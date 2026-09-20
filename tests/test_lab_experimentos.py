"""
Testes da plataforma experimental.

O que precisa ficar garantido:

1. **Uma receita inválida falha na hora, não na décima corrida.** Uma bateria
   de horas que morre no meio por um campo mal digitado é pior que uma que
   recusa antes de começar.

2. **O hash científico muda quando a ciência muda.** É ele que diz se duas
   corridas são comparáveis; se não reagir a `V_TH`, não serve para nada.

3. **A comparação recusa o que não é comparável.** Produzir a tabela com um
   aviso no rodapé é exatamente como um número errado entra num relatório.

4. **A corrida deixa os cinco arquivos, e a série temporal tem as colunas
   declaradas.** Uma coluna faltando que vira vazio parece medida e não é.

    .venv-flygym2\\Scripts\\python tests/test_lab_experimentos.py

O teste de ponta a ponta monta a mosca de verdade e leva ~30 s.
DROSOBOT_SKIP_LENTOS=1 pula só esse.
"""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

from lab import Receita, hash_ciencia, metadata               # noqa: E402
from lab.analise import comparavel, compara                   # noqa: E402
from lab.registro import COLUNAS, Registro                    # noqa: E402


# ------------------------------------------------------------------ receita

def test_receita_recusa_arena_que_nao_existe():
    try:
        Receita(nome="x", arena="marte")
    except ValueError as e:
        assert "marte" in str(e)
        return
    raise AssertionError("aceitou uma arena inexistente")


def test_receita_recusa_campo_com_nome_errado():
    """Campo mal digitado tem que falhar, nao cair no padrao em silencio."""
    try:
        Receita.de_dict({"nome": "x", "semente": 3})     # o certo e `seed`
    except ValueError as e:
        assert "semente" in str(e)
        return
    raise AssertionError("aceitou campo desconhecido e usou o padrao")


def test_receita_recusa_duracao_nao_positiva():
    for d in (0.0, -1.0):
        try:
            Receita(nome="x", duracao_s=d)
        except ValueError:
            continue
        raise AssertionError(f"aceitou duracao {d}")


def test_id_da_corrida_distingue_o_que_varia():
    a = Receita(nome="lo", escopo="whole", condicao="esq", seed=0)
    b = Receita(nome="lo", escopo="whole", condicao="esq", seed=1)
    c = Receita(nome="lo", escopo="circuit", condicao="esq", seed=0)
    d = Receita(nome="lo", escopo="whole", condicao="dir", seed=0)
    ids = {a.id_corrida, b.id_corrida, c.id_corrida, d.id_corrida}
    assert len(ids) == 4, f"ids colidem: {ids}"


# --------------------------------------------------------- hash da ciencia

def test_hash_reage_a_mudanca_de_parametro_cientifico():
    """
    Mexe em `V_TH` em memoria e confere que o hash muda.

    Restaura no fim. Se o hash nao reagisse, duas corridas com limiares
    diferentes pareceriam comparaveis -- e e exatamente pra isso que ele existe.
    """
    from neural import model as m

    antes = hash_ciencia()
    original = m.V_TH
    try:
        m.V_TH = original + 1.0
        depois = hash_ciencia()
    finally:
        m.V_TH = original
    assert antes != depois, "o hash ignorou uma mudanca no limiar"
    assert hash_ciencia() == antes, "o hash nao voltou ao valor original"


def test_metadata_nunca_chama_de_cerebro_completo():
    """O escopo tem nome longo e o aviso vai junto. Sempre."""
    meta = metadata(Receita(nome="x", escopo="whole"))
    assert meta["escopo_declarado"] == "Male CNS whole-connectome simulation"
    texto = json.dumps(meta).lower()
    for proibido in ("complete brain", "cerebro completo", "full brain"):
        # a unica ocorrencia aceitavel e a NEGACAO explicita
        if proibido in texto:
            assert "not a complete functional brain" in texto, (
                f"metadata afirma {proibido!r} sem negar")
    assert "NOT a complete functional brain" in meta["aviso_escopo"]


# ------------------------------------------------------------- comparacao

def _resumo_falso(escopo, seed=0, condicao="centro", hash_c="abc"):
    return {
        "hash_ciencia": hash_c,
        "receita": {"nome": "t", "escopo": escopo, "seed": seed,
                    "arena": "looming", "condicao": condicao,
                    "duracao_s": 1.0, "physics": "flygym2", "colisao": "legs"},
        "gf": {"excitacao_mV": 1.0, "inibicao_mV": -1.0, "liquido_mV": 0.0,
               "v_min_mV": -50.0, "spikes": 1, "por_populacao_mV": {}},
        "sensorial": {"spikes_totais": 5, "hz_max": 1.0, "ttmn_spikes": 0},
        "desfecho": {"fugas": 0},
        "pasta": "x",
    }


def test_comparacao_recusa_ciencia_diferente():
    a = _resumo_falso("circuit", hash_c="aaa")
    b = _resumo_falso("whole", hash_c="bbb")
    ok, problemas = comparavel(a, b)
    assert not ok and any("cientificos" in p for p in problemas)


def test_comparacao_recusa_semente_diferente():
    ok, problemas = comparavel(_resumo_falso("circuit", seed=0),
                               _resumo_falso("whole", seed=1))
    assert not ok and any("seed" in p for p in problemas)


def test_comparacao_recusa_mesmo_escopo():
    ok, problemas = comparavel(_resumo_falso("whole"), _resumo_falso("whole"))
    assert not ok and any("escopo" in p for p in problemas)


def test_comparacao_aceita_o_par_certo():
    ok, problemas = comparavel(_resumo_falso("circuit"), _resumo_falso("whole"))
    assert ok, problemas


# ---------------------------------------------------------------- registro

def test_serie_temporal_exige_todas_as_colunas():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registro("t", {}, raiz=Path(tmp), carimbo=False)
        try:
            reg.linha(t_s=0.0)                    # faltando quase tudo
        except KeyError as e:
            assert "colunas" in str(e)
        else:
            raise AssertionError("aceitou linha incompleta")
        finally:
            reg.fechar()


def test_registro_escreve_os_arquivos():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registro("t", {"a": 1}, raiz=Path(tmp), carimbo=False)
        reg.linha(**{c: 0 for c in COLUNAS})
        reg.evento(0.1, "gf_spike", {"net_mV": -3.0})
        reg.resumo({"ok": True})
        reg.fechar()
        for nome in ("metadata.json", "timeseries.csv", "events.jsonl",
                     "summary.json"):
            assert (reg.pasta / nome).exists(), f"faltou {nome}"
        with (reg.pasta / "timeseries.csv").open(encoding="utf-8") as f:
            assert next(csv.reader(f)) == COLUNAS


# ------------------------------------------------------------ ponta a ponta

def test_corrida_completa_deixa_tudo_em_disco():
    if os.environ.get("DROSOBOT_SKIP_LENTOS"):
        return
    from lab import roda

    with tempfile.TemporaryDirectory() as tmp:
        pasta = roda(Receita(nome="teste", escopo="circuit", duracao_s=0.2,
                             condicao="centro",
                             estimulo={"azimute_graus": 0.0}),
                     raiz=Path(tmp), silencioso=True)
        for nome in ("metadata.json", "timeseries.csv", "events.jsonl",
                     "summary.json", "telemetry.jsonl"):
            assert (pasta / nome).exists(), f"faltou {nome}"

        resumo = json.loads((pasta / "summary.json").read_text(encoding="utf-8"))
        assert resumo["desfecho"]["janelas"] > 0
        assert resumo["gf"]["arestas_entrando"] == 1455, (
            "as arestas que entram no GF mudaram de numero")
        assert resumo["runtime"]["neurons_simulated"] == 1261

        with (pasta / "timeseries.csv").open(encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))
        assert len(linhas) == resumo["desfecho"]["janelas"], (
            "a serie temporal nao tem uma linha por janela")
        print(f"    {len(linhas)} janelas, {resumo['gf']['spikes']} spikes do GF")


def test_mesma_semente_da_o_mesmo_resultado():
    """Reprodutibilidade: e o unico motivo de a semente estar na receita."""
    if os.environ.get("DROSOBOT_SKIP_LENTOS"):
        return
    from lab import roda

    chaves = ("fugas",)
    with tempfile.TemporaryDirectory() as tmp:
        r = Receita(nome="repro", escopo="circuit", duracao_s=0.2, seed=7)
        a = json.loads((roda(r, raiz=Path(tmp), telemetria=False,
                             silencioso=True) / "summary.json")
                       .read_text(encoding="utf-8"))
        b = json.loads((roda(r, raiz=Path(tmp), telemetria=False,
                             silencioso=True) / "summary.json")
                       .read_text(encoding="utf-8"))
    assert a["gf"] == b["gf"], "o gate mudou entre duas corridas iguais"
    for k in chaves:
        assert a["desfecho"][k] == b["desfecho"][k]
    assert a["desfecho"]["posicao_final_mm"] == b["desfecho"]["posicao_final_mm"]
    print(f"    duas corridas com semente 7: gate e posicao identicos")


if __name__ == "__main__":
    falhas = 0
    for nome, fn in sorted(globals().items()):
        if not nome.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok    {nome}")
        except Exception as e:                                # noqa: BLE001
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print("\n" + ("todos passaram" if not falhas else f"{falhas} falha(s)"))
    sys.exit(1 if falhas else 0)
