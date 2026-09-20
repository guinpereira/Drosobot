"""
Todo backend tem que reproduzir a referencia. Os MESMOS testes, pra todos.

    .venv\\Scripts\\python tests\\test_neural_backend.py

O kernel de LIF denso ja tinha sido validado em D3D12
(tests/test_lif_gpu_equivalencia.py: 999/1000 neuronios com contagem de spike
identica). O que faltava era a PROPAGACAO SINAPTICA -- acumulacao no alvo, peso
com sinal, convergencia de varios pre-sinapticos, e o atraso fixo.

Isto fecha essa divida, e fecha pra qualquer backend: a lista BACKENDS abaixo e
o que roda. Quando o host nativo de D3D12 existir, basta acrescentar o nome ali
-- nenhum teste muda.

## Politica de tolerancia

A mesma ja aceita no LIF:

  - contagem de spike por neuronio tem que bater, salvo neuronios que passaram
    a menos de um ULP de fp32 do limiar. Esses sao indistinguiveis por
    construcao, e o teste MEDE a margem pra provar que e o caso.
  - v e g comparados so nos neuronios cuja contagem bateu. Um neuronio um spike
    fora de fase tem v completamente diferente; isso e o trem de spike
    divergindo por um evento, nao "v divergiu", e a checagem anterior ja tratou.

Diferenca real falha. Diferenca indistinguivel passa, e sai no relatorio.
"""
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

from neural import (  # noqa: E402
    BackendIndisponivel, Conectoma, NeuralEngine, V_TH,
    carrega_male_cns, existe, rede_sintetica, subgrafo,
)
from neural.model import Coeficientes, V_REST, V_RESET  # noqa: E402

EPS32 = float(abs(np.spacing(np.float32(V_TH))))
TETO_DIVERGENCIA = 0.005      # 0,5% dos neuronios

# Backends a validar. D3D12 entra aqui quando tiver host nativo -- e os testes
# nao mudam, que e o ponto da interface.
BACKENDS = ["opencl"]


def _engine(c, backend, grupos=None):
    try:
        return NeuralEngine(c, backend=backend, grupos=grupos)
    except (BackendIndisponivel, ImportError):
        return None


def _margens(c: Conectoma, externo, n_passos):
    """Menor distancia ao limiar por neuronio, em fp64. Julga as divergencias."""
    k = Coeficientes.de()
    n = c.n
    v = np.full(n, V_REST); g = np.zeros(n)
    ref = np.full(n, -1, dtype=np.int64)
    anel = np.zeros((k.atraso_passos, n))
    cursor = 0
    menor = np.full(n, np.inf)
    for p in range(n_passos):
        chega = anel[cursor].copy(); anel[cursor] = 0.0
        if externo is not None:
            chega = chega + externo
        u = v - V_REST
        u_novo = u * k.dec_v + g * k.acopla
        g = g * k.dec_g + chega
        livre = p >= ref
        v = np.where(livre, V_REST + u_novo, V_RESET)
        menor = np.minimum(menor, np.where(livre, np.abs(v - V_TH), np.inf))
        disparou = livre & (v > V_TH)
        v = np.where(disparou, V_RESET, v)
        ref = np.where(disparou, p + k.ref_passos, ref)
        quem = np.flatnonzero(disparou)
        if len(quem):
            dest = anel[cursor]
            for i in quem:
                a, b = c.row_offsets[i], c.row_offsets[i + 1]
                if b > a:
                    np.add.at(dest, c.targets[a:b], c.weights[a:b])
        cursor = (cursor + 1) % k.atraso_passos
    return menor


def _compara(ref_eng, alt_eng, rotulo, margens, tol_v, tol_g):
    n = ref_eng.c.n
    cont_r = ref_eng.backend.le_contagem_total()
    cont_a = alt_eng.backend.le_contagem_total()
    v_r, g_r, _ = ref_eng.backend.le_estado_completo()
    v_a, g_a, _ = alt_eng.backend.le_estado_completo()

    difs = np.flatnonzero(cont_r != cont_a)
    print(f"    spikes ref={int(cont_r.sum())} alt={int(cont_a.sum())}  "
          f"divergem em {difs.size}/{n}")

    if difs.size:
        longe = [(int(i), float(margens[i])) for i in difs if margens[i] >= EPS32]
        if longe:
            raise AssertionError(
                f"{rotulo}: {len(longe)} neuronio(s) divergiram LONGE do limiar "
                f"-- erro de formula, nao precisao: {longe[:5]}")
        print(f"      todas de limiar (eps32 {EPS32:.2e}), "
              f"menor margem {min(margens[i] for i in difs):.2e} mV")
    assert difs.size / n <= TETO_DIVERGENCIA, (
        f"{rotulo}: divergencia {difs.size/n:.3%} acima do teto")

    concorda = np.ones(n, dtype=bool); concorda[difs] = False
    dv = float(np.abs(v_a[concorda] - v_r[concorda]).max()) if concorda.any() else 0.0
    dg = float(np.abs(g_a - g_r).max())
    print(f"      |dv| {dv:.3e} mV (nos {int(concorda.sum())} que concordam)   "
          f"|dg| {dg:.3e} mV")
    assert dv < tol_v, f"{rotulo}: v divergiu {dv}"
    assert dg < tol_g, f"{rotulo}: g divergiu {dg}"


def _roda_par(c, backend, externo, n_passos, grupos=None):
    ref = NeuralEngine(c, backend="cpu", grupos=grupos)
    alt = _engine(c, backend, grupos)
    if alt is None:
        return None, None
    for _ in range(n_passos):
        ref.passo(externo)
        alt.passo(externo)
    alt.backend.sincroniza()
    return ref, alt


# ------------------------------------------------------------------ testes

def test_scatter_acumula_no_alvo_com_sinal():
    """
    Caso minimo e deterministico: tres pre-sinapticos convergindo num alvo, um
    deles inibitorio. Se o atomico perder uma contribuicao ou ignorar o sinal,
    aparece aqui.
    """
    ro = np.array([0, 1, 2, 3, 3], dtype=np.int32)
    tg = np.array([3, 3, 3], dtype=np.int32)
    w = np.array([1.0, 2.0, -0.5], dtype=np.float32)   # soma esperada: +2.5 mV
    c = Conectoma(ro, tg, w, np.arange(4, dtype=np.int64), "convergencia")

    for nome in BACKENDS:
        ref = NeuralEngine(c, backend="cpu")
        alt = _engine(c, nome)
        if alt is None:
            print(f"  PULADO {nome}: indisponivel")
            continue
        # estimulo forte e SUSTENTADO nos tres pre-sinapticos: um pulso unico
        # decai antes de cruzar o limiar
        ext = np.array([4.0, 4.0, 4.0, 0.0], dtype=np.float32)
        for _ in range(40):
            ref.passo(ext); alt.passo(ext)
        zero = np.zeros(4, dtype=np.float32)
        for _ in range(8):
            ref.passo(zero); alt.passo(zero)
        alt.backend.sincroniza()

        _, g_r, _ = ref.backend.le_estado_completo()
        _, g_a, _ = alt.backend.le_estado_completo()
        cont_r = ref.backend.le_contagem_total()
        print(f"  {nome}: pre dispararam {cont_r[:3].tolist()}  "
              f"g no alvo ref={g_r[3]:.5f} alt={g_a[3]:.5f}")
        assert cont_r[:3].sum() > 0, "os pre-sinapticos nao dispararam"
        assert abs(g_r[3]) > 1e-6, "a referencia nao acumulou no alvo"
        assert abs(g_a[3] - g_r[3]) < 1e-3, (
            f"{nome}: scatter divergiu no alvo ({g_a[3]} x {g_r[3]})")
        # quem nao e alvo nao pode ter recebido nada de sinapse
        assert abs(g_a[0] - g_r[0]) < 1e-3
    return "ok"


def test_atraso_chega_no_passo_certo():
    """O sinal nao pode chegar antes nem depois dos 4 passos."""
    ro = np.array([0, 1, 1], dtype=np.int32)
    c = Conectoma(ro, np.array([1], np.int32), np.array([3.0], np.float32),
                  np.arange(2, dtype=np.int64), "atraso")
    D = Coeficientes.de().atraso_passos

    for nome in BACKENDS:
        ref = NeuralEngine(c, backend="cpu")
        alt = _engine(c, nome)
        if alt is None:
            print(f"  PULADO {nome}: indisponivel")
            continue
        ext = np.array([4.0, 0.0], dtype=np.float32)
        # dispara o pre e anota em que passo o alvo recebe
        chegou_r = chegou_a = None
        for p in range(60):
            ref.passo(ext); alt.passo(ext)
            alt.backend.sincroniza()
            _, g_r, _ = ref.backend.le_estado_completo()
            _, g_a, _ = alt.backend.le_estado_completo()
            if chegou_r is None and abs(g_r[1]) > 1e-6:
                chegou_r = p
            if chegou_a is None and abs(g_a[1]) > 1e-6:
                chegou_a = p
            if chegou_r is not None and chegou_a is not None:
                break
        cont = ref.backend.le_contagem_total()
        print(f"  {nome}: pre disparou {cont[0]}x; alvo recebeu no passo "
              f"ref={chegou_r} alt={chegou_a} (atraso={D})")
        assert chegou_r is not None, "o sinal nunca chegou nem na referencia"
        assert chegou_r == chegou_a, (
            f"{nome}: discorda de QUANDO o sinal chega ({chegou_a} x {chegou_r})")
    return "ok"


def test_rede_sintetica():
    """Grau zero, grau alto, pesos negativos, convergencia."""
    c = rede_sintetica(n=512, grau=12, semente=3)
    ext = np.full(c.n, 0.9, dtype=np.float32)
    ext[::7] = 1.4                      # desincroniza a rede
    n_passos = 300
    margens = _margens(c, ext.astype(np.float64), n_passos)

    for nome in BACKENDS:
        ref, alt = _roda_par(c, nome, ext, n_passos)
        if ref is None:
            print(f"  PULADO {nome}: indisponivel")
            continue
        print(f"  {nome} / sintetica 512x12")
        _compara(ref, alt, f"{nome}/sintetica", margens, tol_v=1e-2, tol_g=1e-2)
    return "ok"


def test_subgrafo_real_do_conectoma():
    """
    O caso que importa: pedaco REAL do Male CNS, com a distribuicao de grau e os
    pesos do dataset. Escolhemos os de MAIOR grau -- o caso dificil, com muita
    convergencia e muita contencao atomica.
    """
    if not existe():
        print("  PULADO: CSR ausente (rode benchmarks/neural/build_csr.py)")
        return "pulado"

    c = carrega_male_cns()
    graus = np.diff(c.row_offsets.astype(np.int64))
    sub = subgrafo(c, np.argsort(-graus)[:1500])
    print(f"  subgrafo real: {sub.n} neuronios, {sub.e} arestas, "
          f"grau medio {sub.e/sub.n:.1f}")

    rng = np.random.default_rng(11)
    ext = rng.uniform(0.6, 1.1, size=sub.n).astype(np.float32)
    n_passos = 150
    margens = _margens(sub, ext.astype(np.float64), n_passos)

    for nome in BACKENDS:
        ref, alt = _roda_par(sub, nome, ext, n_passos)
        if ref is None:
            print(f"  PULADO {nome}: indisponivel")
            continue
        print(f"  {nome} / subgrafo real")
        # tolerancia maior em g: um alvo de grau alto soma milhares de
        # contribuicoes em ponto fixo, e o erro de quantizacao acumula
        _compara(ref, alt, f"{nome}/subgrafo", margens, tol_v=5e-2, tol_g=5e-1)
    return "ok"


def test_atividade_por_grupo():
    """A agregacao por populacao tem que bater com a contagem individual."""
    c = rede_sintetica(n=256, grau=6, semente=5)
    grupos = (np.arange(c.n) % 4).astype(np.int32)
    ext = np.full(c.n, 1.0, dtype=np.float32)

    for nome in BACKENDS:
        ref, alt = _roda_par(c, nome, ext, 120, grupos=grupos)
        if ref is None:
            print(f"  PULADO {nome}: indisponivel")
            continue
        g_ref = ref.atividade_por_grupo(zerar=False)
        g_alt = alt.atividade_por_grupo(zerar=False)
        # a soma por grupo tem que fechar com a contagem por neuronio
        cont = ref.backend.le_contagem_total()
        esperado = np.bincount(grupos, weights=cont, minlength=4).astype(np.int64)
        print(f"  {nome}: por grupo ref={g_ref.tolist()} alt={g_alt.tolist()}")
        assert (g_ref == esperado).all(), (
            f"referencia: soma por grupo {g_ref} != contagem {esperado}")
        assert (g_alt == g_ref).all(), f"{nome}: agregacao divergiu"
    return "ok"


def test_parametros_batem_com_o_modelo():
    """A copia em neural/model.py nao pode derivar de connectome_model.py."""
    from brian2 import mV, ms
    from connectome_model import (V_REST as VR, V_RESET as VRS, V_THRESHOLD as VT,
                                  TAU_MBR, TAU_SYN, T_REFRACTORY, T_DELAY as TD,
                                  W_SYN)
    from neural import model as nm
    pares = [("V_REST", float(VR / mV), nm.V_REST),
             ("V_RESET", float(VRS / mV), nm.V_RESET),
             ("V_TH", float(VT / mV), nm.V_TH),
             ("TAU_M", float(TAU_MBR / ms), nm.TAU_M),
             ("TAU_S", float(TAU_SYN / ms), nm.TAU_S),
             ("T_REF", float(T_REFRACTORY / ms), nm.T_REF),
             ("T_DELAY", float(TD / ms), nm.T_DELAY),
             ("W_SYN", float(W_SYN / mV), nm.W_SYN_MV)]
    for nome, ref, copia in pares:
        assert math.isclose(ref, copia, rel_tol=1e-12), (
            f"{nome}: modelo {ref!r} x backend {copia!r}")
    print(f"  {len(pares)} parametros conferem com sim/connectome_model.py")
    return "ok"


def _todos():
    return [(n, o) for n, o in sorted(globals().items())
            if n.startswith("test_") and callable(o)]


if __name__ == "__main__":
    falhas = 0
    for nome, fn in _todos():
        try:
            r = fn()
            print(f"  {'PULADO' if r == 'pulado' else 'ok    '} {nome}\n")
        except Exception as e:
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}\n")
    print(f"{len(_todos()) - falhas}/{len(_todos())} passaram")
    sys.exit(1 if falhas else 0)
