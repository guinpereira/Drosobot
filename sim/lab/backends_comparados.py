r"""
A mesma ciencia, dois motores fisicos: onde a divergencia COMECA.

    from lab.backends_comparados import compara_backends, texto
    print(texto(compara_backends(pasta_mujoco, pasta_gpu)))

A comparacao circuito x whole (`analise.py`) responde "o que muda quando o
estimulo atravessa o conectoma inteiro". Esta responde outra coisa: **duas
corridas com a MESMA receita e motores fisicos diferentes ainda sao a mesma
corrida?**

Comparar so a posicao final nao serve. Duas trajetorias que terminam a 0,3 mm
uma da outra podem ter divergido no primeiro contato e se reencontrado por
acaso, ou ter andado juntas 590 ms e se separado no fim. Sao diagnosticos
opostos e o numero final nao os distingue.

Entao o que sai daqui e a **primeira janela em que cada camada se separa**, na
ordem causal:

    FISICA      posicao do torax
    SENSORIAL   taxa de entrada e spikes sensoriais
    NEURAL      excitacao, inibicao e spikes do Giant Fiber
    MOTOR       drive descendente
    DESFECHO    fugas

Se a fisica diverge na janela 12 e o sensorial na 12, a causa e o motor fisico.
Se o sensorial diverge sem a fisica ter divergido, o problema nao e o solver --
e a transducao ou a retina. Se tudo bate e so o desfecho muda, foi um limiar
que caiu do lado errado de uma comparacao, e a divergencia e amplificacao, nao
erro.

## O que esta comparacao AINDA nao ve

`qpos`, `qvel`, forcas de restricao e contatos nao estao no `timeseries.csv`:
as colunas dele sao fixas de proposito, e uma coluna nova quebraria as analises
que ja leem as corridas gravadas. Enquanto nao houver um arquivo proprio para o
estado fisico por passo, a camada FISICA aqui e a trajetoria do torax -- que e
integral do resto, e por isso detecta divergencia mas nao a localiza dentro do
passo.

## O que NAO invalida a comparacao

Backend diferente. E o ponto. `analise.comparavel()` exige `physics` igual
porque la a variavel e o escopo; aqui a variavel e o motor, e quem tem que
bater e o `hash_ciencia` e o `physics_model_hash` -- a mesma ciencia sobre o
mesmo corpo.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Camadas, na ordem causal. Cada uma com as colunas do `timeseries.csv` que a
# representam e a tolerancia abaixo da qual duas janelas contam como iguais.
CAMADAS = (
    ("FISICA", ("pos_x_mm", "pos_y_mm", "pos_z_mm"), 1e-6),
    ("SENSORIAL", ("entrada_hz", "spikes_sensoriais"), 1e-9),
    ("NEURAL", ("gf_exc_mV", "gf_inib_mV", "gf_liquido_mV", "gf_spikes"), 1e-9),
    ("MOTOR", ("drive_esq", "drive_dir"), 1e-9),
    ("DESFECHO", ("fugas_ate_agora",), 1e-9),
)


def _le_serie(pasta) -> list[dict]:
    with (Path(pasta) / "timeseries.csv").open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in linha.items()}
                for linha in csv.DictReader(f)]


def _le_json(pasta, nome) -> dict:
    p = Path(pasta) / nome
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def comparavel(ma: dict, mb: dict) -> tuple[bool, list[str]]:
    """As duas corridas medem a mesma coisa com motores diferentes?"""
    problemas = []
    if ma.get("hash_ciencia") != mb.get("hash_ciencia"):
        problemas.append(
            f"hash_ciencia difere ({ma.get('hash_ciencia')} vs "
            f"{mb.get('hash_ciencia')}): nao e comparacao de backend, e "
            "comparacao de ciencias diferentes")
    ba = ma.get("backend_fisico", {})
    bb = mb.get("backend_fisico", {})
    ha, hb = ba.get("physics_model_hash"), bb.get("physics_model_hash")
    if ha and hb and ha != hb:
        problemas.append(
            f"physics_model_hash difere ({ha} vs {hb}): os dois motores nao "
            "rodaram o mesmo corpo, entao a divergencia nao mede o solver")
    if ba.get("backend") == bb.get("backend"):
        problemas.append(
            f"os dois usam {ba.get('backend')!r}; nao ha backend a comparar")
    ra = ma.get("receita", {})
    rb = mb.get("receita", {})
    for campo in ("seed", "arena", "condicao", "duracao_s", "escopo", "colisao"):
        if ra.get(campo) != rb.get(campo):
            problemas.append(
                f"{campo} difere: {ra.get(campo)!r} vs {rb.get(campo)!r}")
    return (not problemas), problemas


def _primeira_divergencia(a: list[dict], b: list[dict], colunas, tol):
    """(indice, t_s, maior diferenca) da primeira janela que se separa."""
    for i, (la, lb) in enumerate(zip(a, b)):
        pior = max(abs(la[c] - lb[c]) for c in colunas)
        if pior > tol:
            return {"janela": i, "t_s": round(la["t_s"], 4),
                    "diferenca": float(f"{pior:.6g}")}
    return None


def compara_backends(pasta_a, pasta_b) -> dict:
    """
    Duas corridas da mesma receita, motores diferentes. Levanta se nao batem.

    `pasta_a` e a REFERENCIA (tipicamente `flygym2-mujoco`). A ordem importa so
    para o relatorio; a divergencia e simetrica.
    """
    ma, mb = _le_json(pasta_a, "metadata.json"), _le_json(pasta_b, "metadata.json")
    ok, problemas = comparavel(ma, mb)
    if not ok:
        raise ValueError("corridas nao comparaveis:\n  - " + "\n  - ".join(problemas))

    sa, sb = _le_serie(pasta_a), _le_serie(pasta_b)
    n = min(len(sa), len(sb))
    camadas = {}
    for nome, colunas, tol in CAMADAS:
        camadas[nome] = {
            "colunas": list(colunas),
            "tolerancia": tol,
            "primeira_divergencia": _primeira_divergencia(
                sa[:n], sb[:n], colunas, tol),
            "diferenca_final": float(f"{max(abs(sa[n-1][c] - sb[n-1][c]) for c in colunas):.6g}")
            if n else None,
        }

    ra, rb = _le_json(pasta_a, "summary.json"), _le_json(pasta_b, "summary.json")
    ordem = [nome for nome, _, _ in CAMADAS]
    primeiras = [(nome, camadas[nome]["primeira_divergencia"])
                 for nome in ordem if camadas[nome]["primeira_divergencia"]]
    return {
        "hash_ciencia": ma.get("hash_ciencia"),
        "physics_model_hash": ma.get("backend_fisico", {}).get("physics_model_hash"),
        "receita": {k: v for k, v in ma.get("receita", {}).items()
                    if k != "physics"},
        "backends": {
            "a": ma.get("backend_fisico", {}),
            "b": mb.get("backend_fisico", {}),
        },
        "janelas_comparadas": n,
        "camadas": camadas,
        "camada_que_diverge_primeiro": primeiras[0][0] if primeiras else None,
        "custo": {
            "a": ra.get("custo", {}), "b": rb.get("custo", {}),
        },
        "pastas": {"a": str(pasta_a), "b": str(pasta_b)},
    }


def texto(cmp: dict) -> str:
    """A comparacao como tabela de terminal."""
    a, b = cmp["backends"]["a"], cmp["backends"]["b"]
    r = cmp["receita"]
    out = [
        "  MESMA CIENCIA, DOIS MOTORES FISICOS",
        f"  {r.get('nome')} / {r.get('condicao') or 'padrao'} / "
        f"seed {r.get('seed')} / {r.get('duracao_s')} s / "
        f"escopo {r.get('escopo')}",
        f"  ciencia {cmp['hash_ciencia']}   corpo {cmp['physics_model_hash']}",
        f"  A = {a.get('backend')}  ({a.get('familia')})",
        f"  B = {b.get('backend')}  ({b.get('familia')}"
        + (f", {b.get('precisao')}" if b.get("precisao") else "") + ")",
        "",
        f"  {'camada':<12s} {'primeira divergencia':>26s} {'dif. final':>14s}",
        "  " + "-" * 55,
    ]
    for nome, _, _ in CAMADAS:
        c = cmp["camadas"][nome]
        d = c["primeira_divergencia"]
        quando = (f"janela {d['janela']} (t={d['t_s']}s)" if d else "nenhuma")
        final = (f"{c['diferenca_final']:.3e}"
                 if c["diferenca_final"] is not None else "-")
        out.append(f"  {nome:<12s} {quando:>26s} {final:>14s}")

    out.append("")
    primeira = cmp["camada_que_diverge_primeiro"]
    if primeira is None:
        out.append("  Nenhuma camada divergiu acima da tolerancia: os dois "
                   "motores produziram")
        out.append("  a mesma corrida dentro do que estas colunas enxergam.")
    else:
        out.append(f"  Primeira camada a divergir: {primeira}. O que vem depois "
                   "dela na cadeia")
        out.append("  causal herda a divergencia; procurar a causa nas camadas "
                   "seguintes e")
        out.append("  perseguir o sintoma.")
    ca, cb = cmp["custo"]["a"], cmp["custo"]["b"]
    if ca and cb:
        out.append("")
        out.append(f"  custo   A rtf={ca.get('rtf')}   B rtf={cb.get('rtf')}")
    out.append("")
    out.append("  qpos, qvel, forcas de restricao e contatos NAO entram aqui: o")
    out.append("  timeseries.csv nao os grava. A camada FISICA e a trajetoria do")
    out.append("  torax, que detecta divergencia mas nao a localiza no passo.")
    return "\n".join(out)
