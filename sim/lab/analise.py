"""
Circuito × conectoma inteiro, com número.

A pergunta que este projeto faz desde o começo é: **o que muda quando o mesmo
estímulo atravessa o conectoma inteiro em vez do circuito isolado?** Responder
"a mosca não foge" é observação; responder com o balanço de entrada no Giant
Fiber, quebrado por população pré-sináptica, é medição.

Esta análise não roda nada. Ela lê duas corridas já gravadas e as põe lado a
lado. Se as duas não forem comparáveis — semente diferente, hash científico
diferente, condição diferente — ela **diz isso** em vez de produzir uma tabela
que parece válida.

## O que é comparável

Duas corridas comparam quando têm o mesmo `hash_ciencia`, a mesma semente, a
mesma arena e a mesma condição. O escopo é, por construção, a única coisa que
difere — é o que está sendo medido.

O backend **não** entra: trocar de GPU não muda o resultado, e a equivalência
CPU/OpenCL é verificada em `tests/test_neural_backend.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .registro import le_resumo

# Grandezas que a comparação reporta, na ordem em que fazem sentido ler: da
# entrada sensorial ao desfecho comportamental.
LINHAS = [
    ("spikes sensoriais", ("sensorial", "spikes_totais"), "{:>12,d}"),
    ("entrada max (Hz)", ("sensorial", "hz_max"), "{:>12.2f}"),
    ("excitacao no GF (mV)", ("gf", "excitacao_mV"), "{:>12,.1f}"),
    ("inibicao no GF (mV)", ("gf", "inibicao_mV"), "{:>12,.1f}"),
    ("liquido no GF (mV)", ("gf", "liquido_mV"), "{:>12,.1f}"),
    ("v minimo do GF (mV)", ("gf", "v_min_mV"), "{:>12,.1f}"),
    ("spikes do GF", ("gf", "spikes"), "{:>12,d}"),
    ("spikes do TTMn", ("sensorial", "ttmn_spikes"), "{:>12,d}"),
    ("fugas", ("desfecho", "fugas"), "{:>12,d}"),
]


def _pega(d: dict, caminho: tuple):
    for k in caminho:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if not isinstance(d, dict) else None


def comparavel(a: dict, b: dict) -> tuple[bool, list[str]]:
    """As duas corridas podem ser comparadas? Se não, por quê."""
    problemas = []
    if a.get("hash_ciencia") != b.get("hash_ciencia"):
        problemas.append(
            f"parametros cientificos diferentes ({a.get('hash_ciencia')} vs "
            f"{b.get('hash_ciencia')}): as duas nao medem a mesma coisa")
    ra, rb = a.get("receita", {}), b.get("receita", {})
    for campo in ("seed", "arena", "condicao", "duracao_s", "physics", "colisao"):
        if ra.get(campo) != rb.get(campo):
            problemas.append(
                f"{campo} difere: {ra.get(campo)!r} vs {rb.get(campo)!r}")
    if ra.get("escopo") == rb.get("escopo"):
        problemas.append(
            f"as duas tem escopo {ra.get('escopo')!r}; nao ha o que comparar")
    return (not problemas), problemas


def compara(pasta_circuito, pasta_whole) -> dict:
    """
    Põe as duas corridas lado a lado e devolve o resultado estruturado.

    Levanta se não forem comparáveis. Produzir a tabela assim mesmo, com um
    aviso no rodapé, é como o número errado entra num relatório.
    """
    a = le_resumo(pasta_circuito)
    b = le_resumo(pasta_whole)
    ok, problemas = comparavel(a, b)
    if not ok:
        raise ValueError("corridas nao comparaveis:\n  - " + "\n  - ".join(problemas))

    grandezas = {}
    for rotulo, caminho, _ in LINHAS:
        va, vb = _pega(a, caminho), _pega(b, caminho)
        razao = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            razao = round(vb / va, 3) if va != 0 else None
        grandezas[rotulo] = {"circuito": va, "whole": vb, "razao": razao}

    return {
        "receita": {k: v for k, v in a["receita"].items() if k != "escopo"},
        "hash_ciencia": a["hash_ciencia"],
        "grandezas": grandezas,
        "gf_por_populacao": {
            "circuito": a["gf"].get("por_populacao_mV", {}),
            "whole": b["gf"].get("por_populacao_mV", {}),
        },
        "pastas": {"circuito": a["pasta"], "whole": b["pasta"]},
    }


def texto(cmp: dict, n_populacoes: int = 8) -> str:
    """A comparação como tabela de terminal."""
    r = cmp["receita"]
    out = []
    out.append(f"  CIRCUITO x MALE CNS WHOLE-CONNECTOME SIMULATION")
    out.append(f"  {r.get('nome')} / {r.get('condicao') or 'padrao'} / "
               f"seed {r.get('seed')} / {r.get('duracao_s')} s / "
               f"ciencia {cmp['hash_ciencia']}")
    out.append("")
    out.append(f"  {'grandeza':<24s} {'circuito':>12s} {'whole':>12s} {'razao':>9s}")
    out.append("  " + "-" * 61)
    for rotulo, _, fmt in LINHAS:
        g = cmp["grandezas"][rotulo]
        va = fmt.format(g["circuito"]) if g["circuito"] is not None else f"{'-':>12s}"
        vb = fmt.format(g["whole"]) if g["whole"] is not None else f"{'-':>12s}"
        razao = f"{g['razao']:>9.2f}" if g["razao"] is not None else f"{'-':>9s}"
        out.append(f"  {rotulo:<24s} {va} {vb} {razao}")

    for escopo in ("circuito", "whole"):
        pop = cmp["gf_por_populacao"].get(escopo) or {}
        if not pop:
            continue
        out.append("")
        out.append(f"  maiores contribuicoes no GF -- {escopo} "
                   f"({len(pop)} populacoes)")
        itens = sorted(pop.items(), key=lambda kv: kv[1])[:n_populacoes]
        for nome, mv in itens:
            out.append(f"    {nome:<14s} {mv:>12,.1f} mV")
    out.append("")
    out.append("  Whole = Male CNS whole-connectome simulation. NAO e um cerebro")
    out.append("  funcional completo: so a via de looming e a motora tem semantica")
    out.append("  sensorial/motora modelada.")
    return "\n".join(out)


def salva(cmp: dict, destino: Path) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(cmp, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    return destino
