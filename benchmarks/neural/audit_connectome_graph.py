"""
O grafo real do Male CNS: quantas ARESTAS, nao quantas sinapses.

    .venv\\Scripts\\python benchmarks\\neural\\audit_connectome_graph.py

## Por que esta distincao decide o projeto

O modelo do Drosobot ja agrega por par: `signed_weights()` soma as sinapses entre
dois neuronios e produz UM peso. Ou seja, a unidade computacional e a ARESTA
neuronio->neuronio, nao a sinapse biologica.

Se dimensionarmos o kernel esparso por "125M sinapses" quando o grafo tem muito
menos arestas, projetamos memoria e banda pra um problema que nao temos.

Nos dados que ja tinhamos isso ja aparecia: o circuito do Giant Fiber tem 36.781
sinapses em 1.465 pares -- razao ~25x. Mas aquele circuito e de conexoes fortes,
entao nao serve pra extrapolar. Este script mede o grafo inteiro.

Dado: connectome-weights-male-cns-v1.0-minconf-0.5.feather, de
https://male-cns.janelia.org/download/ -- descrito la como "segment-to-segment
connection strengths for all segments", isto e, o grafo ja agregado por par.

## Segmento nao e neuronio -- e a diferenca e de tres ordens de grandeza

O arquivo traz TODOS os segmentos: 88,4 milhoes de corpos, a esmagadora maioria
fragmentos minusculos nao revisados. O Male CNS tem ~166.700 NEURONIOS.

Medir o grafo sem filtrar da 151M arestas e superdimensiona o kernel por um fator
enorme. Entao filtramos pelas anotacoes (body-annotations), mantendo so corpos
tracados/revisados que nao sejam glia, orfaos ou fora de escopo -- que e a
definicao de "neuronio" que o dataset usa.

As duas medicoes ficam no relatorio, porque a diferenca entre elas E o resultado.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data" / "male-cns"
SAIDA = RAIZ / "benchmarks" / "neural"

# bytes por elemento na representacao CSR que a Fase 11 propoe
B_INDICES = 4      # int32: id do alvo
B_WEIGHT = 4       # float32: peso com sinal
B_OFFSET = 4       # int32: row_offsets
B_ESTADO = 4       # float32 por neuronio, por campo


def ids_de_neuronios() -> np.ndarray:
    """
    Os bodyIds que o dataset considera NEURONIO.

    Criterio: corpo com traçado util (revisado ou roughly traced) e que nao seja
    glia, orfao, artefato ou fora de escopo. E a mesma nocao que produz o numero
    publicado de ~166.700 neuronios; conferimos abaixo.
    """
    arq = DADOS / "body-annotations.feather"
    if not arq.exists():
        raise SystemExit(f"falta {arq}. Baixe de https://male-cns.janelia.org/download/")
    t = feather.read_table(arq, columns=["bodyId", "statusLabel", "superclass"])
    body = t.column("bodyId").to_numpy()
    status = np.array(t.column("statusLabel").to_pylist(), dtype=object)
    supers = np.array(t.column("superclass").to_pylist(), dtype=object)

    tracado = np.isin(status, ["Roughly traced", "Reviewed",
                               "Prelim Roughly traced", "RT Hard to trace"])
    nao_glia = supers != "glia"
    sel = tracado & nao_glia
    ids = np.sort(body[sel])
    print(f"  neuronios anotados: {len(ids):,} "
          f"(publicado pela Janelia: ~166.700)")
    return ids


def main():
    arq = DADOS / "connectome-weights.feather"
    if not arq.exists():
        raise SystemExit(f"falta {arq}. Baixe de https://male-cns.janelia.org/download/")

    print(f"lendo {arq.name} ({arq.stat().st_size/1e9:.2f} GB)...")
    t = feather.read_table(arq)
    print(f"  colunas: {t.column_names}")
    print(f"  linhas:  {t.num_rows:,}")

    cols = {c.lower(): c for c in t.column_names}
    c_pre = cols.get("bodyid_pre") or cols.get("pre") or t.column_names[0]
    c_post = cols.get("bodyid_post") or cols.get("post") or t.column_names[1]
    c_w = cols.get("weight") or cols.get("count") or t.column_names[2]

    pre = t.column(c_pre).to_numpy()
    post = t.column(c_post).to_numpy()
    w = t.column(c_w).to_numpy()

    # --- antes de filtrar: o grafo de SEGMENTOS, pra registro ---
    seg_arestas = len(pre)
    seg_sin = int(w.sum())
    seg_corpos = len(np.union1d(np.unique(pre), np.unique(post)))
    print()
    print("== grafo de SEGMENTOS (sem filtrar) ==")
    print(f"  corpos   {seg_corpos:>14,}")
    print(f"  arestas  {seg_arestas:>14,}")
    print(f"  sinapses {seg_sin:>14,}")
    print("  (a maioria e fragmento nao revisado -- nao sao neuronios)")

    # --- filtra pros neuronios de verdade ---
    neur_ids = ids_de_neuronios()
    print()
    print(f"== filtrando pelos {len(neur_ids):,} neuronios anotados ==")
    manter = np.isin(pre, neur_ids) & np.isin(post, neur_ids)
    pre, post, w = pre[manter], post[manter], w[manter]
    print(f"  arestas que sobrevivem: {manter.sum():,} de {seg_arestas:,} "
          f"({manter.sum()/seg_arestas*100:.2f}%)")

    n_arestas = len(pre)
    n_sinapses = int(w.sum())
    neuronios = np.union1d(np.unique(pre), np.unique(post))
    n_neuronios = len(neuronios)

    print()
    print("== o numero que importa ==")
    print(f"  neuronios (ids distintos)     {n_neuronios:>14,}")
    print(f"  ARESTAS (pares dirigidos)     {n_arestas:>14,}")
    print(f"  sinapses (soma dos pesos)     {n_sinapses:>14,}")
    print(f"  sinapses por aresta           {n_sinapses/n_arestas:>14.2f}")

    # -------- grau de saida: e ele que dita o custo do scatter --------
    _, cont_out = np.unique(pre, return_counts=True)
    _, cont_in = np.unique(post, return_counts=True)
    # neuronios sem nenhuma saida nao aparecem em `pre`
    out = np.zeros(n_neuronios, dtype=np.int64)
    idx = np.searchsorted(neuronios, np.unique(pre))
    out[idx] = cont_out
    inn = np.zeros(n_neuronios, dtype=np.int64)
    idx = np.searchsorted(neuronios, np.unique(post))
    inn[idx] = cont_in

    def dist(nome, a):
        print(f"  {nome:10s} media {a.mean():8.1f}  mediana {np.median(a):7.0f}  "
              f"p95 {np.percentile(a,95):7.0f}  p99 {np.percentile(a,99):7.0f}  "
              f"max {a.max():7d}")

    print()
    print("== distribuicao de grau ==")
    dist("out-degree", out)
    dist("in-degree", inn)
    # um punhado de neuronios muito conectados domina o kernel -- este numero
    # decide se vale estrategia hibrida (Fase 12)
    topo = np.sort(out)[::-1]
    for k in (10, 100, 1000):
        print(f"  top {k:5d} neuronios por out-degree concentram "
              f"{topo[:k].sum()/n_arestas*100:5.2f}% das arestas")

    print()
    print("== pesos ==")
    print(f"  peso por aresta: media {w.mean():.2f}  mediana {np.median(w):.0f}  "
          f"p99 {np.percentile(w,99):.0f}  max {w.max()}")
    print(f"  arestas com peso 1: {(w==1).sum():,} ({(w==1).sum()/n_arestas*100:.1f}%)")

    # -------- memoria --------
    print()
    print("== memoria da representacao CSR ==")
    mb = lambda b: b / (1024**2)
    m_idx = n_arestas * B_INDICES
    m_w = n_arestas * B_WEIGHT
    m_off = (n_neuronios + 1) * B_OFFSET
    # delay FIXO no nosso modelo (T_DELAY = 1.8 ms): nao precisa de campo por
    # aresta. E a simplificacao que a Fase "DELAYS" pedia pra explorar.
    m_delay_por_aresta = n_arestas * 1
    m_estado = n_neuronios * B_ESTADO * 3        # v, g, ref_ate
    m_spike = n_neuronios // 8                    # bitmask
    total_estatico = m_idx + m_w + m_off
    print(f"  row_offsets      {mb(m_off):9.1f} MiB")
    print(f"  targets (int32)  {mb(m_idx):9.1f} MiB")
    print(f"  weights (fp32)   {mb(m_w):9.1f} MiB")
    print(f"  ---- estatico    {mb(total_estatico):9.1f} MiB  ({mb(total_estatico)/1024:.2f} GiB)")
    print(f"  estado dinamico  {mb(m_estado):9.1f} MiB")
    print(f"  spike bitmask    {mb(m_spike):9.1f} MiB")
    print(f"  TOTAL            {mb(total_estatico + m_estado + m_spike):9.1f} MiB "
          f"de 12288 MiB de VRAM")
    print(f"  (delay por aresta CUSTARIA +{mb(m_delay_por_aresta):.1f} MiB; "
          f"nosso delay e fixo, entao nao pagamos)")

    relatorio = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "fonte": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "segment_graph": {"bodies": int(seg_corpos), "edges": int(seg_arestas),
                          "synapses": int(seg_sin)},
        "neurons": int(n_neuronios),
        "edges": int(n_arestas),
        "synapses": n_sinapses,
        "synapses_per_edge": round(float(n_sinapses / n_arestas), 3),
        "out_degree": {"mean": float(out.mean()), "median": float(np.median(out)),
                       "p95": float(np.percentile(out, 95)),
                       "p99": float(np.percentile(out, 99)), "max": int(out.max())},
        "in_degree": {"mean": float(inn.mean()), "median": float(np.median(inn)),
                      "p95": float(np.percentile(inn, 95)),
                      "p99": float(np.percentile(inn, 99)), "max": int(inn.max())},
        "weight": {"mean": float(w.mean()), "median": float(np.median(w)),
                   "p99": float(np.percentile(w, 99)), "max": int(w.max()),
                   "frac_weight_1": float((w == 1).sum() / n_arestas)},
        "csr_memory_mib": {
            "row_offsets": round(mb(m_off), 2),
            "targets": round(mb(m_idx), 2),
            "weights": round(mb(m_w), 2),
            "static_total": round(mb(total_estatico), 2),
            "dynamic_state": round(mb(m_estado), 2),
            "total": round(mb(total_estatico + m_estado + m_spike), 2),
        },
    }
    SAIDA.mkdir(parents=True, exist_ok=True)
    destino = SAIDA / "male_cns_graph_stats.json"
    destino.write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
