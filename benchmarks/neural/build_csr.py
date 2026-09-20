"""
Constroi o CSR do Male CNS real e grava num binario que a GPU le direto.

    .venv\\Scripts\\python benchmarks\\neural\\build_csr.py
    .venv\\Scripts\\python benchmarks\\neural\\build_csr.py --n 50000

Sai em data/male-cns/csr/. O formato e cru de proposito: a Unity precisa apenas
carregar tres arrays em ComputeBuffer, sem parser.

## Sinal

O peso ja sai COM SINAL, pela regra de Dale aplicada ao neurotransmissor do
pre-sinaptico -- a mesma de sim/connectome_model.py (`NT_SIGN`). Isso mantem a
aresta como unico numero e evita um array de sinal separado, sem mudar a
matematica: `w_mV = sinapses * sinal * W_SYN`.

Neuronio sem neurotransmissor conhecido nao vira excitatorio por omissao. Ele e
CONTADO e reportado, porque assumir sinal onde o dado nao diz e inventar
conectoma.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data" / "male-cns"
SAIDA = DADOS / "csr"

# de sim/connectome_model.py -- convencao de Shiu et al.
NT_SIGN = {
    "acetylcholine": +1, "dopamine": +1, "octopamine": +1, "serotonin": +1,
    "gaba": -1, "glutamate": -1, "histamine": -1,
}
W_SYN_MV = 0.275     # o unico parametro livre do modelo


def ids_de_neuronios():
    t = feather.read_table(DADOS / "body-annotations.feather",
                           columns=["bodyId", "statusLabel", "superclass"])
    body = t.column("bodyId").to_numpy()
    status = np.array(t.column("statusLabel").to_pylist(), dtype=object)
    supers = np.array(t.column("superclass").to_pylist(), dtype=object)
    sel = np.isin(status, ["Roughly traced", "Reviewed",
                           "Prelim Roughly traced", "RT Hard to trace"])
    sel &= supers != "glia"
    return np.sort(body[sel])


def sinais(ids: np.ndarray):
    """sinal por neuronio, na ordem de `ids`. 0 = neurotransmissor desconhecido."""
    t = feather.read_table(DADOS / "body-neurotransmitters.feather")
    cols = {c.lower(): c for c in t.column_names}
    c_body = cols.get("body") or cols.get("bodyid") or t.column_names[0]
    # consensus_nt e a coluna curada; predicted_nt e so o classificador. Usamos a
    # consensus e caimos pra predicted so quando ela esta vazia -- que e a mesma
    # ordem de preferencia de sim/connectome_model.py.
    c_nt = (cols.get("consensus_nt") or cols.get("consensusnt")
            or cols.get("predicted_nt") or t.column_names[1])
    body = t.column(c_body).to_numpy()
    nt = np.array([str(x).lower() if x else "" for x in t.column(c_nt).to_pylist()],
                  dtype=object)

    # vetorizado: sao 1,8M linhas e 164k neuronios; laco Python aqui custa minutos
    sinal_tab = np.array([NT_SIGN.get(x, 0) for x in nt], dtype=np.int8)
    ordem = np.argsort(body)
    body_ord, sinal_ord = body[ordem], sinal_tab[ordem]
    pos = np.searchsorted(body_ord, ids)
    pos = np.clip(pos, 0, len(body_ord) - 1)
    achou = body_ord[pos] == ids
    s = np.where(achou, sinal_ord[pos], 0).astype(np.int8)
    return s, c_nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="usar so os N neuronios de maior grau (0 = todos)")
    args = ap.parse_args()

    print("lendo anotacoes...")
    ids = ids_de_neuronios()
    print(f"  {len(ids):,} neuronios anotados")

    print("lendo o grafo...")
    t = feather.read_table(DADOS / "connectome-weights.feather")
    pre = t.column("body_pre").to_numpy()
    post = t.column("body_post").to_numpy()
    w = t.column("weight").to_numpy()

    manter = np.isin(pre, ids) & np.isin(post, ids)
    pre, post, w = pre[manter], post[manter], w[manter]
    print(f"  {len(pre):,} arestas entre neuronios")

    if args.n and args.n < len(ids):
        # subconjunto por grau de saida: mantem os hubs, que sao o caso dificil.
        # Cortar aleatoriamente daria um grafo mais facil que o real.
        u, c = np.unique(pre, return_counts=True)
        ordem = u[np.argsort(-c)][:args.n]
        ids = np.sort(ordem)
        manter = np.isin(pre, ids) & np.isin(post, ids)
        pre, post, w = pre[manter], post[manter], w[manter]
        print(f"  subconjunto: {len(ids):,} neuronios, {len(pre):,} arestas")

    print("aplicando o sinal do neurotransmissor...")
    sinal, col_nt = sinais(ids)
    n_desconhecido = int((sinal == 0).sum())
    print(f"  coluna usada: {col_nt}")
    print(f"  sinal conhecido: {len(ids) - n_desconhecido:,}  "
          f"desconhecido: {n_desconhecido:,} "
          f"({n_desconhecido/len(ids)*100:.1f}%)")
    print("  neuronio sem NT conhecido fica com sinal 0 (aresta muda), nao +1 --"
          " assumir excitatorio por omissao inventaria conectoma")

    # ---- indices densos 0..N-1 ----
    idx_pre = np.searchsorted(ids, pre).astype(np.int32)
    idx_post = np.searchsorted(ids, post).astype(np.int32)

    # ---- ordena por pre pra montar o CSR ----
    print("montando CSR...")
    ordem = np.argsort(idx_pre, kind="stable")
    idx_pre, idx_post, w = idx_pre[ordem], idx_post[ordem], w[ordem]

    n = len(ids)
    contagem = np.bincount(idx_pre, minlength=n)
    row_offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(contagem, out=row_offsets[1:])

    peso_mv = (w.astype(np.float32)
               * sinal[idx_pre].astype(np.float32)
               * np.float32(W_SYN_MV))

    SAIDA.mkdir(parents=True, exist_ok=True)
    sufixo = f"_{n}" if args.n else "_full"
    row_offsets.tofile(SAIDA / f"row_offsets{sufixo}.bin")
    idx_post.tofile(SAIDA / f"targets{sufixo}.bin")
    peso_mv.tofile(SAIDA / f"weights{sufixo}.bin")
    ids.astype(np.int64).tofile(SAIDA / f"body_ids{sufixo}.bin")

    mudas = int((peso_mv == 0).sum())
    meta = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "neurons": int(n),
        "edges": int(len(idx_post)),
        "synapses": int(w.sum()),
        "w_syn_mV": W_SYN_MV,
        "edges_excitatorias": int((peso_mv > 0).sum()),
        "edges_inibitorias": int((peso_mv < 0).sum()),
        "edges_mudas_nt_desconhecido": mudas,
        "neurons_sem_nt": n_desconhecido,
        "out_degree": {"mean": float(contagem.mean()),
                       "median": float(np.median(contagem)),
                       "p99": float(np.percentile(contagem, 99)),
                       "max": int(contagem.max())},
        "bytes": {"row_offsets": int(row_offsets.nbytes),
                  "targets": int(idx_post.nbytes),
                  "weights": int(peso_mv.nbytes)},
    }
    (SAIDA / f"meta{sufixo}.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")
    total_mb = (row_offsets.nbytes + idx_post.nbytes + peso_mv.nbytes) / 1048576
    print()
    print(f"  {n:,} neuronios, {len(idx_post):,} arestas, {total_mb:.1f} MiB")
    print(f"  excitatorias {meta['edges_excitatorias']:,}  "
          f"inibitorias {meta['edges_inibitorias']:,}  "
          f"mudas (NT desconhecido) {mudas:,}")
    print(f"  salvo em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
