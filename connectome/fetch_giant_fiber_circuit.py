"""Passo 2: puxa 1 hop de entrada e 1 hop de saida dos dois Giant Fiber (DNp01),
monta o circuito sensorio-motor completo pra virar rede Brian2 depois.
"""
from pathlib import Path
import pandas as pd
from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC

token = Path(__file__).parent.parent.joinpath(".neuprint_token").read_text().strip()
client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)

GF_IDS = [10001, 10010]  # DNp01(GF)_R, DNp01(GF)_L

pd.set_option("display.max_rows", 30)
pd.set_option("display.width", 140)

# conn_in/conn_out só trazem bodyId_pre/bodyId_post/roi/weight -- tipo vem do dataframe de neuronios, precisa merge
type_by_id = lambda neurons: neurons.set_index("bodyId")["type"]

# Quem alimenta o GF (candidatos a "sensor" -- visual looming etc)
neurons_in, conn_in = fetch_adjacencies(None, NC(bodyId=GF_IDS))
conn_in = conn_in.merge(neurons_in[["bodyId", "type"]], left_on="bodyId_pre", right_on="bodyId")
upstream = (
    conn_in.groupby(["bodyId_pre", "type"])["weight"]
    .sum()
    .reset_index()
    .sort_values("weight", ascending=False)
)
print("=== TOP 20 ENTRADAS do Giant Fiber (candidatos a estimulo) ===")
print(upstream.head(20).to_string(index=False))

# Pra onde o GF manda sinal (candidatos a "motor")
neurons_out, conn_out = fetch_adjacencies(NC(bodyId=GF_IDS), None)
conn_out = conn_out.merge(neurons_out[["bodyId", "type"]], left_on="bodyId_post", right_on="bodyId")
downstream = (
    conn_out.groupby(["bodyId_post", "type"])["weight"]
    .sum()
    .reset_index()
    .sort_values("weight", ascending=False)
)
print("\n=== TOP 20 SAIDAS do Giant Fiber (candidatos a motor) ===")
print(downstream.head(20).to_string(index=False))

conn_in.to_csv(Path(__file__).parent / "gf_upstream_connections.csv", index=False)
conn_out.to_csv(Path(__file__).parent / "gf_downstream_connections.csv", index=False)
print("\nSalvo: gf_upstream_connections.csv, gf_downstream_connections.csv")
