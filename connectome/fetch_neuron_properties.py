"""
Puxa as propriedades por neuronio que as simulacoes precisam alem do peso:

- neurotransmissor (define se a sinapse excita ou inibe o alvo)
- somaSide (L/R/M) -- define a que hemisferio o neuronio pertence

Sem isso as redes tratavam tudo como excitatorio e nao separavam lado, as duas
simplificacoes documentadas no README que este script existe pra remover.

Roda depois dos outros fetch_*, le os bodyIds que eles ja salvaram nos CSVs.
Sai em connectome/neuron_properties.csv.
"""
from pathlib import Path
import pandas as pd
from neuprint import Client, fetch_neurons, NeuronCriteria as NC

HERE = Path(__file__).parent
ROOT = HERE.parent

token = (ROOT / ".neuprint_token").read_text().strip()
client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)

# junta todo bodyId que aparece em qualquer CSV de conectividade ja gerado
ids = set()
for csv in sorted(HERE.glob("*.csv")):
    if csv.name == "neuron_properties.csv":
        continue
    df = pd.read_csv(csv)
    for col in ("bodyId_pre", "bodyId_post"):
        if col in df.columns:
            ids |= set(df[col].dropna().astype(int))

print(f"bodyIds unicos nos CSVs de conectividade: {len(ids)}")

COLS = [
    "bodyId", "instance", "type", "somaSide",
    "consensusNt", "predictedNt", "predictedNtConfidence",
    "class", "subclass", "superclass",
]

neurons, _ = fetch_neurons(NC(bodyId=sorted(ids)))
out = neurons[[c for c in COLS if c in neurons.columns]].copy()
out.to_csv(HERE / "neuron_properties.csv", index=False)

print(f"\nSalvo: neuron_properties.csv ({len(out)} neuronios)")
print("\nconsensusNt:")
print(out["consensusNt"].value_counts(dropna=False).to_string())
print("\nsomaSide:")
print(out["somaSide"].value_counts(dropna=False).to_string())
