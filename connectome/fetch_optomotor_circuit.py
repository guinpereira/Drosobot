"""
Circuito 2: T4/T5 (deteccao de movimento) -> HS (wide-field) -> DNa02 (steering)
-> Sternal anterior rotator MN (motoneuronio de perna, gira a coxa == virar).
Confirmado real e conectado em check_optomotor_connectivity.py antes de puxar tudo.
"""
from pathlib import Path
import pandas as pd
from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

token = Path(__file__).parent.parent.joinpath(".neuprint_token").read_text().strip()
client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)

pd.set_option("display.max_rows", 30)
pd.set_option("display.width", 140)

HERE = Path(__file__).parent

# HS e DNa02 sao poucos, pega todos
hs_neurons, _ = fetch_neurons(NC(type=".*HS.*", regex=True))
hs_ids = hs_neurons["bodyId"].tolist()
dna02_neurons, _ = fetch_neurons(NC(type="DNa02"))
dna02_ids = dna02_neurons["bodyId"].tolist()
print("HS ids:", hs_ids)
print("DNa02 ids:", dna02_ids)

# T4/T5 -> HS: pega top 20 pre-sinapticos por peso total (proxy "sensor de movimento")
_, conn_t_hs = fetch_adjacencies(NC(type="T4.*|T5.*", regex=True), NC(bodyId=hs_ids))
sensor_w = conn_t_hs.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False)
sensor_ids = sensor_w.head(20).index.tolist()
print(f"\nTop 20 T4/T5 (peso total {sensor_w.head(20).sum()}):", sensor_ids[:5], "...")

conn_sensor_hs = conn_t_hs[conn_t_hs["bodyId_pre"].isin(sensor_ids)]
conn_sensor_hs.to_csv(HERE / "opto_sensor_hs.csv", index=False)

# HS -> DNa02 (todos, ja e pouco: 12 pares)
_, conn_hs_dna02 = fetch_adjacencies(NC(bodyId=hs_ids), NC(bodyId=dna02_ids))
conn_hs_dna02.to_csv(HERE / "opto_hs_dna02.csv", index=False)
print(f"\nHS->DNa02: {len(conn_hs_dna02)} pares, peso total {conn_hs_dna02['weight'].sum()}")

# DNa02 -> Sternal anterior rotator MN
motor_neurons, _ = fetch_neurons(NC(type="Sternal anterior rotator MN"))
motor_ids = motor_neurons["bodyId"].tolist()
_, conn_dna02_motor = fetch_adjacencies(NC(bodyId=dna02_ids), NC(bodyId=motor_ids))
conn_dna02_motor.to_csv(HERE / "opto_dna02_motor.csv", index=False)
print(f"DNa02->MotorPerna: {len(conn_dna02_motor)} pares, peso total {conn_dna02_motor['weight'].sum()}")
print("Motor ids:", motor_ids)

print("\nSalvo: opto_sensor_hs.csv, opto_hs_dna02.csv, opto_dna02_motor.csv")
