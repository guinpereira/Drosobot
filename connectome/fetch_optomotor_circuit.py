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

# T4/T5 -> HS: top pre-sinapticos POR HEMISFERIO.
# Antes era top-20 no geral, e isso caia 17 do lado R contra 3 do L. Com a
# amostra tao torta nao da pra tirar direcao de giro do circuito, porque um dos
# lados quase nao existe na rede. Amostrando N por lado, os dois canais
# ipsilaterais (T4/T5 -> HS -> DNa02 -> motor de perna, que nos dados nao cruzam
# a linha media) ficam comparaveis.
TOP_POR_LADO = 10

_, conn_t_hs = fetch_adjacencies(NC(type="T4.*|T5.*", regex=True), NC(bodyId=hs_ids))
t_neurons, _ = fetch_neurons(NC(bodyId=sorted(conn_t_hs["bodyId_pre"].unique().tolist())))
side_of = dict(zip(t_neurons["bodyId"], t_neurons["somaSide"]))

sensor_w = conn_t_hs.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False)
sensor_ids = []
for lado in ("L", "R"):
    do_lado = [b for b in sensor_w.index if side_of.get(b) == lado][:TOP_POR_LADO]
    sensor_ids.extend(do_lado)
    print("Top %d T4/T5 do lado %s (peso total %d): %s ..."
          % (TOP_POR_LADO, lado, int(sensor_w[do_lado].sum()), do_lado[:3]))

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
