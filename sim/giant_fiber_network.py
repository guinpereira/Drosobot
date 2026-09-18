"""
Circuito Giant Fiber (DNp01) com peso sinaptico real do Male CNS.
sensor (proxy visual, top upstream por peso) -> GF (DNp01 L/R) -> TTMn (motor jump)

Simplificacao v1 (documentada, nao escondida):
- todo mundo tratado como excitatorio (nao puxamos neurotransmitter prediction ainda)
- peso sinaptico = contagem de sinapse EM * ganho global (nao eh condutancia calibrada)
- LIF unitless (v 0->1), padrao tutorial Brian2, sem unidade biofisica real
Objetivo aqui NAO eh precisao biofisica, eh provar que o pipeline
neuprint -> peso real -> rede spiking -> comportamento de limiar funciona.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, PopulationRateMonitor,
    run, ms, second, start_scope, defaultclock
)
import matplotlib.pyplot as plt

start_scope()
defaultclock.dt = 0.1 * ms

HERE = Path(__file__).parent
up = pd.read_csv(HERE.parent / "connectome" / "gf_upstream_connections.csv")
down = pd.read_csv(HERE.parent / "connectome" / "gf_downstream_connections.csv")

GF_IDS = [10001, 10010]  # DNp01 R, L

# top 8 pre-sinapticos por peso total = proxy do "sensor visual" (maioria PVLP/DNp -- via visual)
up_agg = up.merge(up[["bodyId_pre"]].drop_duplicates(), on="bodyId_pre")
up_w = up.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False)
sensor_ids = up_w.head(8).index.tolist()

# TTMn = motoneuronio do pulo, alvo motor de interesse
down_ttmn = down[down["type"] == "TTMn"]
motor_ids = down_ttmn["bodyId_post"].unique().tolist()
if not motor_ids:
    motor_ids = down.groupby("bodyId_post")["weight"].sum().sort_values(ascending=False).head(2).index.tolist()

print("Sensor (upstream proxy):", sensor_ids)
print("GF:", GF_IDS)
print("Motor (TTMn):", motor_ids)

n_sensor, n_gf, n_motor = len(sensor_ids), len(GF_IDS), len(motor_ids)

# --- Populacoes LIF unitless ---
eqs = """
dv/dt = -v / tau : 1
tau : second
"""
GF = NeuronGroup(n_gf, eqs, threshold="v>1", reset="v=0", refractory=50 * ms, method="euler")
GF.tau = 10 * ms

Motor = NeuronGroup(n_motor, eqs, threshold="v>1", reset="v=0", refractory=2 * ms, method="euler")
Motor.tau = 10 * ms

# --- Estimulo: "looming" = taxa de disparo do sensor sobe em rampa ao longo do tempo ---
# simula objeto se aproximando: rate baixo (nada acontece) -> rate alto (starts a escape)
stim_duration = 300 * ms
ramp_rate_hz = np.linspace(5, 400, int(stim_duration / defaultclock.dt))
from brian2 import TimedArray, Hz
rate_ta = TimedArray(ramp_rate_hz * Hz, dt=defaultclock.dt)

Sensor = NeuronGroup(
    n_sensor, "rate : Hz", threshold="rand()<rate*dt", method="euler"
)
Sensor.run_regularly("rate = rate_ta(t)", dt=defaultclock.dt)

# --- Sinapses com peso real do conectoma (contagem sinapse * ganho, MVP) ---
# dois ganhos separados: sensor->GF precisa de varios sensores somando pra cruzar limiar
# (integra evidencia de olhar/looming). GF->Motor eh sinapse unica muito forte na biologia
# real (eletrica+quimica), quase 1-pra-1 -- ganho escolhido pra isso: peso max 70 * 0.02 = 1.4 (cruza sozinho)
GAIN_SENSOR_GF = 0.001
GAIN_GF_MOTOR = 0.02

sensor_index = {bid: i for i, bid in enumerate(sensor_ids)}
gf_index = {bid: i for i, bid in enumerate(GF_IDS)}
motor_index = {bid: i for i, bid in enumerate(motor_ids)}

# sensor -> GF
rows_sg = up[up["bodyId_pre"].isin(sensor_ids) & up["bodyId_post"].isin(GF_IDS)]
w_sg = rows_sg.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S_sensor_gf = Synapses(Sensor, GF, "w : 1", on_pre="v_post += w")
i_list = [sensor_index[b] for b in w_sg["bodyId_pre"]]
j_list = [gf_index[b] for b in w_sg["bodyId_post"]]
S_sensor_gf.connect(i=i_list, j=j_list)
S_sensor_gf.w = (w_sg["weight"].values * GAIN_SENSOR_GF)

# GF -> motor
rows_gm = down[down["bodyId_pre"].isin(GF_IDS) & down["bodyId_post"].isin(motor_ids)]
w_gm = rows_gm.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S_gf_motor = Synapses(GF, Motor, "w : 1", on_pre="v_post += w")
i_list2 = [gf_index[b] for b in w_gm["bodyId_pre"]]
j_list2 = [motor_index[b] for b in w_gm["bodyId_post"]]
S_gf_motor.connect(i=i_list2, j=j_list2)
S_gf_motor.w = (w_gm["weight"].values * GAIN_GF_MOTOR)

# --- Monitores ---
mon_sensor = SpikeMonitor(Sensor)
mon_gf = SpikeMonitor(GF)
mon_motor = SpikeMonitor(Motor)

run(stim_duration)

print(f"\nSpikes sensor: {mon_sensor.num_spikes}")
print(f"Spikes GF: {mon_gf.num_spikes}  (tempos: {np.round(mon_gf.t/ms, 1).tolist()} ms)")
print(f"Spikes Motor (TTMn): {mon_motor.num_spikes}  (tempos: {np.round(mon_motor.t/ms, 1).tolist()} ms)")

if mon_motor.num_spikes > 0:
    t_first_escape = float(mon_motor.t[0] / ms)
    print(f"\n>>> ESCAPE disparado em t={t_first_escape:.1f}ms (estimulo looming cruzou limiar)")
else:
    print("\n>>> Sem escape -- GAIN baixo demais ou estimulo nao ficou forte o bastante, ajustar.")

# --- Plot raster ---
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
axes[0].plot(mon_sensor.t / ms, mon_sensor.i, ".", color="gray", ms=2)
axes[0].set_ylabel("Sensor")
axes[0].set_title("Giant Fiber escape circuit -- dado real Male CNS (DNp01)")
axes[1].plot(mon_gf.t / ms, mon_gf.i, "o", color="crimson")
axes[1].set_ylabel("GF (DNp01)")
axes[2].plot(mon_motor.t / ms, mon_motor.i, "s", color="darkgreen")
axes[2].set_ylabel("Motor (TTMn)")
axes[2].set_xlabel("tempo (ms)  -- rate do sensor sobe em rampa (looming)")
plt.tight_layout()
out_path = HERE / "giant_fiber_raster.png"
plt.savefig(out_path, dpi=120)
print(f"\nRaster salvo em {out_path}")
