"""
Item 2 (versao leve, sem engine externo): pega o spike do motor (TTMn) da
simulacao Giant Fiber e anima um "robo digital" 1D reagindo -- pulo pra
tras a cada disparo, igual o musculo de pulo real faria.

Sem Unity, sem Webots -- ainda nao precisa. So prova que da pra conectar
spike -> movimento antes de qualquer hardware.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, TimedArray, Hz,
    run, ms, start_scope, defaultclock
)
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
up = pd.read_csv(HERE.parent / "connectome" / "gf_upstream_connections.csv")
down = pd.read_csv(HERE.parent / "connectome" / "gf_downstream_connections.csv")
GF_IDS = [10001, 10010]

up_w = up.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False)
sensor_ids = up_w.head(8).index.tolist()
down_ttmn = down[down["type"] == "TTMn"]
motor_ids = down_ttmn["bodyId_post"].unique().tolist()

start_scope()
defaultclock.dt = 0.1 * ms
STIM_MS = 300

eqs = "dv/dt = -v / tau : 1\ntau : second"
GF = NeuronGroup(len(GF_IDS), eqs, threshold="v>1", reset="v=0", refractory=50 * ms, method="euler")
GF.tau = 10 * ms
Motor = NeuronGroup(len(motor_ids), eqs, threshold="v>1", reset="v=0", refractory=5 * ms, method="euler")
Motor.tau = 10 * ms

rate_arr = np.linspace(5, 400, int(STIM_MS * ms / defaultclock.dt)) * Hz
rate_ta = TimedArray(rate_arr, dt=defaultclock.dt)
Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
Sensor.run_regularly("rate = rate_ta(t)", dt=defaultclock.dt)

sensor_index = {b: i for i, b in enumerate(sensor_ids)}
gf_index = {b: i for i, b in enumerate(GF_IDS)}
motor_index = {b: i for i, b in enumerate(motor_ids)}

rows_sg = up[up["bodyId_pre"].isin(sensor_ids) & up["bodyId_post"].isin(GF_IDS)]
w_sg = rows_sg.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S1 = Synapses(Sensor, GF, "w : 1", on_pre="v_post += w")
S1.connect(i=[sensor_index[b] for b in w_sg["bodyId_pre"]], j=[gf_index[b] for b in w_sg["bodyId_post"]])
S1.w = w_sg["weight"].values * 0.001

rows_gm = down[down["bodyId_pre"].isin(GF_IDS) & down["bodyId_post"].isin(motor_ids)]
w_gm = rows_gm.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S2 = Synapses(GF, Motor, "w : 1", on_pre="v_post += w")
S2.connect(i=[gf_index[b] for b in w_gm["bodyId_pre"]], j=[motor_index[b] for b in w_gm["bodyId_post"]])
S2.w = w_gm["weight"].values * 0.02

mon_motor = SpikeMonitor(Motor)
run(STIM_MS * ms)

escape_times_ms = np.sort(np.array(mon_motor.t / ms))
print("Disparos motor (escape):", escape_times_ms)

# --- "Fisica" 1D bem simples do robo ---
dt = 0.5  # ms
t = np.arange(0, STIM_MS, dt)
x = np.zeros_like(t)          # posicao do robo (0 = parado)
JUMP = -18.0                  # pulo pra tras a cada escape
RELAX_TAU = 25.0              # volta pra posicao neutra depois do pulo (ms)

pos = 0.0
for i, ti in enumerate(t):
    # relaxa de volta pro zero
    pos += (0.0 - pos) * (dt / RELAX_TAU)
    # se cruzou um tempo de escape desde o ultimo passo, aplica pulo
    if np.any((escape_times_ms > ti - dt) & (escape_times_ms <= ti)):
        pos += JUMP
    x[i] = pos

# "objeto se aproximando" (looming) -- mesma semantica da rampa de estimulo:
# rate 5->400Hz mapeado pra distancia 100cm -> 0cm (quanto maior rate, mais perto)
looming_distance = np.interp(t, [0, STIM_MS], [100, 0])

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
axes[0].plot(t, looming_distance, color="black")
axes[0].invert_yaxis()
axes[0].set_ylabel("distancia\nobjeto (cm)")
axes[0].set_title("Robo digital reagindo ao circuito real Giant Fiber (sem hardware)")
for et in escape_times_ms:
    axes[0].axvline(et, color="crimson", alpha=0.3, lw=1)

axes[1].plot(t, x, color="darkgreen")
axes[1].set_ylabel("posicao robo\n(0=parado, <0=pulou p/ tras)")
axes[1].set_xlabel("tempo (ms)")
for et in escape_times_ms:
    axes[1].axvline(et, color="crimson", alpha=0.3, lw=1, label="_nolegend_")

plt.tight_layout()
out = HERE / "digital_robot_reaction.png"
plt.savefig(out, dpi=120)
print(f"Salvo: {out}")
