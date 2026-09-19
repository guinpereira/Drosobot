"""
Item 2 (versao leve, sem engine externo): pega o spike do motor (TTMn) da
simulacao Giant Fiber e anima um "robo digital" 1D reagindo -- pulo pra
tras a cada disparo, igual o musculo de pulo real faria.

Sem Unity, sem Webots -- ainda nao precisa. So prova que da pra conectar
spike -> movimento antes de qualquer hardware.

A rede aqui e a mesma de sim/giant_fiber_network.py: sinal da sinapse pelo
neurotransmissor real e biofisica de Shiu et al. 2024 (sim/connectome_model.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    Network, NeuronGroup, Synapses, SpikeMonitor, TimedArray,
    ms, mV, Hz, defaultclock,
)

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST, TONIC_INHIB_HZ,
    load_properties, signed_weights, gf_input_population,
)

HERE = Path(__file__).parent
props = load_properties()

up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
GF_IDS = [10001, 10010]
motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()
sensor_ids, is_looming, is_inhib = gf_input_population(props, up)

defaultclock.dt = 0.1 * ms
STIM_MS = 300
LOOM_MAX_HZ = 20  # por celula LC4/LPLC2 (sao 311)

GF = NeuronGroup(len(GF_IDS), LIF_EQS, **LIF_KWARGS)
GF.v = V_REST
Motor = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS)
Motor.v = V_REST

# rampa de looming nos LC4/LPLC2, tonico nos inibitorios, 0 Hz no resto
# (mesma estrutura de giant_fiber_network.py, ver comentarios la)
ramp = np.linspace(0, LOOM_MAX_HZ, int(STIM_MS * ms / defaultclock.dt)) * Hz
rate_ta = TimedArray(ramp, dt=defaultclock.dt)
Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
Sensor.variables.add_array("looming", size=len(sensor_ids), dtype=bool)
Sensor.looming = np.array(is_looming)
Sensor.variables.add_array("tonico", size=len(sensor_ids), dtype=bool)
Sensor.tonico = np.array(is_inhib)
Sensor.run_regularly(
    "rate = int(looming) * rate_ta(t) + int(tonico) * TONIC_INHIB_HZ * Hz",
    dt=defaultclock.dt,
)

sensor_ix = {b: i for i, b in enumerate(sensor_ids)}
gf_ix = {b: i for i, b in enumerate(GF_IDS)}
motor_ix = {b: i for i, b in enumerate(motor_ids)}

synapses = []
for src, tgt, conn, si, ti in [
    (Sensor, GF, up, sensor_ix, gf_ix),
    (GF, Motor, down, gf_ix, motor_ix),
]:
    agg = signed_weights(conn, props)
    agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
    S = Synapses(src, tgt, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S.connect(i=[si[b] for b in agg["bodyId_pre"]], j=[ti[b] for b in agg["bodyId_post"]])
    S.w = agg["w_mV"].values * mV
    synapses.append(S)

mon_motor = SpikeMonitor(Motor)
# Network explicito: o run() magico nao enxerga objetos guardados dentro de lista
net = Network(Sensor, GF, Motor, *synapses, mon_motor)
net.run(STIM_MS * ms)

escape_times_ms = np.sort(np.array(mon_motor.t / ms))
print("Disparos motor (escape):", np.round(escape_times_ms, 1))

# --- "Fisica" 1D bem simples do robo ---
dt = 0.5  # ms
t = np.arange(0, STIM_MS, dt)
x = np.zeros_like(t)
JUMP = -18.0        # pulo pra tras a cada escape
RELAX_TAU = 25.0    # volta pra posicao neutra depois do pulo (ms)

pos = 0.0
for i, ti in enumerate(t):
    pos += (0.0 - pos) * (dt / RELAX_TAU)
    if np.any((escape_times_ms > ti - dt) & (escape_times_ms <= ti)):
        pos += JUMP
    x[i] = pos

# "objeto se aproximando": a rampa 0->20 Hz mapeada em distancia 100cm -> 0cm
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
out = HERE.parent / "docs" / "images" / "digital_robot_reaction.png"
plt.savefig(out, dpi=120)
print(f"Salvo: {out}")
