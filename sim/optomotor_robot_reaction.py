"""Mesma logica do digital_robot_reaction.py, agora pro circuito optomotor:
disparo do motor de perna -> impulso de giro no heading do robo (nao mais
pulo pra tras, agora vira, que eh o papel real do DNa02/Sternal rotator MN).

Agora com os dois hemisferios: roda o circuito duas vezes, estimulando um olho
de cada vez, e o heading do robo vai pra lados OPOSTOS. A direcao nao esta
escrita em lugar nenhum do codigo -- ela sai de qual motoneuronio de perna
disparou, e os tres estagios do circuito nao cruzam a linha media no conectoma.

Unica suposicao nossa: que motor de um lado gira o robo PARA aquele lado. O
circuito entrega o lado; a biomecanica da coxa nao esta no conectoma.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    Network, NeuronGroup, Synapses, SpikeMonitor, ms, mV, Hz, defaultclock,
)

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST,
    load_properties, signed_weights, side_of,
)

HERE = Path(__file__).parent
props = load_properties()

sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")

sensor_ids = sorted(sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(sensor_hs["bodyId_post"]) | set(hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(hs_dna02["bodyId_post"]) | set(dna02_motor["bodyId_pre"]))
motor_ids = sorted(dna02_motor["bodyId_post"].unique().tolist())

sensor_side = [side_of(props, b) for b in sensor_ids]
motor_side = [side_of(props, b) for b in motor_ids]

STIM_MS = 300
YAW_FORTE_HZ, YAW_FRACO_HZ = 200, 15


def run_eye(olho, seed=0):
    """Estimula um olho. Devolve os tempos de spike do motor de cada lado."""
    np.random.seed(seed)
    defaultclock.dt = 0.1 * ms

    six = {b: i for i, b in enumerate(sensor_ids)}
    hix = {b: i for i, b in enumerate(hs_ids)}
    dix = {b: i for i, b in enumerate(dna02_ids)}
    mix = {b: i for i, b in enumerate(motor_ids)}

    Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
    HS = NeuronGroup(len(hs_ids), LIF_EQS, **LIF_KWARGS)
    HS.v = V_REST
    DNa02 = NeuronGroup(len(dna02_ids), LIF_EQS, **LIF_KWARGS)
    DNa02.v = V_REST
    Motor = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS)
    Motor.v = V_REST

    synapses = []
    for src, tgt, conn, si, ti in [
        (Sensor, HS, sensor_hs, six, hix),
        (HS, DNa02, hs_dna02, hix, dix),
        (DNa02, Motor, dna02_motor, dix, mix),
    ]:
        agg = signed_weights(conn, props)
        agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
        S = Synapses(src, tgt, "w : volt", on_pre="g_post += w", delay=T_DELAY)
        S.connect(i=[si[b] for b in agg["bodyId_pre"]], j=[ti[b] for b in agg["bodyId_post"]])
        S.w = agg["w_mV"].values * mV
        synapses.append(S)

    Sensor.rate = [YAW_FORTE_HZ if s == olho else YAW_FRACO_HZ for s in sensor_side] * Hz
    mon = SpikeMonitor(Motor)
    Network(Sensor, HS, DNa02, Motor, *synapses, mon).run(STIM_MS * ms)

    t_ms = np.array(mon.t / ms)
    idx = np.array(mon.i)
    out = {}
    for lado in ("L", "R"):
        alvo = [i for i, s in enumerate(motor_side) if s == lado]
        out[lado] = np.sort(t_ms[np.isin(idx, alvo)])
    return out


def heading_trace(spikes_L, spikes_R):
    """Integra os spikes num heading. Esquerda gira negativo, direita positivo."""
    dt = 0.5
    t = np.arange(0, STIM_MS, dt)
    heading = np.zeros_like(t)
    TURN_IMPULSE = 4.0   # graus por spike de motor
    RELAX_TAU = 40.0     # volta pro reto entre disparos (ms)
    h = 0.0
    for i, ti in enumerate(t):
        h += (0.0 - h) * (dt / RELAX_TAU)
        h -= TURN_IMPULSE * np.sum((spikes_L > ti - dt) & (spikes_L <= ti))
        h += TURN_IMPULSE * np.sum((spikes_R > ti - dt) & (spikes_R <= ti))
        heading[i] = h
    return t, heading


fig, ax = plt.subplots(figsize=(9, 5))
cores = {"L": "#1f77b4", "R": "#ff7f0e"}
for olho in ("L", "R"):
    sp = run_eye(olho)
    t, heading = heading_trace(sp["L"], sp["R"])
    print(f"olho {olho}: motor L {len(sp['L'])} spikes, motor R {len(sp['R'])} spikes, "
          f"heading final {heading[-1]:+.1f} graus")
    ax.plot(t, heading, color=cores[olho], lw=2,
            label=f"fluxo optico no olho {olho}  "
                  f"(motor L={len(sp['L'])}, R={len(sp['R'])})")

ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("tempo (ms)")
ax.set_ylabel("heading do robo (graus)\n<0 = virou p/ esquerda   >0 = p/ direita")
ax.set_title("Robo vira para lados opostos conforme o olho estimulado\n"
             "a direcao sai do motoneuronio que disparou, nao esta hardcoded", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out = HERE.parent / "docs" / "images" / "optomotor_robot_reaction.png"
plt.savefig(out, dpi=120)
print(f"Salvo: {out}")
