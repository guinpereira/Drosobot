"""Mesma logica do digital_robot_reaction.py, agora pro circuito optomotor:
disparo do motor de perna -> impulso de giro no heading do robo (nao mais
pulo pra tras, agora vira, que eh o papel real do DNa02/Sternal rotator MN).
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
sensor_hs = pd.read_csv(HERE.parent / "connectome" / "opto_sensor_hs.csv")
hs_dna02 = pd.read_csv(HERE.parent / "connectome" / "opto_hs_dna02.csv")
dna02_motor = pd.read_csv(HERE.parent / "connectome" / "opto_dna02_motor.csv")

sensor_ids = sorted(sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(sensor_hs["bodyId_post"]) | set(hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(hs_dna02["bodyId_post"]) | set(dna02_motor["bodyId_pre"]))
motor_ids = sorted(dna02_motor["bodyId_post"].unique().tolist())

start_scope()
defaultclock.dt = 0.1 * ms
STIM_MS = 300
eqs = "dv/dt = -v / tau : 1\ntau : second"

HS = NeuronGroup(len(hs_ids), eqs, threshold="v>1", reset="v=0", refractory=20 * ms, method="euler")
HS.tau = 10 * ms
DNa02 = NeuronGroup(len(dna02_ids), eqs, threshold="v>1", reset="v=0", refractory=20 * ms, method="euler")
DNa02.tau = 10 * ms
Motor = NeuronGroup(len(motor_ids), eqs, threshold="v>1", reset="v=0", refractory=10 * ms, method="euler")
Motor.tau = 10 * ms

rate_arr = np.linspace(5, 300, int(STIM_MS * ms / defaultclock.dt)) * Hz
rate_ta = TimedArray(rate_arr, dt=defaultclock.dt)
Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
Sensor.run_regularly("rate = rate_ta(t)", dt=defaultclock.dt)

sensor_index = {b: i for i, b in enumerate(sensor_ids)}
hs_index = {b: i for i, b in enumerate(hs_ids)}
dna02_index = {b: i for i, b in enumerate(dna02_ids)}
motor_index = {b: i for i, b in enumerate(motor_ids)}

w1 = sensor_hs.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S1 = Synapses(Sensor, HS, "w : 1", on_pre="v_post += w")
S1.connect(i=[sensor_index[b] for b in w1["bodyId_pre"]], j=[hs_index[b] for b in w1["bodyId_post"]])
S1.w = w1["weight"].values * 0.002

w2 = hs_dna02.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S2 = Synapses(HS, DNa02, "w : 1", on_pre="v_post += w")
S2.connect(i=[hs_index[b] for b in w2["bodyId_pre"]], j=[dna02_index[b] for b in w2["bodyId_post"]])
S2.w = w2["weight"].values * 0.03

w3 = dna02_motor.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S3 = Synapses(DNa02, Motor, "w : 1", on_pre="v_post += w")
S3.connect(i=[dna02_index[b] for b in w3["bodyId_pre"]], j=[motor_index[b] for b in w3["bodyId_post"]])
S3.w = w3["weight"].values * 0.02

mon_motor = SpikeMonitor(Motor)
run(STIM_MS * ms)

turn_times_ms = np.sort(np.array(mon_motor.t / ms))
print("Disparos motor perna (giro):", turn_times_ms)

dt = 0.5
t = np.arange(0, STIM_MS, dt)
heading = np.zeros_like(t)
TURN_IMPULSE = 4.0     # graus por spike de motor
RELAX_TAU = 40.0       # volta pro reto (graus/ms de decaimento)

h = 0.0
for i, ti in enumerate(t):
    h += (0.0 - h) * (dt / RELAX_TAU)
    n_spikes_now = np.sum((turn_times_ms > ti - dt) & (turn_times_ms <= ti))
    h += TURN_IMPULSE * n_spikes_now
    heading[i] = h

motion_speed = np.interp(t, [0, STIM_MS], [5, 300])  # Hz equivalente, mesma rampa do estimulo

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
axes[0].plot(t, motion_speed, color="black")
axes[0].set_ylabel("velocidade\nmovimento visual (Hz)")
axes[0].set_title("Robo digital virando por optomotor real (T4/T5->HS->DNa02->perna)")
for et in turn_times_ms:
    axes[0].axvline(et, color="darkorange", alpha=0.3, lw=1)

axes[1].plot(t, heading, color="darkgreen")
axes[1].set_ylabel("heading do robo\n(graus, 0=reto)")
axes[1].set_xlabel("tempo (ms)")
for et in turn_times_ms:
    axes[1].axvline(et, color="darkorange", alpha=0.3, lw=1)

plt.tight_layout()
out = HERE / "optomotor_robot_reaction.png"
plt.savefig(out, dpi=120)
print(f"Salvo: {out}")
