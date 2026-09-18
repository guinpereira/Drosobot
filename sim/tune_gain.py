"""Varre GAIN + refractory pra achar ponto onde GF fica em silencio
e so dispara (poucas vezes) quando o looming fica forte -- mais fiel
ao comportamento real documentado do Giant Fiber (resposta rara e precisa,
nao disparo continuo).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, TimedArray, Hz,
    run, ms, start_scope, defaultclock
)

HERE = Path(__file__).parent
up = pd.read_csv(HERE.parent / "connectome" / "gf_upstream_connections.csv")
down = pd.read_csv(HERE.parent / "connectome" / "gf_downstream_connections.csv")
GF_IDS = [10001, 10010]

up_w = up.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False)
sensor_ids = up_w.head(8).index.tolist()
down_ttmn = down[down["type"] == "TTMn"]
motor_ids = down_ttmn["bodyId_post"].unique().tolist()

rows_sg = up[up["bodyId_pre"].isin(sensor_ids) & up["bodyId_post"].isin(GF_IDS)]
w_sg = rows_sg.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
rows_gm = down[down["bodyId_pre"].isin(GF_IDS) & down["bodyId_post"].isin(motor_ids)]
w_gm = rows_gm.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()

sensor_index = {b: i for i, b in enumerate(sensor_ids)}
gf_index = {b: i for i, b in enumerate(GF_IDS)}
motor_index = {b: i for i, b in enumerate(motor_ids)}


def run_trial(gain, gf_refrac_ms, stim_ms=300):
    start_scope()
    defaultclock.dt = 0.1 * ms
    eqs = "dv/dt = -v / tau : 1\ntau : second"
    GF = NeuronGroup(len(GF_IDS), eqs, threshold="v>1", reset="v=0",
                      refractory=gf_refrac_ms * ms, method="euler")
    GF.tau = 10 * ms
    Motor = NeuronGroup(len(motor_ids), eqs, threshold="v>1", reset="v=0",
                         refractory=5 * ms, method="euler")
    Motor.tau = 10 * ms

    rate_arr = np.linspace(5, 400, int(stim_ms * ms / defaultclock.dt)) * Hz
    rate_ta = TimedArray(rate_arr, dt=defaultclock.dt)
    Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
    Sensor.run_regularly("rate = rate_ta(t)", dt=defaultclock.dt)

    S1 = Synapses(Sensor, GF, "w : 1", on_pre="v_post += w")
    S1.connect(i=[sensor_index[b] for b in w_sg["bodyId_pre"]],
               j=[gf_index[b] for b in w_sg["bodyId_post"]])
    S1.w = w_sg["weight"].values * gain

    S2 = Synapses(GF, Motor, "w : 1", on_pre="v_post += w")
    S2.connect(i=[gf_index[b] for b in w_gm["bodyId_pre"]],
               j=[motor_index[b] for b in w_gm["bodyId_post"]])
    S2.w = w_gm["weight"].values * gain

    mon_gf = SpikeMonitor(GF)
    mon_motor = SpikeMonitor(Motor)
    run(stim_ms * ms)

    first_gf = float(mon_gf.t[0] / ms) if mon_gf.num_spikes else None
    first_motor = float(mon_motor.t[0] / ms) if mon_motor.num_spikes else None
    return mon_gf.num_spikes, mon_motor.num_spikes, first_gf, first_motor


print(f"{'GAIN':>8} {'refrac':>7} {'GF#':>5} {'Motor#':>7} {'t_GF':>7} {'t_Motor':>8}")
for gain in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]:
    for refrac in [20, 50]:
        n_gf, n_motor, t_gf, t_motor = run_trial(gain, refrac)
        print(f"{gain:>8} {refrac:>7} {n_gf:>5} {n_motor:>7} "
              f"{t_gf if t_gf else '-':>7} {t_motor if t_motor else '-':>8}")
