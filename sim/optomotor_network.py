"""
Circuito 2: T4/T5 (sensor de movimento) -> HS (wide-field) -> DNa02 (steering)
-> Sternal anterior rotator MN (motor de perna, vira). 4 camadas, peso real
do Male CNS em cada uma. Mesma logica/simplificacoes do circuito 1 (Giant Fiber):
tudo excitatorio, LIF unitless, ganho por camada ajustado manualmente (peso
de sinapse EM nao eh condutancia calibrada).
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

print(f"Sensor(T4/T5): {len(sensor_ids)}  HS: {len(hs_ids)}  DNa02: {len(dna02_ids)}  Motor(perna): {len(motor_ids)}")

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

# estimulo: "velocidade de movimento visual" sobe em rampa (pan/rotacao ficando mais rapida)
rate_arr = np.linspace(5, 300, int(STIM_MS * ms / defaultclock.dt)) * Hz
rate_ta = TimedArray(rate_arr, dt=defaultclock.dt)
Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
Sensor.run_regularly("rate = rate_ta(t)", dt=defaultclock.dt)

sensor_index = {b: i for i, b in enumerate(sensor_ids)}
hs_index = {b: i for i, b in enumerate(hs_ids)}
dna02_index = {b: i for i, b in enumerate(dna02_ids)}
motor_index = {b: i for i, b in enumerate(motor_ids)}

GAIN_SENSOR_HS = 0.002
GAIN_HS_DNA02 = 0.03
GAIN_DNA02_MOTOR = 0.02

w1 = sensor_hs.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S1 = Synapses(Sensor, HS, "w : 1", on_pre="v_post += w")
S1.connect(i=[sensor_index[b] for b in w1["bodyId_pre"]], j=[hs_index[b] for b in w1["bodyId_post"]])
S1.w = w1["weight"].values * GAIN_SENSOR_HS

w2 = hs_dna02.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S2 = Synapses(HS, DNa02, "w : 1", on_pre="v_post += w")
S2.connect(i=[hs_index[b] for b in w2["bodyId_pre"]], j=[dna02_index[b] for b in w2["bodyId_post"]])
S2.w = w2["weight"].values * GAIN_HS_DNA02

w3 = dna02_motor.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S3 = Synapses(DNa02, Motor, "w : 1", on_pre="v_post += w")
S3.connect(i=[dna02_index[b] for b in w3["bodyId_pre"]], j=[motor_index[b] for b in w3["bodyId_post"]])
S3.w = w3["weight"].values * GAIN_DNA02_MOTOR

mon_sensor = SpikeMonitor(Sensor)
mon_hs = SpikeMonitor(HS)
mon_dna02 = SpikeMonitor(DNa02)
mon_motor = SpikeMonitor(Motor)

run(STIM_MS * ms)

print(f"Spikes -- Sensor:{mon_sensor.num_spikes}  HS:{mon_hs.num_spikes}  "
      f"DNa02:{mon_dna02.num_spikes}  Motor(perna):{mon_motor.num_spikes}")
if mon_motor.num_spikes:
    print(f"Primeiro comando de giro em t={float(mon_motor.t[0]/ms):.1f}ms")
else:
    print("Sem giro -- ganho fraco demais, ajustar.")

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 7))
for ax, mon, label, color in [
    (axes[0], mon_sensor, "T4/T5\n(movimento)", "gray"),
    (axes[1], mon_hs, "HS\n(wide-field)", "orange"),
    (axes[2], mon_dna02, "DNa02\n(steering)", "crimson"),
    (axes[3], mon_motor, "Motor perna\n(giro)", "darkgreen"),
]:
    ax.plot(mon.t / ms, mon.i, ".", color=color, ms=4)
    ax.set_ylabel(label)
axes[0].set_title("Circuito optomotor -- dado real Male CNS (T4/T5 -> HS -> DNa02 -> motor de perna)")
axes[3].set_xlabel("tempo (ms) -- velocidade de movimento visual sobe em rampa")
plt.tight_layout()
out = HERE / "optomotor_raster.png"
plt.savefig(out, dpi=120)
print(f"Salvo: {out}")
