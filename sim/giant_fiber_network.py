"""
Circuito Giant Fiber (DNp01) com peso E SINAL reais do Male CNS.
sensor (top upstream por peso) -> GF (DNp01 L/R) -> TTMn (motoneuronio de pulo)

Duas coisas mudaram em relacao a v1:

- o sinal da sinapse agora vem do neurotransmissor do neuronio pre-sinaptico,
  nao e mais "tudo excitatorio". Isso importa muito aqui: 6 dos 8 upstream mais
  fortes do Giant Fiber sao GABA ou glutamato, e carregam 59% do peso total.
- LIF unitless com ganho por camada deu lugar a biofisica publicada de
  Shiu et al. 2024 (ver sim/connectome_model.py). W_SYN e o unico parametro livre.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, StateMonitor, TimedArray,
    run, ms, mV, Hz, start_scope, defaultclock,
)

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST, V_RESET, V_THRESHOLD,
    TAU_MBR, TAU_SYN, load_properties, signed_weights, describe_drive, nt_sign,
)

HERE = Path(__file__).parent
props = load_properties()

up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")

GF_IDS = [10001, 10010]  # DNp01 R, L
sensor_ids = up.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False).head(8).index.tolist()
motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()

print("Sensores (top-8 upstream do GF):")
for b in sensor_ids:
    peso = int(up[up["bodyId_pre"] == b]["weight"].sum())
    s = nt_sign(props, b)
    print(f"  {props.at[b,'instance']:<16} {props.at[b,'consensusNt']:<14} "
          f"{'excita' if s > 0 else 'INIBE ':<7} peso {peso}")

start_scope()
defaultclock.dt = 0.1 * ms
STIM = 300 * ms

GF = NeuronGroup(len(GF_IDS), LIF_EQS, **LIF_KWARGS)
GF.v = V_REST
Motor = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS)
Motor.v = V_REST

# Estimulo. Nem todo upstream do GF e um detector de looming: dos 8 mais fortes,
# so os DNp70 (colinergicos) estao no caminho visual de aproximacao. SAD073/091/109
# sao subesofagicos e PVLP010 e glutamatergico -- disparar todos eles em rampa junto
# com o objeto se aproximando nao e biologia, e artefato de ter escolhido "sensor"
# por peso bruto, quando o sinal ainda era ignorado.
#
# Entao: rampa de looming so no caminho excitatorio; os inibitorios ficam em taxa
# tonica de base. Isso da a inibicao o papel que a literatura do Giant Fiber
# descreve -- portao que segura o pulo, evitando alarme falso -- em vez de um
# freio que cresce junto com o proprio estimulo.
#
# TONIC_INHIB_HZ e SUPOSICAO nossa, nao vem do conectoma: o dado diz quem inibe e
# com que forca, nao a que taxa esses neuronios disparam em repouso.
TONIC_INHIB_HZ = 20 * Hz

# 5 -> 120 Hz. A faixa importa agora que o modelo tem unidade: Shiu et al.
# calibraram W_SYN com entrada sensorial em torno de 100 Hz. A rampa ate 400 Hz
# da versao antiga foi ajustada pro LIF unitless e aqui satura o circuito.
ramp = np.linspace(5, 120, int(STIM / defaultclock.dt)) * Hz
rate_ta = TimedArray(ramp, dt=defaultclock.dt)
Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
excitatory_mask = np.array([nt_sign(props, b) > 0 for b in sensor_ids])
Sensor.rate = TONIC_INHIB_HZ
Sensor.run_regularly("rate = int(is_looming) * rate_ta(t) + (1 - int(is_looming)) * TONIC_INHIB_HZ",
                     dt=defaultclock.dt)
Sensor.variables.add_array("is_looming", size=len(sensor_ids), dtype=bool)
Sensor.is_looming = excitatory_mask

sensor_ix = {b: i for i, b in enumerate(sensor_ids)}
gf_ix = {b: i for i, b in enumerate(GF_IDS)}
motor_ix = {b: i for i, b in enumerate(motor_ids)}


def connect(source, target, conn, src_ix, tgt_ix, label):
    agg = signed_weights(conn, props)
    agg = agg[agg["bodyId_pre"].isin(src_ix) & agg["bodyId_post"].isin(tgt_ix)]
    describe_drive(agg, props, f"  {label}")
    S = Synapses(source, target, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S.connect(i=[src_ix[b] for b in agg["bodyId_pre"]],
              j=[tgt_ix[b] for b in agg["bodyId_post"]])
    S.w = agg["w_mV"].values * mV
    return S


print("\nBalanco do drive (sinapses EM, sinal pelo neurotransmissor):")
S_sg = connect(Sensor, GF, up, sensor_ix, gf_ix, "sensor -> GF   ")
S_gm = connect(GF, Motor, down, gf_ix, motor_ix, "GF     -> TTMn ")

mon_sensor = SpikeMonitor(Sensor)
mon_gf = SpikeMonitor(GF)
mon_motor = SpikeMonitor(Motor)
trace_gf = StateMonitor(GF, "v", record=True)

run(STIM)

print(f"\nSpikes -- Sensor: {mon_sensor.num_spikes}  GF: {mon_gf.num_spikes}  TTMn: {mon_motor.num_spikes}")
if mon_motor.num_spikes:
    print(f">>> ESCAPE em t={float(mon_motor.t[0]/ms):.1f}ms")
else:
    v_max = float(trace_gf.v.max() / mV)
    print(f">>> Sem escape. Pico do GF: {v_max:.2f} mV (limiar {float(V_THRESHOLD/mV):.0f} mV, "
          f"repouso {float(V_REST/mV):.0f} mV)")

# ---------- raster ----------
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8),
                         gridspec_kw={"height_ratios": [2, 2, 1, 1]})

for i, b in enumerate(sensor_ids):
    t = mon_sensor.t[mon_sensor.i == i] / ms
    exc = nt_sign(props, b) > 0
    axes[0].plot(t, np.full_like(t, i), ".", ms=2,
                 color="#2a7fff" if exc else "#d62728")
axes[0].set_yticks(range(len(sensor_ids)))
axes[0].set_yticklabels([props.at[b, "instance"] for b in sensor_ids], fontsize=7)
axes[0].set_ylabel("Sensores")
axes[0].set_title("Giant Fiber -- Male CNS, sinal da sinapse pelo neurotransmissor real\n"
                  "azul = excitatorio (ACh)   vermelho = inibitorio (GABA/glutamato)", fontsize=10)

for i, b in enumerate(GF_IDS):
    axes[1].plot(trace_gf.t / ms, trace_gf.v[i] / mV, lw=1,
                 label=props.at[b, "instance"])
axes[1].axhline(float(V_THRESHOLD / mV), color="k", ls="--", lw=0.8, label="limiar")
axes[1].axhline(float(V_REST / mV), color="gray", ls=":", lw=0.8, label="repouso")
axes[1].set_ylabel("GF  v (mV)")
axes[1].legend(fontsize=7, loc="upper left")

axes[2].plot(mon_gf.t / ms, mon_gf.i, "o", color="crimson", ms=5)
axes[2].set_ylabel("GF spikes")
axes[2].set_ylim(-0.5, len(GF_IDS) - 0.5)

axes[3].plot(mon_motor.t / ms, mon_motor.i, "s", color="darkgreen", ms=5)
axes[3].set_ylabel("TTMn")
axes[3].set_ylim(-0.5, max(len(motor_ids) - 0.5, 0.5))
axes[3].set_xlabel("tempo (ms) -- taxa do sensor sobe em rampa (looming)")

plt.tight_layout()
out = HERE.parent / "docs" / "images" / "giant_fiber_raster.png"
plt.savefig(out, dpi=120)
print(f"Raster salvo em {out}")
