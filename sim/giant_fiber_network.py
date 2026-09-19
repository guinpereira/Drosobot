"""
Circuito Giant Fiber (DNp01) com a entrada REAL do neuronio, nao um recorte.

LC4 e LPLC2 (deteccao de looming) -> GF (DNp01 L/R) -> TTMn (motoneuronio de pulo),
com os 1271 neuronios pre-sinapticos do GF no modelo, nao os 8 de maior peso.

Tres coisas mudaram em relacao a v1:

- **sensor certo**. Antes o "sensor visual" eram os 8 upstream de maior peso, que
  pegava DNp70 e neuronios SAD e nao incluia nenhum LC4 nem LPLC2 -- justamente os
  tipos que o dataset aponta como a via visual de aproximacao (ver comentario em
  connectome_model.py). Ordenar por peso por celula escondia a via certa, porque
  cada LC4 tem peso pequeno e sao centenas delas.
- **sinal da sinapse** vem do neurotransmissor do pre-sinaptico, nao e mais
  "tudo excitatorio". Dos 1271 upstream, 501 sao inibitorios (13122 sinapses).
- **biofisica publicada** de Shiu et al. 2024 no lugar de LIF sem unidade com
  ganho por camada. W_SYN e o unico parametro livre.

Quem dispara e quem fica quieto, seguindo o protocolo de Shiu et al. (baseline de
0 Hz, so o sensorio recebe estimulo):

- LC4/LPLC2      -> rampa de looming
- inibitorios    -> taxa tonica de base (SUPOSICAO nossa: no cerebro inteiro quem
                    dispara eles e o resto da rede, que nao simulamos)
- todo o resto   -> 0 Hz

Poisson de fundo em todos os 1271 nao funciona: o GF passa a disparar ate em
repouso, porque com W_SYN do Shiu qualquer conexao de 25 sinapses ou mais ja
cruza o limiar com um spike so.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    Network, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, TimedArray,
    ms, mV, Hz, defaultclock,
)

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST, V_THRESHOLD,
    TONIC_INHIB_HZ, load_properties, signed_weights, describe_drive,
    gf_input_population,
)

HERE = Path(__file__).parent
props = load_properties()

up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")

GF_IDS = [10001, 10010]  # DNp01 R, L
motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()
sensor_ids, is_looming, is_inhib = gf_input_population(props, up)

peso = up.groupby("bodyId_pre")["weight"].sum()
n_silencioso = sum(1 for a, b in zip(is_looming, is_inhib) if not a and not b)
print(f"Upstream do GF no modelo: {len(sensor_ids)} neuronios")
print(f"  looming (LC4/LPLC2): {sum(is_looming):4d} celulas, "
      f"{int(peso[np.array(sensor_ids)[is_looming]].sum()):6d} sinapses  -> rampa")
print(f"  inibitorios:         {sum(is_inhib):4d} celulas, "
      f"{int(peso[np.array(sensor_ids)[is_inhib]].sum()):6d} sinapses  -> {TONIC_INHIB_HZ:.0f} Hz tonico")
print(f"  resto:               {n_silencioso:4d} celulas                    -> 0 Hz")

defaultclock.dt = 0.1 * ms
STIM = 300 * ms
# 0 -> 20 Hz POR CELULA LC4/LPLC2. Parece baixo perto da rampa ate 120 Hz da
# versao antiga, mas antes eram 2 celulas colinergicas carregando o estimulo e
# agora sao 311 -- o que chega no GF e muito maior.
LOOM_MAX_HZ = 20

GF = NeuronGroup(len(GF_IDS), LIF_EQS, **LIF_KWARGS)
GF.v = V_REST
Motor = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS)
Motor.v = V_REST

ramp = np.linspace(0, LOOM_MAX_HZ, int(STIM / defaultclock.dt)) * Hz
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

print("\nBalanco do drive (sinapses EM, sinal pelo neurotransmissor):")
synapses = []
for src, tgt, conn, si, ti, label in [
    (Sensor, GF, up, sensor_ix, gf_ix, "upstream -> GF  "),
    (GF, Motor, down, gf_ix, motor_ix, "GF       -> TTMn"),
]:
    agg = signed_weights(conn, props)
    agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
    describe_drive(agg, props, "  " + label)
    S = Synapses(src, tgt, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S.connect(i=[si[b] for b in agg["bodyId_pre"]], j=[ti[b] for b in agg["bodyId_post"]])
    S.w = agg["w_mV"].values * mV
    synapses.append(S)

mon_sensor = SpikeMonitor(Sensor)
mon_gf = SpikeMonitor(GF)
mon_motor = SpikeMonitor(Motor)
trace_gf = StateMonitor(GF, "v", record=True)

Network(Sensor, GF, Motor, *synapses,
        mon_sensor, mon_gf, mon_motor, trace_gf).run(STIM)

print(f"\nSpikes -- Sensor: {mon_sensor.num_spikes}  GF: {mon_gf.num_spikes}  "
      f"TTMn: {mon_motor.num_spikes}")
if mon_motor.num_spikes:
    t0 = float(mon_motor.t[0] / ms)
    print(f">>> ESCAPE em t={t0:.1f}ms (looming em {t0 / float(STIM/ms) * LOOM_MAX_HZ:.1f} Hz por celula)")
else:
    print(f">>> Sem escape. Pico do GF: {float(trace_gf.v.max()/mV):.2f} mV "
          f"(limiar {float(V_THRESHOLD/mV):.0f} mV)")

# ---------- raster ----------
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8),
                         gridspec_kw={"height_ratios": [2, 2, 1, 1]})

loom_idx = {i for i, v in enumerate(is_looming) if v}
inh_idx = {i for i, v in enumerate(is_inhib) if v}
ti_all = mon_sensor.t / ms
ii_all = np.asarray(mon_sensor.i)
for nome, idxs, cor in [("looming (LC4/LPLC2)", loom_idx, "#2a7fff"),
                        ("inibitorios", inh_idx, "#d62728")]:
    m = np.isin(ii_all, list(idxs))
    axes[0].plot(ti_all[m], ii_all[m], ".", ms=1, color=cor, label=nome)
axes[0].set_ylabel("upstream do GF\n(1271 neuronios)")
axes[0].legend(fontsize=7, loc="upper left", markerscale=6)
axes[0].set_title("Giant Fiber -- entrada real do neuronio (1271 upstream), "
                  "sinal pelo neurotransmissor", fontsize=10)

for i, b in enumerate(GF_IDS):
    axes[1].plot(trace_gf.t / ms, trace_gf.v[i] / mV, lw=1, label=props.at[b, "instance"])
axes[1].axhline(float(V_THRESHOLD / mV), color="k", ls="--", lw=0.8, label="limiar")
axes[1].axhline(float(V_REST / mV), color="gray", ls=":", lw=0.8, label="repouso")
axes[1].set_ylabel("GF  v (mV)")
axes[1].legend(fontsize=7, loc="upper left")

axes[2].plot(mon_gf.t / ms, mon_gf.i, "o", color="crimson", ms=4)
axes[2].set_ylabel("GF spikes")
axes[2].set_ylim(-0.5, len(GF_IDS) - 0.5)

axes[3].plot(mon_motor.t / ms, mon_motor.i, "s", color="darkgreen", ms=5)
axes[3].set_ylabel("TTMn")
axes[3].set_ylim(-0.5, max(len(motor_ids) - 0.5, 0.5))
axes[3].set_xlabel(f"tempo (ms) -- looming sobe de 0 a {LOOM_MAX_HZ} Hz por celula LC4/LPLC2")

plt.tight_layout()
out = HERE.parent / "docs" / "images" / "giant_fiber_raster.png"
plt.savefig(out, dpi=120)
print(f"Raster salvo em {out}")
