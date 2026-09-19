"""
Mede o que a inibicao faz no circuito Giant Fiber.

6 dos 8 neuronios pre-sinapticos mais fortes do GF sao GABA ou glutamato e
carregam 59% do peso total. Enquanto o modelo tratava tudo como excitatorio,
esse peso empurrava na direcao errada. Aqui a pergunta e direta: com o sinal
correto, a inibicao so atrapalha a fuga, ou faz alguma coisa util?

Roda o circuito em varios niveis de looming, com e sem as sinapses inibitorias,
e conta disparo do motoneuronio de pulo (TTMn) -- ou seja, quantas vezes o robo
teria pulado.
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

DURATION = 300 * ms
N_TRIALS = 8  # Poisson e estocastico; uma corrida so nao diz nada


def trial(loom_hz, inhib_on, seed):
    np.random.seed(seed)
    defaultclock.dt = 0.1 * ms

    GF = NeuronGroup(len(GF_IDS), LIF_EQS, **LIF_KWARGS)
    GF.v = V_REST
    Motor = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS)
    Motor.v = V_REST

    # LC4/LPLC2 recebem o looming, os inibitorios ficam tonicos, o resto em 0 Hz
    Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
    taxa = np.zeros(len(sensor_ids))
    taxa[np.array(is_looming)] = loom_hz
    taxa[np.array(is_inhib)] = TONIC_INHIB_HZ
    Sensor.rate = taxa * Hz

    six = {b: i for i, b in enumerate(sensor_ids)}
    gix = {b: i for i, b in enumerate(GF_IDS)}
    mix = {b: i for i, b in enumerate(motor_ids)}

    a = signed_weights(up, props)
    a = a[a["bodyId_pre"].isin(six) & a["bodyId_post"].isin(gix)]
    if not inhib_on:
        a = a[a["sign"] > 0]  # remove so as sinapses inibitorias, mantem o resto igual
    S1 = Synapses(Sensor, GF, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S1.connect(i=[six[b] for b in a["bodyId_pre"]], j=[gix[b] for b in a["bodyId_post"]])
    S1.w = a["w_mV"].values * mV

    b = signed_weights(down, props)
    b = b[b["bodyId_pre"].isin(gix) & b["bodyId_post"].isin(mix)]
    S2 = Synapses(GF, Motor, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S2.connect(i=[gix[x] for x in b["bodyId_pre"]], j=[mix[x] for x in b["bodyId_post"]])
    S2.w = b["w_mV"].values * mV

    mon = SpikeMonitor(Motor)
    Network(Sensor, GF, Motor, S1, S2, mon).run(DURATION)
    return mon.num_spikes


# faixa por celula LC4/LPLC2 -- sao 311 delas, entao o que chega no GF e bem maior
LOOM = [0, 1, 2, 3, 5, 7, 10, 15, 20]
res = {True: [], False: []}
for inhib_on in (True, False):
    for loom in LOOM:
        counts = [trial(loom, inhib_on, seed=s) for s in range(N_TRIALS)]
        res[inhib_on].append((np.mean(counts), np.std(counts)))
        print(f"looming {loom:3d} Hz  inibicao {'ON ' if inhib_on else 'OFF'}  "
              f"TTMn {np.mean(counts):5.1f} +/- {np.std(counts):.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
for inhib_on, color, label in [
    (True, "#1f77b4", "circuito real (com as sinapses inibitorias)"),
    (False, "#d62728", "so as excitatorias (o modelo antigo)"),
]:
    m = np.array([x[0] for x in res[inhib_on]])
    sd = np.array([x[1] for x in res[inhib_on]])
    ax.plot(LOOM, m, "o-", color=color, label=label)
    ax.fill_between(LOOM, m - sd, m + sd, color=color, alpha=0.15)

ax.set_xlabel("intensidade do looming (Hz por celula LC4/LPLC2)")
ax.set_ylabel(f"spikes do TTMn em {int(DURATION/ms)} ms\n(= comandos de pulo)")
ax.set_title("A inibicao nao desliga a fuga: ela suprime alarme falso\n"
             f"media de {N_TRIALS} corridas, sombra = desvio padrao", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out = HERE.parent / "docs" / "images" / "inhibition_gate.png"
plt.savefig(out, dpi=120)
print(f"\nSalvo: {out}")
