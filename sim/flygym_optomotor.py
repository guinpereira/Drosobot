"""
Laco sensorio-motor fechado: mosca biomecanica em MuJoCo, circuito real no meio.

    retina do flygym (2 olhos x 721 omatideos)
      -> energia de movimento por olho
      -> T4/T5 -> HS -> DNa02 -> motoneuronio de perna   (peso e sinal do conectoma)
      -> acao de shape (2,) do HybridTurningController
      -> a mosca anda e vira, o mundo muda na retina, fecha o laco

Por que encaixa tao direto: o HybridTurningController do NeuroMechFly recebe o que
a documentacao dele chama de "descending signal encoding turning", de 2 elementos.
DNa02 E um neuronio descendente de giro, e o circuito ja produz o par esquerdo /
direito porque as tres etapas nao cruzam a linha media no conectoma. A interface da
ferramenta e a anatomia coincidem sem adaptador.

O que vem do dado e o que e nosso:

  do conectoma  - quem conecta em quem, com que peso, com que sinal, de que lado
  nosso         - como virar imagem de retina em taxa de disparo de T4/T5
                  (energia de movimento = |dI/dt| por olho)
                - como virar spike de motoneuronio em comando de marcha
                  (a biomecanica da coxa nao esta no conectoma)

Roda:  .venv\\Scripts\\python sim\\flygym_optomotor.py
Sai:   docs/images/flygym_optomotor.mp4 e flygym_optomotor.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    Network, NeuronGroup, Synapses, SpikeMonitor, ms, mV, Hz, defaultclock,
)
from flygym import Fly, Camera
from flygym.arena import BlocksTerrain
from flygym.examples.locomotion import HybridTurningController

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST,
    load_properties, signed_weights, side_of,
)

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "images"
props = load_properties()

# ---------------- circuito optomotor (mesmo de optomotor_network.py) ----------------
sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")

sensor_ids = sorted(sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(sensor_hs["bodyId_post"]) | set(hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(hs_dna02["bodyId_post"]) | set(dna02_motor["bodyId_pre"]))
motor_ids = sorted(dna02_motor["bodyId_post"].unique().tolist())

sensor_side = np.array([side_of(props, b) for b in sensor_ids])
motor_side = np.array([side_of(props, b) for b in motor_ids])
MOTOR_L = np.where(motor_side == "L")[0]
MOTOR_R = np.where(motor_side == "R")[0]

defaultclock.dt = 0.5 * ms

Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
HS = NeuronGroup(len(hs_ids), LIF_EQS, **LIF_KWARGS)
HS.v = V_REST
DNa02 = NeuronGroup(len(dna02_ids), LIF_EQS, **LIF_KWARGS)
DNa02.v = V_REST
Motor = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS)
Motor.v = V_REST

idx = lambda ids: {b: i for i, b in enumerate(ids)}
synapses = []
for src, tgt, conn, si, ti in [
    (Sensor, HS, sensor_hs, idx(sensor_ids), idx(hs_ids)),
    (HS, DNa02, hs_dna02, idx(hs_ids), idx(dna02_ids)),
    (DNa02, Motor, dna02_motor, idx(dna02_ids), idx(motor_ids)),
]:
    agg = signed_weights(conn, props)
    agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
    S = Synapses(src, tgt, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S.connect(i=[si[b] for b in agg["bodyId_pre"]], j=[ti[b] for b in agg["bodyId_post"]])
    S.w = agg["w_mV"].values * mV
    synapses.append(S)

mon_motor = SpikeMonitor(Motor)
mon_dna02 = SpikeMonitor(DNa02)
net = Network(Sensor, HS, DNa02, Motor, *synapses, mon_motor, mon_dna02)

print(f"circuito: T4/T5 {len(sensor_ids)} (L={np.sum(sensor_side=='L')} R={np.sum(sensor_side=='R')})"
      f" -> HS {len(hs_ids)} -> DNa02 {len(dna02_ids)} -> motor {len(motor_ids)}")

# ---------------- corpo em MuJoCo ----------------
# terreno de blocos em vez de chao liso: sem contraste visual nao existe fluxo
# optico, e o circuito nao teria o que detectar
arena = BlocksTerrain(height_range=(0.2, 0.2), block_size=1.3)
# o HybridTurningController usa contato de tibia/tarso pra detectar tropeco,
# entao esses sensores precisam estar ligados explicitamente
CONTATOS = [f"{perna}{seg}"
            for perna in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]]
fly = Fly(enable_vision=True, vision_refresh_rate=500, spawn_pos=(0, 0, 0.3),
          contact_sensor_placements=CONTATOS)
cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_top",
             targeted_fly_names=[fly.name], window_size=(600, 450), fps=30)
sim = HybridTurningController(fly=fly, cameras=[cam], arena=arena, timestep=1e-4)

# ---------------- como a retina vira taxa de T4/T5 ----------------
# SUPOSICAO nossa, nao vem do conectoma: energia de movimento = modulo da variacao
# temporal da intensidade, media sobre os omatideos daquele olho. T4/T5 sao
# seletivos a direcao, mas nossa amostra e quase toda T5a (um subtipo so), entao
# usar magnitude e mais honesto do que fingir seletividade que a amostra nao tem.
# medido nesta arena: |dI/dt| por olho fica em ~0.005 de media e 0.038 de pico.
# O ganho e o unico parametro livre da nossa transducao sensorial -- escolhido pra
# por o circuito na faixa em que ele responde (ver optomotor_network.py: o canal
# esquerdo so alcanca o motoneuronio perto de 200 Hz).
FLOW_GAIN = 30000.0
FLOW_MAX_HZ = 250.0
BASE_DRIVE = 1.0       # marcha reta quando o circuito esta em silencio
TURN_GAIN = 1.2        # quanto o desequilibrio entre os lados encurta a perna de dentro
# Media movel do comando de giro. Numa janela de 2 ms costuma cair 1 spike so, e o
# balanco cru salta entre -1 e +1 a cada quadro -- comando descendente nao se
# comporta assim, ele integra no tempo (e a perna tambem filtra). TAU em numero de
# atualizacoes de retina.
TURN_TAU = 12.0

N_STEPS = 12000        # 1.2 s de mundo fisico
BRIAN_STEP = 2 * ms

obs, info = sim.reset(seed=0)
retina_ant = np.asarray(obs["vision"]).mean(axis=2)   # (2 olhos, 721)
drive = np.array([BASE_DRIVE, BASE_DRIVE])
visto_motor = 0
hz = np.zeros(2)
bal_suave = 0.0

hist = {"t": [], "flow_L": [], "flow_R": [], "hz_L": [], "hz_R": [],
        "motor_L": [], "motor_R": [], "drive_L": [], "drive_R": [],
        "x": [], "y": []}
acc_L = acc_R = 0

for passo in range(N_STEPS):
    obs, _, _, _, info = sim.step(drive)
    sim.render()

    # a retina so atualiza na taxa de vision_refresh_rate, nao a cada passo de
    # fisica. Sem essa checagem, 19 de cada 20 passos dao diferenca exatamente
    # zero, a taxa vai pra 0 Hz e o circuito nunca dispara.
    if info.get("vision_updated", False):
        retina = np.asarray(obs["vision"]).mean(axis=2)
        flow = np.abs(retina - retina_ant).mean(axis=1)   # um valor por olho
        retina_ant = retina
        hz = np.clip(flow * FLOW_GAIN, 0, FLOW_MAX_HZ)
        Sensor.rate = np.where(sensor_side == "L", hz[0], hz[1]) * Hz

        net.run(BRIAN_STEP)
        novos = np.asarray(mon_motor.i[visto_motor:])
        visto_motor = mon_motor.num_spikes
        n_L = int(np.isin(novos, MOTOR_L).sum())
        n_R = int(np.isin(novos, MOTOR_R).sum())
        acc_L += n_L
        acc_R += n_R

        # SUPOSICAO nossa: motoneuronio de um lado encurta o passo daquele lado,
        # e a mosca vira pra la. O conectoma da o LADO, nao a biomecanica.
        total = n_L + n_R
        bal = 0.0 if total == 0 else (n_R - n_L) / total
        bal_suave += (bal - bal_suave) / TURN_TAU
        drive = np.array([BASE_DRIVE, BASE_DRIVE])
        if bal_suave > 0:
            drive[1] -= bal_suave * TURN_GAIN
        else:
            drive[0] -= -bal_suave * TURN_GAIN
        drive = np.clip(drive, -0.5, 1.5)

        hist["t"].append(passo * 1e-4)
        hist["flow_L"].append(float(flow[0])); hist["flow_R"].append(float(flow[1]))
        hist["hz_L"].append(hz[0]); hist["hz_R"].append(hz[1])
        hist["motor_L"].append(n_L); hist["motor_R"].append(n_R)
        hist["drive_L"].append(drive[0]); hist["drive_R"].append(drive[1])
        hist["x"].append(float(obs["fly"][0][0])); hist["y"].append(float(obs["fly"][0][1]))

print(f"passos: {N_STEPS}  |  spikes de motor  L={acc_L}  R={acc_R}")
print(f"posicao final da mosca: {np.round(obs['fly'][0], 2)}")

video = OUT / "flygym_optomotor.mp4"
cam.save_video(video)
print(f"video salvo em {video}")

# ---------------- figura ----------------
t = np.array(hist["t"])
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
for a in axes[:2]:
    a.sharex(axes[2])
ax_traj = fig.add_subplot(gs[:, 1])

axes[0].plot(t, hist["hz_L"], color="#1f77b4", label="olho esquerdo")
axes[0].plot(t, hist["hz_R"], color="#ff7f0e", label="olho direito")
axes[0].set_ylabel("T4/T5 (Hz)\nenergia de movimento")
axes[0].legend(fontsize=8)
axes[0].set_title("Laco fechado: retina do MuJoCo -> circuito real -> marcha", fontsize=11)

axes[1].plot(t, hist["motor_L"], color="#1f77b4", label="motor perna L")
axes[1].plot(t, hist["motor_R"], color="#ff7f0e", label="motor perna R")
axes[1].set_ylabel("spikes por passo")
axes[1].legend(fontsize=8)

axes[2].plot(t, hist["drive_L"], color="#1f77b4", label="amplitude perna L")
axes[2].plot(t, hist["drive_R"], color="#ff7f0e", label="amplitude perna R")
axes[2].axhline(BASE_DRIVE, color="k", ls=":", lw=0.8, label="marcha reta")
axes[2].set_ylabel("acao (2,) do\nHybridTurningController")
axes[2].set_xlabel("tempo (s)")
axes[2].legend(fontsize=8)

for ax in axes:
    ax.grid(alpha=0.3)

ax_traj.plot(hist["x"], hist["y"], color="#2ca02c", lw=1.5)
ax_traj.plot(hist["x"][0], hist["y"][0], "o", color="k", ms=6, label="inicio")
ax_traj.plot(hist["x"][-1], hist["y"][-1], "*", color="crimson", ms=14, label="fim")
ax_traj.set_aspect("equal", adjustable="datalim")
ax_traj.set_xlabel("x (mm)")
ax_traj.set_ylabel("y (mm)")
ax_traj.set_title("trajetoria da mosca", fontsize=10)
ax_traj.legend(fontsize=8)
ax_traj.grid(alpha=0.3)
plt.tight_layout()
png = OUT / "flygym_optomotor.png"
plt.savefig(png, dpi=120)
print(f"figura salva em {png}")
