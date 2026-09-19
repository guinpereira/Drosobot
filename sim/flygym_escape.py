"""
Giant Fiber no laco 3D: objeto se aproxima, a mosca biomecanica foge.

    esfera aproximando -> retina do flygym -> expansao por olho
      -> LC4/LPLC2 -> DNp01 (Giant Fiber) -> TTMn        (peso e sinal do conectoma)
      -> comando de fuga na marcha em MuJoCo

Complementa sim/flygym_optomotor.py, que fez o mesmo pro circuito de giro. Aqui o
circuito de fuga sai do grafico e vai pra um corpo com fisica.

A esfera se aproxima de frente em ciclos: some, reaparece longe, vem de novo. Entre
um ciclo e outro a mosca anda reto, entao da pra ver que a fuga acontece SO quando
o objeto chega perto -- que e o comportamento de limiar que o circuito deveria ter.

Fronteira dado / suposicao:

  do conectoma  - quem conecta em quem, com que peso, com que sinal, de que lado.
                  LC4 e LPLC2 sao a via de looming segundo o proprio dataset
                  (ver connectome_model.py), e vem separados por hemisferio:
                  L 165 celulas, R 146.
  nosso         - como virar retina em taxa de LC4/LPLC2 (expansao = escurecimento
                  do campo visual)
                - o que "fugir" significa num modelo de CAMINHADA. O NeuroMechFly
                  nao pula, e o TTMn real move o musculo de pulo. Mapeamos spike
                  de TTMn em recuo rapido, que e o mais proximo disponivel.

Roda:  .venv\\Scripts\\python sim\\flygym_escape.py
Sai:   docs/images/flygym_escape.mp4 e flygym_escape.png
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    Network, NeuronGroup, Synapses, SpikeMonitor, ms, mV, Hz, defaultclock,
)
import flygym.examples as flygym_examples
from flygym import Fly, Camera
from flygym.examples.locomotion import HybridTurningController

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST, TONIC_INHIB_HZ,
    load_properties, signed_weights, side_of, gf_input_population,
)

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "images"
props = load_properties()

# o __init__ de flygym.examples.vision importa torch, que nao precisamos --
# carrega o modulo de arena apontando direto pro arquivo
_spec = importlib.util.spec_from_file_location(
    "flygym_vision_arena", os.path.join(flygym_examples.__path__[0], "vision", "arena.py"))
_arena_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arena_mod)


class LoomingArena(_arena_mod.MovingObjArena):
    """Mesma arena da bola flutuante, mas quem move a bola somos nos (ver loop)."""

    def step(self, dt, physics):
        pass  # posicao vem do laco principal, que sabe onde a mosca esta


# ---------------- circuito Giant Fiber ----------------
up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
GF_IDS = [10001, 10010]
motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()
sensor_ids, is_looming, is_inhib = gf_input_population(props, up)

sensor_side = np.array([side_of(props, b) for b in sensor_ids])
is_looming = np.array(is_looming)
is_inhib = np.array(is_inhib)
LOOM_L = is_looming & (sensor_side == "L")
LOOM_R = is_looming & (sensor_side == "R")
print(f"upstream do GF: {len(sensor_ids)}  |  looming L={LOOM_L.sum()} R={LOOM_R.sum()}"
      f"  |  inibitorios={is_inhib.sum()}")

defaultclock.dt = 0.5 * ms

Sensor = NeuronGroup(len(sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
GF = NeuronGroup(len(GF_IDS), LIF_EQS, **LIF_KWARGS)
GF.v = V_REST
TTMn = NeuronGroup(len(motor_ids), LIF_EQS, **LIF_KWARGS)
TTMn.v = V_REST

idx = lambda ids: {b: i for i, b in enumerate(ids)}
synapses = []
for src, tgt, conn, si, ti in [
    (Sensor, GF, up, idx(sensor_ids), idx(GF_IDS)),
    (GF, TTMn, down, idx(GF_IDS), idx(motor_ids)),
]:
    agg = signed_weights(conn, props)
    agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
    S = Synapses(src, tgt, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S.connect(i=[si[b] for b in agg["bodyId_pre"]], j=[ti[b] for b in agg["bodyId_post"]])
    S.w = agg["w_mV"].values * mV
    synapses.append(S)

mon_gf = SpikeMonitor(GF)
mon_ttmn = SpikeMonitor(TTMn)
net = Network(Sensor, GF, TTMn, *synapses, mon_gf, mon_ttmn)

# ---------------- corpo ----------------
CONTATOS = [f"{perna}{seg}"
            for perna in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]]
arena = LoomingArena(obj_radius=3, init_ball_pos=(30, 0))
fly = Fly(enable_vision=True, vision_refresh_rate=500, spawn_pos=(0, 0, 0.3),
          contact_sensor_placements=CONTATOS)
cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_top",
             targeted_fly_names=[fly.name], window_size=(600, 450), fps=30)
sim = HybridTurningController(fly=fly, cameras=[cam], arena=arena, timestep=1e-4)

# ---------------- parametros ----------------
# SUPOSICAO nossa: expansao = crescimento da FRACAO DE OMATIDEOS ESCUROS, e a
# taxa dessa expansao vira taxa de disparo de LC4/LPLC2. So a parte positiva conta
# (objeto crescendo); se afastando nao e looming.
#
# A primeira versao usava diferenca de intensidade media e nao funcionou: o fluxo
# optico da propria caminhada sobre o chao xadrez dominava o sinal e o objeto sumia
# no meio (taxa colada no teto o tempo todo, sem correlacao com a distancia). A
# fracao escura e robusta a isso -- andar sobre chao plano nao muda quanto do campo
# visual esta escuro, objeto se aproximando muda.
#
# Medido com a mosca parada: fracao escura vai de 0.506 a 30 mm ate 0.731 a 4 mm.
#
# Nao da pra usar a diferenca entre quadros consecutivos: a mosca balanca o corpo
# ao andar, a fracao escura treme, e derivada de sinal tremido e ruido (a taxa
# ficava pulando 0-20 Hz sem seguir a aproximacao). Comparamos com uma media LENTA
# em vez do quadro anterior -- robusto ao tremor, e e o que detector de looming
# real faz de qualquer jeito: adapta ao fundo e responde ao que destoa dele.
DARK_THRESHOLD = 0.4
DARK_TAU = 150.0        # atualizacoes de retina (2 ms cada) = ~0.3 s de adaptacao
LOOM_GAIN = 240.0       # poe o pico da aproximacao perto de 20 Hz
LOOM_MAX_HZ = 20.0      # mesma faixa por celula calibrada em giant_fiber_network.py

BASE_DRIVE = 1.0
ESCAPE_DRIVE = -0.5     # recuo: o mais proximo de "pulo pra tras" num modelo que anda
ESCAPE_MS = 120.0       # quanto tempo o recuo dura

CICLO_S = 0.8           # de quanto em quanto tempo o objeto volta a se aproximar
DIST_LONGE = 30.0       # mm na frente da mosca quando o ciclo comeca
DIST_PERTO = 4.0        # mm quando termina -- abaixo disso a esfera passa dos olhos
                        # e a fracao escura desaba, virando artefato e nao looming

N_STEPS = 24000        # 2.4 s de mundo fisico = 3 ciclos
BRIAN_STEP = 2 * ms

obs, info = sim.reset(seed=0)
_r0 = np.asarray(obs["vision"]).mean(axis=2)
escuro_lento = (_r0 < DARK_THRESHOLD).mean(axis=1)
drive = np.array([BASE_DRIVE, BASE_DRIVE])
visto_ttmn = 0
escape_ate = -1.0

hist = {"t": [], "dist": [], "hz_L": [], "hz_R": [], "gf": [], "ttmn": [],
        "drive": [], "x": [], "y": []}
n_escapes = 0
gf_ant = 0

for passo in range(N_STEPS):
    t_s = passo * 1e-4

    # -- objeto se aproxima de frente, em ciclos --
    fase = (t_s % CICLO_S) / CICLO_S
    dist = DIST_LONGE + (DIST_PERTO - DIST_LONGE) * fase
    pos_mosca = np.asarray(obs["fly"][0])
    ang = float(obs["fly_orientation"][0]) if np.ndim(obs["fly_orientation"]) else 0.0
    alvo = np.array([pos_mosca[0] + dist, pos_mosca[1], 2.5], dtype="float32")
    arena.ball_pos = alvo
    sim.physics.bind(arena.object_body).mocap_pos = alvo

    obs, _, _, _, info = sim.step(drive)
    sim.render()

    if info.get("vision_updated", False):
        retina = np.asarray(obs["vision"]).mean(axis=2)
        escuro = (retina < DARK_THRESHOLD).mean(axis=1)      # um valor por olho
        expansao = np.clip(escuro - escuro_lento, 0, None)   # acima do fundo adaptado
        escuro_lento += (escuro - escuro_lento) / DARK_TAU
        hz = np.clip(expansao * LOOM_GAIN, 0, LOOM_MAX_HZ)

        taxa = np.zeros(len(sensor_ids))
        taxa[LOOM_L] = hz[0]
        taxa[LOOM_R] = hz[1]
        taxa[is_inhib] = TONIC_INHIB_HZ
        Sensor.rate = taxa * Hz

        net.run(BRIAN_STEP)

        novos_ttmn = mon_ttmn.num_spikes - visto_ttmn
        visto_ttmn = mon_ttmn.num_spikes
        if novos_ttmn > 0 and t_s > escape_ate:
            escape_ate = t_s + ESCAPE_MS / 1000.0
            n_escapes += 1

        em_fuga = t_s < escape_ate
        drive = np.array([ESCAPE_DRIVE, ESCAPE_DRIVE]) if em_fuga \
            else np.array([BASE_DRIVE, BASE_DRIVE])

        hist["t"].append(t_s)
        hist["dist"].append(dist)
        hist["hz_L"].append(float(hz[0])); hist["hz_R"].append(float(hz[1]))
        hist["gf"].append(mon_gf.num_spikes - gf_ant)
        gf_ant = mon_gf.num_spikes
        hist["ttmn"].append(novos_ttmn)
        hist["drive"].append(float(drive[0]))
        hist["x"].append(float(pos_mosca[0])); hist["y"].append(float(pos_mosca[1]))

print(f"passos {N_STEPS} ({N_STEPS*1e-4:.1f} s)  |  GF {mon_gf.num_spikes} spikes  "
      f"|  TTMn {mon_ttmn.num_spikes} spikes  |  episodios de fuga: {n_escapes}")

video = OUT / "flygym_escape.mp4"
cam.save_video(video)
print(f"video salvo em {video}")

# ---------------- figura ----------------
t = np.array(hist["t"])
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8),
                         gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1]})

axes[0].plot(t, hist["dist"], color="k")
axes[0].invert_yaxis()
axes[0].set_ylabel("distancia do\nobjeto (mm)")
axes[0].set_title("Giant Fiber no corpo biomecanico: objeto aproxima, a mosca recua", fontsize=11)

axes[1].plot(t, hist["hz_L"], color="#1f77b4", label="LC4/LPLC2 esquerdo")
axes[1].plot(t, hist["hz_R"], color="#ff7f0e", label="LC4/LPLC2 direito")
axes[1].set_ylabel("taxa por celula\n(Hz)")
axes[1].legend(fontsize=8)

axes[2].plot(t, hist["gf"], color="crimson", lw=0.8, label="GF (DNp01)")
axes[2].plot(t, hist["ttmn"], color="darkgreen", lw=1.2, label="TTMn")
axes[2].set_ylabel("spikes por\njanela de 2 ms")
axes[2].legend(fontsize=8)

axes[3].plot(t, hist["drive"], color="#9467bd")
axes[3].axhline(BASE_DRIVE, color="k", ls=":", lw=0.8)
axes[3].axhline(ESCAPE_DRIVE, color="crimson", ls=":", lw=0.8)
axes[3].set_ylabel("comando de\nmarcha")
axes[3].set_xlabel("tempo (s)")

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
png = OUT / "flygym_escape.png"
plt.savefig(png, dpi=120)
print(f"figura salva em {png}")
