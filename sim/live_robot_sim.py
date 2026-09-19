"""
Simulador ao vivo (pygame) com os DOIS circuitos reais rodando junto e continuo.
Estimulo pelo teclado -> circuito real -> robo se move na tela.

Controles:
  SETA CIMA    -- objeto se aproxima (looming) -> pode disparar escape (Giant Fiber)
  SETA ESQ/DIR -- fluxo optico naquele olho -> pode disparar giro (Optomotor)
  ESC / fechar janela -- sai

O que mudou em relacao a v1:

- a DIRECAO do giro agora sai do circuito. Antes o pool optomotor era unico e o
  desenho girava pro lado da tecla que voce estava segurando. Agora a tecla so
  escolhe em QUAL OLHO entra o fluxo optico; quem decide o lado do giro e o
  motoneuronio de perna que disparou, e os tres estagios (T4/T5 -> HS -> DNa02
  -> motor de perna) nao cruzam a linha media no conectoma.
- sinal da sinapse vem do neurotransmissor real e a biofisica e a de
  Shiu et al. 2024 (ver sim/connectome_model.py), no lugar do LIF unitless com
  ganho ajustado a mao por camada.

Continua sendo suposicao nossa (o conectoma nao diz): que disparo do motor de um
lado gira o robo PARA aquele lado. O circuito entrega o LADO; a biomecanica de
como uma coxa girando vira o corpo esta fora do dado.

Prioridade igual ao firmware real: escape sempre interrompe um giro em andamento.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from brian2 import (
    Network, NeuronGroup, Synapses, SpikeMonitor, ms, mV, Hz, defaultclock,
)

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, LIF_EQS, LIF_KWARGS, T_DELAY, V_REST,
    load_properties, signed_weights, side_of, nt_sign,
)

props = load_properties()

# ---------- carrega os dois circuitos ----------
gf_up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
gf_down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
GF_IDS = [10001, 10010]
gf_sensor_ids = gf_up.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False).head(8).index.tolist()
gf_motor_ids = gf_down[gf_down["type"] == "TTMn"]["bodyId_post"].unique().tolist()

om_sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
om_hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
om_dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")
om_sensor_ids = sorted(om_sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(om_sensor_hs["bodyId_post"]) | set(om_hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(om_hs_dna02["bodyId_post"]) | set(om_dna02_motor["bodyId_pre"]))
om_motor_ids = sorted(om_dna02_motor["bodyId_post"].unique().tolist())

# quem e de que lado -- e isso que da a direcao do giro
om_sensor_side = [side_of(props, b) for b in om_sensor_ids]
om_motor_side = [side_of(props, b) for b in om_motor_ids]
MOTOR_L = np.array([i for i, s in enumerate(om_motor_side) if s == "L"])
MOTOR_R = np.array([i for i, s in enumerate(om_motor_side) if s == "R"])

# os upstream do GF que nao sao do caminho de looming ficam em taxa tonica
gf_sensor_is_looming = np.array([nt_sign(props, b) > 0 for b in gf_sensor_ids])
TONIC_INHIB_HZ = 20.0

# ---------- rede Brian2 unica (os dois circuitos coexistem) ----------
defaultclock.dt = 0.5 * ms

GF_Sensor = NeuronGroup(len(gf_sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
GF = NeuronGroup(len(GF_IDS), LIF_EQS, **LIF_KWARGS)
GF.v = V_REST
GF_Motor = NeuronGroup(len(gf_motor_ids), LIF_EQS, **LIF_KWARGS)
GF_Motor.v = V_REST

OM_Sensor = NeuronGroup(len(om_sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
HS = NeuronGroup(len(hs_ids), LIF_EQS, **LIF_KWARGS)
HS.v = V_REST
DNa02 = NeuronGroup(len(dna02_ids), LIF_EQS, **LIF_KWARGS)
DNa02.v = V_REST
OM_Motor = NeuronGroup(len(om_motor_ids), LIF_EQS, **LIF_KWARGS)
OM_Motor.v = V_REST


def index_of(ids):
    return {b: i for i, b in enumerate(ids)}


synapses = []
for src, tgt, conn, si, ti in [
    (GF_Sensor, GF, gf_up, index_of(gf_sensor_ids), index_of(GF_IDS)),
    (GF, GF_Motor, gf_down, index_of(GF_IDS), index_of(gf_motor_ids)),
    (OM_Sensor, HS, om_sensor_hs, index_of(om_sensor_ids), index_of(hs_ids)),
    (HS, DNa02, om_hs_dna02, index_of(hs_ids), index_of(dna02_ids)),
    (DNa02, OM_Motor, om_dna02_motor, index_of(dna02_ids), index_of(om_motor_ids)),
]:
    agg = signed_weights(conn, props)
    agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
    S = Synapses(src, tgt, "w : volt", on_pre="g_post += w", delay=T_DELAY)
    S.connect(i=[si[b] for b in agg["bodyId_pre"]], j=[ti[b] for b in agg["bodyId_post"]])
    S.w = agg["w_mV"].values * mV
    synapses.append(S)

mon_escape = SpikeMonitor(GF_Motor)
mon_turn = SpikeMonitor(OM_Motor)
# Network explicito: o run() magico varre as locais de quem chama e nao olha
# dentro de listas, entao as Synapses em `synapses` ficariam de fora.
net = Network(GF_Sensor, GF, GF_Motor, OM_Sensor, HS, DNa02, OM_Motor,
              *synapses, mon_escape, mon_turn)

# ---------- pygame ----------
pygame.init()
W, H = 900, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Drosobot -- simulador ao vivo (dado real Male CNS)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

PLAY_LEFT, PLAY_RIGHT = 20, W - 20
PLAY_TOP, PLAY_BOTTOM = 130, H - 20

robot_x, robot_y, heading = W / 2, (PLAY_TOP + PLAY_BOTTOM) / 2, -math.pi / 2
trail = []
distance_cm = 100.0
flash_text, flash_until = "", 0
seen_escape, seen_turn = 0, 0
turns_L, turns_R = 0, 0
escape_lockout_until_ms = 0.0
sim_time_ms = 0.0

FORWARD_SPEED = 25.0
JUMP_DIST = 40.0
TURN_STEP_DEG = 2.0
# mesmas faixas calibradas dos scripts de rede (ver comentarios la)
GF_LOOM_MAX_HZ, GF_LOOM_MIN_HZ = 120.0, 5.0
OM_FORTE_HZ, OM_FRACO_HZ = 200.0, 15.0

running = True
while running:
    dt_real_s = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    # -- looming so no caminho excitatorio; inibitorios em taxa tonica --
    if keys[pygame.K_UP]:
        distance_cm = max(2.0, distance_cm - 80 * dt_real_s)
    else:
        distance_cm = min(100.0, distance_cm + 30 * dt_real_s)
    gf_rate_hz = float(np.interp(distance_cm, [2, 100], [GF_LOOM_MAX_HZ, GF_LOOM_MIN_HZ]))
    GF_Sensor.rate = np.where(gf_sensor_is_looming, gf_rate_hz, TONIC_INHIB_HZ) * Hz

    # -- a tecla escolhe o OLHO, nao a direcao do giro --
    olho = "L" if keys[pygame.K_LEFT] else ("R" if keys[pygame.K_RIGHT] else None)
    if olho:
        OM_Sensor.rate = [OM_FORTE_HZ if s == olho else OM_FRACO_HZ
                          for s in om_sensor_side] * Hz
    else:
        OM_Sensor.rate = OM_FRACO_HZ * Hz
        # contadores do HUD sao POR EPISODIO, nao acumulados da sessao: soltando
        # a tecla eles zeram. Acumulado engana -- depois de um giro pra direita,
        # um giro pra esquerda aparecia como "L 48 R 101", com o lado errado
        # maior so por causa do episodio anterior.
        turns_L, turns_R = 0, 0

    STEP_MS = 5
    net.run(STEP_MS * ms)
    sim_time_ms += STEP_MS

    # -- spikes novos desde o ultimo frame --
    new_escapes = mon_escape.num_spikes - seen_escape
    seen_escape = mon_escape.num_spikes

    novos_turn = np.asarray(mon_turn.i[seen_turn:])
    seen_turn = mon_turn.num_spikes
    n_turn_L = int(np.isin(novos_turn, MOTOR_L).sum())
    n_turn_R = int(np.isin(novos_turn, MOTOR_R).sum())
    turns_L += n_turn_L
    turns_R += n_turn_R

    if new_escapes > 0:
        escape_lockout_until_ms = sim_time_ms + 150
        robot_x -= JUMP_DIST * math.cos(heading)
        robot_y -= JUMP_DIST * math.sin(heading)
        flash_text, flash_until = "ESCAPE!", pygame.time.get_ticks() + 400

    in_escape_lockout = sim_time_ms < escape_lockout_until_ms
    # direcao pelo SALDO entre os dois motoneuronios, nao pela tecla
    saldo = n_turn_R - n_turn_L
    if saldo != 0 and not in_escape_lockout:
        heading += math.radians(TURN_STEP_DEG * saldo)
        flash_text = "TURN " + ("R" if saldo > 0 else "L")
        flash_until = pygame.time.get_ticks() + 400

    robot_x += FORWARD_SPEED * dt_real_s * math.cos(heading)
    robot_y += FORWARD_SPEED * dt_real_s * math.sin(heading)
    if robot_x <= PLAY_LEFT or robot_x >= PLAY_RIGHT:
        heading = math.pi - heading
    if robot_y <= PLAY_TOP or robot_y >= PLAY_BOTTOM:
        heading = -heading
    robot_x = max(PLAY_LEFT, min(PLAY_RIGHT, robot_x))
    robot_y = max(PLAY_TOP, min(PLAY_BOTTOM, robot_y))
    trail.append((robot_x, robot_y))
    if len(trail) > 300:
        trail.pop(0)

    # ---------- desenho ----------
    screen.fill((15, 15, 20))
    if len(trail) > 1:
        pygame.draw.lines(screen, (60, 90, 60), False, trail, 2)

    tip = (robot_x + 18 * math.cos(heading), robot_y + 18 * math.sin(heading))
    left = (robot_x + 12 * math.cos(heading + 2.5), robot_y + 12 * math.sin(heading + 2.5))
    right = (robot_x + 12 * math.cos(heading - 2.5), robot_y + 12 * math.sin(heading - 2.5))
    color = (220, 60, 60) if in_escape_lockout else (60, 200, 120)
    pygame.draw.polygon(screen, color, [tip, left, right])

    hud_lines = [
        f"distancia (SETA CIMA aproxima): {distance_cm:5.1f} cm   looming: {gf_rate_hz:5.0f} Hz",
        f"fluxo optico entrando no olho (SETA ESQ/DIR): {olho or '--'}",
        f"motoneuronio de perna (episodio)  L: {turns_L:4d}   R: {turns_R:4d}   <- a direcao sai daqui",
        f"spikes de escape (TTMn): {seen_escape}      ESC pra sair",
    ]
    for i, line in enumerate(hud_lines):
        screen.blit(font.render(line, True, (200, 200, 200)), (10, 10 + i * 22))

    if flash_text and pygame.time.get_ticks() < flash_until:
        big = pygame.font.SysFont("consolas", 40, bold=True)
        screen.blit(big.render(flash_text, True, (255, 220, 60)), (W / 2 - 100, PLAY_TOP + 10))

    pygame.display.flip()

pygame.quit()
