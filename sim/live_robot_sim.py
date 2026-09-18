"""
Simulador ao vivo (pygame) rodando os DOIS circuitos reais ao mesmo tempo,
continuo (nao mais "roda 300ms e para"). Testa o fluxo completo antes do
hardware chegar: estimulo controlado por teclado -> circuito real -> robo
se move na tela.

Controles:
  SETA CIMA    -- objeto se aproxima (looming) -> pode disparar escape (Giant Fiber)
  SETA ESQ/DIR -- movimento visual naquela direcao -> pode disparar giro (Optomotor)
  ESC / fechar janela -- sai

Simplificacao documentada: o circuito optomotor aqui nao separa os dois
hemisferios (pool unico dos top-20 T4/T5 por peso). O que vem do dado REAL
e SE e QUANDO o circuito dispara; a DIRECAO do giro no desenho usa a tecla
que voce esta segurando no momento do disparo -- nao vem do dado.

Prioridade igual ao firmware real: escape sempre interrompe um giro em andamento.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from brian2 import NeuronGroup, Synapses, SpikeMonitor, run, ms, Hz, start_scope, defaultclock

HERE = Path(__file__).parent
CONNECTOME = HERE.parent / "connectome"

# ---------- carrega os dois circuitos ----------
gf_up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
gf_down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
GF_IDS = [10001, 10010]
gf_up_w = gf_up.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False)
gf_sensor_ids = gf_up_w.head(8).index.tolist()
gf_motor_ids = gf_down[gf_down["type"] == "TTMn"]["bodyId_post"].unique().tolist()

opto_sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
opto_hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
opto_dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")
om_sensor_ids = sorted(opto_sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(opto_sensor_hs["bodyId_post"]) | set(opto_hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(opto_hs_dna02["bodyId_post"]) | set(opto_dna02_motor["bodyId_pre"]))
om_motor_ids = sorted(opto_dna02_motor["bodyId_post"].unique().tolist())

# ---------- rede Brian2 unica (os dois circuitos coexistem) ----------
start_scope()
defaultclock.dt = 0.5 * ms
eqs = "dv/dt = -v / tau : 1\ntau : second"

GF_Sensor = NeuronGroup(len(gf_sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
GF = NeuronGroup(len(GF_IDS), eqs, threshold="v>1", reset="v=0", refractory=50 * ms, method="euler")
GF.tau = 10 * ms
GF_Motor = NeuronGroup(len(gf_motor_ids), eqs, threshold="v>1", reset="v=0", refractory=5 * ms, method="euler")
GF_Motor.tau = 10 * ms

OM_Sensor = NeuronGroup(len(om_sensor_ids), "rate : Hz", threshold="rand()<rate*dt", method="euler")
HS = NeuronGroup(len(hs_ids), eqs, threshold="v>1", reset="v=0", refractory=20 * ms, method="euler")
HS.tau = 10 * ms
DNa02 = NeuronGroup(len(dna02_ids), eqs, threshold="v>1", reset="v=0", refractory=20 * ms, method="euler")
DNa02.tau = 10 * ms
OM_Motor = NeuronGroup(len(om_motor_ids), eqs, threshold="v>1", reset="v=0", refractory=10 * ms, method="euler")
OM_Motor.tau = 10 * ms

gf_s_idx = {b: i for i, b in enumerate(gf_sensor_ids)}
gf_g_idx = {b: i for i, b in enumerate(GF_IDS)}
gf_m_idx = {b: i for i, b in enumerate(gf_motor_ids)}

w_sg = gf_up[gf_up["bodyId_pre"].isin(gf_sensor_ids) & gf_up["bodyId_post"].isin(GF_IDS)]
w_sg = w_sg.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S1 = Synapses(GF_Sensor, GF, "w : 1", on_pre="v_post += w")
S1.connect(i=[gf_s_idx[b] for b in w_sg["bodyId_pre"]], j=[gf_g_idx[b] for b in w_sg["bodyId_post"]])
S1.w = w_sg["weight"].values * 0.001

w_gm = gf_down[gf_down["bodyId_pre"].isin(GF_IDS) & gf_down["bodyId_post"].isin(gf_motor_ids)]
w_gm = w_gm.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S2 = Synapses(GF, GF_Motor, "w : 1", on_pre="v_post += w")
S2.connect(i=[gf_g_idx[b] for b in w_gm["bodyId_pre"]], j=[gf_m_idx[b] for b in w_gm["bodyId_post"]])
S2.w = w_gm["weight"].values * 0.02

om_s_idx = {b: i for i, b in enumerate(om_sensor_ids)}
hs_idx = {b: i for i, b in enumerate(hs_ids)}
dna_idx = {b: i for i, b in enumerate(dna02_ids)}
om_m_idx = {b: i for i, b in enumerate(om_motor_ids)}

w1 = opto_sensor_hs.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S3 = Synapses(OM_Sensor, HS, "w : 1", on_pre="v_post += w")
S3.connect(i=[om_s_idx[b] for b in w1["bodyId_pre"]], j=[hs_idx[b] for b in w1["bodyId_post"]])
S3.w = w1["weight"].values * 0.002

w2 = opto_hs_dna02.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S4 = Synapses(HS, DNa02, "w : 1", on_pre="v_post += w")
S4.connect(i=[hs_idx[b] for b in w2["bodyId_pre"]], j=[dna_idx[b] for b in w2["bodyId_post"]])
S4.w = w2["weight"].values * 0.03

w3 = opto_dna02_motor.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
S5 = Synapses(DNa02, OM_Motor, "w : 1", on_pre="v_post += w")
S5.connect(i=[dna_idx[b] for b in w3["bodyId_pre"]], j=[om_m_idx[b] for b in w3["bodyId_post"]])
S5.w = w3["weight"].values * 0.02

mon_escape = SpikeMonitor(GF_Motor)
mon_turn = SpikeMonitor(OM_Motor)

# ---------- pygame ----------
pygame.init()
W, H = 900, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Drosobot -- simulador ao vivo (dado real Male CNS)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

PLAY_LEFT, PLAY_RIGHT = 20, W - 20
PLAY_TOP, PLAY_BOTTOM = 130, H - 20  # margem de cima maior, fora da faixa do HUD

robot_x, robot_y, heading = W / 2, (PLAY_TOP + PLAY_BOTTOM) / 2, -math.pi / 2  # aponta pra cima
trail = []
distance_cm = 100.0
flash_text, flash_until = "", 0
last_escape_n, last_turn_n = 0, 0
escape_lockout_until_ms = 0.0
sim_time_ms = 0.0

FORWARD_SPEED = 25.0    # px/s
JUMP_DIST = 40.0        # px por escape
TURN_STEP_DEG = 2.0     # graus por spike de giro (6 fazia loop completo rapido demais; sem freio artificial de cooldown -- so a taxa real de disparo do DNa02, que ja tem refratario 20ms, decide a velocidade)

running = True
while running:
    dt_real_s = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    # -- estimulo controlado por teclado --
    if keys[pygame.K_UP]:
        distance_cm = max(2.0, distance_cm - 80 * dt_real_s)
    else:
        distance_cm = min(100.0, distance_cm + 30 * dt_real_s)
    gf_rate_hz = np.interp(distance_cm, [2, 100], [400, 5])
    GF_Sensor.rate = gf_rate_hz * Hz

    turn_key = 0
    if keys[pygame.K_LEFT]:
        turn_key = -1
    elif keys[pygame.K_RIGHT]:
        turn_key = 1
    om_rate_hz = 250.0 if turn_key != 0 else 5.0
    OM_Sensor.rate = om_rate_hz * Hz

    # -- avanca a simulacao real (peso sinaptico real do conectoma) --
    STEP_MS = 5
    run(STEP_MS * ms)
    sim_time_ms += STEP_MS

    # -- le spikes NOVOS desde o ultimo frame --
    n_escape_now = mon_escape.num_spikes
    n_turn_now = mon_turn.num_spikes
    new_escapes = n_escape_now - last_escape_n
    new_turns = n_turn_now - last_turn_n
    last_escape_n, last_turn_n = n_escape_now, n_turn_now

    if new_escapes > 0:
        escape_lockout_until_ms = sim_time_ms + 150  # mesma janela do firmware real
        robot_x -= JUMP_DIST * math.cos(heading)
        robot_y -= JUMP_DIST * math.sin(heading)
        flash_text, flash_until = "ESCAPE!", pygame.time.get_ticks() + 400

    in_escape_lockout = sim_time_ms < escape_lockout_until_ms
    if new_turns > 0 and not in_escape_lockout and turn_key != 0:
        heading += math.radians(TURN_STEP_DEG * turn_key * new_turns)
        flash_text = "TURN " + ("L" if turn_key < 0 else "R")
        flash_until = pygame.time.get_ticks() + 400

    # -- caminhada idle constante --
    robot_x += FORWARD_SPEED * dt_real_s * math.cos(heading)
    robot_y += FORWARD_SPEED * dt_real_s * math.sin(heading)
    if robot_x <= PLAY_LEFT or robot_x >= PLAY_RIGHT:
        heading = math.pi - heading   # quica na parede esquerda/direita
    if robot_y <= PLAY_TOP or robot_y >= PLAY_BOTTOM:
        heading = -heading            # quica no teto/chao
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
        f"distancia (SETA CIMA aproxima): {distance_cm:5.1f} cm   sensor GF: {gf_rate_hz:5.0f} Hz",
        f"movimento visual (SETA ESQ/DIR): {'ESQ' if turn_key<0 else 'DIR' if turn_key>0 else '--'}   sensor OM: {om_rate_hz:5.0f} Hz",
        f"spikes escape total: {n_escape_now}   spikes giro total: {n_turn_now}",
        "ESC pra sair",
    ]
    for i, line in enumerate(hud_lines):
        screen.blit(font.render(line, True, (200, 200, 200)), (10, 10 + i * 22))

    if flash_text and pygame.time.get_ticks() < flash_until:
        big = pygame.font.SysFont("consolas", 40, bold=True)
        screen.blit(big.render(flash_text, True, (255, 220, 60)), (W / 2 - 100, PLAY_TOP + 10))

    pygame.display.flip()

pygame.quit()
