"""
Janela ao vivo: ve a mosca biomecanica reagindo, com o circuito real no meio.

Os scripts flygym_optomotor.py e flygym_escape.py rodam headless e cospem video.
Este abre o viewer do MuJoCo e mostra a simulacao acontecendo, com o estado do
circuito no terminal.

    .venv\\Scripts\\python sim\\flygym_live.py            # Giant Fiber (objeto aproximando)
    .venv\\Scripts\\python sim\\flygym_live.py optomotor  # giro por fluxo optico

Roda mais devagar que tempo real de proposito: o passo de fisica e 1e-4 s e a
rede spiking anda junto, entao 1 segundo de mosca leva varios segundos de relogio.
Pra assistir isso ate ajuda.

Controles do viewer sao do proprio MuJoCo: arrastar gira a camera, scroll da zoom,
espaco pausa. Fechar a janela encerra.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import flygym.examples as flygym_examples
from flygym import Fly
from flygym.arena import BlocksTerrain
from flygym.examples.locomotion import HybridTurningController

sys.path.insert(0, str(Path(__file__).parent))
from connectome_model import (  # noqa: E402
    CONNECTOME, TONIC_INHIB_HZ, load_properties, signed_weights, side_of,
    gf_input_population,
)
import fast_lif  # noqa: E402

MODO = "optomotor" if len(sys.argv) > 1 and sys.argv[1].startswith("opto") else "escape"
props = load_properties()

# Aqui a rede roda no integrador proprio (sim/fast_lif.py), nao no Brian2. Motivo:
# o Brian2 cobra ~76 ms de overhead FIXO por chamada de run(), e o laco ao vivo
# precisa alternar entre fisica e rede centenas de vezes por segundo de mosca --
# 7,7 s de relogio por segundo de mosca so de overhead. O integrador resolve as
# mesmas equacoes em forma fechada e sai 70x mais barato (0,1 s/s).
#
# O Brian2 continua sendo a referencia: todos os scripts de figura usam ele, e
# `python sim/fast_lif.py` compara os dois pra garantir que nao divergiram.
DT_MS = 0.5

CONTATOS = [f"{perna}{seg}"
            for perna in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]]


def monta(tabelas):
    """(conexoes, ix_pre, ix_pos) -> lista de (i_pre, j_pos, w_mV) pro fast_lif."""
    out = []
    for conn, si, ti in tabelas:
        agg = signed_weights(conn, props)
        agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
        out.append(([si[b] for b in agg["bodyId_pre"]],
                    [ti[b] for b in agg["bodyId_post"]],
                    agg["w_mV"].values))
    return out


ix = lambda ids: {b: i for i, b in enumerate(ids)}

if MODO == "escape":
    # ---- arena com esfera que se aproxima ----
    _spec = importlib.util.spec_from_file_location(
        "flygym_vision_arena",
        os.path.join(flygym_examples.__path__[0], "vision", "arena.py"))
    _am = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_am)

    class LoomingArena(_am.MovingObjArena):
        def step(self, dt, physics):
            pass

    arena = LoomingArena(obj_radius=3, init_ball_pos=(30, 0))

    up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
    down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
    GF_IDS = [10001, 10010]
    motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()
    sensor_ids, is_loom, is_inhib = gf_input_population(props, up)
    lado = np.array([side_of(props, b) for b in sensor_ids])
    is_loom = np.array(is_loom); is_inhib = np.array(is_inhib)
    LOOM_L = is_loom & (lado == "L")
    LOOM_R = is_loom & (lado == "R")

    conexoes = monta([(up, ix(sensor_ids), ix(GF_IDS)),
                      (down, ix(GF_IDS), ix(motor_ids))])
    rede = fast_lif.Rede(len(sensor_ids), [len(GF_IDS), len(motor_ids)], conexoes, DT_MS)
    CAMADA_SAIDA = 1
    print(f"Giant Fiber ao vivo -- {len(sensor_ids)} upstream, "
          f"looming L={LOOM_L.sum()} R={LOOM_R.sum()}")
else:
    arena = BlocksTerrain(height_range=(0.2, 0.2), block_size=1.3)
    sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
    hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
    dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")
    sensor_ids = sorted(sensor_hs["bodyId_pre"].unique().tolist())
    hs_ids = sorted(set(sensor_hs["bodyId_post"]) | set(hs_dna02["bodyId_pre"]))
    dna_ids = sorted(set(hs_dna02["bodyId_post"]) | set(dna02_motor["bodyId_pre"]))
    motor_ids = sorted(dna02_motor["bodyId_post"].unique().tolist())
    lado = np.array([side_of(props, b) for b in sensor_ids])
    lado_motor = np.array([side_of(props, b) for b in motor_ids])
    MOTOR_L = np.where(lado_motor == "L")[0]
    MOTOR_R = np.where(lado_motor == "R")[0]

    conexoes = monta([(sensor_hs, ix(sensor_ids), ix(hs_ids)),
                      (hs_dna02, ix(hs_ids), ix(dna_ids)),
                      (dna02_motor, ix(dna_ids), ix(motor_ids))])
    rede = fast_lif.Rede(len(sensor_ids), [len(hs_ids), len(dna_ids), len(motor_ids)],
                         conexoes, DT_MS)
    CAMADA_SAIDA = 2
    print(f"Optomotor ao vivo -- T4/T5 L={np.sum(lado=='L')} R={np.sum(lado=='R')}")

# retina a 100 Hz em vez de 500: renderizar os dois olhos e o que mais custa no
# passo de fisica, e 100 Hz ja da 80 amostras por ciclo de aproximacao.
VISION_HZ = 100
fly = Fly(enable_vision=True, vision_refresh_rate=VISION_HZ, spawn_pos=(0, 0, 0.3),
          contact_sensor_placements=CONTATOS)
sim = HybridTurningController(fly=fly, arena=arena, timestep=1e-4)
obs, info = sim.reset(seed=0)

# mesmos parametros dos scripts headless (ver comentarios la)
DARK_THRESHOLD, LOOM_GAIN, LOOM_MAX_HZ = 0.4, 240.0, 20.0
# a adaptacao do fundo e 0.3 s de tempo REAL. Nos scripts headless isso virou 150
# atualizacoes porque la a retina roda a 500 Hz; aqui roda a 100, entao o numero
# tem que ser recalculado, senao a adaptacao fica 5x mais lenta sem ninguem notar.
DARK_TAU = 0.3 * VISION_HZ
FLOW_GAIN, FLOW_MAX_HZ = 30000.0, 250.0
BASE_DRIVE, ESCAPE_DRIVE, ESCAPE_MS = 1.0, -0.5, 120.0
TURN_GAIN, TURN_TAU = 1.2, 12.0
CICLO_S, DIST_LONGE, DIST_PERTO = 0.8, 30.0, 4.0
# janela em que a rede avanca entre duas leituras da retina
JANELA_MS = 10.0

_r0 = np.asarray(obs["vision"]).mean(axis=2)
escuro_lento = (_r0 < DARK_THRESHOLD).mean(axis=1)
retina_ant = _r0
drive = np.array([BASE_DRIVE, BASE_DRIVE])
total_saida = 0
escape_ate = -1.0
bal_suave = 0.0
passo = 0

print()
print("Abrindo o viewer do MuJoCo.")
print("  camera segue a mosca sozinha; arrastar gira, scroll da zoom, espaco pausa")
print("  a ESFERA acima da mosca e o estado do circuito:")
if MODO == "escape":
    print("     cinza = quieto   |   amarelo = LC4/LPLC2 disparando   |   VERMELHO = fuga")
else:
    print("     cinza = quieto   |   azul = virando p/ esquerda   |   laranja = p/ direita")
print("  roda ~29x mais devagar que tempo real: o movimento e lento de proposito,")
print("  nao e travamento. O terminal atualiza a cada segundo.")
print()

_modelo = sim.physics.model.ptr
_torax = mujoco.mj_name2id(_modelo, mujoco.mjtObj.mjOBJ_BODY, "0/Thorax")

with mujoco.viewer.launch_passive(_modelo, sim.physics.data.ptr) as viewer:
    # camera colada na mosca. Sem isso ela vira um ponto no meio de uma arena
    # enorme e a cena parece parada, porque a 29x mais devagar o deslocamento
    # por quadro e minusculo.
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = _torax
    viewer.cam.distance = 12.0
    viewer.cam.elevation = -35.0
    viewer.cam.azimuth = 135.0
    viewer.user_scn.ngeom = 1        # a esfera de estado, atualizada a cada quadro

    ultimo_log = time.time()
    while viewer.is_running():
        t_s = passo * 1e-4

        if MODO == "escape":
            fase = (t_s % CICLO_S) / CICLO_S
            dist = DIST_LONGE + (DIST_PERTO - DIST_LONGE) * fase
            pos = np.asarray(obs["fly"][0])
            alvo = np.array([pos[0] + dist, pos[1], 2.5], dtype="float32")
            arena.ball_pos = alvo
            sim.physics.bind(arena.object_body).mocap_pos = alvo

        obs, _, _, _, info = sim.step(drive)
        passo += 1

        if info.get("vision_updated", False):
            retina = np.asarray(obs["vision"]).mean(axis=2)

            if MODO == "escape":
                escuro = (retina < DARK_THRESHOLD).mean(axis=1)
                expansao = np.clip(escuro - escuro_lento, 0, None)
                escuro_lento += (escuro - escuro_lento) / DARK_TAU
                hz = np.clip(expansao * LOOM_GAIN, 0, LOOM_MAX_HZ)
                taxa = np.zeros(len(sensor_ids))
                taxa[LOOM_L] = hz[0]
                taxa[LOOM_R] = hz[1]
                taxa[is_inhib] = TONIC_INHIB_HZ
                taxas = taxa
            else:
                flow = np.abs(retina - retina_ant).mean(axis=1)
                retina_ant = retina
                hz = np.clip(flow * FLOW_GAIN, 0, FLOW_MAX_HZ)
                taxas = np.where(lado == "L", hz[0], hz[1])

            contagem = rede.roda(JANELA_MS, taxas)
            saida = contagem[CAMADA_SAIDA]      # spikes por neuronio nesta janela
            total_saida += int(saida.sum())

            if MODO == "escape":
                if saida.sum() and t_s > escape_ate:
                    escape_ate = t_s + ESCAPE_MS / 1000.0
                drive = (np.array([ESCAPE_DRIVE, ESCAPE_DRIVE]) if t_s < escape_ate
                         else np.array([BASE_DRIVE, BASE_DRIVE]))
            else:
                n_L = int(saida[MOTOR_L].sum())
                n_R = int(saida[MOTOR_R].sum())
                total = n_L + n_R
                bal = 0.0 if total == 0 else (n_R - n_L) / total
                bal_suave += (bal - bal_suave) / TURN_TAU
                drive = np.array([BASE_DRIVE, BASE_DRIVE])
                if bal_suave > 0:
                    drive[1] -= bal_suave * TURN_GAIN
                else:
                    drive[0] -= -bal_suave * TURN_GAIN
                drive = np.clip(drive, -0.5, 1.5)

            # esfera de estado acima da mosca: sem ela a janela nao diz se o
            # circuito esta fazendo alguma coisa, e a simulacao parece morta
            if MODO == "escape":
                if t_s < escape_ate:
                    cor = (1.0, 0.1, 0.1, 1.0)          # vermelho: fugindo
                elif max(hz) > 1.0:
                    cor = (1.0, 0.85, 0.1, 1.0)         # amarelo: LC4/LPLC2 ativo
                else:
                    cor = (0.45, 0.45, 0.45, 1.0)       # cinza: quieto
            else:
                if bal_suave < -0.05:
                    cor = (0.2, 0.45, 1.0, 1.0)         # azul: virando p/ esquerda
                elif bal_suave > 0.05:
                    cor = (1.0, 0.55, 0.1, 1.0)         # laranja: p/ direita
                else:
                    cor = (0.45, 0.45, 0.45, 1.0)
            centro = np.asarray(obs["fly"][0], dtype=float)
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.6, 0.0, 0.0]),
                centro + np.array([0.0, 0.0, 3.5]),
                np.eye(3).ravel(),
                np.array(cor, dtype=np.float32),
            )

            viewer.sync()

            agora = time.time()
            if agora - ultimo_log > 1.0:
                ultimo_log = agora
                if MODO == "escape":
                    estado = "FUGA" if t_s < escape_ate else "andando"
                    print(f"t={t_s:5.2f}s  objeto {dist:5.1f}mm  "
                          f"LC4/LPLC2 L={hz[0]:5.1f} R={hz[1]:5.1f} Hz  "
                          f"TTMn total={total_saida:4d}  {estado}")
                else:
                    print(f"t={t_s:5.2f}s  T4/T5 L={hz[0]:5.0f} R={hz[1]:5.0f} Hz  "
                          f"motor total={total_saida:4d}  "
                          f"marcha L={drive[0]:5.2f} R={drive[1]:5.2f}")

print("encerrado.")
