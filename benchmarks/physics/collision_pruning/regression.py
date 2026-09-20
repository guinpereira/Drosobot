"""
REFERENCE vs TARSI nos quatro comportamentos. Uma regressao, nao uma campanha.

    .venv\\Scripts\\python benchmarks\\physics\\collision_pruning\\regression.py

A poda `tarsi` foi validada so em caminhada reta. Isso e cedo demais pra virar
padrao: curva fechada, fuga de re e campo de obstaculos poem as pernas em poses
que aquela corrida nunca visitou.

Quatro comportamentos, mesma semente, timestep, pose, controlador, duracao e
estimulo nos dois lados. So muda o conjunto de auto-colisao.

    caminhada reta     drive constante
    curva (optomotor)  drive assimetrico sustentado
    looming/escape     drive alterna entre andar e recuar
    campo de obstaculos  postes, com o drive do controle cego

Criterio: se os quatro passarem, `tarsi` vira padrao. Se algum divergir, aquele
experimento fica em REFERENCE e o fato e documentado. **Nao se ajusta a fisica
pra fazer passar.**
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco

RAIZ = Path(__file__).resolve().parents[3]
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "sim"))

DT = 1e-4
DURACAO_S = 1.5
CONTATOS = [f"{p}{s}" for p in ("LF", "LM", "LH", "RF", "RM", "RH")
            for s in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")]

# Tolerancias declaradas. Nao sao chutadas: a caminhada reta deu desvio
# exatamente ZERO, entao qualquer coisa acima de ruido de ponto flutuante ja
# merece investigacao. Os valores abaixo sao generosos de proposito -- se algo
# passar deles, e diferenca de comportamento, nao de arredondamento.
TOL_POS_MM = 0.5          # ~4% de um passo de mosca
TOL_QPOS_RAD = 0.02       # ~1 grau
TOL_ORIENT_DEG = 2.0
TOL_CONTATOS = 0.15       # 15% na media de ncon


# ----------------------------------------------------------- experimentos

def _arena_obstaculos():
    import importlib.util
    import os
    import flygym.examples as fex
    from flygym.arena import FlatTerrain
    spec = importlib.util.spec_from_file_location(
        "flygym_vision_arena", os.path.join(fex.__path__[0], "vision", "arena.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    postes = np.array([(12.0 + 10.0 * k, 7.0 if k % 2 == 0 else -7.0)
                       for k in range(6)])
    return mod.ObstacleOdorArena(
        terrain=FlatTerrain(), obstacle_positions=postes,
        obstacle_colors=(0, 0, 0, 1), obstacle_radius=3.0, obstacle_height=4.0,
        odor_source=np.array([[1000.0, 0.0, 2.0]]), marker_colors=[])


def _arena_blocos():
    from flygym.arena import BlocksTerrain
    return BlocksTerrain(height_range=(0.2, 0.2), block_size=1.3)


def _arena_plana():
    from flygym.arena import FlatTerrain
    return FlatTerrain()


def drive_reto(passo, t_s):
    return np.array([1.0, 1.0])


def drive_curva(passo, t_s):
    """Curva sustentada: e o que poe as pernas em pose assimetrica."""
    return np.array([1.0, 0.3])


def drive_fuga(passo, t_s):
    """Alterna andar e recuar, como o Giant Fiber faz ao disparar."""
    ciclo = t_s % 0.8
    return np.array([-0.5, -0.5]) if ciclo < 0.12 else np.array([1.0, 1.0])


EXPERIMENTOS = [
    ("caminhada_reta", _arena_plana, drive_reto),
    ("curva_optomotor", _arena_blocos, drive_curva),
    ("looming_escape", _arena_plana, drive_fuga),
    ("campo_obstaculos", _arena_obstaculos, drive_reto),
]


def roda(nome_arena, fabrica_arena, fabrica_drive, self_collisions):
    from flygym import Fly
    from flygym.examples.locomotion import HybridTurningController

    fly = Fly(enable_vision=False, spawn_pos=(0, 0, 0.3),
              contact_sensor_placements=CONTATOS,
              self_collisions=self_collisions)
    sim = HybridTurningController(fly=fly, arena=fabrica_arena(), timestep=DT)
    sim.reset(seed=0)
    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    torax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "0/Thorax")

    n = int(DURACAO_S / DT)
    traj, quat, qpos, ncon = [], [], [], []
    t0 = time.perf_counter()
    for passo in range(n):
        sim.step(fabrica_drive(passo, passo * DT))
        if passo % 100 == 0:
            traj.append(d.xpos[torax].copy())
            quat.append(d.xquat[torax].copy())
            qpos.append(d.qpos.copy())
            ncon.append(int(d.ncon))
    wall = time.perf_counter() - t0
    return {
        "npair": int(m.npair),
        "traj": np.array(traj), "quat": np.array(quat),
        "qpos": np.array(qpos), "ncon": np.array(ncon),
        "us_por_passo": wall / n * 1e6,
        "avanco_mm": float(np.array(traj)[-1][0] - np.array(traj)[0][0]),
    }


def compara(ref, alt):
    dpos = np.linalg.norm(ref["traj"] - alt["traj"], axis=1)
    dq = float(np.abs(ref["qpos"] - alt["qpos"]).max())
    dot = np.abs(np.sum(ref["quat"] * alt["quat"], axis=1)).clip(-1, 1)
    dang = float(np.degrees(2 * np.arccos(dot)).max())
    nref, nalt = float(ref["ncon"].mean()), float(alt["ncon"].mean())
    dcon = abs(nalt - nref) / max(1e-9, nref)
    return {
        "desvio_pos_max_mm": float(dpos.max()),
        "desvio_pos_final_mm": float(dpos[-1]),
        "desvio_qpos_max_rad": dq,
        "desvio_orient_max_deg": dang,
        "ncon_ref": round(nref, 3), "ncon_alt": round(nalt, 3),
        "ncon_desvio_rel": round(dcon, 4),
        "avanco_ref_mm": round(ref["avanco_mm"], 3),
        "avanco_alt_mm": round(alt["avanco_mm"], 3),
        "passa": bool(dpos.max() < TOL_POS_MM and dq < TOL_QPOS_RAD
                      and dang < TOL_ORIENT_DEG and dcon < TOL_CONTATOS),
    }


def main():
    print(f"regressao REFERENCE vs TARSI -- {DURACAO_S}s, semente 0, dt {DT}")
    print(f"tolerancias: pos {TOL_POS_MM} mm, qpos {TOL_QPOS_RAD} rad, "
          f"orient {TOL_ORIENT_DEG} deg, contatos {TOL_CONTATOS*100:.0f}%")
    print()

    resultados = {}
    for nome, arena, drive in EXPERIMENTOS:
        print(f"== {nome} ==")
        ref = roda(nome, arena, drive, "legs")
        alt = roda(nome, arena, drive, "tarsi")
        c = compara(ref, alt)
        ganho = ref["us_por_passo"] / alt["us_por_passo"]
        veredito = "PASSA" if c["passa"] else "DIVERGE"
        print(f"  pares {ref['npair']} -> {alt['npair']}   "
              f"passo {ref['us_por_passo']:.0f} -> {alt['us_por_passo']:.0f} us "
              f"({ganho:.2f}x)")
        print(f"  pos max {c['desvio_pos_max_mm']:.4f} mm   "
              f"qpos {c['desvio_qpos_max_rad']:.6f} rad   "
              f"orient {c['desvio_orient_max_deg']:.3f} deg   "
              f"ncon {c['ncon_ref']:.2f}->{c['ncon_alt']:.2f}")
        print(f"  avanco {c['avanco_ref_mm']:.2f} -> {c['avanco_alt_mm']:.2f} mm"
              f"   >>> {veredito}")
        resultados[nome] = {**c, "npair_ref": ref["npair"],
                            "npair_alt": alt["npair"],
                            "us_ref": round(ref["us_por_passo"], 1),
                            "us_alt": round(alt["us_por_passo"], 1),
                            "ganho": round(ganho, 3)}
        print()

    todos = all(r["passa"] for r in resultados.values())
    print("=" * 62)
    print(f"  VEREDITO: {'tarsi pode virar padrao' if todos else 'tarsi NAO generaliza'}")
    if not todos:
        for n, r in resultados.items():
            if not r["passa"]:
                print(f"    {n} divergiu -> manter REFERENCE nesse experimento")

    destino = AQUI / f"regression_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    destino.write_text(json.dumps({
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "duracao_s": DURACAO_S, "timestep": DT, "semente": 0,
        "tolerancias": {"pos_mm": TOL_POS_MM, "qpos_rad": TOL_QPOS_RAD,
                        "orient_deg": TOL_ORIENT_DEG, "ncon_rel": TOL_CONTATOS},
        "todos_passaram": todos,
        "experimentos": resultados,
    }, indent=2), encoding="utf-8")
    print(f"  salvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
