"""
Reduz os pares de colisao e prova se a trajetoria muda.

    .venv\\Scripts\\python benchmarks\\physics\\collision_pruning\\validate_pruning.py

Todas as configuracoes rodam com a MESMA semente, pose inicial, drive, arena,
timestep e duracao. Nada de solver, atrito, geometria ou timestep e tocado -- a
unica variavel e QUAIS geoms entram na auto-colisao.

## Como se desliga um par (e como NAO se desliga)

O FlyGym monta os pares em `Fly.init_self_contacts()`, cruzando todos os geoms
da lista `self_collisions` (fly.py:715 no v1.2.1). Entao o botao suportado e
essa lista, e e ele que usamos.

Uma tentativa anterior mexeu em `m.pair_margin[k]` pra "desligar" pares depois do
modelo compilado. **Nao funcionou**: `ncon` ficou identico nas tres configuracoes
e o passo ficou 10x mais lento. `margin` filtra o resultado do narrowphase, nao
impede o narrowphase de rodar -- e o dm_control ainda pode resincronizar o modelo
por cima. Ficou registrado aqui pra ninguem tentar de novo.

## Configuracoes

    REFERENCE   self_collisions="legs"   o que o Drosobot roda hoje
    TARSI       self_collisions="tarsi"  so os tarsos se auto-colidem
    NONE        self_collisions="none"   nenhuma auto-colisao

NONE nao e uma proposta: e o LIMITE SUPERIOR do ganho possivel por esta via. Se
NONE nao for muito mais rapido que REFERENCE, nao ha o que buscar aqui e o
esforco vai pra outro lugar.
"""
from __future__ import annotations

import argparse
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

DURACAO_S = 2.0
CONTATOS = [f"{p}{s}" for p in ("LF", "LM", "LH", "RF", "RM", "RH")
            for s in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")]
CONFIGS = [("REFERENCE", "legs"), ("TARSI", "tarsi"), ("NONE", "none")]


def monta(self_collisions: str):
    from flygym import Fly
    from flygym.examples.locomotion import HybridTurningController
    fly = Fly(enable_vision=False, spawn_pos=(0, 0, 0.3),
              contact_sensor_placements=CONTATOS,
              self_collisions=self_collisions)
    sim = HybridTurningController(fly=fly, timestep=1e-4)
    sim.reset(seed=0)
    return sim


def perfil_mj(sim, n_passos: int = 600) -> dict:
    """
    Perfil do mj_step PURO, sem o wrapper do flygym.

    Separado de proposito: `sim.step()` faz mais do que um `mj_step`, entao os
    timers internos do MuJoCo lidos por cima dele nao fecham em 100% -- foi
    assim que uma medicao anterior reportou COL_NARROW em 189% do passo, que e
    impossivel e deveria ter sido um sinal de parada na hora.
    """
    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    for _ in range(200):
        mujoco.mj_step(m, d)
    mujoco.set_mjcb_time(time.perf_counter)
    for i in range(mujoco.mjtTimer.mjNTIMER):
        d.timer[i].duration = 0.0
        d.timer[i].number = 0
    for _ in range(n_passos):
        mujoco.mj_step(m, d)
    mujoco.set_mjcb_time(None)
    total = d.timer[int(mujoco.mjtTimer.mjTIMER_STEP)].duration
    narrow = d.timer[int(mujoco.mjtTimer.mjTIMER_COL_NARROW)].duration
    return {
        "npair": int(m.npair),
        "mj_step_ms": total / n_passos * 1000,
        "col_narrow_ms": narrow / n_passos * 1000,
        "col_narrow_frac": (narrow / total) if total else 0.0,
    }


def roda(sim, duracao_s: float) -> dict:
    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    n_passos = int(duracao_s / 1e-4)
    drive = np.array([1.0, 1.0])
    torax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "0/Thorax")

    traj, quats, qpos_am, ncon = [], [], [], []
    t0 = time.perf_counter()
    for passo in range(n_passos):
        sim.step(drive)
        if passo % 100 == 0:
            traj.append(d.xpos[torax].copy())
            quats.append(d.xquat[torax].copy())
            qpos_am.append(d.qpos.copy())
            ncon.append(int(d.ncon))
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "rtf": duracao_s / wall,
        "us_por_passo": wall / n_passos * 1e6,
        "traj": np.array(traj), "quat": np.array(quats),
        "qpos": np.array(qpos_am), "ncon_medio": float(np.mean(ncon)),
    }


def compara(ref: dict, alt: dict) -> dict:
    dt = np.linalg.norm(ref["traj"] - alt["traj"], axis=1)
    dq = np.abs(ref["qpos"] - alt["qpos"]).max()
    dot = np.abs(np.sum(ref["quat"] * alt["quat"], axis=1)).clip(-1, 1)
    dang = np.degrees(2 * np.arccos(dot))
    return {
        "desvio_pos_final_mm": float(dt[-1]),
        "desvio_pos_max_mm": float(dt.max()),
        "desvio_qpos_max_rad": float(dq),
        "desvio_orient_max_deg": float(dang.max()),
        "avanco_ref_mm": float(ref["traj"][-1][0] - ref["traj"][0][0]),
        "avanco_alt_mm": float(alt["traj"][-1][0] - alt["traj"][0][0]),
        "ncon_ref": ref["ncon_medio"], "ncon_alt": alt["ncon_medio"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duracao", type=float, default=DURACAO_S)
    args = ap.parse_args()

    resultados, perfis = {}, {}
    for nome, sc in CONFIGS:
        print(f"== {nome}  (self_collisions={sc!r}) ==")
        sim = monta(sc)
        perfis[nome] = perfil_mj(sim)
        p = perfis[nome]
        print(f"  {p['npair']:5d} pares   mj_step {p['mj_step_ms']:7.3f} ms   "
              f"COL_NARROW {p['col_narrow_ms']:7.3f} ms ({p['col_narrow_frac']*100:5.1f}%)")
        sim = monta(sc)     # remonta: o perfil acima avancou o estado
        resultados[nome] = roda(sim, args.duracao)
        r = resultados[nome]
        print(f"  passo completo {r['us_por_passo']:7.1f} us   RTF {r['rtf']:.4f}   "
              f"ncon medio {r['ncon_medio']:.2f}")

    ref = resultados["REFERENCE"]
    pref = perfis["REFERENCE"]
    print("\n== contra REFERENCE ==")
    rel = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "duracao_s": args.duracao, "timestep": 1e-4,
        "configs": {"REFERENCE": {**pref, "us_por_passo": ref["us_por_passo"],
                                  "rtf": ref["rtf"], "ncon_medio": ref["ncon_medio"]}},
    }
    for nome, _ in CONFIGS[1:]:
        c = compara(ref, resultados[nome])
        ganho = pref["mj_step_ms"] / perfis[nome]["mj_step_ms"]
        ganho_total = ref["us_por_passo"] / resultados[nome]["us_por_passo"]
        print(f"  {nome}: pares {pref['npair']} -> {perfis[nome]['npair']}")
        print(f"    mj_step        {pref['mj_step_ms']:.3f} -> "
              f"{perfis[nome]['mj_step_ms']:.3f} ms   ({ganho:.2f}x)")
        print(f"    passo completo {ref['us_por_passo']:.1f} -> "
              f"{resultados[nome]['us_por_passo']:.1f} us  ({ganho_total:.2f}x)")
        print(f"    avanco         {c['avanco_ref_mm']:.2f} -> {c['avanco_alt_mm']:.2f} mm")
        print(f"    desvio pos max {c['desvio_pos_max_mm']:.4f} mm   "
              f"qpos max {c['desvio_qpos_max_rad']:.6f} rad   "
              f"orient max {c['desvio_orient_max_deg']:.3f} deg")
        rel["configs"][nome] = {**perfis[nome], **c,
                                "us_por_passo": resultados[nome]["us_por_passo"],
                                "rtf": resultados[nome]["rtf"],
                                "ganho_mj_step": ganho,
                                "ganho_passo_completo": ganho_total}

    destino = AQUI / f"pruning_result_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    destino.write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
