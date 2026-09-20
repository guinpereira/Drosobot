r"""
O que o modelo do NeuroMechFly realmente usa do MuJoCo.

Antes de escrever um solver, e preciso saber o que ele tem que resolver. A
alternativa -- implementar "o MuJoCo" -- nao termina: sao dezenas de tipos de
junta, restricao, atuador e sensor, e o NeuroMechFly usa um punhado deles.

Este modulo abre o `mjModel` ja COMPILADO (nao o XML: o compilador do MuJoCo
expande defaults, resolve `<pair>`, converte inercia) e conta o que aparece.
O resultado e o contrato do backend GPU: tudo que aparecer aqui com contagem
maior que zero tem que ser suportado; o que nao aparecer pode esperar.

    .venv-flygym2\Scripts\python -m sim.gpu_physics.inventario

Roda nas tres arenas dos experimentos, porque elas nao usam o mesmo subconjunto
-- a de obstaculos declara `<pair>` explicitos que as outras nao tem.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]

# Nomes legiveis dos enums, para o relatorio nao ser uma lista de inteiros.
JNT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
GEOM = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
        5: "cylinder", 6: "box", 7: "mesh", 8: "sdf"}
TRN = {0: "joint", 1: "jointinparent", 2: "slidercrank", 3: "tendon",
       4: "site", 5: "body"}
DYN = {0: "none", 1: "integrator", 2: "filter", 3: "filterexact", 4: "muscle",
       5: "user"}
GAIN = {0: "fixed", 1: "affine", 2: "muscle", 3: "user"}
BIAS = {0: "none", 1: "affine", 2: "muscle", 3: "user"}
EQ = {0: "connect", 1: "weld", 2: "joint", 3: "tendon", 4: "flex", 5: "distance"}
INTEGRADOR = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
SOLVER = {0: "PGS", 1: "CG", 2: "Newton"}
CONE = {0: "pyramidal", 1: "elliptic"}
JACOBIANO = {0: "dense", 1: "sparse", 2: "auto"}


def _nomes_sensor() -> dict:
    """mjtSensor nao e iteravel no binding; os membros vem do namespace."""
    return {int(v): k.replace("mjSENS_", "")
            for k, v in vars(mj.mjtSensor).items() if k.startswith("mjSENS_")}


def _conta(valores, nomes) -> dict:
    c = Counter(int(v) for v in valores)
    return {nomes.get(k, f"?{k}"): v for k, v in sorted(c.items())}


def inventario(m: mj.MjModel) -> dict:
    """Tudo que decide o que um solver precisa implementar."""
    return {
        "dimensoes": {
            "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
            "nbody": int(m.nbody), "njnt": int(m.njnt), "ngeom": int(m.ngeom),
            "nsite": int(m.nsite), "ntendon": int(m.ntendon),
            "nmesh": int(m.nmesh), "nhfield": int(m.nhfield),
            "npair": int(m.npair), "nexclude": int(m.nexclude),
            "neq": int(m.neq), "nsensor": int(m.nsensor),
            "nmocap": int(m.nmocap), "nflex": int(m.nflex),
            "nM": int(m.nM), "nD": int(m.nD),
        },
        "opcoes": {
            "timestep": float(m.opt.timestep),
            "integrator": INTEGRADOR.get(int(m.opt.integrator)),
            "solver": SOLVER.get(int(m.opt.solver)),
            "cone": CONE.get(int(m.opt.cone)),
            "jacobian": JACOBIANO.get(int(m.opt.jacobian)),
            "iterations": int(m.opt.iterations),
            "ls_iterations": int(m.opt.ls_iterations),
            "tolerance": float(m.opt.tolerance),
            "impratio": float(m.opt.impratio),
            "gravity": [float(g) for g in m.opt.gravity],
            "wind": [float(g) for g in m.opt.wind],
            "density": float(m.opt.density),
            "viscosity": float(m.opt.viscosity),
            "o_margin": float(m.opt.o_margin),
            "disableflags": int(m.opt.disableflags),
            "enableflags": int(m.opt.enableflags),
        },
        "juntas": _conta(m.jnt_type, JNT),
        "geoms_todos": _conta(m.geom_type, GEOM),
        "geoms_colidiveis": _conta(
            [t for t, c, a in zip(m.geom_type, m.geom_contype, m.geom_conaffinity)
             if c or a], GEOM),
        "atuadores": {
            "transmissao": _conta(m.actuator_trntype, TRN),
            "dyntype": _conta(m.actuator_dyntype, DYN),
            "gaintype": _conta(m.actuator_gaintype, GAIN),
            "biastype": _conta(m.actuator_biastype, BIAS),
            "ctrllimited": int(np.sum(m.actuator_ctrllimited)),
            "forcelimited": int(np.sum(m.actuator_forcelimited)),
            "actlimited": int(np.sum(m.actuator_actlimited)),
            "na": int(m.na),
        },
        "restricoes": {
            "equality": _conta(m.eq_type, EQ) if m.neq else {},
            "jnt_limited": int(np.sum(m.jnt_limited)),
            "tendon_limited": int(np.sum(m.tendon_limited)) if m.ntendon else 0,
        },
        "colisao": {
            "npair_explicitos": int(m.npair),
            "nexclude": int(m.nexclude),
            "geoms_com_mask_nao_nula": int(np.sum(
                (np.asarray(m.geom_contype) != 0)
                | (np.asarray(m.geom_conaffinity) != 0))),
            "geoms_totais": int(m.ngeom),
            "condim_usados": sorted({int(c) for c in m.geom_condim}),
            "pair_condim_usados": (sorted({int(c) for c in m.pair_dim})
                                   if m.npair else []),
            "margin_max": float(np.max(m.geom_margin)) if m.ngeom else 0.0,
            "gap_max": float(np.max(m.geom_gap)) if m.ngeom else 0.0,
        },
        "sensores": _conta(m.sensor_type, _nomes_sensor()),
    }


def _model_da_arena(arena: str, estimulo: dict | None = None) -> mj.MjModel:
    sys.path.insert(0, str(RAIZ / "sim"))
    from physics.flygym2 import FlyGym2Adapter

    ad = FlyGym2Adapter(arena=arena, estimulo=estimulo or {}, com_visao=True)
    ad.reset(seed=0)
    return ad.sim.mj_model


def main() -> int:
    arenas = {
        "looming": {},
        "obstaculos": {"lado": "centro"},
        "optomotor": {},
    }
    saida = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
             "mujoco": mj.__version__, "arenas": {}}
    for nome, est in arenas.items():
        m = _model_da_arena(nome, est)
        saida["arenas"][nome] = inventario(m)
        print(f"[ok] {nome}: nq={m.nq} nv={m.nv} nu={m.nu} "
              f"ngeom={m.ngeom} npair={m.npair}")

    dest = RAIZ / "benchmarks" / "physics" / "mjmodel_inventario.json"
    dest.write_text(json.dumps(saida, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
