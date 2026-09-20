r"""
Cinematica direta: Drosobot GPU contra MuJoCo CPU, mesmo modelo, mesma pose.

    .venv-flygym2\Scripts\python benchmarks\physics\gpu\fk_gpu_vs_cpu.py

Mede duas coisas que nao podem ser separadas:

  **Corretude.** O maior erro absoluto entre os campos da GPU e os do `mjData`
  depois de `mj_kinematics`, campo a campo. Em fp64 tem que ficar na ordem do
  epsilon da maquina; se subir, e o porte que esta errado, nao a precisao.

  **Latencia de um mundo.** Microssegundos por cinematica completa. O custo do
  dispatch e separado do custo da conta rodando o mesmo kernel com 1 e com 21
  repeticoes internas e dividindo a diferenca -- sem isso mediriamos os ~100 us
  de ida e volta a cada chamada, que nao e o que o laco real pagaria.

O varrimento de tamanho de work-group existe para responder uma pergunta
especifica: **o kernel esta limitado por paralelismo?** Se o tempo cair quando
o grupo cresce, sim, e a saida e mais threads. Se ficar plano de 16 a 256
threads, nao -- e ai o que limita e a cadeia serial de dependencias da arvore,
que nenhum numero de threads encurta.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "sim"))

CAMPOS = ("xpos", "xquat", "xmat", "xipos", "ximat",
          "xanchor", "xaxis", "geom_xpos", "geom_xmat")
GRUPOS = (16, 32, 64, 128, 256)
PASSOS_ASSENTA = 200


def _cena(arena: str, estimulo: dict):
    from physics.adapter import MotorFrame
    from physics.flygym2 import FlyGym2Adapter

    ad = FlyGym2Adapter(arena=arena, estimulo=estimulo, com_visao=False)
    ad.reset(seed=0)
    # A pose importa: com a mosca no ar, metade dos atalhos de quaternio
    # identidade do MuJoCo dispara e a comparacao mede menos do que parece.
    for _ in range(PASSOS_ASSENTA):
        ad.passo(MotorFrame(drive=np.ones(2)))
    return ad


def _marginal(fn, reps_a: int, reps_b: int, n: int = 200) -> float:
    """us por execucao interna, com o custo do dispatch subtraido."""
    def cron(reps):
        fn(reps)
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn(reps)
        return (time.perf_counter_ns() - t0) / n / 1000.0
    a, b = cron(reps_a), cron(reps_b)
    return (b - a) / (reps_b - reps_a)


def mede(arena: str, estimulo: dict) -> dict:
    from sim.gpu_physics.cinematica import CinematicaGPU
    from sim.gpu_physics.compilador import compila
    from sim.gpu_physics.device import Device

    ad = _cena(arena, estimulo)
    m, d = ad.sim.mj_model, ad.sim.mj_data
    mod = compila(m)
    mj.mj_kinematics(m, d)
    ref = {k: np.asarray(getattr(d, k)).copy() for k in CAMPOS}

    dev = Device()
    r = {
        "arena": arena,
        "hash_modelo": mod.hash_modelo,
        "dims": {k: mod.dims[k] for k in ("nq", "nv", "nbody", "njnt", "ngeom")},
        "device": dev.capacidades(),
        "precisao": {},
    }

    for fp64 in (True, False):
        nome = "fp64" if fp64 else "fp32"
        gpu = CinematicaGPU(mod, dev=dev, fp64=fp64, grupo=256)
        gpu.escreve_estado(d.qpos, d.mocap_pos, d.mocap_quat)
        gpu.passo()
        erros = {}
        for k, v in ref.items():
            got = gpu.le(k).reshape(v.shape).astype(np.float64)
            erros[k] = float(np.abs(got - v).max())
        por_grupo = {}
        for W in GRUPOS:
            g = CinematicaGPU(mod, dev=dev, fp64=fp64, grupo=W)
            g.escreve_estado(d.qpos, d.mocap_pos, d.mocap_quat)
            por_grupo[W] = round(_marginal(lambda reps: g.passo(reps), 1, 21), 3)
        r["precisao"][nome] = {
            "erro_max_por_campo": {k: float(f"{v:.3e}") for k, v in erros.items()},
            "erro_max": float(f"{max(erros.values()):.3e}"),
            "us_por_fk_por_work_group": por_grupo,
            "us_por_fk_melhor": min(por_grupo.values()),
            "arvore_profundidade": gpu.profundidade,
        }

    # referencia: mj_kinematics sozinho, sem o resto do passo
    n = 5000
    mj.mj_kinematics(m, d)
    t0 = time.perf_counter_ns()
    for _ in range(n):
        mj.mj_kinematics(m, d)
    r["mujoco_cpu_us_por_fk"] = round((time.perf_counter_ns() - t0) / n / 1000.0, 3)

    melhor = min(r["precisao"][p]["us_por_fk_melhor"] for p in r["precisao"])
    r["razao_gpu_sobre_cpu_melhor_caso"] = round(
        melhor / r["mujoco_cpu_us_por_fk"], 2)
    espalhamento = [r["precisao"]["fp32"]["us_por_fk_por_work_group"][W]
                    for W in GRUPOS]
    r["limitado_por_paralelismo"] = bool(
        (max(espalhamento) - min(espalhamento)) / max(espalhamento) > 0.5)
    r["leitura"] = (
        "tempo plano de 16 a 256 threads: o kernel NAO esta limitado por "
        "paralelismo. O que limita e a cadeia serial da arvore -- 10 niveis de "
        "operacoes escalares dependentes -- e nenhum numero de threads a "
        "encurta."
        if not r["limitado_por_paralelismo"] else
        "tempo cai com o tamanho do grupo: ha paralelismo a explorar.")
    return r


def main() -> int:
    saida = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
             "mujoco": mj.__version__, "arenas": {}}
    for arena, est in (("looming", {}), ("obstaculos", {"lado": "centro"})):
        r = mede(arena, est)
        saida["arenas"][arena] = r
        print(f"\n=== {arena} (nv={r['dims']['nv']}, nbody={r['dims']['nbody']})")
        print(f"  MuJoCo CPU            {r['mujoco_cpu_us_por_fk']:>8.2f} us")
        for p, v in r["precisao"].items():
            print(f"  Drosobot GPU {p}      {v['us_por_fk_melhor']:>8.2f} us   "
                  f"erro max {v['erro_max']:.2e}")
            print(f"      por work-group: {v['us_por_fk_por_work_group']}")
        print(f"  -> {r['leitura']}")

    dest = RAIZ / "benchmarks" / "physics" / "gpu" / "fk_gpu_vs_cpu.json"
    dest.write_text(json.dumps(saida, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
