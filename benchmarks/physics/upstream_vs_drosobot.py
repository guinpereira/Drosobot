r"""
Por que o upstream anuncia numero grande e o nosso vertical anda a 0,2x.

    .venv-flygym2\Scripts\python benchmarks\physics\upstream_vs_drosobot.py

A suspeita razoavel era que o MuJoCo estivesse lento aqui. Este arquivo mostra
que o problema nao e esse: **o upstream mede outra coisa.**

## O que o benchmark oficial do FlyGym 2.x mede

`research/upstream/flygym/scripts/dev/run_gpu_benchmark.py`, lido no clone
local:

    n_worlds       16 -> 16384, dobrando
    backend        flygym.warp.GPUSimulation, que exige CUDA
    rendering      desligado
    controle       angulos gravados, reproduzidos -- sem malha fechada
    geometria      tambem com ALL_TO_CAPSULES, malhas viram capsulas

e a metrica de "tempo real" dele e

    realtime_factor = sim_steps * n_worlds / walltime * sim_timestep

O `n_worlds` esta DENTRO da conta. Um fator de 100x com 16.384 mundos e
0,006x por mundo. Nao ha nada de errado nisso -- para treinar politica, o que
importa e vazao agregada. Mas nao e a mesma grandeza que a nossa, e comparar os
dois numeros de frente compara vazao com latencia.

Duas consequencias praticas:

  1. **O benchmark oficial nao roda nesta maquina.** `check_gpu()` e
     `nvidia-smi` sao chamados direto, e o `flygym.warp` gera CUDA em tempo de
     execucao. Nao ha caminho HIP, Vulkan ou OpenCL no Warp upstream. Isto e
     verificado aqui, nao suposto: o import e tentado e o erro e registrado.

  2. **O que da para comparar honestamente e um mundo so.** E o que este script
     mede, no mesmo PC, com o mesmo modelo, em tres niveis:

         fisica pura           `sim.step()`, cronometrado POR DENTRO do laco
                               real -- ver `_CronometraStep`
         passo do experimento  + controlador de marcha + atuadores
         vertical completo     + retina + transducao + CNS + telemetria

A diferenca entre o primeiro e o segundo e o custo de Python em volta da
fisica. A diferenca entre o segundo e o terceiro e o cerebro. Sem essa
separacao, "a mosca anda devagar" nao aponta para lugar nenhum.

O terceiro nivel nao e remedido aqui: ele ja esta gravado em toda corrida da
plataforma experimental (`runs/*/summary.json`, campo `custo.rtf`), medido com
a receita inteira. Repetir a medicao com outro laco seria um segundo laco a
divergir do primeiro.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

PASSOS = 3000
AQUECE = 300


def upstream_roda_aqui() -> dict:
    """O benchmark oficial roda nesta placa? Tentar e registrar o erro."""
    r = {"caminho": "flygym.warp.GPUSimulation (NVIDIA Warp -> CUDA)"}
    try:
        import warp as wp

        r["warp_versao"] = getattr(wp, "__version__", "?")
        r["warp_devices"] = [str(d) for d in wp.get_devices()]
        r["warp_cuda"] = bool(wp.is_cuda_available())
    except Exception as e:                                    # noqa: BLE001
        r["warp_import"] = f"{type(e).__name__}: {e}"
    try:
        from flygym.warp import check_gpu

        check_gpu()
        r["check_gpu"] = "ok"
    except Exception as e:                                    # noqa: BLE001
        r["check_gpu"] = f"{type(e).__name__}: {e}"
    r["conclusao"] = (
        "o benchmark oficial exige CUDA; nao ha backend HIP/Vulkan/OpenCL no "
        "Warp upstream, e a RX 6700 XT nao o executa. A comparacao possivel no "
        "mesmo PC e de um mundo so, que e o que este script mede."
        if r.get("check_gpu", "").startswith(("Runtime", "Import", "Module",
                                              "Exception", "Assertion", "Value"))
        or not r.get("warp_cuda", False)
        else "o caminho CUDA existe nesta maquina; da para rodar o oficial")
    return r


class _CronometraStep:
    """
    Mede `sim.step()` DENTRO do laco real, sem tirar o controlador do caminho.

    A alternativa obvia -- rodar `mj_step` num laco apertado depois de
    aquecer -- mede outra fisica: sem o controlador reescrevendo `ctrl` a cada
    passo, os atuadores congelam, a mosca desaba no chao, `ncon` sobe e o
    solver passa a trabalhar mais. Na pratica isso inflou a medida de 94 us
    para 149 us sem que nada no modelo tivesse mudado.
    """

    def __init__(self, sim):
        self._sim = sim
        self._orig = sim.step
        self.ns = 0
        self.n = 0
        sim.step = self._step

    def _step(self, *a, **kw):
        t0 = time.perf_counter_ns()
        r = self._orig(*a, **kw)
        self.ns += time.perf_counter_ns() - t0
        self.n += 1
        return r

    def solta(self) -> float:
        """Devolve us por `mj_step` e restaura o metodo original."""
        self._sim.step = self._orig
        return (self.ns / self.n / 1000.0) if self.n else 0.0


def _corpo(arena: str, visao: bool):
    from physics import cria

    c = cria("flygym2-mujoco", arena=arena, timestep=1e-4, com_visao=visao)
    c.reset(seed=0)
    return c


def um_mundo(arena: str = "looming") -> dict:
    """Latencia de um mundo, em tres niveis, no mesmo modelo."""
    from physics.adapter import MotorFrame

    corpo = _corpo(arena, visao=False)
    motor = MotorFrame(drive=np.ones(2))
    m = corpo.sim.mj_model
    for _ in range(AQUECE):
        corpo.passo(motor)

    # Os dois numeros saem da MESMA passada: o de dentro do `sim.step()` e o de
    # fora. Medir em passadas separadas compararia dois estados diferentes da
    # mosca, e o contato nao e o mesmo nos dois.
    crono = _CronometraStep(corpo.sim)
    t0 = time.perf_counter_ns()
    for _ in range(PASSOS):
        corpo.passo(motor)
    us_passo = (time.perf_counter_ns() - t0) / PASSOS / 1000.0
    us_fisica = crono.solta()
    nv, npair, ncon = int(m.nv), int(m.npair), int(corpo.sim.mj_data.ncon)
    corpo.fecha()

    return {
        "arena": arena, "nv": nv, "npair": npair, "ncon_final": ncon,
        "dt_s": 1e-4,
        "fisica_pura_us_por_passo": round(us_fisica, 2),
        "passo_do_experimento_us": round(us_passo, 2),
        "overhead_python_us": round(us_passo - us_fisica, 2),
        "rtf_fisica_pura": round(1e-4 / (us_fisica * 1e-6), 4),
        "rtf_passo_do_experimento": round(1e-4 / (us_passo * 1e-6), 4),
    }


def vertical_gravado() -> dict:
    """
    O RTF do vertical completo, lido das corridas ja gravadas.

    Nao remede: a plataforma experimental ja grava `custo.rtf` com a receita
    inteira, e um segundo laco aqui divergiria do primeiro.
    """
    corridas = sorted((RAIZ / "runs").glob("*/summary.json"))
    saida = []
    for p in corridas:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            continue
        saida.append({
            "corrida": p.parent.name,
            "escopo": s.get("receita", {}).get("escopo"),
            "arena": s.get("receita", {}).get("arena"),
            "rtf": s.get("custo", {}).get("rtf"),
            "ms_por_seg_simulado": s.get("custo", {}).get("ms_por_seg_simulado"),
        })
    return {"n": len(saida), "corridas": saida}


def main() -> int:
    r = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "mujoco": mj.__version__,
        "benchmark_upstream": upstream_roda_aqui(),
        "um_mundo": {a: um_mundo(a) for a in ("looming", "obstaculos")},
        "vertical_completo_gravado": vertical_gravado(),
    }
    print("=== benchmark oficial do FlyGym 2.x nesta maquina")
    for k, v in r["benchmark_upstream"].items():
        print(f"  {k}: {v}")
    print("\n=== um mundo, mesmo PC, mesmo modelo")
    for arena, v in r["um_mundo"].items():
        print(f"  {arena:<12} fisica pura {v['fisica_pura_us_por_passo']:>7.2f} us "
              f"(rtf {v['rtf_fisica_pura']:.3f})   passo do experimento "
              f"{v['passo_do_experimento_us']:>7.2f} us "
              f"(rtf {v['rtf_passo_do_experimento']:.3f})")
        print(f"  {'':<12} overhead de Python em volta da fisica: "
              f"{v['overhead_python_us']:.2f} us/passo")
    v = r["vertical_completo_gravado"]
    print(f"\n=== vertical completo, {v['n']} corridas gravadas")
    for c in v["corridas"][:6]:
        print(f"  {c['corrida'][:46]:<46} rtf {c['rtf']}")

    dest = RAIZ / "benchmarks" / "physics" / "upstream_vs_drosobot.json"
    dest.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
