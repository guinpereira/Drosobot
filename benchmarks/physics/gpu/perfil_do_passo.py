r"""
Onde vai o tempo de GPU do passo, e quanto disso e execucao de kernel.

    .venv-flygym2\Scripts\python benchmarks\physics\gpu\perfil_do_passo.py

Duas medidas por etapa, e a diferenca entre elas e o achado:

    eventos   `profile.end - start` do OpenCL: o relogio da PLACA, so o tempo
              em que o kernel esteve executando
    parede    o relogio do host em volta de N repeticoes da etapa ISOLADA,
              com uma unica espera no fim

A etapa isolada importa. Medir um kernel DENTRO da sequencia do passo atribui a
ele a espera pela dependencia anterior: com a mesma implementacao, o
`solver_newton_rapido` mede 330 us dentro do passo e 39 us medido sozinho. Foi
seguindo o numero errado que uma rodada inteira de "otimizacao" foi gasta no
lugar errado.

## O que este perfil mostrou

1. `factor_M` e `solve_M` -- a fatoracao esparsa da matriz de massa e as duas
   substituicoes triangulares -- dominavam. Sao serials por natureza: a linha
   `k` depende das anteriores, e dentro de cada linha ha no maximo
   `rownnz <= 17` elementos, ou seja 16 threads ativas de 256. Com tao pouco
   paralelismo o que manda e a latencia de cada acesso, e em memoria global
   eram ~20 mil acessos quase todos dependentes.

   Passar a matriz para `__local` nao muda a ordem das operacoes -- so de onde
   se le -- e corta o custo por tres.

2. O passo continua com uma lacuna grande entre o tempo de parede e a soma dos
   eventos. Ela nao escala com o NUMERO de despachos (6 ou 15 dao a mesma), e
   fica registrada como o maior gargalo nao explicado.
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

ASSENTA = 200
N = 300


def main() -> int:
    from gpu_physics.compilador import compila
    from gpu_physics.device import Device
    from gpu_physics.dinamica import MotorFisicoGPU
    from physics import cria
    from physics.adapter import MotorFrame

    corpo = cria("flygym2-mujoco", arena="looming", timestep=1e-4,
                 com_visao=False)
    corpo.reset(seed=0)
    for _ in range(ASSENTA):
        corpo.passo(MotorFrame(drive=np.ones(2)))
    m, d = corpo.sim.mj_model, corpo.sim.mj_data
    mj.mj_forward(m, d)

    dev = Device(perfil=True)
    g = MotorFisicoGPU(compila(m), dev=dev, fp64=True)
    g.escreve_estado(qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
                     mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
    g.passo_fundido()

    def isolado(fn):
        for _ in range(20):
            fn()
        dev.espera()
        dev.tempos()
        t0 = time.perf_counter_ns()
        for _ in range(N):
            fn()
        dev.espera()
        parede = (time.perf_counter_ns() - t0) / N / 1000.0
        ev = dev.tempos()
        return parede, sum(v["ns"] for v in ev.values()) / N / 1000.0

    etapas = [
        ("cinematica", g.cinematica),
        ("arvore (com_pos+massa+com_vel)", g.grupo_arvore_f),
        ("com_pos", g.com_pos_f),
        ("massa (CRB + factor_M)", g.massa_f),
        ("com_vel", g.com_vel_f),
        ("colisao", g.colisao),
        ("restricoes", g.restricoes_f),
        ("bias (RNE)", g.bias_f),
        ("forcas (passivo+atuacao+adesao)", g.forcas_f),
        ("smooth (solve_M)", g.smooth_f),
        ("solver (Newton)", g.solver_rapido),
        ("euler (factor+solve+integra)", g.euler_f),
    ]
    r = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
         "etapas": {}, "passo": {}}
    print(f"  {'etapa':<34} {'parede':>9} {'eventos':>9}")
    for nome, fn in etapas:
        parede, ev = isolado(fn)
        r["etapas"][nome] = {"parede_us": round(parede, 1),
                             "eventos_us": round(ev, 1)}
        print(f"  {nome:<34} {parede:>8.1f}u {ev:>8.1f}u")

    for nome, fn in (("passo fundido (15 despachos)",
                      lambda: g.passo_fundido(esperar=False)),
                     ("passo em grupos (6 despachos)",
                      lambda: g.passo_grupos(esperar=False))):
        parede, ev = isolado(fn)
        r["passo"][nome] = {"parede_us": round(parede, 1),
                            "eventos_us": round(ev, 1),
                            "lacuna_us": round(parede - ev, 1)}
        print(f"\n  {nome:<34} {parede:>8.1f}u {ev:>8.1f}u  "
              f"lacuna {parede-ev:.1f}u")

    t0 = time.perf_counter_ns()
    for _ in range(N):
        mj.mj_step(m, d)
    cpu = (time.perf_counter_ns() - t0) / N / 1000.0
    r["mujoco_cpu_us"] = round(cpu, 1)
    print(f"  {'MuJoCo CPU mj_step':<34} {cpu:>8.1f}u")

    dest = RAIZ / "benchmarks" / "physics" / "gpu" / "perfil_do_passo.json"
    dest.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    corpo.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
