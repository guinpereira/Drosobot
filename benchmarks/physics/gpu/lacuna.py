r"""
De onde vem a lacuna entre o relogio de parede do passo e os eventos OpenCL.

    .venv-flygym2\Scripts\python benchmarks\physics\gpu\lacuna.py

Por tres rodadas este foi o maior numero sem explicacao do backend: o passo
media ~1200 us de parede e a soma dos `profile.end - start` dava ~600. Este
arquivo roda as quatro medidas que fecharam a questao, na ordem em que elas
descartam alternativas.

1.  **Decomposicao T0->T3** do passo pipelinado: enfileirar, `flush`, esperar.
    Mostra que `flush` custa 0,0 us -- nao ha sincronizacao escondida no host.

2.  **`set_args` por passo.** O cache de argumentos de `MotorFisicoGPU._roda`
    religa argumento so quando ele muda; se este numero nao for 0, o cache
    quebrou e o custo por despacho voltou.

3.  **A lei da lacuna.** Um kernel sintetico de duracao ajustavel, em 1 e em 20
    work-groups. O resultado e `lacuna ~= 1,3 us + 0,62 x exec`: a lacuna
    escala com o TEMPO, nao com a contagem de despachos, e nao muda com mais
    grupos. E por isso que agrupar 15 despachos em 6 nao ganhou nada.

4.  **Execucao a frio.** Roda o passo inteiro e entao repete `smooth_fundido`
    oito vezes, no mesmo passo e sobre os mesmos dados -- aritmetica identica
    nas nove. A primeira custa ~145 us e a oitava ~9 us. A diferenca nao e
    conta: e busca de codigo e de dados a frio. O programa tem ~360 KB de ISA
    em 58 kernels contra 32 KB de L1 de instrucoes no gfx1031, e um work-group
    de 256 threads sao quatro wavefronts numa SIMD -- nenhuma outra para ocupar
    o lugar enquanto elas esperam.

Consequencia de metodo, e a parte que mais importa para quem continuar: medir
uma etapa ISOLADA repetindo-a 300 vezes a mantem quente e SUBESTIMA; medir
dentro do passo atribui a ela a espera pela dependencia anterior e
SUPERESTIMA. Nenhuma das duas serve sozinha. **So o relogio de parede do passo
inteiro e confiavel.**

Esta maquina varia +-30% entre corridas. As razoes se mantem; os absolutos nao.
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

ASSENTA = 150
N = 200
REP = 8

CARGA = """
__kernel void carga(__global double* x, const int K){
  __local double l[256];
  int t = get_local_id(0);
  double v = x[t & 63];
  for (int k = 0; k < K; ++k) {
    l[t] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    v = v * 0.999 + 0.001 * l[(t + 1) & 255];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) x[0] = v;
}
"""


def _eventos(dev):
    ev = [(n, e.profile.start, e.profile.end) for n, e in dev.eventos]
    dev.eventos.clear()
    return ev


def main() -> int:
    import pyopencl as cl

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
    for _ in range(30):
        g.passo_fundido()
    dev.eventos.clear()
    r = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
         "device": dev.capacidades()["nome"]}

    # 1 -- decomposicao do passo ------------------------------------------
    t0 = time.perf_counter_ns()
    for _ in range(N):
        g.passo_fundido(esperar=False)
    t1 = time.perf_counter_ns()
    dev.fila.flush()
    t2 = time.perf_counter_ns()
    dev.espera()
    t3 = time.perf_counter_ns()
    ev = _eventos(dev)
    exe = sum(b - a for _, a, b in ev) / N / 1000.0
    lac = sum(max(0, ev[i][1] - ev[i - 1][2])
              for i in range(1, len(ev))) / N / 1000.0
    r["passo"] = {
        "despachos": len(ev) // N,
        "host_enfileira_us": round((t1 - t0) / N / 1000.0, 1),
        "flush_us": round((t2 - t1) / N / 1000.0, 1),
        "espera_us": round((t3 - t2) / N / 1000.0, 1),
        "parede_us": round((t3 - t0) / N / 1000.0, 1),
        "exec_us": round(exe, 1),
        "lacuna_us": round(lac, 1),
    }
    print("  1. decomposicao do passo pipelinado")
    for k, v in r["passo"].items():
        print(f"     {k:<22} {v}")

    # 2 -- set_args no laco quente ----------------------------------------
    ligados = dict(g._args_ligados)
    g.passo_fundido()
    mudou = sum(1 for k, v in g._args_ligados.items() if ligados.get(k) != v)
    r["set_args_por_passo"] = mudou
    print(f"\n  2. set_args por passo        {mudou}   "
          f"({len(ligados)} kernels com argumentos ligados)")

    # 3 -- a lei da lacuna -------------------------------------------------
    prg = cl.Program(dev.ctx, CARGA).build()
    kc = cl.Kernel(prg, "carga")
    buf = dev.vazio(4096, np.float64)
    lei = []
    print("\n  3. lei da lacuna")
    print(f"     {'K':>6} {'grupos':>7} {'exec':>9} {'lacuna':>9} {'razao':>7}")
    for K, ng in ((10, 1), (100, 1), (1000, 1), (4000, 1),
                  (1000, 20), (4000, 20)):
        kc.set_args(buf, np.int32(K))
        for _ in range(20):
            cl.enqueue_nd_range_kernel(dev.fila, kc, (256 * ng,), (256,))
        dev.espera()
        dev.eventos.clear()
        for _ in range(4 * N):
            e = cl.enqueue_nd_range_kernel(dev.fila, kc, (256 * ng,), (256,))
            dev.eventos.append(("carga", e))
        dev.espera()
        c = _eventos(dev)
        e_ = sum(b - a for _, a, b in c) / len(c) / 1000.0
        g_ = sum(max(0, c[i][1] - c[i - 1][2])
                 for i in range(1, len(c))) / (len(c) - 1) / 1000.0
        lei.append({"K": K, "grupos": ng, "exec_us": round(e_, 1),
                    "lacuna_us": round(g_, 1)})
        print(f"     {K:>6} {ng:>7} {e_:>8.1f}u {g_:>8.1f}u "
              f"{g_ / max(e_, 1e-9):>7.2f}")
    r["lei_da_lacuna"] = lei

    # 4 -- execucao a frio -------------------------------------------------
    dev.eventos.clear()
    for _ in range(N // 4):
        g.passo_fundido(esperar=False)
        for _ in range(REP):
            g.smooth_f()
    dev.espera()
    ev = _eventos(dev)
    K = r["passo"]["despachos"] + REP
    col = [[] for _ in range(REP + 1)]
    for p in range(len(ev) // K):
        bl = ev[p * K:(p + 1) * K]
        dentro = [x for x in bl[:K - REP] if x[0] == "smooth_fundido"][0]
        col[0].append(dentro[2] - dentro[1])
        for j in range(REP):
            col[j + 1].append(bl[K - REP + j][2] - bl[K - REP + j][1])
    frio = [round(float(np.mean(c)) / 1000.0, 1) for c in col]
    r["smooth_frio_para_quente_us"] = frio
    print("\n  4. mesmo kernel, mesmos dados, mesmo passo")
    for j, v in enumerate(frio):
        rotulo = "no passo" if j == 0 else "repeticao %d" % j
        print(f"     {rotulo:<14} {v:>7.1f}u")

    isa = sum(len(b) for p in dev._programas.values() for b in p.binaries)
    r["isa_bytes"] = isa
    r["l1_instrucoes_bytes"] = 32 * 1024
    print(f"\n     ISA compilada {isa / 1024:.0f} KB   "
          f"L1 de instrucoes do gfx1031 32 KB")

    t = time.perf_counter_ns()
    for _ in range(N):
        mj.mj_step(m, d)
    r["mujoco_cpu_us"] = round((time.perf_counter_ns() - t) / N / 1000.0, 1)
    print(f"     MuJoCo CPU mj_step {r['mujoco_cpu_us']}u")

    dest = RAIZ / "benchmarks" / "physics" / "gpu" / "lacuna.json"
    dest.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    corpo.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
