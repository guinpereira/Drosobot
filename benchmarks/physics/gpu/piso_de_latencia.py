r"""
O piso de latencia da GPU, medido antes de escrever um solver.

    .venv-flygym2\Scripts\python benchmarks\physics\gpu\piso_de_latencia.py

A pergunta que decide a arquitetura inteira do backend fisico: **a nossa fisica
cabe no orcamento de tempo da GPU?** O modelo do NeuroMechFly tem nv=72 e
nefc~24. Isso e minusculo. Nao ha duvida de que a GPU tem FLOPs de sobra; a
duvida e se o custo de FALAR com ela ja estoura o orcamento de 100 us/passo.

Mede quatro coisas, e cada uma refuta uma arquitetura possivel:

  1. `despacho_sincrono`   enfileirar 1 kernel e esperar. Se isto for >=
     100 us, qualquer desenho com CPU no laco por passo esta morto.
  2. `despacho_em_lote`    N kernels enfileirados, UM `finish`. E o custo
     marginal de um dispatch quando o comando ja esta na fila -- o que um
     laco de fisica gravado de antemao pagaria por passo.
  3. `kernel_persistente`  UM dispatch que roda K iteracoes por dentro, com
     `barrier()` entre elas, num unico work-group. E o desenho em que o passo
     de fisica NAO custa um dispatch: custa uma barreira.
  4. `banda_1_grupo`       quanto um unico work-group consegue ler da memoria
     global por segundo. Limita a varredura de vertices da colisao (55 pares
     x ~1000 vertices), que e o unico trecho com paralelismo de verdade.

Nao simula nada. Nao depende do MuJoCo. E um teste de hardware, e o numero que
sai dele e o teto do que um backend GPU pode prometer nesta placa.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyopencl as cl

RAIZ = Path(__file__).resolve().parents[3]

FONTE = r"""
__kernel void nulo(__global float* x) {
    if (get_global_id(0) == 0xffffffu) x[0] = 1.0f;   // nunca executa
}

// K iteracoes com barreira entre elas, um work-group so.
// Imita o laco de fisica rodando inteiro por dentro de um dispatch.
__kernel void persistente(__global float* estado, const int K, const int N) {
    __local float buf[256];
    int t = get_local_id(0);
    float v = estado[t % N];
    for (int k = 0; k < K; ++k) {
        buf[t] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        // dependencia real entre iteracoes: le o vizinho
        v = v * 0.999f + 0.001f * buf[(t + 1) & 255];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t < N) estado[t] = v;
}

// varredura de vertices: max do produto escalar, que e a funcao de suporte
__kernel void suporte(__global const float4* verts, const int nvert,
                      const float4 dir, __global float* saida, const int K) {
    __local float red[256];
    int t = get_local_id(0);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float best = -1e30f;
        for (int i = t; i < nvert; i += 256) {
            float4 v = verts[i];
            float d = v.x*dir.x + v.y*dir.y + v.z*dir.z;
            best = fmax(best, d);
        }
        red[t] = best;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 128; s > 0; s >>= 1) {
            if (t < s) red[t] = fmax(red[t], red[t + s]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        acc += red[0];
    }
    if (t == 0) saida[0] = acc;
}
"""


def _device():
    for p in cl.get_platforms():
        for d in p.get_devices(cl.device_type.GPU):
            return d
    raise SystemExit("sem GPU OpenCL nesta maquina")


def _mede(fn, repeticoes: int, aquece: int = 20) -> float:
    for _ in range(aquece):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(repeticoes):
        fn()
    return (time.perf_counter_ns() - t0) / repeticoes / 1000.0     # us


def main() -> int:
    dev = _device()
    ctx = cl.Context([dev])
    fila = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, FONTE).build()
    # Os kernels sao recuperados UMA vez. `prog.nome(...)` cria um objeto novo
    # por chamada, e esse custo -- centenas de us -- e maior que tudo que este
    # arquivo tenta medir. Foi assim que a primeira versao mediu 250 us por
    # dispatch nulo.
    k_nulo = cl.Kernel(prog, "nulo")
    k_pers = cl.Kernel(prog, "persistente")
    k_sup = cl.Kernel(prog, "suporte")
    mf = cl.mem_flags

    buf = cl.Buffer(ctx, mf.READ_WRITE, size=4 * 4096)
    cl.enqueue_fill_buffer(fila, buf, np.float32(1.0), 0, 4 * 4096)
    fila.finish()

    r = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "device": dev.name.strip(),
        "cus": int(dev.max_compute_units),
        "clock_mhz": int(dev.max_clock_frequency),
        "opencl": dev.version.strip(),
    }

    # 1. dispatch sincrono: enfileira e espera
    k_nulo.set_args(buf)

    def sinc():
        cl.enqueue_nd_range_kernel(fila, k_nulo, (64,), (64,))
        fila.finish()
    r["despacho_sincrono_us"] = round(_mede(sinc, 2000), 3)

    # 2. dispatch em lote: N kernels, um finish
    for n in (10, 100, 1000):
        def lote(n=n):
            for _ in range(n):
                cl.enqueue_nd_range_kernel(fila, k_nulo, (64,), (64,))
            fila.finish()
        total = _mede(lote, max(5, 2000 // n))
        r[f"despacho_em_lote_{n}_us_por_kernel"] = round(total / n, 4)

    # 3. kernel persistente: K iteracoes com barreira, um work-group
    for k in (1, 100, 1000, 10000):
        def pers(k=k):
            k_pers.set_args(buf, np.int32(k), np.int32(72))
            cl.enqueue_nd_range_kernel(fila, k_pers, (256,), (256,))
            fila.finish()
        total = _mede(pers, max(5, 2000 // max(1, k // 10)))
        r[f"persistente_{k}_iters_total_us"] = round(total, 3)
    base = r["persistente_1_iters_total_us"]
    r["persistente_us_por_iteracao"] = round(
        (r["persistente_10000_iters_total_us"] - base) / 9999.0, 4)

    # 4. banda de um work-group: varredura de vertices
    nvert = 1024
    verts = np.random.default_rng(0).standard_normal((nvert, 4)).astype(np.float32)
    vbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=verts)
    obuf = cl.Buffer(ctx, mf.WRITE_ONLY, size=4)
    direcao = np.array([0, 0, -1, 0], dtype=np.float32)
    # K grande de proposito: o piso de ~90 us de um dispatch sincrono e maior
    # que a varredura inteira, entao medir 55 varreduras direto mediria o piso.
    # Com K=5500 o custo marginal aparece, e dividimos por 100.
    for k in (100, 5500):
        def sup(k=k):
            k_sup.set_args(vbuf, np.int32(nvert),
                           cl.cltypes.make_float4(*direcao), obuf, np.int32(k))
            cl.enqueue_nd_range_kernel(fila, k_sup, (256,), (256,))
            fila.finish()
        total = _mede(sup, 60 if k > 1000 else 200)
        r[f"suporte_{k}x{nvert}verts_us"] = round(total, 3)
    marginal = ((r[f"suporte_5500x{nvert}verts_us"]
                 - r[f"suporte_100x{nvert}verts_us"]) / 5400.0)
    r["suporte_1_varredura_us"] = round(marginal, 4)
    r["suporte_55_varreduras_us"] = round(55 * marginal, 3)
    r["banda_1_grupo_GBs"] = round(nvert * 16 / (marginal * 1e-6) / 1e9, 2)

    dest = RAIZ / "benchmarks" / "physics" / "gpu" / "piso_de_latencia.json"
    dest.write_text(json.dumps(r, indent=1), encoding="utf-8")
    for k, v in r.items():
        print(f"{k:<40} {v}")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
