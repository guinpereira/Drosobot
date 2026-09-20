"""
O mesmo passo LIF na CPU, pros mesmos tamanhos que o kernel da GPU roda.

    .venv\\Scripts\\python benchmarks\\neural\\bench_cpu.py

Sem isto o numero da GPU nao quer dizer nada: "9,4 G atualizacoes/s" so vira
informacao ao lado do que a maquina ja fazia.

A aritmetica e a de fast_lif.Camada.passo(), e a entrada e exatamente a mesma do
LifBenchmark (0.80 + 0.20*sin(i*0.001), em fp32 antes de promover) -- porque
comparar backends com estimulos diferentes nao compara nada.
"""
from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "benchmarks" / "neural"

V_REST = V_RESET = -52.0
V_TH = -45.0
TAU_M, TAU_S, T_REF, DT = 20.0, 5.0, 2.2, 0.5

TAMANHOS = [1000, 10000, 50000, 166691]
PASSOS = 2000


def passo_lif(n: int, passos: int, dtype=np.float64) -> float:
    dec_g = math.exp(-DT / TAU_S)
    dec_v = math.exp(-DT / TAU_M)
    a = 1.0 / TAU_M - 1.0 / TAU_S
    acopla = (dec_g - dec_v) / (TAU_M * a)
    ref_passos = int(round(T_REF / DT))

    i = np.arange(n)
    entrada = (np.float32(0.80) + np.float32(0.20) *
               np.sin(i.astype(np.float32) * np.float32(0.001))).astype(dtype)
    v = np.full(n, V_REST, dtype=dtype)
    g = np.zeros(n, dtype=dtype)
    ref_ate = np.full(n, -1, dtype=np.int64)

    # aquecimento: primeira passada aloca temporarios e aquece o cache
    for p in range(50):
        u = v - V_REST
        u_novo = u * dec_v + g * acopla
        g = g * dec_g + entrada
        livre = p >= ref_ate
        v = np.where(livre, V_REST + u_novo, V_RESET)
        disparou = livre & (v > V_TH)
        v = np.where(disparou, V_RESET, v)
        ref_ate = np.where(disparou, p + ref_passos, ref_ate)

    t0 = time.perf_counter()
    for p in range(passos):
        u = v - V_REST
        u_novo = u * dec_v + g * acopla
        g = g * dec_g + entrada
        livre = p >= ref_ate
        v = np.where(livre, V_REST + u_novo, V_RESET)
        disparou = livre & (v > V_TH)
        v = np.where(disparou, V_RESET, v)
        ref_ate = np.where(disparou, p + ref_passos, ref_ate)
    return time.perf_counter() - t0


def main():
    print(f"CPU: {platform.processor()}")
    print(f"{PASSOS} passos por tamanho, NumPy {np.__version__}")
    print()
    resultados = []
    for dtype, rotulo in [(np.float64, "fp64"), (np.float32, "fp32")]:
        for n in TAMANHOS:
            t = passo_lif(n, PASSOS, dtype)
            us = t / PASSOS * 1e6
            ups = n * PASSOS / t
            print(f"  {rotulo}  n={n:7d}  {us:9.1f} us/passo  "
                  f"{ups / 1e9:6.3f} G atualizacoes/s")
            resultados.append({"dtype": rotulo, "n": n,
                               "us_per_step": round(us, 3),
                               "updates_per_s": round(ups)})

    SAIDA.mkdir(parents=True, exist_ok=True)
    destino = SAIDA / f"lif_cpu_numpy_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    destino.write_text(json.dumps({
        "probed_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "numpy_cpu",
        "cpu": platform.processor(),
        "numpy": np.__version__,
        "steps": PASSOS,
        "results": resultados,
    }, indent=2), encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
