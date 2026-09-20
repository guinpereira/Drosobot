"""
Onde vai o tempo do passo neural: compute de verdade, ou overhead em volta?

    .venv-flygym2\\Scripts\\python benchmarks\\neural\\profile_neural_step.py

Depois que a fisica encolheu 2,5x, o neural passou a ser 35% do relogio -- ~3,1
s por segundo simulado, com 2.000 passos neurais nesse segundo. Sao ~1,5 ms por
passo. A pergunta e se isso e a GPU trabalhando ou a gente atrapalhando.

Separa, por passo neural:

    host_rng        sorteio Poisson na CPU
    host_mascara    escrever a mascara de spike forcado
    enq_forcado     enfileirar a copia da mascara pro device
    enq_lif         enfileirar o kernel do LIF
    enq_scatter     enfileirar o kernel de propagacao
    enq_acumula     enfileirar as reducoes (contagem e por grupo)
    finish          esperar a fila esvaziar

E, pelos EVENTOS do OpenCL (fila com profiling ligado), o tempo de GPU de cada
kernel. A soma dos kernels contra o relogio de parede diz, sem discussao, se o
custo e compute ou overhead de lancamento/transferencia.

Nao muda nada: mesmo conectoma, mesmo kernel, mesma semente.
"""
from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

SAIDA = RAIZ / "benchmarks" / "neural"
JANELAS = 20          # 200 ms de mosca: 400 passos neurais
JANELA_MS = 10.0


class Relogio:
    def __init__(self):
        self.ns: OrderedDict[str, int] = OrderedDict()
        self.n: OrderedDict[str, int] = OrderedDict()

    def marca(self, etapa, t0):
        dt = time.perf_counter_ns() - t0
        self.ns[etapa] = self.ns.get(etapa, 0) + dt
        self.n[etapa] = self.n.get(etapa, 0) + 1


def main() -> None:
    import pyopencl as cl
    from drosobot_lab import monta_cerebro

    print("montando o Male CNS inteiro na GPU...")
    eng, papeis, _ = monta_cerebro("whole", "opencl")
    b = eng.backend
    r = eng.resumo()
    print(f"  {r['neurons_simulated']:,} neuronios, {r['edges_simulated']:,} "
          f"arestas, {r['device']}")

    # Fila com PROFILING: os eventos passam a carregar inicio e fim em ns de
    # device. Sem isso nao da pra separar compute de overhead -- o relogio da
    # CPU so ve a chamada de enfileiramento, que retorna antes de a GPU comecar.
    fila_perfil = cl.CommandQueue(
        b.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    b.fila = fila_perfil

    sens = papeis.get("LC4/LPLC2", np.zeros(0, np.int32))
    idx = np.asarray(sens, dtype=np.int64)
    rng = np.random.default_rng(0)
    # taxa tipica do looming perto do pico, pra que o scatter tenha trabalho
    p = np.full(len(idx), 20.0 * (eng.dt / 1000.0))
    mascara = np.zeros(eng.c.n, dtype=np.uint8)

    passos_por_janela = int(round(JANELA_MS / eng.dt))
    c = Relogio()
    gpu_ns = OrderedDict()
    eventos = []

    # aquecimento: primeira compilacao de kernel e primeira alocacao
    for _ in range(2):
        eng.roda_poisson(JANELA_MS, np.zeros(eng.c.n), rng, indices=sens)

    print(f"rodando {JANELAS} janelas ({JANELAS * passos_por_janela} passos "
          f"neurais)...")
    parede0 = time.perf_counter_ns()
    for _ in range(JANELAS):
        b.escreve_externo(None)
        for _ in range(passos_por_janela):
            t0 = time.perf_counter_ns()
            sorteio = rng.random(len(idx)) < p
            c.marca("host_rng", t0)

            t0 = time.perf_counter_ns()
            mascara[idx] = sorteio.astype(np.uint8)
            c.marca("host_mascara", t0)

            t0 = time.perf_counter_ns()
            e = cl.enqueue_copy(b.fila, b.b_forcado, mascara)
            c.marca("enq_forcado", t0)
            eventos.append(("copia_forcado", e))

            t0 = time.perf_counter_ns()
            b.lif(eng._passo, eng.cursor)
            c.marca("enq_lif", t0)

            t0 = time.perf_counter_ns()
            b.scatter(eng.cursor)
            c.marca("enq_scatter", t0)

            t0 = time.perf_counter_ns()
            b.acumula()
            c.marca("enq_acumula", t0)

            eng.cursor = (eng.cursor + 1) % eng.coef.atraso_passos
            eng._passo += 1

        t0 = time.perf_counter_ns()
        b.fila.finish()
        c.marca("finish", t0)
    parede_ns = time.perf_counter_ns() - parede0

    # ------------------------------------------------- tempo de GPU por evento
    for nome, ev in eventos:
        try:
            gpu_ns[nome] = gpu_ns.get(nome, 0) + (ev.profile.end - ev.profile.start)
        except cl.RuntimeError:
            pass

    n_passos = JANELAS * passos_por_janela
    sim_ms = JANELAS * JANELA_MS

    def por_seg(ns):
        return ns / 1e6 / (sim_ms / 1000.0)

    print()
    print("  HOST -- onde a CPU gasta (por passo neural)")
    print(f"  {'etapa':<16s} {'us/passo':>10s} {'ms/s sim':>10s} {'%parede':>8s}")
    print("  " + "-" * 48)
    medido = 0
    linhas = {}
    for etapa, ns in sorted(c.ns.items(), key=lambda kv: -kv[1]):
        medido += ns
        us = ns / 1000 / c.n[etapa]
        linhas[etapa] = {"us_por_passo": round(us, 2),
                         "ms_por_seg_simulado": round(por_seg(ns), 1),
                         "pct": round(100 * ns / parede_ns, 1)}
        print(f"  {etapa:<16s} {us:10.2f} {por_seg(ns):10.1f} "
              f"{100 * ns / parede_ns:7.1f}%")
    resto = parede_ns - medido
    print(f"  {'outro':<16s} {'':>10s} {por_seg(resto):10.1f} "
          f"{100 * resto / parede_ns:7.1f}%")
    print("  " + "-" * 48)
    print(f"  {'parede':<16s} {parede_ns / 1000 / n_passos:10.2f} "
          f"{por_seg(parede_ns):10.1f} {100.0:7.1f}%")

    print()
    print("  GPU -- tempo de device pelos eventos do OpenCL")
    total_gpu = sum(gpu_ns.values())
    for nome, ns in gpu_ns.items():
        print(f"  {nome:<16s} {ns / 1000 / n_passos:10.2f} us/passo")
    print(f"  {'total medido':<16s} {total_gpu / 1000 / n_passos:10.2f} us/passo"
          f"   ({100 * total_gpu / parede_ns:.1f}% da parede)")
    print()
    print("  NOTA: so a copia da mascara esta instrumentada por evento aqui --")
    print("  os kernels sao enfileirados por dentro do backend, que nao devolve")
    print("  o evento. O que este numero prova e o custo da TRANSFERENCIA.")

    SAIDA.mkdir(parents=True, exist_ok=True)
    caminho = SAIDA / "neural_step_profile.json"
    caminho.write_text(json.dumps({
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "neuronios": int(r["neurons_simulated"]),
        "arestas": int(r["edges_simulated"]),
        "device": r["device"],
        "passos_neurais": n_passos,
        "sim_ms": sim_ms,
        "parede_ms": round(parede_ns / 1e6, 1),
        "host": linhas,
        "gpu_us_por_passo": {k: round(v / 1000 / n_passos, 2)
                             for k, v in gpu_ns.items()},
    }, indent=1), encoding="utf-8")
    print(f"\n  salvo em {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
