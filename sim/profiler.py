"""
Profiler do runtime. Um so, em vez de um benchmark por pergunta.

    prof = Profiler()
    with prof("physics"):
        sim.step(drive)
    with prof("neural"):
        eng.roda(10.0, externo)
    print(prof.relatorio())

Substitui a pratica anterior de escrever um script de medicao separado pra cada
duvida. O custo por etapa passa a vir da corrida de verdade, no mesmo lugar onde
o experimento acontece, e vai junto na telemetria.

## Por que RTF e nao FPS

FPS da Unity mede a interface, nao a simulacao. O numero que importa aqui e

    RTF = tempo de mosca simulado / tempo de relogio

Um RTF de 0,04 quer dizer que 1 segundo de mosca custa 25 segundos. Isso nao e
travamento -- e o preco, e ele tem que ficar visivel.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager


class Profiler:
    """Acumula tempo por etapa e devolve percentuais e RTF."""

    def __init__(self, etapas: list[str] | None = None):
        # ordem fixa pra tabela nao dancar entre quadros
        self.ns: OrderedDict[str, int] = OrderedDict(
            (e, 0) for e in (etapas or
                             ["physics", "vision", "neural_lif",
                              "neural_scatter", "telemetry", "render"]))
        self.contagem: OrderedDict[str, int] = OrderedDict(
            (k, 0) for k in self.ns)
        self.sim_ms = 0.0
        self._t0 = time.perf_counter()

    @contextmanager
    def __call__(self, etapa: str):
        ini = time.perf_counter_ns()
        try:
            yield
        finally:
            dt = time.perf_counter_ns() - ini
            self.ns[etapa] = self.ns.get(etapa, 0) + dt
            self.contagem[etapa] = self.contagem.get(etapa, 0) + 1

    def soma(self, etapa: str, ns: int) -> None:
        """Pra quem ja mediu por fora (kernel de GPU, por exemplo)."""
        self.ns[etapa] = self.ns.get(etapa, 0) + ns
        self.contagem[etapa] = self.contagem.get(etapa, 0) + 1

    def avanca_sim(self, ms: float) -> None:
        self.sim_ms += ms

    @property
    def wall_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def rtf(self) -> float:
        w = self.wall_s
        return (self.sim_ms / 1000.0) / w if w > 0 else 0.0

    @property
    def total_ns(self) -> int:
        return sum(self.ns.values())

    def zera(self) -> None:
        for k in self.ns:
            self.ns[k] = 0
            self.contagem[k] = 0
        self.sim_ms = 0.0
        self._t0 = time.perf_counter()

    def valores(self) -> dict:
        """Pro telemetry e pra UI. Tempo MEDIO por chamada, em us."""
        total = max(1, self.total_ns)
        saida = {}
        for k, ns in self.ns.items():
            n = max(1, self.contagem.get(k, 0))
            saida[k] = {
                "us": round(ns / n / 1000, 1),
                "total_ms": round(ns / 1e6, 2),
                "pct": round(ns / total * 100, 1),
                "calls": self.contagem.get(k, 0),
            }
        saida["_total"] = {
            "ms": round(self.total_ns / 1e6, 2),
            "wall_s": round(self.wall_s, 2),
            "sim_ms": round(self.sim_ms, 2),
            "rtf": round(self.rtf, 4),
        }
        return saida

    def relatorio(self) -> str:
        v = self.valores()
        largura = max((len(k) for k in self.ns), default=10)
        linhas = []
        for k in self.ns:
            d = v[k]
            if d["calls"] == 0:
                continue
            linhas.append(f"  {k:<{largura}}  {d['us']:9.1f} us  "
                          f"{d['pct']:5.1f}%  {d['calls']:6d} chamadas")
        t = v["_total"]
        linhas.append("  " + "-" * (largura + 34))
        linhas.append(f"  {'total':<{largura}}  {t['ms']:9.2f} ms de trabalho   "
                      f"RTF {t['rtf']:.4f}x")
        linhas.append(f"  {'':<{largura}}  {t['sim_ms']:9.1f} ms de mosca em "
                      f"{t['wall_s']:.1f} s de relogio")
        return "\n".join(linhas)
