"""
Profiler do runtime. Um so, e com grandezas comparaveis.

    prof = Profiler()
    with prof("physics"):
        sim.step(drive)
    with prof("neural_lif"):
        eng.roda_poisson(...)
    prof.avanca_sim(dt_ms)
    print(prof.relatorio())

## O problema que este arquivo resolve

A versao anterior imprimia, na mesma tabela:

    physics   2130 us   (por passo de MuJoCo, dt 1e-4)
    neural   32800 us   (por JANELA de 10 ms)

Os dois numeros sao verdadeiros e nao sao comparaveis -- um e por passo de
fisica, o outro por janela neural, e eles diferem por 100x em tempo simulado.
Lado a lado davam a impressao errada de que o neural custava 15x a fisica,
quando na verdade a fisica e que domina.

A tabela principal agora normaliza tudo por **segundo simulado**, que e a unica
base em que as etapas se somam. Os custos por passo continuam disponiveis, numa
secao secundaria e rotulados com a unidade de cada um.

## Por que RTF e nao FPS

FPS da Unity mede a interface, nao a simulacao. O numero que importa e

    RTF = tempo de mosca simulado / tempo de relogio

RTF 0,04 quer dizer que 1 segundo de mosca custa 25 segundos. Nao e travamento:
e o preco, e ele fica visivel.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager

PADRAO = ["physics", "vision", "neural_lif", "neural_scatter",
          "telemetry", "render"]


class Profiler:
    """Acumula por etapa e normaliza por segundo simulado."""

    def __init__(self, etapas: list[str] | None = None):
        # ordem fixa pra tabela nao dancar entre quadros
        self.ns: OrderedDict[str, int] = OrderedDict(
            (e, 0) for e in (etapas or PADRAO))
        self.contagem: OrderedDict[str, int] = OrderedDict((k, 0) for k in self.ns)
        # unidade de cada etapa, pra secao secundaria nao misturar grandeza
        self.unidade: dict[str, str] = {}
        self.sim_ms = 0.0
        self._t0 = time.perf_counter()
        self._pausa_em: float | None = None

    @contextmanager
    def __call__(self, etapa: str, unidade: str = "chamada"):
        ini = time.perf_counter_ns()
        try:
            yield
        finally:
            dt = time.perf_counter_ns() - ini
            self.ns[etapa] = self.ns.get(etapa, 0) + dt
            self.contagem[etapa] = self.contagem.get(etapa, 0) + 1
            self.unidade[etapa] = unidade

    def soma(self, etapa: str, ns: int, unidade: str = "chamada") -> None:
        """Pra quem ja mediu por fora (kernel de GPU, por exemplo)."""
        self.ns[etapa] = self.ns.get(etapa, 0) + ns
        self.contagem[etapa] = self.contagem.get(etapa, 0) + 1
        self.unidade[etapa] = unidade

    def avanca_sim(self, ms: float) -> None:
        self.sim_ms += ms

    def pausa(self) -> None:
        """Marca o inicio de um intervalo em que nada sera simulado."""
        if self._pausa_em is None:
            self._pausa_em = time.perf_counter()

    def retoma(self) -> None:
        """
        Empurra o t0 pela duracao da pausa.

        Sem isto, pausar pela interface por trinta segundos faz o RTF despencar
        e parece que a simulacao ficou lenta -- quando na verdade ela estava
        parada a pedido. O tempo parado nao e tempo de calculo e nao pode
        entrar no denominador.
        """
        if self._pausa_em is not None:
            self._t0 += time.perf_counter() - self._pausa_em
            self._pausa_em = None

    # ------------------------------------------------------------ numeros

    @property
    def wall_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def rtf(self) -> float:
        w = self.wall_s
        return (self.sim_ms / 1000.0) / w if w > 0 else 0.0

    @property
    def medido_ns(self) -> int:
        return sum(self.ns.values())

    @property
    def nao_medido_ns(self) -> int:
        """
        O que o relogio viu e nenhuma etapa reivindicou.

        Existe de proposito: sem esta linha os percentuais somariam 100% do que
        foi INSTRUMENTADO, escondendo o que ficou de fora. Aqui eles somam 100%
        do tempo de parede, que e o unico total honesto.
        """
        return max(0, int(self.wall_s * 1e9) - self.medido_ns)

    def por_segundo_simulado(self) -> dict[str, float]:
        """ms de relogio gastos por segundo de mosca, por etapa."""
        s = self.sim_ms / 1000.0
        if s <= 0:
            return {k: 0.0 for k in self.ns}
        saida = {k: (ns / 1e6) / s for k, ns in self.ns.items()}
        saida["_outro"] = (self.nao_medido_ns / 1e6) / s
        return saida

    def zera(self) -> None:
        for k in self.ns:
            self.ns[k] = 0
            self.contagem[k] = 0
        self.sim_ms = 0.0
        self._t0 = time.perf_counter()

    def valores(self) -> dict:
        """Pra telemetria e pra UI."""
        por_s = self.por_segundo_simulado()
        parede_ns = max(1, int(self.wall_s * 1e9))
        saida = {}
        for k, ns in self.ns.items():
            n = max(1, self.contagem.get(k, 0))
            saida[k] = {
                "ms_por_seg_simulado": round(por_s.get(k, 0.0), 1),
                "pct": round(ns / parede_ns * 100, 1),
                "us_por_chamada": round(ns / n / 1000, 1),
                "unidade": self.unidade.get(k, "chamada"),
                "chamadas": self.contagem.get(k, 0),
            }
        saida["_outro"] = {
            "ms_por_seg_simulado": round(por_s.get("_outro", 0.0), 1),
            "pct": round(self.nao_medido_ns / parede_ns * 100, 1),
            "nota": "tempo de parede nao reivindicado por nenhuma etapa",
        }
        saida["_total"] = {
            "wall_s": round(self.wall_s, 2),
            "sim_ms": round(self.sim_ms, 2),
            "ms_por_seg_simulado": round(sum(por_s.values()), 1),
            "rtf": round(self.rtf, 4),
        }
        return saida

    def relatorio(self) -> str:
        v = self.valores()
        larg = max([len(k) for k in self.ns] + [8])
        linhas = ["  por segundo simulado:"]
        for k in self.ns:
            d = v[k]
            if d["chamadas"] == 0:
                continue
            linhas.append(f"    {k:<{larg}}  {d['ms_por_seg_simulado']:9.1f} ms  "
                          f"{d['pct']:5.1f}%")
        o = v["_outro"]
        linhas.append(f"    {'outro':<{larg}}  {o['ms_por_seg_simulado']:9.1f} ms  "
                      f"{o['pct']:5.1f}%   (nao instrumentado)")
        t = v["_total"]
        linhas.append("    " + "-" * (larg + 24))
        linhas.append(f"    {'parede':<{larg}}  {t['ms_por_seg_simulado']:9.1f} ms  "
                      f"100.0%   RTF {t['rtf']:.4f}x")
        linhas.append("")
        linhas.append("  por chamada (grandezas DIFERENTES, nao somar):")
        for k in self.ns:
            d = v[k]
            if d["chamadas"] == 0:
                continue
            linhas.append(f"    {k:<{larg}}  {d['us_por_chamada']:9.1f} us "
                          f"por {d['unidade']:<14s} x{d['chamadas']}")
        linhas.append("")
        linhas.append(f"  {t['sim_ms']:.1f} ms de mosca em {t['wall_s']:.1f} s "
                      f"de relogio")
        return "\n".join(linhas)
