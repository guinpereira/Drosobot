r"""
Trace do estado fisico por passo: o arquivo que faltava para localizar divergencia.

    from lab.trace_fisico import TraceFisico
    tr = TraceFisico(pasta, cada=10)
    tr.linha(passo, t_s, qpos=..., qvel=..., qacc=..., contatos=..., forcas=...)
    tr.fecha()

## Por que um arquivo novo, e nao mais colunas no timeseries.csv

As colunas do `timeseries.csv` sao fixas de proposito: uma coluna que muda de
lugar entre corridas quebra qualquer analise que as compare, e o arquivo e lido
por quem quer plotar, nao por quem quer depurar fisica.

O que o depurador de fisica precisa -- `qpos` (73), `qvel` (72), `qacc` (72),
contatos e forcas de restricao -- sao centenas de numeros por PASSO, nao por
janela neural. Isso e 10.000 linhas por segundo simulado contra 100. Numa
corrida de 600 s seriam milhoes de linhas de algo que ninguem plota.

Entao: arquivo proprio, **desligado por padrao**, com passo configuravel.

## Formato

JSONL, uma linha por amostra. Nao CSV: o numero de contatos muda de passo para
passo, e uma tabela de largura fixa ou mentiria (truncando) ou desperdicaria
(preenchendo com vazio).

    {"passo": 3, "t_s": 0.0003, "qpos": [...], "qvel": [...], "qacc": [...],
     "ncon": 5, "contatos": [{"geom": [0, 30], "dist": -0.0006, "pos": [...]}],
     "nefc": 20, "efc_force": [...], "qfrc_constraint": [...]}

Os campos ausentes simplesmente nao aparecem -- quem escreve decide o que tem.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _lista(v, casas: int = 12):
    """Array -> lista de float, arredondada. `None` continua `None`."""
    if v is None:
        return None
    a = np.asarray(v, dtype=float).ravel()
    return [round(float(x), casas) for x in a]


class TraceFisico:
    """
    Escreve `physics_trace.jsonl` na pasta da corrida.

    `cada`: amostra um passo a cada N. `cada=1` grava tudo -- util para um
    punhado de passos, caro para uma corrida inteira. `ativo=False` faz de
    todos os metodos um no-op, que e como ele fica nas corridas normais.
    """

    NOME = "physics_trace.jsonl"

    def __init__(self, pasta, cada: int = 1, ativo: bool = True,
                 casas: int = 12):
        self.ativo = bool(ativo)
        self.cada = max(1, int(cada))
        self.casas = int(casas)
        self.n = 0
        self._f = None
        if self.ativo:
            p = Path(pasta)
            p.mkdir(parents=True, exist_ok=True)
            self._f = (p / self.NOME).open("w", encoding="utf-8")

    def linha(self, passo: int, t_s: float, *, qpos=None, qvel=None, qacc=None,
              contatos=None, efc_force=None, qfrc_constraint=None,
              extras: dict | None = None) -> None:
        if not self.ativo or (passo % self.cada):
            return
        d = {"passo": int(passo), "t_s": round(float(t_s), 9)}
        for nome, v in (("qpos", qpos), ("qvel", qvel), ("qacc", qacc),
                        ("efc_force", efc_force),
                        ("qfrc_constraint", qfrc_constraint)):
            if v is not None:
                d[nome] = _lista(v, self.casas)
        if contatos is not None:
            d["ncon"] = len(contatos)
            d["contatos"] = [
                {"geom": [int(c["geom"][0]), int(c["geom"][1])],
                 "dist": round(float(c["dist"]), self.casas),
                 "pos": _lista(c["pos"], self.casas)}
                for c in contatos]
        if efc_force is not None:
            d["nefc"] = len(d["efc_force"])
        if extras:
            d.update(extras)
        self._f.write(json.dumps(d) + "\n")
        self.n += 1

    def fecha(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def le(pasta) -> list[dict]:
    """Le um trace inteiro. Nao e streaming: quem grava muito usa `cada`."""
    p = Path(pasta) / TraceFisico.NOME
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def primeira_divergencia(a: list[dict], b: list[dict], campos=("qpos", "qvel",
                                                              "qacc"),
                         tol: float = 1e-9) -> dict | None:
    """
    O primeiro passo, e o primeiro CAMPO, em que dois traces se separam.

    Olhar so o estado final nao distingue "divergiu no primeiro contato e se
    reencontrou" de "andou junto e se separou no fim". A ordem dos campos
    importa: eles estao na ordem causal, entao o primeiro a divergir e a causa
    e os seguintes herdam.
    """
    for la, lb in zip(a, b):
        if la["passo"] != lb["passo"]:
            continue
        for campo in campos:
            if campo not in la or campo not in lb:
                continue
            va = np.asarray(la[campo], dtype=float)
            vb = np.asarray(lb[campo], dtype=float)
            if va.shape != vb.shape:
                return {"passo": la["passo"], "campo": campo,
                        "motivo": f"formas diferentes {va.shape} vs {vb.shape}"}
            err = float(np.abs(va - vb).max())
            esc = max(1e-30, float(np.abs(vb).max()))
            if err / esc > tol:
                return {"passo": la["passo"], "campo": campo,
                        "erro_abs": err, "erro_rel": err / esc,
                        "indice": int(np.argmax(np.abs(va - vb)))}
        if la.get("ncon") != lb.get("ncon"):
            return {"passo": la["passo"], "campo": "ncon",
                    "motivo": f"{la.get('ncon')} vs {lb.get('ncon')}"}
    return None
