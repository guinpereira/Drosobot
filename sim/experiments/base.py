"""
O que e um experimento no Drosobot.

Antes disto cada experimento era um script com laco proprio (flygym_escape.py,
flygym_optomotor.py, flygym_avoidance.py). Isso funcionou pra descobrir as
coisas, mas nao da pra escolher um experimento na interface se cada um so existe
como `python script.py`.

Esta classe nao substitui aqueles scripts -- eles continuam sendo a forma de
rodar um experimento sozinho e gerar as figuras do README. O que ela faz e dar um
CONTRATO comum pra quem precisa trocar de experimento em tempo de execucao.

Contrato minimo:

    setup(seed)           monta arena, mosca e circuito
    step()                avanca um passo de fisica; devolve o que mudou
    reset(seed)           volta ao inicio
    telemetry_metadata()  circuitos, parametros e PROCEDENCIA de cada campo
    results()             agregados da corrida

## Fronteira que nao se mexe

O experimento decide o que simular. Ele NAO recebe ordem da interface sobre o
que a mosca faz. A interface escolhe qual experimento roda e quando comeca; o
resto vem do circuito rodando sobre o conectoma, como sempre.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Experiment:
    """Base. Subclasse implementa setup/step/telemetry_metadata."""

    id: str = "abstract"
    name: str = "Experimento"
    description: str = ""

    def __init__(self, **config):
        self.config = dict(config)
        self.seed = int(config.get("seed", 0))
        self.step_index = 0
        self.sim_time = 0.0
        self._pronto = False

    # ------------------------------------------------------------- ciclo

    def setup(self, seed: int | None = None) -> None:
        raise NotImplementedError

    def step(self) -> dict[str, Any]:
        """
        Avanca UM passo de fisica.

        Devolve um dicionario com o que o runner deve publicar neste passo.
        Chaves reconhecidas: position, drive, layers, retina, derived, events.
        Ausencia de chave significa "nada novo" -- nao e erro.
        """
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        """Padrao: refaz o setup. Subclasse otimiza se valer a pena."""
        self.step_index = 0
        self.sim_time = 0.0
        self.setup(seed if seed is not None else self.seed)

    # ------------------------------------------------------- descricao

    def telemetry_metadata(self) -> dict[str, Any]:
        """
        Deve trazer `circuits`, `parameters` e `provenance`.

        A procedencia nao e enfeite: e como a interface sabe marcar o que e
        conectoma medido, o que sai do modelo de Shiu et al. e o que e suposicao
        nossa. Experimento que nao declara isso nao deveria aparecer no Lab.
        """
        raise NotImplementedError

    def scene_metadata(self) -> dict[str, Any]:
        """Arena, obstaculos e estimulo, pra Unity desenhar o ambiente."""
        return {}

    def results(self) -> dict[str, Any]:
        return {"steps": self.step_index, "sim_time": round(self.sim_time, 4)}

    # -------------------------------------------------------- utilidades

    @property
    def pronto(self) -> bool:
        return self._pronto

    def __repr__(self):
        return f"<{type(self).__name__} id={self.id} seed={self.seed}>"


_REGISTRO: dict[str, type[Experiment]] = {}


def registrar(cls: type[Experiment]) -> type[Experiment]:
    """Decorador. Todo experimento se registra pra aparecer no seletor."""
    if not cls.id or cls.id == "abstract":
        raise ValueError(f"{cls.__name__} precisa de um id proprio")
    _REGISTRO[cls.id] = cls
    return cls


def disponiveis() -> list[dict[str, str]]:
    """Lista pro seletor da interface."""
    return [{"id": c.id, "name": c.name, "description": c.description}
            for c in sorted(_REGISTRO.values(), key=lambda c: c.id)]


def criar(experiment_id: str, **config) -> Experiment:
    if experiment_id not in _REGISTRO:
        raise KeyError(f"experimento desconhecido: {experiment_id!r}. "
                       f"Conhecidos: {sorted(_REGISTRO)}")
    return _REGISTRO[experiment_id](**config)
