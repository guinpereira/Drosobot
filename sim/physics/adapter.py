"""
PhysicsAdapter: a fronteira entre o corpo e o cerebro.

    PhysicsAdapter  ->  SensorFrame  ->  NeuralEngine  ->  MotorFrame  ->  PhysicsAdapter

O motor neural NAO importa FlyGym, nao sabe o que e uma arena e nao sabe qual
versao do simulador esta rodando. Ele recebe um `SensorFrame` e devolve um
`MotorFrame`. Isso e o que permite FlyGym 1 e FlyGym 2 enxergarem o MESMO
cerebro, sem duplicar o backend neural.

O criterio e o mesmo de `sim/neural/compute/base.py`, aplicado do outro lado:
trocar de simulador de fisica nao pode exigir mexer em `model.py` nem em
`engine.py`.

## O que atravessa a fronteira

    SensorFrame    retina (2 olhos), pose, orientacao, contatos, tempo
    MotorFrame     drive de marcha, e o que mais um controlador precisar

Nada de objeto do FlyGym, nada de `mjData`, nada de nome de geom. Se um campo
so faz sentido numa das versoes, ele nao atravessa.

## Por que dois adaptadores e nao uma camada de compatibilidade

FlyGym 2.x nao e upgrade do 1.x: e outra API sobre outra base (Python >=3.12,
mujoco >=3.9, nomes de segmento diferentes -- `LFTarsus1` virou `lf_tarsus1`).
Uma camada que tentasse esconder isso ficaria maior e mais fragil que os dois
adaptadores somados. Entao sao adaptadores separados, e a fronteira e estreita
de proposito.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class SensorFrame:
    """O que o corpo entrega ao cerebro num instante."""
    t_s: float
    passo: int
    # retina: (2, n_omatideos) ja reduzida a um valor por omatideo
    retina: np.ndarray | None = None
    retina_atualizou: bool = False
    posicao: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientacao: np.ndarray | None = None      # quaternio (w, x, y, z)
    contatos: np.ndarray | None = None
    # pose dos segmentos do corpo, so quando pedida (ver `pose_corpo()`).
    # Nao vem por passo: sao ~70 segmentos x 7 floats, e a fisica anda 10.000
    # vezes por segundo simulado enquanto a visualizacao precisa de ~30.
    segmentos: list[str] | None = None
    seg_pos: np.ndarray | None = None      # (n, 3) mm
    seg_quat: np.ndarray | None = None     # (n, 4) w,x,y,z
    extras: dict = field(default_factory=dict)


@dataclass
class MotorFrame:
    """O que o cerebro devolve ao corpo."""
    drive: np.ndarray = field(default_factory=lambda: np.ones(2))
    adesao: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


class PhysicsAdapter(Protocol):
    """
    Contrato do corpo. Duas implementacoes: FlyGym1Adapter, FlyGym2Adapter.

    Ciclo:
        adapter.reset(seed)
        while ...:
            adapter.antes_do_passo(t_s)      # move o estimulo, se houver
            frame = adapter.passo(motor)
    """

    nome: str
    versao: str

    def reset(self, seed: int = 0) -> SensorFrame: ...

    def antes_do_passo(self, t_s: float) -> None:
        """Atualiza estimulo controlado pelo laco (esfera de looming etc.)."""
        ...

    def passo(self, motor: MotorFrame) -> SensorFrame:
        """Avanca UM passo de fisica e devolve o que os sensores viram."""
        ...

    @property
    def timestep(self) -> float: ...

    @property
    def n_pares_colisao(self) -> int: ...

    def pose_corpo(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        (nomes, posicoes, quaternios) de todos os segmentos.

        Chamado na cadencia da TELEMETRIA, nao na da fisica. A Unity reconstroi
        a pose a partir disto; interpolar entre quadros na visualizacao e
        permitido, desde que nada volte pra fisica.
        """
        ...

    def resumo(self) -> dict:
        """Pro profiler, pra telemetria e pra interface."""
        ...

    def fecha(self) -> None: ...
