"""
Adaptadores de fisica: o corpo, atras de uma fronteira estreita.

    PhysicsAdapter -> SensorFrame -> NeuralEngine -> MotorFrame -> PhysicsAdapter

FlyGym 1 e FlyGym 2 enxergam o MESMO cerebro porque nenhum deles atravessa esta
fronteira. Ver `adapter.py` pro criterio.

Os dois vivem em ambientes diferentes de proposito:

    .venv           FlyGym 1.2.1 / mujoco 3.2.7   REFERENCIA, nao mexer
    .venv-flygym2   FlyGym 2.1.0 / mujoco 3.9     Python >= 3.12
"""
from .adapter import MotorFrame, PhysicsAdapter, SensorFrame  # noqa: F401


def cria(nome: str, **kw):
    """Fabrica. O import e tardio: cada adaptador so existe no seu venv."""
    nome = (nome or "flygym1").lower()
    if nome in ("flygym1", "1", "fg1"):
        from .flygym1 import FlyGym1Adapter
        return FlyGym1Adapter(**kw)
    if nome in ("flygym2", "2", "fg2"):
        from .flygym2 import FlyGym2Adapter
        return FlyGym2Adapter(**kw)
    raise ValueError(f"adaptador desconhecido: {nome!r} (flygym1 | flygym2)")


__all__ = ["PhysicsAdapter", "SensorFrame", "MotorFrame", "cria"]
