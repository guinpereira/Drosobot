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
from .backends import BACKENDS, adaptador_de, canonico, descreve  # noqa: F401


def cria(nome: str, **kw):
    """
    Fabrica. O import e tardio: cada adaptador so existe no seu venv.

    O nome que entra e o do BACKEND (`flygym2-mujoco`, `drosobot-gpu`), nao o do
    adaptador. Quem traduz um no outro e `backends.py`, porque backends
    diferentes montam o mesmo modelo de proposito -- se `drosobot-gpu` montasse
    um corpo proprio, a comparacao com o MuJoCo mediria dois corpos, nao dois
    solvers.
    """
    alvo = adaptador_de(nome)
    if alvo == "flygym1":
        from .flygym1 import FlyGym1Adapter
        return FlyGym1Adapter(**kw)
    if alvo == "drosobot-gpu":
        from .drosobot_gpu import DrosobotGPUAdapter
        return DrosobotGPUAdapter(**kw)
    from .flygym2 import FlyGym2Adapter
    return FlyGym2Adapter(**kw)


__all__ = ["PhysicsAdapter", "SensorFrame", "MotorFrame", "cria",
           "BACKENDS", "canonico", "descreve", "adaptador_de"]
