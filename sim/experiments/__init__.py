"""
Os experimentos do Drosobot, num registro so.

    from sim.experiments import disponiveis, criar

    disponiveis()                      # pro seletor da interface
    exp = criar("looming_escape", seed=0)
    exp.setup()
    while ...: exp.step()

Cada experimento continua tendo o script headless que produz a figura do README
(sim/flygym_escape.py, sim/flygym_optomotor.py, sim/flygym_avoidance.py). Estes
aqui rodam o MESMO circuito com os MESMOS parametros -- divergencia entre os dois
e bug, nao melhoria.
"""
from .base import Experiment, criar, disponiveis, registrar  # noqa: F401

# importar registra; sem isto o registro fica vazio
from . import looming, obstacles, optomotor  # noqa: F401,E402

__all__ = ["Experiment", "criar", "disponiveis", "registrar"]
