"""
Drosobot Neural Engine: o mesmo circuito, em qualquer backend.

    from sim.neural import NeuralEngine, carrega_male_cns

    c = carrega_male_cns()                      # 164.451 neuronios, 25,5M arestas
    eng = NeuralEngine(c, backend="opencl")     # ou "cpu", "d3d12", "auto"
    eng.roda(10.0, externo_mV)
    est = eng.le(indices_visiveis)

## Onde esta cada coisa

    model.py              parametros de Shiu et al., coeficientes, CSR, estado
    engine.py             a ordem do passo e a semantica do atraso
    connectome_loader.py  o Male CNS real como CSR
    kernels/*.cl          nossos kernels
    compute/base.py       IComputeBackend -- a fronteira
    compute/cpu.py        referencia em fp64
    compute/opencl.py     OpenCL (pyopencl so como binding de host)
    compute/d3d12.py      esqueleto preservado, host nativo pendente

O criterio da fronteira: trocar de API de GPU nao pode exigir mexer em `model.py`
nem em `engine.py`.
"""
from .connectome_loader import (  # noqa: F401
    ConectomaAusente, carrega_male_cns, existe, metadados, rede_sintetica,
    subgrafo,
)
from .compute.base import BackendIndisponivel, IComputeBackend  # noqa: F401
from .compute.cpu import CPUBackend  # noqa: F401
from .compute.opencl import dispositivos  # noqa: F401
from .engine import NeuralEngine  # noqa: F401
from .model import (  # noqa: F401
    Coeficientes, Conectoma, Estado, DT_MS, ESCALA, T_DELAY, T_REF,
    pico_convergente_mV, verifica_escala,
    TAU_M, TAU_S, V_RESET, V_REST, V_TH, W_SYN_MV,
)

__all__ = [
    "NeuralEngine", "Conectoma", "Estado", "Coeficientes",
    "IComputeBackend", "BackendIndisponivel", "CPUBackend", "dispositivos",
    "carrega_male_cns", "subgrafo", "rede_sintetica", "metadados", "existe",
    "ConectomaAusente",
]
