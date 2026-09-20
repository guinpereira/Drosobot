"""
Backends de compute: device, memoria, dispatch, sincronizacao, atomics.

Nenhuma regra cientifica mora aqui. Ver `base.py` pro criterio.
"""
from .base import BackendIndisponivel, IComputeBackend  # noqa: F401
from .cpu import CPUBackend  # noqa: F401

__all__ = ["IComputeBackend", "BackendIndisponivel", "CPUBackend"]
