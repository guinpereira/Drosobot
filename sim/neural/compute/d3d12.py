"""
Backend D3D12/HLSL -- preservado, ainda nao ligado ao runtime em Python.

## Por que ele existe e nao esta ativo

O caminho D3D12 foi o PRIMEIRO a rodar nesta placa e esta medido:

    166.691 estados LIF   17,9 us/passo      (RX 6700 XT, D3D12)
    25,5M arestas         6,44 ms a 100% de disparo, 3,97 G eventos/s

Os kernels HLSL existem e estao validados contra a referencia
(`unity/DrosobotLab/Assets/Compute/`, `tests/test_lif_gpu_equivalencia.py`).
Esse trabalho nao se perde.

O que falta nao e o D3D12: e um HOST nativo. Hoje os kernels rodam dentro do
processo da Unity, que e onde havia um device D3D12 pronto. Para o runtime em
Python, D3D12 precisa de um processo nativo proprio (C++ ou Rust) expondo
device, buffers e dispatch -- o que e perfeitamente possivel headless, sem
Unity, e simplesmente ainda nao foi escrito.

Entao a escolha do OpenCL como primeiro backend de runtime NAO e "D3D12 exige
Unity" -- nao exige. E "temos binding de host pronto pro OpenCL em Python e
ainda nao temos pro D3D12".

## O que este arquivo garante

Que a fronteira esta no lugar certo. Implementar D3D12 e preencher os metodos
abaixo; `engine.py` e `model.py` nao mudam. Se algum dia preencher isto exigir
mexer neles, o acoplamento esta errado e o erro esta la, nao aqui.

## Caminho previsto

  1. host nativo minimo (C++/D3D12) com device, buffer, dispatch e fence
  2. expor por ctypes/pybind11 as mesmas primitivas de IComputeBackend
  3. reusar os .hlsl que ja existem, traduzidos dos .cl de sim/neural/kernels/
  4. rodar tests/test_neural_backend.py contra ele -- os MESMOS testes
  5. comparar OpenCL x D3D12 no mesmo conectoma e no mesmo estimulo

O passo 4 e o que importa: um backend so entra quando passa na mesma validacao.
"""
from __future__ import annotations

import numpy as np

from ..model import Coeficientes, Conectoma, Estado
from .base import BackendIndisponivel

MENSAGEM = (
    "D3D12Backend ainda nao tem host nativo. Os kernels HLSL existem e estao "
    "validados (unity/DrosobotLab/Assets/Compute/), mas falta o processo "
    "nativo que exponha device/buffer/dispatch pro Python. Use backend='opencl' "
    "ou backend='cpu'. Ver o cabecalho de sim/neural/compute/d3d12.py."
)


class D3D12Backend:
    """Esqueleto declarado. Levanta com instrucao, nunca finge funcionar."""

    nome = "drosobot-d3d12"
    device = "(host nativo ausente)"

    def __init__(self):
        raise BackendIndisponivel(MENSAGEM)

    # A interface fica escrita mesmo inalcancavel: ela e o contrato que quem
    # for implementar precisa cumprir, e serve de checklist.
    def prepara(self, conectoma: Conectoma, coef: Coeficientes,
                grupos: np.ndarray | None) -> None: ...
    def reset(self) -> None: ...
    def escreve_externo(self, externo_mV: np.ndarray | None) -> None: ...
    def lif(self, passo: int, cursor: int) -> None: ...
    def scatter(self, cursor: int) -> None: ...
    def acumula(self) -> None: ...
    def sincroniza(self) -> None: ...
    def le_indices(self, indices: np.ndarray) -> Estado: ...
    def le_contagem_grupos(self) -> np.ndarray: ...
    def zera_contagem_grupos(self) -> None: ...
    def le_estado_completo(self): ...
    def le_contagem_total(self) -> np.ndarray: ...
    def resumo(self) -> dict: ...
