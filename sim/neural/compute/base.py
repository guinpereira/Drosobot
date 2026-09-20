"""
IComputeBackend: a fronteira entre a ciencia do Drosobot e a API de GPU.

## O criterio que define esta fronteira

Se um dia removermos o OpenCL e portarmos pro Vulkan ou pro D3D12, a arquitetura
cientifica tem que ficar INTACTA. Se remover o OpenCL exigir reescrever o modelo
neural, o acoplamento esta errado.

Na pratica isso quer dizer: trocar de backend e implementar os metodos abaixo.
`engine.py` nao muda uma linha.

## O que fica de cada lado

    Drosobot (sim/neural/model.py, engine.py)
        representacao do conectoma e layout CSR
        modelo LIF e seus parametros
        semantica do atraso e do anel
        ordem do passo
        regra de sinal por neurotransmissor
        agregacao por populacao
        validacao

    Backend (compute/opencl.py, compute/d3d12.py, compute/cpu.py)
        device
        memoria
        dispatch
        sincronizacao
        atomics

O backend recebe primitivas ja decididas -- "execute o LIF com este cursor",
"espalhe" -- e nao sabe o que e um neurotransmissor, por que o atraso e 4 passos
nem qual populacao e LC4. Nada disso aparece na interface, de proposito.

## Por que nao e um wrapper de biblioteca

Os kernels sao nossos (`sim/neural/kernels/*.cl`). PyOpenCL entra como BINDING
de host -- equivalente a usar C++ com a API de OpenCL. Nenhuma regra cientifica
vive dentro de chamada de biblioteca de terceiro.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..model import Coeficientes, Conectoma, Estado


@runtime_checkable
class IComputeBackend(Protocol):
    """
    Executa as primitivas do passo neural. Nao decide a ordem delas.

    Ciclo de vida:

        backend.prepara(conectoma, coef, grupos)   uma vez, sobe o estatico
        backend.reset()                            zera o estado dinamico
        ... por passo, na ordem que o engine define:
        backend.escreve_externo(arr)
        backend.lif(passo, cursor)
        backend.scatter(cursor)
        backend.acumula()
        backend.sincroniza()
    """

    nome: str
    device: str

    # ----------------------------------------------------------- ciclo

    def prepara(self, conectoma: Conectoma, coef: Coeficientes,
                grupos: np.ndarray | None) -> None:
        """Sobe o conectoma (estatico) e aloca o estado (dinamico)."""
        ...

    def reset(self) -> None:
        """Estado dinamico ao repouso. O conectoma continua carregado."""
        ...

    # -------------------------------------------------------- primitivas

    def escreve_externo(self, externo_mV: np.ndarray | None) -> None:
        """Entrada sensorial deste passo, em mV, por neuronio."""
        ...

    def escreve_forcados(self, forcados: np.ndarray | None) -> None:
        """Mascara de spike forcado: a populacao de entrada Poisson."""
        ...

    def lif(self, passo: int, cursor: int) -> None:
        """Um passo do integrador exato. Le e limpa o slot `cursor` do anel."""
        ...

    def scatter(self, cursor: int) -> None:
        """Propaga os spikes deste passo pro slot `cursor` do anel."""
        ...

    def acumula(self) -> None:
        """Soma a contagem de spike e a atividade por grupo. Instrumentacao."""
        ...

    def sincroniza(self) -> None:
        """Espera tudo que foi enfileirado. So no fim de uma janela."""
        ...

    # ----------------------------------------------------------- leitura

    def le_indices(self, indices: np.ndarray) -> Estado:
        """Estado de um subconjunto. NUNCA do cerebro inteiro."""
        ...

    def le_contagem_grupos(self) -> np.ndarray:
        ...

    def zera_contagem_grupos(self) -> None:
        ...

    def le_estado_completo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(v, g, spike) de todos. So pra validacao -- nao usar no laco."""
        ...

    def le_contagem_total(self) -> np.ndarray:
        """Spikes acumulados por neuronio. So pra validacao."""
        ...

    # ------------------------------------------------------------ info

    def resumo(self) -> dict:
        """Device, memoria, o que couber. Vai pra telemetria e pro profiler."""
        ...


class BackendIndisponivel(RuntimeError):
    """A API existe mas nao da pra usar aqui (driver, device, binding)."""
