"""
NeuralEngine: a ciencia do passo neural. Nao sabe o que e uma GPU.

    from sim.neural import NeuralEngine, carrega_male_cns
    eng = NeuralEngine(carrega_male_cns(), backend="opencl")
    eng.roda(10.0, externo_mV)
    est = eng.le(indices_visiveis)

## O que este arquivo decide, e o backend nao

A ORDEM do passo e a semantica do anel de atraso:

    1. le o que chega agora   = anel[cursor]        (dentro do kernel de LIF)
    2. zera anel[cursor]
    3. integra o LIF          -> spikes deste passo
    4. espalha os spikes em anel[cursor]   (sera lido D passos depois)
    5. cursor = (cursor + 1) % D

Com D = 4 (T_DELAY 1,8 ms / dt 0,5 ms, arredondado), escrever em anel[cursor]
depois de le-lo faz o sinal ser lido de novo D passos adiante. E exatamente a
semantica de `fast_lif.Conexao.empurra()`, e e por isso que os dois concordam
sobre QUANDO o sinal chega.

Trocar OpenCL por D3D12 ou Vulkan nao mexe numa linha daqui. Esse e o criterio.

## WHOLE CONNECTOME SIMULATED nao e WHOLE SENSORIMOTOR MODEL

Este motor simula o conectoma inteiro: 164.451 neuronios e 25,5M arestas
participam da dinamica. Isso NAO quer dizer que o modelo sensorimotor esteja
completo -- a maioria das populacoes nao tem entrada sensorial modelada, e
recebe apenas o que chega pela conectividade.

`resumo()` reporta as duas coisas separadamente, e a interface deve mostra-las
separadas. Confundir as duas seria a afirmacao mais facil e mais errada que este
projeto poderia fazer.
"""
from __future__ import annotations

import numpy as np

from .compute.base import BackendIndisponivel
from .compute.cpu import CPUBackend
from .model import DT_MS, Coeficientes, Conectoma, Estado, verifica_escala


def _constroi_backend(nome: str, preferir_gpu: bool = True):
    """Fabrica. Backend novo entra aqui e em nenhum outro lugar."""
    nome = (nome or "auto").lower()

    if nome in ("cpu", "reference", "cpu-reference"):
        return CPUBackend()

    if nome in ("opencl", "auto"):
        try:
            from .compute.opencl import OpenCLBackend
            return OpenCLBackend(preferir_gpu=preferir_gpu)
        except (BackendIndisponivel, ImportError, Exception) as exc:  # noqa: BLE001
            if nome == "opencl":
                raise
            print(f"[neural] OpenCL indisponivel ({type(exc).__name__}: {exc}); "
                  "caindo pra referencia em CPU")
            return CPUBackend()

    if nome in ("d3d12", "directx", "dx12"):
        from .compute.d3d12 import D3D12Backend
        return D3D12Backend()

    raise ValueError(f"backend desconhecido: {nome!r} "
                     "(cpu | opencl | d3d12 | auto)")


class NeuralEngine:
    """
    Um passo neural do Drosobot, sobre qualquer backend.

    `grupos` mapeia neuronio -> id de populacao (ou -1). E o que permite mostrar
    atividade agregada de quem nao tem morfologia 3D sem trazer 164k estados
    pra CPU.
    """

    def __init__(self, conectoma: Conectoma, backend: str = "auto",
                 grupos: np.ndarray | None = None, dt_ms: float = DT_MS,
                 preferir_gpu: bool = True, nomes_grupos: list[str] | None = None):
        self.c = conectoma
        self.coef = Coeficientes.de(dt_ms)
        self.dt = dt_ms
        self.grupos = grupos
        self.nomes_grupos = nomes_grupos or []
        # Antes de subir nada pra GPU: a escala de ponto fixo tem que caber
        # no int32 DESTE conectoma. Levanta se nao couber.
        self.escala = verifica_escala(conectoma)
        self.backend = _constroi_backend(backend, preferir_gpu)
        self.backend.prepara(conectoma, self.coef, grupos)
        self.cursor = 0
        self._passo = 0
        self.instrumentar = True

    # -------------------------------------------------------------- passo

    def passo(self, externo_mV=None, forcados=None) -> None:
        self.backend.escreve_externo(externo_mV)
        self.backend.escreve_forcados(forcados)
        self._um_passo()

    def _um_passo(self) -> None:
        b = self.backend
        b.lif(self._passo, self.cursor)
        b.scatter(self.cursor)
        if self.instrumentar:
            b.acumula()
        self.cursor = (self.cursor + 1) % self.coef.atraso_passos
        self._passo += 1

    def roda_poisson(self, duracao_ms: float, taxas_hz, rng,
                     indices=None) -> int:
        """
        Janela com a populacao de entrada disparando como POISSON.

        E como o modelo define a entrada sensorial: probabilidade de disparo por
        passo = taxa_hz * dt_s, e cada spike entrega o peso sinaptico cheio. A
        conversao taxa -> probabilidade e decisao de MODELO, e por isso mora aqui
        no host, nao no kernel.

        Injetar corrente continua equivalente NAO e a mesma coisa: e mais fraco,
        e foi o que impediu o primeiro laco fechado de disparar o Giant Fiber.
        """
        n = int(round(duracao_ms / self.dt))
        self.backend.escreve_externo(None)

        # `indices` diz QUEM pode disparar. Sem isso sorteariamos um numero
        # aleatorio por neuronio por passo -- 164.451 x 20 = 3,3 milhoes de
        # sorteios por janela de 10 ms, na CPU, pra 311 sensores. Era o que
        # fazia o passo neural do CNS inteiro custar 2,8 ms em vez de ~1 ms.
        if indices is None:
            p = np.asarray(taxas_hz, dtype=np.float64) * (self.dt / 1000.0)
            for _ in range(n):
                self.backend.escreve_forcados(
                    (rng.random(self.c.n) < p).astype(np.uint8))
                self._um_passo()
        else:
            idx = np.asarray(indices, dtype=np.int64)
            p = np.asarray(taxas_hz, dtype=np.float64)[idx] * (self.dt / 1000.0)
            k = len(idx)
            # Caminho ESPARSO quando o backend souber: manda os k valores em
            # vez da mascara inteira. Sao 311 bytes no lugar de 164.451, e sem
            # bloquear o host. O conteudo da mascara no device e identico --
            # so estes indices sao escritos, e o resto ja era zero.
            if self._esparso_pronto(idx):
                for _ in range(n):
                    self.backend.escreve_forcados_esparso(rng.random(k) < p)
                    self._um_passo()
            else:
                mascara = np.zeros(self.c.n, dtype=np.uint8)
                for _ in range(n):
                    mascara[idx] = (rng.random(k) < p).astype(np.uint8)
                    self.backend.escreve_forcados(mascara)
                    self._um_passo()
        self.backend.sincroniza()
        return n

    def _esparso_pronto(self, idx) -> bool:
        """
        O backend aceita o caminho esparso, e ja sabe destes indices?

        Os indices sao fixos durante a corrida, entao sobem uma vez. Se
        mudarem (outro experimento, outro escopo), sobem de novo -- comparar e
        mais barato que reenviar.
        """
        prep = getattr(self.backend, "prepara_forcados_esparsos", None)
        if prep is None:
            return False
        anterior = getattr(self, "_idx_esparso", None)
        if anterior is not None and len(anterior) == len(idx)                 and np.array_equal(anterior, idx):
            return True
        if not prep(idx):
            self._idx_esparso = None
            return False
        self._idx_esparso = np.array(idx, copy=True)
        return True

    def roda(self, duracao_ms: float,
             externo_mV: np.ndarray | None = None) -> int:
        """
        Avanca uma JANELA inteira sem sincronizar no meio.

        Sincronizar por passo seria uma ida e volta GPU->CPU a cada 0,5 ms de
        mosca. A janela tipica do Drosobot e 10 ms = 20 passos, e o laco so
        espera no fim dela.
        """
        n = int(round(duracao_ms / self.dt))
        self.backend.escreve_externo(externo_mV)
        self.backend.escreve_forcados(None)
        for _ in range(n):
            self._um_passo()
        self.backend.sincroniza()
        return n

    def reset(self) -> None:
        self.backend.reset()
        self.cursor = 0
        self._passo = 0

    @property
    def passo_atual(self) -> int:
        return self._passo

    @property
    def tempo_ms(self) -> float:
        return self._passo * self.dt

    # ------------------------------------------------------------ leitura

    def le(self, indices: np.ndarray) -> Estado:
        """Estado de um subconjunto. Nunca do cerebro inteiro."""
        return self.backend.le_indices(indices)

    def atividade_por_grupo(self, zerar: bool = True) -> np.ndarray:
        soma = self.backend.le_contagem_grupos()
        if zerar:
            self.backend.zera_contagem_grupos()
        return soma

    def indices_de(self, body_ids) -> np.ndarray:
        """bodyId real -> indice denso, descartando quem nao esta no grafo."""
        idx = self.c.indice_de(body_ids)
        return idx[idx >= 0]

    # ------------------------------------------------------------- resumo

    def resumo(self) -> dict:
        """
        Pro profiler, pra telemetria e pra interface.

        Separa deliberadamente o que esta SIMULADO do que tem entrada sensorial
        modelada. Ver o cabecalho deste modulo.
        """
        r = dict(self.backend.resumo())
        r.update({
            "connectome": self.c.nome,
            "neurons_simulated": self.c.n,
            "edges_simulated": self.c.e,
            "dt_ms": self.dt,
            "delay_steps": self.coef.atraso_passos,
            "refractory_steps": self.coef.ref_passos,
            "scope": "whole_connectome_simulated",
            "note": ("conectoma inteiro simulado; NAO e um modelo sensorimotor "
                     "completo -- a maioria das populacoes nao tem entrada "
                     "sensorial modelada"),
        })
        return r
