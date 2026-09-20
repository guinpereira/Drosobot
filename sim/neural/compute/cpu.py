"""
Backend de referencia: o integrador em fp64, na CPU.

E a VERDADE do projeto. Nao e uma reimplementacao livre -- e a aritmetica de
`fast_lif.Camada.passo()` mais a de `fast_lif.Conexao.empurra()`, generalizada
pra uma populacao recorrente unica em vez de camadas em serie. E `fast_lif` e
validado contra o Brian2.

Nao serve pro conectoma inteiro em tempo interativo, e nem tenta. Serve pra
provar que os outros backends estao certos.

Implementa `IComputeBackend` como qualquer outro: recebe primitivas do
`engine.py` e nao decide ordem de nada. Assim o teste de equivalencia compara
maca com maca -- os dois passam pelo MESMO caminho de chamada.
"""
from __future__ import annotations

import numpy as np

from ..model import Coeficientes, Conectoma, Estado, V_REST, V_RESET, V_TH


class CPUBackend:
    """Referencia em fp64. Device fictício, execucao imediata."""

    nome = "cpu-reference"

    def __init__(self):
        self.device = "cpu / numpy fp64"
        self.c: Conectoma | None = None
        self.coef: Coeficientes | None = None
        self.grupos: np.ndarray | None = None
        self.n_grupos = 1

    # ------------------------------------------------------------- ciclo

    def prepara(self, conectoma: Conectoma, coef: Coeficientes,
                grupos: np.ndarray | None) -> None:
        self.c = conectoma
        self.coef = coef
        self.grupos = grupos
        self.n_grupos = (int(grupos.max()) + 1
                         if grupos is not None and len(grupos) else 1)
        self.reset()

    def reset(self) -> None:
        n = self.c.n
        self.v = np.full(n, V_REST, dtype=np.float64)
        self.g = np.zeros(n, dtype=np.float64)
        self.ref_ate = np.full(n, -1, dtype=np.int64)
        self.spike = np.zeros(n, dtype=np.uint8)
        self.anel = np.zeros((self.coef.atraso_passos, n), dtype=np.float64)
        self.contagem = np.zeros(n, dtype=np.int64)
        self.soma_grupo = np.zeros(self.n_grupos, dtype=np.int64)
        self.externo = None
        self.forcados = None

    # --------------------------------------------------------- primitivas

    def escreve_externo(self, externo_mV: np.ndarray | None) -> None:
        self.externo = (None if externo_mV is None
                        else np.asarray(externo_mV, dtype=np.float64))

    def escreve_forcados(self, forcados) -> None:
        self.forcados = (None if forcados is None
                         else np.asarray(forcados, dtype=bool))

    def lif(self, passo: int, cursor: int) -> None:
        k = self.coef
        chega = self.anel[cursor].copy()
        self.anel[cursor] = 0.0
        if self.externo is not None:
            chega = chega + self.externo

        u = self.v - V_REST
        u_novo = u * k.dec_v + self.g * k.acopla
        self.g = self.g * k.dec_g + chega

        livre = passo >= self.ref_ate
        self.v = np.where(livre, V_REST + u_novo, V_RESET)
        disparou = livre & (self.v > V_TH)
        if self.forcados is not None:
            # a fonte Poisson dispara independente do refratario, como em fast_lif
            disparou = disparou | self.forcados
        self.v = np.where(disparou, V_RESET, self.v)
        self.ref_ate = np.where(disparou, passo + k.ref_passos, self.ref_ate)
        self.spike = disparou.astype(np.uint8)

    def scatter(self, cursor: int) -> None:
        c = self.c
        quem = np.flatnonzero(self.spike)
        if not len(quem):
            return
        destino = self.anel[cursor]
        for i in quem:
            a, b = c.row_offsets[i], c.row_offsets[i + 1]
            if b > a:
                np.add.at(destino, c.targets[a:b], c.weights[a:b])

    def acumula(self) -> None:
        disparou = self.spike.astype(bool)
        self.contagem += disparou
        if self.grupos is not None and self.n_grupos > 1:
            gs = self.grupos[disparou]
            gs = gs[gs >= 0]
            if len(gs):
                np.add.at(self.soma_grupo, gs, 1)

    def sincroniza(self) -> None:
        pass          # execucao imediata: nada a esperar

    # ------------------------------------------------------------ leitura

    def le_indices(self, indices: np.ndarray) -> Estado:
        idx = np.asarray(indices, dtype=np.int32)
        return Estado(indices=idx,
                      v_mV=self.v[idx].astype(np.float32),
                      g_mV=self.g[idx].astype(np.float32),
                      spike=self.spike[idx])

    def le_contagem_grupos(self) -> np.ndarray:
        return self.soma_grupo.copy()

    def zera_contagem_grupos(self) -> None:
        self.soma_grupo[:] = 0

    def le_estado_completo(self):
        return (self.v.astype(np.float32), self.g.astype(np.float32),
                self.spike.copy())

    def le_contagem_total(self) -> np.ndarray:
        return self.contagem.copy()

    def resumo(self) -> dict:
        return {"backend": self.nome, "device": self.device,
                "precision": "fp64", "vram_mib": 0.0}
