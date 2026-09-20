"""
O MODELO. Nada aqui sabe o que e uma GPU.

Este modulo carrega a ciencia do backend neural do Drosobot:

    parametros de Shiu et al. 2024 (Nature 634:210)
    coeficientes do integrador exato
    representacao do conectoma (CSR, peso ja com sinal)
    o que e um estado lido

Fica separado de `compute/` de proposito. O criterio esta em
`compute/base.py`: se remover o OpenCL exigisse mexer neste arquivo, o
acoplamento estaria errado.

Os parametros sao a copia operacional dos de `sim/connectome_model.py`, e
`tests/test_neural_backend.py::test_parametros_batem_com_o_modelo` falha se os
dois divergirem -- a duplicacao e deliberada e vigiada, nao acidental.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

# Parametros de Shiu et al. 2024 (Nature 634:210), em mV e ms.
# Importados de connectome_model quando ele esta disponivel; os literais abaixo
# sao o fallback pros scripts que rodam sem o Brian2 carregado, e o teste
# `test_parametros_nao_derivaram` garante que os dois nao divergem.
V_REST = -52.0
V_RESET = -52.0
V_TH = -45.0
TAU_M = 20.0
TAU_S = 5.0
T_REF = 2.2
T_DELAY = 1.8
DT_MS = 0.5
W_SYN_MV = 0.275

# Ponto fixo pra acumulacao atomica na GPU: OpenCL garante atomic_add em int32,
# nao em float, e HLSL SM 5.0 tambem nao tem -- entao a escala tem que servir aos
# dois backends.
#
# A escolha e um compromisso MEDIDO no conectoma real:
#
#   passo de quantizacao = 1/ESCALA mV
#   pior caso de acumulacao = soma de todas as arestas que entram num alvo,
#                             medida em 32.972 mV no Male CNS
#
#   escala    passo      pico            int32 max
#     1024   0,98 uV   3,4e7           2,1e9   seguro, mas o erro por aresta
#                                              (~1 uV) acumula sobre ~150
#                                              convergencias e chega a ~0,01 mV
#    16384   0,06 uV   5,4e8           2,1e9   seguro, com 4x de folga
#    65536   0,015 uV  2,2e9           2,1e9   ESTOURA
#
# 1024 foi o primeiro valor e produzia divergencia de ~0,01 mV contra a
# referencia em fp64 -- o bastante pra trocar a decisao de limiar de alguns
# neuronios. 16384 derruba isso 16x e ainda deixa folga de 4x contra overflow.
ESCALA = 16384.0

# int32 e o que os atomicos da GPU somam. O teto e este, nao o do Python.
INT32_MAX = 2_147_483_647


def pico_convergente_mV(c) -> float:
    """
    Maior soma de |peso| que pode chegar num alvo num unico passo.

    E o pior caso REAL do conectoma carregado, nao o numero da tabela acima: se
    um dia o dataset mudar -- mais arestas, pesos maiores, outro filtro -- a
    conta que justifica a escala muda junto, e ninguem vai lembrar de refazer a
    tabela. Medir custa uma passada de bincount sobre as arestas.

    Pior caso de verdade seria todos os pre-sinapticos de um alvo disparando no
    mesmo passo. Nao acontece na pratica, e exatamente por isso serve de teto.
    """
    import numpy as np

    if len(c.targets) == 0:
        return 0.0
    soma = np.bincount(c.targets.astype(np.int64),
                       weights=np.abs(c.weights.astype(np.float64)),
                       minlength=c.n)
    return float(soma.max())


def verifica_escala(c, escala: float = ESCALA) -> dict:
    """
    Confere a escala de ponto fixo contra o conectoma que ESTA carregado.

    Levanta se estourar. Nao e paranoia: overflow de int32 num atomico nao da
    erro, da a volta -- uma inibicao enorme vira excitacao enorme, e o
    resultado sai plausivel e errado. E o tipo de bug que so aparece como
    "comportamento estranho" meses depois.
    """
    pico = pico_convergente_mV(c)
    usado = pico * escala
    folga = INT32_MAX / usado if usado > 0 else float("inf")
    if usado >= INT32_MAX:
        raise OverflowError(
            f"escala {escala:.0f} estoura int32 neste conectoma: pico "
            f"convergente {pico:.1f} mV x {escala:.0f} = {usado:.3e} >= "
            f"{INT32_MAX:.3e}. Baixar a escala perde precisao no limiar; "
            "trocar o acumulador por int64 custa banda. Decidir, nao ignorar.")
    return {"pico_convergente_mV": pico, "escala": escala,
            "pico_em_fixo": usado, "folga_x": folga}


@dataclass
class Coeficientes:
    """Constantes do passo exato. Calculadas em fp64, uma vez."""
    dec_v: float
    dec_g: float
    acopla: float
    ref_passos: int
    atraso_passos: int

    @classmethod
    def de(cls, dt_ms: float = DT_MS) -> "Coeficientes":
        dec_v = math.exp(-dt_ms / TAU_M)
        dec_g = math.exp(-dt_ms / TAU_S)
        a = 1.0 / TAU_M - 1.0 / TAU_S
        return cls(
            dec_v=dec_v,
            dec_g=dec_g,
            acopla=(dec_g - dec_v) / (TAU_M * a),
            # inteiro, como em fast_lif: 2.2/0.5 = 4.4 -> 4 passos. O valor foi
            # calibrado contra o Brian2; nao mexer sem refazer aquela comparacao.
            ref_passos=int(round(T_REF / dt_ms)),
            # 1.8/0.5 = 3.6 -> 4 passos, o mesmo arredondamento de fast_lif.Conexao
            atraso_passos=max(1, int(round(T_DELAY / dt_ms))),
        )


@dataclass
class Conectoma:
    """
    Grafo estatico. Sobe uma vez pra GPU e nao muda durante a corrida.

    `weights` ja vem COM SINAL (regra de Dale sobre o neurotransmissor do
    pre-sinaptico) e ja em mV. Manter o sinal embutido evita um array separado e
    nao muda a matematica: `w_mV = sinapses * sinal * W_SYN`.
    """
    row_offsets: np.ndarray     # int32 [N+1]
    targets: np.ndarray         # int32 [E]
    weights: np.ndarray         # float32 [E], com sinal, em mV
    body_ids: np.ndarray        # int64 [N], pra casar com o conectoma real
    nome: str = "conectoma"

    @property
    def n(self) -> int:
        return len(self.row_offsets) - 1

    @property
    def e(self) -> int:
        return len(self.targets)

    def indice_de(self, body_ids) -> np.ndarray:
        """bodyId real -> indice denso. Fora do grafo vira -1."""
        alvo = np.asarray(body_ids, dtype=np.int64)
        pos = np.searchsorted(self.body_ids, alvo)
        pos = np.clip(pos, 0, self.n - 1)
        ok = self.body_ids[pos] == alvo
        return np.where(ok, pos, -1).astype(np.int32)

    def resumo(self) -> dict:
        w = self.weights
        return {
            "name": self.nome,
            "neurons": self.n,
            "edges": self.e,
            "excitatory_edges": int((w > 0).sum()),
            "inhibitory_edges": int((w < 0).sum()),
            "silent_edges": int((w == 0).sum()),
            "static_bytes": int(self.row_offsets.nbytes + self.targets.nbytes
                                + self.weights.nbytes),
        }


@dataclass
class Estado:
    """Estado dinamico lido de volta. Sempre de um SUBCONJUNTO."""
    indices: np.ndarray
    v_mV: np.ndarray
    g_mV: np.ndarray
    spike: np.ndarray
