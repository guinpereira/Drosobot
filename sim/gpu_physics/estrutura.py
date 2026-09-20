r"""
A arvore do corpo, reorganizada para ser percorrida em paralelo.

O MuJoCo percorre os corpos em ordem de indice, e isso basta na CPU porque o
compilador garante `parentid < id`: um laco de 0 a nbody ja visita todo pai
antes de todo filho. Na GPU esse laco e uma fila de 72 passos seriais com uma
faixa ativa -- 71 threads paradas.

O que substitui: **niveis**. Um corpo esta no nivel `d` se a distancia dele ate
a raiz e `d`. Corpos do mesmo nivel nao dependem uns dos outros, entao o nivel
inteiro anda de uma vez. A cinematica direta vira `profundidade` etapas em vez
de `nbody`, e a profundidade da mosca e uma ordem de grandeza menor que o
numero de corpos.

    nbody = 72        profundidade = 9

Passo para tras (inercia composta, RNE): mesmos niveis, ordem invertida. Ai
aparece um problema que a ida nao tem -- dois filhos escrevem no MESMO pai. No
torax sao oito (seis coxas, cabeca, abdome). A saida aqui nao e atomico de
ponto flutuante (OpenCL 1.2 nao tem): e inverter quem faz a conta. Uma thread
por PAI, varrendo os filhos dele naquele nivel. Sem corrida, sem atomico, e o
desbalanceamento e irrelevante num corpo com oito filhos no maximo.

Tudo aqui e calculado UMA vez, no setup. Nada disto roda por passo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arvore:
    """Agenda de percurso da arvore de corpos, pronta pra virar buffer."""

    niveis: list[np.ndarray]          # niveis[d] = ids dos corpos no nivel d
    nivel_de: np.ndarray             # (nbody,) nivel de cada corpo
    pais_por_nivel: list[np.ndarray]  # pais distintos que recebem do nivel d
    filhos_adr: np.ndarray           # (nbody,) inicio em `filhos`
    filhos_num: np.ndarray           # (nbody,) quantos filhos
    filhos: np.ndarray               # lista achatada de filhos, agrupada por pai
    cadeia_adr: np.ndarray           # (nv,) inicio em `cadeia`
    cadeia_num: np.ndarray           # (nv,) tamanho da cadeia do dof ate a raiz
    cadeia: np.ndarray               # dof, pai(dof), pai(pai(dof)), ... achatado

    @property
    def profundidade(self) -> int:
        return len(self.niveis)

    def resumo(self) -> dict:
        return {
            "nbody": int(self.nivel_de.size),
            "profundidade": self.profundidade,
            "corpos_por_nivel": [int(n.size) for n in self.niveis],
            "max_filhos": int(self.filhos_num.max()) if self.filhos_num.size else 0,
            "max_cadeia_dof": int(self.cadeia_num.max()) if self.cadeia_num.size else 0,
        }


def _niveis(parentid: np.ndarray) -> np.ndarray:
    """Profundidade de cada corpo. O mundo (0) e o nivel 0 e e pai de si mesmo."""
    nivel = np.zeros(parentid.size, dtype=np.int32)
    for i in range(1, parentid.size):
        nivel[i] = nivel[parentid[i]] + 1
    return nivel


def _filhos(parentid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lista de filhos por pai, achatada e agrupada -- sem lista de listas."""
    n = parentid.size
    num = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        num[parentid[i]] += 1
    adr = np.zeros(n, dtype=np.int32)
    adr[1:] = np.cumsum(num)[:-1]
    cursor = adr.copy()
    chapa = np.zeros(int(num.sum()), dtype=np.int32)
    for i in range(1, n):
        p = int(parentid[i])
        chapa[cursor[p]] = i
        cursor[p] += 1
    return adr, num, chapa


def _cadeias(dof_parentid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Para cada dof, a cadeia dele ate a raiz: `[i, pai, avo, ...]`.

    E o padrao de esparsidade de UMA linha da matriz de massa: a entrada
    `M[i][j]` so e diferente de zero se `j` esta na cadeia de `i`. O MuJoCo
    guarda isso implicitamente em `dof_Madr` e caminha o ponteiro; aqui a
    cadeia fica explicita para que cada thread leia a dela sem caminhar.
    """
    nv = dof_parentid.size
    cadeias = []
    for i in range(nv):
        c, j = [], i
        while j >= 0:
            c.append(j)
            j = int(dof_parentid[j])
        cadeias.append(np.asarray(c, dtype=np.int32))
    num = np.asarray([c.size for c in cadeias], dtype=np.int32)
    adr = np.zeros(nv, dtype=np.int32)
    if nv:
        adr[1:] = np.cumsum(num)[:-1]
    return adr, num, (np.concatenate(cadeias) if nv else np.zeros(0, np.int32))


def constroi(mod) -> Arvore:
    """Agenda de percurso a partir de um `ModeloGPU`."""
    parentid = np.asarray(mod.arrays["body_parentid"], dtype=np.int32)
    nivel = _niveis(parentid)
    niveis = [np.flatnonzero(nivel == d).astype(np.int32)
              for d in range(int(nivel.max()) + 1)]
    pais = [np.unique(parentid[n]).astype(np.int32) if d else np.zeros(0, np.int32)
            for d, n in enumerate(niveis)]
    adr, num, chapa = _filhos(parentid)
    c_adr, c_num, cadeia = _cadeias(
        np.asarray(mod.arrays["dof_parentid"], dtype=np.int32))
    return Arvore(niveis=niveis, nivel_de=nivel, pais_por_nivel=pais,
                  filhos_adr=adr, filhos_num=num, filhos=chapa,
                  cadeia_adr=c_adr, cadeia_num=c_num, cadeia=cadeia)
