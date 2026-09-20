r"""
Cinematica direta residente na GPU: buffers, despacho e leitura.

O que este arquivo faz e subir o modelo UMA vez, manter o estado no device e
oferecer duas formas de rodar a mesma conta:

    passo_por_etapa()   um dispatch por nivel da arvore. Lento de proposito:
                        serve pro teste comparar cada etapa com o `mjData`.
    passo_residente()   a arvore inteira num dispatch so. E o caminho medido.

A conta esta nos kernels (`kernels/cinematica.cl`), nao aqui. Aqui so existe
ligacao de argumento -- e isso e proposital: se algum dia a fisica aparecer
neste arquivo, ela vai estar em Python, no host, rodando na CPU, que e
exatamente o que o projeto inteiro esta tentando sair de fazer.

## O que fica na GPU

    qpos, mocap_pos, mocap_quat          entrada (escrita por passo)
    xpos, xquat, xmat, xanchor, xaxis    quadros dos corpos e das juntas
    xipos, ximat                         quadros inerciais
    geom_xpos, geom_xmat                 quadros dos geoms

Nada disso volta pra CPU no laco. `le()` existe pra validacao e pra telemetria,
que roda a 30 Hz, nao a 10.000 Hz.
"""
from __future__ import annotations

import numpy as np

from . import estrutura
from .device import Device

ARQUIVO = "cinematica.cl"

# Arrays do modelo que o kernel de cinematica precisa, com o tipo que ele
# espera. `int32` para tudo que e indice; `real` para o resto. `body_sameframe`
# e `geom_sameframe` chegam como uint8 do MuJoCo e sobem como int32 porque o
# OpenCL nao tem bool em assinatura de kernel.
INTEIROS = ("body_parentid", "body_jntadr", "body_jntnum", "body_mocapid",
            "body_sameframe", "jnt_type", "jnt_qposadr", "geom_bodyid",
            "geom_sameframe")
REAIS = ("body_pos", "body_quat", "body_ipos", "body_iquat", "jnt_axis",
         "jnt_pos", "qpos0", "geom_pos", "geom_quat")

SAIDAS = {
    "xpos": 3, "xquat": 4, "xmat": 9, "xipos": 3, "ximat": 9,
}
SAIDAS_JNT = {"xanchor": 3, "xaxis": 3}
SAIDAS_GEOM = {"geom_xpos": 3, "geom_xmat": 9}


class CinematicaGPU:
    """Modelo residente + estado residente. Um objeto por corrida."""

    def __init__(self, mod, dev: Device | None = None, fp64: bool = True,
                 grupo: int = 256):
        self.mod = mod
        self.dev = dev or Device()
        self.fp64 = bool(fp64)
        self.real = np.float64 if self.fp64 else np.float32
        self.arvore = estrutura.constroi(mod)
        self.grupo = min(int(grupo), int(self.dev.dev.max_work_group_size))

        d = self.dev
        nbody = mod.dims["nbody"]
        njnt = mod.dims["njnt"]
        ngeom = mod.dims["ngeom"]
        nmocap = max(1, mod.dims.get("nmocap", 0))

        self.b = {}
        for nome in INTEIROS:
            self.b[nome] = d.sobe(np.asarray(mod.arrays[nome]).ravel().astype(np.int32))
        for nome in REAIS:
            self.b[nome] = d.sobe(np.asarray(mod.arrays[nome]).ravel().astype(self.real))

        # agenda da arvore
        arv = self.arvore
        self.profundidade = arv.profundidade
        adr, num, chapa = [], [], []
        for n in arv.niveis:
            adr.append(len(chapa))
            num.append(int(n.size))
            chapa.extend(int(x) for x in n)
        self.nivel_adr_np = np.asarray(adr, dtype=np.int32)
        self.nivel_num_np = np.asarray(num, dtype=np.int32)
        self.b["nivel_adr"] = d.sobe(self.nivel_adr_np)
        self.b["nivel_num"] = d.sobe(self.nivel_num_np)
        self.b["nivel_corpos"] = d.sobe(np.asarray(chapa, dtype=np.int32))

        # estado de entrada
        self.b["qpos"] = d.vazio(mod.dims["nq"], self.real)
        self.b["mocap_pos"] = d.vazio(3 * nmocap, self.real)
        self.b["mocap_quat"] = d.vazio(4 * nmocap, self.real)

        # saidas
        self.tam = {}
        for nome, k in SAIDAS.items():
            self.tam[nome] = k * nbody
        for nome, k in SAIDAS_JNT.items():
            self.tam[nome] = k * njnt
        for nome, k in SAIDAS_GEOM.items():
            self.tam[nome] = k * ngeom
        for nome, n in self.tam.items():
            self.b[nome] = d.vazio(n, self.real)

        self._mundo()
        self.k_nivel = d.kernel(ARQUIVO, "fk_nivel", self.fp64)
        self.k_quadros = d.kernel(ARQUIVO, "fk_quadros", self.fp64)
        self.k_geoms = d.kernel(ARQUIVO, "fk_geoms", self.fp64)
        self.k_completa = d.kernel(ARQUIVO, "fk_completa", self.fp64)
        # a versao com estado em __local precisa das dimensoes em tempo de
        # compilacao; o programa e compilado uma vez por modelo
        self._defines = (("NBODY", nbody), ("NJNT", njnt), ("NGEOM", ngeom))
        self.k_lds = d.kernel(ARQUIVO, "fk_lds", self.fp64, self._defines)
        self._args_lds_fixos = None
        d.espera()

    def _mundo(self) -> None:
        """
        Corpo 0 e o mundo: identidade, escrita uma vez.

        O kernel comeca no nivel 1 porque o mundo nao tem pai nem junta. Deixar
        o zero pro kernel custaria um `if` em toda thread para um corpo so.
        """
        nbody = self.mod.dims["nbody"]
        ident = np.zeros(9 * nbody, dtype=self.real)
        ident[0] = ident[4] = ident[8] = 1
        self.dev.escreve(self.b["xmat"], ident)
        self.dev.escreve(self.b["ximat"], ident)
        q = np.zeros(4 * nbody, dtype=self.real)
        q[0] = 1
        self.dev.escreve(self.b["xquat"], q)

    # ------------------------------------------------------------- entrada

    def escreve_estado(self, qpos, mocap_pos=None, mocap_quat=None) -> None:
        self.dev.escreve(self.b["qpos"], np.asarray(qpos, dtype=self.real))
        if mocap_pos is not None and np.size(mocap_pos):
            self.dev.escreve(self.b["mocap_pos"],
                             np.asarray(mocap_pos, dtype=self.real).ravel())
        if mocap_quat is not None and np.size(mocap_quat):
            self.dev.escreve(self.b["mocap_quat"],
                             np.asarray(mocap_quat, dtype=self.real).ravel())

    # ------------------------------------------------------------- execucao

    def _args_corpo(self, nivel_adr: int, nivel_num: int) -> tuple:
        b = self.b
        return (b["nivel_corpos"], np.int32(nivel_adr), np.int32(nivel_num),
                b["body_parentid"], b["body_jntadr"], b["body_jntnum"],
                b["body_mocapid"], b["body_pos"], b["body_quat"],
                b["jnt_type"], b["jnt_qposadr"], b["jnt_axis"], b["jnt_pos"],
                b["qpos0"], b["qpos"], b["mocap_pos"], b["mocap_quat"],
                b["xpos"], b["xquat"], b["xmat"], b["xanchor"], b["xaxis"])

    def passo_por_etapa(self) -> None:
        """Um dispatch por nivel. Para o teste, nao para o laco quente."""
        d, b = self.dev, self.b
        for nivel in range(1, self.profundidade):
            adr = int(self.nivel_adr_np[nivel])
            num = int(self.nivel_num_np[nivel])
            if not num:
                continue
            d.roda(self.k_nivel, num, None, self._args_corpo(adr, num))
            d.espera()
        nbody = self.mod.dims["nbody"]
        d.roda(self.k_quadros, nbody, None,
               (np.int32(nbody), b["body_ipos"], b["body_iquat"],
                b["body_sameframe"], b["xpos"], b["xquat"], b["xmat"],
                b["xipos"], b["ximat"]))
        d.espera()
        ngeom = self.mod.dims["ngeom"]
        d.roda(self.k_geoms, ngeom, None,
               (np.int32(ngeom), b["geom_bodyid"], b["geom_pos"],
                b["geom_quat"], b["geom_sameframe"], b["xpos"], b["xquat"],
                b["xmat"], b["xipos"], b["ximat"], b["geom_xpos"],
                b["geom_xmat"]))
        d.espera()

    def _args_completa(self, repeticoes: int) -> tuple:
        b = self.b
        return (np.int32(self.mod.dims["nbody"]), np.int32(self.mod.dims["ngeom"]),
                np.int32(self.profundidade),
                b["nivel_adr"], b["nivel_num"], b["nivel_corpos"],
                b["body_parentid"], b["body_jntadr"], b["body_jntnum"],
                b["body_mocapid"], b["body_pos"], b["body_quat"],
                b["body_ipos"], b["body_iquat"], b["body_sameframe"],
                b["jnt_type"], b["jnt_qposadr"], b["jnt_axis"], b["jnt_pos"],
                b["qpos0"], b["geom_bodyid"], b["geom_pos"], b["geom_quat"],
                b["geom_sameframe"], b["qpos"], b["mocap_pos"], b["mocap_quat"],
                b["xpos"], b["xquat"], b["xmat"], b["xanchor"], b["xaxis"],
                b["xipos"], b["ximat"], b["geom_xpos"], b["geom_xmat"],
                np.int32(repeticoes))

    def passo_residente(self, repeticoes: int = 1, esperar: bool = True) -> None:
        """A arvore inteira num dispatch, estado em global. Para comparacao."""
        self.dev.roda(self.k_completa, self.grupo, self.grupo,
                      self._args_completa(repeticoes))
        if esperar:
            self.dev.espera()

    def _args_lds(self, repeticoes: int) -> tuple:
        b = self.b
        return (np.int32(self.profundidade),
                b["nivel_adr"], b["nivel_num"], b["nivel_corpos"],
                b["body_parentid"], b["body_jntadr"], b["body_jntnum"],
                b["body_mocapid"], b["body_pos"], b["body_quat"],
                b["body_ipos"], b["body_iquat"], b["body_sameframe"],
                b["jnt_type"], b["jnt_qposadr"], b["jnt_axis"], b["jnt_pos"],
                b["qpos0"], b["geom_bodyid"], b["geom_pos"], b["geom_quat"],
                b["geom_sameframe"], b["qpos"], b["mocap_pos"], b["mocap_quat"],
                b["xpos"], b["xquat"], b["xmat"], b["xanchor"], b["xaxis"],
                b["xipos"], b["ximat"], b["geom_xpos"], b["geom_xmat"],
                np.int32(repeticoes))

    def passo(self, repeticoes: int = 1, esperar: bool = True) -> None:
        """
        O caminho do laco quente: um dispatch, quadros dos corpos em __local.

        Os argumentos sao ligados UMA vez. `set_args` com 35 argumentos custa
        microssegundos por chamada, e a 10.000 passos por segundo simulado isso
        apareceria no numero que estamos tentando medir.
        """
        if self._args_lds_fixos != repeticoes:
            self.k_lds.set_args(*self._args_lds(repeticoes))
            self._args_lds_fixos = repeticoes
        self.dev.roda(self.k_lds, self.grupo, self.grupo)
        if esperar:
            self.dev.espera()

    # -------------------------------------------------------------- leitura

    def le(self, nome: str) -> np.ndarray:
        self.dev.espera()
        return self.dev.le(self.b[nome], self.tam[nome], self.real)
