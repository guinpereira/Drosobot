r"""
O passo de fisica residente na GPU: buffers, estagios e o laco.

    motor = MotorFisicoGPU(modelo_compilado, fp64=True)
    motor.escreve_estado(qpos, qvel, ctrl, mocap_pos, mocap_quat)
    motor.passo()
    qpos = motor.le("qpos")

A conta esta nos kernels (`kernels/cinematica.cl` + `kernels/dinamica.cl`).
Aqui so existe ligacao de argumento e a ORDEM das etapas -- que e a mesma do
`mj_step`, porque cada etapa foi validada contra o campo correspondente do
`mjData` antes de a seguinte ser escrita.

## Por que cada etapa e um dispatch, por enquanto

Porque um bug de fisica num kernel fundido de trinta etapas e indepuravel. A
forma por etapa existe para o teste poder parar em `cinert`, em `qM`, em
`qfrc_bias` e dizer QUAL delas divergiu. A forma residente -- tudo num dispatch
so -- e o caminho do laco quente e vem depois, sobre etapas que ja sao sabidas
corretas.

A medida que justifica isso esta em `docs/GPU_PHYSICS.md`: um despacho
sincrono custa 84 us e uma barreira dentro do kernel custa 0,035 us. A forma por
etapa NAO e um desenho de producao, e nao e apresentada como um.

## O que este passo ainda nao tem

Contato, restricao e solver. `passo()` produz a dinamica SEM restricao: com a
mosca no ar ela e a fisica inteira; com a mosca no chao, falta a forca que a
impede de atravessar. E por isso que `physics.cria('drosobot-gpu')` continua
recusando montar.
"""
from __future__ import annotations

import numpy as np

from . import estrutura
from .device import Device

ARQUIVOS = ("cinematica.cl", "dinamica.cl", "colisao.cl")

# Arrays do modelo, por tipo de destino no device.
INTEIROS = (
    "body_parentid", "body_rootid", "body_jntadr", "body_jntnum", "body_mocapid",
    "body_sameframe", "body_dofadr", "body_dofnum", "body_geomadr",
    "body_geomnum",
    "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid",
    "dof_bodyid", "dof_jntid", "dof_parentid", "dof_Madr",
    "M_rownnz", "M_rowadr", "M_colind",
    "geom_bodyid", "geom_sameframe", "geom_type", "geom_dataid",
    "pair_geom1", "pair_geom2", "pair_dim",
    "mesh_vertadr", "mesh_vertnum", "mesh_graphadr", "mesh_graph",
    "actuator_trntype", "actuator_trnid", "actuator_biastype",
    "actuator_ctrllimited", "actuator_forcelimited",
)
REAIS = (
    "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass",
    "body_inertia", "body_subtreemass",
    "jnt_axis", "jnt_pos", "jnt_stiffness",
    "dof_armature", "dof_damping", "dof_M0",
    "qpos0", "qpos_spring",
    "geom_pos", "geom_quat", "geom_size",
    "actuator_gainprm", "actuator_biasprm", "actuator_gear",
    "actuator_ctrlrange", "actuator_forcerange",
    "geom_rbound", "pair_margin", "pair_friction", "pair_solref",
    "pair_solimp", "pair_gap",
)

# `mesh_vert` e float32 no mjModel e fica float32 no device de proposito: a
# funcao de suporte do MuJoCo tambem le float e promove na conta, e converter
# aqui mudaria o arredondamento do argmax.
FLOAT32 = ("mesh_vert",)

# `maxplanemesh` do MuJoCo 3.9.0: o ponto de suporte mais, no maximo, dois
# vizinhos dele no grafo de hull.
MAX_CON_POR_PAR = 3


class MotorFisicoGPU:
    """Modelo e estado residentes. Um objeto por corrida."""

    def __init__(self, mod, dev: Device | None = None, fp64: bool = True,
                 grupo: int = 256):
        self.mod = mod
        self.dev = dev or Device()
        self.fp64 = bool(fp64)
        self.real = np.float64 if self.fp64 else np.float32
        self.arvore = estrutura.constroi(mod)
        self.grupo = min(int(grupo), int(self.dev.dev.max_work_group_size))

        d = self.dev
        dims = mod.dims
        self.nq, self.nv = dims["nq"], dims["nv"]
        self.nbody, self.njnt = dims["nbody"], dims["njnt"]
        self.ngeom, self.nu = dims["ngeom"], dims["nu"]
        self.nC = dims.get("nC") or dims["nM"]
        nmocap = max(1, dims.get("nmocap", 0))

        self.b = {}
        for nome in INTEIROS:
            self.b[nome] = d.sobe(
                np.asarray(mod.arrays[nome]).ravel().astype(np.int32))
        for nome in REAIS:
            self.b[nome] = d.sobe(
                np.asarray(mod.arrays[nome]).ravel().astype(self.real))
        for nome in FLOAT32:
            self.b[nome] = d.sobe(
                np.asarray(mod.arrays[nome]).ravel().astype(np.float32))
        self.npair = int(dims["npair"])
        # indice do vertice no GRAFO de hull, por vertice global da malha.
        # O 3.9.0 varre os vizinhos do vertice de suporte pelo grafo, e o
        # `obj.meshindex` dele e um indice local; aqui a traducao e uma tabela
        # construida uma vez, no setup.
        self.b["vert_local"] = d.sobe(self._vert_local(mod))

        self._agenda()

        # --- estado de entrada -------------------------------------------
        self.tam = {}
        for nome, n in (("qpos", self.nq), ("qvel", self.nv), ("ctrl", self.nu),
                        ("mocap_pos", 3 * nmocap), ("mocap_quat", 4 * nmocap),
                        ("qfrc_applied", self.nv)):
            self.tam[nome] = n
            self.b[nome] = d.vazio(n, self.real)

        # --- derivados ----------------------------------------------------
        derivados = {
            "xpos": 3 * self.nbody, "xquat": 4 * self.nbody,
            "xmat": 9 * self.nbody, "xipos": 3 * self.nbody,
            "ximat": 9 * self.nbody,
            "xanchor": 3 * self.njnt, "xaxis": 3 * self.njnt,
            "geom_xpos": 3 * self.ngeom, "geom_xmat": 9 * self.ngeom,
            "subtree_com": 3 * self.nbody, "cinert": 10 * self.nbody,
            "crb": 10 * self.nbody, "cdof": 6 * self.nv,
            "cvel": 6 * self.nbody, "cdof_dot": 6 * self.nv,
            "cacc": 6 * self.nbody, "cfrc_body": 6 * self.nbody,
            "M": self.nC, "qLD": self.nC, "qLDiagInv": self.nv,
            "qH": self.nC, "qHDiagInv": self.nv,
            "qfrc_passive": self.nv, "qfrc_bias": self.nv,
            "qfrc_actuator": self.nv, "qfrc_smooth": self.nv,
            "qfrc_constraint": self.nv,
            "qacc_smooth": self.nv, "qacc": self.nv, "rhs": self.nv,
            "actuator_length": self.nu, "actuator_velocity": self.nu,
            "actuator_force": self.nu,
            # colisao: ranhuras fixas por par, depois compactadas em ordem
            "sup_dist": max(1, self.npair),
            "con_dist_bruto": max(1, MAX_CON_POR_PAR * self.npair),
            "con_pos_bruto": max(1, 3 * MAX_CON_POR_PAR * self.npair),
            "con_normal_bruto": max(1, 3 * MAX_CON_POR_PAR * self.npair),
            "con_dist": max(1, MAX_CON_POR_PAR * self.npair),
            "con_pos": max(1, 3 * MAX_CON_POR_PAR * self.npair),
            "con_normal": max(1, 3 * MAX_CON_POR_PAR * self.npair),
        }
        for nome, n in derivados.items():
            self.tam[nome] = n
            self.b[nome] = d.vazio(n, self.real)
        for nome, n in (("sup_vert", max(1, self.npair)),
                        ("n_por_par", max(1, self.npair)),
                        ("con_geom", max(1, 2 * MAX_CON_POR_PAR * self.npair)),
                        ("con_pair", max(1, MAX_CON_POR_PAR * self.npair)),
                        ("ncon", 1)):
            self.tam[nome] = n
            self.b[nome] = d.vazio(n, np.int32)

        self._mundo()
        self._defines = (
            ("NBODY", self.nbody), ("NJNT", self.njnt), ("NGEOM", self.ngeom),
            ("NQ", self.nq), ("NV", self.nv), ("NC", self.nC), ("NU", self.nu),
            ("NPAIR_MAX", max(1, self.npair)),
            ("MAX_CON_POR_PAR", MAX_CON_POR_PAR),
        )
        self.k = {nome: d.kernel(ARQUIVOS, nome, self.fp64, self._defines)
                  for nome in (
                      "fk_registradores", "fk_lds",
                      "com_momento", "com_acumula_nivel", "com_normaliza",
                      "com_inercia", "com_cdof",
                      "crb_inicia", "crb_acumula_nivel", "crb_monta_M",
                      "factor_M", "solve_M", "com_vel_nivel",
                      "passivo_amortecedor", "passivo_mola",
                      "rne_frente_nivel", "rne_tras_nivel", "rne_projeta",
                      "atuacao", "atuacao_projeta", "soma_smooth",
                      "euler_copia_M", "euler_diag_MhD", "euler_rhs",
                      "euler_qvel", "euler_qpos",
                      "suporte_plano_malha", "contatos_plano_malha",
                      "compacta_contatos")}
        self.usa_registradores = self.grupo >= max(self.nbody, self.ngeom)
        self.dt = self.real(mod.opcoes["timestep"])
        self.gravidade = np.asarray(mod.opcoes["gravity"], dtype=float)
        d.espera()

    # -------------------------------------------------------------- agenda

    @staticmethod
    def _vert_local(mod) -> np.ndarray:
        """global -> local no grafo de hull; -1 para vertice fora do hull."""
        vertadr = np.asarray(mod.arrays["mesh_vertadr"], dtype=np.int64)
        vertnum = np.asarray(mod.arrays["mesh_vertnum"], dtype=np.int64)
        graphadr = np.asarray(mod.arrays["mesh_graphadr"], dtype=np.int64)
        graph = np.asarray(mod.arrays["mesh_graph"], dtype=np.int64)
        total = int(vertadr[-1] + vertnum[-1]) if vertadr.size else 0
        out = np.full(max(1, total), -1, dtype=np.int32)
        for k in range(vertadr.size):
            ga = int(graphadr[k])
            if ga < 0:
                continue
            numvert = int(graph[ga])
            gid = graph[ga + 2 + numvert: ga + 2 + 2 * numvert]
            out[int(vertadr[k]) + gid] = np.arange(numvert, dtype=np.int32)
        return out

    def _agenda(self) -> None:
        """Niveis da arvore e listas de filhos, achatados para o device."""
        arv = self.arvore
        d = self.dev
        self.profundidade = arv.profundidade

        adr, num, chapa = [], [], []
        for n in arv.niveis:
            adr.append(len(chapa))
            num.append(int(n.size))
            chapa.extend(int(x) for x in n)
        self.nivel_adr = np.asarray(adr, dtype=np.int32)
        self.nivel_num = np.asarray(num, dtype=np.int32)
        self.b["nivel_adr"] = d.sobe(self.nivel_adr)
        self.b["nivel_num"] = d.sobe(self.nivel_num)
        self.b["nivel_corpos"] = d.sobe(np.asarray(chapa, dtype=np.int32))
        self.b["nivel_de"] = d.sobe(np.asarray(arv.nivel_de, dtype=np.int32))

        # pais que recebem de cada nivel: e por eles que a passada PARA TRAS e
        # paralelizada sem corrida entre irmaos
        padr, pnum, pchapa = [], [], []
        for p in arv.pais_por_nivel:
            padr.append(len(pchapa))
            pnum.append(int(p.size))
            pchapa.extend(int(x) for x in p)
        self.pais_adr = np.asarray(padr, dtype=np.int32)
        self.pais_num = np.asarray(pnum, dtype=np.int32)
        self.b["pais"] = d.sobe(np.asarray(pchapa or [0], dtype=np.int32))
        self.b["filhos_adr"] = d.sobe(np.asarray(arv.filhos_adr, dtype=np.int32))
        self.b["filhos_num"] = d.sobe(np.asarray(arv.filhos_num, dtype=np.int32))
        self.b["filhos"] = d.sobe(
            np.asarray(arv.filhos if arv.filhos.size else [0], dtype=np.int32))

    def _mundo(self) -> None:
        """Corpo 0: identidade. E `cacc[0] = -gravidade`, que o RNE le."""
        ident = np.zeros(9 * self.nbody, dtype=self.real)
        ident[0] = ident[4] = ident[8] = 1
        self.dev.escreve(self.b["xmat"], ident)
        self.dev.escreve(self.b["ximat"], ident)
        q = np.zeros(4 * self.nbody, dtype=self.real)
        q[0] = 1
        self.dev.escreve(self.b["xquat"], q)

    def _zera_raizes(self) -> None:
        """
        `cvel[0] = 0` e `cacc[0] = -g`: as condicoes de contorno do mundo.

        Sao duas escritas de 6 reais por passo. Ficam no host enquanto a forma e
        por etapa; na forma residente elas viram duas linhas no inicio do kernel.
        """
        zero6 = np.zeros(6, dtype=self.real)
        self.dev.escreve(self.b["cvel"], zero6)
        cacc0 = np.zeros(6, dtype=self.real)
        cacc0[3:6] = -self.gravidade
        self.dev.escreve(self.b["cacc"], cacc0)
        self.dev.escreve(self.b["cfrc_body"], zero6)

    # ------------------------------------------------------------- entrada

    def escreve_estado(self, qpos=None, qvel=None, ctrl=None,
                       mocap_pos=None, mocap_quat=None, qfrc_applied=None) -> None:
        for nome, v in (("qpos", qpos), ("qvel", qvel), ("ctrl", ctrl),
                        ("mocap_pos", mocap_pos), ("mocap_quat", mocap_quat),
                        ("qfrc_applied", qfrc_applied)):
            if v is None or not np.size(v):
                continue
            self.dev.escreve(self.b[nome],
                             np.asarray(v, dtype=self.real).ravel())

    # ------------------------------------------------------------- estagios

    def _roda(self, nome: str, n: int, args: tuple, grupo=None) -> None:
        self.dev.roda(self.k[nome], n, grupo, args)

    def cinematica(self) -> None:
        b = self.b
        args = (np.int32(self.profundidade), np.int32(self.nq), b["nivel_de"],
                b["body_parentid"], b["body_jntadr"], b["body_jntnum"],
                b["body_mocapid"], b["body_pos"], b["body_quat"],
                b["body_ipos"], b["body_iquat"], b["body_sameframe"],
                b["jnt_type"], b["jnt_qposadr"], b["jnt_axis"], b["jnt_pos"],
                b["qpos0"], b["geom_bodyid"], b["geom_pos"], b["geom_quat"],
                b["geom_sameframe"], b["qpos"], b["mocap_pos"], b["mocap_quat"],
                b["xpos"], b["xquat"], b["xmat"], b["xanchor"], b["xaxis"],
                b["xipos"], b["ximat"], b["geom_xpos"], b["geom_xmat"],
                np.int32(1))
        self._roda("fk_registradores", self.grupo, args, self.grupo)

    def com_pos(self) -> None:
        b = self.b
        self._roda("com_momento", self.nbody,
                   (b["xipos"], b["body_mass"], b["subtree_com"]))
        for d in range(self.profundidade - 1, 0, -1):
            n = int(self.pais_num[d])
            if not n:
                continue
            self._roda("com_acumula_nivel", n,
                       (b["pais"], np.int32(self.pais_adr[d]), np.int32(n),
                        b["filhos_adr"], b["filhos_num"], b["filhos"],
                        b["nivel_de"], np.int32(d), b["subtree_com"]))
        self._roda("com_normaliza", self.nbody,
                   (b["body_subtreemass"], b["xipos"], b["subtree_com"]))
        self._roda("com_inercia", self.nbody,
                   (b["body_rootid"], b["body_inertia"], b["body_mass"],
                    b["ximat"], b["xipos"], b["subtree_com"], b["cinert"]))
        self._roda("com_cdof", self.njnt,
                   (np.int32(self.njnt), b["jnt_type"], b["jnt_dofadr"],
                    b["jnt_bodyid"], b["body_rootid"], b["xmat"], b["xanchor"],
                    b["xaxis"], b["subtree_com"], b["cdof"]))

    def massa(self) -> None:
        b = self.b
        self._roda("crb_inicia", self.nbody, (b["cinert"], b["crb"]))
        for d in range(self.profundidade - 1, 0, -1):
            n = int(self.pais_num[d])
            if not n:
                continue
            self._roda("crb_acumula_nivel", n,
                       (b["pais"], np.int32(self.pais_adr[d]), np.int32(n),
                        b["filhos_adr"], b["filhos_num"], b["filhos"],
                        b["nivel_de"], np.int32(d), b["crb"]))
        self._roda("crb_monta_M", self.nv,
                   (b["M_rownnz"], b["M_rowadr"], b["dof_parentid"],
                    b["dof_bodyid"], b["dof_armature"], b["crb"], b["cdof"],
                    b["M"]))
        self._roda("factor_M", self.grupo,
                   (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["M"],
                    b["qLD"], b["qLDiagInv"]), self.grupo)

    def com_vel(self) -> None:
        b = self.b
        for d in range(1, self.profundidade):
            n = int(self.nivel_num[d])
            if not n:
                continue
            self._roda("com_vel_nivel", n,
                       (b["nivel_corpos"], np.int32(self.nivel_adr[d]),
                        np.int32(n), b["body_parentid"], b["body_dofadr"],
                        b["body_dofnum"], b["body_jntadr"], b["jnt_type"],
                        b["dof_jntid"], b["cdof"], b["qvel"], b["cvel"],
                        b["cdof_dot"]))

    def colisao(self) -> None:
        """
        Pares explicitos plano x malha -> lista de contatos, em ordem de par.

        Tres despachos: suporte (um work-group por par), geracao (uma thread
        por par) e compactacao. A ordem final e a de PAR, que e a ordem em que
        o MuJoCo os emite e, portanto, a ordem das linhas de restricao.
        """
        b = self.b
        if not self.npair:
            return
        gr = min(256, self.grupo)
        self.dev.roda(
            self.k["suporte_plano_malha"], self.npair * gr, gr,
            (np.int32(self.npair), b["pair_geom1"], b["pair_geom2"],
             b["geom_type"], b["geom_dataid"], b["geom_xpos"], b["geom_xmat"],
             b["geom_rbound"], b["pair_margin"], b["mesh_vertadr"],
             b["mesh_vertnum"], b["mesh_vert"], b["sup_vert"], b["sup_dist"]))
        self._roda("contatos_plano_malha", self.npair,
                   (np.int32(self.npair), b["pair_geom1"], b["pair_geom2"],
                    b["geom_dataid"], b["geom_rbound"], b["geom_xpos"],
                    b["geom_xmat"], b["pair_margin"], b["mesh_vertadr"],
                    b["mesh_vert"], b["mesh_graphadr"], b["mesh_graph"],
                    b["vert_local"], b["sup_vert"], b["sup_dist"],
                    b["n_por_par"], b["con_dist_bruto"], b["con_pos_bruto"],
                    b["con_normal_bruto"]))
        self._roda("compacta_contatos", 1,
                   (np.int32(self.npair), b["n_por_par"], b["con_dist_bruto"],
                    b["con_pos_bruto"], b["con_normal_bruto"], b["pair_geom1"],
                    b["pair_geom2"], b["con_dist"], b["con_pos"],
                    b["con_normal"], b["con_geom"], b["con_pair"], b["ncon"]),
                   1)

    def le_int(self, nome: str) -> np.ndarray:
        self.dev.espera()
        return self.dev.le(self.b[nome], self.tam[nome], np.int32)

    def contatos(self) -> dict:
        """Os contatos como o teste os compara: dist, pos, normal, geoms."""
        n = int(self.le_int("ncon")[0])
        return {
            "ncon": n,
            "dist": self.le("con_dist")[:n],
            "pos": self.le("con_pos")[:3 * n].reshape(-1, 3),
            "normal": self.le("con_normal")[:3 * n].reshape(-1, 3),
            "geom": self.le_int("con_geom")[:2 * n].reshape(-1, 2),
        }

    def passivo(self) -> None:
        b = self.b
        self._roda("passivo_amortecedor", self.nv,
                   (b["qvel"], b["dof_damping"], b["qfrc_passive"]))
        self._roda("passivo_mola", self.njnt,
                   (np.int32(self.njnt), b["jnt_type"], b["jnt_qposadr"],
                    b["jnt_dofadr"], b["jnt_stiffness"], b["qpos"],
                    b["qpos_spring"], b["qfrc_passive"]))

    def bias(self) -> None:
        b = self.b
        for d in range(1, self.profundidade):
            n = int(self.nivel_num[d])
            if not n:
                continue
            self._roda("rne_frente_nivel", n,
                       (b["nivel_corpos"], np.int32(self.nivel_adr[d]),
                        np.int32(n), b["body_parentid"], b["body_dofadr"],
                        b["body_dofnum"], b["cdof_dot"], b["qvel"], b["cinert"],
                        b["cvel"], b["cacc"], b["cfrc_body"]))
        for d in range(self.profundidade - 1, 0, -1):
            n = int(self.pais_num[d])
            if not n:
                continue
            self._roda("rne_tras_nivel", n,
                       (b["pais"], np.int32(self.pais_adr[d]), np.int32(n),
                        b["filhos_adr"], b["filhos_num"], b["filhos"],
                        b["nivel_de"], np.int32(d), b["cfrc_body"]))
        self._roda("rne_projeta", self.nv,
                   (b["dof_bodyid"], b["cdof"], b["cfrc_body"], b["qfrc_bias"]))

    def atuacao(self) -> None:
        b = self.b
        self._roda("atuacao", self.nu,
                   (np.int32(self.nu), b["actuator_trntype"], b["actuator_trnid"],
                    b["actuator_biastype"], b["actuator_gainprm"],
                    b["actuator_biasprm"], b["actuator_gear"],
                    b["actuator_ctrlrange"], b["actuator_ctrllimited"],
                    b["actuator_forcerange"], b["actuator_forcelimited"],
                    b["jnt_qposadr"], b["jnt_dofadr"], b["qpos"], b["qvel"],
                    b["ctrl"], b["actuator_length"], b["actuator_velocity"],
                    b["actuator_force"], b["qfrc_actuator"]))
        self._roda("atuacao_projeta", self.nv,
                   (np.int32(self.nu), b["actuator_trntype"],
                    b["actuator_trnid"], b["actuator_gear"], b["jnt_dofadr"],
                    b["actuator_force"], b["qfrc_actuator"]))

    def smooth(self) -> None:
        b = self.b
        self._roda("soma_smooth", self.nv,
                   (b["qfrc_passive"], b["qfrc_bias"], b["qfrc_actuator"],
                    b["qfrc_applied"], b["qfrc_smooth"]))
        self._roda("solve_M", 1,
                   (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["qLD"],
                    b["qLDiagInv"], b["qfrc_smooth"], b["qacc_smooth"]), 1)

    def euler(self) -> None:
        """`mj_Euler` com amortecimento implicito e integracao semi-implicita."""
        b = self.b
        self._roda("euler_copia_M", self.nC, (b["M"], b["qH"]))
        self._roda("euler_diag_MhD", self.nv,
                   (self.dt, b["M_rownnz"], b["M_rowadr"], b["dof_damping"],
                    b["qH"]))
        self._roda("factor_M", self.grupo,
                   (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["qH"],
                    b["qH"], b["qHDiagInv"]), self.grupo)
        self._roda("euler_rhs", self.nv,
                   (b["qfrc_smooth"], b["qfrc_constraint"], b["rhs"]))
        self._roda("solve_M", 1,
                   (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["qH"],
                    b["qHDiagInv"], b["rhs"], b["qacc"]), 1)
        self._roda("euler_qvel", self.nv, (self.dt, b["qacc"], b["qvel"]))
        self._roda("euler_qpos", self.njnt,
                   (self.dt, np.int32(self.njnt), b["jnt_type"],
                    b["jnt_qposadr"], b["jnt_dofadr"], b["qvel"], b["qpos"]))

    # ----------------------------------------------------------------- laco

    def forward_smooth(self, esperar: bool = True) -> None:
        """Tudo que o `mj_forward` faz ANTES do solver de restricao."""
        self._zera_raizes()
        self.cinematica()
        self.com_pos()
        self.massa()
        self.com_vel()
        self.colisao()
        self.passivo()
        self.bias()
        self.atuacao()
        self.smooth()
        if esperar:
            self.dev.espera()

    def passo(self, esperar: bool = True) -> None:
        """
        Um passo completo -- SEM restricao.

        `qfrc_constraint` fica zerado enquanto contato e solver nao existirem.
        Com a mosca no ar isto e a fisica inteira; com ela no chao, falta a
        forca que a impede de atravessar.
        """
        self.forward_smooth(esperar=False)
        self.euler()
        if esperar:
            self.dev.espera()

    # -------------------------------------------------------------- leitura

    def le(self, nome: str) -> np.ndarray:
        self.dev.espera()
        return self.dev.le(self.b[nome], self.tam[nome], self.real)
