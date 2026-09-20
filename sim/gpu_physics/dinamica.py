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

ARQUIVOS = ("cinematica.cl", "dinamica.cl", "colisao.cl",
            "restricao.cl", "solver.cl", "fundido.cl")

# Arrays do modelo, por tipo de destino no device.
INTEIROS = (
    "body_parentid", "body_rootid", "body_jntadr", "body_jntnum", "body_mocapid",
    "body_sameframe", "body_dofadr", "body_dofnum", "body_geomadr",
    "body_geomnum",
    "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid",
    "dof_bodyid", "dof_jntid", "dof_parentid", "dof_Madr", "body_weldid",
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
    "pair_solimp", "pair_gap", "body_invweight0",
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
        # tetos estaticos: 3 contatos por par (maxplanemesh) e 4 linhas por
        # contato (piramidal com condim=3). Sem alocacao dinamica no laco.
        self.ncon_max = max(1, MAX_CON_POR_PAR * self.npair)
        self.nefc_max = 4 * self.ncon_max
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
            # `qH_in` existe porque passar o MESMO buffer como entrada `const`
            # e como saida de `factor_M` e aliasing, e o compilador tem licenca
            # para supor que nao ha. Custou um passo divergindo para 1e71.
            "qH": self.nC, "qH_in": self.nC, "qHDiagInv": self.nv,
            "qfrc_passive": self.nv, "qfrc_bias": self.nv,
            "qfrc_actuator": self.nv, "qfrc_smooth": self.nv,
            "qfrc_constraint": self.nv,
            "qacc_smooth": self.nv, "qacc": self.nv, "rhs": self.nv,
            "qacc_warmstart": self.nv,
            # `mj_Euler` resolve uma aceleracao PROPRIA, com `M + h*D`, e a usa
            # so para integrar; o `d.qacc` do MuJoCo continua sendo a do
            # solver. Escrever a do Euler por cima faria o trace registrar a
            # grandeza errada -- e foi assim que um diagnostico de
            # estacionariedade leu 1e+02 onde havia 1e-13.
            "qacc_euler": self.nv,
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
            "con_frame": max(1, 9 * MAX_CON_POR_PAR * self.npair),
            "con_includemargin": max(1, MAX_CON_POR_PAR * self.npair),
            "con_mu": max(1, MAX_CON_POR_PAR * self.npair),
            # restricoes: Jacobiana DENSA nefc x nv
            "efc_J": max(1, self.nefc_max * self.nv),
            "efc_pos": max(1, self.nefc_max),
            "efc_margin": max(1, self.nefc_max),
            "efc_diagApprox": max(1, self.nefc_max),
            "efc_R": max(1, self.nefc_max),
            "efc_D": max(1, self.nefc_max),
            "efc_KBIP": max(1, 4 * self.nefc_max),
            "efc_vel": max(1, self.nefc_max),
            "efc_aref": max(1, self.nefc_max),
            "efc_force": max(1, self.nefc_max),
            "efc_jar": max(1, self.nefc_max),
            # M densa: o Hessiano `M + J'DJ` preenche tudo, entao nao ha
            # esparsidade a preservar dentro do solver
            "Md": self.nv * self.nv,
            "ades_momento": self.nu * self.nv,
        }
        for nome, n in derivados.items():
            self.tam[nome] = n
            self.b[nome] = d.vazio(n, self.real)
        for nome, n in (("efc_id", max(1, self.nefc_max)),
                        ("nefc", 1), ("solver_iters", 1),
                        ("con_exclude", max(1, self.ncon_max)),
                        ("con_efcadr", max(1, self.ncon_max)),
                        ("ades_conta", self.nu),
                        ("sup_vert", max(1, self.npair)),
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
            ("NCON_MAX", self.ncon_max), ("NEFC_MAX", self.nefc_max),
        )
        self.k = {nome: d.kernel_proprio(ARQUIVOS, nome, self.fp64, self._defines)
                  for nome in (
                      "fk_registradores", "fk_lds",
                      "zera_raizes",
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
                      "compacta_contatos",
                      "contato_quadro", "contato_jacobiana",
                      "restricao_impedancia", "restricao_aref",
                      "contato_enderecos", "adesao_momento", "adesao_projeta",
                      "M_densa", "M_simetriza", "solver_newton",
                      "com_pos_fundido", "massa_fundida", "com_vel_fundido",
                      "bias_fundido", "forcas_fundidas", "restricoes_fundidas",
                      "smooth_fundido", "euler_fundido")}
        self.usa_registradores = self.grupo >= max(self.nbody, self.ngeom)
        # Argumentos ja ligados, por kernel. Ver `_chave`.
        self._args_ligados: dict[str, tuple] = {}
        self.dt = self.real(mod.opcoes["timestep"])
        self.tolerancia = float(mod.opcoes["tolerance"])
        # `meaninertia` normaliza o criterio de parada do solver, como no
        # mj_solveNewton. O MuJoCo o calcula em `mj_setConst`; aqui vem do
        # mesmo lugar: a media de `dof_M0`.
        self.meaninertia = float(np.mean(np.asarray(mod.arrays["dof_M0"])))
        # Ligado por padrao, como no MuJoCo. Desligar isola o solver do
        # historico, que e util para depurar um passo isolado.
        self.warmstart = True
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
        # Os enderecos por nivel sobem como BUFFER: o kernel fundido percorre
        # os niveis por dentro, e passa-los como escalar obrigaria a religar
        # argumento a cada nivel -- que e exatamente o custo que a fusao remove.
        self.b["pais_adr"] = d.sobe(self.pais_adr)
        self.b["pais_num"] = d.sobe(self.pais_num)
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
        """`cvel[0] = 0`, `cacc[0] = -g`, `cfrc_body[0] = 0`, no device."""
        gx, gy, gz = (self.real(v) for v in self.gravidade)
        self._roda("zera_raizes", 6,
                   (gx, gy, gz, self.b["cvel"], self.b["cacc"],
                    self.b["cfrc_body"]))

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

    @staticmethod
    def _chave(args: tuple):
        """
        Identidade dos argumentos, para saber se `set_args` pode ser pulado.

        Buffer entra por identidade (`id`), escalar por valor. Medido nesta
        placa: `set_args` com 21 argumentos custa 21 us e o enqueue custa
        1,4 us -- 94% do custo por despacho era religar argumentos que nunca
        mudam entre passos.
        """
        return tuple(id(a) if hasattr(a, "get_host_array") or type(a).__name__
                     == "Buffer" else (type(a).__name__, a.item()
                                       if hasattr(a, "item") else a)
                     for a in args)

    def _roda(self, nome: str, n: int, args: tuple, grupo=None) -> None:
        k = self.k[nome]
        if args:
            chave = self._chave(args)
            if self._args_ligados.get(nome) != chave:
                k.set_args(*args)
                self._args_ligados[nome] = chave
        self.dev.roda(k, n, grupo)

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
                       (b["pais"], b["pais_adr"], b["pais_num"],
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
                       (b["pais"], b["pais_adr"], b["pais_num"],
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
                       (b["nivel_corpos"], b["nivel_adr"], b["nivel_num"],
                        np.int32(d), b["body_parentid"], b["body_dofadr"],
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

    def solver(self) -> None:
        """
        Newton projetado: `qacc` e `qfrc_constraint` a partir de `qacc_smooth`.

        Um dispatch. O objetivo e o do MuJoCo 3.9.0 para o subconjunto
        piramidal; como ele e estritamente convexo, a solucao e unica e os
        dois convergem para ela. Ver o cabecalho de `kernels/solver.cl`.
        """
        b = self.b
        self._roda("M_densa", self.nv,
                   (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["M"],
                    b["Md"]))
        self._roda("M_simetriza", self.nv, (b["Md"],))
        self._roda("solver_newton", self.grupo,
                   (b["nefc"], b["efc_J"], b["efc_D"], b["efc_aref"], b["Md"],
                    b["qacc_smooth"], self.real(self.tolerancia),
                    self.real(self.meaninertia), b["qacc_warmstart"],
                    np.int32(1 if self.warmstart else 0),
                    b["qacc"], b["efc_force"],
                    b["qfrc_constraint"], b["efc_jar"], b["solver_iters"]),
                   self.grupo)

    def restricoes(self) -> None:
        """
        Contatos -> linhas de restricao: J, pos, margin, R, D, KBIP, aref.

        Quatro despachos, na ordem de dependencia do `mj_makeConstraint`:
        quadro do contato, Jacobiana (uma thread por contato x dof),
        impedancia (uma thread por contato, porque as quatro linhas dele
        compartilham `imp` e o ajuste piramidal de R) e a referencia.
        """
        b = self.b
        if not self.npair:
            return
        self._roda("contato_quadro", self.ncon_max,
                   (b["ncon"], b["con_normal"], b["con_pair"], b["pair_margin"],
                    b["pair_gap"], b["con_dist"], b["con_frame"],
                    b["con_includemargin"], b["con_exclude"]))
        self._roda("contato_enderecos", 1,
                   (b["ncon"], b["con_exclude"], b["con_efcadr"], b["nefc"]), 1)
        self._roda("contato_jacobiana", self.ncon_max * self.nv,
                   (b["ncon"], b["con_geom"], b["con_pos"], b["con_frame"],
                    b["con_dist"], b["con_includemargin"], b["con_pair"],
                    b["pair_friction"], b["con_efcadr"], b["geom_bodyid"],
                    b["body_rootid"], b["body_weldid"], b["body_dofadr"],
                    b["body_dofnum"], b["dof_parentid"], b["subtree_com"],
                    b["cdof"], b["efc_J"], b["efc_pos"], b["efc_margin"],
                    b["efc_id"]))
        self._roda("restricao_impedancia", self.ncon_max,
                   (b["ncon"], self.real(self.mod.opcoes["impratio"]),
                    b["con_efcadr"],
                    b["con_geom"], b["con_pair"], b["con_dist"],
                    b["con_includemargin"], b["pair_friction"],
                    b["pair_solref"], b["pair_solimp"], b["geom_bodyid"],
                    b["body_invweight0"], b["efc_diagApprox"], b["efc_R"],
                    b["efc_D"], b["efc_KBIP"], b["con_mu"]))
        self._roda("restricao_aref", self.nefc_max,
                   (b["nefc"], b["efc_J"], b["qvel"], b["efc_KBIP"],
                    b["efc_pos"], b["efc_margin"], b["efc_vel"], b["efc_aref"]))

    def efc(self) -> dict:
        """As linhas de restricao como o teste as compara."""
        n = int(self.le_int("nefc")[0])
        return {
            "nefc": n,
            "J": self.le("efc_J")[:n * self.nv].reshape(n, self.nv),
            "pos": self.le("efc_pos")[:n],
            "margin": self.le("efc_margin")[:n],
            "diagApprox": self.le("efc_diagApprox")[:n],
            "R": self.le("efc_R")[:n],
            "D": self.le("efc_D")[:n],
            "aref": self.le("efc_aref")[:n],
            "vel": self.le("efc_vel")[:n],
            "id": self.le_int("efc_id")[:n],
        }

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
                       (b["nivel_corpos"], b["nivel_adr"], b["nivel_num"],
                        np.int32(d), b["body_parentid"], b["body_dofadr"],
                        b["body_dofnum"], b["cdof_dot"], b["qvel"], b["cinert"],
                        b["cvel"], b["cacc"], b["cfrc_body"]))
        for d in range(self.profundidade - 1, 0, -1):
            n = int(self.pais_num[d])
            if not n:
                continue
            self._roda("rne_tras_nivel", n,
                       (b["pais"], b["pais_adr"], b["pais_num"],
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
        # Adesao DEPOIS da projecao das juntas, e depois das restricoes: o
        # momento dela e a media das Jacobianas normais dos contatos, entao ela
        # so existe quando `efc_J` ja existe.
        self._roda("adesao_momento", self.nu * self.nv,
                   (np.int32(self.nu), b["ncon"], b["actuator_trntype"],
                    b["actuator_trnid"], b["con_geom"], b["con_efcadr"],
                    b["geom_bodyid"], b["efc_J"], b["ades_momento"],
                    b["ades_conta"]))
        self._roda("adesao_projeta", self.nv,
                   (np.int32(self.nu), b["actuator_trntype"],
                    b["ades_momento"], b["actuator_force"],
                    b["qfrc_actuator"]))

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
        self._roda("euler_copia_M", self.nC, (b["M"], b["qH_in"]))
        self._roda("euler_diag_MhD", self.nv,
                   (self.dt, b["M_rownnz"], b["M_rowadr"], b["dof_damping"],
                    b["qH_in"]))
        self._roda("factor_M", self.grupo,
                   (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["qH_in"],
                    b["qH"], b["qHDiagInv"]), self.grupo)
        self._roda("euler_rhs", self.nv,
                   (b["qfrc_smooth"], b["qfrc_constraint"], b["rhs"]))
        self._roda("solve_M", 1,
                   (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["qH"],
                    b["qHDiagInv"], b["rhs"], b["qacc_euler"]), 1)
        self._roda("euler_qvel", self.nv, (self.dt, b["qacc_euler"], b["qvel"]))
        self._roda("euler_qpos", self.njnt,
                   (self.dt, np.int32(self.njnt), b["jnt_type"],
                    b["jnt_qposadr"], b["jnt_dofadr"], b["qvel"], b["qpos"]))

    # ----------------------------------------------------------------- laco

    # --------------------------------------------------------- fundido
    #
    # As mesmas etapas, em poucos despachos. Os kernels daqui chamam as MESMAS
    # funcoes `..._um` dos kernels por estagio (ver `kernels/fundido.cl`), entao
    # os dois caminhos nao podem divergir na conta -- so no numero de despachos.

    def _f(self, nome: str, args: tuple) -> None:
        """Despacho fundido: sempre um work-group, sempre `self.grupo`."""
        self._roda(nome, self.grupo, args, self.grupo)

    def com_pos_f(self) -> None:
        b = self.b
        self._f("com_pos_fundido",
                (np.int32(self.profundidade), np.int32(self.njnt),
                 b["pais"], b["pais_adr"], b["pais_num"], b["filhos_adr"],
                 b["filhos_num"], b["filhos"], b["nivel_de"],
                 b["xipos"], b["body_mass"], b["body_subtreemass"],
                 b["body_rootid"], b["body_inertia"], b["ximat"],
                 b["jnt_type"], b["jnt_dofadr"], b["jnt_bodyid"], b["xmat"],
                 b["xanchor"], b["xaxis"],
                 b["subtree_com"], b["cinert"], b["cdof"]))

    def massa_f(self) -> None:
        b = self.b
        self._f("massa_fundida",
                (np.int32(self.profundidade),
                 b["pais"], b["pais_adr"], b["pais_num"], b["filhos_adr"],
                 b["filhos_num"], b["filhos"], b["nivel_de"],
                 b["M_rownnz"], b["M_rowadr"], b["M_colind"],
                 b["dof_parentid"], b["dof_bodyid"], b["dof_armature"],
                 b["cinert"], b["cdof"], b["crb"], b["M"], b["qLD"],
                 b["qLDiagInv"]))

    def com_vel_f(self) -> None:
        b = self.b
        self._f("com_vel_fundido",
                (np.int32(self.profundidade), b["nivel_corpos"],
                 b["nivel_adr"], b["nivel_num"], b["body_parentid"],
                 b["body_dofadr"], b["body_dofnum"], b["body_jntadr"],
                 b["jnt_type"], b["dof_jntid"], b["cdof"], b["qvel"],
                 b["cvel"], b["cdof_dot"]))

    def bias_f(self) -> None:
        b = self.b
        self._f("bias_fundido",
                (np.int32(self.profundidade), b["nivel_corpos"],
                 b["nivel_adr"], b["nivel_num"], b["pais"], b["pais_adr"],
                 b["pais_num"], b["filhos_adr"], b["filhos_num"], b["filhos"],
                 b["nivel_de"], b["body_parentid"], b["body_dofadr"],
                 b["body_dofnum"], b["dof_bodyid"], b["cdof_dot"], b["qvel"],
                 b["cinert"], b["cvel"], b["cdof"], b["cacc"], b["cfrc_body"],
                 b["qfrc_bias"]))

    def forcas_f(self) -> None:
        b = self.b
        self._f("forcas_fundidas",
                (np.int32(self.njnt), np.int32(self.nu),
                 b["qvel"], b["dof_damping"], b["jnt_type"], b["jnt_qposadr"],
                 b["jnt_dofadr"], b["jnt_stiffness"], b["qpos"],
                 b["qpos_spring"], b["actuator_trntype"], b["actuator_trnid"],
                 b["actuator_biastype"], b["actuator_gainprm"],
                 b["actuator_biasprm"], b["actuator_gear"],
                 b["actuator_ctrlrange"], b["actuator_ctrllimited"],
                 b["actuator_forcerange"], b["actuator_forcelimited"],
                 b["ctrl"], b["ncon"], b["con_geom"], b["con_efcadr"],
                 b["geom_bodyid"], b["efc_J"], b["qfrc_applied"],
                 b["qfrc_bias"], b["qfrc_passive"], b["actuator_length"],
                 b["actuator_velocity"], b["actuator_force"],
                 b["qfrc_actuator"], b["ades_momento"], b["ades_conta"],
                 b["qfrc_smooth"]))

    def restricoes_f(self) -> None:
        b = self.b
        if not self.npair:
            return
        self._f("restricoes_fundidas",
                (np.int32(self.ncon_max),
                 self.real(self.mod.opcoes["impratio"]),
                 b["ncon"], b["con_normal"], b["con_pair"], b["pair_margin"],
                 b["pair_gap"], b["con_dist"], b["con_geom"], b["con_pos"],
                 b["pair_friction"], b["pair_solref"], b["pair_solimp"],
                 b["geom_bodyid"], b["body_rootid"], b["body_weldid"],
                 b["body_dofadr"], b["body_dofnum"], b["dof_parentid"],
                 b["body_invweight0"], b["subtree_com"], b["cdof"], b["qvel"],
                 b["con_frame"], b["con_includemargin"], b["con_exclude"],
                 b["con_efcadr"], b["nefc"], b["efc_J"], b["efc_pos"],
                 b["efc_margin"], b["efc_id"], b["efc_diagApprox"], b["efc_R"],
                 b["efc_D"], b["efc_KBIP"], b["con_mu"], b["efc_vel"],
                 b["efc_aref"]))

    def smooth_f(self) -> None:
        b = self.b
        self._f("smooth_fundido",
                (b["M_rownnz"], b["M_rowadr"], b["M_colind"], b["qLD"],
                 b["qLDiagInv"], b["qfrc_smooth"], b["qacc_smooth"]))

    def euler_f(self) -> None:
        b = self.b
        self._f("euler_fundido",
                (self.dt, np.int32(self.njnt), b["M_rownnz"], b["M_rowadr"],
                 b["M_colind"], b["M"], b["dof_damping"], b["qfrc_smooth"],
                 b["qfrc_constraint"], b["jnt_type"], b["jnt_qposadr"],
                 b["jnt_dofadr"], b["qH_in"], b["qH"], b["qHDiagInv"],
                 b["rhs"], b["qacc_euler"], b["qvel"], b["qpos"]))

    def passo_fundido(self, esperar: bool = True) -> None:
        """
        O passo completo em 13 despachos, contra 80 do caminho por estagio.

        Mesma fisica: os kernels fundidos chamam as mesmas funcoes `..._um`.
        O caminho por estagio continua existindo (`passo`), e e para ele que se
        volta quando algo diverge -- com um despacho por nivel da para parar em
        qualquer etapa e comparar com o `mjData`.
        """
        self._zera_raizes()
        self.cinematica()
        self.com_pos_f()
        self.massa_f()
        self.com_vel_f()
        self.colisao()
        self.restricoes_f()
        # `bias_f` ANTES de `forcas_f`: o kernel de forcas termina somando
        # `qfrc_smooth`, que le `qfrc_bias`. A ordem e a mesma do caminho por
        # estagio, onde `smooth` vem depois de `bias`.
        self.bias_f()
        self.forcas_f()
        self.smooth_f()
        self.solver()
        self.euler_f()
        if esperar:
            self.dev.espera()

    def forward_smooth(self, esperar: bool = True) -> None:
        """Tudo que o `mj_forward` faz ANTES do solver de restricao."""
        self._zera_raizes()
        self.cinematica()
        self.com_pos()
        self.massa()
        self.com_vel()
        self.colisao()
        self.restricoes()
        self.passivo()
        self.bias()
        self.atuacao()
        self.smooth()
        if esperar:
            self.dev.espera()

    def forward(self, esperar: bool = True) -> None:
        """`mj_forward` completo: dinamica suave + solver de restricao."""
        self.forward_smooth(esperar=False)
        self.solver()
        if esperar:
            self.dev.espera()

    def passo(self, esperar: bool = True) -> None:
        """
        Um passo completo: `state(t)` -> `state(t + dt)`, so na GPU.

        `mj_forward` (cinematica, inercia, colisao, restricoes, solver) mais
        `mj_Euler`. O MuJoCo nao executa nenhuma etapa dinamica aqui.
        """
        self.forward(esperar=False)
        self.euler()
        if esperar:
            self.dev.espera()

    # -------------------------------------------------------------- leitura

    def le(self, nome: str) -> np.ndarray:
        self.dev.espera()
        return self.dev.le(self.b[nome], self.tam[nome], self.real)
