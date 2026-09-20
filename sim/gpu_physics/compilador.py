r"""
Do `mjModel` compilado para um modelo plano, sem ponteiros, pronto pra GPU.

Duas responsabilidades, e so essas:

  1. **Recusar o que nao suportamos.** O backend GPU cobre um subconjunto do
     MuJoCo -- o subconjunto que o NeuroMechFly usa, medido em
     `inventario.py`, nao um subconjunto escolhido no papel. Qualquer coisa
     fora dele levanta `RecursoNaoSuportado` com o nome do recurso. Um solver
     que ignora em silencio um tendao que existe no modelo produz numero
     errado com cara de numero certo, e esse e o modo de falha que este
     arquivo existe para impedir.

  2. **Achatar.** O `mjModel` e um grafo com indices indiretos; a GPU quer
     arrays contiguos com o tipo escolhido de proposito. A traducao acontece
     UMA vez, no setup, na CPU -- nunca no laco quente.

Nada aqui simula. Nada aqui e chamado por passo.

## O hash do modelo fisico

`hash_modelo()` e o par de `hash_ciencia()`, para a outra metade do problema.
`hash_ciencia` cobre o cerebro: constantes de Shiu, transducao, timestep
neural. Este cobre o corpo: massas, inercias, juntas, damping, atrito, pares de
contato, parametros de geom, solver e integrador.

Ele e **derivado do modelo compilado**, nao de uma lista escrita a mao. Mexer
no MJCF, trocar uma malha, mudar um `<pair>` -- qualquer dos tres muda o hash
sem ninguem precisar lembrar de atualizar nada. E isso e o que permite dizer,
comparando MuJoCo CPU com Drosobot GPU, que os dois rodaram o MESMO corpo.

O que NAO entra: nome, cor, camera, luz, textura e malha de visualizacao. Um
geom que nao colide nem tem massa nao muda dinamica nenhuma, e faze-lo mudar o
hash tornaria o hash inutil -- ele passaria a mudar por motivo estetico.

## Precisao

Os arrays saem no tipo do `mjModel` (`float64` para `mjtNum`). Quem escolhe o
que baixar para `float32` e o backend, estagio por estagio, com medida -- nao o
compilador, em bloco e em silencio. Ver `docs/GPU_PHYSICS.md`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------- subconjunto

# O contrato. Cada entrada foi observada no modelo real do NeuroMechFly; nada
# aqui e aspiracional. Ver benchmarks/physics/mjmodel_inventario.json.
JNT_SUPORTADAS = {0, 3}                       # free, hinge
GEOM_SUPORTADOS = {0, 2, 5, 7}                # plane, sphere, cylinder, mesh
GEOM_COLIDIVEIS = {(0, 7), (5, 7)}            # plane-mesh, cylinder-mesh
TRN_SUPORTADAS = {0, 5}                       # joint, body (adesao)
DYN_SUPORTADAS = {0}                          # none (atuador sem estado)
GAIN_SUPORTADOS = {0}                         # fixed
BIAS_SUPORTADOS = {0, 1}                      # none, affine (servo de posicao)
INTEGRADOR_SUPORTADO = 0                      # Euler
SOLVER_SUPORTADO = 2                          # Newton
CONE_SUPORTADO = 0                            # pyramidal
CONDIM_SUPORTADO = {3}                        # atrito deslizante, sem torcao


# A versao do MuJoCo de onde os kernels foram portados. NAO e cosmetica: o
# `mjc_PlaneConvex` mudou de algoritmo entre 3.9 e 3.13 -- de "vizinhos do
# vertice de suporte no grafo de hull" para "face poligonal podada por area".
# Um porte feito da versao errada bate o contato mais profundo bit a bit e erra
# os contatos extras, que e um modo de falha quase invisivel: a mosca continua
# de pe, com o numero de contatos errado.
VERSAO_MUJOCO_PORTADA = "3.9.0"


class RecursoNaoSuportado(RuntimeError):
    """O modelo usa algo que o backend GPU nao implementa. Falha cedo."""


class VersaoMuJoCoDivergente(RuntimeError):
    """O runtime nao e a versao de onde os kernels foram portados."""


def _exige(cond: bool, msg: str) -> None:
    if not cond:
        raise RecursoNaoSuportado(msg)


def valida(m) -> dict:
    """
    Confere o modelo contra o subconjunto. Levanta no primeiro desvio.

    Devolve o que foi conferido, para ir ao metadata: dizer "validado" sem
    dizer contra o que nao ajuda ninguem seis meses depois.
    """
    tipos_jnt = {int(t) for t in m.jnt_type}
    _exige(tipos_jnt <= JNT_SUPORTADAS,
           f"tipos de junta fora do subconjunto: {sorted(tipos_jnt - JNT_SUPORTADAS)} "
           "(suportado: free=0, hinge=3)")

    tipos_geom = {int(t) for t in m.geom_type}
    _exige(tipos_geom <= GEOM_SUPORTADOS,
           f"tipos de geom fora do subconjunto: {sorted(tipos_geom - GEOM_SUPORTADOS)}")

    pares = set()
    for i in range(int(m.npair)):
        t1 = int(m.geom_type[m.pair_geom1[i]])
        t2 = int(m.geom_type[m.pair_geom2[i]])
        pares.add((min(t1, t2), max(t1, t2)))
    _exige(pares <= GEOM_COLIDIVEIS,
           f"pares de colisao fora do subconjunto: {sorted(pares - GEOM_COLIDIVEIS)}")

    # O achado dos pilares: este modelo NAO usa contype/conaffinity. Os contatos
    # vem de `<pair>` explicitos, e um backend que inferisse a topologia de
    # colisao pelas mascaras atravessaria a mosca pelo obstaculo sem um unico
    # contato -- foi exatamente o que aconteceu antes da correcao.
    mascaras = int(np.sum((np.asarray(m.geom_contype) != 0)
                          | (np.asarray(m.geom_conaffinity) != 0)))

    _exige({int(c) for c in m.geom_condim} <= CONDIM_SUPORTADO,
           "condim fora de {3}: atrito torcional/rolante nao implementado")
    _exige(int(m.opt.integrator) == INTEGRADOR_SUPORTADO, "integrador != Euler")
    _exige(int(m.opt.solver) == SOLVER_SUPORTADO, "solver != Newton")
    _exige(int(m.opt.cone) == CONE_SUPORTADO, "cone != pyramidal")
    _exige(int(m.ntendon) == 0, "tendoes nao implementados")
    _exige(int(m.neq) == 0, "restricoes de igualdade nao implementadas")
    _exige(int(m.nflex) == 0, "corpos deformaveis (flex) nao implementados")
    _exige(int(m.na) == 0, "atuadores com estado (act) nao implementados")
    _exige(int(np.sum(m.jnt_limited)) == 0, "limites de junta nao implementados")
    _exige(float(np.max(m.dof_frictionloss)) == 0.0,
           "atrito seco nas juntas (frictionloss) nao implementado")
    _exige(float(m.opt.viscosity) == 0.0 and float(m.opt.density) == 0.0,
           "forcas de fluido (viscosity/density) nao implementadas")
    _exige(not np.any(np.asarray(m.opt.wind) != 0), "vento nao implementado")

    # Os kernels assumem mola e amortecedor LINEARES: `-x*stiffness` e
    # `-v*damping`. O MuJoCo 3.9 permite um polinomio por cima disso
    # (`mju_polyForce`), e se ele existir o assumido aqui produz forca errada
    # em silencio -- que e o pior desfecho possivel.
    for campo in ("jnt_stiffnesspoly", "dof_dampingpoly", "actuator_dampingpoly"):
        v = getattr(m, campo, None)
        if v is not None:
            _exige(not np.any(np.asarray(v) != 0),
                   f"{campo} nao nulo: mola/amortecedor polinomial nao "
                   "implementado (os kernels assumem termo linear)")

    _exige({int(t) for t in m.actuator_trntype} <= TRN_SUPORTADAS,
           "transmissao de atuador fora de {joint, body}")
    _exige({int(t) for t in m.actuator_dyntype} <= DYN_SUPORTADAS,
           "dyntype de atuador != none")
    _exige({int(t) for t in m.actuator_gaintype} <= GAIN_SUPORTADOS,
           "gaintype de atuador != fixed")
    _exige({int(t) for t in m.actuator_biastype} <= BIAS_SUPORTADOS,
           "biastype de atuador fora de {none, affine}")

    return {
        "juntas": sorted(tipos_jnt),
        "geoms": sorted(tipos_geom),
        "pares_colisao": sorted(pares),
        "condim": sorted({int(c) for c in m.geom_condim}),
        "integrador": "Euler",
        "solver": "Newton",
        "cone": "pyramidal",
        "mujoco_portado": VERSAO_MUJOCO_PORTADA,
        "colisao_por_pares_explicitos": True,
        "geoms_com_mascara_nao_nula": mascaras,
        "nota_mascara": (
            "geom_contype/conaffinity sao (0,0) em toda a mosca: a topologia "
            "de contato deste modelo vem dos <pair> explicitos. Um backend que "
            "a inferisse pelas mascaras nao teria contato nenhum."),
    }


# ------------------------------------------------------------- achatamento

# Tudo que muda a DINAMICA. A lista e explicita de proposito: um hash sobre o
# mjModel inteiro mudaria quando alguem trocasse a cor de um geom.
CAMPOS_DINAMICA = (
    # arvore e poses de referencia
    "body_parentid", "body_rootid", "body_weldid", "body_jntadr", "body_jntnum",
    "body_dofadr", "body_dofnum", "body_pos", "body_quat", "body_ipos",
    "body_iquat", "body_mass", "body_inertia", "body_treeid", "body_mocapid",
    "body_geomadr", "body_geomnum", "body_sameframe", "body_simple",
    "body_subtreemass", "body_invweight0",
    # juntas
    "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_axis",
    "jnt_pos", "jnt_stiffness", "jnt_range", "jnt_limited",
    # graus de liberdade
    "dof_bodyid", "dof_jntid", "dof_parentid", "dof_treeid", "dof_Madr",
    "dof_simplenum", "dof_armature", "dof_damping", "dof_M0", "dof_invweight0",
    # esparsidade da matriz de massa em CSR: uma linha por dof, com a cadeia
    # dele ate a raiz. E o que `mj_factorI` percorre.
    "M_rownnz", "M_rowadr", "M_colind",
    # geometria de colisao
    "geom_bodyid", "geom_type", "geom_pos", "geom_quat", "geom_size",
    "geom_dataid", "geom_rbound", "geom_condim", "geom_friction",
    "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
    "geom_contype", "geom_conaffinity", "geom_priority", "geom_sameframe",
    # pares explicitos -- a topologia de contato REAL deste modelo
    "pair_dim", "pair_geom1", "pair_geom2", "pair_signature", "pair_solref",
    "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap",
    "pair_friction", "exclude_signature",
    # atuadores
    "actuator_trntype", "actuator_trnid", "actuator_dyntype",
    "actuator_gaintype", "actuator_biastype", "actuator_gainprm",
    "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange",
    "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited",
    "actuator_gear", "actuator_cranklength", "actuator_acc0",
    "actuator_length0", "actuator_lengthrange",
    # estado de referencia
    "qpos0", "qpos_spring",
    # termos polinomiais de mola/amortecedor. Sao zero neste modelo, e o
    # compilador confere isso em `valida()` antes de o kernel assumir.
    "jnt_stiffnesspoly", "dof_dampingpoly",
    # malhas: vertices e faces, que sao o que a colisao usa
    "mesh_vertadr", "mesh_vertnum", "mesh_vert", "mesh_graphadr", "mesh_graph",
    "mesh_rbound",
    "mesh_polyadr", "mesh_polynum", "mesh_polyvertadr", "mesh_polyvertnum",
    "mesh_polyvert", "mesh_polynormal", "mesh_polymapadr", "mesh_polymapnum",
    "mesh_polymap",
)

CAMPOS_OPCAO = (
    "timestep", "gravity", "wind", "density", "viscosity", "impratio",
    "o_margin", "o_solref", "o_solimp", "integrator", "cone", "jacobian",
    "solver", "iterations", "ls_iterations", "tolerance", "ls_tolerance",
    "noslip_iterations", "noslip_tolerance", "disableflags", "enableflags",
)


@dataclass(frozen=True)
class ModeloGPU:
    """O corpo, achatado. Sem `mjModel`, sem ponteiro, sem nome."""

    dims: dict
    opcoes: dict
    arrays: dict = field(repr=False)
    subset: dict = field(repr=False)
    hash_modelo: str = ""

    @property
    def nv(self) -> int:
        return self.dims["nv"]

    @property
    def nq(self) -> int:
        return self.dims["nq"]

    def bytes_totais(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())


def _opcao(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return [float(x) for x in np.asarray(v).ravel()]


def _opcoes(m) -> dict:
    return {nome: _opcao(getattr(m.opt, nome))
            for nome in CAMPOS_OPCAO if hasattr(m.opt, nome)}


def confere_versao(estrito: bool = True) -> str:
    """
    O runtime e a versao de onde os kernels foram portados?

    Custou uma depuracao descobrir que nao era: o clone de pesquisa estava em
    3.13.1 e o binario que roda os experimentos em 3.9.0. Os dois concordam na
    dinamica suave e discordam na geracao de contatos.
    """
    import mujoco as mj

    v = getattr(mj, "__version__", "?")
    if estrito and v != VERSAO_MUJOCO_PORTADA:
        raise VersaoMuJoCoDivergente(
            f"mujoco {v} em execucao, kernels portados de "
            f"{VERSAO_MUJOCO_PORTADA}. O `mjc_PlaneConvex` mudou de algoritmo "
            "entre essas versoes; comparar a GPU com este runtime mediria a "
            "diferenca entre versoes do MuJoCo, nao o porte. Atualize os "
            "kernels a partir da fonte certa (e `VERSAO_MUJOCO_PORTADA`) ou "
            "volte o runtime.")
    return v


def compila(m, estrito: bool = True) -> ModeloGPU:
    """`mjModel` -> `ModeloGPU`. Valida antes; levanta se o modelo sai do subset."""
    confere_versao(estrito)
    subset = valida(m)
    arrays = {}
    for nome in CAMPOS_DINAMICA:
        v = getattr(m, nome, None)
        if v is None:
            continue
        arrays[nome] = np.ascontiguousarray(v).copy()

    dims = {k: int(getattr(m, k)) for k in
            ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nmesh",
             "npair", "nexclude", "neq", "ntendon", "nsensor", "nmocap",
             "nM", "nD", "nC", "ntree")
            if hasattr(m, k)}

    mod = ModeloGPU(dims=dims, opcoes=_opcoes(m), arrays=arrays, subset=subset)
    object.__setattr__(mod, "hash_modelo", hash_modelo(mod))
    return mod


def hash_modelo(mod: ModeloGPU) -> str:
    """
    Resumo curto de tudo que muda a dinamica. Derivado, nunca escrito a mao.

    Bytes crus dos arrays, na ordem declarada, mais dimensoes e opcoes. Trocar
    uma malha, uma massa, um `<pair>` ou o timestep muda este valor; trocar a
    cor de um geom nao.
    """
    h = hashlib.sha256()
    h.update(repr(sorted(mod.dims.items())).encode("utf-8"))
    h.update(repr(sorted(mod.opcoes.items())).encode("utf-8"))
    for nome in CAMPOS_DINAMICA:
        a = mod.arrays.get(nome)
        if a is None:
            continue
        h.update(nome.encode("ascii"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]
