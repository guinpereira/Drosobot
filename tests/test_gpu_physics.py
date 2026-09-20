r"""
O que o backend de fisica nao pode quebrar.

    .venv-flygym2\Scripts\python tests/test_gpu_physics.py

Cinco propriedades, cinco motivos distintos. Nao ha um sexto teste medindo a
mesma coisa por outro angulo -- isso so faria a suite demorar mais e falhar
junto.

  1. **O hash da ciencia e invariante ao backend.** Trocar de motor fisico nao
     e mudanca cientifica. Se `hash_ciencia()` mudar durante trabalho de
     backend, alguem mexeu em Shiu, na transducao, no timestep ou no
     conectoma -- e ai o certo e parar, nao atualizar o numero aqui.

  2. **O hash do modelo fisico e derivado, nao escrito.** Compilar duas vezes
     da o mesmo valor; mudar uma massa muda o valor. Sem as duas metades ele
     nao serve: um hash que nao e estavel nao compara nada, e um que nao muda
     nao protege nada.

  3. **O subconjunto MJCF recusa o que nao implementa.** Um solver que ignora
     em silencio um tendao produz numero errado com cara de numero certo.

  4. **O pilar barra a mosca.** A regressao fisica que veio da depuracao do
     campo de obstaculos: este modelo nao usa `contype`/`conaffinity` -- os 69
     geoms da mosca sao (0,0) -- e os contatos vem de `<pair>` explicitos. Um
     backend que inferisse a topologia de contato pelas mascaras atravessaria o
     pilar sem um unico contato, que foi exatamente o que aconteceu antes da
     correcao. O teste mede contato de verdade, nao a contagem de pares.

  5. **A cinematica da GPU bate com a do MuJoCo.** E o unico estagio que hoje
     roda de fato na GPU; a tolerancia e do tamanho do epsilon da maquina em
     fp64, porque o porte replica a ordem das operacoes do original.
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "sim"))

DT = 1e-4

# Pinado de proposito. Uma mudanca aqui tem que ser uma decisao cientifica
# deliberada, com o motivo no commit -- nunca um efeito colateral de trabalho
# de backend.
HASH_CIENCIA_ESPERADO = "99cad7adc23d66e7"

# fp64 replica a ordem das operacoes do MuJoCo, entao o que sobra e
# arredondamento de ULP; `sin`/`cos` da GPU respondem pela maior parte.
TOL_FP64 = 1e-12
TOL_FP32 = 1e-5


def _modelo(arena="looming", estimulo=None, passos=0):
    from physics import cria
    from physics.adapter import MotorFrame

    corpo = cria("flygym2-mujoco", arena=arena, estimulo=estimulo or {},
                 timestep=DT, com_visao=False)
    corpo.reset(seed=0)
    for _ in range(passos):
        corpo.passo(MotorFrame(drive=np.ones(2)))
    return corpo


def test_hash_da_ciencia_nao_muda_com_backend():
    from lab.receita import Receita, hash_ciencia

    atual = hash_ciencia()
    assert atual == HASH_CIENCIA_ESPERADO, (
        f"hash_ciencia mudou: {atual} != {HASH_CIENCIA_ESPERADO}. Trabalho de "
        "backend fisico NAO pode mexer em Shiu, transducao, timestep, pesos, "
        "conectoma nem semantica motora. Se a mudanca foi deliberada, diga qual "
        "no commit e atualize este valor; se nao foi, ela e o bug.")

    # o hash tem que sair igual dos dois backends: a receita muda, a ciencia nao
    a = Receita(nome="x", physics="flygym2-mujoco")
    b = Receita(nome="x", physics="drosobot-gpu")
    assert a.physics != b.physics
    assert hash_ciencia() == atual
    print(f"    hash_ciencia {atual} estavel entre "
          f"{a.physics} e {b.physics}")


def test_hash_do_modelo_e_derivado():
    from gpu_physics.compilador import compila, hash_modelo

    corpo = _modelo()
    try:
        m = corpo.sim.mj_model
        a = compila(m)
        b = compila(m)
        assert a.hash_modelo == b.hash_modelo, "o hash nao e estavel"

        # mudar uma massa muda o hash -- sem ninguem lembrar de atualizar nada
        c = compila(m)
        c.arrays["body_mass"][3] += 1e-9
        assert hash_modelo(c) != a.hash_modelo, (
            "mexer numa massa nao mudou o hash do modelo fisico: ele nao esta "
            "cobrindo o que muda a dinamica")

        # mudar uma cor NAO deveria mudar o hash. `geom_rgba` nem entra nos
        # arrays compilados; o teste e que a lista e por dentro, nao por fora.
        assert "geom_rgba" not in a.arrays
        print(f"    hash_modelo {a.hash_modelo} estavel, sensivel a massa, "
              f"cego a cor ({len(a.arrays)} arrays, {a.bytes_totais()/1e6:.1f} MB)")
    finally:
        corpo.fecha()


def test_subconjunto_recusa_o_que_nao_implementa():
    from gpu_physics.compilador import RecursoNaoSuportado, valida

    corpo = _modelo()
    try:
        m = corpo.sim.mj_model
        valida(m)                                   # o modelo real passa

        class Falso:
            """Copia rasa do mjModel com um campo trocado."""

            def __init__(self, base, **mudancas):
                self._base, self._m = base, mudancas

            def __getattr__(self, nome):
                if nome in self._m:
                    return self._m[nome]
                return getattr(self._base, nome)

        # um tendao que o backend nao implementa
        try:
            valida(Falso(m, ntendon=1))
            raise AssertionError("aceitou um modelo com tendao")
        except RecursoNaoSuportado as e:
            assert "tendo" in str(e).lower()

        # uma junta de tipo fora do subset (ball = 1)
        tipos = np.asarray(m.jnt_type).copy()
        tipos[5] = 1
        try:
            valida(Falso(m, jnt_type=tipos))
            raise AssertionError("aceitou uma junta ball")
        except RecursoNaoSuportado as e:
            assert "junta" in str(e).lower()
        print("    recusou tendao e junta ball com o nome do recurso na mensagem")
    finally:
        corpo.fecha()


def test_pilar_barra_a_mosca():
    """
    A regressao do campo de obstaculos: o pilar precisa OBSTRUIR, nao decorar.

    Mede contato de verdade, com a mosca andando pra frente contra o pilar
    central. Contar `npair` nao serviria: antes da correcao os pares existiam no
    XML e os contatos nao aconteciam, porque o modelo nao usa mascaras de
    colisao e os `<pair>` com o pilar faltavam.
    """
    from physics.adapter import MotorFrame

    corpo = _modelo(arena="obstaculos", estimulo={"lado": "centro"})
    try:
        m, d = corpo.sim.mj_model, corpo.sim.mj_data
        geoms_pilar = {g for g in range(m.ngeom) if m.geom_type[g] == 5}
        assert geoms_pilar, "a arena de obstaculos nao tem cilindro nenhum"

        # as mascaras da mosca sao (0,0): a topologia de contato vem dos <pair>
        mosca = [g for g in range(m.ngeom) if m.geom_type[g] == 7]
        assert all(int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0
                   for g in mosca), (
            "algum geom da mosca ganhou mascara de colisao. Se isso for "
            "deliberado, o compilador GPU precisa saber: hoje ele declara que a "
            "topologia deste modelo vem so dos <pair>.")

        pares_com_pilar = sum(
            1 for i in range(m.npair)
            if int(m.pair_geom1[i]) in geoms_pilar
            or int(m.pair_geom2[i]) in geoms_pilar)
        assert pares_com_pilar > 0, (
            "nenhum <pair> liga a mosca ao pilar: sem eles o obstaculo e "
            "decoracao e a mosca atravessa")

        contatos = 0
        motor = MotorFrame(drive=np.ones(2))
        for _ in range(30000):                       # 3 s de mosca
            corpo.passo(motor)
            for c in range(int(d.ncon)):
                g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
                if g1 in geoms_pilar or g2 in geoms_pilar:
                    contatos += 1
        assert contatos > 0, (
            f"3 s de corrida contra o pilar central e ZERO contatos com ele "
            f"({pares_com_pilar} pares declarados). E o modo de falha original: "
            "o obstaculo esta visivel para a retina e inexistente para a "
            "fisica.")
        print(f"    {pares_com_pilar} pares com o pilar, {contatos} contatos em 3 s")
    finally:
        corpo.fecha()


def test_cinematica_da_gpu_bate_com_o_mujoco():
    import mujoco as mj

    from gpu_physics.compilador import compila

    try:
        from gpu_physics.cinematica import CinematicaGPU
        from gpu_physics.device import Device, DeviceIndisponivel
        dev = Device()
    except Exception as e:                                    # noqa: BLE001
        print(f"    pulado: sem device OpenCL utilizavel ({type(e).__name__})")
        return

    # com a mosca assentada e andando: parada no ar, metade dos atalhos de
    # quaternio identidade do MuJoCo dispara e o teste mede menos do que parece
    corpo = _modelo(passos=200)
    try:
        m, d = corpo.sim.mj_model, corpo.sim.mj_data
        mod = compila(m)
        mj.mj_kinematics(m, d)
        campos = ("xpos", "xquat", "xmat", "xipos", "ximat",
                  "xanchor", "xaxis", "geom_xpos", "geom_xmat")
        ref = {k: np.asarray(getattr(d, k)).copy() for k in campos}

        for fp64, tol in ((True, TOL_FP64), (False, TOL_FP32)):
            gpu = CinematicaGPU(mod, dev=dev, fp64=fp64)
            gpu.escreve_estado(d.qpos, d.mocap_pos, d.mocap_quat)
            gpu.passo()
            pior, onde = 0.0, ""
            for k, v in ref.items():
                got = gpu.le(k).reshape(v.shape).astype(np.float64)
                e = float(np.abs(got - v).max())
                if e > pior:
                    pior, onde = e, k
            assert pior < tol, (
                f"cinematica fp{'64' if fp64 else '32'} divergiu {pior:.2e} em "
                f"{onde} (limite {tol:.0e})")
            print(f"    fp{'64' if fp64 else '32'}: erro maximo {pior:.2e} "
                  f"em {onde}")
    finally:
        corpo.fecha()


if __name__ == "__main__":
    falhas = 0
    for nome, fn in sorted(globals().items()):
        if not nome.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok    {nome}")
        except Exception as e:                                # noqa: BLE001
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print("\n" + ("todos passaram" if not falhas else f"{falhas} falha(s)"))
    sys.exit(1 if falhas else 0)
