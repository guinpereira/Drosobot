r"""
O que o backend de fisica nao pode quebrar.

    .venv-flygym2\Scripts\python tests/test_gpu_physics.py

Doze propriedades, doze motivos distintos. Nao ha um decimo terceiro teste
medindo a mesma coisa por outro angulo -- isso so faria a suite demorar mais e
falhar junto.

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

# A cadeia da dinamica suave acumula arredondamento por ~dez etapas; o limite e
# relativo ao maior valor do campo, nao absoluto.
TOL_DINAMICA = 1e-12


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


def test_versao_do_mujoco_e_a_portada():
    import mujoco as mj

    from gpu_physics.compilador import (VERSAO_MUJOCO_PORTADA,
                                        VersaoMuJoCoDivergente, confere_versao)

    assert confere_versao() == VERSAO_MUJOCO_PORTADA
    try:
        # a guarda existe para FALHAR; um teste que so confere o caminho feliz
        # nao prova que ela dispara
        import gpu_physics.compilador as comp
        antigo = comp.VERSAO_MUJOCO_PORTADA
        comp.VERSAO_MUJOCO_PORTADA = "0.0.0"
        try:
            comp.confere_versao()
            raise AssertionError("a guarda de versao nao disparou")
        except VersaoMuJoCoDivergente:
            pass
        finally:
            comp.VERSAO_MUJOCO_PORTADA = antigo
    finally:
        pass
    print(f"    runtime mujoco {mj.__version__} == portado "
          f"{VERSAO_MUJOCO_PORTADA}, e a guarda dispara quando nao e")


def _motor_e_referencia(arena="looming", passos=200, sem_adesao=True):
    """Estado comum dos testes de dinamica: MuJoCo em `mj_forward`, GPU igual."""
    import mujoco as mj

    from gpu_physics.compilador import compila
    from gpu_physics.dinamica import MotorFisicoGPU
    from gpu_physics.device import Device

    corpo = _modelo(arena=arena, passos=passos)
    m, d = corpo.sim.mj_model, corpo.sim.mj_data
    if sem_adesao:
        # a adesao tira o momento dela dos CONTATOS; enquanto o backend nao
        # monta a Jacobiana de restricao, comparar com ela ligada mediria uma
        # etapa que ainda nao existe
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) == 5:
                d.ctrl[a] = 0.0
    mj.mj_forward(m, d)
    mod = compila(m)
    g = MotorFisicoGPU(mod, dev=Device(), fp64=True)
    g.escreve_estado(qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
                     mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
    g.forward()
    return corpo, m, d, g


def test_dinamica_suave_bate_com_o_mujoco():
    try:
        from gpu_physics.device import Device
        Device()
    except Exception as e:                                    # noqa: BLE001
        print(f"    pulado: sem device OpenCL ({type(e).__name__})")
        return

    corpo, m, d, g = _motor_e_referencia()
    try:
        cadeia = (
            ("subtree_com", d.subtree_com), ("cinert", d.cinert),
            ("cdof", d.cdof), ("crb", d.crb), ("M", d.M),
            ("qLD", d.qLD), ("cvel", d.cvel), ("cdof_dot", d.cdof_dot),
            ("qfrc_passive", d.qfrc_passive), ("qfrc_bias", d.qfrc_bias),
            ("actuator_force", d.actuator_force),
            ("qfrc_actuator", d.qfrc_actuator),
            ("qfrc_smooth", d.qfrc_smooth), ("qacc_smooth", d.qacc_smooth),
        )
        pior, onde = 0.0, ""
        for nome, ref in cadeia:
            ref = np.asarray(ref, dtype=np.float64)
            got = g.le(nome).reshape(ref.shape)
            escala = max(1e-30, float(np.abs(ref).max()))
            rel = float(np.abs(got - ref).max()) / escala
            assert rel < TOL_DINAMICA, (
                f"{nome} divergiu {rel:.2e} (limite {TOL_DINAMICA:.0e}). A "
                "cadeia e sequencial: o primeiro campo a falhar e a causa, os "
                "seguintes herdam.")
            if rel > pior:
                pior, onde = rel, nome
        print(f"    14 campos, de subtree_com a qacc_smooth: pior "
              f"{pior:.2e} em {onde}")
    finally:
        corpo.fecha()


def test_contatos_batem_com_o_mujoco():
    try:
        from gpu_physics.device import Device
        Device()
    except Exception as e:                                    # noqa: BLE001
        print(f"    pulado: sem device OpenCL ({type(e).__name__})")
        return

    corpo, m, d, g = _motor_e_referencia()
    try:
        con = g.contatos()
        assert int(d.ncon) > 0, (
            "a mosca nao esta tocando o chao: o teste passaria com zero "
            "contatos dos dois lados e nao provaria nada")
        assert con["ncon"] == int(d.ncon), (
            f"numero de contatos difere: GPU {con['ncon']}, MuJoCo {d.ncon}")
        pior_d = pior_p = 0.0
        for i in range(con["ncon"]):
            gg = (int(con["geom"][i][0]), int(con["geom"][i][1]))
            gm = (int(d.contact.geom1[i]), int(d.contact.geom2[i]))
            assert gg == gm, (
                f"contato {i}: geoms {gg} != {gm}. A ORDEM dos contatos e a "
                "ordem das linhas de restricao; trocar duas nao e inocente.")
            pior_d = max(pior_d, abs(float(con["dist"][i]) - float(d.contact.dist[i])))
            pior_p = max(pior_p, float(np.abs(
                con["pos"][i] - np.asarray(d.contact.pos[i])).max()))
        assert pior_d < 1e-12 and pior_p < 1e-12, (
            f"contatos divergiram: dist {pior_d:.2e}, pos {pior_p:.2e}")
        print(f"    {con['ncon']} contatos, mesma ordem: "
              f"dist {pior_d:.1e}, pos {pior_p:.1e}")
    finally:
        corpo.fecha()


def test_restricoes_batem_com_o_mujoco():
    try:
        from gpu_physics.device import Device
        Device()
    except Exception as e:                                    # noqa: BLE001
        print(f"    pulado: sem device OpenCL ({type(e).__name__})")
        return

    corpo, m, d, g = _motor_e_referencia()
    try:
        e = g.efc()
        assert e["nefc"] == int(d.nefc) and e["nefc"] > 0, (
            f"nefc difere: GPU {e['nefc']}, MuJoCo {d.nefc}")
        # a efc_J do MuJoCo e esparsa quando nv >= 60; densifica para comparar
        nefc, nv = int(d.nefc), int(m.nv)
        J = np.zeros((nefc, nv))
        rownnz = np.asarray(d.efc_J_rownnz)[:nefc]
        rowadr = np.asarray(d.efc_J_rowadr)[:nefc]
        colind = np.asarray(d.efc_J_colind)
        vals = np.asarray(d.efc_J)
        for i in range(nefc):
            a0, n = int(rowadr[i]), int(rownnz[i])
            J[i, colind[a0:a0 + n]] = vals[a0:a0 + n]

        for nome, got, ref in (("efc_J", e["J"], J),
                               ("efc_R", e["R"], d.efc_R),
                               ("efc_D", e["D"], d.efc_D),
                               ("efc_aref", e["aref"], d.efc_aref)):
            ref = np.asarray(ref, dtype=np.float64).reshape(np.shape(got))
            esc = max(1e-30, float(np.abs(ref).max()))
            rel = float(np.abs(got - ref).max()) / esc
            assert rel < 1e-11, f"{nome} divergiu {rel:.2e}"
        assert np.array_equal(e["id"], np.asarray(d.efc_id)[:e["nefc"]]), (
            "efc_id difere: a ordem das linhas e a ordem dos contatos")
        print(f"    {e['nefc']} linhas, mesma ordem; R e D exatos")
    finally:
        corpo.fecha()


def test_adesao_bate_com_o_mujoco():
    try:
        from gpu_physics.device import Device
        Device()
    except Exception as e:                                    # noqa: BLE001
        print(f"    pulado: sem device OpenCL ({type(e).__name__})")
        return

    # adesao LIGADA: e como os experimentos rodam, e o momento dela depende dos
    # contatos, entao desligar tornaria o teste vazio
    corpo, m, d, g = _motor_e_referencia(sem_adesao=False)
    try:
        ades = [a for a in range(m.nu) if int(m.actuator_trntype[a]) == 5]
        assert ades and float(np.abs(np.asarray(d.ctrl)[ades]).max()) > 0, (
            "nenhum atuador de adesao acionado: o teste passaria com zeros")
        for nome, ref in (("actuator_force", d.actuator_force),
                          ("qfrc_actuator", d.qfrc_actuator)):
            ref = np.asarray(ref, dtype=np.float64)
            got = g.le(nome).reshape(ref.shape)
            esc = max(1e-30, float(np.abs(ref).max()))
            rel = float(np.abs(got - ref).max()) / esc
            assert rel < 1e-12, f"{nome} divergiu {rel:.2e}"
        print(f"    {len(ades)} atuadores de adesao, ctrl=1: "
              f"actuator_force e qfrc_actuator batem")
    finally:
        corpo.fecha()


def test_solver_chega_ao_minimo():
    """
    Estacionariedade, nao igualdade com o MuJoCo.

    O problema e estritamente convexo e C1, entao a propriedade que define a
    resposta certa e `grad = 0` -- verificavel aqui mesmo, sem depender de uma
    segunda implementacao. Medido: o MuJoCo as vezes para longe disso, e exigir
    igualdade com ele faria da parada antecipada dele a especificacao.
    """
    try:
        from gpu_physics.device import Device
        Device()
    except Exception as e:                                    # noqa: BLE001
        print(f"    pulado: sem device OpenCL ({type(e).__name__})")
        return

    corpo, m, d, g = _motor_e_referencia(sem_adesao=False)
    try:
        nv = int(m.nv)
        nefc = int(g.le_int("nefc")[0])
        assert nefc > 0, "sem restricao ativa: o teste nao provaria nada"
        J = g.le("efc_J")[:nefc * nv].reshape(nefc, nv)
        D = g.le("efc_D")[:nefc]
        aref = g.le("efc_aref")[:nefc]
        Md = g.le("Md").reshape(nv, nv)
        qs = g.le("qacc_smooth")

        def grad(a):
            neg = np.minimum(J @ a - aref, 0.0)
            return Md @ (a - qs) + J.T @ (D * neg)

        g_gpu = float(np.linalg.norm(grad(g.le("qacc"))))
        # normaliza pela escala do problema, senao o limiar vira arbitrario
        escala = float(np.linalg.norm(Md @ (g.le("qacc") - qs))) + 1.0
        assert g_gpu / escala < 1e-12, (
            f"o solver nao chegou ao minimo: |grad|/escala = {g_gpu/escala:.2e}")
        g_mj = float(np.linalg.norm(grad(np.asarray(d.qacc))))
        print(f"    |grad| GPU {g_gpu:.2e}  (MuJoCo no mesmo objetivo: "
              f"{g_mj:.2e})")
    finally:
        corpo.fecha()


def test_passo_completo_e_dois_seguidos():
    """
    Dois passos, nao um: o primeiro esconde bug de buffer residente.

    Adesao desligada de proposito -- com ela o solver do MuJoCo para antes do
    minimo ja no primeiro passo (ver `test_solver_chega_ao_minimo`) e a
    trajetoria separa por motivo que nao e o porte. Sem ela os dois solvers
    concordam, e a comparacao mede a cadeia inteira.
    """
    import mujoco as mj

    try:
        from gpu_physics.device import Device
        Device()
    except Exception as e:                                    # noqa: BLE001
        print(f"    pulado: sem device OpenCL ({type(e).__name__})")
        return

    corpo, m, d, g = _motor_e_referencia()
    try:
        q0 = np.array(d.qpos, copy=True)
        v0 = np.array(d.qvel, copy=True)
        c0 = np.array(d.ctrl, copy=True)
        g.escreve_estado(qpos=q0, qvel=v0, ctrl=c0,
                         mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
        d.qpos[:] = q0
        d.qvel[:] = v0
        d.ctrl[:] = c0
        for k in (1, 2):
            g.passo()
            mj.mj_step(m, d)
            eq = float(np.abs(g.le("qpos") - np.asarray(d.qpos)).max())
            ev = float(np.abs(g.le("qvel") - np.asarray(d.qvel)).max())
            ncon_g = int(g.le_int("ncon")[0])
            assert ncon_g == int(d.ncon), (
                f"passo {k}: ncon {ncon_g} != {int(d.ncon)}")
            assert eq < 1e-13 and ev < 1e-11, (
                f"passo {k}: dqpos {eq:.2e}, dqvel {ev:.2e}")
            print(f"    passo {k}: dqpos {eq:.1e}  dqvel {ev:.1e}  "
                  f"ncon {ncon_g}")
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
            assert gpu.usa_registradores, (
                "o work-group nao cobre nbody/ngeom nesta placa; o caminho "
                "normal nao esta sendo exercitado")
            marca = "64" if fp64 else "32"
            for variante, roda in (("registradores", gpu.passo),
                                   ("modelo global", gpu.passo_lds)):
                roda()
                pior, onde = 0.0, ""
                for k, v in ref.items():
                    got = gpu.le(k).reshape(v.shape).astype(np.float64)
                    e = float(np.abs(got - v).max())
                    if e > pior:
                        pior, onde = e, k
                assert pior < tol, (
                    f"cinematica fp{marca} ({variante}) divergiu {pior:.2e} em "
                    f"{onde} (limite {tol:.0e})")
                print(f"    fp{marca} {variante:<14} erro maximo {pior:.2e} "
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
