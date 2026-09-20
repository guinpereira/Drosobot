"""
O caminho rapido produz os MESMOS numeros que o caminho original.

Este teste e a licenca pra usar `sim/physics/fastpath.py`. Sem ele, "otimizei o
laco" e uma afirmacao; com ele, e uma medicao.

Exigencia: **igualdade exata**, nao tolerancia. O caminho rapido nao aproxima
nada -- ele so deixa de refazer trabalho constante. Se aparecer diferenca, ainda
que no ultimo bit, ha um erro de replicacao e o certo e reprovar, nao afrouxar o
limiar. Uma diferenca de 1 ULP por passo, realimentada 10.000 vezes por segundo
simulado num sistema com contato, diverge.

    .venv-flygym2\\Scripts\\python tests/test_fastpath_equivalencia.py
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

DT = 1e-4
PASSOS = 2000          # 200 ms: passa por swing e stance das seis pernas


def _monta(arena="looming"):
    from physics import cria

    corpo = cria("flygym2", arena=arena, self_collisions="legs",
                 timestep=DT, com_visao=False)
    corpo.reset(seed=0)
    return corpo


def test_controlador_rapido_bate_bit_a_bit():
    """
    Dois controladores, o mesmo estado inicial, a mesma sequencia de entradas.

    As duas simulacoes avancam em paralelo e trocam a acao entre si a cada
    passo? Nao -- cada uma avanca a sua, e a comparacao e da ACAO produzida a
    partir de observacoes identicas. Para isso as duas moscas tem que estar no
    mesmo estado, e e por isso que a acao aplicada e sempre a do caminho
    original: se divergissem no corpo, a diferenca na acao nao diria nada sobre
    o controlador.
    """
    from flygym.compose import ActuatorType
    from flygym_demo.complex_terrain import apply_locomotion_action
    from physics.fastpath import ControladorRapido

    corpo = _monta()
    ctl = corpo.controlador
    rapido = ControladorRapido(ctl)
    drive = np.array([1.0, 1.0])

    piores = {"angulos": 0.0, "adesao": 0}
    try:
        for passo in range(PASSOS):
            corpo.antes_do_passo(passo * DT)
            obs = corpo._observacao()

            # O estado do controlador AVANCA em cada chamada (cpg_network,
            # contadores). Entao guardamos, rodamos o rapido, restauramos, e
            # rodamos o original -- as duas chamadas partem do mesmo estado.
            estado = _congela(ctl)
            acao_rapida = rapido.step(drive, obs)
            _descongela(ctl, estado)
            acao_ref = ctl.step(drive, obs)

            d = np.abs(np.asarray(acao_rapida.joint_angles)
                       - np.asarray(acao_ref.joint_angles)).max()
            piores["angulos"] = max(piores["angulos"], float(d))
            dif_ad = int((np.asarray(acao_rapida.adhesion_onoff)
                          != np.asarray(acao_ref.adhesion_onoff)).sum())
            piores["adesao"] += dif_ad
            assert d == 0.0, (
                f"passo {passo}: angulos divergem em {d:.3e} rad")
            assert dif_ad == 0, f"passo {passo}: adesao divergiu"

            apply_locomotion_action(corpo.sim, corpo.fly.name, acao_ref,
                                    actuator_type=ActuatorType.POSITION)
            corpo.sim.step()
            corpo._passo += 1
    finally:
        corpo.fecha()

    print(f"    controlador: {PASSOS} passos, diferenca maxima "
          f"{piores['angulos']:.1e} rad, adesao divergente {piores['adesao']}x")


def _congela(ctl):
    """Fotografia do estado mutavel do controlador."""
    cpg = ctl.cpg_network
    return {
        "fases": np.array(cpg.curr_phases, copy=True),
        "mags": np.array(cpg.curr_magnitudes, copy=True),
        "retr": np.array(ctl.retraction_correction, copy=True),
        "stum": np.array(ctl.stumbling_correction, copy=True),
        "cont": np.array(ctl.retraction_persistence_counter, copy=True),
    }


def _descongela(ctl, e):
    cpg = ctl.cpg_network
    cpg.curr_phases[:] = e["fases"]
    cpg.curr_magnitudes[:] = e["mags"]
    ctl.retraction_correction[:] = e["retr"]
    ctl.stumbling_correction[:] = e["stum"]
    ctl.retraction_persistence_counter[:] = e["cont"]


def test_forcas_de_contato_batem_bit_a_bit():
    """A mesma forca, com as tabelas resolvidas uma vez em vez de por passo."""
    from flygym.compose import ActuatorType
    from flygym_demo.complex_terrain import apply_locomotion_action
    from physics.fastpath import ForcasContato

    corpo = _monta()
    # Os elos de tropeco (tibia, tarsus1, tarsus2) NAO encostam no chao neste
    # experimento -- medido: 0 passos com forca nao nula em 6.000. Testar so
    # com eles compararia zero com zero e nao provaria nada. Entao o teste usa
    # os DOIS conjuntos: o tarsus5, que e quem pisa, pra exercitar a conta; e o
    # de tropeco, que e o que o runtime de fato pede.
    from flygym.anatomy import BodySegment
    pisantes = [BodySegment(f"{p}_tarsus5") for p in corpo.controlador.legs]
    conjuntos = [("tarsus5", pisantes),
                 ("tropeco", list(corpo._segs_stumbling))]
    leitores = {nome: ForcasContato(corpo.sim, corpo.fly.name, segs,
                                    ground_only=True)
                for nome, segs in conjuntos}
    drive = np.array([1.0, 1.0])
    pior, com_contato = 0.0, 0
    try:
        for passo in range(PASSOS):
            corpo.antes_do_passo(passo * DT)
            for nome, segs in conjuntos:
                ref = corpo.sim.get_bodysegment_contact_forces(
                    corpo.fly.name, segs, ground_only=True)
                rap = leitores[nome].le().copy()
                d = float(np.abs(ref - rap).max())
                pior = max(pior, d)
                if nome == "tarsus5" and np.abs(ref).max() > 0:
                    com_contato += 1
                assert d == 0.0, (
                    f"passo {passo}, {nome}: forcas divergem em {d:.3e}")

            obs = corpo._observacao()
            acao = corpo.controlador.step(drive, obs)
            apply_locomotion_action(corpo.sim, corpo.fly.name, acao,
                                    actuator_type=ActuatorType.POSITION)
            corpo.sim.step()
            corpo._passo += 1
    finally:
        corpo.fecha()

    assert com_contato > 0, (
        "nem o tarsus5 encostou no chao -- o teste passaria com zeros e nao "
        "provaria nada")
    print(f"    contato: {PASSOS} passos, {com_contato} com forca nao nula, "
          f"diferenca maxima {pior:.1e}")


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
