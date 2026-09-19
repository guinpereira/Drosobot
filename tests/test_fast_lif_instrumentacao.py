"""
A instrumentacao nao pode mudar a ciencia.

O ponto destes testes: ligar telemetria/inspecao tem que produzir EXATAMENTE o
mesmo resultado neural que rodar sem. Uma visualizacao que altera o que esta
visualizando nao vale nada, e o jeito de garantir isso e comparar bit a bit com
a mesma seed -- nao "parecido", identico.

Tambem checa que os parametros de Shiu et al. continuam intactos, porque e facil
alguem "ajustar" um limiar sem perceber o que esta fazendo.

    .venv\\Scripts\\python tests/test_fast_lif_instrumentacao.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

import fast_lif  # noqa: E402
from connectome_model import (  # noqa: E402
    CONNECTOME, TAU_MBR, TAU_SYN, T_REFRACTORY, V_REST, V_RESET, V_THRESHOLD,
    W_SYN, load_properties, signed_weights,
)
from brian2 import mV, ms  # noqa: E402


def _circuito_optomotor():
    props = load_properties()
    sh = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
    hd = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
    dm = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")
    sens = sorted(sh["bodyId_pre"].unique().tolist())
    hs = sorted(set(sh["bodyId_post"]) | set(hd["bodyId_pre"]))
    dna = sorted(set(hd["bodyId_post"]) | set(dm["bodyId_pre"]))
    mot = sorted(dm["bodyId_post"].unique().tolist())
    ix = lambda ids: {b: i for i, b in enumerate(ids)}
    conexoes = []
    for conn, si, ti in [(sh, ix(sens), ix(hs)), (hd, ix(hs), ix(dna)), (dm, ix(dna), ix(mot))]:
        agg = signed_weights(conn, props)
        agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
        conexoes.append(([si[b] for b in agg["bodyId_pre"]],
                         [ti[b] for b in agg["bodyId_post"]],
                         agg["w_mV"].values))
    return len(sens), [len(hs), len(dna), len(mot)], conexoes


def test_parametros_shiu_intactos():
    """Se alguem mexer nestes numeros, o projeto deixa de ser o que afirma ser."""
    assert float(V_REST / mV) == -52.0
    assert float(V_RESET / mV) == -52.0
    assert float(V_THRESHOLD / mV) == -45.0
    assert abs(float(TAU_MBR / ms) - 20.0) < 1e-9
    assert float(TAU_SYN / ms) == 5.0
    assert float(T_REFRACTORY / ms) == 2.2
    assert float(W_SYN / mV) == 0.275


def test_observar_nao_altera_o_resultado():
    """
    roda() e roda_observado() com a mesma seed tem que dar o MESMO resultado.

    Esta e a garantia central da Fase 2: a camada de visualizacao observa a
    ciencia existente, nao participa dela.
    """
    n_ent, tamanhos, conexoes = _circuito_optomotor()
    taxa = np.full(n_ent, 150.0)

    rede_a = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    sem = [c.copy() for c in rede_a.roda(300.0, taxa, rng=np.random.default_rng(7))]

    vistos = []
    rede_b = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    com = [c.copy() for c in rede_b.roda_observado(
        300.0, taxa, rng=np.random.default_rng(7),
        observador=lambda t, d: vistos.append((t, [x.copy() for x in d])), stride=1)]

    for k, (a, b) in enumerate(zip(sem, com)):
        assert np.array_equal(a, b), f"camada {k} divergiu: {a.sum()} contra {b.sum()}"
    assert len(vistos) == 600, f"observador chamado {len(vistos)} vezes, esperado 600"


def test_estado_final_identico():
    """Nao basta a contagem bater: v e g tem que terminar iguais tambem."""
    n_ent, tamanhos, conexoes = _circuito_optomotor()
    taxa = np.full(n_ent, 120.0)

    a = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    a.roda(200.0, taxa, rng=np.random.default_rng(3))
    b = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    b.roda_observado(200.0, taxa, rng=np.random.default_rng(3),
                     observador=lambda t, d: None, stride=4)

    for k, (ca, cb) in enumerate(zip(a.camadas, b.camadas)):
        assert np.allclose(ca.v, cb.v), f"camada {k}: v divergiu"
        assert np.allclose(ca.g, cb.g), f"camada {k}: g divergiu"
        assert np.array_equal(ca.ref_ate, cb.ref_ate), f"camada {k}: refratario divergiu"


def test_snapshot_nao_altera_a_rede():
    """Tirar fotografia nao pode mexer no estado."""
    n_ent, tamanhos, conexoes = _circuito_optomotor()
    taxa = np.full(n_ent, 150.0)

    a = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    a.roda(100.0, taxa, rng=np.random.default_rng(11))
    a.snapshot(["HS", "DNa02", "motor"])          # <- no meio
    depois_a = [c.copy() for c in a.roda(100.0, taxa, rng=np.random.default_rng(12))]

    b = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    b.roda(100.0, taxa, rng=np.random.default_rng(11))
    depois_b = [c.copy() for c in b.roda(100.0, taxa, rng=np.random.default_rng(12))]

    for k, (x, y) in enumerate(zip(depois_a, depois_b)):
        assert np.array_equal(x, y), f"camada {k} mudou por causa do snapshot"


def test_snapshot_traz_o_que_promete():
    n_ent, tamanhos, conexoes = _circuito_optomotor()
    rede = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    rede.roda(100.0, np.full(n_ent, 150.0), rng=np.random.default_rng(1))
    snap = rede.snapshot(["HS", "DNa02", "motor"])

    assert len(snap) == len(tamanhos)
    for k, camada in enumerate(snap):
        assert camada["index"] == k
        assert len(camada["v_mV"]) == tamanhos[k]
        assert len(camada["g_mV"]) == tamanhos[k]
        assert len(camada["refratario"]) == tamanhos[k]
        assert len(camada["spikes"]) == tamanhos[k]
        # potencial dentro de faixa fisica: reset <= v <= limiar
        v = camada["v_mV"]
        assert v.min() >= float(V_RESET / mV) - 1e-6
        assert v.max() <= float(V_THRESHOLD / mV) + 1e-6
    assert [c["name"] for c in snap] == ["HS", "DNa02", "motor"]


def test_snapshot_devolve_copias():
    """Quem recebe serializa em outra thread; o laco nao pode escrever por cima."""
    n_ent, tamanhos, conexoes = _circuito_optomotor()
    rede = fast_lif.Rede(n_ent, tamanhos, conexoes, 0.5)
    rede.roda(50.0, np.full(n_ent, 150.0), rng=np.random.default_rng(1))
    snap = rede.snapshot()
    guardado = snap[0]["v_mV"].copy()
    rede.roda(50.0, np.full(n_ent, 150.0), rng=np.random.default_rng(2))
    assert np.array_equal(snap[0]["v_mV"], guardado), "snapshot compartilhou memoria com a rede"


def test_integrador_ainda_confere_com_brian2():
    """A verificacao que ja existia continua valendo depois da instrumentacao."""
    assert fast_lif.verificar(verbose=False)


def _todos():
    return [(n, o) for n, o in sorted(globals().items())
            if n.startswith("test_") and callable(o)]


if __name__ == "__main__":
    falhas = 0
    for nome, fn in _todos():
        try:
            fn()
            print(f"  ok    {nome}")
        except Exception as e:
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(_todos()) - falhas}/{len(_todos())} passaram")
    sys.exit(1 if falhas else 0)
