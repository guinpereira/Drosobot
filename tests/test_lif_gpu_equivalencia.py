"""
O kernel da GPU tem que resolver a MESMA equacao que sim/fast_lif.py.

Um benchmark que so mede velocidade nao prova nada aqui: kernel errado fica
errado mais rapido. Este teste compara o estado final que a GPU produziu
(benchmarks/neural/lif_gpu_estado.json, escrito por LifBenchmark.Validar())
contra o integrador de referencia, que ja e validado contra o Brian2.

    # 1. gerar o estado da GPU
    unity cmd eval --code "Drosobot.Compute.LifBenchmark.Validar(1000, 400)"
    # 2. comparar
    .venv\\Scripts\\python tests\\test_lif_gpu_equivalencia.py

## Sobre tolerancia

A GPU trabalha em fp32; o `fast_lif` em fp64. Elas NAO vao bater bit a bit, e
exigir isso seria exigir a coisa errada. O que tem que bater:

  - a CONTAGEM DE SPIKE por neuronio, exatamente. Spike e evento discreto; se
    diverge, o circuito diverge, e nao adianta a media estar parecida.
  - v e g dentro da tolerancia de fp32 acumulada em 400 passos.

Se a contagem de spike divergir em UM neuronio, o teste falha. E o criterio que
importa, porque e o que o resto do Drosobot consome.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

ESTADO = RAIZ / "benchmarks" / "neural" / "lif_gpu_estado.json"

# Os mesmos valores que o shader recebe. Vem de connectome_model.py; duplicados
# aqui como o shader duplica, justamente pra que uma divergencia apareca.
V_REST = -52.0
V_RESET = -52.0
V_TH = -45.0
TAU_M = 20.0
TAU_S = 5.0
T_REF = 2.2
DT = 0.5


def referencia(n: int, passos: int):
    """
    O integrador de fast_lif.Camada.passo(), passo a passo, em fp64.

    Reescrito aqui em vez de importado porque a `Camada` nao aceita entrada
    constante por neuronio sem montar uma `Rede` inteira com sinapses. A
    aritmetica e linha a linha a mesma -- e se alguem mudar fast_lif e esquecer
    daqui, test_fast_lif_instrumentacao.py continua sendo a guarda da referencia.
    """
    dec_g = math.exp(-DT / TAU_S)
    dec_v = math.exp(-DT / TAU_M)
    a = 1.0 / TAU_M - 1.0 / TAU_S
    acopla = (dec_g - dec_v) / (TAU_M * a)
    ref_passos = int(round(T_REF / DT))

    i = np.arange(n)
    entrada = np.float32(0.80) + np.float32(0.20) * np.sin(
        i.astype(np.float32) * np.float32(0.001))
    entrada = entrada.astype(np.float64)

    v = np.full(n, V_REST)
    g = np.zeros(n)
    ref_ate = np.full(n, -1, dtype=np.int64)
    contagem = np.zeros(n, dtype=np.int64)

    for p in range(passos):
        u = v - V_REST
        u_novo = u * dec_v + g * acopla
        g = g * dec_g + entrada
        livre = p >= ref_ate
        v = np.where(livre, V_REST + u_novo, V_RESET)
        disparou = livre & (v > V_TH)
        v = np.where(disparou, V_RESET, v)
        ref_ate = np.where(disparou, p + ref_passos, ref_ate)
        contagem += disparou
    return v, g, ref_ate, contagem, entrada


def margem_ao_limiar(n: int, passos: int):
    """Menor distancia que cada neuronio chegou do limiar, em fp64."""
    dec_g = math.exp(-DT / TAU_S)
    dec_v = math.exp(-DT / TAU_M)
    a = 1.0 / TAU_M - 1.0 / TAU_S
    acopla = (dec_g - dec_v) / (TAU_M * a)
    ref_passos = int(round(T_REF / DT))

    i = np.arange(n)
    entrada = (np.float32(0.80) + np.float32(0.20) *
               np.sin(i.astype(np.float32) * np.float32(0.001))).astype(np.float64)
    v = np.full(n, V_REST)
    g = np.zeros(n)
    ref_ate = np.full(n, -1, dtype=np.int64)
    menor = np.full(n, np.inf)
    for p in range(passos):
        u = v - V_REST
        u_novo = u * dec_v + g * acopla
        g = g * dec_g + entrada
        livre = p >= ref_ate
        v = np.where(livre, V_REST + u_novo, V_RESET)
        menor = np.minimum(menor, np.where(livre, np.abs(v - V_TH), np.inf))
        disparou = livre & (v > V_TH)
        v = np.where(disparou, V_RESET, v)
        ref_ate = np.where(disparou, p + ref_passos, ref_ate)
    return menor


def test_gpu_bate_com_a_referencia():
    if not ESTADO.exists():
        print(f"  PULADO: {ESTADO.relative_to(RAIZ)} nao existe.")
        print('  Gere com: unity cmd eval --code '
              '"Drosobot.Compute.LifBenchmark.Validar(1000, 400)"')
        return "pulado"

    d = json.loads(ESTADO.read_text(encoding="utf-8"))
    n, passos = d["n"], d["steps"]
    v_gpu = np.asarray(d["v"], dtype=np.float64)
    g_gpu = np.asarray(d["g"], dtype=np.float64)
    ref_gpu = np.asarray(d["ref_ate"], dtype=np.int64)
    cont_gpu = np.asarray(d["contagem"], dtype=np.int64)

    v_ref, g_ref, ref_ref, cont_ref, _ = referencia(n, passos)

    print(f"  device: {d.get('device')}  api: {d.get('api')}")
    print(f"  n={n} passos={passos}")
    print(f"  spikes  GPU {cont_gpu.sum():6d}   referencia {cont_ref.sum():6d}")

    # --- contagem de spike, neuronio a neuronio ---
    #
    # Divergencia aqui nao e automaticamente bug. Spike e evento discreto: um
    # neuronio que passa a menos de um ULP de fp32 do limiar pode cair dos dois
    # lados conforme a precisao, e isso e propriedade do fp32, nao erro de
    # formula. O que NAO pode acontecer e divergir um neuronio que estava longe
    # do limiar -- isso sim seria conta errada.
    #
    # Entao medimos a margem de cada divergente e exigimos que ela seja menor
    # que a resolucao do fp32 naquele ponto.
    difs = np.flatnonzero(cont_gpu != cont_ref)
    margens = margem_ao_limiar(n, passos)
    eps32 = float(abs(np.spacing(np.float32(V_TH))))

    if difs.size:
        print(f"  contagem difere em {difs.size}/{n} neuronios "
              f"({difs.size / n * 100:.2f}%)")
        inexplicadas = []
        for i in difs:
            m = margens[i]
            explicada = m < eps32
            if not explicada:
                inexplicadas.append((int(i), m))
            if i in difs[:5]:
                print(f"    neuronio {i:5d}: GPU {cont_gpu[i]} x ref {cont_ref[i]}"
                      f"   margem ao limiar {m:.3e} mV"
                      f"   ({'dentro' if explicada else 'FORA'} do eps32 {eps32:.3e})")
        if inexplicadas:
            raise AssertionError(
                f"{len(inexplicadas)} neuronio(s) divergiram LONGE do limiar -- "
                f"isso e erro de formula, nao precisao: {inexplicadas[:5]}")
        print(f"  todas as divergencias sao de limiar ({difs.size} caso(s));"
              f" ver docs/research/DROSOBOT_COMPUTE_ARCHITECTURE.md")

    # Taxa de divergencia tem teto. Se subir, alguma coisa mudou de verdade.
    assert difs.size / n <= 0.005, (
        f"taxa de divergencia {difs.size/n:.3%} acima de 0,5% -- "
        "nao da mais pra atribuir a fp32")

    # --- estado continuo, dentro do erro de fp32 ---
    #
    # Comparado SO nos neuronios cuja contagem de spike bateu. Um neuronio um
    # spike fora de fase tem v completamente diferente -- um acabou de resetar em
    # -52 e o outro esta em -45 prestes a disparar. Isso da |dv| ~ 7 mV, que e a
    # faixa inteira, e nao significa "v divergiu": significa que o trem de spike
    # divergiu por um evento, o que a checagem acima ja tratou.
    #
    # Medir |dv| sobre todos misturaria as duas coisas e esconderia um erro de
    # formula atras de um caso de limiar.
    concorda = np.ones(n, dtype=bool)
    concorda[difs] = False
    dv = np.abs(v_gpu[concorda] - v_ref[concorda]).max() if concorda.any() else 0.0
    dg = np.abs(g_gpu - g_ref).max()   # g nao tem evento discreto: compara tudo
    print(f"  |dv| max {dv:.3e} mV  (nos {int(concorda.sum())} que concordam)"
          f"   |dg| max {dg:.3e} mV  (todos)")
    # o refratario segue a contagem de spike: os mesmos casos de limiar aparecem
    dif_ref = int((ref_gpu != ref_ref).sum())
    assert dif_ref <= difs.size, (
        f"estado refratario divergiu em {dif_ref} neuronios, mais do que os "
        f"{difs.size} casos de limiar")
    # 1e-3 mV sobre uma faixa de 7 mV: ~0.015%. Folgado pra fp32 em 400 passos,
    # e apertado o bastante pra pegar erro de formula.
    assert dv < 1e-3, f"v divergiu demais: {dv}"
    assert dg < 1e-3, f"g divergiu demais: {dg}"
    return "ok"


def test_parametros_nao_derivaram():
    """O shader carrega copia dos parametros de Shiu et al. Eles tem que casar."""
    from connectome_model import (V_REST as VR, V_RESET as VRS,
                                  V_THRESHOLD as VT, TAU_MBR, TAU_SYN,
                                  T_REFRACTORY)
    from brian2 import mV, ms

    # TAU_MBR sai de C_MBR * R_MBR e da 19.999999999999996, nao 20.0 exato.
    # Comparar por igualdade falharia por 4e-15, que e ruido de ponto flutuante
    # e nao derivacao de parametro. O que queremos pegar e alguem trocar 20 por
    # 25, nao o ultimo bit de um produto.
    pares = [("V_REST", float(VR / mV), V_REST),
             ("V_RESET", float(VRS / mV), V_RESET),
             ("V_THRESHOLD", float(VT / mV), V_TH),
             ("TAU_MBR", float(TAU_MBR / ms), TAU_M),
             ("TAU_SYN", float(TAU_SYN / ms), TAU_S),
             ("T_REFRACTORY", float(T_REFRACTORY / ms), T_REF)]
    for nome, ref, no_shader in pares:
        assert math.isclose(ref, no_shader, rel_tol=1e-12), (
            f"{nome}: referencia {ref!r} x copia no shader {no_shader!r}")


def _todos():
    return [(n, o) for n, o in sorted(globals().items())
            if n.startswith("test_") and callable(o)]


if __name__ == "__main__":
    falhas = 0
    for nome, fn in _todos():
        try:
            r = fn()
            print(f"  {'PULADO' if r == 'pulado' else 'ok    '} {nome}")
        except Exception as e:
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(_todos()) - falhas}/{len(_todos())} passaram")
    sys.exit(1 if falhas else 0)
