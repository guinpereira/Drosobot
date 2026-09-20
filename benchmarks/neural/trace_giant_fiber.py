"""
Por que o Giant Fiber dispara no circuito e nao dispara no CNS inteiro.

    .venv\\Scripts\\python benchmarks\\neural\\trace_giant_fiber.py

O runtime integrado mostrou 4 fugas no escopo de circuito e 0 no conectoma
inteiro, com o mesmo estimulo. Antes de chamar isso de resultado cientifico e
preciso MEDIR a causa -- "nao e bug" e uma afirmacao, e ela precisa de numero.

## Como a comparacao e feita justa

A MESMA realizacao de Poisson nos dois escopos: a mascara de spike sensorial e
sorteada uma vez, guardada, e reaplicada. Sem isso estariamos comparando dois
estimulos diferentes e qualquer diferenca seria ruido.

Pra cada escopo, por passo, medimos no DNp01 (o Giant Fiber):

    excitacao que chegou   (soma das contribuicoes de peso > 0)
    inibicao que chegou    (soma das de peso < 0)
    liquido
    potencial de membrana
    spikes

E identificamos quais populacoes pre-sinapticas produzem a diferenca.

## O que este script NAO faz

Nao ajusta estimulo, limiar, peso, inibicao nem remove neuronio. Se a conclusao
for que o conectoma completo silencia o GF, isso e um resultado do modelo.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

from neural import NeuralEngine, carrega_male_cns, subgrafo  # noqa: E402
from neural.model import Coeficientes  # noqa: E402
from connectome_model import CONNECTOME, load_properties, gf_input_population  # noqa: E402

GF_IDS = [10001, 10010]
TAXA_LOOMING_HZ = 20.0     # o pico que a retina produz nos experimentos
JANELA_MS = 400.0
SEMENTE = 7


def monta(escopo: str, c_full, props, loom_ids, sensor_ids, motor_ids):
    c = c_full
    if escopo == "circuito":
        alvo = np.unique(np.concatenate([np.asarray(sensor_ids),
                                         np.asarray(GF_IDS),
                                         np.asarray(motor_ids)]))
        idx = c_full.indice_de(alvo)
        c = subgrafo(c_full, idx[idx >= 0])
    eng = NeuralEngine(c, backend="opencl")
    return eng, c


def entradas_do_gf(c, idx_gf):
    """
    Quem entra no GF, com peso e sinal. Percorre o CSR procurando quem tem o GF
    como alvo -- e in-edges, que o CSR por linha (out-edges) nao da de graca.
    """
    alvo = set(int(i) for i in idx_gf)
    pre, peso = [], []
    ro, tg, w = c.row_offsets, c.targets, c.weights
    for i in range(c.n):
        a, b = ro[i], ro[i + 1]
        if b <= a:
            continue
        m = np.isin(tg[a:b], list(alvo))
        if m.any():
            pre.append(np.full(int(m.sum()), i, dtype=np.int64))
            peso.append(w[a:b][m])
    if not pre:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(pre), np.concatenate(peso)


def roda_e_mede(eng, c, idx_gf, idx_sens, mascaras, pre, peso):
    """Roda com a realizacao Poisson dada e acumula o balanco no GF."""
    eng.reset()
    exc = inh = 0.0
    v_serie, spk_gf = [], 0
    n_pre_disparos = defaultdict(int)

    for mascara_sens in mascaras:
        m = np.zeros(c.n, dtype=np.uint8)
        m[idx_sens] = mascara_sens
        eng.passo(forcados=m)
        eng.backend.sincroniza()

        # quem disparou entre os pre-sinapticos do GF, e com que peso
        est_pre = eng.le(pre.astype(np.int32)) if len(pre) else None
        if est_pre is not None:
            disp = est_pre.spike.astype(bool)
            if disp.any():
                contrib = peso[disp]
                exc += float(contrib[contrib > 0].sum())
                inh += float(contrib[contrib < 0].sum())
                for p in pre[disp]:
                    n_pre_disparos[int(p)] += 1

        est_gf = eng.le(idx_gf.astype(np.int32))
        v_serie.append(float(est_gf.v_mV.max()))
        spk_gf += int(est_gf.spike.sum())

    return {
        "excitacao_mV": round(exc, 2),
        "inibicao_mV": round(inh, 2),
        "liquido_mV": round(exc + inh, 2),
        "v_max_mV": round(max(v_serie), 3),
        "v_media_mV": round(float(np.mean(v_serie)), 3),
        "spikes_gf": spk_gf,
    }, n_pre_disparos


def main():
    props = load_properties()
    up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
    down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
    sensor_ids, is_loom, _ = gf_input_population(props, up)
    loom_ids = [b for b, f in zip(sensor_ids, is_loom) if f]
    motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()

    c_full = carrega_male_cns()
    coef = Coeficientes.de()

    n_passos = int(round(JANELA_MS / 0.5))

    # --- a MESMA realizacao Poisson pros dois escopos ---
    rng = np.random.default_rng(SEMENTE)
    p = TAXA_LOOMING_HZ * (0.5 / 1000.0)
    mascaras = [(rng.random(len(loom_ids)) < p).astype(np.uint8)
                for _ in range(n_passos)]
    total_spikes_sens = int(sum(m.sum() for m in mascaras))
    print(f"estimulo: {len(loom_ids)} LC4/LPLC2 a {TAXA_LOOMING_HZ} Hz, "
          f"{n_passos} passos ({JANELA_MS} ms)")
    print(f"  {total_spikes_sens} spikes sensoriais na realizacao "
          f"(semente {SEMENTE}, identica nos dois escopos)")
    print()

    saida = {}
    for escopo in ("circuito", "whole"):
        eng, c = monta(escopo, c_full, props, loom_ids, sensor_ids, motor_ids)
        idx_gf = c.indice_de(GF_IDS)
        idx_gf = idx_gf[idx_gf >= 0]
        idx_sens = c.indice_de(loom_ids)
        idx_sens = idx_sens[idx_sens >= 0]
        print(f"== {escopo} ==  {c.n:,} neuronios, {c.e:,} arestas, "
              f"GF em {idx_gf.tolist()}")

        pre, peso = entradas_do_gf(c, idx_gf)
        n_exc = int((peso > 0).sum())
        n_inh = int((peso < 0).sum())
        print(f"  entradas no GF: {len(pre)} arestas "
              f"({n_exc} excitatorias, {n_inh} inibitorias, "
              f"{len(pre)-n_exc-n_inh} mudas)")
        print(f"  peso total: +{peso[peso>0].sum():.1f} mV / "
              f"{peso[peso<0].sum():.1f} mV")

        med, disparos = roda_e_mede(eng, c, idx_gf, idx_sens, mascaras, pre, peso)
        print(f"  excitacao recebida {med['excitacao_mV']:10.2f} mV")
        print(f"  inibicao recebida  {med['inibicao_mV']:10.2f} mV")
        print(f"  liquido            {med['liquido_mV']:10.2f} mV")
        print(f"  v do GF: max {med['v_max_mV']:.3f}  media {med['v_media_mV']:.3f}"
              f"   (limiar -45.0, repouso -52.0)")
        print(f"  spikes do GF: {med['spikes_gf']}")

        # quem mais contribuiu com inibicao
        por_pre = defaultdict(float)
        for i, (pp, ww) in enumerate(zip(pre, peso)):
            if ww < 0 and disparos.get(int(pp), 0):
                por_pre[int(pp)] += ww * disparos[int(pp)]
        top = sorted(por_pre.items(), key=lambda kv: kv[1])[:8]
        if top:
            print("  maiores fontes de inibicao (bodyId, tipo, mV acumulado):")
            for idx, mv in top:
                bid = int(c.body_ids[idx])
                tipo = (str(props.at[bid, "type"]) if bid in props.index else "?")
                print(f"    {bid:12d}  {tipo:14s} {mv:9.2f} mV")
        saida[escopo] = {
            **med,
            "neuronios": c.n, "arestas": c.e,
            "entradas_gf": len(pre), "exc_edges": n_exc, "inh_edges": n_inh,
            "peso_exc_total_mV": round(float(peso[peso > 0].sum()), 2),
            "peso_inh_total_mV": round(float(peso[peso < 0].sum()), 2),
            "top_inibicao": [{"bodyId": int(c.body_ids[i]),
                              "tipo": (str(props.at[int(c.body_ids[i]), "type"])
                                       if int(c.body_ids[i]) in props.index else None),
                              "mV": round(mv, 2)} for i, mv in top],
        }
        print()

    a, b = saida["circuito"], saida["whole"]
    print("== diferenca ==")
    print(f"  arestas entrando no GF: {a['entradas_gf']} -> {b['entradas_gf']}")
    print(f"  inibicao recebida: {a['inibicao_mV']:.1f} -> {b['inibicao_mV']:.1f} mV")
    print(f"  excitacao recebida: {a['excitacao_mV']:.1f} -> {b['excitacao_mV']:.1f} mV")
    print(f"  liquido: {a['liquido_mV']:.1f} -> {b['liquido_mV']:.1f} mV")
    print(f"  spikes do GF: {a['spikes_gf']} -> {b['spikes_gf']}")

    destino = RAIZ / "benchmarks" / "neural" / "giant_fiber_trace.json"
    destino.write_text(json.dumps({
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "estimulo_hz": TAXA_LOOMING_HZ, "janela_ms": JANELA_MS,
        "semente": SEMENTE, "spikes_sensoriais": total_spikes_sens,
        "nota": ("mesma realizacao Poisson nos dois escopos; nenhum parametro "
                 "foi alterado"),
        **saida,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
