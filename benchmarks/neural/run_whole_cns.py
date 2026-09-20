"""
O Male CNS inteiro rodando de verdade na GPU. Uma medicao, nao uma bateria.

    .venv\\Scripts\\python benchmarks\\neural\\run_whole_cns.py

Nao e benchmark de array do mesmo tamanho: e o conectoma real carregado, com os
pesos reais e o sinal real, integrando de verdade.

Estimulo: as populacoes sensoriais que o Drosobot ja tem modeladas (LC4/LPLC2 do
looming). O resto da rede participa pelo que CHEGA pela conectividade -- que e
exatamente a situacao honesta do projeto hoje e o motivo de o relatorio separar

    WHOLE CONNECTOME SIMULATED     164.451 neuronios, 25,5M arestas
    WHOLE SENSORIMOTOR MODEL       nao: a maioria das populacoes nao tem
                                   entrada sensorial modelada
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

from neural import NeuralEngine, carrega_male_cns, metadados  # noqa: E402

JANELA_MS = 10.0        # a janela que o laco do Drosobot usa
N_JANELAS = 20


def populacao_sensorial(eng: NeuralEngine) -> np.ndarray:
    """
    Indices dos LC4/LPLC2 no grafo completo, pelos bodyIds reais.

    Usa o mesmo criterio que sim/connectome_model.py: o dataset anota o tipo, e
    LC4/LPLC2 sao a via de looming. Se os CSVs do circuito nao estiverem la,
    devolve vazio e o script avisa -- nao inventa uma populacao.
    """
    csv = RAIZ / "connectome" / "neuron_properties.csv"
    if not csv.exists():
        return np.zeros(0, dtype=np.int32)
    import pandas as pd
    props = pd.read_csv(csv)
    col_tipo = "type" if "type" in props.columns else None
    col_id = "bodyId" if "bodyId" in props.columns else props.columns[0]
    if col_tipo is None:
        return np.zeros(0, dtype=np.int32)
    alvo = props[props[col_tipo].isin(["LC4", "LPLC2"])][col_id].to_numpy()
    return eng.indices_de(alvo)


def main():
    meta = metadados()
    print("carregando o Male CNS...")
    t0 = time.perf_counter()
    c = carrega_male_cns()
    t_load = time.perf_counter() - t0
    print(f"  {c.n:,} neuronios, {c.e:,} arestas  ({t_load:.2f} s)")

    print("subindo pra GPU...")
    t0 = time.perf_counter()
    eng = NeuralEngine(c, backend="opencl")
    t_upload = time.perf_counter() - t0
    r = eng.resumo()
    print(f"  {r['device']}")
    print(f"  VRAM {r['vram_mib']:.1f} MiB (estatico {r['static_mib']:.1f} MiB)"
          f"  ({t_upload:.2f} s)")

    sens = populacao_sensorial(eng)
    externo = np.zeros(c.n, dtype=np.float32)
    if len(sens):
        # 1,2 mV por passo nos sensores: acima do limiar de disparo isolado
        # (~0,7 mV), que e o regime em que a via de looming opera nos
        # experimentos atuais do Drosobot.
        externo[sens] = 1.2
        print(f"  estimulo: {len(sens):,} neuronios sensoriais (LC4/LPLC2)")
    else:
        print("  AVISO: populacao sensorial nao encontrada; "
              "rodando so com a dinamica interna")

    print(f"\nrodando {N_JANELAS} janelas de {JANELA_MS} ms...")
    # aquecimento: a primeira janela compila/aquece e nao conta
    eng.roda(JANELA_MS, externo)
    eng.atividade_por_grupo(zerar=True)

    tempos = []
    ativos = []
    for _ in range(N_JANELAS):
        t0 = time.perf_counter()
        n_passos = eng.roda(JANELA_MS, externo)
        tempos.append(time.perf_counter() - t0)
        cont = eng.backend.le_contagem_total()
        ativos.append(int((cont > 0).sum()))
        # zera pra medir por janela
        eng.backend.reset() if False else None

    tempos = np.array(tempos)
    passos_por_janela = int(round(JANELA_MS / eng.dt))
    us_por_passo = tempos.mean() / passos_por_janela * 1e6
    # RTF neural: quanto tempo de mosca por segundo de relogio
    rtf = (JANELA_MS / 1000.0) / tempos.mean()

    cont = eng.backend.le_contagem_total()
    n_ativos = int((cont > 0).sum())
    total_spikes = int(cont.sum())

    print()
    print("== resultado ==")
    print(f"  janela de {JANELA_MS} ms ({passos_por_janela} passos): "
          f"{tempos.mean()*1000:7.2f} ms de relogio  "
          f"(min {tempos.min()*1000:.2f}, max {tempos.max()*1000:.2f})")
    print(f"  por passo neural (dt {eng.dt} ms): {us_por_passo:7.1f} us")
    print(f"  orcamento de tempo real: 500 us por passo -> "
          f"{'CABE' if us_por_passo < 500 else 'NAO CABE'} "
          f"({500/us_por_passo:.2f}x de folga)")
    print(f"  RTF neural: {rtf:.3f}x")
    print()
    print(f"  neuronios que dispararam ao menos uma vez: {n_ativos:,} "
          f"de {c.n:,} ({n_ativos/c.n*100:.1f}%)")
    print(f"  spikes totais acumulados: {total_spikes:,}")
    print(f"  tempo de mosca simulado: {eng.tempo_ms:.1f} ms")

    saida = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "scope": "whole_connectome_simulated",
        "nota": ("conectoma inteiro simulado. NAO e um modelo sensorimotor "
                 "completo: so as populacoes de looming tem entrada sensorial "
                 "modelada; o resto participa pelo que chega pela conectividade"),
        "conectoma": {**{k: meta.get(k) for k in
                         ("neurons", "edges", "synapses", "synapses_per_edge")},
                      "carregado_em_s": round(t_load, 2)},
        "device": r["device"],
        "backend": r["backend"],
        "vram_mib": r["vram_mib"],
        "upload_s": round(t_upload, 2),
        "dt_ms": eng.dt,
        "janela_ms": JANELA_MS,
        "ms_por_janela": round(float(tempos.mean() * 1000), 3),
        "us_por_passo_neural": round(float(us_por_passo), 1),
        "orcamento_us": 500,
        "cabe_em_tempo_real": bool(us_por_passo < 500),
        "rtf_neural": round(float(rtf), 4),
        "sensoriais_estimulados": int(len(sens)),
        "neuronios_que_dispararam": n_ativos,
        "spikes_totais": total_spikes,
    }
    destino = RAIZ / "benchmarks" / "neural" / "whole_cns_run.json"
    destino.write_text(json.dumps(saida, indent=2), encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
