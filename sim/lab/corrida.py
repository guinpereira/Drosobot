"""
Roda uma receita e deixa a corrida em disco.

Sem interface, sem Unity, sem terminal interativo: é o que uma bateria de
experimentos chama num laço. A telemetria é opcional e, quando ligada, grava o
`telemetry.jsonl` que o replay serve de volta.

    from lab import Receita, roda
    pasta = roda(Receita(nome="looming", escopo="whole", duracao_s=2.0))

## Uma implementação só da ciência

Este módulo **não** tem laço de simulação. Ele monta a `Laboratorio` do
`drosobot_lab.py` — a mesma que a interface dirige — e a conduz. Se houvesse
aqui um segundo laço, ele divergiria do primeiro, e nenhum dos dois seria
confiável.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

from .receita import Receita, metadata          # noqa: E402
from .registro import Registro                  # noqa: E402


def _args_de(receita: Receita) -> SimpleNamespace:
    """
    A `Laboratorio` foi escrita para receber o namespace do argparse.

    Traduzir aqui é mais honesto que fazer ela aceitar dois formatos: a receita
    é a fonte, e esta função mostra exatamente qual campo vira qual opção.
    """
    return SimpleNamespace(
        physics=receita.physics,
        neural=receita.neural,
        cns=receita.escopo,
        colisao=receita.colisao,
        duracao=receita.duracao_s,
        seed=receita.seed,
        experiment=None,
    )


def _experimento_do_catalogo(receita: Receita) -> str:
    """Id do catálogo que casa com a arena e o escopo pedidos."""
    import drosobot_lab as lab

    for e in lab.CATALOGO:
        if e["arena"] == receita.arena and e["cns"] == receita.escopo:
            return e["id"]
    raise ValueError(
        f"nenhum experimento no catalogo roda arena={receita.arena!r} com "
        f"escopo={receita.escopo!r}. Prometer o que nao existe e pior que "
        "recusar aqui.")


def roda(receita: Receita, raiz: Path | None = None,
         telemetria: bool = True, porta: int = 8765,
         silencioso: bool = False) -> Path:
    """
    Executa a receita e devolve a pasta da corrida.

    `telemetria=True` grava o `telemetry.jsonl` sem precisar de ninguém
    conectado — o gravador é um sink por si só. É o que permite abrir a corrida
    na Unity depois, sem refazer a simulação.
    """
    import drosobot_lab as lab
    from telemetry import protocol
    from telemetry.control import SemControle
    from telemetry.recorder import Gravador
    from telemetry.server import abrir as abrir_telemetria

    exp_id = _experimento_do_catalogo(receita)
    args = _args_de(receita)
    meta = metadata(receita)

    registro = Registro(receita.id_corrida, meta, raiz=raiz)
    # O gravador escreve na MESMA pasta do registro: os cinco arquivos de uma
    # corrida ficam juntos, e o replay acha o telemetry.jsonl onde a análise
    # acha o summary.json.
    sink = abrir_telemetria(porta=porta, ativo=False)
    gravador = Gravador(sink, receita.id_corrida, metadata=meta,
                        raiz=registro.pasta.parent,
                        fechar_sink=True) if telemetria else sink
    if telemetria:
        # o Gravador cria pasta própria com carimbo; usamos a do registro
        _funde_pastas(gravador, registro)

    laboratorio = lab.Laboratorio(args, gravador, SemControle(), protocol,
                                  registro=registro,
                                  estimulo=receita.estimulo)
    t0 = time.perf_counter()
    try:
        if not laboratorio.monta(exp_id, receita.seed):
            raise RuntimeError(f"nao foi possivel montar {exp_id}")
        n_passos = int(receita.duracao_s / lab.DT)
        while laboratorio.passo_atual < n_passos:
            laboratorio.passo()
        laboratorio.estado = "finished"
    finally:
        parede = time.perf_counter() - t0
        resumo = _resumo(laboratorio, receita, parede, exp_id)
        registro.resumo(resumo)
        registro.fechar()
        if telemetria:
            gravador.escrever_resumo(resumo)
            gravador.fechar()
        if laboratorio.corpo is not None:
            laboratorio.corpo.fecha()

    if not silencioso:
        print(f"  {receita.id_corrida:<44s} fugas={resumo['desfecho']['fugas']} "
              f"GFspk={resumo['gf']['spikes']} RTF={resumo['custo']['rtf']:.3f} "
              f"({parede:.0f} s)")
    return registro.pasta


def _funde_pastas(gravador, registro) -> None:
    """
    Faz o gravador escrever na pasta do registro.

    O `Gravador` nasceu para scripts de uma corrida só e cria pasta própria com
    carimbo de tempo. Aqui quem manda no nome é a receita, porque é ela que a
    bateria usa para achar a corrida depois.
    """
    import shutil

    antiga = gravador.pasta
    try:
        gravador._f.close()
    except Exception:                                         # noqa: BLE001
        pass
    gravador.pasta = registro.pasta
    gravador._f = (registro.pasta / "telemetry.jsonl").open("w", encoding="utf-8")
    if antiga != registro.pasta and antiga.exists():
        shutil.rmtree(antiga, ignore_errors=True)


def _resumo(lab_obj, receita: Receita, parede_s: float, exp_id: str) -> dict:
    """Os agregados que uma tabela comparativa lê."""
    # a soma por população só vira dicionário no fim: durante a corrida ela é
    # um vetor, que é o que permite acumular com um `+=` por janela
    lab_obj._fecha_acumulado()
    v = lab_obj.prof.valores()
    r = lab_obj.eng.resumo() if lab_obj.eng is not None else {}
    rc = lab_obj.corpo.resumo() if lab_obj.corpo is not None else {}
    pos = (np.asarray(lab_obj.frame.posicao, dtype=float).tolist()
           if getattr(lab_obj, "frame", None) is not None else [])
    g = lab_obj.gf_acumulado
    return {
        "experiment_id": exp_id,
        "receita": receita.para_dict(),
        "desfecho": {
            "fugas": lab_obj.escapes,
            "posicao_final_mm": [round(x, 4) for x in pos],
            "janelas": lab_obj.janelas,
            "passos_fisica": lab_obj.passo_atual,
            "sim_s": round(lab_obj.t_s, 4),
        },
        "gf": {
            "spikes": g["spikes"],
            "excitacao_mV": round(g["exc"], 2),
            "inibicao_mV": round(g["inib"], 2),
            "liquido_mV": round(g["exc"] + g["inib"], 2),
            "v_min_mV": round(g["v_min"], 2),
            "arestas_entrando": int(len(lab_obj.gf_peso)),
            "por_populacao_mV": g["por_tipo"],
        },
        "sensorial": {
            "spikes_totais": g["sensoriais"],
            "hz_max": round(g["hz_max"], 3),
            "ttmn_spikes": g["ttmn"],
        },
        "custo": {
            "parede_s": round(parede_s, 2),
            "rtf": round(v["_total"]["rtf"], 4),
            "ms_por_seg_simulado": {
                k: round(v[k]["ms_por_seg_simulado"], 1)
                for k in ("physics", "vision", "neural", "leitura", "telemetry")
            },
        },
        "runtime": {
            "physics_backend": rc.get("physics_backend"),
            "neural_backend": r.get("backend"),
            "neural_device": r.get("device"),
            "neurons_simulated": r.get("neurons_simulated"),
            "edges_simulated": r.get("edges_simulated"),
            "collision_pairs": rc.get("collision_pairs"),
        },
    }
