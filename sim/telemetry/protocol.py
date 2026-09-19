"""
Protocolo de telemetria do Drosobot: simulacao (Python) -> visualizador (Unity).

Transporte: TCP com uma mensagem JSON por linha (newline-delimited JSON).
Escolhido em vez de WebSocket porque nao acrescenta dependencia, e trivial de ler
em C# com StreamReader.ReadLine(), e sobrevive bem no Windows. O volume aqui e
pequeno -- dezenas de mensagens por segundo, nao milhares.

REGRA DE MAO UNICA: a telemetria so OBSERVA. Nenhuma mensagem volta pra
simulacao, e nenhum dado daqui entra em conta na fisica ou no circuito. Se o
visualizador cair, a simulacao nao muda de comportamento.

Toda mensagem carrega:

    {"protocol": "drosobot-telemetry", "version": 1, "type": "...", ...}

Tipos:

    hello             identificacao do emissor, uma vez na conexao
    experiment_info   id/nome/descricao do experimento e seus parametros
    scene_info        geometria estatica: obstaculos, arena, estimulo
    frame             pose da mosca + comando motor (o mais frequente)
    neural_activity   spikes/estado por camada do circuito
    retina            os dois olhos e os valores derivados
    event             marco discreto (escape, contato, reset...)
    statistics        agregados da corrida ate agora
    bye               fim da corrida

## Procedencia: DATA / MODEL / ASSUMPTION

Cada campo que o visualizador mostra ao usuario carrega, no `experiment_info`,
de onde ele vem. Isso e requisito do projeto, nao enfeite -- o Drosobot inteiro
existe pra manter separado o que e conectoma medido e o que e suposicao nossa.

    DATA        veio do conectoma (bodyId, contagem de sinapse, neurotransmissor,
                hemisferio)
    MODEL       saiu do modelo biofisico de Shiu et al. rodando sobre esse dado
                (potencial de membrana, spike, taxa de disparo)
    ASSUMPTION  escolha nossa de modelagem, nao esta em lugar nenhum do dado
                (ganho da transducao retina->Hz, mapeamento spike->marcha, taxa
                tonica dos inibitorios)
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterable

PROTOCOL = "drosobot-telemetry"
VERSION = 1

# procedencia de um campo, pro visualizador poder marcar na interface
DATA = "data"              # medido no conectoma
MODEL = "model"            # derivado do modelo biofisico
ASSUMPTION = "assumption"  # escolha nossa de modelagem


def _envelope(tipo: str, corpo: dict[str, Any]) -> dict[str, Any]:
    msg = {"protocol": PROTOCOL, "version": VERSION, "type": tipo}
    msg.update(corpo)
    return msg


def encode(msg: dict[str, Any]) -> bytes:
    """Uma mensagem por linha. separators sem espaco: e stream, nao arquivo lido a olho."""
    return (json.dumps(msg, separators=(",", ":"), default=_json_default) + "\n").encode("utf-8")


def _json_default(o):
    # numpy aparece em quase todo campo; converte sem obrigar quem chama a lembrar
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"nao serializavel: {type(o)!r}")


def decode(linha: str | bytes) -> dict[str, Any]:
    if isinstance(linha, bytes):
        linha = linha.decode("utf-8")
    msg = json.loads(linha)
    if msg.get("protocol") != PROTOCOL:
        raise ValueError(f"protocolo inesperado: {msg.get('protocol')!r}")
    if msg.get("version") != VERSION:
        raise ValueError(f"versao inesperada: {msg.get('version')!r} (esperado {VERSION})")
    return msg


# ---------------------------------------------------------------- construtores

def hello(fonte: str = "drosobot-sim", **extra) -> dict[str, Any]:
    return _envelope("hello", {"source": fonte, "wall_time": time.time(), **extra})


def experiment_info(experiment_id: str, name: str, description: str,
                    parameters: dict[str, Any] | None = None,
                    provenance: dict[str, str] | None = None,
                    circuits: list[dict[str, Any]] | None = None,
                    **extra) -> dict[str, Any]:
    """
    `provenance` mapeia nome do parametro -> DATA / MODEL / ASSUMPTION.
    `circuits` descreve as camadas: nome, bodyIds, grupo, lado, tipo, NT.
    """
    return _envelope("experiment_info", {
        "experiment_id": experiment_id,
        "name": name,
        "description": description,
        "parameters": parameters or {},
        "provenance": provenance or {},
        "circuits": circuits or [],
        **extra,
    })


def scene_info(obstacles: list[dict[str, Any]] | None = None,
               arena: dict[str, Any] | None = None,
               stimulus: dict[str, Any] | None = None, **extra) -> dict[str, Any]:
    return _envelope("scene_info", {
        "obstacles": obstacles or [],
        "arena": arena or {},
        "stimulus": stimulus or {},
        **extra,
    })


def frame(step: int, sim_time: float, wall_time: float, real_time_factor: float,
          position: Iterable[float], orientation: Iterable[float] | None = None,
          drive: Iterable[float] | None = None,
          linear_velocity: Iterable[float] | None = None,
          contacts: Iterable[float] | None = None,
          running: bool = True, **extra) -> dict[str, Any]:
    corpo = {
        "step": step,
        "sim_time": round(float(sim_time), 6),
        "wall_time": round(float(wall_time), 3),
        "rtf": round(float(real_time_factor), 4),
        "running": bool(running),
        "position": [round(float(v), 4) for v in position],
    }
    if orientation is not None:
        corpo["orientation"] = [round(float(v), 5) for v in orientation]
    if drive is not None:
        corpo["drive"] = [round(float(v), 4) for v in drive]
    if linear_velocity is not None:
        corpo["velocity"] = [round(float(v), 4) for v in linear_velocity]
    if contacts is not None:
        corpo["contacts"] = [round(float(v), 3) for v in contacts]
    corpo.update(extra)
    return _envelope("frame", corpo)


def neural_activity(sim_time: float, layers: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """
    Cada item de `layers`:
        {"name": "GF", "spikes": [0,1,...], "rate_hz": [...],
         "v_mV": [...] (opcional), "g_mV": [...] (opcional),
         "refractory": [...] (opcional)}

    `spikes` e a contagem da janela, por neuronio, na ordem dos body_ids que o
    experiment_info declarou pra essa camada. Nao mandamos os bodyIds a cada
    mensagem -- eles nao mudam.
    """
    return _envelope("neural_activity", {
        "sim_time": round(float(sim_time), 6),
        "layers": layers,
        **extra,
    })


def retina(sim_time: float, left: Iterable[float] | None = None,
           right: Iterable[float] | None = None,
           derived: dict[str, Any] | None = None, **extra) -> dict[str, Any]:
    """
    `left`/`right` sao os 721 omatideos ja reduzidos a um valor por omatideo.
    `derived` traz o que o experimento calculou: dark_fraction, looming, flow,
    input_hz -- cada um como {"L": x, "R": y}.
    """
    corpo = {"sim_time": round(float(sim_time), 6), "derived": derived or {}}
    if left is not None:
        corpo["left"] = [round(float(v), 4) for v in left]
    if right is not None:
        corpo["right"] = [round(float(v), 4) for v in right]
    corpo.update(extra)
    return _envelope("retina", corpo)


def event(sim_time: float, kind: str, detail: dict[str, Any] | None = None,
          **extra) -> dict[str, Any]:
    """kind: looming_detected, escape_triggered, turn_left, turn_right,
    obstacle_contact, experiment_reset, stimulus_started, stimulus_finished."""
    return _envelope("event", {
        "sim_time": round(float(sim_time), 6),
        "kind": kind,
        "detail": detail or {},
        **extra,
    })


def statistics(sim_time: float, values: dict[str, Any], **extra) -> dict[str, Any]:
    return _envelope("statistics", {
        "sim_time": round(float(sim_time), 6),
        "values": values,
        **extra,
    })


def bye(reason: str = "finished", **extra) -> dict[str, Any]:
    return _envelope("bye", {"reason": reason, "wall_time": time.time(), **extra})
