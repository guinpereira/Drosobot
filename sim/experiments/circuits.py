"""
Montagem dos circuitos, extraida dos scripts pra nao existir em tres copias.

Isto NAO e um modelo novo. E exatamente o que flygym_escape.py, flygym_optomotor.py
e flygym_avoidance.py ja faziam, no mesmo formato e com os mesmos CSVs. A unica
diferenca e que agora da pra montar o mesmo circuito duas vezes no mesmo processo,
que e o que o seletor precisa pra trocar de experimento sem reiniciar o Python.

Os pesos e os sinais continuam saindo de connectome_model.signed_weights(), que e
onde mora a convencao de Shiu et al.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectome_model import (  # noqa: E402
    CONNECTOME, load_properties, signed_weights, side_of, gf_input_population,
)
import fast_lif  # noqa: E402

DT_MS = 0.5

# Ler os CSVs custa perto de um segundo e eles nao mudam durante a sessao. Com o
# seletor da pra trocar de experimento varias vezes, entao guardamos.
_cache: dict[str, object] = {}


def _props():
    if "props" not in _cache:
        _cache["props"] = load_properties()
    return _cache["props"]


def _csv(nome: str) -> pd.DataFrame:
    chave = f"csv:{nome}"
    if chave not in _cache:
        _cache[chave] = pd.read_csv(CONNECTOME / nome)
    return _cache[chave]


def _ix(ids):
    return {b: i for i, b in enumerate(ids)}


def _monta(tabelas):
    """(conexoes, ix_pre, ix_pos) -> (i_pre, j_pos, w_mV) no formato do fast_lif."""
    props = _props()
    out = []
    for conn, si, ti in tabelas:
        agg = signed_weights(conn, props)
        agg = agg[agg["bodyId_pre"].isin(si) & agg["bodyId_post"].isin(ti)]
        out.append(([si[b] for b in agg["bodyId_pre"]],
                    [ti[b] for b in agg["bodyId_post"]],
                    agg["w_mV"].values))
    return out


class Circuito:
    """Rede montada + os indices que o experimento precisa pra ler a saida."""

    def __init__(self, rede, camadas, entrada, camada_saida, extras=None):
        self.rede = rede                  # fast_lif.Rede
        # Aqui a rede SEMPRE vai pra telemetria, entao a contagem da populacao de
        # entrada fica ligada. Ela e o unico jeito de os LC4/LPLC2 sem morfologia
        # aparecerem: a atividade deles vira barra de populacao na interface.
        rede.contar_entrada = True
        self.camadas = camadas            # [(nome, body_ids, papel)]
        self.entrada = entrada            # (nome, body_ids, papel)
        self.camada_saida = camada_saida  # indice em self.camadas
        self.extras = extras or {}        # mascaras por lado etc.
        # A populacao de entrada da rede nem sempre e a que o experimento
        # DECLARA. No Giant Fiber a rede tem os 1271 pre-sinapticos (inclusive os
        # inibitorios, que so recebem taxa tonica) e o circuito declarado e so os
        # 311 de looming. Publicar o vetor cru alinharia os 311 primeiros spikes
        # com os bodyIds errados, e a interface acenderia neuronio trocado sem
        # nenhum sinal de erro.
        self.mascara_entrada = self.extras.get("mascara_entrada")

    def recorta_entrada(self, item: dict) -> dict:
        """Deixa o vetor de spikes da entrada do tamanho dos bodyIds declarados."""
        if self.mascara_entrada is not None:
            item["spikes"] = item["spikes"][self.mascara_entrada]
        return item

    def nomes_snapshot(self) -> list[str]:
        """Entrada primeiro, depois as camadas LIF -- a ordem que o snapshot usa."""
        return [self.entrada[0]] + [n for n, _, _ in self.camadas]

    def descreve(self) -> list[dict]:
        """Blocos `circuits` do experiment_info, com a procedencia ja implicita."""
        props = _props()

        def um(nome, ids, papel):
            return {
                "name": nome,
                "role": papel,
                "body_ids": [int(b) for b in ids],
                "sides": [side_of(props, b) for b in ids],
                "types": [str(props.at[b, "type"]) if b in props.index else None
                          for b in ids],
                "neurotransmitters": [str(props.at[b, "consensusNt"])
                                      if b in props.index else None for b in ids],
            }

        return [um(*self.entrada)] + [um(n, i, p) for n, i, p in self.camadas]


def giant_fiber() -> Circuito:
    """LC4/LPLC2 -> DNp01 (Giant Fiber) -> TTMn."""
    props = _props()
    up = _csv("gf_upstream_connections.csv")
    down = _csv("gf_downstream_connections.csv")

    GF_IDS = [10001, 10010]
    motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()
    sensor_ids, is_loom, is_inhib = gf_input_population(props, up)

    lado = np.array([side_of(props, b) for b in sensor_ids])
    is_loom = np.array(is_loom)
    is_inhib = np.array(is_inhib)

    conexoes = _monta([(up, _ix(sensor_ids), _ix(GF_IDS)),
                       (down, _ix(GF_IDS), _ix(motor_ids))])
    rede = fast_lif.Rede(len(sensor_ids), [len(GF_IDS), len(motor_ids)],
                         conexoes, DT_MS)

    return Circuito(
        rede=rede,
        camadas=[("DNp01", GF_IDS, "gf_dnp01_giantfiber"),
                 ("TTMn", motor_ids, "gf_ttmn_motor")],
        # So os looming entram na lista de entrada declarada: os inibitorios estao
        # na rede e recebem taxa tonica, mas nao sao o sensor.
        entrada=("LC4/LPLC2", [b for b, f in zip(sensor_ids, is_loom) if f],
                 "gf_sensor_looming"),
        camada_saida=1,
        extras={
            "sensor_ids": sensor_ids,
            "n_sensor": len(sensor_ids),
            "mascara_entrada": is_loom,   # a rede tem 1271; declaramos os looming
            "loom_L": is_loom & (lado == "L"),
            "loom_R": is_loom & (lado == "R"),
            "inhib": is_inhib,
        },
    )


def optomotor() -> Circuito:
    """T4/T5 -> HS -> DNa02 -> motoneuronio de perna."""
    props = _props()
    sensor_hs = _csv("opto_sensor_hs.csv")
    hs_dna02 = _csv("opto_hs_dna02.csv")
    dna02_motor = _csv("opto_dna02_motor.csv")

    sensor_ids = sorted(sensor_hs["bodyId_pre"].unique().tolist())
    hs_ids = sorted(set(sensor_hs["bodyId_post"]) | set(hs_dna02["bodyId_pre"]))
    dna_ids = sorted(set(hs_dna02["bodyId_post"]) | set(dna02_motor["bodyId_pre"]))
    motor_ids = sorted(dna02_motor["bodyId_post"].unique().tolist())

    lado_sensor = np.array([side_of(props, b) for b in sensor_ids])
    lado_motor = np.array([side_of(props, b) for b in motor_ids])

    conexoes = _monta([(sensor_hs, _ix(sensor_ids), _ix(hs_ids)),
                       (hs_dna02, _ix(hs_ids), _ix(dna_ids)),
                       (dna02_motor, _ix(dna_ids), _ix(motor_ids))])
    rede = fast_lif.Rede(len(sensor_ids),
                         [len(hs_ids), len(dna_ids), len(motor_ids)],
                         conexoes, DT_MS)

    return Circuito(
        rede=rede,
        camadas=[("HS", hs_ids, "om_hs_widefield"),
                 ("DNa02", dna_ids, "om_dna02_steering"),
                 ("motor", motor_ids, "om_leg_motor")],
        entrada=("T4/T5", sensor_ids, "om_sensor_t4t5"),
        camada_saida=2,
        extras={
            "sensor_ids": sensor_ids,
            "n_sensor": len(sensor_ids),
            "lado_sensor": lado_sensor,
            "motor_L": np.where(lado_motor == "L")[0],
            "motor_R": np.where(lado_motor == "R")[0],
        },
    )
