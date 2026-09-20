"""
Optomotor Turning.

A mosca anda num terreno de blocos. O movimento dela propria gera fluxo optico
diferente em cada olho, e o circuito optomotor transforma esse desequilibrio em
curva.

    fluxo optico por olho
      -> T4/T5 -> HS -> DNa02 -> motoneuronio de perna      [conectoma]
      -> encurta o passo do lado escolhido

Mesmos parametros de sim/flygym_live.py, que e a versao ao vivo deste circuito.
O script headless sim/flygym_optomotor.py usa os mesmos ganhos, mas roda no
Brian2 e com a retina a 500 Hz.

Este circuito lateraliza bem quando o contraste entre os olhos e grande. Com
contraste pequeno ele reporta QUANTO, nao DE QUE LADO -- e por isso que o
experimento de obstaculos (obstacles.py) da resultado negativo. Aqui o terreno de
blocos produz contraste alto o bastante pra curva aparecer.
"""
from __future__ import annotations

import numpy as np
from flygym.arena import BlocksTerrain

from telemetry import protocol

from . import circuits
from .base import registrar
from .corpo import BASE_DRIVE, ExperimentoFlygym


@registrar
class OptomotorTurning(ExperimentoFlygym):
    id = "optomotor_turning"
    name = "Optomotor Turning"
    description = ("Fluxo optico do proprio caminhar vira curva. "
                   "T4/T5 -> HS -> DNa02 -> motor de perna.")

    FLOW_GAIN = 30000.0
    FLOW_MAX_HZ = 250.0
    TURN_GAIN = 1.2
    TURN_TAU = 12.0

    def __init__(self, **config):
        super().__init__(**config)
        self.retina_ant = None
        self.bal_suave = 0.0
        self.n_L = 0
        self.n_R = 0

    def monta_arena(self):
        return BlocksTerrain(height_range=(0.2, 0.2), block_size=1.3)

    def monta_circuito(self):
        return circuits.optomotor()

    def prepara(self, retina0):
        self.retina_ant = retina0
        self.bal_suave = 0.0

    def processa_retina(self, retina):
        e = self.circuito.extras
        flow = np.abs(retina - self.retina_ant).mean(axis=1)   # um valor por olho
        self.retina_ant = retina
        hz = np.clip(flow * self.FLOW_GAIN, 0, self.FLOW_MAX_HZ)
        taxas = np.where(e["lado_sensor"] == "L", hz[0], hz[1])
        return taxas, {"optic_flow": {"L": float(hz[0]), "R": float(hz[1])},
                       "input_hz": {"L": float(hz[0]), "R": float(hz[1])}}

    def decide(self, saida, t_s):
        e = self.circuito.extras
        self.n_L = int(saida[e["motor_L"]].sum())
        self.n_R = int(saida[e["motor_R"]].sum())
        total = self.n_L + self.n_R
        bal = 0.0 if total == 0 else (self.n_R - self.n_L) / total
        self.bal_suave += (bal - self.bal_suave) / self.TURN_TAU

        drive = np.array([BASE_DRIVE, BASE_DRIVE])
        if self.bal_suave > 0:
            drive[1] -= self.bal_suave * self.TURN_GAIN
        else:
            drive[0] -= -self.bal_suave * self.TURN_GAIN
        drive = np.clip(drive, -0.5, 1.5)

        eventos = []
        if total > 0:
            eventos.append(("turn_right" if self.n_R > self.n_L else "turn_left",
                            {"motor_L": self.n_L, "motor_R": self.n_R}))
        return drive, eventos

    # ------------------------------------------------------------ descricao

    def telemetry_metadata(self):
        p = self.base_parameters()
        p.update({"flow_gain": self.FLOW_GAIN, "flow_max_hz": self.FLOW_MAX_HZ,
                  "turn_gain": self.TURN_GAIN, "turn_tau": self.TURN_TAU})
        return {
            "parameters": p,
            "circuits": self.circuito.descreve(),
            "provenance": {
                "body_ids": protocol.DATA, "side": protocol.DATA,
                "type": protocol.DATA, "neurotransmitter": protocol.DATA,
                "synapse_weight": protocol.DATA,
                "membrane_potential": protocol.MODEL, "spikes": protocol.MODEL,
                "synaptic_conductance": protocol.MODEL, "firing_rate": protocol.MODEL,
                "synapse_sign": protocol.MODEL,
                "flow_gain": protocol.ASSUMPTION,
                "optic_flow_detector": protocol.ASSUMPTION,
                "turn_gain": protocol.ASSUMPTION,
                "motor_to_gait_mapping": protocol.ASSUMPTION,
            },
        }

    def scene_metadata(self):
        return {"arena": {"kind": "blocks_terrain", "block_size": 1.3,
                          "height": 0.2},
                "stimulus": {"kind": "self_motion_flow"}}

    def results(self):
        r = super().results()
        r.update({"motor_L": self.n_L, "motor_R": self.n_R,
                  "turn_balance": round(self.bal_suave, 4)})
        return r
