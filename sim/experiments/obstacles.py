"""
Obstacle Field -- RESULTADO NEGATIVO, e a interface tem que dizer isso.

A mosca anda livre entre postes alternados dos dois lados. O fluxo optico vem do
movimento dela, e o circuito optomotor deveria desviar. Nao desvia.

## Por que nao funciona

Medindo o circuito isolado, com as taxas que um poste lateral realmente produz
(medidas na propria simulacao: ~205 Hz no olho de perto, ~122 no outro):

    entrada          motor L      motor R      L - R
    122/122 (sem)      0.0         16.0        -16.0
    205/122 (ESQ)      5.8         16.0        -10.2
    122/205 (DIR)      0.0         38.6        -38.6

O sinal de (L - R) e SEMPRE negativo, esteja o obstaculo de que lado for. Um
controlador que leia "de que lado" pela comparacao bilateral nao tem como
funcionar -- nessa faixa o circuito reporta QUANTO, nao DE QUE LADO.

A causa e a assimetria de reconstrucao do Male CNS, ja documentada no README: o
hemisferio direito e mais forte em toda etapa (T4/T5 721 contra 637,
DNa02->motor 442 contra 334).

Seis tentativas estao registradas no cabecalho de sim/flygym_avoidance.py, que e
o script que produz a figura. Este arquivo roda a tentativa 6, a mesma, e existe
pra que o resultado negativo possa ser VISTO acontecer, nao so lido num grafico.

Os parametros NAO devem ser mexidos pra fazer funcionar. Se alguem quiser tentar
de novo, que seja uma variante nova e declarada, com a medicao acima refeita.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import flygym.examples as fex
from flygym.arena import FlatTerrain

from telemetry import protocol

from . import circuits
from .base import registrar
from .corpo import BASE_DRIVE, ExperimentoFlygym

# Postes ALTERNANDO dos lados da linha de caminhada, nao em cima dela: ameaca
# frontal produz n_L ~ n_R e o sinal de lado zera. A +/-7 mm a mosca cega passa
# limpa, e sobra espaco pra medir se virar aumenta ou diminui a folga.
POSTES = np.array([(12.0 + 10.0 * k, 7.0 if k % 2 == 0 else -7.0) for k in range(6)])
RAIO_POSTE = 3.0
# "tocou o poste": centro da mosca dentro do raio dele mais um corpo de mosca.
# Contar TRAVESSIAS desse limiar nao funciona -- o corpo balanca ao andar e uma
# unica batida virava dezenas de "encostoes". Contamos postes DISTINTOS tocados.
RAIO_COLISAO = RAIO_POSTE + 1.0


@registrar
class ObstacleField(ExperimentoFlygym):
    id = "obstacle_field"
    name = "Obstacle Field (resultado negativo)"
    description = ("Postes alternados dos dois lados. O circuito optomotor NAO "
                   "guia o desvio: nessa faixa de entrada (L-R) e sempre "
                   "negativo, de que lado estiver o obstaculo.")

    # Ganho calibrado NESTA geometria, nao copiado do optomotor: com o ganho de
    # 30000 os dois olhos saturavam em 250 Hz, e saturacao MATA a assimetria, que
    # e exatamente o sinal. 11000 poe o olho estimulado perto de 200 Hz.
    FLOW_GAIN = 11000.0
    FLOW_MAX_HZ = 250.0
    GIRO = 0.6          # quanto o lado escolhido encurta o passo
    GIRO_TAU = 4.0      # quanto o comando de giro demora a acumular
    VIES_TAU = 120.0    # atualizacoes de retina (~1.2 s a 100 Hz)
    # "para_longe" e a tentativa 6; "para_ameaca" e a 5; "cego" e o controle.
    MODO = "para_longe"

    def __init__(self, **config):
        super().__init__(**config)
        self.modo = str(config.get("modo", self.MODO))
        self.retina_ant = None
        self.bal = 0.0
        self.vies = 0.0
        self.tocados = np.zeros(len(POSTES), dtype=bool)
        self.folga_min = np.full(len(POSTES), np.inf)

    def monta_arena(self):
        spec = importlib.util.spec_from_file_location(
            "flygym_vision_arena",
            os.path.join(fex.__path__[0], "vision", "arena.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ObstacleOdorArena(
            terrain=FlatTerrain(),
            obstacle_positions=POSTES,
            obstacle_colors=(0, 0, 0, 1),
            obstacle_radius=RAIO_POSTE,
            obstacle_height=4.0,
            odor_source=np.array([[1000.0, 0.0, 2.0]]),   # longe: nao usamos odor
            marker_colors=[],
        )

    def monta_circuito(self):
        return circuits.optomotor()

    def prepara(self, retina0):
        self.retina_ant = retina0
        self.bal = 0.0
        self.vies = 0.0
        self.tocados = np.zeros(len(POSTES), dtype=bool)
        self.folga_min = np.full(len(POSTES), np.inf)

    def antes_do_passo(self, t_s):
        # contato e folga saem da posicao que o MuJoCo reporta, nao de conta nossa
        if self.obs is None:
            return
        pos = np.asarray(self.obs["fly"][0])[:2]
        d = np.linalg.norm(POSTES - pos, axis=1)
        self.tocados |= d < RAIO_COLISAO
        self.folga_min = np.minimum(self.folga_min, d)

    def processa_retina(self, retina):
        e = self.circuito.extras
        flow = np.abs(retina - self.retina_ant).mean(axis=1)
        self.retina_ant = retina
        hz = np.clip(flow * self.FLOW_GAIN, 0, self.FLOW_MAX_HZ)
        taxas = np.where(e["lado_sensor"] == "L", hz[0], hz[1])
        return taxas, {"optic_flow": {"L": float(hz[0]), "R": float(hz[1])},
                       "input_hz": {"L": float(hz[0]), "R": float(hz[1])},
                       "obstacles_touched": int(self.tocados.sum())}

    def decide(self, saida, t_s):
        e = self.circuito.extras
        if self.modo == "cego":
            return np.array([BASE_DRIVE, BASE_DRIVE]), []

        n_L = int(saida[e["motor_L"]].sum())
        n_R = int(saida[e["motor_R"]].sum())
        lado = 0.0
        if n_L + n_R > 0:
            lado = (n_R - n_L) / (n_L + n_R)     # +1 obstaculo visto a direita

        # O desequilibrio tem um VIES PERMANENTE (o hemisferio direito do dataset
        # e mais forte em toda etapa). Com o vies entrando direto no comando, a
        # mosca girava sem parar e nem chegava nos postes. Descontamos a media
        # lenta do proprio sinal, entao so DESVIO em relacao ao normal dela vira
        # curva. Suposicao nossa, igual ao detector de looming.
        self.vies += (lado - self.vies) / self.VIES_TAU
        desvio = lado - self.vies
        if self.modo == "para_longe":
            desvio = -desvio
        self.bal += (desvio - self.bal) / self.GIRO_TAU

        drive = np.array([BASE_DRIVE, BASE_DRIVE])
        if self.bal > 0:
            drive[1] -= self.bal * self.GIRO
        else:
            drive[0] -= -self.bal * self.GIRO
        return np.clip(drive, -0.5, 1.5), []

    # ------------------------------------------------------------ descricao

    def telemetry_metadata(self):
        p = self.base_parameters()
        p.update({"flow_gain": self.FLOW_GAIN, "turn_gain": self.GIRO,
                  "turn_tau": self.GIRO_TAU, "bias_tau": self.VIES_TAU,
                  "mode": self.modo, "n_obstacles": len(POSTES),
                  "obstacle_radius_mm": RAIO_POSTE})
        return {
            "parameters": p,
            "circuits": self.circuito.descreve(),
            # A interface le isto pra avisar antes da corrida. Nao e humildade
            # decorativa: quem abre o Lab tem que saber que esta vendo um
            # resultado negativo reproduzido, nao um desvio que quase funciona.
            "known_result": {
                "outcome": "negative",
                "summary": ("O circuito nao guia desvio de obstaculo. A folga "
                            "media fica igual a do controle cego "
                            "(6.91 contra 6.88 mm)."),
                "cause": ("(L-R) e sempre negativo nessa faixa de entrada, de "
                          "que lado estiver o obstaculo. Assimetria de "
                          "reconstrucao do Male CNS: hemisferio direito mais "
                          "forte em toda etapa."),
                "evidence": "sim/flygym_avoidance.py (6 tentativas registradas)",
                "do_not_tune": True,
            },
            "provenance": {
                "body_ids": protocol.DATA, "side": protocol.DATA,
                "type": protocol.DATA, "neurotransmitter": protocol.DATA,
                "synapse_weight": protocol.DATA,
                "obstacle_positions": protocol.ASSUMPTION,
                "membrane_potential": protocol.MODEL, "spikes": protocol.MODEL,
                "synaptic_conductance": protocol.MODEL, "firing_rate": protocol.MODEL,
                "synapse_sign": protocol.MODEL,
                "flow_gain": protocol.ASSUMPTION,
                "optic_flow_detector": protocol.ASSUMPTION,
                "bias_subtraction": protocol.ASSUMPTION,
                "turn_gain": protocol.ASSUMPTION,
                "motor_to_gait_mapping": protocol.ASSUMPTION,
            },
        }

    def scene_metadata(self):
        return {
            "arena": {"kind": "obstacle_field", "terrain": "flat"},
            "obstacles": [{"x": float(x), "y": float(y), "radius": RAIO_POSTE,
                           "height": 4.0} for x, y in POSTES],
            "stimulus": {"kind": "self_motion_flow"},
        }

    def results(self):
        r = super().results()
        # so conta a folga dos postes que a mosca chegou a passar
        passou = self.folga_min < 20.0
        r.update({
            "mode": self.modo,
            "obstacles_touched": int(self.tocados.sum()),
            "obstacles_passed": int(passou.sum()),
            "mean_clearance_mm": (round(float(np.mean(self.folga_min[passou])), 2)
                                  if passou.any() else None),
            "outcome": "negative",
        })
        return r
