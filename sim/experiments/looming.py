"""
Giant Fiber / Looming Escape.

Uma esfera se aproxima de frente em ciclos: some, reaparece longe, vem de novo.
Entre um ciclo e outro a mosca anda reto, entao da pra ver que a fuga acontece SO
quando o objeto chega perto -- que e o comportamento de limiar que o circuito
deveria ter.

    esfera aproximando -> retina -> expansao por olho
      -> LC4/LPLC2 -> DNp01 (Giant Fiber) -> TTMn          [conectoma]
      -> recuo rapido na marcha

Mesmos parametros de sim/flygym_live.py, que e a versao ao vivo deste circuito.
Este arquivo nao recalibra nada: se os numeros divergirem daquele script e bug,
nao melhoria.

Duas diferencas em relacao a sim/flygym_escape.py, que e o script headless que
produz a figura do README, e as duas ja existiam antes deste arquivo:

  - la a rede roda no Brian2, aqui no integrador proprio (sim/fast_lif.py). Os
    dois resolvem as mesmas equacoes e `python sim/fast_lif.py` compara.
  - la a retina roda a 500 Hz, aqui a 100. Por isso DARK_TAU e recalculado
    abaixo em vez de copiado: a adaptacao e 0.3 s de tempo real nos dois.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import flygym.examples as flygym_examples

from connectome_model import TONIC_INHIB_HZ
from telemetry import protocol

from . import circuits
from .base import registrar
from .corpo import VISION_HZ, BASE_DRIVE, ExperimentoFlygym


def _arena_module():
    # o __init__ de flygym.examples.vision importa torch, que nao precisamos --
    # carrega o modulo de arena apontando direto pro arquivo
    spec = importlib.util.spec_from_file_location(
        "flygym_vision_arena",
        os.path.join(flygym_examples.__path__[0], "vision", "arena.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@registrar
class LoomingEscape(ExperimentoFlygym):
    id = "looming_escape"
    name = "Giant Fiber / Looming Escape"
    description = ("Esfera se aproxima de frente. LC4/LPLC2 -> DNp01 -> TTMn. "
                   "A fuga so dispara perto: e o limiar do circuito, nao um "
                   "gatilho por distancia.")

    # transducao retina -> Hz: escolha nossa, nao esta no conectoma
    DARK_THRESHOLD = 0.4
    LOOM_GAIN = 240.0
    LOOM_MAX_HZ = 20.0
    ESCAPE_DRIVE = -0.5
    ESCAPE_MS = 120.0
    CICLO_S = 0.8
    DIST_LONGE = 30.0
    DIST_PERTO = 4.0

    def __init__(self, **config):
        super().__init__(**config)
        # a adaptacao do fundo e 0.3 s de tempo REAL. Nos scripts headless isso
        # virou 150 atualizacoes porque la a retina roda a 500 Hz; aqui roda a
        # 100, entao o numero tem que ser recalculado, senao a adaptacao fica 5x
        # mais lenta sem ninguem notar.
        self.DARK_TAU = 0.3 * VISION_HZ
        self.escape_ate = -1.0
        self.escuro_lento = None
        self.dist = self.DIST_LONGE
        self.escapes = 0

    def monta_arena(self):
        am = _arena_module()

        class LoomingArena(am.MovingObjArena):
            """Mesma arena da bola flutuante, mas quem move a bola somos nos."""

            def step(self, dt, physics):
                pass  # posicao vem do laco principal, que sabe onde a mosca esta

        return LoomingArena(obj_radius=3, init_ball_pos=(30, 0))

    def monta_circuito(self):
        return circuits.giant_fiber()

    def prepara(self, retina0):
        self.escuro_lento = (retina0 < self.DARK_THRESHOLD).mean(axis=1)
        self.escape_ate = -1.0
        self.escapes = 0

    def antes_do_passo(self, t_s):
        # A esfera e mocap: posicao imposta, nao simulada. Isso e estimulo, nao
        # fisica emergente -- a mosca e que responde.
        fase = (t_s % self.CICLO_S) / self.CICLO_S
        self.dist = self.DIST_LONGE + (self.DIST_PERTO - self.DIST_LONGE) * fase
        pos = np.asarray(self.obs["fly"][0])
        alvo = np.array([pos[0] + self.dist, pos[1], 2.5], dtype="float32")
        self.arena.ball_pos = alvo
        self.sim.physics.bind(self.arena.object_body).mocap_pos = alvo

    def processa_retina(self, retina):
        e = self.circuito.extras
        escuro = (retina < self.DARK_THRESHOLD).mean(axis=1)
        expansao = np.clip(escuro - self.escuro_lento, 0, None)
        self.escuro_lento += (escuro - self.escuro_lento) / self.DARK_TAU
        hz = np.clip(expansao * self.LOOM_GAIN, 0, self.LOOM_MAX_HZ)

        taxa = np.zeros(e["n_sensor"])
        taxa[e["loom_L"]] = hz[0]
        taxa[e["loom_R"]] = hz[1]
        # Os inibitorios nao respondem a retina: recebem taxa tonica. Sem isso a
        # inibicao do GF simplesmente nao existiria, e ela e o que faz a fuga ter
        # limiar (ver sim/inhibition_gate.py).
        taxa[e["inhib"]] = TONIC_INHIB_HZ
        return taxa, {"looming": {"L": float(hz[0]), "R": float(hz[1])},
                      "input_hz": {"L": float(hz[0]), "R": float(hz[1])},
                      "distance_mm": round(float(self.dist), 2)}

    def decide(self, saida, t_s):
        eventos = []
        if saida.sum() and t_s > self.escape_ate:
            self.escape_ate = t_s + self.ESCAPE_MS / 1000.0
            self.escapes += 1
            eventos.append(("escape_triggered",
                            {"distance_mm": round(float(self.dist), 2),
                             "ttmn_spikes": int(saida.sum())}))
        drive = (np.array([self.ESCAPE_DRIVE, self.ESCAPE_DRIVE])
                 if t_s < self.escape_ate else np.array([BASE_DRIVE, BASE_DRIVE]))
        return drive, eventos

    # ------------------------------------------------------------ descricao

    def telemetry_metadata(self):
        p = self.base_parameters()
        p.update({"dark_threshold": self.DARK_THRESHOLD,
                  "loom_gain": self.LOOM_GAIN, "loom_max_hz": self.LOOM_MAX_HZ,
                  "tonic_inhib_hz": TONIC_INHIB_HZ,
                  "escape_drive": self.ESCAPE_DRIVE,
                  "escape_ms": self.ESCAPE_MS, "cycle_s": self.CICLO_S})
        return {
            "parameters": p,
            "circuits": self.circuito.descreve(),
            "provenance": {
                # medido no conectoma
                "body_ids": protocol.DATA, "side": protocol.DATA,
                "type": protocol.DATA, "neurotransmitter": protocol.DATA,
                "synapse_weight": protocol.DATA,
                # sai do modelo de Shiu et al. rodando sobre esse dado
                "membrane_potential": protocol.MODEL, "spikes": protocol.MODEL,
                "synaptic_conductance": protocol.MODEL, "firing_rate": protocol.MODEL,
                "synapse_sign": protocol.MODEL,
                # escolha nossa, nao esta em lugar nenhum do dado
                "loom_gain": protocol.ASSUMPTION,
                "dark_fraction_detector": protocol.ASSUMPTION,
                "tonic_inhib_hz": protocol.ASSUMPTION,
                "escape_drive": protocol.ASSUMPTION,
                "motor_to_gait_mapping": protocol.ASSUMPTION,
            },
        }

    def scene_metadata(self):
        return {
            "arena": {"kind": "looming_sphere"},
            "stimulus": {"kind": "approaching_sphere", "radius": 3.0,
                         "start_distance": self.DIST_LONGE,
                         "end_distance": self.DIST_PERTO,
                         "cycle_s": self.CICLO_S},
        }

    def results(self):
        r = super().results()
        r["escapes"] = self.escapes
        return r
