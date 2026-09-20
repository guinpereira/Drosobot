"""
A parte de corpo que os tres experimentos compartilham.

Mosca, controlador de marcha, retina e o passo de fisica sao iguais nos tres --
o que muda e a arena, como a retina vira taxa de disparo e como o spike motor
vira comando de marcha. Essa separacao e o ponto: o corpo e o mesmo animal
biomecanico em todos, so o circuito e a tarefa mudam.

MuJoCo continua sendo a autoridade da fisica. Nada aqui calcula posicao,
velocidade ou contato por conta propria -- tudo vem de `obs`.
"""
from __future__ import annotations

import numpy as np
from flygym import Fly
from flygym.examples.locomotion import HybridTurningController

from .base import Experiment

CONTATOS = [f"{perna}{seg}"
            for perna in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]]

DT_FISICA = 1e-4
# retina a 100 Hz em vez dos 500 do NeuroMechFly: renderizar os dois olhos e o
# que mais custa no passo, e 100 Hz ja da ~80 amostras por ciclo de aproximacao.
VISION_HZ = 100
# janela em que a rede avanca entre duas leituras da retina
JANELA_MS = 10.0
BASE_DRIVE = 1.0


class ExperimentoFlygym(Experiment):
    """
    Base dos experimentos com corpo. Subclasse fornece:

        monta_arena()             a arena do MuJoCo
        monta_circuito()          um circuits.Circuito
        antes_do_passo(t_s)       mexe no estimulo, se houver
        processa_retina(retina)   -> (taxas por neuronio de entrada, derivados)
        decide(saida, t_s)        -> (drive, eventos)
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.sim = None
        self.circuito = None
        self.obs = None
        self.drive = np.array([BASE_DRIVE, BASE_DRIVE])
        self.retina = None
        self.derivados: dict = {}
        self.total_saida = 0
        self._rng = None
        self._retina_atualizou = False

    # ------------------------------------------------------------ montagem

    def setup(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        arena = self.monta_arena()
        fly = Fly(enable_vision=True, vision_refresh_rate=VISION_HZ,
                  spawn_pos=(0, 0, 0.3), contact_sensor_placements=CONTATOS)
        self.sim = HybridTurningController(fly=fly, arena=arena, timestep=DT_FISICA)
        self.obs, _ = self.sim.reset(seed=self.seed)
        self.arena = arena

        self.circuito = self.monta_circuito()
        self.retina = np.asarray(self.obs["vision"]).mean(axis=2)
        self.drive = np.array([BASE_DRIVE, BASE_DRIVE])
        self.total_saida = 0
        self.step_index = 0
        self.sim_time = 0.0
        self.prepara(self.retina)
        self._pronto = True

    def prepara(self, retina0) -> None:
        """Estado inicial dos detectores. Sobrescrever quando houver adaptacao."""

    # ---------------------------------------------------------------- passo

    def step(self) -> dict:
        t_s = self.step_index * DT_FISICA
        self.antes_do_passo(t_s)

        self.obs, _, _, _, info = self.sim.step(self.drive)
        self.step_index += 1
        self.sim_time = self.step_index * DT_FISICA

        saida_tel = {"position": self.obs["fly"][0], "drive": self.drive}
        self._retina_atualizou = bool(info.get("vision_updated", False))
        if not self._retina_atualizou:
            return saida_tel

        # A rede so avanca quando ha retina nova. Ela nao roda por passo de
        # fisica: seriam 10000 chamadas por segundo de mosca pra um sinal que so
        # muda 100 vezes.
        self.retina = np.asarray(self.obs["vision"]).mean(axis=2)
        taxas, self.derivados = self.processa_retina(self.retina)
        # rng proprio, semeado no setup. Os scripts ao vivo usavam o gerador
        # global do numpy; aqui a semente e escolhida na interface, e ela so
        # significa alguma coisa se o sorteio de spike vier dela tambem.
        contagem = self.circuito.rede.roda(JANELA_MS, taxas, rng=self._rng)
        saida = contagem[self.circuito.camada_saida]
        self.total_saida += int(saida.sum())

        self.drive, eventos = self.decide(saida, t_s)
        saida_tel["drive"] = self.drive
        camadas = self.circuito.rede.snapshot(self.circuito.nomes_snapshot())
        self.circuito.recorta_entrada(camadas[0])
        saida_tel["layers"] = camadas
        saida_tel["retina"] = self.retina
        saida_tel["derived"] = self.derivados
        if eventos:
            saida_tel["events"] = eventos
        return saida_tel

    @property
    def retina_atualizou(self) -> bool:
        return self._retina_atualizou

    # ------------------------------------------------- a cargo da subclasse

    def monta_arena(self):
        raise NotImplementedError

    def monta_circuito(self):
        raise NotImplementedError

    def antes_do_passo(self, t_s: float) -> None:
        pass

    def processa_retina(self, retina):
        raise NotImplementedError

    def decide(self, saida, t_s):
        raise NotImplementedError

    # --------------------------------------------------------------- comum

    def base_parameters(self) -> dict:
        return {"dt_physics_s": DT_FISICA, "dt_network_ms": 0.5,
                "vision_hz": VISION_HZ, "network_window_ms": JANELA_MS,
                "base_drive": BASE_DRIVE, "seed": self.seed}

    def results(self) -> dict:
        r = super().results()
        r["output_spikes"] = self.total_saida
        if self.obs is not None:
            r["position"] = [float(v) for v in np.asarray(self.obs["fly"][0])]
        return r
