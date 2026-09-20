"""
Adaptador do FlyGym 1.2.1 -- o caminho de REFERENCIA.

E o que todos os experimentos do Drosobot rodaram ate agora, e continua sendo a
referencia de comportamento contra a qual o FlyGym 2.x e comparado.

Roda no `.venv` (Python 3.11, mujoco 3.2.7). Nao mexer nesse ambiente.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from .adapter import MotorFrame, SensorFrame

VISION_HZ = 100
CONTATOS = [f"{p}{s}" for p in ("LF", "LM", "LH", "RF", "RM", "RH")
            for s in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")]


class FlyGym1Adapter:
    nome = "flygym1"
    versao = "1.2.1 / mujoco 3.2.7"

    def __init__(self, arena: str = "looming", self_collisions: str = "legs",
                 timestep: float = 1e-4, com_visao: bool = True):
        self.arena_tipo = arena
        self.self_collisions = self_collisions
        self._dt = timestep
        self.com_visao = com_visao
        self.sim = None
        self.arena = None
        self._obs = None
        self._passo = 0
        self._estimulo = None
        self._dist_estimulo = None
        self._nomes_seg = None

    def _monta_arena(self):
        if self.arena_tipo != "looming":
            from flygym.arena import FlatTerrain
            return FlatTerrain()
        import flygym.examples as fex
        spec = importlib.util.spec_from_file_location(
            "flygym_vision_arena",
            os.path.join(fex.__path__[0], "vision", "arena.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class LoomingArena(mod.MovingObjArena):
            def step(self, dt, physics):
                pass      # a posicao da esfera vem do laco principal

        return LoomingArena(obj_radius=3, init_ball_pos=(30, 0))

    def reset(self, seed: int = 0) -> SensorFrame:
        from flygym import Fly
        from flygym.examples.locomotion import HybridTurningController

        self.arena = self._monta_arena()
        fly = Fly(enable_vision=self.com_visao, vision_refresh_rate=VISION_HZ,
                  spawn_pos=(0, 0, 0.3), contact_sensor_placements=CONTATOS,
                  self_collisions=self.self_collisions)
        self.sim = HybridTurningController(fly=fly, arena=self.arena,
                                           timestep=self._dt)
        self._obs, _ = self.sim.reset(seed=seed)
        self._passo = 0
        return self._quadro(True)

    def antes_do_passo(self, t_s: float, dist_mm: float | None = None) -> None:
        """
        Move a esfera de looming.

        A regra de distancia vem de `looming_world.Estimulo`, a MESMA que o
        adaptador 2.x usa. Escrever a geometria do experimento duas vezes seria
        o jeito mais facil de os dois caminhos deixarem de rodar a mesma coisa
        sem ninguem notar.
        """
        if self.arena_tipo != "looming":
            return
        if self._estimulo is None:
            from .looming_world import Estimulo
            self._estimulo = Estimulo()
        pos = np.asarray(self._obs["fly"][0])
        alvo = self._estimulo.posicao(t_s, pos)
        self._dist_estimulo = float(self._estimulo.distancia(t_s))
        self.arena.ball_pos = alvo
        self.sim.physics.bind(self.arena.object_body).mocap_pos = alvo

    def passo(self, motor: MotorFrame) -> SensorFrame:
        self._obs, _, _, _, info = self.sim.step(np.asarray(motor.drive))
        self._passo += 1
        return self._quadro(bool(info.get("vision_updated", False)))

    def _quadro(self, retina_atualizou: bool) -> SensorFrame:
        retina = None
        if self.com_visao and retina_atualizou:
            # (2, 721, 2) -> (2, 721): media dos dois canais de omatideo, que e o
            # que os experimentos do Drosobot sempre usaram
            retina = np.asarray(self._obs["vision"]).mean(axis=2)
        return SensorFrame(
            t_s=self._passo * self._dt, passo=self._passo,
            retina=retina, retina_atualizou=retina_atualizou,
            posicao=np.asarray(self._obs["fly"][0], dtype=float),
        )

    @property
    def timestep(self) -> float:
        return self._dt

    @property
    def n_pares_colisao(self) -> int:
        return int(self.sim.physics.model.ptr.npair) if self.sim else 0

    def pose_corpo(self):
        """Pose de todos os corpos do modelo, direto do mjData."""
        import mujoco
        m = self.sim.physics.model.ptr
        d = self.sim.physics.data.ptr
        if self._nomes_seg is None:
            self._nomes_seg = [
                (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}")
                .replace("0/", "")
                for i in range(m.nbody)]
        return self._nomes_seg, d.xpos.copy(), d.xquat.copy()

    def resumo(self) -> dict:
        m = self.sim.physics.model.ptr if self.sim else None
        return {
            "physics_backend": f"FlyGym {self.versao}",
            "adapter": self.nome,
            "timestep": self._dt,
            "collision_set": self.self_collisions,
            "collision_pairs": self.n_pares_colisao,
            "nv": int(m.nv) if m is not None else 0,
            "vision_hz": VISION_HZ if self.com_visao else 0,
        }

    def fecha(self) -> None:
        if self.sim is not None:
            try:
                self.sim.close()
            except Exception:      # noqa: BLE001
                pass
            self.sim = None
