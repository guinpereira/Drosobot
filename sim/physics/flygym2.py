"""
Adaptador do FlyGym 2.x.

Roda no `.venv-flygym2` (Python 3.14, mujoco 3.9). O `.venv` de referencia fica
intocado -- 2.x nao e upgrade de 1.x, e outra API sobre outra base, e rodar os
dois no mesmo ambiente mudaria a fisica dos experimentos existentes.

## O que muda em relacao ao 1.x, e como o adaptador absorve

    montagem      declarativa: Skeleton -> add_joints -> add_actuators ->
                  add_leg_adhesion, e `Simulation(world)` no fim
    passo         `sim.step()` e so `mj_step`; a observacao virou PULL
    visao         `get_ommatidia_readouts()` sob demanda; nao ha flag
                  `vision_updated`, entao a cadencia da retina e nossa
    nomes         `LFTarsus1` virou `lf_tarsus1`
    controlador   `HybridTurningController` existe e mantem o mesmo sinal
                  descendente de 2 elementos

A cadencia da retina e imposta aqui pra bater com os 100 Hz do 1.x -- sem isso
os dois caminhos veriam estimulo diferente e a comparacao nao valeria.

## O que NAO muda

Timestep, pose inicial, semantica do drive e parametros do experimento. A
comparacao FlyGym1 x FlyGym2 so significa alguma coisa se a unica variavel for o
simulador.
"""
from __future__ import annotations

import mujoco as mj
import numpy as np

from .adapter import MotorFrame, SensorFrame

VISION_HZ = 100


class FlyGym2Adapter:
    nome = "flygym2"
    versao = "2.1.0 / mujoco 3.9"

    def __init__(self, arena: str = "looming", self_collisions: str = "legs",
                 timestep: float = 1e-4, com_visao: bool = True):
        self.arena_tipo = arena
        self.self_collisions = self_collisions
        self._dt = timestep
        self.com_visao = com_visao
        self.sim = None
        self.fly = None
        self.controlador = None
        self._passo = 0
        self._retina_cache = None
        self._pos_cache = None
        self._nome_obj = None
        self._estimulo = None
        self._mocapid = None
        self._dist_estimulo = None
        self._passos_por_retina = max(1, int(round(1.0 / (VISION_HZ * timestep))))

    def reset(self, seed: int = 0) -> SensorFrame:
        from flygym.anatomy import (ActuatedDOFPreset, AxisOrder, JointPreset,
                                    Skeleton)
        from flygym.compose import (ActuatorType, FlatGroundWorld,
                                    KinematicPosePreset, NeuroMechFly)
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import HybridTurningController

        neutral = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL)
        skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                            joint_preset=JointPreset.LEGS_ONLY)
        fly = NeuroMechFly(name="fly")
        fly.add_joints(skeleton, neutral_pose=neutral)
        dofs = skeleton.get_actuated_dofs_from_preset(
            ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        fly.add_actuators(dofs, ActuatorType.POSITION, neutral_input=neutral,
                          kp=45.0, forcerange=(-65.0, 65.0))
        fly.add_leg_adhesion(gain=40.0)
        if self.com_visao:
            fly.add_vision()

        if self.arena_tipo == "looming":
            from .looming_world import Estimulo, constroi_mundo_looming
            world, self._nome_obj = constroi_mundo_looming()
            self._estimulo = Estimulo()
        else:
            world = FlatGroundWorld()
            self._nome_obj = None
            self._estimulo = None
        world.add_fly(fly, (0, 0, 0.8), Rotation3D("quat", (1, 0, 0, 0)))

        self.fly = fly
        self.sim = Simulation(world, timestep=self._dt)
        self.sim.reset()
        # deixa a mosca assentar no chao antes de medir qualquer coisa -- o 1.x
        # faz o equivalente no proprio reset
        self.sim.warmup()
        if self._nome_obj is not None:
            # id do corpo mocap, resolvido uma vez. `mocap_pos` e indexado pelo
            # indice de MOCAP, nao pelo bodyid -- body_mocapid faz a traducao.
            bid = mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_BODY,
                                self._nome_obj)
            self._mocapid = int(self.sim.mj_model.body_mocapid[bid])
        self.controlador = HybridTurningController(
            timestep=self._dt,
            output_dof_order=fly.get_actuated_jointdofs_order(
                ActuatorType.POSITION))
        self._passo = 0
        self._retina_cache = None
        self._pos_cache = None
        self._prepara_indices()
        return self._quadro(True)

    def _prepara_indices(self) -> None:
        """
        Pre-calcula os indices que a observacao do controlador precisa.

        `HybridControllerObservation.from_sim()` faz `body_order.index(...)` --
        busca linear numa lista -- pra cada uma das 6 pernas, a cada passo de
        fisica. Sao 10.000 passos por segundo de mosca, e isso aparece: o passo
        custava 3425 us contra 2130 us do FlyGym 1.x.
        """
        from flygym.anatomy import LEGS
        from flygym_demo.complex_terrain.hybrid_controller import (
            _DETECTED_STUMBLING_LINKS)

        cls = type(self.fly).BODY_SEGMENT_CLASS
        ordem = self.fly.get_bodysegs_order()
        self._legs = tuple(LEGS)
        self._stumbling_links = _DETECTED_STUMBLING_LINKS
        self._i_torax = ordem.index(cls("c_thorax"))
        self._i_tarsus5 = [ordem.index(cls(f"{p}_tarsus5")) for p in self._legs]
        self._segs_stumbling = [cls(f"{p}_{l}") for p in self._legs
                                for l in self._stumbling_links]
        self._bodyid_torax = self.sim._internal_bodyids_by_fly[
            self.fly.name][self._i_torax]

    def _observacao(self):
        """A mesma observacao de `from_sim`, com os indices ja resolvidos."""
        from flygym_demo.complex_terrain import HybridControllerObservation

        pos = self.sim.get_body_positions(self.fly.name)
        forcas = self.sim.get_bodysegment_contact_forces(
            self.fly.name, self._segs_stumbling, ground_only=True
        ).reshape(len(self._legs), len(self._stumbling_links), 3)
        self._pos_cache = pos
        return HybridControllerObservation(
            thorax_z=float(pos[self._i_torax, 2]),
            tarsus5_z=pos[self._i_tarsus5, 2].astype(float),
            stumbling_contact_forces=forcas,
            fly_heading=self.sim.mj_data.xmat[self._bodyid_torax]
            .reshape(3, 3)[:, 0].copy(),
        )

    def antes_do_passo(self, t_s: float, dist_mm: float | None = None) -> None:
        """Move a esfera de looming. Mocap: posicao imposta, sem dinamica."""
        if self._estimulo is None or self._mocapid is None:
            return
        pos = getattr(self, "_pos_cache", None)
        if pos is None:
            pos = self.sim.get_body_positions(self.fly.name)
        alvo = self._estimulo.posicao(t_s, pos[self._i_torax])
        self.sim.mj_data.mocap_pos[self._mocapid] = alvo
        self._dist_estimulo = float(self._estimulo.distancia(t_s))

    def passo(self, motor: MotorFrame) -> SensorFrame:
        from flygym.compose import ActuatorType
        from flygym_demo.complex_terrain import apply_locomotion_action

        obs = self._observacao()
        acao = self.controlador.step(np.asarray(motor.drive, dtype=float), obs)
        apply_locomotion_action(self.sim, self.fly.name, acao,
                                actuator_type=ActuatorType.POSITION)
        self.sim.step()
        self._passo += 1

        # a cadencia da retina e NOSSA aqui: o 2.x nao avisa quando ela mudou
        atualizou = self.com_visao and (self._passo % self._passos_por_retina == 0)
        return self._quadro(atualizou)

    def _quadro(self, retina_atualizou: bool) -> SensorFrame:
        retina = None
        if retina_atualizou and self.com_visao:
            # (2, n_omatideos, 2) -> (2, n): media dos canais amarelo/pale, o
            # mesmo reducao que o caminho 1.x faz
            leituras = self.sim.get_ommatidia_readouts(self.fly.name)
            retina = np.asarray(leituras).mean(axis=2)
            self._retina_cache = retina
        # a posicao ja veio junto com a observacao do controlador; buscar de
        # novo seria um segundo gather pelo mesmo dado a cada passo
        pos = getattr(self, "_pos_cache", None)
        if pos is None:
            pos = self.sim.get_body_positions(self.fly.name)
        return SensorFrame(
            t_s=self._passo * self._dt, passo=self._passo,
            retina=retina, retina_atualizou=retina_atualizou,
            posicao=np.asarray(pos[self._i_torax], dtype=float),
        )

    @property
    def timestep(self) -> float:
        return self._dt

    @property
    def n_pares_colisao(self) -> int:
        return int(self.sim.mj_model.npair) if self.sim else 0

    def pose_corpo(self):
        """
        Pose dos segmentos da mosca, com os nomes do 2.x (`lf_tarsus1`).

        Nomes diferentes dos do 1.x de proposito: quem consome e a Unity, e ela
        recebe a lista junto. Traduzir aqui pra nomenclatura antiga esconderia
        qual modelo esta rodando.
        """
        # BodySegment tem repr proprio; queremos o nome, nao o repr
        nomes = [getattr(b, "name", str(b))
                 for b in self.fly.get_bodysegs_order()]
        pos = np.asarray(self.sim.get_body_positions(self.fly.name))
        quat = np.asarray(self.sim.get_body_rotations(self.fly.name))
        return nomes, pos, quat

    def resumo(self) -> dict:
        m = self.sim.mj_model if self.sim else None
        return {
            "physics_backend": f"FlyGym {self.versao}",
            "adapter": self.nome,
            "timestep": self._dt,
            "collision_set": "padrao 2.x",
            "collision_pairs": self.n_pares_colisao,
            "nv": int(m.nv) if m is not None else 0,
            "vision_hz": VISION_HZ if self.com_visao else 0,
            "arena": self.arena_tipo,
            "diferenca_declarada": (
                "arena de looming reproduzida com esfera mocap; o chao e o "
                "FlatGroundWorld do 2.x, com os parametros de contato dele, "
                "nao os da MovingObjArena do 1.x"
                if self.arena_tipo == "looming" else "chao plano"),
        }

    def fecha(self) -> None:
        self.sim = None
