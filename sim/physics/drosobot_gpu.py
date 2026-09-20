r"""
Adaptador do Drosobot GPU Physics: o corpo, com a dinamica na placa.

    corpo = physics.cria("drosobot-gpu", arena="looming")
    corpo.reset(seed=0)
    frame = corpo.passo(motor)

Do lado de fora e o mesmo `PhysicsAdapter` do FlyGym: `SensorFrame` entra,
`MotorFrame` sai, e o motor neural nao sabe qual dos dois esta rodando. Do lado
de dentro, `state(t) -> state(t + dt)` acontece inteiramente nos kernels de
`sim/gpu_physics/`.

## O que o MuJoCo ainda faz, e o que ele NAO faz

FAZ: ler o MJCF, compilar o `mjModel`, carregar malhas, e renderizar a retina.
Nada disso e etapa dinamica -- e leitura de arquivo, aritmetica de compilacao e
rasterizacao.

NAO FAZ: cinematica, inercia, colisao, restricao, solver, integracao. Nenhuma
chamada a `mj_step`, `mj_forward` ou qualquer `mj_*` de dinamica acontece no
laco. O `mjData` existe porque o renderizador da retina o exige, e recebe
`qpos`/`qvel` COPIADOS da GPU -- ele e destino, nunca fonte.

## O custo que este desenho aceita

O controlador de marcha e Python e roda por passo. Ele precisa de tres coisas
que a GPU tem: posicao dos segmentos, altura do torax e forca de contato nos
segmentos de tropeço. Isso e uma leitura de volta por passo -- ~300 numeros --
e ela existe porque o controlador ainda nao esta na GPU.

E deliberado nao esconder isso: `resumo()` reporta `leituras_por_passo`, e esse
numero e o que a proxima fase tem que zerar para o laco fechado nao voltar para
o Python a 10 kHz.

## Retina

A retina roda no renderizador do FlyGym, que le `mjData`. A sincronizacao
GPU -> `mjData` acontece na cadencia da RETINA (100 Hz), nao na da fisica
(10.000 Hz) -- e `mj_kinematics` e chamado nessa hora, porque o renderizador
precisa dos quadros dos corpos e isso e cinematica para DESENHAR, nao para
integrar. A fisica ja avancou sem ele.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

from .adapter import MotorFrame, SensorFrame     # noqa: E402
from .flygym2 import VISION_HZ, FlyGym2Adapter   # noqa: E402


class DrosobotGPUAdapter(FlyGym2Adapter):
    """
    Monta o modelo como o FlyGym 2.x e integra com os kernels proprios.

    Herda a MONTAGEM -- arena, mosca, atuadores, pares de colisao, controlador,
    retina -- porque montar de novo criaria um segundo corpo, e a comparacao
    com o MuJoCo mediria dois modelos em vez de dois solvers. O que a subclasse
    substitui e so o passo.
    """

    nome = "drosobot-gpu"
    versao = "Drosobot GPU Physics / OpenCL"

    def __init__(self, *a, fp64: bool = True, **kw):
        super().__init__(*a, **kw)
        self.fp64 = bool(fp64)
        self.motor = None
        self.device_gpu = None
        self.precisao = "fp64" if fp64 else "fp32"
        self._leituras = 0
        self._passos = 0

    # ------------------------------------------------------------- montagem

    def reset(self, seed: int = 0) -> SensorFrame:
        from gpu_physics.compilador import compila
        from gpu_physics.dinamica import MotorFisicoGPU

        quadro = super().reset(seed)
        m = self.sim.mj_model

        # `compila` RECUSA o que os kernels nao cobrem -- cilindro x malha,
        # `pair_gap` nao nulo, tendao, junta ball. A recusa vem com o nome do
        # recurso, e e melhor que uma corrida silenciosa com zero contatos.
        self.mod = compila(m)
        self.motor = MotorFisicoGPU(self.mod, fp64=self.fp64)
        self.device_gpu = self.motor.dev
        self._tabelas_de_forca()
        self._escreve_estado_inicial()
        self._leituras = 0
        self._passos = 0
        return quadro

    def _tabelas_de_forca(self) -> None:
        """geom -> indice do segmento, e quais geoms sao chao. Uma vez."""
        g = self.motor
        d = g.dev
        ngeom = int(self.sim.mj_model.ngeom)
        saida = np.full(ngeom, -1, dtype=np.int32)
        for geom, i in self._forcas.geom_para_saida.items():
            saida[int(geom)] = i
        chao = np.zeros(ngeom, dtype=np.int32)
        chao[np.asarray(self._forcas.chao, dtype=np.int32)] = 1
        self._nseg = len(self._forcas.geom_para_saida)
        g.b["geom_saida"] = d.sobe(saida)
        g.b["geom_chao"] = d.sobe(chao)
        g.tam["forcas_seg"] = 3 * self._nseg
        g.b["forcas_seg"] = d.vazio(3 * self._nseg, g.real)
        ARQ = __import__("gpu_physics.dinamica", fromlist=["ARQUIVOS"]).ARQUIVOS
        g.k["forcas_segmentos"] = d.kernel_proprio(
            ARQ, "forcas_segmentos", g.fp64, g._defines)
        g.k["empacota_observacao"] = d.kernel_proprio(
            ARQ, "empacota_observacao", g.fp64, g._defines)
        # Tudo que o controlador le por passo, num buffer so: eram tres
        # `enqueue_copy` bloqueantes, cada um com a sua espera pela placa.
        self._nfly = int(self._bodyids_fly.size)
        self._npacote = 3*self._nfly + 9 + 3*self._nseg
        g.b["bodyids_fly"] = d.sobe(self._bodyids_fly.astype(np.int32))
        g.tam["obs_pacote"] = self._npacote
        g.b["obs_pacote"] = d.vazio(self._npacote, g.real)

    def _escreve_estado_inicial(self) -> None:
        d = self.sim.mj_data
        self.motor.escreve_estado(qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
                                  mocap_pos=d.mocap_pos,
                                  mocap_quat=d.mocap_quat)

    # ---------------------------------------------------------------- passo

    def _le_observacao(self):
        """
        O que o controlador precisa, numa leitura so por passo.

        `xpos` para as alturas, `xmat` do torax para a direcao, e a forca de
        contato por segmento -- que a GPU ja reduziu a `nseg x 3` para nao
        obrigar a trazer o estado de restricao inteiro.
        """
        from flygym_demo.complex_terrain import HybridControllerObservation

        g = self.motor
        g._roda("forcas_segmentos", self._nseg,
                (np.int32(self._nseg), g.b["ncon"], g.b["con_geom"],
                 g.b["con_efcadr"], g.b["con_pair"], g.b["pair_friction"],
                 g.b["con_frame"], g.b["efc_force"], g.b["geom_saida"],
                 g.b["geom_chao"], np.int32(1), g.b["forcas_seg"]))
        g._roda("empacota_observacao", max(self._nfly, 3*self._nseg, 9),
                (np.int32(self._nfly), np.int32(self._nseg),
                 np.int32(self._bodyid_torax), g.b["bodyids_fly"],
                 g.b["xpos"], g.b["xmat"], g.b["forcas_seg"],
                 g.b["obs_pacote"]))
        pac = g.le("obs_pacote")
        self._leituras += 1
        n = self._nfly
        pos = pac[:3*n].reshape(n, 3)
        heading = pac[3*n:3*n+9].reshape(3, 3)[:, 0].copy()
        forcas = pac[3*n+9:].reshape(self._nseg, 3)
        self._pos_cache = pos
        return HybridControllerObservation(
            thorax_z=float(pos[self._i_torax, 2]),
            tarsus5_z=pos[self._i_tarsus5, 2].astype(float),
            stumbling_contact_forces=forcas.reshape(
                len(self._legs), len(self._stumbling_links), 3),
            fly_heading=heading,
        )

    def _prepara_indices(self) -> None:
        super()._prepara_indices()
        # mapeamento do indice de corpo da mosca -> id global, para ler `xpos`
        # da GPU sem depender do gather do FlyGym (que le `mjData`)
        self._bodyids_fly = np.asarray(
            self.sim._internal_bodyids_by_fly[self.fly.name], dtype=np.int64)

    def passo(self, motor: MotorFrame) -> SensorFrame:
        from flygym.compose import ActuatorType
        from flygym_demo.complex_terrain import apply_locomotion_action

        obs = self._le_observacao()
        acao = self._rapido.step(np.asarray(motor.drive, dtype=float), obs)
        # `apply_locomotion_action` escreve em `mj_data.ctrl`; de la o vetor vai
        # para a GPU. O `mjData` e caixa de correio do controlador, nao estado.
        apply_locomotion_action(self.sim, self.fly.name, acao,
                                actuator_type=ActuatorType.POSITION)
        self.motor.escreve_estado(ctrl=self.sim.mj_data.ctrl)
        # `esperar=False`: quem sincroniza e a leitura do pacote no proximo
        # passo. Esperar aqui pagaria a ida e volta duas vezes.
        self.motor.passo_fundido(esperar=False)
        self._passo += 1
        self._passos += 1

        atualizou = self.com_visao and (self._passo % self._passos_por_retina == 0)
        if atualizou:
            self._sincroniza_para_render()
        return self._quadro(atualizou)

    def _sincroniza_para_render(self) -> None:
        """
        GPU -> `mjData`, na cadencia da RETINA.

        `mj_kinematics` roda aqui porque o renderizador precisa dos quadros dos
        corpos para desenhar. Nao e integracao: a fisica ja avancou sem ele, e o
        que entra e o `qpos` que a GPU produziu.
        """
        import mujoco as mj

        d = self.sim.mj_data
        d.qpos[:] = self.motor.le("qpos")
        d.qvel[:] = self.motor.le("qvel")
        self._leituras += 2
        mj.mj_kinematics(self.sim.mj_model, d)

    def antes_do_passo(self, t_s: float, dist_mm: float | None = None) -> None:
        super().antes_do_passo(t_s, dist_mm)
        if self._mocapid is not None:
            d = self.sim.mj_data
            self.motor.escreve_estado(mocap_pos=d.mocap_pos,
                                      mocap_quat=d.mocap_quat)

    def _quadro(self, retina_atualizou: bool) -> SensorFrame:
        retina = None
        if retina_atualizou and self.com_visao:
            leituras = self.sim.get_ommatidia_readouts(self.fly.name)
            retina = np.asarray(leituras).mean(axis=2)
            self._retina_cache = retina
        pos = getattr(self, "_pos_cache", None)
        if pos is None:
            pos = self.sim.get_body_positions(self.fly.name)
        return SensorFrame(
            t_s=self._passo * self._dt, passo=self._passo,
            retina=retina, retina_atualizou=retina_atualizou,
            posicao=np.asarray(pos[self._i_torax], dtype=float),
        )

    def pose_corpo(self):
        """Pose dos segmentos, da GPU. Cadencia da telemetria, nao da fisica."""
        nomes = [getattr(b, "name", str(b))
                 for b in self.fly.get_bodysegs_order()]
        xpos = self.motor.le("xpos").reshape(-1, 3)
        xquat = self.motor.le("xquat").reshape(-1, 4)
        self._leituras += 2
        ids = self._bodyids_fly
        return nomes, xpos[ids], xquat[ids]

    # --------------------------------------------------------------- resumo

    def resumo(self) -> dict:
        r = super().resumo()
        cap = self.device_gpu.capacidades() if self.device_gpu else {}
        r.update({
            "physics_backend": f"Drosobot GPU Physics ({self.precisao})",
            "adapter": self.nome,
            "device": cap.get("nome"),
            "precisao": self.precisao,
            "physics_model_hash": getattr(self.mod, "hash_modelo", None),
            # Quantas leituras de volta a CPU fez por passo. E o numero que a
            # fase de laco residente tem que zerar.
            "leituras_por_passo": round(self._leituras / max(1, self._passos), 2),
            "diferenca_declarada": (
                "dinamica inteiramente nos kernels proprios; o MuJoCo le o "
                "MJCF, compila o mjModel e renderiza a retina, e nao executa "
                "nenhuma etapa dinamica"),
        })
        return r

    def fecha(self) -> None:
        self.motor = None
        super().fecha()
