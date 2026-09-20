"""
Runner do Drosobot Lab: um processo, varios experimentos, escolhidos de fora.

    .venv\\Scripts\\python sim\\lab_runner.py                      # espera a Unity escolher
    .venv\\Scripts\\python sim\\lab_runner.py --experiment looming_escape --start
    .venv\\Scripts\\python sim\\lab_runner.py --viewer --record    # janela do MuJoCo + grava
    .venv\\Scripts\\python sim\\lab_runner.py --lista               # so imprime o catalogo

Antes disto, trocar de experimento era matar o Python e rodar outro script. Os
CSVs do conectoma levam quase um segundo pra ler e a rede precisa ser remontada,
entao fazer isso pela interface exigia um processo que sobrevivesse a troca. E o
que este arquivo e: um laco que segura o estado, ouve o canal de controle e roda
o experimento escolhido.

## Quem manda em que

    MuJoCo / FlyGym    fisica. Posicao, contato, velocidade sao lidos de `obs`.
    circuito           comportamento. O drive motor sai de spike, sempre.
    controle (Unity)   qual experimento, com que semente, e quando comeca.

A terceira linha e a unica coisa nova. Ela nao encosta nas outras duas: nao ha
comando que altere peso, limiar ou drive. Ver sim/telemetry/control.py.

Os scripts headless (flygym_escape.py, flygym_optomotor.py, flygym_avoidance.py)
continuam existindo e continuam sendo o que produz as figuras do README. Este
runner roda os MESMOS circuitos com os MESMOS parametros -- divergencia entre os
dois e bug.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import experiments  # noqa: E402
from telemetry import protocol  # noqa: E402
from telemetry.control import ServidorControle  # noqa: E402
from telemetry.control import abrir as abrir_controle  # noqa: E402
from telemetry.recorder import Gravador  # noqa: E402
from telemetry.server import abrir as abrir_telemetria  # noqa: E402

# a retina sao 1442 floats: e a mensagem mais cara, entao vai a cada N janelas
RETINA_A_CADA = 5


class Runner:
    """Maquina de estados: idle -> loading -> running <-> paused -> finished."""

    def __init__(self, tel, ctl, gravar=False, viewer=False):
        self.tel = tel
        self.ctl = ctl
        self.gravar = gravar
        self.quer_viewer = viewer

        self.estado = "idle"
        self.exp = None
        self.exp_id = None
        self.seed = 0
        self.sair = False

        self._t_inicio = 0.0
        self._n_retina = 0
        self._viewer = None
        self._tel_base = tel     # sem o Gravador em volta

        self._publica_catalogo()
        self._publica_estado()

    # ----------------------------------------------------------- publicacao

    def _publica_catalogo(self):
        self.tel.enviar(protocol.experiment_list(
            experiments.disponiveis(), current=self.exp_id))

    def _publica_estado(self, **detalhe):
        self.tel.enviar(protocol.run_state(
            self.estado, experiment_id=self.exp_id, seed=self.seed,
            sim_time=self.exp.sim_time if self.exp else 0.0,
            step=self.exp.step_index if self.exp else 0,
            detail=detalhe))

    def _publica_descricao(self):
        meta = self.exp.telemetry_metadata()
        cena = self.exp.scene_metadata()
        extra = {k: v for k, v in meta.items()
                 if k not in ("parameters", "provenance", "circuits")}
        self.tel.enviar(protocol.experiment_info(
            experiment_id=self.exp.id, name=self.exp.name,
            description=self.exp.description,
            parameters=meta.get("parameters", {}),
            provenance=meta.get("provenance", {}),
            circuits=meta.get("circuits", []),
            **extra))
        self.tel.enviar(protocol.scene_info(
            obstacles=cena.get("obstacles", []),
            arena=cena.get("arena", {}),
            stimulus=cena.get("stimulus", {})))

    # -------------------------------------------------------------- comandos

    def _trata(self, msg):
        cmd = msg.get("command")
        sock = msg.get("_sock")

        if cmd == "list":
            self._publica_catalogo()
            self.ctl.responder(sock, True, experiments=experiments.disponiveis(),
                               current=self.exp_id, state=self.estado)

        elif cmd == "select":
            alvo = msg.get("experiment_id")
            if alvo not in {e["id"] for e in experiments.disponiveis()}:
                self.ctl.responder(sock, False, error=f"id desconhecido: {alvo!r}")
                return
            self._descarta()
            self.exp_id = alvo
            self.seed = int(msg.get("seed", self.seed))
            self.estado = "idle"
            self._publica_estado()
            self._publica_catalogo()
            self.ctl.responder(sock, True, experiment_id=self.exp_id, seed=self.seed)

        elif cmd == "start":
            if msg.get("experiment_id"):
                self.exp_id = msg["experiment_id"]
            if "seed" in msg:
                self.seed = int(msg["seed"])
            if not self.exp_id:
                self.ctl.responder(sock, False, error="nenhum experimento selecionado")
                return
            # O estado vira `loading` ANTES do ack. Se o ack fosse primeiro, o
            # cliente voltaria a olhar o `run_state` e ainda encontraria o
            # experimento ANTERIOR marcado como `running` -- e trataria a corrida
            # velha como se fosse a nova.
            self.estado = "loading"
            self._publica_estado(message="montando arena, mosca e circuito")
            # Ack ANTES de montar, porem. Ler os CSVs e construir arena e mosca
            # leva de 1 a 10 segundos, e segurar a conexao de controle durante
            # isso faz o cliente estourar o timeout achando que o runner morreu.
            self.ctl.responder(sock, True, experiment_id=self.exp_id,
                               seed=self.seed, state="loading")
            self._monta()

        elif cmd == "pause":
            if self.estado == "running":
                self.estado = "paused"
                self._publica_estado()
            self.ctl.responder(sock, self.estado == "paused", state=self.estado)

        elif cmd == "resume":
            if self.estado == "paused":
                self.estado = "running"
                # o relogio de parede precisa esquecer a pausa, senao o fator de
                # tempo real desaba e parece que a simulacao ficou lenta
                self._t_inicio = time.time() - self.exp.sim_time / max(1e-9, self._rtf_alvo)
                self._publica_estado()
            self.ctl.responder(sock, self.estado == "running", state=self.estado)

        elif cmd == "reset":
            if "seed" in msg:
                self.seed = int(msg["seed"])
            if self.exp is None:
                self.ctl.responder(sock, True, seed=self.seed, state="loading")
                self._monta()
                return
            self.estado = "loading"
            self._publica_estado()
            self.ctl.responder(sock, True, seed=self.seed, state="loading")
            try:
                self.exp.reset(self.seed)
            except Exception as e:      # noqa: BLE001
                self._falha(e)
                return
            self._comeca_corrida()
            self.tel.enviar(protocol.event(0.0, "experiment_reset",
                                           {"seed": self.seed}))

        elif cmd == "stop":
            self._encerra("parado pela interface")
            self.ctl.responder(sock, True, state=self.estado)

        elif cmd == "quit":
            self.ctl.responder(sock, True)
            self._encerra("processo encerrado")
            self.sair = True

    # -------------------------------------------------------------- corrida

    def _monta(self):
        self._descarta()
        self.estado = "loading"
        # Publicado ANTES de montar: ler os CSVs e construir arena e mosca leva
        # alguns segundos, e sem este aviso a interface parece travada.
        self._publica_estado(message="montando arena, mosca e circuito")
        print(f"[runner] montando {self.exp_id} (seed={self.seed})...")
        try:
            self.exp = experiments.criar(self.exp_id, seed=self.seed)
            self.exp.setup(self.seed)
        except Exception as e:          # noqa: BLE001
            self._falha(e)
            return False, str(e)

        if self.gravar:
            self.tel = Gravador(self._tel_base, self.exp_id,
                                metadata={"experiment_id": self.exp_id,
                                          "seed": self.seed},
                                fechar_sink=False)
        self._publica_descricao()
        self._abre_viewer()
        self._comeca_corrida()
        print(f"[runner] {self.exp_id} rodando")
        return True, None

    def _comeca_corrida(self):
        self._t_inicio = time.time()
        self._n_retina = 0
        self._rtf_alvo = 1.0
        self.estado = "running"
        self._publica_estado()
        self._publica_catalogo()

    def _falha(self, e):
        traceback.print_exc()
        self.estado = "error"
        self._publica_estado(message=str(e))

    def _descarta(self):
        self._fecha_viewer()
        # Soltar a referencia nao basta: o flygym segura contexto de render por
        # mosca, e num processo que troca de experimento varias vezes isso vaza e
        # chega a impedir o Python de terminar. close() devolve os recursos.
        if self.exp is not None and getattr(self.exp, "sim", None) is not None:
            try:
                self.exp.sim.close()
            except Exception:           # noqa: BLE001
                pass
        self.exp = None

    def _encerra(self, razao):
        if self.exp is not None:
            self.tel.enviar(protocol.statistics(self.exp.sim_time, self.exp.results()))
        if isinstance(self.tel, Gravador):
            if self.exp is not None:
                self.tel.escrever_resumo(self.exp.results())
            self.tel.fechar()
            self.tel = self._tel_base
        self._descarta()
        self.estado = "idle"
        self._publica_estado(message=razao)
        self._publica_catalogo()

    # ---------------------------------------------------------------- viewer

    def _abre_viewer(self):
        if not self.quer_viewer:
            return
        import mujoco
        import mujoco.viewer
        modelo = self.exp.sim.physics.model.ptr
        torax = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, "0/Thorax")
        self._viewer = mujoco.viewer.launch_passive(
            modelo, self.exp.sim.physics.data.ptr)
        # camera colada na mosca: sem isso ela vira um ponto no meio de uma arena
        # enorme e a cena parece parada
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._viewer.cam.trackbodyid = torax
        self._viewer.cam.distance = 12.0
        self._viewer.cam.elevation = -35.0
        self._viewer.cam.azimuth = 135.0

    def _fecha_viewer(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:           # noqa: BLE001
                pass
            self._viewer = None

    # ------------------------------------------------------------------ laco

    def passo(self):
        """Um passo de fisica mais a telemetria dele. Devolve False pra parar."""
        if self.estado != "running" or self.exp is None:
            time.sleep(0.02)            # pausado/parado nao queima CPU
            return True

        try:
            saida = self.exp.step()
        except Exception as e:          # noqa: BLE001
            self._falha(e)
            return True

        if "layers" not in saida:       # sem retina nova: nada a publicar
            return True

        t_s = self.exp.sim_time
        rel = time.time()
        decorrido = max(1e-9, rel - self._t_inicio)
        self._rtf_alvo = t_s / decorrido
        self.tel.enviar(protocol.frame(
            step=self.exp.step_index, sim_time=t_s, wall_time=rel,
            real_time_factor=self._rtf_alvo,
            position=saida["position"], drive=saida["drive"]))

        # A camada de entrada e Poisson: tem spike, nao tem potencial de membrana.
        # Os campos do modelo biofisico ficam de FORA dela em vez de irem zerados
        # -- mandar v_mV=0 pra quem nao tem membrana e inventar dado.
        self.tel.enviar(protocol.neural_activity(t_s, [
            {k: v for k, v in (
                ("name", c.get("name") or f"layer{c['index']}"),
                ("spikes", c["spikes"]),
                ("v_mV", c.get("v_mV")),
                ("g_mV", c.get("g_mV")),
                ("refractory", c.get("refratario")),
            ) if v is not None}
            for c in saida["layers"]]))

        for tipo, detalhe in saida.get("events", []):
            self.tel.enviar(protocol.event(t_s, tipo, detalhe))

        self._n_retina += 1
        if self._n_retina % RETINA_A_CADA == 0 and "retina" in saida:
            r = saida["retina"]
            self.tel.enviar_se_conectado(protocol.retina, t_s, r[0], r[1],
                                         saida.get("derived", {}))

        if self._viewer is not None:
            if not self._viewer.is_running():
                self._encerra("viewer fechado")
                return True
            self._viewer.sync()
        return True

    def laco(self):
        ultimo_log = time.time()
        while not self.sair:
            msg = self.ctl.proximo()
            while msg is not None:
                self._trata(msg)
                msg = self.ctl.proximo()

            self.passo()

            agora = time.time()
            if agora - ultimo_log > 2.0:
                ultimo_log = agora
                if self.estado == "running" and self.exp is not None:
                    print(f"[{self.exp_id}] t={self.exp.sim_time:6.2f}s  "
                          f"rtf={self._rtf_alvo:.3f}  "
                          f"saida={self.exp.total_saida}")


def main():
    ap = argparse.ArgumentParser(description="Drosobot Lab -- runner de experimentos")
    ap.add_argument("--experiment", default=None,
                    help="id inicial (looming_escape, optomotor_turning, obstacle_field)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", action="store_true",
                    help="ja comeca, em vez de esperar a interface")
    ap.add_argument("--porta", type=int, default=8765, help="telemetria")
    ap.add_argument("--porta-controle", type=int, default=8766)
    ap.add_argument("--sem-telemetria", action="store_true")
    ap.add_argument("--sem-controle", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="abre a janela do MuJoCo")
    ap.add_argument("--record", action="store_true", help="grava em runs/")
    ap.add_argument("--lista", action="store_true", help="imprime o catalogo e sai")
    args = ap.parse_args()

    if args.lista:
        for e in experiments.disponiveis():
            print(f"{e['id']:20s} {e['name']}")
            print(f"{'':20s} {e['description']}")
        return

    np.random.seed(args.seed)
    tel = abrir_telemetria(porta=args.porta, ativo=not args.sem_telemetria)
    ctl = abrir_controle(porta=args.porta_controle, ativo=not args.sem_controle)

    runner = Runner(tel, ctl, gravar=args.record, viewer=args.viewer)
    if args.experiment:
        runner.exp_id = args.experiment
        runner.seed = args.seed
        runner._publica_estado()
        if args.start:
            runner._monta()

    print()
    print("Drosobot Lab -- runner no ar.")
    # O banner diz o que de fato SUBIU, nao o que foi pedido. Porta ocupada faz
    # `abrir` devolver o objeto nulo e seguir sem derrubar a corrida -- util, mas
    # anunciar a porta assim mesmo ja custou uma hora de depuracao: a Unity
    # continuava lendo de um runner ANTIGO que ainda segurava a porta, e o banner
    # do novo dizia que estava tudo certo.
    print(f"  telemetria  127.0.0.1:{args.porta}   "
          f"{'(a Unity le daqui)' if tel.ativo else 'INDISPONIVEL -- porta ocupada'}")
    print(f"  controle    127.0.0.1:{args.porta_controle}   "
          f"{'(a Unity escolhe daqui)' if isinstance(ctl, ServidorControle) else 'INDISPONIVEL -- porta ocupada'}")
    if not tel.ativo:
        print("  ja existe um runner no ar nesta porta. Encerre o outro "
              "(taskkill /F /PID <pid>) ou use --porta/--porta-controle.")
    print("  experimentos:", ", ".join(e["id"] for e in experiments.disponiveis()))
    print("  Ctrl+C encerra.")
    print()

    try:
        runner.laco()
    except KeyboardInterrupt:
        print("\n[runner] Ctrl+C")
    finally:
        runner._encerra("runner encerrado")
        ctl.fechar()
        tel.enviar(protocol.bye("runner encerrado"))
        tel.fechar()
    print("encerrado.")


if __name__ == "__main__":
    main()
