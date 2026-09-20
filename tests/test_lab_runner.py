"""
Testes do seletor de experimentos e do canal de controle.

Duas coisas precisam ficar garantidas aqui:

1. **O canal de controle nao pode virar uma porta dos fundos.** Ele existe pra
   escolher experimento e apertar start/pause/reset. Comando fora dessa lista tem
   que ser RECUSADO com erro visivel, nao ignorado em silencio -- ignorar treina
   a gente a achar que funcionou.

2. **Trocar de experimento nao pode mudar o experimento.** Cada um declara
   circuito, parametros e procedencia; o runner so transporta. Se o runner
   comecar a "ajustar" alguma coisa no caminho, isso aparece aqui.

Rodam com pytest ou direto:

    .venv\\Scripts\\python -m pytest tests/test_lab_runner.py -q
    .venv\\Scripts\\python tests/test_lab_runner.py

Os testes de ponta a ponta sobem o runner de verdade e montam a mosca no MuJoCo,
entao levam perto de um minuto. DROSOBOT_SKIP_LENTOS=1 pula so esses.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

import experiments  # noqa: E402
from telemetry import protocol  # noqa: E402
from telemetry.control import COMANDOS, ServidorControle, abrir  # noqa: E402

PULAR_LENTOS = os.environ.get("DROSOBOT_SKIP_LENTOS") == "1"
ESPERADOS = {"looming_escape", "optomotor_turning", "obstacle_field"}


def _porta_livre():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    porta = s.getsockname()[1]
    s.close()
    return porta


# ------------------------------------------------------------------ registro

def test_os_tres_experimentos_estao_no_catalogo():
    ids = {e["id"] for e in experiments.disponiveis()}
    assert ESPERADOS <= ids, f"faltando: {ESPERADOS - ids}"


def test_catalogo_tem_o_que_o_seletor_precisa():
    for e in experiments.disponiveis():
        assert e["id"] and e["name"] and e["description"], e


def test_id_desconhecido_da_erro():
    try:
        experiments.criar("teletransporte")
    except KeyError:
        return
    raise AssertionError("criar() aceitou um id que nao existe")


def test_obstacle_field_se_declara_negativo():
    """
    O resultado negativo tem que chegar na interface COMO negativo.

    Este e o teste que impede o caso mais facil de acontecer sem ninguem querer:
    alguem mexe nos ganhos, o campo some, e a interface passa a mostrar um desvio
    que parece funcionar.
    """
    exp = experiments.criar("obstacle_field")
    texto = (exp.name + " " + exp.description).lower()
    assert "negativ" in texto, f"nao se declara negativo: {texto!r}"
    assert exp.results()["outcome"] == "negative"


# -------------------------------------------------------------- controle

def test_comandos_permitidos_sao_poucos_e_declarados():
    """
    Se esta lista crescer, alguem esta empurrando decisao pra interface.

    Nada aqui pode escrever peso sinaptico, limiar ou drive motor: o conectoma e
    a autoridade do comportamento e o MuJoCo e a da fisica.
    """
    assert set(COMANDOS) == {"list", "select", "start", "pause", "resume",
                             "reset", "stop", "quit"}


def test_comando_desconhecido_e_recusado():
    porta = _porta_livre()
    srv = ServidorControle(porta=porta)
    try:
        c = socket.create_connection(("127.0.0.1", porta), timeout=5)
        f = c.makefile("r", encoding="utf-8")
        c.sendall(b'{"command":"set_synapse_weight","value":99}\n')
        resposta = json.loads(f.readline())
        assert resposta["ok"] is False
        assert "desconhecido" in resposta["error"]
        assert set(resposta["allowed"]) == set(COMANDOS)
        assert srv.proximo() is None, "comando recusado nao pode entrar na fila"
        c.close()
    finally:
        srv.fechar()


def test_comando_valido_entra_na_fila():
    porta = _porta_livre()
    srv = ServidorControle(porta=porta)
    try:
        c = socket.create_connection(("127.0.0.1", porta), timeout=5)
        c.sendall(b'{"command":"select","experiment_id":"looming_escape","seed":3}\n')
        for _ in range(50):
            msg = srv.proximo()
            if msg:
                break
            time.sleep(0.05)
        assert msg is not None and msg["command"] == "select"
        assert msg["seed"] == 3
        c.close()
    finally:
        srv.fechar()


def test_json_quebrado_nao_derruba_o_servidor():
    porta = _porta_livre()
    srv = ServidorControle(porta=porta)
    try:
        c = socket.create_connection(("127.0.0.1", porta), timeout=5)
        f = c.makefile("r", encoding="utf-8")
        c.sendall(b'isto nao e json\n')
        assert json.loads(f.readline())["ok"] is False
        c.sendall(b'{"command":"list"}\n')       # ainda funciona depois
        for _ in range(50):
            msg = srv.proximo()
            if msg:
                break
            time.sleep(0.05)
        assert msg is not None and msg["command"] == "list"
        c.close()
    finally:
        srv.fechar()


def test_porta_ocupada_nao_derruba():
    porta = _porta_livre()
    a = ServidorControle(porta=porta)
    try:
        b = abrir(porta=porta)
        assert b.proximo() is None          # objeto nulo, nao excecao
        b.fechar()
    finally:
        a.fechar()


# ---------------------------------------------------------- ponta a ponta

class _Cliente:
    """Fala com o runner como a Unity vai falar."""

    def __init__(self, porta_tel, porta_ctl):
        self.tipos = Counter()
        self.estado = {}
        self.catalogo = None
        self.info = None
        self._parar = threading.Event()
        self._tel = socket.create_connection(("127.0.0.1", porta_tel), timeout=None)
        threading.Thread(target=self._ouve, daemon=True).start()
        self._ctl = socket.create_connection(("127.0.0.1", porta_ctl), timeout=60)
        self._cf = self._ctl.makefile("r", encoding="utf-8")

    def _ouve(self):
        f = self._tel.makefile("r", encoding="utf-8")
        for linha in f:
            if self._parar.is_set():
                return
            m = protocol.decode(linha)
            self.tipos[m["type"]] += 1
            if m["type"] == "run_state":
                self.estado = m
            elif m["type"] == "experiment_list":
                self.catalogo = m["experiments"]
            elif m["type"] == "experiment_info":
                self.info = m

    def cmd(self, **kw):
        self._ctl.sendall((json.dumps(kw) + "\n").encode("utf-8"))
        return json.loads(self._cf.readline())

    def espera_estado(self, alvo, limite=60.0):
        fim = time.time() + limite
        while time.time() < fim:
            if self.estado.get("state") == alvo:
                return self.estado
            time.sleep(0.1)
        raise AssertionError(f"estado {alvo!r} nao chegou; ultimo: {self.estado}")

    def espera_montagem(self, limite=90.0):
        """
        Espera `loading` e depois `running`.

        Esperar so por `running` nao serve: logo depois de um start, o estado que
        chega pode ainda ser o da corrida ANTERIOR, que estava rodando. O
        `loading` e o que separa as duas.
        """
        self.espera_estado("loading", limite=20)
        return self.espera_estado("running", limite=limite)

    def fechar(self):
        self._parar.set()
        for s in (self._tel, self._ctl):
            try:
                s.close()
            except OSError:
                pass


def _com_runner(fn):
    porta_tel, porta_ctl = _porta_livre(), _porta_livre()
    proc = subprocess.Popen(
        [sys.executable, str(RAIZ / "sim" / "lab_runner.py"),
         "--porta", str(porta_tel), "--porta-controle", str(porta_ctl)],
        cwd=str(RAIZ), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    cli = None
    try:
        for _ in range(100):                 # espera as portas subirem
            try:
                cli = _Cliente(porta_tel, porta_ctl)
                break
            except OSError:
                time.sleep(0.2)
        assert cli is not None, "runner nao abriu as portas"
        fn(cli)
    finally:
        if cli is not None:
            try:
                cli.cmd(command="quit")
            except OSError:
                pass
            cli.fechar()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("runner nao encerrou depois de `quit`")


def test_ponta_a_ponta_ciclo_completo():
    """start / pause / resume / reset com semente, do jeito que a interface faz."""
    if PULAR_LENTOS:
        return

    def corpo(cli):
        assert {e["id"] for e in cli.catalogo} >= ESPERADOS

        assert cli.cmd(command="select", experiment_id="looming_escape", seed=7)["ok"]
        assert cli.cmd(command="start")["ok"]
        st = cli.espera_montagem()
        assert st["experiment_id"] == "looming_escape" and st["seed"] == 7
        assert cli.tipos["experiment_info"] >= 1
        assert cli.tipos["scene_info"] >= 1

        time.sleep(3)
        assert cli.tipos["frame"] > 0, "rodando e sem publicar quadro"
        assert cli.tipos["neural_activity"] > 0

        assert cli.cmd(command="pause")["ok"]
        cli.espera_estado("paused", limite=10)
        n = cli.tipos["frame"]
        time.sleep(2)
        assert cli.tipos["frame"] == n, "pausado e ainda avancando a fisica"

        assert cli.cmd(command="resume")["ok"]
        cli.espera_estado("running", limite=10)
        time.sleep(2)
        assert cli.tipos["frame"] > n

        assert cli.cmd(command="reset", seed=42)["ok"]
        st = cli.espera_montagem()
        assert st["seed"] == 42, f"reset nao trocou a semente: {st}"

        assert cli.cmd(command="stop")["ok"]
        cli.espera_estado("idle", limite=20)

    _com_runner(corpo)


def test_ponta_a_ponta_troca_de_experimento():
    """
    Trocar de experimento tem que trocar o circuito publicado.

    O que se quer evitar e o runner reaproveitar a rede montada e a interface
    seguir mostrando as camadas do anterior com dados do novo.
    """
    if PULAR_LENTOS:
        return

    vistos = {}

    def corpo(cli):
        for eid, camadas in [("looming_escape", {"LC4/LPLC2", "DNp01", "TTMn"}),
                             ("optomotor_turning", {"T4/T5", "HS", "DNa02", "motor"})]:
            cli.info = None
            assert cli.cmd(command="start", experiment_id=eid, seed=0)["ok"]
            st = cli.espera_montagem()
            assert st["experiment_id"] == eid

            fim = time.time() + 20
            while cli.info is None and time.time() < fim:
                time.sleep(0.1)
            assert cli.info is not None, f"{eid} rodou sem publicar experiment_info"
            assert cli.info["experiment_id"] == eid
            nomes = {c["name"] for c in cli.info["circuits"]}
            assert nomes == camadas, f"{eid}: camadas {nomes}, esperado {camadas}"
            # os bodyIds sao DATA: se vierem vazios a interface nao tem o que acender
            assert all(c["body_ids"] for c in cli.info["circuits"])
            vistos[eid] = nomes

    _com_runner(corpo)
    assert set(vistos) == {"looming_escape", "optomotor_turning"}


def test_spikes_batem_com_os_bodyids_declarados():
    """
    O vetor de spikes de cada camada tem que ter o tamanho dos bodyIds dela.

    Este nao e um teste de formato: a interface casa spikes com bodyIds POR
    POSICAO. No Giant Fiber a rede tem 1271 neuronios pre-sinapticos e o circuito
    declarado tem so os 311 de looming; publicar o vetor inteiro acenderia o
    neuronio errado no cerebro 3D, e ninguem veria erro nenhum na tela.
    """
    if PULAR_LENTOS:
        return

    for eid in ("looming_escape", "optomotor_turning"):
        exp = experiments.criar(eid, seed=0)
        exp.setup()
        declarado = {c["name"]: len(c["body_ids"])
                     for c in exp.telemetry_metadata()["circuits"]}
        publicado = {}
        for _ in range(2000):
            s = exp.step()
            if "layers" in s:
                publicado = {c.get("name"): len(c["spikes"]) for c in s["layers"]}
        assert publicado, f"{eid} nao publicou camada nenhuma"
        assert publicado == declarado, f"{eid}: {publicado} != {declarado}"
        exp.sim.close()


def test_entrada_nao_finge_ter_membrana():
    """
    A populacao de entrada e Poisson: tem spike, nao tem potencial de membrana.

    Mandar v_mV zerado pra ela deixaria a mensagem uniforme e a interface
    desenharia uma membrana que nao existe.
    """
    if PULAR_LENTOS:
        return

    exp = experiments.criar("looming_escape", seed=0)
    exp.setup()
    camadas = []
    for _ in range(300):
        s = exp.step()
        if "layers" in s:
            camadas = s["layers"]
    assert camadas, "nao publicou camada nenhuma"
    entrada = camadas[0]
    assert entrada["name"] == "LC4/LPLC2"
    assert "spikes" in entrada
    assert "v_mV" not in entrada and "refratario" not in entrada
    # as camadas LIF continuam trazendo o estado do modelo
    assert "v_mV" in camadas[1]
    exp.sim.close()


def _todos():
    return [(n, o) for n, o in sorted(globals().items())
            if n.startswith("test_") and callable(o)]


if __name__ == "__main__":
    falhas = 0
    for nome, fn in _todos():
        t0 = time.time()
        try:
            fn()
            print(f"  ok    {nome}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(_todos()) - falhas}/{len(_todos())} passaram")
    sys.exit(1 if falhas else 0)
