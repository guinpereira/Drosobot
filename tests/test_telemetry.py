"""
Testes da camada de telemetria.

Rodam com pytest ou direto:

    .venv\\Scripts\\python -m pytest tests/test_telemetry.py -q
    .venv\\Scripts\\python tests/test_telemetry.py
"""
import json
import socket
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

from telemetry import protocol  # noqa: E402
from telemetry.recorder import Gravador, ler, listar  # noqa: E402
from telemetry.server import SemTelemetria, ServidorTelemetria, abrir  # noqa: E402


# ------------------------------------------------------------------ protocolo

def test_envelope_e_versao():
    msg = protocol.frame(step=1, sim_time=0.5, wall_time=123.0,
                         real_time_factor=0.03, position=[1, 2, 3])
    assert msg["protocol"] == protocol.PROTOCOL
    assert msg["version"] == protocol.VERSION
    assert msg["type"] == "frame"


def test_ida_e_volta():
    original = protocol.event(1.25, "escape_triggered", {"distance_mm": 4.2})
    voltou = protocol.decode(protocol.encode(original))
    assert voltou == original


def test_uma_mensagem_por_linha():
    """O transporte depende disso: nenhuma mensagem pode conter newline cru."""
    bruto = protocol.encode(protocol.frame(
        step=0, sim_time=0, wall_time=0, real_time_factor=1,
        position=[0, 0, 0], drive=[1.0, 1.0]))
    assert bruto.endswith(b"\n")
    assert bruto.count(b"\n") == 1


def test_serializa_numpy():
    """numpy aparece em praticamente todo campo; nao pode obrigar quem chama a converter."""
    msg = protocol.neural_activity(0.1, [{
        "name": "GF",
        "spikes": np.array([0, 1, 0]),
        "v_mV": np.array([-52.0, -45.2, -51.9]),
    }])
    voltou = protocol.decode(protocol.encode(msg))
    assert voltou["layers"][0]["spikes"] == [0, 1, 0]


def test_recusa_versao_errada():
    ruim = json.dumps({"protocol": protocol.PROTOCOL, "version": 999, "type": "frame"})
    try:
        protocol.decode(ruim)
    except ValueError:
        return
    raise AssertionError("deveria ter recusado versao desconhecida")


def test_procedencia_declarada():
    """DATA/MODEL/ASSUMPTION e requisito do projeto, nao enfeite."""
    info = protocol.experiment_info(
        "x", "X", "desc",
        provenance={"body_ids": protocol.DATA,
                    "membrane_potential": protocol.MODEL,
                    "flow_gain": protocol.ASSUMPTION})
    assert set(info["provenance"].values()) == {"data", "model", "assumption"}


# --------------------------------------------------------------- objeto nulo

def test_desligada_nao_faz_nada():
    tel = abrir(ativo=False)
    assert isinstance(tel, SemTelemetria)
    assert tel.ativo is False
    assert tel.conectado is False
    tel.enviar(protocol.frame(step=0, sim_time=0, wall_time=0,
                              real_time_factor=1, position=[0, 0, 0]))
    tel.enviar_se_conectado(protocol.retina, 0.0, [0.5] * 721, [0.5] * 721)
    tel.fechar()


def test_desligada_e_barata():
    """
    O custo com telemetria desligada tem que ser desprezivel perto do passo de
    fisica (~2 ms). Aqui exigimos < 5 us por chamada, o que ja e 400x de folga.
    """
    tel = abrir(ativo=False)
    msg = protocol.frame(step=0, sim_time=0, wall_time=0,
                         real_time_factor=1, position=[0, 0, 0])
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        tel.enviar(msg)
    por_chamada = (time.perf_counter() - t0) / n
    assert por_chamada < 5e-6, f"{por_chamada*1e6:.2f} us por chamada"


# ------------------------------------------------------------------ servidor

def _porta_livre():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_servidor_entrega_mensagens():
    porta = _porta_livre()
    srv = ServidorTelemetria(porta=porta)
    try:
        cli = socket.create_connection(("127.0.0.1", porta), timeout=3)
        f = cli.makefile("r", encoding="utf-8")
        ola = protocol.decode(f.readline())
        assert ola["type"] == "hello"

        espera = time.time()
        while not srv.conectado and time.time() - espera < 3:
            time.sleep(0.05)
        srv.enviar(protocol.event(1.0, "escape_triggered"))
        recebida = protocol.decode(f.readline())
        assert recebida["kind"] == "escape_triggered"
        cli.close()
    finally:
        srv.fechar()


def test_servidor_nao_bloqueia_sem_cliente():
    """
    A garantia central: a simulacao nunca pode esperar pelo visualizador.
    Sem ninguem conectado, mandar muito mais que a fila tem que voltar rapido.
    """
    porta = _porta_livre()
    srv = ServidorTelemetria(porta=porta, fila_max=32)
    try:
        msg = protocol.frame(step=0, sim_time=0, wall_time=0,
                             real_time_factor=1, position=[0, 0, 0])
        t0 = time.perf_counter()
        for _ in range(5000):
            srv.enviar(msg)
        gasto = time.perf_counter() - t0
        assert gasto < 1.0, f"levou {gasto:.2f} s -- estaria segurando a simulacao"
        assert srv.descartadas > 0, "deveria ter descartado com a fila cheia"
    finally:
        srv.fechar()


def test_porta_ocupada_nao_derruba():
    """Perder visualizacao nao pode custar um experimento."""
    porta = _porta_livre()
    ocupa = ServidorTelemetria(porta=porta)
    try:
        segundo = abrir(porta=porta, ativo=True)
        assert isinstance(segundo, SemTelemetria)
        segundo.fechar()
    finally:
        ocupa.fechar()


# ------------------------------------------------------------ gravar/reproduzir

def test_gravacao_e_leitura():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        grav = Gravador(SemTelemetria(), "teste_exp",
                        metadata={"seed": 42, "duration": 3.0}, raiz=raiz)
        grav.enviar(protocol.experiment_info("teste_exp", "Teste", "desc"))
        for k in range(5):
            grav.enviar(protocol.frame(step=k, sim_time=k * 0.1, wall_time=k,
                                       real_time_factor=0.03, position=[k, 0, 1]))
        grav.enviar(protocol.event(0.5, "escape_triggered"))
        grav.escrever_resumo({"escapes": 1})
        grav.fechar()

        msgs = list(ler(grav.pasta))
        assert len(msgs) == 7
        assert msgs[0]["type"] == "experiment_info"
        assert [m["step"] for m in msgs if m["type"] == "frame"] == [0, 1, 2, 3, 4]
        assert msgs[-1]["kind"] == "escape_triggered"

        meta = json.loads((grav.pasta / "metadata.json").read_text(encoding="utf-8"))
        assert meta["seed"] == 42
        assert meta["version"] == protocol.VERSION
        resumo = json.loads((grav.pasta / "summary.json").read_text(encoding="utf-8"))
        assert resumo["escapes"] == 1
        assert listar(raiz) == [grav.pasta]


def test_replay_preserva_ordem_exata():
    """Reproduzir tem que devolver a mesma corrida, nao uma parecida."""
    with tempfile.TemporaryDirectory() as tmp:
        grav = Gravador(SemTelemetria(), "ordem", raiz=Path(tmp))
        enviadas = [protocol.frame(step=k, sim_time=k * 0.01, wall_time=k,
                                   real_time_factor=1, position=[k, k, k])
                    for k in range(50)]
        for m in enviadas:
            grav.enviar(m)
        grav.fechar()
        assert list(ler(grav.pasta)) == enviadas


def _todos():
    return [(n, o) for n, o in sorted(globals().items())
            if n.startswith("test_") and callable(o)]


if __name__ == "__main__":
    falhas = 0
    for nome, fn in _todos():
        try:
            fn()
            print(f"  ok    {nome}")
        except Exception as e:
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(_todos()) - falhas}/{len(_todos())} passaram")
    sys.exit(1 if falhas else 0)
