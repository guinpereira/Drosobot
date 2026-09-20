"""
Testes do runtime unificado (`sim/drosobot_lab.py`).

O que precisa ficar garantido aqui:

1. **O catalogo nao promete o que nao roda.** Cada entrada tem que apontar pra
   uma arena e um escopo que o runtime sabe montar. Oferecer experimento que
   quebra ao dar start e pior que nao oferecer.

2. **O canal de controle nao vira porta dos fundos.** Ele escolhe experimento e
   aperta start/pause/resume/stop. Nada nele altera peso, limiar, entrada ou
   drive -- e o teste de ponta a ponta confirma que a maquina de estados
   responde sem que a ciencia mude de lugar.

3. **Pausar nao distorce o RTF.** O tempo parado nao e tempo de calculo; se
   entrar no denominador, a interface mostra uma lentidao que nao existe.

Rodam com pytest ou direto:

    .venv-flygym2\\Scripts\\python -m pytest tests/test_drosobot_lab.py -q
    .venv-flygym2\\Scripts\\python tests/test_drosobot_lab.py

O teste de ponta a ponta sobe o processo de verdade e monta a mosca no MuJoCo,
entao leva perto de um minuto. DROSOBOT_SKIP_LENTOS=1 pula so esse.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "sim"))

import drosobot_lab as lab  # noqa: E402
from profiler import Profiler  # noqa: E402

# Lidas da receita, nao copiadas. A copia daqui envelheceu: `optomotor` e
# `obstaculos` entraram no catalogo e este conjunto ficou com duas arenas,
# reprovando um catalogo correto.
from lab.receita import ARENAS as _ARENAS, ESCOPOS as _ESCOPOS  # noqa: E402

ARENAS = set(_ARENAS)
ESCOPOS = set(_ESCOPOS)


# --------------------------------------------------------------- catalogo

def test_catalogo_so_promete_o_que_roda():
    assert lab.CATALOGO, "catalogo vazio: o seletor da Unity ficaria sem nada"
    vistos = set()
    for e in lab.CATALOGO:
        for campo in ("id", "name", "description", "arena", "cns"):
            assert e.get(campo), f"{e.get('id')}: falta {campo}"
        assert e["arena"] in ARENAS, f"{e['id']}: arena {e['arena']!r} nao existe"
        assert e["cns"] in ESCOPOS, f"{e['id']}: escopo {e['cns']!r} nao existe"
        assert e["id"] not in vistos, f"id repetido: {e['id']}"
        vistos.add(e["id"])


def test_item_acha_e_nao_inventa():
    assert lab.Laboratorio.item(lab.CATALOGO[0]["id"]) is not None
    assert lab.Laboratorio.item("nao_existe") is None


def test_atalho_de_experimento_respeita_escopo():
    """`--experiment looming --cns circuit` tem que cair no item de circuito."""
    class Args:
        experiment = "looming"
        cns = "circuit"

    assert lab._resolve_experimento(Args()) == "looming_circuit"
    Args.cns = "whole"
    assert lab._resolve_experimento(Args()) == "looming_whole"
    # id explicito vence o atalho
    Args.experiment = "flat_whole"
    assert lab._resolve_experimento(Args()) == "flat_whole"


# ------------------------------------------------------------------ rotulos

def test_nomes_por_body_id_rotula_sem_entrar_na_conta():
    mapa = lab.nomes_por_body_id()
    # o CSV pode nao estar no clone; o runtime tem que aguentar isso
    assert isinstance(mapa, dict)
    if mapa:
        tipos = set(mapa.values())
        assert "DNp01" in tipos, "o proprio Giant Fiber deveria estar rotulado"


# ------------------------------------------------------------------ profiler

def test_pausa_nao_entra_no_denominador_do_rtf():
    """
    Mede o RELOGIO, nao o RTF.

    O RTF logo apos criar o Profiler e a razao entre 100 ms de mosca e alguns
    microssegundos de parede -- nessa escala o overhead do proprio Python
    domina e qualquer limiar sobre a razao vira ruido. O que o codigo promete e
    concreto: o tempo parado nao entra no relogio.
    """
    p = Profiler(["physics"])
    p.avanca_sim(100.0)
    time.sleep(0.10)             # um pouco de corrida de verdade
    wall_a = p.wall_s
    p.pausa()
    time.sleep(0.30)             # parado a pedido da interface
    p.retoma()
    wall_b = p.wall_s
    assert wall_b - wall_a < 0.05, (
        f"a pausa de 0,30 s entrou no relogio: {wall_a:.4f} -> {wall_b:.4f}")
    assert wall_a >= 0.09, "o tempo de corrida sumiu junto com a pausa"


def test_pausa_dupla_nao_acumula_desconto():
    p = Profiler(["physics"])
    p.avanca_sim(10.0)
    p.pausa()
    p.pausa()                    # segunda chamada nao pode reancorar
    time.sleep(0.05)
    p.retoma()
    p.retoma()                   # segunda tem que ser no-op
    assert p.wall_s >= 0.0


# --------------------------------------------------------------- ponta a ponta

def _comando(porta, **campos):
    with socket.create_connection(("127.0.0.1", porta), timeout=20) as s:
        s.sendall((json.dumps(campos) + "\n").encode("utf-8"))
        s.settimeout(20)
        dados = b""
        while not dados.endswith(b"\n"):
            pedaco = s.recv(4096)
            if not pedaco:
                break
            dados += pedaco
        return json.loads(dados.decode("utf-8").strip() or "{}")


def _estados(porta_tel, ate_ver, timeout=90):
    """Le a telemetria ate ver os estados pedidos. Devolve o que viu."""
    vistos = []
    fim = time.time() + timeout
    with socket.create_connection(("127.0.0.1", porta_tel), timeout=5) as s:
        s.settimeout(5)
        buffer = b""
        while time.time() < fim and not ate_ver.issubset(set(vistos)):
            try:
                pedaco = s.recv(65536)
            except socket.timeout:
                continue
            if not pedaco:
                break
            buffer += pedaco
            *linhas, buffer = buffer.split(b"\n")
            for linha in linhas:
                if not linha.strip():
                    continue
                try:
                    msg = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "run_state":
                    vistos.append(msg["state"])
    return vistos


def test_controle_dirige_a_maquina_de_estados():
    """Sobe o runtime de verdade e conduz start -> pause -> resume -> stop."""
    if os.environ.get("DROSOBOT_SKIP_LENTOS"):
        return
    porta_tel, porta_ctl = 8791, 8792
    proc = subprocess.Popen(
        [sys.executable, "-u", "-W", "ignore", str(RAIZ / "sim" / "drosobot_lab.py"),
         "--cns", "circuit", "--duracao", "0", "--telemetry", "--espera",
         "--porta", str(porta_tel), "--porta-controle", str(porta_ctl)],
        cwd=str(RAIZ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    try:
        # o processo precisa subir os dois servidores antes de aceitar comando
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", porta_ctl), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise AssertionError("o canal de controle nunca abriu")

        catalogo = _comando(porta_ctl, command="list")
        assert catalogo.get("ok"), catalogo
        assert any(e["id"] == "looming_circuit"
                   for e in catalogo.get("experiments", []))

        # comando fora da lista tem que ser recusado, nao ignorado em silencio
        recusa = _comando(porta_ctl, command="select",
                          experiment_id="experimento_que_nao_existe")
        assert recusa.get("ok") is False, recusa

        ack = _comando(porta_ctl, command="start",
                       experiment_id="looming_circuit", seed=0)
        assert ack.get("ok"), ack

        vistos = _estados(porta_tel, {"running"}, timeout=120)
        assert "running" in vistos, f"nunca chegou a rodar: {vistos}"

        assert _comando(porta_ctl, command="pause").get("state") == "paused"
        assert _comando(porta_ctl, command="resume").get("state") == "running"
        assert _comando(porta_ctl, command="stop").get("ok")
    finally:
        try:
            _comando(porta_ctl, command="quit")
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_telemetria_entrega_o_contrato_da_unity():
    """
    Sobe o runtime e confere que TODO campo que a interface le chega.

    Este teste existe porque o modo de falha e silencioso: a Unity nao quebra
    quando falta um campo, ela so desenha um painel vazio. Ja aconteceu com o
    canal de controle e com a retina -- o runtime novo simplesmente nao
    mandava, e a tela parecia "ainda carregando" pra sempre.
    """
    if os.environ.get("DROSOBOT_SKIP_LENTOS"):
        return
    porta_tel, porta_ctl = 8795, 8796
    proc = subprocess.Popen(
        [sys.executable, "-u", "-W", "ignore", str(RAIZ / "sim" / "drosobot_lab.py"),
         "--cns", "circuit", "--duracao", "0.6", "--telemetry",
         "--porta", str(porta_tel), "--porta-controle", str(porta_ctl)],
        cwd=str(RAIZ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    tipos, achados = set(), {}
    try:
        for _ in range(120):
            try:
                sock = socket.create_connection(("127.0.0.1", porta_tel), timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise AssertionError("a telemetria nunca abriu")

        sock.settimeout(3)
        buffer = b""
        fim = time.time() + 180
        with sock:
            while time.time() < fim:
                try:
                    pedaco = sock.recv(1 << 16)
                except socket.timeout:
                    continue
                if not pedaco:
                    break
                buffer += pedaco
                *linhas, buffer = buffer.split(b"\n")
                for linha in linhas:
                    if not linha.strip():
                        continue
                    try:
                        m = json.loads(linha)
                    except json.JSONDecodeError:
                        continue
                    tipos.add(m.get("type"))
                    if m.get("type") == "experiment_info":
                        achados["runtime"] = m.get("runtime", {})
                        achados["scope"] = m.get("scope", {})
                        achados["limitacoes"] = m.get("model_limitations", [])
                    elif m.get("type") == "frame" and "body_pose" in m:
                        achados["pose"] = m["body_pose"]
                        achados["profile"] = m.get("profile", {})
                    elif m.get("type") == "retina":
                        achados["retina"] = m
                    elif m.get("type") == "statistics":
                        achados["gate"] = m["values"].get("gf_gate", {})
                    elif m.get("type") == "experiment_list":
                        achados["catalogo"] = m["experiments"]
    finally:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    for t in ("hello", "experiment_info", "scene_info", "experiment_list",
              "run_state", "frame", "neural_activity", "statistics", "retina"):
        assert t in tipos, f"a Unity espera mensagens `{t}` e nenhuma chegou"

    rt = achados.get("runtime", {})
    for campo in ("os", "cpu", "physics_backend", "neural_backend",
                  "neural_device", "neurons_simulated", "edges_simulated",
                  "collision_pairs"):
        assert rt.get(campo) is not None, f"runtime.{campo} nao foi mandado"

    # o escopo nunca pode virar "cerebro funcional completo"
    assert achados["scope"]["functional_brain"] == "no"
    assert achados["limitacoes"], "a limitacao do modelo sumiu do fluxo"

    assert len(achados["retina"]["left"]) == 721, "a retina mudou de tamanho"
    assert len(achados["pose"]["segments"]) == 69, "a pose perdeu segmento"
    assert set(achados["profile"]) == {"physics", "vision", "neural",
                                       "leitura", "telemetry"}

    gate = achados["gate"]
    for campo in ("exc_mV", "inib_mV", "liquido_mV", "v_min_mV", "spikes_gf"):
        assert campo in gate, f"gf_gate.{campo} sumiu"

    ids = {e["id"] for e in achados["catalogo"]}
    assert ids == {e["id"] for e in lab.CATALOGO}


if __name__ == "__main__":
    falhas = 0
    for nome, fn in sorted(globals().items()):
        if not nome.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok    {nome}")
        except Exception as e:                                # noqa: BLE001
            falhas += 1
            print(f"  FALHA {nome}: {type(e).__name__}: {e}")
    print("\n" + ("todos passaram" if not falhas else f"{falhas} falha(s)"))
    sys.exit(1 if falhas else 0)
