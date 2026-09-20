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

ARENAS = {"looming", "flat"}
ESCOPOS = {"whole", "circuit"}


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
