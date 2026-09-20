"""
Canal de controle: a interface escolhe QUAL experimento roda e quando.

Socket separado do de telemetria, de proposito. O de telemetria continua sendo
mao unica -- da pra abrir server.py e conferir que nada la le do socket. Tudo que
vem de fora entra por aqui, e o que cabe aqui e curto:

    list                            devolve o catalogo
    select  {experiment_id, seed}   escolhe o proximo a montar
    start                           monta e comeca
    pause / resume                  congela e descongela o laco
    reset   {seed}                  volta ao inicio (mesma semente ou outra)
    stop                            encerra a corrida, volta pro estado parado
    quit                            encerra o processo

## O que este canal NAO pode fazer

Nao existe comando pra mexer em peso sinaptico, taxa de disparo, limiar, drive
motor, posicao da mosca ou parametro de circuito. Isso nao e esquecimento: o
conectoma e a autoridade do comportamento e a fisica e do MuJoCo. Uma interface
que pudesse empurrar drive faria a mosca virar sem que o circuito tivesse
decidido nada, e ai o que aparece na tela nao seria mais resultado.

Se algum dia for preciso variar um parametro pela interface, que seja um campo
declarado do experimento, marcado ASSUMPTION, e registrado no `metadata` da
gravacao -- nao um comando generico de escrita.

Transporte igual ao da telemetria: TCP, uma linha JSON por mensagem. Cada comando
recebe um ack na mesma conexao ({"type":"ack","ok":bool,...}); o estado de
verdade vai por `run_state` na telemetria, que todo mundo ve.
"""
from __future__ import annotations

import json
import queue
import socket
import threading
from typing import Any

COMANDOS = ("list", "select", "start", "pause", "resume", "reset", "stop", "quit")


class ServidorControle:
    """Recebe comandos e os enfileira. Quem consome e o laco da simulacao."""

    # quem roda sem interface precisa saber que nao adianta esperar comando
    ativo = True

    def __init__(self, host: str = "127.0.0.1", porta: int = 8766):
        self.host = host
        self.porta = porta
        self._fila: queue.Queue = queue.Queue()
        self._parar = threading.Event()
        self._clientes: list[socket.socket] = []
        self._lock = threading.Lock()

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # mesma razao do servidor de telemetria: no Windows SO_REUSEADDR deixa
        # dois servidores pegarem a mesma porta e um rouba comandos do outro
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        self._srv.bind((host, porta))
        self._srv.listen(4)
        self._srv.settimeout(0.5)

        threading.Thread(target=self._aceitar, daemon=True).start()
        print(f"[controle] ouvindo em {host}:{porta}")

    # ------------------------------------------------------------ interface

    def proximo(self) -> dict[str, Any] | None:
        """Comando pendente, ou None. Nunca bloqueia o laco de simulacao."""
        try:
            return self._fila.get_nowait()
        except queue.Empty:
            return None

    def responder(self, sock, ok: bool, **campos) -> None:
        if sock is None:
            return
        try:
            sock.sendall((json.dumps({"type": "ack", "ok": ok, **campos},
                                     separators=(",", ":")) + "\n").encode("utf-8"))
        except OSError:
            pass

    def fechar(self) -> None:
        self._parar.set()
        try:
            self._srv.close()
        except OSError:
            pass
        with self._lock:
            for c in self._clientes:
                try:
                    c.close()
                except OSError:
                    pass
            self._clientes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()

    # -------------------------------------------------------------- threads

    def _aceitar(self):
        while not self._parar.is_set():
            try:
                sock, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clientes.append(sock)
            threading.Thread(target=self._ler, args=(sock, addr), daemon=True).start()
            print(f"[controle] cliente conectado: {addr[0]}:{addr[1]}")

    def _ler(self, sock, addr):
        buf = b""
        while not self._parar.is_set():
            try:
                pedaco = sock.recv(4096)
            except OSError:
                break
            if not pedaco:
                break
            buf += pedaco
            while b"\n" in buf:
                linha, buf = buf.split(b"\n", 1)
                if not linha.strip():
                    continue
                try:
                    msg = json.loads(linha.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as e:
                    self.responder(sock, False, error=f"json invalido: {e}")
                    continue
                cmd = msg.get("command")
                if cmd not in COMANDOS:
                    # Recusa explicita em vez de ignorar: comando desconhecido
                    # quase sempre e a interface pedindo algo que este canal nao
                    # deve fazer, e isso tem que aparecer.
                    self.responder(sock, False,
                                   error=f"comando desconhecido: {cmd!r}",
                                   allowed=list(COMANDOS))
                    continue
                msg["_sock"] = sock
                self._fila.put(msg)

        with self._lock:
            if sock in self._clientes:
                self._clientes.remove(sock)
        try:
            sock.close()
        except OSError:
            pass
        print(f"[controle] cliente saiu: {addr[0]}:{addr[1]}")


class SemControle:
    """Objeto nulo, pra rodar sem interface."""

    ativo = False

    def proximo(self):
        return None

    def responder(self, sock, ok: bool, **campos):
        pass

    def fechar(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def abrir(porta: int = 8766, ativo: bool = True, host: str = "127.0.0.1"):
    """Porta ocupada avisa e segue sem controle, em vez de derrubar a corrida."""
    if not ativo:
        return SemControle()
    try:
        return ServidorControle(host=host, porta=porta)
    except OSError as e:
        print(f"[controle] nao foi possivel abrir {host}:{porta} ({e}); "
              "seguindo sem canal de controle")
        return SemControle()
