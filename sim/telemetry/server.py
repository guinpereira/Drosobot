"""
Servidor de telemetria: publica o estado da simulacao sem nunca segura-la.

Duas garantias que valem mais que qualquer recurso aqui:

1. **A simulacao nunca bloqueia.** `enviar()` so encosta numa fila limitada e
   volta. Uma thread separada esvazia a fila no socket. Se o visualizador estiver
   lento, ou a rede engasgar, ou ninguem estiver conectado, a fila enche e as
   mensagens MAIS ANTIGAS sao descartadas. Perder quadro de visualizacao e
   aceitavel; atrasar a fisica nao e.

2. **Desligada custa quase nada.** `SemTelemetria` tem a mesma interface e todos
   os metodos sao `pass`. O codigo do experimento nao precisa de `if` em lugar
   nenhum, e o custo vira uma chamada de funcao vazia.

Uso tipico:

    tel = abrir(porta=8765, ativo=args.telemetry)
    tel.enviar(protocol.experiment_info(...))
    ...
    tel.enviar(protocol.frame(...))
    tel.fechar()
"""
from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any

from . import protocol


class SemTelemetria:
    """Objeto nulo. Mesma interface, custo praticamente zero."""

    ativo = False

    def enviar(self, msg: dict[str, Any]) -> None:
        pass

    def enviar_se_conectado(self, construtor, *args, **kwargs) -> None:
        pass

    @property
    def conectado(self) -> bool:
        return False

    def fechar(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()


class ServidorTelemetria:
    """TCP, uma mensagem JSON por linha, um ou mais visualizadores conectados."""

    ativo = True

    def __init__(self, host: str = "127.0.0.1", porta: int = 8765,
                 fila_max: int = 256, fonte: str = "drosobot-sim"):
        self.host = host
        self.porta = porta
        self.fonte = fonte
        self._fila: queue.Queue = queue.Queue(maxsize=fila_max)
        self._clientes: list[socket.socket] = []
        # Mensagens de ABERTURA guardadas pra reenviar a quem chegar depois.
        # A Unity quase sempre conecta com a simulacao ja rodando, e sem o
        # experiment_info ela nao sabe quais bodyIds acender nem o que e DATA e o
        # que e ASSUMPTION. Sem isso, conectar tarde = tela vazia.
        self._fixas: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._descartadas = 0

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # No Windows SO_REUSEADDR deixa DOIS servidores pegarem a mesma porta, e
        # ai um rouba conexoes do outro em silencio -- o oposto do que o nome
        # sugere pra quem vem de Linux. SO_EXCLUSIVEADDRUSE e a opcao que de fato
        # recusa a segunda ligacao, que e o que queremos: duas simulacoes no ar
        # tem que dar erro claro, nao telemetria misturada.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, porta))
        self._srv.listen(4)
        self._srv.settimeout(0.5)

        self._t_accept = threading.Thread(target=self._aceitar, daemon=True)
        self._t_write = threading.Thread(target=self._escrever, daemon=True)
        self._t_accept.start()
        self._t_write.start()
        print(f"[telemetria] ouvindo em {host}:{porta}")

    # ------------------------------------------------------------- interface

    @property
    def conectado(self) -> bool:
        with self._lock:
            return len(self._clientes) > 0

    @property
    def descartadas(self) -> int:
        return self._descartadas

    # Tipos que descrevem a corrida: todo cliente novo precisa deles, mesmo
    # chegando no meio. experiment_list e run_state entram aqui porque sem eles o
    # seletor abre vazio e sem saber se ha algo rodando -- so a ULTIMA de cada
    # tipo e guardada, entao run_state mudar nao acumula nada.
    FIXAS = ("experiment_list", "experiment_info", "scene_info", "run_state")

    def enviar(self, msg: dict[str, Any]) -> None:
        """Nunca bloqueia. Fila cheia: joga fora a mais antiga e poe esta."""
        if msg.get("type") in self.FIXAS:
            with self._lock:
                self._fixas[msg["type"]] = msg
        try:
            self._fila.put_nowait(msg)
        except queue.Full:
            try:
                self._fila.get_nowait()
                self._descartadas += 1
                self._fila.put_nowait(msg)
            except (queue.Empty, queue.Full):
                self._descartadas += 1

    def enviar_se_conectado(self, construtor, *args, **kwargs) -> None:
        """
        So monta a mensagem se houver alguem ouvindo.

        Serve pras mensagens caras: a retina sao 1442 floats por quadro, e montar
        essa lista sem ninguem do outro lado e trabalho jogado fora.
        """
        if self.conectado:
            self.enviar(construtor(*args, **kwargs))

    def fechar(self) -> None:
        if self._parar.is_set():
            return
        try:
            self.enviar(protocol.bye())
            time.sleep(0.15)      # deixa a thread de escrita drenar o que da
        except Exception:
            pass
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
        if self._descartadas:
            print(f"[telemetria] {self._descartadas} mensagens descartadas "
                  f"(visualizador mais lento que a simulacao -- esperado)")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()

    # ---------------------------------------------------------------- threads

    def _aceitar(self):
        while not self._parar.is_set():
            try:
                sock, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sock.sendall(protocol.encode(protocol.hello(self.fonte)))
                # poe o recem-chegado em dia: sem isso, quem conecta com a
                # simulacao ja rodando nunca recebe experiment_info/scene_info
                with self._lock:
                    atraso = [self._fixas[t] for t in self.FIXAS if t in self._fixas]
                for m in atraso:
                    sock.sendall(protocol.encode(m))
            except OSError:
                sock.close()
                continue
            with self._lock:
                self._clientes.append(sock)
            print(f"[telemetria] visualizador conectado: {addr[0]}:{addr[1]}")

    def _escrever(self):
        while not self._parar.is_set():
            try:
                msg = self._fila.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                if not self._clientes:
                    continue
                dados = protocol.encode(msg)
                mortos = []
                for c in self._clientes:
                    try:
                        c.sendall(dados)
                    except OSError:
                        mortos.append(c)
                for c in mortos:
                    self._clientes.remove(c)
                    try:
                        c.close()
                    except OSError:
                        pass
                    print("[telemetria] visualizador desconectou")


def abrir(porta: int = 8765, ativo: bool = True, host: str = "127.0.0.1",
          fonte: str = "drosobot-sim"):
    """
    Devolve um servidor de verdade ou o objeto nulo.

    Se a porta estiver ocupada, avisa e devolve o nulo em vez de derrubar a
    corrida -- perder visualizacao nao pode custar um experimento.
    """
    if not ativo:
        return SemTelemetria()
    try:
        return ServidorTelemetria(host=host, porta=porta, fonte=fonte)
    except OSError as e:
        print(f"[telemetria] nao foi possivel abrir {host}:{porta} ({e}); seguindo sem telemetria")
        return SemTelemetria()
