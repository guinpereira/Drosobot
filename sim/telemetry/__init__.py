"""
Telemetria do Drosobot: a simulacao publica seu estado, o visualizador observa.

    from sim.telemetry import protocol, abrir

    tel = abrir(porta=8765, ativo=True)
    tel.enviar(protocol.experiment_info(...))
    tel.enviar(protocol.frame(...))
    tel.fechar()

A camada inteira e de MAO UNICA: nada volta pra simulacao. MuJoCo continua sendo
a autoridade da fisica e o circuito continua sendo a autoridade do comportamento;
a Unity so desenha o que chega.

Desligada (`ativo=False`) devolve um objeto nulo cujos metodos sao `pass`.
"""
from .protocol import ASSUMPTION, DATA, MODEL, PROTOCOL, VERSION  # noqa: F401
from .recorder import Gravador, ler, listar, reproduzir  # noqa: F401
from .server import SemTelemetria, ServidorTelemetria, abrir  # noqa: F401

__all__ = [
    "PROTOCOL", "VERSION", "DATA", "MODEL", "ASSUMPTION",
    "abrir", "ServidorTelemetria", "SemTelemetria",
    "Gravador", "ler", "listar", "reproduzir",
]
