"""
Gravar e reproduzir corridas.

Uma corrida gravada permite abrir de novo o que aconteceu sem refazer a
simulacao -- o que importa aqui porque um segundo de mosca custa ~26 s de
relogio. Serve pra comparar experimentos, depurar, gravar video e mostrar um
resultado sem esperar meia hora na frente de todo mundo.

Formato em disco:

    runs/2026-09-19_143002_giant_fiber_looming/
        metadata.json    experimento, seed, parametros, versao do protocolo
        telemetry.jsonl  uma mensagem por linha, na ordem em que saiu
        summary.json     agregados no fim (escrito no fechamento)

`runs/` nao vai pro git.

O gravador e um DECORADOR do servidor: ele repassa tudo e guarda uma copia. Com o
servidor nulo por dentro, grava sem visualizador nenhum conectado.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from . import protocol

RAIZ_RUNS = Path(__file__).resolve().parents[2] / "runs"


class Gravador:
    """Envolve um sink de telemetria e guarda tudo que passa."""

    def __init__(self, sink, experiment_id: str, metadata: dict[str, Any] | None = None,
                 raiz: Path | None = None, fechar_sink: bool = True):
        self._sink = sink
        # Nos scripts de uma corrida so, fechar a gravacao e fechar tudo. O
        # lab_runner reusa o MESMO servidor de telemetria entre experimentos --
        # la, fechar o sink junto derrubaria a conexao da Unity a cada troca.
        self._fechar_sink = fechar_sink
        carimbo = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.pasta = (raiz or RAIZ_RUNS) / f"{carimbo}_{experiment_id}"
        self.pasta.mkdir(parents=True, exist_ok=True)
        self._f = (self.pasta / "telemetry.jsonl").open("w", encoding="utf-8")
        self._n = 0
        self._t0 = time.time()

        meta = {
            "protocol": protocol.PROTOCOL,
            "version": protocol.VERSION,
            "experiment_id": experiment_id,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta.update(metadata or {})
        (self.pasta / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=protocol._json_default), encoding="utf-8")
        print(f"[gravacao] {self.pasta}")

    @property
    def ativo(self) -> bool:
        return True

    @property
    def conectado(self) -> bool:
        # gravando conta como "tem quem ouca": as mensagens caras devem ser montadas
        return True

    def enviar(self, msg: dict[str, Any]) -> None:
        self._sink.enviar(msg)
        self._f.write(protocol.encode(msg).decode("utf-8"))
        self._n += 1

    def enviar_se_conectado(self, construtor, *args, **kwargs) -> None:
        self.enviar(construtor(*args, **kwargs))

    def escrever_resumo(self, resumo: dict[str, Any]) -> None:
        (self.pasta / "summary.json").write_text(
            json.dumps(resumo, indent=2, default=protocol._json_default), encoding="utf-8")

    def fechar(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
        if self._fechar_sink:
            self._sink.fechar()
        print(f"[gravacao] {self._n} mensagens em {time.time() - self._t0:.1f} s -> {self.pasta}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()


def ler(pasta: str | Path) -> Iterator[dict[str, Any]]:
    """Le uma corrida gravada, mensagem por mensagem."""
    caminho = Path(pasta)
    arq = caminho if caminho.suffix == ".jsonl" else caminho / "telemetry.jsonl"
    with arq.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                yield protocol.decode(linha)


def listar(raiz: Path | None = None) -> list[Path]:
    """Corridas gravadas, mais recente primeiro."""
    r = raiz or RAIZ_RUNS
    if not r.exists():
        return []
    return sorted((p for p in r.iterdir() if (p / "telemetry.jsonl").exists()),
                  reverse=True)


def reproduzir(pasta: str | Path, porta: int = 8765, velocidade: float = 1.0,
               host: str = "127.0.0.1") -> None:
    """
    Serve uma corrida gravada na mesma porta que a simulacao usaria.

    Pro visualizador nao ha diferenca: chegam as mesmas mensagens, na mesma
    ordem, com o mesmo espacamento (multiplicado por `velocidade`). Ele nao
    precisa saber se veio de simulacao ao vivo ou de arquivo.
    """
    from .server import ServidorTelemetria

    msgs = list(ler(pasta))
    if not msgs:
        print(f"[replay] nada em {pasta}")
        return

    srv = ServidorTelemetria(porta=porta, host=host, fonte="drosobot-replay")
    print(f"[replay] {len(msgs)} mensagens de {pasta}")
    print("[replay] esperando visualizador conectar...")
    espera = time.time()
    while not srv.conectado and time.time() - espera < 60:
        time.sleep(0.2)
    if not srv.conectado:
        print("[replay] ninguem conectou em 60 s; reproduzindo assim mesmo")

    # respeita o espacamento original usando o sim_time das mensagens que o tem
    t_anterior = None
    t0 = time.time()
    for msg in msgs:
        t_sim = msg.get("sim_time")
        if t_sim is not None and velocidade > 0:
            if t_anterior is not None:
                atraso = (t_sim - t_anterior) / velocidade
                if atraso > 0:
                    alvo = t0 + atraso
                    while time.time() < alvo:
                        time.sleep(min(0.005, alvo - time.time()))
            t_anterior = t_sim
            t0 = time.time()
        srv.enviar(msg)
    time.sleep(0.5)
    srv.fechar()
    print("[replay] fim")
