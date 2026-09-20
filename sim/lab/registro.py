"""
O que cada corrida deixa em disco.

    runs/<id>/
        metadata.json    receita, ambiente, conectoma, hash da ciência
        telemetry.jsonl  toda mensagem que saiu, na ordem (fonte do replay)
        timeseries.csv   uma linha por janela neural, pronta para análise
        events.jsonl     só os eventos científicos
        summary.json     agregados no fim

## Por que quatro arquivos e não um

Cada um responde a uma pergunta diferente e tem um leitor diferente.

`telemetry.jsonl` é a **fonte de verdade**: é literalmente o que a interface
recebeu. O replay serve esse arquivo de volta na mesma porta, e para a Unity não
há diferença entre corrida ao vivo e gravada. Ele é grande e não serve para
análise.

`timeseries.csv` é o mesmo conteúdo reduzido ao que se plota: uma linha por
janela de 10 ms, colunas fixas. Abre em qualquer lugar sem parser.

`events.jsonl` separa o raro do contínuo. Um spike do Giant Fiber numa corrida
de 600 s está perdido dentro de 60.000 linhas de série temporal; aqui ele é uma
linha.

`summary.json` é o que uma tabela comparativa lê. Sem ele, comparar dez corridas
exigiria reprocessar dez séries temporais.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

RAIZ_RUNS = Path(__file__).resolve().parents[2] / "runs"

# Colunas da série temporal, em ordem fixa. Fixa importa: uma coluna que muda
# de lugar entre corridas quebra qualquer análise que as compare.
COLUNAS = [
    "t_s", "passo",
    "entrada_hz", "spikes_sensoriais",
    "gf_exc_mV", "gf_inib_mV", "gf_liquido_mV", "gf_v_min_mV", "gf_spikes",
    "ttmn_spikes", "fugas_ate_agora",
    "drive_esq", "drive_dir",
    "pos_x_mm", "pos_y_mm", "pos_z_mm",
]


class Registro:
    """
    Escreve os quatro arquivos de uma corrida.

    Não decide nada sobre a simulação: recebe o que já foi medido. Se algum
    número aqui estiver errado, ele já estava errado antes de chegar.
    """

    def __init__(self, id_corrida: str, metadata: dict,
                 raiz: Path | None = None, carimbo: bool = True):
        marca = datetime.now().strftime("%Y-%m-%d_%H%M%S_") if carimbo else ""
        self.pasta = (raiz or RAIZ_RUNS) / f"{marca}{id_corrida}"
        self.pasta.mkdir(parents=True, exist_ok=True)
        (self.pasta / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        self._f_ts = (self.pasta / "timeseries.csv").open(
            "w", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._f_ts, fieldnames=COLUNAS,
                                   extrasaction="ignore")
        self._csv.writeheader()
        self._f_ev = (self.pasta / "events.jsonl").open("w", encoding="utf-8")
        self.n_linhas = 0
        self.n_eventos = 0

    def linha(self, **campos) -> None:
        faltando = set(COLUNAS) - set(campos)
        if faltando:
            # Preencher com vazio em silêncio produziria uma coluna que parece
            # medida e não é. Melhor gritar na primeira linha.
            raise KeyError(f"faltam colunas na serie temporal: {sorted(faltando)}")
        self._csv.writerow(campos)
        self.n_linhas += 1

    def evento(self, t_s: float, tipo: str, detalhe: dict | None = None) -> None:
        self._f_ev.write(json.dumps(
            {"t_s": round(float(t_s), 6), "tipo": tipo, "detalhe": detalhe or {}},
            ensure_ascii=False) + "\n")
        self.n_eventos += 1

    def resumo(self, dados: dict) -> None:
        (self.pasta / "summary.json").write_text(
            json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")

    def fechar(self) -> None:
        for f in (self._f_ts, self._f_ev):
            try:
                f.close()
            except Exception:                                 # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()


def le_resumo(pasta: str | Path) -> dict:
    """Resumo de uma corrida, com a receita junto. Para tabelas comparativas."""
    pasta = Path(pasta)
    resumo = json.loads((pasta / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((pasta / "metadata.json").read_text(encoding="utf-8"))
    resumo["receita"] = meta.get("receita", {})
    resumo["hash_ciencia"] = meta.get("hash_ciencia")
    resumo["pasta"] = str(pasta)
    return resumo


def lista_corridas(raiz: Path | None = None) -> list[Path]:
    """Corridas completas, mais recentes primeiro."""
    raiz = raiz or RAIZ_RUNS
    if not raiz.exists():
        return []
    return sorted((p for p in raiz.iterdir()
                   if p.is_dir() and (p / "summary.json").exists()),
                  reverse=True)
