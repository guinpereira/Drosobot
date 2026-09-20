"""
Abre uma corrida gravada no Drosobot Lab, sem refazer a simulacao.

    python -m sim.replay                       # a mais recente
    python -m sim.replay runs/2026-..._looming_whole_centro_seed0
    python -m sim.replay --lista
    python -m sim.replay <pasta> --velocidade 4

Depois: Unity em unity/DrosobotLab -> Play.

## A corrida gravada e a fonte de verdade

O replay serve de volta, na mesma porta, as MESMAS mensagens que sairam da
simulacao, na mesma ordem e com o mesmo espacamento. Pra interface nao ha
diferenca entre ao vivo e gravado -- e por isso ela nao precisa de nenhum
codigo novo pra reproduzir.

Isso importa cientificamente: o que aparece na tela no replay e literalmente o
que a simulacao produziu, nao uma reconstrucao a partir de resumo. Um segundo
de mosca custa ~5 s de relogio com o conectoma inteiro; reencenar seria caro e,
pior, poderia divergir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "sim"))

from lab.registro import lista_corridas                       # noqa: E402
from telemetry.recorder import reproduzir                     # noqa: E402


def descreve(pasta: Path) -> str:
    try:
        meta = json.loads((pasta / "metadata.json").read_text(encoding="utf-8"))
        r = meta.get("receita", {})
        return (f"{r.get('nome','?'):<18s} {r.get('escopo','?'):<8s} "
                f"{r.get('condicao') or '-':<12s} seed {r.get('seed','?')}  "
                f"{r.get('duracao_s','?')} s")
    except Exception:                                         # noqa: BLE001
        return "(sem metadata)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay de uma corrida gravada")
    ap.add_argument("pasta", nargs="?", help="pasta da corrida (padrao: a mais recente)")
    ap.add_argument("--lista", action="store_true")
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--velocidade", type=float, default=1.0,
                    help="1 = tempo original; 4 = quatro vezes mais rapido")
    args = ap.parse_args()

    corridas = lista_corridas()
    if args.lista or (not args.pasta and not corridas):
        if not corridas:
            print("nenhuma corrida gravada em runs/")
            return
        print("corridas gravadas (mais recentes primeiro):")
        for p in corridas[:30]:
            print(f"  {p.name:<52s} {descreve(p)}")
        return

    pasta = Path(args.pasta) if args.pasta else corridas[0]
    if not (pasta / "telemetry.jsonl").exists():
        raise SystemExit(f"{pasta} nao tem telemetry.jsonl -- foi gravada sem "
                         "telemetria?")
    print(f"[replay] {pasta.name}")
    print(f"[replay] {descreve(pasta)}")
    reproduzir(pasta, porta=args.porta, velocidade=args.velocidade)


if __name__ == "__main__":
    main()
