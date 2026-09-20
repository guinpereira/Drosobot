"""
Bateria de experimentos: escreve a configuração, sai, volta com os resultados.

    python -m sim.run_experiments sim/baterias/looming_lateral.json
    python -m sim.run_experiments --lista
    python -m sim.run_experiments <config> --seco      # só mostra o que rodaria

A configuração é o produto cartesiano de três listas — sementes, escopos e
condições — e cada ponto vira uma corrida gravada. Uma bateria de 3 sementes ×
2 escopos × 3 condições são 18 corridas, e a tabela final sai pronta.

```json
{
  "nome": "looming_lateral",
  "duracao_s": 1.0,
  "seeds": [0, 1, 2],
  "escopos": ["circuit", "whole"],
  "condicoes": [
    {"condicao": "esquerda", "arena": "looming",
     "estimulo": {"azimute_graus": 45}},
    {"condicao": "centro", "arena": "looming",
     "estimulo": {"azimute_graus": 0}}
  ]
}
```

## Por que produto cartesiano, e não uma lista de receitas

Porque a pergunta científica quase sempre é "o que muda quando eu vario X,
mantendo o resto". Escrever as 18 receitas à mão convida a erro de digitação
numa delas — e um experimento com um parâmetro errado no meio é pior que um
experimento que não rodou.

## Retomada

Uma bateria de horas não pode perder tudo porque a décima corrida falhou. Cada
corrida é independente e gravada assim que termina; uma falha é registrada e a
bateria continua. No fim, a tabela mostra quais faltaram.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "sim"))

from lab import Receita, le_resumo, roda                      # noqa: E402
from lab.analise import compara, texto                        # noqa: E402

BATERIAS = RAIZ / "sim" / "baterias"
RAIZ_RUNS = RAIZ / "runs"


def expande(cfg: dict) -> list[Receita]:
    """Config -> lista de receitas. Falha cedo se algum campo não existir."""
    base = {k: v for k, v in cfg.items()
            if k not in ("seeds", "escopos", "condicoes", "nome")}
    receitas = []
    for cond in cfg.get("condicoes") or [{}]:
        for escopo in cfg.get("escopos", ["whole"]):
            for seed in cfg.get("seeds", [0]):
                d = dict(base)
                d.update(cond)
                d.update({"nome": cfg["nome"], "escopo": escopo, "seed": seed})
                receitas.append(Receita.de_dict(d))
    return receitas


def roda_bateria(cfg: dict, raiz: Path | None = None,
                 seco: bool = False) -> dict:
    receitas = expande(cfg)
    raiz = raiz or RAIZ_RUNS
    print(f"== bateria {cfg['nome']} ==")
    print(f"  {len(receitas)} corridas: "
          f"{len(cfg.get('condicoes') or [{}])} condicoes x "
          f"{len(cfg.get('escopos', ['whole']))} escopos x "
          f"{len(cfg.get('seeds', [0]))} sementes")
    print(f"  {cfg.get('duracao_s', 2.0)} s de mosca cada")
    if seco:
        for r in receitas:
            print(f"    {r.id_corrida}")
        return {"seco": True, "receitas": [r.para_dict() for r in receitas]}

    t0 = time.perf_counter()
    resultados, falhas = [], []
    for i, r in enumerate(receitas, 1):
        print(f"  [{i}/{len(receitas)}]", end=" ")
        try:
            pasta = roda(r, raiz=raiz)
            resultados.append({"receita": r.para_dict(), "pasta": str(pasta)})
        except Exception as e:                                # noqa: BLE001
            # Uma corrida que falha nao pode derrubar horas de bateria. Fica
            # registrada com o traceback e a bateria segue.
            print(f"FALHOU: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            falhas.append({"receita": r.para_dict(),
                           "erro": f"{type(e).__name__}: {e}"})

    parede = time.perf_counter() - t0
    bateria = {
        "nome": cfg["nome"],
        "config": cfg,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "parede_s": round(parede, 1),
        "corridas": resultados,
        "falhas": falhas,
    }
    destino = raiz / f"bateria_{cfg['nome']}.json"
    destino.write_text(json.dumps(bateria, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\n  {len(resultados)} corridas em {parede/60:.1f} min"
          + (f", {len(falhas)} falharam" if falhas else ""))
    print(f"  {destino}")
    return bateria


def tabela(bateria: dict) -> str:
    """Uma linha por corrida, agrupada por condição."""
    linhas = [
        "",
        "  RESULTADOS",
        f"  {'condicao':<14s} {'escopo':<8s} {'seed':>5s} {'sens':>7s} "
        f"{'exc mV':>10s} {'inib mV':>11s} {'liq mV':>11s} {'GFspk':>6s} "
        f"{'TTMn':>6s} {'fugas':>6s}",
        "  " + "-" * 92,
    ]
    for c in bateria.get("corridas", []):
        try:
            s = le_resumo(c["pasta"])
        except Exception:                                     # noqa: BLE001
            continue
        r = s["receita"]
        linhas.append(
            f"  {r.get('condicao') or '-':<14s} {r['escopo']:<8s} "
            f"{r['seed']:>5d} {s['sensorial']['spikes_totais']:>7d} "
            f"{s['gf']['excitacao_mV']:>10,.1f} {s['gf']['inibicao_mV']:>11,.1f} "
            f"{s['gf']['liquido_mV']:>11,.1f} {s['gf']['spikes']:>6d} "
            f"{s['sensorial']['ttmn_spikes']:>6d} {s['desfecho']['fugas']:>6d}")
    if bateria.get("falhas"):
        linhas.append("")
        linhas.append(f"  {len(bateria['falhas'])} corrida(s) falharam:")
        for f in bateria["falhas"]:
            linhas.append(f"    {f['receita'].get('condicao')}/"
                          f"{f['receita'].get('escopo')}/"
                          f"seed{f['receita'].get('seed')}: {f['erro']}")
    return "\n".join(linhas)


def comparacoes(bateria: dict) -> str:
    """Circuito × whole para cada (condição, semente) que tiver os dois."""
    por_chave = {}
    for c in bateria.get("corridas", []):
        r = c["receita"]
        chave = (r.get("condicao", ""), r.get("seed"))
        por_chave.setdefault(chave, {})[r["escopo"]] = c["pasta"]

    saida = []
    for chave in sorted(por_chave, key=lambda k: (str(k[0]), k[1])):
        par = por_chave[chave]
        if "circuit" not in par or "whole" not in par:
            continue
        try:
            saida.append(texto(compara(par["circuit"], par["whole"])))
        except ValueError as e:
            saida.append(f"  {chave}: {e}")
    return "\n\n".join(saida)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bateria de experimentos")
    ap.add_argument("config", nargs="?", help="JSON da bateria")
    ap.add_argument("--lista", action="store_true",
                    help="lista as baterias disponiveis e sai")
    ap.add_argument("--seco", action="store_true",
                    help="so mostra o que rodaria")
    ap.add_argument("--raiz", default=None, help="onde gravar (padrao runs/)")
    ap.add_argument("--sem-comparacao", action="store_true")
    args = ap.parse_args()

    if args.lista or not args.config:
        print("baterias disponiveis:")
        for p in sorted(BATERIAS.glob("*.json")):
            cfg = json.loads(p.read_text(encoding="utf-8"))
            n = (len(cfg.get("condicoes") or [{}]) * len(cfg.get("escopos", [1]))
                 * len(cfg.get("seeds", [0])))
            print(f"  {p.name:<28s} {n:>3d} corridas  "
                  f"{cfg.get('duracao_s', 2.0)} s cada")
        return

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raiz = Path(args.raiz) if args.raiz else None
    bateria = roda_bateria(cfg, raiz=raiz, seco=args.seco)
    if args.seco:
        return
    print(tabela(bateria))
    if not args.sem_comparacao:
        texto_cmp = comparacoes(bateria)
        if texto_cmp:
            print()
            print(texto_cmp)


if __name__ == "__main__":
    main()
