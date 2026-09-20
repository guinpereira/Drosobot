"""
Uma tabela: FlyGym 1 x FlyGym 2, circuito x Male CNS inteiro.

    # no venv de referencia (FlyGym 1.2.1)
    .venv\\Scripts\\python sim\\compare_runtimes.py --physics flygym1

    # no venv paralelo (FlyGym 2.1.0, Python 3.14)
    .venv-flygym2\\Scripts\\python sim\\compare_runtimes.py --physics flygym2

    # e depois junta os dois
    .venv\\Scripts\\python sim\\compare_runtimes.py --tabela

Cada corrida grava um JSON em benchmarks/runtime/. `--tabela` le todos e imprime
a comparacao.

## Por que dois comandos e nao um

FlyGym 1 e 2 vivem em ambientes diferentes de proposito -- 2.x exige Python
>=3.12 e mujoco >=3.9, e instalar isso por cima do `.venv` mudaria a fisica dos
experimentos existentes. Um unico processo nao consegue importar os dois.

O cerebro, esse, e o mesmo nos dois: `sim/neural/` so precisa de numpy e
pyopencl, e o CSR vem do mesmo arquivo. E o ponto do `PhysicsAdapter`.

## O que e igual nos quatro, e o que nao da pra igualar

Igual: semente, estimulo de looming (a MESMA classe `Estimulo` -- esfera de raio
3 mm, 30 -> 4 mm, ciclo de 0,8 s), duracao, timestep 1e-4, realizacao Poisson
(mesma semente de rng) e a semantica do experimento.

NAO da pra igualar, e vai registrado em cada corrida:

  - o MODELO padrao do 2.x e mais leve: 55 pares de colisao contra 2268, nv 72
    contra 93
  - o chao do 2.x e `FlatGroundWorld`, com os parametros de contato dele; o do
    1.x vem da `MovingObjArena`
  - o controlador do 2.x e o `HybridTurningController` do flygym_demo; o do 1.x
    e o de `flygym.examples.locomotion`. Mesma semantica de sinal descendente,
    implementacoes diferentes

Por isso a analise separa GANHO OBSERVADO de CAUSA ATRIBUIDA. Dizer "o wrapper
do 2.x e N vezes mais rapido" seria errado: o modelo tambem mudou.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))

from drosobot_lab import entradas_do_gf, mede_gate  # noqa: E402
from profiler import Profiler  # noqa: E402

SAIDA = RAIZ / "benchmarks" / "runtime"
PAPEIS = RAIZ / "connectome" / "gf_roles.json"

DARK_THRESHOLD = 0.4
LOOM_GAIN = 240.0
LOOM_MAX_HZ = 20.0
VISION_HZ = 100
JANELA_MS = 10.0
DT = 1e-4
BASE_DRIVE, ESCAPE_DRIVE, ESCAPE_MS = 1.0, -0.5, 120.0
CICLO_S, DIST_LONGE, DIST_PERTO = 0.8, 30.0, 4.0


def monta_cerebro(escopo: str, backend: str):
    from neural import NeuralEngine, carrega_male_cns, subgrafo

    papeis_ids = json.loads(PAPEIS.read_text(encoding="utf-8"))
    c = carrega_male_cns()

    if escopo == "circuito":
        alvo = np.unique(np.concatenate([
            np.asarray(papeis_ids["upstream_gf_total"], dtype=np.int64),
            np.asarray(papeis_ids["DNp01"], dtype=np.int64),
            np.asarray(papeis_ids["TTMn"], dtype=np.int64)]))
        idx = c.indice_de(alvo)
        c = subgrafo(c, idx[idx >= 0])

    grupos = np.full(c.n, -1, dtype=np.int32)
    nomes, papeis = [], {}
    for nome in ("LC4/LPLC2", "DNp01", "TTMn"):
        idx = c.indice_de(papeis_ids[nome])
        idx = idx[idx >= 0]
        papeis[nome] = idx
        if len(idx):
            grupos[idx] = len(nomes)
            nomes.append(nome)

    return NeuralEngine(c, backend=backend, grupos=grupos,
                        nomes_grupos=nomes), papeis


def roda(physics: str, escopo: str, backend: str, duracao_s: float,
         colisao: str) -> dict:
    from physics import cria, MotorFrame

    eng, papeis = monta_cerebro(escopo, backend)
    corpo = cria(physics, self_collisions=colisao, timestep=DT, com_visao=True)
    frame = corpo.reset(seed=0)

    sens = papeis.get("LC4/LPLC2", np.zeros(0, np.int32))
    motor = papeis.get("TTMn", np.zeros(0, np.int32))
    taxas = np.zeros(eng.c.n, dtype=np.float64)
    rng = np.random.default_rng(0)

    idx_gf = papeis.get("DNp01", np.zeros(0, np.int32))
    gf_pre, gf_peso = entradas_do_gf(eng.c, idx_gf)

    escuro_lento = None
    dark_tau = 0.3 * VISION_HZ
    drive = np.array([BASE_DRIVE, BASE_DRIVE])
    escape_ate, escapes = -1.0, 0
    prof = Profiler(["physics", "vision", "neural", "leitura", "telemetry"])
    n_passos = int(duracao_s / DT)
    # o gate e ACUMULADO na corrida, nao so o ultimo valor
    gf_exc = gf_inib = 0.0
    gf_spikes = 0
    gf_vmin = 0.0

    for passo in range(n_passos):
        t_s = passo * DT
        fase = (t_s % CICLO_S) / CICLO_S
        dist = DIST_LONGE + (DIST_PERTO - DIST_LONGE) * fase
        corpo.antes_do_passo(t_s, dist)

        with prof("physics", "passo de fisica"):
            frame = corpo.passo(MotorFrame(drive=drive))
        prof.avanca_sim(DT * 1000)

        if not frame.retina_atualizou or frame.retina is None:
            continue

        with prof("vision", "quadro de retina"):
            escuro = (frame.retina < DARK_THRESHOLD).mean(axis=1)
            if escuro_lento is None:
                escuro_lento = escuro.copy()
            expansao = np.clip(escuro - escuro_lento, 0, None)
            escuro_lento += (escuro - escuro_lento) / dark_tau
            hz = float(np.clip(expansao * LOOM_GAIN, 0, LOOM_MAX_HZ).max())
            taxas[:] = 0.0
            if len(sens):
                taxas[sens] = hz

        with prof("neural", "janela de 10 ms"):
            eng.roda_poisson(JANELA_MS, taxas, rng, indices=sens)

        with prof("leitura", "janela"):
            gate = mede_gate(eng, idx_gf, gf_pre, gf_peso)
            gf_exc += gate["exc_mV"]
            gf_inib += gate["inib_mV"]
            gf_spikes += gate["spikes_gf"]
            gf_vmin = min(gf_vmin, gate["v_min_mV"])
            est = eng.le(motor) if len(motor) else None
            disparou = int(est.spike.sum()) if est is not None else 0

        if disparou and t_s > escape_ate:
            escape_ate = t_s + ESCAPE_MS / 1000.0
            escapes += 1
        drive = (np.array([ESCAPE_DRIVE, ESCAPE_DRIVE]) if t_s < escape_ate
                 else np.array([BASE_DRIVE, BASE_DRIVE]))

    v = prof.valores()
    r_eng = eng.resumo()
    saida = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "physics": physics,
        "escopo": escopo,
        "rotulo": f"{physics}+{escopo}",
        "duracao_s": duracao_s,
        "corpo": corpo.resumo(),
        "cerebro": {k: r_eng.get(k) for k in
                    ("backend", "device", "neurons_simulated",
                     "edges_simulated", "vram_mib")},
        "rtf": v["_total"]["rtf"],
        "ms_por_seg_simulado": {
            k: v[k]["ms_por_seg_simulado"]
            for k in ("physics", "vision", "neural", "leitura")},
        "outro_ms_por_seg_simulado": v["_outro"]["ms_por_seg_simulado"],
        "total_ms_por_seg_simulado": v["_total"]["ms_por_seg_simulado"],
        "desfecho": {"fugas": escapes, "posicao_final": frame.posicao.tolist()},
        "gf": {"spikes": gf_spikes, "excitacao_mV": round(gf_exc, 1),
               "inibicao_mV": round(gf_inib, 1),
               "liquido_mV": round(gf_exc + gf_inib, 1),
               "v_min_mV": round(gf_vmin, 1),
               "arestas_entrando": int(len(gf_pre))},
        "diferencas_declaradas": {
            "collision_pairs": corpo.resumo()["collision_pairs"],
            "nv": corpo.resumo()["nv"],
            "arena": corpo.resumo().get("arena", "looming"),
            "nota": corpo.resumo().get("diferenca_declarada", ""),
        },
    }
    corpo.fecha()
    print(prof.relatorio())
    print(f"\n  fugas: {escapes}   posicao final {frame.posicao.round(3)}")
    return saida


def tabela():
    SAIDA.mkdir(parents=True, exist_ok=True)
    arqs = sorted(SAIDA.glob("run_*.json"))
    if not arqs:
        print("nenhuma corrida em benchmarks/runtime/. Rode com --physics primeiro.")
        return
    corridas = {}
    for a in arqs:
        d = json.loads(a.read_text(encoding="utf-8"))
        corridas[d["rotulo"]] = d       # a mais recente de cada rotulo vence

    ordem = ["flygym1+circuito", "flygym1+whole",
             "flygym2+circuito", "flygym2+whole"]
    print()
    print("  CUSTO -- ms de relogio por SEGUNDO SIMULADO")
    print(f"  {'configuracao':<20s} {'RTF':>7s} {'physics':>9s} {'neural':>9s} "
          f"{'vision':>8s} {'leitura':>8s} {'telem':>7s} {'outro':>7s} {'total':>9s}")
    print("  " + "-" * 92)
    for rot in ordem:
        d = corridas.get(rot)
        if d is None:
            print(f"  {rot:<20s} {'(nao rodado)':>7s}")
            continue
        m = d["ms_por_seg_simulado"]
        print(f"  {rot:<20s} {d['rtf']:7.4f} {m['physics']:9.0f} {m['neural']:9.0f} "
              f"{m['vision']:8.0f} {m['leitura']:8.0f} "
              f"{m.get('telemetry', 0):7.0f} "
              f"{d['outro_ms_por_seg_simulado']:7.0f} "
              f"{d['total_ms_por_seg_simulado']:9.0f}")

    print()
    print("  COMPORTAMENTO e GIANT FIBER")
    print(f"  {'configuracao':<20s} {'fugas':>6s} {'GFspk':>6s} {'exc mV':>9s} "
          f"{'inib mV':>10s} {'liq mV':>10s} {'v min mV':>9s} {'arestas':>8s}")
    print("  " + "-" * 92)
    for rot in ordem:
        d = corridas.get(rot)
        if d is None:
            continue
        g = d.get("gf", {})
        print(f"  {rot:<20s} {d['desfecho']['fugas']:6d} {g.get('spikes', 0):6d} "
              f"{g.get('excitacao_mV', 0):9.1f} {g.get('inibicao_mV', 0):10.1f} "
              f"{g.get('liquido_mV', 0):10.1f} {g.get('v_min_mV', 0):9.1f} "
              f"{g.get('arestas_entrando', 0):8d}")
    print()
    for rot in ordem:
        d = corridas.get(rot)
        if d:
            c, b = d["corpo"], d["cerebro"]
            print(f"  {rot}: {c['physics_backend']}, {c['collision_pairs']} pares, "
                  f"nv={c['nv']} | {b['neurons_simulated']:,} neuronios, "
                  f"{b['edges_simulated']:,} arestas, {b['backend']}")
    print()
    print("  DIFERENCAS DECLARADAS (o que nao da pra igualar entre 1.x e 2.x)")
    for rot in ordem:
        d = corridas.get(rot)
        if d and d.get("diferencas_declaradas"):
            dd = d["diferencas_declaradas"]
            print(f"    {rot}: {dd['collision_pairs']} pares, nv={dd['nv']}, "
                  f"arena={dd['arena']}")
    print()
    print("  GANHO OBSERVADO x CAUSA ATRIBUIDA")
    a = corridas.get("flygym1+whole")
    b = corridas.get("flygym2+whole")
    if a and b:
        ganho = a["total_ms_por_seg_simulado"] / b["total_ms_por_seg_simulado"]
        print(f"    ponta a ponta, whole CNS: {ganho:.2f}x mais rapido no 2.x")
    print("    NAO atribuivel so ao wrapper: o modelo padrao do 2.x tambem e mais")
    print("    leve (55 pares contra 2268, nv 72 contra 93) e o controlador e")
    print("    outra implementacao. As duas causas nao foram separadas -- fazer")
    print("    isso exigiria rodar o 2.x com o modelo do 1.x.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", choices=["flygym1", "flygym2"])
    ap.add_argument("--escopo", choices=["circuito", "whole", "ambos"],
                    default="ambos")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--duracao", type=float, default=0.5)
    ap.add_argument("--colisao", default="legs")
    ap.add_argument("--tabela", action="store_true")
    args = ap.parse_args()

    if args.tabela or not args.physics:
        tabela()
        return

    SAIDA.mkdir(parents=True, exist_ok=True)
    escopos = (["circuito", "whole"] if args.escopo == "ambos" else [args.escopo])
    for escopo in escopos:
        print(f"\n===== {args.physics} + {escopo} =====")
        d = roda(args.physics, escopo, args.backend, args.duracao, args.colisao)
        destino = SAIDA / f"run_{args.physics}_{escopo}.json"
        destino.write_text(json.dumps(d, indent=2, default=float),
                           encoding="utf-8")
        print(f"  salvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
