"""
Drosobot Lab: um comando, o laco inteiro.

    .venv\\Scripts\\python sim\\drosobot_lab.py
    .venv\\Scripts\\python sim\\drosobot_lab.py --cns whole --telemetry
    .venv\\Scripts\\python sim\\drosobot_lab.py --backend cpu --colisao legs
    .venv\\Scripts\\python sim\\drosobot_lab.py --duracao 2.0 --sem-unity

    FlyGym / MuJoCo          <- autoridade da fisica
        retina, contatos, propriocepcao
            v
    Drosobot Neural Engine   <- autoridade do comportamento
        Male CNS na GPU
            v
        saida motora
            v
    corpo no FlyGym
        (repete)

        e em paralelo: telemetria -> Unity Drosobot Lab

## Escopo, dito antes de qualquer numero

    WHOLE CONNECTOME SIMULATED
        164.451 neuronios e 25,5M arestas participam da dinamica

    WHOLE SENSORIMOTOR MODEL
        NAO. So a via de looming (LC4/LPLC2) e a motora (DNp01, TTMn) tem
        semantica sensorial/motora modelada. O resto da rede participa pelo que
        chega pela conectividade -- o que e real, mas nao e um modelo completo
        de comportamento.

As duas coisas sao diferentes e o runtime reporta as duas separadas, em
`--info`, na telemetria e na interface.

## O que NAO acontece aqui

Nao ha readback do cerebro por passo. v, g, refratario e o anel de atraso ficam
na GPU. A CPU le, por janela: a saida motora, os neuronios com morfologia 3D, o
selecionado e a atividade agregada por populacao. Com 164 mil neuronios, copiar
tudo por passo desmontaria o motivo de estar na GPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from profiler import Profiler  # noqa: E402

# --- transducao retina -> taxa: escolha NOSSA, nao esta no conectoma ---
DARK_THRESHOLD = 0.4
LOOM_GAIN = 240.0
LOOM_MAX_HZ = 20.0
# A populacao sensorial dispara como POISSON, nao recebe corrente: e assim que o
# modelo define a entrada (ver sim/fast_lif.py), e cada spike entrega o peso
# sinaptico cheio. A primeira versao deste arquivo injetava corrente continua
# equivalente -- e mais fraco, e o Giant Fiber nunca disparava.
BASE_DRIVE = 1.0
ESCAPE_DRIVE = -0.5
ESCAPE_MS = 120.0
VISION_HZ = 100
JANELA_MS = 10.0
DT_FISICA = 1e-4
CICLO_S, DIST_LONGE, DIST_PERTO = 0.8, 30.0, 4.0

CONTATOS = [f"{p}{s}" for p in ("LF", "LM", "LH", "RF", "RM", "RH")
            for s in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")]


# ------------------------------------------------------------------ corpo

def monta_corpo(self_collisions: str):
    """
    FlyGym 1.x com a arena de looming. `self_collisions` escolhe o conjunto de
    colisao (ver docs/research/COLLISION_PAIR_AUDIT.md).
    """
    import importlib.util
    import os
    import flygym.examples as fex
    from flygym import Fly
    from flygym.examples.locomotion import HybridTurningController

    spec = importlib.util.spec_from_file_location(
        "flygym_vision_arena", os.path.join(fex.__path__[0], "vision", "arena.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class LoomingArena(mod.MovingObjArena):
        def step(self, dt, physics):
            pass          # a posicao da esfera vem do laco principal

    arena = LoomingArena(obj_radius=3, init_ball_pos=(30, 0))
    fly = Fly(enable_vision=True, vision_refresh_rate=VISION_HZ,
              spawn_pos=(0, 0, 0.3), contact_sensor_placements=CONTATOS,
              self_collisions=self_collisions)
    sim = HybridTurningController(fly=fly, arena=arena, timestep=DT_FISICA)
    obs, _ = sim.reset(seed=0)
    return sim, arena, obs


# ---------------------------------------------------------------- cerebro

def monta_cerebro(escopo: str, backend: str):
    """
    Devolve (engine, papeis). `papeis` mapeia nome -> indices no grafo.

    escopo="whole"    o Male CNS inteiro
    escopo="circuito" so o circuito do Giant Fiber, como os experimentos atuais
    """
    from neural import NeuralEngine, carrega_male_cns

    import pandas as pd
    from connectome_model import CONNECTOME, load_properties, gf_input_population

    c = carrega_male_cns()
    props = load_properties()

    # --- quem sao os sensores e os motores, pelos bodyIds reais ---
    up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
    down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
    sensor_ids, is_loom, _ = gf_input_population(props, up)
    loom_ids = [b for b, f in zip(sensor_ids, is_loom) if f]
    gf_ids = [10001, 10010]
    motor_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()

    if escopo == "circuito":
        from neural import subgrafo
        alvo = np.unique(np.concatenate([np.asarray(sensor_ids),
                                         np.asarray(gf_ids),
                                         np.asarray(motor_ids)]))
        idx = c.indice_de(alvo)
        c = subgrafo(c, idx[idx >= 0])

    # grupos pra atividade de populacao: quem nao esta num papel fica em -1
    grupos = np.full(c.n, -1, dtype=np.int32)
    nomes = []
    papeis = {}
    for nome, ids in (("LC4/LPLC2", loom_ids), ("DNp01", gf_ids),
                      ("TTMn", motor_ids)):
        idx = c.indice_de(ids)
        idx = idx[idx >= 0]
        papeis[nome] = idx
        if len(idx):
            grupos[idx] = len(nomes)
            nomes.append(nome)

    eng = NeuralEngine(c, backend=backend, grupos=grupos, nomes_grupos=nomes)
    return eng, papeis, nomes


# ------------------------------------------------------------------ laco

def main():
    ap = argparse.ArgumentParser(description="Drosobot Lab -- laco integrado")
    ap.add_argument("--cns", choices=["whole", "circuito"], default="whole",
                    help="conectoma inteiro ou so o circuito do Giant Fiber")
    ap.add_argument("--backend", default="auto",
                    help="auto | opencl | cpu | d3d12")
    # PADRAO = "legs". A poda `tarsi` deixa o mj_step 10,66x mais rapido em
    # caminhada reta e com trajetoria identica, mas a regressao nos quatro
    # comportamentos REPROVOU: curva/optomotor diverge em 1,36 rad de qpos e 23
    # graus de orientacao. Ver docs/research/COLLISION_PAIR_AUDIT.md.
    ap.add_argument("--colisao", choices=["legs", "tarsi", "none"], default="legs",
                    help="conjunto de auto-colisao. legs = validado; "
                         "tarsi = mais rapido, mas REPROVOU em curva/optomotor")
    ap.add_argument("--duracao", type=float, default=1.0, help="segundos de mosca")
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--info", action="store_true", help="so imprime o setup e sai")
    args = ap.parse_args()

    from telemetry import protocol
    from telemetry.server import abrir as abrir_telemetria

    print("== Drosobot Lab ==")
    t0 = time.perf_counter()
    eng, papeis, nomes_grupos = monta_cerebro(args.cns, args.backend)
    r = eng.resumo()
    print(f"  cerebro   {r['neurons_simulated']:,} neuronios, "
          f"{r['edges_simulated']:,} arestas")
    print(f"            backend {r['backend']} em {r['device']}")
    print(f"            VRAM {r.get('vram_mib', 0):.1f} MiB   "
          f"({time.perf_counter()-t0:.1f} s)")
    for nome, idx in papeis.items():
        print(f"            {nome:10s} {len(idx):5d} neuronios")

    t0 = time.perf_counter()
    sim, arena, obs = monta_corpo(args.colisao)
    npair = sim.physics.model.ptr.npair
    print(f"  corpo     FlyGym 1.2.1 / MuJoCo CPU, colisao={args.colisao} "
          f"({npair} pares)   ({time.perf_counter()-t0:.1f} s)")
    print()
    print("  ESCOPO    WHOLE CONNECTOME SIMULATED"
          if args.cns == "whole" else "  ESCOPO    circuito do Giant Fiber")
    print("            NAO e um modelo sensorimotor completo: so a via de "
          "looming e a motora")
    print("            tem semantica modelada; o resto participa pela "
          "conectividade.")
    if args.info:
        return

    tel = abrir_telemetria(porta=args.porta, ativo=args.telemetry)
    _publica_abertura(tel, protocol, eng, papeis, nomes_grupos, args, npair)

    prof = Profiler(["physics", "vision", "neural", "telemetry"])
    sens = papeis.get("LC4/LPLC2", np.zeros(0, dtype=np.int32))
    motor = papeis.get("TTMn", np.zeros(0, dtype=np.int32))
    # taxa em Hz por neuronio; o engine converte em probabilidade por passo
    taxas = np.zeros(eng.c.n, dtype=np.float64)
    rng = np.random.default_rng(0)

    retina0 = np.asarray(obs["vision"]).mean(axis=2)
    escuro_lento = (retina0 < DARK_THRESHOLD).mean(axis=1)
    # a adaptacao do fundo e 0,3 s de tempo REAL; a retina roda a VISION_HZ
    dark_tau = 0.3 * VISION_HZ
    drive = np.array([BASE_DRIVE, BASE_DRIVE])
    escape_ate = -1.0
    n_passos = int(args.duracao / DT_FISICA)
    passo = 0
    escapes = 0
    ultimo_log = time.time()

    print(f"\nrodando {args.duracao}s de mosca...")
    while passo < n_passos:
        t_s = passo * DT_FISICA

        # estimulo: esfera se aproximando. Mocap, nao fisica emergente.
        fase = (t_s % CICLO_S) / CICLO_S
        dist = DIST_LONGE + (DIST_PERTO - DIST_LONGE) * fase
        pos = np.asarray(obs["fly"][0])
        alvo = np.array([pos[0] + dist, pos[1], 2.5], dtype="float32")
        arena.ball_pos = alvo
        sim.physics.bind(arena.object_body).mocap_pos = alvo

        with prof("physics", "passo de fisica"):
            obs, _, _, _, info = sim.step(drive)
        passo += 1
        prof.avanca_sim(DT_FISICA * 1000)

        if not info.get("vision_updated", False):
            continue

        with prof("vision", "quadro de retina"):
            retina = np.asarray(obs["vision"]).mean(axis=2)
            escuro = (retina < DARK_THRESHOLD).mean(axis=1)
            expansao = np.clip(escuro - escuro_lento, 0, None)
            escuro_lento += (escuro - escuro_lento) / dark_tau
            hz = np.clip(expansao * LOOM_GAIN, 0, LOOM_MAX_HZ)
            # os dois olhos alimentam a mesma populacao declarada; separar por
            # hemisferio exigiria o somaSide de cada LC4/LPLC2 no grafo completo
            taxas[:] = 0.0
            if len(sens):
                taxas[sens] = float(hz.max())

        with prof("neural", "janela de 10 ms"):
            eng.roda_poisson(JANELA_MS, taxas, rng, indices=sens)

        # --- saida motora: le SO os TTMn, nao o cerebro ---
        saida = eng.le(motor) if len(motor) else None
        disparou = int(saida.spike.sum()) if saida is not None else 0
        if disparou and t_s > escape_ate:
            escape_ate = t_s + ESCAPE_MS / 1000.0
            escapes += 1
        drive = (np.array([ESCAPE_DRIVE, ESCAPE_DRIVE]) if t_s < escape_ate
                 else np.array([BASE_DRIVE, BASE_DRIVE]))

        with prof("telemetry", "quadro"):
            if tel.ativo:
                _publica_quadro(tel, protocol, eng, prof, obs, drive, t_s,
                                passo, hz, papeis, nomes_grupos, disparou)

        agora = time.time()
        if agora - ultimo_log > 2.0:
            ultimo_log = agora
            v = prof.valores()
            print(f"  t={t_s:5.2f}s  RTF {v['_total']['rtf']:.4f}  "
                  f"looming {hz.max():5.1f} Hz  TTMn {disparou}  "
                  f"fugas {escapes}   por s simulado: "
                  f"fis {v['physics']['ms_por_seg_simulado']:.0f}ms "
                  f"neural {v['neural']['ms_por_seg_simulado']:.0f}ms")

    print("\n== profiler ==")
    print(prof.relatorio())
    print(f"\n  fugas disparadas: {escapes}")
    if tel.ativo:
        tel.enviar(protocol.statistics(prof.sim_ms / 1000.0, {
            "escapes": escapes, **prof.valores()["_total"]}))
        tel.enviar(protocol.bye("corrida terminada"))
        tel.fechar()


def _publica_abertura(tel, protocol, eng, papeis, nomes_grupos, args, npair):
    r = eng.resumo()
    circuitos = [{"name": nome, "role": nome,
                  "body_ids": [int(b) for b in eng.c.body_ids[idx]],
                  "sides": [None] * len(idx), "types": [nome] * len(idx),
                  "neurotransmitters": [None] * len(idx)}
                 for nome, idx in papeis.items() if len(idx)]
    tel.enviar(protocol.experiment_info(
        experiment_id=f"lab_{args.cns}",
        name=f"Drosobot Lab -- {'Male CNS inteiro' if args.cns=='whole' else 'circuito GF'}",
        description=("Laco fechado: FlyGym/MuJoCo -> retina -> CNS na GPU -> "
                     "TTMn -> marcha."),
        parameters={"dt_physics_s": DT_FISICA, "dt_neural_ms": eng.dt,
                    "vision_hz": VISION_HZ, "loom_gain": LOOM_GAIN,
                    "collision_set": args.colisao, "collision_pairs": int(npair),
                    "entrada": "poisson_por_taxa"},
        provenance={
            "body_ids": protocol.DATA, "synapse_weight": protocol.DATA,
            "neurotransmitter": protocol.DATA,
            "membrane_potential": protocol.MODEL, "spikes": protocol.MODEL,
            "synapse_sign": protocol.MODEL,
            "loom_gain": protocol.ASSUMPTION,
            "motor_to_gait_mapping": protocol.ASSUMPTION,
            "collision_set": protocol.ASSUMPTION,
        },
        circuits=circuitos,
        runtime={
            "os": sys.platform,
            "neural_backend": r["backend"],
            "neural_device": r["device"],
            "physics_backend": "FlyGym 1.2.1 / MuJoCo CPU",
            "neurons_simulated": r["neurons_simulated"],
            "edges_simulated": r["edges_simulated"],
            "vram_mib": r.get("vram_mib", 0),
            "population_names": nomes_grupos,
        },
        scope={"simulated": "whole_connectome" if args.cns == "whole" else "circuit",
               "sensorimotor_model": "partial",
               "note": r["note"]},
    ))
    tel.enviar(protocol.scene_info(
        arena={"kind": "looming_sphere"},
        stimulus={"kind": "approaching_sphere", "radius": 3.0,
                  "start_distance": DIST_LONGE, "end_distance": DIST_PERTO,
                  "cycle_s": CICLO_S}))


def _publica_quadro(tel, protocol, eng, prof, obs, drive, t_s, passo, hz,
                    papeis, nomes_grupos, disparou):
    v = prof.valores()
    tel.enviar(protocol.frame(step=passo, sim_time=t_s, wall_time=time.time(),
                              real_time_factor=v["_total"]["rtf"],
                              position=obs["fly"][0], drive=drive,
                              profile={k: v[k]["ms_por_seg_simulado"] for k in
                                       ("physics", "vision", "neural", "telemetry")}))
    # SO os papeis, nunca o cerebro inteiro
    camadas = []
    for nome, idx in papeis.items():
        if not len(idx):
            continue
        est = eng.le(idx)
        camadas.append({"name": nome, "spikes": est.spike.tolist(),
                        "v_mV": est.v_mV.tolist(), "g_mV": est.g_mV.tolist()})
    tel.enviar(protocol.neural_activity(t_s, camadas))
    # e a atividade agregada de quem nao tem representacao individual
    soma = eng.atividade_por_grupo(zerar=True)
    tel.enviar(protocol.statistics(t_s, {
        "population_activity": {n: int(s) for n, s in zip(nomes_grupos, soma)},
        "looming_hz": float(hz.max()), "ttmn_spikes": disparou,
    }))


if __name__ == "__main__":
    main()
