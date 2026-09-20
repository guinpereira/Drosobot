"""
Drosobot Lab: um comando, o laboratorio inteiro.

    python sim/drosobot_lab.py --physics flygym2 --neural opencl --cns whole \\
                               --experiment looming --telemetry
    python sim/drosobot_lab.py --physics flygym1 --cns circuit --telemetry
    python sim/drosobot_lab.py --info

    PhysicsAdapter          <- autoridade da fisica (FlyGym 1 ou 2 / MuJoCo)
        SensorFrame
            v
    Drosobot Neural Engine  <- autoridade do comportamento (CPU, OpenCL, D3D12)
        Male CNS ou circuito
            v
        MotorFrame
            v
    PhysicsAdapter
        (repete)

        e em paralelo: telemetria -> Unity Drosobot Lab

Este arquivo NAO conhece a API do FlyGym. Toda a fisica passa por
`sim/physics/`, e todo o cerebro por `sim/neural/`. Trocar de simulador ou de
backend de GPU nao mexe aqui.

## ESCOPO -- a distincao que nunca pode sumir

    WHOLE CONNECTOME SIMULATED     164.451 neuronios, 25,5M arestas participam
    FULL FUNCTIONAL BRAIN          NAO

So a via de looming (LC4/LPLC2) e a motora (DNp01, TTMn) tem semantica
sensorial/motora modelada. O resto participa pelo que chega pela conectividade.
As duas coisas sao diferentes, e o runtime reporta as duas separadas -- em
`--info`, na telemetria e na interface.

## Sem entrada artificial

Os 311 sensores sao a unica origem. Nao ha entrada tonica global; o
`TONIC_INHIB_HZ` dos experimentos de circuito NAO e usado aqui e nao deve
voltar -- no conectoma inteiro os inibitorios tem fonte de verdade.

## Sem readback do cerebro

v, g, refratario e o anel ficam na GPU. A CPU le, por janela: a saida motora, os
papeis declarados, o balanco no Giant Fiber e a atividade agregada por
populacao.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))

from profiler import Profiler  # noqa: E402

PAPEIS_JSON = RAIZ / "connectome" / "gf_roles.json"

# transducao retina -> taxa: escolha NOSSA (ASSUMPTION), nao esta no conectoma
DARK_THRESHOLD = 0.4
LOOM_GAIN = 240.0
LOOM_MAX_HZ = 20.0
VISION_HZ = 100
JANELA_MS = 10.0
DT = 1e-4
BASE_DRIVE, ESCAPE_DRIVE, ESCAPE_MS = 1.0, -0.5, 120.0
# pose vai na cadencia da VISUALIZACAO, nao na da fisica
POSE_HZ = 30
# Aviso de limitacao do modelo. O LIF de Shiu et al. nao tem potencial de
# reversao inibitorio, entao inibicao convergente forte leva v a valores nao
# fisiologicos. NAO ha clamp: o valor vai como veio, e o aviso e so registro.
V_AVISO_MV = -150.0


def monta_cerebro(escopo: str, backend: str):
    """Devolve (engine, papeis, nomes_grupos). `papeis`: nome -> indices."""
    from neural import NeuralEngine, carrega_male_cns, subgrafo

    ids = json.loads(PAPEIS_JSON.read_text(encoding="utf-8"))
    c = carrega_male_cns()

    if escopo == "circuit":
        alvo = np.unique(np.concatenate([
            np.asarray(ids["upstream_gf_total"], dtype=np.int64),
            np.asarray(ids["DNp01"], dtype=np.int64),
            np.asarray(ids["TTMn"], dtype=np.int64)]))
        idx = c.indice_de(alvo)
        c = subgrafo(c, idx[idx >= 0])

    grupos = np.full(c.n, -1, dtype=np.int32)
    nomes, papeis = [], {}
    for nome in ("LC4/LPLC2", "DNp01", "TTMn"):
        idx = c.indice_de(ids[nome])
        idx = idx[idx >= 0]
        papeis[nome] = idx
        if len(idx):
            grupos[idx] = len(nomes)
            nomes.append(nome)

    eng = NeuralEngine(c, backend=backend, grupos=grupos, nomes_grupos=nomes)
    return eng, papeis, nomes


def entradas_do_gf(c, idx_gf):
    """
    Quem entra no Giant Fiber, com peso e sinal.

    Sem saber QUEM entra nao da pra mostrar de onde vem a inibicao. Calculado
    uma vez: e uma varredura do CSR, e o grafo nao muda durante a corrida.
    """
    alvo = set(int(i) for i in idx_gf)
    pre, peso = [], []
    ro, tg, w = c.row_offsets, c.targets, c.weights
    for i in range(c.n):
        a, b = ro[i], ro[i + 1]
        if b <= a:
            continue
        m = np.isin(tg[a:b], list(alvo))
        if m.any():
            pre.append(np.full(int(m.sum()), i, dtype=np.int64))
            peso.append(w[a:b][m])
    if not pre:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    return (np.concatenate(pre).astype(np.int32),
            np.concatenate(peso).astype(np.float32))


def mede_gate(eng, idx_gf, gf_pre, gf_peso):
    """
    Balanco de entrada no Giant Fiber neste passo.

    Dado real, medido do estado da GPU -- nao animacao decorativa. E o que
    permite VER o gate acontecer: no circuito o liquido fica positivo e o GF
    dispara; no conectoma inteiro a inibicao dobra e o liquido troca de sinal.
    """
    if not len(idx_gf):
        return {"exc_mV": 0.0, "inib_mV": 0.0, "liquido_mV": 0.0,
                "v_min_mV": 0.0, "spikes_gf": 0, "top_inib": []}
    est_gf = eng.le(idx_gf)
    exc = inib = 0.0
    top = []
    if len(gf_pre):
        est_pre = eng.le(gf_pre)
        disp = est_pre.spike.astype(bool)
        if disp.any():
            contrib = gf_peso[disp]
            exc = float(contrib[contrib > 0].sum())
            inib = float(contrib[contrib < 0].sum())
            neg = np.flatnonzero(disp & (gf_peso < 0))
            if len(neg):
                ordem = neg[np.argsort(gf_peso[neg])][:5]
                top = [{"body_id": int(eng.c.body_ids[gf_pre[k]]),
                        "mV": round(float(gf_peso[k]), 2)} for k in ordem]
    return {"exc_mV": round(exc, 2), "inib_mV": round(inib, 2),
            "liquido_mV": round(exc + inib, 2),
            "v_min_mV": round(float(est_gf.v_mV.min()), 2),
            "spikes_gf": int(est_gf.spike.sum()), "top_inib": top}


def main():
    ap = argparse.ArgumentParser(description="Drosobot Lab")
    ap.add_argument("--physics", choices=["flygym1", "flygym2"], default="flygym1",
                    help="flygym2 e o caminho principal; flygym1 e a referencia")
    ap.add_argument("--neural", default="auto", help="auto | opencl | cpu | d3d12")
    ap.add_argument("--cns", choices=["whole", "circuit"], default="whole")
    ap.add_argument("--experiment", choices=["looming", "flat"], default="looming")
    ap.add_argument("--colisao", choices=["legs", "tarsi", "none"], default="legs",
                    help="legs = padrao cientifico validado; tarsi = otimizacao "
                         "por cenario (REPROVOU em curva, ver COLLISION_PAIR_AUDIT)")
    ap.add_argument("--duracao", type=float, default=2.0)
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()

    from physics import MotorFrame, cria
    from telemetry import protocol
    from telemetry.server import abrir as abrir_telemetria

    print("== Drosobot Lab ==")
    t0 = time.perf_counter()
    eng, papeis, nomes_grupos = monta_cerebro(args.cns, args.neural)
    r = eng.resumo()
    print(f"  cerebro   {r['neurons_simulated']:,} neuronios, "
          f"{r['edges_simulated']:,} arestas")
    print(f"            {r['backend']} em {r['device']}   "
          f"VRAM {r.get('vram_mib', 0):.1f} MiB   ({time.perf_counter()-t0:.1f} s)")
    for nome, idx in papeis.items():
        print(f"            {nome:10s} {len(idx):5d}")

    t0 = time.perf_counter()
    corpo = cria(args.physics, arena=args.experiment,
                 self_collisions=args.colisao, timestep=DT, com_visao=True)
    frame = corpo.reset(seed=0)
    rc = corpo.resumo()
    print(f"  corpo     {rc['physics_backend']}, arena={args.experiment}, "
          f"{rc['collision_pairs']} pares, nv={rc['nv']}   "
          f"({time.perf_counter()-t0:.1f} s)")
    print()
    print(f"  ESCOPO    {'WHOLE CONNECTOME SIMULATED' if args.cns == 'whole' else 'circuito do Giant Fiber'}")
    print("            NAO e um cerebro funcional completo: so a via de looming")
    print("            e a motora tem semantica modelada.")

    idx_gf = papeis.get("DNp01", np.zeros(0, np.int32))
    gf_pre, gf_peso = entradas_do_gf(eng.c, idx_gf)
    print(f"  gate      {len(gf_pre)} arestas entram no GF "
          f"({int((gf_peso > 0).sum())} exc, {int((gf_peso < 0).sum())} inib)")
    if args.info:
        corpo.fecha()
        return

    tel = abrir_telemetria(porta=args.porta, ativo=args.telemetry)
    _abertura(tel, protocol, eng, corpo, papeis, nomes_grupos, args, gf_peso)

    sens = papeis.get("LC4/LPLC2", np.zeros(0, np.int32))
    motor = papeis.get("TTMn", np.zeros(0, np.int32))
    taxas = np.zeros(eng.c.n, dtype=np.float64)
    rng = np.random.default_rng(0)
    prof = Profiler(["physics", "vision", "neural", "leitura", "telemetry"])

    escuro_lento = None
    dark_tau = 0.3 * VISION_HZ
    drive = np.array([BASE_DRIVE, BASE_DRIVE])
    escape_ate, escapes = -1.0, 0
    n_passos = int(args.duracao / DT)
    # Intervalo em TEMPO, nao em modulo de passo. A pose so pode ser enviada
    # dentro do ramo da retina (que roda a cada 100 passos), e um
    # `passo % 333 == 0` quase nunca cai nesses passos -- os dois so coincidem
    # a cada 33.300, ou seja 3,3 s de mosca. Com intervalo em tempo, sai no
    # primeiro quadro de retina depois de cada 1/POSE_HZ.
    intervalo_pose = 1.0 / POSE_HZ
    proxima_pose = 0.0
    ultimo_log = time.time()
    avisou_v = False

    print(f"\nrodando {args.duracao}s de mosca...")
    for passo in range(n_passos):
        t_s = passo * DT
        corpo.antes_do_passo(t_s)

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
            est_m = eng.le(motor) if len(motor) else None
            ttmn = int(est_m.spike.sum()) if est_m is not None else 0

        # eventos cientificos, nao ruido de baixo nivel
        eventos = []
        if hz > 1.0:
            eventos.append(("looming", {"hz": round(hz, 2)}))
        if gate["spikes_gf"]:
            eventos.append(("gf_spike", {"net_mV": gate["liquido_mV"]}))
        if ttmn:
            eventos.append(("ttmn_spike", {"n": ttmn}))
        if ttmn and t_s > escape_ate:
            escape_ate = t_s + ESCAPE_MS / 1000.0
            escapes += 1
            eventos.append(("escape", {"t_s": round(t_s, 3)}))
        if not avisou_v and gate["v_min_mV"] < V_AVISO_MV:
            avisou_v = True
            eventos.append(("model_limitation", {
                "v_mV": gate["v_min_mV"], "limiar_aviso": V_AVISO_MV,
                "id": "no_inhibitory_reversal"}))

        drive = (np.array([ESCAPE_DRIVE, ESCAPE_DRIVE]) if t_s < escape_ate
                 else np.array([BASE_DRIVE, BASE_DRIVE]))

        with prof("telemetry", "quadro"):
            if tel.ativo:
                pose = None
                if t_s >= proxima_pose:
                    pose = corpo.pose_corpo()
                    proxima_pose = t_s + intervalo_pose
                _quadro(tel, protocol, eng, prof, frame, drive, t_s, passo, hz,
                        papeis, nomes_grupos, ttmn, gate, eventos, pose)

        agora = time.time()
        if agora - ultimo_log > 2.0:
            ultimo_log = agora
            v = prof.valores()
            print(f"  t={t_s:5.2f}s  RTF {v['_total']['rtf']:.4f}  "
                  f"looming {hz:5.1f}Hz  GF exc {gate['exc_mV']:7.1f} "
                  f"inib {gate['inib_mV']:8.1f} liq {gate['liquido_mV']:8.1f}  "
                  f"v {gate['v_min_mV']:8.1f}  TTMn {ttmn}  fugas {escapes}")

    print("\n== profiler ==")
    print(prof.relatorio())
    print(f"\n  fugas: {escapes}")
    corpo.fecha()
    if tel.ativo:
        tel.enviar(protocol.bye("corrida terminada"))
        tel.fechar()


def _abertura(tel, protocol, eng, corpo, papeis, nomes_grupos, args, gf_peso):
    r = eng.resumo()
    rc = corpo.resumo()
    circuitos = [{"name": n, "role": n,
                  "body_ids": [int(b) for b in eng.c.body_ids[idx]],
                  "sides": [None] * len(idx), "types": [n] * len(idx),
                  "neurotransmitters": [None] * len(idx)}
                 for n, idx in papeis.items() if len(idx)]
    tel.enviar(protocol.experiment_info(
        experiment_id=f"lab_{args.physics}_{args.cns}",
        name=f"Drosobot Lab -- {args.experiment} / "
             f"{'Male CNS inteiro' if args.cns == 'whole' else 'circuito GF'}",
        description="FlyGym/MuJoCo -> retina -> CNS na GPU -> TTMn -> marcha.",
        parameters={"dt_physics_s": DT, "dt_neural_ms": eng.dt,
                    "vision_hz": VISION_HZ, "loom_gain": LOOM_GAIN,
                    "collision_set": args.colisao,
                    "collision_pairs": rc["collision_pairs"],
                    "entrada": "poisson_por_taxa", "v_aviso_mV": V_AVISO_MV},
        provenance={
            "body_ids": protocol.DATA, "synapse_weight": protocol.DATA,
            "neurotransmitter": protocol.DATA,
            "membrane_potential": protocol.MODEL, "spikes": protocol.MODEL,
            "synapse_sign": protocol.MODEL,
            "gf_excitation": protocol.MODEL, "gf_inhibition": protocol.MODEL,
            "loom_gain": protocol.ASSUMPTION,
            "dark_fraction_detector": protocol.ASSUMPTION,
            "motor_to_gait_mapping": protocol.ASSUMPTION,
            "collision_set": protocol.ASSUMPTION,
        },
        circuits=circuitos,
        runtime={
            "os": sys.platform,
            "physics_backend": rc["physics_backend"],
            "physics_adapter": rc["adapter"],
            "neural_backend": r["backend"], "neural_device": r["device"],
            "neurons_simulated": r["neurons_simulated"],
            "edges_simulated": r["edges_simulated"],
            "vram_mib": r.get("vram_mib", 0),
            "collision_set": args.colisao,
            "collision_pairs": rc["collision_pairs"],
            "population_names": nomes_grupos,
        },
        scope={
            "simulated": "whole_connectome" if args.cns == "whole" else "circuit",
            "functional_brain": "no",
            "note": ("conectoma inteiro simulado; NAO e um cerebro funcional "
                     "completo -- so a via de looming e a motora tem semantica "
                     "sensorial/motora modelada"),
        },
        model_limitations=[{
            "id": "no_inhibitory_reversal",
            "text": ("The Shiu et al. LIF formulation used here has no "
                     "inhibitory reversal potential. In whole-CNS simulations, "
                     "strong convergent inhibition can therefore drive membrane "
                     "potential to non-physiological negative values."),
            "warning_threshold_mV": V_AVISO_MV,
            "action": "reported, never clamped",
        }],
        gf_gate={"edges": int(len(gf_peso)),
                 "excitatory": int((gf_peso > 0).sum()),
                 "inhibitory": int((gf_peso < 0).sum()),
                 "peso_exc_mV": round(float(gf_peso[gf_peso > 0].sum()), 2),
                 "peso_inib_mV": round(float(gf_peso[gf_peso < 0].sum()), 2)},
    ))
    tel.enviar(protocol.scene_info(
        arena={"kind": args.experiment},
        stimulus=({"kind": "approaching_sphere", "radius": 3.0,
                   "start_distance": 30.0, "end_distance": 4.0, "cycle_s": 0.8}
                  if args.experiment == "looming"
                  else {"kind": "self_motion_flow"})))


def _quadro(tel, protocol, eng, prof, frame, drive, t_s, passo, hz, papeis,
            nomes_grupos, ttmn, gate, eventos, pose):
    v = prof.valores()
    extra = {}
    if pose is not None:
        nomes, pos, quat = pose
        # Pose a 30 Hz, nao a 10.000. A Unity pode interpolar entre quadros --
        # nada do que ela fizer volta pra fisica.
        extra["body_pose"] = {
            "segments": nomes,
            "pos": np.asarray(pos, dtype=np.float32).round(4).tolist(),
            "quat": np.asarray(quat, dtype=np.float32).round(5).tolist(),
        }
    tel.enviar(protocol.frame(
        step=passo, sim_time=t_s, wall_time=time.time(),
        real_time_factor=v["_total"]["rtf"], position=frame.posicao, drive=drive,
        profile={k: v[k]["ms_por_seg_simulado"]
                 for k in ("physics", "vision", "neural", "leitura", "telemetry")},
        **extra))

    camadas = []
    for nome, idx in papeis.items():
        if not len(idx):
            continue
        est = eng.le(idx)
        camadas.append({"name": nome, "spikes": est.spike.tolist(),
                        "v_mV": est.v_mV.tolist(), "g_mV": est.g_mV.tolist()})
    tel.enviar(protocol.neural_activity(t_s, camadas))

    soma = eng.atividade_por_grupo(zerar=True)
    tel.enviar(protocol.statistics(t_s, {
        "population_activity": {n: int(s) for n, s in zip(nomes_grupos, soma)},
        "looming_hz": hz, "ttmn_spikes": ttmn, "gf_gate": gate,
    }))
    for tipo, detalhe in eventos:
        tel.enviar(protocol.event(t_s, tipo, detalhe))


if __name__ == "__main__":
    main()
