"""
Emissor de telemetria falsa, pra desenvolver o visualizador sem rodar MuJoCo.

    python -m sim.telemetry.demo

Um segundo de mosca custa ~26 s de relogio na simulacao de verdade, o que
inviabiliza iterar em interface. Este modo produz o MESMO protocolo em tempo
real, com um episodio de looming plausivel: a esfera se aproxima, LC4/LPLC2
sobem, o Giant Fiber dispara, o TTMn dispara, a mosca recua.

Os NUMEROS aqui sao inventados -- e so isso. A forma das mensagens, os nomes das
camadas e a procedencia dos campos sao os mesmos da simulacao real, entao o que
funcionar contra o demo funciona contra a simulacao.

Se os CSVs do conectoma existirem, usa os bodyIds REAIS, e ai o cerebro na Unity
acende os neuronios certos. Sem eles, cai pra ids sinteticos e avisa.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetry import protocol  # noqa: E402
from telemetry.server import abrir  # noqa: E402


def _circuito_real():
    """Tenta montar os grupos com bodyId de verdade. Devolve None se nao der."""
    try:
        from connectome_model import (CONNECTOME, load_properties, side_of,
                                      gf_input_population)
        import pandas as pd
        props = load_properties()
        up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
        down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
        sensor_ids, is_loom, _ = gf_input_population(props, up)
        loom_ids = [b for b, f in zip(sensor_ids, is_loom) if f]
        gf_ids = [10001, 10010]
        ttmn_ids = down[down["type"] == "TTMn"]["bodyId_post"].unique().tolist()

        def descreve(nome, ids, papel):
            return {
                "name": nome,
                "role": papel,
                "body_ids": [int(b) for b in ids],
                "sides": [side_of(props, b) for b in ids],
                "types": [str(props.at[b, "type"]) if b in props.index else None for b in ids],
                "neurotransmitters": [str(props.at[b, "consensusNt"]) if b in props.index else None
                                      for b in ids],
            }

        return [descreve("LC4/LPLC2", loom_ids, "gf_sensor_looming"),
                descreve("DNp01", gf_ids, "gf_dnp01_giantfiber"),
                descreve("TTMn", ttmn_ids, "gf_ttmn_motor")], True
    except Exception as e:
        print(f"[demo] sem CSVs do conectoma ({type(e).__name__}); usando ids sinteticos")
        falso = lambda nome, n, papel, base: {
            "name": nome, "role": papel,
            "body_ids": [base + i for i in range(n)],
            "sides": ["L" if i % 2 else "R" for i in range(n)],
            "types": [None] * n, "neurotransmitters": [None] * n,
        }
        return [falso("LC4/LPLC2", 24, "gf_sensor_looming", 900000),
                falso("DNp01", 2, "gf_dnp01_giantfiber", 910000),
                falso("TTMn", 2, "gf_ttmn_motor", 920000)], False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--duracao", type=float, default=60.0, help="segundos de relogio")
    ap.add_argument("--ciclo", type=float, default=4.0, help="segundos por aproximacao")
    args = ap.parse_args()

    circuitos, reais = _circuito_real()
    n_loom = len(circuitos[0]["body_ids"])
    n_gf = len(circuitos[1]["body_ids"])
    n_ttmn = len(circuitos[2]["body_ids"])

    tel = abrir(porta=args.porta, ativo=True, fonte="drosobot-demo")
    print("[demo] dados SINTETICOS -- a forma do protocolo e real, os numeros nao")
    print(f"[demo] bodyIds {'reais do conectoma' if reais else 'sinteticos'}")
    print("[demo] ctrl-c pra sair")

    tel.enviar(protocol.experiment_info(
        experiment_id="demo_looming",
        name="Demo: looming (dados sinteticos)",
        description="Emissor de teste. Nao roda MuJoCo nem o circuito; serve pra "
                    "desenvolver o visualizador sem esperar a simulacao.",
        parameters={"cycle_s": args.ciclo, "synthetic": True},
        provenance={
            "body_ids": protocol.DATA if reais else protocol.ASSUMPTION,
            "neurotransmitter": protocol.DATA if reais else protocol.ASSUMPTION,
            "side": protocol.DATA if reais else protocol.ASSUMPTION,
            "spikes": protocol.ASSUMPTION,      # sinteticos neste modo
            "membrane_potential": protocol.ASSUMPTION,
            "input_hz": protocol.ASSUMPTION,
        },
        circuits=circuitos,
    ))
    tel.enviar(protocol.scene_info(
        arena={"kind": "flat", "size": [100, 100]},
        stimulus={"kind": "approaching_sphere", "radius": 3.0,
                  "start_distance": 30.0, "end_distance": 4.0},
    ))

    rng = np.random.default_rng(0)
    t0 = time.time()
    passo = 0
    pos = np.array([0.0, 0.0, 1.0])
    escape_ate = -1.0
    ultimo_evento = None

    try:
        while time.time() - t0 < args.duracao:
            agora = time.time()
            t = agora - t0
            fase = (t % args.ciclo) / args.ciclo
            dist = 30.0 + (4.0 - 30.0) * fase

            # quanto mais perto, mais forte -- e nao linear, como looming de verdade
            forca = float(np.clip((30.0 - dist) / 26.0, 0, 1)) ** 2.5
            hz_L = 20.0 * forca * (1.0 + 0.15 * rng.standard_normal())
            hz_R = 20.0 * forca * (1.0 + 0.15 * rng.standard_normal())
            hz_L, hz_R = max(0.0, hz_L), max(0.0, hz_R)

            spikes_loom = (rng.random(n_loom) < forca * 0.35).astype(int)
            spikes_gf = (rng.random(n_gf) < max(0.0, forca - 0.45) * 0.9).astype(int)
            spikes_ttmn = (rng.random(n_ttmn) < max(0.0, forca - 0.6) * 0.8).astype(int)
            spikes_ttmn[1:] = 0 if n_ttmn > 1 else spikes_ttmn[1:]   # so o lado direito, como no dado

            v_gf = -52.0 + 7.0 * min(1.0, forca * 1.3) + rng.standard_normal(n_gf) * 0.4

            if spikes_ttmn.sum() and t > escape_ate:
                escape_ate = t + 0.12
                tel.enviar(protocol.event(t, "escape_triggered",
                                          {"distance_mm": round(dist, 2)}))
                ultimo_evento = "escape"
            if forca > 0.25 and ultimo_evento != "looming" and t < escape_ate - 5:
                ultimo_evento = "looming"

            em_fuga = t < escape_ate
            drive = [-0.5, -0.5] if em_fuga else [1.0, 1.0]
            pos[0] += (-0.4 if em_fuga else 0.22) * 0.05
            pos[1] += 0.01 * math.sin(t * 1.7)

            tel.enviar(protocol.frame(
                step=passo, sim_time=t, wall_time=agora, real_time_factor=1.0,
                position=pos, orientation=[0, 0, 0, 1], drive=drive))

            tel.enviar(protocol.neural_activity(t, [
                {"name": "LC4/LPLC2", "spikes": spikes_loom.tolist(),
                 "rate_hz": [round(hz_L if i % 2 else hz_R, 2) for i in range(n_loom)]},
                {"name": "DNp01", "spikes": spikes_gf.tolist(),
                 "v_mV": [round(float(x), 3) for x in v_gf]},
                {"name": "TTMn", "spikes": spikes_ttmn.tolist()},
            ]))

            if passo % 3 == 0:     # retina mais espacada: e a mensagem mais cara
                base = 0.5 - 0.22 * forca
                ret_L = np.clip(base + rng.standard_normal(721) * 0.02, 0, 1)
                ret_R = np.clip(base + rng.standard_normal(721) * 0.02, 0, 1)
                tel.enviar_se_conectado(
                    protocol.retina, t, ret_L, ret_R,
                    {"looming": {"L": round(hz_L, 2), "R": round(hz_R, 2)},
                     "dark_fraction": {"L": round(float(1 - ret_L.mean()), 4),
                                       "R": round(float(1 - ret_R.mean()), 4)},
                     "input_hz": {"L": round(hz_L, 2), "R": round(hz_R, 2)}})

            if passo % 20 == 0:
                tel.enviar(protocol.statistics(t, {
                    "distance_mm": round(dist, 2),
                    "escapes": int(t // args.ciclo),
                    "sim_time": round(t, 2),
                }))

            passo += 1
            time.sleep(0.033)      # ~30 Hz
    except KeyboardInterrupt:
        print("\n[demo] interrompido")
    finally:
        tel.fechar()


if __name__ == "__main__":
    main()
