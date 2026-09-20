"""
Onde vai o tempo do passo de fisica, no experimento real de looming.

    .venv-flygym2\\Scripts\\python benchmarks\\physics\\profile_flygym2_step.py

A pergunta e uma so: **o custo esta dentro do `mj_step` ou em volta dele?** Se
estiver dentro, a saida e o modelo de colisao. Se estiver em volta, a saida e
overhead de Python/FlyGym e da pra atacar sem encostar na fisica.

Decompoe o passo do `FlyGym2Adapter` em:

    observacao      gather de posicoes e forcas de contato antes do controlador
    controlador     o HybridTurningController (CPG + ajustes), pelo
                    caminho que o adaptador de fato usa
    aplica_acao     escrever nos atuadores
    mj_step         a fisica de verdade, medida por dentro do MuJoCo
    retina          render dos omatideos (so nos passos em que atualiza)
    quadro          montar o SensorFrame
    estimulo        mover a esfera de looming (mocap)

`mj_step` e medido de duas formas, de proposito:

    parede          relogio em volta de `sim.step()`
    mjcontrol/...   o profiler interno do MuJoCo (`mj_data.timer`), que separa
                    colisao (`mjTIMER_POS_COLLISION`) do resto

A diferenca entre as duas e o que o wrapper do FlyGym cobra por cima do
`mj_step` -- e esse numero e o alvo desta rodada.

Nao muda nada: mesmo modelo, mesmo timestep, mesmo controlador, mesma arena.
"""
from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

SAIDA = RAIZ / "benchmarks" / "physics"
DT = 1e-4
PASSOS = 3000          # 300 ms de mosca: 3 quadros de retina, ciclo de marcha inteiro


class Cronometro:
    """Acumula ns por etapa. Um dicionario e um `with`, nada alem disso."""

    def __init__(self):
        self.ns: OrderedDict[str, int] = OrderedDict()
        self.n: OrderedDict[str, int] = OrderedDict()

    def marca(self, etapa: str, t0: int) -> None:
        dt = time.perf_counter_ns() - t0
        self.ns[etapa] = self.ns.get(etapa, 0) + dt
        self.n[etapa] = self.n.get(etapa, 0) + 1


def timers_mujoco(d) -> dict:
    """
    Timers internos do MuJoCo, em ms acumulados.

    Sao os mesmos contadores que o `mjVISUALIZE` mostra. `mj_data.timer` e um
    array de estruturas com `duration` (em microssegundos) e `number`.
    """
    import mujoco as mj

    nomes = {
        "step": mj.mjtTimer.mjTIMER_STEP,
        "forward": mj.mjtTimer.mjTIMER_FORWARD,
        "position": mj.mjtTimer.mjTIMER_POSITION,
        "collision": mj.mjtTimer.mjTIMER_POS_COLLISION,
        "col_broad": mj.mjtTimer.mjTIMER_COL_BROAD,
        "col_narrow": mj.mjtTimer.mjTIMER_COL_NARROW,
        "make_constraint": mj.mjtTimer.mjTIMER_POS_MAKE,
        "project_constraint": mj.mjtTimer.mjTIMER_POS_PROJECT,
        "kinematics": mj.mjtTimer.mjTIMER_POS_KINEMATICS,
        "inertia": mj.mjtTimer.mjTIMER_POS_INERTIA,
        "constraint_solve": mj.mjtTimer.mjTIMER_CONSTRAINT,
    }
    out = {}
    for rotulo, idx in nomes.items():
        t = d.timer[int(idx)]
        out[rotulo] = {"ms": float(t.duration) / 1000.0, "n": int(t.number)}
    return out


def zera_timers(d) -> None:
    """
    Nao ha `mj_resetTimer` no binding Python; os contadores sao zerados no
    lugar. Sem zerar, o aquecimento entra na conta e a colisao aparece inflada.
    """
    for t in d.timer:
        t.duration = 0.0
        t.number = 0


def main() -> None:
    import mujoco as mj
    from physics import MotorFrame, cria

    print("montando FlyGym 2.x, arena de looming...")
    corpo = cria("flygym2", arena="looming", self_collisions="legs",
                 timestep=DT, com_visao=True)
    corpo.reset(seed=0)
    m, d = corpo.sim.mj_model, corpo.sim.mj_data
    print(f"  {m.npair} pares de colisao declarados, ngeom={m.ngeom}, nv={m.nv}")

    from flygym.compose import ActuatorType
    from flygym_demo.complex_terrain import apply_locomotion_action

    c = Cronometro()
    drive = np.array([1.0, 1.0])
    zera_timers(d)

    # aquecimento: primeira chamada compila caminho quente e aloca buffers
    for _ in range(100):
        obs = corpo._observacao()
        acao = corpo._rapido.step(drive, obs)
        apply_locomotion_action(corpo.sim, corpo.fly.name, acao,
                                actuator_type=ActuatorType.POSITION)
        corpo.sim.step()
    zera_timers(d)

    print(f"rodando {PASSOS} passos ({PASSOS * DT * 1000:.0f} ms de mosca)...")
    t_total = time.perf_counter_ns()
    for passo in range(PASSOS):
        t_s = passo * DT

        t0 = time.perf_counter_ns()
        corpo.antes_do_passo(t_s)
        c.marca("estimulo", t0)

        t0 = time.perf_counter_ns()
        obs = corpo._observacao()
        c.marca("observacao", t0)

        t0 = time.perf_counter_ns()
        acao = corpo._rapido.step(drive, obs)
        c.marca("controlador", t0)

        t0 = time.perf_counter_ns()
        apply_locomotion_action(corpo.sim, corpo.fly.name, acao,
                                actuator_type=ActuatorType.POSITION)
        c.marca("aplica_acao", t0)

        t0 = time.perf_counter_ns()
        corpo.sim.step()
        c.marca("mj_step_parede", t0)

        corpo._passo += 1
        atualizou = corpo.com_visao and (
            corpo._passo % corpo._passos_por_retina == 0)

        t0 = time.perf_counter_ns()
        if atualizou:
            leituras = corpo.sim.get_ommatidia_readouts(corpo.fly.name)
            _ = np.asarray(leituras).mean(axis=2)
        c.marca("retina", t0)

    parede_ns = time.perf_counter_ns() - t_total
    timers = timers_mujoco(d)
    corpo.fecha()

    # ------------------------------------------------------------- relatorio
    sim_ms = PASSOS * DT * 1000.0
    def por_seg(ns):                      # ms de relogio por segundo simulado
        return ns / 1e6 / (sim_ms / 1000.0)

    print()
    print("  CUSTO POR SEGUNDO SIMULADO (ms de relogio)")
    print(f"  {'etapa':<18s} {'ms/s sim':>10s} {'%':>7s} {'us/passo':>10s}")
    print("  " + "-" * 50)
    medido = 0
    linhas = {}
    for etapa, ns in sorted(c.ns.items(), key=lambda kv: -kv[1]):
        medido += ns
        linhas[etapa] = {"ms_por_seg_simulado": round(por_seg(ns), 1),
                         "us_por_passo": round(ns / 1000 / c.n[etapa], 2),
                         "pct": round(100 * ns / parede_ns, 1)}
        print(f"  {etapa:<18s} {por_seg(ns):10.1f} {100*ns/parede_ns:6.1f}% "
              f"{ns/1000/c.n[etapa]:10.2f}")
    resto = parede_ns - medido
    print(f"  {'outro':<18s} {por_seg(resto):10.1f} {100*resto/parede_ns:6.1f}%")
    print("  " + "-" * 50)
    print(f"  {'parede':<18s} {por_seg(parede_ns):10.1f} {100.0:6.1f}%")

    print()
    print("  DENTRO DO mj_step (profiler interno do MuJoCo)")
    passo_ms = timers["step"]["ms"]
    print(f"  {'timer':<20s} {'ms total':>10s} {'%do step':>9s} {'us/passo':>10s}")
    print("  " + "-" * 52)
    for k, v in timers.items():
        pct = 100 * v["ms"] / passo_ms if passo_ms else 0
        us = v["ms"] * 1000 / v["n"] if v["n"] else 0
        print(f"  {k:<20s} {v['ms']:10.1f} {pct:8.1f}% {us:10.2f}")

    parede_step_ms = c.ns["mj_step_parede"] / 1e6
    overhead = parede_step_ms - passo_ms
    print()
    print(f"  sim.step() na parede   {parede_step_ms:8.1f} ms")
    print(f"  mjTIMER_STEP           {passo_ms:8.1f} ms")
    print(f"  overhead do wrapper    {overhead:8.1f} ms  "
          f"({100*overhead/parede_step_ms:.1f}% do sim.step)")

    fora = parede_ns / 1e6 - parede_step_ms
    print()
    print(f"  DENTRO do sim.step()   {parede_step_ms:8.1f} ms  "
          f"({100*parede_step_ms/(parede_ns/1e6):.1f}%)")
    print(f"  FORA do sim.step()     {fora:8.1f} ms  "
          f"({100*fora/(parede_ns/1e6):.1f}%)")

    SAIDA.mkdir(parents=True, exist_ok=True)
    caminho = SAIDA / "flygym2_step_profile.json"
    caminho.write_text(json.dumps({
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "passos": PASSOS, "dt_s": DT, "sim_ms": sim_ms,
        "parede_ms": round(parede_ns / 1e6, 1),
        "etapas": linhas,
        "mujoco_timers_ms": {k: round(v["ms"], 2) for k, v in timers.items()},
        "overhead_wrapper_ms": round(overhead, 1),
        "dentro_do_mj_step_pct": round(100 * passo_ms / (parede_ns / 1e6), 1),
    }, indent=1), encoding="utf-8")
    print(f"\n  salvo em {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
