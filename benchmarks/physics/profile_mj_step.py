r"""
Abrir a caixa-preta dos ~160 us/passo do `mj_step`.

    .venv-flygym2\Scripts\python benchmarks\physics\profile_mj_step.py

O perfil anterior (`profile_flygym2_step.py`) mediu o passo por fora e viu
`mjcontrol/...` tudo zerado. O motivo nao e que o MuJoCo nao meca: e que os
timers internos dependem de um CALLBACK de relogio (`mjcb_time`) que, por
padrao, nao existe. Sem ele todo `TM_START/TM_END` vira no-op.

Aqui o callback e instalado (`mujoco.set_mjcb_time`) e o passo e decomposto
pelos timers oficiais do proprio engine:

    mjTIMER_POSITION          cinematica, inercia, colisao, restricoes
      mjTIMER_POS_KINEMATICS    FK + COM + tendao + transmissao
      mjTIMER_POS_INERTIA       massa, CRB, fatoracao de M
      mjTIMER_POS_COLLISION     broadphase + narrowphase
      mjTIMER_POS_MAKE          montagem da Jacobiana das restricoes
      mjTIMER_POS_PROJECT       projecao / A = J M^-1 J^T
    mjTIMER_VELOCITY          forcas dependentes de velocidade
    mjTIMER_ACTUATION         atuadores
    mjTIMER_CONSTRAINT        solver
    mjTIMER_ADVANCE           integracao

Roda sobre o MESMO modelo dos experimentos (arena de looming e de obstaculos),
com o controlador de marcha real acionando, porque perfilar a mosca parada no
ar mediria outra coisa: sem contato nao ha restricao, e o solver e a metade do
problema.

O callback custa uma chamada Python por marcador, entao a soma dos timers NAO
e comparavel ao relogio de parede desta mesma corrida. Por isso o tempo total
sai de uma segunda passada, SEM callback. Os timers dao a PROPORCAO; a passada
limpa da o valor absoluto.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "sim"))

PASSOS = 2000
AQUECE = 300

ETAPAS = [
    ("mjTIMER_POS_KINEMATICS", "cinematica"),
    ("mjTIMER_POS_INERTIA", "inercia"),
    ("mjTIMER_POS_COLLISION", "colisao"),
    ("mjTIMER_POS_MAKE", "make_constraint"),
    ("mjTIMER_POS_PROJECT", "project_constraint"),
    ("mjTIMER_VELOCITY", "velocidade"),
    ("mjTIMER_ACTUATION", "atuacao"),
    ("mjTIMER_CONSTRAINT", "solver"),
    ("mjTIMER_ADVANCE", "integracao"),
]


def _adaptador(arena: str, estimulo: dict):
    from physics.adapter import MotorFrame
    from physics.flygym2 import FlyGym2Adapter

    ad = FlyGym2Adapter(arena=arena, estimulo=estimulo, com_visao=False)
    ad.reset(seed=0)
    return ad, MotorFrame(drive=np.ones(2))


class _CronometraStep:
    """
    Mede `sim.step()` DENTRO do laco real, sem tirar o controlador do caminho.

    A alternativa obvia -- rodar `mj_step` num laco apertado depois de
    aquecer -- mede outra fisica: sem o controlador reescrevendo `ctrl` a cada
    passo, os atuadores congelam, a mosca desaba no chao, `ncon` sobe e o
    solver passa a trabalhar mais. Na pratica isso inflou a medida de 94 us
    para 149 us sem que nada no modelo tivesse mudado.
    """

    def __init__(self, sim):
        self._sim = sim
        self._orig = sim.step
        self.ns = 0
        self.n = 0
        sim.step = self._step

    def _step(self, *a, **kw):
        t0 = time.perf_counter_ns()
        r = self._orig(*a, **kw)
        self.ns += time.perf_counter_ns() - t0
        self.n += 1
        return r

    def solta(self) -> float:
        """Devolve us por `mj_step` e restaura o metodo original."""
        self._sim.step = self._orig
        return (self.ns / self.n / 1000.0) if self.n else 0.0


def _avanca(ad, motor, n: int) -> None:
    """Passos completos do experimento: controlador + atuadores + mj_step."""
    for _ in range(n):
        ad.passo(motor)


def mede(arena: str, estimulo: dict) -> dict:
    ad, motor = _adaptador(arena, estimulo)
    m, d = ad.sim.mj_model, ad.sim.mj_data
    _avanca(ad, motor, AQUECE)

    # --- passada limpa: quanto custa o passo de verdade -------------------
    # `sim.step()` e cronometrado POR DENTRO do laco real. Um laco apartado de
    # `mj_step` mediria a mosca desabada, com mais contato e mais solver.
    mj.set_mjcb_time(None)
    crono = _CronometraStep(ad.sim)
    t0 = time.perf_counter_ns()
    _avanca(ad, motor, PASSOS)
    parede_ns = time.perf_counter_ns() - t0
    us_passo_mj = crono.solta()

    # --- passada instrumentada: onde o tempo se distribui -----------------
    ad, motor = _adaptador(arena, estimulo)
    m, d = ad.sim.mj_model, ad.sim.mj_data
    _avanca(ad, motor, AQUECE)
    # nao ha mj_resetTimer no binding: os acumuladores sao lidos antes e
    # depois, e a diferenca e o que esta passada gastou.
    mj.set_mjcb_time(time.perf_counter_ns)
    antes = [float(t.duration) for t in d.timer]
    _avanca(ad, motor, PASSOS)
    depois = [float(t.duration) for t in d.timer]
    mj.set_mjcb_time(None)

    def _delta(enum_nome: str) -> float:
        i = int(getattr(mj.mjtTimer, enum_nome))
        return depois[i] - antes[i]

    bruto = {rotulo: _delta(enum_nome) for enum_nome, rotulo in ETAPAS}
    passo_total = _delta("mjTIMER_STEP")

    # POSITION engloba os cinco primeiros; o que sobra dele e "outros"
    pos_total = _delta("mjTIMER_POSITION")
    cobertos = sum(bruto[r] for _, r in ETAPAS[:5])
    bruto["posicao_outros"] = max(0.0, pos_total - cobertos)
    soma = sum(bruto.values())
    resto = max(0.0, passo_total - soma)
    if resto > 0:
        bruto["resto_do_passo"] = resto
        soma += resto

    etapas = {
        r: {
            "pct": round(100.0 * v / soma, 2) if soma else 0.0,
            "us_por_passo": round(us_passo_mj * v / soma, 3) if soma else 0.0,
        }
        for r, v in sorted(bruto.items(), key=lambda kv: -kv[1])
    }

    ncon = int(d.ncon)
    nefc = int(d.nefc)
    return {
        "arena": arena,
        "passos": PASSOS,
        "nv": int(m.nv), "nu": int(m.nu), "npair": int(m.npair),
        "ncon_final": ncon, "nefc_final": nefc,
        "us_por_passo_mj_step": round(us_passo_mj, 2),
        "us_por_passo_ciclo_completo": round(parede_ns / PASSOS / 1000.0, 2),
        "etapas": etapas,
        "nota_timers": ("proporcao medida com mjcb_time instalado; o absoluto "
                        "vem da passada limpa, sem callback"),
    }


def main() -> int:
    saida = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "mujoco": mj.__version__,
        "arenas": {},
    }
    for arena, est in (("looming", {}), ("obstaculos", {"lado": "centro"})):
        r = mede(arena, est)
        saida["arenas"][arena] = r
        print(f"\n=== {arena}: {r['us_por_passo_mj_step']} us/mj_step "
              f"({r['us_por_passo_ciclo_completo']} us/ciclo), "
              f"ncon={r['ncon_final']} nefc={r['nefc_final']}")
        for nome, v in r["etapas"].items():
            print(f"  {nome:<22} {v['pct']:>6.2f}%  {v['us_por_passo']:>8.2f} us")

    dest = RAIZ / "benchmarks" / "physics" / "mj_step_profile.json"
    dest.write_text(json.dumps(saida, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
