r"""
Por que MuJoCo e GPU discordam quando a adesao esta ligada.

    .venv-flygym2\Scripts\python benchmarks\physics\gpu\solver_adesao.py

Com adesao DESLIGADA os dois solvers concordam a 1e-14 e a trajetoria coincide
por dois passos em precisao de maquina. Com ela LIGADA, `qacc` difere em 27%.
Este script mostra que a causa nao e o porte.

## O que ele estabelece, em tres medidas

  1. **O problema e o mesmo.** Toda entrada do solver -- `efc_J`, `efc_R`,
     `efc_D`, `efc_aref`, `qacc_smooth` -- bate entre GPU e MuJoCo com adesao
     ligada, nas mesmas ordens de grandeza de quando ela esta desligada. Se as
     entradas batem e as saidas nao, a diferenca esta no solver.

  2. **O MuJoCo nao esta parando por tolerancia.** Rodado com `tolerance` de
     1e-8 ate 1e-15 e `iterations` de 100 ate 5000, ele para no MESMO ponto,
     depois do MESMO numero de iteracoes. Um criterio de parada por tolerancia
     responderia a mudanca de tolerancia; este nao responde.

  3. **O ponto em que ele para nao e estacionario.** Avaliando o gradiente do
     objetivo comum nos dois pontos: o da GPU tem norma ~1e-13 e custo menor; o
     do MuJoCo tem norma ~5e1 e custo maior.

## A conclusao, e o que ela NAO e

Sobram duas saidas no laco do `mj_solveNewton` de 3.9.0
(`src/engine/engine_solver.c`):

    2018:  if (alpha == 0) break;                      // busca de linha
    2058:  if (improvement < tol || gradient < tol)     // tolerancia

A segunda esta descartada pela medida 2. Resta `alpha == 0`: a busca de linha
exata por partes nao encontra passo que melhore, e o solver para onde esta.

Isso NAO quer dizer que o MuJoCo esteja com defeito. A adesao torna o problema
muito mal condicionado -- uma diferenca de 0,4% na forca de restricao vira 27%
em `qacc`, porque a forca quase cancela uma aceleracao livre enorme. Numa
regiao assim, uma busca de linha exata sobre um custo quase plano pode
legitimamente concluir que nao ha progresso.

Tambem NAO quer dizer que a GPU esteja certa e o MuJoCo errado. Quer dizer que
**"resolver o mesmo problema ate estacionariedade" e "reproduzir a trajetoria
numerica do MuJoCo 3.9.0" sao objetivos diferentes neste regime**, e que a
escolha entre eles e uma decisao a ser tomada com o olho aberto. Ver
`docs/GPU_PHYSICS.md`.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "sim"))

ASSENTA = 200
CONFIGS = ((1e-8, 100), (1e-12, 1000), (1e-15, 5000))


def _densa(d, m):
    """`efc_J` do MuJoCo: esparsa quando nv >= 60."""
    nefc, nv = int(d.nefc), int(m.nv)
    J = np.zeros((nefc, nv))
    rn = np.asarray(d.efc_J_rownnz)[:nefc]
    ra = np.asarray(d.efc_J_rowadr)[:nefc]
    ci = np.asarray(d.efc_J_colind)
    vals = np.asarray(d.efc_J)
    for i in range(nefc):
        a0, n = int(ra[i]), int(rn[i])
        J[i, ci[a0:a0 + n]] = vals[a0:a0 + n]
    return J


def main() -> int:
    from gpu_physics.compilador import compila
    from gpu_physics.dinamica import MotorFisicoGPU
    from physics import cria
    from physics.adapter import MotorFrame

    corpo = cria("flygym2-mujoco", arena="looming", timestep=1e-4,
                 com_visao=False)
    corpo.reset(seed=0)
    for _ in range(ASSENTA):
        corpo.passo(MotorFrame(drive=np.ones(2)))
    m, d = corpo.sim.mj_model, corpo.sim.mj_data
    g = MotorFisicoGPU(compila(m), fp64=True)
    ades = [a for a in range(m.nu) if int(m.actuator_trntype[a]) == 5]

    saida = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
             "mujoco": mj.__version__, "casos": {}}

    for rotulo, valor in (("adesao_off", 0.0), ("adesao_on", 1.0)):
        for a in ades:
            d.ctrl[a] = valor
        mj.mj_forward(m, d)
        g.escreve_estado(qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
                         mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
        g.forward()

        nv, nefc = int(m.nv), int(d.nefc)
        J = g.le("efc_J")[:nefc*nv].reshape(nefc, nv)
        D = g.le("efc_D")[:nefc]
        aref = g.le("efc_aref")[:nefc]
        Md = g.le("Md").reshape(nv, nv)
        qs = g.le("qacc_smooth")
        qgpu = g.le("qacc")

        def custo_grad(a):
            neg = np.minimum(J @ a - aref, 0.0)
            return (float(0.5*(a-qs) @ Md @ (a-qs) + 0.5*np.sum(D*neg*neg)),
                    float(np.linalg.norm(Md @ (a-qs) + J.T @ (D*neg))))

        # 1. as entradas do solver batem?
        entradas = {}
        for nome, got, ref in (("efc_J", J, _densa(d, m)),
                               ("efc_R", g.le("efc_R")[:nefc], d.efc_R),
                               ("efc_D", D, d.efc_D),
                               ("efc_aref", aref, d.efc_aref),
                               ("qacc_smooth", qs, d.qacc_smooth)):
            ref = np.asarray(ref, dtype=float).reshape(np.shape(got))
            esc = max(1e-30, float(np.abs(ref).max()))
            entradas[nome] = float(f"{np.abs(got-ref).max()/esc:.3e}")

        # 2. a parada responde a tolerancia?
        tol0, it0 = float(m.opt.tolerance), int(m.opt.iterations)
        varredura = []
        for tol, it in CONFIGS:
            m.opt.tolerance, m.opt.iterations = tol, it
            mj.mj_forward(m, d)
            q = np.asarray(d.qacc)
            c_, gr = custo_grad(q)
            varredura.append({
                "tolerance": tol, "iterations": it,
                "niter": int(d.solver_niter[0]),
                "custo": c_, "grad": gr,
                "rel_vs_gpu": float(f"{np.abs(q-qgpu).max()/max(1e-30, np.abs(qgpu).max()):.3e}"),
            })
        m.opt.tolerance, m.opt.iterations = tol0, it0
        mj.mj_forward(m, d)

        # 3. quem esta no minimo?
        c_gpu, gr_gpu = custo_grad(qgpu)
        c_mj, gr_mj = custo_grad(np.asarray(d.qacc))

        # O que importa nao e o numero de iteracoes -- ele varia em um -- e sim
        # se a SOLUCAO responde a tolerancia. Se o custo final e o mesmo de
        # 1e-8 a 1e-15, a parada nao e por tolerancia.
        custos = {float(f"{v['custo']:.10e}") for v in varredura}
        estavel = len(custos) == 1
        saida["casos"][rotulo] = {
            "nefc": nefc,
            "entradas_rel": entradas,
            "varredura_tolerancia": varredura,
            "solucao_insensivel_a_tolerancia": estavel,
            "gpu": {"custo": c_gpu, "grad": gr_gpu},
            "mujoco": {"custo": c_mj, "grad": gr_mj},
            "conclusao": (
                "MuJoCo para no mesmo ponto para toda tolerancia testada e o "
                "ponto nao e estacionario: a saida e `alpha == 0` na busca de "
                "linha (engine_solver.c:2018), nao o criterio de tolerancia "
                "(:2058)."
                if estavel and gr_mj > 1e-6 else
                "os dois concordam dentro da tolerancia: nao ha fenomeno a "
                "explicar neste estado."),
        }

        print(f"\n=== {rotulo}  (nefc {nefc})")
        print(f"  entradas do solver, erro relativo: {entradas}")
        print(f"  {'tol':>8} {'iter':>6} {'niter':>6} {'custo':>16} "
              f"{'|grad|':>10} {'rel vs GPU':>11}")
        for v in varredura:
            print(f"  {v['tolerance']:>8.0e} {v['iterations']:>6} "
                  f"{v['niter']:>6} {v['custo']:>16.8e} {v['grad']:>10.2e} "
                  f"{v['rel_vs_gpu']:>11.3e}")
        print(f"  {'GPU':>8} {'-':>6} {'-':>6} {c_gpu:>16.8e} {gr_gpu:>10.2e}")
        print(f"  -> {saida['casos'][rotulo]['conclusao']}")

    dest = RAIZ / "benchmarks" / "physics" / "gpu" / "solver_adesao.json"
    dest.write_text(json.dumps(saida, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    corpo.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
