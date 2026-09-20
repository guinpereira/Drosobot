r"""
Trajetoria: MuJoCo CPU contra Drosobot GPU Physics, passo a passo.

    .venv-flygym2\Scripts\python benchmarks\physics\gpu\trajetoria_gpu_vs_mujoco.py

Mesmo `mjModel`, mesmo `qpos`/`qvel`/`ctrl` inicial, mesmo timestep. Os dois
avancam em paralelo e cada passo e comparado -- nao so o estado final, que nao
distingue "divergiu no primeiro contato e se reencontrou" de "andou junto e se
separou no fim".

Grava dois `physics_trace.jsonl` (ver `lab/trace_fisico.py`) e reporta a
PRIMEIRA etapa e o PRIMEIRO campo que se separam.

## O que a divergencia significa aqui

Ate onde os dois solvers concordam, ela mede o porte. Depois disso, mede outra
coisa: **o solver do MuJoCo para antes do minimo**. Isso esta medido, nao
suposto -- o script avalia o MESMO objetivo nos dois pontos e imprime custo e
norma do gradiente de cada um. Quando o ponto da GPU tem custo menor E gradiente
menor, a diferenca nao e erro do porte.

O benchmark de latencia sai junto, porque so faz sentido medir o passo completo
quando ele existe: passo cheio da GPU contra `mj_step`, um mundo, mais a
separacao entre tempo de HOST (enfileirar) e tempo de GPU.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "sim"))

PASSOS = 1000
ASSENTA = 200
CADA = 1
# Passos em que a ESTACIONARIEDADE do solver e conferida. Conferir em todos
# custaria uma leitura da Jacobiana por passo, e o que interessa e se ela se
# degrada ao longo da corrida -- nao o valor em cada um.
MARCOS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)


def _cena(arena="looming"):
    from physics import cria
    from physics.adapter import MotorFrame

    corpo = cria("flygym2-mujoco", arena=arena, timestep=1e-4, com_visao=False)
    corpo.reset(seed=0)
    for _ in range(ASSENTA):
        corpo.passo(MotorFrame(drive=np.ones(2)))
    return corpo


def _contatos_mj(d):
    return [{"geom": (int(d.contact.geom1[i]), int(d.contact.geom2[i])),
             "dist": float(d.contact.dist[i]),
             "pos": np.asarray(d.contact.pos[i])}
            for i in range(int(d.ncon))]


def _contatos_gpu(con):
    return [{"geom": (int(con["geom"][i][0]), int(con["geom"][i][1])),
             "dist": float(con["dist"][i]), "pos": con["pos"][i]}
            for i in range(con["ncon"])]


def _grad_gpu(g, m) -> float:
    """
    |grad| do objetivo no ponto que o solver devolveu.

    `qacc` tem que ser o do SOLVER, nao o do Euler. O `mj_Euler` resolve uma
    aceleracao propria com `M + h*D` e a usa so para integrar; escreve-la por
    cima faria esta medida ler 1e+02 onde ha 1e-14.
    """
    nv, nefc = int(m.nv), int(g.le_int("nefc")[0])
    if nefc == 0:
        return 0.0
    J = g.le("efc_J")[:nefc*nv].reshape(nefc, nv)
    D = g.le("efc_D")[:nefc]
    aref = g.le("efc_aref")[:nefc]
    Md = g.le("Md").reshape(nv, nv)
    qs = g.le("qacc_smooth")
    a = g.le("qacc")
    neg = np.minimum(J @ a - aref, 0.0)
    return float(np.linalg.norm(Md @ (a - qs) + J.T @ (D * neg)))


def _qualidade(g, d, m):
    """Custo e |grad| do MESMO objetivo nos dois pontos. Quem esta no minimo?"""
    nv, nefc = int(m.nv), int(g.le_int("nefc")[0])
    if nefc == 0:
        return None
    J = g.le("efc_J")[:nefc*nv].reshape(nefc, nv)
    D = g.le("efc_D")[:nefc]
    aref = g.le("efc_aref")[:nefc]
    Md = g.le("Md").reshape(nv, nv)
    qs = g.le("qacc_smooth")

    def cg(a):
        jar = J @ a - aref
        neg = np.minimum(jar, 0.0)
        custo = 0.5*(a-qs) @ Md @ (a-qs) + 0.5*np.sum(D*neg*neg)
        grad = Md @ (a-qs) + J.T @ (D*neg)
        return float(custo), float(np.linalg.norm(grad))

    cg_gpu, gg = cg(g.le("qacc"))
    cg_mj, gm = cg(np.asarray(d.qacc))
    return {"custo_gpu": cg_gpu, "grad_gpu": gg,
            "custo_mujoco": cg_mj, "grad_mujoco": gm,
            "gpu_esta_melhor": bool(cg_gpu <= cg_mj and gg <= gm)}


def main() -> int:
    from gpu_physics.compilador import compila
    from gpu_physics.dinamica import MotorFisicoGPU
    from lab.trace_fisico import TraceFisico, primeira_divergencia

    corpo = _cena()
    m, d = corpo.sim.mj_model, corpo.sim.mj_data
    mj.mj_forward(m, d)
    mod = compila(m)

    q0 = np.array(d.qpos, copy=True)
    v0 = np.array(d.qvel, copy=True)
    c0 = np.array(d.ctrl, copy=True)

    g = MotorFisicoGPU(mod, fp64=True)
    g.escreve_estado(qpos=q0, qvel=v0, ctrl=c0,
                     mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
    d.qpos[:] = q0
    d.qvel[:] = v0
    d.ctrl[:] = c0

    saida = RAIZ / "benchmarks" / "physics" / "gpu" / "trace"
    t_mj = TraceFisico(saida / "mujoco", cada=CADA)
    t_gpu = TraceFisico(saida / "gpu", cada=CADA)

    # Qualidade da solucao no estado INICIAL, onde os dois veem exatamente o
    # mesmo problema. Medir depois de a trajetoria separar compararia dois
    # pontos de estados diferentes, e o numero nao diria nada.
    g.forward()
    mj.mj_forward(m, d)
    qualidade = _qualidade(g, d, m)
    if qualidade:
        qualidade["passo"] = 0
    g.escreve_estado(qpos=q0, qvel=v0, ctrl=c0,
                     mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
    d.qpos[:] = q0
    d.qvel[:] = v0
    d.ctrl[:] = c0

    por_passo = []
    for k in range(1, PASSOS + 1):
        g.passo()
        mj.mj_step(m, d)
        gq, gv = g.le("qpos"), g.le("qvel")
        con = g.contatos()
        t_gpu.linha(k, k*1e-4, qpos=gq, qvel=gv, qacc=g.le("qacc"),
                    contatos=_contatos_gpu(con),
                    qfrc_constraint=g.le("qfrc_constraint"))
        t_mj.linha(k, k*1e-4, qpos=d.qpos, qvel=d.qvel, qacc=d.qacc,
                   contatos=_contatos_mj(d),
                   qfrc_constraint=d.qfrc_constraint)
        eq = float(np.abs(gq - np.asarray(d.qpos)).max())
        ev = float(np.abs(gv - np.asarray(d.qvel)).max())
        reg = {"passo": k, "dqpos": eq, "dqvel": ev,
               "ncon_gpu": con["ncon"], "ncon_mj": int(d.ncon)}
        if k in MARCOS:
            reg["grad_gpu"] = _grad_gpu(g, m)
        por_passo.append(reg)
    t_mj.fecha()
    t_gpu.fecha()

    from lab.trace_fisico import le
    div = primeira_divergencia(le(saida / "gpu"), le(saida / "mujoco"),
                               tol=1e-9)

    # --- latencia: so agora faz sentido, porque o passo completo existe ---
    for _ in range(20):
        g.passo()
    N = 200
    t0 = time.perf_counter_ns()
    for _ in range(N):
        g.passo(esperar=False)
    us_host = (time.perf_counter_ns() - t0) / N / 1000.0
    g.dev.espera()
    t0 = time.perf_counter_ns()
    for _ in range(N):
        g.passo()
    us_gpu = (time.perf_counter_ns() - t0) / N / 1000.0
    t0 = time.perf_counter_ns()
    for _ in range(N):
        mj.mj_step(m, d)
    us_cpu = (time.perf_counter_ns() - t0) / N / 1000.0

    r = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "mujoco": mj.__version__,
        "passos": PASSOS,
        "hash_modelo": mod.hash_modelo,
        "por_passo": [x for x in por_passo if x["passo"] in MARCOS],
        "primeira_divergencia": div,
        "qualidade_no_estado_inicial": qualidade,
        "latencia_um_mundo": {
            "gpu_passo_completo_us": round(us_gpu, 1),
            "gpu_host_para_enfileirar_us": round(us_host, 1),
            "gpu_alem_do_enfileiramento_us": round(max(0.0, us_gpu - us_host), 1),
            "mujoco_cpu_mj_step_us": round(us_cpu, 1),
            "razao": round(us_gpu / us_cpu, 2),
        },
    }
    dest = RAIZ / "benchmarks" / "physics" / "gpu" / "trajetoria_gpu_vs_mujoco.json"
    dest.write_text(json.dumps(r, indent=1), encoding="utf-8")

    print(f"  passo   dqpos      dqvel      ncon   |grad| GPU")
    for x in por_passo:
        if x["passo"] not in MARCOS:
            continue
        print(f"  {x['passo']:>5}   {x['dqpos']:.3e}  {x['dqvel']:.3e}  "
              f"{x['ncon_gpu']:>2}/{x['ncon_mj']:<2}   {x.get('grad_gpu', 0):.2e}")
    print(f"\n  primeira divergencia: {div}")
    if qualidade:
        q = qualidade
        print(f"  estado inicial, MESMO objetivo:")
        print(f"    GPU     custo {q['custo_gpu']:.6e}  |grad| {q['grad_gpu']:.3e}")
        print(f"    MuJoCo  custo {q['custo_mujoco']:.6e}  |grad| {q['grad_mujoco']:.3e}")
        print(f"    GPU no ponto melhor: {q['gpu_esta_melhor']}")
    lat = r["latencia_um_mundo"]
    print(f"\n  latencia, um mundo:")
    print(f"    Drosobot GPU passo completo  {lat['gpu_passo_completo_us']:>9.1f} us")
    print(f"      host para enfileirar       {lat['gpu_host_para_enfileirar_us']:>9.1f} us")
    print(f"      alem do enfileiramento     {lat['gpu_alem_do_enfileiramento_us']:>9.1f} us")
    print(f"    MuJoCo CPU mj_step           {lat['mujoco_cpu_mj_step_us']:>9.1f} us")
    print(f"\n-> {dest.relative_to(RAIZ)}")
    corpo.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
