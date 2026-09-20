"""
Classifica os 2220 pares de colisao do NeuroMechFly e mede quais de fato tocam.

    .venv\\Scripts\\python benchmarks\\physics\\collision_pruning\\audit_pairs.py

Contexto: 87% do passo de fisica e narrowphase de malha, e 98% dos pares sao
perna-contra-perna (ver docs/research/NEUROMECHFLY_PHYSICS_FEATURES.md). Antes de
podar qualquer coisa, precisamos saber o que cada par e e o que cada par faz.

## A regra que este script existe pra respeitar

Nao se desliga auto-colisao inteira. Um par so e candidato a poda se, numa
corrida longa e representativa, a distancia minima entre os dois geoms nunca
chegou perto de contato. "Nunca chegou perto" tem que ser medido, com margem, e
nao presumido a partir da anatomia.

Saida: JSON em benchmarks/physics/collision_pruning/ com, por par:
classe, distancia minima observada, se houve contato real, e a margem.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco

RAIZ = Path(__file__).resolve().parents[3]
SAIDA = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "sim"))

PERNAS = ("LF", "LM", "LH", "RF", "RM", "RH")
SEGMENTOS = ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2",
             "Tarsus3", "Tarsus4", "Tarsus5")
ORDEM_SEG = {s: i for i, s in enumerate(SEGMENTOS)}

DURACAO_S = 2.0          # 20.000 passos a 1e-4
AMOSTRA_A_CADA = 20      # medir distancia a cada 20 passos (2 ms de mosca)


def _nome(m, gid):
    return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
            or f"geom{gid}").replace("0/", "")


def classifica(a: str, b: str) -> str:
    """
    Classe de um par, pelos nomes dos dois geoms.

    As classes existem pra separar o que e estruturalmente impossivel do que e
    so improvavel. Segmento vizinho na MESMA perna, por exemplo, e ligado por
    junta: a colisao entre eles e limitada pelo limite da junta, nao pela fisica
    de contato -- candidato forte. Ja perna contralateral e improvavel numa
    caminhada reta, mas possivel numa curva fechada.
    """
    ra = re.match(r"^([LR][FMH])(.+)$", a)
    rb = re.match(r"^([LR][FMH])(.+)$", b)
    if not ra or not rb:
        outro = a if not ra else b
        if "ground" in outro.lower() or "floor" in outro.lower():
            return "perna_x_chao" if (ra or rb) else "chao_x_corpo"
        return "perna_x_corpo" if (ra or rb) else "corpo_x_corpo"

    perna_a, seg_a = ra.groups()
    perna_b, seg_b = rb.groups()

    if perna_a == perna_b:
        ia, ib = ORDEM_SEG.get(seg_a, 99), ORDEM_SEG.get(seg_b, 99)
        if abs(ia - ib) == 1:
            return "mesma_perna_adjacente"     # ligados por junta
        return "mesma_perna_distante"

    lado_a, pos_a = perna_a[0], perna_a[1]
    lado_b, pos_b = perna_b[0], perna_b[1]
    if lado_a != lado_b:
        return "contralateral"
    # mesmo lado, posicoes diferentes
    ordem = {"F": 0, "M": 1, "H": 2}
    if abs(ordem[pos_a] - ordem[pos_b]) == 1:
        return "ipsilateral_vizinha"
    return "ipsilateral_distante"


def monta():
    from flygym import Fly
    from flygym.examples.locomotion import HybridTurningController
    contatos = [f"{p}{s}" for p in PERNAS
                for s in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3",
                          "Tarsus4", "Tarsus5")]
    fly = Fly(enable_vision=False, spawn_pos=(0, 0, 0.3),
              contact_sensor_placements=contatos)
    sim = HybridTurningController(fly=fly, timestep=1e-4)
    sim.reset(seed=0)
    return sim


def main():
    sim = monta()
    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    npair = m.npair
    print(f"modelo: {npair} pares explicitos, {m.ngeom} geoms, nv={m.nv}")

    g1 = np.array([int(m.pair_geom1[k]) for k in range(npair)])
    g2 = np.array([int(m.pair_geom2[k]) for k in range(npair)])
    nomes = [( _nome(m, g1[k]), _nome(m, g2[k]) ) for k in range(npair)]
    classes = [classifica(a, b) for a, b in nomes]

    print("\n== classes ==")
    for c, n in Counter(classes).most_common():
        print(f"  {c:26s} {n:5d}  ({n/npair*100:5.1f}%)")

    # tamanho caracteristico de cada geom, pra dar sentido a "perto"
    raio = np.zeros(m.ngeom)
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[i]
            ini = m.mesh_vertadr[mid]
            n = m.mesh_vertnum[mid]
            vs = m.mesh_vert[ini:ini + n].reshape(-1, 3)
            raio[i] = float(np.linalg.norm(vs, axis=1).max())
        else:
            raio[i] = float(np.max(m.geom_size[i]))

    # ---- corrida: distancia minima por par, ao longo do tempo ----
    n_passos = int(DURACAO_S / 1e-4)
    dist_min = np.full(npair, np.inf)
    contatos_reais = np.zeros(npair, dtype=np.int64)
    drive = np.array([1.0, 1.0])

    print(f"\nrodando {DURACAO_S}s ({n_passos} passos), "
          f"amostrando distancia a cada {AMOSTRA_A_CADA}...")
    t0 = time.perf_counter()
    for passo in range(n_passos):
        sim.step(drive)

        # contatos que de fato aconteceram neste passo
        for ci in range(d.ncon):
            c = d.contact[ci]
            achou = np.flatnonzero(((g1 == c.geom1) & (g2 == c.geom2)) |
                                   ((g1 == c.geom2) & (g2 == c.geom1)))
            for k in achou:
                contatos_reais[k] += 1

        if passo % AMOSTRA_A_CADA:
            continue
        # distancia entre centros menos os dois raios: cota inferior barata da
        # distancia real entre superficies. Nao e exata -- e conservadora, que e
        # o que queremos: ela NUNCA superestima a folga.
        xp = d.geom_xpos
        dd = np.linalg.norm(xp[g1] - xp[g2], axis=1) - (raio[g1] + raio[g2])
        dist_min = np.minimum(dist_min, dd)

    dt = time.perf_counter() - t0
    print(f"  {dt:.1f}s de relogio ({dt/DURACAO_S:.1f}x mais lento que tempo real)")

    # ---- relatorio ----
    por_classe = defaultdict(lambda: {"n": 0, "tocaram": 0, "dists": []})
    linhas = []
    for k in range(npair):
        c = classes[k]
        por_classe[c]["n"] += 1
        por_classe[c]["dists"].append(float(dist_min[k]))
        if contatos_reais[k] > 0:
            por_classe[c]["tocaram"] += 1
        linhas.append({
            "idx": k, "geom1": nomes[k][0], "geom2": nomes[k][1],
            "classe": c,
            "dist_min_mm": round(float(dist_min[k]), 4),
            "passos_em_contato": int(contatos_reais[k]),
        })

    print("\n== por classe: quantos chegaram a tocar ==")
    print(f"  {'classe':26s} {'pares':>6s} {'tocaram':>8s} "
          f"{'folga min':>10s} {'mediana':>9s}")
    for c, v in sorted(por_classe.items(), key=lambda kv: -kv[1]["n"]):
        ds = np.array(v["dists"])
        print(f"  {c:26s} {v['n']:6d} {v['tocaram']:8d} "
              f"{ds.min():10.3f} {np.median(ds):9.3f}")

    relatorio = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "duracao_s": DURACAO_S,
        "n_passos": n_passos,
        "timestep": 1e-4,
        "npair": int(npair),
        "wall_s": round(dt, 2),
        "classes": {c: {"n": v["n"], "tocaram": v["tocaram"],
                        "folga_min_mm": round(float(np.min(v["dists"])), 4),
                        "folga_mediana_mm": round(float(np.median(v["dists"])), 4)}
                    for c, v in por_classe.items()},
        "pares": linhas,
    }
    SAIDA.mkdir(parents=True, exist_ok=True)
    destino = SAIDA / f"pair_audit_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    destino.write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
