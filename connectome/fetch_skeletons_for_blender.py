"""
Puxa a morfologia 3D REAL (esqueleto, nao so grafo de conectividade) dos
neuronios dos nossos dois circuitos, mais a malha do cerebro+VNC do Male CNS
(via navis-flybrains), pra render bonito no Blender depois.

Sai tudo em blender/skeletons/*.swc e blender/mesh_brain.obj + mesh_vnc.obj.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuprint import Client
import navis
import navis.interfaces.neuprint as neu
import flybrains

ROOT = Path(__file__).parent.parent
CONNECTOME = ROOT / "connectome"
OUT = ROOT / "blender" / "skeletons"
OUT.mkdir(parents=True, exist_ok=True)

token = (ROOT / ".neuprint_token").read_text().strip()
_client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)  # precisa manter referencia viva, senao vira weakref morta e "no default client"

# ---------- reconstroi os mesmos ids usados nas duas simulacoes ----------
gf_up = pd.read_csv(CONNECTOME / "gf_upstream_connections.csv")
gf_down = pd.read_csv(CONNECTOME / "gf_downstream_connections.csv")
GF_IDS = [10001, 10010]
gf_sensor_ids = gf_up.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False).head(8).index.tolist()
gf_motor_ids = gf_down[gf_down["type"] == "TTMn"]["bodyId_post"].unique().tolist()

# ---------- LC4/LPLC2: subconjunto REPRESENTATIVO, nao a populacao toda ----------
#
# A simulacao usa os 311 LC4/LPLC2 pre-sinapticos do Giant Fiber. Exportar a
# morfologia dos 311 inflaria o asset e nao e necessario pra entender o circuito.
# Exportamos um subconjunto e registramos exatamente quanto ele cobre -- quem
# olhar a interface tem que saber que ve uma amostra, nao a populacao.
#
# A escolha e DETERMINISTICA e tem tres regras, nesta ordem:
#   1. peso sinaptico pro GF (contribuicao real, nao "40 quaisquer")
#   2. metade de cada hemisferio, pra nao criar assimetria que o dado nao tem
#   3. empate resolvido por bodyId, pra a selecao ser reproduzivel
#
# Pra exportar tudo no futuro: --top-by-weight 0 (sem limite). A arquitetura nao
# muda; so o numero.
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--top-by-weight", type=int, default=40,
                help="quantos LC4/LPLC2 exportar (0 = todos os 311)")
ap.add_argument("--group", default=None,
                help="exporta so este grupo (padrao: todos)")
args = ap.parse_args()

props_csv = CONNECTOME / "neuron_properties.csv"
props = pd.read_csv(props_csv).set_index("bodyId") if props_csv.exists() else None

LOOMING_TYPES = {"LC4", "LPLC2"}
peso_por_id = gf_up.groupby("bodyId_pre")["weight"].sum()


def _tipo(b):
    return str(props.at[b, "type"]) if props is not None and b in props.index else None


def _lado(b):
    return str(props.at[b, "somaSide"]) if props is not None and b in props.index else None


looming_todos = [b for b in peso_por_id.index if _tipo(b) in LOOMING_TYPES]
peso_total_looming = int(peso_por_id[looming_todos].sum())

if args.top_by_weight and args.top_by_weight > 0:
    # metade por hemisferio, cada metade ordenada por peso decrescente e bodyId
    por_lado = {"L": [], "R": [], None: []}
    for b in looming_todos:
        por_lado.setdefault(_lado(b), []).append(b)
    cota = max(1, args.top_by_weight // 2)
    looming_ids = []
    for lado in ("L", "R"):
        candidatos = sorted(por_lado.get(lado, []),
                            key=lambda b: (-int(peso_por_id[b]), int(b)))
        looming_ids.extend(candidatos[:cota])
    # se um lado tiver menos que a cota, completa pelo peso global
    if len(looming_ids) < args.top_by_weight:
        resto = sorted((b for b in looming_todos if b not in looming_ids),
                       key=lambda b: (-int(peso_por_id[b]), int(b)))
        looming_ids.extend(resto[:args.top_by_weight - len(looming_ids)])
else:
    looming_ids = sorted(looming_todos, key=lambda b: (-int(peso_por_id[b]), int(b)))

peso_coberto = int(peso_por_id[looming_ids].sum()) if looming_ids else 0
COBERTURA_LOOMING = {
    "group": "gf_sensor_looming",
    "types": sorted(LOOMING_TYPES),
    "total_simulated": len(looming_todos),
    "total_visualized": len(looming_ids),
    "fraction_visualized": round(len(looming_ids) / max(1, len(looming_todos)), 4),
    "total_synaptic_weight": peso_total_looming,
    "synaptic_weight_covered": peso_coberto,
    "fraction_weight_covered": round(peso_coberto / max(1, peso_total_looming), 4),
    "selection": "top por peso sinaptico pro Giant Fiber, metade por hemisferio, "
                 "empate por bodyId (deterministico)",
    "sides_visualized": {lado: sum(1 for b in looming_ids if _lado(b) == lado)
                         for lado in ("L", "R")},
    # A populacao INTEIRA, nao so a amostra. Sem isto o visualizador nao consegue
    # distinguir "simulado mas sem morfologia exportada" de "bodyId que eu nunca
    # vi" -- e o primeiro e escolha nossa, o segundo e sinal de descompasso.
    "population_body_ids": [int(b) for b in sorted(looming_todos)],
}
print(f"LC4/LPLC2: {len(looming_todos)} simulados -> {len(looming_ids)} exportados "
      f"({COBERTURA_LOOMING['fraction_visualized']:.0%} dos neuronios, "
      f"{COBERTURA_LOOMING['fraction_weight_covered']:.0%} do peso sinaptico) "
      f"| lados {COBERTURA_LOOMING['sides_visualized']}")

opto_sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
opto_hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
opto_dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")
om_sensor_ids = sorted(opto_sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(opto_sensor_hs["bodyId_post"]) | set(opto_hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(opto_hs_dna02["bodyId_post"]) | set(opto_dna02_motor["bodyId_pre"]))
om_motor_ids = sorted(opto_dna02_motor["bodyId_post"].unique().tolist())

groups = {
    "gf_sensor_looming": looming_ids,
    "gf_sensor_visual": gf_sensor_ids,
    "gf_dnp01_giantfiber": GF_IDS,
    "gf_ttmn_motor": gf_motor_ids,
    "om_sensor_t4t5": om_sensor_ids,
    "om_hs_widefield": hs_ids,
    "om_dna02_steering": dna02_ids,
    "om_leg_motor": om_motor_ids,
}

if args.group:
    groups = {k: v for k, v in groups.items() if k == args.group}
    if not groups:
        raise SystemExit(f"grupo desconhecido: {args.group}")

# cobertura vai pro lado dos SWC pra o export_unity.py achar sem refazer a conta
(OUT / "coverage.json").write_text(
    json.dumps({"gf_sensor_looming": COBERTURA_LOOMING}, indent=2), encoding="utf-8")

for name, ids in groups.items():
    print(f"Puxando {name}: {len(ids)} neuronio(s)...")
    skels = neu.fetch_skeletons(ids)
    for n in skels:
        navis.write_swc(n, OUT / f"{name}__{n.id}.swc")

print("\nExportando malha cerebro+VNC (Male CNS, JRCFIB2022M)...")
flybrains.JRCFIB2022M.mesh_brain.export(ROOT / "blender" / "mesh_brain.obj")
flybrains.JRCFIB2022M.mesh_vnc.export(ROOT / "blender" / "mesh_vnc.obj")

print("\nPronto. Arquivos em blender/skeletons/*.swc e blender/mesh_*.obj")
