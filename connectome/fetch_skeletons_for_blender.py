"""
Puxa a morfologia 3D REAL (esqueleto, nao so grafo de conectividade) dos
neuronios dos nossos dois circuitos, mais a malha do cerebro+VNC do Male CNS
(via navis-flybrains), pra render bonito no Blender depois.

Sai tudo em blender/skeletons/*.swc e blender/mesh_brain.obj + mesh_vnc.obj.
"""
from pathlib import Path
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

opto_sensor_hs = pd.read_csv(CONNECTOME / "opto_sensor_hs.csv")
opto_hs_dna02 = pd.read_csv(CONNECTOME / "opto_hs_dna02.csv")
opto_dna02_motor = pd.read_csv(CONNECTOME / "opto_dna02_motor.csv")
om_sensor_ids = sorted(opto_sensor_hs["bodyId_pre"].unique().tolist())
hs_ids = sorted(set(opto_sensor_hs["bodyId_post"]) | set(opto_hs_dna02["bodyId_pre"]))
dna02_ids = sorted(set(opto_hs_dna02["bodyId_post"]) | set(opto_dna02_motor["bodyId_pre"]))
om_motor_ids = sorted(opto_dna02_motor["bodyId_post"].unique().tolist())

groups = {
    "gf_sensor_visual": gf_sensor_ids,
    "gf_dnp01_giantfiber": GF_IDS,
    "gf_ttmn_motor": gf_motor_ids,
    "om_sensor_t4t5": om_sensor_ids,
    "om_hs_widefield": hs_ids,
    "om_dna02_steering": dna02_ids,
    "om_leg_motor": om_motor_ids,
}

for name, ids in groups.items():
    print(f"Puxando {name}: {len(ids)} neuronio(s)...")
    skels = neu.fetch_skeletons(ids)
    for n in skels:
        navis.write_swc(n, OUT / f"{name}__{n.id}.swc")

print("\nExportando malha cerebro+VNC (Male CNS, JRCFIB2022M)...")
flybrains.JRCFIB2022M.mesh_brain.export(ROOT / "blender" / "mesh_brain.obj")
flybrains.JRCFIB2022M.mesh_vnc.export(ROOT / "blender" / "mesh_vnc.obj")

print("\nPronto. Arquivos em blender/skeletons/*.swc e blender/mesh_*.obj")
