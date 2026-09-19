"""
Exporta o CNS real (malha + morfologia dos neuronios) pra Unity.

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --python blender\\export_unity.py

Sai em unity_assets/cns/:
    cns.glb              cerebro, VNC e os neuronios, com a geometria REAL
    neuron_metadata.json bodyId, grupo, lado, tipo, neurotransmissor por objeto

## Por que este script EXECUTA o render_circuits.py em vez de refazer a cena

A montagem da cena carrega quatro correcoes que custaram caro pra descobrir e
que estao documentadas nos comentarios de render_circuits.py:

    1. navis Handler tem scaling=1/10000 por padrao, aplicado EM CIMA da escala
       que voce ja passou -- o esqueleto saia 10.000x menor que a malha
    2. o esqueleto do neuPrint vem em voxel (1 unidade = 8 nm), a malha do
       flybrains ja vem em nanometro puro
    3. bpy.ops.wm.obj_import converte eixo por padrao (OBJ Y-up -> Blender Z-up),
       e os esqueletos entram sem essa conversao
    4. a rotacao de orientacao anatomica gira em torno da ORIGEM DO MUNDO, nao
       do centro da peca

Reimplementar isso aqui seria criar uma segunda copia pra sair de sincronia com
a primeira. Entao o exportador roda o script canonico e trabalha em cima do
resultado. Se alguem corrigir algo la, chega aqui de graca.

Os nomes dos objetos viram `neuron_<bodyId>`, que e como a Unity liga cada malha
ao dado do conectoma.

## Procedencia

O GLB e a morfologia RECONSTRUIDA por microscopia eletronica -- e dado, nao
desenho. Nenhum neuronio aqui foi modelado artisticamente. O metadata separa o
que e medido (bodyId, tipo, neurotransmissor, lado) do que e escolha nossa de
apresentacao (a cor de cada grupo).
"""
import csv
import json
import math
import sys
from pathlib import Path

import bpy

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
# Vai direto pra dentro do projeto Unity: FBX em Assets/ e importado NATIVAMENTE,
# sem pacote nenhum. A alternativa (GLB + glTFast) arrastava Burst, Collections,
# Mathematics, Mono Cecil e o Performance API -- cinco pacotes que a Unity 6
# sinalizou com assinatura invalida. Nenhum deles era necessario pro projeto.
DESTINO = RAIZ / "unity" / "DrosobotLab" / "Assets" / "Resources" / "CNS"
PROPRIEDADES = RAIZ / "connectome" / "neuron_properties.csv"

# ---------------------------------------------------------------------------
# 1. monta a cena com o script canonico (ver docstring)
# ---------------------------------------------------------------------------
canonico = AQUI / "render_circuits.py"
print(f"[export] montando a cena com {canonico.name}")
codigo = compile(canonico.read_text(encoding="utf-8"), str(canonico), "exec")
escopo = {"__file__": str(canonico), "__name__": "__canonico__"}
exec(codigo, escopo)

GROUP_COLORS = escopo["GROUP_COLORS"]
SKEL_DIR = escopo["SKEL_DIR"]

# ---------------------------------------------------------------------------
# 2. propriedades por neuronio (DATA), se o CSV existir
# ---------------------------------------------------------------------------
props = {}
if PROPRIEDADES.exists():
    with PROPRIEDADES.open(encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            try:
                props[int(linha["bodyId"])] = linha
            except (KeyError, ValueError):
                continue
    print(f"[export] {len(props)} neuronios com propriedades de {PROPRIEDADES.name}")
else:
    print(f"[export] AVISO: {PROPRIEDADES} nao existe; metadata sai sem tipo/NT/lado.")
    print("[export] gere com: .venv\\Scripts\\python connectome\\fetch_neuron_properties.py")


def _body_id(nome):
    """
    O navis nomeia os objetos como '#<bodyId> - <grupo>__<bodyId>'.
    Devolve (bodyId, grupo) ou (None, None) se nao for um neuronio.
    """
    if not nome.startswith("#"):
        return None, None
    cabeca, _, cauda = nome.partition(" - ")
    try:
        bid = int(cabeca[1:])
    except ValueError:
        return None, None
    grupo = cauda.split("__")[0] if "__" in cauda else None
    return bid, grupo


# ---------------------------------------------------------------------------
# 3. renomeia e coleta metadata
# ---------------------------------------------------------------------------
neuronios = []
malhas_contexto = []
for obj in list(bpy.data.objects):
    if obj.name in ("CNS_brain", "CNS_vnc"):
        malhas_contexto.append(obj.name)
        continue
    bid, grupo = _body_id(obj.name)
    if bid is None:
        continue

    p = props.get(bid, {})
    nt = (p.get("consensusNt") or "").strip().lower() or None
    # mesma regra de sinal de sim/connectome_model.py (Shiu et al.): GABA e
    # glutamato inibitorios. Repetida aqui porque o Python do Blender e isolado
    # e nao enxerga o pacote sim/.
    if nt in ("gaba", "glutamate", "histamine"):
        sinal = "inhibitory"
    elif nt:
        sinal = "excitatory"
    else:
        sinal = None

    cor = GROUP_COLORS.get(grupo, (0.8, 0.8, 0.8))
    nome_unity = f"neuron_{bid}"
    obj.name = nome_unity
    if obj.data is not None:
        obj.data.name = nome_unity + "_mesh"

    neuronios.append({
        "bodyId": bid,                                    # DATA
        "objectName": nome_unity,
        "group": grupo,                                   # DATA (papel no circuito)
        "side": (p.get("somaSide") or None),              # DATA
        "type": (p.get("type") or None),                  # DATA
        "instance": (p.get("instance") or None),          # DATA
        "neurotransmitter": nt,                           # DATA
        "polarity": sinal,                                # MODEL (regra de Shiu et al.)
        "roleColor": [round(c, 4) for c in cor],          # apresentacao nossa
    })

neuronios.sort(key=lambda n: (n["group"] or "", n["bodyId"]))
print(f"[export] {len(neuronios)} neuronios renomeados para neuron_<bodyId>")
print(f"[export] malhas de contexto: {', '.join(malhas_contexto) or '(nenhuma)'}")

# ---------------------------------------------------------------------------
# 4. tira o que nao deve ir pro GLB
# ---------------------------------------------------------------------------
# camera e luz sao do render estatico; na Unity quem ilumina e a Unity
for nome in ("Câmera", "Camera", "Sol", "Sun", "scene_center"):
    obj = bpy.data.objects.get(nome)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)

# ---------------------------------------------------------------------------
# 5. baixa a tesselacao SO pra exportacao
# ---------------------------------------------------------------------------
# O navis monta as curvas com resolution_u=10, ou seja, cada segmento entre dois
# pontos do SWC vira 10 subdivisoes. Isso e otimo pro render estatico e absurdo
# pra tempo real: com bevel_resolution=2 o GLB saiu com 143,8 MB, que a Unity nao
# carrega bem.
#
# Os pontos do SWC ja sao densos (o Giant Fiber tem 508 splines), entao a
# subdivisao extra nao acrescenta morfologia -- so triangulo. Baixamos aqui, na
# exportacao, e nao no render_circuits.py: a figura do README continua com a
# qualidade que tinha.
#
# Isto e TESSELACAO, nao geometria: nenhum ponto do esqueleto e movido, removido
# ou interpolado. A morfologia exportada e a mesma que o neuPrint reconstruiu.
EXPORT_RESOLUTION_U = 1
EXPORT_BEVEL_RESOLUTION = 0     # tubo de 4 lados em vez de 8

n_curvas = 0
for obj in bpy.data.objects:
    if obj.type == "CURVE":
        obj.data.resolution_u = EXPORT_RESOLUTION_U
        obj.data.bevel_resolution = EXPORT_BEVEL_RESOLUTION
        n_curvas += 1
print(f"[export] tesselacao reduzida em {n_curvas} curvas "
      f"(resolution_u={EXPORT_RESOLUTION_U}, bevel_resolution={EXPORT_BEVEL_RESOLUTION})")

# ---------------------------------------------------------------------------
# 6. exporta
# ---------------------------------------------------------------------------
DESTINO.mkdir(parents=True, exist_ok=True)
fbx = DESTINO / "cns.fbx"

bpy.ops.object.select_all(action="SELECT")

# curvas viram malha: o FBX nao carrega curva com bevel, e a Unity so entende malha.
# Isto e conversao de REPRESENTACAO -- a geometria resultante e a mesma que o
# Blender ja desenhava.
bpy.context.view_layer.objects.active = next(
    (o for o in bpy.data.objects if o.type == "CURVE"), None)
if bpy.context.view_layer.objects.active is not None:
    bpy.ops.object.convert(target="MESH")

bpy.ops.export_scene.fbx(
    filepath=str(fbx),
    use_selection=True,
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_ALL",
    object_types={"MESH"},
    use_mesh_modifiers=True,
    mesh_smooth_type="FACE",
    bake_space_transform=True,
    axis_forward="-Z",
    axis_up="Y",            # Unity e Y-up
    path_mode="COPY",
)
tam = fbx.stat().st_size / 1e6
print(f"[export] {fbx}  ({tam:.1f} MB)")

# ---------------------------------------------------------------------------
# 6b. conectividade entre os neuronios exportados (DATA)
# ---------------------------------------------------------------------------
# As arestas vao junto com a geometria, nao pela telemetria: peso de sinapse e
# neurotransmissor sao dado do conectoma, nao mudam durante a simulacao, e a
# Unity precisa deles antes de qualquer experimento comecar.
#
# Agregamos por PAR de neuronios. O conectoma tem sinapse individual, mas
# desenhar milhoes de linhas nao informa nada -- uma aresta por par, com
# espessura proporcional ao numero de sinapses, informa.
IDS_EXPORTADOS = {n["bodyId"] for n in neuronios}
SINAL_NT = {"gaba": -1, "glutamate": -1, "histamine": -1}

pares = {}
for csv_nome in ("gf_upstream_connections.csv", "gf_downstream_connections.csv",
                 "opto_sensor_hs.csv", "opto_hs_dna02.csv", "opto_dna02_motor.csv"):
    caminho = RAIZ / "connectome" / csv_nome
    if not caminho.exists():
        continue
    with caminho.open(encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            try:
                pre = int(linha["bodyId_pre"]); pos = int(linha["bodyId_post"])
                peso = int(float(linha["weight"]))
            except (KeyError, ValueError):
                continue
            if pre not in IDS_EXPORTADOS or pos not in IDS_EXPORTADOS:
                continue
            pares[(pre, pos)] = pares.get((pre, pos), 0) + peso

arestas = []
for (pre, pos), peso in sorted(pares.items(), key=lambda kv: -kv[1]):
    nt = (props.get(pre, {}).get("consensusNt") or "").strip().lower()
    arestas.append({
        "pre": pre,                                   # DATA
        "post": pos,                                  # DATA
        "weight": peso,                               # DATA (contagem de sinapse EM)
        "sign": SINAL_NT.get(nt, 1) if nt else 1,     # MODEL (regra de Shiu et al.)
    })
print(f"[export] {len(arestas)} arestas entre os neuronios exportados "
      f"(peso de {min(a['weight'] for a in arestas) if arestas else 0} a "
      f"{max(a['weight'] for a in arestas) if arestas else 0})")

# cobertura da amostragem, escrita por fetch_skeletons_for_blender.py.
# Sem isto a interface nao teria como dizer que mostra uma AMOSTRA e nao a
# populacao -- e afirmar que sao todos os LC4/LPLC2 seria falso.
cobertura = {}
cov_path = SKEL_DIR / "coverage.json"
if cov_path.exists():
    try:
        cobertura = json.loads(cov_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[export] AVISO: coverage.json ilegivel ({e})")
for g, c in cobertura.items():
    print(f"[export] cobertura {g}: {c['total_visualized']}/{c['total_simulated']} neuronios, "
          f"{c['fraction_weight_covered']:.0%} do peso sinaptico")

meta = {
    "source": "Male CNS v1.0 (neuPrint, Janelia) + JRCFIB2022M via navis-flybrains",
    "generator": "blender/export_unity.py",
    "note": "Morfologia RECONSTRUIDA por microscopia eletronica. Nenhum neuronio "
            "foi modelado artisticamente. bodyId, type, side e neurotransmitter "
            "sao dado do conectoma; polarity segue a regra de Shiu et al. 2024 "
            "(GABA e glutamato inibitorios); roleColor e escolha nossa de "
            "apresentacao.",
    "contextMeshes": malhas_contexto,
    "groupColors": {g: [round(c, 4) for c in cor] for g, cor in GROUP_COLORS.items()},
    "neuronCount": len(neuronios),
    "coverage": cobertura,
    "edgeCount": len(arestas),
    "edges": arestas,
    "neurons": neuronios,
}
(DESTINO / "neuron_metadata.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[export] {DESTINO / 'neuron_metadata.json'}  ({len(neuronios)} neuronios)")

sem_prop = [n["bodyId"] for n in neuronios if n["type"] is None]
if sem_prop:
    print(f"[export] AVISO: {len(sem_prop)} neuronios sem propriedades "
          f"(rode connectome/fetch_neuron_properties.py): {sem_prop[:5]}...")
print("[export] pronto.")
