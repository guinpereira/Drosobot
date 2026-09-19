"""
Roda ISSO DENTRO do Blender (aba Scripting -> New -> cola -> Run Script).
Precisa ter rodado antes: connectome/fetch_skeletons_for_blender.py (gera os
.swc e os .obj do cerebro/VNC que esse script consome).

Monta a cena: cerebro+VNC translucido de fundo, neuronios reais dos dois
circuitos (Giant Fiber + Optomotor) coloridos por papel no circuito.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "_pylibs"))  # blender ignora user-site, instalamos aqui do lado

import bpy
from mathutils import Vector
import navis
import navis.interfaces.blender  # nao vem junto do "import navis" puro, precisa explicito

ROOT = Path(r"C:\Dev\Drosobot\blender")
SKEL_DIR = ROOT / "skeletons"

# malha (flybrains) ja vem em nanometro puro. esqueleto (neuprint/navis) vem
# em unidade de VOXEL, onde 1 unidade = 8nm reais (navis confirma via n.units
# == "8 nanometer") -- sem multiplicar por 8 primeiro o neuronio fica 8x menor
# que deveria (foi por isso que sumiu da cena, nao era transparencia nem bevel)
VOXEL_TO_NM = 8
SCALE = 1e-5  # escala de visualizacao pra caber na viewport (CNS real tem ~1mm = 1_000_000 nm)
# espessura: usa o raio REAL de cada ponto do SWC (mediana 0.0026, p90 0.0047,
# max 0.077 nessa escala). Bevel chapado de 0.02 era 7.8x a mediana -- inflava
# os arbores finos ate eles se fundirem num bloco solido. RADIUS_GAIN multiplica
# tudo junto, preservando a proporcao entre neurito grosso e fino.
RADIUS_GAIN = 1.0

# limpa cena (mantem camera/luz se ja existirem, senao Blender cria default)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def get_or_create_principled(mat):
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    # Blender 5.x pode nao criar o BSDF default sozinho -- monta na mao
    bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if output is None:
        output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return bsdf


# ---------- cerebro + VNC como contorno translucido ----------
def import_context_mesh(path, name, color, alpha=0.08):
    # forward_axis=Y / up_axis=Z = identidade. O DEFAULT do importador converte
    # OBJ(Y-up) -> Blender(Z-up), trocando Y<->Z e negando um deles. Os esqueletos
    # entram pelo navis em nm cru, SEM essa conversao -- malha e neuronio ficavam
    # em orientacoes diferentes. Mantem os dois no espaco nm original.
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[0]
    obj.name = name
    obj.scale = (SCALE, SCALE, SCALE)
    mat = bpy.data.materials.new(f"mat_{name}")
    mat.use_nodes = True
    mat.blend_method = "HASHED"  # "BLEND" no EEVEE some/oculta objeto emissivo atras da malha translucida
    bsdf = get_or_create_principled(mat)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    obj.data.materials.append(mat)
    return obj

import_context_mesh(ROOT / "mesh_brain.obj", "CNS_brain", (0.6, 0.6, 0.65))
import_context_mesh(ROOT / "mesh_vnc.obj", "CNS_vnc", (0.6, 0.6, 0.65))

# ---------- cor por papel no circuito ----------
GROUP_COLORS = {
    "gf_sensor_visual": (0.3, 0.5, 1.0),      # azul -- entrada visual (PVLP etc)
    "gf_dnp01_giantfiber": (1.0, 0.05, 0.05),  # vermelho vivo -- a estrela do circuito 1
    "gf_ttmn_motor": (1.0, 0.6, 0.0),          # laranja -- motor de pulo
    "om_sensor_t4t5": (0.4, 0.8, 1.0),         # azul claro -- deteccao de movimento
    "om_hs_widefield": (1.0, 0.9, 0.1),        # amarelo -- integracao wide-field
    "om_dna02_steering": (0.1, 1.0, 0.3),      # verde vivo -- comando de giro
    "om_leg_motor": (1.0, 0.1, 0.7),           # magenta -- motor de perna
}

def make_emission_material(name, color):
    # emissao em vez de BSDF refletivo -- fica visivel em qualquer shading,
    # incluindo dentro do cerebro translucido (navis usa object.color, que
    # nao aparece no Material Preview -- material de verdade sempre funciona)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    for node in list(mat.node_tree.nodes):
        mat.node_tree.nodes.remove(node)
    emission = mat.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    # Strength 1.0 = a cor sai exatamente como definida em GROUP_COLORS. Com 2.0
    # o valor estoura acima de 1 e o view transform dessatura: vermelho puro
    # virava salmao, magenta virava rosa -- ruim numa figura onde cor = papel.
    emission.inputs["Strength"].default_value = 1.0
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


# ATENCAO: Handler tem scaling=1/10000 POR PADRAO (navis/interfaces/blender.py).
# Se deixar default ele encolhe o esqueleto 10.000x em cima de qualquer escala
# que voce ja tenha aplicado no neuronio -- era ESSE o motivo de sumir da cena.
# Toda a escala (voxel->nm->viewport) vai aqui, e so aqui.
handler = navis.interfaces.blender.Handler(scaling=VOXEL_TO_NM * SCALE)

for swc_path in sorted(SKEL_DIR.glob("*.swc")):
    group = swc_path.stem.split("__")[0]
    color = GROUP_COLORS.get(group, (0.8, 0.8, 0.8))
    n = navis.read_swc(swc_path)  # sem escalar aqui: quem escala eh o Handler

    before = set(bpy.data.objects.keys())
    handler.add(n, color=color, use_radii=True)
    new_objs = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]

    mat = make_emission_material(f"mat_{swc_path.stem}", color)
    for obj in new_objs:
        obj.show_name = False  # navis liga show_name; 54 nomes viram borrao preto na tela
        if obj.type == "CURVE":
            # use_radii deixa bevel_depth=1 e poe o raio real em cada ponto;
            # aqui so aplica o ganho global de visibilidade
            obj.data.bevel_depth = RADIUS_GAIN
            obj.data.bevel_resolution = 2
        if hasattr(obj.data, "materials"):
            obj.data.materials.clear()
            obj.data.materials.append(mat)

# ---------- orientacao anatomica + centragem ----------
# no espaco do JRCFIB2022M o eixo Z e antero-posterior (cerebro Z baixo -> VNC Z
# alto). Como o Blender usa +Z do mundo como "cima", a mosca sai de cabeca pra
# baixo. Parenteia tudo num root e gira 180 em X: (x,y,z)->(x,-y,-z), rotacao
# rigida (det=+1), nao espelha a morfologia.
root = bpy.data.objects.new("CNS_root", None)
root.empty_display_size = 0.2
bpy.context.scene.collection.objects.link(root)
for obj in list(bpy.data.objects):
    if obj is not root and obj.parent is None:
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
root.rotation_euler = (math.pi, 0, 0)


def scene_bounds():
    """bbox de mundo de tudo que e geometria."""
    bpy.context.view_layer.update()
    mn = Vector((1e18, 1e18, 1e18))
    mx = Vector((-1e18, -1e18, -1e18))
    for obj in bpy.data.objects:
        if obj.type not in ("MESH", "CURVE"):
            continue
        for corner in obj.bound_box:
            p = obj.matrix_world @ Vector(corner)
            for k in range(3):
                mn[k] = min(mn[k], p[k])
                mx[k] = max(mx[k], p[k])
    return mn, mx


# a rotacao acima gira em torno da ORIGEM DO MUNDO, nao do centro da peca -- sem
# esse passo o CNS inteiro fica pendurado em Z negativo, longe da grade. Mede o
# bbox ja rotacionado e desloca o root pra centrar tudo na origem.
mn, mx = scene_bounds()
root.location -= (mn + mx) / 2
mn, mx = scene_bounds()
radius = max((mx - mn).length / 2, 1.0)

# ---------- camera + luz ----------
target = bpy.data.objects.new("scene_center", None)
target.empty_display_size = 0.2
bpy.context.scene.collection.objects.link(target)

bpy.ops.object.camera_add(location=(radius * 2.0, -radius * 2.0, radius * 0.5))
cam = bpy.context.object
cam.parent = None
bpy.context.scene.camera = cam
track = cam.constraints.new("TRACK_TO")
track.target = target
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"

bpy.ops.object.light_add(type="SUN", location=(0, 0, radius * 3))
bpy.context.object.data.energy = 3.0

# ---------- cor fiel ----------
# AgX (default do Blender) e um tonemap cinematografico: comprime e dessatura
# highlights. Otimo pra foto, pessimo pra legenda por cor. "Standard" mapeia
# o valor do material direto no pixel.
bpy.context.scene.view_settings.view_transform = "Standard"

# ---------- viewport pronta pra olhar ----------
# Material Preview direto (material de Emissao nao aparece em Solid) e sem as
# relationship lines, que com 56 filhos num root viram um leque preto na tela.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = "MATERIAL"
                space.overlay.show_relationship_lines = False

bpy.ops.object.select_all(action="DESELECT")

print("Pronto: %d neuronios + cerebro/VNC, centrado na origem, raio %.2f."
      % (len(list(SKEL_DIR.glob("*.swc"))), radius))
print("Dica: Numpad 0 pela camera, Home enquadra tudo na viewport livre.")
