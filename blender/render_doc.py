"""
Render limpo pra documentacao (sem UI do Blender). Headless:
  blender.exe --background --python render_doc.py
Escreve direto em docs/images/ (resolucao dimensionada pro README -- aumente os
dois ultimos args de shot() se precisar de versao pra slide/paper).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "_pylibs"))

import bpy
from mathutils import Vector

HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "docs" / "images"
exec(open(str(HERE / "render_circuits.py"), encoding="utf-8").read())

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.film_transparent = False
sc.world = sc.world or bpy.data.worlds.new("W")
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.02, 0.02, 0.03, 1)

cam.constraints.clear()  # posiciona na mao, sem TRACK_TO


def shot(name, location, rotation, res_x, res_y):
    cam.location = Vector(location)
    cam.rotation_euler = rotation
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.filepath = str(OUT_DIR / name)
    bpy.ops.render.render(write_still=True)
    print("gerado:", name)


d = radius * 2.3
# frontal: camera em -Y olhando pra +Y, +Z pra cima (a vista do print)
shot("cns_circuitos_frontal.png", (0, -d, 0), (1.5708, 0, 0), 1000, 1280)
# perspectiva tres-quartos
shot("cns_circuitos_perspectiva.png", (d * 0.7, -d * 0.7, d * 0.25), (1.35, 0, 0.785), 1100, 1100)
