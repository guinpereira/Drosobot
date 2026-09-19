import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "_pylibs"))

import bpy
from mathutils import Vector

exec(open(str(Path(__file__).parent / "render_circuits.py"), encoding="utf-8").read())

# forca o depsgraph resolver antes de ler .dimensions / matrix_world
bpy.context.view_layer.update()
bpy.context.evaluated_depsgraph_get()


def world_bbox(obj):
    """bbox real em coordenadas de mundo (curve: usa os pontos, nao o bound_box)."""
    mw = obj.matrix_world
    if obj.type == "CURVE":
        pts = []
        for sp in obj.data.splines:
            for p in (sp.points if len(sp.points) else sp.bezier_points):
                pts.append(mw @ Vector(p.co[:3]))
    else:
        pts = [mw @ Vector(c) for c in obj.bound_box]
    if not pts:
        return None, None
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def fmt(v):
    return "(%.3f, %.3f, %.3f)" % (v.x, v.y, v.z)


print("\n\n===== DIAGNOSTICO =====")

brain = bpy.data.objects["CNS_brain"]
bmn, bmx = world_bbox(brain)
print("CNS_brain  min %s  max %s  dim %s" % (fmt(bmn), fmt(bmx), fmt(bmx - bmn)))

vnc = bpy.data.objects.get("CNS_vnc")
if vnc:
    vmn, vmx = world_bbox(vnc)
    print("CNS_vnc    min %s  max %s  dim %s" % (fmt(vmn), fmt(vmx), fmt(vmx - vmn)))

neurons = [o for o in bpy.data.objects if o.type == "CURVE"]
print("\nneuronios (CURVE): %d" % len(neurons))

gf = next((o for o in neurons if "10001" in o.name), None)
if gf:
    mn, mx = world_bbox(gf)
    ctr = (mn + mx) / 2
    print("\n%s" % gf.name)
    print("  min %s  max %s" % (fmt(mn), fmt(mx)))
    print("  dim %s   centro %s" % (fmt(mx - mn), fmt(ctr)))
    print("  obj.dimensions (Blender) %s" % fmt(gf.dimensions))
    print("  bevel_depth %.4f  splines %d" % (gf.data.bevel_depth, len(gf.data.splines)))
    dentro = all(bmn[i] <= ctr[i] <= bmx[i] for i in range(3))
    print("  centro dentro do bbox do cerebro? %s" % dentro)

# bbox agregado de todos os neuronios vs cerebro
amn = Vector((1e18, 1e18, 1e18))
amx = Vector((-1e18, -1e18, -1e18))
for o in neurons:
    mn, mx = world_bbox(o)
    if mn is None:
        continue
    for i in range(3):
        amn[i] = min(amn[i], mn[i])
        amx[i] = max(amx[i], mx[i])
print("\nTODOS neuronios  min %s  max %s  dim %s" % (fmt(amn), fmt(amx), fmt(amx - amn)))
print("razao dim neuronios/cerebro: %.2f, %.2f, %.2f" % tuple(
    (amx[i] - amn[i]) / (bmx[i] - bmn[i]) for i in range(3)))

print("\nTotal objetos:", len(bpy.data.objects))
