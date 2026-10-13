#!/usr/bin/env python3
"""Per-frame movie renders via Blender.

Reads a directory of vertex-coloured PLY frames (`frame_XXXX.ply`, as
written by `scripts/website_demos.py`) and renders a PNG sequence with
a fixed camera framed on the first frame's bounding box. Two material
modes: `emission` (flat data look for field plots) and `shaded`
(Principled + sun light for solid bodies).

Usage:
    python3 scripts/blender_movie_render.py --frames-dir <dir> \
        --out-dir <png-dir> --direction 0.3,-0.4,1 --material shaded
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BLENDER_SCRIPT = '''
import bpy, sys, math, json

argv = sys.argv[sys.argv.index("--") + 1:]
frames_dir, out_dir, spec_json = argv[0], argv[1], argv[2]
spec = json.loads(spec_json)

import mathutils
from pathlib import Path

direction = mathutils.Vector(spec["direction"]).normalized()
distance_factor = spec.get("distance_factor", 3.0)
width, height = spec.get("resolution", [1280, 720])
material_mode = spec.get("material", "emission")
background = spec.get("background", [0.06, 0.08, 0.1, 1.0])

frames = sorted(Path(frames_dir).glob("frame_*.ply"))
if not frames:
    raise SystemExit("no PLY frames found")

def import_ply(path):
    bpy.ops.wm.ply_import(filepath=str(path))
    obj = bpy.context.selected_objects[0]
    return obj

def apply_material(obj):
    mesh = obj.data
    attr = mesh.color_attributes[0].name
    mat = bpy.data.materials.new("demo_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = attr
    out = nodes["Material Output"]
    if material_mode == "shaded":
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = 0.55
        bsdf.inputs["Metallic"].default_value = 0.1
        mat.node_tree.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        emission = nodes.new("ShaderNodeEmission")
        mat.node_tree.links.new(vcol.outputs["Color"], emission.inputs["Color"])
        mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
        for candidate in list(nodes):
            if candidate.type == "BSDF_PRINCIPLED":
                nodes.remove(candidate)
    obj.data.materials.append(mat)

# Frame the camera on the union of the first and last frames (demos
# grow or collapse), then lock it for the sequence.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
corners = []
for probe_path in {frames[0], frames[-1]}:
    probe = import_ply(probe_path)
    corners.extend(probe.matrix_world @ mathutils.Vector(c) for c in probe.bound_box)
    bpy.data.objects.remove(probe, do_unlink=True)
center = sum(corners, mathutils.Vector()) / len(corners)
size = max((c - center).length for c in corners)
cam_data = bpy.data.cameras.new("cam")
cam_data.clip_start = size * 0.01
cam_data.clip_end = size * 100.0
cam_data.angle = math.radians(40.0)
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.scene.collection.objects.link(cam)

# Solve the camera distance so every corner fits the frame (bbox-sphere
# framing ignores FOV and aspect and crops oblique silhouettes).
def fit_distance():
    view = -direction
    up_hint = mathutils.Vector((0.0, 1.0, 0.0))
    right = view.cross(up_hint)
    if right.length < 1e-6:
        right = mathutils.Vector((1.0, 0.0, 0.0))
    else:
        right = right.normalized()
    up = right.cross(view).normalized()
    tan_h = math.tan(cam_data.angle / 2.0)
    tan_v = tan_h * (height / width)
    distance = size * distance_factor
    for _ in range(60):
        cam_pos = center + direction * distance
        fits = True
        for corner in corners:
            rel = corner - cam_pos
            depth = rel.dot(view)
            if depth <= 0 or abs(rel.dot(right)) > depth * tan_h * 0.94 or abs(rel.dot(up)) > depth * tan_v * 0.94:
                fits = False
                break
        if fits:
            return distance
        distance *= 1.12
    return distance

cam.location = center + direction * fit_distance()
# Explicit look-at basis (camera -Z forward, +Y up as close to world +Y
# as possible); to_track_quat rolls unpredictably for near-axial views.
forward = (center - cam.location).normalized()
up_hint = mathutils.Vector((0.0, 1.0, 0.0))
right = forward.cross(up_hint)
if right.length < 1e-6:
    right = mathutils.Vector((1.0, 0.0, 0.0))
else:
    right = right.normalized()
up = right.cross(forward).normalized()
rotation = mathutils.Matrix((right, up, -forward)).transposed()
cam.rotation_euler = rotation.to_euler()
bpy.context.scene.camera = cam

if material_mode == "shaded":
    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 3.0
    sun_data.angle = 0.3
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = (width, height)
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.eevee.taa_render_samples = 16
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds["World"]
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = tuple(background)

out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
for index, frame in enumerate(frames):
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    obj = import_ply(frame)
    apply_material(obj)
    scene.render.filepath = str(out / f"frame_{index:04d}.png")
    bpy.ops.render.render(write_still=True)
print(f"rendered {len(frames)} frames")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--direction", default="0.35,-0.5,0.75")
    parser.add_argument("--distance-factor", type=float, default=3.0)
    parser.add_argument("--material", choices=("emission", "shaded"), default="emission")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    direction = [float(part) for part in args.direction.split(",")]
    if len(direction) != 3:
        print("--direction must be x,y,z", file=sys.stderr)
        return 2
    spec = {
        "direction": direction,
        "distance_factor": args.distance_factor,
        "resolution": [args.width, args.height],
        "material": args.material,
        "background": [0.063, 0.094, 0.125, 1.0],
    }
    blender_script = Path(args.out_dir).parent / "blender_movie_scene.py"
    blender_script.parent.mkdir(parents=True, exist_ok=True)
    blender_script.write_text(BLENDER_SCRIPT, encoding="utf-8")
    completed = subprocess.run(
        [
            "blender",
            "-b",
            "--python",
            str(blender_script),
            "--",
            args.frames_dir,
            args.out_dir,
            json.dumps(spec),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        print("blender render failed", file=sys.stderr)
        return 1
    # Blender exits 0 even when the scene script raises; verify output.
    rendered = list(Path(args.out_dir).glob("frame_*.png"))
    if not rendered:
        print("blender produced no frames", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
