#!/usr/bin/env python3
"""Fixed-camera 3D gallery renders via Blender.

Reads pcfan VTU assets with pyvista (mesh IO only, no GL), converts an
axial mid-slice (coloured by a CFD field with the gallery palette) plus
the fan surface to PLY, and renders deterministic 1920x1080 PNGs with
Blender in background mode.

Usage:
    python3 scripts/blender_gallery_render.py --flow flow_field.vtu \\
        --surface optimized_geometry.vtu --out-dir <gallery> --work-dir <tmp>
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

SIZE = (1920, 1080)
BACKGROUND = (0x10 / 255, 0x18 / 255, 0x20 / 255, 1.0)


def gallery_colour(t: np.ndarray) -> np.ndarray:
    """Mirror sonium_xem gallery `colour()`: blue->cyan->green->yellow->red.

    Written as sRGB bytes: Blender converts vertex colors to linear for
    shading and back to sRGB on display (verified with an RGB test card),
    so pre-linearizing would double-darken the render.
    """
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def write_ply(path: Path, points: np.ndarray, faces: np.ndarray,
              colors: np.ndarray) -> None:
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.uint8)
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for point, color in zip(points, colors):
            fh.write(struct.pack("<fffBBB", *point, *color))
        for face in faces:
            fh.write(struct.pack("<Biii", 3, *face))


BLENDER_SCRIPT = '''
import bpy, sys

argv = sys.argv[sys.argv.index("--") + 1:]
surface_ply, field_ply, output = argv[0], argv[1], argv[2]

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

def import_ply(path, name):
    bpy.ops.wm.ply_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj

def emission_material(obj, name):
    mesh = obj.data
    attr = mesh.color_attributes[0].name
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    node.layer_name = attr
    emission = mat.node_tree.nodes.new("ShaderNodeEmission")
    mat.node_tree.links.new(node.outputs["Color"], emission.inputs["Color"])
    out = mat.node_tree.nodes["Material Output"]
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    # Drop the default Principled node for a flat data look.
    for candidate in list(mat.node_tree.nodes):
        if candidate.type == "BSDF_PRINCIPLED":
            mat.node_tree.nodes.remove(candidate)
    obj.data.materials.append(mat)

surface = import_ply(surface_ply, "fan")
field = import_ply(field_ply, "field")
emission_material(surface, "fan_mat")
emission_material(field, "field_mat")

# Fixed camera: frame both objects from a 3/4 direction.
import mathutils
corners = []
for obj in (surface, field):
    corners += [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
center = sum(corners, mathutils.Vector()) / len(corners)
size = max((c - center).length for c in corners)
direction = mathutils.Vector((1.0, -1.1, 0.55)).normalized()
cam_data = bpy.data.cameras.new("cam")
cam_data.clip_start = size * 0.01
cam_data.clip_end = size * 100.0
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = center + direction * size * 3.0
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = (1920, 1080)
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.eevee.taa_render_samples = 16
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds["World"]
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (
    0x10 / 255, 0x18 / 255, 0x20 / 255, 1.0)
scene.render.filepath = output
bpy.ops.render.render(write_still=True)
'''


def main() -> int:
    import pyvista as pv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir)

    flow = pv.read(args.flow)
    center = np.array(flow.center)
    surface = pv.read(args.surface).extract_surface().triangulate()
    surf_pts = np.asarray(surface.points)
    surf_faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    gray = np.full((len(surf_pts), 3), 150, dtype=np.uint8)
    surface_ply = work / "fan_surface.ply"
    write_ply(surface_ply, surf_pts, surf_faces, gray)

    slice_mesh = flow.slice(normal=(0.0, 0.0, 1.0), origin=center)
    if slice_mesh.n_points == 0:  # fall back to the outer surface
        slice_mesh = flow.extract_surface()
    slice_mesh = slice_mesh.triangulate()
    pts = np.asarray(slice_mesh.points)
    faces = np.asarray(slice_mesh.faces).reshape(-1, 4)[:, 1:]

    for array, name in (("speed_m_s", "flow_3d_speed.png"),
                        ("pressure_pa", "flow_3d_pressure.png")):
        values = np.asarray(slice_mesh.point_data[array], dtype=float)
        lo, hi = values.min(), values.max()
        t = (values - lo) / max(hi - lo, 1e-300)
        colors = (gallery_colour(t) * 255).astype(np.uint8)
        field_ply = work / f"field_{array}.ply"
        write_ply(field_ply, pts, faces, colors)
        script = work / "render.py"
        script.write_text(BLENDER_SCRIPT)
        print(f"+ blender {name}", flush=True)
        subprocess.run(
            ["blender", "--background", "--python", str(script), "--",
             str(surface_ply), str(field_ply), str(out / name)],
            check=True, stdout=subprocess.DEVNULL)
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
