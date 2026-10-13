#!/usr/bin/env python3
"""Generate the website simulation demo movies end to end.

Stages per demo: run the Rust demo driver (`qa-website-demos`, VTU
frame series), convert frames to vertex-coloured PLY (pyvista, IO
only), render a PNG sequence with Blender, and encode WebM/MP4 plus a
poster with ffmpeg. Afterwards the website data files (demo catalogue
and OpenCourant benchmark table) are regenerated.

Usage:
    python3 scripts/website_demos.py all
    python3 scripts/website_demos.py taylor-green crush
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SONIUM_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = SONIUM_ROOT.parent
DEFAULT_WORK_DIR = WORKSPACE_ROOT / "tmp" / "website-demos"
DEFAULT_OUT_DIR = SONIUM_ROOT.parent / "website" / "public" / "demos6"
DEFAULT_SITE_DATA = SONIUM_ROOT.parent / "website" / "src" / "data"

FPS = 12


def gallery_colour(t: np.ndarray) -> np.ndarray:
    """Mirror sonium_xem gallery `colour()`: blue->cyan->green->yellow->red."""
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def write_ply(path: Path, points: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for point, color in zip(points, colors):
            fh.write(struct.pack("<fffBBB", *point, *color))
        for face in faces:
            fh.write(struct.pack("<Biii", 3, *face))


@dataclass(frozen=True)
class DemoSpec:
    slug: str
    title: str
    capability: str
    blurb: str
    field_mode: str  # "vorticity" or a VTU point-field name
    direction: str
    material: str = "emission"
    distance_factor: float = 3.0
    poster_at: float = 0.6
    signed_field: bool = False
    volume_slices: bool = False


DEMOS: dict[str, DemoSpec] = {
    "taylor-green": DemoSpec(
        slug="taylor-green",
        title="Taylor-Green vortex decay",
        capability="Incompressible Navier-Stokes (BDF2, block Schur preconditioner)",
        blurb=(
            "A periodic vortex lattice decays under viscosity. Colour is "
            "vorticity magnitude; the solver is the same BDF2 Navier-Stokes "
            "path gated against the analytic Taylor-Green solution."
        ),
        field_mode="vorticity",
        direction="0.06,-0.10,1",
        distance_factor=3.7,
    ),
    "arch": DemoSpec(
        slug="arch",
        title="Arch snap-through",
        capability="Corotational shell elements (geometric nonlinearity, explicit dynamics)",
        blurb=(
            "A shallow steel arch is driven down at the crown until it "
            "snaps through to the inverted equilibrium. Colour is "
            "membrane strain intensity."
        ),
        field_mode="strain",
        direction="0.85,-1,0.42",
        material="shaded",
        distance_factor=3.0,
        poster_at=0.55,
    ),
    "room-wave": DemoSpec(
        slug="room-wave",
        title="Room impulse response",
        capability="Acoustic wave equation (P1 FEM, central differences, rigid walls)",
        blurb=(
            "A pressure pulse expands in a 3.6 x 2.8 x 2.4 m room and "
            "reflects off the rigid walls. Three mid-planes show the "
            "signed pressure; the field is a P1 finite-element wave "
            "solution at c = 343 m/s."
        ),
        field_mode="pressure",
        signed_field=True,
        volume_slices=True,
        direction="1.15,-1,0.62",
        distance_factor=3.0,
        poster_at=0.5,
    ),
    "plate-modes": DemoSpec(
        slug="plate-modes",
        title="Plate vibration modes",
        capability="Generalized eigenproblem Kv=w2Mv (LOBPCG, shell elements)",
        blurb=(
            "The first four eigenmodes of a clamped steel plate, computed "
            "from the assembled shell stiffness and lumped mass matrices "
            "and animated over one period each."
        ),
        field_mode="disp",
        signed_field=True,
        direction="0.7,-1,0.55",
        material="shaded",
        distance_factor=3.0,
    ),
    "necking": DemoSpec(
        slug="necking",
        title="Necking bar",
        capability="Finite-strain J2 plasticity (Newton equilibrium, Voce hardening)",
        blurb=(
            "A steel bar pulled in tension necks down at a slight "
            "imperfection once the Considere strain is passed. Colour is "
            "axial logarithmic strain; the profile is the computed "
            "cross-section, revolved for display."
        ),
        field_mode="strain",
        direction="0.25,-1,0.35",
        material="shaded",
        distance_factor=3.9,
        poster_at=0.97,
    ),
    "crush": DemoSpec(
        slug="crush",
        title="Thin-walled tube crush",
        capability="Q4 crash shells (Belytschko-Tsay, hourglass control), J2 plasticity, driven end-shortening",
        blurb=(
            "An aluminium tube is crushed against a rigid wall by "
            "prescribed end-shortening and folds progressively. Colour "
            "is membrane strain intensity; plastic hinges form where "
            "the folds localise."
        ),
        field_mode="strain",
        direction="1,-1.35,0.5",
        material="shaded",
        distance_factor=3.2,
        poster_at=0.9,
    ),
}


def run(command: list[str], label: str, cwd: Path | None = None) -> None:
    print(f"+ {label}", flush=True)
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        raise SystemExit(f"stage failed: {label}")


def simulate(spec: DemoSpec, work: Path) -> Path:
    out = work / spec.slug / "vtu"
    out.mkdir(parents=True, exist_ok=True)
    cargo = os.environ.get("SONIUM_CARGO") or os.environ.get("CARGO", "cargo")
    run(
        [
            cargo,
            "run",
            "--release",
            "-q",
            "-p",
            "sonium-qa",
            "--features",
            "qa-binaries",
            "--bin",
            "qa-website-demos",
            "--",
            spec.slug,
            str(out),
        ],
        f"simulate {spec.slug}",
        cwd=SONIUM_ROOT,
    )
    return out


def convert_slices(spec: DemoSpec, vtu_dir: Path, work: Path, manifest: dict) -> Path:
    """Volume VTU frames -> three orthogonal mid-plane slices as PLYs."""
    import pyvista as pv

    ply_dir = work / spec.slug / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    for stale in ply_dir.glob("frame_*.ply"):
        stale.unlink()
    meshes = [pv.read(vtu_dir / name) for name in manifest["frames"]]
    # RMS-referenced scale: the initial spike decays into a reverberant
    # field ~30x weaker, so peak-referencing would wash out the shell.
    rms = [
        float((mesh.point_data[spec.field_mode] ** 2).mean() ** 0.5) for mesh in meshes[5:]
    ]
    scale = 5.0 * float(np.median(rms)) if rms else 1.0
    for index, mesh in enumerate(meshes):
        center = mesh.center
        parts = []
        for normal in ("x", "y", "z"):
            sliced = mesh.slice(normal=normal, origin=center)
            if sliced.n_points:
                parts.append(sliced.triangulate())
        combined = parts[0].merge(parts[1:])
        points = np.asarray(combined.points)
        faces = np.asarray(combined.regular_faces)
        values = np.asarray(combined.point_data[spec.field_mode], dtype=float)
        colours = (gallery_colour(0.5 + 0.5 * values / scale) * 255).astype(np.uint8)
        write_ply(ply_dir / f"frame_{index:04d}.ply", points, faces, colours)
    return ply_dir


def convert(spec: DemoSpec, vtu_dir: Path, work: Path) -> Path:
    """VTU frames -> vertex-coloured PLY frames (fixed colour range)."""
    import pyvista as pv

    manifest = json.loads((vtu_dir / "manifest.json").read_text(encoding="utf-8"))
    if spec.volume_slices:
        return convert_slices(spec, vtu_dir, work, manifest)
    ply_dir = work / spec.slug / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)

    meshes = []
    fields: list[np.ndarray] = []
    for name in manifest["frames"]:
        mesh = pv.read(vtu_dir / name)
        if spec.field_mode == "vorticity":
            derived = mesh.compute_derivative(scalars="velocity", vorticity=True)
            values = np.asarray(derived.point_data["vorticity"])[:, 2]
            field_values = np.abs(values)
        else:
            field_values = np.asarray(mesh.point_data[spec.field_mode], dtype=float)
        meshes.append(mesh)
        fields.append(field_values)
    scale = max(float(np.max(values)) for values in fields) or 1.0

    for stale in ply_dir.glob("frame_*.ply"):
        stale.unlink()
    for index, (mesh, values) in enumerate(zip(meshes, fields)):
        surface = mesh.extract_surface().triangulate()
        points = np.asarray(surface.points)
        faces = np.asarray(surface.regular_faces)
        if spec.field_mode == "vorticity":
            derived = surface.compute_derivative(scalars="velocity", vorticity=True)
            surface_values = np.abs(np.asarray(derived.point_data["vorticity"])[:, 2])
        else:
            surface_values = np.asarray(surface.point_data[spec.field_mode], dtype=float)
        if spec.signed_field:
            colours = (gallery_colour(0.5 + 0.5 * surface_values / scale) * 255).astype(np.uint8)
        else:
            colours = (gallery_colour(surface_values / scale) * 255).astype(np.uint8)
        write_ply(ply_dir / f"frame_{index:04d}.ply", points, faces, colours)
    return ply_dir


def render(spec: DemoSpec, ply_dir: Path, work: Path) -> Path:
    png_dir = work / spec.slug / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(SONIUM_ROOT / "scripts" / "blender_movie_render.py"),
            "--frames-dir",
            str(ply_dir),
            "--out-dir",
            str(png_dir),
            "--direction",
            spec.direction,
            "--distance-factor",
            str(spec.distance_factor),
            "--material",
            spec.material,
        ],
        f"render {spec.slug}",
    )
    return png_dir


def encode(spec: DemoSpec, png_dir: Path, out_dir: Path) -> None:
    target = out_dir / spec.slug
    target.mkdir(parents=True, exist_ok=True)
    frames = sorted(png_dir.glob("frame_*.png"))
    if not frames:
        raise SystemExit(f"no rendered frames for {spec.slug}")
    poster_index = min(len(frames) - 1, max(0, int(len(frames) * spec.poster_at)))
    run(
        ["cp", str(frames[poster_index]), str(target / "poster.png")],
        f"poster {spec.slug}",
    )
    run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(png_dir / "frame_%04d.png"),
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-pix_fmt", "yuv420p",
            str(target / "movie.webm"),
        ],
        f"encode webm {spec.slug}",
    )
    run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(png_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target / "movie.mp4"),
        ],
        f"encode mp4 {spec.slug}",
    )


# OpenCourant comparison, measured 2026-10-12 on a Threadripper PRO
# 3995WX (128 threads) + RTX PRO 6000 Blackwell. Metric: wall time to
# matched accuracy (lid-driven cavity Re=100, Ghia gate 0.03; each code
# on its native discretization — OpenCourant has no incompressible
# solver, so the engine arm is a low-Mach compressible surrogate).
OC_BENCHMARKS = {
    "generated": "2026-10-12",
    "hardware": "AMD Threadripper PRO 3995WX + NVIDIA RTX PRO 6000 Blackwell",
    "metric": (
        "Wall time to matched accuracy: lid-driven cavity at Re=100 against "
        "Ghia reference profiles (max error gate 0.03). Sonium runs its "
        "incompressible Navier-Stokes solver; OpenCourant (Radioss) runs a "
        "low-Mach compressible ALE surrogate, its closest native option."
    ),
    "rows": [
        {"threads": "1", "sonium_s": 15.6, "engine_s": 226.3, "speedup": 14.5,
         "sonium_error": 0.0162, "engine_error": 0.0264},
        {"threads": "8", "sonium_s": 11.0, "engine_s": 60.8, "speedup": 5.5,
         "sonium_error": 0.0162, "engine_error": 0.0264},
        {"threads": "16 / 32", "sonium_s": 10.8, "engine_s": 57.2, "speedup": 5.3,
         "sonium_error": 0.0162, "engine_error": 0.0264},
    ],
    "gpu_rows": [
        {"case": "Lid cavity n=64", "cpu_s": 15.6, "gpu_s": 11.6, "speedup": 1.34},
        {"case": "Lid cavity n=128", "cpu_s": 354.0, "gpu_s": 38.1, "speedup": 9.3},
        {"case": "Taylor-Green n=128", "cpu_s": 23.1, "gpu_s": 11.3, "speedup": 2.0},
    ],
    "notes": [
        "Sonium is faster at every thread count and more accurate at the gate "
        "(0.0162 vs 0.0264 max error). Both codes saturate with threads: "
        "sonium 1.44x self-speedup at 16 threads, the engine 3.95x at 32.",
        "Every GPU row is backend-audited per solve: a silent CPU fallback "
        "marks the row failed instead of relabeling it.",
    ],
}


def explicit_rows() -> list[dict]:
    """Best Sonium vs engine-serial ns/dof/cycle per deck, from perf/oc bundles."""
    decks: dict[str, dict] = {}
    for path in sorted((SONIUM_ROOT / "perf" / "oc").glob("*.json")):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if bundle.get("schema") != "sonium.oc-bench/v1":
            continue
        for row in bundle.get("rows", []):
            if not row.get("correctness_passed"):
                continue
            deck = decks.setdefault(row["deck"], {"deck": row["deck"], "dof": row["dof"]})
            key = "sonium" if row["side"] == "sonium" else "engine"
            # Serial-vs-serial, matching perf/oc-explicit.md "Reading".
            if row["threads"] != 1 or (key == "engine" and row["flavor"] != "serial"):
                continue
            if key not in deck or row["ns_per_dof_cycle"] < deck[key]["ns_per_dof_cycle"]:
                deck[key] = row
    rows = []
    for deck in decks.values():
        if "sonium" not in deck or "engine" not in deck:
            continue
        sonium, engine = deck["sonium"], deck["engine"]
        rows.append({
            "deck": deck["deck"],
            "dof": deck["dof"],
            "sonium_ns_per_dof_cycle": round(sonium["ns_per_dof_cycle"], 1),
            "engine_ns_per_dof_cycle": round(engine["ns_per_dof_cycle"], 1),
            "speedup": round(engine["ns_per_dof_cycle"] / sonium["ns_per_dof_cycle"], 2),
            "sonium_wall_s": sonium["wall_s"],
            "engine_wall_s": engine["wall_s"],
        })
    return sorted(rows, key=lambda row: row["deck"])


def write_site_data(site_data: Path, specs: list[DemoSpec]) -> None:
    site_data.mkdir(parents=True, exist_ok=True)
    # Merge into any existing catalogue so regenerating one demo never
    # drops the other five from the site; order follows DEMOS.
    existing: dict[str, dict] = {}
    catalogue_path = site_data / "demos6.json"
    if catalogue_path.exists():
        try:
            existing = {entry["slug"]: entry for entry in json.loads(catalogue_path.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}
    for spec in specs:
        existing[spec.slug] = {
            "slug": spec.slug,
            "title": spec.title,
            "capability": spec.capability,
            "blurb": spec.blurb,
            "movie_webm": f"/demos6/{spec.slug}/movie.webm",
            "movie_mp4": f"/demos6/{spec.slug}/movie.mp4",
            "poster": f"/demos6/{spec.slug}/poster.png",
        }
    catalogue = [existing[slug] for slug in DEMOS if slug in existing]
    catalogue_path.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    benchmarks = dict(OC_BENCHMARKS, explicit_rows=explicit_rows())
    (site_data / "oc-benchmarks.json").write_text(
        json.dumps(benchmarks, indent=2) + "\n", encoding="utf-8"
    )
    print(f"site data written to {site_data}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demos", nargs="+", help="demo slugs or 'all'")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--site-data", type=Path, default=DEFAULT_SITE_DATA)
    args = parser.parse_args(argv[1:])

    slugs = list(DEMOS) if args.demos == ["all"] else args.demos
    specs: list[DemoSpec] = []
    for slug in slugs:
        spec = DEMOS.get(slug)
        if spec is None:
            print(f"unknown demo {slug!r}; known: {sorted(DEMOS)}", file=sys.stderr)
            return 2
        specs.append(spec)

    for spec in specs:
        vtu_dir = simulate(spec, args.work_dir)
        ply_dir = convert(spec, vtu_dir, args.work_dir)
        png_dir = render(spec, ply_dir, args.work_dir)
        encode(spec, png_dir, args.out_dir)
    write_site_data(args.site_data, specs)
    print("website demos done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
