#!/usr/bin/env python3
"""Compare complete original warping fields at an identical measured time.

All reference nodes and all displacement components are retained, including
loaded nodes. This report provides measurements, never a standalone QA pass.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def evidence(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(native_path, reference_path, vtk_path, axial_cells, section_factor):
    native = json.loads(native_path.read_text())
    reference = json.loads(reference_path.read_text())
    levels = [level for level in reference["levels"]
              if level["axial_cells"] == axial_cells and level["section_factor"] == section_factor]
    if len(levels) != 1:
        raise ValueError("exactly one declared reference level required")
    level = levels[0]
    work = Path(level["work_directory"])
    starter_path = work / "beam_warping_0000.rad"
    engine_path = work / "beam_warping_0001.rad"
    if evidence(starter_path)["sha256"] != level["starter_sha256"]:
        raise ValueError("reference starter provenance differs")
    if evidence(engine_path)["sha256"] != level["engine_deck_sha256"]:
        raise ValueError("reference engine provenance differs")
    lines = vtk_path.read_text().splitlines()
    if lines[:4] != ["# vtk DataFile Version 3.0", "vtk output", "ASCII", "DATASET UNSTRUCTURED_GRID"]:
        raise ValueError("expected complete ASCII reference animation")
    time_index = lines.index("TIME 1 1 double") + 1
    stamp = float(lines[time_index])
    field_time = native["native"]["final_time_s"] if "native" in native else native["nodal_field_time_s"]
    if not (stamp == field_time == native["physical_stop_time_s"] == 0.5):
        raise ValueError("original complete half-second fields must have identical times")
    i = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    count = int(lines[i].split()[1])
    current = [list(map(float, line.split())) for line in lines[i + 1:i + 1 + count]]
    if len(current) != count or any(len(point) != 3 for point in current):
        raise ValueError("complete reference three-component coordinates required")
    i = next(i for i, line in enumerate(lines) if line.startswith("CELLS "))
    cells = int(lines[i].split()[1])
    connectivity = [list(map(int, line.split())) for line in lines[i + 1:i + 1 + cells]]
    if (len(connectivity) != cells or any(cell[0] != 8 or len(cell) != 9
            or any(node < 0 or node >= count for node in cell[1:]) for cell in connectivity)):
        raise ValueError("original reference eight-node brick connectivity required")
    i = lines.index("SCALARS NODE_ID int 1") + 2
    ids = [int(line) for line in lines[i:i + count]]
    if len(set(ids)) != count:
        raise ValueError("duplicate reference node identifiers")
    original = {}
    bricks = {}
    active = False
    brick_active = False
    for line in starter_path.read_text().splitlines():
        if line.startswith("/"):
            active = line == "/NODE"
            brick_active = line == "/BRICK/1"
        elif active and line.strip() and not line.startswith("#"):
            words = line.split()
            original[int(words[0])] = list(map(float, words[1:4]))
        elif brick_active and line.strip() and not line.startswith("#"):
            words = list(map(int, line.split()))
            if len(words) != 9:
                raise ValueError("original eight-node brick card required")
            bricks[words[0]] = words[1:]
    if set(ids) != set(original):
        raise ValueError("animation does not cover every original reference node")
    i = lines.index("SCALARS ELEMENT_ID int 1") + 2
    element_ids = [int(line) for line in lines[i:i + cells]]
    if len(set(element_ids)) != cells or set(element_ids) != set(bricks):
        raise ValueError("every original reference brick must appear exactly once")
    for element, cell in zip(element_ids, connectivity):
        if len(set(cell[1:])) != 8 or {ids[index] for index in cell[1:]} != set(bricks[element]):
            raise ValueError("animation connectivity differs from the physical deck")
    coordinates = [original[node] for node in ids]
    expected = [[x - y for x, y in zip(point, initial)] for point, initial in zip(current, coordinates)]
    formulation = "linear-P2"
    if "native" in native:
        history = native["native"]
        if (native["axial_cells"] != axial_cells or native["section_factor"] != section_factor
                or history["starter_sha256"] != level["starter_sha256"]
                or history["engine_deck_sha256"] != level["engine_deck_sha256"]):
            raise ValueError("objective native and reference physical decks must match exactly")
        native_work = Path(native["work_directory"])
        for suffix, expected_hash in [("0000", history["starter_sha256"]),
                                      ("0001", history["engine_deck_sha256"])]:
            if evidence(native_work / f"beam_warping_{suffix}.rad")["sha256"] != expected_hash:
                raise ValueError("objective native physical deck provenance differs")
        native_ids = history["node_ids"]
        positions = history["final_positions_m"]
        plastic = history["final_element_plastic_strain"]
        if (len(native_ids) != len(set(native_ids)) or set(native_ids) != set(original)
                or len(positions) != len(native_ids) or history["eroded_elements"] != 0
                or len(plastic) != cells or any(value != 0.0 for value in plastic)
                or history["integration_formulation"] != "gauss-bbar"):
            raise ValueError("complete purely elastic objective native body required")
        if any(len(p) != 3 or any(not math.isfinite(x) for x in p) for p in positions):
            raise ValueError("finite objective native positions required")
        points = [original[node] for node in native_ids]
        values = [[a - b for a, b in zip(point, initial)]
                  for point, initial in zip(positions, points)]
        formulation = "objective-Gauss-Bbar-elastic"
    else:
        points = native["nodal_coordinates_m"]
        values = native["nodal_displacements_m"]
    if len(points) != len(values) or not points:
        raise ValueError("complete native nodal field required")
    if any(len(p) != 3 or any(not math.isfinite(x) for x in p) for p in points + values + current):
        raise ValueError("finite three-component coordinates and displacements required")
    key = lambda p: tuple(round(x * 1e12) for x in p)
    lookup = {key(point): i for i, point in enumerate(points)}
    if len(lookup) != len(points):
        raise ValueError("duplicate native nodal coordinates")
    actual = []
    for point in coordinates:
        index = lookup.get(key(point))
        if index is None or max(abs(a - b) for a, b in zip(points[index], point)) > 1e-12:
            raise ValueError("a reference node has no exact native coordinate")
        actual.append(values[index])
    weights = [0.0] * count
    volume = 0.0
    for cell in connectivity:
        nodes = cell[1:]
        corners = [coordinates[node] for node in nodes]
        ranges = [max(p[axis] for p in corners) - min(p[axis] for p in corners) for axis in range(3)]
        if any(size <= 0 for size in ranges):
            raise ValueError("invalid reference brick volume")
        cell_volume = math.prod(ranges)
        volume += cell_volume
        for node in nodes:
            weights[node] += cell_volume / 8
    # Two metres times the original .004-square-metre I-section area.
    if abs(volume - 0.008) > 1e-11 or any(weight <= 0 for weight in weights):
        raise ValueError("complete original I-section volume coverage required")
    components = []
    for axis, name in enumerate(["DX", "DY", "DZ"]):
        error = sum(w * (a[axis] - b[axis]) ** 2 for w, a, b in zip(weights, actual, expected))
        size = sum(w * b[axis] ** 2 for w, b in zip(weights, expected))
        components.append({"component": name, "volume_weighted_relative_l2": math.sqrt(error / size) if size else None,
                           "volume_weighted_rms_error_m": math.sqrt(error / volume),
                           "maximum_absolute_error_m": max(abs(a[axis] - b[axis]) for a, b in zip(actual, expected))})
    return {"status": "completed full-body field comparison; original QA admission remains unproven",
            "inputs": [evidence(p) for p in [native_path, reference_path, vtk_path, starter_path, engine_path]],
            "physical_time_s": stamp, "reference_nodes": count, "native_nodal_positions": len(points),
            "native_formulation": formulation,
            "reference_bricks": cells, "covered_reference_volume_m3": volume, "components": components,
            "scope": "All reference nodes and three components, including load and root nodes, at actual 0.5 s. Exact native coordinate matches and positive original brick-volume nodal weights. Every original node and brick is retained. Objective fields additionally require identical physical decks and zero plasticity/erosion. Reference animation ASCII rounding is retained. Original physics and all release gates remain independently required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("vtk", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--axial-cells", type=int, default=80)
    parser.add_argument("--section-factor", type=int, default=4)
    args = parser.parse_args()
    result = compare(args.native, args.reference, args.vtk, args.axial_cells, args.section_factor)
    if args.output.exists():
        raise ValueError("existing evidence must not be overwritten")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["components"], indent=2))


if __name__ == "__main__":
    main()
