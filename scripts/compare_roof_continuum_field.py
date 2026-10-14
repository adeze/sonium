#!/usr/bin/env python3
"""Measure complete roof midsurface and director parity against saved continuum fields.

This diagnostic retains all nodes and reports measurements without QA admission.
Director comparisons cover two bending rotations, not independent drilling motion.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def evidence(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def unit(vector):
    size = norm(vector)
    if not math.isfinite(size) or size == 0:
        raise ValueError("finite nonzero director required")
    return [value / size for value in vector]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def animation(path, original, bricks):
    lines = path.read_text().splitlines()
    if lines[:4] != ["# vtk DataFile Version 3.0", "vtk output", "ASCII", "DATASET UNSTRUCTURED_GRID"]:
        raise ValueError("complete ASCII continuum animation required")
    stamp = float(lines[lines.index("TIME 1 1 double") + 1])
    i = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    count = int(lines[i].split()[1])
    points = [list(map(float, line.split())) for line in lines[i + 1:i + 1 + count]]
    i = lines.index("SCALARS NODE_ID int 1") + 2
    ids = [int(line) for line in lines[i:i + count]]
    if len(ids) != count or len(set(ids)) != count or set(ids) != set(original):
        raise ValueError("every original continuum node required exactly once")
    if len(points) != count or any(len(p) != 3 or any(not math.isfinite(x) for x in p) for p in points):
        raise ValueError("finite complete continuum coordinates required")
    i = next(i for i, line in enumerate(lines) if line.startswith("CELLS "))
    cells = int(lines[i].split()[1])
    connectivity = [list(map(int, line.split())) for line in lines[i + 1:i + 1 + cells]]
    i = lines.index("SCALARS ELEMENT_ID int 1") + 2
    element_ids = [int(line) for line in lines[i:i + cells]]
    if len(set(element_ids)) != cells or set(element_ids) != set(bricks) or len(connectivity) != cells:
        raise ValueError("every original continuum brick required exactly once")
    for element, cell in zip(element_ids, connectivity):
        if (len(cell) != 9 or cell[0] != 8 or len(set(cell[1:])) != 8
                or any(node < 0 or node >= count for node in cell[1:])
                or {ids[node] for node in cell[1:]} != set(bricks[element])):
            raise ValueError("animation connectivity differs from the original body")
    return stamp, dict(zip(ids, points))


def compare(native_path, reference_path, previous_path, final_path, cells, radial,
            include_reference_field=False):
    native = json.loads(native_path.read_text())
    reference = json.loads(reference_path.read_text())
    levels = [level for level in reference["levels"]
              if level["surface_cells"] == cells and level["thickness_cells"] == radial]
    if len(levels) != 1 or cells % 2 or radial % 2 or cells < 2 or radial < 2:
        raise ValueError("one matching original continuum refinement required")
    level = levels[0]
    work = Path(level["work_directory"])
    starter = work / "scordelis_lo_continuum_0000.rad"
    engine = work / "scordelis_lo_continuum_0001.rad"
    if (evidence(starter)["sha256"] != level["starter_sha256"]
            or evidence(engine)["sha256"] != level["engine_deck_sha256"]):
        raise ValueError("original continuum deck provenance differs")
    original, bricks = {}, {}
    section = ""
    for line in starter.read_text().splitlines():
        if line.startswith("/"):
            section = line
        elif line.strip() and not line.startswith("#"):
            words = line.split()
            if section == "/NODE":
                original[int(words[0])] = list(map(float, words[1:4]))
            elif section == "/BRICK/1":
                bricks[int(words[0])] = list(map(int, words[1:]))
    if len(original) != (cells + 1) ** 2 * (radial + 1) or len(bricks) != cells ** 2 * radial:
        raise ValueError("complete original full-thickness roof required")
    before_time, before = animation(previous_path, original, bricks)
    after_time, after = animation(final_path, original, bricks)
    state = native["history"]["final_displacement"]
    stamp = native["history"]["times_s"][-1]
    if (stamp != 8.0 or native["surface_cells"] != cells or len(state) != 6 * (cells + 1) ** 2
            or any(not math.isfinite(value) for value in state)
            or not (before_time <= stamp <= after_time and before_time < after_time)):
        raise ValueError("complete native eight-second state and bracketed reference times required")
    fraction = (stamp - before_time) / (after_time - before_time)
    current = {node: [a + fraction * (b - a) for a, b in zip(before[node], after[node])]
               for node in original}
    actual, expected, weights, angles, director_errors, director_changes = [], [], [], [], [], []
    reference_directors = []
    body_actual, body_expected, body_weights = [], [], []
    for iz in range(cells + 1):
        for it in range(cells + 1):
            index = iz * (cells + 1) + it
            mid = index * (radial + 1) + radial // 2 + 1
            inner = index * (radial + 1) + 1
            outer = inner + radial
            theta = math.radians(-40 + 80 * it / cells)
            normal = [math.sin(theta), math.cos(theta), 0.0]
            for r in range(radial + 1):
                point = [(25 + .25 * (r / radial - .5)) * normal[j] for j in range(3)]
                point[2] = 50 * iz / cells
                if max(abs(a - b) for a, b in zip(point, original[inner + r])) > 1e-9:
                    raise ValueError("reference geometry differs from the original physical roof")
            displacement = state[6 * index:6 * index + 3]
            omega = state[6 * index + 3:6 * index + 6]
            # The native shell stores small global rotation components.
            # Normalize its first-order director rather than infer drilling
            # rotation from continuum coordinates.
            native_director = unit([a + b for a, b in zip(normal, cross(omega, normal))])
            reference_director = unit([a - b for a, b in zip(current[outer], current[inner])])
            reference_directors.append(reference_director)
            actual.append(displacement)
            expected.append([a - b for a, b in zip(current[mid], original[mid])])
            weights.append((.5 if iz in [0, cells] else 1) * (.5 if it in [0, cells] else 1))
            for r in range(radial + 1):
                node = inner + r
                offset = .25 * (r / radial - .5)
                body_actual.append([displacement[j] + offset * (native_director[j] - normal[j])
                                    for j in range(3)])
                body_expected.append([a - b for a, b in zip(current[node], original[node])])
                # Positive cylindrical volume quadrature, including both
                # skins and every interior layer of the continuum reference.
                body_weights.append(weights[-1] * (25 + offset) * (.5 if r in [0, radial] else 1))
            angles.append(math.atan2(norm(cross(native_director, reference_director)),
                                     sum(a * b for a, b in zip(native_director, reference_director))))
            director_errors.append(norm([a - b for a, b in zip(native_director, reference_director)]))
            director_changes.append(norm([a - b for a, b in zip(reference_director, normal)]))
    components = []
    for axis, name in enumerate(["DX", "DY", "DZ"]):
        error = sum(w * (a[axis] - b[axis]) ** 2 for w, a, b in zip(weights, actual, expected))
        size = sum(w * b[axis] ** 2 for w, b in zip(weights, expected))
        components.append({"component": name, "surface_weighted_relative_l2": math.sqrt(error / size),
                           "maximum_absolute_error_m": max(abs(a[axis] - b[axis]) for a, b in zip(actual, expected))})
    body_components = []
    for axis, name in enumerate(["DX", "DY", "DZ"]):
        error = sum(w * (a[axis] - b[axis]) ** 2 for w, a, b in zip(body_weights, body_actual, body_expected))
        size = sum(w * b[axis] ** 2 for w, b in zip(body_weights, body_expected))
        body_components.append({"component": name, "volume_weighted_relative_l2": math.sqrt(error / size),
                                "maximum_absolute_error_m": max(abs(a[axis] - b[axis])
                                                                 for a, b in zip(body_actual, body_expected))})
    result = {"status": "completed time-aligned midsurface/director diagnostic; original QA admission unproven",
            "physical_time_s": stamp, "reference_frame_times_s": [before_time, after_time],
            "reference_interpolation_fraction": fraction, "all_midsurface_nodes": len(actual),
            "all_reference_nodes": len(original), "all_reference_bricks": len(bricks), "components": components,
            "full_thickness_displacement_components": body_components,
            "director_rms_angle_rad": math.sqrt(sum(w * a * a for w, a in zip(weights, angles)) / sum(weights)),
            "director_maximum_angle_rad": max(angles),
            "director_change_relative_l2": math.sqrt(sum(w * a * a for w, a in zip(weights, director_errors))
                                                     / sum(w * a * a for w, a in zip(weights, director_changes))),
            "maximum_reference_interpolation_adjustment_m": max(norm([a - b for a, b in zip(current[node], after[node])])
                                                                  for node in original),
            "inputs": [evidence(p) for p in [native_path, reference_path, previous_path, final_path, starter, engine]],
            "scope": "All original midsurface translations and through-thickness director directions, including supports and edges. Every continuum layer is compared with the native shell displacement plus thickness offset times director change, with positive cylindrical volume weights. Reference fields linearly interpolated strictly between actual saved times; interpolation convergence remains required. Native small-rotation director normalization is explicit. Drilling rotation, full reference h/time quality and original release gates remain independently required."}
    if include_reference_field:
        result["reference_surface_displacements_m"] = expected
        result["reference_surface_directors"] = reference_directors
        result["reference_body_displacements_m"] = body_expected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["native", "reference", "previous", "final", "output"]:
        parser.add_argument(name, type=Path)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--radial", type=int, default=6)
    args = parser.parse_args()
    result = compare(args.native, args.reference, args.previous, args.final, args.cells, args.radial)
    if args.output.exists():
        raise ValueError("existing physical evidence must not be overwritten")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ["inputs", "scope"]}, indent=2))


if __name__ == "__main__":
    main()
