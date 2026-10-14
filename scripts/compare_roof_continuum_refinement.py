#!/usr/bin/env python3
"""Measure complete original roof reference refinement at eight physical seconds."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

from compare_roof_continuum_field import compare, evidence


def interpolate(field, cells, radial, coordinates):
    counts = [cells, cells] if radial is None else [cells, cells, radial]
    low, fractions = [], []
    for coordinate, count in zip(coordinates, counts):
        index = min(int(coordinate * count), count - 1)
        low.append(index)
        fractions.append(coordinate * count - index)
    value = [0.0, 0.0, 0.0]
    for corner in range(1 << len(counts)):
        nodes = [low[j] + ((corner >> j) & 1) for j in range(len(counts))]
        weight = math.prod(fractions[j] if (corner >> j) & 1 else 1 - fractions[j]
                           for j in range(len(counts)))
        index = nodes[0] * (cells + 1) + nodes[1]
        if radial is not None:
            index = index * (radial + 1) + nodes[2]
        for axis in range(3):
            value[axis] += weight * field[index][axis]
    return value


def measure_difference(a, b, director=False):
    na, ra, data_a = a
    nb, rb, data_b = b
    cuts = sorted({Fraction(i, n) for n in [na, nb] for i in range(n + 1)})
    cuts = list(map(float, cuts))
    radial_cuts = ([0.0, 1.0] if director else
                   list(map(float, sorted({Fraction(i, r) for r in [ra, rb] for i in range(r + 1)}))))
    key = "reference_surface_directors" if director else "reference_body_displacements_m"
    fields = [data_a[key], data_b[key]]
    if director:
        fields = [[[v[0] - math.sin(math.radians(-40 + 80 * (i % (n + 1)) / n)),
                    v[1] - math.cos(math.radians(-40 + 80 * (i % (n + 1)) / n)), v[2]]
                   for i, v in enumerate(field)] for n, field in zip([na, nb], fields)]
    errors, sizes = [0.0] * 3, [0.0] * 3
    integrated_volume = 0.0
    gauss = [.5 - .5 / math.sqrt(3), .5 + .5 / math.sqrt(3)]
    for za, zb in zip(cuts, cuts[1:]):
        for ta, tb in zip(cuts, cuts[1:]):
            for pa, pb in zip(radial_cuts, radial_cuts[1:]):
                for gz in gauss:
                    for gt in gauss:
                        for gr in ([.5] if director else gauss):
                            z, t, r = za + (zb - za) * gz, ta + (tb - ta) * gt, pa + (pb - pa) * gr
                            coordinates = [z, t] if director else [z, t, r]
                            values = [interpolate(field, n, None if director else nr, coordinates)
                                      for field, n, nr in zip(fields, [na, nb], [ra, rb])]
                            weight = (zb - za) * (tb - ta) * math.radians(80) * 50 / 4
                            weight *= 25 if director else (25 + .25 * (r - .5)) * .25 * (pb - pa) / 2
                            integrated_volume += weight
                            for axis in range(3):
                                errors[axis] += weight * (values[0][axis] - values[1][axis]) ** 2
                                sizes[axis] += weight * values[1][axis] ** 2
    if any(value <= 0 or not math.isfinite(value) for value in sizes):
        raise ValueError("finite nonzero physical reference field components required")
    expected_measure = 25 * math.radians(80) * 50 * (1 if director else .25)
    if abs(integrated_volume - expected_measure) > 1e-8 * expected_measure:
        raise ValueError("complete original roof domain must be integrated")
    return {"components": [math.sqrt(error / size) for error, size in zip(errors, sizes)],
            "vector_relative_l2": math.sqrt(sum(errors) / sum(sizes)),
            "covered_physical_measure": integrated_volume}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = args.artifacts
    reference = base / "p2-scordelis-solid18-reference.json"
    levels, comparisons = [], []
    for cells, radial in [(12, 2), (24, 4), (32, 6)]:
        result = compare(base / f"p2-scordelis-mitc4-full-{cells}-1.json", reference,
                         base / f"roof-solid18-reference-{cells}-{radial}-previous-precision9.vtk",
                         base / f"roof-solid18-reference-{cells}-{radial}-final-precision9.vtk",
                         cells, radial, include_reference_field=True)
        levels.append((cells, radial, result))
        comparisons.append({"surface_cells": cells, "thickness_cells": radial,
                            "physical_time_s": result["physical_time_s"],
                            "native_reference_components": result["full_thickness_displacement_components"],
                            "reference_nodes": result["all_reference_nodes"], "inputs": result["inputs"]})
    body = [measure_difference(a, b) for a, b in zip(levels, levels[1:])]
    director = [measure_difference(a, b, director=True) for a, b in zip(levels, levels[1:])]
    contraction = [b / a for a, b in zip(body[0]["components"], body[1]["components"])]
    original = json.loads(reference.read_text())["budgets"]
    result = {"status": "completed full-domain reference refinement measurements; original QA admission unproven",
              "physical_time_s": 8.0, "levels": comparisons,
              "adjacent_full_thickness_displacement_changes": body,
              "adjacent_director_change_differences": director,
              "component_contraction": contraction,
              "original_scalar_refinement_bounds": {key: original[key] for key in
                                                     ["first_relative_change", "final_relative_change", "contraction"]},
              "source": evidence(Path(__file__)),
              "scope": "Every original node/layer participates in piecewise trilinear displacement fields. Positive tensor Gauss integration over the union of all mesh partitions covers the complete original curved roof volume; midsurface director changes use bilinear interpolation. Saved fields strictly bracket actual8s. Full reference time quality and release integration remain independently required; no MacNeal linear target is applied to nonlinear fields."}
    if args.output.exists():
        raise ValueError("existing physical evidence must not be overwritten")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ["levels", "source", "scope"]}, indent=2))


if __name__ == "__main__":
    main()
