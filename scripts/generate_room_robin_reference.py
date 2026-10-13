#!/usr/bin/env python3
"""Independent point-source Helmholtz reference for the QA impedance box.

For exp(-i omega t), G=exp(i k r)/(4 pi r) and p_inc=-i omega rho Q G.
The six finite walls obey dn(p)-i k p=0, not a free-space condition.
Exterior fundamental solutions span the source-free correction. Their
coefficients enforce Robin data at tensor Gauss points on each face.
All basis functions solve the volume PDE analytically; off-grid boundary
residuals and successive complex probe changes assess reference resolution.
This generator imports no Sonium assembly or solver implementation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve

LENGTHS = np.array([1.6, 1.5, 1.4])
SOURCE = np.array([[0.4, 0.35, 0.35]])
PROBES = np.array([[1.2, 1.1, 1.0], [1.0, 0.95, 0.9], [1.25, 0.6, 0.85]])
FREQUENCY = 400.0
WAVENUMBER = 2 * np.pi * FREQUENCY / 343.0
SOURCE_SCALE = -1j * 2 * np.pi * FREQUENCY * 1.204 * 5.88e-4


def faces(order):
    abscissae, weights = leggauss(order)
    points, normals, areas = [], [], []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        for side in [0, 1]:
            for i, a in enumerate(abscissae):
                for j, b in enumerate(abscissae):
                    point = np.zeros(3)
                    point[axis] = side * LENGTHS[axis]
                    point[other] = (np.array([a, b]) + 1) * LENGTHS[other] / 2
                    normal = np.zeros(3)
                    normal[axis] = 2 * side - 1
                    points.append(point)
                    normals.append(normal)
                    areas.append(weights[i] * weights[j] * np.prod(LENGTHS[other]) / 4)
    return np.array(points), np.array(normals), np.array(areas)


def green(points, sources):
    delta = points[:, None, :] - sources[None, :, :]
    radius = np.linalg.norm(delta, axis=2)
    if np.any(radius == 0):
        raise ValueError("reference evaluation meets a fundamental source")
    values = np.exp(1j * WAVENUMBER * radius) / (4 * np.pi * radius)
    return values, delta, radius


def robin(points, normals, sources):
    values, delta, radius = green(points, sources)
    projection = np.sum(delta * normals[:, None, :], axis=2) / radius
    return values * ((1j * WAVENUMBER - 1 / radius) * projection - 1j * WAVENUMBER)


def reference(order, offset):
    points, normals, _ = faces(order)
    exterior = points + offset * normals
    matrix = robin(points, normals, exterior)
    rhs = -SOURCE_SCALE * robin(points, normals, SOURCE).ravel()
    coefficients = solve(matrix, rhs, assume_a="gen")
    algebraic_residual = np.linalg.norm(matrix @ coefficients - rhs) / np.linalg.norm(rhs)
    if not np.isfinite(algebraic_residual) or algebraic_residual > 1e-9:
        raise ValueError(f"reference linear residual {algebraic_residual} exceeds 1e-9")
    pressure = SOURCE_SCALE * green(PROBES, SOURCE)[0].ravel()
    pressure += green(PROBES, exterior)[0] @ coefficients
    # Different Gauss order supplies independent, non-collocated boundary checks.
    checks, check_normals, areas = faces(order + 3)
    incident = SOURCE_SCALE * robin(checks, check_normals, SOURCE).ravel()
    boundary_error = incident + robin(checks, check_normals, exterior) @ coefficients
    boundary_l2 = np.sqrt(np.sum(areas * abs(boundary_error)**2) / np.sum(areas * abs(incident)**2))
    return pressure, {
        "order": order,
        "source_count": len(exterior),
        "exterior_offset_m": offset,
        "linear_relative_residual": float(algebraic_residual),
        "off_grid_relative_robin_l2": float(boundary_l2),
        "pressure_pa": [[float(value.real), float(value.imag)] for value in pressure],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--offset", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.orders) < 3 or any(order < 2 for order in args.orders):
        parser.error("at least three reference orders >=2 are required")
    if any(left >= right for left, right in zip(args.orders, args.orders[1:])):
        parser.error("reference orders must increase strictly")
    if not np.isfinite(args.offset) or args.offset <= 0:
        parser.error("exterior offset must be finite and positive")
    levels, previous = [], None
    for order in args.orders:
        pressure, level = reference(order, args.offset)
        if previous is not None:
            level["previous_level_max_relative_complex_change"] = float(np.max(abs(pressure - previous) / abs(pressure)))
        print(json.dumps(level), flush=True)
        levels.append(level)
        previous = pressure
    args.output.write_text(json.dumps({
        "method": "independent exterior fundamental solutions with finite-box Robin collocation",
        "phasor_convention": "exp(-i omega t)",
        "dimensions_m": LENGTHS.tolist(),
        "source_m": SOURCE[0].tolist(),
        "probes_m": PROBES.tolist(),
        "frequency_hz": FREQUENCY,
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.204,
        "source_volume_velocity_m3_s": 5.88e-4,
        "normalized_wall_admittance": [1.0, 0.0],
        "levels": levels,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
