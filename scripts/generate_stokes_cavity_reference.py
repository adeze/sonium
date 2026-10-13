#!/usr/bin/env python3
"""Generate an independent clamped-biharmonic finite-difference cavity reference.

The steady incompressible Stokes streamfunction satisfies Laplacian^2 psi=0.
Velocity is (psi_y,-psi_x). Psi vanishes on every wall. Its normal derivative
is zero on the stationary walls and psi_y=1 on the top wall, excluding corners.
A 13-point second-order stencil and reflected ghost values impose these data.
The reference uses no Sonium assemblies or solver implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def solve(n, lid_speed=1.0, manufactured=False):
    h = 1.0 / n
    m = n - 1
    rows, cols, values = [], [], []
    rhs = np.zeros(m * m)
    stencil = [(0, 0, 20), (1, 0, -8), (-1, 0, -8),
               (0, 1, -8), (0, -1, -8), (1, 1, 2),
               (1, -1, 2), (-1, 1, 2), (-1, -1, 2),
               (2, 0, 1), (-2, 0, 1), (0, 2, 1), (0, -2, 1)]
    for j in range(1, n):
        for i in range(1, n):
            row = (j - 1) * m + i - 1
            if manufactured:
                cx, cy = np.cos(2*np.pi*i*h), np.cos(2*np.pi*j*h)
                rhs[row] = h**4 * np.pi**4 * (16*cx*cy - 4*(cx+cy))
            for dx, dy, coefficient in stencil:
                x, y = i + dx, j + dy
                if x in (0, n) or y in (0, n):
                    continue
                if x < 0:
                    x = -x
                elif x > n:
                    x = 2*n-x
                if y < 0:
                    y = -y
                elif y > n:
                    y = 2*n-y
                    rhs[row] -= coefficient * 2*h*lid_speed
                rows.append(row)
                cols.append((y - 1) * m + x - 1)
                values.append(coefficient)
    matrix = coo_matrix((values, (rows, cols)), shape=(m*m, m*m)).tocsc()
    solution = spsolve(matrix, rhs, permc_spec='MMD_AT_PLUS_A')
    residual = np.linalg.norm(matrix @ solution-rhs)/np.linalg.norm(rhs)
    if not np.isfinite(residual) or residual > 1e-8:
        raise RuntimeError(f'independent solve residual {residual}')
    psi = np.zeros((n+1, n+1))
    psi[1:n, 1:n] = solution.reshape((m, m))
    return psi, float(residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    errors = []
    for n in (32, 64):
        psi, residual = solve(n, lid_speed=0.0, manufactured=True)
        coordinates = np.arange(n+1)/n
        exact = np.sin(np.pi*coordinates[:, None])**2*np.sin(np.pi*coordinates[None, :])**2
        error = np.linalg.norm(psi-exact)/np.linalg.norm(exact)
        errors.append(float(error))
    mms_rate = float(np.log2(errors[0]/errors[1]))
    if mms_rate < 1.8 or errors[-1] > 0.005:
        raise RuntimeError(f'independent MMS convergence failed: errors={errors}, rate={mms_rate}')
    samples = [i/8 for i in range(1, 8)]
    levels = []
    for n in (128, 256, 512):
        psi, residual = solve(n)
        h = 1/n
        vertical = [(psi[round(y*n)+1, n//2]-psi[round(y*n)-1, n//2])/(2*h) for y in samples]
        horizontal = [-(psi[n//2, round(x*n)+1]-psi[n//2, round(x*n)-1])/(2*h) for x in samples]
        levels.append({'subdivisions': n, 'true_relative_residual': residual,
                       'vertical_ux': vertical, 'horizontal_uy': horizontal})
        print(n, residual, vertical, flush=True)
    differences = [float(np.max(np.abs(np.r_[a['vertical_ux'], a['horizontal_uy']]-np.r_[b['vertical_ux'], b['horizontal_uy']]))) for a, b in zip(levels, levels[1:])]
    if differences[-1] > 0.0002 or differences[-1] > 0.4*differences[0]:
        raise RuntimeError(f'independent reference refinement failed: {differences}')
    # The MMS and three reference levels establish second-order error. Use
    # Richardson extrapolation and bound its remaining change independently.
    samples_by_level = [np.r_[level['vertical_ux'], level['horizontal_uy']] for level in levels]
    coarse_extrapolation = (4*samples_by_level[1]-samples_by_level[0])/3
    fine_extrapolation = (4*samples_by_level[2]-samples_by_level[1])/3
    extrapolation_change = float(np.max(np.abs(fine_extrapolation-coarse_extrapolation)))
    if extrapolation_change > 2e-6:
        raise RuntimeError(f'independent extrapolation did not resolve: {extrapolation_change}')
    record = {'schema_version': 1, 'reference': 'independent 13-point clamped-biharmonic finite differences',
              'equations': 'Delta^2 psi=0; u=psi_y; v=-psi_x',
              'domain': 'unit square, unit tangential top lid, other walls stationary',
              'sample_coordinates': samples, 'levels': levels,
              'mms_relative_l2_errors': errors, 'mms_refinement_rate': mms_rate,
              'maximum_adjacent_centerline_changes': differences,
              'reference_uncertainty_bound_absolute_lid_units': 2e-6,
              'extrapolated_centerlines': fine_extrapolation.tolist(),
              'maximum_extrapolation_change': extrapolation_change,
              'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'numpy_version': np.__version__, 'scipy_version': scipy.__version__}
    args.output.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
