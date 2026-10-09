#!/usr/bin/env python3
"""PETSc Anderson reference for the synthetic contraction benchmark.

Identical map to nonlinear_bench.rs: G_i(x) = rho*x_i + (1-rho)*phi_i(x),
phi_i = sin((x_{i-1}+x_{i+1})/4 + 0.01*i)/2 (periodic), rho = 0.95.
SNES ANDERSON with depth 5 mirrors the sonium AndersonConfig.

Usage: python3 snes_contract.py [n ...]
Prints one JSON line per n:
  {"variant":"petsc-snes-anderson","n":..,"ms":..,"newton_iters":..,
   "linear_iters":..,"residual":..,"converged":true}
"""

import json
import sys
import time

import numpy as np

from petsc4py import PETSc  # noqa: E402

RHO = 0.95


def solve(n):
    idx = np.arange(n)
    phase = 0.01 * idx

    def g(x):
        xm = x[idx - 1]
        xp = x[(idx + 1) % n]
        phi = np.sin((xm + xp) * 0.25 + phase) * 0.5
        return RHO * x + (1.0 - RHO) * phi

    x = PETSc.Vec().createSeq(n)
    f = PETSc.Vec().createSeq(n)

    def form_function(snes, X, F):
        xa = X.getArray()
        F.setArray(g(xa) - xa)

    snes = PETSc.SNES().create()
    snes.setType(PETSc.SNES.Type.ANDERSON)
    snes.setFunction(form_function, f)
    snes.setTolerances(rtol=1e-8, atol=1e-10, max_it=2000)
    snes.setFromOptions()
    x.set(0.0)
    t0 = time.perf_counter()
    snes.solve(None, x)
    ms = (time.perf_counter() - t0) * 1000.0
    print(
        json.dumps(
            {
                "variant": "petsc-snes-anderson",
                "n": n,
                "ms": round(ms, 3),
                "newton_iters": snes.getIterationNumber(),
                "linear_iters": snes.getFunctionEvaluations(),
                "residual": snes.getFunctionNorm(),
                "converged": bool(snes.getConvergedReason() > 0),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:] if not a.startswith("-")] or [2000, 8000, 32000]
    # Depth option must precede PETSc initialisation use; set it here via the
    # options database before any SNES reads options in solve().
    PETSc.Options()["snes_anderson_m"] = 5
    for n in sizes:
        solve(n)
