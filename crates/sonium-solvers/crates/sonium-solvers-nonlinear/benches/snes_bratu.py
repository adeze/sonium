#!/usr/bin/env python3
"""PETSc SNES reference for the 1D Bratu benchmark.

Identical discretization to nonlinear_bench.rs: -u'' = LAMBDA*e^u on (0,1),
u(0) = u(1) = 0, second-order FD with n interior points. SNES NEWTONLS with
KSP GMRES(30) matches the sonium inexact-Newton setup; tolerances mirror the
sonium defaults (rtol 1e-8, atol 1e-10).

Usage: python3 snes_bratu.py [n ...]
Prints one JSON line per n:
  {"variant":"petsc-snes-newtonls","n":..,"ms":..,"newton_iters":..,
   "linear_iters":..,"residual":..,"converged":true}
"""

import json
import sys
import time

import numpy as np
from petsc4py import PETSc

LAMBDA = 1.0


def solve(n):
    h = 1.0 / (n + 1)
    inv_h2 = 1.0 / (h * h)

    x = PETSc.Vec().createSeq(n)
    f = PETSc.Vec().createSeq(n)
    J = PETSc.Mat().createAIJ([n, n], nnz=3)

    def form_function(snes, X, F):
        xa = X.getArray(readonly=True)
        out = np.empty(n)
        xm = np.empty(n)
        xm[0] = 0.0
        xm[1:] = xa[:-1]
        xp = np.empty(n)
        xp[-1] = 0.0
        xp[:-1] = xa[1:]
        out[:] = (-xm + 2.0 * xa - xp) * inv_h2 - LAMBDA * np.exp(xa)
        F.setArray(out)

    def form_jacobian(snes, X, Jmat, Pmat):
        xa = X.getArray(readonly=True)
        diag = 2.0 * inv_h2 - LAMBDA * np.exp(xa)
        Jmat.setValues([0], [0, 1], [diag[0], -inv_h2])
        for i in range(1, n - 1):
            Jmat.setValues([i], [i - 1, i, i + 1], [-inv_h2, diag[i], -inv_h2])
        Jmat.setValues([n - 1], [n - 2, n - 1], [-inv_h2, diag[n - 1]])
        Jmat.assemble()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    snes = PETSc.SNES().create()
    snes.setType(PETSc.SNES.Type.NEWTONLS)
    snes.setFunction(form_function, f)
    snes.setJacobian(form_jacobian, J)
    # rel 1e-6 mirrors the sonium bench: at n = 32000 both solvers stall near
    # 2.4e-6 absolute (attainable rounding floor of the O(1e9) Jacobian).
    snes.setTolerances(rtol=1e-6, atol=1e-10, max_it=50)
    # Exact inner solve (PREONLY+LU), mirroring the Thomas-exact sonium
    # variant: both sides do one O(n) direct tridiagonal solve per step,
    # isolating Newton-loop efficiency.
    ksp = snes.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    snes.setFromOptions()

    x.set(0.0)
    t0 = time.perf_counter()
    snes.solve(None, x)
    ms = (time.perf_counter() - t0) * 1000.0
    print(
        json.dumps(
            {
                "variant": "petsc-snes-newtonls",
                "n": n,
                "ms": round(ms, 3),
                "newton_iters": snes.getIterationNumber(),
                "linear_iters": snes.getLinearSolveIterations(),
                "residual": snes.getFunctionNorm(),
                "converged": bool(snes.getConvergedReason() > 0),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    PETSc.Sys.pushErrorHandler("abort")
    sizes = [int(a) for a in sys.argv[1:] if not a.startswith("-")] or [2000, 8000, 32000]
    for n in sizes:
        solve(n)
