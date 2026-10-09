#!/usr/bin/env python3
"""SciPy L-BFGS-B reference for the extended-Rosenbrock benchmark.

Identical problem to optim_bench.rs: f(x) = sum_pairs [(1-xi)^2 +
100*(xj - xi^2)^2], start (-1.2, 1) per pair, analytic Jacobian,
gtol 1e-6, maxiter 1000.

Usage: python3 scipy_rosen.py [n ...]
Prints one JSON line per n:
  {"variant":"scipy-lbfgsb","n":..,"ms":..,"iters":..,"fun_evals":..,
   "grad_evals":..,"fun":..,"gnorm":..}
"""

import json
import sys
import time

import numpy as np
from scipy.optimize import minimize


def make_problem(n):
    def fun(x):
        xi = x[0::2]
        xj = x[1::2]
        a = 1.0 - xi
        b = xj - xi * xi
        return float(np.sum(a * a + 100.0 * b * b))

    def jac(x):
        xi = x[0::2]
        xj = x[1::2]
        g = np.empty_like(x)
        g[0::2] = -2.0 * (1.0 - xi) - 400.0 * xi * (xj - xi * xi)
        g[1::2] = 200.0 * (xj - xi * xi)
        return g

    x0 = np.empty(n)
    x0[0::2] = -1.2
    x0[1::2] = 1.0
    return fun, jac, x0


def solve(n):
    fun, jac, x0 = make_problem(n)
    t0 = time.perf_counter()
    res = minimize(
        fun, x0, jac=jac, method="L-BFGS-B",
        options={"gtol": 1e-6, "maxiter": 1000},
    )
    ms = (time.perf_counter() - t0) * 1000.0
    print(
        json.dumps(
            {
                "variant": "scipy-lbfgsb",
                "n": n,
                "ms": round(ms, 3),
                "iters": int(res.nit),
                "fun_evals": int(res.nfev),
                "grad_evals": int(res.get("njev", -1)),
                "fun": float(res.fun),
                "gnorm": float(np.max(np.abs(res.jac))),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:] if not a.startswith("-")] or [100, 1000]
    for n in sizes:
        solve(n)
