#!/usr/bin/env python3
"""Compare the two deterministic S3 facade dispatch artifacts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"S3 native-x86 comparison failed: {message}")


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: check_solver_native_x86.py NATIVE_ARTIFACT SCALAR_ARTIFACT")
    native = json.loads(Path(sys.argv[1]).read_text())
    scalar = json.loads(Path(sys.argv[2]).read_text())
    if native.get("dispatch_mode") != "native-avx2-fma":
        fail(f"native dispatch mode is {native.get('dispatch_mode')!r}")
    if scalar.get("dispatch_mode") != "scalar":
        fail(f"scalar dispatch mode is {scalar.get('dispatch_mode')!r}")

    native_kernels = native["dispatch"]["sparse_selected_kernels"]
    scalar_kernels = scalar["dispatch"]["sparse_selected_kernels"]
    for scalar_name in ("f32", "f64", "complex64"):
        if native_kernels[scalar_name] != "avx2-fma":
            fail(f"native {scalar_name} kernel is {native_kernels[scalar_name]!r}")
    if native_kernels["complex32"] != "scalar":
        fail("Complex32 unexpectedly advertises an x86 SIMD sparse kernel")
    if any(kernel != "scalar" for kernel in scalar_kernels.values()):
        fail(f"forced-scalar kernels are {scalar_kernels!r}")

    native_rows = {
        (row["family"], row["algorithm"], row["scalar"]): row
        for row in native["executions"]
    }
    scalar_rows = {
        (row["family"], row["algorithm"], row["scalar"]): row
        for row in scalar["executions"]
    }
    if native_rows.keys() != scalar_rows.keys():
        fail("native and scalar execution keys differ")

    budgets = {"f32": 1.0e-5, "f64": 1.0e-10, "complex32": 1.0e-5, "complex64": 1.0e-10}
    max_solution_error = 0.0
    max_residual_delta = 0.0
    max_backward_delta = 0.0
    for key, native_row in native_rows.items():
        scalar_row = scalar_rows[key]
        if native_row["status"] != scalar_row["status"]:
            fail(f"{key} status differs")
        if native_row["iterations"] != scalar_row["iterations"]:
            fail(f"{key} iteration count differs: {native_row['iterations']} != {scalar_row['iterations']}")
        budget = budgets[native_row["scalar"]]
        residual_delta = abs(native_row["residual"] - scalar_row["residual"])
        backward_delta = abs(native_row["backward_error"] - scalar_row["backward_error"])
        solution_delta = max(
            math.hypot(native_value["re"] - scalar_value["re"], native_value["im"] - scalar_value["im"])
            for native_value, scalar_value in zip(native_row["solution"], scalar_row["solution"])
        )
        max_residual_delta = max(max_residual_delta, residual_delta)
        max_backward_delta = max(max_backward_delta, backward_delta)
        max_solution_error = max(max_solution_error, solution_delta)
        if max(residual_delta, backward_delta, solution_delta) > budget:
            fail(
                f"{key} dispatch delta exceeds {budget}: "
                f"residual={residual_delta}, backward={backward_delta}, solution={solution_delta}"
            )

    native_path = Path(sys.argv[1])
    scalar_path = Path(sys.argv[2])
    print(json.dumps({
        "schema_version": 1,
        "case_id": "solvers.facade-native-x86-comparison.v1",
        "status": "passed",
        "native_dispatch_mode": native["dispatch_mode"],
        "scalar_dispatch_mode": scalar["dispatch_mode"],
        "native_artifact": f"{native_path.parent.name}/{native_path.name}",
        "scalar_artifact": f"{scalar_path.parent.name}/{scalar_path.name}",
        "executions": len(native_rows),
        "max_solution_error": max_solution_error,
        "max_residual_delta": max_residual_delta,
        "max_backward_delta": max_backward_delta,
        "budgets": budgets,
    }))


if __name__ == "__main__":
    main()
