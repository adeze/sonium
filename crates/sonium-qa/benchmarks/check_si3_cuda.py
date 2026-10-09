#!/usr/bin/env python3
"""Fail-closed checker for SI3 iterative CUDA throughput evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn


CASE_ID = "solvers.iterative-cuda-throughput.v2"
EXPECTED_WORKLOADS = {
    "spd_f64_gmres": ("spd", "Gmres", "f64", 2.0e-6),
    "nonsymmetric_f64_fgmres": ("nonsymmetric", "Fgmres", "f64", 2.0e-6),
    "spd_block_f64_gmres": ("block_spd", "BlockGmres", "f64", 2.0e-6),
    "helmholtz_complex64": ("helmholtz", "Fgmres", "complex64", 2.0e-5),
}
EXPECTED_METHODS = {"Gmres", "Fgmres", "BlockGmres", "BlockFgmres"}
EXPECTED_SCALARS = {"F32", "F64", "Complex32", "Complex64"}
REQUIRED_HARDWARE = (
    "device",
    "driver",
    "toolkit",
    "topology",
    "cpu_model",
    "microcode",
    "compiler",
    "os",
    "target_features",
    "runtime_dispatch",
)
RUNTIME_FIELDS = (
    "d2d_bytes",
    "d2d_operations",
    "d2h_bytes",
    "d2h_operations",
    "h2d_bytes",
    "h2d_operations",
    "launches_or_vendor_operations",
    "live_bytes",
    "peak_bytes",
    "synchronizations",
)
MEASUREMENT_CONTRACT_FIELDS = (
    "setup_ms",
    "iteration_ms_per_solve",
    "synchronization_ms",
    "transfers",
    "final_true_residual",
    "vendor_baseline_setup_ms",
    "vendor_baseline_steady_ms_per_operation",
    "vendor_baseline_numerical_gate",
)
CAPABILITY_FIELDS = (
    "scalar_state",
    "blas1",
    "axpy_many",
    "multi_dot",
    "strided_projection",
    "reduction",
    "csr_spmm",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SI3 artifact rejected: {message}")


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        fail(f"{label} is not a valid finite measurement")
    return number


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def counter(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} is not a non-negative integer counter")
    return value


def runtime(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        fail(f"{label} is missing")
    result: dict[str, int] = {}
    for field in RUNTIME_FIELDS:
        result[field] = counter(value.get(field), f"{label}.{field}")
    if result["peak_bytes"] < result["live_bytes"]:
        fail(f"{label} peak_bytes is below live_bytes")
    for operations, bytes_ in (
        ("d2d_operations", "d2d_bytes"),
        ("d2h_operations", "d2h_bytes"),
        ("h2d_operations", "h2d_bytes"),
    ):
        if (result[operations] == 0) != (result[bytes_] == 0):
            fail(f"{label}.{operations} and {bytes_} disagree")
    return result


def check_capabilities(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 16:
        fail("capability_matrix must contain the complete 16-cell method/scalar matrix")
    seen: set[tuple[str, str]] = set()
    for index, cell in enumerate(value):
        if not isinstance(cell, dict):
            fail(f"capability_matrix[{index}] is not an object")
        method = text(cell.get("method"), f"capability_matrix[{index}].method")
        scalar = text(cell.get("scalar"), f"capability_matrix[{index}].scalar")
        if method not in EXPECTED_METHODS or scalar not in EXPECTED_SCALARS:
            fail(f"capability_matrix[{index}] has an unknown method/scalar cell")
        key = (method, scalar)
        if key in seen:
            fail(f"capability_matrix duplicates {key}")
        seen.add(key)
        if cell.get("backend") != "cuda" or cell.get("native_execution") is not True:
            fail(f"capability_matrix[{index}] is not native CUDA execution")
        for field in CAPABILITY_FIELDS:
            state = text(cell.get(field), f"capability_matrix[{index}].{field}")
            if state not in {"native", "vendor_native"}:
                fail(f"capability_matrix[{index}].{field} is not a native state")
    expected = {(method, scalar) for method in EXPECTED_METHODS for scalar in EXPECTED_SCALARS}
    if seen != expected:
        fail("capability_matrix does not cover every method/scalar combination")


def check_run(
    run: Any,
    label: str,
    *,
    backend: str,
    dofs: int,
    rhs_count: int,
    require_device_work: bool,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    if not isinstance(run, dict) or run.get("backend") != backend or run.get("status") != "passed":
        fail(f"{label} is not a passing {backend} result")
    if run.get("converged") is not True or run.get("stopping_reason") != "Converged":
        fail(f"{label} did not converge with the typed reason")
    iterations = counter(run.get("iterations"), f"{label}.iterations")
    if iterations <= 0:
        fail(f"{label} has no positive iteration count")
    counter(run.get("restarts"), f"{label}.restarts")
    for field in (
        "projected_residual",
        "true_residual",
        "backward_error",
        "independent_final_relative_residual",
        "expected_solution_max_abs_error",
    ):
        finite(run.get(field), f"{label}.{field}")
    for field in ("setup_ms", "warmup_ms", "iteration_ms_per_solve", "synchronization_ms"):
        finite(run.get(field), f"{label}.{field}", positive=True)
    throughput = finite(run.get("steady_throughput_dofs_per_second"), f"{label}.throughput", positive=True)
    expected_throughput = dofs * rhs_count * 3 / (float(run["iteration_ms_per_solve"]) / 1_000.0)
    if not math.isclose(throughput, expected_throughput, rel_tol=1.0e-9, abs_tol=1.0e-12):
        fail(f"{label}.throughput does not match the measured iteration time")
    steady = runtime(run.get("steady_runtime"), f"{label}.steady_runtime")
    synchronization = runtime(run.get("synchronization_runtime"), f"{label}.synchronization_runtime")
    final_readback = runtime(run.get("final_readback_runtime"), f"{label}.final_readback_runtime")
    if require_device_work and steady["launches_or_vendor_operations"] <= 0:
        fail(f"{label} steady state contains no device launches/vendor operations")
    if require_device_work and synchronization["synchronizations"] <= 0:
        fail(f"{label} final synchronization is not recorded")
    if final_readback["d2h_operations"] <= 0 or final_readback["d2h_bytes"] <= 0:
        fail(f"{label} final readback is not recorded")
    return steady, synchronization, final_readback


def check_vendor_baseline(
    baseline: Any,
    label: str,
    *,
    dofs: int,
    rhs_count: int,
    parity_budget: float,
) -> dict[str, Any]:
    if not isinstance(baseline, dict):
        fail(f"{label} is missing")
    if baseline.get("backend") != "cuda" or baseline.get("status") != "passed":
        fail(f"{label} is not a passing CUDA vendor baseline")
    operation = text(baseline.get("operation"), f"{label}.operation")
    expected_operation = "cusparseSpMV" if rhs_count == 1 else "cusparseSpMM"
    if operation != expected_operation:
        fail(f"{label}.operation is not {expected_operation!r}")
    primitive = text(baseline.get("primitive"), f"{label}.primitive")
    if "cuSPARSE" not in primitive or "cuBLAS" not in primitive:
        fail(f"{label}.primitive does not identify cuSPARSE and cuBLAS")
    for field in (
        "setup_ms",
        "warmup_ms",
        "steady_ms_per_operation",
        "synchronization_ms",
        "validation_ms",
    ):
        finite(baseline.get(field), f"{label}.{field}", positive=True)
    throughput = finite(
        baseline.get("steady_throughput_dofs_per_second"),
        f"{label}.throughput",
        positive=True,
    )
    expected_throughput = dofs * rhs_count / (
        float(baseline["steady_ms_per_operation"]) / 1_000.0
    )
    if not math.isclose(throughput, expected_throughput, rel_tol=1.0e-9, abs_tol=1.0e-12):
        fail(f"{label}.throughput does not match the measured operation time")
    steady = runtime(baseline.get("steady_runtime"), f"{label}.steady_runtime")
    setup = runtime(baseline.get("setup_runtime"), f"{label}.setup_runtime")
    synchronization = runtime(
        baseline.get("synchronization_runtime"), f"{label}.synchronization_runtime"
    )
    validation = runtime(baseline.get("validation_runtime"), f"{label}.validation_runtime")
    final_readback = runtime(
        baseline.get("final_readback_runtime"), f"{label}.final_readback_runtime"
    )
    if steady["launches_or_vendor_operations"] < 3:
        fail(f"{label} steady state does not record all repeated vendor operations")
    if steady["h2d_operations"] or steady["d2h_operations"] or steady["d2d_operations"]:
        fail(f"{label} steady state includes a device transfer")
    if setup["h2d_operations"] <= 0 or setup["h2d_bytes"] <= 0:
        fail(f"{label} setup does not record device uploads")
    if synchronization["h2d_operations"] or synchronization["d2h_operations"]:
        fail(f"{label} synchronization interval includes a device transfer")
    if validation["launches_or_vendor_operations"] <= 0:
        fail(f"{label} does not record the cuBLAS validation operation")
    if validation["h2d_operations"] or validation["d2h_operations"]:
        fail(f"{label} validation interval includes a host transfer")
    if final_readback["d2h_operations"] <= 0 or final_readback["d2h_bytes"] <= 0:
        fail(f"{label} final readback is not recorded")
    if synchronization["synchronizations"] <= 0:
        fail(f"{label} explicit synchronization is not recorded")
    if steady["synchronizations"] or validation["synchronizations"]:
        fail(f"{label} hides synchronization inside the steady/validation interval")

    gate = baseline.get("numerical_gate")
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        fail(f"{label}.numerical_gate is not passing")
    residual = finite(
        gate.get("independent_final_relative_residual"),
        f"{label}.numerical_gate.residual",
    )
    output_error = finite(
        gate.get("expected_output_max_abs_error"),
        f"{label}.numerical_gate.output_error",
    )
    if residual > 2.0e-7:
        fail(f"{label} independent vendor residual exceeds 2e-7")
    if output_error > parity_budget:
        fail(f"{label} vendor output error exceeds {parity_budget:g}")
    output_nrm2 = finite(
        gate.get("vendor_output_nrm2"),
        f"{label}.numerical_gate.vendor_output_nrm2",
        positive=True,
    )
    expected_nrm2 = finite(
        gate.get("expected_output_nrm2"),
        f"{label}.numerical_gate.expected_output_nrm2",
        positive=True,
    )
    if not math.isclose(output_nrm2, expected_nrm2, rel_tol=parity_budget, abs_tol=1.0e-12):
        fail(f"{label} cuBLAS norm does not match the host reference")

    comparison = baseline.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("status") != "not_claimed":
        fail(f"{label}.comparison must remain explicitly unclaimed")
    if "speedup" in comparison:
        fail(f"{label}.comparison must not publish speedup before a comparison gate")
    return {
        "operation": operation,
        "primitive": primitive,
        "steady_throughput_dofs_per_second": throughput,
        "steady_launches_or_vendor_operations": steady["launches_or_vendor_operations"],
        "independent_final_relative_residual": residual,
    }


def check_workload(workload: Any, index: int) -> dict[str, Any]:
    if not isinstance(workload, dict):
        fail(f"workloads[{index}] is not an object")
    workload_id = text(workload.get("id"), f"workloads[{index}].id")
    if workload_id not in EXPECTED_WORKLOADS:
        fail(f"unexpected workload {workload_id!r}")
    expected_kind, expected_method, expected_scalar, parity_budget = EXPECTED_WORKLOADS[workload_id]
    for field, expected in (
        ("kind", expected_kind),
        ("method", expected_method),
        ("scalar", expected_scalar),
        ("status", "passed"),
    ):
        if workload.get(field) != expected:
            fail(f"{workload_id}.{field} is not {expected!r}")
    finite(workload.get("cpu_cuda_max_abs_parity"), f"{workload_id}.cpu_cuda_max_abs_parity")
    if float(workload["cpu_cuda_max_abs_parity"]) > parity_budget:
        fail(f"{workload_id} CPU/CUDA parity exceeds {parity_budget:g}")

    dofs = counter(workload.get("dofs"), f"{workload_id}.dofs")
    rhs_count = counter(workload.get("rhs_count"), f"{workload_id}.rhs_count")
    if dofs <= 0 or rhs_count <= 0:
        fail(f"{workload_id} must have positive dofs and rhs_count")
    vendor_primitives = text(workload.get("vendor_primitives"), f"{workload_id}.vendor_primitives")
    if "cuBLAS" not in vendor_primitives or "cuSPARSE" not in vendor_primitives:
        fail(f"{workload_id} does not identify both CUDA vendor primitive families")

    cpu = workload.get("cpu_reference")
    if not isinstance(cpu, dict):
        fail(f"{workload_id} CPU reference is missing")
    check_run(
        cpu,
        f"{workload_id}.cpu",
        backend="cpu",
        dofs=dofs,
        rhs_count=rhs_count,
        require_device_work=False,
    )
    if float(cpu["independent_final_relative_residual"]) > 2.0e-7:
        fail(f"{workload_id} CPU independent final residual exceeds 2e-7")

    cuda = workload.get("cuda")
    steady, _, _ = check_run(
        cuda,
        f"{workload_id}.cuda",
        backend="cuda",
        dofs=dofs,
        rhs_count=rhs_count,
        require_device_work=True,
    )
    if float(cuda["independent_final_relative_residual"]) > 2.0e-7:
        fail(f"{workload_id} independent final residual exceeds 2e-7")
    if float(cuda["expected_solution_max_abs_error"]) > parity_budget:
        fail(f"{workload_id} expected-solution error exceeds {parity_budget:g}")
    vendor = check_vendor_baseline(
        workload.get("vendor_baseline"),
        f"{workload_id}.vendor_baseline",
        dofs=dofs,
        rhs_count=rhs_count,
        parity_budget=parity_budget,
    )
    cuda_capability = workload.get("cuda_capability")
    if not isinstance(cuda_capability, dict):
        fail(f"{workload_id}.cuda_capability is missing")
    expected_capability_scalar = "F64" if expected_scalar == "f64" else "Complex64"
    if (
        cuda_capability.get("backend") != "cuda"
        or cuda_capability.get("method") != expected_method
        or cuda_capability.get("scalar") != expected_capability_scalar
        or cuda_capability.get("native_execution") is not True
    ):
        fail(f"{workload_id}.cuda_capability does not match the executed cell")
    for field in CAPABILITY_FIELDS:
        state = text(cuda_capability.get(field), f"{workload_id}.cuda_capability.{field}")
        if state not in {"native", "vendor_native"}:
            fail(f"{workload_id}.cuda_capability.{field} is not a native state")
    return {
        "id": workload_id,
        "scalar": expected_scalar,
        "method": expected_method,
        "parity": float(workload["cpu_cuda_max_abs_parity"]),
        "residual": float(cuda["independent_final_relative_residual"]),
        "throughput_dofs_per_second": float(cuda["steady_throughput_dofs_per_second"]),
        "steady_launches_or_vendor_operations": int(steady["launches_or_vendor_operations"]),
        "vendor_baseline": vendor,
    }


def load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(document, dict):
        fail("artifact is not a JSON object")
    return document


def main() -> int:
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] != "--output"):
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json [--output CHECK.json]")
    artifact_path = Path(sys.argv[1])
    artifact = load(artifact_path)
    if artifact.get("schema_version") != 2 or artifact.get("case_id") != CASE_ID:
        fail("schema or case id is invalid")
    if artifact.get("requested_backend") != "cuda" or artifact.get("actual_backend") != "cuda":
        fail("required CUDA backend was not used")
    if artifact.get("repetitions_per_steady_measurement") != 3:
        fail("steady measurement repetition count must be three")
    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance is missing")
    for field in REQUIRED_HARDWARE:
        text(hardware.get(field), f"hardware.{field}")
    if hardware["os"] != "linux-x86_64":
        fail("artifact is not native Linux x86_64 evidence")
    if hardware["target_features"] != "cuda" or hardware["runtime_dispatch"] != "cuda-native":
        fail("CUDA dispatch provenance is inconsistent")
    contract = artifact.get("measurement_contract")
    if not isinstance(contract, dict):
        fail("measurement_contract is missing")
    for field in MEASUREMENT_CONTRACT_FIELDS:
        text(contract.get(field), f"measurement_contract.{field}")
    vendor_primitives = artifact.get("vendor_primitives")
    if not isinstance(vendor_primitives, dict):
        fail("vendor_primitives is missing")
    cuda_primitives = text(vendor_primitives.get("cuda"), "vendor_primitives.cuda")
    if "cuBLAS" not in cuda_primitives or "cuSPARSE" not in cuda_primitives:
        fail("vendor_primitives.cuda must identify cuBLAS and cuSPARSE")
    text(vendor_primitives.get("cpu_reference"), "vendor_primitives.cpu_reference")
    unsupported_policy = text(artifact.get("unsupported_policy"), "unsupported_policy")
    if "unsupported" not in unsupported_policy.lower():
        fail("unsupported_policy must describe explicit unsupported handling")
    check_capabilities(artifact.get("capability_matrix"))
    workloads = artifact.get("workloads")
    if not isinstance(workloads, list) or len(workloads) != len(EXPECTED_WORKLOADS):
        fail("artifact must contain exactly four workload rows")
    summaries = [check_workload(workload, index) for index, workload in enumerate(workloads)]
    ids = [summary["id"] for summary in summaries]
    if set(ids) != set(EXPECTED_WORKLOADS):
        fail("workload rows are duplicated or incomplete")
    check = {
        "schema": "sonium-solvers-iterative-cuda-throughput-check/v2",
        "status": "passed",
        "artifact": artifact_path.name,
        "workload_count": len(summaries),
        "native_capability_cells": 16,
        "workloads": summaries,
        "budgets": {
            "real_cpu_cuda_parity": 2.0e-6,
            "complex_cpu_cuda_parity": 2.0e-5,
            "independent_final_relative_residual": 2.0e-7,
            "vendor_baseline_comparison": "not claimed until the vendor numerical gate passes",
        },
    }
    if len(sys.argv) == 4:
        output = Path(sys.argv[3])
        output.write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(check, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
