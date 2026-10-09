#!/usr/bin/env python3
"""Strict checker for the bounded native-CUDA FEM3 Stokes action packet."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


SCHEMA = "sonium-compute-cuda-fem-stokes/v1"
CASE_ID = "compute.fem-stokes-cuda.v1"
TOLERANCE = 3.0e-12
NATIVE_ROWS = {
    "h1.tri3.p2.stokes.assembled.f64.forward",
    "h1.tri3.p2.stokes.assembled.f64.transpose",
    "h1.tri3.p2.stokes.assembled.f64.hermitian",
}
UNSUPPORTED_ROWS = {
    "h1.tri3.p1.stokes.assembled.f64.forward",
    "h1.tri3.p2-or-higher.stokes.assembled.f64.forward",
    "h1.tri3.p2.stokes.assembled.Complex64.forward",
    "h1.tri3.p2.stokes.partial-assembly.f64.forward",
    "h1.tri3.p2.stokes.matrix-free.f64.forward",
    "h1.quad.p2.stokes.assembled.f64.forward",
    "h1.tri3.p2.other-form.assembled.f64.forward",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM3 Stokes artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{label} is not finite")
    return number


def finite_or_bool(value: object, label: str) -> float:
    if isinstance(value, bool):
        return float(value)
    return finite(value, label)


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} is missing")
    return value


def check_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar is missing: {error}")
    expected = next(
        (
            fields[0].lower()
            for row in rows
            if len(fields := row.split()) == 2 and fields[1] == path.name
        ),
        None,
    )
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"checksum sidecar has no digest for {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        fail(f"checksum mismatch for {path.name}")


def check_hardware(artifact: dict) -> None:
    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance is missing")
    for key in (
        "device",
        "driver",
        "toolkit",
        "topology",
        "cpu_model",
        "microcode",
        "compiler",
        "target_features",
        "runtime_dispatch",
        "thread_policy",
        "affinity_policy",
        "numa_policy",
        "deterministic_mode",
    ):
        nonempty(hardware.get(key), f"hardware.{key}")
    if "nvidia" not in hardware["device"].lower():
        fail("hardware.device is not NVIDIA")
    if hardware.get("os") != "linux-x86_64":
        fail("hardware.os is not native Linux x86_64")
    if hardware.get("target_features") != "cuda" or hardware.get("runtime_dispatch") != "cuda-native":
        fail("CUDA target/dispatch provenance is inconsistent")


def check_dispatch_and_cell(artifact: dict) -> None:
    expected_cell = {
        "space": "H1",
        "element": "Tri3",
        "order": "P2-P1 Taylor-Hood",
        "physics": "Stokes",
        "mode": "assembled",
        "scalar": "f64",
        "actions": ["forward", "transpose", "hermitian"],
    }
    if artifact.get("cell") != expected_cell:
        fail("cell metadata is not the bounded Taylor-Hood Stokes pair")
    dispatch = artifact.get("dispatch")
    if dispatch != {
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback": "none",
        "admission_before_assembly": True,
        "native_device_csr": True,
    }:
        fail("dispatch does not prove required native CUDA admission")
    assembly = artifact.get("assembly")
    if not isinstance(assembly, dict):
        fail("assembly provenance is missing")
    if "assemble_mixed_checked" not in nonempty(assembly.get("production"), "assembly.production"):
        fail("production mixed Stokes assembly is not identified")
    if "independent direct quadrature" not in nonempty(assembly.get("independent_reference"), "assembly.independent_reference"):
        fail("independent quadrature/CSR oracle is not identified")


def check_dimensions(artifact: dict) -> None:
    dimensions = artifact.get("dimensions")
    if not isinstance(dimensions, dict):
        fail("dimensions are missing")
    if finite(dimensions.get("dofs"), "dimensions.dofs") != 59:
        fail("fixture does not prove the expected 59-DOF mixed system")
    if finite(dimensions.get("velocity_dofs"), "dimensions.velocity_dofs") != 50:
        fail("velocity block dimension is not 50")
    if finite(dimensions.get("pressure_dofs"), "dimensions.pressure_dofs") != 9:
        fail("pressure block dimension is not 9")
    if finite(dimensions.get("nnz"), "dimensions.nnz") <= 59:
        fail("assembled CSR is empty or trivial")
    if dimensions.get("odd_dimension") is not True or dimensions.get("tail_case") is not True:
        fail("odd/tail markers are missing")
    if finite(dimensions.get("tail_remainder_mod_256"), "dimensions.tail_remainder_mod_256") == 0:
        fail("fixture is block-aligned and does not exercise a tail")


def check_capability_rows(artifact: dict) -> None:
    rows = artifact.get("capability_rows")
    if not isinstance(rows, list):
        fail("capability rows are missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if set(by_id) != NATIVE_ROWS | UNSUPPORTED_ROWS:
        fail("capability rows do not cover exactly the bounded Stokes matrix")
    for row_id in NATIVE_ROWS:
        row = by_id[row_id]
        if row.get("status") not in {"native", "native_real_equivalent"}:
            fail(f"{row_id} is not native")
        if row.get("backend") != "cuda" or row.get("requested_backend") != "cuda":
            fail(f"{row_id} lacks CUDA provenance")
        if row.get("actual_backend") != "cuda" or row.get("dispatch") is not True or row.get("fallback") != "none":
            fail(f"{row_id} does not prove native dispatch")
        nonempty(row.get("evidence"), f"{row_id}.evidence")
    for row_id in UNSUPPORTED_ROWS:
        row = by_id[row_id]
        if row.get("status") != "unsupported_before_assembly":
            fail(f"{row_id} is not fail-closed")
        if row.get("actual_backend") != "none" or row.get("dispatch") is not False or row.get("fallback") != "none":
            fail(f"{row_id} has invalid unsupported provenance")
        if "before assembly" not in nonempty(row.get("fallback_reason"), f"{row_id}.fallback_reason"):
            fail(f"{row_id} does not prove pre-assembly rejection")
        nonempty(row.get("reason"), f"{row_id}.reason")


def check_metrics(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        fail("metrics are missing")
    for key in (
        "assembly_error",
        "forward_error",
        "transpose_error",
        "inner_product_error",
        "retry_error",
    ):
        if finite(metrics.get(key), f"metrics.{key}") > TOLERANCE:
            fail(f"{key} exceeds CPU parity budget")
    if finite(metrics.get("warm_action_count"), "metrics.warm_action_count") < 2:
        fail("warm action count is too small")
    if finite(metrics.get("warm_kernel_operations"), "metrics.warm_kernel_operations") < 2:
        fail("warm path has no native operations")
    for key in (
        "warm_h2d_operations",
        "warm_d2h_operations",
        "warm_allocation_operations",
        "warm_staging_allocation_operations",
    ):
        if finite(metrics.get(key), f"metrics.{key}") != 0:
            fail(f"warm residency has nonzero {key}")
    for key in (
        "malformed_csr_rejected",
        "malformed_shape_rejected",
        "malformed_output_preserved",
        "failure_typed",
        "retry_succeeded",
    ):
        if metrics.get(key) is not True:
            fail(f"{key} contract did not pass")
    if metrics.get("failure_published") is not False:
        fail("failed launch published a result")


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime counters are missing")
    if finite(runtime.get("kernel_launches"), "runtime.kernel_launches") + finite(runtime.get("library_operations"), "runtime.library_operations") <= 0:
        fail("runtime has no native CUDA operations")
    if finite(runtime.get("allocation_high_water_bytes"), "runtime.allocation_high_water_bytes") <= 0:
        fail("runtime has no allocation high-water mark")
    for key in ("h2d_operations", "d2h_operations"):
        if finite(runtime.get(key), f"runtime.{key}") <= 0:
            fail(f"runtime.{key} does not prove setup/inspection transfer")
    if finite(runtime.get("live_allocation_bytes"), "runtime.live_allocation_bytes") != 0 or finite(runtime.get("live_staging_bytes"), "runtime.live_staging_bytes") != 0:
        fail("runtime resource ledger is not released")


def check_contracts(artifact: dict) -> None:
    warm = artifact.get("warm_residency")
    if not isinstance(warm, dict) or warm.get("mixed_csr") != "device-resident" or warm.get("input_vectors") != "device-resident" or warm.get("output_vectors") != "device-resident" or warm.get("warm_action_count") != 2 or warm.get("h2d_operations") != 0 or warm.get("d2h_operations") != 0:
        fail("warm residency provenance is incomplete")
    failure = artifact.get("failure_contract")
    if not isinstance(failure, dict) or failure.get("stage") != "launch" or failure.get("error") != "BackendError::Launch" or failure.get("context_disposition") != "SynchronizeBeforeReuse" or failure.get("output_disposition") != "Unusable" or failure.get("typed") is not True or failure.get("partial_result_published") is not False or failure.get("recovery") != "synchronize_before_reuse" or failure.get("retry_succeeded") is not True:
        fail("typed launch failure/recovery contract is incomplete")
    malformed = artifact.get("malformed_input_contract")
    if not isinstance(malformed, dict) or malformed.get("csr_stage") != "before_upload" or malformed.get("csr_error") != "HostCsrError::RowPointerCount" or malformed.get("shape_stage") != "action_admission" or malformed.get("shape_error") != "BackendError::InvalidArgument" or malformed.get("rejected") is not True or malformed.get("output_preserved") is not True:
        fail("malformed-input contract is incomplete")
    teardown = artifact.get("teardown")
    if not isinstance(teardown, dict) or finite(teardown.get("live_before_bytes"), "teardown.live_before_bytes") <= 0 or finite(teardown.get("live_after_bytes"), "teardown.live_after_bytes") != 0 or finite(teardown.get("staging_after_bytes"), "teardown.staging_after_bytes") != 0 or teardown.get("resources_released") is not True:
        fail("teardown resource ledger did not return to zero")


def check_thresholds(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    thresholds = artifact.get("thresholds")
    directions = artifact.get("threshold_directions")
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict) or not isinstance(directions, dict):
        fail("threshold metadata is missing")
    if set(thresholds) != set(directions) or not set(thresholds).issubset(metrics):
        fail("threshold and metric keys differ")
    for key, threshold in thresholds.items():
        value = finite_or_bool(metrics[key], f"metrics.{key}")
        bound = finite(threshold, f"thresholds.{key}")
        direction = directions[key]
        if direction == "maximum" and value > bound:
            fail(f"{key} exceeds maximum threshold")
        if direction == "minimum" and value < bound:
            fail(f"{key} is below minimum threshold")
        if direction not in {"maximum", "minimum"}:
            fail(f"{key} has invalid threshold direction")


def check_artifact(path: Path) -> None:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID or artifact.get("status") != "passed":
        fail("schema, case identifier, or status is not FEM3 Stokes")
    check_hardware(artifact)
    check_dispatch_and_cell(artifact)
    check_dimensions(artifact)
    check_capability_rows(artifact)
    check_metrics(artifact)
    check_runtime(artifact)
    check_contracts(artifact)
    check_thresholds(artifact)
    if finite(artifact.get("workload_elapsed_ms"), "workload_elapsed_ms") <= 0:
        fail("workload_elapsed_ms must be positive")
    check_checksum(path)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    check_artifact(path)
    print(f"FEM3 native CUDA Stokes artifact passed strict checks: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
