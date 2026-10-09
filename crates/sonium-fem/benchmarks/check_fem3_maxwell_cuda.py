#!/usr/bin/env python3
"""Strict checker for the bounded native-CUDA FEM3 Maxwell action cell."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


SCHEMA = "sonium-compute-cuda-fem-maxwell/v1"
CASE_ID = "compute.fem-maxwell-cuda.v1"
NATIVE = {
    "hcurl.tri3.p1.maxwell.assembled.Complex64.forward",
    "hcurl.tri3.p1.maxwell.assembled.Complex64.transpose",
    "hcurl.tri3.p1.maxwell.assembled.Complex64.hermitian",
}
UNSUPPORTED = {
    "h1.tri3.p1.maxwell.assembled.Complex64.forward",
    "hcurl.quad.p1.maxwell.assembled.Complex64.forward",
    "hcurl.tri3.p2-or-higher.maxwell.assembled.Complex64.forward",
    "hcurl.tri3.p1.maxwell.assembled.f64.forward",
    "hcurl.tri3.p1.maxwell.assembled.Complex32.forward",
    "hcurl.tri3.p1.maxwell.partial-assembly.Complex64.forward",
    "hcurl.tri3.p1.maxwell.matrix-free.Complex64.forward",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM3 Maxwell artifact rejected: {message}")


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        fail(f"{label} is not finite")
    return value


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} is missing")
    return value


def checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar missing: {error}")
    expected = next(
        (fields[0].lower() for row in rows if len(fields := row.split()) == 2 and fields[1] == path.name),
        None,
    )
    if expected is None or len(expected) != 64:
        fail("checksum sidecar has no SHA-256 row")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch: expected {expected}, got {actual}")


def check_hardware(artifact: dict) -> None:
    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance missing")
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
    if "nvidia" not in hardware["device"].lower() or hardware.get("os") != "linux-x86_64":
        fail("hardware is not native Linux NVIDIA")
    if hardware.get("target_features") != "cuda" or hardware.get("runtime_dispatch") != "cuda-native":
        fail("CUDA dispatch provenance is inconsistent")


def check_dispatch(artifact: dict) -> None:
    expected_cell = {
        "space": "Hcurl",
        "element": "tri3",
        "order": "P1",
        "physics": "maxwell",
        "mode": "assembled",
        "scalar": "Complex64",
        "actions": ["forward", "transpose", "hermitian"],
    }
    if artifact.get("cell") != expected_cell:
        fail("bounded Maxwell cell metadata is inconsistent")
    if artifact.get("dispatch") != {
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback": "none",
        "admission_before_assembly": True,
        "native_device_csr": True,
    }:
        fail("native dispatch proof is missing")
    matrix = artifact.get("matrix")
    if not isinstance(matrix, dict) or matrix.get("dofs", 0) <= 0 or matrix.get("nnz", 0) <= 0:
        fail("assembled Maxwell CSR dimensions are missing")
    if "MaxwellForm" not in nonempty(matrix.get("assembly"), "matrix.assembly"):
        fail("matrix does not identify production MaxwellForm assembly")
    if "independent" not in nonempty(matrix.get("cpu_oracle"), "matrix.cpu_oracle").lower():
        fail("independent CPU oracle is missing")


def check_rows(artifact: dict) -> None:
    rows = artifact.get("capability_rows")
    if not isinstance(rows, list):
        fail("capability rows missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if set(by_id) != NATIVE | UNSUPPORTED:
        fail("capability matrix is not exactly the bounded Maxwell matrix")
    for row_id in NATIVE:
        row = by_id[row_id]
        if row.get("status") != "native" or row.get("backend") != "cuda" or row.get("requested_backend") != "cuda":
            fail(f"{row_id} is not native CUDA")
        if row.get("actual_backend") != "cuda" or row.get("dispatch") is not True or row.get("fallback") != "none":
            fail(f"{row_id} lacks no-fallback dispatch proof")
        nonempty(row.get("evidence"), f"{row_id}.evidence")
    for row_id in UNSUPPORTED:
        row = by_id[row_id]
        if row.get("status") != "unsupported_before_assembly" or row.get("actual_backend") != "none" or row.get("dispatch") is not False or row.get("fallback") != "none":
            fail(f"{row_id} is not fail-closed")
        if "before assembly" not in nonempty(row.get("fallback_reason"), f"{row_id}.fallback_reason"):
            fail(f"{row_id} does not prove pre-assembly rejection")
        nonempty(row.get("reason"), f"{row_id}.reason")


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime counters missing")
    if number(runtime.get("kernel_launches"), "runtime.kernel_launches") + number(runtime.get("library_operations"), "runtime.library_operations") <= 0:
        fail("no native CUDA operations recorded")
    if number(runtime.get("allocation_high_water_bytes"), "runtime.allocation_high_water_bytes") <= 0:
        fail("no CUDA allocation high-water mark")
    if number(runtime.get("h2d_operations"), "runtime.h2d_operations") <= 0 or number(runtime.get("d2h_operations"), "runtime.d2h_operations") <= 0:
        fail("input/output transfers are missing")
    warm = artifact.get("warm_residency")
    if not isinstance(warm, dict) or warm.get("operator_storage") != "device-resident" or warm.get("input_vectors") != "device-resident" or warm.get("forward_transpose_hermitian") != "prebaked-device-resident":
        fail("warm device residency provenance missing")
    if warm.get("warm_action_count") != 3 or warm.get("h2d_operations") != 0 or warm.get("allocation_operations") != 0:
        fail("warm action path re-uploaded or allocated")


def check_thresholds(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    thresholds = artifact.get("thresholds")
    directions = artifact.get("threshold_directions")
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict) or not isinstance(directions, dict):
        fail("metrics, thresholds, and directions missing")
    if set(thresholds) != set(directions) or not set(thresholds).issubset(metrics):
        fail("threshold and direction keys differ or reference a missing metric")
    for key, threshold in thresholds.items():
        value = metrics[key] if isinstance(metrics[key], bool) else number(metrics[key], f"metrics.{key}")
        bound = number(threshold, f"thresholds.{key}")
        direction = directions[key]
        if direction == "maximum" and value > bound:
            fail(f"{key} exceeds maximum")
        if direction == "minimum" and value < bound:
            fail(f"{key} is below minimum")
        if direction not in {"maximum", "minimum"}:
            fail(f"{key} has invalid direction")
    if number(artifact.get("workload_elapsed_ms"), "workload_elapsed_ms") <= 0:
        fail("workload_elapsed_ms must be positive")


def check_failures(artifact: dict) -> None:
    rows = artifact.get("failure_contracts")
    if not isinstance(rows, list):
        fail("failure contracts missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    expected = {"launch.output_preservation_and_retry", "malformed_csr.admission", "vector_shape.admission"}
    if set(by_id) != expected:
        fail("failure contract IDs are incomplete")
    launch = by_id["launch.output_preservation_and_retry"]
    if launch.get("status") != "typed_failure_recovered" or launch.get("error") != "BackendError::Launch" or launch.get("output_preserved") is not True or launch.get("retry_succeeded") is not True or launch.get("fallback") != "none":
        fail("launch recovery contract is incomplete")
    for key in ("malformed_csr.admission", "vector_shape.admission"):
        row = by_id[key]
        if row.get("status") != "typed_rejection" or row.get("rejected") is not True or row.get("fallback") != "none":
            fail(f"{key} does not prove typed rejection")
    for row in rows:
        nonempty(row.get("evidence"), f"failure contract {row.get('id')}.evidence")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID or artifact.get("status") != "passed":
        fail("schema, case identifier, or status is not FEM3 Maxwell")
    check_hardware(artifact)
    check_dispatch(artifact)
    check_rows(artifact)
    check_runtime(artifact)
    check_thresholds(artifact)
    check_failures(artifact)
    gradient = artifact.get("gradient_reference")
    if not isinstance(gradient, dict) or gradient.get("status") != "cpu_finite_difference_only" or gradient.get("device_gradient") != "not_implemented":
        fail("gradient boundary is not stated honestly")
    checksum(path)
    print(f"FEM3 native CUDA Maxwell artifact passed strict checks: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
