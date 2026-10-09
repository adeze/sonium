#!/usr/bin/env python3
"""Strict checker for the bounded native-CUDA FEM3 structural action packet."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


SCHEMA = "sonium-compute-cuda-fem-structural/v1"
CASE_ID = "compute.fem-structural-cuda.v1"
NATIVE_ROWS = {
    "h1.tetrahedron.p1.linear-elasticity.assembled.f64.forward",
    "h1.tetrahedron.p1.linear-elasticity.assembled.f64.transpose",
    "h1.tetrahedron.p1.linear-elasticity.assembled.f64.hermitian",
}
UNSUPPORTED_ROWS = {
    "h1.tetrahedron.p1.linear-elasticity.assembled.Complex64.forward",
    "h1.tetrahedron.p1.linear-elasticity.assembled.Complex64.hermitian",
    "h1.tetrahedron.p2-or-higher.linear-elasticity.assembled",
    "h1.tri3.p1.linear-elasticity.assembled",
    "h1.tetrahedron.p1.linear-elasticity.partial-assembly",
    "h1.tetrahedron.p1.linear-elasticity.matrix-free",
    "hcurl.tetrahedron.p1.linear-elasticity.assembled",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM3 structural artifact rejected: {message}")


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
        (fields[0].lower() for row in rows if len(fields := row.split()) == 2 and fields[1] == path.name),
        None,
    )
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"checksum sidecar has no digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
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


def check_cell_and_dispatch(artifact: dict) -> None:
    expected = {
        "space": "H1",
        "element": "tetrahedron",
        "order": "P1",
        "physics": "linear-elasticity",
        "mode": "assembled",
        "scalar": "f64",
    }
    if artifact.get("cell") != {**expected, "actions": ["forward", "transpose", "hermitian"]}:
        fail("bounded structural cell metadata is inconsistent")
    dispatch = artifact.get("dispatch")
    if not isinstance(dispatch, dict) or dispatch != {
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback": "none",
        "admission_before_assembly": True,
        "native_device_csr": True,
    }:
        fail("native CUDA dispatch proof is missing or claims fallback")
    matrix = artifact.get("matrix")
    if not isinstance(matrix, dict) or matrix.get("dofs", 0) <= 0 or matrix.get("nnz", 0) <= 0:
        fail("assembled structural CSR dimensions are missing")
    if "ElasticityForm" not in matrix.get("assembly", ""):
        fail("matrix provenance does not identify production ElasticityForm assembly")


def check_capability_rows(artifact: dict) -> None:
    rows = artifact.get("capability_rows")
    if not isinstance(rows, list):
        fail("structural capability rows are missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if set(by_id) != NATIVE_ROWS | UNSUPPORTED_ROWS:
        fail("capability rows do not cover exactly the bounded structural matrix")
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


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime counters are missing")
    if finite(runtime.get("kernel_launches"), "runtime.kernel_launches") + finite(runtime.get("library_operations"), "runtime.library_operations") <= 0:
        fail("runtime has no native CUDA operations")
    if finite(runtime.get("allocation_high_water_bytes"), "runtime.allocation_high_water_bytes") <= 0:
        fail("runtime has no device allocation high-water mark")
    if finite(runtime.get("h2d_operations"), "runtime.h2d_operations") <= 0 or finite(runtime.get("d2h_operations"), "runtime.d2h_operations") <= 0:
        fail("runtime does not prove input/output transfer")
    warm = artifact.get("warm_residency")
    if not isinstance(warm, dict) or warm.get("operator_storage") != "device-resident" or warm.get("input_vectors") != "device-resident":
        fail("warm residency provenance is missing")
    if warm.get("warm_action_count") != 2 or warm.get("h2d_operations") != 0:
        fail("warm actions are not resident or unexpectedly re-uploaded inputs")


def check_thresholds(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    thresholds = artifact.get("thresholds")
    directions = artifact.get("threshold_directions")
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict) or not isinstance(directions, dict):
        fail("metrics, thresholds, and directions are missing")
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
    if finite(artifact.get("workload_elapsed_ms"), "workload_elapsed_ms") <= 0:
        fail("workload_elapsed_ms must be positive")


def check_failure_contracts(artifact: dict) -> None:
    rows = artifact.get("failure_contracts")
    if not isinstance(rows, list):
        fail("failure contracts are missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if set(by_id) != {"launch.output_preservation_and_retry", "malformed_csr.admission"}:
        fail("failure contract IDs are incomplete")
    launch = by_id["launch.output_preservation_and_retry"]
    if launch.get("status") != "typed_failure_recovered" or launch.get("error") != "BackendError::Launch" or launch.get("fallback") != "none" or launch.get("output_preserved") is not True or launch.get("retry_succeeded") is not True:
        fail("launch failure contract does not prove typed recovery and output preservation")
    malformed = by_id["malformed_csr.admission"]
    if malformed.get("status") != "typed_rejection" or malformed.get("rejected") is not True or malformed.get("fallback") != "none":
        fail("malformed CSR contract is missing typed rejection")
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
    if not isinstance(artifact, dict):
        fail("artifact is not an object")
    if artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID or artifact.get("status") != "passed":
        fail("schema, case identifier, or status is not FEM3 structural")
    check_hardware(artifact)
    check_cell_and_dispatch(artifact)
    check_capability_rows(artifact)
    check_runtime(artifact)
    check_thresholds(artifact)
    check_failure_contracts(artifact)
    check_checksum(path)
    print(f"FEM3 native CUDA structural artifact passed strict checks: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
