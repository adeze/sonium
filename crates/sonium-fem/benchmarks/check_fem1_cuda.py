#!/usr/bin/env python3
"""Strict checker for the bounded native-CUDA FEM1 evidence packet."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


SCHEMA = "sonium-compute-cuda-fem-adjoint/v1"
CASE_ID = "compute.fem-adjoint-cuda.v1"
NATIVE_ROWS = {
    "h1.tri3.p1.helmholtz.assembled.f64.forward",
    "h1.tri3.p1.helmholtz.assembled.f64.hermitian",
    "h1.tri3.p1.helmholtz.assembled.f64.transpose",
    "h1.tri3.p1.helmholtz.assembled.Complex64.forward",
    "h1.tri3.p1.helmholtz.assembled.Complex64.hermitian",
}
NATIVE_ROWS |= {
    "h1.tri3.p2.helmholtz.assembled.f64.forward",
    "h1.tri3.p2.helmholtz.assembled.f64.hermitian",
    "h1.tri3.p2.helmholtz.assembled.f64.transpose",
    "h1.tri3.p2.helmholtz.assembled.Complex64.forward",
    "h1.tri3.p2.helmholtz.assembled.Complex64.hermitian",
}
NATIVE_ROWS |= {
    f"h1.tri3.p1.helmholtz.matrix-free.{scalar}.{action}"
    for scalar in ("f64", "Complex64")
    for action in ("forward", "transpose", "hermitian")
}
UNSUPPORTED_ROWS = {
    "h1.tri3.p1.helmholtz.assembled.Complex64.transpose",
    "h1.quad.p1.helmholtz.assembled",
    "h1.tetrahedron.p1.helmholtz.assembled",
    "h1.hexahedron.p1.helmholtz.assembled",
    "h1.tri3.p2-or-higher.helmholtz.assembled",
    "h1.tri3.p1.other-form.assembled",
    "h1.tri3.p1.helmholtz.partial-assembly",
    "hcurl.tri3.p1.helmholtz.assembled",
    "hdiv.tri3.p1.helmholtz.assembled",
    "dg.tri3.p1.helmholtz.assembled",
    "h1.tri3.p1.helmholtz.assembled.f32",
    "h1.tri3.p1.helmholtz.assembled.Complex32",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM1 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{label} is not finite")
    return number


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
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
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


def check_capability_rows(artifact: dict) -> None:
    rows = artifact.get("capability_rows")
    if not isinstance(rows, list):
        fail("structured capability rows are missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    expected = NATIVE_ROWS | UNSUPPORTED_ROWS
    if set(by_id) != expected:
        fail("capability row IDs do not cover the bounded FEM1 matrix")
    for row_id in NATIVE_ROWS:
        row = by_id[row_id]
        if row.get("status") not in {"native", "native_real_equivalent"}:
            fail(f"{row_id} is not marked native")
        expected_mode = "matrix-free" if ".matrix-free." in row_id else "assembled"
        if (
            row.get("backend") != "cuda"
            or row.get("mode") != expected_mode
            or row.get("requested_backend") != "cuda"
            or row.get("actual_backend") != "cuda"
            or row.get("dispatch") is not True
        ):
            fail(f"{row_id} lacks native CUDA provenance")
        if row.get("fallback") != "none":
            fail(f"{row_id} silently advertises a fallback")
        nonempty(row.get("fallback_reason"), f"{row_id}.fallback_reason")
        nonempty(row.get("evidence"), f"{row_id}.evidence")
    for row_id in UNSUPPORTED_ROWS:
        row = by_id[row_id]
        if row.get("status") != "unsupported_before_assembly":
            fail(f"{row_id} is not fail-closed before assembly")
        if (
            row.get("backend") != "cuda"
            or row.get("requested_backend") != "cuda"
            or row.get("actual_backend") != "none"
            or row.get("dispatch") is not False
        ):
            fail(f"{row_id} has invalid unsupported dispatch metadata")
        if row.get("fallback") != "none":
            fail(f"{row_id} silently advertises a fallback")
        fallback_reason = nonempty(row.get("fallback_reason"), f"{row_id}.fallback_reason")
        if "before assembly" not in fallback_reason:
            fail(f"{row_id} fallback provenance does not prove pre-assembly rejection")
        nonempty(row.get("reason"), f"{row_id}.reason")


def check_failure_contracts(artifact: dict) -> None:
    rows = artifact.get("failure_contracts")
    if not isinstance(rows, list):
        fail("failure contracts are missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    expected = {"required_cuda.initialization", "unsupported_cell.admission"}
    if set(by_id) != expected:
        fail("failure contract IDs are incomplete")
    required = by_id["required_cuda.initialization"]
    if (
        required.get("status") != "fail_closed"
        or required.get("stage") != "initialization"
        or required.get("requested_backend") != "cuda"
        or required.get("actual_backend") != "unavailable"
        or required.get("actual_backend_on_failure") != "unavailable"
        or required.get("fallback") != "none"
        or required.get("dispatch") is not False
    ):
        fail("required-CUDA initialization contract is not fail-closed")
    nonempty(required.get("fallback_reason"), "required-CUDA fallback_reason")
    unsupported = by_id["unsupported_cell.admission"]
    if (
        unsupported.get("status") != "reject_before_assembly"
        or unsupported.get("stage") != "admission"
        or unsupported.get("requested_backend") != "cuda"
        or unsupported.get("actual_backend") != "none"
        or unsupported.get("actual_backend_on_failure") != "none"
        or unsupported.get("fallback") != "none"
        or unsupported.get("dispatch") is not False
    ):
        fail("unsupported-cell admission contract is not fail-closed")
    fallback_reason = nonempty(unsupported.get("fallback_reason"), "unsupported-cell fallback_reason")
    if "before assembly" not in fallback_reason:
        fail("unsupported-cell fallback provenance does not prove pre-assembly rejection")
    for row in rows:
        nonempty(row.get("evidence"), f"failure contract {row.get('id')}.evidence")


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime counters are missing")
    kernel_launches = finite(runtime.get("kernel_launches"), "runtime.kernel_launches")
    library_operations = finite(runtime.get("library_operations"), "runtime.library_operations")
    if kernel_launches + library_operations <= 0:
        fail("runtime has no CUDA kernel or vendor-library operations")
    if finite(runtime.get("allocation_high_water_bytes"), "runtime.allocation_high_water_bytes") <= 0:
        fail("runtime has no device allocation high-water mark")
    for key in ("h2d_operations", "d2h_operations"):
        if finite(runtime.get(key), f"runtime.{key}") <= 0:
            fail(f"runtime.{key} does not prove data movement")
    for key in ("live_allocation_bytes", "synchronization_calls"):
        if finite(runtime.get(key), f"runtime.{key}") < 0:
            fail(f"runtime.{key} is negative")


def check_thresholds(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    thresholds = artifact.get("thresholds")
    directions = artifact.get("threshold_directions")
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict) or not isinstance(directions, dict):
        fail("metrics, thresholds, and directions are missing")
    if set(thresholds) != set(directions) or not set(thresholds).issubset(metrics):
        fail("threshold and metric keys differ")
    for key, threshold in thresholds.items():
        value = finite(metrics[key], f"metrics.{key}")
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
    if artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID:
        fail("schema or case identifier is not FEM1")
    if artifact.get("status") != "passed":
        fail("artifact is not a passed result")
    check_hardware(artifact)
    check_capability_rows(artifact)
    check_failure_contracts(artifact)
    check_runtime(artifact)
    check_thresholds(artifact)
    check_checksum(path)
    print(f"FEM1 native CUDA artifact passed strict checks: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
