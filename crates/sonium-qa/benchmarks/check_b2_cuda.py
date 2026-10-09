#!/usr/bin/env python3
"""Strict checker for native CUDA B2 resident H-matrix/FMM evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "bem.cuda-hmatrix-fmm-device-residency.v1"
TOLERANCE = 1.0e-10
DENSE_RESIDUAL_TOLERANCE = 1.0e-8


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def verify_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        fail(f"checksum sidecar has no valid digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}")


def check_log(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"QA result log is missing or unreadable: {error}")
    marker = "QA_RESULT:"
    lines = [line[len(marker) :].strip() for line in text.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        fail("QA result log must contain exactly one QA_RESULT line")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is invalid JSON: {error}")
    if result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT does not prove a passed B2 case")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT backend admission is not native CUDA")
    hardware = result.get("runtime", {}).get("hardware", {})
    if not isinstance(hardware, dict):
        fail("QA_RESULT hardware provenance is missing")
    for key in ("device", "driver", "toolkit", "topology", "os", "compiler"):
        nonempty(hardware.get(key), f"QA_RESULT.runtime.hardware.{key}")
    if "nvidia" not in hardware["device"].lower() or hardware["os"] != "linux-x86_64":
        fail("QA_RESULT hardware identity is not native NVIDIA Linux")
    runtime = result.get("runtime", {})
    counts = runtime.get("counts", {})
    if counts.get("executed") != 1 or counts.get("passed") != 1 or counts.get("failed") != 0:
        fail("QA_RESULT execution counts are not one passed case")
    counters = runtime.get("counters", {})
    if not isinstance(counters, dict) or counters.get("launches", 0) <= 0:
        fail("QA_RESULT does not prove native CUDA launches")
    check_warm_metrics(result.get("metrics"), "QA_RESULT.metrics")


def integral(value: object, label: str) -> int:
    measured = finite(value, label)
    if measured != int(measured):
        fail(f"{label} is not an integer")
    return int(measured)


def check_warm_metrics(metrics: object, label: str) -> None:
    if not isinstance(metrics, dict):
        fail(f"{label} are missing")
    for prefix in ("mlfmm", "hmatrix"):
        replays = integral(metrics.get(f"{prefix}_warm_replays"), f"{label}.{prefix}_warm_replays")
        updates = integral(metrics.get(f"{prefix}_warm_rhs_updates"), f"{label}.{prefix}_warm_rhs_updates")
        allocations = integral(
            metrics.get(f"{prefix}_warm_structural_allocations"),
            f"{label}.{prefix}_warm_structural_allocations",
        )
        h2d = integral(
            metrics.get(f"{prefix}_warm_structural_h2d_operations"),
            f"{label}.{prefix}_warm_structural_h2d_operations",
        )
        d2d = integral(
            metrics.get(f"{prefix}_warm_rhs_d2d_operations"),
            f"{label}.{prefix}_warm_rhs_d2d_operations",
        )
        d2h = integral(
            metrics.get(f"{prefix}_warm_d2h_publications"),
            f"{label}.{prefix}_warm_d2h_publications",
        )
        error = finite(
            metrics.get(f"{prefix}_warm_max_action_relative_error"),
            f"{label}.{prefix}_warm_max_action_relative_error",
        )
        recovery = integral(
            metrics.get(f"{prefix}_warm_failure_recovery"),
            f"{label}.{prefix}_warm_failure_recovery",
        )
        preserved = integral(
            metrics.get(f"{prefix}_warm_failure_output_preserved"),
            f"{label}.{prefix}_warm_failure_output_preserved",
        )
        if replays < 2 or updates != replays:
            fail(f"{label}.{prefix} does not prove repeated RHS updates")
        if allocations != 0 or h2d != 0:
            fail(f"{label}.{prefix} warm replay allocated or staged structure")
        if d2d < updates or d2h != replays:
            fail(f"{label}.{prefix} warm transfer accounting is inconsistent")
        if not math.isfinite(error) or error > TOLERANCE:
            fail(f"{label}.{prefix} warm parity exceeds tolerance")
        if recovery != 1 or preserved != 1:
            fail(f"{label}.{prefix} typed failure recovery is not proven")

    before = integral(metrics.get("cuda_teardown_live_bytes_before"), f"{label}.cuda_teardown_live_bytes_before")
    after = integral(metrics.get("cuda_teardown_live_bytes_after"), f"{label}.cuda_teardown_live_bytes_after")
    released = integral(metrics.get("cuda_teardown_released_bytes"), f"{label}.cuda_teardown_released_bytes")
    if before <= 0 or after < 0 or released <= 0 or before - after != released:
        fail(f"{label} teardown accounting does not prove resident release")


def check_artifact(artifact: dict) -> None:
    if artifact.get("case_id") != CASE_ID:
        fail("case identifier is not the B2 producer")
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        fail("capability environment is missing")
    expected_environment = {
        "capability.bem.hmatrix.c64": "native",
        "capability.bem.hmatrix_setup.c64": "host_shared_fallback",
        "capability.bem.fmm.c64": "native",
        "capability.bem.fmm_setup.c64": "host_shared_fallback",
        "capability.bem.mlfmm.c64": "native",
        "capability.bem.mlfmm_setup.c64": "host_shared_fallback",
        "capability.bem.field_evaluation.c64": "native",
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            fail(f"{key} must be {expected!r}, got {environment.get(key)!r}")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        fail("metrics are missing")
    check_warm_metrics(metrics, "metrics")
    maximums = {
        "mlfmm_max_action_relative_error": TOLERANCE,
        "max_action_relative_error": TOLERANCE,
        "dense_residual_relative_error": DENSE_RESIDUAL_TOLERANCE,
        "compressed_residual_relative_error": DENSE_RESIDUAL_TOLERANCE,
        "adjoint_identity_error": TOLERANCE,
        "reciprocity_identity_error": TOLERANCE,
        "field_relative_error": TOLERANCE,
    }
    for key, bound in maximums.items():
        measured = finite(metrics.get(key), f"metrics.{key}")
        if measured > bound:
            fail(f"{key}={measured:.6e} exceeds {bound:.6e}")
    positive = (
        "mlfmm_cuda_kernel_launches",
        "cuda_kernel_launches",
        "field_cuda_kernel_launches",
        "cuda_allocation_high_water_bytes",
        "cuda_action_allocation_high_water_bytes",
    )
    for key in positive:
        if finite(metrics.get(key), f"metrics.{key}") <= 0:
            fail(f"{key} must prove positive device work")
    if finite(metrics.get("mlfmm_cuda_h2d_operations_steady_state"), "metrics.mlfmm_cuda_h2d_operations_steady_state") != 0.0:
        fail("MLFMM steady-state actions performed a host-to-device transfer")
    if finite(metrics.get("cuda_h2d_operations_steady_state"), "metrics.cuda_h2d_operations_steady_state") != 0.0:
        fail("H-matrix steady-state actions performed a host-to-device transfer")
    for key in (
        "mlfmm_cuda_d2h_publications",
        "cuda_d2h_publications",
        "field_cuda_h2d_operations",
        "field_cuda_d2h_operations",
    ):
        if finite(metrics.get(key), f"metrics.{key}") <= 0:
            fail(f"{key} must record a publication/field transfer")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json QA_RESULT.log")
    artifact_path = Path(sys.argv[1])
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not a JSON object")
    check_artifact(artifact)
    check_log(Path(sys.argv[2]))
    verify_checksum(artifact_path)
    print(f"B2 native CUDA artifact passed strict checks: {artifact_path}")


if __name__ == "__main__":
    main()
