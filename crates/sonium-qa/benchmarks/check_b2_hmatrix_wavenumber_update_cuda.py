#!/usr/bin/env python3
"""Strict checker for the bounded B2 resident H-matrix frequency-update cell."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "bem.cuda-hmatrix-wavenumber-update.v1"
ACTION_TOLERANCE = 1.0e-7
UPDATE_TOLERANCE = 3.0e-5


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B2 H-matrix wavenumber-update artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def integer(value: object, label: str) -> int:
    measured = finite(value, label)
    if measured != int(measured):
        fail(f"{label} is not an integer")
    return int(measured)


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def verify_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"artifact checksum sidecar is missing: {error}")
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


def check_qa_log(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"QA result log is missing: {error}")
    marker = "QA_RESULT:"
    lines = [line[len(marker) :].strip() for line in text.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        fail("QA result log must contain exactly one QA_RESULT line")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is invalid JSON: {error}")
    if result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT does not prove a passed frequency-update case")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT is not admitted as native CUDA")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT runtime evidence is missing")
    hardware = runtime.get("hardware")
    if not isinstance(hardware, dict):
        fail("QA_RESULT hardware provenance is missing")
    for key in ("device", "driver", "toolkit", "topology", "os", "compiler"):
        nonempty(hardware.get(key), f"runtime.hardware.{key}")
    if "nvidia" not in hardware["device"].lower() or hardware["os"] != "linux-x86_64":
        fail("hardware identity is not native NVIDIA Linux")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 1 or counts.get("passed") != 1:
        fail("QA_RESULT execution counts are not one passed case")
    counters = runtime.get("counters")
    if not isinstance(counters, dict) or integer(counters.get("launches"), "runtime.counters.launches") <= 0:
        fail("QA_RESULT does not prove native CUDA launches")


def check_metrics(metrics: object, label: str) -> None:
    if not isinstance(metrics, dict):
        fail(f"{label} are missing")
    bounds = {
        "value_action_relative_error": ACTION_TOLERANCE,
        "derivative_action_relative_error": UPDATE_TOLERANCE,
        "derivative_finite_difference_relative_error": UPDATE_TOLERANCE,
        "initial_update_relative_error": UPDATE_TOLERANCE,
        "retry_update_relative_error": UPDATE_TOLERANCE,
        "warm_update_relative_error": UPDATE_TOLERANCE,
    }
    for key, bound in bounds.items():
        if finite(metrics.get(key), f"{label}.{key}") > bound:
            fail(f"{label}.{key} exceeds {bound:.3e}")
    for key in ("value_low_rank_blocks", "derivative_low_rank_blocks"):
        if integer(metrics.get(key), f"{label}.{key}") <= 0:
            fail(f"{label}.{key} does not prove compressed blocks")
    for key in ("value_compression_ratio", "derivative_compression_ratio"):
        if finite(metrics.get(key), f"{label}.{key}") <= 0.0:
            fail(f"{label}.{key} is not positive")
    if integer(metrics.get("setup_allocations"), f"{label}.setup_allocations") <= 0:
        fail("setup did not allocate resident device state")
    if integer(metrics.get("setup_h2d_operations"), f"{label}.setup_h2d_operations") <= 0:
        fail("setup did not upload resident compressed state")
    if integer(metrics.get("warm_replays"), f"{label}.warm_replays") < 2:
        fail("fewer than two resident frequency updates were exercised")
    if integer(metrics.get("warm_structural_allocations"), f"{label}.warm_structural_allocations") != 0:
        fail("warm frequency update allocated structural state")
    if integer(metrics.get("warm_h2d_operations"), f"{label}.warm_h2d_operations") != 0:
        fail("warm frequency update staged host data")
    if integer(metrics.get("warm_d2d_operations"), f"{label}.warm_d2d_operations") < 4:
        fail("warm updates lack D2D residency accounting")
    if integer(metrics.get("warm_d2h_publications"), f"{label}.warm_d2h_publications") != 2:
        fail("warm updates lack one publication per replay")
    if integer(metrics.get("warm_native_kernel_launches"), f"{label}.warm_native_kernel_launches") <= 0:
        fail("warm updates lack native kernel launches")
    for key in (
        "malformed_rejected",
        "malformed_output_preserved",
        "stale_metadata_rejected",
        "stale_output_preserved",
        "typed_launch_failure",
        "launch_failure_output_preserved",
        "retry_passed",
    ):
        if integer(metrics.get(key), f"{label}.{key}") != 1:
            fail(f"{label}.{key} does not prove the checked failure contract")
    before = integer(metrics.get("cuda_teardown_live_bytes_before"), f"{label}.teardown_before")
    after = integer(metrics.get("cuda_teardown_live_bytes_after"), f"{label}.teardown_after")
    released = integer(metrics.get("cuda_teardown_released_bytes"), f"{label}.teardown_released")
    if before <= 0 or after < 0 or released <= 0 or before - after != released:
        fail("resident teardown accounting is inconsistent")


def check_artifact(path: Path) -> None:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not an object")
    if artifact.get("schema") != "sonium-bem-cuda-hmatrix-wavenumber-update/v1":
        fail("schema is not the B2 frequency-update schema")
    if artifact.get("case_id") != CASE_ID:
        fail("case identifier is not the B2 frequency-update producer")
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        fail("capability environment is missing")
    expected_environment = {
        "capability.bem.hmatrix.c64": "native",
        "capability.bem.hmatrix_wavenumber_update.c64": "native",
        "capability.bem.hmatrix_setup.c64": "host_shared_fallback",
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            fail(f"{key} must be {expected!r}, got {environment.get(key)!r}")
    if "resident Complex64 production H-matrix" not in artifact.get("claim_scope", ""):
        fail("claim scope does not identify the bounded production cell")
    base = finite(artifact.get("base_wavenumber"), "base_wavenumber")
    if base <= 0.0:
        fail("base wavenumber is not positive")
    deltas = artifact.get("delta_wavenumbers")
    if not isinstance(deltas, list) or len(deltas) != 2 or not all(math.isfinite(float(value)) for value in deltas):
        fail("artifact does not record two finite frequency updates")
    matrix = artifact.get("matrix")
    if not isinstance(matrix, dict) or integer(matrix.get("dofs"), "matrix.dofs") != 16:
        fail("matrix fixture dimension is not the bounded 16-panel production cell")
    check_metrics(artifact.get("metrics"), "metrics")
    accuracy = artifact.get("accuracy")
    if not isinstance(accuracy, dict):
        fail("accuracy breakdown is missing")
    check_metrics({**artifact["metrics"], **accuracy}, "accuracy")
    metadata = artifact.get("metadata_contract")
    failure = artifact.get("failure_contract")
    if not isinstance(metadata, dict) or metadata.get("stale_base_rejected") is not True or metadata.get("malformed_dimension_rejected") is not True:
        fail("metadata contract is incomplete")
    if not isinstance(failure, dict) or failure != {
        "typed_launch_failure": True,
        "output_preserved": True,
        "retry_passed": True,
    }:
        fail("failure contract is incomplete")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json QA_RESULT.log")
    artifact = Path(sys.argv[1])
    check_artifact(artifact)
    check_qa_log(Path(sys.argv[2]))
    verify_checksum(artifact)
    print(f"B2 resident H-matrix wavenumber-update strict check: PASS ({artifact})")


if __name__ == "__main__":
    main()
