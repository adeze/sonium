#!/usr/bin/env python3
"""Strict checker for the bounded B2 resident H-matrix view cell."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "bem.cuda-hmatrix-view-device-residency.v1"
ARTIFACT = "bem_cuda_hmatrix_view_report.json"
TOLERANCE = 1.0e-11


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B2 H-matrix view artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def positive(value: object, label: str) -> None:
    if finite(value, label) <= 0:
        fail(f"{label} must be positive")


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


def check_artifact(artifact: dict) -> None:
    if artifact.get("case_id") != CASE_ID:
        fail("case identifier is not the bounded B2 H-matrix view cell")
    if artifact.get("schema") != "sonium-bem-cuda-hmatrix-view/v1":
        fail("artifact schema is not the frozen view contract")
    scope = artifact.get("claim_scope")
    if not isinstance(scope, str) or "resident Complex64 H-matrix" not in scope:
        fail("claim scope is missing or broadened")
    matrix = artifact.get("matrix")
    if matrix != {"num_dofs": 5, "dense_block": "3x3", "low_rank_block": "2x2 rank-2"}:
        fail("matrix fixture does not match the checked bounded cell")
    views = artifact.get("views")
    if not isinstance(views, dict):
        fail("view metadata is missing")
    if views.get("input") != {"offset": 1, "len": 5, "stride": 3}:
        fail("input view metadata is not the checked offset/stride case")
    if views.get("output") != {"offset": 2, "len": 5, "stride": 2}:
        fail("output view metadata is not the checked offset/stride case")
    if views.get("tail_length") != 5 or not views.get("padding_preserved"):
        fail("tail/padding preservation is not proven")
    if not views.get("malformed_rejected") or not views.get("empty_rejected"):
        fail("malformed and empty view rejection is not proven")
    if artifact.get("actions") != ["forward", "transpose", "hermitian"]:
        fail("all three resident action directions are not listed")
    failure = artifact.get("failure_contract")
    if failure != {"typed_launch": True, "output_preserved": True, "retry_passed": True}:
        fail("typed failure/retry/output-preservation contract is incomplete")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        fail("metrics are missing")
    for key in ("max_action_relative_error", "warm_max_action_relative_error"):
        if finite(metrics.get(key), f"metrics.{key}") > TOLERANCE:
            fail(f"{key} exceeds {TOLERANCE:.1e}")
    for key in ("cuda_kernel_launches", "cuda_allocation_high_water_bytes"):
        positive(metrics.get(key), f"metrics.{key}")
    for key in ("cuda_teardown_live_bytes_before", "cuda_teardown_released_bytes"):
        positive(metrics.get(key), f"metrics.{key}")
    if integer(metrics.get("cuda_teardown_live_bytes_after"), "metrics.cuda_teardown_live_bytes_after") < 0:
        fail("teardown after-byte count is negative")
    warm = artifact.get("warm")
    if not isinstance(warm, dict):
        fail("warm residency metrics are missing")
    if integer(warm.get("replays"), "warm.replays") != 2 or integer(warm.get("rhs_d2d_updates"), "warm.rhs_d2d_updates") != 2:
        fail("warm residency does not prove two RHS refreshes")
    if integer(warm.get("structural_allocations"), "warm.structural_allocations") != 0:
        fail("warm replay allocated structural state")
    if integer(warm.get("structural_h2d_operations"), "warm.structural_h2d_operations") != 0:
        fail("warm replay performed host-to-device staging")
    if integer(warm.get("d2d_operations"), "warm.d2d_operations") < 4 or integer(warm.get("d2h_publications"), "warm.d2h_publications") != 2:
        fail("warm transfer accounting is inconsistent")
    if integer(warm.get("native_kernel_launches"), "warm.native_kernel_launches") <= 0:
        fail("warm replay has no native CUDA launch")
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        fail("capability environment is missing")
    if environment.get("capability.bem.hmatrix.c64") != "native" or environment.get("capability.bem.hmatrix_view.c64") != "native":
        fail("H-matrix/view capability is not native")
    if environment.get("capability.bem.hmatrix_setup.c64") != "host_shared_fallback":
        fail("host-shared setup boundary is not explicit")


def check_log(path: Path) -> None:
    try:
        lines = [line[11:].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("QA_RESULT:")]
    except OSError as error:
        fail(f"QA result log is missing or unreadable: {error}")
    if len(lines) != 1:
        fail("QA result log must contain exactly one QA_RESULT line")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is invalid JSON: {error}")
    if result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT does not prove one passed view cell")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT backend admission is not native CUDA")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("typed runtime evidence is missing")
    counts = runtime.get("counts", {})
    if counts.get("executed") != 1 or counts.get("passed") != 1 or counts.get("failed") != 0:
        fail("QA_RESULT execution counts are not one passed cell")
    counters = runtime.get("counters", {})
    if integer(counters.get("launches"), "runtime.counters.launches") <= 0:
        fail("QA_RESULT does not prove native CUDA launches")
    hardware = runtime.get("hardware", {})
    if not isinstance(hardware, dict):
        fail("CUDA hardware provenance is missing")
    for key in ("device", "driver", "toolkit", "topology", "os", "compiler"):
        nonempty(hardware.get(key), f"runtime.hardware.{key}")
    if "nvidia" not in hardware["device"].lower() or hardware["os"] != "linux-x86_64":
        fail("hardware provenance is not native NVIDIA Linux")
    artifacts = result.get("artifacts", [])
    if artifacts != [ARTIFACT]:
        fail("QA_RESULT artifact declaration is not exact")


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
    print("B2 H-matrix view strict check: PASS")


if __name__ == "__main__":
    main()
