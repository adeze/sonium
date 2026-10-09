#!/usr/bin/env python3
"""Strict checker for the specialized CUDA C2 operation matrix."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "compute.cuda-specialized-operation-matrix.v1"
SCHEMA = "sonium-compute-cuda-specialized-evidence/v1"
PHASES = {
    "fmm.phase_accumulate.c64",
    "fmm.compact_gather.c64",
    "fmm.compact_scatter.c64",
    "fmm.indexed_dense_matvec.c64",
    "fmm.dense_scatter_matvec.c64",
    "bem.regular_pair.c64",
    "bem.quadrature_pair.c64",
    "bem.field.c64",
    "bem.regular_pair_derivative.c64",
    "bem.regular_pair_derivative.view.c64",
    "bem.regular_pair_source_x_derivative.view.c64",
    "bem.regular_pair_source_y_derivative.view.c64",
    "bem.quadrature_pair_derivative.view.c64",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"C2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def integral(value: object, label: str) -> int:
    measured = finite(value, label)
    if measured != int(measured):
        fail(f"{label} is not an integer")
    return int(measured)


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
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


def check_hardware(artifact: dict) -> None:
    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance is missing")
    for key in (
        "device",
        "driver",
        "toolkit",
        "topology",
        "os",
        "compiler",
        "target_features",
        "runtime_dispatch",
        "thread_policy",
        "affinity_policy",
        "numa_policy",
        "deterministic_mode",
    ):
        nonempty(hardware.get(key), f"hardware.{key}")
    if "nvidia" not in hardware["device"].lower() or hardware["os"] != "linux-x86_64":
        fail("hardware provenance is not native NVIDIA Linux")
    if hardware["target_features"] != "cuda" or hardware["runtime_dispatch"] != "cuda-native":
        fail("CUDA target/dispatch provenance is inconsistent")


def check_transfers(runtime: object, label: str) -> None:
    if not isinstance(runtime, dict):
        fail(f"{label} runtime is missing")
    if integral(runtime.get("kernel_launches"), f"{label}.kernel_launches") + integral(
        runtime.get("library_operations"), f"{label}.library_operations"
    ) <= 0:
        fail(f"{label} does not prove native CUDA work")
    if integral(runtime.get("allocations"), f"{label}.allocations") != 0:
        fail(f"{label} allocated during a resident action")
    transfers = runtime.get("transfers")
    if not isinstance(transfers, dict):
        fail(f"{label}.transfers are missing")
    for key in (
        "h2d_operations",
        "d2h_operations",
        "d2d_operations",
        "staging_allocation_operations",
    ):
        if integral(transfers.get(key), f"{label}.transfers.{key}") != 0:
            fail(f"{label} used a host transfer or staging allocation")


def check_quadrature(row: dict) -> None:
    tolerance = finite(row.get("tolerance"), "quadrature.tolerance")
    for key in ("relative_error", "warm_relative_error", "retry_relative_error"):
        if finite(row.get(key), f"quadrature.{key}") > tolerance:
            fail(f"quadrature.{key} exceeds declared tolerance")
    typed = row.get("typed_rejections")
    if not isinstance(typed, dict) or not all(
        typed.get(key) is True for key in ("malformed_view", "empty_samples", "empty_view_noop")
    ):
        fail("quadrature does not prove malformed/empty checked admission")
    recovery = row.get("failure_recovery")
    if not isinstance(recovery, dict) or recovery.get("typed") is not True or recovery.get(
        "output_preserved"
    ) is not True:
        fail("quadrature does not prove typed failure recovery and output preservation")
    replay = row.get("warm_replay")
    if not isinstance(replay, dict):
        fail("quadrature warm replay record is missing")
    for key in ("initial", "warm", "retry"):
        action = replay.get(key)
        if not isinstance(action, dict):
            fail(f"quadrature warm replay action {key} is missing")
        check_transfers(action.get("residency", {}).get("runtime"), f"quadrature.{key}")


def check_field(row: dict) -> None:
    tolerance = finite(row.get("tolerance"), "field.tolerance")
    for key in ("relative_error", "warm_relative_error", "retry_relative_error"):
        if finite(row.get(key), f"field.{key}") > tolerance:
            fail(f"field.{key} exceeds declared tolerance")
    typed = row.get("typed_rejections")
    if not isinstance(typed, dict) or not all(
        typed.get(key) is True for key in ("malformed_view", "empty_points_noop", "empty_elements")
    ):
        fail("field does not prove malformed/empty checked admission")
    recovery = row.get("failure_recovery")
    if not isinstance(recovery, dict) or recovery.get("typed") is not True or recovery.get(
        "output_preserved"
    ) is not True:
        fail("field does not prove typed failure recovery and output preservation")
    replay = row.get("warm_replay")
    if not isinstance(replay, dict):
        fail("field warm replay record is missing")
    for key in ("initial", "warm", "retry"):
        action = replay.get(key)
        if not isinstance(action, dict):
            fail(f"field warm replay action {key} is missing")
        check_transfers(action.get("residency", {}).get("runtime"), f"field.{key}")


def check_derivative_view(
    row: dict, *, empty_key: str = "empty_output", label: str = "derivative_view"
) -> None:
    tolerance = finite(row.get("tolerance"), f"{label}.tolerance")
    for key in (
        "relative_error",
        "warm_relative_error",
        "retry_relative_error",
        "centered_reference_relative_error",
    ):
        if finite(row.get(key), f"{label}.{key}") > tolerance:
            fail(f"{label}.{key} exceeds declared tolerance")
    typed = row.get("typed_rejections")
    if not isinstance(typed, dict) or not all(
        typed.get(key) is True for key in ("malformed_view", empty_key, "empty_view_noop")
    ):
        fail(f"{label} does not prove malformed/empty checked admission")
    recovery = row.get("failure_recovery")
    if not isinstance(recovery, dict) or recovery.get("typed") is not True or recovery.get(
        "output_preserved"
    ) is not True:
        fail(f"{label} does not prove typed failure recovery and output preservation")
    replay = row.get("warm_replay")
    if not isinstance(replay, dict):
        fail(f"{label} warm replay record is missing")
    for key in ("initial", "warm", "retry"):
        action = replay.get(key)
        if not isinstance(action, dict):
            fail(f"{label} warm replay action {key} is missing")
        check_transfers(action.get("residency", {}).get("runtime"), f"{label}.{key}")


def check_artifact(path: Path) -> None:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not a JSON object")
    if artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID:
        fail("schema or case identifier is incorrect")
    if artifact.get("status") != "passed":
        fail("artifact is not marked passed")
    check_hardware(artifact)
    counts = artifact.get("counts")
    if not isinstance(counts, dict) or {
        key: integral(counts.get(key), f"counts.{key}")
        for key in ("executed", "passed", "failed", "unsupported")
    } != {"executed": 13, "passed": 13, "failed": 0, "unsupported": 0}:
        fail("specialized matrix counts are not exactly thirteen passed cells")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 13:
        fail("specialized matrix must contain exactly thirteen rows")
    by_phase = {row.get("phase"): row for row in rows if isinstance(row, dict)}
    if set(by_phase) != PHASES:
        fail(f"specialized matrix phases differ: {sorted(by_phase)}")
    for phase, row in by_phase.items():
        if row.get("status") != "native" or row.get("scalar") != "Complex64":
            fail(f"{phase} is not a native Complex64 row")
        check_transfers(row.get("residency", {}).get("runtime"), phase)
    check_quadrature(by_phase["bem.quadrature_pair.c64"])
    check_field(by_phase["bem.field.c64"])
    check_derivative_view(by_phase["bem.regular_pair_derivative.view.c64"])
    check_derivative_view(by_phase["bem.regular_pair_source_x_derivative.view.c64"])
    check_derivative_view(by_phase["bem.regular_pair_source_y_derivative.view.c64"])
    tail = by_phase["bem.regular_pair_source_y_derivative.view.c64"].get("tail_case")
    if not isinstance(tail, dict) or tail.get("logical_len") != 3 or tail.get("non_block_aligned") is not True:
        fail("source-y derivative does not prove the odd tail case")
    check_derivative_view(
        by_phase["bem.quadrature_pair_derivative.view.c64"],
        empty_key="empty_samples",
        label="quadrature_derivative_view",
    )


def check_log(path: Path) -> None:
    try:
        lines = [line[len("QA_RESULT:") :].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("QA_RESULT:")]
    except OSError as error:
        fail(f"QA result log is missing or unreadable: {error}")
    if len(lines) != 1:
        fail("QA result log must contain exactly one QA_RESULT line")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is invalid JSON: {error}")
    if not isinstance(result, dict) or result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT does not prove a passed C2 case")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT backend admission is not native CUDA")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT runtime evidence is missing")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 13 or counts.get("passed") != 13 or counts.get("failed") != 0:
        fail("QA_RESULT execution counts are not thirteen passed cases")
    counters = runtime.get("counters")
    if not isinstance(counters, dict) or integral(counters.get("launches"), "QA_RESULT.counters.launches") <= 0:
        fail("QA_RESULT does not prove native CUDA launches")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json QA_RESULT.log")
    artifact = Path(sys.argv[1])
    check_artifact(artifact)
    check_log(Path(sys.argv[2]))
    verify_checksum(artifact)
    print(f"C2 specialized native CUDA artifact passed strict checks: {artifact}")


if __name__ == "__main__":
    main()
