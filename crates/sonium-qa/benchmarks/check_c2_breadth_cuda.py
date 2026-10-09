#!/usr/bin/env python3
"""Strict checker for the bounded resident Complex32 C2 breadth cell."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "compute.cuda-c2-breadth.c32-resident-bem.v1"
SCHEMA = "sonium-compute-cuda-c2-breadth-evidence/v1"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"C2 breadth artifact rejected: {message}")


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not an integer")
    measured = int(value)
    if not math.isfinite(float(value)) or measured != value:
        fail(f"{label} is not an integer")
    return measured


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not finite")
    measured = float(value)
    if not math.isfinite(measured):
        fail(f"{label} is not finite")
    return measured


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
    expected = next(
        (fields[0].lower() for row in rows if len(fields := row.split()) == 2 and fields[1] == path.name),
        None,
    )
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
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


def check_runtime(runtime: object, label: str) -> None:
    if not isinstance(runtime, dict):
        fail(f"{label} runtime is missing")
    if integer(runtime.get("kernel_launches"), f"{label}.kernel_launches") != 1:
        fail(f"{label} must contain exactly one native kernel launch")
    if integer(runtime.get("allocations"), f"{label}.allocations") != 0:
        fail(f"{label} allocated during resident action")
    if integer(runtime.get("synchronization_calls"), f"{label}.synchronization_calls") < 1:
        fail(f"{label} has no checked synchronization")
    transfers = runtime.get("transfers")
    if not isinstance(transfers, dict):
        fail(f"{label}.transfers are missing")
    for key in (
        "h2d_operations",
        "d2h_operations",
        "d2d_operations",
        "staging_allocation_operations",
    ):
        if integer(transfers.get(key), f"{label}.transfers.{key}") != 0:
            fail(f"{label} used a host transfer or staging allocation")


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

    backend = artifact.get("backend")
    if not isinstance(backend, dict) or any(
        backend.get(key) != expected
        for key, expected in {
            "requested": "cuda",
            "actual": "cuda",
            "fallback_status": "none",
            "operation": "bem_regular_pair_c32_device",
            "scalar": "Complex32",
            "layout": "contiguous-resident",
        }.items()
    ):
        fail("backend provenance is not the expected resident Complex32 CUDA ABI")

    workload = artifact.get("workload")
    if not isinstance(workload, dict) or integer(workload.get("pair_count"), "workload.pair_count") != 3:
        fail("workload does not prove the required odd three-record tail")
    if workload.get("tail") != "odd-three-record CUDA block tail":
        fail("workload tail description is missing")
    nonempty(workload.get("oracle"), "workload.oracle")

    parity = artifact.get("parity")
    if not isinstance(parity, dict):
        fail("parity record is missing")
    tolerance = finite(parity.get("tolerance"), "parity.tolerance")
    if tolerance <= 0.0 or tolerance > 1.0e-3:
        fail("parity tolerance is not bounded")
    for key in ("initial_max_relative_error", "warm_max_relative_error", "retry_max_relative_error"):
        if finite(parity.get(key), f"parity.{key}") > tolerance:
            fail(f"{key} exceeds declared tolerance")

    admission = artifact.get("admission")
    if not isinstance(admission, dict) or admission.get("malformed_shape_typed") is not True or admission.get("empty_noop") is not True:
        fail("typed malformed-shape and empty admission are not both proven")
    if admission.get("operation") != "bem_regular_pair_c32_device":
        fail("admission operation is inconsistent")

    recovery = artifact.get("failure_recovery")
    if not isinstance(recovery, dict) or any(
        recovery.get(key) is not True for key in ("typed_launch", "output_preserved", "retry")
    ):
        fail("typed launch failure, output preservation, and retry are not all proven")
    if recovery.get("context_recovery") != "reusable resident context":
        fail("recovery disposition is not explicit")
    if recovery.get("publication") != "failed launch publishes no replacement output":
        fail("failure publication disposition is not explicit")

    residency = artifact.get("residency")
    if not isinstance(residency, dict):
        fail("residency record is missing")
    setup = residency.get("setup")
    if not isinstance(setup, dict) or integer(setup.get("allocations"), "residency.setup.allocations") <= 0:
        fail("setup does not prove device allocation")
    setup_transfers = setup.get("transfers")
    if not isinstance(setup_transfers, dict) or integer(setup_transfers.get("h2d_operations"), "residency.setup.transfers.h2d_operations") <= 0:
        fail("setup does not prove the input upload")
    for key in ("initial", "warm", "retry"):
        action = residency.get(key)
        if not isinstance(action, dict) or action.get("status") != "native":
            fail(f"{key} resident action is missing")
        check_runtime(action.get("residency", {}).get("runtime"), f"residency.{key}")
    final = residency.get("final")
    if not isinstance(final, dict) or integer(final.get("allocation_high_water_bytes"), "residency.final.allocation_high_water_bytes") <= 0:
        fail("allocation high-water evidence is missing")

    counts = artifact.get("counts")
    if not isinstance(counts, dict) or {
        key: integer(counts.get(key), f"counts.{key}") for key in ("executed", "passed", "unsupported")
    } != {"executed": 3, "passed": 3, "unsupported": 0}:
        fail("C2 breadth counts are not exactly three passed actions")


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
        fail("QA_RESULT does not prove a passed C2 breadth case")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT backend admission is not native CUDA")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT runtime evidence is missing")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 3 or counts.get("passed") != 3 or counts.get("failed") != 0:
        fail("QA_RESULT counts are not three passed actions")
    counters = runtime.get("counters")
    if not isinstance(counters, dict) or integer(counters.get("launches"), "QA_RESULT.counters.launches") <= 0:
        fail("QA_RESULT does not prove native CUDA launches")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json QA_RESULT.log")
    artifact = Path(sys.argv[1])
    check_artifact(artifact)
    check_log(Path(sys.argv[2]))
    verify_checksum(artifact)
    print(f"C2 breadth native Complex32 CUDA artifact passed strict checks: {artifact}")


if __name__ == "__main__":
    main()
