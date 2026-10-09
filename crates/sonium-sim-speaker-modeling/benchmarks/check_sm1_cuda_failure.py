#!/usr/bin/env python3
"""Strict checker for the SM1 modeling CUDA failure/recovery cell.

The checker accepts only the two native one-shot lanes emitted by the
production modeling seam. A successful retry alone is insufficient: each
record must retain typed stage/context provenance, a zero partial-publication
claim, allocation-ledger unwind observations, and independent CPU/f64 parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "speaker.cuda-bounded-coupled-failure-contract.v1"
HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")
SUBSYSTEMS = {
    "magnetic",
    "structural",
    "acoustic",
    "thermal",
    "sparse_coupled",
    "derivative",
    "nonlinear_material",
}
UNSUPPORTED = {"magnetic", "structural", "acoustic", "thermal", "nonlinear_material"}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SM1 failure artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        fail(f"{label} is not finite")
    return value


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    if value.strip().lower() in {"unknown", "unavailable", "unspecified", "n/a"}:
        fail(f"{label} is unavailable")
    return value


def checksum(path: Path) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{path.name} checksum sidecar is missing: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or not HEX_64.fullmatch(expected):
        fail(f"{path.name} checksum sidecar has no SHA-256 digest")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"{path.name} checksum mismatch")
    return actual


def capability_matrix(raw: object) -> None:
    if not isinstance(raw, dict):
        fail("capability matrix is missing")
    if raw.get("requested_backend") != "cuda" or raw.get("actual_backend") != "cuda":
        fail("capability matrix does not prove requested/actual CUDA")
    if raw.get("precision") != "f64":
        fail("capability matrix precision is not f64")
    nonempty(raw.get("device"), "capability.device")
    nonempty(raw.get("toolchain"), "capability.toolchain")
    rows = raw.get("subsystems")
    if not isinstance(rows, list):
        fail("capability subsystem rows are missing")
    by_name = {row.get("subsystem"): row for row in rows if isinstance(row, dict)}
    if set(by_name) != SUBSYSTEMS:
        fail("capability matrix does not cover all seven subsystems")
    for name in UNSUPPORTED:
        row = by_name[name]
        if row.get("state") != "unsupported":
            fail(f"unsupported high-level cell {name} was relabelled")
        nonempty(row.get("reason"), f"capability.{name}.reason")
    for name in ("sparse_coupled", "derivative"):
        row = by_name[name]
        if row.get("state") not in {"native", "vendor_native"}:
            fail(f"{name} is not an admitted native/vendor CUDA cell")
        nonempty(row.get("reason"), f"capability.{name}.reason")


def check_artifact(path: Path, expected_point: str) -> dict:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path.name}: {error}")
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        fail(f"{path.name} schema is not v1")
    if artifact.get("case_id") != CASE_ID or artifact.get("status") != "passed":
        fail(f"{path.name} case/status is not a passed SM1 failure record")
    if artifact.get("requested_backend") != "cuda" or artifact.get("actual_backend") != "cuda":
        fail(f"{path.name} is not admitted as native CUDA")

    injection = artifact.get("injection")
    expected_operation = {"launch": "cuda_launch", "device-loss": "cuda_device_loss"}[expected_point]
    if not isinstance(injection, dict) or (
        injection.get("point") != expected_point
        or injection.get("one_shot") is not True
        or injection.get("native_backend") != "cuda"
        or injection.get("expected_operation") != expected_operation
    ):
        fail(f"{path.name} injection provenance is invalid")
    capability_matrix(artifact.get("capability_matrix"))

    expected_stage, expected_recovery = {
        "launch": ("launch", "synchronize-before-reuse"),
        "device-loss": ("setup", "recreate"),
    }[expected_point]
    failure = artifact.get("failure_diagnostics")
    if not isinstance(failure, dict) or (
        failure.get("backend") != "cuda"
        or failure.get("stage") != expected_stage
        or failure.get("operation") != expected_operation
        or "injected CUDA" not in failure.get("detail", "")
        or failure.get("partial_result_published") is not False
        or failure.get("resources_released_on_unwind") is not True
        or failure.get("context_recovery") != expected_recovery
        or failure.get("output_disposition") != "unusable"
    ):
        fail(f"{path.name} typed failure diagnostics are incomplete")
    nonempty(failure.get("typed_error"), f"{path.name}.failure_diagnostics.typed_error")

    publication = artifact.get("publication")
    if not isinstance(publication, dict) or (
        publication.get("partial_result_published") is not False
        or publication.get("success_artifact_published_before_retry") is not False
        or publication.get("retry_artifact_published") is not True
    ):
        fail(f"{path.name} no-partial-publication provenance is invalid")

    unwind = artifact.get("unwind")
    ledger = unwind.get("allocation_ledger") if isinstance(unwind, dict) else None
    if not isinstance(unwind, dict) or not isinstance(ledger, dict) or (
        unwind.get("owned_resources_drop") != "Rust error-unwind Drop"
        or unwind.get("resources_released_on_unwind") is not True
        or finite(ledger.get("failure_live_allocation_bytes"), "failure live bytes") <= 0
        or finite(ledger.get("failure_allocation_high_water_bytes"), "failure high water")
        < finite(ledger.get("failure_live_allocation_bytes"), "failure live bytes")
        or finite(ledger.get("post_unwind_live_allocation_bytes"), "post-unwind live bytes") != 0
        or finite(ledger.get("post_unwind_live_staging_bytes"), "post-unwind staging bytes") != 0
    ):
        fail(f"{path.name} allocation-ledger unwind evidence is invalid")

    recovery = artifact.get("recovery")
    if not isinstance(recovery, dict) or (
        recovery.get("required") != expected_recovery
        or recovery.get("retry_mode") != "recreated-context"
        or recovery.get("context_recreated") is not True
        or recovery.get("retry_success") is not True
    ):
        fail(f"{path.name} recovery/retry provenance is invalid")

    cpu = artifact.get("cpu_reference")
    if not isinstance(cpu, dict) or (
        cpu.get("backend") != "cpu"
        or cpu.get("precision") != "f64"
        or cpu.get("independent") is not True
        or cpu.get("frequency_count") != 2
    ):
        fail(f"{path.name} independent CPU/f64 reference is missing")
    retry = artifact.get("retry_success")
    retry_backend = retry.get("backend") if isinstance(retry, dict) else None
    transfer = retry.get("transfer") if isinstance(retry, dict) else None
    residency = retry.get("residency") if isinstance(retry, dict) else None
    if not isinstance(retry, dict) or not isinstance(retry_backend, dict) or not isinstance(transfer, dict) or not isinstance(residency, dict):
        fail(f"{path.name} retry evidence is missing")
    if retry_backend.get("requested") != "cuda" or retry_backend.get("actual") != "cuda" or retry_backend.get("fallback_status") != "none" or retry_backend.get("precision") != "f64":
        fail(f"{path.name} retry was not native CUDA")
    nonempty(retry_backend.get("device"), f"{path.name}.retry.device")
    if retry.get("frequency_count") != 2 or retry.get("all_frequencies_converged") is not True:
        fail(f"{path.name} retry convergence evidence is incomplete")
    for key, limit in (
        ("max_relative_residual", 1.0e-6),
        ("max_power_balance_relative_error", 5.0e-2),
        ("max_velocity_relative_error", 1.0e-7),
        ("max_impedance_relative_error", 1.0e-7),
        ("max_derivative_relative_error", 1.0e-7),
    ):
        if finite(retry.get(key), f"{path.name}.retry.{key}") > limit:
            fail(f"{path.name}.{key} exceeds the bounded parity gate")
    for key in ("h2d_operations", "h2d_bytes", "d2h_operations", "d2h_bytes"):
        if finite(transfer.get(key), f"{path.name}.retry.transfer.{key}") <= 0:
            fail(f"{path.name}.{key} does not prove CUDA movement")
    peak = finite(residency.get("allocation_high_water_bytes"), f"{path.name}.retry.peak")
    live = finite(residency.get("live_allocation_bytes"), f"{path.name}.retry.live")
    if peak <= 0 or live < 0 or live > peak:
        fail(f"{path.name} retry residency watermark is invalid")
    checksum(path)
    return artifact


def check_result(log_path: Path) -> None:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read QA log: {error}")
    records = [line.removeprefix("QA_RESULT: ") for line in lines if line.startswith("QA_RESULT: ")]
    if len(records) != 1:
        fail("expected exactly one QA_RESULT record")
    try:
        result = json.loads(records[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is not JSON: {error}")
    if not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT schema/status/case is invalid")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT is not admitted as native CUDA")
    runtime = result.get("runtime")
    hardware = runtime.get("hardware") if isinstance(runtime, dict) else None
    if not isinstance(runtime, dict) or not isinstance(hardware, dict):
        fail("QA_RESULT runtime/hardware evidence is missing")
    for key in ("device", "driver", "toolkit", "cpu_model", "os", "compiler", "target_features", "runtime_dispatch", "deterministic_mode", "topology"):
        nonempty(hardware.get(key), f"QA_RESULT.hardware.{key}")
    if hardware.get("os") != "linux-x86_64" or hardware.get("target_features") != "cuda" or hardware.get("runtime_dispatch") != "cuda-native":
        fail("QA_RESULT host/dispatch provenance is inconsistent")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 2 or counts.get("passed") != 2 or counts.get("failed") != 0 or counts.get("unsupported") != 0:
        fail("QA_RESULT counts do not prove two passing failure lanes")
    counters = runtime.get("counters")
    if not isinstance(counters, dict):
        fail("QA_RESULT runtime counters are missing")
    for key in ("allocations", "h2d_transfers", "d2h_transfers", "synchronizations", "launches"):
        if finite(counters.get(key), f"QA_RESULT.counters.{key}") <= 0:
            fail(f"QA_RESULT.counters.{key} does not prove CUDA work")
    memory = runtime.get("memory")
    if not isinstance(memory, dict) or finite(memory.get("peak_bytes"), "QA_RESULT.peak_bytes") <= 0 or finite(memory.get("live_bytes"), "QA_RESULT.live_bytes") != 0:
        fail("QA_RESULT memory watermarks are invalid")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or finite(metrics.get("failure_modes_exercised"), "failure_modes_exercised") != 2:
        fail("QA_RESULT does not prove both injected lanes")
    if finite(metrics.get("cuda_h2d_operations"), "cuda_h2d_operations") <= 0 or finite(metrics.get("cuda_d2h_operations"), "cuda_d2h_operations") <= 0 or finite(metrics.get("cuda_allocation_high_water_bytes"), "cuda_allocation_high_water_bytes") <= 0:
        fail("QA_RESULT CUDA counters are incomplete")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        fail("QA_RESULT must declare both failure artifacts")
    env = result.get("environment")
    if not isinstance(env, dict) or env.get("gpu.failure_modes_exercised") != "launch,device-loss":
        fail("QA_RESULT failure-mode environment is missing")


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} LAUNCH.json DEVICE_LOSS.json QA_RESULT.log")
    check_artifact(Path(sys.argv[1]), "launch")
    check_artifact(Path(sys.argv[2]), "device-loss")
    check_result(Path(sys.argv[3]))
    print("SM1 modeling CUDA failure/recovery artifacts passed strict checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
