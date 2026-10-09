#!/usr/bin/env python3
"""Strict checker for C1's native CUDA failure/resource contract.

The producer intentionally uses a small deterministic fixture.  This checker
is the admission boundary for the evidence bundle: a copied artifact, a CPU
fallback, an untyped failure, or a retry with an unstable resource census must
fail closed rather than being counted as CUDA evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "solvers.iterative-cuda-failure-matrix.v1"
METHODS = {"gmres", "fgmres", "block_gmres", "block_fgmres"}
SCALARS = {"f32", "f64", "complex32", "complex64"}
FAILURE_STAGES = {
    "Allocation",
    "DescriptorCreation",
    "Launch",
    "TransferH2D",
    "TransferD2H",
    "TransferD2D",
    "synchronization",
    "event_record",
    "event_wait",
    "DeviceLoss",
    "teardown",
    "capture_abort_teardown",
    "cancellation",
    "nonfinite_operator_output",
}
RESOURCE_FIELDS = {
    "live_contexts",
    "live_streams",
    "live_modules",
    "live_dynamic_libraries",
    "live_library_handles",
    "live_fft_plans",
    "live_completion_events",
    "initialization_baseline_free_bytes",
    "initialization_free_bytes",
    "measured_driver_bytes",
}
PHASES = {
    "compute-check",
    "cuda-source-contract",
    "producer-check",
    "native-failure-matrix",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"C1 CUDA artifact rejected: {message}")


def sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        fail(f"cannot read {path}: {error}")


def check_sidecar(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"missing artifact sidecar {sidecar}: {error}")
    digest = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1].lstrip("*") == path.name:
            digest = fields[0].lower()
            break
    if digest is None or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        fail(f"{sidecar} has no valid SHA-256 for {path.name}")
    actual = sha256(path)
    if actual != digest:
        fail(f"{path.name} sidecar mismatch: {actual} != {digest}")
    return actual


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot parse {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} is missing")
    return value


def integer(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        fail(f"{label} is not a valid {'positive' if positive else 'non-negative'} integer")
    return value


def finite(value: object, label: str, *, allow_null: bool = False) -> None:
    if value is None and allow_null:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        fail(f"{label} is not finite")


def check_capabilities(artifact: dict) -> None:
    cells = artifact.get("capability_matrix")
    if not isinstance(cells, list) or len(cells) != 16:
        fail("capability matrix must contain exactly 16 cells")
    seen: set[tuple[str, str]] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            fail(f"capability cell {index} is not an object")
        method = text(cell.get("method"), f"capability[{index}].method")
        scalar = text(cell.get("scalar"), f"capability[{index}].scalar")
        if method not in METHODS or scalar not in SCALARS or (method, scalar) in seen:
            fail(f"capability cell {index} has an invalid or duplicate method/scalar")
        seen.add((method, scalar))
        if cell.get("native_execution") is not True:
            fail(f"capability {method}/{scalar} was not native")
        if cell.get("operator_state") not in {"native", "vendor_native"}:
            fail(f"capability {method}/{scalar} has a non-native operator")
        if cell.get("preconditioner_state") not in {"native", "vendor_native"}:
            fail(f"capability {method}/{scalar} has a non-native preconditioner")
        execution = cell.get("execution")
        if not isinstance(execution, dict) or execution.get("status") != "passed":
            fail(f"capability {method}/{scalar} did not pass")
        if execution.get("converged") is not True or execution.get("stopping_reason") != "Converged":
            fail(f"capability {method}/{scalar} did not converge")
        integer(execution.get("iterations"), f"capability {method}/{scalar}.iterations", positive=True)
        integer(execution.get("workspace_device_buffers"), f"capability {method}/{scalar}.workspace_device_buffers")
        integer(execution.get("workspace_device_capacity"), f"capability {method}/{scalar}.workspace_device_capacity")
        if execution["workspace_device_buffers"] == 0 or execution["workspace_device_capacity"] == 0:
            fail(f"capability {method}/{scalar} has no device workspace")
        for key in ("projected_residual", "true_residual", "backward_error"):
            finite(execution.get(key), f"capability {method}/{scalar}.{key}")
        transfers = execution.get("transfers")
        if not isinstance(transfers, dict):
            fail(f"capability {method}/{scalar} is missing transfer counters")
        for key in (
            "h2d_operations",
            "h2d_bytes",
            "d2h_operations",
            "d2h_bytes",
            "d2d_operations",
            "d2d_bytes",
        ):
            integer(transfers.get(key), f"capability {method}/{scalar}.transfers.{key}")
    if seen != {(method, scalar) for method in METHODS for scalar in SCALARS}:
        fail("capability matrix does not cover every method/scalar pair")

    counts = artifact.get("method_counts")
    if not isinstance(counts, dict) or set(counts) != METHODS:
        fail("method_counts does not cover exactly the four native methods")
    for method in METHODS:
        row = counts[method]
        if not isinstance(row, dict):
            fail(f"method_counts.{method} is not an object")
        if row.get("executed") != 4 or row.get("passed") != 4 or row.get("failed") != 0 or row.get("unsupported") != 0:
            fail(f"method_counts.{method} is not 4/4 native passed")


def diagnostic(row: dict, stage: str) -> dict:
    value = row.get("diagnostic")
    if not isinstance(value, dict):
        fail(f"failure row {stage} is missing structured solver diagnostic")
    expected_reason = "Cancelled" if stage == "cancellation" else "Failed"
    if value.get("actual_backend") != "cuda" or value.get("stopping_reason") != expected_reason:
        fail(f"failure row {stage} has the wrong diagnostic backend/reason")
    integer(value.get("iterations"), f"failure row {stage}.diagnostic.iterations")
    if value.get("partial_iterate") not in {"Preserved", "Unusable"}:
        fail(f"failure row {stage} has invalid partial-iterate disposition")
    if value.get("context_disposition") not in {
        "Reusable",
        "SynchronizeBeforeReuse",
        "MustRecreate",
    }:
        fail(f"failure row {stage} has invalid context disposition")
    for key in ("projected_residual", "true_residual", "backward_error"):
        finite(value.get(key), f"failure row {stage}.diagnostic.{key}", allow_null=True)
    return value


def direct_diagnostic(row: dict, stage: str) -> None:
    if row.get("diagnostic_actual_backend") != "cuda" or row.get("diagnostic_stopping_reason") != "Failed":
        fail(f"failure row {stage} has the wrong direct diagnostic backend/reason")
    integer(row.get("diagnostic_iterations"), f"failure row {stage}.diagnostic_iterations")
    if row.get("diagnostic_partial_iterate", "Unusable") not in {"Preserved", "Unusable"}:
        fail(f"failure row {stage} has invalid direct partial-iterate disposition")
    if row.get("diagnostic_context_disposition") not in {
        "Reusable",
        "SynchronizeBeforeReuse",
        "MustRecreate",
    }:
        fail(f"failure row {stage} has invalid direct context disposition")
    for key in (
        "diagnostic_projected_residual",
        "diagnostic_true_residual",
        "diagnostic_backward_error",
    ):
        finite(row.get(key), f"failure row {stage}.{key}", allow_null=True)


def check_resources(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"live_allocation_bytes", "live_staging_bytes", "driver_resources"}:
        fail(f"{label} is not a complete runtime resource snapshot")
    integer(value["live_allocation_bytes"], f"{label}.live_allocation_bytes")
    integer(value["live_staging_bytes"], f"{label}.live_staging_bytes")
    driver = value["driver_resources"]
    if not isinstance(driver, dict) or set(driver) != RESOURCE_FIELDS:
        fail(f"{label}.driver_resources is incomplete")
    for key in RESOURCE_FIELDS:
        integer(driver[key], f"{label}.driver_resources.{key}")


def check_failures(artifact: dict) -> None:
    rows = artifact.get("failure_injection")
    if not isinstance(rows, list) or len(rows) != 14:
        fail("failure matrix must contain exactly 14 rows")
    seen: set[str] = set()
    expected_context = {
        "Allocation": "Reusable",
        "DescriptorCreation": "SynchronizeBeforeReuse",
        "Launch": "SynchronizeBeforeReuse",
        "TransferH2D": "SynchronizeBeforeReuse",
        "TransferD2H": "SynchronizeBeforeReuse",
        "TransferD2D": "SynchronizeBeforeReuse",
        "synchronization": "MustRecreate",
        "event_record": "SynchronizeBeforeReuse",
        "event_wait": "MustRecreate",
        "DeviceLoss": "MustRecreate",
        "teardown": "SynchronizeBeforeReuse",
        "capture_abort_teardown": "SynchronizeBeforeReuse",
        "cancellation": "Reusable",
        "nonfinite_operator_output": "SynchronizeBeforeReuse",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            fail(f"failure row {index} is not an object")
        stage = row.get("stage")
        if stage not in FAILURE_STAGES or stage in seen:
            fail(f"failure row {index} has an invalid or duplicate stage")
        seen.add(stage)
        if row.get("status") != "passed" or row.get("expected_failure") is not True:
            fail(f"failure row {stage} is not a passed expected failure")
        text(row.get("error"), f"failure row {stage}.error")
        if row.get("context_disposition") != expected_context[stage]:
            fail(f"failure row {stage} has the wrong context policy")
        if stage in {"Allocation", "cancellation"} and row.get("output_disposition") != "Preserved":
            fail(f"failure row {stage} did not preserve output")
        if stage not in {"capture_abort_teardown", "synchronization", "event_record", "event_wait", "teardown", "cancellation", "nonfinite_operator_output"}:
            diagnostic(row, stage)
        elif stage in {"cancellation", "nonfinite_operator_output"}:
            diagnostic(row, stage)
        elif stage != "capture_abort_teardown":
            direct_diagnostic(row, stage)

    if seen != FAILURE_STAGES:
        fail("failure matrix is missing one or more required stages")
    capture = next(row for row in rows if row["stage"] == "capture_abort_teardown")
    for key in ("typed_teardown", "panic_caught", "output_preserved", "recovered", "retry_succeeded"):
        if capture.get(key) is not True:
            fail(f"capture-abort teardown did not prove {key}")
    if capture.get("output_before_recovery") != [7.0, 7.0] or capture.get("output_after_failure") != [7.0, 7.0]:
        fail("capture-abort teardown did not preserve the sentinel output")
    if capture.get("retry_output") != [14.0, 14.0]:
        fail("capture-abort teardown retry did not publish the expected output")
    baseline = capture.get("resource_baseline")
    if not isinstance(baseline, dict) or baseline.get("stable_after_failure") is not True or baseline.get("stable_after_retry") is not True:
        fail("capture-abort teardown is missing stable resource evidence")
    for key in ("before", "after_failure", "after_retry"):
        check_resources(baseline.get(key), f"capture-abort resource_baseline.{key}")
    if baseline["before"] != baseline["after_failure"] or baseline["before"] != baseline["after_retry"]:
        fail("capture-abort resource census changed across failure/retry")


def check_qa_log(path: Path, artifact: dict) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read QA log {path}: {error}")
    results = []
    for line in lines:
        if line.startswith("QA_RESULT:"):
            try:
                results.append(json.loads(line.split(":", 1)[1].strip()))
            except json.JSONDecodeError as error:
                fail(f"QA_RESULT is invalid JSON: {error}")
    if len(results) != 1:
        fail(f"QA log must contain exactly one QA_RESULT, found {len(results)}")
    result = results[0]
    if result.get("schema_version") != 1 or result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT identity/status is not a native C1 pass")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT backend is not CUDA")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT is missing runtime evidence")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 30 or counts.get("passed") != 30 or counts.get("failed") != 0 or counts.get("unsupported") != 0:
        fail("QA_RESULT runtime counts are not 30/30 native passed")
    hardware = runtime.get("hardware")
    if not isinstance(hardware, dict):
        fail("QA_RESULT is missing hardware provenance")
    for key in ("device", "driver", "toolkit", "os", "compiler", "target_features", "runtime_dispatch"):
        text(hardware.get(key), f"QA_RESULT.runtime.hardware.{key}")
    if "nvidia" not in hardware["device"].lower() or hardware["os"] != "linux-x86_64":
        fail("QA_RESULT hardware is not the required native Linux NVIDIA lane")
    if "cuda" not in hardware["target_features"].lower() or hardware["runtime_dispatch"] != "cuda-native":
        fail("QA_RESULT does not prove native CUDA dispatch")
    if artifact["counts"] != counts:
        fail("artifact and QA_RESULT counts differ")


def check_phases(path: Path) -> None:
    try:
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as error:
        fail(f"cannot read phase status {path}: {error}")
    phase_names = {row[0] for row in rows if len(row) == 2}
    # The runner invokes this checker before it can append the strict-check
    # phase, while retained bundles contain that final successful row.  Accept
    # exactly those two lifecycle states so the archived bundle is recheckable.
    if (
        frozenset(phase_names)
        not in {frozenset(PHASES), frozenset(PHASES | {"strict-check"})}
        or len(rows) != len(phase_names)
        or any(len(row) != 2 or row[1] != "0" for row in rows)
    ):
        fail("phase-status.tsv does not prove every C1 phase passed")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT QA_LOG PHASE_STATUS")
    artifact_path, qa_log, phase_status = map(Path, sys.argv[1:])
    artifact = load_json(artifact_path)
    digest = check_sidecar(artifact_path)
    if set(artifact) != {
        "actual_backend",
        "capability_matrix",
        "case_id",
        "counts",
        "failure_diagnostic_contract",
        "failure_injection",
        "hardware",
        "method_counts",
        "requested_backend",
        "schema_version",
        "unsupported_policy",
    }:
        fail("artifact schema fields differ from the pinned C1 producer")
    if artifact["schema_version"] != 1 or artifact["case_id"] != CASE_ID or artifact["requested_backend"] != "cuda" or artifact["actual_backend"] != "cuda":
        fail("artifact identity is not the required native C1 case")
    hardware = artifact["hardware"]
    if not isinstance(hardware, dict) or hardware.get("os") != "linux-x86_64":
        fail("artifact host is not Linux x86_64")
    for key in ("cpu_model", "compiler", "microcode", "thread_policy", "affinity_policy", "numa_policy", "deterministic_mode"):
        text(hardware.get(key), f"artifact.hardware.{key}")
    counts = artifact["counts"]
    if not isinstance(counts, dict) or counts != {"executed": 30, "passed": 30, "failed": 0, "unsupported": 0}:
        fail("artifact counts are not 30/30 native passed")
    if artifact["failure_diagnostic_contract"] != [
        "method", "actual_backend", "iterations", "stopping_reason", "projected_residual", "true_residual", "backward_error", "partial_iterate", "context_disposition"
    ]:
        fail("failure diagnostic contract is incomplete")
    check_capabilities(artifact)
    check_failures(artifact)
    check_qa_log(qa_log, artifact)
    check_phases(phase_status)
    print(json.dumps({
        "status": "passed",
        "case_id": CASE_ID,
        "artifact_sha256": digest,
        "native_capability_cells": 16,
        "failure_rows": 14,
        "executed": 30,
        "passed": 30,
        "failed": 0,
        "unsupported": 0,
        "resource_census": "stable across capture-abort failure and retry",
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
