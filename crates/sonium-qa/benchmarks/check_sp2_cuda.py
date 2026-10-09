#!/usr/bin/env python3
"""Fail-closed checker for the SP2 native CUDA failure/diagnostics matrix."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn


NATIVE_CASE = "solvers.preconditioner-cuda-native.v1"
FAILURE_CASE = "solvers.preconditioner-failure-cuda.v1"
REQUIRED_HARDWARE = (
    "device",
    "driver",
    "toolkit",
    "topology",
    "cpu_model",
    "microcode",
    "compiler",
    "os",
)
REQUIRED_ACCEPTANCE = {
    "scalar_and_block_host_reference_parity",
    "real_and_complex_promoted_cells",
    "native_action_intervals_have_zero_h2d_d2h_and_staging",
    "unsupported_adjoint_actions_fail_before_device_work",
}
REQUIRED_FAILURE_ACCEPTANCE = {
    "all_setup_faults_typed",
    "setup_failure_allocation_ledger_restored",
    "runtime_faults_typed",
    "runtime_diagnostics_recorded",
    "adjoint_counters_recorded",
    "unsupported_adjoints_transactional",
    "transactional_reusable_fault_outputs",
    "runtime_failure_allocation_ledger_restored",
    "device_loss_output_not_read",
    "device_loss_host_allocation_ledger_restored",
    "retry_after_reusable_fault",
    "synchronization_fault_typed",
    "quota_retry_restored",
    "preconditioner_setup_quota_retry_transactional",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SP2 artifact rejected: {message}")


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        fail(f"{label} is not finite and non-negative")
    return number


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def check_hardware(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        fail(f"{label} hardware provenance is missing")
    for field in REQUIRED_HARDWARE:
        text(value.get(field), f"{label}.hardware.{field}")
    if value["os"] != "linux-x86_64":
        fail(f"{label} is not native Linux x86_64 evidence")
    return {field: str(value[field]) for field in REQUIRED_HARDWARE}


def check_acceptance(value: Any, required: set[str], label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label}.acceptance is missing")
    for key in required:
        if value.get(key) is not True:
            fail(f"{label}.acceptance.{key} is not true")


def check_transfer(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} transfer counters are missing")
    for field in (
        "h2d_operations",
        "d2h_operations",
        "d2d_operations",
        "h2d_bytes",
        "d2h_bytes",
        "d2d_bytes",
        "staging_allocation_operations",
        "live_staging_bytes",
    ):
        finite(value.get(field), f"{label}.{field}")


def check_native(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("schema_version") != 1 or artifact.get("case_id") != NATIVE_CASE:
        fail("native artifact schema or case id is invalid")
    if artifact.get("requested_backend") != "cuda" or artifact.get("actual_backend") != "cuda":
        fail("native artifact did not use required CUDA")
    hardware = check_hardware(artifact.get("hardware"), "native")
    resident = artifact.get("resident_state")
    if not isinstance(resident, dict):
        fail("native resident_state is missing")
    for key in ("setup_before_actions", "hierarchy_and_workspaces_reused"):
        if resident.get(key) is not True:
            fail(f"native resident_state.{key} is not true")
    if resident.get("per_application_host_staging") is not False:
        fail("native resident state reports per-application staging")
    check_acceptance(artifact.get("acceptance"), REQUIRED_ACCEPTANCE, "native")
    preconditioners = artifact.get("preconditioners")
    if not isinstance(preconditioners, dict) or len(preconditioners) < 8:
        fail("native artifact lacks the promoted preconditioner action cells")
    passed_actions = 0
    for name, record in preconditioners.items():
        if not isinstance(record, dict):
            fail(f"native.preconditioners.{name} is not an object")
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, dict):
            # The producer keeps two convenience records for the exact
            # Point-Jacobi transpose cells.  Their diagnostics are retained
            # in the corresponding family record; require that the
            # convenience record is explicitly exact before skipping it.
            if name.endswith("_transpose") and record.get("capability") == "Exact":
                continue
            fail(f"native.preconditioners.{name}.diagnostics is missing")
        for field in (
            "setup_count",
            "setup_failures",
            "apply_count",
            "apply_failures",
            "fallback_count",
            "transpose_count",
            "transpose_failures",
            "hermitian_count",
            "hermitian_failures",
        ):
            finite(diagnostics.get(field), f"native.preconditioners.{name}.diagnostics.{field}")
        for action_name, action in record.items():
            if action_name in {"diagnostics", "capabilities"} or not isinstance(action, dict):
                continue
            status = action.get("status")
            if status == "passed":
                error = finite(action.get("max_abs_error"), f"native.preconditioners.{name}.{action_name}.max_abs_error")
                if error > 3.0e-4:
                    fail(f"native.preconditioners.{name}.{action_name} parity exceeds 3e-4")
                check_transfer(action.get("transfers"), f"native.preconditioners.{name}.{action_name}")
                if finite(action["transfers"].get("h2d_operations"), "action h2d") != 0 or finite(action["transfers"].get("d2h_operations"), "action d2h") != 0:
                    fail(f"native.preconditioners.{name}.{action_name} staged host data")
                passed_actions += 1
    unsupported = artifact.get("unsupported_adjoint_actions")
    if not isinstance(unsupported, list) or len(unsupported) < 4:
        fail("native unsupported adjoint matrix is incomplete")
    for index, record in enumerate(unsupported):
        if not isinstance(record, dict) or record.get("status") != "unsupported" or record.get("device_work") is not False:
            fail(f"native unsupported_adjoint_actions[{index}] is not fail-before-device-work")
    if passed_actions < 16:
        fail(f"native action matrix has only {passed_actions} passed action records")
    return {"hardware": hardware, "passed_actions": passed_actions, "unsupported_adjoint_actions": len(unsupported)}


def check_failure(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("schema_version") != 1 or artifact.get("case_id") != FAILURE_CASE:
        fail("failure artifact schema or case id is invalid")
    if artifact.get("requested_backend") != "cuda" or artifact.get("actual_backend") != "cuda":
        fail("failure artifact did not use required CUDA")
    hardware = check_hardware(artifact.get("hardware"), "failure")
    check_acceptance(artifact.get("acceptance"), REQUIRED_FAILURE_ACCEPTANCE, "failure")
    setup = artifact.get("setup_failures")
    runtime = artifact.get("runtime_failures")
    device_loss = artifact.get("device_loss_failures")
    device_loss_context_teardown = artifact.get("device_loss_context_teardown")
    adjoint = artifact.get("adjoint_diagnostics")
    if not isinstance(setup, list) or len(setup) < 15:
        fail("failure setup matrix must contain at least 15 rows")
    if not isinstance(runtime, list) or len(runtime) < 10:
        fail("failure runtime matrix must contain at least 10 rows")
    if not isinstance(device_loss, list) or len(device_loss) < 5:
        fail("failure device-loss matrix must contain at least five rows")
    if not isinstance(device_loss_context_teardown, list) or len(device_loss_context_teardown) < 5:
        fail("failure device-loss context teardown matrix must contain at least five rows")
    if not isinstance(adjoint, list) or len(adjoint) < 5:
        fail("failure adjoint diagnostics matrix must contain five families")
    for index, record in enumerate(setup):
        if not isinstance(record, dict) or record.get("status") != "typed_failure":
            fail(f"setup_failures[{index}] is not typed")
        text(record.get("error_kind"), f"setup_failures[{index}].error_kind")
        text(record.get("first_failure_stage"), f"setup_failures[{index}].first_failure_stage")
        if record.get("allocation_ledger_restored") is not True:
            fail(f"setup_failures[{index}] did not restore allocation ledger")
        finite(record.get("before_live_bytes"), f"setup_failures[{index}].before_live_bytes")
        finite(record.get("after_live_bytes"), f"setup_failures[{index}].after_live_bytes")
        if record["before_live_bytes"] != record["after_live_bytes"]:
            fail(f"setup_failures[{index}] live allocation changed")
    for index, record in enumerate(runtime):
        if not isinstance(record, dict) or record.get("status") != "typed_failure":
            fail(f"runtime_failures[{index}] is not typed")
        text(record.get("error_kind"), f"runtime_failures[{index}].error_kind")
        text(record.get("first_failure_stage"), f"runtime_failures[{index}].first_failure_stage")
        if record.get("diagnostics_recorded") is not True or record.get("retry_succeeded") is not True:
            fail(f"runtime_failures[{index}] diagnostics/retry contract is incomplete")
        if record.get("output_unchanged") is not True:
            fail(f"runtime_failures[{index}] did not preserve output")
        if record.get("allocation_ledger_restored") is not True:
            fail(f"runtime_failures[{index}] did not restore allocation ledger")
        finite(record.get("baseline_live_bytes"), f"runtime_failures[{index}].baseline_live_bytes")
        finite(record.get("live_bytes_after_teardown"), f"runtime_failures[{index}].live_bytes_after_teardown")
        if record["baseline_live_bytes"] != record["live_bytes_after_teardown"]:
            fail(f"runtime_failures[{index}] live allocation changed")
    for index, record in enumerate(device_loss):
        if not isinstance(record, dict) or record.get("status") != "typed_failure" or record.get("error_kind") != "device_loss":
            fail(f"device_loss_failures[{index}] is not typed device loss")
        if record.get("first_failure_stage") != "DeviceLoss" or record.get("diagnostics_recorded") is not True or record.get("output_not_read_after_terminal_failure") is not True:
            fail(f"device_loss_failures[{index}] terminal contract is incomplete")
        if record.get("host_allocation_ledger_restored") is not True:
            fail(f"device_loss_failures[{index}] did not restore host allocation ledger")
        for field in ("baseline_live_bytes", "live_bytes_before_teardown", "live_bytes_after_teardown"):
            finite(record.get(field), f"device_loss_failures[{index}].{field}")
        if record["baseline_live_bytes"] != record["live_bytes_after_teardown"]:
            fail(f"device_loss_failures[{index}] host allocation ledger changed")
    for index, record in enumerate(device_loss_context_teardown):
        if not isinstance(record, dict) or record.get("host_allocation_ledger_restored") is not True:
            fail(f"device_loss_context_teardown[{index}] did not restore host allocation ledger")
        finite(record.get("baseline_live_bytes"), f"device_loss_context_teardown[{index}].baseline_live_bytes")
        finite(record.get("live_bytes_after_teardown"), f"device_loss_context_teardown[{index}].live_bytes_after_teardown")
        if record["baseline_live_bytes"] != record["live_bytes_after_teardown"]:
            fail(f"device_loss_context_teardown[{index}] live allocation changed")
    for index, record in enumerate(adjoint):
        if not isinstance(record, dict) or not isinstance(record.get("transpose"), dict) or not isinstance(record.get("hermitian"), dict):
            fail(f"adjoint_diagnostics[{index}] is incomplete")
    sync = artifact.get("synchronization_failure")
    if not isinstance(sync, dict) or sync.get("status") != "typed_failure" or sync.get("error_kind") != "synchronization":
        fail("synchronization failure record is incomplete")
    quota = artifact.get("quota_failure")
    if not isinstance(quota, dict) or quota.get("status") != "typed_failure_and_retry" or quota.get("retry_succeeded") is not True:
        fail("quota failure/retry record is incomplete")
    setup_quota = artifact.get("setup_quota_retry")
    if not isinstance(setup_quota, dict) or setup_quota.get("status") != "typed_failure_and_retry":
        fail("preconditioner setup quota failure/retry record is incomplete")
    if setup_quota.get("family") != "point_jacobi" or setup_quota.get("setup_stage") != "DiagonalUpload":
        fail("preconditioner setup quota record has the wrong setup provenance")
    if setup_quota.get("error_kind") != "allocation" or setup_quota.get("first_failure_stage") != "Transfer":
        fail("preconditioner setup quota record has the wrong typed failure stage")
    if "quota exceeded" not in text(setup_quota.get("error"), "setup_quota_retry.error").lower():
        fail("preconditioner setup quota record is not an over-cap rejection")
    if setup_quota.get("diagnostics_recorded") is not True or setup_quota.get("partial_state_published") is not False:
        fail("preconditioner setup quota record lacks strict publication diagnostics")
    if setup_quota.get("allocation_ledger_restored") is not True or setup_quota.get("retry_succeeded") is not True:
        fail("preconditioner setup quota record lacks transactional retry evidence")
    for field in (
        "failure_cap_bytes",
        "retry_cap_bytes",
        "before_live_bytes",
        "after_failure_live_bytes",
        "after_success_live_bytes",
        "after_drop_live_bytes",
    ):
        finite(setup_quota.get(field), f"setup_quota_retry.{field}")
    if setup_quota["failure_cap_bytes"] != setup_quota["before_live_bytes"]:
        fail("preconditioner setup quota failure cap was not the live baseline")
    if setup_quota["after_failure_live_bytes"] != setup_quota["before_live_bytes"]:
        fail("preconditioner setup quota failure retained partial allocations")
    if setup_quota["retry_cap_bytes"] <= setup_quota["failure_cap_bytes"]:
        fail("preconditioner setup quota retry did not restore an under-cap budget")
    if setup_quota["after_success_live_bytes"] <= setup_quota["before_live_bytes"]:
        fail("preconditioner setup quota retry did not publish resident state")
    if setup_quota["after_success_live_bytes"] > setup_quota["retry_cap_bytes"]:
        fail("preconditioner setup quota retry exceeded its restored cap")
    if setup_quota["after_drop_live_bytes"] != setup_quota["before_live_bytes"]:
        fail("preconditioner setup quota retry leaked allocations after teardown")
    return {
        "hardware": hardware,
        "setup_failures": len(setup),
        "runtime_failures": len(runtime),
        "device_loss_failures": len(device_loss),
        "adjoint_diagnostics": len(adjoint),
    }


def main() -> int:
    if len(sys.argv) not in (3, 5) or (len(sys.argv) == 5 and sys.argv[3] != "--output"):
        raise SystemExit(f"usage: {sys.argv[0]} NATIVE.json FAILURE.json [--output CHECK.json]")
    native = check_native(load(Path(sys.argv[1])))
    failure = check_failure(load(Path(sys.argv[2])))
    check = {
        "schema": "sonium-solvers-preconditioners-sp2-cuda-check/v1",
        "status": "passed",
        "native": {key: value for key, value in native.items() if key != "hardware"},
        "failure": {key: value for key, value in failure.items() if key != "hardware"},
        "hardware": native["hardware"],
    }
    if native["hardware"] != failure["hardware"]:
        fail("native and failure hardware provenance differs")
    if len(sys.argv) == 5:
        Path(sys.argv[4]).write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(check, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
