#!/usr/bin/env python3
"""Strict checker for the bounded SM2 axisymmetric response replay."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "speaker.cuda-sm2-axisymmetric-response-breadth.v1"
SCHEMA = "sonium-sim-speaker-modeling-sm2-breadth-evidence/v1"
FREQUENCIES = [400.0, 500.0, 600.0]


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SM2 breadth artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{label} is not finite")
    return number


def integer(value: object, label: str) -> int:
    number = finite(value, label)
    if number != int(number):
        fail(f"{label} is not an integer")
    return int(number)


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
        fail(f"{path.name} sidecar is missing: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"{path.name} sidecar has no SHA-256 digest")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"{path.name} checksum mismatch")


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
    if integer(runtime.get("kernel_launches"), f"{label}.kernel_launches") + integer(
        runtime.get("library_operations"), f"{label}.library_operations"
    ) <= 0:
        fail(f"{label} does not prove native CUDA work")
    if integer(runtime.get("allocations"), f"{label}.allocations") != 0:
        fail(f"{label} allocated during the resident action")
    transfers = runtime.get("transfers")
    if not isinstance(transfers, dict):
        fail(f"{label}.transfers is missing")
    if integer(transfers.get("h2d_operations"), f"{label}.transfers.h2d_operations") != 1:
        fail(f"{label} must upload exactly one resident response state")
    if integer(transfers.get("d2h_operations"), f"{label}.transfers.d2h_operations") != 1:
        fail(f"{label} must publish exactly one response output")
    if integer(transfers.get("d2d_operations"), f"{label}.transfers.d2d_operations") != 0:
        fail(f"{label} used an undocumented device copy")
    if integer(transfers.get("staging_allocation_operations"), f"{label}.transfers.staging_allocation_operations") != 0:
        fail(f"{label} used staging")


def check_artifact(path: Path) -> None:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not an object")
    if artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID:
        fail("schema or case identifier is incorrect")
    if artifact.get("status") != "passed":
        fail("artifact is not marked passed")
    check_hardware(artifact)

    backend = artifact.get("backend")
    expected_backend = {
        "requested": "cuda",
        "actual": "cuda",
        "fallback_status": "none",
        "precision": "Complex64",
        "dispatch": "native-cuda-cusparse",
    }
    if not isinstance(backend, dict) or any(backend.get(key) != value for key, value in expected_backend.items()):
        fail("backend provenance is not the expected native Complex64 CUDA path")

    predecessor = artifact.get("sm1_production_output_consumed")
    if not isinstance(predecessor, dict) or predecessor.get("status") != "passed" or predecessor.get("actual_backend") != "cuda":
        fail("SM1 production predecessor is not explicitly consumed")
    if predecessor.get("operator") != "AxisymmetricLinearModel::CoupledSystem::assemble_harmonic":
        fail("SM1 predecessor operator identity is missing")
    sm1_dimension = integer(predecessor.get("dimension"), "sm1.dimension")
    sm1_nnz = integer(predecessor.get("nnz"), "sm1.nnz")

    production = artifact.get("production_model")
    if not isinstance(production, dict):
        fail("production model record is missing")
    if production.get("operator") != "AxisymmetricLinearModel::CoupledSystem::assemble_harmonic":
        fail("production operator identity is missing")
    dimension = integer(production.get("dimension"), "production.dimension")
    nnz = integer(production.get("nnz"), "production.nnz")
    if dimension != sm1_dimension or nnz != sm1_nnz or dimension < 100 or nnz < dimension:
        fail("SM1/SM2 production operator identity is inconsistent")
    frequencies = production.get("frequencies_hz")
    if frequencies != FREQUENCIES:
        fail("SM2 frequency grid is not the declared three-point response grid")
    rows = production.get("rows")
    if not isinstance(rows, list) or len(rows) != len(FREQUENCIES):
        fail("SM2 production response rows are missing")
    for index, (row, frequency) in enumerate(zip(rows, FREQUENCIES)):
        if not isinstance(row, dict) or not math.isclose(finite(row.get("frequency_hz"), f"rows[{index}].frequency_hz"), frequency):
            fail(f"rows[{index}] frequency is inconsistent")
        if row.get("dimension") != dimension or row.get("nnz") != nnz:
            fail(f"rows[{index}] production identity is inconsistent")
        if integer(row.get("value_update_h2d_operations"), f"rows[{index}].value_update_h2d_operations") != (0 if index == 0 else 1):
            fail(f"rows[{index}] matrix-value update counter is inconsistent")
        if row.get("cpu_solver_converged") is not True:
            fail(f"rows[{index}] CPU solver was not converged")
        if finite(row.get("cpu_solver_relative_residual"), f"rows[{index}].solver_residual") > 1.0e-6:
            fail(f"rows[{index}] CPU residual exceeds its gate")
        if finite(row.get("cpu_power_closure_relative_error"), f"rows[{index}].power_closure") > 5.0e-2:
            fail(f"rows[{index}] power closure exceeds its gate")
        for key in (
            "cpu_observation_spl_db",
            "cpu_impedance_real_ohm",
            "cpu_impedance_imag_ohm",
            "cpu_quadratic_work",
            "cuda_quadratic_work",
            "operator_relative_error",
        ):
            finite(row.get(key), f"rows[{index}].{key}")
        if finite(row.get("operator_relative_error"), f"rows[{index}].operator_relative_error") > 1.0e-8:
            fail(f"rows[{index}] CUDA/CPU operator parity exceeds its gate")
        if integer(row.get("action_h2d_operations"), f"rows[{index}].action_h2d_operations") != 1:
            fail(f"rows[{index}] does not prove a state upload")
        if integer(row.get("action_d2h_operations"), f"rows[{index}].action_d2h_operations") != 1:
            fail(f"rows[{index}] does not prove output publication")
        check_runtime(row.get("action_runtime"), f"rows[{index}]")

    observables = artifact.get("observables")
    if not isinstance(observables, dict) or observables.get("kind") != "resident assembled-harmonic quadratic work derivative":
        fail("quadratic-work derivative observable is missing")
    if not isinstance(observables.get("cpu_quadratic_work"), list) or len(observables["cpu_quadratic_work"]) != 3:
        fail("CPU quadratic-work oracle is missing")
    if not isinstance(observables.get("cuda_quadratic_work"), list) or len(observables["cuda_quadratic_work"]) != 3:
        fail("CUDA quadratic-work response is missing")
    for values, label in ((observables["cpu_quadratic_work"], "cpu_work"), (observables["cuda_quadratic_work"], "cuda_work")):
        for index, value in enumerate(values):
            finite(value, f"observables.{label}[{index}]")
    if finite(observables.get("operator_max_relative_error"), "observables.operator_max_relative_error") > 1.0e-8:
        fail("maximum operator residual exceeds its gate")
    derivative_error = finite(observables.get("derivative_relative_error"), "observables.derivative_relative_error")
    if derivative_error > 1.0e-8:
        fail("quadratic-work derivative exceeds its gate")
    finite(observables.get("cpu_centered_derivative_per_hz"), "observables.cpu_derivative")
    finite(observables.get("cuda_centered_derivative_per_hz"), "observables.cuda_derivative")

    admission = artifact.get("admission")
    if not isinstance(admission, dict) or admission.get("stale_values_typed") is not True or admission.get("malformed_input_typed") is not True:
        fail("stale and malformed input rejection is not proven")
    if admission.get("operator") != "persistent_csr_apply":
        fail("admission operator is inconsistent")

    recovery = artifact.get("failure_recovery")
    if not isinstance(recovery, dict) or any(recovery.get(key) is not True for key in ("typed_launch", "output_preserved", "retry_succeeded")):
        fail("typed launch/no-publication/retry evidence is incomplete")
    if recovery.get("context_recovery") != "reusable resident context" or recovery.get("partial_result_published") is not False:
        fail("launch recovery/publication disposition is inconsistent")
    if finite(recovery.get("retry_relative_error"), "failure_recovery.retry_relative_error") > 1.0e-8:
        fail("retry operator parity exceeds its gate")

    residency = artifact.get("residency")
    if not isinstance(residency, dict):
        fail("residency record is missing")
    if residency.get("structure_uploaded_once") is not True or integer(residency.get("frequency_value_updates"), "residency.frequency_value_updates") != 2 or integer(residency.get("response_state_updates"), "residency.response_state_updates") != 3:
        fail("resident structure/value/state update counts are inconsistent")
    if integer(residency.get("action_h2d_operations"), "residency.action_h2d_operations") != 3 or integer(residency.get("action_d2h_operations"), "residency.action_d2h_operations") != 3:
        fail("resident action transfer counters are inconsistent")
    if integer(residency.get("action_allocation_operations"), "residency.action_allocation_operations") != 0:
        fail("resident actions allocated state")
    if integer(residency.get("allocation_high_water_bytes"), "residency.allocation_high_water_bytes") <= 0 or integer(residency.get("live_device_bytes_after_teardown"), "residency.live_device_bytes_after_teardown") != 0 or integer(residency.get("live_staging_bytes_after_teardown"), "residency.live_staging_bytes_after_teardown") != 0:
        fail("allocation/staging teardown ledger is not clean")

    counts = artifact.get("counts")
    if not isinstance(counts, dict) or {key: integer(counts.get(key), f"counts.{key}") for key in ("executed", "passed", "unsupported")} != {"executed": 5, "passed": 5, "unsupported": 0}:
        fail("SM2 breadth action counts are not exactly five passed actions")


def check_log(path: Path) -> None:
    try:
        lines = [line[len("QA_RESULT:") :].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("QA_RESULT:")]
    except OSError as error:
        fail(f"QA log is missing: {error}")
    if len(lines) != 1:
        fail("QA log must contain exactly one QA_RESULT line")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is invalid JSON: {error}")
    if not isinstance(result, dict) or result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT does not prove a passed SM2 breadth case")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT is not native CUDA")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT runtime evidence is missing")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 5 or counts.get("passed") != 5 or counts.get("failed") != 0:
        fail("QA_RESULT counts are not five passed actions")
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
    print(f"SM2 breadth native axisymmetric CUDA artifact passed strict checks: {artifact}")


if __name__ == "__main__":
    main()
