#!/usr/bin/env python3
"""Strict checker for SM1's bounded production FEM frequency-step cell."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "speaker.cuda-production-frequency-step.v1"
RESIDUAL_BUDGET = 1.0e-10
HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SM1 production-frequency artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{label} is not finite")
    return number


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def checksum(path: Path) -> None:
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
    if expected is None or not HEX_64.fullmatch(expected):
        fail(f"checksum sidecar has no SHA-256 digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}")


def check_artifact(path: Path) -> dict:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not an object")
    if artifact.get("case_id") != CASE_ID or artifact.get("schema_version") != 1:
        fail("schema or case identifier is not SM1 production-frequency v1")
    if artifact.get("status") != "passed":
        fail("artifact is not a passed result")
    if artifact.get("requested_backend") != "cuda" or artifact.get("actual_backend") != "cuda":
        fail("artifact does not prove native CUDA admission")
    if artifact.get("execution_mode") != "DeviceResidentOperator":
        fail("execution mode is not device-resident operator")
    if artifact.get("dispatch") != "native-cuda-cusparse":
        fail("dispatch is not native CUDA cuSPARSE")
    if artifact.get("operator") != "AxisymmetricLinearModel::CoupledSystem::assemble_harmonic":
        fail("operator is not the production coupled harmonic assembly")
    scope = text(artifact.get("production_scope"), "production_scope")
    if "magnetic" not in scope.lower() or "acoustic" not in scope.lower() or "structural" not in scope.lower():
        fail("production scope does not name magnetic/acoustic/structural coupling")
    if "host assembly" not in scope.lower():
        fail("production scope does not retain the host-assembly boundary")

    dimension = finite(artifact.get("dimension"), "dimension")
    nnz = finite(artifact.get("nnz"), "nnz")
    if dimension <= 100 or dimension != int(dimension) or nnz < dimension or nnz != int(nnz):
        fail("production matrix dimensions/nnz are not a bounded nontrivial CSR")
    frequencies = artifact.get("frequencies_hz")
    if frequencies != [250.0, 500.0]:
        fail("frequency grid is not the declared two-point fixture")
    actions = artifact.get("frequency_actions")
    if not isinstance(actions, list) or len(actions) != 2:
        fail("exactly two frequency actions are required")
    for index, (action, expected_frequency) in enumerate(zip(actions, frequencies)):
        if not isinstance(action, dict):
            fail(f"frequency action {index} is not an object")
        if finite(action.get("frequency_hz"), f"frequency_actions[{index}].frequency_hz") != expected_frequency:
            fail("frequency action grid does not match the declared fixture")
        if finite(action.get("dimension"), f"frequency_actions[{index}].dimension") != dimension:
            fail("frequency action dimension changed")
        if finite(action.get("nnz"), f"frequency_actions[{index}].nnz") != nnz:
            fail("frequency action nnz changed")
        if finite(action.get("cpu_output_l2"), f"frequency_actions[{index}].cpu_output_l2") <= 0.0:
            fail("CPU oracle output is empty")
        if finite(action.get("cuda_output_l2"), f"frequency_actions[{index}].cuda_output_l2") <= 0.0:
            fail("CUDA output is empty")
        if finite(action.get("max_relative_residual"), f"frequency_actions[{index}].residual") > RESIDUAL_BUDGET:
            fail("frequency action exceeded the CPU parity budget")
        if finite(action.get("action_kernel_operations"), f"frequency_actions[{index}].kernels") <= 0:
            fail("frequency action has no native CUDA kernel/library operation")
        if finite(action.get("action_h2d_operations"), f"frequency_actions[{index}].h2d") != 0:
            fail("warm frequency action performed a host-to-device transfer")
        if finite(action.get("action_d2h_operations"), f"frequency_actions[{index}].d2h") <= 0:
            fail("frequency action did not produce an observed device output")
        if action.get("value_update") != (index == 1):
            fail("frequency value-update marker is inconsistent")

    residency = artifact.get("residency")
    if not isinstance(residency, dict):
        fail("residency evidence is missing")
    if residency.get("structure_uploaded_once") is not True or residency.get("values_updated_once_for_second_frequency") is not True:
        fail("resident structure/value-update contract is missing")
    if finite(residency.get("warm_action_h2d_operations"), "residency.warm_h2d") != 0:
        fail("warm residency has host-to-device traffic")
    if finite(residency.get("warm_action_allocation_operations"), "residency.warm_allocations") != 0:
        fail("warm residency allocates device resources")
    if finite(residency.get("frequency_value_update_h2d_operations"), "residency.value_update_h2d") != 1:
        fail("second-frequency resident value update is not exactly one H2D operation")
    if finite(residency.get("live_device_bytes_before_teardown"), "residency.live_before") <= 0:
        fail("resident device bytes were not observed before teardown")
    if finite(residency.get("live_device_bytes_after_teardown"), "residency.live_after") != 0 or finite(residency.get("live_staging_bytes_after_teardown"), "residency.staging_after") != 0:
        fail("device/staging ledger did not return to zero after teardown")

    reference = artifact.get("cpu_reference")
    if not isinstance(reference, dict) or "independent host CSR SpMV" not in text(reference.get("kind"), "cpu_reference.kind"):
        fail("independent CPU CSR oracle is missing")
    if reference.get("precision") != "Complex64":
        fail("CPU oracle precision is not Complex64")
    if finite(reference.get("max_relative_residual"), "cpu_reference.residual") > RESIDUAL_BUDGET:
        fail("CPU/CUDA residual exceeds budget")

    failure = artifact.get("launch_failure")
    if not isinstance(failure, dict) or failure.get("status") != "typed" or failure.get("stage") != "launch":
        fail("typed launch failure evidence is missing")
    text(failure.get("operation"), "launch_failure.operation")
    text(failure.get("typed_error"), "launch_failure.typed_error")
    if failure.get("context_recovery") != "synchronize_before_reuse" or failure.get("output_disposition") != "Unusable":
        fail("launch recovery/output disposition is not conservative")
    if failure.get("partial_result_published") is not False or failure.get("recovery_succeeded") is not True or failure.get("retry_succeeded") is not True:
        fail("launch failure publication/retry contract failed")
    if finite(failure.get("retry_max_relative_residual"), "launch_failure.retry_residual") > RESIDUAL_BUDGET:
        fail("retry exceeded CPU parity budget")
    if artifact.get("resources_released_on_teardown") is not True:
        fail("teardown release is not proven")
    text(artifact.get("scope_boundary"), "scope_boundary")
    checksum(path)
    return artifact


def check_result(log_path: Path) -> None:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read QA result log: {error}")
    records = [line.removeprefix("QA_RESULT: ") for line in lines if line.startswith("QA_RESULT: ")]
    if len(records) != 1:
        fail("expected exactly one QA_RESULT record")
    try:
        result = json.loads(records[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is not JSON: {error}")
    if not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("case_id") != CASE_ID:
        fail("QA_RESULT schema or case identifier is not SM1 production-frequency v1")
    if result.get("status") != "passed" or result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT does not prove a native CUDA pass")
    if "SUCCESS" not in lines:
        fail("producer did not report SUCCESS")
    environment = result.get("environment")
    if not isinstance(environment, dict) or "host assembly" not in environment.get("gpu.execution_scope", "").lower():
        fail("QA_RESULT does not retain the host-assembly scope boundary")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT runtime evidence is missing")
    hardware = runtime.get("hardware")
    if not isinstance(hardware, dict) or hardware.get("os") != "linux-x86_64" or hardware.get("target_features") != "cuda" or hardware.get("runtime_dispatch") != "cuda-native":
        fail("QA_RESULT hardware provenance is not native Linux CUDA")
    if "nvidia" not in hardware.get("device", "").lower():
        fail("QA_RESULT device is not NVIDIA")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 1 or counts.get("passed") != 1 or counts.get("failed") != 0:
        fail("QA_RESULT execution counts are invalid")
    memory = runtime.get("memory")
    if not isinstance(memory, dict) or finite(memory.get("peak_bytes"), "runtime.memory.peak") <= 0 or finite(memory.get("live_bytes"), "runtime.memory.live") != 0:
        fail("QA_RESULT memory ledger is invalid")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or finite(metrics.get("max_relative_residual"), "metrics.residual") > RESIDUAL_BUDGET or finite(metrics.get("warm_action_h2d_operations"), "metrics.warm_h2d") != 0 or finite(metrics.get("frequency_value_update_h2d_operations"), "metrics.value_update_h2d") != 1:
        fail("QA_RESULT metric contract is invalid")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json QA_RESULT.log")
    check_artifact(Path(sys.argv[1]))
    check_result(Path(sys.argv[2]))
    print(f"SM1 production-frequency CUDA artifact passed strict checks: {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
