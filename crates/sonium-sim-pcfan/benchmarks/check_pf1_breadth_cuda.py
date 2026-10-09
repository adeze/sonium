#!/usr/bin/env python3
"""Fail-closed checker for the accepted-BDF PF1 CUDA breadth cell."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


CASE_ID = "pcfan.bdf-cuda-breadth.v1"
MAX_ERROR = 2.0e-9


def fail(message: str) -> None:
    raise SystemExit(f"PF1 breadth artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def nonempty_string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")


def check_hardware(value: object) -> None:
    if not isinstance(value, dict):
        fail("hardware provenance is missing")
    for key in ("device", "driver", "toolkit", "os", "compiler", "target_features", "runtime_dispatch", "topology"):
        nonempty_string(value.get(key), f"hardware.{key}")
    if value["os"] != "linux-x86_64" or value["target_features"] != "cuda" or value["runtime_dispatch"] != "cuda-native":
        fail("artifact is not native Linux/x86-64 CUDA")


def check_cpu_baseline(value: object) -> None:
    if not isinstance(value, dict) or value.get("status") != "passed":
        fail("production CPU BDF baseline did not pass")
    if value.get("flow_model") != "IncompressibleCfdFull" or value.get("bdf_scheme") != "BDF2_with_first_order_start":
        fail("BDF production formulation is missing")
    steps = value.get("accepted_steps")
    if not isinstance(steps, list) or len(steps) < 2:
        fail("at least two accepted BDF steps are required")
    previous_step = -1
    previous_time = -math.inf
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("accepted") is not True:
            fail(f"accepted CPU step {index} is missing")
        step_number = step.get("step")
        if not isinstance(step_number, int) or step_number <= previous_step:
            fail("accepted BDF step indices are not chronological")
        previous_step = step_number
        time_s = finite(step.get("time_s"), f"accepted_steps[{index}].time_s")
        if time_s <= previous_time:
            fail("accepted BDF times are not increasing")
        previous_time = time_s
        if step.get("accepted_state_step") != step_number + 1:
            fail("accepted FEM BDF state step does not match trajectory step")
        if step.get("state_finite") is not True:
            fail("CPU accepted state is not finite")
        dimension = step.get("state_dimension")
        if not isinstance(dimension, int) or dimension <= 0 or step.get("matrix_dimension") != dimension:
            fail("CPU accepted state/Jacobian dimensions are invalid")
        reference = step.get("cpu_reference")
        if not isinstance(reference, dict) or reference.get("finite") is not True:
            fail("CPU state/residual oracle is missing")
        output = reference.get("state_jacobian_apply")
        if not isinstance(output, list) or len(output) != dimension:
            fail("CPU Jacobian oracle length is invalid")
        for output_index, item in enumerate(output):
            finite(item, f"accepted_steps[{index}].cpu_reference.state_jacobian_apply[{output_index}]")
        finite(reference.get("state_residual_linf"), f"accepted_steps[{index}].state_residual_linf")
        finite(reference.get("mass_imbalance"), f"accepted_steps[{index}].mass_imbalance")
        finite(reference.get("energy_imbalance"), f"accepted_steps[{index}].energy_imbalance")
        finite(reference.get("kinetic_proxy_j"), f"accepted_steps[{index}].kinetic_proxy_j")
    if finite(value.get("observation_residual"), "production_cpu_bdf.observation_residual") < 0:
        fail("CPU observation residual is negative")
    if value.get("observation_converged") is not True:
        fail("CPU BDF observation is not converged")


def check_replay(value: object, cpu_steps: list[object]) -> None:
    if not isinstance(value, dict) or value.get("status") != "passed":
        fail("CUDA accepted-step replay did not pass")
    if value.get("backend") != "Cuda" or value.get("execution_mode") != "DeviceResidentAcceptedStepReplay":
        fail("CUDA replay is not the native resident accepted-step mode")
    if value.get("device_resident") is not True or value.get("host_staging") is not False:
        fail("CUDA replay residency is mislabelled")
    rows = value.get("steps")
    if not isinstance(rows, list) or len(rows) != len(cpu_steps) or len(rows) < 2:
        fail("CUDA replay does not cover every accepted CPU step")
    max_error = 0.0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("accepted") is not True:
            fail(f"CUDA replay step {index} is not accepted")
        if row.get("step") != cpu_steps[index].get("step"):  # type: ignore[union-attr]
            fail("CUDA/CPU accepted-step indices differ")
        dimension = row.get("matrix_dimension")
        if not isinstance(dimension, int) or dimension <= 0:
            fail(f"CUDA replay step {index} dimensions are invalid")
        observation = row.get("cuda_observation")
        if not isinstance(observation, dict) or observation.get("finite") is not True:
            fail(f"CUDA replay step {index} output is not finite")
        output = observation.get("output")
        if not isinstance(output, list) or len(output) != dimension:
            fail(f"CUDA replay step {index} output length is invalid")
        for output_index, item in enumerate(output):
            finite(item, f"cuda_replay.steps[{index}].output[{output_index}]")
        error = finite(observation.get("max_abs_error"), f"cuda_replay.steps[{index}].max_abs_error")
        if error > MAX_ERROR:
            fail(f"CUDA replay step {index} exceeded CPU oracle budget")
        max_error = max(max_error, error)
        counters = row.get("warm_counters")
        if not isinstance(counters, dict):
            fail(f"CUDA replay step {index} warm counters are missing")
        for key in ("device_allocations", "staging_allocations", "h2d_operations", "d2h_operations", "synchronizations", "kernel_and_library_operations"):
            if not isinstance(counters.get(key), int) or counters[key] < 0:
                fail(f"CUDA replay step {index} counter {key} is invalid")
        if any(counters[key] != 0 for key in ("device_allocations", "staging_allocations", "h2d_operations", "d2h_operations")):
            fail(f"CUDA replay step {index} performed warm allocation/host transfer")
        if counters["synchronizations"] < 1 or counters["kernel_and_library_operations"] < 3:
            fail(f"CUDA replay step {index} did not execute repeated resident work")
        if counters.get("live_device_bytes", 0) <= 0 or counters.get("live_staging_bytes") != 0:
            fail(f"CUDA replay step {index} live-memory census is invalid")
        if row.get("operator_retained_for_accepted_step") is not True or row.get("state_retained_for_accepted_step") is not True:
            fail(f"CUDA replay step {index} retention contract is missing")
    if finite(value.get("max_abs_error"), "cuda_accepted_step_replay.max_abs_error") != max_error:
        fail("CUDA replay aggregate error is inconsistent")
    if value.get("resources_released_on_teardown") is not True:
        fail("CUDA replay teardown did not release resources")


def check_admissions(value: object) -> None:
    if not isinstance(value, dict):
        fail("admission/recovery matrix is missing")
    malformed = value.get("malformed_csr")
    if not isinstance(malformed, dict) or malformed.get("status") != "typed_rejection" or malformed.get("typed") is not True or malformed.get("partial_result_published") is not False:
        fail("malformed CSR admission is not typed/no-publish")
    if value.get("empty_csr", {}).get("status") != "accepted_noop" or value.get("empty_csr", {}).get("partial_result_published") is not False:
        fail("empty CSR no-op coverage is missing")
    for key, expected_error in (("nonfinite_state", "NonFiniteAcceptedState"), ("solver_failure", "PcFanCfdTransientError::InvalidTimeStep")):
        row = value.get(key)
        if not isinstance(row, dict) or row.get("status") != "typed_rejection" or row.get("typed_error") != expected_error:
            fail(f"{key} is not a typed rejection")
        if row.get("partial_result_published") is not False or row.get("retry_succeeded") is not True:
            fail(f"{key} published partial output or did not retry")
    failure = value.get("launch_failure")
    if not isinstance(failure, dict) or failure.get("status") != "typed_failure_no_publication":
        fail("launch failure contract is missing")
    if failure.get("output_preserved") is not True or failure.get("recovery_succeeded") is not True or failure.get("retry_succeeded") is not True or failure.get("partial_result_published") is not False:
        fail("launch failure recovery/no-publication contract failed")
    finite(failure.get("preservation_error_max_abs"), "launch_failure.preservation_error_max_abs")
    finite(failure.get("retry_error_max_abs"), "launch_failure.retry_error_max_abs")
    if failure["preservation_error_max_abs"] > MAX_ERROR or failure["retry_error_max_abs"] > MAX_ERROR:
        fail("launch failure output parity exceeded budget")


def check(value: object) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("case_id") != CASE_ID:
        fail("schema or case id mismatch")
    check_hardware(value.get("hardware"))
    device_count = value.get("cuda_device_count")
    if not isinstance(device_count, int) or device_count < 1:
        fail("native CUDA device count is missing")
    check_cpu_baseline(value.get("production_cpu_bdf"))
    cpu_steps = value["production_cpu_bdf"]["accepted_steps"]  # type: ignore[index]
    check_replay(value.get("cuda_accepted_step_replay"), cpu_steps)
    check_admissions(value.get("admission_and_recovery"))
    reuse = value.get("reuse_validity")
    if not isinstance(reuse, dict) or reuse.get("accepted_state_history_consumed") is not True or reuse.get("accepted_state_and_residual_reused_within_step") is not True or reuse.get("operator_values_reuploaded_when_state_changed") is not True:
        fail("accepted-step reuse validity is incomplete")
    if reuse.get("full_device_resident_bdf_assembly") is not False or reuse.get("retained_pcfan_flow_state_across_full_bdf") is not False or reuse.get("retained_pcfan_aeroacoustic_coupling") is not False:
        fail("unsupported full-device boundaries are not explicit")
    acceptance = value.get("acceptance")
    if not isinstance(acceptance, dict) or any(acceptance.get(key) != expected for key, expected in (("cpu_state_residual_mass_energy_oracle", "passed"), ("native_cuda_resident_replay", "passed"), ("malformed_nonfinite_solver_launch_no_publication", "passed"), ("retry_after_typed_failure", "passed"))):
        fail("acceptance summary is incomplete")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_pf1_breadth_cuda.py ARTIFACT")
    path = Path(sys.argv[1])
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON artifact: {error}")
    check(value)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().split()[0]
        if expected != digest:
            fail("artifact sidecar digest mismatch")
    print(f"PF1 breadth CUDA artifact passed: {path} sha256={digest}")


if __name__ == "__main__":
    main()
