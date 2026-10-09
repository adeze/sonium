#!/usr/bin/env python3
"""Fail-closed checker for the PF1 PC-fan CUDA forward contract.

The high-level PC-fan FEM bridge is deliberately allowed to remain staged;
the required CUDA acceptance is the separate bounded resident transport seam
published by the producer.  This checker refuses to promote an artifact that
calls a CPU fallback or a host round trip a device-resident pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


CASE_ID = "pcfan.forward-cuda-contract.v1"
MAX_ERROR = 2.0e-12


def fail(message: str) -> None:
    raise SystemExit(f"PF1 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def check_hardware(value: object) -> None:
    if not isinstance(value, dict):
        fail("CUDA hardware provenance is missing")
    for key in (
        "device",
        "driver",
        "toolkit",
        "os",
        "compiler",
        "target_features",
        "runtime_dispatch",
        "topology",
    ):
        string(value.get(key), f"hardware.{key}")
    if value["os"] != "linux-x86_64":
        fail("artifact is not native Linux/x86_64")
    if value["target_features"] != "cuda":
        fail("artifact target_features does not identify CUDA")
    if value["runtime_dispatch"] != "cuda-native":
        fail("artifact runtime_dispatch is not cuda-native")


def check_resident(value: object) -> None:
    if not isinstance(value, dict):
        fail("resident_forward record is missing")
    if value.get("status") != "passed":
        fail("bounded resident forward seam did not pass")
    if value.get("backend") != "Cuda" or value.get("execution_mode") != "DeviceResident":
        fail("resident seam backend/mode is not native CUDA DeviceResident")
    for key in ("device_resident", "host_staging", "state_and_operator_retained", "source_retained"):
        expected = key != "host_staging"
        if value.get(key) is not expected:
            fail(f"resident_forward.{key} has an invalid value")
    if value.get("steps") != 4 or value.get("dimension") != 4 or value.get("nnz") != 10:
        fail("resident fixture dimensions or step count changed")
    if finite(value.get("max_abs_error"), "resident max_abs_error") > MAX_ERROR:
        fail("resident host parity budget failed")
    if finite(value.get("mass_error"), "resident mass_error") > MAX_ERROR:
        fail("resident conservation budget failed")

    counters = value.get("warm_counters")
    if not isinstance(counters, dict):
        fail("resident warm counters are missing")
    for key in (
        "device_allocations",
        "staging_allocations",
        "h2d_operations",
        "d2h_operations",
        "d2d_operations",
        "synchronizations",
        "kernel_and_library_operations",
    ):
        count = counters.get(key)
        if not isinstance(count, int) or count < 0:
            fail(f"resident warm counter {key} is invalid")
    if any(counters[key] != 0 for key in ("device_allocations", "staging_allocations", "h2d_operations", "d2h_operations")):
        fail("resident warm trajectory allocated or transferred through host")
    if counters["synchronizations"] < 1 or counters["kernel_and_library_operations"] < 1:
        fail("resident warm trajectory did not execute/synchronize")
    if counters.get("live_device_bytes", 0) <= 0 or counters.get("live_staging_bytes", 0) != 0:
        fail("resident live-memory census is invalid")


def check_production_picard(value: object) -> None:
    """Check a real FEM Picard CSR replay, not the bounded transport fixture."""
    if not isinstance(value, dict) or value.get("status") != "passed":
        fail("production Picard resident record did not pass")
    if value.get("backend") != "Cuda" or value.get("execution_mode") != "DeviceResident":
        fail("production Picard action is not native CUDA DeviceResident")
    if value.get("requested_backend") != "Cuda" or value.get("actual_backend") != "Cuda":
        fail("production Picard requested/actual backend provenance is inconsistent")
    if value.get("assembly_residency") != "Host" or value.get("host_staging") is not False:
        fail("production Picard assembly/staging boundary is mislabelled")
    if value.get("operator") != "production PcFanCfdFlowSolution.picard_linearization.state_jacobian":
        fail("production Picard operator is not the published FEM state Jacobian")
    provenance = string(value.get("matrix_provenance"), "resident_production_picard.matrix_provenance")
    if provenance != "fem:navier-stokes:picard-oseen:v1":
        fail("production Picard matrix provenance is not the FEM Oseen path")
    dimension = value.get("dimension")
    nnz = value.get("nnz")
    if not isinstance(dimension, int) or dimension <= 0 or not isinstance(nnz, int) or nnz <= 0:
        fail("production Picard matrix dimensions are invalid")
    if not isinstance(value.get("empty_rows"), int) or value["empty_rows"] < 0:
        fail("production Picard empty-row coverage is missing")
    if not isinstance(value.get("tail_lanes"), int) or value["tail_lanes"] <= 0:
        fail("production Picard CUDA tail coverage is missing")
    if not isinstance(value.get("repeated_actions"), int) or value["repeated_actions"] < 3:
        fail("production Picard resident replay count is too small")
    if value.get("device_resident") is not True or value.get("operator_retained") is not True or value.get("state_retained") is not True:
        fail("production Picard resident ownership is incomplete")

    cpu_reference = value.get("cpu_reference")
    cuda_observation = value.get("cuda_observation")
    if not isinstance(cpu_reference, dict) or not isinstance(cuda_observation, dict):
        fail("production Picard CPU/CUDA observations are missing")
    cpu_output = cpu_reference.get("output")
    cuda_output = cuda_observation.get("output")
    if not isinstance(cpu_output, list) or len(cpu_output) != dimension:
        fail("production Picard CPU oracle length is invalid")
    if not isinstance(cuda_output, list) or len(cuda_output) != dimension:
        fail("production Picard CUDA output length is invalid")
    for index, (reference, actual) in enumerate(zip(cpu_output, cuda_output)):
        finite(reference, f"resident_production_picard.cpu_reference.output[{index}]")
        finite(actual, f"resident_production_picard.cuda_observation.output[{index}]")
    if finite(cuda_observation.get("max_abs_error"), "resident_production_picard.max_abs_error") > 2.0e-9:
        fail("production Picard CUDA/CPU oracle residual exceeded budget")

    counters = value.get("warm_counters")
    if not isinstance(counters, dict):
        fail("production Picard warm counters are missing")
    for key in (
        "device_allocations",
        "staging_allocations",
        "h2d_operations",
        "d2h_operations",
        "d2d_operations",
        "synchronizations",
        "kernel_and_library_operations",
    ):
        if not isinstance(counters.get(key), int) or counters[key] < 0:
            fail(f"production Picard warm counter {key} is invalid")
    if any(counters[key] != 0 for key in ("device_allocations", "staging_allocations", "h2d_operations", "d2h_operations")):
        fail("production Picard warm replay allocated or transferred through host")
    if counters["synchronizations"] < 1 or counters["kernel_and_library_operations"] < 3:
        fail("production Picard warm replay did not execute repeatedly")
    if counters.get("live_device_bytes", 0) <= 0 or counters.get("live_staging_bytes", 0) != 0:
        fail("production Picard live-memory census is invalid")

    admission = value.get("admission")
    if not isinstance(admission, dict):
        fail("production Picard admission matrix is missing")
    malformed = admission.get("malformed_csr")
    if not isinstance(malformed, dict) or malformed.get("status") != "typed_rejection":
        fail("malformed production Picard CSR was not rejected")
    string(malformed.get("typed_error"), "resident_production_picard.malformed_csr.typed_error")
    if malformed.get("partial_result_published") is not False:
        fail("malformed production Picard CSR published output")
    if admission.get("empty_csr", {}).get("status") != "accepted_noop":
        fail("empty production Picard CSR no-op coverage is missing")
    if admission.get("empty_csr", {}).get("partial_result_published") is not False:
        fail("empty production Picard CSR published output")
    tail = admission.get("tail_case")
    if not isinstance(tail, dict) or tail.get("status") != "passed" or tail.get("matrix_action_includes_partial_tail") is not True:
        fail("production Picard tail action was not proven")

    failure = value.get("launch_failure")
    if not isinstance(failure, dict) or failure.get("status") != "typed":
        fail("production Picard launch failure record is missing")
    for key in ("operation", "typed_error", "context_recovery"):
        string(failure.get(key), f"resident_production_picard.launch_failure.{key}")
    if failure.get("recovery_succeeded") is not True or failure.get("output_preserved") is not True:
        fail("production Picard launch failure was not recovered without output mutation")
    if failure.get("partial_result_published") is not False or failure.get("retry_succeeded") is not True:
        fail("production Picard launch failure published or failed to retry")
    if finite(failure.get("output_preservation_residual_max_abs"), "resident_production_picard.preserved_residual") > 2.0e-9:
        fail("production Picard failed output was not preserved")
    if finite(failure.get("retry_residual_max_abs"), "resident_production_picard.retry_residual") > 2.0e-9:
        fail("production Picard retry did not match the successful output")
    if value.get("resources_released_on_teardown") is not True:
        fail("production Picard teardown did not release resources")
    for key in ("live_device_bytes_after_teardown", "live_staging_bytes_after_teardown", "pool_bytes_in_use_after_teardown"):
        if value.get(key) != 0:
            fail(f"production Picard {key} is non-zero")
    unsupported = value.get("unsupported_high_level")
    if not isinstance(unsupported, list) or set(unsupported) != {
        "on_device_pcfan_fem_assembly",
        "retained_pcfan_flow_state_across_full_bdf",
        "retained_pcfan_aeroacoustic_coupling",
    }:
        fail("production Picard high-level unsupported boundary is incomplete")


def check_adjoint(value: object) -> None:
    if not isinstance(value, dict):
        fail("resident_adjoint record is missing")
    if value.get("status") != "passed":
        fail("bounded resident adjoint seam did not pass")
    if value.get("backend") != "Cuda" or value.get("execution_mode") != "DeviceResident":
        fail("resident adjoint backend/mode is not native CUDA DeviceResident")
    if value.get("dimension") != 3 or value.get("nnz") != 7 or value.get("repeated_actions") != 2:
        fail("resident adjoint fixture dimensions or repeat count changed")
    cpu_reference = value.get("cpu_reference")
    cuda_observation = value.get("cuda_observation")
    if not isinstance(cpu_reference, dict) or not isinstance(cuda_observation, dict):
        fail("resident adjoint CPU/CUDA reference records are missing")
    for key in (
        "forward_residual_max_abs",
        "transpose_residual_max_abs",
    ):
        if finite(cuda_observation.get(key), f"resident_adjoint.{key}") > MAX_ERROR:
            fail(f"resident adjoint {key} exceeded CPU reference budget")
    if finite(cpu_reference.get("inner_product_residual"), "resident_adjoint.cpu_inner_product_residual") > MAX_ERROR:
        fail("resident adjoint CPU inner-product identity failed")
    if finite(cuda_observation.get("inner_product_residual"), "resident_adjoint.cuda_inner_product_residual") > MAX_ERROR:
        fail("resident adjoint CUDA inner-product identity failed")

    counters = value.get("warm_counters")
    if not isinstance(counters, dict):
        fail("resident adjoint warm counters are missing")
    for key in (
        "device_allocations",
        "staging_allocations",
        "h2d_operations",
        "d2h_operations",
        "synchronizations",
        "kernel_and_library_operations",
    ):
        count = counters.get(key)
        if not isinstance(count, int) or count < 0:
            fail(f"resident adjoint warm counter {key} is invalid")
    if any(counters[key] != 0 for key in ("device_allocations", "staging_allocations", "h2d_operations")):
        fail("resident adjoint warm trajectory allocated or uploaded through host")
    if counters["d2h_operations"] != 0:
        fail("resident adjoint warm trajectory downloaded through host")
    if counters["synchronizations"] < 1 or counters["kernel_and_library_operations"] < 2:
        fail("resident adjoint warm trajectory did not execute/synchronize repeatedly")
    if counters.get("live_device_bytes", 0) <= 0 or counters.get("live_staging_bytes", 0) != 0:
        fail("resident adjoint live-memory census is invalid")

    failure = value.get("launch_failure")
    if not isinstance(failure, dict) or failure.get("status") != "typed":
        fail("resident adjoint launch failure record is missing")
    for key in ("operation", "typed_error", "context_recovery"):
        string(failure.get(key), f"resident_adjoint.launch_failure.{key}")
    if failure.get("context_recovery") != "synchronize_before_reuse":
        fail("resident adjoint launch failure recovery disposition is invalid")
    if failure.get("recovery_succeeded") is not True:
        fail("resident adjoint launch failure was not recovered")
    if failure.get("output_preserved") is not True or failure.get("partial_result_published") is not False:
        fail("resident adjoint launch failure mutated or published output")
    if failure.get("retry_succeeded") is not True:
        fail("resident adjoint launch failure did not retry successfully")
    if finite(failure.get("output_preservation_residual_max_abs"), "resident_adjoint.output_preservation_residual_max_abs") > MAX_ERROR:
        fail("resident adjoint failed output was not preserved")
    if finite(failure.get("retry_residual_max_abs"), "resident_adjoint.retry_residual_max_abs") > MAX_ERROR:
        fail("resident adjoint retry did not match the successful output")

    if value.get("resources_released_on_teardown") is not True:
        fail("resident adjoint resources were not released")
    for key in (
        "live_device_bytes_after_teardown",
        "live_staging_bytes_after_teardown",
        "pool_bytes_in_use_after_teardown",
    ):
        if value.get(key) != 0:
            fail(f"resident adjoint {key} is non-zero")
    coupling = value.get("high_level_design_residual_coupling")
    if not isinstance(coupling, dict) or coupling.get("status") != "unsupported":
        fail("high-level PC-fan design-residual coupling was overclaimed")
    if coupling.get("partial_result_published") is not False:
        fail("unsupported high-level design coupling published a result")


def check_steady(value: object) -> None:
    if not isinstance(value, dict):
        fail("steady section is missing")
    cuda = value.get("cuda")
    if not isinstance(cuda, dict):
        fail("steady CUDA production row is missing")
    if cuda.get("actual_backend") != "Cuda" or cuda.get("execution_mode") != "AcceleratorLinearSolve":
        fail("steady CUDA row does not identify the staged linear-solve seam")
    if cuda.get("device_resident") is not False or cuda.get("hidden_host_staging") is not True:
        fail("steady staged CUDA row is mislabelled")
    if not isinstance(value.get("resident_admission"), str) or "typed_rejection" not in value["resident_admission"]:
        fail("complete resident admission was not rejected explicitly")


def check_transient(value: object) -> None:
    if not isinstance(value, dict):
        fail("transient section is missing")
    cpu = value.get("cpu")
    if not isinstance(cpu, dict) or cpu.get("accepted_steps", 0) < 2 or cpu.get("converged") is not True:
        fail("CPU BDF baseline is incomplete")
    cuda = value.get("cuda")
    if not isinstance(cuda, dict) or cuda.get("status") != "typed_rejection":
        fail("CUDA BDF substitution was not rejected")
    if cuda.get("accepted_steps") != 0 or cuda.get("actual_backend") is not None:
        fail("rejected CUDA BDF trajectory published output")


def check_failures(value: object) -> None:
    rows = value.get("failure_matrix")
    if not isinstance(rows, list):
        fail("failure matrix is missing")
    by_class = {row.get("failure_class"): row for row in rows if isinstance(row, dict)}
    expected_typed = {"solver", "nonfinite"}
    expected_injected = {
        "setup",
        "allocation",
        "transfer",
        "launch",
        "synchronization",
        "device_loss",
        "cancellation",
        "non_convergence",
        "teardown",
    }
    if set(by_class) != expected_typed | expected_injected:
        fail("failure matrix classes are incomplete")
    for name in expected_typed:
        if by_class[name].get("status") != "typed":
            fail(f"{name} failure is not typed")
    for name in expected_injected:
        row = by_class[name]
        if row.get("status") != "injected":
            fail(f"{name} production injection did not reach a typed failure")
        for key in ("backend", "stage", "operation", "detail", "typed_error", "context_recovery"):
            string(row.get(key), f"failure_matrix.{name}.{key}")
        if row.get("backend") != "cuda" or row.get("stage") != name:
            fail(f"{name} failure provenance is not native CUDA")
        if row.get("partial_result_published") is not False:
            fail(f"{name} failure published a partial result")
        if row.get("resources_released_on_unwind") is not True:
            fail(f"{name} failure did not release resident resources")
        if not isinstance(row.get("recovery_status"), str) or not row["recovery_status"]:
            fail(f"{name} recovery status is missing")
        if not isinstance(row.get("recovery_succeeded"), bool):
            fail(f"{name} recovery result is missing")
        if not isinstance(row.get("live_device_bytes_after_unwind"), int) or row["live_device_bytes_after_unwind"] != 0:
            fail(f"{name} left live device bytes after unwind")
        if not isinstance(row.get("live_staging_bytes_after_unwind"), int) or row["live_staging_bytes_after_unwind"] != 0:
            fail(f"{name} left live staging bytes after unwind")
        if not isinstance(row.get("pool_bytes_in_use_after_unwind"), int) or row["pool_bytes_in_use_after_unwind"] != 0:
            fail(f"{name} left pool bytes after unwind")
    if by_class["device_loss"].get("context_recovery") != "recreate":
        fail("device-loss recovery does not require backend recreation")
    if by_class["synchronization"].get("context_recovery") != "recreate":
        fail("synchronization recovery does not require backend recreation")
    if by_class["teardown"].get("pending_error_retained") is not True:
        fail("teardown injection was not retained by the production backend")
    for name in ("cancellation", "non_convergence"):
        if by_class[name].get("recovery_succeeded") is not True:
            fail(f"{name} gate did not leave the stream reusable")


def check_checksum(path: Path) -> None:
    # The runner writes the artifact-local sidecar before invoking this
    # checker.  The aggregate checksums file is written after the checker, so
    # consulting it here would make a second run compare against stale bytes.
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        fail("artifact checksum sidecar is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    declared = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            fail("malformed checksum sidecar")
        value, name = line.split("  ", 1)
        declared[name] = value
    if declared.get(path.name) != digest:
        fail("artifact checksum is stale")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json [QA_RESULT.log]")
    path = Path(sys.argv[1]).resolve()
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON: {error}")
    if artifact.get("schema_version") != 1 or artifact.get("case_id") != CASE_ID:
        fail("schema or case id mismatch")
    if artifact.get("scope") != "native_linux_x86_64_cuda_forward_contract":
        fail("scope mismatch")
    check_hardware(artifact.get("hardware"))
    device_count = artifact.get("cuda_device_count")
    if not isinstance(device_count, int) or device_count < 1:
        fail("required CUDA device count is unavailable")
    check_resident(artifact.get("resident_forward"))
    check_production_picard(artifact.get("resident_production_picard"))
    check_adjoint(artifact.get("resident_adjoint"))
    check_steady(artifact.get("steady"))
    check_transient(artifact.get("transient"))
    check_failures(artifact)
    acceptance = artifact.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("bounded_device_resident_algebra") != "passed_or_typed_unavailable":
        fail("acceptance summary is missing the bounded resident seam")
    if acceptance.get("complete_device_resident_steady") != "not_admitted" or acceptance.get("complete_device_resident_transient") != "not_admitted":
        fail("unsupported high-level residency was overclaimed")
    if acceptance.get("bounded_device_resident_adjoint") != "passed_or_typed_unavailable":
        fail("bounded device-resident adjoint acceptance summary is missing")
    if acceptance.get("production_picard_device_resident") != "passed_or_typed_unavailable":
        fail("production Picard resident acceptance summary is missing")
    if len(sys.argv) == 3:
        log = Path(sys.argv[2])
        try:
            log_text = log.read_text(encoding="utf-8")
        except OSError as error:
            fail(f"cannot read QA result log: {error}")
        if "QA_RESULT:" not in log_text or "SUCCESS" not in log_text:
            fail("producer QA_RESULT log is not a success")
    check_checksum(path)
    print(json.dumps({"status": "pass", "artifact": str(path), "resident_steps": 4}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
