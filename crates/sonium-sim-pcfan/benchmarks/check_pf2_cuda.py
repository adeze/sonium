#!/usr/bin/env python3
"""Fail-closed checker for PC-fan CUDA adjoint/parity evidence.

PF2 intentionally has an incomplete CUDA derivative surface.  A valid native
artifact therefore needs both sides of that boundary: a real CUDA production
forward/parity run, and explicit typed rejection rows for the unsupported
flow/acoustic adjoint and design-residual cells.  This checker never treats a
CPU fallback, fabricated backend label, or missing derivative as CUDA evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


CASE_ID = "pcfan.adjoint-cuda-contract.v1"
SCOPE = "native_linux_x86_64_pcfan_adjoint_and_parity"
ARTIFACT = "pcfan_adjoint_cuda_contract.json"
PARITY_RELATIVE_BUDGET = 2.0e-3
PARITY_RESIDUAL_BUDGET = 1.0e-10
ADJOINT_FD_BUDGET = 5.0e-3
PAIRED_CSR_RESIDUAL_BUDGET = 2.0e-12


def fail(message: str) -> None:
    raise SystemExit(f"PF2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        fail(f"{label} is not finite")
    return value


def text(value: object, label: str) -> str:
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
    ):
        text(value.get(key), f"hardware.{key}")
    if value["os"] != "linux-x86_64":
        fail("artifact is not native Linux/x86_64")
    if value["target_features"] != "cuda":
        fail("artifact target_features does not identify CUDA")
    if value["runtime_dispatch"] != "cuda-native":
        fail("artifact runtime_dispatch is not native CUDA")


def observation_series(value: object, label: str, backend: str) -> list[dict]:
    if not isinstance(value, dict):
        fail(f"{label} dataset is missing")
    if not text(value.get("provenance"), f"{label}.provenance").startswith(
        "production:sonium-sim-pcfan:resolved-cfd:"
    ):
        fail(f"{label} provenance is not a production CFD run")
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) != 3:
        fail(f"{label} must contain the three RPM observations")
    previous_rpm = 0.0
    fields = (
        "airflow_cfm",
        "pressure_drop_pa",
        "max_velocity_m_s",
        "reynolds_number",
        "relative_mass_imbalance",
        "relative_momentum_imbalance",
        "relative_pressure_closure",
        "relative_energy_imbalance",
        "residual",
    )
    for index, row in enumerate(observations):
        if not isinstance(row, dict):
            fail(f"{label}.observations[{index}] is malformed")
        if row.get("requested_backend") != backend or row.get("actual_backend") != backend:
            fail(f"{label}.observations[{index}] backend provenance is not {backend}/{backend}")
        if row.get("converged") is not True:
            fail(f"{label}.observations[{index}] is not converged")
        rpm = finite(row.get("rpm"), f"{label}.observations[{index}].rpm")
        if rpm <= previous_rpm:
            fail(f"{label} RPM topology is not strictly increasing")
        previous_rpm = rpm
        for field in fields:
            finite(row.get(field), f"{label}.observations[{index}].{field}")
    return observations


def check_forward_parity(value: object) -> None:
    if not isinstance(value, dict):
        fail("production forward parity record is missing")
    if value.get("status") != "passed" or value.get("passed") is not True:
        fail("CPU/CUDA production parity did not pass")
    if value.get("reference_backend") != "Cpu" or value.get("candidate_backend") != "Cuda":
        fail("forward parity backends are not CPU/CUDA")
    if value.get("sample_count") != 3:
        fail("forward parity sample count is not three")
    errors = value.get("max_relative_errors")
    if not isinstance(errors, list) or not errors:
        fail("forward parity relative-error vector is missing")
    if any(finite(error, "forward parity relative error") > PARITY_RELATIVE_BUDGET for error in errors):
        fail("forward parity relative-error budget failed")
    if finite(value.get("max_residual_difference"), "forward parity residual difference") > PARITY_RESIDUAL_BUDGET:
        fail("forward parity residual budget failed")
    finite(value.get("max_field_relative_error"), "forward parity field error")
    finite(value.get("max_production_relative_error"), "forward parity production error")
    tolerance = value.get("tolerance")
    if not isinstance(tolerance, dict) or tolerance.get("allow_host_fallback") is not False:
        fail("forward parity does not reject host fallback")


def check_cpu_reference(value: object) -> None:
    if not isinstance(value, dict) or value.get("status") != "passed" or value.get("backend") != "Cpu":
        fail("CPU reduced-adjoint reference did not pass")
    text(value.get("derivative_provenance"), "CPU derivative provenance")
    if "finite-difference" in value["derivative_provenance"].lower():
        fail("CPU adjoint provenance is a finite-difference substitution")
    if not isinstance(value.get("gradient_dimension"), int) or value["gradient_dimension"] <= 0:
        fail("CPU gradient dimension is invalid")
    if finite(value.get("max_gradient_relative_error_vs_independent_fd"), "CPU gradient error") > ADJOINT_FD_BUDGET:
        fail("CPU gradient/finite-difference budget failed")
    if finite(value.get("directional_finite_difference_relative_error"), "CPU directional FD error") > ADJOINT_FD_BUDGET:
        fail("CPU directional finite-difference budget failed")
    finite(value.get("taylor_remainder_absolute"), "CPU Taylor remainder")
    finite(value.get("primal_residual_max_abs"), "CPU primal residual")
    if value.get("cuda_promotion") != "not_allowed":
        fail("CPU reference is not explicitly barred from CUDA promotion")


def check_capability(value: object, name: str) -> None:
    if not isinstance(value, dict):
        fail(f"CUDA capability row {name} is missing")
    status = value.get("status")
    if status not in {"unsupported", "typed_rejection"}:
        fail(f"CUDA capability row {name} was overclaimed: {status!r}")
    text(value.get("reason") or value.get("error"), f"CUDA capability row {name} reason")


def check_paired_csr(value: object) -> None:
    """Require the one PF2 CUDA derivative cell that is actually executable."""
    if not isinstance(value, dict) or value.get("status") != "passed":
        fail("paired-CSR CUDA Hermitian/transpose action did not pass")
    if value.get("backend") != "Cuda" or value.get("execution_mode") != "DeviceResident":
        fail("paired-CSR action is not native device-resident CUDA")
    if value.get("dispatch") != "native-cuda-cusparse":
        fail("paired-CSR action does not prove native cuSPARSE dispatch")
    if value.get("operator") != "AdjointDeviceCsr::try_prebake + DeviceAdjointOperator::try_apply_hermitian":
        fail("paired-CSR operator dispatch proof is missing")
    if value.get("scalar") != "f64 (Hermitian action is transpose for this real fixture)":
        fail("paired-CSR scalar/Hermitian contract is missing")
    if value.get("dimension") != 3 or value.get("nnz") != 7 or value.get("repeated_actions") != 2:
        fail("paired-CSR fixture topology is invalid")
    if value.get("malformed_csr_rejected") is not True:
        fail("malformed paired-CSR input was not rejected")

    cpu = value.get("cpu_reference")
    cuda = value.get("cuda_observation")
    if not isinstance(cpu, dict) or not isinstance(cuda, dict):
        fail("paired-CSR CPU/CUDA observations are missing")
    for key in ("inner_product_residual",):
        if finite(cpu.get(key), f"paired_csr.cpu.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"paired-CSR CPU {key} exceeds budget")
    for key in ("forward_residual_max_abs", "transpose_residual_max_abs", "inner_product_residual"):
        if finite(cuda.get(key), f"paired_csr.cuda.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"paired-CSR CUDA {key} exceeds budget")

    warm = value.get("warm_counters")
    if not isinstance(warm, dict):
        fail("paired-CSR warm counters are missing")
    for key in ("device_allocations", "staging_allocations", "h2d_operations"):
        if warm.get(key) != 0:
            fail(f"paired-CSR warm {key} is nonzero")
    if finite(warm.get("kernel_and_library_operations"), "paired_csr.warm.kernel_and_library_operations") <= 0:
        fail("paired-CSR warm window has no native CUDA operation")

    failure = value.get("launch_failure")
    if not isinstance(failure, dict) or failure.get("status") != "typed":
        fail("paired-CSR launch failure is not typed")
    if not text(failure.get("operation"), "paired_csr.launch_failure.operation"):
        fail("paired-CSR launch operation is missing")
    if not text(failure.get("typed_error"), "paired_csr.launch_failure.typed_error"):
        fail("paired-CSR typed launch error is missing")
    if failure.get("recovery_succeeded") is not True or failure.get("output_preserved") is not True:
        fail("paired-CSR launch recovery/output preservation failed")
    if failure.get("retry_succeeded") is not True or failure.get("partial_result_published") is not False:
        fail("paired-CSR retry/publish contract failed")
    for key in ("output_preservation_residual_max_abs", "retry_residual_max_abs"):
        if finite(failure.get(key), f"paired_csr.launch_failure.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"paired-CSR {key} exceeds budget")
    if value.get("resources_released_on_teardown") is not True:
        fail("paired-CSR teardown did not release resources")
    if value.get("live_device_bytes_after_teardown") != 0 or value.get("live_staging_bytes_after_teardown") != 0:
        fail("paired-CSR teardown left live allocations")
    coupling = value.get("high_level_design_residual_coupling")
    if not isinstance(coupling, dict) or coupling.get("status") != "unsupported":
        fail("paired-CSR row overclaims high-level design-residual coupling")


def check_production_picard_flow_transpose(value: object) -> None:
    """Check the production Picard/Oseen flow-matrix transpose boundary.

    This is intentionally a state-action contract, not an exact Newton or
    design-residual adjoint.  The provenance and scope strings are checked so
    a synthetic paired-CSR fixture cannot satisfy this production row.
    """
    if not isinstance(value, dict) or value.get("status") != "passed":
        fail("production Picard flow transpose did not pass")
    if value.get("backend") != "Cuda" or value.get("execution_mode") != "DeviceResident":
        fail("production Picard flow action is not native device-resident CUDA")
    if value.get("dispatch") != "native-cuda-cusparse":
        fail("production Picard flow action does not prove native cuSPARSE dispatch")
    if value.get("operator") != "production PcFanResolvedFlowPicardLinearization.state_jacobian":
        fail("production Picard flow operator provenance is missing")
    if value.get("flow_model") != "IncompressibleCfd":
        fail("production Picard flow model is not incompressible CFD")
    provenance = text(value.get("provenance"), "production Picard flow provenance")
    if not provenance.startswith("fem:navier-stokes:picard-oseen:"):
        fail("production Picard flow provenance is not the FEM Oseen seam")
    dimension = value.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 3:
        fail("production Picard flow dimension is not larger than the paired fixture")
    if not isinstance(value.get("nnz"), int) or value["nnz"] < dimension:
        fail("production Picard flow nnz is invalid")
    if value.get("repeated_actions") != 2:
        fail("production Picard flow did not repeat the resident action")
    cpu = value.get("cpu_reference")
    cuda = value.get("cuda_observation")
    if not isinstance(cpu, dict) or not isinstance(cuda, dict):
        fail("production Picard CPU/CUDA observations are missing")
    for key in ("forward_l2", "transpose_l2", "inner_product_residual"):
        finite(cpu.get(key), f"production_picard.cpu.{key}")
    if cpu["forward_l2"] <= 0.0 or cpu["transpose_l2"] <= 0.0:
        fail("production Picard CPU oracle is empty")
    for key in ("forward_residual_max_abs", "transpose_residual_max_abs", "inner_product_residual"):
        if finite(cuda.get(key), f"production_picard.cuda.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"production Picard CUDA {key} exceeds budget")
    warm = value.get("warm_counters")
    if not isinstance(warm, dict):
        fail("production Picard warm counters are missing")
    for key in ("device_allocations", "staging_allocations", "h2d_operations"):
        if warm.get(key) != 0:
            fail(f"production Picard warm {key} is nonzero")
    if finite(warm.get("kernel_and_library_operations"), "production_picard.warm.kernel_and_library_operations") <= 0:
        fail("production Picard warm path has no native CUDA operation")
    failure = value.get("launch_failure")
    if not isinstance(failure, dict) or failure.get("status") != "typed":
        fail("production Picard launch failure is not typed")
    text(failure.get("operation"), "production_picard.launch_failure.operation")
    text(failure.get("typed_error"), "production_picard.launch_failure.typed_error")
    if failure.get("recovery_succeeded") is not True or failure.get("output_preserved") is not True:
        fail("production Picard launch recovery/output preservation failed")
    if failure.get("retry_succeeded") is not True or failure.get("partial_result_published") is not False:
        fail("production Picard retry/publication contract failed")
    for key in ("output_preservation_residual_max_abs", "retry_residual_max_abs"):
        if finite(failure.get(key), f"production_picard.launch_failure.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"production Picard {key} exceeds budget")
    if value.get("resources_released_on_teardown") is not True:
        fail("production Picard teardown did not release resources")
    if value.get("live_device_bytes_after_teardown") != 0 or value.get("live_staging_bytes_after_teardown") != 0:
        fail("production Picard teardown left live allocations")
    scope = text(value.get("adjoint_scope"), "production Picard adjoint scope")
    if "picard/oseen" not in scope.lower() or "not exact newton/design-residual adjoint" not in scope.lower():
        fail("production Picard row does not retain its non-Newton boundary")


def check_production_newton_design_transpose(value: object) -> None:
    """Check the bounded exact-Newton/design-residual transpose cell."""
    if not isinstance(value, dict) or value.get("status") != "passed":
        fail("production Newton/design transpose did not pass")
    if value.get("backend") != "Cuda" or value.get("execution_mode") != "DeviceResident":
        fail("production Newton/design action is not native device-resident CUDA")
    if value.get("dispatch") != "native-cuda-cusparse":
        fail("production Newton/design action does not prove native cuSPARSE dispatch")
    if value.get("operator") != "production PcFanResolvedFlowNewtonLinearization.state_jacobian + residual_design_jacobian":
        fail("production Newton/design operator provenance is missing")
    if value.get("flow_model") != "IncompressibleCfd":
        fail("production Newton/design flow model is not incompressible CFD")
    provenance = text(value.get("provenance"), "production Newton/design provenance")
    if not provenance.startswith("fem:navier-stokes:newton-convection:"):
        fail("production Newton/design provenance is not the analytic FEM Newton seam")
    dimension = value.get("dimension")
    design_dimension = value.get("design_dimension")
    padded_dimension = value.get("padded_operator_dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 3:
        fail("production Newton/design state dimension is invalid")
    if not isinstance(design_dimension, int) or isinstance(design_dimension, bool) or design_dimension <= 0:
        fail("production Newton/design dimension is invalid")
    if not isinstance(padded_dimension, int) or padded_dimension < max(dimension, design_dimension):
        fail("production Newton/design padded dimension is invalid")
    if not isinstance(value.get("nnz"), int) or value["nnz"] < dimension:
        fail("production Newton/design state nnz is invalid")
    if not isinstance(value.get("design_nnz"), int) or value["design_nnz"] <= 0:
        fail("production Newton/design residual nnz is invalid")
    if value.get("repeated_actions") != 2:
        fail("production Newton/design actions were not repeated while resident")

    cpu = value.get("cpu_reference")
    cuda = value.get("cuda_observation")
    if not isinstance(cpu, dict) or not isinstance(cuda, dict):
        fail("production Newton/design CPU/CUDA observations are missing")
    for key in ("state_forward_l2", "state_transpose_l2", "design_transpose_l2"):
        if finite(cpu.get(key), f"production_newton.cpu.{key}") <= 0.0:
            fail(f"production Newton/design CPU {key} is empty")
    for key in (
        "state_inner_product_residual",
        "design_inner_product_residual",
    ):
        if finite(cpu.get(key), f"production_newton.cpu.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"production Newton/design CPU {key} exceeds budget")
    for key in ("state_finite_difference_relative_error", "design_finite_difference_relative_error"):
        if finite(cpu.get(key), f"production_newton.cpu.{key}") > 5.0e-8:
            fail(f"production Newton/design finite-difference gate failed: {key}")
    for key in (
        "state_forward_residual_max_abs",
        "state_transpose_residual_max_abs",
        "design_transpose_residual_max_abs",
        "state_inner_product_residual",
        "design_inner_product_residual",
    ):
        if finite(cuda.get(key), f"production_newton.cuda.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"production Newton/design CUDA {key} exceeds budget")

    warm = value.get("warm_counters")
    if not isinstance(warm, dict):
        fail("production Newton/design warm counters are missing")
    for key in ("device_allocations", "staging_allocations", "h2d_operations"):
        if warm.get(key) != 0:
            fail(f"production Newton/design warm {key} is nonzero")
    if finite(warm.get("kernel_and_library_operations"), "production_newton.warm.kernel_and_library_operations") <= 0:
        fail("production Newton/design warm path has no native CUDA operation")

    metadata = value.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("stale_rejected") is not True or metadata.get("malformed_rejected") is not True:
        fail("production Newton/design stale or malformed metadata was not rejected")
    failure = value.get("launch_failure")
    if not isinstance(failure, dict) or failure.get("status") != "typed":
        fail("production Newton/design launch failure is not typed")
    text(failure.get("operation"), "production_newton.launch_failure.operation")
    text(failure.get("typed_error"), "production_newton.launch_failure.typed_error")
    if failure.get("recovery_succeeded") is not True or failure.get("output_preserved") is not True:
        fail("production Newton/design launch recovery/output preservation failed")
    if failure.get("retry_succeeded") is not True or failure.get("partial_result_published") is not False:
        fail("production Newton/design retry/publication contract failed")
    for key in ("output_preservation_residual_max_abs", "retry_residual_max_abs"):
        if finite(failure.get(key), f"production_newton.launch_failure.{key}") > PAIRED_CSR_RESIDUAL_BUDGET:
            fail(f"production Newton/design {key} exceeds budget")
    if value.get("resources_released_on_teardown") is not True:
        fail("production Newton/design teardown did not release resources")
    if value.get("live_device_bytes_after_teardown") != 0 or value.get("live_staging_bytes_after_teardown") != 0:
        fail("production Newton/design teardown left live allocations")
    scope = text(value.get("adjoint_scope"), "production Newton/design adjoint scope")
    if "exact analytic newton" not in scope.lower() or "design-residual transpose" not in scope.lower():
        fail("production Newton/design row does not retain its bounded scope")
    if "not coupled acoustic/transient adjoint" not in scope.lower():
        fail("production Newton/design row overclaims coupled scope")


def check_failures(value: object) -> None:
    rows = value.get("failure_matrix")
    if not isinstance(rows, list):
        fail("failure matrix is missing")
    expected = {
        "backend_substitution",
        "unconverged_input",
        "missing_derivative_provenance",
        "budget_violation",
        "transactional_publish",
    }
    by_class = {row.get("failure_class"): row for row in rows if isinstance(row, dict)}
    if set(by_class) != expected:
        fail("failure matrix classes are incomplete")
    for name in expected:
        if by_class[name].get("status") != "typed":
            fail(f"{name} failure is not typed")
        text(by_class[name].get("contract"), f"failure_matrix.{name}.contract")


def check_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        fail("artifact checksum sidecar is missing")
    declared = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            fail("malformed artifact checksum sidecar")
        digest, name = line.split("  ", 1)
        declared[name] = digest
    if declared.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
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
    if artifact.get("scope") != SCOPE:
        fail("scope mismatch")
    check_hardware(artifact.get("hardware"))
    device_count = artifact.get("cuda_device_count")
    if not isinstance(device_count, int) or isinstance(device_count, bool) or device_count < 1:
        fail("required CUDA device count is unavailable")
    observation_series(artifact.get("production_cpu"), "production_cpu", "Cpu")
    observation_series(artifact.get("production_cuda"), "production_cuda", "Cuda")
    check_forward_parity(artifact.get("production_forward_parity"))
    check_cpu_reference(artifact.get("cpu_reference"))
    capabilities = artifact.get("cuda_adjoint_capability")
    if not isinstance(capabilities, dict):
        fail("CUDA adjoint capability matrix is missing")
    check_paired_csr(capabilities.get("paired_csr_hermitian_transpose"))
    check_production_picard_flow_transpose(capabilities.get("production_picard_flow_transpose"))
    check_production_newton_design_transpose(capabilities.get("production_newton_design_transpose"))
    for name in ("flow_transpose", "acoustic_transpose", "hermitian_transpose", "design_residual_contraction"):
        check_capability(capabilities.get(name), name)
    check_failures(artifact)
    acceptance = artifact.get("acceptance")
    if not isinstance(acceptance, dict):
        fail("acceptance summary is missing")
    expected_acceptance = {
        "production_artifact_evaluator": "executed",
        "cpu_state_ledger_objective_reference": "passed",
        "cpu_independent_finite_difference_and_taylor": "passed",
        "cuda_forward_state_ledger_comparison": "passed; see production_forward_parity",
        "cuda_flow_transpose": "unsupported_or_typed_rejected",
        "cuda_paired_csr_hermitian_transpose": "passed",
        "cuda_production_picard_flow_transpose": "passed",
        "cuda_production_newton_design_transpose": "passed",
        "cuda_acoustic_transpose": "unsupported",
        "cuda_design_residual_contraction": "unsupported",
        "cuda_adjoint_finite_difference": "blocked_by_cuda_adjoint_capability",
    }
    for key, expected in expected_acceptance.items():
        if acceptance.get(key) != expected:
            fail(f"acceptance.{key} is not {expected!r}")
    if len(sys.argv) == 3:
        try:
            log = Path(sys.argv[2]).read_text(encoding="utf-8")
        except OSError as error:
            fail(f"cannot read QA result log: {error}")
        if "QA_RESULT:" not in log or "SUCCESS" not in log:
            fail("producer QA_RESULT log is not a success")
    check_checksum(path)
    print(json.dumps({"status": "pass", "artifact": str(path), "cuda_observations": 3}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
