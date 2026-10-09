#!/usr/bin/env python3
"""Strict checker for the SM2 native CUDA DriverModel extension.

SM2 consumes a production resident-CSR replay over a bounded multi-frequency
grid. The checker requires the nested CPU/f64 golden and CUDA candidate,
per-frequency response and coupled-power rows, independent frequency
derivatives, and both typed failure/recreated-context artifacts. It never
turns the bounded network result into a full-order FEM/BEM claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import NoReturn

from check_sm1_cuda_failure import check_artifact as check_failure_artifact


CASE_ID = "speaker.cuda-sm2-multifrequency-derivative-reference.v1"
FAILURE_CASE_ID = "speaker.cuda-bounded-coupled-failure-contract.v1"
PRIMARY_FREQUENCIES = (125.0, 500.0, 1000.0, 4000.0, 16000.0)
DERIVATIVE_FRACTION = 1.0e-3
RESPONSE_TOLERANCE = 1.0e-7
DERIVATIVE_TOLERANCE = 1.0e-7
POWER_TOLERANCE = 5.0e-2
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
    raise SystemExit(f"SM2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{label} is not finite")
    return number


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    if value.strip().lower() in {"unknown", "unavailable", "unspecified", "n/a"}:
        fail(f"{label} is unavailable")
    return value


def sidecar_checksum(path: Path) -> str:
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


def expected_solve_frequencies() -> tuple[float, ...]:
    values: list[float] = []
    for frequency in PRIMARY_FREQUENCIES:
        step = frequency * DERIVATIVE_FRACTION
        values.extend((frequency - step, frequency, frequency + step))
    return tuple(values)


def check_capability_matrix(raw: object) -> None:
    if not isinstance(raw, dict):
        fail("CUDA capability matrix is missing")
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


def check_reference(raw: object, label: str, backend: str, expected_count: int) -> dict:
    if not isinstance(raw, dict):
        fail(f"{label} artifact is missing")
    if raw.get("schema_version") != 2:
        fail(f"{label} artifact schema is not v2")
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        fail(f"{label} provenance is missing")
    if provenance.get("backend") != backend or provenance.get("precision") != "f64":
        fail(f"{label} backend/precision provenance is inconsistent")
    expected_profile = "golden" if backend == "cpu" else "projected_reanalysis"
    if provenance.get("execution_profile") != expected_profile or provenance.get("deterministic") is not True:
        fail(f"{label} execution profile/determinism is inconsistent")
    execution = raw.get("execution")
    if not isinstance(execution, dict):
        fail(f"{label} execution evidence is missing")
    if execution.get("requested_backend") != backend or execution.get("actual_backend") != backend:
        fail(f"{label} requested/actual backend identity is inconsistent")
    if execution.get("precision") != "f64":
        fail(f"{label} execution precision is not f64")
    nonempty(execution.get("device"), f"{label}.execution.device")
    nonempty(execution.get("toolchain"), f"{label}.execution.toolchain")
    if execution.get("mesh_identity") != "driver-model:lumped-network:v1":
        fail(f"{label} mesh identity is not the declared bounded fixture")
    if not HEX_64.fullmatch(nonempty(execution.get("frequency_grid_identity"), f"{label}.grid")):
        fail(f"{label} frequency-grid identity is not a digest")
    if not HEX_64.fullmatch(nonempty(execution.get("reference_checksum_sha256"), f"{label}.checksum")):
        fail(f"{label} reference checksum is not a digest")
    points = raw.get("points")
    if not isinstance(points, list) or len(points) != expected_count:
        fail(f"{label} artifact must contain {expected_count} solve points")
    previous = 0.0
    for index, (point, expected_frequency) in enumerate(zip(points, expected_solve_frequencies())):
        if not isinstance(point, dict):
            fail(f"{label} point {index} is not an object")
        frequency = finite(point.get("frequency_hz"), f"{label}.points[{index}].frequency_hz")
        if not math.isclose(frequency, expected_frequency, rel_tol=1.0e-12, abs_tol=1.0e-12):
            fail(f"{label} point {index} is not on the declared perturbation grid")
        if frequency <= previous:
            fail(f"{label} frequencies are not strictly increasing")
        previous = frequency
        for field in (
            "radiation_impedance_real_n_s_per_m",
            "radiation_impedance_imag_n_s_per_m",
            "velocity_real_m_per_s",
            "velocity_imag_m_per_s",
            "electrical_impedance_real_ohm",
            "electrical_impedance_imag_ohm",
        ):
            finite(point.get(field), f"{label}.points[{index}].{field}")
        if finite(point.get("radiation_impedance_real_n_s_per_m"), f"{label}.radiation") < -1.0e-12:
            fail(f"{label} radiation resistance is not passive")
    convergence = execution.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("converged") is not True:
        fail(f"{label} convergence is not accepted")
    if finite(convergence.get("max_relative_residual"), f"{label}.residual") > 1.0e-6:
        fail(f"{label} residual exceeds the gate")
    if finite(convergence.get("power_closure_relative_error"), f"{label}.power_closure") > POWER_TOLERANCE:
        fail(f"{label} power closure exceeds the gate")
    if backend == "cuda" and finite(convergence.get("coupling_iterations"), f"{label}.coupling_iterations") < 1:
        fail(f"{label} has no coupling solve evidence")
    transfer = execution.get("transfer")
    if not isinstance(transfer, dict):
        fail(f"{label} transfer evidence is missing")
    if backend == "cuda":
        for key in ("h2d_operations", "h2d_bytes", "d2h_operations", "d2h_bytes"):
            if finite(transfer.get(key), f"{label}.transfer.{key}") <= 0:
                fail(f"{label}.transfer.{key} does not prove CUDA movement")
        peak = finite(transfer.get("allocation_high_water_bytes"), f"{label}.transfer.peak")
        live = finite(transfer.get("live_allocation_bytes"), f"{label}.transfer.live")
        if peak <= 0 or live < 0 or live > peak:
            fail(f"{label} allocation watermarks are invalid")
    promotion = execution.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("promoted") is not True:
        fail(f"{label} promotion is not accepted")
    for key, limit in (
        ("full_order_relative_residual", 1.0e-6),
        ("projected_relative_residual", 1.0e-6),
        ("gradient_relative_error", 1.0e-5),
        ("maximum_constraint_violation", 1.0e-10),
    ):
        if finite(promotion.get(key), f"{label}.promotion.{key}") > limit:
            fail(f"{label}.promotion.{key} exceeds its gate")
    if finite(promotion.get("rank_agreement"), f"{label}.promotion.rank_agreement") < 1.0 - 1.0e-10:
        fail(f"{label} promotion rank agreement is below its gate")
    return raw


def complex_pair(raw: object, label: str) -> tuple[float, float]:
    if not isinstance(raw, dict):
        fail(f"{label} is missing")
    return finite(raw.get("real"), f"{label}.real"), finite(raw.get("imaginary"), f"{label}.imaginary")


def check_envelope(path: Path) -> dict:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read SM2 envelope: {error}")
    if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
        fail("SM2 envelope schema is not v1")
    if envelope.get("case_id") != CASE_ID or envelope.get("status") != "passed":
        fail("SM2 envelope case/status is not passed")
    if envelope.get("requested_backend") != "cuda" or envelope.get("actual_backend") != "cuda":
        fail("SM2 envelope is not admitted as native CUDA")

    admission = envelope.get("capability_admission")
    if not isinstance(admission, dict) or admission.get("native_cuda_required") is not True or admission.get("fallback_status") != "none":
        fail("SM2 CUDA capability/admission is not explicit")
    check_capability_matrix(admission.get("matrix"))
    if set(admission.get("unsupported_high_level_cells", [])) != UNSUPPORTED:
        fail("SM2 unsupported high-level cells are not preserved")

    cpu_wrapper = envelope.get("cpu_reference")
    cuda_wrapper = envelope.get("cuda_reference")
    if not isinstance(cpu_wrapper, dict) or cpu_wrapper.get("backend") != "cpu" or cpu_wrapper.get("precision") != "f64" or cpu_wrapper.get("independent") is not True:
        fail("independent CPU/f64 golden wrapper is missing")
    if cpu_wrapper.get("frequency_count") != 15:
        fail("CPU golden does not retain all 15 solve points")
    check_reference(cpu_wrapper.get("artifact"), "CPU golden", "cpu", 15)
    if not isinstance(cuda_wrapper, dict) or not isinstance(cuda_wrapper.get("backend"), dict):
        fail("CUDA candidate wrapper is missing")
    if cuda_wrapper["backend"].get("requested") != "cuda" or cuda_wrapper["backend"].get("actual") != "cuda" or cuda_wrapper["backend"].get("precision") != "f64":
        fail("CUDA candidate wrapper provenance is inconsistent")
    if cuda_wrapper.get("frequency_count") != 15:
        fail("CUDA candidate does not retain all 15 solve points")
    candidate = check_reference(cuda_wrapper.get("artifact"), "CUDA candidate", "cuda", 15)
    for key, limit in (
        ("max_relative_residual", 1.0e-6),
        ("max_power_balance_relative_error", POWER_TOLERANCE),
        ("max_velocity_relative_error", RESPONSE_TOLERANCE),
        ("max_impedance_relative_error", RESPONSE_TOLERANCE),
        ("max_derivative_relative_error", RESPONSE_TOLERANCE),
    ):
        if finite(cuda_wrapper.get(key), f"CUDA candidate.{key}") > limit:
            fail(f"CUDA candidate.{key} exceeds its gate")

    sweep = envelope.get("frequency_sweep")
    if not isinstance(sweep, dict) or sweep.get("primary_frequencies_hz") != list(PRIMARY_FREQUENCIES):
        fail("primary frequency sweep identity is missing")
    if sweep.get("solve_frequency_count") != 15 or sweep.get("sparsity_structure_reused") is not True or sweep.get("resident_values_updated_between_points") is not True:
        fail("resident multi-frequency sweep provenance is incomplete")
    rows = sweep.get("points")
    if not isinstance(rows, list) or len(rows) != len(PRIMARY_FREQUENCIES):
        fail("frequency response sweep does not cover all primary points")
    for row, expected in zip(rows, PRIMARY_FREQUENCIES):
        if not isinstance(row, dict) or not math.isclose(finite(row.get("frequency_hz"), "sweep.frequency_hz"), expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            fail("frequency sweep row has the wrong frequency")
        for key in ("cpu_velocity", "cuda_velocity", "cpu_impedance", "cuda_impedance"):
            complex_pair(row.get(key), f"sweep.{key}")
        for key in ("velocity_relative_error", "impedance_relative_error"):
            if finite(row.get(key), f"sweep.{key}") > RESPONSE_TOLERANCE:
                fail(f"{key} exceeds its independent CPU/CUDA gate")

    derivative = envelope.get("frequency_response_derivative")
    if not isinstance(derivative, dict) or derivative.get("primary_point_count") != 5:
        fail("frequency-response derivative evidence is missing")
    derivative_rows = derivative.get("points")
    if not isinstance(derivative_rows, list) or len(derivative_rows) != 5:
        fail("frequency-response derivative does not cover all primary points")
    for row, expected in zip(derivative_rows, PRIMARY_FREQUENCIES):
        if not isinstance(row, dict) or not math.isclose(finite(row.get("frequency_hz"), "derivative.frequency_hz"), expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            fail("derivative row has the wrong frequency")
        if finite(row.get("step_hz"), "derivative.step_hz") <= 0:
            fail("derivative step is not positive")
        if row.get("method") != "central finite difference of independent CPU/CUDA frequency probes":
            fail("derivative method is not declared")
        for key in ("cpu_velocity_derivative", "cuda_velocity_derivative", "cpu_impedance_derivative", "cuda_impedance_derivative"):
            complex_pair(row.get(key), f"derivative.{key}")
        for key in ("velocity_relative_error", "impedance_relative_error"):
            if finite(row.get(key), f"derivative.{key}") > DERIVATIVE_TOLERANCE:
                fail(f"frequency {key} exceeds its independent gate")

    power = envelope.get("coupled_power_observable")
    if not isinstance(power, dict) or power.get("primary_point_count") != 5 or finite(power.get("power_closure_tolerance"), "power.tolerance") != POWER_TOLERANCE:
        fail("per-frequency coupled power evidence is missing")
    power_rows = power.get("points")
    if not isinstance(power_rows, list) or len(power_rows) != 5:
        fail("coupled power evidence does not cover all primary points")
    for row, expected in zip(power_rows, PRIMARY_FREQUENCIES):
        if not isinstance(row, dict) or not math.isclose(finite(row.get("frequency_hz"), "power.frequency_hz"), expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            fail("coupled power row has the wrong frequency")
        for key in ("cpu_input_power_w", "cuda_input_power_w", "cpu_radiated_power_w", "cuda_radiated_power_w"):
            finite(row.get(key), f"power.{key}")
        for key in ("input_power_relative_error", "radiated_power_relative_error"):
            if finite(row.get(key), f"power.{key}") > RESPONSE_TOLERANCE:
                fail(f"coupled {key} exceeds its CPU/CUDA gate")
        for key in ("cpu_power_closure_relative_error", "cuda_power_closure_relative_error"):
            if finite(row.get(key), f"power.{key}") > POWER_TOLERANCE:
                fail(f"coupled {key} exceeds its closure gate")

    residency = envelope.get("residency")
    if not isinstance(residency, dict) or residency.get("resident_csr_operator") is not True or residency.get("host_fem_bem_assembly") is not False:
        fail("SM2 residency scope is not explicit")
    transfer = residency.get("transfer")
    if not isinstance(transfer, dict) or finite(transfer.get("allocation_high_water_bytes"), "residency.peak") <= 0:
        fail("SM2 resident allocation watermark is missing")

    publication = envelope.get("publication")
    if not isinstance(publication, dict) or publication.get("candidate_artifact_published_after_validation") is not True or publication.get("failed_attempt_published") is not False or publication.get("cpu_golden_comparison_passed") is not True:
        fail("SM2 publication provenance is invalid")
    recovery = envelope.get("failure_recovery")
    if not isinstance(recovery, dict) or recovery.get("typed_failure_case") != FAILURE_CASE_ID or recovery.get("no_partial_publication") is not True or recovery.get("recreated_context_retry") is not True or recovery.get("artifacts") != ["speaker_driver_sm2_cuda_launch_failure.json", "speaker_driver_sm2_cuda_device_loss_failure.json"]:
        fail("SM2 failure/recovery provenance is incomplete")
    if not isinstance(envelope.get("scope"), str) or "full-order FEM/BEM" not in envelope["scope"]:
        fail("SM2 scope does not state the full-order gap")
    sidecar_checksum(path)
    return candidate


def read_result(path: Path) -> dict:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read QA_RESULT log: {error}")
    records = [line.removeprefix("QA_RESULT: ") for line in lines if line.startswith("QA_RESULT: ")]
    if len(records) != 1:
        fail("expected exactly one QA_RESULT record")
    try:
        result = json.loads(records[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is not JSON: {error}")
    if not isinstance(result, dict):
        fail("QA_RESULT is not an object")
    return result


def check_result(path: Path, artifact_paths: list[Path]) -> None:
    result = read_result(path)
    if result.get("schema_version") != 1 or result.get("case_id") != CASE_ID or result.get("status") != "passed":
        fail("QA_RESULT schema/case/status is invalid")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT is not admitted as native CUDA")
    artifacts = result.get("artifacts")
    expected_names = {item.name for item in artifact_paths}
    if not isinstance(artifacts, list) or set(artifacts) != expected_names:
        fail("QA_RESULT artifact declarations do not match the SM2 bundle")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        fail("QA_RESULT reproducibility evidence is missing")
    checksums = evidence.get("artifact_checksums")
    provenance = evidence.get("artifact_provenance")
    if not isinstance(checksums, dict) or not isinstance(provenance, dict):
        fail("QA_RESULT artifact checksums/provenance are missing")
    for artifact_path in artifact_paths:
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if checksums.get(artifact_path.name) != f"sha256:{digest}":
            fail(f"QA_RESULT checksum does not match {artifact_path.name}")
        record = provenance.get(artifact_path.name)
        if not isinstance(record, dict) or record.get("source") != "crates/sonium-qa/bin/speaker_driver_sm2_cuda_reference.rs" or record.get("version") != "v1" or record.get("checksum") != f"sha256:{digest}":
            fail(f"QA_RESULT provenance does not bind {artifact_path.name}")

    environment = result.get("environment")
    if not isinstance(environment, dict) or environment.get("platform") != "linux-x86_64" or environment.get("gpu.failure_modes_exercised") != "launch,device-loss":
        fail("QA_RESULT platform/failure-mode provenance is incomplete")
    if "no high-level FEM/BEM residency claim" not in environment.get("gpu.execution_scope", ""):
        fail("QA_RESULT execution scope is not explicit")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT runtime evidence is missing")
    hardware = runtime.get("hardware")
    if not isinstance(hardware, dict):
        fail("QA_RESULT hardware provenance is missing")
    for key in ("device", "driver", "toolkit", "cpu_model", "os", "compiler", "target_features", "runtime_dispatch", "deterministic_mode", "topology"):
        nonempty(hardware.get(key), f"QA_RESULT.hardware.{key}")
    if "nvidia" not in hardware["device"].lower() or hardware.get("os") != "linux-x86_64" or hardware.get("target_features") != "cuda" or hardware.get("runtime_dispatch") != "cuda-native":
        fail("QA_RESULT native SM120 host/dispatch provenance is inconsistent")
    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 19 or counts.get("passed") != 19 or counts.get("failed") != 0 or counts.get("unsupported") != 0:
        fail("QA_RESULT counts do not prove all sweep/failure lanes passed")
    counters = runtime.get("counters")
    if not isinstance(counters, dict):
        fail("QA_RESULT runtime counters are missing")
    for key in ("allocations", "h2d_transfers", "d2h_transfers", "synchronizations", "launches"):
        if finite(counters.get(key), f"QA_RESULT.counters.{key}") <= 0:
            fail(f"QA_RESULT.counters.{key} does not prove CUDA work")
    memory = runtime.get("memory")
    if not isinstance(memory, dict) or finite(memory.get("peak_bytes"), "QA_RESULT.memory.peak_bytes") <= 0 or finite(memory.get("live_bytes"), "QA_RESULT.memory.live_bytes") < 0:
        fail("QA_RESULT memory watermarks are invalid")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        fail("QA_RESULT metrics are missing")
    for key, limit in (
        ("max_response_relative_error", RESPONSE_TOLERANCE),
        ("max_frequency_derivative_relative_error", DERIVATIVE_TOLERANCE),
        ("max_coupled_power_relative_error", RESPONSE_TOLERANCE),
        ("max_power_closure_relative_error", POWER_TOLERANCE),
    ):
        if finite(metrics.get(key), f"QA_RESULT.metrics.{key}") > limit:
            fail(f"QA_RESULT.{key} exceeds its gate")
    if finite(metrics.get("primary_frequency_count"), "primary_frequency_count") != 5 or finite(metrics.get("solve_frequency_count"), "solve_frequency_count") != 15 or finite(metrics.get("typed_failure_lanes"), "typed_failure_lanes") != 2:
        fail("QA_RESULT sweep/failure counts are incomplete")
    for key in ("cuda_h2d_operations", "cuda_d2h_operations", "cuda_allocation_high_water_bytes", "failure_cuda_h2d_operations", "failure_cuda_d2h_operations"):
        if finite(metrics.get(key), f"QA_RESULT.{key}") <= 0:
            fail(f"QA_RESULT.{key} does not prove live CUDA work")


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            f"usage: {sys.argv[0]} ENVELOPE.json LAUNCH_FAILURE.json DEVICE_LOSS_FAILURE.json QA_RESULT.log PHASE_STATUS.tsv"
        )
    envelope_path = Path(sys.argv[1])
    failure_paths = [Path(sys.argv[2]), Path(sys.argv[3])]
    result_path = Path(sys.argv[4])
    check_envelope(envelope_path)
    check_failure_artifact(failure_paths[0], "launch")
    check_failure_artifact(failure_paths[1], "device-loss")
    check_result(result_path, [envelope_path, *failure_paths])
    try:
        rows = [line.split("\t") for line in Path(sys.argv[5]).read_text(encoding="utf-8").splitlines()]
    except OSError as error:
        fail(f"cannot read phase status: {error}")
    statuses = {row[0]: row[1] for row in rows if len(row) == 2}
    if set(statuses) != {"build_sm2_producer", "driver_model_cpu_contracts", "sm2_cuda_replay"} or any(value != "0" for value in statuses.values()):
        fail("SM2 phase status does not prove both successful phases")
    print("SM2 native CUDA multi-frequency/derivative/failure bundle passed strict checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
