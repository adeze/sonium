#!/usr/bin/env python3
"""Strict checker for the bounded native CUDA SK2 speaker evidence.

This checker validates the report contracts emitted by the production speaker
QA binaries.  It does not turn unsupported product configurations into a
pass, and it keeps the four typed-contract failure probes distinct from the
five production-entry-point failure probes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


FAILURE_MODES = [
    "allocation",
    "device_loss",
    "cancellation",
    "non_convergence",
    "residual_mismatch",
    "invalid_geometry",
    "stale_cache",
    "insufficient_sampling",
    "failed_rom_promotion",
]
PRODUCTION_FAILURE_MODES = [
    "allocation",
    "device_loss",
    "cancellation",
    "non_convergence",
    "invalid_geometry",
]
TYPED_FAILURE_MODES = [
    "residual_mismatch",
    "stale_cache",
    "insufficient_sampling",
    "failed_rom_promotion",
]
NATIVE_BACKEND_FAILURE_LANES = [
    "transfer_h2d",
    "launch",
    "synchronization",
    "device_loss",
]
PRODUCTION_LIFECYCLE_FAILURE_LANES = [
    "descriptor_creation",
    "transfer_h2d",
    "launch",
    "synchronization",
]


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SK2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        fail(f"{label} is not a positive integer")
    return value


def exact_bool(value: object, expected: bool, label: str) -> None:
    if value is not expected:
        fail(f"{label} must be {expected}")


def read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"{label} is not readable JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    return value


def verify_checksums(directory: Path) -> None:
    checksum_path = directory / "checksums.sha256"
    try:
        rows = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{directory.name} checksum manifest is missing: {error}")
    if not rows:
        fail(f"{directory.name} checksum manifest is empty")
    seen: set[str] = set()
    for row in rows:
        fields = row.split()
        if len(fields) != 2:
            fail(f"{directory.name} checksum row is malformed: {row!r}")
        digest, name = fields
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            fail(f"{directory.name} checksum for {name} is malformed")
        if Path(name).name != name or name in seen:
            fail(f"{directory.name} checksum path is unsafe or duplicated: {name!r}")
        seen.add(name)
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest().lower() != digest.lower():
            fail(f"{directory.name} checksum mismatch for {name}")


def read_qa_result(path: Path, label: str) -> dict:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{label} QA log is unreadable: {error}")
    records = [line.removeprefix("QA_RESULT: ") for line in lines if line.startswith("QA_RESULT: ")]
    if len(records) != 1:
        fail(f"{label} QA log must contain exactly one QA_RESULT record")
    try:
        result = json.loads(records[0])
    except json.JSONDecodeError as error:
        fail(f"{label} QA_RESULT is not JSON: {error}")
    if not isinstance(result, dict):
        fail(f"{label} QA_RESULT is not an object")
    if result.get("status") != "passed" or result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail(f"{label} QA_RESULT does not prove a passed native CUDA execution")
    return result


def check_qa_artifact_provenance(result: dict, artifact_names: list[str], label: str) -> None:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or any(name not in artifacts for name in artifact_names):
        fail(f"{label} QA_RESULT does not declare every expected artifact")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        fail(f"{label} QA_RESULT evidence is missing")
    checksums = evidence.get("artifact_checksums")
    provenance = evidence.get("artifact_provenance")
    if not isinstance(checksums, dict) or not isinstance(provenance, dict):
        fail(f"{label} QA_RESULT artifact provenance/checksums are missing")
    for name in artifact_names:
        if name not in checksums or name not in provenance:
            fail(f"{label} QA_RESULT provenance is missing {name}")
        expected = str(checksums[name])
        actual = provenance[name].get("checksum") if isinstance(provenance[name], dict) else None
        if not isinstance(actual, str) or actual != expected:
            fail(f"{label} QA_RESULT checksum/provenance disagree for {name}")
        if not expected.startswith("sha256:") or len(expected) != len("sha256:") + 64:
            fail(f"{label} QA_RESULT digest is malformed for {name}")


def check_common_report(
    report: dict,
    label: str,
    *,
    schema_version: int = 2,
    require_evaluation_points: bool = True,
    require_failure_modes: bool = True,
    require_performance_eligible: bool = True,
) -> None:
    if report.get("schema_version") != schema_version:
        fail(f"{label} report schema is not version {schema_version}")
    if report.get("requested_backend") != "cuda" or report.get("actual_backend") != "cuda":
        fail(f"{label} report does not prove requested/actual CUDA")
    if report.get("frequencies") != 2 or (
        require_evaluation_points and report.get("evaluation_points") != 72
    ):
        fail(f"{label} report has an unexpected frequency/evaluation grid")
    positive_int(report.get("active_dofs"), f"{label}.active_dofs")
    if report.get("residual_gate") != "passed":
        fail(f"{label} residual gate did not pass")
    if require_performance_eligible:
        exact_bool(report.get("performance_eligible"), False, f"{label}.performance_eligible")
    timings = report.get("timings")
    if not isinstance(timings, dict):
        fail(f"{label}.timings is missing")
    for key in ("assembly_ms", "solve_ms", "residual_gate_ms", "field_ms", "reference_ms", "total_ms"):
        value = timings.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{label}.timings.{key} is not a non-negative integer")
    if timings["total_ms"] < timings["assembly_ms"]:
        fail(f"{label} total timing is below assembly timing")
    transfers = report.get("transfers")
    if not isinstance(transfers, dict):
        fail(f"{label}.transfers is missing")
    for key in ("h2d_operations", "h2d_bytes", "d2h_operations", "d2h_bytes", "d2d_operations", "d2d_bytes"):
        positive_int(transfers.get(key), f"{label}.transfers.{key}")
    memory = report.get("memory")
    if not isinstance(memory, dict):
        fail(f"{label}.memory is missing")
    for key in ("live_allocation_bytes", "allocation_high_water_bytes", "live_staging_bytes", "staging_high_water_bytes"):
        value = memory.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{label}.memory.{key} is malformed")
    if memory["allocation_high_water_bytes"] <= 0 or memory["live_allocation_bytes"] > memory["allocation_high_water_bytes"]:
        fail(f"{label} memory does not prove a bounded live CUDA allocation")
    if require_failure_modes:
        failures = report.get("failure_modes_fail_closed")
        if failures != FAILURE_MODES:
            fail(f"{label} failure-mode list is incomplete or reordered")
    capability = report.get("capability_matrix")
    if not isinstance(capability, dict):
        fail(f"{label}.capability_matrix is missing")
    for key in (
        "regular_pair_assembly_c32",
        "regular_pair_assembly_c64",
        "near_pair_assembly",
        "singular_pair_assembly",
        "hypersingular_pair_assembly",
        "field_evaluation",
        "derivative_assembly",
    ):
        if capability.get(key) not in {"native", "vendor_native"}:
            fail(f"{label}.capability_matrix.{key} is not native")
    configurations = capability.get("speaker_configurations")
    if not isinstance(configurations, dict):
        fail(f"{label} speaker configuration capability matrix is missing")
    for key in ("crossover_phase", "coordinate_rotation", "body_fitted_projected_reanalysis"):
        cell = configurations.get(key)
        if not isinstance(cell, dict) or cell.get("state") != "unsupported" or not str(cell.get("reason", "")).strip():
            fail(f"{label} unsupported configuration {key} is not fail-closed")


def check_whole(root: Path) -> None:
    report_path = root / "speaker_cuda_execution_report.json"
    adjoint_path = root / "speaker_cuda_adjoint_report.json"
    preview_path = root / "speaker_cuda_reduced_order_preview.json"
    report = read_json(report_path, "whole-speaker report")
    adjoint = read_json(adjoint_path, "whole-speaker adjoint report")
    preview = read_json(preview_path, "whole-speaker preview")
    check_common_report(report, "whole-speaker")
    cache = report.get("cache")
    if not isinstance(cache, dict):
        fail("whole-speaker cache report is missing")
    for key in ("mesh_resident_across_frequencies", "operator_structure_resident", "field_geometry_resident", "assembly_geometry_reused", "rhs_buffer_reused", "solution_buffer_reused"):
        exact_bool(cache.get(key), True, f"whole-speaker.cache.{key}")
    for key in ("matrix_value_updates", "preconditioner_updates"):
        if cache.get(key) != 1:
            fail(f"whole-speaker.cache.{key} does not prove one resident update")
    config = report["capability_matrix"]["speaker_configurations"]
    if config.get("prescribed_forward", {}).get("state") not in {"native", "vendor_native"}:
        fail("whole-speaker prescribed forward is not native")
    for key in ("typed_multi_driver",):
        if config.get(key, {}).get("state") != "unsupported":
            fail(f"whole-speaker {key} was not kept unsupported")
    physical = report.get("max_physical_error")
    residual = report.get("max_relative_residual")
    derivative = report.get("max_derivative_assembly_error")
    if finite(residual, "whole-speaker.max_relative_residual") > 1e-4 or finite(physical, "whole-speaker.max_physical_error") > 1e-3 or finite(derivative, "whole-speaker.max_derivative_assembly_error") > 5e-3:
        fail("whole-speaker numerical gate exceeds its declared budget")
    check_common_report(
        adjoint,
        "whole-speaker adjoint",
        schema_version=1,
        require_evaluation_points=False,
        require_failure_modes=False,
        require_performance_eligible=False,
    )
    if adjoint.get("objective_points") != 72 or not all(adjoint.get(key) is True for key in ("operator_structure_resident", "rhs_buffer_reused", "solution_buffer_reused")):
        fail("whole-speaker adjoint does not prove resident objective solve")
    for key, budget in (("max_relative_residual", 1e-4), ("max_cpu_reference_error", 5e-3), ("max_inner_product_error", 1e-10)):
        if finite(adjoint.get(key), f"whole-speaker adjoint.{key}") > budget:
            fail(f"whole-speaker adjoint.{key} exceeds its budget")
    if preview.get("schema_version") != 1 or not isinstance(preview.get("preview"), dict):
        fail("whole-speaker reduced-order preview schema is invalid")
    reduced = preview["preview"]
    if len(reduced.get("frequencies", [])) != 2 or len(reduced.get("spl", [])) != 2 or len(reduced.get("phase", [])) != 2:
        fail("whole-speaker reduced-order preview shape is invalid")
    for key in ("spl", "phase"):
        if any(not isinstance(row, list) or len(row) != 72 for row in reduced[key]):
            fail(f"whole-speaker preview {key} shape is invalid")


def check_typed(root: Path) -> None:
    report = read_json(root / "speaker_typed_cuda_execution_report.json", "typed report")
    check_common_report(report, "typed")
    cache = report.get("cache")
    if not isinstance(cache, dict):
        fail("typed cache report is missing")
    for key in ("mesh_resident_across_frequencies", "operator_structure_resident", "field_geometry_resident", "coupled_operator_structure_resident", "coupled_rhs_buffer_reused", "coupled_solution_buffer_reused", "coupled_block_rhs_resident", "coupled_block_solution_resident"):
        exact_bool(cache.get(key), True, f"typed.cache.{key}")
    if cache.get("coupled_block_rhs_count") != 2 or not isinstance(cache.get("coupled_block_iterations"), int) or cache["coupled_block_iterations"] <= 0:
        fail("typed cache does not prove the two-driver resident block solve")
    for key in ("matrix_value_updates", "preconditioner_updates", "coupled_matrix_value_updates", "coupled_preconditioner_updates"):
        if cache.get(key) != 1:
            fail(f"typed.cache.{key} does not prove one resident update")
    config = report["capability_matrix"]["speaker_configurations"]
    if config.get("typed_multi_driver", {}).get("state") not in {"native", "vendor_native"}:
        fail("typed multi-driver configuration is not native")
    for key, budget in (("max_relative_residual", 1e-4), ("max_physical_error", 1e-3), ("max_coupled_power_error", 2e-3), ("max_coupled_observable_error", 2e-3), ("max_coupled_derivative_error", 2e-3), ("max_derivative_assembly_error", 5e-3)):
        if finite(report.get(key), f"typed.{key}") > budget:
            fail(f"typed.{key} exceeds its budget")


def check_failure(root: Path) -> None:
    artifact = read_json(root / "speaker_cuda_failure_contract.json", "failure contract")
    if artifact.get("schema_version") != 1 or artifact.get("case_id") != "speaker.cuda-failure-contract.v1" or artifact.get("requested_backend") != "cuda" or artifact.get("actual_backend") != "cuda":
        fail("failure contract identity/backend is invalid")
    if artifact.get("failure_modes_exercised") != FAILURE_MODES:
        fail("failure contract does not exercise the exact nine failure classes")
    if artifact.get("production_entry_point_failure_modes") != PRODUCTION_FAILURE_MODES:
        fail("failure contract production-entry-point list is incomplete")
    if artifact.get("typed_contract_failure_modes") != TYPED_FAILURE_MODES:
        fail("failure contract typed-probe list is incomplete")
    lanes = artifact.get("native_backend_failure_lanes")
    if not isinstance(lanes, list) or [row.get("failure_class") for row in lanes if isinstance(row, dict)] != NATIVE_BACKEND_FAILURE_LANES:
        fail("native backend failure lane list is incomplete or reordered")
    for index, row in enumerate(lanes):
        if not isinstance(row, dict):
            fail(f"native_backend_failure_lanes[{index}] is malformed")
        name = NATIVE_BACKEND_FAILURE_LANES[index]
        if row.get("status") not in {"typed_failure_and_retry", "typed_failure_and_recreate_retry"}:
            fail(f"native backend lane {name} did not record a typed failure")
        if row.get("backend") != "cuda" or row.get("error_kind") != name.removesuffix("_h2d"):
            fail(f"native backend lane {name} has inconsistent provenance")
        if not isinstance(row.get("typed_error"), str) or not row["typed_error"].strip():
            fail(f"native backend lane {name} lacks its typed error")
        if row.get("output_unchanged") is not True or row.get("partial_result_published") is not False:
            fail(f"native backend lane {name} violated no-publication/output preservation")
        if row.get("retry_succeeded") is not True:
            fail(f"native backend lane {name} did not recover/retry")
    lifecycle = artifact.get("production_lifecycle_failure_lanes")
    if not isinstance(lifecycle, list) or [
        row.get("failure_class") for row in lifecycle if isinstance(row, dict)
    ] != PRODUCTION_LIFECYCLE_FAILURE_LANES:
        fail("high-level production lifecycle failure lane list is incomplete or reordered")
    for index, row in enumerate(lifecycle):
        if not isinstance(row, dict):
            fail(f"production_lifecycle_failure_lanes[{index}] is malformed")
        name = PRODUCTION_LIFECYCLE_FAILURE_LANES[index]
        if row.get("status") != "typed_failure_no_publication":
            fail(f"high-level production lane {name} did not record a typed failure")
        if row.get("requested_backend") != "cuda" or row.get("actual_backend") != "cuda":
            fail(f"high-level production lane {name} backend provenance is invalid")
        if not isinstance(row.get("typed_error"), str) or not row["typed_error"].strip():
            fail(f"high-level production lane {name} lacks its typed error")
        if row.get("partial_result_published") is not False or row.get("failure_report_published") is not False:
            fail(f"high-level production lane {name} published a partial result")
        if not isinstance(row.get("resources_released_on_failure"), str) or not row["resources_released_on_failure"].strip():
            fail(f"high-level production lane {name} lacks unwind provenance")
        retry = row.get("retry")
        if not isinstance(retry, dict) or retry.get("succeeded") is not True or retry.get("requested_backend") != "cuda" or retry.get("actual_backend") != "cuda" or retry.get("finite_product_result") is not True:
            fail(f"high-level production lane {name} did not prove native retry")
        if not isinstance(retry.get("allocation_high_water_bytes"), int) or retry["allocation_high_water_bytes"] <= 0:
            fail(f"high-level production lane {name} retry allocation watermark is invalid")
        if not isinstance(retry.get("live_allocation_bytes"), int) or retry["live_allocation_bytes"] > retry["allocation_high_water_bytes"]:
            fail(f"high-level production lane {name} retry exceeded allocation watermark")
        if not isinstance(retry.get("staging_high_water_bytes"), int) or retry["staging_high_water_bytes"] < 0:
            fail(f"high-level production lane {name} retry staging watermark is invalid")
        if not isinstance(retry.get("live_staging_bytes"), int) or retry["live_staging_bytes"] > retry["staging_high_water_bytes"]:
            fail(f"high-level production lane {name} retry exceeded staging watermark")
        if retry.get("post_teardown_live_allocation_bytes") != 0 or retry.get("post_teardown_live_staging_bytes") != 0:
            fail(f"high-level production lane {name} retry retained resources after teardown")
        if retry.get("allocation_ledger_bounded") is not True or retry.get("staging_ledger_bounded") is not True or retry.get("post_teardown_clean") is not True:
            fail(f"high-level production lane {name} retry omitted ledger cleanup evidence")
        for key in ("h2d_operations", "d2h_operations"):
            if not isinstance(retry.get(key), int) or retry[key] <= 0:
                fail(f"high-level production lane {name} retry {key} is missing")
        if retry.get("timing_reported") is not True:
            fail(f"high-level production lane {name} retry timing provenance is missing")


def check_fallback(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"required-CUDA fallback probe is unreadable: {error}")
    if "CUDA fallback fault injection is active" not in text or "FAILED" not in text:
        fail("required-CUDA fallback probe did not fail closed")
    records = [line.removeprefix("QA_RESULT: ") for line in text.splitlines() if line.startswith("QA_RESULT: ")]
    if len(records) != 1:
        fail("required-CUDA fallback probe did not emit exactly one QA_RESULT")
    result = json.loads(records[0])
    if result.get("status") != "failed" or result.get("requested_backend") != "cuda" or result.get("actual_backend") != "host-fallback":
        fail("required-CUDA fallback probe provenance is inconsistent")


def main() -> int:
    if len(sys.argv) != 2:
        fail(f"usage: {sys.argv[0]} SK2_ROOT")
    root = Path(sys.argv[1])
    whole = root / "whole"
    typed = root / "typed"
    failure = root / "failure"
    for directory in (whole, typed, failure):
        if not directory.is_dir():
            fail(f"missing evidence directory {directory}")
        verify_checksums(directory)
    whole_result = read_qa_result(whole / "qa_result.log", "whole-speaker")
    check_qa_artifact_provenance(
        whole_result,
        [
            "speaker_cuda_execution_report.json",
            "speaker_cuda_adjoint_report.json",
            "speaker_cuda_reduced_order_preview.json",
        ],
        "whole-speaker",
    )
    typed_result = read_qa_result(typed / "qa_result.log", "typed")
    check_qa_artifact_provenance(typed_result, ["speaker_typed_cuda_execution_report.json"], "typed")
    failure_result = read_qa_result(failure / "qa_result.log", "failure")
    check_qa_artifact_provenance(failure_result, ["speaker_cuda_failure_contract.json"], "failure")
    check_whole(whole)
    check_typed(typed)
    check_failure(failure)
    check_fallback(whole / "fallback_probe.log")
    print(json.dumps({"status": "passed", "case_id": "speaker.cuda-bounded-whole-workload.v1", "failure_modes": len(FAILURE_MODES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
