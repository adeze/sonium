#!/usr/bin/env python3
"""Fail-closed checker for the SR2 CPU/CUDA physical-parity artifact."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_sr2_cuda.py ARTIFACT.json [FAILURE_ARTIFACT.json ...]", file=sys.stderr)
        return 2
    try:
        artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"SR2 artifact error: {error}", file=sys.stderr)
        return 2

    if artifact.get("schema_version") != 1 or artifact.get("artifact") != "sonium-sim-room.sr2-cuda":
        print("SR2 artifact error: unsupported schema", file=sys.stderr)
        return 2
    if artifact.get("status") != "passed":
        print("SR2 artifact error: required CUDA parity did not pass", file=sys.stderr)
        return 1
    contract = artifact.get("contract")
    if not isinstance(contract, dict) or contract.get("same_fixture") is not True:
        print("SR2 artifact error: identical-input contract is missing", file=sys.stderr)
        return 1
    if min(contract.get("source_count", 0), contract.get("listener_count", 0), contract.get("frequency_count", 0)) < 2:
        print("SR2 artifact error: multi-source/listener/frequency matrix is incomplete", file=sys.stderr)
        return 1
    backend = artifact.get("backend")
    if not isinstance(backend, dict) or backend.get("requested") != "cuda" or backend.get("actual") != "cuda" or backend.get("fallback_status") != "none":
        print("SR2 artifact error: CUDA provenance is not native and fallback-free", file=sys.stderr)
        return 1
    memory = artifact.get("memory")
    if not isinstance(memory, dict) or not isinstance(memory.get("estimated_peak_bytes"), int) or memory["estimated_peak_bytes"] <= 0:
        print("SR2 artifact error: device memory provenance is missing", file=sys.stderr)
        return 1
    transfer = artifact.get("transfer")
    if not isinstance(transfer, dict) or transfer.get("setup", {}).get("h2d_bytes", 0) <= 0 or transfer.get("total", {}).get("d2h_bytes", 0) <= 0:
        print("SR2 artifact error: transfer provenance is missing", file=sys.stderr)
        return 1
    frequencies = artifact.get("frequency_stats")
    if not isinstance(frequencies, list) or len(frequencies) != contract["frequency_count"]:
        print("SR2 artifact error: frequency diagnostics are missing", file=sys.stderr)
        return 2
    high_water = []
    for index, stats in enumerate(frequencies):
        if not isinstance(stats, dict) or stats.get("converged") is not True:
            print(f"SR2 artifact error: frequency {index} did not converge", file=sys.stderr)
            return 1
        residual = stats.get("max_residual")
        if not isinstance(residual, (int, float)) or isinstance(residual, bool) or not math.isfinite(residual):
            print(f"SR2 artifact error: frequency {index} residual is invalid", file=sys.stderr)
            return 2
        if stats.get("transfer", {}).get("d2h_bytes", 0) <= 0:
            print(f"SR2 artifact error: frequency {index} lacks result transfer", file=sys.stderr)
            return 1
        watermark = stats.get("allocation_high_water_bytes")
        if not isinstance(watermark, int) or isinstance(watermark, bool) or watermark <= 0:
            print(f"SR2 artifact error: frequency {index} lacks allocation watermark", file=sys.stderr)
            return 1
        high_water.append(watermark)
    if len(set(high_water)) != 1:
        print("SR2 artifact error: warm allocation watermark is not stable", file=sys.stderr)
        return 1
    parity = artifact.get("parity")
    if not isinstance(parity, dict) or parity.get("sample_count", 0) <= 0:
        print("SR2 artifact error: pressure parity samples are missing", file=sys.stderr)
        return 1
    budgets = parity.get("budgets", {})
    for metric in ("rms_relative_error", "max_relative_error", "physical_max_relative_error", "metric_max_relative_error"):
        value = parity.get(metric)
        budget_name = {
            "rms_relative_error": "pressure_rms_relative",
            "max_relative_error": "pressure_max_relative",
            "physical_max_relative_error": "physical_max_relative",
            "metric_max_relative_error": "metric_max_relative",
        }[metric]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > budgets.get(budget_name, -1):
            print(f"SR2 artifact error: {metric} exceeds its declared budget", file=sys.stderr)
            return 1
    failures = artifact.get("failure_matrix")
    if not isinstance(failures, list) or {item.get("class") for item in failures if isinstance(item, dict)} != {"malformed_input", "cancellation", "solver_nonconvergence", "device_loss"}:
        print("SR2 artifact error: failure matrix is incomplete", file=sys.stderr)
        return 2
    failure_status = {item["class"]: item.get("status") for item in failures}
    if failure_status.get("malformed_input") != "passed" or failure_status.get("cancellation") != "passed":
        print("SR2 artifact error: checked malformed/cancellation contracts are not passed", file=sys.stderr)
        return 1
    if any(failure_status.get(name) != "passed" for name in ("solver_nonconvergence", "device_loss")):
        print("SR2 artifact error: deterministic solver/device failure lanes did not pass", file=sys.stderr)
        return 1
    failure_artifacts = {
        item["class"]: item.get("artifact")
        for item in failures
        if item.get("class") in ("solver_nonconvergence", "device_loss")
    }
    supplied = sys.argv[2:]
    if len(supplied) != 2:
        print("SR2 artifact error: both deterministic failure artifacts are required", file=sys.stderr)
        return 2
    checked_failure_classes = set()
    for supplied_path in supplied:
        try:
            failure_artifact = json.loads(Path(supplied_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"SR2 failure artifact error: {error}", file=sys.stderr)
            return 2
        if failure_artifact.get("schema_version") != 1 or failure_artifact.get("artifact") != "sonium-sim-room.sr2-cuda-failure" or failure_artifact.get("status") != "passed":
            print(f"SR2 failure artifact error: invalid artifact {supplied_path}", file=sys.stderr)
            return 1
        contract = failure_artifact.get("contract", {})
        if contract.get("same_fixture") is not True or contract.get("native_backend") != "cuda" or contract.get("physical_parity_matrix_unchanged") is not True:
            print(f"SR2 failure artifact error: fixture/backend contract missing in {supplied_path}", file=sys.stderr)
            return 1
        injection = failure_artifact.get("injection", {})
        failure_class = injection.get("class")
        if failure_class not in ("solver_nonconvergence", "device_loss") or injection.get("deterministic") is not True or injection.get("native_backend") != "cuda" or injection.get("physical_reset") is not False:
            print(f"SR2 failure artifact error: injection provenance invalid in {supplied_path}", file=sys.stderr)
            return 1
        if failure_artifacts.get(failure_class) != Path(supplied_path).name:
            print(f"SR2 failure artifact error: baseline does not reference {supplied_path}", file=sys.stderr)
            return 1
        diagnostics = failure_artifact.get("failure_diagnostics", {})
        expected = {
            "solver_nonconvergence": ("non-convergence", "cuda_solver_nonconvergence", "synchronize-before-reuse"),
            "device_loss": ("setup", "cuda_device_loss", "recreate"),
        }[failure_class]
        if (diagnostics.get("backend"), diagnostics.get("stage"), diagnostics.get("operation"), diagnostics.get("context_recovery")) != ("cuda", *expected):
            print(f"SR2 failure artifact error: typed failure diagnostics invalid in {supplied_path}", file=sys.stderr)
            return 1
        if diagnostics.get("partial_result_published") is not False or diagnostics.get("resources_released_on_unwind") is not True or not diagnostics.get("detail"):
            print(f"SR2 failure artifact error: unwind/publication provenance invalid in {supplied_path}", file=sys.stderr)
            return 1
        publication = failure_artifact.get("publication", {})
        if publication.get("partial_result_published") is not False or publication.get("success_artifact_published_before_retry") is not False or publication.get("retry_artifact_published") is not True or publication.get("fallback_attempted") is not False:
            print(f"SR2 failure artifact error: partial publication/fallback contract invalid in {supplied_path}", file=sys.stderr)
            return 1
        unwind = failure_artifact.get("unwind", {})
        if unwind.get("resources_released_on_unwind") is not True or not unwind.get("failure_live_allocation_watermark"):
            print(f"SR2 failure artifact error: resource unwind evidence missing in {supplied_path}", file=sys.stderr)
            return 1
        recovery = failure_artifact.get("recovery", {})
        if recovery.get("required") != expected[2] or recovery.get("retry_mode") != "recreated-context" or recovery.get("context_recreated") is not True or recovery.get("retry_success") is not True:
            print(f"SR2 failure artifact error: recovery provenance invalid in {supplied_path}", file=sys.stderr)
            return 1
        retry = failure_artifact.get("retry_success", {})
        retry_backend = retry.get("backend", {})
        if retry_backend.get("requested") != "cuda" or retry_backend.get("actual") != "cuda" or retry_backend.get("fallback_status") != "none" or retry.get("all_frequencies_converged") is not True or retry.get("frequency_count", 0) < 2 or retry.get("allocation_high_water_bytes", 0) <= 0:
            print(f"SR2 failure artifact error: recreated-context retry did not pass in {supplied_path}", file=sys.stderr)
            return 1
        checked_failure_classes.add(failure_class)
    if checked_failure_classes != {"solver_nonconvergence", "device_loss"}:
        print("SR2 failure artifact error: both deterministic classes are required", file=sys.stderr)
        return 2
    print(json.dumps({"status": "passed", "frequency_count": len(frequencies), "parity_samples": parity["sample_count"], "failure_classes": sorted(checked_failure_classes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
