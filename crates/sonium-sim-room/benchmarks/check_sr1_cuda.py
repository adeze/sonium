#!/usr/bin/env python3
"""Fail-closed validation for the production sim-room SR1 CUDA artifact."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def check_cancellation_artifact(artifact: dict) -> int:
    if artifact.get("schema_version") != 1:
        print("SR1 cancellation artifact error: unsupported schema", file=sys.stderr)
        return 2
    if artifact.get("status") != "passed":
        print("SR1 cancellation artifact error: run did not pass", file=sys.stderr)
        return 1
    backend = artifact.get("backend")
    if not isinstance(backend, dict) or (
        backend.get("requested") != "cuda"
        or backend.get("actual") != "cuda"
        or backend.get("fallback_status") != "none"
        or not isinstance(backend.get("precision"), str)
    ):
        print("SR1 cancellation artifact error: CUDA provenance is not native", file=sys.stderr)
        return 1
    cancellation = artifact.get("cancellation")
    if not isinstance(cancellation, dict) or (
        cancellation.get("stage") != "mid-frequency"
        or cancellation.get("typed_error") != "Cancelled"
        or cancellation.get("backend") != "cuda"
        or not isinstance(cancellation.get("reason"), str)
        or not cancellation.get("reason")
        or not isinstance(cancellation.get("frequency_index"), int)
        or isinstance(cancellation.get("frequency_index"), bool)
        or cancellation.get("frequency_index") < 0
        or cancellation.get("partial_result_published") is not False
        or cancellation.get("success_artifact_published") is not False
    ):
        print("SR1 cancellation artifact error: typed no-publish evidence is missing", file=sys.stderr)
        return 1
    failure = artifact.get("failure_diagnostics")
    if not isinstance(failure, dict) or (
        failure.get("backend") != "cuda"
        or failure.get("stage") != "cancellation"
        or not isinstance(failure.get("operation"), str)
        or not failure.get("operation")
        or not isinstance(failure.get("detail"), str)
        or not failure.get("detail")
        or failure.get("partial_result_published") is not False
        or failure.get("resources_released_on_unwind") is not True
        or failure.get("context_recovery") != "reusable"
    ):
        print("SR1 cancellation artifact error: structured failure diagnostics are missing", file=sys.stderr)
        return 1
    if failure.get("frequency_index") is not None or failure.get("source_index") is not None:
        print("SR1 cancellation artifact error: cancellation location must not claim an uncommitted source", file=sys.stderr)
        return 1
    cleanup = artifact.get("cleanup")
    if not isinstance(cleanup, dict) or (
        cleanup.get("owned_resources_drop") != "Rust error-unwind Drop"
        or cleanup.get("context_reacquired") is not True
        or cleanup.get("device_allocation_leak_counter")
        != "not-exposed by sonium-compute"
        or cleanup.get("proof") != "error-unwind-drop-and-context-reacquisition"
    ):
        print("SR1 cancellation artifact error: cleanup/reuse evidence is missing", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "stage": cancellation["stage"],
                "frequency_index": cancellation["frequency_index"],
                "typed_error": cancellation["typed_error"],
                "backend": backend["actual"],
                "cleanup": cleanup,
            },
            sort_keys=True,
        )
    )
    return 0


def check_failure_artifact(artifact: dict) -> int:
    if artifact.get("schema_version") != 1:
        print("SR1 failure artifact error: unsupported schema", file=sys.stderr)
        return 2
    if artifact.get("status") != "passed":
        print("SR1 failure artifact error: run did not pass", file=sys.stderr)
        return 1
    injection = artifact.get("injection")
    if not isinstance(injection, dict) or (
        injection.get("point") not in {"launch", "device-loss"}
        or injection.get("one_shot") is not True
        or injection.get("native_backend") != "cuda"
        or injection.get("expected_operation")
        not in {"cuda_launch", "cuda_device_loss"}
    ):
        print("SR1 failure artifact error: native one-shot injection provenance is missing", file=sys.stderr)
        return 1
    expected = {
        "launch": ("cuda_launch", "launch", "synchronize-before-reuse"),
        "device-loss": ("cuda_device_loss", "setup", "recreate"),
    }[injection["point"]]
    if injection["expected_operation"] != expected[0]:
        print("SR1 failure artifact error: injection operation does not match point", file=sys.stderr)
        return 1

    failure = artifact.get("failure_diagnostics")
    if not isinstance(failure, dict) or (
        failure.get("backend") != "cuda"
        or failure.get("stage") != expected[1]
        or failure.get("operation") != expected[0]
        or not isinstance(failure.get("detail"), str)
        or "injected CUDA" not in failure.get("detail", "")
        or failure.get("partial_result_published") is not False
        or failure.get("resources_released_on_unwind") is not True
        or failure.get("context_recovery") != expected[2]
    ):
        print("SR1 failure artifact error: structured failure provenance is missing", file=sys.stderr)
        return 1

    publication = artifact.get("publication")
    if not isinstance(publication, dict) or (
        publication.get("partial_result_published") is not False
        or publication.get("success_artifact_published_before_retry") is not False
        or publication.get("retry_artifact_published") is not True
    ):
        print("SR1 failure artifact error: no-partial-publication provenance is missing", file=sys.stderr)
        return 1

    unwind = artifact.get("unwind")
    if not isinstance(unwind, dict) or (
        unwind.get("owned_resources_drop") != "Rust error-unwind Drop"
        or not isinstance(unwind.get("allocation_ledger"), str)
        or not unwind.get("allocation_ledger")
        or unwind.get("live_allocation_after_unwind")
        != "not-observable after backend ownership ends"
    ):
        print("SR1 failure artifact error: unwind/resource-ledger provenance is missing", file=sys.stderr)
        return 1

    recovery = artifact.get("recovery")
    if not isinstance(recovery, dict) or (
        recovery.get("required") != expected[2]
        or recovery.get("retry_mode") != "recreated-context"
        or recovery.get("context_recreated") is not True
        or recovery.get("retry_success") is not True
    ):
        print("SR1 failure artifact error: recovery/retry provenance is missing", file=sys.stderr)
        return 1

    retry = artifact.get("retry_success")
    if not isinstance(retry, dict):
        print("SR1 failure artifact error: retry evidence is missing", file=sys.stderr)
        return 1
    retry_backend = retry.get("backend")
    if not isinstance(retry_backend, dict) or (
        retry_backend.get("requested") != "cuda"
        or retry_backend.get("actual") != "cuda"
        or retry_backend.get("fallback_status") != "none"
        or not isinstance(retry_backend.get("precision"), str)
    ):
        print("SR1 failure artifact error: retry is not native CUDA", file=sys.stderr)
        return 1
    if (
        not isinstance(retry.get("frequency_count"), int)
        or retry.get("frequency_count") <= 0
        or retry.get("all_frequencies_converged") is not True
        or not isinstance(retry.get("allocation_high_water_bytes"), int)
        or retry.get("allocation_high_water_bytes") <= 0
        or not isinstance(retry.get("estimated_peak_memory_bytes"), int)
        or retry.get("estimated_peak_memory_bytes") <= 0
    ):
        print("SR1 failure artifact error: retry watermark/convergence evidence is missing", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "injection": injection["point"],
                "failure_stage": failure["stage"],
                "failure_operation": failure["operation"],
                "context_recovery": recovery["required"],
                "retry_frequency_count": retry["frequency_count"],
                "retry_all_frequencies_converged": retry["all_frequencies_converged"],
                "retry_allocation_high_water_bytes": retry["allocation_high_water_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


def check_resident_recovery_artifact(artifact: dict) -> int:
    """Validate cancellation after warm commits and a fresh production retry."""
    if artifact.get("schema_version") != 1 or artifact.get("status") != "passed":
        print("SR1 resident recovery artifact error: unsupported or failed schema", file=sys.stderr)
        return 2
    backend = artifact.get("backend")
    if not isinstance(backend, dict) or (
        backend.get("requested") != "cuda"
        or backend.get("actual") != "cuda"
        or backend.get("fallback_status") != "none"
    ):
        print("SR1 resident recovery artifact error: retry provenance is not native CUDA", file=sys.stderr)
        return 1
    cancellation = artifact.get("cancellation")
    if not isinstance(cancellation, dict) or (
        cancellation.get("stage") != "mid-frequency"
        or cancellation.get("frequency_index") != 2
        or cancellation.get("committed_frequency_count") != 2
        or cancellation.get("typed_error") != "Cancelled"
        or cancellation.get("backend") != "cuda"
        or not isinstance(cancellation.get("reason"), str)
        or "frequency 2" not in cancellation.get("reason", "")
        or cancellation.get("partial_result_published") is not False
        or cancellation.get("success_artifact_published") is not False
    ):
        print("SR1 resident recovery artifact error: warm cancellation no-publish evidence is missing", file=sys.stderr)
        return 1
    failure = artifact.get("failure_diagnostics")
    if not isinstance(failure, dict) or (
        failure.get("backend") != "cuda"
        or failure.get("stage") != "cancellation"
        or failure.get("operation") != "cooperative-cancellation"
        or failure.get("partial_result_published") is not False
        or failure.get("resources_released_on_unwind") is not True
        or failure.get("context_recovery") != "reusable"
    ):
        print("SR1 resident recovery artifact error: typed unwind diagnostics are missing", file=sys.stderr)
        return 1
    cleanup = artifact.get("cleanup")
    if not isinstance(cleanup, dict) or (
        cleanup.get("owned_resources_drop") != "Rust error-unwind Drop"
        or cleanup.get("context_reacquired") is not True
        or cleanup.get("device_allocation_leak_counter") != "not-exposed by sonium-compute"
    ):
        print("SR1 resident recovery artifact error: cleanup evidence is missing", file=sys.stderr)
        return 1
    retry = artifact.get("retry")
    if not isinstance(retry, dict) or (
        retry.get("mode") != "fresh-context"
        or retry.get("context_recreated") is not True
        or retry.get("success") is not True
        or retry.get("frequency_count") != 4
        or retry.get("all_frequencies_converged") is not True
        or retry.get("preconditioner_ages") != [0, 1, 2, 3]
        or retry.get("preconditioner_rebuild_reasons") != ["scheduled", "cached", "cached", "cached"]
    ):
        print("SR1 resident recovery artifact error: four-frequency retry contract is missing", file=sys.stderr)
        return 1
    retry_backend = retry.get("backend")
    if not isinstance(retry_backend, dict) or (
        retry_backend.get("requested") != "cuda"
        or retry_backend.get("actual") != "cuda"
        or retry_backend.get("fallback_status") != "none"
    ):
        print("SR1 resident recovery artifact error: retry backend is not native CUDA", file=sys.stderr)
        return 1
    stats = retry.get("frequency_stats")
    if not isinstance(stats, list) or len(stats) != 4:
        print("SR1 resident recovery artifact error: retry frequency telemetry is incomplete", file=sys.stderr)
        return 1
    watermarks = []
    for index, frequency in enumerate(stats):
        if not isinstance(frequency, dict) or (
            frequency.get("converged") is not True
            or frequency.get("preconditioner_age") != index
            or frequency.get("preconditioner_rebuild_reason")
            != ("scheduled" if index == 0 else "cached")
        ):
            print(f"SR1 resident recovery artifact error: retry frequency {index} cache contract failed", file=sys.stderr)
            return 1
        residual = frequency.get("max_residual")
        if not isinstance(residual, (int, float)) or isinstance(residual, bool) or not math.isfinite(residual):
            print(f"SR1 resident recovery artifact error: retry frequency {index} residual is invalid", file=sys.stderr)
            return 2
        transfer = frequency.get("transfer")
        if not isinstance(transfer, dict) or (
            transfer.get("d2h_bytes", 0) <= 0
            or transfer.get("d2d_operations", 0) <= 0
        ):
            print(f"SR1 resident recovery artifact error: retry frequency {index} transfers are incomplete", file=sys.stderr)
            return 1
        watermark = frequency.get("allocation_high_water_bytes")
        if not isinstance(watermark, int) or isinstance(watermark, bool) or watermark <= 0:
            print(f"SR1 resident recovery artifact error: retry frequency {index} watermark is invalid", file=sys.stderr)
            return 1
        watermarks.append(watermark)
    growth = max(watermarks) - min(watermarks)
    if watermarks != sorted(watermarks) or growth > 4096 or growth * 100 > watermarks[0]:
        print("SR1 resident recovery artifact error: retry watermark growth is unbounded", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "passed",
        "stage": cancellation["stage"],
        "frequency_index": cancellation["frequency_index"],
        "committed_frequency_count": cancellation["committed_frequency_count"],
        "retry_frequency_count": retry["frequency_count"],
        "retry_allocation_high_water_bytes": retry["allocation_high_water_bytes"],
        "retry_allocation_high_water_growth_bytes": growth,
    }, sort_keys=True))
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_sr1_cuda.py ARTIFACT.json", file=sys.stderr)
        return 2
    try:
        artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"SR1 artifact error: {error}", file=sys.stderr)
        return 2

    if artifact.get("artifact") == "sonium-sim-room.sr1-cuda-cancellation":
        return check_cancellation_artifact(artifact)
    if artifact.get("artifact") == "sonium-sim-room.sr1-cuda-failure-injection":
        return check_failure_artifact(artifact)
    if artifact.get("artifact") == "sonium-sim-room.sr1-cuda-resident-recovery":
        return check_resident_recovery_artifact(artifact)
    if artifact.get("schema_version") != 1 or artifact.get("artifact") != "sonium-sim-room.sr1-cuda":
        print("SR1 artifact error: unsupported schema", file=sys.stderr)
        return 2
    if artifact.get("status") != "passed":
        print("SR1 artifact error: required CUDA run did not pass", file=sys.stderr)
        return 1
    backend = artifact.get("backend")
    if not isinstance(backend, dict) or (
        backend.get("requested") != "cuda"
        or backend.get("actual") != "cuda"
        or backend.get("fallback_status") != "none"
    ):
        print("SR1 artifact error: CUDA provenance is not native and fallback-free", file=sys.stderr)
        return 1
    frequencies = artifact.get("frequency_stats")
    if not isinstance(frequencies, list) or not frequencies:
        print("SR1 artifact error: frequency diagnostics are missing", file=sys.stderr)
        return 2
    high_water = []
    for index, stats in enumerate(frequencies):
        if not isinstance(stats, dict) or stats.get("converged") is not True:
            print(f"SR1 artifact error: frequency {index} did not converge", file=sys.stderr)
            return 1
        residual = stats.get("max_residual")
        if not isinstance(residual, (int, float)) or isinstance(residual, bool) or not math.isfinite(residual):
            print(f"SR1 artifact error: frequency {index} residual is invalid", file=sys.stderr)
            return 2
        transfer = stats.get("transfer")
        if not isinstance(transfer, dict) or transfer.get("d2h_bytes", 0) <= 0:
            print(f"SR1 artifact error: frequency {index} lacks device result transfer", file=sys.stderr)
            return 1
        value = stats.get("allocation_high_water_bytes")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            print(f"SR1 artifact error: frequency {index} lacks allocation watermark", file=sys.stderr)
            return 1
        high_water.append(value)

    if artifact.get("workload") == "f64-resident-reuse-4-frequency":
        if len(frequencies) != 4:
            print("SR1 resident artifact error: expected four production frequencies", file=sys.stderr)
            return 1
        ages = [stats.get("preconditioner_age") for stats in frequencies]
        reasons = [stats.get("preconditioner_rebuild_reason") for stats in frequencies]
        if ages != [0, 1, 2, 3] or reasons != ["scheduled", "cached", "cached", "cached"]:
            print("SR1 resident artifact error: measured preconditioner reuse sequence is missing", file=sys.stderr)
            return 1
        if any(stats["transfer"].get("d2d_operations", 0) <= 0 for stats in frequencies):
            print("SR1 resident artifact error: device-resident value refresh telemetry is missing", file=sys.stderr)
            return 1

    # The backend watermark is lifetime-cumulative.  A cached preconditioner
    # changes which temporary vector overlaps the resident workspace on the
    # second frequency, so an exact equality check incorrectly rejects a
    # bounded one-vector peak (the live resident set itself is reused).  Keep
    # the gate fail-closed by permitting only monotonic, sub-1%, sub-4 KiB
    # growth across the bounded two-frequency fixture.
    watermark_growth = max(high_water) - min(high_water)
    if high_water != sorted(high_water) or watermark_growth > 4096 or watermark_growth * 100 > high_water[0]:
        print("SR1 artifact error: warm allocation watermark growth is unbounded", file=sys.stderr)
        return 1
    cancellation = artifact.get("cancellation")
    if not isinstance(cancellation, dict) or cancellation.get("pre_setup_test") != "cuda_cancellation_rejects_before_device_setup":
        print("SR1 artifact error: cancellation provenance is missing", file=sys.stderr)
        return 2
    parity = artifact.get("parity")
    if not isinstance(parity, dict) or parity.get("sample_count", 0) <= 0:
        print("SR1 artifact error: CPU/CUDA parity samples are missing", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "workload": artifact.get("workload"),
                "frequency_count": len(frequencies),
                "allocation_high_water_bytes": high_water[-1],
                "allocation_high_water_growth_bytes": watermark_growth,
                "parity_samples": parity["sample_count"],
                "cancellation": cancellation,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
