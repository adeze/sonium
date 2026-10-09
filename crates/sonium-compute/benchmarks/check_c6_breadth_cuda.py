#!/usr/bin/env python3
"""Fail-closed checker for measured native CUDA BEM breadth evidence.

The checker validates the timing sample statistics and resident runtime
counters independently of the producer.  It deliberately rejects any
roofline, energy, or vendor-comparison claim because this workload has no
profiler/instruction attribution or locally available vendor quadrature path.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn


SCHEMA = "sonium-compute-cuda-bem-quadrature-c6-breadth/v1"
CASE_ID = "compute.cuda-bem-quadrature-c6-breadth.v1"
TOLERANCE = 8.0e-12
TIMING_SAMPLES = 9
ACTIONS_PER_SAMPLE = 4
TIERS = {"small_odd": (7, 5), "medium_odd": (31, 5), "large_odd": (127, 5)}


def fail(message: str) -> NoReturn:
    raise ValueError(f"C6 breadth CUDA evidence rejected: {message}")


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{field} must be non-empty text")
    return value


def integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{field} must be an integer")
    if value < (1 if positive else 0):
        fail(f"{field} must be {'positive' if positive else 'non-negative'}")
    return value


def number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
        fail(f"{field} must be finite and {'positive' if positive else 'non-negative'}")
    return value


def provenance_number(value: Any, field: str) -> float:
    """Parse the QA provenance convention (numeric values are strings)."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            fail(f"{field} must contain a numeric value")
    return number(value, field, positive=True)


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        fail(f"{field} must be finite")
    return value


def close(actual: float, expected: float, field: str, *, rel_tol: float = 1.0e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=1.0e-15):
        fail(f"{field} is {actual!r}, expected derived value {expected!r}")


def check_timing(row: dict[str, Any], field: str, work_units: int) -> dict[str, float]:
    samples = row.get("samples_ms")
    if not isinstance(samples, list) or len(samples) != TIMING_SAMPLES:
        fail(f"{field}.samples_ms must contain exactly {TIMING_SAMPLES} samples")
    measured = [number(value, f"{field}.samples_ms[{index}]", positive=True) for index, value in enumerate(samples)]
    ordered = sorted(measured)
    median = number(row.get("median_ms"), f"{field}.median_ms", positive=True)
    close(median, ordered[len(ordered) // 2], f"{field}.median_ms", rel_tol=1.0e-12)
    deviations = sorted(abs(value - median) for value in measured)
    mad = number(row.get("mad_ms"), f"{field}.mad_ms")
    close(mad, deviations[len(deviations) // 2], f"{field}.mad_ms", rel_tol=1.0e-12)
    minimum = number(row.get("minimum_ms"), f"{field}.minimum_ms", positive=True)
    maximum = number(row.get("maximum_ms"), f"{field}.maximum_ms", positive=True)
    close(minimum, min(measured), f"{field}.minimum_ms", rel_tol=1.0e-12)
    close(maximum, max(measured), f"{field}.maximum_ms", rel_tol=1.0e-12)
    if minimum > median or median > maximum:
        fail(f"{field} timing order is invalid")
    if integer(row.get("sample_count"), f"{field}.sample_count") != TIMING_SAMPLES:
        fail(f"{field}.sample_count is not the declared sample count")
    if integer(row.get("actions_per_sample"), f"{field}.actions_per_sample", positive=True) != ACTIONS_PER_SAMPLE:
        fail(f"{field}.actions_per_sample is not the declared repeat count")
    if integer(row.get("work_units_per_action"), f"{field}.work_units_per_action", positive=True) != work_units:
        fail(f"{field}.work_units_per_action does not match the tier shape")
    throughput = number(row.get("work_units_per_second_at_median"), f"{field}.work_units_per_second_at_median", positive=True)
    close(throughput, work_units * ACTIONS_PER_SAMPLE / (median / 1000.0), f"{field}.work_units_per_second_at_median")
    finite_number(row.get("checksum"), f"{field}.checksum")
    return {"median_ms": median, "mad_ms": mad, "work_units_per_second_at_median": throughput}


def check_stats(stats: Any, field: str) -> dict[str, int]:
    if not isinstance(stats, dict):
        fail(f"{field} runtime counters are missing")
    names = (
        "kernel_launches",
        "library_operations",
        "synchronization_calls",
        "allocation_operations",
        "staging_allocation_operations",
        "h2d_operations",
        "d2h_operations",
        "d2d_operations",
        "live_allocation_bytes",
        "allocation_high_water_bytes",
        "live_staging_bytes",
        "staging_high_water_bytes",
    )
    return {name: integer(stats.get(name), f"{field}.{name}") for name in names}


def check(path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact must be an object")
    if artifact.get("schema") != SCHEMA or artifact.get("version") != 1:
        fail("schema/version is invalid")
    if artifact.get("status") != "passed" or artifact.get("case_id") != CASE_ID:
        fail("status or case identity is invalid")

    claims = artifact.get("claims")
    if not isinstance(claims, dict):
        fail("claims are missing")
    if claims.get("native_cuda") is not True:
        fail("native CUDA dispatch is not asserted")
    if claims.get("cpu_comparator") != "measured_independent_f64_oracle":
        fail("CPU comparator claim is not the measured oracle contract")
    if claims.get("vendor_comparator") != "not_available_for_chart_driven_quadrature":
        fail("vendor comparator must remain explicitly unavailable")
    if claims.get("roofline") != "not_claimed_no_profiler_or_instruction_count":
        fail("roofline status must remain explicitly unclaimed")
    if claims.get("energy") != "not_claimed_no_energy_counter":
        fail("energy status must remain explicitly unclaimed")

    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance is missing")
    for field in (
        "device",
        "driver",
        "toolkit",
        "compiler",
        "topology",
        "peak_flops",
        "memory_bandwidth",
        "os",
        "target_features",
        "runtime_dispatch",
        "deterministic_mode",
    ):
        text(hardware.get(field), f"hardware.{field}")
    if hardware["os"] != "linux-x86_64" or hardware["target_features"] != "cuda":
        fail("artifact is not native Linux CUDA evidence")
    if hardware["runtime_dispatch"] != "cuda-native":
        fail("artifact does not prove native CUDA dispatch")
    if "blackwell" not in hardware["device"].lower() and "sm120" not in hardware["device"].lower():
        fail("artifact device is not the declared SM120/Blackwell target")
    provenance_number(hardware["peak_flops"], "hardware.peak_flops")
    provenance_number(hardware["memory_bandwidth"], "hardware.memory_bandwidth")

    workload = artifact.get("workload")
    if not isinstance(workload, dict):
        fail("workload is missing")
    if workload.get("production_adapter") != "CudaBemQuadratureOperatorC64" or workload.get("scalar") != "Complex64":
        fail("production adapter/scalar is invalid")
    if workload.get("channels") != ["G", "H", "H_transpose", "E"]:
        fail("all four BEM channels are not declared")
    if integer(workload.get("warmup_actions"), "workload.warmup_actions", positive=True) != 3:
        fail("warmup count is not the checked contract")
    if integer(workload.get("timing_samples"), "workload.timing_samples", positive=True) != TIMING_SAMPLES:
        fail("timing sample count is not the checked contract")
    if integer(workload.get("actions_per_sample"), "workload.actions_per_sample", positive=True) != ACTIONS_PER_SAMPLE:
        fail("action repeat count is not the checked contract")

    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != len(TIERS):
        fail("exactly three size-tier rows are required")
    by_id: dict[str, dict[str, Any]] = {}
    summaries = []
    for row in rows:
        if not isinstance(row, dict):
            fail("tier row must be an object")
        tier_id = text(row.get("id"), "rows[].id")
        if tier_id in by_id or tier_id not in TIERS:
            fail(f"unexpected or duplicate tier {tier_id!r}")
        by_id[tier_id] = row
        pair_count, samples_per_pair = TIERS[tier_id]
        if row.get("status") != "passed":
            fail(f"{tier_id} did not pass")
        if integer(row.get("pair_count"), f"{tier_id}.pair_count", positive=True) != pair_count:
            fail(f"{tier_id} pair count is not the checked odd tier")
        if integer(row.get("samples_per_pair"), f"{tier_id}.samples_per_pair", positive=True) != samples_per_pair:
            fail(f"{tier_id} samples per pair is invalid")
        if integer(row.get("sample_count"), f"{tier_id}.sample_count", positive=True) != pair_count * samples_per_pair:
            fail(f"{tier_id} flattened sample count is invalid")
        if row.get("odd_pair_tail") is not True:
            fail(f"{tier_id} does not prove its odd tail")
        correctness = row.get("correctness")
        if not isinstance(correctness, dict) or correctness.get("correctness_before_timing") is not True:
            fail(f"{tier_id} correctness gate did not precede timing")
        tolerance = number(correctness.get("tolerance"), f"{tier_id}.correctness.tolerance", positive=True)
        if tolerance != TOLERANCE:
            fail(f"{tier_id} correctness tolerance differs from the checked contract")
        pre_error = number(correctness.get("pre_timing_relative_error"), f"{tier_id}.correctness.pre_timing_relative_error")
        post_error = number(correctness.get("post_timing_relative_error"), f"{tier_id}.correctness.post_timing_relative_error")
        if pre_error > tolerance or post_error > tolerance:
            fail(f"{tier_id} correctness error exceeds tolerance")
        cpu = row.get("cpu")
        cuda = row.get("cuda")
        if not isinstance(cpu, dict) or not isinstance(cuda, dict):
            fail(f"{tier_id} CPU/CUDA timing rows are missing")
        cpu_summary = check_timing(cpu, f"{tier_id}.cpu", pair_count * samples_per_pair)
        cuda_summary = check_timing(cuda, f"{tier_id}.cuda", pair_count * samples_per_pair)
        if abs(float(cpu["checksum"]) - float(cuda["checksum"])) > max(1.0e-9, abs(float(cpu["checksum"])) * 1.0e-9):
            fail(f"{tier_id} CPU/CUDA checksums disagree")

        runtime = row.get("runtime")
        if not isinstance(runtime, dict):
            fail(f"{tier_id} runtime evidence is missing")
        setup = check_stats(runtime.get("setup"), f"{tier_id}.runtime.setup")
        timed = check_stats(runtime.get("timed_resident"), f"{tier_id}.runtime.timed_resident")
        publication = check_stats(runtime.get("publication"), f"{tier_id}.runtime.publication")
        teardown = check_stats(runtime.get("teardown"), f"{tier_id}.runtime.teardown")
        expected_actions = TIMING_SAMPLES * ACTIONS_PER_SAMPLE
        if timed["kernel_launches"] != expected_actions or timed["synchronization_calls"] != expected_actions:
            fail(f"{tier_id} timed launch/synchronization counts do not match repeated samples")
        if timed["allocation_operations"] != 0 or timed["h2d_operations"] != 0 or timed["d2h_operations"] != 0:
            fail(f"{tier_id} timed resident path reports allocation or host staging")
        if timed["d2d_operations"] != expected_actions:
            fail(f"{tier_id} timed device publication count does not match actions")
        if timed["live_allocation_bytes"] <= 0 or timed["allocation_high_water_bytes"] <= 0:
            fail(f"{tier_id} timed allocation ledger is missing")
        if publication["d2h_operations"] != 1:
            fail(f"{tier_id} has no single explicit output publication")
        if setup["h2d_operations"] < 2 or setup["allocation_operations"] < 3:
            fail(f"{tier_id} setup does not prove resident input allocation/upload")
        if teardown["live_allocation_bytes"] != 0 or teardown["live_staging_bytes"] != 0:
            fail(f"{tier_id} teardown stranded device or staging bytes")

        failure = row.get("failure_recovery")
        if not isinstance(failure, dict):
            fail(f"{tier_id} failure recovery evidence is missing")
        if failure.get("failure_typed") is not True or failure.get("output_preserved") is not True or failure.get("retry_succeeded") is not True:
            fail(f"{tier_id} typed failure/no-publication/retry contract failed")
        retry_error = number(failure.get("retry_relative_error"), f"{tier_id}.failure_recovery.retry_relative_error")
        if retry_error > tolerance:
            fail(f"{tier_id} retry error exceeds tolerance")
        check_stats(failure.get("failure_runtime"), f"{tier_id}.failure_recovery.failure_runtime")
        check_stats(failure.get("retry_runtime"), f"{tier_id}.failure_recovery.retry_runtime")
        if row.get("malformed_input_rejected") is not True:
            fail(f"{tier_id} malformed input was not rejected")
        summaries.append({
            "id": tier_id,
            "pair_count": pair_count,
            "samples_per_pair": samples_per_pair,
            "cpu_median_ms": cpu_summary["median_ms"],
            "cpu_mad_ms": cpu_summary["mad_ms"],
            "cuda_median_ms": cuda_summary["median_ms"],
            "cuda_mad_ms": cuda_summary["mad_ms"],
            "pre_timing_relative_error": pre_error,
            "post_timing_relative_error": post_error,
        })
    if set(by_id) != set(TIERS):
        fail("tier set is incomplete")

    boundaries = artifact.get("unsupported_boundaries")
    if not isinstance(boundaries, list) or not all(isinstance(item, str) and item.strip() for item in boundaries):
        fail("unsupported boundaries must be explicit text")
    required_boundaries = ("Nsight", "energy", "vendor", "multi-GPU")
    for marker in required_boundaries:
        if not any(marker.lower() in item.lower() for item in boundaries):
            fail(f"unsupported boundary {marker!r} is not recorded")
    return {
        "status": "passed",
        "schema": SCHEMA,
        "case_id": CASE_ID,
        "tiers": summaries,
        "vendor_comparator": claims["vendor_comparator"],
        "roofline": claims["roofline"],
        "energy": claims["energy"],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} ARTIFACT.json", file=sys.stderr)
        return 2
    try:
        result = check(Path(sys.argv[1]))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
