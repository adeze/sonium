#!/usr/bin/env python3
"""Fail-closed summarizer for compute CUDA workload artifacts.

Both the BEM and FEM QA producers write a raw JSON object.  This utility only
summarizes a passed, native CUDA run whose numerical thresholds, runtime
counters, hardware fingerprint, and roofline field are present.  An
unavailable required device is reported as a failure so it cannot be promoted
to a release pass by a dashboard or CI wrapper.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


SUPPORTED_SCHEMAS = {
    "sonium-compute-cuda-bem-evidence/v1",
    "sonium-compute-cuda-fem-adjoint/v1",
}
REQUIRED_HARDWARE = (
    "device",
    "driver",
    "toolkit",
    "compiler",
    "topology",
    "peak_flops",
    "memory_bandwidth",
)
REQUIRED_RUNTIME = (
    "kernel_launches",
    "synchronization_calls",
    "allocation_high_water_bytes",
    "live_allocation_bytes",
)


def load(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("artifact must contain a JSON object")
    return value


def nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} is not finite and non-negative")
    return number


def nonnegative_measurement(value: object, field: str) -> float:
    """Parse a numeric JSON value or the QA environment's numeric string."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise ValueError(f"{field} is not numeric") from error
    return nonnegative(value, field)


def validate_thresholds(artifact: dict, metrics: dict) -> None:
    thresholds = artifact.get("thresholds")
    directions = artifact.get("threshold_directions", {})
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("thresholds are missing")
    if not isinstance(directions, dict):
        raise ValueError("threshold_directions is not an object")
    for field, threshold in thresholds.items():
        if field not in metrics:
            raise ValueError(f"threshold has no metric: {field}")
        actual = nonnegative(metrics[field], f"metrics.{field}")
        bound = nonnegative(threshold, f"thresholds.{field}")
        direction = directions.get(field, "maximum")
        if direction == "maximum" and actual > bound:
            raise ValueError(f"{field} exceeds maximum threshold")
        if direction == "minimum" and actual < bound:
            raise ValueError(f"{field} is below minimum threshold")
        if direction not in ("maximum", "minimum"):
            raise ValueError(f"unsupported threshold direction for {field}")


def runtime_counters(artifact: dict) -> dict:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime evidence is missing")
    # FEM stores counters directly; BEM stores the transfer subgroup.
    counters = dict(runtime)
    transfers = runtime.get("transfers")
    if isinstance(transfers, dict):
        counters.update(transfers)
    for field in REQUIRED_RUNTIME:
        nonnegative(counters.get(field), f"runtime.{field}")
    library_operations = nonnegative(
        counters.get("library_operations", 0), "runtime.library_operations"
    )
    # cuBLAS/cuSPARSE launches are represented by library calls in the
    # instrumentation rather than by custom-kernel launches.  Either class is
    # valid device work, but a passing artifact must contain at least one.
    if counters["kernel_launches"] + library_operations <= 0:
        raise ValueError("passing artifact has no CUDA device operations")
    counters["library_operations"] = library_operations
    return counters


def validate(artifact: dict) -> dict:
    if artifact.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("unsupported or missing compute CUDA schema")
    status = artifact.get("status")
    if status in ("unavailable", "failed"):
        raise ValueError(f"required CUDA workload did not run: {status}")
    if status != "passed":
        raise ValueError(f"invalid artifact status {status!r}")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics are missing")
    validate_thresholds(artifact, metrics)
    throughput = nonnegative_measurement(
        metrics.get("throughput_flops_per_second"),
        "metrics.throughput_flops_per_second",
    )
    if throughput <= 0:
        raise ValueError("metrics.throughput_flops_per_second must be positive")
    operation_count = nonnegative(metrics.get("operation_count", artifact.get("operation_count")), "operation_count")
    if operation_count <= 0:
        raise ValueError("operation_count must be positive")
    elapsed = nonnegative(artifact.get("workload_elapsed_ms"), "workload_elapsed_ms")
    if elapsed <= 0:
        raise ValueError("workload_elapsed_ms must be positive")
    runtime = runtime_counters(artifact)
    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("hardware fingerprint is missing")
    for field in REQUIRED_HARDWARE:
        value = hardware.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"hardware.{field} is required")
    for field in ("peak_flops", "memory_bandwidth"):
        if nonnegative_measurement(hardware[field], f"hardware.{field}") <= 0:
            raise ValueError(f"hardware.{field} must be positive")
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("environment evidence is missing")
    if "performance.roofline_fraction" not in environment:
        raise ValueError("performance.roofline_fraction is missing")
    roofline = environment["performance.roofline_fraction"]
    # A passed native-CUDA artifact must carry a measured roofline fraction.
    # Keep the explicit field requirement above so unavailable/failed runs can
    # still be retained as diagnostic artifacts, but never let the string
    # "unavailable" promote a purported pass.
    roofline = nonnegative_measurement(roofline, "performance.roofline_fraction")
    if roofline > 1.0:
        raise ValueError("performance.roofline_fraction exceeds one")
    return {
        "schema": artifact["schema"],
        "case_id": artifact.get("case_id"),
        "operation_count": operation_count,
        "workload_elapsed_ms": elapsed,
        "throughput_flops_per_second": throughput,
        "kernel_launches": runtime["kernel_launches"],
        "library_operations": runtime["library_operations"],
        "roofline_fraction": roofline,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_cuda_evidence.py ARTIFACT.json", file=sys.stderr)
        return 2
    try:
        summary = validate(load(sys.argv[1]))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"evidence contract failure: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
