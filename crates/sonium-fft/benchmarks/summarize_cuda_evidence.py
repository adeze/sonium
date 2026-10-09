#!/usr/bin/env python3
"""Fail-closed summarizer for sonium-fft CUDA benchmark artifacts.

The input is the raw JSON printed by ``fft_cuda_benchmark``.  Unsupported and
unavailable cells remain visible, while a benchmark summary is produced only
when every measured row passed its numerical gate and contains an empirical
95% interval.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REQUIRED_ROW_FIELDS = (
    "case",
    "dtype",
    "n",
    "batch",
    "status",
    "execution",
    "correctness_passed",
    "throughput_gate_passed",
    "median_ns",
    "ci95_low_ns",
    "ci95_high_ns",
    "ci95_method",
    "bytes_per_second",
    "transforms_per_second",
)


def load(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("artifact must contain a JSON object")
    return value


def finite_number(value: object, field: str, case: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{case}: {field} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{case}: {field} is not a finite non-negative number")
    return number


def validate(artifact: dict) -> list[dict]:
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported or missing schema_version")
    if not isinstance(artifact.get("hardware"), dict):
        raise ValueError("hardware fingerprint is missing")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("artifact contains no rows")
    measured = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row is not an object")
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row is missing {', '.join(missing)}")
        case = str(row["case"])
        if case in seen:
            raise ValueError(f"duplicate case {case}")
        seen.add(case)
        if row["status"] in ("unavailable", "unsupported"):
            if row["median_ns"] is not None:
                raise ValueError(f"{case}: unavailable/unsupported row has timing")
            if row["throughput_gate_passed"] is not False:
                raise ValueError(f"{case}: unavailable/unsupported row passed throughput gate")
            continue
        if row["status"] != "passed" or row["execution"] != "vendor-native":
            raise ValueError(f"{case}: measured row did not pass vendor-native gate")
        if row["ci95_method"] != "empirical_order_stat_95":
            raise ValueError(f"{case}: unsupported confidence-interval method")
        if row["correctness_passed"] is not True:
            raise ValueError(f"{case}: numerical gate failed")
        if row["throughput_gate_passed"] is not True:
            raise ValueError(f"{case}: throughput gate failed")
        if row.get("cache_reuse_gate_passed") is not True:
            raise ValueError(f"{case}: cache/workspace reuse gate failed")
        values = {
            field: finite_number(row[field], field, case)
            for field in (
                "median_ns",
                "ci95_low_ns",
                "ci95_high_ns",
                "bytes_per_second",
                "transforms_per_second",
            )
        }
        samples = row.get("samples_ns")
        if not isinstance(samples, list) or not samples or any(
            not isinstance(sample, int) or isinstance(sample, bool) or sample <= 0
            for sample in samples
        ):
            raise ValueError(f"{case}: samples_ns is not a non-empty positive integer list")
        if not values["ci95_low_ns"] <= values["median_ns"] <= values["ci95_high_ns"]:
            raise ValueError(f"{case}: confidence interval does not bracket median")
        measured.append(row)
    if artifact.get("status") not in ("passed", "unavailable"):
        raise ValueError(f"invalid artifact status {artifact.get('status')!r}")
    if artifact.get("status") != "unavailable":
        cancellation = artifact.get("cancellation")
        if not isinstance(cancellation, dict) or cancellation.get("status") != "passed":
            raise ValueError("cancellation cleanup/reuse evidence is missing or failed")
        for field in (
            "regular_c32",
            "regular_c64",
            "strided_c64",
            "pre_cancel_rejected",
            "post_submit_cancelled",
            "completion_fence_drained",
            "plan_reused",
        ):
            if cancellation.get(field) is not True:
                raise ValueError(f"cancellation.{field} must be true")
        hardware = artifact["hardware"]
        for field in ("device", "driver", "sm_clock_mhz", "memory_clock_mhz"):
            if not isinstance(hardware.get(field), str) or not hardware[field]:
                raise ValueError(f"hardware.{field} is required for a passing artifact")
        for field in ("compiler", "cuda_toolkit"):
            if not isinstance(artifact.get(field), str) or not artifact[field]:
                raise ValueError(f"{field} is required for a passing artifact")
    return measured


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_cuda_evidence.py ARTIFACT.json", file=sys.stderr)
        return 2
    try:
        artifact = load(sys.argv[1])
        measured = validate(artifact)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"evidence contract error: {error}", file=sys.stderr)
        return 2
    if artifact.get("status") == "unavailable":
        print(json.dumps({"status": "unavailable", "measured_rows": 0}, sort_keys=True))
        return 0
    if not measured:
        print("evidence contract error: no measured vendor-native rows", file=sys.stderr)
        return 2
    medians = [float(row["median_ns"]) for row in measured]
    summary = {
        "status": "passed",
        "schema_version": artifact["schema_version"],
        "measured_rows": len(measured),
        "median_ns_min": min(medians),
        "median_ns_max": max(medians),
        "cases": [row["case"] for row in measured],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
