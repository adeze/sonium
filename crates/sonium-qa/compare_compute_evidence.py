#!/usr/bin/env python3
"""Fail-closed comparison for sonium-compute performance evidence.

Timing is useful release evidence only when the numerical preflight, benchmark
matrix, hardware identity, and empirical confidence intervals are all valid.
Malformed or incomparable bundles are contract errors, never silent skips.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MAX_MEDIAN_SLOWDOWN = 0.10
ROW_FIELDS = ("implementation", "blas_level", "operation", "regime", "dtype", "shape")
HARDWARE_FIELDS = ("arch", "model", "rayon_threads")


def load(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def row_key(row: dict) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in ROW_FIELDS)


def validate_row(row: object, source: str) -> tuple[str, ...]:
    if not isinstance(row, dict):
        raise ValueError(f"{source} contains a non-object performance row")
    required = (*ROW_FIELDS, "correctness_passed", "median_ns", "ci95_low_ns", "ci95_high_ns")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{source} row is missing fields: {', '.join(missing)}")
    key = row_key(row)
    if row["correctness_passed"] is not True:
        raise ValueError(f"{source} row {key!r} failed its numerical correctness preflight")
    for name in ("median_ns", "ci95_low_ns", "ci95_high_ns"):
        value = row[name]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{source} row {key!r} has invalid {name}={value!r}")
    if row["ci95_low_ns"] > row["median_ns"] or row["median_ns"] > row["ci95_high_ns"]:
        raise ValueError(f"{source} row {key!r} does not bracket its median")
    return key


def index_rows(snapshot: dict, source: str) -> dict[tuple[str, ...], dict]:
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source} contains no performance rows")
    indexed: dict[tuple[str, ...], dict] = {}
    for row in rows:
        key = validate_row(row, source)
        if key in indexed:
            raise ValueError(f"{source} contains duplicate row {key!r}")
        indexed[key] = row
    return indexed


def validate_snapshot(snapshot: dict, source: str) -> dict[tuple[str, ...], dict]:
    schema = snapshot.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"{source} has no non-empty schema")
    hardware = snapshot.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError(f"{source} has no hardware fingerprint")
    missing = [field for field in HARDWARE_FIELDS if field not in hardware]
    if missing:
        raise ValueError(f"{source} hardware is missing fields: {', '.join(missing)}")
    return index_rows(snapshot, source)


def compare(baseline: dict, current: dict):
    if baseline.get("schema") != current.get("schema"):
        raise ValueError("evidence schema mismatch")
    baseline_rows = validate_snapshot(baseline, "baseline")
    current_rows = validate_snapshot(current, "current")
    for field in HARDWARE_FIELDS:
        if baseline["hardware"].get(field) != current["hardware"].get(field):
            raise ValueError(f"hardware/thread mismatch in {field}")
    baseline_keys = set(baseline_rows)
    current_keys = set(current_rows)
    if baseline_keys != current_keys:
        missing = sorted(baseline_keys - current_keys)
        extra = sorted(current_keys - baseline_keys)
        raise ValueError(f"benchmark matrix mismatch: missing={missing!r}, extra={extra!r}")

    failures = []
    inconclusive = []
    for key in sorted(baseline_keys):
        old = baseline_rows[key]
        new = current_rows[key]
        old_median = float(old["median_ns"])
        new_median = float(new["median_ns"])
        if old_median == 0:
            slowdown = math.inf if new_median > 0 else 0.0
        else:
            slowdown = new_median / old_median - 1.0
        overlap = (new["ci95_low_ns"] <= old["ci95_high_ns"]
                   and old["ci95_low_ns"] <= new["ci95_high_ns"])
        if slowdown > MAX_MEDIAN_SLOWDOWN and not overlap:
            failures.append((key, old_median, new_median))
        elif slowdown > MAX_MEDIAN_SLOWDOWN:
            inconclusive.append(key)
    return failures, inconclusive


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_compute_evidence.py BASELINE.json CURRENT.json", file=sys.stderr)
        return 2
    try:
        baseline, current = map(load, sys.argv[1:])
        failures, inconclusive = compare(baseline, current)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"evidence contract error: {error}", file=sys.stderr)
        return 2
    for key, old, new in failures:
        print(f"regression {'/'.join(key)}: {old:.0f}ns -> {new:.0f}ns", file=sys.stderr)
    for key in inconclusive:
        print(f"inconclusive {'/'.join(key)}: intervals overlap", file=sys.stderr)
    if failures:
        return 1
    print("SUCCESS: no statistically separated regression above 10%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
