#!/usr/bin/env python3
"""Fail-closed comparison of a native FastMath run with its pinned baseline.

The benchmark records aggregate and tail timings rather than raw Criterion
samples.  This gate therefore checks deterministic corpus identity and uses
the recorded p95/p99 values as a conservative statistical screen.  It is a
release signal, not a claim that one noisy local run is a replacement for a
maintainer-approved benchmark host.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return value


def clock_hz(document: dict) -> float | None:
    """Parse the governor-visible clock snapshot used for wall-time scaling.

    The benchmark is wall-clock based, while schedutil/boost may select a
    different P-state for each run.  A numeric clock snapshot lets the gate
    compare algorithmic throughput at a common frequency without pretending
    that a 2.2 GHz capped run is directly comparable to a 4.0 GHz capture.
    """
    raw = document.get("clock_state")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    # Only the ratio is used, so the producer's common frequency unit cancels.
    return value


def external_reference_names(document: dict) -> list[str]:
    """Names of independently configured external references, if any.

    The pinned baseline records an empty list: no vendor vector-math
    library is wired into the producer, so only the in-crate scalar path
    and platform libm are compared. Informational only.
    """
    refs = document.get("external_references", [])
    names: list[str] = []
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict) and isinstance(ref.get("name"), str):
                names.append(ref["name"])
            elif isinstance(ref, str):
                names.append(ref)
    return names


def provenance_snapshot(document: dict) -> dict:
    return {
        "energy_available": document.get("energy_available"),
        "runtime_features": document.get("runtime_features"),
        "external_references": external_reference_names(document),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--max-slowdown", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_slowdown < 0:
        raise SystemExit("--max-slowdown must be non-negative")

    baseline = load(args.baseline)
    current = load(args.current)
    failures: list[str] = []
    if baseline.get("schema_version") != current.get("schema_version"):
        failures.append("schema_version mismatch")
    if current.get("architecture") != "x86_64":
        failures.append("current artifact is not x86_64")
    if current.get("dispatch") != "avx2":
        failures.append("current artifact did not select AVX2")
    if current.get("correctness_gate") != "passed":
        failures.append("current correctness gate did not pass")
    for field in ("architecture", "cpu"):
        if baseline.get(field) != current.get(field):
            failures.append(f"pinned {field} mismatch")

    baseline_regimes = {row.get("name"): row for row in baseline.get("regimes", [])}
    current_regimes = {row.get("name"): row for row in current.get("regimes", [])}
    if set(baseline_regimes) != set(current_regimes):
        failures.append("regime set mismatch")

    baseline_clock = clock_hz(baseline)
    current_clock = clock_hz(current)
    failures = list(failures)
    if baseline_clock is None or current_clock is None:
        failures.append("clock_state missing or invalid; wall-time baseline is not comparable")
        clock_scale = 1.0
        clock_status = "unavailable"
    else:
        # At a lower clock, a fixed amount of work takes proportionally more
        # wall time.  Scale the baseline to the current clock before applying
        # the unchanged slowdown threshold.  Raw ratios remain in the report.
        clock_scale = baseline_clock / current_clock
        clock_status = "frequency-normalized"

    comparisons = []
    limit = 1.0 + args.max_slowdown
    for name in sorted(set(baseline_regimes) & set(current_regimes)):
        old = baseline_regimes[name]
        new = current_regimes[name]
        if old.get("input_checksum") != new.get("input_checksum"):
            failures.append(f"{name}: input checksum mismatch")
        old_ns = float(old.get("fast_ns_per_sample", 0.0))
        new_ns = float(new.get("fast_ns_per_sample", 0.0))
        old_p95 = float(old.get("fast_p95_batch_ns", 0.0))
        new_p95 = float(new.get("fast_p95_batch_ns", 0.0))
        old_p99 = float(old.get("fast_p99_batch_ns", 0.0))
        new_p99 = float(new.get("fast_p99_batch_ns", 0.0))
        if min(old_ns, old_p95, old_p99) <= 0:
            failures.append(f"{name}: baseline timing is not positive")
            continue
        raw_ratios = {
            "ns_per_sample": new_ns / old_ns,
            "p95_batch_ns": new_p95 / old_p95,
            "p99_batch_ns": new_p99 / old_p99,
        }
        ratios = {name: ratio / clock_scale for name, ratio in raw_ratios.items()}
        if any(ratio > limit for ratio in ratios.values()):
            failures.append(f"{name}: timing regression exceeds {args.max_slowdown:.0%}")
        comparisons.append({"name": name, "raw_ratios": raw_ratios, "ratios": ratios})

    result = {
        "schema_version": 1,
        "status": "fail" if failures else "pass",
        "method": "paired-regime timing ratio using aggregate, p95, and p99",
        "max_slowdown_fraction": args.max_slowdown,
        "clock_normalization": {
            "status": clock_status,
            "method": "baseline wall time scaled by baseline/current recorded clock state",
            "baseline_clock_state": baseline_clock,
            "current_clock_state": current_clock,
            "baseline_to_current_scale": clock_scale,
        },
        "baseline": str(args.baseline),
        "current": str(args.current),
        "comparisons": comparisons,
        "provenance": {
            "baseline": provenance_snapshot(baseline),
            "current": provenance_snapshot(current),
            "note": "informational only; missing energy or external references never fail the gate",
        },
        "failures": failures,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
