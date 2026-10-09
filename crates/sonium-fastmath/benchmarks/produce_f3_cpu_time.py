#!/usr/bin/env python3
"""Measure real child-process CPU time for the native FastMath benchmark.

The benchmark already reports kernel wall-time distributions.  This producer
adds Unix process accounting around two AVX2 and two forced-scalar child runs,
without turning an unavailable RAPL counter into an energy estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path


SCHEMA = "sonium.fastmath.f3-cpu-time-evidence.v1"
MODES = {
    "avx2": {
        "SONIUM_FASTMATH_REQUIRE_AVX2": "1",
        "SONIUM_FASTMATH_FORCE_AVX2": "1",
        "SONIUM_FASTMATH_FORCE_SCALAR": None,
    },
    "scalar": {
        "SONIUM_FASTMATH_REQUIRE_AVX2": None,
        "SONIUM_FASTMATH_FORCE_AVX2": None,
        "SONIUM_FASTMATH_FORCE_SCALAR": "1",
    },
}
REGIMES = ["reduced", "coherent_ramp", "random_full_range", "reduction_boundaries", "payne_hanek"]
REGIME_FIELDS = (
    "input_checksum",
    "slow_path_rate",
    "fast_checksum",
    "scalar_checksum",
    "libm_checksum",
    "fast_ns_per_sample",
    "scalar_ns_per_sample",
    "libm_ns_per_sample",
    "fast_p95_batch_ns",
    "fast_p99_batch_ns",
    "scalar_p95_batch_ns",
    "scalar_p99_batch_ns",
    "libm_p95_batch_ns",
    "libm_p99_batch_ns",
    "fast_energy_joules",
    "scalar_energy_joules",
    "libm_energy_joules",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def child_cpu_ns(before: resource.struct_rusage, after: resource.struct_rusage) -> tuple[int, int, int]:
    user_ns = int(round((after.ru_utime - before.ru_utime) * 1.0e9))
    system_ns = int(round((after.ru_stime - before.ru_stime) * 1.0e9))
    return user_ns, system_ns, user_ns + system_ns


def clean_environment(mode: str) -> dict[str, str]:
    environment = dict(os.environ)
    for name, value in MODES[mode].items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    environment["RAYON_NUM_THREADS"] = "1"
    return environment


def run_once(binary: Path, mode: str, repeat: int) -> dict[str, object]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start_ns = time.monotonic_ns()
    completed = subprocess.run(
        [str(binary)],
        env=clean_environment(mode),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    wall_ns = time.monotonic_ns() - wall_start_ns
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} repeat {repeat} failed with {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')}"
        )
    payload = json.loads(completed.stdout)
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"{mode} repeat {repeat}: unexpected schema")
    if payload.get("architecture") != "x86_64":
        raise RuntimeError(f"{mode} repeat {repeat}: not native x86_64")
    expected_dispatch = "avx2" if mode == "avx2" else "scalar"
    if payload.get("dispatch") != expected_dispatch:
        raise RuntimeError(
            f"{mode} repeat {repeat}: dispatch={payload.get('dispatch')!r}, "
            f"expected {expected_dispatch!r}"
        )
    if payload.get("correctness_gate") != "passed":
        raise RuntimeError(f"{mode} repeat {repeat}: correctness gate failed")
    if payload.get("samples_per_regime") != 4096 or payload.get("batches") != 32:
        raise RuntimeError(f"{mode} repeat {repeat}: benchmark dimensions changed")
    regimes = payload.get("regimes")
    if not isinstance(regimes, list) or [row.get("name") for row in regimes] != REGIMES:
        raise RuntimeError(f"{mode} repeat {repeat}: regime set changed")
    for row in regimes:
        for field in REGIME_FIELDS:
            value = row.get(field)
            if field.endswith("energy_joules"):
                continue
            if not finite(value) or (field.endswith("ns_per_sample") and value <= 0):
                raise RuntimeError(f"{mode} repeat {repeat}: invalid regime field {field}")
    user_ns, system_ns, process_cpu_ns = child_cpu_ns(before, after)
    if process_cpu_ns <= 0 or wall_ns <= 0:
        raise RuntimeError(f"{mode} repeat {repeat}: process counter did not advance")
    return {
        "mode": mode,
        "repeat": repeat,
        "dispatch": payload["dispatch"],
        "correctness_gate": payload["correctness_gate"],
        "architecture": payload["architecture"],
        "cpu": payload.get("cpu"),
        "samples_per_regime": payload["samples_per_regime"],
        "batches": payload["batches"],
        "cpu_user_ns": user_ns,
        "cpu_system_ns": system_ns,
        "process_cpu_ns": process_cpu_ns,
        "wall_ns": wall_ns,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "energy_available": payload.get("energy_available") is True,
        "energy_source": payload.get("energy_source"),
        "regimes": [{field: row.get(field) for field in ("name", *REGIME_FIELDS)} for row in regimes],
    }


def mean(values: list[int]) -> float:
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark binary is missing: {binary}")
    if os.uname().sysname != "Linux" or os.uname().machine != "x86_64":
        raise SystemExit("F3 CPU-time evidence requires native Linux x86_64")

    runs = [run_once(binary, mode, repeat) for mode in MODES for repeat in range(2)]
    energy_available = {run["energy_available"] for run in runs}
    if len(energy_available) != 1:
        raise RuntimeError("energy counter availability changed across runs")
    energy_measured = next(iter(energy_available))
    mode_cpu_means = {
        mode: mean([run["process_cpu_ns"] for run in runs if run["mode"] == mode])
        for mode in MODES
    }
    mode_wall_means = {
        mode: mean([run["wall_ns"] for run in runs if run["mode"] == mode])
        for mode in MODES
    }
    repeat_spreads = []
    for mode in MODES:
        values = [run["process_cpu_ns"] for run in runs if run["mode"] == mode]
        repeat_spreads.append(abs(values[1] - values[0]) / max(values))
    canonical_input = (
        "binary=fastmath_benchmark;native_linux_x86_64;"
        "modes=avx2,scalar;repeats=2;RAYON_NUM_THREADS=1;"
        "counter=resource.getrusage(RUSAGE_CHILDREN);energy=reported-only"
    )
    canonical_output = json.dumps(runs, separators=(",", ":"), sort_keys=True)
    first = runs[0]
    if not isinstance(first.get("cpu"), str) or not first["cpu"]:
        raise RuntimeError("benchmark did not report a host CPU description")
    artifact = {
        "schema": SCHEMA,
        "status": "pass",
        "scope": "native Linux x86_64 process CPU-time accounting; AVX-512/Arm64/Windows waived",
        "claim_class": "native FastMath process CPU-time and timing evidence",
        "reference": "resource.getrusage(RUSAGE_CHILDREN) around the production fastmath_benchmark binary; wall-time and per-regime timings remain separately reported",
        "canonical_input": canonical_input,
        "canonical_input_sha256": sha256_bytes(canonical_input.encode()),
        "host": {
            "architecture": first["architecture"],
            "cpu": first["cpu"],
        },
        "counter_source": "resource.getrusage(RUSAGE_CHILDREN): ru_utime + ru_stime",
        "energy_status": "measured" if energy_measured else "unavailable",
        "energy_claim": "measured from benchmark-reported counter" if energy_measured else "no energy counter was readable; no energy claim",
        "modes": list(MODES),
        "repeats_per_mode": 2,
        "runs": runs,
        "summary": {
            "avx2_mean_process_cpu_ns": mode_cpu_means["avx2"],
            "scalar_mean_process_cpu_ns": mode_cpu_means["scalar"],
            "avx2_to_scalar_process_cpu_ratio": mode_cpu_means["avx2"] / mode_cpu_means["scalar"],
            "avx2_mean_wall_ns": mode_wall_means["avx2"],
            "scalar_mean_wall_ns": mode_wall_means["scalar"],
            "maximum_repeat_cpu_spread_fraction": max(repeat_spreads),
        },
        "canonical_output": canonical_output,
        "output_sha256": sha256_bytes(canonical_output.encode()),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "runs": len(runs),
                "energy_status": artifact["energy_status"],
                "avx2_to_scalar_process_cpu_ratio": artifact["summary"]["avx2_to_scalar_process_cpu_ratio"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
