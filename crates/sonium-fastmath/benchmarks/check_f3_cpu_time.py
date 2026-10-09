#!/usr/bin/env python3
"""Fail-closed checker for the bounded native FastMath CPU-time evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMA = "sonium.fastmath.f3-cpu-time-evidence.v1"
MODES = ("avx2", "scalar")
REGIMES = ("reduced", "coherent_ramp", "random_full_range", "reduction_boundaries", "payne_hanek")
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
RUN_FIELDS = {
    "mode",
    "repeat",
    "dispatch",
    "correctness_gate",
    "architecture",
    "cpu",
    "samples_per_regime",
    "batches",
    "cpu_user_ns",
    "cpu_system_ns",
    "process_cpu_ns",
    "wall_ns",
    "stdout_sha256",
    "energy_available",
    "energy_source",
    "regimes",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close_enough(actual: object, expected: float) -> bool:
    return finite(actual) and math.isclose(float(actual), expected, rel_tol=1e-12)


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return value


def check(artifact_path: Path) -> list[str]:
    artifact = load(artifact_path)
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(artifact.get("status") == "pass", "status is not pass")
    require(
        artifact.get("scope") == "native Linux x86_64 process CPU-time accounting; AVX-512/Arm64/Windows waived",
        "scope is not the bounded native x86 scope",
    )
    require(
        artifact.get("claim_class") == "native FastMath process CPU-time and timing evidence",
        "claim class mismatch",
    )
    require("RUSAGE_CHILDREN" in str(artifact.get("reference")), "reference omits child CPU counter")
    canonical_input = artifact.get("canonical_input")
    require(isinstance(canonical_input, str) and bool(canonical_input), "canonical input missing")
    if isinstance(canonical_input, str):
        require(
            artifact.get("canonical_input_sha256") == digest(canonical_input.encode()),
            "canonical input digest mismatch",
        )
    require(artifact.get("counter_source") == "resource.getrusage(RUSAGE_CHILDREN): ru_utime + ru_stime", "counter source mismatch")
    require(artifact.get("modes") == list(MODES), "mode declaration mismatch")
    require(artifact.get("repeats_per_mode") == 2, "repeat declaration mismatch")

    host = artifact.get("host")
    require(isinstance(host, dict), "host record missing")
    if isinstance(host, dict):
        require(host.get("architecture") == "x86_64", "host architecture mismatch")
        require(isinstance(host.get("cpu"), str) and bool(host.get("cpu")), "host CPU description missing")

    energy_status = artifact.get("energy_status")
    require(energy_status in ("unavailable", "measured"), "invalid energy status")
    energy_claim = str(artifact.get("energy_claim", ""))
    if energy_status == "unavailable":
        require("no energy claim" in energy_claim, "unavailable energy must carry no-energy claim")
    if energy_status == "measured":
        require("measured" in energy_claim, "measured energy status has no measured claim")

    runs = artifact.get("runs")
    require(isinstance(runs, list) and len(runs) == 4, "expected exactly four child runs")
    expected_keys = {(mode, repeat) for mode in MODES for repeat in range(2)}
    actual_keys: set[tuple[object, object]] = set()
    if isinstance(runs, list):
        for index, run in enumerate(runs):
            prefix = f"run {index}"
            if not isinstance(run, dict):
                failures.append(f"{prefix}: expected object")
                continue
            require(set(run) == RUN_FIELDS, f"{prefix}: unexpected or missing fields")
            mode = run.get("mode")
            repeat = run.get("repeat")
            if isinstance(mode, str) and isinstance(repeat, int) and not isinstance(repeat, bool):
                actual_keys.add((mode, repeat))
            require(mode in MODES, f"{prefix}: invalid mode")
            require(repeat in (0, 1), f"{prefix}: invalid repeat")
            expected_dispatch = "avx2" if mode == "avx2" else "scalar"
            require(run.get("dispatch") == expected_dispatch, f"{prefix}: dispatch mismatch")
            require(run.get("correctness_gate") == "passed", f"{prefix}: correctness gate failed")
            require(run.get("architecture") == "x86_64", f"{prefix}: architecture mismatch")
            require(isinstance(host, dict) and run.get("cpu") == host.get("cpu"), f"{prefix}: CPU mismatch")
            require(run.get("samples_per_regime") == 4096, f"{prefix}: sample count mismatch")
            require(run.get("batches") == 32, f"{prefix}: batch count mismatch")
            for counter in ("cpu_user_ns", "cpu_system_ns", "process_cpu_ns", "wall_ns"):
                value = run.get(counter)
                require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{prefix}: invalid {counter}")
            require(
                run.get("process_cpu_ns") == run.get("cpu_user_ns", 0) + run.get("cpu_system_ns", 0),
                f"{prefix}: CPU total does not equal user plus system",
            )
            stdout_sha = run.get("stdout_sha256")
            require(isinstance(stdout_sha, str) and len(stdout_sha) == 64, f"{prefix}: invalid stdout digest")
            require(isinstance(run.get("energy_available"), bool), f"{prefix}: energy availability is not boolean")
            require(isinstance(run.get("energy_source"), str) and bool(run.get("energy_source")), f"{prefix}: energy source missing")
            regimes = run.get("regimes")
            regime_names = [row.get("name") if isinstance(row, dict) else None for row in regimes] if isinstance(regimes, list) else []
            require(
                isinstance(regimes, list)
                and regime_names == list(REGIMES),
                f"{prefix}: regime order mismatch",
            )
            if isinstance(regimes, list):
                for regime_index, row in enumerate(regimes):
                    regime_prefix = f"{prefix} regime {regime_index}"
                    if not isinstance(row, dict):
                        failures.append(f"{regime_prefix}: expected object")
                        continue
                    require(set(row) == {"name", *REGIME_FIELDS}, f"{regime_prefix}: unexpected or missing fields")
                    require(row.get("name") in REGIMES, f"{regime_prefix}: invalid name")
                    for field in REGIME_FIELDS:
                        value = row.get(field)
                        if field.endswith("energy_joules"):
                            if energy_status == "unavailable":
                                require(value is None, f"{regime_prefix}: unavailable energy is not null")
                            else:
                                require(finite(value) and value >= 0, f"{regime_prefix}: invalid measured energy")
                        else:
                            require(finite(value), f"{regime_prefix}: invalid {field}")
                    for field in ("fast_ns_per_sample", "scalar_ns_per_sample", "libm_ns_per_sample"):
                        require(row.get(field, 0) > 0, f"{regime_prefix}: non-positive {field}")

    require(actual_keys == expected_keys, "run mode/repeat set mismatch")
    if isinstance(runs, list) and runs:
        energy_values = [run.get("energy_available") for run in runs if isinstance(run, dict)]
        boolean_energy_values = [value for value in energy_values if isinstance(value, bool)]
        require(len(boolean_energy_values) == len(energy_values), "child energy availability is not boolean")
        require(set(boolean_energy_values) == {energy_status == "measured"}, "energy status does not match child reports")
        if isinstance(host, dict):
            require({run.get("cpu") for run in runs if isinstance(run, dict)} == {host.get("cpu")}, "host CPU changed across runs")

    summary = artifact.get("summary")
    require(isinstance(summary, dict), "summary missing")
    if isinstance(summary, dict) and isinstance(runs, list) and len(runs) == 4:
        for mode in MODES:
            values = [run["process_cpu_ns"] for run in runs if isinstance(run, dict) and run.get("mode") == mode]
            wall_values = [run["wall_ns"] for run in runs if isinstance(run, dict) and run.get("mode") == mode]
            if len(values) == 2:
                require(close_enough(summary.get(f"{mode}_mean_process_cpu_ns"), sum(values) / 2), f"summary {mode} CPU mean mismatch")
                require(close_enough(summary.get(f"{mode}_mean_wall_ns"), sum(wall_values) / 2), f"summary {mode} wall mean mismatch")
                require(abs(values[1] - values[0]) / max(values) >= 0, f"summary {mode} spread is negative")
        spreads = []
        for mode in MODES:
            values = [run["process_cpu_ns"] for run in runs if isinstance(run, dict) and run.get("mode") == mode]
            if len(values) == 2:
                spreads.append(abs(values[1] - values[0]) / max(values))
        if len(spreads) == len(MODES):
            require(
                close_enough(summary.get("maximum_repeat_cpu_spread_fraction"), max(spreads)),
                "summary maximum repeat spread mismatch",
            )
        avx_mean = summary.get("avx2_mean_process_cpu_ns")
        scalar_mean = summary.get("scalar_mean_process_cpu_ns")
        require(finite(avx_mean) and finite(scalar_mean) and scalar_mean > 0, "summary means invalid")
        if finite(avx_mean) and finite(scalar_mean) and scalar_mean > 0:
            require(close_enough(summary.get("avx2_to_scalar_process_cpu_ratio"), avx_mean / scalar_mean), "summary ratio mismatch")

    canonical_output = artifact.get("canonical_output")
    require(isinstance(canonical_output, str) and bool(canonical_output), "canonical output missing")
    if isinstance(canonical_output, str):
        require(artifact.get("output_sha256") == digest(canonical_output.encode()), "canonical output digest mismatch")
        try:
            require(json.loads(canonical_output) == runs, "canonical output does not match runs")
        except json.JSONDecodeError:
            failures.append("canonical output is not JSON")

    sidecar = artifact_path.parent / "checksums.sha256"
    try:
        lines = [line for line in sidecar.read_text(encoding="utf-8").splitlines() if line]
    except OSError:
        lines = []
    require(len(lines) == 1, "checksums.sha256 must contain exactly one entry")
    if len(lines) == 1:
        parts = lines[0].split("  ", 1)
        require(len(parts) == 2 and parts[1] == artifact_path.name, "checksum sidecar names the wrong artifact")
        if len(parts) == 2:
            require(parts[0] == digest(artifact_path.read_bytes()), "artifact checksum sidecar mismatch")
    return failures


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    artifact_path = Path(sys.argv[1]).resolve()
    failures = check(artifact_path)
    result = {"schema": SCHEMA, "status": "fail" if failures else "pass", "failures": failures}
    print(json.dumps(result, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
