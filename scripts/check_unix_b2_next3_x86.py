#!/usr/bin/env python3
"""Strict checker for the native-x86 B2 MLFMM frequency limitation map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


SCHEMA = "sonium-bem-native-x86-b2-frequency/v1"
FREQUENCIES_HZ = [20.0, 42.0, 60.0, 80.0, 100.0, 140.0, 180.0]
SPEED_OF_SOUND = 343.0
CHARACTERISTIC_LENGTH = 0.8
ACTION_BUDGET = 1.0e-2
FIELD_BUDGET = 1.0e-3


def fail(message: str) -> None:
    raise SystemExit(f"B2 frequency check failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def finite(value: object, name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} is not numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{name} is not finite")
    return number


def positive_integer(value: object, name: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{name} is not positive",
    )
    return value


def provenance(value: object, name: str) -> None:
    require(isinstance(value, str) and value.strip(), f"{name} is empty")
    require(value.strip().lower() not in {"unknown", "unspecified"}, f"{name} is unresolved")


def check(artifact_path: Path, checksum_path: Path) -> dict[str, object]:
    require(artifact_path.is_file(), f"missing artifact {artifact_path}")
    require(checksum_path.is_file(), f"missing checksum sidecar {checksum_path}")
    checksum_fields = checksum_path.read_text(encoding="utf-8").split()
    require(checksum_fields and len(checksum_fields[0]) == 64, "malformed checksum sidecar")
    try:
        int(checksum_fields[0], 16)
    except ValueError:
        fail("checksum sidecar is not hexadecimal")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    require(digest == checksum_fields[0].lower(), "artifact digest does not match sidecar")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot decode artifact: {error}")
    require(isinstance(artifact, dict), "artifact root is not an object")

    require(artifact.get("schema") == SCHEMA, "unexpected schema")
    require(artifact.get("status") == "pass", "artifact status is not pass")
    require(artifact.get("target_arch") == "x86_64", "packet is not x86_64")
    require(artifact.get("os") == "linux-x86_64", "packet is not native Linux x86_64")
    require(artifact.get("selected_kernel") == "cpu-reference", "unexpected selected kernel")
    require(
        artifact.get("runtime_dispatch")
        == "selected_kernel=cpu-reference,cuda_claim=false,avx_claim=false",
        "runtime dispatch makes an unsupported claim",
    )
    features = artifact.get("features")
    require(isinstance(features, str), "features is not text")
    require("cuda_claim=false" in features and "avx_claim=false" in features, "claims are not explicit")
    require("avx512" not in features.lower(), "AVX-512 must not be implied")
    for field in (
        "cpu_model",
        "microcode",
        "compiler",
        "thread_policy",
        "affinity_policy",
        "numa_policy",
        "deterministic_mode",
    ):
        provenance(artifact.get(field), field)
    require(artifact["thread_policy"] == "single", "thread policy is not single")
    require(artifact["affinity_policy"] == "unbound", "affinity policy is not explicit")
    require(artifact["numa_policy"] == "default", "NUMA policy is not explicit")
    require(artifact["deterministic_mode"] == "enabled", "deterministic mode is not enabled")
    require(artifact.get("execution") == "cpu-reference-only", "execution is not CPU reference-only")
    require(artifact.get("cuda_claim") is False and artifact.get("avx_claim") is False, "unsupported capability claim")

    require(finite(artifact.get("patch_separation_m"), "patch separation") == 12.0, "fixture spacing changed")
    characteristic_length = finite(artifact.get("characteristic_length_m"), "characteristic length")
    require(characteristic_length == CHARACTERISTIC_LENGTH, "characteristic length changed")
    require(artifact.get("element_count") == 16, "element count changed")
    require(artifact.get("dof_count") == 16, "DOF count changed")
    require(artifact.get("field_point_count") == 4, "field point count changed")
    action_budget = finite(artifact.get("mlfmm_action_budget"), "MLFMM action budget")
    field_budget = finite(artifact.get("mlfmm_field_budget"), "MLFMM field budget")
    require(action_budget == ACTION_BUDGET and field_budget == FIELD_BUDGET, "budgets changed")

    rows = artifact.get("rows")
    require(isinstance(rows, list) and len(rows) == len(FREQUENCIES_HZ), "frequency row count changed")
    validation = artifact.get("validation")
    require(isinstance(validation, dict) and validation.get("status") == "pass", "validation did not pass")
    required = validation.get("required_frequencies_hz")
    require(required == FREQUENCIES_HZ, "required frequency list changed")
    require(finite(validation.get("low_ka_action_budget"), "validation action budget") == ACTION_BUDGET, "validation action budget changed")
    require(finite(validation.get("low_ka_field_budget"), "validation field budget") == FIELD_BUDGET, "validation field budget changed")

    bounded = exceeded = no_transfer = 0
    for row, expected_frequency in zip(rows, FREQUENCIES_HZ):
        require(isinstance(row, dict), "frequency row is not an object")
        frequency = finite(row.get("frequency_hz"), "row frequency")
        require(frequency == expected_frequency, "frequency ordering or value changed")
        wave_number = finite(row.get("wave_number"), "row wave number")
        expected_wave_number = math.tau * frequency / SPEED_OF_SOUND
        require(math.isclose(wave_number, expected_wave_number, rel_tol=1e-12, abs_tol=1e-12), "wave number is inconsistent")
        ka = finite(row.get("ka"), "row ka")
        require(math.isclose(ka, wave_number * characteristic_length, rel_tol=1e-12, abs_tol=1e-12), "ka is inconsistent")
        levels = positive_integer(row.get("mlfmm_levels"), "MLFMM levels")
        require(levels >= 2, "MLFMM hierarchy is too shallow")
        d_matrices = row.get("mlfmm_d_matrices")
        require(isinstance(d_matrices, int) and not isinstance(d_matrices, bool) and d_matrices >= 0, "D-matrix count is invalid")
        for field in (
            "dense_field_energy",
            "mlfmm_field_energy",
            "mlfmm_action_residual",
            "mlfmm_field_residual",
            "mlfmm_field_energy_residual",
        ):
            number = finite(row.get(field), field)
            require(number >= 0.0, f"{field} is negative")
        require(finite(row["dense_field_energy"], "dense field energy") > 0.0, "dense field energy is not positive")
        require(finite(row["mlfmm_field_energy"], "MLFMM field energy") > 0.0, "MLFMM field energy is not positive")
        status = row.get("status")
        if d_matrices == 0:
            expected_status = "no-transfer-data"
            no_transfer += 1
        elif (
            row["mlfmm_action_residual"] <= action_budget
            and row["mlfmm_field_residual"] <= field_budget
            and row["mlfmm_field_energy_residual"] <= field_budget
        ):
            expected_status = "within-low-ka-budget"
            bounded += 1
        else:
            expected_status = "exceeds-low-ka-budget"
            exceeded += 1
        require(status == expected_status, f"row {frequency:g} has dishonest status")

    require(bounded >= 1, "map has no bounded low-ka row")
    require(exceeded >= 1, "map has no limitation row")
    require(validation.get("bounded_rows") == bounded, "bounded row count mismatch")
    require(validation.get("exceeded_rows") == exceeded, "exceeded row count mismatch")
    require(validation.get("no_transfer_rows") == no_transfer, "no-transfer row count mismatch")
    hundred = next(row for row in rows if row["frequency_hz"] == 100.0)
    require(hundred["mlfmm_action_residual"] > 5.0e-2, "100 Hz limitation observation disappeared")

    return {
        "schema": SCHEMA,
        "status": "pass",
        "artifact": str(artifact_path),
        "artifact_sha256": digest,
        "scope": "native-x86 CPU reference; fixed MLFMM frequency/ka limitation map; no CUDA/AVX claim",
        "rows": len(rows),
        "bounded_rows": bounded,
        "exceeded_rows": exceeded,
        "no_transfer_rows": no_transfer,
        "ka_at_42_hz": next(row["ka"] for row in rows if row["frequency_hz"] == 42.0),
        "action_at_42_hz": next(row["mlfmm_action_residual"] for row in rows if row["frequency_hz"] == 42.0),
        "action_at_100_hz": hundred["mlfmm_action_residual"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.artifact, args.checksum)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
