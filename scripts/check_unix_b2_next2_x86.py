#!/usr/bin/env python3
"""Strict checker for the bounded native-x86 B2 physical packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


SCHEMA = "sonium-bem-native-x86-b2-physical/v1"


def fail(message: str) -> None:
    raise SystemExit(f"B2 physical check failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def finite(value: object, name: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{name} is not finite")
    return number


def positive_integer(value: object, name: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{name} is not positive")
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
    for field in ("cpu_model", "microcode", "compiler", "thread_policy", "affinity_policy", "numa_policy", "deterministic_mode"):
        provenance(artifact.get(field), field)
    require(artifact["thread_policy"] == "single", "thread policy is not single")
    require(artifact["affinity_policy"] == "unbound", "affinity policy is not explicit")
    require(artifact["numa_policy"] == "default", "NUMA policy is not explicit")
    require(artifact["deterministic_mode"] == "enabled", "deterministic mode is not enabled")
    require(artifact.get("execution") == "cpu-reference-only", "execution claim is not CPU reference-only")
    require(artifact.get("cuda_claim") is False and artifact.get("avx_claim") is False, "unsupported capability claim")

    require(finite(artifact.get("patch_separation_m"), "patch separation") == 12.0, "fixture spacing changed")
    require(finite(artifact.get("frequency_hz"), "frequency") == 42.0, "fixture frequency changed")
    require(artifact.get("element_count") == 16, "element count changed")
    require(artifact.get("dof_count") == 16, "DOF count changed")
    require(artifact.get("field_point_count") == 4, "field point count changed")

    validation = artifact.get("validation")
    require(isinstance(validation, dict) and validation.get("status") == "pass", "validation did not pass")
    action_budget = finite(validation.get("action_budget"), "action budget")
    field_budget = finite(validation.get("field_budget"), "field budget")
    mlfmm_action_budget = finite(validation.get("mlfmm_action_budget"), "MLFMM action budget")
    mlfmm_field_budget = finite(validation.get("mlfmm_field_budget"), "MLFMM field budget")
    derivative_budget = finite(validation.get("derivative_budget"), "derivative budget")
    source_derivative_budget = finite(validation.get("source_derivative_budget"), "source derivative budget")
    require(action_budget == 1e-7 and field_budget == 1e-7, "H-matrix budgets changed")
    require(mlfmm_action_budget == 1e-2 and mlfmm_field_budget == 1e-3, "MLFMM budgets changed")
    require(derivative_budget == 1e-5 and source_derivative_budget == 1e-8, "derivative budgets changed")

    hmatrix_low_rank = positive_integer(artifact.get("hmatrix_low_rank_blocks"), "H-matrix low-rank blocks")
    hmatrix_max_rank = positive_integer(artifact.get("hmatrix_max_rank"), "H-matrix maximum rank")
    require(hmatrix_low_rank >= 2 and hmatrix_max_rank < 8, "H-matrix compression evidence is too weak")
    require(artifact.get("hmatrix_saturated_blocks") == 0, "H-matrix rank cap was saturated")
    for field in ("matrix_action_residual", "hmatrix_field_residual", "hmatrix_field_energy_residual"):
        require(finite(artifact.get(field), field) <= (action_budget if field == "matrix_action_residual" else field_budget), f"{field} exceeds budget")
    for field in ("dense_field_energy", "hmatrix_field_energy", "mlfmm_field_energy"):
        require(finite(artifact.get(field), field) > 0.0, f"{field} is not positive")

    require(finite(artifact.get("mlfmm_action_residual"), "MLFMM action residual") <= mlfmm_action_budget, "MLFMM action bound exceeded")
    require(finite(artifact.get("mlfmm_field_residual"), "MLFMM field residual") <= mlfmm_field_budget, "MLFMM field bound exceeded")
    require(finite(artifact.get("mlfmm_field_energy_residual"), "MLFMM field energy residual") <= mlfmm_field_budget, "MLFMM field energy bound exceeded")
    require(positive_integer(artifact.get("mlfmm_levels"), "MLFMM levels") >= 2, "MLFMM hierarchy is too shallow")
    require(positive_integer(artifact.get("mlfmm_d_matrices"), "MLFMM D matrices") > 0, "MLFMM transfer setup is empty")
    for field in ("hmatrix_wavenumber_derivative_residual", "mlfmm_wavenumber_derivative_residual"):
        require(finite(artifact.get(field), field) <= derivative_budget, f"{field} exceeds budget")
    require(finite(artifact.get("source_wavenumber_derivative_residual"), "source derivative residual") <= source_derivative_budget, "source derivative exceeds budget")

    return {
        "schema": SCHEMA,
        "status": "pass",
        "artifact": str(artifact_path),
        "artifact_sha256": digest,
        "scope": "native-x86 CPU reference; H-matrix physical field parity; MLFMM setup and bounded low-ka approximation; no CUDA/AVX claim",
        "elements": artifact["element_count"],
        "hmatrix_low_rank_blocks": hmatrix_low_rank,
        "hmatrix_action_residual": artifact["matrix_action_residual"],
        "hmatrix_field_residual": artifact["hmatrix_field_residual"],
        "mlfmm_action_residual": artifact["mlfmm_action_residual"],
        "mlfmm_field_residual": artifact["mlfmm_field_residual"],
        "mlfmm_wavenumber_derivative_residual": artifact["mlfmm_wavenumber_derivative_residual"],
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
