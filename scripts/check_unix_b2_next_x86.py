#!/usr/bin/env python3
"""Strict checker for the bounded native-x86 B2 setup packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


SCHEMA = "sonium-bem-native-x86-b2-setup/v1"
EXPECTED_DIGEST_NAME = "sha256"


def fail(message: str) -> None:
    raise SystemExit(f"B2 setup check failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def finite_number(value: object, name: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{name} is not finite")
    return number


def positive_integer(value: object, name: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{name} is not positive")
    return value


def nonempty_provenance(value: object, name: str) -> None:
    require(isinstance(value, str) and value.strip(), f"{name} is empty")
    require(value.strip().lower() not in {"unknown", "unspecified"}, f"{name} is unresolved")


def read_digest(checksum_path: Path) -> str:
    require(checksum_path.is_file(), f"missing {EXPECTED_DIGEST_NAME} sidecar {checksum_path}")
    fields = checksum_path.read_text(encoding="utf-8").split()
    require(fields and len(fields[0]) == 64, f"malformed digest sidecar {checksum_path}")
    try:
        int(fields[0], 16)
    except ValueError:
        fail(f"malformed hexadecimal digest in {checksum_path}")
    return fields[0].lower()


def check_artifact(artifact_path: Path, checksum_path: Path) -> dict[str, object]:
    require(artifact_path.is_file(), f"missing artifact {artifact_path}")
    expected_digest = read_digest(checksum_path)
    actual_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    require(actual_digest == expected_digest, "artifact digest does not match sidecar")

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
    require("cuda_claim=false" in features and "avx_claim=false" in features, "feature claims are not explicit")
    require("avx512" not in features.lower(), "AVX-512 must not be implied by this packet")
    for field in ("cpu_model", "microcode", "compiler", "thread_policy", "affinity_policy", "numa_policy", "deterministic_mode"):
        nonempty_provenance(artifact.get(field), field)
    require(artifact["thread_policy"] == "single", "thread policy is not single")
    require(artifact["affinity_policy"] == "unbound", "affinity policy is not explicit")
    require(artifact["numa_policy"] == "default", "NUMA policy is not explicit")
    require(artifact["deterministic_mode"] == "enabled", "deterministic mode is not enabled")

    capability = artifact.get("capability")
    require(isinstance(capability, dict), "capability proof is missing")
    require(
        capability
        == {
            "operator_family": "HMatrix+MLFMM-setup",
            "precision": "Complex64",
            "execution": "cpu-reference-only",
            "accelerated": False,
            "cuda_claim": False,
            "avx_claim": False,
        },
        "capability proof is not the bounded CPU reference claim",
    )

    require(artifact.get("fixture_elements") == 16, "fixture element count changed")
    require(artifact.get("fixture_dofs") == 16, "fixture DOF count changed")
    require(finite_number(artifact.get("fixture_frequency_hz"), "fixture frequency") == 42.0, "fixture frequency changed")
    require(artifact.get("fixture_precision") == "Complex64", "fixture precision changed")

    hierarchy = artifact.get("hierarchy")
    require(isinstance(hierarchy, dict), "hierarchy packet is missing")
    level_count = positive_integer(hierarchy.get("level_count"), "hierarchy level count")
    clusters_per_level = hierarchy.get("clusters_per_level")
    require(isinstance(clusters_per_level, list) and len(clusters_per_level) == level_count, "hierarchy level shape changed")
    require(clusters_per_level[0] == 1, "hierarchy has no singleton root")
    require(all(isinstance(count, int) and 0 < count <= 16 for count in clusters_per_level), "invalid hierarchy cluster counts")
    require(all(left <= right for left, right in zip(clusters_per_level, clusters_per_level[1:])), "hierarchy levels are not monotone")
    require(hierarchy.get("root_element_count") == 16, "root does not cover the fixture")
    require(positive_integer(hierarchy.get("leaf_cluster_count"), "leaf cluster count") >= 2, "leaf topology is not split")
    require(hierarchy.get("leaf_element_count") == 16, "leaf clusters do not cover the fixture")
    require(positive_integer(hierarchy.get("near_relation_count"), "near relation count") > 0, "near topology is empty")
    require(positive_integer(hierarchy.get("far_relation_count"), "far relation count") > 0, "far topology is empty")
    for field in ("expansion_terms", "theta_points", "phi_points"):
        values = hierarchy.get(field)
        require(isinstance(values, list) and len(values) == level_count, f"{field} shape changed")
        require(all(positive_integer(value, f"{field} entry") for value in values), f"{field} contains an invalid entry")

    hmatrix = artifact.get("hmatrix")
    require(isinstance(hmatrix, dict), "H-matrix packet is missing")
    require(hmatrix.get("num_dofs") == 16, "H-matrix DOF count changed")
    dense_blocks = positive_integer(hmatrix.get("dense_block_count"), "H-matrix dense block count")
    low_rank_blocks = positive_integer(hmatrix.get("low_rank_block_count"), "H-matrix low-rank block count")
    require(dense_blocks >= 2 and low_rank_blocks >= 2, "H-matrix setup did not retain both block families")
    require(positive_integer(hmatrix.get("memory_bytes"), "H-matrix memory") > 0, "H-matrix memory is empty")
    require(0.0 < finite_number(hmatrix.get("compression_ratio"), "H-matrix compression ratio") < 1.0, "H-matrix is not compressed")
    min_rank = positive_integer(hmatrix.get("min_rank"), "H-matrix minimum rank")
    max_rank = positive_integer(hmatrix.get("max_rank"), "H-matrix maximum rank")
    require(min_rank <= max_rank < 8, "H-matrix far block is full rank")
    require(finite_number(hmatrix.get("mean_rank"), "H-matrix mean rank") >= min_rank, "H-matrix rank statistics are inconsistent")
    require(hmatrix.get("saturated_count") == 0, "H-matrix rank cap was saturated")
    require(finite_number(hmatrix.get("saturation_fraction"), "H-matrix saturation fraction") == 0.0, "H-matrix saturation fraction changed")

    validation = artifact.get("validation")
    require(isinstance(validation, dict) and validation.get("status") == "pass", "validation did not pass")
    action_budget = finite_number(validation.get("action_residual_budget"), "action residual budget")
    determinism_budget = finite_number(validation.get("determinism_budget"), "determinism budget")
    require(action_budget == 1.0e-8 and determinism_budget == 1.0e-12, "validation budgets changed")
    for field in ("dense_action_residual", "transpose_action_residual", "hermitian_action_residual"):
        require(finite_number(hmatrix.get(field), field) <= action_budget, f"{field} exceeds budget")

    mlfmm = artifact.get("mlfmm")
    require(isinstance(mlfmm, dict), "MLFMM packet is missing")
    require(mlfmm.get("num_dofs") == 16 and mlfmm.get("num_levels") == level_count, "MLFMM dimensions changed")
    for field in ("near_block_count", "t_matrix_count", "d_matrix_count", "s_matrix_count"):
        positive_integer(mlfmm.get(field), f"MLFMM {field}")
    sphere_points = mlfmm.get("sphere_points_per_level")
    require(isinstance(sphere_points, list) and len(sphere_points) == level_count, "MLFMM sphere layout changed")
    require(all(positive_integer(value, "MLFMM sphere point count") for value in sphere_points), "MLFMM sphere layout is empty")
    require(mlfmm.get("checked_actions_passed") == 3, "not all checked MLFMM actions ran")
    for field in ("forward_repeat_residual", "transpose_repeat_residual", "hermitian_repeat_residual"):
        require(finite_number(mlfmm.get(field), field) <= determinism_budget, f"{field} exceeds budget")

    wavenumber = finite_number(validation.get("wavenumber_probe"), "wavenumber probe")
    require(wavenumber > 0.0, "wavenumber probe is not positive")
    return {
        "schema": SCHEMA,
        "status": "pass",
        "artifact": str(artifact_path),
        "artifact_sha256": actual_digest,
        "scope": "native-x86 CPU reference only; no CUDA, AVX, or cross-architecture claim",
        "h_dense_blocks": dense_blocks,
        "h_low_rank_blocks": low_rank_blocks,
        "h_max_rank": max_rank,
        "mlfmm_levels": level_count,
        "mlfmm_checked_actions": mlfmm["checked_actions_passed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_artifact(args.artifact, args.checksum)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
