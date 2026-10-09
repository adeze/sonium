#!/usr/bin/env python3
"""Strict schema and budget gate for the FEM4 native-x86 artifact."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM4 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def verify_checksum(path: Path) -> None:
    checksum_path = path.with_name("checksums.sha256")
    try:
        rows = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0]
            break
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected.lower()
    ):
        fail(f"checksum sidecar has no valid digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected.lower():
        fail(f"checksum mismatch for {path.name}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON: {error}")

    if artifact.get("schema") != "sonium-fem-native-x86-evidence/v1":
        fail("schema/version mismatch")
    if artifact.get("artifact_version") != 1 or artifact.get("status") != "pass":
        fail("artifact is not a passing version-1 record")
    if artifact.get("scope") != "native-linux-x86_64":
        fail("native Linux x86_64 scope is missing")
    verify_checksum(path)

    target = artifact.get("target") or {}
    if target.get("os") != "linux" or target.get("arch") != "x86_64":
        fail("target identity is not native Linux x86_64")
    if not target.get("cpu_model") or not target.get("microcode") or not target.get("compiler"):
        fail("CPU/microcode/compiler provenance is missing")
    runtime = target.get("runtime_features") or {}
    compile_features = target.get("compile_target_features") or {}
    for features, label in ((runtime, "runtime_features"), (compile_features, "compile_target_features")):
        if any(not isinstance(features.get(name), bool) for name in ("avx2", "fma", "avx512f")):
            fail(f"{label} does not record AVX feature booleans")

    deterministic = artifact.get("deterministic_mode") or {}
    if deterministic.get("requested") is not True or deterministic.get("rayon_threads") != "1":
        fail("deterministic-mode provenance is incomplete")
    dispatch = artifact.get("dispatch") or {}
    if dispatch.get("selected_path") != "portable-rust-reference":
        fail("unexpected/unobservable FEM dispatch path")
    if dispatch.get("vendor_library") != "none-linked":
        fail("unverified vendor library was claimed")

    cross_arch = artifact.get("cross_architecture") or {}
    if cross_arch.get("status") != "waived" or cross_arch.get("comparison") != "not-run":
        fail("cross-architecture waiver is not explicit")
    if "no aarch64" not in cross_arch.get("reason", "").lower():
        fail("cross-architecture waiver does not reject fabricated data")

    cases = artifact.get("cases")
    expected_families = {"triangle-2d", "quadrilateral-2d", "tetrahedron-3d", "hexahedron-3d"}
    expected_cell_metadata = {
        "triangle-2d": ("Triangle", 2),
        "quadrilateral-2d": ("Quadrilateral", 2),
        "tetrahedron-3d": ("Tetrahedron", 3),
        "hexahedron-3d": ("Hexahedron", 3),
    }
    if not isinstance(cases, list) or {case.get("cell_family") for case in cases} != expected_families:
        fail("the four required cell families are not present exactly once")
    if len(cases) != len(expected_families):
        fail("duplicate or missing cell-family case")

    for case in cases:
        family = case.get("cell_family", "unknown")
        if case.get("status") != "pass":
            fail(f"{family} is not passing")
        expected_element, expected_dimension = expected_cell_metadata.get(family, (None, None))
        if case.get("element_type") != expected_element or case.get("dimension") != expected_dimension:
            fail(f"{family} element type/dimension metadata is inconsistent")
        topology = case.get("topology") or {}
        for key in ("nodes", "elements", "global_dofs"):
            if not isinstance(topology.get(key), int) or topology[key] <= 0:
                fail(f"{family} topology.{key} is invalid")
        markers = topology.get("boundary_markers")
        if (
            not isinstance(markers, list)
            or not markers
            or any(not isinstance(marker, int) for marker in markers)
        ):
            fail(f"{family} topology.boundary_markers is invalid")
        interpolation = case.get("interpolation") or {}
        interpolation_budgets = interpolation.get("budgets") or {}
        l2_rate = finite(interpolation.get("l2_rate"), f"{family} l2_rate")
        gradient_rate = finite(interpolation.get("gradient_rate"), f"{family} gradient_rate")
        if l2_rate <= finite(interpolation_budgets.get("l2_rate_min"), f"{family} l2_rate_min"):
            fail(f"{family} misses L2 convergence budget")
        if gradient_rate <= finite(interpolation_budgets.get("gradient_rate_min"), f"{family} gradient_rate_min"):
            fail(f"{family} misses gradient convergence budget")
        for key in ("coarse_l2", "fine_l2", "coarse_gradient", "fine_gradient"):
            finite(interpolation.get(key), f"{family} interpolation.{key}")

        assembly = case.get("assembly") or {}
        if not isinstance(assembly.get("element_matrix_triplets"), int) or assembly["element_matrix_triplets"] <= 0:
            fail(f"{family} element matrix evidence is missing")
        if not isinstance(assembly.get("assembled_matrix_entries"), int) or assembly["assembled_matrix_entries"] <= 0:
            fail(f"{family} element/global matrix evidence is missing")
        assembly_budgets = assembly.get("budgets") or {}
        for value_key, budget_key in (
            ("constant_field_residual_l2", "constant_field_residual_l2_max"),
            ("symmetry_max_abs", "symmetry_max_abs"),
            ("mass_domain_abs_error", "mass_domain_abs_error"),
        ):
            if finite(assembly.get(value_key), f"{family} {value_key}") > finite(assembly_budgets.get(budget_key), f"{family} {budget_key}"):
                fail(f"{family} misses {value_key} budget")

        boundary = case.get("boundary_conditions") or {}
        if boundary.get("kind") != "homogeneous-dirichlet" or not isinstance(boundary.get("constrained_dofs"), int) or boundary["constrained_dofs"] <= 0:
            fail(f"{family} boundary contract is missing")
        if finite(boundary.get("destination_preserved_contract_max_abs"), f"{family} BC contract") > finite(boundary.get("budget"), f"{family} BC budget"):
            fail(f"{family} BC contract budget exceeded")

    print(f"FEM4 native artifact passed: {path}")


if __name__ == "__main__":
    main()
