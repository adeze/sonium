#!/usr/bin/env python3
"""Strict checker for native x86 B4 dispatch-parity evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


OBSERVABLE_COUNT = 1634
PARITY_BUDGET = 5.0e-5
QUAD4_SCHEMA = "sonium-bem-native-x86-b4-quad/v1"
QUAD4_CASE_COUNT = 48
QUAD4_PAIR_CALL_COUNT = 186
QUAD4_OBSERVABLE_COUNT = 18042
QUAD4_MATRIX_ACTION_BUDGET = 1.0e-12
QUAD4_FIELD_BUDGET = 1.0e-12
QUAD4_WAVENUMBER_DERIVATIVE_BUDGET = 5.0e-4


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B4 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def required_text(artifact: dict, key: str) -> str:
    value = artifact.get(key)
    if not isinstance(value, str) or value.strip().lower() in {"", "unknown", "unavailable", "unspecified"}:
        fail(f"{key} provenance is missing")
    return value


def verify_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{sidecar} is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        fail(f"{sidecar} has no valid digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}")


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    verify_checksum(path)
    return value


def check_artifact(artifact: dict, label: str, expected_status: str, expected_kernel: str, force_scalar: bool) -> None:
    if artifact.get("schema") != 2 or artifact.get("status") != expected_status:
        fail(f"{label} schema/status is invalid")
    if artifact.get("target_arch") != "x86_64" or artifact.get("selected_kernel") != expected_kernel:
        fail(f"{label} target/kernel identity is invalid")
    if artifact.get("force_scalar") is not force_scalar:
        fail(f"{label} scalar-forcing state is invalid")
    if artifact.get("cases_passed") != 5 or artifact.get("cases_failed") != 0 or artifact.get("cases_skipped") != 0:
        fail(f"{label} case counts are not a complete pass")
    observables = artifact.get("observables")
    if not isinstance(observables, list) or len(observables) != OBSERVABLE_COUNT:
        fail(f"{label} must contain exactly {OBSERVABLE_COUNT} observables")
    for index, value in enumerate(observables):
        finite(value, f"{label}.observables[{index}]")
    for key in ("cpu_model", "microcode", "compiler", "thread_policy", "affinity_policy", "numa_policy", "deterministic_mode"):
        required_text(artifact, key)
    if artifact.get("os") != "linux-x86_64":
        fail(f"{label} OS provenance is not native Linux x86_64")
    required_features = ("avx2=true", "fma=true") if not force_scalar else ("force_scalar=true",)
    features = artifact.get("features", "")
    if not isinstance(features, str) or any(feature not in features for feature in required_features):
        fail(f"{label} feature provenance is incomplete")
    if artifact.get("runtime_dispatch") != f"selected_kernel={expected_kernel},force_scalar={str(force_scalar).lower()}":
        fail(f"{label} runtime dispatch provenance is inconsistent")
    if artifact.get("kernels") != "blas1,blas2,blas3,spmv,bem":
        fail(f"{label} kernel-family coverage is incomplete")
    if artifact.get("bem_elements") != "Tri3-constant:f64/C64,Tri3-constant:f32/C32,Quad4-and-higher:unsupported":
        fail(f"{label} element-family coverage is inconsistent")
    if artifact.get("bem_phase_regimes") != "static,low,acoustic,high":
        fail(f"{label} phase-regime coverage is incomplete")
    if artifact.get("bem_precisions") != "f64/C64,f32/C32":
        fail(f"{label} precision coverage is incomplete")
    if artifact.get("aarch64_comparison_max_abs_error") is not None:
        finite(artifact["aarch64_comparison_max_abs_error"], f"{label} aarch64 comparison")
    physical = artifact.get("physical_validation") or {}
    if physical.get("status") != "not-run" or physical.get("budget") != "not-applicable":
        fail(f"{label} physical scope is not explicitly bounded")
    required_text(physical, "reason")
    finite(artifact.get("max_norm_error"), f"{label} max_norm_error")
    if not isinstance(artifact.get("max_ulp_error"), int) or artifact["max_ulp_error"] < 0:
        fail(f"{label} max_ulp_error is invalid")


def check_quad4_artifact(artifact: dict, native: dict) -> None:
    """Check the production CPU Quad4 companion cell.

    Quad4 has no sonium-compute AVX2/FMA entry point.  The artifact is useful
    evidence because it executes the production sonium-bem assembly on native
    x86, but it is deliberately not a second dispatch lane.
    """
    if artifact.get("schema") != QUAD4_SCHEMA or artifact.get("status") != "pass":
        fail("quad4 schema/status is invalid")
    if artifact.get("target_arch") != "x86_64" or artifact.get("os") != "linux-x86_64":
        fail("quad4 target/OS identity is not native Linux x86_64")
    if artifact.get("selected_kernel") != "cpu-scalar-reference":
        fail("quad4 selected kernel is not the scalar reference")
    if artifact.get("runtime_dispatch") != "selected_kernel=cpu-scalar-reference,avx_claim=false":
        fail("quad4 runtime dispatch claims an accelerated path")
    if artifact.get("cases_passed") != QUAD4_CASE_COUNT or artifact.get("cases_failed") != 0 or artifact.get("cases_skipped") != 0:
        fail("quad4 case counts are not a complete pass")
    if artifact.get("pair_calls") != QUAD4_PAIR_CALL_COUNT:
        fail("quad4 pair-call count is not the complete tail matrix")
    observables = artifact.get("observables")
    if not isinstance(observables, list) or len(observables) != QUAD4_OBSERVABLE_COUNT:
        fail(f"quad4 must contain exactly {QUAD4_OBSERVABLE_COUNT} observables")
    for index, value in enumerate(observables):
        finite(value, f"quad4.observables[{index}]")
    for key in ("cpu_model", "microcode", "compiler", "thread_policy", "affinity_policy", "numa_policy", "deterministic_mode"):
        required_text(artifact, key)
        if artifact.get(key) != native.get(key):
            fail(f"native/quad4 {key} provenance differs")
    if artifact.get("os") != native.get("os") or artifact.get("target_arch") != native.get("target_arch"):
        fail("native/quad4 target provenance differs")
    features = artifact.get("features", "")
    if not isinstance(features, str) or "quad4_acceleration=false" not in features:
        fail("quad4 feature provenance does not record scalar-only execution")
    if "quad4_acceleration=true" in features or "avx_claim=true" in features:
        fail("quad4 feature provenance claims acceleration")
    capability = artifact.get("capability")
    if capability != {
        "element_family": "Quad4",
        "basis_orders": "linear,quadratic",
        "precision": "Complex64",
        "execution": "scalar-reference-only",
        "accelerated": False,
        "avx_claim": False,
    }:
        fail("quad4 typed capability proof is invalid")
    if artifact.get("phase_regimes") != "low,acoustic,high":
        fail("quad4 phase-regime coverage is incomplete")
    if artifact.get("phase_regimes_not_run") != "static (PhysicsParams requires positive frequency)":
        fail("quad4 static-phase boundary is not explicit")
    if artifact.get("tail_lengths") != "0,1,2,3,4,5,7,9":
        fail("quad4 tail coverage is incomplete")
    if artifact.get("singular_pair_families") != "separated,common-edge,common-vertex,self":
        fail("quad4 pair-family coverage is incomplete")
    finite(artifact.get("max_abs_entry"), "quad4 max_abs_entry")
    if artifact.get("norm") != "maximum absolute matrix-entry magnitude":
        fail("quad4 norm declaration is invalid")
    physical = artifact.get("physical_validation") or {}
    if physical.get("status") != "pass" or physical.get("budget") != "strict packet gates":
        fail("quad4 physical packet is not a strict pass")
    required_text(physical, "reason")
    packet = artifact.get("physical_packet")
    if not isinstance(packet, dict) or packet.get("status") != "pass":
        fail("quad4 physical packet is missing or not a pass")
    if packet.get("execution") != "scalar-reference-only":
        fail("quad4 physical packet does not identify scalar-only execution")
    if packet.get("basis") != "linear" or packet.get("precision") != "Complex64":
        fail("quad4 physical packet basis/precision is invalid")
    if packet.get("geometry") != "separated":
        fail("quad4 physical packet geometry is not the bounded separated case")
    matrix_action_budget = finite(packet.get("matrix_action_budget"), "quad4 matrix action budget")
    if matrix_action_budget != QUAD4_MATRIX_ACTION_BUDGET:
        fail("quad4 matrix action budget changed")
    matrix_action_residual = finite(packet.get("matrix_action_residual"), "quad4 matrix action residual")
    if matrix_action_residual > matrix_action_budget:
        fail("quad4 matrix action residual exceeds budget")
    field_budget = finite(packet.get("field_residual_budget"), "quad4 field budget")
    if field_budget != QUAD4_FIELD_BUDGET:
        fail("quad4 field budget changed")
    field_residual = finite(packet.get("field_checked_legacy_residual"), "quad4 field residual")
    if field_residual > field_budget:
        fail("quad4 checked/legacy field residual exceeds budget")
    field_magnitude = finite(packet.get("field_pressure_magnitude"), "quad4 field pressure magnitude")
    if field_magnitude <= 0.0:
        fail("quad4 field pressure observable is empty")
    finite(packet.get("field_pressure_re"), "quad4 field pressure real part")
    finite(packet.get("field_pressure_im"), "quad4 field pressure imaginary part")
    wavenumber_step = finite(packet.get("wavenumber_step"), "quad4 wavenumber step")
    if wavenumber_step <= 0.0:
        fail("quad4 wavenumber step is not positive")
    derivative_budget = finite(
        packet.get("wavenumber_derivative_budget"),
        "quad4 wavenumber derivative budget",
    )
    if derivative_budget != QUAD4_WAVENUMBER_DERIVATIVE_BUDGET:
        fail("quad4 wavenumber derivative budget changed")
    derivative_residual = finite(
        packet.get("wavenumber_derivative_residual"),
        "quad4 wavenumber derivative residual",
    )
    if derivative_residual > derivative_budget:
        fail("quad4 analytic/centered wavenumber derivative exceeds budget")
    finite(packet.get("wavenumber_derivative_action_re"), "quad4 derivative action real part")
    finite(packet.get("wavenumber_derivative_action_im"), "quad4 derivative action imaginary part")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            f"usage: {sys.argv[0]} NATIVE.json SCALAR.json --quad4 QUAD4.json [--output COMPARISON.json]"
        )
    native_path = Path(sys.argv[1])
    scalar_path = Path(sys.argv[2])
    quad4_path = None
    output_path = None
    index = 3
    while index < len(sys.argv):
        if sys.argv[index] == "--quad4" and index + 1 < len(sys.argv):
            quad4_path = Path(sys.argv[index + 1])
            index += 2
        elif sys.argv[index] == "--output" and index + 1 < len(sys.argv):
            output_path = Path(sys.argv[index + 1])
            index += 2
        else:
            raise SystemExit(
                f"usage: {sys.argv[0]} NATIVE.json SCALAR.json --quad4 QUAD4.json [--output COMPARISON.json]"
            )
    if quad4_path is None:
        fail("--quad4 QUAD4.json is required for the bounded Quad4 cell")
    native = load(native_path)
    scalar = load(scalar_path)
    quad4 = load(quad4_path)
    check_artifact(native, "native", "pass", "avx2-fma", False)
    check_artifact(scalar, "scalar", "scalar-baseline", "scalar-fallback", True)
    check_quad4_artifact(quad4, native)
    for key in ("target_arch", "cpu_model", "microcode", "compiler", "os", "thread_policy", "affinity_policy", "numa_policy", "deterministic_mode", "kernels", "bem_elements", "bem_phase_regimes", "bem_precisions"):
        if native.get(key) != scalar.get(key):
            fail(f"native/scalar {key} provenance differs")
    maximum = max(abs(actual - expected) for actual, expected in zip(native["observables"], scalar["observables"]))
    if maximum > PARITY_BUDGET:
        fail(f"native/scalar observable error {maximum:.6e} exceeds {PARITY_BUDGET:.6e}")
    comparison = {
        "schema": "sonium-compute-native-x86-b4-comparison/v1",
        "status": "passed",
        "native_artifact": native_path.name,
        "scalar_artifact": scalar_path.name,
        "observable_count": len(native["observables"]),
        "max_native_scalar_abs_error": maximum,
        "budget": PARITY_BUDGET,
        "physical_validation": {
            "status": "quad4-scalar-reference-pass",
            "budget": "strict packet gates",
            "reason": "The companion production Quad4 packet passes matrix/action, checked field, and analytic-versus-centered wavenumber derivative gates; it remains scalar-reference-only.",
        },
        "quad4_scalar_artifact": quad4_path.name,
        "quad4_scalar_capability": "scalar-reference-only",
        "quad4_scalar_observable_count": len(quad4["observables"]),
        "quad4_physical_packet": {
            "status": quad4["physical_packet"]["status"],
            "execution": quad4["physical_packet"]["execution"],
            "matrix_action_residual": quad4["physical_packet"]["matrix_action_residual"],
            "field_checked_legacy_residual": quad4["physical_packet"]["field_checked_legacy_residual"],
            "wavenumber_derivative_residual": quad4["physical_packet"]["wavenumber_derivative_residual"],
        },
    }
    if output_path is not None:
        output_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
