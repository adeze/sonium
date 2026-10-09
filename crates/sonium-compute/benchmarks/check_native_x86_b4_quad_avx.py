#!/usr/bin/env python3
"""Strict checker for native x86 Quad4 AVX2/FMA parity evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


ARTIFACT_KEYS = {
    "schema",
    "status",
    "target_arch",
    "os",
    "cpu_model",
    "microcode",
    "compiler",
    "capability",
    "element_family",
    "pair_kind",
    "phase_regimes",
    "basis_orders",
    "gauss_orders",
    "tail_policy",
    "parity_budget",
    "cases_expected",
    "cases_passed",
    "cases_failed",
    "max_abs_error",
    "max_relative_error",
    "cases",
    "evidence_sha256",
    "unsupported_scope",
}
CAPABILITY_KEYS = {
    "avx2",
    "fma",
    "avx512f",
    "selected_kernel",
    "scalar_oracle",
    "precision",
}
CASE_KEYS = {
    "phase",
    "frequency_hz",
    "basis",
    "basis_count",
    "gauss_order",
    "effective_order",
    "field_quadrature_points",
    "avx2_batches",
    "scalar_tail_points",
    "complex_observables",
    "max_abs_error",
    "max_relative_error",
}
PHASES = {"low": 42.0, "acoustic": 420.0, "high": 1000.0}
BASES = {"linear": 4, "quadratic": 9}
ORDERS = {1: 8, 2: 9, 4: 11}
CASE_COUNT = len(PHASES) * len(BASES) * len(ORDERS)
PARITY_BUDGET = 5.0e-11
UNSUPPORTED_SCOPE = (
    "no Quad4 singular/touching AVX claim, weak-form derivative AVX claim, "
    "f32/C32 Quad4 claim, AVX512, Arm64, or Windows claim"
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B4 Quad4 AVX artifact rejected: {message}")


def integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        fail(f"{label} is not a non-negative integer")
    return value


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} provenance is missing")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    sidecar = Path(f"{path}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{sidecar} is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1].lstrip("*") == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"{sidecar} has no valid digest for {path.name}")
    actual = sha256(path)
    if actual != expected:
        fail(f"checksum mismatch for {path.name}: {actual} != {expected}")
    return value


def check_artifact(artifact: dict) -> dict:
    if set(artifact) != ARTIFACT_KEYS:
        fail(
            "artifact fields differ: "
            f"missing={ARTIFACT_KEYS - set(artifact)}, extra={set(artifact) - ARTIFACT_KEYS}"
        )
    if (
        artifact["schema"] != "sonium-bem-native-x86-b4-quad-avx/v1"
        or artifact["status"] != "pass"
        or artifact["target_arch"] != "x86_64"
        or artifact["os"] != "linux-x86_64"
    ):
        fail("artifact identity/target is invalid")
    for key in ("cpu_model", "microcode", "compiler"):
        text(artifact[key], key)
    capability = artifact["capability"]
    if not isinstance(capability, dict) or set(capability) != CAPABILITY_KEYS:
        fail("capability fields are incomplete")
    if (
        capability["avx2"] is not True
        or capability["fma"] is not True
        or type(capability["avx512f"]) is not bool
        or capability["selected_kernel"] != "quad4-regular-avx2-fma"
        or capability["scalar_oracle"] != "galerkin_single_layer_regular_quad4_with_basis_scalar"
        or capability["precision"] != "Complex64"
    ):
        fail("runtime AVX2/FMA dispatch or oracle provenance is invalid")
    if (
        artifact["element_family"] != "Quad4"
        or artifact["pair_kind"] != "separated-regular"
        or artifact["phase_regimes"] != "low:42Hz,acoustic:420Hz,high:1000Hz"
        or artifact["basis_orders"] != "linear:4,quadratic:9"
        or artifact["gauss_orders"] != "1,2,4 (effective 8,9,11)"
        or artifact["tail_policy"]
        != "four field quadrature points per AVX2/FMA batch; scalar remainder"
    ):
        fail("bounded Quad4 workload contract is invalid")
    if (
        artifact["parity_budget"] != PARITY_BUDGET
        or artifact["cases_expected"] != CASE_COUNT
        or artifact["cases_passed"] != CASE_COUNT
        or artifact["cases_failed"] != 0
    ):
        fail("case count or parity budget is incomplete")

    cases = artifact["cases"]
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        fail("case matrix cardinality is invalid")
    seen = set()
    max_abs = 0.0
    max_relative = 0.0
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            fail(f"case {index} fields are incomplete")
        phase = case["phase"]
        basis = case["basis"]
        order = case["gauss_order"]
        if phase not in PHASES or case["frequency_hz"] != PHASES[phase]:
            fail(f"case {index} phase/frequency is invalid")
        if basis not in BASES or case["basis_count"] != BASES[basis]:
            fail(f"case {index} basis is invalid")
        if order not in ORDERS or case["effective_order"] != ORDERS[order]:
            fail(f"case {index} quadrature order is invalid")
        identity = (phase, basis, order)
        if identity in seen:
            fail(f"duplicate case {identity}")
        seen.add(identity)
        effective = ORDERS[order]
        field_points = effective * effective
        if (
            case["field_quadrature_points"] != field_points
            or case["avx2_batches"] != field_points * (field_points // 4)
            or case["scalar_tail_points"] != field_points * (field_points % 4)
            or case["complex_observables"] != BASES[basis] * BASES[basis] * 6
        ):
            fail(f"case {index} does not prove vector batches and scalar tails")
        case_abs = finite(case["max_abs_error"], f"case {index}.max_abs_error")
        case_relative = finite(case["max_relative_error"], f"case {index}.max_relative_error")
        if case_abs > PARITY_BUDGET or case_relative > PARITY_BUDGET:
            fail(f"case {index} exceeds scalar-oracle parity budget")
        max_abs = max(max_abs, case_abs)
        max_relative = max(max_relative, case_relative)
    if len(seen) != CASE_COUNT:
        fail("case matrix does not cover every phase/basis/order combination")
    if artifact["max_abs_error"] != max_abs or artifact["max_relative_error"] != max_relative:
        fail("aggregate error does not match case measurements")
    if not isinstance(artifact["evidence_sha256"], str) or len(artifact["evidence_sha256"]) != 64:
        fail("evidence digest is malformed")
    if any(c not in "0123456789abcdef" for c in artifact["evidence_sha256"]):
        fail("evidence digest is not hexadecimal")
    text(artifact["unsupported_scope"], "unsupported_scope")
    if artifact["unsupported_scope"] != UNSUPPORTED_SCOPE:
        fail("unsupported scope is not explicit")
    return {
        "selected_kernel": capability["selected_kernel"],
        "precision": capability["precision"],
        "cases": len(cases),
        "max_abs_error": max_abs,
        "max_relative_error": max_relative,
        "evidence_sha256": artifact["evidence_sha256"],
    }


def main() -> None:
    if len(sys.argv) not in {2, 4} or (len(sys.argv) == 4 and sys.argv[2] != "--output"):
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json [--output SUMMARY.json]")
    path = Path(sys.argv[1])
    evidence = check_artifact(load(path))
    summary = {
        "schema": "sonium-bem-native-x86-b4-quad-avx/v1",
        "status": "passed",
        "target_arch": "x86_64",
        "os": "linux-x86_64",
        "backend": "cpu-avx2-fma",
        "avx_claim": True,
        "artifact_sha256": sha256(path),
        "evidence": evidence,
        "unsupported_scope": UNSUPPORTED_SCOPE,
    }
    if len(sys.argv) == 4:
        output = Path(sys.argv[3])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
