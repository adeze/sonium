#!/usr/bin/env python3
"""Strict native-x86 checker for the SK3 speaker golden artifact."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SK3 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "unavailable":
        fail(f"{label} is missing")
    return value


def verify_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        fail(f"checksum sidecar has no valid digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}")


def check_shape(rows: object, outer: int, inner: int, label: str) -> None:
    if not isinstance(rows, list) or len(rows) != outer:
        fail(f"{label} must have {outer} rows")
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != inner:
            fail(f"{label}[{row_index}] must have {inner} entries")


def check_complex_grid(rows: object, outer: int, inner: int, label: str) -> None:
    check_shape(rows, outer, inner, label)
    for row_index, row in enumerate(rows):
        for point_index, point in enumerate(row):
            if not isinstance(point, dict):
                fail(f"{label}[{row_index}][{point_index}] is not an object")
            finite(point.get("re"), f"{label}[{row_index}][{point_index}].re")
            finite(point.get("im"), f"{label}[{row_index}][{point_index}].im")


def check_driver(payload: object, label: str, backend: str, precision: str) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        fail(f"{label} is not DriverModel schema 2")
    provenance = payload.get("provenance") or {}
    if provenance.get("backend") != backend or provenance.get("precision") != precision:
        fail(f"{label} provenance backend/precision is inconsistent")
    if not isinstance(payload.get("points"), list) or len(payload["points"]) != 2:
        fail(f"{label} must contain two frequency points")
    execution = payload.get("execution") or {}
    if execution.get("requested_backend") != backend or execution.get("actual_backend") != backend:
        fail(f"{label} execution backend is inconsistent")
    if execution.get("precision") != precision or not nonempty(execution.get("device"), f"{label} device"):
        fail(f"{label} execution precision/device is invalid")
    if not nonempty(execution.get("toolchain"), f"{label} toolchain"):
        fail(f"{label} toolchain is missing")
    convergence = execution.get("convergence") or {}
    if convergence.get("converged") is not True:
        fail(f"{label} is not converged")
    finite(convergence.get("max_relative_residual"), f"{label} residual")
    finite(convergence.get("power_closure_relative_error"), f"{label} power closure")
    promotion = execution.get("promotion") or {}
    if promotion.get("promoted") is not True:
        fail(f"{label} is not promotion-valid")
    for key in (
        "full_order_relative_residual",
        "projected_relative_residual",
        "gradient_relative_error",
        "maximum_constraint_violation",
        "rank_agreement",
    ):
        finite(promotion.get(key), f"{label} promotion.{key}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON: {error}")
    if artifact.get("schema_version") != 3 or artifact.get("target_arch") != "x86_64":
        fail("schema version or target architecture is invalid")
    if artifact.get("target_os") != "linux":
        fail("target OS is not native Linux")
    for key in ("cpu_model", "microcode", "compiler", "cpu_vendor"):
        nonempty(artifact.get(key), key)
    runtime = artifact.get("runtime_features") or {}
    for key in ("avx2", "fma", "sse4.2", "avx512f"):
        if not isinstance(runtime.get(key), bool):
            fail(f"runtime_features.{key} is not a boolean")
    if not runtime["avx2"] or not runtime["fma"] or not runtime["sse4.2"]:
        fail("required native x86 runtime features are absent")

    dispatch = artifact.get("dispatch")
    if not isinstance(dispatch, list):
        fail("dispatch provenance is missing")
    for entry in ("target_arch=x86_64", "scalar=f64", "avx2=true", "fma=true", "sse4.2=true"):
        if entry not in dispatch:
            fail(f"dispatch is missing {entry}")
    vendor_entries = [entry for entry in dispatch if entry.startswith("vendor=")]
    if len(vendor_entries) != 1 or vendor_entries[0].removeprefix("vendor=").strip().lower() in {"", "unknown"}:
        fail("dispatch vendor identity is missing")

    frequencies = artifact.get("frequencies")
    if not isinstance(frequencies, list) or len(frequencies) != 2:
        fail("the golden must contain two frequencies")
    for index, frequency in enumerate(frequencies):
        finite(frequency, f"frequencies[{index}]")
    if not frequencies[0] < frequencies[1] or abs(frequencies[0] - 500.0) > 1e-8 or abs(frequencies[1] - 1000.0) > 1e-8:
        fail("frequency grid is not the pinned 500/1000 Hz fixture")

    check_complex_grid(artifact.get("directivity"), 2, 72, "directivity")
    cea = artifact.get("cea_on_axis")
    if not isinstance(cea, list) or len(cea) != 2:
        fail("CEA on-axis result must contain two values")
    for index, value in enumerate(cea):
        finite(value, f"cea_on_axis[{index}]")
    finite(artifact.get("cea_preference_score"), "cea_preference_score")

    preview = artifact.get("reduced_order_preview") or {}
    preview_frequencies = preview.get("frequencies")
    if preview_frequencies != frequencies:
        fail("reduced preview frequency grid differs from the full solve")
    check_shape(preview.get("spl"), 2, 72, "reduced_order_preview.spl")
    check_shape(preview.get("phase"), 2, 72, "reduced_order_preview.phase")
    for name in ("spl", "phase"):
        for row_index, row in enumerate(preview[name]):
            for point_index, value in enumerate(row):
                finite(value, f"reduced_order_preview.{name}[{row_index}][{point_index}]")

    residual_gate = artifact.get("residual_gate") or {}
    if residual_gate.get("status") != "passed":
        fail("residual gate did not pass")
    finite(residual_gate.get("tolerance"), "residual_gate.tolerance")
    check_driver(artifact.get("driver_model_reference"), "full DriverModel", "cpu", "f64")
    check_driver(artifact.get("projected_driver_model_reference"), "projected DriverModel", "wasm", "f32")
    verify_checksum(path)
    print(f"SK3 native speaker artifact passed: {path}")


if __name__ == "__main__":
    main()
