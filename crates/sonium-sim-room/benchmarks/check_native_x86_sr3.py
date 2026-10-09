#!/usr/bin/env python3
"""Gate the native/scalar RoomSim SR3 evidence bundle.

The producer owns the numerical gates.  This checker verifies that the two
effective dispatch runs describe the same physical problem, that the native
run really selected AVX2/FMA, and that the sidecar contains explicit waivers
instead of silently treating absent cross-architecture/vendor evidence as a
pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"SR3 evidence rejected: {message}")


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot parse {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def walk_numbers(left: Any, right: Any, label: str) -> tuple[float, int]:
    if isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), f"{label} shape differs")
        maximum = 0.0
        count = 0
        for index, (a, b) in enumerate(zip(left, right)):
            error, entries = walk_numbers(a, b, f"{label}[{index}]")
            maximum = max(maximum, error)
            count += entries
        return maximum, count
    if isinstance(left, dict) and isinstance(right, dict):
        require(left.keys() == right.keys(), f"{label} keys differ")
        maximum = 0.0
        count = 0
        for key in left:
            error, entries = walk_numbers(left[key], right[key], f"{label}.{key}")
            maximum = max(maximum, error)
            count += entries
        return maximum, count
    require(isinstance(left, (int, float)) and isinstance(right, (int, float)), f"{label} type differs")
    require(math.isfinite(float(left)) and math.isfinite(float(right)), f"{label} is non-finite")
    return abs(float(left) - float(right)), 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: check_native_x86_sr3.py NATIVE.json SCALAR.json")
    native_path, scalar_path = map(Path, sys.argv[1:])
    native = load(native_path)
    scalar = load(scalar_path)

    for name, artifact in (("native", native), ("scalar", scalar)):
        require(artifact.get("schema_version") == 2, f"{name} schema is not v2")
        require(artifact.get("artifact_id") == "sonium-sim-room.native-x86-sr3.v2", f"{name} artifact id differs")
        require(artifact.get("host_architecture") == "x86_64", f"{name} is not native x86_64")
        require(artifact.get("cross_architecture", {}).get("status") == "waived", f"{name} lacks explicit Arm64 waiver")
        require(artifact.get("convergence", {}).get("status") == "pass", f"{name} convergence gate did not pass")
        require(artifact.get("reciprocity", {}).get("status") == "pass", f"{name} reciprocity gate did not pass")
        require(artifact.get("repeatability", {}).get("status") == "pass", f"{name} repeatability gate did not pass")
        require(artifact.get("vendor_reference", "").startswith("not-benchmarked:"), f"{name} vendor provenance is not explicit")

    require(native["cpu_simd_backend"] == "avx2", "native run did not select AVX2")
    require("avx2=true" in native["dispatch_features"] and "fma=true" in native["dispatch_features"], "native run lacks AVX2/FMA runtime facts")
    require(scalar["cpu_simd_backend"] == "scalar", "scalar run did not select scalar dispatch")
    require("force_scalar=true" in scalar["dispatch_features"], "scalar run lacks forced-scalar provenance")
    require(native["node_coordinates"] == scalar["node_coordinates"], "native/scalar mesh coordinates differ")
    require(native["element_connectivity"] == scalar["element_connectivity"], "native/scalar connectivity differs")
    for key in ("output", "field"):
        maximum, count = walk_numbers(native[key], scalar[key], key)
        require(count > 0, f"{key} has no numeric samples")
        require(maximum <= 1.0e-10, f"native/scalar {key} difference {maximum} exceeds 1e-10")

    convergence = native["convergence"]
    levels = convergence.get("levels", [])
    require(len(levels) == 3, "convergence ladder does not contain coarse/medium/fine")
    require(convergence.get("monotonic") is True, "convergence ladder is not monotonic")
    counts = [(level["node_count"], level["element_count"]) for level in levels]
    require(counts[0][0] < counts[1][0] < counts[2][0], "convergence node counts do not refine")
    require(counts[0][1] < counts[1][1] < counts[2][1], "convergence element counts do not refine")
    for evidence_name in ("reciprocity", "repeatability"):
        evidence = native[evidence_name]
        require(evidence["sample_count"] > 0, f"{evidence_name} has no samples")
        require(evidence["rms_relative_error"] <= evidence["max_rms_relative_error"], f"{evidence_name} RMS exceeds budget")
        require(evidence["max_relative_error"] <= evidence["max_max_relative_error"], f"{evidence_name} maximum exceeds budget")

    check = {
        "schema_version": 1,
        "artifact_id": "sonium-sim-room.native-x86-sr3-check.v1",
        "status": "pass",
        "native_sha256": digest(native_path),
        "scalar_sha256": digest(scalar_path),
        "native_scalar_max_abs_error": {
            key: walk_numbers(native[key], scalar[key], key)[0] for key in ("output", "field")
        },
        "cross_architecture": native["cross_architecture"],
        "vendor_reference": native["vendor_reference"],
    }
    output = native_path.with_name("native-x86-sr3-check-v1.json")
    output.write_text(json.dumps(check, indent=2, sort_keys=True) + "\n")
    print(json.dumps(check, sort_keys=True))


if __name__ == "__main__":
    main()
