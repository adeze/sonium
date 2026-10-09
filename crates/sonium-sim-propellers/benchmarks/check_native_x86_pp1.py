#!/usr/bin/env python3
"""Strict gate for the native Linux/x86 PP1 evidence producer.

The gate checks the evidence contract and compares two real runs made with
different Rayon worker counts.  It never treats a missing optional backend or
waived architecture as a passing execution.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REQUIRED_CASES = {
    "source.conventional-smooth.FastSurrogate",
    "source.conventional-leading-serrated.FastSurrogate",
    "source.conventional-trailing-serrated.FastSurrogate",
    "source.toroidal-leading-trailing-serrated.FastSurrogate",
    "source.conventional-smooth.ResearchControl",
    "source.conventional-leading-serrated.ResearchControl",
    "source.conventional-trailing-serrated.ResearchControl",
    "source.toroidal-leading-trailing-serrated.ResearchControl",
    "aerodynamics.direct-bem-wake",
    "aerodynamics.vortex-lattice",
    "references.actuator-couette-weighted-flow",
    "adversarial.hover-invalid-nonfinite",
}


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: evidence must be an object")
    return value


def check_one(path: Path) -> dict:
    evidence = load(path)
    assert evidence["schema"] == "sonium.propellers.native-x86-pp1.v1"
    assert evidence["status"] == "pass"
    assert evidence["scope"].startswith("native Linux x86_64")
    host = evidence["host"]
    assert host["os"] == "linux" and host["target"].startswith("x86_64-")
    assert host["compiler"] and host["kernel"] and host["cpu_model"]
    assert host["target_features"]
    assert evidence["deterministic_mode"] is True
    assert evidence["rayon_workers_requested"] == host["rayon_threads"]
    assert host["rayon_threads"] > 0

    execution = evidence["execution"]
    assert execution["requested_backend"] == "cpu"
    assert execution["actual_backend"] == "cpu"
    assert execution["precision"] == "f64"
    assert execution["scalar_or_simd"] == "scalar"
    # CPU surrogate provenance omits a null fallback_reason via serde's
    # skip_serializing_if; an explicit non-null reason is still forbidden.
    assert execution.get("fallback_reason") is None
    assert execution["cancelled"] is False
    assert execution["device_lost"] is False
    assert execution["allocation_failures"] == 0
    assert len(execution["input_checksum_sha256"]) == 64
    assert len(execution["output_checksum_sha256"]) == 64

    cases = evidence["cases"]
    assert len(cases) == len(REQUIRED_CASES)
    assert {case["id"] for case in cases} == REQUIRED_CASES
    for case in cases:
        assert case["status"] == "pass"
        assert case["oracle"]
        assert case["observables"]
        assert all(isinstance(value, (int, float)) for value in case["observables"])
        assert all(value == value and abs(value) != float("inf") for value in case["observables"])

    capability = evidence["capability"]
    assert capability["total_cells"] == 648
    assert capability["cpu_f64_forward_supported"] > 0
    assert capability["cuda_typed_unsupported"] == 216
    assert capability["metal_typed_unsupported"] == 216
    assert capability["device_requests_without_fallback"] is True

    for failure in evidence["failures"]:
        assert failure["status"] == "typed_rejected"
        assert failure["error_kind"]
        assert failure["partial_publish"] is False

    cross_arch = evidence["cross_architecture"]
    assert cross_arch["status"] == "waived"
    assert cross_arch["targets"]
    assert "waived" in cross_arch["reason"]

    input_bytes = evidence["input_descriptor"].encode()
    assert evidence["canonical_input_sha256"] == hashlib.sha256(input_bytes).hexdigest()
    checksums = evidence["artifact_checksums_sha256"]
    assert checksums["input_descriptor"] == evidence["canonical_input_sha256"]
    assert checksums["output_observables"] == evidence["canonical_output_sha256"]
    assert path.name not in checksums
    assert evidence["gallery_status"] == "pass"
    gallery = path.parent / "gallery"
    assert (gallery / "gallery.json").is_file()
    manifest = load(gallery / "gallery.json")
    names = {figure["filename"] for figure in manifest["figures"]}
    assert len(names) == 10
    assert {
        "source_spectrum.csv",
        "blade_load_distribution.csv",
        "optimized-blade-loading.png",
        "axial-wake-proxy.png",
        "spectrum.svg",
        "thrust-versus-operating-point.svg",
        "torque-versus-operating-point.svg",
        "blade-load-distribution.svg",
        "conventional-toroidal-comparison.svg",
        "rotation.gif",
    } == names
    sidecar = path.parent / "checksums.sha256"
    assert sidecar.is_file(), f"missing checksum sidecar: {sidecar}"
    checksum_lines = [line.split() for line in sidecar.read_text().splitlines() if line.strip()]
    assert checksum_lines == [[hashlib.sha256(path.read_bytes()).hexdigest(), path.name]]
    return evidence


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} SINGLE CONFIGURED")
    single_path, configured_path = map(Path, sys.argv[1:])
    single = check_one(single_path)
    configured = check_one(configured_path)
    assert single["rayon_workers_requested"] == 1
    assert configured["rayon_workers_requested"] >= 2
    assert len(single["output_observables"]) == len(configured["output_observables"])
    maximum_error = max(
        abs(left - right)
        for left, right in zip(single["output_observables"], configured["output_observables"])
    )
    assert maximum_error <= 1.0e-12, f"Rayon determinism error {maximum_error}"
    print(json.dumps({"cases": len(single["cases"]), "rayon_max_abs_error": maximum_error}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
