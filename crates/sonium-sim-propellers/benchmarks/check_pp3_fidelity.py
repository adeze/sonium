#!/usr/bin/env python3
"""Strict checks for the native PP3 fidelity/convergence artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


EXPECTED_CASES = {
    f"source.{label}.{mode}"
    for mode in ("FastSurrogate", "ResearchControl")
    for label in (
        "conventional-smooth",
        "conventional-leading-serrated",
        "conventional-trailing-serrated",
        "toroidal-leading-trailing-serrated",
    )
}
EXPECTED_CASES |= {
    "ladder.direct-bem-steady-and-prescribed-wake",
    "ladder.vortex-lattice-panel",
    "ladder.body-fitted-weighted-couette",
    "observer.dipole-fwh-installation",
    "refinement.geometry-and-response",
    "promotion.optimizer-gallery",
    "adversarial.hover-zero-load",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text())

    assert artifact["schema"] == "sonium.propellers.fidelity-evidence.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64 waived"
    assert hashlib.sha256(artifact["canonical_input"].encode()).hexdigest() == artifact[
        "canonical_input_sha256"
    ]

    cases = artifact["cases"]
    assert {case["id"] for case in cases} == EXPECTED_CASES
    assert len(cases) == len(EXPECTED_CASES)
    for case in cases:
        assert case["status"] == "pass"
        assert case["converged"] is True
        assert case["tolerance"] > 0.0
        assert case["observables"]
        assert len(case["observables"]) == len(case["observable_names"])
        assert case["dimensions"]
        assert case["gates"]
        assert all(isinstance(value, (int, float)) for value in case["observables"])
        assert all(value == value and abs(value) != float("inf") for value in case["observables"])

    failures = artifact["failures"]
    assert failures
    assert all(failure["status"] == "typed_rejected" for failure in failures)
    assert all(failure["partial_publish"] is False for failure in failures)
    assert any("DuctShroudError::InvalidOptions" in failure["typed_error"] for failure in failures)

    assert artifact["external_measured_data"].startswith("unavailable:")
    assert artifact["resolved_cfd_and_full_fwh"].startswith("unavailable:")
    assert artifact["optimizer_gallery_promotion"].startswith("pass:")
    assert len(artifact["output_sha256"]) == 64
    print(json.dumps({"cases": len(cases), "typed_failures": len(failures), "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
