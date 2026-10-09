#!/usr/bin/env python3
"""Strict checks for the native PP3 observer-distance ladder artifact."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


EXPECTED_DISTANCES = [2.0, 4.0, 8.0, 16.0]


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text())

    assert artifact["schema"] == "sonium.propellers.observer-distance-evidence.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64 waived"
    assert artifact["claim_class"] == "analytic stationary compact FW-H observer reference"
    assert "exact inverse-distance identity" in artifact["reference"]
    canonical_input = artifact["canonical_input"]
    assert isinstance(canonical_input, str) and canonical_input
    assert hashlib.sha256(canonical_input.encode()).hexdigest() == artifact["canonical_input_sha256"]

    assert artifact["dimensions"] == [
        "observer_distance_m",
        "harmonic_frequency_hz",
        "surface_area_m2",
        "source_observer_phase",
    ]
    levels = artifact["levels"]
    assert [level["distance_m"] for level in levels] == EXPECTED_DISTANCES
    assert len(levels) == len(EXPECTED_DISTANCES)
    for level in levels:
        for key in (
            "distance_m",
            "expected_pressure_pa",
            "observed_pressure_pa",
            "scaled_pressure_pa_m",
            "relative_error",
            "source_observer_distance_m",
        ):
            assert finite(level[key]), f"non-finite level field {key}"
        assert level["distance_m"] == level["source_observer_distance_m"]
        assert level["expected_pressure_pa"] > 0.0
        assert level["observed_pressure_pa"] > 0.0
        assert level["relative_error"] <= artifact["tolerance"]

    assert artifact["inverse_distance_exponent"] == 1.0
    assert finite(artifact["max_relative_error"])
    assert finite(artifact["max_scaled_relative_change"])
    assert artifact["max_relative_error"] <= artifact["tolerance"]
    assert artifact["max_scaled_relative_change"] <= artifact["tolerance"]
    assert artifact["converged"] is True

    probe = artifact["invalid_probe"]
    assert probe["id"] == "observer-distance-below-minimum"
    assert probe["status"] == "typed_rejected"
    assert "ObserverTooClose" in probe["typed_error"]

    canonical_output = artifact["canonical_output"]
    assert isinstance(canonical_output, str) and canonical_output
    assert hashlib.sha256(canonical_output.encode()).hexdigest() == artifact["output_sha256"]
    assert json.loads(canonical_output) == levels
    print(
        json.dumps(
            {
                "levels": len(levels),
                "max_relative_error": artifact["max_relative_error"],
                "max_scaled_relative_change": artifact["max_scaled_relative_change"],
                "status": "pass",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
