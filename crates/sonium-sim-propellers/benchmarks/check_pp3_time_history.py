#!/usr/bin/env python3
"""Strict checks for the native PP3 moving FW-H time-history ladder."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


EXPECTED_REFINEMENTS = [16, 32, 64, 128, 256]


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text())

    assert artifact["schema"] == "sonium.propellers.time-history-evidence.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64 waived"
    assert artifact["claim_class"] == "analytic stationary moving FW-H time-history reference"
    assert "first order" in artifact["reference"]
    canonical_input = artifact["canonical_input"]
    assert isinstance(canonical_input, str) and canonical_input
    assert hashlib.sha256(canonical_input.encode()).hexdigest() == artifact["canonical_input_sha256"]

    assert artifact["dimensions"] == [
        "source_time_s",
        "retarded_time_s",
        "time_step_s",
        "surface_area_m2",
        "observer_distance_m",
    ]
    levels = artifact["levels"]
    assert [level["refinement_intervals"] for level in levels] == EXPECTED_REFINEMENTS
    assert len(levels) == len(EXPECTED_REFINEMENTS)
    previous_error = None
    for level, refinement in zip(levels, EXPECTED_REFINEMENTS):
        assert level["sample_count"] == refinement + 1
        assert level["step_s"] == 1.0 / refinement
        for key in (
            "step_s",
            "retarded_time_s",
            "source_observer_distance_m",
            "doppler_factor",
            "observed_thickness_pressure_pa",
            "observed_loading_pressure_pa",
            "observed_total_pressure_pa",
            "expected_total_pressure_pa",
            "absolute_error_pa",
            "relative_error",
        ):
            assert finite(level[key]), f"non-finite level field {key}"
        assert abs(level["retarded_time_s"] - 0.5) <= 1.0e-11
        assert abs(level["source_observer_distance_m"] - 10.0) <= 1.0e-12
        assert abs(level["doppler_factor"] - 1.0) <= 1.0e-14
        assert level["expected_total_pressure_pa"] > 0.0
        assert level["absolute_error_pa"] >= 0.0
        assert level["relative_error"] == level["absolute_error_pa"] / level["expected_total_pressure_pa"]
        if previous_error is not None:
            assert level["absolute_error_pa"] < previous_error
        previous_error = level["absolute_error_pa"]

    assert artifact["exact_convergence_order"] == 1.0
    orders = artifact["observed_convergence_orders"]
    assert len(orders) == len(levels) - 1
    assert all(finite(order) and 0.8 < order < 1.2 for order in orders)
    assert abs(artifact["observed_convergence_order"] - sum(orders) / len(orders)) <= 1.0e-14
    assert finite(artifact["max_relative_error"])
    assert finite(artifact["final_relative_error"])
    assert artifact["final_relative_error"] == levels[-1]["relative_error"]
    assert artifact["final_relative_error"] <= artifact["tolerance"]
    assert artifact["converged"] is True

    probe = artifact["invalid_probe"]
    assert probe["id"] == "observer-time-before-history-window"
    assert probe["status"] == "typed_rejected"
    assert "RetardedTimeOutsideHistory" in probe["typed_error"]

    canonical_output = artifact["canonical_output"]
    assert isinstance(canonical_output, str) and canonical_output
    assert hashlib.sha256(canonical_output.encode()).hexdigest() == artifact["output_sha256"]
    assert json.loads(canonical_output) == levels
    print(
        json.dumps(
            {
                "levels": len(levels),
                "observed_convergence_order": artifact["observed_convergence_order"],
                "final_relative_error": artifact["final_relative_error"],
                "status": "pass",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
