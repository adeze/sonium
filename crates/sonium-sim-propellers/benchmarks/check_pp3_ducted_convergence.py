#!/usr/bin/env python3
"""Strict checker for the native PP3 reduced-order ducted convergence ladder."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


EXPECTED_SEGMENTS = [4, 8, 16, 32, 64, 128]
LENGTH_M = 0.8
INLET_AREA_M2 = 0.01
OUTLET_AREA_M2 = 0.025
HYDRAULIC_DIAMETER_M = 0.04
MASS_FLOW_KG_S = 0.01
DENSITY_KG_M3 = 1.2
DYNAMIC_VISCOSITY_PA_S = 1.0e-3
INLET_STATIC_PRESSURE_PA = 101325.0
SPEED_OF_SOUND_M_S = 343.0
FINAL_TOLERANCE = 1.0e-2


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left: float, right: float, *, rel: float = 1.0e-11, absolute: float = 1.0e-12) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=absolute)


def exact_wall_loss() -> float:
    coefficient = 32.0 * DYNAMIC_VISCOSITY_PA_S * MASS_FLOW_KG_S / (
        DENSITY_KG_M3 * HYDRAULIC_DIAMETER_M**2
    )
    return coefficient * LENGTH_M / (OUTLET_AREA_M2 - INLET_AREA_M2) * math.log(
        OUTLET_AREA_M2 / INLET_AREA_M2
    )


def exact_outlet_pressure() -> float:
    inlet_velocity = MASS_FLOW_KG_S / (DENSITY_KG_M3 * INLET_AREA_M2)
    outlet_velocity = MASS_FLOW_KG_S / (DENSITY_KG_M3 * OUTLET_AREA_M2)
    inlet_dynamic = 0.5 * DENSITY_KG_M3 * inlet_velocity**2
    outlet_dynamic = 0.5 * DENSITY_KG_M3 * outlet_velocity**2
    return INLET_STATIC_PRESSURE_PA + inlet_dynamic - outlet_dynamic - exact_wall_loss()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text())

    assert artifact["schema"] == "sonium.propellers.ducted-convergence-evidence.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64/AVX512 waived"
    assert artifact["claim_class"] == "production reduced-order duct/shroud station-resolution convergence"
    assert "independent" in artifact["reference"]
    assert "laminar Darcy" in artifact["independent_oracle"]
    canonical_input = artifact["canonical_input"]
    assert isinstance(canonical_input, str) and canonical_input
    assert hashlib.sha256(canonical_input.encode()).hexdigest() == artifact["canonical_input_sha256"]
    assert artifact["dimensions"] == [
        "axial_segments",
        "station_count",
        "axial_step_m",
        "flow_area_m2",
        "hydraulic_diameter_m",
        "mass_flow_kg_s",
        "density_kg_m3",
        "dynamic_viscosity_pa_s",
        "wall_loss_pa",
        "static_pressure_pa",
        "continuity_residual_kg_s",
        "reynolds_number",
        "mach_number",
    ]

    expected_loss = exact_wall_loss()
    expected_pressure = exact_outlet_pressure()
    levels = artifact["levels"]
    assert [level["axial_segments"] for level in levels] == EXPECTED_SEGMENTS
    assert len(levels) == len(EXPECTED_SEGMENTS)
    wall_errors = []
    pressure_errors = []
    for level, segments in zip(levels, EXPECTED_SEGMENTS):
        assert level["station_count"] == segments + 1
        assert close(level["axial_step_m"], LENGTH_M / segments)
        for key in (
            "axial_step_m",
            "expected_wall_loss_pa",
            "observed_wall_loss_pa",
            "relative_wall_loss_error",
            "expected_outlet_static_pressure_pa",
            "observed_outlet_static_pressure_pa",
            "relative_outlet_pressure_error",
            "maximum_continuity_residual_kg_s",
            "maximum_velocity_m_s",
            "maximum_mach",
        ):
            assert finite(level[key]), f"non-finite level field {key}"
        assert close(level["expected_wall_loss_pa"], expected_loss)
        assert close(level["expected_outlet_static_pressure_pa"], expected_pressure)
        # The coarsest right-endpoint station rule is intentionally not
        # treated as exact; the convergence gate below owns the accuracy
        # claim.  Keep this wide enough to admit the measured coarse error
        # while still rejecting a dimensionally unrelated result.
        assert close(level["observed_wall_loss_pa"], level["expected_wall_loss_pa"], rel=0.15)
        assert close(
            level["observed_outlet_static_pressure_pa"],
            level["expected_outlet_static_pressure_pa"],
            rel=2.0e-5,
        )
        expected_wall_error = abs(level["observed_wall_loss_pa"] - expected_loss) / abs(expected_loss)
        expected_pressure_error = abs(level["observed_outlet_static_pressure_pa"] - expected_pressure) / abs(expected_pressure)
        assert close(level["relative_wall_loss_error"], expected_wall_error, rel=1.0e-10)
        assert close(level["relative_outlet_pressure_error"], expected_pressure_error, rel=1.0e-10)
        assert level["finite"] is True
        assert level["maximum_continuity_residual_kg_s"] <= 1.0e-12
        assert level["maximum_velocity_m_s"] > 0.0
        assert 0.0 < level["maximum_mach"] < 0.6
        reynolds_range = level["reynolds_number_range"]
        assert len(reynolds_range) == 2
        assert all(finite(value) and value > 0.0 for value in reynolds_range)
        assert reynolds_range[0] < reynolds_range[1] < 2300.0
        wall_errors.append(level["relative_wall_loss_error"])
        pressure_errors.append(level["relative_outlet_pressure_error"])

    assert all(right < left for left, right in zip(wall_errors, wall_errors[1:]))
    assert all(right < left for left, right in zip(pressure_errors, pressure_errors[1:]))
    observed_orders = [math.log(left / right, 2.0) for left, right in zip(wall_errors, wall_errors[1:])]
    assert len(artifact["observed_orders"]) == len(observed_orders)
    assert all(close(left, right, rel=1.0e-10) for left, right in zip(artifact["observed_orders"], observed_orders))
    assert all(math.isfinite(order) and order > 0.7 for order in observed_orders)
    assert close(artifact["minimum_observed_order"], min(observed_orders), rel=1.0e-10)
    assert close(artifact["final_relative_wall_loss_error"], wall_errors[-1], rel=1.0e-10)
    assert close(artifact["final_relative_pressure_error"], pressure_errors[-1], rel=1.0e-10)
    assert artifact["final_relative_wall_loss_error"] <= FINAL_TOLERANCE
    assert artifact["final_relative_pressure_error"] <= FINAL_TOLERANCE
    assert artifact["final_tolerance"] == FINAL_TOLERANCE
    assert artifact["converged"] is True
    assert artifact["deterministic_repeat"] is True

    probes = artifact["invalid_probes"]
    assert [probe["id"] for probe in probes] == [
        "non-monotonic-axial-station",
        "station-budget-exceeded",
    ]
    assert all(probe["status"] == "typed_rejected" for probe in probes)
    assert all(probe["publication"] == "none" for probe in probes)
    assert "NonMonotonicAxial" in probes[0]["typed_error"]
    assert "TooManyStations" in probes[1]["typed_error"]

    canonical_output = artifact["canonical_output"]
    assert isinstance(canonical_output, str) and canonical_output
    assert hashlib.sha256(canonical_output.encode()).hexdigest() == artifact["output_sha256"]
    assert json.loads(canonical_output) == levels
    print(
        json.dumps(
            {
                "levels": len(levels),
                "minimum_observed_order": artifact["minimum_observed_order"],
                "final_relative_wall_loss_error": artifact["final_relative_wall_loss_error"],
                "status": "pass",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
