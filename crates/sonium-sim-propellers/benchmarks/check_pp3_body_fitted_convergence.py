#!/usr/bin/env python3
"""Strict checker for native PP3 body-fitted FEM convergence evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


EXPECTED_LEVELS = [(2, 2), (4, 3), (8, 6), (16, 8), (32, 16), (64, 24)]
EXPECTED_DIMENSIONS = [
    "axial_elements",
    "radial_elements",
    "mesh_node_count",
    "mesh_element_count",
    "residual",
    "inlet_mean_axial_velocity_m_s",
    "outlet_mean_axial_velocity_m_s",
    "relative_mean_velocity_mismatch",
    "maximum_speed_m_s",
    "pressure_pa",
]
INLET_VELOCITY_M_S = 0.1


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left: float, right: float, *, rel: float = 1.0e-12, absolute: float = 1.0e-14) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=absolute)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text())

    assert artifact["schema"] == "sonium.propellers.body-fitted-convergence-evidence.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64/AVX512 waived"
    assert artifact["claim_class"] == "production body-fitted FEM meridional-flow mesh convergence"
    assert "outlet mean axial velocity approaches" in artifact["reference"]
    assert "constant-area annulus" in artifact["continuity_identity"]
    canonical_input = artifact["canonical_input"]
    assert isinstance(canonical_input, str) and canonical_input
    assert "dynamic_viscosity_pa_s=1" in canonical_input
    assert "profile=[(0,0.02,0.05),(0.1,0.02,0.05)]" in canonical_input
    assert hashlib.sha256(canonical_input.encode()).hexdigest() == artifact["canonical_input_sha256"]
    assert artifact["dimensions"] == EXPECTED_DIMENSIONS

    levels = artifact["levels"]
    assert len(levels) == len(EXPECTED_LEVELS)
    mismatches = []
    for level, (axial_elements, radial_elements) in zip(levels, EXPECTED_LEVELS):
        assert level["axial_elements"] == axial_elements
        assert level["radial_elements"] == radial_elements
        assert level["expected_mesh_node_count"] == (axial_elements + 1) * (radial_elements + 1)
        assert level["mesh_node_count"] == level["expected_mesh_node_count"]
        assert level["expected_mesh_element_count"] == 2 * axial_elements * radial_elements
        assert level["mesh_element_count"] == level["expected_mesh_element_count"]
        assert level["iterations"] > 0
        assert level["converged"] is True
        for key in (
            "residual",
            "inlet_mean_axial_velocity_m_s",
            "outlet_mean_axial_velocity_m_s",
            "relative_mean_velocity_mismatch",
            "maximum_speed_m_s",
            "minimum_pressure_pa",
            "maximum_pressure_pa",
        ):
            assert finite(level[key]), f"non-finite level field {key}"
        assert 0.0 <= level["residual"] <= 1.0e-8
        assert close(level["inlet_mean_axial_velocity_m_s"], INLET_VELOCITY_M_S)
        assert level["relative_mean_velocity_mismatch"] >= 0.0
        assert level["maximum_speed_m_s"] >= 0.0
        assert level["minimum_pressure_pa"] <= level["maximum_pressure_pa"]
        mismatches.append(level["relative_mean_velocity_mismatch"])

    assert all(right < left for left, right in zip(mismatches, mismatches[1:]))
    improvement_factors = [left / right for left, right in zip(mismatches, mismatches[1:])]
    assert artifact["improvement_factors"] == improvement_factors
    assert all(math.isfinite(factor) and factor > 1.0 for factor in improvement_factors)
    assert close(artifact["minimum_improvement_factor"], min(improvement_factors), rel=1.0e-13)
    assert artifact["final_relative_mean_velocity_mismatch"] == mismatches[-1]
    assert 0.0 < artifact["final_mismatch_tolerance"] <= 2.0e-2
    assert mismatches[-1] <= artifact["final_mismatch_tolerance"]
    assert artifact["converged"] is True

    probes = artifact["invalid_probes"]
    assert [probe["id"] for probe in probes] == ["non-monotonic-profile", "mesh-budget-exceeded"]
    assert all(probe["status"] == "typed_rejected" for probe in probes)
    assert all(probe["publication"] == "none" for probe in probes)
    assert "NonMonotonicProfile" in probes[0]["typed_error"]
    assert "MeshBudgetExceeded" in probes[1]["typed_error"]

    canonical_output = artifact["canonical_output"]
    assert isinstance(canonical_output, str) and canonical_output
    assert hashlib.sha256(canonical_output.encode()).hexdigest() == artifact["output_sha256"]
    assert json.loads(canonical_output) == levels
    print(
        json.dumps(
            {
                "levels": len(levels),
                "minimum_improvement_factor": artifact["minimum_improvement_factor"],
                "final_relative_mean_velocity_mismatch": artifact["final_relative_mean_velocity_mismatch"],
                "status": "pass",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
