#!/usr/bin/env python3
"""Strict checks for the native PP3 compact dipole direction/phase artifact."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


EXPECTED_ANGLES = [0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0, 5.0 * math.pi / 6.0, math.pi]
EXPECTED_DISTANCES = [1.0, 2.0, 4.0]
EXPECTED_AZIMUTHS = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
RELATIVE_TOLERANCE = 1.0e-12
PHASE_TOLERANCE = 1.0e-12


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left: float, right: float, *, rel: float = 1.0e-11, absolute: float = 1.0e-14) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=absolute)


def wrapped_phase_residual(observed: float, expected: float) -> float:
    return (observed - expected + math.pi) % (2.0 * math.pi) - math.pi


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text())

    assert artifact["schema"] == "sonium.propellers.observer-direction-evidence.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64 waived"
    assert artifact["claim_class"] == "analytic compact rotating-dipole observer direction and phase reference"
    assert "sin(theta)" in artifact["reference"]
    assert "exp(-i(k r + phi))" in artifact["reference"]
    canonical_input = artifact["canonical_input"]
    assert isinstance(canonical_input, str) and canonical_input
    assert hashlib.sha256(canonical_input.encode()).hexdigest() == artifact["canonical_input_sha256"]
    assert artifact["dimensions"] == [
        "polar_angle_rad",
        "observer_distance_m",
        "azimuth_angle_rad",
        "complex_pressure_pa",
        "sphere_integrated_power",
    ]

    frequency = artifact["frequency_hz"]
    reference_distance = artifact["reference_distance_m"]
    equatorial = artifact["equatorial_peak_pressure_pa"]
    sound_speed = artifact["speed_of_sound_m_s"]
    for value in (frequency, reference_distance, equatorial, sound_speed):
        assert finite(value) and value > 0.0
    wavenumber = 2.0 * math.pi * frequency / sound_speed

    angular = artifact["angular_levels"]
    assert len(angular) == len(EXPECTED_ANGLES)
    for level, angle in zip(angular, EXPECTED_ANGLES):
        assert close(level["polar_angle_rad"], angle)
        expected = equatorial * abs(math.sin(angle))
        for key in ("expected_pressure_pa", "observed_pressure_pa", "relative_error"):
            assert finite(level[key])
        assert close(level["expected_pressure_pa"], expected)
        assert close(level["observed_pressure_pa"], expected)
        assert level["relative_error"] <= RELATIVE_TOLERANCE
    assert artifact["max_angular_relative_error"] <= RELATIVE_TOLERANCE

    all_phase_levels = artifact["propagation_levels"] + artifact["azimuth_levels"]
    for level in all_phase_levels:
        for key in (
            "observer_distance_m",
            "polar_angle_rad",
            "azimuth_angle_rad",
            "expected_real_pa",
            "observed_real_pa",
            "expected_imaginary_pa",
            "observed_imaginary_pa",
            "expected_phase_rad",
            "observed_phase_rad",
            "phase_residual_rad",
            "relative_complex_error",
            "scaled_magnitude_pa_m",
        ):
            assert finite(level[key]), f"non-finite phase field {key}"
        expected_phase = -(wavenumber * level["observer_distance_m"] + level["azimuth_angle_rad"])
        amplitude = equatorial * reference_distance / level["observer_distance_m"] * math.sin(level["polar_angle_rad"])
        assert close(level["expected_phase_rad"], expected_phase)
        assert close(level["expected_real_pa"], amplitude * math.cos(expected_phase))
        assert close(level["expected_imaginary_pa"], amplitude * math.sin(expected_phase))
        assert close(level["observed_real_pa"], level["expected_real_pa"], rel=1.0e-10, absolute=1.0e-15)
        assert close(level["observed_imaginary_pa"], level["expected_imaginary_pa"], rel=1.0e-10, absolute=1.0e-15)
        assert abs(wrapped_phase_residual(level["observed_phase_rad"], expected_phase)) <= PHASE_TOLERANCE
        assert abs(level["phase_residual_rad"]) <= PHASE_TOLERANCE
        assert level["relative_complex_error"] <= RELATIVE_TOLERANCE

    propagation = artifact["propagation_levels"]
    assert [level["observer_distance_m"] for level in propagation] == EXPECTED_DISTANCES
    assert [level["azimuth_angle_rad"] for level in propagation] == [math.pi / 4.0] * 3
    scaled = [level["scaled_magnitude_pa_m"] for level in propagation]
    assert max(abs(right - left) / abs(left) for left, right in zip(scaled, scaled[1:])) <= RELATIVE_TOLERANCE
    azimuth = artifact["azimuth_levels"]
    assert [level["azimuth_angle_rad"] for level in azimuth] == EXPECTED_AZIMUTHS
    assert [level["observer_distance_m"] for level in azimuth] == [2.0] * 4
    assert artifact["max_complex_relative_error"] <= RELATIVE_TOLERANCE
    assert artifact["max_phase_residual_rad"] <= PHASE_TOLERANCE
    assert artifact["max_scaled_magnitude_relative_change"] <= RELATIVE_TOLERANCE

    assert artifact["expected_power_ratio_to_isotropic"] == 2.0 / 3.0
    assert close(artifact["observed_power_ratio_to_isotropic"], 2.0 / 3.0, rel=1.0e-13, absolute=1.0e-15)
    assert artifact["tolerance"] == RELATIVE_TOLERANCE
    assert artifact["phase_tolerance_rad"] == PHASE_TOLERANCE
    assert artifact["converged"] is True

    probes = artifact["invalid_probes"]
    assert [probe["id"] for probe in probes] == [
        "polar-angle-outside-spherical-domain",
        "nonfinite-azimuth-angle",
    ]
    assert all(probe["status"] == "typed_rejected" for probe in probes)
    assert "InvalidPolarAngle" in probes[0]["typed_error"]
    assert "InvalidInput" in probes[1]["typed_error"]

    canonical_output = artifact["canonical_output"]
    assert isinstance(canonical_output, str) and canonical_output
    assert hashlib.sha256(canonical_output.encode()).hexdigest() == artifact["output_sha256"]
    decoded = json.loads(canonical_output)
    assert decoded == {
        "angular": angular,
        "propagation": propagation,
        "azimuth": azimuth,
    }
    print(
        json.dumps(
            {
                "angular_levels": len(angular),
                "propagation_levels": len(propagation),
                "azimuth_levels": len(azimuth),
                "max_complex_relative_error": artifact["max_complex_relative_error"],
                "max_phase_residual_rad": artifact["max_phase_residual_rad"],
                "power_ratio": artifact["observed_power_ratio_to_isotropic"],
                "status": "pass",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
