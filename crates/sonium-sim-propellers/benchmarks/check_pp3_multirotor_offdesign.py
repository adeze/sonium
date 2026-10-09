#!/usr/bin/env python3
"""Strict checker for PP3's production multi-rotor/off-design packet."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMA = "sonium.propellers.pp3-multirotor-offdesign-evidence.v1"
EXPECTED_INPUT_FRAGMENT = "rpm=12000,V=10,rho=1.204),(0,0,0.12,axis=(0,0,1),rpm=9000,V=8,rho=0.98"
CONVERGENCE_TOLERANCE = 1.0e-3
RADIUS_M = 0.0635
RADIAL_SAMPLES = 3
AZIMUTHAL_SAMPLES = 16
WAKE_RADIUS_GROWTH = 0.25


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left: float, right: float, *, rel: float = 1.0e-10, absolute: float = 1.0e-12) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=absolute)


def momentum(thrust: float, radius: float, density: float, freestream: float) -> float:
    area = math.pi * radius * radius
    loading = thrust / (density * area)
    root = math.sqrt(freestream * freestream + 2.0 * loading)
    numerator = root - freestream
    return 0.5 * numerator if numerator > 0.0 else loading / (root + freestream)


def aligned_wake(thrust: float, density: float, freestream: float, distance: float) -> float:
    source_induced = momentum(thrust, RADIUS_M, density, freestream)
    wake_radius = RADIUS_M * (1.0 + WAKE_RADIUS_GROWTH * distance / RADIUS_M)
    weighted = 0.0
    weight_sum = 0.0
    for radial_index in range(RADIAL_SAMPLES):
        radial_fraction = (radial_index + 0.5) / RADIAL_SAMPLES
        weight = 2.0 * radial_fraction / RADIAL_SAMPLES
        gaussian = math.exp(-((RADIUS_M * radial_fraction) ** 2) / (wake_radius * wake_radius))
        for _ in range(AZIMUTHAL_SAMPLES):
            weighted += weight * 2.0 * source_induced * gaussian
            weight_sum += weight
    return weighted / weight_sum


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    artifact_path = Path(sys.argv[1])
    artifact = json.loads(artifact_path.read_text())
    assert artifact["schema"] == SCHEMA
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64/AVX512 waived"
    assert artifact["claim_class"] == "production load-coupled multi-rotor BEM off-design convergence"
    assert "evaluate_multi_rotor_bem" in artifact["production_path"]["evaluator"]
    assert artifact["production_path"]["rotor_count"] == 2
    assert close(artifact["production_path"]["tip_radius_m"], RADIUS_M)
    canonical_input = artifact["canonical_input"]
    assert isinstance(canonical_input, str) and EXPECTED_INPUT_FRAGMENT in canonical_input
    assert hashlib.sha256(canonical_input.encode()).hexdigest() == artifact["canonical_input_sha256"]

    rotors = artifact["rotors"]
    assert len(rotors) == 2
    expected_points = [
        (12000.0, 10.0, 1.204, [0.0, 0.0, 0.0]),
        (9000.0, 8.0, 0.98, [0.0, 0.0, 0.12]),
    ]
    for row, (rpm, freestream, density, center) in zip(rotors, expected_points):
        assert row["center_m"] == center
        assert row["axis"] == [0.0, 0.0, 1.0]
        point = row["operating_point"]
        assert close(point["rpm"], rpm) and close(point["freestream_m_s"], freestream)
        assert close(point["density_kg_m3"], density)
        for key in (
            "external_induced_velocity_m_s",
            "isolated_induced_velocity_m_s",
            "effective_axial_velocity_m_s",
            "relative_load_change",
            "loads",
        ):
            if key == "loads":
                continue
            assert finite(row[key]), key
        assert row["effective_axial_velocity_m_s"] > 0.0
        assert row["relative_load_change"] >= 0.0
        loads = row["loads"]
        assert all(finite(loads[field]) and loads[field] > 0.0 for field in ("thrust_n", "torque_nm", "shaft_power_w"))

    pairwise = artifact["pairwise_induced_velocity_m_s"]
    assert pairwise == [[0.0, 0.0], pairwise[1]]
    assert close(pairwise[0][0], 0.0, absolute=0.0)
    assert close(pairwise[0][1], 0.0, absolute=0.0)
    assert finite(pairwise[1][0]) and pairwise[1][0] > 0.0
    source_thrust = rotors[0]["loads"]["thrust_n"]
    expected_pair = aligned_wake(source_thrust, 1.204, 10.0, 0.12)
    oracle = artifact["independent_oracle"]
    assert "momentum" in oracle["formula"]
    assert close(oracle["source_rotor_induced_velocity_m_s"], momentum(source_thrust, RADIUS_M, 1.204, 10.0))
    assert close(oracle["downstream_pairwise_velocity_m_s"], expected_pair)
    assert close(pairwise[1][0], expected_pair)
    assert close(oracle["production_pairwise_relative_error"], abs(pairwise[1][0] - expected_pair) / max(abs(expected_pair), 1.0))
    assert oracle["production_pairwise_relative_error"] <= 1.0e-10

    convergence = artifact["convergence"]
    assert convergence["converged"] is True
    assert convergence["iterations"] > 0
    assert finite(convergence["maximum_relative_load_change"])
    assert convergence["maximum_relative_load_change"] <= CONVERGENCE_TOLERANCE
    assert convergence["tolerance"] == CONVERGENCE_TOLERANCE
    assert convergence["pair_evaluation_count"] >= 2
    assert convergence["bem_evaluation_count"] >= 4
    assert convergence["disk_sample_count_per_pair"] == RADIAL_SAMPLES * AZIMUTHAL_SAMPLES
    assert artifact["deterministic_repeat"] is True

    probes = artifact["invalid_probes"]
    assert [probe["id"] for probe in probes] == ["zero-axis", "rotor-budget", "pair-budget", "bem-budget"]
    assert all(probe["status"] == "typed_rejected" for probe in probes)
    assert all(probe["publication"] == "none" for probe in probes)
    assert "InvalidRotor" in probes[0]["typed_error"]
    assert "TooManyRotors" in probes[1]["typed_error"]
    assert "PairEvaluationBudgetExceeded" in probes[2]["typed_error"]
    assert "BemEvaluationBudgetExceeded" in probes[3]["typed_error"]

    canonical_output = artifact["canonical_output"]
    assert hashlib.sha256(canonical_output.encode()).hexdigest() == artifact["output_sha256"]
    output = json.loads(canonical_output)
    assert len(output["rotors"]) == 2
    assert output["pairwise_induced_velocity_m_s"] == pairwise
    assert output["iterations"] == convergence["iterations"]
    assert "CUDA" in " ".join(artifact["unsupported_or_unexercised"])
    print(json.dumps({
        "status": "pass",
        "rotors": 2,
        "iterations": convergence["iterations"],
        "pairwise_oracle_error": oracle["production_pairwise_relative_error"],
        "maximum_relative_load_change": convergence["maximum_relative_load_change"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
