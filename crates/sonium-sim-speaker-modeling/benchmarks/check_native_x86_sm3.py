#!/usr/bin/env python3
"""Strict checker for the native Linux/x86_64 SM3 modeling artifact.

The producer is intentionally CPU-only.  This gate checks that the reported
measurements came from the native modeling replay and that a missing
cross-architecture lane remains a waiver rather than an implicit pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


DRIVER_MODEL_CPU_GOLDEN_SHA256_V1 = (
    "0411bd27041feef181df9468a5713993121dc904144dbdadb601584073d8f20f"
)


def finite(value: object, label: str) -> None:
    assert isinstance(value, (int, float)) and math.isfinite(value), (
        f"{label} must be a finite number"
    )


def complex_pair(value: object, label: str) -> None:
    assert isinstance(value, list) and len(value) == 2, f"{label} must be [re, im]"
    finite(value[0], f"{label}[0]")
    finite(value[1], f"{label}[1]")


def pair_difference(left: list[float], right: list[float]) -> list[float]:
    return [left[0] - right[0], left[1] - right[1]]


def pair_norm(value: list[float]) -> float:
    return math.hypot(value[0], value[1])


def pair_relative_error(left: list[float], right: list[float]) -> float:
    return pair_norm(pair_difference(left, right)) / max(
        pair_norm(left), pair_norm(right), float.fromhex("0x0.0000000000001p-1022")
    )


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), f"{path}: artifact must be an object"
    return value


def check(path: Path) -> dict:
    artifact = load(path)
    assert artifact["schema"] == "sonium.speaker-modeling.native-x86-sm3.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"].startswith("native Linux x86_64")

    producer = artifact["producer"]
    assert producer["schema_version"] == 1
    assert producer["architecture"] == "x86_64"
    assert producer["backend"] == "cpu"
    assert producer["dispatch_path"] == (
        "native-x86_64-cpu/scalar-modeling-and-native-fem-solvers"
    )
    assert producer["cpu_vendor"] and producer["cpu_model"]
    assert producer["compiler"].startswith("rustc ")
    features = producer["simd_features_available"]
    assert all(isinstance(features[key], bool) for key in ("sse2", "avx2", "fma"))
    assert producer["driver_artifact_sha256"] == DRIVER_MODEL_CPU_GOLDEN_SHA256_V1
    finite(producer["driver_derivative"][0], "driver_derivative[0]")
    finite(producer["driver_derivative"][1], "driver_derivative[1]")

    axis = producer["axisymmetric"]
    assert axis["field_dofs"] > 0
    finite(axis["field_dofs"], "axisymmetric.field_dofs")
    complex_pair(axis["integrated_current"], "axisymmetric.integrated_current")
    complex_pair(axis["impedance"], "axisymmetric.impedance")
    for key in ("input_power_w", "power_residual_w", "observation_spl_db"):
        finite(axis[key], f"axisymmetric.{key}")

    sector = producer["sector"]
    assert sector["field_dofs"] > 0 and sector["mode_field_dofs"] > 0
    finite(sector["field_dofs"], "sector.field_dofs")
    finite(sector["mode_field_dofs"], "sector.mode_field_dofs")
    complex_pair(sector["impedance"], "sector.impedance")
    for key in ("eddy_loss_w", "periodic_residual"):
        finite(sector[key], f"sector.{key}")

    monolithic = producer["monolithic"]
    assert monolithic["interior_field_dofs"] > 0
    assert monolithic["aperture_field_dofs"] > 0
    finite(monolithic["interior_field_dofs"], "monolithic.interior_field_dofs")
    finite(monolithic["aperture_field_dofs"], "monolithic.aperture_field_dofs")
    finite(monolithic["relative_residual"], "monolithic.relative_residual")
    finite(monolithic["interface_residual"], "monolithic.interface_residual")

    product = producer["product_coupled"]
    assert product["interior_field_dofs"] > 0
    assert product["exterior_field_dofs"] > 0
    for key in (
        "interior_field_dofs",
        "exterior_field_dofs",
        "relative_residual",
        "interior_residual",
        "exterior_residual",
    ):
        finite(product[key], f"product_coupled.{key}")
    assert all(
        product[key] < 1.0e-10
        for key in ("relative_residual", "interior_residual", "exterior_residual")
    )
    assert product["force_identity"] == (
        "P1 face-area integration of FEM rear and BEM front pressure; net = rear - front"
    )
    for key in (
        "rear_pressure_force_n",
        "front_pressure_force_n",
        "net_pressure_force_n",
        "rear_reference_force_n",
        "front_reference_force_n",
        "net_reference_force_n",
    ):
        complex_pair(product[key], f"product_coupled.{key}")
    finite(product["reference_relative_error"], "product_coupled.reference_relative_error")
    assert product["reference_relative_error"] <= 1.0e-12
    assert pair_relative_error(
        product["rear_pressure_force_n"], product["rear_reference_force_n"]
    ) <= 1.0e-12
    assert pair_relative_error(
        product["front_pressure_force_n"], product["front_reference_force_n"]
    ) <= 1.0e-12
    assert pair_relative_error(
        product["net_pressure_force_n"],
        pair_difference(product["rear_reference_force_n"], product["front_reference_force_n"]),
    ) <= 1.0e-12

    modal = product["modal"]
    assert modal["identity"] == (
        "rigid-wall pressure projection -> generalized force -> modal displacement -> surface velocity"
    )
    assert modal["mode_count"] == 1
    for key in ("mode_count", "frequency_hz", "natural_frequency_hz", "structural_residual"):
        finite(modal[key], f"product_coupled.modal.{key}")
    assert modal["structural_residual"] < 1.0e-10
    for key in (
        "pressure_projection_peak_pa",
        "generalized_force_n",
        "generalized_displacement_peak_m",
        "generalized_displacement_reference_peak_m",
        "surface_velocity_peak_m_per_s",
        "surface_velocity_reference_peak_m_per_s",
        "independent_response_displacement_peak_m",
        "independent_response_velocity_peak_m_per_s",
    ):
        complex_pair(modal[key], f"product_coupled.modal.{key}")
    for key in (
        "reference_relative_error",
        "surface_velocity_reference_relative_error",
    ):
        finite(modal[key], f"product_coupled.modal.{key}")
        assert modal[key] <= 1.0e-10
    assert pair_relative_error(
        modal["generalized_displacement_peak_m"],
        modal["generalized_displacement_reference_peak_m"],
    ) <= 1.0e-10
    assert pair_relative_error(
        modal["surface_velocity_peak_m_per_s"],
        modal["surface_velocity_reference_peak_m_per_s"],
    ) <= 1.0e-10
    assert pair_relative_error(
        modal["generalized_displacement_peak_m"],
        modal["independent_response_displacement_peak_m"],
    ) <= 1.0e-10
    assert pair_relative_error(
        modal["surface_velocity_peak_m_per_s"],
        modal["independent_response_velocity_peak_m_per_s"],
    ) <= 1.0e-10
    for key in (
        "independent_response_displacement_relative_error",
        "independent_response_velocity_relative_error",
    ):
        finite(modal[key], f"product_coupled.modal.{key}")
        assert modal[key] <= 1.0e-10
    assert modal["independent_response_checked"] is True

    replay = artifact["replay"]
    assert replay == {
        "status": "pass",
        "relative_tolerance": 1.0e-12,
        "residual_absolute_floor": 1.0e-15,
        "scalar_potential_gauge": "lowest admissible periodic master node",
        "meaningful_drift_rejected": True,
        "coupled_force_and_modal_observables": "repeated product solve plus independent FEM/BEM and modal-equation references",
        "product_state_replay_exact": True,
    }
    cross_architecture = artifact["cross_architecture"]
    assert cross_architecture["status"] == "waived"
    assert set(cross_architecture["targets"]) == {"aarch64", "windows"}
    assert "waiv" in cross_architecture["reason"]

    sidecar = path.parent / "checksums.sha256"
    assert sidecar.is_file(), f"missing checksum sidecar: {sidecar}"
    lines = [line.split() for line in sidecar.read_text().splitlines() if line.strip()]
    assert lines == [[hashlib.sha256(path.read_bytes()).hexdigest(), path.name]]
    return artifact


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    artifact = check(Path(sys.argv[1]))
    producer = artifact["producer"]
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "schema": artifact["schema"],
                "architecture": producer["architecture"],
                "sector_field_dofs": producer["sector"]["field_dofs"],
                "monolithic_residual": producer["monolithic"]["relative_residual"],
                "product_force_reference_error": producer["product_coupled"][
                    "reference_relative_error"
                ],
                "product_modal_reference_error": producer["product_coupled"]["modal"][
                    "reference_relative_error"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
