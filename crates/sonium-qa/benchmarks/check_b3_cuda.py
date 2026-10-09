#!/usr/bin/env python3
"""Strict checker for native CUDA B3 forward/adjoint physical evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "bem.cuda-capability-and-assembly-matrix.v1"
SCHEMA = "sonium-compute-cuda-bem-evidence/v1"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B3 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} provenance is missing")
    return value


def verify_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        fail(f"checksum sidecar has no valid digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}")


def check_thresholds(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    thresholds = artifact.get("thresholds")
    directions = artifact.get("threshold_directions")
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict) or not isinstance(directions, dict):
        fail("metrics, thresholds, and threshold directions are required")
    if set(thresholds) != set(directions) or not set(thresholds).issubset(metrics):
        fail("thresholds, directions, and measured metrics do not have identical keys")
    for key, threshold in thresholds.items():
        measured = finite(metrics[key], f"metrics.{key}")
        bound = finite(threshold, f"thresholds.{key}")
        direction = directions[key]
        if direction == "maximum" and measured > bound:
            fail(f"{key}={measured:.6e} exceeds maximum {bound:.6e}")
        if direction == "minimum" and measured < bound:
            fail(f"{key}={measured:.6e} is below minimum {bound:.6e}")
        if direction not in {"maximum", "minimum"}:
            fail(f"{key} has invalid threshold direction {direction!r}")


def check_environment(artifact: dict) -> None:
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        fail("capability/physical environment matrix is missing")
    required = {
        "capability.bem.dimension.2d": "unsupported",
        "capability.bem.dimension.axisymmetric": "unsupported",
        "capability.bem.dimension.3d": "supported",
        "capability.bem.element.tri3.constant.c64": "native",
        "capability.bem.element.tri3.higher_order.c64": "unsupported",
        "capability.bem.element.quad.constant.c64": "unsupported",
        "capability.bem.element.quad.higher_order.c64": "unsupported",
        "capability.bem.source_position_x_derivative.c64": "native",
        "capability.bem.source_position_y_derivative.c64": "native",
        "physical_gate.bem.3d_operator": "validated",
        "physical_gate.bem.3d_rigid_sphere_mie": "validated",
        "physical_gate.bem.3d_radiation": "validated",
        "physical_gate.bem.3d_mesh_convergence": "validated",
        "physical_gate.bem.3d_wavenumber_derivative": "validated",
        "physical_gate.bem.3d_source_derivative": "validated",
        "physical_gate.bem.3d_source_y_derivative": "validated",
        "physical_gate.bem.2d_cylinder": "unsupported",
        "physical_gate.bem.axisymmetric_mie": "unsupported",
        "physical_gate.bem.3d_transmission": "unsupported",
        "physical_gate.bem.piston": "unsupported",
        "physical_gate.bem.duct": "unsupported",
        "physical_gate.bem.3d_geometry_derivative": "unsupported",
        "physical_gate.bem.3d_material_derivative": "unsupported",
    }
    for key, expected in required.items():
        if environment.get(key) != expected:
            fail(f"{key} must be {expected!r}, got {environment.get(key)!r}")
    for key in (
        "vendor_baseline.cublas",
        "vendor_baseline.cusparse",
        "vendor_baseline.cufft",
    ):
        nonempty(environment.get(key), f"environment.{key}")


def check_hardware(artifact: dict) -> None:
    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance is missing")
    for key in (
        "device",
        "driver",
        "toolkit",
        "topology",
        "cpu_model",
        "microcode",
        "compiler",
        "target_features",
        "runtime_dispatch",
        "thread_policy",
        "affinity_policy",
        "numa_policy",
        "deterministic_mode",
    ):
        nonempty(hardware.get(key), f"hardware.{key}")
    if "nvidia" not in hardware["device"].lower():
        fail("hardware.device is not an NVIDIA device")
    if hardware["os"] != "linux-x86_64":
        fail("hardware.os is not native Linux x86_64")
    if hardware["target_features"] != "cuda" or hardware["runtime_dispatch"] != "cuda-native":
        fail("CUDA target/dispatch provenance is inconsistent")


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime evidence is missing")
    for key in ("kernel_launches", "library_operations", "synchronization_calls"):
        value = runtime.get(key)
        if not isinstance(value, int) or value <= 0:
            fail(f"runtime.{key} must be a positive integer")
    transfers = runtime.get("transfers")
    if not isinstance(transfers, dict):
        fail("runtime transfer evidence is missing")
    for key in ("h2d_operations", "d2h_operations"):
        value = transfers.get(key)
        if not isinstance(value, int) or value <= 0:
            fail(f"runtime.transfers.{key} must be a positive integer")
    for key in ("allocation_high_water_bytes", "live_allocation_bytes"):
        value = runtime.get(key)
        if not isinstance(value, int) or value <= 0:
            fail(f"runtime.{key} must be a positive integer")
    metrics = artifact["metrics"]
    for metric, runtime_key in (
        ("cuda_kernel_launches", "kernel_launches"),
        ("cuda_library_operations", "library_operations"),
        ("cuda_synchronization_calls", "synchronization_calls"),
        ("cuda_allocation_high_water_bytes", "allocation_high_water_bytes"),
        ("cuda_live_allocation_bytes", "live_allocation_bytes"),
        ("cuda_h2d_operations", "h2d_operations"),
        ("cuda_d2h_operations", "d2h_operations"),
    ):
        if finite(metrics.get(metric), f"metrics.{metric}") != float(
            transfers.get(runtime_key, runtime.get(runtime_key))
        ):
            fail(f"metrics.{metric} disagrees with runtime counters")


def check_metrics(artifact: dict) -> None:
    metrics = artifact["metrics"]
    required = (
        "matrix_parity_relative_error",
        "product_parity_relative_error",
        "inner_product_error",
        "cuda_solve_residual_relative_error",
        "cuda_adjoint_solve_residual_relative_error",
        "field_parity_relative_error",
        "rigid_sphere_cuda_cpu_field_relative_error",
        "rigid_sphere_mie_relative_error",
        "pulsating_sphere_cuda_cpu_field_relative_error",
        "pulsating_sphere_reference_relative_error",
        "pulsating_sphere_radiation_power_relative_error",
        "pulsating_sphere_mesh_convergence_improvement",
        "wavenumber_derivative_relative_error",
        "source_position_x_derivative_relative_error",
        "source_position_x_derivative_warm_relative_error",
        "source_position_x_derivative_retry_relative_error",
        "source_position_y_derivative_relative_error",
        "source_position_y_derivative_warm_relative_error",
        "source_position_y_derivative_retry_relative_error",
    )
    for key in required:
        finite(metrics.get(key), f"metrics.{key}")
    if finite(metrics["dofs"], "metrics.dofs") <= 0 or finite(metrics["pulsating_sphere_panel_count"], "metrics.pulsating_sphere_panel_count") <= 0:
        fail("physical workload sizes must be positive")
    if finite(metrics["pulsating_sphere_radiation_power"], "metrics.pulsating_sphere_radiation_power") <= 0:
        fail("pulsating-sphere radiation power must be positive")
    if finite(metrics["source_position_x_derivative_active_pairs"], "metrics.source_position_x_derivative_active_pairs") <= 0:
        fail("source-position derivative must cover at least one resident pair")
    for key in (
        "source_position_x_derivative_initial_kernel_launches",
        "source_position_x_derivative_warm_kernel_launches",
        "source_position_x_derivative_initial_d2h_operations",
        "source_position_x_derivative_warm_d2h_operations",
    ):
        if finite(metrics[key], f"metrics.{key}") <= 0:
            fail(f"{key} does not prove native derivative work/publication")
    for key in (
        "source_position_x_derivative_initial_allocations",
        "source_position_x_derivative_warm_allocations",
        "source_position_x_derivative_initial_h2d_operations",
        "source_position_x_derivative_warm_h2d_operations",
    ):
        if finite(metrics[key], f"metrics.{key}") != 0:
            fail(f"{key} records an unexpected resident-workspace allocation or H2D")
    if finite(metrics["source_position_x_derivative_failure_typed"], "metrics.source_position_x_derivative_failure_typed") != 1:
        fail("source-position derivative launch failure is not typed")
    if finite(metrics["source_position_x_derivative_output_preserved"], "metrics.source_position_x_derivative_output_preserved") != 1:
        fail("source-position derivative failure did not preserve output")
    before = finite(metrics["source_position_x_derivative_teardown_live_bytes_before"], "metrics.source_position_x_derivative_teardown_live_bytes_before")
    after = finite(metrics["source_position_x_derivative_teardown_live_bytes_after"], "metrics.source_position_x_derivative_teardown_live_bytes_after")
    released = finite(metrics["source_position_x_derivative_teardown_released_bytes"], "metrics.source_position_x_derivative_teardown_released_bytes")
    if before <= after or released <= 0 or before - after != released:
        fail("source-position derivative workspace teardown was not exact")
    if finite(metrics["source_position_y_derivative_active_pairs"], "metrics.source_position_y_derivative_active_pairs") <= 0:
        fail("source-position y derivative must cover at least one resident pair")
    for key in (
        "source_position_y_derivative_initial_kernel_launches",
        "source_position_y_derivative_warm_kernel_launches",
        "source_position_y_derivative_initial_d2h_operations",
        "source_position_y_derivative_warm_d2h_operations",
    ):
        if finite(metrics[key], f"metrics.{key}") <= 0:
            fail(f"{key} does not prove native derivative work/publication")
    for key in (
        "source_position_y_derivative_initial_allocations",
        "source_position_y_derivative_warm_allocations",
        "source_position_y_derivative_initial_h2d_operations",
        "source_position_y_derivative_warm_h2d_operations",
    ):
        if finite(metrics[key], f"metrics.{key}") != 0:
            fail(f"{key} records an unexpected resident-workspace allocation or H2D")
    if finite(metrics["source_position_y_derivative_failure_typed"], "metrics.source_position_y_derivative_failure_typed") != 1:
        fail("source-position y derivative launch failure is not typed")
    if finite(metrics["source_position_y_derivative_output_preserved"], "metrics.source_position_y_derivative_output_preserved") != 1:
        fail("source-position y derivative failure did not preserve output")
    before = finite(metrics["source_position_y_derivative_teardown_live_bytes_before"], "metrics.source_position_y_derivative_teardown_live_bytes_before")
    after = finite(metrics["source_position_y_derivative_teardown_live_bytes_after"], "metrics.source_position_y_derivative_teardown_live_bytes_after")
    released = finite(metrics["source_position_y_derivative_teardown_released_bytes"], "metrics.source_position_y_derivative_teardown_released_bytes")
    if before <= after or released <= 0 or before - after != released:
        fail("source-position y derivative workspace teardown was not exact")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not a JSON object")
    if artifact.get("schema") != SCHEMA or artifact.get("status") != "passed":
        fail("schema/status is not a passed B3 artifact")
    if artifact.get("case_id") != CASE_ID:
        fail("case identifier is not the B3 producer")
    check_hardware(artifact)
    check_environment(artifact)
    check_metrics(artifact)
    check_thresholds(artifact)
    check_runtime(artifact)
    finite(artifact.get("workload_elapsed_ms"), "workload_elapsed_ms")
    print(f"B3 native CUDA artifact passed strict checks: {path}")


if __name__ == "__main__":
    main()
