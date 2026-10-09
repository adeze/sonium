"""Hardware-independent tests for the B3 source-coordinate derivative gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


CHECKER_PATH = Path(__file__).with_name("check_b3_cuda.py")
SPEC = importlib.util.spec_from_file_location("check_b3_cuda", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def source_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {
        "matrix_parity_relative_error": 1.0e-8,
        "product_parity_relative_error": 1.0e-8,
        "inner_product_error": 1.0e-12,
        "cuda_solve_residual_relative_error": 1.0e-10,
        "cuda_adjoint_solve_residual_relative_error": 1.0e-10,
        "field_parity_relative_error": 1.0e-8,
        "rigid_sphere_cuda_cpu_field_relative_error": 1.0e-11,
        "rigid_sphere_mie_relative_error": 1.0e-2,
        "pulsating_sphere_cuda_cpu_field_relative_error": 1.0e-11,
        "pulsating_sphere_reference_relative_error": 1.0e-2,
        "pulsating_sphere_radiation_power_relative_error": 1.0e-2,
        "pulsating_sphere_mesh_convergence_improvement": 1.0e-2,
        "wavenumber_derivative_relative_error": 1.0e-5,
        "source_position_x_derivative_relative_error": 1.0e-8,
        "source_position_x_derivative_warm_relative_error": 1.0e-8,
        "source_position_x_derivative_retry_relative_error": 1.0e-8,
        "source_position_y_derivative_relative_error": 1.0e-8,
        "source_position_y_derivative_warm_relative_error": 1.0e-8,
        "source_position_y_derivative_retry_relative_error": 1.0e-8,
        "dofs": 3.0,
        "pulsating_sphere_panel_count": 12.0,
        "pulsating_sphere_radiation_power": 1.0,
        "source_position_x_derivative_active_pairs": 4.0,
        "source_position_x_derivative_initial_kernel_launches": 1.0,
        "source_position_x_derivative_warm_kernel_launches": 1.0,
        "source_position_x_derivative_initial_allocations": 0.0,
        "source_position_x_derivative_warm_allocations": 0.0,
        "source_position_x_derivative_initial_h2d_operations": 0.0,
        "source_position_x_derivative_warm_h2d_operations": 0.0,
        "source_position_x_derivative_initial_d2h_operations": 1.0,
        "source_position_x_derivative_warm_d2h_operations": 1.0,
        "source_position_x_derivative_failure_typed": 1.0,
        "source_position_x_derivative_output_preserved": 1.0,
        "source_position_x_derivative_teardown_live_bytes_before": 4096.0,
        "source_position_x_derivative_teardown_live_bytes_after": 0.0,
        "source_position_x_derivative_teardown_released_bytes": 4096.0,
        "source_position_y_derivative_active_pairs": 4.0,
        "source_position_y_derivative_initial_kernel_launches": 1.0,
        "source_position_y_derivative_warm_kernel_launches": 1.0,
        "source_position_y_derivative_initial_allocations": 0.0,
        "source_position_y_derivative_warm_allocations": 0.0,
        "source_position_y_derivative_initial_h2d_operations": 0.0,
        "source_position_y_derivative_warm_h2d_operations": 0.0,
        "source_position_y_derivative_initial_d2h_operations": 1.0,
        "source_position_y_derivative_warm_d2h_operations": 1.0,
        "source_position_y_derivative_failure_typed": 1.0,
        "source_position_y_derivative_output_preserved": 1.0,
        "source_position_y_derivative_teardown_live_bytes_before": 4096.0,
        "source_position_y_derivative_teardown_live_bytes_after": 0.0,
        "source_position_y_derivative_teardown_released_bytes": 4096.0,
    }
    return metrics


def test_checker_accepts_source_derivative_metrics() -> None:
    CHECKER.check_metrics({"metrics": source_metrics()})


def test_checker_rejects_source_derivative_warm_allocation() -> None:
    metrics = source_metrics()
    metrics["source_position_x_derivative_warm_allocations"] = 1.0
    with pytest.raises(SystemExit, match="allocation or H2D"):
        CHECKER.check_metrics({"metrics": metrics})


def test_checker_rejects_source_y_derivative_warm_h2d() -> None:
    metrics = source_metrics()
    metrics["source_position_y_derivative_warm_h2d_operations"] = 1.0
    with pytest.raises(SystemExit, match="allocation or H2D"):
        CHECKER.check_metrics({"metrics": metrics})


def test_checker_rejects_unvalidated_source_capability() -> None:
    environment = {
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
        "physical_gate.bem.3d_source_derivative": "unsupported",
        "physical_gate.bem.3d_source_y_derivative": "unsupported",
        "physical_gate.bem.2d_cylinder": "unsupported",
        "physical_gate.bem.axisymmetric_mie": "unsupported",
        "physical_gate.bem.3d_transmission": "unsupported",
        "physical_gate.bem.piston": "unsupported",
        "physical_gate.bem.duct": "unsupported",
        "physical_gate.bem.3d_geometry_derivative": "unsupported",
        "physical_gate.bem.3d_material_derivative": "unsupported",
        "vendor_baseline.cublas": "exercised",
        "vendor_baseline.cusparse": "exercised",
        "vendor_baseline.cufft": "not_applicable",
    }
    with pytest.raises(SystemExit, match="3d_source_derivative"):
        CHECKER.check_environment({"environment": environment})
