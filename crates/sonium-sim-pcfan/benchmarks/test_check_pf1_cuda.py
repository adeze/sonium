"""Focused, hardware-independent checks for the PF1 production Picard cell."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


CHECKER_PATH = Path(__file__).with_name("check_pf1_cuda.py")
SPEC = importlib.util.spec_from_file_location("check_pf1_cuda", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def production_picard_row() -> dict[str, object]:
    return {
        "status": "passed",
        "backend": "Cuda",
        "execution_mode": "DeviceResident",
        "requested_backend": "Cuda",
        "actual_backend": "Cuda",
        "assembly_residency": "Host",
        "host_staging": False,
        "operator": "production PcFanCfdFlowSolution.picard_linearization.state_jacobian",
        "matrix_provenance": "fem:navier-stokes:picard-oseen:v1",
        "dimension": 3,
        "nnz": 5,
        "empty_rows": 0,
        "tail_lanes": 3,
        "repeated_actions": 3,
        "device_resident": True,
        "operator_retained": True,
        "state_retained": True,
        "cpu_reference": {"output": [1.0, 2.0, 3.0]},
        "cuda_observation": {"output": [1.0, 2.0, 3.0], "max_abs_error": 1.0e-12},
        "warm_counters": {
            "device_allocations": 0,
            "staging_allocations": 0,
            "h2d_operations": 0,
            "d2h_operations": 0,
            "d2d_operations": 0,
            "synchronizations": 1,
            "kernel_and_library_operations": 3,
            "live_device_bytes": 512,
            "live_staging_bytes": 0,
        },
        "admission": {
            "malformed_csr": {
                "status": "typed_rejection",
                "typed_error": "InvalidArgument",
                "partial_result_published": False,
            },
            "empty_csr": {"status": "accepted_noop", "partial_result_published": False},
            "tail_case": {"status": "passed", "matrix_action_includes_partial_tail": True},
        },
        "launch_failure": {
            "status": "typed",
            "operation": "persistent_csr_apply",
            "typed_error": "Launch",
            "context_recovery": "synchronize_before_reuse",
            "recovery_succeeded": True,
            "output_preserved": True,
            "output_preservation_residual_max_abs": 1.0e-12,
            "retry_succeeded": True,
            "retry_residual_max_abs": 1.0e-12,
            "partial_result_published": False,
        },
        "resources_released_on_teardown": True,
        "live_device_bytes_after_teardown": 0,
        "live_staging_bytes_after_teardown": 0,
        "pool_bytes_in_use_after_teardown": 0,
        "unsupported_high_level": [
            "on_device_pcfan_fem_assembly",
            "retained_pcfan_flow_state_across_full_bdf",
            "retained_pcfan_aeroacoustic_coupling",
        ],
    }


def test_checker_accepts_production_picard_resident_cell() -> None:
    CHECKER.check_production_picard(production_picard_row())


def test_checker_rejects_production_picard_warm_host_transfer() -> None:
    row = production_picard_row()
    row["warm_counters"]["h2d_operations"] = 1  # type: ignore[index]
    with pytest.raises(SystemExit, match="allocated or transferred through host"):
        CHECKER.check_production_picard(row)


def test_checker_rejects_non_fem_picard_provenance() -> None:
    row = production_picard_row()
    row["matrix_provenance"] = "fixture:synthetic"
    with pytest.raises(SystemExit, match="FEM Oseen path"):
        CHECKER.check_production_picard(row)
