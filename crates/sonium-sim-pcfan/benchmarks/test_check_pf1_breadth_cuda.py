"""Focused hardware-independent tests for the PF1 BDF breadth checker."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


CHECKER_PATH = Path(__file__).with_name("check_pf1_breadth_cuda.py")
SPEC = importlib.util.spec_from_file_location("check_pf1_breadth_cuda", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def counters() -> dict[str, int]:
    return {
        "device_allocations": 0,
        "staging_allocations": 0,
        "h2d_operations": 0,
        "h2d_bytes": 0,
        "d2h_operations": 0,
        "d2h_bytes": 0,
        "synchronizations": 1,
        "kernel_and_library_operations": 3,
        "live_device_bytes": 1024,
        "live_staging_bytes": 0,
    }


def artifact() -> dict[str, object]:
    accepted_steps: list[dict[str, object]] = []
    replay_steps: list[dict[str, object]] = []
    for step in range(2):
        accepted_steps.append(
            {
                "accepted": True,
                "step": step,
                "time_s": float(step + 1),
                "bdf_order": 1 if step == 0 else 2,
                "state_dimension": 2,
                "state_finite": True,
                "accepted_state_step": step + 1,
                "accepted_state_time_s": float(step + 1),
                "matrix_dimension": 2,
                "matrix_nnz": 2,
                "matrix_provenance": "fem:navier-stokes:bdf:v1",
                "requested_backend": "Cpu",
                "actual_backend": "Cpu",
                "cpu_reference": {
                    "state_jacobian_apply": [1.0, 2.0],
                    "state_residual_linf": 1.0,
                    "production_residual_norm": 1.0e-4,
                    "mass_imbalance": 1.0e-3,
                    "energy_imbalance": 2.0e-3,
                    "kinetic_proxy_j": 1.0,
                    "finite": True,
                },
            }
        )
        replay_steps.append(
            {
                "step": step,
                "accepted": True,
                "matrix_dimension": 2,
                "matrix_nnz": 2,
                "repeated_actions": 3,
                "shape_matches_previous_step": step > 0,
                "cpu_reference": accepted_steps[-1]["cpu_reference"],
                "cuda_observation": {
                    "output": [1.0, 2.0],
                    "max_abs_error": 1.0e-12,
                    "finite": True,
                },
                "warm_counters": counters(),
                "device_resident": True,
                "host_staging": False,
                "operator_retained_for_accepted_step": True,
                "state_retained_for_accepted_step": True,
            }
        )
    return {
        "schema_version": 1,
        "case_id": "pcfan.bdf-cuda-breadth.v1",
        "hardware": {
            "device": "SM120",
            "driver": "native",
            "toolkit": "native",
            "os": "linux-x86_64",
            "compiler": "rustc",
            "target_features": "cuda",
            "runtime_dispatch": "cuda-native",
            "topology": "GPU0",
        },
        "cuda_device_count": 1,
        "production_cpu_bdf": {
            "status": "passed",
            "flow_model": "IncompressibleCfdFull",
            "mesh_cells": [2, 2, 2],
            "bdf_scheme": "BDF2_with_first_order_start",
            "accepted_steps": accepted_steps,
            "observation_residual": 1.0e-4,
            "observation_converged": True,
        },
        "cuda_accepted_step_replay": {
            "status": "passed",
            "backend": "Cuda",
            "execution_mode": "DeviceResidentAcceptedStepReplay",
            "steps": replay_steps,
            "max_abs_error": 1.0e-12,
            "device_resident": True,
            "host_staging": False,
            "resources_released_on_teardown": True,
        },
        "admission_and_recovery": {
            "malformed_csr": {
                "status": "typed_rejection",
                "typed": True,
                "typed_error": "InvalidArgument",
                "partial_result_published": False,
            },
            "empty_csr": {"status": "accepted_noop", "partial_result_published": False},
            "nonfinite_state": {
                "status": "typed_rejection",
                "typed_error": "NonFiniteAcceptedState",
                "partial_result_published": False,
                "retry_succeeded": True,
            },
            "solver_failure": {
                "status": "typed_rejection",
                "typed_error": "PcFanCfdTransientError::InvalidTimeStep",
                "partial_result_published": False,
                "retry_succeeded": True,
            },
            "launch_failure": {
                "status": "typed_failure_no_publication",
                "typed_error": "Launch",
                "operation": "persistent_csr_apply",
                "output_preserved": True,
                "preservation_error_max_abs": 1.0e-12,
                "recovery_succeeded": True,
                "retry_succeeded": True,
                "retry_error_max_abs": 1.0e-12,
                "partial_result_published": False,
            },
        },
        "reuse_validity": {
            "accepted_state_history_consumed": True,
            "accepted_state_and_residual_reused_within_step": True,
            "operator_values_reuploaded_when_state_changed": True,
            "full_device_resident_bdf_assembly": False,
            "retained_pcfan_flow_state_across_full_bdf": False,
            "retained_pcfan_aeroacoustic_coupling": False,
        },
        "acceptance": {
            "cpu_state_residual_mass_energy_oracle": "passed",
            "native_cuda_resident_replay": "passed",
            "malformed_nonfinite_solver_launch_no_publication": "passed",
            "retry_after_typed_failure": "passed",
        },
    }


def test_checker_accepts_accepted_bdf_replay() -> None:
    CHECKER.check(artifact())


def test_checker_rejects_warm_host_transfer() -> None:
    value = artifact()
    value["cuda_accepted_step_replay"]["steps"][0]["warm_counters"]["h2d_operations"] = 1  # type: ignore[index]
    with pytest.raises(SystemExit, match="warm allocation/host transfer"):
        CHECKER.check(value)


def test_checker_rejects_published_solver_failure() -> None:
    value = artifact()
    value["admission_and_recovery"]["solver_failure"]["partial_result_published"] = True  # type: ignore[index]
    with pytest.raises(SystemExit, match="published partial output"):
        CHECKER.check(value)


def test_checker_rejects_full_device_claim() -> None:
    value = copy.deepcopy(artifact())
    value["reuse_validity"]["full_device_resident_bdf_assembly"] = True  # type: ignore[index]
    with pytest.raises(SystemExit, match="unsupported full-device boundaries"):
        CHECKER.check(value)
