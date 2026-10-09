"""Focused, hardware-independent checks for the B2 warm contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


CHECKER_PATH = Path(__file__).with_name("check_b2_cuda.py")
SPEC = importlib.util.spec_from_file_location("check_b2_cuda", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def warm_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix in ("mlfmm", "hmatrix"):
        metrics.update(
            {
                f"{prefix}_warm_replays": 2.0,
                f"{prefix}_warm_rhs_updates": 2.0,
                f"{prefix}_warm_structural_allocations": 0.0,
                f"{prefix}_warm_structural_h2d_operations": 0.0,
                f"{prefix}_warm_rhs_d2d_operations": 2.0,
                f"{prefix}_warm_d2h_publications": 2.0,
                f"{prefix}_warm_max_action_relative_error": 1.0e-15,
                f"{prefix}_warm_failure_recovery": 1.0,
                f"{prefix}_warm_failure_output_preserved": 1.0,
            }
        )
    metrics.update(
        {
            "cuda_teardown_live_bytes_before": 1000.0,
            "cuda_teardown_live_bytes_after": 256.0,
            "cuda_teardown_released_bytes": 744.0,
        }
    )
    return metrics


def test_warm_contract_accepts_device_rhs_update_and_teardown() -> None:
    CHECKER.check_warm_metrics(warm_metrics(), "test.metrics")


def test_warm_contract_rejects_structural_h2d() -> None:
    metrics = warm_metrics()
    metrics["hmatrix_warm_structural_h2d_operations"] = 1.0
    with pytest.raises(SystemExit, match="allocated or staged structure"):
        CHECKER.check_warm_metrics(metrics, "test.metrics")
