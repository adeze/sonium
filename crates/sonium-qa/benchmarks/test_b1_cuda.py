"""Focused, hardware-independent checks for the B1 production TBEM row."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


CHECKER_PATH = Path(__file__).with_name("check_b1_cuda.py")
SPEC = importlib.util.spec_from_file_location("check_b1_cuda", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def production_row() -> dict[str, object]:
    return {
        "id": "production_tbem.resident-tri3-regular-near-singular",
        "status": "passed",
        "initial_relative_error": 1.0e-6,
        "warm_relative_error": 2.0e-6,
        "retry_relative_error": 2.0e-6,
        "tolerance": 1.0e-3,
        "initial_kernel_launches": 2,
        "initial_h2d_operations": 0,
        "initial_d2h_operations": 2,
        "warm_kernel_launches": 2,
        "warm_structural_allocations": 0,
        "warm_steady_state_h2d_operations": 0,
        "warm_publication_d2h_operations": 2,
        "failure_typed": True,
        "teardown_live_bytes_before": 4096,
        "teardown_live_bytes_after": 0,
        "teardown_released_bytes": 4096,
    }


def real_f64_row() -> dict[str, object]:
    return {
        "id": "production_tbem.real-f64-laplace-regular",
        "status": "passed",
        "scalar": "real f64 Laplace channels",
        "active_pairs": 2,
        "initial_relative_error": 1.0e-12,
        "warm_relative_error": 1.0e-12,
        "retry_relative_error": 1.0e-12,
        "tolerance": 1.0e-3,
        "initial_kernel_launches": 1,
        "initial_allocations": 0,
        "initial_h2d_operations": 0,
        "initial_d2h_operations": 1,
        "warm_kernel_launches": 1,
        "warm_structural_allocations": 0,
        "warm_steady_state_h2d_operations": 0,
        "warm_publication_d2h_operations": 1,
        "failure_typed": True,
        "teardown_live_bytes_before": 4096,
        "teardown_live_bytes_after": 0,
        "teardown_released_bytes": 4096,
    }


def transmission_row() -> dict[str, object]:
    return {
        "id": "production_transmission.c64.constant-tri3",
        "status": "passed",
        "formulation": "constant-panel Muller two-medium transmission pair/field cell",
        "scalar": "Complex64",
        "element": "Tri3-constant",
        "pair_count": 3,
        "pair_tail": 3,
        "field_points": 5,
        "field_tail": 5,
        "tolerance": 3.0e-12,
        "pair_initial_relative_error": 1.0e-13,
        "pair_warm_relative_error": 1.0e-13,
        "pair_retry_relative_error": 1.0e-13,
        "field_exterior_initial_relative_error": 1.0e-13,
        "field_interior_initial_relative_error": 1.0e-13,
        "field_exterior_warm_relative_error": 1.0e-13,
        "field_interior_warm_relative_error": 1.0e-13,
        "pair_initial_kernel_launches": 2,
        "pair_initial_allocations": 2,
        "pair_initial_h2d_operations": 0,
        "pair_initial_d2h_operations": 2,
        "pair_warm_kernel_launches": 2,
        "pair_warm_structural_allocations": 0,
        "pair_warm_steady_state_h2d_operations": 0,
        "pair_warm_publication_d2h_operations": 2,
        "pair_warm_d2d_operations": 2,
        "field_initial_kernel_launches": 2,
        "field_initial_allocations": 2,
        "field_initial_h2d_operations": 0,
        "field_initial_d2h_operations": 2,
        "field_warm_kernel_launches": 2,
        "field_warm_structural_allocations": 0,
        "field_warm_steady_state_h2d_operations": 0,
        "field_warm_publication_d2h_operations": 2,
        "field_warm_d2d_operations": 2,
        "failure_typed": True,
        "failure_no_publication": True,
        "failure_kernel_launches": 0,
        "failure_d2d_operations": 0,
        "malformed_pair_typed": True,
        "malformed_field_typed": True,
        "empty_succeeds": True,
        "empty_kernel_launches": 0,
        "empty_h2d_operations": 0,
        "empty_d2h_operations": 0,
        "teardown_live_bytes_before": 4096,
        "teardown_live_bytes_after": 0,
        "teardown_released_bytes": 4096,
    }


def valid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_id in (
        "regular_pair.c32",
        "regular_pair.c64",
        "regular_pair.c64.source_x_derivative",
        "quadrature.c64.bulk-singular-edge",
        "field.c64.constant-tri3",
    ):
        row: dict[str, object] = {
            "id": row_id,
            "status": "passed",
            "relative_error": 1.0e-8,
            "tolerance": 1.0e-3,
            "kernel_launches": 1,
            "d2h_operations": 1,
        }
        if row_id.startswith(("quadrature", "field")):
            row["steady_state_h2d_operations"] = 0
        if row_id == "regular_pair.c64.source_x_derivative":
            row["steady_state_h2d_operations"] = 0
            row["failure_typed"] = True
            row["output_preserved"] = True
        rows.append(row)
    rows.append(production_row())
    rows.append(real_f64_row())
    rows.append(transmission_row())
    rows.extend(
        [
            {
                "id": "failure.regular_pair.launch",
                "status": "passed",
                "typed_error": True,
                "output_preserved": True,
                "cleanup": True,
            },
            {
                "id": "failure.field.launch",
                "status": "passed",
                "typed_error": True,
                "output_preserved": True,
                "cleanup": True,
            },
            {
                "id": "admission.quadrature.invalid_range",
                "status": "passed",
                "typed_error": True,
            },
            {
                "id": "admission.field.quad4-and-on-surface",
                "status": "unsupported",
                "reason": "constant Tri3 off-surface only",
            },
        ]
    )
    return rows


def test_checker_accepts_production_resident_batch() -> None:
    CHECKER.check_rows({"rows": valid_rows()})


def test_checker_rejects_production_warm_host_transfer() -> None:
    rows = valid_rows()
    next(row for row in rows if row["id"] == "production_tbem.resident-tri3-regular-near-singular")[
        "warm_steady_state_h2d_operations"
    ] = 1
    with pytest.raises(SystemExit, match="allocated or staged resident geometry"):
        CHECKER.check_rows({"rows": rows})


def test_checker_rejects_complex_row_relabelled_as_real_f64() -> None:
    rows = valid_rows()
    scalar = next(row for row in rows if row["id"] == "production_tbem.real-f64-laplace-regular")
    scalar["scalar"] = "f64 geometry / Complex64 channels"
    with pytest.raises(SystemExit, match="scalar channel declaration"):
        CHECKER.check_rows({"rows": rows})


def test_checker_rejects_transmission_without_device_publication() -> None:
    rows = valid_rows()
    transmission = next(
        row for row in rows if row["id"] == "production_transmission.c64.constant-tri3"
    )
    transmission["pair_warm_d2d_operations"] = 0
    with pytest.raises(SystemExit, match="device publication"):
        CHECKER.check_rows({"rows": rows})
