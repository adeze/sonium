#!/usr/bin/env python3
"""Strict checker for the bounded FEM1 local-element CUDA parity artifact."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMA = "sonium.fem.element-local-cuda-parity.v1"
CASE_ID = "fem1.element-local-cuda.v1"
TOLERANCE = 2.0e-12


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left: float, right: float, rel: float = 1.0e-10, absolute: float = 1.0e-14) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=absolute)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    artifact = json.loads(Path(sys.argv[1]).read_text())
    assert artifact["schema"] == SCHEMA
    assert artifact["status"] == "passed"
    assert artifact["case_id"] == CASE_ID
    assert artifact["scope"] == "native Linux x86_64 required CUDA SM120; Windows/Arm64/AVX512 waived"
    assert artifact["claim_class"] == "independent P1/Tri3 local Helmholtz tensor parity through resident CUDA CSR action"
    assert "area=0.5" in artifact["reference"]
    assert "consistent mass area/12" in artifact["reference"]
    canonical = artifact["canonical_input"]
    assert isinstance(canonical, str) and canonical
    assert hashlib.sha256(canonical.encode()).hexdigest() == artifact["canonical_input_sha256"]
    assert artifact["elements"] == 2
    assert artifact["local_dofs"] == 3
    assert artifact["dofs"] == 6
    assert artifact["nnz"] == 18

    metrics = artifact["metrics"]
    for key in (
        "forward_relative_error",
        "hermitian_relative_error",
        "inner_product_error",
        "isolated_block_relative_error",
        "second_block_leakage",
    ):
        assert finite(metrics[key]) and metrics[key] <= TOLERANCE, key
    assert metrics["local_tensor_entries"] == 9
    thresholds = artifact["thresholds"]
    assert thresholds["forward_relative_error"] == TOLERANCE
    assert thresholds["hermitian_relative_error"] == TOLERANCE
    assert thresholds["inner_product_error"] == TOLERANCE
    assert thresholds["isolated_block_relative_error"] == TOLERANCE
    assert thresholds["second_block_leakage"] == TOLERANCE

    runtime = artifact["runtime"]
    for key in (
        "kernel_launches",
        "library_operations",
        "synchronization_calls",
        "failed_synchronizations",
        "allocation_operations",
        "staging_allocation_operations",
        "h2d_operations",
        "h2d_bytes",
        "d2h_operations",
        "d2h_bytes",
        "d2d_operations",
        "d2d_bytes",
        "live_allocation_bytes",
        "allocation_high_water_bytes",
        "live_staging_bytes",
        "staging_high_water_bytes",
    ):
        assert isinstance(runtime[key], int) and runtime[key] >= 0, key
    assert runtime["kernel_launches"] + runtime["library_operations"] > 0
    assert runtime["allocation_high_water_bytes"] > 0

    admission = artifact["admission"]
    assert admission == {
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "dispatch": True,
        "resident_structure": "device CSR descriptor and vectors",
        "unsupported_element_kernel": "explicitly not claimed",
    }
    assert artifact["unsupported_scope"] == [
        "native CUDA element/quadrature tensor kernel remains unsupported",
        "this packet does not claim partial assembly or general matrix-free FEM execution",
    ]

    contracts = artifact["failure_contracts"]
    assert len(contracts) == 1
    assert contracts[0]["id"] == "wrong_vector_shape"
    assert contracts[0]["status"] == "typed_rejected"
    assert contracts[0]["stage"] == "action_admission"
    assert contracts[0]["dispatch"] is False
    assert "InvalidArgument" in contracts[0]["error"]

    hardware = artifact["hardware"]
    assert isinstance(hardware, dict)
    assert hardware.get("device")
    assert hardware.get("driver")
    assert hardware.get("toolkit")
    print(
        json.dumps(
            {
                "case_id": CASE_ID,
                "elements": artifact["elements"],
                "forward_relative_error": metrics["forward_relative_error"],
                "hermitian_relative_error": metrics["hermitian_relative_error"],
                "inner_product_error": metrics["inner_product_error"],
                "kernel_or_library_operations": runtime["kernel_launches"] + runtime["library_operations"],
                "status": "pass",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
