#!/usr/bin/env python3
"""Strict checker for the bounded FEM1 native matrix-free CUDA packet."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


SCHEMA = "sonium-compute-cuda-fem-matrix-free/v1"
CASE_ID = "compute.fem-matrix-free-cuda.v1"
TOLERANCE = 2.0e-12
NATIVE_ROWS = {
    f"h1.tri3.p1.helmholtz.matrix-free.{scalar}.{action}"
    for scalar in ("f64", "Complex64")
    for action in ("forward", "transpose", "hermitian")
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM1 matrix-free artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{label} is not finite")
    return number


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} is missing")
    return value


def check_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar is missing: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"checksum sidecar has no digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}: {actual} != {expected}")


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
        fail("hardware.device is not NVIDIA")
    if hardware.get("os") != "linux-x86_64":
        fail("hardware.os is not native Linux x86_64")
    if hardware.get("target_features") != "cuda" or hardware.get("runtime_dispatch") != "cuda-native":
        fail("CUDA target/dispatch provenance is inconsistent")


def check_dispatch(artifact: dict) -> None:
    dispatch = artifact.get("dispatch")
    expected = {
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback": "none",
        "admission_before_assembly": True,
        "native_batched_local_kernel": True,
        "matrix_free_global_csr": False,
    }
    if dispatch != expected:
        fail(f"dispatch provenance is not native matrix-free CUDA: {dispatch!r}")


def check_capability_rows(artifact: dict) -> None:
    rows = artifact.get("capability_rows")
    if not isinstance(rows, list):
        fail("matrix-free capability rows are missing")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if set(by_id) != NATIVE_ROWS:
        fail(f"matrix-free capability IDs differ: {set(by_id)!r}")
    for row_id, row in by_id.items():
        if row.get("status") not in {"native", "native_real_equivalent"}:
            fail(f"{row_id} is not native")
        if row.get("backend") != "cuda" or row.get("mode") != "matrix-free":
            fail(f"{row_id} is not a matrix-free CUDA row")
        if row.get("requested_backend") != "cuda" or row.get("actual_backend") != "cuda":
            fail(f"{row_id} lacks actual CUDA provenance")
        if row.get("dispatch") is not True or row.get("fallback") != "none":
            fail(f"{row_id} has invalid dispatch/fallback metadata")
        nonempty(row.get("fallback_reason"), f"{row_id}.fallback_reason")
        nonempty(row.get("evidence"), f"{row_id}.evidence")


def check_metrics(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        fail("metrics are missing")
    for key in (
        "production_reference_error",
        "forward_error",
        "transpose_error",
        "hermitian_error",
        "inner_product_error",
        "taylor_error",
        "warm_forward_error",
        "warm_transpose_error",
        "warm_hermitian_error",
        "retry_error",
    ):
        if finite(metrics.get(key), f"metrics.{key}") > TOLERANCE:
            fail(f"{key} exceeds {TOLERANCE}")
    for key in (
        "warm_kernel_launches",
        "warm_h2d_operations",
        "warm_allocation_operations",
        "warm_staging_allocation_operations",
    ):
        if not isinstance(metrics.get(key), int) or isinstance(metrics[key], bool) or metrics[key] < 0:
            fail(f"{key} is not a non-negative integer")
    if metrics["warm_kernel_launches"] < 3:
        fail("warm action did not launch three native kernels")
    if any(metrics[key] != 0 for key in ("warm_h2d_operations", "warm_allocation_operations", "warm_staging_allocation_operations")):
        fail("warm action performed transfer/allocation work")
    for key in (
        "failure_typed",
        "output_preserved",
        "retry_succeeded",
        "malformed_rejected",
        "malformed_output_preserved",
        "empty_batch_accepted",
    ):
        if metrics.get(key) is not True:
            fail(f"{key} contract did not pass")


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime counters are missing")
    for key in (
        "kernel_launches",
        "library_operations",
        "synchronization_calls",
        "failed_synchronizations",
        "completion_events",
        "completion_waits",
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
        value = runtime.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"runtime.{key} is not a non-negative integer")
    if runtime["kernel_launches"] < 9:
        fail("runtime does not prove initial/warm/retry native actions")
    if runtime["h2d_operations"] <= 0 or runtime["d2h_operations"] <= 0:
        fail("runtime does not prove setup/publication transfers")
    if runtime["allocation_high_water_bytes"] <= 0:
        fail("runtime has no device allocation high-water mark")


def check_contracts(artifact: dict) -> None:
    contracts = artifact.get("failure_contracts")
    if not isinstance(contracts, list):
        fail("failure contracts are missing")
    by_id = {row.get("id"): row for row in contracts if isinstance(row, dict)}
    if set(by_id) != {
        "launch.output_preservation_and_retry",
        "malformed_local_tensor.shape",
        "empty_batch.noop",
    }:
        fail("failure contract IDs are incomplete")
    launch = by_id["launch.output_preservation_and_retry"]
    if launch.get("status") != "typed_failure_recovered" or launch.get("error") != "BackendError::Launch":
        fail("launch contract is not typed/recovered")
    if launch.get("output_preserved") is not True or launch.get("retry_succeeded") is not True:
        fail("launch contract lost output or retry")
    malformed = by_id["malformed_local_tensor.shape"]
    if malformed.get("status") != "typed_rejection" or malformed.get("error") != "BackendError::InvalidArgument":
        fail("malformed shape contract is not typed")
    if malformed.get("rejected") is not True or malformed.get("output_preserved") is not True:
        fail("malformed shape contract did not preserve output")
    empty = by_id["empty_batch.noop"]
    if empty.get("status") != "accepted_noop" or empty.get("accepted") is not True:
        fail("empty batch contract is not an accepted no-op")
    for row in contracts:
        nonempty(row.get("fallback"), f"failure contract {row.get('id')}.fallback")
        if row["fallback"] != "none":
            fail(f"failure contract {row.get('id')} silently fell back")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not an object")
    if artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID:
        fail("schema or case identifier is not FEM1 matrix-free")
    if artifact.get("status") != "passed":
        fail("artifact is not a passed result")
    if artifact.get("scope") != "native Linux x86_64 required CUDA SM120; Windows/Arm64/AVX512 waived":
        fail("scope is not bounded to the authorized native cell")
    if artifact.get("claim_class") != "production H1/Tri3/P1 Helmholtz matrix-free batched local action":
        fail("claim class is not bounded")
    cell = artifact.get("cell")
    if cell != {
        "space": "H1",
        "element": "Tri3",
        "order": "P1",
        "physics": "Helmholtz",
        "mode": "matrix-free",
        "scalar": "Complex64",
        "actions": ["forward", "transpose", "hermitian"],
    }:
        fail("cell declaration is not the supported matrix-free cell")
    assembly = artifact.get("assembly")
    if not isinstance(assembly, dict) or assembly.get("production") != "HelmholtzForm + assemble_bilinear_checked":
        fail("production FEM assembly provenance is missing")
    dimensions = artifact.get("dimensions")
    if dimensions != {
        "elements": 3,
        "local_dofs": 3,
        "batches": 3,
        "output_entries": 9,
        "tail_case": True,
        "tail_output_entries": 9,
        "tail_not_multiple_of_cuda_block_256": True,
    }:
        fail("tail-batch dimensions are not exact")
    check_hardware(artifact)
    check_dispatch(artifact)
    check_capability_rows(artifact)
    check_metrics(artifact)
    check_runtime(artifact)
    check_contracts(artifact)
    if finite(artifact.get("workload_elapsed_ms"), "workload_elapsed_ms") <= 0:
        fail("workload_elapsed_ms must be positive")
    unsupported = artifact.get("unsupported_scope")
    if not isinstance(unsupported, list) or len(unsupported) != 3:
        fail("unsupported scope is missing or overclaims")
    check_checksum(path)
    print(json.dumps({"case_id": CASE_ID, "status": "pass", "native_rows": len(NATIVE_ROWS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
