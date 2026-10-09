#!/usr/bin/env python3
"""Strict checker for the bounded native-CUDA FEM1 P2 action packet."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


SCHEMA = "sonium-compute-cuda-fem-p2/v1"
CASE_ID = "compute.fem-p2-cuda.v1"
TOLERANCE = 2.0e-11


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM1 P2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{label} is not finite")
    return number


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
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
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        fail(f"checksum sidecar has no SHA-256 digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}")


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


def check_metrics(artifact: dict) -> None:
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        fail("metrics are missing")
    for key in (
        "forward_error",
        "hermitian_error",
        "warm_forward_error",
        "warm_hermitian_error",
        "retry_error",
    ):
        if finite(metrics.get(key), f"metrics.{key}") > TOLERANCE:
            fail(f"{key} exceeds CPU parity budget")
    if finite(metrics.get("warm_kernel_operations"), "metrics.warm_kernel_operations") < 2:
        fail("warm path has no native operations")
    for key in ("warm_h2d_operations", "warm_allocation_operations", "warm_staging_allocation_operations"):
        if finite(metrics.get(key), f"metrics.{key}") != 0:
            fail(f"warm residency has nonzero {key}")
    for key in (
        "malformed_csr_rejected",
        "malformed_shape_rejected",
        "malformed_output_preserved",
        "failure_typed",
        "retry_succeeded",
    ):
        if metrics.get(key) is not True:
            fail(f"{key} contract did not pass")
    if metrics.get("failure_published") is not False:
        fail("failed launch published a result")


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime counters are missing")
    if finite(runtime.get("kernel_launches"), "runtime.kernel_launches") + finite(
        runtime.get("library_operations"), "runtime.library_operations"
    ) <= 0:
        fail("runtime has no native CUDA work")
    if finite(runtime.get("allocation_high_water_bytes"), "runtime.allocation_high_water_bytes") <= 0:
        fail("runtime has no allocation high-water mark")
    for key in ("h2d_operations", "d2h_operations"):
        if finite(runtime.get(key), f"runtime.{key}") <= 0:
            fail(f"runtime.{key} does not prove data movement")


def check_artifact(path: Path) -> None:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not an object")
    if artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID:
        fail("schema or case identifier is not FEM1 P2")
    if artifact.get("status") != "passed":
        fail("artifact is not a passed result")

    cell = artifact.get("cell")
    if cell != {
        "space": "H1",
        "element": "Tri3",
        "order": "P2",
        "physics": "Helmholtz",
        "mode": "assembled",
        "scalar": "Complex64",
        "actions": ["forward", "hermitian"],
    }:
        fail("cell is not the bounded H1/Tri3/P2 Helmholtz pair")
    dispatch = artifact.get("dispatch")
    if not isinstance(dispatch, dict) or dispatch.get("requested_backend") != "cuda" or dispatch.get("actual_backend") != "cuda" or dispatch.get("fallback") != "none" or dispatch.get("admission_before_assembly") is not True or dispatch.get("native_device_csr") is not True:
        fail("dispatch does not prove required native CUDA admission")
    assembly = artifact.get("assembly")
    if not isinstance(assembly, dict) or "HelmholtzProblem::try_assemble" not in nonempty(assembly.get("production"), "assembly.production") or "P2" not in assembly["production"] or "independent host CSR" not in nonempty(assembly.get("independent_reference"), "assembly.independent_reference"):
        fail("production assembly or independent oracle provenance is missing")

    dimensions = artifact.get("dimensions")
    if not isinstance(dimensions, dict):
        fail("dimensions are missing")
    dofs = finite(dimensions.get("dofs"), "dimensions.dofs")
    if dofs <= 32 or dofs % 2 != 1:
        fail("fixture does not prove an odd nontrivial DOF count")
    if dimensions.get("odd_dimension") is not True or dimensions.get("tail_case") is not True:
        fail("odd/tail handling markers are missing")
    if finite(dimensions.get("tail_remainder_mod_256"), "dimensions.tail_remainder_mod_256") == 0:
        fail("fixture is block-aligned and does not exercise a tail")
    if finite(dimensions.get("nnz"), "dimensions.nnz") <= dofs:
        fail("assembled CSR is empty or trivial")

    check_metrics(artifact)
    check_runtime(artifact)

    failure = artifact.get("failure_contract")
    if not isinstance(failure, dict) or failure.get("stage") != "launch" or failure.get("error") != "BackendError::Launch" or failure.get("context_disposition") != "SynchronizeBeforeReuse" or failure.get("output_disposition") != "Unusable" or failure.get("typed") is not True or failure.get("partial_result_published") is not False or failure.get("recovery") != "synchronize_before_reuse" or failure.get("retry_succeeded") is not True:
        fail("typed launch failure/recovery contract is incomplete")
    if finite(failure.get("kernel_operations"), "failure_contract.kernel_operations") < 0:
        fail("failure kernel-operation count is negative")
    malformed = artifact.get("malformed_input_contract")
    if not isinstance(malformed, dict) or malformed.get("csr_stage") != "before_upload" or malformed.get("csr_error") != "HostCsrError::RowPointerCount" or malformed.get("shape_stage") != "action_admission" or malformed.get("shape_error") != "BackendError::InvalidArgument" or malformed.get("rejected") is not True or malformed.get("output_preserved") is not True:
        fail("malformed-input rejection/output-preservation contract is incomplete")

    teardown = artifact.get("teardown")
    if not isinstance(teardown, dict) or finite(teardown.get("live_before_bytes"), "teardown.live_before_bytes") <= 0 or finite(teardown.get("live_after_bytes"), "teardown.live_after_bytes") != 0 or finite(teardown.get("staging_after_bytes"), "teardown.staging_after_bytes") != 0 or teardown.get("resources_released") is not True:
        fail("teardown resource ledger did not return to zero")
    if finite(artifact.get("workload_elapsed_ms"), "workload_elapsed_ms") <= 0:
        fail("workload_elapsed_ms must be positive")
    check_hardware(artifact)
    check_checksum(path)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    check_artifact(path)
    print(f"FEM1 P2 native CUDA artifact passed strict checks: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
