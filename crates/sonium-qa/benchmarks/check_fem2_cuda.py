#!/usr/bin/env python3
"""Strict checker for the bounded native CUDA FEM2 device-solver contract."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "fem.device-solver.v1"
SCHEMA = "sonium-fem-device-solver/v1"
RESIDUAL_TOLERANCE = 2.0e-10
TRANSFER_TOLERANCE = 2.0e-13
WARM_REPEATS = 3


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FEM2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} is missing")
    return value


def verify_artifact_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"artifact checksum sidecar is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"artifact checksum sidecar has no valid digest for {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        fail(f"artifact checksum mismatch for {path.name}")


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
    if "nvidia" not in hardware["device"].lower() or hardware.get("os") != "linux-x86_64":
        fail("hardware provenance is not native NVIDIA Linux")
    if hardware["target_features"] != "cuda" or hardware["runtime_dispatch"] != "cuda-native":
        fail("CUDA target/dispatch provenance is inconsistent")


def check_capabilities(artifact: dict) -> None:
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        fail("capability environment is missing")
    expected = {
        "capability.fem.space.h1": "native",
        "capability.fem.space.hcurl": "unsupported",
        "capability.fem.space.hdiv": "unsupported",
        "capability.fem.element.tri3": "native",
        "capability.fem.element.other": "unsupported",
        "capability.fem.order.p1": "native",
        "capability.fem.order.higher": "unsupported",
        "capability.fem.form.helmholtz": "native",
        "capability.fem.form.other": "unsupported",
        "capability.fem.mode.assembled_csr": "native",
        "capability.fem.mode.partial_or_matrix_free": "unsupported",
        "capability.fem.scalar.complex64": "native",
        "capability.fem.scalar.f64_accumulation": "native",
        "capability.fem.constraint.dirichlet_homogeneous": "native_after_host_setup",
        "capability.fem.constraint.periodic": "native_device_transfer_only",
        "capability.fem.constraint.hanging": "native_device_transfer_only",
        "capability.fem.constraint.affine_mpc": "native_device_transfer_only",
        "capability.fem.transfer.h1_affine": "native_device_csr",
        "capability.fem.solver.gmres": "native_device_resident",
        "capability.fem.preconditioner.point_jacobi": "native_device_resident",
        "capability.fem.fallback.required_cuda": "none",
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            fail(f"{key} must be {value!r}, got {environment.get(key)!r}")


def check_solve_contract(artifact: dict) -> None:
    report = artifact.get("fem_solve_report")
    if not isinstance(report, dict):
        fail("fem_solve_report is missing")
    if report.get("backend_requested") != "cuda" or report.get("backend_actual") != "cuda":
        fail("solve report does not prove required native CUDA")
    if report.get("precision") != "Complex64" or report.get("accumulation_precision") != "f64":
        fail("solve precision provenance is inconsistent")
    if report.get("fallback") is not False:
        fail("solve report claims a CPU fallback")
    convergence = report.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("converged") is not True:
        fail("device solve did not converge")
    if int(convergence.get("iterations", 0)) <= 0:
        fail("device solve has no positive iteration count")
    for key in ("projected_residual", "true_residual", "backward_error"):
        finite(convergence.get(key), f"convergence.{key}")
    if convergence["true_residual"] > RESIDUAL_TOLERANCE:
        fail("device true residual exceeds declared budget")
    physical = report.get("physical_diagnostics")
    if not isinstance(physical, dict) or physical.get("reference") != "independent host CSR accumulation":
        fail("independent physical residual reference is missing")
    if finite(physical.get("relative_residual"), "physical_diagnostics.relative_residual") > RESIDUAL_TOLERANCE:
        fail("independent physical residual exceeds declared budget")


def check_residency(artifact: dict) -> None:
    warm = artifact.get("warm_residency")
    if not isinstance(warm, dict) or warm.get("repeats") != WARM_REPEATS:
        fail("warm residency repeat count is incorrect")
    for key in (
        "allocation_operations",
        "device_allocation_operations",
        "staging_allocation_operations",
        "h2d_operations",
        "h2d_bytes",
        "d2h_operations",
        "d2h_bytes",
        "d2d_operations",
        "d2d_bytes",
    ):
        values = warm.get(key)
        if not isinstance(values, list) or len(values) != WARM_REPEATS or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values
        ):
            fail(f"warm residency {key} is malformed")
    if any(warm["device_allocation_operations"]):
        fail("warm residency allocated device buffers")
    if warm["allocation_operations"] != [
        device + staging
        for device, staging in zip(
            warm["device_allocation_operations"], warm["staging_allocation_operations"]
        )
    ]:
        fail("warm residency total allocation classification is inconsistent")
    for key in (
        "staging_allocation_operations",
        "h2d_operations",
        "h2d_bytes",
        "d2h_operations",
        "d2h_bytes",
        "d2d_operations",
        "d2d_bytes",
    ):
        if len(set(warm[key])) != 1:
            fail(f"warm residency {key} is not stable across repeats")
    if not warm["h2d_operations"] or not warm["h2d_bytes"]:
        fail("warm residency does not record the RHS/descriptor update")
    classification = warm.get("transfer_classification")
    if not isinstance(classification, dict):
        fail("warm residency transfer classification is missing")
    if "no Krylov vectors" not in str(classification.get("d2h", "")):
        fail("warm D2H classification does not exclude Krylov vectors")
    if "device-internal" not in str(classification.get("d2d", "")):
        fail("warm D2D classification does not identify device-internal traffic")
    constraints = artifact.get("constraints")
    if not isinstance(constraints, dict) or "none_for_vectors" not in str(
        constraints.get("host_round_trips_during_krylov", "")
    ):
        fail("vector host-round-trip contract is missing")
    if int(warm.get("peak_device_bytes", 0)) <= 0 or int(warm.get("steady_device_bytes", 0)) <= 0:
        fail("warm residency does not prove live device allocation")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        fail("metrics are missing")
    if finite(metrics.get("transfer_error"), "metrics.transfer_error") > TRANSFER_TOLERANCE:
        fail("device transfer exceeds declared budget")
    if metrics.get("malformed_csr_rejected") is not True:
        fail("typed malformed-CSR admission probe did not pass")
    affine = artifact.get("constraints", {}).get("affine_mpc")
    if not isinstance(affine, dict):
        fail("affine MPC transfer evidence is missing")
    if affine.get("status") != "native_device_transfer_only":
        fail("affine MPC transfer is not explicitly device-resident")
    if affine.get("reference") != "independent CPU MpcSystem::recover":
        fail("affine MPC CPU reference is missing")
    if affine.get("lift") != "nonzero affine hanging-node lift":
        fail("affine MPC nonzero lift is missing")
    for key in ("full_dofs", "reduced_dofs", "constraint_count"):
        value = affine.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            fail(f"affine MPC {key} is invalid")
    if affine["full_dofs"] != 5 or affine["reduced_dofs"] != 3 or affine["constraint_count"] != 2:
        fail("affine MPC fixture dimensions changed")
    if finite(affine.get("max_error"), "constraints.affine_mpc.max_error") > 2.0e-13:
        fail("affine MPC device transfer exceeds declared budget")
    if finite(metrics.get("affine_mpc_transfer_error"), "metrics.affine_mpc_transfer_error") > 2.0e-13:
        fail("affine MPC metric exceeds declared budget")
    for key, expected in (
        ("affine_mpc_full_dofs", 5),
        ("affine_mpc_reduced_dofs", 3),
        ("affine_mpc_constraint_count", 2),
    ):
        if metrics.get(key) != expected:
            fail(f"metrics.{key} does not match the affine MPC fixture")


def check_failures(artifact: dict, fallback_log: Path) -> None:
    contract = artifact.get("failure_contract")
    if not isinstance(contract, dict):
        fail("failure contract is missing")
    malformed = contract.get("malformed_csr")
    if not isinstance(malformed, dict) or malformed.get("status") != "rejected_before_device_upload" or malformed.get("typed_error") != "InvalidArgument" or malformed.get("checked") is not True:
        fail("malformed CSR failure contract is incomplete")
    if contract.get("required_cuda_fallback") != "runner_negative_probe":
        fail("required-CUDA fallback contract is missing")
    try:
        log = fallback_log.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"fallback probe log is missing or unreadable: {error}")
    if "CUDA fallback fault injection is active" not in log or "FAILED" not in log:
        fail("required-CUDA fallback probe did not fail closed")


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        fail("usage: check_fem2_cuda.py ARTIFACT [FALLBACK_LOG]")
    artifact_path = Path(sys.argv[1])
    fallback_log = Path(sys.argv[2]) if len(sys.argv) == 3 else artifact_path.with_name("fallback_probe.log")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"artifact is missing or invalid JSON: {error}")
    if artifact.get("schema") != SCHEMA or artifact.get("case_id") != CASE_ID or artifact.get("status") != "passed":
        fail("artifact identity/status is not a passed FEM2 producer")
    verify_artifact_checksum(artifact_path)
    check_hardware(artifact)
    check_capabilities(artifact)
    check_solve_contract(artifact)
    check_residency(artifact)
    check_failures(artifact, fallback_log)
    print(json.dumps({"status": "passed", "case_id": CASE_ID, "schema": SCHEMA}))


if __name__ == "__main__":
    main()
