#!/usr/bin/env python3
"""Strict checker for the bounded SM1 CUDA driver-model evidence.

The canonical DriverModel artifact contains the response and execution
identity, while the QA_RESULT line contains live device counters and hardware
provenance.  Both are required here: a response table by itself cannot prove
that CUDA executed it, and a result line by itself cannot prove that the
serialized model artifact is valid or bound to its CPU reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "speaker.cuda-bounded-typed-driver-reference.v1"
ARTIFACT_SCHEMA_VERSION = 2
PROVENANCE_SCHEMA_VERSION = 1
EXPECTED_FREQUENCIES = (100.0, 1000.0)
SUBSYSTEMS = {
    "magnetic",
    "structural",
    "acoustic",
    "thermal",
    "sparse_coupled",
    "derivative",
    "nonlinear_material",
}
UNSUPPORTED_SUBSYSTEMS = {
    "magnetic",
    "structural",
    "acoustic",
    "thermal",
    "nonlinear_material",
}
HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SM1 artifact rejected: {message}")


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
    if value.strip().lower() in {"unknown", "unavailable", "unspecified", "n/a"}:
        fail(f"{label} is unavailable")
    return value


def checksum(path: Path) -> None:
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
    if expected is None or not HEX_64.fullmatch(expected):
        fail(f"checksum sidecar has no SHA-256 digest for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}")


def check_artifact(path: Path) -> dict:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read canonical artifact: {error}")
    if not isinstance(artifact, dict):
        fail("canonical artifact is not an object")
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        fail("unsupported DriverModel artifact schema")

    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        fail("artifact provenance is missing")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        fail("unsupported artifact provenance schema")
    if provenance.get("backend") != "cuda":
        fail("artifact provenance is not CUDA")
    if provenance.get("execution_profile") != "projected_reanalysis":
        fail("bounded CUDA artifact has an unexpected execution profile")
    if provenance.get("precision") != "f64" or provenance.get("deterministic") is not True:
        fail("artifact precision/determinism provenance is inconsistent")

    execution = artifact.get("execution")
    if not isinstance(execution, dict):
        fail("artifact execution evidence is missing")
    if execution.get("requested_backend") != "cuda" or execution.get("actual_backend") != "cuda":
        fail("artifact backend admission is not required native CUDA")
    if execution.get("precision") != "f64":
        fail("artifact execution precision is not f64")
    nonempty(execution.get("device"), "execution.device")
    nonempty(execution.get("toolchain"), "execution.toolchain")
    if execution.get("mesh_identity") != "driver-model:lumped-network:v1":
        fail("bounded driver mesh identity is not the declared fixture")
    grid_identity = nonempty(execution.get("frequency_grid_identity"), "execution.frequency_grid_identity")
    if not HEX_64.fullmatch(grid_identity):
        fail("frequency-grid identity is not a SHA-256 digest")
    reference_checksum = nonempty(
        execution.get("reference_checksum_sha256"), "execution.reference_checksum_sha256"
    )
    if not HEX_64.fullmatch(reference_checksum):
        fail("reference checksum is not a SHA-256 digest")

    convergence = execution.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("converged") is not True:
        fail("convergence evidence is not marked converged")
    if finite(convergence.get("max_relative_residual"), "convergence.max_relative_residual") > 1.0e-6:
        fail("convergence residual exceeds the extraction gate")
    if finite(
        convergence.get("power_closure_relative_error"),
        "convergence.power_closure_relative_error",
    ) > 5.0e-2:
        fail("power closure exceeds the extraction gate")
    if finite(convergence.get("coupling_iterations"), "convergence.coupling_iterations") < 1:
        fail("coupling iteration count does not prove a solve")

    transfer = execution.get("transfer")
    if not isinstance(transfer, dict):
        fail("transfer/residency evidence is missing")
    for key in ("h2d_operations", "h2d_bytes", "d2h_operations", "d2h_bytes"):
        if finite(transfer.get(key), f"transfer.{key}") <= 0:
            fail(f"transfer.{key} does not prove live device movement")
    high_water = finite(
        transfer.get("allocation_high_water_bytes"), "transfer.allocation_high_water_bytes"
    )
    live = finite(transfer.get("live_allocation_bytes"), "transfer.live_allocation_bytes")
    if high_water <= 0 or live < 0 or live > high_water:
        fail("device allocation watermarks are invalid")
    for key in ("live_staging_bytes", "staging_high_water_bytes"):
        if finite(transfer.get(key), f"transfer.{key}") < 0:
            fail(f"transfer.{key} is negative")

    promotion = execution.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("promoted") is not True:
        fail("ROM promotion is not accepted")
    for key, limit in (
        ("full_order_relative_residual", 1.0e-6),
        ("projected_relative_residual", 1.0e-6),
        ("gradient_relative_error", 1.0e-5),
        ("maximum_constraint_violation", 1.0e-10),
    ):
        if finite(promotion.get(key), f"promotion.{key}") > limit:
            fail(f"promotion.{key} exceeds its gate")
    if finite(promotion.get("rank_agreement"), "promotion.rank_agreement") < 1.0 - 1.0e-10:
        fail("promotion rank agreement is below its gate")

    points = artifact.get("points")
    if not isinstance(points, list) or len(points) != len(EXPECTED_FREQUENCIES):
        fail("bounded artifact must contain exactly two frequency points")
    point_fields = (
        "frequency_hz",
        "radiation_impedance_real_n_s_per_m",
        "radiation_impedance_imag_n_s_per_m",
        "velocity_real_m_per_s",
        "velocity_imag_m_per_s",
        "electrical_impedance_real_ohm",
        "electrical_impedance_imag_ohm",
    )
    for point, expected_frequency in zip(points, EXPECTED_FREQUENCIES):
        if not isinstance(point, dict):
            fail("response point is not an object")
        if finite(point.get("frequency_hz"), "point.frequency_hz") != expected_frequency:
            fail("frequency grid does not match the declared two-point fixture")
        for field in point_fields[1:]:
            finite(point.get(field), f"point.{field}")

    checksum(path)
    return artifact


def check_capability_matrix(raw: object) -> None:
    if not isinstance(raw, str):
        fail("gpu.capability_matrix is missing")
    try:
        matrix = json.loads(raw)
    except json.JSONDecodeError as error:
        fail(f"gpu.capability_matrix is not JSON: {error}")
    if not isinstance(matrix, dict):
        fail("gpu.capability_matrix is not an object")
    if matrix.get("requested_backend") != "cuda" or matrix.get("actual_backend") != "cuda":
        fail("capability matrix does not prove requested/actual CUDA")
    if matrix.get("precision") != "f64":
        fail("capability matrix precision is not f64")
    nonempty(matrix.get("device"), "gpu.capability_matrix.device")
    nonempty(matrix.get("toolchain"), "gpu.capability_matrix.toolchain")
    rows = matrix.get("subsystems")
    if not isinstance(rows, list):
        fail("capability matrix subsystem rows are missing")
    by_name = {row.get("subsystem"): row for row in rows if isinstance(row, dict)}
    if set(by_name) != SUBSYSTEMS:
        fail("capability matrix does not cover all seven SM1 subsystems")
    for name in UNSUPPORTED_SUBSYSTEMS:
        row = by_name[name]
        if row.get("state") != "unsupported":
            fail(f"{name} is advertised as CUDA-supported without an implementation")
        nonempty(row.get("reason"), f"capability.{name}.reason")
    for name in ("sparse_coupled", "derivative"):
        row = by_name[name]
        if row.get("state") not in {"native", "vendor_native"}:
            fail(f"{name} is not a native/vendor CUDA cell")
        nonempty(row.get("reason"), f"capability.{name}.reason")


def check_result(log_path: Path) -> None:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read QA result log: {error}")
    records = [line.removeprefix("QA_RESULT: ") for line in lines if line.startswith("QA_RESULT: ")]
    if len(records) != 1:
        fail("expected exactly one QA_RESULT record")
    try:
        result = json.loads(records[0])
    except json.JSONDecodeError as error:
        fail(f"QA_RESULT is not JSON: {error}")
    if not isinstance(result, dict):
        fail("QA_RESULT is not an object")
    if result.get("schema_version") != 1 or result.get("case_id") != CASE_ID:
        fail("QA_RESULT schema or case identifier is not SM1")
    if result.get("status") != "passed":
        fail("QA_RESULT is not a passed result")
    if result.get("requested_backend") != "cuda" or result.get("actual_backend") != "cuda":
        fail("QA_RESULT backend provenance is not required native CUDA")
    outcomes = result.get("backend_outcomes")
    if not isinstance(outcomes, list) or not any(
        isinstance(row, dict) and row.get("backend") == "cuda" and row.get("status") == "passed"
        for row in outcomes
    ):
        fail("QA_RESULT has no passing CUDA backend outcome")

    environment = result.get("environment")
    if not isinstance(environment, dict):
        fail("QA_RESULT environment is missing")
    if environment.get("platform", "").lower() != "linux-x86_64":
        fail("QA_RESULT platform is not native Linux")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail("QA_RESULT runtime evidence is missing")
    hardware = runtime.get("hardware")
    if not isinstance(hardware, dict):
        fail("QA_RESULT runtime hardware provenance is missing")
    for key in (
        "device",
        "driver",
        "toolkit",
        "cpu_model",
        "microcode",
        "os",
        "compiler",
        "target_features",
        "runtime_dispatch",
        "thread_policy",
        "affinity_policy",
        "numa_policy",
        "deterministic_mode",
        "topology",
    ):
        nonempty(hardware.get(key), f"runtime.hardware.{key}")
    if "nvidia" not in hardware["device"].lower():
        fail("QA_RESULT runtime device is not NVIDIA")
    if hardware.get("os") != "linux-x86_64":
        fail("QA_RESULT runtime OS is not Linux x86_64")
    if hardware.get("target_features") != "cuda" or hardware.get("runtime_dispatch") != "cuda-native":
        fail("QA_RESULT CUDA target/dispatch provenance is inconsistent")

    counts = runtime.get("counts")
    if not isinstance(counts, dict) or counts.get("executed") != 1 or counts.get("passed") != 1:
        fail("QA_RESULT execution counts do not prove one passing run")
    if counts.get("failed") != 0 or counts.get("unsupported") != 0:
        fail("QA_RESULT execution counts contain failure or unsupported cells")
    counters = runtime.get("counters")
    if not isinstance(counters, dict):
        fail("QA_RESULT runtime counters are missing")
    for key in ("allocations", "h2d_transfers", "d2h_transfers", "synchronizations", "launches"):
        if finite(counters.get(key), f"runtime.counters.{key}") <= 0:
            fail(f"runtime.counters.{key} does not prove CUDA work")
    memory = runtime.get("memory")
    if not isinstance(memory, dict):
        fail("QA_RESULT memory evidence is missing")
    peak = finite(memory.get("peak_bytes"), "runtime.memory.peak_bytes")
    live = finite(memory.get("live_bytes"), "runtime.memory.live_bytes")
    if peak <= 0 or live < 0 or live > peak:
        fail("QA_RESULT memory watermarks are invalid")

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        fail("QA_RESULT metrics are missing")
    limits = {
        "max_relative_residual": 1.0e-6,
        "max_power_balance_relative_error": 5.0e-2,
        "max_velocity_relative_error": 1.0e-7,
        "max_impedance_relative_error": 1.0e-7,
        "max_derivative_relative_error": 1.0e-7,
    }
    for key, limit in limits.items():
        if finite(metrics.get(key), f"metrics.{key}") > limit:
            fail(f"{key} exceeds its declared SM1 gate")
    if finite(metrics.get("failure_modes_exercised"), "metrics.failure_modes_exercised") != 2:
        fail("required typed failure probes were not both exercised")
    if finite(metrics.get("unsupported_subsystems"), "metrics.unsupported_subsystems") != 5:
        fail("unsupported high-level subsystem count is not explicit")
    check_capability_matrix(environment.get("gpu.capability_matrix"))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json QA_RESULT.log")
    check_artifact(Path(sys.argv[1]))
    check_result(Path(sys.argv[2]))
    print(f"SM1 native CUDA artifact passed strict checks: {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
