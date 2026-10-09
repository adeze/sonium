#!/usr/bin/env python3
"""Strict checker for native CUDA B1 pair/field coverage evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "bem.cuda-pair-field-matrix.v1"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B1 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
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


def verify_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"checksum sidecar is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        fail(f"checksum sidecar has no valid digest for {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
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
    if "nvidia" not in hardware["device"].lower() or hardware.get("os") != "linux-x86_64":
        fail("hardware provenance is not native NVIDIA Linux")
    if hardware["target_features"] != "cuda" or hardware["runtime_dispatch"] != "cuda-native":
        fail("CUDA target/dispatch provenance is inconsistent")


def check_rows(artifact: dict) -> None:
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) not in {12, 13}:
        fail("B1 matrix must contain the twelve baseline rows or the thirteen-row breadth extension")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    baseline_ids = {
        "regular_pair.c32",
        "regular_pair.c64",
        "regular_pair.c64.source_x_derivative",
        "quadrature.c64.bulk-singular-edge",
        "field.c64.constant-tri3",
        "production_transmission.c64.constant-tri3",
        "production_tbem.resident-tri3-regular-near-singular",
        "production_tbem.real-f64-laplace-regular",
        "failure.regular_pair.launch",
        "failure.field.launch",
        "admission.quadrature.invalid_range",
        "admission.field.quad4-and-on-surface",
    }
    breadth_id = "quadrature.c64.near-singular-batch"
    expected_ids = baseline_ids | ({breadth_id} if breadth_id in by_id else set())
    if set(by_id) != expected_ids or len(by_id) != len(rows):
        fail(f"row IDs differ from the required B1 matrix: {sorted(by_id)}")
    for row_id, row in by_id.items():
        if row.get("status") not in {"passed", "unsupported"}:
            fail(f"{row_id} has invalid status")
    for row_id in (
        "regular_pair.c32",
        "regular_pair.c64",
        "regular_pair.c64.source_x_derivative",
        "quadrature.c64.bulk-singular-edge",
        "field.c64.constant-tri3",
    ):
        row = by_id[row_id]
        if row["status"] != "passed":
            fail(f"{row_id} did not pass")
        if finite(row.get("relative_error"), f"{row_id}.relative_error") > finite(row.get("tolerance"), f"{row_id}.tolerance"):
            fail(f"{row_id} exceeds its declared parity budget")
        if int(row.get("kernel_launches", 0)) <= 0:
            fail(f"{row_id} does not prove a CUDA launch")
        if int(row.get("d2h_operations", row.get("publication_d2h_operations", 0))) <= 0:
            fail(f"{row_id} does not prove an explicit publication")
    for row_id in (
        "regular_pair.c64.source_x_derivative",
        "quadrature.c64.bulk-singular-edge",
        "field.c64.constant-tri3",
    ):
        row = by_id[row_id]
        if int(row.get("steady_state_h2d_operations", -1)) != 0:
            fail(f"{row_id} performs a steady-state host-to-device transfer")
    derivative = by_id["regular_pair.c64.source_x_derivative"]
    if derivative.get("failure_typed") is not True or derivative.get("output_preserved") is not True:
        fail("source-x derivative launch failure was not typed and transactional")
    production = by_id["production_tbem.resident-tri3-regular-near-singular"]
    if production.get("status") != "passed":
        fail("production TBEM resident batch did not pass")
    for key in ("initial_relative_error", "warm_relative_error", "retry_relative_error"):
        if finite(production.get(key), f"production_tbem.{key}") > finite(
            production.get("tolerance"), "production_tbem.tolerance"
        ):
            fail(f"production_tbem.{key} exceeds its declared parity budget")
    for key in ("initial_kernel_launches", "warm_kernel_launches"):
        if int(production.get(key, 0)) <= 0:
            fail(f"production_tbem.{key} does not prove a native batch launch")
    if int(production.get("initial_h2d_operations", -1)) != 0:
        fail("production TBEM first action unexpectedly staged geometry")
    if int(production.get("warm_structural_allocations", -1)) != 0 or int(
        production.get("warm_steady_state_h2d_operations", -1)
    ) != 0:
        fail("production TBEM warm action allocated or staged resident geometry")
    if int(production.get("warm_publication_d2h_operations", 0)) < 2:
        fail("production TBEM warm action did not publish both assemblies")
    if production.get("failure_typed") is not True:
        fail("production TBEM launch failure was not typed")
    before = int(production.get("teardown_live_bytes_before", 0))
    after = int(production.get("teardown_live_bytes_after", -1))
    released = int(production.get("teardown_released_bytes", 0))
    if before <= after or released <= 0 or before - after != released:
        fail("production TBEM resident buffers were not released exactly")
    scalar = by_id["production_tbem.real-f64-laplace-regular"]
    if scalar.get("status") != "passed" or scalar.get("scalar") != "real f64 Laplace channels":
        fail("real-f64 production row is missing its scalar channel declaration")
    if int(scalar.get("active_pairs", 0)) <= 0:
        fail("real-f64 production row has no active pair batch")
    for key in ("initial_relative_error", "warm_relative_error", "retry_relative_error"):
        if finite(scalar.get(key), f"real-f64.{key}") > finite(
            scalar.get("tolerance"), "real-f64.tolerance"
        ):
            fail(f"real-f64.{key} exceeds its declared parity budget")
    for key in ("initial_kernel_launches", "warm_kernel_launches"):
        if int(scalar.get(key, 0)) <= 0:
            fail(f"real-f64.{key} does not prove a native CUDA launch")
    if int(scalar.get("initial_allocations", -1)) != 0 or int(
        scalar.get("initial_h2d_operations", -1)
    ) != 0:
        fail("real-f64 first action allocated or staged resident geometry")
    if int(scalar.get("warm_structural_allocations", -1)) != 0 or int(
        scalar.get("warm_steady_state_h2d_operations", -1)
    ) != 0:
        fail("real-f64 warm action allocated or staged resident geometry")
    if int(scalar.get("initial_d2h_operations", 0)) < 1 or int(
        scalar.get("warm_publication_d2h_operations", 0)
    ) < 1:
        fail("real-f64 action did not publish scalar outputs")
    if scalar.get("failure_typed") is not True:
        fail("real-f64 launch failure was not typed")
    scalar_before = int(scalar.get("teardown_live_bytes_before", 0))
    scalar_after = int(scalar.get("teardown_live_bytes_after", -1))
    scalar_released = int(scalar.get("teardown_released_bytes", 0))
    if (
        scalar_before <= scalar_after
        or scalar_released <= 0
        or scalar_before - scalar_after != scalar_released
    ):
        fail("real-f64 resident buffers were not released exactly")
    transmission = by_id["production_transmission.c64.constant-tri3"]
    if transmission.get("status") != "passed":
        fail("constant-panel transmission cell did not pass")
    if transmission.get("scalar") != "Complex64" or transmission.get("element") != "Tri3-constant":
        fail("transmission cell is missing its scalar/element declaration")
    if transmission.get("formulation") != "constant-panel Muller two-medium transmission pair/field cell":
        fail("transmission cell formulation is not the bounded production cell")
    tolerance = finite(transmission.get("tolerance"), "transmission.tolerance")
    for key in (
        "pair_initial_relative_error",
        "pair_warm_relative_error",
        "pair_retry_relative_error",
        "field_exterior_initial_relative_error",
        "field_interior_initial_relative_error",
        "field_exterior_warm_relative_error",
        "field_interior_warm_relative_error",
    ):
        if finite(transmission.get(key), f"transmission.{key}") > tolerance:
            fail(f"transmission.{key} exceeds its declared parity budget")
    if int(transmission.get("pair_count", 0)) <= 0 or int(
        transmission.get("pair_tail", -1)
    ) != int(transmission.get("pair_count", 0)):
        fail("transmission pair batch does not prove a non-block-aligned tail")
    if int(transmission.get("field_points", 0)) <= 0 or int(
        transmission.get("field_tail", -1)
    ) != int(transmission.get("field_points", 0)):
        fail("transmission field batch does not prove a non-block-aligned tail")
    for prefix in ("pair", "field"):
        if int(transmission.get(f"{prefix}_initial_kernel_launches", 0)) < 2:
            fail(f"transmission {prefix} first action did not launch both media")
        if int(transmission.get(f"{prefix}_initial_allocations", 0)) <= 0:
            fail(f"transmission {prefix} first action did not allocate resident workspace")
        if int(transmission.get(f"{prefix}_initial_h2d_operations", -1)) != 0:
            fail(f"transmission {prefix} first action staged host input")
        if int(transmission.get(f"{prefix}_initial_d2h_operations", 0)) < 2:
            fail(f"transmission {prefix} first action did not publish both media")
        if int(transmission.get(f"{prefix}_warm_kernel_launches", 0)) < 2:
            fail(f"transmission {prefix} warm action did not launch both media")
        if int(transmission.get(f"{prefix}_warm_structural_allocations", -1)) != 0 or int(
            transmission.get(f"{prefix}_warm_steady_state_h2d_operations", -1)
        ) != 0:
            fail(f"transmission {prefix} warm action allocated or staged resident data")
        if int(transmission.get(f"{prefix}_warm_publication_d2h_operations", 0)) < 2:
            fail(f"transmission {prefix} warm action did not publish both media")
        if int(transmission.get(f"{prefix}_warm_d2d_operations", 0)) < 2:
            fail(f"transmission {prefix} warm action did not perform device publication")
    if transmission.get("failure_typed") is not True or transmission.get("failure_no_publication") is not True:
        fail("transmission launch failure was not typed and transactional")
    if int(transmission.get("failure_kernel_launches", -1)) != 0 or int(
        transmission.get("failure_d2d_operations", -1)
    ) != 0:
        fail("transmission failed action touched native output")
    if transmission.get("malformed_pair_typed") is not True or transmission.get("malformed_field_typed") is not True:
        fail("transmission malformed inputs were not rejected with typed errors")
    if transmission.get("empty_succeeds") is not True or any(
        int(transmission.get(key, -1)) != 0
        for key in ("empty_kernel_launches", "empty_h2d_operations", "empty_d2h_operations")
    ):
        fail("transmission empty batch contract is not no-op")
    transmission_before = int(transmission.get("teardown_live_bytes_before", 0))
    transmission_after = int(transmission.get("teardown_live_bytes_after", -1))
    transmission_released = int(transmission.get("teardown_released_bytes", 0))
    if (
        transmission_before <= transmission_after
        or transmission_released <= 0
        or transmission_before - transmission_after != transmission_released
    ):
        fail("transmission resident buffers were not released exactly")
    if breadth_id in by_id:
        batch = by_id[breadth_id]
        if batch.get("status") != "passed":
            fail("near/singular-corrected quadrature batch did not pass")
        if batch.get("scalar") != "Complex64" or batch.get("formulation") != "production chart-driven near/singular-corrected quadrature batch":
            fail("near/singular-corrected quadrature batch declaration is incomplete")
        if finite(batch.get("relative_error"), f"{breadth_id}.relative_error") > finite(
            batch.get("tolerance"), f"{breadth_id}.tolerance"
        ):
            fail("near/singular-corrected quadrature batch exceeds its parity budget")
        if int(batch.get("pair_count", 0)) != 5 or int(batch.get("pair_tail", -1)) != 5:
            fail("near/singular-corrected quadrature batch does not prove its five-pair tail")
        if int(batch.get("samples_per_pair", 0)) != 4 or int(batch.get("sample_count", 0)) != 20:
            fail("near/singular-corrected quadrature batch sample shape changed")
        if int(batch.get("kernel_launches", 0)) != 2 or int(batch.get("repeated_actions", 0)) != 2:
            fail("near/singular-corrected quadrature batch does not prove two CUDA actions")
        if int(batch.get("steady_state_h2d_operations", -1)) != 0 or int(
            batch.get("publication_d2h_operations", 0)
        ) != 1 or int(batch.get("warm_d2d_operations", 0)) != 2:
            fail("near/singular-corrected quadrature batch residency counters are invalid")
    for row_id in ("failure.regular_pair.launch", "failure.field.launch", "admission.quadrature.invalid_range"):
        row = by_id[row_id]
        if row.get("status") != "passed" or row.get("typed_error") is not True:
            fail(f"{row_id} is not a typed passing failure/admission row")
    for row_id in ("failure.regular_pair.launch", "failure.field.launch"):
        row = by_id[row_id]
        if row.get("output_preserved") is not True or row.get("cleanup") is not True:
            fail(f"{row_id} is not transactional/clean")
    unsupported = by_id["admission.field.quad4-and-on-surface"]
    if unsupported.get("status") != "unsupported" or not nonempty(unsupported.get("reason"), "unsupported field reason"):
        fail("unsupported field scope is not explicit")


def check_runtime(artifact: dict) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime evidence is missing")
    for key in ("kernel_launches", "synchronization_calls", "h2d_operations", "d2h_operations", "allocation_high_water_bytes"):
        if int(runtime.get(key, 0)) <= 0:
            fail(f"runtime.{key} does not prove native work")
    finite(artifact.get("elapsed_ms"), "elapsed_ms")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not a JSON object")
    if artifact.get("schema") != "sonium-compute-cuda-bem-b1-evidence/v1" or artifact.get("status") != "passed":
        fail("schema/status is not a passed B1 artifact")
    if artifact.get("case_id") != CASE_ID:
        fail("case identifier is not the B1 producer")
    capabilities = artifact.get("capabilities")
    expected_capabilities = {
        "regular_pair.c32": "native",
        "regular_pair.c64": "native",
        "regular_pair.c64.source_x_derivative": "native",
        "regular_pair.real_f64_laplace": "native",
        "quadrature.c64": "native",
        "field.c64.constant_tri3": "native",
        "transmission.c64.constant_tri3": "native",
        "field.c64.quad4": "unsupported",
        "field.c64.on_surface": "unsupported",
        "higher_order": "unsupported",
    }
    breadth_capabilities = {
        **expected_capabilities,
        "quadrature.c64.near-singular-batch": "native",
    }
    has_breadth_row = isinstance(artifact.get("rows"), list) and any(
        isinstance(row, dict) and row.get("id") == "quadrature.c64.near-singular-batch"
        for row in artifact["rows"]
    )
    expected_capability_matrix = breadth_capabilities if has_breadth_row else expected_capabilities
    if capabilities != expected_capability_matrix:
        fail("capability matrix is incomplete or relabelled")
    check_hardware(artifact)
    check_rows(artifact)
    check_runtime(artifact)
    verify_checksum(path)
    print(f"B1 native CUDA artifact passed strict checks: {path}")


if __name__ == "__main__":
    main()
