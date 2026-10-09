#!/usr/bin/env python3
"""Strict checker for the bounded native CUDA Quad4 BEM breadth artifact."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


CASE_ID = "bem.cuda-quad4-real-breadth.v1"
ROW_IDS = {
    "production.quad4.quadratic-regular",
    "production.quad4.quadratic-near-regular",
    "production.quadrature.tri3-singular-corrected",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"B1 Quad4 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        fail(f"{label} is not a positive integer")
    return value


def verify_checksum(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        fail(f"missing checksum sidecar for {path.name}")
    expected = None
    for row in sidecar.read_text(encoding="utf-8").splitlines():
        fields = row.split()
        if len(fields) == 2 and fields[1] == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64:
        fail(f"invalid checksum sidecar for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        fail(f"checksum mismatch for {path.name}: {actual} != {expected}")


def check_hardware(artifact: dict) -> None:
    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance is missing")
    required = (
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
    )
    for key in required:
        value = hardware.get(key)
        if not isinstance(value, str) or not value.strip():
            fail(f"hardware.{key} is missing")
    if "nvidia" not in hardware["device"].lower():
        fail("artifact is not from an NVIDIA device")
    if hardware.get("os") != "linux-x86_64":
        fail("artifact is not native Linux x86_64")
    if hardware["target_features"] != "cuda" or hardware["runtime_dispatch"] != "cuda-native":
        fail("CUDA target/dispatch provenance is inconsistent")


def check_action(row: dict, *, tail_required: bool) -> None:
    row_id = row.get("id")
    if row.get("status") != "passed":
        fail(f"{row_id} did not pass")
    tolerance = finite(row.get("tolerance"), f"{row_id}.tolerance")
    for key in ("relative_error", "cpu_production_crosscheck_error", "retry_relative_error"):
        if finite(row.get(key), f"{row_id}.{key}") > tolerance:
            fail(f"{row_id}.{key} exceeds tolerance")
    pair_count = positive_int(row.get("pair_count"), f"{row_id}.pair_count")
    samples = positive_int(row.get("samples_per_pair"), f"{row_id}.samples_per_pair")
    if tail_required and row.get("tail_samples") is not True:
        fail(f"{row_id} does not prove an odd/tail sample count")
    if row.get("malformed_rejected") is not True:
        fail(f"{row_id} did not reject malformed input")
    for name in ("initial", "warm", "publication", "failure", "retry"):
        stats = row.get(name)
        if not isinstance(stats, dict):
            fail(f"{row_id}.{name} counters are missing")
        positive_int(stats.get("kernel_launches"), f"{row_id}.{name}.kernel_launches")
    initial = row["initial"]
    warm = row["warm"]
    publication = row["publication"]
    if initial.get("allocations", 0) <= 0:
        fail(f"{row_id} first action did not allocate its persistent workspace")
    if warm.get("allocations") != 0 or warm.get("h2d_operations") != 0:
        fail(f"{row_id} warm action is not device-resident")
    if publication.get("d2h_operations", 0) <= 0:
        fail(f"{row_id} has no explicit host publication")
    if row.get("failure_typed") is not True or row.get("output_preserved") is not True:
        fail(f"{row_id} failure path is not typed and transactional")
    if pair_count == 9 and samples % 2 == 0 and tail_required:
        fail(f"{row_id} advertised a tail but sample count is even")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_b1_quad4_real_cuda.py ARTIFACT")
    path = Path(sys.argv[1])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"artifact is not valid JSON: {error}")
    if artifact.get("schema") != "sonium-bem-cuda-b1-quad4-real-evidence/v1":
        fail("unexpected schema")
    if artifact.get("status") != "passed" or artifact.get("case_id") != CASE_ID:
        fail("artifact status or case ID is invalid")
    check_hardware(artifact)
    dispatch = artifact.get("dispatch")
    if dispatch != {
        "backend": "cuda",
        "native": True,
        "production_adapter": "CudaBemQuadratureOperatorC64",
        "cuda_sm": "SM120",
    }:
        fail("native CUDA dispatch provenance is incomplete")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or {row.get("id") for row in rows} != ROW_IDS:
        fail("row IDs do not match the bounded Quad4/singular matrix")
    by_id = {row["id"]: row for row in rows}
    check_action(by_id["production.quad4.quadratic-regular"], tail_required=True)
    check_action(by_id["production.quad4.quadratic-near-regular"], tail_required=False)
    check_action(by_id["production.quadrature.tri3-singular-corrected"], tail_required=False)
    if len(artifact.get("unsupported_boundaries", [])) < 3:
        fail("unsupported production boundaries are not explicit")
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict) or int(runtime.get("kernel_launches", 0)) <= 0:
        fail("aggregate runtime does not prove a CUDA launch")
    verify_checksum(path)
    print(f"B1 Quad4 strict checker passed: {path}")


if __name__ == "__main__":
    main()
