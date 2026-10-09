#!/usr/bin/env python3
"""Fail-closed checker for measured native-versus-cuBLAS C6 evidence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn


SCHEMA = "sonium-compute-cuda-vendor-comparison/v1"
CASE_ID = "compute.fem-krylov-multi-dot-cuda-vendor-comparison.v1"
TOLERANCE = 2.0e-10


def fail(message: str) -> NoReturn:
    raise ValueError(f"C6 vendor comparison rejected: {message}")


def number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
        fail(f"{field} must be finite and {'positive' if positive else 'non-negative'}")
    return value


def integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{field} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        fail(f"{field} must be {'positive' if positive else 'non-negative'}")
    return value


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{field} must be non-empty text")
    return value


def check(path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    if not isinstance(artifact, dict):
        fail("artifact must be an object")
    if artifact.get("schema") != SCHEMA or artifact.get("version") != 1:
        fail("schema/version is invalid")
    if artifact.get("status") != "passed" or artifact.get("case_id") != CASE_ID:
        fail("status or case identity is invalid")
    claims = artifact.get("claims")
    if not isinstance(claims, dict) or claims.get("vendor_comparison") != (
        "measured_shape_matched_cublas_gemv_vs_native_fused_multi_dot"
    ):
        fail("vendor comparison claim is not the measured contract")
    if claims.get("nsight_counters") != "not_claimed":
        fail("Nsight status must remain explicitly unclaimed")

    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        fail("hardware provenance is missing")
    for field in (
        "device",
        "driver",
        "toolkit",
        "compiler",
        "topology",
        "peak_flops",
        "memory_bandwidth",
        "os",
        "target_features",
        "runtime_dispatch",
        "deterministic_mode",
    ):
        text(hardware.get(field), f"hardware.{field}")
    if hardware["os"] != "linux-x86_64" or hardware["target_features"] != "cuda":
        fail("artifact is not native Linux CUDA evidence")
    if hardware["runtime_dispatch"] != "cuda-native":
        fail("artifact does not prove native CUDA dispatch")
    if "sm120" not in text(hardware.get("device"), "hardware.device").lower() and "blackwell" not in hardware["device"].lower():
        fail("artifact device is not the declared SM120/Blackwell target")
    peak = number(float(hardware["peak_flops"]), "hardware.peak_flops", positive=True)
    bandwidth = number(float(hardware["memory_bandwidth"]), "hardware.memory_bandwidth", positive=True)

    workload = artifact.get("workload")
    if not isinstance(workload, dict):
        fail("workload is missing")
    if workload.get("scalar") != "f64" or workload.get("operation") != "multi_dot":
        fail("workload scalar/operation is invalid")
    text(workload.get("native_dispatch"), "workload.native_dispatch")
    text(workload.get("vendor_dispatch"), "workload.vendor_dispatch")
    vector_len = integer(workload.get("vector_len"), "workload.vector_len", positive=True)
    basis_count = integer(workload.get("basis_count"), "workload.basis_count", positive=True)
    repeats = integer(workload.get("repeats"), "workload.repeats", positive=True)
    if vector_len < 1024 or basis_count < 2 or repeats < 3:
        fail("workload is below the minimum evidence shape")
    if workload.get("correctness_before_timing") is not True:
        fail("correctness gate did not precede timing")
    tolerance = number(workload.get("tolerance"), "workload.tolerance", positive=True)
    if tolerance != TOLERANCE:
        fail("workload tolerance differs from the checked contract")
    native_error = number(workload.get("native_relative_error"), "workload.native_relative_error")
    vendor_error = number(workload.get("vendor_relative_error"), "workload.vendor_relative_error")
    if native_error > tolerance or vendor_error > tolerance:
        fail("pre-timing correctness gate exceeds tolerance")

    flops_per_repeat = integer(workload.get("flops_per_repeat"), "workload.flops_per_repeat", positive=True)
    total_flops = integer(workload.get("total_flops"), "workload.total_flops", positive=True)
    bytes_per_repeat = integer(workload.get("bytes_per_repeat"), "workload.bytes_per_repeat", positive=True)
    total_bytes = integer(workload.get("total_bytes"), "workload.total_bytes", positive=True)
    expected_flops_per_repeat = basis_count * vector_len * 2
    expected_bytes_per_repeat = basis_count * (vector_len * 2 + 1) * 8
    if flops_per_repeat != expected_flops_per_repeat or total_flops != flops_per_repeat * repeats:
        fail("FLOP model does not match the measured shape")
    if bytes_per_repeat != expected_bytes_per_repeat or total_bytes != bytes_per_repeat * repeats:
        fail("byte model does not match the measured shape")
    intensity = number(
        workload.get("arithmetic_intensity_flops_per_byte"),
        "workload.arithmetic_intensity_flops_per_byte",
        positive=True,
    )
    if not math.isclose(intensity, flops_per_repeat / bytes_per_repeat, rel_tol=1e-12):
        fail("arithmetic intensity does not match the workload model")

    measurements = artifact.get("measurements")
    if not isinstance(measurements, dict):
        fail("measurements are missing")
    rows = {}
    for name in ("native", "vendor"):
        row = measurements.get(name)
        if not isinstance(row, dict):
            fail(f"measurements.{name} is missing")
        elapsed = number(row.get("elapsed_ms"), f"measurements.{name}.elapsed_ms", positive=True)
        throughput = number(
            row.get("throughput_flops_per_second"),
            f"measurements.{name}.throughput_flops_per_second",
            positive=True,
        )
        effective_bandwidth = number(
            row.get("effective_memory_bandwidth_bytes_per_second"),
            f"measurements.{name}.effective_memory_bandwidth_bytes_per_second",
            positive=True,
        )
        compute_fraction = number(
            row.get("roofline_compute_fraction"),
            f"measurements.{name}.roofline_compute_fraction",
        )
        memory_fraction = number(
            row.get("roofline_memory_fraction"),
            f"measurements.{name}.roofline_memory_fraction",
        )
        seconds = elapsed / 1000.0
        if not math.isclose(throughput, total_flops / seconds, rel_tol=1e-9):
            fail(f"measurements.{name} throughput is not derived from elapsed time")
        if not math.isclose(effective_bandwidth, total_bytes / seconds, rel_tol=1e-9):
            fail(f"measurements.{name} bandwidth is not derived from elapsed time")
        if not math.isclose(compute_fraction, throughput / peak, rel_tol=1e-9):
            fail(f"measurements.{name} compute roofline is not derived from measured throughput")
        if not math.isclose(memory_fraction, effective_bandwidth / bandwidth, rel_tol=1e-9):
            fail(f"measurements.{name} memory roofline is not derived from measured bandwidth")
        if compute_fraction > 1.0 or memory_fraction > 1.0:
            fail(f"measurements.{name} roofline fraction exceeds one")
        rows[name] = {"elapsed_ms": elapsed, "throughput_flops_per_second": throughput}
    speedup = number(measurements.get("vendor_over_native_speedup"), "measurements.vendor_over_native_speedup", positive=True)
    if not math.isclose(speedup, rows["vendor"]["elapsed_ms"] / rows["native"]["elapsed_ms"], rel_tol=1e-9):
        fail("speedup is not derived from measured elapsed times")

    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime counters are missing")
    native = runtime.get("native")
    vendor = runtime.get("vendor")
    if not isinstance(native, dict) or not isinstance(vendor, dict):
        fail("native/vendor runtime counters are missing")
    if integer(native.get("kernel_launches"), "runtime.native.kernel_launches") != repeats:
        fail("native launch count does not match repeats")
    if integer(native.get("library_operations"), "runtime.native.library_operations") != 0:
        fail("native path unexpectedly reports vendor operations")
    if integer(vendor.get("kernel_launches"), "runtime.vendor.kernel_launches") != 0:
        fail("vendor path reports custom kernel launches")
    if integer(vendor.get("library_operations"), "runtime.vendor.library_operations") != repeats * basis_count:
        fail("vendor operation count does not match repeated shape")
    for name, row in (("native", native), ("vendor", vendor)):
        for field in ("h2d_operations", "d2h_operations"):
            if integer(row.get(field), f"runtime.{name}.{field}") != 0:
                fail(f"runtime.{name}.{field} proves hidden transfer during timing")
        d2d_operations = integer(row.get("d2d_operations"), f"runtime.{name}.d2d_operations")
        expected_d2d_operations = repeats if name == "native" else 0
        if d2d_operations != expected_d2d_operations:
            fail(
                f"runtime.{name}.d2d_operations must be {expected_d2d_operations} "
                "for the resident implementation"
            )
        if integer(row.get("synchronization_calls"), f"runtime.{name}.synchronization_calls") != 1:
            fail(f"runtime.{name} does not have exactly one timed synchronization")
        if integer(row.get("allocation_operations"), f"runtime.{name}.allocation_operations") != 0:
            fail(f"runtime.{name} allocated during timing")
        integer(row.get("allocation_high_water_bytes"), f"runtime.{name}.allocation_high_water_bytes", positive=True)
    if integer(runtime.get("publication_d2h_operations"), "runtime.publication_d2h_operations", positive=True) != basis_count + 1:
        fail("publication transfer count is not explicit")

    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        fail("producer provenance is missing")
    text(provenance.get("producer"), "provenance.producer")
    text(provenance.get("measurement"), "provenance.measurement")
    return {
        "status": "passed",
        "schema": SCHEMA,
        "case_id": CASE_ID,
        "vector_len": vector_len,
        "basis_count": basis_count,
        "repeats": repeats,
        "native_elapsed_ms": rows["native"]["elapsed_ms"],
        "vendor_elapsed_ms": rows["vendor"]["elapsed_ms"],
        "vendor_over_native_speedup": speedup,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} ARTIFACT.json", file=sys.stderr)
        return 2
    try:
        result = check(Path(sys.argv[1]))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
