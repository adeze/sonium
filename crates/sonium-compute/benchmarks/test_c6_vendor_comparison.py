"""Focused tests for the strict measured C6 vendor-comparison contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = ROOT / "crates/sonium-compute/benchmarks/check_c6_vendor_comparison.py"


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("check_c6_vendor_comparison", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact() -> dict:
    vector_len = 1024
    basis_count = 2
    repeats = 3
    flops_per_repeat = vector_len * basis_count * 2
    bytes_per_repeat = basis_count * (vector_len * 2 + 1) * 8

    def measurement(elapsed_ms: float) -> dict:
        seconds = elapsed_ms / 1000.0
        throughput = flops_per_repeat * repeats / seconds
        effective_bandwidth = bytes_per_repeat * repeats / seconds
        return {
            "elapsed_ms": elapsed_ms,
            "throughput_flops_per_second": throughput,
            "effective_memory_bandwidth_bytes_per_second": effective_bandwidth,
            "roofline_compute_fraction": throughput / 125_000_000_000_000,
            "roofline_memory_fraction": effective_bandwidth / 1_792_000_000_000,
        }

    runtime = {
        "kernel_launches": repeats,
        "library_operations": 0,
        "synchronization_calls": 1,
        "h2d_operations": 0,
        "d2h_operations": 0,
        "d2d_operations": repeats,
        "allocation_operations": 0,
        "allocation_high_water_bytes": 1,
    }
    vendor_runtime = dict(
        runtime,
        kernel_launches=0,
        library_operations=repeats * basis_count,
        d2d_operations=0,
    )
    return {
        "schema": "sonium-compute-cuda-vendor-comparison/v1",
        "version": 1,
        "status": "passed",
        "case_id": "compute.fem-krylov-multi-dot-cuda-vendor-comparison.v1",
        "claims": {
            "vendor_comparison": "measured_shape_matched_cublas_gemv_vs_native_fused_multi_dot",
            "nsight_counters": "not_claimed",
        },
        "hardware": {
            "device": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "driver": "610.57.04",
            "toolkit": "CUDA UMD 13.3",
            "compiler": "rustc test",
            "topology": "GPU0 PCIe Gen5 x16",
            "peak_flops": "125000000000000",
            "memory_bandwidth": "1792000000000",
            "os": "linux-x86_64",
            "target_features": "cuda",
            "runtime_dispatch": "cuda-native",
            "deterministic_mode": "enabled",
        },
        "workload": {
            "scalar": "f64",
            "operation": "multi_dot",
            "native_dispatch": "sonium fused CUDA multi_dot kernel",
            "vendor_dispatch": "cuBLAS GEMV",
            "vector_len": vector_len,
            "basis_count": basis_count,
            "repeats": repeats,
            "shape_match": "same shape",
            "correctness_before_timing": True,
            "native_relative_error": 1.0e-12,
            "vendor_relative_error": 1.0e-12,
            "tolerance": 2.0e-10,
            "flops_per_repeat": flops_per_repeat,
            "total_flops": flops_per_repeat * repeats,
            "bytes_per_repeat": bytes_per_repeat,
            "total_bytes": bytes_per_repeat * repeats,
            "arithmetic_intensity_flops_per_byte": flops_per_repeat / bytes_per_repeat,
        },
        "measurements": {
            "native": measurement(10.0),
            "vendor": measurement(20.0),
            "vendor_over_native_speedup": 2.0,
        },
        "runtime": {
            "native": runtime,
            "vendor": vendor_runtime,
            "publication_d2h_operations": basis_count + 1,
        },
        "provenance": {
            "producer": "crates/sonium-qa/bin/compute_cuda_vendor_comparison.rs",
            "artifact": "cuda_vendor_comparison.json",
            "measurement": "resident CUDA submissions plus one synchronization",
        },
    }


def test_valid_contract_passes(checker, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact()), encoding="utf-8")
    assert checker.check(path)["status"] == "passed"


def test_timing_is_derived(checker, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact()), encoding="utf-8")
    assert checker.check(path)["status"] == "passed"
    mutated = artifact()
    mutated["measurements"]["native"]["throughput_flops_per_second"] *= 2.0
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="throughput"):
        checker.check(path)


def test_hidden_transfer_is_rejected(checker, tmp_path):
    mutated = copy.deepcopy(artifact())
    mutated["runtime"]["native"]["h2d_operations"] = 1
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="hidden transfer"):
        checker.check(path)
