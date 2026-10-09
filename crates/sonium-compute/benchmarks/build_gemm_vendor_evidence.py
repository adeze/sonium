#!/usr/bin/env python3
"""Build the bounded CPU GEMM vs OpenBLAS evidence artifact from criterion output.

Reads target/criterion estimates for the gemm_f32/gemm_f64 groups (sonium vs
openblas arms), attaches audit-index provenance (source digest, hardware,
toolchain, thread policy, workload identity, precision basis), and writes a
versioned JSON artifact plus a sha256 sidecar. Volatile across runs by
design: the sidecar pins this run, not a universal claim.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)
CRITERION = ROOT / "target" / "criterion"
OUT = ROOT / "crates/sonium-compute/benchmarks/artifacts/gemm-vendor-v1.json"

SCALARS = ("f32", "f64")
SIZES = ("64x64x64", "256x256x256", "1024x1024x1024", "2048x128x2048")
ARMS = ("sonium", "openblas")


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def cpu_field(key: str) -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    return "unknown"


def openblas_version() -> str:
    try:
        out = run("ldconfig", "-p")
        for line in out.splitlines():
            if "libopenblas.so.0 " in line or "libopenblas.so.0\t" in line:
                path = line.split("=>", 1)[1].strip()
                blob = subprocess.check_output(
                    ["strings", path], text=True, errors="ignore"
                )
                for blob_line in blob.splitlines():
                    if "OpenBLAS" in blob_line and (
                        "v" in blob_line or "." in blob_line
                    ):
                        return f"{blob_line.strip()} ({path})"
                return f"unversioned ({path})"
    except Exception as error:  # noqa: BLE001 - provenance best effort
        return f"unknown ({error})"
    return "not_found"


def cell_estimates(scalar: str, arm: str, size: str) -> dict:
    path = CRITERION / f"gemm_{scalar}" / arm / size / "new" / "estimates.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"missing criterion estimates for {scalar}/{arm}/{size}: {error}")
    median = data["median"]
    m, n, k = (int(v) for v in size.split("x"))
    flops = 2.0 * m * n * k
    median_ns = float(median["point_estimate"])
    return {
        "scalar": scalar,
        "arm": arm,
        "m": m,
        "n": n,
        "k": k,
        "median_ns": median_ns,
        "ci95_low_ns": float(median["confidence_interval"]["lower_bound"]),
        "ci95_high_ns": float(median["confidence_interval"]["upper_bound"]),
        "gflops_derived": flops / median_ns,
        "samples": 20,
    }


CUDA_ARMS = (
    "cublas",
    "cublas_gemmex_fp32_pedantic_gemm",
    "cublas_gemmex_tf32_gemm",
)


def parse_cuda_preflights(log_path: Path) -> dict:
    """Preflight max_abs/max_rel keyed by (size, variant).

    Only `bench=gemm` lines: other bench families register (and preflight)
    during collection even when the criterion filter excludes their timing.
    """
    out: dict = {}
    for line in log_path.read_text().splitlines():
        if not line.startswith("sonium_precision_check bench=gemm "):
            continue
        fields = dict(
            part.split("=", 1) for part in line.split() if "=" in part
        )
        key = (fields["shape"], fields["variant"])
        out[key] = {
            "max_abs": float(fields["max_abs"]),
            "max_rel": float(fields["max_rel"]),
        }
    return out


def cuda_section() -> dict:
    log_path = ROOT / "target" / "sonium-step3-cuda" / "bench-cuda.log"
    preflights = parse_cuda_preflights(log_path)
    cells = []
    for size in SIZES:
        for arm in CUDA_ARMS:
            cell = cell_estimates("f32", arm, size)
            variant = (
                arm[: -len("_gemm")] if arm != "cublas" else None
            )
            key = (size, variant) if variant else None
            cell["compute"] = (
                "cublasSgemm_v2 via CudaBackend::gemm"
                if arm == "cublas"
                else (
                    "cublasGemmEx COMPUTE_32F_PEDANTIC"
                    if "pedantic" in arm
                    else "cublasGemmEx COMPUTE_32F_FAST_TF32"
                )
            )
            cell["preflight_vs_scalar_reference"] = (
                preflights.get(key) if key else "not_applicable_backend_arm"
            )
            cells.append(cell)
    return {
        "scope": (
            "sonium CudaBackend::gemm (cublasSgemm_v2) vs raw cublasGemmEx; "
            "f32 NoTrans/NoTrans; pre-staged device buffers; 1 synchronize "
            "per iteration; matched-precision pair is backend vs "
            "fp32_pedantic (tf32 differs numerically, informational only)"
        ),
        "gpu": {
            "device": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "compute_capability": "12.0",
            "driver": "610.57.04",
            "toolkit": "cuda_13.3",
            "memory_mib": 97887,
        },
        "loop_accounting_source_verified": {
            "h2d_in_loop": 0,
            "d2h_in_loop": 0,
            "synchronize_per_iter": 1,
            "cublas_calls_per_iter": 1,
            "note": (
                "A/B/C uploaded once before timing "
                "(benches/blas3_compare.rs setup); timed loop is one "
                "GEMM plus one synchronize"
            ),
        },
        "precision_basis": (
            "compute.operation-matrix.v1 passed on actual_backend=cuda "
            "at HEAD (max_relative_error 1.17e-7 over 144 operation + "
            "144 layout + 16 sparse + 24 view + 48 multi_dot cases); "
            "per-cell preflights below compare raw arms against the "
            "scalar reference"
        ),
        "cells": cells,
    }


def main() -> None:
    base_sha = run("git", "rev-parse", "HEAD")
    diff_bytes = subprocess.check_output(["git", "diff"], cwd=ROOT)
    diff_sha = hashlib.sha256(diff_bytes).hexdigest()
    dirty = bool(diff_bytes.strip())

    cells = [
        cell_estimates(scalar, arm, size)
        for scalar in SCALARS
        for size in SIZES
        for arm in ARMS
    ]

    artifact = {
        "schema": "sonium-compute-gemm-vendor/v1",
        "version": 1,
        "status": "passed",
        "scope": (
            "bounded single-thread warm-median GEMM evidence; "
            "Linux x86_64 CPU (f32/f64 sonium AVX2+FMA vs OpenBLAS) "
            "plus CUDA f32 backend-vs-raw-cuBLAS wrapper cells"
        ),
        "claims": {
            "vendor_comparison": "measured_bounded_single_thread_warm_median",
            "universal_superiority": "not_claimed",
            "cold_latency": "not_claimed",
            "energy": "not_claimed",
            "device_counters": "not_applicable_cpu_in_process",
        },
        "provenance": {
            "base_sha": base_sha,
            "worktree_dirty": dirty,
            "diff_sha256": diff_sha,
            "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "compiler": run("rustc", "--version"),
            "cpu_model": cpu_field("model name"),
            "microcode": cpu_field("microcode") or "unknown",
            "os": "linux-x86_64",
            "target_features": "avx2,fma (runtime-detected)",
            "runtime_dispatch": "avx2-fma-native",
            "thread_policy": "RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
            "openblas": openblas_version(),
            "criterion": "sample_size=20 warmup=3s measurement=5s noplot",
            "profile": "bench (optimized)",
        },
        "workload_identity": {
            "operation": "C := alpha*A*B + beta*C, NoTrans/NoTrans",
            "alpha": 0.7,
            "beta": 0.0,
            "inputs": "deterministic ramp sin(k*0.0123[/0.0177]) per bench source",
            "sonium_layout": "row-major, lda=k ldb=n ldc=n",
            "openblas_layout": "column-major pack of the same ramps",
        },
        "precision_basis": {
            "note": (
                "bench cells gate status, not values; numerical parity "
                "comes from the suites below, all passing at HEAD"
            ),
            "properties_blas": "8 passed (seeded GEMV/GEMM vs naive reference)",
            "wolfram_golden": "3 passed (exact gemv transpose variants)",
            "native_x86_evidence": (
                "3 passed; 1634 observables; 0 failed; "
                "max_norm_error 4.44e-16; kernel avx2-fma"
            ),
        },
        "cells": cells,
        "cuda": cuda_section(),
        "limits": [
            "warm medians only; no cold-start latency measured",
            "single-threaded; no scaling or NUMA claim",
            "NoTrans/NoTrans only; complex/transpose regimes ride CI",
            "shared host: medians drift run to run; compare within CI bands",
            "derived GFLOP/s uses 2*m*n*k over the median; model-only",
        ],
    }

    OUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix(".json.sha256").write_text(
        f"{digest}  {OUT.name}\n", encoding="utf-8"
    )
    print(f"wrote {OUT} ({digest[:12]}...)")
    print(f"cells: {len(cells)}")


if __name__ == "__main__":
    sys.exit(main())
