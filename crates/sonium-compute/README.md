# sonium-compute

Compute backend abstraction with strict device-memory quotas. Sits below `sonium-solvers`, `sonium-bem`, and `sonium-fem` so they can target CPU, NVIDIA CUDA, or Apple Metal through the same trait.

## Installation

```toml
[dependencies]
sonium-compute = { version = "0.1", default-features = false, features = ["cpu"] }
```

## Architecture

```
Backend (trait)            BLAS-1/2/3 + CSR SpMV/SpMM/transpose + gather/cast +
                           H2D/D2H/D2D + synchronize
   │
   ├── CpuBackend          Vec<T>-backed host buffers (always available)
   ├── CudaBackend         feature `cuda`: cudarc + cuBLAS / cuSPARSE
   └── MetalBackend        objc2-metal + MSL/MPS f32/c32 acceleration,
                           shared-memory CPU fallback for f64/c64

MemoryPool                 Quota-enforcing allocator, the only public path
                           to a device buffer. Fails fast with
                           OomError::QuotaExceeded before driver OOM kills
                           the process.
```

`Backend::raw_alloc` is the unsafe lower-level entry point used by `MemoryPool`. Upper crates must always go through `MemoryPool::try_alloc`.

## Feature Flags

- `cpu` (default) — Host backend backed by `Vec<T>`.
- `cuda` — NVIDIA CUDA backend via `cudarc` (cuBLAS Level-1 + cuSPARSE generic SpMV through dynamic-loaded FFI). Requires `libcuda.so` + `libcublas.so` + `libcusparse.so` at runtime; compiles on any host (dynamic loading). Supports `f32`, `f64`, `Complex32`, `Complex64`.
- `metal` — Apple Metal backend via `objc2-metal`/MPS. macOS only at runtime. Device alloc, h2d, d2h, d2d, and synchronize go through real Metal. BLAS-1, GEMV, CSR SpMV, and BEM regular-pair have native f32/Complex32 MSL paths; large f32/Complex32 GEMM uses MPS. f64/Complex64 intentionally run on the CPU over shared-storage buffers because Apple GPU double precision is not available to MSL.
- `vulkan` — removed (was a reserved no-op flag). Requesting it fails the
  build; it returns when a backend exists.

`oxicuda` is not a `sonium-compute` backend feature. It is retained only as a
benchmark/dev comparator for harnesses that explicitly depend on the
pure-Rust CUDA stack.

## Auto backend selection

Upper-layer dispatchers (e.g. `sonium_fem::physics::stokes::navier_stokes::LinearSolverBackend::Auto`) resolve to the best backend for the host:

| Build | Resolves to |
|---|---|
| macOS + `--features metal` | `Metal` |
| Linux/Windows + `--features cuda` | `Cuda` (mature cudarc path) |
| Otherwise | `Cpu` |

Pure cfg dispatch — no runtime probing. If the chosen backend's `new()` later fails (no driver, no GPU), the dispatcher falls back to CPU at solve time with a warning. This is the recommended way to write code that "just works" on a developer's macOS laptop and a Linux GPU box without per-target plumbing.

## Trait Surface

The `Backend` trait exposes the operations currently used by Krylov solvers,
preconditioners, and matrix assembly:

- BLAS-1: `axpy`, `dot`, `nrm2`, `scale`
- BLAS-2/3: `gemv`, `gemm`, `gemv_batched`, `gemm_batched`
- Sparse: `prebake_csr`, `spmv_prebaked`, `spmm_prebaked_colmajor`, `spmv_csr`,
  checked CSR transpose
- Solver helpers: `multi_dot`, `axpy_many`, device-resident basis AXPY,
  `gather4_weighted`, `cast_to_lower`, `cast_from_lower`
- Memory: `raw_alloc`, `raw_alloc_uninitialized`, `copy_h2d`, `copy_d2h`, `copy_d2d`, `synchronize`, `total_device_memory_bytes`

Every backend also reports a `BackendCapabilities` table. Capability states are
`Native`, `VendorNative`, `HostSharedFallback`, `HostStagedFallback`, or
`Unsupported`, so fallback behavior is explicit and testable.

## Current Capability Summary

| Operation | CPU | CUDA | Metal |
|---|---|---|---|
| BLAS-1 f32/f64 | Native | VendorNative | f32 Native, f64 HostSharedFallback |
| BLAS-1 Complex32/64 | Native | VendorNative | c32 Native, c64 HostSharedFallback |
| GEMV | Native | VendorNative | f32/c32 Native, f64/c64 HostSharedFallback |
| GEMM | Native | VendorNative | f32/c32 VendorNative, f64/c64 HostSharedFallback |
| CSR SpMV/SpMM | Native | VendorNative | SpMV f32/c32 Native, f64/c64 HostSharedFallback; SpMM HostSharedFallback |
| CSR transpose | Native | VendorNative (cuSPARSE CSR-to-CSC) | HostStagedFallback |
| Cast/gather/multi-dot | Native | Native when custom kernels load, otherwise visible fallback | f32/c32 gather + multi-dot Native; casts HostSharedFallback; f64/c64 gather HostStagedFallback and multi-dot HostSharedFallback |
| BEM regular-pair c32 | Native | Native when custom kernel loads, otherwise HostStagedFallback | Native |
| BEM H-matrix/FMM setup | Native | Complex64 HostSharedFallback (tree/compression is host-owned); other scalars Unsupported | HostSharedFallback where the CPU reference supports it |
| BEM H-matrix/FMM actions | Native | Complex64 Native when FMM helper module loads; other scalars Unsupported | Unsupported |
| BEM field evaluation | Native | Constant Tri3/Complex64 Native when the BEM module loads; other cells Unsupported | Unsupported |

The CUDA Complex64 hierarchical action adapter is exposed as
`CudaHMatrixC64`.  `HMatrix::try_to_cuda` uploads dense near-field blocks,
low-rank factors, index maps, and reusable projection scratch once; forward,
transpose, and Hermitian actions then remain on the device. H-matrix/FMM tree
construction, admissibility, and compression are still host-side and are
reported separately as `hmatrix_setup`/`fmm_setup` capability cells.
`CudaBemFieldOperatorC64` uploads
constant Tri3 geometry and Cauchy traces once, then reuses them for
device-resident field actions; Quad4, higher-order, and singular-on-surface
field cells remain explicit CPU/unsupported boundaries.

## Metal Dispatch Policy

Metal is treated as the f32/Complex32 accelerator. f64/Complex64 operations run
on the CPU over shared-storage buffers unless the capability table says
otherwise, because Apple GPU double precision is not exposed to MSL.

For f32 GEMM, large regular matrices use `MPSMatrixMultiplication`; small
matrices route to the CPU/SME2 path over shared memory to avoid MPS dispatch
overhead. Complex32 GEMM follows the same small-shape policy and uses the
4-real-MPS prototype for large shapes.

For f32 GEMV, the production path stays on Sonium's custom MSL kernels. A
short local run on 2026-05-19 showed MSL faster for 64x64 and 256x256 NoTrans,
roughly tied at 1024x1024, and only slightly behind MPS at 4096x1024. MPS Trans
GEMV is only faster when `A^T` is already prepacked as the physical row-major
input; normal `Backend::gemv(Trans)` does not get that layout for free.

## Generic Element Types

`DeviceScalar` covers the arithmetic types that backend BLAS operates on:
`f32`, `f64`, `Complex32`, and `Complex64`. `BufferElement` is broader and also
allows storage-only values such as CSR indices plus `half::f16` and `half::bf16`.

For far-field and low-rank blocks that want lower-precision storage without
pretending f16/bf16 are BLAS scalars, use the explicit-scale helpers:
`quantize_scaled_f32`, `dequantize_scaled_f32`,
`quantize_scaled_complex32`, and `dequantize_scaled_complex32`.

## Memory Pool

```rust
use sonium_compute::{CpuBackend, MemoryPool, MemoryPoolConfig};

let backend = CpuBackend::new();
let pool = MemoryPool::new(backend, MemoryPoolConfig::default());
let buf = pool.try_alloc::<f64>(1024).expect("under quota");
```

The default cap is 80 % of `total_device_memory_bytes()`. Pass an explicit cap when discovery is impossible.

## Testing

```bash
cargo test -p sonium-compute --lib
cargo check -p sonium-compute && cargo clippy -p sonium-compute
cargo test -p sonium-compute --features cuda
cargo test -p sonium-compute --features metal
```

Backend parity tests (CPU vs CUDA vs Metal) live in `tests/parity.rs` and run
only when the corresponding feature is enabled. By default accelerator tests
skip when the backend cannot initialize; set `SONIUM_REQUIRE_CUDA=1` or
`SONIUM_REQUIRE_METAL=1` in CI jobs that are expected to have the device. The
Gitea accelerator workflow uses runner labels `[self-hosted, linux, cuda]` and
`[self-hosted, macos, metal]` for the required jobs.

The native CPU BEM evidence test is compiled for both `x86_64` and `aarch64`.
It records the selected SIMD kernel, compiler/CPU metadata, and deterministic
BLAS/BEM observables in schema-2 JSON when `SONIUM_OUTPUT_DIR` is set. The
accelerator workflow produces the accepted native-aarch64 artifact first, then
sets `SONIUM_AARCH64_REFERENCE` on the native x86_64 job; the latter fails if
the observable shape, declared BEM cells, or numerical difference (maximum
absolute error `5e-5`) does not match. Rosetta is not used for either lane.

## Phase 1 Bandwidth Instrumentation

Phase 1 roofline instrumentation is exposed through:

- Shared byte/flop models in `src/perf.rs` for copy, scale, axpy, dot, gemv,
  gemm, cast, gather4, CSR SpMV, and BEM regular-pair.
- Benchmark-side normalized reporting (`bytes`, `flops`, arithmetic intensity,
  host/device metadata) in `benches/perf_report.rs`.
- Cross-backend preflight in `benches/roofline_preflight.rs` for copy, device
  copy, scale, axpy, dot, gemv, cast, gather4, CSR SpMV, BEM regular-pair
  (CUDA/Metal), and synchronization overhead.

Run:

```bash
cargo bench -p sonium-compute --bench roofline_preflight
```

`Criterion` throughput output provides achieved bytes/s; `sonium_perf_model`
and `sonium_roofline_preflight` lines provide the operation model used for
roofline interpretation.

CUDA solve-level traces can also use `CudaBackend::runtime_stats()` and
`reset_runtime_stats()`. The returned `CudaRuntimeStatsSnapshot` combines
custom-kernel launches, cuBLAS/cuSPARSE/cuSOLVER submissions, synchronization
and completion counts, checked transfer counters, and allocation/staging
watermarks. This is lightweight in-process evidence; Nsight remains the
source of hardware throughput and occupancy counters.

## CUDA Profiling Recipes

Use the CUDA feature with required-device mode when collecting NVIDIA traces so
fallbacks fail loudly:

```bash
SONIUM_REQUIRE_CUDA=1 cargo test -p sonium-compute --features cuda
SONIUM_REQUIRE_CUDA=1 cargo bench -p sonium-compute --features cuda --bench roofline_preflight
```

For Nsight Compute, start with the roofline preflight and the BLAS comparison
benches. These cover memory bandwidth, launch/sync overhead, cuBLAS BLAS-1/2/3,
cuSPARSE CSR SpMV, custom gather/cast/multi-dot kernels, and BEM regular-pair:

```bash
ncu --target-processes all --set full \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
  cargo bench -p sonium-compute --features cuda --bench roofline_preflight

ncu --target-processes all --set full \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
  cargo bench -p sonium-compute --features cuda --bench blas2_compare -- gemv_f64

ncu --target-processes all --set full \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
  cargo bench -p sonium-compute --features cuda --bench blas3_compare -- gemm_f64
```

Record achieved occupancy, DRAM throughput, SM throughput, tensor-core use when
applicable, and the matching `sonium_perf_model` / `sonium_roofline_preflight`
line for each run. Compare cuBLAS/cuSPARSE paths against custom kernels by
shape rather than by peak hardware numbers alone.

The release CUDA BEM/FEM evidence producers normalize their measured
FLOP/s against two runner-provisioned vendor baselines:
`SONIUM_QA_CUDA_PEAK_FLOPS` (FLOP/s) and `SONIUM_QA_CUDA_MEMORY_BANDWIDTH`
(bytes/s).  The accelerator workflow rejects missing or non-positive values,
and `qa-release-accelerators nvidia-cuda` runs
`benchmarks/summarize_cuda_evidence.py` against both raw workload artifacts.
The resulting roofline fraction is therefore derived from the timed workload;
an unavailable baseline cannot be promoted to a release pass.

## Checked-In Measured Evidence

`benchmarks/artifacts/gemm-vendor-v1.json` (+ `.sha256` sidecar) is the
bounded GEMM vendor supplement to the release CUDA evidence: single-thread
warm medians for f32/f64 NoTrans/NoTrans sonium AVX2+FMA vs system
OpenBLAS (16 cells), plus f32 CUDA `CudaBackend::gemm` vs raw
`cublasGemmEx` wrapper cells with per-cell precision preflights (12
cells). Every cell carries median + 95% CI, workload identity, toolchain
provenance, and its precision basis; the artifact claims no universal
superiority, cold latency, or energy. Reproduce with:

```bash
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo bench -p sonium-compute --bench blas3_compare \
  --features bench-openblas -- gemm_f --noplot
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo bench -p sonium-compute --bench blas3_compare \
  --features bench-openblas,cuda -- gemm_f32/cublas --noplot
python3 crates/sonium-compute/benchmarks/build_gemm_vendor_evidence.py
```

What stays accounting-only by design: the C6 `microkernel_roofline`
models (`perf.rs` formulas, no Nsight attribution) and the C6 BEM/FEM
workload observations (volatile fields, excluded from the canonical
payload). Full real/complex/transpose/size coverage rides the
accelerator CI lanes, which upload per-run criterion evidence.

## Metal Profiling Recipes

Use required-device mode on Apple Silicon so skip-friendly local tests cannot
hide a missing Metal run:

```bash
SONIUM_REQUIRE_METAL=1 cargo test -p sonium-compute --features metal
SONIUM_REQUIRE_METAL=1 cargo bench -p sonium-compute --features metal --bench roofline_preflight
```

For Instruments, use the **Metal System Trace** template and start with the
GEMV/GEMM comparison benches plus the roofline preflight:

```bash
cargo bench -p sonium-compute --features metal --bench blas2_compare -- 'gemv_f32/(metal|metal_mps)'
cargo bench -p sonium-compute --features metal --bench blas2_compare -- 'gemv_f32_trans/(metal|metal_mps_packed_trans)'
cargo bench -p sonium-compute --features metal --bench blas3_compare -- gemm_f32
cargo bench -p sonium-compute --features metal --bench roofline_preflight
```

In the trace, verify that native f32/Complex32 BLAS-1, GEMV, SpMV, gather, and
multi-dot calls append to a pending command buffer and only block at explicit
`synchronize`, scalar reductions, `copy_d2h`, or shared-memory CPU fallback
boundaries. Treat unexpected per-op `waitUntilCompleted` events inside Krylov
loops as regressions.

## Notes

- The CPU backend owns production BLAS-1, dense BLAS-2/3, CSR SpMV, and CSR
  SpMM paths. On aarch64 it uses NEON/SME2 where available. On x86_64 BLAS-1
  runtime-dispatches to AVX2/FMA kernels when supported, and real CSR row
  reducers use AVX2 gathers when available; non-aarch64/non-x86_64 targets use
  scalar/autovectorized fallbacks.
- The `cuda` feature resolves NVIDIA shared libraries at runtime, so it
  compiles on hosts without a GPU. Keep oxicuda usage in benchmarks only.
