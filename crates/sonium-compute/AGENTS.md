# sonium-compute (lib: `sonium_compute`, version: 0.1.0)

Compute backend abstraction with strict device-memory quotas. Provides the trait that `sonium-solvers`, `sonium-bem`, and `sonium-fem` depend on so they can target CPU, NVIDIA CUDA, or Apple Metal without changing call sites.

## Key Components

- `Backend` trait — BLAS-1/2/3, batched dense dispatch, CSR SpMV/SpMM,
  gather/cast, memory transfers, synchronization, and a machine-readable
  `BackendCapabilities` contract
- `MemoryPool` — quota-enforcing allocator, the only public path to a device buffer
- `DeviceScalar` — arithmetic scalars: f32, f64, Complex32, Complex64
- `BufferElement` — device-storable values, including storage-only f16/bf16
  and CSR index types
- Concrete backends: `CpuBackend`, `CudaBackend` (feature `cuda`), `MetalBackend` (feature `metal`)

## Layout

```
src/
  backend.rs       Backend trait + DeviceBufferLike
  pool.rs          MemoryPool, MemoryPoolConfig (quota enforcement)
  scalar.rs        BufferElement, DeviceScalar
  scaled_storage.rs f16/bf16 explicit-scale storage helpers
  error.rs         OomError (QuotaExceeded vs DriverOom)
  cpu/             Vec<T>-backed host buffers (always on)
                   blas2/, blas3/ — dispatch to sme2/ on aarch64
  sme2/            ARM SME2 dense BLAS micro-kernels (aarch64 only).
                   FMOPA + grouped 4-Z loads + cache blocking + B/A pack.
                   See src/sme2/README.md for the full design.
  cuda/            mod.rs        cfg-gated façade
                   cublas.rs     cudarc + cuBLAS/cuSPARSE (feature: cuda)
  metal/           objc2 Metal backend (feature: metal)
tests/
  parity.rs        Cross-backend output equivalence
```

## Trait Surface

Operations needed by Krylov solvers, preconditioners, dense frontal updates,
and matrix assembly are exposed on the `Backend` trait. FFT remains outside
this crate's public backend surface for now.

- BLAS-1: `axpy`, `dot` (Hermitian inner product), `nrm2`, `scale`
- BLAS-2/3: `gemv`, `gemm`, `gemv_batched`, `gemm_batched`
- Sparse: `prebake_csr`, `spmv_prebaked`, `spmm_prebaked_colmajor`, `spmv_csr`
- Solver helpers: `multi_dot`, `axpy_many`, `gather4_weighted`,
  `cast_to_lower`, `cast_from_lower`
- BEM assembly: `bem_regular_pair_c32_batch`
- Memory: `raw_alloc`, `raw_alloc_uninitialized`, `copy_h2d`, `copy_d2h`, `copy_d2d`, `synchronize`, `total_device_memory_bytes`

Call `Backend::capabilities()` to distinguish native device paths from
`HostSharedFallback`, `HostStagedFallback`, and `Unsupported` operations. CPU
reports native coverage for the full surface. CUDA reports cuBLAS/cuSPARSE as
`VendorNative` and custom kernels as `Native`. Metal reports native f32/c32
MSL/MPS coverage and explicit shared-storage CPU fallback for f64/c64.

## SME2 micro-kernels (aarch64)

`src/sme2/` is the production CPU BLAS path on Apple silicon and ARM
servers with `FEAT_SME2`. Three public surfaces:

- `sme2::gemm::{sgemm, dgemm}` — f32 / f64 GEMM. Pipeline: multi-tile
  FMOPA on ZA0..ZA3, SME2 grouped 4-Z `LD1{W,D}`, A panel pack (col-major
  in row-panel) + B panel pack (Nr-wide cols), adaptive Kc/Nc cache
  blocking against an auto-detected L2 budget, thread-local growth
  buffers (no per-call alloc).
- `sme2::gemm_bf16::gemm_bf16` — bf16 × bf16 → f32. Apple M4 BFMOPA
  pair-widening is broken in silicon (executes as single-lane), so this
  widens bf16 → f32 at pack time and routes through the f32 pipeline.
  Full asm `BFMOPA` kernel is preserved for the day pair-widening works.
- `sme2::gemv::{sgemv, dgemv}` — f32 / f64 GEMV with grouped 4-Z loads
  + 4-way accumulator unrolling.

Tuning: `SONIUM_L2_BUDGET_BYTES` overrides the auto-detected per-thread
L2 budget used by `choose_kc` / `choose_nc`. Auto-detect uses
`hw.perflevel0.l2cachesize / 2` on macOS, `sysconf(_SC_LEVEL2_CACHE_SIZE)
/ 2` on Linux, 8 MB fallback elsewhere.

Performance on Apple M4 Pro (single-threaded vs Accelerate): 3–9× faster
for GEMM, 10–27× faster for GEMV. See `src/sme2/README.md` for the full
benchmark table and design notes.

## Memory Discipline

- Buffers always come out of `MemoryPool::try_alloc`, never from `Backend::raw_alloc` directly.
- Default cap is 80 % of `total_device_memory_bytes()`; explicit cap required when discovery returns `None`.
- Quota violations fail fast with `OomError::QuotaExceeded` before the driver OOM kills the process.

## Features

- `cpu` (default) — host backend
- `cuda` — mature NVIDIA path: `dep:cudarc` (cuBLAS Level-1 + cuSPARSE generic SpMV via dynamic-loaded FFI). Supports `f32` / `f64` / `Complex32` / `Complex64`.
- `metal` — `dep:metal`, macOS only at runtime
- `vulkan` — removed (was a reserved no-op flag; requesting it fails the build)

## Testing

```bash
cargo test -p sonium-compute --lib
cargo test -p sonium-compute --features cuda      # mature path; requires libcuda.so + libcublas.so + libcusparse.so at runtime
cargo test -p sonium-compute --features metal     # macOS only
cargo check -p sonium-compute && cargo clippy -p sonium-compute
```

## Benchmarks

Each level has its own `cargo bench` target that pits the in-house kernel
against `oxiblas-blas`, the platform vendor BLAS (Apple Accelerate /
OpenBLAS), and cuBLAS (when the CUDA feature is on). Comparators that
aren't relevant for the active target/feature set silently drop out at
`#[cfg]` time.

```bash
# Default (sonium + oxiblas + accelerate on macOS).
cargo bench -p sonium-compute --bench blas1_compare
cargo bench -p sonium-compute --bench blas2_compare
cargo bench -p sonium-compute --bench blas3_compare

# Linux: opt in to libopenblas linkage at link time.
cargo bench -p sonium-compute --bench blas2_compare --features bench-openblas

# Add cuBLAS rows (requires libcuda + libcublas at runtime). The bench
# pre-stages buffers on the device, so reported figures are kernel +
# stream sync only — no h2d cost.
cargo bench -p sonium-compute --bench blas3_compare --features cuda
```

`tests/quota.rs` exercises `MemoryPool` accounting and refund semantics.
`tests/parity.rs` enforces CPU/CUDA/Metal output equivalence on BLAS-1,
SpMV/SpMM, dense BLAS, batched dense dispatch, gather/cast helpers, graph
capture, and capability contracts. GPU-gated tests skip with a visible
message when the driver is absent; set `SONIUM_REQUIRE_CUDA=1` (or
`SONIUM_REQUIRE_METAL=1` for Metal) to turn that skip into a failure, and
keep that fail-instead-of-skip behavior for forced-backend runs.

## Notes

- CPU BLAS-1 dispatches to NEON/SME2 on aarch64, AVX2/FMA on x86_64 when
  available, and scalar/autovectorized fallbacks elsewhere. BLAS-2 routes
  through `cpu::blas2` (NEON on aarch64, AVX2/FMA complex and real kernels
  on x86_64, scalar elsewhere); BLAS-3 routes through `cpu::blas3`
  (`sme2/` on aarch64, AVX2/FMA kernels on x86_64, Accelerate on macOS for
  eligible shapes, scalar elsewhere). Do not claim every non-ARM dense path
  is scalar. CSR SpMV caches row-bucket inspector metadata and real rows
  use explicit AVX2 gathers on x86_64 when available. Benchmark each
  real/complex, transpose, and size regime against OpenBLAS rather than
  assuming a scalar baseline.
- The NVIDIA `cuda` feature is compile-everywhere: cudarc resolves shared
  libraries at runtime, never at link time. Metal uses objc2 bindings and is
  macOS-only at compile time.
- Keep oxicuda usage benchmark-only. It is available as a dev-dependency for
  benchmark comparators, not as a production backend feature.
- Metal has native MSL kernels for f32/c32 BLAS-1, GEMV, CSR SpMV, and BEM regular-pair, plus MPS f32 GEMM. f64/c64 ops fall back to CPU compute over shared-storage buffers (Apple Silicon Metal compiler rejects `double`).
