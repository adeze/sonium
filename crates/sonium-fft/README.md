# sonium-fft

Batched complex and real FFT on top of `sonium-compute`. Designed to be faster than `rustfft` and Apple Accelerate / vDSP for batched workloads, with a partial `rustfft`-shaped adapter so existing call sites can migrate by changing only their `use` statement (provided they stay inside the supported API — see "What is and isn't covered" below).

## Transform families

The native CPU API exposes:

- `RealFftPlanner` / `RealFftPlan` for packed real half-spectra, including
  allocation-free batched methods.
- `dct2` / `dct3` and `dst2` / `dst3` with explicit inverse scaling, plus
  `DctPlan` / `DstPlan` (with Euclidean adjoints). The free functions are
  direct O(N^2) reference summations; the plans use them at or below length
  64 and an O(N log N) FFT kernel above (one length-N complex FFT plus O(N)
  pre/post shuffles for DCT-II, DCT-III, DST-II, and DST-III).
- `Fft2dPlan` and `Fft3dPlan` for row-major multidimensional transforms, plus
  `process_strided` for regularly strided vectors.
- `StftPlan` with rectangular, Hann, Hamming, and Blackman windows, explicit
  centering/end-padding, and weighted-overlap-add ISTFT reconstruction.
- `SixStepFftPlan` for large cache-aware decompositions and
  `BackendFftPlanner` for device buffers. The latter reports its per-plan
  execution class, including explicit host-staged fallback and available
  vendor-native paths, and supports checked regularly strided batches.
- `BackendRealFftPlanner`, `BackendDctPlanner`, and `BackendDstPlanner` for
  device-resident real transforms. CUDA uses cuFFT R2C/C2R for real FFT and
  cached length-2N cuFFT C2C plus pack/extract kernels for DCT/DST.
- Smooth Stockham plans also retain a cache-budgeted `StageSection` schedule;
  the generic stage path executes through those planner-owned groups without
  allocating section metadata on the hot path.

All complex FFTs use the unnormalised DFT convention (`forward` sign `-1`,
`inverse` sign `+1`). Real inverse and ISTFT apply their documented scales.

## Installation

```toml
[dependencies]
sonium-fft = { workspace = true }
```

## Architecture

```
FftPlanner<T>                Cached planner, keyed by (length, direction)
   │
   ├── StockhamFftPlan       Mixed-radix Stockham auto-sort for smooth lengths
   │                         (factors of 2, 3, 4, 5).
   │                           - radix-4 preferred for power-of-2 (halves twiddle count)
   │                           - codelets straight-line, #[inline(always)]
   │                           - radix-4-first ping-pong driver, no bit-reversal pass
   │
   └── BluesteinFftPlan      Chirp-Z transform for arbitrary lengths.
                             Inner length M = next_pow2(2N - 1) → smooth →
                             three internal mixed-radix Stockham passes.

rustfft_compat::FftPlanner   Drop-in shim over the native CpuFftPlanner.
                             Same API surface as rustfft 6.x.
```

`FftPlan::process` takes a buffer whose length is an integer multiple of `plan.len()`; each contiguous chunk is one independent FFT. `process_with_scratch` reuses caller-provided scratch to avoid per-call allocation in hot loops. The owned-scratch `process` path parallelises large batches (at least 8 chunks totalling at least 65_536 elements) across scoped threads with per-thread scratch, bypassing the shared plan scratch; smaller batches stay serial and allocation-free, as do all `process_with_scratch` calls. The `process_strided` helper uses the first `N` scratch elements as its gather buffer and forwards the remaining `plan.scratch_len()` elements to the one-dimensional plan, so prime/Bluestein batches remain allocation-free. The 2D/3D plans follow the same gather-plus-inner-work-region contract. The native CPU planner also reuses owned plan scratch for `process`, while the compatibility adapter preserves rustfft's stricter empty-buffer behavior.

## Algorithms

- **Mixed-radix Stockham** (smooth lengths). For each radix `r` in the factorisation of `N`, one stage applies `N/r` size-`r` butterflies with inter-stage twiddles `w_N^(l·k·s)` precomputed in a `TwiddleTable`. Stockham auto-sort means inputs come from one buffer and outputs go to the other — no bit-reversal scratch pass at the end. Output parity (even/odd radix count) is chosen so the final write lands in the user buffer.
- **Bluestein chirp-Z** (arbitrary lengths). Computes `X[k] = c[k] · Σ_j (x[j] · c[j]) · b[k − j]` where `c[j] = w^(j²/2)` and `b[d] = w^(−d²/2)`; the convolution is realised as `IFFT(FFT(A) · FFT(B))` on a power-of-two length so the inner FFTs run through the smooth-length Stockham kernel.

Direction is encoded in the twiddle table sign at planner construction and as a `dir_sign: ±1` constant passed into each codelet. Inverse plans are intentionally **unnormalised** — multiply the output by `1/N` yourself if you want the round-trip identity, matching `rustfft` and NumPy/SciPy conventions.

## Scalar Types

```rust
pub trait FftScalar { /* implemented for f32, f64 */ }
```

The buffer element type is always `num_complex::Complex<T>`. Complex-input plans support both `f32` and `f64`; scalar execution is the normative fallback, with architecture-specific NEON, AVX2, and wasm SIMD stages selected when available. Real FFT, STFT, multidimensional transforms, six-step execution, and explicit device adapters are also part of the current surface. `sonium-compute` Metal plans use device-resident radix-4-first Stockham Complex32 stages, with a residual radix-2 stage for odd powers of two. CUDA plans can use the optional dynamically-loaded cuFFT C2C/Z2Z path for regular and checked regularly strided batched Complex32/Complex64 lengths, including non-power-of-two lengths; unsupported precisions and unavailable vendor libraries report an explicit host-staged fallback.

## `rustfft` compatibility shim

```rust
// Before
use rustfft::{Fft, FftPlanner};

// After (no other changes for the common case)
use sonium_fft::rustfft_compat::{Fft, FftPlanner};

let mut planner = FftPlanner::<f64>::new();
let plan = planner.plan_fft_forward(1024);
plan.process(&mut buffer);
```

The shim is CPU-only; backend-aware planning uses the native `FftPlanner` instead.

### What is and isn't covered

The `rustfft_compat` module mirrors the *common* surface of `rustfft` 6.4+ — enough for the everyday "plan + process" workflow. Specifically:

| Covered | Not covered |
|---|---|
| `FftPlanner<T>` with `new`, `plan_fft`, `plan_fft_forward`, `plan_fft_inverse` | Algorithm-specific concrete types (`Radix4`, `MixedRadix`, `BluesteinAlgorithm`, …) |
| `Fft<T>` extending `Length` + `Direction` + `Send + Sync` | The `realfft` companion-crate `RealFftPlanner` |
| In-place: `process`, `process_with_scratch` | rustfft ≤ 6.3 trait shapes |
| Out-of-place: `process_outofplace_with_scratch` | |
| Immutable: `process_immutable_with_scratch` (6.4+) | |
| Scratch queries: `get_inplace_scratch_len`, `get_outofplace_scratch_len`, `get_immutable_scratch_len` | |
| `Length`, `Direction`, `FftNum`, `FftDirection` traits / types | |
| Panic-on-short-or-empty-buffer semantics (`buffer.len() < self.len()`, `buffer.len() % self.len() != 0`) | |

The native [`crate::FftPlan`] trait *does* accept an empty buffer as a zero-batch no-op — a deliberate divergence from `rustfft`. The `rustfft_compat` adapter panics at its boundary to preserve the rustfft contract.

## Native API (example)

```rust
use num_complex::Complex64;
use sonium_fft::{FftDirection, FftPlanner};

let planner = FftPlanner::<f64>::new();
// Prefer the checked entry points for externally supplied sizes: they report
// zero lengths and Bluestein-convolution overflow as typed errors instead of
// panicking. `plan` / `process` remain as panicking conveniences for trusted
// static sizes.
let plan = planner
    .try_plan(1024, FftDirection::Forward)
    .expect("power-of-two length must build");

// Batched: 16 contiguous length-1024 FFTs in one call, reusing
// caller-provided scratch so hot loops never allocate.
let mut buf = vec![Complex64::new(0.0, 0.0); 1024 * 16];
let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
plan.try_process_with_scratch(&mut buf, &mut scratch)
    .expect("batched buffer is a multiple of the plan length");

// Planner diagnostics for smooth Stockham schedules.
let _sections = plan.cache_section_count();
let _budget_bytes = plan.cache_section_budget_bytes();

// Reusable FFT-based linear convolution with explicit zero-padding.
use sonium_fft::LinearConvolutionPlan;
let convolution = LinearConvolutionPlan::<f64>::new(256, 129);
let lhs = vec![Complex64::new(0.0, 0.0); 256];
let rhs = vec![Complex64::new(0.0, 0.0); 129];
let mut output = vec![Complex64::new(0.0, 0.0); convolution.output_len()];
let mut scratch = vec![Complex64::new(0.0, 0.0); convolution.scratch_len()];
convolution.process_with_scratch(&lhs, &rhs, &mut output, &mut scratch);

// Circular convolution requires equal-length operands and returns N samples.
let circular = sonium_fft::CircularConvolutionPlan::<f64>::new(256);
```

## Performance

The checked-in Criterion benches compare single and batched transforms with rustfft, and the standalone artifact covers smooth, prime/Bluestein, strided, 2D/3D, and linear-convolution cases while recording plan time, warmed execution percentiles, scratch bytes, effective bandwidth, compiler, and CPU metadata:

```bash
cargo run --release -p sonium-fft --bin fft_benchmark
```

The following historical M1 measurements are retained for context only; they are not a current performance gate:

| N | sonium-fft | rustfft | ratio |
|---|---|---|---|
| 64 | 160 ns | 115 ns | 0.71× |
| 1 024 | 3.36 µs | 2.34 µs | 0.70× |
| 4 096 | 17.2 µs | 12.2 µs | 0.71× |
| 16 384 | 82.2 µs | 70.0 µs | 0.85× |
| 1 024 × 16 batched | 54.5 µs | 39.7 µs | 0.73× |

Use the emitted JSON and Criterion output for current comparisons. Energy and post-warmup allocation blocks are reported as unavailable unless the host provides dedicated instrumentation; the no-allocation contract is covered by `tests/no_alloc_execution.rs`.

## Roadmap

- **CPU shipped**. Scalar codelets, Stockham + Bluestein, real FFT, DCT/DST, 2D/3D transforms, STFT/ISTFT, linear and circular convolution, six-step planning, compressed twiddles, rustfft compatibility, CPU SIMD stages, and explicit device adapters.
- **Next CPU work**. Extend vector-across-batch coverage to remaining
  precision/architecture combinations, validate fully blocked intermediate
  sections on additional cache regimes. An optional native FFTW differential
  test dynamically loads FFTW when available and skips cleanly otherwise;
  independent high-precision DFT coverage is already checked in.
- **Accelerator work**. Metal now has device-resident radix-4-first/radix-2 Complex32 Stockham stages for power-of-two transforms; broader mixed-radix coverage and an MPSGraph fast path remain future work. CUDA has optional dynamically-loaded cuFFT C2C/Z2Z and R2C/C2R/D2Z/Z2D paths for regular batched transforms, plus device-resident DCT/DST kernels; unsupported precisions, missing cuFFT libraries, and unavailable backends remain explicit host-staged fallbacks.

See `crates/sonium-fft/AGENTS.md` for the design rationale and `/Users/pierre/.claude/plans/create-a-new-crate-mossy-balloon.md` for the full plan.

## Testing

```bash
cargo test -p sonium-fft --release
cargo check -p sonium-fft && cargo clippy -p sonium-fft
cargo bench -p sonium-fft --bench fft_vs_rustfft
cargo run --release -p sonium-fft --bin fft_benchmark
```

Correctness tests cover:

- Every length 1..1024 cross-checked against `rustfft` for `f64` (forward and inverse).
- Lengths 1..32, medium smooth lengths up to 2048, and primes up to 211 for
  `f32` (forward and inverse).
- Primes 7…401 via the Bluestein path (`f64`), plus Bluestein batch > 1 and
  large-batch parallel-vs-serial differentials against naive-DFT spot checks.
- Optional FFTW differential coverage for smooth and prime lengths when a
  system FFTW library is installed (`tests/fftw_reference.rs`).
- Non-smooth composites (14, 21, 22, 26, 33, 35, 49, 55, 77, 91).
- Batched dispatch (each chunk independent).
- Edge cases: `n = 1` identity, empty buffer no-op, misaligned length panics, `process_with_scratch` parity, `process_outofplace_with_scratch` parity, planner caching, ulp-tight codelet precision (radix-3, radix-5), 5⁵ = 3125 nested-radix compounding.

## References

- OpenFFT: An Adaptive Tuning Framework for 3D FFT on ARM Multicore CPUs (ICS '23). dl.acm.org/doi/10.1145/3577193.3593735
- Bluestein, L. I. (1970). A linear filtering approach to the computation of the discrete Fourier transform.
- Cooley, J. W., Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series.
- Stockham, T. G. (1966). High-speed convolution and correlation.
