# sonium-fft (lib: `sonium_fft`, version: 0.1.0)

Batched complex / real FFT and STFT on top of `sonium-compute`. Layer 1.5: sits between the compute-backend abstraction and the physics crates (`sonium-wave`, future BEM frequency sweeps, future STFT/spectrogram tooling).

## Key Components

- `FftPlanner<T>` (= `CpuFftPlanner`) — cached planner keyed by `(length, direction)`, shared across CPU and device-adapter plans. `plan` panics on bad lengths; prefer `try_plan` for untrusted sizes.
- `FftPlan<T>` trait — uniform interface for length-`N` plans: `process`, `try_process`, `process_with_scratch`, `try_process_with_scratch`, `scratch_len` (marked `#[must_use]`), `len`, `direction`, plus cache-section diagnostics. Buffer length must be a positive integer multiple of `len`; each chunk is one independent FFT.
- `StockhamFftPlan` — mixed-radix Stockham auto-sort for smooth lengths (factors of 2/3/4/5).
- `BluesteinFftPlan` — chirp-Z transform for arbitrary lengths; wraps a smooth-length Stockham FFT internally. `BluesteinPlan::process` validates buffer/scratch with release-mode `assert!` (not `debug_assert!`); `try_new` reports overflow as `PlanLengthError`.
- `SixStepFftPlan` — Bailey six-step for `N > 64K` (and directly constructible for any `N`); Cooley-Tukey twiddles are precomputed at plan time.
- `RealFftPlan` / `RealFftPlanner` — half-spectrum real packing (odd lengths fall back to full complex plans).
- `DctPlan` / `DstPlan` + free `dct2`/`dct3`/`dst2`/`dst3` + adjoints — two-tier: direct O(N²) reference at or below length 64, O(N log N) FFT kernel above.
- `Fft2dPlan` / `Fft3dPlan` + `process_strided`, `StftPlan` / `StftConfig` / `WindowFunction` (Rectangular/Hann/Hamming/Blackman with WOLA), `LinearConvolutionPlan` / `CircularConvolutionPlan`.
- Backend adapters (`backend.rs`): `BackendFftPlanner`, `DeviceFftPlan` (+ strided + cancellable), `BackendRealFftPlanner`, `BackendDctPlanner`, `BackendDstPlanner`, `FftExecution::{CpuNative,BackendNative,HostStagedFallback,VendorNative}`, `FftCancellationToken`.
- `rustfft_compat::{FftPlanner, Fft, FftNum, Length, Direction}` — partial shim mirroring `rustfft` 6.4+. Covers the in-place / out-of-place / immutable mutator surface, scratch-length queries, and the `Length`/`Direction` supertraits on `Fft`. Algorithm-specific concrete types (`Radix4`, `MixedRadix`, `BluesteinAlgorithm`, …) and `realfft` are out of scope. CPU-only by construction. Panics on empty or short buffers to match rustfft semantics — see "API contract divergence" below.
- `FftScalar` trait — implemented for `f32` and `f64` (`Float + FromPrimitive + Send + Sync + 'static`). Buffer element is always `Complex<T>`.

## Layout

```
src/
  lib.rs           public re-exports + crate docs (DFT convention, batching)
  direction.rs     FftDirection { Forward, Inverse }
  scalar.rs        FftScalar trait (f32, f64)
  plan.rs          FftPlan<T> trait (+ FftPlanError checked contract)
  planner.rs       FftPlanner facade (re-exports CpuFftPlanner)
  factor.rs        plan_length(n) → MixedRadix | Bluestein; try_plan_length,
                   try_next_pow2_pub (checked overflow reports LengthOverflow)
  twiddles.rs      TwiddleTable<T>, CompressedTwiddleTable, StageTwiddles
                   (per-stage contiguous twiddles; plan-time sincos via
                   sonium-fastmath through sincos.rs)
  sincos.rs        fill_sincos batching helper
  codelets/
    mod.rs         module re-exports
    scalar.rs      radix-{2,3,4,5} butterflies, #[inline(always)], generic <T>
  stockham.rs      Stockham auto-sort driver (run_single, run_batched, fused
                   audio/pair paths, radix-2/3/4/5 stages, SIMD dispatch)
  bluestein.rs     BluesteinPlan: chirp + symmetric kernel + inner FFTs
                   (new/try_new constructors)
  six_step.rs      Bailey six-step (plan-time Cooley-Tukey twiddle table)
  real.rs          RealFftPlan/RealFftPlanner (half-spectrum packing)
  dct.rs           DCT-II/III + DST-II/III + adjoints; DctPlan/DstPlan with
                   direct-vs-FFT dispatch (threshold 64)
  nd.rs            Fft2dPlan/Fft3dPlan, process_strided
  stft.rs          StftPlan/StftConfig/WindowFunction (WOLA ISTFT)
  convolution.rs   LinearConvolutionPlan/CircularConvolutionPlan
  backend.rs       device adapters (Metal/CUDA/vendor execution classes)
  wide_avx2.rs     x86_64 AVX2/FMA stages (pair paths)
  wide_neon.rs     aarch64 NEON stages (pair paths)
  wasm_simd.rs     wasm128 stages
  cpu/
    mod.rs         module re-exports
    plan.rs        StockhamFftPlan, BluesteinFftPlan (FftPlan impls);
                   PlanScratch spinlock + threshold-gated scoped-thread batch
                   parallelism (parallel_workers/split_batch)
    planner.rs     CpuFftPlanner (RwLock<HashMap<…>> cache; builds outside
                   the lock; six-step above 64K)
  rustfft_compat/
    mod.rs         FftPlanner/Fft/FftNum drop-in adapter
benches/
  fft_vs_rustfft.rs    Criterion bench: single + batched, vs rustfft 6.x
  fft_vs_accelerate.rs Criterion bench vs Accelerate (Apple targets)
tests/
  correctness_vs_rustfft.rs  parity vs rustfft: f64 1..=64 + medium + primes;
                             f32 1..=32 + medium + primes; Bluestein batch > 1
  batch_parallelism.rs       parallel-vs-serial differentials + naive-DFT spot
                             checks + concurrent shared-plan test
  edge_cases.rs              n=1, empty buf, misaligned len, process_with_scratch parity, etc.
  planner_contracts.rs       try_* typed errors, empty no-op, overflow, adjoint pair
  precision_codelets.rs      ulp-tight radix-3, radix-5; 5⁵ nested
  no_alloc_execution.rs      zero-heap-block steady-state gate (serial paths)
  dispatch_contract.rs       CpuFftDispatch policies
  transform_family_contracts.rs  real/nd/stft/dct/convolution family contracts
  fftw_reference.rs          optional runtime-loaded system FFTW (skips cleanly)
  independent_reference.rs   high-precision DFT cross-checks
  rustfft_compat_contract.rs compat surface parity
```

## Algorithm choice

`factor::plan_length(n)` is deterministic:

- `n == 1` → `MixedRadix(vec![])` (identity, no work).
- All prime factors are in `{2, 3, 4, 5}` → `MixedRadix(radices)`, radix-4 picked before radix-2 to halve the twiddle count for power-of-2 sizes. Lengths containing 7 or 8 (e.g. 7, 14, 28, 56) are **not** smooth and go through Bluestein.
- Otherwise → `Bluestein { padded_len: next_pow2(2n - 1) }`. The padded length is always a product of 4s and at most one 2, so the inner FFT is smooth.

Non-smooth policy is deliberately Bluestein-only: no radix-7/8 codelets and no Rader prime path (Rader would win where `N−1` is smooth, but the added planner/codelet surface is deferred — see Roadmap). `CpuFftPlanner` additionally routes `N > 64K` to `SixStepFftPlan`. DCT/DST plans use direct O(N²) summation at or below length 64 and one length-N complex FFT plus O(N) shuffles above.

Direction is folded into the twiddle table at plan construction and additionally into a `dir_sign: T` constant (`-1` forward, `+1` inverse) threaded through the codelets so the intrinsic radix-N roots of unity also respect the direction. Inverse plans are **unnormalised** — multiply the output by `1/N` for the round-trip identity, matching `rustfft` / NumPy semantics.

## Codelet contract

```rust
#[inline(always)]
fn radixN<T: FftScalar>(x0..xN, dir_sign: T) -> (y0..yN);
```

Codelets are pure functions over `Complex<T>`. Inter-stage twiddle multiplications (`x_k *= w_N^(l·k·s)`) happen in the Stockham driver before the codelet is called, so codelets see "ready" inputs. Internal radix-N roots of unity (`w_r^k`) are precomputed constants in the codelet body — they round to the nearest representable `f64` (17 sig digits), and clippy's `excessive_precision` lint is deliberately suppressed for this reason. See the comment block at the top of `src/codelets/scalar.rs`.

## Performance Philosophy

FFT for the sizes we care about (≲ 64K) is **memory-bandwidth bound**, not compute-bound. The design priorities are:

1. **Minimise DRAM passes.** Stockham auto-sort fuses bit-reversal into the stage indexing → one pass per stage and no separate permutation kernel.
2. **Keep twiddles in L1.** Twiddle-table compression to a quadrant cuts L1 footprint 4×; the dominant win at N = 1k–8k where the table competes with data. Six-step Cooley-Tukey twiddles are likewise precomputed once at plan time (no per-element trig in steady state).
3. **Vectorise across the batch dimension.** When `batch ≥ SIMD width`, the same scalar butterfly graph operates on B complex lanes — twiddles are loaded *once per stage* and broadcast across all batches (NEON / AVX2+FMA / wasm128 pair paths; single-batch stays scalar since one complex lane underfills a vector).
4. **Section-cache-aware stage grouping** (from OpenFFT, ICS '23). Group consecutive Stockham stages into "sections" sized so the intermediate working set stays L1/L2-resident across multiple butterfly passes.
5. **Threshold-gated batch parallelism, no rayon.** The crate has no thread-pool dependency by design: below 8 chunks or 65_536 elements the serial loop wins (thread spawn costs more than microseconds of FFT). Above both gates, owned-scratch `process` splits the batch across `std::thread::scope` workers with per-thread scratch, bypassing the shared `PlanScratch` spinlock / six-step mutex. `process_with_scratch` always stays serial on caller-owned scratch. The parallel path may allocate per-worker scratch; the serial path is allocation-free.
6. **Algorithmic correctness first.** For arbitrary `N`, Bluestein with a power-of-two inner length is the practical path; we don't try clever Rader-of-Rader chains. For smooth `N`, mixed-radix Stockham with codelets 2/3/4/5 covers ~99% of real-world FFT lengths used in acoustics (`2^k`, `2^k · 3`, `2^k · 5`, …).

## Testing

```bash
cargo test -p sonium-fft --release
cargo check -p sonium-fft && cargo clippy -p sonium-fft
cargo bench -p sonium-fft --bench fft_vs_rustfft
```

Correctness is verified by:

- Parity against `rustfft` 6.x for every length 1..1024 in `f64`, lengths 1..32 plus medium smooth lengths and primes in `f32`.
- Parity against a naive O(N²) DFT for all radix codelets and the Stockham driver up to N = 1024, plus naive-DFT spot checks on large parallel batches.
- Parity against `rustfft` for Bluestein on primes 7…401, Bluestein batch > 1, and parallel-vs-serial differentials on large Stockham/Bluestein/six-step batches.
- Forward → inverse round-trip on smooth (Stockham) and prime (Bluestein) lengths; DCT/DST round-trips and adjoint identities on both sides of the direct/FFT threshold.
- `process_with_scratch` and `process_outofplace_with_scratch` match `process`.
- ulp-tight precision tests for radix-3 and radix-5 (single-stage error < 1e-14 vs `sin_cos` reference).
- Nested-radix compounding: 5⁵ = 3125 with five back-to-back radix-5 stages agrees with `rustfft` to < 1e-12 relative.
- FFT-based DCT/DST kernels agree with the direct O(N²) reference for even, odd, smooth, and prime lengths (f32 + f64).

## Roadmap

- **M1 — shipped.** Crate scaffold, scalar radix-2/3/4/5 codelets, Stockham auto-sort, Bluestein, CPU planner with cache, `rustfft`-compat adapter, `sonium-wave` migration.
- **M2 — shipped.** NEON/AVX2/wasm128 vector-across-batch stages; twiddle-table compression; vectorise-across-batch driver; real-input FFT via half-length complex; OpenFFT-style section-cache-aware stage grouping.
- **M3 — shipped (core).** Metal backend: device-resident Stockham stages behind `metal = ["sonium-compute/metal"]`. An MPSGraph fast path and broader mixed-radix device coverage remain open.
- **M4 — shipped (core).** STFT (windowed batched FFT, Hann/Hamming/Blackman); six-step Bailey for `N > 64K`; `cuFFT` wrapper behind `cuda = ["sonium-compute/cuda"]`.
- **Next.** Rader prime path (wins where `N−1` is smooth; deferred against Bluestein-only policy); radix-7/8 codelets or a documented Bluestein-only guarantee; split-radix/tangent-FFT for pow2; full N/2-packed real path for odd lengths; half-precision (f16) path; vkFFT/wgpu portable-GPU evaluation; persistent-pool batch parallelism if per-call scoped threads ever show up in profiles.

Compute-layer additions proposed for `sonium-compute` (not strictly required for the CPU path but unblock GPU FFT and are reusable):

- `pointwise_mul<T>` — Hadamard / elementwise complex product. Critical for twiddle multiplications in six-step and the Bluestein chirp.
- `transpose_2d<T>` — out-of-place blocked transpose; used by six-step and any future dense LU.
- `scale_real<T>` — `x[i] ← alpha · x[i]` with `alpha: T::Real`. Lets FFT inverse normalization (1/N) skip a complex multiply per element.

## API contract divergence

The native [`FftPlan::process`] accepts a zero-length buffer as a no-op (treat batch count = 0 as "nothing to do"). `rustfft::Fft::process` panics on empty input. The `rustfft_compat` adapter restores the panic at its boundary so migrated call sites see identical behaviour to upstream rustfft. If you are writing new code against the native API, pre-filter empty inputs in the caller if you want strict rustfft semantics.

## Notes

- Inverse plans are unnormalised. Multiply by `1/N` for round-trip identity (matches `rustfft`, NumPy, SciPy).
- `BluesteinFftPlan::scratch_len()` is `2 · m` where `m = next_pow2(2n - 1)`. Allocate once and reuse via `process_with_scratch` in hot loops. The no-scratch `process` reuses owned plan scratch below the parallel gates and allocates per-worker scratch above them.
- `CpuFftPlanner` holds an `RwLock` cache and builds plans **outside** the lock: concurrent first-time construction of different shapes proceeds in parallel; a racing duplicate build is canonicalised on insert. Steady-state lookup is `Arc::clone` and is fast.
- `twiddles::TwiddleTable::at` is bounds-checked (`assert!(j < len)`); `StageTwiddles::k_slice` and `debug_assert_stage_shape` remain debug-only with documented caller preconditions. Every `transmute`-on-`TypeId` SIMD dispatch site first asserts identical size/alignment of the complex lanes and twiddle tables, so a future layout change fails loudly at the call site.
- Overflow policy is unified through `PlanLengthError`: `try_plan_length`, `try_next_pow2_pub`, `BluesteinPlan::try_new`, and `CpuFftPlanner::try_plan` report `LengthOverflow` where the panicking twins (`plan_length`, `next_pow2_pub`, `new`, `plan`) `expect`. New code should default to the `try_*` entry points (README examples do).
- Don't add an `f16` `FftScalar` impl until the half-precision codelets actually exist; `num_traits::Float` isn't implemented for `half::f16` in the way the codelets need.
- No emojis in code or documentation in this crate (workspace convention).
