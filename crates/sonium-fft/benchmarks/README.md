# CUDA FFT evidence

Build and run the vendor benchmark with:

```text
cargo run -p sonium-fft --features cuda --bin fft_cuda_benchmark \
  > fft-cuda-v1.json
python3 crates/sonium-fft/benchmarks/summarize_cuda_evidence.py fft-cuda-v1.json
```

The repeatable FFT2 acceptance runner is `just qa-fft2-cuda`. It uses the
repo-local `.target-unix-fft2` Cargo target and `target/sonium-unix-fft2`
evidence directory, requires a real CUDA device, runs the checked CUDA unit
matrix, then validates both the vendor timing summary and the emitted
row-level execution counts with `check_fft2_matrix.py`. The raw artifact's
`matrix` object distinguishes executed, skipped/unavailable, unsupported, and
failed rows; `required_cuda` and `fail_closed` are recorded explicitly.

The raw artifact is versioned (`schema_version: 1`) and contains the device,
driver, toolkit, clocks, target, numerical gate, cold plan cost, warmed
samples, empirical interval, transfers, synchronization counts, and allocation
watermarks. The timer begins after host-to-device staging and ends after stream
completion; transfers are recorded separately. A row is measured only when
`BackendFftPlanner` reports `VendorNative`, so host fallback cannot be counted
as CUDA throughput.

The checked-in `artifacts/cpu-x86_64-v1.json` records the native AVX2 CPU
matrix. `artifacts/cuda-v1-unavailable-host.json` is an honest fail-closed
probe from a host where `nvidia-smi` was visible but the CUDA context returned
`CUDA_ERROR_NO_DEVICE`; it is not a vendor timing baseline. A successful
hardware run should be saved as a new raw artifact alongside it.

The matrix includes Complex32/Complex64 smooth, prime/Bluestein, contiguous
batched, and strided batches, plus device-resident STFT framing and linear
convolution consumers. It also measures contiguous f32/f64 real FFT forward
plans and all DCT-II/III, DST-II/III, and Euclidean-adjoint operations. The
real FFT rows use cuFFT R2C/C2R directly; DCT/DST rows use a cached length-2N
cuFFT C2C work area with device pack/extract kernels. High-level real STFT,
multidimensional transforms, unsupported dtypes/layouts, and in-flight
cancellation remain explicit unsupported cells. Malformed views,
allocation/launch injection, and device-loss handling are covered by the
required CUDA unit suite.

Set `SONIUM_REQUIRE_CUDA=1` for a required hardware run. Without a usable CUDA
context the command emits an `unavailable` artifact and exits successfully for
local inspection; with the variable set it emits the same artifact and exits
non-zero. No CPU fallback is substituted.

# CPU artifact provenance

`fft_benchmark` (CPU) separates plan construction (`plan_ns`) from warmed
steady-state execution and records, per artifact: architecture, runtime
target features, dispatch, compiler, CPU description, crate
`package_version`, build `profile` (`release` unless built with debug
assertions), best-effort `git_commit` / `git_dirty` captured by `build.rs`
at compile time, caller `threads`, batches per case, the transform
convention, the correctness gate, and per-case scratch bytes. Allocation and
energy are reported as unavailable rather than inferred from timing.

Provenance rules:

- `git_commit: "unknown"` (or a missing field in older artifacts) means the
  source revision was not captured. It must never be read as a clean tree.
  Rebuild the binary after committing before recording a release artifact.
- The checked-in `artifacts/cpu-x86_64-v1.json` predates the
  `package_version` / `profile` / `git_commit` / `git_dirty` / `threads`
  fields and its build-profile/commit provenance is insufficient to infer
  current throughput. Refresh it on the pinned host instead of backfilling
  provenance by hand.
- CPU and CUDA artifacts use different shapes, batching, and timing
  protocols (host wall time versus staged device execution with separate
  transfer accounting). Do not divide one by the other for a CPU/GPU ratio.

# DCT/DST cost guard

The free `dct2` / `dct3` / `dst2` / `dst3` functions (and their Euclidean
adjoints) are direct O(N^2) summations at every length. `DctPlan` /
`DstPlan` use them at or below length `DIRECT_THRESHOLD` (64, published from
`sonium_fft`) and an O(N log N) FFT kernel above. Application code working
above the threshold must use the plans; `tests/dct_reference_guard.rs` pins
the threshold, the dispatch boundary, and plan-versus-reference agreement.
