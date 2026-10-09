# sonium-fastmath benchmarks

## Producer and gate

- `cargo run --release -p sonium-fastmath --bin fastmath_benchmark`
  emits one JSON object comparing the dispatched slice path, the scalar Rust
  path, and platform `f64::sin_cos` across five regimes (`reduced`,
  `coherent_ramp`, `random_full_range`, `reduction_boundaries`,
  `payne_hanek`). Correctness is a hard gate that precedes all timing, and
  every regime records its input checksum plus a measured `slow_path_rate`.
- `just fastmath-performance-native-x86` runs the producer twice on a native
  Linux x86_64 AVX2/FMA host — once forced to AVX2
  (`SONIUM_FASTMATH_FORCE_AVX2=1`) and once forced to scalar
  (`SONIUM_FASTMATH_FORCE_SCALAR=1`) as a same-input control — then gates
  the native run against the pinned baseline below.
- `benchmarks/check_x86_baseline.py <baseline> <current> [--max-slowdown F]
  [--output PATH]` is the gate. It fails closed on schema, architecture,
  dispatch, correctness-gate, CPU, regime-set, or input-checksum drift, on
  missing clock state, and on frequency-normalized timing regressions beyond
  the threshold (default 20%). Its output also echoes `provenance` for both
  artifacts (energy availability, runtime feature string, external reference
  names); that block is informational only and never fails the gate.

## Current producer provenance

Each artifact records architecture, compiler, compile-time and runtime
(`runtime_features`, including forced-dispatch overrides) features,
dispatch, CPU, microcode, clock source/state/policy, thermal source/state,
energy source/availability with per-regime joules when powercap is present,
the correctness gate, samples/batches, the in-crate references
(`scalar_rust`, `platform_f64_sin_cos`), and `external_references` vendor
statuses (SLEEF/SVML/oneMKL are reported as available-not-benchmarked or
unavailable — the producer has no vendor FFI, so nothing there is timed).

## Pinned baseline status

`benchmarks/x86_64-baseline.json` (Threadripper PRO 3995WX, 2026-08-29) is a
historical, regime-specific result, not a current measurement:

- It predates the `runtime_features`, `clock_source`, `clock_policy`,
  `thermal_source`, `energy_source`, `energy_available`, and vendor
  `external_references` detail fields the current producer emits.
- `external_references` is empty and every `*_energy_joules` is null: no
  independently configured vector-math library was compared and no energy
  counter was available on the capture host.
- The reduced-input regime shows the fast path slower per sample than
  platform libm (10.48 vs 8.81 ns), while the coherent-ramp regime shows it
  faster (10.57 vs 13.82 ns); the Payne–Hanek regime is far slower than
  libm (425.47 vs 49.94 ns). Quote these per regime, never as a single
  speedup.

Refresh the baseline on the pinned AVX2 host via the Justfile recipe rather
than editing the numbers by hand.
