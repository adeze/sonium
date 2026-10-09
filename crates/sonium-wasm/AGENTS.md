# sonium-roomsim (package `sonium-roomsim`, version: 0.4.1)

Browser (WASM) room-acoustics simulator: rectangular + L-shaped rooms, multi-source
with directivity/crossover, solvers (direct free-field, ISM orders 1–3, modal,
hybrid modal/ISM with Schroeder crossover, BEM / hybrid-BEM via `sonium-bem`),
frequency response + spatial slices, impulse response, binaural rendering,
WebGPU preview only.

## Purpose

Simulates room acoustics for room EQ optimization and visualization, in the
browser via WASM and natively via `rlib` + `cli` binaries.

## Crate Types

- `cdylib` - Dynamic library (for WASM)
- `rlib` - Rust library

## Module Layout

- `lib.rs` - Config types, physics kernels, orchestration, wasm-bindgen API (~5800 lines)
- `acoustics.rs` - RT60 / critical-distance helpers
- `room_modes.rs` - Rectangular rigid-wall modes, image sources, hybrid crossover weight
- `ism.rs` - Image-source reflection-order helper
- `bem_solver.rs` - BEM type conversions and config
- `impulse_response.rs` - FFT-based IR via `sonium-fft`
- `scattering_objects.rs` - Box/sphere/cylinder scattering objects
- `loaders.rs` / `serialization.rs` / `validation.rs` - I/O, provenance, config validation
- `adaptive.rs` - Adaptive refinement helpers
- `meshlang.rs` - MeshLang bridge (feature-gated)
- `webgpu_preview.rs` - Bounded WebGPU preview (not the solver)
- `worker_detect.rs` - Threading/capability detection
- `bin/` - `room_simulator_bem`, `room_simulator_fem` CLI (`cli` feature)

## WASM Support

Compiles to WebAssembly with parallel processing via wasm-bindgen-rayon, with a
deterministic single-threaded fallback when threading is unavailable.

## Dependencies

- `wasm-bindgen`, `wasm-bindgen-rayon` - WASM interop
- `sonium-bem` - BEM solver backend

## Testing

```bash
cargo test -p sonium-roomsim
cargo check -p sonium-roomsim && cargo clippy -p sonium-roomsim
```

Native integration tests live in `tests/` (`native_parity.rs`,
`browser_boundary.rs`, `wasm_review_20260904.rs`, …). Browser tests need
nightly + atomics + `wasm-pack test --release --headless --chrome`.

## Notes

- Frequency sweeps and slices parallelize with rayon only when the
  machine-adaptive gate (`ParallelPolicy`: per-thread grains scaled by
  the active Rayon worker count, `SONIUM_ROOMSIM_PAR_THRESHOLD_SWEEP` /
  `SONIUM_ROOMSIM_PAR_THRESHOLD_SLICES` overrides, `SONIUM_SERIAL` kill
  switch) says dispatch pays off; small problems run the same closures
  serially.
- Slices evaluate the same method-aware field as the main response; slice
  provenance carries the configured solver method.
- Binaural HRTF gains apply per-source before summation.
- The hybrid modal/ISM crossover uses one broadband Eyring-based Schroeder
  frequency (`broadband_schroeder_frequency`), not a per-frequency RT60.
