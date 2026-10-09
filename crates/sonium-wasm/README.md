# sonium-roomsim

Room acoustic simulator.

WASM-compatible room acoustics simulation library supporting:
- Rectangular and L-shaped room geometries
- Image Source Method (ISM) for early reflections
- Modal analysis for low-frequency behavior
- Boundary Element Method (BEM) for complex geometries
- Binaural rendering of simulation results

The frequency-response/impulse-response transform path is isolated in
`src/impulse_response.rs`; its validation, interpolation, FFT, and uncertainty
propagation can therefore be tested independently from room orchestration.

## Browser capability and tests

Threaded execution is selected only when both `SharedArrayBuffer` and
`crossOriginIsolated` are available. Otherwise the module reports and uses a
single-threaded, chunked fallback. `get_wasm_info()` exposes the selected mode,
thread count, SIMD status, and isolation capabilities.
It also reports `webgpu_supported`; the current execution mode remains
explicitly `cpu-wasm-fallback` until a WebGPU roomsim kernel is available.
The separate `compute_webgpu_preview()` API evaluates bounded, pre-generated
coherent acoustic paths on a WebGPU device when available; it returns an
explicit error otherwise and does not claim to replace the calibrated room
solver.

Call `init_thread_pool_if_supported(thread_count)` during browser startup to
request Rayon workers. It returns `true` when the isolated threaded runtime was
initialized and `false` when the page must use the single-threaded fallback;
worker-startup errors are returned to the caller. The low-level
`init_thread_pool` export remains available for applications that manage this
policy themselves.

Serve the generated package with:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The browser boundary tests cover config rejection, capability reporting, and
cancellation. Because `wasm-bindgen-rayon` requires WebAssembly atomics,
install nightly Rust with `rust-src` and run them in release mode with the
runner flags before the crate path:

```bash
rustup component add rust-src --toolchain nightly
RUSTUP_TOOLCHAIN=nightly \
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
wasm-pack test --release --headless --chrome crates/sonium-wasm
```

Use `--firefox` or `--safari` for the corresponding browser runners.
The repository workflow `.gitea/workflows/roomsim-wasm.yml` runs the boundary
suite in Chrome, Firefox, and Safari on macOS.

Large configurations are rejected before simulation when their estimated
working set exceeds 256 MiB, including frequency/source buffers, requested
spatial slices, and impulse-response transforms. Progressive callers can use
`compute_chunk(size)`; each response includes the accumulated `partial_spl`
prefix and only reports `done: true` after the final chunk.

Every result includes `provenance`, including a method-specific `validity`
assumption list. For BEM methods, `bem_fallback_count` and
`bem_fallback_reason` report actual failed BEM evaluations; a nonzero count
means the corresponding values are approximations produced by the documented
modal-only or ISM fallback path. Such results are labeled
`fidelity: "fallback_approximate"` and `uncertainty: "fallback_observed"`;
zero-fallback results remain explicitly `not_calibrated` rather than implying
native golden validation. `get_wasm_info()` reports the selected CPU backend,
the reason GPU execution is not active, and an asynchronous
`probe_webgpu_adapter()` is available when a caller needs to distinguish a
browser API/adapter from actual roomsim GPU execution.

Image-source and modal results also expose adjacent-order sensitivity in
`frequency_uncertainty` (full runs) or `provenance.error_estimate` (single
frequency results). When an impulse response is requested, the same complex
frequency sensitivity is propagated through the shared FFT and is available as
`impulse_response.uncertainty`, aligned with the normalized amplitude samples.
These are a-posteriori sensitivity diagnostics, not confidence intervals or
measured-room error bars.

Custom materials may additionally provide `reflection_phase`, six radians at
the same octave-band frequencies as `absorption`. Omitting it preserves the
legacy zero-phase behavior.

Source configurations may provide `orientation: { x, y, z }` for the acoustic
axis. It defaults to world `+Y`; directivity is evaluated in the source-local
frame (`+Y` forward, `+X` right, `+Z` up), and the axis must be finite and
non-zero.

They may instead provide `impedance: { real, imag }`, with six acoustic
impedance bands. The native normal-incidence coefficient
`(Z - rho*c) / (Z + rho*c)` then supplies the complex reflection; impedance
real parts must be non-negative. Set `solver.air_density` when using a
non-default ambient medium; it is validated as positive and participates in
the model hash and all ISM wall-reflection calculations.

Run the native rectangular ISM parity fixture with:

```text
cargo test -p sonium-roomsim --test native_parity
```

## MeshLang frontend feature

The optional `meshlang` feature adds a filesystem-free editor/frontend
surface to the same WASM package. Build it with:

```bash
cargo build -p sonium-roomsim --features meshlang --target wasm32-unknown-unknown
```

The generated bindings expose `meshlangFrontendContract`, `parseMeshLang`,
`MeshLangEditor`, `validateMeshLangGrammar`, canonical-IR preview operations,
and the bounded `meshProjection` ProjectionMeshArtifact consumer.
`MeshLangEditor` reuses a deterministic file-level parse cache and reports
hit/miss/entry telemetry. The APIs return
the parser/compiler versioned JSON envelopes and never invoke Gmsh, read a
path, or start a simulation. Registry previews accept only explicit JSON
payloads supplied by the caller; signed and trust-pinned variants are
available for browser policy enforcement.

The publishable browser adapter lives in [`frontend/`](frontend/). It validates
the Rust capability contract before exposing the generated exports, provides
TypeScript declarations, and includes Node contract tests:

```bash
npm --prefix crates/sonium-wasm/frontend test
```

`build-wasm.sh` writes generated bindings to `frontend/pkg/`, which is included
by the package manifest for distribution.
