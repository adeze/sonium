# Sonium fuzz targets

This is a standalone `cargo-fuzz` package and is intentionally excluded from
the main workspace. Install `cargo-fuzz`, then run a target from the
repository root:

```text
cargo fuzz run fastmath_sincos
cargo fuzz run fft_planner
cargo fuzz run compute_view_descriptor
cargo fuzz run compute_csr_descriptor
cargo fuzz run compute_csr_spmv
cargo fuzz run compute_fmm
cargo fuzz run compute_perf_models
cargo fuzz run compute_scheduler_sequences
cargo fuzz run solvers_csr_inputs
```

`fastmath_sincos` feeds arbitrary IEEE-754 bit patterns through the scalar
fastmath API and checks finite-output and trigonometric-identity contracts.
`fft_planner` fuzzes bounded transform lengths, batch sizes, generated complex
inputs, advertised scratch sizing, forward/inverse round trips, and the native
and `rustfft`-compatibility in-place, out-of-place, and immutable buffer
contracts.

`compute_view_descriptor` exercises checked vector and multidimensional
binding metadata before native launch adapters see it. `compute_csr_descriptor`
exercises checked host CSR construction before sparse pre-bake and launch.
`compute_scheduler_sequences` exercises bounded admission, completion,
cancellation, device-loss, and drop/refund races against scheduler quotas.
`compute_fmm` exercises bounded checked indexed dense-matvec gather/scatter with
padded leading dimensions and repeated indices; malformed layouts must return
typed errors or complete without panic.
`compute_perf_models` feeds ordinary and extreme dimensions through every
public roofline byte/FLOP model; arithmetic must saturate without panicking.
`solvers_csr_inputs` exercises checked sparse-matrix construction before
factorization and Krylov paths receive untrusted data.
`compute_csr_spmv` exercises the checked CPU pre-baked SpMV steady-state
boundary with arbitrary malformed row pointers, columns, and finite payloads;
malformed descriptors must return typed errors without mutating the destination.

The targets are also compile-checked independently with:

```text
cargo check --manifest-path fuzz/Cargo.toml
```
