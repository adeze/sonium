# Wolfram accuracy gate for `sonium-sim-*`

This gate compares deterministic outputs from all six simulation application
crates with references independently evaluated by Wolfram Engine:

| Crate | Reference case |
|---|---|
| `sonium-sim-room` | Multi-point/multi-frequency Green function, inverse-distance and phase laws, and the direct plus six first-order rectangular-room images |
| `sonium-sim-speaker` | Natural cubic values/derivatives, exact knot interpolation, extrapolation, and affine reproduction |
| `sonium-sim-speaker-modeling` | Baffled-piston directivity, complex far-field pressure and spreading, full mechanical radiation impedance, and low-`ka` asymptotes |
| `sonium-sim-pcfan` | Blade-passing and harmonic frequency lattices, flow continuity, Mach number, pressure/SPL closure, IEC A-weighting, and spherical sound-power conversion |
| `sonium-sim-propellers` | Compact rotating-loading (Gutin) harmonics, linear load scaling, unity conventional-blade handoff, and A-weighted band power |
| `sonium-sim-structural-mechanics` | Full Euler-Bernoulli nodal displacement/rotation field, clamp reactions, strain energy, residuals, and force/energy balance |

The acoustic cases make their conventions explicit: linear homogeneous air,
RMS piston velocity where applicable, `exp(-i omega t)` for room and speaker
phasors, compact rotating sources for fan/propeller kinematics, and far-field
piston radiation. The room cases cover both perfectly absorbing boundaries
and a lossless first-order image sum. They are not diffuse-field,
late-reverberation, or high-order image validations.

Checks intentionally overlap in physically different ways. For example, a
room pressure must match the complex Green function and separately preserve
`1/r` spreading and propagation phase; a beam solution must match every
Euler-Bernoulli nodal DOF and independently close reactions and strain energy.
This redundancy catches indexing, sign, phasor-convention, normalization, and
unit errors that a single scalar comparison can miss.

## Run

Install Wolfram Engine and `wolframscript`, then run from the repository root:

```bash
WOLFRAM_KERNEL="/path/to/WolframKernel" \
  cargo test -p sonium-qa --release --features sim-wolfram \
  --test wolfram_sim_crates -- --nocapture
```

The same test is included in the repository-wide `just qa-rust-wolfram`
recipe, which enables `sim-wolfram` for this target automatically.

The harness prefers a direct kernel supplied by `WOLFRAM_KERNEL` or
`WolframKernel`; on macOS it also detects the standard Wolfram Engine kernel
path automatically. Direct kernel execution avoids launcher overhead and is
the most reliable CI route. If no kernel is found, the harness falls back to
`WOLFRAMSCRIPT` or `wolframscript` on `PATH`.

Set `SONIUM_SIM_ACCURACY_TRACE=1` to print a completion marker after each
crate report when diagnosing a slow Wolfram expression. Normal runs remain
compact and emit only the final JSON report.

The Rust side writes an ephemeral JSON fixture containing physical inputs and
Sonium outputs. `sim_crates_accuracy.wls` recomputes the references and returns
per-check absolute error, relative error, tolerance, and pass/fail status as
JSON. A missing engine, malformed report, omitted crate, nonzero Wolfram exit,
or exceeded tolerance fails the test.

## Add a case

1. Add a deterministic producer function to `tests/wolfram_sim_crates.rs`.
2. Put every physical input needed by the independent equation in its JSON;
   do not pass a Sonium-computed reference value.
3. Add the Wolfram equation and an `accuracyCheck` entry.
4. State the physical regime and convention in a comment.
5. Prefer complex-field, conservation, or convergence errors over a single
   magnitude-only score when extending the gate to a full solver.
