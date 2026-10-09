# sonium-sim-speaker-modeling

Application-level speaker-driver study specifications inspired by detailed finite-element speaker workflows.

This crate is deliberately a layer above `sonium-fem`: it captures geometry, physics, coupling, symmetry, and study choices for electrovibroacoustic driver examples, then exposes adapters into existing FEM transducer and lumped-circuit APIs where Sonium already has them.

It owns the driver unit—motor, moving assembly, enclosure, and FEM golden
models. Product-level cabinet, waveguide, exterior BEM radiation, CEA-2034,
and optimisation remain in `sonium-sim-speaker`, which consumes the typed
`DriverModel` handoff (T/S, BL(x), Le(f), and complex velocity transfer).

The response table is **not yet an end-to-end FEM result**. It uses the configured lumped electrical and mechanical parameters, loaded by the analytical radiation impedance of a rigid circular piston in an infinite baffle. Reports carry an explicit `response_provenance`, and representative magnetic quantities are labeled as such. Complex drive and response phasors use RMS amplitudes, so average power is `Re(Z) |v_rms|^2`.

## Examples

```bash
cargo run -p sonium-sim-speaker-modeling --example detailed_driver
cargo run -p sonium-sim-speaker-modeling --example axisymmetric_vs_3d
cargo run -p sonium-sim-speaker-modeling --example frequency_response
cargo run -p sonium-sim-speaker-modeling --bin motor-extract --release -- [options]
```

## What It Covers

- Stationary magnetic bias followed by small-signal frequency-domain perturbation.
- Time-domain studies for pulses, higher harmonics, and nonlinear large-signal motion.
- 2D axisymmetric, 3D sector, and full 3D modeling choices.
- Voice-coil Lorentz coupling, acoustic-structure coupling, shell/solid part representation, and thermoviscous-loss recommendations.
- FEM bridge objects for magnetic perturbation workflow, representative magnetic bias, distributed Lorentz assembly, acoustic-structure coefficients, sweep points, and response extraction.
- Frequency-dependent viscous/thermal penetration depths and conservative broadband gap-model selection.
- Exact baffled-piston radiation resistance, numerically checked radiation reactance, passive acoustic loading, and RMS power-balance validation.
- Passive lumped sealed and vented rear-enclosure loads, including explicit port end correction, leakage, porous loss, and port radiation resistance.
- Infinite-baffle piston far-field pressure and directivity with explicit angle, range, medium, and RMS-phasor conventions.
- Executable nonlinear lumped Newmark studies with `BL(x)`, `Le(x)`, cubic suspension stiffness, bounded sampling, and harmonic-distortion diagnostics.
- Self-consistent steady-state copper heating and thermal-compression response using the full complex electromechanical load.
- Fallible validation for user/deserialized requirements plus safe frequency-grid generation.
- Typed `DriverModel` export for product simulation, including interpolated
  diaphragm velocity boundary data.

The new enclosure and large-signal paths are application-level lumped models. The nonlinear transient currently runs without an acoustic load and rejects enclosed studies instead of silently ignoring their load. These paths do not turn the report into a distributed FEM field solution: finite-baffle diffraction, breakup, cavity modes, transient thermal feedback, and spatial motor/structure/acoustic assembly remain outside their validity range.

Missing FEM assembly features are tracked in [`reviews/fem-for-speaker-modelling.md`](../../reviews/fem-for-speaker-modelling.md), and the correctness audit is in [`reviews/speaker-modeling-correctness-audit.md`](../../reviews/speaker-modeling-correctness-audit.md).

## Deterministic gallery

Render the validated quick whole-speaker artifact set without interactive
ParaView or PyVista:

```bash
cargo run -p sonium-sim-speaker-modeling --bin whole-speaker -- --quick --gallery target/whole-speaker-gallery
```

The command emits the existing VTU/PVD and CSV/report sources, fixed PNG field
views, SVG engineering curves, and `gallery.json` with per-figure source and
output checksums.
