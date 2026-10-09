# sonium-sim-speaker

Product-level loudspeaker simulation, CEA-2034 output, and cabinet optimisation.
Consumes typed driver-unit models from `sonium-sim-speaker-modeling`.

## Installation

```toml
[dependencies]
sonium-sim-speaker = { workspace = true }
```

## Binaries

```bash
cargo run -p sonium-sim-speaker --bin speaker-sim --release  -- [options]
cargo run -p sonium-sim-speaker-modeling --bin motor-extract --release -- [options]
```

The `speaker-sim` binary requires the `cli` feature (default-on).

## What it covers

- **Parametric speaker model** (`model`) — sealed cabinet with woofer, tweeter, waveguide, and internal separator. Spline-driven cone/surround geometry.
- **Mesh generation** (`mesh_gen`) — Delaunay-triangulated surface meshes, woofer cone honoured under a tweeter waveguide, cone-shell triangle winding aligned with analytical cone normals so element normals point into the front domain.
- **MeshLang loader** (`mlg_loader`) — alternative cabinet/driver geometry from `.mlg` files.
- **BEM acoustic simulation** (`simulation`) — small solves use dense LU; large solves use H-matrix FGMRES. Every solve is checked with the true operator residual, and recycling resets across frequency steps wider than a quarter-octave.
- **Driver handoff** — consumes `DriverModel` (T/S, BL(x), Le(f), and complex velocity transfer) from `sonium-sim-speaker-modeling`.
- **CEA-2034** (`cea2034_output`) — listening window, early reflections, sound power, in-room response. Sound power uses 70 unique CTA measurement rows with spherical-slice weights.
- **Distortion / power compression** (`distortion`, `power_compression`) — large-signal post-processing.
- **Optimisation** (`optimization`) — drives cabinet parameters for flat response and smooth radiation. Scoring delegates to `sonium_roomeq::cea2034::score_peq` (preference rating, listening-window and early-reflection groups).

## Modules

- `model` — `SpeakerConfig`, drivers, cabinet, waveguide, separator
- `spline` — cone/surround spline generation
- `mesh_gen` — Delaunay surface meshing
- `mlg_loader` — MeshLang-described cabinets
- `simulation` — dense/H-matrix BEM acoustic solve, exact residual gating, frequency sweeps
- `cea2034_output` — CEA-2034 measurement set
- `distortion`, `power_compression` — large-signal post
- `optimization` — design parameter optimisation

## Features

- `cli` (default) — `clap` for the `speaker-sim` binary

## Testing

```bash
cargo test -p sonium-sim-speaker --release
cargo check -p sonium-sim-speaker && cargo clippy -p sonium-sim-speaker
```

## Notes

- Above ~3000 BEM DOFs the simulation refuses to return acoustic fields from a non-converged solve — the H-matrix solution is verified against an exact, uncompressed TBEM matvec without allocating a dense matrix.
- Optimisation refuses an under-resolved mesh: `optim_max_mesh_freq` must cover the model's highest simulated frequency.
- Sound power weighting is the spherical-slice form on 70 unique CTA rows, not azimuthal `|cos(angle)|`.
- CEA-2034 scoring uses the `sonium-roomeq` preference-rating implementation.

## Deterministic gallery

The gallery command uses the versioned `examples/two_way_speaker.mlg` when no
model is supplied, with a pinned 4-point 200 Hz–800 Hz reference sweep, so it
is reproducible from a clean checkout:

```bash
cargo run -p sonium-sim-speaker --bin speaker-sim -- --gallery target/speaker-gallery
```

It writes the mesh, CEA-2034 data and manifest, simulation provenance, fixed
PNG geometry/polar views, SVG spinorama/impedance curves, and `gallery.json`.
Every gallery entry includes checksums for both the figure and its source
artifact.

## Conventions and model limits

BEM pressure amplitudes are peak phasors with `exp(-i omega t)`. SPL converts
their magnitude to RMS before referencing 20 µPa. Driver modeling and measured
references use RMS `exp(+i omega t)` phasors; the handoff must conjugate and
convert amplitude. `measurement_reference::bem_peak_to_measurement_rms` provides
the pressure conversion explicitly.

The Hann DFT normalizes by the actual window sum, including short records.
`distortion::ThdResult` measures mechanical displacement harmonics. Acoustic
THD requires propagating each harmonic's surface velocity to the receiver.
`power_compression::CompressionPoint` is a lumped thermal output estimate;
its relative decibels do not include enclosure or changed acoustic loading.

## Measured speaker references

`measurement_reference` compares supplied predictions with independent
observations at exactly matching geometry, drive voltage, frequency and probe
position. It performs no fitting, pressure rescaling, phase alignment or
extrapolation. Acquisition metadata is mandatory. Calibration samples are
reported separately from held-out validation samples.

```bash
cargo run --release -p sonium-sim-speaker --example compare_measurements -- measured.json predicted.json report.json
```

Both input sets use schema version 1 and the same `geometry_id`. A measured
sample has `id`, `role` (`calibration` or `validation`), `observable`,
`frequency_hz`, `drive_voltage_rms_v`, `position_m`, `value` (a `[real, imag]`
RMS phasor), `absolute_tolerance`, and `relative_tolerance`. Predictions carry
the same identity and conditions, their complex `value`, and a set-level
`provenance`. Measurement metadata contains `dataset_id`, `source`, and
`acquisition_method`. Positions use x=right, y=up, z=forward in meters.

Canonical observables are `pressure_rms_pa`, `electrical_impedance_ohm`,
`displacement_rms_m`, `pressure_thd_fraction`, `compression_db`, and
`coil_temperature_k`. Scalar channels use zero imaginary parts. Compression
is the signed hot-minus-cold output difference at matched drive. Acceptance
requires `|predicted-measured| <= absolute + relative*|measured|`.
Complete-speaker acceptance requires held-out evidence for all six channels;
partial coverage is written to the report and exits unsuccessfully. No real
measurement dataset is bundled; unit-test fixtures are synthetic.

## CPU and GPU sweeps

CPU gates use the active Rayon pool size, allowing local thread budgets to
control frequency lanes and modeling batches. Existing memory admission and
contiguous frequency lanes preserve continuation and preconditioner reuse.

CUDA prescribed-driver solves use Complex32 inner GMRES with Complex64 outer
refinement. GPU forward sweeps default to `GpuValidationPolicy::Qualification`.
After qualifying a workload, opt into `GpuValidationPolicy::production()` for
first/last, every tenth, and marginal-residual reference checks. Every frequency
still passes a host f64 residual gate. `GpuExecutionReport` schema 3 records
which samples received independent reference and derivative checks. Physical
error maxima cover those reference samples only.

Wavenumber derivatives are assembled only when
`SolverConfig::gpu_wavenumber_derivative` is requested. Metal cannot supply
that derivative path and rejects the request. Host publication and f64 checks
remain explicit; this change does not make the pipeline fully device-resident.

See [the WebGPU prototype](../../apps/speaker-demo/README.md) for native cache
generation, browser launch instructions, and CPU/CUDA benchmark commands.
