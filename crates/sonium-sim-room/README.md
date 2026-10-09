# sonium-sim-room

Generate simulated room measurement data for testing the RoomEQ optimizer.

This crate uses a FEM (Finite Element Method) acoustic solver to compute
the sound pressure field in a virtual room, then exports the results as CSV
measurement files and a `config.json` that can be fed directly into `roomeq`.

For closed-room interior acoustics with locally-reacting impedance walls
FEM is strictly better than BEM: sparse rather than dense matrices, native
support for frequency-dependent local impedance, easy symmetry exploitation,
and better scaling with frequency. BEM's natural domain — exterior radiation
into free space — is not relevant to a room simulator, so the BEM driver
was removed from this crate.

## Quick Start

```bash
# Generate data for all 20 scenarios
cargo run --bin generate-roomeq-data --release -- \
  --output-dir data_tests/roomeq/generated

# Generate a single scenario
cargo run --bin generate-roomeq-data --release -- \
  --scenario medium_surround_5_1 --output-dir data_tests/roomeq/generated

# Cap memory usage at 16 GiB; auto-reduce max_freq if the estimate exceeds that
cargo run --bin generate-roomeq-data --release -- \
  --memory-limit-gb 16 --auto-reduce-freq \
  --output-dir data_tests/roomeq/generated

# Verbose logging
cargo run --bin generate-roomeq-data --release -- -v \
  --output-dir data_tests/roomeq/generated

# Small deterministic native gallery (CSV/WAV/PNG/SVG + gallery.json).
cargo run --bin generate-roomeq-data --release -- \
  --solver fast-hybrid --scenario small_stereo_2_0 \
  --min-freq 50 --max-freq 100 --num-points 2 \
  --gallery /tmp/roomeq-gallery --output-dir /tmp/roomeq-data
```

## FEM nodal-field gallery

An external solved-reference envelope may add a `spatial_field` object with
schema `sonium-sim-room-reference-field/v1`. It declares the source, planar
grid, frequency-aligned complex pressures, and physical positions. When a
retained FEM field is available, the gallery emits per-frequency reference,
candidate, and relative-error panels plus
`fem_reference_field_comparison.{json,csv}`. This field comparison is
separate from listener transfer-response comparison and fails closed on
source, grid, plane, frequency, or non-finite-sample mismatches.

`fem_runner::run_fem_with_mesh_field_report` is the opt-in full-domain CPU
FEM path for retaining accepted complex nodal pressures. The returned
`FemNodalField` can P1-interpolate physical-room points or export a VTU;
`gallery::write_gallery_with_fem_field` derives its XY slice from those nodal
values. The slice sidecar marks `provenance=solved`, `measured=false`, and
`fem_nodal_field=true`. Listener-response rasters and fast-hybrid slices do
not satisfy this contract.

## CLI Options

The optional dense reference path uses `--dense` with `--dense-max-freq` (the
default cutoff is 250 Hz). `--dense-overlap-hz` defaults to zero, preserving a
strict non-overlapping splice; a positive value solves exact shared frequency
grid points with both dense-BLR and FEM and complex-crossfades the seam. The
cutoff and selected overlap width are recorded in `simulation_metadata.json`.

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir` | `data_tests/roomeq/generated` | Output directory |
| `--scenario` | (all) | Run a single scenario by name |
| `--max-freq` | `2000` | Maximum simulated frequency (Hz). Higher = finer mesh, more memory. |
| `--num-points` | `200` | Number of log-spaced frequency points from 20 Hz to `--max-freq`. |
| `--max-concurrent` | `0` | Cap on concurrent scenario solves (`0` = use all cores). |
| `--mesh-grading` | `1.0` | Far-field coarsening multiplier. `1.0` = uniform mesh. |
| `--mesh-fine-radius` | `0.5` | Radius (m) around each source/LP that keeps the wavelength-resolved spacing. |
| `--modal-damping-ratio` | `0.0005` | Minimum FEM modal damping floor as a fraction of real wavenumber (pole regularizer); `0.0` disables the floor. |
| `--memory-limit-gb` | auto (80% of free RAM at start) | Estimated FEM peak-memory budget. Triggers a prompt to lower `--max-freq` if exceeded. |
| `--auto-reduce-freq` | off | Skip the prompt and accept the auto-lowered `--max-freq`. |
| `--gallery DIR` | none | Emit deterministic native gallery assets and checksummed provenance. |
| `-v, --verbose` | off | Enable debug-level logging |

## FEM Stability Benchmarks

## External reference command bridge

The gallery workflow can execute an independently supplied reference driver
after creating a digest-bound problem manifest:

```bash
cargo run --release -p sonium-sim-room --bin generate-roomeq-data -- \
  --scenario small_stereo_2_0 \
  --gallery roomsim-gallery \
  --gallery-reference-command FreeFem++ \
  --gallery-reference-backend freefem \
  --gallery-reference-arg=room.edp
```

The driver receives absolute paths in
`ROOMSIM_EXTERNAL_REFERENCE_MANIFEST`, `ROOMSIM_EXTERNAL_REFERENCE_DATA`,
and `ROOMSIM_EXTERNAL_REFERENCE_OUTPUT`. Repeat
`--gallery-reference-arg` for additional arguments. The driver must write the
validated `sonium-sim-room-reference-response/v1` or `/v2` envelope to the
declared output path; the command is bounded by the adapter timeout and its
backend, digest, frequency, source, and listener topology are checked before
gallery comparison. This bridge does not itself claim physical parity until a
real external solver and independently validated reference are run.

Use `bench-fem-preconditioners` to compare preconditioner reuse, shifted-
Laplacian shifts, and modal damping before changing production defaults:

```bash
cargo run --release -p sonium-sim-room --bin bench-fem-preconditioners -- \
  --scenario medium_multi_seat --max-freq 150 --num-freqs 25 \
  --preconditioners ilut,block-jacobi,oras,habc \
  --rebuild-every 1,3,5,10 \
  --shift legacy,k2 --shift-alpha 0.3,0.5,0.7 --shift-beta 0.1,0.3,0.5 \
  --modal-damping 0.0,0.003,0.005,0.01 \
  --json-out /tmp/fem-stability.json
```

For the Metal prototype, build with the `metal` feature and switch the backend:

```bash
cargo run --release -p sonium-sim-room --features metal --bin bench-fem-preconditioners -- \
  --backend metal \
  --scenario medium_multi_seat --max-freq 150 --num-freqs 25 \
  --gpu-preconditioners auto,chebyshev-8,poly-jacobi-4,point-jacobi \
  --rebuild-every 1,3,5,10 \
  --shift legacy,k2 --shift-alpha 0.3,0.5,0.7 --shift-beta 0.1,0.3,0.5 \
  --modal-damping 0.0,0.003,0.005,0.01 \
  --json-out /tmp/fem-metal-stability.json
```

- `legacy` keeps the existing `0.5*k^2 + i*0.5*k` preconditioner shift.
- `k2` sweeps `(alpha + i beta)*k^2`, which is the main Helmholtz stability
  experiment.
- Non-convergence is fatal after one retry/reset, so failed cells are not
  mixed into agreement tables as if they were valid solutions.

## FreeFEM Comparison

Use `compare-freefem-room` to run the same rectangular-room Helmholtz problem
through Sonium FEM and a generated FreeFEM script:

```bash
cargo run --release -p sonium-qa --features qa-binaries,legacy-sims --bin compare-freefem-room -- \
  --scenario medium_multi_seat --max-freq 150 --num-freqs 25 \
  --mesh-source sonium \
  --freefem-bin FreeFem++ \
  --output-dir /tmp/sonium-freefem-medium \
  --json-out /tmp/sonium-freefem-medium/compare.json
```

- `--mesh-source sonium` exports Sonium's full-domain tetra mesh to
  `sonium_room.mesh` and loads it in FreeFEM with `readmesh3(...)`. This mode
  also forces Sonium's CPU FEM baseline onto the full domain, bypassing
  automatic symmetry reduction so both solvers use the same mesh.
- `--mesh-source freefem-cube` keeps the faster coarse check where FreeFEM
  generates its own rectangular `cube(...)` mesh.
- The generated FreeFEM script receives Sonium's complex wavenumber, Gaussian
  source scale/images, modal damping, and resolved boundary coefficients.
- Use `--emit-only` to generate `freefem_room.edp` and Sonium output without
  executing FreeFEM.

## Scenarios

All scenarios simulate the 20–500 Hz range (100 log-spaced frequency points),
then extend to 20 kHz with a synthetic speaker response shape and smooth random
noise. Boundaries use realistic absorption coefficients (floor 0.3, ceiling
0.05, walls 0.1).

### Stereo & 2.1

| Name | Room | Speakers | Subs | LPs | Description |
|------|------|----------|------|-----|-------------|
| `small_stereo_2_0` | 3x3x2.4 m | L, R | — | 1 | Fullrange stereo |
| `small_stereo_2_1` | 3x3x2.4 m | L, R | 1 | 1 | HP mains + LP sub |
| `small_multi_sub_2` | 3x3x2.4 m | L, R | 2 | 1 | 2 front-corner subs |
| `medium_stereo_2_0` | 5x4x2.5 m | L, R | — | 1 | Fullrange stereo |
| `medium_stereo_2_1` | 5x4x2.5 m | L, R | 1 | 1 | HP mains + LP sub |
| `medium_multi_sub_4` | 5x4x2.5 m | L, R | 4 | 1 | 4 corner subs |
| `medium_multi_seat` | 5x4x2.5 m | L, R | — | 3 | 3 listening positions |
| `large_stereo_2_0` | 7x5.5x2.6 m | L, R | — | 1 | Fullrange stereo |
| `large_stereo_2_1` | 7x5.5x2.6 m | L, R | 1 | 1 | HP mains + LP sub |
| `large_multi_sub_4` | 7x5.5x2.6 m | L, R | 4 | 1 | 4 corner subs |
| `large_multi_seat_2_1` | 7x5.5x2.6 m | L, R | 1 | 3 | 2.1, 3 seats |
| `medium_multi_sub_multi_seat` | 5x4x2.5 m | L, R | 2 | 2 | 2 subs, 2 seats |

### Surround (5.0 / 5.1 / 5.1.4)

Speaker placement follows ITU-R BS.775 and ITU-R BS.2051 guidelines. Surround
speakers are placed at approximately ±110° behind the listener. Height channels
(5.1.4) are near ceiling level at ±45° front and ±135° rear.

| Name | Room | Speakers | Subs | LPs | Description |
|------|------|----------|------|-----|-------------|
| `medium_surround_5_0` | 5x4x2.5 m | L, R, C, SL, SR | — | 1 | Fullrange 5.0 |
| `medium_surround_5_1` | 5x4x2.5 m | L, R, C, SL, SR | 1 | 1 | HP mains + LP sub |
| `medium_surround_5_1_4` | 5x4x2.5 m | L, R, C, SL, SR, TFL, TFR, TRL, TRR | 1 | 1 | Immersive audio |
| `large_surround_5_1` | 7x5.5x2.6 m | L, R, C, SL, SR | 1 | 1 | HP mains + LP sub |
| `large_surround_5_1_4` | 7x5.5x2.6 m | L, R, C, SL, SR, TFL, TFR, TRL, TRR | 1 | 1 | Immersive audio |

Channel abbreviations: L=left, R=right, C=center, SL=surround_left,
SR=surround_right, TFL=top_front_left, TFR=top_front_right,
TRL=top_rear_left, TRR=top_rear_right.

## Output Structure

For each scenario, the binary creates a directory under `fem/`:

```
data_tests/roomeq/generated/
  fem/
    small_stereo_2_0/
      left_lp0.csv          # Left speaker → listening position 0
      right_lp0.csv         # Right speaker → listening position 0
      config.json           # RoomEQ config referencing these CSVs
      simulation_metadata.json
    medium_surround_5_1/
      left_lp0.csv
      right_lp0.csv
      center_lp0.csv
      surround_left_lp0.csv
      surround_right_lp0.csv
      subwoofer_lp0.csv
      config.json
      simulation_metadata.json
```

### CSV Format

Each CSV has 200 rows (100 simulated + 100 HF extension) with columns:

```
freq,spl,phase
20.0000,49.9664,-172.6980
20.6610,49.5056,-173.2463
...
20000.0000,27.0280,27.3745
```

- **freq**: Frequency in Hz (20–20000, log-spaced)
- **spl**: Sound Pressure Level in dB (relative to 20 µPa)
- **phase**: Phase in degrees (-180 to +180)

### config.json

A `RoomConfig` JSON that can be passed directly to the `roomeq` binary:

```json
{
  "speakers": {
    "left": { "name": "left", "csv": "left_lp0.csv" },
    "right": { "name": "right", "csv": "right_lp0.csv" },
    "lfe": { "name": "subwoofer", "csv": "subwoofer_lp0.csv" }
  },
  "optimizer": {
    "num_filters": 7,
    "seed": 42
  }
}
```

Subwoofer sources (names starting with "sub") are automatically grouped under
the "lfe" key. Multiple subs produce a `MultiSub` config with per-sub
measurements.

### simulation_metadata.json

For dense hybrid runs, `dense_blr_max_freq_hz` records the dense cutoff and
`dense_blr_overlap_hz` records the exact-grid crossfade width (`0` means the
strict splice). CPU FEM runs may also include the validated `fem_diagnostics`
object described below.

Records simulation-model choices that are not part of RoomEQ optimization,
including `modal_damping_ratio`, whether dense BLR supplied the low-frequency
band, the low-band source directivity convention, the atmospheric absorption
model, a `frequency_band_model` value that labels the output as hybrid
numerical/synthetic data, and the synthetic high-frequency extension marker.

## Architecture

### Pipeline

```
Scenario (room geometry + speaker positions + listener positions)
    │
    └──▶ Low-band solve (FEM, or dense BLR below --dense-max-freq)
          │
          └──▶ SimulationOutput (complex pressure per source/LP/freq)
                │
                ▼
           HF Extension (500 Hz → 20 kHz synthetic)
                │
                ├──▶ CSV export (freq, spl, phase)
                ├──▶ config.json (RoomConfig for roomeq)
                │
                └──▶ simulation_metadata.json
```

### Modules

| Module | Purpose |
|--------|---------|
| `scenarios` | Defines room geometry, speaker/listener positions, and crossover filters |
| `fem_runner` | Drives the FEM solver (tetrahedral mesh, graded refinement, symmetry exploitation, GMRES + ILUT) |
| `hf_extension` | Extends 20–500 Hz simulation data to 20 kHz with synthetic speaker response |
| `csv_export` | Converts complex pressures to CSV (SPL + phase) |
| `roomeq_config_gen` | Generates `RoomConfig` JSON and simulation metadata from scenario + CSV paths |

Dense BLR/TBEM is an optional low-frequency path in the generator binary. It
uses the same scenario boundary admittance and source convention as FEM, then
splices into the FEM band above `--dense-max-freq`.

### Dense hybrid boundary

The optional `--dense` reference path uses `--dense-max-freq` (default 250 Hz)
for its low-frequency dense-BLR/TBEM band. A positive `--dense-overlap-hz`
solves exact shared frequency-grid points with both dense-BLR and FEM and
complex-crossfades the seam; zero keeps the strict non-overlapping splice. The
cutoff and overlap width are recorded in `simulation_metadata.json`.

### Solver

**FEM** (Finite Element Method):
- Solves the interior Helmholtz equation on a 3D tetrahedral volume mesh.
- Sparse matrices (CSR), GMRES + shifted-Laplacian ILUT preconditioner,
  warm-started across frequencies for fast convergence.
- Frequency-dependent local impedance BCs (rigid, absorption coefficient,
  complex impedance/admittance, Delany-Bazley / Miki porous layers).
- Graded mesh: fine wavelength-resolved spacing within `--mesh-fine-radius`
  of any source or listening position; up to `--mesh-grading×` coarser
  far-field. Preserves the conforming 6-tetrahedron hex decomposition.
- Symmetry exploitation halves or quarters the mesh for symmetric source
  layouts.

### HF Extension

Below 500 Hz, room interaction dominates; this is the numerically solved
region (FEM, or dense BLR below the configured dense crossover). Above the
simulated cutoff, the response is not a room-acoustics solve. The extension
module synthesizes the remaining band to 20000 Hz and records that hybrid
numerical/synthetic model in `simulation_metadata.json`:

- **Main speakers**: Presence bump at 3 kHz (+1.5 dB), treble rolloff above
  8 kHz (-6 dB at 20 kHz), plus slowly-varying cosine-interpolated random
  noise [-5, +3] dB. Seeded per (source, LP) pair for deterministic output.
- **Subwoofers**: Steep rolloff at -60 dB/octave above the simulation range
  (sub crossover).

### Source Configuration

Sources use crossover filters to model real speaker systems:

| Type | Crossover | Typical use |
|------|-----------|-------------|
| Fullrange | None | Stereo mains without subwoofer |
| Highpass | 80 Hz, 4th-order Butterworth | Mains in systems with subwoofer |
| Lowpass | 80 Hz, 4th-order Butterworth | Subwoofer |

Low-band room excitation is intentionally monopole-like. FEM injects each
source as an isotropic volume source scaled by amplitude and crossover, and
dense BLR uses the same receiver-only directivity convention for boundary
excitation. Directional speaker patterns are applied when evaluating receiver
pressure, not when feeding reflected wall energy. Generated
`simulation_metadata.json` records this as `receiver_only_low_band`.

CPU FEM runs also record an optional `fem_diagnostics` object in
`simulation_metadata.json`. It contains the validated solver configuration,
aggregate convergence counters/residual, and per-frequency timing, iteration,
preconditioner-rebuild, and fallback evidence. The generator publishes this
object only after every FEM frequency has converged; accelerator runs leave it
absent until equivalent device diagnostics are available.

### Physical FEM mesh-convergence evidence

For a promotion-grade refinement study, use
`fem_runner::run_fem_mesh_convergence_ladder` with three labelled
`FemMeshConvergenceSpec` values. The helper performs sequential full-domain
CPU FEM solves, retains each `FemNodalField` and `FemRunDiagnostics` packet,
and compares solved responses using the exact listener coordinates captured by
each field. It fails closed on duplicate labels, invalid mesh configuration,
non-increasing physical mesh counts, listener drift, non-solved provenance, or
the configured RMS/max response budgets.

The manifest-backed `roomsim.fem-mesh-convergence.v1` QA binary exercises this
production path. Focused unit tests cover its preflight boundary without
launching a FEM solve.

## Development

### Adding a New Scenario

1. Add a function `scenario_XX_name()` in `src/scenarios.rs` returning a
   `Scenario` with room, sources, listening positions, and source names
2. Add it to `all_scenarios()` and update the doc comment count
3. Update the `test_all_scenarios_count` assertion
4. Add config validation tests in `tests/pipeline_test.rs`
5. Run: `cargo test -p sonium-sim-room --lib && cargo test -p sonium-sim-room --test pipeline_test`

Source naming conventions:
- Main speakers: `left`, `right`, `center`, `surround_left`, `surround_right`
- Height channels: `top_front_left`, `top_front_right`, `top_rear_left`, `top_rear_right`
- Subwoofers: any name starting with `sub` (e.g., `subwoofer`, `sub1`, `sub2`)
  — these are automatically grouped under `lfe` in the config

### Testing

```bash
# Library unit tests (scenarios, HF extension)
cargo test -p sonium-sim-room --lib

# Integration tests (full FEM pipeline + config validation)
cargo test -p sonium-sim-room --test pipeline_test

# Lint
cargo check -p sonium-sim-room && cargo clippy -p sonium-sim-room
```

### Gallery artifact contract

`--gallery DIR` also emits `spatial_metrics.csv`, `room_metrics.json`, and a
listener-by-frequency `listener_frequency_waterfall.png`. Fast-hybrid galleries
add a labelled `pressure_slice_xy.png`/CSV pair with explicit reduced-order
provenance; FEM galleries derive spatial slices from retained physical nodal
fields rather than synthesizing them from listener data. A reference spatial
field also produces checked reference/candidate/error panels and field-
comparison sidecars.
The metrics and CSV files are authoritative; PNG/SVG files are fixed-viewport presentation
assets. `room_metrics.json` records whether an RT60 estimate was possible and
retains the per-frequency solved/reduced/synthetic provenance labels. The
generated `index.html` links figure sources and every exported WAV, while
`gallery.json` and `gallery-provenance.json` retain checksums.

The source contract is exposed through `source_calibration`: a unit source
amplitude maps to the documented volume velocity and the validated free-field
monopole relation. This is an analytic calibration reference, not measured
loudspeaker evidence.

### Key Types

```rust
/// Output from a FEM simulation.
/// Pressures indexed as [source_idx][lp_idx][freq_idx].
pub struct SimulationOutput {
    pub frequencies: Vec<f64>,
    pub pressures: Vec<Vec<Vec<Complex64>>>,
    pub source_names: Vec<String>,
    /// Per-frequency provenance: Solved (full model), ReducedOrder (fast
    /// hybrid), or Synthetic (HF extension). Exported as `provenance.csv`.
    pub provenance: Vec<FrequencyProvenance>,
}
```

### Dependencies

- `sonium-fem`: FEM acoustic solver
- `sonium-bem`: Delany-Bazley / Miki porous-layer impedance models and optional
  dense BLR/TBEM low-frequency solves in the generator binary.
- `sonium-xem`: Shared types (room geometry, source config, etc.)
- `sonium-roomeq`: RoomConfig types for config generation
- `sysinfo`: Detect free system RAM for the default memory budget
- `rayon`: Parallel scenario / frequency solving
- `rand`: Seeded RNG for HF extension noise
