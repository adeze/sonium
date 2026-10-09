# sonium-sim-speaker (lib: `sonium_sim_speaker`, version: 0.1.3)

Product-level loudspeaker simulation, CEA-2034 output, and optimisation.
Consumes typed driver-unit results from `sonium-sim-speaker-modeling`.

## Layout

```
src/
  model.rs              SpeakerConfig, driver/cabinet/waveguide/separator types
  spline.rs             Cone and surround spline generators
  mesh_gen.rs           Delaunay-triangulated surface meshes
  mlg_loader.rs         MeshLang `.mlg` cabinet loading
  simulation.rs         Dense/H-matrix BEM solve + exact residual gating
  directivity_export.rs Complex far-field → xem `ComplexDirectivityGrid` handoff
  cea2034_output.rs     CEA-2034 measurement set + scoring
  distortion.rs         Distortion post-processing (`ThdConfig::validate`)
  power_compression.rs  Power compression post-processing (`CompressionConfig::validate`)
  optimization.rs       Cabinet/driver parameter optimisation (always available)
  product_metrics.rs    CEA-2034 product metrics (matrix-first impedance)
  coupling.rs           Typed driver-coupling solve
  directivity_reference.rs  Reference-spinorama correlation
bin/
  speaker_sim.rs        `speaker-sim` binary (feature: cli)
benches/
  frequency_lane_scaling.rs  Lane-gate crossover bench
```

## Dependencies

- `sonium-bem` — assembly + post-processing
- `sonium-fem` — magnetostatics and electro-acoustic post-processing
- `sonium-sim-speaker-modeling` — typed T/S, BL(x), Le(f), and velocity handoff
- `sonium-solvers` — GMRES, ILU, optimisation
- `sonium-xem` — unified VTU/PVD writer
- `math-delaunay` — surface mesh generation
- `meshlang-parser`, `meshlang-ast` — MeshLang `.mlg` cabinets
- `sonium-roomeq` — CEA-2034 preference-rating implementation

## Solver Discipline

- Below ~3000 BEM DOFs: dense LU. 3000–5000 DOFs: dense FGMRES+ILU. Above 5000: H-matrix FGMRES. Non-converged solutions are rejected via true-residual gates; the exact uncompressed TBEM matvec check is opt-in (`verify_exact_hmatrix_residual`).
- GCRO-DR subspace recycling (size 20) and ILU reuse are on by default; both reset across frequency steps wider than a quarter-octave.
- Frequency sweeps run in parallel (rayon) by default with per-lane solver state; warm-started guesses across design points are dropped when the DOF count changes.
- Prescribed velocity RHS assembly must include the `iωρ` pressure-gradient conversion and the same explicit Burton-Miller beta as the matrix.
- The BEM side is `e^{-iωt}`; the modeling crate is `e^{+iωt}`. Radiation-impedance loads and driver velocities crossing the `DriverModel` handoff must be conjugated.

## Driver Boundary

- Driver motor, moving assembly, enclosure, and FEM golden models belong to
  `sonium-sim-speaker-modeling`.
- This crate consumes `DriverModel`; do not add BL/Le extraction here.
- Cabinet, waveguide, BEM radiation, CEA-2034, and product optimisation remain here.

## Geometry Discipline

- Cone shell triangle winding matches the analytical cone node normals so element normals point into the front acoustic domain.
- Tweeter waveguide is honoured by the woofer cone geometry.

## CEA-2034 Discipline

- Sound power uses 70 unique CTA measurement rows weighted by spherical slices, not azimuthal `|cos(angle)|`.
- Preference rating goes through `sonium_roomeq::cea2034::score_peq` with the listening-window and early-reflection groups.

## Room Directivity Handoff

- `directivity_export::export_complex_directivity` converts a `SimulationResult` (which retains per-point complex pressures in `SimulationResult::complex_pressure`) into an xem `ComplexDirectivityGrid` for `Directivity::ComplexGrid` room sources. Speaker frame: x=right, y=up, z=forward; the grid's on-axis direction is (theta=90°, phi=90°) and each frequency is normalized by its complex on-axis sample (grid is 1+0i on-axis; absolute level stays with the room source amplitude). Measurement points must form a complete angular sphere at uniform radius including the on-axis direction — CEA-2034 arcs are not sufficient; use `spherical_eval_points`.
- The exported pattern already contains the speaker crossover (magnitude+phase) and multi-driver interference: room sources using it must not apply their own `CrossoverFilter` (sonium-sim-room warns, does not error).
- CLI: `speaker-sim --export-directivity grid.json [--directivity-step 10] [--directivity-radius 1.0]` runs the sim on the sphere grid and writes the serde JSON; room scenario JSONs import it inline as `"directivity": {"type": "complex_grid", ...}` (xem `DirectivityConfig`).

## Features

- `cli` (default) — `clap`, `speaker-sim` binary
- `cuda` / `metal` — GPU backends via `sonium-bem`/`sonium-compute`

`optimization` is NOT feature-gated (it needs only `cea2034_output` +
`sonium-roomeq`, no `clap`).

## Testing

```bash
cargo test -p sonium-sim-speaker --release
cargo check -p sonium-sim-speaker && cargo clippy -p sonium-sim-speaker
cargo check -p sonium-sim-speaker --no-default-features  # optimization visible
```

## Notes

- The `cli` feature is on by default but the lib compiles cleanly without it for use as a dependency.
- `ThdConfig::validate` / `CompressionConfig::validate` name the first bad
  field; `measure_thd` returns the NaN guard on validation failure.
- `SimulationResult::woofer/tweeter_electrical_impedance_ohm` are
  `#[deprecated]` (prescribed-result compat); producers must fill
  `driver_electrical_impedances_ohm` (woofer = driver 0, tweeter = 1).
- Frequency lanes: `use_frequency_lanes` (cached env gate) /
  `use_frequency_lanes_with` (pure, hermetic); `SONIUM_SPEAKER_PAR_THRESHOLD`
  override, `SONIUM_SERIAL=1` hatch.
