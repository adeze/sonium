# sonium-sim-room (lib: `sonium_sim_room`)

Generate simulated room measurement data for RoomEQ testing using a FEM
acoustic solver.

## Why FEM (and not BEM)?

For closed-room interior Helmholtz problems with locally-reacting impedance
walls, FEM is the natural fit:

- Sparse matrices vs BEM's dense surface matrices, so memory scales as
  `O(N_volume)` (and is sparse) instead of `O(N_surface^2)` dense.
- Frequency-dependent local impedance BCs map directly onto the FEM
  weak form (boundary mass with admittance coefficient).
- Symmetry exploitation halves or quarters the volume mesh.

BEM's natural use case is exterior radiation into free space (loudspeaker
scattering, anechoic far-field), which a *room* simulator never needs. The
BEM driver was therefore removed from this crate.

## Key Components

- `scenarios.rs` -- Room geometry, speaker/listener positions, crossover filters (20 scenarios)
- `fem_runner.rs` -- FEM solver driver (sonium-fem) with graded mesh and symmetry exploitation
- `hf_extension.rs` -- Extends 20-500 Hz simulation to 20 kHz with synthetic speaker response
- `csv_export.rs` -- Converts complex pressures to CSV (SPL + phase)
- `roomeq_config_gen.rs` -- Generates `RoomConfig` JSON for roomeq

## Binaries

- `generate-roomeq-data` -- CLI for generating room measurement data, with
  estimated-memory budget enforcement and optional automatic max-frequency
  reduction.

## Dependencies

- `sonium-fem` -- Acoustic solver
- `sonium-bem` -- Used only for the `materials` module (Delany-Bazley / Miki
  porous-layer impedance models). No BEM solver code is invoked.
- `sonium-xem` -- Shared simulation types
- `sonium-roomeq` -- RoomConfig types

## Testing

```bash
cargo test -p sonium-sim-room --lib
cargo test -p sonium-sim-room --test pipeline_test
cargo check -p sonium-sim-room && cargo clippy -p sonium-sim-room
```

## Important Notes

- Subwoofer sources (names starting with "sub") are automatically grouped under "lfe" in config
- HF extension above 500 Hz is synthetic (not physically simulated)
