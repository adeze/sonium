# sonium-sim-propellers (lib: `sonium_sim_propellers`, version: 0.1.0)

Application-level propeller aeroacoustic simulation. Sits on top of `sonium-fem`'s `physics::aeroacoustics::PropellerSource`.

## Layout

```
src/
  source_model.rs       Axisymmetric source model with toroidal/serration modifiers
  blade3d.rs            3D panel-level blade geometry, acoustic summary, VTU export
  optimization3d.rs     3D shape optimization (chord, twist, camber, sweep, dihedral)
  bin/propeller_3d_opt.rs   `propeller-3d-opt` binary
```

## Key Types

- `PropellerDesign` — wraps `PropellerSource` with planform, edge treatment, serration, and evaluation mode.
- `PropellerPlanform::Conventional | ToroidalLoop` — loops are *source-generation modifiers* (the axisymmetric propagator cannot represent the loop geometry directly).
- `EdgeTreatment::Smooth | LeadingEdgeSerrations | TrailingEdgeSerrations | LeadingAndTrailingSerrations` — affects coherent-wake eddy breakup and incoming turbulence pre-conditioning.
- `SerrationGeometry { amplitude_chord_fraction, wavelength_chord_fraction, span_fraction }` — tooth amplitude/wavelength as fractions of local chord, span coverage.
- `EvaluationMode::FastSurrogate | ResearchControl` — switch between optimization-loop speed and higher-order coherence quadrature.
- `Blade3dConfig`, `BladeMesh`, `BladePanel`, `PanelAcousticSummary` — 3D representation.
- `BladeShapeVariables`, `OptimizedBlade`, `Propeller3dOptConfig` — optimisation interface.

## Output

ParaView VTU for both volume and surface fields. The crate is part of the workspace-wide migration to ParaView outputs.

## Testing

```bash
cargo test -p sonium-sim-propellers --release
cargo check -p sonium-sim-propellers && cargo clippy -p sonium-sim-propellers
```

## Notes

- Per-panel acoustic summaries are parallelised over `rayon`.
- The toroidal-loop "modifier" reduces estimated open-tip vortex shedding; do not interpret it as a full geometric solve.
- Default serration geometry mimics owl wing micro-airfoils (1.8 % chord amplitude, 5.5 % wavelength, 72 % span). Effectiveness drops outside `amplitude ∈ [0.6 %, 3.5 %]` and `wavelength ∈ [9 %, 22 %]`.
