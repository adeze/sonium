# sonium-sim-speaker-modeling (lib: `sonium_sim_speaker_modeling`, version: 0.1.0)

COMSOL-style speaker-driver modeling on top of existing Sonium FEM APIs. This
crate owns driver-unit motor, moving assembly, enclosure, and FEM golden models.

## Layout

```
src/
  lib.rs      Study specifications, example builders, recommendations, FEM adapters
  driver_model.rs       Typed product handoff: T/S, BL(x), Le(f), velocity
  motor_extraction.rs   Axisymmetric magnetostatic BL extraction
  motor_topology.rs     Density-based pole/yoke topology optimization (D1-D3):
                        analytic adjoint, filter/projection/continuation,
                        symmetry/bridge/robust constraints, differential
                        permeability from the nonlinear bias
  projected_reanalysis.rs  D5: reanalysis of projected (binary) topology
                        designs on body-fitted, mesh-converged geometry.
                        Binary steel extraction + boundary polygon loops
                        (half-edge chaining with pinch resolution), integer-K
                        structured remesh that embeds every coarse interface
                        edge (exact steel/non-design region preservation), and
                        a true-material magnetostatic re-solve (constant mu_r
                        or D3 differential map re-extracted per level) over a
                        refinement ladder with a 1%-change converged flag.
                        QA: qa-speaker-projected-reanalysis.
  resolved_coupling.rs  ΔZ load feedback: condensed resolved enclosure/port load
                        iterated into the distributed structure (fixed point)
  port_flow/            Resolved incompressible reflex-port path: deterministic
                        2D slot and current 3D round/flared geometry; Taylor–Hood
                        P2–P1 BDF flow coupled weakly to a lumped box/driver in
                        lockstep; pressure-traction end caps; port/diaphragm
                        volume-flow measurement; THD/IMD/SPL analysis and
                        compression sweeps. Owns the current 2D strict QA gates
                        and the opt-in 3D pipe-flow nightly fixture.
  cabinet_design.rs   Parametric cabinet design variables (C4): CabinetDesign
                        (box w×h×d internal + wall material, optional front
                        port, baffle dims, panel braces) lowers to lumped
                        EnclosureSpec, resolved ResolvedAcousticConfig
                        (equivalent cylinder R=√(w·h/π), sealed unsupported),
                        C2 CouplingFixture, and braced sonium-fem Cabinet;
                        design_variables/from_design_variables expose the
                        continuous vector (port presence, brace count, and
                        material stay categorical).
  fem_bem_coupling/     3D interior-FEM/exterior-BEM cabinet/port coupling (C2):
                        exterior BEM condensed to a DtN map on the aperture
                        elements (LU once + m back-solves per frequency), dense
                        low-rank block in the interior Helmholtz FEM;
                        rom.rs adds the C5 fast path: rigid-wall modal basis
                        augmented with Craig-Bampton attachment modes (one
                        shifted-static solve per aperture face) + augmented
                        two-term static tail correction for the drive
  acoustic_rom.rs       Modal-basis ROM for fixed-topology acoustic FEM sweeps
                        (C5): LOBPCG rigid-wall basis (zero mode included),
                        dense m×m per-frequency Galerkin solve, two-term
                        static tail correction per RHS pattern, opt-in exact
                        residual spot-check; attachment_modes exposes the
                        Craig-Bampton augmentation used by the coupled path.
                        QA: qa-speaker-acoustic-rom. The former sonium-solvers
                        LOBPCG initial-block rank cap (~17) is fixed: the
                        seeded xorshift64* initial block has full numerical
                        rank, so requested mode counts are delivered.
  structural_constraints.rs  Optimization-ready constraint evaluators (C3):
                        excursion vs Xmax, per-region recovered stress vs
                        allowables, cabinet panel displacement/bending limits;
                        normalized margins (limit − value ≥ 0) bindable into
                        sonium-solvers Constraints via to_solver_constraints.
                        Cabinet modal displacement X_mn is exposed by
                        sonium-fem cabinet_radiation (additive companion API).
examples/
  detailed_driver.rs       Print the default detailed electrovibroacoustic report
  axisymmetric_vs_3d.rs    Compare axisymmetric, 3D sector, and full 3D choices
  frequency_response.rs    Print the small-signal response sweep
```

## Scope

- Keep application-level modeling choices out of `sonium-fem`.
- Expose product consumers through `DriverModel`; keep cabinet, waveguide,
  exterior BEM radiation, CEA-2034, and product optimisation out of this crate.
- Reuse existing FEM public APIs where they exist.
- Do not modify `sonium-fem`, `sonium-compute`, or `sonium-solvers` from this crate.
  (Reviewed exceptions: C3 added the additive `modal_displacements` /
  `surface_displacement` API to sonium-fem `cabinet_radiation.rs`, since the
  panel modal model lives there. The C1 follow-up relaxed the
  electroacoustics `CoupledIterativeConfig::relative_tolerance` default from
  1e-9 to 1e-8 in sonium-fem `physics/electroacoustics/iterative.rs`: the
  coupled operator's attainable f64 residual floor is ~1e-9, so the old
  default made acceptance a rounding-noise coin flip; all consumer QA gates
  are 1e-8 or looser.)
- Track FEM feature gaps in `review/fem-for-speaker-modelling.md`.

### Resolved port-flow solver discipline

- Close the scalar box-pressure/port-flow interface at the accepted end-step
  state using the affine two-pressure response; use Taylor–Hood P2 velocity–P1
  pressure spaces only and leave residual stabilization off.
- Use the laminar viscosity model unless the optional algebraic closure is
  explicitly selected; drive both end caps through pressure traction.
- Keep runtime configuration, convergence, finite-state, sampling, and QA
  gates strict. The vector-mass and exact-reference implementations are in
  place. The bounded straight-tube linear P2 reference now passes its
  independent `<1%` mesh-convergence gate under refinement. The nonlinear
  Taylor–Hood runtime gate and product-scale mesh/time convergence remain
  unproven; do not report either unresolved gate as passed.

## Testing

```bash
cargo test -p sonium-sim-speaker-modeling
cargo check -p sonium-sim-speaker-modeling && cargo clippy -p sonium-sim-speaker-modeling
```
