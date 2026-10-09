# Changelog

## Unreleased

### Added — aeroacoustics 2-D/axisymmetric QA

- `qa-fem-aeroacoustics-analogies-drp` grows four planar/axisymmetric
  checks: 2-D Lighthill cylindrical spreading, spinning-ring (m = 3)
  and revolved-strip (m = 2) on-axis nulls, and closed-contour
  cylindrical spreading of a compact pulsating contour.

### Added — `bem-step1.md` Phase 1.1 + 1.2 CI gates

- New `qa-bem-mie-ci` binary: strict regression detector for BEM with
  Burton-Miller. Validates against the Mie series at kD ∈ {1, 5, 20} with
  per-`kD` tolerances calibrated to the currently-achievable accuracy of the
  constant-element discretisation (kD=1: 1% L2 / 1.5% max; kD=5: 1.5% L2 /
  2.5% max). The kD=20 case is gated behind `MIE_HEAVY=1` because it requires
  ~80k DOFs. Each case prints elements-per-wavelength and the engaged
  Burton-Miller β so silent CBIE-only regressions are visible at a glance.
- `qa-bem-mie-ci` also runs a preconditioner-parity test: A/B compares
  unpreconditioned BiCGSTAB against GMRES + every `BemPreconditioner`
  variant (`SurfaceMassLumped`, `Jacobi`, `CalderonV`, `CalderonW`) and
  asserts each converges to the same far-field answer to within 1e-7 L²
  (≈16× over the worst observed `CalderonW` divergence of 6e-9). Wall-clock
  is printed alongside so iteration-count wins surface in CI logs.
- `qa-bem-sota-validation` extended with `validate_hmatrix_rank_stats_far_field`
  — builds an H-matrix on two well-separated icospheres, exercises ACA on
  the cross blocks, and prints `RankStats` (compression ratio, rank
  distribution, saturation count) so silent ACA under-resolution is
  observable in routine CI runs. The existing
  `validate_hmatrix_near_field_bookkeeping` validator now also prints rank
  stats for parity.

## 0.1.2

### Added

- Added `convergence_rate_with_sizes(errors, element_sizes)` for convergence suites that need
  the actual mesh size or time step ratio instead of assuming uniform halving.
- Added unit coverage for convergence-rate estimation, zero-reference relative error handling,
  and relative residual normalization.
- Helmholtz FEM analytical QA now routes P1 L2 convergence checks through the shared
  convergence-rate helper and reports mean rates plus finest-mesh errors for each MMS case.
- Added a deterministic rigid shoebox-room modal validation binary for low-frequency room
  acoustics, including axial, tangential, oblique, ordering, and cube-degeneracy checks.
- Added a deterministic loudspeaker vibro-acoustic validation binary for small-signal
  electrodynamic drive, motional impedance, baffled-piston radiation load, and power checks.
- Added computed benchmark binaries for FEM shoebox-room modal Rayleigh quotients and baffled
  circular-piston radiation/directivity reference quantities.
- Added a full application-pipeline physics validation binary for loudspeaker BEM SPL/directivity
  and room BEM modal-response acceptance checks.
- Extended loudspeaker pipeline physics validation to use the full 72-point CEA-2034 evaluation
  fixture, with on-axis duplicate, directivity-shape, rear-response, and SPL regression checks.
- Extended room pipeline physics validation with deterministic rigid cube, rigid rectangular, and
  porous-treated rectangular acceptance cases covering geometry shifts, material damping, SPL
  bounds, modal-band tolerances, and regression SPL references.
- Added a BEM SOTA validation binary covering Burton-Miller resonance-regime assembly, SLFMM
  dense parity, H-matrix near-field bookkeeping, and near-singular quadrature behavior.
- Extended BEM SOTA validation with an MLFMM adjoint/transpose shape-Lagrangian check, guarding
  optimization workflows that depend on `A^T` products.
- Added BEM porous/material impedance validation to the deterministic `qa-deep` tier.
- Extended the computed shoebox-room benchmark with a core TBEM frequency-response check around
  the first axial room mode, complementing the FEM Rayleigh-quotient modal checks.

### Fixed

- `convergence_rate(errors)` now computes the correct order for uniformly halved refinement
  sequences instead of dividing by `ln(n)`.
- `relative_error` now handles zero-norm references deterministically, returning zero for
  identical zero vectors and infinity for non-zero error against a zero reference.
- Added `relative_residual_norm` to normalize residuals by `||b||`, with explicit zero-RHS
  handling.
- Documented that `check_spd` is a cheap necessary screening check, not a full SPD proof.

## 0.1.1

### Added

- Added deterministic QA tiers to the Justfile:
  - `qa-smoke` for fast pre-merge physics smoke validation.
  - `qa-deep` for convergence, analytical scattering, WaveHoltz, and Schwarz-PML validation.
  - `qa-expensive` for explicit room/speaker application pipeline validation.
  - `qa-full` for the complete standalone QA binary sweep.
- QA Just recipes now require standalone validation binaries to exit successfully, print
  `SUCCESS`, avoid `FAILED`, and finish within an explicit timeout.
- `cargo test -p sonium-qa` now runs a fast smoke harness that executes representative BEM,
  FEM, and solver QA binaries with timeout and `SUCCESS`/`FAILED` output checks.

### Fixed

- `convergence_rate` divided by `ln(n)` (sample count, a constant) instead of `ln(h_prev/h_next)`.
  Now takes `element_sizes` as a parameter and computes the correct rate
  `ln(e_i/e_{i+1}) / ln(h_i/h_{i+1})`.
- `relative_error` divided by zero when the reference vector was zero, producing silent NaN.
  Now returns `T::Real::infinity()` for zero-norm references.
- Added `relative_residual_norm` function that normalizes by `||b||`, complementing the existing
  absolute `residual_norm`.
- `check_spd` previously checked only positive diagonal + symmetry. Added documentation noting
  this is a necessary-but-not-sufficient condition (true SPD check requires Cholesky).
- `bem_basic.rs` low-frequency tolerance tightened from 50% to 10%.
- `bem_mie_validation.rs` surface pressure tolerance tightened from 90% to 30%.
- `bem_3d_sphere.rs` renamed `test_3d_sphere_mesh_convergence` to
  `generate_sphere_mesh_convergence_data` to clarify it generates placeholder data, not a real
  BEM validation.
- Added TODO comments to the analytical wave reference binaries (now
  `wave_1d_reference.rs` and `wave_cylinder_2d_reference.rs`) and
  `bem_3d_sphere.rs`
  documenting that they currently validate analytical against itself (no actual BEM solver call).
