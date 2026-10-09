# Changelog

## Unreleased

### Added

- Added machine-adaptive `parallelism::ParallelPolicy` for assembly merges:
  the rayon gate scales with `available_parallelism`
  (`SONIUM_FEM_PAR_THRESHOLD` absolute override, `SONIUM_SERIAL=1` escape
  hatch) instead of a fixed element count.
- `assemble_bilinear` and `assemble_bilinear_per_element` now merge large
  per-element triplet lists on rayon workers via order-preserving sharded
  concatenation, so the push order — and duplicate summation in `to_csr` —
  is bitwise identical to the serial merge; small assembles stay serial.
- Element-tabulation and pointwise-combine loops (generic, region-routed,
  legacy mass/stiffness/Helmholtz assemblers, prebaked coefficient
  application, Helmholtz shift assembly, transport combine, nonlinear
  residual/tangent) auto-select serial below the machine-adaptive
  `ELEMENTS_PER_THREAD` / `POINTWISE_VALUES_PER_THREAD` grains via the new
  `collect_indexed` / `filter_collect_indexed` helpers; both paths visit
  indices in order, so results are identical.

### Fixed

- Fixed the serial triplet-merge bottleneck after parallel element
  tabulation in bilinear assembly: large merges no longer single-thread the
  scatter into the global `TripletMatrix`.
- Replaced the `thread_local! UnsafeCell` quadrature-rule cache with a
  process-wide `OnceLock<Mutex<..>>` table returning owned clones: no more
  `'static` lifetime laundering or aliasing risk under rayon worker churn.
- `ElasticityForm`/`ViscoelasticForm` 2D assembly without a plane
  assumption now panics with a contract message (`plane_2d`, covered by a
  `should_panic` test) instead of a bare `Option::unwrap`; all `% dim`
  index math routes through the documented `vector_component` helper plus
  `BasisTabulation::grad`/`vector_value` accessors with debug layout
  asserts.
- Reconciled `stabilisation_tau` docs with the code: the viscous term is
  `C_inv·ν²/h⁴` with `C_inv = 36` absorbing the textbook `9·C` factor
  (Tezduyar/Osawa, Codina variants), pinned by a hand-computed unit test.
- `AdvectionForm` documents the `[u_x, u_y, …]` closure contract and trips
  a `debug_assert` on short returns instead of silently truncating.
- The 22 meshlang-transfer integration tests declare
  `required-features = ["meshlang"]`, so default-feature
  `cargo check --all-targets` no longer fails with E0432.
- The Navier–Stokes backend path uploads the saddle-point preconditioner
  via `try_from_csr_host_parts_with_pressure_preconditioner` and falls
  back to CPU with a warning instead of aborting on allocation failure.
- Fixed a tabulation rebuild storm that made multi-element H(curl)
  tabulation never finish: `HcurlSpace::try_tabulate` rebuilt the
  whole-mesh edge/face tables on every quadrature point (directly, and
  indirectly through per-call full-mesh validation). Edge signs are now
  computed locally in O(edges) — bit-identical to the global rule,
  pinned by a unit test — and validation is the caller's once-up-front
  duty per the `FunctionSpace` contract. The 48-tet exact-sequence
  check went from >30 min (killed) to ~3 s.

### Added (physics)

- New `JohnsonChampouxAllard` rigid-frame porous model
  (`physics::acoustics::impedance`): five-parameter microstructure
  equivalent fluid (σ, φ, α∞, Λ, Λ′) with `effective_density` /
  `effective_bulk_modulus` accessors, wired into the shared
  `PorousMaterialModel` trait (characteristic impedance, propagation
  constant, backed-layer surface impedance, Robin coefficient). Verified
  against an independent published-data-backed oracle (Z_c/z0 =
  2.321+2.075j at 200 Hz for the σ=20 kPa·s/m², φ=0.98, α∞=1,
  Λ=Λ′=87 µm specimen), DC/high-frequency analytical limits, audio-band
  passivity, and agreement with the in-crate Miki fit.
- "Which model when" decision docs for mean-flow acoustics
  (`physics::aeroacoustics`: LPF vs LEE vs LNS vs Lighthill/FW-H vs DRP,
  with the Galbrun/APE absence noted and justified) and for viscothermal
  modelling (LRF vs BLI vs full LNS).

## 0.5.30

### Added

- Added first-class FEM physics modules for porous flow/poroelasticity, species transport,
  thermoelasticity, compressible-flow building blocks, turbulence transport, phase-field fracture,
  electrochemistry, semiconductor drift-diffusion, shallow-water waves, and composite laminates.
- Added FEM-side speaker multiphysics contracts for axisymmetric pressure/elasticity/shell/coupling
  forms, magnetic bias and perturbation workflows, distributed Lorentz force assembly, solid-shell
  ties, acoustic-structure boundary coupling, cyclic sector constraints, thermoviscous gap model
  selection, coupled frequency-sweep reuse, and stable speaker result extraction.
- Added generalized FEM QA coverage for Maxwell, electrostatics, biharmonic plates, shells,
  magnetostatics, nonlinear solids, plasticity, piezoelectricity, contact, aeroacoustics,
  electroacoustics, Navier-Stokes, and lumped-circuit analytical benchmarks.
- Added shell distributed pressure/load assembly, shell point loads, 6-DOF shell boundary helpers,
  and curved shell surface mesh generators.
- Added biharmonic Hessian cell assembly and plate load helpers.
- Added DSG/MITC-style projected shear stabilization for P1 triangular Mindlin plate elements,
  preserving the compatible shear formulation via `new_consistent` and validating the stabilized
  simply supported sinusoidal plate QA against the Navier Mindlin solution.
- Added Maxwell current-source helpers and standalone Maxwell PML assembly.
- Added electrostatic point-charge/image helpers, two-cylinder mesh support, contact reference
  helpers, and additional analytical QA formulas.

### Fixed

- Shell pressure loads now evaluate spatially varying load functions at the global element
  centroid instead of local coordinates.
- Q2 quadrilateral and Hex27 H1 tabulation now provide physical Hessians, preventing biharmonic
  cell stiffness from silently assembling as zero on tensor-product P2 meshes.
- Maxwell PML assembly now applies anisotropic curl-curl stretching and a separate mass
  coefficient instead of uniformly rescaling the full element contribution.

## 0.5.29

### Added

- Added transient thermal time stepping for `M dT/dt + K T = f`, including backward Euler,
  Crank-Nicolson, reusable effective-matrix/RHS helpers, and a `thermal_bdf_step` convenience API.
- Added vector traction Neumann assembly for elasticity `VectorH1Space` systems.
- Added scalar Saint-Venant torsion helpers for Prandtl stress-function and warping-function
  formulations.
- Added standalone Helmholtz PML assembly via `PmlHelmholtzForm`.
- Added spherical Bessel/Hankel helper wrappers for QA analytical scattering checks.
- Added axisymmetric magnetostatic `A_theta` forms for azimuthal solenoid/coil current,
  distributional point-dipole source assembly, and a tagged C-core mesh generator.

## 0.5.28

### Fixed

- `CachedShapeKey` now uses exact floating-point equality consistent with its hash, restoring the
  `HashMap` Eq/Hash invariant for shape-function caching.
- Added P2 tetrahedron shape-function evaluation and gradients, and made unsupported H1
  element/degree combinations fail explicitly instead of being advertised by DOF mapping and
  failing later during tabulation.
- 15-point tetrahedral quadrature rule (order >= 5) used coordinates with negative barycentric
  values (outside the reference domain) and wrong weight sum. Replaced with the correct Keast
  (1986) degree-5 15-point rule: 4 orbits (centroid, two S31, one S22), all coordinates
  non-negative, weights sum to 1/6.
- `BodyForceForm` in elasticity indexed shape function values as `values[dof * dim + c]` instead
  of `values[dof]`, reading the wrong shape function for all non-x components.
- `VectorMassForm` had the same incorrect indexing pattern (`values[a * dim + comp_a]`). Fixed
  to `values[a]`.
- Added comment documenting the RT0 basis function normalization issue (missing factor of 2 for
  triangles, 3 for tetrahedra) and incorrect divergence values.
- Added comment documenting that the multigrid restriction operator uses `R = P^T` without
  variational scaling, which may degrade convergence for non-uniform meshes.

### Known Issues

- RT0 (Raviart-Thomas) basis uses `N_i = λ_i ∇λ_i` instead of the correctly normalized
  `N_i = dim · λ_i ∇λ_i`. Divergence values are also wrong. This affects H(div) conforming
  discretizations.
- Neumann BC for 3D triangle faces uses centroid (1-point) quadrature, which is inconsistent
  with the consistent mass matrix used in Robin BCs.
