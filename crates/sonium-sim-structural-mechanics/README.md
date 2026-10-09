# sonium-sim-structural-mechanics

COMSOL-style structural mechanics examples built on Sonium FEM.

The first example is an introductory bracket workflow:

- linear static plane-stress solve
- damped transient study definition
- assembled first bending frequency and mass-normalized mode
- assembled complex frequency-domain harmonic response sweep

The bracket is an explicitly identified two-dimensional plane-stress model.
Static results include applied/reaction force balance, free-DOF equilibrium,
strain-energy identity, and centroid stress-recovery provenance. The modal
step solves the generalized assembled stiffness/mass problem with an
initial-stress geometric stiffness from the signed axial preload. Harmonic
response solves the assembled complex system with mass-proportional damping
calibrated at the first mode. Positive preload is tension; compression reduces
the mode toward the assembled model's instability limit.

The crate also exposes `run_axisymmetric_cylinder`, a bounded meridional FEM
reference for a thick-walled cylinder. It assembles Sonium FEM's
`AxisymmetricElasticityForm` on a P1 triangular `(r,z)` mesh, integrates inner
and outer pressure with the `2*pi*r` weight, and constrains axial displacement
to zero for a plane-strain Lamé comparison:

```rust
use sonium_sim_structural_mechanics::{
    run_axisymmetric_cylinder, AxisymmetricCylinderStudy,
};

let result = run_axisymmetric_cylinder(&AxisymmetricCylinderStudy::introductory())?;
assert!(result.converged);
assert!(result.max_hoop_stress_relative_error < 0.10);
```

The result reports displacement, radial/hoop stress, free-equilibrium, and
strain-energy gates with explicit CPU/f64 provenance. This is a plane-strain
thick-cylinder verification case, not a claim of general axisymmetric,
nonlinear, contact, thermal, or external-solver capability.

Use `run_static_mesh_convergence` for coarse/fine displacement and energy
evidence. Centroid stress is retained as a diagnostic; unregularized point or
edge peaks are not optimization constraints. The convergence report exposes
`stress_converged` separately from its displacement/energy `converged` gate so
stress behavior remains visible without turning a singularity-sensitive
centroid diagnostic into an optimization acceptance criterion.

Use `run_static_order_convergence` for same-mesh p-refinement evidence. It runs
the assembled bracket at `StructuralPolynomialOrder::Linear` and
`StructuralPolynomialOrder::Quadratic`; the quadratic solve uses genuine P2
edge DOFs, consistent edge traction integration, a correctly sized consistent
mass matrix, and P2 centroid stress recovery. The returned static results tag
their polynomial order explicitly. `solve_first_bending_mode_order` and
`solve_harmonic_tip_response_order` accept the same order tag, use the
consistent P1/P2 mass and geometric-stiffness matrices, and include all
boundary DOFs in modal influence and harmonic tip extraction. The legacy
modal/harmonic wrappers follow the order carried by their static/modal input.
For independent frequency samples, `solve_harmonic_tip_response_parallel_order`
uses an explicit local thread count and returns the same input frequency order;
each frequency owns its sparse factorization, so the parallel path does not
claim factorization reuse or accelerator execution.

`StructuralReducedPreview` is the portable browser/WASM visualization boundary.
`bracket_static_preview` converts a validated bracket result into versioned
JSON containing coordinates, triangles, displacement, and a scalar colour field.
It is intentionally a reduced visualization payload: the WASM backend remains
unsupported for production structural solving and is reported that way by
`structural_backend_capabilities`.

The library boundary is checked on the native host plus cross-target library
builds for `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`,
`aarch64-pc-windows-msvc`, and `wasm32-unknown-unknown` with commands such as:

```bash
cargo check -p sonium-sim-structural-mechanics --lib --target wasm32-unknown-unknown
cargo check -p sonium-sim-structural-mechanics --lib --target x86_64-unknown-linux-gnu
cargo check -p sonium-sim-structural-mechanics --lib --target aarch64-pc-windows-msvc
```

The WASM check proves compilation of the portable API; it does not turn the
CPU reference solver into an unqualified browser production backend.

For a validated multi-load envelope, compose the same bracket model with
typed load cases:

```rust
use sonium_sim_structural_mechanics::{
    run_multi_load_study, BracketStudy, MultiLoadStudy, StructuralLoadCase,
};

let base = BracketStudy::introductory_bracket();
let study = MultiLoadStudy::new(
    base,
    vec![
        StructuralLoadCase::new("service", -120.0, 250.0),
        StructuralLoadCase::new("reversal", 120.0, 250.0),
    ],
);
let result = run_multi_load_study(&study)?;
println!(
    "critical stress case: {} ({:.3e} Pa)",
    result.envelope.critical_stress_case,
    result.envelope.max_von_mises_pa,
);
```

The bracket family remains a 2D plane-stress/plane-strain model; the crate
also includes the bounded axisymmetric cylinder reference above and an
executable 3D tetrahedral-solid family:

```rust
use sonium_sim_structural_mechanics::{run_solid3d_modal, run_solid3d_static, Solid3DStudy};

let study = Solid3DStudy::introductory();
let static_result = run_solid3d_static(&study)?;
let modal_result = run_solid3d_modal(&study)?;
assert!(static_result.force_balance_residual_n < 1e-6);
assert!(modal_result.frequencies_hz[0] > 0.0);
```

The 3D family marks the clamped x-minimum and loaded x-maximum faces,
assembles 3D elasticity, integrates total traction, recovers centroid
von-Mises stress, and solves the generalized mass/stiffness modal problem.

`run_solid3d_multi_load` provides the bounded P1 multi-case extension. It
assembles the tetrahedral stiffness operator once, factorizes one constrained
CPU sparse system, and solves each named x-maximum-face traction case with a
separate RHS. Every case reports displacement, centroid von-Mises stress,
reaction/equilibrium, force-balance, energy, and solver-residual gates, while
the result records the governing stress case and factorization memory. This is
an assembled linear P1/f64 reference; quadratic or matrix-free block solves,
shell multi-load, nonlinear load paths, and accelerator execution are not
claimed.

`run_solid3d_topology` is the bounded assembled 3D SIMP extension. It uses
linear tetrahedral element-local elasticity matrices, exact maximum-compliance
active-case gradients, one constrained sparse factorization per density
iterate reused across all traction RHSs. The design variables use a
centroid-based 3D density filter, Heaviside projection, volume projection,
and beta continuation before final volume/connectivity, stress, equilibrium,
energy, residual, and final-density first-frequency gates. Set
`Solid3DTopologyStudy::objective` to `Compliance` (the default),
`FirstFrequency`, or `BucklingLoadFactor` to optimize the corresponding
assembled objective. Frequency sensitivities use generalized
eigenvalue differentiation through the assembled mass/stiffness pair. For
explicitly x-compressive load cases, buckling sensitivities use bounded
central finite differences of the re-solved linearized initial-stress factor.
The `buckling_model` result identifies whether the geometric matrix used the
full `-sigma` tensor (when positive-semidefinite) or the bounded `sigma_xx`
fallback for mixed-sign stress states.
Its capability ID is
`sonium-structural.cpu-reference.f64.solid3d-topology.v1`. It intentionally
also reanalyses graph-distance eroded/dilated designs and the active-element
body-fitted mesh. It does not claim nonlinear/post-buckling behavior, general
3D geometric stiffness for indefinite stress states, or CUDA/Metal/WASM
execution. Buckling-objective runs are deliberately bounded reference
workflows because each sensitivity evaluation re-solves the initial-stress
problem at perturbed densities.

Large-case operator selection is explicit through
`run_solid3d_static_with_options`. `AssembledCg` is the compatibility CPU
reference, `AssembledAmgFgmres` uses the solver crate's elasticity AMG profile,
and `MatrixFreeJacobiFgmres` applies element stiffness actions without storing
the global matrix. `run_solid3d_static_order(..., StructuralPolynomialOrder::Quadratic)`
selects the affine-tetrahedral P2 matrix-free reference path: edge-midpoint
DOFs receive consistent face traction, homogeneous Dirichlet rows/columns are
handled in the operator wrapper, and centroid stress recovery uses the full
P2 displacement field. Its result carries the distinct
`sonium-structural.cpu-reference.f64.matrix-free-p2.v1` provenance ID. These
paths are CPU/f64 references; no CUDA, Metal, WASM, factorization-reuse, or
external-solver parity is implied.

The crate also exposes a genuine linear static shell path backed by
`sonium-fem`'s flat three-node CST membrane + DKT bending element:

```rust
use sonium_sim_structural_mechanics::{run_shell_static, ShellStaticStudy};

let result = run_shell_static(&ShellStaticStudy::introductory())?;
assert!(result.converged);
assert!(result.force_balance_residual_n < 1.0e-7);
```

This path assembles the shell stiffness and normal pressure load, clamps all
six DOFs on the x-minimum edge, solves the constrained CSR system with the
CPU GMRES/ILU reference solver, and reports force equilibrium and the
strain-energy identity while retaining moment resultants as diagnostics. Its
provenance is explicitly
`sonium-structural.cpu-reference.f64.shell-tri3-cst-dkt.v1`. It is a bounded
linear flat-shell workflow; it does not claim general shell curvature,
anisotropic materials, nonlinear/contact/fracture behavior, external-solver
parity, or CUDA/Metal/WASM dispatch.

`run_shell_multi_load` is the corresponding P1 load-case path. It assembles
the same flat-shell operator once, reuses one constrained sparse factorization
for each named pressure RHS, and reports per-case displacement, resultants,
load-integration, equilibrium, and energy gates. Its provenance is
`sonium-structural.cpu-reference.f64.shell-tri3-cst-dkt-multi-load.v1`; shell
stress recovery, curved/nonlinear behavior, and accelerator execution remain
outside this bounded reference.

The crate also exposes a bounded complex-stiffness viscoelastic plane-stress
path:

```rust
use sonium_sim_structural_mechanics::{run_viscoelastic_2d, Viscoelastic2DStudy};

let result = run_viscoelastic_2d(&Viscoelastic2DStudy::introductory())?;
assert!(result.converged);
assert!(result.phase_error_rad < 1.0e-8);
```

This workflow uses a rectangular P1 triangular mesh, homogeneous clamping on
the x-minimum edge, and a right/top corner point load. It assembles the FEM
`ViscoelasticForm` over `Complex64`, solves with CPU GMRES/ILU, and checks the
complex free residual, reaction balance, bilinear-form energy identity, and
the exact `1 / (1 + i*eta)` displacement/compliance scaling against an
independent zero-loss real solve. Its provenance is explicitly
`sonium-structural.cpu-reference.complex64.viscoelastic-2d.v1`. It is a
frequency-independent structural-loss reference path, not a time-domain
hereditary material model or a claim of general anisotropic, nonlinear,
contact, fracture, or GPU/WASM viscoelastic dispatch.

For nonlinear 3D evidence, `run_hyperelastic_3d` drives the actual
`sonium-fem` total-Lagrangian `HyperelasticForm<NeoHookean>` through a
displacement-controlled continuation path on a P1 tetrahedral box:

```rust
use sonium_sim_structural_mechanics::{
    run_hyperelastic_3d, Hyperelastic3DStudy,
};

let result = run_hyperelastic_3d(&Hyperelastic3DStudy::introductory())?;
assert!(result.converged);
assert!(result.relative_energy_error < 1.0e-10);
assert!((result.taylor_remainder_ratio - 0.25).abs() < 0.02);
```

The result records Newton residual/iterations, affine displacement and
deformation-gradient recovery, integrated Neo-Hookean strain energy, and a
finite-difference constitutive gradient/Taylor gate. This is explicitly a
displacement-controlled total-Lagrangian material check; it does not claim
force-controlled post-buckling, contact, plasticity, or fracture capability.

The crate also exposes a bounded staggered phase-field damage path:

```rust
use sonium_sim_structural_mechanics::{
    run_phase_field_damage, PhaseFieldDamageStudy,
};

let result = run_phase_field_damage(&PhaseFieldDamageStudy::introductory())?;
assert!(result.converged);
assert!(result.damage_factorizations == 1);
assert!(result.damage_monotone);
```

This study assembles the fracture damage operator once, assembles per-element
history sources, solves mechanics with per-element degraded elasticity, and
enforces irreversible `[0, 1]` damage updates over displacement load steps.

The same staggered operator can also be driven by a ramped right-edge force:

```rust
use sonium_sim_structural_mechanics::{
    run_phase_field_damage_force_controlled, PhaseFieldForceControlledStudy,
    PhaseFieldLoadingControl,
};

let result = run_phase_field_damage_force_controlled(
    &PhaseFieldForceControlledStudy::introductory(),
)?;
assert_eq!(result.loading_control, PhaseFieldLoadingControl::ForceControlled);
assert_eq!(result.applied_force_history_n.len(), result.load_steps);
assert!(result.final_tip_displacement_m > 0.0);
```

Force control integrates a consistent piecewise-linear traction over the
right-edge segments and records the force/displacement continuation history.
It remains a stable seeded continuation path: arc-length control, snap-back
tracking, and calibrated fracture-initiation/toughness claims are outside this
small verification study.
It is seeded displacement-controlled damage evidence; it does not claim
force-controlled crack propagation, crack-initiation calibration, or
post-failure fracture toughness prediction.

The beam family similarly assembles 2D Euler-Bernoulli or Timoshenko elements,
reports translational reactions separately from the clamped reaction moment,
and compares static tip deflection against the cantilever formula:

```rust
use sonium_sim_structural_mechanics::{run_beam2d_modal, run_beam2d_static, Beam2DStudy};

let study = Beam2DStudy::introductory();
let static_result = run_beam2d_static(&study)?;
let modal_result = run_beam2d_modal(&study)?;
assert!(static_result.analytical_relative_error < 1e-8);
assert!(!modal_result.frequencies_hz.is_empty());
```

The plate family provides an assembled P1 triangular Mindlin solve with a
simply-supported Navier benchmark:

```rust
use sonium_sim_structural_mechanics::{run_plate2d_static, Plate2DStudy};

let study = Plate2DStudy::introductory();
let result = run_plate2d_static(&study)?;
assert!(result.analytical_relative_error < 0.02);
assert!(result.force_balance_residual_n < 1e-6);
```

The plate result records the DSG/MITC-style shear treatment, integrated
pressure/reactions, free-DOF equilibrium, strain energy, and analytical
center-deflection error. Shells, composites, nonlinear constitutive laws, and
optimization constraints remain separate validated study families rather than
being inferred from these APIs.

Composite laminate membrane execution is also available as the bounded
anisotropic material path. Each `CompositePly` is an orthotropic material with
its own material-axis angle; the study assembles the laminate A matrix, applies
an integrated axial traction, and recovers local ply stresses with Tsai–Hill
and maximum-stress indices:

```rust
use sonium_sim_structural_mechanics::{
    run_composite_laminate_static, CompositeLaminateStudy,
};

let study = CompositeLaminateStudy::introductory();
let result = run_composite_laminate_static(&study)?;
assert!(result.analytical_relative_error < 1e-8);
assert!(result.max_tsai_hill_index < 1.0);
```

The composite family currently requires negligible membrane–bending B coupling;
unsymmetric layups fail validation instead of silently discarding bending
physics. Ply failure results identify the governing element centroid and retain
both global and material-axis stress components. General anisotropic solid or
curved-shell constitutive paths remain outside this bounded laminate study.

Uniform thermoelastic free expansion is available as an assembled plane-stress
family. It uses the FEM thermal eigenstrain load and reports affine displacement
error, reactions, equilibrium, energy, and the residual mechanical-stress
diagnostic:

```rust
use sonium_sim_structural_mechanics::{run_thermoelastic_static, ThermoelasticStudy};

let result = run_thermoelastic_static(&ThermoelasticStudy::introductory())?;
assert!(result.max_affine_displacement_error_m < 1e-10);
assert!(result.max_mechanical_stress_pa < 1.0);
```

Linear beam buckling is exposed as a generalized elastic/geometric-stiffness
eigenproblem and checked against Euler's critical load:

```rust
use sonium_sim_structural_mechanics::{run_linear_buckling, BucklingStudy};

let result = run_linear_buckling(&BucklingStudy::introductory())?;
assert!(result.analytical_relative_error < 0.02);
```

This is a pre-buckling load-factor result; nonlinear post-buckling and
imperfection sensitivity are not inferred from it.

The remaining roadmap families are exposed as validated reduced reference
executions. `run_elastoplastic_bar` performs monotone small-strain return
mapping with Newton/continuation and reports committed plastic dissipation;
`run_frictionless_contact` solves a one-DOF unilateral penalty law and reports
penetration explicitly. These APIs do not claim 3D plasticity, friction, or
surface-contact capability.

Transient and fatigue evidence is available through
`run_transient_sdof` (Newmark average acceleration with equilibrium and energy
accounting) and `run_fatigue_damage` (Basquin stress-life blocks with Miner
damage). The fatigue result carries the explicit
`sonium-structural.cpu-reference.f64.fatigue-evidence.v1` provenance and states
that no FEM stress-history, multiaxial, mean-stress, or crack-growth claim is
made. `run_fem_fatigue_2d` is the bounded structural extension: it solves two
assembled P1 plane-stress load states per block, recovers the maximum element
stress amplitude, and then applies the scalar S-N/Miner gate. It carries
`sonium-structural.cpu-reference.f64.fem-fatigue-2d.v1`; it does not claim
multiaxial fatigue or crack-growth validation. The contact and reliability results carry analogous capability IDs for
their one-DOF penalty and empirical sample-counting reference models.
Shape optimization uses an analytic compliance/volume gradient and
body-fitted mesh motion, and reports material/load Taylor derivatives plus
bound-active design variables. The same bracket family also supports an
assembled displacement, centroid-stress, and first-bending-frequency target
objective; their values are re-evaluated on each morphed FEM mesh and their
thickness derivatives use a bounded finite difference with a Taylor gate. The
bracket also supports a
bounded linearized-buckling target objective when an explicitly compressive
axial preload is configured. The real 2D FEM topology workflow also supports
`FemTopology2DObjective::FirstFrequency` and
`FemTopology2DObjective::BucklingLoadFactor` in addition to compliance. The
frequency objective differentiates the assembled generalized eigenvalue with
respect to SIMP/RAMP densities, while the buckling objective re-solves each
perturbed density with a bounded finite-difference sensitivity. Results record
the selected objective, objective history, objective-improvement gate, Taylor
error, and the same volume, connectivity, stress, equilibrium, and
projected/body-fitted reanalysis gates.
Topology optimization includes density filtering, projection, SIMP/RAMP
interpolation, volume projection, connectivity, material/load Taylor
derivatives, and projected reanalysis gates.
`run_uncertainty_study` propagates deterministic bar samples
while reporting reuse of one normalized stiffness factorization. All of these
are deliberately reduced models with machine-readable provenance and should
not be read as validated general 3D optimization or fatigue/contact solvers.

For an actual Sonium FEM reanalysis, `run_bracket_shape_optimization` provides
a length/thickness bracket family. It applies `MeshMotion` transactionally,
reassembles the static solve on the accepted moved mesh, solves the FEM
adjoint for the thickness compliance derivative, and reports the adjoint
residual plus a Taylor finite-difference gate. Set
`BracketShapeOptimizationStudy::objective` to `FrequencyTarget` or
`BucklingTarget` to include the corresponding assembled eigenproblem in the
objective; the mode or buckling factor is re-solved for each candidate geometry.
`BucklingTarget` requires compressive axial preload and is a linearized
pre-buckling target, not nonlinear post-buckling analysis. The length derivative
remains a body-fitted reanalysis derivative until a geometry-level analytic
shape derivative is available.

The optional `meshlang` feature adds a parameterized CAD provenance and
meshing seam around that design loop:

```rust
#[cfg(feature = "meshlang")]
use sonium_sim_structural_mechanics::{
    run_meshlang_bracket_shape_optimization, BracketStudy,
    MeshlangBracketShapeOptimizationStudy,
};

#[cfg(feature = "meshlang")]
let result = run_meshlang_bracket_shape_optimization(
    &MeshlangBracketShapeOptimizationStudy::introductory(
        BracketStudy::introductory_bracket(),
    ),
)?;
assert!(!result.final_geometry.geo_code.is_empty());
```

Each initial/final design emits deterministic MeshLang source and retains the
compiler's canonical IR, sensitivity, provenance, and backend manifests. The
FEM loop remains the validated body-fitted plane-stress reanalysis; calling
`mesh_meshlang_bracket` additionally runs the MeshLang → Gmsh → Sonium mesh
bridge and therefore requires Gmsh on `PATH` (or `GMSH_PATH`/
`MESHLANG_GMSH`). This makes the CAD/volume-mesh hand-off explicit without
claiming that the reduced plane-stress optimizer is already a general 3D
shape optimizer.

`run_multi_load_study_reused` is the corresponding linear multi-load path: it
assembles and factorizes one constrained FEM operator, solves each load case
with a separate RHS, and reports factor memory plus residual/equilibrium gates.
The solid, shell, and topology multi-load families schedule independent cases
through a bounded four-worker CPU pool; each worker owns only its case-local
RHS/solution while the constrained factorization remains shared.

`run_structural_rom` is the narrow reduced-order path. It reuses assembled
beam modal vectors as a Galerkin basis for a nearby Young's-modulus query,
computes the exact full-order residual of the reduced displacement, and
enriches the basis with a full-order snapshot when the configured residual
gate is exceeded. The result records the reference/final basis dimensions,
tip error, and both reduced and full-order residuals. Its provenance is
`sonium-structural.cpu-reference.f64.structural-rom-beam.v1`; it is not a
general nonlinear, topology, contact, distributed, or accelerator ROM.

`run_robust_topology_optimization` extends the reduced topology reference with
SIMP or RAMP interpolation, projection-beta continuation, multiple load
factors evaluated through the bounded CPU sample pool, density-filtered analytic gradients, erosion/dilation volume
evidence, minimum feature runs, coarse-to-fine rank preservation, and a
central-difference Taylor gate. Its eroded compliance is an explicit
uncertainty/manufacturing screen; this remains a one-dimensional surrogate,
not a general 3D robust topology or body-fitted FEM reanalysis.

The assembled `run_fem_topology_2d` path also exposes a real P1 initial-stress
generalized eigenproblem for explicitly compressive x-directed load cases.
`buckling_load_factor` is optional for transverse-only cases and
`buckling_feasible` applies the study's minimum factor gate. The geometric
matrix uses the recovered compressive principal stress in the triangular
plane-stress model; mixed-stress shell, nonlinear post-buckling, and
imperfection sensitivity remain outside this bounded reference.

`run_fem_topology_2d_multi_load` verifies the accepted primary-load topology
against named additional load cases. It assembles one constrained operator,
factorizes it once, solves every RHS, and reports per-case compliance,
displacement, stress, equilibrium, energy, and residual gates plus the
critical case and factorization memory. This remains a verification API: the
topology is optimized for the primary case.
The aggregate benchmark suite also runs bounded small-mesh operator smoke cases
named `solid3d-static-amg-operator` and `solid3d-static-matrix-free-p2`; these
exercise solver selection and physical residual gates, but are not large-scale
performance or accelerator-parity evidence.
Its capability ID is
`sonium-structural.cpu-reference.f64.fem-topology-2d-multi-load.v1`.

For a genuine shared-load topology objective, use
`run_fem_topology_2d_envelope`. It solves every named load case from one
constrained sparse factorization at each density evaluation, forms the exact
maximum-compliance objective, and updates the common density field with the
active worst-case subgradient (averaged at ties). The result reports initial,
final, projected, and body-fitted per-case compliances, the governing case,
Taylor derivative evidence, connectivity, and factorization reuse. It is a
deterministic CPU/f64 P1 plane-stress reference; it does not imply a smooth
envelope at tied cases, general 3D topology, or accelerator execution.

`run_fem_topology_2d_robust` extends that genuine FEM path with bounded
manufacturing robustness. It first optimizes the exact load envelope, then
applies graph-distance element erosion and dilation to the accepted physical
density and re-solves every morphology against every named uncertainty case.
Uncertainty is explicit multiplicative load, Young's-modulus, and thickness
scaling; each morphology/uncertainty scenario reuses one constrained sparse
factorization across its load-case RHSs and reports compliance, displacement,
equilibrium, energy, convergence, and compliance-limit gates. Its capability ID
is `sonium-structural.cpu-reference.f64.fem-topology-2d-robust.v1`. This is a
CPU/f64 P1 plane-stress robustness reference, not a calibrated probabilistic or
general 3D robust topology optimizer.

`run_constrained_topology_optimization` adds explicit reduced-model gates for
aggregated axial stress, an effective first frequency, Euler buckling load
factor, connectivity/minimum feature length, density-slope overhang, and
projected reanalysis consistency. These gates are feasibility evidence on the
optimized density; they do not claim 3D stress, mode, buckling, or additive
manufacturing fidelity.

`run_structural_benchmark_suite` aggregates bracket, axisymmetric Lamé-cylinder,
orthotropic composite laminate, analytical beam static and modal, Mindlin plate,
thermoelastic free expansion,
linear and hyperelastic 3D solid, transient/random-vibration/reliability screening, scalar and FEM
fatigue, displacement- and force-controlled staggered phase-field damage,
Euler buckling, elastoplastic, and topology evidence into machine-readable
quality records covering error, residuals, feasibility, iterations, estimated
memory, and elapsed time. Its `external-adapter-decks`
case validates generation of schema-versioned CalculiX, Code_Aster, and
FEniCSx inputs. The suite reports executable availability but never launches
an external process by default. The structured `external_comparison` field
therefore remains `NotRun` until `run_external_benchmark` is explicitly
executed; adapter presence or executable discovery cannot make the parity gate
pass.

`run_nafems_benchmark_suite` provides a versioned validation catalogue with
static, modal, thermal-stress, and nonlinear NAFEMS verification axes. The
native cases use explicit analytical surrogates (cantilever, affine thermal
eigenstrain, and homogeneous Neo-Hookean extension), and the result sets
`official_reference_data_loaded` to `false` until official NAFEMS decks and
reference values are installed. This prevents analytical smoke results from
being reported as published parity. The Lamé thick-cylinder fixture is
available through `run_thick_walled_cylinder_analytical`; the separate
`axisymmetric-cylinder-plane-strain` structural benchmark is the executable
axisymmetric FEM comparison and remains explicitly bounded to that case.

Published values are loaded separately through the versioned
`PublishedReferenceManifest` JSON contract. A manifest must carry a stable ID,
citation, source revision, case applicability, units, tolerances, and declared
authority; loading it validates the schema and rejects duplicate or
non-finite case data. Compare it explicitly with
`compare_nafems_suite_to_published_manifest`:

```rust
use sonium_sim_structural_mechanics::{
    compare_nafems_suite_to_published_manifest, load_published_reference_manifest,
    run_nafems_benchmark_suite,
};

let suite = run_nafems_benchmark_suite()?;
let references = load_published_reference_manifest("references/nafems.json")?;
let report = compare_nafems_suite_to_published_manifest(&suite, &references)?;
assert!(report.reference_manifest_validated);
// `provenance_verified` remains false until the caller independently verifies
// the source and revision; a declared NAFEMS authority is not authentication.
```

The comparison report keeps missing native cases, kind mismatches, solver-gate
failures, unmatched suite cases, applicability limits, and relative errors
machine-readable. `all_passed` is false when the manifest omits any native
case, while `PublishedReferenceComparisonReport::parity_passed()` additionally
requires independent provenance verification. No official NAFEMS data is
bundled, so an analytical suite passing on its own remains an analytical
result rather than published parity; `NafemsBenchmarkSuite::official_parity_passed()`
therefore remains false until official reference data is loaded.

Structural result reports also carry `StructuralExecutionProvenance` and the
`structural_backend_capabilities()` catalogue. CPU/f64 remains the general
reference path. Enabling the `cuda` feature adds exactly one accelerator cell:
the introductory P1, small-strain, isotropic, linear plane-stress bracket. Its
mesh assembly and homogeneous constraint reduction remain on the CPU; the
checked constrained CSR operator, point-Jacobi forward solve, and Hermitian
adjoint solve execute through `sonium-compute`'s CUDA backend. The result
records transfers, peak/live device allocation bytes, timings, true f64 host
residuals, equilibrium/energy/reaction gates, CPU parity, and a compliance
gradient check. It rejects rather than silently falling back. P2, plane strain,
modal, buckling, nonlinear/contact, topology, mixed precision, and general
shape/material derivatives are not CUDA capability claims.

The real-hardware test skips when CUDA is unavailable unless promotion is
explicitly required:

```bash
CARGO_TARGET_DIR=.target-structural cargo test -p sonium-sim-structural-mechanics \
  --features cuda --test cuda_static_bracket
SONIUM_REQUIRE_CUDA_STRUCTURAL=1 CARGO_TARGET_DIR=.target-structural \
  cargo test -p sonium-sim-structural-mechanics --features cuda \
  --test cuda_static_bracket real_cuda_static_forward_adjoint_and_product_evidence -- --exact --nocapture
# Optionally persist the JSON evidence; use an absolute path because tests run
# with the crate as their current directory.
SONIUM_REQUIRE_CUDA_STRUCTURAL=1 \
SONIUM_CUDA_STRUCTURAL_EVIDENCE_OUTPUT=/absolute/path/cuda-static-bracket.json \
  CARGO_TARGET_DIR=.target-structural cargo test -p sonium-sim-structural-mechanics \
  --features cuda --test cuda_static_bracket real_cuda_static_forward_adjoint_and_product_evidence -- --exact
```

Native architecture evidence is an executable packet, not a cross-target
compile proxy. On native x86 it covers bounded linear-static, modal, buckling,
and topology cases and records compiler, CPU features, actual dispatch,
threads, deterministic settings, full observables, and a canonical payload
digest. Cross-architecture promotion requires a packet produced by the same
test on native aarch64; Rosetta output is supplemental only:

```bash
CARGO_TARGET_DIR=.target-structural cargo test -p sonium-sim-structural-mechanics \
  --test native_structural_evidence -- --nocapture
# On each native runner, persist a packet using an absolute output path.
SONIUM_STRUCTURAL_EVIDENCE_OUTPUT=/absolute/path/native-aarch64-packet.json \
  CARGO_TARGET_DIR=.target-structural cargo test -p sonium-sim-structural-mechanics \
  --test native_structural_evidence native_structural_packet_is_deterministic_and_machine_readable -- --exact
SONIUM_AARCH64_STRUCTURAL_REFERENCE=/path/to/native-aarch64-packet.json \
  CARGO_TARGET_DIR=.target-structural cargo test -p sonium-sim-structural-mechanics \
  --test native_structural_evidence native_x86_compares_with_supplied_accepted_aarch64_packet \
  -- --ignored --exact --nocapture
```

Metal and WASM remain explicitly unsupported for structural production solves.
This keeps portable JSON or lower-level backend availability from being
mistaken for a completed structural capability cell.

The external parity API is explicit and compares the 3D axial-solid result to
its closed-form displacement reference:

```rust
use sonium_sim_structural_mechanics::{
    run_external_benchmark, ExternalBenchmarkDefinition, ExternalRunOptions,
    ExternalSolverKind,
};

let definition = ExternalBenchmarkDefinition::default();
let result = run_external_benchmark(
    &definition,
    ExternalSolverKind::Calculix,
    &ExternalRunOptions::new("target/external-calculix"),
)?;
assert!(result.comparison.relative_error.is_finite());
```

Set `SONIUM_CALCULIX`, `SONIUM_CODE_ASTER`, or
`SONIUM_FENICSX_PYTHON` to select executables; FEniCSx additionally requires
the `dolfinx` Python package. Missing tools return a typed unavailable error,
non-zero exits return execution errors, and output without a positive
`displacement_m=`, `U1=`, or supported table value returns a parse error.
External runs write only below the caller-supplied work directory. Deck
generation and discovery are not external-solver parity: the comparison must
be explicitly executed and its tolerance must be reviewed for the selected
solver/version. `ExternalRunResult::evidence_sufficient` is true only when the
displacement came from the explicit common result marker; legacy numeric-table
fallbacks remain visible as `SolverTableHeuristic` and do not count as strong
solver-parity evidence.

`ExternalRunResult::provenance` records the versioned stages
`DeckRendered`, `ExecutableResolved`, `DeckWritten`, `ProcessCompleted`,
`OutputObserved`, and `NumericCompared`. Use `result.parity_passed()` (or the
comparison/provenance gates separately) before treating a run as external
solver parity. The capability-gated integration test is opt-in through
`SONIUM_RUN_EXTERNAL_INTEGRATION=1`; ordinary tests never launch an external
solver.

Dynamic uncertainty is covered by `run_random_vibration`, which integrates a
base-acceleration PSD through the exact SDOF transfer function and applies a
peak-factor displacement constraint. `run_reliability_study` reports empirical
failure probability and a Gaussian-equivalent margin index; it is explicitly
a screening gate, not certified FORM/SORM reliability.

`run_modal_random_vibration` consumes a converged mass-normalized modal
result together with caller-supplied base-acceleration PSD, modal
participation, and damping. It integrates each modal displacement PSD, reports
per-mode variance/peak diagnostics, and applies an aggregate independent-mode
SRSS displacement gate. Non-converged modes, mismatched controls, invalid
grids, and unresolved undamped resonances fail closed. This is bounded modal
screening evidence, not a correlated spatial PSD matrix, product-scale random
vibration FEM, fatigue certification, or measured correlation.

`compare_measured_field_validation` adds a versioned caller-supplied boundary
for multiple aligned displacement, von Mises stress, modal-frequency, and
already normalized modal-shape channels. Every channel has independent
absolute and relative gates, and the aggregate is accepted only when all
channels pass. Finite coordinates, unique channel IDs/locations,
quantity-specific cardinality, and optional SHA-256 input binding fail closed.
The result records CPU/f64 provenance and does not authenticate instruments,
calibration, modal sign/normalization, or product-scale solver coupling.

Run it with:

```bash
cargo run -p sonium-sim-structural-mechanics --bin structural-bracket-demo
```

By default this writes:

- `target/structural-bracket-demo/bracket_deformed_preview.vtu`
- `target/structural-bracket-demo/bracket_result.vtu`
- `target/structural-bracket-demo/bracket_report.json`

Open `bracket_deformed_preview.vtu` in ParaView for an immediate warped
displacement view, or open `bracket_result.vtu` and apply "Warp By Vector" to
the `displacement` field. Color either view by `von_mises_pa` or
`displacement_magnitude`.

## Deterministic gallery

Generate the static structural gallery headlessly:

```bash
cargo run -p sonium-sim-structural-mechanics --bin structural-bracket-demo -- --gallery target/structural-bracket-gallery
```

This writes the numerical report and VTU sources plus fixed PNG and SVG views
and `gallery.json`; each figure records source/output checksums and its display
conventions, including the deformation scale.

## Bounded 3-D topology mesh convergence

`run_solid3d_topology_mesh_convergence` runs the assembled CPU/f64 3-D SIMP
reference twice: once at the supplied structured mesh and once with the
structured `nx`, `ny`, and `nz` resolutions doubled. The returned
`Solid3DTopologyConvergenceReport` keeps the coarse and refined solver,
feasibility, objective, stress, volume, and density-rank diagnostics together.
The report is converged only when both runs pass their existing gates, the
objective/stress/volume relative changes are within the supplied tolerance, and
nearest-centroid projected density ordering stays within the rank-discordance
budget. This is a bounded multidirectional refinement contract, not general
product-scale mesh independence or official NAFEMS validation; the physical
model and structured aspect ratios remain fixed.

```rust
use sonium_sim_structural_mechanics::{
    run_solid3d_topology_mesh_convergence, Solid3DTopologyStudy,
};

let report = run_solid3d_topology_mesh_convergence(
    &Solid3DTopologyStudy::introductory(),
    0.75,
)?;
assert!(report.converged);
assert!(report.rank_preserved);
```
## Bounded 3-D hyperelastic contact

`run_hyperelastic_3d_contact` exposes a genuine small FEM contact reference:
it assembles two disconnected P1 tetrahedral Neo-Hookean blocks with an
initial gap, prescribes incremental motion on the upper block, and calls
Sonium FEM's augmented-Lagrangian Newton/Uzawa contact solver on the opposing
faces. The versioned result retains contact-face counts, per-step Newton/Uzawa
residuals, active-pair counts, engagement step, penetration diagnostics, and
CPU/f64 provenance.

```rust
use sonium_sim_structural_mechanics::{
    run_hyperelastic_3d_contact, Hyperelastic3DContactStudy,
};

let result = run_hyperelastic_3d_contact(&Hyperelastic3DContactStudy::introductory())?;
assert!(result.converged);
assert!(result.contact_engaged_step.is_some());
```

This is displacement-controlled opposing-face engagement evidence, not a claim
of frictional/self-contact, force-controlled contact, post-buckling, or
product-scale contact validation.
## Bounded 3-D hyperelastic force control

`run_hyperelastic_3d_force` exposes a small force-controlled total-Lagrangian
reference using a P1 tetrahedral Neo-Hookean block. The lower face is fixed,
the upper face is guided laterally, and a distributed downward load is applied
through eight incremental load steps using the production FEM Newton solver's
explicit external-load vector:

```rust
use sonium_sim_structural_mechanics::{
    run_hyperelastic_3d_force, Hyperelastic3DForceStudy,
};

let result = run_hyperelastic_3d_force(&Hyperelastic3DForceStudy::introductory())?;
assert!(result.converged);
assert!(result.final_average_top_displacement_m < 0.0);
```

The versioned result retains mesh/DOF counts, per-step force, Newton
iterations and residuals, average top displacement, and CPU/f64 provenance.
The introductory fixture has 18 nodes, 24 tetrahedra, and 54 displacement
DOFs. This is bounded monotone static evidence for external-load assembly; it
does not claim force-controlled contact, snap-back or post-buckling tracking,
calibrated material behavior, or product-scale solver parity.

## Bounded 3-D hyperelastic mesh convergence

`run_hyperelastic_3d_mesh_convergence` runs the production displacement-
controlled total-Lagrangian `run_hyperelastic_3d` FEM/Newton path on a coarse
structured P1 tetrahedral mesh and a uniformly refined mesh with doubled
`nx`, `ny`, and `nz`. The versioned report retains both nonlinear results,
topology/DOF refinement, relative strain-energy and maximum-displacement
changes, and refined-field affine displacement, deformation-gradient, and
constitutive-gradient diagnostics. Newton convergence, refinement topology,
response changes, and all refined-field/gradient gates are explicit checked
controls. The introductory test uses a bounded 1x1x1 to 2x2x2 ladder, two
load steps, and 0.02 m prescribed extension.

This is bounded actual nonlinear CPU/f64 spatial-convergence evidence. It is
not nonlinear transient FEM, shell/solid post-buckling, calibrated material
or manufacturing validation, product-scale mesh/time/reliability validation,
official NAFEMS/current-SOTA or external-solver parity, or accelerator/device
evidence.

## Bounded assembled FEM transient

`run_transient_fem` promotes the bracket's transient step to a checked
assembled 2-D FEM reference. It reuses the selected P1/P2 stiffness and
consistent-mass assembly, applies a caller-supplied load-scale history, and
integrates the reduced system with Newmark average acceleration and
mass-proportional damping calibrated at the assembled first mode:

```rust
use sonium_sim_structural_mechanics::{run_transient_fem, TransientFemStudy};

let result = run_transient_fem(&TransientFemStudy::introductory())?;
assert!(result.converged);
```

The result retains each sample's displacement/velocity/acceleration maxima,
dynamic equilibrium residual, total energy, damping power, final fields, and
CPU/f64 provenance. The initial load sample must be zero and the trajectory is
bounded by a checked step budget. This is linear prescribed-load 2-D FEM
evidence; nonlinear dynamics, contact, transient topology optimization,
distributed/device execution, and external-solver parity remain open.

## Bounded nonlinear 3-D transient FEM

`run_nonlinear_transient_fem_3d` advances a bounded force history with
implicit Newmark average acceleration. Each step solves the actual production
total-Lagrangian P1 tetrahedral Neo-Hookean residual and tangent augmented by
consistent mass and mass-proportional damping. The versioned result retains
per-step Newton iterations, dynamic-equilibrium residual, displacement,
velocity, acceleration, kinetic/strain/total energy, damping power, input
work, dissipation, and CPU/f64 provenance. The first load sample must be zero;
resolution, time-step, force-history, damping, and Newton work are bounded and
fail closed.

The focused reference uses a small guided block and a reversible force history;
the new target passes 4/4, and the existing scalar nonlinear-transient target
passes 3/3. This is bounded actual nonlinear transient FEM evidence, not
contact/fracture dynamics, shell dynamics, solid post-buckling, topology
updates, calibrated measured loading, product-scale time/mesh/reliability
validation, external-solver parity, or accelerator/device evidence.

## Bounded 3-D static mesh convergence

`run_solid3d_static_mesh_convergence` runs the production 3-D solid static
path on a caller-supplied coarse mesh and a uniformly refined mesh with doubled
`nx`, `ny`, and `nz`. The versioned report retains both static results and
relative displacement, strain-energy, and element-centroid von-Mises changes.
Displacement and strain energy are hard convergence gates; centroid stress is
diagnostic-only under the crate's stress-singularity policy. This is bounded
structured CPU/f64 evidence and does not claim product-scale mesh independence,
official NAFEMS/current-SOTA parity, calibrated manufacturing, measured
validation, external-solver execution, or accelerator parity.

## Bounded 3-D harmonic mesh convergence

`run_solid3d_harmonic_mesh_convergence` composes the direct assembled harmonic
solver on a coarse mesh and a uniformly refined mesh. It resolves the selected
observation point by physical coordinate on the refined mesh, then gates
topology refinement, per-frequency displacement changes, wrapped phase
changes, peak displacement, peak frequency, and both solver convergence states.
The report retains both full response results and per-frequency diagnostics.
This is bounded CPU/f64 harmonic mesh evidence, not product-scale mesh
independence, nonlinear/transient or modal-coupled analysis, measured or
official reference parity, external execution, manufacturing validation, or
accelerator/device parity.

## Bounded direct assembled 3-D solid harmonic response

`run_solid3d_harmonic_response` assembles the production P1 3-D elasticity
stiffness and consistent vector mass matrices, applies the production fixed-face
and loaded-face traction policy, and solves one complex dynamic system per
caller-supplied frequency. It retains full nodal complex displacement phasors
per frequency and reports the selected observed phasor, magnitude,
principal-value phase lag under `exp(iωt)`, solver residual, peak frequency,
feasibility, and factorization-memory provenance. Mass-proportional damping is
explicit as `C = 2 ζ ω_ref M`, and complex force phasors preserve caller-
supplied force phase. The field is numerical solver output, not authenticated
measured data.

The focused direct-response target passes 6/6; related modal, convergence, and
measured-validation targets pass, and the structural library suite passes
138/138. No QA binary, full simulation, or external solver ran. This is bounded
linear P1 CPU/f64 reference evidence, not nonlinear/transient or modal-coupled
analysis, product-scale mesh convergence, measured or official reference
parity, manufacturing validation, external execution, or accelerator/device
parity.

## Bounded 3-D modal harmonic response

`run_modal_harmonic_response` consumes a converged mass-normalized
`Solid3DModalResult` and caller-supplied complex modal-force phasors. It
reconstructs the observed displacement phasor at a selected node/component
over a checked frequency grid, reporting per-mode contributions, magnitude,
principal-value phase lag, peak frequency, and an allowable-displacement gate.
The response uses the `exp(iωt)` convention and rejects non-mass-normalized,
malformed, non-converged, or unresolved-resonance inputs. This is bounded linear
modal CPU/f64 evidence, not direct complex 3-D FEM assembly, modal coupling,
nonlinear/product-scale harmonic analysis, measured correlation, external
parity, or accelerator evidence.
