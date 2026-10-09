# sonium-sim-pcfan (lib: `sonium_sim_pcfan`, version: 0.3.0)

## Verification update — checked resolved FEM acoustic assembly bridge (2026-08-27)

Added `assemble_pc_fan_resolved_aeroacoustic_pde`, which validates the structured CFD domain, converts it to a FEM hexahedral mesh, assembles the Linearized Euler operator, and constructs a checked annular harmonic continuity source/RHS with explicit provenance. This is a forward assembly boundary only: it does not solve the complex acoustic system or provide the missing production aeroacoustic derivative bundle/adjoint. Focused assembly tests (3), package check, clippy, edited-file formatting, and diff checks pass. No QA binary or full simulation run.

## Verification update — resolved adjoint admission hardening (2026-08-27)

Newton linearization validation now rejects `black-box` and `derivative-probe` provenance, and production derivative-bundle construction rejects malformed or non-finite objective derivative vectors before the bundle can be retained. Focused resolved-adjoint tests, package check, clippy, edited-file formatting, and diff checks pass; no QA binary or full simulation run.

## Verification update — transient adjoint backend provenance (2026-08-27)

Production transient-adjoint wrappers now carry the concrete backend recorded on each accepted trajectory step, preserving explicit CPU fallback provenance instead of copying the requested configuration. A focused library regression passes; no QA binary or full simulation run.

## Verification update — measured prediction provenance (2026-08-27)

Measured-product comparison now rejects reserved `synthetic` and `measured-product` provenance markers wherever they occur in a prediction label, preventing wrapped or delegated fixture labels from passing the independence gate. A focused library regression passes; no QA binary or full simulation run.

## Verification update — transient backend compatibility admission (2026-08-27)

Transient-adjoint admission now rejects trajectory backend labels incompatible with the configured request while allowing the FEM dispatcher’s explicit CPU fallback for CUDA/Metal requests. Focused library regression passes; no QA binary or full simulation run.

## Verification update — implicit aeroacoustic PDE adjoint seam (2026-08-27)

Added checked `PcFanResolvedAeroacousticPdeDerivativeData` and a coupled implicit adjoint that solves the acoustic transpose system, feeds its flow-state coupling into the resolved-flow transpose, and contracts acoustic plus flow design residuals. Analytic provenance, dimensions, finiteness, and acoustic residual budgets are admitted; finite-difference/black-box data fail closed. Production FEM acoustic source assembly and hardware/scale evidence remain unclaimed. Focused library tests pass; no QA binary or full simulation run.

## Verification update — transient trajectory finite-field admission (2026-08-27)

Production transient state-adjoint admission now rejects non-finite retained
state/history vectors, residual/design rows, transported fields, observations,
or absolute conservation-ledger values even when summary residual budgets are
otherwise finite. Focused regression coverage passes; no QA binary or full
simulation run.

## Verification update — transient BDF backend provenance (2026-08-27)

Cached FEM BDF telemetry now records the resolved linear-solve backend and
iteration count, and each accepted PC-fan transient trajectory step publishes
the actual backend. Configured CUDA/Metal requests use the backend dispatcher
when available; unavailable or failed accelerator setup is explicitly retained
as CPU fallback provenance rather than being promoted to hardware execution
claims. Transient-adjoint admission checks retained per-step conservation
ledgers and transported scalar residuals as well as the final-window budget.
Focused FEM and PC-fan library tests pass; no QA binary or full simulation was
run.

## Verification update — accelerator parity admission (2026-08-27)

`evaluate_pc_fan_accelerator_parity` now requires each observation's requested backend to equal its actual backend, rejects backend substitution, and only accepts a host fallback when `allow_host_fallback` is explicit. This prevents a CPU result from being reported as CUDA/Metal parity. Focused accelerator-parity tests pass; this is an admission contract and does not claim hardware execution. No QA binary or full simulation was run.

## Verification update — adjoint-independence evidence (2026-08-27)

Added `PcFanCfdAdjointIndependenceSample`, per-axis studies, and a checked
`evaluate_pc_fan_cfd_adjoint_independence` report. The contract requires
finite, nonzero, equal-dimensional gradients with analytic adjoint
provenance; monotonic mesh/time/revolution/iteration ladders; bounded state
and adjoint residuals; and adjacent-gradient relative-change/cosine gates.
Malformed, diagnostic, finite-difference, and nonconverged evidence cannot be
promoted. Focused library tests pass; the full library run was stopped before
completion to avoid crossing the no-full-simulation boundary. No QA binary or
full simulation was run.

## Verification update — fixed-topology remesh map design derivative handoff (2026-08-27)

`PcFanCfdTransientRemeshMapDesignDerivative` provides a checked caller
contract for analytic fixed-topology remesh-map design derivatives. The
    schedule adjoint contracts sparse current/previous velocity and pressure map entries
with composed stage-initial history adjoints and reports a separate map
gradient. Diagnostic/black-box provenance, malformed dimensions, non-finite
values, and mismatched predecessor states fail closed; discrete remesher
selection and connectivity/topology derivatives remain inadmissible. Focused
library tests pass; no QA binary or full simulation was run.

## Verification update — fixed-mesh aeroacoustic output-map derivative handoff (2026-08-27)

`PcFanResolvedAeroacousticDerivativeData` now composes caller-retained
analytic one-way acoustic state/design Jacobians into the resolved Newton
adjoint by the exact chain rule. Dimensions, finiteness, and provenance are
checked before publication; diagnostic finite-difference/black-box paths are
rejected. This closes derivative transport infrastructure only: acoustic
source/PDE assembly, topology-changing mesh derivatives, and accelerator or
production-scale evidence remain outside the contract. Focused library tests
pass; no QA binary or full simulation was run.

## Verification update — zero-load acoustic state contract (2026-08-27)

Reduced-order acoustic source and state paths now preserve true zero harmonic
pressure; finite floors remain confined to dB and sound-power diagnostics. The
reduced adjoint rejects a harmonic at that diagnostic floor because the clipped
logarithmic objective is not differentiable there. Focused acoustic and
reduced-adjoint library regressions pass. Full resolved-CFD aeroacoustic and
design derivatives remain fail-closed; no QA binary or full simulation was run.

## Verification update — fixed-map remesh schedule design adjoint (2026-08-27)

`solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_derivatives`
now contracts caller-retained per-step residual/objective design derivatives
across a multi-stage fixed interpolation-map remesh schedule. The result
publishes per-stage and aggregate `phi_d - R_d^T lambda` gradients while
retaining explicit fail-closed scope for remesher selection, topology,
geometry, and interpolation-map derivatives. Focused library tests pass; no
QA binary or full simulation was run.

## Verification update — fixed-map remesh schedule state objective adjoint (2026-08-27)

The production full-CFD laminar transient surface now composes state/objective
adjoints across multiple external remesh stages. It consumes each stage's
published BDF startup-history blocks, routes current and previous velocity
adjoints plus shared pressure through the checked P2/P1 interpolation
transpose, and returns chronological stage reports. The result is explicitly
fixed-map only: remesher topology/geometry and topology-changing design
derivatives remain inadmissible. Focused library regressions cover one-stage,
multi-stage, startup-level routing, and missing-history rejection. No QA binary
or full simulation run.

## Verification update — forward startup history publication (2026-08-27)

The production BDF trajectory now retains checked residual Jacobians for
external startup histories: first-order slot 0 and second-order startup slot
1 use the same mass/history assembly and boundary-row treatment as ordinary
trajectory history blocks. The retained-history adjoint entry point consumes
these blocks directly. This is still per-trajectory fixed-dimensional data;
remesh transfer and topology/geometry derivatives remain separately guarded.
No QA binary or full simulation run.

## Verification update — production stage-initial history wrapper (2026-08-27)

The production resolved transient state-adjoint surface now accepts caller-
retained per-stage incoming BDF history Jacobians and routes them through the
checked generic stage-initial-history contraction. A manufactured trajectory
test verifies the production conversion and `-Hᵀλ` handoff. This remains a
fixed-dimensional incoming-history contract; remesher topology/geometry,
transported production coupling, and full schedule objective derivatives stay
fail-closed. No QA binary or full simulation run.

## Verification update — stage-initial BDF history adjoint contract (2026-08-27)

The generic resolved transient adjoint now accepts checked per-stage,
per-history-slot incoming BDF residual Jacobians and returns their exact
residual-induced `-Hᵀλ_stage` derivatives. Focused direct-contraction,
multi-stage recurrence, and malformed-input tests pass. This remains an
incoming-history contract only; remesher topology/geometry, transported
production, full schedule objective, and full-simulation evidence remain
outside scope. No QA binary or full simulation run.

## Verification update — transported scalar-to-momentum bridge (2026-08-26)

The FEM momentum configuration now accepts a checked lagged transported eddy-viscosity coefficient. P1 k--omega states publish per-element viscosities, and the PC-fan adapter validates and applies their target-mesh mean to the next segregated momentum assembly with zero scalar-to-velocity derivative. Heterogeneous element transport, viscosity publication, momentum override, and adapter regressions pass. This remains a segregated Picard bridge, not an exact coupled turbulent Newton/BDF Jacobian or adjoint; body-fitted remeshing execution, full aeroacoustics, hardware parity, and production-scale evidence remain open. No QA binary or full simulation run.

## Verification update — checked transported k--omega scalar step (2026-08-26)

Added the FEM `KOmegaTransportState`/`advance_komega_bdf_step` contract and a PC-fan P1 adapter over the transferred k--omega history state. The step assembles target-mesh BDF mass/history plus frozen-coefficient transport/reaction operators, solves both scalar equations with sparse LU, checks finite/positive history and solutions, and publishes residual/provenance diagnostics. Focused FEM turbulence and PC-fan adapter tests pass. This is scalar Picard transport only: mean-flow eddy-viscosity coupling, exact coupled turbulent Newton/BDF Jacobians and adjoints, body-fitted remeshing execution, and production-scale evidence remain open. No QA binary or full simulation run.

## Verification update — measured comparison and remesh-chain handoff (2026-08-26)

The checked measured-product surface now compares independently provenance-
labelled predictions against checksummed records using exact RPM topology,
finite-field admission, and explicit per-observable RMSE budgets. The P2/P1
BDF transfer API also supports multiple topology-changing remesh stages,
transfers both retained velocity levels, publishes per-stage diagnostics, and
requires target operator reassembly. A checked P1 k--omega BDF-history handoff
now transfers current and previous scalar levels through the same mesh chain
with positivity/finiteness admission and mandatory target operator assembly.
Focused tests and compile-only synthetic QA targets cover these contracts.
External product data, body-fitted remesher execution, transported k--omega
equations, exact turbulent adjoints, full aeroacoustic derivatives,
hardware parity, and production-scale evidence remain open; no QA binary or
full simulation was run.

## Verification update — transient BDF restart handoff (2026-08-25)

The production rotating-wall transient API now accepts a checked source
`BdfState` from another affine tetrahedral topology. Current and previous P2
velocity levels plus P1 pressure are transferred onto the target ALE mesh;
target BDF matrices and history operators are reassembled rather than copied.
Time/step metadata and finite DOF dimensions are checked, and transfer errors
are returned before solving. Focused state-transfer and runner fail-closed tests
pass. This is a state restart contract, not transported turbulence history,
topology-dependent operator persistence, exact turbulent adjoints, or QA/full
simulation evidence. No QA binary or full simulation was run.

## Verification update — transient P2/P1 state transfer (2026-08-25)

The checked state-transfer surface now also covers the transient Taylor--Hood
layout: interleaved P2 velocity DOFs and P1 pressure DOFs are transferred
component-wise through FEM H1 interpolation across a changed affine
tetrahedral topology. Affine fields, consistent negative FEM orientation, and
malformed/non-finite state rejection have focused regressions. The higher-order
path deliberately does not apply the P1 nodal conservation correction. BDF
field levels are supported by the transient restart handoff, while
topology-dependent BDF operators, turbulence history, and body-fitted
remeshing metadata are reassembled or remain outside scope. No QA binary or
full simulation was run.

## Verification update — production state-transfer runner handoff (2026-08-25)

The production P1 mean-flow API now consumes checked source state transferred from a different generated tetrahedral topology before solving the target mesh. The transfer contract accepts either consistent FEM orientation convention, rejects mixed/inverted elements and source/target convention mismatches, and preserves transactional failure behavior. Focused coarse-to-fine runner and 11-case transfer tests pass. This is a remesh/restart handoff, not a body-fitted remesher or higher-order, turbulence-history, transient-BDF, aeroacoustic, accelerator, measured-product, or production-scale closure. No QA binary or full simulation was run.

## Verification update — transient turbulent adjoint admission (2026-08-25)

The transient state-only resolved adjoint now rejects non-laminar PC-fan configurations. Its published trajectory blocks are not yet exact per-step turbulent Newton/BDF operators; the checked algebraic k-omega tangent remains a steady-state contract only. A focused regression covers the fail-closed boundary. Transported turbulence, topology-changing remeshing, full aeroacoustic derivatives, accelerator parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.

## Verification update — fixed-state rotating-wall ALE objective derivative (2026-08-25)

The checked laminar full-CFD rotating-wall objective design gradient now includes P1/P2 fixed-reference velocity evaluation, ALE centroid and force-position terms, and tetrahedral volume terms for hub, tip, and shroud controls. A focused P2 centered-difference library regression passes for axial loading and shaft power. This covers topology-preserving ALE objective terms; topology-changing remeshing, turbulence, aeroacoustics, accelerator parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.

PC fan aeroacoustic simulation and shape optimization. Couples a structured-grid FEM mean-flow solve with the FEM aeroacoustic source model, driven by L-BFGS over blade and shroud design variables.

## Structural calibration boundary

The `structural_calibration` API is an admission contract for provenance-labelled
RPM-aligned datasets. It compares four thermal/structural observables with
absolute RMSE and maximum-relative-error budgets and only predicts through the
reduced-order flow model. Its QA reference uses a deterministic synthetic
fixture for compile coverage; it must not be described as measured-product or
resolved-flow evidence.

The `measured_product` module is the only adapter that may label a dataset as
independent product evidence: it requires instrument/calibration metadata and
raw-data checksum provenance before producing aero or structural datasets.
`resolved_adjoint` consumes an exact sparse CFD tangent supplied by the solver;
do not replace it with black-box finite differences. `accelerator_parity`
requires explicit actual-backend provenance and rejects silent host fallback.

The production FEM Picard path publishes the map-input state and update
residual through `evaluate_pc_fan_cfd_picard_map`. The checked
`probe_pc_fan_cfd_resolved_flow_derivatives` seam is for centered state/design
Taylor diagnostics only; its finite-difference provenance is rejected by the
exact resolved-flow adjoint until a full Newton tangent and all active
design-residual derivatives are supplied.

The laminar actuator-disk and rotating-wall CFD paths publish frozen-mesh
analytic body-force design residual Jacobians alongside their Newton tangents,
including hub/tip span occupancy controls while the FEM mesh remains fixed.
Topology-changing shroud/mesh motion, turbulence, and full aeroacoustic
objective derivatives remain fail-closed; fixed-mesh shroud occupancy is
covered. Objective derivatives entering the convenience
adjoint API require explicit analytic provenance.

The checked CFD objective-admission boundary independently verifies the
published actuator/wall loading identity `shaft_power = torque * angular_speed`
and cross-checks copied axial-force and shaft-power fields in the conservation
ledger. It also recomputes mass, momentum, pressure-closure, and energy
residual identities from primitive ledger fields. Positive loading and a
self-reported residual are not enough to promote a stale or corrupted ledger.

The production rotating-wall transient runner validates the nonlinear tolerance
before mesh construction and marks each BDF observation converged only when the
step succeeds, all retained fields are finite, and its residual is within the
configured budget. A failed final window returns `NonConvergedObservation`.

Steady mesh-convergence sweeps likewise require the validated
`max_final_residual` budget at every participating level, in addition to
observable agreement and solver convergence.

Transient rotating-wall observations also compute and publish a final-window averaged control-volume conservation ledger. Invalid conservation tolerances fail before mesh setup, and retained BDF steps/final-window admission require mass, axial-momentum, pressure-closure, and mechanical-power residuals within the configured budgets.

## Layout

```
src/
  model.rs              PcFanConfig, design variables, mesh + weights + optimizer config
  geometry.rs           Frame box, shroud, hub, struts, blade panels, VTU + intersection check
  cfd.rs                ReducedOrder + IncompressibleCfd mean-flow, immersed penalties,
                        actuator-line forcing, boundary conditions, VTU
  evaluation.rs         Flow + acoustics + objective + explicit derivative policy
  optimizer.rs          L-BFGS warm-restart loop with airflow-band lock
  meshlang_geometry.rs  Alternative MeshLang DSL geometry path
  bin/pcfan_opt.rs      `pcfan-opt` binary
shapes/
  pcfan.mlg, pcfan.msh  Reference MeshLang fan and compiled Gmsh mesh
```

## Key Types

- `PcFanConfig` — top-level config: dimensions, operating point, mesh resolution, weights, optimizer, broadband, CFD config, flow model
- `PcFanDesignVariables` — `blade_path` controls (×N), per-station `blade_sections` (chord, twist, camber, thickness), hub/tip radius offsets, and `shroud_radius_offsets_m`
- `PcFanFlowModel::ReducedOrder | IncompressibleCfd` — switch between fast objective and CFD validation
- `PcFanCfdBoundaryKind` — Inlet, Outlet, OuterWall, ShroudWall, SupportWall, RotatingHub, RotatingBlade

## Optimization Loop

1. `solve_pc_fan_mean_flow` (or full CFD) → airflow, pressure rise, Reynolds
2. `solve_pc_fan_acoustics` → tonal harmonics, broadband, fixed-hardware noise
3. `evaluate_pc_fan` → objective scalar
4. `solve_pc_fan_shape_gradient` → finite-difference full-objective gradient. `PcFanDerivativePolicy::ReducedOrderDiscreteAdjoint` provides the checked reduced-order full-state adjoint over airflow and harmonic acoustic pressures. Resolved CFD now admits checked laminar full-state adjoints for airflow, pressure rise, actuator loading, and topology-preserving rotating-wall loading/power; full acoustic derivatives, turbulence, topology-changing motion, and external-fidelity gates remain fail-closed. Use `PcFanDerivativePolicy::DerivativeFree` with `optimize_pc_fan_with_derivative_policy` when no gradient contract is appropriate; policy-aware gradient requests fail closed.
5. L-BFGS step under collision and shroud-clearance penalties
6. Airflow-band lock warm-restarts when the optimised design exits the ±band of the target

The reduced-order discrete-adjoint path now solves airflow and harmonic
acoustic-pressure state equations and exposes non-zero pressure multipliers;
its gradient parity-tested full objective. Resolved-CFD requests outside the checked laminar objective subset remain
fail-closed.

## Objective Components

Current derivative policy also exposes the checked reduced-order full-state
airflow/harmonic-pressure adjoint (`solve_pc_fan_shape_gradient_reduced_adjoint`
under `PcFanDerivativePolicy::ReducedOrderDiscreteAdjoint`). Resolved-CFD requests outside the checked laminar objective subset remain
fail-closed. Reduced-order optimizer
candidates undergo explicit bounded feasibility restoration before promotion;
the existing finite-difference and derivative-free policies remain available
for their declared contracts.

- Acoustic level (tonal): finite-difference gradient in the current CPU implementation
- Airflow penalty (squared): finite-difference gradient in the current CPU implementation
- Collision penalty: heavy weight (1e6) — must stay zero
- Spectral spread / directivity spread: FD only
- Tip-leakage broadband: FD only
- Shroud clearance / frame containment penalties

## Testing

```bash
cargo test -p sonium-sim-pcfan --release
cargo check -p sonium-sim-pcfan && cargo clippy -p sonium-sim-pcfan
```

The test suite covers blade intersection validity, CFD boundary tagging, fan-forced (not target-prescribed) flow, immersed-penalty DOF accounting, full-objective finite-difference gradient consistency (rel. error < 5e-3), reduced-order discrete-adjoint parity and routing, nominal and manufacturing-corner operating-envelope validation, bounded feasibility restoration, and end-to-end optimiser convergence.

Focused tests cover robust RPM/manufacturing envelope scoring and the
reduced-order L-SHADE promotion path; resolved-flow and measured-product
optimization remain unsupported.

`predict_pc_fan_correlation_dataset` now supplies provenance-labelled
reduced-order predictions to the measured-correlation contract; external
instrument acquisition and measured-product acceptance remain open.

## Notes

- Frame mounts are fixed; deformation only touches blades and shroud.
- Solid fan surfaces are immersed penalties — boundary DOF count counts only side-wall nodes.
- For adjoint testing, neutralise the band/spread/broadband weights so only the airflow + tonal path is exercised.
- `optimize_pc_fan` is monotone-improving on the deterministic small-mesh case (objective never worse than baseline).
- `PcFanDerivativePolicy` makes the finite-difference, reduced-order discrete-adjoint, and derivative-free contracts explicit; resolved-CFD adjoint requests outside the checked laminar objective subset remain unsupported.
- The reduced structural screen includes a validated Basquin fatigue margin in
  addition to thermal stress, modal separation, imbalance, and manufacturing
  constraints; it is not product-scale S-N or fatigue validation.
`evaluate_pc_fan_fidelity_promotion` is a fail-closed reduced-order-to-higher-fidelity admission contract. It checks provenance, candidate topology, finite metrics, fidelity labels, objective error, rank preservation, feasibility parity, and best-candidate movement. The synthetic QA fixture is compile-only contract evidence, not resolved-flow or measured-product evidence. `PcFanCfdIndependenceConfig::max_final_residual` is a serde-defaulted absolute residual gate; observable agreement and solver convergence flags cannot promote a ladder whose finest residual exceeds that budget.
## Verification update — CFD independence conservation evidence (2026-08-25)

`PcFanCfdIndependenceSample` now retains mass, axial-momentum, pressure-closure, and energy-imbalance metrics. Production mesh, nonlinear, and rotating-wall transient adapters copy the configured conservation budgets; invalid budgets, non-finite metrics, and over-budget samples fail closed before independence promotion. Focused independence tests and the full PC-fan library suite pass. No QA binary or full simulation was run.
## Checked resolved-CFD finite-difference gradients

Resolved-CFD finite-difference gradients admit the base and each plus/minus perturbation only after checked convergence, conservation, loading, field-shape, and objective admission. This is a checked finite-difference fallback, not an exact resolved-flow discrete adjoint. Focused policy tests pass; topology-changing mesh motion, turbulence fidelity, full aeroacoustic derivatives, accelerator runtime parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.
## Verification update — topology-preserving ALE mesh motion (2026-08-25)

Resolved steady, rotating-wall transient, and checked finite-difference paths now apply a compactly supported radial ALE deformation for hub, tip, and shroud-radius offsets. Connectivity is preserved; the outer inlet/outlet/side boundaries stay fixed; non-finite, inverted, or collapsed elements fail closed through an element-measure barrier. Focused motion, steady, transient, and derivative tests pass. This is topology-preserving immersed-mesh motion, not a body-fitted remesher or an exact mesh-motion shape adjoint; turbulence, full aeroacoustic derivatives, accelerator parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.
## Verification update — algebraic turbulence state tangent (2026-08-25)

The algebraic k-omega surrogate now publishes a checked Newton state tangent when its global velocity peak is unique and non-zero, including the viscosity clamp derivative and stabilization-viscosity contribution. Resolved state-only adjoints may consume this provenance-labelled tangent; turbulence design, mesh-motion, transported k-omega/RANS, and full aeroacoustic derivatives remain fail-closed. Focused FEM and PC-fan tests pass. No QA binary or full simulation was run.
