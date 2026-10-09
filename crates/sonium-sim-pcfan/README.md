# sonium-sim-pcfan

## Current capability clarification (audit follow-up)

The dated updates below describe incremental milestones, not a complete current
capability matrix. `src/transported_turbulence.rs` now contains transported
k--omega residuals, coupled/monolithic current-state Jacobians and transient
history operators. Its viscosity bridge retains the element-local viscosity
field in `transported_eddy_viscosity_per_element`; the mean is a summary, not
the only momentum input. These implementations do not establish validated
production turbulent fan CFD, complete aeroacoustic derivatives or accelerator
runtime parity. Admission and tests must be checked for the particular steady,
transient and adjoint entry point.

## Verification update — transported scalar-to-momentum bridge (2026-08-26)

The FEM/PC-fan path supports heterogeneous element-local k--omega transport and a checked scalar-to-momentum eddy-viscosity bridge. P1 states publish target-mesh element viscosities, retained by the momentum config for assembly along with their mean. Coupled derivatives are supplied by separate checked linearization APIs, not by the value-only bridge. Production-scale CFD remains unclaimed; the original milestone reported focused tests, not a QA binary or full simulation run.

## Verification update — measured comparison and remesh-chain handoff (2026-08-26)

Independent measured-product prediction validation now has strict provenance,
RPM-topology, finite-field, and RMSE-budget checks. A checked multi-stage
P2/P1 BDF state-transfer chain carries current/previous velocity levels across
changed tetrahedral meshes and marks target operator reassembly mandatory.
The checked chain also carries P1 k--omega current/previous history with
finite, non-negative k and strictly positive omega admission. This is a
history handoff contract only; it does not itself validate the transported
k--omega equations or coupled turbulent adjoints implemented in other paths.
Coverage is focused tests plus compile-only synthetic QA; it is not external
product evidence or a full remeshed CFD run.

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

The production P1 mean-flow API now consumes checked source state transferred from a different generated tetrahedral topology before solving the target mesh. Either consistent FEM orientation convention is supported; mixed/inverted elements and source/target convention mismatches fail closed. Focused coarse-to-fine runner and 11-case transfer tests pass. This is a remesh/restart handoff, not a body-fitted remesher or higher-order, turbulence-history, transient-BDF, aeroacoustic, accelerator, measured-product, or production-scale closure. No QA binary or full simulation was run.

## Verification update — transient turbulent adjoint admission (2026-08-25)

The transient state-only resolved adjoint now fails closed for non-laminar configurations because the trajectory does not yet publish exact turbulent Newton/BDF history operators. The algebraic k-omega tangent is admitted for checked steady-state paths only. A focused regression covers this boundary; transported turbulence, topology-changing remeshing, full aeroacoustic derivatives, accelerator parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.

## Verification update — fixed-state rotating-wall ALE objective derivative (2026-08-25)

The checked laminar full-CFD rotating-wall objective design gradient now includes P1/P2 fixed-reference velocity evaluation, ALE centroid and force-position terms, and tetrahedral volume terms for hub, tip, and shroud controls. A focused P2 centered-difference library regression passes for axial loading and shaft power. This covers topology-preserving ALE objective terms; topology-changing remeshing, turbulence, aeroacoustics, accelerator parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.

The empirical basis, uncertainty ranges, and measurement plan for airflow,
broadband levels, and chassis damping are documented in
[`CALIBRATION.md`](CALIBRATION.md).

PC fan aeroacoustic simulation and shape optimization.

## What it does

Models a 120 mm PC fan (frame box, shroud, motor mount, support struts, blades) and runs an end-to-end mean-flow + harmonic acoustics solve. The optimizer drives blade-path, blade-section, and shroud-radius design variables to minimize a weighted objective combining acoustic level, airflow penalty, collision penalty, and broadband noise terms.

## Installation

```toml
[dependencies]
sonium-sim-pcfan = { workspace = true }
```

## Binary

```bash
cargo run -p sonium-sim-pcfan --bin pcfan-opt --release -- [options]

# Calibrated presets at 600 RPM: 20 CFM free-air or 15 CFM installed.
cargo run -p sonium-sim-pcfan --bin pcfan-opt --release -- --installation bench
cargo run -p sonium-sim-pcfan --bin pcfan-opt --release -- --installation chassis

# Deterministic native gallery smoke (ParaView script retained; no pvbatch).
cargo run -p sonium-sim-pcfan --bin pcfan-opt --release -- \
  --solve-only --gallery /tmp/pcfan-gallery --no-paraview \
  --cfd-cells-x 2 --cfd-cells-y 2 --cfd-cells-z 2 \
  --cfd-picard-iterations 2 --cfd-linear-iterations 40
```

Set `PVBATCH=/path/to/pvbatch` to opt into strict external ParaView capture;
the native gallery contract does not require a GUI/OpenGL installation.

Gallery manifest admission re-reads every registered PNG, verifies its PNG
chunks and decoded pixels, requires visible pixel variation, and enforces the
fixed 1920×1080 viewport before accepting the checksummed manifest. This is an
image/container-integrity gate; ParaView camera/layout semantics and external
product-image validation remain environment-gated.

The preset targets and empirical intervals scale with RPM. Use
`--target-airflow-cfm` for an explicit target, `--inflow-ti` and
`--inflow-length-scale-m` for measured inflow, and `--modal-loss-factor`
(0.03–0.20) for a calibrated chassis cavity.

## Modules

- `model` — `PcFanConfig`, design variables, mesh resolution, optimization weights
- `geometry` — frame box, shroud, hub, motor mount, support struts, blade panels, VTU export
- `cfd` — reduced-order and incompressible-CFD mean-flow solves on a structured FEM grid
- `evaluation` — couples flow + acoustics + objective + explicit derivative policy
- `optimizer` — L-BFGS warm-restart loop with airflow-band lock
- `meshlang_geometry` — alternative MeshLang DSL geometry (loaded via `meshlang-compiler`)

## Flow Models

| Model | When to use |
|-------|-------------|
| `ReducedOrder` (default) | Fast objective evaluation in optimizer hot loop |
| `IncompressibleCfd` | Validation and final design verification |

The CFD path uses an immersed-penalty representation for solid fan surfaces (no extra Dirichlet DOFs on blades/struts), an actuator-line forcing derived from blade geometry, and a Picard nonlinear iteration capped by `max_picard_iterations`.

Use `try_solve_pc_fan_cfd_mesh_convergence` for a checked isotropic refinement
sweep. It rejects invalid bounds, tolerances, or level budgets before solving;
the returned report records whether every participating level converged and
reported finite flow, residual, and conservation diagnostics.

Resolved-flow runners publish independent mesh, time-step, rotor-revolution,
and nonlinear-iteration ladders through
`evaluate_pc_fan_cfd_independence`. Promotion requires strict level ordering,
finite converged observations, and agreement of airflow, pressure drop, and
recovered axial loading on every required axis.

`evaluate_pc_fan_measured_correlation` compares provenance-labelled measured
and predicted RPM observations with explicit airflow, pressure, and optional
acoustic RMSE budgets. Synthetic fixtures exercise the same contract but do
not count as measured-product correlation.

The reduced-order thermo-structural path carries winding-temperature-driven
thermal stress and blade growth into the structural margin. Its expansion and
constraint parameters are checked at the objective admission boundary; this
is still a reduced-order screen, not product-scale structural validation.

The structural screen also exposes a reduced-order Basquin fatigue margin.
`fatigue_reference_stress_pa`, the reference/design cycle counts, and the
negative `fatigue_exponent` are validated before objective admission. The
alternating bending stress and design-life-scaled allowable are hard
feasibility constraints; this remains a deterministic screen, not an S-N
calibration or product-scale fatigue certification.

## Objective

Weighted sum of:

- Tonal acoustic level (per-harmonic FEM aeroacoustic source model, blade-passing frequency × harmonic count)
- Airflow penalty (deviation from `target_airflow_cfm`, with a band-fraction lock)
- Collision penalty (blade-blade, blade-shroud)
- Spectral spread / directivity spread / operating-point spread (Phase 1)
- Tip-leakage broadband noise (FD only)

current CPU optimizer supports centered finite-difference full-objective gradients and a checked reduced-order full-state adjoint over airflow plus harmonic acoustic pressures. The adjoint exposes non-zero harmonic pressure multipliers and matches the full-objective finite-difference gradient. Resolved CFD admits the checked laminar full-state objective subset: airflow, pressure rise, actuator loading, and topology-preserving rotating-wall loading/power. Full aeroacoustic derivatives, turbulence, topology-changing motion, and other unsupported objectives remain fail-closed. `PcFanDerivativePolicy::DerivativeFree` routes explicitly unsupported or nonsmooth objectives through L-SHADE and rejects shape-gradient requests. Broadband, spectral, and directivity spreads remain independently finite-difference-tested.

The reduced-order discrete-adjoint path now includes the harmonic acoustic
pressure state in its transposed solve and exposes non-zero pressure
multipliers. Its gradient matches the full-objective finite-difference
reference. Resolved-CFD requests outside the checked laminar objective subset remain
fail-closed.

## CFD backend & preconditioner

The current CPU optimizer exposes both centered finite-difference
full-objective gradients and the parity-tested reduced-order full-state
airflow/harmonic-pressure adjoint. Resolved CFD supports only the checked laminar
full-state objective subset documented above; other resolved-CFD gradient requests
remain explicitly unsupported.
Reduced-order candidates pass through explicit bounded
feasibility restoration before promotion.

The Picard-loop linear solve dispatches through `sonium_compute::Backend`. Pick at runtime:

```bash
# Auto-select (default): Metal on macOS+metal feature, CUDA on Linux+cuda, CPU otherwise.
cargo run -p sonium-sim-pcfan --bin pcfan-opt --release --features metal -- --cfd ...

# Explicit backend
--cfd-backend auto|cpu|cuda|metal       # default: auto
```

The GPU-friendly preconditioner choice trades preconditioner strength for parallelism:

```bash
--cfd-precond polynomial-jacobi|chebyshev   # default: polynomial-jacobi
--cfd-precond-degree N                       # polynomial Jacobi degree, default 1
--cfd-precond-damping W                      # polynomial Jacobi ω, default 1.0
--cfd-cheb-degree N                          # Chebyshev degree, default 5
--cfd-cheb-eigenvalue-ratio R                # Chebyshev λ_max/λ_min, default 30.0
```

**Defaults**: `polynomial-jacobi` with K=1, ω=1.0 (= point-Jacobi). Robust on the indefinite NS-Picard saddle-point system. Switch to `chebyshev` only for SPD-shaped sub-problems (e.g. acoustic post-processing or Helmholtz channel-mode tests, not the Stokes block itself).

**Recommendation by problem size** (from `backend_gmres_bench` on 2-D Poisson):
- Small (n < 10k DOFs) — `--cfd-backend cpu` (legacy ILU wins; backend dispatch overhead dominates)
- Medium (10k ≤ n < 50k) — `--cfd-backend auto` with the default polynomial-jacobi
- Large (n ≥ 50k) — `--cfd-backend auto --cfd-precond chebyshev --cfd-cheb-degree 5` if the matrix is SPD-shaped; otherwise `--cfd-precond polynomial-jacobi --cfd-precond-degree 2`

For the stock pcfan grid (≤10k DOFs), defaults are fine — Metal helps only when the grid is refined past `--cfd-cells-x 16 --cfd-cells-y 16 --cfd-cells-z 24`.

## Manufacturing robustness

`PcFanManufacturingConfig::tolerance` defines symmetric thickness, chord,
twist, and shroud manufacturing half-widths. The explicit
`evaluate_pc_fan_manufacturing_ensemble` API evaluates the nominal design and
all deterministic tolerance corners on the reduced-order model. Use
`PcFanManufacturingEnsemble::is_robust()` as a promotion gate. It rejects
resolved-flow configurations rather than silently treating a reduced-order
corner sweep as CFD evidence.

For an operating-envelope promotion check, use
`evaluate_pc_fan_manufacturing_operating_envelope`. It evaluates the same
corners at every configured `operating_point_spread.rpms`, checks finite and
feasible points plus monotonic airflow/pressure trends for each realization,
and requires both non-zero tolerance coverage and at least two RPM samples in
`is_robust()`.

`evaluate_pc_fan_fan_curve` also solves the reduced fan/system-pressure
intersection with a bounded bracketed solve and exposes it as
`PcFanFanCurve::operating_point`; curve validity requires that intersection to
be finite and converged. This prevents an installed operating point from being
silently inferred from a prescribed target flow.

## Robust operating-envelope optimization

`PcFanOperatingEnvelopeObjectiveConfig` and
`evaluate_pc_fan_operating_envelope_objective` score every requested RPM and
manufacturing realization against airflow and pressure targets, with optional
acoustic targets. `optimize_pc_fan_operating_envelope` uses deterministic
reduced-order L-SHADE and promotes only finite, robust candidates. It rejects
resolved-flow configurations and malformed target/weight contracts; this API
does not claim resolved-flow or measured-product evidence.

## Prediction-to-measured correlation

`predict_pc_fan_correlation_dataset` evaluates a reduced-order design at a
strictly increasing RPM set and returns provenance-labelled airflow,
pressure-rise, and acoustic observations for
`evaluate_pc_fan_measured_correlation`. Resolved-flow configurations and
malformed RPM topology fail closed before evaluation. Synthetic fixtures and
solver predictions are evidence of the pipeline contract only; they do not
constitute measured-product correlation.

## Testing

```bash
cargo test -p sonium-sim-pcfan --release
cargo check -p sonium-sim-pcfan && cargo clippy -p sonium-sim-pcfan
```

## Notes

- Frame mounts are fixed — design variables never move them.
- Shroud outer radius must stay inside the frame inner side.
- Blade root contact at the hub is allowed; interior overlap is rejected.
- The MeshLang DSL geometry path (in `meshlang_geometry`) is experimental.
## Fidelity promotion

`evaluate_pc_fan_fidelity_promotion` is a fail-closed admission contract for
promoting a reduced-order candidate set to a higher-fidelity evaluation. It
requires provenance, matching candidate topology, finite objectives and
observables, distinct fidelity labels, objective-error and Spearman-rank
budgets, feasibility parity, and agreement on the best candidate. The
`pcfan.fidelity-promotion-reference.v1` QA case uses a deterministic synthetic
fixture to compile-check this contract; it is not resolved-flow or measured
product evidence.

## Structural calibration contract

`structural_calibration` provides provenance-labelled, RPM-aligned comparison
for winding temperature, maximum stress, thermal expansion, and fatigue
alternating stress. Admission validates finite physical observations and RPM
topology, then reports per-observable absolute RMSE and maximum relative error
against configurable budgets. Prediction is intentionally restricted to the
reduced-order flow model; resolved-flow data is rejected until its full-state
adjoint and evidence path are complete.

The `pcfan.structural-calibration-reference.v1` QA target only compile-checks
this API with a deterministic synthetic fixture. It is not independent
measured-product evidence or product-scale structural/thermal validation.

## Independent evidence and resolved-flow parity

`measured_product` is the ingestion boundary for real product data. It requires
product/serial identity, instrument and calibration records, acquisition time,
raw-source URI, SHA-256 provenance, a fixed-unit CSV schema, finite physical
observations, and strictly increasing RPMs before adapting data into the
correlation or structural-calibration contracts.

`resolved_adjoint` solves the exact sparse state-transpose system supplied by a
resolved CFD assembler and contracts state/design derivatives. It does not
pretend that black-box finite differences are a full-state adjoint; the CFD
assembler must still publish its converged tangent and design residual
derivatives. `accelerator_parity` compares CPU-reference and explicit
CPU/CUDA/Metal backend observations, rejects host fallback by default, and
reports per-observable relative error. Its reference QA fixture is synthetic;
hardware parity evidence still requires backend execution.
## Checked resolved-CFD finite-difference gradients

Resolved-CFD finite-difference gradients now admit the base and every plus/minus perturbation only after checked convergence, conservation, loading, field-shape, and objective admission. This is a checked finite-difference fallback, not an exact resolved-flow discrete adjoint. Focused policy tests pass; topology-changing mesh motion, turbulence fidelity, full aeroacoustic derivatives, accelerator runtime parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.
## Topology-preserving ALE mesh motion

Resolved steady, rotating-wall transient, and checked finite-difference paths now apply a compactly supported radial ALE deformation for hub, tip, and shroud-radius design offsets. The update is transactional: connectivity is preserved, outer inlet/outlet/side boundaries remain fixed, and non-finite, inverted, or collapsed elements fail closed through an element-measure quality barrier. This is topology-preserving immersed-mesh motion, not a body-fitted remesher or an exact mesh-motion shape adjoint. Focused motion, steady, transient, and derivative tests pass; turbulence fidelity, full aeroacoustic derivatives, accelerator runtime parity, measured-product validation, and production-scale evidence remain open. No QA binary or full simulation was run.
## Verification update — algebraic turbulence state tangent (2026-08-25)

The algebraic k-omega surrogate now publishes a checked Newton state tangent when its global velocity peak is unique and non-zero, including the viscosity clamp derivative and stabilization-viscosity contribution. Resolved state-only adjoints may consume this provenance-labelled tangent; turbulence design, mesh-motion, transported k-omega/RANS, and full aeroacoustic derivatives remain fail-closed. Focused FEM and PC-fan tests pass. No QA binary or full simulation was run.
