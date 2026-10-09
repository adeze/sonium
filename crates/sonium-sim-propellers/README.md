# sonium-sim-propellers

The literature basis, matched-thrust caveats, empirical attenuation ranges,
and proposed measurement study are documented in
[`CALIBRATION.md`](CALIBRATION.md).

Propeller aeroacoustic simulation: shrouded, toroidal, and serrated designs.

Application-level shape and noise studies that sit on top of `sonium-fem`'s aeroacoustic source model. The crate keeps these design studies out of the FEM core.

## Installation

```toml
[dependencies]
sonium-sim-propellers = { workspace = true }
```

## Binary

```bash
cargo run -p sonium-sim-propellers --bin propeller-3d-opt --release

# Explicit trade study: the default thrust-loss ceiling is 5%.
cargo run -p sonium-sim-propellers --bin propeller-3d-opt --release -- \
  --rpm 18000 --max-thrust-loss 0.05 \
  --serration-amplitude 0.018 --serration-wavelength 0.055 --serration-span 0.72
```

## What it covers

- **Axisymmetric source model** (`source_model`) — shroud propagation through a meridional duct profile, with explicit modifiers for toroidal loops and leading/trailing-edge serrations. Loops reduce open-tip vortex shedding; LE serrations pre-condition incoming turbulence; TE serrations break coherent wake eddies.
- **3D blade representation** (`blade3d`) — panel-level blade geometry with per-panel acoustic summary, comparison helpers, and ParaView VTU export.
- **3D shape optimization** (`optimization3d`) — `BladeShapeVariables` (chord, twist, camber, sweep, dihedral, …), evaluation function, and an optimisation driver.

## Modules

- `source_model` — `PropellerDesign`, `PropellerSpectrum`, `PropellerPlanform` (`Conventional`, `ToroidalLoop`), `EdgeTreatment`, `SerrationGeometry`, `EvaluationMode` (`FastSurrogate` / `ResearchControl`), `HarmonicNoise`
- `blade3d` — `Blade3dConfig`, `BladeMesh`, `BladePanel`, `PanelAcousticSummary`, `compare_panel_acoustics`, `Point3`
- `optimization3d` — `Propeller3dOptConfig`, `BladeShapeVariables`, `OptimizedBlade`, `evaluate_blade_shape`, `optimize_blade_shape`

## Evaluation Modes

| Mode | Use case |
|------|----------|
| `FastSurrogate` | Fast deterministic surrogate, optimization hot loop |
| `ResearchControl` | Higher-order quadrature + coherence control |

## Edge Treatment

`EdgeTreatment` is `Smooth | LeadingEdgeSerrations | TrailingEdgeSerrations | LeadingAndTrailingSerrations`. Serration effectiveness is parameterised by amplitude/wavelength as fractions of local chord and the span fraction treated. The default `owl_like_micro_airfoil` geometry uses 1.8 % chord amplitude, 5.5 % chord wavelength, 72 % span.

## Output

ParaView (VTU) is the primary output for both volume and surface quantities, consistent with the rest of the workspace.

## Testing

```bash
cargo test -p sonium-sim-propellers --release
cargo check -p sonium-sim-propellers && cargo clippy -p sonium-sim-propellers
```

## Notes

- The axisymmetric source model cannot geometrically represent a toroidal loop. Loops are modelled as a *source-generation modifier* before axisymmetric propagation.
- The 3D path uses `rayon` for per-panel parallelism on acoustic summaries.

## Deterministic gallery

Regenerate the complete, headless propeller gallery from a clean checkout:

```bash
cargo run -p sonium-sim-propellers --bin propeller-3d-opt -- --gallery target/propeller-gallery
```

This writes baseline and optimized VTU sources, deterministic source CSVs for
the harmonic spectrum and spanwise loading, fixed-camera PNG field views, SVG
response/comparison curves, a twelve-frame `rotation.gif`, and `gallery.json`.
The manifest records rendering conventions and checksums for each rendered or
source asset and its source artifact. The exportability gate runs before any
gallery artifact is written. Rendering is transactional: a failed rerun
leaves the previous gallery untouched, while a successful rerun removes stale
files. The bundle also contains `execution_evidence.json`, which records
compiler/host/target facts, selected CPU dispatch, f64 precision, surrogate
provenance, and SHA-256 checksums for every source and rendered artifact.

## Backend capability contract

`PropellerCapabilityMatrix::current` is the authoritative application matrix
for backend, workload, precision, mesh order, direction, and
single/multipoint/transient execution. CPU reference paths are f64. With the
`cuda` feature, the matrix admits `BladeElementBem`, f64, mesh order 1,
forward, single or multipoint requests through the bounded native adapter.
Other CUDA matrix cells and all Metal cells remain typed unsupported results
without implicit fallback. `src/cuda_adapter.rs` implements c64 regular-panel
BEM work, resident acoustic sweeps and a VLM influence action; the latter is
a separate API, not blanket promotion of VLM application matrix cells.
Panel/source construction remains host work, and panel-integrated acoustic
proxies are not resolved observer, wake or end-to-end CFD solutions. Optimized
results carry the same requested/actual backend, precision, fidelity,
convergence, failure flags, host facts, and input/output checksums in
`PropellerResultProvenance`.

The PP2 capability evidence recipe is
`just test-propellers-cuda-capability`. It validates all 648 matrix cells,
records the local NVIDIA inventory when available, and writes
`target/sonium-unix-pp2/cuda-metal-capability-v1.json`. Hardware presence is
reported as inventory only. The supported-cell count depends on compiled
features; do not reuse the older all-432-device-cells-unsupported count for a
CUDA build. Capability enumeration alone is not runtime evidence: native
adapter results must carry actual device counters and CPU/device parity for
the exact workload. This documentation correction adds no new runtime result.

## PP3 fidelity evidence

`just test-propellers-fidelity-native-x86` emits
`target/sonium-unix-pp3/native-x86-pp3-fidelity-v1.json`. The native Linux/x86
producer exercises analytic source families, direct/BEM/VLM/free-wake,
body-fitted/weighted flow, compact observer/installation, zero-load, and
geometry/response refinement gates with named observables, declared budgets,
finite/converged predicates, and typed rejection evidence. Analytic,
caller/synthetic, and reduced-physics rows remain separate claim classes.
No authenticated measured dataset, resolved CFD, or full moving-surface FW-H
result is emitted or inferred; those fields are explicit `unavailable` values.
Windows and Arm64 comparison runs are waived for this host.

## PP4 release evidence

`just test-propellers-release-native-x86` regenerates the CLI gallery in a
clean `target/sonium-unix-pp4/cli-gallery` directory, then validates a
ledger-bound library gallery and writes
`target/sonium-unix-pp4/native-x86-pp4-release-v1.json`. The gate checks the
exact ten-figure bundle, 1920×1080 nonblank PNGs, fixed-size structured SVGs,
twelve-frame GIF, finite CSV rows, VTU validity, manifest/source checksums,
CSV-to-result ledgers, deterministic replacement, corruption rejection, typed
invalid/unwritable output, invalid CLI, and zero-partial-output failures.
The release/QA join is recorded as blocked until PP1–PP3 and the
`sonium-qa`-owned manifest fan-in are complete; no skipped QA case is marked
passed. The native PP4 lane also exercises the real CLI in a child process:
the deterministic pre-publication interruption and forced renderer-unavailable
probes require typed nonzero outcomes, sentinel preservation, no published
partial bundle, and a successful retry. Fault scratch data stays under the
repository-local `.target-unix-pp4-faults` target root.

## Measured harmonic correlation contract

`correlate_measured_harmonics` compares externally supplied peak-pressure harmonics with `PropellerSpectrum` by harmonic number and frequency tolerance. It rejects malformed or duplicate points, reports signed/absolute errors, RMSE, relative RMSE, bias, and explicit admission limits, and supports deliberate partial matching. The API supplies a checked data contract only; this repository does not claim a measured product dataset, calibration, observer phase, propagation, wake, CFD/FW-H, or product acceptance from its synthetic tests.

## Prescribed transient free-wake reference

`advance_prescribed_free_wake` advances a converged fixed-circulation steady wake with explicit Euler steps. Each step recomputes regularized finite-segment Biot--Savart induction, applies the declared uniform background velocity, tracks node age/core growth, and records bounded node/segment work and displacement diagnostics. Zero circulation is valid and all malformed, non-finite, non-converged, over-budget, or unstable states fail closed. This is prescribed-circulation transient kinematics only: it does not model shedding or load-coupled circulation, viscous core dynamics, aeroelasticity, body-fitted duct/shroud flow, CFD, FW-H, or measured correlation.

## Lagged load-coupled transient free-wake reference

`evaluate_transient_load_coupled_free_wake` adds bounded explicit time-step coupling around the checked BEM and steady wake references. Each step samples the current wake at BEM stations, relaxes external inflow, re-evaluates BEM, updates station/tip-seed circulation, inserts one anchored shedding node per blade/seed, and convects the downstream wake. The result retains per-step BEM, inflow, circulation-change, age, displacement, and Biot--Savart work evidence with total BEM/wake budgets. This is lagged quasi-unsteady shedding kinematics only, not dynamic stall, viscous wake physics, aeroelasticity, body-fitted duct/shroud flow, CFD, FW-H, or measured correlation.

## Centrifugal-stiffened modal reference

`evaluate_centrifugal_stiffened_modal_reference` estimates rectangular-section modal mass, elastic bending stiffness, centrifugal geometric stiffness, root centrifugal load, and blade-passing-frequency separation using deterministic spanwise quadrature. It provides a Campbell-style structural screen and RPM-squared scaling evidence, not a manufactured-section finite-element, nonlinear, flutter, aeroelastic time-domain, or measured structural solver.

## Multi-rotor interaction screen

`evaluate_multi_rotor_reference` is a bounded one-way actuator-disk screen for prescribed positive-thrust rotors. It uses momentum-theory isolated induction and finite receiving-disk quadrature over downstream Gaussian wakes, retaining signed pairwise induction and strict work budgets. It is screening evidence only; it does not replace load-coupled, body-fitted, viscous, CFD/FW-H, or measured multi-rotor validation.

`evaluate_multi_rotor_bem` adds a steady load-coupled slice for caller-supplied checked BEM models: pairwise wake induction is injected as external station inflow and BEM loads are iterated to a declared residual. It remains a bounded forward-flight reference, not an unsteady, viscous, aeroelastic, body-fitted, CFD/FW-H, or measured solver.

`evaluate_nonlinear_aeroelastic_modal` provides a bounded single-mode Duffing harmonic-balance screen with deterministic amplitude bracketing, nonlinear/linear response comparison, and stress/deflection gates. It is not a transient or multi-mode aeroelastic solver.

`project_bem_loads_to_panel_mesh` maps checked BEM strip loads onto generated 3-D conventional or toroidal panels and preserves the all-blade thrust/torque ledger. It is a geometry-aware projection, not a resolved panel-flow or toroidal wake solver.

`evaluate_harmonic_signal_reference` independently synthesizes integer harmonics and checks analytical/sample RMS, band power, SPL, phase, and closure. It is a signal-contract reference, not measured or propagated acoustics.

## Analytic observer reference

`RotatingDipoleReference` provides a bounded far-field magnitude check for one
harmonic: it applies the compact dipole envelope `|sin(theta)|` to an
equatorial source magnitude and verifies the exact `2/3` sphere-power factor.
It intentionally contains no observer phase, wake, installation, or resolved
directivity model. The release QA case is
`propeller.rotating-dipole-reference.v1`.

## Multipoint operating conditions

`PropellerOperatingEnvelope::evaluate_conditions` evaluates a deterministic
list of off-design conditions using the explicitly bounded constant-load
surrogate: thrust and torque scale with `density * RPM²`, temperature supplies
ideal-gas speed of sound, and the disk solve uses the axial component of the
inflow frame. `Propeller3dOptConfig::operating_conditions` applies the same
admission contract to every optimized candidate; invalid source points and
candidate power failures are rejected before feasibility or gallery export.
This is robust screening evidence, not a blade-element, free-wake, CFD, or
resolved observer-directivity model. The release case is
`propeller.multipoint-operating-envelope.v1`.

`estimate_shape_gradient` and `taylor_test_shape_derivative` expose a checked
central finite-difference derivative contract for the bounded surrogate. They
reject non-finite steps, bound-clamped probes, and invalid Taylor schedules;
the report records first-order remainders and observed order. This is a
shape-search verification tool, not a discrete adjoint or resolved-flow
gradient.

`evaluate_mesh_convergence` provides a checked geometric panel-refinement
ladder for conventional and toroidal meshes. It records panel counts, total
area, tip-area fraction, adjacent relative area changes, and final admission
status. These are geometry-resolution observables only; they are not
aerodynamic, wake, acoustic, or observer-directivity convergence.

Every `OptimizedBlade` also carries `PropellerResultProvenance`. Current
production results are labeled `PanelGeometrySurrogate` with unresolved-wake,
unresolved-directivity, unconverged, and unmeasured flags; export admission
rejects injected stronger claims.
## S-N/Miner fatigue reference

`evaluate_s_n_fatigue_reference` evaluates caller-supplied mean/alternating stress blocks with a Goodman correction, single-slope Basquin S-N curve, endurance-limit handling, and Miner damage accumulation. It is a bounded fatigue admission contract only: it does not generate aeroelastic loads or claim crack-growth, manufactured-section, measured, CFD/FW-H, or full-QA fatigue validation.
## Viscous-core evolution reference

`evaluate_viscous_core_evolution` applies the analytic Lamb--Oseen relation `a²(t) = a₀² + 4νt` to prescribed signed circulation and reports finite-radius tangential velocity, age, core growth, and explicit step/radius budgets. This is a bounded viscous-core coupling reference only; it is not a resolved viscous wake, turbulence, dynamic-stall, CFD, FW-H, or measured solver.

## Dynamic-stall reference

`evaluate_dynamic_stall_reference` evaluates a prescribed angle-of-attack history with attached lift, smooth separation onset, delayed separation/recovery, and first-order lift-state lag. The checked history, step, sample-count, and finite-state budgets make this a deterministic unsteady blade-element coupling contract. It is a bounded phenomenological reference—not a Beddoes--Leishman model, CFD/URANS solver, or measured dynamic-stall validation.


## Quasi-1D duct/shroud reference

`evaluate_duct_shroud_reference` evaluates a prescribed geometry-derived axial station ladder with constant-density continuity, Darcy wall loss, minor-transition loss, Reynolds/Mach diagnostics, and explicit velocity, Mach, pressure, station-count, and finite-state budgets. This is bounded quasi-one-dimensional duct/shroud coupling evidence; it is not a three-dimensional body-fitted CFD solver, a compressible-flow solver, an acoustic mode-matching model, or measured validation.


## Compact FW-H-style observer reference

`evaluate_fwh_reference` coherently integrates prescribed stationary surface pressure and normal-velocity peak phasors into free-field monopole and dipole observer pressure, using explicit `exp(iωt)`/`exp(-ikr)` conventions and distance/element budgets. This is a bounded source-observer contract only; it is not moving-surface retarded-time FW-H, CFD, mean-flow correction, acoustic mode matching, or measured validation. Gate every compact-observer call with `evaluate_compressibility_reference`: supersonic tips fail closed, and non-compact sources (`kL > 1`) must set `require_compact` before compact propagation may be claimed.
## Checked 3-D vortex-lattice reference

`evaluate_vortex_lattice_reference` is the checked panel-flow lane for generated `BladePanel` geometry. It assembles quarter-chord bound vortices with finite discretized trailing wakes, includes rigid-body rotational velocity, solves the dense collocation system with pivot admission, and reports per-panel circulation, Kutta--Joukowski force/moment, thrust/torque, induced velocity, and normal residuals. Panel, wake, velocity, and finite-result budgets fail closed. The registered contract is `propeller.vortex-lattice-reference.v1`.

This is steady incompressible VLM evidence only. It is not viscous flow, nonlinear free-wake roll-up, RANS/URANS/LES, resolved duct/shroud flow, moving-surface FW-H, measured calibration, or product-level QA. The evidence is covered by focused library/integration tests and a compile-only QA wrapper; no QA binary or full simulation is required for this reference.
## Caller-supplied measured calibration reference

`calibrate_measured_response` fits a bounded affine correction, `measured ≈ gain × predicted + offset`, over caller-supplied operating-point samples. It validates finite inputs, sample/work limits, predictor conditioning, gain bounds, and residual gates, and returns per-sample calibrated values, RMSE, maximum error, and optional R². This is a calibration-contract and provenance hook only: the repository ships no measured dataset and does not authenticate instruments, calibration certificates, or product acceptance.
## Low/high-fidelity promotion reference

`evaluate_fidelity_promotion` compares caller-supplied low- and high-fidelity objective values with strict tie rejection, deterministic one-based ranks, pairwise ordering agreement, Kendall-style rank correlation, and explicit promotion thresholds. It is a ranking gate only: it does not execute either fidelity, estimate a physical error field, or establish CFD/measured provenance.

## Versioned measured-dataset provenance envelope

`MeasuredCalibrationDataset`, `MeasuredCalibrationDatasetMetadata`, and `MEASURED_CALIBRATION_DATASET_SCHEMA_V1` provide a versioned JSON envelope for caller-supplied scalar measured-calibration samples. The envelope binds dataset identity, source URI, license, instrument ID, calibration reference, channel/unit, sample values, and a canonical lowercase SHA-256 digest over the samples and metadata (excluding the digest). `from_parts`, `from_json`, `validate`, `to_json`, and `calibrate` enforce metadata, finite-sample, duplicate-operating-point, digest-integrity, and bounded affine-calibration checks.

This is provenance and content-integrity plumbing only. It does not authenticate an instrument or certificate, establish that a public source or measurement is truthful, provide uncertainty propagation, or constitute external correlation or product acceptance. Coverage is through focused tests and a compile-only QA wrapper; no QA binary or full simulation is required.

## Retarded-time moving-surface FW-H reference

`evaluate_moving_fwh_reference` consumes bounded, strictly ordered histories for moving surface elements and solves the observer-time retarded equation for each element. It interpolates position, normal, area, pressure, and normal velocity; checks source Mach and `1 - M_r`; and evaluates reduced far-field thickness/loading terms with motion-dependent path and Doppler factors. Results retain per-element retarded time, distance, Doppler factor, and pressure contributions.

This is moving-surface retarded-time acoustic evidence under a quiescent homogeneous-medium, far-field, piecewise-linear source-history model. It is not a full mean-flow/Farassat implementation, CFD/URANS/LES coupling, or measured FW-H validation. Coverage is through focused tests and a compile-only QA wrapper; no QA binary or full simulation is required.
## Uniform-mean-flow moving FW-H reference

`evaluate_mean_flow_fwh_reference` reuses the moving-surface history contract with a bounded, subsonic uniform mean flow. It solves the convected retarded-time identity, reports the convected acoustic ray and relative-source Doppler Jacobian, and evaluates reduced far-field thickness/loading terms. This is constant-flow reference evidence only—not full Farassat FW-H, CFD/URANS/LES, turbulence, acoustic mode matching, measured correlation, or runtime QA. The focused integration target and compile-only QA wrapper are covered by tests/checks; no QA binary or full simulation is required.

## Bounded measured-uncertainty reference

`evaluate_measured_uncertainty` wraps the caller-supplied affine measured-response calibration with independent first-order Gaussian propagation for model prediction uncertainty, measurement uncertainty, and calibration RMSE. It reports per-point combined standard deviations and z-scores, coverage, reduced chi-squared, and explicit admission gates. This is a bounded uncertainty contract only: correlated-error models, Bayesian posteriors, instrument traceability, authenticated public datasets, external correlation, and product acceptance remain out of scope. Seven focused integration tests and a feature-enabled compile-only QA wrapper cover the contract; no QA binary or full simulation was run.
## Mesh-state transfer reference (2026-08-25)

`transfer_panel_scalar_state` provides a bounded remesh/state-transfer seam for scalar panel fields on compatible generated 3-D blade meshes. It validates topology, blade identity, finite geometry/state, panel counts, nearest-neighbor distance, and distance-work budgets; it reports source/target area-weighted means and transfer-distance diagnostics. The interpolation is deterministic same-blade inverse-distance geometry, not conservative finite-volume transfer, CFD, wake transport, or aeroelastic state transfer. Focused tests pass and the QA target is compile-only; no QA binary or full simulation was run.
## Bounded Farassat 1A reference (2026-08-25)

`evaluate_farassat_1a_reference` evaluates the time-domain Farassat 1A thickness and loading surface integrals over caller-supplied piecewise-linear moving-surface histories. It solves each retarded-time equation, differentiates source strength and loading/directivity terms on the local source-time segment, applies the moving-source Doppler factors, and retains per-element diagnostics. Nine focused analytical/admission tests pass; the QA target compiles and is registered, but its binary is not run. This is a complete quiescent homogeneous-medium surface-integral reference under its stated assumptions—not CFD coupling, turbulence, or a mean-flow/Farassat extension.
## Body-fitted meridional flow reference

`solve_body_fitted_duct_flow` builds a checked annular `(r,z)` body-fitted mesh from caller-supplied duct/shroud stations and solves the production FEM laminar incompressible Navier–Stokes Picard operator. It returns nodal velocity/pressure and convergence, mesh, inlet/outlet, and finite-state diagnostics. The contract is deliberately limited to meridional laminar flow: it does not claim axisymmetric weighting, rotating walls, turbulence/RANS/LES, wake transport, FW-H coupling, measured validation, or product/runtime QA. Focused tests and compile-only QA coverage are provided; no QA binary or full simulation is required.
## Authenticated measured-dataset attestation (2026-08-25)

`MeasuredDatasetAttestation` signs the canonical `MeasuredCalibrationDataset::to_json()` representation with a domain-separated Ed25519 payload containing the caller-supplied key ID. `from_signed_parts`, `from_json`, `verify`, and `signing_payload` enforce the v1 schema, canonical nested dataset, lowercase hexadecimal key/signature encoding, and signature validity. `MeasuredDatasetTrustPolicy` adds a host-owned allow-list that pins accepted key IDs to public keys; `verify_with_policy` checks that pin before cryptographic verification. The module performs no key discovery, certificate-chain lookup, network access, or claim that the instrument or public source is physically truthful.

Eight focused integration tests pass. The feature-enabled QA wrapper, manifest, and inventory are compile/validation checked only; no QA binary or full simulation was run. This closes byte-level authenticated provenance under a caller-owned pinned policy, not certificate-chain or instrument authentication, public-source truth, independent measured correlation, or product acceptance.
## Axisymmetric weighted-flow and rotating-wall reference (2026-08-25)

`evaluate_axisymmetric_weighted_flow` applies the physical `2πr` meridional-section measure to a checked `BodyFittedFlowResult`, reporting area, volume flow, mass flow, volume-flow-weighted axial velocity, kinetic-energy flux, and inlet/outlet closure. `evaluate_axisymmetric_rotating_wall` provides the exact annular Couette solution at caller-supplied axial stations, including wall speeds, torque per axial length, and viscous-dissipation identity. Nine focused integration tests pass; the feature-enabled QA wrapper, manifest, and inventory are compile/validation checked only. This is a bounded weighted-observable and rotating-wall reference seam—not a fully coupled axisymmetric Navier–Stokes or 3-D rotating-wall CFD solver, turbulence/RANS/LES model, wake/FW-H coupling, measured validation, or product/runtime QA. No QA binary or full simulation was run.
## Gallery export provenance validation (2026-08-25)

`validate_gallery_bundle` now checks the exact ten expected manifest figures and required VTU/manifest artifacts, then validates CSV headers, finite rows, counts, and numerical correspondence to the admitted candidate/baseline spectra and optimized mesh panel ledger. `write_gallery_bundle` runs this check after rendering, and the gallery test suite covers both spectrum and panel-load provenance drift. Focused gallery tests pass and the feature-enabled QA target is compile-checked only. This does not establish browser/perceptual regression, physical CFD or measured validation, or runtime/product QA; no QA binary or full simulation was run.

## Panel-response refinement reference (2026-08-25)

`evaluate_panel_refinement` validates a caller-supplied ordered ladder of panel counts with thrust, torque, and acoustic-pressure observations, reports normalized adjacent changes, and applies a finest-level convergence gate. Four focused tests and a compile-only QA wrapper cover finite, deterministic, non-converged, ordering, non-finite, and budgeted cases. This is a response-ledger contract only; it does not execute panel flow, resolve wake/directivity, establish CFD or measured convergence, or provide product/runtime QA. No QA binary or full simulation was run.
