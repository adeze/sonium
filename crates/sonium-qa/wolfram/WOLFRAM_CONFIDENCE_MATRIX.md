# Wolfram-derived confidence matrix

Audited 2026-08-08 against the public Wolfram documentation and the Sonium QA
ports. `WOLFRAMSCRIPT` remains an optional live cross-check; a row is called
enforced only when it also has a hermetic analytic, manufactured, or checked-in
numeric reference.

## Enforced coverage

| Wolfram source family | Sonium confidence coverage |
|---|---|
| [`NDEigensystem`](https://reference.wolfram.com/language/ref/NDEigensystem.html) | Every distinct family listed in `NDEIGENSYSTEM_PORTS.md`; option contracts in `wolfram_ndeigensystem_option_contracts.rs`; interval searches have inertia-certified FEAST completeness |
| Dense `LinearSolve`, `LeastSquares`, `QRDecomposition`, `SingularValueDecomposition` examples | LU/Cholesky/LDLT, reorthogonalized QR, two-level TSQR, pivoted rank detection, least squares, and thin Jacobi SVD unit references in `sonium-solvers-direct` |
| Global `NMinimize` method examples | CMA-ES plus deterministic-seed Differential Evolution, Simulated Annealing, and multistart Random Search references in `sonium-solvers-optimization` |
| [Heat-transfer verification](https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransferVerificationTests.html?view=all) | Sinusoidal plate, transient slab, convection, convection-radiation, axisymmetric hollow cylinder, segmented flux, radial multilayer sphere, literal non-axisymmetric 3D multilayer sphere, and locally refined MEMS-anemometer tests |
| [Fluid-dynamics verification](https://reference.wolfram.com/language/PDEModels/tutorial/FluidDynamics/FluidDynamicsVerificationTests.html?view=all) | Pipe/Poiseuille and cylinder-flow checks, plus catalytic converter and transport/reactor cases |
| [Acoustics frequency domain](https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/AcousticsFrequencyDomain.html?view=all) | Room modes, horn, diffraction, mirror, PML, porous/two-media propagation, radiation, and time-domain boundary variants |
| [Room eigenfrequencies](https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/ModelCollection/RoomEigenfrequencies.html?view=all) | Empty and furnished modal tests, including mesh/geometry effects |
| Structural mechanics and plasticity | Plane-stress patch, plate with hole, loaded/gravity/thermal cantilevers, beam-spring-mass, hyperelastic cube, plastic loading path, and P2 2D/3D static/modal references with true residual and refinement gates |
| [Joule heating](https://reference.wolfram.com/language/PDEModels/tutorial/Multiphysics/ModelCollection/JouleHeating.html?view=all) | Coupled electrothermal bar reference and broader multiphysics suite |
| Electromagnetics | Spherical capacitor with public solved-field energy/charge/capacitance observables, permanent magnet, and quasistatic straight-wire references |
| [Schrodinger-Poisson quantum well](https://reference.wolfram.com/language/PDEModels/tutorial/Physics/ModelCollection/SchrodingerPoisson1DQuantumWell.html?view=all) | Self-consistent 1D quantum-well reference plus scalar/radial Schrödinger eigensystems |
| Boundary element acoustics | Rigid/soft/impedance spheres using the public positive-admittance Robin convention, normalized public RCS, speaker piston, mixed-port duct power, rectangular-room Neumann Green function, and multiple assembly/FMM parity references |
| Implicit/image geometry | Pixel-exact masks remain available; fitted level-set and bilinear-mask contours use Newton projection, conforming local refinement, and orientation-preserving line search |

The exact test inventory is the set of `crates/sonium-qa/tests/wolfram_*.rs`;
the corresponding live extractors are in `crates/sonium-qa/wolfram/*.wls`.

## Closed audit findings

| Former priority | Implemented acceptance evidence |
|---|---|
| P0 mixed-port BEM | Refined inlet/outlet pressure errors 8.1%/3.6%, power errors 8.1%/7.1%, and 1.05% power imbalance below cutoff |
| P0 rectangular room | Refined BEM receiver pressure is within 4.33% of the independent Neumann modal sum |
| P0 axisymmetric resonator | Refined peak 182.396 Hz versus the published 180.158 Hz |
| P0 anemometer | Locally isotropic heater mesh gives 119.5814 K and 0.07420 K versus 119.519 K and 0.0678515 K (0.052% and 9.36%) |
| P1 public RCS | Public postprocessor and receiver definition differ by 0.385% after restoring the free-space `1/(4 pi)` normalization |
| P1 structural order | P2 2D/3D eigenmodes and bending loads pass published tolerances, improve over P1, conserve applied resultants, and report independently measured residuals |
| P2 fitted geometry | `fitted_implicit_region_mesh_triangles` and `fitted_binary_mask_mesh_triangles` reduce staircase error and refine boundary-adjacent cells |
| P2 dense solvers | `DenseDirectMethod::Ldlt` and `Tsqr` implement factorization, solve, auto-dispatch, rank, singularity, and dimension-error contracts |
| P2 hermetic references | Provenance-tagged goldens now cover structural eigensystems, localized/flux/parallel-plate/surrounding transport, catalytic conversion, multilayer-sphere averages, and 31 multiphysics cases; other enforced tests use analytic, manufactured, or published constants |
| P1 impedance-sphere convention | The public helper now implements `dp/dr - i k beta p = 0`; dense/exact and H-matrix/exact relative errors are both 0.0591%, with 9.84e-11 dense/H-matrix disagreement |
| P1 conforming curved horn | Exact disk/floor/wall projection, sliver removal, and compacted topology satisfy geometry/area checks. At the mesh-sensitive 800 Hz dip, the 120/180-grid first-order extrapolation is 0.03976 versus the published Wolfram raster's 0.0410; the refined 950 Hz value is 0.01720 versus 0.0218 |
| P1 literal 3D multilayer sphere | A conforming marked shell mesh and quadrature-point Neumann adapter drive the angular Wolfram flux directly; coarse/fine layer averages converge, angular spread is 50.4 K, flux error is 1.88%, and energy/free-residual errors are below 4e-13 |
| P2 public electrostatic observables | `ElectrostaticSolvedField` exposes checked electric-field, energy, capacitance, and charge recovery; the spherical-capacitor test uses this API and the unit test exactly recovers a linear field and quadratic energy |
| P2 command-line hyperelastic extractor | The extractor substitutes the fixed published displacement before FEM discretization and configures damped Newton; current command-line Wolfram Engine emits finite displacements at all three published sample points |

## Remaining production backlog found during the wider Wolfram audit

The audit's five production gaps are now implemented. No additional missing
capability remains in the audited Wolfram solver, FEM, optimization, and
`NDEigensystem` example families. Future Wolfram documentation additions should
be assessed with the promotion rule below rather than silently counted as
covered.

## Promotion rule

A diagnostic can move to enforced coverage only when it asserts convergence
toward an independent reference, true residual/backward error, finite output,
and a mesh/order refinement trend. Tests that assert an error remains large
are useful regression diagnostics but are not confidence gates.
