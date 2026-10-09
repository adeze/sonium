# Physics support matrix

This file summarizes the physics exported by `sonium_fem::physics`. Dimension
columns describe the spatial formulation supplied by the physics module, not
merely whether a closed-form reference case exists.

## Legend

- **✓** — dedicated implementation with tests, examples, or QA coverage in that
  dimension.
- **◐** — the weak form is dimension-generic, or reusable building blocks exist,
  but there is no equally complete dimension-specific workflow or QA matrix yet.
- **R** — reduced-order, analytical, lumped, or reference model only; not a
  general spatial FEM formulation in that dimension.
- **0D** — network/lumped model with no spatial mesh.
- **—** — no explicit implementation is currently exposed.
- **Surface** — a 2-D manifold formulation that may be embedded in 3-D.

Axisymmetric support means a cylindrical meridional `(r, z)` formulation with
the required radial weighting and hoop terms. A planar 2-D form is not counted
as axisymmetric merely because it can be assembled on a meridional-looking mesh.

## Supported physics

| Physics | Rust module | 1-D | 2-D | Axi | 3-D | Formulation and properties |
|---|---|:---:|:---:|:---:|:---:|---|
| Pressure acoustics | [`acoustics`](src/physics/acoustics/) | R | ✓ | ✓ | ✓ | Frequency-domain Helmholtz and time-domain wave equations; complex wavenumber, viscothermal/equivalent-fluid loss, impedance and porous boundaries, ports, monopole/normal-motion sources, PML, WaveHoltz, Schwarz-PML, intensity/SPL/TL post-processing, and room acoustics. |
| Aeroacoustics | [`aeroacoustics`](src/physics/aeroacoustics/) | ✓ | ✓ | ✓ | ✓ | Convected potential flow, linearized Euler and linearized Navier–Stokes; nonuniform mean flow, liners and SUPG/GLS stabilization; compact Lighthill quadrupoles, stationary permeable FW-H, and 1-D Tam–Webb DRP Euler with phase/group velocity. |
| Axisymmetric common forms | [`axisymmetric`](src/physics/axisymmetric.rs) | — | — | ✓ | — | Reusable `2πr`-weighted scalar/vector mass and gradient forms, pressure Helmholtz, axisymmetric elasticity with hoop strain, shell hoop stiffness, and pressure/normal coupling. |
| Battery electrochemistry | [`battery`](src/physics/battery/) | R | ✓ | — | ✓ | DFN/P2D-style macro diffusion/capacity operators on P1 tetrahedra, spherical active-particle diffusion, interfacial reaction Jacobian, regional materials, and electrochemical heat-source breakdown. |
| Beams | [`beam`](src/physics/beam/) | ✓ | — | — | — | Two-node 2-D Euler–Bernoulli/corotational beam element, section properties, global beam model, and H-section shape optimization. |
| Biharmonic problems | [`biharmonic`](src/physics/biharmonic.rs) | — | ✓ | — | ✓ | C0 interior-penalty forms for higher-order elements plus a mixed `q = -Δu` P1 triangle/tetrahedron formulation for Navier biharmonic problems. |
| Laminated composites | [`composites`](src/physics/composites/) | — | ✓ | — | Surface | Classical laminate membrane and bending forms, orthotropic plies and strengths, laminate stiffness, and through-thickness layups. |
| Compressible flow | [`compressible_flow`](src/physics/compressible_flow/) | R | ✓ | — | ✓ | P1 DG Euler/Navier–Stokes on triangles/tetrahedra with Rusanov and SIPG fluxes, SSPRK3, characteristic limiting, Zhang–Shu positivity, Sutherland transport, and characteristic/wall boundaries; linearized building blocks remain available. |
| Conjugate heat transfer | [`conjugate_heat_transfer`](src/physics/conjugate_heat_transfer/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron marker-routed solid conduction and fluid advection/capacity, spatially varying advection fields, interface heat balance, and fluid/solid material regions. |
| Contact mechanics | [`contact`](src/physics/contact/) | — | ✓ | — | ✓ | Surface-triangle/AABB detection, closest-point projection and gap evaluation, contact pairs, augmented-Lagrangian enforcement, and self-contact workflows. |
| Eddy currents | [`eddy_current`](src/physics/eddy_current/) | — | ✓ | ✓ | ✓ | Harmonic magnetic vector-potential conductivity/reluctivity forms, current sources, 3-D A–φ with tree–cotree gauge, and an axisymmetric coil problem with flux linkage and unit-current solve. |
| Eigenproblems and modes | [`eigenmodes`](src/physics/eigenmodes.rs) | ✓ | ✓ | ◐ | ✓ | Interval Laplacian, scalar Robin/Dirichlet, acoustic, structural, electromagnetic and thermal-stability eigenproblems; proportional damping, nonsymmetric complex modes, and cross-mesh mode assignment. |
| Linear elasticity | [`elasticity`](src/physics/elasticity/) | R | ✓ | ✓ | ✓ | Isotropic plane stress/plane strain and 3-D elasticity, structural mass/body force, viscoelastic structural damping, materials, and Saint-Venant torsion on 2-D sections. |
| Electroacoustics | [`electroacoustics`](src/physics/electroacoustics/) | R | ✓ | ✓ | ✓ | Coupled electrical–mechanical–acoustic block systems, electrodynamic and electrostatic transducers, distributed Lorentz coupling, iterative frequency sweeps, time-domain response, thermal voice-coil coupling, cabinet radiation, and Thiele–Small extraction. |
| Electrochemistry | [`electrochemistry`](src/physics/electrochemistry/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron Nernst–Planck diffusion/electromigration, ionic charge source, Butler–Volmer boundary flux and consistent tangent, species and kinetic material models. |
| Electrostatics | [`electrostatic`](src/physics/electrostatic.rs) | R | ✓ | ◐ | ✓ | Electrostatic field observables and point-charge source support plus parallel-plate, capacitor, pull-in, linearized transducer, and push–pull reduced models. |
| Electrothermal | [`electrothermal`](src/physics/electrothermal/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron DC electrical conduction, DC/harmonic Joule heating, temperature-dependent electrothermal properties, heated-bar references, and energy-balance reporting. |
| Phase-field fracture | [`fracture`](src/physics/fracture/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron degraded elasticity, phase-field damage, damage-history right-hand side, fracture energy/length-scale material parameters, and coupled assembly. |
| Fluid–structure interaction | [`fsi`](src/physics/fsi/) | R | ✓ | — | ✓ | 3-D nonmatching dual-P1 mortar coupling with vector/normal constraints, conservative saddle and transfer operators, reference-map rebuild, ALE convection, and callback-based Aitken iteration. |
| Thin-film lubrication | [`lubrication`](src/physics/lubrication/) | R | ✓ | — | — | Reynolds pressure equation with wedge and squeeze-film sources, viscosity/density properties, cavitation state, and bearing/squeeze-film analytical references. The mesh is the film plane, not a 3-D fluid volume. |
| Lumped circuits | [`lumped_circuit`](src/physics/lumped_circuit/) | 0D | — | — | — | Modified nodal analysis, linear/nonlinear circuit components, voice-coil model, acoustic compliance/mass/resistance, Helmholtz resonator, and two-port transfer matrices. |
| Magnetostatics | [`magnetostatics`](src/physics/magnetostatics/) | R | ✓ | ✓ | ✓ | Reluctivity/permeability forms, permanent-magnet and point-dipole sources, azimuthal current density, nonlinear B–H curves, axisymmetric nonlinear bias solve, and magnetic energy reporting. |
| Maxwell electromagnetics | [`maxwell`](src/physics/maxwell/) | R | ✓ | — | ✓ | Frequency-domain curl–curl formulation with Nédélec H(curl) elements, complex permittivity/permeability/conductivity, impressed current, mass term, and electromagnetic PML. |
| Magnetohydrodynamics | [`mhd`](src/physics/mhd/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron inductionless electric-potential formulation, motional EMF and Lorentz-force coupling, Lorentz damping, coupled blocks, and conducting-fluid materials. |
| Reissner–Mindlin plates | [`mindlin_plate`](src/physics/mindlin_plate.rs) | — | ✓ | — | Surface | Bending, transverse shear and load forms, mixed plate space, and shear-locking stabilization for moderately thick plates. |
| Nonlinear solids | [`nonlinear_solid`](src/physics/nonlinear_solid/) | R | ✓ | — | ✓ | Hyperelastic residual/tangent forms with Neo-Hookean, Mooney–Rivlin, and St. Venant–Kirchhoff-style material behavior. |
| Phase-field two-phase flow | [`phase_field_flow`](src/physics/phase_field_flow/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron Cahn–Hilliard phase/chemical-potential system, advection, chemical-potential tangent, capillary force, properties, and mass/conservation reports. |
| Piezoelectricity | [`piezoelectric`](src/physics/piezoelectric/) | R | ✓ | — | ✓ | Coupled mechanical stiffness, dielectric and piezoelectric coupling forms, anisotropic piezo material tensors, and a d31-bar reduced reference. |
| Plasticity | [`plasticity`](src/physics/plasticity/) | R | ✓ | — | ✓ | Small-strain J2 return mapping, elastic/perfect/isotropic/kinematic hardening laws, plastic internal state, and shell/solid material-point updates. |
| Porous media | [`porous_flow`](src/physics/porous_flow/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron Darcy pressure/storage/source, Richards flow with van Genuchten–Mualem retention, and Biot pressure–displacement/divergence coupling. |
| Participating-media radiation | [`radiation_transport`](src/physics/radiation_transport/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron grey discrete-ordinates streaming/extinction, scattering coupling, thermal emission, 3-D angular quadrature, materials, and energy-balance references. |
| Reacting flow | [`reacting_flow`](src/physics/reacting_flow/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron species advection–diffusion, Arrhenius reactions and Jacobian, recovered reaction source, heat-release coupling, and species-conservation reporting. |
| Semiconductor transport | [`semiconductor`](src/physics/semiconductor/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron electron/hole drift–diffusion, doping charge/profile, recombination, and carrier material models. |
| Shallow water | [`shallow_water`](src/physics/shallow_water/) | R | ✓ | — | — | Depth-averaged free-surface mass and gravity-wave forms, bed-slope source, Manning friction, and shallow-water material parameters. The physical model is horizontally 2-D. |
| Shells | [`shell`](src/physics/shell/) | — | ✓ | ◐ | Surface | Kirchhoff–Love/DKT and shell elements, section resultants, corotational/nonlinear and elastoplastic updates. The discretization is a 2-D midsurface embedded in 3-D. |
| Stokes and Navier–Stokes | [`stokes`](src/physics/stokes/) | R | ✓ | — | ✓ | Stokes, Oseen and nonlinear incompressible Navier–Stokes; mixed velocity/pressure forms, SUPG/PSPG/LSIC stabilization, BDF stepping, CPU/backend solvers, preconditioners, and turbulence hooks. |
| Thermal transport | [`thermal`](src/physics/thermal/) | R | ✓ | ◐ | ✓ | Conduction, heat capacity, advection, convection matrix/RHS, radiation residual, volumetric/Joule sources, temperature-dependent materials, and transient time stepping. Axisymmetric scalar building blocks are available but not packaged as a dedicated thermal solver. |
| Thermoelasticity | [`thermoelasticity`](src/physics/thermoelasticity/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron thermal-strain load, thermoelastic damping, coupled thermal/structural blocks, layered materials, and constrained-stress references. |
| Thermoelectricity | [`thermoelectric`](src/physics/thermoelectric/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron electrical and thermal conduction with Seebeck, Peltier, Thomson and Joule terms, temperature-dependent coefficients, and coupled assembly. |
| General transport | [`transport`](src/physics/transport/) | R | ✓ | ◐ | ✓ | P1 triangle/tetrahedron scalar advection–diffusion–reaction/capacity/source forms, SUPG stabilization, linear reaction networks and Jacobians. Axisymmetric operators remain compositional. |
| Turbulence transport | [`turbulence`](src/physics/turbulence/) | R | ✓ | — | ✓ | P1 triangle/tetrahedron Spalart–Allmaras and k–ω/SST transport, production and reaction/source forms, constants, and coupling inputs for RANS workflows. |

## Cross-cutting capabilities

These capabilities are shared by multiple physics modules and therefore are not
repeated as separate rows above:

| Capability | Support |
|---|---|
| Scalar spaces | H1 Lagrange P1/P2 on triangles, quadrilaterals, tetrahedra, and hexahedra. |
| Vector spaces | Vector H1 plus Raviart–Thomas H(div) and Nédélec H(curl) elements where required. |
| Assembly | Generic bilinear/linear/nonlinear forms, marker-routed regional assembly, mixed systems, DG/facet terms, and prebaked frequency-sweep operators. |
| Boundary conditions | Dirichlet, Neumann, Robin, periodic/Floquet, impedance/radiation, and PML-specific formulations. |
| Time integration | Explicit and implicit acoustic stepping, thermal stepping, BDF fluid stepping, and coupled transducer dynamics. |
| Solvers | Direct and iterative sparse solvers, AMG/multigrid, shifted-Laplacian, WaveHoltz, Schwarz-PML, and block preconditioners. |
| Optimization | Adjoint, density/topology and shape optimization; optional autograd support. |

## Three-dimensional promotion QA

The fast promotion binaries check tetrahedral assembly, invariants, refinement,
and 2-D versus extruded-3-D behavior. The newly implemented production paths
also have independent solution-level binaries:

| Physics | Additional 3-D QA evidence |
|---|---|
| Biharmonic | Solved `sin(pi x) sin(pi y) sin(pi z)` mixed Navier MMS on two tetrahedral refinements, including `u`/`q` error, observed rate, and algebraic residual. |
| Fluid–structure interaction | Normal-only affine patch with arbitrary tangential slip, nonuniform virtual-work identity, explicit side selection, and below/above 5% reference-map motion. |
| Compressible flow | Nonuniform smooth DG state advanced with SSPRK3 in a closed slip-wall tetrahedral box, checking mass/energy conservation, positivity, and nonzero evolution. |

## Maintenance rule

When adding or extending a physics module, update this matrix only after the
corresponding dimension has a production form/assembler and at least one focused
unit, integration, or standalone QA test. Use **◐** until that evidence exists.
