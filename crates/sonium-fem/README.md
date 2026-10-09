# sonium-fem

Multi-physics Finite Element Method solver with a layered, physics-agnostic architecture.

## Installation

```toml
[dependencies]
sonium-fem = { version = "0.6" }
```

## Architecture

The crate separates physics from infrastructure. Adding a new physics domain only requires implementing `BilinearForm<T>` — no changes to assembly, boundary, solver, or multigrid.

```
Layer 0: Geometry    mesh/, basis/, quadrature/
Layer 1: Forms       forms/traits.rs  (BilinearForm<T>, LinearForm<T>)
Layer 2: Assembly    assembly/generic.rs, prebaked.rs  (physics-agnostic)
Layer 3: BCs         boundary/generic.rs  (FemSystem<T>)
Layer 4: Solver      solver/  (GMRES, CG, AMG, WaveHoltz, Schwarz-PML)
Layer 5: Physics     physics/*/forms.rs  (implement form traits)
```

## Features

### Pressure Acoustics

Full-featured acoustic simulation:

- **Frequency domain**: Helmholtz equation with PrebakedAssembler for O(nnz) frequency sweeps
- **Time domain**: Implicit Newmark-β and explicit leapfrog integrators
- **Viscothermal losses**: Equivalent fluid model (Zwikker-Kosten/LRF) for narrow tubes/slits, thermal boundary layer impedance for larger volumes
- **Boundary conditions**: Wall, impedance (Delany-Bazley, Miki porous layer models), waveguide ports with multimode expansion, radiation (Sommerfeld, BGT 2nd order), Floquet periodic
- **Sources**: Normal velocity, acceleration, displacement, pressure, monopole point sources
- **Post-processing**: Particle velocity, acoustic intensity, SPL, transmission loss, absorption coefficient

### Room Acoustics

- **Ray tracing**: Geometrical acoustics with Möller-Trumbore intersection, specular reflection, frequency-dependent surface absorption/scattering, configurable receiver spheres
- **Atmosphere attenuation**: ISO 9613-1 model (temperature, pressure, humidity dependent)
- **Room metrics**: RT60/RT30/EDT from Schroeder integration, C80, D50, Ts, Sabine/Eyring formulas
- **Acoustic diffusion**: Energy-based room analysis via diffusion equation (for quick estimates in large spaces)

### Transducer Modeling

- **Electrodynamic (voice coil)**: Magnetostatic BL extraction, voice coil circuit (linear + nonlinear with displacement-dependent BL, L, reluctance force), multi-domain coupling (electrical + mechanical + acoustic)
- **Electrostatic (condenser)**: Parallel-plate gap model (force, capacitance, coupling factor, pull-in voltage), linearised + large-signal dynamics
- **Lumped extraction**: Thiele-Small parameters, radiation impedance, effective area from FEM
- **Lumped elements**: Acoustic compliance/mass/resistance, Helmholtz resonator, two-port transfer matrix networks

### Structural Mechanics

- Linear isotropic elasticity (2D plane stress/strain, 3D)
- Viscoelastic materials (structural damping)
- Fluid-structure coupling (block system assembly)
- Predefined materials: steel, aluminum, rubber, loudspeaker surround/spider

### Electromagnetics

- Maxwell equations (H(curl) Nedelec elements)
- Magnetostatics (scalar potential, permanent magnets)
- Eddy currents (frequency-domain A-formulation)

### Additional Physics

- Thermal (conduction, advection, convection, radiation)
- Conjugate heat transfer across marker-routed fluid/solid regions
- Electrothermal DC and induction heating with temperature feedback
- Thermoelectric Seebeck/Peltier/Thomson transport
- Battery DFN/P2D operators and spherical active-particle diffusion
- Inductionless magnetohydrodynamics with Lorentz/motional-EMF coupling
- General ALE fluid-structure interaction with conservative interface transfer
- Thin-film Reynolds lubrication, squeeze films, and cavitation state
- Cahn-Hilliard diffuse-interface two-phase flow
- Nonlinear Arrhenius reacting flow with species and heat-release coupling
- Grey discrete-ordinates participating-media radiation transport
- Nonlinear solid (hyperelasticity, St. Venant-Kirchhoff)
- Beam (Euler-Bernoulli), Shell (Kirchhoff-Love, Reissner-Mindlin)
- Plasticity (return mapping)
- Topology optimization (with autograd)

Each of the ten coupled-physics modules above contains the three analytical or
semi-analytical reference cases specified in `20260806-multiphysics.md`, plus
null-state and conservation checks. The references include Reynolds/Sommerfeld/
Stefan lubrication, Domenicali/Ioffe/Snyder-Ursell thermoelectrics, Luikov/Graetz/
Gardner CHT, Dodd-Deeds induction heating, Hartmann/Shercliff/Hunt MHD,
Causin/Moens-Korteweg/BBO FSI, Cahn-Hilliard/Laplace/capillary waves, Crank/Sand/
de Levie batteries, Fisher/Burke-Schumann/Frank-Kamenetskii combustion, and
Chandrasekhar/Su-Olson radiation transport.

## Usage

```rust
use sonium_fem::physics::acoustics::pressure::{
    build_acoustics_assembler, AcousticsAssemblerExt,
};
use sonium_fem::basis::PolynomialDegree;
use num_complex::Complex64;

// Create mesh and assembler
let mesh = sonium_fem::mesh::unit_square_triangles(20);
let assembler = build_acoustics_assembler(&mesh, PolynomialDegree::P1);

// Assemble Helmholtz system at a specific wavenumber
let k = Complex64::new(5.0, 0.0);
let system = assembler.assemble_helmholtz(k, &Default::default());
```

## Solvers

| Solver | Best For | Complexity |
|--------|----------|------------|
| Direct (LU) | N < 1000 | O(N³) |
| GMRES+ILU | N < 10000 | O(N) per iter |
| GMRES+AMG | Large parallel | O(N) per iter |
| Shifted-Laplacian | High frequency | O(N) per iter |
| **WaveHoltz** | **High frequency, large N** | **O(N) total** |
| **Schwarz-PML** | **High frequency, k-robust** | **O(N) per iter** |

## Element Types

| Element | Dimension | Nodes (P1) | Nodes (P2) |
|---------|-----------|------------|------------|
| Triangle | 2D | 3 | 6 |
| Quadrilateral | 2D | 4 | 9 |
| Tetrahedron | 3D | 4 | 10 |
| Hexahedron | 3D | 8 | 27 |

## Feature Flags

- `native` (default) — Rayon parallelism + hardware BLAS
- `parallel` — Rayon for parallel assembly
- `cli` — CLI binary dependencies
- `mshio` — Gmsh `.msh` loading through `sonium-mesher`
- `meshlang` — MeshLang compilation plus Gmsh loading through `sonium-mesher`
- `autograd` — Automatic differentiation for topology optimization

## References

- D. Appelo et al., "WaveHoltz: Iterative Solution of the Helmholtz Equation via the Wave Equation", SIAM J. Sci. Comput., 2020
- J. Galkowski et al., "Optimised Schwarz methods with PML transmission conditions for the Helmholtz equation", 2024
- Stinson, "The propagation of plane sound waves in narrow and wide circular tubes", JASA, 1991
- ISO 9613-1:1993, "Acoustics — Attenuation of sound during propagation outdoors"
- ISO 3382-1:2009, "Acoustics — Measurement of room acoustic parameters"
