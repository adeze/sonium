# `NDEigensystem` port matrix

Published source: [Wolfram `NDEigensystem` documentation](https://reference.wolfram.com/language/ref/NDEigensystem.html),
updated by Wolfram Research in 2023 and inspected on 2026-08-06.

The documentation contains many presentation and option variants of the same
mathematical problem. This matrix groups them by distinct physics or numerical
capability. `ready` means Sonium already has the required assembly and solver;
`small gap` means the ingredients exist but are not connected through an
eigenproblem API; `missing` identifies a substantive physics or solver addition.

| Published example family | Status | Sonium path or missing capability |
|---|---|---|
| 1D scalar Laplacian on an interval | implemented | Dedicated consistent-mass P1 interval pencil, exact spectrum unit test, and live Wolfram comparison |
| Clamped circular membrane | implemented | `disk_membrane.wls` and `wolfram_disk_membrane.rs` |
| Dirichlet rectangle and disk modes | implemented | Rectangle scalar fixture plus clamped circular-membrane fixture, both live-compared with Wolfram |
| Homogeneous Neumann Laplacian modes | implemented | Natural zero-flux scalar pencil, retained constant mode, exact rectangle check, and live Wolfram comparison |
| L-shaped and irregular scalar regions | implemented | Native concave simple-polygon ear clipping, uniform refinement, boundary tagging, and live Wolfram comparison |
| Periodic Laplacian | implemented | Real periodic/anti-periodic MPC condensation and live Wolfram comparison |
| Second-order wave equation modes | implemented | Scalar stiffness/mass pencil plus mass projection, wave modal evolution (including zero modes), reconstruction, analytic checks, and live Wolfram sample comparison |
| Liouville and weighted Sturm–Liouville operators | implemented | Variable-coefficient scalar builder and `wolfram_scalar_eigenproblems.rs` |
| Schrödinger operator with a potential | implemented | Potential term in scalar builder and `wolfram_scalar_eigenproblems.rs` |
| Acoustic modes from an image-mask car cross section | implemented | Pixel-exact binary-mask meshing (including holes), compact nodes, boundary tagging, and a self-contained live Wolfram `ImageMesh` comparison |
| Structural constraint modes | implemented | Plane-stress vector pencil, vector boundary-DOF elimination, and live fully constrained Wolfram comparison |
| Beam resonant frequency | implemented | Consistent-mass Euler-Bernoulli frame elements, shift-invert fallback for slender pencils, and live comparison with Wolfram's cantilever characteristic roots |
| Wave/heat eigenfunction expansions | implemented | Mass projection, modal reconstruction, wave and heat propagators, Dirichlet reduction mapping, analytic checks, and live Wolfram samples |
| Homogeneous generalized Neumann/Robin eigenmodes | implemented | Stiffness- and mass-side Robin terms in scalar builder; the mass-side equality is enforced by `wolfram_ndeigensystem_option_contracts.rs` |
| Partially constrained eigensystems | implemented | Marker-selective elimination, mixed Dirichlet/natural rectangle exact check, and live Wolfram comparison |
| Damped generalized wave equation | implemented | Arbitrary sparse `M s^2 + C s + K` pencils use first-order complex block linearization and shift-invert Arnoldi; proportional Rayleigh damping also has a cheaper exact modal path. The live Wolfram comparison exercises the general block solver |
| Coupled first-order/PDE eigensystems | implemented | General complex shift-invert Arnoldi plus two-field nonsymmetric reaction-diffusion assembly and live Wolfram comparison |
| Complex Hermitian wave eigenpairs | implemented | Complex Bloch MPC condensation, Hermitian realification, and mass-normalized eigenvectors |
| Nonsymmetric complex spectra | implemented | Sparse complex shift-invert Arnoldi, projected complex QR/Schur extraction, Ritz vectors, and independently checked true residuals |
| FEAST interval/band eigenvalue search | implemented | Complex-contour FEAST subspace iteration with an independent LDLT-inertia completeness certificate, repeated-eigenspace coverage, inclusive endpoints, SPD mass validation, and true residual gates; the live Wolfram comparison uses Arnoldi because the free macOS Engine omits `libFEAST` |
| Radial hydrogen Schrödinger equation | implemented | Reduced radial Coulomb weak form, regular origin condition, centrifugal potential, targeted negative spectrum, and live Wolfram comparison |
| Avoided crossing in a moving two-disk region | implemented | Structured implicit-region meshing, overlapping-disk unions, repeatable remeshing, Dirichlet mode expansion, cross-mesh physical-overlap assignment, sign alignment, and a three-point live Wolfram comparison |
| Periodic multi-well quantum band formation | implemented | Bloch potential sweep with global overlap assignment and phase-aligned modes |

## Accuracy follow-up completed

Smooth level-set and image-mask geometry now has an opt-in fitted path:
`fitted_implicit_region_mesh_triangles` projects boundary vertices with a
safeguarded Newton step and performs conforming boundary-adjacent refinement;
`fitted_binary_mask_mesh_triangles` applies the same machinery to the bilinear
half-value contour. The original pixel-exact staircase API is unchanged for
tests whose semantics are explicitly pixel based.

There are no unimplemented `NDEigensystem` source families in this inventory.

The option-only examples—normalization, extrapolation outside a region,
eigenvector sign ambiguity, mesh granularity, invalid inhomogeneous boundary
conditions, and generalized Robin mass terms—are enforced hermetically by
`tests/wolfram_ndeigensystem_option_contracts.rs`.
