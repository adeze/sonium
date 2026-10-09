# sonium-bem (lib: `sonium_bem`, version: 0.4.16)

Boundary Element Method solver for acoustic scattering and room simulation.

## Key Features

- High-performance BEM solver for acoustic problems (Helmholtz exterior
  problems via the Burton–Miller combined formulation)
- Dense (TBEM), H-matrix/ACA, SLFMM and MLFMM assembly paths
- GMRES iterative solver via `sonium-solvers`
- Porous-material impedance models and XEM exchange admission

## Features

- `cli` - Command-line interface (enables the `numcalc` binary)
- `sofa` - SOFA/HRTF I/O (enables the `output2sofa` binary, requires `cli`)
- `meshlang` - MeshLang compilation plus Gmsh loading through `sonium-mesher`
- `metal` - Metal GPU offload for the regular-pair kernel
- `cuda` - CUDA GPU offload for the regular-pair kernel

## Binaries

- `numcalc` - BEM CLI (requires `cli`)
- `output2sofa` - SOFA output conversion (requires `cli`, `sofa`)

## Dependencies

- `sonium-solvers` - GMRES solver
- `sonium-wave` - Analytical solutions for validation
- `sonium-compute` - SIMD kernels, Metal/CUDA backends

## Testing

```bash
cargo test -p sonium-bem --lib
cargo check -p sonium-bem && cargo clippy -p sonium-bem
```

## Notes

- Single acoustic convention: `e^{-iwt}`, `G=e^{ikr}/4pr`, outward normals
  (see `CONVENTIONS.md`); XEM admission rejects inward normals and sign
  mismatches.
- `apply_row_sum_correction` is a diagnostics-only helper (returns the
  average row-sum magnitude, does not modify the system).
- Use the `try_*` checked constructors (`try_surface_impedance`,
  `try_build_*`, `try_merge`, `try_from_control_points`); the panicking
  legacy wrappers (`surface_impedance`, `apply_row_sum_correction`, …)
  are `#[deprecated]`.
