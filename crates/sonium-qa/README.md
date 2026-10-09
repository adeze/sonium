# sonium-qa

Quality assurance suite for Sonium. Its checked-in
[manifest inventory](manifest-inventory.md) is generated from the release
manifest, so validated-case and binary counts cannot drift from the source.

Each binary prints `SUCCESS` on a passing run and `FAILED` (with details) otherwise. The crate lives in the independent `sonium-accuracy` workspace and is wired into its root `Justfile` as deterministic QA tiers.

## Installation

This crate is internal — it is not published. It consumes the sibling Sonium
numerical crates through path dependencies and `math-dsp` through the accuracy
workspace dependency table.

## QA Tiers

```bash
just qa-smoke       # Fast pre-merge physics smoke validation
just qa-deep        # Convergence, analytical scattering, WaveHoltz, Schwarz-PML
just qa-expensive   # Full room/speaker application pipeline validation
just qa-full        # The complete standalone QA binary sweep
just qa-manifest    # Validate release-critical case metadata
```

## macOS Compute Evidence

The release-evidence producer benchmarks preallocated f32 BLAS-1 (AXPY),
BLAS-2 (GEMV), and BLAS-3 (GEMM) buffers against Accelerate across empty,
unit, tail, cache, square, tall/wide, and memory-oriented regimes. It writes
versioned JSON, Markdown, and SVG artifacts; it does not claim Metal/MPS or
energy measurements when those were not collected.

```bash
RAYON_NUM_THREADS=1 cargo run -p sonium-qa --release --bin bench-compute-macos-evidence
RAYON_NUM_THREADS=<production-thread-count> cargo run -p sonium-qa --release --bin bench-compute-macos-evidence
```

Set `SONIUM_COMPUTE_EVIDENCE_DIR` to archive the bundle outside `target/`.

Each Just recipe enforces:

- Binary exits with `SUCCESS` and zero exit code
- Output never contains `FAILED`
- Run completes within an explicit timeout

## Running a Single Binary

```bash
cargo run -p sonium-qa --release --bin qa-bem-basic
cargo run -p sonium-qa --release --bin qa-fem-analytical-validation
cargo run -p sonium-qa --release --bin qa-solvers-iterative
cargo run -p sonium-qa --release --bin qa-pipeline-physics-validation
```

## Cargo Test Harness

```bash
cargo test -p sonium-qa --release
```

Runs a fast smoke harness that executes representative BEM, FEM, and solver QA binaries with timeout + `SUCCESS`/`FAILED` output checks.

## Library Helpers

`sonium-qa` also exposes a small library used inside the binaries:

- `relative_error`, `l2_norm`, `residual_norm`, `relative_residual_norm` — error norms with explicit zero-reference handling
- `convergence_rate(errors, sizes)` — order estimation from error/element-size pairs
- `check_symmetry`, `check_spd` — cheap necessary-condition screening
- `sphere_mesh_points`, `cylinder_mesh_points`, `spd_tridiagonal`, `generate_rhs`
- `analytical_1d_laplacian_eigenvalues`, `analytical_1d_laplacian_eigenvector`
- `fixtures::*` and `validation::*` modules

Tolerances: `REL_TOL = 1e-2`, `ABS_TOL = 1e-8`, `RESIDUAL_TOL = 1e-8`, `GRADIENT_TOL = 1e-8`, `KKT_TOL = 1e-6`.

## Notes

- `check_spd` is a *necessary* screening check (positive diagonal + symmetry), not a full SPD proof. Use Cholesky for actual validation.
- The explicitly reference-only wave binaries compare analytical models with
  themselves and now emit structured records; they are excluded from BEM
  solver coverage by the manifest. `qa-bem-3d-sphere` is the production Mie
  solver evidence path.
- Binaries that take long-running real BEM/FEM solves only run in `qa-expensive` and `qa-full`.
- Release-facing `QaResult` records must declare threshold direction with the
  `minimum` or `maximum` builder. The runners fail closed when a threshold has
  no metric, no explicit direction, or an unsatisfied bound.
