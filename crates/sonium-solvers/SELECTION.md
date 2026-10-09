# High-Level Solver Selection Evidence

_Generated 2026-09-23T16:56:49Z from commit `9034219e` plus the uncommitted
audit worktrees (selection-evidence bench is new in this tree)._
Answers audit-sonium-solvers-20260921 §Priorities: "publish high-level
selection evidence on representative PDE matrices with setup, solve, memory
and error separated."

## Platform

| Property | Value |
|---|---|
| OS | Linux 6.8.0-139-generic x86_64 |
| Rust | rustc 1.98.1 (48a229cea 2026-09-01) |
| Logical CPUs | 128 |
| Commit | 9034219e (+ uncommitted audit tree) |

## Method

Harness: `crates/sonium-solvers/benches/selection_evidence_bench.rs`
(`cargo bench -p sonium-solvers --bench selection_evidence_bench`).
Every case drives the **public facade** with `Auto` dispatch
(`solve`, `solve_complex`) unless the case names an explicit strategy.
Config: tolerance `1e-8`, max 500 iterations, GMRES restart 30.
Right-hand sides are manufactured (`rhs = A·1`) and the error column is
measured against `1`, independently of the Krylov residual estimator.

- **setup ms**: median of 5 explicit preconditioner builds through the same
  public `try_from_csr*` constructors the facade uses for the resolved
  strategy (ILU(0), AMG with `AmgConfig::for_fem`, Schwarz overlap 1).
- **total ms**: median of 5 facade solves after 1 discarded warm-up.
- **solve ms**: total − setup (derived).
- **memory**: lower-bound model, PERF.md style, never OS RSS: exact CSR
  storage + Krylov basis ((restart+1)·N·w; +1 basis for FGMRES; 5·N·w for CG)
  + 3 work vectors, plus a stated factor rule — ILU(0)/Schwarz-subdomain at
  one matrix pattern (no fill), AMG at 2× (upper-bound model; exact AMG
  complexities live in `preconditioner_bench`), saddle block factors at one
  matrix pattern (setup is internal to `solve_saddle_point`, hence `n/a`).
- **large arms**: `helmholtz-2d-large` and `laplacian-3d-large` set
  `SONIUM_LARGE_PROBLEM_ROWS=1000` / `SONIUM_LARGE_PROBLEM_NNZ=20000`
  (saved/restored around those cases) to exercise the large-problem dispatch
  arms without benchmark-scale matrices.
- A failed (`Err`) solve fails the bench loudly; non-convergence would be
  recorded with its residual rather than hidden. All cases below converged.

## Results

| case | class | scalar | N | nnz | assumption | resolved | setup ms | solve ms | total ms | iters | residual | backward err | mfg err | converged | peak KiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poisson-2d | SPD Laplacian | f64 | 2304 | 11328 | General | GmresIlu | 0.287 | 6.798 | 7.084 | 50 | 5.47e-10 | 7.51e-13 | 8.59e-10 | true | 984.0 |
| poisson-2d-cgamg | SPD Laplacian | f64 | 2304 | 11328 | PositiveDefinite | CgAmg | 7.462 | 200.644 | 208.106 | 49 | 7.52e-9 | 1.03e-11 | 1.93e-8 | true | 693.0 |
| convection-diffusion-2d | nonsymmetric upwind | f64 | 4096 | 20224 | General | GmresIlu | 0.514 | 20.776 | 21.290 | 88 | 6.26e-10 | 5.55e-13 | 4.08e-9 | true | 1752.0 |
| helmholtz-2d | complex indefinite | Complex64 | 1024 | 4992 | General | GmresIlu | 0.191 | 40.464 | 40.655 | 341 | 9.97e-10 | 3.17e-12 | 2.38e-9 | true | 786.0 |
| helmholtz-2d-large | complex indefinite (large arm) | Complex64 | 9216 | 45696 | General | FgmresSchwarz | 4.183 | 692.831 | 697.014 | 523 | 9.83e-10 | 4.67e-13 | 3.29e-8 | true | 7789.5 |
| laplacian-3d-large | SPD Laplacian 3D (large arm) | f64 | 8000 | 53600 | General | GmresIlu | 1.518 | 31.896 | 33.414 | 26 | 7.76e-10 | 8.05e-13 | 7.39e-10 | true | 3862.5 |
| saddle-point-kkt | KKT block (contract scale) | f64 | 4 | 10 | General | solve_saddle_point(explicit) | n/a | n/a | 0.026 | 3 | 1.25e-16 | 3.78e-17 | 1.56e-16 | true | 1.4 |

## Reading

- `Auto` resolves to `GmresIlu` for every small/real case and routes the
  forced-large complex Helmholtz case to `FgmresSchwarz`, which converges
  (523 iterations, residual 9.83e-10, backward error 4.67e-13). ILU on the
  small indefinite Helmholtz case converges but needs 341 iterations: the
  indefinite regime is where the Schwarz arm earns its keep.
- The explicit `CgAmg` + `PositiveDefinite` path converges on SPD Poisson
  but is ~29× slower than Auto's ILU choice at N=2304 with a ~26× dearer
  setup: evidence that Auto's nonsymmetric-safe ILU default is the right
  call at this scale, and that AMG evidence must be gathered at larger N
  before recommending it by default.
- The explicit saddle-point path solves the KKT fixture to machine precision
  in 3 iterations.
- Backward errors sit 1–3 decades below the reported residuals on every
  PDE case (machine-precision tie on the 4×4 saddle fixture): the facade's
  independent original-system check agrees with the Krylov estimator
  throughout.
