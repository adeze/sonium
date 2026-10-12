# Solver conformance inventory

`src/conformance.rs` is the canonical, machine-readable inventory of the
public solver algorithms. Each row records the matrix/model assumptions,
stopping-accuracy contract, breakdown policy, scalar coverage, maturity, and
the executable evidence that backs the row. The table is re-exported by the
facade so release QA can inspect the same data that downstream callers see.

The release gate is `solvers.conformance-matrix.v1` in
`crates/sonium-qa/validation-manifest.toml`. It requires unique, non-empty
rows and rejects algorithms marked `DocumentedOnly`. The current inventory
contains 56 rows: 26 independently referenced rows, 30 cross-checked rows,
and no documented-only rows. Twelve optimization rows are production-ready;
the remaining twelve public optimization algorithms are explicitly marked
experimental or research and are not eligible for automatic facade dispatch.
Optimization rows are algorithm-granular (24 distinct names), so a new public
method cannot hide behind a combined family alias.

## Evidence policy

- `Independent` means an analytic solution, an independent implementation,
  or a deterministic reference fixture is executed and checked.
- `CrossChecked` means the production path is checked against a sibling
  implementation or an equivalent independent execution contract.
- `DocumentedOnly` is reserved for roadmap work and fails the release gate
  until executable evidence is added.

The inventory is descriptive rather than an algorithm selector. Every solver
still reports its native result and typed failure; callers must use the row's
assumptions and stopping rule when interpreting a result. The direct-solver
property packet additionally checks real/complex and f32/f64 LU, dense/sparse
Cholesky, block versus single right-hand sides, independent residuals,
nonsymmetric row/column and symmetric permutations, and explicit rank-deficient
rejection in `tests/factor_solve_properties.rs`; the promoted
`solvers.factor-solve-reference.v1` QA case remains the release-facing subset.

`validation_corpus.rs` is the companion versioned fixture inventory. It
contains deterministic SPD, HPD, symmetric-indefinite, nonsymmetric,
saddle-point, ill-scaled, near-singular, Helmholtz, and convection-diffusion
cases, including fixed 16x16 Poisson and convection-diffusion workloads. Each
row records workload and conditioning notes; the `validation_corpus`
integration test checks independent solutions, residuals, multi-RHS behavior,
factor workload metadata, and iterative convergence.

Constrained optimization has a release-facing KKT gate in
`solvers.optimization-kkt-reference.v1`, covering an equality-constrained SQP
solution and an active-upper-bound L-BFGS-B solution. The gate requires the
original-problem feasibility/stationarity metrics and an attached bound KKT
report before accepting success.

The cross-family failure packet is `solvers.failure-exit-matrix.v1`. Its
seven fixtures cover direct factorization, ILU setup, Newton callback panic,
invalid SQP bounds, LOBPCG non-finite operators, and Newmark dimension
failure; the integration test keeps the same calls compiled in the facade
crate so a QA-build dependency failure cannot hide API drift.

To inspect the gate locally:

```text
cargo test -p sonium-solvers --test factor_solve_properties --offline
cargo run --release -p sonium-qa --bin qa-solvers-conformance-matrix \
  --features qa-binaries --offline
```

Broader external comparisons (PETSc, SuiteSparse, SciPy, IPOPT), sustained
FEM/BEM workloads, accelerator backends, and long-running fuzz campaigns are
tracked as separate review items; they are not implied by this inventory.

The release QA case `solvers.external-differential-reference.v1` exercises a
live PETSc GMRES comparison when `SONIUM_PETSC_REFERENCE` points at a usable
binary. It compares identical MatrixMarket Poisson and convection-diffusion
fixtures, independently recomputed residuals, iteration counts, and timing
metadata. If PETSc is unavailable the case is explicitly `Skipped` (or fails
with `SONIUM_REQUIRE_PETSC=1`); an unavailable external solver is never counted
as a passing differential result.

## Production consumer workload

`solvers.fem-bem-workload.v1` exercises the assembled FEM Helmholtz CSR
path and dense BEM TBEM path over three frequencies each. The release QA
binary records assembly and solve timings, iterations, FEM plane-wave nodal
error, and independently recomputed true residuals. The current CPU run
passes all six solves with maximum residual `6.94e-9` (FEM) and `5.71e-9`
(BEM). This is bounded correctness/workload evidence; it is not a claim of
external-solver competitiveness or large-scale production throughput.

The checked solver fuzz targets `checked_fgmres`, `checked_csr`, and
`checked_preconditioners` each pass a 100-run libFuzzer smoke campaign. The
repository workflow repeats the three bounded campaigns; sustained mutation
and long-running fuzz coverage remain separate gates.
`solvers.optimization-scipy-reference.v1` adds an optional live SciPy
differential packet for the BFGS/L-BFGS lane. It runs the same translated
diagonal quadratic through Sonium Strong-Wolfe BFGS/L-BFGS and SciPy
BFGS/L-BFGS-B, with explicit skip/fail behavior controlled by
`SONIUM_REQUIRE_SCIPY`. The current local SciPy 1.11.4 run passes with
maximum cross-method coordinate gap `8.87e-9`. IPOPT/PRIMA and constrained
large-scale optimizer comparisons remain separate open gates.
