# sonium-solvers

High-performance linear, nonlinear, and optimization solvers for computational acoustics.

Current direct-solver timing, residual, factor-fill, factor-storage, and
modeled-memory evidence against available external backends is recorded in
[`PERF.md`](PERF.md). The modeled memory value is a deterministic lower bound,
not OS peak RSS; the snapshot is correctness/performance evidence, not a claim
of competitiveness across all matrix classes.

All solvers are generic over the `ComplexField` trait: **f32, f64, Complex32, Complex64** are all supported with the same API.

## Iterative Solvers

- **GMRES(m)** with restart — general non-symmetric systems
- **FGMRES** — flexible GMRES for variable preconditioners
- **BiCGSTAB** — non-symmetric, no restart needed
- **CGS** — conjugate gradient squared
- **CG / PCG** — symmetric positive definite systems
- **IDR(s)** — induced dimension reduction, short recurrence
- **GMRES-deflated** — for clustered eigenvalues

All support preconditioned variants and optional initial guess.

## Direct Solver

- **LU decomposition** with partial pivoting (generic over ComplexField)

Checked low-level factor solves can use `DirectFactorization::solve_with_result`
to obtain `DirectSolveResult<T>`, which reports zero direct iterations and a
finite-solution `Converged` status without claiming a residual that the factor
cannot independently recompute. Attach an independently recomputed residual
and backward error with `with_accuracy_check` before publishing an attained
accuracy claim; contradictory or non-finite metadata is rejected by the shared
`SolverResultContract` validator.

## Preconditioners

### CPU (host arrays, in-house BLAS-1)

| Preconditioner | Description |
|---------------|-------------|
| Diagonal (Jacobi) | Simple diagonal scaling, fully parallel |
| ILU(0) | Incomplete LU, best convergence on small/medium |
| ILU-coloring | Parallel ILU via level scheduling |
| ILU-fixed-point | Parallel ILU via Jacobi iteration |
| HILU | Hierarchical ILU with nested dissection |
| ILUT | ILU with drop tolerance |
| Block Jacobi | Block Jacobi with ILUT subdomains |
| Additive Schwarz | Overlapping domain decomposition |
| **AMG** | Algebraic multigrid (parallel coarsening + smoothing) |
| Chebyshev | Polynomial preconditioner with auto eigenvalue estimate |

### GPU-friendly (device buffers via `sonium_compute::Backend`)

Pure SpMV + BLAS-1 preconditioners that run on `CpuBackend`, `CudaBackend`, or `MetalBackend` through the same `gmres_backend_preconditioned` entry point. No triangular solve, no host round-trip.

| Preconditioner | Best for | Cost per apply |
|---|---|---|
| `IdentityBackendPreconditioner` | Sanity checks, well-conditioned problems | 1 d2d copy |
| `PointJacobiBackend` | Robust baseline | 1 SpMV (diagonal CSR) |
| `PolynomialJacobiBackend` (degree K, damping ω) | Indefinite saddle-point (Stokes, Oseen) with `ω ∈ [0.5, 0.8]` | K SpMVs + (K−1) full SpMVs |
| `ChebyshevBackend` (degree K, λ-ratio) | SPD or near-SPD (Poisson, Helmholtz, elasticity, k-equation) | K diagonal SpMVs + (K−1) full SpMVs |

#### Picking a GPU preconditioner

The two big choices are **kind** and **degree K**.

| If your matrix is… | Use | Default degree |
|---|---|---|
| SPD or near-SPD (Laplacian, mass, stiffness, Helmholtz < cutoff) | `Chebyshev` | K=5 (PETSc/HYPRE default) |
| Indefinite saddle-point (Stokes, Oseen, mixed velocity-pressure) | `PolynomialJacobi` with damping ω = 0.7 | K=2 |
| Strongly non-symmetric or you don't know yet | `PolynomialJacobi` with ω=1.0 | K=1 (point-Jacobi) |

`PolynomialJacobi` with K=1 and ω=1.0 is bit-equivalent to `PointJacobi` — start there, increase K (and switch to Chebyshev if SPD-ish) only if GMRES iteration count is too high.

#### When the GPU path beats CPU+ILU

CPU ILU(0) is a useful baseline at small sizes. Whether a parallel
preconditioner wins depends on setup, conditioning, iteration count and
hardware; compare these costs rather than assuming a crossover:

- **Problem size** — at n ≈ 50k+ the legacy ILU's setup cost (sequential factorisation) collapses, while the iterative GPU path's per-iter cost grows linearly with nnz. Bench numbers on a 2-D Poisson `n=49729`: `Cheb K=5` on CpuBackend is ~11× faster than legacy ILU; on Metal it's ~5× faster. See `cargo bench -p sonium-solvers --bench backend_gmres_bench`.
- **Hardware** — `sonium-compute` has native CUDA kernels through cudarc and
  native Metal MSL/MPS paths. Support, precision and host fallback are
  operation-specific; the former blanket claim that Metal BLAS-1/SpMV are
  host-staged through oxicuda is obsolete. Measure the actual backend,
  transfers, setup, iteration count and attained residual together; native
  execution does not guarantee a fixed GPU speedup.

The ~10k/50k thresholds and Poisson ratios above are historical workload
observations, not portable selection rules. Include CPU
`LinearSolverType::GmresIlu` in the comparison and select using matched
matrix, precision, residual tolerance and current platform measurements.

## High-Level API

```rust
use sonium_solvers::{solve, CsrMatrix, LinearSolveConfig, LinearSolverType};

let config = LinearSolveConfig {
    solver: LinearSolverType::GmresIlu,
    ..Default::default()
};
let result = solve(&matrix, &rhs, &config)?;
```

The generic `solve<T>()` function works with any `ComplexField` type.

## Nonlinear Solver

Newton-Raphson with configurable damping, generic over ComplexField:

```rust
use sonium_solvers::{newton_raphson, NewtonConfig};

let result = newton_raphson(&x0, residual_fn, jacobian_solve_fn, &config)?;
```

## Time Integration

Newmark-β for second-order ODEs (`M u'' + C u' + K u = f(t)`), generic over ComplexField:

```rust
use sonium_solvers::{newmark_integrate, NewmarkConfig};

let config = NewmarkConfig::average_acceleration(dt, num_steps);
let result = newmark_integrate(damping, &stiffness, force_fn, &initial, solve, &config);
```

## Optimization

- **Unconstrained**: BFGS, L-BFGS, Nelder-Mead, Powell, nonlinear CG, Newton-CG
- **Constrained**: SQP, augmented Lagrangian, penalty, L-BFGS-B
- **Multi-objective**: Epsilon-constraint method

## Sparse Matrices

`CsrMatrix<T>` with SIMD-optimized SpMV (AVX2 on x86, NEON on ARM), automatic parallelization for large matrices via rayon.

Matrix-free `SumOperator`, `CompositeOperator`, and `FnOperator` also provide
checked in-place/scratch-buffer actions for callers that need typed dimension
errors; checked actions contain callback panics and reject non-finite generated
values or undeclared adjoint operations. Their historical panic-compatible
constructors remain available.

## Feature Flags

- `native` (default) — BLAS/LAPACK + rayon for maximum performance
- `parallel` — Rayon parallelism only
- `wasm` — WebAssembly support (pure-Rust fallback)
