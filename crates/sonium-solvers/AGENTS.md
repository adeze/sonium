# sonium-solvers

`sonium-solvers` (lib: `sonium_solvers`, version 0.4.5) is the compatibility
facade for the focused solver crates below. Public facade paths remain stable;
new code can depend on the narrow crate it needs.

## Crate map

| Crate | Responsibility |
|---|---|
| `sonium-solvers-core` | `ComplexField`, operators, preconditioner traits, dense matrices, vector/BLAS helpers, `PrecisionCast` |
| `sonium-solvers-sys` | Runtime detection, NUMA topology, lane execution, rayon helpers |
| `sonium-solvers-sparse` | CSR/BSR/triplet matrices, sparse operators, SpMV plans |
| `sonium-solvers-eigen` | LOBPCG and adjoint eigen sensitivities |
| `sonium-solvers-direct` | Dense/sparse LU and Cholesky, multifrontal BLR/LDLT |
| `sonium-solvers-preconditioners` | Jacobi, ILU/ILUT/HILU, AMG, Schwarz, Chebyshev, saddle-point |
| `sonium-solvers-iterative` | GMRES/FGMRES, CG, BiCGSTAB, IDR(s), MINRES, TFQMR and block variants |
| `sonium-solvers-optimization` | BFGS/L-BFGS(-B), SQP/SLSQP, MMA, IPM, CMA-ES and related methods |
| `sonium-solvers-nonlinear` | Newton-Raphson and Anderson acceleration |
| `sonium-solvers-time-integration` | Newmark, generalized-alpha, explicit integration and checkpointing |
| `sonium-solvers` | Stable facade, high-level `solve`, mixed precision, differentiable APIs, binaries and benches |

All numerical crates use the shared `ComplexField` abstraction over f32, f64,
Complex32, Complex64, and f16 where supported.

## Testing

Run the focused suite for the crate being changed:

```bash
cargo test -p sonium-solvers-core
cargo test -p sonium-solvers-sys
cargo test -p sonium-solvers-sparse
cargo test -p sonium-solvers-eigen
cargo test -p sonium-solvers-direct
cargo test -p sonium-solvers-preconditioners
cargo test -p sonium-solvers-iterative
cargo test -p sonium-solvers-optimization
cargo test -p sonium-solvers-nonlinear
cargo test -p sonium-solvers-time-integration
cargo test -p sonium-solvers
```

The facade owns `solve.rs`, `mixed_precision.rs`, differentiable contracts,
remaining cross-cutting integration tests, binaries, examples, and benches.
The distributed integration tests live under the owning sub-crate's `tests/`
directory. For a single test, append its name to the corresponding command.

Before merging solver changes, run:

```bash
cargo check --workspace
cargo clippy --workspace --all-targets
```

The `sonium-solvers` facade features `cli`, `cuda`, `metal`, and
`experimental-gnn`; CUDA/Metal and GNN features are forwarded to the relevant
sub-crates.
