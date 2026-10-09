# PETSc Reference Benchmarks

Compare sonium-solvers against PETSc (the industry-standard reference) on the same matrices, same tolerance, same preconditioners.

## Prerequisites

1. Install PETSc 3.20+ with HYPRE support:
   ```bash
   git clone https://gitlab.com/petsc/petsc.git
   cd petsc
   ./configure --with-debugging=0 --with-64-bit-indices=0 \
       --download-hypre --with-shared-libraries=1 \
       COPTFLAGS="-O3" CXXOPTFLAGS="-O3"
   make all
   ```

2. Set environment:
   ```bash
   export PETSC_DIR=/path/to/petsc
   export PETSC_ARCH=arch-darwin-c-opt  # or your arch name
   ```

## Building

```bash
cd crates/sonium-solvers/benches/ffi
./build_petsc.sh
```

## Generating test matrices

Export matrices from the Rust benchmarks in MatrixMarket format, or download from [SuiteSparse](https://sparse.tamu.edu/).

## Running

```bash
# Single run
./petsc_reference matrix.mtx gmres ilu 1e-6 500

# Sweep all solver/precond combinations
for solver in gmres fgmres cg bicgstab; do
  for pc in none jacobi ilu amg; do
    ./petsc_reference matrix.mtx $solver $pc 1e-6 500
  done
done
```

Output is JSON (one line per run):
```json
{"solver":"gmres","precond":"ilu","n":10000,"nnz":49600,"setup_ms":12.3,"solve_ms":45.6,"total_ms":57.9,"iterations":42,"residual":1.2e-07,"converged":true}
```

## Comparison workflow

1. Run Rust benchmarks: `cargo bench -p sonium-solvers --bench solver_bench`
2. Export a test matrix from Rust (or use SuiteSparse .mtx files)
3. Run PETSc on same matrix: `./petsc_reference test.mtx gmres ilu 1e-6 500`
4. Compare: iterations (algorithm quality) and wall time (implementation speed)

## Key solver/precond mappings

| sonium-solvers | PETSc flag |
|---|---|
| `gmres` | `-ksp_type gmres` |
| `fgmres` | `-ksp_type fgmres` |
| `bicgstab` | `-ksp_type bcgs` |
| `cg` / `pcg` | `-ksp_type cg` |
| `DiagonalPreconditioner` | `-pc_type jacobi` |
| `IluPreconditioner` / `IlutPreconditioner` | `-pc_type ilu` |
| `AmgPreconditioner` | `-pc_type hypre -pc_hypre_type boomeramg` |
