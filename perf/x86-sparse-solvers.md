# x86-64 (Linux) — Sparse solvers + preconditioners

_Generated 2026-10-01T10:51:04+00:00 from commit `47d8c891a376-dirty`._

- status: measured
- host: Linux 6.8.0-142-generic x86_64 · 128 logical CPUs · mem 251 GB
- rustc: rustc 1.98.1 (48a229cea 2026-09-01)

## Commands

```text
bash scripts/direct_solver_perf.sh /tmp/direct-x86.md
```
```text
ITERATIVE_PERF_PETSC=$PWD/target/petsc_reference bash scripts/iterative_solver_perf.sh target/petsc-iterative.json
```
## Results

### Sparse direct (LU)

| Grid | N | NNZ | Simplicial (ms) | Simplicial factor (ms) | Simplicial solve (ms) | Supernodal (ms) | Supernodal factor (ms) | Supernodal solve (ms) | Reused symbolic (ms) | Reused factor (ms) | Reused solve (ms) | SuperLU (ms) | Simplicial / SuperLU | Supernodal / SuperLU | Simplicial factor nnz | Supernodal factor nnz | Reused factor nnz | Simplicial factor bytes | Supernodal factor bytes | Reused factor bytes | Modeled peak bytes | Peak RSS bytes | Max residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8×8 | 64 | 288 | 0.105 | 0.096 | 0.002 | 0.173 | 0.168 | 0.002 | 0.112 | 0.104 | 0.002 | 0.069 | 1.51× | 2.50× | 844 | 844 | 844 | 6752 | 6752 | 6752 | 13416 | 31019008 | 8.22e-16 |
| 12×12 | 144 | 672 | 0.283 | 0.272 | 0.005 | 0.533 | 0.533 | 0.005 | 0.389 | 0.378 | 0.006 | 0.190 | 1.49× | 2.81× | 2598 | 2598 | 2631 | 20784 | 20784 | 21048 | 36416 | 31432704 | 6.49e-15 |
| 16×16 | 256 | 1216 | 0.567 | 0.558 | 0.010 | 1.039 | 1.027 | 0.009 | 0.766 | 0.699 | 0.008 | 0.343 | 1.65× | 3.03× | 6060 | 6060 | 6103 | 48480 | 48480 | 48824 | 76480 | 32079872 | 2.58e-15 |
| 20×20 | 400 | 1920 | 0.872 | 0.838 | 0.015 | 1.549 | 1.536 | 0.015 | 1.230 | 1.213 | 0.016 | 0.588 | 1.48× | 2.63× | 10864 | 10864 | 10887 | 86912 | 86912 | 87096 | 130624 | 33017856 | 2.37e-14 |
| 28×28 | 784 | 3808 | 1.968 | 1.939 | 0.035 | 3.288 | 3.248 | 0.038 | 2.677 | 2.651 | 0.035 | 1.301 | 1.51× | 2.53× | 26090 | 26090 | 26177 | 208720 | 208720 | 209416 | 295440 | 34881536 | 1.17e-14 |

### Iterative + preconditioners

108 runs · pairs: bicgstab+none, cg+amg, cg+jacobi, cg+none, fgmres+amg, gmres+amg, gmres+ilu, gmres+ilut, gmres+jacobi, gmres+none, idrs+jacobi, idrs+none

matrices: anisotropic_diffusion_2d, helmholtz_2d, laplacian_3d, poisson_2d, saddle_point_2d

#### CG + AMG on Poisson 2-D (f64)

| n | solve | iters | residual | eff. SpMV BW |
|---|---|---|---|---|
| 961 | 0.60 ms | 12 | 8.5e-07 | 1.97 GB/s |
| 10000 | 13.35 ms | 14 | 5.2e-07 | 1.08 GB/s |
| 99856 | 92.99 ms | 18 | 6.7e-07 | 2.01 GB/s |

#### Sonium vs PETSc KSP (f64, rhs=1)

| solver+precond | matrix | n | sonium | petsc | gap | iters s/p | conv s/p |
|---|---|---|---|---|---|---|---|
| bicgstab+none | poisson_2d | 99856 | 952.02 ms | 456.30 ms | 2.09× | 500/371 | ✗/✓ |
| cg+amg | anisotropic_diffusion_2d | 99856 | 56.70 ms | 35.74 ms | 1.59× | 8/5 | ✓/✓ |
| cg+amg | laplacian_3d | 97336 | 111.13 ms | 75.44 ms | 1.47× | 13/7 | ✓/✓ |
| cg+amg | poisson_2d | 99856 | 92.99 ms | 72.15 ms | 1.29× | 18/8 | ✓/✓ |
| cg+jacobi | anisotropic_diffusion_2d | 99856 | 49.01 ms | 21.40 ms | 2.29× | 13/13 | ✓/✓ |
| cg+jacobi | laplacian_3d | 97336 | 321.11 ms | 116.26 ms | 2.76× | 92/92 | ✓/✓ |
| cg+jacobi | poisson_2d | 99856 | 1200.19 ms | 924.54 ms | 1.30× | 500/500 | ✗/✗ |
| cg+none | laplacian_3d | 97336 | 171.19 ms | 102.44 ms | 1.67× | 92/92 | ✓/✓ |
| cg+none | poisson_2d | 99856 | 411.27 ms | 470.25 ms | 0.87× | 500/500 | ✗/✗ |
| fgmres+amg | poisson_2d | 99856 | 105.11 ms | 99.78 ms | 1.05× | 17/12 | ✓/✓ |
| gmres+amg | poisson_2d | 99856 | 4133.86 ms | 49.02 ms | 84.34× | 510/7 | ✗/✓ |
| gmres+ilu | poisson_2d | 99856 | 3278.67 ms | 1231.43 ms | 2.66× | 917/500 | ✓/✗ |
| gmres+ilu | saddle_point_2d | 99372 | 132.17 ms | 88.54 ms | 1.49× | 30/26 | ✓/✓ |
| gmres+jacobi | poisson_2d | 10000 | 1182.54 ms | 54.33 ms | 21.77× | 1043/500 | ✓/✗ |
| gmres+none | poisson_2d | 10000 | 675.98 ms | 52.53 ms | 12.87× | 1043/500 | ✓/✗ |

Harness validation: 37 passed, 6 failed out of 43 comparisons.

_Caveats: GMRES restart lengths differ between the harnesses (sonium runs longer cycles; PETSc defaults to restart 30), so iteration counts are not directly comparable. PETSc AMG is HYPRE BoomerAMG used as a solver; sonium AMG is its own smoothed-aggregation preconditioner. Rows where either side hit `max_iter` compare a converged solve against a truncated one._
