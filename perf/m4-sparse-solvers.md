# Apple M4 (macOS, aarch64) — Sparse solvers + preconditioners

_Generated 2026-10-07T20:06:36+00:00 from commit `3e4493316758-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 14 logical CPUs · mem unknown
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
bash scripts/direct_solver_perf.sh /var/folders/j4/s3d596x91csf_pdsjx3jf7800000gn/T/perf-cells-i5nh743u/m4-direct.md
```
```text
bash scripts/iterative_solver_perf.sh /var/folders/j4/s3d596x91csf_pdsjx3jf7800000gn/T/perf-cells-i5nh743u/m4-iterative.json
```
## Results

### Sparse direct (LU)

| Grid | N | NNZ | Simplicial (ms) | Simplicial factor (ms) | Simplicial solve (ms) | Supernodal (ms) | Supernodal factor (ms) | Supernodal solve (ms) | Reused symbolic (ms) | Reused factor (ms) | Reused solve (ms) | SuperLU (ms) | Simplicial / SuperLU | Supernodal / SuperLU | Simplicial factor nnz | Supernodal factor nnz | Reused factor nnz | Simplicial factor bytes | Supernodal factor bytes | Reused factor bytes | Modeled peak bytes | Peak RSS bytes | Max residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8×8 | 64 | 288 | 0.039 | 0.036 | 0.001 | 0.072 | 0.069 | 0.001 | 0.045 | 0.040 | 0.001 | 0.029 | 1.34× | 2.49× | 844 | 844 | 844 | 6752 | 6752 | 6752 | 13416 | n/a | 1.05e-15 |
| 12×12 | 144 | 672 | 0.101 | 0.087 | 0.002 | 0.185 | 0.171 | 0.002 | 0.131 | 0.119 | 0.002 | 0.067 | 1.49× | 2.74× | 2598 | 2598 | 2631 | 20784 | 20784 | 21048 | 36416 | n/a | 2.35e-15 |
| 16×16 | 256 | 1216 | 0.241 | 0.193 | 0.007 | 0.569 | 0.423 | 0.006 | 0.312 | 0.381 | 0.006 | 0.147 | 1.64× | 3.86× | 6060 | 6060 | 6103 | 48480 | 48480 | 48824 | 76480 | n/a | 2.67e-15 |
| 20×20 | 400 | 1920 | 0.453 | 0.382 | 0.009 | 0.782 | 0.648 | 0.009 | 0.659 | 0.670 | 0.010 | 0.255 | 1.78× | 3.07× | 10864 | 10864 | 10887 | 86912 | 86912 | 87096 | 130624 | n/a | 1.04e-14 |
| 28×28 | 784 | 3808 | 1.162 | 0.990 | 0.022 | 2.023 | 1.791 | 0.021 | 1.465 | 1.382 | 0.022 | 0.675 | 1.72× | 3.00× | 26090 | 26090 | 26177 | 208720 | 208720 | 209416 | 295440 | n/a | 4.68e-15 |

### Iterative + preconditioners

62 runs · pairs: bicgstab+none, cg+amg, cg+jacobi, cg+none, fgmres+amg, gmres+amg, gmres+ilu, gmres+ilut, gmres+jacobi, gmres+none, idrs+jacobi, idrs+none

matrices: anisotropic_diffusion_2d, helmholtz_2d, laplacian_3d, poisson_2d, saddle_point_2d

#### CG + AMG on Poisson 2-D (f64)

| n | solve | iters | residual | eff. SpMV BW |
|---|---|---|---|---|
| 961 | 0.41 ms | 12 | 8.5e-07 | 2.88 GB/s |
| 10000 | 3.61 ms | 14 | 5.2e-07 | 4.01 GB/s |
| 99856 | 72.86 ms | 18 | 6.6e-07 | 2.56 GB/s |

