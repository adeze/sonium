# Direct Solver Performance

_Generated 2026-09-26T13:34:49Z from commit `bc1e4fc5-dirty`._

## Platform

| Property | Value |
|---|---|
| OS | Linux 6.8.0-142-generic #142-Ubuntu SMP PREEMPT_DYNAMIC Wed Sep  2 14:24:27 UTC 2026 |
| Architecture | x86_64 |
| CPU | AMD Ryzen Threadripper PRO 3995WX 64-Cores |
| Memory | 251.5 GiB |
| Logical CPUs | 128 |
| Rust | rustc 1.98.1 (48a229cea 2026-09-01) |

## Backend availability

| Backend | Status | Workload |
|---|---|---|
| Sonium blocked LU | available | dense |
| Apple Accelerate | unavailable on this platform | dense |
| Sonium sparse simplicial | available | sparse |
| Sonium sparse supernodal | available | sparse |
| SuperLU | available (6.0.1) | sparse |
| CUDA | adapter pending | dense and sparse |
| ROCm | adapter pending | dense and sparse |

## Dense LU: factorization and solve

| N | Sonium blocked (ms) | Sonium factor (ms) | Sonium solve (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes | Peak RSS bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.296 | 0.202 | 0.024 | — | — | 1.22e-15 | — | 16384 | 131072 | 265216 | 10571776 |
| 256 | 0.948 | 0.859 | 0.088 | — | — | 1.55e-15 | — | 65536 | 524288 | 1054720 | 23191552 |
| 512 | 5.694 | 4.091 | 0.176 | — | — | 2.43e-15 | — | 262144 | 2097152 | 4206592 | 33955840 |
| 1024 | 18.753 | 17.716 | 0.692 | — | — | 3.11e-15 | — | 1048576 | 8388608 | 16801792 | 73916416 |

## Sparse LU: 2D nonsymmetric convection-diffusion

| Grid | N | NNZ | Simplicial (ms) | Simplicial factor (ms) | Simplicial solve (ms) | Supernodal (ms) | Supernodal factor (ms) | Supernodal solve (ms) | Reused symbolic (ms) | Reused factor (ms) | Reused solve (ms) | SuperLU (ms) | Simplicial / SuperLU | Supernodal / SuperLU | Simplicial factor nnz | Supernodal factor nnz | Reused factor nnz | Simplicial factor bytes | Supernodal factor bytes | Reused factor bytes | Modeled peak bytes | Peak RSS bytes | Max residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16×16 | 256 | 1216 | 0.652 | 0.632 | 0.017 | 0.837 | 0.814 | 0.016 | 0.616 | 0.596 | 0.016 | 0.344 | 1.90× | 2.44× | 6060 | 6060 | 6103 | 48480 | 48480 | 48824 | 76480 | 57499648 | 2.58e-15 |
| 24×24 | 576 | 2784 | 1.944 | 1.892 | 0.049 | 2.168 | 2.111 | 0.047 | 1.679 | 1.614 | 0.047 | 0.905 | 2.15× | 2.39× | 17854 | 17854 | 17865 | 142832 | 142832 | 142920 | 205904 | 57499648 | 1.76e-15 |
| 32×32 | 1024 | 4992 | 20.552 | 4.416 | 14.792 | 20.476 | 4.069 | 15.510 | 19.778 | 3.201 | 15.886 | 1.699 | 12.10× | 12.05× | 38988 | 38988 | 39017 | 311904 | 311904 | 312136 | 424784 | 57905152 | 1.05e-14 |
| 40×40 | 1600 | 7840 | 30.027 | 7.701 | 22.087 | 31.420 | 6.634 | 23.403 | 27.909 | 5.362 | 20.891 | 2.712 | 11.07× | 11.58× | 65924 | 65924 | 66216 | 527392 | 527392 | 529728 | 706376 | 59211776 | 2.60e-15 |

## Methodology

- Times are medians of 7 measured runs after one warm-up run.
- Each measured operation includes one factorization and one right-hand-side solve; the factor/solve columns split the two stages (solve reuses the timed factorization's factor).
- The reused-symbolic column excludes Sonium ordering and symbolic analysis; all other sparse columns are one-shot.
- Factor nnz is the stored numeric factor count; factor bytes use scalar width and exclude allocator overhead.
- Modeled peak bytes include the input matrix, largest reported factor, and three scalar work vectors; this is an explicit lower-bound model, not OS RSS.
- Peak RSS bytes is the process high-water mark (`VmHWM` on Linux, `n/a` elsewhere), sampled after each size; it covers the harness, allocator overhead, and all backends.
- Every backend must pass a relative infinity-norm residual gate of `1e-10` before timing.
- Ratios above 1.0 mean Sonium is slower than the platform reference; below 1.0 means faster.
- Regenerate with `scripts/direct_solver_perf.sh`. Override sizes with `DIRECT_PERF_DENSE_SIZES`, `DIRECT_PERF_SPARSE_GRIDS`, and `DIRECT_PERF_REPEATS`.
