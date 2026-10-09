# CUDA (NVIDIA GPU, Linux) — Dense LAPACK

_Generated 2026-10-01T11:05:39+00:00 from commit `47d8c891a376-dirty`._

- status: measured
- host: Linux 6.8.0-142-generic x86_64 · 128 logical CPUs · mem 251 GB
- rustc: rustc 1.98.1 (48a229cea 2026-09-01)
- gpu: GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (UUID: GPU-fdc918d0-2387-567e-92d3-fcce47a2facc)

## Commands

```text
RAYON_NUM_THREADS=16 cargo bench -p sonium-solvers-direct --bench lapack_compare --features cuda -- getrf_f64
```
```text
RAYON_NUM_THREADS=16 cargo bench -p sonium-solvers-direct --bench lapack_compare --features cuda -- potrf_f64
```
## Results

### Factorization kernels (`lapack_compare`)

Median of criterion samples; gap = sonium ÷ cusolver.

#### `getrf_f64`

| size | sonium | cusolver | gap |
|---|---|---|---|
| 512 | 2.837 ms | 5.622 ms | 0.50× |
| 1024 | 9.622 ms | 15.122 ms | 0.64× |
| 2048 | 41.828 ms | 45.878 ms | 0.91× |
| 4096 | 266.579 ms | 174.395 ms | 1.53× |

#### `potrf_f64`

| size | sonium | cusolver | gap |
|---|---|---|---|
| 512 | 1.390 ms | 5.000 ms | 0.28× |
| 1024 | 5.634 ms | 10.708 ms | 0.53× |
| 2048 | 19.475 ms | 26.010 ms | 0.75× |
| 4096 | 118.076 ms | 83.833 ms | 1.41× |

New arm 2026-10-01 (`perf/dense-x86-cuda`): `dense_cholesky_f64`
cuSOLVER delegation (`Dpotrf`, lower). One-shot with H2D upload, like
the getrf arm; crossover between 2048 and 4096, same as LU. The 4096
CPU arm ran contended (+26% vs the x86 table median), so the 1.41×
overstates the crossover margin — the GPU arm itself (83.8 ms incl
~5 ms upload) is the stable number.

### Dense LU via `direct-solver-perf`

One factorization + one solve per timing; end-to-end reading lives in `cuda-dense-solvers.md`.

GPU LU via cuSOLVER delegation at 1.00× of direct cuSOLVER calls (4096: 163.36 ms vs 162.99 ms, 2026-09-26): solver loops use the persistent prebaked path at parity; the one-shot kernel table above includes H2D upload and crosses over between 2048 and 4096.
