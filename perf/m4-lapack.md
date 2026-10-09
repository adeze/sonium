# Apple M4 (macOS, aarch64) — Dense LAPACK

_Generated 2026-10-07T20:04:45+00:00 from commit `3e4493316758-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 14 logical CPUs · mem unknown
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
bash scripts/direct_solver_perf.sh /var/folders/j4/s3d596x91csf_pdsjx3jf7800000gn/T/perf-cells-i5nh743u/m4-direct.md
```
```text
mbx bench -p sonium-solvers-direct --bench lapack_compare
```
## Results

### Factorization kernels (`lapack_compare`)

Median of criterion samples; gap = sonium ÷ accelerate.

#### `geqrf_f64`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 512 | 6.434 ms | 3.991 ms | 1.61× |
| 1024 | 31.915 ms | 21.042 ms | 1.52× |
| 2048 | 215.277 ms | 148.843 ms | 1.45× |
| 4096 | 1.626 s | 1.043 s | 1.56× |

#### `getrf_f32`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 1024 | 8.733 ms | 3.830 ms | 2.28× |
| 2048 | 23.110 ms | 40.650 ms | 0.57× |

#### `getrf_f64`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 512 | 6.859 ms | 910.61 µs | 7.53× |
| 1024 | 28.671 ms | 5.213 ms | 5.50× |
| 2048 | 234.660 ms | 50.370 ms | 4.66× |
| 4096 | 781.804 ms | 247.696 ms | 3.16× |

#### `potrf_f32`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 1024 | 21.369 ms | 513.81 µs | 41.59× |
| 2048 | 22.242 ms | 3.715 ms | 5.99× |

#### `potrf_f64`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 512 | 9.446 ms | 385.60 µs | 24.50× |
| 1024 | 29.151 ms | 1.857 ms | 15.69× |
| 2048 | 23.632 ms | 19.055 ms | 1.24× |
| 4096 | 141.269 ms | 119.643 ms | 1.18× |

### Dense LU via `direct-solver-perf`

One factorization + one solve per timing; end-to-end reading lives in `m4-dense-solvers.md`.

| N | Sonium blocked (ms) | Sonium factor (ms) | Sonium solve (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes | Peak RSS bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.027 | 0.024 | 0.003 | 0.014 | 1.91× | 8.71e-16 | 1.09e-15 | 4096 | 32768 | 67072 | n/a |
| 128 | 0.262 | 0.300 | 0.008 | 0.057 | 4.56× | 1.22e-15 | 1.44e-15 | 16384 | 131072 | 265216 | n/a |
| 256 | 1.003 | 0.841 | 0.029 | 0.307 | 3.27× | 1.77e-15 | 2.65e-15 | 65536 | 524288 | 1054720 | n/a |
| 512 | 2.989 | 3.184 | 0.110 | 1.612 | 1.85× | 2.87e-15 | 3.32e-15 | 262144 | 2097152 | 4206592 | n/a |
