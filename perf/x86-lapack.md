# x86-64 (Linux) — Dense LAPACK

_Generated 2026-10-01T10:30:42+00:00 from commit `47d8c891a376-dirty`.
`getrf_f64` and `potrf_f64` rows refreshed same-day from
`perf/dense-x86-cuda` (LU size-dependent nb + Cholesky all-wide compact
pool); see the notes under each table._

- status: measured
- host: Linux 6.8.0-142-generic x86_64 · 128 logical CPUs · mem 251 GB
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
bash scripts/direct_solver_perf.sh /tmp/direct-x86.md
```
```text
RAYON_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 cargo bench -p sonium-solvers-direct --bench lapack_compare --features bench-openblas
```
## Results

### Factorization kernels (`lapack_compare`)

Median of criterion samples; gap = sonium ÷ openblas.

#### `geqrf_f64`

| size | sonium | openblas | gap |
|---|---|---|---|
| 512 | 12.679 ms | 13.147 ms | 0.96× |
| 1024 | 64.156 ms | 54.479 ms | 1.18× |
| 2048 | 308.901 ms | 271.877 ms | 1.14× |
| 4096 | 1.431 s | 1.575 s | 0.91× |

#### `getrf_f32`

| size | sonium | openblas | gap |
|---|---|---|---|
| 1024 | 7.420 ms | 4.778 ms | 1.55× |
| 2048 | 25.145 ms | 16.174 ms | 1.55× |

#### `getrf_f64`

| size | sonium | openblas | gap |
|---|---|---|---|
| 512 | 2.791 ms | 1.719 ms | 1.62× |
| 1024 | 9.966 ms | 6.122 ms | 1.63× |
| 2048 | 42.828 ms | 29.947 ms | 1.43× |
| 4096 | 227.362 ms | 153.165 ms | 1.48× |

Refresh 2026-10-01 (`perf/dense-x86-cuda`, same harness/flags):
threaded f64 LU now narrows 48→32 at n ≥ 3072 (`LU_NARROW_PANEL_THRESHOLD`,
11/12 interleaved pairs at 4096). 2048/4096 are fresh medians from the
split-era binary (contended box — both arms ~10% above the clean
baseline, ratio robust); 512/1024 keep the clean-box medians (code path
identical, still nb48).
Merged 2026-10-03: `opt/compute-x86` kept HEAD LU (evo-owned); branch
measured 1.33/1.29/1.08/1.06× on its superseded LU — see `a9dc2a3e`.

#### `potrf_f32`

| size | sonium | openblas | gap |
|---|---|---|---|
| 1024 | 2.913 ms | 2.456 ms | 1.19× |
| 2048 | 8.750 ms | 7.183 ms | 1.22× |

#### `potrf_f64`

| size | sonium | openblas | gap |
|---|---|---|---|
| 512 | 1.205 ms | 0.854 ms | 1.41× |
| 1024 | 4.437 ms | 2.614 ms | 1.70× |
| 2048 | 19.603 ms | 14.830 ms | 1.32× |
| 4096 | 94.288 ms | 68.345 ms | 1.38× |

Refresh 2026-10-01 (`perf/dense-x86-cuda`, same harness/flags):
f64 Cholesky moved from the narrow (4-worker) to the wide compact pool
(`cholesky_compact_wide` all-wide; wide beats narrow 6/6 pairs at 512
and 1024, ties at 256). 512/1024 run on the wide pool now; 2048/4096
are outside the compact envelope (mirror pool before and after —
ratios unchanged within noise).
Merged 2026-10-03: `opt/compute-x86` kept HEAD Cholesky (evo compact
routing); branch measured 1.18/1.24/1.17/1.14× with TRSM-64 + compact
bypass — see `a9dc2a3e`. Re-regen on merged main recommended.

### Dense LU via `direct-solver-perf`

One factorization + one solve per timing; end-to-end reading lives in `x86-dense-solvers.md`.

| N | Sonium blocked (ms) | Sonium factor (ms) | Sonium solve (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes | Peak RSS bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.029 | 0.026 | 0.003 | — | — | 5.44e-16 | — | 4096 | 32768 | 67072 | 22745088 |
| 128 | 0.258 | 0.170 | 0.010 | — | — | 7.74e-16 | — | 16384 | 131072 | 265216 | 23482368 |
| 256 | 0.978 | 0.556 | 0.039 | — | — | 1.11e-15 | — | 65536 | 524288 | 1054720 | 27394048 |
| 512 | 3.146 | 2.478 | 0.142 | — | — | 2.43e-15 | — | 262144 | 2097152 | 4206592 | 32641024 |
