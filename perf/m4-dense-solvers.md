# Apple M4 (macOS, aarch64) — Dense solvers

_Generated 2026-10-07T20:06:36+00:00 from commit `3e4493316758-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 14 logical CPUs · mem unknown
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
bash scripts/direct_solver_perf.sh /var/folders/j4/s3d596x91csf_pdsjx3jf7800000gn/T/perf-cells-i5nh743u/m4-direct.md
```
## Results

End-to-end dense LU: one factorization + one right-hand-side solve per timing.

| N | Sonium blocked (ms) | Sonium factor (ms) | Sonium solve (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes | Peak RSS bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.027 | 0.024 | 0.003 | 0.014 | 1.91× | 8.71e-16 | 1.09e-15 | 4096 | 32768 | 67072 | n/a |
| 128 | 0.262 | 0.300 | 0.008 | 0.057 | 4.56× | 1.22e-15 | 1.44e-15 | 16384 | 131072 | 265216 | n/a |
| 256 | 1.003 | 0.841 | 0.029 | 0.307 | 3.27× | 1.77e-15 | 2.65e-15 | 65536 | 524288 | 1054720 | n/a |
| 512 | 2.989 | 3.184 | 0.110 | 1.612 | 1.85× | 2.87e-15 | 3.32e-15 | 262144 | 2097152 | 4206592 | n/a |
