# Apple M4 (macOS, aarch64) — Sparse matrix

_Generated 2026-10-07T20:06:35+00:00 from commit `3e4493316758-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 14 logical CPUs · mem unknown
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
mbx bench -p sonium-compute --bench roofline_preflight
```
```text
bash scripts/iterative_solver_perf.sh /var/folders/j4/s3d596x91csf_pdsjx3jf7800000gn/T/perf-cells-i5nh743u/m4-iterative.json
```
## Results

### CSR SpMV (roofline preflight)

| benchmark | median | bytes | eff. BW |
|---|---|---|---|
| `roofline_preflight_spmv/cpu/csr_f32_prebaked` | 112.18 µs | 589812 | 5.3 GB/s |

### Effective SpMV bandwidth inside iterative solves

| n | solve | iters | residual | eff. SpMV BW |
|---|---|---|---|---|
| 961 | 0.41 ms | 12 | 8.5e-07 | 2.88 GB/s |
| 10000 | 3.61 ms | 14 | 5.2e-07 | 4.01 GB/s |
| 99856 | 72.86 ms | 18 | 6.6e-07 | 2.56 GB/s |

