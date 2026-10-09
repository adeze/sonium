# x86-64 (Linux) — Sparse matrix

_Generated 2026-10-01T10:35:49+00:00 from commit `47d8c891a376-dirty`._

- status: measured
- host: Linux 6.8.0-142-generic x86_64 · 128 logical CPUs · mem 251 GB
- rustc: rustc 1.98.1 (48a229cea 2026-09-01)

## Commands

```text
cargo bench -p sonium-compute --bench roofline_preflight
```
```text
bash scripts/iterative_solver_perf.sh /tmp/perf-cells-_48n9em9/x86-iterative.json
```
## Results

### CSR SpMV (roofline preflight)

| benchmark | median | bytes | eff. BW |
|---|---|---|---|
| `roofline_preflight_spmv/cpu/csr_f32_prebaked` | 111.26 µs | 589812 | 5.3 GB/s |

### Effective SpMV bandwidth inside iterative solves

| n | solve | iters | residual | eff. SpMV BW |
|---|---|---|---|---|
| 961 | 0.57 ms | 12 | 8.5e-07 | 2.05 GB/s |
| 10000 | 6.82 ms | 14 | 5.2e-07 | 2.12 GB/s |
| 99856 | 95.15 ms | 18 | 6.7e-07 | 1.96 GB/s |

