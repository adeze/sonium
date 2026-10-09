# x86-64 (Linux) — Dense solvers

_Seeded 2026-10-01 from root `PERF.md` D3 records (Threadripper PRO 3995WX, 2026-09-26 → 27). Regenerate: `just perf-x86`._

- status: seed

## Results

End-to-end factor+solve. dgetrf-1024 in-place passes the project's
≤1.2× gate; dgetrf-2048 at 1.44× needs panel/Schur work.

| workload | Sonium | OpenBLAS | gap | gate |
|---|---|---|---|---|
| LU dgetrf-1024, in-place | 8.67–9.06 ms | 7.42 ms | 1.17× | ≤1.2× PASS |
| LU dgetrf-2048 | — | — | 1.44× | FAIL |
