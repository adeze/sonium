# CUDA (NVIDIA GPU, Linux) — Dense BLAS

_Seeded 2026-10-01 from root `PERF.md` compute-202609 worklog (idle RTX PRO 6000, 2026-09-27). Regenerate: `just perf-cuda`._

- status: seed

## Results

Bitwise-verified parity with cuBLAS across sizes 1024/2048/4096.

| workload | gap vs cuBLAS | correctness |
|---|---|---|
| dgemm | 1.000× | bitwise |
| sgemm | 1.001× | bitwise |
| dgemv | 0.994× | bitwise |
| sgemv | 1.008× | bitwise |
