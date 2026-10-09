# CUDA (NVIDIA GPU, Linux) — Sparse matrix

_Seeded 2026-10-01 from root `PERF.md` x86/GPU record (RTX PRO 6000, 2026-09-26 → 27). Regenerate: `just perf-cuda`._

- status: seed

## Results

| workload | size | Sonium | vendor | gap |
|---|---|---|---|---|
| SpMV f64 (prebaked) | 512k rows | 0.101 ms · 740 GB/s | cuSPARSE / roofline | 1.007× · near-roof |
| SpMM (prebaked) | 512k rows, 16 rhs | 1.255 ms | cuSPARSE best-alg 1.254 ms | 1.00× |
