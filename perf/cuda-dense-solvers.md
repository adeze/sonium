# CUDA (NVIDIA GPU, Linux) — Dense solvers

_Seeded 2026-10-01 from root `PERF.md` x86/GPU record (RTX PRO 6000, 2026-09-26). Regenerate: `just perf-cuda`._

- status: seed

## Results

GPU LU via delegation at 1.00× (see `cuda-lapack.md`). One-shot
entries carry documented non-live costs (validating 266× = full
device-to-host validation; unchecked 9× = per-call prebake) — solver
loops use the persistent prebaked path at parity.
