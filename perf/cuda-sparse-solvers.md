# CUDA (NVIDIA GPU, Linux) — Sparse solvers + preconditioners

_Generated 2026-10-01T11:07:29+00:00 from commit `47d8c891a376-dirty`._

- status: measured
- host: Linux 6.8.0-142-generic x86_64 · 128 logical CPUs · mem 251 GB
- rustc: rustc 1.98.1 (48a229cea 2026-09-01)
- gpu: GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (UUID: GPU-fdc918d0-2387-567e-92d3-fcce47a2facc)

## Commands

```text
cargo bench -p sonium-compute --bench block_gmres_accelerator --features cuda
```
## Results

### Block GMRES workstation (N=512 diagonal f32, 8 rhs)

| backend | median |
|---|---|
| cpu | 2.980 ms |
| cuda | 2.924 ms |

_Tiny diagonal smoke system: both arms are overhead-dominated and tie.
A production-size comparison (larger N, harder spectrum) is still open._

### Mixed-precision GMRES-IR

CUDA GMRES-IR converges with outer/inner residuals digit-identical to
the CPU baseline (2026-09-25 record). Clean-window timing remains queued:
tenant contention dominates current numbers.
