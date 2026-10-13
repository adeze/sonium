# ns_lid_cavity deck

Unit-square lid-driven cavity at Re=100 for the sonium-vs-OpenCourant
Navier-Stokes benchmark (`qa-oc-ns-bench --deck ns_lid_cavity`).

## Files

- `generate.py` — deck generator. Regenerate with `python3 generate.py`.
  Hand edits to the `.rad` files are lost on regeneration.
- `ns_lid_cavity_0000.rad` — starter deck (SI units: kg, m, s).
- `ns_lid_cavity_0001.rad` — engine deck (run controls, TH output).
- `probes.json` — engine-arm accuracy probes (see below).

## Setup

- Domain `[0,1] x [0,1]`, one brick layer deep (`dz = 1/64`), 64x64x1
  bricks, 8450 nodes. Node id `(i,j,k)` = `k*65^2 + j*65 + i + 1`.
- Material `/MAT/HYD_VISC/1`: `rho = 1.0`, `Knu = 0.01` (Re = 100),
  computed in the fixed-grid `/EULER/MAT` framework (no grid motion,
  no convection through boundary faces).
- Equation of state `/EOS/POLYNOMIAL/1`: ideal gas, `gamma = 1.4`
  (`C4 = C5 = 0.4`), `E0 = 714.29`, `Psh = 0`, giving `p0 = 285.7`
  and sound speed `c = 20`, i.e. Mach 0.05 at lid speed. The
  compressibility modeling error (~M^2 = 2.5e-3) sits an order of
  magnitude below the accuracy gate.
- Walls: `/BCS` no-slip (`111`) on `x = 0`, `x = 1`, `y = 0` nodes.
- Lid: `/IMPVEL` `Dir = X` with a ramped `/FUNCT` (`0 -> 1` over the
  first 2 seconds, held to `Tstop`); `Y`/`Z` locked by `/BCS` (`011`).
  Lid corners move with the lid, matching the sonium arm. The ramp
  keeps impulsive-start acoustics out of the closed box.
- Interior: `/BCS` `001` locks `z` (2D flow).
- Engine: `Tstop = 30`, `/TFILE` every `Tstop/1000`, `/TH/TITLE` for
  named TH channels, `/PRINT/-1000`. No `/ANIM` output (timed runs
  skip animation I/O).

## Probes

`probes.json` carries 9 `VX/<node>` channels on the `x = 0.5`
centerline (`y = 0, 0.125, ..., 1.0`, `k = 0` layer). Expected values
are the linear interpolation of the Ghia, Ghia & Shin (1982) Re=100
centerline table (authority:
`sonium_qa::analytical::ghia_lid_cavity_re100_u`) at the TH stations.

Calibrated 2026-10-12 on the settled run (90,054 cycles, late drift
`t = 22.5..30` of 5.3e-5): max probe error 0.0264 at `y = 0.875`;
every probe passes `abs_tol = 0.02`, `rel_tol = 0.25` with >= 3.9x
margin.

## Equivalence basis (vs sonium arm)

Same Reynolds number (100), same unit-square geometry, same lid speed
(`U = 1`), same no-slip walls, same lid-corner treatment (corners move
with the lid), same accuracy target (Ghia Re=100 centerline within a
few percent of lid speed). Deliberate modeling differences:

- Sonium solves steady incompressible NS directly (Picard); the engine
  marches explicit compressible ALE from rest to the same steady state
  (`Tstop = 30`, settled: late drift 5.3e-5).
- Sonium uses P1-P1 triangles (`n = 64`, 12,675 dofs); the engine uses
  64x64x1 bricks (8,450 nodes, 25,350 velocity dofs).
- Low-Mach compressibility (`M = 0.05`) is the engine-side modeling
  error; it is verified small by the Ghia comparison, not assumed.

The head-to-head verdict is time-to-matched-accuracy (both arms'
probes passing), reported in the bundle's `NS verdict` section. The
per-step `ns_per_dof_cycle` ratio is diagnostic only: explicit engine
cycles and implicit Picard iterations are not the same unit.

## Baseline (2026-10-12, Threadripper PRO 3995WX, 1 thread)

Accuracy-vs-time Pareto front (Ghia max error vs wall seconds):

| side   | mesh | wall (s) | Ghia error |
| ------ | ---- | -------- | ---------- |
| sonium | n=40 | 6.5      | 0.0283     |
| sonium | n=48 | 12.4     | 0.0230     |
| sonium | n=64 | 15.6     | 0.0162     |
| engine | n=32 | 29.8     | 0.0275     |
| engine | n=64 | 226.3    | 0.0264     |

(The n=64 sonium row is post-BVH-rewire; the pre-fix value was 39.4 s
at identical accuracy.) Sonium dominates at every accuracy the engine
reaches: at matched accuracy (~0.028) sonium-n40 is 4.6x faster than
engine-n32 (6.5 s vs 29.8 s); at the engine's accuracy floor (0.0264)
sonium-n48 is both more accurate (0.0230) and 18x faster (12.4 s vs
226 s). The engine's error floor (~0.026 at any mesh) is
modeling-dominated (low-Mach compressibility plus lid-shear-layer
diffusion), not discretization.

The checked-in engine deck is n=64 and the harness default sonium mesh
is n=64 (the `NS verdict` row reports 14.5x); both pass the 0.03 gate,
the engine with 1.14x margin on a frozen vendor binary (deterministic,
cannot regress) and sonium with 1.85x margin for development headroom.

## Scaling (same box, cavity n=64, wall seconds)

| threads | sonium | engine | speedup |
| ------- | ------ | ------ | ------- |
| 1       | 15.58  | 226.3  | 14.5x   |
| 8       | 11.02  | 60.8   | 5.5x    |
| 16/32   | 10.79  | 57.2   | 5.3x    |

Both sides saturate: sonium self-speedup is 1.44x at 16 threads
(assembly scales 2.45x; the GMRES+ILU solve is serial-triangular-solve
bound), the engine 3.95x at 32 threads. Absolute wall time favors
sonium 5x or more at every thread count. A level-scheduled parallel
triangular solve is the principled follow-up for sonium's
self-scaling.

## GPU (RTX PRO 6000 Blackwell, cavity + Taylor-Green)

| case       | CPU (s) | GPU (s) | speedup | backend proof |
| ---------- | ------- | ------- | ------- | ------------- |
| cavity n64 | 15.6    | 11.64   | 1.34x   | all-GPU, mixed FGMRES-IR + poly |
| cavity n128| 354 (stalled, gate failed) | 38.1 | 9.3x | all-GPU, error 0.0064 (mesh-convergent) |
| TG n128    | 23.07   | 14.40   | 1.60x   | all-CUDA, F64 + Schur/AMG |

Every GPU row is audited: the cavity arm checks the sticky resolved
backend per repeat, the Taylor-Green arm per BDF step. Any CPU
fallback marks the row `correct=false` (never silently relabeled).
Known limits: poly/Chebyshev GPU preconditioners cannot build on
saddle-point zero diagonals and fail closed to CPU (graceful, no
panic); mixed-precision Schur is f64-only and falls back likewise.
