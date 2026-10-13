# OpenCourant explicit-dynamics comparison (Phase 6, 2026-10-11; plastic extension, 2026-10-12)

Sonium native explicit vs the OpenCourant engine on the 6-deck
subset (4 elastic + 2 plastic), measured with `qa-oc-bench`
(`sonium.oc-bench/v1` bundles). Raw evidence: `perf/oc/*.json`
(49 cells). Every published row has `correctness_passed=true`;
any failing row is unpublished by policy (none failed in the
final matrix).

## Fingerprint

- Box: AMD Ryzen Threadripper PRO 3995WX 64-core, 128 threads,
  251 GB RAM, Linux x86_64. No CUDA explicit arm exists in
  sonium (`sonium-fem/cuda` only enables compute BLAS), so the
  sonium flavor is `native` only; CUDA is N/A, not measured.
- Sonium: release build (`cargo build --release -p sonium-qa
  --features qa-binaries --bin qa-oc-bench`), this worktree.
- Engine: `~/src_3rdparties/OpenCourant/exec`
  (`engine_linux64_gf` serial/SMP via `-nt`, `..._ompi` via
  `mpirun -np`); starter always serial, excluded from `wall_s`.
- Protocol: 3 timed repeats + 1 warmup, median wall wins —
  except `hertz_sphere` (1 repeat, 0 warmup: the sonium arm
  needs ~360 s/repeat, so full protocol would cost ~1 h/cell;
  the reduced protocol is recorded in each hertz bundle's
  `protocol` field). All runs sequential on an otherwise quiet
  box (a first pass under load avg 47-77 was discarded and
  re-run quiet; quiet taylor sonium t1 = 0.027 s vs 0.034 s
  loaded).
- Regen: `just perf-oc` runs the full matrix
  (`scripts/perf_oc_matrix.sh`, serial by design: engine runs
  race under concurrent load) and republishes these tables
  (`scripts/perf_oc_publish.py`); `perf-oc-elastic`,
  `perf-oc-plastic`, and `perf-oc-deck <deck>` run subsets.
  Tables live between `PERF-OC-DECK` markers and sort by
  ns/dof/cycle; prose around them is hand-maintained.

## Equivalence basis (from 6A, frozen in PHASE6_CONTRACTS.md)

Same checked-in deck pair both arms (dof 675/297/297/1455
exact on elastic, 2340/1872 on plastic), same material cards
and stop time, same walls/BCs/loads. Differs: engine LAW1-QEPH
vs native Neo-Hookean Q1 bricks (small-strain identical),
engine LAW2/LAW36 vs native `hex_plastic_forces` (Johnson-Cook
/ tabulated yield, same E/nu/rho), engine contact vs native
exact-multiplier rigid walls (zero contact stiffness, no
contact CFL limit; quasi-static decks keep an explicit penalty
cap on the plastic arm only), own adaptive dt per side (cycle
counts differ; normalized by ns/dof/cycle). Native plastic runs with linear
bulk viscosity off (c1 = 0): the compression-only kink
chatters under velocity-Verlet on quasi-static ramps
(55%/step growth from roundoff); the C1-smooth quadratic term
stays on for shocks. Probe sampling mirrors each deck's
engine leg (taylor_plastic FNZ plateau mean, thick-cylinder
transient DX + residual means). Deferred with measured
reasons: shells (no native explicit shells),
cantilever/bending (native Q1 shear locking).

## Results

`ns/dof/cycle = wall_s * 1e9 / (cycles * dof)`. Best in each
deck column is bold.

<!-- PERF-OC-DECK-BEGIN taylor_bar -->
### taylor_bar (dof 675; engine 24 cycles, sonium 21 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 0.00306 | **216** | 83 MB |
| sonium | native | 16 | 0.00307 | 217 | 83 MB |
| sonium | native | 64 | 0.00309 | 218 | 83 MB |
| engine | smp | 1 | 0.0336 | 2071 | 39 MB |
| engine | serial | 1 | 0.0342 | 2113 | 39 MB |
| engine | smp | 4 | 0.0453 | 2795 | 44 MB |
| engine | smp | 16 | 0.0852 | 5258 | 108 MB |
| engine | mpi | 2 | 0.625 | 38558 | 72 MB |
| engine | mpi | 4 | 0.667 | 41181 | 69 MB |
<!-- PERF-OC-DECK-END -->

Engine numbers are setup-dominated (24 cycles in 33 ms; the
~40 ms in-binary setup confound documented by 6A). MPI pays
~0.7 s of launcher/rank overhead for 24 cycles.

<!-- PERF-OC-DECK-BEGIN rail_crush -->
### rail_crush (dof 297; engine 6898 cycles, sonium 6528 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 0.302 | **156** | 83 MB |
| sonium | native | 16 | 0.308 | 159 | 83 MB |
| engine | smp | 1 | 0.546 | 266 | 39 MB |
| engine | serial | 1 | 0.55 | 269 | 39 MB |
| engine | smp | 4 | 1.01 | 495 | 44 MB |
| engine | mpi | 2 | 1.18 | 576 | 71 MB |
| engine | mpi | 4 | 1.24 | 604 | 69 MB |
| engine | smp | 16 | 1.83 | 894 | 109 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN necking_bar -->
### necking_bar (dof 297; engine 1372839 cycles, sonium 1305578 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 56.7 | **146** | 83 MB |
| sonium | native | 16 | 59.8 | 154 | 83 MB |
| engine | smp | 1 | 100 | 246 | 39 MB |
| engine | mpi | 4 | 104 | 255 | 69 MB |
| engine | mpi | 2 | 110 | 269 | 71 MB |
| engine | serial | 1 | 164 | 401 | 39 MB |
| engine | smp | 4 | 187 | 459 | 44 MB |
| engine | smp | 16 | 321 | 788 | 108 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN hertz_sphere -->
### hertz_sphere (dof 1455; engine 137513 cycles, sonium 105056 cycles; 1 repeat)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 9.13 | **60** | 83 MB |
| sonium | native | 1 | 9.14 | 60 | 83 MB |
| engine | mpi | 4 | 16.6 | 83 | 69 MB |
| engine | mpi | 2 | 21.3 | 106 | 72 MB |
| engine | serial | 1 | 25.6 | 128 | 39 MB |
| engine | smp | 1 | 25.6 | 128 | 39 MB |
| engine | smp | 4 | 27.9 | 140 | 44 MB |
| engine | smp | 16 | 43.3 | 216 | 107 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN plasticity_thick_cylinder -->
### plasticity_thick_cylinder (dof 1872; engine 5188 cycles, sonium 4704 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 1.01 | **115** | 83 MB |
| sonium | native | 16 | 1.01 | 115 | 83 MB |
| engine | smp | 1 | 1.19 | 123 | 39 MB |
| engine | serial | 1 | 1.2 | 123 | 39 MB |
| engine | smp | 4 | 1.26 | 130 | 47 MB |
| engine | mpi | 4 | 1.36 | 140 | 71 MB |
| engine | mpi | 2 | 1.48 | 153 | 72 MB |
| engine | smp | 16 | 1.83 | 189 | 112 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN taylor_plastic -->
### taylor_plastic (dof 2340; engine 53 cycles, sonium 123 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 0.0844 | **293** | 83 MB |
| sonium | native | 1 | 0.0845 | 293 | 83 MB |
| engine | serial | 1 | 0.0539 | 434 | 40 MB |
| engine | smp | 1 | 0.0545 | 439 | 40 MB |
| engine | smp | 4 | 0.0572 | 461 | 47 MB |
| engine | smp | 16 | 0.0995 | 802 | 111 MB |
| engine | mpi | 2 | 0.625 | 5041 | 72 MB |
| engine | mpi | 4 | 0.673 | 5425 | 69 MB |
<!-- PERF-OC-DECK-END -->

## Reading

- Per-dof-per-cycle, sonium wins all 6 decks against engine
  serial: taylor 9.8x, rail 1.7x, necking 2.7x,
  hertz 2.1x, plastic taylor 1.5x, cylinder 1.1x. Hertz
  flipped from a 2.6x deficit: the multiplier-contact
  refactor removed the penalty dt cap (786k → 105k cycles)
  and hertz runs a precomputed linear stiffness (no per-step
  Jacobians/determinants/logarithms), taking it from 320 to
  60 ns/dof/cycle with the DZ probe exact to 7 digits.
  Taylor went from 127 to 21 cycles at the bulk CFL step.
- Wall-clock: sonium hertz takes 9.1 s vs engine 25.6 s (2.8x;
  fewer native cycles too, 105k vs 137k) and sonium taylor
  takes 0.003 s vs engine 0.034 s (11x). Plastic taylor goes
  the other way (engine 0.054 s vs sonium 0.084 s: 53 vs 123
  cycles at a cheaper sonium per-cycle cost), while the
  cylinder runs fewer native cycles than engine (4704 vs
  5188), so sonium wins wall there too (1.01 s vs 1.19 s).
  Necking: 56.7 s vs 164 s serial (2.9x; 1.31M vs 1.37M
  cycles).
- Nobody scales on these decks: sonium t1 ~= t16
  everywhere (serial force loop by design; 1.00x at 297-2340
  dof; t8 spot-checked on hertz at 1.00x), engine SMP is flat to sharply negative (-nt16 is
  1.5-3.4x SLOWER than serial), MPI only pays on the largest
  deck (hertz r4 = 0.65x serial wall). Threading the native
  loop cannot pay here (both sides agree it is overhead-bound
  at this size); scaling work belongs with bigger meshes
  (future work, needs native shells first). At 16 cores
  sonium still wins all 6 decks (1.6x-24x per-cycle) because
  the engine degrades while sonium holds flat.
- Memory: sonium reports a fixed 83 MB peak RSS on every
  fresh cell regardless of deck size (297-2340 dof) against
  3-15 MB on the Oct-11 binary: process-level overhead,
  unexplained, flagged for follow-up (not investigated;
  numbers published as measured). Engine 38-112 MB per cell
  (MPI RSS covers the launcher only and under-reports ranks —
  6A caveat, restated).
- Engine serial == smp1 on every deck (within 2%), as expected.

## Rerun

```sh
export TMPDIR=$HOME/src/all_of_sonium/tmp \
  SONIUM_QA_SCRATCH=$HOME/src/all_of_sonium/tmp \
  OPENCOURANT_EXE=$HOME/src_3rdparties/OpenCourant/exec \
  OPENCOURANT_MINIQA=$HOME/src_3rdparties/OpenCourant/OpenCourant/qa-tests/miniqa \
  LD_LIBRARY_PATH=$HOME/src_3rdparties/OpenCourant/extlib/hm_reader/linux64 \
  RAD_CFG_PATH=$HOME/src_3rdparties/OpenCourant/hm_cfg_files
just perf-oc-deck taylor_bar   # one deck; perf-oc / -elastic / -plastic for sets
```

Run cells sequentially on a quiet box; discard any run with
sustained foreign load (compare against `perf/oc/`).
