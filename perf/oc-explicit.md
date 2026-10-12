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
penalty walls, own adaptive dt per side (cycle counts differ;
normalized by ns/dof/cycle). Native plastic runs with linear
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
### taylor_bar (dof 675; engine 24 cycles, sonium 127 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 64 | 0.027 | **316** | 3 MB |
| sonium | native | 1 | 0.0272 | 317 | 3 MB |
| sonium | native | 16 | 0.0273 | 318 | 3 MB |
| engine | serial | 1 | 0.033 | 2035 | 38 MB |
| engine | smp | 1 | 0.0334 | 2062 | 38 MB |
| engine | smp | 4 | 0.0451 | 2787 | 44 MB |
| engine | smp | 16 | 0.0862 | 5322 | 108 MB |
| engine | mpi | 2 | 0.754 | 46543 | 71 MB |
| engine | mpi | 4 | 0.802 | 49479 | 69 MB |
<!-- PERF-OC-DECK-END -->

Engine numbers are setup-dominated (24 cycles in 33 ms; the
~40 ms in-binary setup confound documented by 6A). MPI pays
~0.7 s of launcher/rank overhead for 24 cycles.

<!-- PERF-OC-DECK-BEGIN rail_crush -->
### rail_crush (dof 297; engine 6898 cycles, sonium 81020 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 5.39 | **224** | 13 MB |
| sonium | native | 1 | 5.48 | 228 | 15 MB |
| engine | serial | 1 | 0.534 | 261 | 38 MB |
| engine | smp | 1 | 0.534 | 261 | 38 MB |
| engine | smp | 4 | 1.01 | 495 | 43 MB |
| engine | mpi | 2 | 1.33 | 648 | 71 MB |
| engine | mpi | 4 | 1.39 | 678 | 69 MB |
| engine | smp | 16 | 1.81 | 885 | 108 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN necking_bar -->
### necking_bar (dof 297; engine 1372839 cycles, sonium 1305578 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 81.9 | **211** | 3 MB |
| sonium | native | 1 | 85.4 | 220 | 3 MB |
| engine | serial | 1 | 97.7 | 240 | 38 MB |
| engine | smp | 1 | 97.8 | 240 | 38 MB |
| engine | mpi | 4 | 105 | 257 | 69 MB |
| engine | mpi | 2 | 109 | 266 | 71 MB |
| engine | smp | 4 | 182 | 446 | 42 MB |
| engine | smp | 16 | 316 | 776 | 108 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN hertz_sphere -->
### hertz_sphere (dof 1455; engine 137513 cycles, sonium 786523 cycles; 1 repeat)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | mpi | 4 | 18.1 | **90** | 69 MB |
| engine | mpi | 2 | 21.7 | 108 | 72 MB |
| engine | serial | 1 | 24.6 | 123 | 38 MB |
| engine | smp | 1 | 24.9 | 125 | 38 MB |
| engine | smp | 4 | 27 | 135 | 43 MB |
| engine | smp | 16 | 44 | 220 | 109 MB |
| sonium | native | 16 | 354 | 309 | 8 MB |
| sonium | native | 1 | 367 | 320 | 9 MB |
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

- Per-dof-per-cycle, sonium wins 5 decks (elastic taylor
  6.4x, rail 1.1x, necking 1.1x; plastic taylor 1.5x,
  cylinder 1.1x vs engine serial) and the engine wins hertz
  (2.6x). The hertz gap is the honest headline on the sonium
  side: 5.7x more cycles at ~2.6x the per-cycle cost — dt
  policy and contact cost, not a broken run (probes pass).
- Wall-clock tells the same story except where cycle counts
  diverge: sonium rail takes 5.4 s vs engine 0.5 s because it
  runs 11.7x more cycles (own adaptive dt) at a slightly
  cheaper per-cycle cost. Plastic taylor flips the other way
  (engine 0.054 s vs sonium 0.084 s: 53 vs 123 cycles at a
  cheaper sonium per-cycle cost), while the cylinder runs
  fewer native cycles than engine (4704 vs 5188), so sonium
  wins wall there too (1.01 s vs 1.19 s).
- Nobody scales on these decks: sonium t1 ~= t16 everywhere
  (1.00-1.04x; expected at 297-2340 dof), engine SMP is flat
  to sharply negative (-nt16 is 1.5-3.4x SLOWER than serial),
  MPI only pays on the largest deck (hertz r4 = 0.73x serial
  wall). These are tiny decks; scaling claims need bigger
  meshes (future work, needs native shells first — plasticity
  has now landed).
- Memory: sonium 3-15 MB elastic vs engine 38-112 MB peak RSS
  per cell (MPI RSS covers the launcher only and
  under-reports ranks — 6A caveat, restated). Plastic sonium
  cells report 83 MB RSS for ~2k dof against 3 MB on the
  smaller elastic taylor: unexplained, flagged for follow-up
  (not investigated; numbers published as measured).
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
