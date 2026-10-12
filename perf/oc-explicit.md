# OpenCourant explicit-dynamics comparison (Phase 6, 2026-10-11)

Sonium native explicit vs the OpenCourant engine on the frozen
4-deck subset, measured with `qa-oc-bench` (lane 6A harness,
`sonium.oc-bench/v1` bundles). Raw evidence: `perf/oc/*.json`
(33 cells). Every published row has `correctness_passed=true`;
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

## Equivalence basis (from 6A, frozen in PHASE6_CONTRACTS.md)

Same checked-in deck pair both arms (dof 675/297/297/1455
exact), same LAW1 E/nu/rho and stop time, same walls/BCs/loads.
Differs: engine LAW1-QEPH vs native Neo-Hookean Q1 bricks
(small-strain identical), engine contact vs native penalty
walls, own adaptive dt per side (cycle counts differ;
normalized by ns/dof/cycle). Deferred with measured reasons:
shells (no native explicit shells), plasticity (no native
explicit plasticity), cantilever/bending (native Q1 shear
locking).

## Results

`ns/dof/cycle = wall_s * 1e9 / (cycles * dof)`. Best in each
deck column is bold.

### taylor_bar (dof 675; engine 24 cycles, sonium 127 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 0.027 | **317** | 3 MB |
| sonium | native | 16 | 0.027 | 318 | 3 MB |
| sonium | native | 64 | 0.027 | 316 | 3 MB |
| engine | serial | 1 | 0.033 | 2034 | 38 MB |
| engine | smp | 1 | 0.033 | 2062 | 38 MB |
| engine | smp | 4 | 0.045 | 2787 | 44 MB |
| engine | smp | 16 | 0.086 | 5322 | 108 MB |
| engine | mpi | 2 | 0.754 | 46543 | 71 MB |
| engine | mpi | 4 | 0.802 | 49479 | 69 MB |

Engine numbers are setup-dominated (24 cycles in 33 ms; the
~40 ms in-binary setup confound documented by 6A). MPI pays
~0.7 s of launcher/rank overhead for 24 cycles.

### rail_crush (dof 297; engine 6898 cycles, sonium 81020 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 5.48 | **228** | 15 MB |
| sonium | native | 16 | 5.39 | 224 | 13 MB |
| engine | serial | 1 | 0.53 | 260 | 38 MB |
| engine | smp | 1 | 0.53 | 261 | 38 MB |
| engine | smp | 4 | 1.02 | 495 | 43 MB |
| engine | smp | 16 | 1.81 | 885 | 108 MB |
| engine | mpi | 2 | 1.33 | 648 | 71 MB |
| engine | mpi | 4 | 1.39 | 678 | 69 MB |

### necking_bar (dof 297; engine 1372839 cycles, sonium 1305578 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 81.9 | **211** | 3 MB |
| sonium | native | 1 | 85.4 | 220 | 3 MB |
| engine | serial | 1 | 97.7 | 240 | 38 MB |
| engine | smp | 1 | 97.9 | 240 | 38 MB |
| engine | mpi | 4 | 104.9 | 257 | 69 MB |
| engine | mpi | 2 | 108.6 | 266 | 71 MB |
| engine | smp | 4 | 182.0 | 446 | 42 MB |
| engine | smp | 16 | 316.4 | 776 | 108 MB |

### hertz_sphere (dof 1455; engine 137513 cycles, sonium 786523 cycles; 1 repeat)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | mpi | 4 | 18.1 | **90** | 69 MB |
| engine | mpi | 2 | 21.7 | 108 | 72 MB |
| engine | serial | 1 | 24.6 | 123 | 38 MB |
| engine | smp | 1 | 24.9 | 125 | 38 MB |
| engine | smp | 4 | 27.0 | 135 | 43 MB |
| engine | smp | 16 | 44.0 | 220 | 108 MB |
| sonium | native | 16 | 353.6 | 309 | 8 MB |
| sonium | native | 1 | 366.7 | 320 | 9 MB |

## Reading

- Per-dof-per-cycle, sonium wins 3 decks (taylor 6.4x, rail
  1.1x, necking 1.1x vs engine serial) and the engine wins
  hertz (2.6x). The hertz gap is the honest headline on the
  sonium side: 5.7x more cycles at ~2.6x the per-cycle cost —
  dt policy and contact cost, not a broken run (probes pass).
- Wall-clock tells the same story except where cycle counts
  diverge: sonium rail takes 5.4 s vs engine 0.5 s because it
  runs 11.7x more cycles (own adaptive dt) at a slightly
  cheaper per-cycle cost.
- Nobody scales on these decks: sonium t1 ~= t16 everywhere
  (1.00-1.04x; expected at 297-1455 dof), engine SMP is flat
  to sharply negative (-nt16 is 1.8-3.4x SLOWER than serial),
  MPI only pays on the largest deck (hertz r4 = 0.73x serial
  wall). These are tiny decks; scaling claims need bigger
  meshes (future work, needs native shells + plasticity first).
- Memory: sonium 3-15 MB vs engine 38-108 MB peak RSS per
  cell (MPI RSS covers the launcher only and under-reports
  ranks — 6A caveat, restated).
- Engine serial == smp1 on every deck (within 1%), as expected.

## Rerun

```sh
export TMPDIR=$HOME/src/all_of_sonium/tmp \
  SONIUM_QA_SCRATCH=$HOME/src/all_of_sonium/tmp \
  OPENCOURANT_EXE=$HOME/src_3rdparties/OpenCourant/exec
cargo build --release -p sonium-qa --features qa-binaries --bin qa-oc-bench
./target/release/qa-oc-bench --deck taylor_bar --side both \
  --threads 1 --emit-bundle /tmp/cell.json   # one cell; sweep decks/sides/flavors
```

Run cells sequentially on a quiet box; discard any run with
sustained foreign load (compare against `perf/oc/`).
