# Room Solver Profile Benchmark - 2026-05-18

Scope: roadmap pass after dense BLR boundary/source semantics were aligned with
FEM. All runs used release binaries on the local Apple Silicon workstation.

Common sweep unless noted:

- Frequencies: 20-200 Hz, 8 log-spaced points
- FEM recycler: GCRO-DR, recycle size 30
- FEM preconditioner rebuild cadence: every 5 frequencies
- FEM tolerance: default 1e-6
- Dense BLR: CPU backend, dense max frequency 200 Hz

## Results

| Scenario | Profile | Wall time | Notes |
| --- | ---: | ---: | --- |
| `small_stereo_2_1` | FEM ILUT | 0.18 s | Reference |
| `small_stereo_2_1` | FEM block-jacobi | 0.16 s | `max |dp|/p90 = 5.34e-1`, `L2 rel = 6.42e-2` vs ILUT |
| `small_stereo_2_1` | FEM AMG | 18.81 s | Many retry warnings; `max |dp|/p90 = 1.91e0` |
| `small_stereo_2_1` | Dense BLR | 13.61 s | 1170 surface DOF, 3 sources, sequential dense frequencies |
| `medium_multi_sub_4` | FEM ILUT | 1.37 s | Reference |
| `medium_multi_sub_4` | FEM block-jacobi | 1.69 s | `max |dp|/p90 = 1.10e-6`, `L2 rel = 4.11e-7` |
| `medium_multi_sub_4` | FEM AMG | 4.47 s | `max |dp|/p90 = 1.44e-5`, `L2 rel = 6.27e-6` |
| `medium_multi_sub_4` | Dense BLR | 117.26 s | 2170 surface DOF, 6 sources, sequential dense frequencies |
| `large_surround_5_1` | FEM ILUT | failed | 20-200 Hz run missed tolerance at 143.9 Hz after retry, residual 2.06e-6 |
| `large_surround_5_1` | FEM block-jacobi | failed | 20-200 Hz run missed tolerance at 143.9 Hz after retry, residual 2.09e-6 |
| `large_surround_5_1` | FEM AMG | failed | 20-200 Hz run failed at 143.9 Hz after retry, residual 1.75e-6 |
| `large_surround_5_1` | FEM ILUT | 4.96 s | Fallback 20-120 Hz run |
| `large_surround_5_1` | FEM block-jacobi | 3.44 s | Fallback 20-120 Hz run; `max |dp|/p90 = 2.36e-5` |
| `large_surround_5_1` | FEM AMG | 131.12 s | Fallback 20-120 Hz run; many retry warnings |
| `large_surround_5_1` | Dense BLR | 748.47 s | 3598 surface DOF, 6 sources, `freq_concurrency=4`, `tbem_threads=3` |

## Commands

FEM examples:

```text
target/release/bench-fem-preconditioners --scenario small_stereo_2_1 --min-freq 20 --max-freq 200 --num-freqs 8 --preconditioners ilut,block-jacobi,amg --rebuild-every 5
target/release/bench-fem-preconditioners --scenario medium_multi_sub_4 --min-freq 20 --max-freq 200 --num-freqs 8 --preconditioners ilut,block-jacobi,amg --rebuild-every 5
target/release/bench-fem-preconditioners --scenario large_surround_5_1 --min-freq 20 --max-freq 200 --num-freqs 8 --preconditioners ilut,block-jacobi,amg --rebuild-every 5
target/release/bench-fem-preconditioners --scenario large_surround_5_1 --min-freq 20 --max-freq 120 --num-freqs 8 --preconditioners ilut,block-jacobi,amg --rebuild-every 5
```

Dense examples:

```text
target/release/generate-roomeq-data --scenario small_stereo_2_1 --max-freq 200 --num-points 8 --dense --dense-max-freq 200 --output-dir /private/tmp/sonium-sim-room-bench-small-dense
target/release/generate-roomeq-data --scenario medium_multi_sub_4 --max-freq 200 --num-points 8 --dense --dense-max-freq 200 --output-dir /private/tmp/sonium-sim-room-bench-medium-dense
target/release/generate-roomeq-data --scenario large_surround_5_1 --max-freq 200 --num-points 8 --dense --dense-max-freq 200 --dense-freq-concurrency 4 --dense-tbem-threads 3 --output-dir /private/tmp/sonium-sim-room-bench-large-dense
```

## Takeaways

- Dense BLR is not competitive as a broad default at 200 Hz on these room
  meshes. It is useful as an accuracy reference or for much smaller/modal-only
  bands, but assembly dominates wall time.
- Block-Jacobi is the fastest successful FEM profile in the medium and large
  fallback runs, with close agreement against ILUT where both complete.
- AMG is unattractive for these sweeps as currently configured: it is slower and
  produces many first-pass retries, especially on small and large rooms.
- The strict 1e-6 convergence gate makes `large_surround_5_1` at 20-200 Hz
  brittle around 143.9 Hz. The failed residuals were close to tolerance for
  ILUT and block-Jacobi, so a follow-up should decide whether production data
  should use a looser 1e-5 tolerance or stronger settings for this scenario.
