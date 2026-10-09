# Parallelism / solver / compute fix plan (generate-roomeq-data sweep audit)

Scope: what blocks or risks a 20-scenario x 20-2000 Hz sweep on 60 cores /
160 GB + RTX PRO 6000 Blackwell. Findings are ordered by priority. Each item
names the owning crate, the evidence, and how to verify the fix.

Status 2026-09-18: P1-P4 implemented and verified (see per-item notes).
P5 needs no code change; P6 are non-issues.
Status 2026-09-19: production GPU sweeps with `--gpu-precond auto` aborted in
CUDA AMG setup (singular f32 coarse operator at 20 Hz, zero data written).
Fixed with a regularized coarse retry + sweep-wide AMG-to-Chebyshev downgrade
(see P2b). New finding same day: f32-pure stalls just above the 1e-4 promoted
gate in the high band (res floor ~1.3e-4 at ~1586-1861 Hz, 2100+ outers, no
progress) — f64-outer IR completes the same band in <2 min. Per-frequency
f32→IR auto-escalation is future work; until then GPU-strict runs should use
`--gpu-precision f64-outer`, and `run_gpu.sh` CPU fallback covers the rest.
CPU full-grid (20-2000 Hz) convergence itself is still unverified — validation
running. Immediate workaround if ever needed: `--gpu-precond chebyshev-8`
(low/mid band only).
Status 2026-09-24: GPU high band (>~500 Hz) was not just slow but broken —
f64-outer IR burned its full 60-outer budget and stalled at residual ~3e-2
at 700 Hz (abort, zero data), and f32-pure ground 3000+ iterations per
frequency to barely pass the 1e-4 gate. Root cause: the CUDA path built its
AMG hierarchy with generic `AmgConfig::for_aggregation()` defaults
(`strong_threshold = 0.08`) while the CPU path uses the tuned
shifted-Laplacian setup (`for_parallel()` + Chebyshev + threshold 0.5), so
the high-k shifted operator got a poor coarsening and every f32 inner solve
ran to its cap without reaching 1e-3. Fixed with `gpu_amg_config()` (same
tuned setup as CPU) + IR inner budget 80→160
(`crates/sonium-sim-room/src/fem_runner/gpu.rs`): the same 700 Hz case now
converges to ~1e-6 (outer 79→32 with the deeper inner budget), low band
unchanged (4-6 outers), and the low-frequency AMG setup-regularization test
still passes. Guarded by
`fem_runner::tests::gpu_amg_config_uses_tuned_shifted_laplacian_setup`.
Follow-up same day: 1500 Hz on a 312k-node mesh converges to 9.6e-7, but
1600 Hz burned the 60-outer budget and stopped at 1.4e-6 against the 1e-6
gate (near-miss abort, zero data). Raised `GPU_IR_MAX_OUTER_ITERATIONS`
60→100: 1600 Hz now converges at 9.99e-7 using 67 outers, and
early-converging bands are unaffected (the loop still exits on tolerance).
Guarded by `fem_runner::tests::gpu_ir_budgets_cover_high_band_convergence`.
Cost note: high-k frequencies now take minutes each on large meshes (1600 Hz
needed 144 s on 312k nodes with the inner solve still running to its cap
every outer step), so the 20-2000 Hz production grid remains hours per
scenario — device-side Krylov recycling is the next structural lever.
Status 2026-09-24 (CPU): GCRO-DR subspace recycling is HARMFUL on the high
band and is now OFF in the sweep defaults (`--recycle-subspace-size 0` in
`common.sh`). Matched A/B, block-jacobi, 700-800 Hz: 5-point grid with
recycle-30 took ~8.2 min wall with a 236 s / 13,665-iter outlier at 773 Hz;
without recycling ~2.9 min, max 3000 iters, no outlier. MSO scenario 2-point:
freq-2 with recycling cost 2.2-2.4x freq-1 (21.8k/26.3k vs ~10k iters);
without, freq-2 matches freq-1 (9-10.8k). Mechanism: the deflated subspace
goes stale across %-level k-steps while adding per-iteration augmentation
overhead (~19% slower per iter measured). Low band unaffected either way
(ms-scale, tiny mesh). Numerically safe: same 1e-6 gate enforced by
true-residual verification + defect correction regardless of path — max drift
0.0006 dB SPL / 0.0054 deg over 1020 common rows. Revisit only with
production-step (2.3%) A/B evidence; a healthy recycler would also reopen
frequency-parallel sweep designs. CPU AMG was also measured (setup 9x cheaper
at 244 ms vs 2.2 s, but solve ~2x slower with borderline 1.00e-6 residuals),
so it stays out.
Status 2026-09-24 (CPU): sweep default preconditioner is now ORAS
(`--preconditioner oras` in `common.sh`), replacing block-jacobi. Matched
A/B at equal iteration counts: 24k mesh high band 16-18 s vs 40-44 s per
mode (setup 70 ms vs 2.4 s); 283k mesh at 2000 Hz 221/279 s vs 328/412 s per
mode (setup 0.26 s vs 4.7 s); full-domain 3-source scenario healthy at
3000-3300 iters/source with 92 ms setup. Outputs at parity (max 0.0002 dB /
0.0031 deg over 1020 rows; same 1e-6 gate + defect correction). HABC measures
identically on this path. This also collapses the "first case" setup cost
that dominated small-mesh starts (morning production logs showed 96-140 s
ILUT builds on large meshes). Negative result: `--gmres-restart 80` vs 50
with ORAS saves ~5% iterations but wall-clock is flat (denser Arnoldi
orthogonalization offsets it), so the restart-50 default stands.
Validation 2026-09-24: full-band 12-point CPU run, small_stereo_2_0 on the
production 283k-node mesh with the new defaults (oras, recycle 0) — all 24
mode/frequency solves converged under the 1e-6 gate, zero failures, ~71 min
wall on 32 threads. Iters (mode1/mode0): 20 Hz 169/550, 30 Hz 184/550,
46 Hz 240/900, 70 Hz 700/4800, 107 Hz 1300/1350, 162 Hz 7100/8250,
247 Hz 6700/8100, 375 Hz 4650/5500, 570 Hz 5000/4850, 866 Hz 4700/5350,
1316 Hz 4300/4450, 2000 Hz 4600/5900. Setup is ms everywhere (0.3 s first
build). Cost concentrates at isolated low-mid resonance poles (162-247 Hz,
7-8k iters) and the high-band grind (~5k); both converge via the
true-residual + defect-correction net. First case now 11-35 s vs 135-325 s
in the morning baseline.
Measured ping-pong stays negligible (matrix 0-2 ms, upload 0-4 ms vs
seconds of solve) thanks to the reusable CSR structure + device-side value
refresh — no transfer work needed. Remaining structural costs, not yet
addressed: no Krylov recycling across frequencies on the GPU path (CPU has
GCRO-DR(30); GPU has warm-start only), per-source serial solves, and one
2000 Hz mesh for all 200 frequencies (low band is ~100x oversolved).
CPU sweep tuning (`run_cpu.sh`, ILUT fill 20 setup cost) is deferred follow-up.

## Already healthy (no action)

- **Sweep CPU preconditioner is `oras`** (see status entry above): one-level
  optimized Schwarz with Robin transmission, ms-scale setup, parallel apply.
  `block-jacobi` (independent local ILUT solves) remains available but is no
  longer the sweep default. Global `ilut` apply is documented "mostly serial"
  and is correctly NOT the default.
- **Within-frequency sources solve in parallel** (`fem_runner/run.rs`, `par_iter`
  over sources); **frequencies stay sequential** to preserve GCRO-DR Krylov
  recycling across adjacent freqs. Do not "fix" this into per-frequency
  parallelism: it would cold-restart the recycler at every step.
- **Scenarios run sequentially in-binary** (deliberate: one scenario owns the
  whole rayon pool). Scale-out belongs at script level (`run_cpu.sh` does this).
- **CPU SpMV is rayon-parallel** (`sonium-compute/src/cpu/mod.rs`).
- **GPU host↔device traffic is negligible** (measured 2026-09-24: per
  frequency matrix 0-2 ms, upload 0-4 ms vs seconds of solve): reusable CSR
  structure uploaded once, per-frequency values refreshed by device-side
  linear combinations, only listener pressures + evidence cross back. No
  transfer optimization needed; the bottleneck is purely solver iterations.

## P1 - Lane-pool worker stack + thread cap (`sonium-sim-room`, `fem_runner/run.rs`)

Evidence: the candidate-scheduler lane pools (`ThreadPoolBuilder::new()
.num_threads(threads_per_lane).build()`, ~line 630) omit the 32 MiB worker
stack that `install_fem_rayon_pool` sets for the global pool because "nested
FEM assembly + GMRES + ILUT blow past the default ~2 MiB stack". Any full FEM
solve executing inside a lane pool risks stack overflow. Additionally
`threads_per_lane = available_parallelism() / lanes` ignores `--max-concurrent`
(and our 60-core cap on a 128-CPU box), oversubscribing the cgroup.
Fix: build lane pools through the same helper (32 MiB stack) with
`min(user_cap, available) / lanes` threads. Verify: 2-lane candidate run under
`--max-concurrent 4` + thread-count assertion test.
Implemented 2026-09-18: `FEM_WORKER_STACK_BYTES` shared const
(`sim-room/src/runtime.rs`); lane pools in
`run_fem_candidates_concurrent_with_commit` (`fem_runner/run.rs`) derive
`threads_per_lane` from the global pool width via `lane_thread_budget()` and
carry the FEM stack + `fem-lane-{lane}-{index}` thread names. Tests:
`lane_pool_tests::lane_budget_splits_pool_without_oversubscription` (budget
math incl. the `--max-concurrent 4` → 2-per-lane regression case) and
`concurrent_candidates_solve_inside_lane_pools` (two full tiny-room solves
through the concurrent path). `cargo test -p sonium-sim-room --lib`: 329
passed, 0 failed.

## P2 - CUDA FEM path is prototype-grade (`sonium-sim-room`, `fem_runner/consts.rs`)

Evidence: code comments say "Match the current CUDA prototype: GPU runners do
not yet handle ..."; Metal path has explicit `prototype` guards.
Validation 2026-09-18 on RTX PRO 6000 Blackwell (cc 12.0, driver 610.57):
`small_stereo_2_0`, 20-30 Hz / 3 pts, `--backend cuda --gpu-precision f32
--gpu-precond auto` solved end-to-end - actual backend `cuda`, `f32-pure`,
`AMG(P,Chebyshev smoother)`, 0 nonconverged, max residual 1.7e-6, full
CSV/WAV/config/metadata exports. Kernels (incl. shipped PTX) load on sm_120.
Against the CPU f64/block-jacobi reference the max deviation was 0.89 dB SPL
/ 0.017 deg phase (right source, ~58 dB SPL; 10 files x 3 freqs) - no in-tree
dB gate exists (`backend_parity.rs` checks topology/correlation only), so a
production parity gate (peak-SPL tolerance + cross-backend diff report) is
still the promotion criterion before trusting GPU sweeps outright. Until then
`run_gpu.sh` treats GPU results as provisional (validates outputs, keeps CPU
fallback). Implemented 2026-09-18: `cuda_fem_f32_smoke_matches_cpu_tiny_room`
in `fem_runner/tests.rs` (tiny room, F32Pure + AMG, asserts convergence and
<25% relative drift vs CPU; skips cleanly without a CUDA device) — passes on
Blackwell in 1.2 s. Remaining: (a) ensure the repo's CUDA CI job runs it;
(b) prefer JIT `.cu` over shipped PTX or ship multi-arch PTX.

## P2b - CUDA AMG setup failure aborts sweep with zero data (fixed 2026-09-19)

Symptom: every `--backend cuda --gpu-precond auto` scenario failed at
frequency 1/200 (20 Hz) with `CUDA AMG coarse LU factorization failed:
cusolverDnCgetrf: factorization failed at one-based pivot 12-15`, leaving
empty `fem/<scenario>/` dirs (this is why no data was found in data_tests).
Root cause: the runner builds the AMG hierarchy on the f32-demoted
shifted-Laplacian and factors the coarse grid with dense f32 LU; at low
frequencies the coarse Galerkin operator inherits the stiffness nullspace
and the factor hits an exact zero pivot. The error propagated with `?` out
of the per-frequency AMG arm, killing the scenario.
Root-cause fix (`sonium-solvers-preconditioners/src/amg/mod.rs`):
`build_cuda_c32_coarse_lu` retries a zero-pivot factor once with a 1e-6
relative diagonal shift (`COARSE_LU_DIAGONAL_REGULARIZATION`). A coarse
solve is only an approximate correction, so the perturbation is far below
AMG's own approximation error while lifting the inherited nullspace pivot.
Second-line defense (`fem_runner/consts.rs`, CUDA path only): if setup still
fails, the first failure arms a sticky sweep-wide downgrade to the
Chebyshev(8) smoother with a `log::warn!` and
`fallback_status: "amg-setup-fallback:chebyshev-8"` in diagnostics.
`P_inner` device CSR is now allocated upfront for Auto/AMG (one extra
A-sized structure) so the fallback has live values. Note: Chebyshev alone
does not converge the high band (stalls ~1.6e-4 vs the 1e-4 gate at
~1860 Hz), so the regularization — not the fallback — is what makes full
production grids possible; the fallback only converts other setup errors
into data+provenance instead of silence.
Tests: `cuda_low_frequency_sweep_completes_with_amg` uses the
production-faithful trigger (small_stereo_2_0 meshed for 100 Hz max, 20 Hz
first frequency) and asserts completion with fallback `none` (trips if the
regularization regresses) + finite pressures; preconditioners crate AMG
suite green (65 passed).
Verified: focused test green; full 20-2000 Hz / 200-pt small_stereo_2_0 GPU
sweep validation running (complete exports pending confirmation).
Not changed: explicit-solve error paths (device loss, non-convergence)
still abort loudly; launch-injection acceptance tests unaffected (they fail
before precond setup).

## P3 - Guard the known-stalling GPU preconditioner (`sonium-sim-room` CLI)

Evidence: `--gpu-precond point-jacobi` is documented to stall ~1e-2 residual on
indefinite Helmholtz, yet the CLI accepts it silently for any sweep.
Fix: warn (or refuse without `--i-know`) when `--backend cuda` combines
`point-jacobi` with a modal-band sweep; scripts already pin `auto` (-> CUDA AMG).
Implemented 2026-09-18 (warn variant, non-breaking): `is_point_jacobi_gpu_precond()`
in `generate_roomeq_data.rs` mirrors the `GpuPrecond::parse` aliases without
feature gates; `main()` logs a warn suggesting auto/poly-jacobi/chebyshev.
Unit tests in `cli_guard_tests`; verified live on the CUDA release binary.

## P4 - Serial-apply CPU preconditioners under a 60-core budget (`sonium-solvers*`)

`ilut` / `ilu-coloring` / `ilu-fixedpoint` triangular solves do not scale with
cores; they are legitimate algorithmically (strong convergence, low memory) but
misleading under `--max-concurrent 60`. Fix: emit a one-line warning suggesting
`block-jacobi`/`additive-schwarz`/`amg` when a serial-apply precond is combined
with >8 threads; add a `precond_scaling` CI check (the `bin/precond_scaling.rs`
harness exists) asserting apply speedup vs threads for the parallel family.
Implemented 2026-09-18 (warning half): `uses_serial_apply_preconditioner()` +
`effective_thread_count()` in `generate_roomeq_data.rs`; `main()` warns when a
serial-apply precond meets >8 effective threads. Unit tests in
`cli_guard_tests`; verified live (`--preconditioner ilut --max-concurrent 60`
warns, run completes). The `precond_scaling` CI gate remains future work.

## P5 - Silent-grid-change and stdin-hang hazards (CLI contract, no code change needed)

- `--auto-reduce-freq` silently shrinks the contracted 20-2000 Hz grid: scripts
  must never set it for production (they don't).
- `prompt_yes_no` on budget exceed hangs headless runners: scripts always pass
  explicit `--memory-limit-gb` (they do). Consider a future `--no-prompt` flag
  that fails fast instead.

## P6 - Non-issues, do not "fix"

- `--dense` BLR TBEM capped at 8 samples / single scenario: reference-check path
  by design, not a production route. No change.
- `metal` backend on this Linux box: irrelevant. No change.
- `block_source_solve` default off: correct with a full pool (parallel
  per-source path wins); only enable under frequency-lane parallelism.
