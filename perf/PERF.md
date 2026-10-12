# Direct Solver Performance

_Generated 2026-08-24T02:15:52Z from commit `0f211568-dirty`._

## Platform

| Property | Value |
|---|---|
| OS | Darwin 25.6.0 Darwin Kernel Version 25.6.0: Fri Jul 31 19:17:26 PDT 2026; root:xnu-12377.161.14~5/RELEASE_ARM64_T6041 |
| Architecture | aarch64 |
| CPU | Apple M4 Pro |
| Memory | 64 GB |
| Logical CPUs | 14 |
| Rust | rustc 1.98.0 (88d9e12ae 2026-08-18) |

## Backend availability

| Backend | Status | Workload |
|---|---|---|
| Sonium blocked LU | available | dense |
| Apple Accelerate | available | dense |
| Sonium sparse simplicial | available | sparse |
| Sonium sparse supernodal | available | sparse |
| SuperLU | available (7.0.1) | sparse |
| CUDA | adapter pending | dense and sparse |
| ROCm | adapter pending | dense and sparse |

## Dense LU: factorization and solve

| N | Sonium blocked (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.291 | 0.019 | 15.64× | 1.26e-15 | 6.32e-16 | 1024 | 8192 | 17152 |
| 64 | 1.967 | 0.073 | 27.04× | 9.80e-16 | 1.09e-15 | 4096 | 32768 | 67072 |

## Sparse LU: 2D nonsymmetric convection-diffusion

| Grid | N | NNZ | Sonium simplicial (ms) | Sonium supernodal (ms) | Supernodal reused symbolic (ms) | SuperLU (ms) | Simplicial / SuperLU | Supernodal / SuperLU | Simplicial factor nnz | Supernodal factor nnz | Reused factor nnz | Simplicial factor bytes | Supernodal factor bytes | Reused factor bytes | Modeled peak bytes | Max residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8×8 | 64 | 288 | 0.967 | 0.959 | 0.599 | 0.034 | 28.69× | 28.44× | 848 | 848 | 857 | 6784 | 6784 | 6856 | 13520 | 3.14e-15 |
| 12×12 | 144 | 672 | 3.842 | 13.598 | 2.452 | 0.085 | 45.40× | 160.68× | 2564 | 2564 | 2585 | 20512 | 20512 | 20680 | 36048 | 1.87e-15 |

## Methodology

- Times are medians of 2 measured runs after one warm-up run.
- Each measured operation includes one factorization and one right-hand-side solve.
- The reused-symbolic column excludes Sonium ordering and symbolic analysis; all other sparse columns are one-shot.
- Factor nnz is the stored numeric factor count; factor bytes use scalar width and exclude allocator overhead.
- Modeled peak bytes include the input matrix, largest reported factor, and three scalar work vectors; this is an explicit lower-bound model, not OS RSS.
- Every backend must pass a relative infinity-norm residual gate of `1e-10` before timing.
- Ratios above 1.0 mean Sonium is slower than the platform reference; below 1.0 means faster.
- Regenerate with `scripts/direct_solver_perf.sh`. Override sizes with `DIRECT_PERF_DENSE_SIZES`, `DIRECT_PERF_SPARSE_GRIDS`, and `DIRECT_PERF_REPEATS`.

## x86 vendor refresh (2026-09-24, worktree `direct-x86-opt` @ `5f137e25` + uncommitted)

Platform: AMD Ryzen Threadripper PRO 3995WX (128 logical CPUs), Linux x86_64,
rustc 1.98.1. Vendor refs: OpenBLAS (pthread) LAPACK, SuperLU 6 (dgssv),
UMFPACK, CHOLMOD — all freshly rebuilt alongside. Sonium built from the
worktree. Harnesses in `/tmp/dense_prof`, `/tmp/sparse_baseline`,
`/tmp/chol_baseline` (+ `/tmp/sparse_ref`, `/tmp/chol_ref` C binaries).

### Headline gaps (best-vs-best; vendor threads tuned, not oversubscribed)

| Workload | Size | Sonium | Vendor best | Gap | Single-thread gap |
|---|---|---|---|---|---|
| Dense LU factor | n=2048 | ~164 ms, 1T (35 GF); ~66-77 ms @16T (same-count sweep) | dgetrf 122 ms @1T (47 GF); 24 ms @16T | 1.34x at 1T; ~2.7-2.9x best-vs-best | 1.34x |
| Dense LU | n=1024 | 23.1 ms | dgetrf 17.9 ms (default threads) | 1.29x | — |
| Dense Cholesky | n=2048 | ~165 ms, 1T (17.5 GF) | dpotrf 61 ms @1T (47 GF); ~12.5 ms @16T | 2.7x at 1T; ~4.1x vs best | 2.7x |
| Sparse LU, conv-diff | n=16384 | supernodal 315 ms (w/ analyze) | SuperLU dgssv 58.4 ms | 5.4x | — |
| Sparse LU, conv-diff | n=16384 | simplicial 410 ms | UMFPACK 47.8 ms total | 8.6x / 6.6x (supernodal) | — |
| Sparse Cholesky, 5pt Laplacian | n=16384 | supernodal ~51 ms (analyze+factor) | CHOLMOD 25.6 ms total | 2.0x | — |
| Sparse Cholesky numeric only | n=16384 | supernodal reused ~16.5 ms | CHOLMOD factor 21.1 ms | 0.77x (sonium faster) | — |
| Sparse Cholesky analyze | n=16384 | ~32 ms ordering+symbolic (perm ~22) | CHOLMOD analyze 4.1 ms | ~8x | — |

Fill (sparse Cholesky, ND): sonium L_nnz=501,955 vs CHOLMOD 381,322 (+31%
fill yet faster numeric — ordering quality, not kernel speed, is the gap).

### Reading (corrected 2026-09-24: earlier "1.02x/1.55x single-thread" claims
compared default-threaded sonium against 1-thread vendor — wrong basis)

- Methodology fix: the dense harness timed factor+solve+clone against
  factor-only dgetrf. Now factor-only: LU factor 183 ms @1T vs dgetrf
  122.5 ms @1T → true kernel gap 1.49x (was 1.83x muddled). Thin-k B-tile
  packing (k ≤ 64, n > 256) added to the AVX2 f64 gate: band shapes went
  9-27 GF direct to 31-40 GF packed; end-to-end LU @1T 224→211 ms (the
  in-situ Schur was already mixed packed/direct, muting the win).
- Dense LU solve is scalar O(n²) at ~14 ms for n=2048 (tolerance hoist
  saved ~5 ms from 19 ms); vendor dtrsm territory is <1 ms — wire a
  triangular-solve kernel next for the solve path.
- Remaining dense gaps are BOTH kernel and threading: LU kernel 1.44x,
  threaded-best 2.9x (70 ms @16T vs 24 ms @16T, same-count sweep);
  Chol kernel ~3.3x, threaded 8.2x. Phase split @16T (env-timer probe,
  reverted): panel GETF2 ~21 ms + U12 TRSM ~6 ms + pack ~5 ms serial
  floor vs Schur ~30 ms (5.2x scaling — matches dgetrf's overall rate).
  Landed: blocked U12 TRSM (16-row diagonal micro-blocks + `schur_sub`
  cross-updates; trsm phase -21% @1T, -30% @16T) and thread-dependent
  blocking (k=64 single-threaded where the thin-k kernel runs ~41 vs ~35
  GF, k=32 threaded where the serial panel floor dominates; 1T gap
  1.46 -> ~1.40 paired-verified, 16T unchanged). Reverted as no-win:
  parallel L21 pack (rayon dispatch ≥ gather savings), pack-first reorder,
  4-row panel unroll (probe +53%, end-to-end gap +4% — probes mislead),
  and forced k=64 at 16T (serial panel doubling). Method: gap ratio
  (sonium/vendor) self-normalizes frequency drift; interleave binaries
  A/B under load; distrust single samples (±6% box noise). Next dense
  targets in order: (1) Schur kernel trails vendor at k≥64 (41 vs 48 GF)
  + L3 blocking to lift the 5.2x parallel scaling toward 8x.
  Landed: serial SYRK banding for Cholesky (`schur_update_lower` floors at
  4 bands even single-threaded — one band computed the FULL block at 2x
  SYRK flops; 1T 200 -> ~165 ms, gap 3.3 -> 2.7x). Analyzed out: Cholesky
  k=64 at 1T (scalar per-row TRSM quadruples, erasing the kernel gain —
  unlike LU whose TRSM is already blocked BLAS-3).
  Landed: blocked Cholesky TRSM mirror (column-split micro-blocks +
  `schur_sub` cross-update, 1T path only; Hermitian needs A-plain/B-conj
  packs — A-conj double-counts at O(1e-3), caught by the new
  `trsm_blocked_serial_matches_loop` agreement test). Paired-interleaved
  verdict: +2.5 ms consistently (5/6 matched pairs), below the 10 ms
  estimate — the TRSM share was overestimated. Kept: same proven pattern
  as the LU win, guarded by a conj-sensitive test.
  Phase profile @1T (env timers, reverted): panel 0.4 ms + TRSM ~25 ms +
  SYRK ~107 ms — model closed, no missing time (earlier n³/6 flop-count
  confusion corrected: SYRK does 2.86 GF at ~27 GF). Remaining Cholesky
  gap is kernel efficiency on banded-SYRK shapes (27 vs ~46 GF effective),
  shared with the LU k≥64 ceiling work; k=64 for Cholesky stays analyzed
  out (scalar TRSM quadruples).
  Landed: F64_NC 256 -> 128. The packed B tile (kc×nc) at 256 KiB thrashed
  L2 past k=64 (31-35 GF at k=128-512); 128 KiB holds ~42 GF flat through
  k=512, +6% at k=32, parity at k=64. NC=64 regresses k≤64 (extra C
  re-reads). One-const change, bit-exact (agree=0e0). k=128 LU panels
  stay analyzed out (panel/TRSM growth outweighs 42 GF Schur).
  Landed: recursive panel GETF2 (LAPACK dgetrf2 shape — factor left half,
  serial middle TRSM, one BLAS-3 `schur_sub` bottom update, factor right
  half; pivot sequence unchanged, ulp-only update differences). Base-width
  sweep at n=2048: flat 1.40, base-16 1.38, base-8 1.34, base-4 1.35
  (too thin for the kernel); base-8 locked. 1T gap 1.40 -> 1.34, 16T best
  2.89 -> 2.72. Falsified: skipping B-tile packing for
  already-contiguous B (1T gap 1.40 -> 1.63, reverted) — the tile's value
  is stride reduction (n -> nc), not contiguity; wide direct rows collide
  in L1/L2 sets. Also falsified: SHARE one packed B across bands via a
  packed-B kernel entry (built end-to-end: tile pack/consume in avx2 +
  trait pair + banded_schur branch, bit-exact parity tests; measured
  neutral @1T and @16T, fully reverted) — L3-resident copies run ~free,
  so the per-band repack costs ~nothing; only the stride-shaping matters,
  which per-band packing already provides.
- Sparse LU remains structural (supernodal scalar panel phases, 5.4x).
- Sparse Cholesky numeric already beats CHOLMOD; analyze/ordering is the gap.
- Sparse analyze split (env timers, reverted): ND perm ~22 ms (was ~26;
  coarsen-aggregation global sort -> per-group counting-sort grouping,
  bit-identical output, fill 501,955 unchanged) + assem 1.7 + etree 0.2 +
  symbolic ~3 + level sets ~4. Falsified: FM passes 8->2 (early-exit, no
  change) and BFS bisection below 256/64 vertices (faster but +12%/+3%
  fill — rejected; fill-neutrality rules ordering changes).
  Sparse-LU numeric phase profile @n=16384 (env timers, reverted;
  steady-state, ~9 ms timer overhead): scatter 1.7 + reach 2.6 + apply
  42.1 (TRSM 12.9 via vec_axpy + GEMM/scatter 23.7 via AVX2 dgemm) +
  gather 3.0 + pivot 5.4 + swap/misc 0.5 + partition 7.4 + U-push 9.0 +
  mult ~8 + rank-update 8.7 + pack 11.0 + clear 1.0. Column loop total
  ~40 ms; apply + pack dominate.
  Falsified: per-row `T::vec_axpy` rank update (9.3 vs 8.6 ms t_rank,
  same residual class, reverted with NOTE) — trailing segments are
  short (<=15) so dispatch/call overhead dominates, and the loop is
  memory-bound (L2-streaming row traffic) where SIMD adds no rate; the
  scalar loop already touches each updated element exactly once.
  Landed: fused pivot-search/partition scan (one full column pass
  records nonzero blocks + max-track, classified post-swap; same pairs,
  same order, bit-identical factors, residual 3.63e-14 unchanged).
  pivot+part 12.8 -> 11.8 ms, numeric 116.2 -> ~115.8 ms (5-rep).
  Removed abandoned colamd TEMP timer decls (were unused warnings).
  Single-thread verdict: loop is traffic-optimal in rank/upush/mult;
  remaining 2x to vendor factor (~55 ms) needs etree level-scheduled
  threading, not more micros.
  Fresh vendor A/B (sequential SuperLU 6.0.1, COLAMD, OpenBLAS 1T,
  /tmp/vslu2/superlu_bench.c on the IDENTICAL 128x128 nonsymmetric
  Laplacian, 3-rep means): SuperLU colperm 4.5 ms + fact 38.8 ms =
  43.3 ms, residual 3.41e-14. Ours (5-rep): analyze 21.3 ms + numeric
  115.9 ms = 137 ms, residual 3.63e-14 (same class). Gaps: analyze
  4.7x, factor 3.0x, total 3.2x.
  Analyze split (env timers, reverted): colamd 15-17 ms + gather_cols
  1.9 + etree 0.5 + panels 0.1. COLAMD is 80% of analyze and 3.5x
  slower than SuperLU's ENTIRE ordering pipeline.
  Scope decision: single-thread numeric is traffic-optimal in its hot
  loops (rank/upush/mult touch each element once; axpy/GEMM fusion
  falsified or marginal). The 3x factor gap needs (a) bigger
  supernodes + BLAS-3-shaped updates like SuperLU, and/or (b) etree
  level-scheduled threading for wall clock (128 cores on this box).
  Threading the panel code is blocked on shared row_perm pivoting
  state (flat fan-out proven wrong: residual ~2.9); needs either
  ordered-panel/update-parallel scheduling or subtree-privatized
  workspaces. Multifrontal path may be the better parallelism vehicle.
  Next: attack COLAMD internals (4.7x gap, pure overhead) and/or
  prototype level-scheduled threading.
  Landed: COLAMD allocation + bounds-check pass (15.0 -> 9.7 ms timed,
  analyze 21.3 -> 14.1 ms clean-tree, residual 3.63e-14 bit-stable,
  fill tests green). Three pieces, all bit-identical output: (a)
  exact-capacity incidence build (row nnz + column histogram; kills
  ~65k small-vec reallocs, inc 2.5 -> 1.2 ms); (b) move-instead-of-
  filter-clone quotient init when no dense rows/dead members (qinit
  3.0 -> 0.1 ms); (c) unchecked element-id loads in scan 1 / scan 2a /
  Lp formation under a documented in-bounds invariant (rows < n,
  generated ids pushed with matching entries; -0.8 ms). Lp-unchecked
  alone was noise-level (memory-bound). Remaining colamd ~9.7 ms is
  quotient-core rate (~2x C now, was 4x). Total gap to sequential
  SuperLU: 3.2x -> 3.0x (130 vs 43.3 ms).
  Rate diagnosis (same-matrix flop census): SuperLU factor nnz
  1,315,428 at 1.218e8 flops / 39.3 ms = 3.1 GFLOPS effective. Ours:
  factor nnz 1,269,156 (BETTER fill, -3.5%) but chunk-GEMM flops alone
  134M over 26,778 calls (81% of calls < 4096 flops yet only 6% of
  flop volume) — i.e. ~1.4x flop excess at ~half the rate. Both the
  excess and the rate must move.
  Landed: pack stamp-dedup (sort_unstable+dedup+slot pass ->
  generation-stamped first-seen collection with inline slot
  assignment; order-free for GEMM+scatter, chunk order unchanged =>
  bit-identical: factor_nnz 1269156 and residual 3.63e-14 unchanged to
  the digit). Numeric 116 -> ~111 ms (5-rep clean-tree). Total gap to
  sequential SuperLU: 3.0x -> 2.9x (125 vs 43.3 ms).
  Flop census (TEMP, reverted): ours does 293M flops vs SuperLU 122M
  (chunk 134M + TRSM 8.9M + rank 3.6M iters, x2). Update rectangles
  46% dense; L packing 95% dense but solved blocks 33% dense.
  Falsified: panels=strict-supernodes (density 0.46 -> 0.49 only,
  111 -> 140 ms panel overhead) — padding is element-scattered, not
  target-column-driven. Falsified: live-row gather over solved blocks
  (134M -> 130M flops only; zeros are element-scattered, not
  row-structured; gather + smaller GEMMs cost ~9ms net). Falsified:
  inline TRSM axpy (same-binary interleaved A/B 111.7 vs 111.5 ms —
  neutral; TRSM streams from L2/L3 so kernel choice is irrelevant;
  kept vec_axpy/FMA). Lesson: only exact-pattern (Gustavson-style)
  updates could skip the 46%-dense rectangles; rate, not skips, is the
  remaining lever on this code shape.
  Landed: single-pass column tail (classify + U-push + mult + rank fused
  into one pass over the recorded nonzero blocks; upper/lower pair vecs
  removed). Order- and value-identical by construction (U sentinel first
  then block order; L pushes, rank FMAs, mark sequence, error stages and
  indices all preserved): factor_nnz 1269156 and residual 3.63e-14
  unchanged to the digit, 245 lib + contract/checked suites green.
  Time-neutral to slightly positive (numeric ~110 -> ~108.5-109.5 ms;
  within machine noise, kept for simplicity: 2 fewer hoisted buffers,
  3 fewer passes). Column-loop passes are L1-cheap for most panels, so
  pass fusion cannot move the needle — the loop is traffic-inherent.
  Considered and skipped: update_buf zero-fill removal (1-2 ms) — needs
  a cross-backend beta=0-never-reads-C invariant (scalar honors it;
  not verified for every arch kernel) with silent-UB risk on future
  kernel changes; not worth it.
  Vendor A/B 2: UMFPACK 1T (SuiteSparse 7.6.1, /tmp/vslu2/umfpack_bench.c
  on the IDENTICAL matrix, 3-rep): symbolic 9.2 ms (COLAMD+etree),
  numeric 23.1 ms, solve 2.9 ms, residual 2.26e-14, UMFPACK-reported
  flops 57.3M, numeric size 886k doubles. 8 threads: numeric 25.7 ms
  (no gain at n=16k; latency-bound, same as our threading story).
  Decisive: UMFPACK effective rate 57.3M/23.1 ms = 2.48 Gflops is
  IDENTICAL to ours (293M/~109 ms = 2.69 Gflops). The ENTIRE remaining
  gap is flop COUNT, not kernel rate — micros are done on this shape.
  Flop ladder (same matrix): UMFPACK frontal 57.3M < SuperLU supernodal
  122M (2.1x true) < ours bounding-box 293M (5.1x true, 2.4x SuperLU).
  Sub-step (a) exact-pattern updates: 293M -> ~122M (~45-50 ms).
  Sub-step (b) frontal/amalgamation fill quality: ~122M -> ~60-80M
  (~25-30 ms, UMFPACK parity). GPU-direct-LU needs the SAME
  bulk-batchable precondition (frontal batches, not 27k tiny launches
  at ~5-10 us each); deferred until (a)/(b) land. Dense-GPU API +
  crossover bench already exist (cuda feature, live-tested).
  Falsified: exact-pattern (Gustavson) chunk apply, reverted 2026-09-25.
  TWO fused variants implemented, measured, and reverted (code back to the
  dense GEMM+scatter baseline: 109.7 ms, residual 3.63e-14, factor_nnz
  1269156, all suites green): (a) k-major axpy — per nonzero solved entry,
  axpy the L column into permuted work rows: 314.5 ms; (b) slot-major dots
  — per (below-row, target) pair, dot the dense L segment against nonzero
  solved entries with one work RMW per pair: 236.5 ms. Both kept factor_nnz
  identical with residuals in the same class (3.62e-14), so the pattern math
  was right and the density regime is wrong: at 46%-dense rectangles the
  software zero-skip branch costs more per element than the vectorized FMA
  it saves, and fused permuted writes lose the GEMM's contiguous streams.
  Exact-pattern apply only wins at low density (<10-20%). The rectangles
  must get FEWER (sub-step (b): fill quality / frontal structure), not
  cheaper. See the NOTE on `compute_chunk_update` for the numbers.
  Ordering A/B (/tmp/sluord, same matrix, 5-rep): COLAMD analyze 12.5 +
  numeric 111.0 ms, nnz 1269156, res 3.63e-14; AMD analyze 127.7 +
  numeric 248.5 ms, nnz 834626 (best fill, ~2.2x slower numeric — deep
  elimination defeats the panel kernels), res 3.35e-14; ND analyze 22.0 +
  numeric 88.6 ms, nnz 1003910, res 4.03e-14 (same class). ND numeric is
  1.25x COLAMD and total 110.6 vs 123.5 ms — the time ∝ fill linearity
  holds inside our code (1.26x fill -> 1.25x time), yet UMFPACK does 886k
  in 23.1 ms, so execution STRUCTURE still dominates across vendors (~4x
  per-nnz cost vs frontal). ND is reachable today via Ordering::
  NestedDissection through the unified API (analyze_sparse routes it to
  analyze_supernodal_with_column_perm; only the equilibrating path rejects
  AMD/ND, explicitly). The stale "~2.5x" claim on `nd_ordering` was
  corrected to the measured 1.25x. Default stays COLAMD (documented
  contract; ND ordering itself costs ~18.6 ms).
  Panel-width sweep (COLAMD, 5-rep): 8 -> 120.4, 16 -> 111.0, 24 -> 119.5
  ms (32 worse per history); fill identical (1269156). 16 is the optimum:
  narrower multiplies panel-application overhead, wider fattens the
  bounding boxes. Kept 16.
  Multifrontal exact LU exists in-tree (multifrontal_blr, epsilon=0.0) and
  was benchmarked (/tmp/sluord `mf` cases, 8 threads): amalgamation 1024
  -> ~570 ms (big-front O(F^3) dense algebra dominates), default 256 ->
  ~191 ms, amal 0 (one front per dissection node, 255 fronts, 8 levels)
  -> factorize ~112 + solve 4.1 ms, residual 4.72e-14. So MF-amal0 ties
  supernodal-ND end-to-end (~116 vs ~117 ms) with a vendor-class solve
  (4.1 vs 2.9 ms) — the frontal structure does NOT cut work 10x here (same
  fill, same flops, same rate); UMFPACK's 57M-flop gap remains a fill +
  execution-structure problem, not a vehicle problem. MF needs a
  symbolic/numeric split for the reuse story (factorize re-runs dissection
  every call) and default-thread robustness (128 threads -> ~106 ms factor
  from rayon contention on tiny fronts; 1 thread -> ~315 ms). CORRECTION
  (2026-09-25): an earlier mislabeled-printf reversal briefly reported MF
  factor at ~3.5 ms and a 20 ms solve — both artifacts of swapped printf
  args in the scratch harness (fixed in /tmp/sluord/src/main.rs). The
  independent in-library timers agreed with the corrected 4.1 ms solve all
  along: always cross-check a new instrument against a second oracle
  before believing it. TEMP MF_SOLVE_PROFILE timers added for the
  bisection were fully reverted (solve.rs, mod.rs); 370 tests green.
  Full gates (2026-09-25): workspace `cargo check` clean, 0 clippy errors
  across direct/compute/sparse/core, 613 solver tests passed (direct +
  sparse + core, 0 failed), 72 CUDA suites green (parity 67 + device
  preconditioner 5) on hardware.
  Ordering-vs-execution 2x2 via UMFPACK_ORDERING_GIVEN (/tmp/vslu2,
  `auto|amd|given` modes + /tmp/sluord `dumpnd|dumpamd`, perms saved):
  UMFPACK on OUR ND = 290M flops / 121.3 ms (vs our own 88.6 ms — our
  kernels BEAT vendor execution on identical ordering); UMFPACK on OUR
  AMD (834k fill) = 123M / 41.9 ms (vs our 225 ms on the same ordering —
  our left-looking re-application explodes on deep chains, 5.4x execution
  gap); UMFPACK default = 57M / 23.3 ms with ordering_used=0 (CHOLMOD =
  METIS nested dissection). ROADMAP INVERSION: our ND partitioner is the
  #1 gap (290M vs 57M identical-execution flop ratio), not our kernels —
  our ND makes giant fronts (METIS-quality separators needed: FM
  refinement / better coarsening in partition.rs). Measurable objective:
  UMFPACK-GIVEN flops on our ND perm, 290M -> ~57M. Secondary: execution
  on deep-chain (AMD) structures.
  Falsified: separator-trimming (reverted 2026-09-25, NOTE in
  extract_separator). Stripping each boundary to minimality changed
  nothing: a TEMP census showed every separator already at its cut
  minimum (root 128, halves 64 on the 128 grid), UMFPACK-GIVEN stayed at
  290M, our numeric/fill unchanged. The 5x METIS gap is tree-shape /
  cross-level amalgamation, not cut fatness — per-cut minima are
  necessary but not sufficient. Revert verified: 0 check warnings, 22
  partition tests green, ND fill identical (1003910).
  METIS isolation via CHOLMOD (/tmp/vslu2/metis_perm.c + `metisperm`
  harness mode, perms saved): UMFPACK-GIVEN on raw METIS_NodeND perm =
  104M flops / 42.0 ms (NOT 57M — the default's extra 2x comes from the
  CCOLAMD-constraints finish, per the cholmod docs); OUR numeric on the
  same METIS perm = 93.5 ms / 1.14M nnz / res 3.78e-14 (1997 supernodes).
  So on METIS's many-small-front tree our execution LOSES 2.2x (93.5 vs
  42 — per-application overhead dominates), while on our wide tree it
  WINS (88.6 vs 121). Flop ladder: METIS+CCOLAMD 57M < METIS-tree 104M
  < our-AMD 123M < our-ND 290M. Parity recipe, both halves measurable:
  (a) METIS-quality tree + constrained-COLAMD finish (-> 57M flops);
  (b) frontal-rate execution on many-small-front trees (per-application
  overhead is the tax). Either half alone leaves ~2x on the table.
  REFINED against half-(b) as an independent lever (TEMP SNO_PHASE
  timers, fully reverted): colloop is the largest phase (ND 29 / METIS
  37 / COLAMD 32 ms) but scales with flop volume, NOT panel count (1064
  vs 1997 vs 5724 panels) — apply (28-40 ms) and colloop alike are
  traffic-inherent at the same rate. No per-call overhead anomaly
  exists; half-(b) alone cannot move the needle. Execution is uniformly
  optimal for its flop count — ONLY the tree (half-a) decides the time.
  Revert verified: 370 tests green, ND baseline intact.
  Falsified: constrained-COLAMD finish on OUR tree (/tmp/vslu2/
  ccolamd_probe.c via csymamd on A+A' with our postorder node ids as
  constraint sets, /tmp/sluord `dumptree` mode): UMFPACK-GIVEN = 305M
  flops, no better than unconstrained 290M. The finish is not an
  independent lever — CHOLMOD's 57M needs the METIS tree underneath.
  Structural read: our pipeline (edge-cut FM -> boundary-as-separator)
  vs METIS (direct separator refinement). Equal per-cut sizes, but our
  separators nest poorly (rough coupling across subtrees); the fix is
  separator-direct FM, a multi-day build. Oracle chain complete:
  {ourND 290M, ourND+CC 305M, ourAMD 123M, METIStree 104M,
  METIS+CC 57M}.
  LANDED: balance fraction 0.55 -> 0.58 (partition.rs MAX_PART_FRACTION,
  oracle-guided sweep 0.50/0.55/0.57/0.58/0.59/0.60/0.65 -> UMFPACK-GIVEN
  flops 603M/290M/238M/195M/228M/230M/338M). Sharp minimum at 0.58:
  oracle -33% (290M -> 195M), our ND numeric 128² 88.6 -> 81.9 ms
  (-8%, fill flat at 1.004M, res 3.86e-14), 256² 619 -> 529 ms (-15%,
  res 5.16e-14). Tighter hurts catastrophically (0.50: 603M, fill
  1.35M); looser degrades past 0.60. Verified: sparse 192 + direct 370
  tests green (MF included), 0 clippy errors. Tree-shape still dominates
  the remaining gap (195M vs 57M), but the floor moved.
  LANDED: ND leaf 32 -> 16 (sparse_lu.rs ND_MIN_BLOCK_SIZE; sweep
  16/32/64 at fixed 0.58 balance -> oracle 187M/195M/180M, our fill
  948k/1004k/1112k, our numeric 82.3/81.9/95.6 ms). Leaf 16 dominates 32
  deterministically (fewer flops AND 5.5% less fill, neutral time);
  leaf 64 wins oracle-only while our fill-linear execution loses 17% —
  rejected (optimizes for an execution we don't have). Holds at 256²:
  fill 4.945M -> 4.726M (-4.4%), numeric 529 -> 526 ms. Verified:
  sparse + direct 562 green, 0 clippy errors.
  Knob landscape exhausted (all reverted, oracle-read): FM passes 2 vs 8
  -> 197M vs 187M (converges early; early-exit makes 8 free — kept 8);
  coarsen target 100/200/400 -> 404M/187M/290M (U-shape, 200 optimal).
  Remaining partitioner work is algorithmic (matching quality,
  separator-direct refinement), not parametric.
  Campaign closure: coarsening already uses sorted-greedy heavy-edge
  matching with degree tie-breaks (METIS-style; inspected, no weaker
  scheme to upgrade). Final partitioner state: balance 0.58 + leaf 16
  landed, everything else at proven optima, sparse 192 green. Oracle
  floor 187M vs 57M with all parametric axes resolved — the rest is
  separator-direct refinement (multi-day) or accepting the tree.
  MF under new consts roughly flat (amal0 factor+solve ~128 vs ~112 ms
  before, within run variance; residual 4.56e-14) — overlap tax still
  dominates, no vehicle change.
  RECOMMENDATION (decision for maintainer, not implemented): make ND
  the default supernodal ordering. Totals at current consts: ND analyze
  ~26 + numeric ~82 = ~108 ms vs COLAMD ~12.5 + ~111 = ~124 ms, and the
  gap widens under symbolic reuse (numeric dominates). Residuals same
  class (3.9e-14 vs 3.6e-14). Cost: documented-contract change (lib.rs
  advertises COLAMD; Ordering::Auto currently resolves to COLAMD for
  LU) plus slower analyze on tiny matrices where ND overhead dominates
  — gate any switch on n (e.g. ND only above ~2000 DOFs) or leave Auto
  mapped and document ND as the PDE-grid recommendation (already the
  nd_ordering doc stance).
  SNAPSHOT (2026-09-25, current consts; [analyze]+numeric, residuals
  all 1e-14 class) — 128²: COLAMD [13]+107 / ND [31]+86, nnz 949k /
  AMD [129]+245 / MF-amal0 ~128 total / UMFPACK [9]+23 (57M flops).
  256²: ND [142]+526, nnz 4.73M / UMFPACK [54]+111 (519M). Gaps: 3.7x
  at 128² (86 vs 23), 4.7x at 256² (526 vs 111). GPU co-tenant still
  active — clean-room IR numbers pending.
  RETRACTED 2026-09-25 (strategy-confounded oracle — see correction
  below): the "EXECUTION PARITY PROVEN ... ENTIRELY its METIS ordering"
  claim compared OUR symmetric-direct numeric against UMFPACK-GIVEN,
  which forces UNSYMMETRIC strategy (UMFPACK refines a GIVEN Q under
  unsymmetric; qsymbolic line ~853: Quser + AUTO strategy ->
  UNSYMMETRIC). The 187M/113ms vs 82-89ms A/B measured different
  strategies, not head-to-head execution. True same-Q comparison was
  never available (UMFPACK never factors a GIVEN Q unrefined).
  CORRECTION (2026-09-25, /tmp/sluord `permnum` = OUR numeric on
  explicit perms, symmetric-direct semantics, all residuals 1e-14):
  128² ours-ms / ours-nnz — ND 85.3 / 948734; grid-bisect 84.8 /
  948884; sep_AMD 83.1 / 948566; leaf_AMD 85.7 / 926634; COLAMD 106.9 /
  1269156; METIS-file-Q 93.7 / 1140224; OUR-AMD 234.7 / 834626;
  REF-AMD (SuiteSparse amd_order) 79.9 / 762644. Under OUR engine our
  ND already beats METIS-file-Q, and the best known Q (refAMD) is only
  ~6% faster than ND. But UMFPACK-auto does 886K nnz in 40.6 ms (21.8M
  nnz/s) vs our best 763K in 79.9 ms (9.5M nnz/s): at SMALLER factor
  size we run ~2x SLOWER. The gap is EXECUTION RATE, not ordering.
  Companion UMFPACK-oracle bugs found the same day: (1) umfpack_bench
  passes Info=NULL for non-GIVEN modes, so ordering_used/strategy_used
  printed there are uninitialized stack (stably 0) — flops/numericsize
  /timings are real (set by numeric); (2) 6.3.2 quirk — symbolic with
  Info=NULL takes a different ordering path than with real Info
  (explicit SYMMETRIC+AMD: 57M-class object vs extracted-AMD-Q
  replaying 659M); mechanism unexplained, do not rely on it; (3)
  SYMMETRIC+GIVEN ignores Quser entirely (ordering_used=NONE, same
  533M for every perm) — there is NO same-Q UMFPACK replay path.
  REVISED CPU PLAN: (a) numeric kernel rate + robustness first —
  phase split (TEMP SONIUM_SLU_PROFILE instrumentation, still in tree)
  128² base: TRSM 6.0 / GEMM+scatter 10.4 / tail 40.0 / ~29 gather+reach
  (n_apply 60K, n_gemm 6.4K) vs ourAMD: 13.3 / 74.8 / 62.6 / ~92
  (n_apply 349K, n_gemm 21.5K): chain etrees cause O(P²) contributor
  applications, each paying reach-walk + panel-wide TRSM + dispatch.
  Fix directions: the 349K applies are REAL (nonzero updates) at 0.06
  GEMMs each — fragmentation, not skippable work. Cure is adaptive
  chain-panelization (merge consecutive chain-run panels beyond 16:
  pairs drop quadratically; uniform 32/64 was already falsified
  134/189ms on ND trees, so the rule must be etree-shape-aware), queued
  as the next implementation. Alternative: MF vehicle (already wins at
  256²: 396 vs 752 total, immune by right-looking scheduling, but 153
  vs 35 MB factor tax needs structural work); (b) analyze 31ms vs 12.6ms
  symbolic (2.5x) second; (c) ordering last — native refAMD-class AMD
  (fix amd.rs tie-breaking/shape) buys ~20% nnz, ND separator-mass
  excess (6795 vs ~1300 optimal, SEPPROF) costs little time under our
  engine. Separator-direct FM deprioritized accordingly.
  FM buckets BTreeSet -> lazy global BinaryHeap (2026-09-25): BIT-IDENTICAL
  perm (cmp vs pre-change reference), FM 24.4 -> 21.3ms serial, all 144
  sparse + 248 direct tests green. Allocation theory falsified (only -13%:
  FM cost is algorithmic passes x O(E), not bucket overhead). Ordering TIME
  retired as a gap: full vendor ND path (metis_perm) = 110ms wall vs ours
  40ms; UMFPACK symbolic 12.6ms vs our ordering+symbolic ~27ms = -14ms
  possible, vs numeric -44ms. Priority stays numeric rate.
  FLOP CENSUS (2026-09-25, TEMP counters, 128² ND): ours executes 179.5M
  flops (GEMM 161.6 / TRSM 12.9 / rank 5.0) in 85.5ms = 2.1 GF effective
  vs UMFPACK 57.3M in 40.6ms = 1.4 GF. WE ARE 1.5x FASTER PER FLOP —
  the gap is 3.1x EXCESS FLOPS, decomposing into (a) rectangle padding
  ~1.9x (update density 0.517 measured on update_buf; dense GEMM pays
  zeros; strict-supernode chunks already near the density ceiling 0.49,
  so padding is structural, not amalgamation slop) and (b) structure
  ~1.7x (necessary ~95M vs vendor 57M at similar nnz: update-term
  concentration from ordering/pivot/amalgamation choices). NOTE: our
  amalgamation is well-matched to ND (METIS-Q through OUR symbolic =
  1140K nnz vs ND 949K). Next vehicles: exact-pattern execution
  (fresh approach — Gustavson variants already falsified) and/or
  fill-aware amalgamation (SuperLU-style, not pure chain relaxation).
  AMALGAMATION PROBE (2026-09-25): relaxed width 16->8/4/2/1 moves GEMM
  flops 161.6->117.8/100.3/94.0/81.5M and density 0.52->0.69, but time
  94->96/104/120/147ms (panels stay ~1030: numeric_panels re-merges to
  16; factor_nnz bit-identical 948734 throughout). Finer chunks are a
  dead end on time: per-chunk dispatch+scatter ~0.5µs dominates. w16
  stands. Exact-pattern must FUSE chunk dispatch (one pass per panel),
  not refine it; structure gap (~95M necessary vs 57M vendor) is separate
  and needs ordering/pivot-shape work, not amalgamation tuning.
  PRE-TRSM SKIP falsified (2026-09-25, same-binary interleaved A/B 98.9
  vs 95.3ms): full-slice pre-scan costs 11.5ms (no early exit) while the
  TRSMs it saves were ~free (0.3ms); the in-TRSM early-out already
  skipped chunk work for empties. Reverted with NOTE. Lesson: reach
  tightening (fewer empty applies), not cheaper detection.
  PAIR-LOOP (2026-09-25): the 539K (source,target) visit loop is only
  ~7ms; panel-ancestor pruning is UNSOUND (row-links bypass panel
  ancestry), and reach-derived flagged-panel lists save at most ~5ms.
  Not the lever. Biggest compute bucket is crest 22ms (push/branch
  overhead, not arithmetic: rank is 5M flops) — next target.
  CREST CENSUS + TRAIL HOIST (2026-09-25, TEMP counters): per ND-128²
  run crest visits 932k blocks, 3.37M (block x trailing) pairs but only
  2.48M FMAs (26% zero-skip), 474k above-diag pushes. Hoisting the
  pivot-row trailing pattern once per column (pivot row never written
  in-loop; same FMAs, same order) LANDED bit-identical (nnz 948734,
  res 3.78e-14 exact): crest 28.6 -> 24.8ms profiled, numeric medians
  89.5 -> 88.0ms unprofiled (-1.5ms). Remaining ~21ms crest is ~2.5M
  FMA+finite-checks, 458k mults, ~1.9M Vec pushes — micro-cuts only;
  structural (ordering) and GEMM-padding work outranks further crest
  surgery.
  DRIVER LIVENESS STAMPS (2026-09-25): TEMP timer showed the O(P²)
  contributor loop at 35.5ms/run for 539K calls of which 478K vacuous
  (call + slice scan each). Reach-marking now stamps
  source_seen[chunk_source[chunk]] on flag transitions (chunk_source
  extends 1:1 as panels complete) and the driver skips unstamped
  sources with one compare — ascending order preserved, all FP
  bit-identical. TEMP census proved exactness (zero apply-side vacuous
  entries), so the redundant in-apply any() became a debug_assert.
  needed clear+resize per panel replaced by set_list stamp-clear.
  LANDED (interleaved same-machine A/B): numeric medians 90.5 ->
  85.2ms (-5.3ms, -5.9%; trail hoist ~-1.5 + liveness/stamp-clear ~-3.8),
  nnz 948734 / res 3.78e-14 exact, all direct tests green, clippy clean
  in touched file. Gap to UMFPACK-auto 40.6ms now 2.10x.
  WIDTH UNIVERSALITY + SCALING (2026-09-25): w32 re-tested post-fixes —
  128² 99 vs 83ms, 256² 622 vs 541ms. w16 stands at both sizes; the
  width falsification is not size-specific (fill bit-identical 4725972
  at 256² across widths: wider panels add pure padding). 256² profile:
  applies 572K (9.5x), vacuous 8.2M (17x), GEMM 1020M @ density 0.651
  (up from 0.517), flop ratio vs vendor IMPROVED to 2.1x (1107M vs
  519M) — but vendor rate rises 3.3x with size (1.4 -> 4.7 GF via
  BLAS3 fronts) while ours stays flat (~2.1 GF): the remaining gap is
  rate-at-scale (fragmented chunks + scalar tail), not flop count.
  FM SMALL-GRAPH GATE (2026-09-25, partition.rs FM_MIN_REFINE=128):
  serial census showed FM 21.9ms/ordering with 82% of passes (2073/2543)
  on <128-vertex graphs. Gating FM polish below 128 vertices (coarsened
  initial bisection stands) does NOT speed parallel ordering (19.6 vs
  19.1ms recursion: small-FM lives off the critical path) but fills
  BETTER at both sizes — 128²: 936582 vs 948734 (-1.3%), numeric 85.2
  class (interleaved pairs all favor gated); 256² reps=1: 4672196 vs
  4725972 (-1.1%), numeric 481.7 vs 517.9 (-7%), residuals 5.0e-14 both.
  Small-FM polish appears to overfit tiny cuts; skipping it is
  fill-positive here. KEPT (tests green both crates, clippy clean).
  Arg-convention warning: sluord takes (nx, reps, which, key) — earlier
  "permnum nd" runs were fresh-ND-ordering mode (no ndvar_nd.perm ever
  existed); true saved-perm A/B needs keys fm128/ndfull.
  ENGINE CROSSOVER (2026-09-25): MF-amal0 (exact, multifrontal_blr,
  current binary) BEATS supernodal-ND at both sizes — 128²: 96.2/96.5/
  95.8ms total (residual exactly 4.56e-14, fronts=287, 31MB) vs SN ~108
  (21 ordering + 3 symbolic + 85 numeric); 256²: 367.1 (146MB) vs SN
  ~620. Gap to UMFPACK totals: 1.8x at 128² (96 vs 53), 2.2x at 256²
  (367 vs 165). MF uses the same nested_dissection (FM gate applies).
  Rate hypothesis for MF factor (~70ms/287 fronts ≈ 240µs per front at
  128²; ~260ms/1195 ≈ 220µs at 256² — suspiciously constant): per-front
  fixed costs (allocation/scratch/BLAS dispatch on small fronts), not
  arithmetic. MF solve 6.8ms also ~3x expectation. Next: MF factor
  profile (assembly vs dense vs Schur) + per-front alloc audit. The
  supernodal line stands landed; MF is now the lead CPU vehicle.
  MF EXACT PACK FAST PATH (2026-09-25, blr.rs compress_block): worker-sum
  profile showed pack 411ms of 649ms front work — every block ran full
  truncate_rrqr at eps=0 only to conclude "keep dense". Exact mode now
  skips the analysis (nonzero -> Dense; zeros keep rank-0 LowRank, half
  mode unchanged). LANDED bit-identical residual 4.56e-14 x4 runs: pack
  411 -> 35 worker-sum, factor wall 55.5 -> 16.7, total 96 -> 52-56ms.
  Gap to UMFPACK total (53ms) now 1.0-1.06x — effectively PARITY at
  128² (within machine variance; needs interleaved vendor A/B to call).
  Remaining MF buckets (worker-sums): trsm 156, pivot 61, assemble 19
  serial, schur 21, pack 35. Next: TRSM internals + assembly.
  MF TRSM QUAD INTERLEAVE (2026-09-25, front.rs): FMA census pinned TRSM
  at 59.3M FMAs / 169ms worker-sum = 0.7 GF (scalar latency-bound), so
  4-column/4-row interleaves (u12 then l21) overlap the independent
  chains — each lane's FMA sequence bit-identical, 2x2 Cont skip hoisted
  (column-independent), per-j reciprocals hoisted across l21 lanes.
  LANDED residual 4.56e-14 exact: trsm medians 164 -> 109 (u12) -> 66
  (both), factor wall ~16.7 -> ~12, total avg ~52. Direct tests green;
  the one front.rs clippy hit is pre-existing type_complexity (shifted
  line). Tenant noise note: untouched-bucket worker-sums (pack 25-130)
  swing 3-5x run to run — all bucket claims use non-overlapping ranges
  or medians, never single samples.
  REVERTED THIS ROUND (both with NOTEs in tree): (1) vec_axpy rank-1 in
  factor_frontal_block — pivot medians 61 -> 54 but residual moved
  4.56 -> 4.53e-14 (SIMD FMA fusion, one rounding vs two); same-class
  gain inside noise is a bad trade vs bit-identity. (2) parallel MF
  assembly — bit-identical and faster cold (11.2 vs 19.6 rep1) but
  slower warm (9.4-9.9 vs 8.0-8.5; top levels too narrow for fan-out);
  hot path is warm. Serial stands; total back to ~53ms, 4.56e-14.
  MF ASSEMBLY: CB-index BTreeSet -> bulk sort+dedup AND direct block
  extend-add (no per-front to_dense alloc+copy; rank-0 blocks skipped)
  both measured NEUTRAL at 128² (assemble warm 8.0-8.5 -> 6.9-9.2,
  inside tenant noise) with residual exactly 4.56e-14. Kept: correct,
  tested (new extend_add_panel_matches_materialized_path equivalence
  test incl. -0.0 disclosure), and the copy it removes scales with CB
  area (~6x larger at 256²). The per-front fixed cost is therefore in
  the frontal zeros + validation + dispatch, not indices/copies.
  128² PARITY CALLED (2026-09-25, 5 interleaved pairs same machine):
  vendor totals 56.5/56.3/56.3/56.4/56.6 (median 56.4, variance <0.5%)
  vs MF-amal0 55.9/59.3/57.5/54.4/53.4 (median 55.9) — 0.99x vendor.
  Residuals 2.27e-14 vs 4.56e-14, same 1e-14 class. Scope of the call:
  2D-Laplacian 128² total wall (ordering+factor+solve) only. Open: 256²
  CPU (367 vs 165, 2.2x), GPU 256², other matrices/sizes, TEMP revert.
  MF SOLVE (2026-09-25): wall split showed front compute 5.6-5.9 of 6.7ms
  (panel gemvs + substitution; reduce/scatter ~1ms). Removed the
  redundant per-front validate in forward/backward_front (checked
  variants re-validate at the boundary; factorize-pack validated once):
  solve 6.3 -> 5.6ms, residual exact, tests green. Remaining solve work
  (gemv dot->axpy reformulation, substitution interleave) is same-class
  territory — queued behind factor buckets.
  (GPU tenant turned over 12005 -> 109672, same binary — windows may
  open between jobs; keep checking.)
  MF memory tax (/tmp/sluord MFSTATS): amal0 factor stores 32.6 MB at
  n=16384 vs ~12-15 MB supernodal factor (~2.5x front-overlap tax) and 7 MB
  UMFPACK numeric — bytes moved explain the time ladder better than flops
  do. 256x256 scale probe (8 threads): MF-amal0 factor 383 + solve 13 ms
  (153 MB!, res 6.15e-14) vs supernodal-ND analyze 133 + numeric 619 ms
  (4.94M nnz, res 5.24e-14) vs UMFPACK symbolic 64 + numeric 125.5 +
  solve 17 ms (519M flops, 4.45M doubles, res 2.78e-14). Same ~3-5x gap at
  both sizes, same cause (bytes + executed flops, not rate: UMFPACK 4.1
  Gflops effective at scale). MF is the better vehicle at scale (383 vs
  619) but its overlap tax (153 vs 35 MB) needs structural work. Retracted:
  no MF default-thread pathology exists (128T factor ~106 ms ~= 8T ~112;
  only 1T is slow at ~315 ms) — the parallel_front_threshold sweep was
  stood down, fronts at amal0 never reach the 1024 default.
  Falsified: BLR compression as a fast-approximate path on this matrix
  (/tmp/sluord `mfblr` cases, 8 threads, harness-only config sweep, no
  library changes). epsilons 1e-14/1e-12/1e-10/1e-8 at amal256:
  204.5/151.8/168.2/159.6 ms vs 191 exact (<=20% gain) with residuals
  7.9e-14/4.3e-12/9.5e-10/1.0e-7 (~1000x epsilon amplification); at
  amal1024: 545/546/541 ms vs ~570 exact (~5%). The rank-detection and
  low-rank algebra overhead eats the savings — Laplacian-like grids at
  this size have no exploitable low-rank structure at these block sizes
  (BLR's design target is smooth-kernel BEM matrices, untested here).
  Exact-amal0 (112 ms, 4.72e-14) stays the fastest MF operating point.
  GPU crossover verdict (/tmp/gpuprobe, RTX PRO 6000 Blackwell + 128-core
  Threadripper CPU, cublas vs OpenBLAS square GEMM): FP64 — CPU wins at
  EVERY size to 4096 (0.11 vs 0.17 ms at 128; 151.5 vs 434 ms at 4096;
  cublas plateaus ~316 Gflops while 128-core OpenBLAS reaches ~906).
  GPU-direct (FP64 sparse LU) is correctly NOT pursued on this box. FP32 —
  GPU compute wins 7-8x from N=512 up (14.3 TFLOPS at 4096: 9.6 vs 70.5
  ms), but PCIe transfers eat most of it (with-transfer wins only 1.25x
  at 1024, 2.7x at 4096). Scoped GPU leg: FP32-or-lower, N>=1024,
  resident-data mixed-precision iterative solves (HBM bandwidth), NOT
  FP64 direct factorization. That points at sonium-solvers-iterative +
  the facade's mixed-precision API on the cuda feature, not at direct.
  CUDA stack live-verified on the hardware (2026-09-25): sonium-compute
  --features cuda `device_preconditioner_cuda` 5/5 and `parity` 67/67
  green on the RTX PRO 6000. Present on-device: cublas/cufft/custom
  kernels, device preconditioner, dense-LU API; CPU-only:
  MixedPrecisionRefinement (zero cuda references). No device CSR SpMV
  exists (cuda/kernels are BEM/FMM-oriented). Precise GPU-leg remainder:
  device SpMV + resident-FP32 Krylov iteration + precision-cast
  refinement loop — a multi-day build, scoped but not started.
  CORRECTION: the loop already exists — `gmres_ir_backend_preconditioned`
  (GMRES-IR, backend-generic, on-device CUDA casts) — and 2026-09-25 saw
  its first live CUDA run (/tmp/girprobe, Laplacian, Identity precond,
  f64-outer/f32-inner): n=16384 — CPU 559 ms vs CUDA 2779 ms (GPU 5x
  SLOWER: ~640 tiny inner SpMVs are launch-latency-bound); n=65536 —
  CPU 10815 ms vs CUDA 5094 ms (GPU wins 2.1x, same 4-outer/~1000-inner
  path, true residual 6.8e-13 both). The GPU leg has a measured operating
  window: large-n iterative where per-launch arithmetic amortizes. Next:
  preconditioned (non-Identity) CUDA runs to cut the ~1000 inner iters,
  then the crossover moves to smaller n.
  Preconditioned follow-up (PolynomialJacobi degree 3 / damping 0.75 on
  the f32 inner side, /tmp/girprobe `jacobi`): n=16384 — CPU 498 ms (327
  inner) vs CUDA 122 ms (312 inner, res 7.0e-13); n=65536 — CPU ~6775 ms
  (491, rock-solid across processes) vs CUDA median ~1360-1400 ms (492,
  res 6.3e-13). Degree sweep at n=65k (CUDA iters): deg1 1000, deg2 594,
  deg3 492, deg5 360, deg8 276 — deg3 is the wall-time knee (higher
  degrees cost more per apply than they save in iters). Damping 0.75 vs
  1.0 unresolved (same iter class; walls overlap). CONTAMINATION CAVEAT
  (partially retired): a co-tenant GPU job (generate-roomeq-data) made
  early CUDA walls highly variable while CPU medians vary <1% and all
  iter counts/residuals are exactly deterministic. A clean-room window
  (0-1% util) then verified 128² jacobi3-0.75: CPU 487 (486/487/487) vs
  CUDA 108 (108/108/108) ms, res 7.0e-13 — 4.5x GPU win, zero variance.
  At 256² the tenant woke mid-run (86% by end): CPU 6710 tight vs CUDA
  292/709/1301 (median 709, res 6.3e-13) — the 292 rep plausibly near
  clean but unprovable; 256² clean re-measurement still queued.
  Methodology hardened to warmup + median-of-3 in-harness because of
  this. Mechanism (robust): unpreconditioned inner hits the 500-iter
  cap, so O(500^2) Arnoldi orthogonalization dominates; Jacobi keeps
  inner runs short and its extra SpMVs are resident throughput work the
  GPU does well.

## 256² MF re-verify (2026-09-25, after exact-pack + TRSM quads + jumps)
  mfamal0 @256² (sluord rel, same binary): factor_plus_solve 218.0 ms,
  res 5.86e-14 (was 367 ms pre-pack/quads/jumps: 1.68x from kept changes).
  UMFPACK 165 ms -> gap now 1.32x (was 2.2x). MFPROF worker-sums @256²:
  pack 427.5 / trsm 343.2 / pivot 216.1 / schur 206.3 / assemble 70.3.
  Pack is the biggest bucket at scale; next target: FactoredFront
  panel-vs-block redundancy (panels retained post-pack?) / zero-copy solve
  reads. 128² parity 0.99x vendor stands.

## Owned-pack conversion (2026-09-25, worktree direct-x86-opt)
  `BlrPanel::from_dense_owned_*` + `Panel::from_dense_owned_*` + schur->pack
  `DenseOrPanel` plumbing: single-block panels move the source Vec into
  `BlrBlock::Dense` (no alloc-zero-copy); multi-block copies row segments by
  memcpy into capacity-exact buffers; f11/cb scratch copies also memcpy.
  Bit-identical: residuals exactly 4.56e-14 @128² and 5.86e-14 @256².
  Pack worker-sum: 128² 53->~16 (3.3x, 6 reps), 256² 427->~145-169 (2.6-2.9x,
  3 reps). Wall @256² 218.0->~212 (206.0/211.8/215.4).
  MACHINE CAVEAT: compute-bound worker-sums inflated ~1.75x between the
  pre-change window and now (trsm 343->~600, schur 206->~420, identical
  trsm_mfma=262.1 flop counts; memcpy phases pack/assemble flat-or-better),
  consistent with tenant CPU contention (load ~4.3, tenant
  generate-roomeq-data active, GPU 96%; /proc/cpuinfo 2200 MHz vs
  lscpu max 4308 MHz = 1.96x clock ratio, matching the 1.75-2.0x compute
  inflation). The wall delta understates the win;
  re-measure totals on an idle window. check/clippy/tests green (only
  pre-existing lints; factorize.rs BlrPanel unused-import predates change).
  NOTE: stash-A/B vs HEAD does not compile (prior worktree changes
  interlock); A/B was same-work binary + exact-residual identity instead.

## U12 unchecked-quad falsified (2026-09-25, throttled machine)
  Raw-pointer lane access in `u12_quad` (same ops/order, SAFETY+debug_assert):
  4 reps trsm 67-83ms vs indexed 71-82ms — inside the variance band, no
  separable win; LLVM already hoists the checks or the loop is FMA-bound.
  Reverted with NOTE at site; L21 mirror not attempted. TRSM remains a
  BLAS-2 scalar loop; structural options (dtrsm wiring needs 2x2-pivot
  audit) deferred to an idle-machine window where A/B resolves.

## dtrsm audit + delayed-pivot robustness gap (2026-09-25, throttled machine)
  AUDIT (throttle-proof, counts not timings): the mfamal0 benchmark uses
  `BlrConfig::default()` => `PivotStrategy::Partial`, and `select_pivot`
  returns `BkPivot::OneByOne` unconditionally under Partial (col fixed at
  k); only BunchKaufman can emit 2x2. So on the benchmark path U12 =
  dtrsm(Left,Lower,NoTrans,Unit) shape and L21 =
  dtrsm(Right,Upper,NoTrans,NonUnit) shape — no 2x2 fallback needed for
  Partial/Rook. Locked in by new test
  `partial_pivoting_never_emits_2x2` (multi-front DD + row-reversed
  single-front with swap assertion; asserts all-OneByOne + col identity).
  BLOCKERS for wiring: (1) no trsm kernel exists in sonium-compute/core
  traits — new kernel work; (2) blocked/GEMM/dtrsm reorder summation =>
  residual moves at ~1e-16 level, which the vec_axpy precedent
  (4.56->4.53e-14, reverted) says the bit-identity bar forbids. Needs a
  user decision: relax bit-identity for TRSM or keep scalar.
  GAP (out of scope, locked): row-reversed DD tridiagonal (cond ~3,
  det verified nonzero via numpy) aborts `SingularFront(14)` under the
  tiny-block multi-front config, while single-front and non-reversed
  multi-front (15 fronts) factorise fine — mass-delay + multi-front
  interaction. Ignored regression test
  `partial_row_reversed_multi_front_factorises` added; re-enable after
  the delay path is fixed. Benchmark path (natural/ND orderings) never
  hits this.

## Solve assume_validated split (2026-09-25, throttled machine)
  `forward/backward_apply_checked` split into validate + `*_assume_validated`
  inner; solve.rs uses the inner (pack-time validation + immutability make
  per-solve re-validation redundant; `b` length check kept). Bit-identical
  (validate has no side effects), tests green. NEUTRAL on wall: solve_ms
  4.9/4.4/5.4 @128² vs 4.1-6.1 band before — `validate()` is O(nfs) scalar
  len/bound scans (~0.3ms/solve), never the bottleneck (GEMV/gather/allocs
  are). Kept as trust-chain refactor for future solve work, not as a win.
  256² health: 213.1 total, res exactly 5.86e-14.
  NOTE: solve GEMV is scalar per block; SIMD gemv exists in sonium-compute
  but reorders summation => barred by bit-identity (same as TRSM).

## Symmetric CB-index skip (2026-09-25, throttled machine) — LANDED
  `is_structurally_symmetric` prepass (one binary search per upper-triangle
  entry, O(nnz log deg), once per factorisation); when symmetric, the
  column-incoming buckets are neither built nor consulted (`a_perm_in:
  Option`, part (2) of `compute_cb_indices` skipped). Output is the
  identical sorted-unique set (redundant pushes, not missing ones), so
  bit-identity holds: residuals exactly 4.56e-14 @128², 5.86e-14 @256².
  Assemble bucket: 128² ~20->~18 (17.0/18.4/19.5/18.5), 256² 70-73->62.6/
  62.8 (-10ms, -14%). Totals: 128² bests 54.3/54.6 (was 55.4+), 256²
  203.6/204.6 best-ever (was 206-218). Nonsymmetric path unchanged
  (existing `structurally_nonsymmetric_matrix` test covers it) + new
  `structural_symmetry_prepass` unit test. Suite 248 green, clippy clean
  incl. a doc-list relocation fix (fn inserted mid-doc-list). Scratch-Vec
  reuse considered and dropped: the CB buffer escapes into front storage.
  Gap to UMFPACK @256² now ~1.23x (203.6 vs 165, throttled clocks).

## Assembly sub-bucket ranking (2026-09-25, MFAPROF TEMP, 256²)
  cb 2.2 / valid 0.2 / zeros 31.1 / scatter 20.6 / extend 10.1 (sums to
  assemble 65.2 — full accounting). Zeros dominates and is a bandwidth
  floor: ~1.6GB of front materialization streaming at ~50GB/s; every page
  is eventually touched (scatter/extend/schur-read), so first-touch cost
  cannot be avoided within the dense-front algorithm — vendor pays the
  same shape. Extend (10ms, per-element map indirection on single-block
  Dense panels) has no cheap fix (scatter-add is inherently random).
  Scatter (21ms) likely CB-row repeat visits + cold-front streaming, same
  floor class. CONCLUSION: serial assembly ≈ at floor; remaining 256² gap
  is compute rate (pivot/trsm/schur, currently 2x-throttled) and flop
  volume (census 293M vs 122M vendor — needs structure work, idle-machine
  validation). MFAPROF counters stay under the pending TEMP revert.

## GPU window opened/closed + TRSM rate analysis (2026-09-25)
  Tenant turnover observed live: old generate-roomeq gone, clocks briefly
  4142 MHz, GPU 0% idle — then a NEW generate-roomeq (PID 730774, 1.7GB
  GPU) claimed the GPU within ~1 min (96% by the time the 7.7s girprobe
  rebuild finished). No CUDA run: launching into contention repeats the
  original contamination failure. girprobe binary is rebuilt fresh against
  the worktree (cuda feature) and ready for the next idle window; the
  queued clean 256² jacobi3-0.75 CUDA re-measurement stays queued.
  New tenant also burns ~500% CPU (clocks back to 2200 cap) — no clean CPU
  window either (the 3×256² reps taken during the turnover: 203-211 wall,
  same throttled worker-sums; kept as extra confirmations, not clean data).
  RATE ANALYSIS (throttle-proof direction): TRSM census 262.1M FMAs over
  ~620ms worker-sum = ~0.42 GFMA/s aggregate vs ~35 GFMA/s scalar ceiling
  (16c × 2.2GHz) = ~1.2% — TRSM is NOT FMA-bound but latency/dispatch-bound
  (tiny fronts: loop/closure/dispatch overhead + L-streaming vs FMAs).
  This confirms the pending fused-chunk dispatch attack as the right
  direction over SIMD; interleaved A/B stays fair under stable throttle.

## Same-window vendor A/B @256² (2026-09-25) — CPU PARITY EXCEEDED
  Interleaved reps, identical 2D-Laplacian 65536², throttled clocks (fair):
  - Sonium MF-amal0: 197/204/213 (factor+1 solve, ordering incl.), res 5.86e-14
  - UMFPACK auto (AMD default): sym 75-76 + num 198-200 + solve 19 = ~294, res 2.79e-14
  - UMFPACK metis: sym 257 + num 176 + solve 19 = ~451 (ordering search dominates; worse single-shot total)
  - UMFPACK best: num 173.5 (fill 3.53e8 vs auto 5.19e8) but sym 496 search cost — not a runnable total
  - SuperLU COLAMD: perm 25 + fact 488 + solve = ~535, res 4.40e-14 (fact flops 1.06e9)
  Ratios (sonium/vendor total wall): 0.69x UMFPACK-auto, 0.45x metis, 0.38x SuperLU.
  The stale "UMFPACK 165ms / 1.23x gap" is RETRACTED: 165 was numeric-only
  from an unthrottled window; like-for-like totals were never behind.
  CAVEATS (do not overclaim): vendor ran AMD-default ordering vs our ND
  (ordering quality differs; `given` mode failed status=-3 — our_nd.perm is
  128²-sized, kernel-gap isolation unavailable); memory NOT at parity
  (146MB vs 35MB numericsize); no CHOLMOD comparison yet (true SPD SOTA —
  no harness, queued); single matrix class; one grid size pair.
  CPU leg of the goal is therefore satisfied on the lead vehicle + class to
  the extent the workspace can measure it; GPU leg (clean 256² CUDA) still
  queued behind the tenant.

## Min-block sweep + default 128->64 (2026-09-25) — LANDED
  TEMP MFCENSUS (closed-form counts, throttle-proof): @256²-min128
  pivot 182 / trsm 265 (FMA) / schur 573 MFLOP-ish, front bins
  tiny/small/med/big = 16/519/659/1 (1195 fronts).
  AMALGAMATION sweep (config-only): amal 0/16/32/64/128 totals
  ~201-204 flat — halving front count saves nothing measurable, fill
  growth eats it; amal0 vehicle choice reconfirmed, residual creeps
  5.86->6.47e-14 with coarser trees.
  MIN-BLOCK sweep (config-only, the money): @256² min128 203.6 / min64
  195.8 / min48 191.6 / min32 186.4 / min24 189.2 / min16 196.9 (U-curve,
  min32 best); @128² min128 54.7 / min64 46.1 / min32 48.3 (min64 best).
  Means-of-3: 256² min32 182.0 vs min64 175.9 (min64 takes it — within
  noise but sweeps both sizes); 128² min64 46.2 vs min32 48.3.
  Mechanism: finer trees collapse fill/flops (pivot 182->21, trsm FMA
  265->84 @min32; assemble 63->48, pack 165->50) faster than tail costs
  grow; rebound at min16 (7487 fronts: schur worker 647 for 446M flops =
  0.66 GFLOPS tiny-GEMM disease + solve 16.7). Residuals IMPROVE with
  finer trees (5.86->5.17e-14 @256², 4.56->3.95e-14 @128²).
  CHANGE: `BlrConfig::default().min_block_size` 128 -> 64 (BLR only;
  LDLT default untouched — no evidence there). Suite 248 green, clippy
  clean. Harness keeps explicit configs (comparisons stay pinned).
  NOTE: min32-vs-min64 @256² is inside machine noise; revisit on idle.
  NOTE2: ordering (nd_proxy) itself swings 61-92ms under tenant — the
  partition code is a future target, not today's.
  CHOLMOD scoped OUT: the benchmark Laplacian is numerically nonsymmetric
  (-1.1/-1.2/4.01/-0.8/-0.9; pattern-symmetric only), CHOLMOD needs SPD
  values — UMFPACK/SuperLU are the applicable SOTA, both beaten on totals.

## schur_sub for MF: investigated, BARRED by bit-identity (2026-09-25)
  The MF RightLook path materializes `update = L21·U12` (with redundant
  zeros) then subtracts (2x traffic). `T::schur_sub` (in-place C -= A·B,
  already wired+tested for dense LU) would halve that traffic (~1-2ms
  wall) — but the f64 override is a single alpha=-1/beta=1 GEMM whose
  summation order differs from GEMM-then-subtract, so the residual would
  move at ~1e-16 (same trade class as the reverted vec_axpy). Barred under
  the bit-identity bar; needs the user decision with the TRSM item.
  The separable zeros-before-overwrite alone needs unsafe for ~0.5ms:
  not worth it, not attempted.
## Ordering fill gap identified (queued, needs idle validation)
  Census-implied MF flop volume ~1.3G flop-eq vs UMFPACK-auto 519M and
  SuperLU 1.06G on the same matrix: our ND(+min64) tree carries ~2.5x
  UMFPACK's fill (also the 146MB-vs-35MB memory gap). Wall parity holds
  DESPITE fill (rate advantage), but fill parity is the deep lever for
  both wall and memory. Ordering work is validated only on idle (ahead:
  partition sweep measuring fill + wall + residual together).

## Self-review of session diffs (2026-09-25, review skill)
  Re-verified: symmetry-skip redundancy proof (upper-triangle coverage,
  self-loop/empty-row/delayed-set edge cases sound); DenseOrPanel ownership
  + LeftLook join-shape change (equivalent results); census k=0/overflow
  edges. FOUND+FIXED: MFCENSUS TRSM label (was "mflop", counts FMAs —
  renamed trsm_mfma with note). Performance proof: owned-pack, symmetry,
  min64 all benchmarked with residuals + vendor A/B. Backends: changes are
  CPU-factor-path only, no cuda/metal surface touched. Accuracy: residuals
  bit-identical to prior self, vendor cross-check same class; new tests are
  property tests appropriate to the change; known delay-path gap locked as
  ignored regression test. `cargo check --workspace`: 0 errors (no public
  API changes — all new items pub(crate)/private; full workspace test run
  deferred as disproportionate: touched crate suite 248 green).

## 128² scoreboard completed same-window (2026-09-25)
  Sonium MF min64: 46.6 (res 4.15e-14) / UMFPACK auto: 55.8 (12.6+39.9+3.3)
  / SuperLU: 84.4 (6.7+73.1). Ratios: 0.84x UMFPACK, 0.55x SuperLU.
  Combined with 256² (0.60-0.69x / 0.33-0.38x): ahead of both vendors on
  totals at both sizes, same-window, throttled clocks.
  Declined this turn (logged, not attempted): solve-phase alloc pooling
  (~1-2ms wall for moderate complexity + unsafe-adjacent work) — below the
  bar set by falsified micro-items; solve stays as-is pending a structural
  reason to touch it.

## Default-change blast radius closed (2026-09-25)
  No external users of `BlrConfig::default()` / MF default path outside
  sonium-solvers-direct (grep: only dense/simplicial LU hits elsewhere).
  Facade `sonium-solvers` suite: 116+ passed, 0 failed (96s). The min64
  default is safe downstream.

## Small-n default guardrail (2026-09-25)
  min128 vs min64: 32² 10.4 vs 10.7 (wash), 64² 16.6 vs 15.8 (min64
  better). Residuals 1-3e-14 both. No small-problem regression from the
  min64 default.

## Worktree hygiene (2026-09-25)
  Untracked files banded_schur.rs / gpu.rs / sparse_lu/amd.rs all date
  from 2026-09-24 (prior sessions) — not this session's, left untouched.
  This session created no new worktree files; all changes are modifications
  to tracked files + PERF.md. Scratch stays in /tmp.

## Wall decomposition: SERIAL-DOMINATED (2026-09-25) — strategic re-aim
  256²-min32 wall 186.4 = ordering ~86 + assembly ~48 + solve ~13 + factor
  wall only ~39 (parallel region; worker-sums are 5-10x larger but
  slack-absorbed). 128² even starker: factor wall ~0-5 of 46. So compute
  kernels (throttled or not, SIMD or not) can move at most ~40ms of wall;
  TRSM/schur_sub under relaxed identity ≈ 2-4ms — DOWNSIZED, no longer the
  headline ask (still logged for the record).
  The wall lives in serial ordering+assembly (~70%) and both are at
  vendor parity / bandwidth floor. Remaining wall lever of any size is
  FILL (ordering quality → less assembly AND factor work together):
  census table @256² (MFLOP P/T×2/S, fronts, mem, wall):
   min16: 15/141/446, 7487f, ~109MB?, 196.9 | min24: 17/153/453, 5387f, 108.6MB, 197.9
   min32: 21/168/460, 4169f, ~?MB, 186.4 | min48: 31/201/474, 3087f, 114.5MB, 192.3
   min64: 52/257/494, 2237f, 121.1MB, ~196-201 | min128: 182/531/573, 1195f, 146.5MB, 203.6
  Wall is FLAT (186-203) across 2x flop range: flop cuts convert to wall
  only via the serial pieces. Fill work (idle validation) is now THE CPU
  project; factor-kernel work is maintenance.

## Tree-injection probe: architecturally wrong, not pursued (2026-09-25)
  Idea was METIS-ordered MF via external-perm injection. Rejected on
  design: the assembler needs the DISSECTION structure (separators,
  fronts, levels), which is not derivable from a flat perm file —
  honouring a foreign perm would collapse MF to a simplicial chain.
  Fill stays a partitioner-internal lever (queued, idle).
  Correction to earlier fill math: min32 census ≈ 649M flop-eq vs
  UMFPACK-AMD 519M = 1.25x (not 2.5x — that was min128); min64 ≈ 803M
  = 1.55x. The min-block change nearly closed fill parity as a side
  effect. Memory (panels store full rectangles) still trails at ~3x.

## min32-vs-min64 bracket resolved: TIE, keep 64 (2026-09-25)
  Interleaved pairs @256² (4x{32,64}): 192.3/196.7, 200.2/192.1,
  190.8/191.9, 188.7/195.1 → min32 takes 3/4 pairs but means are 193.0
  vs 194.0 (1ms amid ±8 swings from nd_proxy 72-99 + factor noise).
  min32 keeps better residual (5.17 vs 5.43e-14, every rep) and fill
  (649 vs 803M) without converting to wall. With 128² decisively min64
  (46.2 vs 48.3), default 64 stands. Revisit only with a structural
  change that moves the tie.

## CUDA GMRES-IR: per-step device memsets removed (2026-09-25)
  REVIEW finding: every Krylov basis vector alloc (`V[j+1]` inner+outer,
  `z_j_outer`) went through zeroing `raw_alloc` (device memset + allocator
  call per step; ~500 steps/solve at 256²) then was fully overwritten.
  Switched the 3 per-step sites to `raw_alloc_uninitialized` with SAFETY
  justification (copy+scale / cast fully overwrite before any read;
  cast kernel + host fallback verified full-length). check/clippy clean,
  compute gmres suite 41 green (incl. CUDA tests on busy GPU).
  CORRECTNESS (contention-proof): 256² jacobi3-0.75 CUDA run converges
  with outer/inner iters (4/492) and residual history IDENTICAL
  digit-for-digit to the stashed-baseline run — no uninit leak.
  TIMING inconclusive as predicted: baseline reps 293/898/1374 vs changed
  2513/2562/2576 (tenant contention dominates; no mechanism for slowdown
  — strictly less work). Clean-window number still queued.

## TEMP profiling reverted (2026-09-25)
  Removed all env-gated TEMP instrumentation: `SONIUM_MF_PROFILE`
  (factorize.rs wall split + MF_A_* assembly buckets + MFPROF/MFCENSUS/
  MFAPROF reports; front.rs MF_T_*/MF_F_*/MF_C_* statics, span timers,
  census; solve.rs wall-split timers + report), `SONIUM_SLU_PROFILE`
  (sparse_supernodal.rs thread-local phase/census/flop buckets, scan/crest
  toggles, SLUPROF reports), `SONIUM_PART_PROFILE` (partition.rs thread-
  local buckets, pprof_timed wrappers, dead pub pprof_report). All sites
  purely observational (timers/counters/eprintln); falsification NOTEs kept.
  check --all-targets clean (moved test-only BlrPanel import into tests
  mod; remaining warnings pre-existing); direct+sparse suites green
  (248 passed + all integration targets, 0 failed). Tenant still holds
  machine (generate-roomeq ~55min); clean CUDA/CPU timing still queued.

## x86/GPU local-optimum record (2026-09-26)

Platform: Threadripper PRO 3995WX (128 logical), Linux x86_64, rustc 1.98.1,
RTX PRO 6000 Blackwell (driver 610.57). Vendor refs: OpenBLAS (pthread,
threads=16, not oversubscribed), cuSOLVER/cuSPARSE/cuBLAS. Sonium at
RAYON_NUM_THREADS=16 (default 128-thread pool oversubscribes: dpotrf-2048
25.8ms default vs 20-22ms at 16). Affinity default-on. Method: interleaved
same-process A/B, median-of-reps; box noise ±10-15% MT.

### Headline gaps (fresh 09-26 unless noted)

| Workload | Size | Sonium | Vendor | Gap |
|---|---|---|---|---|
| dpotrf checked (shipped) | 2048 | 24.6 med | dpotrf 12.2 med | 2.02x |
| dpotrf in-place | 2048 | 22.0 med / 20.2 min | 12.2 / 11.8 | 1.80x med, 1.70x min |
| dpotrf | 1024 | ~6.0-6.2 | ~4.5 | 1.30-1.35x |
| sgetrf full/body | 1024 | 7.3-8.6 / 7.1-8.4 | 4.9-5.9 | 1.3-1.7x |
| sgetrf full/body | 2048 | 30.2 / 28.8 | 15.1 | 2.00x / 1.91x |
| dgetrf full/body | 2048 | 41.9 / 43.2 | 25.7 | 1.63x / 1.68x |
| dgeqrf+dorgqr (blocked Householder, new) | 1024 | 104-110 | 55 | 1.90x |
| dgeqrf+dorgqr | 2048 | 432 | 245 | 1.76x (was 1.92x pre trans-A gate) |
| dgeqrf+dorgqr, RAYON=32 | 2048 | 418 | 236 (OB @32t) | 1.70-1.78x |
| dgeqrf+dorgqr, OB @4t | 2048 | 423 (ours @32t) | 402 | 1.05x |
| dgeqrf+dorgqr, single-thread | 2048 | 1009 | 787 | 1.28x |
| gpu-dgetrf (delegation) | 4096 | 163.36 | cuSOLVER 162.99 | 1.00x |
| gpu-SpMM 80^3 Lapl, nrhs=16 | 512k rows | 1.255 | cuSPARSE best-alg 1.254 | 1.00x |
| gpu-SpMV f64 | 512k rows | 0.101ms, 740GB/s | roofline | near-roof |

Record correction: the earlier "dpotrf-2048 ~16ms / 1.3x" does not
reproduce — no current config yields 16ms (1024 runs ~6ms, 2048 runs
20-22ms min). Treat pre-09-26 dense-Cholesky bests as stale; the table
above supersedes them.

### Landed 09-26: blocked Householder QR (dense_qr.rs, +541 lines)

Correct compact-WY with LAPACK-convention T (upper, NB=32, dispatch at
ncols>=64): forward trailing uses Tᴴ (reverse-order product), Q pass uses
T; GEMM-V copies carry zero strict-upper + explicit unit diagonal (dlarfb's
DTRMM treatment — feeding R entries as V broke orthogonality to 1e15);
betas retained out-of-band for R's diagonal; forward T cached per panel
(2.15x -> 1.89x). Correctness at machine precision (orth/recon ~1e-15,
Q/R match MGS to 5e-16/9e-15); 266/266 direct tests green, clippy clean.
Test fixture replaced: the old sin/cos mixture had true rank ~4
(SVD: 60/64 singular values < 1e-12*max) — rank verdicts on it were noise.

### Falsified 09-26 (reverted, noted in code where applicable)

- QR NB=64: 1024 1.90x->2.35x, 2048 1.92x->2.11x (4x Level-2 panel work
  outweighs fatter GEMMs). Noted on QR_BLOCK_NB.
- Cholesky SONIUM_CHOL_NB sweep at 2048 (min-ratio): 32: 1.81x,
  48: 1.74x, 64: 1.70x, 128: 1.77x, 256: 2.38x, 512: 3.71x. Default
  64 optimal, no change (thinner GEMMs lose what smaller panels save).
- Trans-A panel GEMM split knobs (M_SPLIT / NO_RANKK_M / NOROWBAND):
  all within ±15% box noise, no win. LANDED instead: serial-only A^T
  pre-pack gate (pack wins ST on all 5 mapped shapes, gather wins MT
  everywhere: 32x300x2048T 1.60x vs 2.59x, 256x2048x2048T 0.82x vs
  1.26x; bit-exact either way). QR-2048 1.92x -> 1.76x; no collateral
  (potrf/sgetrf re-gated in-band). NN panel shapes BEAT OB (0.5-0.6x).
- Earlier window (kept as local optimum): team/broadcast/pack/coarser-panel/
  SYRK-64/LU-48/L3-spread, LU in-place funnel (neutral end-to-end).
- Probe hygiene 09-26: `obslu`/`obgetrf` body arms cloned inside the
  timer (cold mmap + 33MB faults inflated dgetrf-2048 body to 2.48x);
  clones hoisted out (matching `potrf_ab`), true body 1.68x. Lesson:
  time the factor, not the allocator; full (warm pool) was fair all
  along. Follow-up order-alternation control (`obgetrf` order B):
  second arm always wins ~1-1.7ms either way — L3-pollution artifact,
  no workspace-reuse penalty exists.
- LU gap localized 09-26 (new probe `f32shapes`): f32 panel GEMMs healthy
  (MT 0.48x/1.09x/0.65x/0.74x on schur/tn/schur2/square-1024; ST ~1.15x),
  so the sgetrf-2048 body gap (~1.85x) lives in the serial
  panel-GETF2/TRSM/pivot floor, not GEMM. Per lu.rs notes the
  prerequisite is panel L2 residency at 2048 (786KB compact spills L2;
  intermediate widths swept via temp hook, all tied) — threaded panels
  only after that.
- dgetrf-1024: ST gap (1.9-2.2x) EXCEEDS MT gap (1.6-1.8x) — kernel /
  structure, not threading. Panel primitives probed healthy 09-26 (new
  probe `bl1shapes`: pivot 0.74x, ger 0.47x, scal 0.95x vs OB) yet
  per-column arithmetic closes the model: ~10us x 1024 sequential
  columns ~= 10ms of our 13.8ms ST total. Panel area heavily mined
  already (in-place/compact sweeps, scan trials); no primitive defect.
- Cholesky SYRK probed healthy 09-26 (new probe `syrkshapes`: MT
  0.58x/1.00x/0.97x/0.85x, ST ~1.1x, bit-exact) — dpotrf gap is not SYRK.
- QR thread-scaling honesty 09-26 (new probe `dotshapes` for the
  Level-1 side): dot at parity (1.00-1.04x), axpy 1.1-1.3x — no
  primitive defect behind the ST 1.28x. Gap is scale-free (~1.7-1.9x
  MT at 512/1024/2048) and mostly serial-kernel volume + per-call
  costs over ~500 BLAS calls. Config note: QR likes RAYON=32
  (418ms, 1.70x); at equal 4-thread budgets we are at parity (1.05x).
- Cholesky panel-row solve (right-Trans TRSM, new probe `trsmshapes`):
  2.1x MT / 1.62x ST at 2048x64 (~2ms weight at n=2048); right-NoTrans
  wins (0.75x). Both directions share the tile+left-core machinery
  (dir flag only), so the gap is some mix of backward-substitution
  direction efficiency and tile pack/unpack traffic (4MB moved for
  8.4M MACs). Known dead ends, do not retry without a new idea: native
  strided-row AVX2 kernel (hsum epilogue slower, noted in trsm.rs),
  task pairing/coarsening (tied end-to-end). Residual is pool-rendezvous
  floor on 0.14ms calls + kernel direction, not sizing. Negative result
  09-26: the AVX2 left core carries symmetric two-row unrolls in both
  directions (forward `!trans` + descending mirror, same `nb>=64` gate),
  so there is no missing-unroll defect to fix — the trans deficit is
  subtler (direction efficiency), not worth chasing at ~1ms weight.

### Remaining levers (architectural, multi-turn)

1. Trans-A small-m MT efficiency: SHIPPED 2026-09-27
   (pack-once-shared `A^T`, `SONIUM_GEMM_PACK_A_MT=0` kill-switch):
   one shared `A^T` pack + `NoTrans` recurse replaces per-band
   strided gathers (one cache line per element, re-read per band).
   `32..256x1984x1984T` 0.50-0.61x -> 0.55-0.66x vs OpenBLAS,
   bit-for-bit identical (probe max-rel-diff unchanged). Gated to
   MACs >= 100M (below: pack is pure overhead, `256^3T` 0.60x ->
   0.41x) and pack <= 8M elements. Residual trans-A gap is true
   16T-hot NoTrans scaling (lever 5: 4T-hot is ~parity, 16T-hot
   SQ1024 0.65x), not transpose handling. (Earlier "0.57-0.62x NT
   ceiling" quotes were heat-depressed interleaved medians.)
2. dpotrf-2048 checked-path cost (+12% over in-place): validation is
   contract, not waste — cheaper validation is the only lever.
3. MT thread placement: FALSIFIED for non-L3-fitting factors (taskset
   A/B 09-26: compact 2-domain 21.99/21.52, spread-16 21.64/20.87,
   unpinned 22.0/20.9 med/min — identical; OB degraded equally both
   ways, i.e. box contention not placement). L3-fitting 1024 already
   gets compact-pool pinning by design (spread placement: 13.5ms vs
   6.0ms unpinned). Remaining 20-27ms run variance is not placement;
   box runs frequency-capped (53% scaling MHz).
4. Single-threaded kernels are at parity (QR 1.28x, potrf-1024 1.3x) —
   no ST work left; all remaining gaps are MT scaling/traffic.
5. GEMM-probe heat artifact, found 2026-09-27 (methodology, not a
   kernel defect): interleaved ours/OB microbenchmarks let our pinned
   workers' cores park during the OB phases (schedutil floor 2.2GHz),
   depressing our medians ~2x at 4T (SQ1024 102GF interleaved vs
   173GF ours-only vs 185GF taskset-confined, 3/3 interleaved
   rounds). Ours-only sustained (hot) at 4T is 0.9-0.95x of OB-hot —
   the 4T "NoTrans ceiling" was heat, not driver. Discipline: judge
   GEMM rates hot-vs-hot (sustained ours-only vs vendor best), never
   from one side of an interleaved pair. End-to-end factorizations
   self-heat (funnel + lookahead keep workers loaded), so lever 3's
   factorization verdict stands — this bites microbenchmarks and any
   solve with alternating idle phases. Real 16T-hot lever: SQ1024
   277GF (0.65x of OB 416), SQ2048 465GF (0.82x of 569) — scaling
   4T->16T, not heat, is the remaining GEMM work (band sizing /
   cross-CCX traffic).
6. 16T-hot square follow-ups, all FALSIFIED 2026-09-27 (hot ours-only
   A/B, temp `SONIUM_GEMM_BANDS` gate added then removed): N-split
   band count 16 vs 32 ties on squares (SQ2048 462/464 and 475/474
   interleaved pairs — the single-sample 558GF "win" was a transient,
   replicate before theorizing); 8 bands catastrophic (underfill),
   48/64 noisy shape-dependent wiggles. `SONIUM_GEMM_M_SPLIT=1`:
   SQ1024 +2-5% (3 samples), SQ2048 tied (direction flips), thin-m
   W256T destroyed 267GF -> ~130GF (4 samples, B re-reads) — not
   shippable; a square-conditioned M default needs portfolio-wide A/B
   first. Cross-CCX placement confirmed as a mechanism (8T packed in
   2 CCXs 352GF vs spread over 8 groups 147GF on SQ2048), but our
   default pin already packs workers (0-15) and the sharing pattern
   (N-split A re-reads vs M-split B re-reads, both ~128MB at 1024^3)
   shows no clean lever — the 16T-hot gap stays open work.
7. TRSM single-band inline, shipped default-on 2026-09-27
   (`SONIUM_TRSM_ONEBAND=0` kill-switch): `unblocked_left_lower` at
   `nrhs <= TRSM_NB` (=128) runs the serial band, skipping scope +
   pool install (~25-40µs) the second band never repays (m=nb=128:
   124µs -> ~100µs at 16T, replicated; bitwise identical — columns
   independent, same kernel ops). Narrow by construction (only
   nrhs == 128 changes; <128 was already serial). End-to-end weight
   ~0.1% of dgetrf (one last-panel call per solve); the larger TRSM
   share (1.18-1.38x at n=640-1920) is band-kernel rate, still open:
   16 workers deliver only ~104GF total — but re-framed 2026-09-27,
   that is 7.2x over serial (vs OB 9.8x; the "6.5 vs 23GF/core" mixes
   shapes), i.e. only ~25% scaling efficiency gap, worth ~0.1ms of
   dgetrf. Analyzed out: the scalar 4-wide-SLP band kernel already
   extracts ~50% of FMA peak on a latency-bound triangular solve;
   chasing 1.18x -> 1.0x large-n buys ~0.1ms end-to-end. TRSM work
   stops here unless a cheaper dispatch (sub-10µs pooled) appears.
8. LU-2048 phase split, sized 2026-09-27 via temporary `SONIUM_LU_PROFILE`
   timers (reverted): factor 0.6 + trsm 2.7 + schur 31.8 + valid 1.1ms
   (hot, 16T) — model closes, no missing time. Thin-probe Schur rates
   (150-250GF) are real; in-solve Schur runs ~170-216GF. Lookahead A/B:
   OFF = factor 10.4 + schur 24.1 (full-parallel 216GF), ON = factor
   0.6 + schur 31.8 — net +2ms win stands, keep. `SONIUM_COMPACT=0`
   ties compact at 2048 (31.75 vs 31.83 schur) — compact choice
   neutral here, keep. OB-8T total 25.2ms implies OB-hot Schur
   ~220GF+ (their interleaved thin numbers ~154GF look heat-depressed
   too). LU gap anatomy: serial panels + block-column tax − overlap +
   trsm + valid; no single fat — every phase must move. Next LU lever:
   deeper overlap (serial block update ~10ms runs un-overlapped) or
   parallel panels; both structural, untested.
9. Parallel block-column update FALSIFIED 2026-09-27 (temp
   `SONIUM_LU_BLKPAR` gate, reverted): routing the lookahead block
   update through the parallel strided kernel regresses 39.55 ->
   40.97ms at 2048 (non-overlapping quartiles), rows>=1024-gated
   still 40.39, wide-pool still 40.94 — thin-shape B-pack + dispatch
   outweigh the win at every scope width. The serial author was
   right; do not retry without addressing thin-shape pack cost.
10. LU-2048 decomposition variants FALSIFIED 2026-09-27 (env-only,
    nothing to revert): `SONIUM_GEMM_FORCE_2D=1` regresses 41.36 ->
    45.32ms (non-overlapping) — 2D row-splitting hurts k=128 Schur
    here. `RAYON_NUM_THREADS=8` (matching OB's 8): 47.93ms vs 40ms
    at 16T — we scale 8->16 (+20%), so not over-threaded; OB-8's
    25.4ms stands. perf counters unavailable (paranoid=4), so no
    L3/DRAM telemetry — mechanism work proceeds by A/B only.
11. TRSM uplo-enum probe bug + left-chunk win, 2026-09-27. First, a
    methodology catch: the TRSM probes passed uplo=121 (CblasUpper)
    while meaning Lower=122 — OB silently solved the identity
    (output == input, verified) and all prior "1.18-2.32x" TRSM
    ratios were void. Fixed in-probe (scratch only); true TRSM
    standing is 0.26-0.73x, kernel 23GF vs OB 34GF serial. Second,
    shipped: left chunk width 64 -> 16 (`SONIUM_TRSM_LCHUNK`
    override retained, capped to scratch): the nb x nw kernel block
    fits L1 (32KB at nb=128) — same-binary u12 A/B +19-28% at 16T
    across ncols 384-1920 (1920: 0.343->0.287ms, 1408: 0.302->0.245,
    896: 0.207->0.170, 384: 0.168->0.131), +3-4% serial; end-to-end
    dgetrf-2048 neutral-tie (+0.3ms, overlapping quartiles — TRSM is
    2.7 of 41ms). Bitwise identical (exact-copy pack/unpack, kernel
    order untouched); 469 compute tests green.
12. Thread-count sweet spot, 2026-09-27 (env-only, no code): 16->32T
    nearly doubles GEMM micro throughput (SQ1024 280->497GF, SQ2048
    465->702, W256T 268->384; corroborates the old "596GF @32T"
    comment). End-to-end knee at 32T: QR-2048 384->338ms (1.37x vs
    OB-16T), LU-2048 39.6->38.3 (1.50x vs OB-8T); 48T regresses both
    (QR 371, LU 40.9). Default-128 behaves sanely (caps work: SQ1024
    560GF, LU 40.9) — no default-config bug. Best-vs-best noted with
    thread asymmetry (ours-32 vs OB-16/8); OB's own sweet spots.
13b. LU block size 64 FALSIFIED 2026-09-27 (const probe, reverted):
    dgetrf-2048 38.3 -> 42.1ms with variance blowout [40-50], -1024
    9.1 -> 10.8. Unlike QR, LU's serial GETF2 doubles with nb and the
    lookahead covers only one panel — no overlap gain to fund it.
    Correction logged: LU uses 32-wide panels (64 panels at 2048),
    not 128 — the u12/oneband probes' p=128 shapes are
    LU-unrepresentative (they match TRSM_NB, not LU_BLOCK_SIZE);
    future TRSM probes should use m=32 panel shapes.
13. QR block size 32 -> 64 shipped 2026-09-27 (one-const change):
    halves dispatches, fattens panel GEMMs (W64 274GF vs W32 212GF
    at 32T) — dgeqrf-2048 338->317ms, -512 18.2->17.1 (nb=128
    falsified: serial panel cost dominates, 414ms). Residual cover
    exists (`blocked_is_orthogonal_and_reconstructs`, 128x96,
    multi-panel at both values) and passes; 400 direct + 116 facade
    green, clippy clean. QR-2048 now 1.30x vs OB-16T at 32T.
14. PARNORM (parallel QR tail norms) FALSIFIED 2026-09-27 (branch
    + timers, fully reverted — zero remnants): parallel copy+dlassq
    over 32 bands runs in 3ms (15x the serial norm loop) yet totals
    move only ~5ms (noise-level) — first-touch faults + copies/scans
    dominate the tail, not the norms. Last-ulp tolerance deviation
    for 1.5% was correctly declined; the exact fix is workspace reuse
    (exact), not parallel norms. Do not retry.
15. Box clock drift 2026-09-27: scaling_max_freq read 2.7GHz uniform
    mid-session, scaling_cur_freq 4.1GHz later the same day (cap is
    opportunistic, ~1.9x clock range). Post-revert QR reads 342ms vs
    317 pre-experiment with OB flat — code verified identical
    (tests green, zero temp remnants), so absolute times are
    incomparable across >10-min spans on this box. Discipline
    hardened: same-run OB ratios ONLY; re-baseline every finding.
16. QR tail micro-surgery FALSIFIED/REVERTED 2026-09-27 (fully reverted
    — zero diff vs session baseline, verified: 0 deletions, 0 remnant
    strings, 270 lib tests green): (a) parallel validate loses to
    serial (valscan 33MB: par 3.96ms vs serial 2.83ms — bandwidth-bound
    scan, 128-way dispatch pure overhead); (b) fused Q/R finite checks
    save ~1ms (standalone scans hit L3-hot data, not DRAM); (c) parallel
    dlassq norms re-banned per lever 14 (same ~5ms noise, same ulp
    order change). Phase medians (temp timers, reverted): pre ~20,
    fwd ~130 (reflect 24 + wy 9 + trail 74), qpass ~150, thinq ~41,
    R+checks ~31 per 2048 factorize; GEMM phases run at vendor rate
    (~125GF vs OB ~117GF) — the whole 1.4x gap is tails+panels.
    Do not retry tail micros; the exact fixes remain QrWorkspace
    (faults/zeros/clone) and recursive panels (BLAS-3 panel factor).
17. QrWorkspace SHIPPED 2026-09-27 (opt-in, LuWorkspace precedent):
    `QrWorkspace<T>` + `qr_factorize_with_workspace` (direct + facade
    re-exports) reuses a/qmat and all panel scratch across calls;
    Q/R outputs stay owned-fresh. Same flop sequence -> bitwise
    identical (new f64 + Complex64 equivalence tests, incl. shape-change
    reuse exercising all_t/vbuf clears). dgeqrf-2048 same-run: ws 275
    (1.12x) vs plain 318 (1.29x), reproduced 290 (1.18x) vs 331
    (1.35x) — steady −41ms (13%). Recursive panels killed in design
    (one level saves only ~5ms: halves BLAS-2 but pays gather/scatter
    + T glue; not worth ulp risk). Remaining ws gap ~29ms: owned Q/R
    zeros+faults + serial panels (reflect 24 + wy 9).
18. LU phase anatomy 2026-09-27 (temp timers, fully reverted — zero
    remnants, 272 lib tests green): dgetrf-2048 @32T = 39ms (1.55x vs
    OB-8T 25.3): pre 1.7 + panel 0.4 (lookahead hides) + trsm 3.4 +
    schur 29 + valid 1.1. Big-trailings run 200-287GF (vendor rate);
    trailing<1024 collapses 89GF -> 6GF (dispatch+pack overhead).
    TRSM MACs floor FALSIFIED (tied 39.48 vs 39.39): the 114µs-vs-7µs
    micro used L2-hot data; in-situ panels are strided L3-cold so both
    paths are pack-bound — micro overstated dispatch. k-sweep:
    k=64 runs ~2x the GF rate of k=32 at small t (49 vs 24 @256,
    131 vs 81 @512). Next: 2-panel Schur/TRSM fusion in the tail
    (k=64 halves dispatches; last-ulp, residuals govern) for ~-5ms.
19. 2-panel Schur/TRSM tail fusion FALSIFIED/REVERTED 2026-09-27
    (fully reverted — zero remnants: no TAILFUSE switch, no pair
    branches, `trsm_panel` back to the full-trailing signature; 273
    lib tests green, clippy clean for lu.rs): the fused form was
    correct (off-vs-on maxdiff 1.4e-12 @1024, 3.6e-12 @2048, same
    pivots; n=1024 reconstruction held at 1e-12) but measured a
    noise-level tie in same-process order-rotated A/B @32T
    (`tailfuseab` probe, $HOME/tmp/getrfprobe): 1024 on 9.28 vs off
    9.15ms (1.01x, quartiles barely overlap — on slower), 2048 on
    41.19 vs off 42.12 (0.98x, heavily overlapping quartiles).
    Direction flips with n = noise, not signal; the predicted ~-5ms
    never appeared (the tail is only 8 pairs, and the 64-wide fused
    TRSM pack eats the dispatch savings). The n=1024 reconstruction
    test stays, renamed `large_threaded_reconstruction_n1024`.
    Do not retry pair fusion without a bigger win region (wider
    tail or cheaper fused-TRSM pack).
20. Serial-Schur threshold + worker-side block update FALSIFIED/
    REVERTED 2026-09-27 (both fully reverted — zero remnants,
    403 tests green, clippy clean for lu.rs). (a) Routing
    trailing<512 Schur updates through the serial kernel ties
    (1.01x, overlapping quartiles @32T dgetrf-2048; maxdiff 0e0,
    confirming the serial/parallel bitwise contract) — the
    small-trailing updates sit inside the lookahead overlap, so
    kernel choice never reaches the critical path. (b) Temp timers
    sized the caller-side serial lookahead block update at ~8.6ms
    per dgetrf-2048 (62 calls, fully un-overlapped). (c) Moving it
    into the lookahead worker onto the gathered compact panel
    (L2-resident C, same serial kernel/order — maxdiff 0e0) still
    ties (0.99x, heavily overlapping quartiles): the worker factor
    is the long pole, so relocated work just lengthens the join —
    hiding caller time only helps when the caller is critical.
    Lesson: overlap moves work off the critical path only if the
    receiving side has slack; measure pole balance before
    restructuring. Next LU candidates: cheaper worker factor
    itself (gather+overlay at STREAM floor — irreducible; scalar
    core ties vendor dgetf2), TRSM 3.4 vs 1.21 (dispatch-bound,
    on hold), or entry/valid scans (~2.8ms combined).
21. Strided-pack SW prefetch FALSIFIED/REVERTED 2026-09-27 (fully
    reverted — zero remnants): T0 prefetch of strided source rows
    in the f64 B tile/panel packs loses at every distance
    (dist-8 1.05x, dist-4 1.02x, same-process order-rotated
    strided arena 1500x1500x32@ld2048; maxdiff 0e0) — extra
    uops/traffic on a non-bottleneck, joining the two earlier
    falsified B-walk prefetches. Do not retry pack/streamer
    prefetches. Key reframe from the same probe: at this strided
    thin shape we now measure 249-257GF vs vendor 137-142GF — we
    WIN the kernel-rate battle head-to-head, so the in-situ Schur
    gap is dispatch count x scope cost + hetero interference, not
    pack/stream rate. Next: fresh phase anatomy (the last one
    predates lookahead tuning) before any more restructuring.
    REFRAME 2026-09-27 (obgetrf @32T dgetrf-2048, same run):
    thin-kernel head-to-head we now BEAT vendor 1.5-2.5x (992^2x32
    181 vs 74GF; x64 226 vs 124; x128 240 vs 163; x256 303 vs 196)
    yet end-to-end is 1.59x (40.3 vs 25.3ms) with clone ~0
    (body≈full). Model closes around overhead, not rate: serial
    block 8.6 + TRSM 3.4 + entry 1.7 + valid 1.1 ≈ 15ms serial
    against ~25ms kernel-at-rate. Kernels are done; every
    remaining LU lever must delete or hide serial/overhead work:
    TRSM dispatch (2.2ms prize, on hold), entry-fold fuselet
    (~0.5-1ms, exact), fast-core pinning/pole balance (untested),
    persistent lookahead worker (spawn x62). No more kernel-rate
    trials for LU Schur.
22. Entry clone-fold fuselet FALSIFIED/REVERTED 2026-09-27 (fully
    reverted — zero remnants, 403 tests green, clippy clean): fusing
    the factor-buffer clone with the entry max/finite fold into one
    scalar copy-fold loop loses decisively, 1.18x at dgetrf-2048
    (45.6 vs 38.7ms same-process order-rotated; the pair was exact,
    maxdiff 0e0 both API paths, same pivots). The scalar fold+copy
    runs ~7ms slower than memcpy + SIMD fold — the bool-AND fold
    chain blocks vectorization (known defeat). Bound the idea
    before any retry: even an ideal NT-store SIMD copy-fold saves
    <0.5ms over today's two fast passes, so the line stays dead
    unless entry grows. Remaining overhead shortlist: fast-core
    pinning/pole balance (untested), persistent lookahead worker
    (spawn x62, ~0.5ms), TRSM dispatch (2.2ms prize, on hold).
24. Last-panel tail merge FALSIFIED/REVERTED 2026-09-27 (fully
    reverted — zero remnants, 403 tests green, clippy clean):
    absorbing the final W columns into one wide last factor is
    exactly correct (same pivots, maxdiff 1.8-3.6e-12) but
    measures a clean tie at every width — W64 1.01x, W96 0.99x,
    W192 1.00x (7 reps each, overlapping), and decisively at
    W96 with 101 order-rotated pairs (median on-off diff
    +0.27ms, IQR [-1.03,+1.36]). The removed dispatches were
    mostly hidden (worker-factored) while the fatter GETF2 and
    the lost tail lookahead overlap cost real time — pole
    balance again. Method note: paired same-rep differences
    resolve ~0.3ms effects the median-ratio cannot; in-place
    probe arms bypass NUMA first-touch (+4ms both arms — keep
    the full API for absolute numbers, pairs for verdicts).
    Do not retry tail widening without reducing the wide-GETF2
    cost itself.
25. Fresh LU anatomy 2026-09-27 (temp accumulators, fully reverted —
    zero remnants, 403 tests green): per dgetrf-2048 @32T =
    entry-fold 1.74 + factor 0.58 (caller-side exposed only; the
    ~11ms worker factor is fully hidden) + trsm 3.26 + serial-blk
    8.13 + schur-remainder 21.3 + valid 1.15 ≈ 36.2 vs ~37.5 wall
    (funnel/clone/scopes close the rest). Caller IS the critical
    path — every relocation trial (tail fusion, worker block,
    tail merge) tied because the worker has slack but the join
    granularity re-exposes it. Exhausted with verdicts: serial
    blk hide (tie), serial-Schur (tie), entry fuselet (1.18x
    loss), pack prefetch (1.02-1.05x loss), tail merge (tie at
    101 pairs), placement (OB-interference artifact), TRSM chain
    (at ~50% FMA peak — STOP stands), scans (bandwidth floor).
    Best-vs-best confirmed honest: vendor plateaus at 8-16T
    (24-25ms; 32T regresses to 30.8), so 1.5-1.6x is the true
    standing, not a config artifact. dgetrf LU is now a
    sub-2ms game (persistent worker ~0.3 risky, serial-tail
    unpromising) — next battlefield should be Cholesky (2.7x 1T,
    SYRK mass) or QR tails/panels, or the cuBLAS leg.
26. GPU leg re-verified 2026-09-27 (Blackwell RTX PRO 6000, CUDA
    13): dense-LU delegation 1.04-1.05x vs cuSOLVER+H2D at
    1024/2048/4096 (cuvendorlu same-process pair); SpMM wrapper
    1.263ms vs cuSPARSE best-alg (csr_alg2) 1.255ms at nrhs=16
    (1.007x); SpMV 0.101ms 740GB/s relerr 1.6e-15 (roof holds).
    Delegation parities all stand. cuBLAS-GEMM wrapper-vs-raw
    closed 2026-09-27: stale probe rewritten to the backend trait
    API (`pool.backend().gemm` vs raw `cublasDgemm_v2` on the same
    handle/stream/buffers, order-rotated) — ratio exactly 1.000
    at 1024/2048/4096, maxdiff 0e0. Zero wrapper overhead, as
    constructed. GPU leg complete: every vendor pair is at
    1.00-1.05x or roof.
23. Placement x count 2x2, dgetrf-2048 @taskset (OB flat 25.0-25.2
    across all four — drift-anchored): unpinned/R32 39.06 (1.55x),
    unpinned/R16 40.89 (1.63x), pinned0-31/R32 37.95 (1.51x),
    pinned0-15/R16 37.12 (1.49x). Unpinned SMT exclusion (0-63/R32)
    ties unpinned (39.13) — SMT collision is NOT the mechanism;
    tight packing is (concentration beats migration/L3 scatter).
    Interaction: unpinned wants more threads (32<16, reproduces
    the 32T knee), pinned wants fewer (16<32: less per-scope
    overhead x63+63+62 joins, no stragglers). Best (pinned-16)
    beats default (unpinned-32) by ~2ms. Prescription: extend
    packed-pool routing to big factorizations (wide packed pool,
    matched count — placement win needs no L3 fit, cf. QR
    pinned-8 0.95x) behind a kill-switch, with portfolio A/B
    (GEMM shapes, QR, Cholesky, sgetrf) before shipping as a
    default. No code changed this lever (taskset only).
23b. CORRECTION 2026-09-27 — lever 23's prescription is VOID
    (methodology catch, same class as lever 11's uplo bug): an
    OB-free per-rep probe (`lutime2`, $HOME/tmp) shows ZERO
    placement/count sensitivity in our solve alone — warmed
    dgetrf-2048: R32 unpinned 37.3-38.3, R32 pinned 37.0-38.7,
    R16 unpinned 36.7-38.4, R16 pinned 36.2-37.5 (all
    overlapping). The taskset wins appear ONLY with OpenBLAS
    active in-process (OB threads/pollution interacting with mask
    width), so the 2x2 measured interference geometry, not library
    placement. DO NOT build the packed-pool default on this
    evidence. Real-user processes don't carry a spinning vendor
    pool next to the solve; our pools (mirror/small/tiny pinned,
    compact L3-packed) already place correctly. Next: persistent
    lookahead worker (~0.5ms), TRSM dispatch (2.2ms, on hold).
27. Fused SIMD dlassq thin-Q norms SHIPPED 2026-09-27 (default-on,
    kill-switch `SONIUM_QR_DLASQ_SCALAR`): `ComplexField::
    dlassq_row_update` hook (default = scalar equations; f32/f64
    overrides route to the AVX2 kernels in sonium-compute blas1)
    wired into the blocked-QR thin-Q norm loop. Same-run
    order-rotated A/B (`qrdlassqab` probe, $HOME/tmp/getrfprobe,
    16T): 512 fused 16.09 vs scalar 16.30 (1.013x), 1024 67.93 vs
    69.04 (1.016x), 2048 330.39 vs 334.08 (1.011x), reproduced
    15.93/66.89/332.46 vs 16.10/67.25/337.30 — steady ~1.0-1.6%.
    BITWISE_MATCH (Q+R+rank checksums) at all three sizes:
    maxdiff 0e0, gate on vs off. Review caught two kernel bugs
    pre-ship: (a) ±inf passed the NaN-EQ finiteness mask
    (`inf == inf`), updating state where the scalar errors — now
    excluded via a `mag <= MAX` bounded compare; (b) the f32 abs
    mask `-1i32 >> 1` is an arithmetic shift (all-ones no-op), so
    negative magnitudes leaked into the update — now explicit
    `0x7FFF_FFFF` (f64 used the correct u64 literal all along).
    Committed guards: 3 bitwise unit tests (dispatch + raw AVX2 vs
    scalar, adversarial ±0/subnormals/±inf/NaN/overflow-scale
    ratios, vector/tail widths). TEMP-QRPHASES anatomy fully
    reverted (zero remnants — no QRPHASES/qr_phase/__t_ strings).
    472 compute lib + full direct suite green, clippy clean at
    edited lines.
28. LU trial batch FALSIFIED/REVERTED 2026-09-27 (all fully reverted
    — zero TEMP remnants verified by grep, 273 direct + 10 trsm tests
    green, clippy clean at edited lines; one doc correction kept).
    Fresh TEMP-LUPHASES anatomy (dgetrf-2048 ~39.8: entry 1.6 +
    panel 0.5 + trsm 3.6 + schur 32.9 + valid 1.2) with per-trailing
    Schur buckets (t>=1536 211GF, 1024-1536 171GF, 512-1024 118GF,
    256-512 54GF, <256 31GF) and isolated serial lookahead block
    (62 calls, 8.85ms — the serial kernel runs ~15GF = single-core
    ceiling on Zen2, so parallelism is the only lever there).
    (a) TRSM serial-small (nb<128 to serial band): end-to-end tie
    (serial band slower at 9GF + per-call fixed costs survive).
    Correctness catch: serial-vs-parallel MISMATCHED checksums
    (residuals equal at 2.6e-18) — bands wider than lchunk re-chunk
    locally and chunk edges change blocked order; corrected the
    "bitwise identical under any split" comment in trsm.rs (kept).
    Routing changes need checksums, not independence reasoning.
    (b) TRSM chunk-pack skip at nb<=32: decisive loss (scope
    43->73us, serial 83->251us at n=1024) — the L1-set-collision
    pack is load-bearing. Do not retry.
    (c) GEMM one-wave grid: bucket 256-512 regresses +0.34ms, big
    buckets -0.4 (stragglers) — net noise. Reverted.
    (d) Parallel lookahead block (same slices, parallel hook): tie
    (~-1ms, noise) — nested tiny-pool install + per-task packs on
    1M-MAC calls cost what the serial sweep costs. Reverted.
    (e) Packed-A + parallel block: end-to-end tie (same pathology
    as (d)) BUT BITWISE_MATCH vs the serial baseline at 64/256/2048
    (identical factor checksums) — extends the parallel==serial hook
    contract to this shape. Evidence only, gate reverted.
    (f) TRSM force-unroll below nb=64: micro +8% at n=2016 only,
    flat at 256/1024 — tuning confirmed. Reverted.
    (g) GEMM force-2D on full LU: -0.7ms (largest bucket only);
    portfolio risk per its note — not shipped.
    Standing model: the 8.85ms serial block is at ceiling, so the
    next block lever must pack L21 ONCE into contiguous storage and
    amortize that pack over block + remainder (shared packed A, one
    scope) — packing per call re-pays the disease. Same prerequisite
    unlocks nb=64 (which doubles total block MACs 134M->268M, so a
    packed-parallel block is required, not optional). Vendor TRSM
    19us/call vs ours 41us at n=1024 (kernel + dispatch) stays hard;
    mid-size Schur routing tried 3 variants, all weak — the
    remaining LU gap is structural (contiguous panels), not tuning.
29. LU tiny-pool funnel SHIPPED 2026-09-27 (default-on f64,
    `SONIUM_LU_TINY=0` opt-out): whole dgetrf on the 16-worker tiny
    pool (one install; `run_tiny` in blas3/mod.rs sets IN_TINY so all
    nested scopes route inline with 16-wide sizings). Root cause of
    the RAYON=16-vs-128 delta (~2.5-5.5ms OB-free): ~200 nested
    tiny/small pool installs (~10-30µs each) under the default
    mirror/compact funnels. OB-free order-rotated: 2048 on 34.6-59.6
    vs off 40.0-71.7 (1.16-1.20x, five runs); same pivots at
    1024/2048, maxdiff 5.6e-17 at 2048 (pool-geometry TRSM band
    re-chunking), residuals equal (8.7e-19 at 1024). Envelope:
    f64-only via `lu_prefers_tiny_pool` (sgetrf-1024 regresses 2x on
    tiny — f32 kernels oversubscribe 16 workers; complex unmeasured,
    stays old path), n>=2048 with 64MB cap (512 loses 0.93x,
    1024 conflicted 1.08x/0.88x under ±30% drift with compact's 8.3ms
    best intact, 4096 loses), inert at global<=16. OB-in-process
    shows only ~-1.5ms (vendor pool interference, 23b class — ship
    basis is OB-free). dgetrf-2048 now 1.36x vs OB-standalone
    (34.6 vs 25.4). 273 direct + 472 compute + 47 core green both
    arms, clippy clean at edited lines. Contamination re-check
    (co-tenant `generator_admis` burning ~12 cores, load 8+): separate-
    process A/B degrades to noise, but order-rotated in-process still
    resolves on 35.4 vs off 40.4 (1.14x) with identical ship hashes
    (a522… vs 8122…, same pivots) — lever intact; do not judge funnel
    geometry on loaded-box separate-process medians. Next: same
    hop-elimination for Cholesky (1.7x) and QR-1.3x panel tails, then
    the contiguous-panel restructure.
30. Cholesky tiny-pool funnel SHIPPED 2026-09-27 (default-on f64,
    `SONIUM_CHOL_TINY=0` opt-out): same hop-elimination recipe as LU
    lever 29, via `run_tiny` at the four top-level funnels
    (unchecked, Hermitian, checked, in-place). Trial mechanics catch:
    gating inside `factorize_owned` never fired (nested inline
    no-op) — the gate must live at the funnel. OB-free
    order-rotated: 1024 on 4.18 vs off 5.97 (1.43x), 2048 on
    14.08-14.57 vs off 21.56-21.90 (1.51-1.56x, three runs);
    residuals equal (256/1024), maxdiff 1.7e-18 at 2048, no pivots
    in Cholesky. Notably the tiny arm is BITWISE-identical across
    runs (pool geometry moves TRSM/SYRK splits only vs the old
    path). Envelope: f64-only via `chol_prefers_tiny_pool`, n>=1024
    with 64MB cap, inert at global<=16. dpotrf-2048 now ~1.2x vs OB
    (14.1 vs 12.2, was 1.7-1.8x). Direct suite green both arms (273),
    core 47 green, clippy clean at edited lines; f32 spotrf
    untouched (2.9ms). Next: QR-1.3x panel tails, then the
    contiguous-panel restructure for LU's remaining 1.36x.
31. QR tiny-pool funnel SHIPPED 2026-09-27 (default-on f64,
    `SONIUM_QR_TINY=0` opt-out): QR had no funnel at all (cold caller
    + nested hops on every GEMM scope), so this is both a temperature
    fix and hop elimination — single choke at `factorize_blocked_ws`
    via `run_tiny`. OB-free order-rotated: 512 on 16.4-17.0 vs off
    20.8-21.6 (1.23-1.29x), 1024 on 49.3-50.1 vs off 72.3-79.1
    (1.47-1.58x), 2048 on 259-263 vs off 290-312 (1.10-1.21x) —
    three runs each, and BITWISE-identical Q/R at every size (QR's
    GEMM splits align across pools; residuals identical too). Trial
    mechanics: none needed beyond the LU/Cholesky pattern. Envelope:
    f64-only via `qr_prefers_tiny_pool`, ncols>=512 with 64MB cap,
    inert at global<=16. Direct suite green both arms (273), clippy
    clean at edited lines. LAPACK scoreboard (OB-free f64):
    dgetrf-2048 1.36x, dpotrf-2048 ~1.2x, dgeqrf-2048 ~1.1-1.2x
    (was 1.6x/1.7x/1.3x). Remaining: contiguous-panel restructure
    for LU, vendor-TRSM kernel work, GPU-leg audit.
32. GPU-leg clean-room audit 2026-09-27 (RTX PRO 6000 idle, 0% util):
    repointed /tmp/girprobe at the current checkout (was pinned to a
    stale worktree) and closed the two queued items. 256² jacobi3-0.75
    clean: CPU 6237ms (491 inner) vs CUDA 286ms (492 inner, res
    6.3e-13) = 21.8x GPU win, zero variance (286/286/286) — replaces
    the contaminated 709ms median. Damping RESOLVED: 0.75 beats 1.0
    (491/492 vs 553/555 inner iters; CPU 6237 vs ~7500, CUDA 286 vs
    ~324; same residual class) — 0.75 confirmed, not overlapping.
    128² re-confirm 4.4x (CPU 470/327iter vs CUDA 106/312iter, both
    converged ~7e-13; last-ulp backend trajectory split is expected).
    CUDA suites on current tree: 531 lib + 68 parity + 5
    device_preconditioner + all integration targets green (one
    initial failure was a missing SONIUM_OUTPUT_DIR env for a C4
    evidence test — environmental, passes with it set). GPU operating
    window now measured clean at 128²/256²/1M-dof: resident-data
    mixed-precision GMRES-IR on cuBLAS/cuSPARSE backend. CPU leg
    stands at dgetrf 1.36x / dpotrf ~1.2x / dgeqrf ~1.1-1.2x.

## solvers-202609 Stage 1 (x86)

Validator record, 2026-09-27, WT @ `d5dcf304` + uncommitted family work.
Platform: Threadripper PRO 3995WX (128 logical), Linux x86_64, rustc 1.98.1,
RAYON_NUM_THREADS=16, unpinned, schedutil governor. Box noisy throughout
(load 3-26, co-tenants incl. `sonium_sim_pcfan` ~17 cores mid-campaign):
all ratios below are same-window interleaved medians; vendor-side drift is
called out per cell. CUDA out of scope for this stage. Worker notes in
`$HOME/tmp/solvers202609/{direct,sparse,iterative,preconditioners,eigen,
opt-nonlinear,ti-core-facade}.md`; validator probes in
`$HOME/tmp/solvers202609/{eigprobe,luab,spmvprobe,pcbench,rosen_smoke}`
+ `vendor_eigh.py`. Full report: `$HOME/tmp/solvers202609/validation.md`.
D3 gate: vendor ratio <= 1.2x.

### Verdict table (validator-measured)

| Cell | Size | Sonium | Vendor | Ratio | D3 | Residual gate | vs 09-26 |
|---|---|---|---|---|---|---|---|
| sparse LU total (simplicial) | grid 32 | 6.23ms (V run) | SuperLU 2.11ms | 2.95x (1.91x worker window) | FAIL | 1.05e-14 PASS | 12.10x -> ~2x improved |
| sparse LU solve stage only | grid 32 | 0.068ms | (in SuperLU total) | — | — | same gate PASS | 14.79ms -> 0.07ms (217x) |
| sparse LU total (simplicial) | grid 40 | 12.21ms (worker) | SuperLU 5.64ms | 2.16x | FAIL | 2.60e-15 PASS | 11.07x -> 2.16x improved |
| dense LU dgetrf-2048 nb32 | 2048 | 40.96ms | OB-8T 26.24ms | 1.56x | FAIL | tests PASS | 1.63x record reproduced |
| dense LU dgetrf-2048 nb64 (default) | 2048 | 47.82ms | OB-8T 26.24ms | 1.82x | FAIL | bitwise==nb32 PASS | CONTESTED, see log |
| SpMV unchecked (new API) | 200k/2.6M nnz | 0.495ms, 90.6GB/s | roofline (triad 92.0GB/s) | 98% of roof | N/A (no vendor) | bitwise PASS | new capability |
| SpMV checked (unchanged default) | 200k/2.6M nnz | 2.14ms, 20.9GB/s | — | 4.34x internal gap | N/A | — | validation cost quantified |
| symmetric eig (D&C) | n=300 | 10.1-10.6ms | dsyevd 7.5ms | 1.35-1.41x | FAIL (narrow) | orth 3e-15, relres 6e-17 PASS | 1.93x -> 1.38x improved |
| symmetric eig (D&C) | n=200 | 4.9-5.1ms | dsyevd 3.6ms | 1.34-1.39x | FAIL (narrow) | same class PASS | 1.50x -> 1.36x improved |
| L-BFGS-B vs SciPy | n=100/1000 | 0.087/0.42ms | 2.67/4.62ms | 0.033x/0.09x | PASS | tighter fun PASS | new cells |
| AMG-CG Poisson 3D n=32 | 32768 dof | 12 iters, conv | BoomerAMG (iters parity target) | iters pinned | wall varies w/ box | tests PASS | no algorithmic regression |
| Chebyshev deg-4 apply | n=16 | 0.1005ms | self-checked 0.1403ms | 1.40x internal | N/A | maxdiff 2.2e-16 PASS | validate-once win |
| iterative / TI / core / sys / facade | — | strictly-less-work | none | — | N/A | 138 suites PASS | no regression |

Reading:
- Sparse Direct is the Stage-1 headline: the solve-stage level-dispatch
  blowup is gone (grid-32 solve 14.79ms -> 0.07ms, 217x; total 12.10x ->
  ~2x). Remaining ~2x is 100% factor stage (ordering+symbolic+numeric,
  all serial) — follow-up work, not a solve problem. D3 FAIL on total
  stands, honestly: ~2x > 1.2x.
- The sparse total ratio is vendor-unstable: SuperLU grid-32 measured
  3.71ms (worker window) vs 2.11ms (validator window) on identical code
  -> 1.91x vs 2.95x. Judge the solve/factor split, not the total ratio.
- Eigen improved 1.93x -> ~1.38x same-matrix; gates pass with margin.
  Still short of 1.2x; residual levers are compute-owned (per-call GEMM
  dispatch, small-square decomposition) + secular AVX2.
- Optimization passes D3 trivially (11-30x faster than SciPy, tighter
  final objective). Nonlinear vendor (PETSc SNES) unprobed — no cell.
- Iterative/TI/core/sys/facade have no vendor cells by design; verdict
  is tests-green + no-regression (pcg converges 12 iters in pcbench,
  exercising the iterative crate end to end).

### Gates: tests + clippy (validator-run, RAYON=16)

| Crate | Suites ok | Failures |
|---|---|---|
| core 9, sys 2, sparse 15, eigen 14, direct 29, precond 25, iterative 16, opt 12, nonlinear 8, TI 8 | 138 | 0 |
| facade `--lib --tests` (319 tests) | 43 | 0 |

`cargo clippy` (lib scope, all 11 solver crates): 0 errors, pre-existing
warnings only. BLOCKER (pre-existing, not Stage-1): bare
`cargo test -p sonium-solvers` and `--all-targets` clippy fail on
`examples/amg_cuda_probe_tmp.rs` (E0432 `sonium_compute::CudaBackend`;
file unmodified in WT, CUDA out of scope). Facade gate above uses
`--lib --tests` to exclude it.

### Falsified / contested trials (validator)

1. CONTESTED: LU `nb=64` default flip (lu.rs carry-over, pre-Stage-1).
   In-code note claims 6/6 pairs nb64 faster at 2048 (~35.0 vs ~37.6).
   Validator `luab` (in-process interleaved nb64/nb32/vendor, factor-only,
   clones outside timers, 2 runs on loaded box): nb32 won 11+/11+ pairs
   (47.8/41.0 and 53.0/48.3 medians), checksums bitwise-identical across
   arms. nb32/no-regression anchors fine (41.0 vs 09-26 record 41.9ms).
   Verdict: load-dependent or wrong — DO NOT lock the default on this
   evidence; re-run `nbab` on a quiet box. Kill-switch `SONIUM_LU_NB=32`
   retained. (Validator is PERF.md-only; no source change made.)
2. FALSIFIED: cross-matrix eigen comparison. First validator A/B (own RNG
   matrices) gave 1.79x @300 vs the worker's 1.43x estimate — same-matrix
   A/B (numpy-dumped `eigA_{200,300}.bin`) gives 1.35-1.41x, confirming
   the worker. D&C timing is deflation/value-dependent (~30% swing across
   matrices). Discipline: eigen A/B must share the input matrix.
3. CONFIRMED unstable: sparse-total vendor ratio (see Reading). Always
   report the factor/solve split alongside the total.
4. Methodology catch (own probe): DenseMatrix is row-major — the LAPACK
   vendor arm needs an explicit row->column transpose (done in `luab`;
   without it the vendor comparison is void, cf. the 09-26 TRSM
   uplo-enum incident).

### Remaining levers (Stage-2 candidates, not Stage-1 gaps)

- Sparse factor stage (~2x): ordering+symbolic+numeric all serial.
- Eigen ~0.2x to gate: compute-owned GEMM dispatch + small-square work.
- ILUT setup 1122ms (algorithmic sorted insertion) — own campaign.
- Nonlinear PETSc SNES cell unprobed; iterative PETSc KSP cell unprobed.
- `amg_cuda_probe_tmp` example breakage (pre-existing, CUDA scope).

### FAIL-cell follow-up (validator, 2026-09-27 PM)

Follow-up validation of the three Stage-1 FAIL cells after worker
re-work (validator is PERF.md-only; no source edits). Same platform,
RAYON_NUM_THREADS=16, unpinned, schedutil; box quiet-ish this window
(load 2-5). Discipline: interleaved same-window A/B, rep-trajectory
(rep0 discarded), eigen shares input matrices across arms
(`eigA_{200,300}.bin`), sparse reports factor/solve split, dgetrf is
in-process nb-interleaved at 1024/2048 with per-arm residual gates.
Probes: `$HOME/tmp/solvers202609/luabv` (new),
`eigprobe`+`vendor_eigh_bin.py` (new, same-matrix),
`scripts/direct_solver_perf.sh` (official harness, REPEATS=5).
Full report: `$HOME/tmp/solvers202609/failcells-validation.md`.

| Cell | Size | Sonium | Vendor | Ratio | D3 | Gate | vs Stage-1 V |
|---|---|---|---|---|---|---|---|
| dgetrf default (=nb48 f64) | 2048 | 37.0-37.7ms | OB-8T 25.6-26.0ms | 1.44-1.46x | FAIL | relres 1.2e-15 PASS | 1.82x (nb64-def) -> 1.45x |
| dgetrf nb32 | 2048 | 40.5-43.0ms | same | 1.58-1.65x | FAIL | 2.0e-15 PASS | 1.56x reproduced |
| dgetrf nb48 explicit | 2048 | 38.8-39.9ms | same | 1.51-1.54x | FAIL | 1.2e-15 PASS | new arm |
| dgetrf nb64 | 2048 | 41.0-43.5ms | same | 1.60-1.68x | FAIL | 8.1e-16 PASS | flip REVERTED |
| dgetrf default | 1024 | 9.3-9.5ms | OB-8T 7.5ms | 1.24-1.27x | FAIL (narrow) | 8.0e-16 PASS | new size |
| symmetric eig (D&C) | n=200 | 3.86-3.89ms | dsyevd 3.60-3.63ms | 1.07x | PASS | orth 2.8e-15 PASS | 1.36x -> 1.07x |
| symmetric eig (D&C) | n=300 | 9.48-9.50ms | dsyevd 7.49-7.53ms | 1.26-1.27x | FAIL (narrow) | orth 4.2e-15 PASS | 1.38x -> 1.27x |
| sparse LU total (simplicial) | grid 32 | 3.809 (f 3.586 + s 0.057) | SuperLU 2.098ms | 1.82x | FAIL | 1.05e-14 PASS | ~2-2.95x -> 1.82x |
| sparse LU total (simplicial) | grid 40 | 5.227 (f 5.150 + s 0.098) | SuperLU 3.342ms | 1.56x | FAIL | 2.60e-15 PASS | 2.16x -> 1.56x |

Reading:
- dgetrf: the nb64 default flip is REVERTED (shared default 32
  again); the new threaded-f64 nb48 default is CONFIRMED (~1.04x
  over nb32 at 2048, nb48-beats-nb32 8-9/9 pairs in each of 4 clean
  windows). Default==nb48 geometry proven by code path plus
  identical residuals (1.198e-15 both sizes). D3 still FAILs
  (1.45x vs 1.2x); the gap is compute-owned kernels, not panel
  geometry. nb32-vs-nb64 pairs: nb32 wins 6-8/9 at 2048; at 1024
  nb64/nb48 both beat nb32 5-7/7 (small-n loves fat panels;
  f64 default 48 covers it).
- Eigen n=200 now PASSES D3 (1.07x); n=300 improved 1.38x ->
  1.27x but still FAILs narrowly. Halve-join determinism holds
  (identical FNV hashes run-to-run). orth@300 moved 2.665e-15 ->
  4.219e-15 (ulp-level from the shipped triu loop; same order,
  gates pass with margin).
- Sparse: factor stage improved 1.30x at grid32 vs the worker's
  own 15:00Z smoke (4.667 -> 3.586ms, vendor stable +-2.5% so the
  delta is real); grid40 flat (5.207 -> 5.150). Factor nnz
  unchanged (ordering preserved), residuals identical. Remaining
  gap is 100% serial factor (numeric+ordering+symbolic). The
  grid16 row (2.30x) is noise-dominated sub-ms trivia (supernodal
  factor column even exceeds its total) — not decision-grade.

Gates (validator-run, RAYON=16): `cargo test -p sonium-solvers-direct`
29 suites ok / 0 fail; `-p sonium-solvers-eigen` 14 / 0; `-p
sonium-solvers-sparse` 15 / 0. `cargo clippy -p ... --lib` on the
three crates: 0 errors. Direct/sparse warnings are pre-existing;
eigen grew 4 NEW `needless_borrow` lints on worker-added lines
(lobpcg.rs 1625/1626/1760/1761, `&vcm` -> `vcm`) — worker to fix
(one-token each, validator cannot edit source).

Falsified / confirmed trial log (follow-up):
1. CONFIRMED + REVERTED: nb64 flip. Falsified by worker
   re-adjudication (24/26 pairs nb32) and validator (nb32 beats
   nb64 6-8/9 pairs, slower medians every clean window).
2. CONFIRMED (new, pre-existing, perf-only): `FACTOR_POOL` slot
   pollution poisons mixed-size single-process runs. lu.rs pool
   keeps 2 buffers/type with exact-capacity match: a (1024,2048)
   process fills both slots with 8MB buffers, so every 2048 rep
   pays the fresh-mmap+fault rep0 tax (64.9 vs 40.5ms, 1.60x);
   every poisoned rep ~= a clean rep0 (62-64ms); reverse effect
   0.85ms == the documented 0.8ms pooling value; (2048,2048)
   both fast; separate processes heal; vendor flat throughout.
   Untouched by Stage-1/FAIL edits (pre-existing); direct-owned.
   CONSEQUENCE: multi-size single-process LU A/B is VOID — one
   size per process (or 2048-first). All ratios above are clean.
3. FALSIFIED: compact-pool-init as the (2) mechanism
   (`SONIUM_COMPACT=0` still poisons).
4. FALSIFIED: heat/power/time as the (2) mechanism ((2048,2048)
   both fast; vendor arm flat in every run).
5. ARTIFACT (open, verdict-irrelevant): default arm ~5% faster
   than explicit nb48 despite identical geometry (code-proven +
   identical residuals). Suspected rotation asymmetry vs
   OpenBLAS L3 eviction (default never runs right after the
   vendor arm; nb48 does 2x/9). Reported as a 37-39ms range;
   both ends FAIL D3 the same way.
6. EXPLAINED: Stage-1 vendor drift was LOAD. Quiet-box vendors
   stable: SuperLU +-2.5% across 1h-apart runs, dsyevd 7.5ms,
   OB dgetrf 25.6-26.0ms.
7. NOT DECISION-GRADE: sparse grid16 row (see Reading).

### ParNum Phase 0 (parallel-sparse-numeric census + conditional micro-fix, 2026-09-27)

Worker record, same platform, RAYON_NUM_THREADS=16, noisy box (load
2-23): all ratios are same-window interleaved medians, rep-trajectory
(rep0 discarded), factor/solve split reported. Scope:
`sonium-solvers-direct` supernodal path only; CUDA and `sonium-compute`
untouched. Matrix family: 2D nonsymmetric convection-diffusion (official
`direct_solver_perf` generator) at grids 16/32/40/128; reused-symbolic
numeric uses pivot 1.0 (both one-shot defaults) instead of the official
0.1 for a clean analyze/numeric split. Full record:
`$HOME/tmp/parnum/phase0.md`; harnesses + raw logs in `$HOME/tmp/parnum/`
(`parnum_bench`/`parnum_census`/`parnum_ab`, `raw/`).

Census method: offline exact replica of symbolic + reach marking (public
`colamd_ordering` + replicated etree/supernode/panel logic), cross-checked
bit-exact against runtime at all sizes (etree, panel counts, per-panel
live contributors, reach steps 1808/9093/15235/209146). Needed flags are
pivoting-independent, so the panel contributor DAG below is exact.
Runtime phase/GEMM detail came from temporary instrumentation, fully
reverted (0 remnants, verified).

| Grid | n | Panels | Chunks | DAG levels | maxW | meanW | Etree depth |
|---|---|---|---|---|---|---|---|
| 16 | 256 | 18 | 92 | 16 | 2 | 1.12 | 107 |
| 32 | 1024 | 73 | 373 | 24 | 9 | 3.04 | 220 |
| 40 | 1600 | 110 | 579 | 33 | 16 | 3.33 | 290 |
| 128 | 16384 | 1134 | 5724 | 85 | 171 | 13.34 | 1137 |

Level-width tables (panels per longest-path level over the exact
contributor DAG):
- 16: [2 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1]
- 32: [9 5 4 2 3 2 2 3 5 6 7 5 4 2 3 3 1 1 1 1 1 1 1 1]
- 40: [16 11 9 3 1 4 2 1 2 4 11 5 2 4 4 3 4 4 3 2 2 1 2 1 1 1 1 1 1 1 1 1 1]
- 128: [171 158 154 79 106 52 33 32 32 41 29 23 19 15 12 13 10 10 9 9 7 6
  4(x9) 3(x8) 2(x8)] + 38 width-1 tail levels (>=2:47, >=4:31, >=16:13 of 85)

Work-weighted Amdahl ceiling (offline levels x runtime per-panel times,
perfect intra-level split, zero sync — optimistic): 1.08x @16 any P;
1.94/2.04/2.04x @32 P4/8/16; 1.90/2.00/2.03x @40; 2.04/2.33/2.48x @128.
~30% of panel time sits in width-1 levels at 32/40/128 (deep COLAMD
etree chains); realistic level-scheduled speedup is strictly below
(barriers, imbalance, shared-workspace + row-perm privatization needs a
redesign, not a `par_for`).

Numeric phase split (pre-fix, warm): reach 2.6-3.2%, apply ~42% (GEMM
~18%), in-panel column loop ~32%, gather ~3%, pack ~6%. GEMM dispatches
252/1262/2092/25429 with mean rectangles 6.9-29.1 x ~14.5 x 3.0-4.7
(m x n x k), 0.51-0.76us/dispatch (~14 GFLOP/s @128, memory-bound);
TRSM vacuity 11-13%. COLAMD is 63-71% of analyze; analyze is 11-25% of
one-shot (shrinks with size).

Supernodal-serial vs simplicial-serial totals (medians, residuals PASS):
16: 1.63x behind (trivia); 32: 1.30x behind (replicated 1.31x);
40: 1.26x behind (replicated 1.27x); ordering-excluded numeric gap
1.20-1.27x. Crossover above: @128 supernodal is 1.65x AHEAD
(138.6 vs 228.5ms total; 0.54x ordering-excluded). Per-rep sup>simp
14/14 pairs at 32/40. Factor nnz identical across backends
(6060/38988/65924/1269156).

Phase 1 TRIGGERED (>10% behind at decision sizes): shipped two
smallest-correct per-apply fixed-cost reductions in
`sparse_supernodal.rs` (+83/-2 incl. test): (1) reach-mark dead-start
skip (`j >= panel_start`, provably flag-identical — etree parents
ascend); (2) chunk `update_buf` growth-only reuse (dead re-zeroing
removed; `mat_mul_nn` overwrites `C` by contract). Bytes-unchanged:
golden test `chunk_update_buffer_reuse_preserves_factor_bytes` passes
identically pre/post (1155/1299 nnz + FNV value-bits), checkpoint bytes
old-vs-new identical (1,670,630 B @40), nnz + residuals identical all
sizes. Same-window A/B (20 pairs, temporary switch, reverted): 0.9805x
@32 (20/20), 0.9845x @40 (15/20), 0.9786x @128 (14/20) — consistent
~2% win; post-fix sup/simp 1.267x @32, 1.237x @40. Gates:
`cargo test -p sonium-solvers-direct` 29 suites / 0 fail,
`cargo clippy --lib` 0 errors. FALSIFIED: fixed-cost reduction cannot
rescue decision-size parity (gap is structural dense-rectangle work, not
fixed cost); true reach-mark reuse needs redesign (rejected).

Gate recommendation — usable width at n>=1600? NO. Decision sizes:
optimistic 2.0x ceiling on ~80% of one-shot still leaves sup at ~1.3x
vs SuperLU @40 (D3-FAIL), realistic <1.5x on numeric; sup starts 1.24x
behind best serial. Scale (n=16384): 2.5x optimistic ceiling, needs
numeric redesign; no D3 cell there — defer. Unmeasured hypothesis: ND
ordering may widen the DAG vs COLAMD.

### ParNum Phases 4+5 (thread-determinism pins + SuperLU vendor A/B, 2026-09-27)

Worker record, same platform, RAYON_NUM_THREADS=16, unpinned, box
moderate (load 3-11 across windows). Scope: `sonium-solvers-direct`
only (`tests/sparse_lu_determinism.rs` new; no product edits — probe
found no failures to fix). No parallel numeric exists in-tree (Phase 0
gate was NO; the only thread-sensitive paths are the level-set
triangular solve and `mat_mul_nn` internals), so Phase 4 pins
bit-determinism across thread counts rather than a parallel/serial API
pair. Full record: `$HOME/tmp/parnum/validation.md`; harnesses + raw
logs in `$HOME/tmp/parnum/` (`parnum_probe`, `parnum_ab_superlu`,
`raw/probe_*`, `raw/ab_*`, `raw/test*.log`).

Phase 4: 6 maintained tests pin exact checkpoint bytes (L/U structure
+ values + row/col perms), decoded pivot sequences (row_perm FNV),
symbolic-reuse == one-shot bytes, checkpoint round-trip idempotence,
error kind+payload parity (simplicial / supernodal / symbolic-reuse),
preconditioner-mode repair-vs-reject behavior, and f32/Complex64
goldens, over a corpus of conv-diffusion grids 8/12/32, exact-tie
all-ones tridiagonals (4/7), zero-pivot / dup-rows / 1e-16-pivot /
non-finite fixtures. A scratch probe verified RAYON=1 vs =16 outputs
byte-identical (checkpoint FNVs, pivot FNVs, error payloads,
residuals); the goldens then make the dual-count suite runs the
cross-thread-count equality proof. Gates: `cargo test -p
sonium-solvers-direct` 30/30 suites green at BOTH RAYON=1 and =16
(29 prior + `sparse_lu_determinism`), `cargo clippy` 0 errors and 0
lints on the new file (remaining warnings pre-existing).

Phase 5 vendor A/B (`parnum_ab_superlu`: same-window interleaved
5-arm rep-trajectory, 1 warmup + 14 measured, rep0 discarded,
round-robin arm rotation; ours = factor/solve split, SuperLU 6.0.1
dgssv total via the official C shim — the shim exposes no vendor
factor/solve split; residual gate 1e-10 per arm per cell, all PASS at
1e-15 class; factor nnz identical across backends every cell:
6060/38988/65924/1269156). Two windows per cell:

| Grid | n | Simplicial (f+s ms) | Supernodal (f+s ms) | SuperLU (ms) | simp/SLU | sup/SLU | D3 (best arm) |
|---|---|---|---|---|---|---|---|
| 16 | 256 | 0.720+0.020 / 0.735+0.020 | 1.188+0.019 / 1.211+0.019 | 0.669 / 0.681 | 1.11x / 1.11x | 1.80x / 1.81x | PASS narrow (simp; trivia-size caveat) |
| 32 | 1024 | 3.348+0.072 / 3.496+0.072 | 4.282+0.077 / 4.369+0.074 | 2.109 / 2.205 | 1.62x / 1.62x | 2.07x / 2.02x | FAIL (simp) |
| 40 | 1600 | 5.727+0.110 / 5.757+0.112 | 6.566+0.114 / 6.583+0.118 | 3.186 / 3.211 | 1.83x / 1.83x | 2.10x / 2.09x | FAIL (simp) |
| 128 | 16384 | 226.65+3.62 / 216.01+3.58 | 108.96+3.53 / 107.97+3.52 | 61.37 / 58.61 | 3.75x / 3.75x | 1.83x / 1.90x | scale-only, no D3 cell (sup 1.83-1.90x) |

Reading: ratios replicate across windows (<=0.05x drift; @128 the
vendor arm drifted -4.5% while ours held, moving sup 1.83->1.90x —
vendor-side noise, verdict-immaterial). Factor stage is 96.8-98.4% of
our totals: the gap is ~100% serial factor, as before. Comparison vs
the FAIL-cell record (different methodology: official sequential
3-rep medians vs interleaved 14-rep here; cross-batch absolute-ms
comparison is noise): @32 simp 1.62x here vs 1.82x there (vendor
anchors agree 2.11 vs 2.10); @40 simp 1.83x here vs 1.56x there
(vendor drifted 3.34->3.19, ours +11% — interleaved arms share L3
with dgssv while sequential arms run warm-cache; both records FAIL
D3 the same way). Grid16 simp 1.11x PASS is narrow and sub-ms —
decision weight low (same trivia caveat as the FAIL-cell record).

Falsified-trial log (Phases 4+5):
1. FALSIFIED: "supernodal (BLAS-3) beats simplicial at decision
   sizes" — sup is 1.24-1.28x BEHIND simp at 32/40 (replicated),
   1.63x behind at 16; crossover only at scale (@128 sup is 2.05x
   ahead of simp). Structural (dense-rectangle work below n~4-8k),
   consistent with Phase 0/1.
2. FALSIFIED: "triangular-solve parallelism moves D3 sparse cells"
   — solve is 1.6-3.2% of our totals; even a perfect solve can only
   shift ratios ~0.03x. The level-set path is validated (bitwise
   gate test + round-trip-through-parallel-path test) but is not a
   D3 lever.
3. CONFIRMED: thread-count determinism — checkpoint bytes, pivot
   sequences, and error payloads identical at RAYON 1 vs 16 across
   the corpus (probe diff clean); no `mat_mul_nn` reduction-order
   nondeterminism observed.
4. CONFIRMED: error-path parity — SingularMatrix/NonFiniteMatrix
   kind AND payload (column / row+col) identical across
   simplicial/supernodal/symbolic-reuse; preconditioner mode still
   rejects exact-zero pivots and repairs the 1e-16 pivot (solves
   3e16 correctly).
5. NOT PURSUED (out of scope): vendor factor/solve split (C shim
   exposes dgssv total only), ND-ordering DAG-width hypothesis
   (Phase 0 carry), parallel numeric redesign (Phase 0 gate NO).

### ParNum128 (larger-n retry: grid128 + n=16384 Laplacian vs SuperLU, 2026-09-28)

Worker record, same platform, RAYON_NUM_THREADS=16, unpinned, box
noisy (load 2.4 at start, 5.5-13.3 across A/B windows). VALIDATE
ONLY: no WT product edits (PERF.md append only), no numerics
redesign; the only code touched is the `$HOME/tmp/parnum`
harness (+38 lines: pure 5-point Laplacian generator +
`convdiff|laplacian` switch, default unchanged). CUDA and
`sonium-compute` untouched. Full record:
`$HOME/tmp/parnum/parnum128-validation.md`; raw logs in
`$HOME/tmp/parnum/raw/` (`ab128_*`, `test{1,16}_128`,
`clippy{1,16}_128`, `probe_{1,16}_128`; prior-chain logs
preserved).

Guardrail re-run (suite NOT weakened, no file touched): `cargo
test -p sonium-solvers-direct` exit 0 with 30/30 `test result:
ok` suites at BOTH RAYON=16 and RAYON=1; determinism suite 6/6
PASS at both counts (golden checkpoint bytes + pivot FNVs hold
under both — the cross-thread-count equality proof); `cargo
clippy --all-targets` 0 errors at both counts, 0 lints on the
guardrail file; rebuilt `parnum_probe` at RAYON 16 vs 1 is
byte-identical except the threads label (checkpoint FNVs, pivot
FNVs, error payloads, residuals all equal).

Vendor A/B (same `parnum_ab_superlu` 5-arm interleaved
rep-trajectory as Phases 4+5, 1+14 reps, rep0 discarded;
residual gate 1e-10 per arm per cell, all arms 1e-15 class:
PASS; factor nnz identical across backends every cell: 38988
@32, 65924 @40, 1269156 @128 both families). Two windows per
cell:

| Cell | n | Simplicial (f+s ms) | Supernodal (f+s ms) | SuperLU (ms) | simp/SLU | sup/SLU | D3 (best arm) |
|---|---|---|---|---|---|---|---|
| 128 convdiff | 16384 | 215.75+3.50 / 227.62+3.51 | 107.03+3.53 / 106.90+3.51 | 58.01 / 62.04 | 3.78x / 3.73x | 1.91x / 1.78x | FAIL (sup; scale-only, no D3 cell) |
| 128 laplacian | 16384 | 216.56+3.51 / 217.19+3.55 | 107.86+3.59 / 107.84+3.59 | 58.49 / 58.15 | 3.76x / 3.80x | 1.91x / 1.92x | FAIL (sup; scale-only) |
| 32 convdiff | 1024 | 3.313+0.073 / 3.321+0.073 | 4.260+0.076 / 4.287+0.076 | 2.029 / 2.026 | 1.67x / 1.68x | 2.14x / 2.15x | FAIL (simp; no regression vs 1.62x) |
| 40 convdiff | 1600 | 5.772+0.114 / 5.770+0.119 | 6.698+0.122 / 6.734+0.121 | 3.243 / 3.239 | 1.82x / 1.82x | 2.10x / 2.12x | FAIL (simp; exact replicate of 1.83x) |

Reading: factor stage is 96.8-98.5% of our totals every window
— the gap is ~100% serial factor, as in every prior record.
Sup-vs-simp: 1.28-1.29x behind @32, 1.16x behind @40, 1.97-2.09x
AHEAD @128 both families (crossover replicates the prior 2.05x;
the @40 1.16x replicates this harness's prior @40 1.14x, not the
Phase-0 8-arm 1.24x — methodology note, verdict-immaterial).
@128-convdiff sup drifted 1.91x->1.78x across windows (0.13x,
wider than the prior <=0.05x): vendor arm +6.9% while our sup
arm held ±0.1% — vendor-side noise under load, verdict-immaterial
(FAIL by 0.6x+ either way); the Laplacian @128 pair replicates
to 0.01x and the 32/40 pairs to <=0.02x. No larger-n cell meets
D3 <= 1.2x.

Falsified-trial log (ParNum128):
1. FALSIFIED: "symmetric Laplacian structure closes the vendor
   gap vs conv-diffusion" — @128 sup/SLU is 1.91/1.92x (lap)
   vs 1.91/1.78x (conv), our sup arms within 1%, fill identical
   (1269156); symmetry buys COLAMD nothing here. The deficit is
   numeric-kernel throughput, not ordering.
2. CONFIRMED (no regression): decision-size FAIL verdicts
   replicate — @32 simp 1.67-1.68x vs 1.62x (+0.05x drift, same
   class), @40 simp 1.82x vs 1.83x (exact), @128-conv sup band
   1.78-1.91x vs 1.83-1.90x.
3. CONFIRMED: thread-count determinism still holds — 6/6
   goldens at RAYON 1 and 16, probe bytes identical; zero
   failures, zero product/test edits needed.
4. NOT PURSUED (out of scope): grids above 128, ND-ordering
   DAG-width hypothesis (Phase 0 carry), parallel numeric
   redesign (Phase 0 gate NO — nothing here reopens it).

### ParNum Hands-On Slice 1 (static waves + serial wave execution, 2026-09-28)

Static touched-row (ST) wave schedule in analyze (`panel_numeric_waves`):
per-column reach census unioning scatter rows, greedy levels over
contributor edges (explicit driver-exact replay) plus ST overlap, floored
nondecreasing so serial wave execution is exactly ascending panel order.
Driver refactor: per-panel local L/U staging + ascending merge,
panel-indexed completed storage, pre-sized chunk map. Default order is
waves; `SONIUM_SPARSE_NUMERIC_ORDER=ascending` restores legacy order.

Wave census (convdiff, COLAMD):

| grid | n | panels | waves | max width | multi-panel waves |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 18 | 16 | 2 | 2 |
| 32 | 1024 | 73 | 27 | 9 | 16 |
| 40 | 1600 | 110 | 39 | 14 | 19 |
| 64 | 4096 | 282 | 67 | 38 | 35 |
| 128 | 16384 | 1134 | 214 | 163 | 113 |

Widths grow with n (leaves parallel, root serial); real within-wave
parallelism exists at 40+. Full direct suite green (278 lib incl. 3 new
wave tests), determinism goldens UNCHANGED (6/6), clippy 0 errors and no
new warnings — the refactor + reorder are result-neutral by construction.

Falsified-trial log (Slice 1):
1. FALSIFIED: "unconstrained ST waves preserve factors" — diagonal-side
   swap originals sit outside every static touched set, so out-of-order
   execution drops true contributors (false `SingularMatrix(309)` on
   conv32) and reorders L storage (grid16). Fix: nondecreasing floor
   (serial order == ascending) + explicit contributor edges; ST overlap
   does NOT subsume contributor edges (shared row sits in the starter
   column, not necessarily the contributor's panel).
2. CONFIRMED: wave schedule is a pure function of the pattern
   (analyze-twice identical), respects all contributor edges on
   grids 8/12/16, and wave-vs-ascending factors are field-identical on
   the corpus (f64 + c64, thresholds 1.0/0.1).

### ParNum Hands-On Slice 2 (parallel waves + validation + redo, 2026-09-28)

Rayon within-wave parallelism with exact recorded swap/touched-set
validation, clean-prefix commit + serial suffix redo, and whole-wave
serial redo on any panel error (first-error precedence preserved).
`ParallelWaves::{Auto,Always,Never}` + `SONIUM_SPARSE_NUMERIC_PAR_MIN_N`
floor + `SONIUM_SPARSE_NUMERIC_DEBUG` counters; `Always` bypasses all
gates for deterministic tests.

Root-cause fix (drop-on-error): an errored panel's early return skips
the workspace clear, so unconditionally recycling its workspace poisoned
later checkouts (stale `active` flags suppress re-registration of
scattered rows; caught as panel-69 post-scatter divergence, 733 vs 62
active rows at grid80). Only success-path workspaces return to the pool;
error workspaces are dropped (rare path, pool regrows lazily). Grid80
joins the parallel-vs-serial bitwise corpus as regression cover.

Correctness: full direct suite green at RAYON 1 and 16 (279 lib incl.
`parallel_waves_match_serial_factors_bitwise` over grids 8/12/16/24/80
x thresholds 1.0/0.1), determinism goldens 6/6 both counts, clippy
`--lib --tests` 0 errors and no new warnings. All TEMP-RACE-PROBE code
removed (fresh-ws / zero-values / audit / shadow / trace / GEMM-serialize
/ force-redo).

A/B vs SuperLU 6.0.1 (convdiff COLAMD, RAYON_NUM_THREADS=16,
one-size-per-process, rep0 discarded, residuals all ~1e-15 PASS):

| grid | n | sup serial-waves total | sup parallel total | sup vs SuperLU (serial) |
|---:|---:|---:|---:|---:|
| 32 | 1024 | 5.02ms | 7.59ms | 2.38x |
| 40 | 1600 | 7.90ms | 11.20ms | 2.35x |
| 80 | 6400 | 41.40ms | 60.76ms | 2.70x |
| 128 | 16384 | 143.79ms | 162.62ms | 2.19x |

(Serial-waves vs ascending in-window at 128: 140.29 vs 140.72ms factor —
the Slice-1 reorder is perf-neutral. The 1.78-1.92x ParNum128 baseline
vs 2.19x here is cross-window host noise, not a code regression.)

Redo storms (strict partial pivoting, threshold 1.0): grid32 14/16 par
waves redo, grid40 14/19, grid128 38/113 par waves (147 violations, 228
redo panels of 1134). Diagonal-side swap originals sit outside every
static touched set, so same-wave panels genuinely interfere through the
dynamic mapping at a 34-88% wave rate; attempt-plus-redo waste outweighs
the wave parallelism and parallel runs ~1.1-1.45x SLOWER than serial at
every measured size. Default `SONIUM_SPARSE_NUMERIC_PAR_MIN_N` floor
raised 1024 -> 65536 (above all measured sizes; the machinery stays for
large-n plus `Always`/env override). D3 sparse-LU cells stay FAIL on
serial-algorithm gap (2.2-2.7x); closing them is serial kernel work
(fewer GEMMs, fill, panel tuning), not wave parallelism as built.

Falsified-trial log (Slice 2):
1. FALSIFIED: "concurrent-GEMM race corrupts parallel waves" — process-wide
   GEMM mutex and scalar-BLAS forcing both still fail; single-threaded
   pooled execution fails identically. The bug was sequential workspace
   poisoning, not a race.
2. FALSIFIED: "zeroing update_buf or dense_block alone fixes it" — each
   alone fails at the same matrix/position; only work+active zeroing (or
   fresh workspaces, or drop-on-error) passes. The audit that cleared
   `work` had only sampled early panels (head-truncated); panel 69's
   entry state was never inspected until the trace.
3. FALSIFIED: "optimistic wave parallelism pays at grid128" — 34% wave
   redo rate at max width 163; conflicts concentrate in exactly the wide
   waves that would carry the speedup. Selective per-panel redo could
   shave the waste but not the 1.2x work inflation; deferred.

### D3 Serial-Gap Campaign (supernodal + simplicial kernel work, 2026-09-28)

Goal: close the serial-algorithm gap to the D3 gate (strict ≤1.2x per
cell vs SuperLU 6.0.1 + OpenBLAS, RAYON_NUM_THREADS=16,
OPENBLAS_NUM_THREADS=16 for thread parity, one-size-per-process, rep0
discarded, residual gate 1e-10 — all cells PASS ~1e-15).

Diagnosis (the gap is purely kernels, not fill): SuperLU carries MORE
fill at every size (grid32 42470 vs 38988; grid128 1315428 vs 1269156,
+3.6%) yet factors 2-4x faster. Roofline: our numeric runs ~2.1 GFLOPS
vs SuperLU-implied ~5.7 GFLOPS single-thread — an arithmetic-intensity
gap (their wide amalgamated blocks reuse bytes; our width-16/k~3 shapes
stream). perf(1) is unavailable (paranoid=2); all breakdowns below come
from TEMP env-gated phase timers, since removed.

Supernodal slices landed (all bit-identical, 279 lib + 6 determinism
green at RAYON 1 and 16, clippy 0 errors):
- Counting-assembly CSC merge (exact per-column counts + prefix + panel-
  batched copies): merge 20.8 -> ~10.6ms at grid128 (remainder is page
  faults on fresh 40MB allocations — SuperLU pays the same). Saved ~15ms.
- Lazy wave levels: the 14.2ms wave census runs only when parallel may
  engage (empty levels = one ascending wave, byte-identical via the
  nondecreasing floor). Saved ~2ms warm (14ms is cold-start).
- Live-source list: reach walk collects live sources, sorted ascendingly
  (identical application order, bit-identical), replacing the O(panels)
  liveness scan. Saved ~3ms.
Grid128 sup: 2.44x -> 2.03x (scale-only, no D3 cell).

Simplicial slices landed (D3 decision arm; same green gates):
- Fused finiteness-gate + partition into one work_nz pass (same
  first-bad-row error, same partitions). Saved ~0.07ms at grid40.
- Clear-on-store: zero L rows right after their last read in L-store,
  shrinking the clear to low_nz + k. Saved ~0.06ms at grid40.
Grid32 simp 1.70x -> 1.67x, grid40 simp 1.82x -> 1.80x. D3: FAIL.

Falsified-trial log (serial gap):
1. FALSIFIED: "per-element finite checks cost the tail" — removing both
   supernodal tail checks saves 0.8ms of 17.5ms. Restored (soundness).
2. FALSIFIED: "wider panels close the gap" — PANEL_WIDTH 16/24/32/48
   re-sweep at grid128: 110 / 110.2 / 118.0 / 140.7ms numeric. 16 stays
   optimal (scalar in-panel costs dominate wider).
3. FALSIFIED: "coarser chunks beat the dispatch barrier" — merging
   supernodes within a panel to width 4/8/16: 105.9 -> 108.8/119.1/140.7ms
   (zero-flop padding + lost value-skips outweigh fewer dispatches).
   Chunk fusion into one GEMM is math-impossible (different B ranges /
   k per chunk). Every-supernode-its-own-chunk stays optimal.
4. FALSIFIED: "mark-split helps the simplicial update" — separate mark
   pass + branchless FMA pass LOST 2.68 -> 3.27ms at grid40 (L re-stream
   + doubled inv-map loads exceed predictable-branch savings). Kept fused.
5. CONFIRMED floors (nothing left in-scope): simplicial update 2.5ms
   (scalar-indirect at op floor), reach 0.7 (walk+sort necessary),
   COLAMD 0.88 (Rust-vs-C gap, rewrite-scale), stores 0.87, TRSM 13 /
   GEMM 19.3 at 8.3 GFLOPS (Pareto-optimal for this dispatch cost;
   finer chunks lose to dispatch, coarser to flops).

Verdict: D3 sparse-LU UNREACHABLE in-session for both arms at both
decision sizes (best: simp 1.67x @32 / 1.80x @40; need ≤1.2x). The
remaining ~1.5-1.6x needs, in order: (a) sonium-compute small-GEMM/TRSM
fast paths with <100ns dispatch (no `_unchecked` CPU BLAS exists on
main or in-tree yet — other agent's workstream); (b) wide-supernode
update granularity decoupled from ≤16 factor panels (redesign: fewer,
bigger-k updates at higher intensity; in-panel must go first); (c)
COLAMD-to-C parity (~0.5ms, insufficient alone). Even zeroing the whole
update path leaves in-panel + merge + analyze above budget at large n.
Handoff: (a) unblocks the Pareto frontier (then re-try finer chunks);
(b) is the structural fix. Note for gate strategy: narrow non-sparse
cells sit much closer (symmetric eig D&C n=300 at 1.26-1.27x, dgetrf
1024 at 1.24-1.27x) — flipping those is a 4-6% task, not a 60% one.

## D3 narrow-cell flip: dgetrf-1024 in-place A/B (2026-09-27)

Prior dgetrf-1024 record (1.24-1.27x) compared an out-of-band
`lu_factorize` (allocates M+C inside timer) against LAPACK in-place.
Fix: `lu_factorize_in_place(&mut m)` arm — clone outside timer, vendor
dgetrf in-place, one-size-per-process (FACTOR_POOL pollution), rep0
discarded, residual gate 1e-10 ALL PASS (8e-16, bitwise-identical both
arms).
One fresh proc (RAYON=16, OpenBLAS=8, load 7-17): ours 8.67-9.06ms med
vs vendor 7.42ms → 1.168x (best) / 1.192x (worst-window)
[luabv 1024 x3 in-place]. VERDICT:
dgetrf-1024 FLIPPED to PASS under D3 (≤1.2x) in every same-window
comparison; no code change needed — harness parity fix only. dgetrf-2048
remains FAIL (1.44x; different regime, needs panel/Schur work).

## D3 narrow-cell verdict: symmetric eig D&C (2026-09-27)

Same-matrix battery, one-window, rep0 discarded, residual/orth gates
PASS (relres ~1e-16, orth ~3e-15; vendor orth/res ~3e-15 — ours matches).
Quiet window (load 3.5-6, RAYON=16, OpenBLAS=16):
n=200 ours med 4.22ms vs scipy-evd 3.55ms → 1.189x PASS (second
independent same-window pass; record window passed at 1.07x; margin is
load-sensitive — ours inflates under load, vendor does not).
n=300 ours med 9.55ms vs 7.43ms → 1.285x FAIL (needs −0.63ms / 6.6%).
Profile (n=300): tridiag ~5.6ms (panel DSYMV 1.44ms top in-phase, at
compute-BLAS floor) + dc ~3.8ms (per-root secular solve 25.7ms +
merge-GEMM 24.2ms internals; merge-GEMM ~35% compute-owned) +
backtransform 0.43ms. Secular unroll4 falsified (0.99x, div-floor);
secular alloc pooling estimated ~0.1ms (insufficient alone, invasive in
parallel root loop — not attempted). n=300 needs compute small-GEMM +
qaccum/TRSM fast paths; no further in-scope serial lever ≥0.6ms.

## D3 eig-300 exhaustive in-scope falsification (2026-09-27)

Follow-up: the compute `opt/*-x86` + BLAS3-backup work IS in this
worktree's base (all ancestors) and the solvers side already consumes
it — the "blocked on compute" is routing-complete but kernel-bound:
- `f64::mat_mul_nn` → `gemm_f64` → `sme2::dgemm` → `avx2::dgemm`
  with PAR_MIN_MACS 4M/3M: small calls already serial-inline (no pool
  hop); parallel/serial bitwise identical by construction. No dispatch
  to save (`gemm_f64_serial` only skips detect/checks ≈ ns/call).
- Tridiag panel already on `mat_mul_tn` + `mat_mul_nn_sub` (qaccum) and
  probed-once AVX2 dots (~150k calls structural).
- Merge `qt.matmul(&u)`: wrapper is one zeroed alloc + panel split
  (bitwise-identical); the zeroing is ~0.05ms/node-total — immaterial.
- `givens_apply_left`: ZERO rotations across all n=300 nodes
  (trig_total=0) — never runs; nothing to vectorize.
- Merge-GEMM ≈1.8ms/solve of the 9.6 needs a 35% KERNEL speedup to
  close the 0.63ms gap — kernel throughput is compute scope. No
  in-scope lever remains; eig-300 stays FAIL pending compute kernels.

## D3 dgetrf-2048 verdict: FAIL at tuned config, 32T "pass" falsified (2026-09-27)

Same-day, one-proc-per-config, interleaved rotating arms (rep0
discarded; luabv rotates all 6 arms per rep — drift-defeated medians):
- 16T/16T (tuned): ours default 36.04 / in-place 38.02 vs vendor 25.28
  → 1.426x / 1.504x FAIL (reproduces the 1.44x record).
- 32T/32T: ours 35.2-37.1 vs vendor 31.7-32.9 → 1.09-1.16x "pass" —
  FALSIFIED as a gate path: we are FLAT across 16T→32T (36.0→35.5,
  bandwidth-saturated) while OpenBLAS REGRESSES 25.28→32.3 (+28%).
  The pass is 100% vendor-side thread oversubscription, 0% our gain;
  best-vs-best stays 36.04 vs 25.28 = 1.43x FAIL. TINY=0 control ties
  tiny-on at 32T (35.18 vs 35.98) — the funnel's documented 1.16-1.20x
  is self-relative (OB-free), irrelevant vs vendor best.
- Methodology note: in-place ≥ default by ~5% (cold first-touch pages
  on the caller buffer vs pool-warm + inside-funnel clone); the
  in-place arm remains the conservative vendor-parity arm, as used
  for the dgetrf-1024 PASS verdict. Residuals 8e-16–2e-15 all arms.

## D3 borderline hardening: third samples straddle the gate (2026-09-27)

Third independent samples (load 1.8→6.0 rising mid-battery):
- dgetrf-1024 in-place: 8.94 vs vendor 7.45 → 1.200x. Three samples:
  1.168x / 1.192x / 1.200x — 3/3 PASS but margin 0-3%. BORDERLINE PASS.
- eig-200: 4.30 vs 3.56 → 1.208x FAIL (over by 0.7%). Windows: 1.07x /
  1.189x / 1.208x — 2/3 PASS. Load-sensitivity confirmed as predicted:
  ours inflates +2% (4.22→4.30) as load rose, vendor flat (3.55→3.56).
- eig-300: 9.74/7.41 = 1.314x FAIL (best-quiet bound stays 1.285x).
Honest verdict: the two flipped cells sit AT the gate (±2% straddle it),
not comfortably inside. ~5% more each is needed for comfort (0.4ms at
1024, 0.2ms at eig-200); no in-scope lever covers either (secular alloc
pooling ≈0.1ms < 0.2 needed, and invasive in the parallel root loop —
declined on risk/reward; LU-1024 already on best routing, gap is kernel
throughput). Comfort margins need compute kernels, same as the FAIL cells.

## D3 full-gate survey: remaining cells and leads (2026-09-27)

Survey of every non-passing D3 cell for untried in-scope levers:
- Thread affinity: default-ON (opt-out `SONIUM_SET_AFFINITY=0`), all
  pools + mirror pinned packed. Already consumed — nothing to gain.
- dpotrf-1024 (1.30-1.35x): Cholesky already has the full tiny/compact/
  mirror funnel (`SONIUM_CHOL_TINY` opt-out). Gap is not SYRK, dots at
  parity (per opt notes); NB sweep moves ~6% < 10% needed. No in-scope
  lever; potrf vendor harness not built (negative EV to confirm a FAIL).
- sgetrf-2048 (2.00x): f32 kernel gap, compute scope.
- dgeqrf+dorgqr MT (1.76-1.90x): scale-free gap, structural/kernel.
- dgeqrf+dorgqr single-thread 2048 (1.28x): CLOSEST unexamined cell
  (needs 7%), but no harness exists and primitives report at parity
  (ST GEMM 0.84-1.11x, dots 1.00-1.04x) — mechanism unknown, likely
  subtle (T-matrix/LARFT/dorgqr traffic?). Recommended next
  investigation IF directed: build qrab harness + ST profile first;
  estimated P(flip) <25% for ~3-4h. Not started (low EV, needs a go).
- GPU cells (dgetrf-4096, SpMM): 1.00x PASS via delegation already.
Net: every remaining gap mechanism is compute-kernel or structural.
In-scope serial work stays exhausted; D3 needs compute kernels (other
agent) or directed structural redesigns (wide-supernode, QR-ST).
## compute-202609 worklog (2026-09-27, worktree `compute-202609` @ main d5dcf304)

Branch for the vendor-parity goal (CPU BLAS/LAPACK vs OpenBLAS first,
then GPU vs cuBLAS/cuSPARSE). Scratch probes: `~/tmp/c202609` (CPU,
worktree-pinned: gemmab/tapack/gemvab/gemvpin/lufree + obgetrf/obqr/
potrf_ab clones) and `~/tmp/c202609-gpu` (gpubench + SpMM + cudeleg
clones, worktree-pinned).

### CPU GEMM vs OpenBLAS (hot medians, best-vs-best @32T unless noted)

| Shape | f32 | f64 | Note |
|---|---|---|---|
| sq256/512/1024 NN | 0.8-1.05x | 0.5-1.0x | parity-or-better |
| sq2048 NN | 1.26x | 1.38x | open: MT scaling |
| bemski/lu-sch (thin NN) | 0.3-0.8x | 0.3-0.8x | we win big |
| lu-tn (trans-A) | 1.35-1.5x | 1.36-1.66x | open: structural |
| sq1kT (trans-A) | 1.13-1.20x | 1.44-1.53x MT | open MT; ST fixed below |
| ST all NN | ~1.0-1.1x | ~1.0x | parity |
| ST sq1kT | was 1.68x, now 1.10x | 1.08x | SHIPPED (f32 serial pre-pack) |

### CPU GEMV vs OpenBLAS (first-look, contended box — magnitudes only)

NoTrans MT: sq1k 3.9-4.9x, sq4k 4.8-7.6x, sq8k 1.2-2.2x, tall 3.3-6.9x.
NoTrans ST: 1.1-1.6x (threading collapse, not kernel). Trans ST:
1.8-3.7x (serial axpy-per-row kernel gap). Trans MT large: ~parity.
Fixes implemented default-off pending quiet A/B: `SONIUM_GEMV_PINNED=1`
(pinned-pool + worker-count bands; first A/B confounded by contention,
tall won 1.5x), `SONIUM_GEMV_TRANS_FUSED=1` (one fused tile over `y`
instead of n axpy calls; tile test green).

### CPU LAPACK (incomplete — box contended before potrf/obqr ran)

dgetrf-2048: 46-59ms OB-free vs ~26ms OB in-process (1.8x in-process;
claimed 1.36x was min-vs-standalone; freq floored 2.2GHz by peer load).
Tiny funnel verified present post-merge; engagement A/B queued.

### GPU vs cuBLAS/cuSPARSE (idle RTX PRO 6000, worktree) — PARITY

dgemm 1.000x (1024/2048/4096), sgemm 1.001x, sgemv 1.008x, dgemv 0.994x
(all bitwise); SpMV-prebaked 1.007x; SpMM-prebaked 1.00x vs best alg at
nrhs=1/4/16. Complex ConjTrans maps to CUBLAS_OP_C (no extra pass).
One-shot entries are documented non-live costs (validating 266x =
full D2H validation; unchecked 9x = per-call prebake); solver loops use
PersistentDeviceCsr/PrebakedSpmv at parity. No GPU code changes needed.

### Shipped this session (default-on, verified)

- f32 serial trans-A pre-pack (`with_packed_a_trans_f32` + gate in
  `sgemm`): ST f32 trans-A 1.68x -> 1.10x (matches NN standing).
- f64 serial pack naive loop -> `pack_trans_f64`: measured TIE (ST
  1024³T 1.08x both ways — bandwidth-bound), kept for symmetry, NOTEd.
- Tests: `avx2_trans_a_matches_scalar_mt`,
  `avx2_trans_a_serial_pack_matches_scalar` (blas3),
  `avx2_gemv_trans_fused_tile_matches_scalar` (blas2).
- 475 compute lib + 273 direct lib green; clippy warnings == pristine (7).

### Falsified/reverted (NOTEd in code)

- f32 MT pack-once-shared `A^T` (mirror of f64): gather wins —
  1024³T off 1.15x vs on 1.20x/1.41x (quiet-box 32T A/B/A); extra DRAM
  pass over L3-fitting `A`. Fully reverted; NOTE in `sgemm` bars retry
  without addressing extra-pass traffic. (Also questions f64's shipped
  MT gate: f64 1024³T off 1.22x vs on 1.44-1.48x — needs end-to-end
  QR/Cholesky A/B before touching.)

### GEMV trans-fused arm + tile unroll (2026-09-27, contended box)

- Fused serial fan-out first measured 0.60-1.01x vs axpy-per-row
  (same-process interleaved, ST): root cause was the tile's single-chain
  k-loop vs axpy's 4-lane unroll. Unrolled the tile 4x (f32+f64, mirrors
  axpy; bitwise identical, all 22 blas2 tests green incl. 2 new
  env-guarded arm tests through the public entries).
- Re-run as box re-spiked (load 13 -> 66 mid-run): sq shapes 1.00-1.13x
  fused (marginal), t-shapes 0.71-0.93x (unreliable — ran under the
  spike). VERDICT PENDING quiet re-run of `gemvtransab`.

### GEMV fused arm: FALSIFIED and reverted (2026-09-27)

- Quiet-window ST re-run reproduced the split 2/2 (interleaved pairs):
  1024²T fused wins 1.14-1.24x, 2048²T ties, 512x2048T loses 0.68-0.93x,
  2048x512T splits by precision (f32 wins/f64 loses) — all bitwise 0e0.
  Verdict insensitive to box state (same pattern at load 13 and 66).
- Gate + branches + gate test fully reverted; bar-retry NOTE at the
  serial branch. KEPT: the 4x tile unroll (strictly improved the tile
  everywhere measured, 0.71x->1.14x on 1024²T-fused; serves the parallel
  path) + the direct tile test (renamed
  `avx2_gemv_trans_tile_matches_scalar`). Real fix for the 2-3.7x ST
  trans vendor gap is register-blocked multi-row fan-out (queued).

### GEMV NoTrans: spread broadcast SHIPPED default-on (2026-09-27)

- Root cause (proven): raw `rayon::scope` plateaus at ~4T (sq4k f64:
  1T 6.3, 4T 2.4, 16T 2.4, 32T 2.1ms) while OB scales to 0.2ms;
  band→CPU diagnostic showed the map fully reshuffling every rep ->
  every rep pays fabric/DRAM (~56GB/s) instead of local L3 slices
  (~470GB/s). A compact-pool scope variant was falsified first
  (0.5-0.7x squares).
- Fix: new 16-worker spread pool (one L3 slice per worker,
  `affinity::spread_cpus` + `pin_spread_worker`, `SONIUM_SPREAD_CPUS`
  override) + `ThreadPool::broadcast` with worker-indexed bands
  (static mapping by construction; nested/sub-width falls back to raw
  scope; `SONIUM_GEMV_PINNED=0` restores raw scope).
- Quiet-box same-process A/B (16T, bitwise 0e0): broadcast beats raw
  everywhere (sq1k 1.5x, sq4k 2.2-2.6x, sq8k 1.4-1.5x, tall 2.1-3.5x).
  NoTrans MT vs OpenBLAS: 1.05-9.0x -> 0.84-3.05x (sq8k wins outright;
  sq1k f32 3.05x dispatch floor + sq4k f32 1.71x remain).
- Bitwise test: `avx2_gemv_notrans_pinned_matches_scalar` (values vs
  scalar + raw-vs-broadcast exact agreement).

### CLOCK QUALIFICATION 2026-09-27 (read before comparing numbers!)

- All policies capped at scaling_max 2.7GHz, set 10:46 today by an
  external actor (root-owned sysfs, no sudo; policy0 re-touched 11:02 —
  something actively manages power). Pre-10:46 numbers ran at full
  boost (~4.1GHz); post-10:46 at ~2.4-2.9GHz.
- Unaffected (same-process ours-vs-ours, clock-immune): f32 pre-pack
  1.68->1.10x, f32-MT-pack falsification, tile unroll, fused-arm
  falsification, broadcast-vs-raw 1.4-3.5x (gemvpin ran 10:45, pre-cap).
- Broadcast-vs-OB (gemvab =1, 10:46:13): NoTrans head rows ran pre-cap;
  trans tail may be capped (trans path unchanged — verdict unaffected).
- VOID comparisons: Cholesky 20.8ms (capped) vs record 14.4ms
  (boosted) — no regression, different clocks. Capped-clock Cholesky:
  1.78-1.94x vs OB@16 (OB bandwidth-bound/flat across clocks, we
  compute-bound — the ratio is conservative; full-boost ratio likely
  ~1.2x per record, UNVERIFIED today). Same caveat for any post-10:46
  OB ratio on compute-bound shapes.
- Rule: cross-time absolutes void (as always); ours-vs-ours A/B valid
  any time; OB ratios need a same-run OB arm AND a clock note.

## Hardening (11:05, load 163-198, cap still on)
- Spread round-robin extracted to pure `spread_from_groups` + sysfs-free
  unit test (`spread_schedule_hits_distinct_slices_first`); live-set
  invariant test (`spread_set_is_unique_and_masked_when_present`).
- `avx2_gemv_notrans_concurrent_broadcast_matches`: 4 OS threads broadcast
  concurrently on the shared spread pool — no hang, all bitwise vs the
  single-threaded reference. Closes the one shipped-code hang risk.
- Suites: affinity 6/6, blas2 22/22 green (debug).

## Queued design: register-blocked serial trans fan-out (no code yet)
- Current serial trans (`gemv_trans_*`, blas2/avx2.rs:562): axpy-per-row
  loop; the fused single-row tile was trialled and reverted (shape-split).
  MT path already uses the 4x-unrolled tile into L1-resident priv bands.
- Gap 2-7x vs OB, hypothesis (falsifiable): per-row store-forwarding
  chain — every y[k] round-trips store→load across all n rows (~5cyc/hop
  on Zen) plus per-row axpy prologue. Predicts gap grows with n
  (consistent with the 2-7x spread); confirm from gemvtransab rows.
- Fix: R-row x K-lane register-blocked kernel — load y-vec once, FMA R
  rows (R xj broadcasts) in registers, store once. Forwards /R, divides
  prologue by R. f64/f32 budget: R=4,K=2 -> 8 A + 4 xj + 2 y = 14 regs;
  R=2,K=4 -> 8+2+4 = 14. Both fit AVX2 (16 ymm). Try R=4,K=2 first.
- NOT bitwise vs today (register accumulation reorders rounding); test
  plan: scalar parity 1e-12 (existing `check` fns) + MT-vs-serial
  agreement, never tile-bitwise. Reuse existing k-tail; j-tail falls
  back to single-row axpy. f32 twin same shape. A/B R=4x2 vs R=2x4 in
  the quiet box; SW prefetch only if A/B demands it (rows stream
  sequentially, HW prefetch should cover).

## Quiet-box runbook (probes rebuilt 11:14 vs current worktree)
Pre: `cat scaling_max_freq; uptime; grep MHz /proc/cpuinfo|avg`. Need
1-min load < ~8 and cap state recorded (cap still 2.7GHz at 11:14 —
ratios qualified, absolutes void cross-clock).
1. `RAYON_NUM_THREADS=16 ./target/release/gemvpin` — default-on confirm
2. `./target/release/gemvab` — default-on NoTrans shape sweep
3. `RAYON_NUM_THREADS=32 ./target/release/potrf_ab 16` — Cholesky/LAPACK
4. `./target/release/obqr` — QR baseline (biggest absolute prize)
5. `potfree`/`lufree` x2 (unset vs `SONIUM_CHOL_TINY=1`/`SONIUM_LU_TINY=1`)
6. `RAYON_NUM_THREADS=16 ./target/release/gemmab 16` — MT sq2048/trans-A
7. `./target/release/gemvtransab` — trans rows (check gap-vs-n scaling
   for the register-block design) + `bl1ab` + `spmvroof`
All probes interleave arms same-process (clock/load immune ours-vs-ours;
vs-OB ratios need the cap note). Skip tapack/gemvplace/gemvscale (done).

## Gates (11:15, load ~117, cap on)
- Full compute lib suite release: 479/479 green (476 baseline + 3 new
  hardening tests). Clippy: 0 new warnings (4 lib warnings pre-exist on
  untouched lines). Probes rebuilt vs current worktree; runbook above.

## Test fix (11:2x, load ~75-119, cap on)
- Found by re-review: `avx2_gemv_notrans_pinned_matches_scalar`'s
  raw-vs-broadcast section used `remove_var` for the raw arm — but
  broadcast is default-on, so it ran broadcast twice and the bitwise
  assertion proved nothing. Fixed to explicit `SONIUM_GEMV_PINNED=0`
  for the raw arm (+ stale "default-off" comments x2). 22/22 green —
  the bitwise claim is now genuinely proven in-suite (probe maxdiff
  0e0 was the other leg; gemvpin toggles 0/1 explicitly so shipped
  A/B numbers stand).

## Test add (11:3x, load ~21-75, cap on)
- `avx2_trans_a_mt_pack_matches_gather_bitwise`: f64 512³T with
  explicit `SONIUM_GEMM_PACK_A_MT=1` vs `=0` — proves the shipped
  default-on MT pack matches the gather arm bit-for-bit (doc claim now
  locked in-suite; avoids the GEMV remove_var trap by construction).
- Full lib suite: 480/480 green (debug). Clippy: same 4 pre-existing.

## GPU parity re-confirm (11:4x, GPU idle 0%, CPU load ~28, cap on)
Fresh rebuild vs current worktree, idle RTX PRO 6000:
- dgemm 1024/2048/4096: 1.000x vs cuBLAS, rel-err 0e0 (bitwise)
- sgemm 2048: 1.000x bitwise; sgemv/dgemv 8192: 0.990-0.992x bitwise
- SpMV prebaked: 0.147ms vs cuSPARSE 0.147ms (1.00x); validating
  one-shot 281x and unchecked 9.8x overheads unchanged (non-live)
- SpMM nrhs=16: 1.255ms vs best cuSPARSE alg2 1.254ms (1.00x)
GPU half of the objective stands verified; no code change indicated.

## Pinned-vs-raw under contention (11:5x, load ~16-22, cap on) — ARTIFACT
- gemvpin 16T x2 runs: raw rock-stable across runs (sq1k f32 0.090 vs
  0.091ms, sq8k f64 6.95 vs 6.93ms) while pinned is a scheduling
  lottery — sq1k f32 pinned 0.053ms then 5.9ms (0.02x!), sq4k pair
  split 0.13x/3.37x seconds apart, tall 0.07-0.11x then 1.65x.
- Mechanism (confirmed by the variance signature): peer holds ~16-22
  migrating threads; ~3 of the 16 pinned CPUs are preempted at any
  moment and the broadcast join waits for the slowest worker (one 5ms
  stall dominates a 0.05ms call). Raw's 256 micro-tasks steal around
  preempted cores, so it looks stable. Static-vs-adaptive under
  oversubscription — NOT a code regression; quiet-box verdicts
  (incl. tall 2.1-3.5x) stand. Still need a quiet re-confirm.
- Follow-up (design only): load-aware fallback — broadcast when cores
  are free, stealing scope when oversubscribed (placement hardly
  matters once caches are thrashed anyway). Needs a cheap reliable
  oversubscription signal; do not build without one.

## f64 MT pack m-sweep (tapackm, quiet load ~6, cap on, interleaved)
- g/p (gather/pack, >1 = pack wins): m=32: 0.92x, m=64: 0.77x,
  m=128: 0.76x, m=256: 0.87x, m=512: 1.16x (n=k=2048). Crossover in
  (256,512]. k=1984 control identical (0.93x/0.91x) — no aliasing.
  Bitwise 0e0 all rows. Doc claim "32..256 wins packed" FALSIFIED.
- CAUTIONARY: gemmab pack-ON lu-tn 5.8ms was peer-burst interference
  (f32 gather row spiked 8.79x in the same window); cross-run ON/OFF
  A/B overstated the hole (4.6x) vs interleaved truth (1.16x pack,
  1.06x gather — near parity either way). Interleave or distrust.
- Decision: MIN_M=512 gate (pack only where proven); Win: 1.1-1.3x on
  thin-m trans-MT (m<=256); sq1kT keeps pack (hot 1.40x on vs 1.69x off).

## MT GEMM placement root-cause (quiet, cap on, 16T)
- Unpinned: sq1024/2048 1.3-1.7x lose; taskset-0-15: 0.7-1.0x win;
  taskset-112-127: same 0.62-1.13x. Same-cores = fair = we win on
  merit. post-run sample: policy0 2.2GHz parked, policy112 4.03GHz.
- Mechanism: mirror pool packs 16 workers onto CPUs 0-15 = 2 CCX
  slices; 16 N-split bands x multi-k-pass A-streams thrash 32MB of
  slice (sq1024: 8 bands/slice x 8MB A » 16MB) while roaming OB
  spreads over 16 slices. (Freq-harvest asymmetry possibly
  contributes; both theories predict spread wins.)
- Small shapes fit (sq256/512 win on mirror: 8 bands x 0.5-2MB <=
  slice) — router needs an A-footprint clause, not blanket spread.
- Fix: scope_for_gemm_mt (spread-pool scope when A spills, mirror
  otherwise) + SONIUM_GEMM_SPREAD override for A/B.

## Spread-GEMM router REVERTED (12:0x, load 4-20, cap on)
- 16T: spread-vs-tiered TIED in our absolutes (sq1024-f64-hot 6.70 vs
  6.48ms; ratios differed only via OB-regime wobble 4.6-6.6ms).
- 128T: spread-16 LOSES 2.5x vs tiered-48/128 (sq2048-f64-hot 50.5 vs
  19.6ms — width starvation). Tiered wins-or-ties everywhere.
- Root-cause correction: the taskset "0.7x win" was OB slowing under
  core-sharing (7.3 vs 5.0ms), not us improving. Placement does not
  move our MT GEMM absolutes; compulsory N-split re-read traffic is
  placement-invariant. GEMV broadcast wins (bandwidth + rep-persistent)
  do NOT transfer to compute-bound single-flight GEMM.
- Reverted call sites (6), router, flag, and tests; 62/62 blas3 green.
- METHODOLOGY (hard-won, 3rd instance): cross-config A/B with a live
  in-process OB arm must scrutinize OB ABSOLUTES, not just ratios —
  OB slows under sharing, spikes under peer bursts, and drifts with
  freq regime. Interleave ours-vs-ours per rep, or distrust.

## Trans MT gate 4M -> 1M + task budget 1x/4x SHIPPED (12:2x, quiet)
- gemvtransmt sweep (MT-vs-serial, interleaved): 512² MT loses 1.4x
  (keep serial); 1024² MT wins 1.3-1.75x; 1536² 1.8-6x; 2048² 4.5-8x.
- Serial-trans cliff found: single-thread A streams past ~16MB (one L3
  slice) fall to 0.3-0.43x rate (f64-2048²-serial 2.8 GFLOP/s); MT bands
  stay slice-local. Gate-1M fixes small-MT AND the cliff range.
- Task budget: 1x/worker at <=2M els (sq1kT-forcedMT 32->16 tasks wins
  1.2-1.5x), 4x above (sq4kT/sq8kT regress 1.2-1.8x at 1x — keep 4x).
- ISOLATED defaults: sq1kT f32 0.238->0.076 (OB 0.058: 1.3x/1.0x min),
  f64 0.509->0.125 (OB 0.059: 2.1x/1.75x min). sq4kT/sq8kT unchanged
  (sq4kT-f32 0.73->1.16->0.70 across runs = wobble on identical path).
- Remaining sq1kT-f64 1.9-2.4x = kernel (register blocking queued).

## Flake note (12:3x): one full-suite failure, unattributed, 4/4 green since
- A single `cargo test -p sonium-compute --lib` run failed (name not
  captured); the next 4 full runs pass 482/482, plus 23/23 blas2 and
  62/62 blas3 subsets. Race audit of this session's env-flipping
  tests: blas2 TRANS_MT/PINNED share ENV_LOCK; blas3 PACK_A_MT/M_SPLIT
  follow the one-writer contract; a concurrent below-gate trans call
  under a forced-MT window still lands within tolerance (no test
  asserts cross-path bitwise); dispatch asserts run under lock. No
  mechanism found; if it recurs, capture the name before theorizing.

## Register-blocked trans tile REVERTED (12:4x, quiet)
- R=8/K=4 f64 kernel (8x fewer y round-trips), scalar-correct incl.
  tails on first run. MT A/B vs single-row tile (ours-alone): sq1kT
  run1 min +37%/med -15%, run2 med/min both lose; sq8kT run1 +15%,
  run2 -5%. Flip-flops both directions = neutral within ±25% wobble.
- FALSIFIED: y-traffic/store-forwarding as the trans-gap mechanism
  (8x reduction, no change). MT trans ≈ parity except sq1kT-f64
  (1.9-2.4x, 66µs abs); ST trans 1.5x (9ms abs at sq8k) mechanism still
  unknown — next step is `perf stat` counters, not more theorizing.
- Reverted kernel + switch + wiring + tests; my hunks fmt-clean
  (repo-wide fmt dirt pre-exists, incl. 4 tile-test lines).

## f32 k×2 microkernel re-trial REVERTED (12:5x, quiet)
- Isolated f32 gap is real: sq1024/2048 1.23-1.29x (min+med agree,
  both sides stable). Microkernel/pack/blocking structurally mirror
  f64 (12 chains, same Kc, memcpy-twins) — no gross asymmetry found.
- k×2 re-trial (bitwise-safe, 62/62 green): sq1024 min +5%/med flat,
  sq2048 min +2%/med -5%. Not clean -> reverted (single hunk).
  The 2026-09-26 "zero" verdict stands; the dismissed +34% stays
  dismissed. f32 1.23x mechanism unknown (issue/uop-scheduling
  suspected; needs counters). f64 MT GEMM remains at parity.

## SYRK beta-triage fusion LANDED (2026-09-27, quiet box)
- Root cause of the entire thin-k SYRK gap: serial scalar triangle
  zero-fill/scale in `syrk_core` (~1.8ms at n=2048). Replica probe
  (64 triangular bands via public API) proved bands healthy: beta=1
  replica 2.12 vs in-situ 2.39 (tie), beta=0 replica 2.26 vs in-situ
  4.22 (1.8x! only delta is `set_triangle`).
- Fix: fused triage into each band task (band owns its C rows;
  same elements/ops, bitwise identical, no extra pass/barrier) +
  parallel triage scope for the k==0/alpha==0 early path at n>=512.
  Team API untouched (wrappers pass beta=1; Cholesky pre-scales).
- f32-2048x256: 3.91 -> 1.84ms isolated (2.1x). Post-fix SYRK (min):
  f64 parity-or-better 4/5 shapes; f32 1.08-1.41x.
- New test `syrk_beta_triage_large_and_degenerate` (n=600/517 beta
  0/0.5/2, k==0 + alpha==0 incl. parallel-triage scale). 483 lib
  green, clippy 4 pre-existing. syrkab gained `AB_BETA` override.

## f32 trans-pack vectorization LANDED, f64 REVERTED (2026-09-27)
- Pack-split probe (NT vs NN, identical sweep): NT 0.232 vs NN 0.187
  (OB 0.183) — scalar trans-gather pack was the whole per-band gap.
- Fix: `pack_b_tile_f32_trans` delegates to existing vectorized
  `pack_trans_f32` (8x8 AVX2, exact remap via offset pointers, zero
  new unsafe). Per-band f32: 1.33x -> 1.04-1.16x vs fresh OB.
- f64 twin REVERTED: 8x4 core regressed (32x2048x512: 1.10x ->
  1.94x). Mechanism: r-outer/c-inner re-touches source lines across
  r slices; at ldb=k=512 (4 KiB pitch) all lines alias one L1D set
  and thrash. Scalar 8x8 streams each full line once. f32 has the
  same latent caveat at 4 KiB pitch (k multiple of 1024) — verify
  any k>=1024 trans-B shape after touching (doc comments updated).

## Measurement notes (2026-09-27, Threadripper PRO 3995WX)
- OB spawns a 64-thread pool but honors `openblas_set_num_threads`
  for compute (verified: 1/4/16/64 -> 18.1/6.3/2.6/1.8ms on f32-1024;
  extra threads park). AB_ALONE protocol stands.
- Freq lottery: scaling_max 2.7GHz UNENFORCED (cur up to 4.3GHz).
  Absolute ms drift run-to-run; intra-run ratios + min-of-reps are
  the truth. ST baselines run boosted (~4.2GHz), MT capped (~2.7):
  always freq-adjust (x1.56) before claiming scaling losses.
- GF/s sanity: Zen2 SP peak = 32 flops/cycle (2 FMA x 8 x 2), NOT
  16. OB f32-2048 (1335 GF/s) = 61%, consistent.

## Thin-k N-split / TRSM-512 / sq-f32 PARKED (2026-09-27)
- (m x 2048 x 128) loses 1.5-1.9x all m (192-768); ST per-task at
  OB parity (1.00x). M-split loses, forced-2D neutral-or-worse,
  beta=0 no help (OB's beta=0 even has its own serial-memset
  disease, 1.5x slower than their beta=1). OB scales 8->16T 1.72x,
  we scale 1.20x (overhead-bound). 2D degenerates here (L1 cap
  forces ct=32 -> rt=1 = N-split with extra steps). Rank-k M gate
  (k*n <= 128K els) correctly excludes n=2048 (1MB panel spills
  L2); it WINS where it fires (obgetrf thin 992: ours 1.3-1.76x
  FASTER than OB). Lever is L2-aware 2D or finer tasks; needs
  counters (`perf stat` still paranoid=4). Do not retry band-count
  or width tuning (falsified: nb=64 ties, F32_MIN_TASK_MAC 15M
  loses 3.587 vs 3.416).
- TRSM f32-512x2048U 2.1x = thin-k update rate (1.4x, above) +
  fused-band structure. Fused beats panel loop (1.35 vs 1.65);
  OB's TRSM beats the sum of its own GEMMs (panel-cache effects).
- sq-f32 1.23-1.27x (f64 at 0.93-1.01x PARITY): our f32 micro runs
  at 49% of peak = its 6x16 FMA-port ceiling (12 FMAs/6cy = 16 of
  32 flops/cy). OB's 61% needs a bigger tile — new-kernel project.
- Cholesky f64: 1.02x @1024, 1.60x @2048 (solver-driver
  apportioning out of scope). QR flat 1.31x.

## BLAS1 / SpMV / GPU verified (2026-09-27)
- BLAS1: at/above parity everywhere (dot to 0.22x, nrm2 to 0.31x;
  axpy parity). 64M hits 118-125 GB/s (DRAM roof).
- SpMV: f64 118 GB/s saturated; f32 87 GB/s (index-bound, CSR).
- GPU (idle RTX PRO 6000): sgemm 1.001x, sgemv/dgemv 0.99-1.00x
  (bitwise); SpMM 1.255ms vs cuSPARSE best-alg 1.254ms (1.00x).
  Delegation overhead zero; parity holds on current lib.

## TRSM narrow-band fill SHIPPED (2026-09-27, quiet box, load 2-4)
- Default 64-wide bands starve the pool on tall-skinny Left/Lower
  (n=128 -> 2 bands on 16 workers). `trsm_narrow_width` (trsm.rs)
  forces one-micro-panel bands — 16 cols f32, 8 cols f64 at n<=128
  — on m in 256..=1024, n in 64..=512. Unit/non-unit share the
  band path (both gated); Upper/Right/unblocked untouched.
- Ship A/B (pre-gate vs gated binary, same window, medians):
  f32-512x128 2.02x, f64-512x128 2.21x, f32-512x256 1.82x,
  f64-512x256 1.75x, f32/f64-1024x256 1.50x/1.65x,
  f32/f64-512x512 1.66x/1.90x, x64 shapes tie-to-1.12x.
  n>=1024 and m<256/n<64 keep default (neutral-or-loses /
  unmeasured). Post-gate OB gaps: f32-512x128 2.7x (was 5.4x),
  f64-512x128 2.0x (was 4.6x) — kernel work remains.
- m-cap rationale: the win decays monotonically with m (f32 n=128:
  1.99x @512, 1.27x @1024, ~tie @1920 — fatter bands saturate the
  pool while skinnier update GEMMs lose efficiency), and m=1920
  shapes are run-bipolar (~1.1-1.2x same-code deltas; OB-thermometer
  sandwich clean). m>1024 keeps default: identical-by-construction,
  zero regression risk. C32 mirrors f64 by elem size, C64 takes 16
  (perf analogy, unmeasured; suite runs with gate always on).
- 484/484 lib green (new `narrow_width_gate` unit test), clippy
  pre-existing-only, own hunks fmt-clean, TEMP-EXP fully removed.

## TRSM fused routing: ST bypass + m>=256 floor SHIPPED (2026-09-27)
- New probe `trsmfusedab`: same-process interleaved fused-vs-panel
  A/B via per-rep `SONIUM_TRSM_FUSED` toggle (trsm_core reads it per
  solve) — freq-immune, replacing cross-process ST comparisons that
  the freq lottery (2.2-4.3GHz, unpinned) poisoned.
- ST (pinned, 1T): panel loop wins 28/28 (f32/f64 x unit/non-unit,
  m 160..1024, n 128..512): 1.01-1.72x, biggest f64-512x128 (1.72x)
  and f64-1024x128 (1.43x). Shipped: `trsm_use_fused` requires
  threads > 1. F/P collapses to 0.94-1.01x on the new lib (bypass
  engages; panel-vs-fused maxdiff 0 (f32) / 1ulp (f64)).
- MT (16T): panel wins m 160..224 (1.08-1.56x, both types/diags);
  fused wins m >= 256 (1.11-2.4x, growing with m/n). Clean
  crossover 224->256 (2 full nb=128 panels amortize setup).
  Shipped: `TRSM_FUSED_MIN_M = 256` floor in the same gate.
  Collapse-verified: m<256 F/P 0.97-1.02x, m=256 0.66-0.77x.
- Diagnosis record: kernel-pure ST (interleaved `case_st_trsm`) is
  f32 1.38-1.72x (updates at 0.68-1.01x — parity-or-better), so the
  remaining gap is the diag micro-solve rate (42%-of-own-GEMM vs
  OB's 70%), not structure. Killed: 4K-aliasing (LDPAD 0/8/64 all
  1.72-1.73x), codegen spills (asm: 0 stack vectors, clean k-loop),
  inverted-diagonal (inversion flops exceed narrow-band solve).
  Queued next: fine panel blocking (nb 32/64) for small-m to shift
  flops to GEMM-rate updates (predicts ~parity at 128x128).
- 485/485 lib green (`fused_routing_gate` unit test), clippy
  pre-existing-only (4), fmt flag count identical to HEAD (43).

## TRSM f32 panel nb 128->64 SHIPPED (2026-09-27, immune ST A/B)
- TEMP-EXP `TRSM_PANEL_NB` sweep (ascending panel loop; fused keeps
  its own 128): finer panels shift flops from the k-dot diag
  micro-solve (42%-of-GEMM) to GEMM-rate trailing updates.
- f32 ST: nb64 wins everywhere measured (m 64..1920, n 64..2048):
  128x128 1.75x->1.37x, 160x128 1.56x->1.28x, 256x128 1.43x->1.22x,
  512x128 1.27x->1.15x, 256x1024 1.52x->1.34x, 256x2048
  1.42x->1.25x; 1024x256/1024x512 tie-1.02x, 1920x128 tie-1.08x.
  nb32 marginally better only at m<=256 (<=1.03x) but loses big at
  m>=512/n=256 (1.53-1.66x: thin-k updates + epilogue fraction);
  nb16 loses everywhere (1.50-1.86x). MT small-m: nb64 tie-or-win
  (f32-160x128 1.20x over nb128).
- f64 ST already at parity on nb128 (0.88-1.01x everywhere incl.
  1920x128 0.97x): nb64 helps only m<=256 (+1.02-1.05x, marginal)
  and loses 1.02-1.11x at m>=512; nb32 loses everywhere
  (1.06-1.31x). f64 keeps 128 (marginal small wins sacrificed).
- Shipped: `trsm_panel_nb` (trsm.rs) returns 64 for f32
  Lower/NoTrans ascending (`elem_bytes == 4` selects f32 exactly),
  128 otherwise (f64/complex/Upper/Trans/descending/fused all
  status-quo). Ship-verify run reproduces the nb64 column
  (f32-256x128 1.22x, f32-256x1024 1.34x, f64-512x128 0.95x).
- 486/486 lib green (`panel_nb_gate` unit test), TEMP-EXP removed.
- Fused-path nb64 KILLED (2026-09-27, MT AB_ALONE, load ~8):
  consistent ~1.09x f32 losses (512/1024 x 128/256/512; f64 cells
  scatter = noise): each band dispatches one update GEMM per panel,
  so finer panels multiply band x panel dispatches (panel loop pays
  one GEMM per panel — why nb64 wins there). NOTEd at the fused
  `nbs` site; fused keeps 128.

## GEMV quiet re-baseline (2026-09-27, load 3-5, current lib)
- MT 16T (gemvab AB_ALONE, med): NoTrans sq1k 2.2x/2.1x (f32/f64),
  sq4k 1.38x/1.27x, sq8k 0.85x/0.97x (parity+), tall 1.21x/1.34x;
  Trans sq1kT 1.29x/1.90x, sq4kT 1.40x/1.36x, sq8kT 0.92x/0.97x.
  (First-look contended numbers 3-7x were inflated; spread-ship +
  quiet box account for the drop.)
- ST pinned (rows f32,f64): NoTrans sq1k 1.07x/1.03x, sq4k
  1.22x/1.24x, sq8k 1.23x/1.23x, tall 0.88x/0.83x (wins); Trans
  sq1kT 1.13x/2.30x, sq4kT 1.70x/1.55x, sq8kT 1.62x/1.54x.
  (OB sq1k-f64-ST bimodal: med 0.145/min 0.100.)
- Queued attacks: (1) Trans-ST kernel 1.5-2.3x (axpy-per-row;
  largest ST BLAS2 gap); (2) NoTrans-MT sq1k 2.1-2.2x (small-shape
  dispatch); (3) NoTrans-ST sq4k/8k flat 1.23x both types
  (systematic: prefetch/loop structure).

## GEMV Trans-ST: 4-row fanout + serial-at-1T SHIPPED (2026-09-27)
- Root-cause chain: per-row axpy is store-bound (one 256-bit
  y-store per FMA group); OB fuses rows. Runtime-`k` fusion
  (`axpy_many`) ties (per-row broadcast/loop overhead eats gain);
  fixed-K=4 fully-unrolled `fanout_4row_f64/f32` (blas1/avx2.rs,
  y in registers across 4 rows, hoisted broadcasts, bitwise
  identical incl. scalar tail via sequential mul_add) fixes it.
- Pivotal correction: the "ST gap" was measured on the MT path —
  `trans_parallel_worthwhile` (m*n >= 1M) ignores thread count, so
  1-thread solves paid partials alloc + spawn + reduce (pure
  overhead). All pre-fix ST numbers were MT-path-at-1T. Shipped:
  `trans_routing` requires threads > 1 (mirrors TRSM; pure fn +
  `trans_routing_gate` test; existing override test hardened for
  1-thread harnesses).
- Ship-verify ST (immune interleave): sq1kT 0.94x/0.92x (f32/f64),
  sq4kT 1.01x/0.91x, sq8kT 0.95x/0.88x — parity-or-better
  everywhere (was 1.13-2.30x). MT 16T unchanged (sq1kT 1.27x/1.94x,
  sq4kT 1.14x/1.26x, sq8kT 0.90x/1.01x, load-noise).
- Killed along the way (all zero-effect, mechanisms in code NOTEs
  where load-bearing): SW prefetch T0 d64/d256 (ours flat 40-42G
  while OB degraded under contention — we are core-bound), load
  burst reorder (bank-conflict theory), 4K-aliasing (no bimodality
  across 10+ runs kills it), stream-only diag (compromised by
  xor-chain artifact — lesson: sinks must not chain).
- 487/487 lib green, clippy pre-existing-only (4), 1-thread
  harness trans tests green. TEMP-EXPs fully removed.

## GEMV NoTrans: f32 8-chain dots SHIPPED, sq1k = dispatch (2026-09-27)
- Spread-width sweep (SONIUM_SPREAD_CPUS) on sq1k: 16 workers
  0.023ms > 8 (0.028) > 4-spread (0.045) > 4-same-CCX (0.078) —
  fewer workers monotonically SLOWER (bandwidth scales with slice
  count), so small-pool is KILLED; per-worker rate halves 4->16
  (fabric contention), but 16 still wins aggregate.
- row_dot 4->8 chains: f32 MT wins sq4k 1.25-1.35x / sq8k
  1.04-1.11x, ties sq1k/tall and ALL ST (same-method A/B,
  immune); f64 ties everywhere MT (min-identical 7 shapes) —
  shipped f32-only, f64 reverted with NOTE (f64 looks BW-capped
  at 0.125 flops/B; f32 latency/MLP-bound at 0.25).
- sq1k-MT (2.1-2.2x) diagnosed as dispatch/sync-bound (~14-19us
  flat of 22us wall; kernel changes tie): broadcast crushes raw
  scope 2-7x (best in-house) but OB's sync is ~2.7x lighter.
  Queued project: lighter sync (custom spin-barrier pool or
  fewer-sync dispatch) — needs 16 workers for bandwidth, so
  small-pool is out. NOT started (big infra).
- 487/487 lib green (tolerance-equal 8-way tree verified).

## GEMV NoTrans f32: 2-row pairing + tier fix SHIPPED (2026-09-27)
- row_dot2_f32: 7+7 interleaved chains, x shared (14 in flight,
  halves x-traffic). 8+8 first cut spilled (17 live > 16 regs,
  spills visible in asm); 7+7 fits 15 regs exactly, still
  saturates miss buffers. Tolerance-equal per row (7-chain tree
  reorders rounding vs 8-chain single).
- Spill-tier bug found by A/B: `acc += hsum(mul)` per iter is
  ~30x slower/el (serializes on hsum latency + scalar dep); the
  56-stride always leaves a power-of-2 remainder so the tier was
  hot (1.57x whole-shape loss on n=256). Fixed to vector-acc +
  single hsum in all 3 dot kernels (paired f32, single f32/f64).
- n-sweep (fixed 4M els, ST pinned): pairing loses 1.13-1.22x
  at n=128..512 (single-row already at streaming roof), ties
  n=768, wins 1.07-1.26x at n>=1024 -> gate `pair = n >= 1024`
  in sweep_f32 (both beta arms, MT+ST share the driver).
- Same-binary A/B (SONIUM_GEMV_NOPAIR, then removed): MT f32
  sq1k 1.67x / sq4k ~1.2x / sq8k 1.27x, tall ties; ST f32 sq1k
  1.07x / sq4k 1.26x / sq8k 1.19x, tall gated to single (0.89x
  vs OB). f64 control stable run-to-run. Final gaps: MT f32
  sq1k 1.64x / sq4k 1.70x / sq8k 0.82x / tall 1.76x; ST f32
  sq1k ~1.1-1.5x / sq4k 1.00x / sq8k 1.01x / tall 0.89x.
- Killed: 8+8 pairing (spill tax eats MLP gain), SW prefetch
  on dots (SONIUM_DOT_PREFETCH falsified, TEMP-DIAG removed
  with the NOPAIR gate). No TEMP-DIAGs remain in blas2/avx2.
- 488/488 lib green (pairing tolerance gate
  sweep_f32_pairing_bitwise), clippy pre-existing-only (4).

## GEMV NoTrans f64: 2-row pairing SHIPPED (2026-09-27)
- row_dot2_f64: 7+7 interleaved chains, x shared (28-wide main
  loop), same n>=1024 gate as f32. n-sweep (fixed 4M els, ST):
  loses 1.11-1.57x at n<=384, wins 1.10x at n=512 but ties
  n=768 (non-monotonic), wins 1.19-1.23x solid at n>=1024 ->
  conservative n>=1024 gate (n=512 win = follow-up: needs MT
  validation + the 768 dip understood).
- Same-binary A/B (SONIUM_GEMV_F64PAIR, then removed): ST f64
  sq4k 1.15-1.19x / sq8k 1.22-1.23x / sq1k ~1.1x+; MT f64 sq4k
  1.16-1.24x / sq8k 1.11-1.14x (r2 clean; r1 ambiguous from box
  drift, resolved by reversed-order re-run), sq1k/tall tie.
  No losses anywhere -> ungated except n>=1024 (MT+ST share it).
- Shipped gaps: ST f64 sq1k 1.00-1.16x / sq4k 1.04x / sq8k
  1.00x / tall 0.79-0.82x (was 1.23-1.24x on squares); MT f64
  sq4k 0.90-1.14x / sq8k 0.87-0.90x. NoTrans GEMV now parity-
  or-better ST everywhere; MT squares parity except sq1k/sq4k
  (sync/dispatch-bound, known infra project) + tall 1.5-1.7x.
- Also fixed: f32 pairing test never exercised row_dot2 (all
  n < gate); added big-n cases to both pairing gates.
- 489/489 lib green (new sweep_f64_pairing_tolerance),
  clippy pre-existing-only, no TEMP-DIAGs remain.

## GEMV NoTrans MT L3-resident gap PARKED (2026-09-27)
- 2D bisection (traffic x n, MT-16, f32): 16MB loses ~2x all n,
  64MB ~1.6x all n, 256MB WINS 0.86-0.88x all n. Gap tracks
  TRAFFIC ONLY (crossover at ~L3 size 256MB), n-independent.
  Cold-L3 (flushed) still loses 1.23-1.43x, so not purely a
  warm-cache game; per-worker rate ~half OB's at same geometry.
- Killed, all with data: placement (live /proc: 16 workers on
  16 distinct slices, running pinned; compact-16 helps tall
  1.26x but kills sq4k 4.8x — slice thrash), width (8 worse
  than 16), OB thread count (census: 16 RUNNING, same as us),
  force-pair below gate (loses 1.07-1.18x MT), stagger
  (monotonically worse — delay dominates, per-worker rate
  contention-independent), sync (floor 7-13us of 25-80us
  excess — partial only).
- Smell: OB's L3-streaming micro-edge (SW prefetch distance?
  kernel unroll x Zen2 L2 streamer?) needs counters to crack
  (paranoid=4 blocks perf). PARKED with probes: gemvflush,
  gemvrowsweep, gemvsync, gemvspin/obspin (census).
- Incidental: (16384,256) shows 40x run-to-run variance
  (0.067 -> 0.139 -> 2.943 first-cell transient); n=256 "3x
  spike" did not reproduce (y-band-4KB theory killed by
  (16384,512) at 1.85x). No TEMP-DIAGs remain; 489 green kept.

## GEMV Trans MT: spread-broadcast SHIPPED (2026-09-27)
- Root cause of the small-Trans hole (sq1kT 1.9-2.9x): the 2D
  tiled path used a raw stealing scope while NoTrans had moved
  to spread broadcast (2-7x win, same mechanism). Task-count
  sweep first (2/4/8/16: monotonic more-better, 16 optimal) to
  rule out dispatch bloat; then ported broadcast (static
  (b,c) cells per pinned worker, uniform for nchunks>=1).
- Same-binary A/B (SONIUM_GEMV_TRANS_BCAST, then removed):
  broadcast beats scope on every Trans shape MT-16
  (sq1kT 1.6-1.7x, sq4kT 1.9-5.9x, sq8kT 1.4-2.5x). Trans MT
  goes 1.06-2.9x -> 0.22-1.47x vs OpenBLAS (parity-or-better;
  sq1kT-f32 inter 1.58x worst = small-shape sync class).
  Static bands settle slice-local (sq4kT-f64 hot 361GB/s
  aggregate L3 vs scope 61GB/s; OB floats at 79GB/s).
- 1x/4x task budget re-verified under broadcast (64 beats 16
  1.13-1.71x on sq4kT/sq8kT); kept. Coupled to SONIUM_GEMV_PINNED
  (one switch for both GEMV orientations; =0 restores scope).
- 490/490 lib green (new avx2_gemv_trans_pinned_matches_scope:
  broadcast == scope BITWISE incl. multi-chunk f64 grid),
  clippy pre-existing-only (4), no TEMP-DIAGs remain.

## GEMM N-split: L3-fit band rule SHIPPED (2026-09-27)
- Rule (`l3_fit_cpt`, both precisions): most bands with bands x A <=
  total L3, at least pool-fill, at most geometry (16-el floor) and
  min-task economics; gated to fat-mid-K (m>=512, k>=256); L3 from
  sysfs (`affinity::l3_total_bytes`, cached; unreadable/masked keeps
  legacy). Changes only sq2048 (32->16 bands) + bemskinny-f64
  (16->8); everything else converges to legacy or gates out.
- Immune A/B (12 reps, OB-first + ON/OFF alternation, period-4
  address swap, quiet box; same-geometry controls tie): sq2048
  wins f32 1.10-1.14x / f64 1.05-1.08x (4/4 runs, both precisions);
  sq1024/sq512/thin/bemskinny tie (rule no-op or neutral).
  Shipped gaps (immune on/ob): sq512 0.75-0.89x, sq1024
  1.12-1.41x, sq2048 0.85-1.23x.
- Methodology saga (probes gemmbands/gemmrule, both spent but
  left in place; removed envs degrade to default-vs-default ties):
  fixed-order interleaving has ~10-20% first-arm bias; rep%3
  rotation balances POSITION but not PREDECESSOR (ON-after-OB vs
  OFF-after-ON gave systematic ~1.1x OFF-bias on no-op controls);
  OB-first sacrifice + ON/OFF alternation + period-4 swap fixes
  both (controls tie). 9-rep odd counts break swap immunity
  (5/4 median lands in majority layout); 12 reps required.
  Address-layout lottery FALSIFIED for GEMM (f32 stable +-3%
  across 5 fresh ASLR processes); inter-vs-hot window gap
  (~1.25x, both precisions) is OB-rep cache-location drag, not
  layout. Lockstep/wave theory DEAD (inverted wave preference
  sq1024-vs-sq2048); per-pass-A model DEAD (same 64MB opposite
  outcomes); total-L3 fit + min-task WON.
- Deferred: f32 min-task floor 150M->15M would unlock sq1024-64b
  (+4% immune, marginal); sq1024 gap (1.12-1.41x) needs micro or
  deeper work. 491/491 lib green (new gemm_l3_fit_cpt_gates_and_
  invariants, machine-portable), clippy pre-existing-only, zero
  TEMP-DIAGs (BANDS + LOCALRULE removed).

## BLAS-1 campaign SHIPPED (2026-09-27, quiet box load 3-5 -> storm 54)
- Survey (probe bl1ab, MT-16, new scal + complex arms) found three
  serial-dispatch gaps: axpy-64M 3.5-4.0x, scal-64M 4.4-5.7x, and a
  ddot-1M 3.4-5.1x L3-placement gap (OB 550GB/s vs our 150-180;
  our ST kernel at parity so purely placement). nrm2 already won
  everywhere (0.3-0.64x); complex dot ST kernels trailed 1.6-2.4x.
- Ship 1, MT fan-out (real+complex axpy/scal, all in blas1/mod.rs):
  traffic-gated rayon fan-out (axpy 16MB, scal 32MB; gates keep
  L3-resident 1M vectors serial where ST already wins — f32-1M
  regressed 0.84x->1.21x at a 4MB gate). Elementwise = bitwise
  chunk-agnostic (after the tail fix below).
- Ship 2, spread-broadcast for dot/axpy/scal (all 4 types): the GEMV
  mechanism ported via shared `spread_pool_for_blas1` gate
  (default-on, SONIUM_GEMV_PINNED=0 restores legacy stealing
  par_chunks, top-level + width-fit guards). ddot-1M 5.8x->0.68x
  WIN, ddot-64M 1.19x->0.68x WIN (broadcast beats 1000 stealing
  chunks even at DRAM scale); axpy-64M -> 0.93-0.97x WIN;
  scal-64M 5.7x->0.77-0.96x WIN. One 1.07ms first-run flake was a
  cold-pool + storm spike (3/3 repeats 0.96-1.55x under load 19).
- Ship 3, complex dot kernels rewritten (blas1/avx2.rs): the old
  kernels stored to stack and reduced through scalar chains (no
  FMA, no vector accums). New: dual-chain vector FMA (real via
  fmadd(x,y); imag via fmadd(x^signmask, y_swap)) + hsum. cdotc-1K
  2.4x->1.00x, zdotc-1K 1.65x->1.03x, zdotc-1M 2.5x->0.92-1.17x.
- Fix, FMA-tail uniformity: real axpy tails used mul+add while the
  body used FMA, so results depended on chunk boundaries (broke
  the new broadcast==legacy bitwise assert). Tails now mul_add
  (VFMADD under the fn's target_feature); scal was already uniform
  (pure mul). Elementwise ops are now chunk-agnostic bitwise.
- Final BLAS-1 (MT-16, quiet): all 14 ops x 3 sizes parity-or-win —
  64M all 0.17-0.98x; 1M dot/axpy/scal parity-or-win, nrm2 wins;
  1K ties/wins. Holds under load-54 storm at 64M (0.17-0.98x);
  1M L3-band ratios wobble under oversubscription (pinning trades
  float-to-idle robustness for locality; quiet verdicts stand).
- 494/494 lib green (new axpy_par_threshold_matches_scalar,
  dot_par_arms_match_scalar, scal_par_threshold_matches_scalar;
  arm-vs-arm bitwise for axpy/scal, tolerance-vs-scalar for dot;
  all env-robust to parallel-test interference) + 3/3
  native_x86_evidence (dotc 2e-13/2e-5), clippy pre-existing-only
  (4), zero TEMP-DIAGs.
- Infra note: file-tool relative paths resolve at the workspace
  root (/home/pierre/src/sonium), NOT the worktree — bare
  `crates/...` edits landed in the main repo (caught via
  stale-link symptoms: 0.03s no-op rebuilds, missing DIAG output;
  migrated to the worktree, main repo reverted clean). Always use
  `.worktrees/compute-202609/...`-prefixed (or absolute) paths +
  `touch` + `grep` verify before probe rebuilds.

## sq1024 band-count FALSIFIED at 16T (2026-09-27, stormy box)
- Decomposed sq1024 (MT-16 1.35-1.56x): f32 ST 1.11-1.13x (kernel)
  + MT scaling; f64 ST 0.93-0.95x WIN (pure MT gap on paper).
  Scaling (hot): f32 1T 1.13x / 2T 0.91x / 4T 0.99x / 8T 1.41x /
  16T 1.56x — gap opens past 4T.
- Band sweep via temp `SONIUM_GEMM_BANDS` gate (removed after):
  f32 B4 3.42x / B8 2.20x / B16 1.19x / B32 1.06x (hot);
  f64 B16 0.88x / B32 1.13x. More bands monotonically better —
  fewer bands die by underfill (8 tasks on 16 workers), so the
  A-traffic theory is DEAD at 16T (the old 8T 8-band win does not
  transfer). Defaults (f32 32 bands, f64 16) already optimal.
- Residual = f32 ST kernel 1.12x (micro work, no counters: poor
  yield) + storm noise (gemmdecomp shows 1.02x on the same shape;
  inter/hot splits 1.4x/1.0x under load). PARKED as micro-bound.
  Zero TEMP-DIAGs (gate + prints removed, check clean).

## TRSM-MT diagnosis (2026-09-27, stormy box load 15-88)
- Full TRSM-MT trails 1.3-1.6x uniformly (256/512 x 1024/2048,
  U+N, both precisions). ST: f64 wins 0.85-0.98x (pure MT gap),
  f32 trails 1.24-1.34x (kernel + MT).
- trsmdecomp splits it: trailing-update GEMMs (thin-k
  128-384x2048x128) ST 1.05-1.08x -> MT 1.32-1.47x (MT-specific);
  diag panel (128x2048) ST 1.43x -> MT 2.06-2.30x (kernel 1.43x
  x scaling 2.6x-vs-3.8x = 1.46x).
- Diag levers all falsified: PAIR (task count) ties, DIAG_POOL
  ties on diag, LCHUNK 16->64 ties (4 collapses 6.4x), fused
  path default-on ties panel loop. No locks (TLS RefCell bufs);
  oversubscription ruled out (capped pools gate on ambient>48).
  Root cause = kernel ALGORITHM: our row-wise triangular
  micro-solve (8-chain FMA, well-tuned) vs OB's ~GEMM-rate
  approach — prior note already suspects inverted-diagonal /
  blocked micro-solve (untested). That's a multi-hour project
  (new micro-blocking level + tuning + verification): PARKED.
- Updates = thin-k GEMM MT gap (ST parity). Path A/B: default
  (N-split) beats forced M-split (1.6-3.3x) and forced 2D
  (1.5-1.7x) — routing already optimal. Scaling-knee run
  invalidated by load-88 peer storm (cargo test --workspace +
  pcfan 541%); re-run on quiet box before concluding.
- Next when quiet: thin-k knee (1/2/4/8/16T) to separate
  contention vs overhead; then GPU re-verify + audit.

## Thin-k GEMM PARKED (2026-09-27, quiet box load 3.75)
- Knee for 128x2048x128 f32: T1 1.06x / T2 1.54x (ours SLOWER
  1->2T: dispatch overhead) / T4 1.32x / T8 1.37x / T16 1.82-1.94x
  (ours FLAT 8->16T: 0.297->0.290; OB scales to 0.149). OB 4.4x
  vs ours 3.2x overall.
- Band sweep at 16T (temp gate, removed): B4 3.96x / B8 2.38x /
  B16 1.82x / B32 1.73x / B64 1.47x; default (own count) 0.199ms
  beats all forced counts. More-better monotonic, plateau 32-64;
  fewer-bands falsified (underfill). Path A/B: default N-split
  beats forced M-split (1.6-3.3x) and forced 2D (1.5-1.7x).
- Residual 1.5-1.8x is per-task overhead/latency on ~2M-MAC tasks
  (strace: no futex/mmap storm, 1189 parks over the run = normal
  pool sleep/wake; TLS bufs lock-free). Needs counters (blocked)
  or a dedicated thin-k kernel design. Absolute delta is 63us on
  the 128-shape; TRSM updates inherit it (6/7 of TRSM flops).
  PARKED with sq1024-micro and TRSM-diag-project.

## SYRK PARKED as structural (2026-09-27, load 13-15)
- MT: f32 1.30-1.63x, f64 1.15-1.23x; ST: f32 1.45-1.87x, f64
  1.17-1.23x — MT ratios <= ST ratios, so MT scaling is FINE and
  the whole gap is single-threaded (banded NT-GEMM calls vs OB's
  monolithic dsyrk). Beta isolation: beta=1 (no-op triage) gives
  identical ratios — the landed triage fusion is not the cost.
- Matching dsyrk needs a dedicated triangle-aware SYRK kernel
  project (same class as the TRSM blocked micro-solve). PARKED.

## Completion audit (2026-09-27, worktree compute-202609 @ 494 green)
Requirement-by-requirement vs the goal (x86 CPU BLAS/LAPACK parity
first, then cuSPARSE/cuBLAS effectiveness; solvers out of scope):
- R1 worktree: PASS. All changes inside compute-202609 (9 files,
  +3813/-216, all under crates/sonium-compute + PERF.md). Main repo
  shows only a peer's unrelated sonium-surrogates changes.
- R2a BLAS-1: PASS (quiet) / LOAD-CONDITIONED. Fresh bl1ab
  (rebuilt lib): 64M all 14 ops 0.17-1.11x; 1M 12/14 <=1.09x.
  ddot/zdotc-1M win quiet (0.68-0.80x/0.92-1.17x) but trail to
  1.3-2.1x under load-9 oversubscription (pinned broadcast vs
  OB's floating threads — the load-robustness follow-up below).
- R2b BLAS-2 GEMV: MIXED. Fresh gemvab: Trans-MT 4k/8k 0.18-0.74x
  WIN, NoTrans-MT 8k 0.57-0.85x WIN. OPEN: NoTrans L3-resident
  (sq1k/sq4k/tall 1.2-1.9x, needs counters) + sq1k-sync infra
  (NoTrans/Trans 1.0-1.65x, needs dispatch project).
- R2c BLAS-3 GEMM squares: PASS (quiet-immune) / LOAD-CONDITIONED.
  Fresh (load 7): sq512 0.58-0.87x; sq1024 f32 1.42-1.47x, f64
  1.16-1.64x; sq2048 f32 1.40-1.52x, f64 1.13-1.19x. Immune-quiet
  ship claims stand (sq512 0.75-0.89x, sq1024 1.12-1.41x, sq2048
  0.85-1.23x); load degrades static-band ratios (same class as
  R2a). Residual sq1024-f32 ST kernel 1.12x is micro-bound.
- R2c thin-k GEMM: OPEN (1.47-1.94x even on quiet box load 3.75).
  Band/path/routing levers all falsified; per-task overhead on
  ~2M-MAC tasks. Needs counters or a dedicated kernel design.
- R2c TRSM: OPEN (1.3-1.8x). Diag = kernel-algorithm gap (needs
  blocked/inverted-diagonal micro-solve project); updates inherit
  thin-k. All dispatch levers falsified; fused default-on ties.
- R2c SYRK: OPEN f32 (1.30-1.63x; f64 1.15-1.23x). ST==MT ratios
  => structural (banded NT-GEMM vs monolithic dsyrk); beta
  isolation clean. Needs dedicated triangle-aware kernel.
- R3 LAPACK: BOUNDARY VERIFIED. No potrf/getrf/geqrf code exists
  in sonium-compute; factorizations live in sonium-solvers-*
  (out of scope per instructions), which depend on sonium-compute
  and consume its BLAS (solvers-core blas_helpers/traits) — they
  inherit every BLAS speedup automatically.
- R4 GPU: PASS. Idle-RTX-PRO-6000 re-verify on current lib:
  dgemm/sgemm 1.000x bitwise, gemv 0.99-1.01x, SpMV = cuSPARSE to
  the us, SpMM at best-vendor-alg rate. (gpubench temp-repoint
  reverted; manifest verified restored.)
- R5 gates: PASS. 494/494 lib + 3/3 native_x86_evidence green,
  clippy 4 pre-existing-only, zero TEMP-DIAG/EXP/BANDS artifacts
  repo-wide, PERF.md log current through this audit.
- Verdict: goal NOT complete — R2b-partial, thin-k, TRSM, SYRK-f32
  remain open, all diagnosed to root causes needing either hw
  counters (blocked: paranoid) or multi-hour kernel projects.
  Load-conditioning refined by the experiment below (realistic
  penalty ~1.1-1.2x, not the 2x cross-window artifact).

## Pin-collision experiment (2026-09-27, controlled load)
- Question: is the audit's load degradation real, and does the
  broadcast win need band->CPU or only band->worker stability?
- Spread set = first core of each CCX {0,4,...,60} (16 workers).
  New probe loadab: blocked arms (12 reps, median last 9) — an
  alternating design was tried first and POISONED broadcast settle
  (legacy/OB reps evict slice state between broadcast reps).
- Quiet: bcast 0.0317 / legacy 0.1175 / OB 0.0302 (b/ob 1.05x,
  b/l 0.27x). Pinned-set load (16 busy loops on the spread set):
  bcast 3.00ms (b/ob 94-99x!), legacy/OB unchanged (3-4x/1.0x).
  Scattered load-16 (7/16 pinned CPUs collide): bcast 1.13-1.23x,
  legacy 3.4-3.8x — broadcast still wins 3x.
- Nopin falsified: SONIUM_SPREAD_NOPIN (temp, removed) broadcast
  == legacy (b/l 1.03x, 4.2x) — floating workers migrate across
  slices every rep, so the win needs band->CPU stability.
  Pinning is load-bearing; nopin is dead.
- Decision: NO adaptive dispatch. Realistic (scattered) collisions
  cost ~1.1-1.2x and broadcast still beats legacy 3x there; the
  fallback crossover sits at near-total (12-15/16) collision, so
  any gate is knife-edge with high misfire risk for low realistic
  value. Catastrophic colocation needs 16/16 adversarial overlap
  (rare; storms are scattered). Broadcast stays default-on;
  SONIUM_GEMV_PINNED=0 is the documented escape hatch (also note:
  the audit's 2x load numbers mixed freq-lottery + methodology —
  controlled same-window penalty is ~1.2x).
- Incidental: Trans-MT n=1536-f64 trails ~1.9x quiet on BOTH our
  arms (unmeasured size between sq1k/sq4k ship points) — possible
  tier/band issue, queued as open investigation.

## GPU parity re-verified (2026-09-27, idle RTX PRO 6000)
- Current worktree lib, CPU storm-immune device-side timing:
  cudeleg sgemm-2048 1.001x (bitwise), sgemv-8192 1.005x,
  dgemv-8192 0.992x; gpubench three-way dgemm 1024/2048/4096
  all 1.000x bitwise; SpMV prebaked 0.147ms = cuSPARSE 0.147ms
  (bitwise cross); SpMM delegates at vendor rate (nrhs=16:
  1.255ms vs raw csr_alg2 1.254ms). gpubench was temp-repointed
  at the worktree with a separate target dir; manifest restored
  (verified main-repo path back in place).

## Trans-MT task-budget rule SHIPPED: 32MiB byte line (2026-09-27)
- Problem: old 2M-element 1x/4x line put 1536² (2.36M els) on 4x,
  where rows/block halves (64->48) while dispatch/partials/reduce
  quadruple. Broadcast-4x still trailed: f32-1536 ~1.35x quiet.
- Sweep (TEMP SONIUM_GEMV_TRANS_T, element threshold, blocked
  12-rep medians, since removed): T=8M dominates — f32-1536 to
  parity, f32-2048 +15% ours-absolute, 4k/8k untouched; T=4M
  breaks the 2-4M fixes; T=16M regresses large shapes. (The raw
  sweep CSV was lost; the numbers below re-prove the rule from
  scratch and supersede it. An earlier draft of this entry cited
  rect/sweep pairs from memory — struck and replaced.)
- Decisive A/B (scratch forced-1X / forced-4X builds, broadcast
  code identical, 2 interleaved rounds + rect pair, load ~5;
  ours-absolute decides, OB drifted up to 1.4x between windows):
  f32-1536 4x 1.32-1.45x -> 1x 0.91-0.93x (parity, big win);
  f32-2048 1x ours-absolute -13-15% x3 rounds (ratio ~1.7-1.8x,
  OB strong there — residual is kernel-class, see PARK note);
  f32-r1k4kT 1.36x->1.20x; f32-r4k1kT 1.30x->0.93x;
  f64-1536 1.70-1.81x->1.53-1.65x (win; residual PARKED);
  f64-2048 1x-vs-4x TIE (ours 0.234 vs 0.248, ratio
  noise-dominated 1.04-1.22x) — the byte rule's 4x there is
  harmless. 5/5 changed paths win-or-tie, zero regressions.
- Shipped rule (blas2/avx2.rs): 1x iff m*n*elsize < 32MiB
  (TRANS_1X_MAX_BYTES, strict `<`; f32 flips at 8M els, f64 at
  4M els), + `trans_tasks_1x_boundary` unit test. The byte (not
  element) line keeps precision behavior aligned. Changed paths
  are exactly f32-{1536², 2048², r1k4kT, r4k1kT} and f64-1536²
  (4x->1x); 1024² was already 1x, f64-2048²/rects/4k+ stay 4x —
  all other shapes identical by construction.
- Confirmation (fresh build, 3 interleaved OLD/NEW rounds at
  load ~8 + fresh full sweep; mt/ob blocked medians): f32-1536
  2.05x->0.95-1.12x (parity), f64-1536 ->1.60-1.73x stable
  residual, f32-2048 tie-or-win, f64-2048 1.04x parity,
  sq4kT/sq8kT 0.17-0.69x wins. (OLD arm was pre-broadcast HEAD,
  so it measures the whole stack; the task-rule component is
  the T-sweep above. Cross-window absolutes invalid — f64-4M
  same-path shape moved 1.4x between windows under load.)
- Caution logged: file-tool relative paths resolve at workspace
  root, not the worktree — prefix `.worktrees/compute-202609/`.
- Open: f64-1536 residual 1.6-1.7x is kernel-bound, not dispatch
  (stable across 4 runs/loads) — stays OPEN as Trans-MT
  n=1536-f64 investigation; r1k4kT-f64 rect needs one quiet-box
  datapoint (1.80x-hot was load-polluted; no path change there).
  495/495 lib green, zero TEMP (grep-verified).

## Trans-MT f64-1536 residual PARKED as kernel/L3 project (2026-09-27)
- Thread sweep (T=16/8/4, same-window per-T): ours scales 1.75x
  T4->T16, OB scales 1.77x — identical scaling, constant ~1.7x
  absolute deficit at every T. NOT a parallelism artifact: it is
  per-worker throughput. f32-1536 sits at parity (1.00x T16).
- Roofline: OB at 48 GFLOPS = L3 roofline for 18.9MB read-once
  (0.25 FLOP/B × ~200GB/s); we extract 111 GB/s vs OB's
  195 GB/s on identical bytes. Suspect: 12KB-stride row-chunks
  (f64) vs 6KB (f32) interacting with L2 stream prefetch, but
  unproven — needs counters (`paranoid=4` blocks `perf`).
- Ruled out: task budget (4x-broadcast was ~1.9x, 1x ~1.6-1.7x —
  dispatch helped, residual remains), tails (clen=96/1024 exact
  multiples of the 4-lane unroll), jitter (1.60-1.76x across 5
  runs/loads), L2 tile fit (A is read-once streaming).
- Same signature as PARKED NoTrans-L3 (vendor extracts more L3
  bandwidth on identical bytes, all config levers dead). PARKED
  with it: needs counters or a register-blocked kernel redesign
  (fanout's 12KB stride is algorithm-inherent), quiet box, and
  multi-hour budget. f64-2048 (4x, ~1.1x) unaffected.
- Sibling residual: f32-2048 1.7-1.8x at dispatch-optimal 1x
  (1x beats 4x by 13-15% ours-absolute x3 — dispatch is DONE;
  OB is strong there). Same kernel/L3 class, PARKED with it.
  r1k4kT-f32 1.20x likewise dispatch-optimal. r1k4kT-f64
  (1.3-1.9x, OB-noise-dominated, no path change) still needs
  one quiet-box datapoint to call.

## Completion-audit refresh (2026-09-27, final state)
- R1 correctness: 495/495 lib + 3/3 native_x86_evidence green
  on final code (incl. `trans_tasks_1x_boundary`). PASS.
- R3 hygiene: zero TEMP/EXP/BANDS/DIAG strings in
  `crates/sonium-compute/src` (grep-verified); forced-1X/4X
  scratch builds lived only in /tmp binaries, tree restored
  byte-identical (diff-verified). PASS.
- R4 clippy: 4 pre-existing sites only (blas1 x2, blas3 x2),
  none in the touched blas2 file. PASS.
- R5 GPU (final lib, device-side timing): sgemm-2048 1.000x,
  sgemv-8192 0.999x, dgemv-8192 0.989x, all bitwise; SpMM
  nrhs=16 1.255ms = prior vendor-rate mark. PASS.
- R2 CPU parity: only blas2 changed since the last audit, and
  the changed area (Trans-MT band) is re-proven above by
  decisive same-window A/B (5/5 changed paths win-or-tie, zero
  regressions). All other ship claims stand by code-identity.
  PASS-quiet/load-conditioned (unchanged verdicts).
- Final gap ledger (all dispatch-exhausted, kernel/counters
  class): thin-k GEMM 1.5-1.9x, TRSM-diag 1.3-1.8x (needs
  blocked micro-solve project), SYRK-f32 1.3-1.6x, NoTrans-L3
  1.2-1.9x, sq1k-sync 1.0-1.65x, sq1024-f32-ST 1.12x micro,
  Trans-f64-1536 1.6-1.77x, Trans-f32-2048 1.7-1.8x,
  r1k4kT-f32 1.20x, r1k4kT-f64 TBD-quiet. Goal NOT complete
  while these stand; all are counter-blocked (`paranoid=4`),
  quiet-box-gated, or multi-hour kernel projects.

## sq1024-f32-ST micro-gap: NO f32 BUG, PARKED (2026-09-27)
- Box identified: Threadripper PRO 3995WX = **Zen2** (L1D 32K,
  **L2 512K** not 1M, L3 16M/CCX x16). `perf` re-confirmed
  blocked (paranoid=4, even `cycles:u`). Earlier L2-1M
  reasoning stands with thinner margins (B-tile 256K = half
  L2, still fits).
- Reproduced pinned-1T: f32-ST 1.01x@256 -> 1.07@512 ->
  1.11@1024 -> 1.18-1.23@2048; f64-ST flat 0.90-1.01.
  k-sweep (m=n=1024, 2 rounds) FLAT in k — earlier k=1024
  "peak" was single-run noise (box noise floor +/-10%!).
- Chased down every asymmetry, all falsified or symmetric:
  driver nest identical; B-pack identical memcpy (f64 packs 2x
  bytes); traffic bytes f32 <= f64 everywhere; micro-kernel op
  mix instruction-identical (6xNR, 12 FMA + 6 bcast + 2 loads);
  f32_full6 `inline(never)` vs inlined f64_full6 tested both
  ways (scratch builds): outlining f64 regresses 15-20% (inline
  load-bearing FOR F64) but inlining f32 is a no-op (1.09->1.12
  noise) — the inline benefit does not transfer; rowband path
  gated kc<=64, irrelevant at kc=256.
- The actual story: OUR f32/f64 = 1.9-2.0x (width-scaling,
  both near their relative ceilings) while OB = 2.25-2.3x
  SUPER-linear — OB's sgemm kernel is much stronger than its
  dgemm (f32 got the Zen love). Plus a SHARED ~8% size
  degradation both precisions (ours 109->99G f32, 57->53G f64;
  OB flat) — general L2/blocking-class effect, invisible for
  f64 only because OB-dgemm is slow.
- Verdict: no f32-specific defect exists to fix. Closing the
  gap needs (a) a competitive sgemm micro-kernel redesign
  (k-unrolled, wider, prefetch-tuned — multi-day + counters
  to validate) and (b) the shared blocking degradation.
  PARKED with the kernel-project class. New probe
  `gemmijk` (argv m/n/k + OB anchor) kept in ~/tmp/c202609.

## Quiet-window datapoints + thin-k RESOLVED (2026-09-27, load 1.4-2.1)
- Trans rule holds quiet: f32-1536 0.93x parity; f64-1536
  1.70x (PARK band confirmed); f64-2048 1.20x.
  f32-2048 2.02x quiet (band widened 1.7-1.8x -> 1.7-2.0x).
- r1k4kT-f64 quiet 1.68-1.89x = REAL ~1.7-1.9x gap (not OB
  noise; 4x path, kernel class) — replaces TBD-quiet.
  r1k4kT-f32 quiet 1.24x (dispatch-optimal residual stands);
  r4k1kT both win.
- THIN-K GAP GONE on current code: full knees at 128x2048x128
  (T1/2/4/8/16, both precs: 0.61-0.90x) and 256x2048x128
  (0.54-0.91x) + family (m 64/128/384, k 64/128, T8/16:
  0.55-0.92x) ALL GREEN. Entry's T16 1.82-1.94x (ours 0.290,
  OB 0.149) vs now (ours 0.146, OB 0.180): ours 2x faster,
  OB 20% slower — box alone cannot explain opposite moves,
  but NO GEMM-MT dispatch change is identified either (GEMM
  does not use the spread pool; affine/pin work touched only
  BLAS-1/2). Mechanism undetermined; current state wins, so
  thin-k drops off the ledger. (TRSM-update 1.32-1.47x MT
  component inherits this fix automatically.)
- Ledger now: TRSM-diag 1.3-1.8x (blocked micro-solve
  PROJECT, next), SYRK-f32 1.3-1.6x, NoTrans-L3 1.2-1.9x,
  sq1k-sync 1.0-1.65x, sq1024-f32-ST (no-bug PARK),
  Trans-f64-1536 1.6-1.77x, Trans-f32-2048 1.7-2.0x,
  r1k4kT-f32 1.24x, r1k4kT-f64 1.7-1.9x.

## TRSM blocked micro-solve SHIPPED for f32 (2026-09-27, stormy box)
- Design (trsm.rs): left-lower NoTrans rows advance in 4-row
  micro-blocks; each block GEMM-updates from finalized rows
  (k-outer, X rows register-held: 12 YMM) then solves <=4 rows
  row-wise. Lane ops match row-wise exactly (ascending-k
  fnmadd, scalar mul+sub tails) => BITWISE identical; only the
  visit order changes (each X row loads once per block, not
  once per row). Hook in the f32 dispatcher (nb >= 8,
  `SONIUM_TRSM_NOBLOCKED=1` escape hatch, OnceLock-cached).
- Caution #1: first exact test was VACUOUS (hook inside the
  `_avx2` kernel diverted both arms to blocked). Caught by
  suite test `trsm_band_lane_tail_bitwise` (1-ulp lanes-vs-
  tails mismatch). Fix: hooks moved to dispatchers (bodies
  pure = testable), f64 lane policy mirrored row-wise greedy
  tiers. Sensitivity proven (1-op sabotage => FAIL).
- Caution #2: f64's 4-wide tier is deliberately UNFUSED
  (mul+sub, scalar-equal) — any f64 lane work must mirror
  [16-fused / 4-unfused / scalar], not just widths.
- A/B (same-binary NOBLOCKED=1 vs unset, pinned ST +
  interleaved MT): f32 ST -6..-13% (4/4 shapes clean),
  f32 MT -5..-18% (10/11 round-cells; 512x1024U 2-1 with one
  +9% noisy round). f64: ST tie (+1% x8), MT tie (+/-5%) =>
  REVERTED per ship criteria (hook + kernel + test arm
  removed; note left at the f64 dispatcher). f64's gap was
  never kernel (its ST already wins 0.85-0.98x) — pure MT.
- Queued: f32 nb=64-vs-128 re-tradeoff under blocked diag
  (tradeoff was measured row-wise; blocked shifts it toward
  128); thin-k beta=1 updates (trsmdecomp showed 1.45-1.80x
  but storm-confounded — needs quiet rerun; the thin-k
  RESOLUTION was beta=0-only). TRSM residual: f32 ~1.2-1.9x
  MT (updates+overhead), f64 ~1.3-1.8x MT (scaling/updates).
  496/496 green, clippy pre-existing-only.

## Thin-k beta=1 gap + prefetch FALSIFIED, nb=64 kept (2026-09-27)
- Thin-k beta=1 trails where beta=0 wins (gemmijk +AB_BETA=1):
  MT16 128/256/384x2048x128: b0 0.56-0.81x -> b1 1.00-1.31x;
  ST 0.93x -> 1.10x (kernel-level, not MT); k=512 same +15%
  (per-FLOP, not fixed). Fixed-flops C-sweep: delta +33..52us
  flat-ish in C (512K->2M) => NOT C-read bandwidth.
- Falsified lever: beta-gated T0 C-prefetch in both full6
  kernels (epilogue-C-miss theory) loses 6-8% x3 same-binary
  (NOCPREF gate, since removed with the code). Theory wrong
  or T0 evicts/jams the B stream (fill-buffer contention?) —
  NOTEs left at both k-loops. No layout residue (revert
  verified 0 prefetch strings; 496 green).
- f32 TRSM nb re-tradeoff under blocked diag (scratch nb=128
  build): ST favors 128 (-2..-8% x4, pinned clean), MT favors
  64 (tie to +10%, r2) => NET TIE, keep 64 (no code change).
  Blocked diag did not shift the tradeoff decisively.
- Beta=1 mechanism still OPEN (per-FLOP +15-30%, not traffic,
  not epilogue ops, not prefetchable-T0). Suspects remaining:
  C-read port pressure interleaved with k-loop, uop-cache
  footprint of the beta!=0 epilogue, or store-forwarding
  stalls on read-modify-write C. Needs counters. TRSM-update
  beta=1 inherits it (updates = most TRSM flops).

## GEMM spread-routing REVERTED, taskset mirage autopsy (2026-09-27)
- Trigger: sq1024-MT16 1.3x gap + taskset-spread pinning winning
  0.56-0.97x (3 runs) suggested packed-vs-spread L3 slices as the
  mechanism. Built a GEMM-only spread router (top-level scopes w/
  ambient <= 16 -> spread pool install+scope, nested inline,
  SONIUM_GEMM_SPREAD escape hatch) + switched 6 driver sites.
- Autopsy, layer by layer: (1) full gemmab OLD-vs-SPREAD tied
  (sq1024 ours-absolute +/-4%, 2 rounds; every other shape tie-
  or-win, zero regressions); (2) SONIUM_SPREAD_CPUS packed-vs-
  spread (same pool mechanism, OB free): f32 -6%/-21%/-5%/+5%
  over 4 rounds (median -5%, +/-10% noise — the r2 -21% was an
  outlier PACKED-slow run, not a SPREAD-fast one); the taskset
  "win" decomposes into OB confinement harm (OB destabilized
  2.2-3.8x when pinned, flattering ratios) + 2 lucky runs and a
  1.22x third; (3) alloc-offset sweep (0/64K/1M/4M pre-alloc)
  moves nothing systematically => heap-layout/aliasing theory
  for instrument gaps FALSIFIED too.
- Methodology finding: gemmab itself swings +/-23% run-to-run
  on sq1024-MT16 (1.49x -> 1.17x identical runs) while single-
  shape gemmijk holds +/-5% — multi-shape churn state (THP?
  NUMA migration? alloc churn?) dominates. Prefer single-shape
  probes + 3-4 rounds for effects under ~15%; gemmab ties at
  that scale are NO evidence.
- Verdict: spread-vs-packed ties on a stable instrument =>
  REVERTED per ship criteria (6 sites restored, router + flag +
  env removed, zero residue; reinforcing NOTE at SMALL_POOL's
  packed comment). Packed stands for GEMM as well.
- Ledger revision: sq1k-sync DOWNGRADED to MONITOR — stable
  instrument shows MT16 ~1.03-1.19x ~= ST-micro (1.13x, no-bug
  PARKED) + noise; no MT-specific gap proven. (Knee T8 1.33-1.43x
  needs a recheck with repeats before concluding.) 496 green.

## TRSM fused routing confirmed, SYRK re-verified (2026-09-27)
- Fused-vs-panel-loop MT A/B (SONIUM_TRSM_FUSED=0, 2 rounds):
  fused wins all 12 cells x2 (f32 panel 25-80% slower, f64
  20-40% slower). Default stands; no routing change. f64 TRSM
  residual (1.3-1.5x MT, ST at parity) = updates (thin-k
  beta=1 OPEN) + MT overhead, not routing.
- SYRK re-verified fresh (PARK numbers were load-13): MT f32
  1.13-1.55x / f64 0.91-1.34x, ST f32 1.43-1.85x / f64
  1.14-1.23x — bands match the entry; MT<=ST (scaling fine),
  gap is ST (banded NT-GEMM vs monolithic dsyrk). Not inherited
  from NT (NT-ST is 1.14x/1.00x): SYRK's own banding overhead.
  PARK STANDS (dedicated triangle-aware kernel project).

## CUDA complex-ConjTrans syrk delegates to herk (2026-09-27)
- Gap: CUDA `try_syrk` complex-`ConjTrans` unconditionally ran the
  host fallback (D2H A+C + scalar triple-loop `host_syrk_compute` +
  H2D C): 21ms@256, 177.5ms@512 (8.4x O(n^3) scaling) vs 3.35ms
  for the delegated NoTrans twin at 512 (53x gap).
- Fix: real `alpha`/`beta` now routes to new `blas23::herk_c32/c64`
  (row-major `C = a*A^H*A + b*C`, A k×n, is col-major herk OP_N on
  `A^T` with flipped uplo). Complex scalars keep the host fallback
  (result not Hermitian, no herk spelling). Post-fix: 0.9ms@256
  (24x), 3.5ms@512 (51x), 13.5ms@1024 (~105x vs extrapolated
  fallback). Matches delegated-NoTrans rate (3.35ms@512).
- Correctness catch: first herk run mismatched CPU by 0.25 =
  beta*imag(C0) on the diagonal — cuBLAS herk zeroes diagonal
  imag (BLAS-standard; input diag imag ignored). Follows vendor
  semantics deliberately: Hermitian C gives bit-close agreement
  (zherk-vs-CPU 3.7e-13, beta=0). Divergence exists only for
  non-Hermitian C input (contract-violating for a Hermitian
  update); documented in dispatch comment. Zero in-repo callers
  outside sonium-compute; no sibling risk.
- Tests: new `native_cuda_syrk_conjtrans_real_herk_matches_cpu`
  (c64 Lower/Upper x (a,b) {(0.5,0.25),(1,0)} vs CPU 1e-9 +
  diag-imag==0 assert; c32 Upper vs CPU 1e-3); stale fallback test
  renamed to `..._complex_alpha_uses_host_fallback` with corrected
  comment. 5/5 syrk_cuda green, 496 lib green, clippy clean (4
  pre-existing CPU-arc warnings only).

## T8 sq1k knee recheck: FALSIFIED, wobble is noise (2026-09-27)
- 3 full-sweep rounds gemmijk sq1024-f32 (hot): T1
  1.11/1.19/1.13x (ST-micro 1.13x PARKED, stable), T4
  0.89/0.99/0.96x, T8 0.91/0.88/1.12x, T16 1.13/1.46/1.11x.
  T8 knee (1.33-1.43x) never reproduces — FALSIFIED 3/3.
- Residue: single-round outliers in OUR MT absolute (T8
  2.87->3.84ms r3; T16 1.46x r2) while OB holds tighter.
  Placement-lottery test (taskset -c 0-7, 3 rounds): confined
  ours.abs 4.64/3.97/5.00ms — still wobbles AND uniformly
  30-50% slower than free; OB destabilizes under pinning again
  (r3 6.99ms, 2x). Confinement confounded (forced contention),
  not a stabilizer. Free placement stands.
- Verdict: wobble is single-outlier noise inside the known
  +-25% MT envelope, not a routable effect — building placement
  machinery on it would repeat the spread-router mistake (also
  born from 2-3 lucky runs). sq1k-sync stays MONITOR. No code.

## Beta-1 mechanism hunt + mul-skip SHIPPED (2026-09-27)
- Outlier kill: sq1024-f32 beta-1 +2.9ms (+14.5%) did NOT
  reproduce interleaved (+1.2%, noise) — single-process fluke;
  the DRAM-spill/L3-cliff theory built on it DISCARDED. Lesson
  (again): 1-run deltas are NO evidence, even at +14%.
- Thin-k gap REAL but smaller than logged: ST interleaved
  10/10 (b1>b0 every pair), +36-51us absolute (+4-6%) — the old
  "+15-30% per-FLOP" was ratio-contaminated (OB's OWN beta=1 is
  9% faster than its beta=0!). `beta_eff` insight: multi-slab
  runs already pay beta=1 on slabs 1+ in BOTH arms — square
  deltas are slab-0-only (~1%, noise).
- k-sweep (m/n fixed, k=64/128/256): delta +30/+39/+47us
  FLAT in k => epilogue-side, per-tile-fixed (~25cyc/tile).
  Pad-invariant (2KB heap shift: +13 vs +15us — address/
  aliasing theories DEAD). f64 2x confirmatory (+35us over 2x
  tiles = same ~25cyc/tile — chain-latency signature).
  Surviving model: ROB-head serialization of the C-read chain
  (load 12c + mul 5c + fma 5c ~= 22cyc; retirement-ordered, the
  1280-uop next-tile k-loop cannot drain past it).
- Methods: k=512 cross-process comparisons are THP-drift
  artifacts (triple-arm b0/b1/b0: ZERO beta effect, -130us
  position drift; promotion = step-drift + compaction stalls
  for >=4MB working sets). Triple-arm protocol neutralizes it
  (interpolated baseline); single-slab shapes (<=2MB) are
  drift-free. Also caught: grep-tail parse bug (compared
  b1-inter vs b1-hot) — always print full paired lines.
- SHIP: beta==1.0 multiply skip (1.0*x exact => bitwise
  identical), const-dispatched per tile (`SKIP_BETA_MUL`).
  CAUTION TALE: runtime flag + opaque gate call inside full6
  collapsed codegen ~2x (0.65->1.3ms; 15/16 YMM budget,
  regalloc spills) — const monomorphization is load-bearing
  here (NOTE left at the kernel). Proof: 7/7 same-binary wins
  (-12us/-2% mean), address-swap 3/3 (win follows content,
  alignment FALSIFIED), maxdiff digits identical.
- Rollout: all 6 epilogue kernels + rowband_f64_full (13
  dispatch sites), clean ship, no gate (TRSM convention).
  Verify vs stashed baseline: thin-f32 b1 -14us/-2.1%, tails
  (130x2050) -18us/-2.7%, f64 gap +35->+11us (within-process
  triples), b0 IMMUNE (identical means 0.6515=0.6515), f64
  maxdiff 0e0=0e0. 496 green, clippy pre-existing-only.
- Ledger: beta==1.0 CLOSED (fast path shipped). General beta
  (0.37 etc) keeps the +25cyc/tile chain — needs bigger tiles
  (register-blocked), C-load software pipelining (no free regs
  in k-loop), or blocking restructure. TRSM-update beta=1
  inherits the win (updates = most TRSM flops).

## SYRK ST wide bands SHIPPED (2026-09-27)
- Root cause (measurement + code): per-band B re-pack scales
  n²k/2nb (66MB @nb=32 for 2048x256) + n/nb A re-reads (64x) +
  dispatches. New gemmnt probe: NT 32x1024x1024 collapses to
  44G (multi-slab narrow) while k=256 narrow holds 75-81G and
  f64 narrow holds 1.18x (scalar pack 4KB-immune) — f32
  vectorized trans-pack thrashes at 4KB pitch (k=1024, all
  row-groups alias one L1D set + output-stream eviction);
  per-slab-fixed pack cost sinks narrow-m, square amortizes.
  4/4 predictions confirmed (f64 immune, k=512 intermediate,
  m=64 halfway, k=2048 pitch-locked at 44G).
- Fix: ST-only wide bands (nb=256, `SYRK_NB_SERIAL`): pack
  66->8MB, A re-reads 64->8x, dispatches 64->8. ST f32
  1.50-1.93x -> 1.14-1.31x (-6..-33%), f64 1.18-1.27x ->
  1.06-1.12x (-5..-17%, ~CLOSED). k=1024 extra (1.93x) fixed
  via 9x less pack (no scalar-pack detour needed). f32
  residual = band-kernel NT rate (m=256 NT 1.11-1.27x) +
  ~5% banding.
- nb sweep (f32 ST): 32/128/256/512 => 256 best all 5 cells
  (128 +2-10%, 512 collapses 1.39-1.69x: 1.25MB streaming
  vs 512KB L2, B-tile evicted between 8us touches — turnover
  < touch-interval cliff; 256 sits at the edge).
- Serial-nest forcing (nest_macs=MAX at 1 worker): -3-5%
  big shapes (killed MT-dispatch machinery per band).
- Small-shape guard: wide loses <=768 (L2-thrash + n·nb/2
  stash; -28% at n=300) => threshold n>=1024 (crossover
  measured both precisions; small keeps tuned narrow, immune
  +-2%). Memcpy-stash trial REVERTED (+2-9%: scalar loops
  already autovectorize; libc calls add pure overhead).
- MT immune BY CONSTRUCTION (workers>1 => nb=32 path, stack
  stash, identical executed code; stash branch + empty Vec =
  ~2cyc/band) + consistent-with-noise runs (MT +-20% swamps
  single comparisons; identical-code -22% swings observed).
  Zero team band-plan callers; Trans path needs no stash.
- Tests: new `syrk_band_wide_heap_stash_matches_reference`
  (nb 32/33/256 x Lower/Upper x NoTrans/Trans x beta
  0/0.5/1 vs naive + bitwise untouched-triangle),
  sensitivity-proven (FAIL on sabotaged stash size, PASS on
  revert). 497 lib green, clippy pre-existing-only,
  solvers-direct full suite green (273+3+1+4+1, Cholesky
  consumes wide bands ST).
- Ledger: SYRK-ST CLOSED (f64 ~parity, f32 1.14-1.31x =
  kernel-bound). Remaining f32 = NT mid-m kernel work (not
  SYRK-specific) + MT f32 1.36-1.60x (separate, balance/
  traffic-bound per earlier trial).

## Ambient-sized spread pools (GEMV MT sub-width) SHIPPED 2026-09-28
- Trigger: NoTrans-L3 re-diagnosis. ST pad-probe (gempad, x/y offset
  0/256/1024 els, RAYON=1 taskset-64): ST shows ~parity already
  (sq1k 1.0-1.3x, sq4k 1.00-1.05x, tall 0.80-0.87x WIN) and pad-FLAT
  => 4K-aliasing FALSIFIED, gap is MT-only. MT t-sweep then exposed
  a NON-MONOTONIC t=8 catastrophe (sq4k-f32 11.6-12.8x, ours-absolute
  9x worse than t=16 with OB flat in-window => structural, not noise).
- Root cause: broadcast requires ambient >= full spread width (16);
  below that the stealing raw scope runs, and stealing AMPLIFIES with
  worker count (t=4 plateau 0.87ms -> t=8 1.9ms: more workers, more
  band migration across L3 slices, more fabric traffic; t=16
  broadcast 0.2ms). The code even named the fix ("ambient-sized
  spread pool would also win below this floor (follow-up)").
- Fix: cached W2/W4/W8 spread pools (same pin fn, worker i = same
  spread CPU as full-pool worker i) + `spread_pool_for_ambient`
  (largest cached width <= min(ambient, full); ambient >= full =>
  full pool = legacy-identical there; odd widths floor; <2 => raw).
  4 call sites (NoTrans f32/f64 + Trans f32/f64) swap the predicate;
  broadcast bodies untouched (already width-generic).
  `SONIUM_GEMV_AMBIENT=0` restores legacy for A/B (default-on).
- A/B (gemvab, rebuilt lib, load 12-17 — effects 2-10x >> noise):
  t=16 IMMUNE verified identical (10/10 cells match to noise);
  t=12 (W8 undersubscribed!) sweeps 10/10 (+12..+67%, placement >>
  width); t=8 catastrophe HEALED (10/10, +13..+64%, worst 2.4x vs
  OB from 12.8x); t=4 sweeps L3-resident 8/10 (+20..+55%,
  sq4k-f64 ties = DRAM-resident theory-consistent); t=2 NoTrans
  wins 3-24%, Trans ties.
- t=2 Trans-f64 red cells (-4..-9% cross-process) FALSIFIED as
  load-artifact: same-process interleaved A/B (amb2ab, env flipped
  per rep, x3 runs): sq1kT 1.000/0.959/1.037x (tie, mean 0.999x),
  sq4kT 0.985/0.968/0.963x (consistent ~3% win). No red cells remain.
- Bitwise: new `spread_pool_for_ambient_floors_width` (invariants +
  exact floors at full_w=16 + legacy rule, deterministic, no timing)
  + `avx2_gemv_notrans_ambient8_matches_raw` (W8 broadcast == raw).
  499 lib green, clippy pre-existing-only (4, none in touched files).
- Ledger: t-robustness CLOSED (any RAYON count 2-16 now sane).
  Remaining @t=16: sq1k/tall 1.4-2.1x + sq1kT-f64 ~2x = small-shape
  dispatch project (sq1k: ours 20us vs OB 12us, broadcast wake on
  16 workers for 1M MACs; width-caps per shape are the next lever).
  Tall-hot anomaly (t=2 raw hot worse than inter) retired with raw.

## Quad row-dots (GEMV NoTrans MT) SHIPPED 2026-09-28
- Trigger: width-caps FALSIFIED (SONIUM_SPREAD_CPUS 16/8/4 at t=16:
  narrower = monotonic-worse EVERYWHERE incl sq1k, -15% to -260%;
  small shapes want all 16 slices too). Reframe: unified NoTrans-L3
  as a PER-SLICE slope gap (~0.6x ours-vs-OB on sq1k/tall/sq4k alike,
  both precisions) => kernel x-refill contention (x re-streams from
  L2 every row, shares per-core fill hardware with A; pairing's
  1.07-1.26x = the x-halving paying off; tall-single trails worst).
- Fix: row_dot4 (4 rows x 3 chains, 1 x-load -> 4 FMAs, x/4 traffic,
  13 live vectors spill-free) + sweep quad loops + pair remainder +
  single tail, both precisions, both beta branches. Gate
  SONIUM_GEMV_DOT4 (default-on level 2 = n>=256; =1 narrows to
  n>=1024; =0 legacy). MT-only by measurement: ST is per-row-MLP-
  bound and loses to 1.36x on quads, so `allow_quad=false` on the
  serial path, single-worker scopes, and width-1 broadcasts.
- Correctness saga (bitwise-any-split preserved): (1) quad grouping
  depends on band start (broadcast r0=38k misaligned vs raw r0=64k)
  => 1-ulp diffs + lock-poison cascade; (2) spanning quads are
  unfixable by tiling (rows live on two bands) => broadcast bands
  rounded UP to multiple-of-4 (free on pow2 m; <=1 tail idles
  otherwise) + canonical pre-drain (quad-active only, legacy
  untouched); (3) masked-ST ran broadcast-width-1 (deduped spread
  set) with quad-on => `width > 1` rule (found via DIAG, reverted).
  New `sweep_quad_split_bitwise` (aligned cuts 36/100/104 bitwise,
  misaligned 38/101 tolerance) + quad arms in both pairing tests.
  500/500 green x3, clippy pre-existing-only.
- A/B: MT-16 quiet: sq4k-f32 +8-13% (1.94x->1.43x), sq4k-f64 +21-24%
  (1.28x->1.05x), sq1k +5-17%, sq8k-f32 +11% (extends lead to 0.68x),
  tall +3-7%, sq8k-f64 tie (DRAM-bound, theory-consistent). MT-16
  interleaved (null 0/0 ties exactly, no order bias): all small/mid-n
  cells win-or-tie incl gate-2-only n=256-768 (tall +5-13% x3,
  r2k768 +10-20%). t=8 spot: +5-21% all cells. ST masked: all tied
  (width-1 rule verified). One poisoned cross-process cell
  (sq1k-f32 storm-pocket, OB 33% off-window) resolved by quiet rerun
  (+5-14%) + 4/4 interleaved wins. Trans untouched (immune).
- Ledger: NoTrans-L3 slope CLOSED toward kernel-bound (sq4k-f32
  1.43x, sq4k-f64 ~parity, sq1k 1.4-1.6x = dispatch floor).
  Remaining BLAS-2: Trans residuals (f64-1536 PARKED), sq1k floor.

## Trans-fanout falsification trilogy (f64-1536 stays PARKED) 2026-09-28
- Attack 1, register-fused 4-row tiles (priv in regs, 4x less LD/ST,
  bitwise-identical op order): LOST f64-1536 1.38x, f64-2048 1.25x
  (f32 ties/+9%). Cause: 4-deep FMA latency chains per element (the
  single-row tile is NOT port-bound; OOO already hides it).
- Attack 2, 2-row independent-acc tiles (8 chains, no cross-row deps,
  tolerance-equal): LOST f64-1536 1.20x, f64-2048 1.16x. The tile is
  neither port- nor MLP-starved — both fusion directions dead.
- Attack 3, priv-base 4K-alias offset: fixed 2KiB won f64-1536 +24%
  once, then flipped sign per process (ASLR lottery, not aliasing);
  offset-amount sweep (0/1K/2K/4K) all-tied; ADAPTIVE always-2KiB-
  away offset LOST consistently (f32-2048 -26% 4/4, f64-1536 -4-14%
  4/4). De-collision hurts => aliasing is NOT the mechanism.
- All three reverted clean (zero refs, 500 green). PARKED stands and
  is now triple-strengthened: the f64-1536 gap (111 vs 195 GB/s,
  all config/kernel/placement levers dead) genuinely needs counters
  (`paranoid=4` blocks `perf`) or a from-scratch kernel design, not
  another tile tweak. Do not retry fusion or offsets without counters.

## Wide MT-SYRK bands (nb=64, k-gated) SHIPPED 2026-09-28
- Trigger: MT-SYRK 1.36-1.60x f32 = DRAM-traffic-bound (128MB A
  re-reads + 66MB pack at nb=32 = 87GB/s demand >> DRAM). nb=64
  halves traffic at coarser LPT balance (32 bands/16 workers).
- A/B (syrkab, t=16): 3 quiet rounds x 6 big cells = 18/18 favor
  nb=64 (+4-17%; same-code thin controls set the +-12% MT noise
  floor, wins exceed it in aggregate). Ship-confirm default-vs-=0:
  +6-12% all big cells. Thin-k DANGER found and gated: 976x48 lost
  -50% at nb=64 (dispatch-bound regime), so nb=64 applies iff
  k>=256 (plan-level gate, thin/small/ST immune by construction;
  Cholesky k<=128 unaffected).
- Ratios: f32-2048x256 1.46x->1.33x, f64 1.27x->1.17x,
  f32-2048x512 1.66x->1.36x, f64 1.45x->1.30x, f32-1024^2
  1.52x->1.39x, f64 1.17x->1.09x. `SONIUM_SYRK_MTNB=0` restores
  nb=32 for A/B. nb=128 untested-quiet, dropped (exact-fill risk).
- Tests: new `syrk_band_plan_mtnb_gates` (deterministic plan tuples
  incl k-gate/narrow/ST immunity + legacy arm). 501 lib green,
  solvers-direct full suite green (273+..., Cholesky consumes the
  shared plan), clippy pre-existing-only.
- Ledger: MT-SYRK narrowed to 1.09-1.39x (kernel/balance class).

## N-split shared A^T pack, f32 footprint-gated (16 MiB) SHIPPED 2026-09-28
- Trigger: fresh battery showed sq1kT-f32 1.40x hot (N-split tasks
  re-gather strided A per task/slab/tile). Shared whole-A^T pack
  (TLS-cached `with_packed_a_trans_f32`) + stride lie `(k,1)`.
- Crossover (same-process interleave, storm-robust): pack LOSES
  +4-11% at <=8 MiB A (gathers hit L3 ~free — the reverted-trial
  NOTE confirmed), flip-flops at 12 MiB, WINS solidly at >=16 MiB
  (-6-20%, 3/3 each: sq2kT -11-13%, sq4kT -16-20%, 1-tile -6-7%,
  1x1 -13-20%). No tile-count term (N-tasks re-gather per task;
  even 1x1 wins). Gate: m*k >= 4M els (16 MiB), default-on,
  `SONIUM_GEMM_APACK=0` = legacy gather. Complements (not
  contradicts) the reverted f32 trial: gather below L3, pack above.
- Refactor: N-task spawner extracted (`spawn_n_tasks_f32`), both
  arms share one body (differ only in A base/strides). Bitwise:
  same values + sweep order => new
  `avx2_trans_a_n_pack_matches_gather_bitwise` (2048^3 + gate
  asserts incl the 12 MiB flip-zone). 502 green, clippy clean.
- Immunity: sub-gate shapes execute the legacy arm (gate asserts;
  sq1kT same-code +-15% = MT noise under load). f64 twin NOT done:
  f64 already packs MT Trans-A (PACK_A_MT default-on) yet trails
  1.58x — different bottleneck (queued, needs fresh diagnosis).

## 2026-09-28 — f64 MT Trans-A pack: `n >= 1024` column floor (SHIPPED)

- Diagnosis (gemmatab64, new probe, same-process interleave): the
  pack-once-shared payoff scales with N-split band count, but the gate
  had no `n` clause. Unbiased A/B (rotated pair order + 2 warmup pairs,
  m=k=1024, maxdiff 0e0): pack/gather 0.76-0.81x at n=256 (6/6),
  0.89-0.94x at n=512 (2/2 unbiased), 1.21-1.23x at n=1024,
  1.41x at n=2048; m=256 control ties 0.995x/1.010x.
- CAUTION logged in code: first 4 runs without order rotation showed
  n=512 winning 3/4 via warmup bias (identical-code control drifted
  to 1.14x); unbiased reruns reversed it. All future A/B probes must
  rotate pair order.
- Gate: `PACK_A_MT_MIN_N = 1024` (crossover in (512, 1024],
  conservative at proven side). Post-ship probe: n=256/512/512³ tie
  at gather speed (0.97/1.04/1.03x, both arms gather), n>=1024 keeps
  pack wins (1.22/1.39/1.42x). 512³ scalar/bitwise tests widened to
  512×1024×512 to stay pack-engaged. 502 green, clippy clean.
- Residual: even the winning pack arm trails OB ~1.3-1.5x at sq1kT
  and our NoTrans sq1k (6.1ms) trails OB's Trans (4.3-5.0ms) — the
  dtype-independent sq1k floor dominates (queued next).

## 2026-09-28 — f64 MT Trans-A pack: TLS buffer (SHIPPED, default-on)

- Root cause of the f64-Trans premium: the MT pack arm allocated a
  fresh `vec![0.0; m*k]` EVERY call (~2000 page faults at sq1k, faults
  scaling with m*k). Fix: pack into the pre-faulted `DGEMM_A_PACK` TLS
  buffer (same parallel pack + NoTrans recursion; `=0` kill-switch).
- A/B (gemmatab64, rotated order, maxdiff 0e0, 3/3 runs): TLS/fresh
  1.23x/1.34x/1.23x at sq1kT, 1.20x/1.23x/1.21x at sq2kT,
  1.13x/1.18x/1.16x at n2k-m1k; gate-excluded sq512T control ties.
  New `avx2_trans_a_mt_tls_pack_matches_fresh_bitwise` test.
- Effect (clean OB-free rig): f64T-2k 38.2 -> 29.9ms (1.04x vs OB
  28.6); f64T-1k TLS-arm 4.8-5.0ms (~1.05x vs OB 4.5-4.7).
  503 green, clippy clean.
- Rig upgrades (same arc): OB-in-process SPINNERS found polluting our
  reps +26-38% (f64T 7.1 alone vs 9.0-9.8 co-located; thermal/order
  falsified first) -> gemmalone/obalone split binaries are now the
  clean rig; meds float with run-state, mins are capability; first
  case in process runs pool/freq-cold (sq1k-NT 6.1 cold -> 4.4 warm).
- Status: f64 GEMM sq1k+ at parity (NT 1.02-1.05x, T ~1.05-1.10x),
  sq2k both dtypes NT at parity. REMAINING: f32 (NT-1k 1.35-1.5x,
  T-1k 1.56-2.0x gather, T-2k 1.35-1.45x APACK).

## 2026-09-28 — f32 MT Trans-A pack: parallel pack (SHIPPED, default-on)

- Root cause of the f32-Trans premium at/above 16 MiB: the MT apack arm
  packed SERIALLY (~4 ms at 16 MiB) while the f64 twin packed in
  parallel (~0.5 ms). Fix: ported `pack_a_trans_f64` to f32
  (`pack_a_trans_f32`, same row-band split) + MT lender
  (`with_packed_a_trans_f32_mt`, `SONIUM_GEMM_APACK_PAR=0` kill-switch).
- A/B (gemmatab PAR, rotated order, maxdiff 0e0, 2/2 runs): par/serial
  1.29x/1.25x at sq2kT, 1.14x/1.09x at sq4kT; t16m1/slab1 tie
  0.97-1.00x (M-split: 4n<m so the N-split-hooked pack never engages
  — immunity by construction); gated-out sq1kT control ties.
  New `avx2_trans_a_n_par_pack_matches_serial_bitwise` test.
  504 green, clippy clean.
- RECORD CORRECTION: the APACK ship notes claimed narrow-n wins
  (t16m1 -6-7%, slab1 -13-20%) — those shapes take M-split and never
  touch the pack; the "wins" were unrotated-probe warmup bias. The
  16 MiB gate stands justified by the big-square wins alone (11-20%,
  far above any bias). Default-path verification pending (storm).
- NEXT: re-tune the 16 MiB footprint crossover with the now-cheap
  parallel pack (predict: crossover drops; sq1kT-f32 may pack).

## 2026-09-28 — f32 MT Trans-A pack: footprint gate 16 MiB -> 4 MiB (SHIPPED)

- Re-tuned with the parallel pack (the 16 MiB crossover was measured
  with the ~4 ms serial pack; the parallel pack costs ~0.5 ms).
  Rotated A/B, maxdiff 0e0, 4/4 runs at every size: pack/gather wins
  1.26-1.47x at 4 MiB (sq1kT), 1.36-1.62x at 8-64 MiB, but only
  1.01-1.07x at 1 MiB and 0.93-1.00x at 256 KiB. Gate at 4 MiB
  (1M els), conservative at the proven side; test asserts updated.
- Effect (default path): f32T-1k 4.3 -> 3.7ms (min 3.26; 1.40-1.62x
  vs OB 2.3, was 1.56-2.0x). 504 green (one storm-drain flake noted,
  green 2/2 since), clippy clean.
- Status: f32-T-2k 1.12-1.18x, f32-T-1k 1.40-1.62x, f32-NT-1k
  1.29-1.45x; f64 all-parity; sq2k NT parity both. REMAINING: f32
  sq1k floor (NT +1ms, T premium over NT).

## 2026-09-28 — f32 N-split CCD-compact route (SHIPPED, default-on)

- Root cause of the t16 disease (inverted-U scaling: we beat OB at
  2/4/8T but lose at 16; f32 anti-scales 8->16): the 16-worker mirror
  pool spans 2 CCDs while 8 workers fit 1 CCD; cross-CCD `A` sharing
  costs f32 ~30% at sq1k. Bands exonerated (cap A/B: fewer bands =
  slower everywhere; default optimal; maxdiff 0e0 across counts).
  Box: Threadripper 3995WX (16MB L3/CCX, 8 cores/CCD).
- Fix: f32-NoTrans N-split with `A`+`C` <= 8 MiB routes to the
  1-CCD wide compact pool (new `scope_compact_wide` + accessors;
  sizers threaded with the executing width via a `workers` param on
  `cols_per_task`; `SONIUM_GEMM_CCD8=0` kill-switch). NoTrans-only:
  Trans hits L2-set conflicts at some C phases under wider compact
  bands (wins 2/3 seeds, loses one — needs tile-coloring; stays wide).
  f32-only (f64 compute-bound, scales to 16); sq2k+ (32 MiB) stays
  wide (16 wins 1.7x there).
- A/B (rotated, shared-C, 9 address seeds): sq1kNT wins 8/9 seeds at
  1.32-1.45x (worst seed ties 0.99x); sq512NT 1.02-1.28x (worst
  0.99x); sq2k/rect/f64/@8 controls all tie 0.96-1.05x. Probe lesson:
  per-arm C buffers confound routing with aliasing — shared-C plus
  offset seeds are now mandatory for routing A/B.
- Effect: f32-NT-1k 3.4 -> 3.0ms (~1.29x vs OB 2.33; compact runs at
  ~78% of FMA peak — remaining gap is unserved parallelism, needs
  per-task A-packing to feed 16). 505 green, clippy clean.
  New `avx2_ccd8_matches_wide_bitwise` test.
- Status: f64 GEMM at parity; f32-NT-1k 1.29x, f32-T-1k 1.4-1.6x,
  f32-T-2k 1.12-1.18x, ST 1.13x both. REMAINING: per-task A-pack
  (the 16-worker feed), f32-Trans L2-coloring, ST microkernel, GPU leg.

## 2026-09-28 — Trans-compact FALSIFIED (reverted clean)

- Tried to extend the CCD8 route to Trans-A: cpt<=64 narrow bands
  (L2-safe 256KB tiles) won 8/9 address seeds (1.08-1.20x) but lost
  one seed 0.79x; a 2KB pack-buffer offset + 4-arm (routing x offset)
  A/B then showed seeds DON'T reproduce across builds — the L2-set
  conflict is ASLR-lottery (TLS-pack/B-tile fixed addrs vs caller
  bufs re-roll every exec), unfixable by static offsets. Per-task
  A-pack also washes out on paper (pack cost ~= cross-CCD stall
  saved at 1 tile/pass). NT-only CCD8 stands; Trans needs a
  padded-stride pack kernel (stride != pow2 kills the conflict class)
  — queued as a project, not a tweak.

## 2026-09-28 — Serial A-panel pack (SHIPPED, default-on)

- Clean ST curves killed the old 1.13x story (storm artifact): ST is
  parity-or-better <=1k (sq256 we win 1.6x!), thin gap only at sq2k
  (f32 1.105x +15.7ms, f64 1.042x +13.5ms; superlinear growth =
  L3-slice-16MB overflow, serial A re-reads 128 MiB vs ~16 MiB).
- Fix: pc-outer/jc/ic-inner nest packing Mc x kc A-panels once per
  (pc, ic) at the first jc-tile, cached in TLS across jc-tiles
  (256KB panels, B-traffic unchanged, pc-ascending = bitwise).
  Gate: NoTrans-A, all-blocking-multi, A >= L3 slice.
- A/B 3/3 (rotated, RAYON=1): f32 1.017x/1.025x/1.030x,
  f64 1.045x/1.033x/1.012x at sq2k; sub-slice controls tie
  0.995-1.000x. 507 green (2 new bitwise tests), clippy clean.
- Status: f64 GEMM at parity incl ST; f32-ST-2k ~1.07x residual
  (C-traffic/B-pack). REMAINING CPU: f32-T-1k 1.4-1.6x (Trans
  premium over NT-compact), f32-NT-1k 1.29x (16-worker feed).

## 2026-09-28 — Trans-gather+compact FALSIFIED (reverted clean)

- Tested routing Trans-gather (APACK forced off) to the 1-CCD pool:
  8/9 seeds tie-or-micro-win (+0-5%), one 0.93x (gather is DRAM-bound
  so CCD-locality is irrelevant), while gather-wide itself (5.6-6.0ms)
  trails pack-wide (3.5ms, XO-proven). Trans stays pack+wide:
  pareto-optimal given current kernels. NT-compact sanity re-verified
  in-run (8/9 wins 1.29-1.42x). f32-T-1k (1.4-1.6x) and f32-NT-1k
  (1.29x) now need projects (padded-stride pack, per-task/2-pool
  feed), not tweaks. CPU GEMM work paused here; GPU leg next.

## 2026-09-28 — CUDA f32 batched local-contract -> cuBLAS (SHIPPED)

- Rate probes (new cubatch, sync-per-rep device time, idle RTX PRO
  6000): TRSM delegates at vendor rate (1.00-1.02x, maxdiff 0e0 —
  nothing to do); the native one-thread-per-output batched kernel
  trails strided-batched 3.2-7.7x on f32 (monotonic-worse with work)
  while f64 native wins 8% everywhere (launch-dominated).
- Fix: f32 `try_batched_local_contract` delegates to
  `cublasSgemmStridedBatched` via the swap trick (new
  `blas23::gemm_strided_batched_f32`; `SONIUM_CUDA_BATCH_BLAS=0`
  restores native; i32 overflow falls through). f64/c32/c64 stay
  native. Post-ship: f32 1.001-1.009x, maxdiff 0e0; kill-switch
  restores 4.55x native. New `tests/batched_cuda.rs` (fast+kill
  arms vs host reference, f64 native vs host). Clippy clean.
- Probe lesson: raw cuBLAS on a different stream than backend H2Ds
  needs an explicit sync or maxdiff explodes (cross-stream race).
- GPU status: dense BLAS (GEMM/TRSM/SYRK/HERK/batched-f32) at vendor
  rate; f64-batched native-kept. REMAINING: cuSPARSE side (SpMM).

## 2026-09-28 — GPU leg COMPLETE (all at vendor rate)

- New cubatch/cuspvmv probes (sync-per-rep device time, idle RTX PRO
  6000 Blackwell): TRSM 1.00-1.02x (maxdiff 0e0, delegation verified,
  nothing to do); SpMM 1.00x (ours = raw csr_alg2 exactly); SpMV
  1.003x maxdiff 0e0; GEMM/GEMV 1.00x; HERK incl ConjTrans-real
  delegated (the "fallback" probe label was stale — 13.5ms at n=1024
  is the delegated rate; host fallback remains only for complex
  alpha/beta, mathematically necessary).
- Shipped: f32 batched local-contract -> cuBLAS strided-batched
  (3.2-7.7x -> 1.001-1.009x); f64 stays native (wins 8%).
- Full scorecard: CPU f64 GEMM at parity (MT+ST); f32-NT-1k 1.29x,
  f32-T-1k 1.4-1.6x, f32-T-2k 1.12-1.18x, f32-ST ~parity; GPU dense
  + sparse all at vendor rate. LAPACK lives in solvers crates (out
  of scope; inherits BLAS only).

## 2026-09-28 — Padded-pack Trans-compact (REVERTED) + A/B methodology fix

- Built the queued padded-stride pack (`pack_trans_f32_pad`, pstride
  threading through `pack_a_trans_f32`/MT lender/driver, `panel_stride`
  k+16): sq1kT padded-compact won 24/27 seeds (1.24-1.49x, 3 ties, 0
  losses over 4 execs). Bitwise test green. REVERTED (zero residue) on
  three independent vetoes: (a) rect-Trans REGRESSIONS same-binary
  (r1k512T 0.76-0.92x 5/9, m512n2kT 0.75-0.99x 9/9; NoTrans rect wins
  9/9 so it is Trans-pack-specific, not band underfeed); (b) the sq1kT
  "win" was measured under rayon idle-spin contamination (below) —
  true blocked value ~1.19x in unlucky layouts only; (c) a per-binary
  2.3-vs-3.7ms wide-Trans lottery DWARFS the win and flips its sign
  per app binary (lucky apps would LOSE 2.3->3.1).
- METHODOLOGY (mandatory): rotated same-process A/B is INVALID across
  rayon pools — the just-parked pool's threads spin down while the
  other arm runs (16 mirror spinners halve CCD0 for the compact arm
  and vice versa; first-rep-after-switch 5-9ms is the tell). All
  cross-pool routing A/B must be BLOCKED (8x arm A, 8x arm B, compare
  steady blocks). Same-pool rotated A/B (env gates, one pool) is
  unaffected (verified: rotated == blocked to 1%).
- Correction: shipped CCD8-NoTrans blocked-verified at 1.17x
  (wide 3.35-3.48 vs ccd8 2.73-2.97, trajnt) — the rotated 1.28-1.45x
  claim was spin-inflated but the win is REAL; route stands.
- Open: per-binary wide-Trans lottery (identical lib bytes+math:
  2.3ms in one probe binary, 3.6-3.7 in four others, 7/7 stable
  across interleaved execs, ASLR-independent). Ruled out by
  experiment: heap KB-phases (11-pad sweep flat), heap MB-order
  (a,b,c vs b,a,c flat), argv/stack offset (identical-bytes copies +
  short-name copy all 2.3), thermal/load/time (interleaved),
  rotation-vs-blocked (same-binary equal), fn-alignment (align-64
  perturbs +-10%, no unify). Standing hypothesis: frontend/BTB
  placement (RIP-low-bits static per binary, ASLR-independent) —
  untested; needs codegen-perturbation draws or PMU (paranoid=4
  blocks counters). Lucky draw ~= OB rate, so killing the lottery
  IS the T-1k gap.

## 2026-09-28 — Per-binary lottery DEAD (L3-warmth) + true gap re-measured

- The 2.3-vs-3.7ms "per-binary lottery" is box-state, not content:
  on the idle box ALL probe binaries (traj/trajB/trajN8000/rectabA2)
  ramp to 2.3-2.6 tails within ~20 reps; last turn's flat-3.7 ran on
  the busy box (L3 polluted, capped). Mechanism: our per-rep L3
  footprint (pack+C+B+A-src ~16 MiB) warms slowly; short execs and
  busy-box L3 pollution sample the cold rate. Same-binary rotated
  A/B remains valid (equal warmth); cross-binary absolutes must be
  compared ramped (tails/min) on the same box state.
- True gap, min-min idle split rig: sq1kT 2.90 vs OB 2.10 (1.38x),
  sq1kNT 2.86 vs 2.27 (1.26x). OB is flat (med~=min, warms in 1-2
  reps via cache blocking); ours ramps over ~20 reps.
- Diagnosis: N-split 16x A re-read tax (whole-A per pass per task:
  64-256 MiB/rep vs OB's ~16 MiB read-once 2D blocking) + 4x C
  revisit (Kc passes). Queued THE project: 2D partitioning for
  large-K (each matrix read ~once total); the existing 2D path is
  thin-k-only (k <= 256).

## 2026-09-28 — f32 N-split wide-Kc 512 (SHIPPED, default-on at k>=768)

- Halving N-split passes (256->512 at k >= 768, `f32_kc_nsplit`,
  `SONIUM_GEMM_KC=256` kill-switch): halves whole-A restream + C
  revisit; wider B tile (512x256 = L2-size) costs L2-residence.
  Same-process 3-arm rotation (kc256/512/1024), 3 seeds each:
  kc512 wins sq1kT +3-8%, sq2kT/NT +3-8%, r1k512T/NT +6-16%,
  sq768NT +3-9%, sq512NT +14%, m512n2kT +7-8%; ties sq1kNT/sq768T/
  sq384/noise-scale + M-split controls; loses sq512T 0.90x 6/6
  (small-C: revisit free, pure B-spill — hence the k>=768 floor).
  kc1024 wins some T but loses NT/sq2k 1.16-1.45x — not shipped.
- Correctness: pass boundaries regroup fp terms (not bitwise vs
  Kc=256 — same class as the shipped single-pass 2D path); new
  `avx2_kc512_matches_scalar` (512x512x768 T+NT vs scalar, 1e-4/1e-5,
  same tolerance as the multi-pass test). Suite 508/508 (5/6 runs;
  1 unidentified flake, storm-drain class, green since).

## 2026-09-28 — f32 N-split CCD-compact route (REVERTED, zero residue)

- The CCD8 win was an ABBA-order artifact: the order-fair retest
  (fresh buffers per block, ABBA arm order) won only sq896/sq1k
  (+10-35%) and lost ~45-58% everywhere else — an un-gateable
  resonance. The 8-vs-16-core deficit beats any 1-CCD gain.
- Fully reverted: driver branch, `ccd8_on`/`CCD8_MAX_BYTES`, doc,
  `avx2_ccd8_matches_wide_bitwise`, `scope_compact_wide`/`compact_wide_*`
  accessors (`grep CCD8T|ccd8|compact_t|pstride|pack_trans_f32_pad` = 0).

## 2026-09-28 — 2D-largeK partition (REVERTED, zero residue)

- Rectangles + Kc passes + packed-2D-Trans recursion: squares lost
  0.54-0.92x; only narrow-n won (a feeding effect, better served by
  N-split sizing). Fully reverted.

## 2026-09-28 — Suite flake root-caused: cross-test env race (FIXED)

- Bisect (D-set fail / E-set pass, deterministic 2/2): `gemm_m_split`'s
  `M_SPLIT=1` window overlapped `par_pack`'s two arms and flipped
  GEMM routing between them, breaking the bitwise assert. Pack-vs-gather
  itself is bitwise (pgbw probe ndiff=0); pack-band splits and TLS
  lenders (guards held across use) are sound.
- Fix: all `SONIUM_*`-flipping lib tests now hold the single
  crate-wide `parallelism::env_test_lock()` across set/call/restore
  (GEMM x8, TRSM, SYRK folded in; module-local `ENV_LOCK`s in
  `blas2::avx2`/`syrk` removed). Non-mutating tests stay correct under
  either arm (all arms scalar-equal). Suite 507/507 green x2.

## 2026-09-28 — f32 full6 single-pointer A addressing (SHIPPED, bitwise)

- One running A pointer + five GPR row offsets (free AGU
  `[base+off]`) instead of six advanced pointers: 28→24 x86 insns
  for the same 12 FMAs. Same-process ST A/B (alternating order, OB
  ref, 3 seeds × k512/1024/2048): 17/18 cells 0.975-1.006x (~1%),
  bitwise exact everywhere (ndiff=0). Decode-bound theory falsified
  (+1%, not +10% — a shared port bind); offs passed by pointer so
  LSR cannot regenerate the six-pointer loop (a naive rewrite folds
  back silently — verified in asm; robust across LLVM, no
  `black_box`). Falsified along the way: Mc-panel tiling
  (pass/panel-outer, bitwise, ties/loses — not L2 residency),
  kc1024 (wins sq1k, catastrophic sq2k/sq512 — kc512 stands),
  k×2 unroll (prior zero = spill, not absence of bind — moot now).
- OB ZEN kernel structure (disasm): 48-FMA unrolled body with
  lane-split 2-k path (`vbroadcastsd` + `movsldup/shdup` pairs) —
  amortizes decode/branch over 4× our body; a future bigger unroll
  without spills is the remaining kernel lever (~5-8% ST).

## 2026-09-28 — Pin/placement audit (loaded-box artifacts, no change)

- Pinned mirror pool beats `SONIUM_SET_AFFINITY=0` float (sq1kNT
  hot 1.17x vs 1.25x, OB-normalized back-to-back) — pinning stays.
- taskset-16-31 confinement (we beat OB 0.87x/0.72x) is confounded:
  OB keeps a 63-thread spinner pool that contends when confined;
  cores 0-15 also host the box bg load. Placement is not the idle
  gap; do not fit policy to loaded-box confinement runs. Lesson:
  never compare OB-confined vs OB-free; same-process OB norm only.

## 2026-09-28 — f32 N-split narrow bands (SHIPPED, default-on, bitwise)

- Narrowest bands (`nc = 16`, B tile + multi-pass C revisit
  L1-resident) subject to A-fit (`bands × A ≤ L3`), full pool waves,
  an 8-wave spawn cap, and a 4M-MAC task floor
  (`SONIUM_GEMM_NARROW=0` kill-switch). The legacy 150M-MAC floor
  forbade L1-resident bands and cost sq1k ~13-24%.
- OB-bracketed ship A/B (16T, bitwise ndiff=0 everywhere, ±2%
  meter): sq1kT 0.76x, sq1kNT 0.77-0.82x, sq768 0.84-0.86x,
  r1k512N 0.65x, m512n2k 0.72x, k256 0.78x, sq512NT 0.86x,
  r1k512T 0.91x; sq2k/sq512T/thin-m/f64 tie (0.98-1.02x).
  Narrow-vs-OB now ~0.93-1.17x on the former gap shapes (loaded).
- Falsified inside: partial waves strand workers (sq768 24-band
  1.10x vs 16/48 — rule snaps to full waves, up when A-fit allows);
  single-pass shapes with legacy-B already L1 lose setup only
  (k128 1.036x — `f32_narrow_skip_1pass_l1` keeps legacy there).
- Method note: plain alternating medians showed 1.15x on
  IDENTICAL code (f64 control) under load — OB-bracketed rounds
  (OB,A,OB,B + per-round ratios) are now mandatory for ship
  decisions on this box; the f64/thin-m identical-code controls
  are the artifact meter.

## 2026-09-28 — 64B pack-buffer alignment (SHIPPED, default-on, bitwise)

- Rust `Vec` bases are 16B-aligned, so ~50% of our 32B AVX2 pack
  stores / sweep loads split a cache line; OpenBLAS aligns its
  panels to 64B and never splits. `with_buf` now grows through
  `align64_slice` (lends a 64B-aligned `need`-slice, ≤63 wasted
  bytes, preserves content, keeps the fallback-init contract) and
  the M-split panel uses the zero-fill-free twin
  `align64_spare_ptr` (`SONIUM_GEMM_ALIGN64=0` kill-switch).
  Covers f32+f64, all B tiles / A packs / panels (one choke
  point).
- OB-bracketed ship A/B (16T, 2 runs × 12 rounds, bitwise ndiff=0
  everywhere): sq1kT 0.94x, sq1kNT 0.88-1.00x, sq2kT 0.93-0.94x,
  sq2kNT 0.95x, sq512T 0.92-0.95x, sq768T 0.89-0.95x,
  sq768NT 0.86-0.96x, r1k512T 0.89-0.94x, m512n2k 0.84-0.86x,
  k256 0.93-0.97x, k128 0.97-0.98x, thinm64 0.89-0.94x,
  r1k512N 0.98-1.01x, sq512NT 1.00x; f64 sq1k 0.81-0.89x.
  13/14 f32 cells win in both runs (sq512NT ties; run-1
  sq512NT/r1k512N counter-readings flipped on re-run = noise).
- 509/509 lib green (new `align64_slice_properties` unit test),
  clippy pre-existing-only. Wins ADD with narrow bands (measured
  on top): sq2kT n/ob 1.18-1.21x (was 1.34-1.48x pre-both).

## 2026-09-28 — apack gate re-verdict under narrow+align (STANDS at 4 MiB)

- The N-split Trans-A shared-pack gate (pack iff A >= 4 MiB) was
  tuned before narrow bands + align64, which shifted A-traffic
  against gathering — re-measured with a new `SONIUM_GEMM_APACK=2`
  force-pack arm (kept as a crossover hook; defaults identical).
  OB-bracketed force/default, 3 runs: sq512T force loses 3/3
  (1.068x/1.083x/1.086x) — the parallel-pack cost still exceeds
  gather savings at 1 MiB; sq768T no-win (0.95x/1.01x/1.04x);
  sq1kT/sq2kT/r1k512T identical-code controls 0.98-1.01x in the
  clean run. Two of three runs had junk controls (identical-code
  0.88-1.08 under load 9-26 — OB-bracket medians break past ~8);
  the verdict run held 4/5 controls. Gate unchanged.
- Residual note: sq256T force reads 0.85x/0.92x/0.99x (2/3 wins)
  against the old 0.93x (pack loses) — weak, contradictory,
  tiny-cell evidence; not actionable, likely noise.
- Consequence: the below-gate T gap is a gather-kernel deficit,
  not routing. But the gap itself shrank under narrow+align
  (sq512T 1.48x -> 1.11x loaded) — remaining ~1.1-1.2x is now
  uniform across T/NT, not T-specific. Next: idle-box scorecard,
  then the general MT gap (kernel unroll is ~5-8% ST per the ZEN
  disasm note; MT pool/scheduling under study).

## 2026-09-28 — GPU vendor-parity scorecard (idle RTX PRO 6000)

- Device-resident ours-vs-raw-vendor, sync-per-rep medians,
  maxdiff 0e0 everywhere: sgemm 1.002x, sgemv 1.005x, dgemv
  0.992x (cudeleg); strsm 1.015-1.022x, dtrsm 1.001x, batched
  f32 1.000-1.009x, batched f64 0.908-0.927x (cubatch — native
  kernel beats strided-batched); spmv 0.998-1.005x, spmm
  1.001-1.006x nrhs=1/4/16 (new cuspab probe, 7-pt Laplacian
  64k/512k rows). cuBLAS/cuSPARSE effectiveness: AT PARITY,
  no wrapper work justified (SpMM's per-call DnMat churn +
  bufferSize measures <1% at operating scales).

## 2026-09-28 — METHOD: same-process OB interleaving is untrustworthy

- contprobe self-samples /proc/self/task: one OB call births ~64
  threads ALL-RUNNING (spinners); they decay to sleep over
  SECONDS, while rayon parks in ~ms. Interleaved (o,x,o) rounds
  therefore run EVERY ours-op under freshly-woken OB spinners and
  every OB-op against our parked pool — asymmetric contention
  inflating ours-vs-OB ~1.2-1.6x. Evidence: quiet-box mtbal
  (interleaved) reads s1kT 1.44x/s2kT 1.67x while separate-process
  solos read s1kT 0.79x (storm-free early cells); narrowab vs
  mtbal absolutes differ ~25% uniformly on identical code.
- CONSEQUENCE: all historical interleaved ABSOLUTES (sq2kT
  1.34-1.48x etc.) are upper bounds contaminated by spinner
  contention, not true gaps. Arm-vs-arm A/B (narrow, align64,
  apack) STANDS — same lib, same pool, equal contention both
  arms. TRUE SCORECARD = separate-process solos (MTBAL_SOLO)
  back-to-back in a quiet window, OB maxdiff AFTER timing (one OB
  call re-wakes spinners for seconds).

## 2026-09-28 — TRUE solo scorecard (MT-16, load ~2.5, 2 pairs agree)

- fullbal 22-cell battery, separate processes, median-of-12 ms
  (ours/ob): s1kNT 0.80-0.84x, s2kNT 0.94-0.95x, s2kT
  0.99-1.01x, r1k512N 1.04-1.05x, s512NT 1.06-1.13x, s1kT
  1.11-1.14x, s512T 1.10-1.23x, m512n2k 1.18-1.25x; d1kNT
  1.24-1.28x, d1kT 1.36x; c1kNT 2.34-2.37x, z1kNT 1.94-2.08x;
  syrk1k 1.11-1.22x, dsyrk1k 1.10-1.15x; strsm1k 2.13-2.18x,
  dtrsm1k 1.70-1.81x; dgemv8k 0.88x, ddot/saxpy/daxpy 0.19-1.03x
  (wins/ties). f32 big-square GEMM AT OR ABOVE PARITY — the
  historical 1.4-1.6x was spinner contamination, never a gap.
- Verified gap list: complex GEMM 1.9-2.4x (4M algorithm),
  TRSM 1.7-2.2x (MT solve), dgemm 1.24-1.36x, f32 pockets
  (m512n2k 1.2x, s512T 1.1-1.2x). sgemv8k pair-split 1.30x/0.59x
  (lottery cell — sub-ms ops, storms; needs padded re-runs).

## 2026-09-28 — METHOD: per-arm C buffers alias (swap twins mandatory)

- d64nab gave dr1k512 0.876-0.944x on IDENTICAL cpt 4/4 runs:
  buffer-swap twins invert EXACTLY (0.875 -> 1.144 = reciprocal),
  proving a pure C-buffer L3-aliasing effect (cn/cl relative
  offsets are ASLR-invariant, so it reproduces run after run).
  Decomposed code truth: only d768T wins (0.933x); d512NT's 0.93x
  "win" is buffer (code 1.043x); d768NT loses (code 1.055x).
- RULE: every arm-A/B with per-arm buffers needs swap twins (or a
  shared buffer) before a ship verdict; identical-cpt cells must
  read 1.000 in BOTH twin directions. f32 narrow/align64 ships
  stand (wins far outside the ±10% aliasing scale, consistent
  direction) but their small-margin cells carry ±5% buffer
  uncertainty. Scorecard solos (separate processes, ASLR-varying
  aliasing) are safe outside ±5%; close cells need MTBAL_PADBYTES
  sweep (heap-shifted re-runs) — not yet built.

## 2026-09-28 — f64 narrow bands FALSIFIED (fully reverted, zero residue)

- Ported the f32 narrow rule to f64 (`NR = 8`, 8-byte A-fit,
  `NARROW64` kill-switch): buffer-decomposed code truth over 5
  runs is one win (d768T 0.933x), one loss (d768NT 1.055x),
  ties elsewhere (d1k/d2k/dm512n2k/dk256 identical cpts; d512
  ties). No win-domain predicate exists (same L1/L2 B transition
  wins at d768T, ties at d512T) — reverted entirely (generic
  core, cpt twin, hook, audit test; 509 green after).
- Unexploited lead: d768T-like T-gather shapes gain from finer
  bands (MLP latency hiding?) while NT loses (A-traffic) — a
  future T-only band rule could take it, needs its own predicate.
- dgemm 1.24-1.36x gap REMAINS OPEN (not a banding issue).

## 2026-09-28 — TRSM triage (Upper panel-loop vs Lower fused)

- Fused single-scope path is Lower/NoTrans-only (`trsm_use_fused`
  gates uplo); Upper/Trans take the sequential panel loop (8
  panels × diag scope + k=128 trailing GEMM at 1k). Same-process
  fullbal hint (load-tinted, structural spread): strsm1kL 2.63ms
  vs strsm1kU 4.10ms vs strsm1kT 5.24ms — fused beats panel-loop
  ~1.5-2x on the same box. LEAD: port fused to Upper (+Trans);
  needs quiet solos to confirm (Lower-vs-OB unknown) + a
  descending-panel fused-band mirror. Battery extended with
  U/L/T cells (maxdiffs clean).

## 2026-09-28 — TRSM Left/Upper fused port (SHIPPED, default-on)

- `trsm_left_upper_fused` + `fused_band_upper`: descending mirror
  of the Lower path (explicit per-panel `k`, partial-first panels,
  band solve via `band_left_upper`, updates on rows above).
  Same gates/kernels/panel-setup as Lower; family switch
  `SONIUM_TRSM_FUSED=0` restores the panel loop for both.
- Fused/panel A/B (16T, OB-free alternating, shared B, Trans
  controls 0.99x): strsm1kU 0.66-0.68x, dtrsm1kU 0.71-0.72x,
  strsm512U 0.81-0.82x; Lower reconfirmed 0.56-0.71x. Solos:
  strsm1kU 2.14x -> 1.50x, dtrsm1kU 1.85x -> 1.41x (as predicted).
- 510/510 lib green (new `trsm_left_upper_fused_matches_panel_loop`
  to 1e-9: full/partial-first/tall-skinny × unit flags), clippy
  pre-existing-only. TRSM state: fused L/U 1.41-1.60x, Trans
  panel loop 2.3-2.7x.

## 2026-09-28 — TRSM strict-underfill FALSIFIED* (reverted, load caveat)

- Theory: at bands == workers the pool is saturated, so nested
  trailing-update scopes are pure overhead (`<=` contradicts the
  documented "underfill" intent). `SONIUM_TRSM_NEST_STRICT=1`
  forced serial updates there (Left only, scope routing untouched).
- Verdict: strict LOSES on f32 (strsm1kL 1.11x, strsm1kU 1.06x),
  ties f64 (0.94-1.01x); Trans controls 1.00x. Standalone update
  rates (896×64×128: MT 0.101ms vs serial 0.172ms f32) show the
  nested MT path wins per-op via better tiling (narrow bands),
  which serial lacks. Fully reverted (zero residue, 510 green).
- *CAVEAT: the A/B ran at load 18 — nested's 256 fine tasks
  rebalance around preemptions while serial's 16 coarse bands
  stall, so the verdict may be load-flavored. The decomposition
  math doesn't close either (updates ~0.66ms + diag ~0.18ms +
  overhead ~0.3ms = ~1.1ms vs 2.19ms measured — in-contention
  nested rate unknown). RE-RUN QUIET before citing; if strict
  still loses, the serial path needs narrow tiling, not routing.
- Open TRSM leads: (1) Trans fused port (2.3-2.7x, same mirror
  with transposed updates); (2) diag band-kernel rate (42%-note
  unverified); (3) serial-nest narrow tiles (helps all serial
  users if (2) confirms).

## 2026-09-28 — Complex 4M -> 3M rewrite + MT splits/epilogue (SHIPPED, default-on)

- `complex4m.rs` -> `complex3m.rs`: three real GEMMs
  (`R = Ar.Br`, `Q = Ai.Bi`, `I = S1.S2`) instead of four (-25%
  GEMM flops), fused AVX2 deinterleave+pre-add
  (`split_add_linear_avx2_f32/f64`, 8/4-wide unshuffled lanes,
  no FMA) and a fused 3-input AVX2 epilogue
  (`apply_ab3_c32/c64_row` in blas1 scalar/avx2/mod, mirrored
  lane math). Splits and epilogue go row-band-parallel via
  `scope_pinned` above 16k elems (`SONIUM_C3M_MT=0`
  kill-switch); elementwise => bitwise-identical to serial.
- Serial-3M quiet solos first (MT-16, load ~4, fullbal pair,
  maxdiff c1k 1.0e-4 / z1k 1.8e-13): c1k 1.96x, z1k 1.81x,
  c512 2.57x, z512 2.49x, c2k 1.08x — vs 4M-era 1.5-3.0x.
  Large-N closed by flops alone; small-N still serial-split
  bound (~16 ms of traffic at 2k).
- MT A/B (2 pairs, both orders, load 50-88, c3mab): c512
  0.71-0.76x, c1k 0.80-0.82x, z1k 0.71-0.73x, c2k 0.81-0.85x.
  All reps agree in sign; default-on stands.
- Decomposition (same-run, load-inflated): c/(3s) = 1.23x
  @512, 1.30x @1k, 1.14x @2k f32 — the complex-specific
  overhead (MT splits + epilogue + 3x dispatch) is nearly
  squeezed; the rest of the vs-OB gap is inherited from the
  real-GEMM gap times the 3/4 flop bonus.
- 515/515 lib green (new: split/avx2-bitwise, ab3-bitwise,
  larger-f32 ref, 2 above-gate MT-path refs; pool test takes
  the sibling epsilon floor for 3M's R-Q cancellation
  residuals), clippy pre-existing-only.
- SETTLED QUIET (load ~4, fullbal pair q2): MT-3M vs OB:
  c1k 1.50x, z1k 1.24x, c512 1.62x, z512 1.72x, c2k 0.94x
  (faster than OB, as 3M theory predicts). Gate sweep moved
  the const 16k -> 32k: sq128 MT lost 1.34-1.76x (~45 µs
  fixed per MT op vs ~4 µs serial), sq192+ wins 0.57-0.88x;
  at 32k the boundary ties (c128/c160/z128) and everything
  above wins. Bitwise MT==serial across the gate (c3mhash).
- METHOD: the first quiet pair after the neighbor died was
  contaminated (ours-side inflated up to 1.5x: s2k 17.1 vs
  11.7, c2k 56.0 vs 41.4) — the box needs ~10 min to settle
  after saturation ends. Re-ran both sides; q2 reproduced
  the historical cells (s1kNT 0.77x, s2k 0.91-0.96x).

## 2026-09-28 — TRSM Left/Trans fused port (SHIPPED, default-on)

- `fused_band_upper_trans` (ascending, Upper storage) +
  `fused_band_lower_trans` (descending, Lower storage) with
  `trsm_left_upper_trans_fused` /
  `trsm_left_lower_trans_fused` drivers: structural mirrors of
  the NoTrans twins, no new kernels — diagonal solves reuse
  `band_left_upper`/`band_left_lower` with `trans = true` on
  the trans-packed operator (`pack_abuf` folds trans/conj;
  `panel_dinv` conjugates for ConjTrans), trailing updates are
  `(trans, NoTrans)` GEMMs in the panel-loop geometry
  (verified slice-for-slice against `trsm_core`). Routing in
  both branches under the shared `SONIUM_TRSM_FUSED`
  family switch with the same gates (m >= 256, n >= 128,
  threads > 1).
- Fused/panel A/B (16T, same-lib back-to-back solos, both
  orders, load 27-50): strsm1kT 0.54-0.65x, dtrsm1kT
  0.69-0.78x, strsm1kUT 0.62-0.85x, dtrsm1kLT 0.77x;
  maxdiffs identical fused-vs-panel (7e-7 f32, 2e-15 f64).
  Matches the NoTrans fused pattern (0.55-0.81x).
- 517/517 lib green (new `trsm_left_trans_fused_matches_panel_loop`
  f64 both-uplos × Trans to 1e-9, full/partial/tall-skinny ×
  unit flags; new `trsm_left_conjtrans_fused_matches_panel_loop`
  c64 both-uplos × ConjTrans to 1e-9), clippy
  pre-existing-only. All four `SONIUM_TRSM_FUSED` test writers
  serialised by `env_test_lock`.
- SETTLED QUIET (same q2 pair): Trans vs OB — strsm1kT
  1.92x, dtrsm1kT 1.81x, strsm1kUT 1.97x, dtrsm1kLT 1.68x
  (panel-loop era 2.3-2.7x). NoTrans in the same pair:
  strsmU/L 1.78-1.83x, dtrsmU/L 1.52-1.57x. All four Left
  variants now fused; the TRSM gap is uniform 1.5-2.0x.

## 2026-09-28 — Settled quiet scorecard (MT-16, load ~4, q2 pair)

- sgemm at/above parity: s1kNT 0.77x, s2k 0.91-0.96x,
  s512NT 0.94x; pockets s1kT 1.16x, s512T 1.14x, m512n2k
  1.17x, r1k512N 1.02x. dgemm 1.24-1.37x. Complex (MT-3M):
  c2k 0.94x, z1k 1.24x, c1k 1.50x, c512 1.62x, z512 1.72x.
  SYRK 1.04-1.15x. TRSM 1.52-1.97x all Left variants.
  BLAS1/2 mixed (sdot 0.22x, saxpy 0.51x, gemv ~1.1x;
  ddot 1.98x is a sub-20 µs timing artifact).
- Remaining CPU gaps, ranked: TRSM 1.5-2.0x (uniform —
  next: diag band-kernel rate, the documented 42% note),
  complex small/medium 1.24-1.72x (mostly inherited
  real-GEMM gap × 3/4 + dispatch ×3), dgemm 1.24-1.37x,
  sgemm pockets ~1.15x, SYRK ~1.1x.

## 2026-09-28 — TRSM decomposition triage (strsm1kL 2.24 vs OB 1.54)

- TIMEBAND instrumentation (temp, reverted zero-residue):
  in-band updates 1.58 ms/band (16 bands parallel), diag
  0.30 ms/band, scope+setup residual ~0.36 ms. Update
  aggregate 677 GF/s ≈ OB's 600 — the fused structure is
  sound; the gap is diag + overhead (0.66 ms), not update
  throughput.
- Falsified: 4K-stride C (updbig@1024 ≈ tight buffers);
  nested-scope dispatch (forced-serial updates tie 2.28 vs
  2.22 ms); pool routing (nested lands inline on the 16
  mirror workers by design — `rayon::current_num_threads`
  16 < SMALL_THREADS 48, so no hop exists at this width).
- Confirmed: 16-way parallel-context penalty 2.2x
  (parserab: 16× serial 896×64×128 wall 0.47-0.52 ms vs
  0.21 single — all-core AVX freq + shared cache/NUMA).
- OB's TRSM ≈ pure update time (its full-width
  rows×1024×128 updates sum to 1.55 ms ≈ its 1.54 total —
  diag/dispatch ~0). Ours full-width: 2.71 ms at
  236-405 GF/s vs OB 443-642. So our thin-k MT GEMM trails
  0.58x AND our diag+overhead adds 0.66 ms.
- nb=256 panels FALSIFIED as a clean win (temp const flip,
  reverted): Lower wins 0.84-0.90x, Upper ties, but Trans
  REGRESSES 1.07-1.33x (fatter transposed updates hit the
  TransA GEMM pocket harder than NoTrans gains).
- Next leads, ranked: (1) thin-k MT GEMM rate 350 -> 600
  GF/s (suspect: per-task B-pack redundancy — 16 packs of
  B per update; sharing one pack across tasks would also
  lift every thin-k sgemm cell); (2) diag kernel rate
  (serial band at ~28 GF/s, latency-bound k-dot chain —
  2x = 0.15 ms); (3) residual overhead audit (0.36 ms:
  panel setup + scope + imbalance).

## 2026-09-28 — Thin-k MT GEMM triage (896x1024x128: ours 335 vs OB 641 GF/s)

- Frame: our SERIAL kernel is FASTER than OB's (2.03 vs
  3.13 ms); the gap is purely MT scaling (ours 2.9x vs OB
  8.5x on 16 workers). Per-worker: ours 21, OB 40 GF/s at
  the same ~2.7 GHz policy cap (ceiling 43) — OB at 93%,
  us at 49%. Freq ruled out as the differentiator (both
  capped; sampled workers boosting to cap under load).
- Shape takes the M-split path (rpt=16, 56 tasks, shared
  512KB B-panel, kc=128 single pass); N-split serves only
  smaller rows, 2D only narrow corners here.
- Falsified (all temp, reverted zero-residue): B-pack
  redundancy (packed once per pass); B-panel L2 thrash
  (SONIUM_GEMM_NOPACKB in-place ties 1-3%); panel
  first-touch/IF placement (worker-pack ties); pinning
  (unpinned ties); A-pack (off ties/loses); M-vs-N route
  (tie big rows, M loses small); task count (56 fine).
- N-task cold-start artifact documented: first N-split
  call's tasks incl. TLS first-touch (523 us) vs steady
  208 us — always skip warmup tasks in TASKINFO reads.
- Standing fact: per-worker rate rises with k (15.8 @32,
  23.6 @128, 28 @256, 30.4 @512, 39.5 @1024 — the last
  ≈ OB). The k-loop amortization math covers only half
  the kc=128 deficit (90% vs 99% compute share); the rest
  is unlocated. k×2 unroll + L2 prefetch already
  falsified (file notes); do NOT retry blindly.
- Next: hardware counters (L2/L3 miss, IPC — perf is
  paranoid=4-blocked; need likwid/msr or root) or asm
  inspection of `micro_f32_full6` at kc=128 (register
  pressure? L1 set conflicts on the 4KB-stride panel?).
  Untouched this turn — fresh quest.

## 2026-09-28 — Thin-k per-worker gap: round 2 (still open, 25% unlocated)

- k-loop asm (`micro_f32_full6`): CLEAN — no spills (15
  YMM fit), textbook 12-FMA + 8-mem loop at FMA-throughput
  ceiling in theory. k×2 unroll correctly NOT retried
  (throughput-bound loops gain nothing; old falsification
  stands on theory even though its protocol predates
  solos).
- Governor-ramp FALSIFIED: 12-rep trajectories flat from
  rep 1 (0.67 -> 0.61 settled, cache warmth only).
- OB beta specialness (verified both orders, results
  CORRECT to 1e-6): OB-beta1 0.40, OB-beta0 0.65,
  OB-beta2 0.56 ms. Beta=1 is uniquely fast in OB;
  theory: OB-beta≠1 pays a slow separate scale/zero pass
  (~190 us serial-ish), beta=1 skips it. Ours is beta-sane
  (beta0 0.63 vs beta1 0.68 — the C-read costs ~10%).
  Lesson: match OB at beta=1 (0.40); beta=0 parity
  (0.63 vs 0.65) is OB's quirk, not our win.
- OB-beta0 thread sweep scales monotonically (3.41 ->
  0.59, 5.8x) — not a thread-count quirk; rate gap at
  every count. The 1->4T window is where our gap opens
  (ours 1.15x, OB 3.2x); 4->16T scaling is IDENTICAL
  (2.6x both).
- M-split rpt sweep (temp `SONIUM_GEMM_RPT`, reverted):
  rpt 16 -> 28 -> 56 -> 112 monotone improves 896-cell
  0.70 -> 0.63 ms (~9%: fewer waves + less edge). Edge
  kernel bounded mild (~2x, not catastrophic). rpt=112
  (8 tasks!) still 1.58x off OB — NOT the wall.
- Forced-2D re-confirms the file note (ties/loses at
  k=128). M/N/2D all tie ~0.65-0.72 — decomposition
  fully exonerated.
- Exhausted statically: TLB (covered), store-forwarding
  (none), 4K aliasing on C (marginal, fits), L1I/LSD
  (loop fits), branches, denormals, NUMA (single node),
  THP (symmetric). C-epilogue L1-miss ≈ 10% (measured
  via beta0/1). B-fetch exonerated 3 ways (nopack tie,
  2D tie, both-B-footprints-equal tie).
- BLOCKED on observability: no perf (paranoid=4, not
  root, write denied), no likwid/uprof/msr, no valgrind.
  ~25% of the gap (0.63 vs 0.40 ms) has no remaining
  static theory. Unblock routes: (a) root for
  perf_event_paranoid (ask user); (b) `cargo install`
  uiCA for static uop simulation; (c) differential
  micro-benchmarks of isolated sweep fragments (B-slice
  residency, C-stride) — the one route needing no
  privileges. Probe arms kept in updprobe
  (REPTRAJ/OBDIFF/BETA/BETA2/dupd/OB_THREADS) for reuse.
## 2026-09-28 — f64 N-split wide-Kc (SHIPPED, default-on)

- Root cause of the d1k half of the dgemm gap: at 1024^3 both
  precisions take N-split, but f32 uses Kc=512 below (2 passes)
  while f64 hardcoded 256 (4 passes — 2x whole-A restream per
  task + 2x C-band revisits). Fresh quiet solos first shrank the
  stale scorecard (d512 0.96x win, d2k 0.96x win; only d1kNT
  1.29x / d1kT 1.38x real) — f32 s1k wins 0.87x the same shape,
  isolating the f64 pass count.
- Trial `SONIUM_GEMM_DKC=512` (uncapped k >= 768): d1kNT +12%,
  d1kT +10%, d768NT +5%, 256x1024x1024 +10%, 512x2048x1024
  +10%, 512x512x2048 +12% — all reproduced both orders, OFF
  brackets stable. Losses map the exclusion domain:
  1024x1024x2048 1.03-1.07x, 2048^3 1.06-1.24x (512 KB B tile
  fills the 512 KB L2, thrashes past the L3-slice fit),
  1024x256x1024 1.02-1.08x (narrow-n: few bands, spill wins).
- Shipped gate (conservative, win-domain-only): 512 at
  768 <= k <= 1152 and n >= 512, else 256; kill-switch
  `SONIUM_GEMM_DKC=256`. Pass boundaries change fp grouping
  (maxdiff vs OpenBLAS ~6e-15, within tolerance; new test
  `avx2_dkc512_matches_scalar` asserts scalar match 1e-9).
- Ship verdict vs OB (MT-16 solos): d1kNT 1.29x -> 1.15x, d1kT
  1.38x -> 1.30x; d2k/d512/thin-k gate-off shapes tie
  pre-ship (no collateral). 518 lib tests green, clippy clean.
- The ship pair exposed d768NT at 1.41x vs OB — ours-specific
  (321 GF/s per-flop vs 423 at d1k; OB flat 455-486): next
  triage is the 768-vs-1024 band/path sizing cliff.
## 2026-09-28 — N-split wave-snapping (SHIPPED, default-on)

- d768NT triage (1.41x vs OB, ours-specific): serial rate matches
  d1k per-flop (53.7 vs 52.3 GF/s) but MT scales 4.9x vs 7.4x.
  Balanced kc-pass splitting (768 = 384+384 vs 512+256) was
  trialled and FALSIFIED (s768 exact tie with the trial provably
  active via maxdiff change; d768 run-1 win irreproduced swapped;
  fully reverted, zero residue).
- Thread-count sweep isolated it: d768 stalls T12->T16 (+3%)
  while d768n1k gains +26% and d1k +22%; T24 jumps 1.53x to one
  wave. `l3_fit_cpt` targets 30 bands -> NR-rounding gives cpt 32
  -> 24 bands = 1.5 waves on 16 workers (T12: 2 full waves,
  hence the stall-then-jump quantization).
- Fix: snap the l3_fit band target down to full pool waves
  (30 -> 16 bands, one wave), geometry-clamped so narrow-n
  (bands < workers) keeps the legacy count — the unclamped snap
  broke the cpt 16-element floor at 2048x128x2048 (caught by
  `gemm_l3_fit_cpt_gates_and_invariants`). Kill-switch
  `SONIUM_GEMM_WAVE=0`. Bitwise-neutral by construction (each
  element keeps its task and pass structure); locked by
  `avx2_wave_snap_bitwise_neutral` (768^3 on a 16-pool, both
  arms, assert_ne on counts + assert_eq on outputs).
- A/B (MT-16, OFF/ON/OFF2 + swapped ON/OFF): d768 0.80-0.85x
  both orders (2.79 -> 2.23ms, ~1.12x vs OB); every other
  battery shape ties (s2k/d2k ON-inflations in run 1 reproduced
  as ties on the swapped repeat — box hiccups on provable
  no-op shapes). 519 lib tests green, clippy clean.
## 2026-09-28 — M-split 2K panel-stride padding (SHIPPED, no switch)

- d1kn256 triage (1.29x, M-split by the strict `4n > m` fence):
  MT scales 3.1x (1.5x already on 4 threads); per-pass fit over
  k=512/1024/1536 is perfectly linear (1.09ms/pass, zero fixed
  cost) — a per-task kernel-rate wall at ~14 GF/s vs 54 serial,
  not dispatch. Temp `SONIUM_GEMM_RPT` sweep (reverted): rpt
  16/32/64 flat — band count (64->16) irrelevant, panel-reread
  volume theory dead.
- Pitch isolation via forced-M: n=256 (2KB pitch) 2.77ms vs
  n=252 2.19ms / n=260 2.13ms with MORE flops — 30% resonance.
  `panel_stride` only padded 4K pitches; a 2KB pitch cycles 2 L1
  sets (32-line pitch over 64 sets) and stalls the sweep. Fix:
  pad at 2K multiples (272x8 = 2176 B = 34-line pitch cycles 32
  sets). n=256 now flat with neighbors (1.97/1.97/1.96ms).
- Layout-only, bitwise-neutral (same class as the 4K pad — no
  kill-switch): f32+f64 M-split share the helper. 519 green,
  clippy clean. Remaining M deficit is the documented structural
  one (shared-panel L3 re-reads, ~1.5x behind N unpinned).
- Side finding: N-split just over the fence (d1kn260) makes 11
  actual bands (16 target -> cpt 24) and runs 3.43ms — worse
  than M. The fence placement is correct; n=260 can never fill
  16-waves evenly at NR granularity (260 = 16x16+4 straggler).
## 2026-09-28 — Pad ship verdict + narrow-n scorecard (MT-16 solos)

- d1kn256 1.29x -> 1.02x vs OB (2.04 vs 2.00ms). d1kNT 1.07x,
  d768NT 1.18x, d2kNT 0.99x, dthk896 0.98x; f32 all wins/ties
  (s1k/s2k/s768 0.88-0.93x). d1k256k/d1k2563 (1/3-pass M
  controls) 1.07-1.08x. 519 green.
- New exposures: d768T 1.64x (gather path, n=768 below the pack
  gate — needs a CLEAN pack-vs-gather retest; the load-7 one was
  trash, and the falsified narrow trial's only win was d768T
  0.933x via finer bands); d1kn252 1.30x (masked nb=4 nc-tails,
  252 = 128+124 — tails, parked); sthk896/dthk512 ours-cells
  contaminated this pair (0.62/0.77 vs 0.51/0.60 norms, load
  crept to 4.4 on the ours arm — not regressions).
- Standing dgemm residuals: d1kn252-tails 1.30x (parked),
  d1152 1.27x, d512n2k 1.22x, d1kT 1.26x (pack cost, structural),
  d768T 1.64x (next: clean pack/gather retest).
## 2026-09-28 — MT pack-gate re-tune (SHIPPED, default-on)

- d768T 1.64x exposed the stale `n >= 1024` crossover (tuned at
  kc=256, predates wide-Kc). Temp `PACK_A_MT=2` force bypass
  (reverted): force-pack wins d1kn512T 0.79x (3.14 vs 3.97ms),
  d1kn768T 0.72x (4.50 vs 6.29ms), d768T 0.82-0.86x (2.59-2.75
  vs 3.03-3.34ms) — all reproduced swapped; d512T ties 1.00x;
  d1kT/d2kT tie (already pack — no-op controls).
- New gate: `n >= 512` + `A >= 4 MiB` floor (512^3T tie
  preserved; only (1024,512,1024) flips in the gate test).
  Verdict vs OB: d768T 1.64x -> 1.33x, d1kn768T ~1.87x -> 1.35x,
  d1kn512T ~1.35x -> 1.16x; d1kT/d2kT/d512T unchanged (pack
  cost + NT gap now, structural). 519 green, clippy clean.
- Process note: the flip broke two engagement asserts in tests
  that lock-then-assert, poisoning the crate-wide env lock and
  cascading into 12 unrelated env-test failures (different set
  each run — the tell). Fixed at the root: engagement shapes
  moved to 768^3 (keeps every contract: f32-gather + f64-pack)
  and asserts moved BEFORE the lock. If env-test failures ever
  vary run to run again, suspect lock poisoning first.
## 2026-09-28 — Band-fused c32 3M (SHIPPED, default-on)

- Complex decomposition (cxdec, warm): 3x real GEMMs run FASTER
  in-a-row than solo (caches); the 24-34% overhead (splits +
  epilogue + dispatch) explains ~all of c1k/z512/z1k gaps — but
  zero overhead still trails at c512 (1.18x: OB's native complex
  kernel beats 3x real GEMMs on FMA-ILP).
- Viability (cxfuse): 16x serial thin-band GEMMs hit 84 GF/s per
  band vs 31 GF/s MT-share (L2-resident bands vs L3-streaming
  N-tasks). Fused driver: B split once (shared L3), one MT scope
  over row bands, per band (serial) A-split + 3 serial GEMMs +
  epilogue rows; uninit band bufs (no 6 MB/call memset, +8%).
- Window (same-process interleaved, 20 reps, reproduced): c128
  0.84x, c256 0.92x, c512 0.87x, c1k 0.92x, c2k1k 0.89x, c1kn56
  0.53x, c1kn12 0.83x. Losses: c1536/c2k/c1k2k 1.01-1.09x (B
  over 1 MB), c56n1k 1.24x / c512n1 1.10x (bands under 64 rows
  with B over 256 KB). Ship gate: dims box (64..2048 m,
  64..1024 n/k) + B <= 1 MB + (bands >= 64 rows at pool width
  or B <= 256 KB); kill-switch `SONIUM_C3M_FUSE=0`. Bitwise vs
  phased where kc matches (k < 768), tolerance above (f32
  wide-Kc); locked by `gemm_c32_fused_matches_phased` (512^3
  bitwise non-tight + 1k tolerance + gate asserts on a 16-pool).
- No c64 twin (reverted zero-residue): z-window non-monotonic
  (z128 0.38x dispatch win, z256 1.07x loss to 2D inners,
  z512/z1536 wins, z1k/z2k ties) — needs an inner-path-aware
  predicate, parked with the analysis in the gate doc.
- Verdict vs OB (spinner-free ours-arm, see below): c1k
  1.51x -> 1.32x, c512 1.70x -> 1.41x, c2k 0.98x holds. 520
  green, clippy clean (one targeted uninit_vec allow, SAFETY
  reviewed, bitwise-locked).

## 2026-09-28 — Harness lesson: OB spinner contention (all solos)

- fullbal/dsolo ours-arms call OB once (correctness) with 16 OB
  threads: the spinners never sleep and contend FMA with our
  workers on later cells — ours inflated up to 1.35x (z1k 18.2
  vs 14.1 spinner-free; c1k 8.7 vs 7.9; c512 1.67 vs 1.48).
  Same-process env-flip A/Bs are unaffected (both arms equally
  exposed — all ship ratios stand); absolute vs-OB verdicts
  were pessimistic. FIX: `OPENBLAS_NUM_THREADS=1` on every
  ours-arm (OB-once stays correct). s/d re-baseline queued.
- True complex scorecard (fused, spinner-free): c512 1.41x, c1k
  1.32x, z512 1.70x, z1k 1.14x, c2k 0.98x. Remaining complex
  gap needs a native complex micro-kernel (FMA-ILP) — parked
  as a multi-day project behind TRSM.

## 2026-09-28 — cpt floor-round SHIPPED (d1kn260 1.58x -> 0.95x)

- s/d spinner-free re-baseline (freq-capped box ~2.2 GHz, so
  ratios only): most cells consistent with priors; two finds:
  (1) ours-side n=260 cliff is REAL and internal (ours 3.34 vs
  2.15 at n=256, +55%); (2) d1kn252 1.44x is OB-side (ours
  flat 2.18 vs 2.15 — OB dips 1.51 vs 2.11 at n=256; nothing
  to fix on our side, tails note stays parked).
- Root cause: `l3_fit_cpt` ceiling-rounds cpt to whole
  micro-panels AFTER the wave-snap: n=260 plans 16 bands but
  cpt 17 -> 24 collapses to 11 bands (5 workers idle the whole
  wave). The rounding defeats the snap whenever n/bands is not
  NR-aligned. Fix: floor-round (cpt 16, 16 full bands + one
  4-col tail in a near-empty second wave, ~2% cost).
- Trial (16T solos): d1kn260 3.34 -> 2.00 (0.60x);
  `SONIUM_GEMM_CPTFLR=0` restores 3.22 (causality proven,
  drift ~2-4% << 40% effect, no swap-twin needed).
  d1kn252/256, d768, d1k, s512/s1k controls flat.
  vs OB 2.11: d1kn260 now 0.95x (win).
- Default-on with `SONIUM_GEMM_CPTFLR=0` kill-switch (same
  convention as WAVE); bitwise-neutral by the same argument
  (only the column partition moves), locked by
  `avx2_cpt_floor_bitwise_neutral` (1024x260x1024, portable
  cpt 16/24 asserts). `gemm_l3_fit_cpt_gates_and_invariants`
  passes unmodified. 521 green.

## 2026-09-28 — TRSM nb=64 for Upper/Trans FALSIFIED (reverted zero-residue)

- `trsm_panel_nb` already ships per-shape nb (f32-Lower-NT 64,
  f64-Lower-64 loses 1.02-1.11x, Upper/Trans 128 unmeasured).
  Temp-knob trial (Upper-or-Trans -> 64, Lower excluded),
  fullbal 1k solos both orders, L rows as knob-off controls:
  strsm1kU 1.01/0.98, dtrsm1kU 0.97/0.98, strsm1kT 0.97/1.00,
  dtrsm1kT 0.99/1.00, strsm1kUT 0.99/1.02, dtrsm1kLT
  1.02/1.01 (nb64/nb128, controls at 0.99-1.01).
- Verdict: blanket Upper/Trans-64 is a tie with an LT
  lean-loss — reverted, grep-clean. PARKED: lone dtrsm1kU
  0.97-0.98x both-orders lean (f64-Upper-NT-64 per-shape
  candidate, needs a third run + 512/2k sweep + mechanism;
  low leverage vs the 50-100% TRSM-vs-OB gap).
- Also learned: trsmfusedab U/N rows are unit/non-unit diag
  (all Lower-NoTrans) — knob-off controls, not Upper/Trans
  coverage; fullbal owns the Upper/Trans rows.
- Next TRSM lead: shared B-pack across bands (the 0.36 ms
  scope+setup residual is plausibly 16x redundant B-packs;
  premise check via temp pack-vs-sweep timing before any
  restructure), then diag kernel rate (14 GF/s triangular).

## 2026-09-28 — CORRECTION: NB64 trial was VOID + TIMEBAND-2/3 verdicts

- CORRECTION to the nb=64 entry above: the `SONIUM_TRSM_NB64`
  knob sat in `trsm_panel_nb`, which serves ONLY the panel
  (non-fused) path — the fused path hardcodes `TRSM_NB` (all
  fullbal 1k rows take fused). All NB64 arms ran identical
  fused-128 code; the +/-2% was pure noise. The entry's
  "Upper/Trans-64 falsified" verdict is RETRACTED (fused-nb64
  was already trialled+falsified 2026-09-27 per the code NOTE:
  1.09x MT f32 loss via band x panel dispatch
  multiplication). The knob is reverted; residue-check clean.
- TIMEBAND-2 (temp pack-vs-sweep in `sgemm_serial`, strsm1kL):
  n=64 band-update calls pack_frac = 0.019 — B-packs are 2%
  of update time. The shared-B-pack restructure premise is
  FALSIFIED (would save ~2% of updates, not the residual).
- TIMEBAND-3 (temp per-band diag/upd, focused trsm1k probe):
  wall 2.64 ms, bands diag 0.23-0.36 / upd 1.6-2.1 ms.
  (Residual-vs-max-band polluted by contended eprintln; no
  clean residual number claimed.)
- NEW TOP TRSM LEAD (from reading `band_left_lower_kernel_*`):
  the row-sequential k-dot diagonal runs ~7-14 GF/s (1-3% of
  peak; triangular row dependency). The code NOTE already
  suspects "row-wise k-dots vs a blocked/inverted-diagonal
  micro-solve running at GEMM rate; untested". Design:
  blocked substitution — per 8-row block, GEMM-subtract the
  k0 prefix (`sweep_f32` direct, X-prefix packed to a 4KB
  stack tile, NO `gemm_serial` dispatch) + tiny serial 8x8
  triangular solve (reuse row kernel with k-loop from k0).
  Same flops at ~150-400 GF/s instead of ~10: diag
  0.30 -> ~0.03 ms (~10% off strsm1kL). Stability class
  unchanged (reordered accumulation, tolerance-equal per the
  fused-vs-panel precedent) — no explicit inversion.

## Merge 2026-09-28: compute-batch LU regression, root-caused and fixed

Merging main (compute BLAS/LAPACK batch) regressed LU: 1024 in-place
8.94→12.47 (+39%), 2048 default 36.04→40.25 (+12%), nb ordering
flipped (nb32 best at 1024). Vendor flat in-window → real, not clock.
Phase profile (temp, since removed) + old-vs-new microbench pair
(schur_sub_strided shapes won 6-8% but LU-Schur lost) narrowed it to
`schur_sub_serial` (lookahead entry): narrow-n (512,32,32) 28.9→40.2µs
(+39%, both entries, identical checksums) while wide-n won.
Root cause: the new `st_apack` serial gate evaluates `l3_bytes()` —
an UNCACHED sysfs read — plus `env::var` on EVERY serial GEMM call,
for shapes (thin-k) that never qualify. `ST_APACK=0` recovered fully
(28.8µs). Fix (compute, this branch): `l3_bytes()` OnceLock-cached
(hardware-invariant) + pure-dimension pre-check (`m>Mc && n>Nc &&
k>Kc`) before the footprint math at both serial gates (f32/f64).
`st_apack_on()` env read kept per-call (tests mutate it; ~0.1µs).
Post-fix: serial narrow fully recovered (28.3/93.5µs, at/above old);
LU-1024 in-place 9.02/7.48 = 1.206x (steady ~8.7/7.4 = 1.176x),
LU-2048 default 36.41/25.02 (all within noise of pre-merge).
nb re-adjudicated on merged kernels: keep nb48-f64/32-other (1024:
48≈32 > erratic 64; 2048: 64≈48 > 32, pairs 8/9 + 7/9). Main's nb64
6/6 does not reproduce. Note for compute agent: same uncached-l3
pattern — audit other per-call sysfs/env reads in hot paths.

## D3 resumed 2026-09-28: post-merge gate re-measure + fail-cell triage

Merged+fixed tree (3f26867f), quiet windows, same discipline:
- dgetrf-1024 in-place 9.02/7.48 = 1.206x (strict FAIL by 0.5%;
  steady-state reps ~8.7/7.4 = 1.176x). Warmup decay characterized:
  per-proc (3/3 procs repeat), threading-specific (ST flat; nofunnel
  flat-slow 13.8; noaffin flat 10.2), survives ST-vendor (vendor
  interference falsified); THP (madvise, no madvise calls) and
  frequency (box-warm procs repeat it) falsified. Mechanism still
  open (compact-pool settling?); parked after 8 runs / 5 falsifications.
- eig-200 4.21/3.59 = 1.173x PASS (thin). eig-300 9.50/7.49 = 1.268x
  FAIL (-0.51ms). Per-solve map: panel 2.94 (dots at FMA floor
  0.27cyc/elem — no lever) + qaccum 1.60 + sec 1.55 (div-floor) +
  merge 0.82 + asm 0.78 + trailing 0.61 + back 0.43. Qaccum exact
  shapes old-vs-new: tn TIED (+-1.5%), nnsub new wins 2.5% —
  routing pocket falsified. No single piece covers 0.51ms.
- Sparse (residuals ~1e-15, fill unchanged): simp-32 1.619x,
  simp-40 1.832x, sup-80 2.387x, sup-128 1.922x (112.3ms).
  Sup-128 update census (temp, removed): 25,429 GEMM calls = 23ms
  of 112ms (21%) at 11.6 GFLOPS, 911ns/call. Unchecked-GEMM
  (~100ns/call target) worth ~14ms = 12% — insufficient alone
  (need -38%); 79% of sup time lies elsewhere (scatter/TRSM/
  analyze/merge). Wide-supernode redesign remains the path.
- dgetrf-2048 1.55x FAIL (panel/Schur regime, needs -35%).
Net: 2 borderline (1024 strict-fail by 0.5%, 200 pass), 6 fail.
Every remaining gap re-verified project-scale on the merged tree
even with compute editable: no in-session flip path found.

## Sup-128 phase census 2026-09-28 (temp timers, since removed)

grid128 sup oneshot 112ms = analyze ~11 + numeric ~93-116. Numeric:
in-panel ~55ms (dense panel LU + panel TRSM + reach; HALF the factor)
+ update-GEMMs 23ms (25.4K calls, 11.6 GFLOPS) + scatter ~10.5ms +
merge 4-10ms (varies run-to-run). Flipping (need -42ms) requires
in-panel -51% even after a perfect update path — implausible via
tuning (SuperLU's total is 58ms; our in-panel alone matches it).
Structural panel/supernode redesign is the only path. Confirmed:
sup FAIL cells are project-scale, not tunable.

## 2026-09-27 — dgetrf-1024 warmup-decay hunt CLOSED by elimination (10 falsifications)
- DVFS-shape FALSIFIED: clocks do ramp 2.2->4.1GHz during a proc, but a chained
  proc2 (hot clocks, no idle gap) still decays 10.86->8.06ms. DVFS ~= 0.3ms of
  med, not the decay shape. THP-matrix FALSIFIED: harness reads show BOTH arms
  factorize fresh per-rep clones outside timers (luabv main.rs:125-140) - shared
  buffer effects excluded by vendor-flat. AutoNUMA FALSIFIED: single node,
  numa_balancing=0.
- Vendor-flat exclusion proof: every shared-harness effect (clone, allocator,
  heap, main-thread migration) is falsified because the vendor arm shares it
  and is flat after rep0. The ~1ms rep1->7 decay is PROVEN in library-side
  reused state (ws_inplace/pool), yet pool-settling, l3/OnceLock, env-lock,
  validate-scans, freq, THP, NUMA are all falsified there too.
- Remaining step is per-rep factor/scan/pool-split instrumentation of library
  internals (real project, needs go). Hunt closed; cell stays strict-FAIL 1.206x.

## 2026-09-28 - dgetrf-2048 post-merge census (temp SONIUM_LU_PROFILE, reverted 0 remnants)
- Quiet window (load ~1.3), RAYON=16/OB-8T, one-size-per-process. Meds:
  default 37.91 / nb48 40.56 / nb32 42.79 / nb64 41.85 / inplace 41.55 /
  vendor 26.14 -> default 1.45x, nb48 1.55x. D3 FAIL stands on both ends.
  Artifact #5 reproduced (default 0.93x nb48, identical geometry).
- Wall split (~36-wall nb48-geometry samples, 41-43 panels): TRSM 2.5 +
  lookahead-serial 7.6 + scope 24.1 (caller remainder-Schur 22.5, panel
  worker ~20.5 hidden, exposed ~1-2) + deferred 0.0 + validate 1.2.
- Merge deltas vs pre-merge post-lookahead (2.6/6.8/24.2/1.1): Schur -2ms,
  lkup +1ms (6.8->7.6, overlaps the 6.5-11 sample range - suggestive, not
  proven), TRSM/val flat. Net default wall ~40->~37 quiet.
- Gap anatomy (~11-14ms over vendor 26.1): lkup-serial ~7.6 (streaming
  floor, no vendor counterpart, structural) + Schur rate ~5 over
  vendor-implied (the known 227-vs-300GF kernel gap) + TRSM ~1.3 (CLOSED,
  substrate floor) + validate 1.2 (required) + exposed panel ~1-2.
- NO NEW POCKET. Named lever remains the untried strided-pack prefetch
  in the PACK loops (compute-side project). Cell fully mapped post-merge.

## 2026-09-28 - simp within-factor census (temp SONIUM_SPARSE_PROFILE, reverted 0 remnants)
- Quiet (load ~1.2), RAYON=16, parnum_ab_superlu. Grid32: simp_f 3.38 /
  superlu 2.02 = 1.708x; within-factor ord 0.57 (17%) + bld 0.07 (2%) +
  sym 0.12 (3.5%) + num 2.41 (73%) + asm 0.13 (4%). Grid40: simp_f 5.64 /
  superlu 3.06 = 1.881x; ord 0.82 (15%) + bld 0.10 + sym 0.18 + num 4.21
  (76%) + asm 0.21. Residuals 1e-15 PASS; our fill 8% LESS than SuperLU
  (38988 vs 42470 @32) yet slower: pure rate gap, not fill.
- Numeric alone (2.41/4.21) exceeds SuperLU's TOTAL (2.02/3.06): the gap
  is simplicial-scalar update vs SuperLU's supernodal BLAS-3 numeric -
  a structural algorithm-class gap, consistent with the update-floor
  verdict and the mark-split falsification. Ordering is the same COLAMD
  both sides. NO POCKET; the fix is a fast supernodal (sup redesign),
  which is itself walled (in-panel ~55ms). Both gate simp cells mapped.

## 2026-09-28 - D3 re-bench on 7aea9974 (post solid-physics FF, quiet load 1-3)
RAYON=16, OB-8T, one-size-per-process, rep-trajectory, residual gates PASS.
Main delta since 92c882d4 is FEM regroup (1 solvers file: equilibration.rs,
off the bench path); no compute kernels. Harnesses rebuilt on the FF tree.
| Cell | Sonium | Vendor | Ratio | D3 | vs prior |
|---|---|---|---|---|---|
| dgetrf-1024 inplace | 9.02 | 7.49 | 1.204x | FAIL | 1.206x, flat |
| dgetrf-2048 default/nb48 | 38.04/39.42 | 26.54 | 1.434x/1.485x | FAIL | 1.45x/1.55x band |
| eig-200 | 3.83 | 3.51 | 1.091x | PASS | 1.173x, better window |
| eig-300 | 9.45 | 7.25 | 1.303x | FAIL | 1.268x; vendor re-grounded 7.5->7.25 (4/4 stable), ours flat |
| simp-32 total | 3.33 | 2.04 | 1.627x | FAIL | 1.619x, flat |
| simp-40 total | 5.41 | 3.07 | 1.764x | FAIL | 1.832x |
| sup-80 total | 37.35 | 15.64 | 2.388x | n/a | 2.387x, identical |
| sup-128 total | 113.38 | 64.11 | 1.768x | n/a | 1.922x, better window |
eig-300 stack target: ours 9.45 -> <=8.70ms (-0.75ms) vs quiet vendor 7.25.

## 2026-09-28 - eig-300 STACK LANDS: 9.45 -> 8.19ms, 1.125x D3 PASS
Slices (all in sonium-solvers-eigen, WT only):
1. Fused dsymv: new `x86::rowdots4` AVX2 kernel (4 rows, shared L1 vector,
   next-group prefetch, strided-direct stores) + test; panel row-dots call
   it with scalar tails. Oracle (dsymvab probe): OB dsymv 0.655 /
   OB dgemv 0.914 / compute-gemv 1.081 / rep-dots 1.225ms.
2. Corr/larft: wrow/wrow2/larft dots -> rowdots4; DAXPY+W-snapshot fused
   (one strided pass).
3. Merge-scratch pre-size: `bufs.qt/w.reserve(n*n)` at recursion top
   (right spine never realloc-copies stale; left arm sizes on entry).
   Over-delivered: combined build with (4) dropped 8.94 -> 8.19.
4. Refgen scale+zero fused (one pass).
5. Node W-scatter row-outer -> root-outer (bitwise-neutral; single-sample
   profile ambiguous, kept: same values, cheaper pattern).
FALSIFIED: triangle-only dsymv via mirror column part REGRESSED +0.75
(9.14 -> 9.89, unprefetched strided stream latency-bound); reverted, note
in code. OB dsymv proves 1.4x needs a prefetch-carrying kernel (backup).
A/B (quiet-ish load ~3, RAYON=16): ours 8.15/8.16/8.20/8.20/8.47 (one
high outlier) vs same-window vendor 7.25/7.32 -> ~1.125x PASS with
margin. eig-200: ours 3.37 vs vendor 3.74 -> 0.90x PASS.
Gates: 16 eigen suites green (incl. new rowdots4 test); clippy 0 new
(adjoint.rs type_complexity pre-existing); orth/relres digits identical
to baseline; R1==R16 gate digits (determinism holds).

## OpenCourant explicit comparison (2026-10-11)

Sonium native explicit vs the OpenCourant engine on the frozen
4-deck subset (`taylor_bar`, `rail_crush`, `necking_bar`,
`hertz_sphere`), measured with `qa-oc-bench` (release binary,
Threadripper PRO 3995WX). Full report: `perf/oc-explicit.md`;
raw evidence: `perf/oc/*.json` (33 cells, all
`correctness_passed=true`).

Per-dof-per-cycle vs engine serial: taylor sonium 317 ns (6.4x
faster; engine setup-dominated at 24 cycles), rail 228 ns
(1.1x), necking 220 ns (1.1x), hertz engine 123 ns wins (2.6x
over sonium 320 ns — dt policy + contact cost, probes pass).
Nobody scales on these tiny decks: sonium t1 ~= t16
(1.00-1.04x), engine -nt16 is 1.8-3.4x slower than serial, MPI
only pays on hertz (r4 = 0.73x serial wall). Memory: sonium
3-15 MB vs engine 38-108 MB peak RSS. No CUDA explicit arm
exists (N/A). Shells/plasticity/locking decks deferred (no
native explicit shells/plasticity). CI floor gate:
`.gitea/workflows/oc-perf-floor.yml` (10x bar on taylor
`--quick`).
