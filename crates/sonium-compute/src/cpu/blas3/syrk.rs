//! Blocked symmetric rank-k update (`SYRK`).
//!
//! Computes `C := alpha * op(A) * op(A)ᵀ + beta * C` for row-major `A`/`C`,
//! writing only the triangle named by `uplo` (`op` is `NoTrans` with `A`
//! `n × k`, or `Trans` with `A` `k × n`). This is the LAPACK `*SYRK` entry
//! our blocked `Cholesky` trailing update is shaped like, and the BLAS
//! coverage vendor libraries ship that `sonium-compute` lacked.
//!
//! Algorithm: partition `C` into `NB = 64` band-rows. Each band-row is an
//! independent task (rayon scope): off-diagonal blocks are full rectangles
//! updated by one threaded [`crate::cpu::blas3`] `gemm` each, diagonal
//! blocks by a small unblocked rank-k kernel that writes only the
//! requested triangle. `C` is read exactly once per band plus the beta
//! pre-scale; `A` row bands stream once per band-row in `NoTrans` mode.
//! In `Trans` mode each task packs its `k × nb` column panel contiguously
//! first (strided column reads would stall exactly like the GEMV-`Trans`
//! dots did), then runs the identical kernels on the packed panel — with
//! conjugation folded into the pack for complex `ConjTrans` (Hermitian
//! update), which otherwise reduces to the plain `Trans` path.
//!
//! Only rounding differs from a fully unblocked update; no zero-skips, so
//! NaN propagates per IEEE. Padding outside the triangle is never touched.

use crate::backend::Transpose;
use crate::cpu::blas3::trsm::TrsmScalar;
use crate::cpu::blas3::{
    gemm_c32, gemm_c32_serial, gemm_c64, gemm_f32, gemm_f32_serial, gemm_f64, gemm_f64_serial,
    trsm::TrsmUplo,
};
use num_complex::{Complex32, Complex64};

/// Band width of the blocked update. 32 stands: finer bands balance the
/// triangular skew better without nested scopes — spotrf-1024 wins
/// 3.35→3.12ms, dpotrf-1024 ties, potrf-2048 ties (43.7 vs 44.6ms), and
/// the switch is bitwise identical (disjoint rows, same k order), so no
/// numerics risk. 128 halves the per-band dispatches and lifts mid-size
/// rates in isolation (T=1024: 96→109 GF/s), but end-to-end potrf-2048
/// regresses 26.5→29.1ms — the 128-wide diagonal's 2x-flop full GEMM
/// waste plus small-trailing collapse (T=256: 34→18 GF/s) outweigh it.
///
/// NOTE (reverted trial): k-aware wide bands (128, `k <= 128 && n >=
/// 512`) with interleaved band pairing lost back-to-back A/B
/// (n=992,k=32: 0.45ms wide vs 0.30ms base, 2/2 rounds). Paired tasks
/// underfill the workers (half the aggregate bandwidth) and the
/// task-count underfill test then nests bands aggressively (`/16`
/// threshold: scope overhead on sub-1M-MAC bands). Any future
/// coarsening must keep task count >= worker count, or pair without
/// tripping aggressive nesting.
/// 64-wide bands were trialled 2026-09-26 for the small-`k` Cholesky
/// shapes (halving per-band GEMM dispatches) and lost standalone at
/// every size (n=976,k=48 f32: 0.19 -> 0.26ms) plus in-situ spotrf-1024
/// (2.39 -> 2.56ms med, 3/3 rounds) — the 32-wide L1-resident tiles beat
/// coarser bands despite the dispatch count.
const SYRK_NB: usize = 32;

/// Single-threaded band width. ST SYRK is pack/dispatch-bound, not
/// balance-bound: every band re-packs its `A_top` span, so total pack
/// traffic scales as `n²k/2nb` (66MB at nb=32 vs 8MB at nb=256 for
/// n=2048,k=256) plus `n/nb` A re-reads and dispatches. Wide bands cut
/// all three ~8x while the sweep keeps L1-resident micro-panels
/// (unlike the hoisted-panel trial, consumption stays sequential).
/// MT keeps 32 (balance + nesting; coarser lost 37% on thin-k MT).
const SYRK_NB_SERIAL: usize = 256;

type GemmFn<T> =
    fn(Transpose, Transpose, usize, usize, usize, T, &[T], usize, &[T], usize, T, &mut [T], usize);

// NOTE (reverted trial): hoisting the per-band B transpose-packs into
// one shared stride-`n` panel pack per call (bands sweeping their column
// ranges from it, no per-band packs) loses back-to-back, standalone and
// in-Cholesky (f64 n=960,k=64: 0.39 -> 0.55ms standalone, spotrf-1024
// SYRK phase 1.25 -> 1.90ms). The shared 0.5MB panel falls out of L1, so
// every micro call's B reads come from L2/L3 and stall the FMA chains,
// while the per-band 8KB tiles stay L1-resident — the 30x pack-count
// redundancy is absorbed by L3 and is cheaper than the slower sweep.
// Do not re-hoist without changing the consumption cache level.

/// Pointer wrappers for `rayon::scope` dispatch: disjoint `C` band-rows,
/// read-only `A`.
#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// NOTE (reverted trial 2026-09-26): once-per-call pre-transpose of `A`
/// (`n × k` → dense `k × n`, vectorised `pack_trans_f32`/`pack_trans_f64`,
/// bands consuming `(NoTrans, NoTrans)` row-copy tiles) wins serially
/// (+7-18% at RAYON=1: 9x fewer pack elements) but loses in parallel —
/// 4/4 in-situ spotrf-1024 rounds (2.3 → 2.55ms) and RAYON=8 probes.
/// The shared transpose's lines sit `M`-state in the transposing core, so
/// every band worker's prefix re-reads pay cross-core transfers, and the
/// per-band tile packs re-read the shared prefixes anyway (the 9x pack
/// redundancy is inherent to band-parallel SYRK — the transpose only
/// changes strided→contiguous, it does not remove it). Bitwise agreement
/// held throughout. Do not re-hoist without removing the prefix re-reads
/// (OB-style even partition), not just re-patterning them.
///
/// NOTE (reverted trial 2026-09-26): worker-partitioned SYRK (one LPT task
/// per worker, each with a private prefix pack consumed `(NoTrans,
/// NoTrans)`, TLS-pooled tiles) loses in-situ 3/3 (spotrf-1024 2.27 →
/// 2.45ms) and regresses standalone small shapes. Lesson: the wall is the
/// single biggest unit's pack+sweep — worker ranges replicate the biggest
/// band's work on the critical path, so total-work savings (3x fewer pack
/// bytes) never move it, while LPT assignment, private-pack fan-out, and
/// heap stashes add coordination. Lowering `SYRK_NEST_MIN_MACS` to nest
/// big bands also ties (nested threaded GEMMs at these shapes run the
/// serial nest internally plus scope overhead). The remaining lever is the
/// serial kernel rate on the critical `(32, ~1000, 48)` band itself.
///
/// Minimum `n` before the band-rows fan out over rayon.
const SYRK_PAR_MIN_N: usize = 128;

/// Minimum multiply-accumulates before an off-diagonal band GEMM nests the
/// parallel entry even when the outer scope already fills the machine.
/// SYRK bands are triangularly skewed (the last band holds ~30x the first),
/// so a huge band serialised onto one worker would own the critical path
/// while nesting it costs one scope. Either entry accumulates per element
/// in the same order, so the choice is bitwise invisible. 4M (not 64M):
/// Cholesky calls with `k = panel width` (<= 128), so no band ever reached
/// the old 64M and large bands serialised; interleaved A/B gives -9..-11%
/// at potrf-2048 and -12..-15% at 1024 for 4M over 64M (1M ties at 2048
/// but adds scope overhead; 8M keeps half the gain; 512 neutral),
/// residuum unchanged. Override with `SONIUM_SYRK_NEST_MACS`.
const SYRK_NEST_MIN_MACS: usize = 4_000_000;

/// Unblocked diagonal-block update for the packed `Trans` path: `C_blk +=
/// alpha * P1ᵀ * P2` with `k × nb` contiguous panels, stride `ldp`. The two
/// panels coincide except for Hermitian updates, where `P1` is conjugated
/// and `P2` is plain.
fn diag_update_packed<T: TrsmScalar>(
    uplo: TrsmUplo,
    nb: usize,
    k: usize,
    alpha: T,
    p1: &[T],
    p2: &[T],
    ldp: usize,
    c_blk: &mut [T],
    ldc: usize,
) {
    if alpha == T::zero() {
        return;
    }
    match uplo {
        TrsmUplo::Lower => {
            for i in 0..nb {
                for j in 0..=i {
                    let mut s = T::zero();
                    for r in 0..k {
                        s = s + p1[r * ldp + i] * p2[r * ldp + j];
                    }
                    c_blk[i * ldc + j] = c_blk[i * ldc + j] + alpha * s;
                }
            }
        }
        TrsmUplo::Upper => {
            for i in 0..nb {
                for j in i..nb {
                    let mut s = T::zero();
                    for r in 0..k {
                        s = s + p1[r * ldp + i] * p2[r * ldp + j];
                    }
                    c_blk[i * ldc + j] = c_blk[i * ldc + j] + alpha * s;
                }
            }
        }
    }
}

/// Pack column panel `A[:, c0..c0 + nb]` (`k × n` stored) contiguously into
/// `out` (`k × nb`), conjugating when `conj` (complex `ConjTrans`).
fn pack_panel<T: TrsmScalar>(
    a: &[T],
    lda: usize,
    k: usize,
    c0: usize,
    nb: usize,
    conj: bool,
    out: &mut [T],
) {
    for r in 0..k {
        for c in 0..nb {
            let v = a[r * lda + c0 + c];
            out[r * nb + c] = if conj { v.conj() } else { v };
        }
    }
}

/// Core driver shared by the four per-type wrappers. `conj_pack`
/// conjugates while packing (complex `ConjTrans` only).
fn syrk_core<T: TrsmScalar + Send + Sync>(
    uplo: TrsmUplo,
    trans: Transpose,
    conj_pack: bool,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
    gemm: GemmFn<T>,
    gemm_diag: GemmFn<T>,
) {
    if n == 0 {
        return;
    }
    let zero = T::zero();
    let one = T::one();
    if k == 0 || alpha == zero {
        // No update: `C := beta * C`. Band-fused triage never runs (no
        // bands), so triage here — parallel past
        // `SYRK_TRIAGE_PAR_MIN_N` (the serial scalar loop costs ~1.8ms
        // at n=2048).
        if beta != one {
            triage_triangle(uplo, n, beta, c, ldc);
        }
        return;
    }
    // Beta triage is fused into each band task (see `syrk_band`): band
    // `bi` owns `C` rows `[bi*nb, ...)` and zeroes/scales its own
    // referenced-triangle elements before accumulating — same elements
    // and scalar ops as the old serial whole-triangle pre-scale
    // (bitwise identical), with no extra pass or barrier.
    syrk_bands(
        uplo, trans, conj_pack, n, k, alpha, beta, a, lda, c, ldc, gemm, gemm_diag,
    );
}

/// Band dispatch of [`syrk_core`] over a packed (`lda == k`) or
/// hoisted-strided `NoTrans` `A`, or any `Trans` `A`.
#[allow(clippy::too_many_arguments)]
fn syrk_bands<T: TrsmScalar + Send + Sync>(
    uplo: TrsmUplo,
    trans: Transpose,
    conj_pack: bool,
    n: usize,
    k: usize,
    alpha: T,
    beta: T,
    a: &[T],
    lda: usize,
    c: &mut [T],
    ldc: usize,
    gemm: GemmFn<T>,
    gemm_diag: GemmFn<T>,
) {
    let notrans = matches!(trans, Transpose::NoTrans);
    // Band-rows over C: task bi owns diagonal block bi plus its
    // triangle-side off-diagonal blocks — disjoint across tasks.
    // Bands are triangularly skewed (band bi owns ~bi x the first band's
    // work), so exact fill (bands == workers) parks each worker behind
    // one band and the wall is the biggest band. Target ~2x bands per
    // worker at small n: workers average a big band with a small one and
    // the wall drops toward the mean (n=512,k=64 f64: 0.198ms -> ~0.10
    // at 16T). Narrower serial tiles run slightly slower per flop, and
    // extra spawns cost, so only split at or below fill parity, never
    // below 16 wide, never on the serial path, and never above n=512
    // (big shapes keep the tuned 32-wide bands).
    let (bands, nb) = syrk_band_plan(n, k);
    let parallel = n >= SYRK_PAR_MIN_N;
    // Nested fan-out reaches idle cores when the band scope underfills the
    // machine (bands <= workers: at equality every worker holds one band,
    // and nesting the skewed tail still wins); otherwise bands run the
    // serial kernel except past `SYRK_NEST_MIN_MACS` (see
    // `pick_offdiag_gemm`). NOTE (trial 2026-09-26): gating per-band GEMM
    // nesting to severe underfill (bands <= workers/8) tied at 2048
    // (26.4 vs 26.6ms) — 2048 is traffic-bound (out-of-L3 A re-reads), not
    // dispatch-bound, so the gate stays as-is.
    // Nesting-threshold override, parsed once per call (see
    // `syrk_nest_macs`).
    let (nest_outer, nest_macs) = syrk_band_nest(bands);
    // Serial: never nest — one worker gains nothing from the MT GEMM
    // dispatch machinery (pool checks, scope, routing ≈ 5-20µs/band),
    // and wide serial bands would trip `SYRK_NEST_MIN_MACS` constantly.
    // Forcing `MAX` routes every band to the serial kernel.
    let nest_macs = if rayon::current_num_threads().max(1) == 1 {
        usize::MAX
    } else {
        nest_macs
    };
    if parallel {
        let a_send = SendConstPtr(a.as_ptr());
        let c_send = SendMutPtr(c.as_mut_ptr());
        // Keep a nesting tree on one pool: bands that fan out again join
        // the small pool instead of straddling pools (see
        // `scope_for_nested_outer`).
        // Largest-first band order: bands are triangularly skewed (band
        // `bi` owns ~`bi` x the first band's work), so spawning small-first
        // parks the big bands in the last wave while workers idle (tail
        // latency). Reversed, the caller and thieves take big bands first
        // and small ones fill the tail (LPT scheduling). Measured 2026-09:
        // interleaved REV/FWD x3 at 8T wins ~6% at n=960,k=64 (0.43 vs
        // 0.46) and ties at n=992,k=256 — do not flip back.
        // Hot-workers tier for short top-level updates (see
        // [`super::hot_tier_applies`]): the band scope concentrates on
        // L3-neighbour hot cores instead of spraying parked ones. The
        // `nest_outer` fan-out policy passes through unchanged — only the
        // outer pool differs.
        let macs = n.saturating_mul(n).saturating_mul(k) / 2;
        // Route by size, not by nesting: below `SMALL_MAX_MACS` the
        // capped pools (hot/tiny/small) beat the 48-wide small pool the
        // nested-outer hop lands on for mid-size updates (29M MACs:
        // tiny-16 recovers the 16-thread rate the 48-wide tree loses to
        // barrier/steal coordination), and nested bands still run inline
        // on whichever capped pool is picked. Huge updates keep the
        // all-48 nesting tree (see `scope_for_nested_outer`).
        if macs < super::SMALL_MAX_MACS {
            super::scope_for_macs(macs, |s| {
                spawn_syrk_bands(
                    s, uplo, notrans, conj_pack, bands, n, k, alpha, beta, a_send, lda, c_send,
                    ldc, gemm, gemm_diag, nest_outer, nest_macs, nb,
                );
            });
        } else {
            super::scope_for_nested_outer(nest_outer, |s| {
                spawn_syrk_bands(
                    s, uplo, notrans, conj_pack, bands, n, k, alpha, beta, a_send, lda, c_send,
                    ldc, gemm, gemm_diag, nest_outer, nest_macs, nb,
                );
            });
        }
    } else {
        for bi in 0..bands {
            syrk_band(
                uplo,
                notrans,
                conj_pack,
                bi,
                nb,
                n,
                k,
                alpha,
                beta,
                a.as_ptr(),
                lda,
                c.as_mut_ptr(),
                ldc,
                gemm,
                gemm_diag,
                nest_outer,
                nest_macs,
            );
        }
    }
}

/// Diagnostic tuning gate: `SONIUM_SYRK_NEST_MACS=n` overrides
/// [`SYRK_NEST_MIN_MACS`] (parsed once per SYRK call; the small-band
/// threshold keeps the 1/16 ratio). The default was tuned when `k` was the
/// trailing order; Cholesky now calls with `k = panel width` (<= 128), so
/// no band ever reaches 64M macs and large bands serialise onto one
/// worker. Interleaved A/B only — host drift exceeds small deltas across
/// runs.
fn syrk_nest_macs() -> usize {
    std::env::var("SONIUM_SYRK_NEST_MACS")
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|macs| *macs > 0)
        .unwrap_or(SYRK_NEST_MIN_MACS)
}

/// Pick the off-diagonal band GEMM entry. Nest the parallel kernel when the
/// outer band scope underfills the machine (`nest_outer`: the fan-out is
/// the only way to reach idle cores — measured faster at few bands), or
/// when the band itself is huge (serialising it onto one worker would own
/// the critical path). Otherwise the serial kernel: the outer scope
/// already spreads bands over threads, and a nested fan-out (pack scopes
/// plus N-/M-split scopes per Kc pass) costs more than it gains.
#[inline]
#[allow(clippy::too_many_arguments)]
fn pick_offdiag_gemm<T: TrsmScalar + Send + Sync>(
    nest_outer: bool,
    nest_macs: usize,
    nbi: usize,
    ncols: usize,
    k: usize,
    gemm: GemmFn<T>,
    gemm_diag: GemmFn<T>,
) -> GemmFn<T> {
    let macs = nbi.saturating_mul(ncols).saturating_mul(k);
    if macs >= nest_macs || (nest_outer && macs >= nest_macs / 16) {
        gemm
    } else {
        gemm_diag
    }
}

/// Spawn every band-row task of [`syrk_core`] onto `s` in largest-first
/// order (see the LPT note at the call site). Shared by the hot-tier and
/// the nested-outer scope drivers so both run the identical task tree.
#[allow(clippy::too_many_arguments)]
fn spawn_syrk_bands<'scope, T: TrsmScalar + Send + Sync + 'scope>(
    s: &rayon::Scope<'scope>,
    uplo: TrsmUplo,
    notrans: bool,
    conj_pack: bool,
    bands: usize,
    n: usize,
    k: usize,
    alpha: T,
    beta: T,
    a_send: SendConstPtr<T>,
    lda: usize,
    c_send: SendMutPtr<T>,
    ldc: usize,
    gemm: GemmFn<T>,
    gemm_diag: GemmFn<T>,
    nest_outer: bool,
    nest_macs: usize,
    nb: usize,
) {
    // NOTE (reverted trial): LPT-mirrored band pairing (one task per
    // largest+smallest pair, halving spawns; order-free and bitwise
    // identical) lost in-situ dpotrf-2048 decisively (16.7 -> 19.0ms med).
    // Pairing serialises each small band behind its big twin's nested
    // parallel GEMM join on the same worker, destroying the overlap the
    // 4x oversubscription buys. Do not re-pair without preserving
    // nest/small overlap.
    for bi in (0..bands).rev() {
        s.spawn(move |_| {
            let a_send: SendConstPtr<T> = a_send;
            let c_send: SendMutPtr<T> = c_send;
            // SAFETY: task `bi` owns `C` rows `[i0, i1)` (hence its
            // triangle blocks); `A` is read-only. Bounds per the
            // caller contract. (`syrk_band` is safe; its own
            // unsafe blocks document their contracts inside.)
            syrk_band(
                uplo, notrans, conj_pack, bi, nb, n, k, alpha, beta, a_send.0, lda, c_send.0, ldc,
                gemm, gemm_diag, nest_outer, nest_macs,
            );
        });
    }
}

/// One band-row task of [`syrk_core`]: diagonal block plus triangle-side
/// off-diagonal blocks for band `bi`. Raw pointers so the rayon scope and
/// the serial fallback share the body.
fn syrk_band<T: TrsmScalar + Send + Sync>(
    uplo: TrsmUplo,
    notrans: bool,
    conj_pack: bool,
    bi: usize,
    nb: usize,
    n: usize,
    k: usize,
    alpha: T,
    beta: T,
    a: *const T,
    lda: usize,
    c: *mut T,
    ldc: usize,
    gemm: GemmFn<T>,
    gemm_diag: GemmFn<T>,
    nest_outer: bool,
    nest_macs: usize,
) {
    // SAFETY: caller guarantees task `bi` owns `C` rows `[i0, i1)` with
    // `i1 - i0 <= nb`, and `A` holds its bands/panels.
    unsafe {
        let i0 = bi * nb;
        let nbi = (n - i0).min(nb);
        // Fused beta triage: this task owns `C` rows `[i0, i0 + nbi)`,
        // so it zeroes/scales its own referenced-triangle elements
        // before the beta=1 accumulations below — same elements and
        // scalar ops as the old serial whole-triangle pre-scale in
        // `syrk_core` (bitwise identical), with no extra pass, no
        // barrier, and the rows hot in cache for the kernels. Only the
        // referenced triangle is touched, so the untouched triangle
        // the stash/restore preserves keeps its bit-identical values.
        // No-op at `beta == 1`.
        triage_band_rows(uplo, i0, nbi, n, beta, c, ldc);
        if notrans {
            // Hermitian NoTrans (see `herk_c64`): the second factor is
            // conjugated (`A_bi · A_topᴴ`), consumed by the sweep's
            // Trans-B block pack — no strided repack of a `(k × n)`
            // operand. Symmetric updates keep the plain transpose.
            let transb = if conj_pack {
                Transpose::ConjTrans
            } else {
                Transpose::Trans
            };
            let a_band = core::slice::from_raw_parts(a.add(i0 * lda), (nbi - 1) * lda + k);
            // Diagonal merged into the off-diagonal update: one GEMM per
            // band covers C[bi, 0..i1) (Lower) instead of a separate
            // diagonal call plus a merged off-diagonal call — the diagonal
            // call's dispatch/setup costs more than its ~130k flops at
            // small `k`. The GEMM's beta=1 accumulate would pollute the
            // diagonal block's untouched triangle, so its strict triangle
            // (<= 496 elements) is stashed on the stack and restored after
            // (the public contract leaves the untouched triangle
            // bit-identical). FP summation order differs from the old
            // tmp-then-add: tolerance-equal, not bitwise.
            // Off-diagonal, merged: C[bi, 0..j1] += alpha * A_bi * A_topᵀ
            // in one GEMM instead of one per block pair — fewer dispatches,
            // larger shapes. Hoisting the per-band `A_top` packs into one
            // `Aᵀ` pack per call was trialled twice (off-diag+diag, then
            // off-diag-only with a streaming build) and loses 20-40% on
            // every shape: the small-stride transposed gather is already
            // streaming, and the pack build plus contiguous-`B` dispatch
            // change cost more than the redundant strided reads save. A
            // third trial (transpose once, consume slices via
            // `(NoTrans, NoTrans)` row-copy packs — no double transpose)
            // also lost back-to-back on small-`k` (n=992,k=32: 0.33-0.49
            // vs 0.24-0.26ms base): the serial scalar transpose dominates
            // small shapes. Hoist stays dead unless the transpose itself
            // gets cheap (parallel/vectorised) — the per-band consumption
            // half of the idea is validated (bitwise-identical, maxdiff 0).
            match uplo {
                TrsmUplo::Lower => {
                    let j1 = bi * nb;
                    // Extended width covers the diagonal block (no separate
                    // diagonal call, no `j1 > 0` guard: band 0's call is
                    // the diagonal update).
                    let i1 = j1 + nbi;
                    let a_top = core::slice::from_raw_parts(a, (i1 - 1) * lda + k);
                    let c_band =
                        core::slice::from_raw_parts_mut(c.add(i0 * ldc), (nbi - 1) * ldc + i1);
                    // Stash the diagonal block's strict upper triangle
                    // (local (i, j'), j' > i); restored after the GEMM.
                    // Wide serial bands overflow the stack array: heap
                    // fallback (the MT path always fits the stack array,
                    // behaviour identical).
                    let mut stash_stack =
                        core::array::from_fn::<_, { SYRK_NB * SYRK_NB }, _>(|_| T::zero());
                    let mut stash_vec = Vec::new();
                    if nbi > SYRK_NB {
                        stash_vec = vec![T::zero(); nbi * nbi];
                    }
                    let stash: &mut [T] = if nbi > SYRK_NB {
                        stash_vec.as_mut_slice()
                    } else {
                        &mut stash_stack[..]
                    };
                    for i in 0..nbi {
                        for jp in (i + 1)..nbi {
                            stash[i * nbi + jp] = c_band[i * ldc + j1 + jp];
                        }
                    }
                    // Small bands run serial: the outer scope already
                    // spreads bands over threads (see SYRK_PAR_MIN_MACS).
                    let g = pick_offdiag_gemm(nest_outer, nest_macs, nbi, i1, k, gemm, gemm_diag);
                    g(
                        Transpose::NoTrans,
                        transb,
                        nbi,
                        i1,
                        k,
                        alpha,
                        a_band,
                        lda,
                        a_top,
                        lda,
                        T::one(),
                        c_band,
                        ldc,
                    );
                    for i in 0..nbi {
                        for jp in (i + 1)..nbi {
                            c_band[i * ldc + j1 + jp] = stash[i * nbi + jp];
                        }
                    }
                }
                TrsmUplo::Upper => {
                    // Merged mirror of the Lower case: C[bi, i0..n],
                    // extended left to cover the diagonal block (no
                    // separate diagonal call, no `j0 < n` guard: the last
                    // band's call is the diagonal update). The diagonal
                    // block's strict lower triangle is stashed and
                    // restored around the GEMM.
                    let ncols = n - i0;
                    let a_bot = core::slice::from_raw_parts(a.add(i0 * lda), (ncols - 1) * lda + k);
                    let c_band = core::slice::from_raw_parts_mut(
                        c.add(i0 * ldc + i0),
                        (nbi - 1) * ldc + ncols,
                    );
                    // Heap fallback for wide serial bands (see Lower).
                    let mut stash_stack =
                        core::array::from_fn::<_, { SYRK_NB * SYRK_NB }, _>(|_| T::zero());
                    let mut stash_vec = Vec::new();
                    if nbi > SYRK_NB {
                        stash_vec = vec![T::zero(); nbi * nbi];
                    }
                    let stash: &mut [T] = if nbi > SYRK_NB {
                        stash_vec.as_mut_slice()
                    } else {
                        &mut stash_stack[..]
                    };
                    for i in 0..nbi {
                        for jp in 0..i.min(nbi) {
                            stash[i * nbi + jp] = c_band[i * ldc + jp];
                        }
                    }
                    let g =
                        pick_offdiag_gemm(nest_outer, nest_macs, nbi, ncols, k, gemm, gemm_diag);
                    g(
                        Transpose::NoTrans,
                        transb,
                        nbi,
                        ncols,
                        k,
                        alpha,
                        a_band,
                        lda,
                        a_bot,
                        lda,
                        T::one(),
                        c_band,
                        ldc,
                    );
                    for i in 0..nbi {
                        for jp in 0..i.min(nbi) {
                            c_band[i * ldc + jp] = stash[i * nbi + jp];
                        }
                    }
                }
            }
        } else {
            // Trans path: pack this band's k × nbi column panel (with
            // conjugation for Hermitian updates), then run the packed
            // kernels. The scratch is task-local.
            let mut pack = vec![T::zero(); k * nbi];
            {
                let a_ref = core::slice::from_raw_parts(a, k * lda);
                pack_panel(a_ref, lda, k, i0, nbi, conj_pack, &mut pack);
            }
            // Hermitian updates need a plain second panel alongside the
            // conjugated first panel; symmetric updates reuse one pack.
            let mut pack_plain = Vec::new();
            if conj_pack {
                pack_plain.resize(k * nbi, T::zero());
                let a_ref = core::slice::from_raw_parts(a, k * lda);
                pack_panel(a_ref, lda, k, i0, nbi, false, &mut pack_plain);
            }
            let (p1, p2): (&[T], &[T]) = match pack_plain.is_empty() {
                true => (&pack, &pack),
                false => (&pack, &pack_plain),
            };
            let c_blk =
                core::slice::from_raw_parts_mut(c.add(i0 * ldc + i0), (nbi - 1) * ldc + nbi);
            diag_update_packed(uplo, nbi, k, alpha, p1, p2, nbi, c_blk, ldc);
            // Off-diagonal, merged: pack columns 0..j1 once, then one
            // GEMM C[bi, 0..j1] += alpha * P_biᵀ * P_top. The second
            // factor is never conjugated (see the diagonal above).
            match uplo {
                TrsmUplo::Lower => {
                    let j1 = bi * nb;
                    if j1 > 0 {
                        let mut pack_top = vec![T::zero(); k * j1];
                        {
                            let a_ref = core::slice::from_raw_parts(a, k * lda);
                            pack_panel(a_ref, lda, k, 0, j1, false, &mut pack_top);
                        }
                        let c_band =
                            core::slice::from_raw_parts_mut(c.add(i0 * ldc), (nbi - 1) * ldc + j1);
                        let g =
                            pick_offdiag_gemm(nest_outer, nest_macs, nbi, j1, k, gemm, gemm_diag);
                        g(
                            Transpose::Trans,
                            Transpose::NoTrans,
                            nbi,
                            j1,
                            k,
                            alpha,
                            &pack,
                            nbi,
                            &pack_top,
                            j1,
                            T::one(),
                            c_band,
                            ldc,
                        );
                    }
                }
                TrsmUplo::Upper => {
                    // Merged mirror: pack columns j0..n once, one GEMM.
                    let j0 = (bi + 1) * nb;
                    if j0 < n {
                        let ncols = n - j0;
                        let mut pack_right = vec![T::zero(); k * ncols];
                        {
                            let a_ref = core::slice::from_raw_parts(a, k * lda);
                            pack_panel(a_ref, lda, k, j0, ncols, false, &mut pack_right);
                        }
                        let c_band = core::slice::from_raw_parts_mut(
                            c.add(i0 * ldc + j0),
                            (nbi - 1) * ldc + ncols,
                        );
                        let g = pick_offdiag_gemm(
                            nest_outer, nest_macs, nbi, ncols, k, gemm, gemm_diag,
                        );
                        g(
                            Transpose::Trans,
                            Transpose::NoTrans,
                            nbi,
                            ncols,
                            k,
                            alpha,
                            &pack,
                            nbi,
                            &pack_right,
                            ncols,
                            T::one(),
                            c_band,
                            ldc,
                        );
                    }
                }
            }
        }
    }
}

/// Minimum `n` for the parallel early-path beta triage in
/// [`triage_triangle`]: below this the serial scalar loop beats a scope
/// (~30µs at n=256 vs ~15-30µs scope overhead); at n=512 the serial
/// ~120µs already loses to the fanned-out rows.
const SYRK_TRIAGE_PAR_MIN_N: usize = 512;

/// Zero/scale the referenced-triangle elements of `C` rows `[i0, i0 +
/// nbi)` (`n × n`, stride `ldc`): the per-band slice of
/// [`set_triangle`]/[`scale_triangle`], fused into each [`syrk_band`]
/// task and shared by the parallel early-path triage. Same elements
/// and scalar ops as the whole-triangle forms (bitwise identical), a
/// no-op at `beta == 1`. Only the referenced triangle is touched.
///
/// # Safety
///
/// Rows `[i0, i0 + nbi)` are exclusively writable, `i0 + nbi <= n`,
/// and every indexed element is in bounds.
unsafe fn triage_band_rows<T: TrsmScalar>(
    uplo: TrsmUplo,
    i0: usize,
    nbi: usize,
    n: usize,
    beta: T,
    c: *mut T,
    ldc: usize,
) {
    if beta == T::one() {
        return;
    }
    // Hoisted out of the element loops: one branch per band, not per
    // element (the flag is loop-invariant either way).
    if beta == T::zero() {
        // SAFETY: per the caller contract.
        unsafe {
            match uplo {
                TrsmUplo::Lower => {
                    for i in i0..i0 + nbi {
                        for j in 0..=i {
                            *c.add(i * ldc + j) = T::zero();
                        }
                    }
                }
                TrsmUplo::Upper => {
                    for i in i0..i0 + nbi {
                        for j in i..n {
                            *c.add(i * ldc + j) = T::zero();
                        }
                    }
                }
            }
        }
    } else {
        // SAFETY: per the caller contract.
        unsafe {
            match uplo {
                TrsmUplo::Lower => {
                    for i in i0..i0 + nbi {
                        for j in 0..=i {
                            let p = c.add(i * ldc + j);
                            *p = beta * *p;
                        }
                    }
                }
                TrsmUplo::Upper => {
                    for i in i0..i0 + nbi {
                        for j in i..n {
                            let p = c.add(i * ldc + j);
                            *p = beta * *p;
                        }
                    }
                }
            }
        }
    }
}

/// Whole-triangle beta triage for [`syrk_core`]'s early path (`k == 0`
/// or `alpha == 0`: `C := beta * C`, no update, so band-fused triage
/// never runs). Serial below [`SYRK_TRIAGE_PAR_MIN_N`], else one scope
/// of disjoint band-row tasks over the pinned pool.
fn triage_triangle<T: TrsmScalar + Send + Sync>(
    uplo: TrsmUplo,
    n: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    if beta == T::one() {
        return;
    }
    if n < SYRK_TRIAGE_PAR_MIN_N {
        if beta == T::zero() {
            set_triangle(uplo, n, c, ldc, T::zero());
        } else {
            scale_triangle(uplo, n, c, ldc, beta);
        }
        return;
    }
    let c_send = SendMutPtr(c.as_mut_ptr());
    super::scope_pinned(|s| {
        let (bands, nb) = syrk_band_plan(n, 0);
        for bi in (0..bands).rev() {
            s.spawn(move |_| {
                let c_send: SendMutPtr<T> = c_send;
                let i0 = bi * nb;
                let nbi = (n - i0).min(nb);
                // SAFETY: task `bi` owns `C` rows `[i0, i0 + nbi)`,
                // disjoint across tasks; bounds per the plan.
                unsafe { triage_band_rows(uplo, i0, nbi, n, beta, c_send.0, ldc) };
            });
        }
    });
}

/// Overwrite the referenced triangle of `C` (`n × n`, stride `ldc`).
fn set_triangle<T: TrsmScalar>(uplo: TrsmUplo, n: usize, c: &mut [T], ldc: usize, v: T) {
    match uplo {
        TrsmUplo::Lower => {
            for i in 0..n {
                for j in 0..=i {
                    c[i * ldc + j] = v;
                }
            }
        }
        TrsmUplo::Upper => {
            for i in 0..n {
                for j in i..n {
                    c[i * ldc + j] = v;
                }
            }
        }
    }
}

/// Scale the referenced triangle of `C` in place.
fn scale_triangle<T: TrsmScalar>(uplo: TrsmUplo, n: usize, c: &mut [T], ldc: usize, beta: T) {
    match uplo {
        TrsmUplo::Lower => {
            for i in 0..n {
                for j in 0..=i {
                    c[i * ldc + j] = beta * c[i * ldc + j];
                }
            }
        }
        TrsmUplo::Upper => {
            for i in 0..n {
                for j in i..n {
                    c[i * ldc + j] = beta * c[i * ldc + j];
                }
            }
        }
    }
}

// ---------- team-dispatch API --------------------------------------------------
// Band split plus serial band slices for SYRK, so a caller-owned team scope
// (one scope per Cholesky panel, bands statically assigned to tasks, an
// internal barrier before the next phase) can run bands without paying the
// per-band spawn volume. Same split and same nest decisions
// [`syrk_bands`] makes (which is refactored onto [`syrk_band_plan`] and
// [`syrk_band_nest`] below); assembling a call from slices is bitwise
// identical to the whole call (bands write disjoint `C` rows; nesting is
// order-free per the `SYRK_NEST_MIN_MACS` docs).

/// Band split for an `n`-row update: `(bands, nb)`, exactly as
/// [`syrk_bands`] computes it (narrow-`n` adaptation included). Must be
/// called in the scope the bands will run in (`workers` is read from the
/// executing pool, as the default path does).
pub fn syrk_band_plan(n: usize, k: usize) -> (usize, usize) {
    let workers = rayon::current_num_threads().max(1);
    if workers == 1 {
        // Serial: no balance to keep — wide bands minimise pack/A
        // re-read/dispatch redundancy (see `SYRK_NB_SERIAL`), but only
        // past n = 1024: below that the 656KB band footprint thrashes
        // L2 and the n·nb/2 stash dominates (A/B 2026-09-27: narrow
        // wins 5-30% at n <= 768, wide wins 7-31% at n >= 1024, both
        // precisions). Small n keeps the tuned narrow path.
        let nb = if n >= 1024 { SYRK_NB_SERIAL } else { SYRK_NB };
        let nb = nb.min(n.max(1));
        return (n.div_ceil(nb), nb);
    }
    // Wide MT bands (shipped 2026-09-28, 3-round A/B 18/18 big-cell
    // wins +4-17%): nb=64 halves A re-read + pack traffic vs nb=32 at
    // coarser LPT balance (32 bands over 16 workers still balances;
    // nb=128 untested-quiet, dropped — 16 bands risks an exact-fill
    // skew wall). Thin-k (k < 256) keeps legacy
    // `SYRK_NB`: thin updates are dispatch-bound and coarser bands
    // lose to 2x there (976x48 measured -50% at nb=64). The narrow-n
    // adaptation below keeps legacy `SYRK_NB` so small shapes are
    // immune. `SONIUM_SYRK_MTNB=0` restores nb=32 everywhere for A/B.
    let mut nb = if k >= 256
        && std::env::var("SONIUM_SYRK_MTNB")
            .map(|v| v.as_bytes().first() != Some(&b'0'))
            .unwrap_or(true)
    {
        64
    } else {
        SYRK_NB
    };
    if (SYRK_PAR_MIN_N..=512).contains(&n) && n.div_ceil(SYRK_NB) <= workers {
        nb = n.div_ceil(2 * workers).clamp(16, SYRK_NB);
    }
    (n.div_ceil(nb), nb)
}

/// Nest decisions for `bands` bands: `(nest_outer, nest_macs)`, exactly as
/// [`syrk_bands`] computes them.
pub fn syrk_band_nest(bands: usize) -> (bool, usize) {
    (
        bands <= rayon::current_num_threads().max(1),
        syrk_nest_macs(),
    )
}

/// Beta pre-scale of the referenced triangle, exactly as [`syrk_core`]
/// applies it before bands (zero-fill at `beta == 0`, scale otherwise,
/// no-op at `beta == 1`). Team callers run this once per call, then the
/// band slices.
pub fn syrk_beta_scale<T: TrsmScalar>(uplo: TrsmUplo, n: usize, beta: T, c: &mut [T], ldc: usize) {
    if beta == T::zero() {
        set_triangle(uplo, n, c, ldc, T::zero());
    } else if beta != T::one() {
        scale_triangle(uplo, n, c, ldc, beta);
    }
}

/// Serial SYRK band slice for band `bi` (of [`syrk_band_plan`'s split).
/// `nest_*` come from [`syrk_band_nest`]; remaining params mirror
/// [`syrk_f64`]. Band `bi` owns `C` rows `[bi*nb, ...)`. Team callers
/// pre-scale `C` via [`syrk_beta_scale`]; the fused triage runs with
/// beta = 1 (no-op).
///
/// # Safety
///
/// `a` holds the band's rows/panels, `c` band `bi`'s rows are exclusively
/// writable; all reads in bounds (see [`syrk_band`]).
pub unsafe fn syrk_band_f64(
    uplo: TrsmUplo,
    trans: Transpose,
    bi: usize,
    nb: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    c: *mut f64,
    ldc: usize,
    nest_outer: bool,
    nest_macs: usize,
) {
    let t = match trans {
        Transpose::ConjTrans => Transpose::Trans,
        t => t,
    };
    // SAFETY: per the caller contract (`syrk_band` is safe; only its body
    // is `unsafe` internally). This entry stays `unsafe` because the
    // disjoint-rows contract is on the cross-crate caller.
    syrk_band(
        uplo,
        matches!(t, Transpose::NoTrans),
        false,
        bi,
        nb,
        n,
        k,
        alpha,
        1.0,
        a,
        lda,
        c,
        ldc,
        gemm_f64,
        gemm_f64_serial,
        nest_outer,
        nest_macs,
    )
}

/// [`syrk_band_f64`] for `f32`.
///
/// # Safety
///
/// See [`syrk_band_f64`].
pub unsafe fn syrk_band_f32(
    uplo: TrsmUplo,
    trans: Transpose,
    bi: usize,
    nb: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    c: *mut f32,
    ldc: usize,
    nest_outer: bool,
    nest_macs: usize,
) {
    let t = match trans {
        Transpose::ConjTrans => Transpose::Trans,
        t => t,
    };
    syrk_band(
        uplo,
        matches!(t, Transpose::NoTrans),
        false,
        bi,
        nb,
        n,
        k,
        alpha,
        1.0,
        a,
        lda,
        c,
        ldc,
        gemm_f32,
        gemm_f32_serial,
        nest_outer,
        nest_macs,
    )
}

/// [`syrk_band_f64`] for `Complex32` (symmetric update, as in [`syrk_c32`]).
///
/// # Safety
///
/// See [`syrk_band_f64`].
pub unsafe fn syrk_band_c32(
    uplo: TrsmUplo,
    trans: Transpose,
    bi: usize,
    nb: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: *const Complex32,
    lda: usize,
    c: *mut Complex32,
    ldc: usize,
    nest_outer: bool,
    nest_macs: usize,
) {
    let (t, conj) = match trans {
        Transpose::ConjTrans => (Transpose::Trans, true),
        t => (t, false),
    };
    syrk_band(
        uplo,
        matches!(t, Transpose::NoTrans),
        conj,
        bi,
        nb,
        n,
        k,
        alpha,
        Complex32::new(1.0, 0.0),
        a,
        lda,
        c,
        ldc,
        gemm_c32,
        gemm_c32,
        nest_outer,
        nest_macs,
    )
}

/// [`syrk_band_f64`] for `Complex64` (as in [`syrk_c64`]).
///
/// # Safety
///
/// See [`syrk_band_f64`].
pub unsafe fn syrk_band_c64(
    uplo: TrsmUplo,
    trans: Transpose,
    bi: usize,
    nb: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: *const Complex64,
    lda: usize,
    c: *mut Complex64,
    ldc: usize,
    nest_outer: bool,
    nest_macs: usize,
) {
    let (t, conj) = match trans {
        Transpose::ConjTrans => (Transpose::Trans, true),
        t => (t, false),
    };
    syrk_band(
        uplo,
        matches!(t, Transpose::NoTrans),
        conj,
        bi,
        nb,
        n,
        k,
        alpha,
        Complex64::new(1.0, 0.0),
        a,
        lda,
        c,
        ldc,
        gemm_c64,
        gemm_c64,
        nest_outer,
        nest_macs,
    )
}

// ---------- per-type wrappers ------------------------------------------------

/// `C := alpha * op(A) * op(A)ᵀ + beta * C` for `f32`, row-major, writing
/// only the triangle named by `uplo`.
///
/// `A` is `n × k` (`NoTrans`) or `k × n` (`Trans`) with row stride `lda`;
/// `C` is `n × n` with row stride `ldc`. Padding and the opposite triangle
/// are never touched.
pub fn syrk_f32(
    uplo: TrsmUplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    let t = match trans {
        Transpose::ConjTrans => Transpose::Trans,
        t => t,
    };
    syrk_core(
        uplo,
        t,
        false,
        n,
        k,
        alpha,
        a,
        lda,
        beta,
        c,
        ldc,
        gemm_f32,
        gemm_f32_serial,
    );
}

/// `f64` twin of [`syrk_f32`].
pub fn syrk_f64(
    uplo: TrsmUplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    let t = match trans {
        Transpose::ConjTrans => Transpose::Trans,
        t => t,
    };
    syrk_core(
        uplo,
        t,
        false,
        n,
        k,
        alpha,
        a,
        lda,
        beta,
        c,
        ldc,
        gemm_f64,
        gemm_f64_serial,
    );
}

/// `Complex32` twin of [`syrk_f32`] (symmetric update, no conjugation).
/// `ConjTrans` requests the Hermitian update `C := alpha * Aᴴ * A + beta *
/// C`, folded into the panel pack.
pub fn syrk_c32(
    uplo: TrsmUplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    let (t, conj) = match trans {
        Transpose::ConjTrans => (Transpose::Trans, true),
        t => (t, false),
    };
    // Diag bands run the true serial sweep: `gemm_c32` nests a parallel
    // sweep above 100K MACs, so passing it as the diag entry fans every
    // big band out over the shared band pool (cross-band working-set
    // thrash + oversubscription). The outer band scope already spreads
    // bands over workers; nested fan-out only pays past `nest_macs`.
    syrk_core(
        uplo,
        t,
        conj,
        n,
        k,
        alpha,
        a,
        lda,
        beta,
        c,
        ldc,
        gemm_c32,
        gemm_c32_serial,
    );
}

/// `Complex64` twin of [`syrk_f32`]; `ConjTrans` requests the Hermitian
/// update like [`syrk_c32`].
pub fn syrk_c64(
    uplo: TrsmUplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    let (t, conj) = match trans {
        Transpose::ConjTrans => (Transpose::Trans, true),
        t => (t, false),
    };
    // Complex GEMM is already serial, so the diag entry is the same kernel.
    syrk_core(
        uplo, t, conj, n, k, alpha, a, lda, beta, c, ldc, gemm_c64, gemm_c64,
    );
}

/// Hermitian rank-`k` update from a NoTrans operand (BLAS `zherk` with
/// `TRANS = 'N'`): `C := alpha * A * Aᴴ + beta * C`, `A` is `n × k`
/// with row stride `lda`, writing only the triangle named by `uplo`.
/// Unlike [`syrk_c64`] with `ConjTrans` (which repacks a `(k × n)`
/// operand with strided reads), this consumes the `(n × k)` operand
/// streaming through the NoTrans band arm, conjugating the second GEMM
/// factor (free inside the sweep's B-block pack). Cholesky's Hermitian
/// path uses this on the already-packed `l21` instead of building and
/// re-reading the transposed `l21h`.
pub fn herk_c64(
    uplo: TrsmUplo,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    syrk_core(
        uplo,
        Transpose::NoTrans,
        true,
        n,
        k,
        alpha,
        a,
        lda,
        beta,
        c,
        ldc,
        gemm_c64,
        gemm_c64,
    );
}

/// `Complex32` twin of [`herk_c64`] (BLAS `cherk`, `TRANS = 'N'`).
pub fn herk_c32(
    uplo: TrsmUplo,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    syrk_core(
        uplo,
        Transpose::NoTrans,
        true,
        n,
        k,
        alpha,
        a,
        lda,
        beta,
        c,
        ldc,
        gemm_c32,
        gemm_c32_serial,
    );
}

// ---------- correctness tests ------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::parallelism::env_test_lock;

    /// Wide-MT-band plan selection is deterministic (no timing):
    /// default-on nb=64 for big-k MT ((2048,256)->(32,64)), thin-k
    /// keeps 32 (k-gate), small-n keeps the narrow adaptation
    /// (immune), `SONIUM_SYRK_MTNB=0` restores nb=32 everywhere, and
    /// the serial path (1-thread pool) keeps wide/narrow ST bands
    /// regardless of the var.
    #[test]
    fn syrk_band_plan_mtnb_gates() {
        let _guard = env_test_lock();
        let pool16 = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        // SAFETY: `env_test_lock` serializes env flips; the switch is read
        // per call on the calling thread only.
        unsafe {
            std::env::remove_var("SONIUM_SYRK_MTNB");
        }
        pool16.install(|| {
            assert_eq!(syrk_band_plan(2048, 256), (32, 64));
            assert_eq!(syrk_band_plan(2048, 512), (32, 64));
            assert_eq!(syrk_band_plan(2048, 48), (64, 32), "thin-k keeps 32");
            assert_eq!(syrk_band_plan(300, 200), (19, 16), "narrow-n immune");
            assert_eq!(syrk_band_plan(100, 64), (4, 32), "below narrow floor");
        });
        pool1.install(|| {
            assert_eq!(syrk_band_plan(2048, 256), (8, 256), "ST wide immune");
            assert_eq!(syrk_band_plan(300, 200), (10, 32), "ST narrow immune");
        });
        // SAFETY: as above; selects the legacy arm.
        unsafe {
            std::env::set_var("SONIUM_SYRK_MTNB", "0");
        }
        pool16.install(|| {
            assert_eq!(syrk_band_plan(2048, 256), (64, 32));
            assert_eq!(syrk_band_plan(2048, 512), (64, 32));
            assert_eq!(syrk_band_plan(300, 200), (19, 16));
        });
        pool1.install(|| {
            assert_eq!(syrk_band_plan(2048, 256), (8, 256));
        });
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_SYRK_MTNB");
        }
    }

    fn prng(t: usize) -> f64 {
        let h = t
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 11;
        (h % 2000) as f64 / 1000.0 - 1.0
    }

    // Naive obviously-correct reference over the full triangle.
    fn ref_syrk_f64(
        uplo: TrsmUplo,
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    ) {
        let t = !matches!(trans, Transpose::NoTrans);
        // Scale the referenced triangle.
        for i in 0..n {
            for j in 0..n {
                let inside = match uplo {
                    TrsmUplo::Lower => j <= i,
                    TrsmUplo::Upper => j >= i,
                };
                if inside {
                    c[i * ldc + j] = beta * c[i * ldc + j];
                }
            }
        }
        if k == 0 || alpha == 0.0 {
            return;
        }
        // op(A) row r: NoTrans → A row r; Trans → A column r.
        let op = |r: usize, p: usize| -> f64 { if t { a[p * lda + r] } else { a[r * lda + p] } };
        for i in 0..n {
            for j in 0..n {
                let inside = match uplo {
                    TrsmUplo::Lower => j <= i,
                    TrsmUplo::Upper => j >= i,
                };
                if !inside {
                    continue;
                }
                let mut s = 0.0;
                for p in 0..k {
                    s += op(i, p) * op(j, p);
                }
                c[i * ldc + j] += alpha * s;
            }
        }
    }

    fn check_f64(
        uplo: TrsmUplo,
        trans: Transpose,
        n: usize,
        k: usize,
        pad_a: usize,
        pad_c: usize,
        alpha: f64,
        beta: f64,
    ) {
        let (arows, lda) = match trans {
            Transpose::NoTrans => (n, k.max(1) + pad_a),
            _ => (k, n.max(1) + pad_a),
        };
        let ldc = n.max(1) + pad_c;
        let a: Vec<f64> = (0..arows.max(1) * lda).map(prng).collect();
        let c0: Vec<f64> = (0..n.max(1) * ldc).map(|t| prng(100_000 + t)).collect();
        let mut got = c0.clone();
        syrk_f64(uplo, trans, n, k, alpha, &a, lda, beta, &mut got, ldc);
        let mut want = c0;
        ref_syrk_f64(uplo, trans, n, k, alpha, &a, lda, beta, &mut want, ldc);
        // Valid triangle only; padding and the opposite triangle are
        // untouched by contract (and equal on both sides by construction).
        for i in 0..n {
            for j in 0..n {
                let inside = match uplo {
                    TrsmUplo::Lower => j <= i,
                    TrsmUplo::Upper => j >= i,
                };
                if !inside {
                    continue;
                }
                assert!(
                    (got[i * ldc + j] - want[i * ldc + j]).abs()
                        <= 1e-9 * want[i * ldc + j].abs().max(1.0),
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn syrk_band_wide_heap_stash_matches_reference() {
        // Wide serial bands (`nbi > SYRK_NB`) take the heap stash;
        // narrow bands keep the stack array. Drive `syrk_band`
        // directly (serial GEMMs, no threads) over all bands and
        // compare owned rows against the naive reference, plus
        // bitwise preservation of the untouched triangle (the
        // stash/restore contract around the merged diagonal GEMM).
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            for trans in [Transpose::NoTrans, Transpose::Trans] {
                for nb in [32usize, 33, 256] {
                    for beta in [0.0, 0.5, 1.0] {
                        let (n, k) = (300usize, 48usize);
                        let notrans = matches!(trans, Transpose::NoTrans);
                        let (arows, lda) = if notrans { (n, k) } else { (k, n) };
                        let ldc = n + 3;
                        let a: Vec<f64> = (0..arows * lda).map(prng).collect();
                        let c0: Vec<f64> = (0..n * ldc).map(|t| prng(777 + t)).collect();
                        let mut got = c0.clone();
                        let bands = n.div_ceil(nb);
                        for bi in 0..bands {
                            syrk_band(
                                uplo,
                                notrans,
                                false,
                                bi,
                                nb,
                                n,
                                k,
                                2.5,
                                beta,
                                a.as_ptr(),
                                lda,
                                got.as_mut_ptr(),
                                ldc,
                                crate::cpu::blas3::gemm_f64_serial,
                                crate::cpu::blas3::gemm_f64_serial,
                                false,
                                usize::MAX,
                            );
                        }
                        let mut want = c0.clone();
                        ref_syrk_f64(uplo, trans, n, k, 2.5, &a, lda, beta, &mut want, ldc);
                        for i in 0..n {
                            for j in 0..n {
                                let inside = match uplo {
                                    TrsmUplo::Lower => j <= i,
                                    TrsmUplo::Upper => j >= i,
                                };
                                if inside {
                                    assert!(
                                        (got[i * ldc + j] - want[i * ldc + j]).abs()
                                            <= 1e-9 * want[i * ldc + j].abs().max(1.0),
                                        "mismatch uplo={uplo:?} trans={trans:?} nb={nb} beta={beta} at ({i},{j})"
                                    );
                                } else {
                                    assert!(
                                        got[i * ldc + j].to_bits() == c0[i * ldc + j].to_bits(),
                                        "untouched triangle clobbered uplo={uplo:?} trans={trans:?} nb={nb} beta={beta} at ({i},{j})"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn syrk_f64_all_variants() {
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            for trans in [Transpose::NoTrans, Transpose::Trans] {
                for (alpha, beta) in [(1.0, 0.0), (2.5, 0.5), (0.0, 1.0), (1.0, 2.0)] {
                    for (n, k, pa, pc) in [
                        (1, 1, 0, 0),
                        (5, 3, 0, 0),
                        (3, 5, 2, 1),
                        (17, 16, 0, 0),
                        (16, 17, 3, 2),
                        (70, 65, 0, 0),
                        (65, 70, 1, 1),
                        (130, 129, 0, 0),
                        (33, 200, 0, 0),
                    ] {
                        check_f64(uplo, trans, n, k, pa, pc, alpha, beta);
                    }
                }
            }
        }
    }

    #[test]
    fn syrk_beta_triage_large_and_degenerate() {
        // Fused per-band triage (main path at many-band scale) and the
        // early-path triage (`k == 0` / `alpha == 0`, parallel at `n >=
        // SYRK_TRIAGE_PAR_MIN_N`): the variants test tops out at n=130
        // and pairs `alpha == 0` only with the `beta == 1` no-op, so
        // neither triage shape is covered there.
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            for trans in [Transpose::NoTrans, Transpose::Trans] {
                // Main path at parallel scale (fused per-band triage),
                // tight and padded strides.
                for (alpha, beta) in [(1.0, 0.0), (2.5, 0.5), (1.0, 2.0)] {
                    check_f64(uplo, trans, 600, 48, 0, 0, alpha, beta);
                    check_f64(uplo, trans, 517, 48, 1, 1, alpha, beta);
                }
                // Early path with real triage (serial and parallel).
                for (n, k, alpha, beta) in [
                    (600, 0, 1.0, 0.0),
                    (600, 0, 1.0, 2.0),
                    (600, 48, 0.0, 0.0),
                    (600, 48, 0.0, 2.0),
                    (517, 0, 2.5, 0.5),
                    (33, 0, 1.0, 2.0),
                    (33, 48, 0.0, 0.0),
                ] {
                    check_f64(uplo, trans, n, k, 0, 0, alpha, beta);
                }
            }
        }
    }

    #[test]
    fn syrk_complex_spot_check() {
        // c64/c32 share the generic core; spot-check symmetric NoTrans and
        // Hermitian ConjTrans against explicit references.
        for trans in [Transpose::NoTrans, Transpose::ConjTrans] {
            let (n, k) = (40, 37);
            let (arows, lda) = match trans {
                Transpose::NoTrans => (n, k + 2),
                _ => (k, n + 2),
            };
            let ldc = n + 3;
            let a: Vec<Complex64> = (0..arows * lda)
                .map(|t| Complex64::new(prng(t), prng(50_000 + t)))
                .collect();
            let c0: Vec<Complex64> = (0..n * ldc)
                .map(|t| Complex64::new(prng(100_000 + t), prng(200_000 + t)))
                .collect();
            let (alpha, beta) = (Complex64::new(1.5, -0.5), Complex64::new(0.5, 0.25));
            let mut got = c0.clone();
            syrk_c64(
                TrsmUplo::Lower,
                trans,
                n,
                k,
                alpha,
                &a,
                lda,
                beta,
                &mut got,
                ldc,
            );
            let conj = trans == Transpose::ConjTrans;
            let tr = !matches!(trans, Transpose::NoTrans);
            // First factor conjugated iff Hermitian; second never is.
            let av = |r: usize, p: usize| -> Complex64 {
                let v = if tr { a[p * lda + r] } else { a[r * lda + p] };
                if conj { v.conj() } else { v }
            };
            let mut want = c0;
            for i in 0..n {
                for j in 0..=i {
                    want[i * ldc + j] = beta * want[i * ldc + j];
                }
            }
            let bv = |r: usize, p: usize| -> Complex64 {
                if tr { a[p * lda + r] } else { a[r * lda + p] }
            };
            for i in 0..n {
                for j in 0..=i {
                    let mut s = Complex64::new(0.0, 0.0);
                    for p in 0..k {
                        s += av(i, p) * bv(j, p);
                    }
                    want[i * ldc + j] += alpha * s;
                }
            }
            let mut err = 0.0;
            let mut norm = 0.0;
            for i in 0..n {
                for j in 0..=i {
                    err += (got[i * ldc + j] - want[i * ldc + j]).norm_sqr();
                    norm += want[i * ldc + j].norm_sqr();
                }
            }
            assert!(
                err.sqrt() <= 1e-12 * norm.sqrt().max(1.0),
                "{trans:?}: {} / {}",
                err.sqrt(),
                norm.sqrt()
            );
        }
        // c32 smoke test.
        {
            let (n, k) = (40, 37);
            let (lda, ldc) = (k + 2, n + 3);
            let a: Vec<Complex32> = (0..n * lda)
                .map(|t| Complex32::new(prng(t) as f32, prng(50_000 + t) as f32))
                .collect();
            let mut c = vec![Complex32::new(0.0, 0.0); n * ldc];
            syrk_c32(
                TrsmUplo::Lower,
                Transpose::NoTrans,
                n,
                k,
                Complex32::new(1.0, 0.0),
                &a,
                lda,
                Complex32::new(0.0, 0.0),
                &mut c,
                ldc,
            );
            assert!(c.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
            // Symmetry of the written triangle: C[i,j] == C[j,i]... only
            // Lower is written; check diagonal real and finite instead.
            for i in 0..n {
                assert!(c[i * ldc + i].re.is_finite());
            }
        }
    }

    #[test]
    fn herk_notrans_matches_reference() {
        // `herk_c64`/`herk_c32` (Hermitian update from a NoTrans `(n × k)`
        // operand) against an explicit `A·Aᴴ` reference, plus agreement
        // with the `(k × n)` ConjTrans spelling on transposed operands.
        let (n, k) = (40, 37);
        let (lda, ldc) = (k + 2, n + 3);
        let a: Vec<Complex64> = (0..n * lda)
            .map(|t| Complex64::new(prng(t), prng(50_000 + t)))
            .collect();
        let c0: Vec<Complex64> = (0..n * ldc)
            .map(|t| Complex64::new(prng(100_000 + t), prng(200_000 + t)))
            .collect();
        let (alpha, beta) = (Complex64::new(1.5, -0.5), Complex64::new(0.5, 0.25));
        let mut got = c0.clone();
        herk_c64(TrsmUplo::Lower, n, k, alpha, &a, lda, beta, &mut got, ldc);
        let mut want = c0.clone();
        for i in 0..n {
            for j in 0..=i {
                let mut s = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    s += a[i * lda + p] * a[j * lda + p].conj();
                }
                want[i * ldc + j] = beta * want[i * ldc + j] + alpha * s;
            }
        }
        let mut err = 0.0;
        let mut norm = 0.0;
        for i in 0..n {
            for j in 0..=i {
                err += (got[i * ldc + j] - want[i * ldc + j]).norm_sqr();
                norm += want[i * ldc + j].norm_sqr();
            }
        }
        assert!(
            err.sqrt() <= 1e-12 * norm.sqrt().max(1.0),
            "{} / {}",
            err.sqrt(),
            norm.sqrt()
        );
        // Same math via the ConjTrans spelling: its `(k × n)` operand
        // must be the conjugate-transpose (`C += Aᴴ·A` conjugates the
        // first factor, while NoTrans-HERK conjugates the second).
        let lda_t = n + 2;
        let mut at = vec![Complex64::new(0.0, 0.0); k * lda_t];
        for i in 0..n {
            for p in 0..k {
                at[p * lda_t + i] = a[i * lda + p].conj();
            }
        }
        let mut got_t = c0.clone();
        syrk_c64(
            TrsmUplo::Lower,
            Transpose::ConjTrans,
            n,
            k,
            alpha,
            &at,
            lda_t,
            beta,
            &mut got_t,
            ldc,
        );
        let mut err_t = 0.0;
        for i in 0..n {
            for j in 0..=i {
                err_t += (got[i * ldc + j] - got_t[i * ldc + j]).norm_sqr();
            }
        }
        assert!(
            err_t.sqrt() <= 1e-12 * norm.sqrt().max(1.0),
            "{}",
            err_t.sqrt()
        );
        // c32 smoke test vs its own reference.
        {
            let ac: Vec<Complex32> = (0..n * lda)
                .map(|t| Complex32::new(prng(t) as f32, prng(50_000 + t) as f32))
                .collect();
            let c0c: Vec<Complex32> = (0..n * ldc)
                .map(|t| Complex32::new(prng(100_000 + t) as f32, prng(200_000 + t) as f32))
                .collect();
            let (ac_alpha, ac_beta) = (Complex32::new(1.5, -0.5), Complex32::new(0.5, 0.25));
            let mut gotc = c0c.clone();
            herk_c32(
                TrsmUplo::Lower,
                n,
                k,
                ac_alpha,
                &ac,
                lda,
                ac_beta,
                &mut gotc,
                ldc,
            );
            let mut errc = 0.0f32;
            let mut normc = 0.0f32;
            for i in 0..n {
                for j in 0..=i {
                    let mut s = Complex32::new(0.0, 0.0);
                    for p in 0..k {
                        s += ac[i * lda + p] * ac[j * lda + p].conj();
                    }
                    let w = ac_beta * c0c[i * ldc + j] + ac_alpha * s;
                    errc += (gotc[i * ldc + j] - w).norm_sqr();
                    normc += w.norm_sqr();
                }
            }
            assert!(
                errc.sqrt() <= 1e-4 * normc.sqrt().max(1.0),
                "{} / {}",
                errc.sqrt(),
                normc.sqrt()
            );
        }
    }

    #[test]
    fn syrk_f32_spot_check() {
        let (n, k) = (70, 65);
        let (lda, ldc) = (n + 1, n + 1);
        // Trans: A is k × n.
        let a: Vec<f32> = (0..k * lda).map(|t| prng(t) as f32).collect();
        let c0: Vec<f32> = (0..n * ldc).map(|t| prng(100_000 + t) as f32).collect();
        let mut got = c0.clone();
        syrk_f32(
            TrsmUplo::Upper,
            Transpose::Trans,
            n,
            k,
            2.0,
            &a,
            lda,
            0.5,
            &mut got,
            ldc,
        );
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let mut want64: Vec<f64> = c0.iter().map(|&v| v as f64).collect();
        ref_syrk_f64(
            TrsmUplo::Upper,
            Transpose::Trans,
            n,
            k,
            2.0,
            &a64,
            lda,
            0.5,
            &mut want64,
            ldc,
        );
        for i in 0..n {
            for j in i..n {
                assert!(
                    (got[i * ldc + j] as f64 - want64[i * ldc + j]).abs()
                        <= 1e-4 * want64[i * ldc + j].abs().max(1.0),
                    "f32 mismatch at ({i},{j})"
                );
            }
        }
    }

    /// Team-slice equality: the beta pre-scale plus serial band slices
    /// reproduce the whole SYRK call bitwise (the safety property the
    /// Cholesky team loop relies on). Covers `NoTrans`/`Trans`, `beta`
    /// 1 and 0.5 (the scale path), `f64`/`f32`, and a nesting-exercising
    /// large shape alongside a small one.
    #[test]
    fn team_slices_match_whole_syrk() {
        for (n, k) in [(200usize, 64usize), (1984usize, 64usize)] {
            for trans in [Transpose::NoTrans, Transpose::Trans] {
                let (a_rows, a_cols, lda) = match trans {
                    Transpose::NoTrans => (n, k, k),
                    _ => (k, n, n),
                };
                let a: Vec<f64> = (0..a_rows * a_cols)
                    .map(|i| 0.01 * ((i % 97) as f64))
                    .collect();
                for beta in [1.0f64, 0.5f64] {
                    let c0 = vec![1.0f64; n * n];
                    let mut whole = c0.clone();
                    syrk_f64(
                        TrsmUplo::Lower,
                        trans,
                        n,
                        k,
                        -1.0,
                        &a,
                        lda,
                        beta,
                        &mut whole,
                        n,
                    );
                    let (bands, nb) = syrk_band_plan(n, k);
                    let (nest_outer, nest_macs) = syrk_band_nest(bands);
                    let mut sliced = c0.clone();
                    syrk_beta_scale(TrsmUplo::Lower, n, beta, &mut sliced, n);
                    for bi in 0..bands {
                        // SAFETY: bands own disjoint `C` rows (in bounds by
                        // the plan's split); `a` read-only shared.
                        unsafe {
                            syrk_band_f64(
                                TrsmUplo::Lower,
                                trans,
                                bi,
                                nb,
                                n,
                                k,
                                -1.0,
                                a.as_ptr(),
                                lda,
                                sliced.as_mut_ptr(),
                                n,
                                nest_outer,
                                nest_macs,
                            );
                        }
                    }
                    assert_eq!(whole, sliced, "f64 n={n} {trans:?} beta={beta}");
                }
            }
        }
        // f32 smoke (in-situ single-precision shape).
        {
            let (n, k) = (960usize, 64usize);
            let a: Vec<f32> = (0..n * k).map(|i| 0.01 * ((i % 97) as f32)).collect();
            let c0 = vec![1.0f32; n * n];
            let mut whole = c0.clone();
            syrk_f32(
                TrsmUplo::Lower,
                Transpose::NoTrans,
                n,
                k,
                -1.0,
                &a,
                k,
                1.0,
                &mut whole,
                n,
            );
            let (bands, nb) = syrk_band_plan(n, k);
            let (nest_outer, nest_macs) = syrk_band_nest(bands);
            let mut sliced = c0.clone();
            syrk_beta_scale(TrsmUplo::Lower, n, 1.0f32, &mut sliced, n);
            for bi in 0..bands {
                // SAFETY: as above.
                unsafe {
                    syrk_band_f32(
                        TrsmUplo::Lower,
                        Transpose::NoTrans,
                        bi,
                        nb,
                        n,
                        k,
                        -1.0,
                        a.as_ptr(),
                        k,
                        sliced.as_mut_ptr(),
                        n,
                        nest_outer,
                        nest_macs,
                    );
                }
            }
            assert_eq!(whole, sliced, "f32 n={n}");
        }
    }
}
