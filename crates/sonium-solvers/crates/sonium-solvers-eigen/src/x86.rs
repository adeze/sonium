//! x86_64 SIMD micro-kernels for the blocked Householder panel.
//!
//! The panel factorisation ([`crate::lobpcg::householder_tridiag_blocked`])
//! is level-2 work LLVM will not auto-vectorize (runtime trip counts), so
//! the long reductions go through explicit AVX2+FMA kernels here. Same
//! convention as `sonium-fastmath`: `unsafe` `target_feature` kernels with
//! the CPU dispatch (runtime detection) as the caller's responsibility —
//! except the dispatch lives in [`dot`]/[`axpy`] themselves, so panel code
//! stays branch-free. Non-x86_64 builds compile the scalar fallbacks only.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Dot product over contiguous slices, AVX2+FMA when available.
///
/// Lengths below [`DOT_AVX2_MIN`] stay scalar: the vector kernel's
/// horizontal-reduction latency exceeds its throughput win there (measured
/// crossover on Zen 2 well above the 8-wide loop minimum).
const DOT_AVX2_MIN: usize = 64;

/// Dot product over contiguous slices, AVX2+FMA when available.
#[inline]
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= DOT_AVX2_MIN && has_avx2_fma() {
        // SAFETY: lengths are equal (debug-checked in the fallback and by
        // every panel call site); AVX2+FMA presence was just detected.
        return unsafe { dot_avx2(x, y) };
    }
    dot_scalar(x, y)
}

/// Dot product with 4 independent accumulators: exposes ILP/FMA on long
/// rows where the auto-vectorizer stays out. Reassociated summation differs
/// from a naive scalar loop by a few ulps at most; all consumers here are
/// verified by the tridiagonal reconstruction and eigenvalue-agreement
/// tests.
#[inline(always)]
pub(crate) fn dot_scalar(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut p = 0;
    while p + 4 <= len {
        s0 += x[p] * y[p];
        s1 += x[p + 1] * y[p + 1];
        s2 += x[p + 2] * y[p + 2];
        s3 += x[p + 3] * y[p + 3];
        p += 4;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    while p < len {
        s += x[p] * y[p];
        p += 1;
    }
    s
}

/// Cached x86_64 AVX2+FMA availability.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Minimum `t_` trip for the AVX2 mirror kernel: below it the horizontal
/// reduction exceeds the throughput win (same spirit as [`DOT_AVX2_MIN`]).
const MIRROR_AVX2_MIN_J: usize = 8;

/// Four-row mirror GEMV: `out[i] = Σ_{t_<j} mirror[t_·n + r0 + i]·u[t_·st]`
/// for `i` in `0..4`.
///
/// Covers the panel's strided-mirror reductions — the `dlatrd` column
/// update (`V·W[c]ᵀ + W·V[c]ᵀ`) and the W-column correction (`V·wrow`) —
/// which a scalar loop would stream at ~1 element per cycle. Mirror rows
/// are contiguous 4-packs; `u` is read through a strided broadcast. The
/// scalar fallback accumulates `t_` ascending per row, exactly matching the
/// panel loops it replaces; the AVX2 path splits even/odd `t_` (like
/// [`dot_avx2`'s two accumulators), so results differ by reassociation
/// ulps — panel consumers are backward-stable under that (same convention
/// as [`dot_scalar`]).
#[inline]
pub(crate) fn mirror_gemv4(
    mirror: &[f64],
    n: usize,
    j: usize,
    r0: usize,
    u: &[f64],
    ustride: usize,
) -> [f64; 4] {
    #[cfg(target_arch = "x86_64")]
    if j >= MIRROR_AVX2_MIN_J && has_avx2_fma() {
        // SAFETY: bounds debug-checked in the fallback; AVX2+FMA presence
        // was just detected.
        return unsafe { mirror_gemv4_avx2(mirror, n, j, r0, u, ustride) };
    }
    let mut out = [0.0f64; 4];
    debug_assert!(j == 0 || (j - 1) * n + r0 + 4 <= mirror.len());
    for t_ in 0..j {
        let mrow = t_ * n + r0;
        let ut = u[t_ * ustride];
        for i in 0..4 {
            out[i] += mirror[mrow + i] * ut;
        }
    }
    out
}

/// 4-row strided-mirror AVX2+FMA GEMV (see [`mirror_gemv4`]).
///
/// # Safety
///
/// - x86_64 with AVX2 and FMA (see [`has_avx2_fma`]); `target_feature`
///   function, no in-function check.
/// - `mirror` must hold `j` columns of stride `n` with rows `r0..r0+4`
///   readable; `u` must hold `j` elements at stride `ustride`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn mirror_gemv4_avx2(
    mirror: &[f64],
    n: usize,
    j: usize,
    r0: usize,
    u: &[f64],
    ustride: usize,
) -> [f64; 4] {
    // SAFETY: `target_feature = "avx2,fma"` body; unaligned mirror loads
    // (no alignment); strided `u` through 8-byte broadcast loads only.
    unsafe {
        debug_assert!(j == 0 || (j - 1) * n + r0 + 4 <= mirror.len());
        let mp = mirror.as_ptr();
        let up = u.as_ptr();
        // Even/odd accumulators keep the 4–5 cycle FMA latency off the
        // critical path (each accumulator's updates are 2 FMAs apart).
        let mut acc_e = _mm256_setzero_pd();
        let mut acc_o = _mm256_setzero_pd();
        let mut t_ = 0;
        while t_ + 2 <= j {
            let me = _mm256_loadu_pd(mp.add(t_ * n + r0));
            acc_e = _mm256_fmadd_pd(me, _mm256_broadcast_sd(&*up.add(t_ * ustride)), acc_e);
            let mo = _mm256_loadu_pd(mp.add((t_ + 1) * n + r0));
            acc_o = _mm256_fmadd_pd(mo, _mm256_broadcast_sd(&*up.add((t_ + 1) * ustride)), acc_o);
            t_ += 2;
        }
        if t_ < j {
            let m = _mm256_loadu_pd(mp.add(t_ * n + r0));
            acc_e = _mm256_fmadd_pd(m, _mm256_broadcast_sd(&*up.add(t_ * ustride)), acc_e);
        }
        // Lanes are four INDEPENDENT row sums: extract each directly,
        // never added across rows (there is no horizontal reduction here).
        let sum = _mm256_add_pd(acc_e, acc_o);
        let hi = _mm256_extractf128_pd(sum, 1);
        let lo = _mm256_castpd256_pd128(sum);
        let mut out = [0.0f64; 4];
        out[0] = _mm_cvtsd_f64(lo);
        out[1] = _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
        out[2] = _mm_cvtsd_f64(hi);
        out[3] = _mm_cvtsd_f64(_mm_unpackhi_pd(hi, hi));
        out
    }
}

/// Four-row strided dots sharing one vector: `out[o0 + i·ostride] =
/// Σ_k a[(r0+i)·n + k0 + k]·v[k]` for `i` in `0..4`, `k` in `0..len`.
///
/// Covers the panel's `dlatrd` W-column symmetric matvec (one row-dot per
/// trailing row sharing `vcol`): fusing 4 rows amortizes dispatch over 4x
/// fewer calls and keeps one vector in L1 across 4 row streams, while the
/// AVX2 path hides FMA latency across 8 accumulators (2 per row, like
/// [`dot_avx2`]). Reassociation vs per-row [`dot_scalar`] differs by ulps
/// (panel-accepted convention). Out-of-range indices panic in every build
/// mode (use scalar [`dot`] for the tail).
///
/// # Panics
///
/// Panics when rows `r0..r0+4`, the `len` segment at `k0`, `v[..len]`, or
/// any of the four strided outputs falls outside its slice.
#[inline]
pub(crate) fn rowdots4(
    a: &[f64],
    n: usize,
    r0: usize,
    k0: usize,
    len: usize,
    v: &[f64],
    out: &mut [f64],
    o0: usize,
    ostride: usize,
) {
    assert!(
        rowdots4_bounds_ok(a.len(), n, r0, k0, len, v.len(), out.len(), o0, ostride),
        "rowdots4 index out of bounds: a.len={} n={n} r0={r0} k0={k0} len={len} v.len={} out.len={} o0={o0} ostride={ostride}",
        a.len(),
        v.len(),
        out.len(),
    );
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: bounds release-checked above; AVX2+FMA presence was
        // just detected.
        return unsafe { rowdots4_avx2(a, n, r0, k0, len, v, out, o0, ostride) };
    }
    rowdots4_scalar(a, n, r0, k0, len, v, out, o0, ostride);
}

/// Bounds contract for [`rowdots4`]: rows `r0..r0+4` each holding `len`
/// elements at offset `k0` inside `a` (row stride `n`), `len` elements
/// inside `v`, and the four strided outputs inside `out`. Checked
/// arithmetic throughout so adversarial indices fail instead of wrapping.
fn rowdots4_bounds_ok(
    a_len: usize,
    n: usize,
    r0: usize,
    k0: usize,
    len: usize,
    v_len: usize,
    out_len: usize,
    o0: usize,
    ostride: usize,
) -> bool {
    if v_len < len {
        return false;
    }
    for i in 0..4usize {
        let end = r0
            .checked_add(i)
            .and_then(|row| row.checked_mul(n))
            .and_then(|base| base.checked_add(k0))
            .and_then(|start| start.checked_add(len));
        match end {
            Some(end) if end <= a_len => {}
            _ => return false,
        }
        let o = i.checked_mul(ostride).and_then(|off| o0.checked_add(off));
        match o {
            Some(o) if o < out_len => {}
            _ => return false,
        }
    }
    true
}

/// Scalar [`rowdots4`] fallback: one [`dot_scalar`] per row, ascending.
/// Callers must uphold the [`rowdots4`] bounds contract (release-checked
/// by the wrapper); extracted so tests can pin the scalar path directly
/// on AVX2 hosts, where the wrapper always dispatches to the kernel.
fn rowdots4_scalar(
    a: &[f64],
    n: usize,
    r0: usize,
    k0: usize,
    len: usize,
    v: &[f64],
    out: &mut [f64],
    o0: usize,
    ostride: usize,
) {
    for i in 0..4 {
        let row = &a[(r0 + i) * n + k0..(r0 + i) * n + k0 + len];
        out[o0 + i * ostride] = dot_scalar(row, &v[..len]);
    }
}

/// 4-row strided-dots AVX2+FMA kernel (see [`rowdots4`]).
///
/// # Safety
///
/// - x86_64 with AVX2 and FMA (see [`has_avx2_fma`]); `target_feature`
///   function, no in-function check.
/// - `a` must hold rows `r0..r0+4` with `len` elements at offset `k0`
///   (row stride `n`); `v` must hold `len` elements; `out` must hold
///   `o0 + i·ostride` for `i` in `0..4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn rowdots4_avx2(
    a: &[f64],
    n: usize,
    r0: usize,
    k0: usize,
    len: usize,
    v: &[f64],
    out: &mut [f64],
    o0: usize,
    ostride: usize,
) {
    // SAFETY: body runs inside a `target_feature = "avx2,fma"` function;
    // unaligned loads throughout (panel rows carry no alignment).
    unsafe {
        debug_assert!(v.len() >= len);
        let ap = a.as_ptr();
        let vp = v.as_ptr();
        let op = out.as_mut_ptr();
        // Prefetch the next row-group's A starts: each group is a fresh
        // stream set for the L2 streamer, so hide the ramp behind this
        // group's FMAs. x86 prefetch never faults, but forming the address
        // with `add` past the object end is still UB, so the offset uses
        // wrapping arithmetic: on the last group the address is arbitrary
        // yet well-defined, and prefetch never dereferences it.
        for i in 0..4 {
            let off = r0
                .wrapping_add(4)
                .wrapping_add(i)
                .wrapping_mul(n)
                .wrapping_add(k0);
            _mm_prefetch(ap.wrapping_add(off) as *const i8, _MM_HINT_T0);
        }
        let r1 = ap.add((r0 + 1) * n + k0);
        let r2 = ap.add((r0 + 2) * n + k0);
        let r3 = ap.add((r0 + 3) * n + k0);
        let r0p = ap.add(r0 * n + k0);
        // 8 accumulators (even/odd k per row): each accumulator's updates
        // are 8 FMAs apart, clearing the 4-5 cycle FMA latency.
        let mut a0e = _mm256_setzero_pd();
        let mut a0o = _mm256_setzero_pd();
        let mut a1e = _mm256_setzero_pd();
        let mut a1o = _mm256_setzero_pd();
        let mut a2e = _mm256_setzero_pd();
        let mut a2o = _mm256_setzero_pd();
        let mut a3e = _mm256_setzero_pd();
        let mut a3o = _mm256_setzero_pd();
        let mut p = 0;
        while p + 8 <= len {
            let ve = _mm256_loadu_pd(vp.add(p));
            let vo = _mm256_loadu_pd(vp.add(p + 4));
            a0e = _mm256_fmadd_pd(_mm256_loadu_pd(r0p.add(p)), ve, a0e);
            a1e = _mm256_fmadd_pd(_mm256_loadu_pd(r1.add(p)), ve, a1e);
            a2e = _mm256_fmadd_pd(_mm256_loadu_pd(r2.add(p)), ve, a2e);
            a3e = _mm256_fmadd_pd(_mm256_loadu_pd(r3.add(p)), ve, a3e);
            a0o = _mm256_fmadd_pd(_mm256_loadu_pd(r0p.add(p + 4)), vo, a0o);
            a1o = _mm256_fmadd_pd(_mm256_loadu_pd(r1.add(p + 4)), vo, a1o);
            a2o = _mm256_fmadd_pd(_mm256_loadu_pd(r2.add(p + 4)), vo, a2o);
            a3o = _mm256_fmadd_pd(_mm256_loadu_pd(r3.add(p + 4)), vo, a3o);
            p += 8;
        }
        // Horizontal reduce per row (inlined 4x: closures do not inherit
        // `target_feature`), then the scalar tail in k-ascending order.
        let mut s = [0.0f64; 4];
        let accs = [(a0e, a0o), (a1e, a1o), (a2e, a2o), (a3e, a3o)];
        for (i, (e, o)) in accs.iter().enumerate() {
            let sum = _mm256_add_pd(*e, *o);
            let hi = _mm256_extractf128_pd(sum, 1);
            let lo = _mm256_castpd256_pd128(sum);
            let s128 = _mm_add_pd(hi, lo);
            s[i] = _mm_cvtsd_f64(_mm_add_sd(s128, _mm_unpackhi_pd(s128, s128)));
        }
        while p < len {
            let vv = *vp.add(p);
            s[0] += *r0p.add(p) * vv;
            s[1] += *r1.add(p) * vv;
            s[2] += *r2.add(p) * vv;
            s[3] += *r3.add(p) * vv;
            p += 1;
        }
        *op.add(o0) = s[0];
        *op.add(o0 + ostride) = s[1];
        *op.add(o0 + 2 * ostride) = s[2];
        *op.add(o0 + 3 * ostride) = s[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xBF58476D1CE4E5B9);
        let z = *seed;
        ((z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9) >> 11) as f64 / (u64::MAX as f64) * 2.0
            - 1.0
    }

    /// (n, r0, k0, len, ostride): wide AVX2 loops, k-tails around the
    /// 8-wide boundary, minimal/last groups, and both production strides
    /// (1 for the corr/larft sites, kb for the panel site).
    const ROWDOTS4_CASES: &[(usize, usize, usize, usize, usize)] = &[
        (300, 0, 0, 300, 32),
        (300, 1, 1, 299, 32),
        (300, 150, 151, 149, 32),
        (300, 296, 200, 100, 32),
        (300, 290, 295, 5, 32),
        (17, 3, 2, 15, 1),
        (9, 5, 0, 9, 1),
        (64, 60, 60, 4, 1),
        (4, 0, 0, 4, 1),
        (5, 1, 2, 3, 1),
        (9, 5, 0, 0, 1),
        (9, 5, 0, 1, 1),
        (9, 5, 0, 7, 1),
        (16, 12, 0, 8, 1),
        (16, 12, 0, 9, 1),
        (32, 28, 3, 16, 32),
        (32, 28, 3, 17, 32),
    ];

    fn check_rowdots4_case(
        seed: &mut u64,
        n: usize,
        r0: usize,
        k0: usize,
        len: usize,
        ostride: usize,
        write: impl Fn(&[f64], &[f64], &mut [f64]),
    ) {
        let a: Vec<f64> = (0..n * n).map(|_| splitmix(seed)).collect();
        let v: Vec<f64> = (0..n).map(|_| splitmix(seed)).collect();
        let mut got = vec![0.0f64; (n + 4) * ostride.max(1) + 8];
        let o0 = r0 * ostride + 3;
        write(&a, &v, &mut got);
        for i in 0..4 {
            let row = &a[(r0 + i) * n + k0..(r0 + i) * n + k0 + len];
            let want = dot_scalar(row, &v[k0..k0 + len]);
            let tol = 1e-12 * want.abs().max(1.0);
            assert!(
                (got[o0 + i * ostride] - want).abs() <= tol,
                "n={n} r0={r0} k0={k0} len={len} ostride={ostride} row {i}: got {} want {}",
                got[o0 + i * ostride],
                want
            );
        }
    }

    #[test]
    fn rowdots4_matches_per_row_dots() {
        // Reference: plain per-row dot_scalar (the panel's old order).
        let mut seed = 0xCAFE_F00D_DEAD_BEEF;
        for &(n, r0, k0, len, ostride) in ROWDOTS4_CASES {
            check_rowdots4_case(&mut seed, n, r0, k0, len, ostride, |a, v, got| {
                let o0 = r0 * ostride + 3;
                rowdots4(a, n, r0, k0, len, &v[k0..k0 + len], got, o0, ostride);
            });
        }
    }

    #[test]
    fn rowdots4_scalar_matches_per_row_dots() {
        // Pins the scalar path directly: on AVX2 hosts the wrapper never
        // takes it, so dispatch coverage alone would leave it untested.
        let mut seed = 0x5CA1_AB1E_0123_4567;
        for &(n, r0, k0, len, ostride) in ROWDOTS4_CASES {
            check_rowdots4_case(&mut seed, n, r0, k0, len, ostride, |a, v, got| {
                let o0 = r0 * ostride + 3;
                rowdots4_scalar(a, n, r0, k0, len, &v[k0..k0 + len], got, o0, ostride);
            });
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rowdots4_avx2_matches_scalar() {
        if !has_avx2_fma() {
            return;
        }
        // Direct kernel agreement, including last-group calls whose
        // next-group prefetch runs past the matrix end.
        let mut seed = 0xA9E2_4C7B_1D0F_9E31;
        for &(n, r0, k0, len, ostride) in ROWDOTS4_CASES {
            let a: Vec<f64> = (0..n * n).map(|_| splitmix(&mut seed)).collect();
            let v: Vec<f64> = (0..n).map(|_| splitmix(&mut seed)).collect();
            let mut want = vec![0.0f64; (n + 4) * ostride.max(1) + 8];
            let mut got = want.clone();
            let o0 = r0 * ostride + 3;
            rowdots4_scalar(&a, n, r0, k0, len, &v[k0..k0 + len], &mut want, o0, ostride);
            // SAFETY: same in-bounds inputs the scalar call just consumed.
            unsafe {
                rowdots4_avx2(&a, n, r0, k0, len, &v[k0..k0 + len], &mut got, o0, ostride);
            }
            for i in 0..4 {
                let (g, w) = (got[o0 + i * ostride], want[o0 + i * ostride]);
                let tol = 1e-12 * w.abs().max(1.0);
                assert!(
                    (g - w).abs() <= tol,
                    "n={n} r0={r0} k0={k0} len={len} ostride={ostride} row {i}: avx2 {g} scalar {w}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "rowdots4 index out of bounds")]
    fn rowdots4_rejects_short_rows() {
        let a = vec![0.0f64; 4 * 8];
        let v = vec![0.0f64; 8];
        let mut out = vec![0.0f64; 8];
        // Rows r0..r0+4 = 1..5 exceed the 4-row matrix.
        rowdots4(&a, 8, 1, 0, 8, &v, &mut out, 0, 1);
    }

    #[test]
    #[should_panic(expected = "rowdots4 index out of bounds")]
    fn rowdots4_rejects_short_vector() {
        let a = vec![0.0f64; 4 * 8];
        let v = vec![0.0f64; 7];
        let mut out = vec![0.0f64; 8];
        rowdots4(&a, 8, 0, 0, 8, &v, &mut out, 0, 1);
    }

    #[test]
    #[should_panic(expected = "rowdots4 index out of bounds")]
    fn rowdots4_rejects_short_output() {
        let a = vec![0.0f64; 4 * 8];
        let v = vec![0.0f64; 8];
        let mut out = vec![0.0f64; 3];
        rowdots4(&a, 8, 0, 0, 8, &v, &mut out, 0, 1);
    }

    #[test]
    fn mirror_gemv4_matches_scalar_order() {
        // Reference: plain t_-ascending accumulation per row.
        let reference =
            |mirror: &[f64], n: usize, j: usize, r0: usize, u: &[f64], st: usize| -> [f64; 4] {
                let mut out = [0.0f64; 4];
                for t_ in 0..j {
                    for i in 0..4 {
                        out[i] += mirror[t_ * n + r0 + i] * u[t_ * st];
                    }
                }
                out
            };
        let mut seed = 0x1234_5678_9ABC_DEF0;
        for &(n, j, r0, st) in &[
            (300usize, 0usize, 0usize, 1usize),
            (300, 1, 5, 300),
            (300, 7, 11, 300),
            (300, 8, 13, 1),
            (300, 9, 17, 300),
            (300, 31, 23, 1),
            (300, 32, 296 - 4, 300),
            (150, 15, 100, 150),
        ] {
            let mirror: Vec<f64> = (0..32 * n).map(|_| splitmix(&mut seed)).collect();
            let ufull: Vec<f64> = (0..32 * st).map(|_| splitmix(&mut seed)).collect();
            let got = mirror_gemv4(&mirror, n, j, r0, &ufull, st);
            let want = reference(&mirror, n, j, r0, &ufull, st);
            for i in 0..4 {
                let tol = 1e-12 * want[i].abs().max(1.0);
                assert!(
                    (got[i] - want[i]).abs() <= tol,
                    "n={n} j={j} r0={r0} st={st} lane {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}

/// 8-wide AVX2+FMA dot over contiguous slices.
///
/// # Safety
///
/// - The caller must be running on x86_64 with both AVX2 and FMA (see
///   [`has_avx2_fma`]); this function carries the `target_feature`
///   attribute but performs no in-function check.
/// - `x` and `y` must have equal lengths; out-of-range lanes are never
///   touched (the tail is scalar), but the 8-wide main loop reads both
///   slices in bounds only under equal lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_avx2(x: &[f64], y: &[f64]) -> f64 {
    // SAFETY: body runs inside a `target_feature = "avx2,fma"` function;
    // every intrinsic is sound under the caller's contract. Unaligned
    // loads/stores throughout (panel rows/columns carry no alignment).
    unsafe {
        let len = x.len().min(y.len());
        let xp = x.as_ptr();
        let yp = y.as_ptr();
        // NOTE: a 4-accumulator 16-wide variant was tried: no measurable
        // gain at n ≤ 300 (rows are L1/L2-hit, so extra latency hiding is
        // idle), so the leaner 2-accumulator form stands.
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut p = 0;
        while p + 8 <= len {
            let a0 = _mm256_loadu_pd(xp.add(p));
            let b0 = _mm256_loadu_pd(yp.add(p));
            acc0 = _mm256_fmadd_pd(a0, b0, acc0);
            let a1 = _mm256_loadu_pd(xp.add(p + 4));
            let b1 = _mm256_loadu_pd(yp.add(p + 4));
            acc1 = _mm256_fmadd_pd(a1, b1, acc1);
            p += 8;
        }
        let sum = _mm256_add_pd(acc0, acc1);
        let hi = _mm256_extractf128_pd(sum, 1);
        let lo = _mm256_castpd256_pd128(sum);
        let s128 = _mm_add_pd(hi, lo);
        let mut s = _mm_cvtsd_f64(_mm_add_sd(s128, _mm_unpackhi_pd(s128, s128)));
        while p < len {
            s += *xp.add(p) * *yp.add(p);
            p += 1;
        }
        s
    }
}
