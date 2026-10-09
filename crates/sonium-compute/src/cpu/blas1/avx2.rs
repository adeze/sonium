//! x86_64 AVX2/FMA BLAS-1 kernels.

#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]

use core::arch::x86_64::*;
use num_complex::{Complex32, Complex64};

#[inline]
pub fn is_available() -> bool {
    !crate::cpu::force_scalar_dispatch()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    let va = _mm256_set1_ps(alpha);
    while i + 32 <= n {
        for lane in 0..4 {
            let offset = i + lane * 8;
            let xv = _mm256_loadu_ps(x.as_ptr().add(offset));
            let yv = _mm256_loadu_ps(y.as_ptr().add(offset));
            _mm256_storeu_ps(y.as_mut_ptr().add(offset), _mm256_fmadd_ps(va, xv, yv));
        }
        i += 32;
    }
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = _mm256_loadu_ps(y.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(va, xv, yv));
        i += 8;
    }
    // FMA tail (not mul+add): bitwise-uniform with the vector body, so
    // results don't depend on chunk boundaries (MT band sizes).
    while i < n {
        y[i] = alpha.mul_add(x[i], y[i]);
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    let va = _mm256_set1_pd(alpha);
    while i + 16 <= n {
        for lane in 0..4 {
            let offset = i + lane * 4;
            let xv = _mm256_loadu_pd(x.as_ptr().add(offset));
            let yv = _mm256_loadu_pd(y.as_ptr().add(offset));
            _mm256_storeu_pd(y.as_mut_ptr().add(offset), _mm256_fmadd_pd(va, xv, yv));
        }
        i += 16;
    }
    while i + 4 <= n {
        let xv = _mm256_loadu_pd(x.as_ptr().add(i));
        let yv = _mm256_loadu_pd(y.as_ptr().add(i));
        _mm256_storeu_pd(y.as_mut_ptr().add(i), _mm256_fmadd_pd(va, xv, yv));
        i += 4;
    }
    // FMA tail (not mul+add): bitwise-uniform with the vector body, so
    // results don't depend on chunk boundaries (MT band sizes).
    while i < n {
        y[i] = alpha.mul_add(x[i], y[i]);
        i += 1;
    }
}

/// Four-row fused fan-out `y[i] += c0*r0[i] + c1*r1[i] + c2*r2[i] +
/// c3*r3[i]` (f64): the serial trans-GEMV inner kernel. One axpy per
/// row is store-bound (one 256-bit y-store per FMA group) while the
/// vendor fuses rows; holding `y` in registers across 4 rows quarters
/// y-load/store traffic at the same A-traffic (measured ST
/// 2026-09: 1.5-2x over the axpy loop on square trans shapes).
/// Fully unrolled rows (a runtime-`k` loop version ties the axpy loop
/// — per-row broadcast/loop overhead eats the fusion gain).
/// Bitwise identical to 4 sequential `axpy` calls: same FMA chain per
/// element in the same row order.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fanout_4row_f64(
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    r0: &[f64],
    r1: &[f64],
    r2: &[f64],
    r3: &[f64],
    y: &mut [f64],
) {
    debug_assert_eq!(r0.len(), y.len());
    debug_assert_eq!(r1.len(), y.len());
    debug_assert_eq!(r2.len(), y.len());
    debug_assert_eq!(r3.len(), y.len());
    let n = y.len();
    let v0 = _mm256_set1_pd(c0);
    let v1 = _mm256_set1_pd(c1);
    let v2 = _mm256_set1_pd(c2);
    let v3 = _mm256_set1_pd(c3);
    let mut i = 0;
    while i + 16 <= n {
        let mut a0 = _mm256_loadu_pd(y.as_ptr().add(i));
        let mut a1 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
        let mut a2 = _mm256_loadu_pd(y.as_ptr().add(i + 8));
        let mut a3 = _mm256_loadu_pd(y.as_ptr().add(i + 12));
        a0 = _mm256_fmadd_pd(v0, _mm256_loadu_pd(r0.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_pd(v0, _mm256_loadu_pd(r0.as_ptr().add(i + 4)), a1);
        a2 = _mm256_fmadd_pd(v0, _mm256_loadu_pd(r0.as_ptr().add(i + 8)), a2);
        a3 = _mm256_fmadd_pd(v0, _mm256_loadu_pd(r0.as_ptr().add(i + 12)), a3);
        a0 = _mm256_fmadd_pd(v1, _mm256_loadu_pd(r1.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_pd(v1, _mm256_loadu_pd(r1.as_ptr().add(i + 4)), a1);
        a2 = _mm256_fmadd_pd(v1, _mm256_loadu_pd(r1.as_ptr().add(i + 8)), a2);
        a3 = _mm256_fmadd_pd(v1, _mm256_loadu_pd(r1.as_ptr().add(i + 12)), a3);
        a0 = _mm256_fmadd_pd(v2, _mm256_loadu_pd(r2.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_pd(v2, _mm256_loadu_pd(r2.as_ptr().add(i + 4)), a1);
        a2 = _mm256_fmadd_pd(v2, _mm256_loadu_pd(r2.as_ptr().add(i + 8)), a2);
        a3 = _mm256_fmadd_pd(v2, _mm256_loadu_pd(r2.as_ptr().add(i + 12)), a3);
        a0 = _mm256_fmadd_pd(v3, _mm256_loadu_pd(r3.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_pd(v3, _mm256_loadu_pd(r3.as_ptr().add(i + 4)), a1);
        a2 = _mm256_fmadd_pd(v3, _mm256_loadu_pd(r3.as_ptr().add(i + 8)), a2);
        a3 = _mm256_fmadd_pd(v3, _mm256_loadu_pd(r3.as_ptr().add(i + 12)), a3);
        _mm256_storeu_pd(y.as_mut_ptr().add(i), a0);
        _mm256_storeu_pd(y.as_mut_ptr().add(i + 4), a1);
        _mm256_storeu_pd(y.as_mut_ptr().add(i + 8), a2);
        _mm256_storeu_pd(y.as_mut_ptr().add(i + 12), a3);
        i += 16;
    }
    while i + 4 <= n {
        let mut acc = _mm256_loadu_pd(y.as_ptr().add(i));
        acc = _mm256_fmadd_pd(v0, _mm256_loadu_pd(r0.as_ptr().add(i)), acc);
        acc = _mm256_fmadd_pd(v1, _mm256_loadu_pd(r1.as_ptr().add(i)), acc);
        acc = _mm256_fmadd_pd(v2, _mm256_loadu_pd(r2.as_ptr().add(i)), acc);
        acc = _mm256_fmadd_pd(v3, _mm256_loadu_pd(r3.as_ptr().add(i)), acc);
        _mm256_storeu_pd(y.as_mut_ptr().add(i), acc);
        i += 4;
    }
    while i < n {
        // Sequential mul_add: same 4 roundings as 4 axpy calls (bitwise).
        y[i] = c0.mul_add(r0[i], y[i]);
        y[i] = c1.mul_add(r1[i], y[i]);
        y[i] = c2.mul_add(r2[i], y[i]);
        y[i] = c3.mul_add(r3[i], y[i]);
        i += 1;
    }
}

/// f32 twin of [`fanout_4row_f64`]: 8-wide lanes, 32 elements per
/// block. Same bitwise-vs-4-axpys contract.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fanout_4row_f32(
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    y: &mut [f32],
) {
    debug_assert_eq!(r0.len(), y.len());
    debug_assert_eq!(r1.len(), y.len());
    debug_assert_eq!(r2.len(), y.len());
    debug_assert_eq!(r3.len(), y.len());
    let n = y.len();
    let v0 = _mm256_set1_ps(c0);
    let v1 = _mm256_set1_ps(c1);
    let v2 = _mm256_set1_ps(c2);
    let v3 = _mm256_set1_ps(c3);
    let mut i = 0;
    while i + 32 <= n {
        let mut a0 = _mm256_loadu_ps(y.as_ptr().add(i));
        let mut a1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
        let mut a2 = _mm256_loadu_ps(y.as_ptr().add(i + 16));
        let mut a3 = _mm256_loadu_ps(y.as_ptr().add(i + 24));
        a0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(r0.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(r0.as_ptr().add(i + 8)), a1);
        a2 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(r0.as_ptr().add(i + 16)), a2);
        a3 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(r0.as_ptr().add(i + 24)), a3);
        a0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(r1.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(r1.as_ptr().add(i + 8)), a1);
        a2 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(r1.as_ptr().add(i + 16)), a2);
        a3 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(r1.as_ptr().add(i + 24)), a3);
        a0 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(r2.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(r2.as_ptr().add(i + 8)), a1);
        a2 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(r2.as_ptr().add(i + 16)), a2);
        a3 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(r2.as_ptr().add(i + 24)), a3);
        a0 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(r3.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(r3.as_ptr().add(i + 8)), a1);
        a2 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(r3.as_ptr().add(i + 16)), a2);
        a3 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(r3.as_ptr().add(i + 24)), a3);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), a0);
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), a1);
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 16), a2);
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 24), a3);
        i += 32;
    }
    while i + 8 <= n {
        let mut acc = _mm256_loadu_ps(y.as_ptr().add(i));
        acc = _mm256_fmadd_ps(v0, _mm256_loadu_ps(r0.as_ptr().add(i)), acc);
        acc = _mm256_fmadd_ps(v1, _mm256_loadu_ps(r1.as_ptr().add(i)), acc);
        acc = _mm256_fmadd_ps(v2, _mm256_loadu_ps(r2.as_ptr().add(i)), acc);
        acc = _mm256_fmadd_ps(v3, _mm256_loadu_ps(r3.as_ptr().add(i)), acc);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), acc);
        i += 8;
    }
    while i < n {
        // Sequential mul_add: same 4 roundings as 4 axpy calls (bitwise).
        y[i] = c0.mul_add(r0[i], y[i]);
        y[i] = c1.mul_add(r1[i], y[i]);
        y[i] = c2.mul_add(r2[i], y[i]);
        y[i] = c3.mul_add(r3[i], y[i]);
        i += 1;
    }
}

/// Fused rank-1 update `block[i*bs+j] += mults[i*ms] * pivot[j]` (f32).
///
/// Single-dispatch replacement for a row loop of `axpy` calls over the same
/// rows: the LU panel factorization issues one tiny axpy per row per column
/// (millions of dispatches at n = 2048). For `alpha = ±1` (the LU use)
/// bitwise identical to that axpy loop — exact sign flip plus the same
/// FMA-per-element arithmetic, each element written once.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ger_update_f32(
    alpha: f32,
    mults: &[f32],
    mult_stride: usize,
    pivot: &[f32],
    block: &mut [f32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    let va = _mm256_set1_ps(alpha);
    let mut i = 0;
    // Four-row unrolled main loop for tall-skinny blocks (LU panels):
    // mirrors `ger_update_f64` — the pivot row stays in registers across
    // the four rows instead of being reloaded per row, and per-row
    // broadcast/loop overhead is quartered. Each element computes the
    // identical op sequence (`vm = alpha*mult` once, then one FMA), so
    // results are bitwise identical to the single-row loop below.
    while i + 4 <= rows {
        let m0 = *mults.as_ptr().add(i * mult_stride);
        let m1 = *mults.as_ptr().add((i + 1) * mult_stride);
        let m2 = *mults.as_ptr().add((i + 2) * mult_stride);
        let m3 = *mults.as_ptr().add((i + 3) * mult_stride);
        // NOTE: `alpha` is ±1 in the LU use, so these multiplies are exact
        // sign flips and each `vm` matches the axpy loop's scalar-negated
        // factor bit-for-bit (other `alpha` values add one rounding here,
        // exactly as the single-row loop does).
        let v0 = _mm256_mul_ps(va, _mm256_set1_ps(m0));
        let v1 = _mm256_mul_ps(va, _mm256_set1_ps(m1));
        let v2 = _mm256_mul_ps(va, _mm256_set1_ps(m2));
        let v3 = _mm256_mul_ps(va, _mm256_set1_ps(m3));
        let r0 = block.as_mut_ptr().add(i * block_stride);
        let r1 = block.as_mut_ptr().add((i + 1) * block_stride);
        let r2 = block.as_mut_ptr().add((i + 2) * block_stride);
        let r3 = block.as_mut_ptr().add((i + 3) * block_stride);
        let mut j = 0;
        while j + 8 <= cols {
            let pv = _mm256_loadu_ps(pivot.as_ptr().add(j));
            let a0 = _mm256_loadu_ps(r0.add(j));
            let a1 = _mm256_loadu_ps(r1.add(j));
            let a2 = _mm256_loadu_ps(r2.add(j));
            let a3 = _mm256_loadu_ps(r3.add(j));
            _mm256_storeu_ps(r0.add(j), _mm256_fmadd_ps(v0, pv, a0));
            _mm256_storeu_ps(r1.add(j), _mm256_fmadd_ps(v1, pv, a1));
            _mm256_storeu_ps(r2.add(j), _mm256_fmadd_ps(v2, pv, a2));
            _mm256_storeu_ps(r3.add(j), _mm256_fmadd_ps(v3, pv, a3));
            j += 8;
        }
        while j < cols {
            let pv = *pivot.as_ptr().add(j);
            *r0.add(j) += alpha * m0 * pv;
            *r1.add(j) += alpha * m1 * pv;
            *r2.add(j) += alpha * m2 * pv;
            *r3.add(j) += alpha * m3 * pv;
            j += 1;
        }
        i += 4;
    }
    while i < rows {
        let mult = *mults.as_ptr().add(i * mult_stride);
        // NOTE: `alpha` is ±1 in the LU use, so this multiply is an exact
        // sign flip and `vm` matches the axpy loop's scalar-negated factor
        // bit-for-bit (other `alpha` values add one rounding here).
        let vm = _mm256_mul_ps(va, _mm256_set1_ps(mult));
        let row = block.as_mut_ptr().add(i * block_stride);
        let mut j = 0;
        while j + 32 <= cols {
            for lane in 0..4 {
                let offset = j + lane * 8;
                let pv = _mm256_loadu_ps(pivot.as_ptr().add(offset));
                let rv = _mm256_loadu_ps(row.add(offset));
                _mm256_storeu_ps(row.add(offset), _mm256_fmadd_ps(vm, pv, rv));
            }
            j += 32;
        }
        while j + 8 <= cols {
            let pv = _mm256_loadu_ps(pivot.as_ptr().add(j));
            let rv = _mm256_loadu_ps(row.add(j));
            _mm256_storeu_ps(row.add(j), _mm256_fmadd_ps(vm, pv, rv));
            j += 8;
        }
        while j < cols {
            *row.add(j) += alpha * mult * *pivot.as_ptr().add(j);
            j += 1;
        }
        i += 1;
    }
}

/// Pivot scan `max |base[(first+i)*stride]|` over `len` rows (f32).
///
/// Vectorized idamax for the LU panel factorization (millions of scalar
/// iterations at n = 2048). Returns the same `(row, value)` as the scalar
/// loop bit-for-bit: per-lane strict `>` keeps the first maximum across
/// blocks, the cross-lane fix-up breaks ties by lowest row, and a
/// NaN/+Inf initial pivot keeps its row (scalar: nothing exceeds it via
/// `>`). NaN later values never win on either path.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pivot_scan_f32(
    base: &[f32],
    stride: usize,
    first: usize,
    len: usize,
) -> (usize, f32) {
    debug_assert!(len >= 1);
    let p = base.as_ptr();
    let init = (*p.add(first * stride)).abs();
    if !init.is_finite() && init != f32::NEG_INFINITY {
        return (first, init);
    }
    let signbit = _mm256_set1_ps(-0.0);
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let r0 = first as i32;
    let mut imax = _mm256_set_epi32(r0 + 7, r0 + 6, r0 + 5, r0 + 4, r0 + 3, r0 + 2, r0 + 1, r0);
    let blocks = len / 8;
    let mut b = 0;
    while b < blocks {
        let r = r0 + b as i32 * 8;
        let v = _mm256_setr_ps(
            *p.add((r as usize) * stride),
            *p.add((r as usize + 1) * stride),
            *p.add((r as usize + 2) * stride),
            *p.add((r as usize + 3) * stride),
            *p.add((r as usize + 4) * stride),
            *p.add((r as usize + 5) * stride),
            *p.add((r as usize + 6) * stride),
            *p.add((r as usize + 7) * stride),
        );
        let v = _mm256_andnot_ps(signbit, v);
        let cmp = _mm256_cmp_ps(v, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_ps(vmax, v, cmp);
        let idx = _mm256_set_epi32(r + 7, r + 6, r + 5, r + 4, r + 3, r + 2, r + 1, r);
        imax = _mm256_blendv_epi8(imax, idx, _mm256_castps_si256(cmp));
        b += 1;
    }
    let mut vals = [f32::NEG_INFINITY; 8];
    let mut idxs = [0i32; 8];
    _mm256_storeu_ps(vals.as_mut_ptr(), vmax);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, imax);
    let mut best_row = first;
    let mut best_val = init;
    // Lane order is not row order across blocks, so ties must compare rows
    // explicitly (first maximum wins). `vals` holds finite magnitudes here:
    // NaN never wins a lane (`>` is false) and a NaN/+Inf `init` returned
    // early, so `==` is an exact tie test.
    for lane in 0..8 {
        if vals[lane] > best_val || (vals[lane] == best_val && (idxs[lane] as usize) < best_row) {
            best_val = vals[lane];
            best_row = idxs[lane] as usize;
        }
    }
    // Tail rows (strict `>`, first wins) — also covers len < 8.
    let mut i = first + blocks * 8;
    let end = first + len;
    while i < end {
        let val = (*p.add(i * stride)).abs();
        if val > best_val {
            best_val = val;
            best_row = i;
        }
        i += 1;
    }
    (best_row, best_val)
}

/// Pivot scan `max |base[(first+i)*stride]|` over `len` rows (f64).
/// Same bit-for-bit contract as [`pivot_scan_f32`].
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pivot_scan_f64(
    base: &[f64],
    stride: usize,
    first: usize,
    len: usize,
) -> (usize, f64) {
    debug_assert!(len >= 1);
    let p = base.as_ptr();
    let init = (*p.add(first * stride)).abs();
    if !init.is_finite() && init != f64::NEG_INFINITY {
        return (first, init);
    }
    let signbit = _mm256_set1_pd(-0.0);
    let mut vmax = _mm256_set1_pd(f64::NEG_INFINITY);
    let r0 = first as i64;
    let mut imax = _mm256_set_epi64x(r0 + 3, r0 + 2, r0 + 1, r0);
    let blocks = len / 4;
    let mut b = 0;
    while b < blocks {
        let r = r0 + b as i64 * 4;
        let v = _mm256_setr_pd(
            *p.add(r as usize * stride),
            *p.add((r as usize + 1) * stride),
            *p.add((r as usize + 2) * stride),
            *p.add((r as usize + 3) * stride),
        );
        let v = _mm256_andnot_pd(signbit, v);
        let cmp = _mm256_cmp_pd(v, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_pd(vmax, v, cmp);
        let idx = _mm256_set_epi64x(r + 3, r + 2, r + 1, r);
        imax = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(imax),
            _mm256_castsi256_ps(idx),
            _mm256_castpd_ps(cmp),
        ));
        b += 1;
    }
    let mut vals = [f64::NEG_INFINITY; 4];
    let mut idxs = [0i64; 4];
    _mm256_storeu_pd(vals.as_mut_ptr(), vmax);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, imax);
    let mut best_row = first;
    let mut best_val = init;
    // Lane order is not row order across blocks, so ties must compare rows
    // explicitly (first maximum wins). `vals` holds finite magnitudes here:
    // NaN never wins a lane (`>` is false) and a NaN/+Inf `init` returned
    // early, so `==` is an exact tie test.
    for lane in 0..4 {
        if vals[lane] > best_val || (vals[lane] == best_val && (idxs[lane] as usize) < best_row) {
            best_val = vals[lane];
            best_row = idxs[lane] as usize;
        }
    }
    // Tail rows (strict `>`, first wins) — also covers len < 4.
    let mut i = first + blocks * 4;
    let end = first + len;
    while i < end {
        let val = (*p.add(i * stride)).abs();
        if val > best_val {
            best_val = val;
            best_row = i;
        }
        i += 1;
    }
    (best_row, best_val)
}

/// Fused rank-1 update `block[i*bs+j] += mults[i*ms] * pivot[j]` (f64).
/// Same single-dispatch motivation and bitwise parity as [`ger_update_f32`].
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ger_update_f64(
    alpha: f64,
    mults: &[f64],
    mult_stride: usize,
    pivot: &[f64],
    block: &mut [f64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    let va = _mm256_set1_pd(alpha);
    let mut i = 0;
    // Four-row unrolled main loop for tall-skinny blocks (LU panels): the
    // pivot row stays in registers across the four rows instead of being
    // reloaded per row, and per-row broadcast/loop overhead is quartered.
    // Each element computes the identical op sequence (`vm = alpha*mult`
    // once, then one FMA), so results are bitwise identical to the
    // single-row loop below.
    while i + 4 <= rows {
        let m0 = *mults.as_ptr().add(i * mult_stride);
        let m1 = *mults.as_ptr().add((i + 1) * mult_stride);
        let m2 = *mults.as_ptr().add((i + 2) * mult_stride);
        let m3 = *mults.as_ptr().add((i + 3) * mult_stride);
        // NOTE: `alpha` is ±1 in the LU use, so these multiplies are exact
        // sign flips and each `vm` matches the axpy loop's scalar-negated
        // factor bit-for-bit (other `alpha` values add one rounding here,
        // exactly as the single-row loop does).
        let v0 = _mm256_mul_pd(va, _mm256_set1_pd(m0));
        let v1 = _mm256_mul_pd(va, _mm256_set1_pd(m1));
        let v2 = _mm256_mul_pd(va, _mm256_set1_pd(m2));
        let v3 = _mm256_mul_pd(va, _mm256_set1_pd(m3));
        let r0 = block.as_mut_ptr().add(i * block_stride);
        let r1 = block.as_mut_ptr().add((i + 1) * block_stride);
        let r2 = block.as_mut_ptr().add((i + 2) * block_stride);
        let r3 = block.as_mut_ptr().add((i + 3) * block_stride);
        let mut j = 0;
        while j + 4 <= cols {
            let pv = _mm256_loadu_pd(pivot.as_ptr().add(j));
            let a0 = _mm256_loadu_pd(r0.add(j));
            let a1 = _mm256_loadu_pd(r1.add(j));
            let a2 = _mm256_loadu_pd(r2.add(j));
            let a3 = _mm256_loadu_pd(r3.add(j));
            _mm256_storeu_pd(r0.add(j), _mm256_fmadd_pd(v0, pv, a0));
            _mm256_storeu_pd(r1.add(j), _mm256_fmadd_pd(v1, pv, a1));
            _mm256_storeu_pd(r2.add(j), _mm256_fmadd_pd(v2, pv, a2));
            _mm256_storeu_pd(r3.add(j), _mm256_fmadd_pd(v3, pv, a3));
            j += 4;
        }
        while j < cols {
            let pv = *pivot.as_ptr().add(j);
            *r0.add(j) += alpha * m0 * pv;
            *r1.add(j) += alpha * m1 * pv;
            *r2.add(j) += alpha * m2 * pv;
            *r3.add(j) += alpha * m3 * pv;
            j += 1;
        }
        i += 4;
    }
    while i < rows {
        let mult = *mults.as_ptr().add(i * mult_stride);
        // NOTE: `alpha` is ±1 in the LU use, so this multiply is an exact
        // sign flip and `vm` matches the axpy loop's scalar-negated factor
        // bit-for-bit (other `alpha` values add one rounding here).
        let vm = _mm256_mul_pd(va, _mm256_set1_pd(mult));
        let row = block.as_mut_ptr().add(i * block_stride);
        let mut j = 0;
        while j + 16 <= cols {
            for lane in 0..4 {
                let offset = j + lane * 4;
                let pv = _mm256_loadu_pd(pivot.as_ptr().add(offset));
                let rv = _mm256_loadu_pd(row.add(offset));
                _mm256_storeu_pd(row.add(offset), _mm256_fmadd_pd(vm, pv, rv));
            }
            j += 16;
        }
        while j + 4 <= cols {
            let pv = _mm256_loadu_pd(pivot.as_ptr().add(j));
            let rv = _mm256_loadu_pd(row.add(j));
            _mm256_storeu_pd(row.add(j), _mm256_fmadd_pd(vm, pv, rv));
            j += 4;
        }
        while j < cols {
            *row.add(j) += alpha * mult * *pivot.as_ptr().add(j);
            j += 1;
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn scal_f32(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    let mut i = 0;
    let va = _mm256_set1_ps(alpha);
    while i + 32 <= n {
        for lane in 0..4 {
            let offset = i + lane * 8;
            let xv = _mm256_loadu_ps(x.as_ptr().add(offset));
            _mm256_storeu_ps(x.as_mut_ptr().add(offset), _mm256_mul_ps(va, xv));
        }
        i += 32;
    }
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(va, xv));
        i += 8;
    }
    while i < n {
        x[i] *= alpha;
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn scal_f64(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let mut i = 0;
    let va = _mm256_set1_pd(alpha);
    while i + 16 <= n {
        for lane in 0..4 {
            let offset = i + lane * 4;
            let xv = _mm256_loadu_pd(x.as_ptr().add(offset));
            _mm256_storeu_pd(x.as_mut_ptr().add(offset), _mm256_mul_pd(va, xv));
        }
        i += 16;
    }
    while i + 4 <= n {
        let xv = _mm256_loadu_pd(x.as_ptr().add(i));
        _mm256_storeu_pd(x.as_mut_ptr().add(i), _mm256_mul_pd(va, xv));
        i += 4;
    }
    while i < n {
        x[i] *= alpha;
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    while i + 32 <= n {
        a0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(i)),
            _mm256_loadu_ps(y.as_ptr().add(i)),
            a0,
        );
        a1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(i + 8)),
            _mm256_loadu_ps(y.as_ptr().add(i + 8)),
            a1,
        );
        a2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(i + 16)),
            _mm256_loadu_ps(y.as_ptr().add(i + 16)),
            a2,
        );
        a3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(i + 24)),
            _mm256_loadu_ps(y.as_ptr().add(i + 24)),
            a3,
        );
        i += 32;
    }
    let mut acc = hsum_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    while i + 8 <= n {
        let prod = _mm256_mul_ps(
            _mm256_loadu_ps(x.as_ptr().add(i)),
            _mm256_loadu_ps(y.as_ptr().add(i)),
        );
        acc += hsum_ps(prod);
        i += 8;
    }
    while i < n {
        acc += x[i] * y[i];
        i += 1;
    }
    acc
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    let mut a0 = _mm256_setzero_pd();
    let mut a1 = _mm256_setzero_pd();
    let mut a2 = _mm256_setzero_pd();
    let mut a3 = _mm256_setzero_pd();
    while i + 16 <= n {
        a0 = _mm256_fmadd_pd(
            _mm256_loadu_pd(x.as_ptr().add(i)),
            _mm256_loadu_pd(y.as_ptr().add(i)),
            a0,
        );
        a1 = _mm256_fmadd_pd(
            _mm256_loadu_pd(x.as_ptr().add(i + 4)),
            _mm256_loadu_pd(y.as_ptr().add(i + 4)),
            a1,
        );
        a2 = _mm256_fmadd_pd(
            _mm256_loadu_pd(x.as_ptr().add(i + 8)),
            _mm256_loadu_pd(y.as_ptr().add(i + 8)),
            a2,
        );
        a3 = _mm256_fmadd_pd(
            _mm256_loadu_pd(x.as_ptr().add(i + 12)),
            _mm256_loadu_pd(y.as_ptr().add(i + 12)),
            a3,
        );
        i += 16;
    }
    let mut acc = hsum_pd(_mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)));
    while i + 4 <= n {
        let prod = _mm256_mul_pd(
            _mm256_loadu_pd(x.as_ptr().add(i)),
            _mm256_loadu_pd(y.as_ptr().add(i)),
        );
        acc += hsum_pd(prod);
        i += 4;
    }
    while i < n {
        acc += x[i] * y[i];
        i += 1;
    }
    acc
}

/// Four simultaneous dot products sharing one left operand:
/// `out[r] = Σ_{i<n} l[i]·x_r[i]`.
///
/// Vectorized twin of the scalar row loop in the Cholesky TRSM kernel
/// (`trsm_row4`): the `l` vector loads amortize over four rows instead of
/// streaming per row. FMA-per-element like the scalar multiply-add, but
/// chunked accumulation order differs from serial → same tolerance class,
/// not bitwise. Slices must share one length (checked in debug).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot4_f32(l: &[f32], x0: &[f32], x1: &[f32], x2: &[f32], x3: &[f32]) -> [f32; 4] {
    debug_assert_eq!(l.len(), x0.len());
    debug_assert_eq!(l.len(), x1.len());
    debug_assert_eq!(l.len(), x2.len());
    debug_assert_eq!(l.len(), x3.len());
    let n = l.len();
    let mut i = 0;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    while i + 8 <= n {
        let lv = _mm256_loadu_ps(l.as_ptr().add(i));
        a0 = _mm256_fmadd_ps(lv, _mm256_loadu_ps(x0.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_ps(lv, _mm256_loadu_ps(x1.as_ptr().add(i)), a1);
        a2 = _mm256_fmadd_ps(lv, _mm256_loadu_ps(x2.as_ptr().add(i)), a2);
        a3 = _mm256_fmadd_ps(lv, _mm256_loadu_ps(x3.as_ptr().add(i)), a3);
        i += 8;
    }
    let mut out = [hsum_ps(a0), hsum_ps(a1), hsum_ps(a2), hsum_ps(a3)];
    while i < n {
        let lv = *l.as_ptr().add(i);
        out[0] += lv * *x0.as_ptr().add(i);
        out[1] += lv * *x1.as_ptr().add(i);
        out[2] += lv * *x2.as_ptr().add(i);
        out[3] += lv * *x3.as_ptr().add(i);
        i += 1;
    }
    out
}

/// f64 twin of [`dot4_f32`] (4-wide lanes).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot4_f64(l: &[f64], x0: &[f64], x1: &[f64], x2: &[f64], x3: &[f64]) -> [f64; 4] {
    debug_assert_eq!(l.len(), x0.len());
    debug_assert_eq!(l.len(), x1.len());
    debug_assert_eq!(l.len(), x2.len());
    debug_assert_eq!(l.len(), x3.len());
    let n = l.len();
    let mut i = 0;
    let mut a0 = _mm256_setzero_pd();
    let mut a1 = _mm256_setzero_pd();
    let mut a2 = _mm256_setzero_pd();
    let mut a3 = _mm256_setzero_pd();
    while i + 4 <= n {
        let lv = _mm256_loadu_pd(l.as_ptr().add(i));
        a0 = _mm256_fmadd_pd(lv, _mm256_loadu_pd(x0.as_ptr().add(i)), a0);
        a1 = _mm256_fmadd_pd(lv, _mm256_loadu_pd(x1.as_ptr().add(i)), a1);
        a2 = _mm256_fmadd_pd(lv, _mm256_loadu_pd(x2.as_ptr().add(i)), a2);
        a3 = _mm256_fmadd_pd(lv, _mm256_loadu_pd(x3.as_ptr().add(i)), a3);
        i += 4;
    }
    let mut out = [hsum_pd(a0), hsum_pd(a1), hsum_pd(a2), hsum_pd(a3)];
    while i < n {
        let lv = *l.as_ptr().add(i);
        out[0] += lv * *x0.as_ptr().add(i);
        out[1] += lv * *x1.as_ptr().add(i);
        out[2] += lv * *x2.as_ptr().add(i);
        out[3] += lv * *x3.as_ptr().add(i);
        i += 1;
    }
    out
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn nrm2_f32(x: &[f32]) -> f32 {
    sumsq_f32(x.as_ptr(), x.len()).sqrt()
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn nrm2_f64(x: &[f64]) -> f64 {
    sumsq_f64(x.as_ptr(), x.len()).sqrt()
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dotc_c32(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    debug_assert_eq!(x.len(), y.len());
    let floats = x.len() * 2;
    let xp = x.as_ptr().cast::<f32>();
    let yp = y.as_ptr().cast::<f32>();
    // Sign mask flipping odd (imaginary) lanes: with y_swap = [yi, yr]
    // per pair, fmadd(xv ^ mask, y_swap) accumulates xr*yi - xi*yr.
    // `_mm256_set_ps` takes lanes e7..e0, so odd lanes read -0.0 here.
    let neg_im = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    let mut r0 = _mm256_setzero_ps();
    let mut r1 = _mm256_setzero_ps();
    let mut r2 = _mm256_setzero_ps();
    let mut r3 = _mm256_setzero_ps();
    let mut i0 = _mm256_setzero_ps();
    let mut i1 = _mm256_setzero_ps();
    let mut i2 = _mm256_setzero_ps();
    let mut i3 = _mm256_setzero_ps();
    let mut i = 0;
    // Four-way unroll like the real dot kernels: two FMADD chains cannot
    // cover the 4-5 cycle FMA latency and trailed OpenBLAS ~1.5x.
    while i + 32 <= floats {
        let x0 = _mm256_loadu_ps(xp.add(i));
        let y0 = _mm256_loadu_ps(yp.add(i));
        r0 = _mm256_fmadd_ps(x0, y0, r0);
        i0 = _mm256_fmadd_ps(
            _mm256_xor_ps(x0, neg_im),
            _mm256_permute_ps(y0, 0b1011_0001),
            i0,
        );
        let x1 = _mm256_loadu_ps(xp.add(i + 8));
        let y1 = _mm256_loadu_ps(yp.add(i + 8));
        r1 = _mm256_fmadd_ps(x1, y1, r1);
        i1 = _mm256_fmadd_ps(
            _mm256_xor_ps(x1, neg_im),
            _mm256_permute_ps(y1, 0b1011_0001),
            i1,
        );
        let x2 = _mm256_loadu_ps(xp.add(i + 16));
        let y2 = _mm256_loadu_ps(yp.add(i + 16));
        r2 = _mm256_fmadd_ps(x2, y2, r2);
        i2 = _mm256_fmadd_ps(
            _mm256_xor_ps(x2, neg_im),
            _mm256_permute_ps(y2, 0b1011_0001),
            i2,
        );
        let x3 = _mm256_loadu_ps(xp.add(i + 24));
        let y3 = _mm256_loadu_ps(yp.add(i + 24));
        r3 = _mm256_fmadd_ps(x3, y3, r3);
        i3 = _mm256_fmadd_ps(
            _mm256_xor_ps(x3, neg_im),
            _mm256_permute_ps(y3, 0b1011_0001),
            i3,
        );
        i += 32;
    }
    while i + 16 <= floats {
        let x0 = _mm256_loadu_ps(xp.add(i));
        let y0 = _mm256_loadu_ps(yp.add(i));
        r0 = _mm256_fmadd_ps(x0, y0, r0);
        i0 = _mm256_fmadd_ps(
            _mm256_xor_ps(x0, neg_im),
            _mm256_permute_ps(y0, 0b1011_0001),
            i0,
        );
        let x1 = _mm256_loadu_ps(xp.add(i + 8));
        let y1 = _mm256_loadu_ps(yp.add(i + 8));
        r1 = _mm256_fmadd_ps(x1, y1, r1);
        i1 = _mm256_fmadd_ps(
            _mm256_xor_ps(x1, neg_im),
            _mm256_permute_ps(y1, 0b1011_0001),
            i1,
        );
        i += 16;
    }
    while i + 8 <= floats {
        let xv = _mm256_loadu_ps(xp.add(i));
        let yv = _mm256_loadu_ps(yp.add(i));
        r0 = _mm256_fmadd_ps(xv, yv, r0);
        i0 = _mm256_fmadd_ps(
            _mm256_xor_ps(xv, neg_im),
            _mm256_permute_ps(yv, 0b1011_0001),
            i0,
        );
        i += 8;
    }
    let r01 = _mm256_add_ps(r0, r1);
    let r23 = _mm256_add_ps(r2, r3);
    let i01 = _mm256_add_ps(i0, i1);
    let i23 = _mm256_add_ps(i2, i3);
    let mut real = hsum_ps(_mm256_add_ps(r01, r23));
    let mut imag = hsum_ps(_mm256_add_ps(i01, i23));
    let mut ci = i / 2;
    while ci < x.len() {
        real += x[ci].re * y[ci].re + x[ci].im * y[ci].im;
        imag += x[ci].re * y[ci].im - x[ci].im * y[ci].re;
        ci += 1;
    }
    Complex32::new(real, imag)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dotc_c64(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    debug_assert_eq!(x.len(), y.len());
    let floats = x.len() * 2;
    let xp = x.as_ptr().cast::<f64>();
    let yp = y.as_ptr().cast::<f64>();
    // Sign mask flipping odd (imaginary) lanes: with y_swap = [yi, yr]
    // per pair, fmadd(xv ^ mask, y_swap) accumulates xr*yi - xi*yr.
    // `_mm256_set_pd` takes lanes e3..e0, so odd lanes read -0.0 here.
    let neg_im = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    let mut r0 = _mm256_setzero_pd();
    let mut r1 = _mm256_setzero_pd();
    let mut r2 = _mm256_setzero_pd();
    let mut r3 = _mm256_setzero_pd();
    let mut i0 = _mm256_setzero_pd();
    let mut i1 = _mm256_setzero_pd();
    let mut i2 = _mm256_setzero_pd();
    let mut i3 = _mm256_setzero_pd();
    let mut i = 0;
    // Four-way unroll like the real dot kernels (see `dotc_c32`).
    while i + 16 <= floats {
        let x0 = _mm256_loadu_pd(xp.add(i));
        let y0 = _mm256_loadu_pd(yp.add(i));
        r0 = _mm256_fmadd_pd(x0, y0, r0);
        i0 = _mm256_fmadd_pd(_mm256_xor_pd(x0, neg_im), _mm256_permute_pd(y0, 0b0101), i0);
        let x1 = _mm256_loadu_pd(xp.add(i + 4));
        let y1 = _mm256_loadu_pd(yp.add(i + 4));
        r1 = _mm256_fmadd_pd(x1, y1, r1);
        i1 = _mm256_fmadd_pd(_mm256_xor_pd(x1, neg_im), _mm256_permute_pd(y1, 0b0101), i1);
        let x2 = _mm256_loadu_pd(xp.add(i + 8));
        let y2 = _mm256_loadu_pd(yp.add(i + 8));
        r2 = _mm256_fmadd_pd(x2, y2, r2);
        i2 = _mm256_fmadd_pd(_mm256_xor_pd(x2, neg_im), _mm256_permute_pd(y2, 0b0101), i2);
        let x3 = _mm256_loadu_pd(xp.add(i + 12));
        let y3 = _mm256_loadu_pd(yp.add(i + 12));
        r3 = _mm256_fmadd_pd(x3, y3, r3);
        i3 = _mm256_fmadd_pd(_mm256_xor_pd(x3, neg_im), _mm256_permute_pd(y3, 0b0101), i3);
        i += 16;
    }
    while i + 8 <= floats {
        let x0 = _mm256_loadu_pd(xp.add(i));
        let y0 = _mm256_loadu_pd(yp.add(i));
        r0 = _mm256_fmadd_pd(x0, y0, r0);
        i0 = _mm256_fmadd_pd(_mm256_xor_pd(x0, neg_im), _mm256_permute_pd(y0, 0b0101), i0);
        let x1 = _mm256_loadu_pd(xp.add(i + 4));
        let y1 = _mm256_loadu_pd(yp.add(i + 4));
        r1 = _mm256_fmadd_pd(x1, y1, r1);
        i1 = _mm256_fmadd_pd(_mm256_xor_pd(x1, neg_im), _mm256_permute_pd(y1, 0b0101), i1);
        i += 8;
    }
    while i + 4 <= floats {
        let xv = _mm256_loadu_pd(xp.add(i));
        let yv = _mm256_loadu_pd(yp.add(i));
        r0 = _mm256_fmadd_pd(xv, yv, r0);
        i0 = _mm256_fmadd_pd(_mm256_xor_pd(xv, neg_im), _mm256_permute_pd(yv, 0b0101), i0);
        i += 4;
    }
    let r01 = _mm256_add_pd(r0, r1);
    let r23 = _mm256_add_pd(r2, r3);
    let i01 = _mm256_add_pd(i0, i1);
    let i23 = _mm256_add_pd(i2, i3);
    let mut real = hsum_pd(_mm256_add_pd(r01, r23));
    let mut imag = hsum_pd(_mm256_add_pd(i01, i23));
    let mut ci = i / 2;
    while ci < x.len() {
        real += x[ci].re * y[ci].re + x[ci].im * y[ci].im;
        imag += x[ci].re * y[ci].im - x[ci].im * y[ci].re;
        ci += 1;
    }
    Complex64::new(real, imag)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_c32(alpha: Complex32, x: &[Complex32], y: &mut [Complex32]) {
    debug_assert_eq!(x.len(), y.len());
    let floats = x.len() * 2;
    let xp = x.as_ptr().cast::<f32>();
    let yp = y.as_mut_ptr().cast::<f32>();
    let mut i = 0;
    let ar = _mm256_set1_ps(alpha.re);
    let ai = _mm256_set1_ps(alpha.im);
    // Four-way unroll like the real kernels: the single-stream version
    // leaves FMA/shuffle ports idle behind the permute+mul+addsub chain
    // (serial complex axpy trailed OpenBLAS ~1.4-1.9x). Per-element op
    // order is unchanged, so results stay bitwise-identical.
    while i + 32 <= floats {
        for lane in 0..4 {
            let offset = i + lane * 8;
            let xv = _mm256_loadu_ps(xp.add(offset));
            let yv = _mm256_loadu_ps(yp.add(offset));
            let prod = complex_mul_ps(xv, ar, ai);
            _mm256_storeu_ps(yp.add(offset), _mm256_add_ps(yv, prod));
        }
        i += 32;
    }
    while i + 8 <= floats {
        let xv = _mm256_loadu_ps(xp.add(i));
        let yv = _mm256_loadu_ps(yp.add(i));
        let prod = complex_mul_ps(xv, ar, ai);
        _mm256_storeu_ps(yp.add(i), _mm256_add_ps(yv, prod));
        i += 8;
    }
    let mut ci = i / 2;
    while ci < x.len() {
        y[ci] += alpha * x[ci];
        ci += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_c64(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    debug_assert_eq!(x.len(), y.len());
    let floats = x.len() * 2;
    let xp = x.as_ptr().cast::<f64>();
    let yp = y.as_mut_ptr().cast::<f64>();
    let mut i = 0;
    let ar = _mm256_set1_pd(alpha.re);
    let ai = _mm256_set1_pd(alpha.im);
    // Four-way unroll like the real kernels (see `axpy_c32`): the
    // single-stream version trails OpenBLAS ~1.4-1.9x serially. Op order
    // per element is unchanged (bitwise-identical).
    while i + 16 <= floats {
        for lane in 0..4 {
            let offset = i + lane * 4;
            let xv = _mm256_loadu_pd(xp.add(offset));
            let yv = _mm256_loadu_pd(yp.add(offset));
            let prod = complex_mul_pd(xv, ar, ai);
            _mm256_storeu_pd(yp.add(offset), _mm256_add_pd(yv, prod));
        }
        i += 16;
    }
    while i + 4 <= floats {
        let xv = _mm256_loadu_pd(xp.add(i));
        let yv = _mm256_loadu_pd(yp.add(i));
        let prod = complex_mul_pd(xv, ar, ai);
        _mm256_storeu_pd(yp.add(i), _mm256_add_pd(yv, prod));
        i += 4;
    }
    let mut ci = i / 2;
    while ci < x.len() {
        y[ci] += alpha * x[ci];
        ci += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn scal_c32(alpha: Complex32, x: &mut [Complex32]) {
    let floats = x.len() * 2;
    let xp = x.as_mut_ptr().cast::<f32>();
    let mut i = 0;
    let ar = _mm256_set1_ps(alpha.re);
    let ai = _mm256_set1_ps(alpha.im);
    // Four-way unroll (see `axpy_c32`): single-stream trailed OpenBLAS
    // ~1.2-1.5x. Op order per element unchanged (bitwise-identical).
    while i + 32 <= floats {
        for lane in 0..4 {
            let offset = i + lane * 8;
            let xv = _mm256_loadu_ps(xp.add(offset));
            _mm256_storeu_ps(xp.add(offset), complex_mul_ps(xv, ar, ai));
        }
        i += 32;
    }
    while i + 8 <= floats {
        let xv = _mm256_loadu_ps(xp.add(i));
        _mm256_storeu_ps(xp.add(i), complex_mul_ps(xv, ar, ai));
        i += 8;
    }
    let mut ci = i / 2;
    while ci < x.len() {
        x[ci] *= alpha;
        ci += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn scal_c64(alpha: Complex64, x: &mut [Complex64]) {
    let floats = x.len() * 2;
    let xp = x.as_mut_ptr().cast::<f64>();
    let mut i = 0;
    let ar = _mm256_set1_pd(alpha.re);
    let ai = _mm256_set1_pd(alpha.im);
    // Four-way unroll (see `axpy_c32`): single-stream trailed OpenBLAS
    // ~1.2x. Op order per element unchanged (bitwise-identical).
    while i + 16 <= floats {
        for lane in 0..4 {
            let offset = i + lane * 4;
            let xv = _mm256_loadu_pd(xp.add(offset));
            _mm256_storeu_pd(xp.add(offset), complex_mul_pd(xv, ar, ai));
        }
        i += 16;
    }
    while i + 4 <= floats {
        let xv = _mm256_loadu_pd(xp.add(i));
        _mm256_storeu_pd(xp.add(i), complex_mul_pd(xv, ar, ai));
        i += 4;
    }
    let mut ci = i / 2;
    while ci < x.len() {
        x[ci] *= alpha;
        ci += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn nrm2_c32(x: &[Complex32]) -> f32 {
    sumsq_f32(x.as_ptr().cast::<f32>(), x.len() * 2).sqrt()
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn nrm2_c64(x: &[Complex64]) -> f64 {
    sumsq_f64(x.as_ptr().cast::<f64>(), x.len() * 2).sqrt()
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn complex_mul_ps(v: __m256, ar: __m256, ai: __m256) -> __m256 {
    let swapped = _mm256_permute_ps(v, 0b1011_0001);
    _mm256_addsub_ps(_mm256_mul_ps(v, ar), _mm256_mul_ps(swapped, ai))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn complex_mul_pd(v: __m256d, ar: __m256d, ai: __m256d) -> __m256d {
    let swapped = _mm256_permute_pd(v, 0b0101);
    _mm256_addsub_pd(_mm256_mul_pd(v, ar), _mm256_mul_pd(swapped, ai))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sumsq_f32(ptr: *const f32, n: usize) -> f32 {
    let mut i = 0;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    while i + 16 <= n {
        let x0 = _mm256_loadu_ps(ptr.add(i));
        let x1 = _mm256_loadu_ps(ptr.add(i + 8));
        a0 = _mm256_fmadd_ps(x0, x0, a0);
        a1 = _mm256_fmadd_ps(x1, x1, a1);
        i += 16;
    }
    let mut acc = hsum_ps(_mm256_add_ps(a0, a1));
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(ptr.add(i));
        acc += hsum_ps(_mm256_mul_ps(xv, xv));
        i += 8;
    }
    while i < n {
        let v = *ptr.add(i);
        acc += v * v;
        i += 1;
    }
    acc
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sumsq_f64(ptr: *const f64, n: usize) -> f64 {
    let mut i = 0;
    let mut a0 = _mm256_setzero_pd();
    let mut a1 = _mm256_setzero_pd();
    while i + 8 <= n {
        let x0 = _mm256_loadu_pd(ptr.add(i));
        let x1 = _mm256_loadu_pd(ptr.add(i + 4));
        a0 = _mm256_fmadd_pd(x0, x0, a0);
        a1 = _mm256_fmadd_pd(x1, x1, a1);
        i += 8;
    }
    let mut acc = hsum_pd(_mm256_add_pd(a0, a1));
    while i + 4 <= n {
        let xv = _mm256_loadu_pd(ptr.add(i));
        acc += hsum_pd(_mm256_mul_pd(xv, xv));
        i += 4;
    }
    while i < n {
        let v = *ptr.add(i);
        acc += v * v;
        i += 1;
    }
    acc
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn hsum_ps(v: __m256) -> f32 {
    let mut tmp = [0.0_f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), v);
    tmp.iter().sum()
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn hsum_pd(v: __m256d) -> f64 {
    let mut tmp = [0.0_f64; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), v);
    tmp.iter().sum()
}

// ---------- axpy_many: y[i] += Σ_bb alphas[bb] * x_basis[bb][i] ----------

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_many_f32(alphas: &[f32], x_basis: &[&[f32]], y: &mut [f32]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    let mut i = 0;
    while i + 32 <= n {
        let mut a0 = _mm256_loadu_ps(y.as_ptr().add(i));
        let mut a1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
        let mut a2 = _mm256_loadu_ps(y.as_ptr().add(i + 16));
        let mut a3 = _mm256_loadu_ps(y.as_ptr().add(i + 24));
        for bb in 0..k {
            let va = _mm256_set1_ps(alphas[bb]);
            let xb0 = _mm256_loadu_ps(x_basis[bb].as_ptr().add(i));
            let xb1 = _mm256_loadu_ps(x_basis[bb].as_ptr().add(i + 8));
            let xb2 = _mm256_loadu_ps(x_basis[bb].as_ptr().add(i + 16));
            let xb3 = _mm256_loadu_ps(x_basis[bb].as_ptr().add(i + 24));
            a0 = _mm256_fmadd_ps(va, xb0, a0);
            a1 = _mm256_fmadd_ps(va, xb1, a1);
            a2 = _mm256_fmadd_ps(va, xb2, a2);
            a3 = _mm256_fmadd_ps(va, xb3, a3);
        }
        _mm256_storeu_ps(y.as_mut_ptr().add(i), a0);
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), a1);
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 16), a2);
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 24), a3);
        i += 32;
    }
    while i + 8 <= n {
        let mut acc = _mm256_loadu_ps(y.as_ptr().add(i));
        for bb in 0..k {
            let va = _mm256_set1_ps(alphas[bb]);
            let xb = _mm256_loadu_ps(x_basis[bb].as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, xb, acc);
        }
        _mm256_storeu_ps(y.as_mut_ptr().add(i), acc);
        i += 8;
    }
    while i < n {
        let mut acc = y[i];
        for bb in 0..k {
            acc += alphas[bb] * x_basis[bb][i];
        }
        y[i] = acc;
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_many_f64(alphas: &[f64], x_basis: &[&[f64]], y: &mut [f64]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    let mut i = 0;
    while i + 16 <= n {
        let mut a0 = _mm256_loadu_pd(y.as_ptr().add(i));
        let mut a1 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
        let mut a2 = _mm256_loadu_pd(y.as_ptr().add(i + 8));
        let mut a3 = _mm256_loadu_pd(y.as_ptr().add(i + 12));
        for bb in 0..k {
            let va = _mm256_set1_pd(alphas[bb]);
            let xb0 = _mm256_loadu_pd(x_basis[bb].as_ptr().add(i));
            let xb1 = _mm256_loadu_pd(x_basis[bb].as_ptr().add(i + 4));
            let xb2 = _mm256_loadu_pd(x_basis[bb].as_ptr().add(i + 8));
            let xb3 = _mm256_loadu_pd(x_basis[bb].as_ptr().add(i + 12));
            a0 = _mm256_fmadd_pd(va, xb0, a0);
            a1 = _mm256_fmadd_pd(va, xb1, a1);
            a2 = _mm256_fmadd_pd(va, xb2, a2);
            a3 = _mm256_fmadd_pd(va, xb3, a3);
        }
        _mm256_storeu_pd(y.as_mut_ptr().add(i), a0);
        _mm256_storeu_pd(y.as_mut_ptr().add(i + 4), a1);
        _mm256_storeu_pd(y.as_mut_ptr().add(i + 8), a2);
        _mm256_storeu_pd(y.as_mut_ptr().add(i + 12), a3);
        i += 16;
    }
    while i + 4 <= n {
        let mut acc = _mm256_loadu_pd(y.as_ptr().add(i));
        for bb in 0..k {
            let va = _mm256_set1_pd(alphas[bb]);
            let xb = _mm256_loadu_pd(x_basis[bb].as_ptr().add(i));
            acc = _mm256_fmadd_pd(va, xb, acc);
        }
        _mm256_storeu_pd(y.as_mut_ptr().add(i), acc);
        i += 4;
    }
    while i < n {
        let mut acc = y[i];
        for bb in 0..k {
            acc += alphas[bb] * x_basis[bb][i];
        }
        y[i] = acc;
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_many_c32(alphas: &[Complex32], x_basis: &[&[Complex32]], y: &mut [Complex32]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    let floats = n * 2;
    let yp = y.as_mut_ptr().cast::<f32>();
    let mut i = 0;
    while i + 16 <= floats {
        let yv0 = _mm256_loadu_ps(yp.add(i));
        let yv1 = _mm256_loadu_ps(yp.add(i + 8));
        let mut acc0 = yv0;
        let mut acc1 = yv1;
        for bb in 0..k {
            let ar = _mm256_set1_ps(alphas[bb].re);
            let ai = _mm256_set1_ps(alphas[bb].im);
            let xp = x_basis[bb].as_ptr().cast::<f32>();
            let xv0 = _mm256_loadu_ps(xp.add(i));
            let xv1 = _mm256_loadu_ps(xp.add(i + 8));
            let xs0 = _mm256_permute_ps(xv0, 0b1011_0001);
            let xs1 = _mm256_permute_ps(xv1, 0b1011_0001);
            acc0 = _mm256_add_ps(
                acc0,
                _mm256_addsub_ps(_mm256_mul_ps(xv0, ar), _mm256_mul_ps(xs0, ai)),
            );
            acc1 = _mm256_add_ps(
                acc1,
                _mm256_addsub_ps(_mm256_mul_ps(xv1, ar), _mm256_mul_ps(xs1, ai)),
            );
        }
        _mm256_storeu_ps(yp.add(i), acc0);
        _mm256_storeu_ps(yp.add(i + 8), acc1);
        i += 16;
    }
    let mut ci = i / 2;
    while ci < n {
        let mut acc_re = y[ci].re;
        let mut acc_im = y[ci].im;
        for bb in 0..k {
            let alpha = alphas[bb];
            let xb = x_basis[bb][ci];
            acc_re += alpha.re * xb.re - alpha.im * xb.im;
            acc_im += alpha.re * xb.im + alpha.im * xb.re;
        }
        y[ci].re = acc_re;
        y[ci].im = acc_im;
        ci += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_many_c64(alphas: &[Complex64], x_basis: &[&[Complex64]], y: &mut [Complex64]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    let floats = n * 2;
    let yp = y.as_mut_ptr().cast::<f64>();
    let mut i = 0;
    while i + 4 <= floats {
        let yv = _mm256_loadu_pd(yp.add(i));
        let mut acc = yv;
        for bb in 0..k {
            let ar = _mm256_set1_pd(alphas[bb].re);
            let ai = _mm256_set1_pd(alphas[bb].im);
            let xp = x_basis[bb].as_ptr().cast::<f64>();
            let xv = _mm256_loadu_pd(xp.add(i));
            let xs = _mm256_permute_pd(xv, 0b0101);
            let prod = _mm256_addsub_pd(_mm256_mul_pd(xv, ar), _mm256_mul_pd(xs, ai));
            acc = _mm256_add_pd(acc, prod);
        }
        _mm256_storeu_pd(yp.add(i), acc);
        i += 4;
    }
    let mut ci = i / 2;
    while ci < n {
        let mut acc_re = y[ci].re;
        let mut acc_im = y[ci].im;
        for bb in 0..k {
            let alpha = alphas[bb];
            let xb = x_basis[bb][ci];
            acc_re += alpha.re * xb.re - alpha.im * xb.im;
            acc_im += alpha.re * xb.im + alpha.im * xb.re;
        }
        y[ci].re = acc_re;
        y[ci].im = acc_im;
        ci += 1;
    }
}

/// First index of a non-finite lane (NaN or ±inf), f32.
///
/// Vectorized scan for the LU/Cholesky entry and end-of-factorize checks:
/// 8-wide exponent-bit test per block (`(bits & 0x7F80_0000) == mask` holds
/// exactly for inf/NaN), with an exact scalar fixup inside the first bad
/// block. Same contract as the scalar loop: the FIRST bad index (fixup is
/// scalar-ordered, and blocks are scanned in order).
#[target_feature(enable = "avx2")]
pub unsafe fn find_non_finite_f32(values: &[f32]) -> Option<usize> {
    let exp = _mm256_set1_epi32(0x7F80_0000u32 as i32);
    let n = values.len();
    let p = values.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(p.add(i));
        let bits = _mm256_castps_si256(v);
        let bad = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exp), exp);
        if _mm256_movemask_ps(_mm256_castsi256_ps(bad)) as u32 != 0 {
            // Rare (never on valid input): exact first index, scalar order.
            let mut j = i;
            while (*p.add(j)).is_finite() {
                j += 1;
            }
            return Some(j);
        }
        i += 8;
    }
    while i < n {
        if !(*p.add(i)).is_finite() {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// f64 twin of [`find_non_finite_f32`]: 4-wide exponent-bit test
/// (`(bits & 0x7FF0_..._0) == mask`), same first-index contract.
#[target_feature(enable = "avx2")]
pub unsafe fn find_non_finite_f64(values: &[f64]) -> Option<usize> {
    let exp = _mm256_set1_epi64x(0x7FF0_0000_0000_0000u64 as i64);
    let n = values.len();
    let p = values.as_ptr();
    let mut i = 0;
    while i + 4 <= n {
        let v = _mm256_loadu_pd(p.add(i));
        let bits = _mm256_castpd_si256(v);
        let bad = _mm256_cmpeq_epi64(_mm256_and_si256(bits, exp), exp);
        if _mm256_movemask_pd(_mm256_castsi256_pd(bad)) as u32 != 0 {
            // Rare (never on valid input): exact first index, scalar order.
            let mut j = i;
            while (*p.add(j)).is_finite() {
                j += 1;
            }
            return Some(j);
        }
        i += 4;
    }
    while i < n {
        if !(*p.add(i)).is_finite() {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Max `|v|` over f32 lanes (`0` when empty).
///
/// Vectorized reduction for the LU entry scale fold (input already proven
/// finite by the entry scan, so plain `max` — which ignores NaN — is safe).
/// Same value as the scalar fold: `abs` is exact and the max of a finite
/// set is unique.
#[target_feature(enable = "avx2")]
pub unsafe fn max_norm_f32(values: &[f32]) -> f32 {
    let signbit = _mm256_set1_ps(-0.0);
    let mut vmax0 = _mm256_setzero_ps();
    let mut vmax1 = _mm256_setzero_ps();
    let mut vmax2 = _mm256_setzero_ps();
    let mut vmax3 = _mm256_setzero_ps();
    let n = values.len();
    let p = values.as_ptr();
    let mut i = 0;
    // Four independent accumulators (32 lanes/iter); see
    // `max_norm_checked_f64` for why a single chain is latency-bound.
    while i + 32 <= n {
        let v0 = _mm256_loadu_ps(p.add(i));
        let v1 = _mm256_loadu_ps(p.add(i + 8));
        let v2 = _mm256_loadu_ps(p.add(i + 16));
        let v3 = _mm256_loadu_ps(p.add(i + 24));
        vmax0 = _mm256_max_ps(vmax0, _mm256_andnot_ps(signbit, v0));
        vmax1 = _mm256_max_ps(vmax1, _mm256_andnot_ps(signbit, v1));
        vmax2 = _mm256_max_ps(vmax2, _mm256_andnot_ps(signbit, v2));
        vmax3 = _mm256_max_ps(vmax3, _mm256_andnot_ps(signbit, v3));
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(p.add(i));
        vmax0 = _mm256_max_ps(vmax0, _mm256_andnot_ps(signbit, v));
        i += 8;
    }
    let vmax01 = _mm256_max_ps(vmax0, vmax1);
    let vmax23 = _mm256_max_ps(vmax2, vmax3);
    let vmax = _mm256_max_ps(vmax01, vmax23);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), vmax);
    let mut best = 0.0f32;
    for &v in &buf {
        if v > best {
            best = v;
        }
    }
    while i < n {
        let a = (*p.add(i)).abs();
        if a > best {
            best = a;
        }
        i += 1;
    }
    best
}

/// f64 twin of [`max_norm_f32`]: 4-wide abs+max reduction.
#[target_feature(enable = "avx2")]
pub unsafe fn max_norm_f64(values: &[f64]) -> f64 {
    let signbit = _mm256_set1_pd(-0.0);
    let mut vmax0 = _mm256_setzero_pd();
    let mut vmax1 = _mm256_setzero_pd();
    let mut vmax2 = _mm256_setzero_pd();
    let mut vmax3 = _mm256_setzero_pd();
    let n = values.len();
    let p = values.as_ptr();
    let mut i = 0;
    // Four independent accumulators (16 lanes/iter); see
    // `max_norm_checked_f64` for why a single chain is latency-bound.
    while i + 16 <= n {
        let v0 = _mm256_loadu_pd(p.add(i));
        let v1 = _mm256_loadu_pd(p.add(i + 4));
        let v2 = _mm256_loadu_pd(p.add(i + 8));
        let v3 = _mm256_loadu_pd(p.add(i + 12));
        vmax0 = _mm256_max_pd(vmax0, _mm256_andnot_pd(signbit, v0));
        vmax1 = _mm256_max_pd(vmax1, _mm256_andnot_pd(signbit, v1));
        vmax2 = _mm256_max_pd(vmax2, _mm256_andnot_pd(signbit, v2));
        vmax3 = _mm256_max_pd(vmax3, _mm256_andnot_pd(signbit, v3));
        i += 16;
    }
    while i + 4 <= n {
        let v = _mm256_loadu_pd(p.add(i));
        vmax0 = _mm256_max_pd(vmax0, _mm256_andnot_pd(signbit, v));
        i += 4;
    }
    let vmax01 = _mm256_max_pd(vmax0, vmax1);
    let vmax23 = _mm256_max_pd(vmax2, vmax3);
    let vmax = _mm256_max_pd(vmax01, vmax23);
    let mut buf = [0.0f64; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), vmax);
    let mut best = 0.0f64;
    for &v in &buf {
        if v > best {
            best = v;
        }
    }
    while i < n {
        let a = (*p.add(i)).abs();
        if a > best {
            best = a;
        }
        i += 1;
    }
    best
}

/// Fused f32 entry-validation fold `(max |v|, all_finite)` in one
/// streaming pass (LU entry scan+scale fusion): the [`max_norm_f32`]
/// abs+max fold plus the [`find_non_finite_f32`] exponent-bit test, the
/// latter as a per-lane comparison OR-accumulated across blocks (OR-ing
/// raw masked fields would forge phantom all-ones exponents from
/// complementary finite lanes, e.g. `0x409 | 0x3FE == 0x7FF`) and reduced
/// once at the end. Same max value as [`max_norm_f32`] whenever the flag
/// holds (finite input); the max is unspecified when it does not —
/// callers rescan for the first bad index then (rare).
#[target_feature(enable = "avx2")]
pub unsafe fn max_norm_checked_f32(values: &[f32]) -> (f32, bool) {
    let signbit = _mm256_set1_ps(-0.0);
    let exp = _mm256_set1_epi32(0x7F80_0000u32 as i32);
    let mut vmax0 = _mm256_setzero_ps();
    let mut vmax1 = _mm256_setzero_ps();
    let mut vmax2 = _mm256_setzero_ps();
    let mut vmax3 = _mm256_setzero_ps();
    let mut bad0 = _mm256_setzero_si256();
    let mut bad1 = _mm256_setzero_si256();
    let mut bad2 = _mm256_setzero_si256();
    let mut bad3 = _mm256_setzero_si256();
    let n = values.len();
    let p = values.as_ptr();
    let mut i = 0;
    // Four independent accumulator pairs (32 lanes/iter); see the f64
    // twin for why a single chain is latency-bound.
    while i + 32 <= n {
        let v0 = _mm256_loadu_ps(p.add(i));
        let v1 = _mm256_loadu_ps(p.add(i + 8));
        let v2 = _mm256_loadu_ps(p.add(i + 16));
        let v3 = _mm256_loadu_ps(p.add(i + 24));
        bad0 = _mm256_or_si256(
            bad0,
            _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp), exp),
        );
        bad1 = _mm256_or_si256(
            bad1,
            _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v1), exp), exp),
        );
        bad2 = _mm256_or_si256(
            bad2,
            _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v2), exp), exp),
        );
        bad3 = _mm256_or_si256(
            bad3,
            _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v3), exp), exp),
        );
        vmax0 = _mm256_max_ps(vmax0, _mm256_andnot_ps(signbit, v0));
        vmax1 = _mm256_max_ps(vmax1, _mm256_andnot_ps(signbit, v1));
        vmax2 = _mm256_max_ps(vmax2, _mm256_andnot_ps(signbit, v2));
        vmax3 = _mm256_max_ps(vmax3, _mm256_andnot_ps(signbit, v3));
        i += 32;
    }
    // Drain a trailing partial block through lane 0 (same arithmetic).
    while i + 8 <= n {
        let v = _mm256_loadu_ps(p.add(i));
        let bits = _mm256_castps_si256(v);
        let cmp = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exp), exp);
        bad0 = _mm256_or_si256(bad0, cmp);
        vmax0 = _mm256_max_ps(vmax0, _mm256_andnot_ps(signbit, v));
        i += 8;
    }
    let vmax01 = _mm256_max_ps(vmax0, vmax1);
    let vmax23 = _mm256_max_ps(vmax2, vmax3);
    let vmax = _mm256_max_ps(vmax01, vmax23);
    let bad01 = _mm256_or_si256(bad0, bad1);
    let bad23 = _mm256_or_si256(bad2, bad3);
    let bad = _mm256_or_si256(bad01, bad23);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), vmax);
    let mut best = 0.0f32;
    for &v in &buf {
        if v > best {
            best = v;
        }
    }
    let mut ok = _mm256_movemask_epi8(bad) == 0;
    while i < n {
        let a = (*p.add(i)).abs();
        if a > best {
            best = a;
        }
        ok &= a.is_finite();
        i += 1;
    }
    (best, ok)
}

/// f64 twin of [`max_norm_checked_f32`]: 4-wide abs+max fold with
/// OR-accumulated per-lane exponent-bit comparison.
///
/// Four independent accumulator pairs (16 lanes/iter): a single
/// `vmax`/`bad` chain is latency-bound (~4 cycles per 4 lanes through
/// `max`). Four chains saturate load throughput instead, bringing the
/// fold to the box streaming floor (~19GB/s single-thread cold here —
/// memcpy itself; the 33MB dgetrf-2048 entry scan stays 1.6ms cold,
/// so further single-thread wins need fewer passes, not wider folds).
/// Reduction order is partition-dependent, but `max` over finite
/// lanes is exact in any order and OR-accumulation is order-free; the
/// NaN-input max stays unspecified per contract.
#[target_feature(enable = "avx2")]
pub unsafe fn max_norm_checked_f64(values: &[f64]) -> (f64, bool) {
    let signbit = _mm256_set1_pd(-0.0);
    let exp = _mm256_set1_epi64x(0x7FF0_0000_0000_0000u64 as i64);
    let mut vmax0 = _mm256_setzero_pd();
    let mut vmax1 = _mm256_setzero_pd();
    let mut vmax2 = _mm256_setzero_pd();
    let mut vmax3 = _mm256_setzero_pd();
    let mut bad0 = _mm256_setzero_si256();
    let mut bad1 = _mm256_setzero_si256();
    let mut bad2 = _mm256_setzero_si256();
    let mut bad3 = _mm256_setzero_si256();
    let n = values.len();
    let p = values.as_ptr();
    let mut i = 0;
    while i + 16 <= n {
        let v0 = _mm256_loadu_pd(p.add(i));
        let v1 = _mm256_loadu_pd(p.add(i + 4));
        let v2 = _mm256_loadu_pd(p.add(i + 8));
        let v3 = _mm256_loadu_pd(p.add(i + 12));
        bad0 = _mm256_or_si256(
            bad0,
            _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(v0), exp), exp),
        );
        bad1 = _mm256_or_si256(
            bad1,
            _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(v1), exp), exp),
        );
        bad2 = _mm256_or_si256(
            bad2,
            _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(v2), exp), exp),
        );
        bad3 = _mm256_or_si256(
            bad3,
            _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(v3), exp), exp),
        );
        vmax0 = _mm256_max_pd(vmax0, _mm256_andnot_pd(signbit, v0));
        vmax1 = _mm256_max_pd(vmax1, _mm256_andnot_pd(signbit, v1));
        vmax2 = _mm256_max_pd(vmax2, _mm256_andnot_pd(signbit, v2));
        vmax3 = _mm256_max_pd(vmax3, _mm256_andnot_pd(signbit, v3));
        i += 16;
    }
    // Drain a trailing partial block through lane 0 (same arithmetic).
    while i + 4 <= n {
        let v = _mm256_loadu_pd(p.add(i));
        let bits = _mm256_castpd_si256(v);
        let cmp = _mm256_cmpeq_epi64(_mm256_and_si256(bits, exp), exp);
        bad0 = _mm256_or_si256(bad0, cmp);
        vmax0 = _mm256_max_pd(vmax0, _mm256_andnot_pd(signbit, v));
        i += 4;
    }
    let vmax01 = _mm256_max_pd(vmax0, vmax1);
    let vmax23 = _mm256_max_pd(vmax2, vmax3);
    let vmax = _mm256_max_pd(vmax01, vmax23);
    let bad01 = _mm256_or_si256(bad0, bad1);
    let bad23 = _mm256_or_si256(bad2, bad3);
    let bad = _mm256_or_si256(bad01, bad23);
    let mut buf = [0.0f64; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), vmax);
    let mut best = 0.0f64;
    for &v in &buf {
        if v > best {
            best = v;
        }
    }
    let mut ok = _mm256_movemask_epi8(bad) == 0;
    while i < n {
        let a = (*p.add(i)).abs();
        if a > best {
            best = a;
        }
        ok &= a.is_finite();
        i += 1;
    }
    (best, ok)
}

// ---------- complex pivot scan / ger ----------

/// Complex64 pivot scan: `max base[(first+i)*stride].norm()` (true modulus)
/// over `len` rows, first maximum wins. Gather-free like
/// [`pivot_scan_f64`]: each complex value needs its `re`/`im` pair, so one
/// iteration loads 4 strided pairs into `re`/`im` vectors and folds
/// `sqrt(re²+im²)` through the same blend/index machinery.
///
/// Exactness: the metric is `sqrt(re*re + im*im)` while the scalar
/// reference uses `hypot` — identical except sub-ulp ties and
/// overflow/underflow edges (overflow reads as +inf, i.e. still wins the
/// scan, which is the correct selection). The index contract matches
/// [`pivot_scan_f64`] bit-for-bit otherwise: first maximum wins, a
/// non-finite initial pivot keeps its row, later non-finite values never
/// win. Entry validation guarantees finite input in the LU use.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pivot_scan_c64(
    base: &[Complex64],
    stride: usize,
    first: usize,
    len: usize,
) -> (usize, f64) {
    debug_assert!(len >= 1);
    let p = base.as_ptr() as *const f64;
    let init_re = *p.add(first * stride * 2);
    let init_im = *p.add(first * stride * 2 + 1);
    let init = init_re.hypot(init_im);
    if !init.is_finite() && init != f64::NEG_INFINITY {
        return (first, init);
    }
    let mut vmax = _mm256_set1_pd(f64::NEG_INFINITY);
    let r0 = first as i64;
    let mut imax = _mm256_set_epi64x(r0 + 3, r0 + 2, r0 + 1, r0);
    let blocks = len / 4;
    let mut b = 0;
    while b < blocks {
        let r = r0 + b as i64 * 4;
        let re = _mm256_setr_pd(
            *p.add(r as usize * stride * 2),
            *p.add((r as usize + 1) * stride * 2),
            *p.add((r as usize + 2) * stride * 2),
            *p.add((r as usize + 3) * stride * 2),
        );
        let im = _mm256_setr_pd(
            *p.add(r as usize * stride * 2 + 1),
            *p.add((r as usize + 1) * stride * 2 + 1),
            *p.add((r as usize + 2) * stride * 2 + 1),
            *p.add((r as usize + 3) * stride * 2 + 1),
        );
        let v = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(re, re), _mm256_mul_pd(im, im)));
        let cmp = _mm256_cmp_pd(v, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_pd(vmax, v, cmp);
        let idx = _mm256_set_epi64x(r + 3, r + 2, r + 1, r);
        imax = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(imax),
            _mm256_castsi256_ps(idx),
            _mm256_castpd_ps(cmp),
        ));
        b += 1;
    }
    let mut vals = [f64::NEG_INFINITY; 4];
    let mut idxs = [0i64; 4];
    _mm256_storeu_pd(vals.as_mut_ptr(), vmax);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, imax);
    let mut best_row = first;
    let mut best_val = init;
    // Same tie-break rationale as `pivot_scan_f64`: `vals` holds finite
    // magnitudes here (NaN never wins a lane, non-finite `init` returned
    // early), so `==` is an exact tie test.
    for lane in 0..4 {
        if vals[lane] > best_val || (vals[lane] == best_val && (idxs[lane] as usize) < best_row) {
            best_val = vals[lane];
            best_row = idxs[lane] as usize;
        }
    }
    // Tail rows (strict `>`, first wins) — also covers len < 4. Scalar
    // `hypot` here, matching the reference metric exactly.
    let mut i = first + blocks * 4;
    let end = first + len;
    while i < end {
        let val = (*p.add(i * stride * 2)).hypot(*p.add(i * stride * 2 + 1));
        if val > best_val {
            best_val = val;
            best_row = i;
        }
        i += 1;
    }
    (best_row, best_val)
}

/// Complex32 twin of [`pivot_scan_c64`]: 8 values per iteration, same
/// index contract and metric caveat.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pivot_scan_c32(
    base: &[Complex32],
    stride: usize,
    first: usize,
    len: usize,
) -> (usize, f32) {
    debug_assert!(len >= 1);
    let p = base.as_ptr() as *const f32;
    let init_re = *p.add(first * stride * 2);
    let init_im = *p.add(first * stride * 2 + 1);
    let init = init_re.hypot(init_im);
    if !init.is_finite() && init != f32::NEG_INFINITY {
        return (first, init);
    }
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let r0 = first as i32;
    let mut imax = _mm256_set_epi32(r0 + 7, r0 + 6, r0 + 5, r0 + 4, r0 + 3, r0 + 2, r0 + 1, r0);
    let blocks = len / 8;
    let mut b = 0;
    while b < blocks {
        let r = r0 + b as i32 * 8;
        let re = _mm256_setr_ps(
            *p.add(r as usize * stride * 2),
            *p.add((r as usize + 1) * stride * 2),
            *p.add((r as usize + 2) * stride * 2),
            *p.add((r as usize + 3) * stride * 2),
            *p.add((r as usize + 4) * stride * 2),
            *p.add((r as usize + 5) * stride * 2),
            *p.add((r as usize + 6) * stride * 2),
            *p.add((r as usize + 7) * stride * 2),
        );
        let im = _mm256_setr_ps(
            *p.add(r as usize * stride * 2 + 1),
            *p.add((r as usize + 1) * stride * 2 + 1),
            *p.add((r as usize + 2) * stride * 2 + 1),
            *p.add((r as usize + 3) * stride * 2 + 1),
            *p.add((r as usize + 4) * stride * 2 + 1),
            *p.add((r as usize + 5) * stride * 2 + 1),
            *p.add((r as usize + 6) * stride * 2 + 1),
            *p.add((r as usize + 7) * stride * 2 + 1),
        );
        let v = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(re, re), _mm256_mul_ps(im, im)));
        let cmp = _mm256_cmp_ps(v, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_ps(vmax, v, cmp);
        let idx = _mm256_set_epi32(r + 7, r + 6, r + 5, r + 4, r + 3, r + 2, r + 1, r);
        imax = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(imax),
            _mm256_castsi256_ps(idx),
            cmp,
        ));
        b += 1;
    }
    let mut vals = [f32::NEG_INFINITY; 8];
    let mut idxs = [0i32; 8];
    _mm256_storeu_ps(vals.as_mut_ptr(), vmax);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, imax);
    let mut best_row = first;
    let mut best_val = init;
    for lane in 0..8 {
        if vals[lane] > best_val || (vals[lane] == best_val && (idxs[lane] as usize) < best_row) {
            best_val = vals[lane];
            best_row = idxs[lane] as usize;
        }
    }
    let mut i = first + blocks * 8;
    let end = first + len;
    while i < end {
        let val = (*p.add(i * stride * 2)).hypot(*p.add(i * stride * 2 + 1));
        if val > best_val {
            best_val = val;
            best_row = i;
        }
        i += 1;
    }
    (best_row, best_val)
}

/// Fused complex rank-1 update `block[i*bs+j] += alpha*mults[i*ms] *
/// pivot[j]` (Complex64). Same single-dispatch motivation as
/// [`ger_update_f64`]: the LU leaf issues one tiny axpy per row per panel
/// column otherwise.
///
/// Vectorization: one `__m256d` holds 2 complex values `[re0, im0, re1,
/// im1]`. With `m = alpha*mult` broadcast to `vmr`/`vmi`, the update delta
/// is `d = vmr*p +/− vmi*permute(p)` where the permute swaps each pair
/// (`[im0, re0, im1, re1]`) and the sign mask negates the even lanes:
/// `d = [mr*pr0 − mi*pi0, mr*pi0 + mi*pr0, ...]`. Two rows share one pivot
/// load (register pressure: ~11 live vectors, fits).
///
/// Rounding: `m` uses the scalar complex multiply (same as the reference),
/// then 2 roundings for the product lanes and 1 for the accumulate — the
/// same count and structure as the scalar axpy loop, but not
/// bit-guaranteed (NaN payloads aside, finite results can differ in the
/// last ulp on adversarial cancellation). LU residuals are unaffected.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ger_update_c64(
    alpha: Complex64,
    mults: &[Complex64],
    mult_stride: usize,
    pivot: &[Complex64],
    block: &mut [Complex64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    let signmask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
    let pp = pivot.as_ptr() as *const f64;
    let mut i = 0;
    while i + 2 <= rows {
        let m0 = alpha * *mults.as_ptr().add(i * mult_stride);
        let m1 = alpha * *mults.as_ptr().add((i + 1) * mult_stride);
        let vmr0 = _mm256_set1_pd(m0.re);
        let vmi0 = _mm256_set1_pd(m0.im);
        let vmr1 = _mm256_set1_pd(m1.re);
        let vmi1 = _mm256_set1_pd(m1.im);
        let r0 = block.as_mut_ptr().add(i * block_stride) as *mut f64;
        let r1 = block.as_mut_ptr().add((i + 1) * block_stride) as *mut f64;
        let mut j = 0;
        while j + 2 <= cols {
            let pv = _mm256_loadu_pd(pp.add(j * 2));
            let pvs = _mm256_permute_pd(pv, 0b0101);
            let b0 = _mm256_loadu_pd(r0.add(j * 2));
            let b1 = _mm256_loadu_pd(r1.add(j * 2));
            let d0 = _mm256_add_pd(
                _mm256_mul_pd(vmr0, pv),
                _mm256_xor_pd(_mm256_mul_pd(vmi0, pvs), signmask),
            );
            let d1 = _mm256_add_pd(
                _mm256_mul_pd(vmr1, pv),
                _mm256_xor_pd(_mm256_mul_pd(vmi1, pvs), signmask),
            );
            _mm256_storeu_pd(r0.add(j * 2), _mm256_add_pd(b0, d0));
            _mm256_storeu_pd(r1.add(j * 2), _mm256_add_pd(b1, d1));
            j += 2;
        }
        while j < cols {
            let m0s = alpha * *mults.as_ptr().add(i * mult_stride);
            let m1s = alpha * *mults.as_ptr().add((i + 1) * mult_stride);
            let p0 = *pivot.as_ptr().add(j);
            let b0r = r0.add(j * 2);
            let b1r = r1.add(j * 2);
            let d0r = m0s.re * p0.re - m0s.im * p0.im;
            let d0i = m0s.re * p0.im + m0s.im * p0.re;
            let d1r = m1s.re * p0.re - m1s.im * p0.im;
            let d1i = m1s.re * p0.im + m1s.im * p0.re;
            *b0r += d0r;
            *b0r.add(1) += d0i;
            *b1r += d1r;
            *b1r.add(1) += d1i;
            j += 1;
        }
        i += 2;
    }
    while i < rows {
        let m = alpha * *mults.as_ptr().add(i * mult_stride);
        let vmr = _mm256_set1_pd(m.re);
        let vmi = _mm256_set1_pd(m.im);
        let row = block.as_mut_ptr().add(i * block_stride) as *mut f64;
        let mut j = 0;
        while j + 8 <= cols {
            for lane in 0..4 {
                let off = (j + lane * 2) * 2;
                let pv = _mm256_loadu_pd(pp.add(off));
                let pvs = _mm256_permute_pd(pv, 0b0101);
                let bv = _mm256_loadu_pd(row.add(off));
                let d = _mm256_add_pd(
                    _mm256_mul_pd(vmr, pv),
                    _mm256_xor_pd(_mm256_mul_pd(vmi, pvs), signmask),
                );
                _mm256_storeu_pd(row.add(off), _mm256_add_pd(bv, d));
            }
            j += 8;
        }
        while j + 2 <= cols {
            let pv = _mm256_loadu_pd(pp.add(j * 2));
            let pvs = _mm256_permute_pd(pv, 0b0101);
            let bv = _mm256_loadu_pd(row.add(j * 2));
            let d = _mm256_add_pd(
                _mm256_mul_pd(vmr, pv),
                _mm256_xor_pd(_mm256_mul_pd(vmi, pvs), signmask),
            );
            _mm256_storeu_pd(row.add(j * 2), _mm256_add_pd(bv, d));
            j += 2;
        }
        while j < cols {
            let p0 = *pivot.as_ptr().add(j);
            let dr = m.re * p0.re - m.im * p0.im;
            let di = m.re * p0.im + m.im * p0.re;
            *row.add(j * 2) += dr;
            *row.add(j * 2 + 1) += di;
            j += 1;
        }
        i += 1;
    }
}

/// Complex32 twin of [`ger_update_c64`]: 4 values per vector, pair swap
/// via `_mm256_permute_ps(_, 0xB1)`, same rounding caveat.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ger_update_c32(
    alpha: Complex32,
    mults: &[Complex32],
    mult_stride: usize,
    pivot: &[Complex32],
    block: &mut [Complex32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    let signmask = _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    let pp = pivot.as_ptr() as *const f32;
    let mut i = 0;
    while i + 2 <= rows {
        let m0 = alpha * *mults.as_ptr().add(i * mult_stride);
        let m1 = alpha * *mults.as_ptr().add((i + 1) * mult_stride);
        let vmr0 = _mm256_set1_ps(m0.re);
        let vmi0 = _mm256_set1_ps(m0.im);
        let vmr1 = _mm256_set1_ps(m1.re);
        let vmi1 = _mm256_set1_ps(m1.im);
        let r0 = block.as_mut_ptr().add(i * block_stride) as *mut f32;
        let r1 = block.as_mut_ptr().add((i + 1) * block_stride) as *mut f32;
        let mut j = 0;
        while j + 4 <= cols {
            let pv = _mm256_loadu_ps(pp.add(j * 2));
            let pvs = _mm256_permute_ps(pv, 0xB1);
            let b0 = _mm256_loadu_ps(r0.add(j * 2));
            let b1 = _mm256_loadu_ps(r1.add(j * 2));
            let d0 = _mm256_add_ps(
                _mm256_mul_ps(vmr0, pv),
                _mm256_xor_ps(_mm256_mul_ps(vmi0, pvs), signmask),
            );
            let d1 = _mm256_add_ps(
                _mm256_mul_ps(vmr1, pv),
                _mm256_xor_ps(_mm256_mul_ps(vmi1, pvs), signmask),
            );
            _mm256_storeu_ps(r0.add(j * 2), _mm256_add_ps(b0, d0));
            _mm256_storeu_ps(r1.add(j * 2), _mm256_add_ps(b1, d1));
            j += 4;
        }
        while j < cols {
            let m0s = alpha * *mults.as_ptr().add(i * mult_stride);
            let m1s = alpha * *mults.as_ptr().add((i + 1) * mult_stride);
            let p0 = *pivot.as_ptr().add(j);
            let b0r = r0.add(j * 2);
            let b1r = r1.add(j * 2);
            *b0r += m0s.re * p0.re - m0s.im * p0.im;
            *b0r.add(1) += m0s.re * p0.im + m0s.im * p0.re;
            *b1r += m1s.re * p0.re - m1s.im * p0.im;
            *b1r.add(1) += m1s.re * p0.im + m1s.im * p0.re;
            j += 1;
        }
        i += 2;
    }
    while i < rows {
        let m = alpha * *mults.as_ptr().add(i * mult_stride);
        let vmr = _mm256_set1_ps(m.re);
        let vmi = _mm256_set1_ps(m.im);
        let row = block.as_mut_ptr().add(i * block_stride) as *mut f32;
        let mut j = 0;
        while j + 16 <= cols {
            for lane in 0..4 {
                let off = (j + lane * 4) * 2;
                let pv = _mm256_loadu_ps(pp.add(off));
                let pvs = _mm256_permute_ps(pv, 0xB1);
                let bv = _mm256_loadu_ps(row.add(off));
                let d = _mm256_add_ps(
                    _mm256_mul_ps(vmr, pv),
                    _mm256_xor_ps(_mm256_mul_ps(vmi, pvs), signmask),
                );
                _mm256_storeu_ps(row.add(off), _mm256_add_ps(bv, d));
            }
            j += 16;
        }
        while j + 4 <= cols {
            let pv = _mm256_loadu_ps(pp.add(j * 2));
            let pvs = _mm256_permute_ps(pv, 0xB1);
            let bv = _mm256_loadu_ps(row.add(j * 2));
            let d = _mm256_add_ps(
                _mm256_mul_ps(vmr, pv),
                _mm256_xor_ps(_mm256_mul_ps(vmi, pvs), signmask),
            );
            _mm256_storeu_ps(row.add(j * 2), _mm256_add_ps(bv, d));
            j += 4;
        }
        while j < cols {
            let p0 = *pivot.as_ptr().add(j);
            *row.add(j * 2) += m.re * p0.re - m.im * p0.im;
            *row.add(j * 2 + 1) += m.re * p0.im + m.im * p0.re;
            j += 1;
        }
        i += 1;
    }
}

/// Complex64 entry-validation fold `(max norm, all_finite)` in one pass:
/// vectorized twin of [`scalar::max_norm_checked_c64`](crate::cpu::blas1::scalar::max_norm_checked_c64)
/// for the LU entry scan. Four values per iteration: two loads cover the 8
/// lanes, `hadd` folds each `(re, im)` pair to `re²+im²`, `sqrt` gives the
/// modulus folded into `vmax`, and the exponent-bit test ORs any
/// non-finite lane into `bad`.
///
/// Exactness: the flag is bit-exact; the max uses `sqrt(re²+im²)` vs the
/// scalar `hypot` (identical except sub-ulp ties and the >1e308-component
/// overflow edge, where it reads +inf — the factorization itself overflows
/// on such input regardless, so only which error fires changes). NaN lanes
/// never poison `vmax` (`max_pd` propagates the finite side) while still
/// tripping `bad`, so the caller's rescan contract holds.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn max_norm_checked_c64(values: &[Complex64]) -> (f64, bool) {
    let exp = _mm256_set1_epi64x(0x7FF0_0000_0000_0000u64 as i64);
    let mut vmax = _mm256_setzero_pd();
    let mut bad = _mm256_setzero_si256();
    let n = values.len();
    let p = values.as_ptr() as *const f64;
    let mut i = 0;
    while i + 4 <= n {
        let v0 = _mm256_loadu_pd(p.add(i * 2));
        let v1 = _mm256_loadu_pd(p.add(i * 2 + 4));
        let b0 = _mm256_castpd_si256(v0);
        let b1 = _mm256_castpd_si256(v1);
        bad = _mm256_or_si256(
            bad,
            _mm256_or_si256(
                _mm256_cmpeq_epi64(_mm256_and_si256(b0, exp), exp),
                _mm256_cmpeq_epi64(_mm256_and_si256(b1, exp), exp),
            ),
        );
        let sq = _mm256_hadd_pd(_mm256_mul_pd(v0, v0), _mm256_mul_pd(v1, v1));
        vmax = _mm256_max_pd(vmax, _mm256_sqrt_pd(sq));
        i += 4;
    }
    let mut buf = [0.0f64; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), vmax);
    let mut best = 0.0f64;
    for &v in &buf {
        if v > best {
            best = v;
        }
    }
    let mut ok = _mm256_movemask_epi8(bad) == 0;
    while i < n {
        let m = (*p.add(i * 2)).hypot(*p.add(i * 2 + 1));
        if m > best {
            best = m;
        }
        ok &= m.is_finite();
        i += 1;
    }
    (best, ok)
}

/// Complex32 twin of [`max_norm_checked_c64`]: 8 values per iteration
/// (four loads, two `hadd`s), same flag-exact / max-near-exact contract.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn max_norm_checked_c32(values: &[Complex32]) -> (f32, bool) {
    let exp = _mm256_set1_epi32(0x7F80_0000u32 as i32);
    let mut vmax = _mm256_setzero_ps();
    let mut bad = _mm256_setzero_si256();
    let n = values.len();
    let p = values.as_ptr() as *const f32;
    let mut i = 0;
    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(p.add(i * 2));
        let v1 = _mm256_loadu_ps(p.add(i * 2 + 8));
        let v2 = _mm256_loadu_ps(p.add(i * 2 + 16));
        let v3 = _mm256_loadu_ps(p.add(i * 2 + 24));
        let t0 = _mm256_castps_si256(v0);
        let t1 = _mm256_castps_si256(v1);
        let t2 = _mm256_castps_si256(v2);
        let t3 = _mm256_castps_si256(v3);
        let c0 = _mm256_cmpeq_epi32(_mm256_and_si256(t0, exp), exp);
        let c1 = _mm256_cmpeq_epi32(_mm256_and_si256(t1, exp), exp);
        let c2 = _mm256_cmpeq_epi32(_mm256_and_si256(t2, exp), exp);
        let c3 = _mm256_cmpeq_epi32(_mm256_and_si256(t3, exp), exp);
        bad = _mm256_or_si256(
            bad,
            _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3)),
        );
        let s0 = _mm256_mul_ps(v0, v0);
        let s1 = _mm256_mul_ps(v1, v1);
        let s2 = _mm256_mul_ps(v2, v2);
        let s3 = _mm256_mul_ps(v3, v3);
        // Each `hadd` folds 8 squared lanes into 8 moduli (m0..m7,
        // m8..m15); both fold into `vmax` separately.
        let h01 = _mm256_hadd_ps(s0, s1);
        let h23 = _mm256_hadd_ps(s2, s3);
        vmax = _mm256_max_ps(vmax, _mm256_sqrt_ps(h01));
        vmax = _mm256_max_ps(vmax, _mm256_sqrt_ps(h23));
        i += 16;
    }
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), vmax);
    let mut best = 0.0f32;
    for &v in &buf {
        if v > best {
            best = v;
        }
    }
    let mut ok = _mm256_movemask_epi8(bad) == 0;
    while i < n {
        let m = (*p.add(i * 2)).hypot(*p.add(i * 2 + 1));
        if m > best {
            best = m;
        }
        ok &= m.is_finite();
        i += 1;
    }
    (best, ok)
}

/// 4M epilogue row `C := α·(R + i·I) + β·C` (Complex64): vectorized twin of
/// [`scalar::apply_ab_c64_row`](crate::cpu::blas1::scalar::apply_ab_c64_row)
/// for the complex-GEMM combine pass (the Schur-update epilogue). Four
/// values per iteration; the caller runs it per contiguous output row so
/// strided `C` (`ldc > n`) vectorizes too.
///
/// Exactness: bitwise-identical to the scalar reference. The α half uses
/// separate multiply then add/sub (same two roundings as the scalar, NOT a
/// fused contraction), the β half reproduces the scalar association
/// (`(ab + br·c) −/+ bi·c`) on interleaved lanes via the pair-swap +
/// sign-mask trick, and tails run the scalar reference itself.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_ab_c64_row(
    r: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    let n = c.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(imag.len(), n);
    let rp = r.as_ptr();
    let ip = imag.as_ptr();
    let cp = c.as_mut_ptr() as *mut f64;
    let var = _mm256_set1_pd(alpha.re);
    let vai = _mm256_set1_pd(alpha.im);
    let signmask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
    if beta_zero {
        let mut j = 0;
        while j + 4 <= n {
            let rv = _mm256_loadu_pd(rp.add(j));
            let iv = _mm256_loadu_pd(ip.add(j));
            let ab_re = _mm256_sub_pd(_mm256_mul_pd(var, rv), _mm256_mul_pd(vai, iv));
            let ab_im = _mm256_add_pd(_mm256_mul_pd(var, iv), _mm256_mul_pd(vai, rv));
            // unpack splits elements even/odd across the 128-bit lanes;
            // permute2f128 resequence to sequential pairs.
            let lo = _mm256_unpacklo_pd(ab_re, ab_im);
            let hi = _mm256_unpackhi_pd(ab_re, ab_im);
            _mm256_storeu_pd(cp.add(j * 2), _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(cp.add(j * 2 + 4), _mm256_permute2f128_pd(lo, hi, 0x31));
            j += 4;
        }
        if j < n {
            super::scalar::apply_ab_c64_row(
                &r[j..],
                &imag[j..],
                &mut c[j..],
                alpha,
                beta,
                beta_zero,
            );
        }
        return;
    }
    let vbr = _mm256_set1_pd(beta.re);
    let vbi = _mm256_set1_pd(beta.im);
    let mut j = 0;
    while j + 4 <= n {
        let rv = _mm256_loadu_pd(rp.add(j));
        let iv = _mm256_loadu_pd(ip.add(j));
        let ab_re = _mm256_sub_pd(_mm256_mul_pd(var, rv), _mm256_mul_pd(vai, iv));
        let ab_im = _mm256_add_pd(_mm256_mul_pd(var, iv), _mm256_mul_pd(vai, rv));
        let lo = _mm256_unpacklo_pd(ab_re, ab_im);
        let hi = _mm256_unpackhi_pd(ab_re, ab_im);
        let ab0 = _mm256_permute2f128_pd(lo, hi, 0x20);
        let ab1 = _mm256_permute2f128_pd(lo, hi, 0x31);
        let cv0 = _mm256_loadu_pd(cp.add(j * 2));
        let cv1 = _mm256_loadu_pd(cp.add(j * 2 + 4));
        let cp0 = _mm256_permute_pd(cv0, 0b0101);
        let cp1 = _mm256_permute_pd(cv1, 0b0101);
        // (ab + br·c) −/+ bi·cswap: scalar association, sign mask picks
        // the − on real lanes and + on imaginary lanes.
        let o0 = _mm256_add_pd(
            _mm256_add_pd(ab0, _mm256_mul_pd(vbr, cv0)),
            _mm256_xor_pd(_mm256_mul_pd(vbi, cp0), signmask),
        );
        let o1 = _mm256_add_pd(
            _mm256_add_pd(ab1, _mm256_mul_pd(vbr, cv1)),
            _mm256_xor_pd(_mm256_mul_pd(vbi, cp1), signmask),
        );
        _mm256_storeu_pd(cp.add(j * 2), o0);
        _mm256_storeu_pd(cp.add(j * 2 + 4), o1);
        j += 4;
    }
    if j < n {
        super::scalar::apply_ab_c64_row(&r[j..], &imag[j..], &mut c[j..], alpha, beta, beta_zero);
    }
}

/// 3M epilogue row: [`apply_ab_c64_row`] with `Re = R - Q`,
/// `Im = I - R - Q` fused ahead of the α/β math (+1 load, +2 subs per
/// 4 elements over the 4M kernel). Same unpack/permute interleave and
/// exactness contract (scalar association); tail calls the scalar twin.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_ab3_c64_row(
    r: &[f64],
    q: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    let n = c.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(q.len(), n);
    debug_assert_eq!(imag.len(), n);
    let rp = r.as_ptr();
    let qp = q.as_ptr();
    let ip = imag.as_ptr();
    let cp = c.as_mut_ptr() as *mut f64;
    let var = _mm256_set1_pd(alpha.re);
    let vai = _mm256_set1_pd(alpha.im);
    let signmask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
    if beta_zero {
        let mut j = 0;
        while j + 4 <= n {
            let rv0 = _mm256_loadu_pd(rp.add(j));
            let qv = _mm256_loadu_pd(qp.add(j));
            let iv0 = _mm256_loadu_pd(ip.add(j));
            let rv = _mm256_sub_pd(rv0, qv);
            let iv = _mm256_sub_pd(_mm256_sub_pd(iv0, rv0), qv);
            let ab_re = _mm256_sub_pd(_mm256_mul_pd(var, rv), _mm256_mul_pd(vai, iv));
            let ab_im = _mm256_add_pd(_mm256_mul_pd(var, iv), _mm256_mul_pd(vai, rv));
            let lo = _mm256_unpacklo_pd(ab_re, ab_im);
            let hi = _mm256_unpackhi_pd(ab_re, ab_im);
            _mm256_storeu_pd(cp.add(j * 2), _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(cp.add(j * 2 + 4), _mm256_permute2f128_pd(lo, hi, 0x31));
            j += 4;
        }
        if j < n {
            super::scalar::apply_ab3_c64_row(
                &r[j..],
                &q[j..],
                &imag[j..],
                &mut c[j..],
                alpha,
                beta,
                beta_zero,
            );
        }
        return;
    }
    let vbr = _mm256_set1_pd(beta.re);
    let vbi = _mm256_set1_pd(beta.im);
    let mut j = 0;
    while j + 4 <= n {
        let rv0 = _mm256_loadu_pd(rp.add(j));
        let qv = _mm256_loadu_pd(qp.add(j));
        let iv0 = _mm256_loadu_pd(ip.add(j));
        let rv = _mm256_sub_pd(rv0, qv);
        let iv = _mm256_sub_pd(_mm256_sub_pd(iv0, rv0), qv);
        let ab_re = _mm256_sub_pd(_mm256_mul_pd(var, rv), _mm256_mul_pd(vai, iv));
        let ab_im = _mm256_add_pd(_mm256_mul_pd(var, iv), _mm256_mul_pd(vai, rv));
        let lo = _mm256_unpacklo_pd(ab_re, ab_im);
        let hi = _mm256_unpackhi_pd(ab_re, ab_im);
        let ab0 = _mm256_permute2f128_pd(lo, hi, 0x20);
        let ab1 = _mm256_permute2f128_pd(lo, hi, 0x31);
        let cv0 = _mm256_loadu_pd(cp.add(j * 2));
        let cv1 = _mm256_loadu_pd(cp.add(j * 2 + 4));
        let cp0 = _mm256_permute_pd(cv0, 0b0101);
        let cp1 = _mm256_permute_pd(cv1, 0b0101);
        let o0 = _mm256_add_pd(
            _mm256_add_pd(ab0, _mm256_mul_pd(vbr, cv0)),
            _mm256_xor_pd(_mm256_mul_pd(vbi, cp0), signmask),
        );
        let o1 = _mm256_add_pd(
            _mm256_add_pd(ab1, _mm256_mul_pd(vbr, cv1)),
            _mm256_xor_pd(_mm256_mul_pd(vbi, cp1), signmask),
        );
        _mm256_storeu_pd(cp.add(j * 2), o0);
        _mm256_storeu_pd(cp.add(j * 2 + 4), o1);
        j += 4;
    }
    if j < n {
        super::scalar::apply_ab3_c64_row(
            &r[j..],
            &q[j..],
            &imag[j..],
            &mut c[j..],
            alpha,
            beta,
            beta_zero,
        );
    }
}

/// Complex32 twin of [`apply_ab_c64_row`]: 8 values per iteration, pair
/// swap via `_mm256_permute_ps(_, 0xB1)`, same exactness contract.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_ab_c32_row(
    r: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    let n = c.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(imag.len(), n);
    let rp = r.as_ptr();
    let ip = imag.as_ptr();
    let cp = c.as_mut_ptr() as *mut f32;
    let var = _mm256_set1_ps(alpha.re);
    let vai = _mm256_set1_ps(alpha.im);
    let signmask = _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    if beta_zero {
        let mut j = 0;
        while j + 8 <= n {
            let rv = _mm256_loadu_ps(rp.add(j));
            let iv = _mm256_loadu_ps(ip.add(j));
            let ab_re = _mm256_sub_ps(_mm256_mul_ps(var, rv), _mm256_mul_ps(vai, iv));
            let ab_im = _mm256_add_ps(_mm256_mul_ps(var, iv), _mm256_mul_ps(vai, rv));
            let lo = _mm256_unpacklo_ps(ab_re, ab_im);
            let hi = _mm256_unpackhi_ps(ab_re, ab_im);
            _mm256_storeu_ps(cp.add(j * 2), _mm256_permute2f128_ps(lo, hi, 0x20));
            _mm256_storeu_ps(cp.add(j * 2 + 8), _mm256_permute2f128_ps(lo, hi, 0x31));
            j += 8;
        }
        if j < n {
            super::scalar::apply_ab_c32_row(
                &r[j..],
                &imag[j..],
                &mut c[j..],
                alpha,
                beta,
                beta_zero,
            );
        }
        return;
    }
    let vbr = _mm256_set1_ps(beta.re);
    let vbi = _mm256_set1_ps(beta.im);
    let mut j = 0;
    while j + 8 <= n {
        let rv = _mm256_loadu_ps(rp.add(j));
        let iv = _mm256_loadu_ps(ip.add(j));
        let ab_re = _mm256_sub_ps(_mm256_mul_ps(var, rv), _mm256_mul_ps(vai, iv));
        let ab_im = _mm256_add_ps(_mm256_mul_ps(var, iv), _mm256_mul_ps(vai, rv));
        let lo = _mm256_unpacklo_ps(ab_re, ab_im);
        let hi = _mm256_unpackhi_ps(ab_re, ab_im);
        let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
        let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
        let cv0 = _mm256_loadu_ps(cp.add(j * 2));
        let cv1 = _mm256_loadu_ps(cp.add(j * 2 + 8));
        let cp0 = _mm256_permute_ps(cv0, 0xB1);
        let cp1 = _mm256_permute_ps(cv1, 0xB1);
        let o0 = _mm256_add_ps(
            _mm256_add_ps(ab0, _mm256_mul_ps(vbr, cv0)),
            _mm256_xor_ps(_mm256_mul_ps(vbi, cp0), signmask),
        );
        let o1 = _mm256_add_ps(
            _mm256_add_ps(ab1, _mm256_mul_ps(vbr, cv1)),
            _mm256_xor_ps(_mm256_mul_ps(vbi, cp1), signmask),
        );
        _mm256_storeu_ps(cp.add(j * 2), o0);
        _mm256_storeu_ps(cp.add(j * 2 + 8), o1);
        j += 8;
    }
    if j < n {
        super::scalar::apply_ab_c32_row(&r[j..], &imag[j..], &mut c[j..], alpha, beta, beta_zero);
    }
}

/// 3M epilogue row (Complex32 twin of [`apply_ab3_c64_row`]): 8 values
/// per iteration, `Re = R - Q`, `Im = I - R - Q` fused ahead of the
/// α/β math. Same exactness contract; tail calls the scalar twin.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_ab3_c32_row(
    r: &[f32],
    q: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    let n = c.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(q.len(), n);
    debug_assert_eq!(imag.len(), n);
    let rp = r.as_ptr();
    let qp = q.as_ptr();
    let ip = imag.as_ptr();
    let cp = c.as_mut_ptr() as *mut f32;
    let var = _mm256_set1_ps(alpha.re);
    let vai = _mm256_set1_ps(alpha.im);
    let signmask = _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    if beta_zero {
        let mut j = 0;
        while j + 8 <= n {
            let rv0 = _mm256_loadu_ps(rp.add(j));
            let qv = _mm256_loadu_ps(qp.add(j));
            let iv0 = _mm256_loadu_ps(ip.add(j));
            let rv = _mm256_sub_ps(rv0, qv);
            let iv = _mm256_sub_ps(_mm256_sub_ps(iv0, rv0), qv);
            let ab_re = _mm256_sub_ps(_mm256_mul_ps(var, rv), _mm256_mul_ps(vai, iv));
            let ab_im = _mm256_add_ps(_mm256_mul_ps(var, iv), _mm256_mul_ps(vai, rv));
            let lo = _mm256_unpacklo_ps(ab_re, ab_im);
            let hi = _mm256_unpackhi_ps(ab_re, ab_im);
            _mm256_storeu_ps(cp.add(j * 2), _mm256_permute2f128_ps(lo, hi, 0x20));
            _mm256_storeu_ps(cp.add(j * 2 + 8), _mm256_permute2f128_ps(lo, hi, 0x31));
            j += 8;
        }
        if j < n {
            super::scalar::apply_ab3_c32_row(
                &r[j..],
                &q[j..],
                &imag[j..],
                &mut c[j..],
                alpha,
                beta,
                beta_zero,
            );
        }
        return;
    }
    let vbr = _mm256_set1_ps(beta.re);
    let vbi = _mm256_set1_ps(beta.im);
    let mut j = 0;
    while j + 8 <= n {
        let rv0 = _mm256_loadu_ps(rp.add(j));
        let qv = _mm256_loadu_ps(qp.add(j));
        let iv0 = _mm256_loadu_ps(ip.add(j));
        let rv = _mm256_sub_ps(rv0, qv);
        let iv = _mm256_sub_ps(_mm256_sub_ps(iv0, rv0), qv);
        let ab_re = _mm256_sub_ps(_mm256_mul_ps(var, rv), _mm256_mul_ps(vai, iv));
        let ab_im = _mm256_add_ps(_mm256_mul_ps(var, iv), _mm256_mul_ps(vai, rv));
        let lo = _mm256_unpacklo_ps(ab_re, ab_im);
        let hi = _mm256_unpackhi_ps(ab_re, ab_im);
        let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
        let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
        let cv0 = _mm256_loadu_ps(cp.add(j * 2));
        let cv1 = _mm256_loadu_ps(cp.add(j * 2 + 8));
        let cp0 = _mm256_permute_ps(cv0, 0xB1);
        let cp1 = _mm256_permute_ps(cv1, 0xB1);
        let o0 = _mm256_add_ps(
            _mm256_add_ps(ab0, _mm256_mul_ps(vbr, cv0)),
            _mm256_xor_ps(_mm256_mul_ps(vbi, cp0), signmask),
        );
        let o1 = _mm256_add_ps(
            _mm256_add_ps(ab1, _mm256_mul_ps(vbr, cv1)),
            _mm256_xor_ps(_mm256_mul_ps(vbi, cp1), signmask),
        );
        _mm256_storeu_ps(cp.add(j * 2), o0);
        _mm256_storeu_ps(cp.add(j * 2 + 8), o1);
        j += 8;
    }
    if j < n {
        super::scalar::apply_ab3_c32_row(
            &r[j..],
            &q[j..],
            &imag[j..],
            &mut c[j..],
            alpha,
            beta,
            beta_zero,
        );
    }
}

/// Fused `dlassq` row accumulation (f64): fold one row of magnitudes
/// into per-column `(scale, sums)` with the exact scalar update
/// equations. Vectorizes ACROSS independent columns (4-wide), keeping
/// each column's op sequence in row order — bitwise identical to
/// [`super::scalar::dlassq_row_update_f64`], including NaN/±inf/zero
/// lanes (masked out of the state update) and the two-rounding
/// `1 + (sums*ratio)*ratio` / `sums + ratio*ratio` forms (separate
/// mul/add, never FMA). Returns row finiteness.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dlassq_row_update_f64(row: &[f64], scales: &mut [f64], sums: &mut [f64]) -> bool {
    debug_assert_eq!(row.len(), scales.len());
    debug_assert_eq!(row.len(), sums.len());
    let n = row.len();
    // abs() == clear sign bit; exact for all inputs including -0.0.
    let absmask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));
    let one = _mm256_set1_pd(1.0);
    let zero = _mm256_setzero_pd();
    let mut i = 0;
    let mut ok = true;
    while i + 4 <= n {
        let mag = _mm256_and_pd(_mm256_loadu_pd(row.as_ptr().add(i)), absmask);
        let sc = _mm256_loadu_pd(scales.as_ptr().add(i));
        let su = _mm256_loadu_pd(sums.as_ptr().add(i));
        // Finite lanes: NaN fails EQ with itself, but ±inf passes, so
        // also exclude mag > f64::MAX (fails for NaN and ±inf alike).
        // Zero lanes excluded below. Active lanes update state exactly
        // like the scalar body.
        let fin = _mm256_cmp_pd(mag, mag, _CMP_EQ_OQ);
        let bounded = _mm256_cmp_pd(mag, _mm256_set1_pd(f64::MAX), _CMP_LE_OQ);
        let fin = _mm256_and_pd(fin, bounded);
        if _mm256_movemask_pd(fin) != 0xF {
            ok = false;
        }
        let nz = _mm256_cmp_pd(mag, zero, _CMP_NEQ_OQ);
        let upd = _mm256_and_pd(fin, nz);
        let lt = _mm256_cmp_pd(sc, mag, _CMP_LT_OQ);
        // iflt: ratio = sc/mag; sums = 1 + (sums*ratio)*ratio (two
        // roundings, scalar order — not FMA).
        let r_lt = _mm256_div_pd(sc, mag);
        let su_lt = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(su, r_lt), r_lt), one);
        // else: ratio = mag/sc; sums += ratio*ratio.
        let r_ge = _mm256_div_pd(mag, sc);
        let su_ge = _mm256_add_pd(su, _mm256_mul_pd(r_ge, r_ge));
        let su_new = _mm256_blendv_pd(su, _mm256_blendv_pd(su_ge, su_lt, lt), upd);
        _mm256_storeu_pd(sums.as_mut_ptr().add(i), su_new);
        // scale = mag where active and greater, else unchanged.
        let sc_new = _mm256_blendv_pd(sc, mag, _mm256_and_pd(upd, lt));
        _mm256_storeu_pd(scales.as_mut_ptr().add(i), sc_new);
        i += 4;
    }
    if i < n {
        ok &= super::scalar::dlassq_row_update_f64(&row[i..], &mut scales[i..], &mut sums[i..]);
    }
    ok
}

/// f32 twin of [`dlassq_row_update_f64`]: same equations, 8-wide.
/// Same bitwise-identity contract vs
/// [`super::scalar::dlassq_row_update_f32`] (NaN/±inf/zero masked).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dlassq_row_update_f32(row: &[f32], scales: &mut [f32], sums: &mut [f32]) -> bool {
    debug_assert_eq!(row.len(), scales.len());
    debug_assert_eq!(row.len(), sums.len());
    let n = row.len();
    // abs() == clear sign bit; NOTE: `(-1i32) >> 1` is an arithmetic
    // shift (stays all-ones), so spell the mask out explicitly.
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    let mut ok = true;
    while i + 8 <= n {
        let mag = _mm256_and_ps(_mm256_loadu_ps(row.as_ptr().add(i)), absmask);
        let sc = _mm256_loadu_ps(scales.as_ptr().add(i));
        let su = _mm256_loadu_ps(sums.as_ptr().add(i));
        // Finite lanes: NaN fails EQ with itself, but ±inf passes, so
        // also exclude mag > f32::MAX (fails for NaN and ±inf alike).
        let fin = _mm256_cmp_ps(mag, mag, _CMP_EQ_OQ);
        let bounded = _mm256_cmp_ps(mag, _mm256_set1_ps(f32::MAX), _CMP_LE_OQ);
        let fin = _mm256_and_ps(fin, bounded);
        if _mm256_movemask_ps(fin) != 0xFF {
            ok = false;
        }
        let nz = _mm256_cmp_ps(mag, zero, _CMP_NEQ_OQ);
        let upd = _mm256_and_ps(fin, nz);
        let lt = _mm256_cmp_ps(sc, mag, _CMP_LT_OQ);
        let r_lt = _mm256_div_ps(sc, mag);
        let su_lt = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(su, r_lt), r_lt), one);
        let r_ge = _mm256_div_ps(mag, sc);
        let su_ge = _mm256_add_ps(su, _mm256_mul_ps(r_ge, r_ge));
        let su_new = _mm256_blendv_ps(su, _mm256_blendv_ps(su_ge, su_lt, lt), upd);
        _mm256_storeu_ps(sums.as_mut_ptr().add(i), su_new);
        let sc_new = _mm256_blendv_ps(sc, mag, _mm256_and_ps(upd, lt));
        _mm256_storeu_ps(scales.as_mut_ptr().add(i), sc_new);
        i += 8;
    }
    if i < n {
        ok &= super::scalar::dlassq_row_update_f32(&row[i..], &mut scales[i..], &mut sums[i..]);
    }
    ok
}
