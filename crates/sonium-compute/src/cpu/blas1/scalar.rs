//! Portable scalar BLAS-1 reference kernels.
//!
//! All kernels use four parallel accumulators in reductions to break the
//! serial dependency chain on the FMA pipeline. Loop tails are handled
//! scalar. These run on every architecture and serve as the fallback
//! when no SIMD-specific path applies.

use num_complex::{Complex32, Complex64};

// ---------- axpy: y <- alpha * x + y ----------

#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        y[i] += alpha * x[i];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        i += 4;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        y[i] += alpha * x[i];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        i += 4;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

// ---------- ger_update: block[i*bs+j] += mults[i*ms] * pivot[j] ----------

/// Portable rank-1 update used by the LU panel factorization (single call
/// per panel column instead of one axpy per row). Same per-element operation
/// order as the axpy loop it replaces; SIMD kernels use one FMA per element
/// exactly like the AVX2 axpy, so results match the vectorized axpy loop
/// bit-for-bit (scalar mul+add may differ in the last ulp, as with axpy).
#[inline]
pub fn ger_update_f32(
    alpha: f32,
    mults: &[f32],
    mult_stride: usize,
    pivot: &[f32],
    block: &mut [f32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    for i in 0..rows {
        let m = alpha * mults[i * mult_stride];
        let row = &mut block[i * block_stride..i * block_stride + cols];
        let mut j = 0;
        while j + 4 <= cols {
            row[j] += m * pivot[j];
            row[j + 1] += m * pivot[j + 1];
            row[j + 2] += m * pivot[j + 2];
            row[j + 3] += m * pivot[j + 3];
            j += 4;
        }
        while j < cols {
            row[j] += m * pivot[j];
            j += 1;
        }
    }
}

/// Portable rank-1 update used by the LU panel factorization (single call
/// per panel column instead of one axpy per row). Same per-element operation
/// order as the axpy loop it replaces; SIMD kernels use one FMA per element
/// exactly like the AVX2 axpy, so results match the vectorized axpy loop
/// bit-for-bit (scalar mul+add may differ in the last ulp, as with axpy).
#[inline]
pub fn ger_update_f64(
    alpha: f64,
    mults: &[f64],
    mult_stride: usize,
    pivot: &[f64],
    block: &mut [f64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    for i in 0..rows {
        let m = alpha * mults[i * mult_stride];
        let row = &mut block[i * block_stride..i * block_stride + cols];
        let mut j = 0;
        while j + 4 <= cols {
            row[j] += m * pivot[j];
            row[j + 1] += m * pivot[j + 1];
            row[j + 2] += m * pivot[j + 2];
            row[j + 3] += m * pivot[j + 3];
            j += 4;
        }
        while j < cols {
            row[j] += m * pivot[j];
            j += 1;
        }
    }
}

/// Portable pivot scan: first row `>= first` with maximum `|base[i*stride]|`
/// over `len` rows, returning `(row, value)` with strict `>` comparison
/// (first maximum wins, exactly the LU GETF2 scalar loop). SIMD kernels
/// must reproduce the index bit-for-bit, including the non-finite-initial
/// fast path (NaN/+Inf initial pivot keeps its row: nothing exceeds it).
#[inline]
pub fn pivot_scan_f32(base: &[f32], stride: usize, first: usize, len: usize) -> (usize, f32) {
    debug_assert!(len >= 1);
    let mut max_row = first;
    let mut max_val = base[first * stride].abs();
    for i in 1..len {
        let val = base[(first + i) * stride].abs();
        if val > max_val {
            max_val = val;
            max_row = first + i;
        }
    }
    (max_row, max_val)
}

/// Portable pivot scan: first row `>= first` with maximum `|base[i*stride]|`
/// over `len` rows. Same exactness contract as [`pivot_scan_f32`].
#[inline]
pub fn pivot_scan_f64(base: &[f64], stride: usize, first: usize, len: usize) -> (usize, f64) {
    debug_assert!(len >= 1);
    let mut max_row = first;
    let mut max_val = base[first * stride].abs();
    for i in 1..len {
        let val = base[(first + i) * stride].abs();
        if val > max_val {
            max_val = val;
            max_row = first + i;
        }
    }
    (max_row, max_val)
}

#[inline]
pub fn axpy_c32(alpha: Complex32, x: &[Complex32], y: &mut [Complex32]) {
    debug_assert_eq!(x.len(), y.len());
    let ar = alpha.re;
    let ai = alpha.im;
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        let r = ar * xi.re - ai * xi.im;
        let i = ar * xi.im + ai * xi.re;
        yi.re += r;
        yi.im += i;
    }
}

#[inline]
pub fn axpy_c64(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    debug_assert_eq!(x.len(), y.len());
    let ar = alpha.re;
    let ai = alpha.im;
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        let r = ar * xi.re - ai * xi.im;
        let i = ar * xi.im + ai * xi.re;
        yi.re += r;
        yi.im += i;
    }
}

// ---------- scal: x <- alpha * x ----------

#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    for xi in x {
        *xi *= alpha;
    }
}

#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    for xi in x {
        *xi *= alpha;
    }
}

#[inline]
pub fn scal_c32(alpha: Complex32, x: &mut [Complex32]) {
    let ar = alpha.re;
    let ai = alpha.im;
    for xi in x {
        let r = ar * xi.re - ai * xi.im;
        let i = ar * xi.im + ai * xi.re;
        xi.re = r;
        xi.im = i;
    }
}

#[inline]
pub fn scal_c64(alpha: Complex64, x: &mut [Complex64]) {
    let ar = alpha.re;
    let ai = alpha.im;
    for xi in x {
        let r = ar * xi.re - ai * xi.im;
        let i = ar * xi.im + ai * xi.re;
        xi.re = r;
        xi.im = i;
    }
}

// ---------- dot (real) / dotc (Hermitian for complex) ----------

#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut i = 0;
    while i + 4 <= n {
        s0 += x[i] * y[i];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
        i += 4;
    }
    let mut tail = 0.0_f32;
    while i < n {
        tail += x[i] * y[i];
        i += 1;
    }
    (s0 + s1) + (s2 + s3) + tail
}

#[inline]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut i = 0;
    while i + 4 <= n {
        s0 += x[i] * y[i];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
        i += 4;
    }
    let mut tail = 0.0_f64;
    while i < n {
        tail += x[i] * y[i];
        i += 1;
    }
    (s0 + s1) + (s2 + s3) + tail
}

/// Hermitian dot `Σ conj(x_i) * y_i`.
#[inline]
pub fn dotc_c32(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sr0 = 0.0_f32;
    let mut sr1 = 0.0_f32;
    let mut si0 = 0.0_f32;
    let mut si1 = 0.0_f32;
    let n = x.len();
    let mut i = 0;
    while i + 2 <= n {
        // conj(x).re = x.re; conj(x).im = -x.im
        // (conj(x)*y).re = x.re*y.re + x.im*y.im
        // (conj(x)*y).im = x.re*y.im - x.im*y.re
        sr0 += x[i].re * y[i].re + x[i].im * y[i].im;
        si0 += x[i].re * y[i].im - x[i].im * y[i].re;
        sr1 += x[i + 1].re * y[i + 1].re + x[i + 1].im * y[i + 1].im;
        si1 += x[i + 1].re * y[i + 1].im - x[i + 1].im * y[i + 1].re;
        i += 2;
    }
    let mut srt = 0.0_f32;
    let mut sit = 0.0_f32;
    while i < n {
        srt += x[i].re * y[i].re + x[i].im * y[i].im;
        sit += x[i].re * y[i].im - x[i].im * y[i].re;
        i += 1;
    }
    Complex32::new(sr0 + sr1 + srt, si0 + si1 + sit)
}

#[inline]
pub fn dotc_c64(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    debug_assert_eq!(x.len(), y.len());
    let mut sr0 = 0.0_f64;
    let mut sr1 = 0.0_f64;
    let mut si0 = 0.0_f64;
    let mut si1 = 0.0_f64;
    let n = x.len();
    let mut i = 0;
    while i + 2 <= n {
        sr0 += x[i].re * y[i].re + x[i].im * y[i].im;
        si0 += x[i].re * y[i].im - x[i].im * y[i].re;
        sr1 += x[i + 1].re * y[i + 1].re + x[i + 1].im * y[i + 1].im;
        si1 += x[i + 1].re * y[i + 1].im - x[i + 1].im * y[i + 1].re;
        i += 2;
    }
    let mut srt = 0.0_f64;
    let mut sit = 0.0_f64;
    while i < n {
        srt += x[i].re * y[i].re + x[i].im * y[i].im;
        sit += x[i].re * y[i].im - x[i].im * y[i].re;
        i += 1;
    }
    Complex64::new(sr0 + sr1 + srt, si0 + si1 + sit)
}

// ---------- nrm2 ----------

#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let n = x.len();
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut i = 0;
    while i + 4 <= n {
        s0 += x[i] * x[i];
        s1 += x[i + 1] * x[i + 1];
        s2 += x[i + 2] * x[i + 2];
        s3 += x[i + 3] * x[i + 3];
        i += 4;
    }
    let mut tail = 0.0_f32;
    while i < n {
        tail += x[i] * x[i];
        i += 1;
    }
    ((s0 + s1) + (s2 + s3) + tail).sqrt()
}

#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut i = 0;
    while i + 4 <= n {
        s0 += x[i] * x[i];
        s1 += x[i + 1] * x[i + 1];
        s2 += x[i + 2] * x[i + 2];
        s3 += x[i + 3] * x[i + 3];
        i += 4;
    }
    let mut tail = 0.0_f64;
    while i < n {
        tail += x[i] * x[i];
        i += 1;
    }
    ((s0 + s1) + (s2 + s3) + tail).sqrt()
}

#[inline]
pub fn nrm2_c32(x: &[Complex32]) -> f32 {
    let n = x.len();
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut i = 0;
    while i + 4 <= n {
        s0 += x[i].re * x[i].re + x[i].im * x[i].im;
        s1 += x[i + 1].re * x[i + 1].re + x[i + 1].im * x[i + 1].im;
        s2 += x[i + 2].re * x[i + 2].re + x[i + 2].im * x[i + 2].im;
        s3 += x[i + 3].re * x[i + 3].re + x[i + 3].im * x[i + 3].im;
        i += 4;
    }
    let mut tail = 0.0_f32;
    while i < n {
        tail += x[i].re * x[i].re + x[i].im * x[i].im;
        i += 1;
    }
    ((s0 + s1) + (s2 + s3) + tail).sqrt()
}

#[inline]
pub fn nrm2_c64(x: &[Complex64]) -> f64 {
    let n = x.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut i = 0;
    while i + 4 <= n {
        s0 += x[i].re * x[i].re + x[i].im * x[i].im;
        s1 += x[i + 1].re * x[i + 1].re + x[i + 1].im * x[i + 1].im;
        s2 += x[i + 2].re * x[i + 2].re + x[i + 2].im * x[i + 2].im;
        s3 += x[i + 3].re * x[i + 3].re + x[i + 3].im * x[i + 3].im;
        i += 4;
    }
    let mut tail = 0.0_f64;
    while i < n {
        tail += x[i].re * x[i].re + x[i].im * x[i].im;
        i += 1;
    }
    ((s0 + s1) + (s2 + s3) + tail).sqrt()
}

// ---------- axpy_many: y[i] += Σ_bb alphas[bb] * x_basis[bb][i] ----------

#[inline]
pub fn axpy_many_f32(alphas: &[f32], x_basis: &[&[f32]], y: &mut [f32]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    let mut i = 0;
    while i + 4 <= n {
        let mut acc0 = y[i];
        let mut acc1 = y[i + 1];
        let mut acc2 = y[i + 2];
        let mut acc3 = y[i + 3];
        for bb in 0..k {
            let alpha = alphas[bb];
            let xb = x_basis[bb];
            acc0 += alpha * xb[i];
            acc1 += alpha * xb[i + 1];
            acc2 += alpha * xb[i + 2];
            acc3 += alpha * xb[i + 3];
        }
        y[i] = acc0;
        y[i + 1] = acc1;
        y[i + 2] = acc2;
        y[i + 3] = acc3;
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

#[inline]
pub fn axpy_many_f64(alphas: &[f64], x_basis: &[&[f64]], y: &mut [f64]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    let mut i = 0;
    while i + 4 <= n {
        let mut acc0 = y[i];
        let mut acc1 = y[i + 1];
        let mut acc2 = y[i + 2];
        let mut acc3 = y[i + 3];
        for bb in 0..k {
            let alpha = alphas[bb];
            let xb = x_basis[bb];
            acc0 += alpha * xb[i];
            acc1 += alpha * xb[i + 1];
            acc2 += alpha * xb[i + 2];
            acc3 += alpha * xb[i + 3];
        }
        y[i] = acc0;
        y[i + 1] = acc1;
        y[i + 2] = acc2;
        y[i + 3] = acc3;
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

#[inline]
pub fn axpy_many_c32(alphas: &[Complex32], x_basis: &[&[Complex32]], y: &mut [Complex32]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    for i in 0..n {
        let mut acc_re = y[i].re;
        let mut acc_im = y[i].im;
        for bb in 0..k {
            let alpha = alphas[bb];
            let xb = x_basis[bb][i];
            acc_re += alpha.re * xb.re - alpha.im * xb.im;
            acc_im += alpha.re * xb.im + alpha.im * xb.re;
        }
        y[i].re = acc_re;
        y[i].im = acc_im;
    }
}

#[inline]
pub fn axpy_many_c64(alphas: &[Complex64], x_basis: &[&[Complex64]], y: &mut [Complex64]) {
    debug_assert_eq!(alphas.len(), x_basis.len());
    let k = alphas.len();
    if k == 0 || y.is_empty() {
        return;
    }
    let n = y.len();
    for i in 0..n {
        let mut acc_re = y[i].re;
        let mut acc_im = y[i].im;
        for bb in 0..k {
            let alpha = alphas[bb];
            let xb = x_basis[bb][i];
            acc_re += alpha.re * xb.re - alpha.im * xb.im;
            acc_im += alpha.re * xb.im + alpha.im * xb.re;
        }
        y[i].re = acc_re;
        y[i].im = acc_im;
    }
}

// ---------- find_non_finite: first index with a non-finite lane ----------

/// Portable first-non-finite scan: index of the first NaN or ±inf lane,
/// or `None` when all lanes are finite. Fallback for
/// [`super::find_non_finite_f32`](super::find_non_finite_f32) where AVX2 is
/// unavailable, and the implementation on other architectures.
#[inline]
pub fn find_non_finite_f32(values: &[f32]) -> Option<usize> {
    values.iter().position(|v| !v.is_finite())
}

/// f64 twin of [`find_non_finite_f32`].
#[inline]
pub fn find_non_finite_f64(values: &[f64]) -> Option<usize> {
    values.iter().position(|v| !v.is_finite())
}

// ---------- max_norm: max |lane| over real lanes ----------

/// Portable max-abs reduction: `max |v|` over real lanes (`0` when empty).
/// Fallback for [`super::max_norm_f32`](super::max_norm_f32) where AVX2 is
/// unavailable, and the implementation on other architectures. Exact: `abs`
/// then `max` per lane, same value as the scalar fold.
#[inline]
pub fn max_norm_f32(values: &[f32]) -> f32 {
    let mut best = 0.0f32;
    for &v in values {
        let a = v.abs();
        if a > best {
            best = a;
        }
    }
    best
}

/// f64 twin of [`max_norm_f32`].
#[inline]
pub fn max_norm_f64(values: &[f64]) -> f64 {
    let mut best = 0.0f64;
    for &v in values {
        let a = v.abs();
        if a > best {
            best = a;
        }
    }
    best
}

// ---------- max_norm_checked: fused max-abs + finiteness fold ----------

/// Portable fused entry-validation fold `(max |v|, all_finite)` in one
/// pass. Same max value as [`max_norm_f32`]; the flag is false iff some
/// lane is NaN or ±inf. Callers use the max only when the flag holds and
/// rescan for the first bad index otherwise (rare).
#[inline]
pub fn max_norm_checked_f32(values: &[f32]) -> (f32, bool) {
    let mut best = 0.0f32;
    let mut ok = true;
    for &v in values {
        let a = v.abs();
        if a > best {
            best = a;
        }
        ok &= a.is_finite();
    }
    (best, ok)
}

/// f64 twin of [`max_norm_checked_f32`].
#[inline]
pub fn max_norm_checked_f64(values: &[f64]) -> (f64, bool) {
    let mut best = 0.0f64;
    let mut ok = true;
    for &v in values {
        let a = v.abs();
        if a > best {
            best = a;
        }
        ok &= a.is_finite();
    }
    (best, ok)
}

/// Portable fused `dlassq` row accumulation: fold one row of magnitudes
/// into per-column `(scale, sums)` state with the exact scalar update
/// equations. Returns row finiteness. Reference for the AVX2 twins,
/// which reproduce it bitwise by vectorizing across independent
/// columns (per-column op order kept).
#[inline]
pub fn dlassq_row_update_f64(row: &[f64], scales: &mut [f64], sums: &mut [f64]) -> bool {
    debug_assert_eq!(row.len(), scales.len());
    debug_assert_eq!(row.len(), sums.len());
    let mut ok = true;
    for (c, &v) in row.iter().enumerate() {
        let magnitude = v.abs();
        if !magnitude.is_finite() {
            ok = false;
        } else if magnitude != 0.0 {
            if scales[c] < magnitude {
                let ratio = scales[c] / magnitude;
                sums[c] = 1.0 + sums[c] * ratio * ratio;
                scales[c] = magnitude;
            } else {
                let ratio = magnitude / scales[c];
                sums[c] += ratio * ratio;
            }
        }
    }
    ok
}

/// f32 twin of [`dlassq_row_update_f64`]: same equations, same contract.
#[inline]
pub fn dlassq_row_update_f32(row: &[f32], scales: &mut [f32], sums: &mut [f32]) -> bool {
    debug_assert_eq!(row.len(), scales.len());
    debug_assert_eq!(row.len(), sums.len());
    let mut ok = true;
    for (c, &v) in row.iter().enumerate() {
        let magnitude = v.abs();
        if !magnitude.is_finite() {
            ok = false;
        } else if magnitude != 0.0 {
            if scales[c] < magnitude {
                let ratio = scales[c] / magnitude;
                sums[c] = 1.0 + sums[c] * ratio * ratio;
                scales[c] = magnitude;
            } else {
                let ratio = magnitude / scales[c];
                sums[c] += ratio * ratio;
            }
        }
    }
    ok
}

// ---------- complex pivot scan / ger (portable reference) ----------

/// Portable complex pivot scan: first row `>= first` with maximum
/// `base[i*stride].norm()` (true modulus, same metric as the
/// `ComplexField` default) over `len` rows. Same first-wins contract as
/// [`pivot_scan_f64`]; AVX2 twins reproduce the index.
#[inline]
pub fn pivot_scan_c32(base: &[Complex32], stride: usize, first: usize, len: usize) -> (usize, f32) {
    debug_assert!(len >= 1);
    let mut max_row = first;
    let mut max_val = base[first * stride].norm();
    for i in 1..len {
        let val = base[(first + i) * stride].norm();
        if val > max_val {
            max_val = val;
            max_row = first + i;
        }
    }
    (max_row, max_val)
}

/// Complex64 twin of [`pivot_scan_c32`].
#[inline]
pub fn pivot_scan_c64(base: &[Complex64], stride: usize, first: usize, len: usize) -> (usize, f64) {
    debug_assert!(len >= 1);
    let mut max_row = first;
    let mut max_val = base[first * stride].norm();
    for i in 1..len {
        let val = base[(first + i) * stride].norm();
        if val > max_val {
            max_val = val;
            max_row = first + i;
        }
    }
    (max_row, max_val)
}

/// Portable complex rank-1 update `block[i*bs+j] += alpha*mults[i*ms] *
/// pivot[j]`: one [`axpy_c32`] per row, exactly the `ComplexField` default
/// it replaces (same per-element operation order).
#[inline]
pub fn ger_update_c32(
    alpha: Complex32,
    mults: &[Complex32],
    mult_stride: usize,
    pivot: &[Complex32],
    block: &mut [Complex32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    for i in 0..rows {
        let m = alpha * mults[i * mult_stride];
        let row = &mut block[i * block_stride..i * block_stride + cols];
        axpy_c32(m, &pivot[..cols], row);
    }
}

/// Complex64 twin of [`ger_update_c32`] (via [`axpy_c64`]).
#[inline]
pub fn ger_update_c64(
    alpha: Complex64,
    mults: &[Complex64],
    mult_stride: usize,
    pivot: &[Complex64],
    block: &mut [Complex64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    for i in 0..rows {
        let m = alpha * mults[i * mult_stride];
        let row = &mut block[i * block_stride..i * block_stride + cols];
        axpy_c64(m, &pivot[..cols], row);
    }
}

/// Portable complex entry-validation fold `(max norm, all_finite)` in one
/// pass (Complex32): exact reference for the AVX2 twin and the
/// `ComplexField::vec_max_norm_checked` default (same `hypot` metric).
#[inline]
pub fn max_norm_checked_c32(values: &[Complex32]) -> (f32, bool) {
    let mut best = 0.0f32;
    let mut ok = true;
    for &v in values {
        let m = v.norm();
        if m > best {
            best = m;
        }
        ok &= m.is_finite();
    }
    (best, ok)
}

/// Complex64 twin of [`max_norm_checked_c32`].
#[inline]
pub fn max_norm_checked_c64(values: &[Complex64]) -> (f64, bool) {
    let mut best = 0.0f64;
    let mut ok = true;
    for &v in values {
        let m = v.norm();
        if m > best {
            best = m;
        }
        ok &= m.is_finite();
    }
    (best, ok)
}

// ---------- complex 4M epilogue (portable reference) ----------

/// Portable 4M combine `C := α·(R + i·I) + β·C` over one contiguous row of
/// `n` values (`r`/`imag` hold the real/imaginary product halves, `c` the
/// interleaved output row). Exact reference for the AVX2 twins: same
/// per-element operation order as the historical inline loop.
#[inline]
pub fn apply_ab_c64_row(
    r: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    let (ar, ai) = (alpha.re, alpha.im);
    let (br, bi) = (beta.re, beta.im);
    for ((&rij, &iij), cij) in r.iter().zip(imag.iter()).zip(c.iter_mut()) {
        let ab_re = ar * rij - ai * iij;
        let ab_im = ar * iij + ai * rij;
        *cij = if beta_zero {
            Complex64::new(ab_re, ab_im)
        } else {
            Complex64::new(
                ab_re + br * cij.re - bi * cij.im,
                ab_im + br * cij.im + bi * cij.re,
            )
        };
    }
}

/// 3M epilogue row (Complex64 twin of [`apply_ab3_c32_row`]).
pub fn apply_ab3_c64_row(
    r: &[f64],
    q: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    let (ar, ai) = (alpha.re, alpha.im);
    let (br, bi) = (beta.re, beta.im);
    for (((&rij, &qij), &iij), cij) in r.iter().zip(q.iter()).zip(imag.iter()).zip(c.iter_mut()) {
        let re = rij - qij;
        let im = iij - rij - qij;
        let ab_re = ar * re - ai * im;
        let ab_im = ar * im + ai * re;
        *cij = if beta_zero {
            Complex64::new(ab_re, ab_im)
        } else {
            Complex64::new(
                ab_re + br * cij.re - bi * cij.im,
                ab_im + br * cij.im + bi * cij.re,
            )
        };
    }
}

/// Complex32 twin of [`apply_ab_c64_row`].
#[inline]
pub fn apply_ab_c32_row(
    r: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    let (ar, ai) = (alpha.re, alpha.im);
    let (br, bi) = (beta.re, beta.im);
    for ((&rij, &iij), cij) in r.iter().zip(imag.iter()).zip(c.iter_mut()) {
        let ab_re = ar * rij - ai * iij;
        let ab_im = ar * iij + ai * rij;
        *cij = if beta_zero {
            Complex32::new(ab_re, ab_im)
        } else {
            Complex32::new(
                ab_re + br * cij.re - bi * cij.im,
                ab_im + br * cij.im + bi * cij.re,
            )
        };
    }
}

/// 3M epilogue row: `Re = R - Q`, `Im = I - R - Q` (with `R = Ar·Br`,
/// `Q = Ai·Bi`, `I = (Ar+Ai)·(Br+Bi)`), then `C := α·(Re + i·Im) + β·C`.
/// Same α/β math as [`apply_ab_c32_row`], one fused pass (no separate
/// subtract pass over the intermediates).
pub fn apply_ab3_c32_row(
    r: &[f32],
    q: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    let (ar, ai) = (alpha.re, alpha.im);
    let (br, bi) = (beta.re, beta.im);
    for (((&rij, &qij), &iij), cij) in r.iter().zip(q.iter()).zip(imag.iter()).zip(c.iter_mut()) {
        let re = rij - qij;
        let im = iij - rij - qij;
        let ab_re = ar * re - ai * im;
        let ab_im = ar * im + ai * re;
        *cij = if beta_zero {
            Complex32::new(ab_re, ab_im)
        } else {
            Complex32::new(
                ab_re + br * cij.re - bi * cij.im,
                ab_im + br * cij.im + bi * cij.re,
            )
        };
    }
}
