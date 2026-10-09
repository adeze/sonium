//! Native interleaved complex `gemv` kernels.
//!
//! Unlike [`crate::cpu::blas2::complex4m`], which deinterleaves `op(A)` and
//! `x` into split real/imag scratch and runs four real GEMV calls per
//! invocation, these kernels operate directly on the interleaved
//! `Complex32`/`Complex64` row-major storage with zero per-call repacking:
//!
//! - On `target_arch = "aarch64"` we use NEON intrinsics (always available
//!   on AArch64 — no runtime feature check). One `Complex64` per 128-bit
//!   register (two `Complex32`); the complex multiply uses the
//!   multiply + lane-swap trick with the sign fold deferred to the
//!   horizontal reduction, and FMA throughout.
//! - On `target_arch = "x86_64"` we runtime-dispatch to AVX2/FMA kernels
//!   (two `Complex64` / four `Complex32` per 256-bit register) when the
//!   host supports them, otherwise the scalar reference.
//! - On every other target we fall back to the scalar reference.
//!
//! `NoTrans` computes one row dot-product per output element; `Trans` and
//! `ConjTrans` fan out into contiguous row axpys (mirroring
//! `gemv_trans_f64` in [`crate::cpu::blas2`]), folding conjugation into the
//! per-row broadcast constants so the inner loop is identical for both ops.

#![allow(clippy::too_many_arguments)]

use crate::backend::Transpose;
use crate::cpu::blas1;
use num_complex::{Complex32, Complex64};

/// Pointer wrappers for `rayon::scope` dispatch: disjoint `y` bands (or
/// disjoint partials bands), read-only `A`/`x`. Mirrors the wrappers in
/// [`crate::cpu::blas2::avx2`], which is `x86_64`-only while this module
/// compiles everywhere.
#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Minimum `m * n` (complex elements) before paying scope spawn/join, and
/// minimum rows for a parallel split at all. Below this the serial sweep
/// wins. A complex element is twice the flops of a real one, so the
/// product bar matches the real `PAR_MIN_MN` in flops at half the count.
const PAR_MIN_MN: usize = 131_072;
const PAR_MIN_M: usize = 16;

#[inline]
fn complex_parallel_worthwhile(m: usize, n: usize) -> bool {
    m >= PAR_MIN_M && m.saturating_mul(n) >= PAR_MIN_MN
}

/// Parallel `NoTrans` complex GEMV: row bands own disjoint `y` slices, so no
/// privatisation buffer is needed. `beta` must already be folded into `y`.
fn gemv_nt_parallel_c64(
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    y: &mut [Complex64],
) {
    let workers = rayon::current_num_threads().max(1);
    let tasks = (m.div_ceil(64)).clamp(1, workers.saturating_mul(8).max(1));
    let a_send = SendConstPtr(a.as_ptr());
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    rayon::scope(|s| {
        let mut r0 = 0;
        for _ in 0..tasks {
            if r0 >= m {
                break;
            }
            let r1 = (r0 + m.div_ceil(tasks)).min(m);
            s.spawn(move |_| {
                let a_send: SendConstPtr<Complex64> = a_send;
                let x_send: SendConstPtr<Complex64> = x_send;
                let y_send: SendMutPtr<Complex64> = y_send;
                // SAFETY: band rows `[r0, r1)` of `A`/`y` are disjoint
                // across tasks; `x` is read-only.
                let mb = r1 - r0;
                let a_band = unsafe {
                    core::slice::from_raw_parts(a_send.0.add(r0 * lda), (mb - 1) * lda + n)
                };
                let x_all = unsafe { core::slice::from_raw_parts(x_send.0, n) };
                let y_band = unsafe { core::slice::from_raw_parts_mut(y_send.0.add(r0), mb) };
                active::gemv_nt_c64(mb, n, alpha, a_band, lda, x_all, y_band);
            });
            r0 = r1;
        }
    });
}

/// Parallel `Trans`/`ConjTrans` complex GEMV: the serial fan-out accumulates
/// every row into the shared `y`, so each task folds its row block into a
/// private band and an ordered reduce adds the bands into `y` after the
/// scope joins — the same privatised pattern as `avx2::gemv_trans_f64`,
/// except the output is additionally tiled into `m`-chunks. Without tiling,
/// each task's private band spans the whole `y` (e.g. 64 KiB at `m=4096`),
/// spilling out of L1 on every one of its rows; with chunks of at most
/// 512 elements (8 KiB) the band stays L1-resident while the `A` traffic
/// stays at exactly one read of the matrix.
/// `beta` must already be folded into `y`.
fn gemv_tct_parallel_c64(
    conj: bool,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    y: &mut [Complex64],
) {
    /// Output tile: 512 `Complex64` = 8 KiB, comfortably L1-resident.
    const CHUNK: usize = 512;
    // NOTE: an output-split variant (disjoint `y`-chunks, no private bands)
    // was tried here and measured ~40% slower than the tiled fan-out below:
    // reusing the axpy fan-out kernel per chunk keeps the 3x `y` traffic
    // while turning the `A` reads strided, losing on both counts. Closing
    // the rest of the cache-resident `Trans` gap needs a dedicated
    // strided-column dot kernel, not a dispatch change.
    let workers = rayon::current_num_threads().max(1);
    // Task sizing mirrors `avx2::gemv_trans_f64`: `workers * 4` tasks
    // total, at least ~32 rows per block.
    let tasks = workers.saturating_mul(4).max(2);
    let nchunks = m.div_ceil(CHUNK).max(1);
    // Row blocks per chunk column, sized so blocks × chunks ≈ tasks.
    let blocks = (n / 32).clamp(1, tasks.div_ceil(nchunks)).max(1);
    let zero = Complex64::new(0.0, 0.0);
    let mut partials = vec![zero; blocks * nchunks * CHUNK];
    let a_send = SendConstPtr(a.as_ptr());
    let x_send = SendConstPtr(x.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    rayon::scope(|s| {
        for c in 0..nchunks {
            let i0 = c * CHUNK;
            let clen = (m - i0).min(CHUNK);
            let mut j0 = 0;
            let mut b = 0;
            while j0 < n {
                let j1 = (j0 + n.div_ceil(blocks)).min(n);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<Complex64> = a_send;
                    let x_send: SendConstPtr<Complex64> = x_send;
                    let p_send: SendMutPtr<Complex64> = p_send;
                    // SAFETY: task `(b, c)` writes only band `b * nchunks + c`;
                    // `A`/`x` are read-only. Bands start zeroed.
                    let nb = j1 - j0;
                    // Row `j` of the tile starts at `a + (j0 + j) * lda + i0`
                    // and spans `clen <= m` elements; the slice must cover
                    // `(nb - 1) * lda + clen` entries from the tile origin.
                    let a_tile = unsafe {
                        core::slice::from_raw_parts(
                            a_send.0.add(j0 * lda + i0),
                            (nb - 1) * lda + clen,
                        )
                    };
                    let x_band = unsafe { core::slice::from_raw_parts(x_send.0.add(j0), nb) };
                    let priv_y = unsafe {
                        core::slice::from_raw_parts_mut(
                            p_send.0.add((b * nchunks + c) * CHUNK),
                            clen,
                        )
                    };
                    if conj {
                        active::gemv_ct_c64(clen, nb, alpha, a_tile, lda, x_band, priv_y);
                    } else {
                        active::gemv_t_c64(clen, nb, alpha, a_tile, lda, x_band, priv_y);
                    }
                });
                b += 1;
                j0 = j1;
            }
        }
    });
    let one = Complex64::new(1.0, 0.0);
    for c in 0..nchunks {
        let i0 = c * CHUNK;
        let clen = (m - i0).min(CHUNK);
        let y_chunk = &mut y[i0..i0 + clen];
        for b in 0..blocks {
            // Band `(b, c)` zero-pads `clen..CHUNK`; reduce only the live
            // prefix (bands start zeroed, so the padding is exactly zero).
            blas1::axpy_c64(
                one,
                &partials[(b * nchunks + c) * CHUNK..(b * nchunks + c) * CHUNK + clen],
                y_chunk,
            );
        }
    }
}

/// `f32` twin of [`gemv_nt_parallel_c64`].
fn gemv_nt_parallel_c32(
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    y: &mut [Complex32],
) {
    let workers = rayon::current_num_threads().max(1);
    let tasks = (m.div_ceil(64)).clamp(1, workers.saturating_mul(8).max(1));
    let a_send = SendConstPtr(a.as_ptr());
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    rayon::scope(|s| {
        let mut r0 = 0;
        for _ in 0..tasks {
            if r0 >= m {
                break;
            }
            let r1 = (r0 + m.div_ceil(tasks)).min(m);
            s.spawn(move |_| {
                let a_send: SendConstPtr<Complex32> = a_send;
                let x_send: SendConstPtr<Complex32> = x_send;
                let y_send: SendMutPtr<Complex32> = y_send;
                // SAFETY: band rows `[r0, r1)` of `A`/`y` are disjoint
                // across tasks; `x` is read-only.
                let mb = r1 - r0;
                let a_band = unsafe {
                    core::slice::from_raw_parts(a_send.0.add(r0 * lda), (mb - 1) * lda + n)
                };
                let x_all = unsafe { core::slice::from_raw_parts(x_send.0, n) };
                let y_band = unsafe { core::slice::from_raw_parts_mut(y_send.0.add(r0), mb) };
                active::gemv_nt_c32(mb, n, alpha, a_band, lda, x_all, y_band);
            });
            r0 = r1;
        }
    });
}

/// `f32` twin of [`gemv_tct_parallel_c64`]: same 2D-tiled privatised fan-out.
/// 1024 `Complex32` = 8 KiB per band, L1-resident.
fn gemv_tct_parallel_c32(
    conj: bool,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    y: &mut [Complex32],
) {
    const CHUNK: usize = 1024;
    let workers = rayon::current_num_threads().max(1);
    // Output-split deliberately not used here; see the `Complex64` twin.
    // Task sizing mirrors the `Complex64` twin above.
    let tasks = workers.saturating_mul(4).max(2);
    let nchunks = m.div_ceil(CHUNK).max(1);
    let blocks = (n / 32).clamp(1, tasks.div_ceil(nchunks)).max(1);
    let zero = Complex32::new(0.0, 0.0);
    let mut partials = vec![zero; blocks * nchunks * CHUNK];
    let a_send = SendConstPtr(a.as_ptr());
    let x_send = SendConstPtr(x.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    rayon::scope(|s| {
        for c in 0..nchunks {
            let i0 = c * CHUNK;
            let clen = (m - i0).min(CHUNK);
            let mut j0 = 0;
            let mut b = 0;
            while j0 < n {
                let j1 = (j0 + n.div_ceil(blocks)).min(n);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<Complex32> = a_send;
                    let x_send: SendConstPtr<Complex32> = x_send;
                    let p_send: SendMutPtr<Complex32> = p_send;
                    // SAFETY: task `(b, c)` writes only band `b * nchunks + c`;
                    // `A`/`x` are read-only. Bands start zeroed.
                    let nb = j1 - j0;
                    let a_tile = unsafe {
                        core::slice::from_raw_parts(
                            a_send.0.add(j0 * lda + i0),
                            (nb - 1) * lda + clen,
                        )
                    };
                    let x_band = unsafe { core::slice::from_raw_parts(x_send.0.add(j0), nb) };
                    let priv_y = unsafe {
                        core::slice::from_raw_parts_mut(
                            p_send.0.add((b * nchunks + c) * CHUNK),
                            clen,
                        )
                    };
                    if conj {
                        active::gemv_ct_c32(clen, nb, alpha, a_tile, lda, x_band, priv_y);
                    } else {
                        active::gemv_t_c32(clen, nb, alpha, a_tile, lda, x_band, priv_y);
                    }
                });
                b += 1;
                j0 = j1;
            }
        }
    });
    let one = Complex32::new(1.0, 0.0);
    for c in 0..nchunks {
        let i0 = c * CHUNK;
        let clen = (m - i0).min(CHUNK);
        let y_chunk = &mut y[i0..i0 + clen];
        for b in 0..blocks {
            blas1::axpy_c32(
                one,
                &partials[(b * nchunks + c) * CHUNK..(b * nchunks + c) * CHUNK + clen],
                y_chunk,
            );
        }
    }
}

// ---------- shared driver --------------------------------------------------

/// `y := alpha * op(A) * x + beta * y` for `Complex64`, interleaved native.
pub fn gemv_c64(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
) {
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    if m == 0 {
        return;
    }
    let zero = Complex64::new(0.0, 0.0);
    if beta == zero {
        for v in y.iter_mut() {
            *v = zero;
        }
    } else if beta != Complex64::new(1.0, 0.0) {
        for v in y.iter_mut() {
            *v *= beta;
        }
    }
    if alpha == zero || n == 0 {
        return;
    }
    // Large shapes fan out over rayon: disjoint row bands for `NoTrans`,
    // privatised fan-out blocks for `Trans`/`ConjTrans` (same pattern as
    // the real `avx2::gemv_trans_f64`). Block partials reduce in order, so
    // only rounding differs from the serial sweep.
    if complex_parallel_worthwhile(m, n) {
        match op {
            Transpose::NoTrans => {
                debug_assert!(lda >= n);
                gemv_nt_parallel_c64(m, n, alpha, a, lda, x, y);
            }
            Transpose::Trans => {
                debug_assert!(lda >= m);
                gemv_tct_parallel_c64(false, m, n, alpha, a, lda, x, y);
            }
            Transpose::ConjTrans => {
                debug_assert!(lda >= m);
                gemv_tct_parallel_c64(true, m, n, alpha, a, lda, x, y);
            }
        }
        return;
    }
    match op {
        Transpose::NoTrans => {
            debug_assert!(lda >= n);
            active::gemv_nt_c64(m, n, alpha, a, lda, x, y);
        }
        Transpose::Trans => {
            debug_assert!(lda >= m);
            active::gemv_t_c64(m, n, alpha, a, lda, x, y);
        }
        Transpose::ConjTrans => {
            debug_assert!(lda >= m);
            active::gemv_ct_c64(m, n, alpha, a, lda, x, y);
        }
    }
}

/// `y := alpha * op(A) * x + beta * y` for `Complex32`, interleaved native.
pub fn gemv_c32(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    beta: Complex32,
    y: &mut [Complex32],
) {
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    if m == 0 {
        return;
    }
    let zero = Complex32::new(0.0, 0.0);
    if beta == zero {
        for v in y.iter_mut() {
            *v = zero;
        }
    } else if beta != Complex32::new(1.0, 0.0) {
        for v in y.iter_mut() {
            *v *= beta;
        }
    }
    if alpha == zero || n == 0 {
        return;
    }
    if complex_parallel_worthwhile(m, n) {
        match op {
            Transpose::NoTrans => {
                debug_assert!(lda >= n);
                gemv_nt_parallel_c32(m, n, alpha, a, lda, x, y);
            }
            Transpose::Trans => {
                debug_assert!(lda >= m);
                gemv_tct_parallel_c32(false, m, n, alpha, a, lda, x, y);
            }
            Transpose::ConjTrans => {
                debug_assert!(lda >= m);
                gemv_tct_parallel_c32(true, m, n, alpha, a, lda, x, y);
            }
        }
        return;
    }
    match op {
        Transpose::NoTrans => {
            debug_assert!(lda >= n);
            active::gemv_nt_c32(m, n, alpha, a, lda, x, y);
        }
        Transpose::Trans => {
            debug_assert!(lda >= m);
            active::gemv_t_c32(m, n, alpha, a, lda, x, y);
        }
        Transpose::ConjTrans => {
            debug_assert!(lda >= m);
            active::gemv_ct_c32(m, n, alpha, a, lda, x, y);
        }
    }
}

// ---------- scalar reference ------------------------------------------------

// On aarch64 only the NEON module is active; the scalar reference is the
// fallback for x86_64-without-AVX2 and every other target.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod scalar {
    use num_complex::{Complex32, Complex64};

    pub fn gemv_nt_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for i in 0..m {
            let row = &a[i * lda..i * lda + n];
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, &aij) in row.iter().enumerate() {
                acc += aij * x[j];
            }
            y[i] += alpha * acc;
        }
    }

    pub fn gemv_t_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex64::new(0.0, 0.0) {
                continue;
            }
            let row = &a[j * lda..j * lda + m];
            for (i, &aji) in row.iter().enumerate() {
                y[i] += aji * xj;
            }
        }
    }

    pub fn gemv_ct_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex64::new(0.0, 0.0) {
                continue;
            }
            let row = &a[j * lda..j * lda + m];
            for (i, &aji) in row.iter().enumerate() {
                y[i] += aji.conj() * xj;
            }
        }
    }

    pub fn gemv_nt_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for i in 0..m {
            let row = &a[i * lda..i * lda + n];
            let mut acc = Complex32::new(0.0, 0.0);
            for (j, &aij) in row.iter().enumerate() {
                acc += aij * x[j];
            }
            y[i] += alpha * acc;
        }
    }

    pub fn gemv_t_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex32::new(0.0, 0.0) {
                continue;
            }
            let row = &a[j * lda..j * lda + m];
            for (i, &aji) in row.iter().enumerate() {
                y[i] += aji * xj;
            }
        }
    }

    pub fn gemv_ct_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex32::new(0.0, 0.0) {
                continue;
            }
            let row = &a[j * lda..j * lda + m];
            for (i, &aji) in row.iter().enumerate() {
                y[i] += aji.conj() * xj;
            }
        }
    }
}

// ---------- AArch64 NEON ----------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;
    use num_complex::{Complex32, Complex64};

    // One Complex64 per 128-bit register. The product a*x is accumulated in
    // two registers: acc1 += a * (xr, xr) gives (ar*xr, ai*xr) and
    // acc2 += swap(a) * (xi, xi) gives (ai*xi, ar*xi), so the final lanes
    // combine as re = acc1[0] - acc2[0], im = acc1[1] + acc2[1] — the sign
    // fold is deferred to the reduction.
    #[inline]
    fn dot_c64(row: &[Complex64], x: &[Complex64]) -> Complex64 {
        let n = row.len();
        debug_assert_eq!(x.len(), n);
        unsafe {
            let rp = row.as_ptr() as *const f64;
            let xp = x.as_ptr() as *const f64;
            let mut a1 = vdupq_n_f64(0.0);
            let mut a2 = vdupq_n_f64(0.0);
            let mut b1 = vdupq_n_f64(0.0);
            let mut b2 = vdupq_n_f64(0.0);
            let mut i = 0;
            while i + 2 <= n {
                let av0 = vld1q_f64(rp.add(2 * i));
                let xv0 = vld1q_f64(xp.add(2 * i));
                let av1 = vld1q_f64(rp.add(2 * i + 2));
                let xv1 = vld1q_f64(xp.add(2 * i + 2));
                a1 = vfmaq_f64(a1, av0, vdupq_laneq_f64(xv0, 0));
                a2 = vfmaq_f64(a2, vextq_f64(av0, av0, 1), vdupq_laneq_f64(xv0, 1));
                b1 = vfmaq_f64(b1, av1, vdupq_laneq_f64(xv1, 0));
                b2 = vfmaq_f64(b2, vextq_f64(av1, av1, 1), vdupq_laneq_f64(xv1, 1));
                i += 2;
            }
            if i < n {
                let av = vld1q_f64(rp.add(2 * i));
                let xv = vld1q_f64(xp.add(2 * i));
                a1 = vfmaq_f64(a1, av, vdupq_laneq_f64(xv, 0));
                a2 = vfmaq_f64(a2, vextq_f64(av, av, 1), vdupq_laneq_f64(xv, 1));
            }
            let s1 = vaddq_f64(a1, b1);
            let s2 = vaddq_f64(a2, b2);
            let sign: [f64; 2] = [-1.0, 1.0];
            let s2s = vmulq_f64(s2, vld1q_f64(sign.as_ptr()));
            let comb = vaddq_f64(s1, s2s);
            Complex64::new(vgetq_lane_f64(comb, 0), vgetq_lane_f64(comb, 1))
        }
    }

    /// `y[i] += c * row[i]`, or `y[i] += c * conj(row[i])` when `conj`.
    /// Conjugation folds into the broadcast constants so both ops share the
    /// same two-FMA inner loop.
    #[inline]
    fn axpy_row_c64(c: Complex64, conj: bool, row: &[Complex64], y: &mut [Complex64]) {
        let m = row.len();
        debug_assert!(y.len() >= m);
        unsafe {
            let rp = row.as_ptr() as *const f64;
            let yp = y.as_mut_ptr() as *mut f64;
            // y += row*t + swap(row)*s. NoTrans: t = (cr, cr), s = (-ci, ci).
            // ConjTrans: t = (cr, -cr), s = (ci, ci).
            let (t_arr, s_arr): ([f64; 2], [f64; 2]) = if conj {
                ([c.re, -c.re], [c.im, c.im])
            } else {
                ([c.re, c.re], [-c.im, c.im])
            };
            let t = vld1q_f64(t_arr.as_ptr());
            let s = vld1q_f64(s_arr.as_ptr());
            let mut i = 0;
            while i + 2 <= m {
                let av0 = vld1q_f64(rp.add(2 * i));
                let yv0 = vld1q_f64(yp.add(2 * i));
                let yv0 = vfmaq_f64(yv0, av0, t);
                let yv0 = vfmaq_f64(yv0, vextq_f64(av0, av0, 1), s);
                vst1q_f64(yp.add(2 * i), yv0);
                let av1 = vld1q_f64(rp.add(2 * i + 2));
                let yv1 = vld1q_f64(yp.add(2 * i + 2));
                let yv1 = vfmaq_f64(yv1, av1, t);
                let yv1 = vfmaq_f64(yv1, vextq_f64(av1, av1, 1), s);
                vst1q_f64(yp.add(2 * i + 2), yv1);
                i += 2;
            }
            if i < m {
                let av = vld1q_f64(rp.add(2 * i));
                let yv = vld1q_f64(yp.add(2 * i));
                let yv = vfmaq_f64(yv, av, t);
                let yv = vfmaq_f64(yv, vextq_f64(av, av, 1), s);
                vst1q_f64(yp.add(2 * i), yv);
            }
        }
    }

    // Two Complex32 per 128-bit register. `vtrn1q`/`vtrn2q` broadcast the
    // re/im lanes of x across each complex pair; the sign fold is again
    // deferred to the reduction.
    #[inline]
    fn dot_c32(row: &[Complex32], x: &[Complex32]) -> Complex32 {
        let n = row.len();
        debug_assert_eq!(x.len(), n);
        unsafe {
            let rp = row.as_ptr() as *const f32;
            let xp = x.as_ptr() as *const f32;
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut b1 = vdupq_n_f32(0.0);
            let mut b2 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= n {
                let av0 = vld1q_f32(rp.add(2 * i));
                let xv0 = vld1q_f32(xp.add(2 * i));
                let av1 = vld1q_f32(rp.add(2 * i + 4));
                let xv1 = vld1q_f32(xp.add(2 * i + 4));
                a1 = vfmaq_f32(a1, av0, vtrn1q_f32(xv0, xv0));
                a2 = vfmaq_f32(a2, vrev64q_f32(av0), vtrn2q_f32(xv0, xv0));
                b1 = vfmaq_f32(b1, av1, vtrn1q_f32(xv1, xv1));
                b2 = vfmaq_f32(b2, vrev64q_f32(av1), vtrn2q_f32(xv1, xv1));
                i += 4;
            }
            while i + 2 <= n {
                let av = vld1q_f32(rp.add(2 * i));
                let xv = vld1q_f32(xp.add(2 * i));
                a1 = vfmaq_f32(a1, av, vtrn1q_f32(xv, xv));
                a2 = vfmaq_f32(a2, vrev64q_f32(av), vtrn2q_f32(xv, xv));
                i += 2;
            }
            let s1 = vaddq_f32(a1, b1);
            let s2 = vaddq_f32(a2, b2);
            let sign: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
            let s2s = vmulq_f32(s2, vld1q_f32(sign.as_ptr()));
            let comb = vaddq_f32(s1, s2s);
            let pair = vadd_f32(vget_low_f32(comb), vget_high_f32(comb));
            let mut re = vget_lane_f32(pair, 0);
            let mut im = vget_lane_f32(pair, 1);
            while i < n {
                let (a, b) = (row[i], x[i]);
                re += a.re * b.re - a.im * b.im;
                im += a.re * b.im + a.im * b.re;
                i += 1;
            }
            Complex32::new(re, im)
        }
    }

    /// `y[i] += c * row[i]`, or `y[i] += c * conj(row[i])` when `conj`.
    #[inline]
    fn axpy_row_c32(c: Complex32, conj: bool, row: &[Complex32], y: &mut [Complex32]) {
        let m = row.len();
        debug_assert!(y.len() >= m);
        unsafe {
            let rp = row.as_ptr() as *const f32;
            let yp = y.as_mut_ptr() as *mut f32;
            let neg: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
            let pos: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
            let (t, s) = if conj {
                (
                    vmulq_f32(vdupq_n_f32(c.re), vld1q_f32(pos.as_ptr())),
                    vdupq_n_f32(c.im),
                )
            } else {
                (
                    vdupq_n_f32(c.re),
                    vmulq_f32(vdupq_n_f32(c.im), vld1q_f32(neg.as_ptr())),
                )
            };
            let mut i = 0;
            while i + 4 <= m {
                let av0 = vld1q_f32(rp.add(2 * i));
                let yv0 = vld1q_f32(yp.add(2 * i));
                let yv0 = vfmaq_f32(yv0, av0, t);
                let yv0 = vfmaq_f32(yv0, vrev64q_f32(av0), s);
                vst1q_f32(yp.add(2 * i), yv0);
                let av1 = vld1q_f32(rp.add(2 * i + 4));
                let yv1 = vld1q_f32(yp.add(2 * i + 4));
                let yv1 = vfmaq_f32(yv1, av1, t);
                let yv1 = vfmaq_f32(yv1, vrev64q_f32(av1), s);
                vst1q_f32(yp.add(2 * i + 4), yv1);
                i += 4;
            }
            while i + 2 <= m {
                let av = vld1q_f32(rp.add(2 * i));
                let yv = vld1q_f32(yp.add(2 * i));
                let yv = vfmaq_f32(yv, av, t);
                let yv = vfmaq_f32(yv, vrev64q_f32(av), s);
                vst1q_f32(yp.add(2 * i), yv);
                i += 2;
            }
            while i < m {
                let a = if conj { row[i].conj() } else { row[i] };
                y[i] += c * a;
                i += 1;
            }
        }
    }

    pub fn gemv_nt_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for i in 0..m {
            let acc = dot_c64(&a[i * lda..i * lda + n], x);
            y[i] += alpha * acc;
        }
    }

    pub fn gemv_t_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex64::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c64(xj, false, &a[j * lda..j * lda + m], y);
        }
    }

    pub fn gemv_ct_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex64::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c64(xj, true, &a[j * lda..j * lda + m], y);
        }
    }

    pub fn gemv_nt_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for i in 0..m {
            let acc = dot_c32(&a[i * lda..i * lda + n], x);
            y[i] += alpha * acc;
        }
    }

    pub fn gemv_t_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex32::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c32(xj, false, &a[j * lda..j * lda + m], y);
        }
    }

    pub fn gemv_ct_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex32::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c32(xj, true, &a[j * lda..j * lda + m], y);
        }
    }
}

// ---------- x86_64 AVX2/FMA -------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::missing_safety_doc)]
mod avx2 {
    use core::arch::x86_64::*;
    use num_complex::{Complex32, Complex64};

    #[inline]
    pub fn is_available() -> bool {
        !crate::cpu::force_scalar_dispatch()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
    }

    // Two Complex64 per 256-bit register; same two-accumulator sign-fold
    // scheme as the NEON kernel.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_c64(row: &[Complex64], x: &[Complex64]) -> Complex64 {
        let n = row.len();
        debug_assert_eq!(x.len(), n);
        let rp = row.as_ptr() as *const f64;
        let xp = x.as_ptr() as *const f64;
        let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd();
        let mut b1 = _mm256_setzero_pd();
        let mut b2 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 4 <= n {
            let av0 = _mm256_loadu_pd(rp.add(2 * i));
            let xv0 = _mm256_loadu_pd(xp.add(2 * i));
            let av1 = _mm256_loadu_pd(rp.add(2 * i + 4));
            let xv1 = _mm256_loadu_pd(xp.add(2 * i + 4));
            a1 = _mm256_fmadd_pd(av0, _mm256_permute_pd(xv0, 0b0000), a1);
            a2 = _mm256_fmadd_pd(
                _mm256_permute_pd(av0, 0b0101),
                _mm256_permute_pd(xv0, 0b1111),
                a2,
            );
            b1 = _mm256_fmadd_pd(av1, _mm256_permute_pd(xv1, 0b0000), b1);
            b2 = _mm256_fmadd_pd(
                _mm256_permute_pd(av1, 0b0101),
                _mm256_permute_pd(xv1, 0b1111),
                b2,
            );
            i += 4;
        }
        while i + 2 <= n {
            let av = _mm256_loadu_pd(rp.add(2 * i));
            let xv = _mm256_loadu_pd(xp.add(2 * i));
            a1 = _mm256_fmadd_pd(av, _mm256_permute_pd(xv, 0b0000), a1);
            a2 = _mm256_fmadd_pd(
                _mm256_permute_pd(av, 0b0101),
                _mm256_permute_pd(xv, 0b1111),
                a2,
            );
            i += 2;
        }
        let s1 = _mm256_add_pd(a1, b1);
        let s2 = _mm256_add_pd(a2, b2);
        // Negate the re lanes of s2, then fold the two complex lanes.
        let sign = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
        let comb = _mm256_add_pd(s1, _mm256_xor_pd(s2, sign));
        let pair = _mm_add_pd(_mm256_castpd256_pd128(comb), _mm256_extractf128_pd(comb, 1));
        let mut buf = [0.0_f64; 2];
        _mm_storeu_pd(buf.as_mut_ptr(), pair);
        let (mut re, mut im) = (buf[0], buf[1]);
        while i < n {
            let (a, b) = (row[i], x[i]);
            re += a.re * b.re - a.im * b.im;
            im += a.re * b.im + a.im * b.re;
            i += 1;
        }
        Complex64::new(re, im)
    }

    /// `y[i] += c * row[i]`, or `y[i] += c * conj(row[i])` when `conj`.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn axpy_row_c64(c: Complex64, conj: bool, row: &[Complex64], y: &mut [Complex64]) {
        let m = row.len();
        debug_assert!(y.len() >= m);
        let rp = row.as_ptr() as *const f64;
        let yp = y.as_mut_ptr() as *mut f64;
        let (t, s) = if conj {
            (
                _mm256_set_pd(-c.re, c.re, -c.re, c.re),
                _mm256_set1_pd(c.im),
            )
        } else {
            (
                _mm256_set1_pd(c.re),
                _mm256_set_pd(c.im, -c.im, c.im, -c.im),
            )
        };
        let mut i = 0;
        while i + 2 <= m {
            let av = _mm256_loadu_pd(rp.add(2 * i));
            let yv = _mm256_loadu_pd(yp.add(2 * i));
            let yv = _mm256_fmadd_pd(av, t, yv);
            let yv = _mm256_fmadd_pd(_mm256_permute_pd(av, 0b0101), s, yv);
            _mm256_storeu_pd(yp.add(2 * i), yv);
            i += 2;
        }
        while i < m {
            let a = if conj { row[i].conj() } else { row[i] };
            y[i] += c * a;
            i += 1;
        }
    }

    // Four Complex32 per 256-bit register.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_c32(row: &[Complex32], x: &[Complex32]) -> Complex32 {
        let n = row.len();
        debug_assert_eq!(x.len(), n);
        let rp = row.as_ptr() as *const f32;
        let xp = x.as_ptr() as *const f32;
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 4 <= n {
            let av = _mm256_loadu_ps(rp.add(2 * i));
            let xv = _mm256_loadu_ps(xp.add(2 * i));
            let xr = _mm256_permute_ps(xv, 0b10_10_00_00);
            let xi = _mm256_permute_ps(xv, 0b11_11_01_01);
            let avs = _mm256_permute_ps(av, 0b10_11_00_01);
            a1 = _mm256_fmadd_ps(av, xr, a1);
            a2 = _mm256_fmadd_ps(avs, xi, a2);
            i += 4;
        }
        let sign = _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
        let comb = _mm256_add_ps(a1, _mm256_xor_ps(a2, sign));
        // comb holds per-complex (re, im) pairs; gather re/im per 128-lane.
        let g = _mm256_permute_ps(comb, 0b11_01_10_00);
        let sums = _mm_add_ps(_mm256_castps256_ps128(g), _mm256_extractf128_ps(g, 1));
        // sums = (re02, re13, im02, im13); hadd pairs the adjacent lanes.
        let folded = _mm_hadd_ps(sums, sums);
        let mut buf = [0.0_f32; 4];
        _mm_storeu_ps(buf.as_mut_ptr(), folded);
        let (mut re, mut im) = (buf[0], buf[1]);
        while i < n {
            let (a, b) = (row[i], x[i]);
            re += a.re * b.re - a.im * b.im;
            im += a.re * b.im + a.im * b.re;
            i += 1;
        }
        Complex32::new(re, im)
    }

    /// `y[i] += c * row[i]`, or `y[i] += c * conj(row[i])` when `conj`.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn axpy_row_c32(c: Complex32, conj: bool, row: &[Complex32], y: &mut [Complex32]) {
        let m = row.len();
        debug_assert!(y.len() >= m);
        let rp = row.as_ptr() as *const f32;
        let yp = y.as_mut_ptr() as *mut f32;
        let (t, s) = if conj {
            (
                _mm256_set_ps(-c.re, c.re, -c.re, c.re, -c.re, c.re, -c.re, c.re),
                _mm256_set1_ps(c.im),
            )
        } else {
            (
                _mm256_set1_ps(c.re),
                _mm256_set_ps(c.im, -c.im, c.im, -c.im, c.im, -c.im, c.im, -c.im),
            )
        };
        let mut i = 0;
        while i + 4 <= m {
            let av = _mm256_loadu_ps(rp.add(2 * i));
            let yv = _mm256_loadu_ps(yp.add(2 * i));
            let yv = _mm256_fmadd_ps(av, t, yv);
            let yv = _mm256_fmadd_ps(_mm256_permute_ps(av, 0b10_11_00_01), s, yv);
            _mm256_storeu_ps(yp.add(2 * i), yv);
            i += 4;
        }
        while i < m {
            let a = if conj { row[i].conj() } else { row[i] };
            y[i] += c * a;
            i += 1;
        }
    }

    pub unsafe fn gemv_nt_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for i in 0..m {
            let acc = dot_c64(&a[i * lda..i * lda + n], x);
            y[i] += alpha * acc;
        }
    }

    pub unsafe fn gemv_t_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex64::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c64(xj, false, &a[j * lda..j * lda + m], y);
        }
    }

    pub unsafe fn gemv_ct_c64(
        m: usize,
        n: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex64::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c64(xj, true, &a[j * lda..j * lda + m], y);
        }
    }

    pub unsafe fn gemv_nt_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for i in 0..m {
            let acc = dot_c32(&a[i * lda..i * lda + n], x);
            y[i] += alpha * acc;
        }
    }

    pub unsafe fn gemv_t_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex32::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c32(xj, false, &a[j * lda..j * lda + m], y);
        }
    }

    pub unsafe fn gemv_ct_c32(
        m: usize,
        n: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        x: &[Complex32],
        y: &mut [Complex32],
    ) {
        for j in 0..n {
            let xj = alpha * x[j];
            if xj == Complex32::new(0.0, 0.0) {
                continue;
            }
            axpy_row_c32(xj, true, &a[j * lda..j * lda + m], y);
        }
    }
}

// ---------- x86_64 runtime dispatch -----------------------------------------

#[cfg(target_arch = "x86_64")]
mod dispatch {
    use super::{avx2, scalar};
    use num_complex::{Complex32, Complex64};

    macro_rules! dispatch_fn {
        ($name:ident, $t:ty) => {
            pub fn $name(
                m: usize,
                n: usize,
                alpha: $t,
                a: &[$t],
                lda: usize,
                x: &[$t],
                y: &mut [$t],
            ) {
                if avx2::is_available() {
                    unsafe { avx2::$name(m, n, alpha, a, lda, x, y) }
                } else {
                    scalar::$name(m, n, alpha, a, lda, x, y)
                }
            }
        };
    }

    dispatch_fn!(gemv_nt_c64, Complex64);
    dispatch_fn!(gemv_t_c64, Complex64);
    dispatch_fn!(gemv_ct_c64, Complex64);
    dispatch_fn!(gemv_nt_c32, Complex32);
    dispatch_fn!(gemv_t_c32, Complex32);
    dispatch_fn!(gemv_ct_c32, Complex32);
}

// ---------- active backend selection ----------------------------------------

#[cfg(target_arch = "x86_64")]
use dispatch as active;
#[cfg(target_arch = "aarch64")]
use neon as active;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
use scalar as active;

// ---------- correctness tests ------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::blas2::scalar as reference;

    // Deterministic pseudo-random value in [-1, 1) (same scheme as the
    // blas2 mod.rs tests).
    fn prng(k: usize) -> f64 {
        let h = k
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 11;
        (h % 2000) as f64 / 1000.0 - 1.0
    }

    fn assert_close_c64(actual: &[Complex64], reference: &[Complex64], tol: f64, ctx: &str) {
        let err = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let norm = reference.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            err <= tol * norm.max(1.0),
            "{ctx}: err={err} norm={norm} tol={tol}"
        );
    }

    fn assert_close_c32(actual: &[Complex32], reference: &[Complex32], tol: f64, ctx: &str) {
        let err = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).norm_sqr() as f64)
            .sum::<f64>()
            .sqrt();
        let norm = reference
            .iter()
            .map(|v| v.norm_sqr() as f64)
            .sum::<f64>()
            .sqrt();
        assert!(
            err <= tol * norm.max(1.0),
            "{ctx}: err={err} norm={norm} tol={tol}"
        );
    }

    fn run_case_c64(
        op: Transpose,
        m: usize,
        n: usize,
        pad: usize,
        alpha: Complex64,
        beta: Complex64,
    ) {
        // Storage: m rows of length >= n for NoTrans; n rows of length >= m
        // for Trans/ConjTrans.
        let (rows, lda) = match op {
            Transpose::NoTrans => (m, n + pad),
            _ => (n, m + pad),
        };
        let a: Vec<Complex64> = (0..rows * lda)
            .map(|k| Complex64::new(prng(k), prng(rows * lda + k)))
            .collect();
        let x: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(prng(2 * rows * lda + k), prng(3 * rows * lda + k)))
            .collect();
        let y0: Vec<Complex64> = (0..m)
            .map(|k| Complex64::new(prng(4 * rows * lda + k), prng(5 * rows * lda + k)))
            .collect();
        let mut y_ref = y0.clone();
        reference::gemv_c64(op, m, n, alpha, &a, lda, &x, beta, &mut y_ref);
        let mut y_new = y0.clone();
        gemv_c64(op, m, n, alpha, &a, lda, &x, beta, &mut y_new);
        let ctx = format!("c64 op={op:?} m={m} n={n} pad={pad} alpha={alpha:?} beta={beta:?}");
        assert_close_c64(&y_new, &y_ref, 1e-12, &ctx);
    }

    fn run_case_c32(
        op: Transpose,
        m: usize,
        n: usize,
        pad: usize,
        alpha: Complex32,
        beta: Complex32,
    ) {
        let (rows, lda) = match op {
            Transpose::NoTrans => (m, n + pad),
            _ => (n, m + pad),
        };
        let a: Vec<Complex32> = (0..rows * lda)
            .map(|k| Complex32::new(prng(k) as f32, prng(rows * lda + k) as f32))
            .collect();
        let x: Vec<Complex32> = (0..n)
            .map(|k| {
                Complex32::new(
                    prng(2 * rows * lda + k) as f32,
                    prng(3 * rows * lda + k) as f32,
                )
            })
            .collect();
        let y0: Vec<Complex32> = (0..m)
            .map(|k| {
                Complex32::new(
                    prng(4 * rows * lda + k) as f32,
                    prng(5 * rows * lda + k) as f32,
                )
            })
            .collect();
        let mut y_ref = y0.clone();
        reference::gemv_c32(op, m, n, alpha, &a, lda, &x, beta, &mut y_ref);
        let mut y_new = y0.clone();
        gemv_c32(op, m, n, alpha, &a, lda, &x, beta, &mut y_new);
        let ctx = format!("c32 op={op:?} m={m} n={n} pad={pad} alpha={alpha:?} beta={beta:?}");
        assert_close_c32(&y_new, &y_ref, 1e-4, &ctx);
    }

    #[test]
    fn gemv_c64_matches_scalar_all_ops() {
        let alpha = Complex64::new(0.7, -0.2);
        let beta = Complex64::new(0.3, 0.1);
        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            // Odd / non-lane-multiple sizes, plus sizes around the
            // sonium-solvers blas2 dispatch threshold (128×128).
            for (m, n) in [
                (1, 1),
                (2, 3),
                (3, 2),
                (5, 7),
                (8, 8),
                (13, 9),
                (16, 16),
                (31, 17),
                (33, 65),
                (64, 64),
                (100, 128),
                (128, 128),
                (129, 131),
            ] {
                for pad in [0, 3] {
                    run_case_c64(op, m, n, pad, alpha, beta);
                }
            }
        }
    }

    #[test]
    fn gemv_c64_alpha_beta_edge_cases() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            for (m, n) in [(4, 5), (17, 16), (128, 127)] {
                run_case_c64(op, m, n, 0, one, zero);
                run_case_c64(op, m, n, 1, zero, one);
                run_case_c64(op, m, n, 2, Complex64::new(-1.5, 0.5), one);
            }
        }
    }

    #[test]
    fn gemv_c32_matches_scalar_all_ops() {
        let alpha = Complex32::new(0.7, -0.2);
        let beta = Complex32::new(0.3, 0.1);
        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            for (m, n) in [
                (1, 1),
                (2, 3),
                (3, 2),
                (5, 7),
                (8, 8),
                (13, 9),
                (16, 16),
                (31, 17),
                (33, 65),
                (64, 64),
                (100, 128),
                (128, 128),
                (129, 131),
            ] {
                for pad in [0, 3] {
                    run_case_c32(op, m, n, pad, alpha, beta);
                }
            }
        }
    }

    #[test]
    fn gemv_complex_parallel_path_matches_scalar() {
        // Shapes above `PAR_MIN_MN` exercise the rayon row-band
        // (`NoTrans`) and privatised fan-out (`Trans`/`ConjTrans`) paths,
        // including a tall and a wide rectangle.
        let alpha64 = Complex64::new(0.7, -0.2);
        let beta64 = Complex64::new(0.3, 0.1);
        let alpha32 = Complex32::new(0.7, -0.2);
        let beta32 = Complex32::new(0.3, 0.1);
        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            for (m, n) in [(512, 512), (2048, 256), (256, 2048)] {
                run_case_c64(op, m, n, 0, alpha64, beta64);
                run_case_c32(op, m, n, 0, alpha32, beta32);
            }
        }
    }

    #[test]
    fn gemv_c32_alpha_beta_edge_cases() {
        let one = Complex32::new(1.0, 0.0);
        let zero = Complex32::new(0.0, 0.0);
        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            for (m, n) in [(4, 5), (17, 16), (128, 127)] {
                run_case_c32(op, m, n, 0, one, zero);
                run_case_c32(op, m, n, 1, zero, one);
                run_case_c32(op, m, n, 2, Complex32::new(-1.5, 0.5), one);
            }
        }
    }
}
