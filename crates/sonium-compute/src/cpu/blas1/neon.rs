//! AArch64 NEON BLAS-1 kernels.
//!
//! These are unconditionally compiled on `target_arch = "aarch64"` — NEON
//! is part of the AArch64 base ISA, no runtime feature check needed.
//!
//! Each kernel processes a SIMD-wide chunk and finishes with a scalar
//! tail. Reductions use multiple parallel accumulators followed by a
//! horizontal sum so the FMA pipeline isn't serial.

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;
use num_complex::{Complex32, Complex64};

// ---------- f32 ----------

/// `y <- alpha * x + y`. Four-lane f32 NEON with FMA.
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    unsafe {
        let va = vdupq_n_f32(alpha);
        while i + 16 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let x2 = vld1q_f32(x.as_ptr().add(i + 8));
            let x3 = vld1q_f32(x.as_ptr().add(i + 12));
            let y0 = vld1q_f32(y.as_ptr().add(i));
            let y1 = vld1q_f32(y.as_ptr().add(i + 4));
            let y2 = vld1q_f32(y.as_ptr().add(i + 8));
            let y3 = vld1q_f32(y.as_ptr().add(i + 12));
            let r0 = vfmaq_f32(y0, va, x0);
            let r1 = vfmaq_f32(y1, va, x1);
            let r2 = vfmaq_f32(y2, va, x2);
            let r3 = vfmaq_f32(y3, va, x3);
            vst1q_f32(y.as_mut_ptr().add(i), r0);
            vst1q_f32(y.as_mut_ptr().add(i + 4), r1);
            vst1q_f32(y.as_mut_ptr().add(i + 8), r2);
            vst1q_f32(y.as_mut_ptr().add(i + 12), r3);
            i += 16;
        }
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let yv = vld1q_f32(y.as_ptr().add(i));
            let r = vfmaq_f32(yv, va, xv);
            vst1q_f32(y.as_mut_ptr().add(i), r);
            i += 4;
        }
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    let mut i = 0;
    unsafe {
        let va = vdupq_n_f32(alpha);
        while i + 16 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let x2 = vld1q_f32(x.as_ptr().add(i + 8));
            let x3 = vld1q_f32(x.as_ptr().add(i + 12));
            vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(va, x0));
            vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(va, x1));
            vst1q_f32(x.as_mut_ptr().add(i + 8), vmulq_f32(va, x2));
            vst1q_f32(x.as_mut_ptr().add(i + 12), vmulq_f32(va, x3));
            i += 16;
        }
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(va, xv));
            i += 4;
        }
    }
    while i < n {
        x[i] *= alpha;
        i += 1;
    }
}

#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    let mut tail: f32;
    unsafe {
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            a0 = vfmaq_f32(
                a0,
                vld1q_f32(x.as_ptr().add(i)),
                vld1q_f32(y.as_ptr().add(i)),
            );
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(x.as_ptr().add(i + 4)),
                vld1q_f32(y.as_ptr().add(i + 4)),
            );
            a2 = vfmaq_f32(
                a2,
                vld1q_f32(x.as_ptr().add(i + 8)),
                vld1q_f32(y.as_ptr().add(i + 8)),
            );
            a3 = vfmaq_f32(
                a3,
                vld1q_f32(x.as_ptr().add(i + 12)),
                vld1q_f32(y.as_ptr().add(i + 12)),
            );
            i += 16;
        }
        while i + 4 <= n {
            a0 = vfmaq_f32(
                a0,
                vld1q_f32(x.as_ptr().add(i)),
                vld1q_f32(y.as_ptr().add(i)),
            );
            i += 4;
        }
        let sum = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        tail = vaddvq_f32(sum);
    }
    while i < n {
        tail += x[i] * y[i];
        i += 1;
    }
    tail
}

#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let n = x.len();
    let mut i = 0;
    let mut sum_sq: f32;
    unsafe {
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let x2 = vld1q_f32(x.as_ptr().add(i + 8));
            let x3 = vld1q_f32(x.as_ptr().add(i + 12));
            a0 = vfmaq_f32(a0, x0, x0);
            a1 = vfmaq_f32(a1, x1, x1);
            a2 = vfmaq_f32(a2, x2, x2);
            a3 = vfmaq_f32(a3, x3, x3);
            i += 16;
        }
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            a0 = vfmaq_f32(a0, xv, xv);
            i += 4;
        }
        let sum = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        sum_sq = vaddvq_f32(sum);
    }
    while i < n {
        sum_sq += x[i] * x[i];
        i += 1;
    }
    sum_sq.sqrt()
}

// ---------- f64 ----------

#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    unsafe {
        let va = vdupq_n_f64(alpha);
        while i + 8 <= n {
            let x0 = vld1q_f64(x.as_ptr().add(i));
            let x1 = vld1q_f64(x.as_ptr().add(i + 2));
            let x2 = vld1q_f64(x.as_ptr().add(i + 4));
            let x3 = vld1q_f64(x.as_ptr().add(i + 6));
            let y0 = vld1q_f64(y.as_ptr().add(i));
            let y1 = vld1q_f64(y.as_ptr().add(i + 2));
            let y2 = vld1q_f64(y.as_ptr().add(i + 4));
            let y3 = vld1q_f64(y.as_ptr().add(i + 6));
            vst1q_f64(y.as_mut_ptr().add(i), vfmaq_f64(y0, va, x0));
            vst1q_f64(y.as_mut_ptr().add(i + 2), vfmaq_f64(y1, va, x1));
            vst1q_f64(y.as_mut_ptr().add(i + 4), vfmaq_f64(y2, va, x2));
            vst1q_f64(y.as_mut_ptr().add(i + 6), vfmaq_f64(y3, va, x3));
            i += 8;
        }
        while i + 2 <= n {
            let xv = vld1q_f64(x.as_ptr().add(i));
            let yv = vld1q_f64(y.as_ptr().add(i));
            vst1q_f64(y.as_mut_ptr().add(i), vfmaq_f64(yv, va, xv));
            i += 2;
        }
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let mut i = 0;
    unsafe {
        let va = vdupq_n_f64(alpha);
        while i + 8 <= n {
            let x0 = vld1q_f64(x.as_ptr().add(i));
            let x1 = vld1q_f64(x.as_ptr().add(i + 2));
            let x2 = vld1q_f64(x.as_ptr().add(i + 4));
            let x3 = vld1q_f64(x.as_ptr().add(i + 6));
            vst1q_f64(x.as_mut_ptr().add(i), vmulq_f64(va, x0));
            vst1q_f64(x.as_mut_ptr().add(i + 2), vmulq_f64(va, x1));
            vst1q_f64(x.as_mut_ptr().add(i + 4), vmulq_f64(va, x2));
            vst1q_f64(x.as_mut_ptr().add(i + 6), vmulq_f64(va, x3));
            i += 8;
        }
        while i + 2 <= n {
            let xv = vld1q_f64(x.as_ptr().add(i));
            vst1q_f64(x.as_mut_ptr().add(i), vmulq_f64(va, xv));
            i += 2;
        }
    }
    while i < n {
        x[i] *= alpha;
        i += 1;
    }
}

#[inline]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    let mut tail: f64;
    unsafe {
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        while i + 8 <= n {
            a0 = vfmaq_f64(
                a0,
                vld1q_f64(x.as_ptr().add(i)),
                vld1q_f64(y.as_ptr().add(i)),
            );
            a1 = vfmaq_f64(
                a1,
                vld1q_f64(x.as_ptr().add(i + 2)),
                vld1q_f64(y.as_ptr().add(i + 2)),
            );
            a2 = vfmaq_f64(
                a2,
                vld1q_f64(x.as_ptr().add(i + 4)),
                vld1q_f64(y.as_ptr().add(i + 4)),
            );
            a3 = vfmaq_f64(
                a3,
                vld1q_f64(x.as_ptr().add(i + 6)),
                vld1q_f64(y.as_ptr().add(i + 6)),
            );
            i += 8;
        }
        while i + 2 <= n {
            a0 = vfmaq_f64(
                a0,
                vld1q_f64(x.as_ptr().add(i)),
                vld1q_f64(y.as_ptr().add(i)),
            );
            i += 2;
        }
        let sum = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        tail = vaddvq_f64(sum);
    }
    while i < n {
        tail += x[i] * y[i];
        i += 1;
    }
    tail
}

#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let n = x.len();
    let mut i = 0;
    let mut sum_sq: f64;
    unsafe {
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        while i + 8 <= n {
            let x0 = vld1q_f64(x.as_ptr().add(i));
            let x1 = vld1q_f64(x.as_ptr().add(i + 2));
            let x2 = vld1q_f64(x.as_ptr().add(i + 4));
            let x3 = vld1q_f64(x.as_ptr().add(i + 6));
            a0 = vfmaq_f64(a0, x0, x0);
            a1 = vfmaq_f64(a1, x1, x1);
            a2 = vfmaq_f64(a2, x2, x2);
            a3 = vfmaq_f64(a3, x3, x3);
            i += 8;
        }
        while i + 2 <= n {
            let xv = vld1q_f64(x.as_ptr().add(i));
            a0 = vfmaq_f64(a0, xv, xv);
            i += 2;
        }
        let sum = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        sum_sq = vaddvq_f64(sum);
    }
    while i < n {
        sum_sq += x[i] * x[i];
        i += 1;
    }
    sum_sq.sqrt()
}

// ---------- Complex32 ----------
//
// Memory layout of `num_complex::Complex<f32>` is `#[repr(C)]`
// `{ re: f32, im: f32 }` — a `&[Complex32]` is two complex values per
// `float32x4_t` in interleaved `[re0, im0, re1, im1]` order.
//
// For `y += alpha * x` with `alpha = ar + i*ai`:
//   (alpha * x).re = ar*x.re - ai*x.im
//   (alpha * x).im = ar*x.im + ai*x.re
//
// Strategy per 128-bit chunk:
//   x   = [x0.re, x0.im, x1.re, x1.im]
//   xs  = vrev64q_f32(x) → [x0.im, x0.re, x1.im, x1.re]   (swap re/im within each complex)
//   alpha_re_broadcast = [ar, ar, ar, ar]
//   alpha_im_swizzle   = [-ai, ai, -ai, ai]                (build once via vmulq)
//   acc = vfmaq(y, alpha_re_broadcast, x)                  + ar*[xr, xi]
//   acc = vfmaq(acc, alpha_im_swizzle, xs)                 + [-ai*xi, ai*xr] = [(-ai*xi), (ai*xr)]
// Combined: lane 0 = y0.re + ar*x0.re - ai*x0.im ✓
//           lane 1 = y0.im + ar*x0.im + ai*x0.re ✓

#[inline]
pub fn axpy_c32(alpha: Complex32, x: &[Complex32], y: &mut [Complex32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    unsafe {
        let ar = vdupq_n_f32(alpha.re);
        // `[-ai, ai, -ai, ai]`
        let pos_neg: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
        let ai_signed = vmulq_f32(vdupq_n_f32(alpha.im), vld1q_f32(pos_neg.as_ptr()));
        let xp = x.as_ptr() as *const f32;
        let yp = y.as_mut_ptr() as *mut f32;
        while i + 2 <= n {
            let xv = vld1q_f32(xp.add(2 * i));
            let xs = vrev64q_f32(xv);
            let yv = vld1q_f32(yp.add(2 * i));
            let t = vfmaq_f32(yv, ar, xv);
            let t = vfmaq_f32(t, ai_signed, xs);
            vst1q_f32(yp.add(2 * i), t);
            i += 2;
        }
    }
    let ar = alpha.re;
    let ai = alpha.im;
    while i < n {
        let r = ar * x[i].re - ai * x[i].im;
        let im = ar * x[i].im + ai * x[i].re;
        y[i].re += r;
        y[i].im += im;
        i += 1;
    }
}

#[inline]
pub fn scal_c32(alpha: Complex32, x: &mut [Complex32]) {
    let n = x.len();
    let mut i = 0;
    unsafe {
        let ar = vdupq_n_f32(alpha.re);
        let pos_neg: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
        let ai_signed = vmulq_f32(vdupq_n_f32(alpha.im), vld1q_f32(pos_neg.as_ptr()));
        let xp = x.as_mut_ptr() as *mut f32;
        while i + 2 <= n {
            let xv = vld1q_f32(xp.add(2 * i));
            let xs = vrev64q_f32(xv);
            let t = vmulq_f32(ar, xv);
            let t = vfmaq_f32(t, ai_signed, xs);
            vst1q_f32(xp.add(2 * i), t);
            i += 2;
        }
    }
    let ar = alpha.re;
    let ai = alpha.im;
    while i < n {
        let r = ar * x[i].re - ai * x[i].im;
        let im = ar * x[i].im + ai * x[i].re;
        x[i].re = r;
        x[i].im = im;
        i += 1;
    }
}

#[inline]
pub fn nrm2_c32(x: &[Complex32]) -> f32 {
    let n = x.len();
    let mut i;
    let mut sum_sq: f32;
    unsafe {
        // Treat as flat &[f32] of length 2n and reduce sum of squares.
        let xp = x.as_ptr() as *const f32;
        let m = 2 * n;
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut j = 0;
        while j + 16 <= m {
            let v0 = vld1q_f32(xp.add(j));
            let v1 = vld1q_f32(xp.add(j + 4));
            let v2 = vld1q_f32(xp.add(j + 8));
            let v3 = vld1q_f32(xp.add(j + 12));
            a0 = vfmaq_f32(a0, v0, v0);
            a1 = vfmaq_f32(a1, v1, v1);
            a2 = vfmaq_f32(a2, v2, v2);
            a3 = vfmaq_f32(a3, v3, v3);
            j += 16;
        }
        while j + 4 <= m {
            let v = vld1q_f32(xp.add(j));
            a0 = vfmaq_f32(a0, v, v);
            j += 4;
        }
        let sum = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        sum_sq = vaddvq_f32(sum);
        i = j / 2;
    }
    while i < n {
        sum_sq += x[i].re * x[i].re + x[i].im * x[i].im;
        i += 1;
    }
    sum_sq.sqrt()
}

/// Hermitian dot for Complex32. Same lane trick as axpy_c32, with
/// `y` filling the role that `alpha` played there.
///
/// `(conj(x) * y).re = x.re*y.re + x.im*y.im`
/// `(conj(x) * y).im = x.re*y.im - x.im*y.re`
#[inline]
pub fn dotc_c32(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    let mut tail: Complex32;
    unsafe {
        let xp = x.as_ptr() as *const f32;
        let yp = y.as_ptr() as *const f32;
        // Accumulate two complex partial sums per 128-bit lane pair so
        // we get four parallel chains.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let pos_neg: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
        let sign = vld1q_f32(pos_neg.as_ptr());
        while i + 4 <= n {
            // First pair.
            let xv = vld1q_f32(xp.add(2 * i));
            let yv = vld1q_f32(yp.add(2 * i));
            // lane: [x.re*y.re, x.im*y.im, x.re*y.re, x.im*y.im]
            let p_diag = vmulq_f32(xv, yv);
            // [x.re*y.im, x.im*y.re, ...]: swap re/im in y
            let yr = vrev64q_f32(yv);
            let p_cross = vmulq_f32(xv, yr);
            // Combine real part: diag.0 + diag.1 (pairwise add within each complex).
            // The result lays out [sum_re_chunk_0, sum_im_chunk_0, sum_re_chunk_1, sum_im_chunk_1].
            // Real lane = diag.even + diag.odd
            // Imag lane = cross.even - cross.odd   (signed via `sign`)
            let p_cross_signed = vmulq_f32(p_cross, sign);
            // re-lane gets `vpaddq_f32(p_diag, p_cross_signed)` would interleave;
            // simpler: directly accumulate diag with `+` for real, cross with sign for imag.
            // Build per-complex contribution: [p_diag.lane0 + p_diag.lane1, p_cross_signed.lane0 + p_cross_signed.lane1, ...]
            let re_im_pair = vpaddq_f32(p_diag, p_cross_signed);
            // re_im_pair now: [re0, re1, im0, im1] (vpaddq output ordering).
            // We need to accumulate alternating re/im per complex into a 2-lane stream.
            // Rearrange to [re0, im0, re1, im1] for direct accumulation alongside the input layout.
            // re0 lives at lane 0, re1 at lane 1, im0 at lane 2, im1 at lane 3.
            let perm = vcombine_f32(
                vzip1_f32(vget_low_f32(re_im_pair), vget_high_f32(re_im_pair)),
                vzip2_f32(vget_low_f32(re_im_pair), vget_high_f32(re_im_pair)),
            );
            acc0 = vaddq_f32(acc0, perm);

            // Second pair (offset by 2 complex values).
            let xv2 = vld1q_f32(xp.add(2 * (i + 2)));
            let yv2 = vld1q_f32(yp.add(2 * (i + 2)));
            let p_diag2 = vmulq_f32(xv2, yv2);
            let yr2 = vrev64q_f32(yv2);
            let p_cross2 = vmulq_f32(xv2, yr2);
            let p_cross_signed2 = vmulq_f32(p_cross2, sign);
            let re_im_pair2 = vpaddq_f32(p_diag2, p_cross_signed2);
            let perm2 = vcombine_f32(
                vzip1_f32(vget_low_f32(re_im_pair2), vget_high_f32(re_im_pair2)),
                vzip2_f32(vget_low_f32(re_im_pair2), vget_high_f32(re_im_pair2)),
            );
            acc1 = vaddq_f32(acc1, perm2);

            i += 4;
        }
        let acc = vaddq_f32(acc0, acc1);
        // acc holds 2 complex partial sums: [r0, i0, r1, i1].
        let re = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 2);
        let im = vgetq_lane_f32(acc, 1) + vgetq_lane_f32(acc, 3);
        tail = Complex32::new(re, im);
    }
    while i < n {
        tail.re += x[i].re * y[i].re + x[i].im * y[i].im;
        tail.im += x[i].re * y[i].im - x[i].im * y[i].re;
        i += 1;
    }
    tail
}

// ---------- Complex64 ----------
//
// One Complex64 per 128-bit register — NEON parallelism is marginal vs.
// a well-scheduled scalar version. We still SIMD it for the FMA pipeline.

#[inline]
pub fn axpy_c64(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut i = 0;
    unsafe {
        let ar = vdupq_n_f64(alpha.re);
        let pos_neg: [f64; 2] = [-1.0, 1.0];
        let ai_signed = vmulq_f64(vdupq_n_f64(alpha.im), vld1q_f64(pos_neg.as_ptr()));
        let xp = x.as_ptr() as *const f64;
        let yp = y.as_mut_ptr() as *mut f64;
        while i < n {
            let xv = vld1q_f64(xp.add(2 * i));
            // Swap (re, im).
            let xs = vextq_f64(xv, xv, 1);
            let yv = vld1q_f64(yp.add(2 * i));
            let t = vfmaq_f64(yv, ar, xv);
            let t = vfmaq_f64(t, ai_signed, xs);
            vst1q_f64(yp.add(2 * i), t);
            i += 1;
        }
    }
}

#[inline]
pub fn scal_c64(alpha: Complex64, x: &mut [Complex64]) {
    let n = x.len();
    let mut i = 0;
    unsafe {
        let ar = vdupq_n_f64(alpha.re);
        let pos_neg: [f64; 2] = [-1.0, 1.0];
        let ai_signed = vmulq_f64(vdupq_n_f64(alpha.im), vld1q_f64(pos_neg.as_ptr()));
        let xp = x.as_mut_ptr() as *mut f64;
        while i < n {
            let xv = vld1q_f64(xp.add(2 * i));
            let xs = vextq_f64(xv, xv, 1);
            let t = vmulq_f64(ar, xv);
            let t = vfmaq_f64(t, ai_signed, xs);
            vst1q_f64(xp.add(2 * i), t);
            i += 1;
        }
    }
}

#[inline]
pub fn nrm2_c64(x: &[Complex64]) -> f64 {
    let n = x.len();
    let sum_sq: f64;
    unsafe {
        let xp = x.as_ptr() as *const f64;
        let m = 2 * n;
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        let mut j = 0;
        while j + 8 <= m {
            let v0 = vld1q_f64(xp.add(j));
            let v1 = vld1q_f64(xp.add(j + 2));
            let v2 = vld1q_f64(xp.add(j + 4));
            let v3 = vld1q_f64(xp.add(j + 6));
            a0 = vfmaq_f64(a0, v0, v0);
            a1 = vfmaq_f64(a1, v1, v1);
            a2 = vfmaq_f64(a2, v2, v2);
            a3 = vfmaq_f64(a3, v3, v3);
            j += 8;
        }
        while j + 2 <= m {
            let v = vld1q_f64(xp.add(j));
            a0 = vfmaq_f64(a0, v, v);
            j += 2;
        }
        let sum = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        sum_sq = vaddvq_f64(sum);
    }
    sum_sq.sqrt()
}

#[inline]
pub fn dotc_c64(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    for i in 0..n {
        re += x[i].re * y[i].re + x[i].im * y[i].im;
        im += x[i].re * y[i].im - x[i].im * y[i].re;
    }
    Complex64::new(re, im)
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
    unsafe {
        while i + 16 <= n {
            let mut a0 = vld1q_f32(y.as_ptr().add(i));
            let mut a1 = vld1q_f32(y.as_ptr().add(i + 4));
            let mut a2 = vld1q_f32(y.as_ptr().add(i + 8));
            let mut a3 = vld1q_f32(y.as_ptr().add(i + 12));
            for bb in 0..k {
                let va = vdupq_n_f32(alphas[bb]);
                let xb0 = vld1q_f32(x_basis[bb].as_ptr().add(i));
                let xb1 = vld1q_f32(x_basis[bb].as_ptr().add(i + 4));
                let xb2 = vld1q_f32(x_basis[bb].as_ptr().add(i + 8));
                let xb3 = vld1q_f32(x_basis[bb].as_ptr().add(i + 12));
                a0 = vfmaq_f32(a0, va, xb0);
                a1 = vfmaq_f32(a1, va, xb1);
                a2 = vfmaq_f32(a2, va, xb2);
                a3 = vfmaq_f32(a3, va, xb3);
            }
            vst1q_f32(y.as_mut_ptr().add(i), a0);
            vst1q_f32(y.as_mut_ptr().add(i + 4), a1);
            vst1q_f32(y.as_mut_ptr().add(i + 8), a2);
            vst1q_f32(y.as_mut_ptr().add(i + 12), a3);
            i += 16;
        }
        while i + 4 <= n {
            let mut acc = vld1q_f32(y.as_ptr().add(i));
            for bb in 0..k {
                let va = vdupq_n_f32(alphas[bb]);
                let xb = vld1q_f32(x_basis[bb].as_ptr().add(i));
                acc = vfmaq_f32(acc, va, xb);
            }
            vst1q_f32(y.as_mut_ptr().add(i), acc);
            i += 4;
        }
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
    unsafe {
        while i + 8 <= n {
            let mut a0 = vld1q_f64(y.as_ptr().add(i));
            let mut a1 = vld1q_f64(y.as_ptr().add(i + 2));
            let mut a2 = vld1q_f64(y.as_ptr().add(i + 4));
            let mut a3 = vld1q_f64(y.as_ptr().add(i + 6));
            for bb in 0..k {
                let va = vdupq_n_f64(alphas[bb]);
                let xb0 = vld1q_f64(x_basis[bb].as_ptr().add(i));
                let xb1 = vld1q_f64(x_basis[bb].as_ptr().add(i + 2));
                let xb2 = vld1q_f64(x_basis[bb].as_ptr().add(i + 4));
                let xb3 = vld1q_f64(x_basis[bb].as_ptr().add(i + 6));
                a0 = vfmaq_f64(a0, va, xb0);
                a1 = vfmaq_f64(a1, va, xb1);
                a2 = vfmaq_f64(a2, va, xb2);
                a3 = vfmaq_f64(a3, va, xb3);
            }
            vst1q_f64(y.as_mut_ptr().add(i), a0);
            vst1q_f64(y.as_mut_ptr().add(i + 2), a1);
            vst1q_f64(y.as_mut_ptr().add(i + 4), a2);
            vst1q_f64(y.as_mut_ptr().add(i + 6), a3);
            i += 8;
        }
        while i + 2 <= n {
            let mut acc = vld1q_f64(y.as_ptr().add(i));
            for bb in 0..k {
                let va = vdupq_n_f64(alphas[bb]);
                let xb = vld1q_f64(x_basis[bb].as_ptr().add(i));
                acc = vfmaq_f64(acc, va, xb);
            }
            vst1q_f64(y.as_mut_ptr().add(i), acc);
            i += 2;
        }
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
    let mut i = 0;
    unsafe {
        let pos_neg: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
        let sign = vld1q_f32(pos_neg.as_ptr());
        let yp = y.as_mut_ptr() as *mut f32;
        while i + 4 <= n {
            let mut acc0 = vld1q_f32(yp.add(2 * i));
            let mut acc1 = vld1q_f32(yp.add(2 * i + 4));
            for bb in 0..k {
                let ar = vdupq_n_f32(alphas[bb].re);
                let ai_signed = vmulq_f32(vdupq_n_f32(alphas[bb].im), sign);
                let xp = x_basis[bb].as_ptr() as *const f32;
                let xv0 = vld1q_f32(xp.add(2 * i));
                let xv1 = vld1q_f32(xp.add(2 * i + 4));
                let xs0 = vrev64q_f32(xv0);
                let xs1 = vrev64q_f32(xv1);
                acc0 = vfmaq_f32(acc0, ar, xv0);
                acc0 = vfmaq_f32(acc0, ai_signed, xs0);
                acc1 = vfmaq_f32(acc1, ar, xv1);
                acc1 = vfmaq_f32(acc1, ai_signed, xs1);
            }
            vst1q_f32(yp.add(2 * i), acc0);
            vst1q_f32(yp.add(2 * i + 4), acc1);
            i += 4;
        }
    }
    while i < n {
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
        i += 1;
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
    let mut i = 0;
    unsafe {
        let pos_neg: [f64; 2] = [-1.0, 1.0];
        let sign = vld1q_f64(pos_neg.as_ptr());
        let yp = y.as_mut_ptr() as *mut f64;
        while i < n {
            let mut acc = vld1q_f64(yp.add(2 * i));
            for bb in 0..k {
                let ar = vdupq_n_f64(alphas[bb].re);
                let ai_signed = vmulq_f64(vdupq_n_f64(alphas[bb].im), sign);
                let xp = x_basis[bb].as_ptr() as *const f64;
                let xv = vld1q_f64(xp.add(2 * i));
                let xs = vextq_f64(xv, xv, 1);
                acc = vfmaq_f64(acc, ar, xv);
                acc = vfmaq_f64(acc, ai_signed, xs);
            }
            vst1q_f64(yp.add(2 * i), acc);
            i += 1;
        }
    }
}
