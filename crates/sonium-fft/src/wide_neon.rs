//! NEON-specialised wide-butterfly stages. AArch64-only.
//!
//! Processes consecutive radix-r butterflies per loop iteration using
//! the planar split-complex layout in NEON registers. Loads / stores go
//! through `vld2q_*` / `vst2q_*` which deinterleave / interleave the
//! contiguous complex stride pattern in a single instruction.
//!
//! Stage twiddles are pre-shuffled into contiguous per-k arrays, which
//! gives the hot loops stride-1 vector twiddle loads.
//!
//! All stages preserve the exact memory layout and numerical contract of
//! the scalar codelets in [`crate::codelets::scalar`] — they are drop-in
//! replacements for matching scalar types.
//!
//! Every `pub unsafe fn` in this module requires AArch64 NEON (provided by
//! the module's target feature), valid stage dimensions and complete twiddle
//! slices. Source and destination slices must be disjoint unless the
//! function name explicitly says `in_place`; fused pair routines additionally
//! require the documented fixed lengths in their debug assertions.

#![allow(dead_code)]

use std::arch::aarch64::*;

use num_complex::Complex;

use crate::twiddles::StageTwiddles;

/// Load two consecutive Complex64 twiddles into a `(re_v, im_v)` pair
/// of NEON 2-lane vectors via the `vld2q_f64` deinterleave intrinsic.
/// The slice must be stride-1 contiguous in memory — guaranteed by the
/// [`StageTwiddles`] layout.
#[inline]
unsafe fn load_twiddles2(tw: *const Complex<f64>) -> (float64x2_t, float64x2_t) {
    // SAFETY: AArch64 mandates NEON; caller must ensure 4 f64 of valid
    // memory at `tw`.
    unsafe {
        let pair = vld2q_f64(tw as *const f64);
        (pair.0, pair.1)
    }
}

/// Load four consecutive Complex32 twiddles into a `(re_v, im_v)` pair.
/// `StageTwiddles` stores twiddles stride-1, so this is one deinterleave
/// load for four lane-parallel butterflies.
#[inline]
unsafe fn load_twiddles4_f32(tw: *const Complex<f32>) -> (float32x4_t, float32x4_t) {
    unsafe {
        let pair = vld2q_f32(tw as *const f32);
        (pair.0, pair.1)
    }
}

/// Alternate twiddle pack via two scalar loads + lane writes. Used to
/// A/B the vld2q deinterleave path; kept private and unused in the hot
/// path. Remove once vld2q is verified faster across N.
#[inline]
#[allow(dead_code)]
unsafe fn pack_two_twiddles(tw: *const Complex<f64>) -> (float64x2_t, float64x2_t) {
    unsafe {
        let t0 = *tw;
        let t1 = *tw.add(1);
        let re = vsetq_lane_f64(t1.re, vdupq_n_f64(t0.re), 1);
        let im = vsetq_lane_f64(t1.im, vdupq_n_f64(t0.im), 1);
        (re, im)
    }
}

/// Lane-parallel radix-2 butterfly. Same shape as the scalar codelet,
/// just operating on two consecutive butterflies at once via planar
/// `float64x2_t` lanes.
///
/// b is implicitly multiplied by the supplied `tw_re_v + i·tw_im_v` before
/// the butterfly add/sub; pass `(1, 0)` in both lanes to skip (e.g. when
/// `l = 0`).
#[inline]
unsafe fn radix2_simd(
    a_re: float64x2_t,
    a_im: float64x2_t,
    b_re: float64x2_t,
    b_im: float64x2_t,
    tw_re: float64x2_t,
    tw_im: float64x2_t,
) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) {
    // SAFETY: NEON is mandatory on AArch64; intrinsics are sound under
    // the caller's `target_feature` gate.
    unsafe {
        // b *= tw:
        //   re' = b_re·tw_re − b_im·tw_im
        //   im' = b_re·tw_im + b_im·tw_re
        let bt_re = vfmsq_f64(vmulq_f64(b_re, tw_re), b_im, tw_im);
        let bt_im = vfmaq_f64(vmulq_f64(b_re, tw_im), b_im, tw_re);

        let y0_re = vaddq_f64(a_re, bt_re);
        let y0_im = vaddq_f64(a_im, bt_im);
        let y1_re = vsubq_f64(a_re, bt_re);
        let y1_im = vsubq_f64(a_im, bt_im);
        (y0_re, y0_im, y1_re, y1_im)
    }
}

/// NEON radix-2 stage. Memory layout and semantics identical to the
/// scalar `stockham::stage_radix2` for `T = f64`. Falls back to scalar
/// for tail `l` positions when `m` isn't a multiple of 2 (so length-3,
/// length-5 etc. factors compose cleanly with this path).
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix2_f64(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
) {
    // SAFETY: AArch64 + NEON is enforced by the `target_feature` attribute.
    unsafe {
        stage_tw.debug_assert_stage_shape(2, stride, src.len(), dst.len());
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            while l + 2 <= m {
                // Single contiguous load for {a[l], a[l+1]} via deinterleave.
                let a_ptr = src.as_ptr().add(base_in + l) as *const f64;
                let b_ptr = src.as_ptr().add(base_in + l + stride) as *const f64;
                let a = vld2q_f64(a_ptr);
                let b = vld2q_f64(b_ptr);

                // Twiddles are now stride-1: one vld2q for two of them.
                let (tw_re, tw_im) = load_twiddles2(tw1.as_ptr().add(l));

                let (y0_re, y0_im, y1_re, y1_im) = radix2_simd(a.0, a.1, b.0, b.1, tw_re, tw_im);

                let y0_ptr = dst.as_mut_ptr().add(base_out + l) as *mut f64;
                let y1_ptr = dst.as_mut_ptr().add(base_out + m + l) as *mut f64;
                vst2q_f64(y0_ptr, float64x2x2_t(y0_re, y0_im));
                vst2q_f64(y1_ptr, float64x2x2_t(y1_re, y1_im));

                l += 2;
            }

            // Scalar tail for the odd-l remainder (and for the whole
            // stage when m < 2). Skip the identity multiply at l = 0.
            while l < m {
                let a = src[base_in + l];
                let b = if l == 0 {
                    src[base_in + l + stride]
                } else {
                    src[base_in + l + stride] * tw1[l]
                };
                dst[base_out + l] = a + b;
                dst[base_out + m + l] = a - b;
                l += 1;
            }
        }
    }
}

/// Complex multiply `a * b`, planar lanes. Returns `(re, im)` lane vecs.
#[inline]
unsafe fn cmul_simd(
    a_re: float64x2_t,
    a_im: float64x2_t,
    b_re: float64x2_t,
    b_im: float64x2_t,
) -> (float64x2_t, float64x2_t) {
    // SAFETY: NEON mandatory on AArch64.
    unsafe {
        // (a_re + i·a_im) * (b_re + i·b_im) =
        //   (a_re·b_re − a_im·b_im) + i·(a_re·b_im + a_im·b_re)
        let re = vfmsq_f64(vmulq_f64(a_re, b_re), a_im, b_im);
        let im = vfmaq_f64(vmulq_f64(a_re, b_im), a_im, b_re);
        (re, im)
    }
}

type ComplexPairF64 = (float64x2_t, float64x2_t);
type Radix4PairF64 = (
    ComplexPairF64,
    ComplexPairF64,
    ComplexPairF64,
    ComplexPairF64,
);

#[inline(always)]
unsafe fn load_packed_radix4_twiddles(ptr: *const f64) -> [ComplexPairF64; 3] {
    unsafe {
        [
            (vld1q_f64(ptr), vld1q_f64(ptr.add(2))),
            (vld1q_f64(ptr.add(4)), vld1q_f64(ptr.add(6))),
            (vld1q_f64(ptr.add(8)), vld1q_f64(ptr.add(10))),
        ]
    }
}

/// Two lane-parallel radix-4 butterflies, including the three twiddle
/// multiplies. Each lane is an adjacent Stockham `l` value.
#[inline(always)]
unsafe fn radix4_twiddled_simd<const FORWARD: bool>(
    x0: ComplexPairF64,
    x1: ComplexPairF64,
    x2: ComplexPairF64,
    x3: ComplexPairF64,
    tw1: ComplexPairF64,
    tw2: ComplexPairF64,
    tw3: ComplexPairF64,
) -> Radix4PairF64 {
    unsafe {
        let x1 = cmul_simd(x1.0, x1.1, tw1.0, tw1.1);
        let x2 = cmul_simd(x2.0, x2.1, tw2.0, tw2.1);
        let x3 = cmul_simd(x3.0, x3.1, tw3.0, tw3.1);

        let a = (vaddq_f64(x0.0, x2.0), vaddq_f64(x0.1, x2.1));
        let b = (vsubq_f64(x0.0, x2.0), vsubq_f64(x0.1, x2.1));
        let c = (vaddq_f64(x1.0, x3.0), vaddq_f64(x1.1, x3.1));
        let d = (vsubq_f64(x1.0, x3.0), vsubq_f64(x1.1, x3.1));
        let id = if FORWARD {
            (d.1, vnegq_f64(d.0))
        } else {
            (vnegq_f64(d.1), d.0)
        };

        (
            (vaddq_f64(a.0, c.0), vaddq_f64(a.1, c.1)),
            (vaddq_f64(b.0, id.0), vaddq_f64(b.1, id.1)),
            (vsubq_f64(a.0, c.0), vsubq_f64(a.1, c.1)),
            (vsubq_f64(b.0, id.0), vsubq_f64(b.1, id.1)),
        )
    }
}

/// Complex multiply `a * b`, planar f32 lanes.
#[inline]
unsafe fn cmul_simd_f32(
    a_re: float32x4_t,
    a_im: float32x4_t,
    b_re: float32x4_t,
    b_im: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    unsafe {
        let re = vfmsq_f32(vmulq_f32(a_re, b_re), a_im, b_im);
        let im = vfmaq_f32(vmulq_f32(a_re, b_im), a_im, b_re);
        (re, im)
    }
}

/// NEON radix-4 stage. Mirrors `stockham::stage_radix4::<f64>`.
///
/// Per `g`, sweeps `l` in chunks of 2. Each chunk pulls 4 contiguous
/// Complex64 lanes via `vld2q_f64` for the four input streams, applies
/// the three pre-baked twiddles, runs the radix-4 butterfly fused as
/// two radix-2 butterflies + the ±i rotation, and writes the four output
/// streams back interleaved.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix4_f64(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    // SAFETY: NEON enforced by target_feature.
    unsafe {
        stage_tw.debug_assert_stage_shape(4, stride, src.len(), dst.len());
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let dir_v = vdupq_n_f64(dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            while l + 2 <= m {
                let x0_ptr = src.as_ptr().add(base_in + l) as *const f64;
                let x1_ptr = src.as_ptr().add(base_in + l + stride) as *const f64;
                let x2_ptr = src.as_ptr().add(base_in + l + 2 * stride) as *const f64;
                let x3_ptr = src.as_ptr().add(base_in + l + 3 * stride) as *const f64;
                let x0 = vld2q_f64(x0_ptr);
                let x1 = vld2q_f64(x1_ptr);
                let x2 = vld2q_f64(x2_ptr);
                let x3 = vld2q_f64(x3_ptr);

                // Stride-1 twiddle loads: one vld2q each.
                let (tw1_re, tw1_im) = load_twiddles2(tw1.as_ptr().add(l));
                let (tw2_re, tw2_im) = load_twiddles2(tw2.as_ptr().add(l));
                let (tw3_re, tw3_im) = load_twiddles2(tw3.as_ptr().add(l));

                let (x1_re, x1_im) = cmul_simd(x1.0, x1.1, tw1_re, tw1_im);
                let (x2_re, x2_im) = cmul_simd(x2.0, x2.1, tw2_re, tw2_im);
                let (x3_re, x3_im) = cmul_simd(x3.0, x3.1, tw3_re, tw3_im);

                // Radix-4 butterfly. Fuse-as-two-radix-2:
                //   a = x0 + x2,  b = x0 − x2,  c = x1 + x3,  d = x1 − x3
                //   id = ±i · d (sign from dir_sign)
                //   y0 = a + c,  y2 = a − c,  y1 = b + id,  y3 = b − id
                let a_re = vaddq_f64(x0.0, x2_re);
                let a_im = vaddq_f64(x0.1, x2_im);
                let b_re = vsubq_f64(x0.0, x2_re);
                let b_im = vsubq_f64(x0.1, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                // id = (±i) · d = (∓d_im, ±d_re).
                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                let y0_ptr = dst.as_mut_ptr().add(base_out + l) as *mut f64;
                let y1_ptr = dst.as_mut_ptr().add(base_out + m + l) as *mut f64;
                let y2_ptr = dst.as_mut_ptr().add(base_out + 2 * m + l) as *mut f64;
                let y3_ptr = dst.as_mut_ptr().add(base_out + 3 * m + l) as *mut f64;
                vst2q_f64(y0_ptr, float64x2x2_t(y0_re, y0_im));
                vst2q_f64(y1_ptr, float64x2x2_t(y1_re, y1_im));
                vst2q_f64(y2_ptr, float64x2x2_t(y2_re, y2_im));
                vst2q_f64(y3_ptr, float64x2x2_t(y3_re, y3_im));

                l += 2;
            }

            // Scalar tail — also runs the whole stage when m < 2 (the
            // SIMD `while l + 2 <= m` never enters). Skip the identity
            // multiply at l = 0; matters most for the m = 1 first stage.
            while l < m {
                let x0 = src[base_in + l];
                let (x1, x2, x3) = if l == 0 {
                    (
                        src[base_in + l + stride],
                        src[base_in + l + 2 * stride],
                        src[base_in + l + 3 * stride],
                    )
                } else {
                    (
                        src[base_in + l + stride] * tw1[l],
                        src[base_in + l + 2 * stride] * tw2[l],
                        src[base_in + l + 3 * stride] * tw3[l],
                    )
                };
                let a = x0 + x2;
                let b = x0 - x2;
                let c = x1 + x3;
                let d = x1 - x3;
                let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
                dst[base_out + l] = a + c;
                dst[base_out + m + l] = b + id;
                dst[base_out + 2 * m + l] = a - c;
                dst[base_out + 3 * m + l] = b - id;
                l += 1;
            }
        }
    }
}

/// NEON radix-4 final stage for `s = 1`.
///
/// This keeps the same lane-parallel butterfly as `stage_radix4_f64`, but
/// removes the outer group loop and fixed base-offset arithmetic from the
/// audio-rate final radix-4 stage (`stride == m`).
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix4_f64_s1(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    // SAFETY: NEON enforced by target_feature.
    unsafe {
        let m = stage_tw.m;
        debug_assert_eq!(stage_tw.s, 1);
        let dir_v = vdupq_n_f64(dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        let mut l = 0usize;
        while l + 2 <= m {
            let x0_ptr = src.as_ptr().add(l) as *const f64;
            let x1_ptr = src.as_ptr().add(l + m) as *const f64;
            let x2_ptr = src.as_ptr().add(l + 2 * m) as *const f64;
            let x3_ptr = src.as_ptr().add(l + 3 * m) as *const f64;
            let x0 = vld2q_f64(x0_ptr);
            let x1 = vld2q_f64(x1_ptr);
            let x2 = vld2q_f64(x2_ptr);
            let x3 = vld2q_f64(x3_ptr);

            let (tw1_re, tw1_im) = load_twiddles2(tw1.as_ptr().add(l));
            let (tw2_re, tw2_im) = load_twiddles2(tw2.as_ptr().add(l));
            let (tw3_re, tw3_im) = load_twiddles2(tw3.as_ptr().add(l));

            let (x1_re, x1_im) = cmul_simd(x1.0, x1.1, tw1_re, tw1_im);
            let (x2_re, x2_im) = cmul_simd(x2.0, x2.1, tw2_re, tw2_im);
            let (x3_re, x3_im) = cmul_simd(x3.0, x3.1, tw3_re, tw3_im);

            let a_re = vaddq_f64(x0.0, x2_re);
            let a_im = vaddq_f64(x0.1, x2_im);
            let b_re = vsubq_f64(x0.0, x2_re);
            let b_im = vsubq_f64(x0.1, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            let y0_ptr = dst.as_mut_ptr().add(l) as *mut f64;
            let y1_ptr = dst.as_mut_ptr().add(m + l) as *mut f64;
            let y2_ptr = dst.as_mut_ptr().add(2 * m + l) as *mut f64;
            let y3_ptr = dst.as_mut_ptr().add(3 * m + l) as *mut f64;
            vst2q_f64(y0_ptr, float64x2x2_t(y0_re, y0_im));
            vst2q_f64(y1_ptr, float64x2x2_t(y1_re, y1_im));
            vst2q_f64(y2_ptr, float64x2x2_t(y2_re, y2_im));
            vst2q_f64(y3_ptr, float64x2x2_t(y3_re, y3_im));

            l += 2;
        }

        while l < m {
            let x0 = src[l];
            let (x1, x2, x3) = if l == 0 {
                (src[l + m], src[l + 2 * m], src[l + 3 * m])
            } else {
                (
                    src[l + m] * tw1[l],
                    src[l + 2 * m] * tw2[l],
                    src[l + 3 * m] * tw3[l],
                )
            };
            let a = x0 + x2;
            let b = x0 - x2;
            let c = x1 + x3;
            let d = x1 - x3;
            let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
            dst[l] = a + c;
            dst[m + l] = b + id;
            dst[2 * m + l] = a - c;
            dst[3 * m + l] = b - id;
            l += 1;
        }
    }
}

/// NEON radix-4 stage for `Complex<f32>`. Same memory layout and semantics
/// as `stockham::stage_radix4::<f32>`, processing four consecutive
/// butterflies per loop iteration.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix4_f32(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let dir_v = vdupq_n_f32(dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            while l + 4 <= m {
                let x0_ptr = src.as_ptr().add(base_in + l) as *const f32;
                let x1_ptr = src.as_ptr().add(base_in + l + stride) as *const f32;
                let x2_ptr = src.as_ptr().add(base_in + l + 2 * stride) as *const f32;
                let x3_ptr = src.as_ptr().add(base_in + l + 3 * stride) as *const f32;
                let x0 = vld2q_f32(x0_ptr);
                let x1 = vld2q_f32(x1_ptr);
                let x2 = vld2q_f32(x2_ptr);
                let x3 = vld2q_f32(x3_ptr);

                let (tw1_re, tw1_im) = load_twiddles4_f32(tw1.as_ptr().add(l));
                let (tw2_re, tw2_im) = load_twiddles4_f32(tw2.as_ptr().add(l));
                let (tw3_re, tw3_im) = load_twiddles4_f32(tw3.as_ptr().add(l));

                let (x1_re, x1_im) = cmul_simd_f32(x1.0, x1.1, tw1_re, tw1_im);
                let (x2_re, x2_im) = cmul_simd_f32(x2.0, x2.1, tw2_re, tw2_im);
                let (x3_re, x3_im) = cmul_simd_f32(x3.0, x3.1, tw3_re, tw3_im);

                let a_re = vaddq_f32(x0.0, x2_re);
                let a_im = vaddq_f32(x0.1, x2_im);
                let b_re = vsubq_f32(x0.0, x2_re);
                let b_im = vsubq_f32(x0.1, x2_im);
                let c_re = vaddq_f32(x1_re, x3_re);
                let c_im = vaddq_f32(x1_im, x3_im);
                let d_re = vsubq_f32(x1_re, x3_re);
                let d_im = vsubq_f32(x1_im, x3_im);

                let id_re = vnegq_f32(vmulq_f32(d_im, dir_v));
                let id_im = vmulq_f32(d_re, dir_v);

                let y0_re = vaddq_f32(a_re, c_re);
                let y0_im = vaddq_f32(a_im, c_im);
                let y2_re = vsubq_f32(a_re, c_re);
                let y2_im = vsubq_f32(a_im, c_im);
                let y1_re = vaddq_f32(b_re, id_re);
                let y1_im = vaddq_f32(b_im, id_im);
                let y3_re = vsubq_f32(b_re, id_re);
                let y3_im = vsubq_f32(b_im, id_im);

                let y0_ptr = dst.as_mut_ptr().add(base_out + l) as *mut f32;
                let y1_ptr = dst.as_mut_ptr().add(base_out + m + l) as *mut f32;
                let y2_ptr = dst.as_mut_ptr().add(base_out + 2 * m + l) as *mut f32;
                let y3_ptr = dst.as_mut_ptr().add(base_out + 3 * m + l) as *mut f32;
                vst2q_f32(y0_ptr, float32x4x2_t(y0_re, y0_im));
                vst2q_f32(y1_ptr, float32x4x2_t(y1_re, y1_im));
                vst2q_f32(y2_ptr, float32x4x2_t(y2_re, y2_im));
                vst2q_f32(y3_ptr, float32x4x2_t(y3_re, y3_im));

                l += 4;
            }

            while l < m {
                let x0 = src[base_in + l];
                let (x1, x2, x3) = if l == 0 {
                    (
                        src[base_in + l + stride],
                        src[base_in + l + 2 * stride],
                        src[base_in + l + 3 * stride],
                    )
                } else {
                    (
                        src[base_in + l + stride] * tw1[l],
                        src[base_in + l + 2 * stride] * tw2[l],
                        src[base_in + l + 3 * stride] * tw3[l],
                    )
                };
                let a = x0 + x2;
                let b = x0 - x2;
                let c = x1 + x3;
                let d = x1 - x3;
                let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
                dst[base_out + l] = a + c;
                dst[base_out + m + l] = b + id;
                dst[base_out + 2 * m + l] = a - c;
                dst[base_out + 3 * m + l] = b - id;
                l += 1;
            }
        }
    }
}
// ============================================================================
// Batched vectorisation across the batch dimension (B = SIMD width = 2)
// ============================================================================
//
// For f64 on AArch64 the SIMD width is 2 lanes.  Instead of vectorising
// *within* one FFT (two consecutive butterflies), we process two independent
// FFTs in parallel: lane-0 = FFT-0, lane-1 = FFT-1.
//
// The data is transposed into an interleaved layout where the B values for
// the same logical index are contiguous:
//   tmp[2*i + 0] = fft0[i]
//   tmp[2*i + 1] = fft1[i]
//
// In this layout a single `vld2q_f64` loads one logical index from both FFTs
// directly into planar (re, im) vectors.

/// Transpose two contiguous FFT chunks into interleaved layout.
/// `dst.len() == 2 * src0.len()` must hold.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn transpose_pair_in(
    src0: &[Complex<f64>],
    src1: &[Complex<f64>],
    dst: &mut [Complex<f64>],
) {
    unsafe {
        let n = src0.len();
        debug_assert_eq!(src1.len(), n);
        debug_assert_eq!(dst.len(), 2 * n);
        let mut i = 0usize;
        while i + 2 <= n {
            let x0 = vld2q_f64(src0.as_ptr().add(i) as *const f64);
            let x1 = vld2q_f64(src1.as_ptr().add(i) as *const f64);
            let re0 = vtrn1q_f64(x0.0, x1.0);
            let re1 = vtrn2q_f64(x0.0, x1.0);
            let im0 = vtrn1q_f64(x0.1, x1.1);
            let im1 = vtrn2q_f64(x0.1, x1.1);
            vst2q_f64(
                dst.as_mut_ptr().add(2 * i) as *mut f64,
                float64x2x2_t(re0, im0),
            );
            vst2q_f64(
                dst.as_mut_ptr().add(2 * i + 2) as *mut f64,
                float64x2x2_t(re1, im1),
            );
            i += 2;
        }
        while i < n {
            dst[2 * i] = src0[i];
            dst[2 * i + 1] = src1[i];
            i += 1;
        }
    }
}

/// Undo `transpose_pair_in`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn transpose_pair_out(
    src: &[Complex<f64>],
    dst0: &mut [Complex<f64>],
    dst1: &mut [Complex<f64>],
) {
    unsafe {
        let n = dst0.len();
        debug_assert_eq!(dst1.len(), n);
        debug_assert_eq!(src.len(), 2 * n);
        let mut i = 0usize;
        while i + 2 <= n {
            let x0 = vld2q_f64(src.as_ptr().add(2 * i) as *const f64);
            let x1 = vld2q_f64(src.as_ptr().add(2 * i + 2) as *const f64);
            let re0 = vtrn1q_f64(x0.0, x1.0);
            let re1 = vtrn2q_f64(x0.0, x1.0);
            let im0 = vtrn1q_f64(x0.1, x1.1);
            let im1 = vtrn2q_f64(x0.1, x1.1);
            vst2q_f64(
                dst0.as_mut_ptr().add(i) as *mut f64,
                float64x2x2_t(re0, im0),
            );
            vst2q_f64(
                dst1.as_mut_ptr().add(i) as *mut f64,
                float64x2x2_t(re1, im1),
            );
            i += 2;
        }
        while i < n {
            dst0[i] = src[2 * i];
            dst1[i] = src[2 * i + 1];
            i += 1;
        }
    }
}

/// Batched radix-2 stage for two FFTs in interleaved layout.
/// `stride`, `src` and `dst` are in *logical* element units; the physical
/// offset for logical index `l` is `l * 2`.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix2_f64_batched2(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
) {
    unsafe {
        const B: usize = 2;
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);
        for g in 0..s {
            let base_in = g * m * B;
            let base_out = g * m_new * B;
            let mut l = 0usize;
            while l < m {
                let a = vld2q_f64(src.as_ptr().add(base_in + l * B) as *const f64);
                let b = vld2q_f64(src.as_ptr().add(base_in + (l + stride) * B) as *const f64);
                let tw = tw1[l];
                let tw_re = vdupq_n_f64(tw.re);
                let tw_im = vdupq_n_f64(tw.im);
                let (y0_re, y0_im, y1_re, y1_im) = radix2_simd(a.0, a.1, b.0, b.1, tw_re, tw_im);
                vst2q_f64(
                    dst.as_mut_ptr().add(base_out + l * B) as *mut f64,
                    float64x2x2_t(y0_re, y0_im),
                );
                vst2q_f64(
                    dst.as_mut_ptr().add(base_out + (m + l) * B) as *mut f64,
                    float64x2x2_t(y1_re, y1_im),
                );
                l += 1;
            }
        }
    }
}

/// Batched radix-4 stage for two FFTs in interleaved layout.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix4_f64_batched2(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        const B: usize = 2;
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let dir_v = vdupq_n_f64(dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        for g in 0..s {
            let base_in = g * m * B;
            let base_out = g * m_new * B;
            let mut l = 0usize;
            while l < m {
                let x0 = vld2q_f64(src.as_ptr().add(base_in + l * B) as *const f64);
                let x1 = vld2q_f64(src.as_ptr().add(base_in + (l + stride) * B) as *const f64);
                let x2 = vld2q_f64(src.as_ptr().add(base_in + (l + 2 * stride) * B) as *const f64);
                let x3 = vld2q_f64(src.as_ptr().add(base_in + (l + 3 * stride) * B) as *const f64);

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                    (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
                } else {
                    let t1 = tw1[l];
                    let t2 = tw2[l];
                    let t3 = tw3[l];
                    let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0.0, x2_re);
                let a_im = vaddq_f64(x0.1, x2_im);
                let b_re = vsubq_f64(x0.0, x2_re);
                let b_im = vsubq_f64(x0.1, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                vst2q_f64(
                    dst.as_mut_ptr().add(base_out + l * B) as *mut f64,
                    float64x2x2_t(y0_re, y0_im),
                );
                vst2q_f64(
                    dst.as_mut_ptr().add(base_out + (m + l) * B) as *mut f64,
                    float64x2x2_t(y1_re, y1_im),
                );
                vst2q_f64(
                    dst.as_mut_ptr().add(base_out + (2 * m + l) * B) as *mut f64,
                    float64x2x2_t(y2_re, y2_im),
                );
                vst2q_f64(
                    dst.as_mut_ptr().add(base_out + (3 * m + l) * B) as *mut f64,
                    float64x2x2_t(y3_re, y3_im),
                );

                l += 1;
            }
        }
    }
}

#[inline]
unsafe fn load_complex2_f32_batched(ptr: *const Complex<f32>) -> (float32x4_t, float32x4_t) {
    unsafe {
        let values = vld1q_f32(ptr as *const f32);
        // Physical layout is [re0, im0, re1, im1]. Duplicate the two
        // useful lanes so the same arithmetic handles both transforms.
        (vuzp1q_f32(values, values), vuzp2q_f32(values, values))
    }
}

#[inline]
unsafe fn store_complex2_f32_batched(ptr: *mut Complex<f32>, re: float32x4_t, im: float32x4_t) {
    unsafe {
        vst1q_f32(ptr as *mut f32, vzip1q_f32(re, im));
    }
}

/// Batched radix-2 stage for two f32 FFTs in interleaved layout.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix2_f32_batched2(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for (l, tw) in tw1.iter().copied().enumerate().take(m) {
                let (a_re, a_im) = load_complex2_f32_batched(src.as_ptr().add(2 * (base_in + l)));
                let (b_re, b_im) =
                    load_complex2_f32_batched(src.as_ptr().add(2 * (base_in + l + stride)));
                let (b_re, b_im) =
                    cmul_simd_f32(b_re, b_im, vdupq_n_f32(tw.re), vdupq_n_f32(tw.im));
                store_complex2_f32_batched(
                    dst.as_mut_ptr().add(2 * (base_out + l)),
                    vaddq_f32(a_re, b_re),
                    vaddq_f32(a_im, b_im),
                );
                store_complex2_f32_batched(
                    dst.as_mut_ptr().add(2 * (base_out + m + l)),
                    vsubq_f32(a_re, b_re),
                    vsubq_f32(a_im, b_im),
                );
            }
        }
    }
}

/// Batched radix-4 stage for two f32 FFTs in interleaved layout.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix4_f32_batched2(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let dir_v = vdupq_n_f32(dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for (l, ((t1, t2), t3)) in tw1
                .iter()
                .copied()
                .zip(tw2.iter().copied())
                .zip(tw3.iter().copied())
                .enumerate()
                .take(m)
            {
                let (x0_re, x0_im) = load_complex2_f32_batched(src.as_ptr().add(2 * (base_in + l)));
                let (mut x1_re, mut x1_im) =
                    load_complex2_f32_batched(src.as_ptr().add(2 * (base_in + l + stride)));
                let (mut x2_re, mut x2_im) =
                    load_complex2_f32_batched(src.as_ptr().add(2 * (base_in + l + 2 * stride)));
                let (mut x3_re, mut x3_im) =
                    load_complex2_f32_batched(src.as_ptr().add(2 * (base_in + l + 3 * stride)));

                (x1_re, x1_im) =
                    cmul_simd_f32(x1_re, x1_im, vdupq_n_f32(t1.re), vdupq_n_f32(t1.im));
                (x2_re, x2_im) =
                    cmul_simd_f32(x2_re, x2_im, vdupq_n_f32(t2.re), vdupq_n_f32(t2.im));
                (x3_re, x3_im) =
                    cmul_simd_f32(x3_re, x3_im, vdupq_n_f32(t3.re), vdupq_n_f32(t3.im));

                let a_re = vaddq_f32(x0_re, x2_re);
                let a_im = vaddq_f32(x0_im, x2_im);
                let b_re = vsubq_f32(x0_re, x2_re);
                let b_im = vsubq_f32(x0_im, x2_im);
                let c_re = vaddq_f32(x1_re, x3_re);
                let c_im = vaddq_f32(x1_im, x3_im);
                let d_re = vsubq_f32(x1_re, x3_re);
                let d_im = vsubq_f32(x1_im, x3_im);
                let id_re = vnegq_f32(vmulq_f32(d_im, dir_v));
                let id_im = vmulq_f32(d_re, dir_v);

                store_complex2_f32_batched(
                    dst.as_mut_ptr().add(2 * (base_out + l)),
                    vaddq_f32(a_re, c_re),
                    vaddq_f32(a_im, c_im),
                );
                store_complex2_f32_batched(
                    dst.as_mut_ptr().add(2 * (base_out + m + l)),
                    vaddq_f32(b_re, id_re),
                    vaddq_f32(b_im, id_im),
                );
                store_complex2_f32_batched(
                    dst.as_mut_ptr().add(2 * (base_out + 2 * m + l)),
                    vsubq_f32(a_re, c_re),
                    vsubq_f32(a_im, c_im),
                );
                store_complex2_f32_batched(
                    dst.as_mut_ptr().add(2 * (base_out + 3 * m + l)),
                    vsubq_f32(b_re, id_re),
                    vsubq_f32(b_im, id_im),
                );
            }
        }
    }
}

/// Batched radix-4 final stage (`s == 1`) for two FFTs in interleaved layout.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix4_f64_s1_batched2(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        const B: usize = 2;
        let m = stage_tw.m;
        debug_assert_eq!(stage_tw.s, 1);
        let dir_v = vdupq_n_f64(dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        let mut l = 0usize;
        while l < m {
            let x0 = vld2q_f64(src.as_ptr().add(l * B) as *const f64);
            let x1 = vld2q_f64(src.as_ptr().add((l + m) * B) as *const f64);
            let x2 = vld2q_f64(src.as_ptr().add((l + 2 * m) * B) as *const f64);
            let x3 = vld2q_f64(src.as_ptr().add((l + 3 * m) * B) as *const f64);

            let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
            } else {
                let t1 = tw1[l];
                let t2 = tw2[l];
                let t3 = tw3[l];
                let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                (r1, i1, r2, i2, r3, i3)
            };

            let a_re = vaddq_f64(x0.0, x2_re);
            let a_im = vaddq_f64(x0.1, x2_im);
            let b_re = vsubq_f64(x0.0, x2_re);
            let b_im = vsubq_f64(x0.1, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            vst2q_f64(
                dst.as_mut_ptr().add(l * B) as *mut f64,
                float64x2x2_t(y0_re, y0_im),
            );
            vst2q_f64(
                dst.as_mut_ptr().add((m + l) * B) as *mut f64,
                float64x2x2_t(y1_re, y1_im),
            );
            vst2q_f64(
                dst.as_mut_ptr().add((2 * m + l) * B) as *mut f64,
                float64x2x2_t(y2_re, y2_im),
            );
            vst2q_f64(
                dst.as_mut_ptr().add((3 * m + l) * B) as *mut f64,
                float64x2x2_t(y3_re, y3_im),
            );

            l += 1;
        }
    }
}

// ============================================================================
// Planar (split-complex) batched pair path — f64 only
// ============================================================================
//
// The interleaved pair layout above stores [fft0[i], fft1[i]] as consecutive
// Complex<f64> elements.  Every load/store uses vld2q_f64 / vst2q_f64 to
// deinterleave / interleave, which costs ~2-3 cycles each.
//
// In planar layout the same buffer is reinterpreted as:
//   [all real parts (2*n f64s), all imag parts (2*n f64s)]
// so a logical position p maps to:
//   real_addr = base + p * B   (B = 2 SIMD lanes)
//   imag_addr = base + 2*n + p * B
// This lets us use vld1q_f64 / vst1q_f64 (~1 cycle) for every access.
// The transpose-in / transpose-out cost is amortised across all stages.

/// Batched radix-4 stage for two FFTs in planar layout.
#[target_feature(enable = "neon")]
pub unsafe fn stage_radix4_f64_batched2_planar(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        const B: usize = 2;
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let dir_v = vdupq_n_f64(dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        let n_complex = src.len();
        let src_real = src.as_ptr() as *const f64;
        let src_imag = src_real.add(n_complex);
        let dst_real = dst.as_mut_ptr() as *mut f64;
        let dst_imag = dst_real.add(n_complex);

        for g in 0..s {
            let base_in = g * m * B;
            let base_out = g * m_new * B;
            let mut l = 0usize;
            while l < m {
                let x0_re = vld1q_f64(src_real.add(base_in + l * B));
                let x0_im = vld1q_f64(src_imag.add(base_in + l * B));
                let x1_re = vld1q_f64(src_real.add(base_in + (l + stride) * B));
                let x1_im = vld1q_f64(src_imag.add(base_in + (l + stride) * B));
                let x2_re = vld1q_f64(src_real.add(base_in + (l + 2 * stride) * B));
                let x2_im = vld1q_f64(src_imag.add(base_in + (l + 2 * stride) * B));
                let x3_re = vld1q_f64(src_real.add(base_in + (l + 3 * stride) * B));
                let x3_im = vld1q_f64(src_imag.add(base_in + (l + 3 * stride) * B));

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                    (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im)
                } else {
                    let t1 = tw1[l];
                    let t2 = tw2[l];
                    let t3 = tw3[l];
                    let (r1, i1) = cmul_simd(x1_re, x1_im, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2_re, x2_im, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3_re, x3_im, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0_re, x2_re);
                let a_im = vaddq_f64(x0_im, x2_im);
                let b_re = vsubq_f64(x0_re, x2_re);
                let b_im = vsubq_f64(x0_im, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                vst1q_f64(dst_real.add(base_out + l * B), y0_re);
                vst1q_f64(dst_imag.add(base_out + l * B), y0_im);
                vst1q_f64(dst_real.add(base_out + (m + l) * B), y1_re);
                vst1q_f64(dst_imag.add(base_out + (m + l) * B), y1_im);
                vst1q_f64(dst_real.add(base_out + (2 * m + l) * B), y2_re);
                vst1q_f64(dst_imag.add(base_out + (2 * m + l) * B), y2_im);
                vst1q_f64(dst_real.add(base_out + (3 * m + l) * B), y3_re);
                vst1q_f64(dst_imag.add(base_out + (3 * m + l) * B), y3_im);

                l += 1;
            }
        }
    }
}

/// Fused final two radix-4 stages for one 4096-point transform.
///
/// The operation is safe in place because both adjacent `l` values are fully
/// loaded before any of their output locations are overwritten. Processing
/// two adjacent values per iteration keeps the complete final third of the
/// FFT in NEON instead of falling back to scalar complex arithmetic.
#[inline(always)]
pub unsafe fn fused_final_radix4x4_4096_in_place(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        if dir_sign < 0.0 {
            fused_final_radix4x4_4096_in_place_impl::<true>(buffer, radix4_stage, final_stage);
        } else {
            fused_final_radix4x4_4096_in_place_impl::<false>(buffer, radix4_stage, final_stage);
        }
    }
}

#[inline(always)]
unsafe fn fused_final_radix4x4_4096_in_place_impl<const FORWARD: bool>(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
) {
    unsafe {
        debug_assert_eq!(buffer.len(), 4096);
        debug_assert_eq!(
            (radix4_stage.radix, radix4_stage.m, radix4_stage.s),
            (4, 256, 4)
        );
        debug_assert_eq!(
            (final_stage.radix, final_stage.m, final_stage.s),
            (4, 1024, 1)
        );

        let ptr = buffer.as_mut_ptr();

        let mut l = 0usize;
        while l < 256 {
            let [r4_t1, r4_t2, r4_t3] =
                load_packed_radix4_twiddles(radix4_stage.radix4_pair_ptr(l));

            let mut stage4 = [
                (
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                ),
                (
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                ),
                (
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                ),
                (
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                    (vdupq_n_f64(0.0), vdupq_n_f64(0.0)),
                ),
            ];

            for (g, outputs) in stage4.iter_mut().enumerate() {
                let base = g * 256 + l;
                let x0 = vld2q_f64(ptr.add(base) as *const f64);
                let x1 = vld2q_f64(ptr.add(base + 1024) as *const f64);
                let x2 = vld2q_f64(ptr.add(base + 2048) as *const f64);
                let x3 = vld2q_f64(ptr.add(base + 3072) as *const f64);
                *outputs = radix4_twiddled_simd::<FORWARD>(
                    (x0.0, x0.1),
                    (x1.0, x1.1),
                    (x2.0, x2.1),
                    (x3.0, x3.1),
                    r4_t1,
                    r4_t2,
                    r4_t3,
                );
            }

            for q in 0..4 {
                let tw_offset = q * 256 + l;
                let [final_t1, final_t2, final_t3] =
                    load_packed_radix4_twiddles(final_stage.radix4_pair_ptr(tw_offset));
                let outputs = radix4_twiddled_simd::<FORWARD>(
                    stage4[0].0,
                    stage4[1].0,
                    stage4[2].0,
                    stage4[3].0,
                    final_t1,
                    final_t2,
                    final_t3,
                );
                let base = tw_offset;
                vst2q_f64(
                    ptr.add(base) as *mut f64,
                    float64x2x2_t(outputs.0.0, outputs.0.1),
                );
                vst2q_f64(
                    ptr.add(base + 1024) as *mut f64,
                    float64x2x2_t(outputs.1.0, outputs.1.1),
                );
                vst2q_f64(
                    ptr.add(base + 2048) as *mut f64,
                    float64x2x2_t(outputs.2.0, outputs.2.1),
                );
                vst2q_f64(
                    ptr.add(base + 3072) as *mut f64,
                    float64x2x2_t(outputs.3.0, outputs.3.1),
                );

                // Select the corresponding output of each stage-4 butterfly
                // for the next final-stage group.
                if q != 3 {
                    stage4[0].0 = match q {
                        0 => stage4[0].1,
                        1 => stage4[0].2,
                        _ => stage4[0].3,
                    };
                    stage4[1].0 = match q {
                        0 => stage4[1].1,
                        1 => stage4[1].2,
                        _ => stage4[1].3,
                    };
                    stage4[2].0 = match q {
                        0 => stage4[2].1,
                        1 => stage4[2].2,
                        _ => stage4[2].3,
                    };
                    stage4[3].0 = match q {
                        0 => stage4[3].1,
                        1 => stage4[3].2,
                        _ => stage4[3].3,
                    };
                }
            }

            l += 2;
        }
    }
}

/// Fused first stage (radix-4, m=1) with transpose-in for a pair of FFTs.
/// Reads from original chunk layout, writes planar layout.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn fused_first_stage_radix4_m1_transpose_in_pair_planar(
    src0: &[Complex<f64>],
    src1: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    dir_sign: f64,
) {
    unsafe {
        let n = src0.len();
        debug_assert_eq!(src1.len(), n);
        debug_assert_eq!(dst.len(), 2 * n);
        let s = n / 4;
        let dir_v = vdupq_n_f64(dir_sign);

        let dst_real = dst.as_mut_ptr() as *mut f64;
        let dst_imag = dst_real.add(2 * n);

        for g in 0..s {
            let x0_0 = vld1q_f64(src0.as_ptr().add(g) as *const f64);
            let x1_0 = vld1q_f64(src0.as_ptr().add(g + s) as *const f64);
            let x2_0 = vld1q_f64(src0.as_ptr().add(g + 2 * s) as *const f64);
            let x3_0 = vld1q_f64(src0.as_ptr().add(g + 3 * s) as *const f64);

            let x0_1 = vld1q_f64(src1.as_ptr().add(g) as *const f64);
            let x1_1 = vld1q_f64(src1.as_ptr().add(g + s) as *const f64);
            let x2_1 = vld1q_f64(src1.as_ptr().add(g + 2 * s) as *const f64);
            let x3_1 = vld1q_f64(src1.as_ptr().add(g + 3 * s) as *const f64);

            let x0_re = vtrn1q_f64(x0_0, x0_1);
            let x0_im = vtrn2q_f64(x0_0, x0_1);
            let x1_re = vtrn1q_f64(x1_0, x1_1);
            let x1_im = vtrn2q_f64(x1_0, x1_1);
            let x2_re = vtrn1q_f64(x2_0, x2_1);
            let x2_im = vtrn2q_f64(x2_0, x2_1);
            let x3_re = vtrn1q_f64(x3_0, x3_1);
            let x3_im = vtrn2q_f64(x3_0, x3_1);

            let a_re = vaddq_f64(x0_re, x2_re);
            let a_im = vaddq_f64(x0_im, x2_im);
            let b_re = vsubq_f64(x0_re, x2_re);
            let b_im = vsubq_f64(x0_im, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            let base = 8 * g;
            vst1q_f64(dst_real.add(base), y0_re);
            vst1q_f64(dst_imag.add(base), y0_im);
            vst1q_f64(dst_real.add(base + 2), y1_re);
            vst1q_f64(dst_imag.add(base + 2), y1_im);
            vst1q_f64(dst_real.add(base + 4), y2_re);
            vst1q_f64(dst_imag.add(base + 4), y2_im);
            vst1q_f64(dst_real.add(base + 6), y3_re);
            vst1q_f64(dst_imag.add(base + 6), y3_im);
        }
    }
}

/// Fused final two radix-4 stages (m=16,s=4 then m=64,s=1) for n=256,
/// with transpose-out.  Reads planar pair, writes original chunks.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn fused_final_radix4x4_256_transpose_out_pair_planar(
    src: &[Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
    dst0: &mut [Complex<f64>],
    dst1: &mut [Complex<f64>],
) {
    unsafe {
        debug_assert_eq!(radix4_stage.radix, 4);
        debug_assert_eq!(radix4_stage.m, 16);
        debug_assert_eq!(radix4_stage.s, 4);
        debug_assert_eq!(final_stage.radix, 4);
        debug_assert_eq!(final_stage.m, 64);
        debug_assert_eq!(final_stage.s, 1);
        debug_assert_eq!(src.len(), 512);
        debug_assert_eq!(dst0.len(), 256);
        debug_assert_eq!(dst1.len(), 256);

        let tw1_r4 = radix4_stage.k_slice(1);
        let tw2_r4 = radix4_stage.k_slice(2);
        let tw3_r4 = radix4_stage.k_slice(3);

        let tw1_f = final_stage.k_slice(1);
        let tw2_f = final_stage.k_slice(2);
        let tw3_f = final_stage.k_slice(3);

        let dir_v = vdupq_n_f64(dir_sign);

        let mut pre = [Complex::new(0.0, 0.0); 512];
        let pre_real = pre.as_mut_ptr() as *mut f64;
        let pre_imag = pre_real.add(512);

        let src_real = src.as_ptr() as *const f64;
        let src_imag = src_real.add(512);

        for g in 0..4usize {
            for inner_l in 0..16usize {
                let x0_re = vld1q_f64(src_real.add((g * 16 + inner_l) * 2));
                let x0_im = vld1q_f64(src_imag.add((g * 16 + inner_l) * 2));
                let x1_re = vld1q_f64(src_real.add((g * 16 + inner_l + 64) * 2));
                let x1_im = vld1q_f64(src_imag.add((g * 16 + inner_l + 64) * 2));
                let x2_re = vld1q_f64(src_real.add((g * 16 + inner_l + 128) * 2));
                let x2_im = vld1q_f64(src_imag.add((g * 16 + inner_l + 128) * 2));
                let x3_re = vld1q_f64(src_real.add((g * 16 + inner_l + 192) * 2));
                let x3_im = vld1q_f64(src_imag.add((g * 16 + inner_l + 192) * 2));

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if inner_l == 0 {
                    (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im)
                } else {
                    let t1 = tw1_r4[inner_l];
                    let t2 = tw2_r4[inner_l];
                    let t3 = tw3_r4[inner_l];
                    let (r1, i1) = cmul_simd(x1_re, x1_im, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2_re, x2_im, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3_re, x3_im, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0_re, x2_re);
                let a_im = vaddq_f64(x0_im, x2_im);
                let b_re = vsubq_f64(x0_re, x2_re);
                let b_im = vsubq_f64(x0_im, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                let base_out = (g * 64 + inner_l) * 2;
                vst1q_f64(pre_real.add(base_out), y0_re);
                vst1q_f64(pre_imag.add(base_out), y0_im);
                vst1q_f64(pre_real.add(base_out + 32), y1_re);
                vst1q_f64(pre_imag.add(base_out + 32), y1_im);
                vst1q_f64(pre_real.add(base_out + 64), y2_re);
                vst1q_f64(pre_imag.add(base_out + 64), y2_im);
                vst1q_f64(pre_real.add(base_out + 96), y3_re);
                vst1q_f64(pre_imag.add(base_out + 96), y3_im);
            }
        }

        for l in 0..64usize {
            let x0_re = vld1q_f64(pre_real.add(l * 2));
            let x0_im = vld1q_f64(pre_imag.add(l * 2));
            let x1_re = vld1q_f64(pre_real.add((l + 64) * 2));
            let x1_im = vld1q_f64(pre_imag.add((l + 64) * 2));
            let x2_re = vld1q_f64(pre_real.add((l + 128) * 2));
            let x2_im = vld1q_f64(pre_imag.add((l + 128) * 2));
            let x3_re = vld1q_f64(pre_real.add((l + 192) * 2));
            let x3_im = vld1q_f64(pre_imag.add((l + 192) * 2));

            let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im)
            } else {
                let t1 = tw1_f[l];
                let t2 = tw2_f[l];
                let t3 = tw3_f[l];
                let (r1, i1) = cmul_simd(x1_re, x1_im, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                let (r2, i2) = cmul_simd(x2_re, x2_im, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                let (r3, i3) = cmul_simd(x3_re, x3_im, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                (r1, i1, r2, i2, r3, i3)
            };

            let a_re = vaddq_f64(x0_re, x2_re);
            let a_im = vaddq_f64(x0_im, x2_im);
            let b_re = vsubq_f64(x0_re, x2_re);
            let b_im = vsubq_f64(x0_im, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            dst0[l] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
            dst1[l] = Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
            dst0[64 + l] = Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
            dst1[64 + l] = Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
            dst0[128 + l] = Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
            dst1[128 + l] = Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
            dst0[192 + l] = Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
            dst1[192 + l] = Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
        }
    }
}

/// Fused final two radix-4 stages (m=64,s=4 then m=256,s=1) for n=1024,
/// with transpose-out.  Reads planar pair, writes original chunks.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn fused_final_radix4x4_1024_transpose_out_pair_planar(
    src: &[Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
    dst0: &mut [Complex<f64>],
    dst1: &mut [Complex<f64>],
) {
    unsafe {
        debug_assert_eq!(radix4_stage.radix, 4);
        debug_assert_eq!(radix4_stage.m, 64);
        debug_assert_eq!(radix4_stage.s, 4);
        debug_assert_eq!(final_stage.radix, 4);
        debug_assert_eq!(final_stage.m, 256);
        debug_assert_eq!(final_stage.s, 1);
        debug_assert_eq!(src.len(), 2048);
        debug_assert_eq!(dst0.len(), 1024);
        debug_assert_eq!(dst1.len(), 1024);

        let tw1_r4 = radix4_stage.k_slice(1);
        let tw2_r4 = radix4_stage.k_slice(2);
        let tw3_r4 = radix4_stage.k_slice(3);

        let tw1_f = final_stage.k_slice(1);
        let tw2_f = final_stage.k_slice(2);
        let tw3_f = final_stage.k_slice(3);

        let dir_v = vdupq_n_f64(dir_sign);

        let mut pre = [Complex::new(0.0, 0.0); 2048];
        let pre_real = pre.as_mut_ptr() as *mut f64;
        let pre_imag = pre_real.add(2048);

        let src_real = src.as_ptr() as *const f64;
        let src_imag = src_real.add(2048);

        for g in 0..4usize {
            for inner_l in 0..64usize {
                let x0_re = vld1q_f64(src_real.add((g * 64 + inner_l) * 2));
                let x0_im = vld1q_f64(src_imag.add((g * 64 + inner_l) * 2));
                let x1_re = vld1q_f64(src_real.add((g * 64 + inner_l + 256) * 2));
                let x1_im = vld1q_f64(src_imag.add((g * 64 + inner_l + 256) * 2));
                let x2_re = vld1q_f64(src_real.add((g * 64 + inner_l + 512) * 2));
                let x2_im = vld1q_f64(src_imag.add((g * 64 + inner_l + 512) * 2));
                let x3_re = vld1q_f64(src_real.add((g * 64 + inner_l + 768) * 2));
                let x3_im = vld1q_f64(src_imag.add((g * 64 + inner_l + 768) * 2));

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if inner_l == 0 {
                    (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im)
                } else {
                    let t1 = tw1_r4[inner_l];
                    let t2 = tw2_r4[inner_l];
                    let t3 = tw3_r4[inner_l];
                    let (r1, i1) = cmul_simd(x1_re, x1_im, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2_re, x2_im, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3_re, x3_im, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0_re, x2_re);
                let a_im = vaddq_f64(x0_im, x2_im);
                let b_re = vsubq_f64(x0_re, x2_re);
                let b_im = vsubq_f64(x0_im, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                let base_out = (g * 256 + inner_l) * 2;
                vst1q_f64(pre_real.add(base_out), y0_re);
                vst1q_f64(pre_imag.add(base_out), y0_im);
                vst1q_f64(pre_real.add(base_out + 128), y1_re);
                vst1q_f64(pre_imag.add(base_out + 128), y1_im);
                vst1q_f64(pre_real.add(base_out + 256), y2_re);
                vst1q_f64(pre_imag.add(base_out + 256), y2_im);
                vst1q_f64(pre_real.add(base_out + 384), y3_re);
                vst1q_f64(pre_imag.add(base_out + 384), y3_im);
            }
        }

        for l in 0..256usize {
            let x0_re = vld1q_f64(pre_real.add(l * 2));
            let x0_im = vld1q_f64(pre_imag.add(l * 2));
            let x1_re = vld1q_f64(pre_real.add((l + 256) * 2));
            let x1_im = vld1q_f64(pre_imag.add((l + 256) * 2));
            let x2_re = vld1q_f64(pre_real.add((l + 512) * 2));
            let x2_im = vld1q_f64(pre_imag.add((l + 512) * 2));
            let x3_re = vld1q_f64(pre_real.add((l + 768) * 2));
            let x3_im = vld1q_f64(pre_imag.add((l + 768) * 2));

            let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im)
            } else {
                let t1 = tw1_f[l];
                let t2 = tw2_f[l];
                let t3 = tw3_f[l];
                let (r1, i1) = cmul_simd(x1_re, x1_im, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                let (r2, i2) = cmul_simd(x2_re, x2_im, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                let (r3, i3) = cmul_simd(x3_re, x3_im, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                (r1, i1, r2, i2, r3, i3)
            };

            let a_re = vaddq_f64(x0_re, x2_re);
            let a_im = vaddq_f64(x0_im, x2_im);
            let b_re = vsubq_f64(x0_re, x2_re);
            let b_im = vsubq_f64(x0_im, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            dst0[l] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
            dst1[l] = Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
            dst0[256 + l] = Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
            dst1[256 + l] = Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
            dst0[512 + l] = Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
            dst1[512 + l] = Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
            dst0[768 + l] = Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
            dst1[768 + l] = Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
        }
    }
}

/// Fused final two radix-4 stages (m=256,s=4 then m=1024,s=1) for n=4096,
/// with transpose-out.  Reads planar pair, writes original chunks.
#[inline(never)]
#[cold]
#[target_feature(enable = "neon")]
pub unsafe fn fused_final_radix4x4_4096_transpose_out_pair_planar(
    src: &[Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
    dst0: &mut [Complex<f64>],
    dst1: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    unsafe {
        debug_assert_eq!(radix4_stage.radix, 4);
        debug_assert_eq!(radix4_stage.m, 256);
        debug_assert_eq!(radix4_stage.s, 4);
        debug_assert_eq!(final_stage.radix, 4);
        debug_assert_eq!(final_stage.m, 1024);
        debug_assert_eq!(final_stage.s, 1);
        debug_assert_eq!(src.len(), 8192);
        debug_assert_eq!(dst0.len(), 4096);
        debug_assert_eq!(dst1.len(), 4096);

        let tw1_r4 = radix4_stage.k_slice(1);
        let tw2_r4 = radix4_stage.k_slice(2);
        let tw3_r4 = radix4_stage.k_slice(3);

        let tw1_f = final_stage.k_slice(1);
        let tw2_f = final_stage.k_slice(2);
        let tw3_f = final_stage.k_slice(3);

        let dir_v = vdupq_n_f64(dir_sign);
        debug_assert!(scratch.len() >= 8192);
        let pre = &mut scratch[..8192];
        let pre_real = pre.as_mut_ptr() as *mut f64;
        let pre_imag = pre_real.add(8192);

        let src_real = src.as_ptr() as *const f64;
        let src_imag = src_real.add(8192);

        for g in 0..4usize {
            for inner_l in 0..256usize {
                let x0_re = vld1q_f64(src_real.add((g * 256 + inner_l) * 2));
                let x0_im = vld1q_f64(src_imag.add((g * 256 + inner_l) * 2));
                let x1_re = vld1q_f64(src_real.add((g * 256 + inner_l + 1024) * 2));
                let x1_im = vld1q_f64(src_imag.add((g * 256 + inner_l + 1024) * 2));
                let x2_re = vld1q_f64(src_real.add((g * 256 + inner_l + 2048) * 2));
                let x2_im = vld1q_f64(src_imag.add((g * 256 + inner_l + 2048) * 2));
                let x3_re = vld1q_f64(src_real.add((g * 256 + inner_l + 3072) * 2));
                let x3_im = vld1q_f64(src_imag.add((g * 256 + inner_l + 3072) * 2));

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if inner_l == 0 {
                    (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im)
                } else {
                    let t1 = tw1_r4[inner_l];
                    let t2 = tw2_r4[inner_l];
                    let t3 = tw3_r4[inner_l];
                    let (r1, i1) = cmul_simd(x1_re, x1_im, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2_re, x2_im, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3_re, x3_im, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0_re, x2_re);
                let a_im = vaddq_f64(x0_im, x2_im);
                let b_re = vsubq_f64(x0_re, x2_re);
                let b_im = vsubq_f64(x0_im, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                let base_out = (g * 1024 + inner_l) * 2;
                vst1q_f64(pre_real.add(base_out), y0_re);
                vst1q_f64(pre_imag.add(base_out), y0_im);
                vst1q_f64(pre_real.add(base_out + 512), y1_re);
                vst1q_f64(pre_imag.add(base_out + 512), y1_im);
                vst1q_f64(pre_real.add(base_out + 1024), y2_re);
                vst1q_f64(pre_imag.add(base_out + 1024), y2_im);
                vst1q_f64(pre_real.add(base_out + 1536), y3_re);
                vst1q_f64(pre_imag.add(base_out + 1536), y3_im);
            }
        }

        for l in 0..1024usize {
            let x0_re = vld1q_f64(pre_real.add(l * 2));
            let x0_im = vld1q_f64(pre_imag.add(l * 2));
            let x1_re = vld1q_f64(pre_real.add((l + 1024) * 2));
            let x1_im = vld1q_f64(pre_imag.add((l + 1024) * 2));
            let x2_re = vld1q_f64(pre_real.add((l + 2048) * 2));
            let x2_im = vld1q_f64(pre_imag.add((l + 2048) * 2));
            let x3_re = vld1q_f64(pre_real.add((l + 3072) * 2));
            let x3_im = vld1q_f64(pre_imag.add((l + 3072) * 2));

            let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im)
            } else {
                let t1 = tw1_f[l];
                let t2 = tw2_f[l];
                let t3 = tw3_f[l];
                let (r1, i1) = cmul_simd(x1_re, x1_im, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                let (r2, i2) = cmul_simd(x2_re, x2_im, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                let (r3, i3) = cmul_simd(x3_re, x3_im, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                (r1, i1, r2, i2, r3, i3)
            };

            let a_re = vaddq_f64(x0_re, x2_re);
            let a_im = vaddq_f64(x0_im, x2_im);
            let b_re = vsubq_f64(x0_re, x2_re);
            let b_im = vsubq_f64(x0_im, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            dst0[l] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
            dst1[l] = Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
            dst0[1024 + l] = Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
            dst1[1024 + l] = Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
            dst0[2048 + l] = Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
            dst1[2048 + l] = Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
            dst0[3072 + l] = Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
            dst1[3072 + l] = Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
        }
    }
}

/// Fused first stage (radix-4, m=1) with transpose-in for a pair of FFTs.
/// Reads from original chunk layout, writes interleaved (transposed) layout.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn fused_first_stage_radix4_m1_transpose_in_pair(
    src0: &[Complex<f64>],
    src1: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    dir_sign: f64,
) {
    unsafe {
        let n = src0.len();
        debug_assert_eq!(src1.len(), n);
        debug_assert_eq!(dst.len(), 2 * n);
        let s = n / 4;
        let dir_v = vdupq_n_f64(dir_sign);

        for g in 0..s {
            // Load one Complex64 from each FFT at the four input positions.
            // vld1q_f64 on a *const f64 loads 2 consecutive f64s = 1 Complex64.
            let x0_0 = vld1q_f64(src0.as_ptr().add(g) as *const f64);
            let x1_0 = vld1q_f64(src0.as_ptr().add(g + s) as *const f64);
            let x2_0 = vld1q_f64(src0.as_ptr().add(g + 2 * s) as *const f64);
            let x3_0 = vld1q_f64(src0.as_ptr().add(g + 3 * s) as *const f64);

            let x0_1 = vld1q_f64(src1.as_ptr().add(g) as *const f64);
            let x1_1 = vld1q_f64(src1.as_ptr().add(g + s) as *const f64);
            let x2_1 = vld1q_f64(src1.as_ptr().add(g + 2 * s) as *const f64);
            let x3_1 = vld1q_f64(src1.as_ptr().add(g + 3 * s) as *const f64);

            // Interleave lane-0 = FFT0, lane-1 = FFT1
            let x0_re = vtrn1q_f64(x0_0, x0_1);
            let x0_im = vtrn2q_f64(x0_0, x0_1);
            let x1_re = vtrn1q_f64(x1_0, x1_1);
            let x1_im = vtrn2q_f64(x1_0, x1_1);
            let x2_re = vtrn1q_f64(x2_0, x2_1);
            let x2_im = vtrn2q_f64(x2_0, x2_1);
            let x3_re = vtrn1q_f64(x3_0, x3_1);
            let x3_im = vtrn2q_f64(x3_0, x3_1);

            // Radix-4 butterfly (vectorized, no twiddles for m=1)
            let a_re = vaddq_f64(x0_re, x2_re);
            let a_im = vaddq_f64(x0_im, x2_im);
            let b_re = vsubq_f64(x0_re, x2_re);
            let b_im = vsubq_f64(x0_im, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            // Store interleaved: each logical position p occupies 2 physical slots.
            let base = 8 * g;
            vst2q_f64(
                dst.as_mut_ptr().add(base) as *mut f64,
                float64x2x2_t(y0_re, y0_im),
            );
            vst2q_f64(
                dst.as_mut_ptr().add(base + 2) as *mut f64,
                float64x2x2_t(y1_re, y1_im),
            );
            vst2q_f64(
                dst.as_mut_ptr().add(base + 4) as *mut f64,
                float64x2x2_t(y2_re, y2_im),
            );
            vst2q_f64(
                dst.as_mut_ptr().add(base + 6) as *mut f64,
                float64x2x2_t(y3_re, y3_im),
            );
        }
    }
}

/// Fused final two radix-4 stages (m=16,s=4 then m=64,s=1) for n=256,
/// with transpose-out.  Reads interleaved pair, writes original chunks.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn fused_final_radix4x4_256_transpose_out_pair(
    src: &[Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
    dst0: &mut [Complex<f64>],
    dst1: &mut [Complex<f64>],
) {
    unsafe {
        debug_assert_eq!(radix4_stage.radix, 4);
        debug_assert_eq!(radix4_stage.m, 16);
        debug_assert_eq!(radix4_stage.s, 4);
        debug_assert_eq!(final_stage.radix, 4);
        debug_assert_eq!(final_stage.m, 64);
        debug_assert_eq!(final_stage.s, 1);
        debug_assert_eq!(src.len(), 512);
        debug_assert_eq!(dst0.len(), 256);
        debug_assert_eq!(dst1.len(), 256);

        let tw1_r4 = radix4_stage.k_slice(1);
        let tw2_r4 = radix4_stage.k_slice(2);
        let tw3_r4 = radix4_stage.k_slice(3);

        let tw1_f = final_stage.k_slice(1);
        let tw2_f = final_stage.k_slice(2);
        let tw3_f = final_stage.k_slice(3);

        let dir_v = vdupq_n_f64(dir_sign);

        // Pre-compute the first sub-stage (m=16, s=4) outputs into a temp buffer.
        // We need 256 positions * 2 lanes = 512 elements.
        let mut pre = [Complex::new(0.0, 0.0); 512];
        for g in 0..4usize {
            for inner_l in 0..16usize {
                let x0 = vld2q_f64(src.as_ptr().add((g * 16 + inner_l) * 2) as *const f64);
                let x1 = vld2q_f64(src.as_ptr().add((g * 16 + inner_l + 64) * 2) as *const f64);
                let x2 = vld2q_f64(src.as_ptr().add((g * 16 + inner_l + 128) * 2) as *const f64);
                let x3 = vld2q_f64(src.as_ptr().add((g * 16 + inner_l + 192) * 2) as *const f64);

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if inner_l == 0 {
                    (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
                } else {
                    let t1 = tw1_r4[inner_l];
                    let t2 = tw2_r4[inner_l];
                    let t3 = tw3_r4[inner_l];
                    let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0.0, x2_re);
                let a_im = vaddq_f64(x0.1, x2_im);
                let b_re = vsubq_f64(x0.0, x2_re);
                let b_im = vsubq_f64(x0.1, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                let base_out = (g * 64 + inner_l) * 2;
                pre[base_out] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
                pre[base_out + 1] =
                    Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
                pre[base_out + 32] =
                    Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
                pre[base_out + 33] =
                    Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
                pre[base_out + 64] =
                    Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
                pre[base_out + 65] =
                    Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
                pre[base_out + 96] =
                    Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
                pre[base_out + 97] =
                    Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
            }
        }

        // Now apply the final stage (m=64, s=1) and transpose-out.
        for l in 0..64usize {
            let x0 = vld2q_f64(pre.as_ptr().add(l * 2) as *const f64);
            let x1 = vld2q_f64(pre.as_ptr().add((l + 64) * 2) as *const f64);
            let x2 = vld2q_f64(pre.as_ptr().add((l + 128) * 2) as *const f64);
            let x3 = vld2q_f64(pre.as_ptr().add((l + 192) * 2) as *const f64);

            let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
            } else {
                let t1 = tw1_f[l];
                let t2 = tw2_f[l];
                let t3 = tw3_f[l];
                let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                (r1, i1, r2, i2, r3, i3)
            };

            let a_re = vaddq_f64(x0.0, x2_re);
            let a_im = vaddq_f64(x0.1, x2_im);
            let b_re = vsubq_f64(x0.0, x2_re);
            let b_im = vsubq_f64(x0.1, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            dst0[l] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
            dst1[l] = Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
            dst0[64 + l] = Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
            dst1[64 + l] = Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
            dst0[128 + l] = Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
            dst1[128 + l] = Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
            dst0[192 + l] = Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
            dst1[192 + l] = Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::codelets::scalar as scalar_codelets;
    use approx::assert_relative_eq;
    use num_complex::{Complex32, Complex64};

    fn scalar_stage_radix2(
        stride: usize,
        src: &[Complex64],
        dst: &mut [Complex64],
        stage_tw: &StageTwiddles<f64>,
    ) {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for l in 0..m {
                let a = src[base_in + l];
                let b = src[base_in + l + stride] * tw1[l];
                let (y0, y1) = scalar_codelets::radix2(a, b);
                dst[base_out + l] = y0;
                dst[base_out + m + l] = y1;
            }
        }
    }

    fn assert_close(a: &[Complex64], b: &[Complex64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_relative_eq!(x.re, y.re, max_relative = 1e-13, epsilon = 1e-13);
            assert_relative_eq!(x.im, y.im, max_relative = 1e-13, epsilon = 1e-13);
        }
    }

    fn assert_close32(a: &[Complex32], b: &[Complex32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_relative_eq!(x.re, y.re, max_relative = 1e-5, epsilon = 1e-5);
            assert_relative_eq!(x.im, y.im, max_relative = 1e-5, epsilon = 1e-5);
        }
    }

    fn rand_complex(seed: u64, n: usize) -> Vec<Complex64> {
        let mut s = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let im = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                Complex64::new(re, im)
            })
            .collect()
    }

    fn rand_complex32(seed: u64, n: usize) -> Vec<Complex32> {
        rand_complex(seed, n)
            .into_iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect()
    }

    fn check_stage_for(m: usize, s: usize) {
        let n = m * 2 * s;
        let stride = n / 2;
        // Synthesise a `StageTwiddles` with this stage's m, s and the
        // radix-2 twiddles `(l · s mod n)`.
        let stage_tw = make_stage_twiddles_radix2(n.max(2), m, s);
        let src = rand_complex(m as u64 * 1009 + s as u64 * 251, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_neon = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix2(stride, &src, &mut dst_scalar, &stage_tw);
        unsafe {
            stage_radix2_f64(stride, &src, &mut dst_neon, &stage_tw);
        }
        assert_close(&dst_scalar, &dst_neon);
    }

    fn make_stage_twiddles_radix2(n: usize, m: usize, s: usize) -> StageTwiddles<f64> {
        // Direct construction: one length-m array of stride-1 twiddles,
        // tw1[l] = exp(-2π·i · l·s / n).
        use std::f64::consts::TAU;
        let mut flat = Vec::with_capacity(m);
        for l in 0..m {
            let j = (l * s) % n;
            let theta = -TAU * (j as f64) / (n as f64);
            let (sv, cv) = theta.sin_cos();
            flat.push(Complex64::new(cv, sv));
        }
        StageTwiddles::from_parts(2, m, s, flat)
    }

    #[test]
    fn neon_radix2_matches_scalar_various_shapes() {
        // Cover m even, odd, small, large; s = 1, 2, 4.
        check_stage_for(1, 1);
        check_stage_for(2, 1);
        check_stage_for(3, 1);
        check_stage_for(4, 1);
        check_stage_for(8, 4);
        check_stage_for(16, 2);
        check_stage_for(64, 8);
        check_stage_for(7, 3); // odd m to exercise the scalar tail
        check_stage_for(15, 5);
    }

    fn scalar_stage_radix4(
        stride: usize,
        src: &[Complex64],
        dst: &mut [Complex64],
        stage_tw: &StageTwiddles<f64>,
        dir_sign: f64,
    ) {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for l in 0..m {
                let x0 = src[base_in + l];
                let x1 = src[base_in + l + stride] * tw1[l];
                let x2 = src[base_in + l + 2 * stride] * tw2[l];
                let x3 = src[base_in + l + 3 * stride] * tw3[l];
                let (y0, y1, y2, y3) = scalar_codelets::radix4(x0, x1, x2, x3, dir_sign);
                dst[base_out + l] = y0;
                dst[base_out + m + l] = y1;
                dst[base_out + 2 * m + l] = y2;
                dst[base_out + 3 * m + l] = y3;
            }
        }
    }

    fn make_stage_twiddles_radix4(
        n: usize,
        m: usize,
        s: usize,
        dir_sign: f64,
    ) -> StageTwiddles<f64> {
        use std::f64::consts::TAU;
        let mut flat = Vec::with_capacity(3 * m);
        for k in 1..=3 {
            for l in 0..m {
                let j = (k * l * s) % n;
                let theta = dir_sign * TAU * (j as f64) / (n as f64);
                let (sv, cv) = theta.sin_cos();
                flat.push(Complex64::new(cv, sv));
            }
        }
        StageTwiddles::from_parts(4, m, s, flat)
    }

    fn scalar_stage_radix4_f32(
        stride: usize,
        src: &[Complex32],
        dst: &mut [Complex32],
        stage_tw: &StageTwiddles<f32>,
        dir_sign: f32,
    ) {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for l in 0..m {
                let x0 = src[base_in + l];
                let x1 = src[base_in + l + stride] * tw1[l];
                let x2 = src[base_in + l + 2 * stride] * tw2[l];
                let x3 = src[base_in + l + 3 * stride] * tw3[l];
                let (y0, y1, y2, y3) = scalar_codelets::radix4(x0, x1, x2, x3, dir_sign);
                dst[base_out + l] = y0;
                dst[base_out + m + l] = y1;
                dst[base_out + 2 * m + l] = y2;
                dst[base_out + 3 * m + l] = y3;
            }
        }
    }

    fn make_stage_twiddles_radix4_f32(
        n: usize,
        m: usize,
        s: usize,
        dir_sign: f32,
    ) -> StageTwiddles<f32> {
        use std::f32::consts::TAU;
        let mut flat = Vec::with_capacity(3 * m);
        for k in 1..=3 {
            for l in 0..m {
                let j = (k * l * s) % n;
                let theta = dir_sign * TAU * (j as f32) / (n as f32);
                let (sv, cv) = theta.sin_cos();
                flat.push(Complex32::new(cv, sv));
            }
        }
        StageTwiddles::from_parts(4, m, s, flat)
    }

    fn check_radix4_for(m: usize, s: usize, dir_sign: f64) {
        let n = m * 4 * s;
        let stride = n / 4;
        let stage_tw = make_stage_twiddles_radix4(n.max(4), m, s, dir_sign);
        let src = rand_complex(m as u64 * 1013 + s as u64 * 257, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_neon = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix4(stride, &src, &mut dst_scalar, &stage_tw, dir_sign);
        unsafe {
            stage_radix4_f64(stride, &src, &mut dst_neon, &stage_tw, dir_sign);
        }
        assert_close(&dst_scalar, &dst_neon);
    }

    fn check_radix4_f32_for(m: usize, s: usize, dir_sign: f32) {
        let n = m * 4 * s;
        let stride = n / 4;
        let stage_tw = make_stage_twiddles_radix4_f32(n.max(4), m, s, dir_sign);
        let src = rand_complex32(m as u64 * 1019 + s as u64 * 263, n);
        let mut dst_scalar = vec![Complex32::new(0.0, 0.0); n];
        let mut dst_neon = vec![Complex32::new(0.0, 0.0); n];
        scalar_stage_radix4_f32(stride, &src, &mut dst_scalar, &stage_tw, dir_sign);
        unsafe {
            stage_radix4_f32(stride, &src, &mut dst_neon, &stage_tw, dir_sign);
        }
        assert_close32(&dst_scalar, &dst_neon);
    }

    #[test]
    fn neon_radix4_matches_scalar_forward() {
        check_radix4_for(1, 1, -1.0);
        check_radix4_for(2, 1, -1.0);
        check_radix4_for(4, 1, -1.0);
        check_radix4_for(8, 8, -1.0);
        check_radix4_for(16, 4, -1.0);
        check_radix4_for(64, 4, -1.0);
        check_radix4_for(3, 2, -1.0); // odd m
        check_radix4_for(7, 3, -1.0);
    }

    #[test]
    fn neon_radix4_matches_scalar_inverse() {
        check_radix4_for(4, 1, 1.0);
        check_radix4_for(16, 4, 1.0);
        check_radix4_for(64, 4, 1.0);
        check_radix4_for(7, 3, 1.0);
    }

    #[test]
    fn neon_radix4_f32_matches_scalar_forward() {
        check_radix4_f32_for(1, 1, -1.0);
        check_radix4_f32_for(4, 1, -1.0);
        check_radix4_f32_for(8, 8, -1.0);
        check_radix4_f32_for(16, 4, -1.0);
        check_radix4_f32_for(64, 4, -1.0);
        check_radix4_f32_for(5, 2, -1.0);
        check_radix4_f32_for(7, 3, -1.0);
    }

    #[test]
    fn neon_radix4_f32_matches_scalar_inverse() {
        check_radix4_f32_for(4, 1, 1.0);
        check_radix4_f32_for(16, 4, 1.0);
        check_radix4_f32_for(64, 4, 1.0);
        check_radix4_f32_for(7, 3, 1.0);
    }

    #[test]
    fn neon_fused_final_4096_matches_two_scalar_stages() {
        let n = 4096;
        for &dir_sign in &[-1.0, 1.0] {
            let radix4_stage = make_stage_twiddles_radix4(n, 256, 4, dir_sign);
            let final_stage = make_stage_twiddles_radix4(n, 1024, 1, dir_sign);
            let input = rand_complex(0xF00D + dir_sign.to_bits(), n);

            let mut after_first = vec![Complex64::new(0.0, 0.0); n];
            scalar_stage_radix4(n / 4, &input, &mut after_first, &radix4_stage, dir_sign);
            let mut expected = vec![Complex64::new(0.0, 0.0); n];
            scalar_stage_radix4(n / 4, &after_first, &mut expected, &final_stage, dir_sign);

            let mut actual = input;
            unsafe {
                fused_final_radix4x4_4096_in_place(
                    &mut actual,
                    &radix4_stage,
                    &final_stage,
                    dir_sign,
                );
            }
            assert_close(&actual, &expected);
        }
    }
}
/// Fused final two radix-4 stages (m=64,s=4 then m=256,s=1) for n=1024,
/// with transpose-out.  Reads interleaved pair, writes original chunks.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn fused_final_radix4x4_1024_transpose_out_pair(
    src: &[Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
    dst0: &mut [Complex<f64>],
    dst1: &mut [Complex<f64>],
) {
    unsafe {
        debug_assert_eq!(radix4_stage.radix, 4);
        debug_assert_eq!(radix4_stage.m, 64);
        debug_assert_eq!(radix4_stage.s, 4);
        debug_assert_eq!(final_stage.radix, 4);
        debug_assert_eq!(final_stage.m, 256);
        debug_assert_eq!(final_stage.s, 1);
        debug_assert_eq!(src.len(), 2048);
        debug_assert_eq!(dst0.len(), 1024);
        debug_assert_eq!(dst1.len(), 1024);

        let tw1_r4 = radix4_stage.k_slice(1);
        let tw2_r4 = radix4_stage.k_slice(2);
        let tw3_r4 = radix4_stage.k_slice(3);

        let tw1_f = final_stage.k_slice(1);
        let tw2_f = final_stage.k_slice(2);
        let tw3_f = final_stage.k_slice(3);

        let dir_v = vdupq_n_f64(dir_sign);

        let mut pre = [Complex::new(0.0, 0.0); 2048];
        for g in 0..4usize {
            for inner_l in 0..64usize {
                let x0 = vld2q_f64(src.as_ptr().add((g * 64 + inner_l) * 2) as *const f64);
                let x1 = vld2q_f64(src.as_ptr().add((g * 64 + inner_l + 256) * 2) as *const f64);
                let x2 = vld2q_f64(src.as_ptr().add((g * 64 + inner_l + 512) * 2) as *const f64);
                let x3 = vld2q_f64(src.as_ptr().add((g * 64 + inner_l + 768) * 2) as *const f64);

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if inner_l == 0 {
                    (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
                } else {
                    let t1 = tw1_r4[inner_l];
                    let t2 = tw2_r4[inner_l];
                    let t3 = tw3_r4[inner_l];
                    let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0.0, x2_re);
                let a_im = vaddq_f64(x0.1, x2_im);
                let b_re = vsubq_f64(x0.0, x2_re);
                let b_im = vsubq_f64(x0.1, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                let base_out = (g * 256 + inner_l) * 2;
                pre[base_out] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
                pre[base_out + 1] =
                    Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
                pre[base_out + 128] =
                    Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
                pre[base_out + 129] =
                    Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
                pre[base_out + 256] =
                    Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
                pre[base_out + 257] =
                    Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
                pre[base_out + 384] =
                    Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
                pre[base_out + 385] =
                    Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
            }
        }

        for l in 0..256usize {
            let x0 = vld2q_f64(pre.as_ptr().add(l * 2) as *const f64);
            let x1 = vld2q_f64(pre.as_ptr().add((l + 256) * 2) as *const f64);
            let x2 = vld2q_f64(pre.as_ptr().add((l + 512) * 2) as *const f64);
            let x3 = vld2q_f64(pre.as_ptr().add((l + 768) * 2) as *const f64);

            let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
            } else {
                let t1 = tw1_f[l];
                let t2 = tw2_f[l];
                let t3 = tw3_f[l];
                let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                (r1, i1, r2, i2, r3, i3)
            };

            let a_re = vaddq_f64(x0.0, x2_re);
            let a_im = vaddq_f64(x0.1, x2_im);
            let b_re = vsubq_f64(x0.0, x2_re);
            let b_im = vsubq_f64(x0.1, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            dst0[l] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
            dst1[l] = Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
            dst0[256 + l] = Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
            dst1[256 + l] = Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
            dst0[512 + l] = Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
            dst1[512 + l] = Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
            dst0[768 + l] = Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
            dst1[768 + l] = Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
        }
    }
}
/// Fused final two radix-4 stages (m=256,s=4 then m=1024,s=1) for n=4096,
/// with transpose-out.  Reads interleaved pair, writes original chunks.
/// Marked cold and never-inline to avoid icache pollution on small-size paths.
#[inline(never)]
#[cold]
#[target_feature(enable = "neon")]
pub unsafe fn fused_final_radix4x4_4096_transpose_out_pair(
    src: &[Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
    dst0: &mut [Complex<f64>],
    dst1: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    unsafe {
        debug_assert_eq!(radix4_stage.radix, 4);
        debug_assert_eq!(radix4_stage.m, 256);
        debug_assert_eq!(radix4_stage.s, 4);
        debug_assert_eq!(final_stage.radix, 4);
        debug_assert_eq!(final_stage.m, 1024);
        debug_assert_eq!(final_stage.s, 1);
        debug_assert_eq!(src.len(), 8192);
        debug_assert_eq!(dst0.len(), 4096);
        debug_assert_eq!(dst1.len(), 4096);

        let tw1_r4 = radix4_stage.k_slice(1);
        let tw2_r4 = radix4_stage.k_slice(2);
        let tw3_r4 = radix4_stage.k_slice(3);

        let tw1_f = final_stage.k_slice(1);
        let tw2_f = final_stage.k_slice(2);
        let tw3_f = final_stage.k_slice(3);

        let dir_v = vdupq_n_f64(dir_sign);
        debug_assert!(scratch.len() >= 8192);
        let pre = &mut scratch[..8192];
        for g in 0..4usize {
            for inner_l in 0..256usize {
                let x0 = vld2q_f64(src.as_ptr().add((g * 256 + inner_l) * 2) as *const f64);
                let x1 = vld2q_f64(src.as_ptr().add((g * 256 + inner_l + 1024) * 2) as *const f64);
                let x2 = vld2q_f64(src.as_ptr().add((g * 256 + inner_l + 2048) * 2) as *const f64);
                let x3 = vld2q_f64(src.as_ptr().add((g * 256 + inner_l + 3072) * 2) as *const f64);

                let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if inner_l == 0 {
                    (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
                } else {
                    let t1 = tw1_r4[inner_l];
                    let t2 = tw2_r4[inner_l];
                    let t3 = tw3_r4[inner_l];
                    let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                    let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                    let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                    (r1, i1, r2, i2, r3, i3)
                };

                let a_re = vaddq_f64(x0.0, x2_re);
                let a_im = vaddq_f64(x0.1, x2_im);
                let b_re = vsubq_f64(x0.0, x2_re);
                let b_im = vsubq_f64(x0.1, x2_im);
                let c_re = vaddq_f64(x1_re, x3_re);
                let c_im = vaddq_f64(x1_im, x3_im);
                let d_re = vsubq_f64(x1_re, x3_re);
                let d_im = vsubq_f64(x1_im, x3_im);

                let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
                let id_im = vmulq_f64(d_re, dir_v);

                let y0_re = vaddq_f64(a_re, c_re);
                let y0_im = vaddq_f64(a_im, c_im);
                let y2_re = vsubq_f64(a_re, c_re);
                let y2_im = vsubq_f64(a_im, c_im);
                let y1_re = vaddq_f64(b_re, id_re);
                let y1_im = vaddq_f64(b_im, id_im);
                let y3_re = vsubq_f64(b_re, id_re);
                let y3_im = vsubq_f64(b_im, id_im);

                let base_out = (g * 1024 + inner_l) * 2;
                pre[base_out] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
                pre[base_out + 1] =
                    Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
                pre[base_out + 512] =
                    Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
                pre[base_out + 513] =
                    Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
                pre[base_out + 1024] =
                    Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
                pre[base_out + 1025] =
                    Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
                pre[base_out + 1536] =
                    Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
                pre[base_out + 1537] =
                    Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
            }
        }

        for l in 0..1024usize {
            let x0 = vld2q_f64(pre.as_ptr().add(l * 2) as *const f64);
            let x1 = vld2q_f64(pre.as_ptr().add((l + 1024) * 2) as *const f64);
            let x2 = vld2q_f64(pre.as_ptr().add((l + 2048) * 2) as *const f64);
            let x3 = vld2q_f64(pre.as_ptr().add((l + 3072) * 2) as *const f64);

            let (x1_re, x1_im, x2_re, x2_im, x3_re, x3_im) = if l == 0 {
                (x1.0, x1.1, x2.0, x2.1, x3.0, x3.1)
            } else {
                let t1 = tw1_f[l];
                let t2 = tw2_f[l];
                let t3 = tw3_f[l];
                let (r1, i1) = cmul_simd(x1.0, x1.1, vdupq_n_f64(t1.re), vdupq_n_f64(t1.im));
                let (r2, i2) = cmul_simd(x2.0, x2.1, vdupq_n_f64(t2.re), vdupq_n_f64(t2.im));
                let (r3, i3) = cmul_simd(x3.0, x3.1, vdupq_n_f64(t3.re), vdupq_n_f64(t3.im));
                (r1, i1, r2, i2, r3, i3)
            };

            let a_re = vaddq_f64(x0.0, x2_re);
            let a_im = vaddq_f64(x0.1, x2_im);
            let b_re = vsubq_f64(x0.0, x2_re);
            let b_im = vsubq_f64(x0.1, x2_im);
            let c_re = vaddq_f64(x1_re, x3_re);
            let c_im = vaddq_f64(x1_im, x3_im);
            let d_re = vsubq_f64(x1_re, x3_re);
            let d_im = vsubq_f64(x1_im, x3_im);

            let id_re = vnegq_f64(vmulq_f64(d_im, dir_v));
            let id_im = vmulq_f64(d_re, dir_v);

            let y0_re = vaddq_f64(a_re, c_re);
            let y0_im = vaddq_f64(a_im, c_im);
            let y2_re = vsubq_f64(a_re, c_re);
            let y2_im = vsubq_f64(a_im, c_im);
            let y1_re = vaddq_f64(b_re, id_re);
            let y1_im = vaddq_f64(b_im, id_im);
            let y3_re = vsubq_f64(b_re, id_re);
            let y3_im = vsubq_f64(b_im, id_im);

            dst0[l] = Complex::new(vgetq_lane_f64(y0_re, 0), vgetq_lane_f64(y0_im, 0));
            dst1[l] = Complex::new(vgetq_lane_f64(y0_re, 1), vgetq_lane_f64(y0_im, 1));
            dst0[1024 + l] = Complex::new(vgetq_lane_f64(y1_re, 0), vgetq_lane_f64(y1_im, 0));
            dst1[1024 + l] = Complex::new(vgetq_lane_f64(y1_re, 1), vgetq_lane_f64(y1_im, 1));
            dst0[2048 + l] = Complex::new(vgetq_lane_f64(y2_re, 0), vgetq_lane_f64(y2_im, 0));
            dst1[2048 + l] = Complex::new(vgetq_lane_f64(y2_re, 1), vgetq_lane_f64(y2_im, 1));
            dst0[3072 + l] = Complex::new(vgetq_lane_f64(y3_re, 0), vgetq_lane_f64(y3_im, 0));
            dst1[3072 + l] = Complex::new(vgetq_lane_f64(y3_re, 1), vgetq_lane_f64(y3_im, 1));
        }
    }
}
