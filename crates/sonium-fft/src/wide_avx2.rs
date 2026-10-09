//! AVX2-specialised wide-butterfly stages. x86_64-only.
//!
//! Processes four consecutive `Complex<f64>` butterflies per loop iteration.
//! Complex input is interleaved in memory, so each stream uses two unaligned
//! 256-bit loads plus a lane shuffle to form planar real/imaginary vectors.
//! Stage twiddles are stride-1, which keeps twiddle loads contiguous too.

use std::arch::x86_64::*;

use num_complex::Complex;

use crate::twiddles::StageTwiddles;

const DEINTERLEAVE_PD: i32 = 0b11_01_10_00;

#[inline]
unsafe fn load_complex4(ptr: *const Complex<f64>) -> (__m256d, __m256d) {
    unsafe {
        let lo = _mm256_loadu_pd(ptr as *const f64);
        let hi = _mm256_loadu_pd((ptr as *const f64).add(4));
        let re = _mm256_permute4x64_pd(_mm256_unpacklo_pd(lo, hi), DEINTERLEAVE_PD);
        let im = _mm256_permute4x64_pd(_mm256_unpackhi_pd(lo, hi), DEINTERLEAVE_PD);
        (re, im)
    }
}

#[inline]
unsafe fn store_complex4(ptr: *mut Complex<f64>, re: __m256d, im: __m256d) {
    unsafe {
        let re_perm = _mm256_permute4x64_pd(re, DEINTERLEAVE_PD);
        let im_perm = _mm256_permute4x64_pd(im, DEINTERLEAVE_PD);
        let lo = _mm256_unpacklo_pd(re_perm, im_perm);
        let hi = _mm256_unpackhi_pd(re_perm, im_perm);
        _mm256_storeu_pd(ptr as *mut f64, lo);
        _mm256_storeu_pd((ptr as *mut f64).add(4), hi);
    }
}

#[inline]
unsafe fn cmul(a_re: __m256d, a_im: __m256d, b_re: __m256d, b_im: __m256d) -> (__m256d, __m256d) {
    unsafe {
        let re = _mm256_fmsub_pd(a_re, b_re, _mm256_mul_pd(a_im, b_im));
        let im = _mm256_fmadd_pd(a_re, b_im, _mm256_mul_pd(a_im, b_re));
        (re, im)
    }
}

#[inline]
unsafe fn load_complex2(ptr: *const Complex<f64>) -> (__m128d, __m128d) {
    unsafe {
        let v0 = _mm_loadu_pd(ptr as *const f64);
        let v1 = _mm_loadu_pd((ptr as *const f64).add(2));
        let re = _mm_shuffle_pd(v0, v1, 0b00);
        let im = _mm_shuffle_pd(v0, v1, 0b11);
        (re, im)
    }
}

#[inline]
unsafe fn store_complex2(ptr: *mut Complex<f64>, re: __m128d, im: __m128d) {
    unsafe {
        let v0 = _mm_shuffle_pd(re, im, 0b00);
        let v1 = _mm_shuffle_pd(re, im, 0b11);
        _mm_storeu_pd(ptr as *mut f64, v0);
        _mm_storeu_pd((ptr as *mut f64).add(2), v1);
    }
}

#[inline]
unsafe fn cmul2(a_re: __m128d, a_im: __m128d, b_re: __m128d, b_im: __m128d) -> (__m128d, __m128d) {
    unsafe {
        let re = _mm_fmsub_pd(a_re, b_re, _mm_mul_pd(a_im, b_im));
        let im = _mm_fmadd_pd(a_re, b_im, _mm_mul_pd(a_im, b_re));
        (re, im)
    }
}

#[inline]
unsafe fn load_complex2_f32(ptr: *const Complex<f32>) -> (__m128, __m128) {
    unsafe {
        let values = _mm_loadu_ps(ptr as *const f32);
        // The physical layout is [re0, im0, re1, im1]. Keep the first two
        // logical lanes duplicated so one pair of SIMD arithmetic operations
        // handles both independent transforms.
        let re = _mm_shuffle_ps(values, values, 0x88);
        let im = _mm_shuffle_ps(values, values, 0xdd);
        (re, im)
    }
}

#[inline]
unsafe fn store_complex2_f32(ptr: *mut Complex<f32>, re: __m128, im: __m128) {
    unsafe {
        let values = _mm_unpacklo_ps(re, im);
        _mm_storeu_ps(ptr as *mut f32, values);
    }
}

#[inline]
unsafe fn cmul_f32(a_re: __m128, a_im: __m128, b_re: __m128, b_im: __m128) -> (__m128, __m128) {
    unsafe {
        let re = _mm_fmsub_ps(a_re, b_re, _mm_mul_ps(a_im, b_im));
        let im = _mm_fmadd_ps(a_re, b_im, _mm_mul_ps(a_im, b_re));
        (re, im)
    }
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-2 stage with `src.len() == dst.len() == 2 * stage_tw.m *
/// stage_tw.s`, `stride == src.len() / 2`, and disjoint source/destination
/// slices. The twiddle slices must therefore contain the stage's complete
/// `m` entries.
pub unsafe fn stage_radix2_f64(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
) {
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
            if l + 4 <= m {
                let (mut tw_re, mut tw_im) = load_complex4(tw1.as_ptr().add(l));
                loop {
                    let (a_re, a_im) = load_complex4(src.as_ptr().add(base_in + l));
                    let (b_re, b_im) = load_complex4(src.as_ptr().add(base_in + l + stride));
                    let (bt_re, bt_im) = cmul(b_re, b_im, tw_re, tw_im);

                    let y0_re = _mm256_add_pd(a_re, bt_re);
                    let y0_im = _mm256_add_pd(a_im, bt_im);
                    let y1_re = _mm256_sub_pd(a_re, bt_re);
                    let y1_im = _mm256_sub_pd(a_im, bt_im);

                    store_complex4(dst.as_mut_ptr().add(base_out + l), y0_re, y0_im);
                    store_complex4(dst.as_mut_ptr().add(base_out + m + l), y1_re, y1_im);

                    l += 4;
                    if l + 4 > m {
                        break;
                    }
                    let (tw_re_next, tw_im_next) = load_complex4(tw1.as_ptr().add(l));
                    tw_re = tw_re_next;
                    tw_im = tw_im_next;
                }
            }

            if l + 2 <= m {
                let (a_re, a_im) = load_complex2(src.as_ptr().add(base_in + l));
                let (b_re, b_im) = load_complex2(src.as_ptr().add(base_in + l + stride));
                let (tw_re, tw_im) = load_complex2(tw1.as_ptr().add(l));
                let (bt_re, bt_im) = cmul2(b_re, b_im, tw_re, tw_im);
                let y0_re = _mm_add_pd(a_re, bt_re);
                let y0_im = _mm_add_pd(a_im, bt_im);
                let y1_re = _mm_sub_pd(a_re, bt_re);
                let y1_im = _mm_sub_pd(a_im, bt_im);
                store_complex2(dst.as_mut_ptr().add(base_out + l), y0_re, y0_im);
                store_complex2(dst.as_mut_ptr().add(base_out + m + l), y1_re, y1_im);
                l += 2;
            }

            if l < m {
                let a = src[base_in + l];
                let b = if l == 0 {
                    src[base_in + l + stride]
                } else {
                    src[base_in + l + stride] * tw1[l]
                };
                dst[base_out + l] = a + b;
                dst[base_out + m + l] = a - b;
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(unused_assignments)]
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-4 stage with `src.len() == dst.len() == 4 * stage_tw.m *
/// stage_tw.s`, `stride == src.len() / 4`, and disjoint source/destination
/// slices. The twiddle slices must contain the stage's complete `m` entries.
pub unsafe fn stage_radix4_f64(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        stage_tw.debug_assert_stage_shape(4, stride, src.len(), dst.len());
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let _dir_v = _mm256_set1_pd(dir_sign);
        let _sign_flip = _mm256_set1_pd(-0.0);
        let dir_v2 = _mm_set1_pd(dir_sign);
        let sign_flip2 = _mm_set1_pd(-0.0);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            if l + 2 <= m {
                let (mut tw1_re, mut tw1_im) = load_complex2(tw1.as_ptr().add(l));
                let (mut tw2_re, mut tw2_im) = load_complex2(tw2.as_ptr().add(l));
                let (mut tw3_re, mut tw3_im) = load_complex2(tw3.as_ptr().add(l));
                loop {
                    let (x0_re, x0_im) = load_complex2(src.as_ptr().add(base_in + l));
                    let (x1_re, x1_im) = load_complex2(src.as_ptr().add(base_in + l + stride));
                    let (x2_re, x2_im) = load_complex2(src.as_ptr().add(base_in + l + 2 * stride));
                    let (x3_re, x3_im) = load_complex2(src.as_ptr().add(base_in + l + 3 * stride));

                    let (x1_re, x1_im) = cmul2(x1_re, x1_im, tw1_re, tw1_im);
                    let (x2_re, x2_im) = cmul2(x2_re, x2_im, tw2_re, tw2_im);
                    let (x3_re, x3_im) = cmul2(x3_re, x3_im, tw3_re, tw3_im);

                    let a_re = _mm_add_pd(x0_re, x2_re);
                    let a_im = _mm_add_pd(x0_im, x2_im);
                    let b_re = _mm_sub_pd(x0_re, x2_re);
                    let b_im = _mm_sub_pd(x0_im, x2_im);
                    let c_re = _mm_add_pd(x1_re, x3_re);
                    let c_im = _mm_add_pd(x1_im, x3_im);
                    let d_re = _mm_sub_pd(x1_re, x3_re);
                    let d_im = _mm_sub_pd(x1_im, x3_im);

                    let id_re = _mm_xor_pd(_mm_mul_pd(d_im, dir_v2), sign_flip2);
                    let id_im = _mm_mul_pd(d_re, dir_v2);

                    let y0_re = _mm_add_pd(a_re, c_re);
                    let y0_im = _mm_add_pd(a_im, c_im);
                    let y2_re = _mm_sub_pd(a_re, c_re);
                    let y2_im = _mm_sub_pd(a_im, c_im);
                    let y1_re = _mm_add_pd(b_re, id_re);
                    let y1_im = _mm_add_pd(b_im, id_im);
                    let y3_re = _mm_sub_pd(b_re, id_re);
                    let y3_im = _mm_sub_pd(b_im, id_im);

                    store_complex2(dst.as_mut_ptr().add(base_out + l), y0_re, y0_im);
                    store_complex2(dst.as_mut_ptr().add(base_out + m + l), y1_re, y1_im);
                    store_complex2(dst.as_mut_ptr().add(base_out + 2 * m + l), y2_re, y2_im);
                    store_complex2(dst.as_mut_ptr().add(base_out + 3 * m + l), y3_re, y3_im);

                    l += 2;
                    if l + 2 > m {
                        break;
                    }
                    let (tw1_re_next, tw1_im_next) = load_complex2(tw1.as_ptr().add(l));
                    let (tw2_re_next, tw2_im_next) = load_complex2(tw2.as_ptr().add(l));
                    let (tw3_re_next, tw3_im_next) = load_complex2(tw3.as_ptr().add(l));
                    tw1_re = tw1_re_next;
                    tw1_im = tw1_im_next;
                    tw2_re = tw2_re_next;
                    tw2_im = tw2_im_next;
                    tw3_re = tw3_re_next;
                    tw3_im = tw3_im_next;
                }
            }

            if l < m {
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
            if l < m {
                let x0 = src[base_in + l];
                let x1 = src[base_in + l + stride] * tw1[l];
                let x2 = src[base_in + l + 2 * stride] * tw2[l];
                let x3 = src[base_in + l + 3 * stride] * tw3[l];
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
            if l < m {
                let x0 = src[base_in + l];
                let x1 = src[base_in + l + stride] * tw1[l];
                let x2 = src[base_in + l + 2 * stride] * tw2[l];
                let x3 = src[base_in + l + 3 * stride] * tw3[l];
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

#[target_feature(enable = "avx2,fma")]
#[allow(unused_assignments)]
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw.s` must equal
/// one, `src.len() == dst.len() == 4 * stage_tw.m`, and source and destination
/// slices must be disjoint. The stage twiddle slices must contain `m` entries.
pub unsafe fn stage_radix4_f64_s1(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        let m = stage_tw.m;
        debug_assert_eq!(stage_tw.s, 1);
        let dir_v2 = _mm_set1_pd(dir_sign);
        let sign_flip2 = _mm_set1_pd(-0.0);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        let mut l = 0usize;
        if l + 2 <= m {
            let (mut tw1_re, mut tw1_im) = load_complex2(tw1.as_ptr().add(l));
            let (mut tw2_re, mut tw2_im) = load_complex2(tw2.as_ptr().add(l));
            let (mut tw3_re, mut tw3_im) = load_complex2(tw3.as_ptr().add(l));
            loop {
                let (x0_re, x0_im) = load_complex2(src.as_ptr().add(l));
                let (x1_re, x1_im) = load_complex2(src.as_ptr().add(l + m));
                let (x2_re, x2_im) = load_complex2(src.as_ptr().add(l + 2 * m));
                let (x3_re, x3_im) = load_complex2(src.as_ptr().add(l + 3 * m));

                let (x1_re, x1_im) = cmul2(x1_re, x1_im, tw1_re, tw1_im);
                let (x2_re, x2_im) = cmul2(x2_re, x2_im, tw2_re, tw2_im);
                let (x3_re, x3_im) = cmul2(x3_re, x3_im, tw3_re, tw3_im);

                let a_re = _mm_add_pd(x0_re, x2_re);
                let a_im = _mm_add_pd(x0_im, x2_im);
                let b_re = _mm_sub_pd(x0_re, x2_re);
                let b_im = _mm_sub_pd(x0_im, x2_im);
                let c_re = _mm_add_pd(x1_re, x3_re);
                let c_im = _mm_add_pd(x1_im, x3_im);
                let d_re = _mm_sub_pd(x1_re, x3_re);
                let d_im = _mm_sub_pd(x1_im, x3_im);

                let id_re = _mm_xor_pd(_mm_mul_pd(d_im, dir_v2), sign_flip2);
                let id_im = _mm_mul_pd(d_re, dir_v2);

                let y0_re = _mm_add_pd(a_re, c_re);
                let y0_im = _mm_add_pd(a_im, c_im);
                let y2_re = _mm_sub_pd(a_re, c_re);
                let y2_im = _mm_sub_pd(a_im, c_im);
                let y1_re = _mm_add_pd(b_re, id_re);
                let y1_im = _mm_add_pd(b_im, id_im);
                let y3_re = _mm_sub_pd(b_re, id_re);
                let y3_im = _mm_sub_pd(b_im, id_im);

                store_complex2(dst.as_mut_ptr().add(l), y0_re, y0_im);
                store_complex2(dst.as_mut_ptr().add(m + l), y1_re, y1_im);
                store_complex2(dst.as_mut_ptr().add(2 * m + l), y2_re, y2_im);
                store_complex2(dst.as_mut_ptr().add(3 * m + l), y3_re, y3_im);

                l += 2;
                if l + 2 > m {
                    break;
                }
                let (tw1_re_next, tw1_im_next) = load_complex2(tw1.as_ptr().add(l));
                let (tw2_re_next, tw2_im_next) = load_complex2(tw2.as_ptr().add(l));
                let (tw3_re_next, tw3_im_next) = load_complex2(tw3.as_ptr().add(l));
                tw1_re = tw1_re_next;
                tw1_im = tw1_im_next;
                tw2_re = tw2_re_next;
                tw2_im = tw2_im_next;
                tw3_re = tw3_re_next;
                tw3_im = tw3_im_next;
            }
        }

        if l < m {
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
        if l < m {
            let x0 = src[l];
            let x1 = src[l + m] * tw1[l];
            let x2 = src[l + 2 * m] * tw2[l];
            let x3 = src[l + 3 * m] * tw3[l];
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
        if l < m {
            let x0 = src[l];
            let x1 = src[l + m] * tw1[l];
            let x2 = src[l + 2 * m] * tw2[l];
            let x3 = src[l + 3 * m] * tw3[l];
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

/// AVX2/FMA radix-2 stage for two independent f64 FFTs interleaved as
/// `[fft0[0], fft1[0], fft0[1], fft1[1], …]`.
///
/// The logical `stride` and stage dimensions are expressed in elements of
/// one FFT; the physical slices therefore contain twice as many `Complex`
/// values. Broadcasting each logical twiddle keeps both transforms in the
/// two lanes of an SSE register while the surrounding AVX2 stage kernels
/// remain available for single-transform execution.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-2 stage, `src.len() == dst.len() == 2 * (2 * stage_tw.m *
/// stage_tw.s)`, and the source and destination slices must be disjoint.
/// `stride` must equal the logical distance between the two radix inputs
/// (`2 * stage_tw.m * stage_tw.s / 2`).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix2_f64_batched2(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
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
                let (a_re, a_im) = load_complex2(src.as_ptr().add(2 * (base_in + l)));
                let (b_re, b_im) = load_complex2(src.as_ptr().add(2 * (base_in + l + stride)));
                let tw_re = _mm_set1_pd(tw.re);
                let tw_im = _mm_set1_pd(tw.im);
                let (b_re, b_im) = cmul2(b_re, b_im, tw_re, tw_im);
                let y0_re = _mm_add_pd(a_re, b_re);
                let y0_im = _mm_add_pd(a_im, b_im);
                let y1_re = _mm_sub_pd(a_re, b_re);
                let y1_im = _mm_sub_pd(a_im, b_im);
                store_complex2(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
            }
        }
    }
}

/// AVX2/FMA radix-4 stage for two independent f64 FFTs interleaved as
/// `[fft0[0], fft1[0], fft0[1], fft1[1], …]`.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-4 stage, `src.len() == dst.len() == 2 * (4 * stage_tw.m *
/// stage_tw.s)`, and the source and destination slices must be disjoint.
/// `stride` is the logical distance between radix inputs for one FFT.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f64_batched2(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        let dir_v = _mm_set1_pd(dir_sign);
        let sign_flip = _mm_set1_pd(-0.0);

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
                let (x0_re, x0_im) = load_complex2(src.as_ptr().add(2 * (base_in + l)));
                let (mut x1_re, mut x1_im) =
                    load_complex2(src.as_ptr().add(2 * (base_in + l + stride)));
                let (mut x2_re, mut x2_im) =
                    load_complex2(src.as_ptr().add(2 * (base_in + l + 2 * stride)));
                let (mut x3_re, mut x3_im) =
                    load_complex2(src.as_ptr().add(2 * (base_in + l + 3 * stride)));

                (x1_re, x1_im) = cmul2(x1_re, x1_im, _mm_set1_pd(t1.re), _mm_set1_pd(t1.im));
                (x2_re, x2_im) = cmul2(x2_re, x2_im, _mm_set1_pd(t2.re), _mm_set1_pd(t2.im));
                (x3_re, x3_im) = cmul2(x3_re, x3_im, _mm_set1_pd(t3.re), _mm_set1_pd(t3.im));

                let a_re = _mm_add_pd(x0_re, x2_re);
                let a_im = _mm_add_pd(x0_im, x2_im);
                let b_re = _mm_sub_pd(x0_re, x2_re);
                let b_im = _mm_sub_pd(x0_im, x2_im);
                let c_re = _mm_add_pd(x1_re, x3_re);
                let c_im = _mm_add_pd(x1_im, x3_im);
                let d_re = _mm_sub_pd(x1_re, x3_re);
                let d_im = _mm_sub_pd(x1_im, x3_im);
                let id_re = _mm_xor_pd(_mm_mul_pd(d_im, dir_v), sign_flip);
                let id_im = _mm_mul_pd(d_re, dir_v);

                let y0_re = _mm_add_pd(a_re, c_re);
                let y0_im = _mm_add_pd(a_im, c_im);
                let y2_re = _mm_sub_pd(a_re, c_re);
                let y2_im = _mm_sub_pd(a_im, c_im);
                let y1_re = _mm_add_pd(b_re, id_re);
                let y1_im = _mm_add_pd(b_im, id_im);
                let y3_re = _mm_sub_pd(b_re, id_re);
                let y3_im = _mm_sub_pd(b_im, id_im);

                store_complex2(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
                store_complex2(
                    dst.as_mut_ptr().add(2 * (base_out + 2 * m + l)),
                    y2_re,
                    y2_im,
                );
                store_complex2(
                    dst.as_mut_ptr().add(2 * (base_out + 3 * m + l)),
                    y3_re,
                    y3_im,
                );
            }
        }
    }
}

/// AVX2/FMA radix-2 stage for two independent f32 FFTs interleaved as
/// `[fft0[0], fft1[0], fft0[1], fft1[1], …]`.
///
/// The two logical transforms occupy the two useful lanes in each `__m128`
/// real/imaginary vector. The remaining duplicate lanes make the complex
/// arithmetic uniform while the load/store helpers preserve the public
/// interleaved `Complex<f32>` layout.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-2 stage, `src.len() == dst.len() == 2 * n` where `n` is the logical
/// FFT length, and the source and destination slices must be disjoint.
#[target_feature(enable = "avx2,fma")]
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
                let (a_re, a_im) = load_complex2_f32(src.as_ptr().add(2 * (base_in + l)));
                let (b_re, b_im) = load_complex2_f32(src.as_ptr().add(2 * (base_in + l + stride)));
                let (b_re, b_im) = cmul_f32(b_re, b_im, _mm_set1_ps(tw.re), _mm_set1_ps(tw.im));
                let y0_re = _mm_add_ps(a_re, b_re);
                let y0_im = _mm_add_ps(a_im, b_im);
                let y1_re = _mm_sub_ps(a_re, b_re);
                let y1_im = _mm_sub_ps(a_im, b_im);
                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
            }
        }
    }
}

/// AVX2/FMA radix-4 stage for two independent f32 FFTs in the same
/// interleaved layout as [`stage_radix2_f32_batched2`].
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-4 stage, `src.len() == dst.len() == 2 * n`, and source and
/// destination slices must be disjoint.
#[target_feature(enable = "avx2,fma")]
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
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        let dir_v = _mm_set1_ps(dir_sign);
        let sign_flip = _mm_set1_ps(-0.0);

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
                let (x0_re, x0_im) = load_complex2_f32(src.as_ptr().add(2 * (base_in + l)));
                let (mut x1_re, mut x1_im) =
                    load_complex2_f32(src.as_ptr().add(2 * (base_in + l + stride)));
                let (mut x2_re, mut x2_im) =
                    load_complex2_f32(src.as_ptr().add(2 * (base_in + l + 2 * stride)));
                let (mut x3_re, mut x3_im) =
                    load_complex2_f32(src.as_ptr().add(2 * (base_in + l + 3 * stride)));

                (x1_re, x1_im) = cmul_f32(x1_re, x1_im, _mm_set1_ps(t1.re), _mm_set1_ps(t1.im));
                (x2_re, x2_im) = cmul_f32(x2_re, x2_im, _mm_set1_ps(t2.re), _mm_set1_ps(t2.im));
                (x3_re, x3_im) = cmul_f32(x3_re, x3_im, _mm_set1_ps(t3.re), _mm_set1_ps(t3.im));

                let a_re = _mm_add_ps(x0_re, x2_re);
                let a_im = _mm_add_ps(x0_im, x2_im);
                let b_re = _mm_sub_ps(x0_re, x2_re);
                let b_im = _mm_sub_ps(x0_im, x2_im);
                let c_re = _mm_add_ps(x1_re, x3_re);
                let c_im = _mm_add_ps(x1_im, x3_im);
                let d_re = _mm_sub_ps(x1_re, x3_re);
                let d_im = _mm_sub_ps(x1_im, x3_im);
                let id_re = _mm_xor_ps(_mm_mul_ps(d_im, dir_v), sign_flip);
                let id_im = _mm_mul_ps(d_re, dir_v);

                let y0_re = _mm_add_ps(a_re, c_re);
                let y0_im = _mm_add_ps(a_im, c_im);
                let y2_re = _mm_sub_ps(a_re, c_re);
                let y2_im = _mm_sub_ps(a_im, c_im);
                let y1_re = _mm_add_ps(b_re, id_re);
                let y1_im = _mm_add_ps(b_im, id_im);
                let y3_re = _mm_sub_ps(b_re, id_re);
                let y3_im = _mm_sub_ps(b_im, id_im);

                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
                store_complex2_f32(
                    dst.as_mut_ptr().add(2 * (base_out + 2 * m + l)),
                    y2_re,
                    y2_im,
                );
                store_complex2_f32(
                    dst.as_mut_ptr().add(2 * (base_out + 3 * m + l)),
                    y3_re,
                    y3_im,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codelets::scalar as scalar_codelets;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

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

    fn assert_close(a: &[Complex64], b: &[Complex64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_relative_eq!(x.re, y.re, max_relative = 1e-13, epsilon = 1e-13);
            assert_relative_eq!(x.im, y.im, max_relative = 1e-13, epsilon = 1e-13);
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

    fn make_stage_twiddles_radix2(n: usize, m: usize, s: usize) -> StageTwiddles<f64> {
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

    fn check_radix2_for(m: usize, s: usize) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * 2 * s;
        let stride = n / 2;
        let stage_tw = make_stage_twiddles_radix2(n.max(2), m, s);
        let src = rand_complex(m as u64 * 1009 + s as u64 * 251, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_avx2 = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix2(stride, &src, &mut dst_scalar, &stage_tw);
        unsafe {
            stage_radix2_f64(stride, &src, &mut dst_avx2, &stage_tw);
        }
        assert_close(&dst_scalar, &dst_avx2);
    }

    fn check_radix4_for(m: usize, s: usize, dir_sign: f64) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * 4 * s;
        let stride = n / 4;
        let stage_tw = make_stage_twiddles_radix4(n.max(4), m, s, dir_sign);
        let src = rand_complex(m as u64 * 1013 + s as u64 * 257, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_avx2 = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix4(stride, &src, &mut dst_scalar, &stage_tw, dir_sign);
        unsafe {
            stage_radix4_f64(stride, &src, &mut dst_avx2, &stage_tw, dir_sign);
        }
        assert_close(&dst_scalar, &dst_avx2);
    }

    fn check_radix4_s1_for(m: usize, dir_sign: f64) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * 4;
        let stage_tw = make_stage_twiddles_radix4(n.max(4), m, 1, dir_sign);
        let src = rand_complex(m as u64 * 1031 + 271, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_avx2 = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix4(n / 4, &src, &mut dst_scalar, &stage_tw, dir_sign);
        unsafe {
            stage_radix4_f64_s1(&src, &mut dst_avx2, &stage_tw, dir_sign);
        }
        assert_close(&dst_scalar, &dst_avx2);
    }

    fn check_batched2_f64_for(m: usize, s: usize, dir_sign: f64) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let radix = if m.is_multiple_of(2) { 2 } else { 4 };
        let n = m * radix * s;
        let stride = n / radix;
        let stage_tw = if radix == 2 {
            make_stage_twiddles_radix2(n.max(2), m, s)
        } else {
            make_stage_twiddles_radix4(n.max(4), m, s, dir_sign)
        };
        let src = rand_complex(0xB2F6_4000 + n as u64 + dir_sign as u64, 2 * n);
        let mut expected = vec![Complex64::new(0.0, 0.0); 2 * n];
        let mut actual = vec![Complex64::new(0.0, 0.0); 2 * n];
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m * radix;
            for l in 0..m {
                for transform in 0..2 {
                    let lane = transform;
                    let x0 = src[2 * (base_in + l) + lane];
                    if radix == 2 {
                        let x1 = src[2 * (base_in + l + stride) + lane] * stage_tw.k_slice(1)[l];
                        let (y0, y1) = scalar_codelets::radix2(x0, x1);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                    } else {
                        let tw1 = stage_tw.k_slice(1);
                        let tw2 = stage_tw.k_slice(2);
                        let tw3 = stage_tw.k_slice(3);
                        let x1 = src[2 * (base_in + l + stride) + lane] * tw1[l];
                        let x2 = src[2 * (base_in + l + 2 * stride) + lane] * tw2[l];
                        let x3 = src[2 * (base_in + l + 3 * stride) + lane] * tw3[l];
                        let (y0, y1, y2, y3) = scalar_codelets::radix4(x0, x1, x2, x3, dir_sign);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                        expected[2 * (base_out + 2 * m + l) + lane] = y2;
                        expected[2 * (base_out + 3 * m + l) + lane] = y3;
                    }
                }
            }
        }
        unsafe {
            if radix == 2 {
                stage_radix2_f64_batched2(stride, &src, &mut actual, &stage_tw);
            } else {
                stage_radix4_f64_batched2(stride, &src, &mut actual, &stage_tw, dir_sign);
            }
        }
        assert_close(&actual, &expected);
    }

    fn check_batched2_f32_for(m: usize, s: usize, dir_sign: f32, radix: usize) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * radix * s;
        let stride = n / radix;
        let stage_tw_f64 = if radix == 2 {
            make_stage_twiddles_radix2(n.max(2), m, s)
        } else {
            make_stage_twiddles_radix4(n.max(4), m, s, dir_sign as f64)
        };
        let mut twiddles = Vec::with_capacity((radix - 1) * m);
        for k in 1..radix {
            twiddles.extend(
                stage_tw_f64
                    .k_slice(k)
                    .iter()
                    .map(|value| Complex::new(value.re as f32, value.im as f32)),
            );
        }
        let stage_tw = StageTwiddles::from_parts(radix, m, s, twiddles);
        let src64 = rand_complex(0xB2F6_3200 + n as u64 + radix as u64, 2 * n);
        let src: Vec<_> = src64
            .into_iter()
            .map(|value| Complex::new(value.re as f32, value.im as f32))
            .collect();
        let mut expected = vec![Complex::new(0.0_f32, 0.0_f32); 2 * n];
        let mut actual = vec![Complex::new(0.0_f32, 0.0_f32); 2 * n];
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m * radix;
            for l in 0..m {
                for transform in 0..2 {
                    let lane = transform;
                    let x0 = src[2 * (base_in + l) + lane];
                    if radix == 2 {
                        let x1 = src[2 * (base_in + l + stride) + lane] * stage_tw.k_slice(1)[l];
                        let (y0, y1) = scalar_codelets::radix2(x0, x1);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                    } else {
                        let x1 = src[2 * (base_in + l + stride) + lane] * stage_tw.k_slice(1)[l];
                        let x2 =
                            src[2 * (base_in + l + 2 * stride) + lane] * stage_tw.k_slice(2)[l];
                        let x3 =
                            src[2 * (base_in + l + 3 * stride) + lane] * stage_tw.k_slice(3)[l];
                        let (y0, y1, y2, y3) = scalar_codelets::radix4(x0, x1, x2, x3, dir_sign);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                        expected[2 * (base_out + 2 * m + l) + lane] = y2;
                        expected[2 * (base_out + 3 * m + l) + lane] = y3;
                    }
                }
            }
        }
        unsafe {
            if radix == 2 {
                stage_radix2_f32_batched2(stride, &src, &mut actual, &stage_tw);
            } else {
                stage_radix4_f32_batched2(stride, &src, &mut actual, &stage_tw, dir_sign);
            }
        }
        for (actual, expected) in actual.iter().zip(expected) {
            assert_relative_eq!(actual.re, expected.re, max_relative = 5e-5, epsilon = 5e-5);
            assert_relative_eq!(actual.im, expected.im, max_relative = 5e-5, epsilon = 5e-5);
        }
    }

    #[test]
    fn avx2_radix2_matches_scalar_various_shapes() {
        for &(m, s) in &[(1, 1), (4, 1), (8, 4), (16, 2), (64, 8), (7, 3), (15, 5)] {
            check_radix2_for(m, s);
        }
    }

    #[test]
    fn avx2_radix4_matches_scalar_forward() {
        for &(m, s) in &[(1, 1), (4, 1), (8, 8), (16, 4), (64, 4), (5, 2), (7, 3)] {
            check_radix4_for(m, s, -1.0);
        }
    }

    #[test]
    fn avx2_radix4_matches_scalar_inverse() {
        for &(m, s) in &[(4, 1), (16, 4), (64, 4), (7, 3)] {
            check_radix4_for(m, s, 1.0);
        }
    }

    #[test]
    fn avx2_radix4_s1_matches_scalar_both_directions() {
        for &m in &[1, 2, 3, 4, 7, 16, 63, 257] {
            check_radix4_s1_for(m, -1.0);
            check_radix4_s1_for(m, 1.0);
        }
    }

    #[test]
    fn avx2_batched2_radix_stages_match_scalar_both_directions() {
        for &(m, s) in &[(1usize, 1usize), (2, 1), (3, 2), (4, 3), (7, 5), (15, 2)] {
            for &dir_sign in &[-1.0, 1.0] {
                check_batched2_f64_for(m, s, dir_sign);
            }
        }
    }

    #[test]
    fn avx2_batched2_f32_radix_stages_match_scalar_both_directions() {
        for &(radix, m, s) in &[
            (2usize, 1usize, 1usize),
            (2, 5, 3),
            (4, 1, 1),
            (4, 3, 2),
            (4, 7, 3),
        ] {
            for &dir_sign in &[-1.0_f32, 1.0_f32] {
                check_batched2_f32_for(m, s, dir_sign, radix);
            }
        }
    }
}
