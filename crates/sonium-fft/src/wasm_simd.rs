//! WebAssembly `simd128` Stockham stage kernels.
//!
//! The kernels operate on two adjacent `Complex<f32>` values at a time.
//! Values are kept in a structure-of-arrays representation while the
//! butterfly is evaluated, then interleaved only at the load/store boundary.
//! This keeps the arithmetic lane-parallel without imposing a new public
//! buffer layout on the FFT API.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use num_complex::Complex;

use crate::codelets::scalar::{radix2, radix4};
use crate::scalar::FftScalar;
use crate::twiddles::StageTwiddles;

#[derive(Clone, Copy)]
struct ComplexPair {
    re: v128,
    im: v128,
}

#[inline(always)]
unsafe fn load_pair<T>(values: &[Complex<T>], index: usize) -> ComplexPair
where
    T: FftScalar,
{
    debug_assert!(index + 1 < values.len());
    // The source and twiddle slices are interleaved `[re, im, re, im]`.
    // `i32x4_shuffle` is used as a lane deinterleave; all lanes after the
    // first two are duplicates and are harmless for the pair operations.
    let raw = unsafe { v128_load(values.as_ptr().add(index).cast()) };
    ComplexPair {
        re: i32x4_shuffle::<0, 2, 0, 2>(raw, raw),
        im: i32x4_shuffle::<1, 3, 1, 3>(raw, raw),
    }
}

#[inline(always)]
unsafe fn store_pair<T>(values: &mut [Complex<T>], index: usize, pair: ComplexPair)
where
    T: FftScalar,
{
    debug_assert!(index + 1 < values.len());
    let raw = i32x4_shuffle::<0, 4, 1, 5>(pair.re, pair.im);
    unsafe { v128_store(values.as_mut_ptr().add(index).cast(), raw) };
}

#[inline(always)]
fn add_pair(a: ComplexPair, b: ComplexPair) -> ComplexPair {
    ComplexPair {
        re: f32x4_add(a.re, b.re),
        im: f32x4_add(a.im, b.im),
    }
}

#[inline(always)]
fn sub_pair(a: ComplexPair, b: ComplexPair) -> ComplexPair {
    ComplexPair {
        re: f32x4_sub(a.re, b.re),
        im: f32x4_sub(a.im, b.im),
    }
}

#[inline(always)]
fn scale_pair(a: ComplexPair, scale: f32) -> ComplexPair {
    let scale = f32x4_splat(scale);
    ComplexPair {
        re: f32x4_mul(a.re, scale),
        im: f32x4_mul(a.im, scale),
    }
}

#[inline(always)]
fn mul_pair(a: ComplexPair, b: ComplexPair) -> ComplexPair {
    ComplexPair {
        re: f32x4_sub(f32x4_mul(a.re, b.re), f32x4_mul(a.im, b.im)),
        im: f32x4_add(f32x4_mul(a.re, b.im), f32x4_mul(a.im, b.re)),
    }
}

#[inline(always)]
fn rotate_i_pair(a: ComplexPair, dir_sign: f32) -> ComplexPair {
    // i * (re + i im) = -im + i re, with the DFT direction applied.
    scale_pair(
        ComplexPair {
            re: f32x4_neg(a.im),
            im: a.re,
        },
        dir_sign,
    )
}

#[inline(always)]
unsafe fn scalar_tail_radix2(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    base_in: usize,
    base_out: usize,
    l: usize,
) {
    let tw = stage_tw.k_slice(1)[l];
    let a = src[base_in + l];
    let b = src[base_in + l + stride] * tw;
    let (y0, y1) = radix2(a, b);
    dst[base_out + l] = y0;
    dst[base_out + stage_tw.m + l] = y1;
}

#[inline(always)]
unsafe fn scalar_tail_radix4(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    dir_sign: f32,
    base_in: usize,
    base_out: usize,
    l: usize,
) {
    let tw1 = stage_tw.k_slice(1)[l];
    let tw2 = stage_tw.k_slice(2)[l];
    let tw3 = stage_tw.k_slice(3)[l];
    let x0 = src[base_in + l];
    let x1 = src[base_in + l + stride] * tw1;
    let x2 = src[base_in + l + 2 * stride] * tw2;
    let x3 = src[base_in + l + 3 * stride] * tw3;
    let (y0, y1, y2, y3) = radix4(x0, x1, x2, x3, dir_sign);
    dst[base_out + l] = y0;
    dst[base_out + stage_tw.m + l] = y1;
    dst[base_out + 2 * stage_tw.m + l] = y2;
    dst[base_out + 3 * stage_tw.m + l] = y3;
}

/// SIMD radix-2 Stockham stage for `Complex<f32>`.
///
/// # Safety
///
/// The function must only be called on a `wasm32` target with `simd128`.
/// The slices and stage metadata must satisfy the same dimensions as the
/// generic Stockham stage.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn stage_radix2_f32(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
) {
    stage_tw.debug_assert_stage_shape(2, stride, src.len(), dst.len());
    let m = stage_tw.m;
    let m_new = m * 2;
    for g in 0..stage_tw.s {
        let base_in = g * m;
        let base_out = g * m_new;
        let mut l = 0;
        while l + 1 < m {
            let a = unsafe { load_pair(src, base_in + l) };
            let b = mul_pair(unsafe { load_pair(src, base_in + l + stride) }, unsafe {
                load_pair(stage_tw.k_slice(1), l)
            });
            unsafe {
                store_pair(dst, base_out + l, add_pair(a, b));
                store_pair(dst, base_out + m + l, sub_pair(a, b));
            }
            l += 2;
        }
        if l < m {
            unsafe {
                scalar_tail_radix2(stride, src, dst, stage_tw, base_in, base_out, l);
            }
        }
    }
}

/// SIMD radix-4 Stockham stage for `Complex<f32>`.
///
/// # Safety
///
/// The function must only be called on a `wasm32` target with `simd128`.
/// The slices and stage metadata must satisfy the same dimensions as the
/// generic Stockham stage.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn stage_radix4_f32(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    stage_tw.debug_assert_stage_shape(4, stride, src.len(), dst.len());
    let m = stage_tw.m;
    let m_new = m * 4;
    for g in 0..stage_tw.s {
        let base_in = g * m;
        let base_out = g * m_new;
        let mut l = 0;
        while l + 1 < m {
            let x0 = unsafe { load_pair(src, base_in + l) };
            let x1 = mul_pair(unsafe { load_pair(src, base_in + l + stride) }, unsafe {
                load_pair(stage_tw.k_slice(1), l)
            });
            let x2 = mul_pair(
                unsafe { load_pair(src, base_in + l + 2 * stride) },
                unsafe { load_pair(stage_tw.k_slice(2), l) },
            );
            let x3 = mul_pair(
                unsafe { load_pair(src, base_in + l + 3 * stride) },
                unsafe { load_pair(stage_tw.k_slice(3), l) },
            );

            let a = add_pair(x0, x2);
            let b = sub_pair(x0, x2);
            let c = add_pair(x1, x3);
            let d = sub_pair(x1, x3);
            let id = rotate_i_pair(d, dir_sign);
            let y0 = add_pair(a, c);
            let y1 = add_pair(b, id);
            let y2 = sub_pair(a, c);
            let y3 = sub_pair(b, id);
            unsafe {
                store_pair(dst, base_out + l, y0);
                store_pair(dst, base_out + m + l, y1);
                store_pair(dst, base_out + 2 * m + l, y2);
                store_pair(dst, base_out + 3 * m + l, y3);
            }
            l += 2;
        }
        if l < m {
            unsafe {
                scalar_tail_radix4(stride, src, dst, stage_tw, dir_sign, base_in, base_out, l);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    #[test]
    fn wasm_radix_stages_match_scalar_stockham_stage() {
        for &(n, radix) in &[(8usize, 2usize), (16, 4)] {
            let radices = if radix == 2 {
                vec![2, 2, 2]
            } else {
                vec![4, 4]
            };
            let stages = crate::twiddles::build_stage_twiddles::<f32>(
                n,
                &radices,
                crate::FftDirection::Forward,
            );
            let input: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new(i as f32 * 0.13 - 0.4, (i as f32 * 0.21).cos()))
                .collect();
            let stage = &stages[0];
            let stride = n / radix;
            let mut actual = vec![Complex32::new(0.0, 0.0); n];
            unsafe {
                if radix == 2 {
                    stage_radix2_f32(stride, &input, &mut actual, stage);
                } else {
                    stage_radix4_f32(stride, &input, &mut actual, stage, -1.0);
                }
            }

            let mut expected = input.clone();
            let mut scratch = vec![Complex32::new(0.0, 0.0); n];
            crate::stockham::run_single(
                std::slice::from_ref(stage),
                -1.0,
                &mut expected,
                &mut scratch,
            );
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!((actual - expected).norm() <= 2.0e-5);
            }
        }
    }
}
