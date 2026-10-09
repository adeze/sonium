//! AVX-512 f64 sine/cosine kernel.
//!
//! Valid Cody-Waite lanes use vector range-reduced polynomial evaluation.
//! Non-finite values, signed zero, and phases outside the documented fast
//! domain are evaluated through the scalar Payne–Hanek/reference paths so the
//! public contract remains identical across dispatches.

use std::arch::x86_64::*;

use crate::coeffs::{
    COS_C0, COS_C1, COS_C2, COS_C3, COS_C4, COS_C5, COS_C6, PI_OVER_2_HI, PI_OVER_2_LO, SIN_S0,
    SIN_S1, SIN_S2, SIN_S3, SIN_S4, SIN_S5, SIN_S6, SIN_S7, SIN_S8, TWO_OVER_PI,
};
use crate::scalar::{CODY_WAITE_LIMIT, sincos_scalar};

#[allow(clippy::excessive_precision)]
const PI_OVER_2_HI_F32: f32 = 1.570_796_251_3;
const PI_OVER_2_LO_F32: f32 = 7.549_789_4e-8;
const TWO_OVER_PI_F32: f32 = 0.636_619_75;

#[target_feature(enable = "avx512f")]
unsafe fn poly_sin(r: __m512d, s: __m512d) -> __m512d {
    let mut u = _mm512_set1_pd(SIN_S0);
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S1));
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S2));
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S3));
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S4));
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S5));
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S6));
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S7));
    u = _mm512_fmadd_pd(u, s, _mm512_set1_pd(SIN_S8));
    _mm512_fmadd_pd(_mm512_mul_pd(r, s), u, r)
}

#[target_feature(enable = "avx512f")]
unsafe fn poly_cos(s: __m512d) -> __m512d {
    let mut v = _mm512_set1_pd(COS_C0);
    v = _mm512_fmadd_pd(v, s, _mm512_set1_pd(COS_C1));
    v = _mm512_fmadd_pd(v, s, _mm512_set1_pd(COS_C2));
    v = _mm512_fmadd_pd(v, s, _mm512_set1_pd(COS_C3));
    v = _mm512_fmadd_pd(v, s, _mm512_set1_pd(COS_C4));
    v = _mm512_fmadd_pd(v, s, _mm512_set1_pd(COS_C5));
    v = _mm512_fmadd_pd(v, s, _mm512_set1_pd(COS_C6));
    _mm512_fmadd_pd(s, v, _mm512_set1_pd(1.0))
}

/// Lane-parallel `(sin, cos)` for eight f64 values.
///
/// # Safety
///
/// The caller must provide AVX-512F at runtime.
#[target_feature(enable = "avx512f")]
pub unsafe fn sincos_f64x8(x: __m512d) -> (__m512d, __m512d) {
    let mut input = [0.0_f64; 8];
    unsafe { _mm512_storeu_pd(input.as_mut_ptr(), x) };

    if input
        .iter()
        .all(|&value| value != 0.0 && value.is_finite() && value.abs() <= CODY_WAITE_LIMIT)
    {
        return unsafe { sincos_f64x8_vector(x) };
    }
    // Mixed batch: the vector kernel still runs for every lane — its output
    // on fallback lanes is well-defined FP arithmetic (no traps under the
    // default MXCSR mask) but discarded — then only those lanes are
    // overwritten with scalar results.
    let (vec_sin, vec_cos) = unsafe { sincos_f64x8_vector(x) };
    let mut sins = [0.0_f64; 8];
    let mut coss = [0.0_f64; 8];
    unsafe {
        _mm512_storeu_pd(sins.as_mut_ptr(), vec_sin);
        _mm512_storeu_pd(coss.as_mut_ptr(), vec_cos);
    }
    for (i, &value) in input.iter().enumerate() {
        if value == 0.0 || !value.is_finite() || value.abs() > CODY_WAITE_LIMIT {
            // Scalar path preserves NaN and signed-zero semantics.
            (sins[i], coss[i]) = sincos_scalar(value);
        }
    }
    unsafe {
        (
            _mm512_loadu_pd(sins.as_ptr()),
            _mm512_loadu_pd(coss.as_ptr()),
        )
    }
}

/// Vector kernel for [`sincos_f64x8`]: caller guarantees every lane holds a
/// non-zero finite value within [`CODY_WAITE_LIMIT`], or discards that
/// lane's output.
///
/// # Safety
///
/// The caller must provide AVX-512F at runtime.
#[target_feature(enable = "avx512f")]
unsafe fn sincos_f64x8_vector(x: __m512d) -> (__m512d, __m512d) {
    let mut input = [0.0_f64; 8];
    unsafe { _mm512_storeu_pd(input.as_mut_ptr(), x) };

    // The integer quadrant is formed scalar-wise because AVX-512F alone is
    // deliberately sufficient for this module; integer conversion and mask
    // extensions from AVX-512DQ/VL are not required. Polynomial work remains
    // fully vectorized across all eight valid lanes.
    let mut k_f = [0.0_f64; 8];
    for (k, &value) in k_f.iter_mut().zip(input.iter()) {
        *k = (value * TWO_OVER_PI).round();
    }
    let k_v = unsafe { _mm512_loadu_pd(k_f.as_ptr()) };
    let x_hi = _mm512_mul_pd(k_v, _mm512_set1_pd(PI_OVER_2_HI));
    let x_lo = _mm512_mul_pd(k_v, _mm512_set1_pd(PI_OVER_2_LO));
    let r = _mm512_sub_pd(_mm512_sub_pd(x, x_hi), x_lo);
    let s = _mm512_mul_pd(r, r);
    let sin_reduced = unsafe { poly_sin(r, s) };
    let cos_reduced = unsafe { poly_cos(s) };

    let mut sin_values = [0.0_f64; 8];
    let mut cos_values = [0.0_f64; 8];
    unsafe {
        _mm512_storeu_pd(sin_values.as_mut_ptr(), sin_reduced);
        _mm512_storeu_pd(cos_values.as_mut_ptr(), cos_reduced);
    }
    let mut sins = [0.0_f64; 8];
    let mut coss = [0.0_f64; 8];
    for lane in 0..8 {
        match (k_f[lane] as i64) & 3 {
            0 => {
                sins[lane] = sin_values[lane];
                coss[lane] = cos_values[lane];
            }
            1 => {
                sins[lane] = cos_values[lane];
                coss[lane] = -sin_values[lane];
            }
            2 => {
                sins[lane] = -sin_values[lane];
                coss[lane] = -cos_values[lane];
            }
            _ => {
                sins[lane] = -cos_values[lane];
                coss[lane] = sin_values[lane];
            }
        }
    }

    unsafe {
        (
            _mm512_loadu_pd(sins.as_ptr()),
            _mm512_loadu_pd(coss.as_ptr()),
        )
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn poly_sin_f32(r: __m512, s: __m512) -> __m512 {
    let mut u = _mm512_set1_ps(SIN_S0 as f32);
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S1 as f32));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S2 as f32));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S3 as f32));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S4 as f32));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S5 as f32));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S6 as f32));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S7 as f32));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(SIN_S8 as f32));
    _mm512_fmadd_ps(_mm512_mul_ps(r, s), u, r)
}

#[target_feature(enable = "avx512f")]
unsafe fn poly_cos_f32(s: __m512) -> __m512 {
    let mut v = _mm512_set1_ps(COS_C0 as f32);
    v = _mm512_fmadd_ps(v, s, _mm512_set1_ps(COS_C1 as f32));
    v = _mm512_fmadd_ps(v, s, _mm512_set1_ps(COS_C2 as f32));
    v = _mm512_fmadd_ps(v, s, _mm512_set1_ps(COS_C3 as f32));
    v = _mm512_fmadd_ps(v, s, _mm512_set1_ps(COS_C4 as f32));
    v = _mm512_fmadd_ps(v, s, _mm512_set1_ps(COS_C5 as f32));
    v = _mm512_fmadd_ps(v, s, _mm512_set1_ps(COS_C6 as f32));
    _mm512_fmadd_ps(s, v, _mm512_set1_ps(1.0))
}

/// Lane-parallel `(sin, cos)` for sixteen f32 values.
///
/// # Safety
///
/// The caller must provide AVX-512F at runtime.
#[target_feature(enable = "avx512f")]
pub unsafe fn sincos_f32x16(x: __m512) -> (__m512, __m512) {
    let mut input = [0.0_f32; 16];
    unsafe { _mm512_storeu_ps(input.as_mut_ptr(), x) };
    if input
        .iter()
        .all(|&value| value != 0.0 && value.is_finite() && value.abs() <= 1_000.0)
    {
        return unsafe { sincos_f32x16_vector(x) };
    }
    // Mixed batch: run the vector kernel for every lane, then overwrite
    // only the fallback lanes (zero, non-finite, out-of-range) with scalar
    // results so one outlier no longer forces all 16 lanes scalar.
    let (vec_sin, vec_cos) = unsafe { sincos_f32x16_vector(x) };
    let mut sins = [0.0_f32; 16];
    let mut coss = [0.0_f32; 16];
    unsafe {
        _mm512_storeu_ps(sins.as_mut_ptr(), vec_sin);
        _mm512_storeu_ps(coss.as_mut_ptr(), vec_cos);
    }
    for (i, &value) in input.iter().enumerate() {
        if value == 0.0 || !value.is_finite() || value.abs() > 1_000.0 {
            (sins[i], coss[i]) = crate::sincos_f32(value);
        }
    }
    unsafe {
        (
            _mm512_loadu_ps(sins.as_ptr()),
            _mm512_loadu_ps(coss.as_ptr()),
        )
    }
}

/// Vector kernel for [`sincos_f32x16`]: caller guarantees every lane holds
/// a non-zero finite value within `±1000`, or discards that lane's output.
///
/// # Safety
///
/// The caller must provide AVX-512F at runtime.
#[target_feature(enable = "avx512f")]
unsafe fn sincos_f32x16_vector(x: __m512) -> (__m512, __m512) {
    let mut input = [0.0_f32; 16];
    unsafe { _mm512_storeu_ps(input.as_mut_ptr(), x) };

    let mut k_f = [0.0_f32; 16];
    for (k, &value) in k_f.iter_mut().zip(input.iter()) {
        *k = (value * TWO_OVER_PI_F32).round();
    }
    let k_v = unsafe { _mm512_loadu_ps(k_f.as_ptr()) };
    let r = _mm512_sub_ps(
        _mm512_sub_ps(x, _mm512_mul_ps(k_v, _mm512_set1_ps(PI_OVER_2_HI_F32))),
        _mm512_mul_ps(k_v, _mm512_set1_ps(PI_OVER_2_LO_F32)),
    );
    let s = _mm512_mul_ps(r, r);
    let sin_reduced = unsafe { poly_sin_f32(r, s) };
    let cos_reduced = unsafe { poly_cos_f32(s) };
    let mut sin_values = [0.0_f32; 16];
    let mut cos_values = [0.0_f32; 16];
    unsafe {
        _mm512_storeu_ps(sin_values.as_mut_ptr(), sin_reduced);
        _mm512_storeu_ps(cos_values.as_mut_ptr(), cos_reduced);
    }
    let mut sins = [0.0_f32; 16];
    let mut coss = [0.0_f32; 16];
    for lane in 0..16 {
        match (k_f[lane] as i32) & 3 {
            0 => {
                sins[lane] = sin_values[lane];
                coss[lane] = cos_values[lane];
            }
            1 => {
                sins[lane] = cos_values[lane];
                coss[lane] = -sin_values[lane];
            }
            2 => {
                sins[lane] = -sin_values[lane];
                coss[lane] = -cos_values[lane];
            }
            _ => {
                sins[lane] = -cos_values[lane];
                coss[lane] = sin_values[lane];
            }
        }
    }
    unsafe {
        (
            _mm512_loadu_ps(sins.as_ptr()),
            _mm512_loadu_ps(coss.as_ptr()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordered_bits(value: f64) -> i64 {
        let bits = value.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    }

    fn ulp_distance(a: f64, b: f64) -> u64 {
        (ordered_bits(a) - ordered_bits(b)).unsigned_abs()
    }

    #[test]
    fn avx512_parity_when_available() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let input = [-1000.0, -3.0, -0.25, 0.25, 1.0, 3.0, 1000.0, 999.0];
        unsafe {
            let x = _mm512_loadu_pd(input.as_ptr());
            let (sv, cv) = sincos_f64x8(x);
            let mut sins = [0.0; 8];
            let mut coss = [0.0; 8];
            _mm512_storeu_pd(sins.as_mut_ptr(), sv);
            _mm512_storeu_pd(coss.as_mut_ptr(), cv);
            for i in 0..8 {
                let (sin, cos) = sincos_scalar(input[i]);
                assert!(ulp_distance(sins[i], sin) <= 4, "sin lane {i}");
                assert!(ulp_distance(coss[i], cos) <= 4, "cos lane {i}");
            }
        }
    }

    #[test]
    fn avx512_f32_parity_when_available() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let input: [f32; 16] = [
            -1000.0, -31.0, -3.0, -0.25, -0.01, 0.01, 0.25, 1.0, 2.0, 3.0, 10.0, 31.0, 100.0,
            512.0, 999.0, 1000.0,
        ];
        unsafe {
            let x = _mm512_loadu_ps(input.as_ptr());
            let (sv, cv) = sincos_f32x16(x);
            let mut sins = [0.0; 16];
            let mut coss = [0.0; 16];
            _mm512_storeu_ps(sins.as_mut_ptr(), sv);
            _mm512_storeu_ps(coss.as_mut_ptr(), cv);
            for i in 0..16 {
                let (sin, cos) = input[i].sin_cos();
                assert!((sins[i] - sin).abs() <= 4.0e-6, "sin lane {i}");
                assert!((coss[i] - cos).abs() <= 4.0e-6, "cos lane {i}");
            }
        }
    }
}
