//! wasm32 `simd128` f32 sine/cosine kernel.
//!
//! WebAssembly SIMD has no elementary sine/cosine instruction. This module
//! therefore uses the same Cody-Waite shape as the scalar implementation,
//! with f32 constants and a vector Horner evaluation on the reduced interval.
//! Non-finite values and phases outside the f32 fast-path contract fall back
//! lane-wise to the normative scalar implementation.

use std::arch::wasm32::*;

use crate::coeffs::{
    COS_C0, COS_C1, COS_C2, COS_C3, COS_C4, COS_C5, COS_C6, SIN_S0, SIN_S1, SIN_S2, SIN_S3, SIN_S4,
    SIN_S5, SIN_S6, SIN_S7, SIN_S8,
};
use crate::sincos_f32;

const PI_OVER_2_HI_F32: f32 = 1.570_796_251_3;
const PI_OVER_2_LO_F32: f32 = 7.549_789_4e-8;
const TWO_OVER_PI_F32: f32 = 0.636_619_75;

#[inline(always)]
fn lanes(x: v128) -> [f32; 4] {
    [
        f32x4_extract_lane::<0>(x),
        f32x4_extract_lane::<1>(x),
        f32x4_extract_lane::<2>(x),
        f32x4_extract_lane::<3>(x),
    ]
}

#[inline(always)]
fn from_lanes(values: [f32; 4]) -> v128 {
    let mut result = f32x4_splat(values[0]);
    result = f32x4_replace_lane::<1>(result, values[1]);
    result = f32x4_replace_lane::<2>(result, values[2]);
    f32x4_replace_lane::<3>(result, values[3])
}

#[inline(always)]
fn horner(x: v128, coefficients: &[f64]) -> v128 {
    let mut result = f32x4_splat(coefficients[0] as f32);
    for &coefficient in &coefficients[1..] {
        result = f32x4_add(f32x4_mul(result, x), f32x4_splat(coefficient as f32));
    }
    result
}

/// Lane-parallel f32 sine/cosine for wasm `simd128`.
///
/// # Safety
///
/// The caller must execute on a wasm target with the `simd128` target feature.
#[target_feature(enable = "simd128")]
pub unsafe fn sincos_f32x4(x: v128) -> (v128, v128) {
    let input = lanes(x);
    // Zero lanes take the scalar path: the vector reduction computes
    // `(-0.0) - (-0.0)·HI = +0.0` and would lose the sine's signed zero.
    if input
        .iter()
        .all(|&value| value != 0.0 && value.is_finite() && value.abs() <= 1_000.0)
    {
        // SAFETY: same simd128 contract as the entry point.
        return unsafe { sincos_f32x4_vector(x, input) };
    }
    // Mixed batch: run the vector kernel for every lane, then overwrite
    // only the fallback lanes with scalar results.
    // SAFETY: same simd128 contract as the entry point.
    let (vec_sin, vec_cos) = unsafe { sincos_f32x4_vector(x, input) };
    let mut sins = lanes(vec_sin);
    let mut coss = lanes(vec_cos);
    for lane in 0..4 {
        if input[lane] == 0.0 || !input[lane].is_finite() || input[lane].abs() > 1_000.0 {
            (sins[lane], coss[lane]) = sincos_f32(input[lane]);
        }
    }
    (from_lanes(sins), from_lanes(coss))
}

/// Vector kernel for [`sincos_f32x4`]: caller guarantees every lane holds a
/// non-zero finite value within `±1000`, or discards that lane's output.
///
/// # Safety
///
/// The caller must execute on a wasm target with the `simd128` target feature.
#[target_feature(enable = "simd128")]
unsafe fn sincos_f32x4_vector(x: v128, input: [f32; 4]) -> (v128, v128) {
    // Compute the integer quadrant scalar-wise, then keep range reduction and
    // both polynomial evaluations in SIMD registers. f32's split constants
    // retain useful low bits over the complete fast-path interval.
    let k = input.map(|value| (value * TWO_OVER_PI_F32).round());
    let k_v = from_lanes(k);
    let r = f32x4_sub(
        f32x4_sub(x, f32x4_mul(k_v, f32x4_splat(PI_OVER_2_HI_F32))),
        f32x4_mul(k_v, f32x4_splat(PI_OVER_2_LO_F32)),
    );
    let s = f32x4_mul(r, r);
    let sin_poly = horner(
        s,
        &[
            SIN_S0, SIN_S1, SIN_S2, SIN_S3, SIN_S4, SIN_S5, SIN_S6, SIN_S7, SIN_S8,
        ],
    );
    let sin_reduced = f32x4_add(r, f32x4_mul(f32x4_mul(r, s), sin_poly));
    let cos_poly = horner(s, &[COS_C0, COS_C1, COS_C2, COS_C3, COS_C4, COS_C5, COS_C6]);
    let cos_reduced = f32x4_add(f32x4_splat(1.0), f32x4_mul(s, cos_poly));

    // Applying the four quadrant choices scalar-wise is deliberately kept at
    // the boundary. The expensive range reduction and polynomial work above
    // is lane-parallel, while this preserves exact signed-zero behaviour at
    // the quadrant boundaries.
    let sin_values = lanes(sin_reduced);
    let cos_values = lanes(cos_reduced);
    let mut sins = [0.0; 4];
    let mut coss = [0.0; 4];
    for lane in 0..4 {
        match (k[lane] as i32) & 3 {
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
    (from_lanes(sins), from_lanes(coss))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_simd_matches_scalar_across_reduction_edges() {
        let input = [
            -1000.0_f32,
            -31.0,
            -std::f32::consts::FRAC_PI_2,
            -0.0,
            0.0,
            0.25,
            31.0,
            1000.0,
        ];
        let (sin_v, cos_v) =
            unsafe { sincos_f32x4(from_lanes([input[0], input[1], input[2], input[3]])) };
        let sins = lanes(sin_v);
        let coss = lanes(cos_v);
        for (index, &x) in input[..4].iter().enumerate() {
            let (sin, cos) = x.sin_cos();
            assert!((sins[index] - sin).abs() <= 4.0e-6, "sin lane {index}");
            assert!((coss[index] - cos).abs() <= 4.0e-6, "cos lane {index}");
        }

        let (sin_v, cos_v) =
            unsafe { sincos_f32x4(from_lanes([input[4], input[5], input[6], input[7]])) };
        let sins = lanes(sin_v);
        let coss = lanes(cos_v);
        for (index, &x) in input[4..].iter().enumerate() {
            let (sin, cos) = x.sin_cos();
            assert!(
                (sins[index] - sin).abs() <= 4.0e-6,
                "sin lane {}",
                index + 4
            );
            assert!(
                (coss[index] - cos).abs() <= 4.0e-6,
                "cos lane {}",
                index + 4
            );
        }
    }
}
