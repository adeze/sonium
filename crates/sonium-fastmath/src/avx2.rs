//! AVX2 + FMA f64x4 implementation of [`sincos_f64x4`]. x86_64-only.
//!
//! Same algorithm as [`crate::scalar::sincos_scalar`] and
//! [`crate::neon::sincos_f64x2`]: two-term Cody–Waite range reduction,
//! Sleef u10 minimax polynomials in Horner form (FMA-fused), quadrant
//! select via integer masks reinterpreted as `__m256d`. The vector kernel
//! runs all 4 lanes lane-parallel; lanes outside its domain (zero,
//! non-finite, `|x| > CODY_WAITE_LIMIT`) are blended per lane with scalar
//! results so one outlier no longer forces the whole batch scalar.

use std::arch::x86_64::*;

use crate::coeffs::{
    COS_C0, COS_C1, COS_C2, COS_C3, COS_C4, COS_C5, COS_C6, PI_OVER_2_HI, PI_OVER_2_LO, SIN_S0,
    SIN_S1, SIN_S2, SIN_S3, SIN_S4, SIN_S5, SIN_S6, SIN_S7, SIN_S8, TWO_OVER_PI,
};
use crate::scalar::{CODY_WAITE_LIMIT, sincos_scalar};

/// Lane-parallel `(sin(x), cos(x))` for four f64 inputs. For finite lanes
/// within [`CODY_WAITE_LIMIT`], the fast path follows the scalar contract;
/// larger finite lanes use the scalar fallback (three-term mid-range
/// reduction, then Payne–Hanek); non-finite lanes use `f64::sin_cos` so
/// their quadrants and special-value semantics are preserved.
///
/// # Safety
///
/// - The caller must be running on x86_64 with both AVX2 and FMA. The CPU
///   dispatch (runtime feature detection via `is_x86_feature_detected!`) is
///   the caller's responsibility; this function carries the `target_feature`
///   attribute but performs no in-function check.
/// - `x` must be a valid `__m256d` value produced by the x86_64 ABI. The
///   lanes may contain any IEEE-754 values; non-finite and out-of-domain lanes
///   are deliberately handled by the scalar fallback.
///
/// The function does not dereference caller-provided memory and does not
/// require alignment beyond the value ABI. Public slice dispatch uses
/// unaligned loads only after checking AVX2+FMA at runtime.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sincos_f64x4(x: __m256d) -> (__m256d, __m256d) {
    // SAFETY: the body runs inside a `target_feature = "avx2,fma"`
    // function. Every intrinsic is sound under the caller's contract.
    unsafe {
        // Keep the vector kernel on its documented domain. Lanes needing
        // the robust scalar fallback (zero, non-finite, out-of-range) are
        // blended per lane so ordinary lanes keep vector throughput.
        let mut inputs = [0.0_f64; 4];
        _mm256_storeu_pd(inputs.as_mut_ptr(), x);
        if inputs
            .iter()
            .all(|&value| value != 0.0 && value.is_finite() && value.abs() <= CODY_WAITE_LIMIT)
        {
            return sincos_f64x4_vector(x);
        }
        // Mixed batch: the vector kernel still runs for every lane — its
        // outputs on fallback lanes are well-defined (finite or NaN/Inf FP
        // arithmetic, no traps under the default MXCSR mask) but discarded.
        let (vec_sin, vec_cos) = sincos_f64x4_vector(x);
        let mut sins = [0.0_f64; 4];
        let mut coss = [0.0_f64; 4];
        _mm256_storeu_pd(sins.as_mut_ptr(), vec_sin);
        _mm256_storeu_pd(coss.as_mut_ptr(), vec_cos);
        for (i, &value) in inputs.iter().enumerate() {
            if value == 0.0 || !value.is_finite() || value.abs() > CODY_WAITE_LIMIT {
                // Scalar path preserves NaN and signed-zero semantics.
                (sins[i], coss[i]) = sincos_scalar(value);
            }
        }
        (
            _mm256_loadu_pd(sins.as_ptr()),
            _mm256_loadu_pd(coss.as_ptr()),
        )
    }
}

/// Vector kernel for [`sincos_f64x4`]: caller guarantees every lane is a
/// non-zero finite value within [`CODY_WAITE_LIMIT`], or discards that
/// lane's output.
///
/// # Safety
///
/// Same contract as [`sincos_f64x4`]: AVX2 + FMA, valid `__m256d`.
#[target_feature(enable = "avx2,fma")]
unsafe fn sincos_f64x4_vector(x: __m256d) -> (__m256d, __m256d) {
    // SAFETY: same intrinsic soundness argument as the entry point.
    unsafe {
        // 1. Range reduce: x = k · π/2 + r, r ∈ [-π/4, π/4]
        let two_over_pi = _mm256_set1_pd(TWO_OVER_PI);
        // Round-to-nearest-even, suppress exceptions. AVX rounding mode 0
        // = nearest with ties-to-even when combined with _MM_FROUND_NO_EXC.
        let k_f = _mm256_round_pd(
            _mm256_mul_pd(x, two_over_pi),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        // f64 → i32 (truncating-cast is OK because k_f is already integer-valued
        // after _mm256_round_pd). Returns __m128i with 4 i32 lanes.
        let k_i32 = _mm256_cvtpd_epi32(k_f);

        // Two-term Cody–Waite: r = ((x - k·HI) - k·LO).
        // _mm256_fnmadd_pd(a, b, c) = -a*b + c = c - a*b.
        let pi_hi = _mm256_set1_pd(PI_OVER_2_HI);
        let pi_lo = _mm256_set1_pd(PI_OVER_2_LO);
        let r = _mm256_fnmadd_pd(k_f, pi_hi, x);
        let r = _mm256_fnmadd_pd(k_f, pi_lo, r);
        let s = _mm256_mul_pd(r, r);

        // 2. Polynomial sin(r), cos(r) using the Sleef u10 coefficients.
        let sin_r = poly_sin(r, s);
        let cos_r = poly_cos(s);

        // 3. Quadrant select on k mod 4.
        //    Integer-side: q_i32 = k_i32 & 3. Build 4 boolean masks in i32,
        //    sign-extend each to i64, reinterpret as __m256d for blendv.
        let three_i32 = _mm_set1_epi32(3);
        let one_i32 = _mm_set1_epi32(1);
        let two_i32 = _mm_set1_epi32(2);

        let q = _mm_and_si128(k_i32, three_i32);

        // swap = (q & 1) == 1
        let swap_i32 = _mm_cmpeq_epi32(_mm_and_si128(q, one_i32), one_i32);
        let swap_i64 = _mm256_cvtepi32_epi64(swap_i32);
        let swap_pd = _mm256_castsi256_pd(swap_i64);

        // neg_sin = (q & 2) != 0  i.e. (q & 2) == 2
        let neg_sin_i32 = _mm_cmpeq_epi32(_mm_and_si128(q, two_i32), two_i32);
        let neg_sin_i64 = _mm256_cvtepi32_epi64(neg_sin_i32);
        let neg_sin_pd = _mm256_castsi256_pd(neg_sin_i64);

        // neg_cos = ((q + 1) & 2) != 0  i.e. ((q+1) & 2) == 2
        let q_plus_one = _mm_add_epi32(q, one_i32);
        let neg_cos_i32 = _mm_cmpeq_epi32(_mm_and_si128(q_plus_one, two_i32), two_i32);
        let neg_cos_i64 = _mm256_cvtepi32_epi64(neg_cos_i32);
        let neg_cos_pd = _mm256_castsi256_pd(neg_cos_i64);

        // blendv_pd: select b where mask high bit is 1, a otherwise.
        // swap_pd has lanes that are all-1s (true) or all-0s (false), so
        // the high bit drives the blend correctly.
        let sin_out = _mm256_blendv_pd(sin_r, cos_r, swap_pd);
        let cos_out = _mm256_blendv_pd(cos_r, sin_r, swap_pd);

        // Sign flip via XOR against the sign bit pattern of f64.
        let sign_bit = _mm256_castsi256_pd(_mm256_set1_epi64x(i64::MIN));
        let sin_neg = _mm256_xor_pd(sin_out, sign_bit);
        let cos_neg = _mm256_xor_pd(cos_out, sign_bit);
        let sin_final = _mm256_blendv_pd(sin_out, sin_neg, neg_sin_pd);
        let cos_final = _mm256_blendv_pd(cos_out, cos_neg, neg_cos_pd);

        (sin_final, cos_final)
    }
}

/// Lane-parallel f32 sine/cosine for eight values.
///
/// The f32 lanes are widened in two AVX2 groups, evaluated through the
/// normative f64 kernel, and narrowed once. This is a deterministic SIMD f32
/// path until a dedicated f32 polynomial is introduced.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. This function performs no
/// runtime feature check itself; public slice dispatch checks both features
/// before calling it. The input lanes may contain non-finite values because
/// the widened f64 kernel preserves the scalar fallback contract.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sincos_f32x8(x: __m256) -> (__m256, __m256) {
    unsafe {
        let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
        let (sin_lo, cos_lo) = sincos_f64x4(lo);
        let (sin_hi, cos_hi) = sincos_f64x4(hi);
        let sin_lo = _mm256_cvtpd_ps(sin_lo);
        let sin_hi = _mm256_cvtpd_ps(sin_hi);
        let cos_lo = _mm256_cvtpd_ps(cos_lo);
        let cos_hi = _mm256_cvtpd_ps(cos_hi);
        (
            _mm256_insertf128_ps(_mm256_castps128_ps256(sin_lo), sin_hi, 1),
            _mm256_insertf128_ps(_mm256_castps128_ps256(cos_lo), cos_hi, 1),
        )
    }
}

/// SIMD Horner for the Sleef u10 sin polynomial: `sin(r) = r + r·s·U(s)`,
/// where `s = r²`. FMA-fused so each line is one VFMADD231PD.
#[inline]
unsafe fn poly_sin(r: __m256d, s: __m256d) -> __m256d {
    unsafe {
        let mut u = _mm256_set1_pd(SIN_S0);
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S1));
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S2));
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S3));
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S4));
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S5));
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S6));
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S7));
        u = _mm256_fmadd_pd(u, s, _mm256_set1_pd(SIN_S8));
        // sin(r) = r + (r·s) · u
        _mm256_fmadd_pd(_mm256_mul_pd(r, s), u, r)
    }
}

/// SIMD Horner for the Sleef u10 cos polynomial: `cos(r) = 1 + s·V(s)`.
#[inline]
unsafe fn poly_cos(s: __m256d) -> __m256d {
    unsafe {
        let mut v = _mm256_set1_pd(COS_C0);
        v = _mm256_fmadd_pd(v, s, _mm256_set1_pd(COS_C1));
        v = _mm256_fmadd_pd(v, s, _mm256_set1_pd(COS_C2));
        v = _mm256_fmadd_pd(v, s, _mm256_set1_pd(COS_C3));
        v = _mm256_fmadd_pd(v, s, _mm256_set1_pd(COS_C4));
        v = _mm256_fmadd_pd(v, s, _mm256_set1_pd(COS_C5));
        v = _mm256_fmadd_pd(v, s, _mm256_set1_pd(COS_C6));
        // cos(r) = 1 + s · v
        _mm256_fmadd_pd(s, v, _mm256_set1_pd(1.0))
    }
}

#[cfg(all(test, target_feature = "avx2", target_feature = "fma"))]
mod tests {
    use super::*;
    use crate::scalar::sincos_scalar;

    #[inline]
    unsafe fn pack(a: f64, b: f64, c: f64, d: f64) -> __m256d {
        // _mm256_set_pd packs in reverse: lane 0 = last arg.
        unsafe { _mm256_set_pd(d, c, b, a) }
    }

    fn extract(v: __m256d) -> [f64; 4] {
        let mut out = [0.0_f64; 4];
        // SAFETY: 4×f64 aligned-store, target is well-typed.
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), v) };
        out
    }

    #[test]
    fn avx2_matches_scalar_at_known_points() {
        use std::f64::consts::PI;
        let xs = [
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 3.0,
            PI / 2.0,
            PI,
            -PI / 2.0,
            3.0 * PI / 2.0,
            10.0 * PI,
            -10.0 * PI,
            1e-12,
            -1e-12,
        ];
        for chunk in xs.chunks(4) {
            let a = chunk[0];
            let b = *chunk.get(1).unwrap_or(&a);
            let c = *chunk.get(2).unwrap_or(&a);
            let d = *chunk.get(3).unwrap_or(&a);
            let xv = unsafe { pack(a, b, c, d) };
            let (sv, cv) = unsafe { sincos_f64x4(xv) };
            let s = extract(sv);
            let cc = extract(cv);
            for (i, &xi) in [a, b, c, d].iter().enumerate() {
                let (sref, cref) = sincos_scalar(xi);
                assert!(
                    (s[i] - sref).abs() < 1e-14,
                    "sin lane {i} at {xi}: {} vs {sref}",
                    s[i]
                );
                assert!(
                    (cc[i] - cref).abs() < 1e-14,
                    "cos lane {i} at {xi}: {} vs {cref}",
                    cc[i]
                );
            }
        }
    }

    #[test]
    fn avx2_matches_scalar_at_reduction_edges_and_specials() {
        let xs = [
            -CODY_WAITE_LIMIT - 1.0,
            -CODY_WAITE_LIMIT,
            -std::f64::consts::PI / 2.0,
            -0.0,
            f64::MIN_POSITIVE,
            std::f64::consts::PI / 2.0,
            CODY_WAITE_LIMIT,
            CODY_WAITE_LIMIT + 1.0,
            f64::MAX,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for chunk in xs.chunks(4) {
            let a = chunk[0];
            let b = *chunk.get(1).unwrap_or(&a);
            let c = *chunk.get(2).unwrap_or(&a);
            let d = *chunk.get(3).unwrap_or(&a);
            let (sv, cv) = unsafe { sincos_f64x4(pack(a, b, c, d)) };
            let sins = extract(sv);
            let coss = extract(cv);
            for (x, s, c) in [
                (a, sins[0], coss[0]),
                (b, sins[1], coss[1]),
                (c, sins[2], coss[2]),
                (d, sins[3], coss[3]),
            ] {
                let (want_s, want_c) = sincos_scalar(x);
                if want_s.is_nan() {
                    assert!(s.is_nan() && c.is_nan());
                } else {
                    assert_eq!(s, want_s, "sin mismatch at {x}");
                    assert_eq!(c, want_c, "cos mismatch at {x}");
                }
            }
        }
    }
}
