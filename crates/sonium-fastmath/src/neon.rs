//! NEON f64x2 implementation of [`sincos_f64x2`]. AArch64-only.
//!
//! Same algorithm as [`crate::scalar::sincos_scalar`]: two-term Cody–Waite
//! range reduction, Sleef u10 minimax polynomials in Horner form, quadrant
//! select via bitwise masks. The vector kernel runs both `float64x2_t`
//! lanes; lanes outside its domain (zero, non-finite,
//! `|x| > CODY_WAITE_LIMIT`) are blended per lane with scalar results so
//! one outlier no longer forces the whole pair scalar.
//!
//! The polynomial chain is intentionally shortened relative to the scalar
//! path (see [`poly_sin`]/[`poly_cos`]): the sine chain drops the two
//! smallest terms and the cosine chain folds the omitted highest term into
//! compensated `COS_C*_COMP` constants. The pair agrees with scalar output
//! within the ULP envelope in `lib.rs`, pinned by
//! `tests/lane_equivalence.rs`, rather than bit-for-bit.

use std::arch::aarch64::*;

use crate::coeffs::{
    COS_C1, COS_C2, COS_C3, COS_C4, COS_C5, COS_C6, PI_OVER_2_HI, PI_OVER_2_LO, SIN_S2, SIN_S3,
    SIN_S4, SIN_S5, SIN_S6, SIN_S7, SIN_S8, TWO_OVER_PI,
};
use crate::scalar::{CODY_WAITE_LIMIT, sincos_scalar};

// Degree-6 minimax compensation for the omitted COS_C0*s^7 contribution over
// the reduced-domain interval s in [0, (pi/4)^2]. This keeps the shortened
// Horner chain's instruction count while preserving the fastmath ULP envelope.
const COS_C1_COMP: f64 = COS_C1 - 2.102_509_801_408_321_3e-11;
const COS_C2_COMP: f64 = COS_C2 + 1.459_050_467_998_576_3e-11;
const COS_C3_COMP: f64 = COS_C3 - 4.666_747_983_453_317e-12;
const COS_C4_COMP: f64 = COS_C4 + 6.746_917_446_718_613e-13;
const COS_C5_COMP: f64 = COS_C5 - 3.567_289_613_888_604e-14;

/// Lane-parallel `(sin(x), cos(x))` for two f64 inputs. For finite lanes
/// within [`CODY_WAITE_LIMIT`], the fast path follows the scalar contract;
/// larger finite lanes use the scalar fallback (three-term mid-range
/// reduction, then Payne–Hanek); non-finite lanes use `f64::sin_cos` so
/// their quadrants and special-value semantics are preserved.
///
/// # Safety
///
/// Caller must be running on aarch64 with NEON enabled. NEON is mandatory
/// on AArch64, so the compile-time `#[cfg(target_arch = "aarch64")]` gate
/// in `lib.rs` is sufficient.
#[target_feature(enable = "neon")]
pub unsafe fn sincos_f64x2(x: float64x2_t) -> (float64x2_t, float64x2_t) {
    // SAFETY: the whole body runs inside a `target_feature = "neon"`
    // function. NEON is mandatory on AArch64, so every intrinsic is sound
    // under the function's calling contract.
    unsafe {
        // Keep the vector reduction on its documented domain. Lanes needing
        // the scalar fallback are blended per lane so the ordinary lane of
        // a mixed pair keeps vector throughput.
        let x0 = vgetq_lane_f64(x, 0);
        let x1 = vgetq_lane_f64(x, 1);
        let fallback0 = !x0.is_finite() || x0 == 0.0 || x0.abs() > CODY_WAITE_LIMIT;
        let fallback1 = !x1.is_finite() || x1 == 0.0 || x1.abs() > CODY_WAITE_LIMIT;
        if fallback0 && fallback1 {
            let (s0, c0) = sincos_scalar(x0);
            let (s1, c1) = sincos_scalar(x1);
            return (
                vsetq_lane_f64(s1, vdupq_n_f64(s0), 1),
                vsetq_lane_f64(c1, vdupq_n_f64(c0), 1),
            );
        }
        if fallback0 || fallback1 {
            // Mixed pair: run the vector kernel for both lanes — its output
            // on a fallback lane is well-defined FP arithmetic (no traps)
            // but discarded — then overwrite only that lane.
            let (vec_sin, vec_cos) = sincos_f64x2_vector(x);
            let (mut s0, mut s1) = (vgetq_lane_f64(vec_sin, 0), vgetq_lane_f64(vec_sin, 1));
            let (mut c0, mut c1) = (vgetq_lane_f64(vec_cos, 0), vgetq_lane_f64(vec_cos, 1));
            if fallback0 {
                // Scalar path preserves NaN and signed-zero semantics.
                (s0, c0) = sincos_scalar(x0);
            }
            if fallback1 {
                (s1, c1) = sincos_scalar(x1);
            }
            return (
                vsetq_lane_f64(s1, vdupq_n_f64(s0), 1),
                vsetq_lane_f64(c1, vdupq_n_f64(c0), 1),
            );
        }

        return sincos_f64x2_vector(x);
    }
}

/// Vector kernel for [`sincos_f64x2`]: caller guarantees both lanes hold
/// non-zero finite values within [`CODY_WAITE_LIMIT`], or discards that
/// lane's output.
///
/// # Safety
///
/// Caller must be running on aarch64 with NEON enabled.
#[target_feature(enable = "neon")]
unsafe fn sincos_f64x2_vector(x: float64x2_t) -> (float64x2_t, float64x2_t) {
    // SAFETY: same intrinsic soundness argument as the entry point.
    unsafe {
        // 1. Range reduce: x = k · π/2 + r, r ∈ [-π/4, π/4]
        let two_over_pi = vdupq_n_f64(TWO_OVER_PI);
        let k_f = vrndnq_f64(vmulq_f64(x, two_over_pi)); // round-to-nearest-even
        let k_i = vcvtq_s64_f64(k_f);

        // Two-term Cody–Waite: r = ((x - k·HI) - k·LO).
        // vfmsq_f64(a, b, c) = a - b*c.
        let pi_hi = vdupq_n_f64(PI_OVER_2_HI);
        let pi_lo = vdupq_n_f64(PI_OVER_2_LO);
        let r = vfmsq_f64(x, k_f, pi_hi);
        let r = vfmsq_f64(r, k_f, pi_lo);
        let s = vmulq_f64(r, r);

        // 2. Polynomial sin(r), cos(r), evaluated via vfmaq_f64 Horner.
        //    The sine chain skips the two smallest terms; the cosine chain
        //    uses compensated coefficients for the omitted highest term.
        let sin_r = poly_sin(r, s);
        let cos_r = poly_cos(s);

        // 3. Quadrant select on k mod 4.
        //    bit 0 of k → swap sin and cos
        //    bit 1 of k → negate sin     (i.e. q ∈ {2, 3})
        //    bit 1 of (k+1) → negate cos (i.e. q ∈ {1, 2})
        let three = vdupq_n_s64(3);
        let one = vdupq_n_s64(1);
        let two = vdupq_n_s64(2);

        let k_mod4 = vandq_s64(k_i, three);
        let swap = vceqq_s64(vandq_s64(k_mod4, one), one); // k_mod4 & 1 == 1
        let neg_sin = vtstq_s64(k_mod4, two); // k_mod4 & 2 != 0
        let neg_cos = vtstq_s64(vaddq_s64(k_mod4, one), two); // (k_mod4 + 1) & 2

        // vbslq_f64 takes a uint64x2_t mask: vceqq/vtstq return u64x2,
        // which is exactly the right type for vbslq_f64.
        let sin_out = vbslq_f64(swap, cos_r, sin_r);
        let cos_out = vbslq_f64(swap, sin_r, cos_r);

        let sign_bit = vdupq_n_u64(0x8000_0000_0000_0000);
        let sin_bits = vreinterpretq_u64_f64(sin_out);
        let cos_bits = vreinterpretq_u64_f64(cos_out);
        let sin_neg = veorq_u64(sin_bits, sign_bit);
        let cos_neg = veorq_u64(cos_bits, sign_bit);
        let sin_final = vreinterpretq_f64_u64(vbslq_u64(neg_sin, sin_neg, sin_bits));
        let cos_final = vreinterpretq_f64_u64(vbslq_u64(neg_cos, cos_neg, cos_bits));

        (sin_final, cos_final)
    }
}

/// Lane-parallel f32 sine/cosine for four values, widened through the f64
/// kernel and narrowed once.
///
/// # Safety
///
/// The caller must be running on AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn sincos_f32x4(x: float32x4_t) -> (float32x4_t, float32x4_t) {
    unsafe {
        let lo = vcvt_f64_f32(vget_low_f32(x));
        let hi = vcvt_f64_f32(vget_high_f32(x));
        let (sin_lo, cos_lo) = sincos_f64x2(lo);
        let (sin_hi, cos_hi) = sincos_f64x2(hi);
        (
            vcombine_f32(vcvt_f32_f64(sin_lo), vcvt_f32_f64(sin_hi)),
            vcombine_f32(vcvt_f32_f64(cos_lo), vcvt_f32_f64(cos_hi)),
        )
    }
}

/// SIMD Horner for the Sleef u10 sin polynomial: `sin(r) = r + r·s·U(s)`,
/// where `s = r²`. NEON is mandatory on AArch64; the caller wrap-in
/// (`sincos_f64x2`) already enables the feature, so this function inherits.
#[inline]
unsafe fn poly_sin(r: float64x2_t, s: float64x2_t) -> float64x2_t {
    unsafe {
        let mut u = vdupq_n_f64(SIN_S2);
        u = vfmaq_f64(vdupq_n_f64(SIN_S3), u, s);
        u = vfmaq_f64(vdupq_n_f64(SIN_S4), u, s);
        u = vfmaq_f64(vdupq_n_f64(SIN_S5), u, s);
        u = vfmaq_f64(vdupq_n_f64(SIN_S6), u, s);
        u = vfmaq_f64(vdupq_n_f64(SIN_S7), u, s);
        u = vfmaq_f64(vdupq_n_f64(SIN_S8), u, s);
        // sin(r) = r + (r·s)·u = r + r·s·u
        vfmaq_f64(r, vmulq_f64(r, s), u)
    }
}

/// SIMD Horner for the Sleef u10 cos polynomial: `cos(r) = 1 + s·V(s)`.
#[inline]
unsafe fn poly_cos(s: float64x2_t) -> float64x2_t {
    unsafe {
        let mut v = vdupq_n_f64(COS_C1_COMP);
        v = vfmaq_f64(vdupq_n_f64(COS_C2_COMP), v, s);
        v = vfmaq_f64(vdupq_n_f64(COS_C3_COMP), v, s);
        v = vfmaq_f64(vdupq_n_f64(COS_C4_COMP), v, s);
        v = vfmaq_f64(vdupq_n_f64(COS_C5_COMP), v, s);
        v = vfmaq_f64(vdupq_n_f64(COS_C6), v, s);
        // cos(r) = 1 + s·v
        vfmaq_f64(vdupq_n_f64(1.0), s, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::sincos_scalar;

    /// Pack scalar `a, b` into `float64x2_t` with lane 0 = a, lane 1 = b.
    #[inline]
    unsafe fn pack(a: f64, b: f64) -> float64x2_t {
        unsafe { vsetq_lane_f64(b, vdupq_n_f64(a), 1) }
    }

    fn extract(v: float64x2_t) -> (f64, f64) {
        unsafe { (vgetq_lane_f64(v, 0), vgetq_lane_f64(v, 1)) }
    }

    #[test]
    fn neon_matches_scalar_at_known_points() {
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
        ];
        for chunk in xs.chunks(2) {
            let a = chunk[0];
            let b = *chunk.get(1).unwrap_or(&chunk[0]);
            let xv = unsafe { pack(a, b) };
            let (sv, cv) = unsafe { sincos_f64x2(xv) };
            let (sa, sb) = extract(sv);
            let (ca, cb) = extract(cv);

            let (sa_ref, ca_ref) = sincos_scalar(a);
            let (sb_ref, cb_ref) = sincos_scalar(b);

            assert!(
                (sa - sa_ref).abs() < 1e-14,
                "sin lane 0 at {a}: {sa} vs {sa_ref}"
            );
            assert!(
                (sb - sb_ref).abs() < 1e-14,
                "sin lane 1 at {b}: {sb} vs {sb_ref}"
            );
            assert!(
                (ca - ca_ref).abs() < 1e-14,
                "cos lane 0 at {a}: {ca} vs {ca_ref}"
            );
            assert!(
                (cb - cb_ref).abs() < 1e-14,
                "cos lane 1 at {b}: {cb} vs {cb_ref}"
            );
        }
    }

    #[test]
    fn neon_vs_scalar_max_diff_stays_in_envelope() {
        // Pins the documented compensated-shortened-chain agreement: a
        // dense sweep over the fast-path domain must stay within the ULP
        // envelope in `lib.rs` rather than drifting toward bit-for-bit
        // equality (which the shortened chain does not provide).
        let n = 50_000;
        let mut max_sin = 0.0_f64;
        let mut max_cos = 0.0_f64;
        for i in 0..n {
            let a = -CODY_WAITE_LIMIT + 2.0 * CODY_WAITE_LIMIT * (i as f64) / (n as f64);
            let xv = unsafe { pack(a, -a) };
            let (sv, cv) = unsafe { sincos_f64x2(xv) };
            let (sa, sb) = extract(sv);
            let (ca, cb) = extract(cv);
            let (sra, cra) = sincos_scalar(a);
            let (srb, crb) = sincos_scalar(-a);
            max_sin = max_sin.max((sa - sra).abs()).max((sb - srb).abs());
            max_cos = max_cos.max((ca - cra).abs()).max((cb - crb).abs());
        }
        assert!(
            max_sin <= 8.0 * f64::EPSILON,
            "NEON-vs-scalar max sin diff {max_sin:e}"
        );
        assert!(
            max_cos <= 8.0 * f64::EPSILON,
            "NEON-vs-scalar max cos diff {max_cos:e}"
        );
    }

    #[test]
    fn neon_mixed_pair_blends_per_lane() {
        // One outlier must not disturb the ordinary lane: the fallback
        // lane takes the scalar path (bitwise equal), the vector lane
        // stays within the envelope.
        for (a, b) in [
            (0.5, f64::NAN),
            (f64::INFINITY, 1.0),
            (-0.0, 2.0),
            (CODY_WAITE_LIMIT + 1.0, -1.0),
        ] {
            let (sv, cv) = unsafe { sincos_f64x2(pack(a, b)) };
            let (sa, sb) = extract(sv);
            let (ca, cb) = extract(cv);
            for (x, s, c) in [(a, sa, ca), (b, sb, cb)] {
                let (want_s, want_c) = sincos_scalar(x);
                if want_s.is_nan() {
                    assert!(s.is_nan() && c.is_nan(), "NaN lane at {x}");
                } else if x == 0.0 || !x.is_finite() || x.abs() > CODY_WAITE_LIMIT {
                    assert_eq!(s, want_s, "fallback sin lane at {x}");
                    assert_eq!(c, want_c, "fallback cos lane at {x}");
                } else {
                    assert!(
                        (s - want_s).abs() <= 8.0 * f64::EPSILON,
                        "vector sin lane at {x}"
                    );
                    assert!(
                        (c - want_c).abs() <= 8.0 * f64::EPSILON,
                        "vector cos lane at {x}"
                    );
                }
            }
        }
    }

    #[test]
    fn neon_matches_scalar_at_reduction_edges_and_specials() {
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
        for chunk in xs.chunks(2) {
            let a = chunk[0];
            let b = *chunk.get(1).unwrap_or(&a);
            let (sv, cv) = unsafe { sincos_f64x2(pack(a, b)) };
            let (sa, sb) = extract(sv);
            let (ca, cb) = extract(cv);
            for (x, s, c) in [(a, sa, ca), (b, sb, cb)] {
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
