//! Scalar f64 sin/cos via Cody–Waite range reduction + Sleef u10 minimax
//! polynomials. Serves three purposes:
//!
//! 1. **Reference implementation** — the SIMD path uses the same polynomial,
//!    so this is the scalar version of the same algorithm and a useful
//!    standalone tool when only one input is in flight.
//! 2. **Fallback** — exposed via [`sincos_f64`] in `lib.rs` for `cfg`
//!    targets without a vectorised implementation.
//! 3. **Test reference** — the ULP harness compares both this scalar path
//!    and the SIMD path against `libm` via `f64::sin_cos`.

use crate::coeffs::{
    COS_C0, COS_C1, COS_C2, COS_C3, COS_C4, COS_C5, COS_C6, PI_OVER_2_HI, PI_OVER_2_LO, SIN_S0,
    SIN_S1, SIN_S2, SIN_S3, SIN_S4, SIN_S5, SIN_S6, SIN_S7, SIN_S8, TWO_OVER_PI,
};
use crate::payne_hanek;
use std::sync::atomic::{AtomicU64, Ordering};

static SLOW_PATH_CALLS: AtomicU64 = AtomicU64::new(0);
static MID_PATH_CALLS: AtomicU64 = AtomicU64::new(0);

/// Largest magnitude for which the two-term Cody–Waite reduction is used.
///
/// Two double-double terms keep `k·HI` rounding error below ~1 ULP while
/// `|k| ≲ 640` (`|x| ≤ 10³ ≪ 2²⁰`, the exactness bound noted in
/// `coeffs.rs`). Past this cutover the reduced argument needs a third
/// correction term (see [`MID_RANGE_LIMIT`]); the two-term path is kept
/// because it is one FMA shorter per lane in the SIMD kernels.
pub const CODY_WAITE_LIMIT: f64 = 1.0e3;

/// Largest magnitude for which the three-term Cody–Waite reduction is used.
///
/// `PI_OVER_2_MID_HI` carries 25 significant bits and `PI_OVER_2_MID_MID`
/// carries 24, so for `|k| < 2²⁶` (i.e. `|x| ≤ 10⁸`) both `k·HI` and `k·MID`
/// are exact in f64 and the two subtractions lose at most ~2 ULP(1.0)
/// absolute. Past this cutover only the fixed-point Payne–Hanek path in
/// `payne_hanek` retains the quadrant reliably.
pub const MID_RANGE_LIMIT: f64 = 1.0e8;

/// High 25 significant bits of π/2 (`0x3FF921FB50000000`).
const PI_OVER_2_MID_HI: f64 = 1.5707963109016418;
/// Next 24 significant bits of π/2 (`0x3E5110B460000000`).
const PI_OVER_2_MID_MID: f64 = 1.5893254712295857e-08;
/// f64-rounded remainder of π/2 after the two head terms.
const PI_OVER_2_MID_LO: f64 = 6.123233995736766e-17;

#[inline]
pub(crate) fn record_slow_path() {
    SLOW_PATH_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_mid_path() {
    MID_PATH_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Return the number of scalar evaluations that used the robust fallback.
///
/// The counter is process-global and intentionally approximate under heavy
/// concurrent use. It is a diagnostic for workload profiling, not a timing or
/// synchronisation primitive.
pub fn slow_path_count() -> u64 {
    SLOW_PATH_CALLS.load(Ordering::Relaxed)
}

/// Return the number of scalar evaluations that used the three-term
/// mid-range reduction (`CODY_WAITE_LIMIT < |x| <= MID_RANGE_LIMIT`).
///
/// Same approximate-process-global semantics as [`slow_path_count`].
pub fn mid_path_count() -> u64 {
    MID_PATH_CALLS.load(Ordering::Relaxed)
}

/// Reset the process-global slow-path diagnostic counter.
pub fn reset_slow_path_count() {
    SLOW_PATH_CALLS.store(0, Ordering::Relaxed);
}

/// Reset the process-global mid-range diagnostic counter.
pub fn reset_mid_path_count() {
    MID_PATH_CALLS.store(0, Ordering::Relaxed);
}

/// Range-reduce + polynomial-evaluate sin and cos at `x`.
///
/// For finite `|x| <= [`CODY_WAITE_LIMIT`]`, this uses the fast Cody–Waite
/// plus Sleef-u10 polynomial path. For `CODY_WAITE_LIMIT < |x| <=
/// [`MID_RANGE_LIMIT`]` it uses a three-term Cody–Waite reduction (same
/// polynomial, no heap allocation); larger finite values use the
/// fixed-point Payne–Hanek slow path; infinities and NaN delegate to
/// [`f64::sin_cos`]. This partition is essential: a two-term reduction cannot
/// retain the quadrant for the full finite `f64` domain.
///
/// The fast path's published acceptance envelope is at most 8 scaled ULPs
/// against `f64::sin_cos` over its finite domain, and the mid-range path
/// holds the same 8-ULP envelope (measured max 0.5 ULP, mean 0.06 ULP over
/// 2·10⁶ log-uniform samples in `(10³, 10⁸]`; see `tests/sincos_ulp.rs`).
/// Near a mathematical zero, the contract is
/// absolute (the same scaled-ULP metric), not relative; relative error is
/// undefined at the zero. Signed zero is preserved for sine and cosine is
/// `+1`; non-finite inputs follow `f64::sin_cos`.
#[inline]
pub fn sincos_scalar(x: f64) -> (f64, f64) {
    if x == 0.0 {
        return (x, 1.0);
    }
    if !x.is_finite() {
        record_slow_path();
        return x.sin_cos();
    }
    if x.abs() > CODY_WAITE_LIMIT {
        if x.abs() <= MID_RANGE_LIMIT {
            record_mid_path();
            return sincos_mid_range(x);
        }
        record_slow_path();
        let reduction = payne_hanek::reduce(x);
        let s = reduction.remainder * reduction.remainder;
        return select_quadrant(
            poly_sin(reduction.remainder, s),
            poly_cos(s),
            reduction.quadrant,
        );
    }

    // 1. Range reduce: x = k * π/2 + r, r ∈ [-π/4, π/4]
    let k_f = (x * TWO_OVER_PI).round();
    let k = k_f as i64;

    // Two-term Cody–Waite: r = ((x - k·HI) - k·LO)
    let r = x - k_f * PI_OVER_2_HI;
    let r = r - k_f * PI_OVER_2_LO;
    let s = r * r;

    // 2. Polynomial evaluation on [-π/4, π/4]
    let sinr = poly_sin(r, s);
    let cosr = poly_cos(s);

    // 3. Quadrant select on k mod 4
    select_quadrant(sinr, cosr, (k & 3) as u8)
}

/// Three-term Cody–Waite reduction for `CODY_WAITE_LIMIT < |x| <=
/// MID_RANGE_LIMIT`.
///
/// The caller guarantees a non-zero finite input in range. `|k| < 2²⁶`
/// there, and the head constants carry only 25/24 significant bits, so
/// `k·HI` and `k·MID` are exact in f64; the first subtraction is exact by
/// Sterbenz (`x` and `k·HI` agree to ~24 bits) and the remaining two
/// roundings contribute ≲ 2 ULP(1.0) absolute to `r`. The Sleef-u10
/// polynomial then evaluates on `r ∈ [-π/4, π/4]` exactly as in the fast
/// path, so no second polynomial implementation exists to drift.
///
/// Measured single-call latency (release, x86_64): ~63 ns/elem at `1e6`
/// vs ~720 ns/elem for the fixed-point bigint path at `1e20` — over 10×
/// lower with no heap allocation (`benches/sincos_bench.rs`, `mid_range`
/// regime vs `huge_fallback`).
#[inline]
fn sincos_mid_range(x: f64) -> (f64, f64) {
    debug_assert!(
        x.is_finite() && x != 0.0 && x.abs() > CODY_WAITE_LIMIT && x.abs() <= MID_RANGE_LIMIT
    );
    let f = x * TWO_OVER_PI;
    let k_f = f.round();
    if (f - k_f).abs() > 0.5 - 1e-6 {
        // `f` is within 10⁻⁶ of a half-integer, closer than the single
        // rounding's ~10⁻⁸ error bar can resolve: the rounded `k` may name
        // the wrong quadrant. These inputs are measure-zero in practice;
        // resolve them through the exact fixed-point path instead.
        record_slow_path();
        let reduction = payne_hanek::reduce(x);
        let s = reduction.remainder * reduction.remainder;
        return select_quadrant(
            poly_sin(reduction.remainder, s),
            poly_cos(s),
            reduction.quadrant,
        );
    }
    let k = k_f as i64;
    // Three-term reduction: r = ((x - k·HI) - k·MID) - k·LO.
    let r = x - k_f * PI_OVER_2_MID_HI;
    let r = r - k_f * PI_OVER_2_MID_MID;
    let r = r - k_f * PI_OVER_2_MID_LO;
    let s = r * r;
    select_quadrant(poly_sin(r, s), poly_cos(s), (k & 3) as u8)
}

#[inline]
fn select_quadrant(sinr: f64, cosr: f64, quadrant: u8) -> (f64, f64) {
    match quadrant & 3 {
        0 => (sinr, cosr),
        1 => (cosr, -sinr),
        2 => (-sinr, -cosr),
        // q == 3
        _ => (-cosr, sinr),
    }
}

/// Evaluate `sin(r)` for `r ∈ [-π/4, π/4]` using the Sleef u10 odd
/// polynomial. Layout: `sin(r) = r + r·s·U(s)` where `s = r²`.
#[inline]
fn poly_sin(r: f64, s: f64) -> f64 {
    let u = SIN_S0;
    let u = u.mul_add(s, SIN_S1);
    let u = u.mul_add(s, SIN_S2);
    let u = u.mul_add(s, SIN_S3);
    let u = u.mul_add(s, SIN_S4);
    let u = u.mul_add(s, SIN_S5);
    let u = u.mul_add(s, SIN_S6);
    let u = u.mul_add(s, SIN_S7);
    let u = u.mul_add(s, SIN_S8);
    r + r * s * u
}

/// Evaluate `cos(r)` for `r ∈ [-π/4, π/4]` using the Sleef u10 even
/// polynomial. Layout: `cos(r) = 1 + s·V(s)` where `s = r²`.
#[inline]
fn poly_cos(s: f64) -> f64 {
    let v = COS_C0;
    let v = v.mul_add(s, COS_C1);
    let v = v.mul_add(s, COS_C2);
    let v = v.mul_add(s, COS_C3);
    let v = v.mul_add(s, COS_C4);
    let v = v.mul_add(s, COS_C5);
    let v = v.mul_add(s, COS_C6);
    1.0 + s * v
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn matches_libm_at_well_known_points() {
        for &(x, sin_ref, cos_ref) in &[
            (0.0, 0.0, 1.0),
            (PI / 6.0, 0.5, 0.8660254037844387),
            (
                PI / 4.0,
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ),
            (PI / 3.0, 0.8660254037844386, 0.5),
            (PI / 2.0, 1.0, 0.0_f64),
            (PI, 0.0_f64, -1.0),
            (-PI / 2.0, -1.0, 0.0_f64),
            (3.0 * PI / 2.0, -1.0, 0.0_f64),
            (2.0 * PI, 0.0_f64, 1.0),
        ] {
            let (s, c) = sincos_scalar(x);
            assert_relative_eq!(s, sin_ref, epsilon = 1e-14, max_relative = 1e-14);
            assert_relative_eq!(c, cos_ref, epsilon = 1e-14, max_relative = 1e-14);
        }
    }

    #[test]
    fn pythagorean_identity_holds_across_range() {
        // sin² + cos² = 1 over a dense sweep.
        let n = 10_000;
        let lo = -50.0 * PI;
        let hi = 50.0 * PI;
        for i in 0..=n {
            let x = lo + (hi - lo) * (i as f64) / (n as f64);
            let (s, c) = sincos_scalar(x);
            let id = s * s + c * c;
            assert!((id - 1.0).abs() < 5e-15, "sin²+cos² at x={x} drifted: {id}");
        }
    }

    #[test]
    fn randomized_pythagorean_identity() {
        use rand::Rng;
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let x = rng.random_range(-1e3..1e3);
            let (s, c) = sincos_scalar(x);
            let id = s * s + c * c;
            assert!((id - 1.0).abs() < 5e-15, "sin²+cos² at x={x} drifted: {id}");
        }
    }

    #[test]
    fn randomized_matches_libm_within_ulp() {
        use rand::Rng;
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let x = rng.random_range(-1e3..1e3);
            let (s, c) = sincos_scalar(x);
            let (s_ref, c_ref) = x.sin_cos();
            let s_err = (s - s_ref).abs();
            let c_err = (c - c_ref).abs();
            // ULP envelope over the Cody–Waite domain.
            let ulp = f64::EPSILON;
            assert!(s_err < 2.0 * ulp, "sin error at x={x}: {s_err:e}");
            assert!(c_err < 2.0 * ulp, "cos error at x={x}: {c_err:e}");
        }
    }

    #[test]
    fn special_values_and_cutover_follow_libm() {
        for &x in &[
            CODY_WAITE_LIMIT,
            -CODY_WAITE_LIMIT,
            CODY_WAITE_LIMIT + 1.0,
            -CODY_WAITE_LIMIT - 1.0,
            1.0e20,
            -1.0e20,
            f64::MAX,
            -f64::MAX,
        ] {
            let got = sincos_scalar(x);
            let want = x.sin_cos();
            let tolerance = 32.0 * f64::EPSILON;
            assert!((got.0 - want.0).abs() <= tolerance, "sin mismatch at {x}");
            assert!((got.1 - want.1).abs() <= tolerance, "cos mismatch at {x}");
        }

        let (sin_neg_zero, cos_neg_zero) = sincos_scalar(-0.0);
        assert!(sin_neg_zero.is_sign_negative());
        assert_eq!(cos_neg_zero, 1.0);

        for &x in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (sin_x, cos_x) = sincos_scalar(x);
            assert!(sin_x.is_nan() && cos_x.is_nan(), "special input {x}");
        }
    }

    #[test]
    fn numerical_derivatives_match_sincos_pair() {
        // Central differences expose discontinuous quadrant selection even
        // when pointwise values happen to look plausible. Keep the probes in
        // the polynomial domain and away from the cutover.
        let h = 1.0e-5;
        for &x in &[
            -999.0,
            -1234.5,
            -std::f64::consts::PI / 2.0 + 1.0e-3,
            -0.7,
            0.0,
            0.7,
            std::f64::consts::PI / 2.0 - 1.0e-3,
            1234.5,
            999.0,
        ] {
            let (sin_x, cos_x) = sincos_scalar(x);
            let (sin_p, cos_p) = sincos_scalar(x + h);
            let (sin_m, cos_m) = sincos_scalar(x - h);
            let d_sin = (sin_p - sin_m) / (2.0 * h);
            let d_cos = (cos_p - cos_m) / (2.0 * h);
            assert!(
                (d_sin - cos_x).abs() < 2.0e-8,
                "d sin at {x}: {d_sin} vs {cos_x}"
            );
            assert!(
                (d_cos + sin_x).abs() < 2.0e-8,
                "d cos at {x}: {d_cos} vs {}",
                -sin_x
            );
        }
    }
}
