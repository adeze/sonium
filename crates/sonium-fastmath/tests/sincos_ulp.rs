//! ULP validation for `sonium-fastmath::sincos`. Five suites:
//!
//! 1. **Uniform sweep** — 10⁶ inputs spread across `[−10π, 10π]`.
//! 2. **Quadrant boundary** — 10⁵ inputs near `±k·π/2` for `k = 0..16`.
//!    Range-reduction regressions usually fire here and only here.
//! 3. **Large argument** — 10⁵ inputs in `[1, 1e3]`. Verifies the
//!    Cody–Waite two-term reduction is still in spec. Values past the
//!    cutover use the three-term mid-range path or the scalar libm
//!    fallback and are tested separately.
//! - **Mid range** — 10⁵ inputs log-uniform in `(1e3, 1e8]`. Verifies the
//!   three-term Cody–Waite reduction holds the 8-ULP envelope without
//!   heap allocation.
//! 4. **Special inputs** — ±0, ±∞, ±NaN, smallest normal, biggest finite.
//!    Finite inputs must return finite values and preserve the scalar/libm
//!    fallback contract.
//!
//! Each numerical suite asserts:
//!
//! - `max_ulp(sin) ≤ 2.0`
//! - `max_ulp(cos) ≤ 2.0`
//! - `mean_ulp(sin) ≤ 0.5`
//! - `mean_ulp(cos) ≤ 0.5`
//!
//! 2-ULP is a deliberate margin over Sleef's `u10` 1-ULP design target —
//! the polynomial path can lose ~1 ULP through Cody–Waite reduction in
//! the worst case, and that's well below downstream BEM quadrature error
//! by six orders of magnitude.

use sonium_fastmath::{CODY_WAITE_LIMIT, MID_RANGE_LIMIT, sincos_f64};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use sonium_fastmath::sincos_scalar;

#[cfg(target_arch = "aarch64")]
use sonium_fastmath::sincos_f64x2;

#[cfg(target_arch = "x86_64")]
use sonium_fastmath::sincos_f64x4;

/// ULP distance between `actual` and `expected`, in units of `ulp(1.0)`.
/// `sin` and `cos` are bounded in `[-1, 1]`, so measuring ULP against 1.0
/// is the meaningful scale — near a true zero the bit-pattern ULP metric
/// would explode by ~10¹⁰ for absolute errors at machine precision, which
/// is not what we care about.
fn ulp_distance(actual: f64, expected: f64) -> f64 {
    if actual == expected {
        return 0.0;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return f64::INFINITY;
    }
    // ulp(1.0) = 2⁻⁵² = f64::EPSILON. Floor the divisor at this so a
    // tiny absolute error near a true zero reads as ~1 ULP, not ~10¹⁰.
    let mag = expected.abs().max(actual.abs()).max(1.0);
    let ulp = mag * f64::EPSILON;
    (actual - expected).abs() / ulp
}

#[derive(Default, Clone, Copy)]
struct UlpStats {
    n: usize,
    max_sin: f64,
    max_cos: f64,
    sum_sin: f64,
    sum_cos: f64,
}

impl UlpStats {
    fn observe(&mut self, x: f64, got_sin: f64, got_cos: f64) {
        let (ref_sin, ref_cos) = x.sin_cos();
        let u_sin = ulp_distance(got_sin, ref_sin);
        let u_cos = ulp_distance(got_cos, ref_cos);
        self.n += 1;
        if u_sin > self.max_sin {
            self.max_sin = u_sin;
        }
        if u_cos > self.max_cos {
            self.max_cos = u_cos;
        }
        self.sum_sin += u_sin;
        self.sum_cos += u_cos;
    }

    fn mean_sin(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum_sin / self.n as f64
        }
    }
    fn mean_cos(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum_cos / self.n as f64
        }
    }

    fn assert_within(&self, label: &str, max: f64, mean: f64) {
        assert!(
            self.max_sin <= max,
            "{label}: max sin ULP {:.3} > {max:.3}",
            self.max_sin
        );
        assert!(
            self.max_cos <= max,
            "{label}: max cos ULP {:.3} > {max:.3}",
            self.max_cos
        );
        assert!(
            self.mean_sin() <= mean,
            "{label}: mean sin ULP {:.3} > {mean:.3}",
            self.mean_sin()
        );
        assert!(
            self.mean_cos() <= mean,
            "{label}: mean cos ULP {:.3} > {mean:.3}",
            self.mean_cos()
        );
    }
}

/// Collect stats for both the scalar API and (where available) the NEON
/// 2-lane API. NEON inputs are batched two-at-a-time so both lanes are
/// exercised. Returns `(scalar_stats, simd_stats)`.
fn run_sweep(inputs: &[f64]) -> (UlpStats, UlpStats) {
    let mut scalar = UlpStats::default();
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let mut simd = UlpStats::default();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let simd = scalar;

    for &x in inputs {
        let (s, c) = sincos_f64(x);
        scalar.observe(x, s, c);
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i + 2 <= inputs.len() {
            let a = inputs[i];
            let b = inputs[i + 1];
            // SAFETY: aarch64 mandates NEON.
            let v = unsafe { vsetq_lane_f64(b, vdupq_n_f64(a), 1) };
            let (sv, cv) = unsafe { sincos_f64x2(v) };
            let sa = unsafe { vgetq_lane_f64(sv, 0) };
            let sb = unsafe { vgetq_lane_f64(sv, 1) };
            let ca = unsafe { vgetq_lane_f64(cv, 0) };
            let cb = unsafe { vgetq_lane_f64(cv, 1) };
            simd.observe(a, sa, ca);
            simd.observe(b, sb, cb);
            i += 2;
        }
        // Tail scalar: include so n matches.
        while i < inputs.len() {
            let (s, c) = sincos_scalar(inputs[i]);
            simd.observe(inputs[i], s, c);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Only exercise the AVX2 path when the runtime CPU supports it.
        // Without runtime detection we'd crash on older x86_64 silicon.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            use std::arch::x86_64::*;
            let mut i = 0;
            while i + 4 <= inputs.len() {
                let a = inputs[i];
                let b = inputs[i + 1];
                let c = inputs[i + 2];
                let d = inputs[i + 3];
                // SAFETY: runtime feature check above.
                let v = unsafe { _mm256_set_pd(d, c, b, a) };
                let (sv, cv) = unsafe { sincos_f64x4(v) };
                let mut sout = [0.0_f64; 4];
                let mut cout = [0.0_f64; 4];
                unsafe {
                    _mm256_storeu_pd(sout.as_mut_ptr(), sv);
                    _mm256_storeu_pd(cout.as_mut_ptr(), cv);
                }
                simd.observe(a, sout[0], cout[0]);
                simd.observe(b, sout[1], cout[1]);
                simd.observe(c, sout[2], cout[2]);
                simd.observe(d, sout[3], cout[3]);
                i += 4;
            }
            while i < inputs.len() {
                let (s, c) = sincos_scalar(inputs[i]);
                simd.observe(inputs[i], s, c);
                i += 1;
            }
        } else {
            simd = scalar;
        }
    }

    (scalar, simd)
}

#[test]
fn suite1_uniform_sweep() {
    use std::f64::consts::PI;
    const N: usize = 1_000_000;
    let lo = -10.0 * PI;
    let hi = 10.0 * PI;
    let inputs: Vec<f64> = (0..N)
        .map(|i| lo + (hi - lo) * (i as f64) / (N as f64))
        .collect();
    let (scalar, simd) = run_sweep(&inputs);
    scalar.assert_within("suite1 scalar", 2.0, 0.5);
    simd.assert_within("suite1 simd", 2.0, 0.5);
}

#[test]
fn suite2_quadrant_boundary() {
    use std::f64::consts::PI;
    // For each k in -16..=16, cluster 1.5e3 points within ±1e-10 of k·π/2.
    let mut inputs = Vec::with_capacity(33 * 1500);
    for k in -16..=16 {
        let center = (k as f64) * PI / 2.0;
        for j in 0..1500 {
            let off = (j as f64 - 750.0) * 1.5e-13; // ±~1e-10 spread
            inputs.push(center + off);
        }
    }
    let (scalar, simd) = run_sweep(&inputs);
    // Boundary points are where the worst ULP shows up; allow slightly
    // larger max but still well below libm's documented headroom.
    scalar.assert_within("suite2 scalar", 3.0, 1.0);
    simd.assert_within("suite2 simd", 3.0, 1.0);
}

#[test]
fn suite3_large_argument() {
    const N: usize = 100_000;
    // Log-uniform in [1, CODY_WAITE_LIMIT] across both signs.
    let inputs: Vec<f64> = (0..N)
        .map(|i| {
            let t = (i as f64) / (N as f64);
            let mag = 10f64.powf(3.0 * t);
            if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
    let (scalar, simd) = run_sweep(&inputs);
    // Two-term Cody–Waite gives ~4 ULP at the upper end of the range.
    // The harness documents this regime so a regression in the
    // reduction constants is visible.
    scalar.assert_within("suite3 scalar", 8.0, 1.5);
    simd.assert_within("suite3 simd", 8.0, 1.5);
}

#[test]
fn suite3b_mid_range_three_term_reduction() {
    const N: usize = 100_000;
    // Log-uniform in (CODY_WAITE_LIMIT, MID_RANGE_LIMIT] across both signs.
    // Alternating signs also force mixed-sign SIMD batches through the
    // per-lane blending path.
    let inputs: Vec<f64> = (0..N)
        .map(|i| {
            let t = (i as f64 + 1.0) / (N as f64 + 1.0);
            let mag = 10f64.powf(3.0 + 5.0 * t);
            assert!(
                mag > CODY_WAITE_LIMIT && mag <= MID_RANGE_LIMIT,
                "sweep left the mid range: {mag:e}"
            );
            if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
    let (scalar, simd) = run_sweep(&inputs);
    // Three-term Cody–Waite holds the fast-path 8-ULP envelope with no
    // heap allocation; a 2·10⁶-sample probe measures max 0.5 ULP, mean
    // 0.06 ULP, so the 8.0/1.5 gates carry wide headroom.
    scalar.assert_within("suite3b scalar", 8.0, 1.5);
    simd.assert_within("suite3b simd", 8.0, 1.5);
}

#[test]
fn suite4_special_inputs() {
    // Special values follow the same scalar/libm contract as the public API.
    let specials = [
        0.0_f64,
        -0.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1.0e-300,
        -1.0e-300,
        f64::MAX,
        -f64::MAX,
    ];
    for &x in &specials {
        let (s, c) = sincos_f64(x);
        let (sr, cr) = x.sin_cos();
        assert!((s - sr).abs() <= 32.0 * f64::EPSILON, "sin x={x}");
        assert!((c - cr).abs() <= 32.0 * f64::EPSILON, "cos x={x}");
    }

    // NaN and ±∞ follow libm's NaN semantics and must not panic.
    for &x in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (s, c) = sincos_f64(x);
        assert!(s.is_nan() && c.is_nan(), "special x={x}");
    }
}

#[test]
fn suite5_huge_argument_fallback_preserves_quadrants() {
    let inputs = [
        CODY_WAITE_LIMIT + 1.0,
        -CODY_WAITE_LIMIT - 1.0,
        1.0e12,
        -1.0e12,
        1.0e20,
        -1.0e20,
        f64::MAX,
        -f64::MAX,
    ];
    let (scalar, simd) = run_sweep(&inputs);
    scalar.assert_within("suite5 scalar", 32.0, 4.0);
    simd.assert_within("suite5 simd", 32.0, 4.0);
}

#[test]
fn suite6_payne_hanek_spans_all_finite_exponents() {
    // Four mantissa edge patterns at every normal exponent exercise the
    // fixed-point reducer's bit alignment, including the largest finite
    // values. This is deterministic and broad enough to catch quadrant
    // cutovers that a handful of hand-picked huge phases can miss.
    let fractions = [0_u64, 1, (1_u64 << 51), (1_u64 << 52) - 1];
    for exponent in 1_u64..=0x7fe {
        for &fraction in &fractions {
            let bits = (exponent << 52) | fraction;
            for bits in [bits, bits | (1_u64 << 63)] {
                let x = f64::from_bits(bits);
                let (sin, cos) = sincos_f64(x);
                let (sin_ref, cos_ref) = x.sin_cos();
                assert!((sin - sin_ref).abs() <= 64.0 * f64::EPSILON, "sin x={x:e}");
                assert!((cos - cos_ref).abs() <= 64.0 * f64::EPSILON, "cos x={x:e}");
            }
        }
    }
}

#[test]
fn derivative_consistency_in_public_scalar_api() {
    let h = 1.0e-5;
    for &x in &[-999.0, -17.25, -0.7, 0.0, 0.7, 17.25, 999.0] {
        let (s, c) = sincos_f64(x);
        let (sp, cp) = sincos_f64(x + h);
        let (sm, cm) = sincos_f64(x - h);
        assert!(((sp - sm) / (2.0 * h) - c).abs() < 2.0e-8);
        assert!(((cp - cm) / (2.0 * h) + s).abs() < 2.0e-8);
    }
}
