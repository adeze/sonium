//! Native x86 validation for every SIMD implementation and public API.
//!
//! The corpus is deterministic and deliberately mixes ordinary reduced
//! arguments, exact reduction neighbours, Cody--Waite cutover values,
//! subnormals, all normal exponents, and non-finite inputs.  AVX2 and AVX-512
//! are invoked directly so the test is independent of the public dispatch
//! choice; the public API checks below exercise the actual runtime dispatch.

#![cfg(target_arch = "x86_64")]

use num_complex::{Complex32, Complex64};
use sonium_fastmath::{
    AccuracyProfile, CODY_WAITE_LIMIT, SimdBackend, dispatch_features, exp_i, exp_i_f32,
    exp_i_f32_slice, exp_i_f32_with_derivative_slice, exp_i_f32_with_profile, exp_i_slice,
    exp_i_with_derivative, exp_i_with_derivative_profile, exp_i_with_derivative_slice,
    exp_i_with_derivative_slice_with_profile, exp_i_with_profile, selected_backend, sincos_f32,
    sincos_f32_slice, sincos_f32_with_derivatives, sincos_f64, sincos_f64_slice, sincos_scalar,
    sincos_with_derivatives, sincos_with_profile,
};

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value.is_sign_positive() {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn next_down(value: f64) -> f64 {
    -next_up(-value)
}

fn validation_corpus() -> Vec<f64> {
    let mut values = vec![
        -0.0,
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        CODY_WAITE_LIMIT,
        next_down(CODY_WAITE_LIMIT),
        next_up(CODY_WAITE_LIMIT),
        -CODY_WAITE_LIMIT,
        next_down(-CODY_WAITE_LIMIT),
        next_up(-CODY_WAITE_LIMIT),
        1.0e12,
        -1.0e12,
        1.0e20,
        -1.0e20,
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    // Exact floating-point neighbours of the principal reduction boundaries.
    for k in -128..=128 {
        let center = k as f64 * std::f64::consts::FRAC_PI_2;
        values.extend([next_down(center), center, next_up(center)]);
    }

    // Every normal exponent and four mantissa patterns, in both signs.  This
    // reaches every fixed-point alignment used by the Payne--Hanek fallback.
    let fractions = [0_u64, 1, 1_u64 << 51, (1_u64 << 52) - 1];
    for exponent in 1_u64..=0x7fe {
        for &fraction in &fractions {
            let bits = (exponent << 52) | fraction;
            values.push(f64::from_bits(bits));
            values.push(f64::from_bits(bits | (1_u64 << 63)));
        }
    }

    // Deterministic pseudo-random bit patterns add coverage between the
    // structured exponent/mantissa probes without depending on a RNG crate.
    let mut state = 0x5eed_5eed_d00d_u64;
    for _ in 0..8192 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (state & 0x7fff_ffff_ffff_ffff) | (state & (1_u64 << 63));
        let value = f64::from_bits(bits);
        if value.is_finite() {
            values.push(value);
        }
    }
    values
}

fn assert_f64_close(input: f64, actual: (f64, f64), expected: (f64, f64), label: &str) {
    if expected.0.is_nan() {
        assert!(
            actual.0.is_nan() && actual.1.is_nan(),
            "{label}: NaN at {input:e}"
        );
        return;
    }
    if input == 0.0 {
        assert_eq!(
            actual.0.to_bits(),
            expected.0.to_bits(),
            "{label}: sine zero"
        );
        assert_eq!(
            actual.1.to_bits(),
            expected.1.to_bits(),
            "{label}: cosine zero"
        );
        return;
    }
    let budget = if input.abs() <= CODY_WAITE_LIMIT {
        8.0 * f64::EPSILON
    } else {
        64.0 * f64::EPSILON
    };
    assert!(
        (actual.0 - expected.0).abs() <= budget,
        "{label}: sine at {input:e}: {} vs {} (budget {budget:e})",
        actual.0,
        expected.0
    );
    assert!(
        (actual.1 - expected.1).abs() <= budget,
        "{label}: cosine at {input:e}: {} vs {} (budget {budget:e})",
        actual.1,
        expected.1
    );
}

fn assert_backend_requirements() {
    if std::env::var_os("SONIUM_FASTMATH_REQUIRE_AVX2").is_some() {
        assert!(
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            "SONIUM_FASTMATH_REQUIRE_AVX2=1 but the runner lacks AVX2+FMA"
        );
    }
    if std::env::var_os("SONIUM_FASTMATH_REQUIRE_AVX512").is_some() {
        assert!(
            std::is_x86_feature_detected!("avx512f"),
            "SONIUM_FASTMATH_REQUIRE_AVX512=1 but the runner lacks AVX-512F"
        );
    }
}

#[test]
fn runtime_dispatch_is_observable_and_never_overstates_features() {
    assert_backend_requirements();
    let force_scalar = std::env::var_os("SONIUM_FASTMATH_FORCE_SCALAR")
        .is_some_and(|value| value == "1" || value == "true");
    let force_avx2 = std::env::var_os("SONIUM_FASTMATH_FORCE_AVX2")
        .is_some_and(|value| value == "1" || value == "true");
    if force_avx2 {
        assert!(
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            "SONIUM_FASTMATH_FORCE_AVX2=1 requires AVX2+FMA; refusing a silent scalar run"
        );
    }
    let backend = selected_backend();
    if force_scalar {
        assert_eq!(backend, SimdBackend::Scalar);
    } else if force_avx2 {
        assert_eq!(backend, SimdBackend::Avx2);
    } else {
        match backend {
            SimdBackend::Avx512 => assert!(std::is_x86_feature_detected!("avx512f")),
            SimdBackend::Avx2 => assert!(
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            ),
            SimdBackend::Scalar => assert!(
                !std::is_x86_feature_detected!("avx512f")
                    || !std::is_x86_feature_detected!("avx2")
                    || !std::is_x86_feature_detected!("fma")
            ),
            SimdBackend::Neon | SimdBackend::WasmSimd => {
                panic!("non-x86 backend selected on x86_64")
            }
        }
    }
    let features = dispatch_features();
    assert!(features.contains("arch=x86_64"));
    assert!(features.contains(&format!("force_scalar={force_scalar}")));
    assert!(features.contains(&format!("force_avx2={force_avx2}")));
}

#[test]
fn avx2_matches_scalar_and_libm_over_full_validation_corpus() {
    assert_backend_requirements();
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }

    use sonium_fastmath::avx2::sincos_f64x4;
    use std::arch::x86_64::*;

    let values = validation_corpus();
    for chunk in values.chunks(4) {
        let mut input = [0.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: the feature checks above establish AVX2 + FMA support.
        let vector = unsafe { _mm256_loadu_pd(input.as_ptr()) };
        // SAFETY: the feature checks above establish AVX2 + FMA support.
        let (sin, cos) = unsafe { sincos_f64x4(vector) };
        let mut sins = [0.0; 4];
        let mut coss = [0.0; 4];
        // SAFETY: the output arrays have four f64 lanes.
        unsafe {
            _mm256_storeu_pd(sins.as_mut_ptr(), sin);
            _mm256_storeu_pd(coss.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            let scalar = sincos_scalar(chunk[lane]);
            assert_f64_close(chunk[lane], (sins[lane], coss[lane]), scalar, "AVX2/scalar");
            assert_f64_close(
                chunk[lane],
                (sins[lane], coss[lane]),
                chunk[lane].sin_cos(),
                "AVX2/libm",
            );
        }
    }
}

#[test]
fn avx512_matches_scalar_and_libm_over_full_validation_corpus() {
    assert_backend_requirements();
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    use sonium_fastmath::avx512::sincos_f64x8;
    use std::arch::x86_64::*;

    let values = validation_corpus();
    for chunk in values.chunks(8) {
        let mut input = [0.0; 8];
        input[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: the feature check above establishes AVX-512F support.
        let vector = unsafe { _mm512_loadu_pd(input.as_ptr()) };
        // SAFETY: the feature check above establishes AVX-512F support.
        let (sin, cos) = unsafe { sincos_f64x8(vector) };
        let mut sins = [0.0; 8];
        let mut coss = [0.0; 8];
        // SAFETY: the output arrays have eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(sins.as_mut_ptr(), sin);
            _mm512_storeu_pd(coss.as_mut_ptr(), cos);
        }
        for lane in 0..chunk.len() {
            let scalar = sincos_scalar(chunk[lane]);
            assert_f64_close(
                chunk[lane],
                (sins[lane], coss[lane]),
                scalar,
                "AVX-512/scalar",
            );
            assert_f64_close(
                chunk[lane],
                (sins[lane], coss[lane]),
                chunk[lane].sin_cos(),
                "AVX-512/libm",
            );
        }
    }
}

#[test]
fn avx2_and_avx512_f32_lanes_match_scalar() {
    assert_backend_requirements();
    let input: [f32; 32] = [
        -0.0,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        -1000.0,
        -31.0,
        -std::f32::consts::FRAC_PI_2,
        -1.0,
        -0.25,
        -0.01,
        0.01,
        0.25,
        1.0,
        std::f32::consts::FRAC_PI_2,
        31.0,
        1000.0,
        1.0e12,
        -1.0e12,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        3.0,
        10.0,
        100.0,
        512.0,
        999.0,
        -999.0,
        0.7,
        -0.7,
        42.0,
    ];

    fn check(input: &[f32], actual_sin: &[f32], actual_cos: &[f32], label: &str) {
        for i in 0..input.len() {
            let (expected_sin, expected_cos) = sincos_f32(input[i]);
            if expected_sin.is_nan() {
                assert!(
                    actual_sin[i].is_nan() && actual_cos[i].is_nan(),
                    "{label} lane {i}"
                );
            } else if input[i] == 0.0 {
                assert_eq!(
                    actual_sin[i].to_bits(),
                    expected_sin.to_bits(),
                    "{label} lane {i}"
                );
                assert_eq!(
                    actual_cos[i].to_bits(),
                    expected_cos.to_bits(),
                    "{label} lane {i}"
                );
            } else {
                assert!(
                    (actual_sin[i] - expected_sin).abs() <= 8.0e-6,
                    "{label} sin lane {i}"
                );
                assert!(
                    (actual_cos[i] - expected_cos).abs() <= 8.0e-6,
                    "{label} cos lane {i}"
                );
            }
        }
    }

    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        use sonium_fastmath::avx2::sincos_f32x8;
        use std::arch::x86_64::*;
        for chunk in input.chunks(8) {
            let mut values = [0.0_f32; 8];
            values[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: the feature checks above establish AVX2 + FMA support.
            let vector = unsafe { _mm256_loadu_ps(values.as_ptr()) };
            // SAFETY: the feature checks above establish AVX2 + FMA support.
            let (sin, cos) = unsafe { sincos_f32x8(vector) };
            let mut sins = [0.0_f32; 8];
            let mut coss = [0.0_f32; 8];
            // SAFETY: the output arrays have eight f32 lanes.
            unsafe {
                _mm256_storeu_ps(sins.as_mut_ptr(), sin);
                _mm256_storeu_ps(coss.as_mut_ptr(), cos);
            }
            check(&values[..chunk.len()], &sins, &coss, "AVX2 f32");
        }
    }

    if std::is_x86_feature_detected!("avx512f") {
        use sonium_fastmath::avx512::sincos_f32x16;
        use std::arch::x86_64::*;
        for chunk in input.chunks(16) {
            let mut values = [0.0_f32; 16];
            values[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: the feature check above establishes AVX-512F support.
            let vector = unsafe { _mm512_loadu_ps(values.as_ptr()) };
            // SAFETY: the feature check above establishes AVX-512F support.
            let (sin, cos) = unsafe { sincos_f32x16(vector) };
            let mut sins = [0.0_f32; 16];
            let mut coss = [0.0_f32; 16];
            // SAFETY: the output arrays have sixteen f32 lanes.
            unsafe {
                _mm512_storeu_ps(sins.as_mut_ptr(), sin);
                _mm512_storeu_ps(coss.as_mut_ptr(), cos);
            }
            check(&values[..chunk.len()], &sins, &coss, "AVX-512 f32");
        }
    }
}

#[test]
fn public_api_corpus_covers_tails_phasors_and_derivatives() {
    assert_backend_requirements();
    let input: Vec<f64> = (0..37)
        .map(|i| -17.0 + i as f64 * 0.91)
        .chain([CODY_WAITE_LIMIT + 1.0, -1.0e20])
        .collect();

    let mut sins = vec![0.0; input.len()];
    let mut coss = vec![0.0; input.len()];
    let mut values = vec![Complex64::default(); input.len()];
    let mut derivatives = vec![Complex64::default(); input.len()];
    sincos_f64_slice(&input, &mut sins, &mut coss);
    exp_i_slice(&input, &mut values);
    exp_i_with_derivative_slice(&input, &mut values, &mut derivatives);
    exp_i_with_derivative_slice_with_profile(
        &input,
        &mut values,
        &mut derivatives,
        AccuracyProfile::Fast,
    );

    for i in 0..input.len() {
        let expected = sincos_scalar(input[i]);
        assert_f64_close(input[i], (sins[i], coss[i]), expected, "public slice");
        assert_f64_close(input[i], sincos_f64(input[i]), expected, "public scalar");
        assert_f64_close(
            input[i],
            sincos_with_profile(input[i], AccuracyProfile::Fast),
            expected,
            "profile scalar",
        );
        let derivatives_value = sincos_with_derivatives(input[i]);
        assert_eq!(derivatives_value.d_sin, derivatives_value.cos);
        assert_eq!(derivatives_value.d_cos, -derivatives_value.sin);
        assert_eq!(
            exp_i(input[i]),
            exp_i_with_profile(input[i], AccuracyProfile::Fast)
        );
        assert_eq!(
            exp_i_with_profile(input[i], AccuracyProfile::Reference),
            Complex64::new(input[i].cos(), input[i].sin())
        );
        let (phasor, derivative) = exp_i_with_derivative(input[i]);
        assert_eq!(
            exp_i_with_derivative_profile(input[i], AccuracyProfile::Reference),
            (
                Complex64::new(input[i].cos(), input[i].sin()),
                Complex64::new(-input[i].sin(), input[i].cos())
            )
        );
        assert_f64_close(
            input[i],
            (values[i].im, values[i].re),
            expected,
            "public phasor",
        );
        assert!((derivatives[i].re + values[i].im).abs() <= 8.0 * f64::EPSILON);
        assert!((derivatives[i].im - values[i].re).abs() <= 8.0 * f64::EPSILON);
        assert_f64_close(input[i], (phasor.im, phasor.re), expected, "scalar phasor");
        assert_eq!(derivative, Complex64::new(-phasor.im, phasor.re));
    }

    let input_f32: Vec<f32> = (0..39).map(|i| -17.0 + i as f32 * 0.91).collect();
    let mut f32_sins = vec![0.0; input_f32.len()];
    let mut f32_coss = vec![0.0; input_f32.len()];
    let mut f32_values = vec![Complex32::default(); input_f32.len()];
    let mut f32_derivatives = vec![Complex32::default(); input_f32.len()];
    sincos_f32_slice(&input_f32, &mut f32_sins, &mut f32_coss);
    exp_i_f32_slice(&input_f32, &mut f32_values);
    exp_i_f32_with_derivative_slice(&input_f32, &mut f32_values, &mut f32_derivatives);
    for i in 0..input_f32.len() {
        let (expected_sin, expected_cos) = sincos_f32(input_f32[i]);
        assert_eq!(f32_values[i], exp_i_f32(input_f32[i]));
        assert_eq!(
            exp_i_f32_with_profile(input_f32[i], AccuracyProfile::Reference),
            Complex32::new(input_f32[i].cos(), input_f32[i].sin())
        );
        assert!((f32_sins[i] - expected_sin).abs() <= 8.0e-6);
        assert!((f32_coss[i] - expected_cos).abs() <= 8.0e-6);
        let derivatives = sincos_f32_with_derivatives(input_f32[i]);
        assert_eq!(derivatives.2, derivatives.1);
        assert_eq!(derivatives.3, -derivatives.0);
        assert_eq!(
            f32_derivatives[i],
            Complex32::new(-expected_sin, expected_cos)
        );
    }
}

#[test]
fn every_public_phase_entry_point_preserves_tail_and_special_contracts() {
    assert_backend_requirements();
    let all_f64 = [
        -0.0,
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        -std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2,
        CODY_WAITE_LIMIT,
        CODY_WAITE_LIMIT + 1.0,
        -1.0e20,
        1.0e20,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for length in 0..=all_f64.len() {
        let input = &all_f64[..length];
        let mut sins = vec![0.0; length];
        let mut coss = vec![0.0; length];
        let mut values = vec![Complex64::default(); length];
        let mut derivatives = vec![Complex64::default(); length];
        sincos_f64_slice(input, &mut sins, &mut coss);
        exp_i_slice(input, &mut values);
        exp_i_with_derivative_slice(input, &mut values, &mut derivatives);
        exp_i_with_derivative_slice_with_profile(
            input,
            &mut values,
            &mut derivatives,
            AccuracyProfile::Fast,
        );
        for (i, &phase) in input.iter().enumerate() {
            let expected = sincos_scalar(phase);
            assert_f64_close(phase, (sins[i], coss[i]), expected, "f64 tail slice");
            let (sin, cos) = sincos_with_profile(phase, AccuracyProfile::Fast);
            assert_f64_close(phase, (sin, cos), expected, "f64 profile");
            let derivatives_value = sincos_with_derivatives(phase);
            assert_f64_close(
                phase,
                (derivatives_value.sin, derivatives_value.cos),
                expected,
                "f64 value-plus-derivative",
            );
            let expected_phasor = Complex64::new(expected.1, expected.0);
            if expected.0.is_nan() {
                assert!(values[i].re.is_nan() && values[i].im.is_nan());
            } else {
                assert_f64_close(phase, (values[i].im, values[i].re), expected, "f64 phasor");
                assert_eq!(derivatives[i], Complex64::new(-values[i].im, values[i].re));
            }
            if expected.0.is_nan() {
                let exp = exp_i(phase);
                let exp_profile = exp_i_with_profile(phase, AccuracyProfile::Fast);
                assert!(
                    exp.re.is_nan()
                        && exp.im.is_nan()
                        && exp_profile.re.is_nan()
                        && exp_profile.im.is_nan()
                );
            } else {
                assert_eq!(
                    exp_i(phase),
                    exp_i_with_profile(phase, AccuracyProfile::Fast)
                );
                assert_eq!(exp_i(phase), expected_phasor);
                assert_eq!(
                    exp_i_with_derivative(phase),
                    exp_i_with_derivative_profile(phase, AccuracyProfile::Fast)
                );
            }
        }
    }

    let all_f32 = [
        -0.0_f32,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        -std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_2,
        CODY_WAITE_LIMIT as f32,
        (CODY_WAITE_LIMIT + 1.0) as f32,
        -1.0e20,
        1.0e20,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for length in 0..=all_f32.len() {
        let input = &all_f32[..length];
        let mut sins = vec![0.0; length];
        let mut coss = vec![0.0; length];
        let mut values = vec![Complex32::default(); length];
        let mut derivatives = vec![Complex32::default(); length];
        sincos_f32_slice(input, &mut sins, &mut coss);
        exp_i_f32_slice(input, &mut values);
        exp_i_f32_with_derivative_slice(input, &mut values, &mut derivatives);
        for (i, &phase) in input.iter().enumerate() {
            let (expected_sin, expected_cos) = sincos_f32(phase);
            if expected_sin.is_nan() {
                assert!(sins[i].is_nan() && coss[i].is_nan());
                assert!(values[i].re.is_nan() && values[i].im.is_nan());
            } else {
                assert!((sins[i] - expected_sin).abs() <= 8.0e-6);
                assert!((coss[i] - expected_cos).abs() <= 8.0e-6);
                assert_eq!(values[i], exp_i_f32(phase));
                assert_eq!(derivatives[i], Complex32::new(-values[i].im, values[i].re));
                let derivative = sincos_f32_with_derivatives(phase);
                assert_eq!(derivative.2, derivative.1);
                assert_eq!(derivative.3, -derivative.0);
                assert_eq!(
                    exp_i_f32_with_profile(phase, AccuracyProfile::Fast),
                    exp_i_f32(phase)
                );
            }
        }
    }
}
