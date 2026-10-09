//! Metamorphic properties for the public fastmath APIs.

use num_complex::Complex64;
use sonium_fastmath::{
    AccuracyProfile, CODY_WAITE_LIMIT, exp_i_f32_with_profile, exp_i_with_derivative_profile,
    exp_i_with_derivative_slice_with_profile, exp_i_with_profile, sincos_f32, sincos_f64,
};

#[test]
fn public_reference_phasor_profile_preserves_value_and_derivative_convention() {
    let phases = [-1.0, -0.0, 0.7, 1.0e20];
    let mut values = [Complex64::default(); 4];
    let mut derivatives = [Complex64::default(); 4];
    exp_i_with_derivative_slice_with_profile(
        &phases,
        &mut values,
        &mut derivatives,
        AccuracyProfile::Reference,
    );

    for ((&phase, value), derivative) in phases.iter().zip(values).zip(derivatives) {
        let (sin, cos) = phase.sin_cos();
        assert_eq!(exp_i_with_profile(phase, AccuracyProfile::Reference), value);
        assert_eq!(value, Complex64::new(cos, sin));
        assert_eq!(derivative, Complex64::new(-sin, cos));
        assert_eq!(
            exp_i_with_derivative_profile(phase, AccuracyProfile::Reference),
            (value, derivative)
        );
    }

    let phase_f32 = 1.0e10_f32;
    let (sin, cos) = phase_f32.sin_cos();
    assert_eq!(
        exp_i_f32_with_profile(phase_f32, AccuracyProfile::Reference),
        num_complex::Complex32::new(cos, sin)
    );
}

#[test]
fn pythagorean_identity_holds_for_public_scalar_apis() {
    for &x in &[
        -CODY_WAITE_LIMIT + 1.0,
        -999.0_f64,
        -17.25,
        -0.0,
        0.0,
        0.17,
        17.25,
        999.0,
        CODY_WAITE_LIMIT - 1.0,
    ] {
        let (sin, cos) = sincos_f64(x);
        assert!(
            (sin * sin + cos * cos - 1.0).abs() < 8.0e-15,
            "f64 identity x={x:e}"
        );
    }

    for &x in &[-999.0_f32, -17.25, -0.0, 0.0, 0.17, 17.25, 999.0] {
        let (sin, cos) = sincos_f32(x);
        assert!(
            (sin * sin + cos * cos - 1.0).abs() < 3.0e-6,
            "f32 identity x={x:e}"
        );
    }
}

#[test]
fn sine_is_odd_and_cosine_is_even_across_the_fast_domain() {
    for &x in &[
        0.0_f64,
        1.0e-300,
        0.17,
        1.0,
        17.25,
        999.0,
        CODY_WAITE_LIMIT - 1.0,
    ] {
        let (sin, cos) = sincos_f64(x);
        let (negative_sin, negative_cos) = sincos_f64(-x);
        assert!((negative_sin + sin).abs() < 2.0e-15, "odd sine x={x:e}");
        assert!((negative_cos - cos).abs() < 2.0e-15, "even cosine x={x:e}");
    }
}

#[test]
fn periodicity_is_stable_inside_the_documented_reduction_domain() {
    let period = 2.0 * std::f64::consts::PI;
    for i in 0..257 {
        let x = -900.0 + i as f64 * 7.0;
        let (sin, cos) = sincos_f64(x);
        let (shifted_sin, shifted_cos) = sincos_f64(x + period);
        assert!((shifted_sin - sin).abs() < 3.0e-12, "periodic sine x={x:e}");
        assert!(
            (shifted_cos - cos).abs() < 3.0e-12,
            "periodic cosine x={x:e}"
        );
    }
}

#[test]
fn f32_special_values_preserve_signed_zero_and_nan_contracts() {
    let (negative_zero_sin, negative_zero_cos) = sincos_f32(-0.0);
    assert_eq!(negative_zero_sin.to_bits(), (-0.0_f32).to_bits());
    assert_eq!(negative_zero_cos, 1.0);

    for &x in &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let (sin, cos) = sincos_f32(x);
        assert!(sin.is_nan() && cos.is_nan(), "special x={x}");
    }
}

#[test]
fn f64_negative_zero_preserves_the_sine_sign() {
    let (sin, cos) = sincos_f64(-0.0);
    assert_eq!(sin.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(cos, 1.0);
}

#[test]
fn f64_special_values_preserve_nan_contracts() {
    for &x in &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let (sin, cos) = sincos_f64(x);
        assert!(sin.is_nan() && cos.is_nan(), "special x={x}");
    }
}
