use half::f16;
use num_complex::{Complex32, Complex64};
use sonium_solvers_core::traits::ComplexField;

/// The checked wrappers across the solvers stack validate vectors with
/// `value.norm().is_finite()`. That check must accept every finite scalar,
/// including top-of-range magnitudes where a naive `sqrt(re*re + im*im)`
/// would overflow to infinity and raise a false positive.
#[test]
fn top_of_range_finite_scalars_pass_norm_finiteness() {
    assert!(f64::MAX.norm().is_finite());
    assert!(f32::MAX.norm().is_finite());
    assert!(f16::MAX.norm().is_finite());

    // Magnitudes above sqrt(MAX) where squaring first would overflow.
    assert!(1.0e200f64.norm().is_finite());
    assert!(1.0e30f32.norm().is_finite());
    assert!(Complex64::new(1.0e200, -1.0e200).norm().is_finite());
    assert!(Complex32::new(1.0e30, 1.0e30).norm().is_finite());
    assert!(Complex64::new(f64::MAX, 0.0).norm().is_finite());
}

#[test]
fn genuinely_non_finite_scalars_fail_norm_finiteness() {
    assert!(!f64::INFINITY.norm().is_finite());
    assert!(!f64::NAN.norm().is_finite());
    assert!(!f32::NEG_INFINITY.norm().is_finite());
    assert!(!Complex64::new(1.0, f64::INFINITY).norm().is_finite());
    assert!(!Complex64::new(f64::NAN, 0.0).norm().is_finite());
    assert!(!Complex32::new(0.0, f32::NAN).norm().is_finite());
    assert!(!f16::INFINITY.norm().is_finite());
}
