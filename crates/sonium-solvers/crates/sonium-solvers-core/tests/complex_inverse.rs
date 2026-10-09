use num_complex::{Complex32, Complex64};
use sonium_solvers_core::traits::ComplexField;

#[test]
fn scaled_complex32_inverse_preserves_tiny_finite_reciprocal() {
    let value = Complex32::new(1.0e-31, -2.0e-31);
    let inverse = ComplexField::inv(&value);
    assert!(inverse.re.is_finite());
    assert!(inverse.im.is_finite());
    assert!((value * inverse - Complex32::new(1.0, 0.0)).norm() < 1.0e-6);
}

#[test]
fn scaled_complex64_inverse_preserves_tiny_finite_reciprocal() {
    let value = Complex64::new(1.0e-31, -2.0e-31);
    let inverse = ComplexField::inv(&value);
    assert!(inverse.re.is_finite());
    assert!(inverse.im.is_finite());
    assert!((value * inverse - Complex64::new(1.0, 0.0)).norm() < 1.0e-14);
}
