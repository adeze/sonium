//! Scalar-family admission coverage for relaxed sparse-LU preconditioner pivots.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, One, ToPrimitive};
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;

fn real<T: ComplexField>(value: f64) -> T {
    T::from_real(T::Real::from_f64(value).expect("fixture value representable"))
}

fn check<T: ComplexField + FromPrimitive>(pivot: f64, tolerance: f64) {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![real::<T>(pivot)]);
    let rhs = [real::<T>(1.0)];
    let expected = rhs[0] * real::<T>(pivot).inv();
    let factor = SparseLuFactorization::factorize_for_preconditioner(&matrix)
        .expect("preconditioner mode should admit finite tiny pivot");
    let solution = factor.solve(&rhs).expect("relaxed pivot solve");
    let relative_error = (solution[0] - expected).norm() / expected.norm().max(T::Real::one());
    assert!(
        relative_error.to_f64().expect("test error representable") <= tolerance,
        "relative error {relative_error:?} exceeds {tolerance}"
    );
}

#[test]
fn relaxed_preconditioner_pivot_supports_f32() {
    check::<f32>(1.0e-31, 3.0e-6);
}

#[test]
fn relaxed_preconditioner_pivot_supports_f64() {
    check::<f64>(1.0e-31, 3.0e-13);
}

#[test]
fn relaxed_preconditioner_pivot_supports_complex32() {
    check::<Complex32>(1.0e-31, 3.0e-6);
}

#[test]
fn relaxed_preconditioner_complex32_tiny_imaginary_pivot() {
    let pivot = Complex32::new(1.0e-31, -2.0e-31);
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![pivot]);
    let factor = SparseLuFactorization::factorize_for_preconditioner(&matrix)
        .expect("complex tiny pivot should be admitted");
    let rhs = [Complex32::new(1.0, 0.5)];
    let expected = rhs[0] * ComplexField::inv(&pivot);
    let solution = factor.solve(&rhs).expect("complex tiny pivot solve");
    let error = (solution[0] - expected).norm();
    assert!(error < 3.0e24, "absolute error {error:e} too large");
    let relative = error / expected.norm().max(1.0);
    assert!(relative < 3.0e-6, "relative error {relative:e} too large");
}

#[test]
fn relaxed_preconditioner_pivot_supports_complex64() {
    check::<Complex64>(1.0e-31, 3.0e-13);
}
