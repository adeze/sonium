//! Independent failure-admission matrix for the generic solver facade.
//!
//! Every supported scalar family must classify the same malformed public
//! inputs with the same typed `LinearSolveError` variant before entering a
//! solver iteration.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, Zero};
use sonium_solvers::{
    ComplexField, CsrMatrix, LinearSolveConfig, LinearSolveError, LinearSolverType, solve,
};

fn scalar<T: ComplexField>(value: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_re_im(
        T::Real::from_f64(value).expect("finite scalar fixture"),
        T::Real::zero(),
    )
}

fn nan_scalar<T: ComplexField>() -> T
where
    T::Real: FromPrimitive,
{
    T::from_re_im(
        T::Real::from_f64(f64::NAN).expect("NaN scalar fixture"),
        T::Real::zero(),
    )
}

fn config() -> LinearSolveConfig {
    LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        tolerance: 1.0e-6,
        max_iterations: 16,
        ..LinearSolveConfig::default()
    }
}

fn run_scalar<T: ComplexField>() -> usize
where
    T::Real: FromPrimitive + Sync + Send,
    T: Sync + Send,
{
    let matrix = CsrMatrix::<T>::identity(2);

    let dimension = solve(&matrix, &[scalar::<T>(1.0)], &config())
        .expect_err("short RHS must be rejected before iteration");
    assert!(
        matches!(
            dimension,
            LinearSolveError::DimensionMismatch {
                operation: "linear solve rhs",
                expected: 2,
                actual: 1,
            }
        ),
        "unexpected dimension error: {dimension:?}"
    );

    let nonfinite = solve(&matrix, &[nan_scalar::<T>(), scalar::<T>(1.0)], &config())
        .expect_err("non-finite RHS must be rejected before iteration");
    assert!(
        matches!(
            nonfinite,
            LinearSolveError::NonFiniteInput {
                operation: "rhs",
                index: 0,
            }
        ),
        "unexpected non-finite error: {nonfinite:?}"
    );

    let mut malformed = CsrMatrix::<T>::identity(2);
    malformed.row_ptrs = vec![0, 2, 1];
    let malformed = solve(&malformed, &[scalar::<T>(1.0), scalar::<T>(1.0)], &config())
        .expect_err("non-monotone CSR row pointers must be rejected before iteration");
    assert!(
        matches!(malformed, LinearSolveError::InvalidConfig(ref message) if message.contains("monotonically")),
        "unexpected malformed CSR error: {malformed:?}"
    );

    let mut out_of_range = CsrMatrix::<T>::identity(1);
    out_of_range.col_indices[0] = 1;
    let out_of_range = solve(&out_of_range, &[scalar::<T>(1.0)], &config())
        .expect_err("out-of-range CSR columns must be rejected before iteration");
    assert!(
        matches!(out_of_range, LinearSolveError::InvalidConfig(ref message) if message.contains("out of range")),
        "unexpected out-of-range CSR error: {out_of_range:?}"
    );

    let mut nonfinite_matrix = CsrMatrix::<T>::identity(1);
    nonfinite_matrix.values[0] = nan_scalar::<T>();
    let nonfinite_matrix = solve(&nonfinite_matrix, &[scalar::<T>(1.0)], &config())
        .expect_err("non-finite matrix values must be rejected before iteration");
    assert!(
        matches!(
            nonfinite_matrix,
            LinearSolveError::NonFiniteInput {
                operation: "matrix",
                index: 0,
            }
        ),
        "unexpected non-finite matrix error: {nonfinite_matrix:?}"
    );

    let invalid_config = solve(
        &matrix,
        &[scalar::<T>(1.0), scalar::<T>(-2.0)],
        &LinearSolveConfig {
            max_iterations: 0,
            ..config()
        },
    )
    .expect_err("zero iteration configuration must be rejected");
    assert!(
        matches!(invalid_config, LinearSolveError::InvalidConfig(_)),
        "unexpected configuration error: {invalid_config:?}"
    );

    6
}

pub fn run_compatibility_failure_matrix() -> usize {
    run_scalar::<f32>()
        + run_scalar::<f64>()
        + run_scalar::<Complex32>()
        + run_scalar::<Complex64>()
}

#[test]
fn compatibility_failure_matrix_across_scalar_families() {
    assert_eq!(run_compatibility_failure_matrix(), 24);
}
