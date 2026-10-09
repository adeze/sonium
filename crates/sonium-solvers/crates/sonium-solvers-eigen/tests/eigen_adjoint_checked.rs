use sonium_solvers_core::traits::SolverError;
use sonium_solvers_eigen::{
    generalized_eigenvalue_sensitivity, generalized_eigenvector_sensitivity,
};
use sonium_solvers_sparse::CsrMatrix;

#[test]
fn eigenvalue_sensitivity_rejects_non_finite_eigenpair_input() {
    let identity = CsrMatrix::identity(2);
    let error = generalized_eigenvalue_sensitivity(
        f64::NAN,
        &[1.0, 0.0],
        &[1.0, 0.0],
        &identity,
        None::<&CsrMatrix<f64>>,
        &identity,
    )
    .expect_err("non-finite eigenvalue must be rejected before callbacks");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn eigenvector_sensitivity_rejects_non_finite_vector_before_singular_solve() {
    let identity = CsrMatrix::identity(2);
    let error = generalized_eigenvector_sensitivity(
        1.0,
        &[f64::NAN, 0.0],
        &[1.0, 0.0],
        &identity,
        None::<&CsrMatrix<f64>>,
        &identity,
        |_| panic!("the singular solve must not be called"),
    )
    .expect_err("non-finite eigenvector must be rejected before callbacks");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}
