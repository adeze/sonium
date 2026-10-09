use num_complex::{Complex32, Complex64};
use num_traits::FromPrimitive;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverResultContract, SolverStatus,
};
use sonium_solvers_direct::{
    DenseDirectConfig, DirectFactorization, DirectSolveResult, MatrixClass, SparseDirectConfig,
    SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers_sparse::CsrMatrix;

fn assert_unchecked_result<T: ComplexField>(result: &DirectSolveResult<T>, expected: &[T]) {
    assert_eq!(result.solution().len(), expected.len());
    assert_eq!(result.iterations, 0);
    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
    assert!(result.residual.is_none());
    assert!(result.backward_error.is_none());
    assert!(
        result
            .solution()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - *expected).norm()
                < T::Real::from_f64(5.0e-4).unwrap())
    );
    result
        .validate_contract()
        .expect("finite direct result must satisfy the shared contract");
}

#[test]
fn dense_direct_result_contract_covers_all_scalar_families() {
    let real_matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f32, 1.0, 1.0, 3.0]);
    let real_rhs = real_matrix.matvec(&[1.0_f32, 2.0]);
    let real_factor = factorize_dense(&real_matrix, &DenseDirectConfig::default()).unwrap();
    let real_result = real_factor.solve_with_result(&real_rhs).unwrap();
    assert_unchecked_result(&real_result, &[1.0_f32, 2.0]);

    let real_matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
    let real_rhs = real_matrix.matvec(&[1.0_f64, 2.0]);
    let real_factor = factorize_dense(&real_matrix, &DenseDirectConfig::default()).unwrap();
    let real_result = real_factor.solve_with_result(&real_rhs).unwrap();
    assert_unchecked_result(&real_result, &[1.0_f64, 2.0]);

    let complex_matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex32::new(4.0, 0.5),
            Complex32::new(1.0, -0.25),
            Complex32::new(1.0, 0.25),
            Complex32::new(3.0, -0.5),
        ],
    );
    let complex_expected = [Complex32::new(1.0, 0.5), Complex32::new(2.0, -0.25)];
    let complex_rhs = complex_matrix.matvec(&complex_expected);
    let complex_factor = factorize_dense(&complex_matrix, &DenseDirectConfig::default()).unwrap();
    let complex_result = complex_factor.solve_with_result(&complex_rhs).unwrap();
    assert_unchecked_result(&complex_result, &complex_expected);

    let complex_matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(1.0, 0.25),
            Complex64::new(3.0, -0.5),
        ],
    );
    let complex_expected = [Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)];
    let complex_rhs = complex_matrix.matvec(&complex_expected);
    let complex_factor = factorize_dense(&complex_matrix, &DenseDirectConfig::default()).unwrap();
    let complex_result = complex_factor.solve_with_result(&complex_rhs).unwrap();
    assert_unchecked_result(&complex_result, &complex_expected);
}

#[test]
fn sparse_direct_result_contract_covers_real_and_complex_lu() {
    let real_matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0_f64), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
    );
    let real_expected = [1.0_f64, 2.0];
    let real_rhs = real_matrix.matvec(&real_expected);
    let real_factor = factorize_sparse(
        &real_matrix,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let real_result = real_factor.solve_with_result(&real_rhs).unwrap();
    assert_unchecked_result(&real_result, &real_expected);

    let complex_matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(4.0, 0.5)),
            (0, 1, Complex64::new(1.0, -0.25)),
            (1, 0, Complex64::new(1.0, 0.25)),
            (1, 1, Complex64::new(3.0, -0.5)),
        ],
    );
    let complex_expected = [Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)];
    let complex_rhs = complex_matrix.matvec(&complex_expected);
    let complex_factor = factorize_sparse(
        &complex_matrix,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let complex_result = complex_factor.solve_with_result(&complex_rhs).unwrap();
    assert_unchecked_result(&complex_result, &complex_expected);
}

#[test]
fn direct_result_contract_covers_cholesky_and_rectangular_qr() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
    let expected = [1.0_f64, 2.0];
    let rhs = matrix.matvec(&expected);
    let cholesky = factorize_dense(
        &matrix,
        &sonium_solvers_direct::DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: sonium_solvers_direct::DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .unwrap();
    let cholesky_result = cholesky.solve_with_result(&rhs).unwrap();
    assert_unchecked_result(&cholesky_result, &expected);

    let rectangular = DenseMatrix::from_row_major_vec(3, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let expected = [1.0_f64, 2.0];
    let rhs = rectangular.matvec(&expected);
    let qr = factorize_dense(
        &rectangular,
        &sonium_solvers_direct::DenseDirectConfig {
            matrix_class: MatrixClass::RectangularFullRank,
            method: sonium_solvers_direct::DenseDirectMethod::Qr,
            ..Default::default()
        },
    )
    .unwrap();
    let qr_result = qr.solve_with_result(&rhs).unwrap();
    assert_unchecked_result(&qr_result, &expected);

    let sparse = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0_f64), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
    );
    let sparse_factor = factorize_sparse(
        &sparse,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let sparse_rhs = sparse.matvec(&[1.0_f64, 2.0]);
    let sparse_result = sparse_factor.solve_with_result(&sparse_rhs).unwrap();
    assert_unchecked_result(&sparse_result, &[1.0_f64, 2.0]);
}

#[test]
fn direct_result_contract_covers_indefinite_multifrontal_wrappers() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![0.0_f64, 1.0, 1.0, 0.0]);
    let expected = [2.0_f64, 3.0];
    let rhs = matrix.matvec(&expected);
    let dense = factorize_dense(
        &matrix,
        &sonium_solvers_direct::DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: sonium_solvers_direct::DenseDirectMethod::Ldlt,
            ..Default::default()
        },
    )
    .unwrap();
    let dense_result = dense.solve_with_result(&rhs).unwrap();
    assert_unchecked_result(&dense_result, &expected);

    let sparse = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 1.0_f64), (1, 0, 1.0)]);
    let sparse_factor = factorize_sparse(
        &sparse,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: SparseDirectMethod::MultifrontalLdlt,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let sparse_rhs = sparse.matvec(&expected);
    let sparse_result = sparse_factor.solve_with_result(&sparse_rhs).unwrap();
    assert_unchecked_result(&sparse_result, &expected);
}

#[test]
fn direct_result_accuracy_check_is_explicit_and_fail_closed() {
    let result = DirectSolveResult::from_solution(vec![1.0_f64, 2.0]);
    let checked = result
        .with_accuracy_check(2.0e-15, 3.0e-16, 1.0e-12)
        .unwrap();
    assert_eq!(checked.solver_status(), SolverStatus::Converged);
    assert_eq!(checked.accuracy_status(), AccuracyStatus::Attained);
    checked.validate_contract().unwrap();

    let failed = DirectSolveResult::from_solution(vec![1.0_f64, 2.0])
        .with_accuracy_check(1.0, 1.0e-3, 1.0e-6)
        .unwrap();
    assert_eq!(failed.solver_status(), SolverStatus::NumericalError);
    assert_eq!(failed.accuracy_status(), AccuracyStatus::NotAttained);
    failed.validate_contract().unwrap();

    assert!(
        DirectSolveResult::from_solution(vec![1.0_f64])
            .with_accuracy_check(f64::NAN, 0.0, 1.0e-6)
            .is_err()
    );
    assert!(
        DirectSolveResult::from_solution(vec![1.0_f64])
            .with_accuracy_check(0.0, 0.0, -1.0)
            .is_err()
    );
}

#[test]
fn direct_result_contract_rejects_contradictory_public_metadata() {
    let result = DirectSolveResult {
        x: vec![1.0_f64],
        iterations: 0,
        residual: None,
        backward_error: None,
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
    };
    assert!(result.validate_contract().is_err());

    let result = DirectSolveResult {
        x: vec![f64::NAN],
        iterations: 0,
        residual: Some(0.0),
        backward_error: Some(0.0),
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
    };
    assert!(result.validate_contract().is_err());
}
