use std::panic::{AssertUnwindSafe, catch_unwind};

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use sonium_solvers_direct::{
    BlrError, CholeskyError, DenseDirectConfig, DenseDirectError, DirectFactorization, LuError,
    MatrixClass, QrError, SparseCholeskyError, SparseDirectConfig, SparseDirectError,
    SparseDirectMethod, SparseFactorization, SparseLuError, SparseLuFactorization,
    SparseRefinementStopReason, cholesky_factorize, dense_backward_error, factorize_dense,
    factorize_sparse,
};
use sonium_solvers_sparse::CsrMatrix;

pub fn dense_facade_normalizes_dimensions_and_non_finite_inputs() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            ..Default::default()
        },
    )
    .unwrap();

    let mut output = vec![9.0, 9.0];
    assert!(matches!(
        factor.solve_into(&[1.0], &mut output),
        Err(DenseDirectError::RhsDimension {
            expected: 2,
            actual: 1
        })
    ));
    assert_eq!(output, vec![9.0, 9.0]);
    assert!(matches!(
        factor.solve_into(&[f64::NAN, 1.0], &mut output),
        Err(DenseDirectError::NonFiniteRhs { index: 0 })
    ));
    assert!(matches!(
        factor.solve_into(&[1.0, 1.0], &mut [0.0]),
        Err(DenseDirectError::SolutionDimension {
            expected: 2,
            actual: 1
        })
    ));

    let non_finite = DenseMatrix::from_row_major_vec(1, 1, vec![f64::NAN]);
    assert!(matches!(
        factorize_dense(
            &non_finite,
            &DenseDirectConfig {
                matrix_class: MatrixClass::SymmetricPositiveDefinite,
                ..Default::default()
            }
        ),
        Err(DenseDirectError::NonFiniteMatrix { index: 0 })
    ));
}

pub fn dense_backward_error_rejects_non_finite_metadata() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![f64::MAX]);
    assert!(matches!(
        dense_backward_error(&matrix, &[f64::MAX], &[f64::MAX]),
        Err(DenseDirectError::NonFiniteMetadata { .. })
    ));
    assert!(matches!(
        dense_backward_error(&DenseMatrix::eye(1), &[f64::NAN], &[1.0]),
        Err(DenseDirectError::NonFiniteSolution { index: 0 })
    ));
}

pub fn sparse_facade_normalizes_checked_solve_and_preserves_destination() {
    let matrix = CsrMatrix::from_dense(&DenseMatrix::<f64>::eye(2), 0.0);
    let factor = factorize_sparse(
        &matrix,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert!(matches!(
        factor.solve(&[1.0]),
        Err(SparseDirectError::RhsDimension {
            expected: 2,
            actual: 1
        })
    ));
    let mut output = vec![7.0, 7.0];
    assert!(matches!(
        factor.solve_into(&[f64::INFINITY, 1.0], &mut output),
        Err(SparseDirectError::NonFiniteRhs { index: 0 })
    ));
    assert_eq!(output, vec![7.0, 7.0]);
    assert!(matches!(
        factor.solve_into(&[1.0, 1.0], &mut [0.0]),
        Err(SparseDirectError::SolutionDimension {
            expected: 2,
            actual: 1
        })
    ));

    let mut non_finite = CsrMatrix::identity(1);
    non_finite.values[0] = f64::NAN;
    assert!(matches!(
        factorize_sparse(
            &non_finite,
            &SparseDirectConfig {
                matrix_class: MatrixClass::SymmetricPositiveDefinite,
                ..Default::default()
            },
            None,
        ),
        Err(SparseDirectError::NonFiniteMatrix { index: 0 })
    ));
}

pub fn sparse_direct_facade_many_rhs_preserves_checked_contract() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0_f64), (1, 1, 3.0_f64)]);
    let factor = factorize_sparse(&matrix, &SparseDirectConfig::default(), None).unwrap();
    let mut output = vec![7.0; 4];
    factor
        .solve_many_into(&[2.0, 4.0, 3.0, 6.0], &mut output, 2)
        .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 1.0, 2.0]);

    let before = output.clone();
    assert!(matches!(
        factor.solve_many_into(&[2.0, f64::NAN, 3.0, 6.0], &mut output, 2),
        Err(SparseDirectError::NonFiniteRhs { index: 1 })
    ));
    assert_eq!(output, before);
    assert!(matches!(
        factor.solve_many_into(&[], &mut [], usize::MAX),
        Err(SparseDirectError::RhsSizeOverflow {
            dimension: 2,
            nrhs: usize::MAX
        })
    ));
}

pub fn dense_factorization_many_rhs_preserves_checked_contract() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            ..Default::default()
        },
    )
    .unwrap();
    let mut output = vec![7.0; 4];
    factor
        .solve_many_into(&[6.0, 9.0, 7.0, 5.0], &mut output, 2)
        .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 2.0, 1.0]);

    let before = output.clone();
    assert!(matches!(
        factor.solve_many_into(&[6.0, f64::NAN, 7.0, 5.0], &mut output, 2),
        Err(DenseDirectError::NonFiniteRhs { index: 1 })
    ));
    assert_eq!(output, before);
    assert!(matches!(
        factor.solve_many_into(&[], &mut [], usize::MAX),
        Err(DenseDirectError::SizeOverflow {
            dimension: 2,
            nrhs: usize::MAX
        })
    ));
}

pub fn sparse_multifrontal_adapter_contains_legacy_panicking_solve_path() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 1.0_f64), (1, 0, 1.0_f64)]);
    let factor = factorize_sparse(
        &matrix,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: SparseDirectMethod::MultifrontalLdlt,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| factor.solve(&[f64::NAN, 1.0])));
    let result = result.expect("checked facade must not panic on invalid RHS");
    assert!(matches!(
        result,
        Err(SparseDirectError::NonFiniteRhs { index: 0 })
    ));
}

pub fn sparse_compatibility_uses_checked_direct_contract_and_matches_reference() {
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0_f64), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 3.0)],
    );
    let factor = factorize_sparse(
        &matrix,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let solution =
        <SparseFactorization<f64> as DirectFactorization<f64>>::solve(&factor, &[1.0, 7.0])
            .unwrap();
    assert!((solution[0] + 0.4).abs() < 1.0e-12);
    assert!((solution[1] - 2.6).abs() < 1.0e-12);
    let residual = matrix.matvec(&solution);
    assert!((residual[0] - 1.0).abs() < 1.0e-12);
    assert!((residual[1] - 7.0).abs() < 1.0e-12);

    let mut destination = [9.0, 9.0];
    <SparseFactorization<f64> as DirectFactorization<f64>>::solve_into(
        &factor,
        &[1.0, 7.0],
        &mut destination,
    )
    .unwrap();
    assert!((destination[0] + 0.4).abs() < 1.0e-12);
    assert!((destination[1] - 2.6).abs() < 1.0e-12);

    let before = destination;
    let error = <SparseFactorization<f64> as DirectFactorization<f64>>::solve_into(
        &factor,
        &[f64::NAN, 7.0],
        &mut destination,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SparseDirectError::NonFiniteRhs { index: 0 }
    ));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(destination, before);
}

pub fn cholesky_rejects_multi_rhs_shape_overflow_without_panicking() {
    let factor = cholesky_factorize(&DenseMatrix::<f64>::eye(2)).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        factor.solve_many_into(&[], &mut [], usize::MAX)
    }));
    let result = result.expect("checked Cholesky shape validation must not panic");
    assert!(matches!(
        result,
        Err(sonium_solvers_direct::CholeskyError::SizeOverflow {
            dimension: 2,
            nrhs: usize::MAX
        })
    ));
}

pub fn sparse_refinement_never_publishes_non_finite_residual_metadata() {
    let factor_matrix = CsrMatrix::identity(1);
    let factor = SparseLuFactorization::factorize(&factor_matrix).unwrap();
    let updated_matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0_f64)]);

    let result = factor.solve_refined(&updated_matrix, &[f64::MAX], 1.0e-12, 0);
    assert!(matches!(
        result,
        Err(SparseLuError::NumericalBreakdown {
            stage: "residual norm",
            ..
        })
    ));

    let result = factor
        .solve_refined(&factor_matrix, &[1.0], 1.0e-12, 1)
        .unwrap();
    assert_eq!(
        result.stop_reason,
        SparseRefinementStopReason::ToleranceReached
    );
    assert!(result.initial_relative_residual.is_finite());
    assert!(result.relative_residual.is_finite());
    assert!(result.tolerance.is_finite());
    assert_eq!(result.status(), SolverStatus::Converged);
    assert_eq!(result.attainable_accuracy(), AccuracyStatus::Attained);
}

pub fn direct_errors_expose_canonical_status_and_accuracy() {
    let dense = DenseDirectError::NonFiniteRhs { index: 0 };
    assert_eq!(dense.status(), SolverStatus::NumericalError);
    assert_eq!(dense.attainable_accuracy(), AccuracyStatus::NotChecked);

    let sparse = SparseDirectError::RhsDimension {
        expected: 2,
        actual: 1,
    };
    assert_eq!(sparse.status(), SolverStatus::NumericalError);
    assert_eq!(sparse.attainable_accuracy(), AccuracyStatus::NotChecked);

    let cholesky = sonium_solvers_direct::CholeskyError::NotPositiveDefinite { pivot: 1 };
    assert_eq!(cholesky.status(), SolverStatus::Breakdown);
    assert_eq!(cholesky.attainable_accuracy(), AccuracyStatus::NotChecked);
}

pub fn direct_composite_errors_preserve_nested_breakdown_and_aliases() {
    fn assert_breakdown<E: SolverFailure>(error: E) {
        assert_eq!(error.status(), SolverStatus::Breakdown);
        assert_eq!(error.solver_status(), SolverStatus::Breakdown);
        assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
        assert_eq!(error.attainable_accuracy(), AccuracyStatus::NotChecked);
    }

    assert_breakdown(DenseDirectError::Lu(LuError::SingularMatrix));
    assert_breakdown(DenseDirectError::Cholesky(
        CholeskyError::NotPositiveDefinite { pivot: 0 },
    ));
    assert_breakdown(DenseDirectError::Qr(QrError::RankDeficient {
        rank: 1,
        columns: 2,
    }));
    assert_breakdown(SparseDirectError::Lu(SparseLuError::SingularMatrix(0)));
    assert_breakdown(SparseDirectError::Cholesky(
        SparseCholeskyError::NotPositiveDefinite { pivot: 0 },
    ));
    assert_breakdown(SparseDirectError::Multifrontal(BlrError::SingularFront(0)));
}

pub fn lu_multi_rhs_into_rejects_size_overflow() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 3.0]);
    let factor = sonium_solvers_direct::lu_factorize(&matrix).unwrap();
    let mut destination = Vec::new();
    assert!(matches!(
        factor.solve_many_into(&[], &mut destination, usize::MAX),
        Err(LuError::SizeOverflow {
            dimension: 2,
            nrhs: usize::MAX,
        })
    ));
}

#[test]
fn dense_facade_normalization_test() {
    dense_facade_normalizes_dimensions_and_non_finite_inputs();
}
#[test]
fn dense_backward_error_test() {
    dense_backward_error_rejects_non_finite_metadata();
}
#[test]
fn sparse_facade_normalization_test() {
    sparse_facade_normalizes_checked_solve_and_preserves_destination();
}

#[test]
fn sparse_multifrontal_many_rhs_preserves_destination_on_later_breakdown() {
    let matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, f64::MIN_POSITIVE)]);
    let factor = factorize_sparse(
        &matrix,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: SparseDirectMethod::MultifrontalLdlt,
            ..Default::default()
        },
        None,
    )
    .expect("multifrontal LDLT factorization");
    let mut destination = vec![17.0; 2];
    let before = destination.clone();
    assert!(
        factor
            .solve_many_into(&[0.0, f64::MAX], &mut destination, 2)
            .is_err()
    );
    assert_eq!(destination, before);
}

#[test]
fn dense_ldlt_many_rhs_preserves_destination_on_later_breakdown() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![f64::MIN_POSITIVE]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: sonium_solvers_direct::DenseDirectMethod::Ldlt,
            ..Default::default()
        },
    )
    .expect("dense LDLT factorization");
    let mut destination = vec![19.0; 2];
    let before = destination.clone();
    assert!(
        factor
            .solve_many_into(&[0.0, f64::MAX], &mut destination, 2)
            .is_err()
    );
    assert_eq!(destination, before);
}
#[test]
fn sparse_many_rhs_test() {
    sparse_direct_facade_many_rhs_preserves_checked_contract();
}
#[test]
fn dense_many_rhs_test() {
    dense_factorization_many_rhs_preserves_checked_contract();
}
#[test]
fn multifrontal_legacy_path_test() {
    sparse_multifrontal_adapter_contains_legacy_panicking_solve_path();
}
#[test]
fn sparse_compatibility_reference_test() {
    sparse_compatibility_uses_checked_direct_contract_and_matches_reference();
}
#[test]
fn cholesky_shape_test() {
    cholesky_rejects_multi_rhs_shape_overflow_without_panicking();
}
#[test]
fn sparse_refinement_metadata_test() {
    sparse_refinement_never_publishes_non_finite_residual_metadata();
}
#[test]
fn direct_status_accuracy_test() {
    direct_errors_expose_canonical_status_and_accuracy();
}
#[test]
fn direct_nested_error_test() {
    direct_composite_errors_preserve_nested_breakdown_and_aliases();
}
#[test]
fn lu_many_rhs_overflow_test() {
    lu_multi_rhs_into_rejects_size_overflow();
}

pub fn dense_single_rhs_preserves_destination_after_lu_failure() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![1.0e-200_f64]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: sonium_solvers_direct::DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("tiny finite LU factor should be admitted");
    let mut destination = [37.0];
    let before = destination;
    let error = factor
        .solve_into(&[f64::MAX], &mut destination)
        .expect_err("finite RHS should overflow the tiny LU pivot");
    assert_eq!(error.status(), SolverStatus::Breakdown);
    assert_eq!(destination, before);
}

#[test]
fn dense_single_rhs_preserves_destination_after_lu_failure_test() {
    dense_single_rhs_preserves_destination_after_lu_failure();
}

pub fn dense_single_rhs_preserves_destination_after_cholesky_failure() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![1.0e-200_f64]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: sonium_solvers_direct::DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .expect("tiny finite Cholesky factor should be admitted");
    let mut destination = [41.0];
    let before = destination;
    let error = factor
        .solve_into(&[f64::MAX], &mut destination)
        .expect_err("finite RHS should overflow the tiny Cholesky pivot");
    assert_eq!(error.status(), SolverStatus::Breakdown);
    assert_eq!(destination, before);
}

#[test]
fn dense_single_rhs_preserves_destination_after_cholesky_failure_test() {
    dense_single_rhs_preserves_destination_after_cholesky_failure();
}

pub fn dense_single_rhs_preserves_destination_after_qr_failure() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![1.0e-10_f64]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::RectangularFullRank,
            method: sonium_solvers_direct::DenseDirectMethod::Qr,
            ..Default::default()
        },
    )
    .expect("tiny finite QR factor should be admitted");
    let mut destination = [43.0];
    let before = destination;
    let error = factor
        .solve_into(&[f64::MAX], &mut destination)
        .expect_err("finite RHS should overflow the tiny QR pivot");
    assert_eq!(error.status(), SolverStatus::Breakdown);
    assert_eq!(destination, before);
}

#[test]
fn dense_single_rhs_preserves_destination_after_qr_failure_test() {
    dense_single_rhs_preserves_destination_after_qr_failure();
}

pub fn dense_many_rhs_preserves_destination_after_later_lu_failure() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![1.0e-200_f64]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: sonium_solvers_direct::DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("tiny finite LU factor should be admitted");
    let mut destination = vec![37.0; 2];
    let before = destination.clone();
    let error = factor
        .solve_many_into(&[1.0, f64::MAX], &mut destination, 2)
        .expect_err("later RHS should overflow the tiny LU pivot");
    assert!(matches!(
        error,
        DenseDirectError::Lu(LuError::NumericalBreakdown { .. })
    ));
    assert_eq!(destination, before);
}

#[test]
fn dense_many_rhs_preserves_destination_after_later_lu_failure_test() {
    dense_many_rhs_preserves_destination_after_later_lu_failure();
}

pub fn dense_many_rhs_preserves_destination_after_later_cholesky_failure() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![1.0e-200_f64]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: sonium_solvers_direct::DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .expect("tiny finite Cholesky factor should be admitted");
    let mut destination = vec![41.0; 2];
    let before = destination.clone();
    let error = factor
        .solve_many_into(&[1.0, f64::MAX], &mut destination, 2)
        .expect_err("later RHS should overflow the tiny Cholesky pivot");
    assert_eq!(error.status(), SolverStatus::Breakdown);
    assert_eq!(destination, before);
}

#[test]
fn dense_many_rhs_preserves_destination_after_later_cholesky_failure_test() {
    dense_many_rhs_preserves_destination_after_later_cholesky_failure();
}

pub fn dense_many_rhs_preserves_destination_after_later_qr_failure() {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![1.0e-10_f64]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::RectangularFullRank,
            method: sonium_solvers_direct::DenseDirectMethod::Qr,
            ..Default::default()
        },
    )
    .expect("tiny finite QR factor should be admitted");
    let mut destination = vec![43.0; 2];
    let before = destination.clone();
    let error = factor
        .solve_many_into(&[1.0, f64::MAX], &mut destination, 2)
        .expect_err("later RHS should overflow the tiny QR pivot");
    assert_eq!(error.status(), SolverStatus::Breakdown);
    assert_eq!(destination, before);
}

#[test]
fn dense_many_rhs_preserves_destination_after_later_qr_failure_test() {
    dense_many_rhs_preserves_destination_after_later_qr_failure();
}
