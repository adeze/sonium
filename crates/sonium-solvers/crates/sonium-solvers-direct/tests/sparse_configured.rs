use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::SolverResultContract;
use sonium_solvers_direct::{
    DirectFactorization, MatrixClass, SparseDirectConfig, SparseDirectError, SparseDirectMethod,
    factorize_sparse, factorize_sparse_configured,
};
use sonium_solvers_sparse::CsrMatrix;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_dense(
        &DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 0.3, 0.2, 1.1]),
        0.0,
    )
}

fn configured(right_hand_sides: usize) -> SparseDirectConfig {
    SparseDirectConfig {
        matrix_class: MatrixClass::General,
        method: SparseDirectMethod::SimplicialLu,
        right_hand_sides,
        backward_error_tolerance: 0.0,
        ..SparseDirectConfig::default()
    }
}

#[test]
fn configured_sparse_factor_enforces_rhs_count_and_solves_block() {
    let factor = factorize_sparse_configured(&matrix(), &configured(2), None).unwrap();
    let single = factor.solve(&[2.0, 4.0]).unwrap();
    let single_residual = matrix().matvec(&single);
    assert!((single_residual[0] - 2.0).abs() < 1.0e-12);
    assert!((single_residual[1] - 4.0).abs() < 1.0e-12);
    let mut solution = vec![0.0; 4];
    factor
        .solve_many_into(&[2.0, 3.0, 4.0, 5.0], &mut solution, 2)
        .unwrap();
    let first = matrix().matvec(&[solution[0], solution[2]]);
    let second = matrix().matvec(&[solution[1], solution[3]]);
    assert!((first[0] - 2.0).abs() < 1.0e-12);
    assert!((first[1] - 4.0).abs() < 1.0e-12);
    assert!((second[0] - 3.0).abs() < 1.0e-12);
    assert!((second[1] - 5.0).abs() < 1.0e-12);
    assert!(matches!(
        factor.solve_many_into(&[2.0, 3.0], &mut [0.0; 2], 1),
        Err(SparseDirectError::RightHandSideCountMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn configured_sparse_refinement_reports_bounded_diagnostics() {
    let factor = factorize_sparse_configured(&matrix(), &configured(1), None).unwrap();
    let result = factor.solve_refined(&[1.23456789, 2.34567891], 2).unwrap();
    assert!(result.iterations <= 2);
    assert!(result.initial_backward_error.is_finite());
    assert!(result.backward_error.is_finite());
    assert_eq!(result.tolerance, 0.0);
    assert!(result.x.iter().all(|value| value.is_finite()));
}

#[test]
fn configured_sparse_result_contract_reports_retained_diagnostics() {
    let factor = factorize_sparse_configured(&matrix(), &configured(1), None).unwrap();
    let result = factor.solve_with_result(&[1.4, -2.0]).unwrap();

    result.validate_contract().unwrap();
    assert!(result.residual.is_some());
    assert!(result.backward_error.is_some());
    assert!(result.backward_error.unwrap() <= 1.0e-12);
}

#[test]
fn configured_sparse_refinement_supports_configured_block_columns() {
    let factor = factorize_sparse_configured(&matrix(), &configured(2), None).unwrap();
    let results = factor
        .solve_refined_many(&[1.4, -0.775, -2.0, 0.725], 2, 2)
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].x.len(), 2);
    assert_eq!(results[1].x.len(), 2);
    for result in &results {
        result.validate_contract().unwrap();
        assert!(result.backward_error.is_finite());
    }
}

#[test]
fn configured_sparse_refinement_contract_rejects_tampered_metadata() {
    let factor = factorize_sparse_configured(&matrix(), &configured(1), None).unwrap();
    let mut result = factor.solve_refined(&[1.4, -2.0], 1).unwrap();
    result.backward_error = -1.0;

    assert!(result.validate_contract().is_err());
}

#[test]
fn configured_sparse_rejects_invalid_options_and_budget() {
    let mut invalid = configured(0);
    assert!(matches!(
        factorize_sparse_configured(&matrix(), &invalid, None),
        Err(SparseDirectError::InvalidConfiguration {
            field: "right_hand_sides",
            ..
        })
    ));

    invalid = configured(1);
    invalid.backward_error_tolerance = f64::NAN;
    assert!(matches!(
        factorize_sparse_configured(&matrix(), &invalid, None),
        Err(SparseDirectError::InvalidConfiguration {
            field: "backward_error_tolerance",
            ..
        })
    ));

    let mut budget = configured(1);
    budget.memory_budget_bytes = Some(1);
    assert!(matches!(
        factorize_sparse_configured(&matrix(), &budget, None),
        Err(SparseDirectError::MemoryBudgetExceeded {
            budget_bytes: 1,
            ..
        })
    ));
}

#[test]
fn legacy_sparse_factor_rejects_configured_options_instead_of_ignoring_them() {
    let config = configured(2);
    assert!(matches!(
        factorize_sparse(&matrix(), &config, None),
        Err(SparseDirectError::OptionUnavailable {
            option: "configured sparse options (use factorize_sparse_configured)",
            ..
        })
    ));
}
