use sonium_solvers_direct::{
    SparseLuError, SparseLuFactorization, SparseRefinementProvenance, SparseRefinementStopReason,
};
use sonium_solvers_sparse::CsrMatrix;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
    )
}

fn small_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![3.0, 0.5, 0.5, 2.0],
    )
}

fn reference_matvec(x: &[f64; 3]) -> [f64; 3] {
    [
        4.0 * x[0] + x[1],
        x[0] + 3.0 * x[1] + x[2],
        x[1] + 2.0 * x[2],
    ]
}

#[test]
fn many_refinement_reuses_factor_and_reports_each_column() {
    let matrix = matrix();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");
    let expected = [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5], [2.0, 0.25, -1.0]];
    let rhs = [
        reference_matvec(&expected[0])[0],
        reference_matvec(&expected[1])[0],
        reference_matvec(&expected[2])[0],
        reference_matvec(&expected[0])[1],
        reference_matvec(&expected[1])[1],
        reference_matvec(&expected[2])[1],
        reference_matvec(&expected[0])[2],
        reference_matvec(&expected[1])[2],
        reference_matvec(&expected[2])[2],
    ];

    let results = factor
        .solve_refined_many(&matrix, &rhs, 3, 1.0e-12, 3)
        .expect("multi-RHS refinement");
    assert_eq!(results.len(), 3);
    for (column, result) in results.iter().enumerate() {
        assert!(result.converged);
        assert_eq!(
            result.stop_reason,
            SparseRefinementStopReason::ToleranceReached
        );
        assert_eq!(
            result.provenance,
            SparseRefinementProvenance::ReusedValueOnlyPattern {
                structure_fingerprint: factor.structure_fingerprint(),
            }
        );
        for (actual, expected) in result.x.iter().zip(expected[column]) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert!(result.relative_residual < 1.0e-12);
    }
}

#[test]
fn many_refinement_rejects_bad_batch_shape_before_solving() {
    let matrix = matrix();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");
    let error = factor
        .solve_refined_many(&matrix, &[1.0, 2.0, 3.0], 2, 1.0e-12, 2)
        .expect_err("short row-major RHS must be rejected");
    assert!(matches!(
        error,
        SparseLuError::DimensionMismatch {
            expected: 6,
            actual: 3
        }
    ));
}

#[test]
fn many_refinement_rejects_changed_pattern_before_partial_batch() {
    let matrix = matrix();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");
    let changed = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 0, 2],
        vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
    );
    let error = factor
        .solve_refined_many(&changed, &[1.0; 6], 2, 1.0e-12, 2)
        .expect_err("changed CSR pattern must be rejected");
    assert!(matches!(error, SparseLuError::PatternMismatch));
}

#[test]
fn many_refinement_supports_supernodal_and_equilibrated_factors() {
    let matrix = matrix();
    let expected_columns = [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5]];
    let mut rhs = Vec::with_capacity(6);
    let expected_rhs = expected_columns.map(|expected| reference_matvec(&expected));
    for row in 0..3 {
        for column in &expected_rhs {
            rhs.push(column[row]);
        }
    }

    let factors = [
        SparseLuFactorization::factorize_supernodal(&matrix).unwrap(),
        SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true).unwrap(),
        SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true).unwrap(),
    ];
    for factor in factors {
        let results = factor
            .solve_refined_many(&matrix, &rhs, 2, 1.0e-12, 3)
            .unwrap();
        assert_eq!(results.len(), expected_columns.len());
        for (result, expected) in results.iter().zip(expected_columns) {
            assert!(result.converged);
            assert!(result.relative_residual < 1.0e-12);
            assert!(
                result
                    .x
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| { (actual - expected).abs() < 1.0e-10 })
            );
            assert!(matches!(
                result.provenance,
                SparseRefinementProvenance::ReusedValueOnlyPattern { .. }
            ));
        }
    }
}

#[test]
fn refinement_rejects_duplicate_and_unsorted_rows_for_all_factor_variants() {
    let matrix = small_matrix();
    let factors = [
        SparseLuFactorization::factorize(&matrix).unwrap(),
        SparseLuFactorization::factorize_supernodal(&matrix).unwrap(),
        SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true).unwrap(),
        SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true).unwrap(),
    ];
    let invalid_rows = [
        CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 3, 4],
            vec![0, 0, 1, 1],
            vec![3.0, 1.0, 0.5, 2.0],
        ),
        CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![1, 0, 0, 1],
            vec![0.5, 3.0, 0.5, 2.0],
        ),
    ];
    for factor in factors {
        for invalid in &invalid_rows {
            assert!(matches!(
                factor.solve_refined(invalid, &[1.0, 2.0], 1.0e-12, 1),
                Err(SparseLuError::InvalidCsrStructure(_))
            ));
            assert!(matches!(
                factor.solve_refined_many(invalid, &[1.0, 2.0], 1, 1.0e-12, 1),
                Err(SparseLuError::InvalidCsrStructure(_))
            ));
        }
    }
}
