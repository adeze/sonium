use sonium_solvers_direct::{SparseLuError, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn base_matrix() -> CsrMatrix<f64> {
    CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 0, 2],
        vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0],
    )
    .expect("base matrix is canonical")
}

fn assert_rejected_update(factor: &SparseLuFactorization<f64>, matrix: &CsrMatrix<f64>) {
    assert!(!factor.accepts_value_update(matrix));
    assert!(matches!(
        factor.try_accepts_value_update(matrix),
        Err(SparseLuError::PatternMismatch)
    ));
    assert!(matches!(
        factor.solve_refined(matrix, &[1.0, 2.0, 3.0], 1.0e-12, 2),
        Err(SparseLuError::PatternMismatch)
    ));
}

fn reference_matvec(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * x[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

fn relative_residual(matrix: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let ax = reference_matvec(matrix, x);
    let residual_norm = ax
        .iter()
        .zip(b)
        .map(|(actual, rhs)| (rhs - actual).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    residual_norm / rhs_norm.max(f64::MIN_POSITIVE)
}

#[test]
fn refinement_rejects_canonical_row_and_column_permutations() {
    let factor = SparseLuFactorization::factorize(&base_matrix()).expect("factorization");

    // Swap the first and last rows while retaining canonical column order in
    // every row. The values are irrelevant: the stored row structure changed.
    let row_permuted = CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 2, 0, 1, 2, 0, 1],
        vec![1.0, 4.0, 1.0, 3.0, 1.0, 4.0, 1.0],
    )
    .expect("row permutation remains canonical");
    assert_rejected_update(&factor, &row_permuted);

    // Apply the 0 <-> 2 column permutation to the same sparsity pattern.
    // Each row remains strictly sorted, so this is not merely an unsorted-row
    // rejection masquerading as a pattern-compatibility test.
    let column_permuted = CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![1, 2, 0, 1, 2, 0, 2],
        vec![1.0, 4.0, 1.0, 3.0, 1.0, 4.0, 1.0],
    )
    .expect("column permutation remains canonical");
    assert_rejected_update(&factor, &column_permuted);
}

#[test]
fn refinement_reports_independent_residual_for_same_pattern_value_update() {
    let base = base_matrix();
    let factor = SparseLuFactorization::factorize(&base).expect("factorization");
    let updated = CsrMatrix::try_from_raw_parts(
        3,
        3,
        base.row_ptrs.clone(),
        base.col_indices.clone(),
        vec![4.5, 0.5, 1.25, 3.5, 0.75, 0.8, 4.25],
    )
    .expect("same-pattern value update is canonical");
    let expected = [1.0, -2.0, 0.5];
    let b = reference_matvec(&updated, &expected);

    let initial = factor.solve(&b).expect("initial reusable solve");
    let initial_reference = relative_residual(&updated, &initial, &b);
    let refined = factor
        .solve_refined(&updated, &b, 1.0e-12, 100)
        .expect("same-pattern value update must be refined");
    let final_reference = relative_residual(&updated, &refined.x, &b);

    assert!((refined.initial_relative_residual - initial_reference).abs() < 1.0e-15);
    assert!((refined.relative_residual - final_reference).abs() < 1.0e-15);
    assert!(
        final_reference <= 1.0e-12,
        "independent final relative residual is {final_reference:.3e}"
    );
    for (actual, expected) in refined.x.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    assert!(refined.initial_relative_residual > refined.relative_residual);
}
