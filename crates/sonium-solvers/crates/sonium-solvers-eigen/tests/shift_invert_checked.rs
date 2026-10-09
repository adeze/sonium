use sonium_solvers_core::traits::{AccuracyStatus, SolverError, SolverStatus};
use sonium_solvers_eigen::{
    ShiftInvertConfig, eigenvalue_count_in_interval, shift_invert_lanczos_generalized,
};
use sonium_solvers_sparse::CsrMatrix;

fn identity(dimension: usize) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        dimension,
        dimension,
        (0..dimension).map(|index| (index, index, 1.0)).collect(),
    )
}

fn config(target: f64) -> ShiftInvertConfig {
    ShiftInvertConfig {
        target,
        num_eigenvalues: 1,
        krylov_dimension: 2,
        tolerance: 1.0e-8,
        breakdown_tolerance: 1.0e-14,
    }
}

#[test]
fn shift_invert_rejects_malformed_or_non_finite_public_csr_storage() {
    let malformed = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        values: vec![1.0],
        col_indices: vec![0],
        row_ptrs: vec![0, 1],
    };
    let error = shift_invert_lanczos_generalized(&malformed, &identity(2), &config(0.0))
        .expect_err("mutated public CSR storage must be rejected before row access");
    assert!(matches!(error, SolverError::OperationFailed { .. }));

    let non_finite_mass = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        values: vec![1.0, f64::NAN],
        col_indices: vec![0, 1],
        row_ptrs: vec![0, 1, 2],
    };
    let error = shift_invert_lanczos_generalized(&identity(2), &non_finite_mass, &config(0.0))
        .expect_err("non-finite mass coefficients must be rejected");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn shift_invert_accepts_krylov_dimension_equal_to_requested_pairs() {
    let operator = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 2.0)]);
    let result = shift_invert_lanczos_generalized(
        &operator,
        &identity(2),
        &ShiftInvertConfig {
            target: 1.5,
            num_eigenvalues: 2,
            krylov_dimension: 2,
            tolerance: 1.0e-10,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect("a full-dimensional Krylov space can return every requested pair");

    assert!(result.converged);
    assert_eq!(result.eigenvalues.len(), 2);
    assert!((result.eigenvalues[0] - 2.0).abs() < 1.0e-10);
    assert!((result.eigenvalues[1] - 1.0).abs() < 1.0e-10);
}

#[test]
fn shift_invert_rejects_more_pairs_than_operator_dimension() {
    let error = shift_invert_lanczos_generalized(
        &identity(2),
        &identity(2),
        &ShiftInvertConfig {
            target: 0.0,
            num_eigenvalues: 3,
            krylov_dimension: 3,
            tolerance: 1.0e-8,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect_err("the requested pair count must fit the operator dimension");

    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn shift_invert_rejects_overflow_in_shifted_matrix_construction() {
    let huge_mass = CsrMatrix::from_triplets(2, 2, vec![(0, 0, f64::MAX), (1, 1, f64::MAX)]);
    let error = shift_invert_lanczos_generalized(&identity(2), &huge_mass, &config(2.0))
        .expect_err("finite coefficients must not hide shifted-matrix overflow");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn shift_invert_rejects_overflow_in_mass_application() {
    let huge_dense_mass = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, f64::MAX),
            (0, 1, f64::MAX),
            (1, 0, f64::MAX),
            (1, 1, f64::MAX),
        ],
    );
    let error = shift_invert_lanczos_generalized(&identity(2), &huge_dense_mass, &config(0.0))
        .expect_err("finite coefficients must not hide sparse-application overflow");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn interval_count_rejects_endpoint_shift_overflow() {
    let huge_mass = CsrMatrix::from_triplets(2, 2, vec![(0, 0, f64::MAX), (1, 1, f64::MAX)]);
    let error = eigenvalue_count_in_interval(&identity(2), &huge_mass, 1.0, 2.0)
        .expect_err("endpoint expansion must not panic on shifted-matrix overflow");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn shift_invert_preserves_sparse_factor_breakdown_status() {
    let error = shift_invert_lanczos_generalized(&identity(2), &identity(2), &config(1.0))
        .expect_err("a singular shifted factor must be reported as a breakdown");
    assert!(matches!(error, SolverError::Breakdown { .. }));
    assert_eq!(error.status(), SolverStatus::Breakdown);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn shift_invert_restart_covers_degenerate_spectrum_after_lucky_breakdown() {
    // A = B = I has the single eigenvalue 1 (multiplicity 2). The first
    // stream breaks down after one vector (its Krylov space is an exact
    // one-dimensional invariant subspace), so the old code returned
    // Breakdown here. The restart stream is orthogonalized against the found
    // Ritz vector, explores the complementary direction, and the merged runs
    // recover both B-orthogonal eigenvectors instead of erroring.
    let result = shift_invert_lanczos_generalized(
        &identity(2),
        &identity(2),
        &ShiftInvertConfig {
            target: 0.0,
            num_eigenvalues: 2,
            krylov_dimension: 3,
            tolerance: 1.0e-8,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect("restart must cover the degenerate spectrum");
    assert!(result.converged);
    assert_eq!(result.eigenvalues.len(), 2);
    for value in &result.eigenvalues {
        assert!(
            (value - 1.0).abs() < 1.0e-12,
            "unexpected eigenvalue {value}"
        );
    }
    let columns = result.eigenvectors;
    let dot: f64 = (0..2)
        .map(|row| columns[[row, 0]] * columns[[row, 1]])
        .sum();
    assert!(
        dot.abs() < 1.0e-9,
        "restart must find distinct eigenvectors, overlap {dot}"
    );
}
