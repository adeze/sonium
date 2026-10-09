use num_complex::Complex64;
use sonium_solvers_core::traits::{AccuracyStatus, SolverError, SolverStatus};
use sonium_solvers_eigen::{ComplexArnoldiConfig, complex_shift_invert_arnoldi_generalized};
use sonium_solvers_sparse::CsrMatrix;

fn identity(dimension: usize) -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        dimension,
        dimension,
        (0..dimension)
            .map(|index| (index, index, Complex64::new(1.0, 0.0)))
            .collect(),
    )
}

fn config(target: Complex64) -> ComplexArnoldiConfig {
    ComplexArnoldiConfig {
        target,
        num_eigenvalues: 1,
        krylov_dimension: 2,
        tolerance: 1.0e-8,
        max_qr_iterations: 100,
        breakdown_tolerance: 1.0e-14,
    }
}

#[test]
fn complex_arnoldi_rejects_non_finite_matrix_coefficients() {
    let identity = identity(2);
    let non_finite_operator = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        values: vec![Complex64::new(f64::NAN, 0.0), Complex64::new(2.0, 0.0)],
        col_indices: vec![0, 1],
        row_ptrs: vec![0, 1, 2],
    };
    let error = complex_shift_invert_arnoldi_generalized(
        &non_finite_operator,
        &identity,
        &config(Complex64::new(0.0, 0.0)),
    )
    .expect_err("non-finite operator coefficients must be rejected");
    assert!(matches!(error, SolverError::OperationFailed { .. }));

    let non_finite_mass = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        values: vec![Complex64::new(1.0, 0.0), Complex64::new(f64::INFINITY, 0.0)],
        col_indices: vec![0, 1],
        row_ptrs: vec![0, 1, 2],
    };
    let error = complex_shift_invert_arnoldi_generalized(
        &identity,
        &non_finite_mass,
        &config(Complex64::new(0.0, 0.0)),
    )
    .expect_err("non-finite mass coefficients must be rejected");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn complex_arnoldi_rejects_malformed_public_csr_storage() {
    let malformed = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        values: vec![Complex64::new(1.0, 0.0)],
        col_indices: vec![0],
        row_ptrs: vec![0, 1],
    };
    let error = complex_shift_invert_arnoldi_generalized(
        &malformed,
        &identity(2),
        &config(Complex64::new(0.0, 0.0)),
    )
    .expect_err("mutated public CSR storage must be rejected before row access");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn complex_arnoldi_rejects_overflow_in_shifted_matrix_construction() {
    let operator = identity(2);
    let huge_mass = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(f64::MAX, 0.0)),
            (1, 1, Complex64::new(f64::MAX, 0.0)),
        ],
    );
    let error = complex_shift_invert_arnoldi_generalized(
        &operator,
        &huge_mass,
        &config(Complex64::new(2.0, 0.0)),
    )
    .expect_err("finite coefficients must not hide shifted-matrix overflow");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn complex_arnoldi_rejects_more_eigenpairs_than_operator_dimension() {
    let error = complex_shift_invert_arnoldi_generalized(
        &identity(2),
        &identity(2),
        &ComplexArnoldiConfig {
            target: Complex64::new(0.0, 0.0),
            num_eigenvalues: 3,
            krylov_dimension: 3,
            tolerance: 1.0e-8,
            max_qr_iterations: 100,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect_err("the complex eigensolver must reject requests beyond operator dimension");

    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn complex_arnoldi_rejects_overflow_in_mass_application() {
    let operator = identity(2);
    let huge_dense_mass = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(f64::MAX, 0.0)),
            (0, 1, Complex64::new(f64::MAX, 0.0)),
            (1, 0, Complex64::new(f64::MAX, 0.0)),
            (1, 1, Complex64::new(f64::MAX, 0.0)),
        ],
    );
    let error = complex_shift_invert_arnoldi_generalized(
        &operator,
        &huge_dense_mass,
        &config(Complex64::new(0.0, 0.0)),
    )
    .expect_err("finite coefficients must not hide sparse-application overflow");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn complex_arnoldi_reports_unattained_qr_accuracy_as_convergence_error() {
    let error = complex_shift_invert_arnoldi_generalized(
        &CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(1.0, 0.0)),
                (0, 1, Complex64::new(0.5, 0.2)),
                (1, 1, Complex64::new(2.0, 0.0)),
            ],
        ),
        &identity(2),
        &ComplexArnoldiConfig {
            target: Complex64::new(0.0, 0.0),
            num_eigenvalues: 1,
            krylov_dimension: 2,
            tolerance: 1.0e-12,
            max_qr_iterations: 1,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect_err("a deliberately exhausted QR budget must remain a convergence failure");

    assert!(matches!(
        error,
        SolverError::ConvergenceError {
            status: SolverStatus::MaxIterationsReached,
            ..
        }
    ));
    assert_eq!(error.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotAttained);
}

#[test]
fn complex_arnoldi_restart_covers_degenerate_spectrum_after_lucky_breakdown() {
    // Mirror of the real-path restart test: A = B = I has the single
    // eigenvalue 1 (multiplicity 2). The first stream breaks down after one
    // vector; the restart stream explores the complementary direction and
    // the merged runs recover both orthogonal eigenvectors.
    let result = complex_shift_invert_arnoldi_generalized(
        &identity(2),
        &identity(2),
        &ComplexArnoldiConfig {
            target: Complex64::new(0.0, 0.0),
            num_eigenvalues: 2,
            krylov_dimension: 2,
            tolerance: 1.0e-8,
            max_qr_iterations: 100,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect("restart must cover the degenerate spectrum");
    assert!(result.converged);
    assert_eq!(result.eigenvalues.len(), 2);
    for value in &result.eigenvalues {
        assert!(
            (*value - Complex64::new(1.0, 0.0)).norm() < 1.0e-12,
            "unexpected eigenvalue {value:?}"
        );
    }
    let columns = &result.eigenvectors;
    let overlap = (0..2)
        .map(|row| columns[[row, 0]].conj() * columns[[row, 1]])
        .sum::<Complex64>()
        .norm();
    assert!(
        overlap < 1.0e-9,
        "restart must find distinct eigenvectors, overlap {overlap}"
    );
}
