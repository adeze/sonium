use num_complex::Complex64;
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, SolverResultContract, SolverStatus},
};
use sonium_solvers_eigen::{
    ComplexArnoldiConfig, ComplexTargetedEigenResult, LobpcgConfig, LobpcgResult,
    ShiftInvertConfig, TargetedEigenResult, complex_shift_invert_arnoldi_generalized, lobpcg,
    shift_invert_lanczos_generalized, symmetric_eigen,
};
use sonium_solvers_sparse::CsrMatrix;

fn real_diagonal(values: &[f64]) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, index, value))
            .collect(),
    )
}

fn complex_diagonal(values: &[f64]) -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, index, Complex64::new(value, 0.0)))
            .collect(),
    )
}

#[test]
fn real_eigen_families_match_independent_diagonal_fixture() {
    let operator = real_diagonal(&[1.0, 3.0, 5.0]);
    let mass = real_diagonal(&[1.0, 1.0, 1.0]);

    let lobpcg_result = lobpcg(
        &operator,
        &LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 80,
            tolerance: 1.0e-8,
            block_padding: 0,
        },
    )
    .expect("LOBPCG should solve the diagonal fixture");
    assert_eq!(lobpcg_result.status(), SolverStatus::Converged);
    assert_eq!(
        lobpcg_result.attainable_accuracy(),
        AccuracyStatus::Attained
    );
    assert!((lobpcg_result.eigenvalues[0] - 1.0).abs() < 1.0e-6);
    assert!(lobpcg_result.residual_norms[0].is_finite());

    let shift_result = shift_invert_lanczos_generalized(
        &operator,
        &mass,
        &ShiftInvertConfig {
            target: 1.2,
            num_eigenvalues: 1,
            krylov_dimension: 3,
            tolerance: 1.0e-8,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect("shift-invert Lanczos should solve the diagonal fixture");
    assert_eq!(shift_result.status(), SolverStatus::Converged);
    assert_eq!(shift_result.accuracy_status(), AccuracyStatus::Attained);
    SolverResultContract::validate_contract(&shift_result)
        .expect("real eigen result should satisfy shared contract");
    assert!((shift_result.eigenvalues[0] - 1.0).abs() < 1.0e-6);
    assert!(shift_result.residual_norms[0].is_finite());
}

#[test]
fn complex_eigen_fixture_reports_canonical_result_status() {
    let operator = complex_diagonal(&[1.0, 3.0, 5.0]);
    let mass = complex_diagonal(&[1.0, 1.0, 1.0]);
    let result = complex_shift_invert_arnoldi_generalized(
        &operator,
        &mass,
        &ComplexArnoldiConfig {
            target: Complex64::new(1.2, 0.0),
            num_eigenvalues: 1,
            krylov_dimension: 3,
            tolerance: 1.0e-8,
            max_qr_iterations: 500,
            breakdown_tolerance: 1.0e-14,
        },
    )
    .expect("complex Arnoldi should solve the diagonal fixture");

    assert_eq!(result.status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    SolverResultContract::validate_contract(&result)
        .expect("complex eigen result should satisfy shared contract");
    assert!((result.eigenvalues[0] - Complex64::new(1.0, 0.0)).norm() < 1.0e-6);
    assert!(result.residual_norms[0].is_finite());
}

#[test]
fn eigen_result_shape_corruption_is_reported_as_numerical_error() {
    let result = TargetedEigenResult {
        eigenvalues: vec![1.0],
        eigenvectors: DenseMatrix::zeros(2, 0),
        residual_norms: vec![],
        iterations: 0,
        converged: true,
    };
    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.attainable_accuracy(), AccuracyStatus::NotChecked);
}

#[test]
fn all_eigen_result_families_reject_non_finite_vectors_and_invalid_residuals() {
    let real_shift = TargetedEigenResult {
        eigenvalues: vec![1.0],
        eigenvectors: DenseMatrix::from_row_major_vec(2, 1, vec![1.0, f64::NAN]),
        residual_norms: vec![0.0],
        iterations: 1,
        converged: true,
    };
    assert_eq!(real_shift.status(), SolverStatus::NumericalError);
    assert_eq!(real_shift.accuracy_status(), AccuracyStatus::NotChecked);
    assert!(SolverResultContract::validate_contract(&real_shift).is_err());

    let real_lobpcg = LobpcgResult {
        eigenvalues: vec![1.0],
        eigenvectors: DenseMatrix::from_row_major_vec(2, 1, vec![1.0, 0.0]),
        iterations: 1,
        residual_norms: vec![-1.0],
        converged: true,
    };
    assert_eq!(real_lobpcg.status(), SolverStatus::NumericalError);
    assert_eq!(real_lobpcg.accuracy_status(), AccuracyStatus::NotChecked);

    let complex_arnoldi = ComplexTargetedEigenResult {
        eigenvalues: vec![Complex64::new(1.0, 0.0)],
        eigenvectors: DenseMatrix::from_row_major_vec(
            2,
            1,
            vec![Complex64::new(1.0, 0.0), Complex64::new(f64::NAN, 0.0)],
        ),
        residual_norms: vec![0.0],
        iterations: 1,
        converged: true,
    };
    assert_eq!(complex_arnoldi.status(), SolverStatus::NumericalError);
    assert_eq!(
        complex_arnoldi.accuracy_status(),
        AccuracyStatus::NotChecked
    );
}

#[test]
fn empty_interval_eigen_result_remains_a_converged_valid_result() {
    let result = TargetedEigenResult {
        eigenvalues: vec![],
        eigenvectors: DenseMatrix::zeros(3, 0),
        residual_norms: vec![],
        iterations: 0,
        converged: true,
    };
    assert_eq!(result.status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
}

#[test]
fn incomplete_real_and_complex_results_report_unattained_accuracy() {
    let real = TargetedEigenResult {
        eigenvalues: vec![],
        eigenvectors: DenseMatrix::zeros(3, 0),
        residual_norms: vec![],
        iterations: 4,
        converged: false,
    };
    assert_eq!(real.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(real.accuracy_status(), AccuracyStatus::NotAttained);

    let complex = ComplexTargetedEigenResult {
        eigenvalues: vec![Complex64::new(1.0, 0.0)],
        eigenvectors: DenseMatrix::from_row_major_vec(
            2,
            1,
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ),
        residual_norms: vec![0.25],
        iterations: 2,
        converged: false,
    };
    assert_eq!(complex.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(complex.attainable_accuracy(), AccuracyStatus::NotAttained);
}

#[test]
fn symmetric_eigen_rejects_malformed_and_non_finite_inputs() {
    let malformed = symmetric_eigen(&DenseMatrix::zeros(2, 3), 2)
        .expect_err("symmetric eigensolver must reject a mismatched matrix shape");
    assert!(matches!(
        malformed,
        sonium_solvers_core::traits::SolverError::DimensionMismatch {
            expected: 2,
            got: 3
        }
    ));

    let non_finite = symmetric_eigen(
        &DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 0.0, 0.0, f64::NAN]),
        2,
    )
    .expect_err("symmetric eigensolver must reject non-finite input");
    assert!(matches!(
        non_finite,
        sonium_solvers_core::traits::SolverError::OperationFailed { .. }
    ));
}
