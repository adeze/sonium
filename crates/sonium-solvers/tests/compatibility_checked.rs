use num_complex::{Complex32, Complex64};
use sonium_solvers::{
    AccuracyStatus, CsrMatrix, LinearSolveConfig, LinearSolveError, LinearSolverType,
    SolverResultContract, SolverStatus, solve, solve_real,
};

#[test]
fn facade_result_implements_shared_result_contract() {
    let matrix = CsrMatrix::identity(2);
    let result = solve_real(&matrix, &[1.0, -2.0], &LinearSolveConfig::default())
        .expect("identity solve should converge");

    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    SolverResultContract::validate_contract(&result).expect("shared contract should validate");
}

#[test]
fn facade_wrapper_preserves_checked_result_metadata() {
    let matrix = CsrMatrix::identity(2);
    let result = solve_real(
        &matrix,
        &[2.0, -3.0],
        &LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            ..Default::default()
        },
    )
    .expect("identity fixture should solve through the compatibility facade");
    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    assert_eq!(result.x, vec![2.0, -3.0]);
    result
        .validate_contract()
        .expect("facade result must satisfy the public contract");
}

#[test]
fn facade_wrapper_maps_invalid_inputs_to_canonical_failure_metadata() {
    let matrix = CsrMatrix::identity(2);
    let error = solve_real(&matrix, &[1.0], &LinearSolveConfig::default())
        .expect_err("RHS dimension mismatch must remain checked at the facade");
    assert!(matches!(
        error,
        LinearSolveError::DimensionMismatch {
            operation: "linear solve rhs",
            expected: 2,
            actual: 1,
        }
    ));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn facade_wrapper_maps_nonfinite_inputs_to_canonical_failure_metadata() {
    let matrix = CsrMatrix::identity(1);
    let error = solve_real(&matrix, &[f64::NAN], &LinearSolveConfig::default())
        .expect_err("non-finite RHS must remain a typed facade failure");
    assert!(matches!(error, LinearSolveError::NonFiniteInput { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.attainable_accuracy(), AccuracyStatus::NotChecked);
}

#[test]
fn generic_facade_executes_all_supported_scalar_families() {
    let config = LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        tolerance: 1.0e-6,
        ..Default::default()
    };
    let real32 = solve::<f32>(&CsrMatrix::identity(2), &[1.0, -2.0], &config)
        .expect("f32 generic facade solve should converge");
    let real64 = solve::<f64>(&CsrMatrix::identity(2), &[1.0, -2.0], &config)
        .expect("f64 generic facade solve should converge");
    let complex32 = solve::<Complex32>(
        &CsrMatrix::identity(2),
        &[Complex32::new(1.0, 0.5), Complex32::new(-2.0, 0.25)],
        &config,
    )
    .expect("Complex32 generic facade solve should converge");
    let complex64 = solve::<Complex64>(
        &CsrMatrix::identity(2),
        &[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 0.25)],
        &config,
    )
    .expect("Complex64 generic facade solve should converge");

    macro_rules! assert_valid {
        ($result:expr) => {
            assert_eq!($result.solver_status(), SolverStatus::Converged);
            assert_eq!($result.accuracy_status(), AccuracyStatus::Attained);
            $result
                .validate_contract()
                .expect("generic facade result must satisfy the public contract");
        };
    }
    assert_valid!(real32);
    assert_valid!(real64);
    assert_valid!(complex32);
    assert_valid!(complex64);
}
