use sonium_solvers_core::traits::{
    AccuracyStatus, IdentityPreconditioner, LinearOperator, Preconditioner, SolverError,
    SolverStatus,
};
use sonium_solvers_iterative::{CaCgConfig, ca_cg, try_ca_cg, try_ca_pcg};
use sonium_solvers_sparse::CsrMatrix;

fn identity(size: usize) -> CsrMatrix<f64> {
    CsrMatrix::identity(size)
}

fn config() -> CaCgConfig<f64> {
    CaCgConfig {
        max_iterations: 8,
        tolerance: 1.0e-12,
        step_size: 2,
        ..CaCgConfig::default()
    }
}

#[test]
fn ca_cg_matches_independent_identity_fixture_and_reports_canonical_metadata() {
    let rhs = [2.0, -3.0];
    let result = try_ca_cg(&identity(rhs.len()), &rhs, &config()).expect("identity must converge");

    assert_eq!(result.status, SolverStatus::Converged);
    assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
    assert!(result.residual <= 1.0e-12);
    assert!(result.backward_error.is_finite());
    for (actual, expected) in result.x.iter().zip(rhs) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
}

#[test]
fn ca_cg_and_ca_pcg_preserve_typed_dimensions_and_family_operation_names() {
    let matrix = identity(2);
    let dimension_error =
        try_ca_cg(&matrix, &[1.0], &config()).expect_err("RHS dimension must be checked");
    assert!(matches!(
        dimension_error,
        SolverError::DimensionMismatch {
            expected: 2,
            got: 1
        }
    ));
    assert_eq!(dimension_error.status(), SolverStatus::NumericalError);
    assert_eq!(
        dimension_error.accuracy_status(),
        AccuracyStatus::NotChecked
    );

    let mut invalid = config();
    invalid.max_iterations = 0;
    let ca_cg_error =
        try_ca_cg(&matrix, &[1.0, 2.0], &invalid).expect_err("invalid config must fail");
    assert!(matches!(
        ca_cg_error,
        SolverError::OperationFailed {
            operation: "CA-CG",
            ..
        }
    ));

    let ca_pcg_error = try_ca_pcg(&matrix, &[1.0, 2.0], &IdentityPreconditioner, &invalid)
        .expect_err("invalid preconditioned config must fail");
    assert!(matches!(
        ca_pcg_error,
        SolverError::OperationFailed {
            operation: "CA-PCG",
            reason,
        } if reason == "invalid CA-PCG configuration"
    ));
}

struct PanicOperator;

impl LinearOperator<f64> for PanicOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _input: &[f64], _output: &mut [f64]) {
        panic!("independent CA-CG callback failure");
    }

    fn apply_transpose_into(&self, _input: &[f64], _output: &mut [f64]) {
        panic!("independent CA-CG transpose callback failure");
    }
}

#[test]
fn ca_cg_and_ca_pcg_contain_callback_failures_with_family_statuses() {
    let ca_cg_error =
        try_ca_cg(&PanicOperator, &[1.0], &config()).expect_err("panic must be typed");
    assert!(matches!(
        ca_cg_error,
        SolverError::OperationFailed {
            operation: "CA-CG",
            ..
        }
    ));
    assert_eq!(ca_cg_error.status(), SolverStatus::NumericalError);

    let ca_pcg_error = try_ca_pcg(&PanicOperator, &[1.0], &IdentityPreconditioner, &config())
        .expect_err("preconditioned panic must be typed");
    assert!(matches!(
        ca_pcg_error,
        SolverError::OperationFailed {
            operation: "CA-PCG",
            ..
        }
    ));
    assert_eq!(ca_pcg_error.status(), SolverStatus::NumericalError);
}

struct NonFiniteOperator;

impl LinearOperator<f64> for NonFiniteOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = f64::NAN;
    }

    fn apply_transpose_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = f64::NAN;
    }
}

struct NonFinitePreconditioner;

impl Preconditioner<f64> for NonFinitePreconditioner {
    fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
        output.fill(f64::INFINITY);
    }
}

#[test]
fn ca_pcg_keeps_operator_and_preconditioner_failures_distinct() {
    let operator_error = try_ca_pcg(
        &NonFiniteOperator,
        &[1.0],
        &IdentityPreconditioner,
        &config(),
    )
    .expect_err("non-finite operator output must be typed");
    assert!(matches!(
        operator_error,
        SolverError::OperationFailed {
            operation: "CA-PCG operator",
            ..
        }
    ));

    let preconditioner_error =
        try_ca_pcg(&identity(1), &[1.0], &NonFinitePreconditioner, &config())
            .expect_err("non-finite preconditioner output must be typed");
    assert!(matches!(
        preconditioner_error,
        SolverError::OperationFailed {
            operation: "CA-PCG preconditioner",
            ..
        }
    ));
}

struct ZeroOperator;

impl LinearOperator<f64> for ZeroOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = 0.0;
    }

    fn apply_transpose_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = 0.0;
    }
}

#[test]
fn ca_cg_zero_operator_returns_finite_breakdown_metadata() {
    let result = try_ca_cg(&ZeroOperator, &[1.0], &config()).expect("breakdown is a result status");

    assert_eq!(result.status, SolverStatus::Breakdown);
    assert!(!result.converged);
    assert_eq!(result.attainable_accuracy, AccuracyStatus::NotAttained);
    assert!(result.residual.is_finite());
    assert!(result.backward_error.is_finite());

    let legacy = ca_cg(&ZeroOperator, &[1.0], &config());
    assert_eq!(legacy.status, SolverStatus::Breakdown);
    assert_eq!(legacy.attainable_accuracy, AccuracyStatus::NotAttained);
}
