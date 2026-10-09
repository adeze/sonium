use sonium_solvers_core::{
    AccuracyStatus, SOLVER_RESULT_CONTRACT_VERSION, SolverError, SolverErrorKind, SolverFailure,
    SolverStatus,
};

#[test]
fn status_and_accuracy_vocabulary_is_explicit_and_non_ambiguous() {
    assert_eq!(SOLVER_RESULT_CONTRACT_VERSION, 1);
    assert!(SolverStatus::Converged.is_converged());
    assert!(!SolverStatus::Breakdown.is_converged());
    assert!(SolverStatus::NumericalError.is_failure());
    assert!(AccuracyStatus::Attained.is_attained());
    assert!(!AccuracyStatus::NotChecked.is_attained());
}

#[derive(Debug)]
struct ExternalFailure(SolverStatus);

impl std::fmt::Display for ExternalFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "external failure: {:?}", self.0)
    }
}

impl SolverFailure for ExternalFailure {
    fn status(&self) -> SolverStatus {
        self.0
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }
}

#[test]
fn solver_failure_default_kind_preserves_status_taxonomy() {
    let cases = [
        (
            SolverStatus::MaxIterationsReached,
            SolverErrorKind::Convergence,
        ),
        (SolverStatus::Stagnated, SolverErrorKind::Convergence),
        (SolverStatus::Diverged, SolverErrorKind::Convergence),
        (SolverStatus::Breakdown, SolverErrorKind::Breakdown),
        (
            SolverStatus::NumericalError,
            SolverErrorKind::OperationFailed,
        ),
        (SolverStatus::Converged, SolverErrorKind::OperationFailed),
    ];

    for (status, expected_kind) in cases {
        assert_eq!(ExternalFailure(status).error_kind(), expected_kind);
    }
}

#[test]
fn every_solver_error_variant_maps_to_one_canonical_kind_and_status() {
    let cases = [
        (
            SolverError::ConvergenceError {
                status: SolverStatus::MaxIterationsReached,
                iterations: 10,
                residual: 1.0,
            },
            SolverErrorKind::Convergence,
            SolverStatus::MaxIterationsReached,
        ),
        (
            SolverError::Breakdown {
                operation: "gmres",
                reason: "zero pivot".into(),
            },
            SolverErrorKind::Breakdown,
            SolverStatus::Breakdown,
        ),
        (
            SolverError::DimensionMismatch {
                expected: 2,
                got: 3,
            },
            SolverErrorKind::DimensionMismatch,
            SolverStatus::NumericalError,
        ),
        (
            SolverError::UnsupportedOperation {
                operation: "transpose",
                preconditioner: "test",
            },
            SolverErrorKind::UnsupportedOperation,
            SolverStatus::NumericalError,
        ),
        (
            SolverError::UnsupportedOperatorOperation {
                operation: "transpose",
                operator: "test",
            },
            SolverErrorKind::UnsupportedOperatorOperation,
            SolverStatus::NumericalError,
        ),
        (
            SolverError::OperationFailed {
                operation: "apply",
                reason: "non-finite output".into(),
            },
            SolverErrorKind::OperationFailed,
            SolverStatus::NumericalError,
        ),
    ];

    for (error, kind, status) in cases {
        assert_eq!(error.kind(), kind);
        assert_eq!(error.status(), status);
        assert_eq!(error.solver_status(), status);
        let expected_accuracy = match status {
            SolverStatus::MaxIterationsReached
            | SolverStatus::Stagnated
            | SolverStatus::Diverged => AccuracyStatus::NotAttained,
            SolverStatus::Converged | SolverStatus::Breakdown | SolverStatus::NumericalError => {
                AccuracyStatus::NotChecked
            }
        };
        assert_eq!(error.accuracy_status(), expected_accuracy);
    }
}
