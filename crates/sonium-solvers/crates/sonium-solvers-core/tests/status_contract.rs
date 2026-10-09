use sonium_solvers_core::traits::{AccuracyStatus, SolverError, SolverStatus};

#[test]
fn checked_core_errors_expose_canonical_status_and_accuracy() {
    let convergence = SolverError::ConvergenceError {
        status: SolverStatus::Stagnated,
        iterations: 7,
        residual: 1.0e-3,
    };
    assert_eq!(convergence.status(), SolverStatus::Stagnated);
    assert_eq!(
        convergence.attainable_accuracy(),
        AccuracyStatus::NotAttained
    );

    let dimension = SolverError::DimensionMismatch {
        expected: 4,
        got: 3,
    };
    assert_eq!(dimension.status(), SolverStatus::NumericalError);
    assert_eq!(dimension.attainable_accuracy(), AccuracyStatus::NotChecked);
}

#[test]
fn checked_core_failures_distinguish_iteration_and_breakdown_outcomes() {
    for status in [
        SolverStatus::MaxIterationsReached,
        SolverStatus::Stagnated,
        SolverStatus::Diverged,
    ] {
        let error = SolverError::ConvergenceError {
            status,
            iterations: 3,
            residual: 1.0,
        };
        assert_eq!(error.status(), status);
        assert_eq!(error.accuracy_status(), AccuracyStatus::NotAttained);
    }

    let breakdown = SolverError::Breakdown {
        operation: "factorization",
        reason: "zero pivot".into(),
    };
    assert_eq!(breakdown.solver_status(), SolverStatus::Breakdown);
    assert_eq!(breakdown.attainable_accuracy(), AccuracyStatus::NotChecked);

    let numerical = SolverError::OperationFailed {
        operation: "residual check",
        reason: "non-finite residual".into(),
    };
    assert_eq!(numerical.status(), SolverStatus::NumericalError);
    assert_eq!(numerical.accuracy_status(), AccuracyStatus::NotChecked);
}
