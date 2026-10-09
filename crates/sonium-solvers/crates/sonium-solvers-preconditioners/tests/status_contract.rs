use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use sonium_solvers_preconditioners::ChebyshevError;

#[test]
fn chebyshev_admission_errors_use_shared_failure_vocabulary() {
    let errors = [
        ChebyshevError::ComplexEntries { count: 2 },
        ChebyshevError::NonFiniteEntries { count: 1 },
        ChebyshevError::NonPositiveDiagonal { index: 3 },
        ChebyshevError::InvalidCsrStructure,
    ];

    for error in errors {
        assert_eq!(error.status(), SolverStatus::NumericalError);
        assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
    }
}
