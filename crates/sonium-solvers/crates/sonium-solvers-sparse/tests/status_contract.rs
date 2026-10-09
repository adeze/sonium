use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use sonium_solvers_sparse::NumaSpmvError;

#[test]
fn numa_spmv_errors_use_shared_failure_vocabulary() {
    let errors = [
        NumaSpmvError::DimensionMismatch {
            expected: 4,
            got: 3,
        },
        NumaSpmvError::InvalidPartition {
            partition: 1,
            reason: "row range is empty".to_owned(),
        },
        NumaSpmvError::NonFinite { operation: "spmv" },
        NumaSpmvError::WorkerPanicked,
    ];

    for error in errors {
        assert_eq!(error.status(), SolverStatus::NumericalError);
        assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
    }
}
