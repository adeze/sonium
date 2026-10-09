use sonium_solvers::direct::{
    BlrError, CholeskyError, DenseDirectError, IrError, LuError, QrError, SparseCholeskyError,
    SparseDirectError, SparseLuError,
};
use sonium_solvers::nonlinear::{AndersonError, NewtonError};
use sonium_solvers::optimization::OptimError;
use sonium_solvers::preconditioners::ChebyshevError;
use sonium_solvers::sparse::NumaSpmvError;
use sonium_solvers::time_integration::RevolveError;
use sonium_solvers::{LinearSolveError, MatrixMarketError, SolverErrorKind, SolverFailure};

fn assert_kind(error: &impl SolverFailure, expected: SolverErrorKind) {
    assert_eq!(error.error_kind(), expected);
}

#[test]
fn every_public_solver_family_exposes_the_shared_failure_taxonomy() {
    assert_kind(
        &NumaSpmvError::DimensionMismatch {
            expected: 2,
            got: 1,
        },
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &ChebyshevError::InvalidCsrStructure,
        SolverErrorKind::OperationFailed,
    );

    assert_kind(
        &LuError::DimensionMismatch {
            expected: 2,
            got: 1,
        },
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &CholeskyError::NotPositiveDefinite { pivot: 0 },
        SolverErrorKind::Breakdown,
    );
    assert_kind(
        &QrError::RhsDimension {
            expected: 2,
            actual: 1,
        },
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &SparseLuError::SingularMatrix(0),
        SolverErrorKind::Breakdown,
    );
    assert_kind(
        &SparseCholeskyError::SolutionDimensionMismatch {
            expected: 2,
            actual: 1,
        },
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &BlrError::DimensionMismatch {
            expected: 2,
            actual: 1,
        },
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &IrError::DirectSolve(BlrError::SingularFront(0)),
        SolverErrorKind::Breakdown,
    );
    assert_kind(
        &DenseDirectError::Lu(LuError::SingularMatrix),
        SolverErrorKind::Breakdown,
    );
    assert_kind(
        &SparseDirectError::RhsDimension {
            expected: 2,
            actual: 1,
        },
        SolverErrorKind::DimensionMismatch,
    );

    assert_kind(
        &NewtonError::NotConverged(3, 1.0),
        SolverErrorKind::Convergence,
    );
    assert_kind(
        &AndersonError::DimensionMismatch {
            expected: 2,
            got: 1,
            stage: "test",
        },
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &OptimError::NotConverged {
            iterations: 3,
            grad_norm: 1.0,
        },
        SolverErrorKind::Convergence,
    );
    assert_kind(
        &RevolveError::NoCheckpoint { num_steps: 1 },
        SolverErrorKind::OperationFailed,
    );
    assert_kind(
        &LinearSolveError::DimensionMismatch {
            operation: "rhs",
            expected: 2,
            actual: 1,
        },
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &MatrixMarketError::Parse("malformed banner".into()),
        SolverErrorKind::OperationFailed,
    );
    assert_kind(
        &MatrixMarketError::Io(std::io::Error::other("fixture")),
        SolverErrorKind::OperationFailed,
    );
}

#[test]
fn nested_direct_errors_preserve_the_leaf_category() {
    assert_kind(
        &DenseDirectError::Lu(LuError::SingularMatrix),
        SolverErrorKind::Breakdown,
    );
    assert_kind(
        &SparseDirectError::Multifrontal(BlrError::NotSquare { rows: 2, cols: 3 }),
        SolverErrorKind::DimensionMismatch,
    );
    assert_kind(
        &IrError::DirectSolve(BlrError::NumericalBreakdown {
            stage: "factorization",
            index: 0,
        }),
        SolverErrorKind::Breakdown,
    );
}
