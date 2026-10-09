//! Core traits and dense matrix primitives for the sonium solver family.

/// Construct a vector or a rectangular row-major matrix, evaluating every
/// element exactly once in row order. Matrix rows must have equal length.
///
/// ```compile_fail
/// let _ = sonium_solvers_core::array![[1.0, 2.0], [3.0]];
/// ```
#[macro_export]
macro_rules! array {
    ( [ $( $only:expr ),* $(,)? ] $(,)? ) => {{
        let data = vec![$($only),*];
        $crate::dense::DenseMatrix::from_row_major_vec(1, data.len(), data)
    }};
    ( [ $( $first:expr ),* $(,)? ] $(, [ $( $rest:expr ),* $(,)? ] )+ $(,)? ) => {{
        let rows = [[$($first),*], $([$($rest),*]),+];
        let row_count = rows.len();
        let cols = rows[0].len();
        let data = rows.into_iter().flatten().collect();
        $crate::dense::DenseMatrix::from_row_major_vec(row_count, cols, data)
    }};
    ( $( $value:expr ),* $(,)? ) => {
        vec![$($value),*]
    };
}

#[doc(hidden)]
pub mod blas_helpers;
pub mod dense;
pub mod precision;
pub mod traits;
pub mod vec_ops;

pub use precision::PrecisionCast;
/// BLAS transpose selector re-exported so downstream solver crates can name
/// transposition modes (e.g. [`traits::ComplexField::syrk_lower`]) without
/// depending on `sonium-compute` directly.
pub use sonium_compute::backend::Transpose;
pub use traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner,
    SOLVER_RESULT_CONTRACT_VERSION, SolverError, SolverErrorKind, SolverFailure,
    SolverResultContract, SolverStatus,
};
