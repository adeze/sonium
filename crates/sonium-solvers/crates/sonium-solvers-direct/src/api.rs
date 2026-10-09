//! Property-aware public interfaces for direct factorization dispatch.

use super::cholesky::{
    CholeskyError, CholeskyFactorization, cholesky_factorize, cholesky_factorize_hermitian,
};
use super::dense_qr::{QrError, QrFactorization};
use super::lu::{LuError, LuFactorization, LuWorkspace, lu_factorize_with_workspace};
use super::multifrontal_blr::{BlrConfig, BlrError, MultifrontalBlrFactorization, PanelKind};
use super::multifrontal_ldlt::{LdltConfig, MultifrontalLdltFactorization};
use super::sparse_cholesky::{
    CholeskyOrdering, SparseCholeskyError, SparseCholeskyFactorization, SparseCholeskySymbolic,
};
use super::sparse_lu::{
    SparseLuError, SparseLuFactorization, SparseLuSymbolic, amd_ordering, colamd_ordering,
    nd_ordering,
};
use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverError, SolverFailure, SolverResultContract, SolverStatus,
};
use sonium_solvers_sparse::CsrMatrix;
use thiserror::Error;

/// Mathematical structure declared by the caller.
///
/// `Auto` dispatch never guesses a stronger property than this declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixClass {
    /// No symmetry or definiteness is promised.
    #[default]
    General,
    /// `A = Aᵀ`, without a definiteness promise.
    SymmetricIndefinite,
    /// `A = Aᴴ`, without a definiteness promise.
    HermitianIndefinite,
    /// `A = Aᵀ` and the matrix is positive definite.
    SymmetricPositiveDefinite,
    /// `A = Aᴴ` and the matrix is positive definite.
    HermitianPositiveDefinite,
    /// Rectangular full-rank matrix.
    RectangularFullRank,
    /// Rectangular or square matrix for which rank detection is required.
    RankSensitive,
}

/// Dense direct factorization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseDirectMethod {
    #[default]
    Auto,
    Lu,
    Cholesky,
    Ldlt,
    Qr,
    PivotedQr,
    Tsqr,
}

/// Sparse direct factorization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseDirectMethod {
    #[default]
    Auto,
    SimplicialCholesky,
    SupernodalCholesky,
    MultifrontalLdlt,
    SimplicialLu,
    SupernodalLu,
    ExactMultifrontalLu,
    MultifrontalBlr,
    MultifrontalHss,
}

/// Fill-reducing ordering requested for sparse symbolic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ordering {
    #[default]
    Auto,
    Natural,
    Amd,
    Colamd,
    NestedDissection,
}

/// Dense direct-solver configuration.
#[derive(Debug, Clone)]
pub struct DenseDirectConfig {
    pub matrix_class: MatrixClass,
    pub method: DenseDirectMethod,
    pub equilibrate: bool,
    pub estimate_condition: bool,
    pub iterative_refinement_steps: usize,
}

impl Default for DenseDirectConfig {
    fn default() -> Self {
        Self {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Auto,
            equilibrate: false,
            estimate_condition: false,
            iterative_refinement_steps: 0,
        }
    }
}

/// Sparse direct-solver configuration.
#[derive(Debug, Clone)]
pub struct SparseDirectConfig {
    pub matrix_class: MatrixClass,
    pub method: SparseDirectMethod,
    pub ordering: Ordering,
    /// MC64-style max-weight matching plus two-sided row/column
    /// equilibration before the numeric sparse LU factorization.
    ///
    /// Off by default: preprocessing changes the numerics of the
    /// factorization (an extra value-dependent column permutation and
    /// row/column scaling) and costs an `O(n·nnz·log n)` matching pass, so
    /// existing callers keep bit-stable behavior unless they opt in. Enable
    /// it for badly scaled matrices, where the matched large-magnitude
    /// diagonal makes partial pivoting far more reliable.
    ///
    /// Only the sparse LU methods implement equilibration; requesting it
    /// for Cholesky / LDLᵀ / multifrontal methods returns
    /// [`SparseDirectError::OptionUnavailable`]. Because the matching is
    /// value-dependent, it is also incompatible with a reused pattern-only
    /// symbolic analysis: pass `symbolic: None` to [`factorize_sparse`]
    /// when `equilibrate` is set.
    pub equilibrate: bool,
    pub pivot_threshold: f64,
    /// Optional admission limit for the reported numeric-factor storage.
    ///
    /// Use [`factorize_sparse_configured`] to enforce this limit. The legacy
    /// [`factorize_sparse`] entry point rejects non-default configured fields
    /// instead of silently ignoring them.
    pub memory_budget_bytes: Option<usize>,
    /// Number of RHS columns expected by the configured block-solve wrapper.
    pub right_hand_sides: usize,
    /// Normwise backward-error tolerance used by configured refinement.
    pub backward_error_tolerance: f64,
}

impl Default for SparseDirectConfig {
    fn default() -> Self {
        Self {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::Auto,
            ordering: Ordering::Auto,
            equilibrate: false,
            pivot_threshold: 1.0,
            memory_budget_bytes: None,
            right_hand_sides: 1,
            backward_error_tolerance: 1e-12,
        }
    }
}

/// Scalar-independent reusable sparse analysis.
#[derive(Debug, Clone)]
pub enum SparseSymbolic {
    Lu(SparseLuSymbolic),
    Cholesky(SparseCholeskySymbolic),
}

/// Typed sparse factors available through the compatibility foundation.
pub enum SparseFactorization<T: ComplexField> {
    Lu(SparseLuFactorization<T>),
    Cholesky(SparseCholeskyFactorization<T>),
    Multifrontal(MultifrontalBlrFactorization<T>),
    MultifrontalLdlt(MultifrontalLdltFactorization<T>),
}

impl<T: ComplexField + FromPrimitive> SparseFactorization<T> {
    pub fn dimension(&self) -> usize {
        match self {
            Self::Lu(factor) => factor.dimension(),
            Self::Cholesky(factor) => factor.dimension(),
            Self::Multifrontal(factor) => factor.dimension(),
            Self::MultifrontalLdlt(factor) => factor.dimension(),
        }
    }

    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SparseDirectError> {
        let expected = self.dimension();
        if b.len() != expected {
            return Err(SparseDirectError::RhsDimension {
                expected,
                actual: b.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(SparseDirectError::NonFiniteRhs { index });
        }
        match self {
            Self::Lu(factor) => factor.solve(b).map_err(Into::into),
            Self::Cholesky(factor) => factor.solve(b).map_err(Into::into),
            Self::Multifrontal(factor) => factor.try_solve(b).map_err(Into::into),
            Self::MultifrontalLdlt(factor) => factor.try_solve(b).map_err(Into::into),
        }
    }

    /// Checked row-major multi-right-hand-side solve into caller-owned storage.
    ///
    /// The destination is transactional: it is copied only after every RHS
    /// has solved successfully, so a later numerical failure cannot publish
    /// a partial block result.
    pub fn solve_many_into(
        &self,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<(), SparseDirectError> {
        let dimension = self.dimension();
        let expected = dimension
            .checked_mul(nrhs)
            .ok_or(SparseDirectError::RhsSizeOverflow { dimension, nrhs })?;
        if b.len() != expected {
            return Err(SparseDirectError::RhsDimension {
                expected,
                actual: b.len(),
            });
        }
        if x.len() != expected {
            return Err(SparseDirectError::SolutionDimension {
                expected,
                actual: x.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(SparseDirectError::NonFiniteRhs { index });
        }

        match self {
            Self::Lu(factor) => factor.solve_many_into(b, x, nrhs).map_err(Into::into),
            Self::Cholesky(factor) => factor.solve_many_into(b, x, nrhs).map_err(Into::into),
            Self::Multifrontal(_) | Self::MultifrontalLdlt(_) => {
                use rayon::prelude::*;
                // Each RHS column solves independently through the shared
                // factor (`&self`), so columns fan out over the pool with
                // thread-local repack buffers instead of paying full latency
                // per column on one thread.
                let columns: Result<Vec<Vec<T>>, SparseDirectError> = (0..nrhs)
                    .into_par_iter()
                    .map(|rhs| {
                        let column_rhs: Vec<T> =
                            (0..dimension).map(|row| b[row * nrhs + rhs]).collect();
                        self.solve(&column_rhs)
                    })
                    .collect();
                let columns = columns?;
                let mut solutions = vec![T::zero(); expected];
                for (rhs, solution) in columns.iter().enumerate() {
                    for row in 0..dimension {
                        solutions[row * nrhs + rhs] = solution[row];
                    }
                }
                x.copy_from_slice(&solutions);
                Ok(())
            }
        }
    }

    /// Checked solve into caller-owned storage.
    ///
    /// The multifrontal variants use a checked temporary solution because
    /// their low-level checked APIs currently return an owned vector.
    pub fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), SparseDirectError> {
        let expected = self.dimension();
        if b.len() != expected {
            return Err(SparseDirectError::RhsDimension {
                expected,
                actual: b.len(),
            });
        }
        if x.len() != expected {
            return Err(SparseDirectError::SolutionDimension {
                expected,
                actual: x.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(SparseDirectError::NonFiniteRhs { index });
        }
        // Keep the destination transactional: no partial solution is exposed
        // when a checked factor solve reports a numerical failure.
        let solution = self.solve(b)?;
        if let Some(index) = first_non_finite(&solution) {
            return Err(SparseDirectError::NonFiniteSolution { index });
        }
        x.copy_from_slice(&solution);
        Ok(())
    }

    pub fn factor_nonzeros(&self) -> usize {
        match self {
            Self::Lu(factor) => factor.factor_nonzeros(),
            Self::Cholesky(factor) => factor.factor_nonzeros(),
            Self::Multifrontal(factor) => factor.factor_memory_bytes() / std::mem::size_of::<T>(),
            Self::MultifrontalLdlt(factor) => {
                factor.factor_memory_bytes() / std::mem::size_of::<T>()
            }
        }
    }

    pub fn factor_memory_bytes(&self) -> usize {
        match self {
            Self::Lu(factor) => factor
                .factor_nonzeros()
                .saturating_mul(std::mem::size_of::<T>()),
            Self::Cholesky(factor) => factor
                .factor_nonzeros()
                .saturating_mul(std::mem::size_of::<T>()),
            Self::Multifrontal(factor) => factor.factor_memory_bytes(),
            Self::MultifrontalLdlt(factor) => factor.factor_memory_bytes(),
        }
    }
}

/// Errors from property-aware dense dispatch.
#[derive(Debug, Error)]
pub enum DenseDirectError {
    #[error(transparent)]
    Lu(#[from] LuError),
    #[error(transparent)]
    Cholesky(#[from] CholeskyError),
    #[error(transparent)]
    Qr(#[from] QrError),
    #[error(transparent)]
    Ldlt(#[from] BlrError),
    #[error("right-hand side has length {actual}, expected {expected}")]
    RhsDimension { expected: usize, actual: usize },
    #[error("multi-RHS shape overflows usize (dimension={dimension}, nrhs={nrhs})")]
    SizeOverflow { dimension: usize, nrhs: usize },
    #[error("solution has length {actual}, expected {expected}")]
    SolutionDimension { expected: usize, actual: usize },
    #[error("dense direct matrix contains a non-finite value at storage index {index}")]
    NonFiniteMatrix { index: usize },
    #[error("dense direct right-hand side contains a non-finite value at index {index}")]
    NonFiniteRhs { index: usize },
    #[error("dense direct solution contains a non-finite value at index {index}")]
    NonFiniteSolution { index: usize },
    #[error("dense direct metadata `{field}` is non-finite")]
    NonFiniteMetadata { field: &'static str },
    #[error("{method:?} is not valid for declared matrix class {matrix_class:?}")]
    InvalidMethod {
        matrix_class: MatrixClass,
        method: DenseDirectMethod,
    },
    #[error("{method:?} is part of the public roadmap but is not implemented yet")]
    MethodUnavailable { method: DenseDirectMethod },
    #[error("dense direct option {option} is not implemented yet")]
    OptionUnavailable { option: &'static str },
    #[error("dense direct iterative-refinement tolerance must be finite and non-negative")]
    InvalidRefinementTolerance,
}

impl DenseDirectError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::Lu(error) => error.status(),
            Self::Cholesky(error) => error.status(),
            Self::Qr(error) => error.status(),
            Self::Ldlt(error) => error.status(),
            Self::SizeOverflow { .. }
            | Self::RhsDimension { .. }
            | Self::SolutionDimension { .. }
            | Self::NonFiniteMatrix { .. }
            | Self::NonFiniteRhs { .. }
            | Self::NonFiniteSolution { .. }
            | Self::NonFiniteMetadata { .. }
            | Self::InvalidMethod { .. }
            | Self::MethodUnavailable { .. }
            | Self::OptionUnavailable { .. }
            | Self::InvalidRefinementTolerance => SolverStatus::NumericalError,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    /// Direct failures occur before a valid accuracy result is available.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverFailure for DenseDirectError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::Lu(error) => error.error_kind(),
            Self::Cholesky(error) => error.error_kind(),
            Self::Qr(error) => error.error_kind(),
            Self::Ldlt(error) => error.error_kind(),
            Self::RhsDimension { .. }
            | Self::SizeOverflow { .. }
            | Self::SolutionDimension { .. } => SolverErrorKind::DimensionMismatch,
            Self::MethodUnavailable { .. } | Self::OptionUnavailable { .. } => {
                SolverErrorKind::UnsupportedOperation
            }
            Self::InvalidRefinementTolerance => SolverErrorKind::OperationFailed,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        DenseDirectError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        DenseDirectError::accuracy_status(self)
    }
}

/// Errors from property-aware sparse dispatch.
#[derive(Debug, Error)]
pub enum SparseDirectError {
    #[error(transparent)]
    Lu(#[from] SparseLuError),
    #[error(transparent)]
    Cholesky(#[from] SparseCholeskyError),
    #[error(transparent)]
    Multifrontal(#[from] BlrError),
    #[error("right-hand side has length {actual}, expected {expected}")]
    RhsDimension { expected: usize, actual: usize },
    #[error("multi-RHS shape overflows usize (dimension={dimension}, nrhs={nrhs})")]
    RhsSizeOverflow { dimension: usize, nrhs: usize },
    #[error("solution has length {actual}, expected {expected}")]
    SolutionDimension { expected: usize, actual: usize },
    #[error("sparse direct matrix contains a non-finite value at storage index {index}")]
    NonFiniteMatrix { index: usize },
    #[error("sparse direct right-hand side contains a non-finite value at index {index}")]
    NonFiniteRhs { index: usize },
    #[error("sparse direct solution contains a non-finite value at index {index}")]
    NonFiniteSolution { index: usize },
    #[error("sparse direct metadata `{field}` is non-finite")]
    NonFiniteMetadata { field: &'static str },
    #[error("{method:?} is not valid for declared matrix class {matrix_class:?}")]
    InvalidMethod {
        matrix_class: MatrixClass,
        method: SparseDirectMethod,
    },
    #[error("{method:?} is part of the public roadmap but is not implemented yet")]
    MethodUnavailable { method: SparseDirectMethod },
    #[error("reusable symbolic analysis does not match {method:?}")]
    SymbolicMismatch { method: SparseDirectMethod },
    #[error("sparse direct option {option} is not available for {method:?}")]
    OptionUnavailable {
        option: &'static str,
        method: SparseDirectMethod,
    },
    #[error("sparse direct configuration field `{field}` is invalid: {reason}")]
    InvalidConfiguration {
        field: &'static str,
        reason: &'static str,
    },
    #[error(
        "sparse direct factorization requires {required_bytes} bytes, exceeding the configured {budget_bytes}-byte budget for {method:?}"
    )]
    MemoryBudgetExceeded {
        method: SparseDirectMethod,
        budget_bytes: usize,
        required_bytes: usize,
    },
    #[error("configured sparse factor expects {expected} right-hand sides, received {actual}")]
    RightHandSideCountMismatch { expected: usize, actual: usize },
    #[error("{ordering:?} ordering is not available for {method:?}")]
    OrderingUnavailable {
        ordering: Ordering,
        method: SparseDirectMethod,
    },
}

/// Metadata returned by a successful checked direct solve.
///
/// A direct factor does not retain the original operator, so a successful
/// factor solve can establish that a finite solution was produced but cannot
/// claim a true residual or backward-error tolerance on its own.  The default
/// result therefore reports `Converged` with `NotChecked` accuracy and no
/// residual fields.  Call [`DirectSolveResult::with_accuracy_check`] only when
/// the caller has independently recomputed those diagnostics against the
/// original operator.
#[derive(Debug, Clone)]
pub struct DirectSolveResult<T: ComplexField> {
    /// Solution vector, in the same ordering as the right-hand side.
    pub x: Vec<T>,
    /// Direct factor solves do not perform Krylov iterations; configured
    /// operator-retaining refinement reports its correction-solve count here.
    pub iterations: usize,
    /// Independently recomputed residual norm, when supplied.
    pub residual: Option<T::Real>,
    /// Independently recomputed normwise backward error, when supplied.
    pub backward_error: Option<T::Real>,
    /// Canonical operation status.
    pub status: SolverStatus,
    /// Whether an independently supplied accuracy check passed.
    pub attainable_accuracy: AccuracyStatus,
}

impl<T: ComplexField> DirectSolveResult<T> {
    /// Construct the conservative result for a finite direct solution.
    pub fn from_solution(x: Vec<T>) -> Self {
        Self {
            x,
            iterations: 0,
            residual: None,
            backward_error: None,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::NotChecked,
        }
    }

    /// Borrow the computed solution without exposing metadata internals.
    pub fn solution(&self) -> &[T] {
        &self.x
    }

    /// Attach independently recomputed residual diagnostics.
    ///
    /// `tolerance` is applied to the backward error.  A failed check is
    /// represented as a typed `NumericalError` result rather than silently
    /// publishing `Converged` with unattained accuracy.
    pub fn with_accuracy_check(
        mut self,
        residual: T::Real,
        backward_error: T::Real,
        tolerance: T::Real,
    ) -> Result<Self, SolverError> {
        if !residual.is_finite() || residual < T::Real::zero() {
            return Err(SolverError::OperationFailed {
                operation: "direct solve accuracy check",
                reason: "residual must be finite and non-negative".into(),
            });
        }
        if !backward_error.is_finite() || backward_error < T::Real::zero() {
            return Err(SolverError::OperationFailed {
                operation: "direct solve accuracy check",
                reason: "backward error must be finite and non-negative".into(),
            });
        }
        if !tolerance.is_finite() || tolerance < T::Real::zero() {
            return Err(SolverError::OperationFailed {
                operation: "direct solve accuracy check",
                reason: "tolerance must be finite and non-negative".into(),
            });
        }

        self.residual = Some(residual);
        self.backward_error = Some(backward_error);
        if backward_error <= tolerance {
            self.status = SolverStatus::Converged;
            self.attainable_accuracy = AccuracyStatus::Attained;
        } else {
            self.status = SolverStatus::NumericalError;
            self.attainable_accuracy = AccuracyStatus::NotAttained;
        }
        Ok(self)
    }
}

impl<T: ComplexField> SolverResultContract for DirectSolveResult<T> {
    fn solver_status(&self) -> SolverStatus {
        self.status
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.x.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "direct solve result contract",
                reason: "solution contains a non-finite value".into(),
            });
        }

        match (self.residual, self.backward_error) {
            (Some(residual), Some(backward_error)) => {
                if !residual.is_finite()
                    || residual < T::Real::zero()
                    || !backward_error.is_finite()
                    || backward_error < T::Real::zero()
                {
                    return Err(SolverError::OperationFailed {
                        operation: "direct solve result contract",
                        reason: "diagnostics must be finite and non-negative".into(),
                    });
                }
            }
            (None, None) => {}
            _ => {
                return Err(SolverError::OperationFailed {
                    operation: "direct solve result contract",
                    reason: "residual and backward error must be supplied together".into(),
                });
            }
        }

        let metadata_matches = match self.status {
            SolverStatus::Converged => matches!(
                self.attainable_accuracy,
                AccuracyStatus::NotChecked | AccuracyStatus::Attained
            ),
            _ => self.attainable_accuracy != AccuracyStatus::Attained,
        };
        if !metadata_matches {
            return Err(SolverError::OperationFailed {
                operation: "direct solve result contract",
                reason: "status and attainable accuracy disagree".into(),
            });
        }
        if self.attainable_accuracy == AccuracyStatus::Attained
            && (self.residual.is_none() || self.backward_error.is_none())
        {
            return Err(SolverError::OperationFailed {
                operation: "direct solve result contract",
                reason: "attained accuracy requires residual diagnostics".into(),
            });
        }
        if self.status == SolverStatus::Converged
            && self.attainable_accuracy == AccuracyStatus::NotChecked
            && (self.residual.is_some() || self.backward_error.is_some())
        {
            return Err(SolverError::OperationFailed {
                operation: "direct solve result contract",
                reason: "unchecked accuracy cannot carry diagnostics".into(),
            });
        }
        Ok(())
    }
}

impl SparseDirectError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::Lu(error) => error.status(),
            Self::Cholesky(error) => error.status(),
            Self::Multifrontal(error) => error.status(),
            Self::RhsSizeOverflow { .. }
            | Self::RhsDimension { .. }
            | Self::SolutionDimension { .. }
            | Self::NonFiniteMatrix { .. }
            | Self::NonFiniteRhs { .. }
            | Self::NonFiniteSolution { .. }
            | Self::NonFiniteMetadata { .. }
            | Self::InvalidMethod { .. }
            | Self::MethodUnavailable { .. }
            | Self::SymbolicMismatch { .. }
            | Self::OptionUnavailable { .. }
            | Self::InvalidConfiguration { .. }
            | Self::MemoryBudgetExceeded { .. }
            | Self::RightHandSideCountMismatch { .. }
            | Self::OrderingUnavailable { .. } => SolverStatus::NumericalError,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    /// Direct failures occur before a valid accuracy result is available.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverFailure for SparseDirectError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::Lu(error) => error.error_kind(),
            Self::Cholesky(error) => error.error_kind(),
            Self::Multifrontal(error) => error.error_kind(),
            Self::RhsSizeOverflow { .. }
            | Self::RhsDimension { .. }
            | Self::SolutionDimension { .. } => SolverErrorKind::DimensionMismatch,
            Self::MethodUnavailable { .. }
            | Self::OptionUnavailable { .. }
            | Self::MemoryBudgetExceeded { .. }
            | Self::OrderingUnavailable { .. } => SolverErrorKind::UnsupportedOperation,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        SparseDirectError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        SparseDirectError::accuracy_status(self)
    }
}

/// Dense iterative-refinement result produced by a configured factor.
#[derive(Debug, Clone)]
pub struct DenseRefinementResult<T: ComplexField> {
    /// Refined solution vector.
    pub x: Vec<T>,
    /// Number of correction solves performed.
    pub iterations: usize,
    /// Backward error before the first correction solve.
    pub initial_backward_error: T::Real,
    /// Backward error after the final correction solve.
    pub backward_error: T::Real,
    /// Requested stopping tolerance.
    pub tolerance: T::Real,
    /// Whether the requested tolerance was reached.
    pub converged: bool,
}

/// Status and contract methods for dense iterative-refinement results.
impl<T: ComplexField> DenseRefinementResult<T> {
    /// Compatibility alias for callers using the shorter status spelling.
    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    /// Compatibility alias for legacy attainable-accuracy callers.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }

    /// Return the canonical status represented by this refinement result.
    pub fn solver_status(&self) -> SolverStatus {
        let finite = self.initial_backward_error.is_finite()
            && self.backward_error.is_finite()
            && self.tolerance.is_finite()
            && self.initial_backward_error >= T::Real::zero()
            && self.backward_error >= T::Real::zero()
            && self.tolerance >= T::Real::zero()
            && self.x.iter().all(|value| value.norm().is_finite());
        if !finite {
            SolverStatus::NumericalError
        } else if self.converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        }
    }

    /// Return whether the requested backward-error tolerance was attained.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached => AccuracyStatus::NotAttained,
            _ => AccuracyStatus::NotChecked,
        }
    }
}

impl<T: ComplexField> SolverResultContract for DenseRefinementResult<T> {
    fn solver_status(&self) -> SolverStatus {
        DenseRefinementResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        DenseRefinementResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if !self.initial_backward_error.is_finite()
            || !self.backward_error.is_finite()
            || !self.tolerance.is_finite()
            || self.initial_backward_error < T::Real::zero()
            || self.backward_error < T::Real::zero()
            || self.tolerance < T::Real::zero()
            || self.x.iter().any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "dense refinement result contract",
                reason: "result contains non-finite or negative diagnostics".into(),
            });
        }
        if self.converged && self.backward_error > self.tolerance {
            return Err(SolverError::OperationFailed {
                operation: "dense refinement result contract",
                reason: "convergence metadata disagrees with backward-error tolerance".into(),
            });
        }
        Ok(())
    }
}

/// Sparse iterative-refinement result produced by a configured factor.
#[derive(Debug, Clone)]
pub struct SparseConfiguredRefinementResult<T: ComplexField> {
    /// Refined solution vector.
    pub x: Vec<T>,
    /// Number of correction solves performed.
    pub iterations: usize,
    /// Normwise backward error before the first correction solve.
    pub initial_backward_error: T::Real,
    /// Normwise backward error after the final correction solve.
    pub backward_error: T::Real,
    /// Configured stopping tolerance.
    pub tolerance: T::Real,
    /// Whether the configured tolerance was reached.
    pub converged: bool,
}

impl<T: ComplexField> SparseConfiguredRefinementResult<T> {
    /// Return the canonical status for checked configured refinement.
    pub fn solver_status(&self) -> SolverStatus {
        let finite = self.initial_backward_error.is_finite()
            && self.backward_error.is_finite()
            && self.tolerance.is_finite()
            && self.initial_backward_error >= T::Real::zero()
            && self.backward_error >= T::Real::zero()
            && self.tolerance >= T::Real::zero()
            && self.x.iter().all(|value| value.norm().is_finite());
        if !finite {
            SolverStatus::NumericalError
        } else if self.converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        }
    }

    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached => AccuracyStatus::NotAttained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: ComplexField> SolverResultContract for SparseConfiguredRefinementResult<T> {
    fn solver_status(&self) -> SolverStatus {
        SparseConfiguredRefinementResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        SparseConfiguredRefinementResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.initial_backward_error < T::Real::zero()
            || self.backward_error < T::Real::zero()
            || self.tolerance < T::Real::zero()
            || !self.initial_backward_error.is_finite()
            || !self.backward_error.is_finite()
            || !self.tolerance.is_finite()
            || self.x.iter().any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "configured sparse refinement result contract",
                reason: "refinement values or metadata are invalid".into(),
            });
        }
        if self.converged != (self.solver_status() == SolverStatus::Converged)
            || (self.converged && self.backward_error > self.tolerance)
        {
            return Err(SolverError::OperationFailed {
                operation: "configured sparse refinement result contract",
                reason: "convergence metadata disagrees with backward-error tolerance".into(),
            });
        }
        Ok(())
    }
}

/// Sparse factor together with the source matrix and opt-in diagnostics.
///
/// The legacy [`SparseFactorization`] remains a matrix-free factor. This
/// wrapper is the explicit owner of the source matrix required for checked
/// refinement and of the configured multi-RHS contract.
pub struct SparseConfiguredFactorization<T: ComplexField> {
    factor: SparseFactorization<T>,
    matrix: CsrMatrix<T>,
    right_hand_sides: usize,
    backward_error_tolerance: T::Real,
}

impl<T: ComplexField + FromPrimitive> SparseConfiguredFactorization<T> {
    /// Borrow the underlying sparse factor.
    pub fn factor(&self) -> &SparseFactorization<T> {
        &self.factor
    }

    /// Borrow the retained source matrix used by diagnostics and refinement.
    pub fn matrix(&self) -> &CsrMatrix<T> {
        &self.matrix
    }

    /// Number of RHS columns admitted by the configured block-solve contract.
    pub fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Backward-error tolerance used by [`Self::solve_refined`].
    pub fn backward_error_tolerance(&self) -> T::Real {
        self.backward_error_tolerance
    }

    /// Solve one configured RHS with a caller-supplied correction budget.
    pub fn solve_refined(
        &self,
        b: &[T],
        max_iterations: usize,
    ) -> Result<SparseConfiguredRefinementResult<T>, SparseDirectError> {
        if b.len() != self.matrix.num_rows {
            return Err(SparseDirectError::RhsDimension {
                expected: self.matrix.num_rows,
                actual: b.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(SparseDirectError::NonFiniteRhs { index });
        }

        let mut x = self.factor.solve(b)?;
        let (initial_backward_error, _) = sparse_diagnostics(&self.matrix, &x, b)?;
        let mut backward_error = initial_backward_error;
        let mut iterations = 0;
        if backward_error <= self.backward_error_tolerance {
            return Ok(SparseConfiguredRefinementResult {
                x,
                iterations,
                initial_backward_error,
                backward_error,
                tolerance: self.backward_error_tolerance,
                converged: true,
            });
        }

        while iterations < max_iterations {
            let ax = self.matrix.matvec(&x);
            let residual: Vec<T> = b.iter().zip(ax).map(|(rhs, value)| *rhs - value).collect();
            if let Some(index) = first_non_finite(&residual) {
                return Err(SparseDirectError::NonFiniteSolution { index });
            }
            let correction = self.factor.solve(&residual)?;
            for (value, delta) in x.iter_mut().zip(correction) {
                *value += delta;
            }
            if let Some(index) = first_non_finite(&x) {
                return Err(SparseDirectError::NonFiniteSolution { index });
            }
            iterations += 1;
            backward_error = sparse_diagnostics(&self.matrix, &x, b)?.0;
            if backward_error <= self.backward_error_tolerance {
                break;
            }
        }

        Ok(SparseConfiguredRefinementResult {
            x,
            iterations,
            initial_backward_error,
            backward_error,
            tolerance: self.backward_error_tolerance,
            converged: backward_error <= self.backward_error_tolerance,
        })
    }

    /// Refine every configured row-major RHS column transactionally.
    ///
    /// The returned vector is published only after every column has passed
    /// the same checked solve/refinement boundary.  A failure in a later
    /// column therefore cannot expose a partial block result.
    pub fn solve_refined_many(
        &self,
        b: &[T],
        nrhs: usize,
        max_iterations: usize,
    ) -> Result<Vec<SparseConfiguredRefinementResult<T>>, SparseDirectError> {
        if nrhs != self.right_hand_sides {
            return Err(SparseDirectError::RightHandSideCountMismatch {
                expected: self.right_hand_sides,
                actual: nrhs,
            });
        }
        let expected =
            self.matrix
                .num_rows
                .checked_mul(nrhs)
                .ok_or(SparseDirectError::RhsSizeOverflow {
                    dimension: self.matrix.num_rows,
                    nrhs,
                })?;
        if b.len() != expected {
            return Err(SparseDirectError::RhsDimension {
                expected,
                actual: b.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(SparseDirectError::NonFiniteRhs { index });
        }

        let mut results = Vec::with_capacity(nrhs);
        let mut column_rhs = vec![T::zero(); self.matrix.num_rows];
        for rhs in 0..nrhs {
            for row in 0..self.matrix.num_rows {
                column_rhs[row] = b[row * nrhs + rhs];
            }
            results.push(self.solve_refined(&column_rhs, max_iterations)?);
        }
        Ok(results)
    }

    /// Checked row-major block solve using the configured RHS count.
    pub fn solve_many_into(
        &self,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<(), SparseDirectError> {
        if nrhs != self.right_hand_sides {
            return Err(SparseDirectError::RightHandSideCountMismatch {
                expected: self.right_hand_sides,
                actual: nrhs,
            });
        }
        self.factor.solve_many_into(b, x, nrhs)
    }
}

impl<T: ComplexField + FromPrimitive> DirectFactorization<T> for SparseConfiguredFactorization<T> {
    type Error = SparseDirectError;

    fn dimension(&self) -> usize {
        self.factor.dimension()
    }

    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error> {
        self.factor.solve_into(b, x)
    }

    fn factor_nonzeros(&self) -> usize {
        self.factor.factor_nonzeros()
    }

    fn factor_memory_bytes(&self) -> usize {
        self.factor.factor_memory_bytes()
    }

    fn solve_with_result(&self, b: &[T]) -> Result<DirectSolveResult<T>, Self::Error> {
        let solution = self.factor.solve(b)?;
        let (backward_error, residual) = sparse_diagnostics(&self.matrix, &solution, b)?;
        let result = DirectSolveResult::from_solution(solution);
        result
            .with_accuracy_check(residual, backward_error, self.backward_error_tolerance)
            .map_err(|_| SparseDirectError::NonFiniteMetadata {
                field: "configured sparse direct result",
            })
    }
}

/// Dense factorization selected from a declared matrix class.
#[derive(Clone)]
pub enum DenseFactorization<T: ComplexField> {
    /// General square LU factorization.
    Lu(LuFactorization<T>),
    /// Symmetric/Hermitian positive-definite Cholesky factorization.
    Cholesky(CholeskyFactorization<T>),
    /// Symmetric/Hermitian indefinite multifrontal LDLT/LDLH factorization.
    Ldlt(std::sync::Arc<MultifrontalLdltFactorization<T>>),
    /// Thin QR factorization, including the TSQR dispatch path.
    Qr(QrFactorization<T>),
}

impl<T: ComplexField> std::fmt::Debug for DenseFactorization<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (method, dimension) = match self {
            Self::Lu(factor) => ("Lu", factor.n),
            Self::Cholesky(factor) => ("Cholesky", factor.dimension()),
            Self::Ldlt(factor) => ("Ldlt", factor.dimension()),
            Self::Qr(factor) => ("Qr", factor.columns()),
        };
        formatter
            .debug_struct("DenseFactorization")
            .field("method", &method)
            .field("dimension", &dimension)
            .finish_non_exhaustive()
    }
}

impl<T: ComplexField> DenseFactorization<T> {
    /// Number of entries expected in a right-hand side for this factor.
    /// Rectangular QR factors consume one value per matrix row, while square
    /// direct factors consume one value per factor dimension.
    pub fn rhs_dimension(&self) -> usize {
        match self {
            Self::Lu(factor) => factor.n,
            Self::Cholesky(factor) => factor.dimension(),
            Self::Ldlt(factor) => factor.dimension(),
            Self::Qr(factor) => factor.rows(),
        }
    }

    /// Checked row-major multi-right-hand-side solve into caller-owned storage.
    ///
    /// The destination is transactional across all factor classes; a failure
    /// on any RHS leaves the caller's existing values untouched.
    pub fn solve_many_into(&self, b: &[T], x: &mut [T], nrhs: usize) -> Result<(), DenseDirectError>
    where
        T: FromPrimitive,
    {
        let rhs_dimension = self.rhs_dimension();
        let solution_dimension = self.dimension();
        let expected_rhs =
            rhs_dimension
                .checked_mul(nrhs)
                .ok_or(DenseDirectError::SizeOverflow {
                    dimension: rhs_dimension,
                    nrhs,
                })?;
        let expected_solution =
            solution_dimension
                .checked_mul(nrhs)
                .ok_or(DenseDirectError::SizeOverflow {
                    dimension: solution_dimension,
                    nrhs,
                })?;
        if b.len() != expected_rhs {
            return Err(DenseDirectError::RhsDimension {
                expected: expected_rhs,
                actual: b.len(),
            });
        }
        if x.len() != expected_solution {
            return Err(DenseDirectError::SolutionDimension {
                expected: expected_solution,
                actual: x.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(DenseDirectError::NonFiniteRhs { index });
        }

        match self {
            Self::Lu(factor) if rhs_dimension == solution_dimension => {
                let mut solutions = vec![T::zero(); expected_solution];
                factor
                    .solve_many_into(b, &mut solutions, nrhs)
                    .map_err(DenseDirectError::Lu)?;
                x.copy_from_slice(&solutions);
                Ok(())
            }
            Self::Cholesky(factor) if rhs_dimension == solution_dimension => {
                let mut solutions = vec![T::zero(); expected_solution];
                factor
                    .solve_many_into(b, &mut solutions, nrhs)
                    .map_err(DenseDirectError::Cholesky)?;
                x.copy_from_slice(&solutions);
                Ok(())
            }
            Self::Qr(factor) => {
                let mut solutions = vec![T::zero(); expected_solution];
                factor
                    .solve_least_squares_many_into(b, &mut solutions, nrhs)
                    .map_err(DenseDirectError::Qr)?;
                x.copy_from_slice(&solutions);
                Ok(())
            }
            Self::Ldlt(_) | Self::Lu(_) | Self::Cholesky(_) => {
                let mut solutions = vec![T::zero(); expected_solution];
                let mut column_rhs = vec![T::zero(); rhs_dimension];
                for rhs in 0..nrhs {
                    for row in 0..rhs_dimension {
                        column_rhs[row] = b[row * nrhs + rhs];
                    }
                    let mut solution = vec![T::zero(); solution_dimension];
                    self.solve_into(&column_rhs, &mut solution)?;
                    for row in 0..solution_dimension {
                        solutions[row * nrhs + rhs] = solution[row];
                    }
                }
                x.copy_from_slice(&solutions);
                Ok(())
            }
        }
    }
}

/// A dense factor together with the source matrix and opt-in diagnostics.
///
/// [`DenseFactorization`] deliberately stores only factor data so it remains
/// cheap to move and serialise.  Operations that need the original operator
/// (condition estimation and iterative refinement) therefore use this
/// explicit owning wrapper instead of silently retaining a hidden matrix.
#[derive(Debug, Clone)]
pub struct DenseConfiguredFactorization<T: ComplexField> {
    factor: DenseFactorization<T>,
    matrix: DenseMatrix<T>,
    refinement_steps: usize,
    condition_estimate: Option<T::Real>,
}

impl<T: ComplexField + FromPrimitive> DenseConfiguredFactorization<T> {
    /// Borrow the underlying factor for callers that need method-specific APIs.
    pub fn factor(&self) -> &DenseFactorization<T> {
        &self.factor
    }

    /// Borrow the retained source matrix used by diagnostics and refinement.
    pub fn matrix(&self) -> &DenseMatrix<T> {
        &self.matrix
    }

    /// Return the optional one-norm condition estimate requested at setup.
    pub fn condition_estimate(&self) -> Option<T::Real> {
        self.condition_estimate
    }

    /// Solve using the configured number of iterative-refinement corrections.
    pub fn solve_refined(&self, b: &[T]) -> Result<DenseRefinementResult<T>, DenseDirectError> {
        let tolerance = T::Real::epsilon()
            * T::Real::from_usize(self.matrix.nrows().max(1)).unwrap_or_else(T::Real::one)
            * T::Real::from_f64(10.0).unwrap_or_else(T::Real::one);
        self.solve_refined_with_tolerance(b, tolerance, self.refinement_steps)
    }

    /// Solve with a caller-supplied backward-error tolerance and correction cap.
    pub fn solve_refined_with_tolerance(
        &self,
        b: &[T],
        tolerance: T::Real,
        max_iterations: usize,
    ) -> Result<DenseRefinementResult<T>, DenseDirectError> {
        if !tolerance.is_finite() || tolerance < T::Real::zero() {
            return Err(DenseDirectError::InvalidRefinementTolerance);
        }
        if b.len() != self.matrix.nrows() {
            return Err(DenseDirectError::RhsDimension {
                expected: self.matrix.nrows(),
                actual: b.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(DenseDirectError::NonFiniteRhs { index });
        }
        if self.matrix.nrows() != self.matrix.ncols() && max_iterations != 0 {
            return Err(DenseDirectError::OptionUnavailable {
                option: "iterative refinement for rectangular factors",
            });
        }

        let mut x = self.factor.solve(b)?;
        let initial_backward_error = dense_backward_error(&self.matrix, &x, b)?;
        let mut backward_error = initial_backward_error;
        let mut iterations = 0;
        if backward_error <= tolerance {
            return Ok(DenseRefinementResult {
                x,
                iterations,
                initial_backward_error,
                backward_error,
                tolerance,
                converged: true,
            });
        }

        while iterations < max_iterations {
            let mut ax = vec![T::zero(); self.matrix.nrows()];
            self.matrix.matvec_into(&x, &mut ax);
            let residual: Vec<T> = b.iter().zip(ax).map(|(rhs, value)| *rhs - value).collect();
            if first_non_finite(&residual).is_some() {
                return Err(DenseDirectError::NonFiniteMetadata {
                    field: "iterative-refinement residual",
                });
            }
            let correction = self.factor.solve(&residual)?;
            for (value, delta) in x.iter_mut().zip(correction) {
                *value += delta;
            }
            if let Some(index) = first_non_finite(&x) {
                return Err(DenseDirectError::NonFiniteSolution { index });
            }
            iterations += 1;
            backward_error = dense_backward_error(&self.matrix, &x, b)?;
            if backward_error <= tolerance {
                break;
            }
        }

        Ok(DenseRefinementResult {
            x,
            iterations,
            initial_backward_error,
            backward_error,
            tolerance,
            converged: backward_error <= tolerance,
        })
    }
}

impl<T: ComplexField + FromPrimitive> DirectFactorization<T> for DenseConfiguredFactorization<T> {
    type Error = DenseDirectError;

    fn dimension(&self) -> usize {
        self.factor.dimension()
    }

    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error> {
        if x.len() != self.dimension() {
            return Err(DenseDirectError::SolutionDimension {
                expected: self.dimension(),
                actual: x.len(),
            });
        }
        let result = self.solve_refined(b)?;
        x.copy_from_slice(&result.x);
        Ok(())
    }

    fn factor_nonzeros(&self) -> usize {
        self.factor.factor_nonzeros()
    }

    fn factor_memory_bytes(&self) -> usize {
        self.factor.factor_memory_bytes()
    }

    fn condition_estimate(&self) -> Option<T::Real> {
        self.condition_estimate
    }

    fn solve_with_result(&self, b: &[T]) -> Result<DirectSolveResult<T>, Self::Error> {
        let refined = self.solve_refined(b)?;
        let residual = dense_residual_norm(&self.matrix, &refined.x, b)?;
        let mut result = DirectSolveResult::from_solution(refined.x);
        result.iterations = refined.iterations;
        result
            .with_accuracy_check(residual, refined.backward_error, refined.tolerance)
            .map_err(|_| DenseDirectError::NonFiniteMetadata {
                field: "iterative-refinement result",
            })
    }
}

/// Factor a dense matrix with explicit operator-retaining diagnostics.
///
/// This additive API is the supported path for `DenseDirectConfig`'s
/// `estimate_condition` and `iterative_refinement_steps` options.  The legacy
/// [`factorize_dense`] function continues to reject those options because its
/// returned factor intentionally does not retain the matrix.  Equilibration
/// remains unavailable for dense factors until a scale-aware factor format is
/// added.
pub fn factorize_dense_configured<T: ComplexField + FromPrimitive>(
    matrix: &DenseMatrix<T>,
    config: &DenseDirectConfig,
) -> Result<DenseConfiguredFactorization<T>, DenseDirectError> {
    if config.equilibrate {
        return Err(DenseDirectError::OptionUnavailable {
            option: "dense equilibration",
        });
    }
    if config.iterative_refinement_steps != 0 && matrix.nrows() != matrix.ncols() {
        return Err(DenseDirectError::OptionUnavailable {
            option: "iterative refinement for rectangular factors",
        });
    }
    if config.estimate_condition && matrix.nrows() != matrix.ncols() {
        return Err(DenseDirectError::OptionUnavailable {
            option: "condition estimation for rectangular factors",
        });
    }

    let mut factor_config = config.clone();
    factor_config.estimate_condition = false;
    factor_config.iterative_refinement_steps = 0;
    let factor = factorize_dense(matrix, &factor_config)?;
    let condition_estimate = if config.estimate_condition {
        Some(estimate_dense_condition(matrix, &factor)?)
    } else {
        None
    };

    Ok(DenseConfiguredFactorization {
        factor,
        matrix: matrix.clone(),
        refinement_steps: config.iterative_refinement_steps,
        condition_estimate,
    })
}

fn estimate_dense_condition<T: ComplexField + FromPrimitive>(
    matrix: &DenseMatrix<T>,
    factor: &DenseFactorization<T>,
) -> Result<T::Real, DenseDirectError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DenseDirectError::OptionUnavailable {
            option: "condition estimation for rectangular factors",
        });
    }
    let n = matrix.nrows();
    if n == 0 {
        return Ok(T::Real::one());
    }
    let matrix_norm = (0..n)
        .map(|column| {
            (0..n)
                .map(|row| matrix[(row, column)].norm())
                .fold(T::Real::zero(), |sum, value| sum + value)
        })
        .fold(T::Real::zero(), T::Real::max);
    let mut inverse_norm = T::Real::zero();
    for column in 0..n {
        let mut rhs = vec![T::zero(); n];
        rhs[column] = T::one();
        let solution = factor.solve(&rhs)?;
        let column_norm = solution
            .iter()
            .map(ComplexField::norm)
            .fold(T::Real::zero(), |sum, value| sum + value);
        inverse_norm = inverse_norm.max(column_norm);
    }
    let estimate = matrix_norm * inverse_norm;
    if estimate.is_finite() {
        Ok(estimate)
    } else {
        Err(DenseDirectError::NonFiniteMetadata {
            field: "condition estimate",
        })
    }
}

fn dense_residual_norm<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    x: &[T],
    b: &[T],
) -> Result<T::Real, DenseDirectError> {
    if x.len() != matrix.ncols() {
        return Err(DenseDirectError::SolutionDimension {
            expected: matrix.ncols(),
            actual: x.len(),
        });
    }
    if b.len() != matrix.nrows() {
        return Err(DenseDirectError::RhsDimension {
            expected: matrix.nrows(),
            actual: b.len(),
        });
    }
    let mut ax = vec![T::zero(); matrix.nrows()];
    matrix.matvec_into(x, &mut ax);
    let residual = ax
        .iter()
        .zip(b)
        .map(|(value, rhs)| (*value - *rhs).norm())
        .fold(T::Real::zero(), T::Real::max);
    if residual.is_finite() {
        Ok(residual)
    } else {
        Err(DenseDirectError::NonFiniteMetadata {
            field: "iterative-refinement residual",
        })
    }
}

/// Common metadata and allocation-free solve interface for direct factors.
pub trait DirectFactorization<T: ComplexField> {
    /// Checked failures expose the shared solver status/accuracy vocabulary.
    type Error: SolverFailure;

    fn dimension(&self) -> usize;
    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error>;
    fn factor_nonzeros(&self) -> usize;

    fn factor_memory_bytes(&self) -> usize {
        self.factor_nonzeros()
            .saturating_mul(std::mem::size_of::<T>())
    }

    fn condition_estimate(&self) -> Option<T::Real> {
        None
    }

    fn solve(&self, b: &[T]) -> Result<Vec<T>, Self::Error> {
        let mut x = vec![T::zero(); self.dimension()];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }

    /// Checked solve returning the canonical direct-result metadata object.
    ///
    /// Implementations validate RHS and solution finiteness in
    /// [`DirectFactorization::solve_into`].  Because the factor does not retain
    /// the original matrix, the returned result deliberately leaves accuracy
    /// as [`AccuracyStatus::NotChecked`] until the caller supplies an
    /// independent residual check.
    fn solve_with_result(&self, b: &[T]) -> Result<DirectSolveResult<T>, Self::Error> {
        let solution = self.solve(b)?;
        Ok(DirectSolveResult::from_solution(solution))
    }
}

impl<T: ComplexField> DirectFactorization<T> for LuFactorization<T> {
    type Error = LuError;

    fn dimension(&self) -> usize {
        self.n
    }

    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error> {
        LuFactorization::solve_into(self, b, x)
    }

    fn factor_nonzeros(&self) -> usize {
        self.n.saturating_mul(self.n)
    }
}

impl<T: ComplexField> DirectFactorization<T> for CholeskyFactorization<T> {
    type Error = CholeskyError;

    fn dimension(&self) -> usize {
        CholeskyFactorization::dimension(self)
    }

    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error> {
        CholeskyFactorization::solve_into(self, b, x)
    }

    fn factor_nonzeros(&self) -> usize {
        CholeskyFactorization::factor_nonzeros(self)
    }
}

impl<T: ComplexField> DirectFactorization<T> for SparseCholeskyFactorization<T> {
    type Error = SparseCholeskyError;

    fn dimension(&self) -> usize {
        SparseCholeskyFactorization::dimension(self)
    }

    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error> {
        SparseCholeskyFactorization::solve_into(self, b, x)
    }

    fn factor_nonzeros(&self) -> usize {
        SparseCholeskyFactorization::factor_nonzeros(self)
    }
}

impl<T: ComplexField + FromPrimitive> DirectFactorization<T> for DenseFactorization<T> {
    type Error = DenseDirectError;

    fn dimension(&self) -> usize {
        match self {
            Self::Lu(factor) => factor.n,
            Self::Cholesky(factor) => factor.dimension(),
            Self::Ldlt(factor) => factor.dimension(),
            Self::Qr(factor) => factor.columns(),
        }
    }

    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error> {
        let expected_solution = self.dimension();
        if b.len() != self.rhs_dimension() {
            return Err(DenseDirectError::RhsDimension {
                expected: self.rhs_dimension(),
                actual: b.len(),
            });
        }
        if x.len() != expected_solution {
            return Err(DenseDirectError::SolutionDimension {
                expected: expected_solution,
                actual: x.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(DenseDirectError::NonFiniteRhs { index });
        }
        // Stage every factor-class result before publishing it.  The checked
        // facade promises caller-owned destination preservation on numerical
        // failure, including a breakdown during the final substitution step.
        let solution = match self {
            Self::Lu(factor) => {
                let mut solution = vec![T::zero(); expected_solution];
                factor
                    .solve_into(b, &mut solution)
                    .map_err(DenseDirectError::Lu)?;
                solution
            }
            Self::Cholesky(factor) => {
                let mut solution = vec![T::zero(); expected_solution];
                factor
                    .solve_into(b, &mut solution)
                    .map_err(DenseDirectError::Cholesky)?;
                solution
            }
            Self::Ldlt(factor) => factor.try_solve(b)?,
            Self::Qr(factor) => {
                let mut solution = vec![T::zero(); expected_solution];
                factor
                    .solve_least_squares_into(b, &mut solution)
                    .map_err(DenseDirectError::Qr)?;
                solution
            }
        };
        if let Some(index) = first_non_finite(&solution) {
            return Err(DenseDirectError::NonFiniteSolution { index });
        }
        x.copy_from_slice(&solution);
        Ok(())
    }

    fn factor_nonzeros(&self) -> usize {
        match self {
            Self::Lu(factor) => factor.n.saturating_mul(factor.n),
            Self::Cholesky(factor) => factor.factor_nonzeros(),
            Self::Ldlt(factor) => factor.factor_memory_bytes() / std::mem::size_of::<T>(),
            Self::Qr(factor) => factor.factor_nonzeros(),
        }
    }
}

impl<T: ComplexField + FromPrimitive> DirectFactorization<T> for SparseFactorization<T> {
    type Error = SparseDirectError;

    fn dimension(&self) -> usize {
        SparseFactorization::dimension(self)
    }

    fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), Self::Error> {
        SparseFactorization::solve_into(self, b, x)
    }

    fn factor_nonzeros(&self) -> usize {
        SparseFactorization::factor_nonzeros(self)
    }

    fn factor_memory_bytes(&self) -> usize {
        SparseFactorization::factor_memory_bytes(self)
    }
}

/// Dispatch a dense factorization from an explicit matrix-class declaration.
pub fn factorize_dense<T: ComplexField + FromPrimitive>(
    matrix: &DenseMatrix<T>,
    config: &DenseDirectConfig,
) -> Result<DenseFactorization<T>, DenseDirectError> {
    if config.equilibrate {
        return Err(DenseDirectError::OptionUnavailable {
            option: "equilibration",
        });
    }
    if config.estimate_condition {
        return Err(DenseDirectError::OptionUnavailable {
            option: "condition estimation",
        });
    }
    if config.iterative_refinement_steps > 0 {
        return Err(DenseDirectError::OptionUnavailable {
            option: "iterative refinement",
        });
    }
    if let Some(index) = first_non_finite(matrix.as_slice()) {
        return Err(DenseDirectError::NonFiniteMatrix { index });
    }
    let method = match config.method {
        DenseDirectMethod::Auto => match config.matrix_class {
            MatrixClass::General => DenseDirectMethod::Lu,
            MatrixClass::SymmetricPositiveDefinite | MatrixClass::HermitianPositiveDefinite => {
                DenseDirectMethod::Cholesky
            }
            MatrixClass::SymmetricIndefinite | MatrixClass::HermitianIndefinite => {
                DenseDirectMethod::Ldlt
            }
            MatrixClass::RectangularFullRank => {
                if matrix.nrows() >= 4 * matrix.ncols() {
                    DenseDirectMethod::Tsqr
                } else {
                    DenseDirectMethod::Qr
                }
            }
            MatrixClass::RankSensitive => DenseDirectMethod::PivotedQr,
        },
        explicit => explicit,
    };

    match method {
        DenseDirectMethod::Lu if config.matrix_class == MatrixClass::General => {
            let mut workspace = LuWorkspace::new();
            Ok(DenseFactorization::Lu(lu_factorize_with_workspace(
                matrix,
                &mut workspace,
            )?))
        }
        DenseDirectMethod::Cholesky
            if config.matrix_class == MatrixClass::SymmetricPositiveDefinite =>
        {
            Ok(DenseFactorization::Cholesky(cholesky_factorize(matrix)?))
        }
        DenseDirectMethod::Cholesky
            if config.matrix_class == MatrixClass::HermitianPositiveDefinite =>
        {
            Ok(DenseFactorization::Cholesky(cholesky_factorize_hermitian(
                matrix,
            )?))
        }
        DenseDirectMethod::Ldlt
            if matches!(
                config.matrix_class,
                MatrixClass::SymmetricIndefinite | MatrixClass::HermitianIndefinite
            ) =>
        {
            let sparse = CsrMatrix::from_dense(matrix, T::Real::zero());
            let factor = MultifrontalLdltFactorization::factorize_with_config(
                &sparse,
                &LdltConfig {
                    hermitian: config.matrix_class == MatrixClass::HermitianIndefinite,
                    ..LdltConfig::default()
                },
            )?;
            Ok(DenseFactorization::Ldlt(std::sync::Arc::new(factor)))
        }
        DenseDirectMethod::Qr if config.matrix_class == MatrixClass::RectangularFullRank => {
            Ok(DenseFactorization::Qr(QrFactorization::factorize(matrix)?))
        }
        DenseDirectMethod::Tsqr if config.matrix_class == MatrixClass::RectangularFullRank => Ok(
            DenseFactorization::Qr(QrFactorization::factorize_tsqr(matrix)?),
        ),
        DenseDirectMethod::PivotedQr if config.matrix_class == MatrixClass::RankSensitive => Ok(
            DenseFactorization::Qr(QrFactorization::factorize_pivoted(matrix)?),
        ),
        _ => Err(DenseDirectError::InvalidMethod {
            matrix_class: config.matrix_class,
            method,
        }),
    }
}

/// Reusable symbolic analysis for currently supported sparse methods.
pub fn analyze_sparse<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: &SparseDirectConfig,
) -> Result<SparseSymbolic, SparseDirectError> {
    let method = resolved_sparse_method(config);
    validate_sparse_method(config, method)?;
    if let Some(index) = first_non_finite(&matrix.values) {
        return Err(SparseDirectError::NonFiniteMatrix { index });
    }
    match method {
        SparseDirectMethod::SimplicialLu | SparseDirectMethod::SupernodalLu => Ok(
            SparseSymbolic::Lu(SparseLuFactorization::analyze_supernodal_with_column_perm(
                matrix,
                lu_column_perm(matrix, config.ordering),
            )?),
        ),
        SparseDirectMethod::SimplicialCholesky | SparseDirectMethod::SupernodalCholesky => {
            Ok(SparseSymbolic::Cholesky(SparseCholeskySymbolic::analyze(
                matrix,
                cholesky_ordering(config.ordering),
            )?))
        }
        _ => Err(SparseDirectError::MethodUnavailable { method }),
    }
}

/// Dispatch a sparse factorization from the declared matrix class and method.
///
/// The legacy return type is matrix-free. Non-default
/// `memory_budget_bytes`, `right_hand_sides`, or
/// `backward_error_tolerance` values are rejected; use
/// [`factorize_sparse_configured`] when those contracts are needed.
pub fn factorize_sparse<T>(
    matrix: &CsrMatrix<T>,
    config: &SparseDirectConfig,
    symbolic: Option<&SparseSymbolic>,
) -> Result<SparseFactorization<T>, SparseDirectError>
where
    T: ComplexField + FromPrimitive,
{
    let method = resolved_sparse_method(config);
    if sparse_configured_options_requested(config) {
        return Err(SparseDirectError::OptionUnavailable {
            option: "configured sparse options (use factorize_sparse_configured)",
            method,
        });
    }
    factorize_sparse_impl(matrix, config, symbolic)
}

/// Factor a sparse matrix while retaining the source operator for configured
/// RHS-count, memory-budget, and iterative-refinement diagnostics.
pub fn factorize_sparse_configured<T>(
    matrix: &CsrMatrix<T>,
    config: &SparseDirectConfig,
    symbolic: Option<&SparseSymbolic>,
) -> Result<SparseConfiguredFactorization<T>, SparseDirectError>
where
    T: ComplexField + FromPrimitive,
{
    let method = resolved_sparse_method(config);
    if config.right_hand_sides == 0 {
        return Err(SparseDirectError::InvalidConfiguration {
            field: "right_hand_sides",
            reason: "must be greater than zero",
        });
    }
    if !config.backward_error_tolerance.is_finite() || config.backward_error_tolerance < 0.0 {
        return Err(SparseDirectError::InvalidConfiguration {
            field: "backward_error_tolerance",
            reason: "must be finite and non-negative",
        });
    }
    let factor = factorize_sparse_impl(matrix, config, symbolic)?;
    if let Some(budget_bytes) = config.memory_budget_bytes {
        let required_bytes = factor.factor_memory_bytes();
        if required_bytes > budget_bytes {
            return Err(SparseDirectError::MemoryBudgetExceeded {
                method,
                budget_bytes,
                required_bytes,
            });
        }
    }
    let backward_error_tolerance = T::Real::from_f64(config.backward_error_tolerance).ok_or(
        SparseDirectError::InvalidConfiguration {
            field: "backward_error_tolerance",
            reason: "cannot be represented by the scalar type",
        },
    )?;
    Ok(SparseConfiguredFactorization {
        factor,
        matrix: matrix.clone(),
        right_hand_sides: config.right_hand_sides,
        backward_error_tolerance,
    })
}

fn factorize_sparse_impl<T>(
    matrix: &CsrMatrix<T>,
    config: &SparseDirectConfig,
    symbolic: Option<&SparseSymbolic>,
) -> Result<SparseFactorization<T>, SparseDirectError>
where
    T: ComplexField + FromPrimitive,
{
    let method = resolved_sparse_method(config);
    validate_sparse_method(config, method)?;
    if let Some(index) = first_non_finite(&matrix.values) {
        return Err(SparseDirectError::NonFiniteMatrix { index });
    }
    if config.equilibrate
        && !matches!(
            method,
            SparseDirectMethod::SimplicialLu | SparseDirectMethod::SupernodalLu
        )
    {
        return Err(SparseDirectError::OptionUnavailable {
            option: "equilibration",
            method,
        });
    }
    if config.equilibrate && symbolic.is_some() {
        return Err(SparseDirectError::OptionUnavailable {
            option: "equilibration with a reused symbolic analysis (the matching is value-dependent)",
            method,
        });
    }
    if config.equilibrate
        && matches!(config.ordering, Ordering::Amd | Ordering::NestedDissection)
        && matches!(
            method,
            SparseDirectMethod::SimplicialLu | SparseDirectMethod::SupernodalLu
        )
    {
        // The equilibrating path analyses the matched-and-scaled matrix
        // internally with COLAMD; an AMD/ND request would be silently ignored.
        return Err(SparseDirectError::OptionUnavailable {
            option: "AMD/ND ordering with equilibration",
            method,
        });
    }
    let threshold =
        T::Real::from_f64(config.pivot_threshold.clamp(0.0, 1.0)).unwrap_or_else(T::Real::one);
    match method {
        SparseDirectMethod::SimplicialLu => Ok(SparseFactorization::Lu(
            SparseLuFactorization::factorize_with_preprocessing_and_column_perm(
                matrix,
                threshold,
                config.equilibrate,
                lu_column_perm(matrix, config.ordering),
            )?,
        )),
        SparseDirectMethod::SupernodalLu => {
            let factor = if config.equilibrate {
                SparseLuFactorization::factorize_supernodal_with_preprocessing(
                    matrix, threshold, true,
                )?
            } else {
                match symbolic {
                    Some(SparseSymbolic::Lu(symbolic)) => {
                        SparseLuFactorization::factorize_supernodal_with_symbolic(
                            matrix, symbolic, threshold,
                        )?
                    }
                    Some(_) => {
                        return Err(SparseDirectError::SymbolicMismatch { method });
                    }
                    // A reused symbolic carries the ordering it was analyzed
                    // with; `config.ordering` applies to fresh analyses.
                    None => {
                        if matches!(config.ordering, Ordering::Amd | Ordering::NestedDissection) {
                            let symbolic =
                                SparseLuFactorization::analyze_supernodal_with_column_perm(
                                    matrix,
                                    lu_column_perm(matrix, config.ordering),
                                )?;
                            SparseLuFactorization::factorize_supernodal_with_symbolic(
                                matrix, &symbolic, threshold,
                            )?
                        } else {
                            SparseLuFactorization::factorize_supernodal_with_pivot_threshold(
                                matrix, threshold,
                            )?
                        }
                    }
                }
            };
            Ok(SparseFactorization::Lu(factor))
        }
        SparseDirectMethod::SimplicialCholesky => {
            let hermitian = config.matrix_class == MatrixClass::HermitianPositiveDefinite;
            let factor = match symbolic {
                Some(SparseSymbolic::Cholesky(symbolic)) => {
                    SparseCholeskyFactorization::factorize_with_symbolic(
                        matrix, symbolic, hermitian,
                    )?
                }
                Some(_) => {
                    return Err(SparseDirectError::SymbolicMismatch { method });
                }
                None => SparseCholeskyFactorization::factorize_with_ordering(
                    matrix,
                    cholesky_ordering(config.ordering),
                    hermitian,
                )?,
            };
            Ok(SparseFactorization::Cholesky(factor))
        }
        SparseDirectMethod::SupernodalCholesky => {
            let hermitian = config.matrix_class == MatrixClass::HermitianPositiveDefinite;
            let factor = match symbolic {
                Some(SparseSymbolic::Cholesky(symbolic)) => {
                    SparseCholeskyFactorization::factorize_supernodal_with_symbolic(
                        matrix, symbolic, hermitian,
                    )?
                }
                Some(_) => {
                    return Err(SparseDirectError::SymbolicMismatch { method });
                }
                None => SparseCholeskyFactorization::factorize_supernodal_with_ordering(
                    matrix,
                    cholesky_ordering(config.ordering),
                    hermitian,
                )?,
            };
            Ok(SparseFactorization::Cholesky(factor))
        }
        SparseDirectMethod::ExactMultifrontalLu
        | SparseDirectMethod::MultifrontalBlr
        | SparseDirectMethod::MultifrontalHss => {
            if symbolic.is_some() {
                return Err(SparseDirectError::MethodUnavailable { method });
            }
            let mut multifrontal = BlrConfig::default();
            match method {
                SparseDirectMethod::ExactMultifrontalLu => multifrontal.epsilon = 0.0,
                SparseDirectMethod::MultifrontalHss => multifrontal.panel_kind = PanelKind::Hss,
                SparseDirectMethod::MultifrontalBlr => {}
                _ => unreachable!(),
            }
            Ok(SparseFactorization::Multifrontal(
                MultifrontalBlrFactorization::factorize_with_config(matrix, &multifrontal)?,
            ))
        }
        SparseDirectMethod::MultifrontalLdlt => {
            if symbolic.is_some() {
                return Err(SparseDirectError::MethodUnavailable { method });
            }
            let ldlt = LdltConfig {
                hermitian: config.matrix_class == MatrixClass::HermitianIndefinite,
                ..LdltConfig::default()
            };
            Ok(SparseFactorization::MultifrontalLdlt(
                MultifrontalLdltFactorization::factorize_with_config(matrix, &ldlt)?,
            ))
        }
        _ => Err(SparseDirectError::MethodUnavailable { method }),
    }
}

fn sparse_configured_options_requested(config: &SparseDirectConfig) -> bool {
    config.memory_budget_bytes.is_some()
        || config.right_hand_sides != SparseDirectConfig::default().right_hand_sides
        || config.backward_error_tolerance.to_bits()
            != SparseDirectConfig::default()
                .backward_error_tolerance
                .to_bits()
}

fn resolved_sparse_method(config: &SparseDirectConfig) -> SparseDirectMethod {
    match config.method {
        SparseDirectMethod::Auto => match config.matrix_class {
            MatrixClass::General => SparseDirectMethod::SupernodalLu,
            MatrixClass::SymmetricPositiveDefinite | MatrixClass::HermitianPositiveDefinite => {
                SparseDirectMethod::SimplicialCholesky
            }
            MatrixClass::SymmetricIndefinite | MatrixClass::HermitianIndefinite => {
                SparseDirectMethod::MultifrontalLdlt
            }
            MatrixClass::RectangularFullRank | MatrixClass::RankSensitive => {
                SparseDirectMethod::Auto
            }
        },
        explicit => explicit,
    }
}

/// Map the public ordering request onto an LU column permutation.
/// `Auto` resolves to COLAMD (the historical default); `Amd` selects the
/// approximate-minimum-degree ordering and `NestedDissection` the
/// METIS-style multilevel dissection, both of which win decisively on
/// symmetric-structure problems (SuiteSparse UMFPACK territory) — ND
/// additionally cuts supernodal numeric time on PDE grids.
fn lu_column_perm<T: ComplexField>(matrix: &CsrMatrix<T>, ordering: Ordering) -> Vec<usize> {
    match ordering {
        Ordering::Amd => amd_ordering(matrix),
        Ordering::NestedDissection => nd_ordering(matrix),
        _ => colamd_ordering(matrix),
    }
}

/// Map the public ordering request onto the Cholesky symbolic analysis.
/// `Auto` resolves to nested dissection (best for PDE matrices).
fn cholesky_ordering(ordering: Ordering) -> CholeskyOrdering {
    match ordering {
        Ordering::Natural => CholeskyOrdering::Natural,
        Ordering::Colamd => CholeskyOrdering::Colamd,
        _ => CholeskyOrdering::NestedDissection,
    }
}

fn validate_sparse_method(
    config: &SparseDirectConfig,
    method: SparseDirectMethod,
) -> Result<(), SparseDirectError> {
    let general_method = matches!(
        method,
        SparseDirectMethod::SimplicialLu
            | SparseDirectMethod::SupernodalLu
            | SparseDirectMethod::ExactMultifrontalLu
            | SparseDirectMethod::MultifrontalBlr
            | SparseDirectMethod::MultifrontalHss
    );
    let cholesky_method = matches!(
        method,
        SparseDirectMethod::SimplicialCholesky | SparseDirectMethod::SupernodalCholesky
    );
    let ldlt_method = matches!(method, SparseDirectMethod::MultifrontalLdlt);
    let class_supported = if cholesky_method {
        matches!(
            config.matrix_class,
            MatrixClass::SymmetricPositiveDefinite | MatrixClass::HermitianPositiveDefinite
        )
    } else if ldlt_method {
        matches!(
            config.matrix_class,
            MatrixClass::SymmetricIndefinite | MatrixClass::HermitianIndefinite
        )
    } else {
        config.matrix_class == MatrixClass::General && general_method
    };
    if !class_supported {
        return Err(SparseDirectError::InvalidMethod {
            matrix_class: config.matrix_class,
            method,
        });
    }
    let ordering_supported = match method {
        SparseDirectMethod::SimplicialLu | SparseDirectMethod::SupernodalLu => {
            matches!(
                config.ordering,
                Ordering::Auto | Ordering::Colamd | Ordering::Amd | Ordering::NestedDissection
            )
        }
        SparseDirectMethod::SimplicialCholesky | SparseDirectMethod::SupernodalCholesky => {
            matches!(
                config.ordering,
                Ordering::Auto | Ordering::Natural | Ordering::NestedDissection | Ordering::Colamd
            )
        }
        _ => matches!(config.ordering, Ordering::Auto | Ordering::NestedDissection),
    };
    if !ordering_supported {
        return Err(SparseDirectError::OrderingUnavailable {
            ordering: config.ordering,
            method,
        });
    }
    Ok(())
}

/// Scaled infinity-norm backward error for a computed dense solution.
pub fn dense_backward_error<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    x: &[T],
    b: &[T],
) -> Result<T::Real, DenseDirectError> {
    if matrix.ncols() != x.len() {
        return Err(DenseDirectError::SolutionDimension {
            expected: matrix.ncols(),
            actual: x.len(),
        });
    }
    if matrix.nrows() != b.len() {
        return Err(DenseDirectError::RhsDimension {
            expected: matrix.nrows(),
            actual: b.len(),
        });
    }
    if let Some(index) = first_non_finite(matrix.as_slice()) {
        return Err(DenseDirectError::NonFiniteMatrix { index });
    }
    if let Some(index) = first_non_finite(x) {
        return Err(DenseDirectError::NonFiniteSolution { index });
    }
    if let Some(index) = first_non_finite(b) {
        return Err(DenseDirectError::NonFiniteRhs { index });
    }
    let actual = matrix.matvec(x);
    let residual = actual
        .iter()
        .zip(b)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(T::Real::zero(), T::Real::max);
    if !residual.is_finite() {
        return Err(DenseDirectError::NonFiniteMetadata { field: "residual" });
    }
    let matrix_norm = (0..matrix.nrows())
        .map(|row| {
            matrix
                .row(row)
                .iter()
                .map(ComplexField::norm)
                .fold(T::Real::zero(), |sum, value| sum + value)
        })
        .fold(T::Real::zero(), T::Real::max);
    if !matrix_norm.is_finite() {
        return Err(DenseDirectError::NonFiniteMetadata {
            field: "matrix norm",
        });
    }
    let x_norm = x
        .iter()
        .map(ComplexField::norm)
        .fold(T::Real::zero(), T::Real::max);
    if !x_norm.is_finite() {
        return Err(DenseDirectError::NonFiniteMetadata {
            field: "solution norm",
        });
    }
    let b_norm = b
        .iter()
        .map(ComplexField::norm)
        .fold(T::Real::zero(), T::Real::max);
    if !b_norm.is_finite() {
        return Err(DenseDirectError::NonFiniteMetadata {
            field: "right-hand-side norm",
        });
    }
    let denominator = matrix_norm * x_norm + b_norm;
    if !denominator.is_finite() {
        return Err(DenseDirectError::NonFiniteMetadata {
            field: "backward-error denominator",
        });
    }
    if denominator == T::Real::zero() {
        Ok(residual)
    } else {
        let error = residual / denominator;
        if error.is_finite() {
            Ok(error)
        } else {
            Err(DenseDirectError::NonFiniteMetadata {
                field: "backward error",
            })
        }
    }
}

fn sparse_diagnostics<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    x: &[T],
    b: &[T],
) -> Result<(T::Real, T::Real), SparseDirectError> {
    if matrix.num_cols != x.len() {
        return Err(SparseDirectError::SolutionDimension {
            expected: matrix.num_cols,
            actual: x.len(),
        });
    }
    if matrix.num_rows != b.len() {
        return Err(SparseDirectError::RhsDimension {
            expected: matrix.num_rows,
            actual: b.len(),
        });
    }
    if let Some(index) = first_non_finite(&matrix.values) {
        return Err(SparseDirectError::NonFiniteMatrix { index });
    }
    if let Some(index) = first_non_finite(x) {
        return Err(SparseDirectError::NonFiniteSolution { index });
    }
    if let Some(index) = first_non_finite(b) {
        return Err(SparseDirectError::NonFiniteRhs { index });
    }

    let actual = matrix.matvec(x);
    let residual = actual
        .iter()
        .zip(b)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(T::Real::zero(), T::Real::max);
    let matrix_norm = (0..matrix.num_rows)
        .map(|row| matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
        .map(|range| {
            range
                .map(|index| matrix.values[index].norm())
                .fold(T::Real::zero(), |sum, value| sum + value)
        })
        .fold(T::Real::zero(), T::Real::max);
    let x_norm = x
        .iter()
        .map(ComplexField::norm)
        .fold(T::Real::zero(), T::Real::max);
    let b_norm = b
        .iter()
        .map(ComplexField::norm)
        .fold(T::Real::zero(), T::Real::max);
    if !residual.is_finite() {
        return Err(SparseDirectError::NonFiniteMetadata { field: "residual" });
    }
    if !matrix_norm.is_finite() {
        return Err(SparseDirectError::NonFiniteMetadata {
            field: "matrix norm",
        });
    }
    if !x_norm.is_finite() {
        return Err(SparseDirectError::NonFiniteMetadata {
            field: "solution norm",
        });
    }
    if !b_norm.is_finite() {
        return Err(SparseDirectError::NonFiniteMetadata {
            field: "right-hand-side norm",
        });
    }
    let denominator = matrix_norm * x_norm + b_norm;
    if !denominator.is_finite() {
        return Err(SparseDirectError::NonFiniteMetadata {
            field: "backward-error denominator",
        });
    }
    let backward_error = if denominator == T::Real::zero() {
        if residual == T::Real::zero() {
            T::Real::zero()
        } else {
            T::Real::infinity()
        }
    } else {
        residual / denominator
    };
    if !backward_error.is_finite() {
        return Err(SparseDirectError::NonFiniteMetadata {
            field: "backward error",
        });
    }
    Ok((backward_error, residual))
}

fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    values.iter().position(|value| !value.norm().is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_uses_declared_property_and_reports_backward_error() {
        let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
        let config = DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            ..DenseDirectConfig::default()
        };
        let factor = factorize_dense(&matrix, &config).unwrap();
        assert!(matches!(factor, DenseFactorization::Cholesky(_)));
        let rhs = vec![1.0, 2.0];
        let solution = factor.solve(&rhs).unwrap();
        assert!(dense_backward_error(&matrix, &solution, &rhs).unwrap() < 1e-14);
    }

    #[test]
    fn invalid_property_method_pair_does_not_fall_back() {
        let matrix = DenseMatrix::<f64>::eye(2);
        let config = DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Cholesky,
            ..DenseDirectConfig::default()
        };
        assert!(matches!(
            factorize_dense(&matrix, &config),
            Err(DenseDirectError::InvalidMethod { .. })
        ));
    }

    #[test]
    fn rectangular_auto_dispatches_to_qr_least_squares() {
        let matrix =
            DenseMatrix::from_row_major_vec(4, 2, vec![1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let factor = factorize_dense(
            &matrix,
            &DenseDirectConfig {
                matrix_class: MatrixClass::RectangularFullRank,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(factor, DenseFactorization::Qr(_)));
        let solution = factor.solve(&[6.0, 5.0, 7.0, 10.0]).unwrap();
        assert!((solution[0] - 3.5).abs() < 1.0e-12);
        assert!((solution[1] - 1.4).abs() < 1.0e-12);
    }

    #[test]
    fn dense_indefinite_auto_dispatches_to_ldlt() {
        let matrix = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![0.0_f64, 2.0, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0, 3.0],
        );
        let config = DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            ..DenseDirectConfig::default()
        };
        let factor = factorize_dense(&matrix, &config).unwrap();
        assert!(matches!(factor, DenseFactorization::Ldlt(_)));
        let rhs = [1.0, -2.0, 4.0];
        let solution = factor.solve(&rhs).unwrap();
        assert!(dense_backward_error(&matrix, &solution, &rhs).unwrap() < 1.0e-13);
    }

    #[test]
    fn dense_ldlt_reports_singular_and_dimension_errors() {
        let singular = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![1.0_f64, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
        );
        let config = DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: DenseDirectMethod::Ldlt,
            ..DenseDirectConfig::default()
        };
        assert!(matches!(
            factorize_dense(&singular, &config),
            Err(DenseDirectError::Ldlt(_))
        ));

        let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![0.0_f64, 1.0, 1.0, 0.0]);
        let factor = factorize_dense(&matrix, &config).unwrap();
        assert!(matches!(
            factor.solve(&[1.0]),
            Err(DenseDirectError::RhsDimension {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn dense_ldlt_rejects_non_finite_rhs_without_panicking() {
        let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![0.0_f64, 1.0, 1.0, 0.0]);
        let config = DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: DenseDirectMethod::Ldlt,
            ..DenseDirectConfig::default()
        };
        let factor = factorize_dense(&matrix, &config).unwrap();

        assert!(matches!(
            factor.solve(&[f64::NAN, 1.0]),
            Err(DenseDirectError::NonFiniteRhs { index: 0 })
        ));
    }

    #[test]
    fn tall_rectangular_auto_dispatches_to_tsqr() {
        let matrix = DenseMatrix::from_fn(1_024, 3, |row, column| {
            let x = row as f64 / 1024.0;
            match column {
                0 => 1.0,
                1 => x,
                _ => x * x,
            }
        });
        let factor = factorize_dense(
            &matrix,
            &DenseDirectConfig {
                matrix_class: MatrixClass::RectangularFullRank,
                ..DenseDirectConfig::default()
            },
        )
        .unwrap();
        let exact = [2.0, -1.0, 0.25];
        let rhs = matrix.matvec(&exact);
        let solution = factor.solve(&rhs).unwrap();
        for (actual, expected) in solution.iter().zip(exact) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
    }

    #[test]
    fn sparse_auto_reuses_symbolic_for_general_matrix() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0_f64, -1.0, 0.0, -0.5, 3.0, -1.0, 0.0, -1.0, 2.0],
        );
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let config = SparseDirectConfig::default();
        let symbolic = analyze_sparse(&matrix, &config).unwrap();
        let factor = factorize_sparse(&matrix, &config, Some(&symbolic)).unwrap();
        let rhs = vec![1.0, 2.0, 3.0];
        let solution = factor.solve(&rhs).unwrap();
        let actual = matrix.matvec(&solution);
        for (actual, expected) in actual.iter().zip(rhs) {
            assert!((*actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_lu_amd_ordering_dispatches_and_rejects_equilibrate_combo() {
        // Symmetric-pattern system through the configured AMD path, with
        // and without a reused symbolic analysis.
        let dense = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                4.0_f64, -1.0, 0.0, -0.5, -1.0, 4.0, -1.0, 0.0, 0.0, -1.0, 4.0, -1.0, -0.5, 0.0,
                -1.0, 4.0,
            ],
        );
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SupernodalLu,
            ordering: Ordering::Amd,
            ..SparseDirectConfig::default()
        };
        let rhs = vec![1.0, 2.0, 3.0, 4.0];
        for symbolic in [None, Some(analyze_sparse(&matrix, &config).unwrap())] {
            let factor = factorize_sparse(&matrix, &config, symbolic.as_ref()).unwrap();
            assert!(matches!(factor, SparseFactorization::Lu(_)));
            let solution = factor.solve(&rhs).unwrap();
            let actual = matrix.matvec(&solution);
            for (actual, expected) in actual.iter().zip(&rhs) {
                assert!((*actual - expected).abs() < 1e-10);
            }
        }
        // AMD with equilibration is rejected rather than silently ignored
        // (the equilibrating path analyses the matched matrix with COLAMD).
        let bad = SparseDirectConfig {
            equilibrate: true,
            ..config.clone()
        };
        assert!(matches!(
            factorize_sparse(&matrix, &bad, None),
            Err(SparseDirectError::OptionUnavailable { .. })
        ));
    }

    #[test]
    fn sparse_lu_nd_ordering_dispatches_and_rejects_equilibrate_combo() {
        // Symmetric-pattern system through the configured ND path, with
        // and without a reused symbolic analysis.
        let dense = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                4.0_f64, -1.0, 0.0, -0.5, -1.0, 4.0, -1.0, 0.0, 0.0, -1.0, 4.0, -1.0, -0.5, 0.0,
                -1.0, 4.0,
            ],
        );
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let rhs = vec![1.0, 2.0, 3.0, 4.0];
        for method in [
            SparseDirectMethod::SimplicialLu,
            SparseDirectMethod::SupernodalLu,
        ] {
            let config = SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method,
                ordering: Ordering::NestedDissection,
                ..SparseDirectConfig::default()
            };
            for symbolic in [None, Some(analyze_sparse(&matrix, &config).unwrap())] {
                let factor = factorize_sparse(&matrix, &config, symbolic.as_ref()).unwrap();
                assert!(matches!(factor, SparseFactorization::Lu(_)));
                let solution = factor.solve(&rhs).unwrap();
                let actual = matrix.matvec(&solution);
                for (actual, expected) in actual.iter().zip(&rhs) {
                    assert!((*actual - expected).abs() < 1e-10);
                }
            }
        }
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SupernodalLu,
            ordering: Ordering::NestedDissection,
            ..SparseDirectConfig::default()
        };
        // ND with equilibration is rejected rather than silently ignored
        // (the equilibrating path analyses the matched matrix with COLAMD).
        let bad = SparseDirectConfig {
            equilibrate: true,
            ..config.clone()
        };
        assert!(matches!(
            factorize_sparse(&matrix, &bad, None),
            Err(SparseDirectError::OptionUnavailable { .. })
        ));
    }

    #[test]
    fn sparse_spd_auto_dispatches_to_cholesky() {
        let matrix = CsrMatrix::from_dense(&DenseMatrix::<f64>::eye(2), 0.0);
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            ..SparseDirectConfig::default()
        };
        let factor = factorize_sparse(&matrix, &config, None).unwrap();
        assert!(matches!(factor, SparseFactorization::Cholesky(_)));
        let solution = factor.solve(&[1.0, 2.0]).unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-14);
        assert!((solution[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn sparse_spd_symbolic_reuse_through_api() {
        // 1D Laplacian, SPD.
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![2.0_f64, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
        );
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ..SparseDirectConfig::default()
        };
        let symbolic = analyze_sparse(&matrix, &config).unwrap();
        assert!(matches!(symbolic, SparseSymbolic::Cholesky(_)));
        let factor = factorize_sparse(&matrix, &config, Some(&symbolic)).unwrap();
        let rhs = vec![1.0, 0.0, 1.0];
        let solution = factor.solve(&rhs).unwrap();
        let actual = matrix.matvec(&solution);
        for (actual, expected) in actual.iter().zip(rhs) {
            assert!((*actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_supernodal_cholesky_dispatches_and_reuses_symbolic() {
        // 2D Laplacian 8x8, SPD.
        let nx = 8;
        let n = nx * nx;
        let mut triplets = Vec::new();
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                triplets.push((i, i, 4.0_f64));
                if ix > 0 {
                    triplets.push((i, i - 1, -1.0));
                }
                if ix + 1 < nx {
                    triplets.push((i, i + 1, -1.0));
                }
                if iy > 0 {
                    triplets.push((i, i - nx, -1.0));
                }
                if iy + 1 < nx {
                    triplets.push((i, i + nx, -1.0));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, triplets);
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SupernodalCholesky,
            ..SparseDirectConfig::default()
        };
        let rhs: Vec<f64> = (0..n).map(|i| (i % 5) as f64 + 1.0).collect();

        // With and without a reused symbolic analysis, the supernodal method
        // must produce a Cholesky factor that solves the system.
        for symbolic in [None, Some(analyze_sparse(&matrix, &config).unwrap())] {
            let factor = factorize_sparse(&matrix, &config, symbolic.as_ref()).unwrap();
            assert!(matches!(factor, SparseFactorization::Cholesky(_)));
            let solution = factor.solve(&rhs).unwrap();
            let actual = matrix.matvec(&solution);
            for (actual, expected) in actual.iter().zip(&rhs) {
                assert!((*actual - expected).abs() < 1e-10);
            }
        }

        // A simplicial symbolic is the same type and is equally valid here.
        let simplicial_config = SparseDirectConfig {
            method: SparseDirectMethod::SimplicialCholesky,
            ..config.clone()
        };
        let symbolic = analyze_sparse(&matrix, &simplicial_config).unwrap();
        assert!(factorize_sparse(&matrix, &config, Some(&symbolic)).is_ok());
    }

    #[test]
    fn sparse_cholesky_rejects_general_class() {
        let matrix = CsrMatrix::from_dense(&DenseMatrix::<f64>::eye(2), 0.0);
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialCholesky,
            ..SparseDirectConfig::default()
        };
        assert!(matches!(
            factorize_sparse(&matrix, &config, None),
            Err(SparseDirectError::InvalidMethod { .. })
        ));
    }

    #[test]
    fn sparse_indefinite_auto_dispatches_to_multifrontal_ldlt() {
        // 1D shifted Laplacian K − σM, indefinite for σ between eigenvalues
        // (σ = 1.3 lies strictly between the 2nd and 3rd eigenvalue of the
        // n = 8 chain; σ = 1.0 would be an exact eigenvalue).
        let n = 8;
        let sigma = 1.3_f64;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            dense[[i, i]] = 2.0 - sigma;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i + 1 < n {
                dense[[i, i + 1]] = -1.0;
            }
        }
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        for class in [
            MatrixClass::SymmetricIndefinite,
            MatrixClass::HermitianIndefinite,
        ] {
            let config = SparseDirectConfig {
                matrix_class: class,
                ..SparseDirectConfig::default()
            };
            let factor = factorize_sparse(&matrix, &config, None).unwrap();
            assert!(
                matches!(factor, SparseFactorization::MultifrontalLdlt(_)),
                "{class:?} must dispatch to MultifrontalLdlt"
            );
            let rhs: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();
            let solution = factor.solve(&rhs).unwrap();
            let actual = matrix.matvec(&solution);
            for (actual, expected) in actual.iter().zip(&rhs) {
                assert!((*actual - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn sparse_multifrontal_ldlt_rejects_non_finite_rhs_without_panicking() {
        let matrix = CsrMatrix::from_dense(
            &DenseMatrix::from_row_major_vec(2, 2, vec![0.0_f64, 1.0, 1.0, 0.0]),
            0.0,
        );
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: SparseDirectMethod::MultifrontalLdlt,
            ..SparseDirectConfig::default()
        };
        let factor = factorize_sparse(&matrix, &config, None).unwrap();

        assert!(matches!(
            factor.solve(&[f64::NAN, 1.0]),
            Err(SparseDirectError::NonFiniteRhs { index: 0 })
        ));
    }

    #[test]
    fn sparse_ldlt_rejects_general_class() {
        let matrix = CsrMatrix::from_dense(&DenseMatrix::<f64>::eye(2), 0.0);
        let config = SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::MultifrontalLdlt,
            ..SparseDirectConfig::default()
        };
        assert!(matches!(
            factorize_sparse(&matrix, &config, None),
            Err(SparseDirectError::InvalidMethod { .. })
        ));
    }

    /// Tridiagonal core with row/column scalings spanning 10^±6: the
    /// pattern is well conditioned but the numerics are badly scaled.
    fn badly_scaled_matrix(n: usize) -> CsrMatrix<f64> {
        let dense = DenseMatrix::from_fn(n, n, |i, j| {
            let base = if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            };
            base * 10f64.powi(i as i32 - 6) * 10f64.powi(5 - j as i32)
        });
        CsrMatrix::from_dense(&dense, 0.0)
    }

    #[test]
    fn sparse_equilibration_available_through_standard_dispatch() {
        let n = 12;
        let matrix = badly_scaled_matrix(n);
        let exact: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        let rhs = matrix.matvec(&exact);

        // Equilibration must be reachable through the plain config/dispatch
        // surface (Auto and both explicit LU methods) with no special entry
        // points, and must produce an accurate solution on this badly
        // scaled matrix.
        for method in [
            SparseDirectMethod::Auto,
            SparseDirectMethod::SimplicialLu,
            SparseDirectMethod::SupernodalLu,
        ] {
            let config = SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method,
                equilibrate: true,
                ..SparseDirectConfig::default()
            };
            let factor = factorize_sparse(&matrix, &config, None)
                .unwrap_or_else(|e| panic!("{method:?} with equilibration failed: {e}"));
            let solution = factor.solve(&rhs).unwrap();
            let max_error = solution
                .iter()
                .zip(&exact)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_error < 1e-6,
                "{method:?} equilibrated solution error {max_error:e}"
            );
        }
    }

    #[test]
    fn sparse_equilibration_rejects_non_lu_methods_and_symbolic_reuse() {
        let n = 6;
        let matrix = badly_scaled_matrix(n);

        // SPD dispatch: equilibration is only implemented for sparse LU.
        let spd_config = SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            equilibrate: true,
            ..SparseDirectConfig::default()
        };
        assert!(matches!(
            factorize_sparse(&matrix, &spd_config, None),
            Err(SparseDirectError::OptionUnavailable { .. })
        ));

        // The matching is value-dependent, so a pattern-only reusable
        // symbolic cannot be combined with equilibration.
        let lu_config = SparseDirectConfig {
            matrix_class: MatrixClass::General,
            equilibrate: true,
            ..SparseDirectConfig::default()
        };
        let symbolic = analyze_sparse(&matrix, &lu_config).unwrap();
        assert!(matches!(
            factorize_sparse(&matrix, &lu_config, Some(&symbolic)),
            Err(SparseDirectError::OptionUnavailable { .. })
        ));
    }
}
