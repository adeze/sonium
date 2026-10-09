//! Sparse LU factorization with partial pivoting
//!
//! Implements a left-looking sparse LU factorization for mark_general square matrices.
//! Uses COLAMD fill-reducing ordering and elimination-tree-guided column reach
//! to achieve O(nnz(L)) complexity instead of O(n^2).
//!
//! # Algorithm
//!
//! 1. **COLAMD ordering**: Compute fill-reducing column permutation Q
//! 2. **Column elimination tree**: Compute etree of A^T*A for sparse reach
//! 3. **Numeric factorization**: Left-looking column-by-column LU with
//!    etree-guided sparse triangular solve and partial pivoting
//! 4. **Triangular solve**: Exact level-scheduled forward (L) and backward
//!    (U) substitution for large factors, with a serial path for small ones.
//!
//! Works with all `ComplexField` types (f32, f64, Complex32, Complex64).

use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, SolverError, SolverFailure,
    SolverResultContract, SolverStatus,
};
use sonium_solvers_sparse::CsrMatrix;

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rayon::prelude::*;
use thiserror::Error;

use super::sparse_supernodal;
pub use super::sparse_supernodal::SparseLuSymbolic;

pub mod amd;
pub mod colamd;
pub(crate) mod equilibration;

/// Errors from sparse LU factorization.
#[derive(Error, Debug)]
pub enum SparseLuError {
    #[error("Matrix is singular (zero pivot at column {0})")]
    SingularMatrix(usize),
    #[error("Matrix must be square ({rows} x {cols})")]
    NotSquare { rows: usize, cols: usize },
    #[error("Matrix dimensions {rows} x {cols} do not match factorization size {expected}")]
    MatrixDimensionMismatch {
        expected: usize,
        rows: usize,
        cols: usize,
    },
    #[error("Right-hand side length {actual} does not match factorization size {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("caller-supplied column permutation is not a permutation (bad entry {column})")]
    InvalidColumnPermutation { column: usize },
    #[error("multi-RHS shape overflows usize (dimension={dimension}, nrhs={nrhs})")]
    SizeOverflow { dimension: usize, nrhs: usize },
    #[error("U diagonal at column {0} is non-finite")]
    NonFiniteDiagonal(usize),
    #[error("matrix entry at row {row}, column {col} is non-finite")]
    NonFiniteMatrix { row: usize, col: usize },
    #[error("right-hand side at index {0} is non-finite")]
    NonFiniteRhs(usize),
    #[error("pivot threshold must be finite")]
    InvalidPivotThreshold,
    #[error("non-finite value generated during {stage} at index {index}")]
    NumericalBreakdown { stage: &'static str, index: usize },
    #[error("invalid sparse LU checkpoint: {0}")]
    InvalidCheckpoint(String),
    #[error("CSR sparsity pattern does not match reusable symbolic analysis")]
    PatternMismatch,
    #[error("invalid CSR structure: {0}")]
    InvalidCsrStructure(String),
    #[error("invalid iterative-refinement tolerance: {0}")]
    InvalidRefinementTolerance(String),
    #[error("iterative solver failed: {0}")]
    IterativeSolve(String),
}

impl SparseLuError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::SingularMatrix(_) | Self::NumericalBreakdown { .. } => SolverStatus::Breakdown,
            _ => SolverStatus::NumericalError,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverFailure for SparseLuError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotSquare { .. }
            | Self::MatrixDimensionMismatch { .. }
            | Self::DimensionMismatch { .. }
            | Self::InvalidColumnPermutation { .. }
            | Self::SizeOverflow { .. } => SolverErrorKind::DimensionMismatch,
            Self::SingularMatrix(_) | Self::NumericalBreakdown { .. } => SolverErrorKind::Breakdown,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        SparseLuError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        SparseLuError::accuracy_status(self)
    }
}

/// Result and convergence metadata from sparse iterative refinement.
#[derive(Debug)]
pub struct SparseRefinementResult<T: ComplexField> {
    pub x: Vec<T>,
    pub iterations: usize,
    /// Relative residual of the initial solve before any correction step.
    pub initial_relative_residual: T::Real,
    pub relative_residual: T::Real,
    /// The relative-residual tolerance supplied to [`Self::solve_refined`].
    pub tolerance: T::Real,
    pub converged: bool,
    pub stop_reason: SparseRefinementStopReason,
    /// Provenance of the reusable factor used for every correction solve.
    /// Refinement accepts value updates only when this exact structural
    /// fingerprint is preserved.
    pub provenance: SparseRefinementProvenance,
}

impl<T: ComplexField> SparseRefinementResult<T> {
    /// Return the canonical status for the independently checked refinement.
    pub fn solver_status(&self) -> SolverStatus {
        let finite = self.initial_relative_residual.is_finite()
            && self.relative_residual.is_finite()
            && self.tolerance.is_finite()
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

impl<T: ComplexField> SolverResultContract for SparseRefinementResult<T> {
    fn solver_status(&self) -> SolverStatus {
        SparseRefinementResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        SparseRefinementResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.initial_relative_residual < T::Real::zero()
            || self.relative_residual < T::Real::zero()
            || self.tolerance < T::Real::zero()
            || !self.initial_relative_residual.is_finite()
            || !self.relative_residual.is_finite()
            || !self.tolerance.is_finite()
            || self.x.iter().any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "sparse refinement result contract",
                reason: "solution or residual metadata is invalid".into(),
            });
        }
        if self.converged != (self.solver_status() == SolverStatus::Converged)
            || self.converged != (self.stop_reason == SparseRefinementStopReason::ToleranceReached)
        {
            return Err(SolverError::OperationFailed {
                operation: "sparse refinement result contract",
                reason: "convergence metadata and stop reason disagree".into(),
            });
        }
        Ok(())
    }
}

/// Structural provenance attached to an iterative-refinement result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseRefinementProvenance {
    /// The numeric values may have changed, but the canonical CSR pattern was
    /// identical to the pattern used to build the reusable factor.
    ReusedValueOnlyPattern { structure_fingerprint: u64 },
}

/// Terminal condition reported by sparse iterative refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseRefinementStopReason {
    /// The reported relative residual satisfied the requested tolerance.
    ToleranceReached,
    /// The correction budget was exhausted before the tolerance was met.
    MaxIterations,
}

/// Numeric backend used to build a sparse LU factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseLuBackend {
    /// Etree-guided scalar left-looking columns.
    Simplicial,
    /// SuperLU-style dense panels with packed BLAS-3 updates.
    Supernodal,
}

/// Version of the portable sparse-LU factor checkpoint format.
///
/// Numeric values are stored as little-endian `f64` real/imaginary pairs.
/// This is lossless for the supported `f32`/`f64` scalar families and avoids
/// depending on Rust's in-memory representation.
pub const SPARSE_LU_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const SPARSE_LU_CHECKPOINT_MAGIC: &[u8; 8] = b"SNLUCP01";

fn validate_csr_lu_structure<T: ComplexField>(matrix: &CsrMatrix<T>) -> Result<(), SparseLuError> {
    let expected_row_ptr_len = matrix.num_rows.checked_add(1).ok_or_else(|| {
        SparseLuError::InvalidCsrStructure("row count overflows the row-pointer length".into())
    })?;
    if matrix.row_ptrs.len() != expected_row_ptr_len {
        return Err(SparseLuError::InvalidCsrStructure(format!(
            "row-pointer length {} does not equal {}",
            matrix.row_ptrs.len(),
            expected_row_ptr_len
        )));
    }
    if matrix.col_indices.len() != matrix.values.len() {
        return Err(SparseLuError::InvalidCsrStructure(format!(
            "column-index length {} does not equal value length {}",
            matrix.col_indices.len(),
            matrix.values.len()
        )));
    }
    let nnz = matrix.values.len();
    let terminal = *matrix
        .row_ptrs
        .last()
        .ok_or_else(|| SparseLuError::InvalidCsrStructure("row-pointer array is empty".into()))?;
    if terminal != nnz {
        return Err(SparseLuError::InvalidCsrStructure(format!(
            "terminal row pointer {terminal} does not equal nnz {nnz}"
        )));
    }

    for row in 0..matrix.num_rows {
        let start = matrix.row_ptrs[row];
        let end = matrix.row_ptrs[row + 1];
        if start > end || end > nnz {
            return Err(SparseLuError::InvalidCsrStructure(format!(
                "row {row} has invalid range [{start}, {end}) for nnz {nnz}"
            )));
        }
        let mut previous_column = None;
        for index in start..end {
            let column = matrix.col_indices[index];
            if column >= matrix.num_cols {
                return Err(SparseLuError::InvalidCsrStructure(format!(
                    "entry {index} in row {row} has column {column}, outside {} columns",
                    matrix.num_cols
                )));
            }
            if let Some(previous) = previous_column
                && column <= previous
            {
                return Err(SparseLuError::InvalidCsrStructure(format!(
                    "row {row} columns are not strictly increasing at entry {index} ({previous} then {column})"
                )));
            }
            previous_column = Some(column);
        }
    }
    Ok(())
}

fn validate_finite_matrix_values<T: ComplexField>(
    matrix: &CsrMatrix<T>,
) -> Result<(), SparseLuError> {
    validate_csr_lu_structure(matrix)?;
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            if !matrix.values[index].norm().is_finite() {
                return Err(SparseLuError::NonFiniteMatrix {
                    row,
                    col: matrix.col_indices[index],
                });
            }
        }
    }
    Ok(())
}

fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    values.iter().position(|value| !value.norm().is_finite())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CsrPattern {
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
}

impl CsrPattern {
    fn from_matrix<T: ComplexField>(matrix: &CsrMatrix<T>) -> Self {
        Self {
            row_ptrs: matrix.row_ptrs.clone(),
            col_indices: matrix.col_indices.clone(),
        }
    }

    fn empty() -> Self {
        Self {
            row_ptrs: vec![0],
            col_indices: Vec::new(),
        }
    }

    fn matches<T: ComplexField>(&self, matrix: &CsrMatrix<T>) -> bool {
        self.row_ptrs == matrix.row_ptrs && self.col_indices == matrix.col_indices
    }

    fn fingerprint(&self) -> u64 {
        // FNV-1a is deterministic across processes and covers only the
        // structural arrays; numeric values may change during refinement.
        let mut hash = 0xcbf29ce484222325_u64;
        for value in self.row_ptrs.iter().chain(self.col_indices.iter()) {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3_u64);
            }
        }
        hash
    }
}

fn checkpoint_invalid(message: impl Into<String>) -> SparseLuError {
    SparseLuError::InvalidCheckpoint(message.into())
}

fn validate_checkpoint_ptrs(
    name: &str,
    ptrs: &[usize],
    entry_count: usize,
    n: usize,
) -> Result<(), SparseLuError> {
    let expected = n
        .checked_add(1)
        .ok_or_else(|| checkpoint_invalid(format!("{name} pointer length overflows")))?;
    if ptrs.len() != expected || ptrs.first().copied() != Some(0) {
        return Err(checkpoint_invalid(format!(
            "{name} pointers must have length {expected} and start at zero"
        )));
    }
    if ptrs.windows(2).any(|window| window[0] > window[1]) {
        return Err(checkpoint_invalid(format!(
            "{name} pointers are not monotone"
        )));
    }
    if ptrs.last().copied() != Some(entry_count) {
        return Err(checkpoint_invalid(format!(
            "{name} pointer terminus does not match entry count"
        )));
    }
    Ok(())
}

fn validate_checkpoint_rows(name: &str, rows: &[usize], n: usize) -> Result<(), SparseLuError> {
    if rows.iter().any(|&row| row >= n) {
        return Err(checkpoint_invalid(format!(
            "{name} contains an out-of-range row"
        )));
    }
    Ok(())
}

fn validate_checkpoint_permutation(
    name: &str,
    permutation: &[usize],
    n: usize,
) -> Result<(), SparseLuError> {
    if permutation.len() != n {
        return Err(checkpoint_invalid(format!(
            "{name} has length {}, expected {n}",
            permutation.len()
        )));
    }
    let mut seen = vec![false; n];
    for &value in permutation {
        if value >= n || seen[value] {
            return Err(checkpoint_invalid(format!("{name} is not a permutation")));
        }
        seen[value] = true;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_checkpoint_state<T: ComplexField>(
    n: usize,
    pattern: &CsrPattern,
    l_col_ptrs: &[usize],
    l_row_indices: &[usize],
    l_values: &[T],
    u_col_ptrs: &[usize],
    u_row_indices: &[usize],
    u_values: &[T],
    row_perm: &[usize],
    col_perm: &[usize],
    preprocess: Option<&equilibration::Equilibration<T::Real>>,
) -> Result<(), SparseLuError> {
    validate_checkpoint_ptrs("CSR row", &pattern.row_ptrs, pattern.col_indices.len(), n)?;
    validate_checkpoint_rows("CSR column indices", &pattern.col_indices, n)?;
    if l_row_indices.len() != l_values.len() {
        return Err(checkpoint_invalid("L row/value lengths differ"));
    }
    if u_row_indices.len() != u_values.len() {
        return Err(checkpoint_invalid("U row/value lengths differ"));
    }
    validate_checkpoint_ptrs("L", l_col_ptrs, l_row_indices.len(), n)?;
    validate_checkpoint_ptrs("U", u_col_ptrs, u_row_indices.len(), n)?;
    validate_checkpoint_rows("L row indices", l_row_indices, n)?;
    validate_checkpoint_rows("U row indices", u_row_indices, n)?;
    validate_checkpoint_permutation("row permutation", row_perm, n)?;
    validate_checkpoint_permutation("column permutation", col_perm, n)?;
    if l_values.iter().any(|value| !value.norm().is_finite()) {
        return Err(checkpoint_invalid("L contains a non-finite value"));
    }
    if u_values.iter().any(|value| !value.norm().is_finite()) {
        return Err(checkpoint_invalid("U contains a non-finite value"));
    }
    if let Some(preprocess) = preprocess {
        if preprocess.row_scale.len() != n
            || preprocess.col_perm.len() != n
            || preprocess.col_scale.len() != n
        {
            return Err(checkpoint_invalid("equilibration vector length mismatch"));
        }
        validate_checkpoint_permutation(
            "equilibration column permutation",
            &preprocess.col_perm,
            n,
        )?;
        if preprocess
            .row_scale
            .iter()
            .chain(preprocess.col_scale.iter())
            .any(|scale| !scale.is_finite() || *scale <= T::Real::zero())
        {
            return Err(checkpoint_invalid(
                "equilibration scales must be finite and positive",
            ));
        }
    }
    Ok(())
}

fn checkpoint_push_usize(output: &mut Vec<u8>, value: usize) -> Result<(), SparseLuError> {
    let value = u64::try_from(value)
        .map_err(|_| checkpoint_invalid("usize value does not fit checkpoint format"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn checkpoint_push_usize_vec(output: &mut Vec<u8>, values: &[usize]) -> Result<(), SparseLuError> {
    checkpoint_push_usize(output, values.len())?;
    for &value in values {
        checkpoint_push_usize(output, value)?;
    }
    Ok(())
}

fn checkpoint_push_values<T: ComplexField>(
    output: &mut Vec<u8>,
    values: &[T],
) -> Result<(), SparseLuError> {
    checkpoint_push_usize(output, values.len())?;
    for value in values {
        let re = value
            .re()
            .to_f64()
            .ok_or_else(|| checkpoint_invalid("real factor value cannot convert to f64"))?;
        let im = value
            .im()
            .to_f64()
            .ok_or_else(|| checkpoint_invalid("imaginary factor value cannot convert to f64"))?;
        if !re.is_finite() || !im.is_finite() {
            return Err(checkpoint_invalid("factor contains a non-finite value"));
        }
        output.extend_from_slice(&re.to_le_bytes());
        output.extend_from_slice(&im.to_le_bytes());
    }
    Ok(())
}

fn checkpoint_push_reals<R: ToPrimitive>(
    output: &mut Vec<u8>,
    values: &[R],
) -> Result<(), SparseLuError> {
    checkpoint_push_usize(output, values.len())?;
    for value in values {
        let value = value
            .to_f64()
            .ok_or_else(|| checkpoint_invalid("equilibration value cannot convert to f64"))?;
        if !value.is_finite() {
            return Err(checkpoint_invalid(
                "equilibration contains a non-finite value",
            ));
        }
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

struct SparseLuCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SparseLuCheckpointReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], SparseLuError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| checkpoint_invalid("checkpoint offset overflows"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| checkpoint_invalid("checkpoint is truncated"))?;
        let mut result = [0_u8; N];
        result.copy_from_slice(bytes);
        self.offset = end;
        Ok(result)
    }

    fn read_u8(&mut self) -> Result<u8, SparseLuError> {
        Ok(self.take::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SparseLuError> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }

    fn read_u64(&mut self) -> Result<u64, SparseLuError> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    fn read_f64(&mut self) -> Result<f64, SparseLuError> {
        let value = f64::from_le_bytes(self.take::<8>()?);
        if !value.is_finite() {
            return Err(checkpoint_invalid(
                "checkpoint contains a non-finite scalar",
            ));
        }
        Ok(value)
    }

    fn read_len(&mut self, element_bytes: usize) -> Result<usize, SparseLuError> {
        let value = usize::try_from(self.read_u64()?)
            .map_err(|_| checkpoint_invalid("checkpoint length does not fit usize"))?;
        if value > self.bytes.len().saturating_sub(self.offset) / element_bytes {
            return Err(checkpoint_invalid(
                "checkpoint vector length exceeds payload",
            ));
        }
        Ok(value)
    }

    fn read_usize_vec(&mut self) -> Result<Vec<usize>, SparseLuError> {
        let len = self.read_len(8)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(
                usize::try_from(self.read_u64()?)
                    .map_err(|_| checkpoint_invalid("checkpoint index does not fit usize"))?,
            );
        }
        Ok(values)
    }

    fn read_values<T: ComplexField>(&mut self) -> Result<Vec<T>, SparseLuError> {
        let len = self.read_len(16)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let re = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| checkpoint_invalid("real factor value cannot decode"))?;
            let im = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| checkpoint_invalid("imaginary factor value cannot decode"))?;
            values.push(T::from_re_im(re, im));
        }
        Ok(values)
    }

    fn read_reals<R: FromPrimitive>(&mut self) -> Result<Vec<R>, SparseLuError> {
        let len = self.read_len(8)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(
                R::from_f64(self.read_f64()?)
                    .ok_or_else(|| checkpoint_invalid("equilibration value cannot decode"))?,
            );
        }
        Ok(values)
    }

    fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Sparse LU factorization result.
///
/// Stores L (unit lower triangular) and U (upper triangular) in CSC-like
/// compressed column storage, plus row and column permutations. Precomputes
/// level metadata for future parallel triangular solves.
#[derive(Debug)]
pub struct SparseLuFactorization<T: ComplexField> {
    n: usize,
    backend: SparseLuBackend,
    supernode_count: usize,
    /// Column pointers for L (length n+1)
    l_col_ptrs: Vec<usize>,
    /// Row indices for L
    l_row_indices: Vec<usize>,
    /// Values for L (unit diagonal not stored)
    l_values: Vec<T>,
    /// Column pointers for U (length n+1)
    u_col_ptrs: Vec<usize>,
    /// Row indices for U
    u_row_indices: Vec<usize>,
    /// Values for U (diagonal stored)
    u_values: Vec<T>,
    /// Row permutation (partial pivoting): perm[k] = original row of k-th pivot
    row_perm: Vec<usize>,
    /// Column permutation (COLAMD ordering): col_perm[k] = original column of k-th step
    col_perm: Vec<usize>,
    /// Level sets for parallel forward substitution (L solve).
    /// Each level contains column indices that can be processed in parallel.
    l_level_sets: Vec<Vec<usize>>,
    /// Level sets for parallel backward substitution (U solve).
    u_level_sets: Vec<Vec<usize>>,
    /// Inverse row permutation (`row_perm_inv[original] = factor row`),
    /// derived once at construction: every solve previously rebuilt it
    /// (alloc + O(n) scatter).
    row_perm_inv: Vec<usize>,
    /// Index into `u_row_indices`/`u_values` of each column's diagonal
    /// entry (`usize::MAX` when structurally absent — the simplicial
    /// backend skips storing exact-zero pivots under threshold-0
    /// pivoting). Derived once at construction: every solve previously
    /// linear-searched each U column for its diagonal (O(nnz) per solve,
    /// twice over in the substitution itself).
    u_diag_idx: Vec<usize>,
    /// Optional MC64-style preprocessing (max-weight matching + scaling)
    /// applied before factorization. When present, the factors describe
    /// `Dr · A · Q_m · Dc` and `solve` applies the (un)scaling transparently.
    /// Boxed to keep the default (no preprocessing) struct small.
    preprocess: Option<Box<equilibration::Equilibration<T::Real>>>,
    /// Exact CSR structure used to build these factors. Refinement rejects
    /// matrices whose sparsity pattern, ordering, or index base differs.
    pattern: CsrPattern,
    /// Built by [`factorize_for_preconditioner`](Self::factorize_for_preconditioner):
    /// the singularity guard was relaxed and tiny pivots repaired, and
    /// `solve` normalizes the right-hand side to unit inf-norm (exact
    /// power-of-two scaling) so triangular-solve intermediates stay inside
    /// narrow exponent ranges (f16).
    preconditioner_mode: bool,
}

impl<T: ComplexField> SparseLuFactorization<T> {
    /// Matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Numeric backend that produced these factors.
    pub fn backend(&self) -> SparseLuBackend {
        self.backend
    }

    /// Deterministic fingerprint of the CSR row-pointer and column-index
    /// arrays used to build this factorization.
    pub fn structure_fingerprint(&self) -> u64 {
        self.pattern.fingerprint()
    }

    /// Whether `matrix` is a supported value-only refinement update.
    /// Dimensions and the complete CSR index structure must be unchanged;
    /// numeric values may differ.
    pub fn accepts_value_update(&self, matrix: &CsrMatrix<T>) -> bool {
        self.try_accepts_value_update(matrix).is_ok()
    }

    /// Validate the value-only update contract used by [`Self::solve_refined`].
    ///
    /// A reusable factor is tied to the exact canonical CSR pattern used to
    /// build it. This checked form also revalidates mutable public CSR storage
    /// and matrix values before the first residual matvec, so duplicate,
    /// unsorted, malformed, or non-finite updates cannot reach refinement.
    pub fn try_accepts_value_update(&self, matrix: &CsrMatrix<T>) -> Result<(), SparseLuError> {
        if matrix.num_rows != self.n || matrix.num_cols != self.n {
            return Err(SparseLuError::MatrixDimensionMismatch {
                expected: self.n,
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        validate_finite_matrix_values(matrix)?;
        if !self.pattern.matches(matrix) {
            return Err(SparseLuError::PatternMismatch);
        }
        Ok(())
    }

    /// Number of relaxed symbolic supernodes (`0` for the simplicial backend).
    pub fn supernode_count(&self) -> usize {
        self.supernode_count
    }

    /// Number of explicitly stored factor entries, including L's unit diagonal.
    pub fn factor_nonzeros(&self) -> usize {
        self.l_values
            .len()
            .saturating_add(self.u_values.len())
            .saturating_add(self.n)
    }

    /// Factorize a sparse matrix: P*A*Q = L*U
    ///
    /// Uses COLAMD fill-reducing ordering, elimination-tree-guided sparse
    /// column reach (O(nnz) instead of O(n^2)), and partial pivoting.
    /// Serialize this factor into a portable, versioned checkpoint.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SparseLuError> {
        validate_checkpoint_state(
            self.n,
            &self.pattern,
            &self.l_col_ptrs,
            &self.l_row_indices,
            &self.l_values,
            &self.u_col_ptrs,
            &self.u_row_indices,
            &self.u_values,
            &self.row_perm,
            &self.col_perm,
            self.preprocess.as_deref(),
        )?;
        let backend = match self.backend {
            SparseLuBackend::Simplicial => 0_u8,
            SparseLuBackend::Supernodal => 1_u8,
        };
        let mut flags = u8::from(self.preconditioner_mode);
        if self.preprocess.is_some() {
            flags |= 1 << 1;
        }
        let mut output = Vec::new();
        output.extend_from_slice(SPARSE_LU_CHECKPOINT_MAGIC);
        output.extend_from_slice(&SPARSE_LU_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(backend);
        output.push(flags);
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        checkpoint_push_usize(&mut output, self.n)?;
        checkpoint_push_usize(&mut output, self.supernode_count)?;
        checkpoint_push_usize_vec(&mut output, &self.pattern.row_ptrs)?;
        checkpoint_push_usize_vec(&mut output, &self.pattern.col_indices)?;
        checkpoint_push_usize_vec(&mut output, &self.l_col_ptrs)?;
        checkpoint_push_usize_vec(&mut output, &self.l_row_indices)?;
        checkpoint_push_values(&mut output, &self.l_values)?;
        checkpoint_push_usize_vec(&mut output, &self.u_col_ptrs)?;
        checkpoint_push_usize_vec(&mut output, &self.u_row_indices)?;
        checkpoint_push_values(&mut output, &self.u_values)?;
        checkpoint_push_usize_vec(&mut output, &self.row_perm)?;
        checkpoint_push_usize_vec(&mut output, &self.col_perm)?;
        if let Some(preprocess) = self.preprocess.as_deref() {
            checkpoint_push_reals(&mut output, &preprocess.row_scale)?;
            checkpoint_push_usize_vec(&mut output, &preprocess.col_perm)?;
            checkpoint_push_reals(&mut output, &preprocess.col_scale)?;
        }
        Ok(output)
    }

    /// Restore a factor produced by [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SparseLuError> {
        let mut reader = SparseLuCheckpointReader::new(bytes);
        if &reader.take::<8>()? != SPARSE_LU_CHECKPOINT_MAGIC {
            return Err(checkpoint_invalid("bad magic"));
        }
        let version = reader.read_u16()?;
        if version != SPARSE_LU_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_invalid(format!(
                "unsupported format version {version}"
            )));
        }
        let backend = match reader.read_u8()? {
            0 => SparseLuBackend::Simplicial,
            1 => SparseLuBackend::Supernodal,
            _ => return Err(checkpoint_invalid("unknown backend tag")),
        };
        let flags = reader.read_u8()?;
        if flags & !0b11 != 0 {
            return Err(checkpoint_invalid("unknown checkpoint flags"));
        }
        let scalar_size = reader.read_u8()?;
        let real_size = reader.read_u8()?;
        if scalar_size as usize != std::mem::size_of::<T>()
            || real_size as usize != std::mem::size_of::<T::Real>()
        {
            return Err(checkpoint_invalid(
                "checkpoint scalar family does not match restore type",
            ));
        }
        let preconditioner_mode = flags & 1 != 0;
        let has_preprocess = flags & (1 << 1) != 0;
        let n = usize::try_from(reader.read_u64()?)
            .map_err(|_| checkpoint_invalid("dimension does not fit usize"))?;
        let supernode_count = usize::try_from(reader.read_u64()?)
            .map_err(|_| checkpoint_invalid("supernode count does not fit usize"))?;
        let pattern = CsrPattern {
            row_ptrs: reader.read_usize_vec()?,
            col_indices: reader.read_usize_vec()?,
        };
        let l_col_ptrs = reader.read_usize_vec()?;
        let l_row_indices = reader.read_usize_vec()?;
        let l_values = reader.read_values::<T>()?;
        let u_col_ptrs = reader.read_usize_vec()?;
        let u_row_indices = reader.read_usize_vec()?;
        let u_values = reader.read_values::<T>()?;
        let row_perm = reader.read_usize_vec()?;
        let col_perm = reader.read_usize_vec()?;
        let preprocess = if has_preprocess {
            Some(Box::new(equilibration::Equilibration {
                row_scale: reader.read_reals::<T::Real>()?,
                col_perm: reader.read_usize_vec()?,
                col_scale: reader.read_reals::<T::Real>()?,
            }))
        } else {
            None
        };
        if !reader.is_at_end() {
            return Err(checkpoint_invalid("trailing bytes"));
        }
        validate_checkpoint_state(
            n,
            &pattern,
            &l_col_ptrs,
            &l_row_indices,
            &l_values,
            &u_col_ptrs,
            &u_row_indices,
            &u_values,
            &row_perm,
            &col_perm,
            preprocess.as_deref(),
        )?;
        let l_level_sets = compute_l_level_sets(&l_col_ptrs, &l_row_indices, &row_perm, n);
        let u_level_sets = compute_u_level_sets(&u_col_ptrs, &u_row_indices, n);
        let (row_perm_inv, u_diag_idx) =
            derive_solve_caches(&u_col_ptrs, &u_row_indices, &row_perm, n);
        Ok(Self {
            n,
            backend,
            supernode_count,
            l_col_ptrs,
            l_row_indices,
            l_values,
            u_col_ptrs,
            u_row_indices,
            u_values,
            row_perm,
            col_perm,
            l_level_sets,
            u_level_sets,
            row_perm_inv,
            u_diag_idx,
            preprocess,
            pattern,
            preconditioner_mode,
        })
    }

    pub fn factorize(matrix: &CsrMatrix<T>) -> Result<Self, SparseLuError> {
        Self::factorize_with_pivot_threshold(matrix, T::Real::one())
    }

    /// Factorize with the SuperLU-style panel backend and strict partial pivoting.
    pub fn factorize_supernodal(matrix: &CsrMatrix<T>) -> Result<Self, SparseLuError> {
        Self::factorize_supernodal_with_pivot_threshold(matrix, T::Real::one())
    }

    /// Analyze ordering, etree, relaxed supernodes, and panel schedule once.
    pub fn analyze_supernodal(matrix: &CsrMatrix<T>) -> Result<SparseLuSymbolic, SparseLuError> {
        validate_csr_lu_structure(matrix)?;
        sparse_supernodal::analyze(matrix)
    }

    /// Analyze with a caller-supplied column permutation (`col_perm[k]` is
    /// the original column eliminated at step `k`; see [`amd_ordering`] and
    /// [`colamd_ordering`]). The numeric phase is ordering-agnostic, so a
    /// permutation from any fill-reducing ordering works here.
    pub fn analyze_supernodal_with_column_perm(
        matrix: &CsrMatrix<T>,
        col_perm: Vec<usize>,
    ) -> Result<SparseLuSymbolic, SparseLuError> {
        validate_csr_lu_structure(matrix)?;
        sparse_supernodal::analyze_with_column_perm(matrix, col_perm)
    }

    /// Factorize using an explicitly selected numeric backend.
    pub fn factorize_with_backend(
        matrix: &CsrMatrix<T>,
        backend: SparseLuBackend,
        pivot_threshold: T::Real,
    ) -> Result<Self, SparseLuError> {
        match backend {
            SparseLuBackend::Simplicial => {
                Self::factorize_with_pivot_threshold(matrix, pivot_threshold)
            }
            SparseLuBackend::Supernodal => {
                Self::factorize_supernodal_with_pivot_threshold(matrix, pivot_threshold)
            }
        }
    }

    /// Factorize with SuperLU-style panels and threshold partial pivoting.
    pub fn factorize_supernodal_with_pivot_threshold(
        matrix: &CsrMatrix<T>,
        pivot_threshold: T::Real,
    ) -> Result<Self, SparseLuError> {
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(SparseLuError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        validate_finite_matrix_values(matrix)?;
        if !pivot_threshold.is_finite() {
            return Err(SparseLuError::InvalidPivotThreshold);
        }
        if n == 0 {
            return Ok(Self::empty(SparseLuBackend::Supernodal));
        }

        let numeric = sparse_supernodal::factorize(matrix, pivot_threshold)?;
        let mut result = Self::from_supernodal_numeric(numeric)?;
        result.pattern = CsrPattern::from_matrix(matrix);
        Ok(result)
    }

    /// Factorize with SuperLU-style panels, threshold partial pivoting, and
    /// optional MC64-style preprocessing (max-weight matching + row/column
    /// equilibration, see [`equilibration`]).
    ///
    /// When `equilibrate` is set and the sparsity pattern admits a perfect
    /// matching, the factors are built for `Dr · A · Q_m · Dc` and `solve`
    /// applies the scaling transparently. When no perfect matching exists
    /// the matrix is factorized as-is.
    ///
    /// Note: the explicit symbolic-reuse entry points
    /// (`analyze_supernodal` / `factorize_supernodal_with_symbolic`) do not
    /// equilibrate, because the matching is value-dependent while the
    /// symbolic analysis is pattern-only.
    pub fn factorize_supernodal_with_preprocessing(
        matrix: &CsrMatrix<T>,
        pivot_threshold: T::Real,
        equilibrate: bool,
    ) -> Result<Self, SparseLuError> {
        if !equilibrate {
            return Self::factorize_supernodal_with_pivot_threshold(matrix, pivot_threshold);
        }
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(SparseLuError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        validate_finite_matrix_values(matrix)?;
        if !pivot_threshold.is_finite() {
            return Err(SparseLuError::InvalidPivotThreshold);
        }
        if n == 0 {
            return Ok(Self::empty(SparseLuBackend::Supernodal));
        }

        let mut preprocess = None;
        let equilibrated;
        let matrix = match equilibration::equilibrate(matrix) {
            Some((scaled, pre)) => {
                preprocess = Some(Box::new(pre));
                equilibrated = scaled;
                &equilibrated
            }
            None => matrix,
        };
        let numeric = sparse_supernodal::factorize(matrix, pivot_threshold)?;
        let mut result = Self::from_supernodal_numeric(numeric)?;
        result.preprocess = preprocess;
        result.pattern = CsrPattern::from_matrix(matrix);
        Ok(result)
    }

    /// Reuse symbolic analysis for another matrix with the same CSR pattern.
    pub fn factorize_supernodal_with_symbolic(
        matrix: &CsrMatrix<T>,
        symbolic: &SparseLuSymbolic,
        pivot_threshold: T::Real,
    ) -> Result<Self, SparseLuError> {
        validate_finite_matrix_values(matrix)?;
        let numeric =
            sparse_supernodal::factorize_with_symbolic(matrix, symbolic, pivot_threshold)?;
        let mut result = Self::from_supernodal_numeric(numeric)?;
        result.pattern = CsrPattern::from_matrix(matrix);
        Ok(result)
    }

    fn from_supernodal_numeric(
        numeric: sparse_supernodal::NumericFactors<T>,
    ) -> Result<Self, SparseLuError> {
        if let Some(index) = first_non_finite(&numeric.l_values) {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "supernodal L factorization",
                index,
            });
        }
        if let Some(index) = first_non_finite(&numeric.u_values) {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "supernodal U factorization",
                index,
            });
        }
        let n = numeric.row_perm.len();
        let l_level_sets = compute_l_level_sets(
            &numeric.l_col_ptrs,
            &numeric.l_row_indices,
            &numeric.row_perm,
            n,
        );
        let u_level_sets = compute_u_level_sets(&numeric.u_col_ptrs, &numeric.u_row_indices, n);
        let (row_perm_inv, u_diag_idx) = derive_solve_caches(
            &numeric.u_col_ptrs,
            &numeric.u_row_indices,
            &numeric.row_perm,
            n,
        );
        Ok(Self {
            n,
            backend: SparseLuBackend::Supernodal,
            supernode_count: numeric.supernode_count,
            l_col_ptrs: numeric.l_col_ptrs,
            l_row_indices: numeric.l_row_indices,
            l_values: numeric.l_values,
            u_col_ptrs: numeric.u_col_ptrs,
            u_row_indices: numeric.u_row_indices,
            u_values: numeric.u_values,
            row_perm: numeric.row_perm,
            col_perm: numeric.col_perm,
            l_level_sets,
            u_level_sets,
            row_perm_inv,
            u_diag_idx,
            preprocess: None,
            pattern: CsrPattern::empty(),
            preconditioner_mode: false,
        })
    }

    fn empty(backend: SparseLuBackend) -> Self {
        Self {
            n: 0,
            backend,
            supernode_count: 0,
            l_col_ptrs: vec![0],
            l_row_indices: Vec::new(),
            l_values: Vec::new(),
            u_col_ptrs: vec![0],
            u_row_indices: Vec::new(),
            u_values: Vec::new(),
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            l_level_sets: Vec::new(),
            u_level_sets: Vec::new(),
            row_perm_inv: Vec::new(),
            u_diag_idx: Vec::new(),
            preprocess: None,
            pattern: CsrPattern::empty(),
            preconditioner_mode: false,
        }
    }

    /// Factorize with threshold partial pivoting.
    ///
    /// A threshold of `1` selects the largest available entry (conventional
    /// partial pivoting). Smaller values retain the diagonal candidate when
    /// `|a_kk| >= threshold * max_i |a_ik|`, reducing row interchanges and
    /// preserving more of the symbolic structure. Values are clamped to
    /// `[0, 1]`.
    pub fn factorize_with_pivot_threshold(
        matrix: &CsrMatrix<T>,
        pivot_threshold: T::Real,
    ) -> Result<Self, SparseLuError> {
        Self::factorize_simplicial(matrix, pivot_threshold, false)
    }

    /// Factorize with threshold partial pivoting and optional MC64-style
    /// preprocessing (max-weight matching + row/column equilibration, see
    /// [`equilibration`]).
    ///
    /// Preprocessing is off by default in [`factorize`](Self::factorize)
    /// and [`factorize_with_pivot_threshold`](Self::factorize_with_pivot_threshold)
    /// to preserve bit-stable behavior for existing callers; enable it here
    /// for badly scaled matrices. When the pattern admits no perfect
    /// matching the matrix is factorized as-is.
    pub fn factorize_with_preprocessing(
        matrix: &CsrMatrix<T>,
        pivot_threshold: T::Real,
        equilibrate: bool,
    ) -> Result<Self, SparseLuError> {
        Self::factorize_simplicial_guard(matrix, pivot_threshold, equilibrate, false, None)
    }

    /// [`factorize_with_preprocessing`](Self::factorize_with_preprocessing)
    /// with a caller-supplied column permutation (`col_perm[k]` is the
    /// original column eliminated at step `k`; see [`amd_ordering`] and
    /// [`colamd_ordering`]).
    pub fn factorize_with_preprocessing_and_column_perm(
        matrix: &CsrMatrix<T>,
        pivot_threshold: T::Real,
        equilibrate: bool,
        column_perm: Vec<usize>,
    ) -> Result<Self, SparseLuError> {
        Self::factorize_simplicial_guard(
            matrix,
            pivot_threshold,
            equilibrate,
            false,
            Some(column_perm),
        )
    }

    /// Factorize in *preconditioner mode*: the numerical-singularity guard
    /// is relaxed from the direct-solver contract (`eps·n·‖A‖` pivot floor)
    /// to rejecting only exact-zero pivots.
    ///
    /// A direct solve must refuse factors whose pivots sit below rounding
    /// level, because its triangular solves would otherwise return garbage
    /// silently. A preconditioner has no such contract: the outer Krylov
    /// iteration (e.g. GMRES-IR, see `crate::mixed_precision`) measures the
    /// true residual and decides convergence itself, so a factor that is a
    /// poor inverse in its worst rows is still a usable preconditioner.
    /// This matters for very-low-precision factors: with f16 the strict
    /// guard already rejects condition numbers ≳ 1e3, while GMRES-IR theory
    /// (Carson–Higham) tolerates `cond(A)` up to roughly `u_low⁻¹` ≈ 2e3
    /// for f16 factors. The returned factor may be a poor inverse; only use
    /// it where an outer iteration certifies the result.
    ///
    /// To keep the relaxed factor usable, pivots below `eps²·‖A‖` (already
    /// destroyed by rounding, and overflow-prone on inversion) are repaired
    /// by clamping them to that floor, so the factor stays finite; and
    /// [`solve`](Self::solve) on a preconditioner-mode factor normalizes the
    /// right-hand side to unit inf-norm with an exact power-of-two scaling
    /// so triangular-solve intermediates stay inside the f16 exponent range.
    pub fn factorize_for_preconditioner(matrix: &CsrMatrix<T>) -> Result<Self, SparseLuError> {
        Self::factorize_simplicial_guard(matrix, T::Real::one(), false, true, None)
    }

    fn factorize_simplicial(
        matrix: &CsrMatrix<T>,
        pivot_threshold: T::Real,
        equilibrate: bool,
    ) -> Result<Self, SparseLuError> {
        Self::factorize_simplicial_guard(matrix, pivot_threshold, equilibrate, false, None)
    }

    fn factorize_simplicial_guard(
        matrix: &CsrMatrix<T>,
        pivot_threshold: T::Real,
        equilibrate: bool,
        preconditioner_mode: bool,
        column_perm: Option<Vec<usize>>,
    ) -> Result<Self, SparseLuError> {
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(SparseLuError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }

        validate_finite_matrix_values(matrix)?;

        if !pivot_threshold.is_finite() {
            return Err(SparseLuError::InvalidPivotThreshold);
        }

        if n == 0 {
            return Ok(Self::empty(SparseLuBackend::Simplicial));
        }

        let mut preprocess = None;
        let equilibrated;
        let matrix = if equilibrate {
            match equilibration::equilibrate(matrix) {
                Some((scaled, pre)) => {
                    preprocess = Some(Box::new(pre));
                    equilibrated = scaled;
                    &equilibrated
                }
                None => matrix,
            }
        } else {
            matrix
        };

        // Step 1: fill-reducing column ordering (COLAMD by default, or a
        // caller-supplied permutation such as AMD).
        let col_perm = column_perm.unwrap_or_else(|| colamd(matrix));
        // Reject non-permutations before any indexing (caller-supplied).
        {
            let mut seen = vec![false; n];
            for &column in &col_perm {
                if column >= n || seen[column] {
                    return Err(SparseLuError::InvalidColumnPermutation { column });
                }
                seen[column] = true;
            }
        }
        let mut col_perm_inv = vec![0usize; n];
        for (i, &p) in col_perm.iter().enumerate() {
            col_perm_inv[p] = i;
        }

        // Step 2: Build column-oriented representation of A*Q
        let mut a_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        for row in 0..n {
            let start = matrix.row_ptrs[row];
            let end = matrix.row_ptrs[row + 1];
            for idx in start..end {
                let col = matrix.col_indices[idx];
                let perm_col = col_perm_inv[col];
                a_cols[perm_col].push((row, matrix.values[idx]));
            }
        }
        // No per-column sort: pushes above visit CSR rows in ascending order
        // and each row contributes at most one entry per column (strictly
        // increasing columns are enforced by `validate_csr_lu_structure`),
        // so every column is already in strictly ascending row order.

        // Step 3: Compute column elimination tree of A^T*A.
        // This provides an upper bound on L's structure for any pivot sequence,
        // enabling O(nnz) sparse triangular solve instead of O(n^2).
        let etree = compute_col_etree(&a_cols, n);

        // Build row -> columns mapping for reach computation.
        let nrows = a_cols
            .iter()
            .flat_map(|col| col.iter().map(|&(r, _)| r))
            .max()
            .map_or(0, |m| m + 1);
        let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); nrows];
        for (k, col) in a_cols.iter().enumerate() {
            for &(row, _) in col {
                row_to_cols[row].push(k);
            }
        }

        // Step 4: Left-looking numeric factorization with etree-guided reach
        let mut l_col_ptrs = vec![0usize; n + 1];
        let mut l_row_indices: Vec<usize> = Vec::new();
        let mut l_values: Vec<T> = Vec::new();
        let mut u_col_ptrs = vec![0usize; n + 1];
        let mut u_row_indices: Vec<usize> = Vec::new();
        let mut u_values: Vec<T> = Vec::new();

        let mut row_perm: Vec<usize> = (0..n).collect();
        let mut row_perm_inv: Vec<usize> = (0..n).collect();

        let mut work: Vec<T> = vec![T::zero(); n];
        // Touched rows of the current column, each pushed exactly once in
        // first-occurrence order (see the touch generation below); doubles
        // as the batch finite-check set and the pivot-scan source.
        let mut work_nz: Vec<usize> = Vec::with_capacity(n);
        // Generation marks shared by the column-reach walk and the
        // load/update touch set (two generations per column, never
        // concurrent, so one array suffices).
        let mut marks = vec![0usize; n];
        let mut mark_gen = 0usize;
        // Column scratch, reused across columns instead of reallocated:
        // `reach` (sorted column reach), `l_nz` (unique touched rows > k in
        // first-occurrence order: the pivot-scan order and the L-column
        // store order), and `low_nz` (unique touched rows <= k, for the
        // workspace clear).
        let mut reach: Vec<usize> = Vec::new();
        let mut l_nz: Vec<usize> = Vec::new();
        let mut low_nz: Vec<usize> = Vec::new();

        let pivot_threshold = pivot_threshold.max(T::Real::zero()).min(T::Real::one());
        let matrix_scale = matrix
            .values
            .iter()
            .fold(T::Real::zero(), |scale, value| scale.max(value.norm()));
        // In preconditioner mode the guard rejects only exact-zero pivots
        // (see factorize_for_preconditioner); the direct-solver contract
        // keeps the conservative eps·n·‖A‖ floor.
        let pivot_tol = if preconditioner_mode {
            T::Real::zero()
        } else {
            let dimension =
                T::Real::from_usize(n.max(1)).ok_or(SparseLuError::NumericalBreakdown {
                    stage: "pivot tolerance dimension conversion",
                    index: n,
                })?;
            let tolerance =
                T::Real::epsilon() * dimension * matrix_scale.max(T::Real::min_positive_value());
            if !tolerance.is_finite() {
                return Err(SparseLuError::NumericalBreakdown {
                    stage: "pivot tolerance formation",
                    index: n,
                });
            }
            tolerance
        };

        for k in 0..n {
            // Compute sparse column reach via etree, before any `work`
            // write: it reads only `a_cols`/`row_to_cols`/the etree, so it
            // may precede the load. Starting columns: all j < k sharing an
            // original row with column k. Walk each up the etree to find all
            // columns that need updating.
            mark_gen += 1;
            marks[k] = mark_gen;
            reach.clear();
            for &(row, _) in &a_cols[k] {
                for &j in &row_to_cols[row] {
                    if j >= k || marks[j] == mark_gen {
                        continue;
                    }
                    let mut col = j;
                    while marks[col] != mark_gen {
                        marks[col] = mark_gen;
                        reach.push(col);
                        let p = etree[col];
                        if p >= k || p == usize::MAX {
                            break;
                        }
                        col = p;
                    }
                }
            }
            reach.sort_unstable();

            // Touch generation shared by the load and the update below.
            // Invariant: `marks[i] == touch_gen` iff `i` is in `work_nz`, so
            // each touched row is pushed exactly once, in first-occurrence
            // order — the old separate dedup scan over the heavily duplicated
            // `work_nz` is gone and the partition below only splits the
            // already-unique set. Push order is unchanged, so pivot
            // tie-breaks and stored values are bitwise identical.
            mark_gen += 1;
            let touch_gen = mark_gen;

            // Load column k of permuted A into work vector
            for &(row, val) in &a_cols[k] {
                let perm_row = row_perm_inv[row];
                work[perm_row] = val;
                if marks[perm_row] != touch_gen {
                    marks[perm_row] = touch_gen;
                    work_nz.push(perm_row);
                }
            }

            // Left-looking update: sparse triangular solve using only reach columns
            // Exact LU: do NOT drop small entries — every non-zero must be kept.
            for &j in &reach {
                let u_jk = work[j];
                if !u_jk.norm().is_finite() {
                    return Err(SparseLuError::NumericalBreakdown {
                        stage: "simplicial reach update",
                        index: j,
                    });
                }
                if u_jk == T::zero() {
                    continue;
                }
                // One range check per (column, reach) pair instead of one per
                // element: same iteration, fewer bounds checks.
                let l_rows = &l_row_indices[l_col_ptrs[j]..l_col_ptrs[j + 1]];
                let l_vals = &l_values[l_col_ptrs[j]..l_col_ptrs[j + 1]];
                // NOTE (mark-split attempt, 2026-09-28): a separate mark
                // pass plus a branchless FMA pass LOST by ~0.6ms at grid40
                // (2.68 -> 3.27ms update): the L re-stream plus the doubled
                // row_perm_inv loads cost more than the predictable mark
                // branches save. Keep the fused loop.
                for (&stored_row, &multiplier) in l_rows.iter().zip(l_vals.iter()) {
                    let i = row_perm_inv[stored_row];
                    work[i] -= multiplier * u_jk;
                    if marks[i] != touch_gen {
                        marks[i] = touch_gen;
                        work_nz.push(i);
                    }
                }
            }

            // Batched finiteness gate over the touched rows: once non-finite
            // a `work` entry stays non-finite (accumulation only, cleared
            // after the column). Fused into the partition loop below (one
            // pass): reports the first touched row in first-occurrence
            // order, exactly as the old standalone scan did.
            // Partition the already-unique touched set: `work_nz` holds each
            // touched row exactly once in first-occurrence order (see the
            // touch marks above), so this keeps the pivot scan's tie-breaks
            // exactly as the old dedup scan did, without re-scanning
            // duplicates. Fused with the finiteness gate above into one
            // pass: same first-bad-row error (the check precedes the push
            // per element and aborts identically), same partitions.
            l_nz.clear();
            low_nz.clear();
            for &i in &work_nz {
                if !work[i].norm().is_finite() {
                    return Err(SparseLuError::NumericalBreakdown {
                        stage: "simplicial left-looking update",
                        index: i,
                    });
                }
                if i > k {
                    l_nz.push(i);
                } else {
                    low_nz.push(i);
                }
            }

            // Partial pivoting: find largest |work[i]| for i >= k
            // Only scan the deduplicated touched rows tracked in l_nz
            let diagonal_val = work[k].norm();
            if !diagonal_val.is_finite() {
                return Err(SparseLuError::NumericalBreakdown {
                    stage: "simplicial pivot search",
                    index: k,
                });
            }
            let mut max_row = k;
            let mut max_val = diagonal_val;
            // Scan the unique touched set in first-occurrence order (strict
            // `>` keeps the first maximum). All entries exceed `k` by
            // construction, so the `i <= k` skip below never triggers.
            for &i in &l_nz {
                if i <= k {
                    continue;
                }
                let v = work[i].norm();
                if !v.is_finite() {
                    return Err(SparseLuError::NumericalBreakdown {
                        stage: "simplicial pivot search",
                        index: i,
                    });
                }
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }

            if max_val <= pivot_tol {
                return Err(SparseLuError::SingularMatrix(k));
            }
            let pivot_row = if diagonal_val >= pivot_threshold * max_val {
                k
            } else {
                max_row
            };

            // Swap rows k and pivot_row
            if pivot_row != k {
                work.swap(k, pivot_row);
                let orig_k = row_perm[k];
                let orig_p = row_perm[pivot_row];
                row_perm[k] = orig_p;
                row_perm[pivot_row] = orig_k;
                row_perm_inv[orig_p] = k;
                row_perm_inv[orig_k] = pivot_row;
            }

            // Pivot repair (preconditioner mode only): a pivot below
            // eps²·‖A‖ carries no reliable information in T arithmetic —
            // rounding has already destroyed it — and its reciprocal would
            // overflow to infinity in the triangular solves (observed with
            // f16 factors, producing NaN GMRES-IR residuals). Clamp it to
            // the repair floor, preserving the sign of its real part; the
            // factor then solves a slightly perturbed problem. The floor is
            // deliberately far below the eps·‖A‖ direct-solver guard so
            // rounding-meaningful pivots of ill-conditioned matrices are
            // left intact; it only catches values already lost to
            // underflow/cancellation. Acceptable only where an outer
            // iteration certifies the result — never for a direct solve.
            if preconditioner_mode {
                let repair_floor = T::Real::epsilon()
                    * T::Real::epsilon()
                    * matrix_scale.max(T::Real::min_positive_value());
                if work[k].norm() < repair_floor {
                    work[k] = T::from_real(if work[k].re() < T::Real::zero() {
                        -repair_floor
                    } else {
                        repair_floor
                    });
                }
            }

            let diag = work[k];

            // Store U column k: entries from reach + diagonal
            // Exact LU: keep every non-zero, do not drop small entries.
            for &j in &reach {
                if work[j] != T::zero() {
                    u_row_indices.push(j);
                    u_values.push(work[j]);
                }
            }
            if work[k] != T::zero() {
                u_row_indices.push(k);
                u_values.push(work[k]);
            }
            u_col_ptrs[k + 1] = u_row_indices.len();

            // Store L column k: entries at rows > k from work_nz
            // Exact LU: keep every non-zero, do not drop small entries.
            let diag_inv = diag.inv();
            if !diag_inv.norm().is_finite() {
                return Err(SparseLuError::NumericalBreakdown {
                    stage: "simplicial pivot inversion",
                    index: k,
                });
            }
            // Store the L column in first-occurrence order (no sort): every
            // consumer iterates a stored column commutatively — the numeric
            // update and forward substitution scatter each entry into a
            // disjoint row, level scheduling reduces with max, checkpoints
            // round-trip the multiset — so sorted order buys nothing. The
            // order is deterministic (same input, same bytes); the only
            // order-sensitive downstream consumer is pivot tie-breaking
            // (first maximum in first-occurrence order), which is unchanged
            // on the validated corpus including exact-tie matrices.
            for &i in &l_nz {
                if work[i] != T::zero() {
                    let multiplier = work[i] * diag_inv;
                    if !multiplier.norm().is_finite() {
                        return Err(SparseLuError::NumericalBreakdown {
                            stage: "simplicial multiplier formation",
                            index: i,
                        });
                    }
                    l_row_indices.push(row_perm[i]);
                    l_values.push(multiplier);
                }
                // Clear-on-store: nothing reads `work[l_nz]` after this
                // loop except the workspace clear below, so zero each row
                // right after its last read and shrink the clear to
                // `low_nz + k`. Same values cleared, one less pass over
                // the L rows.
                work[i] = T::zero();
            }
            l_col_ptrs[k + 1] = l_row_indices.len();

            // Clear workspace: only touched entries (O(nnz) instead of O(n)).
            // `l_nz` rows were already zeroed on store above; `low_nz`
            // covers the rest exactly once; `k` is added defensively (an
            // untouched diagonal is already zero).

            for &idx in low_nz.iter().chain(std::iter::once(&k)) {
                work[idx] = T::zero();
            }
            work_nz.clear();
        }

        // Compute level sets for parallel triangular solves
        let l_level_sets = compute_l_level_sets(&l_col_ptrs, &l_row_indices, &row_perm, n);
        let u_level_sets = compute_u_level_sets(&u_col_ptrs, &u_row_indices, n);
        let (row_perm_inv, u_diag_idx) =
            derive_solve_caches(&u_col_ptrs, &u_row_indices, &row_perm, n);

        Ok(Self {
            n,
            backend: SparseLuBackend::Simplicial,
            supernode_count: 0,
            l_col_ptrs,
            l_row_indices,
            l_values,
            u_col_ptrs,
            u_row_indices,
            u_values,
            row_perm,
            col_perm,
            l_level_sets,
            u_level_sets,
            row_perm_inv,
            u_diag_idx,
            preprocess,
            pattern: CsrPattern::from_matrix(matrix),
            preconditioner_mode,
        })
    }

    /// Solve A*x = b using the pre-computed factorization.
    ///
    /// Computes x = Q * U^{-1} * L^{-1} * P * b.
    /// Uses dependency-complete level scheduling for large factors and exact
    /// serial forward and backward substitution for small factors.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SparseLuError> {
        let n = self.n;
        if b.len() != n {
            return Err(SparseLuError::DimensionMismatch {
                expected: n,
                actual: b.len(),
            });
        }
        if let Some(index) = b.iter().position(|value| !value.norm().is_finite()) {
            return Err(SparseLuError::NonFiniteRhs(index));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        self.validate_solve_diagonals()?;
        self.solve_prevalidated(b)
    }

    /// Reject missing, non-finite, or below-floor U diagonals before a
    /// solve writes into fresh storage. O(n) over the cached diagonal
    /// index (previously a per-column linear search, O(nnz) per solve);
    /// same first-bad-column errors, same order.
    fn validate_solve_diagonals(&self) -> Result<(), SparseLuError> {
        // Preconditioner-mode factors intentionally admit pivots below
        // the direct-solver floor; the outer iteration is responsible
        // for certifying that approximate inverse. Still reject an
        // exact zero pivot in every mode.
        let pivot_floor = T::Real::from_f64(1e-30).unwrap_or_else(T::Real::min_positive_value);
        for column in 0..self.n {
            let diagonal = match self.u_diag_idx[column] {
                usize::MAX => return Err(SparseLuError::SingularMatrix(column)),
                index => self.u_values[index],
            };
            let norm = diagonal.norm();
            if !norm.is_finite() {
                return Err(SparseLuError::NonFiniteDiagonal(column));
            }
            if norm == T::Real::zero() || (!self.preconditioner_mode && norm <= pivot_floor) {
                return Err(SparseLuError::SingularMatrix(column));
            }
        }
        Ok(())
    }

    /// Solve core behind [`Self::solve`] and [`Self::solve_many_into`]:
    /// `b` has length `n`, is finite, and the diagonals passed
    /// [`Self::validate_solve_diagonals`].
    ///
    /// No stored-factor finite rescan here (validate-once): the fields are
    /// private with no `&mut` API so factors are immutable after
    /// construction, and every constructor establishes finiteness — the
    /// simplicial kernel checks each stored multiplier and update, the
    /// supernodal path scans once in `from_supernodal_numeric`, and
    /// checkpoint restore validates in `validate_checkpoint_state`.
    /// Intermediates computed below are still scanned per solve.
    fn solve_prevalidated(&self, b: &[T]) -> Result<Vec<T>, SparseLuError> {
        let n = self.n;
        // Preconditioner mode (f16-range safety): triangular-solve
        // intermediates live near the ‖U·x‖ scale, which can overflow the
        // f16 range at the 2^13 cast scale even when the final answer is
        // tiny (observed as inf/NaN GMRES-IR residuals on a cond=1e3
        // matrix). Normalize the right-hand side to unit inf-norm with an
        // exact power-of-two scaling and undo it on the result; power-of-two
        // scaling introduces no rounding, so this is free accuracy-wise.
        let mut output_scale = T::Real::one();
        let b_scaled_storage;
        let b: &[T] = if self.preconditioner_mode {
            let b_max = b
                .iter()
                .map(ComplexField::norm)
                .fold(T::Real::zero(), T::Real::max);
            match b_max.to_f64() {
                Some(m) if m.is_finite() && m > 1.0 => {
                    let shift = m.log2().ceil() as i32;
                    let down = T::Real::from_f64(2f64.powi(-shift)).unwrap_or_else(T::Real::one);
                    output_scale = T::Real::from_f64(2f64.powi(shift)).unwrap_or_else(T::Real::one);
                    b_scaled_storage = b
                        .iter()
                        .map(|v| *v * T::from_real(down))
                        .collect::<Vec<T>>();
                    &b_scaled_storage
                }
                _ => b,
            }
        } else {
            b
        };

        // Step 1: Apply row scaling (when preprocessed) and row permutation:
        // y = P * Dr * b.
        let mut y = vec![T::zero(); n];
        match &self.preprocess {
            Some(pre) => {
                for (i, slot) in y.iter_mut().enumerate() {
                    let original_row = self.row_perm[i];
                    *slot = b[original_row] * T::from_real(pre.row_scale[original_row]);
                }
            }
            None => {
                for i in 0..n {
                    y[i] = b[self.row_perm[i]];
                }
            }
        }
        if let Some(index) = first_non_finite(&y) {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "right-hand-side preprocessing",
                index,
            });
        }

        // Step 2: Forward substitution: L * z = y (L is unit lower triangular).
        let row_perm_inv = &self.row_perm_inv;

        // Thin waves run serially inline: a rayon dispatch costs more than
        // their flops (see `PAR_LEVEL_MIN_NNZ`). Serial-in-wave is bitwise
        // identical to parallel-in-wave (disjoint writes), so waves below
        // the floor take the same serial column loop either way.
        let par_min_nnz = par_level_min_nnz();
        let use_parallel = n >= 1000 && self.l_level_sets.len() > 1;

        if use_parallel {
            // Level-set parallel forward substitution.
            // Columns within each level are independent and can be processed
            // in parallel. Between levels, we synchronise.
            let y_raw = y.as_mut_ptr();
            for level in &self.l_level_sets {
                if level.len() > 1 && wave_nnz(level, &self.l_col_ptrs) >= par_min_nnz {
                    // SAFETY: columns in the same level write to disjoint rows
                    // (guaranteed by level-set construction). y[j] is only read
                    // (it was written in a previous level).
                    unsafe {
                        parallel_l_level(
                            level,
                            y_raw,
                            &self.l_col_ptrs,
                            &self.l_row_indices,
                            &self.l_values,
                            row_perm_inv,
                        );
                    }
                } else {
                    for &j in level {
                        let start = self.l_col_ptrs[j];
                        let end = self.l_col_ptrs[j + 1];
                        let yj = y[j];
                        for idx in start..end {
                            let i = row_perm_inv[self.l_row_indices[idx]];
                            y[i] -= self.l_values[idx] * yj;
                        }
                    }
                }
            }
        } else {
            for j in 0..n {
                let start = self.l_col_ptrs[j];
                let end = self.l_col_ptrs[j + 1];
                let yj = y[j];
                for idx in start..end {
                    let i = row_perm_inv[self.l_row_indices[idx]];
                    y[i] -= self.l_values[idx] * yj;
                }
            }
        }
        if let Some(index) = first_non_finite(&y) {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "forward substitution",
                index,
            });
        }

        // Step 3: Backward substitution: U * w = z
        let mut w = vec![T::zero(); n];

        let use_parallel_u = n >= 1000 && self.u_level_sets.len() > 1;

        // Diagonals are cached (`u_diag_idx`): `validate_solve_diagonals`
        // already rejected missing/zero/non-finite pivots, so the index is
        // valid for every column here (no per-column search, no zero risk
        // beyond what validation admitted in preconditioner mode).
        if use_parallel_u {
            let w_raw = w.as_mut_ptr();
            let y_raw = y.as_mut_ptr();
            for level in &self.u_level_sets {
                if level.len() > 1 && wave_nnz(level, &self.u_col_ptrs) >= par_min_nnz {
                    unsafe {
                        parallel_u_level(
                            level,
                            w_raw,
                            y_raw,
                            &self.u_col_ptrs,
                            &self.u_row_indices,
                            &self.u_values,
                        );
                    }
                } else {
                    for &j in level {
                        let start = self.u_col_ptrs[j];
                        let end = self.u_col_ptrs[j + 1];
                        debug_assert!(self.u_diag_idx[j] != usize::MAX);
                        let diag = self.u_values[self.u_diag_idx[j]];
                        w[j] = y[j] * diag.inv();
                        for idx in start..end {
                            let i = self.u_row_indices[idx];
                            if i < j {
                                y[i] -= self.u_values[idx] * w[j];
                            }
                        }
                    }
                }
            }
        } else {
            for j in (0..n).rev() {
                let start = self.u_col_ptrs[j];
                let end = self.u_col_ptrs[j + 1];
                debug_assert!(self.u_diag_idx[j] != usize::MAX);
                let diag = self.u_values[self.u_diag_idx[j]];
                w[j] = y[j] * diag.inv();
                for idx in start..end {
                    let i = self.u_row_indices[idx];
                    if i < j {
                        y[i] -= self.u_values[idx] * w[j];
                    }
                }
            }
        }
        if let Some(index) = first_non_finite(&w) {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "backward substitution",
                index,
            });
        }

        // Step 4: Apply inverse column permutation: x = Q_m * Dc * Q * w.
        let mut x = vec![T::zero(); n];
        match &self.preprocess {
            Some(pre) => {
                for (i, &wi) in w.iter().enumerate() {
                    let equilibrated_col = self.col_perm[i];
                    let original_col = pre.col_perm[equilibrated_col];
                    x[original_col] = wi * T::from_real(pre.col_scale[equilibrated_col]);
                }
            }
            None => {
                for i in 0..n {
                    x[self.col_perm[i]] = w[i];
                }
            }
        }

        // Undo the preconditioner-mode power-of-two rhs normalization.
        if output_scale != T::Real::one() {
            let scale = T::from_real(output_scale);
            for value in &mut x {
                *value *= scale;
            }
        }

        if let Some(index) = first_non_finite(&x) {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "solution scaling",
                index,
            });
        }

        Ok(x)
    }

    /// Solve `Aᵀ*x = b` using the pre-computed factorization.
    ///
    /// The solve reuses the stored permutations and triangular factors rather
    /// than refactorizing a transposed matrix. This is exact for both
    /// simplicial and supernodal factors, and preserves the optional
    /// equilibration and preconditioner-mode scaling contracts.
    pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, SparseLuError> {
        self.solve_adjoint(b, false)
    }

    /// Solve `Aᴴ*x = b` using the pre-computed factorization.
    pub fn solve_hermitian(&self, b: &[T]) -> Result<Vec<T>, SparseLuError> {
        self.solve_adjoint(b, true)
    }

    fn solve_adjoint(&self, b: &[T], hermitian: bool) -> Result<Vec<T>, SparseLuError> {
        let n = self.n;
        if b.len() != n {
            return Err(SparseLuError::DimensionMismatch {
                expected: n,
                actual: b.len(),
            });
        }
        if let Some(index) = b.iter().position(|value| !value.norm().is_finite()) {
            return Err(SparseLuError::NonFiniteRhs(index));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        self.validate_solve_diagonals()?;
        self.solve_adjoint_prevalidated(b, hermitian)
    }

    /// Adjoint core behind [`Self::solve_adjoint`] and
    /// [`Self::solve_adjoint_many_into`]: `b` has length `n`, is finite,
    /// and the diagonals passed [`Self::validate_solve_diagonals`]. Same
    /// validate-once contract as [`Self::solve_prevalidated`] (no
    /// stored-factor rescan; intermediates still scanned).
    fn solve_adjoint_prevalidated(
        &self,
        b: &[T],
        hermitian: bool,
    ) -> Result<Vec<T>, SparseLuError> {
        let n = self.n;
        // Keep the same power-of-two normalization used by forward
        // preconditioner-mode solves. It is a scalar linear rescaling and
        // therefore applies identically to transpose/Hermitian systems.
        let mut output_scale = T::Real::one();
        let b_scaled_storage = if self.preconditioner_mode {
            let b_max = b
                .iter()
                .map(ComplexField::norm)
                .fold(T::Real::zero(), T::Real::max);
            match b_max.to_f64() {
                Some(m) if m.is_finite() && m > 1.0 => {
                    let shift = m.log2().ceil() as i32;
                    let down = T::Real::from_f64(2f64.powi(-shift)).unwrap_or_else(T::Real::one);
                    output_scale = T::Real::from_f64(2f64.powi(shift)).unwrap_or_else(T::Real::one);
                    Some(
                        b.iter()
                            .map(|value| *value * T::from_real(down))
                            .collect::<Vec<_>>(),
                    )
                }
                _ => None,
            }
        } else {
            None
        };
        let rhs = b_scaled_storage.as_deref().unwrap_or(b);

        let row_perm_inv = &self.row_perm_inv;

        // P B Q = L U, where B is A (or Dr*A*Qm*Dc when preprocessing is
        // enabled). For the adjoint, first form Qᵀ Dc Qmᵀ b.
        let mut rhs_factor = vec![T::zero(); n];
        match &self.preprocess {
            Some(pre) => {
                for (factor_col, slot) in rhs_factor.iter_mut().enumerate() {
                    let equilibrated_col = self.col_perm[factor_col];
                    let original_col = pre.col_perm[equilibrated_col];
                    *slot = rhs[original_col] * T::from_real(pre.col_scale[equilibrated_col]);
                }
            }
            None => {
                for factor_col in 0..n {
                    rhs_factor[factor_col] = rhs[self.col_perm[factor_col]];
                }
            }
        }

        // Uᴴ/Uᵀ is lower triangular. U is stored in CSC, so column `k`
        // contains exactly the coefficients needed for row `k` of Uᵀ.
        let mut y = vec![T::zero(); n];
        for k in 0..n {
            let mut value = rhs_factor[k];
            let start = self.u_col_ptrs[k];
            let end = self.u_col_ptrs[k + 1];
            // Cached diagonal: `validate_solve_diagonals` already rejected
            // missing pivots, so the index is valid for every column.
            debug_assert!(self.u_diag_idx[k] != usize::MAX);
            let diagonal = self.u_values[self.u_diag_idx[k]];
            for index in start..end {
                let row = self.u_row_indices[index];
                if row < k {
                    let coefficient = if hermitian {
                        self.u_values[index].conj()
                    } else {
                        self.u_values[index]
                    };
                    value -= coefficient * y[row];
                }
            }
            let diagonal = if hermitian { diagonal.conj() } else { diagonal };
            y[k] = value / diagonal;
        }

        // Lᴴ/Lᵀ is upper triangular with unit diagonal. L row indices are
        // retained in original-row numbering, so map them back to factor
        // numbering before applying the transpose.
        let mut z = y;
        for k in (0..n).rev() {
            let start = self.l_col_ptrs[k];
            let end = self.l_col_ptrs[k + 1];
            let mut value = z[k];
            for index in start..end {
                let row = row_perm_inv[self.l_row_indices[index]];
                if row > k {
                    let coefficient = if hermitian {
                        self.l_values[index].conj()
                    } else {
                        self.l_values[index]
                    };
                    value -= coefficient * z[row];
                }
            }
            z[k] = value;
        }

        // Undo P and (when present) Dr. Column scaling is real, so the
        // transpose and Hermitian paths share the same equilibration factors.
        let mut x = vec![T::zero(); n];
        match &self.preprocess {
            Some(pre) => {
                for (factor_row, &zi) in z.iter().enumerate() {
                    let original_row = self.row_perm[factor_row];
                    x[original_row] = zi * T::from_real(pre.row_scale[original_row]);
                }
            }
            None => {
                for factor_row in 0..n {
                    x[self.row_perm[factor_row]] = z[factor_row];
                }
            }
        }
        if output_scale != T::Real::one() {
            let scale = T::from_real(output_scale);
            for value in &mut x {
                *value *= scale;
            }
        }
        if let Some(index) = first_non_finite(&x) {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "adjoint solution scaling",
                index,
            });
        }
        Ok(x)
    }

    /// Solve row-major multiple right-hand sides into caller-owned storage.
    ///
    /// Both `b` and `x` use `[row * nrhs + rhs]` layout. All dimensions and
    /// input values are validated before `x` is changed. The implementation
    /// reuses the checked single-RHS path so the block contract has identical
    /// pivot, equilibration, and failure semantics.
    pub fn solve_many_into(&self, b: &[T], x: &mut [T], nrhs: usize) -> Result<(), SparseLuError> {
        let expected = self
            .n
            .checked_mul(nrhs)
            .ok_or(SparseLuError::SizeOverflow {
                dimension: self.n,
                nrhs,
            })?;
        if b.len() != expected || x.len() != expected {
            return Err(SparseLuError::DimensionMismatch {
                expected,
                actual: if b.len() != expected {
                    b.len()
                } else {
                    x.len()
                },
            });
        }
        if let Some(rhs) = b.iter().position(|value| !value.norm().is_finite()) {
            return Err(SparseLuError::NonFiniteRhs(rhs));
        }
        // Validate the block and the diagonals once; each column below is
        // copied from the validated block so per-column revalidation would
        // be pure overhead (same pivot/equilibration/failure semantics —
        // the triangular cores still scan their own intermediates).
        self.validate_solve_diagonals()?;

        // Keep the caller-owned destination transactional. A later RHS can
        // still fail during triangular solution (for example when a finite
        // but extremely large RHS overflows a tiny pivot); publishing earlier
        // columns would leave a misleading partial block result behind.
        let mut solutions = vec![T::zero(); expected];
        let mut column_rhs = vec![T::zero(); self.n];
        for rhs in 0..nrhs {
            for row in 0..self.n {
                column_rhs[row] = b[row * nrhs + rhs];
            }
            let solution = self.solve_prevalidated(&column_rhs)?;
            for row in 0..self.n {
                solutions[row * nrhs + rhs] = solution[row];
            }
        }
        x.copy_from_slice(&solutions);
        Ok(())
    }

    /// Solve row-major multiple right-hand sides, allocating the result.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, SparseLuError> {
        let expected = self
            .n
            .checked_mul(nrhs)
            .ok_or(SparseLuError::SizeOverflow {
                dimension: self.n,
                nrhs,
            })?;
        let mut x = vec![T::zero(); expected];
        self.solve_many_into(b, &mut x, nrhs)?;
        Ok(x)
    }

    /// Solve row-major multiple transpose right-hand sides, allocating result.
    pub fn solve_transpose_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, SparseLuError> {
        let expected = self
            .n
            .checked_mul(nrhs)
            .ok_or(SparseLuError::SizeOverflow {
                dimension: self.n,
                nrhs,
            })?;
        let mut x = vec![T::zero(); expected];
        self.solve_transpose_many_into(b, &mut x, nrhs)?;
        Ok(x)
    }

    /// Solve multiple transpose right-hand sides into caller-owned storage.
    ///
    /// Both `b` and `x` use row-major `[row * nrhs + rhs]` layout. All
    /// right-hand sides are validated and solved before `x` is published, so
    /// a later singular or numerical failure cannot expose a partial block.
    pub fn solve_transpose_many_into(
        &self,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<(), SparseLuError> {
        self.solve_adjoint_many_into(b, x, nrhs, false)
    }

    /// Solve row-major multiple Hermitian-adjoint right-hand sides,
    /// allocating result.
    pub fn solve_hermitian_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, SparseLuError> {
        let expected = self
            .n
            .checked_mul(nrhs)
            .ok_or(SparseLuError::SizeOverflow {
                dimension: self.n,
                nrhs,
            })?;
        let mut x = vec![T::zero(); expected];
        self.solve_hermitian_many_into(b, &mut x, nrhs)?;
        Ok(x)
    }

    /// Solve multiple Hermitian-adjoint right-hand sides into caller-owned
    /// storage.
    ///
    /// Both `b` and `x` use row-major `[row * nrhs + rhs]` layout. The
    /// destination remains unchanged if any column fails validation or solve.
    pub fn solve_hermitian_many_into(
        &self,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<(), SparseLuError> {
        self.solve_adjoint_many_into(b, x, nrhs, true)
    }

    /// Solve multiple transpose/Hermitian-adjoint right-hand sides.
    fn solve_adjoint_many_into(
        &self,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
        hermitian: bool,
    ) -> Result<(), SparseLuError> {
        let expected = self
            .n
            .checked_mul(nrhs)
            .ok_or(SparseLuError::SizeOverflow {
                dimension: self.n,
                nrhs,
            })?;
        if b.len() != expected || x.len() != expected {
            return Err(SparseLuError::DimensionMismatch {
                expected,
                actual: if b.len() != expected {
                    b.len()
                } else {
                    x.len()
                },
            });
        }
        if let Some(index) = b.iter().position(|value| !value.norm().is_finite()) {
            return Err(SparseLuError::NonFiniteRhs(index));
        }
        // Validate once for the whole block (see `solve_many_into`).
        self.validate_solve_diagonals()?;

        let mut solutions = vec![T::zero(); expected];
        let mut column_rhs = vec![T::zero(); self.n];
        for rhs in 0..nrhs {
            for row in 0..self.n {
                column_rhs[row] = b[row * nrhs + rhs];
            }
            let solution = self.solve_adjoint_prevalidated(&column_rhs, hermitian)?;
            for row in 0..self.n {
                solutions[row * nrhs + rhs] = solution[row];
            }
        }
        x.copy_from_slice(&solutions);
        Ok(())
    }

    /// Solve and apply residual-correction iterative refinement.
    ///
    /// The matrix must be the same matrix used to build this factorization.
    /// Refinement stops once `||b - A x||₂ / ||b||₂ <= tolerance` or after
    /// `max_iterations` correction solves.
    pub fn solve_refined(
        &self,
        matrix: &CsrMatrix<T>,
        b: &[T],
        tolerance: T::Real,
        max_iterations: usize,
    ) -> Result<SparseRefinementResult<T>, SparseLuError> {
        if !tolerance.is_finite() || tolerance < T::Real::zero() {
            return Err(SparseLuError::InvalidRefinementTolerance(
                "tolerance must be finite and non-negative".to_string(),
            ));
        }
        if matrix.num_rows != self.n || matrix.num_cols != self.n {
            return Err(SparseLuError::MatrixDimensionMismatch {
                expected: self.n,
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        self.try_accepts_value_update(matrix)?;
        if b.len() != self.n {
            return Err(SparseLuError::DimensionMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }

        let mut x = self.solve(b)?;
        let provenance = SparseRefinementProvenance::ReusedValueOnlyPattern {
            structure_fingerprint: self.structure_fingerprint(),
        };
        let rhs_norm =
            checked_refinement_norm(b, "right-hand-side norm")?.max(T::Real::min_positive_value());
        let mut initial_relative_residual = T::Real::zero();
        let mut relative_residual = T::Real::zero();
        for iteration in 0..=max_iterations {
            let ax = matrix.matvec(&x);
            let residual: Vec<T> = b.iter().zip(ax).map(|(rhs, value)| *rhs - value).collect();
            let residual_norm = checked_refinement_norm(&residual, "residual norm")?;
            relative_residual = checked_refinement_ratio::<T>(residual_norm, rhs_norm)?;
            if iteration == 0 {
                initial_relative_residual = relative_residual;
            }
            if relative_residual <= tolerance {
                return Ok(SparseRefinementResult {
                    x,
                    iterations: iteration,
                    initial_relative_residual,
                    relative_residual,
                    tolerance,
                    converged: true,
                    stop_reason: SparseRefinementStopReason::ToleranceReached,
                    provenance,
                });
            }
            if iteration == max_iterations {
                break;
            }
            let correction = self.solve(&residual)?;
            for (value, delta) in x.iter_mut().zip(correction) {
                *value += delta;
            }
            if let Some(index) = first_non_finite(&x) {
                return Err(SparseLuError::NumericalBreakdown {
                    stage: "refinement correction accumulation",
                    index,
                });
            }
        }
        Ok(SparseRefinementResult {
            x,
            iterations: max_iterations,
            initial_relative_residual,
            relative_residual,
            tolerance,
            converged: false,
            stop_reason: SparseRefinementStopReason::MaxIterations,
            provenance,
        })
    }

    /// Refine several right-hand sides with one reusable sparse-LU factor.
    ///
    /// The RHS and returned results use row-major `n × nrhs` storage.  The
    /// matrix dimensions and exact CSR structure are checked once before any
    /// column solve, so a changed sparsity pattern cannot expose a partial
    /// batch of refined solutions. Numeric value updates that preserve the
    /// row-pointer/column-index pattern remain supported and are reported in
    /// each result's provenance metadata.
    pub fn solve_refined_many(
        &self,
        matrix: &CsrMatrix<T>,
        b: &[T],
        nrhs: usize,
        tolerance: T::Real,
        max_iterations: usize,
    ) -> Result<Vec<SparseRefinementResult<T>>, SparseLuError> {
        if !tolerance.is_finite() || tolerance < T::Real::zero() {
            return Err(SparseLuError::InvalidRefinementTolerance(
                "tolerance must be finite and non-negative".to_string(),
            ));
        }
        if matrix.num_rows != self.n || matrix.num_cols != self.n {
            return Err(SparseLuError::MatrixDimensionMismatch {
                expected: self.n,
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        self.try_accepts_value_update(matrix)?;
        let expected = self
            .n
            .checked_mul(nrhs)
            .ok_or(SparseLuError::SizeOverflow {
                dimension: self.n,
                nrhs,
            })?;
        if b.len() != expected {
            return Err(SparseLuError::DimensionMismatch {
                expected,
                actual: b.len(),
            });
        }

        let mut results = Vec::with_capacity(nrhs);
        for rhs_index in 0..nrhs {
            let mut column_rhs = Vec::with_capacity(self.n);
            for row in 0..self.n {
                column_rhs.push(b[row * nrhs + rhs_index]);
            }
            results.push(self.solve_refined(matrix, &column_rhs, tolerance, max_iterations)?);
        }
        Ok(results)
    }
}

fn checked_refinement_norm<T: ComplexField>(
    values: &[T],
    stage: &'static str,
) -> Result<T::Real, SparseLuError> {
    let norm = T::vec_norm(values);
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(SparseLuError::NumericalBreakdown {
            stage,
            index: values.len(),
        })
    }
}

fn checked_refinement_ratio<T: ComplexField>(
    numerator: T::Real,
    denominator: T::Real,
) -> Result<T::Real, SparseLuError> {
    let ratio = numerator / denominator;
    if ratio.is_finite() {
        Ok(ratio)
    } else {
        Err(SparseLuError::NumericalBreakdown {
            stage: "relative residual formation",
            index: 0,
        })
    }
}

/// Parallel forward substitution for one level.
///
/// # Safety
/// Columns in `level` must write to disjoint rows. `y_ptr` must point to
/// a valid allocation of at least `max(row_perm_inv) + 1` elements.
unsafe fn parallel_l_level<T: ComplexField>(
    level: &[usize],
    y_ptr: *mut T,
    l_col_ptrs: &[usize],
    l_row_indices: &[usize],
    l_values: &[T],
    row_perm_inv: &[usize],
) {
    // Wrap pointer so it can cross thread boundaries.
    let ptr = y_ptr as usize;
    level.par_iter().for_each(|&j| {
        let y = ptr as *mut T;
        let start = l_col_ptrs[j];
        let end = l_col_ptrs[j + 1];
        // SAFETY: disjoint write regions guaranteed by level-set construction.
        let yj = unsafe { *y.add(j) };
        for idx in start..end {
            let i = row_perm_inv[l_row_indices[idx]];
            unsafe { *y.add(i) -= l_values[idx] * yj };
        }
    });
}

/// Parallel backward substitution for one level.
///
/// # Safety
/// Columns in `level` must write to disjoint rows/positions. Pointers must
/// be valid with sufficient allocation.
unsafe fn parallel_u_level<T: ComplexField>(
    level: &[usize],
    w_ptr: *mut T,
    y_ptr: *mut T,
    u_col_ptrs: &[usize],
    u_row_indices: &[usize],
    u_values: &[T],
) {
    let wp = w_ptr as usize;
    let yp = y_ptr as usize;
    level.par_iter().for_each(|&j| {
        let w = wp as *mut T;
        let y = yp as *mut T;
        let start = u_col_ptrs[j];
        let end = u_col_ptrs[j + 1];
        let mut diag = T::zero();
        for idx in start..end {
            if u_row_indices[idx] == j {
                diag = u_values[idx];
                break;
            }
        }
        // SAFETY: disjoint write regions guaranteed by level-set construction.
        unsafe {
            *w.add(j) = (*y.add(j)) * diag.inv();
            let wj = *w.add(j);
            for idx in start..end {
                let i = u_row_indices[idx];
                if i < j {
                    *y.add(i) -= u_values[idx] * wj;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Elimination tree and level-set computation
// ---------------------------------------------------------------------------

/// Compute the column elimination tree of A^T*A without forming A^T*A.
///
/// Returns `parent[j]` = smallest column index k > j such that columns j and k
/// share at least one row nonzero (possibly through fill-in). Root columns have
/// `parent[j] = usize::MAX`.
///
/// Algorithm from Tim Davis (CSPARSE), O(nnz * alpha(n)) via path compression.
pub(super) fn compute_col_etree<T: ComplexField>(
    a_cols: &[Vec<(usize, T)>],
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    let nrows = a_cols
        .iter()
        .flat_map(|col| col.iter().map(|&(r, _)| r))
        .max()
        .map_or(0, |m| m + 1);

    let mut parent = vec![usize::MAX; n];
    let mut ancestor = vec![usize::MAX; n];
    let mut prev = vec![usize::MAX; nrows];

    for (k, col) in a_cols.iter().enumerate().take(n) {
        for &(row, _) in col {
            let mut i = prev[row];
            while i != usize::MAX && i < k {
                let inext = ancestor[i];
                ancestor[i] = k;
                if inext == usize::MAX {
                    parent[i] = k;
                }
                i = inext;
            }
            prev[row] = k;
        }
    }

    parent
}

/// Compute level sets for parallel forward substitution (L solve).
///
/// Columns in the same level write to disjoint rows and can be processed
/// in parallel. A column is scheduled only after every earlier column that
/// updates its RHS entry.
fn compute_l_level_sets(
    l_col_ptrs: &[usize],
    l_row_indices: &[usize],
    row_perm: &[usize],
    n: usize,
) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }

    let mut row_perm_inv = vec![0usize; n];
    for k in 0..n {
        row_perm_inv[row_perm[k]] = k;
    }

    // `ready_level[i]` is the first level at which y[i] may be read or
    // written. Every update to y[i] advances it past the writer's level.
    // This encodes both read-after-write dependencies and serialises
    // otherwise independent columns that scatter into the same row.
    let mut ready_level = vec![0usize; n];
    let mut col_level: Vec<usize> = vec![0; n];
    // One scratch buffer for all columns (was: one allocation per column).
    let mut write_rows: Vec<usize> = Vec::new();

    for j in 0..n {
        let start = l_col_ptrs[j];
        let end = l_col_ptrs[j + 1];

        let mut level = ready_level[j];
        write_rows.clear();
        for idx in start..end {
            let i = row_perm_inv[l_row_indices[idx]];
            if i < n && i != j {
                write_rows.push(i);
                level = level.max(ready_level[i]);
            }
        }

        col_level[j] = level;
        for &row in &write_rows {
            ready_level[row] = level + 1;
        }
    }

    let max_level = col_level.iter().copied().max().unwrap_or(0);
    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for j in 0..n {
        levels[col_level[j]].push(j);
    }
    levels
}

/// Compute level sets for parallel backward substitution (U solve).
///
/// Columns in the same level are independent. Columns are discovered from
/// right to left, and the resulting levels are processed in ascending order.
fn compute_u_level_sets(
    u_col_ptrs: &[usize],
    u_row_indices: &[usize],
    n: usize,
) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }

    // Same dependency-complete algorithm as L, but processed right-to-left.
    let mut ready_level = vec![0usize; n];
    let mut col_level: Vec<usize> = vec![0; n];
    // One scratch buffer for all columns (was: one allocation per column).
    let mut write_rows: Vec<usize> = Vec::new();

    for j in (0..n).rev() {
        let start = u_col_ptrs[j];
        let end = u_col_ptrs[j + 1];

        let mut level = ready_level[j];
        write_rows.clear();
        for &i in &u_row_indices[start..end] {
            if i < j {
                write_rows.push(i);
                level = level.max(ready_level[i]);
            }
        }

        col_level[j] = level;
        for &row in &write_rows {
            ready_level[row] = level + 1;
        }
    }

    let max_level = col_level.iter().copied().max().unwrap_or(0);
    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for j in 0..n {
        levels[col_level[j]].push(j);
    }
    // Within each level, reverse order (largest j first) for backward sub
    for lvl in &mut levels {
        lvl.reverse();
    }
    levels
}

/// Derive the per-solve caches (`row_perm_inv`, `u_diag_idx`) from
/// validated factor structure. Every constructor calls this exactly once;
/// `row_perm` is a permutation and the column pointers are bounds-checked
/// at every call site, so the scatters below cannot panic.
fn derive_solve_caches(
    u_col_ptrs: &[usize],
    u_row_indices: &[usize],
    row_perm: &[usize],
    n: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut row_perm_inv = vec![0usize; n];
    for (factor_row, &original_row) in row_perm.iter().enumerate() {
        row_perm_inv[original_row] = factor_row;
    }
    let mut u_diag_idx = vec![usize::MAX; n];
    for column in 0..n {
        let start = u_col_ptrs[column];
        let end = u_col_ptrs[column + 1];
        if let Some(offset) = u_row_indices[start..end]
            .iter()
            .position(|&row| row == column)
        {
            u_diag_idx[column] = start + offset;
        }
    }
    (row_perm_inv, u_diag_idx)
}

/// Minimum aggregate column-nnz for a level-set wave to pay a rayon
/// dispatch instead of running serially inline.
///
/// Rationale (measured 2026-09-27): 2D-PDE orderings produce hundreds of
/// thin waves, and each `par_iter` wave costs ~10-30µs while its flops
/// finish in ~1µs serially — the n>=1000 solve spent 15-25ms in dispatch
/// on ~0.1ms of work (grid-32 convection-diffusion: 24ms default-pool,
/// 2.9ms at 16 threads, vs 0.022ms for the n=256 all-serial path).
/// Serial-in-wave is bitwise identical to parallel-in-wave (disjoint
/// write sets, each target written once), so the threshold only moves
/// work between identical-execution paths — no numerical effect.
/// Tuned over {2048, 8192, 32768, ∞} at n=1024/1600/4096/16384: 8192
/// still dispatched marginal waves at n=16384 (~13% slower than
/// serial-only there); 32768 ties serial-only at every size while
/// keeping the parallel path for truly fat waves (>~30-100µs of serial
/// work, clearly above dispatch cost).
/// `SONIUM_SPARSE_PAR_MIN_NNZ=0` restores dispatch-every-wave.
const PAR_LEVEL_MIN_NNZ: usize = 32768;

/// Runtime override for [`PAR_LEVEL_MIN_NNZ`] (diagnostic/A-B +
/// kill-switch). Read once per solve, like the dense `SONIUM_LU_*` gates.
pub(crate) fn par_level_min_nnz() -> usize {
    std::env::var("SONIUM_SPARSE_PAR_MIN_NNZ")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(PAR_LEVEL_MIN_NNZ)
}

/// Aggregate stored nonzeros of one wave's columns (the dispatch
/// breakeven metric for [`PAR_LEVEL_MIN_NNZ`]). Shared with the sparse
/// Cholesky solves, which have the identical wave structure.
pub(crate) fn wave_nnz(level: &[usize], col_ptrs: &[usize]) -> usize {
    level.iter().map(|&j| col_ptrs[j + 1] - col_ptrs[j]).sum()
}

// ---------------------------------------------------------------------------
// COLAMD ordering
// ---------------------------------------------------------------------------

/// Column Approximate Minimum Degree (COLAMD) fill-reducing ordering.
///
/// Returns a column permutation that heuristically minimises fill-in in the
/// L and U factors of a sparse LU factorization with partial pivoting.
///
/// This is a genuine AMD over the column intersection graph — quotient
/// graph with element absorption, approximate external-degree updates, and
/// dense-variable removal — following Amestoy/Davis/Duff (1996) and
/// CSparse's `cs_amd`. See [`colamd`] for the implementation. It replaces a
/// previous static minimum-degree heuristic that never simulated fill.
pub(crate) fn colamd<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    colamd::colamd(matrix)
}

/// Compute a fill-reducing column permutation for a sparse matrix.
pub fn colamd_ordering<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    colamd::colamd(matrix)
}

/// Compute an approximate-minimum-degree variable permutation for a sparse
/// matrix with (near-)symmetric structure.
///
/// The pattern is symmetrised (`A + Aᵀ`), so this is the ordering of choice
/// for LU on symmetric-structure problems — the SuiteSparse UMFPACK/CHOLMOD
/// territory where AMD decisively beats column orderings. For structurally
/// unsymmetric matrices prefer [`colamd_ordering`].
pub fn amd_ordering<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    amd::amd(matrix)
}

/// Compute a nested-dissection variable permutation for a sparse matrix
/// with (near-)symmetric structure, via METIS-style multilevel bisection
/// of the symmetrised (`A + Aᵀ`) pattern.
///
/// On PDE grids this beats COLAMD on fill *and* numeric time: separators
/// concentrate fill into dense diagonal blocks that the supernodal kernels
/// process efficiently (128x128 nonsymmetric Laplacian, 5-rep means:
/// ~1.00M vs 1.27M COLAMD nonzeros at 88.6 vs 111.0 ms numeric, i.e. ~1.25x;
/// total with ordering cost 110.6 vs 123.5 ms). AMD reaches still less fill
/// (~0.83M) but its deep elimination structure runs ~2.2x slower through
/// the panel kernels (248.5 ms), so ND is the time winner on grids. For
/// structurally unsymmetric matrices prefer [`colamd_ordering`].
pub fn nd_ordering<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    use sonium_solvers_sparse::partition::{DissectionNode, nested_dissection};

    const ND_MIN_BLOCK_SIZE: usize = 16;
    let n = matrix.num_rows;
    if n <= ND_MIN_BLOCK_SIZE {
        return (0..n).collect();
    }
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            if j != i {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }
    }
    let all: Vec<usize> = (0..n).collect();
    let max_levels = ((n as f64) / (ND_MIN_BLOCK_SIZE as f64)).log2().ceil() as usize + 1;
    let mut nodes: Vec<DissectionNode> = Vec::new();
    nested_dissection(
        &adjacency,
        &all,
        ND_MIN_BLOCK_SIZE,
        max_levels,
        0,
        &mut nodes,
    );
    // `nodes` is post-order by construction: concatenate each node's DOFs
    // to get the elimination permutation.
    let mut perm = Vec::with_capacity(n);
    for node in &nodes {
        perm.extend_from_slice(&node.global_indices);
    }
    debug_assert_eq!(perm.len(), n, "nested dissection must cover all DOFs");
    perm
}

/// Implement `Preconditioner<T>` so the sparse LU can be used as a
/// direct preconditioner or coarse-grid solver in AMG.
impl<T: ComplexField> sonium_solvers_core::traits::Preconditioner<T> for SparseLuFactorization<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        self.solve(r)
            .expect("preconditioner input must match a valid sparse LU factorization")
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        let x = self
            .solve(r)
            .expect("preconditioner input must match a valid sparse LU factorization");
        y.copy_from_slice(&x);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        self.solve_transpose(r)
            .expect("preconditioner input must match valid sparse LU factorization")
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        self.solve_hermitian(r)
            .expect("preconditioner input must match valid sparse LU factorization")
    }

    fn try_apply_transpose(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        self.solve_transpose(r)
            .map_err(|error| SolverError::from_failure("sparse LU transpose", &error))
    }

    fn try_apply_hermitian(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        self.solve_hermitian(r)
            .map_err(|error| SolverError::from_failure("sparse LU Hermitian transpose", &error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_core::dense::DenseMatrix;

    #[test]
    fn solve_rejects_rhs_dimension_mismatch() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let factorization = SparseLuFactorization::factorize(&matrix).unwrap();
        assert!(matches!(
            factorization.solve(&[1.0]),
            Err(SparseLuError::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn solve_rejects_zero_and_non_finite_u_diagonals() {
        for diagonal in [0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let matrix = CsrMatrix::<f64>::identity(2);
            let mut factorization = SparseLuFactorization::factorize(&matrix).unwrap();
            let diagonal_index = (factorization.u_col_ptrs[0]..factorization.u_col_ptrs[1])
                .find(|&index| factorization.u_row_indices[index] == 0)
                .expect("identity factorization must store its first U diagonal");
            factorization.u_values[diagonal_index] = diagonal;

            let result = factorization.solve(&[1.0, 2.0]);
            if diagonal == 0.0 {
                assert!(matches!(result, Err(SparseLuError::SingularMatrix(0))));
            } else {
                assert!(matches!(result, Err(SparseLuError::NonFiniteDiagonal(0))));
            }
        }
    }

    #[test]
    fn factorization_rejects_non_finite_matrix_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
            matrix.values[0] = value;
            for result in [
                SparseLuFactorization::factorize(&matrix),
                SparseLuFactorization::factorize_supernodal(&matrix),
            ] {
                assert!(
                    matches!(
                        result,
                        Err(SparseLuError::NonFiniteMatrix { row: 0, col: 0 })
                    ),
                    "non-finite matrix entry must be rejected before numeric factorization"
                );
            }
        }
    }

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(2.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    #[test]
    fn structure_fingerprint_distinguishes_pattern_from_value_updates() {
        let base = CsrMatrix::from_dense(&DenseMatrix::<f64>::eye(3), 0.0);
        let factor = SparseLuFactorization::factorize(&base).unwrap();

        let mut values_changed = base.clone();
        values_changed.values[0] = 2.0;
        assert!(factor.accepts_value_update(&values_changed));
        assert!(factor.try_accepts_value_update(&values_changed).is_ok());
        assert_eq!(
            factor.structure_fingerprint(),
            SparseLuFactorization::factorize(&values_changed)
                .unwrap()
                .structure_fingerprint()
        );

        let mut pattern_changed = base.clone();
        pattern_changed.col_indices[0] = 1;
        assert!(!factor.accepts_value_update(&pattern_changed));
        assert!(matches!(
            factor.solve_refined(&pattern_changed, &[1.0, 2.0, 3.0], 1e-12, 1),
            Err(SparseLuError::PatternMismatch)
        ));
    }

    #[test]
    fn sparse_lu_rejects_duplicate_and_unsorted_csr_rows() {
        let factor = SparseLuFactorization::factorize(&CsrMatrix::<f64>::identity(2)).unwrap();
        for columns in [vec![0, 0, 1], vec![1, 0, 1]] {
            let matrix = CsrMatrix::try_from_raw_parts(
                2,
                2,
                vec![0, 2, 3],
                columns,
                vec![1.0_f64, 2.0, 3.0],
            )
            .expect("raw CSR structure is valid before canonical-row policy");

            assert!(matches!(
                SparseLuFactorization::factorize(&matrix),
                Err(SparseLuError::InvalidCsrStructure(_))
            ));
            assert!(matches!(
                SparseLuFactorization::analyze_supernodal(&matrix),
                Err(SparseLuError::InvalidCsrStructure(_))
            ));
            assert!(matches!(
                factor.try_accepts_value_update(&matrix),
                Err(SparseLuError::InvalidCsrStructure(_))
            ));
            assert!(matches!(
                factor.solve_refined(&matrix, &[1.0, 2.0], 1e-12, 1),
                Err(SparseLuError::InvalidCsrStructure(_))
            ));
        }
    }

    #[test]
    fn test_sparse_lu_small() {
        // 3×3 system
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![2.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![1.0_f64, 2.0, 3.0];

        let lu = SparseLuFactorization::factorize(&csr).unwrap();
        let x = lu.solve(&b).unwrap();

        // Check A*x ≈ b
        let ax = csr.matvec(&x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Residual at {i}: {:.2e}",
                (ax[i] - b[i]).abs()
            );
        }
    }

    #[test]
    fn test_sparse_lu_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let lu = SparseLuFactorization::factorize(&a).unwrap();
        let x = lu.solve(&b).unwrap();

        let ax = a.matvec(&x);
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|&bi| bi.powi(2)).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-10,
            "Sparse LU residual too large: {:.2e}",
            residual / b_norm
        );
    }

    #[test]
    fn amd_ordering_beats_colamd_fill_on_symmetric_grid() {
        // 16x16 symmetric-pattern grid through the caller-perm entry
        // points: AMD must clearly beat COLAMD fill here (this mirrors the
        // n=4096 vendor shootout where reference AMD reached 137K vs 248K
        // for COLAMD), and the AMD-path solve must be accurate.
        let matrix = laplacian_2d(16, 16);
        let n = 256;
        let amd_symbolic = SparseLuFactorization::analyze_supernodal_with_column_perm(
            &matrix,
            amd_ordering(&matrix),
        )
        .expect("AMD analyze should succeed");
        let colamd_symbolic = SparseLuFactorization::analyze_supernodal_with_column_perm(
            &matrix,
            colamd_ordering(&matrix),
        )
        .expect("COLAMD analyze should succeed");
        let amd_factor =
            SparseLuFactorization::factorize_supernodal_with_symbolic(&matrix, &amd_symbolic, 1.0)
                .expect("AMD numeric should succeed");
        let colamd_factor = SparseLuFactorization::factorize_supernodal_with_symbolic(
            &matrix,
            &colamd_symbolic,
            1.0,
        )
        .expect("COLAMD numeric should succeed");
        assert!(
            amd_factor.factor_nonzeros() < colamd_factor.factor_nonzeros(),
            "AMD fill {} should beat COLAMD fill {}",
            amd_factor.factor_nonzeros(),
            colamd_factor.factor_nonzeros()
        );
        let b = vec![1.0_f64; n];
        let x = amd_factor.solve(&b).expect("AMD solve should succeed");
        let ax = matrix.matvec(&x);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            num += (ax[i] - b[i]) * (ax[i] - b[i]);
            den += b[i] * b[i];
        }
        assert!(
            (num / den).sqrt() < 1e-10,
            "AMD-path residual too large: {:e}",
            (num / den).sqrt()
        );
    }

    #[test]
    fn nd_ordering_beats_colamd_fill_on_symmetric_grid() {
        // 64x64 symmetric-pattern grid through the caller-perm entry
        // points: ND must clearly beat COLAMD fill here (nested dissection
        // needs size for separators to pay off — it loses on tiny grids;
        // 128x128 vendor shootout: ~1.0M ND vs 1.34M COLAMD nonzeros), and
        // the ND-path solve must be accurate.
        let matrix = laplacian_2d(64, 64);
        let n = 4096;
        let nd_symbolic = SparseLuFactorization::analyze_supernodal_with_column_perm(
            &matrix,
            nd_ordering(&matrix),
        )
        .expect("ND analyze should succeed");
        let colamd_symbolic = SparseLuFactorization::analyze_supernodal_with_column_perm(
            &matrix,
            colamd_ordering(&matrix),
        )
        .expect("COLAMD analyze should succeed");
        let nd_factor =
            SparseLuFactorization::factorize_supernodal_with_symbolic(&matrix, &nd_symbolic, 1.0)
                .expect("ND numeric should succeed");
        let colamd_factor = SparseLuFactorization::factorize_supernodal_with_symbolic(
            &matrix,
            &colamd_symbolic,
            1.0,
        )
        .expect("COLAMD numeric should succeed");
        assert!(
            nd_factor.factor_nonzeros() < colamd_factor.factor_nonzeros(),
            "ND fill {} should beat COLAMD fill {}",
            nd_factor.factor_nonzeros(),
            colamd_factor.factor_nonzeros()
        );
        let b = vec![1.0_f64; n];
        let x = nd_factor.solve(&b).expect("ND solve should succeed");
        let ax = matrix.matvec(&x);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            num += (ax[i] - b[i]) * (ax[i] - b[i]);
            den += b[i] * b[i];
        }
        assert!(
            (num / den).sqrt() < 1e-10,
            "ND-path residual too large: {:e}",
            (num / den).sqrt()
        );
    }

    #[test]
    fn invalid_column_perm_rejected() {
        let matrix = laplacian_2d(4, 4);
        // Duplicate entry.
        let mut perm: Vec<usize> = (0..16).collect();
        perm[0] = 1;
        assert!(matches!(
            SparseLuFactorization::analyze_supernodal_with_column_perm(&matrix, perm),
            Err(SparseLuError::InvalidColumnPermutation { .. })
        ));
        // Out-of-range entry.
        let mut perm: Vec<usize> = (0..16).collect();
        perm[15] = 16;
        assert!(matches!(
            SparseLuFactorization::analyze_supernodal_with_column_perm(&matrix, perm),
            Err(SparseLuError::InvalidColumnPermutation { .. })
        ));
        // Wrong length.
        assert!(matches!(
            SparseLuFactorization::analyze_supernodal_with_column_perm(&matrix, vec![0, 1, 2]),
            Err(SparseLuError::DimensionMismatch { .. })
        ));
        // Simplicial entry point enforces the same contract (duplicate 4,
        // missing 3).
        let mut perm: Vec<usize> = (0..16).collect();
        perm[3] = 4;
        assert!(matches!(
            SparseLuFactorization::factorize_with_preprocessing_and_column_perm(
                &matrix, 1.0, false, perm
            ),
            Err(SparseLuError::InvalidColumnPermutation { .. })
        ));
    }

    #[test]
    fn test_sparse_lu_nonsymmetric() {
        let dense = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                4.0_f64, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 2.0, 4.0, 1.0, 0.0, 0.0, 3.0, 4.0,
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![1.0_f64, 2.0, 3.0, 4.0];

        let lu = SparseLuFactorization::factorize(&csr).unwrap();
        let x = lu.solve(&b).unwrap();

        let ax = csr.matvec(&x);
        for i in 0..4 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Row {i}: ax={:.6}, b={:.6}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_simplicial_dedup_paths_2d_laplacian() {
        // 2D 5-point Laplacian (20x20): heavy fill-in exercises the
        // mark-dedup pivot scan, the unique L-column store, and the unique
        // workspace clear (a missed touched row would corrupt the factor).
        let nx = 20;
        let n = nx * nx;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                // Ascending column order within each row (up, left, diag,
                // right, down): i-nx < i-1 < i < i+1 < i+nx always holds.
                if iy > 0 {
                    col_indices.push(i - nx);
                    values.push(-1.0);
                }
                if ix > 0 {
                    col_indices.push(i - 1);
                    values.push(-1.0);
                }
                col_indices.push(i);
                values.push(4.0);
                if ix + 1 < nx {
                    col_indices.push(i + 1);
                    values.push(-1.0);
                }
                if iy + 1 < nx {
                    col_indices.push(i + nx);
                    values.push(-1.0);
                }
                row_ptrs.push(col_indices.len());
            }
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let b = vec![1.0_f64; n];

        let lu = SparseLuFactorization::factorize(&a).unwrap();
        let x = lu.solve(&b).unwrap();

        let ax = a.matvec(&x);
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1e-10, "2D Laplacian residual: {residual:.2e}");
    }

    #[test]
    fn test_sparse_lu_as_preconditioner() {
        let n = 30;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let lu = SparseLuFactorization::factorize(&a).unwrap();

        // Use as preconditioner (should give exact solution in 1 step)
        use sonium_solvers_core::traits::Preconditioner;
        let x = <SparseLuFactorization<f64> as Preconditioner<f64>>::apply(&lu, &b);

        let ax = a.matvec(&x);
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1e-10, "Preconditioner residual: {residual:.2e}");
    }

    #[test]
    fn test_colamd_valid_permutation() {
        let n = 20;
        let a = laplacian_1d(n);
        let perm = colamd(&a);
        assert_eq!(perm.len(), n);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_colamd_arrowhead_correctness() {
        // Arrowhead matrix: column 0 (the hub) connects to all others.
        // COLAMD should not place the hub first (it has the highest degree),
        // and the factorization must produce a correct solve regardless of ordering.
        let dense = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                2.0_f64, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0,
            ],
        );
        let a = CsrMatrix::from_dense(&dense, 1e-15);

        // COLAMD ordering: hub (col 0, degree 3) must not be first
        let perm = colamd(&a);
        assert_eq!(perm.len(), 4);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        // The hub's approximate degree is capped at `remaining - 1`, so it
        // can tie the last surviving leaf and lose the queue-order
        // tie-break; then the leaf is mass eliminated immediately after it.
        // Either way the hub is last or second-to-last with zero extra fill.
        let hub_position = perm.iter().position(|&p| p == 0).unwrap();
        assert!(
            hub_position >= 2,
            "the denser hub column should be eliminated after every leaf but the tied one"
        );

        // Factorization must solve correctly
        let b = vec![1.0_f64, 2.0, 3.0, 4.0];
        let lu = SparseLuFactorization::factorize(&a).unwrap();
        let x = lu.solve(&b).unwrap();
        let ax = a.matvec(&x);
        for i in 0..4 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "arrowhead row {i}: ax={:.6}, b={:.6}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn threshold_pivoting_retains_acceptable_diagonal() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![0.9_f64, 1.0, 1.0, 2.0]);
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let threshold =
            SparseLuFactorization::factorize_with_pivot_threshold(&matrix, 0.5).unwrap();
        let strict = SparseLuFactorization::factorize_with_pivot_threshold(&matrix, 1.0).unwrap();
        assert_eq!(threshold.row_perm[0], 0);
        assert_eq!(strict.row_perm[0], 1);

        let b = vec![1.0, -2.0];
        for factor in [&threshold, &strict] {
            let x = factor.solve(&b).unwrap();
            let ax = matrix.matvec(&x);
            for (actual, expected) in ax.iter().zip(&b) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn iterative_refinement_reports_and_reduces_residual() {
        let n = 12;
        let dense = DenseMatrix::from_fn(n, n, |i, j| {
            1.0_f64 / (i + j + 1) as f64 + if i == j { 1e-5 } else { 0.0 }
        });
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let factor = SparseLuFactorization::factorize(&matrix).unwrap();
        let expected = vec![1.0_f64; n];
        let b = matrix.matvec(&expected);
        let plain = factor.solve(&b).unwrap();
        let plain_ax = matrix.matvec(&plain);
        let plain_residual = plain_ax
            .iter()
            .zip(&b)
            .map(|(actual, rhs)| (actual - rhs) * (actual - rhs))
            .sum::<f64>()
            .sqrt();

        let refined = factor.solve_refined(&matrix, &b, 1e-14, 5).unwrap();
        let refined_ax = matrix.matvec(&refined.x);
        let refined_residual = refined_ax
            .iter()
            .zip(&b)
            .map(|(actual, rhs)| (actual - rhs) * (actual - rhs))
            .sum::<f64>()
            .sqrt();
        assert!(refined_residual <= plain_residual * 1.01 + 1e-16);
        assert!(refined.initial_relative_residual.is_finite());
        assert!(refined.relative_residual.is_finite());
        assert_eq!(refined.tolerance, 1e-14);
        assert_eq!(
            refined.stop_reason,
            SparseRefinementStopReason::ToleranceReached
        );
        assert_eq!(
            refined.provenance,
            SparseRefinementProvenance::ReusedValueOnlyPattern {
                structure_fingerprint: factor.structure_fingerprint()
            }
        );
        assert!(refined.converged);
        assert!(refined.iterations <= 5);
    }

    #[test]
    fn iterative_refinement_reports_max_iteration_provenance() {
        let matrix = CsrMatrix::identity(3);
        let factor = SparseLuFactorization::factorize(&matrix).unwrap();
        let mut updated = matrix.clone();
        updated.values.fill(2.0);

        let result = factor
            .solve_refined(&updated, &[1.0, 1.0, 1.0], 0.0, 0)
            .expect("same-pattern value update must be accepted");

        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.initial_relative_residual, result.relative_residual);
        assert_eq!(result.tolerance, 0.0);
        assert_eq!(
            result.stop_reason,
            SparseRefinementStopReason::MaxIterations
        );
        assert_eq!(
            result.provenance,
            SparseRefinementProvenance::ReusedValueOnlyPattern {
                structure_fingerprint: factor.structure_fingerprint()
            }
        );
    }

    #[test]
    fn solve_refined_rejects_same_size_matrix_with_changed_pattern() {
        let matrix = CsrMatrix::from_dense(&DenseMatrix::<f64>::eye(3), 0.0);
        let factor = SparseLuFactorization::factorize(&matrix).unwrap();

        let mut changed_dense = DenseMatrix::<f64>::eye(3);
        changed_dense[[0, 1]] = 0.25;
        let changed = CsrMatrix::from_dense(&changed_dense, 0.0);

        let error = factor
            .solve_refined(&changed, &[1.0, 2.0, 3.0], 1e-12, 2)
            .expect_err("refinement must reject a changed CSR pattern");
        assert!(matches!(error, SparseLuError::PatternMismatch));
    }

    #[test]
    fn scaled_sparse_system_is_not_rejected_by_absolute_pivot_floor() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![
                2e-30_f64, -1e-30, 0.0, -1e-30, 2e-30, -1e-30, 0.0, -1e-30, 2e-30,
            ],
        );
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let factor = SparseLuFactorization::factorize(&matrix).unwrap();
        let b = vec![1e-30, 0.0, 1e-30];
        let x = factor.solve(&b).unwrap();
        let ax = matrix.matvec(&x);
        for (actual, expected) in ax.iter().zip(&b) {
            assert!((actual - expected).abs() < 1e-42);
        }
    }

    #[test]
    fn test_sparse_lu_complex() {
        use num_complex::Complex64;
        let dense = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, -1.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let lu = SparseLuFactorization::factorize(&csr).unwrap();
        let x = lu.solve(&b).unwrap();

        let ax = csr.matvec(&x);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).norm() < 1e-10,
                "Complex residual at {i}: {:.2e}",
                (ax[i] - b[i]).norm()
            );
        }
    }

    /// Build a 2D Laplacian on an nx * ny grid (5-point stencil).
    fn laplacian_2d(nx: usize, ny: usize) -> CsrMatrix<f64> {
        let n = nx * ny;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iy * nx + ix;
                if iy > 0 {
                    col_indices.push(idx - nx);
                    values.push(-1.0);
                }
                if ix > 0 {
                    col_indices.push(idx - 1);
                    values.push(-1.0);
                }
                col_indices.push(idx);
                values.push(4.0);
                if ix < nx - 1 {
                    col_indices.push(idx + 1);
                    values.push(-1.0);
                }
                if iy < ny - 1 {
                    col_indices.push(idx + nx);
                    values.push(-1.0);
                }
                row_ptrs.push(col_indices.len());
            }
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    #[test]
    fn test_sparse_lu_2d_laplacian() {
        // Keep this as a unit-scale 2D stencil. Larger sparse-LU scaling cases
        // belong in benchmarks or QA, not the default `cargo test` path.
        let a = laplacian_2d(8, 6);
        let n = a.num_rows;
        let b = vec![1.0_f64; n];

        let lu = SparseLuFactorization::factorize(&a).unwrap();
        let x = lu.solve(&b).unwrap();

        let ax = a.matvec(&x);
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|&bi| bi.powi(2)).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-10,
            "2D Laplacian residual too large: {:.2e}",
            residual / b_norm
        );
    }

    #[test]
    fn test_etree_tridiagonal() {
        // 1D Laplacian: tridiagonal => etree should be a chain: parent[j] = j+1
        let n = 10;
        let a = laplacian_1d(n);
        let col_perm = colamd(&a);
        let mut col_perm_inv = vec![0usize; n];
        for (i, &p) in col_perm.iter().enumerate() {
            col_perm_inv[p] = i;
        }
        let mut a_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for row in 0..n {
            for idx in a.row_ptrs[row]..a.row_ptrs[row + 1] {
                let perm_col = col_perm_inv[a.col_indices[idx]];
                a_cols[perm_col].push((row, a.values[idx]));
            }
        }
        let etree = compute_col_etree(&a_cols, n);

        // Etree should have n-1 non-root entries (one root)
        let roots = etree.iter().filter(|&&p| p == usize::MAX).count();
        assert!(
            roots >= 1,
            "etree should have at least one root, got {roots}"
        );
        // Every non-root should point to a strictly larger column
        for (j, &parent) in etree.iter().enumerate().take(n) {
            if parent != usize::MAX {
                assert!(parent > j, "etree[{j}] = {parent} should be > {j}");
            }
        }
    }

    #[test]
    fn test_sparse_lu_nonsymmetric_large() {
        // Convection-diffusion: nonsymmetric, tests pivoting + etree interaction
        let n = 100;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0 - 0.5); // diffusion + upwind convection
            }
            col_indices.push(i);
            values.push(2.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0 + 0.5); // diffusion - upwind convection
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let b = vec![1.0_f64; n];

        let lu = SparseLuFactorization::factorize(&a).unwrap();
        let x = lu.solve(&b).unwrap();

        let ax = a.matvec(&x);
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|&bi| bi.powi(2)).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-10,
            "Nonsymmetric residual too large: {:.2e}",
            residual / b_norm
        );
    }

    #[test]
    fn test_sparse_lu_no_dropping() {
        // Regression test for issue #15: exact LU must not drop small entries.
        // A matrix with a small off-diagonal (1e-12) should produce an LU
        // factor that includes that entry, not discard it because |entry| < 1e-14.
        let n = 3;
        let mut dense = DenseMatrix::from_elem(n, n, 0.0_f64);
        dense[[0, 0]] = 1.0;
        dense[[0, 1]] = 1e-12; // tiny off-diagonal
        dense[[1, 0]] = 1e-12;
        dense[[1, 1]] = 1.0;
        dense[[2, 2]] = 1.0;

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let lu = SparseLuFactorization::factorize(&csr).unwrap();

        // Count total non-zeros in L and U (excluding unit diagonal of L)
        let l_nnz = lu.l_values.len();
        let u_nnz = lu.u_values.len();

        // With the old dropping code, these tiny entries would be discarded
        // and l_nnz + u_nnz would be 0.  Exact LU must keep them.
        assert!(
            l_nnz + u_nnz >= 2,
            "Exact LU dropped small entries: L_nnz={}, U_nnz={}",
            l_nnz,
            u_nnz
        );

        // Verify the solve is still accurate
        let b = vec![1.0_f64, 2.0, 3.0];
        let x = lu.solve(&b).unwrap();
        let ax = csr.matvec(&x);
        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Solve inaccurate after removing drop tolerance: ax[{}] = {}, b[{}] = {}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn test_l_level_sets_no_write_conflicts() {
        // Regression: columns in the same level must not write to the same row.
        // Directly construct an L factor where columns 0 and 1 both write to row 2.
        let n = 4;
        let l_col_ptrs = vec![0, 1, 2, 2, 2];
        let l_row_indices = vec![2, 2]; // col 0 -> row 2, col 1 -> row 2
        let _l_values = [1.0_f64, 1.0];
        let row_perm: Vec<usize> = (0..n).collect();

        let levels = compute_l_level_sets(&l_col_ptrs, &l_row_indices, &row_perm, n);

        // Verify that columns 0 and 1 are NOT in the same level
        for level in &levels {
            let has_0 = level.contains(&0);
            let has_1 = level.contains(&1);
            assert!(
                !(has_0 && has_1),
                "Columns 0 and 1 both write to row 2 and must not share a level"
            );
        }
    }

    #[test]
    fn test_l_level_sets_respect_read_after_write_dependencies() {
        // Column 0 updates y[1], which column 1 must read before updating y[2].
        // A race-free scheduler must therefore put column 1 after column 0.
        let n = 3;
        let l_col_ptrs = vec![0, 1, 2, 2];
        let l_row_indices = vec![1, 2];
        let row_perm: Vec<usize> = (0..n).collect();

        let levels = compute_l_level_sets(&l_col_ptrs, &l_row_indices, &row_perm, n);
        let level_of = |column| {
            levels
                .iter()
                .position(|level| level.contains(&column))
                .unwrap()
        };

        assert!(
            level_of(0) < level_of(1),
            "column 1 reads y[1] after column 0 writes it"
        );
    }

    #[test]
    fn test_u_level_sets_no_write_conflicts() {
        // Regression: columns in the same level must not write to the same row.
        // Directly construct a U factor where columns 2 and 3 both write to row 0.
        let n = 4;
        let u_col_ptrs = vec![0, 0, 0, 1, 2];
        let u_row_indices = vec![0, 0]; // col 2 -> row 0, col 3 -> row 0
        let _u_values = [1.0_f64, 1.0];

        let levels = compute_u_level_sets(&u_col_ptrs, &u_row_indices, n);

        // Verify that columns 2 and 3 are NOT in the same level
        for level in &levels {
            let has_2 = level.contains(&2);
            let has_3 = level.contains(&3);
            assert!(
                !(has_2 && has_3),
                "U columns 2 and 3 both write to row 0 and must not share a level"
            );
        }
    }

    #[test]
    fn test_u_level_sets_respect_read_after_write_dependencies() {
        // Back substitution processes column 2 first. It updates y[1], which
        // column 1 must read before updating y[0].
        let n = 3;
        let u_col_ptrs = vec![0, 0, 1, 2];
        let u_row_indices = vec![0, 1];

        let levels = compute_u_level_sets(&u_col_ptrs, &u_row_indices, n);
        let level_of = |column| {
            levels
                .iter()
                .position(|level| level.contains(&column))
                .unwrap()
        };

        assert!(
            level_of(2) < level_of(1),
            "column 1 reads y[1] after column 2 writes it"
        );
    }

    fn assert_supernodal_reconstruction(a: &CsrMatrix<f64>, factor: &SparseLuFactorization<f64>) {
        let n = a.num_rows;
        let mut row_perm_inv = vec![0; n];
        for (position, &original) in factor.row_perm.iter().enumerate() {
            row_perm_inv[original] = position;
        }
        let mut l = DenseMatrix::<f64>::eye(n);
        for column in 0..n {
            for index in factor.l_col_ptrs[column]..factor.l_col_ptrs[column + 1] {
                let row = row_perm_inv[factor.l_row_indices[index]];
                l[[row, column]] = factor.l_values[index];
            }
        }
        let mut u = DenseMatrix::<f64>::zeros(n, n);
        for column in 0..n {
            for index in factor.u_col_ptrs[column]..factor.u_col_ptrs[column + 1] {
                u[[factor.u_row_indices[index], column]] = factor.u_values[index];
            }
        }
        let product = l.matmul(&u);
        let dense = a.to_dense();
        let mut maximum_error = 0.0_f64;
        let mut maximum_value = 0.0_f64;
        for row in 0..n {
            for column in 0..n {
                let expected = dense[[factor.row_perm[row], factor.col_perm[column]]];
                maximum_error = maximum_error.max((product[[row, column]] - expected).abs());
                maximum_value = maximum_value.max(expected.abs());
            }
        }
        assert!(
            maximum_error <= 2e-11 * n as f64 * maximum_value.max(1.0),
            "P*A*Q reconstruction error {maximum_error:e}"
        );
    }

    #[test]
    fn supernodal_reconstructs_and_solves_multiple_panels() {
        let n = 97;
        let dense = DenseMatrix::from_fn(n, n, |row, column| {
            if row == column {
                5.0 + row as f64 / n as f64
            } else if row.abs_diff(column) == 1 {
                -1.0 + ((row * 17 + column * 11) % 13) as f64 * 0.01
            } else if (row * 7 + column * 19) % 89 == 0 {
                0.05
            } else {
                0.0
            }
        });
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let factor = SparseLuFactorization::factorize_supernodal(&matrix).unwrap();
        assert_eq!(factor.backend(), SparseLuBackend::Supernodal);
        assert!(factor.supernode_count() > 0);
        assert_supernodal_reconstruction(&matrix, &factor);

        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();
        let x = factor.solve(&b).unwrap();
        let ax = matrix.matvec(&x);
        let error = ax
            .iter()
            .zip(&b)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(error < 2e-11, "solve residual {error:e}");
    }

    #[test]
    fn supernodal_matches_simplicial_on_nonsymmetric_laplacian() {
        let nx = 9;
        let n = nx * nx;
        let dense = DenseMatrix::from_fn(n, n, |row, column| {
            if row == column {
                4.1
            } else if row.abs_diff(column) == 1 && row / nx == column / nx {
                if row < column { -0.8 } else { -1.2 }
            } else if row.abs_diff(column) == nx {
                if row < column { -0.9 } else { -1.1 }
            } else {
                0.0
            }
        });
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let scalar = SparseLuFactorization::factorize(&matrix).unwrap();
        let supernodal = SparseLuFactorization::factorize_supernodal(&matrix).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.25).cos()).collect();
        let scalar_x = scalar.solve(&b).unwrap();
        let supernodal_x = supernodal.solve(&b).unwrap();
        for (scalar_value, supernodal_value) in scalar_x.iter().zip(&supernodal_x) {
            assert!((scalar_value - supernodal_value).abs() < 2e-11);
        }
        assert_supernodal_reconstruction(&matrix, &supernodal);
    }

    #[test]
    fn supernodal_threshold_pivoting_handles_delayed_pivots() {
        let n = 70;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            dense[[i, i]] = 0.25;
            dense[[i, (i + 1) % n]] = 2.0;
            dense[[i, (i + n - 1) % n]] = -0.125;
        }
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let factor =
            SparseLuFactorization::factorize_supernodal_with_pivot_threshold(&matrix, 0.5).unwrap();
        assert!(
            factor
                .row_perm
                .iter()
                .enumerate()
                .any(|(position, original)| position != *original)
        );
        assert_supernodal_reconstruction(&matrix, &factor);
    }

    #[test]
    fn supernodal_complex_solve_matches_residual() {
        let n = 45;
        let dense = DenseMatrix::from_fn(n, n, |row, column| {
            if row == column {
                Complex64::new(4.0 + row as f64 * 0.01, 0.25)
            } else if row.abs_diff(column) == 1 {
                Complex64::new(-1.0, (row as f64 * 0.1).sin() * 0.05)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let factor = SparseLuFactorization::factorize_supernodal(&matrix).unwrap();
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let x = factor.solve(&b).unwrap();
        let ax = matrix.matvec(&x);
        let error = ax
            .iter()
            .zip(&b)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max);
        assert!(error < 2e-11, "complex residual {error:e}");
    }

    #[test]
    fn supernodal_symbolic_analysis_reuses_pattern_and_refines() {
        let n = 66;
        let base = DenseMatrix::from_fn(n, n, |row, column| {
            if row == column {
                3.0
            } else if row.abs_diff(column) == 1 {
                -0.75
            } else {
                0.0
            }
        });
        let matrix = CsrMatrix::from_dense(&base, 0.0);
        let symbolic = SparseLuFactorization::analyze_supernodal(&matrix).unwrap();
        assert_eq!(symbolic.dimension(), n);
        assert!(symbolic.panel_count() >= 3);
        assert_eq!(symbolic.elimination_tree().len(), n);

        let shifted = DenseMatrix::from_fn(n, n, |row, column| {
            if row == column {
                3.5 + row as f64 * 1e-3
            } else if row.abs_diff(column) == 1 {
                -0.75
            } else {
                0.0
            }
        });
        let shifted_matrix = CsrMatrix::from_dense(&shifted, 0.0);
        let factor = SparseLuFactorization::factorize_supernodal_with_symbolic(
            &shifted_matrix,
            &symbolic,
            0.1,
        )
        .unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let refined = factor.solve_refined(&shifted_matrix, &b, 1e-13, 3).unwrap();
        assert!(refined.relative_residual < 1e-12);
    }

    #[test]
    fn supernodal_symbolic_rejects_different_pattern() {
        let first = CsrMatrix::from_dense(&DenseMatrix::<f64>::eye(8), 0.0);
        let symbolic = SparseLuFactorization::analyze_supernodal(&first).unwrap();
        let mut changed = DenseMatrix::<f64>::eye(8);
        changed[[0, 1]] = 0.5;
        let second = CsrMatrix::from_dense(&changed, 0.0);
        assert!(matches!(
            SparseLuFactorization::factorize_supernodal_with_symbolic(&second, &symbolic, 1.0),
            Err(SparseLuError::PatternMismatch)
        ));
    }

    #[test]
    fn supernodal_deterministic_sparse_family_crosses_panel_boundaries() {
        for n in [7_usize, 31, 32, 33, 64, 65] {
            let mut dense = DenseMatrix::<f64>::zeros(n, n);
            for row in 0..n {
                let mut row_sum = 0.0;
                for column in 0..n {
                    if row != column && (row * 37 + column * 19 + row * column * 3) % 23 == 0 {
                        let value = (((row * 11 + column * 17 + 5) % 29) as f64 - 14.0) / 31.0;
                        dense[[row, column]] = value;
                        row_sum += value.abs();
                    }
                }
                dense[[row, row]] = row_sum + 1.0 + row as f64 * 1e-3;
            }
            let matrix = CsrMatrix::from_dense(&dense, 0.0);
            let factor = SparseLuFactorization::factorize_supernodal(&matrix).unwrap();
            assert_supernodal_reconstruction(&matrix, &factor);
            let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
            let x = factor.solve(&b).unwrap();
            let ax = matrix.matvec(&x);
            let max_error = ax
                .iter()
                .zip(&b)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            assert!(max_error < 2e-11, "n={n}, residual={max_error:e}");
        }
    }

    #[test]
    fn supernodal_f32_residual_is_precision_appropriate() {
        let n = 65;
        let dense = DenseMatrix::from_fn(n, n, |row, column| {
            if row == column {
                3.0_f32
            } else if row.abs_diff(column) == 1 {
                -0.75_f32
            } else {
                0.0_f32
            }
        });
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let factor = SparseLuFactorization::factorize_supernodal(&matrix).unwrap();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let x = factor.solve(&b).unwrap();
        let ax = matrix.matvec(&x);
        let max_error = ax
            .iter()
            .zip(&b)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 2e-5, "f32 residual {max_error:e}");
    }

    #[test]
    fn supernodal_large_tridiagonal_solve_regression() {
        let n = 1001;
        let dense = DenseMatrix::from_fn(n, n, |row, column| {
            if row == column {
                2.5_f64
            } else if row.abs_diff(column) == 1 {
                -0.5
            } else {
                0.0
            }
        });
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let symbolic = SparseLuFactorization::analyze_supernodal(&matrix).unwrap();
        let factor =
            SparseLuFactorization::factorize_supernodal_with_symbolic(&matrix, &symbolic, 0.1)
                .unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let x = factor.solve(&b).unwrap();
        let ax = matrix.matvec(&x);
        let max_error = ax
            .iter()
            .zip(&b)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 2e-11, "parallel solve residual {max_error:e}");
    }

    #[test]
    fn parallel_triangular_solve_handles_pivoted_independent_blocks() {
        // More than 1,000 unknowns selects the parallel triangular solves.
        // Independent 2x2 blocks create wide levels, while the small leading
        // diagonal in every block forces partial pivoting.
        let n = 1002;
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::with_capacity(2 * n);
        let mut values = Vec::with_capacity(2 * n);
        row_ptrs.push(0);
        for row in 0..n {
            let block = row / 2;
            let first = 2 * block;
            col_indices.extend([first, first + 1]);
            if row % 2 == 0 {
                values.extend([0.01_f64, 1.0]);
            } else {
                values.extend([2.0_f64, 3.0]);
            }
            row_ptrs.push(col_indices.len());
        }
        let matrix = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let expected: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).cos()).collect();
        let rhs = matrix.matvec(&expected);

        let factor = SparseLuFactorization::factorize(&matrix).unwrap();
        assert!(
            factor
                .row_perm
                .iter()
                .enumerate()
                .any(|(position, &original)| position != original),
            "test matrix must exercise partial pivoting"
        );
        assert!(factor.l_level_sets.iter().any(|level| level.len() > 1));
        assert!(factor.u_level_sets.iter().any(|level| level.len() > 1));

        let actual = factor.solve(&rhs).unwrap();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 2e-12, "parallel solve error {max_error:e}");
    }

    #[test]
    fn supernodal_reports_singular_matrix() {
        let n = 40;
        let mut dense = DenseMatrix::<f64>::eye(n);
        for column in 0..n {
            dense[[n - 1, column]] = dense[[n - 2, column]];
        }
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        assert!(matches!(
            SparseLuFactorization::factorize_supernodal(&matrix),
            Err(SparseLuError::SingularMatrix(_))
        ));
    }

    #[test]
    fn supernodal_large_convection_diffusion_matches_residual() {
        // 32x32 convection-diffusion: exercises multi-panel threshold
        // pivoting and the split compute/scatter apply path at scale,
        // gated by the solve residual. (A flat parallel fan-out of the
        // apply compute was tried and reverted here: contributors form a
        // left-looking dependence chain, so later contributors read rows
        // earlier ones update — this test caught residual ~2.9.)
        let nx = 32;
        let n = nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for y in 0..nx {
            for x in 0..nx {
                let i = y * nx + x;
                dense[[i, i]] = 4.01;
                if x > 0 {
                    dense[[i, i - 1]] = -1.2;
                }
                if x + 1 < nx {
                    dense[[i, i + 1]] = -0.8;
                }
                if y > 0 {
                    dense[[i, i - nx]] = -1.1;
                }
                if y + 1 < nx {
                    dense[[i, i + nx]] = -0.9;
                }
            }
        }
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let factor = SparseLuFactorization::factorize_supernodal(&matrix).unwrap();
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 0.5).collect();
        let x = factor.solve(&rhs).unwrap();
        let actual = dense.matvec(&x);
        let max_error = actual
            .iter()
            .zip(&rhs)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_error < 1e-10,
            "supernodal large-grid solve error {max_error:e}"
        );
    }

    /// Badly scaled matrix: base tridiagonal scaled by row and column
    /// diagonals spanning 1e-8..1e8.
    fn badly_scaled_matrix(n: usize) -> CsrMatrix<f64> {
        let dense = DenseMatrix::from_fn(n, n, |i, j| {
            let base = if i == j {
                4.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            };
            let row_scale = 10f64.powi(-8 + (16 * i) as i32 / (n - 1) as i32);
            let col_scale = 10f64.powi(-8 + (16 * ((j * 7 + 3) % n)) as i32 / (n - 1) as i32);
            base * row_scale * col_scale
        });
        CsrMatrix::from_dense(&dense, 0.0)
    }

    #[test]
    fn equilibration_improves_accuracy_on_badly_scaled_matrix() {
        let n = 24;
        let matrix = badly_scaled_matrix(n);
        let x_true: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).sin() + 1.5).collect();
        let b = matrix.matvec(&x_true);

        let relative_error = |factor: &SparseLuFactorization<f64>| {
            let x = factor.solve(&b).unwrap();
            x.iter()
                .zip(&x_true)
                .map(|(actual, expected)| (actual - expected).abs() / expected.abs())
                .fold(0.0_f64, f64::max)
        };

        let equilibrated =
            SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true).unwrap();
        let equilibrated_error = relative_error(&equilibrated);

        match SparseLuFactorization::factorize(&matrix) {
            Err(_) => {
                // Plain LU cannot even factor this matrix (the absolute pivot
                // floor rejects every pivot in the 1e-8-scaled columns) —
                // the strongest possible demonstration that equilibration is
                // necessary here.
            }
            Ok(plain) => {
                let plain_error = relative_error(&plain);
                assert!(
                    equilibrated_error < plain_error,
                    "equilibration should improve accuracy: {equilibrated_error:.3e} vs {plain_error:.3e}"
                );
            }
        }
        // f64 forward accuracy on a 1e16-conditioned matrix: the equilibrated
        // solve is backward-stable, so worst-component relative error is
        // small but not machine precision.
        assert!(
            equilibrated_error < 1e-2,
            "equilibrated solve should be accurate: {equilibrated_error:.3e}"
        );
    }

    #[test]
    fn equilibration_is_harmless_on_well_scaled_matrix() {
        // Robustness check: preprocessing must not degrade a well-scaled system.
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let factor = SparseLuFactorization::factorize_with_preprocessing(&a, 1.0, true).unwrap();
        let x = factor.solve(&b).unwrap();
        let ax = a.matvec(&x);
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|&bi| bi.powi(2)).sum::<f64>().sqrt();
        assert!(
            residual / b_norm < 1e-10,
            "equilibrated residual too large: {:.2e}",
            residual / b_norm
        );
    }

    #[test]
    fn supernodal_equilibration_solves_badly_scaled_matrix() {
        let n = 24;
        let matrix = badly_scaled_matrix(n);
        let x_true: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).sin() + 1.5).collect();
        let b = matrix.matvec(&x_true);

        let factor =
            SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true)
                .unwrap();
        assert_eq!(factor.backend(), SparseLuBackend::Supernodal);
        let x = factor.solve(&b).unwrap();
        let error = x
            .iter()
            .zip(&x_true)
            .map(|(actual, expected)| (actual - expected).abs() / expected.abs())
            .fold(0.0_f64, f64::max);
        // Same f64 forward-accuracy bound as the simplicial backend on a
        // 1e16-conditioned matrix.
        assert!(error < 1e-2, "supernodal equilibrated error {error:.3e}");
    }

    #[test]
    fn preconditioner_mode_relaxes_singularity_guard() {
        // f16 diagonal matrix with cond = 1e3. The strict pivot floor is
        // eps·n·‖A‖ ≈ 9.8e-4 · 16 ≈ 1.6e-2, far above the smallest pivot
        // (1e-3): the direct-solver contract rejects the factorization.
        // Preconditioner mode rejects only exact-zero pivots and succeeds,
        // which is what GMRES-IR needs (the outer FGMRES certifies the
        // result against the true residual).
        use half::f16;
        let n = 16;
        let row_ptrs: Vec<usize> = (0..=n).collect();
        let col_indices: Vec<usize> = (0..n).collect();
        let values: Vec<f16> = (0..n)
            .map(|i| f16::from_f32(10f32.powf(-3.0 * i as f32 / (n - 1) as f32)))
            .collect();
        let matrix = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);

        assert!(
            matches!(
                SparseLuFactorization::factorize(&matrix),
                Err(SparseLuError::SingularMatrix(_))
            ),
            "strict guard must reject cond=1e3 in f16"
        );

        let factor = SparseLuFactorization::factorize_for_preconditioner(&matrix)
            .expect("preconditioner mode must accept cond=1e3 in f16");
        // The factor is still usable: the diagonal system solves exactly.
        let b = vec![f16::ONE; n];
        let x = factor.solve(&b).unwrap();
        assert!((f32::from(x[0]) - 1.0).abs() < 1e-3);
        assert!((f32::from(x[n - 1]) - 1000.0).abs() < 1.0);
    }
}
