//! Simplicial sparse Cholesky factorization for SPD / HPD matrices.
//!
//! Computes `P·A·Pᵀ = L·Lᵀ` (declared symmetric) or `P·A·Pᵀ = L·Lᴴ`
//! (declared Hermitian) without pivoting, at roughly half the flops and
//! memory of the nonsymmetric sparse LU path.
//!
//! # Algorithm
//!
//! 1. **Ordering**: nested dissection ([`sonium_solvers_sparse::partition`]) by
//!    default — the right fill-reducing ordering for PDE matrices. Natural
//!    and COLAMD orderings are available as fallbacks.
//! 2. **Symbolic factorization**: elimination tree of the permuted pattern,
//!    then the nonzero pattern of `L` via row-subtree reach (one marked
//!    etree walk per column, `O(nnz(L))` work). Stored as CSC with the
//!    diagonal first in each column. Reusable across matrices that share
//!    a sparsity pattern.
//! 3. **Numeric factorization**: up-looking row-by-row Cholesky in the
//!    style of CSparse's `cs_chol`, generic over `f32`/`f64`/`Complex32`/
//!    `Complex64` via [`ComplexField`]. A non-positive (or non-real, or
//!    non-finite) pivot reports [`SparseCholeskyError::NotPositiveDefinite`].
//! 4. **Solves**: forward (`L`) and backward (`Lᴴ`) substitution with
//!    level-scheduled parallelism for large factors, mirroring the
//!    triangular-solve infrastructure in [`super::sparse_lu`].
//!
//! The factor is stored column-compressed: column `j` holds the diagonal
//! `L(j,j)` (real, positive) at `l_col_ptrs[j]`, followed by the strictly
//! lower entries `L(i,j)` for increasing `i > j`.

use super::sparse_cholesky_supernodal::{self, SupernodeMap};
use super::sparse_lu::{par_level_min_nnz, wave_nnz};
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, SolverError, SolverFailure, SolverStatus,
};
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sparse::partition::{DissectionNode, nested_dissection};

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rayon::prelude::*;
use thiserror::Error;

/// Errors from sparse Cholesky factorization and solves.
#[derive(Debug, Error)]
pub enum SparseCholeskyError {
    #[error("Matrix must be square ({rows} x {cols})")]
    NotSquare { rows: usize, cols: usize },
    #[error("Matrix is not positive definite (pivot {pivot})")]
    NotPositiveDefinite { pivot: usize },
    #[error("Right-hand side length {actual} does not match factorization size {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("solution length {actual} does not match factorization size {expected}")]
    SolutionDimensionMismatch { expected: usize, actual: usize },
    #[error("matrix entry at row {row}, column {col} is non-finite")]
    NonFiniteMatrix { row: usize, col: usize },
    #[error("right-hand side at index {0} is non-finite")]
    NonFiniteRhs(usize),
    #[error("non-finite value generated during {stage} at index {index}")]
    NumericalBreakdown { stage: &'static str, index: usize },
    #[error("CSR sparsity pattern does not match reusable symbolic analysis")]
    PatternMismatch,
    #[error("invalid sparse Cholesky checkpoint: {0}")]
    InvalidCheckpoint(String),
    #[error("invalid CSR structure: {0}")]
    InvalidCsrStructure(String),
    /// The requested row-major block size cannot be represented.
    #[error("multi-RHS size overflows for dimension {dimension} and {nrhs} RHS")]
    SizeOverflow { dimension: usize, nrhs: usize },
}

impl SparseCholeskyError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NotPositiveDefinite { .. } | Self::NumericalBreakdown { .. } => {
                SolverStatus::Breakdown
            }
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

impl SolverFailure for SparseCholeskyError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotSquare { .. }
            | Self::DimensionMismatch { .. }
            | Self::SolutionDimensionMismatch { .. }
            | Self::SizeOverflow { .. } => SolverErrorKind::DimensionMismatch,
            Self::NotPositiveDefinite { .. } | Self::NumericalBreakdown { .. } => {
                SolverErrorKind::Breakdown
            }
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        SparseCholeskyError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        SparseCholeskyError::accuracy_status(self)
    }
}

/// Versioned portable sparse-Cholesky factor checkpoint format.
/// Numeric values use little-endian f64 real/imaginary pairs; all index
/// metadata is encoded as checked u64 values so checkpoints are portable
/// across host index widths.
pub const SPARSE_CHOLESKY_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const SPARSE_CHOLESKY_CHECKPOINT_MAGIC: &[u8; 8] = b"SNCHCP01";

fn cholesky_checkpoint_invalid(reason: impl Into<String>) -> SparseCholeskyError {
    SparseCholeskyError::InvalidCheckpoint(reason.into())
}

struct CholeskyCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CholeskyCheckpointReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], SparseCholeskyError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| cholesky_checkpoint_invalid("checkpoint offset overflows"))?;
        if end > self.bytes.len() {
            return Err(cholesky_checkpoint_invalid("checkpoint is truncated"));
        }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_u8(&mut self) -> Result<u8, SparseCholeskyError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SparseCholeskyError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u64(&mut self) -> Result<u64, SparseCholeskyError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
            cholesky_checkpoint_invalid("invalid u64 field")
        })?))
    }

    fn read_f64(&mut self) -> Result<f64, SparseCholeskyError> {
        let bytes = self.take(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().map_err(|_| {
            cholesky_checkpoint_invalid("invalid f64 field")
        })?))
    }

    fn read_usize(&mut self, field: &str) -> Result<usize, SparseCholeskyError> {
        usize::try_from(self.read_u64()?)
            .map_err(|_| cholesky_checkpoint_invalid(format!("{field} does not fit usize")))
    }
}

fn checkpoint_push_usize(
    output: &mut Vec<u8>,
    value: usize,
    field: &str,
) -> Result<(), SparseCholeskyError> {
    output.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| cholesky_checkpoint_invalid(format!("{field} does not fit u64")))?
            .to_le_bytes(),
    );
    Ok(())
}

fn checkpoint_push_vec(
    output: &mut Vec<u8>,
    values: &[usize],
    field: &str,
) -> Result<(), SparseCholeskyError> {
    checkpoint_push_usize(output, values.len(), field)?;
    for &value in values {
        checkpoint_push_usize(output, value, field)?;
    }
    Ok(())
}

fn checkpoint_read_vec(
    reader: &mut CholeskyCheckpointReader<'_>,
    field: &str,
) -> Result<Vec<usize>, SparseCholeskyError> {
    let count = reader.read_usize(field)?;
    if count > reader.remaining() / 8 {
        return Err(cholesky_checkpoint_invalid(format!(
            "{field} length exceeds remaining checkpoint"
        )));
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.read_usize(field)?);
    }
    Ok(values)
}

fn validate_checkpoint_permutation(
    values: &[usize],
    n: usize,
    field: &str,
) -> Result<(), SparseCholeskyError> {
    if values.len() != n {
        return Err(cholesky_checkpoint_invalid(format!(
            "{field} length {} does not equal dimension {n}",
            values.len()
        )));
    }
    let mut seen = vec![false; n];
    for &value in values {
        if value >= n || std::mem::replace(&mut seen[value], true) {
            return Err(cholesky_checkpoint_invalid(format!(
                "{field} is not a permutation"
            )));
        }
    }
    Ok(())
}

fn validate_checkpoint_ptrs(
    values: &[usize],
    n: usize,
    entry_count: usize,
    field: &str,
) -> Result<(), SparseCholeskyError> {
    let expected = n
        .checked_add(1)
        .ok_or_else(|| cholesky_checkpoint_invalid(format!("{field} length overflows")))?;
    if values.len() != expected || values.first().copied() != Some(0) {
        return Err(cholesky_checkpoint_invalid(format!(
            "{field} pointers have invalid length or start"
        )));
    }
    if values.windows(2).any(|window| window[0] > window[1])
        || values.last().copied() != Some(entry_count)
    {
        return Err(cholesky_checkpoint_invalid(format!(
            "{field} pointers are not monotone or terminal value mismatches"
        )));
    }
    Ok(())
}

fn validate_checkpoint_rows(
    values: &[usize],
    n: usize,
    field: &str,
) -> Result<(), SparseCholeskyError> {
    if values.iter().any(|&row| row >= n) {
        return Err(cholesky_checkpoint_invalid(format!(
            "{field} contains an out-of-range row"
        )));
    }
    Ok(())
}

fn cholesky_scalar_is_complex<T: ComplexField>() -> u8 {
    let one = T::Real::from_f64(1.0).unwrap_or_else(T::Real::zero);
    u8::from(T::from_re_im(T::Real::zero(), one).norm() > T::Real::zero())
}

fn validate_checkpoint_symbolic<T: ComplexField>(
    symbolic: &SparseCholeskySymbolic,
    l_values: &[T],
) -> Result<(), SparseCholeskyError> {
    let n = symbolic.n;
    validate_checkpoint_permutation(&symbolic.perm, n, "permutation")?;
    validate_checkpoint_permutation(&symbolic.perm_inv, n, "inverse permutation")?;
    if symbolic
        .perm
        .iter()
        .enumerate()
        .any(|(position, &original)| symbolic.perm_inv[original] != position)
    {
        return Err(cholesky_checkpoint_invalid(
            "permutation and inverse permutation disagree",
        ));
    }
    if symbolic.parent.len() != n {
        return Err(cholesky_checkpoint_invalid("parent length mismatch"));
    }
    if symbolic
        .parent
        .iter()
        .enumerate()
        .any(|(child, &parent)| parent != usize::MAX && (parent >= n || parent <= child))
    {
        return Err(cholesky_checkpoint_invalid(
            "parent contains an invalid elimination-tree edge",
        ));
    }
    validate_checkpoint_ptrs(&symbolic.l_col_ptrs, n, symbolic.l_row_indices.len(), "L")?;
    validate_checkpoint_rows(&symbolic.l_row_indices, n, "L row indices")?;
    for column in 0..n {
        let rows =
            &symbolic.l_row_indices[symbolic.l_col_ptrs[column]..symbolic.l_col_ptrs[column + 1]];
        if rows.first().copied() != Some(column)
            || rows.windows(2).any(|window| window[0] >= window[1])
            || rows.iter().skip(1).any(|&row| row <= column)
        {
            return Err(cholesky_checkpoint_invalid(
                "L pattern is not diagonal-first and strictly lower",
            ));
        }
    }
    if symbolic.a_col_ptrs.len() != n.saturating_add(1) {
        return Err(cholesky_checkpoint_invalid("A pointer length mismatch"));
    }
    validate_checkpoint_ptrs(&symbolic.a_col_ptrs, n, symbolic.a_row_indices.len(), "A")?;
    validate_checkpoint_rows(&symbolic.a_row_indices, n, "A row indices")?;
    for column in 0..n {
        let rows =
            &symbolic.a_row_indices[symbolic.a_col_ptrs[column]..symbolic.a_col_ptrs[column + 1]];
        if rows.windows(2).any(|window| window[0] >= window[1])
            || rows.iter().any(|&row| row > column)
        {
            return Err(cholesky_checkpoint_invalid(
                "A pattern is not sorted upper-triangular",
            ));
        }
    }
    if l_values.len() != symbolic.l_row_indices.len()
        || l_values.iter().any(|value| !value.norm().is_finite())
    {
        return Err(cholesky_checkpoint_invalid(
            "factor values have invalid shape or non-finite entries",
        ));
    }
    for column in 0..n {
        let diagonal = l_values[symbolic.l_col_ptrs[column]];
        if diagonal.re() <= T::Real::zero() || diagonal.im() != T::Real::zero() {
            return Err(cholesky_checkpoint_invalid(
                "factor diagonal is not finite positive real",
            ));
        }
    }
    Ok(())
}

fn validate_csr_structure<T: ComplexField>(
    matrix: &CsrMatrix<T>,
) -> Result<(), SparseCholeskyError> {
    let expected_row_ptr_len = matrix.num_rows.checked_add(1).ok_or_else(|| {
        SparseCholeskyError::InvalidCsrStructure(
            "row count overflows the row-pointer length".into(),
        )
    })?;
    if matrix.row_ptrs.len() != expected_row_ptr_len {
        return Err(SparseCholeskyError::InvalidCsrStructure(format!(
            "row-pointer length {} does not equal {}",
            matrix.row_ptrs.len(),
            expected_row_ptr_len
        )));
    }
    if matrix.col_indices.len() != matrix.values.len() {
        return Err(SparseCholeskyError::InvalidCsrStructure(format!(
            "column-index length {} does not equal value length {}",
            matrix.col_indices.len(),
            matrix.values.len()
        )));
    }
    if matrix.row_ptrs.first().copied() != Some(0) {
        return Err(SparseCholeskyError::InvalidCsrStructure(
            "row-pointer array must start at zero".into(),
        ));
    }
    if matrix
        .row_ptrs
        .windows(2)
        .any(|window| window[0] > window[1])
    {
        return Err(SparseCholeskyError::InvalidCsrStructure(
            "row pointers must be monotonically nondecreasing".into(),
        ));
    }
    let nnz = matrix.values.len();
    if matrix.row_ptrs.last().copied() != Some(nnz) {
        return Err(SparseCholeskyError::InvalidCsrStructure(format!(
            "terminal row pointer {:?} does not equal nnz {nnz}",
            matrix.row_ptrs.last()
        )));
    }
    for row in 0..matrix.num_rows {
        let start = matrix.row_ptrs[row];
        let end = matrix.row_ptrs[row + 1];
        let mut previous_column = None;
        for index in start..end {
            let column = matrix.col_indices[index];
            if column >= matrix.num_cols {
                return Err(SparseCholeskyError::InvalidCsrStructure(format!(
                    "entry {index} in row {row} has column {column}, outside {} columns",
                    matrix.num_cols
                )));
            }
            if let Some(previous) = previous_column
                && column <= previous
            {
                return Err(SparseCholeskyError::InvalidCsrStructure(format!(
                    "row {row} columns are not strictly increasing at entry {index} ({previous} then {column})"
                )));
            }
            if !matrix.values[index].norm().is_finite() {
                return Err(SparseCholeskyError::NonFiniteMatrix { row, col: column });
            }
            previous_column = Some(column);
        }
    }
    Ok(())
}

/// Fill-reducing ordering for the simplicial Cholesky symbolic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CholeskyOrdering {
    /// Nested dissection (default; best for PDE matrices).
    #[default]
    NestedDissection,
    /// Identity permutation.
    Natural,
    /// COLAMD column ordering (fallback; tuned for nonsymmetric LU).
    Colamd,
}

/// Cutoff below which the serial triangular-solve path is used (same
/// heuristic as the sparse LU solves).
const PARALLEL_SOLVE_MIN_N: usize = 1000;

/// Default nested-dissection leaf size.
const ND_MIN_BLOCK_SIZE: usize = 32;

/// Scalar-independent symbolic analysis for simplicial sparse Cholesky.
///
/// Holds the fill-reducing permutation, the elimination tree, and the
/// nonzero pattern of `L`, all derived from the sparsity pattern alone.
/// Reusable for any matrix with the same pattern (see
/// [`SparseCholeskyFactorization::factorize_with_symbolic`]).
#[derive(Debug, Clone)]
pub struct SparseCholeskySymbolic {
    n: usize,
    /// `perm[k]` is the original row/column eliminated at step `k`.
    perm: Vec<usize>,
    /// `perm_inv[g]` is the elimination step of original index `g`.
    perm_inv: Vec<usize>,
    /// Elimination tree of the permuted pattern (`usize::MAX` at roots).
    parent: Vec<usize>,
    /// CSC column pointers of `L` (length `n + 1`); diagonal first per column.
    l_col_ptrs: Vec<usize>,
    /// CSC row indices of `L`.
    l_row_indices: Vec<usize>,
    /// Pattern of `triu(P·A·Pᵀ)`, CSC, kept to validate symbolic reuse.
    a_col_ptrs: Vec<usize>,
    /// Row indices of the `triu(P·A·Pᵀ)` pattern.
    a_row_indices: Vec<usize>,
    /// Level sets for parallel forward substitution (`L` solve). Columns in
    /// the same level scatter into disjoint rows.
    l_level_sets: Vec<Vec<usize>>,
    /// Level sets for parallel backward substitution (`Lᴴ` solve). Columns in
    /// the same level only read entries solved in earlier levels.
    lh_level_sets: Vec<Vec<usize>>,
    /// Lazily computed relaxed-supernode partition (supernodal numeric path).
    supernode_cache: std::sync::OnceLock<SupernodeMap>,
}

impl SparseCholeskySymbolic {
    /// Matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// `perm[k]` is the original row/column eliminated at step `k`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Number of nonzeros in the `L` factor, including the diagonal.
    pub fn factor_nonzeros(&self) -> usize {
        self.l_row_indices.len()
    }

    /// Analyze the sparsity pattern of a square matrix.
    pub fn analyze<T: ComplexField>(
        matrix: &CsrMatrix<T>,
        ordering: CholeskyOrdering,
    ) -> Result<Self, SparseCholeskyError> {
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(SparseCholeskyError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        validate_csr_structure(matrix)?;

        let perm = compute_permutation(matrix, ordering);
        let mut perm_inv = vec![0usize; n];
        for (k, &g) in perm.iter().enumerate() {
            perm_inv[g] = k;
        }

        // triu(P·A·Pᵀ) in CSC; values are assembled but only the pattern
        // (and the etree derived from it) is kept.
        let upper = assemble_upper_triangle(matrix, &perm_inv, false);
        let parent = elimination_tree(n, &upper.col_ptrs, &upper.row_indices);
        let (l_col_ptrs, l_row_indices) = symbolic_pattern(n, &upper, &parent);
        let l_level_sets = compute_l_level_sets(&l_col_ptrs, &l_row_indices, n);
        let lh_level_sets = compute_lh_level_sets(&l_col_ptrs, &l_row_indices, n);

        Ok(Self {
            n,
            perm,
            perm_inv,
            parent,
            l_col_ptrs,
            l_row_indices,
            a_col_ptrs: upper.col_ptrs,
            a_row_indices: upper.row_indices,
            l_level_sets,
            lh_level_sets,
            supernode_cache: std::sync::OnceLock::new(),
        })
    }

    /// CSC column pointers of the `L` factor pattern.
    pub(crate) fn factor_col_ptrs(&self) -> &[usize] {
        &self.l_col_ptrs
    }

    /// CSC row indices of the `L` factor pattern.
    pub(crate) fn factor_row_indices(&self) -> &[usize] {
        &self.l_row_indices
    }

    /// Relaxed-supernode partition of the factor pattern, computed on first
    /// use and cached. Used by the supernodal numeric factorization and by
    /// supernode statistics queries.
    pub fn supernode_map(&self) -> SupernodeMap {
        self.supernode_map_ref().clone()
    }

    /// Borrow the cached supernode partition, computing it on first use.
    pub(crate) fn supernode_map_ref(&self) -> &SupernodeMap {
        self.supernode_cache
            .get_or_init(|| SupernodeMap::analyze(self))
    }
}

/// Numeric simplicial sparse Cholesky factorization.
///
/// Owns its symbolic analysis so a factorization is self-contained; clone
/// the [`SparseCholeskySymbolic`] beforehand if it should be reused for
/// another matrix with the same pattern.
#[derive(Debug, Clone)]
pub struct SparseCholeskyFactorization<T: ComplexField> {
    symbolic: SparseCholeskySymbolic,
    /// CSC values of `L`, layout matching `symbolic.l_col_ptrs/l_row_indices`.
    l_values: Vec<T>,
    /// Hermitian (`L·Lᴴ`) vs symmetric (`L·Lᵀ`) form; only differs for
    /// complex scalar types.
    hermitian: bool,
}

impl<T: ComplexField> SparseCholeskyFactorization<T> {
    /// Factorize a declared symmetric positive-definite matrix (`A = L·Lᵀ`)
    /// with nested-dissection ordering.
    pub fn factorize(matrix: &CsrMatrix<T>) -> Result<Self, SparseCholeskyError> {
        Self::factorize_with_ordering(matrix, CholeskyOrdering::NestedDissection, false)
    }

    /// Factorize a declared Hermitian positive-definite matrix (`A = L·Lᴴ`)
    /// with nested-dissection ordering.
    pub fn factorize_hermitian(matrix: &CsrMatrix<T>) -> Result<Self, SparseCholeskyError> {
        Self::factorize_with_ordering(matrix, CholeskyOrdering::NestedDissection, true)
    }

    /// Factorize with an explicit ordering. `hermitian` selects the `L·Lᴴ`
    /// form (relevant only for complex scalar types).
    pub fn factorize_with_ordering(
        matrix: &CsrMatrix<T>,
        ordering: CholeskyOrdering,
        hermitian: bool,
    ) -> Result<Self, SparseCholeskyError> {
        let symbolic = SparseCholeskySymbolic::analyze(matrix, ordering)?;
        Self::factorize_with_symbolic(matrix, &symbolic, hermitian)
    }

    /// Supernodal (BLAS-3) factorization of a declared symmetric
    /// positive-definite matrix with nested-dissection ordering.
    ///
    /// Same factor, solves, and symbolic-reuse contract as [`Self::factorize`];
    /// only the numeric path differs (relaxed supernodes, dense rank-k
    /// updates). See [`super::sparse_cholesky_supernodal`].
    pub fn factorize_supernodal(matrix: &CsrMatrix<T>) -> Result<Self, SparseCholeskyError> {
        Self::factorize_supernodal_with_ordering(matrix, CholeskyOrdering::NestedDissection, false)
    }

    /// Supernodal factorization of a declared Hermitian positive-definite
    /// matrix (`A = L·Lᴴ`) with nested-dissection ordering.
    pub fn factorize_supernodal_hermitian(
        matrix: &CsrMatrix<T>,
    ) -> Result<Self, SparseCholeskyError> {
        Self::factorize_supernodal_with_ordering(matrix, CholeskyOrdering::NestedDissection, true)
    }

    /// Supernodal factorization with an explicit ordering.
    pub fn factorize_supernodal_with_ordering(
        matrix: &CsrMatrix<T>,
        ordering: CholeskyOrdering,
        hermitian: bool,
    ) -> Result<Self, SparseCholeskyError> {
        let symbolic = SparseCholeskySymbolic::analyze(matrix, ordering)?;
        Self::factorize_supernodal_with_symbolic(matrix, &symbolic, hermitian)
    }

    /// Supernodal factorization reusing a prior symbolic analysis. The matrix
    /// must have exactly the same sparsity pattern as the analyzed one —
    /// the same contract as [`Self::factorize_with_symbolic`].
    pub fn factorize_supernodal_with_symbolic(
        matrix: &CsrMatrix<T>,
        symbolic: &SparseCholeskySymbolic,
        hermitian: bool,
    ) -> Result<Self, SparseCholeskyError> {
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(SparseCholeskyError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        if n != symbolic.n {
            return Err(SparseCholeskyError::PatternMismatch);
        }
        validate_csr_structure(matrix)?;
        let upper = assemble_upper_triangle(matrix, &symbolic.perm_inv, hermitian);
        if upper.col_ptrs != symbolic.a_col_ptrs || upper.row_indices != symbolic.a_row_indices {
            return Err(SparseCholeskyError::PatternMismatch);
        }

        let map = symbolic.supernode_map_ref();
        let l_values =
            sparse_cholesky_supernodal::numeric_factorize(&upper, symbolic, map, hermitian)?;
        if let Some(index) = first_non_finite(&l_values) {
            return Err(SparseCholeskyError::NumericalBreakdown {
                stage: "supernodal factorization",
                index,
            });
        }
        Ok(Self {
            symbolic: symbolic.clone(),
            l_values,
            hermitian,
        })
    }

    /// Factorize reusing a prior symbolic analysis. The matrix must have
    /// exactly the same sparsity pattern as the analyzed one.
    pub fn factorize_with_symbolic(
        matrix: &CsrMatrix<T>,
        symbolic: &SparseCholeskySymbolic,
        hermitian: bool,
    ) -> Result<Self, SparseCholeskyError> {
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(SparseCholeskyError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        if n != symbolic.n {
            return Err(SparseCholeskyError::PatternMismatch);
        }
        validate_csr_structure(matrix)?;
        let upper = assemble_upper_triangle(matrix, &symbolic.perm_inv, hermitian);
        if upper.col_ptrs != symbolic.a_col_ptrs || upper.row_indices != symbolic.a_row_indices {
            return Err(SparseCholeskyError::PatternMismatch);
        }

        let l_values = numeric_factorize(&upper, symbolic, hermitian)?;
        if let Some(index) = first_non_finite(&l_values) {
            return Err(SparseCholeskyError::NumericalBreakdown {
                stage: "simplicial factorization",
                index,
            });
        }
        Ok(Self {
            symbolic: symbolic.clone(),
            l_values,
            hermitian,
        })
    }

    /// Matrix dimension.
    pub fn dimension(&self) -> usize {
        self.symbolic.n
    }

    /// Whether the factor represents the Hermitian form `L·Lᴴ`.
    pub fn is_hermitian(&self) -> bool {
        self.hermitian
    }

    /// The symbolic analysis (permutation, etree, pattern of `L`).
    pub fn symbolic(&self) -> &SparseCholeskySymbolic {
        &self.symbolic
    }

    /// Number of explicitly stored factor entries, including the diagonal.
    pub fn factor_nonzeros(&self) -> usize {
        self.l_values.len()
    }

    /// Serialize the numeric factor and symbolic metadata to a portable,
    /// versioned checkpoint. The supernode cache is intentionally rebuilt
    /// lazily after restore because it is derived from the symbolic pattern.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SparseCholeskyError> {
        validate_checkpoint_symbolic(&self.symbolic, &self.l_values)?;
        let mut output = Vec::new();
        output.extend_from_slice(SPARSE_CHOLESKY_CHECKPOINT_MAGIC);
        output.extend_from_slice(&SPARSE_CHOLESKY_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(u8::try_from(std::mem::size_of::<T::Real>()).unwrap_or(u8::MAX));
        output.push(cholesky_scalar_is_complex::<T>());
        output.push(u8::from(self.hermitian));
        output.push(0);
        checkpoint_push_usize(&mut output, self.symbolic.n, "dimension")?;
        checkpoint_push_vec(&mut output, &self.symbolic.perm, "permutation")?;
        checkpoint_push_vec(&mut output, &self.symbolic.perm_inv, "inverse permutation")?;
        checkpoint_push_vec(&mut output, &self.symbolic.parent, "parent")?;
        checkpoint_push_vec(&mut output, &self.symbolic.l_col_ptrs, "L pointers")?;
        checkpoint_push_vec(&mut output, &self.symbolic.l_row_indices, "L rows")?;
        checkpoint_push_vec(&mut output, &self.symbolic.a_col_ptrs, "A pointers")?;
        checkpoint_push_vec(&mut output, &self.symbolic.a_row_indices, "A rows")?;
        checkpoint_push_usize(&mut output, self.l_values.len(), "factor values")?;
        for value in &self.l_values {
            let re = value
                .re()
                .to_f64()
                .ok_or_else(|| cholesky_checkpoint_invalid("real factor value cannot convert"))?;
            let im = value.im().to_f64().ok_or_else(|| {
                cholesky_checkpoint_invalid("imaginary factor value cannot convert")
            })?;
            if !re.is_finite() || !im.is_finite() {
                return Err(cholesky_checkpoint_invalid(
                    "factor values contain NaN or infinity",
                ));
            }
            output.extend_from_slice(&re.to_le_bytes());
            output.extend_from_slice(&im.to_le_bytes());
        }
        Ok(output)
    }

    /// Restore a sparse Cholesky factor checkpoint with strict metadata,
    /// scalar-family, shape, finite-value, and trailing-byte validation.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SparseCholeskyError> {
        let mut reader = CholeskyCheckpointReader { bytes, offset: 0 };
        if reader.take(SPARSE_CHOLESKY_CHECKPOINT_MAGIC.len())? != SPARSE_CHOLESKY_CHECKPOINT_MAGIC
        {
            return Err(cholesky_checkpoint_invalid("checkpoint magic mismatch"));
        }
        if reader.read_u16()? != SPARSE_CHOLESKY_CHECKPOINT_FORMAT_VERSION {
            return Err(cholesky_checkpoint_invalid("checkpoint version mismatch"));
        }
        let expected_width = u8::try_from(std::mem::size_of::<T::Real>()).unwrap_or(u8::MAX);
        if reader.read_u8()? != expected_width {
            return Err(cholesky_checkpoint_invalid(
                "checkpoint scalar width mismatch",
            ));
        }
        if reader.read_u8()? != cholesky_scalar_is_complex::<T>() {
            return Err(cholesky_checkpoint_invalid(
                "checkpoint scalar family mismatch",
            ));
        }
        let hermitian = match reader.read_u8()? {
            0 => false,
            1 => true,
            _ => return Err(cholesky_checkpoint_invalid("invalid Hermitian flag")),
        };
        if reader.read_u8()? != 0 {
            return Err(cholesky_checkpoint_invalid(
                "checkpoint reserved field non-zero",
            ));
        }
        let n = reader.read_usize("dimension")?;
        let perm = checkpoint_read_vec(&mut reader, "permutation")?;
        let perm_inv = checkpoint_read_vec(&mut reader, "inverse permutation")?;
        let parent = checkpoint_read_vec(&mut reader, "parent")?;
        let l_col_ptrs = checkpoint_read_vec(&mut reader, "L pointers")?;
        let l_row_indices = checkpoint_read_vec(&mut reader, "L rows")?;
        let a_col_ptrs = checkpoint_read_vec(&mut reader, "A pointers")?;
        let a_row_indices = checkpoint_read_vec(&mut reader, "A rows")?;
        let value_count = reader.read_usize("factor values")?;
        if value_count > reader.remaining() / 16 {
            return Err(cholesky_checkpoint_invalid(
                "factor value count exceeds remaining checkpoint",
            ));
        }
        let mut l_values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            let re = reader.read_f64()?;
            let im = reader.read_f64()?;
            if !re.is_finite() || !im.is_finite() {
                return Err(cholesky_checkpoint_invalid(
                    "factor values contain NaN or infinity",
                ));
            }
            l_values.push(T::from_re_im(
                T::Real::from_f64(re)
                    .ok_or_else(|| cholesky_checkpoint_invalid("real value conversion failed"))?,
                T::Real::from_f64(im).ok_or_else(|| {
                    cholesky_checkpoint_invalid("imaginary value conversion failed")
                })?,
            ));
        }
        if reader.remaining() != 0 {
            return Err(cholesky_checkpoint_invalid(
                "checkpoint contains trailing bytes",
            ));
        }
        // Validate all serialized pointer and row metadata before deriving
        // level sets; otherwise malformed ranges could panic during restore.
        let mut symbolic = SparseCholeskySymbolic {
            n,
            perm,
            perm_inv,
            parent,
            l_col_ptrs,
            l_row_indices,
            a_col_ptrs,
            a_row_indices,
            l_level_sets: Vec::new(),
            lh_level_sets: Vec::new(),
            supernode_cache: std::sync::OnceLock::new(),
        };
        validate_checkpoint_symbolic(&symbolic, &l_values)?;
        symbolic.l_level_sets =
            compute_l_level_sets(&symbolic.l_col_ptrs, &symbolic.l_row_indices, n);
        symbolic.lh_level_sets =
            compute_lh_level_sets(&symbolic.l_col_ptrs, &symbolic.l_row_indices, n);
        Ok(Self {
            symbolic,
            l_values,
            hermitian,
        })
    }

    /// Solve `A·x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SparseCholeskyError> {
        let n = self.dimension();
        let mut x = vec![T::zero(); n];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }

    /// Allocation-free solve into caller-owned storage.
    pub fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), SparseCholeskyError> {
        let n = self.dimension();
        if b.len() != n {
            return Err(SparseCholeskyError::DimensionMismatch {
                expected: n,
                actual: b.len(),
            });
        }
        if x.len() != n {
            return Err(SparseCholeskyError::SolutionDimensionMismatch {
                expected: n,
                actual: x.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(SparseCholeskyError::NonFiniteRhs(index));
        }
        if n == 0 {
            return Ok(());
        }
        self.solve_into_prevalidated(b, x)
    }

    /// Solve core behind [`Self::solve_into`] and [`Self::solve_many_into`]:
    /// `b`/`x` have length `n` and `b` is finite.
    ///
    /// No stored-factor finite rescan here (validate-once): the fields are
    /// private with no `&mut` API so factors are immutable after
    /// construction, and every constructor establishes finiteness — both
    /// numeric paths scan once at completion and checkpoint restore
    /// validates in `validate_checkpoint_symbolic`. Intermediates computed
    /// below are still scanned per solve.
    fn solve_into_prevalidated(&self, b: &[T], x: &mut [T]) -> Result<(), SparseCholeskyError> {
        let n = self.dimension();
        let sym = &self.symbolic;

        // y = P·b (elimination order).
        let mut y = vec![T::zero(); n];
        for (k, &g) in sym.perm.iter().enumerate() {
            y[k] = b[g];
        }

        self.forward_solve(&mut y);
        if let Some(index) = first_non_finite(&y) {
            return Err(SparseCholeskyError::NumericalBreakdown {
                stage: "forward substitution",
                index,
            });
        }
        self.backward_solve(&mut y);
        if let Some(index) = first_non_finite(&y) {
            return Err(SparseCholeskyError::NumericalBreakdown {
                stage: "backward substitution",
                index,
            });
        }

        // x = Pᵀ·y (original order).
        for (k, &g) in sym.perm.iter().enumerate() {
            x[g] = y[k];
        }
        Ok(())
    }

    /// Solve an `n × nrhs` row-major block without mutating the destination
    /// until every right-hand side has passed validation and solved.
    pub fn solve_many_into(
        &self,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<(), SparseCholeskyError> {
        let n = self.dimension();
        let expected = n
            .checked_mul(nrhs)
            .ok_or(SparseCholeskyError::SizeOverflow { dimension: n, nrhs })?;
        if b.len() != expected || x.len() != expected {
            return Err(SparseCholeskyError::DimensionMismatch {
                expected,
                actual: if b.len() != expected {
                    b.len()
                } else {
                    x.len()
                },
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(SparseCholeskyError::NonFiniteRhs(index));
        }

        // The block is validated once above; each column below is copied
        // from it, so per-column revalidation would be pure overhead (the
        // core still scans its own intermediates per column).
        let mut solutions = vec![T::zero(); expected];
        let mut rhs = vec![T::zero(); n];
        let mut solution = vec![T::zero(); n];
        for column in 0..nrhs {
            for row in 0..n {
                rhs[row] = b[row * nrhs + column];
            }
            self.solve_into_prevalidated(&rhs, &mut solution)?;
            for row in 0..n {
                solutions[row * nrhs + column] = solution[row];
            }
        }
        x.copy_from_slice(&solutions);
        Ok(())
    }

    /// Forward substitution `L·y = y`, column-scatter form.
    fn forward_solve(&self, y: &mut [T]) {
        let n = self.dimension();
        let sym = &self.symbolic;
        let ptrs = &sym.l_col_ptrs;
        let rows = &sym.l_row_indices;
        let values = &self.l_values;

        // Thin waves run serially inline: a rayon dispatch costs more than
        // their flops (shared `SONIUM_SPARSE_PAR_MIN_NNZ` gate; serial-in-
        // wave is bitwise identical to parallel-in-wave, so no numerical
        // effect — see the sparse-LU rationale).
        let par_min_nnz = par_level_min_nnz();
        if n >= PARALLEL_SOLVE_MIN_N && sym.l_level_sets.len() > 1 {
            let y_raw = y.as_mut_ptr();
            for level in &sym.l_level_sets {
                if level.len() > 1 && wave_nnz(level, ptrs) >= par_min_nnz {
                    // SAFETY: columns in the same level write to disjoint
                    // rows (guaranteed by level-set construction); y[j] is
                    // only read after every earlier-level update landed.
                    unsafe { parallel_forward_level(level, y_raw, ptrs, rows, values) };
                } else {
                    for &j in level {
                        scatter_forward_column(j, y, ptrs, rows, values);
                    }
                }
            }
        } else {
            for j in 0..n {
                scatter_forward_column(j, y, ptrs, rows, values);
            }
        }
    }

    /// Backward substitution `Lᴴ·x = x` (or `Lᵀ·x = x`), column-gather form.
    fn backward_solve(&self, x: &mut [T]) {
        let n = self.dimension();
        let sym = &self.symbolic;
        let ptrs = &sym.l_col_ptrs;
        let rows = &sym.l_row_indices;
        let values = &self.l_values;
        let hermitian = self.hermitian;

        let par_min_nnz = par_level_min_nnz();
        if n >= PARALLEL_SOLVE_MIN_N && sym.lh_level_sets.len() > 1 {
            let x_raw = x.as_mut_ptr();
            for level in &sym.lh_level_sets {
                if level.len() > 1 && wave_nnz(level, ptrs) >= par_min_nnz {
                    // SAFETY: columns in the same level write only their own
                    // x[j] and read entries computed in earlier levels.
                    unsafe { parallel_backward_level(level, x_raw, ptrs, rows, values, hermitian) };
                } else {
                    for &j in level {
                        gather_backward_column(j, x, ptrs, rows, values, hermitian);
                    }
                }
            }
        } else {
            for j in (0..n).rev() {
                gather_backward_column(j, x, ptrs, rows, values, hermitian);
            }
        }
    }
}

/// Implement `Preconditioner<T>` so the sparse Cholesky can serve as a
/// direct preconditioner or coarse-grid solver, mirroring the sparse LU.
impl<T: ComplexField> sonium_solvers_core::traits::Preconditioner<T>
    for SparseCholeskyFactorization<T>
{
    fn apply(&self, r: &[T]) -> Vec<T> {
        self.solve(r)
            .expect("preconditioner input must match a valid sparse Cholesky factorization")
    }

    fn transpose_capability(&self) -> AdjointCapability {
        // A symmetric factor (`L·Lᵀ`) has an exact transpose action.  A
        // complex Hermitian factor does not generally solve `Aᵀ`, while real
        // scalars are simultaneously symmetric and Hermitian.
        if !self.is_hermitian() || cholesky_scalar_is_complex::<T>() == 0 {
            AdjointCapability::Exact
        } else {
            AdjointCapability::Unsupported
        }
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        assert_eq!(
            self.transpose_capability(),
            AdjointCapability::Exact,
            "transpose action requires a symmetric sparse Cholesky factor"
        );
        self.solve(r)
            .expect("preconditioner input must match a valid sparse Cholesky factorization")
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        if self.is_hermitian() || cholesky_scalar_is_complex::<T>() == 0 {
            AdjointCapability::Exact
        } else {
            AdjointCapability::Unsupported
        }
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        assert!(
            self.hermitian_capability() == AdjointCapability::Exact,
            "Hermitian action requires a Hermitian or real symmetric Cholesky factor"
        );
        self.solve(r)
            .expect("preconditioner input match sparse Cholesky factorization")
    }

    fn try_apply_hermitian(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if self.hermitian_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "preconditioner Hermitian transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        self.solve(r).map_err(|error| {
            SolverError::from_failure("sparse Cholesky Hermitian transpose", &error)
        })
    }
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

/// Compute the elimination permutation for the requested ordering.
fn compute_permutation<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    ordering: CholeskyOrdering,
) -> Vec<usize> {
    let n = matrix.num_rows;
    match ordering {
        CholeskyOrdering::Natural => (0..n).collect(),
        CholeskyOrdering::Colamd => {
            // COLAMD returns `col_perm[k]` = original column of step `k`.
            super::sparse_lu::colamd(matrix)
        }
        CholeskyOrdering::NestedDissection => {
            if n <= ND_MIN_BLOCK_SIZE {
                return (0..n).collect();
            }
            let adjacency = symmetrized_adjacency(matrix);
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
            // `nodes` is post-order by construction: concatenate each node's
            // DOFs to get the elimination permutation.
            let mut perm = Vec::with_capacity(n);
            for node in &nodes {
                perm.extend_from_slice(&node.global_indices);
            }
            debug_assert_eq!(perm.len(), n, "nested dissection must cover all DOFs");
            perm
        }
    }
}

/// Symmetric adjacency of the pattern (union of row and column neighbours,
/// self-loops dropped). Unlike `partition::build_adjacency`, this tolerates
/// matrices that store only one triangle.
fn symmetrized_adjacency<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = matrix.num_rows;
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            if j != i {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

// ---------------------------------------------------------------------------
// Permuted upper-triangle assembly
// ---------------------------------------------------------------------------

/// CSC storage of `triu(P·A·Pᵀ)` with values.
pub(crate) struct UpperTriangle<T: ComplexField> {
    pub(crate) col_ptrs: Vec<usize>,
    pub(crate) row_indices: Vec<usize>,
    pub(crate) values: Vec<T>,
}

/// Assemble the upper triangle of the permuted matrix in CSC, rows sorted
/// within each column.
///
/// An original entry `(r, c)` maps to permuted position `(i, j)`; the upper
/// position is `(min(i,j), max(i,j))`. Entries below the diagonal supply
/// their (conjugate, for Hermitian) transpose value, so matrices storing
/// only one triangle are handled. When both triangles are stored the last
/// writer wins — for a consistent symmetric/Hermitian input the values are
/// identical anyway.
pub(crate) fn assemble_upper_triangle<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    perm_inv: &[usize],
    hermitian: bool,
) -> UpperTriangle<T> {
    let n = matrix.num_rows;

    // Pass 1: count deduplicated entries per column. `stamp[row]` records
    // the column (+1) for which `row` was last counted.
    let mut counts = vec![0usize; n];
    let mut stamp = vec![0usize; n];
    for r in 0..n {
        for idx in matrix.row_ptrs[r]..matrix.row_ptrs[r + 1] {
            let i = perm_inv[r];
            let j = perm_inv[matrix.col_indices[idx]];
            let (row, col) = if i <= j { (i, j) } else { (j, i) };
            if stamp[row] != col + 1 {
                stamp[row] = col + 1;
                counts[col] += 1;
            }
        }
    }

    let mut col_ptrs = vec![0usize; n + 1];
    for j in 0..n {
        col_ptrs[j + 1] = col_ptrs[j] + counts[j];
    }

    // Pass 2: fill rows and values (last write wins on duplicates).
    let mut row_indices = vec![0usize; col_ptrs[n]];
    let mut values = vec![T::zero(); col_ptrs[n]];
    let mut cursors = col_ptrs[..n].to_vec();
    let mut slot = vec![usize::MAX; n];
    stamp.fill(0);
    for r in 0..n {
        for idx in matrix.row_ptrs[r]..matrix.row_ptrs[r + 1] {
            let i = perm_inv[r];
            let j = perm_inv[matrix.col_indices[idx]];
            let value = matrix.values[idx];
            let (row, col, value) = if i <= j {
                (i, j, value)
            } else if hermitian {
                (j, i, value.conj())
            } else {
                (j, i, value)
            };
            let position = if stamp[row] == col + 1 {
                slot[row]
            } else {
                stamp[row] = col + 1;
                let position = cursors[col];
                cursors[col] += 1;
                slot[row] = position;
                row_indices[position] = row;
                position
            };
            values[position] = value;
        }
    }

    // Sort rows within each column for deterministic structure.
    for j in 0..n {
        let segment = &mut row_indices[col_ptrs[j]..col_ptrs[j + 1]];
        let value_segment = &mut values[col_ptrs[j]..col_ptrs[j + 1]];
        let mut order: Vec<usize> = (0..segment.len()).collect();
        order.sort_by_key(|&p| segment[p]);
        let sorted_rows: Vec<usize> = order.iter().map(|&p| segment[p]).collect();
        let sorted_values: Vec<T> = order.iter().map(|&p| value_segment[p]).collect();
        segment.copy_from_slice(&sorted_rows);
        value_segment.copy_from_slice(&sorted_values);
    }

    UpperTriangle {
        col_ptrs,
        row_indices,
        values,
    }
}

// ---------------------------------------------------------------------------
// Elimination tree and symbolic pattern of L
// ---------------------------------------------------------------------------

/// Elimination tree of a symmetric pattern given as CSC of its upper
/// triangle. `parent[j]` is the first row index above `j` that fills in
/// column `j` (`usize::MAX` at roots). Same algorithm as the column etree
/// in `sparse_lu` (Tim Davis, path compression), specialized to a
/// pattern-only CSC.
fn elimination_tree(n: usize, col_ptrs: &[usize], row_indices: &[usize]) -> Vec<usize> {
    let mut parent = vec![usize::MAX; n];
    let mut ancestor = vec![usize::MAX; n];
    for k in 0..n {
        for &start in &row_indices[col_ptrs[k]..col_ptrs[k + 1]] {
            let mut i = start;
            while i != usize::MAX && i < k {
                let next = ancestor[i];
                ancestor[i] = k;
                if next == usize::MAX {
                    parent[i] = k;
                }
                i = next;
            }
        }
    }
    parent
}

/// Nonzero pattern of `L` from the etree and the upper-triangle pattern.
///
/// For each row `k`, the columns of `L(k,:)` form the row subtree: the
/// union of etree paths from each row `i < k` of `A(:,k)` up to (but
/// excluding) `k`. Each etree edge is traversed at most once per row, so
/// the total work is `O(nnz(L))`. Column `j` of the result stores the
/// diagonal row `j` first, followed by rows `> j` in increasing order —
/// exactly the order the numeric phase appends them.
fn symbolic_pattern<T: ComplexField>(
    n: usize,
    upper: &UpperTriangle<T>,
    parent: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut counts = vec![0usize; n];
    let mut marker = vec![usize::MAX; n];
    for k in 0..n {
        for p in upper.col_ptrs[k]..upper.col_ptrs[k + 1] {
            let mut i = upper.row_indices[p];
            while i != usize::MAX && i < k && marker[i] != k {
                marker[i] = k;
                counts[i] += 1;
                i = parent[i];
            }
        }
    }

    let mut col_ptrs = vec![0usize; n + 1];
    for j in 0..n {
        col_ptrs[j + 1] = col_ptrs[j] + counts[j] + 1; // +1 for the diagonal
    }

    let mut row_indices = vec![0usize; col_ptrs[n]];
    let mut cursors = vec![0usize; n];
    for (j, cursor) in cursors.iter_mut().enumerate() {
        row_indices[col_ptrs[j]] = j; // diagonal first
        *cursor = col_ptrs[j] + 1;
    }
    marker.fill(usize::MAX);
    for k in 0..n {
        for p in upper.col_ptrs[k]..upper.col_ptrs[k + 1] {
            let mut i = upper.row_indices[p];
            while i != usize::MAX && i < k && marker[i] != k {
                marker[i] = k;
                row_indices[cursors[i]] = k;
                cursors[i] += 1;
                i = parent[i];
            }
        }
    }

    (col_ptrs, row_indices)
}

/// Row-subtree reach for row `k` (CSparse `cs_ereach`): the columns of
/// `L(k,:)` in topological order — every node appears before its etree
/// ancestors, which is the order the numeric triangular solve requires.
/// Returns the pattern in `s[..len]`.
fn ereach(
    k: usize,
    upper_ptrs: &[usize],
    upper_rows: &[usize],
    parent: &[usize],
    marker: &mut [usize],
    s: &mut [usize],
    path: &mut Vec<usize>,
) -> usize {
    let n = parent.len();
    let mut top = n;
    for &start in &upper_rows[upper_ptrs[k]..upper_ptrs[k + 1]] {
        let mut i = start;
        path.clear();
        while i != usize::MAX && i < k && marker[i] != k {
            marker[i] = k;
            path.push(i);
            i = parent[i];
        }
        // Push the path reversed so descendants precede ancestors in `s`.
        for &node in path.iter().rev() {
            top -= 1;
            s[top] = node;
        }
    }
    s.copy_within(top..n, 0);
    n - top
}

// ---------------------------------------------------------------------------
// Numeric factorization
// ---------------------------------------------------------------------------

/// Up-looking numeric Cholesky (CSparse `cs_chol` structure), reusing the
/// symbolic pattern. Column `i` of `L` is appended in increasing row order
/// as row `k` sweeps past; the workspace `x` accumulates `A(:,k)` (upper
/// part) minus the contributions of every previously computed `L(k,j)`.
fn numeric_factorize<T: ComplexField>(
    upper: &UpperTriangle<T>,
    symbolic: &SparseCholeskySymbolic,
    hermitian: bool,
) -> Result<Vec<T>, SparseCholeskyError> {
    let n = symbolic.n;
    let ptrs = &symbolic.l_col_ptrs;
    let mut l_values = vec![T::zero(); symbolic.l_row_indices.len()];

    // Pivot tolerance relative to the diagonal scale, mirroring the dense
    // Cholesky's scale-aware check.
    let scale = (0..n)
        .map(|k| {
            let mut diag = T::Real::zero();
            for p in upper.col_ptrs[k]..upper.col_ptrs[k + 1] {
                if upper.row_indices[p] == k {
                    diag = upper.values[p].norm();
                }
            }
            diag
        })
        .fold(T::Real::zero(), T::Real::max)
        .max(T::Real::min_positive_value());
    let tolerance =
        T::Real::epsilon() * T::Real::from_usize(n.max(1)).unwrap_or_else(T::Real::one) * scale;

    let mut x = vec![T::zero(); n]; // dense workspace
    let mut marker = vec![usize::MAX; n];
    let mut s = vec![0usize; n]; // row-subtree pattern scratch
    let mut path = Vec::new(); // etree walk scratch
    let mut cursors = ptrs[..n].to_vec(); // append cursor per column

    for k in 0..n {
        let pattern_len = ereach(
            k,
            &upper.col_ptrs,
            &upper.row_indices,
            &symbolic.parent,
            &mut marker,
            &mut s,
            &mut path,
        );

        // Scatter the upper part of A(:,k) into the workspace.
        for p in upper.col_ptrs[k]..upper.col_ptrs[k + 1] {
            let i = upper.row_indices[p];
            if i <= k {
                x[i] = upper.values[p];
            }
        }
        let mut d = x[k];
        x[k] = T::zero();

        // Triangular solve L(0:k-1, 0:k-1)·x = A(0:k-1, k) in etree
        // topological order, appending L(k,i) to column i as it finalizes.
        for &i in s.iter().take(pattern_len) {
            let diag = l_values[ptrs[i]];
            let lki = x[i] * diag.inv();
            x[i] = T::zero();
            for p in ptrs[i] + 1..cursors[i] {
                x[symbolic.l_row_indices[p]] -= l_values[p] * lki;
            }
            d -= if hermitian {
                lki * lki.conj()
            } else {
                lki * lki
            };
            let p = cursors[i];
            debug_assert_eq!(symbolic.l_row_indices[p], k);
            l_values[p] = if hermitian { lki.conj() } else { lki };
            cursors[i] += 1;
        }

        // Pivot: must be real, positive, and finite.
        if !d.re().is_finite() || !d.im().is_finite() {
            return Err(SparseCholeskyError::NumericalBreakdown {
                stage: "simplicial pivot formation",
                index: k,
            });
        }
        if d.im().abs() > tolerance || d.re() <= tolerance {
            return Err(SparseCholeskyError::NotPositiveDefinite { pivot: k });
        }
        l_values[ptrs[k]] = T::from_real(d.re().sqrt());
        cursors[k] += 1; // diagonal occupies the first slot of column k
    }

    Ok(l_values)
}

fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    values.iter().position(|value| !value.norm().is_finite())
}

// ---------------------------------------------------------------------------
// Triangular solve kernels and level sets
// ---------------------------------------------------------------------------

/// One column of the forward sweep: `y[j] /= L(j,j)`, then scatter
/// `y[i] -= L(i,j)·y[j]` for `i > j` in the column pattern.
fn scatter_forward_column<T: ComplexField>(
    j: usize,
    y: &mut [T],
    ptrs: &[usize],
    rows: &[usize],
    values: &[T],
) {
    let diag = values[ptrs[j]];
    let yj = y[j] * diag.inv();
    y[j] = yj;
    for p in ptrs[j] + 1..ptrs[j + 1] {
        y[rows[p]] -= values[p] * yj;
    }
}

/// One column of the backward sweep: gather
/// `x[j] = (x[j] - Σ_{i>j} conj(L(i,j))·x[i]) / L(j,j)`.
fn gather_backward_column<T: ComplexField>(
    j: usize,
    x: &mut [T],
    ptrs: &[usize],
    rows: &[usize],
    values: &[T],
    hermitian: bool,
) {
    let mut acc = x[j];
    for p in ptrs[j] + 1..ptrs[j + 1] {
        let coefficient = if hermitian {
            values[p].conj()
        } else {
            values[p]
        };
        acc -= coefficient * x[rows[p]];
    }
    x[j] = acc * values[ptrs[j]].inv();
}

/// Parallel forward substitution for one level.
///
/// # Safety
/// Columns in `level` must write to disjoint rows (guaranteed by
/// [`compute_l_level_sets`]). `y_ptr` must be valid for at least
/// `max(rows) + 1` elements.
unsafe fn parallel_forward_level<T: ComplexField>(
    level: &[usize],
    y_ptr: *mut T,
    ptrs: &[usize],
    rows: &[usize],
    values: &[T],
) {
    let ptr = y_ptr as usize;
    level.par_iter().for_each(|&j| {
        let y = ptr as *mut T;
        // SAFETY: disjoint write regions guaranteed by level-set construction.
        let diag = values[ptrs[j]];
        let yj = unsafe { *y.add(j) } * diag.inv();
        unsafe { *y.add(j) = yj };
        for p in ptrs[j] + 1..ptrs[j + 1] {
            unsafe { *y.add(rows[p]) -= values[p] * yj };
        }
    });
}

/// Parallel backward substitution for one level.
///
/// # Safety
/// Columns in `level` write only their own `x[j]` and read entries solved
/// in earlier levels. `x_ptr` must be valid for `n` elements.
unsafe fn parallel_backward_level<T: ComplexField>(
    level: &[usize],
    x_ptr: *mut T,
    ptrs: &[usize],
    rows: &[usize],
    values: &[T],
    hermitian: bool,
) {
    let ptr = x_ptr as usize;
    level.par_iter().for_each(|&j| {
        let x = ptr as *mut T;
        // SAFETY: each column writes only x[j]; reads target earlier levels.
        let mut acc = unsafe { *x.add(j) };
        for p in ptrs[j] + 1..ptrs[j + 1] {
            let coefficient = if hermitian {
                values[p].conj()
            } else {
                values[p]
            };
            acc -= coefficient * unsafe { *x.add(rows[p]) };
        }
        unsafe { *x.add(j) = acc * values[ptrs[j]].inv() };
    });
}

/// Level sets for the parallel forward solve: columns in the same level
/// scatter into disjoint rows and depend only on earlier levels. Mirrors
/// `compute_l_level_sets` in `sparse_lu` with an identity row permutation
/// (the Cholesky factor stores permuted row indices directly).
fn compute_l_level_sets(ptrs: &[usize], rows: &[usize], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    // `ready_level[i]` is the first level at which y[i] may be read or
    // written. Every update to y[i] advances it past the writer's level,
    // serializing columns that scatter into the same row.
    let mut ready_level = vec![0usize; n];
    let mut col_level = vec![0usize; n];
    // One scratch buffer for all columns (was: one allocation per column).
    let mut write_rows = Vec::new();
    for j in 0..n {
        let mut level = ready_level[j];
        write_rows.clear();
        for &i in &rows[ptrs[j] + 1..ptrs[j + 1]] {
            write_rows.push(i);
            level = level.max(ready_level[i]);
        }
        col_level[j] = level;
        for &row in &write_rows {
            ready_level[row] = level + 1;
        }
    }
    let max_level = col_level.iter().copied().max().unwrap_or(0);
    let mut levels = vec![Vec::new(); max_level + 1];
    for (j, &level) in col_level.iter().enumerate() {
        levels[level].push(j);
    }
    levels
}

/// Level sets for the parallel backward solve (gather form): `x[j]` reads
/// `x[i]` for `i > j` in column `j`'s pattern, so `level[j]` must exceed
/// every dependency's level. Columns within a level write only their own
/// slot, so no write conflicts are possible. Levels are processed in
/// ascending order.
fn compute_lh_level_sets(ptrs: &[usize], rows: &[usize], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    let mut col_level = vec![0usize; n];
    for j in (0..n).rev() {
        let mut level = 0;
        for p in ptrs[j] + 1..ptrs[j + 1] {
            level = level.max(col_level[rows[p]] + 1);
        }
        col_level[j] = level;
    }
    let max_level = col_level.iter().copied().max().unwrap_or(0);
    let mut levels = vec![Vec::new(); max_level + 1];
    for (j, &level) in col_level.iter().enumerate() {
        levels[level].push(j);
    }
    levels
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_lu::SparseLuFactorization;
    use num_complex::{Complex32, Complex64};

    /// 5-point 2D Laplacian on an `nx × nx` grid (SPD).
    fn laplacian_2d(nx: usize) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut triplets = Vec::with_capacity(5 * n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                triplets.push((i, i, 4.0));
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
        CsrMatrix::from_triplets(n, n, triplets)
    }

    /// Reference solution and RHS for an exact-residual check.
    fn exact_system<T: ComplexField>(matrix: &CsrMatrix<T>) -> (Vec<T>, Vec<T>) {
        let n = matrix.num_rows;
        let seven = T::Real::from_usize(7).unwrap();
        let x_exact: Vec<T> = (0..n)
            .map(|i| {
                T::from_re_im(
                    T::Real::from_usize(i % 7 + 1).unwrap() / seven,
                    T::Real::zero(),
                )
            })
            .collect();
        let b = matrix.matvec(&x_exact);
        (x_exact, b)
    }

    fn residual_norm<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real {
        let actual = matrix.matvec(x);
        actual
            .iter()
            .zip(b)
            .map(|(a, e)| (*a - *e).norm())
            .fold(T::Real::zero(), T::Real::max)
    }

    #[test]
    fn spd_laplacian_solve_residual() {
        let matrix = laplacian_2d(40); // n = 1600, exercises the parallel solve path
        let (x_exact, b) = exact_system(&matrix);
        let factor = SparseCholeskyFactorization::factorize(&matrix).unwrap();
        let x = factor.solve(&b).unwrap();
        let residual = residual_norm(&matrix, &x, &b);
        assert!(residual < 1e-10, "residual {residual:e}");
        let error = x
            .iter()
            .zip(&x_exact)
            .map(|(a, e)| (*a - *e).norm())
            .fold(0.0_f64, f64::max);
        assert!(error < 1e-9, "solution error {error:e}");
    }

    #[test]
    fn natural_ordering_also_solves() {
        let matrix = laplacian_2d(16);
        let (_, b) = exact_system(&matrix);
        let factor = SparseCholeskyFactorization::factorize_with_ordering(
            &matrix,
            CholeskyOrdering::Natural,
            false,
        )
        .unwrap();
        let x = factor.solve(&b).unwrap();
        assert!(residual_norm(&matrix, &x, &b) < 1e-10);
        // ND should produce no more fill than natural ordering on a grid.
        let nd = SparseCholeskyFactorization::factorize(&matrix).unwrap();
        assert!(nd.factor_nonzeros() <= factor.factor_nonzeros());
    }

    #[test]
    fn colamd_ordering_also_solves() {
        let matrix = laplacian_2d(16);
        let (_, b) = exact_system(&matrix);
        let factor = SparseCholeskyFactorization::factorize_with_ordering(
            &matrix,
            CholeskyOrdering::Colamd,
            false,
        )
        .unwrap();
        let x = factor.solve(&b).unwrap();
        assert!(residual_norm(&matrix, &x, &b) < 1e-10);
    }

    /// HPD matrix: diagonal 6, Laplacian off-diagonals, plus `±0.5i` skew
    /// on horizontal neighbours. Hermitian and strictly diagonally dominant
    /// (Gershgorin radius ≤ 4·|−1+0.5i| ≈ 4.47 < 6), hence genuinely HPD.
    /// (A plain `L + iS` with diagonal 4 is only Hermitian, not positive
    /// definite — both this code and the dense Cholesky reject it.)
    fn hpd_matrix(nx: usize) -> CsrMatrix<Complex64> {
        let lap = laplacian_2d(nx);
        let n = lap.num_rows;
        let mut triplets = Vec::with_capacity(lap.nnz() + n);
        for r in 0..n {
            for idx in lap.row_ptrs[r]..lap.row_ptrs[r + 1] {
                let c = lap.col_indices[idx];
                let v = lap.values[idx];
                let skew = if c == r + 1 {
                    Complex64::new(0.0, 0.5)
                } else if c + 1 == r {
                    Complex64::new(0.0, -0.5)
                } else {
                    Complex64::zero()
                };
                let diagonal_boost = if c == r { 2.0 } else { 0.0 };
                triplets.push((r, c, Complex64::new(v + diagonal_boost, 0.0) + skew));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    #[test]
    fn hpd_complex_solve() {
        let matrix = hpd_matrix(12);
        let (x_exact, b) = exact_system(&matrix);
        let factor = SparseCholeskyFactorization::factorize_hermitian(&matrix).unwrap();
        let x = factor.solve(&b).unwrap();
        let residual = residual_norm(&matrix, &x, &b);
        assert!(residual < 1e-10, "residual {residual:e}");
        let error = x
            .iter()
            .zip(&x_exact)
            .map(|(a, e)| (*a - *e).norm())
            .fold(0.0_f64, f64::max);
        assert!(error < 1e-9, "solution error {error:e}");
    }

    #[test]
    fn non_spd_input_reports_not_positive_definite() {
        let nx = 10;
        let n = nx * nx;
        let mut triplets = Vec::new();
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                // One negative diagonal entry makes the matrix indefinite.
                triplets.push((i, i, if i == n / 2 { -1.0 } else { 4.0 }));
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
        assert!(matches!(
            SparseCholeskyFactorization::factorize(&matrix),
            Err(SparseCholeskyError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn agrees_with_sparse_lu() {
        let matrix = laplacian_2d(24);
        let (_, b) = exact_system(&matrix);
        let chol = SparseCholeskyFactorization::factorize(&matrix).unwrap();
        let lu = SparseLuFactorization::factorize(&matrix).unwrap();
        let x_chol = chol.solve(&b).unwrap();
        let x_lu = lu.solve(&b).unwrap();
        for (c, l) in x_chol.iter().zip(&x_lu) {
            assert!((*c - *l).abs() < 1e-9, "chol {c} vs lu {l}");
        }
    }

    #[test]
    fn symbolic_reuse_across_values() {
        let nx = 16;
        let a1 = laplacian_2d(nx);
        // Same pattern, different values (anisotropic coefficient).
        let n = a1.num_rows;
        let mut triplets = Vec::with_capacity(a1.nnz());
        for r in 0..n {
            for idx in a1.row_ptrs[r]..a1.row_ptrs[r + 1] {
                let c = a1.col_indices[idx];
                let v = if c == r { 8.0 } else { -2.0 };
                triplets.push((r, c, v));
            }
        }
        let a2 = CsrMatrix::from_triplets(n, n, triplets);

        let symbolic = SparseCholeskySymbolic::analyze(&a1, CholeskyOrdering::NestedDissection)
            .expect("analyze a1");
        let f1 = SparseCholeskyFactorization::factorize_with_symbolic(&a1, &symbolic, false)
            .expect("factorize a1");
        let f2 = SparseCholeskyFactorization::factorize_with_symbolic(&a2, &symbolic, false)
            .expect("factorize a2 with reused symbolic");

        let (_, b1) = exact_system(&a1);
        let (_, b2) = exact_system(&a2);
        let x1 = f1.solve(&b1).unwrap();
        let x2 = f2.solve(&b2).unwrap();
        assert!(residual_norm(&a1, &x1, &b1) < 1e-10);
        assert!(residual_norm(&a2, &x2, &b2) < 1e-10);

        // A different pattern of the same dimension must be rejected.
        let mut triplets = Vec::with_capacity(a1.nnz() + 2);
        for r in 0..n {
            for idx in a1.row_ptrs[r]..a1.row_ptrs[r + 1] {
                triplets.push((r, a1.col_indices[idx], a1.values[idx]));
            }
        }
        triplets.push((0, 2, -0.25));
        triplets.push((2, 0, -0.25));
        let a3 = CsrMatrix::from_triplets(n, n, triplets);
        assert!(matches!(
            SparseCholeskyFactorization::factorize_with_symbolic(&a3, &symbolic, false),
            Err(SparseCholeskyError::PatternMismatch)
        ));
    }

    #[test]
    fn malformed_csr_is_rejected_before_cholesky_analysis_or_reuse() {
        let valid = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]);
        let symbolic = SparseCholeskySymbolic::analyze(&valid, CholeskyOrdering::Natural)
            .expect("valid symbolic analysis");

        let mut malformed_rows = valid.clone();
        malformed_rows.row_ptrs[2] = 1;
        assert!(matches!(
            SparseCholeskySymbolic::analyze(&malformed_rows, CholeskyOrdering::Natural),
            Err(SparseCholeskyError::InvalidCsrStructure(_))
        ));

        let mut malformed_columns = valid;
        malformed_columns.col_indices[0] = 2;
        assert!(matches!(
            SparseCholeskyFactorization::factorize_with_symbolic(
                &malformed_columns,
                &symbolic,
                false
            ),
            Err(SparseCholeskyError::InvalidCsrStructure(_))
        ));
    }

    #[test]
    fn badly_scaled_anisotropic_spd() {
        // A = D·L·D with D varying over six orders of magnitude.
        let nx = 16;
        let n = nx * nx;
        let scale: Vec<f64> = (0..n).map(|i| 10f64.powi((i % 7) as i32 - 3)).collect();
        let lap = laplacian_2d(nx);
        let mut triplets = Vec::with_capacity(lap.nnz());
        for r in 0..n {
            for idx in lap.row_ptrs[r]..lap.row_ptrs[r + 1] {
                let c = lap.col_indices[idx];
                triplets.push((r, c, scale[r] * lap.values[idx] * scale[c]));
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, triplets);
        let (_, b) = exact_system(&matrix);
        let factor = SparseCholeskyFactorization::factorize(&matrix).unwrap();
        let x = factor.solve(&b).unwrap();
        // Scaled backward error: ||b - Ax|| / (||A||∞·||x||∞ + ||b||∞).
        let residual = residual_norm(&matrix, &x, &b);
        let x_norm = x.iter().map(ComplexField::norm).fold(0.0, f64::max);
        let b_norm = b.iter().map(ComplexField::norm).fold(0.0, f64::max);
        let a_norm = (0..n)
            .map(|r| {
                (matrix.row_ptrs[r]..matrix.row_ptrs[r + 1])
                    .map(|idx| matrix.values[idx].abs())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let backward = residual / (a_norm * x_norm + b_norm);
        assert!(backward < 1e-9, "backward error {backward:e}");
    }

    #[test]
    fn f32_and_complex32_smoke() {
        let lap = laplacian_2d(8);
        let n = lap.num_rows;
        let f32_matrix: CsrMatrix<f32> = CsrMatrix::from_raw_parts(
            n,
            n,
            lap.row_ptrs.clone(),
            lap.col_indices.clone(),
            lap.values.iter().map(|&v| v as f32).collect(),
        );
        let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 + 1.0).collect();
        let factor = SparseCholeskyFactorization::factorize(&f32_matrix).unwrap();
        let x = factor.solve(&b).unwrap();
        assert!(residual_norm(&f32_matrix, &x, &b) < 1e-3);

        let c32_matrix: CsrMatrix<Complex32> = CsrMatrix::from_raw_parts(
            n,
            n,
            lap.row_ptrs.clone(),
            lap.col_indices.clone(),
            lap.values
                .iter()
                .map(|&v| Complex32::new(v as f32, 0.0))
                .collect(),
        );
        let b32: Vec<Complex32> = b.iter().map(|&v| Complex32::new(v, 0.5 * v)).collect();
        let factor = SparseCholeskyFactorization::factorize_hermitian(&c32_matrix).unwrap();
        let x = factor.solve(&b32).unwrap();
        assert!(residual_norm(&c32_matrix, &x, &b32) < 1e-2);
    }

    #[test]
    fn lower_triangle_only_storage_solves() {
        // Only the lower triangle stored: the assembly must symmetrize.
        let lap = laplacian_2d(10);
        let n = lap.num_rows;
        let mut triplets = Vec::new();
        for r in 0..n {
            for idx in lap.row_ptrs[r]..lap.row_ptrs[r + 1] {
                let c = lap.col_indices[idx];
                if c <= r {
                    triplets.push((r, c, lap.values[idx]));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, triplets);
        // The factorization sees only the lower triangle and must
        // symmetrize; verify against the full SPD Laplacian.
        let full = laplacian_2d(10);
        let (x_exact, _) = exact_system(&full);
        let b = full.matvec(&x_exact);
        let factor = SparseCholeskyFactorization::factorize(&matrix).unwrap();
        let x = factor.solve(&b).unwrap();
        assert!(residual_norm(&full, &x, &b) < 1e-10);
    }

    #[test]
    fn empty_and_singleton_matrices() {
        let empty: CsrMatrix<f64> = CsrMatrix::new(0, 0);
        let factor = SparseCholeskyFactorization::factorize(&empty).unwrap();
        assert_eq!(factor.dimension(), 0);
        assert!(factor.solve(&[]).unwrap().is_empty());

        let one = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0)]);
        let factor = SparseCholeskyFactorization::factorize(&one).unwrap();
        let x = factor.solve(&[8.0]).unwrap();
        assert!((x[0] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn factor_fill_and_time_vs_sparse_lu() {
        let nx = 64; // n = 4096
        let matrix = laplacian_2d(nx);
        let (_, b) = exact_system(&matrix);

        let t0 = std::time::Instant::now();
        let chol = SparseCholeskyFactorization::factorize(&matrix).unwrap();
        let chol_time = t0.elapsed();

        let t0 = std::time::Instant::now();
        let lu = SparseLuFactorization::factorize(&matrix).unwrap();
        let lu_time = t0.elapsed();

        let chol_nnz = chol.factor_nonzeros();
        let lu_nnz = lu.factor_nonzeros();
        let x = chol.solve(&b).unwrap();
        assert!(residual_norm(&matrix, &x, &b) < 1e-10);
        assert!(
            chol_nnz <= lu_nnz,
            "chol nnz {chol_nnz} should not exceed lu nnz {lu_nnz}"
        );
        println!(
            "2D Laplacian {nx}x{nx} (n={}): nnz(L)={chol_nnz} vs nnz(L+U)={lu_nnz} \
             ({:.2}x); factor time chol={chol_time:?} vs lu={lu_time:?} ({:.2}x)",
            matrix.num_rows,
            lu_nnz as f64 / chol_nnz as f64,
            lu_time.as_secs_f64() / chol_time.as_secs_f64().max(1e-9),
        );
    }
}
