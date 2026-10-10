//! Dense reorthogonalized QR, column-pivoted rank detection, least squares,
//! and a thin real SVD for reference-sized problems.

use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, ComplexField, SolverFailure, SolverStatus},
};
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq)]
pub enum QrError {
    #[error("QR requires rows >= columns, got {rows}x{columns}")]
    Underdetermined { rows: usize, columns: usize },
    #[error("QR input contains a non-finite value")]
    NonFiniteInput,
    #[error("QR right-hand side contains a non-finite value")]
    NonFiniteRhs,
    #[error("QR {operation} generated a non-finite value")]
    NumericalBreakdown { operation: &'static str },
    #[error("matrix has numerical rank {rank}, expected {columns}")]
    RankDeficient { rank: usize, columns: usize },
    #[error("right-hand side has length {actual}, expected {expected}")]
    RhsDimension { expected: usize, actual: usize },
    #[error("solution has length {actual}, expected {expected}")]
    SolutionDimension { expected: usize, actual: usize },
    #[error("QR multi-RHS shape overflows usize: rows={rows}, columns={columns}, nrhs={nrhs}")]
    SizeOverflow {
        rows: usize,
        columns: usize,
        nrhs: usize,
    },
}

impl QrError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NumericalBreakdown { .. } | Self::RankDeficient { .. } => SolverStatus::Breakdown,
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

impl SolverFailure for QrError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::RhsDimension { .. }
            | Self::SolutionDimension { .. }
            | Self::SizeOverflow { .. } => SolverErrorKind::DimensionMismatch,
            Self::NumericalBreakdown { .. } | Self::RankDeficient { .. } => {
                SolverErrorKind::Breakdown
            }
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        QrError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        QrError::accuracy_status(self)
    }
}

/// Thin `A P = Q R` factorization. `Q` has shape `m x n`, `R` is `n x n`,
/// and `permutation[k]` is the original column represented by factor column
/// `k`.
#[derive(Debug, Clone)]
pub struct QrFactorization<T: ComplexField> {
    rows: usize,
    columns: usize,
    q: DenseMatrix<T>,
    r: DenseMatrix<T>,
    permutation: Vec<usize>,
    rank: usize,
}

impl<T: ComplexField> QrFactorization<T> {
    pub fn factorize(matrix: &DenseMatrix<T>) -> Result<Self, QrError> {
        // Wide non-pivoted shapes take the blocked Householder path (same
        // result up to rounding, BLAS-3 rate); pivoted, narrow, and tiny
        // shapes keep reorthogonalized MGS.
        if matrix.ncols() >= QR_BLOCKED_MIN_COLS {
            return factorize_blocked(matrix);
        }
        factorize_impl(matrix, false)
    }

    pub fn factorize_pivoted(matrix: &DenseMatrix<T>) -> Result<Self, QrError> {
        factorize_impl(matrix, true)
    }

    /// Communication-avoiding tall-skinny QR.
    ///
    /// Rows are split into independently factorized blocks, their triangular
    /// factors are reduced by a second QR, and the thin global Q is formed as
    /// a block product. Small matrices use the ordinary reorthogonalized QR.
    pub fn factorize_tsqr(matrix: &DenseMatrix<T>) -> Result<Self, QrError> {
        let rows = matrix.nrows();
        let columns = matrix.ncols();
        validate_qr_input(matrix)?;
        let target_block_rows = (4 * columns).max(256);
        let block_count = (rows / target_block_rows).max(1);
        if block_count == 1 {
            return Self::factorize(matrix);
        }

        let mut blocks = Vec::with_capacity(block_count);
        let mut row_start = 0;
        for block in 0..block_count {
            let remaining_rows = rows - row_start;
            let remaining_blocks = block_count - block;
            let block_rows = remaining_rows / remaining_blocks;
            let local = DenseMatrix::from_fn(block_rows, columns, |row, column| {
                matrix[[row_start + row, column]]
            });
            // A row block may be rank deficient even when the stacked matrix
            // is full rank. Keep its rank-revealing basis and let the global
            // reduction decide the rank of the complete matrix.
            let factor = Self::factorize_pivoted(&local)?;
            blocks.push((row_start, factor));
            row_start += block_rows;
        }

        let mut stacked_r = DenseMatrix::zeros(block_count * columns, columns);
        for (block, (_, factor)) in blocks.iter().enumerate() {
            for column in 0..columns {
                let factor_column = factor
                    .permutation
                    .iter()
                    .position(|&original_column| original_column == column)
                    .ok_or(QrError::NumericalBreakdown {
                        operation: "TSQR permutation reconstruction",
                    })?;
                for local_row in 0..columns {
                    stacked_r[[block * columns + local_row, column]] =
                        factor.r[[local_row, factor_column]];
                }
            }
        }
        ensure_finite_matrix("TSQR stacked factor", &stacked_r)?;
        let reduction = Self::factorize(&stacked_r)?;
        let q = DenseMatrix::from_fn(rows, columns, |row, column| {
            let block = blocks
                .partition_point(|(start, _)| *start <= row)
                .saturating_sub(1);
            let (start, local) = &blocks[block];
            let local_row = row - *start;
            (0..columns).fold(T::zero(), |sum, inner| {
                sum + local.q[[local_row, inner]] * reduction.q[[block * columns + inner, column]]
            })
        });
        ensure_finite_matrix("TSQR global basis", &q)?;
        Ok(Self {
            rows,
            columns,
            q,
            r: reduction.r,
            permutation: (0..columns).collect(),
            rank: columns,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn q(&self) -> &DenseMatrix<T> {
        &self.q
    }

    pub fn r(&self) -> &DenseMatrix<T> {
        &self.r
    }

    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    pub fn solve_least_squares(&self, rhs: &[T]) -> Result<Vec<T>, QrError> {
        let mut solution = vec![T::zero(); self.columns];
        self.solve_least_squares_into(rhs, &mut solution)?;
        Ok(solution)
    }

    pub fn solve_least_squares_into(&self, rhs: &[T], solution: &mut [T]) -> Result<(), QrError> {
        if rhs.len() != self.rows {
            return Err(QrError::RhsDimension {
                expected: self.rows,
                actual: rhs.len(),
            });
        }
        if solution.len() != self.columns {
            return Err(QrError::SolutionDimension {
                expected: self.columns,
                actual: solution.len(),
            });
        }
        if self.rank != self.columns {
            return Err(QrError::RankDeficient {
                rank: self.rank,
                columns: self.columns,
            });
        }
        if rhs.iter().any(|value| !value.norm().is_finite()) {
            return Err(QrError::NonFiniteRhs);
        }
        let mut transformed = vec![T::zero(); self.columns];
        for (column, transformed_value) in transformed.iter_mut().enumerate() {
            let q_column: Vec<_> = (0..self.rows).map(|row| self.q[[row, column]]).collect();
            *transformed_value = T::vec_dot(&q_column, rhs);
            ensure_finite_value("least-squares projection", *transformed_value)?;
        }
        let mut factor_order = vec![T::zero(); self.columns];
        for row in (0..self.columns).rev() {
            let mut value = transformed[row];
            for (column, &factor_value) in factor_order.iter().enumerate().skip(row + 1) {
                value -= self.r[[row, column]] * factor_value;
            }
            factor_order[row] = value * self.r[[row, row]].inv();
            ensure_finite_value("least-squares back substitution", factor_order[row])?;
        }
        solution.fill(T::zero());
        for (factor_column, &original_column) in self.permutation.iter().enumerate() {
            solution[original_column] = factor_order[factor_column];
        }
        if solution.iter().any(|value| !value.norm().is_finite()) {
            return Err(QrError::NumericalBreakdown {
                operation: "least-squares solution",
            });
        }
        Ok(())
    }

    /// Solve multiple row-major right-hand sides without mutating the
    /// destination until every column has passed validation and solve.
    pub fn solve_least_squares_many_into(
        &self,
        rhs: &[T],
        solution: &mut [T],
        nrhs: usize,
    ) -> Result<(), QrError> {
        let rhs_len = self.rows.checked_mul(nrhs).ok_or(QrError::SizeOverflow {
            rows: self.rows,
            columns: self.columns,
            nrhs,
        })?;
        let solution_len = self
            .columns
            .checked_mul(nrhs)
            .ok_or(QrError::SizeOverflow {
                rows: self.rows,
                columns: self.columns,
                nrhs,
            })?;
        if rhs.len() != rhs_len {
            return Err(QrError::RhsDimension {
                expected: rhs_len,
                actual: rhs.len(),
            });
        }
        if solution.len() != solution_len {
            return Err(QrError::SolutionDimension {
                expected: solution_len,
                actual: solution.len(),
            });
        }
        if rhs.iter().any(|value| !value.norm().is_finite()) {
            return Err(QrError::NonFiniteRhs);
        }

        let mut computed = vec![T::zero(); solution_len];
        let mut column_rhs = vec![T::zero(); self.rows];
        let mut column_solution = vec![T::zero(); self.columns];
        for column in 0..nrhs {
            for row in 0..self.rows {
                column_rhs[row] = rhs[row * nrhs + column];
            }
            self.solve_least_squares_into(&column_rhs, &mut column_solution)?;
            for row in 0..self.columns {
                computed[row * nrhs + column] = column_solution[row];
            }
        }
        solution.copy_from_slice(&computed);
        Ok(())
    }

    pub fn factor_nonzeros(&self) -> usize {
        self.rows
            .saturating_mul(self.columns)
            .saturating_add(self.columns.saturating_mul(self.columns.saturating_add(1)) / 2)
    }
}

fn factorize_impl<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    pivoted: bool,
) -> Result<QrFactorization<T>, QrError> {
    validate_qr_input(matrix)?;
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    let mut work: Vec<Vec<T>> = (0..columns)
        .map(|column| (0..rows).map(|row| matrix[[row, column]]).collect())
        .collect();
    let mut permutation: Vec<_> = (0..columns).collect();
    let mut q = DenseMatrix::zeros(rows, columns);
    let mut r = DenseMatrix::zeros(columns, columns);
    let largest_norm = work.iter().try_fold(T::Real::zero(), |largest, column| {
        checked_norm(column, "initial column norm").map(|norm| largest.max(norm))
    })?;
    let dimension_scale =
        T::Real::from_usize(rows.max(columns)).ok_or(QrError::NumericalBreakdown {
            operation: "dimension conversion",
        })?;
    let tolerance = T::Real::epsilon() * dimension_scale * largest_norm.max(T::Real::one());
    if !tolerance.is_finite() {
        return Err(QrError::NumericalBreakdown {
            operation: "rank tolerance",
        });
    }
    let mut rank = columns;

    for step in 0..columns {
        if pivoted {
            let mut pivot = step;
            let mut pivot_norm = checked_norm(&work[step], "pivot column norm")?;
            for (column, work_column) in work.iter().enumerate().skip(step + 1) {
                let norm = checked_norm(work_column, "pivot column norm")?;
                if norm > pivot_norm {
                    pivot = column;
                    pivot_norm = norm;
                }
            }
            if pivot != step {
                work.swap(step, pivot);
                permutation.swap(step, pivot);
                for row in 0..step {
                    let value = r[[row, step]];
                    r[[row, step]] = r[[row, pivot]];
                    r[[row, pivot]] = value;
                }
            }
        }
        let norm = checked_norm(&work[step], "factor column norm")?;
        if norm <= tolerance {
            rank = step;
            break;
        }
        r[[step, step]] = T::from_real(norm);
        let inverse_norm = T::from_real(norm.recip());
        ensure_finite_value("factor column normalization", inverse_norm)?;
        for row in 0..rows {
            q[[row, step]] = work[step][row] * inverse_norm;
            ensure_finite_value("factor column normalization", q[[row, step]])?;
        }
        let q_column: Vec<_> = (0..rows).map(|row| q[[row, step]]).collect();
        for column in step + 1..columns {
            // Two passes make modified Gram-Schmidt reliable enough for the
            // rank/least-squares contract while retaining a simple generic
            // real/complex implementation.
            for _ in 0..2 {
                let coefficient = T::vec_dot(&q_column, &work[column]);
                ensure_finite_value("reorthogonalization coefficient", coefficient)?;
                r[[step, column]] += coefficient;
                ensure_finite_value("triangular factor update", r[[step, column]])?;
                T::vec_axpy(-coefficient, &q_column, &mut work[column]);
                ensure_finite_slice("reorthogonalization update", &work[column])?;
            }
        }
    }
    if !pivoted && rank != columns {
        return Err(QrError::RankDeficient { rank, columns });
    }
    ensure_finite_matrix("orthogonal factor", &q)?;
    ensure_finite_matrix("triangular factor", &r)?;
    Ok(QrFactorization {
        rows,
        columns,
        q,
        r,
        permutation,
        rank,
    })
}

fn validate_qr_input<T: ComplexField>(matrix: &DenseMatrix<T>) -> Result<(), QrError> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows < columns {
        return Err(QrError::Underdetermined { rows, columns });
    }
    if matrix
        .as_slice()
        .iter()
        .any(|value| !value.norm().is_finite())
    {
        return Err(QrError::NonFiniteInput);
    }
    Ok(())
}

fn checked_norm<T: ComplexField>(
    values: &[T],
    operation: &'static str,
) -> Result<T::Real, QrError> {
    let mut scale = T::Real::zero();
    let mut sum_squares = T::Real::one();
    for value in values {
        let magnitude = value.norm();
        if !magnitude.is_finite() {
            return Err(QrError::NumericalBreakdown { operation });
        }
        if magnitude == T::Real::zero() {
            continue;
        }
        if scale < magnitude {
            let ratio = scale / magnitude;
            sum_squares = T::Real::one() + sum_squares * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            sum_squares += ratio * ratio;
        }
    }
    let norm = if scale == T::Real::zero() {
        T::Real::zero()
    } else {
        scale * sum_squares.sqrt()
    };
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(QrError::NumericalBreakdown { operation })
    }
}

fn ensure_finite_value<T: ComplexField>(operation: &'static str, value: T) -> Result<(), QrError> {
    if value.norm().is_finite() {
        Ok(())
    } else {
        Err(QrError::NumericalBreakdown { operation })
    }
}

fn ensure_finite_slice<T: ComplexField>(
    operation: &'static str,
    values: &[T],
) -> Result<(), QrError> {
    if values.iter().all(|value| value.norm().is_finite()) {
        Ok(())
    } else {
        Err(QrError::NumericalBreakdown { operation })
    }
}

fn ensure_finite_matrix<T: ComplexField>(
    operation: &'static str,
    matrix: &DenseMatrix<T>,
) -> Result<(), QrError> {
    ensure_finite_slice(operation, matrix.as_slice())
}

/// Thin real SVD `A = U diag(s) V^T`, computed by one-sided Jacobi rotations.
/// This is intended for reference-sized least-squares/rank contracts; large
/// production problems should use QR or an iterative method.
#[derive(Debug, Clone)]
pub struct DenseSvd {
    pub u: DenseMatrix<f64>,
    pub singular_values: Vec<f64>,
    pub v: DenseMatrix<f64>,
    pub rank: usize,
}

/// Thin real SVD `A = U diag(s) Vᵀ` via one-sided Jacobi rotations.
///
/// Tall or square inputs (`rows >= columns`) return `U` as `rows x columns`,
/// `s` with length `columns`, and `V` as `columns x columns`. Wide inputs
/// (`rows < columns`) are handled through the transpose identity
/// `Aᵀ = U' Σ V'ᵀ ⟹ A = V' Σ U'ᵀ`, returning `U` as `rows x rows`, `s` with
/// length `rows`, and `V` as `columns x rows`, so the reconstruction
/// `A[i,j] = Σₖ U[i,k] s[k] V[j,k]` holds for every shape.
///
/// # Errors
///
/// Returns [`QrError::NonFiniteInput`] for non-finite entries and
/// [`QrError::NumericalBreakdown`] when an intermediate quantity overflows
/// or a Jacobi rotation fails to converge to finite values.
pub fn dense_svd(matrix: &DenseMatrix<f64>) -> Result<DenseSvd, QrError> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows < columns {
        // Wide input: factor the tall transpose, then swap the factors.
        let transposed = dense_svd(&matrix.transposed())?;
        return Ok(DenseSvd {
            u: transposed.v,
            singular_values: transposed.singular_values,
            v: transposed.u,
            rank: transposed.rank,
        });
    }
    if matrix.as_slice().iter().any(|value| !value.is_finite()) {
        return Err(QrError::NonFiniteInput);
    }
    let mut work = matrix.clone();
    let mut vectors = DenseMatrix::from_fn(
        columns,
        columns,
        |row, column| {
            if row == column { 1.0 } else { 0.0 }
        },
    );
    for _ in 0..100 {
        let mut changed = false;
        for left in 0..columns {
            for right in left + 1..columns {
                let (alpha, beta, coupling) =
                    scaled_column_pair_gram(&work, rows, left, right, "SVD column Gram matrix")?;
                if coupling.abs() <= 8.0 * f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                changed = true;
                let zeta = (beta - alpha) / (2.0 * coupling);
                let tangent = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = cosine * tangent;
                if !tangent.is_finite() || !cosine.is_finite() || !sine.is_finite() {
                    return Err(QrError::NumericalBreakdown {
                        operation: "SVD Jacobi rotation",
                    });
                }
                for row in 0..rows {
                    let a = work[[row, left]];
                    let b = work[[row, right]];
                    work[[row, left]] = cosine * a - sine * b;
                    work[[row, right]] = sine * a + cosine * b;
                }
                for row in 0..columns {
                    let a = vectors[[row, left]];
                    let b = vectors[[row, right]];
                    vectors[[row, left]] = cosine * a - sine * b;
                    vectors[[row, right]] = sine * a + cosine * b;
                }
                ensure_finite_matrix("SVD Jacobi matrix update", &work)?;
                ensure_finite_matrix("SVD Jacobi vector update", &vectors)?;
            }
        }
        if !changed {
            break;
        }
    }
    let unsorted_values: Vec<f64> = (0..columns)
        .map(|column| {
            let values: Vec<_> = (0..rows).map(|row| work[[row, column]]).collect();
            checked_norm(&values, "SVD singular value")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut order: Vec<_> = (0..columns).collect();
    order.sort_by(|&left, &right| unsorted_values[right].total_cmp(&unsorted_values[left]));
    let singular_values: Vec<_> = order.iter().map(|&index| unsorted_values[index]).collect();
    let v = DenseMatrix::from_fn(columns, columns, |row, column| {
        vectors[[row, order[column]]]
    });
    let tolerance = f64::EPSILON
        * rows.max(columns) as f64
        * singular_values.first().copied().unwrap_or(0.0).max(1.0);
    if !tolerance.is_finite() {
        return Err(QrError::NumericalBreakdown {
            operation: "SVD rank tolerance",
        });
    }
    let rank = singular_values
        .iter()
        .take_while(|&&value| value > tolerance)
        .count();
    let u = DenseMatrix::from_fn(rows, columns, |row, column| {
        if singular_values[column] <= tolerance {
            0.0
        } else {
            work[[row, order[column]]] / singular_values[column]
        }
    });
    if singular_values.iter().any(|value| !value.is_finite()) {
        return Err(QrError::NumericalBreakdown {
            operation: "SVD singular values",
        });
    }
    ensure_finite_matrix("SVD left singular vectors", &u)?;
    ensure_finite_matrix("SVD right singular vectors", &v)?;
    Ok(DenseSvd {
        u,
        singular_values,
        v,
        rank,
    })
}

fn scaled_column_pair_gram(
    matrix: &DenseMatrix<f64>,
    rows: usize,
    left: usize,
    right: usize,
    operation: &'static str,
) -> Result<(f64, f64, f64), QrError> {
    let scale = (0..rows).fold(0.0_f64, |largest, row| {
        largest
            .max(matrix[[row, left]].abs())
            .max(matrix[[row, right]].abs())
    });
    if !scale.is_finite() {
        return Err(QrError::NumericalBreakdown { operation });
    }
    if scale == 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut coupling = 0.0;
    for row in 0..rows {
        let left_value = matrix[[row, left]] / scale;
        let right_value = matrix[[row, right]] / scale;
        alpha += left_value * left_value;
        beta += right_value * right_value;
        coupling += left_value * right_value;
    }
    if alpha.is_finite() && beta.is_finite() && coupling.is_finite() {
        Ok((alpha, beta, coupling))
    } else {
        Err(QrError::NumericalBreakdown { operation })
    }
}

/// Blocked right-looking Householder QR (LAPACK `dgeqrf` structure) for the
/// non-pivoted path.
///
/// The generic reorthogonalized MGS above is BLAS-1 bound (~2 GFLOPS: column
/// arrays, per-element checked indexing, two passes) and trails OpenBLAS
/// `dgeqrf` 25-52x at 1024-2048. The blocked form moves all O(n^3) work into
/// three threaded GEMM calls per panel — the same entries thin-GEMM probes
/// show beating OpenBLAS — leaving only the O(n^2·nb) panel on serial
/// BLAS-1, exactly the vendor structure. Only plain `factorize` above
/// `QR_BLOCKED_MIN_COLS` routes here; pivoted, TSQR-narrow, and tiny shapes
/// keep MGS. Results match MGS up to rounding (same reflectors, same order
/// of accumulation at the panel level); complex types use the Hermitian dot
/// and the phase-rotated beta, mirroring LAPACK `zlarfg`.
/// Panel width. NB=64 was trialled and reverted: the 4x Level-2 panel/T
/// work outweighs the fatter GEMM shapes (1024: 1.90x -> 2.35x,
/// 2048: 1.92x -> 2.11x vs OB dgeqrf+dorgqr, 16 threads). NB=16 also
/// falsified (thinner GEMMs + 2x call count dominate: 1.79x/2.05x/2.13x
/// at 512/1024/2048). NB=32 stands.
///
/// Forward lookahead (next-panel WY update, then panel prep on a worker
/// overlapped with the update remainder, ping-pong buffers) was trialled
/// 2026-09-26 and fully reverted: bitwise-exact Q and R (maxdiff 0e0,
/// same orthogonality/residuals) but 1.03x (tie hot, +20ms cold at
/// 2048, order-rotated). The per-panel serial prep (~1.2ms) barely fits
/// under the per-panel update remainder (~1.3ms), so scope + doubled
/// dispatch volume consume the prize, and worker prep lands on slow
/// cores cold. Do not retry overlap here; the Q-pass thin-GEMM rate
/// (m-thin 0.77-0.86x) is the better QR lever.
// Blocked-QR panel width (shipped 64 on 2026-09-27; 32 -> 64 wins
// ~6% at n = 512/2048 by halving dispatches and fattening panel
// GEMMs, 128 loses everywhere to serial panel cost — dgeqrf-2048
// 338 -> 317ms at 32T; nb = 128 falsified, do not raise blindly).
const QR_BLOCK_NB: usize = 64;
/// Minimum column count for the blocked path.
const QR_BLOCKED_MIN_COLS: usize = 64;

/// Length-set a fully-overwritten scratch buffer without re-zeroing.
///
/// `clear` + `resize` memsets the whole buffer every panel; the WY `W`
/// buffers and the gather panels are overwritten element-for-element
/// before any read (the `gemm_*_strided` hooks document `beta = 0`
/// overwrite; the gather loops write every panel element), so only
/// capacity growth zero-fills (safe-`Vec` length extension on a
/// fresh/short buffer) and shrinking truncates without touching memory.
/// The first call on a fresh workspace still faults pages on the GEMM /
/// gather writes, exactly as before — only the redundant memset goes
/// away (~15ms of `wbuf`/`w2` + ~3ms of `panel`/`pt` zeroing at n=2048,
/// `SONIUM_QR_PHASES` 2026-10-02). Bitwise-neutral: no read ever sees a
/// stale element.
///
/// Callers must overwrite all `n` elements before reading any of them.
/// `SONIUM_QR_SCRATCH_REUSE=0` restores `clear` + `resize` (diagnostic /
/// A-B and kill-switch).
fn reuse_scratch<T: ComplexField>(buf: &mut Vec<T>, n: usize) {
    if std::env::var("SONIUM_QR_SCRATCH_REUSE")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        buf.clear();
        buf.resize(n, T::zero());
        return;
    }
    if buf.len() < n {
        buf.resize(n, T::zero());
    } else {
        buf.truncate(n);
    }
}

/// Square-Q output steal for [`factorize_blocked_ws_direct`]: when `m == n`
/// the `m x m` Q accumulator already holds the thin Q bytes, so the output
/// takes over its storage instead of copying (`SONIUM_QR_SQUARE_Q=0`
/// restores the copy). Bitwise-identical bytes, one 33MB copy + alloc +
/// first-touch saved at n=2048 (~28ms, `SONIUM_QR_PHASES` 2026-10-02).
/// Tall shapes keep the thin copy. Note: stealing empties the workspace's
/// `qmat` capacity, so a `_with_workspace` repeat caller re-grows it next
/// call — still a net win (re-grow faults ~15ms once vs ~28ms copy saved
/// every call); the `factorize` path drops the workspace either way.
fn qr_square_q_on() -> bool {
    !std::env::var("SONIUM_QR_SQUARE_Q")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Fused overlay for [`factorize_blocked_ws_direct`]: one pass per panel
/// column writes the R block + reflectors to `a`, the GEMM panel copy,
/// the beta diagonal restore, and the panel zero-upper together, instead
/// of four sequential passes (`SONIUM_QR_FUSED_OVERLAY=0` restores the
/// sequential passes). Same final bytes in both buffers.
fn qr_fused_overlay_on() -> bool {
    !std::env::var("SONIUM_QR_FUSED_OVERLAY")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Fused overlay branch: one pass per panel column writes the R block +
/// reflectors to `a`, the GEMM panel copy, the beta diagonal restore, and
/// the panel zero-upper together. Must produce byte-identical `a`/`panel`
/// to [`overlay_panel_sequential`] (pinned by
/// `fused_overlay_matches_sequential`).
fn overlay_panel_fused<T: ComplexField>(
    a: &mut [T],
    n: usize,
    k0: usize,
    rows: usize,
    nb: usize,
    pt: &[T],
    betas: &[T],
    panel: &mut [T],
) {
    for j in 0..nb {
        let beta = betas[j];
        let pt_row = &pt[j * rows..(j + 1) * rows];
        // Strict-upper: R entries to `a`, zeros to the GEMM panel.
        for r in 0..j.min(rows) {
            a[(k0 + r) * n + (k0 + j)] = pt_row[r];
            panel[r * nb + j] = T::zero();
        }
        // Diagonal: beta (= R[j][j]) to `a`, explicit unit to panel.
        if j < rows {
            a[(k0 + j) * n + (k0 + j)] = beta;
            panel[j * nb + j] = pt_row[j];
        }
        // Strict-lower: reflectors to both.
        for r in j + 1..rows {
            let v = pt_row[r];
            a[(k0 + r) * n + (k0 + j)] = v;
            panel[r * nb + j] = v;
        }
    }
}

/// Sequential overlay branch: panel copy, `a` copy, beta diagonal
/// restore, then the panel zero-upper. Kill-switch twin of
/// [`overlay_panel_fused`]; same contract, four passes.
fn overlay_panel_sequential<T: ComplexField>(
    a: &mut [T],
    n: usize,
    k0: usize,
    rows: usize,
    nb: usize,
    pt: &[T],
    betas: &[T],
    panel: &mut [T],
) {
    for j in 0..nb {
        for r in 0..rows {
            panel[r * nb + j] = pt[j * rows + r];
        }
    }
    for j in 0..nb {
        for r in 0..rows {
            a[(k0 + r) * n + (k0 + j)] = panel[r * nb + j];
        }
    }
    // Restore R's diagonal: the panel copy carries explicit unit
    // diagonal entries (needed by the GEMM copies of V), while the
    // work matrix must hold beta = R[j][j].
    for j in 0..nb {
        a[(k0 + j) * n + (k0 + j)] = betas[j];
    }
    for j in 0..nb {
        for r in 0..j.min(rows) {
            panel[r * nb + j] = T::zero();
        }
    }
}

/// Fused triangular-factor finiteness for [`factorize_blocked_ws_direct`]:
/// each R row is checked while L1-hot right after its upper-copy instead of
/// a second full-matrix walk, and the strict-lower triangle is skipped
/// (`zeros` output, finite by construction, so the verdict matches
/// [`ensure_finite_slice`] exactly, including the `operation` name and the
/// orthogonal-before-triangular error precedence).
/// (`SONIUM_QR_FUSED_RCHECK=0` restores the separate walks). Same factors,
/// same errors, one 33MB cold stream removed (13/14 same-binary toggle
/// pairs, ~-2 to -6ms wall at n=2048, RAYON=16, 2026-10-02).
fn qr_fused_rcheck_on() -> bool {
    !std::env::var("SONIUM_QR_FUSED_RCHECK")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Copy R's upper triangle from the `n`-strided work matrix `a` into the
/// zeroed `n x n` output `rs` (the strict-lower keeps its zeros).
fn copy_r_upper<T: ComplexField>(rs: &mut [T], a: &[T], n: usize) {
    for i in 0..n {
        rs[i * n + i..i * n + n].copy_from_slice(&a[i * n + i..i * n + n]);
    }
}

/// Fused twin of [`copy_r_upper`] + [`ensure_finite_slice`]: each row is
/// checked L1-hot right after its copy, and the strict-lower triangle is
/// skipped (zeros by construction, finite unconditionally), so the verdict
/// and the `operation` name match the separate walk on every input.
fn copy_r_checked<T: ComplexField>(rs: &mut [T], a: &[T], n: usize) -> Result<(), QrError> {
    for i in 0..n {
        let dst = &mut rs[i * n + i..i * n + n];
        dst.copy_from_slice(&a[i * n + i..i * n + n]);
        if dst.iter().any(|value| !value.norm().is_finite()) {
            return Err(QrError::NumericalBreakdown {
                operation: "triangular factor",
            });
        }
    }
    Ok(())
}

/// Sub-block width for the blocked trailing update in [`panel_reflect`]:
/// each sub-block's reflectors are built serially (with in-sub-block
/// updates), then applied to the trailing columns in one parallel
/// region. Eight sub-blocks per 64-wide panel cut the ~2000 per-column
/// joins of a naive per-column fan-out to ~250 panel-lifetime joins,
/// amortizing scope dispatch over ~8x more work per join.
const REFLECT_BLOCK_K: usize = 8;

/// Minimum trailing columns for the parallel trailing update (see
/// [`panel_apply_trailing`]): fewer columns stay serial (fan-out
/// overhead exceeds the update).
const REFLECT_PAR_MIN_COLS: usize = 8;

/// Minimum panel rows (at the sub-block start) for the parallel
/// trailing update: short late panels stay serial.
const REFLECT_PAR_MIN_ROWS: usize = 512;

/// `SONIUM_QR_REFLECT_PAR=0` restores the fully-serial trailing update
/// in [`panel_reflect`] (diagnostic/A-B and kill-switch).
fn qr_reflect_par_on() -> bool {
    !std::env::var("SONIUM_QR_REFLECT_PAR")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Minimum panel rows for the two-phase [`wy_transform`] (see
/// [`wy_transform_two_phase`]): short late panels stay serial.
const WY_PAR_MIN_ROWS: usize = 512;

/// Minimum panel columns for the two-phase [`wy_transform`]: narrow
/// tail panels stay serial (too few dots per join).
const WY_PAR_MIN_COLS: usize = 16;

/// `SONIUM_QR_WY_PAR=0` restores the fully-serial [`wy_transform`]
/// (diagnostic/A-B and kill-switch).
fn qr_wy_par_on() -> bool {
    !std::env::var("SONIUM_QR_WY_PAR")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Minimum columns for the single-install funnel below: the scored
/// 2048 cell (and the 1024/4096 overhead cells) funnel; smaller shapes
/// keep the per-scope installs exactly as before.
const QR_FUNNEL_MIN_COLS: usize = 1024;

/// `SONIUM_QR_FUNNEL=0` restores per-scope pool installs (diagnostic /
/// A-B and kill-switch).
fn qr_funnel_on() -> bool {
    !std::env::var("SONIUM_QR_FUNNEL")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Blocked non-pivoted QR with explicit thin `Q`/`R` and rank detection.
/// `matrix` needs `rows >= columns` and finite entries (checked, same
/// contract as the MGS path).
/// Reusable scratch storage for blocked Householder QR factorization.
///
/// Keeping this workspace across factorizations removes the per-call
/// allocations and first-touch page faults on the work matrix (`a`),
/// the Q accumulator (`qmat`), and every panel scratch buffer. The
/// flop sequence is untouched, so results are bitwise identical to the
/// allocating path. Only the blocked path (`ncols >= QR_BLOCKED_MIN_COLS`)
/// uses it; narrow shapes keep the MGS path. Like [`LuWorkspace`], storage
/// grows on the first factorization and is retained after.
#[derive(Debug, Clone)]
pub struct QrWorkspace<T: ComplexField> {
    a: Vec<T>,
    qmat: Vec<T>,
    pt: Vec<T>,
    panel: Vec<T>,
    vtmp: Vec<T>,
    tmat: Vec<T>,
    tmat_t: Vec<T>,
    wbuf: Vec<T>,
    w2: Vec<T>,
    all_t: Vec<T>,
    betas: Vec<T>,
    taus: Vec<T>,
    vbuf: Vec<T>,
}

impl<T: ComplexField> Default for QrWorkspace<T> {
    fn default() -> Self {
        Self {
            a: Vec::new(),
            qmat: Vec::new(),
            pt: Vec::new(),
            panel: Vec::new(),
            vtmp: Vec::new(),
            tmat: Vec::new(),
            tmat_t: Vec::new(),
            wbuf: Vec::new(),
            w2: Vec::new(),
            all_t: Vec::new(),
            betas: Vec::new(),
            taus: Vec::new(),
            vbuf: Vec::new(),
        }
    }
}

impl<T: ComplexField> QrWorkspace<T> {
    /// Create an empty workspace. Storage grows on the first factorization
    /// and is retained for subsequent matrices.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Blocked QR factorization reusing caller-held scratch storage.
///
/// Bitwise identical to [`QrFactorization::factorize`] on the blocked path;
/// steady-state (same shape repeated) performs no allocations and touches
/// no fresh pages outside the owned `Q`/`R` outputs.
pub fn qr_factorize_with_workspace<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    workspace: &mut QrWorkspace<T>,
) -> Result<QrFactorization<T>, QrError> {
    if matrix.ncols() >= QR_BLOCKED_MIN_COLS {
        return factorize_blocked_ws(matrix, workspace);
    }
    QrFactorization::factorize(matrix)
}

/// Maximum retained storage per thread-local workspace: the sum of all
/// buffer capacities, in bytes. Bounds per-thread waste after one huge
/// factorization — the retained footprint of the bench shapes fits (2048²
/// f64 ≈ 36MB including scratch, 4096² ≈ 140MB with the stolen `qmat`
/// already gone), while an 8192² work matrix alone (512MB) trips the cap
/// and is dropped, re-grown on the next call. Shape-keyed retention only:
/// capacities are reused, every element is overwritten before any read
/// (see `reuse_scratch` and the per-call clears), so retained bytes never
/// affect results.
const QR_RETAIN_MAX_BYTES: usize = 256 << 20;

/// `SONIUM_QR_RETAIN_WS=0` restores a fresh workspace per factorization
/// on the allocating blocked path (diagnostic / A-B and kill-switch).
fn qr_retain_ws_on() -> bool {
    !std::env::var("SONIUM_QR_RETAIN_WS")
        .map(|v| v == "0")
        .unwrap_or(false)
}

thread_local! {
    /// Retained allocating-path workspaces, one slot per scalar type that
    /// has factorized on this thread. `ComplexField: 'static` lets the
    /// generic path key slots by `TypeId` with no API change and no
    /// per-type statics; each slot holds only scratch capacities, never
    /// input-keyed values. Perf-only state: dropping the map (cap trip,
    /// reentrancy, thread teardown) only costs re-allocation.
    static QR_RETAINED: RefCell<HashMap<TypeId, Box<dyn Any>>> =
        RefCell::new(HashMap::new());
}

/// Sum of buffer capacities in `ws`, in bytes (saturating).
fn qr_workspace_bytes<T: ComplexField>(ws: &QrWorkspace<T>) -> usize {
    let elem = std::mem::size_of::<T>();
    let elems =
        ws.a.capacity()
            .saturating_add(ws.qmat.capacity())
            .saturating_add(ws.pt.capacity())
            .saturating_add(ws.panel.capacity())
            .saturating_add(ws.vtmp.capacity())
            .saturating_add(ws.tmat.capacity())
            .saturating_add(ws.tmat_t.capacity())
            .saturating_add(ws.wbuf.capacity())
            .saturating_add(ws.w2.capacity())
            .saturating_add(ws.all_t.capacity())
            .saturating_add(ws.betas.capacity())
            .saturating_add(ws.taus.capacity())
            .saturating_add(ws.vbuf.capacity());
    elems.saturating_mul(elem)
}

/// Run `op` on this thread's retained workspace for `T`, enforcing the
/// [`QR_RETAIN_MAX_BYTES`] cap afterwards. Returns `None` under
/// same-thread reentrancy or thread teardown (the caller falls back to a
/// fresh workspace); never panics.
fn with_retained_workspace<T: ComplexField, R>(
    op: impl FnOnce(&mut QrWorkspace<T>) -> R,
) -> Option<R> {
    QR_RETAINED
        .try_with(|cell| {
            let mut map = cell.try_borrow_mut().ok()?;
            let (out, over_cap) = {
                let slot = map
                    .entry(TypeId::of::<T>())
                    .or_insert_with(|| Box::new(QrWorkspace::<T>::new()));
                // The slot is keyed by `T`, so this downcast always
                // succeeds; `None` would only fall back to a fresh
                // workspace, never to a wrong-typed one.
                let ws = slot.downcast_mut::<QrWorkspace<T>>()?;
                let out = op(ws);
                let over_cap = qr_workspace_bytes(ws) > QR_RETAIN_MAX_BYTES;
                (out, over_cap)
            };
            if over_cap {
                map.remove(&TypeId::of::<T>());
            }
            Some(out)
        })
        .ok()
        .flatten()
}

fn factorize_blocked<T: ComplexField>(
    matrix: &DenseMatrix<T>,
) -> Result<QrFactorization<T>, QrError> {
    // Thread-local retained workspace (default; `SONIUM_QR_RETAIN_WS=0`
    // restores the fresh workspace): repeat factorizations on a thread
    // reuse the work-matrix and scratch capacities instead of
    // re-allocating + re-faulting ~40MB per call at n=2048. The square-Q
    // steal still empties `qmat` every call (conservation: one buffer
    // stolen per call means one allocated per call in steady state, so no
    // pooling scheme can retain it — ping-pong included), and the `R`
    // output stays caller-owned; retention covers `a` + panel scratch.
    if qr_retain_ws_on()
        && let Some(out) = with_retained_workspace(|ws| factorize_blocked_ws(matrix, ws))
    {
        return out;
    }
    factorize_blocked_ws(matrix, &mut QrWorkspace::new())
}

/// Tiny-pool funnel (shipped default-on 2026-09-27, `SONIUM_QR_TINY=0`
/// opt-out): QR has no funnel today (cold caller + nested hops on every
/// GEMM scope), so this is both a temperature fix and hop elimination.
/// One install; nested scopes inline at 16-wide sizings. Covers
/// `factorize`/`factorize_with_workspace` (both funnel through here);
/// nested calls run inline. Measured OB-free order-rotated (f64):
/// 1.27-1.29x at 512, 1.47-1.58x at 1024, 1.10-1.13x at 2048 — and
/// BITWISE-identical Q/R at all sizes (QR's GEMM splits align across
/// pools, unlike LU/Cholesky band re-chunking). Envelope: f64-only
/// via [`ComplexField::qr_prefers_tiny_pool`], ncols>=512 with 64MB
/// cap, inert at global<=16.
fn factorize_blocked_ws<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    workspace: &mut QrWorkspace<T>,
) -> Result<QrFactorization<T>, QrError> {
    let est = matrix.nrows() as u64 * matrix.ncols() as u64 * std::mem::size_of::<T>() as u64;
    if !std::env::var("SONIUM_QR_TINY")
        .map(|v| v == "0")
        .unwrap_or(false)
        && T::qr_prefers_tiny_pool()
        && matrix.ncols() >= 512
        && rayon::current_num_threads() > 16
        && est <= (64 << 20)
    {
        return sonium_compute::cpu::blas3::run_tiny(|| {
            factorize_blocked_ws_direct(matrix, workspace)
        });
    }
    // Single-install funnel (shipped default-on 2026-10-02,
    // `SONIUM_QR_FUNNEL=0` opt-out): one mirror-pool install per
    // factorization instead of a nested hop per GEMM/reflect/WY scope
    // (~470 at n=2048). Same pool the per-scope installs would use, so
    // task geometry is unchanged — nested scopes just inline. Without
    // it the parallel reflect/WY regions regress (~+15ms at n=2048,
    // unpinned): the caller bounces in/out of the pool hundreds of
    // times and lands on parked/low-frequency cores for the serial
    // phases (gather/overlay/untimed tail slow ~1.5x while pool phases
    // stay flat; pinning the process recovers the win). With it the
    // caller stays put and the parallel regions pay only their joins.
    if qr_funnel_on() && matrix.ncols() >= QR_FUNNEL_MIN_COLS {
        return sonium_compute::cpu::blas3::run_mirror(|| {
            factorize_blocked_ws_direct(matrix, workspace)
        });
    }
    factorize_blocked_ws_direct(matrix, workspace)
}

fn factorize_blocked_ws_direct<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    workspace: &mut QrWorkspace<T>,
) -> Result<QrFactorization<T>, QrError> {
    validate_qr_input(matrix)?;
    let m = matrix.nrows();
    let n = matrix.ncols();
    let src = matrix.as_slice();
    let QrWorkspace {
        a,
        qmat,
        pt,
        panel,
        vtmp,
        tmat,
        tmat_t,
        wbuf,
        w2,
        all_t,
        betas,
        taus,
        vbuf,
    } = workspace;
    // Owned row-major work matrix; reflectors persist below the diagonal
    // (LAPACK convention) for the backward Q accumulation. Retained
    // capacity across calls: copy, don't reallocate.
    a.clear();
    a.extend_from_slice(src);
    // Compact panel transpose (nb x rows), reflector temp, WY T, W buffers.
    tmat.clear();
    tmat.resize(QR_BLOCK_NB * QR_BLOCK_NB, T::zero());
    // Retained WY blocks, one full QR_BLOCK_NB² tile per panel in panel
    // order: the backward Q pass reuses the forward T bit-for-bit instead
    // of recomputing it (same V in, same flops, ~130M serial Level-1
    // flops saved at n=2048). Cleared per factorization: stale tiles from
    // a previous (possibly differently sized) call must never leak into
    // the Q pass indexing below.
    all_t.clear();
    // Phase timers (diagnostic, `SONIUM_QR_PHASES` set prints sums to
    // stderr). Same shape as `SONIUM_LU_PHASES`/`SONIUM_CHOL_PHASES`.
    let time_phases = std::env::var("SONIUM_QR_PHASES").is_ok();
    let (mut t_gather, mut t_reflect, mut t_tform, mut t_overlay) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );
    let (mut t_wy, mut t_qgather, mut t_qwy) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );
    let mut k0 = 0;
    while k0 < n {
        let nb = (n - k0).min(QR_BLOCK_NB);
        let rows = m - k0;
        // Compact panel + its transpose (reflector j = row j of `pt`).
        let t0 = std::time::Instant::now();
        // Fully overwritten by the gather loops below: skip the re-zero.
        reuse_scratch(panel, rows * nb);
        reuse_scratch(pt, nb * rows);
        for j in 0..nb {
            for r in 0..rows {
                let v = a[(k0 + r) * n + (k0 + j)];
                panel[r * nb + j] = v;
                pt[j * rows + r] = v;
            }
        }
        t_gather += t0.elapsed();
        // Fully overwritten by `panel_reflect` (`vtmp[..len]` per column,
        // every `taus`/`betas` entry on both branches): skip the re-zero.
        reuse_scratch(vtmp, rows);
        reuse_scratch(taus, nb);
        reuse_scratch(betas, nb);
        let t0 = std::time::Instant::now();
        panel_reflect(pt, rows, nb, vtmp, taus, betas)?;
        t_reflect += t0.elapsed();
        let t0 = std::time::Instant::now();
        wy_transform(pt, rows, nb, taus, tmat)?;
        t_tform += t0.elapsed();
        all_t.extend_from_slice(tmat);
        let t0 = std::time::Instant::now();
        // Overlay R block (upper triangle) + reflectors (strict lower).
        // The reflector columns of `panel` are refreshed from `pt` rows
        // (the factorization ran on the transpose).
        //
        // Fused form (default; `SONIUM_QR_FUSED_OVERLAY=0` restores the
        // sequential passes below): one pass per column writes the R
        // block + reflectors to `a`, the GEMM panel copy, the beta
        // diagonal restore, and the panel zero-upper together. The
        // sequential form wrote every panel element twice plus a third
        // zero-upper pass; the fused form writes each destination once
        // with its final value. Same final bytes in both buffers.
        //
        // The GEMM copy of V must have a zero strict-upper: the panel
        // carries the R block above the diagonal, but the compact-WY
        // identity needs true Householder vectors (LAPACK's dlarfb never
        // reads those entries: it applies V1 through lower-triangular
        // DTRMM). Feeding R entries as V silently breaks orthogonality.
        // The R block reaches `a` in the same pass, so `a` keeps it.
        //
        // Overlay direction (row-outer trialled 2026-09-26, reverted):
        // ntmicro shows 303 -> 38us per 2048x32 panel for row-outer
        // (contiguous both sides, vectorized), but interleaved
        // order-rotated A/B in-situ ties within noise (j-outer lines
        // stay hot in L3 across panels) — no measurable win, so the
        // original j-outer stands. Bitwise identical either way.
        if qr_fused_overlay_on() {
            overlay_panel_fused(a, n, k0, rows, nb, pt, betas, panel);
        } else {
            overlay_panel_sequential(a, n, k0, rows, nb, pt, betas, panel);
        }
        t_overlay += t0.elapsed();
        // Trailing update on the big matrix (cols k0+nb..n), V from the
        // retained compact panel (disjoint from `a`: no aliasing). The
        // middle factor here is Tᴴ, not T: sequential application hits the
        // trailing matrix as H_last···H_first, the reverse-order product,
        // which the compact form reaches through the conjugate-transpose
        // (this is LAPACK dlarfb's Forward/Backward distinction: the same
        // T is applied transposed for the trailing update). The Q pass
        // below keeps T as-is.
        let trailing = n - (k0 + nb);
        if trailing > 0 {
            // Fully overwritten by the conjugate-transpose below.
            reuse_scratch(tmat_t, nb * nb);
            for i in 0..nb {
                for j in 0..nb {
                    tmat_t[j * nb + i] = tmat[i * QR_BLOCK_NB + j].conj();
                }
            }
            let t0 = std::time::Instant::now();
            wy_update(
                panel,
                nb,
                rows,
                nb,
                tmat_t,
                nb,
                a,
                k0,
                k0 + nb,
                trailing,
                n,
                wbuf,
                w2,
            );
            t_wy += t0.elapsed();
        }
        k0 += nb;
    }

    // Backward Q accumulation: Q = I, then Q -= V·(T·(Vᴴ·Q)) per panel.
    // Retained capacity: refill the identity in place.
    qmat.clear();
    qmat.resize(m * m, T::zero());
    for i in 0..m {
        qmat[i * m + i] = T::one();
    }
    let mut k0 = ((n - 1) / QR_BLOCK_NB) * QR_BLOCK_NB;
    loop {
        let nb = (n - k0).min(QR_BLOCK_NB);
        let rows = m - k0;
        // Cached forward T for this panel (k0 is always a multiple of
        // QR_BLOCK_NB on this loop, and every non-tail panel has full
        // width, so panel index k0/QR_BLOCK_NB lines up with push order).
        let tblock = &all_t[k0 / QR_BLOCK_NB * QR_BLOCK_NB * QR_BLOCK_NB..];
        // Q panel rows k0..m (all m cols): Q -= V·(T·(Vᴴ·Q)).
        // V lives in the work matrix (stride n); gather the panel block
        // into `vbuf` once (the GEMM hook needs a strided slice it can
        // address, and the work-matrix rows are longer than nb). Retained
        // capacity across panels and calls: clear, don't reallocate.
        let t0 = std::time::Instant::now();
        vbuf.clear();
        for r in 0..rows {
            let s = (k0 + r) * n + k0;
            vbuf.extend_from_slice(&a[s..s + nb]);
        }
        // Same zero strict-upper as the forward panel: the work matrix
        // carries the R block above the diagonal here. The diagonal holds
        // R's beta: restore V's explicit unit diagonal.
        for j in 0..nb {
            for r in 0..j.min(rows) {
                vbuf[r * nb + j] = T::zero();
            }
            if j < rows {
                vbuf[j * nb + j] = T::one();
            }
        }
        t_qgather += t0.elapsed();
        let t0 = std::time::Instant::now();
        wy_update_q::<T>(vbuf, rows, nb, tblock, qmat, k0, m, wbuf, w2);
        t_qwy += t0.elapsed();
        if k0 == 0 {
            break;
        }
        k0 -= QR_BLOCK_NB;
    }
    if time_phases {
        eprintln!(
            "qr phases gather {:.3}ms reflect {:.3}ms tform {:.3}ms overlay {:.3}ms wy {:.3}ms qgather {:.3}ms qwy {:.3}ms",
            t_gather.as_secs_f64() * 1000.0,
            t_reflect.as_secs_f64() * 1000.0,
            t_tform.as_secs_f64() * 1000.0,
            t_overlay.as_secs_f64() * 1000.0,
            t_wy.as_secs_f64() * 1000.0,
            t_qgather.as_secs_f64() * 1000.0,
            t_qwy.as_secs_f64() * 1000.0
        );
    }
    // Thin Q (first n cols), upper R, rank scan with the MGS tolerance.
    // Square shapes steal the accumulator storage instead of copying
    // (see `qr_square_q_on`); tall shapes copy the thin block as before.
    let square_q = m == n && qr_square_q_on();
    let mut q = if square_q {
        DenseMatrix::from_row_major_vec(m, n, std::mem::take(&mut *qmat))
    } else {
        DenseMatrix::zeros(m, n)
    };
    // Fused column-norm state (LAPACK dlassq form, one scale/sum-squares
    // pair per column): the thin-Q copy below already streams the input
    // rows, so accumulation rides along prefetch-friendly. A separate
    // column pass would re-read the matrix strided (16KB jumps over the
    // L3-cold 33MB work matrix, ~50ms at n=2048). Same update equations
    // as `checked_norm` (overflow-safe by construction); only the
    // summation order differs (row-outer here), so norms agree up to
    // rounding.
    let mut col_scale = vec![T::Real::zero(); n];
    let mut col_sums = vec![T::Real::one(); n];
    let mut col_finite = true;
    let qs = q.as_mut_slice();
    // Fused SIMD row update (f32/f64 hook routes to the AVX2 kernel,
    // bitwise-identical to the scalar equations); opt out with
    // SONIUM_QR_DLASQ_SCALAR=1 to restore the scalar loop below.
    let dlassq_scalar = std::env::var_os("SONIUM_QR_DLASQ_SCALAR").is_some();
    for r in 0..m {
        if !square_q {
            qs[r * n..(r + 1) * n].copy_from_slice(&qmat[r * m..r * m + n]);
        }
        if !dlassq_scalar {
            if !T::dlassq_row_update(&src[r * n..(r + 1) * n], &mut col_scale, &mut col_sums) {
                col_finite = false;
            }
            continue;
        }
        for c in 0..n {
            let magnitude = src[r * n + c].norm();
            if !magnitude.is_finite() {
                col_finite = false;
            } else if magnitude != T::Real::zero() {
                if col_scale[c] < magnitude {
                    let ratio = col_scale[c] / magnitude;
                    col_sums[c] = T::Real::one() + col_sums[c] * ratio * ratio;
                    col_scale[c] = magnitude;
                } else {
                    let ratio = magnitude / col_scale[c];
                    col_sums[c] += ratio * ratio;
                }
            }
        }
    }
    let mut rmat = DenseMatrix::zeros(n, n);
    if qr_fused_rcheck_on() {
        // Q first: preserves the legacy orthogonal-before-triangular error
        // precedence when both factors are non-finite.
        ensure_finite_slice("orthogonal factor", q.as_slice())?;
        copy_r_checked(rmat.as_mut_slice(), a, n)?;
    } else {
        copy_r_upper(rmat.as_mut_slice(), a, n);
        ensure_finite_slice("orthogonal factor", q.as_slice())?;
        ensure_finite_slice("triangular factor", rmat.as_slice())?;
    }
    // Largest input column norm (same tolerance formula as MGS), finalized
    // from the fused accumulators above.
    if !col_finite {
        return Err(QrError::NumericalBreakdown {
            operation: "initial column norm",
        });
    }
    let mut largest_norm = T::Real::zero();
    for c in 0..n {
        let norm = if col_scale[c] == T::Real::zero() {
            T::Real::zero()
        } else {
            col_scale[c] * col_sums[c].sqrt()
        };
        if !norm.is_finite() {
            return Err(QrError::NumericalBreakdown {
                operation: "initial column norm",
            });
        }
        largest_norm = largest_norm.max(norm);
    }
    let dimension_scale = T::Real::from_usize(m.max(n)).ok_or(QrError::NumericalBreakdown {
        operation: "dimension conversion",
    })?;
    let tolerance = T::Real::epsilon() * dimension_scale * largest_norm.max(T::Real::one());
    if !tolerance.is_finite() {
        return Err(QrError::NumericalBreakdown {
            operation: "rank tolerance",
        });
    }
    let mut rank = n;
    for i in 0..n {
        if rmat.as_slice()[i * n + i].norm() <= tolerance {
            rank = i;
            break;
        }
    }
    if rank != n {
        return Err(QrError::RankDeficient { rank, columns: n });
    }
    Ok(QrFactorization {
        rows: m,
        columns: n,
        q,
        r: rmat,
        permutation: (0..n).collect(),
        rank,
    })
}

/// Unblocked Householder on the transposed compact panel `pt` (`nb` rows
/// of length `rows`; row `j` holds matrix column `k0+j` over panel rows).
/// Reflectors accumulate in place with explicit unit diagonals; `taus`
/// alongside. Uses the vectorized `vec_dot`/`vec_axpy` hooks on the long
/// contiguous rows.
fn panel_reflect<T: ComplexField>(
    pt: &mut [T],
    rows: usize,
    nb: usize,
    vtmp: &mut [T],
    taus: &mut [T],
    betas: &mut [T],
) -> Result<(), QrError> {
    if !qr_reflect_par_on() {
        return panel_reflect_serial(pt, rows, nb, vtmp, taus, betas);
    }
    panel_reflect_parallel(pt, rows, nb, vtmp, taus, betas)
}

/// Blocked parallel [`panel_reflect`] (see `panel_apply_trailing`).
/// Separate entry point so parity tests can force this branch
/// regardless of ambient `SONIUM_QR_REFLECT_PAR` state.
fn panel_reflect_parallel<T: ComplexField>(
    pt: &mut [T],
    rows: usize,
    nb: usize,
    vtmp: &mut [T],
    taus: &mut [T],
    betas: &mut [T],
) -> Result<(), QrError> {
    // Blocked form: each sub-block builds its reflectors serially (with
    // in-sub-block updates), then applies them to the trailing columns
    // in one parallel region (see `panel_apply_trailing`). Per-column op
    // order matches the serial loop exactly, so bytes are identical.
    let mut j0 = 0;
    while j0 < nb {
        let j1 = (j0 + REFLECT_BLOCK_K).min(nb);
        for j in j0..j1 {
            let tau = panel_reflector_build(pt, rows, j, vtmp, taus, betas)?;
            if tau == T::zero() {
                continue;
            }
            for c in (j + 1)..j1 {
                panel_apply_one(pt, rows, j, c, tau)?;
            }
        }
        if j1 < nb {
            panel_apply_trailing(pt, rows, j0, j1, nb, taus)?;
        }
        j0 = j1;
    }
    Ok(())
}

/// Fully-serial [`panel_reflect`] (original loop, helper-extracted):
/// the `SONIUM_QR_REFLECT_PAR=0` fallback and the small-panel route.
fn panel_reflect_serial<T: ComplexField>(
    pt: &mut [T],
    rows: usize,
    nb: usize,
    vtmp: &mut [T],
    taus: &mut [T],
    betas: &mut [T],
) -> Result<(), QrError> {
    for j in 0..nb {
        let tau = panel_reflector_build(pt, rows, j, vtmp, taus, betas)?;
        // Apply to trailing rows c in j+1..nb (contiguous, vectorized).
        if tau == T::zero() {
            continue;
        }
        for c in (j + 1)..nb {
            panel_apply_one(pt, rows, j, c, tau)?;
        }
    }
    Ok(())
}

/// Build reflector `j` of the transposed panel: norm, phase-rotated
/// beta, tau, and the unit-diagonal `v` written back to row `j`.
/// Returns tau (`T::zero()` when the column is already zero below the
/// diagonal and the reflector is the identity).
fn panel_reflector_build<T: ComplexField>(
    pt: &mut [T],
    rows: usize,
    j: usize,
    vtmp: &mut [T],
    taus: &mut [T],
    betas: &mut [T],
) -> Result<T, QrError> {
    let len = rows - j;
    let row_j = &pt[j * rows + j..(j + 1) * rows];
    // Fast norm (vectorized dot); robust fallback on overflow, same
    // contract spirit as the MGS checked norms.
    let dot: T = T::vec_dot(row_j, row_j);
    let sq = dot.norm();
    let mut norm = sq.sqrt();
    if !norm.is_finite() {
        norm = checked_norm(row_j, "panel reflector norm")?;
    }
    if norm == T::Real::zero() {
        taus[j] = T::zero();
        // Column already zero below the diagonal: R[j][j] is zero and
        // the reflector is the identity.
        betas[j] = T::zero();
        return Ok(T::zero());
    }
    let alpha = pt[j * rows + j];
    // Phase-rotated beta (LAPACK zlarfg convention; real: -copysign).
    let abs_alpha = alpha.norm();
    let beta = if abs_alpha == T::Real::zero() {
        T::from_real(-norm)
    } else {
        T::from_real(-norm) * (alpha / T::from_real(abs_alpha))
    };
    let tau = (beta - alpha) / beta;
    ensure_finite_value("reflector scale", tau)?;
    // The diagonal holds beta (= R[j][j]) in LAPACK storage, but the
    // compact panel keeps the unit diagonal explicit for the GEMM
    // copies, so beta is retained out-of-band for the overlay.
    betas[j] = beta;
    // v into the temp (unit diagonal first), then back to the row.
    vtmp[0] = T::one();
    vtmp[1..len].copy_from_slice(&row_j[1..len]);
    let denom = alpha - beta;
    for v in vtmp[1..len].iter_mut() {
        *v /= denom;
    }
    ensure_finite_slice("reflector vector", &vtmp[..len])?;
    pt[j * rows + j..(j + 1) * rows].copy_from_slice(&vtmp[..len]);
    taus[j] = tau;
    Ok(tau)
}

/// Apply reflector `j` (`v` in row `j` from `j`, unit diagonal) to panel
/// row `c`: `col -= tau·v·(vᴴ·col)`.
#[inline]
fn panel_apply_one<T: ComplexField>(
    pt: &mut [T],
    rows: usize,
    j: usize,
    c: usize,
    tau: T,
) -> Result<(), QrError> {
    debug_assert!(j < c);
    let (head, tail) = pt.split_at_mut((j + 1) * rows);
    let v = &head[j * rows + j..(j + 1) * rows];
    let col = &mut tail[(c - (j + 1)) * rows + j..(c - j) * rows];
    let w = tau * T::vec_dot(v, col);
    ensure_finite_value("panel update coefficient", w)?;
    T::vec_axpy(-w, v, col);
    Ok(())
}

/// Apply sub-block reflectors `j0..j1` to the trailing panel rows
/// `j1..nb`, in ascending reflector order per column (the serial op
/// order, so bytes match). Large sub-blocks fan out over static
/// trailing-column chunks through the mirror pool; small ones run the
/// serial double loop (same per-column order). Zero-tau reflectors are
/// skipped up front, preserving the serial `continue` semantics.
fn panel_apply_trailing<T: ComplexField>(
    pt: &mut [T],
    rows: usize,
    j0: usize,
    j1: usize,
    nb: usize,
    taus: &[T],
) -> Result<(), QrError> {
    let mut active = [0usize; REFLECT_BLOCK_K];
    let mut nactive = 0;
    for (k, tau) in taus.iter().enumerate().take(j1).skip(j0) {
        if *tau != T::zero() {
            active[nactive] = k;
            nactive += 1;
        }
    }
    if nactive == 0 {
        return Ok(());
    }
    let trailing = nb - j1;
    if trailing < REFLECT_PAR_MIN_COLS || rows - j0 < REFLECT_PAR_MIN_ROWS {
        for &k in &active[..nactive] {
            for c in j1..nb {
                panel_apply_one(pt, rows, k, c, taus[k])?;
            }
        }
        return Ok(());
    }
    use crate::banded_schur::SendMutPtr;
    use std::sync::atomic::{AtomicBool, Ordering};
    let base = SendMutPtr::new(pt.as_mut_ptr());
    let workers = rayon::current_num_threads().max(1);
    let chunk = trailing.div_ceil(workers).max(1);
    let failed = AtomicBool::new(false);
    let failed_ref = &failed;
    sonium_compute::cpu::blas3::scope_pinned(|s| {
        let mut c0 = j1;
        while c0 < nb {
            let c1 = (c0 + chunk).min(nb);
            s.spawn(move |_| {
                for c in c0..c1 {
                    for &k in &active[..nactive] {
                        // SAFETY: reflector rows `j0..j1` are read-only
                        // and disjoint from trailing rows `j1..nb`; this
                        // task owns columns `c0..c1` exclusively (static
                        // chunk split, no overlap across tasks). Lengths
                        // `rows - k` match the serial indexing (row `k`
                        // from `k`, row `c` from `k`).
                        let v =
                            unsafe { std::slice::from_raw_parts(base.add(k * rows + k), rows - k) };
                        let col = unsafe {
                            std::slice::from_raw_parts_mut(base.add(c * rows + k), rows - k)
                        };
                        let w = taus[k] * T::vec_dot(v, col);
                        if !w.norm().is_finite() {
                            failed_ref.store(true, Ordering::Relaxed);
                        }
                        T::vec_axpy(-w, v, col);
                    }
                }
            });
            c0 = c1;
        }
    });
    if failed.load(Ordering::Relaxed) {
        return Err(QrError::NumericalBreakdown {
            operation: "panel update coefficient",
        });
    }
    Ok(())
}

/// Compact-WY `T` (upper `nb × nb` in the leading block of `tmat`, row
/// stride `QR_BLOCK_NB`) from the transposed panel `pt` and `taus`.
/// Forward recurrence: `T[i,j] = -tau_j·Σ_{k=i}^{j-1} T[i,k]·z[k]` with
/// `z[k] = V[:,k]ᴴ·v_j`, `T[j,j] = tau_j`; all sources are final when
/// read (columns `< j` are complete).
fn wy_transform<T: ComplexField>(
    pt: &[T],
    rows: usize,
    nb: usize,
    taus: &[T],
    tmat: &mut [T],
) -> Result<(), QrError> {
    if qr_wy_par_on() && rows >= WY_PAR_MIN_ROWS && nb >= WY_PAR_MIN_COLS {
        return wy_transform_two_phase(pt, rows, nb, taus, tmat);
    }
    wy_transform_serial(pt, rows, nb, taus, tmat)
}

/// Fully-serial [`wy_transform`] (original loop, helper-extracted):
/// the `SONIUM_QR_WY_PAR=0` fallback and the small-panel route.
fn wy_transform_serial<T: ComplexField>(
    pt: &[T],
    rows: usize,
    nb: usize,
    taus: &[T],
    tmat: &mut [T],
) -> Result<(), QrError> {
    for v in tmat.iter_mut() {
        *v = T::zero();
    }
    // One scratch row for z (nb <= QR_BLOCK_NB): a single small buffer per
    // panel, reused across columns.
    let mut zbuf = vec![T::zero(); nb];
    for j in 0..nb {
        let tau = taus[j];
        if tau == T::zero() {
            continue;
        }
        // z[k] = V[:,k]ᴴ·v_j once per k (was recomputed per i: O(nb³/6)
        // dots per panel instead of O(nb²/2) — 10x redundant at nb=32).
        // Same dot inputs and order, so values are bitwise identical.
        for (k, slot) in zbuf[..j].iter_mut().enumerate() {
            *slot = zk(k, j, pt, rows)?;
        }
        wy_column(tmat, &zbuf[..j], taus, j)?;
    }
    Ok(())
}

/// Two-phase [`wy_transform`]: every `z[k,j] = V[:,k]ᴴ·v_j` dot reads
/// only final reflector data, so all `nb²/2` dots run up front in one
/// parallel region (static chunks over `j`), then the tiny serial
/// recurrence consumes the rows. Same dots in the same per-dot order
/// and the same recurrence, so bytes match the serial loop. Breakdown
/// reporting matches serial exactly too: the dot phase stores every `z`
/// unchecked, and the recurrence re-validates each consumed row in
/// serial column order — so a non-finite dot in a skipped zero-tau
/// column is never observed, and an early accumulation failure wins
/// over a later dot failure, exactly as in [`wy_transform_serial`].
fn wy_transform_two_phase<T: ComplexField>(
    pt: &[T],
    rows: usize,
    nb: usize,
    taus: &[T],
    tmat: &mut [T],
) -> Result<(), QrError> {
    for v in tmat.iter_mut() {
        *v = T::zero();
    }
    use crate::banded_schur::{SendConstPtr, SendMutPtr};
    let cbase = SendConstPtr::new(pt.as_ptr());
    let mut zmat = vec![T::zero(); nb * nb];
    let zbase = SendMutPtr::new(zmat.as_mut_ptr());
    let workers = rayon::current_num_threads().max(1);
    let chunk = nb.saturating_sub(1).div_ceil(workers).max(1);
    sonium_compute::cpu::blas3::scope_pinned(|s| {
        let mut c0 = 1;
        while c0 < nb {
            let c1 = (c0 + chunk).min(nb);
            s.spawn(move |_| {
                for j in c0..c1 {
                    for k in 0..j {
                        // SAFETY: `pt` is read-only and shared; this task
                        // owns `zmat` rows `c0..c1` exclusively (static
                        // chunk split, no overlap across tasks; each row
                        // writes only its `k < j` prefix). Lengths
                        // `rows - j` match [`zk`] exactly.
                        let vk = unsafe {
                            std::slice::from_raw_parts(cbase.add(k * rows + j), rows - j)
                        };
                        let vj = unsafe {
                            std::slice::from_raw_parts(cbase.add(j * rows + j), rows - j)
                        };
                        let z = T::vec_dot(vk, vj);
                        unsafe {
                            *zbase.add(j * nb + k) = z;
                        }
                    }
                }
            });
            c0 = c1;
        }
    });
    for j in 0..nb {
        if taus[j] == T::zero() {
            continue;
        }
        // Ordered re-validation: same dots-then-accumulation failure order
        // as the serial loop (see [`zk`).
        let row = &zmat[j * nb..j * nb + j];
        for &z in row {
            ensure_finite_value("WY inner product", z)?;
        }
        wy_column(tmat, row, taus, j)?;
    }
    Ok(())
}

/// Recurrence column `j` of the compact-WY `T`: `T[i,j] =
/// -tau_j·Σ_{k=i}^{j-1} T[i,k]·z[k]` over the final leading block, with
/// `T[j,j] = tau_j`. Shared by the serial and two-phase transforms.
fn wy_column<T: ComplexField>(
    tmat: &mut [T],
    z: &[T],
    taus: &[T],
    j: usize,
) -> Result<(), QrError> {
    let tau = taus[j];
    for i in 0..j {
        // No tau_i skip: a zero-tau row still carries coupling from
        // later columns (T[i,k], k>i), exactly as LAPACK's DTRMV over
        // the full leading block computes. (T[i,i] itself is zero, so
        // the k=i term contributes nothing.)
        let mut acc = T::zero();
        for k in i..j {
            acc += tmat[i * QR_BLOCK_NB + k] * z[k];
        }
        ensure_finite_value("WY accumulation", acc)?;
        tmat[i * QR_BLOCK_NB + j] = T::from_real(T::Real::zero() - T::Real::one()) * tau * acc;
    }
    tmat[j * QR_BLOCK_NB + j] = tau;
    Ok(())
}

/// Helper: `z_k = V[:,k]ᴴ·v_j` over panel rows `j..` (contiguous rows of
/// the transposed panel). Split out so the recurrence reads clearly.
fn zk<T: ComplexField>(k: usize, j: usize, pt: &[T], rows: usize) -> Result<T, QrError> {
    let vk = &pt[k * rows + j..(k + 1) * rows];
    let vj = &pt[j * rows + j..(j + 1) * rows];
    let z = T::vec_dot(vk, vj);
    ensure_finite_value("WY inner product", z)?;
    Ok(z)
}

/// Trailing update `A_rows -= V·(U·(Vᴴ·A_rows))` with zero copies: `v` is
/// the reflector block (`rows × nb`, row stride `v_stride`), `a` the
/// target rows at columns `a_col0..a_col0+ncols` with row stride `lda`.
/// `umat` is the `nb × nb` middle factor with row stride `t_stride`.
/// Three threaded GEMM calls through the strided hooks; `wbuf`/`w2` are
/// reused `nb × ncols` scratch.
#[allow(clippy::too_many_arguments)]
fn wy_update<T: ComplexField>(
    v: &[T],
    v_stride: usize,
    rows: usize,
    nb: usize,
    umat: &[T],
    t_stride: usize,
    a: &mut [T],
    a_row0: usize,
    a_col0: usize,
    ncols: usize,
    lda: usize,
    wbuf: &mut Vec<T>,
    w2: &mut Vec<T>,
) {
    if ncols == 0 || rows == 0 {
        return;
    }
    // Fully overwritten by the `beta = 0` GEMM hooks below: skip the
    // per-panel re-zero (the dominant `walloc` term at n=2048).
    reuse_scratch(wbuf, nb * ncols);
    reuse_scratch(w2, nb * ncols);
    let a_off = a_row0 * lda + a_col0;
    // W = Vᴴ · A_panel.
    T::gemm_hermitian_strided(nb, ncols, rows, v, v_stride, &a[a_off..], lda, wbuf, ncols);
    // W2 = U · W.
    T::gemm_plain_strided(nb, ncols, nb, umat, t_stride, wbuf, ncols, w2, ncols);
    // A_panel -= V · W2.
    T::schur_sub_strided(
        rows,
        ncols,
        nb,
        v,
        v_stride,
        w2,
        ncols,
        &mut a[a_off..],
        lda,
    );
}

/// Q-panel variant: identical math, but `V` arrives as a gathered compact
/// block (the work-matrix rows are longer than `nb`, so the Q pass gathers
/// once per panel and reuses this same entry point).
#[allow(clippy::too_many_arguments)]
fn wy_update_q<T: ComplexField>(
    v: &[T],
    rows: usize,
    nb: usize,
    tmat: &[T],
    qmat: &mut [T],
    q_row0: usize,
    q_stride: usize,
    wbuf: &mut Vec<T>,
    w2: &mut Vec<T>,
) {
    // Q accumulation uses U = T as-is: each panel factor must equal
    // H_first···H_last, which is exactly what the upper-triangular compact
    // WY form represents.
    wy_update(
        v,
        nb,
        rows,
        nb,
        tmat,
        QR_BLOCK_NB,
        qmat,
        q_row0,
        0,
        q_stride,
        q_stride,
        wbuf,
        w2,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_least_squares_matches_wolfram_reference() {
        let matrix =
            DenseMatrix::from_row_major_vec(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let rhs = [6.0, 5.0, 7.0, 10.0];
        let factor = QrFactorization::factorize(&matrix).unwrap();
        let solution = factor.solve_least_squares(&rhs).unwrap();
        assert!((solution[0] - 3.5).abs() < 1.0e-12);
        assert!((solution[1] - 1.4).abs() < 1.0e-12);
    }

    #[test]
    fn pivoted_qr_reports_rank_deficiency() {
        let matrix = DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        let factor = QrFactorization::factorize_pivoted(&matrix).unwrap();
        assert_eq!(factor.rank(), 1);
        assert!(matches!(
            factor.solve_least_squares(&[1.0, 2.0, 3.0]),
            Err(QrError::RankDeficient { rank: 1, .. })
        ));
    }

    #[test]
    fn qr_scaled_norm_accepts_large_finite_column() {
        let matrix = DenseMatrix::from_row_major_vec(2, 1, vec![1.0e308, 1.0]);
        let factor = QrFactorization::factorize_pivoted(&matrix)
            .expect("a representable large column norm must remain factorable");
        assert_eq!(factor.rank(), 1);
        assert!(factor.q().as_slice().iter().all(|value| value.is_finite()));
        assert!(factor.r().as_slice().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn qr_rejects_unrepresentable_generated_column_norm() {
        let matrix = DenseMatrix::from_row_major_vec(2, 1, vec![f64::MAX, f64::MAX]);
        assert!(matches!(
            QrFactorization::factorize_pivoted(&matrix),
            Err(QrError::NumericalBreakdown {
                operation: "initial column norm"
            })
        ));
    }

    #[test]
    fn qr_least_squares_rejects_non_finite_or_overflowing_rhs_projection() {
        let matrix = DenseMatrix::from_row_major_vec(2, 1, vec![1.0, 1.0]);
        let factor = QrFactorization::factorize(&matrix).unwrap();
        assert_eq!(
            factor.solve_least_squares(&[f64::NAN, 1.0]),
            Err(QrError::NonFiniteRhs)
        );
        assert!(matches!(
            factor.solve_least_squares(&[f64::MAX, f64::MAX]),
            Err(QrError::NumericalBreakdown {
                operation: "least-squares projection"
            })
        ));
    }

    #[test]
    fn tsqr_reconstructs_tall_matrix_and_solves_least_squares() {
        let rows = 1_024;
        let matrix = DenseMatrix::from_fn(rows, 3, |row, column| {
            let x = row as f64 / rows as f64;
            match column {
                0 => 1.0,
                1 => x,
                _ => x * x - 0.25,
            }
        });
        let factor = QrFactorization::factorize_tsqr(&matrix).unwrap();
        for row in 0..rows {
            for column in 0..3 {
                let reconstructed: f64 = (0..3)
                    .map(|inner| factor.q()[[row, inner]] * factor.r()[[inner, column]])
                    .sum();
                assert!((reconstructed - matrix[[row, column]]).abs() < 2.0e-14);
            }
        }
        let exact = [2.0, -3.0, 0.5];
        let rhs = matrix.matvec(&exact);
        let solution = factor.solve_least_squares(&rhs).unwrap();
        for (actual, expected) in solution.iter().zip(exact) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
    }

    #[test]
    fn tsqr_reports_rank_deficient_blocks() {
        let matrix = DenseMatrix::from_fn(600, 2, |row, column| {
            let value = row as f64 + 1.0;
            if column == 0 { value } else { 2.0 * value }
        });
        assert!(matches!(
            QrFactorization::factorize_tsqr(&matrix),
            Err(QrError::RankDeficient {
                rank: 1,
                columns: 2
            })
        ));
    }

    #[test]
    fn tsqr_accepts_rank_deficient_blocks_when_global_matrix_is_full_rank() {
        let matrix = DenseMatrix::from_fn(1_024, 2, |row, column| match (row < 512, column) {
            (true, 0) => 1.0_f64 + row as f64 / 1024.0,
            (false, 1) => 1.0_f64 + row as f64 / 1024.0,
            _ => 0.0,
        });
        let factor = QrFactorization::factorize_tsqr(&matrix).unwrap();
        let exact = [2.0, -3.0];
        let rhs = matrix.matvec(&exact);
        let solution = factor.solve_least_squares(&rhs).unwrap();
        for (actual, expected) in solution.iter().zip(exact) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                let reconstructed: f64 = (0..matrix.ncols())
                    .map(|inner| factor.q()[[row, inner]] * factor.r()[[inner, column]])
                    .sum();
                assert!((reconstructed - matrix[[row, column]]).abs() < 1.0e-13);
            }
        }
    }

    #[test]
    fn thin_svd_reconstructs_and_detects_rank() {
        let matrix = DenseMatrix::from_row_major_vec(3, 2, vec![3.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let svd = dense_svd(&matrix).unwrap();
        assert_eq!(svd.rank, 2);
        assert!((svd.singular_values[0] - 3.0).abs() < 1.0e-12);
        assert!((svd.singular_values[1] - 2.0).abs() < 1.0e-12);
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                let reconstructed: f64 = (0..matrix.ncols())
                    .map(|mode| {
                        svd.u[[row, mode]] * svd.singular_values[mode] * svd.v[[column, mode]]
                    })
                    .sum();
                assert!((reconstructed - matrix[[row, column]]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn wide_svd_reconstructs_with_swapped_factors() {
        let matrix =
            DenseMatrix::from_row_major_vec(2, 4, vec![3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let svd = dense_svd(&matrix).unwrap();
        assert_eq!(svd.rank, 2);
        assert_eq!(svd.u.shape(), (2, 2));
        assert_eq!(svd.v.shape(), (4, 2));
        assert_eq!(svd.singular_values.len(), 2);
        assert!((svd.singular_values[0] - 3.0).abs() < 1.0e-12);
        assert!((svd.singular_values[1] - 2.0).abs() < 1.0e-12);
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                let reconstructed: f64 = (0..matrix.nrows())
                    .map(|mode| {
                        svd.u[[row, mode]] * svd.singular_values[mode] * svd.v[[column, mode]]
                    })
                    .sum();
                assert!((reconstructed - matrix[[row, column]]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn wide_svd_matches_transpose_path_and_stays_orthonormal() {
        // Fixed non-diagonal wide matrix (ESPRIT-style: few rows, many columns).
        let matrix = DenseMatrix::from_row_major_vec(
            3,
            7,
            vec![
                0.5, -1.2, 0.3, 2.1, -0.7, 1.4, 0.9, //
                1.1, 0.4, -2.0, 0.6, 1.8, -0.3, 0.2, //
                -0.9, 1.7, 0.8, -1.5, 0.1, 2.2, -0.6,
            ],
        );
        let svd = dense_svd(&matrix).unwrap();
        let reference = dense_svd(&matrix.transposed()).unwrap();
        assert_eq!(svd.rank, reference.rank);
        for (actual, expected) in svd
            .singular_values
            .iter()
            .zip(reference.singular_values.iter())
        {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                let reconstructed: f64 = (0..matrix.nrows())
                    .map(|mode| {
                        svd.u[[row, mode]] * svd.singular_values[mode] * svd.v[[column, mode]]
                    })
                    .sum();
                assert!((reconstructed - matrix[[row, column]]).abs() < 1.0e-11);
            }
        }
        // Thin factors keep orthonormal columns: UᵀU = I and VᵀV = I.
        for (factor, modes) in [(&svd.u, 3), (&svd.v, 3)] {
            for left in 0..modes {
                for right in 0..modes {
                    let overlap: f64 = (0..factor.nrows())
                        .map(|row| factor[[row, left]] * factor[[row, right]])
                        .sum();
                    let expected = if left == right { 1.0 } else { 0.0 };
                    assert!((overlap - expected).abs() < 1.0e-12);
                }
            }
        }
    }

    #[test]
    fn thin_svd_scaled_norm_accepts_large_representable_column() {
        let matrix = DenseMatrix::from_row_major_vec(2, 1, vec![1.0e308, 1.0]);
        let svd = dense_svd(&matrix)
            .expect("a representable large singular value must remain factorable");
        assert_eq!(svd.rank, 1);
        assert!(svd.singular_values.iter().all(|value| value.is_finite()));
        assert!(svd.u.as_slice().iter().all(|value| value.is_finite()));
        assert!(svd.v.as_slice().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn thin_svd_rejects_unrepresentable_generated_singular_value() {
        let matrix = DenseMatrix::from_row_major_vec(2, 1, vec![f64::MAX, f64::MAX]);
        assert!(matches!(
            dense_svd(&matrix),
            Err(QrError::NumericalBreakdown {
                operation: "SVD singular value"
            })
        ));
    }

    /// Deterministic full-rank fixture for blocked-vs-MGS agreement.
    /// Full-rank, well-conditioned fixture (σ_min ≈ 1.8 at every tested
    /// shape, verified by SVD): a 2.0 diagonal spine plus 1% sinusoidal
    /// texture. The old sin/cos-mixture fixture had true rank ≈ 4, which
    /// made rank verdicts algorithm-dependent noise.
    fn blocked_fixture(rows: usize, cols: usize) -> DenseMatrix<f64> {
        DenseMatrix::from_fn(rows, cols, |row, col| {
            let x = (row * 31 + col * 17) as f64;
            (if row == col { 2.0 } else { 0.0 }) + 0.01 * (x * 0.61803398875).sin()
        })
    }

    #[test]
    fn blocked_matches_mgs_on_full_rank() {
        // 96x64 exercises two blocked panels plus a narrow tail. MGS uses
        // a positive R diagonal while Householder mirrors LAPACK's
        // negative one, so columns are compared up to per-column sign.
        let (rows, cols) = (96, 64);
        let matrix = blocked_fixture(rows, cols);
        let mgs = factorize_impl(&matrix, false).unwrap();
        let blocked = factorize_blocked(&matrix).unwrap();
        assert_eq!(blocked.rank(), 64);
        let mq = mgs.q().as_slice();
        let bq = blocked.q().as_slice();
        let mut max_diff = 0.0;
        for j in 0..cols {
            let mut plus = 0.0;
            let mut minus = 0.0;
            for i in 0..rows {
                let d = mq[i * cols + j] - bq[i * cols + j];
                let s = mq[i * cols + j] + bq[i * cols + j];
                plus = plus.max(d.abs());
                minus = minus.max(s.abs());
            }
            max_diff = max_diff.max(plus.min(minus));
        }
        assert!(max_diff < 1.0e-9, "Q differs by {max_diff:e}");
        // Flipping Q column i flips R row i (A = QR); align R rows by the
        // corresponding Q-column sign.
        let mr = mgs.r().as_slice();
        let br = blocked.r().as_slice();
        let mut max_r = 0.0;
        for i in 0..cols {
            let mut qp = 0.0;
            let mut qm = 0.0;
            for r in 0..rows {
                let d = mq[r * cols + i] - bq[r * cols + i];
                let s = mq[r * cols + i] + bq[r * cols + i];
                qp = qp.max(d.abs());
                qm = qm.max(s.abs());
            }
            let flipped = qp > qm;
            for j in 0..cols {
                let d = (mr[i * cols + j] - br[i * cols + j]).abs();
                let s = (mr[i * cols + j] + br[i * cols + j]).abs();
                max_r = max_r.max(if flipped { s } else { d });
            }
        }
        assert!(max_r < 1.0e-9, "R differs by {max_r:e}");
    }

    #[test]
    fn blocked_is_orthogonal_and_reconstructs() {
        let (rows, cols) = (128, 96);
        let matrix = blocked_fixture(rows, cols);
        let factor = factorize_blocked(&matrix).unwrap();
        let q = factor.q().as_slice();
        // ||Q^T Q - I||_max.
        let mut orth = 0.0;
        for i in 0..cols {
            for j in 0..cols {
                let mut acc = 0.0;
                for r in 0..rows {
                    acc += q[r * cols + i] * q[r * cols + j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                orth = orth.max((acc - target).abs());
            }
        }
        assert!(orth < 1.0e-12, "orthogonality {orth:e}");
        // ||A - QR||_max / ||A||_max.
        let r = factor.r().as_slice();
        let a = matrix.as_slice();
        let mut resid = 0.0;
        let mut anorm = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0.0;
                for k in 0..cols {
                    acc += q[i * cols + k] * r[k * cols + j];
                }
                resid = resid.max((a[i * cols + j] - acc).abs());
                anorm = anorm.max(a[i * cols + j].abs());
            }
        }
        assert!(resid / anorm < 1.0e-12, "residual {}", resid / anorm);
    }

    #[test]
    fn blocked_reports_matching_rank_deficiency() {
        // Rank-60 70x65 product: both paths must agree on the rank.
        let b = blocked_fixture(70, 60);
        let c = blocked_fixture(60, 65);
        let mut data = vec![0.0; 70 * 65];
        for i in 0..70 {
            for j in 0..65 {
                let mut acc = 0.0;
                for k in 0..60 {
                    acc += b.as_slice()[i * 60 + k] * c.as_slice()[k * 65 + j];
                }
                data[i * 65 + j] = acc;
            }
        }
        let matrix = DenseMatrix::from_row_major_vec(70, 65, data);
        let mgs_rank = match factorize_impl(&matrix, false) {
            Ok(f) => f.rank(),
            Err(QrError::RankDeficient { rank, .. }) => rank,
            Err(e) => panic!("MGS unexpected error: {e:?}"),
        };
        let blocked_rank = match factorize_blocked(&matrix) {
            Ok(f) => f.rank(),
            Err(QrError::RankDeficient { rank, .. }) => rank,
            Err(e) => panic!("blocked unexpected error: {e:?}"),
        };
        assert_eq!(mgs_rank, 60);
        assert_eq!(blocked_rank, mgs_rank);
    }

    #[test]
    fn workspace_matches_allocating_bitwise() {
        // Same flop sequence through reused buffers: Q/R bitwise identical
        // to the allocating path, including across workspace reuse with
        // different shapes (growth + all_t/vbuf clearing).
        let mut ws = QrWorkspace::<f64>::new();
        for (rows, cols) in [(96, 64), (128, 96), (96, 64)] {
            let matrix = blocked_fixture(rows, cols);
            let plain = QrFactorization::factorize(&matrix).unwrap();
            let reused = qr_factorize_with_workspace(&matrix, &mut ws).unwrap();
            assert_eq!(reused.rank(), plain.rank());
            assert_eq!(reused.q().as_slice(), plain.q().as_slice());
            assert_eq!(reused.r().as_slice(), plain.r().as_slice());
        }
    }

    /// Fill every workspace buffer with NaN garbage at mixed capacities:
    /// some larger than the coming factorization needs (shrink/truncate
    /// path), some smaller (growth path). Any stale read surfaces as a
    /// wrong or non-finite factor, so bitwise equality with the fresh
    /// path proves the overwrite contract retention relies on.
    fn poison_workspace(ws: &mut QrWorkspace<f64>, big: usize, small: usize) {
        let nan = f64::NAN;
        ws.a = vec![nan; big];
        ws.qmat = vec![nan; big];
        ws.pt = vec![nan; big];
        ws.panel = vec![nan; big];
        ws.vtmp = vec![nan; small];
        ws.tmat = vec![nan; big];
        ws.tmat_t = vec![nan; small];
        ws.wbuf = vec![nan; big];
        ws.w2 = vec![nan; small];
        ws.all_t = vec![nan; big];
        ws.betas = vec![nan; small];
        ws.taus = vec![nan; small];
        ws.vbuf = vec![nan; big];
    }

    #[test]
    fn retained_garbage_matches_fresh_bitwise() {
        // Poisoned capacities + shapes sequenced to hit every reuse path:
        // square (steal empties `qmat`), tall (thin copy keeps it), then
        // square again at a smaller size (shrink over post-steal state).
        // Each matches a fresh workspace bitwise, twice in a row.
        let mut ws = QrWorkspace::<f64>::new();
        poison_workspace(&mut ws, 200 * 200, 8);
        for (rows, cols) in [(96, 96), (128, 96), (64, 64)] {
            let matrix = blocked_fixture(rows, cols);
            let fresh = factorize_blocked_ws_direct(&matrix, &mut QrWorkspace::new()).unwrap();
            for _ in 0..2 {
                let reused = factorize_blocked_ws_direct(&matrix, &mut ws).unwrap();
                assert_eq!(reused.rank(), fresh.rank());
                assert_eq!(reused.q().as_slice(), fresh.q().as_slice());
                assert_eq!(reused.r().as_slice(), fresh.r().as_slice());
            }
        }
    }

    #[test]
    fn retained_garbage_matches_fresh_complex() {
        // Same overwrite-contract proof for the complex path (interleaved
        // NaN in both parts; any stale read corrupts a factor).
        use num_complex::Complex64;
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let mut ws = QrWorkspace::<Complex64>::new();
        ws.a = vec![nan; 160 * 160];
        ws.qmat = vec![nan; 160 * 160];
        ws.pt = vec![nan; 160 * 160];
        ws.panel = vec![nan; 160 * 160];
        ws.vtmp = vec![nan; 8];
        ws.tmat = vec![nan; 160 * 160];
        ws.tmat_t = vec![nan; 8];
        ws.wbuf = vec![nan; 160 * 160];
        ws.w2 = vec![nan; 8];
        ws.all_t = vec![nan; 160 * 160];
        ws.betas = vec![nan; 8];
        ws.taus = vec![nan; 8];
        ws.vbuf = vec![nan; 160 * 160];
        for (rows, cols) in [(96, 96), (128, 96)] {
            let matrix = DenseMatrix::from_fn(rows, cols, |row, col| {
                let x = (row * 31 + col * 17) as f64;
                let re = (if row == col { 2.0 } else { 0.0 }) + 0.01 * (x * 0.61803398875).sin();
                Complex64::new(re, 0.005 * (x * 0.38196601125).cos())
            });
            let fresh = factorize_blocked_ws_direct(&matrix, &mut QrWorkspace::new()).unwrap();
            for _ in 0..2 {
                let reused = factorize_blocked_ws_direct(&matrix, &mut ws).unwrap();
                assert_eq!(reused.rank(), fresh.rank());
                assert_eq!(reused.q().as_slice(), fresh.q().as_slice());
                assert_eq!(reused.r().as_slice(), fresh.r().as_slice());
            }
        }
    }

    #[test]
    fn retained_workspace_mechanism() {
        // The thread-local dispatch retains capacities across calls on
        // this thread, drops over-cap workspaces, and degrades to `None`
        // (fresh-workspace fallback) under same-thread reentrancy instead
        // of panicking. Env-independent: the kill-switch lives one layer
        // up in `factorize_blocked`.
        let first = with_retained_workspace(|ws: &mut QrWorkspace<f64>| {
            ws.a.resize(100, 1.0);
            ws.a.capacity()
        })
        .unwrap();
        assert!(first >= 100);
        // Second call observes the retained capacity.
        let second = with_retained_workspace(|ws: &mut QrWorkspace<f64>| ws.a.capacity()).unwrap();
        assert_eq!(second, first);
        // Over-cap workspaces are dropped: the next call starts fresh.
        with_retained_workspace(|ws: &mut QrWorkspace<f64>| {
            ws.a.reserve(QR_RETAIN_MAX_BYTES / std::mem::size_of::<f64>() + 1);
        })
        .unwrap();
        let after_cap =
            with_retained_workspace(|ws: &mut QrWorkspace<f64>| ws.a.capacity()).unwrap();
        assert_eq!(after_cap, 0);
        // Reentrant borrow fails closed, outer borrow unaffected.
        let outer = with_retained_workspace(|_: &mut QrWorkspace<f64>| {
            with_retained_workspace(|_: &mut QrWorkspace<f64>| ()).is_none()
        })
        .unwrap();
        assert!(outer);
        // A fresh slot starts empty on an untouched type/thread.
        let other = std::thread::spawn(|| {
            with_retained_workspace(|ws: &mut QrWorkspace<f32>| ws.a.capacity()).unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(other, 0);
    }

    #[test]
    fn reflect_blocked_matches_serial_bitwise() {
        // 640x64 panel: rows >= REFLECT_PAR_MIN_ROWS and trailing >=
        // REFLECT_PAR_MIN_COLS route most sub-blocks through the parallel
        // trailing update, while the serial twin runs the original loop.
        // The parallel branch is forced directly (not via the env-gated
        // dispatcher) so the test stays meaningful under ambient
        // `SONIUM_QR_REFLECT_PAR=0`. Per-column op order is identical, so
        // pt/taus/betas must match bitwise. Row 40 is all zeros: it stays
        // zero under earlier updates (dot with zeros is exactly zero),
        // exercising the norm==0 identity branch and the active-list
        // zero-tau skip on both paths.
        let (rows, nb) = (640usize, 64usize);
        let mut pt0 = vec![0.0f64; nb * rows];
        for j in 0..nb {
            for r in 0..rows {
                let x = (r * 31 + j * 17) as f64;
                pt0[j * rows + r] = 0.01 * (x * 0.61803398875).sin();
            }
        }
        for r in 0..rows {
            pt0[40 * rows + r] = 0.0;
        }
        let mut pt_par = pt0.clone();
        let mut pt_ser = pt0.clone();
        let mut taus_par = vec![0.0f64; nb];
        let mut taus_ser = vec![0.0f64; nb];
        let mut betas_par = vec![0.0f64; nb];
        let mut betas_ser = vec![0.0f64; nb];
        let mut vtmp = vec![0.0f64; rows];
        panel_reflect_parallel(
            &mut pt_par,
            rows,
            nb,
            &mut vtmp,
            &mut taus_par,
            &mut betas_par,
        )
        .unwrap();
        panel_reflect_serial(
            &mut pt_ser,
            rows,
            nb,
            &mut vtmp,
            &mut taus_ser,
            &mut betas_ser,
        )
        .unwrap();
        assert_eq!(taus_par[40], 0.0);
        assert_eq!(pt_par, pt_ser);
        assert_eq!(taus_par, taus_ser);
        assert_eq!(betas_par, betas_ser);
    }

    #[test]
    fn funnel_install_matches_direct_bitwise() {
        // The single-install funnel (see `factorize_blocked_ws`) only
        // hoists the mirror-pool install: same pool, same task geometry,
        // so Q/R bytes match the per-scope path. Direct mechanism check
        // at small size (the 1024-column gate keeps this shape on the
        // per-scope path in production).
        let matrix = blocked_fixture(96, 64);
        let direct = factorize_blocked_ws_direct(&matrix, &mut QrWorkspace::<f64>::new()).unwrap();
        let funneled = sonium_compute::cpu::blas3::run_mirror(|| {
            factorize_blocked_ws_direct(&matrix, &mut QrWorkspace::<f64>::new()).unwrap()
        });
        assert_eq!(funneled.rank(), direct.rank());
        assert_eq!(funneled.q().as_slice(), direct.q().as_slice());
        assert_eq!(funneled.r().as_slice(), direct.r().as_slice());
    }

    #[test]
    fn funnels_match_direct_at_gate_width() {
        // Gate-width coverage for both production funnels. f64 1024 goes
        // through the real dispatcher (tiny funnel on wide hosts, mirror
        // below its thread floor — either way a funneled path) vs the
        // per-scope direct path. f32 pins the mirror mechanism itself at
        // gate width via an explicit install, so the arm stays meaningful
        // even under ambient `SONIUM_QR_FUNNEL=0`.
        let matrix = blocked_fixture(1024, 1024);
        let funneled = factorize_blocked_ws(&matrix, &mut QrWorkspace::<f64>::new()).unwrap();
        let direct = factorize_blocked_ws_direct(&matrix, &mut QrWorkspace::<f64>::new()).unwrap();
        assert_eq!(funneled.rank(), direct.rank());
        assert_eq!(funneled.q().as_slice(), direct.q().as_slice());
        assert_eq!(funneled.r().as_slice(), direct.r().as_slice());
        let matrix32 = DenseMatrix::from_fn(1024, 1024, |row, col| {
            let x = (row * 31 + col * 17) as f32;
            (if row == col { 2.0 } else { 0.0 }) + 0.01 * (x * 0.61803398875).sin()
        });
        let funneled32 = sonium_compute::cpu::blas3::run_mirror(|| {
            factorize_blocked_ws_direct(&matrix32, &mut QrWorkspace::<f32>::new()).unwrap()
        });
        let direct32 =
            factorize_blocked_ws_direct(&matrix32, &mut QrWorkspace::<f32>::new()).unwrap();
        assert_eq!(funneled32.rank(), direct32.rank());
        assert_eq!(funneled32.q().as_slice(), direct32.q().as_slice());
        assert_eq!(funneled32.r().as_slice(), direct32.r().as_slice());
    }

    #[test]
    fn wy_two_phase_matches_serial_bitwise() {
        // 640x64 panel with real reflector data (built by panel_reflect,
        // including a zero-tau column at 40): rows >= WY_PAR_MIN_ROWS and
        // nb >= WY_PAR_MIN_COLS route through the parallel two-phase
        // transform, forced directly (not via the env-gated dispatcher)
        // so the test stays meaningful under ambient
        // `SONIUM_QR_WY_PAR=0`. Same dots in the same order and the same
        // recurrence, so tmat must match bitwise — including the skipped
        // zero-tau column (its dots are exactly zero).
        let (rows, nb) = (640usize, 64usize);
        let mut pt = vec![0.0f64; nb * rows];
        for j in 0..nb {
            for r in 0..rows {
                let x = (r * 31 + j * 17) as f64;
                pt[j * rows + r] = 0.01 * (x * 0.61803398875).sin();
            }
        }
        for r in 0..rows {
            pt[40 * rows + r] = 0.0;
        }
        let mut taus = vec![0.0f64; nb];
        let mut betas = vec![0.0f64; nb];
        let mut vtmp = vec![0.0f64; rows];
        panel_reflect_serial(&mut pt, rows, nb, &mut vtmp, &mut taus, &mut betas).unwrap();
        assert_eq!(taus[40], 0.0);
        let mut tmat_par = vec![0.0f64; QR_BLOCK_NB * QR_BLOCK_NB];
        let mut tmat_ser = vec![0.0f64; QR_BLOCK_NB * QR_BLOCK_NB];
        wy_transform_two_phase(&pt, rows, nb, &taus, &mut tmat_par).unwrap();
        wy_transform_serial(&pt, rows, nb, &taus, &mut tmat_ser).unwrap();
        assert_eq!(tmat_par, tmat_ser);
    }

    #[test]
    fn fused_overlay_matches_sequential() {
        // Both overlay branches must produce byte-identical `a`/`panel`
        // buffers, on a mid-matrix panel and a minimal square one.
        // Distinct garbage init catches any one-sided missed write.
        for (n, k0, rows, nb) in [(128usize, 32usize, 96usize, 64usize), (64, 0, 64, 64)] {
            let pt: Vec<f64> = (0..nb * rows)
                .map(|i| 0.01 * ((i * 31 + 7) as f64 * 0.61803398875).sin())
                .collect();
            let betas: Vec<f64> = (0..nb).map(|j| 2.0 + 0.1 * j as f64).collect();
            let mut a_fused = vec![-1.25f64; n * n];
            let mut panel_fused = vec![-1.25f64; rows * nb];
            let mut a_seq = vec![-1.25f64; n * n];
            let mut panel_seq = vec![-1.25f64; rows * nb];
            overlay_panel_fused(&mut a_fused, n, k0, rows, nb, &pt, &betas, &mut panel_fused);
            overlay_panel_sequential(&mut a_seq, n, k0, rows, nb, &pt, &betas, &mut panel_seq);
            assert_eq!(panel_fused, panel_seq, "panel mismatch at k0={k0}");
            assert_eq!(a_fused, a_seq, "work-matrix mismatch at k0={k0}");
        }
    }

    #[test]
    fn wy_two_phase_ignores_skipped_column_dots() {
        // Serial skips zero-tau columns before computing their z-dots; the
        // two-phase path must not observe them either. NaN sits where only
        // z[0,1] reads it, and taus[1] == 0: serial succeeds, so two-phase
        // must succeed with identical bytes.
        let (rows, nb) = (8usize, 3usize);
        let mut pt = vec![0.25f64; nb * rows];
        pt[1 * rows + 1] = f64::NAN;
        let taus = vec![2.0f64, 0.0, 2.0];
        let mut tmat_par = vec![0.0f64; QR_BLOCK_NB * QR_BLOCK_NB];
        let mut tmat_ser = vec![0.0f64; QR_BLOCK_NB * QR_BLOCK_NB];
        wy_transform_serial(&pt, rows, nb, &taus, &mut tmat_ser).unwrap();
        wy_transform_two_phase(&pt, rows, nb, &taus, &mut tmat_par).unwrap();
        assert_eq!(tmat_par, tmat_ser);
    }

    #[test]
    fn wy_two_phase_reports_serial_first_failure_site() {
        // Serial fails at the j=1 accumulation (2·1e308 overflows) before
        // ever reaching the NaN dots of column 3; two-phase must report
        // the same first site, not its own up-front dot scan.
        let (rows, nb) = (8usize, 4usize);
        let mut pt = vec![0.0f64; nb * rows];
        pt[0 * rows + 1] = 1e308;
        pt[1 * rows + 1] = 1.0;
        pt[3 * rows + 3] = f64::NAN;
        let taus = vec![2.0f64; nb];
        let mut tmat_ser = vec![0.0f64; QR_BLOCK_NB * QR_BLOCK_NB];
        let mut tmat_par = vec![0.0f64; QR_BLOCK_NB * QR_BLOCK_NB];
        let ser = wy_transform_serial(&pt, rows, nb, &taus, &mut tmat_ser).unwrap_err();
        let par = wy_transform_two_phase(&pt, rows, nb, &taus, &mut tmat_par).unwrap_err();
        assert!(matches!(
            ser,
            QrError::NumericalBreakdown {
                operation: "WY accumulation"
            }
        ));
        assert!(matches!(
            par,
            QrError::NumericalBreakdown {
                operation: "WY accumulation"
            }
        ));
    }

    #[test]
    fn fused_rcheck_matches_separate_walk() {
        // The fused R check must agree with the legacy copy + full walk on
        // both bytes and errors. The `a` strict-lower holds NaN garbage:
        // neither path may see it (only the upper reaches the zeroed
        // output), proving the skipped triangle is truly ignored.
        let n = 8usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if j < i {
                    f64::NAN
                } else {
                    0.01 * ((i * 31 + j * 17) as f64 * 0.61803398875).sin()
                        + if i == j { 2.0 } else { 0.0 }
                };
            }
        }
        // Finite upper: identical bytes, both Ok.
        let mut rs_fused = vec![0.0f64; n * n];
        let mut rs_sep = vec![0.0f64; n * n];
        assert!(copy_r_checked(&mut rs_fused, &a, n).is_ok());
        copy_r_upper(&mut rs_sep, &a, n);
        assert!(ensure_finite_slice("triangular factor", &rs_sep).is_ok());
        assert_eq!(rs_fused, rs_sep);
        // Non-finite upper entries (inf diagonal, NaN strict-upper):
        // identical errors. Bytes match through the failing row (the
        // fused path copies each row before checking it).
        for bad in [2 * n + 2, n + 5] {
            let saved = a[bad];
            a[bad] = if bad == 2 * n + 2 {
                f64::INFINITY
            } else {
                f64::NAN
            };
            let mut rs_fused = vec![0.0f64; n * n];
            let mut rs_sep = vec![0.0f64; n * n];
            let err_fused = copy_r_checked(&mut rs_fused, &a, n);
            copy_r_upper(&mut rs_sep, &a, n);
            let err_sep = ensure_finite_slice("triangular factor", &rs_sep);
            assert_eq!(
                err_fused,
                Err(QrError::NumericalBreakdown {
                    operation: "triangular factor",
                })
            );
            assert_eq!(err_fused, err_sep);
            let fail_row = bad / n;
            assert_eq!(rs_fused[..fail_row * n], rs_sep[..fail_row * n]);
            // Bitwise compare through the failing row (NaN != NaN under
            // `PartialEq`, so plain slice equality cannot see it).
            for v in rs_fused[..(fail_row + 1) * n]
                .iter()
                .zip(rs_sep[..(fail_row + 1) * n].iter())
            {
                assert_eq!(v.0.to_bits(), v.1.to_bits());
            }
            a[bad] = saved;
        }
    }

    #[test]
    fn workspace_matches_allocating_complex() {
        use num_complex::Complex64;
        let matrix = DenseMatrix::from_fn(96, 64, |row, col| {
            let x = (row * 31 + col * 17) as f64;
            Complex64::new(
                (if row == col { 2.0 } else { 0.0 }) + 0.01 * (x * 0.61803398875).sin(),
                0.01 * (x * 0.38196601125).cos(),
            )
        });
        let mut ws = QrWorkspace::<Complex64>::new();
        let plain = QrFactorization::factorize(&matrix).unwrap();
        let reused = qr_factorize_with_workspace(&matrix, &mut ws).unwrap();
        assert_eq!(reused.rank(), plain.rank());
        assert_eq!(reused.q().as_slice(), plain.q().as_slice());
        assert_eq!(reused.r().as_slice(), plain.r().as_slice());
    }

    #[test]
    fn jacobi_svd_matches_nontrivial_singular_values() {
        let matrix = DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let svd = dense_svd(&matrix).unwrap();
        assert!((svd.singular_values[0] - 9.525_518_091_565_107).abs() < 1.0e-11);
        assert!((svd.singular_values[1] - 0.514_300_580_658_644_3).abs() < 1.0e-11);
        let overlap: f64 = (0..3).map(|row| svd.u[[row, 0]] * svd.u[[row, 1]]).sum();
        assert!(overlap.abs() < 1.0e-12);

        let deficient = DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        assert_eq!(dense_svd(&deficient).unwrap().rank, 1);
    }
}
