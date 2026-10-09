//! Parallel ILU preconditioners
//!
//! Two parallel approaches for ILU triangular solves:
//!
//! 1. **Graph Coloring (Level Scheduling)**: Groups rows into independent sets (levels)
//!    that can be solved in parallel. Rows within the same level have no dependencies.
//!
//! 2. **Fine-Grained Fixed-Point Iteration**: Uses Jacobi-style iteration to
//!    approximate the triangular solve. Each iteration is embarrassingly parallel.
//!
//! 3. **ParILUT** (Chow/Anzt/Dongarra): parallel fixed-point sweeps over the
//!    factor *values* during setup, with optional threshold-based candidate
//!    augmentation of the nonzero pattern.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, Preconditioner, SolverError};
use sonium_solvers_sparse::CsrMatrix;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::ilu::{
    Ilu0Factors, IluDiagnostics, IluPreconditioner, finish_scaled_norm, ilu0_factorize,
    merge_scaled_norm, update_scaled_norm,
};

/// Version of the portable ParILUT factor checkpoint format.
pub const PARILUT_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const PARILUT_CHECKPOINT_MAGIC: &[u8; 8] = b"SPILTCP1";
/// Version of the portable graph-coloring ILU checkpoint format.
pub const ILU_COLORING_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const ILU_COLORING_CHECKPOINT_MAGIC: &[u8; 8] = b"SICLCP01";
/// Version of the portable fixed-point ILU checkpoint format.
pub const ILU_FIXED_POINT_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const ILU_FIXED_POINT_CHECKPOINT_MAGIC: &[u8; 8] = b"SIFPCP01";

fn parilut_checkpoint_error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "ParILUT checkpoint",
        reason: message.into(),
    }
}

fn parilut_checkpoint_push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value = u64::try_from(value)
        .map_err(|_| parilut_checkpoint_error("usize value does not fit checkpoint format"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

struct ParIlutCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ParIlutCheckpointReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| parilut_checkpoint_error("checkpoint offset overflow"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| parilut_checkpoint_error("truncated checkpoint"))?;
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(value))
    }

    fn read_u64(&mut self) -> Result<u64, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(value))
    }

    fn read_len(&mut self, element_bytes: usize) -> Result<usize, SolverError> {
        let length = usize::try_from(self.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("checkpoint length does not fit usize"))?;
        if length > self.bytes.len().saturating_sub(self.offset) / element_bytes {
            return Err(parilut_checkpoint_error(
                "checkpoint vector length exceeds payload",
            ));
        }
        Ok(length)
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        let value = f64::from_le_bytes(value);
        if !value.is_finite() {
            return Err(parilut_checkpoint_error(
                "checkpoint contains non-finite value",
            ));
        }
        Ok(value)
    }

    fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn parallel_ilu_payload<T: ComplexField>(factors: Ilu0Factors<T>) -> Result<Vec<u8>, SolverError> {
    IluPreconditioner::from_factors(factors, None)?.to_checkpoint_bytes()
}

fn parallel_ilu_factors<T: ComplexField>(payload: &[u8]) -> Result<Ilu0Factors<T>, SolverError> {
    let ilu = IluPreconditioner::<T>::from_checkpoint_bytes(payload)?;
    let (factors, permutation) = ilu.into_factors();
    if permutation.is_some() {
        return Err(parilut_checkpoint_error(
            "parallel ILU checkpoint cannot contain a permutation",
        ));
    }
    Ok(factors)
}

fn parallel_ilu_push_levels(
    output: &mut Vec<u8>,
    levels: &[Vec<usize>],
) -> Result<(), SolverError> {
    parilut_checkpoint_push_u64(output, levels.len())?;
    for level in levels {
        parilut_checkpoint_push_u64(output, level.len())?;
        for &row in level {
            parilut_checkpoint_push_u64(output, row)?;
        }
    }
    Ok(())
}

fn parallel_ilu_read_levels(
    reader: &mut ParIlutCheckpointReader<'_>,
    n: usize,
) -> Result<Vec<Vec<usize>>, SolverError> {
    let level_count = reader.read_len(8)?;
    let mut levels = Vec::with_capacity(level_count);
    for _ in 0..level_count {
        let row_count = reader.read_len(8)?;
        let mut level = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            let row = usize::try_from(reader.read_u64()?)
                .map_err(|_| parilut_checkpoint_error("level row does not fit usize"))?;
            if row >= n {
                return Err(parilut_checkpoint_error("level row is out of range"));
            }
            level.push(row);
        }
        levels.push(level);
    }
    Ok(levels)
}

fn validate_parallel_ilu_matrix<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    operation: &'static str,
) -> Result<(), SolverError> {
    matrix.validate_canonical_structure()?;
    if matrix.num_rows != matrix.num_cols {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "matrix must be square, got {}x{}",
                matrix.num_rows, matrix.num_cols
            ),
        });
    }
    if matrix.values.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "matrix contains NaN or infinity".to_string(),
        });
    }
    CsrMatrix::try_from_raw_parts(
        matrix.num_rows,
        matrix.num_cols,
        matrix.row_ptrs.clone(),
        matrix.col_indices.clone(),
        matrix.values.clone(),
    )
    .map(|_| ())
}

fn check_parallel_ilu_diagnostics<R: Float>(
    diagnostics: IluDiagnostics<R>,
    operation: &'static str,
) -> Result<(), SolverError> {
    if diagnostics.non_finite_entries != 0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "{} non-finite matrix entries",
                diagnostics.non_finite_entries
            ),
        });
    }
    if diagnostics.skipped_pivots != 0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "{} pivots were below the configured relative threshold",
                diagnostics.skipped_pivots
            ),
        });
    }
    Ok(())
}

// ============================================================================
// Graph Coloring (Level Scheduling) ILU
// ============================================================================

/// Parallel ILU using Graph Coloring / Level Scheduling
///
/// This approach analyzes the sparsity pattern of L and U to find independent
/// rows that can be solved in parallel. Rows are grouped into "levels" where
/// all rows in a level depend only on rows from previous levels.
///
/// Good for matrices with limited fill-in and structured sparsity patterns.
#[derive(Debug, Clone)]
pub struct IluColoringPreconditioner<T: ComplexField> {
    /// Lower triangular factor values
    l_values: Vec<T>,
    l_col_indices: Vec<u32>,
    l_row_ptrs: Vec<usize>,
    /// Upper triangular factor values (off-diagonal only, j > i)
    u_values: Vec<T>,
    u_col_indices: Vec<u32>,
    u_row_ptrs: Vec<usize>,
    /// Inverse diagonal of U (phase-preserving fallback for near-zero pivots)
    u_diag_inv: Vec<T>,
    /// Exact diagonal U retained for portable factor checkpoints.
    u_diag: Vec<T>,
    /// Forward solve levels: each level contains row indices that can be solved in parallel
    forward_levels: Vec<Vec<usize>>,
    /// Backward solve levels: each level contains row indices that can be solved in parallel
    backward_levels: Vec<Vec<usize>>,
    /// Matrix dimension
    n: usize,
    /// Canonical setup diagnostics retained for checked construction.
    diagnostics: IluDiagnostics<T::Real>,
}

impl<T: ComplexField> IluColoringPreconditioner<T> {
    /// Create ILU preconditioner with graph coloring from a CSR matrix
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Self {
        let n = matrix.num_rows;

        // Canonical ILU(0) factorization (shared with IluPreconditioner).
        let factors = ilu0_factorize(matrix);

        // Compute level scheduling for L (forward solve)
        let forward_levels = compute_forward_levels(n, &factors.l_col_indices, &factors.l_row_ptrs);

        // Compute level scheduling for U (backward solve)
        let backward_levels =
            compute_backward_levels(n, &factors.u_col_indices, &factors.u_row_ptrs);

        Self {
            l_values: factors.l_values,
            l_col_indices: factors.l_col_indices,
            l_row_ptrs: factors.l_row_ptrs,
            u_values: factors.u_values,
            u_col_indices: factors.u_col_indices,
            u_row_ptrs: factors.u_row_ptrs,
            u_diag_inv: factors.u_diag_inv,
            u_diag: factors.u_diag,
            forward_levels,
            backward_levels,
            n,
            diagnostics: factors.diagnostics,
        }
    }

    /// Fallible graph-coloring ILU construction.
    pub fn try_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        validate_parallel_ilu_matrix(matrix, "colored ILU setup")?;
        let preconditioner = Self::from_csr(matrix);
        check_parallel_ilu_diagnostics(preconditioner.diagnostics, "colored ILU setup")?;
        Ok(preconditioner)
    }

    /// Get statistics about the level structure
    pub fn level_stats(&self) -> (usize, usize, f64, f64) {
        let fwd_levels = self.forward_levels.len();
        let bwd_levels = self.backward_levels.len();
        let avg_fwd = self.n as f64 / fwd_levels as f64;
        let avg_bwd = self.n as f64 / bwd_levels as f64;
        (fwd_levels, bwd_levels, avg_fwd, avg_bwd)
    }

    /// Setup diagnostics retained by both compatibility and checked
    /// constructors. Checked construction rejects non-zero failure counters;
    /// compatibility callers can inspect the policy that was applied.
    pub fn diagnostics(&self) -> &IluDiagnostics<T::Real> {
        &self.diagnostics
    }

    /// Serialize the colored ILU factors and level schedule into a portable
    /// checkpoint.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let payload = parallel_ilu_payload(Ilu0Factors {
            l_values: self.l_values.clone(),
            l_col_indices: self.l_col_indices.clone(),
            l_row_ptrs: self.l_row_ptrs.clone(),
            u_values: self.u_values.clone(),
            u_col_indices: self.u_col_indices.clone(),
            u_row_ptrs: self.u_row_ptrs.clone(),
            u_diag: self.u_diag.clone(),
            u_diag_inv: self.u_diag_inv.clone(),
            n: self.n,
            diagnostics: self.diagnostics,
        })?;
        let mut output = Vec::new();
        output.extend_from_slice(ILU_COLORING_CHECKPOINT_MAGIC);
        output.extend_from_slice(&ILU_COLORING_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(0);
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(0);
        parilut_checkpoint_push_u64(&mut output, payload.len())?;
        output.extend_from_slice(&payload);
        parallel_ilu_push_levels(&mut output, &self.forward_levels)?;
        parallel_ilu_push_levels(&mut output, &self.backward_levels)?;
        Ok(output)
    }

    /// Restore colored ILU factors and level schedules from a checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = ParIlutCheckpointReader::new(bytes);
        if reader.take(ILU_COLORING_CHECKPOINT_MAGIC.len())? != ILU_COLORING_CHECKPOINT_MAGIC {
            return Err(parilut_checkpoint_error(
                "invalid colored ILU checkpoint magic",
            ));
        }
        if reader.read_u16()? != ILU_COLORING_CHECKPOINT_FORMAT_VERSION {
            return Err(parilut_checkpoint_error(
                "unsupported colored ILU checkpoint version",
            ));
        }
        if reader.read_u8()? != 0
            || reader.read_u8()? != std::mem::size_of::<T>() as u8
            || reader.read_u8()? != std::mem::size_of::<T::Real>() as u8
            || reader.read_u8()? != 0
        {
            return Err(parilut_checkpoint_error(
                "invalid colored ILU checkpoint header",
            ));
        }
        let payload_length = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("colored ILU payload length overflow"))?;
        let factors = parallel_ilu_factors(reader.take(payload_length)?)?;
        let forward_levels = parallel_ilu_read_levels(&mut reader, factors.n)?;
        let backward_levels = parallel_ilu_read_levels(&mut reader, factors.n)?;
        if !reader.is_at_end() {
            return Err(parilut_checkpoint_error(
                "trailing colored ILU checkpoint bytes",
            ));
        }
        if forward_levels
            != compute_forward_levels(factors.n, &factors.l_col_indices, &factors.l_row_ptrs)
            || backward_levels
                != compute_backward_levels(factors.n, &factors.u_col_indices, &factors.u_row_ptrs)
        {
            return Err(parilut_checkpoint_error(
                "colored ILU level schedule does not match factors",
            ));
        }
        Ok(Self {
            l_values: factors.l_values,
            l_col_indices: factors.l_col_indices,
            l_row_ptrs: factors.l_row_ptrs,
            u_values: factors.u_values,
            u_col_indices: factors.u_col_indices,
            u_row_ptrs: factors.u_row_ptrs,
            u_diag_inv: factors.u_diag_inv,
            u_diag: factors.u_diag,
            forward_levels,
            backward_levels,
            n: factors.n,
            diagnostics: factors.diagnostics,
        })
    }
}

/// Compute forward substitution levels (for lower triangular L)
fn compute_forward_levels(
    n: usize,
    l_col_indices: &[u32],
    l_row_ptrs: &[usize],
) -> Vec<Vec<usize>> {
    // Level of each row: level[i] = max(level[j] for all j where L[i,j] != 0) + 1
    let mut level = vec![0usize; n];

    for i in 0..n {
        let mut max_dep_level = 0;
        for &j in &l_col_indices[l_row_ptrs[i]..l_row_ptrs[i + 1]] {
            max_dep_level = max_dep_level.max(level[j as usize] + 1);
        }
        level[i] = max_dep_level;
    }

    // Group rows by level
    let max_level = *level.iter().max().unwrap_or(&0);
    let mut levels = vec![Vec::new(); max_level + 1];
    for (i, &lvl) in level.iter().enumerate() {
        levels[lvl].push(i);
    }

    levels
}

/// Compute backward substitution levels (for upper triangular U)
fn compute_backward_levels(
    n: usize,
    u_col_indices: &[u32],
    u_row_ptrs: &[usize],
) -> Vec<Vec<usize>> {
    // For backward solve, we process in reverse order
    // Level of each row: level[i] = max(level[j] for all j > i where U[i,j] != 0) + 1
    // U stores only off-diagonal entries (j > i), so all entries qualify.
    let mut level = vec![0usize; n];

    for i in (0..n).rev() {
        let mut max_dep_level = 0;
        for &j in &u_col_indices[u_row_ptrs[i]..u_row_ptrs[i + 1]] {
            max_dep_level = max_dep_level.max(level[j as usize] + 1);
        }
        level[i] = max_dep_level;
    }

    // Group rows by level
    let max_level = *level.iter().max().unwrap_or(&0);
    let mut levels = vec![Vec::new(); max_level + 1];
    for (i, &lvl) in level.iter().enumerate() {
        levels[lvl].push(i);
    }

    levels
}

impl<T: ComplexField + Send + Sync> Preconditioner<T> for IluColoringPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(r.len(), self.n, "ILU coloring RHS length mismatch");
        assert_eq!(y.len(), self.n, "ILU coloring output length mismatch");

        if self.n >= 246 {
            self.apply_parallel_into(r, y);
        } else {
            self.apply_sequential_into(r, y);
        }
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_adjoint_sequential_into(r, &mut y, false);
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.apply_adjoint_sequential_into(r, y, false);
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_adjoint_sequential_into(r, &mut y, true);
        y
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        self.apply_adjoint_sequential_into(r, y, true);
    }
}

impl<T: ComplexField + Send + Sync> IluColoringPreconditioner<T> {
    fn apply_sequential_into(&self, r: &[T], y: &mut [T]) {
        y.copy_from_slice(r);

        // Forward substitution
        for i in 0..self.n {
            for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                let j = self.l_col_indices[idx] as usize;
                let l_ij = self.l_values[idx];
                y[i] -= l_ij * y[j];
            }
        }

        // Backward substitution
        // U stores only off-diagonal entries (j > i); diagonal handled via u_diag_inv.
        for i in (0..self.n).rev() {
            for idx in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                let j = self.u_col_indices[idx] as usize;
                let u_ij = self.u_values[idx];
                y[i] -= u_ij * y[j];
            }

            y[i] *= self.u_diag_inv[i];
        }
    }

    fn apply_parallel_into(&self, r: &[T], y: &mut [T]) {
        // Wrapper to allow shared mutable access (safe because each level
        // writes each row at most once, after dependencies from earlier levels).
        #[derive(Clone, Copy)]
        struct UnsafeSlice<U>(*mut U);
        unsafe impl<U: Send> Sync for UnsafeSlice<U> {}

        impl<U: Copy> UnsafeSlice<U> {
            fn get(&self, i: usize) -> U {
                unsafe { *self.0.add(i) }
            }
            fn set(&self, i: usize, val: U) {
                unsafe { *self.0.add(i) = val }
            }
        }

        y.copy_from_slice(r);
        let y = UnsafeSlice(y.as_mut_ptr());

        // Per-level parallelism only pays when each worker has enough work
        // to amortise the fork/join cost. With many short level sets a
        // fixed `> 32` threshold dispatches tiny par_iters that wreck
        // scaling on high-core-count machines (negative apply× past 4
        // threads). Require ≥ MIN_PER_WORKER rows per available thread.
        const MIN_PER_WORKER: usize = 128;
        let par_threshold = rayon::current_num_threads().saturating_mul(MIN_PER_WORKER);

        // Forward substitution by levels
        for level in &self.forward_levels {
            if level.len() >= par_threshold {
                // Parallel within level
                level.par_iter().for_each(|&i| {
                    let mut sum = T::zero();
                    for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                        let j = self.l_col_indices[idx] as usize;
                        let l_ij = self.l_values[idx];
                        // Safe: j < i always for lower triangular, all j in previous levels
                        sum += l_ij * y.get(j);
                    }
                    // Safe: each thread writes to unique index
                    y.set(i, y.get(i) - sum);
                });
            } else {
                // Sequential for small levels
                for &i in level {
                    let mut sum = T::zero();
                    for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                        let j = self.l_col_indices[idx] as usize;
                        let l_ij = self.l_values[idx];
                        sum += l_ij * y.get(j);
                    }
                    y.set(i, y.get(i) - sum);
                }
            }
        }

        // Backward substitution by levels
        // U stores only off-diagonal entries (j > i); diagonal handled via u_diag_inv.
        for level in &self.backward_levels {
            if level.len() >= par_threshold {
                let u_diag_inv = &self.u_diag_inv;
                level.par_iter().for_each(|&i| {
                    let mut sum = T::zero();
                    for idx in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                        let j = self.u_col_indices[idx] as usize;
                        let u_ij = self.u_values[idx];
                        sum += u_ij * y.get(j);
                    }
                    let xi = y.get(i) - sum;
                    y.set(i, xi * u_diag_inv[i]);
                });
            } else {
                for &i in level {
                    let mut sum = T::zero();
                    for idx in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                        let j = self.u_col_indices[idx] as usize;
                        let u_ij = self.u_values[idx];
                        sum += u_ij * y.get(j);
                    }
                    let xi = y.get(i) - sum;
                    y.set(i, xi * self.u_diag_inv[i]);
                }
            }
        }
    }

    /// Apply the exact transpose or Hermitian action of the stored ILU
    /// factors. The coloring schedule is specific to forward triangular
    /// dependencies; the adjoint path uses the equivalent O(nnz) sequential
    /// scatter solve rather than advertising a schedule that is not valid for
    /// the transposed dependency graph.
    fn apply_adjoint_sequential_into(&self, r: &[T], y: &mut [T], hermitian: bool) {
        assert_eq!(r.len(), self.n, "ILU coloring adjoint RHS length mismatch");
        assert_eq!(
            y.len(),
            self.n,
            "ILU coloring adjoint output length mismatch"
        );
        y.copy_from_slice(r);

        // Uᵀ/Uᴴ is lower triangular.
        for i in 0..self.n {
            y[i] = if hermitian {
                y[i] * self.u_diag_inv[i].conj()
            } else {
                y[i] * self.u_diag_inv[i]
            };
            for idx in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                let j = self.u_col_indices[idx] as usize;
                let value = if hermitian {
                    self.u_values[idx].conj()
                } else {
                    self.u_values[idx]
                };
                y[j] -= value * y[i];
            }
        }

        // Lᵀ/Lᴴ is upper triangular with unit diagonal.
        for i in (0..self.n).rev() {
            for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                let j = self.l_col_indices[idx] as usize;
                let value = if hermitian {
                    self.l_values[idx].conj()
                } else {
                    self.l_values[idx]
                };
                y[j] -= value * y[i];
            }
        }
    }
}

// ============================================================================
// Fine-Grained Fixed-Point Iteration ILU
// ============================================================================

/// Diagnostics for the fixed-point triangular solves (cumulative counters).
///
/// Shared across clones of the preconditioner; updated on every `apply`.
#[derive(Debug, Default)]
pub struct IluFixedPointStats {
    /// Total Jacobi sweeps executed (forward + backward).
    sweeps: AtomicUsize,
    /// Number of times a diverging sweep loop fell back to exact substitution.
    fallbacks: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriangularSolve {
    Forward,
    Backward,
}

/// Parallel ILU using Fine-Grained Fixed-Point Iteration
///
/// Instead of exact triangular solves, this approach uses Jacobi-style
/// fixed-point iteration: x^{k+1} = D^{-1}(b - (L+U)x^k)
///
/// Each iteration is embarrassingly parallel. Typically 2-5 iterations
/// are enough for a good approximation.
///
/// The sweeps are convergence-checked: the iteration stops early once the
/// fixpoint residual `||x^{k+1} - x^k||` drops below
/// `tolerance * ||rhs||` (note the fixpoint residual equals the true
/// triangular-solve residual `||b - L x^k||` for the unit-diagonal forward
/// solve, and `||y - U x^k||` scaled by D^{-1} for the backward solve).
/// If the residual grows for two consecutive sweeps (or becomes non-finite)
/// the iteration is diverging: a warning is emitted and the apply falls back
/// to exact sequential substitution, so a hostile matrix degrades to the
/// sequential ILU instead of returning garbage.
///
/// This trades exact solves for parallelism and can be faster on
/// highly parallel hardware.
#[derive(Debug, Clone)]
pub struct IluFixedPointPreconditioner<T: ComplexField> {
    /// Lower triangular factor values
    l_values: Vec<T>,
    l_col_indices: Vec<u32>,
    l_row_ptrs: Vec<usize>,
    /// Upper triangular factor values (excluding diagonal)
    u_off_values: Vec<T>,
    u_off_col_indices: Vec<u32>,
    u_off_row_ptrs: Vec<usize>,
    /// Inverse diagonal of U (for fast multiplication)
    u_diag_inv: Vec<T>,
    /// Exact diagonal U retained for portable factor checkpoints.
    u_diag: Vec<T>,
    /// Maximum number of fixed-point sweeps per triangular solve
    iterations: usize,
    /// Relative fixpoint-residual tolerance for early exit
    tolerance: T::Real,
    /// Cumulative sweep/fallback diagnostics
    stats: Arc<IluFixedPointStats>,
    /// Matrix dimension
    n: usize,
    /// Canonical setup diagnostics retained for checked construction.
    diagnostics: IluDiagnostics<T::Real>,
}

impl<T: ComplexField> IluFixedPointPreconditioner<T> {
    /// Create ILU preconditioner with fixed-point iteration
    ///
    /// # Arguments
    /// * `matrix` - The sparse matrix to precondition
    /// * `iterations` - Maximum number of fixed-point sweeps per triangular
    ///   solve (typically 2-5). Sweeps stop early on convergence; on
    ///   divergence the apply falls back to exact sequential substitution.
    pub fn from_csr(matrix: &CsrMatrix<T>, iterations: usize) -> Self {
        // Tight default: only stop early when the iteration has essentially
        // reached substitution precision, so the default behavior matches the
        // previous fixed-sweep-count semantics on typical (convergent) inputs.
        Self::from_csr_with_tolerance(matrix, iterations, 1e-12)
    }

    /// Create with an explicit relative fixpoint-residual tolerance.
    ///
    /// Larger tolerances (e.g. 1e-2) stop the sweeps as soon as the iterate
    /// stabilises at preconditioner accuracy, saving sweeps on easy systems.
    pub fn from_csr_with_tolerance(
        matrix: &CsrMatrix<T>,
        iterations: usize,
        tolerance: f64,
    ) -> Self {
        let n = matrix.num_rows;

        // Canonical ILU(0) factorization (shared with IluPreconditioner).
        let factors = ilu0_factorize(matrix);

        Self {
            l_values: factors.l_values,
            l_col_indices: factors.l_col_indices,
            l_row_ptrs: factors.l_row_ptrs,
            u_off_values: factors.u_values,
            u_off_col_indices: factors.u_col_indices,
            u_off_row_ptrs: factors.u_row_ptrs,
            u_diag_inv: factors.u_diag_inv,
            u_diag: factors.u_diag,
            iterations,
            tolerance: T::Real::from_f64(tolerance).unwrap(),
            stats: Arc::new(IluFixedPointStats::default()),
            n,
            diagnostics: factors.diagnostics,
        }
    }

    /// Fallible fixed-point ILU construction with the default tolerance.
    pub fn try_from_csr(matrix: &CsrMatrix<T>, iterations: usize) -> Result<Self, SolverError> {
        Self::try_from_csr_with_tolerance(matrix, iterations, 1e-12)
    }

    /// Fallible fixed-point ILU construction with explicit tolerance.
    pub fn try_from_csr_with_tolerance(
        matrix: &CsrMatrix<T>,
        iterations: usize,
        tolerance: f64,
    ) -> Result<Self, SolverError> {
        validate_parallel_ilu_matrix(matrix, "fixed-point ILU setup")?;
        if iterations == 0 {
            return Err(SolverError::OperationFailed {
                operation: "fixed-point ILU setup",
                reason: "iterations must be greater than zero".to_string(),
            });
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "fixed-point ILU setup",
                reason: "tolerance must be finite and non-negative".to_string(),
            });
        }
        let preconditioner = Self::from_csr_with_tolerance(matrix, iterations, tolerance);
        check_parallel_ilu_diagnostics(preconditioner.diagnostics, "fixed-point ILU setup")?;
        Ok(preconditioner)
    }

    /// Create with default 3 iterations
    pub fn from_csr_default(matrix: &CsrMatrix<T>) -> Self {
        Self::from_csr(matrix, 3)
    }

    /// Cumulative diagnostics: `(sweeps executed, divergence fallbacks)`.
    pub fn stats(&self) -> (usize, usize) {
        (
            self.stats.sweeps.load(Ordering::Relaxed),
            self.stats.fallbacks.load(Ordering::Relaxed),
        )
    }

    /// Setup diagnostics retained by the canonical ILU(0) factorization.
    pub fn diagnostics(&self) -> &IluDiagnostics<T::Real> {
        &self.diagnostics
    }

    /// Serialize fixed-point ILU factors, iteration policy, and runtime counters.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        if !self.tolerance.is_finite() || self.tolerance < T::Real::zero() {
            return Err(parilut_checkpoint_error(
                "fixed-point ILU tolerance is invalid",
            ));
        }
        let payload = parallel_ilu_payload(Ilu0Factors {
            l_values: self.l_values.clone(),
            l_col_indices: self.l_col_indices.clone(),
            l_row_ptrs: self.l_row_ptrs.clone(),
            u_values: self.u_off_values.clone(),
            u_col_indices: self.u_off_col_indices.clone(),
            u_row_ptrs: self.u_off_row_ptrs.clone(),
            u_diag: self.u_diag.clone(),
            u_diag_inv: self.u_diag_inv.clone(),
            n: self.n,
            diagnostics: self.diagnostics,
        })?;
        let tolerance = self
            .tolerance
            .to_f64()
            .ok_or_else(|| parilut_checkpoint_error("fixed-point tolerance cannot convert"))?;
        let mut output = Vec::new();
        output.extend_from_slice(ILU_FIXED_POINT_CHECKPOINT_MAGIC);
        output.extend_from_slice(&ILU_FIXED_POINT_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&[
            0,
            std::mem::size_of::<T>() as u8,
            std::mem::size_of::<T::Real>() as u8,
            0,
        ]);
        parilut_checkpoint_push_u64(&mut output, payload.len())?;
        output.extend_from_slice(&payload);
        parilut_checkpoint_push_u64(&mut output, self.iterations)?;
        output.extend_from_slice(&tolerance.to_le_bytes());
        parilut_checkpoint_push_u64(&mut output, self.stats.sweeps.load(Ordering::Relaxed))?;
        parilut_checkpoint_push_u64(&mut output, self.stats.fallbacks.load(Ordering::Relaxed))?;
        Ok(output)
    }

    /// Restore fixed-point ILU factors, policy, and counters from a checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = ParIlutCheckpointReader::new(bytes);
        if reader.take(ILU_FIXED_POINT_CHECKPOINT_MAGIC.len())? != ILU_FIXED_POINT_CHECKPOINT_MAGIC
        {
            return Err(parilut_checkpoint_error(
                "invalid fixed-point ILU checkpoint magic",
            ));
        }
        if reader.read_u16()? != ILU_FIXED_POINT_CHECKPOINT_FORMAT_VERSION {
            return Err(parilut_checkpoint_error(
                "unsupported fixed-point ILU checkpoint version",
            ));
        }
        if reader.read_u8()? != 0
            || reader.read_u8()? != std::mem::size_of::<T>() as u8
            || reader.read_u8()? != std::mem::size_of::<T::Real>() as u8
            || reader.read_u8()? != 0
        {
            return Err(parilut_checkpoint_error(
                "invalid fixed-point ILU checkpoint header",
            ));
        }
        let payload_length = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("fixed-point ILU payload length overflow"))?;
        let factors = parallel_ilu_factors(reader.take(payload_length)?)?;
        let iterations = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("fixed-point iterations overflow"))?;
        let tolerance = T::Real::from_f64(reader.read_f64()?)
            .ok_or_else(|| parilut_checkpoint_error("fixed-point tolerance cannot decode"))?;
        if !tolerance.is_finite() || tolerance < T::Real::zero() {
            return Err(parilut_checkpoint_error("fixed-point tolerance is invalid"));
        }
        let sweeps = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("fixed-point sweep count overflow"))?;
        let fallbacks = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("fixed-point fallback count overflow"))?;
        if !reader.is_at_end() {
            return Err(parilut_checkpoint_error(
                "trailing fixed-point ILU checkpoint bytes",
            ));
        }
        Ok(Self {
            l_values: factors.l_values,
            l_col_indices: factors.l_col_indices,
            l_row_ptrs: factors.l_row_ptrs,
            u_off_values: factors.u_values,
            u_off_col_indices: factors.u_col_indices,
            u_off_row_ptrs: factors.u_row_ptrs,
            u_diag: factors.u_diag,
            u_diag_inv: factors.u_diag_inv,
            iterations,
            tolerance,
            stats: Arc::new(IluFixedPointStats {
                sweeps: AtomicUsize::new(sweeps),
                fallbacks: AtomicUsize::new(fallbacks),
            }),
            n: factors.n,
            diagnostics: factors.diagnostics,
        })
    }

    fn vector_norm(v: &[T]) -> T::Real {
        let mut scale = T::Real::zero();
        let mut sum_squares = T::Real::one();
        for value in v {
            update_scaled_norm(&mut scale, &mut sum_squares, value.norm());
        }
        finish_scaled_norm(scale, sum_squares)
    }
}

impl<T: ComplexField + Send + Sync> Preconditioner<T> for IluFixedPointPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(r.len(), self.n, "ILU fixed-point RHS length mismatch");
        assert_eq!(y.len(), self.n, "ILU fixed-point output length mismatch");

        if self.n >= 246 {
            self.apply_parallel_into(r, y);
        } else {
            self.apply_sequential_into(r, y);
        }
    }
}

/// Status of the convergence-checked sweep loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SweepStatus {
    Continue,
    Converged,
    Diverged,
}

/// Tracks the fixpoint residual `||x^{k+1} - x^k||` across sweeps: stops
/// early once it drops below `tolerance * scale`, and flags divergence when
/// the residual grows for two consecutive sweeps or becomes non-finite.
struct SweepMonitor<R: Float> {
    tolerance: R,
    scale: R,
    prev_delta: Option<R>,
    growth: usize,
}

impl<R: Float + FromPrimitive> SweepMonitor<R> {
    fn new(tolerance: R, scale: R) -> Self {
        Self {
            tolerance,
            scale: scale.max(R::from_f64(1e-30).unwrap()),
            prev_delta: None,
            growth: 0,
        }
    }

    fn update(&mut self, delta: R) -> SweepStatus {
        if delta <= self.tolerance * self.scale {
            return SweepStatus::Converged;
        }
        if !delta.is_finite() {
            return SweepStatus::Diverged;
        }
        if let Some(prev) = self.prev_delta {
            if delta > prev {
                self.growth += 1;
            } else {
                self.growth = 0;
            }
        }
        self.prev_delta = Some(delta);
        if self.growth >= 2 {
            return SweepStatus::Diverged;
        }
        SweepStatus::Continue
    }
}

impl<T: ComplexField + Send + Sync> IluFixedPointPreconditioner<T> {
    fn vector_norm_diff(a: &[T], b: &[T]) -> T::Real {
        let mut scale = T::Real::zero();
        let mut sum_squares = T::Real::one();
        for (&x, &y) in a.iter().zip(b) {
            update_scaled_norm(&mut scale, &mut sum_squares, (x - y).norm());
        }
        finish_scaled_norm(scale, sum_squares)
    }

    fn vector_norm_diff_parallel(a: &[T], b: &[T]) -> T::Real {
        let (scale, sum_squares) = a
            .par_iter()
            .zip(b.par_iter())
            .fold(
                || (T::Real::zero(), T::Real::one()),
                |(mut scale, mut sum_squares), (&x, &y)| {
                    update_scaled_norm(&mut scale, &mut sum_squares, (x - y).norm());
                    (scale, sum_squares)
                },
            )
            .reduce(
                || (T::Real::zero(), T::Real::one()),
                |(mut scale, mut sum_squares), (other_scale, other_sum_squares)| {
                    merge_scaled_norm(&mut scale, &mut sum_squares, other_scale, other_sum_squares);
                    (scale, sum_squares)
                },
            );
        finish_scaled_norm(scale, sum_squares)
    }

    /// Approximate `L^{-1} r` (unit-diagonal L) with convergence-checked
    /// Jacobi sweeps. Returns `None` when the iteration diverges.
    fn forward_sweeps_sequential(&self, r: &[T]) -> Option<Vec<T>> {
        let mut monitor = SweepMonitor::new(self.tolerance, Self::vector_norm(r));
        let mut y = r.to_vec();
        let mut y_new = vec![T::zero(); self.n];

        for _ in 0..self.iterations {
            for i in 0..self.n {
                let mut sum = r[i];
                for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                    let j = self.l_col_indices[idx] as usize;
                    sum -= self.l_values[idx] * y[j];
                }
                y_new[i] = sum;
            }
            self.stats.sweeps.fetch_add(1, Ordering::Relaxed);
            let delta = Self::vector_norm_diff(&y_new, &y);
            std::mem::swap(&mut y, &mut y_new);
            match monitor.update(delta) {
                SweepStatus::Continue => {}
                SweepStatus::Converged => return Some(y),
                SweepStatus::Diverged => return None,
            }
        }
        Some(y)
    }

    /// Approximate `U^{-1} y` (U = D + strict upper) with convergence-checked
    /// Jacobi sweeps. Returns `None` when the iteration diverges.
    fn backward_sweeps_sequential(&self, y: &[T]) -> Option<Vec<T>> {
        let mut monitor = SweepMonitor::new(self.tolerance, Self::vector_norm(y));
        let mut x: Vec<T> = y
            .iter()
            .zip(self.u_diag_inv.iter())
            .map(|(&yi, &di)| yi * di)
            .collect();
        let mut x_new = vec![T::zero(); self.n];

        for _ in 0..self.iterations {
            for i in (0..self.n).rev() {
                let mut sum = y[i];
                for idx in self.u_off_row_ptrs[i]..self.u_off_row_ptrs[i + 1] {
                    let j = self.u_off_col_indices[idx] as usize;
                    sum -= self.u_off_values[idx] * x[j];
                }
                x_new[i] = sum * self.u_diag_inv[i];
            }
            self.stats.sweeps.fetch_add(1, Ordering::Relaxed);
            let delta = Self::vector_norm_diff(&x_new, &x);
            std::mem::swap(&mut x, &mut x_new);
            match monitor.update(delta) {
                SweepStatus::Continue => {}
                SweepStatus::Converged => return Some(x),
                SweepStatus::Diverged => return None,
            }
        }
        Some(x)
    }

    fn forward_sweeps_parallel(&self, r: &[T]) -> Option<Vec<T>> {
        let mut monitor = SweepMonitor::new(self.tolerance, Self::vector_norm(r));
        let mut y = r.to_vec();
        let mut y_new = vec![T::zero(); self.n];

        for _ in 0..self.iterations {
            y_new.par_iter_mut().enumerate().for_each(|(i, yi_new)| {
                let mut sum = r[i];
                for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                    let j = self.l_col_indices[idx] as usize;
                    sum -= self.l_values[idx] * y[j];
                }
                *yi_new = sum;
            });
            self.stats.sweeps.fetch_add(1, Ordering::Relaxed);
            let delta = Self::vector_norm_diff_parallel(&y_new, &y);
            std::mem::swap(&mut y, &mut y_new);
            match monitor.update(delta) {
                SweepStatus::Continue => {}
                SweepStatus::Converged => return Some(y),
                SweepStatus::Diverged => return None,
            }
        }
        Some(y)
    }

    fn backward_sweeps_parallel(&self, y: &[T]) -> Option<Vec<T>> {
        let mut monitor = SweepMonitor::new(self.tolerance, Self::vector_norm(y));
        let mut x: Vec<T> = y
            .iter()
            .zip(self.u_diag_inv.iter())
            .map(|(&yi, &di)| yi * di)
            .collect();
        let mut x_new = vec![T::zero(); self.n];

        for _ in 0..self.iterations {
            x_new.par_iter_mut().enumerate().for_each(|(i, xi_new)| {
                let mut sum = y[i];
                for idx in self.u_off_row_ptrs[i]..self.u_off_row_ptrs[i + 1] {
                    let j = self.u_off_col_indices[idx] as usize;
                    sum -= self.u_off_values[idx] * x[j];
                }
                *xi_new = sum * self.u_diag_inv[i];
            });
            self.stats.sweeps.fetch_add(1, Ordering::Relaxed);
            let delta = Self::vector_norm_diff_parallel(&x_new, &x);
            std::mem::swap(&mut x, &mut x_new);
            match monitor.update(delta) {
                SweepStatus::Continue => {}
                SweepStatus::Converged => return Some(x),
                SweepStatus::Diverged => return None,
            }
        }
        Some(x)
    }

    /// Exact sequential forward/backward substitution (divergence fallback).
    fn solve_exact_into(&self, r: &[T], y_out: &mut [T]) {
        y_out.copy_from_slice(r);

        // Forward substitution (L has unit diagonal)
        for i in 0..self.n {
            for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                let j = self.l_col_indices[idx] as usize;
                y_out[i] -= self.l_values[idx] * y_out[j];
            }
        }

        // Backward substitution (U = D + strict upper)
        for i in (0..self.n).rev() {
            let mut sum = T::zero();
            for idx in self.u_off_row_ptrs[i]..self.u_off_row_ptrs[i + 1] {
                let j = self.u_off_col_indices[idx] as usize;
                sum += self.u_off_values[idx] * y_out[j];
            }
            y_out[i] = (y_out[i] - sum) * self.u_diag_inv[i];
        }
    }

    fn fallback(&self, solve: TriangularSolve, r: &[T], y_out: &mut [T]) {
        self.stats.fallbacks.fetch_add(1, Ordering::Relaxed);
        log::warn!(
            "ILU fixed-point {solve:?} solve diverged; \
             falling back to exact sequential substitution"
        );
        self.solve_exact_into(r, y_out);
    }

    fn apply_sequential_into(&self, r: &[T], y_out: &mut [T]) {
        // Approximate the triangular solves (L*y = r,  U*x = y) via
        // convergence-checked fixed-point sweeps. Each sweep is a Jacobi
        // iteration over the triangular system, which is embarrassingly
        // parallel. Divergence falls back to exact sequential substitution.
        let Some(y) = self.forward_sweeps_sequential(r) else {
            self.fallback(TriangularSolve::Forward, r, y_out);
            return;
        };
        let Some(x) = self.backward_sweeps_sequential(&y) else {
            self.fallback(TriangularSolve::Backward, r, y_out);
            return;
        };
        y_out.copy_from_slice(&x);
    }

    fn apply_parallel_into(&self, r: &[T], y_out: &mut [T]) {
        let Some(y) = self.forward_sweeps_parallel(r) else {
            self.fallback(TriangularSolve::Forward, r, y_out);
            return;
        };
        let Some(x) = self.backward_sweeps_parallel(&y) else {
            self.fallback(TriangularSolve::Backward, r, y_out);
            return;
        };
        y_out.copy_from_slice(&x);
    }
}

// ============================================================================
// ParILUT: Parallel Fixed-Point ILU with Threshold (Chow/Anzt/Dongarra)
// ============================================================================

/// Configuration for [`ParIlutPreconditioner`].
#[derive(Debug, Clone)]
pub struct ParIlutConfig {
    /// Number of parallel fixed-point sweeps over the factor values
    /// (typically 1-10). Each sweep performs one Jacobi-style update of every
    /// L/U entry from the previous iterate, driving `L U` towards the exact
    /// LU factorization restricted to the current pattern.
    pub sweeps: usize,
    /// Chow's threshold-based pattern augmentation: after each sweep, the
    /// largest-magnitude entries of the residual `R = A - L U` that fall
    /// outside the current pattern are added as zero-valued candidates, and
    /// weak off-diagonal pattern entries that are not part of the pattern of
    /// A are dropped.
    pub add_candidates: bool,
    /// Maximum candidate entries added per matrix row per sweep.
    pub max_candidates_per_row: usize,
    /// Relative threshold (scaled by the row norm of A) used both for
    /// selecting residual candidates and for dropping weak pattern entries.
    pub drop_tolerance: f64,
    /// Record `||A - L U||_F` before the first sweep and after every sweep
    /// (see [`ParIlutPreconditioner::residual_history`]). Costs one sparse
    /// product evaluation per sweep; off by default.
    pub track_residual: bool,
}

impl Default for ParIlutConfig {
    fn default() -> Self {
        Self {
            sweeps: 5,
            add_candidates: true,
            max_candidates_per_row: 3,
            drop_tolerance: 1e-4,
            track_residual: false,
        }
    }
}

/// Per-row L/U pattern entries produced by the augmentation pass:
/// `(new L row, new U row)` per matrix row.
type PatternRows<T> = Vec<(Vec<(u32, T)>, Vec<(u32, T)>)>;

/// Working L/U factors for the ParILUT fixed-point iteration.
///
/// Unlike [`Ilu0Factors`], U stores the diagonal explicitly (first entry of
/// each row, columns sorted) so the Jacobi update can read the current
/// pivot. The pattern is rebuilt whenever candidates are added or dropped.
#[derive(Debug, Clone)]
struct ParLuFactors<T: ComplexField> {
    /// Lower triangular factor (unit diagonal; strictly lower entries j < i), CSR.
    l_values: Vec<T>,
    l_col_indices: Vec<u32>,
    l_row_ptrs: Vec<usize>,
    /// Upper triangular factor including the diagonal (j >= i), CSR.
    u_values: Vec<T>,
    u_col_indices: Vec<u32>,
    u_row_ptrs: Vec<usize>,
    /// Row index of every flat L entry (for the parallel Jacobi update).
    l_entry_row: Vec<u32>,
    /// Row index of every flat U entry (for the parallel Jacobi update).
    u_entry_row: Vec<u32>,
    /// Diagnostics from the canonical ILU(0) setup, preserved through ParILUT.
    diagnostics: IluDiagnostics<T::Real>,
}

/// Dot product of an L row slice with a U column slice (CSC), restricted to
/// shared indices `k < k_max`. Both index slices must be sorted ascending.
fn pattern_dot<T: ComplexField>(
    l_cols: &[u32],
    l_vals: &[T],
    u_rows: &[u32],
    u_vals: &[T],
    k_max: usize,
) -> T {
    let mut sum = T::zero();
    let (mut a, mut b) = (0, 0);
    while a < l_cols.len() && b < u_rows.len() {
        let k_l = l_cols[a] as usize;
        let k_u = u_rows[b] as usize;
        if k_l >= k_max || k_u >= k_max {
            break;
        }
        match k_l.cmp(&k_u) {
            std::cmp::Ordering::Less => a += 1,
            std::cmp::Ordering::Greater => b += 1,
            std::cmp::Ordering::Equal => {
                sum += l_vals[a] * u_vals[b];
                a += 1;
                b += 1;
            }
        }
    }
    sum
}

/// Inverse of a U diagonal entry, using the same phase-preserving fallback
/// as the canonical ILU(0) kernel in `ilu.rs`.
fn invert_pivot<T: ComplexField>(d: T, row_norm: T::Real) -> T {
    let d_norm: T::Real = d.norm();
    if d_norm > T::Real::from_f64(1e-20).unwrap() {
        d.inv()
    } else {
        // Fallback: near-zero diagonal. Use row norm as scale so the inverse
        // has the right magnitude. Preserve the complex phase.
        let fallback: T::Real = row_norm.max(T::Real::from_f64(1.0).unwrap());
        if d_norm > T::Real::from_f64(1e-30).unwrap() {
            let scale: T::Real = T::Real::one() / (d_norm * fallback);
            d.conj() * T::from_real(scale)
        } else {
            // d is effectively zero; no phase to preserve
            T::from_real(fallback).inv()
        }
    }
}

impl<T: ComplexField + Send + Sync> ParLuFactors<T> {
    /// Build the initial guess from the canonical ILU(0) factorization,
    /// restoring the U diagonal (stored only as an inverse in
    /// [`Ilu0Factors`]) and adding a unit diagonal to rows that have no
    /// diagonal entry in the pattern of A.
    fn from_ilu0(factors: &Ilu0Factors<T>) -> Self {
        let n = factors.n;

        let u_diag = factors.u_diag.clone();

        // Merge the diagonal into the off-diagonal U pattern (both are
        // sorted, diagonal sorts first within its row).
        let mut u_values = Vec::with_capacity(factors.u_values.len() + n);
        let mut u_col_indices = Vec::with_capacity(factors.u_col_indices.len() + n);
        let mut u_row_ptrs = Vec::with_capacity(n + 1);
        u_row_ptrs.push(0);
        for (i, &diag) in u_diag[..n].iter().enumerate() {
            u_values.push(diag);
            u_col_indices.push(i as u32);
            for idx in factors.u_row_ptrs[i]..factors.u_row_ptrs[i + 1] {
                u_values.push(factors.u_values[idx]);
                u_col_indices.push(factors.u_col_indices[idx]);
            }
            u_row_ptrs.push(u_values.len());
        }

        let mut factors = Self {
            l_values: factors.l_values.clone(),
            l_col_indices: factors.l_col_indices.clone(),
            l_row_ptrs: factors.l_row_ptrs.clone(),
            u_values,
            u_col_indices,
            u_row_ptrs,
            l_entry_row: Vec::new(),
            u_entry_row: Vec::new(),
            diagnostics: factors.diagnostics,
        };
        factors.rebuild_entry_rows(n);
        factors
    }

    /// Recompute the flat-entry row indices after a pattern change.
    fn rebuild_entry_rows(&mut self, n: usize) {
        self.l_entry_row = (0..n)
            .flat_map(|i| {
                let count = self.l_row_ptrs[i + 1] - self.l_row_ptrs[i];
                std::iter::repeat_n(i as u32, count)
            })
            .collect();
        self.u_entry_row = (0..n)
            .flat_map(|i| {
                let count = self.u_row_ptrs[i + 1] - self.u_row_ptrs[i];
                std::iter::repeat_n(i as u32, count)
            })
            .collect();
    }

    /// Transpose U (including the diagonal) to CSC for column access during
    /// the fixed-point sweeps. Returns `(col_ptrs, row_indices, values)`.
    fn build_u_csc(&self, n: usize) -> (Vec<usize>, Vec<u32>, Vec<T>) {
        let nnz = self.u_values.len();
        let mut col_ptrs = vec![0usize; n + 1];
        for &col in &self.u_col_indices {
            col_ptrs[col as usize + 1] += 1;
        }
        for j in 0..n {
            col_ptrs[j + 1] += col_ptrs[j];
        }
        let mut row_indices = vec![0u32; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut next = col_ptrs[..n].to_vec();
        for i in 0..n {
            for idx in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                let col = self.u_col_indices[idx] as usize;
                let dest = next[col];
                row_indices[dest] = i as u32;
                values[dest] = self.u_values[idx];
                next[col] += 1;
            }
        }
        (col_ptrs, row_indices, values)
    }

    /// One parallel Jacobi sweep over all L and U values.
    ///
    /// For every pattern entry the ILU recurrence is applied using the
    /// *previous* iterate only:
    ///
    /// - U part (`i <= j`): `u_ij ← a_ij - Σ_{k<i} l_ik u_kj`
    /// - L part (`i > j`): `l_ij ← (a_ij - Σ_{k<j} l_ik u_kj) / u_jj`
    ///
    /// Rows with a numerically tiny pivot `u_jj` keep their old multiplier
    /// (same relative threshold as the canonical ILU(0) kernel).
    fn sweep(
        &mut self,
        a: &CsrMatrix<T>,
        u_csc_col_ptrs: &[usize],
        u_csc_row_indices: &[u32],
        u_csc_values: &[T],
        row_norms: &[T::Real],
    ) {
        let pivot_rel = T::Real::from_f64(1e-2).unwrap();
        let pivot_abs = T::Real::from_f64(1e-30).unwrap();

        let new_l: Vec<T> = (0..self.l_values.len())
            .into_par_iter()
            .map(|idx| {
                let i = self.l_entry_row[idx] as usize;
                let j = self.l_col_indices[idx] as usize;
                let l_range = self.l_row_ptrs[i]..self.l_row_ptrs[i + 1];
                let c_range = u_csc_col_ptrs[j]..u_csc_col_ptrs[j + 1];
                let sum = pattern_dot(
                    &self.l_col_indices[l_range.clone()],
                    &self.l_values[l_range],
                    &u_csc_row_indices[c_range.clone()],
                    &u_csc_values[c_range],
                    j,
                );
                // Diagonal is the first entry of U row j.
                let u_jj = self.u_values[self.u_row_ptrs[j]];
                let tol = pivot_rel * row_norms[j].max(pivot_abs);
                if u_jj.norm() < tol {
                    return self.l_values[idx];
                }
                (a.get(i, j) - sum) * u_jj.inv()
            })
            .collect();

        let new_u: Vec<T> = (0..self.u_values.len())
            .into_par_iter()
            .map(|idx| {
                let i = self.u_entry_row[idx] as usize;
                let j = self.u_col_indices[idx] as usize;
                let l_range = self.l_row_ptrs[i]..self.l_row_ptrs[i + 1];
                let c_range = u_csc_col_ptrs[j]..u_csc_col_ptrs[j + 1];
                let sum = pattern_dot(
                    &self.l_col_indices[l_range.clone()],
                    &self.l_values[l_range],
                    &u_csc_row_indices[c_range.clone()],
                    &u_csc_values[c_range],
                    i,
                );
                a.get(i, j) - sum
            })
            .collect();

        self.l_values = new_l;
        self.u_values = new_u;
    }

    /// Chow's threshold-based pattern augmentation.
    ///
    /// Computes the residual `R = A - L U` at the positions of the product
    /// pattern `L·U` outside the current pattern; per row, the up to
    /// `max_candidates_per_row` largest entries above
    /// `drop_tolerance * ||A(i,:)||` are added as zero-valued nonzeros (L
    /// part for `j < i`, U part for `j > i`). Off-diagonal pattern entries
    /// that fall below the same threshold and are not present in the pattern
    /// of A are dropped. The diagonal is never touched.
    fn augment_pattern(&mut self, a: &CsrMatrix<T>, row_norms: &[T::Real], config: &ParIlutConfig) {
        let n = a.num_rows;
        let tol_abs = T::Real::from_f64(1e-30).unwrap();
        let drop_rel = T::Real::from_f64(config.drop_tolerance).unwrap();
        let max_cand = config.max_candidates_per_row;

        let rows: PatternRows<T> = (0..n)
            .into_par_iter()
            .map(|i| {
                let tol = drop_rel * row_norms[i].max(tol_abs);
                let a_range = a.row_ptrs[i]..a.row_ptrs[i + 1];
                let a_cols = &a.col_indices[a_range];

                // Accumulate row i of the product L·U (unit-diagonal L).
                let mut prod: Vec<(u32, T)> = Vec::new();
                let push_scaled_u_row = |k: usize, scale: T, prod: &mut Vec<(u32, T)>| {
                    for idx in self.u_row_ptrs[k]..self.u_row_ptrs[k + 1] {
                        prod.push((self.u_col_indices[idx], scale * self.u_values[idx]));
                    }
                };
                for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                    push_scaled_u_row(
                        self.l_col_indices[idx] as usize,
                        self.l_values[idx],
                        &mut prod,
                    );
                }
                push_scaled_u_row(i, T::one(), &mut prod);
                prod.sort_unstable_by_key(|&(j, _)| j);

                let l_range = self.l_row_ptrs[i]..self.l_row_ptrs[i + 1];
                let u_range = self.u_row_ptrs[i]..self.u_row_ptrs[i + 1];
                let in_pattern = |j: u32| {
                    self.l_col_indices[l_range.clone()]
                        .binary_search(&j)
                        .is_ok()
                        || self.u_col_indices[u_range.clone()]
                            .binary_search(&j)
                            .is_ok()
                };

                // Residual candidates outside the current pattern.
                let mut candidates: Vec<(u32, T::Real)> = Vec::new();
                let mut p = 0;
                while p < prod.len() {
                    let j = prod[p].0;
                    let mut acc = prod[p].1;
                    p += 1;
                    while p < prod.len() && prod[p].0 == j {
                        acc += prod[p].1;
                        p += 1;
                    }
                    if in_pattern(j) {
                        continue;
                    }
                    let residual = a.get(i, j as usize) - acc;
                    let magnitude = residual.norm();
                    if magnitude > tol {
                        candidates.push((j, magnitude));
                    }
                }
                candidates
                    .sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates.truncate(max_cand);

                // Keep strong entries; A-pattern entries and the diagonal are
                // never dropped.
                let in_a = |j: u32| a_cols.binary_search(&(j as usize)).is_ok();
                let mut new_l: Vec<(u32, T)> = self.l_col_indices[l_range.clone()]
                    .iter()
                    .zip(&self.l_values[l_range])
                    .filter(|&(j, v)| v.norm() >= tol || in_a(*j))
                    .map(|(&j, &v)| (j, v))
                    .collect();
                let mut new_u: Vec<(u32, T)> = self.u_col_indices[u_range.clone()]
                    .iter()
                    .zip(&self.u_values[u_range])
                    .filter(|&(j, v)| *j as usize == i || v.norm() >= tol || in_a(*j))
                    .map(|(&j, &v)| (j, v))
                    .collect();
                for &(j, _) in &candidates {
                    if (j as usize) < i {
                        new_l.push((j, T::zero()));
                    } else {
                        new_u.push((j, T::zero()));
                    }
                }
                new_l.sort_unstable_by_key(|&(j, _)| j);
                new_u.sort_unstable_by_key(|&(j, _)| j);
                (new_l, new_u)
            })
            .collect();

        // Flatten per-row results back into CSR.
        self.l_values.clear();
        self.l_col_indices.clear();
        self.u_values.clear();
        self.u_col_indices.clear();
        self.l_row_ptrs.clear();
        self.u_row_ptrs.clear();
        self.l_row_ptrs.push(0);
        self.u_row_ptrs.push(0);
        for (l_row, u_row) in rows {
            for (j, v) in l_row {
                self.l_col_indices.push(j);
                self.l_values.push(v);
            }
            for (j, v) in u_row {
                self.u_col_indices.push(j);
                self.u_values.push(v);
            }
            self.l_row_ptrs.push(self.l_values.len());
            self.u_row_ptrs.push(self.u_values.len());
        }
        self.rebuild_entry_rows(n);
    }

    /// Frobenius norm `||A - L U||_F` over the full product pattern.
    fn residual_norm(&self, a: &CsrMatrix<T>) -> f64 {
        let n = a.num_rows;
        let mut ws = vec![T::zero(); n];
        let mut touched: Vec<usize> = Vec::new();
        let mut residual_scale = T::Real::zero();
        let mut residual_sum_squares = T::Real::one();

        for i in 0..n {
            debug_assert!(touched.is_empty());
            // Row i of L·U (unit-diagonal L), accumulated densely.
            for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                let k = self.l_col_indices[idx] as usize;
                let scale = self.l_values[idx];
                for uidx in self.u_row_ptrs[k]..self.u_row_ptrs[k + 1] {
                    let j = self.u_col_indices[uidx] as usize;
                    if ws[j] == T::zero() {
                        touched.push(j);
                    }
                    ws[j] += scale * self.u_values[uidx];
                }
            }
            for uidx in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                let j = self.u_col_indices[uidx] as usize;
                if ws[j] == T::zero() {
                    touched.push(j);
                }
                ws[j] += self.u_values[uidx];
            }
            // Subtract A's contribution; A columns not touched by the product
            // join the residual set.
            for idx in a.row_ptrs[i]..a.row_ptrs[i + 1] {
                let j = a.col_indices[idx];
                if ws[j] == T::zero() {
                    touched.push(j);
                }
                ws[j] = a.values[idx] - ws[j];
            }
            for &j in &touched {
                update_scaled_norm(&mut residual_scale, &mut residual_sum_squares, ws[j].norm());
                ws[j] = T::zero();
            }
            touched.clear();
        }

        finish_scaled_norm(residual_scale, residual_sum_squares)
            .to_f64()
            .unwrap_or(f64::INFINITY)
    }

    /// Convert to the canonical factor layout (off-diagonal U + inverse
    /// diagonal) used by the sequential triangular solve.
    fn to_ilu0(&self, n: usize, row_norms: &[T::Real]) -> Ilu0Factors<T> {
        let mut u_diag_inv = Vec::with_capacity(n);
        let mut u_diag = Vec::with_capacity(n);
        let mut u_values = Vec::with_capacity(self.u_values.len());
        let mut u_col_indices = Vec::with_capacity(self.u_col_indices.len());
        let mut u_row_ptrs = Vec::with_capacity(n + 1);
        u_row_ptrs.push(0);
        for (i, &rn) in row_norms[..n].iter().enumerate() {
            let range = self.u_row_ptrs[i]..self.u_row_ptrs[i + 1];
            // Diagonal is the first entry of the row (columns sorted).
            let u_ii = self.u_values[range.start];
            u_diag.push(u_ii);
            u_diag_inv.push(invert_pivot(u_ii, rn));
            for idx in (range.start + 1)..range.end {
                u_values.push(self.u_values[idx]);
                u_col_indices.push(self.u_col_indices[idx]);
            }
            u_row_ptrs.push(u_values.len());
        }

        Ilu0Factors {
            l_values: self.l_values.clone(),
            l_col_indices: self.l_col_indices.clone(),
            l_row_ptrs: self.l_row_ptrs.clone(),
            u_values,
            u_col_indices,
            u_row_ptrs,
            u_diag,
            u_diag_inv,
            n,
            diagnostics: self.diagnostics,
        }
    }
}

/// ParILUT: parallel ILU with threshold via fixed-point sweeps
/// (Chow/Anzt/Dongarra style).
///
/// The factorization starts from the canonical ILU(0) factors and refines
/// them with parallel Jacobi sweeps of the ILU recurrence over the nonzero
/// pattern. With [`ParIlutConfig::add_candidates`] enabled, the pattern is
/// augmented between sweeps with the largest residual positions of the
/// product `L·U` (and pruned of weak entries), which recovers the most
/// valuable fill-in of the exact factorization — the fixed-point analogue
/// of ILUT's dropping rule.
///
/// The factor values are produced entirely in parallel (rayon); the
/// triangular solve at apply time reuses the canonical sequential
/// substitution kernel, matching ParILUT on shared-memory CPUs.
#[derive(Debug, Clone)]
pub struct ParIlutPreconditioner<T: ComplexField> {
    /// Final factors in the canonical layout (unit-diagonal L, off-diagonal
    /// U, inverse U diagonal).
    factors: Ilu0Factors<T>,
    /// `||A - L U||_F` before the first sweep and after every sweep; only
    /// populated when [`ParIlutConfig::track_residual`] is set.
    residual_history: Vec<f64>,
    /// Number of nonzeros in the final (L, U) patterns.
    l_nnz: usize,
    u_nnz: usize,
}

impl<T: ComplexField + Send + Sync> ParIlutPreconditioner<T> {
    /// Factorize with the default configuration (5 sweeps, candidate
    /// augmentation enabled).
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Self {
        Self::from_csr_with_config(matrix, &ParIlutConfig::default())
    }

    /// Fallible ParILUT construction with the default configuration.
    pub fn try_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        Self::try_from_csr_with_config(matrix, &ParIlutConfig::default())
    }

    /// Factorize with an explicit configuration.
    pub fn from_csr_with_config(matrix: &CsrMatrix<T>, config: &ParIlutConfig) -> Self {
        assert_eq!(
            matrix.num_rows, matrix.num_cols,
            "ParILUT requires a square matrix"
        );
        let n = matrix.num_rows;

        // Row norms of the original matrix, shared by the pivot guards and
        // the relative drop threshold (same construction as `ilu.rs`).
        let row_norms: Vec<T::Real> = (0..n)
            .map(|i| {
                let mut scale = T::Real::zero();
                let mut sum_squares = T::Real::one();
                for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                    update_scaled_norm(&mut scale, &mut sum_squares, matrix.values[idx].norm());
                }
                finish_scaled_norm(scale, sum_squares)
            })
            .collect();

        let ilu0 = ilu0_factorize(matrix);
        let mut lu = ParLuFactors::from_ilu0(&ilu0);

        let mut residual_history = Vec::new();
        if config.track_residual {
            residual_history.push(lu.residual_norm(matrix));
        }

        for _ in 0..config.sweeps {
            let (col_ptrs, row_indices, values) = lu.build_u_csc(n);
            lu.sweep(matrix, &col_ptrs, &row_indices, &values, &row_norms);
            if config.add_candidates {
                lu.augment_pattern(matrix, &row_norms, config);
            }
            if config.track_residual {
                residual_history.push(lu.residual_norm(matrix));
            }
        }

        let l_nnz = lu.l_values.len();
        // U nonzeros excluding the diagonal, matching Ilu0Factors' convention.
        let u_nnz = lu.u_values.len() - n;
        let factors = lu.to_ilu0(n, &row_norms);

        Self {
            factors,
            residual_history,
            l_nnz,
            u_nnz,
        }
    }

    /// Fallible ParILUT construction with setup and canonical ILU diagnostics.
    pub fn try_from_csr_with_config(
        matrix: &CsrMatrix<T>,
        config: &ParIlutConfig,
    ) -> Result<Self, SolverError> {
        matrix.validate_canonical_structure()?;
        if matrix.num_rows != matrix.num_cols {
            return Err(SolverError::OperationFailed {
                operation: "ParILUT setup",
                reason: format!(
                    "matrix must be square, got {}x{}",
                    matrix.num_rows, matrix.num_cols
                ),
            });
        }
        if !config.drop_tolerance.is_finite() || config.drop_tolerance < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "ParILUT setup",
                reason: "drop_tolerance must be finite and non-negative".to_string(),
            });
        }
        if matrix.values.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "ParILUT setup",
                reason: "matrix contains NaN or infinity".to_string(),
            });
        }
        CsrMatrix::try_from_raw_parts(
            matrix.num_rows,
            matrix.num_cols,
            matrix.row_ptrs.clone(),
            matrix.col_indices.clone(),
            matrix.values.clone(),
        )?;
        let preconditioner = Self::from_csr_with_config(matrix, config);
        if preconditioner.factors.diagnostics.non_finite_entries != 0 {
            return Err(SolverError::OperationFailed {
                operation: "ParILUT setup",
                reason: format!(
                    "{} non-finite matrix entries",
                    preconditioner.factors.diagnostics.non_finite_entries
                ),
            });
        }
        if preconditioner.factors.diagnostics.skipped_pivots != 0 {
            return Err(SolverError::OperationFailed {
                operation: "ParILUT setup",
                reason: format!(
                    "{} pivots were below the configured relative threshold",
                    preconditioner.factors.diagnostics.skipped_pivots
                ),
            });
        }
        if preconditioner
            .residual_history
            .iter()
            .any(|residual| !residual.is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "ParILUT setup",
                reason: "factor residual history contains NaN or infinity".into(),
            });
        }
        Ok(preconditioner)
    }

    /// `||A - L U||_F` before the first sweep and after every sweep
    /// (length `sweeps + 1`); empty unless
    /// [`ParIlutConfig::track_residual`] was set.
    pub fn residual_history(&self) -> &[f64] {
        &self.residual_history
    }

    /// Number of nonzeros in the final L (strictly lower) and U
    /// (off-diagonal) patterns.
    pub fn factor_nnz(&self) -> (usize, usize) {
        (self.l_nnz, self.u_nnz)
    }

    /// Setup diagnostics retained by the canonical ILU(0) initialization.
    pub fn diagnostics(&self) -> &IluDiagnostics<T::Real> {
        &self.factors.diagnostics
    }

    /// Serialize the final ParILUT factors and setup metadata into a portable
    /// checkpoint. The nested ILU payload preserves scalar-family tags,
    /// factor structure, finite-value checks, and diagnostics.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        if self.l_nnz != self.factors.l_values.len()
            || self.u_nnz != self.factors.u_values.len()
            || self.residual_history.iter().any(|value| !value.is_finite())
        {
            return Err(parilut_checkpoint_error(
                "ParILUT metadata is inconsistent or non-finite",
            ));
        }
        let ilu = IluPreconditioner::from_factors(self.factors.clone(), None)?;
        let payload = ilu.to_checkpoint_bytes()?;

        let mut output = Vec::with_capacity(32 + payload.len() + self.residual_history.len() * 8);
        output.extend_from_slice(PARILUT_CHECKPOINT_MAGIC);
        output.extend_from_slice(&PARILUT_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(0); // reserved flags
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(0); // reserved byte
        parilut_checkpoint_push_u64(&mut output, payload.len())?;
        output.extend_from_slice(&payload);
        parilut_checkpoint_push_u64(&mut output, self.residual_history.len())?;
        for &value in &self.residual_history {
            output.extend_from_slice(&value.to_le_bytes());
        }
        parilut_checkpoint_push_u64(&mut output, self.l_nnz)?;
        parilut_checkpoint_push_u64(&mut output, self.u_nnz)?;
        Ok(output)
    }

    /// Restore ParILUT factors and setup metadata from a portable checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = ParIlutCheckpointReader::new(bytes);
        if reader.take(PARILUT_CHECKPOINT_MAGIC.len())? != PARILUT_CHECKPOINT_MAGIC {
            return Err(parilut_checkpoint_error("invalid checkpoint magic"));
        }
        if reader.read_u16()? != PARILUT_CHECKPOINT_FORMAT_VERSION {
            return Err(parilut_checkpoint_error("unsupported checkpoint version"));
        }
        if reader.read_u8()? != 0 {
            return Err(parilut_checkpoint_error("unsupported checkpoint flags"));
        }
        if reader.read_u8()? != std::mem::size_of::<T>() as u8
            || reader.read_u8()? != std::mem::size_of::<T::Real>() as u8
        {
            return Err(parilut_checkpoint_error(
                "checkpoint scalar family mismatch",
            ));
        }
        if reader.read_u8()? != 0 {
            return Err(parilut_checkpoint_error(
                "non-zero reserved checkpoint byte",
            ));
        }

        let payload_length = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("ILU payload length does not fit usize"))?;
        let payload = reader.take(payload_length)?;
        let ilu = IluPreconditioner::<T>::from_checkpoint_bytes(payload)?;
        let (factors, permutation) = ilu.into_factors();
        if permutation.is_some() {
            return Err(parilut_checkpoint_error(
                "ParILUT checkpoint cannot contain a permutation",
            ));
        }

        let residual_length = reader.read_len(8)?;
        let mut residual_history = Vec::with_capacity(residual_length);
        for _ in 0..residual_length {
            residual_history.push(reader.read_f64()?);
        }
        let l_nnz = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("L nnz does not fit usize"))?;
        let u_nnz = usize::try_from(reader.read_u64()?)
            .map_err(|_| parilut_checkpoint_error("U nnz does not fit usize"))?;
        if !reader.is_at_end() {
            return Err(parilut_checkpoint_error("trailing bytes"));
        }
        if l_nnz != factors.l_values.len() || u_nnz != factors.u_values.len() {
            return Err(parilut_checkpoint_error(
                "checkpoint nnz metadata does not match factors",
            ));
        }

        Ok(Self {
            factors,
            residual_history,
            l_nnz,
            u_nnz,
        })
    }
}

impl<T: ComplexField + Send + Sync> Preconditioner<T> for ParIlutPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = r.to_vec();
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(r.len(), self.factors.n, "ParILUT RHS length mismatch");
        assert_eq!(y.len(), self.factors.n, "ParILUT output length mismatch");
        y.copy_from_slice(r);
        self.factors.solve_in_place(y);
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        let mut y = r.to_vec();
        self.apply_transpose_into(r, &mut y);
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(
            r.len(),
            self.factors.n,
            "ParILUT transpose RHS length mismatch"
        );
        assert_eq!(
            y.len(),
            self.factors.n,
            "ParILUT transpose output length mismatch"
        );
        y.copy_from_slice(r);
        self.factors.solve_transpose_in_place(y);
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut y = r.to_vec();
        self.apply_hermitian_into(r, &mut y);
        y
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(
            r.len(),
            self.factors.n,
            "ParILUT Hermitian RHS length mismatch"
        );
        assert_eq!(
            y.len(),
            self.factors.n,
            "ParILUT Hermitian output length mismatch"
        );
        y.copy_from_slice(r);
        self.factors.solve_hermitian_in_place(y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};

    fn create_test_matrix() -> CsrMatrix<Complex64> {
        let n = 10;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }

        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn checked_parilut_accepts_a_well_pivoted_matrix() {
        let matrix = create_test_matrix();
        assert!(ParIlutPreconditioner::try_from_csr(&matrix).is_ok());
    }

    #[test]
    fn checked_parilut_norms_do_not_overflow_on_large_finite_diagonal() {
        let matrix =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![f64::MAX, f64::MAX]);
        let config = ParIlutConfig {
            sweeps: 2,
            add_candidates: false,
            track_residual: true,
            ..ParIlutConfig::default()
        };
        let preconditioner = ParIlutPreconditioner::try_from_csr_with_config(&matrix, &config)
            .expect("large finite diagonal must remain representable");
        assert_eq!(preconditioner.diagnostics().non_finite_entries, 0);
        assert_eq!(preconditioner.diagnostics().skipped_pivots, 0);
        assert_eq!(preconditioner.residual_history().len(), config.sweeps + 1);
        assert!(
            preconditioner
                .residual_history()
                .iter()
                .all(|residual| residual.is_finite() && *residual == 0.0)
        );
    }

    #[test]
    fn checked_parilut_rejects_singular_and_non_finite_setup() {
        let dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(3, 3, Complex64::new(0.0, 0.0));
        let singular = CsrMatrix::from_dense(&dense, 1e-15);
        let error = match ParIlutPreconditioner::try_from_csr(&singular) {
            Ok(_) => panic!("singular ParILUT setup must report pivot diagnostics"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("pivots"));

        let mut bad =
            CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![Complex64::new(1.0, 0.0)]);
        bad.values[0] = Complex64::new(f64::NAN, 0.0);
        let error = match ParIlutPreconditioner::try_from_csr(&bad) {
            Ok(_) => panic!("non-finite ParILUT input must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("NaN or infinity"));
    }

    #[test]
    fn checked_parallel_ilu_variants_reject_singular_setup() {
        let dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(3, 3, Complex64::new(0.0, 0.0));
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let colored = IluColoringPreconditioner::try_from_csr(&matrix);
        assert!(
            colored
                .expect_err("colored ILU must reject singular pivots")
                .to_string()
                .contains("pivots")
        );
        let fixed = IluFixedPointPreconditioner::try_from_csr(&matrix, 2);
        assert!(
            fixed
                .expect_err("fixed-point ILU must reject singular pivots")
                .to_string()
                .contains("pivots")
        );

        let matrix = create_test_matrix();
        assert!(IluColoringPreconditioner::try_from_csr(&matrix).is_ok());
        assert!(IluFixedPointPreconditioner::try_from_csr(&matrix, 2).is_ok());
    }

    #[test]
    fn parallel_ilu_variants_expose_setup_diagnostics() {
        let matrix = CsrMatrix::<f64>::identity(3);

        let coloring = IluColoringPreconditioner::try_from_csr(&matrix)
            .expect("well-pivoted coloring ILU should construct");
        assert_eq!(coloring.diagnostics().non_finite_entries, 0);
        assert!(coloring.diagnostics().min_abs_pivot.is_some());

        let fixed = IluFixedPointPreconditioner::try_from_csr(&matrix, 2)
            .expect("well-pivoted fixed-point ILU should construct");
        assert_eq!(fixed.diagnostics().non_finite_entries, 0);
        assert!(fixed.diagnostics().min_abs_pivot.is_some());

        let parilut = ParIlutPreconditioner::try_from_csr(&matrix)
            .expect("well-pivoted ParILUT should construct");
        assert_eq!(parilut.diagnostics().non_finite_entries, 0);
        assert!(parilut.diagnostics().min_abs_pivot.is_some());
    }

    #[test]
    fn test_ilu_coloring_basic() {
        let matrix = create_test_matrix();
        let precond = IluColoringPreconditioner::from_csr(&matrix);

        let r = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let result = precond.apply(&r);

        // Should produce a reasonable result
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|x| x.norm() < 100.0));
    }

    #[test]
    fn test_ilu_fixedpoint_basic() {
        let matrix = create_test_matrix();
        let precond = IluFixedPointPreconditioner::from_csr(&matrix, 3);

        let r = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let result = precond.apply(&r);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|x| x.norm() < 100.0));
    }

    #[test]
    fn fixed_point_monitor_norms_remain_finite_for_large_vectors() {
        let large = f64::MAX / 2.0;
        let values = [large, large];
        let zeros = [0.0, 0.0];

        let norm = IluFixedPointPreconditioner::<f64>::vector_norm(&values);
        let serial_diff = IluFixedPointPreconditioner::<f64>::vector_norm_diff(&values, &zeros);
        let parallel_diff =
            IluFixedPointPreconditioner::<f64>::vector_norm_diff_parallel(&values, &zeros);

        assert!(norm.is_finite() && norm > large);
        assert_eq!(serial_diff, norm);
        assert!((parallel_diff - norm).abs() <= 2.0 * f64::EPSILON * norm);
    }

    #[test]
    fn test_ilu_coloring_with_gmres() {
        let matrix = create_test_matrix();
        let precond = IluColoringPreconditioner::from_csr(&matrix);

        let b = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_preconditioned(&matrix, &precond, &b, &config);
        assert!(sol.converged, "GMRES with ILU-coloring should converge");
    }

    #[test]
    fn test_ilu_fixedpoint_with_gmres() {
        let matrix = create_test_matrix();
        let precond = IluFixedPointPreconditioner::from_csr(&matrix, 3);

        let b = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_preconditioned(&matrix, &precond, &b, &config);
        assert!(sol.converged, "GMRES with ILU-fixedpoint should converge");
    }

    #[test]
    fn test_level_stats() {
        let matrix = create_test_matrix();
        let precond = IluColoringPreconditioner::from_csr(&matrix);

        let (fwd, bwd, avg_fwd, avg_bwd) = precond.level_stats();

        // Tridiagonal matrix should have limited parallelism
        assert!(fwd > 0);
        assert!(bwd > 0);
        assert!(avg_fwd > 0.0);
        assert!(avg_bwd > 0.0);
    }

    #[test]
    fn test_ilu_fixedpoint_converges_to_exact_solve() {
        // Regression test for issue #14: the fixed-point iteration must
        // approximate the LU solve (forward + backward substitution), not
        // the L+U Jacobi sweep.
        //
        // With enough iterations the fixed-point preconditioner should
        // converge to the same result as the exact coloring preconditioner.
        let matrix = create_test_matrix();
        let exact = IluColoringPreconditioner::from_csr(&matrix);
        let approx = IluFixedPointPreconditioner::from_csr(&matrix, 20);

        let r = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let x_exact = exact.apply(&r);
        let x_approx = approx.apply(&r);

        for i in 0..10 {
            let diff = (x_exact[i] - x_approx[i]).norm();
            assert!(
                diff < 1e-10,
                "Fixed-point did not converge to exact LU solve at {}: diff={:.2e}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_ilu_fixedpoint_early_exit_on_easy_system() {
        // Strongly diagonally dominant matrix: the Jacobi sweeps converge
        // almost immediately, so the convergence check must stop the loop
        // long before the maximum sweep count.
        let n = 12;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-0.001, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-0.001, 0.0);
            }
        }
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluFixedPointPreconditioner::from_csr_with_tolerance(&matrix, 10, 1e-3);

        let r = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let (sweeps_before, fallbacks_before) = precond.stats();
        let result = precond.apply(&r);
        let (sweeps_after, fallbacks_after) = precond.stats();

        let sweeps_used = sweeps_after - sweeps_before;
        eprintln!("ILU fixed-point early exit: {sweeps_used} sweeps (max 20)");
        assert!(
            sweeps_used <= 4,
            "easy system should exit early, used {sweeps_used} sweeps of 20 max"
        );
        assert_eq!(
            fallbacks_after, fallbacks_before,
            "easy system must not trigger the fallback"
        );

        // Result should still approximate the exact triangular solves.
        let exact = IluColoringPreconditioner::from_csr(&matrix);
        let x_exact = exact.apply(&r);
        for i in 0..n {
            assert!(
                (result[i] - x_exact[i]).norm() < 1e-2,
                "early-exit result too far from exact solve at {i}"
            );
        }
    }

    #[test]
    fn test_ilu_fixedpoint_fallback_on_divergence() {
        // Lower bidiagonal with |subdiagonal| = 4: the Jacobi iteration on
        // L y = r amplifies the fixpoint residual by ~4x per sweep (nilpotent
        // iteration matrix, but with huge transient growth). The convergence
        // check must detect the growth and fall back to exact sequential
        // substitution instead of returning garbage.
        let n = 8;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(1.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-4.0, 0.0);
            }
        }
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluFixedPointPreconditioner::from_csr(&matrix, 10);

        let r = (0..n)
            .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.0))
            .collect::<Vec<_>>();
        let (_, fallbacks_before) = precond.stats();
        let result = precond.apply(&r);
        let (_, fallbacks_after) = precond.stats();

        assert!(
            fallbacks_after > fallbacks_before,
            "diverging sweeps should trigger the exact-substitution fallback"
        );

        // Fallback must produce the exact LU solve (ILU(0) is exact on a
        // bidiagonal matrix).
        let exact = IluColoringPreconditioner::from_csr(&matrix);
        let x_exact = exact.apply(&r);
        let scale = x_exact
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        for i in 0..n {
            let diff = (result[i] - x_exact[i]).norm();
            assert!(
                diff < 1e-12 * scale,
                "fallback result deviates from exact solve at {i}: diff={diff:.3e}"
            );
        }
        assert!(
            result.iter().all(|v| v.norm().is_finite()),
            "fallback result must be finite"
        );
    }

    #[test]
    fn test_ilu_fixedpoint_parallel_path_checks() {
        // n >= 246 exercises the parallel apply path. Easy system: early
        // exit. Hostile system (subdiagonal -1.5): divergence -> fallback.
        let n = 300;

        let mut easy =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        let mut hostile =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            easy[[i, i]] = Complex64::new(4.0, 0.0);
            hostile[[i, i]] = Complex64::new(1.0, 0.0);
            if i > 0 {
                easy[[i, i - 1]] = Complex64::new(-0.001, 0.0);
                hostile[[i, i - 1]] = Complex64::new(-1.5, 0.0);
            }
            if i < n - 1 {
                easy[[i, i + 1]] = Complex64::new(-0.001, 0.0);
            }
        }
        let easy = CsrMatrix::from_dense(&easy, 1e-15);
        let hostile = CsrMatrix::from_dense(&hostile, 1e-15);

        let r = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), 0.0))
            .collect::<Vec<_>>();

        // Easy: early exit, no fallback.
        let precond = IluFixedPointPreconditioner::from_csr_with_tolerance(&easy, 12, 1e-3);
        let (sweeps_before, fallbacks_before) = precond.stats();
        let result = precond.apply(&r);
        let (sweeps_after, fallbacks_after) = precond.stats();
        assert!(
            sweeps_after - sweeps_before <= 6,
            "parallel easy system should exit early, used {} sweeps",
            sweeps_after - sweeps_before
        );
        assert_eq!(fallbacks_after, fallbacks_before);
        assert!(result.iter().all(|v| v.norm().is_finite()));

        // Hostile: fallback to exact substitution.
        let precond = IluFixedPointPreconditioner::from_csr(&hostile, 12);
        let (_, fallbacks_before) = precond.stats();
        let result = precond.apply(&r);
        let (_, fallbacks_after) = precond.stats();
        assert!(
            fallbacks_after > fallbacks_before,
            "parallel hostile system should trigger the fallback"
        );
        let exact = IluColoringPreconditioner::from_csr(&hostile);
        let x_exact = exact.apply(&r);
        let scale = x_exact
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        for i in 0..n {
            let diff = (result[i] - x_exact[i]).norm();
            assert!(
                diff < 1e-9 * scale,
                "parallel fallback deviates from exact solve at {i}: diff={diff:.3e}"
            );
        }
    }

    /// 2D 5-point Poisson matrix on an `nx × nx` grid (row-major numbering).
    fn poisson_2d(nx: usize) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut rows = vec![Vec::new(); n];
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                rows[i].push((i, 4.0));
                if ix > 0 {
                    rows[i].push((i - 1, -1.0));
                }
                if ix + 1 < nx {
                    rows[i].push((i + 1, -1.0));
                }
                if iy > 0 {
                    rows[i].push((i - nx, -1.0));
                }
                if iy + 1 < nx {
                    rows[i].push((i + nx, -1.0));
                }
            }
        }
        CsrMatrix::from_triplets(
            n,
            n,
            rows.iter()
                .enumerate()
                .flat_map(|(i, row)| row.iter().map(move |&(j, v)| (i, j, v)))
                .collect(),
        )
    }

    /// Nonsymmetric 2D convection-diffusion matrix: 5-point Laplacian plus
    /// central-difference convection of strength `c` in x.
    fn convection_diffusion_2d(nx: usize, c: f64) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut rows = vec![Vec::new(); n];
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                rows[i].push((i, 4.0));
                if ix > 0 {
                    rows[i].push((i - 1, -1.0 - c));
                }
                if ix + 1 < nx {
                    rows[i].push((i + 1, -1.0 + c));
                }
                if iy > 0 {
                    rows[i].push((i - nx, -1.0));
                }
                if iy + 1 < nx {
                    rows[i].push((i + nx, -1.0));
                }
            }
        }
        CsrMatrix::from_triplets(
            n,
            n,
            rows.iter()
                .enumerate()
                .flat_map(|(i, row)| row.iter().map(move |&(j, v)| (i, j, v)))
                .collect(),
        )
    }

    #[test]
    fn test_par_ilut_residual_decreases_with_sweeps() {
        let matrix = poisson_2d(16);

        // Fixed pattern = pattern of A: ILU(0) is already the exact solution
        // of the ILU recurrence restricted to that pattern, so the sweeps
        // must leave the residual unchanged (fixed-point property).
        let fixed = ParIlutPreconditioner::from_csr_with_config(
            &matrix,
            &ParIlutConfig {
                sweeps: 5,
                add_candidates: false,
                track_residual: true,
                ..Default::default()
            },
        );
        let history = fixed.residual_history();
        assert_eq!(history.len(), 6, "history must hold initial + 5 sweeps");
        for (k, &r) in history.iter().enumerate().skip(1) {
            assert!(
                (r - history[0]).abs() < 1e-10 * history[0].max(1.0),
                "fixed-pattern sweep {k} changed the residual: {history:?}"
            );
        }

        // With candidate augmentation the pattern grows beyond A and the
        // sweeps must drive ||A - LU||_F below the ILU(0) baseline.
        let augmented = ParIlutPreconditioner::from_csr_with_config(
            &matrix,
            &ParIlutConfig {
                sweeps: 5,
                add_candidates: true,
                max_candidates_per_row: 3,
                drop_tolerance: 1e-3,
                track_residual: true,
            },
        );
        let history = augmented.residual_history();
        eprintln!(
            "ParILUT residuals: ILU(0) baseline {:.4e}, augmented {:?}",
            fixed.residual_history()[0],
            history
        );
        assert!(
            *history.last().unwrap() < history[0],
            "residual must decrease with sweeps: {history:?}"
        );
        assert!(
            *history.last().unwrap() < 0.8 * fixed.residual_history()[0],
            "sweeps should cut the residual at least 20% below the ILU(0) baseline: {history:?}"
        );
    }

    #[test]
    fn test_par_ilut_candidate_augmentation() {
        let matrix = poisson_2d(16);

        let base = ParIlutPreconditioner::from_csr_with_config(
            &matrix,
            &ParIlutConfig {
                sweeps: 0,
                ..Default::default()
            },
        );
        let config = ParIlutConfig {
            sweeps: 4,
            add_candidates: true,
            max_candidates_per_row: 3,
            drop_tolerance: 1e-3,
            track_residual: true,
        };
        let precond = ParIlutPreconditioner::from_csr_with_config(&matrix, &config);

        let (l0, u0) = base.factor_nnz();
        let (l1, u1) = precond.factor_nnz();
        assert!(
            l1 + u1 > l0 + u0,
            "candidate augmentation should grow the pattern: {l0}+{u0} -> {l1}+{u1}"
        );

        let history = precond.residual_history();
        eprintln!("ParILUT augmented residuals: {history:?}");
        assert!(
            *history.last().unwrap() < history[0],
            "augmented residual must decrease: {history:?}"
        );

        // Apply must remain sane (finite, right size).
        let r = vec![1.0_f64; matrix.num_rows];
        let y = precond.apply(&r);
        assert_eq!(y.len(), matrix.num_rows);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_par_ilut_beats_ilu0_gmres() {
        use crate::IluPreconditioner;

        // Nonsymmetric convection-dominated matrix: ILU(0) misses the fill
        // that ParILUT recovers via candidate augmentation.
        let matrix = convection_diffusion_2d(16, 1.5);
        let n = matrix.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() + 1.0).collect();

        let config = GmresConfig {
            max_iterations: 300,
            restart: 40,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            ..Default::default()
        };

        let ilu0 = IluPreconditioner::from_csr(&matrix);
        let sol_ilu0 = gmres_preconditioned(&matrix, &ilu0, &b, &config);

        let par_ilut = ParIlutPreconditioner::from_csr_with_config(
            &matrix,
            &ParIlutConfig {
                sweeps: 5,
                add_candidates: true,
                max_candidates_per_row: 3,
                drop_tolerance: 1e-3,
                track_residual: false,
            },
        );
        let sol_par = gmres_preconditioned(&matrix, &par_ilut, &b, &config);

        eprintln!(
            "GMRES iterations: ILU(0) = {}, ParILUT = {}",
            sol_ilu0.iterations, sol_par.iterations
        );
        assert!(
            sol_par.converged,
            "ParILUT-preconditioned GMRES must converge"
        );
        assert!(
            sol_par.iterations < sol_ilu0.iterations,
            "ParILUT ({}) should need fewer iterations than ILU(0) ({})",
            sol_par.iterations,
            sol_ilu0.iterations
        );

        // Both solutions must agree (same linear system).
        if sol_ilu0.converged {
            for i in 0..n {
                assert!(
                    (sol_par.x[i] - sol_ilu0.x[i]).abs() < 1e-5,
                    "solution mismatch at {i}"
                );
            }
        }
    }

    #[test]
    fn test_par_ilut_complex_smoke() {
        // Complex64 genericity: factorize, apply, and precondition GMRES.
        let matrix = create_test_matrix();
        let precond = ParIlutPreconditioner::from_csr_with_config(
            &matrix,
            &ParIlutConfig {
                sweeps: 3,
                track_residual: true,
                ..Default::default()
            },
        );

        let b = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let y = precond.apply(&b);
        assert_eq!(y.len(), 10);
        assert!(y.iter().all(|v| v.norm().is_finite()));

        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            ..Default::default()
        };
        let sol = gmres_preconditioned(&matrix, &precond, &b, &config);
        assert!(sol.converged, "GMRES with ParILUT should converge");
        assert_eq!(precond.residual_history().len(), 4);
    }

    #[test]
    fn par_ilut_checkpoint_round_trip_preserves_complex_action_and_metadata() {
        let matrix = create_test_matrix();
        let source = ParIlutPreconditioner::from_csr_with_config(
            &matrix,
            &ParIlutConfig {
                sweeps: 3,
                track_residual: true,
                ..Default::default()
            },
        );
        let bytes = source.to_checkpoint_bytes().expect("serialize ParILUT");
        let restored = ParIlutPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
            .expect("restore ParILUT");
        let rhs = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect::<Vec<_>>();
        let expected = source.apply(&rhs);
        let actual = restored.apply(&rhs);
        let max_error = expected
            .iter()
            .zip(&actual)
            .map(|(left, right)| (*left - *right).norm())
            .fold(0.0, f64::max);
        assert!(max_error < 1e-12, "checkpoint action error {max_error:e}");
        assert_eq!(source.factor_nnz(), restored.factor_nnz());
        assert_eq!(source.residual_history(), restored.residual_history());
        assert_eq!(source.diagnostics(), restored.diagnostics());
    }

    #[test]
    fn par_ilut_checkpoint_rejects_corruption_and_trailing_bytes() {
        let matrix = poisson_2d(4);
        let source = ParIlutPreconditioner::from_csr_with_config(
            &matrix,
            &ParIlutConfig {
                sweeps: 2,
                track_residual: true,
                ..Default::default()
            },
        );
        let bytes = source.to_checkpoint_bytes().expect("serialize ParILUT");
        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xff;
        assert!(ParIlutPreconditioner::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(ParIlutPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err());
    }

    #[test]
    fn colored_ilu_checkpoint_round_trip_preserves_levels_and_action() {
        let matrix = create_test_matrix();
        let source = IluColoringPreconditioner::try_from_csr(&matrix).expect("colored ILU");
        let bytes = source.to_checkpoint_bytes().expect("serialize colored ILU");
        let restored = IluColoringPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
            .expect("restore colored ILU");
        let rhs = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect::<Vec<_>>();
        let max_error = source
            .apply(&rhs)
            .iter()
            .zip(restored.apply(&rhs))
            .map(|(left, right)| (*left - right).norm())
            .fold(0.0, f64::max);
        assert!(max_error < 1e-12, "colored ILU action error {max_error:e}");
        assert_eq!(source.level_stats(), restored.level_stats());
        assert_eq!(source.diagnostics(), restored.diagnostics());
    }

    #[test]
    fn fixed_point_ilu_checkpoint_round_trip_preserves_policy_and_stats() {
        let matrix = create_test_matrix();
        let source = IluFixedPointPreconditioner::try_from_csr_with_tolerance(&matrix, 3, 1e-12)
            .expect("fixed-point ILU");
        let bytes = source
            .to_checkpoint_bytes()
            .expect("serialize fixed-point ILU");
        let restored = IluFixedPointPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
            .expect("restore fixed-point ILU");
        let rhs = (0..10)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let source_action = source.apply(&rhs);
        let source_stats = source.stats();
        let max_error = source_action
            .iter()
            .zip(restored.apply(&rhs))
            .map(|(left, right)| (*left - right).norm())
            .fold(0.0, f64::max);
        assert!(
            max_error < 1e-12,
            "fixed-point ILU action error {max_error:e}"
        );
        assert_eq!(restored.stats(), source_stats);
        assert_eq!(source.diagnostics(), restored.diagnostics());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(
            IluFixedPointPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err()
        );
    }
}
