//! Fixed-size block-Jacobi preconditioner over dense diagonal blocks.
//!
//! SIM-PERF-20261013.md section 2, gap 3: the speaker CUDA path solved with
//! point-Jacobi only, so device iteration counts could never compete
//! per-iteration with the CPU path. This module adds the next rung:
//! exact solves over fixed contiguous diagonal blocks (default 128,
//! PERF-REPORT-20261013.md step 4.3 prescription).
//!
//! Design: each diagonal block is factored on the host (LU with partial
//! pivoting) and its explicit inverse uploaded once as a block-diagonal
//! resident CSR matrix, so `apply` is a single device SpMV with no new
//! kernels — backend-neutral across CPU/CUDA/Metal. Blocks whose LU
//! factorisation fails degrade to the Jacobi diagonal for their rows
//! (same convention as the CPU `HMatrixBlockJacobi`), and `block_size =
//! 1` reproduces point Jacobi exactly, which is how callers keep
//! point-Jacobi as the small-system fallback without a second slot type.
//!
//! Per-frequency sweeps keep the block partition (hence the CSR
//! structure) fixed and push numeric-only value updates through
//! [`BlockJacobiBackend::try_update_dense_host`], mirroring how the
//! resident operator reuses its prebaked descriptor.

// Rust guideline compliant 2026-02-21

use crate::operator::PersistentDeviceCsr;
use crate::{
    Backend, DeviceBufferLike, DevicePreconditioner, DeviceScalar, HostCsrRef, LowerPrecision,
    PreconditionerFailureInjectionPoint,
};
use num_traits::Float;
use rayon::prelude::*;
use std::sync::Mutex;

/// Default fixed block size for the device block-Jacobi preconditioner.
///
/// 128x128 follows the PERF-REPORT step 4.3 prescription: large enough to
/// capture near-diagonal coupling, small enough that host-side explicit
/// inversion stays negligible next to assembly (one 128^3 LU per block).
pub const DEFAULT_BLOCK_SIZE: usize = 128;

/// Block-Jacobi preconditioner with explicit host-factored block inverses.
///
/// Owns a resident block-diagonal CSR matrix holding one explicit inverse
/// per diagonal block; `apply` is a single SpMV. Built from dense
/// row-major host data (the speaker BEM dense operator layout).
pub struct BlockJacobiBackend<B: Backend, T: DeviceScalar = f64> {
    block_inv: PersistentDeviceCsr<B, T>,
    n: usize,
    block_size: usize,
    num_blocks: usize,
    fallback_blocks: usize,
    workspace: Mutex<Option<B::Buffer<T>>>,
    runtime_diagnostics: crate::DevicePreconditionerDiagnostics,
}

/// Row range of one contiguous diagonal block plus its CSR value offset.
struct BlockLayout {
    row_begin: usize,
    len: usize,
    values_offset: usize,
}

impl BlockLayout {
    fn plan(n: usize, block_size: usize) -> Vec<BlockLayout> {
        let mut layouts = Vec::new();
        let mut offset = 0;
        let mut row = 0;
        while row < n {
            let len = block_size.min(n - row);
            layouts.push(BlockLayout {
                row_begin: row,
                len,
                values_offset: offset,
            });
            offset += len * len;
            row += len;
        }
        layouts
    }
}

/// Explicit inverse of one dense row-major block via LU with partial
/// pivoting, or `None` when the block is numerically singular (pivot
/// `<= epsilon`). Generic over [`DeviceScalar`] so the same code serves
/// f32/f64 and complex sweeps without a solvers dependency.
fn invert_dense_block<T: DeviceScalar>(block: &[T], len: usize) -> Option<Vec<T>> {
    debug_assert_eq!(block.len(), len * len);
    let eps = T::Real::epsilon();
    let mut lu = block.to_vec();
    let mut perm: Vec<usize> = (0..len).collect();
    for k in 0..len {
        let mut pivot_row = k;
        let mut pivot_mag = lu[k * len + k].norm_sqr().sqrt();
        for i in (k + 1)..len {
            let mag = lu[i * len + k].norm_sqr().sqrt();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_mag <= eps || !pivot_mag.is_finite() {
            return None;
        }
        if pivot_row != k {
            for j in 0..len {
                lu.swap(k * len + j, pivot_row * len + j);
            }
            perm.swap(k, pivot_row);
        }
        let pivot = lu[k * len + k];
        for i in (k + 1)..len {
            lu[i * len + k] = lu[i * len + k] / pivot;
            let factor = lu[i * len + k];
            for j in (k + 1)..len {
                lu[i * len + j] = lu[i * len + j] - factor * lu[k * len + j];
            }
        }
    }
    // Solve LU X = P I one column at a time: column j of P I is the
    // unit vector at the row i with perm[i] == j.
    let mut inverse = vec![T::zero(); len * len];
    let mut y = vec![T::zero(); len];
    for j in 0..len {
        for (i, slot) in y.iter_mut().enumerate() {
            *slot = if perm[i] == j { T::one() } else { T::zero() };
        }
        for i in 0..len {
            let mut acc = y[i];
            for k in 0..i {
                acc = acc - lu[i * len + k] * y[k];
            }
            y[i] = acc;
        }
        for i in (0..len).rev() {
            let mut acc = y[i];
            for k in (i + 1)..len {
                acc = acc - lu[i * len + k] * inverse[k * len + j];
            }
            inverse[i * len + j] = acc / lu[i * len + i];
        }
    }
    if inverse.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(inverse)
}

/// Extract the `len x len` diagonal block at `row_begin` from dense
/// row-major `n x n` data.
fn extract_block<T: Copy>(dense: &[T], n: usize, row_begin: usize, len: usize) -> Vec<T> {
    let mut block = Vec::with_capacity(len * len);
    for local_row in 0..len {
        let base = (row_begin + local_row) * n + row_begin;
        block.extend_from_slice(&dense[base..base + len]);
    }
    block
}

/// Inverted block values plus a singular-block flag.
///
/// Returns the explicit inverse in row-major order, or Jacobi-diagonal
/// fallback rows (identity rows where the diagonal itself is tiny, the
/// same floor convention as the CPU H-matrix block-Jacobi) with the
/// flag set when LU factorisation fails.
fn invert_block_values<T: DeviceScalar>(block: &[T], len: usize) -> (Vec<T>, usize) {
    debug_assert_eq!(block.len(), len * len);
    if let Some(inverse) = invert_dense_block(block, len) {
        return (inverse, 0);
    }
    let eps = T::Real::epsilon();
    let mut values = vec![T::zero(); len * len];
    for local in 0..len {
        let diag = block[local * len + local];
        values[local * len + local] = if diag.norm_sqr().sqrt() > eps {
            T::one() / diag
        } else {
            T::one()
        };
    }
    (values, 1)
}

/// Block-diagonal CSR values for `dense` (row-major `n x n`).
///
/// Returns the values in CSR row order plus the number of blocks that
/// degraded to the diagonal. Blocks invert in parallel; the per-block
/// results merge back serially in layout order, so no shared
/// mutable state crosses threads.
fn block_inverse_values<T: DeviceScalar>(
    dense: &[T],
    n: usize,
    layouts: &[BlockLayout],
) -> (Vec<T>, usize) {
    let per_block: Vec<(usize, Vec<T>, usize)> = layouts
        .par_iter()
        .map(|layout| {
            let block = extract_block(dense, n, layout.row_begin, layout.len);
            let (values, downgraded) = invert_block_values(&block, layout.len);
            (layout.values_offset, values, downgraded)
        })
        .collect();
    let total: usize = layouts.iter().map(|layout| layout.len * layout.len).sum();
    let mut values = vec![T::zero(); total];
    let mut fallback_blocks = 0;
    for (offset, block_values, downgraded) in &per_block {
        values[*offset..*offset + block_values.len()].copy_from_slice(block_values);
        fallback_blocks += *downgraded;
    }
    (values, fallback_blocks)
}

/// CSR structure (row pointers, column indices) for a block-diagonal
/// matrix over `layouts`. Structure depends only on `n`/`block_size`,
/// never on values, so sweeps reuse it across frequencies.
fn block_diagonal_structure(n: usize, layouts: &[BlockLayout]) -> (Vec<usize>, Vec<usize>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0);
    for layout in layouts {
        for _ in 0..layout.len {
            for k in 0..layout.len {
                col_idx.push(layout.row_begin + k);
            }
            row_ptr.push(col_idx.len());
        }
    }
    debug_assert_eq!(row_ptr.len(), n + 1);
    (row_ptr, col_idx)
}

impl<B: Backend, T: DeviceScalar> BlockJacobiBackend<B, T> {
    /// Build from dense row-major host data.
    ///
    /// `dense_rowmajor` holds the `n x n` system in row-major order and
    /// `block_size` names the fixed contiguous block width; the trailing
    /// block takes the remainder. `block_size = 1` reproduces point
    /// Jacobi exactly. Singular blocks degrade to the diagonal (see
    /// [`Self::fallback_blocks`]) instead of failing.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_from_dense_host`] reports a shape or
    /// finiteness error (wrong length, non-finite input, empty system,
    /// zero block size).
    pub fn from_dense_host(backend: &B, dense_rowmajor: &[T], n: usize, block_size: usize) -> Self {
        Self::try_from_dense_host(backend, dense_rowmajor, n, block_size)
            .expect("block-Jacobi construction failed")
    }

    /// Build from dense data, reporting shape errors.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` for an empty system, a zero block
    /// size, a length other than `n * n`, or non-finite input.
    pub fn try_from_dense_host(
        backend: &B,
        dense_rowmajor: &[T],
        n: usize,
        block_size: usize,
    ) -> crate::BackendResult<Self> {
        Self::check_dense_inputs(
            dense_rowmajor.len(),
            n,
            block_size,
            "block_jacobi_constructor",
        )?;
        if dense_rowmajor.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_constructor",
                reason: "dense values must be finite".into(),
            });
        }
        let layouts = BlockLayout::plan(n, block_size);
        let (values, fallback_blocks) = block_inverse_values(dense_rowmajor, n, &layouts);
        Self::upload(backend, n, block_size, &layouts, &values, fallback_blocks)
    }

    /// Build from higher-precision dense host data.
    ///
    /// Casts the extracted diagonal blocks on the host. This is the
    /// mixed-precision sweep path: only the `n x block_size`
    /// diagonal-block entries are cast, never a temporary full `n^2`
    /// buffer. Length, finiteness, and shape checks mirror
    /// [`Self::try_from_dense_host`].
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_from_dense_host_cast`] reports a shape
    /// or finiteness error.
    pub fn from_dense_host_cast<UpperT>(
        backend: &B,
        dense_rowmajor: &[UpperT],
        n: usize,
        block_size: usize,
    ) -> Self
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        Self::try_from_dense_host_cast(backend, dense_rowmajor, n, block_size)
            .expect("block-Jacobi cast construction failed")
    }

    /// Build from higher-precision data, reporting shape errors.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` for an empty system, a zero block
    /// size, a length other than `n * n`, or non-finite input.
    pub fn try_from_dense_host_cast<UpperT>(
        backend: &B,
        dense_rowmajor: &[UpperT],
        n: usize,
        block_size: usize,
    ) -> crate::BackendResult<Self>
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        Self::check_dense_inputs(
            dense_rowmajor.len(),
            n,
            block_size,
            "block_jacobi_constructor",
        )?;
        if dense_rowmajor.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_constructor",
                reason: "dense values must be finite".into(),
            });
        }
        let layouts = BlockLayout::plan(n, block_size);
        // Cast only the diagonal blocks the preconditioner consumes.
        let cast_layouts = layouts
            .par_iter()
            .map(|layout| {
                let mut cast = Vec::with_capacity(layout.len * layout.len);
                for local_row in 0..layout.len {
                    let base = (layout.row_begin + local_row) * n + layout.row_begin;
                    cast.extend(
                        dense_rowmajor[base..base + layout.len]
                            .iter()
                            .map(|value| value.cast_to_lower()),
                    );
                }
                cast
            })
            .collect::<Vec<_>>();
        let total: usize = layouts.iter().map(|layout| layout.len * layout.len).sum();
        let mut values = vec![T::zero(); total];
        let mut fallback_blocks = 0;
        for (layout, cast) in layouts.iter().zip(cast_layouts.iter()) {
            fallback_blocks += Self::invert_into(&mut values, layout, cast);
        }
        Self::upload(backend, n, block_size, &layouts, &values, fallback_blocks)
    }

    /// Numeric-only update for new dense values.
    ///
    /// The block partition (hence the resident CSR structure and
    /// prebaked descriptor) is retained; only values are recomputed
    /// and re-uploaded.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` for a length other than `n * n` or
    /// non-finite input.
    pub fn try_update_dense_host(
        &mut self,
        backend: &B,
        dense_rowmajor: &[T],
    ) -> crate::BackendResult<()> {
        if dense_rowmajor.len() != self.n * self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_update",
                reason: format!(
                    "expected {} dense values, got {}",
                    self.n * self.n,
                    dense_rowmajor.len()
                ),
            });
        }
        if dense_rowmajor.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_update",
                reason: "dense values must be finite".into(),
            });
        }
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let layouts = BlockLayout::plan(self.n, self.block_size);
        let (values, fallback_blocks) = block_inverse_values(dense_rowmajor, self.n, &layouts);
        self.fallback_blocks = fallback_blocks;
        self.block_inv.try_copy_values_from_host(backend, &values)
    }

    /// Mixed-precision numeric-only update for new dense values.
    ///
    /// Casts only the diagonal blocks, never a temporary full `n^2`
    /// buffer. See [`Self::try_update_dense_host`].
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` for a length other than `n * n` or
    /// non-finite input.
    pub fn try_update_dense_host_cast<UpperT>(
        &mut self,
        backend: &B,
        dense_rowmajor: &[UpperT],
    ) -> crate::BackendResult<()>
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        if dense_rowmajor.len() != self.n * self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_update",
                reason: format!(
                    "expected {} dense values, got {}",
                    self.n * self.n,
                    dense_rowmajor.len()
                ),
            });
        }
        if dense_rowmajor.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_update",
                reason: "dense values must be finite".into(),
            });
        }
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let layouts = BlockLayout::plan(self.n, self.block_size);
        let total: usize = layouts.iter().map(|layout| layout.len * layout.len).sum();
        let mut values = vec![T::zero(); total];
        let mut fallback_blocks = 0;
        for layout in &layouts {
            let mut cast = Vec::with_capacity(layout.len * layout.len);
            for local_row in 0..layout.len {
                let base = (layout.row_begin + local_row) * self.n + layout.row_begin;
                cast.extend(
                    dense_rowmajor[base..base + layout.len]
                        .iter()
                        .map(|value| value.cast_to_lower()),
                );
            }
            fallback_blocks += Self::invert_into(&mut values, layout, &cast);
        }
        self.fallback_blocks = fallback_blocks;
        self.block_inv.try_copy_values_from_host(backend, &values)
    }

    /// Number of diagonal blocks.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Number of blocks that failed LU factorisation and degraded to the
    /// Jacobi diagonal. Zero in the healthy case.
    pub fn fallback_blocks(&self) -> usize {
        self.fallback_blocks
    }

    /// Fixed block width (the trailing block may be narrower).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// System dimension.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the system is empty (never true: construction rejects `n == 0`).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn check_dense_inputs(
        values_len: usize,
        n: usize,
        block_size: usize,
        operation: &'static str,
    ) -> crate::BackendResult<()> {
        if n == 0 {
            return Err(crate::BackendError::InvalidArgument {
                operation,
                reason: "block-Jacobi needs a non-empty system".into(),
            });
        }
        if block_size == 0 {
            return Err(crate::BackendError::InvalidArgument {
                operation,
                reason: "block size must be at least 1".into(),
            });
        }
        if values_len != n * n {
            return Err(crate::BackendError::InvalidArgument {
                operation,
                reason: format!("expected {} dense values, got {values_len}", n * n),
            });
        }
        Ok(())
    }

    /// Invert one already-cast block into the CSR value buffer.
    ///
    /// Returns 1 when the block degraded to the diagonal, else 0.
    fn invert_into(values: &mut [T], layout: &BlockLayout, cast: &[T]) -> usize {
        let (block_values, downgraded) = invert_block_values(cast, layout.len);
        values[layout.values_offset..layout.values_offset + block_values.len()]
            .copy_from_slice(&block_values);
        downgraded
    }

    fn upload(
        backend: &B,
        n: usize,
        block_size: usize,
        layouts: &[BlockLayout],
        values: &[T],
        fallback_blocks: usize,
    ) -> crate::BackendResult<Self> {
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let (row_ptr, col_idx) = block_diagonal_structure(n, layouts);
        let block_inv = PersistentDeviceCsr::try_from_host_csr_ref(
            backend,
            HostCsrRef::new(n, n, &row_ptr, &col_idx, values),
        )?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        let workspace = backend
            .raw_alloc::<T>(n)
            .map_err(crate::BackendError::from)?;
        let runtime_diagnostics = crate::DevicePreconditionerDiagnostics::default();
        runtime_diagnostics.record_setup();
        for _ in 0..fallback_blocks {
            runtime_diagnostics.record_fallback();
        }
        Ok(Self {
            block_inv,
            n,
            block_size,
            num_blocks: layouts.len(),
            fallback_blocks,
            workspace: Mutex::new(Some(workspace)),
            runtime_diagnostics,
        })
    }
}

impl<B: Backend, T: DeviceScalar> DevicePreconditioner<B, T> for BlockJacobiBackend<B, T> {
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "block-Jacobi application failed: {error}"
            );
        }
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_untracked(backend, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_untracked(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        if r.len() != self.n || z.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_apply",
                reason: format!(
                    "expected input/output length {}, got {}/{}",
                    self.n,
                    r.len(),
                    z.len()
                ),
            });
        }
        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "block_jacobi_apply",
                    reason: "workspace mutex poisoned".into(),
                })?;
        if workspace
            .as_ref()
            .is_none_or(|buffer| buffer.len() != self.n)
        {
            backend.try_inject_preconditioner_failure(
                PreconditionerFailureInjectionPoint::WorkspaceAllocation,
            )?;
            *workspace = Some(
                backend
                    .raw_alloc::<T>(self.n)
                    .map_err(crate::BackendError::from)?,
            );
        }
        let scratch = workspace
            .as_mut()
            .ok_or(crate::BackendError::OperationFailed {
                operation: "block_jacobi_apply",
                reason: "workspace was not initialized".into(),
            })?;
        self.block_inv.try_apply(backend, r, scratch)?;
        backend.try_copy_d2d(scratch, z)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "block-Jacobi multi-RHS application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        let result = self.try_apply_colmajor_untracked(backend, rhs, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_colmajor_untracked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        let len = self
            .n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_apply_colmajor",
                reason: format!("workspace length overflow: n={}, rhs={rhs}", self.n),
            })?;
        if r.len() != len || z.len() != len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_jacobi_apply_colmajor",
                reason: format!(
                    "expected input/output length {len}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "block_jacobi_apply_colmajor",
                    reason: "workspace mutex poisoned".into(),
                })?;
        if workspace.as_ref().is_none_or(|buffer| buffer.len() != len) {
            backend.try_inject_preconditioner_failure(
                PreconditionerFailureInjectionPoint::WorkspaceAllocation,
            )?;
            *workspace = Some(
                backend
                    .raw_alloc::<T>(len)
                    .map_err(crate::BackendError::from)?,
            );
        }
        let scratch = workspace
            .as_mut()
            .ok_or(crate::BackendError::OperationFailed {
                operation: "block_jacobi_apply_colmajor",
                reason: "workspace was not initialized".into(),
            })?;
        self.block_inv
            .try_apply_colmajor(backend, rhs, r, scratch)?;
        backend.try_copy_d2d(scratch, z)
    }

    fn diagnostics(&self) -> crate::DevicePreconditionerDiagnosticsSnapshot {
        self.runtime_diagnostics.snapshot()
    }

    fn try_apply_transpose(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        // Explicit block inverses are not symmetric in general; the
        // speaker GPU adjoint path solves the explicit adjoint operator
        // instead, so no caller needs this action.
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "block_jacobi_transpose",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "block_jacobi_hermitian",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }

    fn try_apply_transpose_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "block_jacobi_transpose_colmajor",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "block_jacobi_hermitian_colmajor",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    fn make_buf(backend: &CpuBackend, host: &[f64]) -> crate::cpu::HostBuffer<f64> {
        let mut buffer = backend.raw_alloc::<f64>(host.len()).unwrap();
        backend.copy_h2d(host, &mut buffer);
        buffer
    }

    fn apply_host(
        backend: &CpuBackend,
        precond: &BlockJacobiBackend<CpuBackend, f64>,
        r: &[f64],
    ) -> Vec<f64> {
        let r_buf = make_buf(backend, r);
        let mut z_buf = backend.raw_alloc::<f64>(r.len()).unwrap();
        precond.apply(backend, &r_buf, &mut z_buf);
        let mut out = vec![0.0; r.len()];
        backend.copy_d2h(&z_buf, &mut out);
        out
    }

    #[test]
    fn block_jacobi_is_exact_on_block_diagonal_system() {
        let backend = CpuBackend::new();
        // Two 2x2 blocks with off-diagonal coupling inside each block.
        let dense = vec![
            4.0, 1.0, 0.0, 0.0, //
            1.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 5.0, 2.0, //
            0.0, 0.0, 2.0, 6.0, //
        ];
        let precond = BlockJacobiBackend::from_dense_host(&backend, &dense, 4, 2);
        assert_eq!(precond.num_blocks(), 2);
        assert_eq!(precond.fallback_blocks(), 0);
        let z = apply_host(&backend, &precond, &[5.0, 5.0, 7.0, 8.0]);
        // Block 1: inv([[4,1],[1,3]]) [5,5] = [10/11, 15/11].
        // Block 2: inv([[5,2],[2,6]]) [7,8] = [26/26, 26/26] = [1, 1].
        let expected = [10.0 / 11.0, 15.0 / 11.0, 1.0, 1.0];
        for (got, want) in z.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn block_size_one_matches_point_jacobi() {
        let backend = CpuBackend::new();
        let dense = vec![
            4.0, 1.0, 0.5, //
            1.0, 3.0, 0.25, //
            0.5, 0.25, 2.0, //
        ];
        let block = BlockJacobiBackend::from_dense_host(&backend, &dense, 3, 1);
        assert_eq!(block.num_blocks(), 3);
        let point =
            crate::preconditioner::PointJacobiBackend::from_diagonal(&backend, &[4.0, 3.0, 2.0]);
        let r = [8.0, 9.0, 10.0];
        let z_block = apply_host(&backend, &block, &r);
        let r_buf = make_buf(&backend, &r);
        let mut z_buf = backend.raw_alloc::<f64>(3).unwrap();
        point.apply(&backend, &r_buf, &mut z_buf);
        let mut z_point = vec![0.0; 3];
        backend.copy_d2h(&z_buf, &mut z_point);
        for (got, want) in z_block.iter().zip(z_point.iter()) {
            assert!((got - want).abs() < 1.0e-14, "got {got} want {want}");
        }
    }

    #[test]
    fn singular_block_falls_back_to_diagonal() {
        let backend = CpuBackend::new();
        // Second block is exactly singular (rank 1).
        let dense = vec![
            4.0, 1.0, 0.0, 0.0, //
            1.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 2.0, //
            0.0, 0.0, 2.0, 4.0, //
        ];
        let precond = BlockJacobiBackend::from_dense_host(&backend, &dense, 4, 2);
        assert_eq!(precond.fallback_blocks(), 1);
        let z = apply_host(&backend, &precond, &[5.0, 5.0, 3.0, 4.0]);
        // First block exact; fallback rows divide by the diagonal.
        assert!((z[0] - 10.0 / 11.0).abs() < 1.0e-12);
        assert!((z[1] - 15.0 / 11.0).abs() < 1.0e-12);
        assert!((z[2] - 3.0).abs() < 1.0e-12);
        assert!((z[3] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn numeric_update_tracks_new_values() {
        let backend = CpuBackend::new();
        let dense = vec![4.0, 1.0, 1.0, 3.0];
        let mut precond = BlockJacobiBackend::from_dense_host(&backend, &dense, 2, 2);
        let z_before = apply_host(&backend, &precond, &[5.0, 5.0]);
        assert!((z_before[0] - 10.0 / 11.0).abs() < 1.0e-12);
        let rotated = vec![3.0, 1.0, 1.0, 4.0];
        precond.try_update_dense_host(&backend, &rotated).unwrap();
        let z_after = apply_host(&backend, &precond, &[5.0, 5.0]);
        // inv([[3,1],[1,4]]) [5,5] = [15/11, 10/11].
        assert!((z_after[0] - 15.0 / 11.0).abs() < 1.0e-12);
        assert!((z_after[1] - 10.0 / 11.0).abs() < 1.0e-12);
        assert!(
            precond
                .try_update_dense_host(&backend, &[1.0, 2.0, 3.0])
                .is_err()
        );
    }

    #[test]
    fn cast_construction_matches_same_precision() {
        let backend = CpuBackend::new();
        let dense64 = vec![4.0, 1.0, 1.0, 3.0];
        let casted: BlockJacobiBackend<CpuBackend, f32> =
            BlockJacobiBackend::from_dense_host_cast(&backend, &dense64, 2, 2);
        assert_eq!(casted.fallback_blocks(), 0);
        let r = [5.0_f32, 5.0];
        let mut r_buf = backend.raw_alloc::<f32>(2).unwrap();
        backend.copy_h2d(&r, &mut r_buf);
        let mut z_buf = backend.raw_alloc::<f32>(2).unwrap();
        casted.apply(&backend, &r_buf, &mut z_buf);
        let mut z = [0.0_f32; 2];
        backend.copy_d2h(&z_buf, &mut z);
        assert!((z[0] - 10.0 / 11.0).abs() < 1.0e-5);
        assert!((z[1] - 15.0 / 11.0).abs() < 1.0e-5);
    }

    #[test]
    fn constructor_rejects_bad_shapes() {
        let backend = CpuBackend::new();
        assert!(
            BlockJacobiBackend::<CpuBackend, f64>::try_from_dense_host(&backend, &[], 0, 2)
                .is_err()
        );
        assert!(
            BlockJacobiBackend::<CpuBackend, f64>::try_from_dense_host(
                &backend,
                &[1.0, 2.0, 3.0],
                2,
                2
            )
            .is_err()
        );
        assert!(
            BlockJacobiBackend::<CpuBackend, f64>::try_from_dense_host(
                &backend,
                &[1.0, 0.0, 0.0, 1.0],
                2,
                0
            )
            .is_err()
        );
        assert!(
            BlockJacobiBackend::<CpuBackend, f64>::try_from_dense_host(
                &backend,
                &[1.0, 0.0, 0.0, f64::NAN],
                2,
                2
            )
            .is_err()
        );
    }
}
