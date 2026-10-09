//! Block Sparse Row (BSR) matrix format
//!
//! Stores a sparse matrix where non-zeros come in dense blocks of size `BS x BS`.
//! This is natural for systems of PDEs (Stokes BS=2, elasticity BS=3) where
//! each node has multiple DOFs that couple to every DOF of neighbouring nodes.
//!
//! **Advantages over scalar CSR:**
//! - Index storage reduced by BS² (one column index per block, not per scalar)
//! - Register blocking: the BS×BS block multiply is fully unrolled
//! - Better cache utilisation: block values are contiguous in memory
//!
//! The matrix dimensions must be multiples of BS.

use num_traits::Float;
use sonium_solvers_core::traits::{ComplexField, LinearOperator, SolverError};

/// Block Sparse Row matrix with compile-time block size `BS`.
///
/// Storage layout:
/// - `block_row_ptrs[i]` = index into `block_col_indices`/`block_values` where block-row i starts
/// - `block_col_indices[k]` = block-column index of the k-th stored block
/// - `block_values[k * BS * BS .. (k+1) * BS * BS]` = row-major dense BS×BS block
///
/// Scalar dimensions: `num_block_rows * BS` rows, `num_block_cols * BS` columns.
#[derive(Debug, Clone)]
pub struct BsrMatrix<T: ComplexField, const BS: usize> {
    /// Number of block rows
    pub num_block_rows: usize,
    /// Number of block columns
    pub num_block_cols: usize,
    /// Block row pointers (length = num_block_rows + 1)
    pub block_row_ptrs: Vec<usize>,
    /// Block column indices (length = number of stored blocks)
    pub block_col_indices: Vec<usize>,
    /// Block values stored row-major: block k at offset k*BS*BS (length = nnz_blocks * BS * BS)
    pub block_values: Vec<T>,
}

impl<T: ComplexField, const BS: usize> BsrMatrix<T, BS> {
    /// Scalar number of rows.
    pub fn num_rows(&self) -> usize {
        self.num_block_rows * BS
    }

    /// Scalar number of columns.
    pub fn num_cols(&self) -> usize {
        self.num_block_cols * BS
    }

    /// Number of stored blocks.
    pub fn nnz_blocks(&self) -> usize {
        self.block_col_indices.len()
    }

    /// Number of stored scalar non-zeros.
    pub fn nnz(&self) -> usize {
        self.nnz_blocks() * BS * BS
    }

    fn validate_checked_structure(
        &self,
        operation: &'static str,
    ) -> Result<(usize, usize), SolverError> {
        if BS == 0 {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "block size must be greater than zero".into(),
            });
        }
        let rows =
            self.num_block_rows
                .checked_mul(BS)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation,
                    reason: "scalar row dimension overflows usize".into(),
                })?;
        let cols =
            self.num_block_cols
                .checked_mul(BS)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation,
                    reason: "scalar column dimension overflows usize".into(),
                })?;
        let expected_row_ptrs =
            self.num_block_rows
                .checked_add(1)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation,
                    reason: "block row-pointer length overflows usize".into(),
                })?;
        if self.block_row_ptrs.len() != expected_row_ptrs
            || self.block_row_ptrs.first().copied() != Some(0)
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "invalid BSR block row-pointer storage".into(),
            });
        }
        if self
            .block_row_ptrs
            .windows(2)
            .any(|window| window[0] > window[1])
            || self.block_row_ptrs.last().copied() != Some(self.block_col_indices.len())
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "BSR block row pointers are not monotone or do not cover blocks".into(),
            });
        }
        let block_entries = BS
            .checked_mul(BS)
            .ok_or_else(|| SolverError::OperationFailed {
                operation,
                reason: "BSR block size overflows block storage".into(),
            })?;
        let expected_values = self
            .block_col_indices
            .len()
            .checked_mul(block_entries)
            .ok_or_else(|| SolverError::OperationFailed {
                operation,
                reason: "BSR value storage length overflows usize".into(),
            })?;
        if self.block_values.len() != expected_values {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "BSR block-value length {} != expected {expected_values}",
                    self.block_values.len()
                ),
            });
        }
        if self
            .block_col_indices
            .iter()
            .any(|&column| column >= self.num_block_cols)
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "BSR block column index exceeds matrix width".into(),
            });
        }
        if self
            .block_values
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "BSR block values contain NaN or infinity".into(),
            });
        }
        Ok((rows, cols))
    }

    /// Lightweight structural check for the infallible BSR hot path.
    ///
    /// Full [`validate_checked_structure`](Self::validate_checked_structure)
    /// additionally scans every block value for finiteness — a second full
    /// streaming pass over the matrix data with no memory-safety effect
    /// (non-finite values propagate arithmetically; they cannot cause
    /// out-of-bounds access). The hot path only needs the memory-safety
    /// invariants, checked here: dimension agreement and overflow freedom,
    /// row-pointer shape and monotonicity, value-length agreement, and
    /// in-bounds block columns. This mirrors CSR's
    /// `validate_hot_structure` split: infallible entry points propagate
    /// non-finite values arithmetically, while the `try_*` boundaries still
    /// reject them.
    fn validate_hot_structure(
        &self,
        operation: &'static str,
    ) -> Result<(usize, usize), SolverError> {
        if BS == 0 {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "block size must be greater than zero".into(),
            });
        }
        let rows =
            self.num_block_rows
                .checked_mul(BS)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation,
                    reason: "scalar row dimension overflows usize".into(),
                })?;
        let cols =
            self.num_block_cols
                .checked_mul(BS)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation,
                    reason: "scalar column dimension overflows usize".into(),
                })?;
        let expected_row_ptrs =
            self.num_block_rows
                .checked_add(1)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation,
                    reason: "block row-pointer length overflows usize".into(),
                })?;
        if self.block_row_ptrs.len() != expected_row_ptrs
            || self.block_row_ptrs.first().copied() != Some(0)
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "invalid BSR block row-pointer storage".into(),
            });
        }
        if self
            .block_row_ptrs
            .windows(2)
            .any(|window| window[0] > window[1])
            || self.block_row_ptrs.last().copied() != Some(self.block_col_indices.len())
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "BSR block row pointers are not monotone or do not cover blocks".into(),
            });
        }
        let block_entries = BS
            .checked_mul(BS)
            .ok_or_else(|| SolverError::OperationFailed {
                operation,
                reason: "BSR block size overflows block storage".into(),
            })?;
        let expected_values = self
            .block_col_indices
            .len()
            .checked_mul(block_entries)
            .ok_or_else(|| SolverError::OperationFailed {
                operation,
                reason: "BSR value storage length overflows usize".into(),
            })?;
        if self.block_values.len() != expected_values {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "BSR block-value length {} != expected {expected_values}",
                    self.block_values.len()
                ),
            });
        }
        if self
            .block_col_indices
            .iter()
            .any(|&column| column >= self.num_block_cols)
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "BSR block column index exceeds matrix width".into(),
            });
        }
        Ok((rows, cols))
    }

    fn validate_checked_input(
        operation: &'static str,
        x: &[T],
        expected: usize,
    ) -> Result<(), SolverError> {
        if x.len() != expected {
            return Err(SolverError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "input vector contains NaN or infinity".into(),
            });
        }
        Ok(())
    }

    fn validate_checked_output(
        operation: &'static str,
        y: &[T],
        expected: usize,
    ) -> Result<(), SolverError> {
        if y.len() != expected {
            return Err(SolverError::DimensionMismatch {
                expected,
                got: y.len(),
            });
        }
        if y.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "output vector contains NaN or infinity".into(),
            });
        }
        Ok(())
    }

    /// Convert a scalar CSR matrix into BSR format.
    ///
    /// The CSR matrix dimensions must be multiples of `BS`. Scalar entries
    /// within each BS×BS block are gathered into dense blocks; any entry
    /// that falls inside a block makes the entire block stored (zero-padded).
    pub fn from_csr(csr: &crate::CsrMatrix<T>) -> Self {
        Self::try_from_csr(csr)
            .unwrap_or_else(|error| panic!("invalid CSR matrix for BSR conversion: {error}"))
    }

    /// Convert a scalar CSR matrix into BSR format after validating both the
    /// mutable CSR storage and the block-dimension contract.
    pub fn try_from_csr(csr: &crate::CsrMatrix<T>) -> Result<Self, SolverError> {
        if BS == 0 {
            return Err(SolverError::OperationFailed {
                operation: "BSR construction",
                reason: "block size must be greater than zero".into(),
            });
        }
        csr.validate_structure()?;
        if !csr.num_rows.is_multiple_of(BS) {
            return Err(SolverError::OperationFailed {
                operation: "BSR construction",
                reason: format!(
                    "CSR row count {} is not a multiple of block size {BS}",
                    csr.num_rows
                ),
            });
        }
        if !csr.num_cols.is_multiple_of(BS) {
            return Err(SolverError::OperationFailed {
                operation: "BSR construction",
                reason: format!(
                    "CSR column count {} is not a multiple of block size {BS}",
                    csr.num_cols
                ),
            });
        }

        let nbr = csr.num_rows / BS;
        let nbc = csr.num_cols / BS;

        let mut block_row_ptrs = Vec::with_capacity(nbr + 1);
        let mut block_col_indices = Vec::new();
        let mut block_values = Vec::new();

        block_row_ptrs.push(0);

        // For each block row, scan all BS scalar rows to find which
        // block columns have any non-zeros.
        let mut block_cols_in_row: Vec<usize> = Vec::new();

        for br in 0..nbr {
            block_cols_in_row.clear();

            // Collect all block column indices touched by scalar rows in this block row
            for local_r in 0..BS {
                let scalar_row = br * BS + local_r;
                let start = csr.row_ptrs[scalar_row];
                let end = csr.row_ptrs[scalar_row + 1];
                for idx in start..end {
                    let bc = csr.col_indices[idx] / BS;
                    block_cols_in_row.push(bc);
                }
            }

            // Deduplicate and sort
            block_cols_in_row.sort_unstable();
            block_cols_in_row.dedup();

            // For each block column, extract the BS×BS block
            for &bc in &block_cols_in_row {
                let mut block = vec![T::zero(); BS * BS];

                for local_r in 0..BS {
                    let scalar_row = br * BS + local_r;
                    let start = csr.row_ptrs[scalar_row];
                    let end = csr.row_ptrs[scalar_row + 1];
                    for idx in start..end {
                        let scalar_col = csr.col_indices[idx];
                        if scalar_col / BS == bc {
                            let local_c = scalar_col % BS;
                            // CSR SpMV accumulates duplicate entries. Preserve
                            // that contract when packing a row into a dense
                            // block instead of letting the last duplicate win.
                            block[local_r * BS + local_c] += csr.values[idx];
                        }
                    }
                }

                block_col_indices.push(bc);
                block_values.extend_from_slice(&block);
            }

            block_row_ptrs.push(block_col_indices.len());
        }

        // Packing preserves CSR's additive duplicate semantics.  A finite
        // CSR can nevertheless overflow while duplicate entries are summed
        // into a dense block (for example, `f64::MAX + f64::MAX`).  Reject
        // that generated non-finite state at the checked construction
        // boundary instead of allowing a BSR factor/action to fail later.
        if block_values.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "BSR construction",
                reason: "generated block values contain NaN or infinity".into(),
            });
        }

        Ok(Self {
            num_block_rows: nbr,
            num_block_cols: nbc,
            block_row_ptrs,
            block_col_indices,
            block_values,
        })
    }

    /// Block-SpMV: y = A * x
    ///
    /// Each block multiply is a BS×BS dense matvec, fully unrolled by the
    /// compiler for small BS (2, 3, 4).
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        let (rows, cols) = self
            .validate_hot_structure("BSR SpMV")
            .expect("invalid BSR SpMV structure");
        assert_eq!(x.len(), cols, "Input vector size mismatch");
        let mut y = vec![T::zero(); rows];
        // SAFETY: the hot check established the memory-safety contract
        // below (dimensions, monotonicity, in-bounds block columns, value
        // length); values only affect arithmetic, never indexing.
        unsafe { self.matvec_into_buf_unchecked(x, &mut y) };
        y
    }

    /// Block-SpMV into a pre-allocated buffer: y = A * x.
    ///
    /// Overwrites `y` with the result. No allocation.
    pub fn matvec_into_buf(&self, x: &[T], y: &mut [T]) {
        let (rows, cols) = self
            .validate_hot_structure("BSR SpMV")
            .expect("invalid BSR SpMV structure");
        assert_eq!(x.len(), cols, "Input vector size mismatch");
        assert_eq!(y.len(), rows, "Output vector size mismatch");
        // SAFETY: the hot check established the memory-safety contract
        // below (dimensions, monotonicity, in-bounds block columns, value
        // length); values only affect arithmetic, never indexing.
        unsafe { self.matvec_into_buf_unchecked(x, y) };
    }

    /// Block-SpMV into a buffer without structural validation.
    ///
    /// # Safety
    ///
    /// The caller guarantees the memory-safety invariants (`BS > 0`,
    /// scalar dimensions free of overflow, block row pointers shaped
    /// `num_block_rows + 1`, rooted at zero, monotone and covering the
    /// stored blocks, value length equal to
    /// `nnz_blocks * BS * BS`, in-bounds block columns; `x`/`y` matching
    /// the scalar column/row counts) — e.g. via
    /// [`Self::validate_hot_structure`] or the stronger
    /// [`Self::validate_checked_structure`]. Violating this is immediate
    /// undefined behavior: the kernel below uses unchecked indexing
    /// throughout. Only call this on matrices the caller owns and never
    /// mutates after a validating construction; user-supplied matrices must
    /// go through the checked entry points.
    pub unsafe fn matvec_into_buf_unchecked(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_cols(), "Input vector size mismatch");
        assert_eq!(y.len(), self.num_rows(), "Output vector size mismatch");
        self.matvec_forward_kernel(x, y);
    }

    /// Dense BS×BS block-row kernel shared by every forward entry point.
    ///
    /// Overwrites `y`. Called only after structure and length validation;
    /// every index below is in bounds by that contract.
    fn matvec_forward_kernel(&self, x: &[T], y: &mut [T]) {
        y.fill(T::zero());
        let x_slice = x;
        let y_slice = y;

        for br in 0..self.num_block_rows {
            let bk_start = self.block_row_ptrs[br];
            let bk_end = self.block_row_ptrs[br + 1];
            let y_off = br * BS;

            for bk in bk_start..bk_end {
                let bc = self.block_col_indices[bk];
                let x_off = bc * BS;
                let v_off = bk * BS * BS;

                // Dense BS×BS block multiply: y[y_off..] += block * x[x_off..]
                // The compiler unrolls this for const BS.
                // SAFETY: upheld by the caller per the entry-point contract:
                // `bk` is inside a validated block-row range, `bc` is an
                // in-bounds block column, `v_off` addresses a stored block,
                // and `x`/`y` have the required scalar lengths.
                for lr in 0..BS {
                    let mut sum = T::zero();
                    for lc in 0..BS {
                        unsafe {
                            sum += *self.block_values.get_unchecked(v_off + lr * BS + lc)
                                * *x_slice.get_unchecked(x_off + lc);
                        }
                    }
                    unsafe {
                        *y_slice.get_unchecked_mut(y_off + lr) += sum;
                    }
                }
            }
        }
    }

    // Called only after structure and input validation. Public metadata can
    // change between calls, so construction-time validation is insufficient.
    fn matvec_validated(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.num_rows()];
        self.matvec_forward_kernel(x, &mut y);
        y
    }

    /// Parallel block-SpMV using rayon.
    pub fn matvec_parallel(&self, x: &[T]) -> Vec<T>
    where
        T: Send + Sync,
    {
        use rayon::prelude::*;

        // Hot-path validation only (memory-safety invariants); non-finite
        // values propagate arithmetically, mirroring the CSR hot path. The
        // `try_*` boundaries still reject them.
        let (rows, cols) = self
            .validate_hot_structure("BSR parallel SpMV")
            .expect("invalid BSR parallel SpMV structure");
        assert_eq!(x.len(), cols, "Input vector size mismatch");
        let x_slice = x;

        let mut y = vec![T::zero(); rows];

        sonium_solvers_sys::runtime::with_bandwidth_pool(|| {
            y.par_chunks_mut(BS).enumerate().for_each(|(br, local_y)| {
                let bk_start = self.block_row_ptrs[br];
                let bk_end = self.block_row_ptrs[br + 1];

                for bk in bk_start..bk_end {
                    let bc = self.block_col_indices[bk];
                    let x_off = bc * BS;
                    let v_off = bk * BS * BS;

                    #[allow(clippy::needless_range_loop)]
                    for lr in 0..BS {
                        let mut sum = T::zero();
                        for lc in 0..BS {
                            unsafe {
                                sum += *self.block_values.get_unchecked(v_off + lr * BS + lc)
                                    * *x_slice.get_unchecked(x_off + lc);
                            }
                        }
                        local_y[lr] += sum;
                    }
                }
            });
        });

        debug_assert_eq!(y.len(), rows);
        y
    }

    /// Checked block-SpMV. Public BSR metadata is validated before the
    /// unchecked block kernel reads row pointers, columns, or values.
    pub fn try_matvec(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        let (rows, cols) = self.validate_checked_structure("BSR SpMV")?;
        Self::validate_checked_input("BSR SpMV", x, cols)?;
        let y = self.matvec_validated(x);
        Self::validate_checked_output("BSR SpMV", &y, rows)?;
        Ok(y)
    }

    /// Checked transpose block-SpMV.
    pub fn try_matvec_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        let (rows, cols) = self.validate_checked_structure("BSR transpose SpMV")?;
        Self::validate_checked_input("BSR transpose SpMV", x, rows)?;
        let mut y = vec![T::zero(); cols];
        self.matvec_adjoint_kernel::<false>(x, &mut y);
        Self::validate_checked_output("BSR transpose SpMV", &y, cols)?;
        Ok(y)
    }

    /// Checked Hermitian-transpose block-SpMV.
    pub fn try_matvec_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        let (rows, cols) = self.validate_checked_structure("BSR Hermitian SpMV")?;
        Self::validate_checked_input("BSR Hermitian SpMV", x, rows)?;
        let mut y = vec![T::zero(); cols];
        self.matvec_adjoint_kernel::<true>(x, &mut y);
        Self::validate_checked_output("BSR Hermitian SpMV", &y, cols)?;
        Ok(y)
    }

    /// Transpose block-SpMV into a pre-allocated buffer: y = A^T * x.
    ///
    /// Overwrites `y` with the result. No allocation.
    pub fn matvec_transpose_into_buf(&self, x: &[T], y: &mut [T]) {
        let (rows, cols) = self
            .validate_hot_structure("BSR transpose SpMV")
            .expect("invalid BSR transpose SpMV structure");
        assert_eq!(x.len(), rows, "Input vector size mismatch");
        assert_eq!(y.len(), cols, "Output vector size mismatch");
        // SAFETY: the hot check established the memory-safety contract
        // below; values only affect arithmetic, never indexing.
        unsafe { self.matvec_transpose_into_buf_unchecked(x, y) };
    }

    /// Transpose block-SpMV into a buffer without structural validation.
    ///
    /// # Safety
    ///
    /// Same memory-safety contract as
    /// [`Self::matvec_into_buf_unchecked`], with `x`/`y` matching the
    /// scalar row/column counts (transposed). Only call this on matrices
    /// the caller owns and never mutates after a validating construction;
    /// user-supplied matrices must go through the checked entry points.
    pub unsafe fn matvec_transpose_into_buf_unchecked(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_rows(), "Input vector size mismatch");
        assert_eq!(y.len(), self.num_cols(), "Output vector size mismatch");
        self.matvec_adjoint_kernel::<false>(x, y);
    }

    /// Hermitian-transpose block-SpMV into a pre-allocated buffer: y = A^H * x.
    ///
    /// Overwrites `y` with the result. No allocation.
    pub fn matvec_hermitian_into_buf(&self, x: &[T], y: &mut [T]) {
        let (rows, cols) = self
            .validate_hot_structure("BSR Hermitian SpMV")
            .expect("invalid BSR Hermitian SpMV structure");
        assert_eq!(x.len(), rows, "Input vector size mismatch");
        assert_eq!(y.len(), cols, "Output vector size mismatch");
        // SAFETY: the hot check established the memory-safety contract
        // below; values only affect arithmetic, never indexing.
        unsafe { self.matvec_hermitian_into_buf_unchecked(x, y) };
    }

    /// Hermitian block-SpMV into a buffer without structural validation.
    ///
    /// # Safety
    ///
    /// Same memory-safety contract as
    /// [`Self::matvec_into_buf_unchecked`], with `x`/`y` matching the
    /// scalar row/column counts (transposed). Only call this on matrices
    /// the caller owns and never mutates after a validating construction;
    /// user-supplied matrices must go through the checked entry points.
    pub unsafe fn matvec_hermitian_into_buf_unchecked(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_rows(), "Input vector size mismatch");
        assert_eq!(y.len(), self.num_cols(), "Output vector size mismatch");
        self.matvec_adjoint_kernel::<true>(x, y);
    }

    /// Transposed/Hermitian block scatter kernel shared by every adjoint
    /// entry point (`CONJ` selects the Hermitian action).
    ///
    /// Overwrites `y`. A^T for BSR transposes each block and swaps block
    /// row/column. Called only after structure and length validation;
    /// every index below is in bounds by that contract.
    fn matvec_adjoint_kernel<const CONJ: bool>(&self, x: &[T], y: &mut [T]) {
        y.fill(T::zero());
        let x_slice = x;
        let y_slice = y;

        for br in 0..self.num_block_rows {
            let bk_start = self.block_row_ptrs[br];
            let bk_end = self.block_row_ptrs[br + 1];
            let x_off = br * BS;

            for bk in bk_start..bk_end {
                let bc = self.block_col_indices[bk];
                let y_off = bc * BS;
                let v_off = bk * BS * BS;

                // Transposed block multiply: y[y_off..] += block^T * x[x_off..]
                // SAFETY: upheld by the caller per the entry-point contract:
                // `bk` is inside a validated block-row range, `bc` is an
                // in-bounds block column, `v_off` addresses a stored block,
                // and `x`/`y` have the required scalar lengths, so all of
                // `x_off + lr`, `y_off + lc`, `v_off + lr * BS + lc` are in
                // bounds.
                for lc in 0..BS {
                    let mut sum = T::zero();
                    for lr in 0..BS {
                        unsafe {
                            let value = *self.block_values.get_unchecked(v_off + lr * BS + lc);
                            let value = if CONJ { value.conj() } else { value };
                            sum += value * *x_slice.get_unchecked(x_off + lr);
                        }
                    }
                    unsafe {
                        *y_slice.get_unchecked_mut(y_off + lc) += sum;
                    }
                }
            }
        }
    }

    /// Shared parallel gate for the forward BSR action.
    ///
    /// Same policy as CSR forward SpMV: enough total work (nnz) plus enough
    /// rows to distribute (a few ultra-dense rows would pay full
    /// thread-pool overhead with ~no row parallelism). Uses the crate-wide
    /// SpMV thresholds (defaults 50_000 nnz / 4_000 rows, matching the
    /// previous hardcoded BSR gate) so `SONIUM_SPARSE_PAR_*` overrides
    /// apply uniformly.
    fn should_parallelize_forward(&self) -> bool {
        let nnz = self.nnz();
        nnz >= crate::spmv_plan::par_nnz_threshold()
            && self.num_rows() >= crate::spmv_plan::par_min_rows()
            && sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(nnz) > 1
    }
}

impl<T: ComplexField, const BS: usize> LinearOperator<T> for BsrMatrix<T, BS> {
    fn num_rows(&self) -> usize {
        self.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.num_cols()
    }

    fn apply(&self, x: &[T]) -> Vec<T> {
        if self.should_parallelize_forward() {
            return self.matvec_parallel(x);
        }
        self.matvec(x)
    }

    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec(x)
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        if self.should_parallelize_forward() {
            let parallel = self.matvec_parallel(x);
            y.copy_from_slice(&parallel);
            return;
        }
        self.matvec_into_buf(x, y);
    }

    fn apply_transpose(&self, x: &[T]) -> Vec<T> {
        let (rows, cols) = self
            .validate_hot_structure("BSR transpose SpMV")
            .expect("invalid BSR transpose SpMV structure");
        assert_eq!(x.len(), rows, "Input vector size mismatch");
        let mut y = vec![T::zero(); cols];
        // SAFETY: the hot check established the memory-safety contract;
        // values only affect arithmetic, never indexing.
        unsafe { self.matvec_transpose_into_buf_unchecked(x, &mut y) };
        y
    }

    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_transpose(x)
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_transpose_into_buf(x, y);
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        let (rows, cols) = self
            .validate_hot_structure("BSR Hermitian SpMV")
            .expect("invalid BSR Hermitian SpMV structure");
        assert_eq!(x.len(), rows, "Input vector size mismatch");
        let mut y = vec![T::zero(); cols];
        // SAFETY: the hot check established the memory-safety contract;
        // values only affect arithmetic, never indexing.
        unsafe { self.matvec_hermitian_into_buf_unchecked(x, &mut y) };
        y
    }

    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_hermitian(x)
    }

    fn transpose_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn is_square(&self) -> bool {
        self.num_block_rows == self.num_block_cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrMatrix;
    use sonium_solvers_core::dense::DenseMatrix;

    fn stokes_like_4x4() -> CsrMatrix<f64> {
        let dense = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                4.0_f64, 1.0, 0.5, 0.0, 1.0, 4.0, 0.0, 0.5, 0.5, 0.0, 3.0, 1.0, 0.0, 0.5, 1.0, 3.0,
            ],
        );
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn test_bsr_from_csr_dimensions() {
        let csr = stokes_like_4x4();
        let bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&csr);

        assert_eq!(bsr.num_block_rows, 2);
        assert_eq!(bsr.num_block_cols, 2);
        assert_eq!(bsr.num_rows(), 4);
        assert_eq!(bsr.num_cols(), 4);
    }

    #[test]
    fn checked_bsr_constructor_rejects_non_multiple_dimensions() {
        let csr = CsrMatrix::from_raw_parts(3, 2, vec![0, 1, 1, 1], vec![0], vec![1.0_f64]);
        let error = BsrMatrix::<f64, 2>::try_from_csr(&csr).unwrap_err();
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "BSR construction",
                ..
            }
        ));
    }

    #[test]
    fn checked_bsr_constructor_rejects_mutated_csr_structure() {
        let mut csr = stokes_like_4x4();
        csr.row_ptrs[4] = csr.row_ptrs[4].saturating_sub(1);
        let error = BsrMatrix::<f64, 2>::try_from_csr(&csr).unwrap_err();
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "CSR validation",
                ..
            }
        ));
    }

    #[test]
    fn checked_bsr_execution_rejects_mutated_public_metadata() {
        let csr = stokes_like_4x4();
        let mut bsr = BsrMatrix::<f64, 2>::from_csr(&csr);
        bsr.block_row_ptrs[1] = bsr.block_col_indices.len() + 1;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bsr.try_apply(&[1.0, 2.0, 3.0, 4.0])
        }))
        .expect("checked BSR apply must not panic");
        assert!(matches!(
            result,
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "BSR SpMV"
        ));
    }

    #[test]
    fn checked_bsr_execution_rejects_nonfinite_block_values() {
        let csr = stokes_like_4x4();
        let mut bsr = BsrMatrix::<f64, 2>::from_csr(&csr);
        bsr.block_values[0] = f64::NAN;

        assert!(matches!(
            bsr.try_apply(&[1.0, 2.0, 3.0, 4.0]),
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "BSR SpMV"
        ));
        assert!(matches!(
            bsr.try_apply_transpose(&[1.0, 2.0, 3.0, 4.0]),
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "BSR transpose SpMV"
        ));
    }

    #[test]
    fn bsr_duplicate_entries_match_csr_accumulation() {
        let csr = CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![0, 0], vec![1.0_f64, 2.0]);
        let bsr = BsrMatrix::<f64, 2>::try_from_csr(&csr).expect("valid BSR conversion");
        let x = [3.0, 4.0];
        assert_eq!(bsr.matvec(&x), csr.matvec(&x));
    }

    #[test]
    fn checked_bsr_constructor_rejects_generated_non_finite_duplicate_sum() {
        let csr =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![0, 0], vec![f64::MAX, f64::MAX]);

        let error = BsrMatrix::<f64, 2>::try_from_csr(&csr).unwrap_err();
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "BSR construction",
                ref reason,
            } if reason.contains("generated block values")
        ));
    }

    #[test]
    fn test_bsr_matvec_matches_csr() {
        let csr = stokes_like_4x4();
        let bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&csr);

        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y_csr = csr.matvec(&x);
        let y_bsr = bsr.matvec(&x);

        for i in 0..4 {
            assert!(
                (y_csr[i] - y_bsr[i]).abs() < 1e-12,
                "Mismatch at {i}: csr={} bsr={}",
                y_csr[i],
                y_bsr[i]
            );
        }
    }

    #[test]
    fn test_bsr_transpose_matches_csr() {
        let csr = stokes_like_4x4();
        let bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&csr);

        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y_csr = csr.matvec_transpose(&x);
        let y_bsr = bsr.apply_transpose(&x);

        for i in 0..4 {
            assert!(
                (y_csr[i] - y_bsr[i]).abs() < 1e-12,
                "Transpose mismatch at {i}: csr={} bsr={}",
                y_csr[i],
                y_bsr[i]
            );
        }
    }

    #[test]
    fn test_bsr_block_size_3() {
        // 6x6 matrix with 3x3 blocks (elasticity-like)
        let mut dense = DenseMatrix::from_elem(6, 6, 0.0_f64);
        for i in 0..6 {
            dense[[i, i]] = 10.0;
        }
        // Off-diagonal coupling within blocks
        dense[[0, 1]] = 1.0;
        dense[[1, 0]] = 1.0;
        dense[[3, 4]] = 2.0;
        dense[[4, 3]] = 2.0;
        // Inter-block coupling
        dense[[0, 3]] = 0.5;
        dense[[3, 0]] = 0.5;

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let bsr: BsrMatrix<f64, 3> = BsrMatrix::from_csr(&csr);

        assert_eq!(bsr.num_block_rows, 2);
        assert_eq!(bsr.num_block_cols, 2);

        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_csr = csr.matvec(&x);
        let y_bsr = bsr.matvec(&x);

        for i in 0..6 {
            assert!(
                (y_csr[i] - y_bsr[i]).abs() < 1e-12,
                "BS=3 mismatch at {i}: csr={} bsr={}",
                y_csr[i],
                y_bsr[i]
            );
        }
    }

    #[test]
    fn test_bsr_linear_operator() {
        let csr = stokes_like_4x4();
        let bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&csr);

        assert_eq!(LinearOperator::<f64>::num_rows(&bsr), 4);
        assert_eq!(LinearOperator::<f64>::num_cols(&bsr), 4);
        assert!(bsr.is_square());

        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y = LinearOperator::<f64>::apply(&bsr, &x);
        let y_ref = csr.matvec(&x);

        for i in 0..4 {
            assert!((y[i] - y_ref[i]).abs() < 1e-12);
        }
    }
}
