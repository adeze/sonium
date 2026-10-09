//! Compressed Sparse Row (CSR) matrix format
//!
//! CSR format stores:
//! - `values`: Non-zero entries in row-major order
//! - `col_indices`: Column index for each value
//! - `row_ptrs`: Index into values/col_indices where each row starts

use crate::partition::{build_adjacency, nested_dissection, nnz_balanced_ranges};
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, Zero};
use sonium_solvers_core::PrecisionCast;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{ComplexField, LinearOperator, SolverError};
use std::any::TypeId;
use std::collections::VecDeque;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

use rayon::prelude::*;

const TRANSPOSE_PAR_WORKSPACE_ELEMS_LIMIT: usize = 16_000_000;

/// Kernel family selected by the sparse action dispatch.
///
/// `Avx2Fma` and `Avx2` are selected at runtime only after checking the host
/// feature bits. The FMA variant is kept distinct in reports so a baseline
/// build and an explicitly `+avx2,+fma` build cannot be confused.
/// The AVX2 implementation is compiled into an isolated function, so a
/// baseline build remains safe on hosts without AVX2. Set
/// `SONIUM_SPARSE_FORCE_SCALAR=1` before process start to collect a scalar
/// baseline with the same binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmvKernel {
    Scalar,
    Avx2,
    Avx2Fma,
    Neon,
}

impl SpmvKernel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx2Fma => "avx2-fma",
            Self::Neon => "neon",
        }
    }
}

fn force_scalar_dispatch() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| {
        std::env::var_os("SONIUM_SPARSE_FORCE_SCALAR")
            .is_some_and(|value| value == "1" || value == "true")
    })
}

fn disable_fma_dispatch() -> bool {
    static DISABLE_FMA: OnceLock<bool> = OnceLock::new();
    *DISABLE_FMA.get_or_init(|| {
        std::env::var_os("SONIUM_SPARSE_DISABLE_FMA")
            .is_some_and(|value| value == "1" || value == "true")
    })
}

#[inline]
fn avx2_dispatch_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !force_scalar_dispatch() && std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Return whether the host can safely enter kernels compiled with AVX2+FMA.
///
/// This is crate-visible because the compact and SELL representations share
/// the same runtime dispatch contract as direct CSR. Keeping the feature
/// check in one place also makes the selected kernel reported by every
/// representation observable and fail-closed on unsupported hosts.
#[inline]
pub(crate) fn fma_dispatch_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        avx2_dispatch_available() && !disable_fma_dispatch() && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Return the runtime-selected kernel family for a concrete sparse scalar.
///
/// This is also the dispatch value recorded by [`SpmvPlanReport`]. Complex32
/// intentionally remains scalar on x86 because no x86 Complex32 SIMD kernel
/// is provided.
pub fn selected_spmv_kernel<T: ComplexField>() -> SpmvKernel {
    if avx2_dispatch_available()
        && (TypeId::of::<T>() == TypeId::of::<f32>()
            || TypeId::of::<T>() == TypeId::of::<f64>()
            || TypeId::of::<T>() == TypeId::of::<Complex64>())
    {
        return if fma_dispatch_available() {
            SpmvKernel::Avx2Fma
        } else {
            SpmvKernel::Avx2
        };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if !force_scalar_dispatch() {
        return SpmvKernel::Neon;
    }

    SpmvKernel::Scalar
}

/// Return the host features relevant to sparse dispatch in a stable form for
/// machine-readable evidence.
pub fn detected_spmv_features() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::is_x86_feature_detected!("avx2");
        let fma = std::is_x86_feature_detected!("fma");
        format!(
            "arch=x86_64;avx2={avx2};fma={fma};avx2_fma={};disable_fma={};force_scalar={}",
            avx2 && fma && !disable_fma_dispatch() && !force_scalar_dispatch(),
            disable_fma_dispatch(),
            force_scalar_dispatch()
        )
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return format!(
            "arch=aarch64;neon=true;force_scalar={}",
            force_scalar_dispatch()
        );
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        "arch=unknown;scalar=true".to_string()
    }
}

/// Compressed Sparse Row (CSR) matrix format
///
/// Memory-efficient storage for sparse matrices with O(nnz) space complexity.
/// Matrix-vector products are O(nnz) instead of O(n²) for dense matrices.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T: ComplexField> {
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_cols: usize,
    /// Non-zero values in row-major order
    pub values: Vec<T>,
    /// Column indices for each value
    pub col_indices: Vec<usize>,
    /// Row pointers: `row_ptrs[i]` is the start index in values/col_indices for row `i`
    /// `row_ptrs[num_rows]` = nnz (total number of non-zeros)
    pub row_ptrs: Vec<usize>,
}

#[cfg(test)]
mod checked_raw_parts_tests {
    use super::CsrMatrix;
    use num_complex::Complex64;
    use sonium_solvers_core::traits::{AdjointCapability, LinearOperator};

    #[test]
    fn checked_constructor_rejects_non_monotone_row_pointers() {
        let result =
            CsrMatrix::<f64>::try_from_raw_parts(2, 2, vec![0, 2, 1], vec![0, 1], vec![1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn checked_constructor_rejects_out_of_range_columns() {
        let result = CsrMatrix::<f64>::try_from_raw_parts(1, 2, vec![0, 1], vec![2], vec![1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn checked_constructor_rejects_non_finite_values() {
        let result =
            CsrMatrix::<f64>::try_from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::NAN]);
        assert!(result.is_err());
    }

    #[test]
    fn checked_empty_constructors_reject_row_pointer_overflow() {
        assert!(CsrMatrix::<f64>::try_new(usize::MAX, 0).is_err());
        assert!(CsrMatrix::<f64>::try_with_capacity(usize::MAX, 0, 0).is_err());
    }

    #[test]
    fn canonical_validation_rejects_duplicate_and_unsorted_columns() {
        let unsorted =
            CsrMatrix::<f64>::try_from_raw_parts(1, 2, vec![0, 2], vec![1, 0], vec![1.0, 2.0])
                .expect("structural CSR validation should allow noncanonical rows");
        assert!(unsorted.validate_structure().is_ok());
        assert!(unsorted.validate_canonical_structure().is_err());

        let duplicate =
            CsrMatrix::<f64>::try_from_raw_parts(1, 2, vec![0, 2], vec![0, 0], vec![1.0, 2.0])
                .expect("structural CSR validation should allow duplicate entries");
        assert!(duplicate.validate_canonical_structure().is_err());
    }

    #[test]
    fn checked_triplet_constructor_rejects_invalid_indices_and_values() {
        assert!(CsrMatrix::<f64>::try_from_triplets(1, 1, vec![(1, 0, 1.0)]).is_err());
        assert!(CsrMatrix::<f64>::try_from_triplets(1, 1, vec![(0, 1, 1.0)]).is_err());
        assert!(CsrMatrix::<f64>::try_from_triplets(1, 1, vec![(0, 0, f64::NAN)]).is_err());
    }

    #[test]
    fn checked_adjoint_contract_matches_complex_inner_product() {
        let matrix = CsrMatrix::from_triplets(
            2,
            3,
            vec![
                (0, 0, Complex64::new(1.0, 2.0)),
                (0, 2, Complex64::new(-0.5, 1.0)),
                (1, 1, Complex64::new(2.0, -1.0)),
            ],
        );
        assert_eq!(matrix.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(matrix.hermitian_capability(), AdjointCapability::Exact);

        let x = [
            Complex64::new(0.5, -1.0),
            Complex64::new(1.5, 0.25),
            Complex64::new(-0.75, 2.0),
        ];
        let y = [Complex64::new(2.0, 0.5), Complex64::new(-1.0, 1.25)];
        let ax = matrix.apply(&x);
        let ah_y = matrix
            .try_apply_hermitian(&y)
            .expect("CSR Hermitian action must be supported");
        let lhs: Complex64 = ax.iter().zip(y.iter()).map(|(a, b)| a.conj() * b).sum();
        let rhs: Complex64 = x.iter().zip(ah_y.iter()).map(|(a, b)| a.conj() * b).sum();
        assert!((lhs - rhs).norm() < 1e-12);

        assert!(
            matrix
                .try_apply_hermitian(&[Complex64::new(1.0, 0.0)])
                .is_err()
        );
    }
}

impl<T: ComplexField> CsrMatrix<T> {
    /// Create a new empty CSR matrix
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self::try_new(num_rows, num_cols)
            .unwrap_or_else(|error| panic!("invalid CSR dimensions: {error}"))
    }

    /// Create a new empty CSR matrix after checking the row-pointer length.
    pub fn try_new(
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        let row_ptr_len = num_rows.checked_add(1).ok_or_else(|| {
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!("row count {num_rows} overflows row-pointer length"),
            }
        })?;
        Ok(Self {
            num_rows,
            num_cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptrs: vec![0; row_ptr_len],
        })
    }

    /// Create a CSR matrix with pre-allocated capacity.
    pub fn with_capacity(num_rows: usize, num_cols: usize, nnz_estimate: usize) -> Self {
        Self::try_with_capacity(num_rows, num_cols, nnz_estimate)
            .unwrap_or_else(|error| panic!("invalid CSR dimensions: {error}"))
    }

    /// Create a CSR matrix with pre-allocated capacity after checking the
    /// row-pointer length.
    pub fn try_with_capacity(
        num_rows: usize,
        num_cols: usize,
        nnz_estimate: usize,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        let row_ptr_len = num_rows.checked_add(1).ok_or_else(|| {
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!("row count {num_rows} overflows row-pointer length"),
            }
        })?;
        Ok(Self {
            num_rows,
            num_cols,
            values: Vec::with_capacity(nnz_estimate),
            col_indices: Vec::with_capacity(nnz_estimate),
            row_ptrs: vec![0; row_ptr_len],
        })
    }

    /// Create a CSR matrix from raw components.
    ///
    /// This compatibility constructor retains the historical panic-on-invalid
    /// API. New callers should use [`Self::try_from_raw_parts`], which reports
    /// malformed row pointers and column indices as a typed error.
    pub fn from_raw_parts(
        num_rows: usize,
        num_cols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        Self::try_from_raw_parts(num_rows, num_cols, row_ptrs, col_indices, values)
            .unwrap_or_else(|error| panic!("invalid CSR matrix: {error}"))
    }

    /// Create a CSR matrix from raw components with complete structural
    /// validation.
    pub fn try_from_raw_parts(
        num_rows: usize,
        num_cols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        let expected_row_ptr_len = num_rows.checked_add(1).ok_or_else(|| {
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!("row count {num_rows} overflows row-pointer length"),
            }
        })?;
        if row_ptrs.len() != expected_row_ptr_len {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!(
                    "row_ptrs has length {}, expected {}",
                    row_ptrs.len(),
                    expected_row_ptr_len
                ),
            });
        }
        if col_indices.len() != values.len() {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!(
                    "col_indices has length {}, values has length {}",
                    col_indices.len(),
                    values.len()
                ),
            });
        }
        if row_ptrs.first().copied() != Some(0) {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: "row_ptrs must start at zero".into(),
            });
        }
        if row_ptrs.windows(2).any(|window| window[0] > window[1]) {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: "row_ptrs must be monotonically nondecreasing".into(),
            });
        }
        if row_ptrs.last().copied() != Some(values.len()) {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!(
                    "terminal row pointer {:?} does not equal nnz {}",
                    row_ptrs.last(),
                    values.len()
                ),
            });
        }
        if let Some((index, &column)) = col_indices
            .iter()
            .enumerate()
            .find(|&(_, &column)| column >= num_cols)
        {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!("column index {column} at position {index} >= {num_cols}"),
            });
        }
        if let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.norm().is_finite())
        {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR construction",
                reason: format!("value at position {index} is non-finite: {value:?}"),
            });
        }

        Ok(Self {
            num_rows,
            num_cols,
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Create a CSR matrix from an explicitly typed index representation.
    ///
    /// Sparse files and foreign interfaces commonly carry CSR offsets and
    /// columns as 32-bit or 64-bit integers. The solver storage remains
    /// `usize`, but conversion is checked before the existing structural
    /// validation runs. This keeps narrow-index inputs portable while
    /// rejecting values that cannot be represented on the host.
    pub fn try_from_raw_parts_indexed<RI, CI>(
        num_rows: usize,
        num_cols: usize,
        row_ptrs: impl IntoIterator<Item = RI>,
        col_indices: impl IntoIterator<Item = CI>,
        values: Vec<T>,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError>
    where
        RI: TryInto<usize>,
        RI::Error: std::fmt::Debug,
        CI: TryInto<usize>,
        CI::Error: std::fmt::Debug,
    {
        let row_ptrs = row_ptrs
            .into_iter()
            .map(|index| {
                index.try_into().map_err(|error| {
                    sonium_solvers_core::traits::SolverError::OperationFailed {
                        operation: "CSR construction",
                        reason: format!("row pointer index conversion failed: {error:?}"),
                    }
                })
            })
            .collect::<Result<Vec<usize>, _>>()?;
        let col_indices = col_indices
            .into_iter()
            .map(|index| {
                index.try_into().map_err(|error| {
                    sonium_solvers_core::traits::SolverError::OperationFailed {
                        operation: "CSR construction",
                        reason: format!("column index conversion failed: {error:?}"),
                    }
                })
            })
            .collect::<Result<Vec<usize>, _>>()?;
        Self::try_from_raw_parts(num_rows, num_cols, row_ptrs, col_indices, values)
    }

    /// Create a CSR matrix from a dense matrix
    ///
    /// Only stores entries with magnitude > threshold
    /// Validate the public CSR storage before passing it to a lower-level
    /// consumer.
    ///
    /// The storage fields remain public for compatibility, so a caller can
    /// mutate a previously valid matrix after construction. Checked plan and
    /// solver entry points must call this method at their boundary instead of
    /// assuming the constructor invariant still holds.
    pub fn validate_structure(&self) -> Result<(), sonium_solvers_core::traits::SolverError> {
        let expected_row_ptr_len = self.num_rows.checked_add(1).ok_or_else(|| {
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: format!("row count {} overflows row-pointer length", self.num_rows),
            }
        })?;

        if self.row_ptrs.len() != expected_row_ptr_len {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: format!(
                    "row_ptrs has length {}, expected {}",
                    self.row_ptrs.len(),
                    expected_row_ptr_len
                ),
            });
        }
        if self.col_indices.len() != self.values.len() {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: format!(
                    "col_indices has length {}, values has length {}",
                    self.col_indices.len(),
                    self.values.len()
                ),
            });
        }
        if self.row_ptrs.first().copied() != Some(0) {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: "row_ptrs must start at zero".into(),
            });
        }
        if self.row_ptrs.windows(2).any(|window| window[0] > window[1]) {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: "row_ptrs must be monotone".into(),
            });
        }
        if self.row_ptrs.last().copied() != Some(self.values.len()) {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: format!(
                    "terminal row pointer must equal values length {}, got {:?}",
                    self.values.len(),
                    self.row_ptrs.last()
                ),
            });
        }
        if let Some((index, &column)) = self
            .col_indices
            .iter()
            .enumerate()
            .find(|(_, column)| **column >= self.num_cols)
        {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: format!(
                    "column index {} at entry {} is outside {} columns",
                    column, index, self.num_cols
                ),
            });
        }
        if let Some((index, _value)) = self
            .values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.norm().is_finite())
        {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                reason: format!("matrix contains NaN or infinity at entry {}", index),
            });
        }
        Ok(())
    }

    /// Lightweight structural check for the infallible SpMV/SpMM hot path.
    ///
    /// Full [`validate_structure`](Self::validate_structure) additionally scans
    /// every value for finiteness. That scan is a second full streaming pass
    /// over the matrix data with no memory-safety effect (non-finite values
    /// propagate arithmetically; they cannot cause out-of-bounds access), and
    /// on bandwidth-bound problems it costs more than the SpMV itself. The
    /// hot path only needs the memory-safety invariants, checked here:
    ///
    /// - O(1) length agreement between `row_ptrs` / `values` / `col_indices`
    ///   plus first/terminal row-pointer consistency, so every `start..end`
    ///   row range resolved below is in bounds;
    /// - monotone row pointers (one streaming pass over `row_ptrs`, O(rows)),
    ///   so `end - start` row lengths cannot underflow in the SIMD kernels;
    /// - every column index in bounds (one streaming pass over `col_indices`,
    ///   O(nnz)), so gathers into `x` cannot go out of bounds.
    ///
    /// Duplicate or unsorted entries are accepted exactly like the full
    /// validation (rows accumulate). Callers that need value-finiteness or
    /// canonical ordering must use [`validate_structure`](Self::validate_structure)
    /// or [`validate_canonical_structure`](Self::validate_canonical_structure).
    fn validate_hot_structure(&self) -> Result<(), sonium_solvers_core::traits::SolverError> {
        use sonium_solvers_core::traits::SolverError;
        if self.row_ptrs.len() != self.num_rows + 1 {
            return Err(SolverError::OperationFailed {
                operation: "CSR hot-path validation",
                reason: format!(
                    "row_ptrs has length {}, expected {}",
                    self.row_ptrs.len(),
                    self.num_rows + 1
                ),
            });
        }
        if self.col_indices.len() != self.values.len() {
            return Err(SolverError::OperationFailed {
                operation: "CSR hot-path validation",
                reason: format!(
                    "col_indices has length {}, values has length {}",
                    self.col_indices.len(),
                    self.values.len()
                ),
            });
        }
        if self.row_ptrs.first().copied() != Some(0) {
            return Err(SolverError::OperationFailed {
                operation: "CSR hot-path validation",
                reason: "row_ptrs must start at zero".into(),
            });
        }
        if self.row_ptrs.last().copied() != Some(self.values.len()) {
            return Err(SolverError::OperationFailed {
                operation: "CSR hot-path validation",
                reason: format!(
                    "terminal row pointer must equal values length {}, got {:?}",
                    self.values.len(),
                    self.row_ptrs.last()
                ),
            });
        }
        if self.row_ptrs.windows(2).any(|window| window[0] > window[1]) {
            return Err(SolverError::OperationFailed {
                operation: "CSR hot-path validation",
                reason: "row_ptrs must be monotone".into(),
            });
        }
        // NOTE: a branchless max-reduction was tried here but is slower on
        // x86-64: AVX2 has no packed u64 max, so a `usize` fold scalarises,
        // while the early-exit scan below vectorises to packed compares.
        if let Some((index, &column)) = self
            .col_indices
            .iter()
            .enumerate()
            .find(|(_, column)| **column >= self.num_cols)
        {
            return Err(SolverError::OperationFailed {
                operation: "CSR hot-path validation",
                reason: format!(
                    "column index {} at entry {} is outside {} columns",
                    column, index, self.num_cols
                ),
            });
        }
        Ok(())
    }

    /// Validate the canonical CSR representation required by factorization
    /// and triangular-preconditioner setup.
    ///
    /// [`Self::validate_structure`] deliberately permits duplicate or
    /// unsorted entries because ordinary SpMV can define those rows by
    /// accumulation. Factorization consumers need a unique ordered row
    /// representation for diagonal lookup and binary-search-based updates;
    /// they must call this stricter boundary.
    pub fn validate_canonical_structure(
        &self,
    ) -> Result<(), sonium_solvers_core::traits::SolverError> {
        self.validate_structure()?;
        for row in 0..self.num_rows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            let mut previous = None;
            for &column in &self.col_indices[start..end] {
                if let Some(previous) = previous
                    && column <= previous
                {
                    return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                        operation: "CSR canonical structure",
                        reason: format!(
                            "row {row} columns must be strictly increasing; found {previous} then {column}"
                        ),
                    });
                }
                previous = Some(column);
            }
        }
        Ok(())
    }

    /// This infallible constructor is retained for compatibility. New code
    /// accepting untrusted values should use [`Self::try_from_dense`].
    pub fn from_dense(dense: &DenseMatrix<T>, threshold: T::Real) -> Self {
        Self::try_from_dense(dense, threshold)
            .unwrap_or_else(|error| panic!("invalid dense CSR matrix: {error}"))
    }

    /// Create a CSR matrix from a dense matrix with checked input admission.
    ///
    /// Every dense value is checked before thresholding. A NaN must not
    /// disappear because `NaN > threshold` is false; callers receive a
    /// typed construction error instead.
    pub fn try_from_dense(
        dense: &DenseMatrix<T>,
        threshold: T::Real,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        if !threshold.is_finite() {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR dense construction",
                reason: format!("threshold must be finite, got {threshold:?}"),
            });
        }

        let num_rows = dense.nrows();
        let num_cols = dense.ncols();

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let row_ptr_len = num_rows.checked_add(1).ok_or_else(|| {
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR dense construction",
                reason: format!("row count {num_rows} overflows row-pointer length"),
            }
        })?;
        let mut row_ptrs = vec![0usize; row_ptr_len];

        for i in 0..num_rows {
            for j in 0..num_cols {
                let val = dense[[i, j]];
                if !val.norm().is_finite() {
                    return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                        operation: "CSR dense construction",
                        reason: format!("dense value at ({i}, {j}) is non-finite"),
                    });
                }
                if val.norm() > threshold {
                    values.push(val);
                    col_indices.push(j);
                }
            }
            row_ptrs[i + 1] = values.len();
        }

        Self::try_from_raw_parts(num_rows, num_cols, row_ptrs, col_indices, values)
    }

    /// Create a CSR matrix from COO (Coordinate) format triplets
    ///
    /// Triplets are (row, col, value). Duplicate entries are summed.
    pub fn from_triplets(
        num_rows: usize,
        num_cols: usize,
        triplets: Vec<(usize, usize, T)>,
    ) -> Self {
        Self::try_from_triplets(num_rows, num_cols, triplets)
            .unwrap_or_else(|error| panic!("invalid CSR triplets: {error}"))
    }

    /// Create a CSR matrix from COO triplets with complete validation.
    ///
    /// Rows and columns must be within the declared matrix shape and every
    /// value must be finite. Duplicate entries are summed after validation.
    pub fn try_from_triplets(
        num_rows: usize,
        num_cols: usize,
        triplets: Vec<(usize, usize, T)>,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        num_rows.checked_add(1).ok_or_else(|| {
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR triplet construction",
                reason: format!("row count {num_rows} overflows row-pointer length"),
            }
        })?;
        for (index, &(row, column, value)) in triplets.iter().enumerate() {
            if row >= num_rows {
                return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                    operation: "CSR triplet construction",
                    reason: format!("row index {row} at triplet {index} >= {num_rows}"),
                });
            }
            if column >= num_cols {
                return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                    operation: "CSR triplet construction",
                    reason: format!("column index {column} at triplet {index} >= {num_cols}"),
                });
            }
            if !value.norm().is_finite() {
                return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                    operation: "CSR triplet construction",
                    reason: format!("value at triplet {index} is non-finite: {value:?}"),
                });
            }
        }

        let matrix = Self::from_triplets_unchecked(num_rows, num_cols, triplets);
        if let Some((index, value)) = matrix
            .values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.norm().is_finite())
        {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR triplet construction",
                reason: format!(
                    "duplicate coalescing generated non-finite value at storage index {index}: {value:?}"
                ),
            });
        }
        Ok(matrix)
    }

    fn from_triplets_unchecked(
        num_rows: usize,
        num_cols: usize,
        triplets: Vec<(usize, usize, T)>,
    ) -> Self {
        if triplets.is_empty() {
            return Self::new(num_rows, num_cols);
        }

        // 1. Count non-zeros per row
        let mut row_counts = vec![0usize; num_rows];
        for (r, _, _) in &triplets {
            row_counts[*r] += 1;
        }

        // 2. Compute row pointers (prefix sums)
        let mut row_ptrs = vec![0usize; num_rows + 1];
        for i in 0..num_rows {
            row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
        }

        // 3. Scatter entries into pre-allocated arrays
        let nnz = triplets.len();
        let mut values = vec![T::zero(); nnz];
        let mut col_indices = vec![0usize; nnz];

        let mut row_current = row_ptrs.clone();
        for (r, c, v) in triplets {
            let idx = row_current[r];
            values[idx] = v;
            col_indices[idx] = c;
            row_current[r] += 1;
        }

        // 4. Sort columns per row and compactly merge duplicates.
        // Each row's sort+dedupe is independent given its slice of the
        // staged (col_indices, values) arrays — process them in parallel.
        // Phase A: produce a per-row deduped Vec<(usize, T)>.
        // Phase B: prefix-sum the row lengths into final_row_ptrs and
        //          flatten into the contiguous output arrays.
        const PARALLEL_FROM_TRIPLETS_THRESHOLD: usize = 4_096;
        let parallel = num_rows >= PARALLEL_FROM_TRIPLETS_THRESHOLD;

        let dedup_row = |i: usize| -> Vec<(usize, T)> {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            if start == end {
                return Vec::new();
            }
            let mut row_entries: Vec<(usize, T)> = col_indices[start..end]
                .iter()
                .copied()
                .zip(values[start..end].iter().copied())
                .collect();
            row_entries.sort_unstable_by_key(|&(c, _)| c);

            let mut deduped: Vec<(usize, T)> = Vec::with_capacity(row_entries.len());
            for (c, v) in row_entries {
                if let Some(last) = deduped.last_mut()
                    && last.0 == c
                {
                    last.1 += v;
                } else {
                    deduped.push((c, v));
                }
            }
            deduped
        };

        let per_row: Vec<Vec<(usize, T)>> = if parallel {
            (0..num_rows).into_par_iter().map(dedup_row).collect()
        } else {
            (0..num_rows).map(dedup_row).collect()
        };

        // Prefix-sum the per-row lengths to get final_row_ptrs.
        let mut final_row_ptrs = vec![0usize; num_rows + 1];
        for i in 0..num_rows {
            final_row_ptrs[i + 1] = final_row_ptrs[i] + per_row[i].len();
        }
        let total = final_row_ptrs[num_rows];

        let mut final_values = vec![T::zero(); total];
        let mut final_col_indices = vec![0usize; total];

        if parallel {
            // Disjoint mutable slices per row — split the output buffers
            // by the prefix-summed row pointers and fill in parallel.
            let mut val_rest = final_values.as_mut_slice();
            let mut col_rest = final_col_indices.as_mut_slice();
            let mut val_chunks: Vec<&mut [T]> = Vec::with_capacity(num_rows);
            let mut col_chunks: Vec<&mut [usize]> = Vec::with_capacity(num_rows);
            for row in &per_row {
                let len = row.len();
                let (vh, vt) = val_rest.split_at_mut(len);
                let (ch, ct) = col_rest.split_at_mut(len);
                val_chunks.push(vh);
                col_chunks.push(ch);
                val_rest = vt;
                col_rest = ct;
            }
            per_row
                .into_par_iter()
                .zip(val_chunks.into_par_iter())
                .zip(col_chunks.into_par_iter())
                .for_each(|((row, val_chunk), col_chunk)| {
                    for (k, (c, v)) in row.into_iter().enumerate() {
                        val_chunk[k] = v;
                        col_chunk[k] = c;
                    }
                });
        } else {
            let mut idx = 0;
            for row in per_row {
                for (c, v) in row {
                    final_values[idx] = v;
                    final_col_indices[idx] = c;
                    idx += 1;
                }
            }
        }

        Self {
            num_rows,
            num_cols,
            values: final_values,
            col_indices: final_col_indices,
            row_ptrs: final_row_ptrs,
        }
    }

    /// Assemble CSR directly from row-major triplets with sorted, unique
    /// columns per row.
    ///
    /// Skips the validation, per-row sort, and duplicate merge of
    /// [`Self::from_triplets`]: the caller guarantees every `(row, col)`
    /// is in bounds, columns are ascending with no repeats within each
    /// row, rows appear in non-decreasing order, and values are finite.
    /// Gustavson SpGEMM kernels satisfy this by construction (occupied-set
    /// accumulation with sorted active indices), so routing their output
    /// here avoids two validation passes plus a redundant sort.
    fn from_sorted_unique_triplets(
        num_rows: usize,
        num_cols: usize,
        triplets: Vec<(usize, usize, T)>,
    ) -> Self {
        let mut row_ptrs = vec![0usize; num_rows + 1];
        for (r, _, _) in &triplets {
            row_ptrs[*r] += 1;
        }
        let mut acc = 0usize;
        for count in row_ptrs.iter_mut() {
            let c = *count;
            *count = acc;
            acc += c;
        }
        row_ptrs[num_rows] = acc;
        let nnz = triplets.len();
        let mut values = vec![T::zero(); nnz];
        let mut col_indices = vec![0usize; nnz];
        let mut row_current = row_ptrs[..num_rows].to_vec();
        for (r, c, v) in triplets {
            let idx = row_current[r];
            values[idx] = v;
            col_indices[idx] = c;
            row_current[r] += 1;
        }
        Self {
            num_rows,
            num_cols,
            values,
            col_indices,
            row_ptrs,
        }
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Maximum distance `|row - col|` over stored entries.
    ///
    /// For square matrices this is the usual sparse matrix bandwidth metric
    /// used by locality/fill-reducing reorderings. Rectangular matrices are
    /// accepted so callers can still quantify row/column spread.
    pub fn bandwidth(&self) -> usize {
        let mut bandwidth = 0usize;
        for row in 0..self.num_rows {
            for idx in self.row_ptrs[row]..self.row_ptrs[row + 1] {
                bandwidth = bandwidth.max(row.abs_diff(self.col_indices[idx]));
            }
        }
        bandwidth
    }

    /// Compute a reverse Cuthill-McKee ordering for square CSR sparsity.
    ///
    /// The returned permutation is `new_index -> old_index`, which is the same
    /// convention used by [`CsrMatrix::symmetric_permute`]. The graph is treated
    /// as undirected using the union of `A` and `A^T`; diagonal entries do not
    /// contribute to degree.
    pub fn reverse_cuthill_mckee_ordering(&self) -> Vec<usize> {
        assert_eq!(
            self.num_rows, self.num_cols,
            "reverse Cuthill-McKee requires a square matrix"
        );

        let adjacency = self.symmetric_adjacency();
        let degrees = adjacency
            .iter()
            .map(|neighbors| neighbors.len())
            .collect::<Vec<_>>();
        let n = self.num_rows;
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut queue = VecDeque::new();

        while order.len() < n {
            let Some(start) = (0..n)
                .filter(|&node| !visited[node])
                .min_by_key(|&node| (degrees[node], node))
            else {
                break;
            };

            visited[start] = true;
            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                order.push(node);
                let mut neighbors = adjacency[node]
                    .iter()
                    .copied()
                    .filter(|&neighbor| !visited[neighbor])
                    .collect::<Vec<_>>();
                neighbors.sort_unstable_by_key(|&neighbor| (degrees[neighbor], neighbor));

                for neighbor in neighbors {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        order.reverse();
        order
    }

    /// Apply a symmetric permutation `B = P * A * P^T`.
    ///
    /// `new_to_old[i]` gives the original row/column index that should appear
    /// at new row/column `i`.
    pub fn symmetric_permute(&self, new_to_old: &[usize]) -> Self {
        assert_eq!(
            self.num_rows, self.num_cols,
            "symmetric permutation requires a square matrix"
        );
        assert_eq!(
            new_to_old.len(),
            self.num_rows,
            "permutation length must match matrix dimension"
        );

        let old_to_new = invert_permutation(new_to_old);
        let mut triplets = Vec::with_capacity(self.nnz());
        for (new_row, &old_row) in new_to_old.iter().enumerate() {
            for idx in self.row_ptrs[old_row]..self.row_ptrs[old_row + 1] {
                let old_col = self.col_indices[idx];
                let new_col = old_to_new[old_col];
                triplets.push((new_row, new_col, self.values[idx]));
            }
        }
        Self::from_triplets(self.num_rows, self.num_cols, triplets)
    }

    /// Convenience wrapper for `self.symmetric_permute(self.reverse_cuthill_mckee_ordering())`.
    pub fn reverse_cuthill_mckee_permuted(&self) -> (Self, Vec<usize>) {
        let permutation = self.reverse_cuthill_mckee_ordering();
        let reordered = self.symmetric_permute(&permutation);
        (reordered, permutation)
    }

    /// Compute a nested-dissection ordering for square CSR sparsity.
    ///
    /// The returned permutation is `new_index -> old_index`, matching
    /// [`CsrMatrix::symmetric_permute`]. This exposes the same graph
    /// partitioning hook used by the multifrontal-BLR direct solver and H-ILU
    /// setup, so callers can inspect or apply the ordering without building a
    /// factorization.
    pub fn nested_dissection_ordering(
        &self,
        min_block_size: usize,
        max_levels: Option<usize>,
    ) -> Vec<usize> {
        assert_eq!(
            self.num_rows, self.num_cols,
            "nested dissection requires a square matrix"
        );

        nested_dissection_ordering_from_adjacency(
            &build_adjacency(self),
            min_block_size,
            max_levels,
        )
    }

    /// Convenience wrapper for `self.symmetric_permute(self.nested_dissection_ordering(...))`.
    pub fn nested_dissection_permuted(
        &self,
        min_block_size: usize,
        max_levels: Option<usize>,
    ) -> (Self, Vec<usize>) {
        let permutation = self.nested_dissection_ordering(min_block_size, max_levels);
        let reordered = self.symmetric_permute(&permutation);
        (reordered, permutation)
    }

    /// Compute a block/node ordering for system matrices with fixed-size DOF blocks.
    ///
    /// The scalar sparsity is collapsed to a block graph, nested dissection is
    /// applied to that graph, and each ordered node is expanded back to its
    /// contiguous scalar DOFs. This gives FEM/system PDE callers a node-level
    /// permutation hook without needing a separate block matrix type.
    pub fn block_node_nested_dissection_ordering(
        &self,
        block_size: usize,
        min_block_size: usize,
        max_levels: Option<usize>,
    ) -> Vec<usize> {
        assert_eq!(
            self.num_rows, self.num_cols,
            "block/node ordering requires a square matrix"
        );
        assert!(block_size > 0, "block size must be nonzero");
        assert!(
            self.num_rows.is_multiple_of(block_size),
            "matrix dimension must be divisible by block size"
        );

        let block_count = self.num_rows / block_size;
        let mut adjacency = vec![Vec::new(); block_count];
        for row in 0..self.num_rows {
            let row_block = row / block_size;
            for idx in self.row_ptrs[row]..self.row_ptrs[row + 1] {
                let col_block = self.col_indices[idx] / block_size;
                if row_block != col_block {
                    adjacency[row_block].push(col_block);
                    adjacency[col_block].push(row_block);
                }
            }
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let block_permutation =
            nested_dissection_ordering_from_adjacency(&adjacency, min_block_size, max_levels);
        let mut permutation = Vec::with_capacity(self.num_rows);
        for block in block_permutation {
            let start = block * block_size;
            permutation.extend(start..start + block_size);
        }
        permutation
    }

    /// Convenience wrapper for [`CsrMatrix::block_node_nested_dissection_ordering`].
    pub fn block_node_nested_dissection_permuted(
        &self,
        block_size: usize,
        min_block_size: usize,
        max_levels: Option<usize>,
    ) -> (Self, Vec<usize>) {
        let permutation =
            self.block_node_nested_dissection_ordering(block_size, min_block_size, max_levels);
        let reordered = self.symmetric_permute(&permutation);
        (reordered, permutation)
    }

    fn symmetric_adjacency(&self) -> Vec<Vec<usize>> {
        let mut adjacency = vec![Vec::new(); self.num_rows];
        for row in 0..self.num_rows {
            for idx in self.row_ptrs[row]..self.row_ptrs[row + 1] {
                let col = self.col_indices[idx];
                if row == col {
                    continue;
                }
                adjacency[row].push(col);
                adjacency[col].push(row);
            }
        }

        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        adjacency
    }

    /// Borrow this matrix in the compute crate's CSR interchange format.
    pub fn as_host_csr_ref(&self) -> sonium_compute::HostCsrRef<'_, T> {
        sonium_compute::HostCsrRef::new(
            self.num_rows,
            self.num_cols,
            &self.row_ptrs,
            &self.col_indices,
            &self.values,
        )
    }

    /// Sparsity ratio (fraction of non-zero entries)
    pub fn sparsity(&self) -> f64 {
        let total = self.num_rows * self.num_cols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Get the range of indices in values/col_indices for a given row
    pub fn row_range(&self, row: usize) -> Range<usize> {
        self.row_ptrs[row]..self.row_ptrs[row + 1]
    }

    /// Get the (col, value) pairs for a row
    pub fn row_entries(&self, row: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let range = self.row_range(row);
        self.col_indices[range.clone()]
            .iter()
            .copied()
            .zip(self.values[range].iter().copied())
    }

    /// Matrix-vector product: y = A * x
    ///
    /// Uses parallel processing when the `rayon` feature is enabled and the
    /// matrix is large enough to benefit from parallelization.
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        // Storage is publicly mutable; the hot path re-checks the
        // memory-safety invariants (lengths, monotone row pointers,
        // in-bounds columns) without the full value scan.
        self.validate_hot_structure().expect("invalid CSR storage");
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");

        let mut y = vec![T::zero(); self.num_rows];

        // Use parallel version for large matrices when rayon is available.
        // Heuristic: parallel dispatch needs both enough total work (nnz) and
        // enough rows to distribute across threads without excessive per-row
        // overhead. The SIMD kernel is most efficient when rows have enough
        // non-zeros to amortize the gather+multiply loop setup.
        {
            let nnz = self.row_ptrs[self.num_rows];
            if nnz >= crate::spmv_plan::par_nnz_threshold()
                && self.num_rows >= crate::spmv_plan::par_min_rows()
                && sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(nnz) > 1
            {
                self.matvec_parallel_into(x, &mut y);
                return y;
            }
        }

        self.matvec_sequential_into(x, &mut y);
        y
    }

    /// Checked variant of [`Self::matvec`] that revalidates public CSR storage
    /// before any kernel or unsafe indexing is reached.
    pub fn try_matvec(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_spmv_input("CSR SpMV", x, self.num_cols)?;
        let y = self.matvec(x);
        Self::reject_non_finite_spmv_output("CSR SpMV", &y)?;
        Ok(y)
    }

    /// Checked variant of [`Self::matvec_into_buf`].
    pub fn try_matvec_into_buf(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        self.validate_checked_spmv_input("CSR SpMV", x, self.num_cols)?;
        self.validate_checked_spmv_output("CSR SpMV", y, self.num_rows)?;
        let mut staged = vec![T::zero(); self.num_rows];
        self.matvec_into_buf(x, &mut staged);
        Self::reject_non_finite_spmv_output("CSR SpMV", &staged)?;
        y.copy_from_slice(&staged);
        Ok(())
    }

    /// Sequential matrix-vector product into a pre-allocated output slice.
    ///
    /// For `f64` matrices on x86-64 (AVX2) or aarch64 (NEON), this dispatches
    /// to a SIMD-optimized gather-multiply-accumulate kernel. All other types
    /// use the generic scalar loop.
    #[inline]
    fn matvec_sequential_into(&self, x: &[T], y: &mut [T]) {
        // Compile-time type check: dispatch f64 matrices to the SIMD path.
        // TypeId comparison is a single integer compare that the compiler can
        // often constant-fold when T is concrete.
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // AVX2 gather intrinsics use i32 indices; skip SIMD for huge matrices.
            if self.num_cols <= i32::MAX as usize {
                // SAFETY: We just verified T == f64. The pointer casts are
                // transmuting &[T] <-> &[f64] which is layout-identical.
                unsafe {
                    let x_f64 = &*(x as *const [T] as *const [f64]);
                    let y_f64 = &mut *(y as *mut [T] as *mut [f64]);
                    let values_f64 = &*(self.values.as_slice() as *const [T] as *const [f64]);
                    spmv_f64_simd(
                        values_f64,
                        self.col_indices.as_slice(),
                        self.row_ptrs.as_slice(),
                        x_f64,
                        y_f64,
                        self.num_rows,
                        self.num_cols <= i32::MAX as usize,
                    );
                }
                return;
            }
        }

        if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            // SAFETY: We just verified T == Complex64. Complex64 is
            // `Complex<f64>` with layout `(f64, f64)`, so the cast is valid.
            unsafe {
                let x_c64 = &*(x as *const [T] as *const [Complex64]);
                let y_c64 = &mut *(y as *mut [T] as *mut [Complex64]);
                let values_c64 = &*(self.values.as_slice() as *const [T] as *const [Complex64]);
                spmv_complex64_simd(
                    values_c64,
                    self.col_indices.as_slice(),
                    self.row_ptrs.as_slice(),
                    x_c64,
                    y_c64,
                    self.num_rows,
                );
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // AVX2 gather intrinsics use i32 indices; skip SIMD for huge matrices.
            if self.num_cols <= i32::MAX as usize {
                // SAFETY: We just verified T == f32. The pointer cast is valid.
                unsafe {
                    let x_f32 = &*(x as *const [T] as *const [f32]);
                    let y_f32 = &mut *(y as *mut [T] as *mut [f32]);
                    let values_f32 = &*(self.values.as_slice() as *const [T] as *const [f32]);
                    spmv_f32_simd(
                        values_f32,
                        self.col_indices.as_slice(),
                        self.row_ptrs.as_slice(),
                        x_f32,
                        y_f32,
                        self.num_rows,
                    );
                }
                return;
            }
        }

        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            // SAFETY: We just verified T == Complex32. Complex32 is
            // `Complex<f32>` with layout `(f32, f32)`, so the cast is valid.
            unsafe {
                let x_c32 = &*(x as *const [T] as *const [Complex32]);
                let y_c32 = &mut *(y as *mut [T] as *mut [Complex32]);
                let values_c32 = &*(self.values.as_slice() as *const [T] as *const [Complex32]);
                spmv_complex32_simd(
                    values_c32,
                    self.col_indices.as_slice(),
                    self.row_ptrs.as_slice(),
                    x_c32,
                    y_c32,
                    self.num_rows,
                );
            }
            return;
        }

        // Generic scalar fallback
        let values = self.values.as_slice();
        let col_indices = self.col_indices.as_slice();
        let row_ptrs = self.row_ptrs.as_slice();

        for i in 0..self.num_rows {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            let mut sum = T::zero();
            for idx in start..end {
                // SAFETY: col_indices[idx] is guaranteed < num_cols by CSR construction,
                // and x.len() == num_cols (checked in matvec). Use unchecked for hot loop.
                unsafe {
                    let j = *col_indices.get_unchecked(idx);
                    sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
                }
            }
            y[i] = sum;
        }
    }

    /// Parallel matrix-vector product into a pre-allocated output slice.
    ///
    /// Rows are partitioned into contiguous ranges with ≈ equal nonzero
    /// counts (one per rayon thread, via
    /// [`partition::nnz_balanced_ranges`](crate::partition::nnz_balanced_ranges)),
    /// so skewed matrices do not leave a straggler thread holding most of
    /// the non-zeros. For `f64` matrices each task uses SIMD (AVX2/NEON)
    /// per row. Runs on the bandwidth-limited thread pool for small
    /// matrices and on the wider compute pool for large matrices (see
    /// [`runtime::bandwidth_threads_for_nnz`](sonium_solvers_sys::runtime::bandwidth_threads_for_nnz)).
    fn matvec_parallel_into(&self, x: &[T], y: &mut [T])
    where
        T: Send + Sync,
    {
        let values = &self.values;
        let col_indices = &self.col_indices;
        let row_ptrs = &self.row_ptrs;
        let nnz = row_ptrs[self.num_rows];

        sonium_solvers_sys::runtime::with_bandwidth_pool_for_nnz(nnz, || {
            let nthreads = rayon::current_num_threads().max(1);
            let ranges = nnz_balanced_ranges(row_ptrs.as_slice(), nthreads);

            // Split `y` into one contiguous chunk per nnz-balanced row range
            // so each rayon task writes a disjoint slice of the output.
            let mut y_rest = &mut *y;
            let mut prev_end = 0usize;
            let mut y_chunks: Vec<&mut [T]> = Vec::with_capacity(ranges.len());
            for range in &ranges {
                let (head, tail) = y_rest.split_at_mut(range.end - prev_end);
                prev_end = range.end;
                y_chunks.push(head);
                y_rest = tail;
            }

            // SIMD path for f64: each rayon task computes its nnz-balanced
            // row range using the SIMD-optimised gather-multiply-accumulate
            // kernel per row.
            if TypeId::of::<T>() == TypeId::of::<f64>() && self.num_cols <= i32::MAX as usize {
                // SAFETY: We verified T == f64. The pointer casts are
                // transmuting &[T] <-> &[f64] which is layout-identical.
                // Each chunk is a disjoint &mut [f64], so there are no data
                // races between tasks.
                let values_f64 = unsafe { &*(values.as_slice() as *const [T] as *const [f64]) };
                let x_f64 = unsafe { &*(x as *const [T] as *const [f64]) };
                let y_chunks_f64: Vec<&mut [f64]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [f64]) })
                    .collect();

                y_chunks_f64
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i];
                            let end = row_ptrs[i + 1];
                            // SAFETY: CSR invariants guarantee valid indices
                            *yi = unsafe {
                                spmv_row_f64(
                                    values_f64,
                                    col_indices,
                                    x_f64,
                                    start,
                                    end,
                                    self.num_cols <= i32::MAX as usize,
                                )
                            };
                        }
                    });
                return;
            }

            // SIMD path for Complex64
            if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                let values_c64 =
                    unsafe { &*(values.as_slice() as *const [T] as *const [Complex64]) };
                let x_c64 = unsafe { &*(x as *const [T] as *const [Complex64]) };
                let y_chunks_c64: Vec<&mut [Complex64]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [Complex64]) })
                    .collect();

                y_chunks_c64
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i];
                            let end = row_ptrs[i + 1];
                            *yi = unsafe {
                                spmv_row_complex64(values_c64, col_indices, x_c64, start, end)
                            };
                        }
                    });
                return;
            }

            // SIMD path for f32
            if TypeId::of::<T>() == TypeId::of::<f32>() && self.num_cols <= i32::MAX as usize {
                let values_f32 = unsafe { &*(values.as_slice() as *const [T] as *const [f32]) };
                let x_f32 = unsafe { &*(x as *const [T] as *const [f32]) };
                let y_chunks_f32: Vec<&mut [f32]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [f32]) })
                    .collect();

                y_chunks_f32
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i];
                            let end = row_ptrs[i + 1];
                            *yi =
                                unsafe { spmv_row_f32(values_f32, col_indices, x_f32, start, end) };
                        }
                    });
                return;
            }

            // SIMD path for Complex32
            if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                let values_c32 =
                    unsafe { &*(values.as_slice() as *const [T] as *const [Complex32]) };
                let x_c32 = unsafe { &*(x as *const [T] as *const [Complex32]) };
                let y_chunks_c32: Vec<&mut [Complex32]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [Complex32]) })
                    .collect();

                y_chunks_c32
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i];
                            let end = row_ptrs[i + 1];
                            *yi = unsafe {
                                spmv_row_complex32(values_c32, col_indices, x_c32, start, end)
                            };
                        }
                    });
                return;
            }

            // Generic scalar fallback for all other types
            y_chunks
                .into_par_iter()
                .zip(ranges.par_iter())
                .for_each(|(chunk, range)| {
                    for (offset, yi) in chunk.iter_mut().enumerate() {
                        let i = range.start + offset;
                        let start = row_ptrs[i];
                        let end = row_ptrs[i + 1];
                        let mut sum = T::zero();
                        for idx in start..end {
                            unsafe {
                                let j = *col_indices.get_unchecked(idx);
                                sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
                            }
                        }
                        *yi = sum;
                    }
                });
        });
    }

    /// Matrix-vector product into pre-allocated buffer: y = A * x
    ///
    /// Overwrites `y` with the result. No allocation.
    /// Uses the same parallel/SIMD dispatch as [`matvec`](Self::matvec).
    #[inline]
    pub fn matvec_into_buf(&self, x: &[T], y: &mut [T]) {
        // Storage is publicly mutable; the hot path re-checks the
        // memory-safety invariants (lengths, monotone row pointers,
        // in-bounds columns) without the full value scan. Non-finite
        // values propagate arithmetically instead of panicking (pinned by
        // `hot_validation_accepts_non_finite_values`); full
        // `validate_structure` still rejects them for checked boundaries.
        self.validate_hot_structure().expect("invalid CSR storage");
        // SAFETY: the hot check established the memory-safety contract
        // below (lengths, monotonicity, in-bounds columns); values only
        // affect arithmetic, never indexing.
        unsafe { self.matvec_into_buf_unchecked(x, y) };
    }

    /// SpMV into a buffer without structural validation.
    ///
    /// # Safety
    ///
    /// The caller guarantees the memory-safety invariants (monotone row
    /// pointers rooted at zero with terminal count equal to the value
    /// count, in-bounds column indices; `x`/`y` matching
    /// `num_cols`/`num_rows`) — e.g. via [`Self::validate_hot_structure`]
    /// or the stronger [`Self::validate_structure`]. Violating this is
    /// immediate undefined
    /// behavior: the kernels below use unchecked indexing throughout.
    /// Only call this on matrices the caller owns and never mutates after
    /// a validating construction (e.g. hierarchy operators built once at
    /// AMG setup); user-supplied matrices must go through the checked
    /// entry points. Non-finite *values* remain the caller's problem and
    /// are still rejected at API boundaries.
    pub unsafe fn matvec_into_buf_unchecked(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_rows, "Output vector size mismatch");

        {
            let nnz = self.row_ptrs[self.num_rows];
            if nnz >= crate::spmv_plan::par_nnz_threshold()
                && self.num_rows >= crate::spmv_plan::par_min_rows()
                && sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(nnz) > 1
            {
                self.matvec_parallel_into(x, y);
                return;
            }
        }

        self.matvec_sequential_into(x, y);
    }

    /// Matrix-vector product with accumulation: y += A * x
    #[inline]
    pub fn matvec_add(&self, x: &[T], y: &mut [T]) {
        // Storage is publicly mutable; the hot path re-checks the
        // memory-safety invariants (lengths, monotone row pointers,
        // in-bounds columns) without the full value scan.
        self.validate_hot_structure().expect("invalid CSR storage");
        // SAFETY: the hot check established the memory-safety contract
        // below (lengths, monotonicity, in-bounds columns); values only
        // affect arithmetic, never indexing.
        unsafe { self.matvec_add_unchecked(x, y) };
    }

    /// Accumulating SpMV without structural validation: y += A * x.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::matvec_into_buf_unchecked`]: the caller
    /// guarantees monotone row pointers rooted at zero with terminal count
    /// equal to the value count, in-bounds column indices, and
    /// `x`/`y` matching `num_cols`/`num_rows`. Only call this on matrices
    /// the caller owns and never mutates after a validating construction;
    /// user-supplied matrices must go through the checked entry points.
    pub unsafe fn matvec_add_unchecked(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_rows, "Output vector size mismatch");

        let values = self.values.as_slice();
        let col_indices = self.col_indices.as_slice();
        let row_ptrs = self.row_ptrs.as_slice();

        for i in 0..self.num_rows {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            let mut sum = T::zero();
            for idx in start..end {
                // SAFETY: upheld by the caller per the method contract:
                // `idx` is inside a validated row range, `j` is an
                // in-bounds column, and `x`/`y` have the required lengths.
                unsafe {
                    let j = *col_indices.get_unchecked(idx);
                    sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
                }
            }
            y[i] += sum;
        }
    }

    /// Checked variant of [`Self::matvec_add`].
    pub fn try_matvec_add(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        self.validate_checked_spmv_input("CSR SpMV add", x, self.num_cols)?;
        self.validate_checked_spmv_output("CSR SpMV add", y, self.num_rows)?;

        // Compute the contribution through the same runtime-dispatched
        // sequential/parallel SIMD path as ordinary SpMV. Accumulation is
        // then published transactionally so generated non-finite values leave
        // the caller-owned destination unchanged.
        let mut contribution = vec![T::zero(); self.num_rows];
        self.matvec_into_buf(x, &mut contribution);
        Self::reject_non_finite_spmv_output("CSR SpMV add", &contribution)?;

        let mut staged = y.to_vec();
        for (destination, value) in staged.iter_mut().zip(contribution) {
            *destination += value;
        }
        Self::reject_non_finite_spmv_output("CSR SpMV add", &staged)?;
        y.copy_from_slice(&staged);
        Ok(())
    }

    /// Transpose matrix-vector product: y = A^T * x
    pub fn matvec_transpose(&self, x: &[T]) -> Vec<T> {
        // Validated by `matvec_transpose_into_buf` below; no redundant scan here.
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");

        let mut y = vec![T::zero(); self.num_cols];
        self.matvec_transpose_into_buf(x, &mut y);
        y
    }

    /// Checked variant of [`Self::matvec_transpose`].
    pub fn try_matvec_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_spmv_input("CSR transpose SpMV", x, self.num_rows)?;
        let y = self.matvec_transpose(x);
        Self::reject_non_finite_spmv_output("CSR transpose SpMV", &y)?;
        Ok(y)
    }

    /// Transpose matrix-vector product into pre-allocated buffer: y = A^T * x
    pub fn matvec_transpose_into_buf(&self, x: &[T], y: &mut [T]) {
        // Storage is publicly mutable; the hot path re-checks the
        // memory-safety invariants (lengths, monotone row pointers,
        // in-bounds columns) without the full value scan.
        self.validate_hot_structure().expect("invalid CSR storage");
        // SAFETY: the hot check established the memory-safety contract
        // below (lengths, monotonicity, in-bounds columns); values only
        // affect arithmetic, never indexing.
        unsafe { self.matvec_transpose_into_buf_unchecked(x, y) };
    }

    /// Transpose SpMV into a buffer without structural validation.
    ///
    /// # Safety
    ///
    /// The caller guarantees the memory-safety invariants (monotone row
    /// pointers rooted at zero with terminal count equal to the value
    /// count, in-bounds column indices; `x`/`y` matching
    /// `num_rows`/`num_cols`) — e.g. via [`Self::validate_hot_structure`]
    /// or the stronger [`Self::validate_structure`]. Only call this on
    /// matrices the caller owns and never mutates after a validating
    /// construction; user-supplied matrices must go through the checked
    /// entry points.
    pub unsafe fn matvec_transpose_into_buf_unchecked(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_cols, "Output vector size mismatch");

        if self.should_parallelize_transpose() {
            self.matvec_transpose_parallel_into::<false>(x, y);
        } else {
            self.matvec_transpose_sequential_into::<false>(x, y);
        }
    }

    /// Checked variant of [`Self::matvec_transpose_into_buf`].
    pub fn try_matvec_transpose_into_buf(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        self.validate_checked_spmv_input("CSR transpose SpMV", x, self.num_rows)?;
        self.validate_checked_spmv_output("CSR transpose SpMV", y, self.num_cols)?;
        let mut staged = vec![T::zero(); self.num_cols];
        self.matvec_transpose_into_buf(x, &mut staged);
        Self::reject_non_finite_spmv_output("CSR transpose SpMV", &staged)?;
        y.copy_from_slice(&staged);
        Ok(())
    }

    /// Hermitian (conjugate transpose) matrix-vector product: y = A^H * x
    pub fn matvec_hermitian(&self, x: &[T]) -> Vec<T> {
        // Validated by `matvec_hermitian_into_buf` below; no redundant scan here.
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");

        let mut y = vec![T::zero(); self.num_cols];
        self.matvec_hermitian_into_buf(x, &mut y);
        y
    }

    /// Checked variant of [`Self::matvec_hermitian`].
    pub fn try_matvec_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_spmv_input("CSR Hermitian SpMV", x, self.num_rows)?;
        let y = self.matvec_hermitian(x);
        Self::reject_non_finite_spmv_output("CSR Hermitian SpMV", &y)?;
        Ok(y)
    }

    /// Hermitian matrix-vector product into pre-allocated buffer: y = A^H * x
    pub fn matvec_hermitian_into_buf(&self, x: &[T], y: &mut [T]) {
        // Storage is publicly mutable; the hot path re-checks the
        // memory-safety invariants (lengths, monotone row pointers,
        // in-bounds columns) without the full value scan.
        self.validate_hot_structure().expect("invalid CSR storage");
        // SAFETY: the hot check established the memory-safety contract
        // below (lengths, monotonicity, in-bounds columns); values only
        // affect arithmetic, never indexing.
        unsafe { self.matvec_hermitian_into_buf_unchecked(x, y) };
    }

    /// Hermitian SpMV into a buffer without structural validation.
    ///
    /// # Safety
    ///
    /// The caller guarantees the memory-safety invariants (monotone row
    /// pointers rooted at zero with terminal count equal to the value
    /// count, in-bounds column indices; `x`/`y` matching
    /// `num_rows`/`num_cols`) — e.g. via [`Self::validate_hot_structure`]
    /// or the stronger [`Self::validate_structure`]. Only call this on
    /// matrices the caller owns and never mutates after a validating
    /// construction; user-supplied matrices must go through the checked
    /// entry points.
    pub unsafe fn matvec_hermitian_into_buf_unchecked(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_cols, "Output vector size mismatch");

        if self.should_parallelize_transpose() {
            self.matvec_transpose_parallel_into::<true>(x, y);
        } else {
            self.matvec_transpose_sequential_into::<true>(x, y);
        }
    }

    /// Checked variant of [`Self::matvec_hermitian_into_buf`].
    pub fn try_matvec_hermitian_into_buf(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        self.validate_checked_spmv_input("CSR Hermitian SpMV", x, self.num_rows)?;
        self.validate_checked_spmv_output("CSR Hermitian SpMV", y, self.num_cols)?;
        let mut staged = vec![T::zero(); self.num_cols];
        self.matvec_hermitian_into_buf(x, &mut staged);
        Self::reject_non_finite_spmv_output("CSR Hermitian SpMV", &staged)?;
        y.copy_from_slice(&staged);
        Ok(())
    }

    fn validate_checked_spmv_input(
        &self,
        operation: &'static str,
        x: &[T],
        expected: usize,
    ) -> Result<(), SolverError> {
        self.validate_structure()?;
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

    fn validate_checked_spmv_output(
        &self,
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

    fn reject_non_finite_spmv_output(
        operation: &'static str,
        values: &[T],
    ) -> Result<(), SolverError> {
        if values.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "computed output contains NaN or infinity".into(),
            });
        }
        Ok(())
    }

    /// Sparse matrix–multi-vector product: `Y = A * X`.
    ///
    /// `x` is `num_cols × k` and `y` is `num_rows × k`, both row-major with
    /// one right-hand side per column (the same layout as
    /// [`LinearOperator::apply_block`]). Each row of `A` is walked once per
    /// call: the row's non-zeros are FMA'd into `k` accumulators reading the
    /// *contiguous* length-`k` rows of `x`, so the matrix values and column
    /// indices stream from memory exactly once for all `k` right-hand sides
    /// (vs `k` full matrix walks for `k` separate [`matvec`](Self::matvec)
    /// calls — the bandwidth win that motivates SpMM).
    ///
    /// For `f64` and `Complex64` with `k <= 8` the row kernel is dispatched
    /// to a const-`K` instantiation so the accumulator loop is fully unrolled
    /// and auto-vectorised (the multi-vector access pattern is contiguous, so
    /// the gather-based `spmv_row_*` SIMD kernels do not apply here). All
    /// other types, and `k > 8`, use the generic runtime-`k` loop. Uses the
    /// same nnz-balanced rayon partitioning as
    /// [`matvec_parallel_into`](Self::matvec) for large matrices.
    pub fn spmm_into(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        // Storage is publicly mutable; the hot path re-checks the
        // memory-safety invariants (lengths, monotone row pointers,
        // in-bounds columns) without the full value scan.
        self.validate_hot_structure().expect("invalid CSR storage");
        // SAFETY: the hot check established the memory-safety contract
        // below (lengths, monotonicity, in-bounds columns); values only
        // affect arithmetic, never indexing.
        unsafe { self.spmm_into_unchecked(x, y) };
    }

    /// Sparse matrix–multi-vector product without structural validation.
    ///
    /// Same layout contract as [`Self::spmm_into`]: `x` is `num_cols × k`
    /// and `y` is `num_rows × k`, both row-major.
    ///
    /// # Safety
    ///
    /// The caller guarantees the memory-safety invariants (monotone row
    /// pointers rooted at zero with terminal count equal to the value
    /// count, in-bounds column indices; `x`/`y` dimensions as above) —
    /// e.g. via [`Self::validate_hot_structure`] or the stronger
    /// [`Self::validate_structure`]. Only call this on matrices the caller
    /// owns and never mutates after a validating construction;
    /// user-supplied matrices must go through the checked entry points.
    pub unsafe fn spmm_into_unchecked(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        assert_eq!(x.nrows(), self.num_cols, "SpMM input row mismatch");
        assert_eq!(y.nrows(), self.num_rows, "SpMM output row mismatch");
        assert_eq!(x.ncols(), y.ncols(), "SpMM column count mismatch");
        let k = x.ncols();
        if k == 0 || self.num_rows == 0 {
            return;
        }

        let nnz = self.row_ptrs[self.num_rows];
        // The total work scales with k, so the parallel threshold is checked
        // against nnz·k; the row budget still has to be large enough to split.
        if nnz.saturating_mul(k) >= crate::spmv_plan::par_nnz_threshold()
            && self.num_rows >= crate::spmv_plan::par_min_rows()
            && sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(nnz) > 1
        {
            self.spmm_parallel_into(x.as_slice(), y.as_mut_slice(), k);
            return;
        }

        self.spmm_sequential_into(x.as_slice(), y.as_mut_slice(), 0, self.num_rows, k);
    }

    /// Checked sparse matrix-matrix multiplication into caller-owned storage.
    ///
    /// The native [`Self::spmm_into`] kernel retains its infallible shape
    /// assertions for compatibility. This boundary validates the CSR
    /// structure and block input before dispatch, stages the complete result,
    /// and publishes it only after all generated values are finite. A malformed
    /// or overflowing block therefore leaves `y` unchanged.
    pub fn try_spmm_into(
        &self,
        x: &DenseMatrix<T>,
        y: &mut DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        self.validate_structure()?;
        if x.nrows() != self.num_cols {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols,
                got: x.nrows(),
            });
        }
        if y.nrows() != self.num_rows {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows,
                got: y.nrows(),
            });
        }
        if x.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: x.ncols(),
                got: y.ncols(),
            });
        }
        if x.as_slice().iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "CSR block SpMV",
                reason: "input block contains NaN or infinity".into(),
            });
        }

        let mut staged = DenseMatrix::zeros(self.num_rows, x.ncols());
        catch_unwind(AssertUnwindSafe(|| self.spmm_into(x, &mut staged))).map_err(|_| {
            SolverError::OperationFailed {
                operation: "CSR block SpMV",
                reason: "SpMM kernel panicked".into(),
            }
        })?;
        Self::reject_non_finite_spmv_output("CSR block SpMV", staged.as_slice())?;
        y.as_mut_slice().copy_from_slice(staged.as_slice());
        Ok(())
    }

    /// Sequential SpMM over rows `row_start..row_end` (see
    /// [`spmm_into`](Self::spmm_into) for the layout contract).
    fn spmm_sequential_into(
        &self,
        x: &[T],
        y: &mut [T],
        row_start: usize,
        row_end: usize,
        k: usize,
    ) {
        let values = self.values.as_slice();
        let col_indices = self.col_indices.as_slice();
        let row_ptrs = self.row_ptrs.as_slice();

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: We just verified T == f64. The pointer casts are
            // transmuting &[T] <-> &[f64] which is layout-identical.
            unsafe {
                let x_f64 = &*(x as *const [T] as *const [f64]);
                let y_f64 = &mut *(y as *mut [T] as *mut [f64]);
                let values_f64 = &*(values as *const [T] as *const [f64]);
                spmm_dispatch_f64(
                    values_f64,
                    col_indices,
                    row_ptrs,
                    x_f64,
                    y_f64,
                    row_start,
                    row_end,
                    k,
                );
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            // SAFETY: We just verified T == Complex64. Complex64 is
            // `Complex<f64>` with layout `(f64, f64)`, so the cast is valid.
            unsafe {
                let x_c64 = &*(x as *const [T] as *const [Complex64]);
                let y_c64 = &mut *(y as *mut [T] as *mut [Complex64]);
                let values_c64 = &*(values as *const [T] as *const [Complex64]);
                spmm_dispatch_complex64(
                    values_c64,
                    col_indices,
                    row_ptrs,
                    x_c64,
                    y_c64,
                    row_start,
                    row_end,
                    k,
                );
            }
            return;
        }

        // Generic scalar fallback (all other types).
        let mut acc = vec![T::zero(); k];
        spmm_rows_generic(
            values,
            col_indices,
            row_ptrs,
            x,
            y,
            row_start,
            row_end,
            k,
            &mut acc,
        );
    }

    /// Parallel SpMM: rows are partitioned into contiguous ranges with
    /// ≈ equal nonzero counts (one per rayon thread, via
    /// [`partition::nnz_balanced_ranges`]), each task walking its rows with
    /// the sequential kernel. Row-major `y` means a contiguous row range is
    /// a contiguous output slice, so tasks write disjoint memory.
    fn spmm_parallel_into(&self, x: &[T], y: &mut [T], k: usize) {
        let row_ptrs = &self.row_ptrs;
        let nnz = row_ptrs[self.num_rows];

        sonium_solvers_sys::runtime::with_bandwidth_pool_for_nnz(nnz, || {
            let nthreads = rayon::current_num_threads().max(1);
            let ranges = nnz_balanced_ranges(row_ptrs.as_slice(), nthreads);

            // Split `y` into one contiguous chunk per nnz-balanced row range
            // (a row range covers `rows·k` contiguous elements).
            let mut y_rest = &mut *y;
            let mut prev_end = 0usize;
            let mut y_chunks: Vec<&mut [T]> = Vec::with_capacity(ranges.len());
            for range in &ranges {
                let (head, tail) = y_rest.split_at_mut((range.end - prev_end) * k);
                prev_end = range.end;
                y_chunks.push(head);
                y_rest = tail;
            }

            y_chunks
                .into_par_iter()
                .zip(ranges.par_iter())
                .for_each(|(chunk, range)| {
                    // Each chunk starts at row `range.start`, so index it as a
                    // local slice with the global row offset applied here.
                    self.spmm_sequential_into(x, chunk, range.start, range.end, k);
                });
        });
    }

    fn should_parallelize_transpose(&self) -> bool {
        let nnz = self.row_ptrs[self.num_rows];
        if nnz < crate::spmv_plan::par_nnz_threshold()
            || sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(nnz) <= 1
        {
            return false;
        }

        let nthreads = rayon::current_num_threads().max(1);
        self.num_cols
            .checked_mul(nthreads)
            .is_some_and(|workspace| workspace <= TRANSPOSE_PAR_WORKSPACE_ELEMS_LIMIT)
    }

    fn matvec_transpose_sequential_into<const CONJ: bool>(&self, x: &[T], y: &mut [T]) {
        y.fill(T::zero());

        for (i, &xi) in x.iter().enumerate().take(self.num_rows) {
            for idx in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let j = self.col_indices[idx];
                let value = if CONJ {
                    self.values[idx].conj()
                } else {
                    self.values[idx]
                };
                y[j] += value * xi;
            }
        }
    }

    /// Race-safety: CSR transpose is a scatter (`y[col] += ...`), so naive
    /// row-parallelism would race. This instead gives every row-band its own
    /// `num_cols` partial buffer (two-phase reduction) and combines them in
    /// a second parallel pass over disjoint `y` lanes — no two threads ever
    /// write the same element concurrently.
    fn matvec_transpose_parallel_into<const CONJ: bool>(&self, x: &[T], y: &mut [T]) {
        let nnz = self.row_ptrs[self.num_rows];

        sonium_solvers_sys::runtime::with_bandwidth_pool_for_nnz(nnz, || {
            let nthreads = rayon::current_num_threads().max(1);
            let ranges = nnz_balanced_ranges(self.row_ptrs.as_slice(), nthreads);
            let partials: Vec<Vec<T>> = ranges
                .par_iter()
                .map(|range| {
                    let mut local = vec![T::zero(); self.num_cols];

                    for (offset, &xi) in x[range.clone()].iter().enumerate() {
                        let i = range.start + offset;
                        for idx in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                            let j = self.col_indices[idx];
                            let value = if CONJ {
                                self.values[idx].conj()
                            } else {
                                self.values[idx]
                            };
                            local[j] += value * xi;
                        }
                    }

                    local
                })
                .collect();

            y.par_iter_mut().enumerate().for_each(|(j, yj)| {
                let mut sum = T::zero();
                for partial in &partials {
                    sum += partial[j];
                }
                *yj = sum;
            });
        });
    }

    /// Get element at (i, j), returns 0 if not stored
    pub fn get(&self, i: usize, j: usize) -> T {
        for idx in self.row_range(i) {
            if self.col_indices[idx] == j {
                return self.values[idx];
            }
        }
        T::zero()
    }

    /// Extract diagonal elements
    pub fn diagonal(&self) -> Vec<T> {
        let n = self.num_rows.min(self.num_cols);
        let mut diag = vec![T::zero(); n];

        for (i, diagonal) in diag.iter_mut().enumerate() {
            for idx in self.row_range(i) {
                if self.col_indices[idx] == i {
                    *diagonal = self.values[idx];
                    break;
                }
            }
        }

        diag
    }

    /// Scale all values by a scalar.
    ///
    /// This compatibility wrapper preserves the infallible API and panics
    /// when the checked operation rejects malformed or non-finite state.
    pub fn scale(&mut self, scalar: T) {
        self.try_scale(scalar)
            .unwrap_or_else(|error| panic!("invalid CSR scale: {error}"));
    }

    /// Scale all values by a scalar after validating the complete update.
    ///
    /// The result is staged before publication, so an invalid scalar or an
    /// overflow-generated value leaves the matrix unchanged.
    pub fn try_scale(&mut self, scalar: T) -> Result<(), SolverError> {
        self.validate_structure()?;
        if !scalar.norm().is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "CSR scale",
                reason: "scale scalar must be finite".into(),
            });
        }

        let mut updated = Vec::with_capacity(self.values.len());
        for (index, &value) in self.values.iter().enumerate() {
            let scaled = value * scalar;
            if !scaled.norm().is_finite() {
                return Err(SolverError::OperationFailed {
                    operation: "CSR scale",
                    reason: format!("scaled value at entry {index} is non-finite"),
                });
            }
            updated.push(scaled);
        }
        self.values = updated;
        Ok(())
    }

    /// Add a scalar to the diagonal.
    ///
    /// This compatibility wrapper preserves the infallible API and panics
    /// when the checked operation rejects malformed or non-finite state.
    pub fn add_diagonal(&mut self, scalar: T) {
        self.try_add_diagonal(scalar)
            .unwrap_or_else(|error| panic!("invalid CSR diagonal update: {error}"));
    }

    /// Add a scalar to each stored diagonal entry after validating the update.
    ///
    /// As with the legacy method, rows without a stored diagonal are left
    /// unchanged. The complete result is staged before publication, so an
    /// invalid scalar or an overflow-generated value leaves the matrix intact.
    pub fn try_add_diagonal(&mut self, scalar: T) -> Result<(), SolverError> {
        self.validate_structure()?;
        if !scalar.norm().is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "CSR add_diagonal",
                reason: "diagonal scalar must be finite".into(),
            });
        }

        let mut updated = self.values.clone();
        let n = self.num_rows.min(self.num_cols);
        for i in 0..n {
            for idx in self.row_range(i) {
                if self.col_indices[idx] == i {
                    let diagonal = self.values[idx] + scalar;
                    if !diagonal.norm().is_finite() {
                        return Err(SolverError::OperationFailed {
                            operation: "CSR add_diagonal",
                            reason: format!("diagonal value at row {i} is non-finite"),
                        });
                    }
                    updated[idx] = diagonal;
                    break;
                }
            }
        }
        self.values = updated;
        Ok(())
    }

    /// Create identity matrix in CSR format
    pub fn identity(n: usize) -> Self {
        Self {
            num_rows: n,
            num_cols: n,
            values: vec![T::one(); n],
            col_indices: (0..n).collect(),
            row_ptrs: (0..=n).collect(),
        }
    }

    /// Create diagonal matrix from vector
    pub fn from_diagonal(diag: &[T]) -> Self {
        let n = diag.len();
        Self {
            num_rows: n,
            num_cols: n,
            values: diag.to_vec(),
            col_indices: (0..n).collect(),
            row_ptrs: (0..=n).collect(),
        }
    }

    /// Convert to dense matrix (for debugging/small matrices)
    pub fn to_dense(&self) -> DenseMatrix<T> {
        let mut dense = DenseMatrix::from_elem(self.num_rows, self.num_cols, T::zero());

        for i in 0..self.num_rows {
            for idx in self.row_range(i) {
                let j = self.col_indices[idx];
                dense[[i, j]] = self.values[idx];
            }
        }

        dense
    }

    /// Cast the matrix to a different precision level.
    ///
    /// The sparsity structure (row_ptrs, col_indices) is preserved; only the
    /// values are converted via [`PrecisionCast`].
    ///
    /// This is a raw per-value cast with **no range scaling**: casting into
    /// a narrow-range type (f16, usable range ~[6e-5, 65504]) silently
    /// overflows entries to ±inf or underflows them to zero. That is fine
    /// for inspecting values, but a matrix destined for a low-precision
    /// *factorization* (GMRES-IR) should go through the equilibrating cast
    /// in `crate::mixed_precision` instead, which scales the matrix into
    /// the target range first.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let matrix_f64: CsrMatrix<f64> = /* ... */;
    /// let matrix_f32: CsrMatrix<f32> = matrix_f64.cast::<f32>();
    /// ```
    pub fn cast<TLow>(&self) -> CsrMatrix<TLow>
    where
        T: PrecisionCast<TLow>,
        TLow: ComplexField,
    {
        let values_low: Vec<TLow> = self.values.iter().map(|&v| T::cast_down(v)).collect();
        CsrMatrix {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            values: values_low,
            col_indices: self.col_indices.clone(),
            row_ptrs: self.row_ptrs.clone(),
        }
    }
}

// =============================================================================
// SIMD-optimized SpMV for f64
// =============================================================================
//
// The gather-multiply-accumulate pattern `sum += values[k] * x[col_indices[k]]`
// is the hot loop in SpMV. For f64:
//
// - x86-64 AVX2: process 4 f64s at once with `_mm256_i32gather_pd` for the
//   gathered x[] loads. Indices are narrowed from usize to i32 which is safe
//   for matrices with < 2^31 columns.
//
// - aarch64 NEON: process 2 f64s at once. NEON lacks gather instructions so
//   x[] values are loaded individually and combined into a vector register.
//   FMA via `vfmaq_f64` is used when available.
//
// - Scalar fallback: identical to the generic loop, included for platforms
//   without SIMD support.

// ---------------------------------------------------------------------------
// SpMM kernels (multi-vector products, row-major X/Y with k RHS columns)
//
// Unlike the gather-based SpMV row kernels below, the multi-vector access
// `x[j·k .. j·k + k]` is contiguous, so the win comes from walking each
// matrix row once and FMA-ing into k accumulators. The const-K
// instantiations let LLVM unroll and vectorise the accumulator loop (NEON
// on aarch64, AVX2+FMA on x86_64) without per-call gather setup.
// ---------------------------------------------------------------------------

/// Generic SpMM row kernel for any `ComplexField` (runtime `k`).
///
/// `y` is the local output slice for rows `row_start..row_end`.
#[allow(clippy::too_many_arguments)]
fn spmm_rows_generic<T: ComplexField>(
    values: &[T],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[T],
    y: &mut [T],
    row_start: usize,
    row_end: usize,
    k: usize,
    acc: &mut [T],
) {
    debug_assert_eq!(acc.len(), k);
    debug_assert!(y.len() >= (row_end - row_start) * k);
    for (yi, row) in y.chunks_exact_mut(k).zip(row_start..row_end) {
        for a in acc.iter_mut() {
            *a = T::zero();
        }
        for idx in row_ptrs[row]..row_ptrs[row + 1] {
            // SAFETY: col_indices[idx] < num_cols by CSR construction and
            // x.len() == num_cols·k (checked in spmm_into).
            unsafe {
                let j = *col_indices.get_unchecked(idx);
                let v = *values.get_unchecked(idx);
                let xrow = x.get_unchecked(j * k..j * k + k);
                for (a, &xv) in acc.iter_mut().zip(xrow.iter()) {
                    *a += v * xv;
                }
            }
        }
        yi.copy_from_slice(acc);
    }
}

/// Const-`K` f64 SpMM row kernel: one walk per row, `K` unrolled FMA
/// accumulators over the contiguous length-`K` rows of `x`.
#[allow(clippy::too_many_arguments)]
fn spmm_rows_f64<const K: usize>(
    values: &[f64],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[f64],
    y: &mut [f64],
    row_start: usize,
    row_end: usize,
) {
    debug_assert!(y.len() >= (row_end - row_start) * K);
    for (yi, row) in y.as_chunks_mut::<K>().0.iter_mut().zip(row_start..row_end) {
        let mut acc = [0.0f64; K];
        for idx in row_ptrs[row]..row_ptrs[row + 1] {
            // SAFETY: CSR invariants guarantee valid indices; x has
            // num_cols·K elements (checked in spmm_into).
            unsafe {
                let j = *col_indices.get_unchecked(idx);
                let v = *values.get_unchecked(idx);
                let xrow = x.get_unchecked(j * K..j * K + K);
                for c in 0..K {
                    acc[c] += v * xrow[c];
                }
            }
        }
        yi.copy_from_slice(&acc);
    }
}

/// Const-`K` Complex64 SpMM row kernel (interleaved re/im layout).
#[allow(clippy::too_many_arguments)]
fn spmm_rows_complex64<const K: usize>(
    values: &[Complex64],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[Complex64],
    y: &mut [Complex64],
    row_start: usize,
    row_end: usize,
) {
    debug_assert!(y.len() >= (row_end - row_start) * K);
    for (yi, row) in y.as_chunks_mut::<K>().0.iter_mut().zip(row_start..row_end) {
        let mut acc = [Complex64::new(0.0, 0.0); K];
        for idx in row_ptrs[row]..row_ptrs[row + 1] {
            // SAFETY: CSR invariants guarantee valid indices; x has
            // num_cols·K elements (checked in spmm_into).
            unsafe {
                let j = *col_indices.get_unchecked(idx);
                let v = *values.get_unchecked(idx);
                let xrow = x.get_unchecked(j * K..j * K + K);
                for c in 0..K {
                    acc[c] += v * xrow[c];
                }
            }
        }
        yi.copy_from_slice(&acc);
    }
}

/// Dispatch f64 SpMM to the const-`K` kernel for `k <= 8`, else the generic
/// runtime-`k` loop.
#[allow(clippy::too_many_arguments)]
fn spmm_dispatch_f64(
    values: &[f64],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[f64],
    y: &mut [f64],
    row_start: usize,
    row_end: usize,
    k: usize,
) {
    match k {
        1 => spmm_rows_f64::<1>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        2 => spmm_rows_f64::<2>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        3 => spmm_rows_f64::<3>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        4 => spmm_rows_f64::<4>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        5 => spmm_rows_f64::<5>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        6 => spmm_rows_f64::<6>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        7 => spmm_rows_f64::<7>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        8 => spmm_rows_f64::<8>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        _ => {
            let mut acc = vec![0.0f64; k];
            spmm_rows_generic(
                values,
                col_indices,
                row_ptrs,
                x,
                y,
                row_start,
                row_end,
                k,
                &mut acc,
            );
        }
    }
}

/// Dispatch Complex64 SpMM to the const-`K` kernel for `k <= 8`, else the
/// generic runtime-`k` loop.
#[allow(clippy::too_many_arguments)]
fn spmm_dispatch_complex64(
    values: &[Complex64],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[Complex64],
    y: &mut [Complex64],
    row_start: usize,
    row_end: usize,
    k: usize,
) {
    match k {
        1 => spmm_rows_complex64::<1>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        2 => spmm_rows_complex64::<2>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        3 => spmm_rows_complex64::<3>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        4 => spmm_rows_complex64::<4>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        5 => spmm_rows_complex64::<5>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        6 => spmm_rows_complex64::<6>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        7 => spmm_rows_complex64::<7>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        8 => spmm_rows_complex64::<8>(values, col_indices, row_ptrs, x, y, row_start, row_end),
        _ => {
            let mut acc = vec![Complex64::new(0.0, 0.0); k];
            spmm_rows_generic(
                values,
                col_indices,
                row_ptrs,
                x,
                y,
                row_start,
                row_end,
                k,
                &mut acc,
            );
        }
    }
}

/// SIMD-accelerated SpMV for f64 matrices.
///
/// Processes all `num_rows` rows, dispatching each row to AVX2/NEON/scalar.
/// Exposed as `pub(super)` so the NUMA module can call it per-partition.
///
/// # Safety
///
/// Caller must ensure:
///   - `row_ptrs` has at least `num_rows + 1` elements
///   - `y` has at least `num_rows` elements
///   - All indices in `col_indices` are valid indices into `x`
///   - All ranges `row_ptrs[i]..row_ptrs[i+1]` are valid indices into `values` and `col_indices`
///   - `allow_avx2_gather` is true only when every column index is at most
///     `i32::MAX`
#[inline]
pub(super) unsafe fn spmv_f64_simd(
    values: &[f64],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[f64],
    y: &mut [f64],
    num_rows: usize,
    allow_avx2_gather: bool,
) {
    for i in 0..num_rows {
        // SAFETY: bounds guaranteed by caller's CSR invariants.
        unsafe {
            let start = *row_ptrs.get_unchecked(i);
            let end = *row_ptrs.get_unchecked(i + 1);
            *y.get_unchecked_mut(i) =
                spmv_row_f64(values, col_indices, x, start, end, allow_avx2_gather);
        }
    }
}

/// Row lengths at or below this many non-zeros take the scalar loop even when
/// AVX2/FMA is available: gather setup plus the horizontal reduction costs
/// more than the arithmetic itself, and on Zen-class cores with slow
/// `vgather` the scalar loop wins by ~35% at 4-5 nnz/row. Measured crossover
/// is 8-12 nnz/row on x86-64 (fixed-row-length sweep, 60k rows); the
/// conservative end keeps longer rows on the gather path. Re-tune with the
/// same sweep before changing.
const SHORT_ROW_SCALAR_MAX_NNZ: usize = 8;

/// Compute a single sparse dot-product for one row, dispatching to the best
/// available SIMD path at runtime.
///
/// # Safety
///
/// Caller must ensure all indices in `col_indices[start..end]` are valid
/// indices into `x`, and that `start..end` is within bounds for both
/// `values` and `col_indices`.
/// If `allow_avx2_gather` is true, those column indices must also be at most
/// `i32::MAX`.
#[inline]
pub(super) unsafe fn spmv_row_f64(
    values: &[f64],
    col_indices: &[usize],
    x: &[f64],
    start: usize,
    end: usize,
    allow_avx2_gather: bool,
) -> f64 {
    // SAFETY: forwarding the caller's safety invariants to the platform-
    // specific implementation.
    #[cfg(target_arch = "x86_64")]
    {
        // Short rows never amortize gather setup + horizontal reduction.
        // `end >= start` is established by hot-path validation, so the
        // subtraction cannot underflow.
        if end - start <= SHORT_ROW_SCALAR_MAX_NNZ {
            // SAFETY: the caller's CSR bounds invariants are unchanged.
            return unsafe { spmv_row_scalar(values, col_indices, x, start, end) };
        }
        if fma_dispatch_available() && allow_avx2_gather {
            // SAFETY: `fma_dispatch_available` established AVX2+FMA before
            // entering the target-feature-isolated implementation.
            return unsafe { spmv_row_avx2_fma(values, col_indices, x, start, end) };
        }
        if avx2_dispatch_available() && allow_avx2_gather {
            // SAFETY: `avx2_dispatch_available` established the required
            // host feature immediately before entering the AVX2 function.
            return unsafe { spmv_row_avx2(values, col_indices, x, start, end) };
        }
        // SAFETY: the caller's CSR bounds invariants are unchanged.
        unsafe { spmv_row_scalar(values, col_indices, x, start, end) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { spmv_row_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { spmv_row_scalar(values, col_indices, x, start, end) }
    }
}

// ---------------------------------------------------------------------------
// x86-64 AVX2
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn spmv_row_avx2(
    values: &[f64],
    col_indices: &[usize],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    use std::arch::x86_64::*;

    // SAFETY: all AVX2 intrinsics require the avx2 target feature, which is
    // established by the runtime dispatch immediately before this function is
    // called. Slice accesses are within bounds by CSR construction invariants
    // documented at the call-site.
    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();

        let mut acc = _mm256_setzero_pd(); // 4 x f64 accumulator
        let simd_end = start + (len / 4) * 4;

        let mut idx = start;
        while idx < simd_end {
            // Load 4 consecutive values (contiguous in memory).
            let v = _mm256_loadu_pd(values.as_ptr().add(idx));

            // Narrow usize indices to i32 for the gather intrinsic.
            // This is safe for column indices < 2^31 (~2 billion columns).
            let i0 = *col_indices.get_unchecked(idx) as i32;
            let i1 = *col_indices.get_unchecked(idx + 1) as i32;
            let i2 = *col_indices.get_unchecked(idx + 2) as i32;
            let i3 = *col_indices.get_unchecked(idx + 3) as i32;
            let indices = _mm_set_epi32(i3, i2, i1, i0);

            // Gather 4 f64 values from x[] at the given indices.
            // Scale factor 8 = sizeof(f64).
            let xv = _mm256_i32gather_pd::<8>(x_ptr, indices);

            // Multiply and accumulate: acc += v * xv
            acc = _mm256_add_pd(acc, _mm256_mul_pd(v, xv));

            idx += 4;
        }

        // Horizontal sum of 4-lane accumulator -> scalar
        // [a, b, c, d] -> [a+c, b+d] -> a+c+b+d
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let pair = _mm_add_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(pair, pair);
        let mut sum = _mm_cvtsd_f64(_mm_add_sd(pair, hi64));

        // Scalar remainder (0..3 elements)
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn spmv_row_avx2_fma(
    values: &[f64],
    col_indices: &[usize],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and CSR bounds before dispatch.
    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();
        let mut acc = _mm256_setzero_pd();
        let simd_end = start + (len / 4) * 4;
        let mut idx = start;
        while idx < simd_end {
            let v = _mm256_loadu_pd(values.as_ptr().add(idx));
            let i0 = *col_indices.get_unchecked(idx) as i32;
            let i1 = *col_indices.get_unchecked(idx + 1) as i32;
            let i2 = *col_indices.get_unchecked(idx + 2) as i32;
            let i3 = *col_indices.get_unchecked(idx + 3) as i32;
            let indices = _mm_set_epi32(i3, i2, i1, i0);
            let xv = _mm256_i32gather_pd::<8>(x_ptr, indices);
            acc = _mm256_fmadd_pd(v, xv, acc);
            idx += 4;
        }
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let pair = _mm_add_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(pair, pair);
        let mut sum = _mm_cvtsd_f64(_mm_add_sd(pair, hi64));
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn spmv_row_neon(
    values: &[f64],
    col_indices: &[usize],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    use std::arch::aarch64::*;

    let len = end - start;

    // SAFETY: all NEON intrinsics require the neon target feature which is
    // guaranteed by the #[cfg] gate. Slice accesses are within bounds by
    // CSR construction invariants documented at the call-site.
    unsafe {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);

        // 4-wide unrolled loop: 2 NEON registers × 2 f64 = 4 elements/iter
        let simd4_end = start + (len / 4) * 4;
        let mut idx = start;
        while idx < simd4_end {
            let v0 = vld1q_f64(values.as_ptr().add(idx));
            let v1 = vld1q_f64(values.as_ptr().add(idx + 2));

            let j0 = *col_indices.get_unchecked(idx);
            let j1 = *col_indices.get_unchecked(idx + 1);
            let j2 = *col_indices.get_unchecked(idx + 2);
            let j3 = *col_indices.get_unchecked(idx + 3);

            let xv0 = vsetq_lane_f64::<1>(*x.get_unchecked(j1), vdupq_n_f64(*x.get_unchecked(j0)));
            let xv1 = vsetq_lane_f64::<1>(*x.get_unchecked(j3), vdupq_n_f64(*x.get_unchecked(j2)));

            acc0 = vfmaq_f64(acc0, v0, xv0);
            acc1 = vfmaq_f64(acc1, v1, xv1);

            idx += 4;
        }

        // 2-wide remainder (0 or 2 elements)
        let simd2_end = start + (len / 2) * 2;
        if idx < simd2_end {
            let v = vld1q_f64(values.as_ptr().add(idx));
            let j0 = *col_indices.get_unchecked(idx);
            let j1 = *col_indices.get_unchecked(idx + 1);
            let xv = vsetq_lane_f64::<1>(*x.get_unchecked(j1), vdupq_n_f64(*x.get_unchecked(j0)));
            acc0 = vfmaq_f64(acc0, v, xv);
            idx += 2;
        }

        // Reduce both accumulators
        let combined = vaddq_f64(acc0, acc1);
        let mut sum = vaddvq_f64(combined);

        // Scalar remainder (0 or 1 element)
        if idx < end {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn spmv_row_scalar(
    values: &[f64],
    col_indices: &[usize],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    let mut sum = 0.0;
    for idx in start..end {
        // SAFETY: bounds guaranteed by CSR construction invariants.
        unsafe {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
    }
    sum
}

// =============================================================================
// SIMD-optimized SpMV for Complex64
// =============================================================================
//
// Complex64 is stored as (re, im) pairs of f64. Each complex multiply-accumulate
// `sum += val * x[col]` requires two FMA operations:
//   sum += re(val) * x[col]                    (real-part broadcast × x)
//   sum += im(val) * rotate_negate(x[col])     (imag-part broadcast × rotated x)
//
// On aarch64 NEON: 1 complex element per iteration using float64x2_t.
// On other platforms: scalar loop with the type known at compile time (enables
// autovectorization).

/// Full SpMV driver for Complex64 — iterates rows and calls the per-row kernel.
///
/// # Safety
///
/// Same invariants as [`spmv_f64_simd`]: CSR index arrays must be consistent
/// and all column indices must be valid indices into `x`.
pub(super) unsafe fn spmv_complex64_simd(
    values: &[Complex64],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[Complex64],
    y: &mut [Complex64],
    num_rows: usize,
) {
    for i in 0..num_rows {
        unsafe {
            let start = *row_ptrs.get_unchecked(i);
            let end = *row_ptrs.get_unchecked(i + 1);
            *y.get_unchecked_mut(i) = spmv_row_complex64(values, col_indices, x, start, end);
        }
    }
}

/// Compute a single sparse complex dot-product for one row.
///
/// # Safety
///
/// All indices in `col_indices[start..end]` must be valid indices into `x`,
/// and `start..end` must be within bounds for `values` and `col_indices`.
#[inline]
pub(super) unsafe fn spmv_row_complex64(
    values: &[Complex64],
    col_indices: &[usize],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    #[cfg(target_arch = "x86_64")]
    {
        if fma_dispatch_available() {
            // SAFETY: `fma_dispatch_available` established AVX2+FMA before
            // entering the target-feature-isolated implementation.
            return unsafe { spmv_row_complex64_avx2_fma(values, col_indices, x, start, end) };
        }
        if avx2_dispatch_available() {
            // SAFETY: `avx2_dispatch_available` established the required
            // host feature immediately before entering the AVX2 function.
            return unsafe { spmv_row_complex64_avx2(values, col_indices, x, start, end) };
        }
        // SAFETY: the caller's CSR bounds invariants are unchanged.
        unsafe { spmv_row_complex64_scalar(values, col_indices, x, start, end) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { spmv_row_complex64_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { spmv_row_complex64_scalar(values, col_indices, x, start, end) }
    }
}

// ---------------------------------------------------------------------------
// x86-64 AVX2 — Complex64
// ---------------------------------------------------------------------------
//
// Each Complex64 is 16 bytes = 2 f64. We use 128-bit SSE2 (always available on
// x86-64) to process one complex element per iteration with the broadcast-rotate
// pattern, avoiding scalar re/im extraction overhead.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn spmv_row_complex64_avx2(
    values: &[Complex64],
    col_indices: &[usize],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    use std::arch::x86_64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;

        let mut acc = _mm_setzero_pd(); // [sum_re, sum_im]
        // Mask for sign-flip of lane 0: [-0.0, 0.0]
        let neg_mask = _mm_set_pd(0.0, -0.0_f64);

        for idx in start..end {
            // Load value as [re_v, im_v]
            let val = _mm_loadu_pd(values_ptr.add(idx * 2));

            // Load x[col_indices[idx]] as [re_x, im_x]
            let j = *col_indices.get_unchecked(idx);
            let xv = _mm_loadu_pd(x_ptr.add(j * 2));

            // Complex multiply-accumulate:
            // acc += [re_v*re_x - im_v*im_x, re_v*im_x + im_v*re_x]
            //
            // Step 1: acc += [re_v, re_v] * [re_x, im_x]
            let re_v = _mm_shuffle_pd::<0x00>(val, val); // [re_v, re_v]
            acc = _mm_add_pd(acc, _mm_mul_pd(re_v, xv));

            // Step 2: acc += [im_v, im_v] * [-im_x, re_x]
            let im_v = _mm_shuffle_pd::<0x03>(val, val); // [im_v, im_v]
            let xv_swap = _mm_shuffle_pd::<0x01>(xv, xv); // [im_x, re_x]
            let xv_neg_rot = _mm_xor_pd(xv_swap, neg_mask); // [-im_x, re_x]
            acc = _mm_add_pd(acc, _mm_mul_pd(im_v, xv_neg_rot));
        }

        let mut result = [0.0_f64; 2];
        _mm_storeu_pd(result.as_mut_ptr(), acc);
        Complex64::new(result[0], result[1])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn spmv_row_complex64_avx2_fma(
    values: &[Complex64],
    col_indices: &[usize],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and CSR bounds before dispatch.
    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;
        let mut acc = _mm_setzero_pd();
        let neg_mask = _mm_set_pd(0.0, -0.0_f64);
        for idx in start..end {
            let val = _mm_loadu_pd(values_ptr.add(idx * 2));
            let j = *col_indices.get_unchecked(idx);
            let xv = _mm_loadu_pd(x_ptr.add(j * 2));
            let re_v = _mm_shuffle_pd::<0x00>(val, val);
            acc = _mm_fmadd_pd(re_v, xv, acc);
            let im_v = _mm_shuffle_pd::<0x03>(val, val);
            let xv_swap = _mm_shuffle_pd::<0x01>(xv, xv);
            let xv_neg_rot = _mm_xor_pd(xv_swap, neg_mask);
            acc = _mm_fmadd_pd(im_v, xv_neg_rot, acc);
        }
        let mut result = [0.0_f64; 2];
        _mm_storeu_pd(result.as_mut_ptr(), acc);
        Complex64::new(result[0], result[1])
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON — Complex64
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn spmv_row_complex64_neon(
    values: &[Complex64],
    col_indices: &[usize],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    use std::arch::aarch64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;

        let mut acc = vdupq_n_f64(0.0); // [sum_re, sum_im]
        // Mask for negating the first lane: [-1.0, 1.0]
        let neg_mask = vsetq_lane_f64::<0>(-1.0, vdupq_n_f64(1.0));

        for idx in start..end {
            // Load value as [re_v, im_v]
            let val = vld1q_f64(values_ptr.add(idx * 2));

            // Load x[col_indices[idx]] as [re_x, im_x]
            let j = *col_indices.get_unchecked(idx);
            let xv = vld1q_f64(x_ptr.add(j * 2));

            // Complex multiply-accumulate:
            // acc += [re_v*re_x - im_v*im_x, re_v*im_x + im_v*re_x]
            //
            // Step 1: acc += [re_v, re_v] * [re_x, im_x]
            let re_v = vdupq_laneq_f64::<0>(val);
            acc = vfmaq_f64(acc, re_v, xv);

            // Step 2: acc += [im_v, im_v] * [-im_x, re_x]
            let im_v = vdupq_laneq_f64::<1>(val);
            let xv_swap = vextq_f64(xv, xv, 1); // [im_x, re_x]
            let xv_neg_rot = vmulq_f64(xv_swap, neg_mask); // [-im_x, re_x]
            acc = vfmaq_f64(acc, im_v, xv_neg_rot);
        }

        let re = vgetq_lane_f64::<0>(acc);
        let im = vgetq_lane_f64::<1>(acc);
        Complex64::new(re, im)
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback — Complex64
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn spmv_row_complex64_scalar(
    values: &[Complex64],
    col_indices: &[usize],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for idx in start..end {
        unsafe {
            let j = *col_indices.get_unchecked(idx);
            let v = *values.get_unchecked(idx);
            let xj = *x.get_unchecked(j);
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            sum_re += v.re * xj.re - v.im * xj.im;
            sum_im += v.re * xj.im + v.im * xj.re;
        }
    }
    Complex64::new(sum_re, sum_im)
}

// =============================================================================
// SIMD-optimized SpMV for f32
// =============================================================================
//
// On aarch64 NEON: process 4 f32s at once using float32x4_t.
// On other platforms: scalar loop with the type known at compile time.

/// Full SpMV driver for f32.
///
/// # Safety
///
/// Same invariants as [`spmv_f64_simd`].
unsafe fn spmv_f32_simd(
    values: &[f32],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[f32],
    y: &mut [f32],
    num_rows: usize,
) {
    for i in 0..num_rows {
        unsafe {
            let start = *row_ptrs.get_unchecked(i);
            let end = *row_ptrs.get_unchecked(i + 1);
            *y.get_unchecked_mut(i) = spmv_row_f32(values, col_indices, x, start, end);
        }
    }
}

/// Per-row f32 sparse dot-product with runtime SIMD dispatch.
///
/// # Safety
///
/// All indices in `col_indices[start..end]` must be valid indices into `x`.
#[inline]
unsafe fn spmv_row_f32(
    values: &[f32],
    col_indices: &[usize],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if fma_dispatch_available() {
            // SAFETY: `fma_dispatch_available` established AVX2+FMA before
            // entering the target-feature-isolated implementation.
            return unsafe { spmv_row_f32_avx2_fma(values, col_indices, x, start, end) };
        }
        if avx2_dispatch_available() {
            // SAFETY: `avx2_dispatch_available` established the required
            // host feature immediately before entering the AVX2 function.
            return unsafe { spmv_row_f32_avx2(values, col_indices, x, start, end) };
        }
        // SAFETY: the caller's CSR bounds invariants are unchanged.
        unsafe { spmv_row_f32_scalar(values, col_indices, x, start, end) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { spmv_row_f32_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { spmv_row_f32_scalar(values, col_indices, x, start, end) }
    }
}

// ---------------------------------------------------------------------------
// x86-64 AVX2 — f32
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn spmv_row_f32_avx2(
    values: &[f32],
    col_indices: &[usize],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: all AVX2 intrinsics require the avx2 target feature, which is
    // established by the runtime dispatch immediately before this function is
    // called. Slice accesses are within bounds by CSR construction invariants
    // documented at the call-site.
    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();

        let mut acc = _mm256_setzero_ps(); // 8 x f32 accumulator
        let simd_end = start + (len / 8) * 8;

        let mut idx = start;
        while idx < simd_end {
            // Load 8 consecutive values (contiguous in memory).
            let v = _mm256_loadu_ps(values.as_ptr().add(idx));

            // Narrow usize indices to i32 for the gather intrinsic.
            let i0 = *col_indices.get_unchecked(idx) as i32;
            let i1 = *col_indices.get_unchecked(idx + 1) as i32;
            let i2 = *col_indices.get_unchecked(idx + 2) as i32;
            let i3 = *col_indices.get_unchecked(idx + 3) as i32;
            let i4 = *col_indices.get_unchecked(idx + 4) as i32;
            let i5 = *col_indices.get_unchecked(idx + 5) as i32;
            let i6 = *col_indices.get_unchecked(idx + 6) as i32;
            let i7 = *col_indices.get_unchecked(idx + 7) as i32;
            let indices = _mm256_set_epi32(i7, i6, i5, i4, i3, i2, i1, i0);

            // Gather 8 f32 values from x[] at the given indices.
            // Scale factor 4 = sizeof(f32).
            let xv = _mm256_i32gather_ps::<4>(x_ptr, indices);

            // Multiply and accumulate: acc += v * xv
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, xv));

            idx += 8;
        }

        // Horizontal sum of 8-lane accumulator -> scalar
        // [a,b,c,d,e,f,g,h] -> extract hi128=[e,f,g,h], add to lo128=[a+e,b+f,c+g,d+h]
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128); // [a+e, b+f, c+g, d+h]
        // Horizontal add pairs: [a+e+b+f, c+g+d+h, ...]
        let shuf = _mm_movehdup_ps(sum128); // [b+f, b+f, d+h, d+h]
        let sums = _mm_add_ps(sum128, shuf); // [a+e+b+f, ..., c+g+d+h, ...]
        let hi = _mm_movehl_ps(sums, sums); // move bits 64..127 to 0..63
        let final_sum = _mm_add_ss(sums, hi);
        let mut sum = _mm_cvtss_f32(final_sum);

        // Scalar remainder (0..7 elements)
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn spmv_row_f32_avx2_fma(
    values: &[f32],
    col_indices: &[usize],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and CSR bounds before dispatch.
    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();
        let mut acc = _mm256_setzero_ps();
        let simd_end = start + (len / 8) * 8;
        let mut idx = start;
        while idx < simd_end {
            let v = _mm256_loadu_ps(values.as_ptr().add(idx));
            let i0 = *col_indices.get_unchecked(idx) as i32;
            let i1 = *col_indices.get_unchecked(idx + 1) as i32;
            let i2 = *col_indices.get_unchecked(idx + 2) as i32;
            let i3 = *col_indices.get_unchecked(idx + 3) as i32;
            let i4 = *col_indices.get_unchecked(idx + 4) as i32;
            let i5 = *col_indices.get_unchecked(idx + 5) as i32;
            let i6 = *col_indices.get_unchecked(idx + 6) as i32;
            let i7 = *col_indices.get_unchecked(idx + 7) as i32;
            let indices = _mm256_set_epi32(i7, i6, i5, i4, i3, i2, i1, i0);
            let xv = _mm256_i32gather_ps::<4>(x_ptr, indices);
            acc = _mm256_fmadd_ps(v, xv, acc);
            idx += 8;
        }
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let hi = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, hi);
        let mut sum = _mm_cvtss_f32(final_sum);
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON — f32
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn spmv_row_f32_neon(
    values: &[f32],
    col_indices: &[usize],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let len = end - start;

    unsafe {
        let mut acc = vdupq_n_f32(0.0); // 4 x f32 accumulator

        let simd_end = start + (len / 4) * 4;
        let mut idx = start;
        while idx < simd_end {
            let v = vld1q_f32(values.as_ptr().add(idx));

            // NEON has no f32 gather — load individually and combine
            let j0 = *col_indices.get_unchecked(idx);
            let j1 = *col_indices.get_unchecked(idx + 1);
            let j2 = *col_indices.get_unchecked(idx + 2);
            let j3 = *col_indices.get_unchecked(idx + 3);

            let mut xv = vdupq_n_f32(*x.get_unchecked(j0));
            xv = vsetq_lane_f32::<1>(*x.get_unchecked(j1), xv);
            xv = vsetq_lane_f32::<2>(*x.get_unchecked(j2), xv);
            xv = vsetq_lane_f32::<3>(*x.get_unchecked(j3), xv);

            acc = vfmaq_f32(acc, v, xv);
            idx += 4;
        }

        let mut sum = vaddvq_f32(acc);

        // Scalar remainder (0..3 elements)
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn spmv_row_f32_scalar(
    values: &[f32],
    col_indices: &[usize],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for idx in start..end {
        unsafe {
            let j = *col_indices.get_unchecked(idx);
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
    }
    sum
}

// =============================================================================
// SIMD-optimized SpMV for Complex32
// =============================================================================
//
// Complex32 is stored as (re, im) pairs of f32. Each complex multiply-accumulate
// `sum += val * x[col]` requires two FMA operations (same pattern as Complex64).
//
// On aarch64 NEON: float32x4_t holds 2 Complex32 elements ([re0, im0, re1, im1]),
// but we process 1 at a time using the lower 2 lanes for correctness, matching
// the Complex64 NEON pattern but with f32 intrinsics.
//
// On other platforms: scalar loop.

/// Full SpMV driver for Complex32.
///
/// # Safety
///
/// Same invariants as [`spmv_f64_simd`].
unsafe fn spmv_complex32_simd(
    values: &[Complex32],
    col_indices: &[usize],
    row_ptrs: &[usize],
    x: &[Complex32],
    y: &mut [Complex32],
    num_rows: usize,
) {
    for i in 0..num_rows {
        unsafe {
            let start = *row_ptrs.get_unchecked(i);
            let end = *row_ptrs.get_unchecked(i + 1);
            *y.get_unchecked_mut(i) = spmv_row_complex32(values, col_indices, x, start, end);
        }
    }
}

/// Per-row Complex32 sparse dot-product.
///
/// # Safety
///
/// All indices in `col_indices[start..end]` must be valid indices into `x`.
#[inline]
unsafe fn spmv_row_complex32(
    values: &[Complex32],
    col_indices: &[usize],
    x: &[Complex32],
    start: usize,
    end: usize,
) -> Complex32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { spmv_row_complex32_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        unsafe { spmv_row_complex32_scalar(values, col_indices, x, start, end) }
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON — Complex32
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn spmv_row_complex32_neon(
    values: &[Complex32],
    col_indices: &[usize],
    x: &[Complex32],
    start: usize,
    end: usize,
) -> Complex32 {
    use std::arch::aarch64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f32;
        let x_ptr = x.as_ptr() as *const f32;

        // Accumulate in a float32x2_t: [sum_re, sum_im]
        let mut acc = vdup_n_f32(0.0);
        // Mask for negating the first lane: [-1.0, 1.0]
        let neg_mask = vset_lane_f32::<0>(-1.0, vdup_n_f32(1.0));

        for idx in start..end {
            // Load value as [re_v, im_v]
            let val = vld1_f32(values_ptr.add(idx * 2));

            // Load x[col_indices[idx]] as [re_x, im_x]
            let j = *col_indices.get_unchecked(idx);
            let xv = vld1_f32(x_ptr.add(j * 2));

            // Complex multiply-accumulate:
            // acc += [re_v*re_x - im_v*im_x, re_v*im_x + im_v*re_x]
            //
            // Step 1: acc += [re_v, re_v] * [re_x, im_x]
            let re_v = vdup_lane_f32::<0>(val);
            acc = vfma_f32(acc, re_v, xv);

            // Step 2: acc += [im_v, im_v] * [-im_x, re_x]
            let im_v = vdup_lane_f32::<1>(val);
            let xv_swap = vext_f32::<1>(xv, xv); // [im_x, re_x]
            let xv_neg_rot = vmul_f32(xv_swap, neg_mask); // [-im_x, re_x]
            acc = vfma_f32(acc, im_v, xv_neg_rot);
        }

        let re = vget_lane_f32::<0>(acc);
        let im = vget_lane_f32::<1>(acc);
        Complex32::new(re, im)
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback — Complex32
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn spmv_row_complex32_scalar(
    values: &[Complex32],
    col_indices: &[usize],
    x: &[Complex32],
    start: usize,
    end: usize,
) -> Complex32 {
    let mut sum_re = 0.0_f32;
    let mut sum_im = 0.0_f32;
    for idx in start..end {
        unsafe {
            let j = *col_indices.get_unchecked(idx);
            let v = *values.get_unchecked(idx);
            let xj = *x.get_unchecked(j);
            sum_re += v.re * xj.re - v.im * xj.im;
            sum_im += v.re * xj.im + v.im * xj.re;
        }
    }
    Complex32::new(sum_re, sum_im)
}

fn invert_permutation(permutation: &[usize]) -> Vec<usize> {
    let n = permutation.len();
    let mut inverse = vec![usize::MAX; n];
    for (new_index, &old_index) in permutation.iter().enumerate() {
        assert!(
            old_index < n,
            "permutation entry {old_index} out of bounds for length {n}"
        );
        assert_eq!(
            inverse[old_index],
            usize::MAX,
            "permutation contains duplicate index {old_index}"
        );
        inverse[old_index] = new_index;
    }
    assert!(
        inverse.iter().all(|&index| index != usize::MAX),
        "permutation must contain every index exactly once"
    );
    inverse
}

fn nested_dissection_ordering_from_adjacency(
    adjacency: &[Vec<usize>],
    min_block_size: usize,
    max_levels: Option<usize>,
) -> Vec<usize> {
    let n = adjacency.len();
    if n == 0 {
        return Vec::new();
    }

    let min_block_size = min_block_size.max(1);
    let max_levels = max_levels.unwrap_or_else(|| {
        let mut levels = 0;
        let mut size = n;
        while size > min_block_size {
            levels += 1;
            size = size.div_ceil(2);
        }
        levels
    });

    let indices = (0..n).collect::<Vec<_>>();
    let mut nodes = Vec::new();
    nested_dissection(
        adjacency,
        &indices,
        min_block_size,
        max_levels,
        0,
        &mut nodes,
    );

    let mut permutation = Vec::with_capacity(n);
    for node in &nodes {
        permutation.extend_from_slice(&node.global_indices);
    }
    let _ = invert_permutation(&permutation);
    permutation
}

impl<T: ComplexField> LinearOperator<T> for CsrMatrix<T> {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn apply(&self, x: &[T]) -> Vec<T> {
        self.matvec(x)
    }

    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec(x)
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_into_buf(x, y);
    }

    /// Multi-vector apply via the SpMM kernel: one matrix walk for all `k`
    /// right-hand sides instead of `k` gather/scatter + matvec rounds.
    fn apply_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        self.spmm_into(x, y);
    }

    fn try_apply_block(
        &self,
        x: &DenseMatrix<T>,
        y: &mut DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        self.try_spmm_into(x, y)
    }

    fn apply_transpose(&self, x: &[T]) -> Vec<T> {
        self.matvec_transpose(x)
    }

    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_transpose(x)
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_transpose_into_buf(x, y);
    }

    /// Multi-vector transpose apply with validate-once structure checking.
    ///
    /// The default trait loop calls the single-vector callback per column,
    /// which would re-run the O(nnz) hot validation k times. The structure
    /// cannot change between columns of one call, so one validation covers
    /// all `k` unchecked column applies.
    fn apply_transpose_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        self.validate_hot_structure().expect("invalid CSR storage");
        assert_eq!(
            x.nrows(),
            self.num_rows,
            "block transpose input row mismatch"
        );
        assert_eq!(
            y.nrows(),
            self.num_cols,
            "block transpose output row mismatch"
        );
        assert_eq!(
            x.ncols(),
            y.ncols(),
            "block transpose column count mismatch"
        );
        let m = self.num_rows;
        let n = self.num_cols;
        let mut xj = vec![T::zero(); m];
        let mut yj = vec![T::zero(); n];
        for j in 0..x.ncols() {
            for i in 0..m {
                xj[i] = x[[i, j]];
            }
            // SAFETY: validated once above; `&self` is shared (not mutated)
            // for the whole loop, so the memory-safety contract holds for
            // every column. Length asserts are re-checked per column inside.
            unsafe { self.matvec_transpose_into_buf_unchecked(&xj, &mut yj) };
            y.set_column(j, &yj);
        }
    }

    fn try_apply_transpose_block(
        &self,
        x: &DenseMatrix<T>,
        y: &mut DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        self.validate_structure()?;
        if x.nrows() != self.num_rows {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows,
                got: x.nrows(),
            });
        }
        if x.as_slice().iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "CSR block transpose SpMV",
                reason: "input block contains NaN or infinity".into(),
            });
        }
        if y.nrows() != self.num_cols {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols,
                got: y.nrows(),
            });
        }
        if x.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: x.ncols(),
                got: y.ncols(),
            });
        }

        let mut staged = DenseMatrix::zeros(self.num_cols, x.ncols());
        catch_unwind(AssertUnwindSafe(|| {
            self.apply_transpose_block(x, &mut staged)
        }))
        .map_err(|_| SolverError::OperationFailed {
            operation: "CSR block transpose SpMV",
            reason: "transpose SpMM kernel panicked".into(),
        })?;
        Self::reject_non_finite_spmv_output("CSR block transpose SpMV", staged.as_slice())?;
        y.as_mut_slice().copy_from_slice(staged.as_slice());
        Ok(())
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        self.matvec_hermitian(x)
    }

    /// Multi-vector Hermitian apply with validate-once structure checking
    /// (same `k`-column validation saving as the transpose override above).
    fn apply_hermitian_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        self.validate_hot_structure().expect("invalid CSR storage");
        assert_eq!(
            x.nrows(),
            self.num_rows,
            "block Hermitian input row mismatch"
        );
        assert_eq!(
            y.nrows(),
            self.num_cols,
            "block Hermitian output row mismatch"
        );
        assert_eq!(
            x.ncols(),
            y.ncols(),
            "block Hermitian column count mismatch"
        );
        let m = self.num_rows;
        let n = self.num_cols;
        let mut xj = vec![T::zero(); m];
        let mut yj = vec![T::zero(); n];
        for j in 0..x.ncols() {
            for i in 0..m {
                xj[i] = x[[i, j]];
            }
            // SAFETY: validated once above; `&self` is shared (not mutated)
            // for the whole loop, so the memory-safety contract holds for
            // every column. Length asserts are re-checked per column inside.
            unsafe { self.matvec_hermitian_into_buf_unchecked(&xj, &mut yj) };
            y.set_column(j, &yj);
        }
    }

    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_hermitian(x)
    }

    fn try_apply_hermitian_block(
        &self,
        x: &DenseMatrix<T>,
        y: &mut DenseMatrix<T>,
    ) -> Result<(), SolverError> {
        self.validate_structure()?;
        if x.nrows() != self.num_rows {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_rows,
                got: x.nrows(),
            });
        }
        if x.as_slice().iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "CSR block Hermitian SpMV",
                reason: "input block contains NaN or infinity".into(),
            });
        }
        if y.nrows() != self.num_cols {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols,
                got: y.nrows(),
            });
        }
        if x.ncols() != y.ncols() {
            return Err(SolverError::DimensionMismatch {
                expected: x.ncols(),
                got: y.ncols(),
            });
        }

        let mut staged = DenseMatrix::zeros(self.num_cols, x.ncols());
        catch_unwind(AssertUnwindSafe(|| {
            self.apply_hermitian_block(x, &mut staged)
        }))
        .map_err(|_| SolverError::OperationFailed {
            operation: "CSR block Hermitian SpMV",
            reason: "Hermitian transpose SpMM kernel panicked".into(),
        })?;
        Self::reject_non_finite_spmv_output("CSR block Hermitian SpMV", staged.as_slice())?;
        y.as_mut_slice().copy_from_slice(staged.as_slice());
        Ok(())
    }

    fn transpose_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }
}

/// Builder for constructing CSR matrices row by row
pub struct CsrBuilder<T: ComplexField> {
    num_rows: usize,
    num_cols: usize,
    values: Vec<T>,
    col_indices: Vec<usize>,
    row_ptrs: Vec<usize>,
    current_row: usize,
}

impl<T: ComplexField> CsrBuilder<T> {
    /// Create a new CSR builder
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptrs: vec![0],
            current_row: 0,
        }
    }

    /// Create a new CSR builder with estimated non-zeros
    pub fn with_capacity(num_rows: usize, num_cols: usize, nnz_estimate: usize) -> Self {
        let mut row_ptrs = Vec::with_capacity(num_rows + 1);
        row_ptrs.push(0);
        Self {
            num_rows,
            num_cols,
            values: Vec::with_capacity(nnz_estimate),
            col_indices: Vec::with_capacity(nnz_estimate),
            row_ptrs,
            current_row: 0,
        }
    }

    /// Add entries for the current row (must be added in column order)
    pub fn add_row_entries(&mut self, entries: impl Iterator<Item = (usize, T)>) {
        for (col, val) in entries {
            if val.norm() > T::Real::zero() {
                self.values.push(val);
                self.col_indices.push(col);
            }
        }
        self.row_ptrs.push(self.values.len());
        self.current_row += 1;
    }

    /// Finish building and return the CSR matrix
    pub fn finish(mut self) -> CsrMatrix<T> {
        // Fill remaining rows if not all rows were added
        while self.current_row < self.num_rows {
            self.row_ptrs.push(self.values.len());
            self.current_row += 1;
        }

        CsrMatrix {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            values: self.values,
            col_indices: self.col_indices,
            row_ptrs: self.row_ptrs,
        }
    }
}

/// Blocked CSR format for hierarchical matrices
///
/// Stores the matrix as a collection of dense blocks at the leaf level
/// of a hierarchical decomposition.
#[derive(Debug, Clone)]
pub struct BlockedCsr<T: ComplexField> {
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_cols: usize,
    /// Block size (rows and columns per block)
    pub block_size: usize,
    /// Number of block rows
    pub num_block_rows: usize,
    /// Number of block columns
    pub num_block_cols: usize,
    /// Dense blocks stored in CSR-like format
    pub blocks: Vec<DenseMatrix<T>>,
    /// Block column indices
    pub block_col_indices: Vec<usize>,
    /// Block row pointers
    pub block_row_ptrs: Vec<usize>,
}

impl<T: ComplexField> BlockedCsr<T> {
    /// Create a new blocked CSR matrix
    pub fn new(num_rows: usize, num_cols: usize, block_size: usize) -> Self {
        let num_block_rows = num_rows.div_ceil(block_size);
        let num_block_cols = num_cols.div_ceil(block_size);

        Self {
            num_rows,
            num_cols,
            block_size,
            num_block_rows,
            num_block_cols,
            blocks: Vec::new(),
            block_col_indices: Vec::new(),
            block_row_ptrs: vec![0; num_block_rows + 1],
        }
    }

    /// Matrix-vector product using blocked structure
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");

        let mut y = vec![T::zero(); self.num_rows];

        for block_i in 0..self.num_block_rows {
            let row_start = block_i * self.block_size;
            let row_end = (row_start + self.block_size).min(self.num_rows);
            let local_rows = row_end - row_start;

            for idx in self.block_row_ptrs[block_i]..self.block_row_ptrs[block_i + 1] {
                let block_j = self.block_col_indices[idx];
                let block = &self.blocks[idx];

                let col_start = block_j * self.block_size;
                let col_end = (col_start + self.block_size).min(self.num_cols);
                let local_cols = col_end - col_start;

                // Apply block — index x directly (no per-block allocation)
                for i in 0..local_rows {
                    let mut sum = T::zero();
                    for j in 0..local_cols {
                        sum += block[[i, j]] * x[col_start + j];
                    }
                    y[row_start + i] += sum;
                }
            }
        }

        y
    }
}

/// Optimized sparse matrix-matrix multiplication: C = A * B
///
/// Uses a sorted accumulation approach instead of HashMap for better cache locality,
/// providing 2-4x speedup for the AMG Galerkin product.
///
/// For CSR matrices A (m×k) and B (k×n), computes C (m×n).
impl<T: ComplexField> CsrMatrix<T> {
    /// Checked sparse matrix-matrix multiplication.
    ///
    /// The compatibility [`Self::matmul`] and [`Self::matmul_with_drop`]
    /// methods retain their historical panic behavior. This entry point
    /// validates both operands and the drop policy before dispatch, contains
    /// legacy arithmetic panics, and rejects non-finite values generated by
    /// products or accumulation as a typed construction failure.
    pub fn try_matmul(&self, other: &CsrMatrix<T>) -> Result<CsrMatrix<T>, SolverError> {
        self.try_matmul_with_drop(other, T::Real::zero())
    }

    /// Checked sparse matrix-matrix multiplication with explicit drop policy.
    pub fn try_matmul_with_drop(
        &self,
        other: &CsrMatrix<T>,
        tol: T::Real,
    ) -> Result<CsrMatrix<T>, SolverError> {
        self.validate_structure()?;
        other.validate_structure()?;
        if self.num_cols != other.num_rows {
            return Err(SolverError::DimensionMismatch {
                expected: self.num_cols,
                got: other.num_rows,
            });
        }
        if !tol.is_finite() || tol < T::Real::zero() {
            return Err(SolverError::OperationFailed {
                operation: "CSR matmul",
                reason: "drop tolerance must be finite and non-negative".into(),
            });
        }

        self.validate_matmul_finite(other)?;

        let product = catch_unwind(AssertUnwindSafe(|| self.matmul_with_drop(other, tol)))
            .map_err(|_| SolverError::OperationFailed {
                operation: "CSR matmul",
                reason: "matrix products or accumulation generated a non-finite value".into(),
            })?;
        product
            .validate_structure()
            .map_err(|error| SolverError::OperationFailed {
                operation: "CSR matmul",
                reason: format!("generated product failed CSR validation: {error}"),
            })?;
        Ok(product)
    }

    fn validate_matmul_finite(&self, other: &CsrMatrix<T>) -> Result<(), SolverError> {
        let mut accumulator = vec![T::zero(); other.num_cols];
        let mut seen = vec![false; other.num_cols];
        let mut touched = Vec::new();
        for row in 0..self.num_rows {
            touched.clear();
            for (k, left_value) in self.row_entries(row) {
                for (column, right_value) in other.row_entries(k) {
                    let product = left_value * right_value;
                    if !product.norm().is_finite() {
                        return Err(SolverError::OperationFailed {
                            operation: "CSR matmul",
                            reason: format!(
                                "matrix product generated non-finite value at ({row}, {column})"
                            ),
                        });
                    }
                    if !seen[column] {
                        seen[column] = true;
                        touched.push(column);
                    }
                    accumulator[column] += product;
                    if !accumulator[column].norm().is_finite() {
                        return Err(SolverError::OperationFailed {
                            operation: "CSR matmul",
                            reason: format!(
                                "matrix accumulation generated non-finite value at ({row}, {column})"
                            ),
                        });
                    }
                }
            }
            for &column in &touched {
                accumulator[column] = T::zero();
                seen[column] = false;
            }
        }
        Ok(())
    }

    /// Compute C = A * B using optimized approach (Gustavson's algorithm)
    ///
    /// This is an *exact* SpGEMM: every structurally computed nonzero is
    /// kept — no magnitude-based dropping. Entries whose contributions
    /// cancel to an exact numerical zero are still omitted (they carry no
    /// information).
    ///
    /// Callers that want thresholded output (e.g. the AMG Galerkin product,
    /// which drops numerically tiny entries to control coarse-operator
    /// complexity) should opt in explicitly via
    /// [`CsrMatrix::matmul_with_drop`].
    pub fn matmul(&self, other: &CsrMatrix<T>) -> CsrMatrix<T> {
        self.matmul_with_drop(other, T::Real::from_f64(0.0).unwrap())
    }

    /// Compute C = A * B, dropping accumulated entries with `|c_ij| <= tol`.
    ///
    /// `tol` is an absolute threshold on the entry norm. Passing zero gives
    /// the exact product (identical to [`CsrMatrix::matmul`]). Dropping is a
    /// numerical policy of the caller, not of general-purpose SpGEMM, so it
    /// lives behind this explicit opt-in.
    pub fn matmul_with_drop(&self, other: &CsrMatrix<T>, tol: T::Real) -> CsrMatrix<T> {
        assert_eq!(
            self.num_cols, other.num_rows,
            "Matrix dimension mismatch: A.cols ({}) != B.rows ({})",
            self.num_cols, other.num_rows
        );

        let m = self.num_rows;
        let n = other.num_cols;

        if m == 0 || n == 0 || self.nnz() == 0 || other.nnz() == 0 {
            return CsrMatrix::new(m, n);
        }

        // Below the parallel threshold the workspace allocation per thread
        // (n × 17 bytes) outweighs the work; stay sequential.
        const PARALLEL_MATMUL_THRESHOLD: usize = 4_096;
        if m < PARALLEL_MATMUL_THRESHOLD || n < PARALLEL_MATMUL_THRESHOLD {
            return self.matmul_sequential(other, tol);
        }

        // Parallel: each thread takes a contiguous row range with its own
        // Gustavson workspace. The dense `sparse_accumulator + occupied`
        // pair costs n × 17 bytes per thread; capping chunk count at the
        // current rayon worker count keeps total workspace bounded.
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_size = m.div_ceil(nthreads);
        let chunks: Vec<(usize, usize)> = (0..nthreads)
            .map(|t| (t * chunk_size, ((t + 1) * chunk_size).min(m)))
            .filter(|(s, e)| e > s)
            .collect();

        let per_chunk: Vec<Vec<(usize, usize, T)>> = chunks
            .into_par_iter()
            .map(|(start, end)| {
                let mut sparse_accumulator = vec![T::zero(); n];
                let mut active_indices: Vec<usize> = Vec::with_capacity(64);
                let mut occupied = vec![false; n];
                let mut local: Vec<(usize, usize, T)> = Vec::new();

                for i in start..end {
                    for (k, a_ik) in self.row_entries(i) {
                        for (j, b_kj) in other.row_entries(k) {
                            if !occupied[j] {
                                occupied[j] = true;
                                active_indices.push(j);
                            }
                            sparse_accumulator[j] += a_ik * b_kj;
                        }
                    }

                    if active_indices.is_empty() {
                        continue;
                    }

                    active_indices.sort_unstable();

                    for &j in &active_indices {
                        let val = sparse_accumulator[j];
                        if val.norm() > tol {
                            local.push((i, j, val));
                        }
                        sparse_accumulator[j] = T::zero();
                        occupied[j] = false;
                    }
                    active_indices.clear();
                }

                local
            })
            .collect();

        let total: usize = per_chunk.iter().map(|v| v.len()).sum();
        let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(total);
        for chunk in per_chunk {
            triplets.extend(chunk);
        }

        Self::from_sorted_unique_triplets(m, n, triplets)
    }

    fn matmul_sequential(&self, other: &CsrMatrix<T>, tol: T::Real) -> CsrMatrix<T> {
        let m = self.num_rows;
        let n = other.num_cols;
        let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(self.nnz() * 2);

        // Workspace for Gustavson's algorithm
        let mut sparse_accumulator = vec![T::zero(); n];
        let mut active_indices = Vec::with_capacity(n);
        let mut occupied = vec![false; n];

        for i in 0..m {
            for (k, a_ik) in self.row_entries(i) {
                for (j, b_kj) in other.row_entries(k) {
                    if !occupied[j] {
                        occupied[j] = true;
                        active_indices.push(j);
                    }
                    sparse_accumulator[j] += a_ik * b_kj;
                }
            }

            if active_indices.is_empty() {
                continue;
            }

            // To keep C's rows sorted by column index, we sort active_indices
            active_indices.sort_unstable();

            for &j in &active_indices {
                let val = sparse_accumulator[j];
                if val.norm() > tol {
                    triplets.push((i, j, val));
                }
                // Reset for next row
                sparse_accumulator[j] = T::zero();
                occupied[j] = false;
            }
            active_indices.clear();
        }

        Self::from_sorted_unique_triplets(m, n, triplets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn dense2(rows: usize, cols: usize, data: Vec<Complex64>) -> DenseMatrix<Complex64> {
        DenseMatrix::from_row_major_vec(rows, cols, data)
    }

    fn small_poisson_f64() -> CsrMatrix<f64> {
        // 3x3 5-point Poisson-like pattern: short rows exercise the scalar
        // short-row dispatch on the hot path.
        CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 4.0),
            ],
        )
    }

    #[test]
    fn hot_validation_accepts_valid_short_row_matrix() {
        let a = small_poisson_f64();
        assert!(a.validate_hot_structure().is_ok());
        assert!(a.validate_structure().is_ok());
        let mut y = vec![0.0; 3];
        a.matvec_into_buf(&[1.0, 1.0, 1.0], &mut y);
        assert_relative_eq!(y[0], 3.0);
        assert_relative_eq!(y[1], 2.0);
        assert_relative_eq!(y[2], 3.0);
    }

    #[test]
    #[should_panic(expected = "invalid CSR storage")]
    fn hot_validation_rejects_out_of_bounds_column() {
        let mut a = small_poisson_f64();
        a.col_indices[0] = 99;
        let mut y = vec![0.0; 3];
        a.matvec_into_buf(&[1.0, 1.0, 1.0], &mut y);
    }

    #[test]
    #[should_panic(expected = "invalid CSR storage")]
    fn hot_validation_rejects_non_monotone_row_ptrs() {
        let mut a = small_poisson_f64();
        a.row_ptrs[1] = a.row_ptrs[2] + 1;
        let mut y = vec![0.0; 3];
        a.matvec_into_buf(&[1.0, 1.0, 1.0], &mut y);
    }

    #[test]
    fn hot_validation_accepts_non_finite_values() {
        // The hot path only enforces memory-safety invariants; non-finite
        // values propagate arithmetically instead of panicking. Full
        // `validate_structure` still rejects them for checked boundaries.
        let mut a = small_poisson_f64();
        a.values[0] = f64::INFINITY;
        assert!(a.validate_hot_structure().is_ok());
        assert!(a.validate_structure().is_err());
        let mut y = vec![0.0; 3];
        a.matvec_into_buf(&[1.0, 1.0, 1.0], &mut y);
        assert!(y[0].is_infinite());
    }

    #[test]
    fn test_csr_from_dense() {
        let dense = dense2(
            3,
            3,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(5.0, 0.0),
            ],
        );

        let csr = CsrMatrix::from_dense(&dense, 1e-15);

        assert_eq!(csr.num_rows, 3);
        assert_eq!(csr.num_cols, 3);
        assert_eq!(csr.nnz(), 5);

        assert_relative_eq!(csr.get(0, 0).re, 1.0);
        assert_relative_eq!(csr.get(0, 2).re, 2.0);
        assert_relative_eq!(csr.get(1, 1).re, 3.0);
        assert_relative_eq!(csr.get(2, 0).re, 4.0);
        assert_relative_eq!(csr.get(2, 2).re, 5.0);
    }

    #[test]
    fn test_csr_matvec() {
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let y = csr.matvec(&x);

        // [1 2] * [1]   [5]
        // [3 4]   [2] = [11]
        assert_relative_eq!(y[0].re, 5.0, epsilon = 1e-10);
        assert_relative_eq!(y[1].re, 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_csr_matvec_complex64_imaginary() {
        // Matrix with non-trivial imaginary parts to exercise the SIMD
        // complex multiply-accumulate path:
        //   [(1+2i)  (3+4i)]   [(1+i)]   [(1+2i)(1+i) + (3+4i)(2-i)]
        //   [(5+6i)  (7+8i)] * [(2-i)] = [(5+6i)(1+i) + (7+8i)(2-i)]
        //
        // Row 0: (1+2i)(1+i) = 1+i+2i+2i² = 1+3i-2 = -1+3i
        //        (3+4i)(2-i) = 6-3i+8i-4i² = 6+5i+4 = 10+5i
        //        sum = 9+8i
        //
        // Row 1: (5+6i)(1+i) = 5+5i+6i+6i² = 5+11i-6 = -1+11i
        //        (7+8i)(2-i) = 14-7i+16i-8i² = 14+9i+8 = 22+9i
        //        sum = 21+20i
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, 6.0),
                Complex64::new(7.0, 8.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        // Test matvec (allocating)
        let y = csr.matvec(&x);
        assert_relative_eq!(y[0].re, 9.0, epsilon = 1e-10);
        assert_relative_eq!(y[0].im, 8.0, epsilon = 1e-10);
        assert_relative_eq!(y[1].re, 21.0, epsilon = 1e-10);
        assert_relative_eq!(y[1].im, 20.0, epsilon = 1e-10);

        // Test matvec_into_buf (pre-allocated, same SIMD path)
        let mut y_buf = vec![Complex64::new(0.0, 0.0); 2];
        csr.matvec_into_buf(&x, &mut y_buf);
        assert_relative_eq!(y_buf[0].re, 9.0, epsilon = 1e-10);
        assert_relative_eq!(y_buf[0].im, 8.0, epsilon = 1e-10);
        assert_relative_eq!(y_buf[1].re, 21.0, epsilon = 1e-10);
        assert_relative_eq!(y_buf[1].im, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_csr_apply_into() {
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![Complex64::new(5.0, 1.0), Complex64::new(7.0, -2.0)];

        let y_alloc = csr.apply(&x);
        let mut y_into = vec![Complex64::new(0.0, 0.0); 2];
        csr.apply_into(&x, &mut y_into);

        for i in 0..2 {
            assert_relative_eq!(y_into[i].re, y_alloc[i].re, epsilon = 1e-14);
            assert_relative_eq!(y_into[i].im, y_alloc[i].im, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_csr_from_triplets() {
        let triplets = vec![
            (0, 0, Complex64::new(1.0, 0.0)),
            (0, 2, Complex64::new(2.0, 0.0)),
            (1, 1, Complex64::new(3.0, 0.0)),
            (2, 0, Complex64::new(4.0, 0.0)),
            (2, 2, Complex64::new(5.0, 0.0)),
        ];

        let csr = CsrMatrix::from_triplets(3, 3, triplets);

        assert_eq!(csr.nnz(), 5);
        assert_relative_eq!(csr.get(0, 0).re, 1.0);
        assert_relative_eq!(csr.get(1, 1).re, 3.0);
    }

    #[test]
    fn test_csr_triplets_duplicate() {
        let triplets = vec![
            (0, 0, Complex64::new(1.0, 0.0)),
            (0, 0, Complex64::new(2.0, 0.0)), // Duplicate!
            (1, 1, Complex64::new(3.0, 0.0)),
        ];

        let csr = CsrMatrix::from_triplets(2, 2, triplets);

        assert_relative_eq!(csr.get(0, 0).re, 3.0); // 1 + 2 = 3
    }

    fn path_matrix_in_label_order(order: &[usize]) -> CsrMatrix<f64> {
        let n = order.len();
        let mut triplets = (0..n).map(|i| (i, i, 2.0_f64)).collect::<Vec<_>>();
        for edge in order.windows(2) {
            let a = edge[0];
            let b = edge[1];
            triplets.push((a, b, -1.0));
            triplets.push((b, a, -1.0));
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    fn vector_in_new_order(values: &[f64], new_to_old: &[usize]) -> Vec<f64> {
        new_to_old.iter().map(|&old| values[old]).collect()
    }

    fn assert_valid_permutation(permutation: &[usize]) {
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..permutation.len()).collect::<Vec<_>>());
    }

    fn assert_symmetric_permute_preserves_matvec(csr: &CsrMatrix<f64>, permutation: &[usize]) {
        let reordered = csr.symmetric_permute(permutation);
        let x_old = (0..csr.num_cols)
            .map(|i| 1.0_f64 + 0.5 * i as f64)
            .collect::<Vec<_>>();
        let x_new = vector_in_new_order(&x_old, permutation);
        let y_old = csr.matvec(&x_old);
        let y_new_expected = vector_in_new_order(&y_old, permutation);
        let y_new = reordered.matvec(&x_new);

        for i in 0..y_new.len() {
            assert_relative_eq!(y_new[i], y_new_expected[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_rcm_reduces_path_bandwidth() {
        let csr = path_matrix_in_label_order(&[0, 4, 1, 5, 2, 6, 3, 7]);
        let original_bandwidth = csr.bandwidth();
        let (reordered, permutation) = csr.reverse_cuthill_mckee_permuted();

        assert_valid_permutation(&permutation);
        assert!(reordered.bandwidth() < original_bandwidth);
        assert_eq!(reordered.bandwidth(), 1);
    }

    #[test]
    fn test_symmetric_permute_preserves_matvec_under_reordered_vectors() {
        let csr = path_matrix_in_label_order(&[0, 4, 1, 5, 2, 6, 3, 7]);
        let permutation = csr.reverse_cuthill_mckee_ordering();
        let reordered = csr.symmetric_permute(&permutation);
        let x_old = (0..csr.num_cols)
            .map(|i| 1.0_f64 + 0.5 * i as f64)
            .collect::<Vec<_>>();
        let x_new = vector_in_new_order(&x_old, &permutation);
        let y_old = csr.matvec(&x_old);
        let y_new_expected = vector_in_new_order(&y_old, &permutation);
        let y_new = reordered.matvec(&x_new);

        for i in 0..y_new.len() {
            assert_relative_eq!(y_new[i], y_new_expected[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_nested_dissection_permuted_preserves_matvec() {
        let csr = path_matrix_in_label_order(&[0, 4, 1, 5, 2, 6, 3, 7]);
        let (reordered, permutation) = csr.nested_dissection_permuted(2, Some(3));

        assert_valid_permutation(&permutation);
        assert_eq!(reordered.num_rows, csr.num_rows);
        assert_eq!(reordered.num_cols, csr.num_cols);
        assert_eq!(reordered.nnz(), csr.nnz());
        assert_symmetric_permute_preserves_matvec(&csr, &permutation);
    }

    fn block_path_matrix(block_count: usize, block_size: usize) -> CsrMatrix<f64> {
        let n = block_count * block_size;
        let mut triplets = Vec::new();
        for block in 0..block_count {
            let block_start = block * block_size;
            for local_row in 0..block_size {
                let row = block_start + local_row;
                triplets.push((row, row, 4.0));
                for local_col in 0..block_size {
                    if local_row != local_col {
                        triplets.push((row, block_start + local_col, 0.25));
                    }
                }
            }
        }
        for block in 0..block_count - 1 {
            for local in 0..block_size {
                let a = block * block_size + local;
                let b = (block + 1) * block_size + local;
                triplets.push((a, b, -1.0));
                triplets.push((b, a, -1.0));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    #[test]
    fn test_block_node_nested_dissection_keeps_node_dofs_contiguous() {
        let block_size = 2;
        let csr = block_path_matrix(4, block_size);
        let (reordered, permutation) =
            csr.block_node_nested_dissection_permuted(block_size, 1, Some(3));

        assert_valid_permutation(&permutation);
        assert_eq!(reordered.nnz(), csr.nnz());
        for chunk in permutation.chunks(block_size) {
            assert_eq!(chunk.len(), block_size);
            assert!(
                chunk
                    .windows(2)
                    .all(|pair| pair[0] / block_size == pair[1] / block_size)
            );
        }
        assert_symmetric_permute_preserves_matvec(&csr, &permutation);
    }

    #[test]
    fn test_csr_identity() {
        let id: CsrMatrix<Complex64> = CsrMatrix::identity(3);

        assert_eq!(id.nnz(), 3);
        assert_relative_eq!(id.get(0, 0).re, 1.0);
        assert_relative_eq!(id.get(1, 1).re, 1.0);
        assert_relative_eq!(id.get(2, 2).re, 1.0);
        assert_relative_eq!(id.get(0, 1).norm(), 0.0);
    }

    #[test]
    fn test_csr_builder() {
        let mut builder: CsrBuilder<Complex64> = CsrBuilder::new(3, 3);

        builder.add_row_entries(
            [(0, Complex64::new(1.0, 0.0)), (2, Complex64::new(2.0, 0.0))].into_iter(),
        );
        builder.add_row_entries([(1, Complex64::new(3.0, 0.0))].into_iter());
        builder.add_row_entries(
            [(0, Complex64::new(4.0, 0.0)), (2, Complex64::new(5.0, 0.0))].into_iter(),
        );

        let csr = builder.finish();

        assert_eq!(csr.nnz(), 5);
        assert_relative_eq!(csr.get(0, 0).re, 1.0);
        assert_relative_eq!(csr.get(1, 1).re, 3.0);
    }

    #[test]
    fn test_csr_to_dense_roundtrip() {
        let original = dense2(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        );

        let csr = CsrMatrix::from_dense(&original, 1e-15);
        let recovered = csr.to_dense();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    (original[[i, j]] - recovered[[i, j]]).norm(),
                    0.0,
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_linear_operator_impl() {
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        // Test via LinearOperator trait
        let y = csr.apply(&x);
        assert_relative_eq!(y[0].re, 5.0, epsilon = 1e-10);
        assert_relative_eq!(y[1].re, 11.0, epsilon = 1e-10);

        assert!(csr.is_square());
        assert_eq!(csr.num_rows(), 2);
        assert_eq!(csr.num_cols(), 2);
    }

    #[test]
    fn test_f64_csr() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![1.0_f64, 2.0];

        let y = csr.matvec(&x);
        assert_relative_eq!(y[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_csr_empty_matrix() {
        // 0x0 matrix should not panic on basic operations
        let csr: CsrMatrix<f64> = CsrMatrix::new(0, 0);
        assert_eq!(csr.num_rows, 0);
        assert_eq!(csr.num_cols, 0);
        assert_eq!(csr.nnz(), 0);

        let dense = csr.to_dense();
        assert_eq!(dense.shape(), (0, 0));
    }

    #[test]
    fn test_csr_rectangular() {
        // 3x5 rectangular matrix matvec
        let mut dense = DenseMatrix::from_elem(3, 5, Complex64::new(0.0, 0.0));
        dense[[0, 0]] = Complex64::new(1.0, 0.0);
        dense[[0, 2]] = Complex64::new(2.0, 0.0);
        dense[[1, 1]] = Complex64::new(3.0, 0.0);
        dense[[1, 4]] = Complex64::new(4.0, 0.0);
        dense[[2, 3]] = Complex64::new(5.0, 0.0);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        assert_eq!(csr.num_rows, 3);
        assert_eq!(csr.num_cols, 5);

        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ];

        let y = csr.matvec(&x);
        // row 0: 1*1 + 2*3 = 7
        // row 1: 3*2 + 4*5 = 26
        // row 2: 5*4 = 20
        assert_relative_eq!(y[0].re, 7.0, epsilon = 1e-10);
        assert_relative_eq!(y[1].re, 26.0, epsilon = 1e-10);
        assert_relative_eq!(y[2].re, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_csr_parallel_transpose_and_hermitian_match_serial() {
        let rows = 1_400;
        let cols = 1_024;
        let entries_per_row = 40;
        let mut triplets = Vec::with_capacity(rows * entries_per_row);

        for i in 0..rows {
            for k in 0..entries_per_row {
                let j = (i * 37 + k * 17) % cols;
                let scale = ((i + 1 + k) % 11) as f64 + 1.0;
                triplets.push((i, j, Complex64::new(scale * 0.125, -scale * 0.0625)));
            }
        }

        let csr = CsrMatrix::from_triplets(rows, cols, triplets);
        let x = (0..rows)
            .map(|i| Complex64::new((i as f64 * 0.01).sin(), (i as f64 * 0.02).cos()))
            .collect::<Vec<_>>();

        let mut transpose_serial = vec![Complex64::new(0.0, 0.0); cols];
        let mut transpose_parallel = vec![Complex64::new(0.0, 0.0); cols];
        csr.matvec_transpose_sequential_into::<false>(&x, &mut transpose_serial);
        csr.matvec_transpose_parallel_into::<false>(&x, &mut transpose_parallel);

        for (serial, parallel) in transpose_serial.iter().zip(&transpose_parallel) {
            assert_relative_eq!(serial.re, parallel.re, epsilon = 1e-10);
            assert_relative_eq!(serial.im, parallel.im, epsilon = 1e-10);
        }

        let mut hermitian_serial = vec![Complex64::new(0.0, 0.0); cols];
        let mut hermitian_parallel = vec![Complex64::new(0.0, 0.0); cols];
        csr.matvec_transpose_sequential_into::<true>(&x, &mut hermitian_serial);
        csr.matvec_transpose_parallel_into::<true>(&x, &mut hermitian_parallel);

        for (serial, parallel) in hermitian_serial.iter().zip(&hermitian_parallel) {
            assert_relative_eq!(serial.re, parallel.re, epsilon = 1e-10);
            assert_relative_eq!(serial.im, parallel.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_csr_matvec_add() {
        // Verify y += Ax accumulation
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        // Pre-fill y with known values
        let mut y = vec![Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)];
        csr.matvec_add(&x, &mut y);

        // y should be old_y + Ax = [10 + 5, 20 + 11] = [15, 31]
        assert_relative_eq!(y[0].re, 15.0, epsilon = 1e-10);
        assert_relative_eq!(y[1].re, 31.0, epsilon = 1e-10);
    }

    #[test]
    fn test_csr_matmul() {
        // Verify A*B against dense result on small matrices
        let a_dense = dense2(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let b_dense = dense2(
            2,
            2,
            vec![
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
            ],
        );

        let a = CsrMatrix::from_dense(&a_dense, 1e-15);
        let b = CsrMatrix::from_dense(&b_dense, 1e-15);

        let c = a.matmul(&b);
        let c_dense = c.to_dense();

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_relative_eq!(c_dense[[0, 0]].re, 19.0, epsilon = 1e-10);
        assert_relative_eq!(c_dense[[0, 1]].re, 22.0, epsilon = 1e-10);
        assert_relative_eq!(c_dense[[1, 0]].re, 43.0, epsilon = 1e-10);
        assert_relative_eq!(c_dense[[1, 1]].re, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_csr_matmul_exact_keeps_tiny_entries() {
        // A (1e-9) * B (1e-9) = C (1e-18): a general-purpose SpGEMM must
        // keep the tiny entry. Only the explicit opt-in `matmul_with_drop`
        // may threshold it away.
        let a = CsrMatrix::from_triplets(2, 2, vec![(0, 0, Complex64::new(1e-9, 0.0))]);
        let b = CsrMatrix::from_triplets(2, 2, vec![(0, 1, Complex64::new(1e-9, 0.0))]);

        let exact = a.matmul(&b);
        assert_relative_eq!(exact.get(0, 1).re, 1e-18, epsilon = 1e-30);
        assert_eq!(exact.nnz(), 1, "exact matmul must keep the tiny entry");

        let dropped = a.matmul_with_drop(&b, 1e-15);
        assert_eq!(
            dropped.nnz(),
            0,
            "matmul_with_drop(1e-15) must drop the tiny entry"
        );

        // A drop tolerance of zero is the exact product.
        let exact_via_drop = a.matmul_with_drop(&b, 0.0);
        assert_eq!(exact_via_drop.nnz(), 1);
    }

    #[test]
    fn test_csr_matmul_exact_omits_numerical_cancellation() {
        // Contributions that cancel to an exact zero carry no information
        // and are omitted even from the exact product.
        let a = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(1.0, 0.0)),
                (0, 1, Complex64::new(1.0, 0.0)),
            ],
        );
        let b = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(1.0, 0.0)),
                (1, 0, Complex64::new(-1.0, 0.0)),
            ],
        );
        let c = a.matmul(&b);
        assert_eq!(c.nnz(), 0, "exact cancellation should not emit an entry");
    }

    #[test]
    fn test_csr_matvec_huge_cols_f64() {
        // Regression: AVX2 gather uses i32 indices, which overflow for
        // matrices with >= 2^31 columns.  Verify we fall back to scalar.
        let num_cols = (i32::MAX as usize) + 1;
        let row_ptrs = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0_f64, 2.0];
        let a = CsrMatrix::from_raw_parts(2, num_cols, row_ptrs, col_indices, values);

        let x = vec![1.0_f64; num_cols];
        let mut y = vec![0.0_f64; 2];
        a.matvec_into_buf(&x, &mut y);

        assert_eq!(y[0], 1.0, "Row 0: col 0 * x[0] = 1*1");
        assert_eq!(y[1], 2.0, "Row 1: col 1 * x[1] = 2*1");
    }

    #[test]
    fn test_csr_matvec_huge_cols_f32() {
        let num_cols = (i32::MAX as usize) + 1;
        let row_ptrs = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0_f32, 2.0];
        let a = CsrMatrix::from_raw_parts(2, num_cols, row_ptrs, col_indices, values);

        let x = vec![1.0_f32; num_cols];
        let mut y = vec![0.0_f32; 2];
        a.matvec_into_buf(&x, &mut y);

        assert_eq!(y[0], 1.0, "Row 0: col 0 * x[0] = 1*1");
        assert_eq!(y[1], 2.0, "Row 1: col 1 * x[1] = 2*1");
    }

    /// Skewed matrix with row lengths varying 1..=1000, heavy rows clustered
    /// at the top — the straggler case for equal-rows parallel partitioning.
    fn skewed_f64_matrix(rows: usize) -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for i in 0..rows {
            let deg = 1 + (i * 7) % 1000;
            for k in 0..deg {
                let col = (i + k * 37) % rows;
                triplets.push((i, col, (i % 13) as f64 + (k % 7) as f64 * 0.25 + 1.0));
            }
        }
        CsrMatrix::from_triplets(rows, rows, triplets)
    }

    #[test]
    fn test_matvec_parallel_skewed_matches_sequential() {
        let rows = 4_096;
        let a = skewed_f64_matrix(rows);
        assert!(a.nnz() >= 50_000, "must exercise the parallel path");

        let x: Vec<f64> = (0..rows).map(|i| (i as f64).sin()).collect();
        let mut y_par = vec![0.0; rows];
        let mut y_seq = vec![0.0; rows];
        a.matvec_parallel_into(&x, &mut y_par);
        a.matvec_sequential_into(&x, &mut y_seq);

        for i in 0..rows {
            assert_relative_eq!(y_par[i], y_seq[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_matvec_transpose_parallel_skewed_matches_sequential() {
        let rows = 4_096;
        let a = skewed_f64_matrix(rows);
        assert!(a.nnz() >= crate::spmv_plan::par_nnz_threshold());

        let x: Vec<f64> = (0..rows).map(|i| (i as f64).cos()).collect();
        let mut y_par = vec![0.0; rows];
        let mut y_seq = vec![0.0; rows];
        a.matvec_transpose_parallel_into::<false>(&x, &mut y_par);
        a.matvec_transpose_sequential_into::<false>(&x, &mut y_seq);

        for i in 0..rows {
            assert_relative_eq!(y_par[i], y_seq[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_matvec_hermitian_parallel_skewed_matches_sequential() {
        let rows = 4_096;
        let real = skewed_f64_matrix(rows);
        let triplets: Vec<(usize, usize, Complex64)> = real
            .values
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                // Rebuild (row, col) from the CSR layout.
                let row = real.row_ptrs.partition_point(|&p| p <= idx) - 1;
                (row, real.col_indices[idx], Complex64::new(v, -0.5 * v))
            })
            .collect();
        let a = CsrMatrix::from_triplets(rows, rows, triplets);

        let x: Vec<Complex64> = (0..rows)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut y_par = vec![Complex64::new(0.0, 0.0); rows];
        let mut y_seq = vec![Complex64::new(0.0, 0.0); rows];
        a.matvec_transpose_parallel_into::<true>(&x, &mut y_par);
        a.matvec_transpose_sequential_into::<true>(&x, &mut y_seq);

        for i in 0..rows {
            assert_relative_eq!(y_par[i].re, y_seq[i].re, epsilon = 1e-8);
            assert_relative_eq!(y_par[i].im, y_seq[i].im, epsilon = 1e-8);
        }
    }

    // ------------------------------------------------------------------
    // SpMM (multi-vector) tests
    // ------------------------------------------------------------------

    /// Rectangular matrix with empty rows: 60×40, rows divisible by 3 are
    /// empty, others carry 1-4 entries.
    fn spmm_test_matrix_f64() -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for row in 0..60usize {
            if row % 3 == 0 {
                continue; // empty row
            }
            let deg = row % 4 + 1;
            for d in 0..deg {
                let col = (row * 7 + d * 13) % 40;
                triplets.push((row, col, 0.5 + (row * d) as f64 % 3.0));
            }
        }
        CsrMatrix::from_triplets(60, 40, triplets)
    }

    /// Reference: per-column matvec into a row-major n×k block.
    fn block_from_columns<T: ComplexField>(cols: &[Vec<T>], n: usize) -> DenseMatrix<T> {
        let k = cols.len();
        let mut out = DenseMatrix::from_elem(n, k, T::zero());
        for (j, col) in cols.iter().enumerate() {
            out.set_column(j, col);
        }
        out
    }

    #[test]
    fn test_spmm_parity_f64() {
        let a = spmm_test_matrix_f64();
        for k in [1usize, 2, 3, 4, 5, 8, 9] {
            let x = DenseMatrix::from_fn(40, k, |i, j| ((i * 31 + j * 17) % 19) as f64 - 9.0);
            let mut y = DenseMatrix::from_elem(60, k, 0.0f64);
            a.spmm_into(&x, &mut y);

            let reference: Vec<Vec<f64>> = (0..k).map(|j| a.matvec(&x.column_to_vec(j))).collect();
            let y_ref = block_from_columns(&reference, 60);
            for i in 0..60 {
                for j in 0..k {
                    assert_relative_eq!(y[[i, j]], y_ref[[i, j]], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_spmm_parity_complex64() {
        let real = spmm_test_matrix_f64();
        let triplets: Vec<(usize, usize, Complex64)> = real
            .values
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                let row = real.row_ptrs.partition_point(|&p| p <= idx) - 1;
                (row, real.col_indices[idx], Complex64::new(v, -0.25 * v))
            })
            .collect();
        let a = CsrMatrix::from_triplets(60, 40, triplets);

        for k in [1usize, 2, 4, 8] {
            let x = DenseMatrix::from_fn(40, k, |i, j| {
                Complex64::new(
                    ((i * 31 + j * 17) % 19) as f64 - 9.0,
                    ((i * 13 + j * 7) % 11) as f64 - 5.0,
                )
            });
            let mut y = DenseMatrix::from_elem(60, k, Complex64::new(0.0, 0.0));
            a.spmm_into(&x, &mut y);

            let reference: Vec<Vec<Complex64>> =
                (0..k).map(|j| a.matvec(&x.column_to_vec(j))).collect();
            let y_ref = block_from_columns(&reference, 60);
            for i in 0..60 {
                for j in 0..k {
                    assert_relative_eq!(y[[i, j]].re, y_ref[[i, j]].re, epsilon = 1e-11);
                    assert_relative_eq!(y[[i, j]].im, y_ref[[i, j]].im, epsilon = 1e-11);
                }
            }
        }
    }

    #[test]
    fn test_spmm_parallel_matches_per_column_spmv() {
        // Large enough to take the rayon path: nnz·k >= 50_000, rows >= 4_000.
        let rows = 4_096;
        let a = skewed_f64_matrix(rows);
        assert!(a.nnz() * 2 >= 50_000 && a.num_rows >= 4_000);

        for k in [2usize, 4, 8] {
            let x = DenseMatrix::from_fn(rows, k, |i, j| ((i * 31 + j * 17) % 23) as f64 - 11.0);
            let mut y = DenseMatrix::from_elem(rows, k, 0.0f64);
            a.spmm_into(&x, &mut y);

            let reference: Vec<Vec<f64>> = (0..k).map(|j| a.matvec(&x.column_to_vec(j))).collect();
            let y_ref = block_from_columns(&reference, rows);
            // The parallel kernel changes the summation order per row only
            // through partitioning (rows stay whole), so parity is tight.
            for i in 0..rows {
                for j in 0..k {
                    assert_relative_eq!(y[[i, j]], y_ref[[i, j]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_spmm_parallel_complex64_matches_per_column_spmv() {
        let rows = 4_096;
        let real = skewed_f64_matrix(rows);
        let triplets: Vec<(usize, usize, Complex64)> = real
            .values
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                let row = real.row_ptrs.partition_point(|&p| p <= idx) - 1;
                (row, real.col_indices[idx], Complex64::new(v, 0.5 * v))
            })
            .collect();
        let a = CsrMatrix::from_triplets(rows, rows, triplets);

        let k = 4;
        let x = DenseMatrix::from_fn(rows, k, |i, j| {
            Complex64::new(
                ((i * 31 + j * 17) % 23) as f64 - 11.0,
                ((i * 7 + j * 13) % 17) as f64 - 8.0,
            )
        });
        let mut y = DenseMatrix::from_elem(rows, k, Complex64::new(0.0, 0.0));
        a.spmm_into(&x, &mut y);

        let reference: Vec<Vec<Complex64>> =
            (0..k).map(|j| a.matvec(&x.column_to_vec(j))).collect();
        let y_ref = block_from_columns(&reference, rows);
        for i in 0..rows {
            for j in 0..k {
                assert_relative_eq!(y[[i, j]].re, y_ref[[i, j]].re, epsilon = 1e-9);
                assert_relative_eq!(y[[i, j]].im, y_ref[[i, j]].im, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_csr_apply_block_matches_per_column_apply() {
        // Trait path: LinearOperator::apply_block must route through SpMM and
        // agree with per-column apply for both small (sequential) and large
        // (parallel) matrices.
        let a = spmm_test_matrix_f64();
        let k = 3;
        let x = DenseMatrix::from_fn(40, k, |i, j| (i as f64).sin() + j as f64);
        let mut y = DenseMatrix::from_elem(60, k, 0.0f64);
        LinearOperator::apply_block(&a, &x, &mut y);
        for j in 0..k {
            let yj = LinearOperator::apply(&a, &x.column_to_vec(j));
            for i in 0..60 {
                assert_relative_eq!(y[[i, j]], yj[i], epsilon = 1e-12);
            }
        }

        let rows = 4_096;
        let big = skewed_f64_matrix(rows);
        let x = DenseMatrix::from_fn(rows, 2, |i, j| ((i + j * 7) % 13) as f64 - 6.0);
        let mut y = DenseMatrix::from_elem(rows, 2, 0.0f64);
        LinearOperator::apply_block(&big, &x, &mut y);
        for j in 0..2 {
            let yj = LinearOperator::apply(&big, &x.column_to_vec(j));
            for i in 0..rows {
                assert_relative_eq!(y[[i, j]], yj[i], epsilon = 1e-10);
            }
        }
    }
}
