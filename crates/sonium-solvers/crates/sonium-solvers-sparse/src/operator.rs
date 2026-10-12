//! Matrix-free operator combinators
//!
//! Building blocks for constructing operators without assembling a global matrix.
//! All combinators implement [`LinearOperator<T>`] and can be used with any Krylov solver.

use crate::CsrMatrix;
use num_traits::Float;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, LinearOperator, SolverError};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn combine_adjoint_capabilities(
    left: AdjointCapability,
    right: AdjointCapability,
) -> AdjointCapability {
    if left == AdjointCapability::Unsupported || right == AdjointCapability::Unsupported {
        AdjointCapability::Unsupported
    } else if left == AdjointCapability::Unknown || right == AdjointCapability::Unknown {
        AdjointCapability::Unknown
    } else {
        AdjointCapability::Exact
    }
}

fn reject_non_finite<T: ComplexField>(
    operation: &'static str,
    values: &[T],
) -> Result<(), SolverError> {
    if values.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "operator callback produced a non-finite value".into(),
        });
    }
    Ok(())
}

fn callback_error(operation: &'static str) -> SolverError {
    SolverError::OperationFailed {
        operation,
        reason: "operator callback panicked".into(),
    }
}

/// Sum of two operators: `(A + B) x = A x + B x`
pub struct SumOperator<'a, T: ComplexField> {
    a: &'a dyn LinearOperator<T>,
    b: &'a dyn LinearOperator<T>,
}

impl<'a, T: ComplexField> SumOperator<'a, T> {
    /// Checked constructor for a sum of shape-compatible operators.
    pub fn try_new(
        a: &'a dyn LinearOperator<T>,
        b: &'a dyn LinearOperator<T>,
    ) -> Result<Self, SolverError> {
        if a.num_rows() != b.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_rows(),
                got: b.num_rows(),
            });
        }
        if a.num_cols() != b.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_cols(),
                got: b.num_cols(),
            });
        }
        Ok(Self { a, b })
    }

    /// Compatibility constructor preserving the historical panic boundary.
    pub fn new(a: &'a dyn LinearOperator<T>, b: &'a dyn LinearOperator<T>) -> Self {
        Self::try_new(a, b)
            .unwrap_or_else(|error| panic!("SumOperator dimension mismatch: {error}"))
    }

    /// Checked forward sum that contains callback failures and validates
    /// generated values before committing them to `y`.
    pub fn try_apply_into(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        if x.len() != self.a.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.a.num_cols(),
                got: x.len(),
            });
        }
        if y.len() != self.a.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.a.num_rows(),
                got: y.len(),
            });
        }
        reject_non_finite("SumOperator forward input", x)?;
        let mut ax = vec![T::zero(); y.len()];
        let mut bx = vec![T::zero(); y.len()];
        catch_unwind(AssertUnwindSafe(|| self.a.apply_into(x, &mut ax)))
            .map_err(|_| callback_error("SumOperator left apply"))?;
        reject_non_finite("SumOperator left output", &ax)?;
        catch_unwind(AssertUnwindSafe(|| self.b.apply_into(x, &mut bx)))
            .map_err(|_| callback_error("SumOperator right apply"))?;
        reject_non_finite("SumOperator right output", &bx)?;
        for (yi, (ai, bi)) in y.iter_mut().zip(ax.iter().zip(bx.iter())) {
            *yi = *ai + *bi;
        }
        reject_non_finite("SumOperator forward output", y)
    }

    /// Checked transpose sum that requires exact transpose actions.
    pub fn try_apply_transpose_into(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        if x.len() != self.a.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.a.num_rows(),
                got: x.len(),
            });
        }
        if y.len() != self.a.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.a.num_cols(),
                got: y.len(),
            });
        }
        if self.a.transpose_capability() != AdjointCapability::Exact
            || self.b.transpose_capability() != AdjointCapability::Exact
        {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "SumOperator transpose",
                operator: "summed operator",
            });
        }
        reject_non_finite("SumOperator transpose input", x)?;
        let mut ax = vec![T::zero(); y.len()];
        let mut bx = vec![T::zero(); y.len()];
        catch_unwind(AssertUnwindSafe(|| self.a.apply_transpose_into(x, &mut ax)))
            .map_err(|_| callback_error("SumOperator left transpose"))?;
        reject_non_finite("SumOperator left transpose output", &ax)?;
        catch_unwind(AssertUnwindSafe(|| self.b.apply_transpose_into(x, &mut bx)))
            .map_err(|_| callback_error("SumOperator right transpose"))?;
        reject_non_finite("SumOperator right transpose output", &bx)?;
        for (yi, (ai, bi)) in y.iter_mut().zip(ax.iter().zip(bx.iter())) {
            *yi = *ai + *bi;
        }
        reject_non_finite("SumOperator transpose output", y)
    }
}

impl<T: ComplexField> LinearOperator<T> for SumOperator<'_, T> {
    fn num_rows(&self) -> usize {
        self.a.num_rows()
    }
    fn num_cols(&self) -> usize {
        self.a.num_cols()
    }
    fn apply_into(&self, x: &[T], y: &mut [T]) {
        self.a.apply_into(x, y);
        let mut bx = vec![T::zero(); self.b.num_rows()];
        self.b.apply_into(x, &mut bx);
        for (yi, bi) in y.iter_mut().zip(bx.iter()) {
            *yi += *bi;
        }
    }
    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.a.apply_transpose_into(x, y);
        let mut bx = vec![T::zero(); self.b.num_cols()];
        self.b.apply_transpose_into(x, &mut bx);
        for (yi, bi) in y.iter_mut().zip(bx.iter()) {
            *yi += *bi;
        }
    }
    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        let mut ax = self.a.apply_hermitian(x);
        let bx = self.b.apply_hermitian(x);
        for (ai, bi) in ax.iter_mut().zip(bx.iter()) {
            *ai += *bi;
        }
        ax
    }

    fn transpose_capability(&self) -> AdjointCapability {
        combine_adjoint_capabilities(self.a.transpose_capability(), self.b.transpose_capability())
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        combine_adjoint_capabilities(self.a.hermitian_capability(), self.b.hermitian_capability())
    }
}

/// Scalar-scaled operator: `(alpha * A) x = alpha * (A x)`
pub struct ScaledOperator<'a, T: ComplexField> {
    op: &'a dyn LinearOperator<T>,
    alpha: T,
}

impl<'a, T: ComplexField> ScaledOperator<'a, T> {
    pub fn new(op: &'a dyn LinearOperator<T>, alpha: T) -> Self {
        Self { op, alpha }
    }
}

impl<T: ComplexField> LinearOperator<T> for ScaledOperator<'_, T> {
    fn num_rows(&self) -> usize {
        self.op.num_rows()
    }
    fn num_cols(&self) -> usize {
        self.op.num_cols()
    }
    fn apply_into(&self, x: &[T], y: &mut [T]) {
        self.op.apply_into(x, y);
        for v in y.iter_mut() {
            *v = self.alpha * *v;
        }
    }
    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.op.apply_transpose_into(x, y);
        for v in y.iter_mut() {
            *v = self.alpha * *v;
        }
    }
    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        let mut y = self.op.apply_hermitian(x);
        let alpha_conj = self.alpha.conj();
        for v in y.iter_mut() {
            *v = alpha_conj * *v;
        }
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.op.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.op.hermitian_capability()
    }
}

/// Composition of two operators: `(A * B) x = A(B(x))`
pub struct CompositeOperator<'a, T: ComplexField> {
    outer: &'a dyn LinearOperator<T>,
    inner: &'a dyn LinearOperator<T>,
}

impl<'a, T: ComplexField> CompositeOperator<'a, T> {
    /// Checked constructor for a dimension-compatible operator composition.
    pub fn try_new(
        outer: &'a dyn LinearOperator<T>,
        inner: &'a dyn LinearOperator<T>,
    ) -> Result<Self, SolverError> {
        if outer.num_cols() != inner.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: outer.num_cols(),
                got: inner.num_rows(),
            });
        }
        Ok(Self { outer, inner })
    }

    /// Compatibility constructor preserving the historical panic boundary.
    pub fn new(outer: &'a dyn LinearOperator<T>, inner: &'a dyn LinearOperator<T>) -> Self {
        Self::try_new(outer, inner)
            .unwrap_or_else(|error| panic!("CompositeOperator dimension mismatch: {error}"))
    }

    /// Number of entries needed for reusable composition scratch.
    pub fn scratch_len(&self) -> usize {
        self.inner.num_rows()
    }

    /// Apply `(outer * inner) x` using caller-provided scratch for `inner * x`.
    pub fn apply_into_with_scratch(&self, x: &[T], y: &mut [T], scratch: &mut [T]) {
        self.try_apply_into_with_scratch(x, y, scratch)
            .unwrap_or_else(|error| panic!("CompositeOperator checked apply failed: {error}"));
    }

    /// Checked forward composition using caller-provided scratch storage.
    pub fn try_apply_into_with_scratch(
        &self,
        x: &[T],
        y: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SolverError> {
        if x.len() != self.inner.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.inner.num_cols(),
                got: x.len(),
            });
        }
        if y.len() != self.outer.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.outer.num_rows(),
                got: y.len(),
            });
        }
        if scratch.len() < self.scratch_len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.scratch_len(),
                got: scratch.len(),
            });
        }
        reject_non_finite("CompositeOperator forward input", x)?;
        let bx = &mut scratch[..self.scratch_len()];
        bx.fill(T::zero());
        y.fill(T::zero());
        catch_unwind(AssertUnwindSafe(|| self.inner.apply_into(x, bx)))
            .map_err(|_| callback_error("CompositeOperator inner apply"))?;
        reject_non_finite("CompositeOperator inner output", bx)?;
        catch_unwind(AssertUnwindSafe(|| self.outer.apply_into(bx, y)))
            .map_err(|_| callback_error("CompositeOperator outer apply"))?;
        reject_non_finite("CompositeOperator forward output", y)?;
        Ok(())
    }

    /// Apply `(outer * inner)^T x` using caller-provided scratch for `outer^T * x`.
    pub fn apply_transpose_into_with_scratch(&self, x: &[T], y: &mut [T], scratch: &mut [T]) {
        self.try_apply_transpose_into_with_scratch(x, y, scratch)
            .unwrap_or_else(|error| panic!("CompositeOperator checked transpose failed: {error}"));
    }

    /// Checked transpose composition using caller-provided scratch storage.
    pub fn try_apply_transpose_into_with_scratch(
        &self,
        x: &[T],
        y: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SolverError> {
        if x.len() != self.outer.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.outer.num_rows(),
                got: x.len(),
            });
        }
        if y.len() != self.inner.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.inner.num_cols(),
                got: y.len(),
            });
        }
        if scratch.len() < self.scratch_len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.scratch_len(),
                got: scratch.len(),
            });
        }
        if self.outer.transpose_capability() != AdjointCapability::Exact
            || self.inner.transpose_capability() != AdjointCapability::Exact
        {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "CompositeOperator transpose",
                operator: "composed operator",
            });
        }
        reject_non_finite("CompositeOperator transpose input", x)?;
        let atx = &mut scratch[..self.scratch_len()];
        atx.fill(T::zero());
        y.fill(T::zero());
        catch_unwind(AssertUnwindSafe(|| self.outer.apply_transpose_into(x, atx)))
            .map_err(|_| callback_error("CompositeOperator outer transpose"))?;
        reject_non_finite("CompositeOperator outer transpose output", atx)?;
        catch_unwind(AssertUnwindSafe(|| self.inner.apply_transpose_into(atx, y)))
            .map_err(|_| callback_error("CompositeOperator inner transpose"))?;
        reject_non_finite("CompositeOperator transpose output", y)?;
        Ok(())
    }
}

impl<T: ComplexField> LinearOperator<T> for CompositeOperator<'_, T> {
    fn num_rows(&self) -> usize {
        self.outer.num_rows()
    }
    fn num_cols(&self) -> usize {
        self.inner.num_cols()
    }
    fn apply_into(&self, x: &[T], y: &mut [T]) {
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.apply_into_with_scratch(x, y, &mut scratch);
    }
    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        // (AB)^T = B^T A^T
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.apply_transpose_into_with_scratch(x, y, &mut scratch);
    }
    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        // (AB)^H = B^H A^H
        let ahx = self.outer.apply_hermitian(x);
        self.inner.apply_hermitian(&ahx)
    }

    fn transpose_capability(&self) -> AdjointCapability {
        combine_adjoint_capabilities(
            self.outer.transpose_capability(),
            self.inner.transpose_capability(),
        )
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        combine_adjoint_capabilities(
            self.outer.hermitian_capability(),
            self.inner.hermitian_capability(),
        )
    }
}

/// Callback-based operator for fully custom matrix-free apply.
#[allow(clippy::type_complexity)]
pub struct FnOperator<T: ComplexField> {
    rows: usize,
    cols: usize,
    apply_into_fn: FnAction<T>,
    transpose_into_fn: Option<FnAction<T>>,
    hermitian_into_fn: Option<FnAction<T>>,
}

type FnAction<T> = Box<dyn Fn(&[T], &mut [T]) + Send + Sync>;

impl<T: ComplexField> FnOperator<T> {
    pub fn new(
        rows: usize,
        cols: usize,
        apply_fn: impl Fn(&[T]) -> Vec<T> + Send + Sync + 'static,
    ) -> Self {
        Self::new_into(rows, cols, move |x, y| {
            let result = apply_fn(x);
            assert_eq!(result.len(), y.len(), "FnOperator output length mismatch");
            y.copy_from_slice(&result);
        })
    }

    pub fn new_into(
        rows: usize,
        cols: usize,
        apply_into_fn: impl Fn(&[T], &mut [T]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            rows,
            cols,
            apply_into_fn: Box::new(apply_into_fn),
            transpose_into_fn: None,
            hermitian_into_fn: None,
        }
    }

    /// Checked forward callback application without allocating an output.
    pub fn try_apply_into(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        if x.len() != self.cols {
            return Err(SolverError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }
        if y.len() != self.rows {
            return Err(SolverError::DimensionMismatch {
                expected: self.rows,
                got: y.len(),
            });
        }
        reject_non_finite("FnOperator forward input", x)?;
        y.fill(T::zero());
        catch_unwind(AssertUnwindSafe(|| (self.apply_into_fn)(x, y)))
            .map_err(|_| callback_error("FnOperator forward callback"))?;
        reject_non_finite("FnOperator forward output", y)
    }

    /// Checked transpose callback application without allocating an output.
    pub fn try_apply_transpose_into(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        if x.len() != self.rows {
            return Err(SolverError::DimensionMismatch {
                expected: self.rows,
                got: x.len(),
            });
        }
        if y.len() != self.cols {
            return Err(SolverError::DimensionMismatch {
                expected: self.cols,
                got: y.len(),
            });
        }
        if self.transpose_into_fn.is_none() {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "FnOperator transpose",
                operator: "forward-only callback operator",
            });
        }
        reject_non_finite("FnOperator transpose input", x)?;
        y.fill(T::zero());
        let Some(action) = self.transpose_into_fn.as_ref() else {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "FnOperator transpose",
                operator: "forward-only callback operator",
            });
        };
        catch_unwind(AssertUnwindSafe(|| action(x, y)))
            .map_err(|_| callback_error("FnOperator transpose callback"))?;
        reject_non_finite("FnOperator transpose output", y)
    }

    /// Checked Hermitian callback application without allocating an output.
    pub fn try_apply_hermitian_into(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        if x.len() != self.rows {
            return Err(SolverError::DimensionMismatch {
                expected: self.rows,
                got: x.len(),
            });
        }
        if y.len() != self.cols {
            return Err(SolverError::DimensionMismatch {
                expected: self.cols,
                got: y.len(),
            });
        }
        if self.hermitian_into_fn.is_none() {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "FnOperator Hermitian transpose",
                operator: "operator without an exact Hermitian callback",
            });
        }
        reject_non_finite("FnOperator Hermitian input", x)?;
        y.fill(T::zero());
        let Some(action) = self.hermitian_into_fn.as_ref() else {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: "FnOperator Hermitian transpose",
                operator: "operator without an exact Hermitian callback",
            });
        };
        catch_unwind(AssertUnwindSafe(|| action(x, y)))
            .map_err(|_| callback_error("FnOperator Hermitian callback"))?;
        reject_non_finite("FnOperator Hermitian output", y)
    }

    /// Create a callback operator with explicitly supplied exact adjoint actions.
    ///
    /// Forward-only [`FnOperator::new`] and [`FnOperator::new_into`] instances
    /// explicitly report both adjoint operations as unsupported. Callers that
    /// can provide mathematically exact actions can opt into the checked
    /// transpose/Hermitian contract with this constructor.
    pub fn new_into_with_adjoint(
        rows: usize,
        cols: usize,
        apply_into_fn: impl Fn(&[T], &mut [T]) + Send + Sync + 'static,
        transpose_into_fn: impl Fn(&[T], &mut [T]) + Send + Sync + 'static,
        hermitian_into_fn: impl Fn(&[T], &mut [T]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            rows,
            cols,
            apply_into_fn: Box::new(apply_into_fn),
            transpose_into_fn: Some(Box::new(transpose_into_fn)),
            hermitian_into_fn: Some(Box::new(hermitian_into_fn)),
        }
    }

    fn apply_hermitian_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(y.len(), self.cols, "FnOperator output length mismatch");
        let Some(action) = self.hermitian_into_fn.as_ref() else {
            panic!("FnOperator does not provide an exact Hermitian action");
        };
        action(x, y);
    }
}

impl<T: ComplexField> LinearOperator<T> for FnOperator<T> {
    fn num_rows(&self) -> usize {
        self.rows
    }
    fn num_cols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.rows];
        self.apply_into(x, &mut y);
        y
    }
    fn apply_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(y.len(), self.rows, "FnOperator output length mismatch");
        (self.apply_into_fn)(x, y);
    }
    fn apply_transpose(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.cols];
        self.apply_transpose_into(x, &mut y);
        y
    }
    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(y.len(), self.cols, "FnOperator output length mismatch");
        let Some(action) = self.transpose_into_fn.as_ref() else {
            panic!("FnOperator does not provide an exact transpose action");
        };
        action(x, y);
    }
    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.cols];
        self.apply_hermitian_into(x, &mut y);
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        if self.transpose_into_fn.is_some() {
            AdjointCapability::Exact
        } else {
            AdjointCapability::Unsupported
        }
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        if self.hermitian_into_fn.is_some() {
            AdjointCapability::Exact
        } else {
            AdjointCapability::Unsupported
        }
    }
}

/// Borrowed CSR matrix validated once for repeated applies.
///
/// Every `CsrMatrix` hot apply re-scans the full structure (an O(nnz)
/// column-bounds pass plus an O(rows) monotonicity pass) because the
/// storage is publicly mutable. That re-validation dominates the
/// per-iteration cost of Krylov solvers on large matrices. This wrapper
/// runs the same hot-structure check once at construction; the shared
/// borrow then freezes the storage for the wrapper lifetime, so every
/// apply takes the unchecked kernel without re-scanning.
///
/// # Examples
///
/// ```rust
/// use sonium_solvers_sparse::{CsrMatrix, ValidatedCsr};
/// use sonium_solvers_core::traits::LinearOperator;
///
/// let matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]);
/// let op = ValidatedCsr::try_new(&matrix).expect("valid structure");
/// let mut y = vec![0.0; 2];
/// op.apply_into(&[1.0, 1.0], &mut y);
/// assert_eq!(y, vec![2.0, 3.0]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ValidatedCsr<'a, T: ComplexField> {
    matrix: &'a CsrMatrix<T>,
}

impl<'a, T: ComplexField> ValidatedCsr<'a, T> {
    /// Checked constructor validating the hot-structure contract once.
    ///
    /// # Errors
    ///
    /// Returns the [`SolverError`] from the CSR hot-structure check when
    /// row pointers are not monotone rooted at zero with terminal count
    /// equal to the value count, when storage lengths disagree, or when
    /// any column index is out of bounds.
    pub fn try_new(matrix: &'a CsrMatrix<T>) -> Result<Self, SolverError> {
        matrix.validate_hot_structure()?;
        Ok(Self { matrix })
    }

    /// Compatibility constructor preserving the historical panic boundary.
    ///
    /// # Panics
    ///
    /// Panics with the same message as the `CsrMatrix` hot applies when
    /// the storage fails the hot-structure check.
    pub fn new(matrix: &'a CsrMatrix<T>) -> Self {
        Self::try_new(matrix).expect("invalid CSR storage")
    }

    /// Borrow the wrapped matrix.
    pub fn inner(&self) -> &'a CsrMatrix<T> {
        self.matrix
    }
}

impl<T: ComplexField> LinearOperator<T> for ValidatedCsr<'_, T> {
    fn num_rows(&self) -> usize {
        self.matrix.num_rows
    }

    fn num_cols(&self) -> usize {
        self.matrix.num_cols
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        // SAFETY: `try_new` validated the hot-structure contract and the
        // shared borrow freezes the public storage for this lifetime, so
        // no mutation can invalidate it. Length asserts re-run per call.
        unsafe { self.matrix.matvec_into_buf_unchecked(x, y) };
    }

    /// Multi-vector apply via the SpMM kernel without re-validation.
    fn apply_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        // SAFETY: same borrow-frozen contract as `apply_into` above.
        unsafe { self.matrix.spmm_into_unchecked(x, y) };
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        // SAFETY: same borrow-frozen contract as `apply_into` above.
        unsafe { self.matrix.matvec_transpose_into_buf_unchecked(x, y) };
    }

    /// Multi-vector transpose apply with validate-once checking.
    ///
    /// Mirrors the `CsrMatrix` override: the structure cannot change
    /// between columns of one call, so the construction-time validation
    /// covers all `k` unchecked column applies.
    fn apply_transpose_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        assert_eq!(
            x.nrows(),
            self.matrix.num_rows,
            "block transpose input row mismatch"
        );
        assert_eq!(
            y.nrows(),
            self.matrix.num_cols,
            "block transpose output row mismatch"
        );
        assert_eq!(
            x.ncols(),
            y.ncols(),
            "block transpose column count mismatch"
        );
        let m = self.matrix.num_rows;
        let n = self.matrix.num_cols;
        let mut xj = vec![T::zero(); m];
        let mut yj = vec![T::zero(); n];
        for j in 0..x.ncols() {
            for i in 0..m {
                xj[i] = x[[i, j]];
            }
            // SAFETY: same borrow-frozen contract as `apply_into` above.
            unsafe {
                self.matrix
                    .matvec_transpose_into_buf_unchecked(&xj, &mut yj)
            };
            y.set_column(j, &yj);
        }
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.matrix.num_cols];
        // SAFETY: same borrow-frozen contract as `apply_into` above.
        unsafe { self.matrix.matvec_hermitian_into_buf_unchecked(x, &mut y) };
        y
    }

    /// Multi-vector Hermitian apply with validate-once checking.
    fn apply_hermitian_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        assert_eq!(
            x.nrows(),
            self.matrix.num_rows,
            "block Hermitian input row mismatch"
        );
        assert_eq!(
            y.nrows(),
            self.matrix.num_cols,
            "block Hermitian output row mismatch"
        );
        assert_eq!(
            x.ncols(),
            y.ncols(),
            "block Hermitian column count mismatch"
        );
        let m = self.matrix.num_rows;
        let n = self.matrix.num_cols;
        let mut xj = vec![T::zero(); m];
        let mut yj = vec![T::zero(); n];
        for j in 0..x.ncols() {
            for i in 0..m {
                xj[i] = x[[i, j]];
            }
            // SAFETY: same borrow-frozen contract as `apply_into` above.
            unsafe {
                self.matrix
                    .matvec_hermitian_into_buf_unchecked(&xj, &mut yj)
            };
            y.set_column(j, &yj);
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrMatrix;
    use num_complex::Complex64;

    fn _identity_2x2() -> CsrMatrix<f64> {
        CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0])
    }

    fn diag_2x2(a: f64, b: f64) -> CsrMatrix<f64> {
        CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![a, b])
    }

    #[test]
    fn test_sum_operator() {
        let a = diag_2x2(2.0, 3.0);
        let b = diag_2x2(1.0, 1.0);
        let sum = SumOperator::new(&a, &b);

        let x = vec![1.0_f64, 1.0];
        let y = sum.apply(&x);
        assert!((y[0] - 3.0).abs() < 1e-12);
        assert!((y[1] - 4.0).abs() < 1e-12);
        assert_eq!(sum.hermitian_capability(), AdjointCapability::Exact);
        assert!(sum.try_apply_hermitian(&x).is_ok());
    }

    #[test]
    fn test_scaled_operator() {
        let a = diag_2x2(2.0, 3.0);
        let scaled = ScaledOperator::new(&a, 0.5);

        let x = vec![1.0_f64, 1.0];
        let y = scaled.apply(&x);
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 1.5).abs() < 1e-12);
        assert_eq!(scaled.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(scaled.hermitian_capability(), AdjointCapability::Exact);
    }

    #[test]
    fn test_composite_operator() {
        let a = diag_2x2(2.0, 3.0);
        let b = diag_2x2(4.0, 5.0);
        let ab = CompositeOperator::new(&a, &b);

        let x = vec![1.0_f64, 1.0];
        let y = ab.apply(&x); // A * (B * x) = diag(2,3) * diag(4,5) * [1,1] = [8,15]
        assert!((y[0] - 8.0).abs() < 1e-12);
        assert!((y[1] - 15.0).abs() < 1e-12);
        assert_eq!(ab.hermitian_capability(), AdjointCapability::Exact);
        assert!(ab.try_apply_hermitian(&x).is_ok());
    }

    #[test]
    fn test_composite_operator_reuses_scratch() {
        let a = diag_2x2(2.0, 3.0);
        let b = diag_2x2(4.0, 5.0);
        let ab = CompositeOperator::new(&a, &b);

        let x = vec![1.0_f64, 1.0];
        let mut y = vec![0.0; 2];
        let mut scratch = vec![0.0; ab.scratch_len()];
        ab.apply_into_with_scratch(&x, &mut y, &mut scratch);

        assert!((y[0] - 8.0).abs() < 1e-12);
        assert!((y[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn checked_operator_combinators_reject_shape_and_scratch_mismatches() {
        let a = diag_2x2(2.0, 3.0);
        let b =
            CsrMatrix::from_raw_parts(3, 2, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0, 1.0, 1.0]);
        assert!(matches!(
            SumOperator::try_new(&a, &b),
            Err(sonium_solvers_core::traits::SolverError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            CompositeOperator::try_new(&a, &b),
            Err(sonium_solvers_core::traits::SolverError::DimensionMismatch { .. })
        ));

        let composition = CompositeOperator::try_new(&a, &a).unwrap();
        let mut output = vec![0.0; 2];
        let mut scratch = vec![0.0; 1];
        assert!(matches!(
            composition.try_apply_into_with_scratch(&[1.0, 1.0], &mut output, &mut scratch),
            Err(sonium_solvers_core::traits::SolverError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            composition
                .try_apply_transpose_into_with_scratch(&[1.0], &mut output, &mut [0.0, 0.0],),
            Err(sonium_solvers_core::traits::SolverError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_fn_operator_apply_into_callback() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_op = Arc::clone(&calls);
        let op = FnOperator::new_into(2, 2, move |x: &[f64], y: &mut [f64]| {
            calls_for_op.fetch_add(1, Ordering::Relaxed);
            y[0] = 2.0 * x[0];
            y[1] = 3.0 * x[1];
        });

        let mut y = vec![0.0; 2];
        op.apply_into(&[4.0, 5.0], &mut y);
        assert_eq!(y, vec![8.0, 15.0]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let y = op.apply(&[1.0, 2.0]);
        assert_eq!(y, vec![2.0, 6.0]);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn checked_fn_operator_rejects_non_finite_forward_output() {
        let operator = FnOperator::new_into(2, 2, |_x: &[f64], y: &mut [f64]| {
            y.fill(f64::NAN);
        });
        let mut output = vec![0.0; 2];
        assert!(matches!(
            operator.try_apply_into(&[1.0, 2.0], &mut output),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn checked_fn_operator_rejects_undeclared_and_non_finite_adjoint_actions() {
        let forward_only = FnOperator::new(2, 2, |x: &[f64]| x.to_vec());
        let mut output = vec![0.0; 2];
        assert!(matches!(
            forward_only.try_apply_transpose_into(&[1.0, 2.0], &mut output),
            Err(SolverError::UnsupportedOperatorOperation { .. })
        ));

        let operator = FnOperator::new_into_with_adjoint(
            2,
            2,
            |_x: &[f64], y: &mut [f64]| y.fill(0.0),
            |_x: &[f64], y: &mut [f64]| y.fill(0.0),
            |_x: &[f64], y: &mut [f64]| y.fill(f64::INFINITY),
        );
        assert!(matches!(
            operator.try_apply_hermitian_into(&[1.0, 2.0], &mut output),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn checked_composite_rejects_non_finite_callback_output() {
        let inner = FnOperator::new_into(2, 2, |_x: &[f64], y: &mut [f64]| {
            y.fill(f64::NAN);
        });
        let outer = diag_2x2(1.0, 1.0);
        let composition = CompositeOperator::try_new(&outer, &inner).unwrap();
        let mut output = vec![0.0; 2];
        let mut scratch = vec![0.0; composition.scratch_len()];
        assert!(matches!(
            composition.try_apply_into_with_scratch(&[1.0, 2.0], &mut output, &mut scratch),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn checked_sum_rejects_non_finite_callback_output() {
        let non_finite = FnOperator::new_into(2, 2, |_x: &[f64], y: &mut [f64]| {
            y.fill(f64::NAN);
        });
        let other = diag_2x2(1.0, 1.0);
        let sum = SumOperator::try_new(&other, &non_finite).unwrap();
        let mut output = vec![0.0; 2];
        assert!(matches!(
            sum.try_apply_into(&[1.0, 2.0], &mut output),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn checked_sum_transpose_rejects_missing_adjoint() {
        let forward_only = FnOperator::new(2, 2, |x: &[f64]| x.to_vec());
        let other = diag_2x2(1.0, 1.0);
        let sum = SumOperator::try_new(&forward_only, &other).unwrap();
        let mut output = vec![0.0; 2];
        assert!(matches!(
            sum.try_apply_transpose_into(&[1.0, 2.0], &mut output),
            Err(SolverError::UnsupportedOperatorOperation { .. })
        ));
    }

    #[test]
    fn checked_composite_contains_callback_panics() {
        let inner = FnOperator::new_into(2, 2, |_x: &[f64], _y: &mut [f64]| {
            panic!("synthetic operator failure");
        });
        let outer = diag_2x2(1.0, 1.0);
        let composition = CompositeOperator::try_new(&outer, &inner).unwrap();
        let mut output = vec![0.0; 2];
        let mut scratch = vec![0.0; composition.scratch_len()];
        assert!(matches!(
            composition.try_apply_into_with_scratch(&[1.0, 2.0], &mut output, &mut scratch),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn checked_composite_transpose_rejects_non_finite_callback_output() {
        let inner = FnOperator::new_into_with_adjoint(
            2,
            2,
            |_x: &[f64], y: &mut [f64]| y.fill(0.0),
            |_x: &[f64], y: &mut [f64]| y.fill(f64::INFINITY),
            |_x: &[f64], y: &mut [f64]| y.fill(0.0),
        );
        let outer = diag_2x2(1.0, 1.0);
        let composition = CompositeOperator::try_new(&outer, &inner).unwrap();
        let mut output = vec![0.0; 2];
        let mut scratch = vec![0.0; composition.scratch_len()];
        assert!(matches!(
            composition.try_apply_transpose_into_with_scratch(
                &[1.0, 2.0],
                &mut output,
                &mut scratch,
            ),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn fn_operator_explicitly_rejects_adjoint_actions() {
        let op = FnOperator::new(2, 2, |x: &[f64]| vec![x[0], 2.0 * x[1]]);

        assert_eq!(op.transpose_capability(), AdjointCapability::Unsupported);
        assert_eq!(op.hermitian_capability(), AdjointCapability::Unsupported);
        assert!(matches!(
            op.try_apply_transpose(&[1.0, 2.0]),
            Err(sonium_solvers_core::traits::SolverError::UnsupportedOperatorOperation { .. })
        ));
        assert!(matches!(
            op.try_apply_hermitian(&[1.0, 2.0]),
            Err(sonium_solvers_core::traits::SolverError::UnsupportedOperatorOperation { .. })
        ));
    }

    #[test]
    fn fn_operator_accepts_explicit_complex_adjoint_actions() {
        let op = FnOperator::new_into_with_adjoint(
            2,
            2,
            |x: &[Complex64], y: &mut [Complex64]| {
                y[0] = Complex64::new(1.0, 1.0) * x[0] + Complex64::new(2.0, 0.0) * x[1];
                y[1] = Complex64::new(3.0, 0.0) * x[0] + Complex64::new(4.0, -1.0) * x[1];
            },
            |x: &[Complex64], y: &mut [Complex64]| {
                y[0] = Complex64::new(1.0, 1.0) * x[0] + Complex64::new(3.0, 0.0) * x[1];
                y[1] = Complex64::new(2.0, 0.0) * x[0] + Complex64::new(4.0, -1.0) * x[1];
            },
            |x: &[Complex64], y: &mut [Complex64]| {
                y[0] = Complex64::new(1.0, -1.0) * x[0] + Complex64::new(3.0, 0.0) * x[1];
                y[1] = Complex64::new(2.0, 0.0) * x[0] + Complex64::new(4.0, 1.0) * x[1];
            },
        );

        assert_eq!(op.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(op.hermitian_capability(), AdjointCapability::Exact);
        let rhs = [Complex64::new(0.5, -1.0), Complex64::new(-2.0, 0.25)];
        let hermitian = op
            .try_apply_hermitian(&rhs)
            .expect("exact Hermitian action");
        assert_eq!(
            hermitian,
            vec![
                Complex64::new(1.0, -1.0) * rhs[0] + Complex64::new(3.0, 0.0) * rhs[1],
                Complex64::new(2.0, 0.0) * rhs[0] + Complex64::new(4.0, 1.0) * rhs[1],
            ]
        );
    }

    #[test]
    fn test_helmholtz_like_combination() {
        let k = diag_2x2(4.0, 4.0); // stiffness
        let m = diag_2x2(1.0, 1.0); // mass
        let k_sq = 2.0_f64; // wavenumber squared

        let neg_k2_m = ScaledOperator::new(&m, -k_sq);
        let helmholtz = SumOperator::new(&k, &neg_k2_m);

        let x = vec![1.0_f64, 1.0];
        let y = helmholtz.apply(&x); // (4 - 2) * [1,1] = [2, 2]
        assert!((y[0] - 2.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
    }

    fn nonsymmetric_3x3() -> CsrMatrix<f64> {
        // [[2, -1, 0], [0, 3, -1], [1, 0, 4]]
        CsrMatrix::from_raw_parts(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![2.0, -1.0, 3.0, -1.0, 1.0, 4.0],
        )
    }

    #[test]
    fn validated_csr_matches_checked_applies() {
        let matrix = nonsymmetric_3x3();
        let op = ValidatedCsr::try_new(&matrix).expect("valid structure");
        assert_eq!(op.num_rows(), 3);
        assert_eq!(op.num_cols(), 3);
        assert_eq!(op.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(op.hermitian_capability(), AdjointCapability::Exact);

        let x = vec![1.0_f64, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        op.apply_into(&x, &mut y);
        assert_eq!(y, matrix.matvec(&x));
        assert_eq!(op.apply(&x), matrix.matvec(&x));

        let mut yt = vec![0.0; 3];
        op.apply_transpose_into(&x, &mut yt);
        assert_eq!(yt, matrix.matvec_transpose(&x));
        assert_eq!(op.apply_transpose(&x), matrix.matvec_transpose(&x));

        assert_eq!(op.apply_hermitian(&x), matrix.matvec_hermitian(&x));

        // Checked trait defaults still gate shapes and finiteness.
        assert!(op.try_apply(&x).is_ok());
        assert!(op.try_apply_transpose(&x).is_ok());
        assert!(op.try_apply_hermitian(&x).is_ok());
        assert!(op.try_apply(&[1.0, 2.0]).is_err());
        assert!(op.try_apply(&[1.0, f64::NAN, 3.0]).is_err());
    }

    #[test]
    fn validated_csr_matches_checked_block_applies() {
        use sonium_solvers_core::dense::DenseMatrix;

        let matrix = nonsymmetric_3x3();
        let op = ValidatedCsr::try_new(&matrix).expect("valid structure");
        let x = DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut y = DenseMatrix::zeros(3, 2);
        let mut expected = DenseMatrix::zeros(3, 2);
        op.apply_block(&x, &mut y);
        matrix.spmm_into(&x, &mut expected);
        assert_eq!(y.as_slice(), expected.as_slice());

        let mut yt = DenseMatrix::zeros(3, 2);
        let mut expected_t = DenseMatrix::zeros(3, 2);
        op.apply_transpose_block(&x, &mut yt);
        matrix.apply_transpose_block(&x, &mut expected_t);
        assert_eq!(yt.as_slice(), expected_t.as_slice());

        let mut yh = DenseMatrix::zeros(3, 2);
        let mut expected_h = DenseMatrix::zeros(3, 2);
        op.apply_hermitian_block(&x, &mut yh);
        matrix.apply_hermitian_block(&x, &mut expected_h);
        assert_eq!(yh.as_slice(), expected_h.as_slice());
    }

    #[test]
    fn validated_csr_rejects_corrupt_structure() {
        // Out-of-bounds column.
        let mut bad_col = nonsymmetric_3x3();
        bad_col.col_indices[0] = 3;
        assert!(ValidatedCsr::try_new(&bad_col).is_err());

        // Non-monotone row pointers.
        let mut bad_rows = nonsymmetric_3x3();
        bad_rows.row_ptrs[1] = 5;
        assert!(ValidatedCsr::try_new(&bad_rows).is_err());

        // Terminal pointer diverging from the value count.
        let mut bad_terminal = nonsymmetric_3x3();
        bad_terminal.row_ptrs[3] = 5;
        assert!(ValidatedCsr::try_new(&bad_terminal).is_err());

        // Length disagreement.
        let mut bad_len = nonsymmetric_3x3();
        bad_len.values.pop();
        assert!(ValidatedCsr::try_new(&bad_len).is_err());
    }

    #[test]
    fn validated_csr_matches_complex_hermitian() {
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![
                Complex64::new(2.0, 1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(3.0, -2.0),
            ],
        );
        let op = ValidatedCsr::try_new(&matrix).expect("valid structure");
        let x = vec![Complex64::new(1.0, -1.0), Complex64::new(0.5, 2.0)];
        assert_eq!(op.apply_hermitian(&x), matrix.matvec_hermitian(&x));
        let mut y = vec![Complex64::new(0.0, 0.0); 2];
        op.apply_into(&x, &mut y);
        assert_eq!(y, matrix.matvec(&x));
    }
}
