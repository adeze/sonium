//! Checked adapters for legacy Krylov callback boundaries.

use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};

use num_traits::Float;
use sonium_solvers_core::traits::{
    AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
};

/// An operator adapter that turns checked callback failures into panics for
/// legacy result-shaped solver cores. The public `try_*` wrapper catches the
/// panic and returns the original typed failure.
pub(crate) struct CheckedLinearOperator<'a, T, A> {
    inner: &'a A,
    marker: PhantomData<fn() -> T>,
}

impl<'a, T, A> CheckedLinearOperator<'a, T, A> {
    pub(crate) fn new(inner: &'a A) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }
}

impl<T, A> LinearOperator<T> for CheckedLinearOperator<'_, T, A>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_cols()
    }

    fn apply(&self, x: &[T]) -> Vec<T> {
        self.inner
            .try_apply(x)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        if x.len() != self.inner.num_cols() {
            panic!(
                "operator input length mismatch: expected {}, got {}",
                self.inner.num_cols(),
                x.len()
            );
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            panic!("operator received NaN or infinity input");
        }
        if y.len() != self.inner.num_rows() {
            panic!(
                "operator output length mismatch: expected {}, got {}",
                self.inner.num_rows(),
                y.len()
            );
        }
        self.inner.apply_into(x, y);
        if y.iter().any(|value| !value.norm().is_finite()) {
            panic!("operator generated NaN or infinity");
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.inner.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.inner.hermitian_capability()
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        if x.len() != self.inner.num_rows() {
            panic!(
                "operator transpose input length mismatch: expected {}, got {}",
                self.inner.num_rows(),
                x.len()
            );
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            panic!("operator transpose received NaN or infinity input");
        }
        if y.len() != self.inner.num_cols() {
            panic!(
                "operator transpose output length mismatch: expected {}, got {}",
                self.inner.num_cols(),
                y.len()
            );
        }
        self.inner.apply_transpose_into(x, y);
        if y.iter().any(|value| !value.norm().is_finite()) {
            panic!("operator transpose generated NaN or infinity");
        }
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        if x.len() != self.inner.num_rows() {
            panic!(
                "operator Hermitian input length mismatch: expected {}, got {}",
                self.inner.num_rows(),
                x.len()
            );
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            panic!("operator Hermitian received NaN or infinity input");
        }
        let y = self.inner.apply_hermitian(x);
        if y.len() != self.inner.num_cols() {
            panic!(
                "operator Hermitian output length mismatch: expected {}, got {}",
                self.inner.num_cols(),
                y.len()
            );
        }
        if y.iter().any(|value| !value.norm().is_finite()) {
            panic!("operator Hermitian generated NaN or infinity");
        }
        y
    }
}

/// A preconditioner adapter with the same panic-to-typed-error bridge as
/// [`CheckedLinearOperator`].
pub(crate) struct CheckedPreconditioner<'a, T, P> {
    inner: &'a P,
    marker: PhantomData<fn() -> T>,
}

impl<'a, T, P> CheckedPreconditioner<'a, T, P> {
    pub(crate) fn new(inner: &'a P) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }
}

impl<T, P> Preconditioner<T> for CheckedPreconditioner<'_, T, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply(&self, r: &[T]) -> Vec<T> {
        self.inner
            .try_apply(r)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        if r.len() != y.len() {
            panic!(
                "preconditioner output length mismatch: expected {}, got {}",
                r.len(),
                y.len()
            );
        }
        if r.iter().any(|value| !value.norm().is_finite()) {
            panic!("preconditioner received NaN or infinity input");
        }
        self.inner.apply_into(r, y);
        if y.iter().any(|value| !value.norm().is_finite()) {
            panic!("preconditioner generated NaN or infinity");
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.inner.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.inner.hermitian_capability()
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        if r.len() != y.len() {
            panic!(
                "preconditioner transpose output length mismatch: expected {}, got {}",
                r.len(),
                y.len()
            );
        }
        if r.iter().any(|value| !value.norm().is_finite()) {
            panic!("preconditioner transpose received NaN or infinity input");
        }
        self.inner.apply_transpose_into(r, y);
        if y.iter().any(|value| !value.norm().is_finite()) {
            panic!("preconditioner transpose generated NaN or infinity");
        }
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        if r.iter().any(|value| !value.norm().is_finite()) {
            panic!("preconditioner Hermitian received NaN or infinity input");
        }
        let y = self.inner.apply_hermitian(r);
        if y.len() != r.len() {
            panic!(
                "preconditioner Hermitian output length mismatch: expected {}, got {}",
                r.len(),
                y.len()
            );
        }
        if y.iter().any(|value| !value.norm().is_finite()) {
            panic!("preconditioner Hermitian generated NaN or infinity");
        }
        y
    }
}

/// Catch a legacy solver core panic and retain a typed callback-boundary
/// failure for checked callers.
pub(crate) fn catch_solver_callback<T, F>(operation: &'static str, f: F) -> Result<T, SolverError>
where
    F: FnOnce() -> Result<T, SolverError>,
{
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| SolverError::OperationFailed {
        operation,
        reason: panic_reason(payload),
    })?
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "solver callback panicked".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroOperator;

    impl LinearOperator<f64> for ZeroOperator {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
            y.fill(0.0);
        }

        fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
            y.fill(0.0);
        }
    }

    struct ZeroPreconditioner;

    impl Preconditioner<f64> for ZeroPreconditioner {
        fn apply_into(&self, _r: &[f64], y: &mut [f64]) {
            y.fill(0.0);
        }
    }

    #[test]
    fn checked_operator_rejects_non_finite_in_place_input() {
        let operator = ZeroOperator;
        let checked = CheckedLinearOperator::<f64, _>::new(&operator);
        let result = catch_solver_callback("checked operator", || {
            let mut output = [0.0];
            checked.apply_into(&[f64::NAN], &mut output);
            Ok(())
        });
        let error = result.expect_err("non-finite operator input must be typed failure");
        assert!(matches!(
            error,
            SolverError::OperationFailed { reason, .. }
                if reason.contains("operator received NaN or infinity input")
        ));
    }

    #[test]
    fn checked_operator_rejects_non_finite_hermitian_input() {
        let operator = ZeroOperator;
        let checked = CheckedLinearOperator::<f64, _>::new(&operator);
        let result = catch_solver_callback("checked operator", || {
            let _ = checked.apply_hermitian(&[f64::NAN]);
            Ok(())
        });
        let error = result.expect_err("non-finite Hermitian input must be typed failure");
        assert!(matches!(
            error,
            SolverError::OperationFailed { reason, .. }
                if reason.contains("operator Hermitian received NaN or infinity input")
        ));
    }

    #[test]
    fn checked_preconditioner_rejects_non_finite_adjoint_input() {
        let preconditioner = ZeroPreconditioner;
        let checked = CheckedPreconditioner::<f64, _>::new(&preconditioner);
        let result = catch_solver_callback("checked preconditioner", || {
            let mut output = [0.0];
            checked.apply_transpose_into(&[f64::INFINITY], &mut output);
            Ok(())
        });
        let error = result.expect_err("non-finite preconditioner input must be typed failure");
        assert!(matches!(
            error,
            SolverError::OperationFailed { reason, .. }
                if reason.contains("preconditioner transpose received NaN or infinity input")
        ));
    }

    #[test]
    fn checked_preconditioner_rejects_non_finite_forward_input() {
        let preconditioner = ZeroPreconditioner;
        let checked = CheckedPreconditioner::<f64, _>::new(&preconditioner);
        let result = catch_solver_callback("checked preconditioner", || {
            let mut output = [0.0];
            checked.apply_into(&[f64::INFINITY], &mut output);
            Ok(())
        });
        let error = result.expect_err("non-finite preconditioner input must be typed failure");
        assert!(matches!(
            error,
            SolverError::OperationFailed { reason, .. }
                if reason.contains("preconditioner received NaN or infinity input")
        ));
    }
}
