//! Optimizers for differentiable DSP parameters.

use crate::autodiff::error::AutodiffError;
use crate::autodiff::module::Scalar;
use crate::autodiff::tensor::NdArray;

/// Stochastic gradient descent optimizer with a constant learning rate.
#[derive(Debug, Clone)]
pub struct Sgd<T = f64> {
    /// Learning rate.
    pub lr: T,
}

impl<T> Sgd<T> {
    /// Create a new SGD optimizer with the given learning rate.
    #[must_use]
    pub const fn new(lr: T) -> Self {
        Self { lr }
    }
}

impl<T: Scalar> Sgd<T> {
    /// Apply a single SGD update: `param -= lr * grad`.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter and gradient counts or shapes differ.
    ///
    /// # Panics
    ///
    /// Panics on internal indexing violations; unreachable after shape validation.
    pub fn step(
        &self,
        params: &mut [&mut NdArray<T>],
        grads: &[&NdArray<T>],
    ) -> Result<(), AutodiffError> {
        if params.len() != grads.len() {
            return Err(AutodiffError::Message(format!(
                "SGD: received {} parameters but {} gradients",
                params.len(),
                grads.len()
            )));
        }
        for (index, (param, grad)) in params.iter().zip(grads).enumerate() {
            if param.shape() != grad.shape() {
                return Err(AutodiffError::Message(format!(
                    "SGD: parameter {index} shape {:?} does not match gradient shape {:?}",
                    param.shape(),
                    grad.shape()
                )));
            }
        }
        for (p, g) in params.iter_mut().zip(grads) {
            let grad = g.as_slice().expect("gradient storage must be contiguous");
            for (parameter, &gradient) in p
                .as_slice_mut()
                .expect("parameter storage must be contiguous")
                .iter_mut()
                .zip(grad.iter())
            {
                *parameter -= self.lr * gradient;
            }
        }
        Ok(())
    }
}
