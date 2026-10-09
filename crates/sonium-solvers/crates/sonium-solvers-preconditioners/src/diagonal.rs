//! Diagonal (Jacobi) preconditioner
//!
//! Simple but effective preconditioner that scales by the diagonal of A.
//!
//! This preconditioner is embarrassingly parallel since it only involves
//! element-wise operations.

use num_traits::{Float, FromPrimitive};
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, Preconditioner, SolverError};
use sonium_solvers_sparse::CsrMatrix;

use rayon::prelude::*;

/// Diagonal (Jacobi) preconditioner
///
/// M = diag(A), so M^(-1) scales each component by 1/A_ii
#[derive(Debug, Clone)]
pub struct DiagonalPreconditioner<T: ComplexField> {
    /// Inverse diagonal elements
    inv_diag: Vec<T>,
}

impl<T: ComplexField> DiagonalPreconditioner<T> {
    /// Construct a Jacobi preconditioner after validating the CSR structure
    /// and all stored values. The compatibility [`Self::from_csr`] constructor
    /// remains available for existing callers, while new fallible solver
    /// paths should prefer this method.
    pub fn try_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        matrix.validate_canonical_structure()?;
        Self::try_from_diagonal(&matrix.diagonal())
    }

    /// Create a diagonal preconditioner from a CSR matrix
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Self {
        let diag = matrix.diagonal();
        let inv_diag = diag
            .iter()
            .map(|&d| {
                if d.norm() > T::Real::from_f64(1e-20).unwrap() {
                    d.inv()
                } else {
                    T::one()
                }
            })
            .collect();
        Self { inv_diag }
    }

    /// Create from a diagonal vector directly
    pub fn from_diagonal(diag: &[T]) -> Self {
        let inv_diag = diag
            .iter()
            .map(|&d| {
                if d.norm() > T::Real::from_f64(1e-20).unwrap() {
                    d.inv()
                } else {
                    T::one()
                }
            })
            .collect();
        Self { inv_diag }
    }

    /// Construct from a diagonal after rejecting non-finite entries.
    pub fn try_from_diagonal(diag: &[T]) -> Result<Self, SolverError> {
        if let Some(index) = diag.iter().position(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "diagonal preconditioner setup",
                reason: format!("diagonal entry at index {index} is non-finite"),
            });
        }
        Ok(Self::from_diagonal(diag))
    }

    /// Create from inverse diagonal vector directly
    pub fn from_inverse_diagonal(inv_diag: Vec<T>) -> Self {
        Self { inv_diag }
    }

    /// Construct from inverse diagonal values after rejecting non-finite
    /// preconditioner state.
    pub fn try_from_inverse_diagonal(inv_diag: Vec<T>) -> Result<Self, SolverError> {
        if let Some(index) = inv_diag.iter().position(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "diagonal preconditioner setup",
                reason: format!("inverse diagonal entry at index {index} is non-finite"),
            });
        }
        Ok(Self::from_inverse_diagonal(inv_diag))
    }
}

impl<T: ComplexField> Preconditioner<T> for DiagonalPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); r.len()];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(r.len(), y.len(), "preconditioner output size mismatch");
        assert_eq!(
            r.len(),
            self.inv_diag.len(),
            "diagonal preconditioner size mismatch"
        );

        if r.len() >= 1000 {
            self.apply_parallel_into(r, y);
        } else {
            self.apply_sequential_into(r, y);
        }
    }

    /// Transpose of a diagonal operator is itself.
    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        self.apply(r)
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.apply_into(r, y);
    }

    /// Hermitian scales by the conjugated inverse diagonal.
    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); r.len()];
        self.apply_hermitian_into(r, &mut y);
        y
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(r.len(), y.len(), "preconditioner output size mismatch");
        assert_eq!(
            r.len(),
            self.inv_diag.len(),
            "diagonal preconditioner size mismatch"
        );
        y.iter_mut()
            .zip(r.iter())
            .zip(self.inv_diag.iter())
            .for_each(|((yi, &ri), &di)| {
                *yi = ri * di.conj();
            });
    }
}

impl<T: ComplexField> DiagonalPreconditioner<T> {
    fn apply_sequential_into(&self, r: &[T], y: &mut [T]) {
        y.iter_mut()
            .zip(r.iter())
            .zip(self.inv_diag.iter())
            .for_each(|((yi, &ri), &di)| {
                *yi = ri * di;
            });
    }

    fn apply_parallel_into(&self, r: &[T], y: &mut [T])
    where
        T: Send + Sync,
    {
        y.par_iter_mut()
            .zip(r.par_iter())
            .zip(self.inv_diag.par_iter())
            .for_each(|((yi, &ri), &di)| {
                *yi = ri * di;
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_diagonal_preconditioner() {
        let diag = sonium_solvers_core::array![
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0)
        ];

        let precond = DiagonalPreconditioner::from_diagonal(&diag);

        let r = sonium_solvers_core::array![
            Complex64::new(2.0, 0.0),
            Complex64::new(8.0, 0.0),
            Complex64::new(3.0, 0.0)
        ];

        let result = precond.apply(&r);

        assert_relative_eq!(result[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[2].re, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_from_csr() {
        use sonium_solvers_sparse::CsrMatrix;

        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        ];

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = DiagonalPreconditioner::from_csr(&matrix);

        let r = sonium_solvers_core::array![Complex64::new(4.0, 0.0), Complex64::new(4.0, 0.0)];
        let result = precond.apply(&r);

        assert_relative_eq!(result[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1].re, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn checked_diagonal_constructors_reject_non_finite_state() {
        assert!(DiagonalPreconditioner::<f64>::try_from_diagonal(&[1.0, f64::NAN]).is_err());
        assert!(
            DiagonalPreconditioner::<f64>::try_from_inverse_diagonal(vec![1.0, f64::INFINITY])
                .is_err()
        );

        let mut matrix = CsrMatrix::from_diagonal(&[1.0_f64, 2.0]);
        matrix.values[1] = f64::NAN;
        assert!(DiagonalPreconditioner::try_from_csr(&matrix).is_err());
    }

    #[test]
    fn test_diagonal_zero_entries() {
        // Near-zero diagonal entries should be handled gracefully (not produce NaN)
        let diag = sonium_solvers_core::array![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),   // zero entry
            Complex64::new(1e-30, 0.0), // near-zero entry
            Complex64::new(3.0, 0.0),
        ];

        let precond = DiagonalPreconditioner::from_diagonal(&diag);
        let r = sonium_solvers_core::array![
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(9.0, 0.0),
        ];

        let result = precond.apply(&r);

        // First entry: 4/2 = 2
        assert_relative_eq!(result[0].re, 2.0, epsilon = 1e-10);
        // Last entry: 9/3 = 3
        assert_relative_eq!(result[3].re, 3.0, epsilon = 1e-10);
        // Near-zero and zero entries should not produce NaN or Inf
        assert!(
            result[1].re.is_finite(),
            "Zero diagonal should not produce NaN/Inf"
        );
        assert!(
            result[2].re.is_finite(),
            "Near-zero diagonal should not produce NaN/Inf"
        );
    }

    #[test]
    fn test_diagonal_complex() {
        // Complex64 diagonal preconditioner with non-zero imaginary parts
        let diag =
            sonium_solvers_core::array![Complex64::new(2.0, 1.0), Complex64::new(1.0, -1.0),];

        let precond = DiagonalPreconditioner::from_diagonal(&diag);
        let r = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),];

        let result = precond.apply(&r);

        // result[0] = r[0] / diag[0] = 1/(2+i) = (2-i)/5
        let expected_0 = Complex64::new(1.0, 0.0) / Complex64::new(2.0, 1.0);
        assert_relative_eq!(result[0].re, expected_0.re, epsilon = 1e-10);
        assert_relative_eq!(result[0].im, expected_0.im, epsilon = 1e-10);

        // result[1] = r[1] / diag[1] = i/(1-i) = i(1+i)/2 = (-1+i)/2
        let expected_1 = Complex64::new(0.0, 1.0) / Complex64::new(1.0, -1.0);
        assert_relative_eq!(result[1].re, expected_1.re, epsilon = 1e-10);
        assert_relative_eq!(result[1].im, expected_1.im, epsilon = 1e-10);
    }
}
