//! Dense matrix exponential via scaling and squaring.
//!
//! [`expm`] evaluates `exp(A)` for a square [`DenseMatrix`] with the
//! scaling-and-squaring method combined with a `[13/13]` Padé approximant
//! (Higham, *Functions of Matrices*, Algorithm 2.3). The implementation is
//! generic over [`ComplexField`], covering `f32`, `f64`, `Complex32`, and
//! `Complex64`, and is intended for reference-sized dense problems such as
//! orthogonal maps `exp(skew)`; large sparse exponentials need Krylov
//! methods, which this crate does not provide.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, ComplexField, SolverFailure, SolverStatus},
};
use thiserror::Error;

/// Padé degree used by [`expm`].
///
/// A single `[13/13]` approximant covers every input through scaling, which
/// keeps one code path (and one coefficient recurrence) instead of the
/// textbook degree table. Small-norm inputs pay a few redundant
/// matrix-matrix products; that cost is negligible at reference sizes.
const PADE_DEGREE: usize = 13;

/// Largest 1-norm handled without scaling by the `[13/13]` approximant.
///
/// Backward-error bound θ₁₃ from Higham (*Functions of Matrices*,
/// Table 2.3): `‖A‖₁/2ˢ ≤ θ₁₃` keeps the Padé evaluation at full
/// working precision. Inputs above it are scaled by `1/2ˢ` and the result
/// is squared `s` times.
const THETA_13: f64 = 5.371_920_351_148_152;

#[derive(Debug, Clone, Error, PartialEq)]
pub enum ExpmError {
    #[error("matrix exponential requires a square matrix, got {rows}x{columns}")]
    NonSquare { rows: usize, columns: usize },
    #[error("matrix exponential input contains a non-finite value")]
    NonFiniteInput,
    #[error("matrix exponential {operation} generated a non-finite value")]
    NumericalBreakdown { operation: &'static str },
}

impl ExpmError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NumericalBreakdown { .. } => SolverStatus::Breakdown,
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

impl SolverFailure for ExpmError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NumericalBreakdown { .. } => SolverErrorKind::Breakdown,
            Self::NonSquare { .. } => SolverErrorKind::DimensionMismatch,
            Self::NonFiniteInput => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        ExpmError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        ExpmError::accuracy_status(self)
    }
}

/// Evaluate the exponential of a square dense matrix.
///
/// Computes `exp(A) = Σₖ Aᵏ/k!` by scaling `A` down until its 1-norm fits
/// the `[13/13]` Padé validity radius, evaluating the approximant as
/// `(V - U)⁻¹(V + U)` from its even (`V`) and odd (`U`) parts, and squaring
/// the result back up. The Padé coefficients come from the closed-form
/// recurrence `b[j+1] = b[j]·(m-j)/((2m-j)·(j+1))` with `b[0] = 1`.
///
/// The zero matrix returns the exact identity without floating-point work.
/// Eigenvalue order and multiplicities play no role: the method only uses
/// matrix products and one LU solve of the Padé denominator.
///
/// # Examples
///
/// A skew-symmetric matrix exponentiates to a rotation (the orthogonal
/// `exp(skew)` map):
///
/// ```rust
/// use sonium_solvers_core::dense::DenseMatrix;
/// use sonium_solvers_direct::expm;
///
/// let theta = 0.7_f64;
/// let skew = DenseMatrix::from_row_major_vec(2, 2, vec![0.0, -theta, theta, 0.0]);
/// let rotation = expm(&skew).unwrap();
/// assert!((rotation[[0, 0]] - theta.cos()).abs() < 1.0e-14);
/// assert!((rotation[[0, 1]] + theta.sin()).abs() < 1.0e-14);
/// assert!((rotation[[1, 0]] - theta.sin()).abs() < 1.0e-14);
/// assert!((rotation[[1, 1]] - theta.cos()).abs() < 1.0e-14);
/// ```
///
/// # Errors
///
/// Returns [`ExpmError::NonSquare`] for non-square input,
/// [`ExpmError::NonFiniteInput`] for non-finite entries, and
/// [`ExpmError::NumericalBreakdown`] when the Padé denominator solve fails
/// or the result overflows the scalar range.
pub fn expm<T: ComplexField>(a: &DenseMatrix<T>) -> Result<DenseMatrix<T>, ExpmError> {
    let rows = a.nrows();
    let columns = a.ncols();
    if rows != columns {
        return Err(ExpmError::NonSquare { rows, columns });
    }
    if a.as_slice().iter().any(|value| !value.norm().is_finite()) {
        return Err(ExpmError::NonFiniteInput);
    }
    if rows == 0 || a.as_slice().iter().all(|value| value.is_zero()) {
        return Ok(DenseMatrix::eye(rows));
    }
    let n = rows;

    // 1-norm (maximum absolute column sum): the subordinate norm the θ₁₃
    // backward-error bound is stated in.
    let mut norm = T::Real::zero();
    for j in 0..n {
        let mut column_sum = T::Real::zero();
        for i in 0..n {
            column_sum += a[[i, j]].norm();
        }
        if column_sum > norm {
            norm = column_sum;
        }
    }
    if !norm.is_finite() {
        return Err(ExpmError::NumericalBreakdown {
            operation: "input 1-norm",
        });
    }
    let theta = T::Real::from_f64(THETA_13).ok_or(ExpmError::NumericalBreakdown {
        operation: "Padé radius conversion",
    })?;
    let squarings = if norm <= theta {
        0
    } else {
        let ratio = norm / theta;
        let steps = ratio.log2().ceil();
        if !steps.is_finite() {
            return Err(ExpmError::NumericalBreakdown {
                operation: "scaling parameter",
            });
        }
        steps.to_usize().ok_or(ExpmError::NumericalBreakdown {
            operation: "scaling parameter",
        })?
    };

    // Scale A by 1/2ˢ. An underflowing scale means the norm is so large the
    // exponential cannot be represented; report it instead of returning exp(0).
    let mut scale = T::Real::one();
    for _ in 0..squarings {
        scale /= T::Real::from_f64(2.0).ok_or(ExpmError::NumericalBreakdown {
            operation: "scaling factor",
        })?;
    }
    if scale <= T::Real::zero() {
        return Err(ExpmError::NumericalBreakdown {
            operation: "scaling factor",
        });
    }
    let scaled = DenseMatrix::from_fn(n, n, |i, j| a[[i, j]] * T::from_real(scale));

    // [13/13] Padé coefficients from the closed-form recurrence.
    let degree = T::Real::from_usize(PADE_DEGREE).ok_or(ExpmError::NumericalBreakdown {
        operation: "Padé coefficient conversion",
    })?;
    let mut b = [T::Real::zero(); PADE_DEGREE + 1];
    b[0] = T::Real::one();
    for j in 0..PADE_DEGREE {
        let step = T::Real::from_usize(j).ok_or(ExpmError::NumericalBreakdown {
            operation: "Padé coefficient conversion",
        })?;
        let next = T::Real::from_usize(j + 1).ok_or(ExpmError::NumericalBreakdown {
            operation: "Padé coefficient conversion",
        })?;
        b[j + 1] = b[j] * (degree - step) / ((degree + degree - step) * next);
    }
    if b.iter().any(|value| !value.is_finite()) {
        return Err(ExpmError::NumericalBreakdown {
            operation: "Padé coefficients",
        });
    }

    // Even/odd split evaluated in factored form so only A², A⁴, A⁶ plus
    // three products are needed:
    //   U = A·(A⁶·(b₁₃A⁶ + b₁₁A⁴ + b₉A²) + b₇A⁶ + b₅A⁴ + b₃A² + b₁I)
    //   V = A⁶·(b₁₂A⁶ + b₁₀A⁴ + b₈A²) + b₆A⁶ + b₄A⁴ + b₂A² + b₀I
    let a2 = scaled.matmul(&scaled);
    let a4 = a2.matmul(&a2);
    let a6 = a2.matmul(&a4);
    let u_inner = a6.matmul(&pade_sum(
        n,
        T::Real::zero(),
        &[(b[13], &a6), (b[11], &a4), (b[9], &a2)],
    ));
    let u = scaled.matmul(&pade_sum(
        n,
        b[1],
        &[
            (T::Real::one(), &u_inner),
            (b[7], &a6),
            (b[5], &a4),
            (b[3], &a2),
        ],
    ));
    let v_inner = a6.matmul(&pade_sum(
        n,
        T::Real::zero(),
        &[(b[12], &a6), (b[10], &a4), (b[8], &a2)],
    ));
    let v = pade_sum(
        n,
        b[0],
        &[
            (T::Real::one(), &v_inner),
            (b[6], &a6),
            (b[4], &a4),
            (b[2], &a2),
        ],
    );
    let numerator = DenseMatrix::from_fn(n, n, |i, j| v[[i, j]] + u[[i, j]]);
    let denominator = DenseMatrix::from_fn(n, n, |i, j| v[[i, j]] - u[[i, j]]);
    let factor =
        crate::lu::lu_factorize(&denominator).map_err(|_| ExpmError::NumericalBreakdown {
            operation: "Padé denominator factorization",
        })?;
    let solved =
        factor
            .solve_many(numerator.as_slice(), n)
            .map_err(|_| ExpmError::NumericalBreakdown {
                operation: "Padé denominator solve",
            })?;
    let mut result = DenseMatrix::from_row_major_vec(n, n, solved);
    for _ in 0..squarings {
        result = result.matmul(&result);
    }
    if result
        .as_slice()
        .iter()
        .any(|value| !value.norm().is_finite())
    {
        return Err(ExpmError::NumericalBreakdown {
            operation: "squaring phase",
        });
    }
    Ok(result)
}

/// Evaluate `ident·I + Σ cₖ·Mₖ` for Padé partial sums.
fn pade_sum<T: ComplexField>(
    n: usize,
    ident: T::Real,
    terms: &[(T::Real, &DenseMatrix<T>)],
) -> DenseMatrix<T> {
    DenseMatrix::from_fn(n, n, |i, j| {
        let mut acc = if i == j {
            T::from_real(ident)
        } else {
            T::zero()
        };
        for (coefficient, matrix) in terms {
            acc += T::from_real(*coefficient) * matrix[[i, j]];
        }
        acc
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn assert_matrix_close(actual: &DenseMatrix<f64>, expected: &[f64], tol: f64) {
        assert_eq!(actual.as_slice().len(), expected.len());
        for (got, want) in actual.as_slice().iter().zip(expected) {
            assert!((got - want).abs() <= tol, "got={got} want={want} tol={tol}");
        }
    }

    #[test]
    fn expm_zero_is_exact_identity() {
        let zero = DenseMatrix::<f64>::zeros(3, 3);
        let result = expm(&zero).unwrap();
        assert_eq!(result.as_slice(), DenseMatrix::<f64>::eye(3).as_slice());
    }

    #[test]
    fn expm_diagonal_exponentiates_entries() {
        let diag = DenseMatrix::from_diag(&[0.5, -1.0, 2.0]);
        let result = expm(&diag).unwrap();
        assert_matrix_close(
            &result,
            &[
                0.5_f64.exp(),
                0.0,
                0.0,
                0.0,
                (-1.0_f64).exp(),
                0.0,
                0.0,
                0.0,
                2.0_f64.exp(),
            ],
            1.0e-14,
        );
    }

    #[test]
    fn expm_skew_symmetric_is_rotation() {
        // The orthogonal exp(skew) map: exp of a 2D generator is a rotation.
        for theta in [0.7, 10.0] {
            let skew = DenseMatrix::from_row_major_vec(2, 2, vec![0.0, -theta, theta, 0.0]);
            let result = expm(&skew).unwrap();
            assert_matrix_close(
                &result,
                &[theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
                1.0e-12,
            );
            // Orthogonality: RᵀR = I.
            let gram = result.matmul_transa(&result);
            assert_matrix_close(&gram, &[1.0, 0.0, 0.0, 1.0], 1.0e-12);
        }
    }

    #[test]
    fn expm_nilpotent_matches_truncated_series() {
        // A³ = 0, so exp(A) = I + A + A²/2 exactly.
        let nilpotent = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![0.0, 1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        );
        let result = expm(&nilpotent).unwrap();
        assert_matrix_close(
            &result,
            &[1.0, 1.0, 1.5, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0],
            1.0e-14,
        );
    }

    #[test]
    fn expm_inverse_property_on_nonsymmetric() {
        // Fixed nonsymmetric matrix with 1-norm above θ₁₃: exercises scaling.
        let a = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                1.0, 2.0, -1.0, 0.5, //
                -3.0, 0.5, 1.5, 2.0, //
                0.5, -1.5, 2.5, -2.0, //
                1.5, 1.0, -0.5, 3.0,
            ],
        );
        let forward = expm(&a).unwrap();
        let negated = DenseMatrix::from_fn(4, 4, |i, j| -a[[i, j]]);
        let backward = expm(&negated).unwrap();
        let product = forward.matmul(&backward);
        assert_matrix_close(
            &product,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
            1.0e-11,
        );
    }

    #[test]
    fn expm_scalar_case() {
        let one = DenseMatrix::from_row_major_vec(1, 1, vec![1.5_f64]);
        let result = expm(&one).unwrap();
        let expected = 1.5_f64.exp();
        assert!((result[[0, 0]] - expected).abs() < 1.0e-14 * expected);
    }

    #[test]
    fn expm_rejects_non_square_and_non_finite() {
        let wide = DenseMatrix::from_row_major_vec(2, 3, vec![0.0; 6]);
        assert!(matches!(
            expm(&wide),
            Err(ExpmError::NonSquare {
                rows: 2,
                columns: 3
            })
        ));
        let nan = DenseMatrix::from_row_major_vec(2, 2, vec![0.0, f64::NAN, 0.0, 0.0]);
        assert!(matches!(expm(&nan), Err(ExpmError::NonFiniteInput)));
    }

    #[test]
    fn expm_reports_overflow_instead_of_garbage() {
        // ‖A‖₁ is so large the exponential overflows f64.
        let big = DenseMatrix::from_diag(&[1.0e300, 1.0]);
        assert!(matches!(
            expm(&big),
            Err(ExpmError::NumericalBreakdown { .. })
        ));
    }

    #[test]
    fn expm_supports_f32_and_complex() {
        let theta = 0.7_f32;
        let skew = DenseMatrix::from_row_major_vec(2, 2, vec![0.0, -theta, theta, 0.0]);
        let result = expm(&skew).unwrap();
        for (got, want) in
            result
                .as_slice()
                .iter()
                .zip([theta.cos(), -theta.sin(), theta.sin(), theta.cos()])
        {
            assert!((got - want).abs() < 1.0e-5, "got={got} want={want}");
        }

        let diag = DenseMatrix::from_diag(&[Complex64::new(0.0, 1.0), Complex64::new(1.0, -0.5)]);
        let result = expm(&diag).unwrap();
        let expected = [
            Complex64::new(0.0, 1.0).exp(),
            Complex64::new(1.0, -0.5).exp(),
        ];
        assert!((result[[0, 0]] - expected[0]).norm() < 1.0e-13);
        assert!((result[[1, 1]] - expected[1]).norm() < 1.0e-13);
        assert!(result[[0, 1]].norm() < 1.0e-14);
        assert!(result[[1, 0]].norm() < 1.0e-14);
    }
}
