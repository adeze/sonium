use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;

pub(crate) fn real_result_shape_valid(
    eigenvalues: &[f64],
    eigenvectors: &DenseMatrix<f64>,
    residual_norms: &[f64],
    allow_empty: bool,
) -> bool {
    (allow_empty || !eigenvalues.is_empty())
        && eigenvalues.len() == residual_norms.len()
        && eigenvectors.nrows() > 0
        && eigenvectors.ncols() == eigenvalues.len()
        && eigenvalues.iter().all(|value| value.is_finite())
        && residual_norms
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && eigenvectors
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
}

pub(crate) fn complex_result_shape_valid(
    eigenvalues: &[Complex64],
    eigenvectors: &DenseMatrix<Complex64>,
    residual_norms: &[f64],
    allow_empty: bool,
) -> bool {
    (allow_empty || !eigenvalues.is_empty())
        && eigenvalues.len() == residual_norms.len()
        && eigenvectors.nrows() > 0
        && eigenvectors.ncols() == eigenvalues.len()
        && eigenvalues
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
        && residual_norms
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && eigenvectors
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
}
