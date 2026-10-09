use approx::assert_relative_eq;
use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_sparse::CsrMatrix;

#[test]
fn checked_dense_constructor_rejects_nonfinite_values_and_thresholds() {
    let mut matrix = DenseMatrix::from_elem(2, 2, 0.0_f64);
    matrix[[0, 0]] = f64::NAN;
    let error = CsrMatrix::try_from_dense(&matrix, 0.0).expect_err("NaN must be rejected");
    assert!(error.to_string().contains("dense value at (0, 0)"));

    let finite = DenseMatrix::from_row_major_vec(1, 1, vec![1.0_f64]);
    let error = CsrMatrix::try_from_dense(&finite, f64::NAN)
        .expect_err("non-finite threshold must be rejected");
    assert!(error.to_string().contains("threshold must be finite"));
}

#[test]
fn checked_dense_constructor_preserves_complex_values_and_thresholding() {
    let matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(2.0, -1.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-3.0, 4.0),
        ],
    );
    let csr = CsrMatrix::try_from_dense(&matrix, 0.5).expect("finite dense matrix should pass");
    assert_eq!(csr.num_rows, 2);
    assert_eq!(csr.num_cols, 2);
    assert_eq!(csr.nnz(), 2);
    assert_relative_eq!(csr.get(0, 0).re, 2.0, epsilon = 1.0e-12);
    assert_relative_eq!(csr.get(0, 0).im, -1.0, epsilon = 1.0e-12);
    assert_relative_eq!(csr.get(1, 1).re, -3.0, epsilon = 1.0e-12);
    assert_relative_eq!(csr.get(1, 1).im, 4.0, epsilon = 1.0e-12);
}
