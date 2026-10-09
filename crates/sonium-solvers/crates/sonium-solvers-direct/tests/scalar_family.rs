use num_complex::{Complex32, Complex64};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{DenseDirectConfig, DirectFactorization, factorize_dense};

#[test]
fn dense_direct_lu_supports_f32() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f32, 1.0, 1.0, 3.0]);
    let factor = factorize_dense(&matrix, &DenseDirectConfig::default()).unwrap();
    let solution = factor.solve(&[9.0_f32, 8.0]).unwrap();

    assert!((solution[0] - 19.0 / 11.0).abs() < 2.0e-6);
    assert!((solution[1] - 23.0 / 11.0).abs() < 2.0e-6);
}

#[test]
fn dense_direct_lu_supports_complex32() {
    let matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex32::new(2.0, 1.0),
            Complex32::new(1.0, -0.5),
            Complex32::new(0.5, 0.25),
            Complex32::new(3.0, -1.0),
        ],
    );
    let expected = [Complex32::new(1.0, 0.5), Complex32::new(-0.25, 1.25)];
    let rhs = matrix.matvec(&expected);
    let factor = factorize_dense(&matrix, &DenseDirectConfig::default()).unwrap();
    let solution = factor.solve(&rhs).unwrap();

    assert!(
        solution
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).norm() < 3.0e-5)
    );
}

#[test]
fn dense_direct_lu_supports_f64() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 2.0, 3.0]);
    let expected = [1.0_f64, 2.0];
    let rhs = matrix.matvec(&expected);
    let factor = factorize_dense(&matrix, &DenseDirectConfig::default()).unwrap();
    let solution = factor.solve(&rhs).unwrap();

    assert!(
        solution
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-12)
    );
}

#[test]
fn dense_direct_lu_supports_complex64() {
    let matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, -2.0),
        ],
    );
    let expected = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let rhs = matrix.matvec(&expected);
    let factor = factorize_dense(&matrix, &DenseDirectConfig::default()).unwrap();
    let solution = factor.solve(&rhs).unwrap();

    assert!(
        solution
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-12)
    );
}
