use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{CholeskyError, cholesky_factorize};

#[test]
fn dense_cholesky_into_preserves_destination_on_generated_overflow() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![1.0e-20_f64, 0.0, 0.0, 1.0e-20]);
    let factor = cholesky_factorize(&matrix).expect("finite tiny SPD factor should be admitted");

    let mut destination = [53.0_f64; 2];
    let before = destination;
    let error = factor
        .solve_into(&[1.0e-20, f64::MAX], &mut destination)
        .expect_err("finite RHS should overflow the second forward pivot");

    assert!(matches!(
        error,
        CholeskyError::NumericalBreakdown {
            stage: "forward substitution",
            index: 1,
        }
    ));
    assert_eq!(destination, before);
}
