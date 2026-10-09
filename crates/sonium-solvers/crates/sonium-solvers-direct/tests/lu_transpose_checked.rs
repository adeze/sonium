use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{LuError, lu_factorize};

#[test]
fn transpose_and_adjoint_into_preserve_destination_on_generated_overflow() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![1.0e-200_f64, 0.0, 0.0, 1.0e-200]);
    let factor = lu_factorize(&matrix).expect("finite tiny LU factor should be admitted");

    let mut transpose_destination = [37.0_f64; 2];
    let transpose_before = transpose_destination;
    let transpose_error = factor
        .solve_transpose_into(&[1.0e-200, f64::MAX], &mut transpose_destination)
        .expect_err("finite RHS should overflow the transpose pivot");
    assert!(matches!(
        transpose_error,
        LuError::NumericalBreakdown {
            stage: "transpose forward substitution",
            index: 1,
        }
    ));
    assert_eq!(transpose_destination, transpose_before);

    let mut adjoint_destination = [41.0_f64; 2];
    let adjoint_before = adjoint_destination;
    let adjoint_error = factor
        .solve_conj_transpose_into(&[1.0e-200, f64::MAX], &mut adjoint_destination)
        .expect_err("finite RHS should overflow the adjoint pivot");
    assert!(matches!(
        adjoint_error,
        LuError::NumericalBreakdown {
            stage: "transpose forward substitution",
            index: 1,
        }
    ));
    assert_eq!(adjoint_destination, adjoint_before);
}
