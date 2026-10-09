use num_complex::Complex64;
use sonium_bem::core::solver::{DenseOperator, DenseOperatorError};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::LinearOperator;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn operator() -> DenseOperator {
    DenseOperator::new(DenseMatrix::from_row_major_vec(
        2,
        3,
        vec![
            c(1.0, 0.5),
            c(-2.0, 0.25),
            c(0.75, -1.0),
            c(0.5, -0.75),
            c(3.0, 0.0),
            c(-1.25, 0.5),
        ],
    ))
}

#[test]
fn checked_direct_actions_match_dense_matrix_for_all_adjoint_kinds() {
    let operator = operator();
    let x = vec![c(0.25, -0.5), c(-1.0, 0.75), c(0.5, 0.125)];
    let transpose_x = vec![c(-0.5, 0.25), c(1.25, -0.75)];

    assert_eq!(operator.apply_checked(&x).unwrap(), operator.apply(&x));
    assert_eq!(
        operator.apply_transpose_checked(&transpose_x).unwrap(),
        operator.apply_transpose(&transpose_x)
    );
    assert_eq!(
        operator.apply_hermitian_checked(&transpose_x).unwrap(),
        operator.apply_hermitian(&transpose_x)
    );

    let block = DenseMatrix::from_row_major_vec(
        3,
        2,
        vec![
            c(0.25, 0.0),
            c(-0.5, 0.25),
            c(1.0, -0.75),
            c(0.5, 0.125),
            c(-1.0, 0.5),
            c(0.75, 0.0),
        ],
    );
    let forward = operator.apply_block_checked(&block).unwrap();
    let transpose_block = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![c(0.25, -0.5), c(-1.0, 0.75), c(0.5, 0.125), c(-0.25, 1.0)],
    );
    let transpose = operator
        .apply_transpose_block_checked(&transpose_block)
        .unwrap();
    let hermitian = operator
        .apply_hermitian_block_checked(&transpose_block)
        .unwrap();
    for column in 0..block.ncols() {
        let rhs = (0..block.nrows())
            .map(|row| block[[row, column]])
            .collect::<Vec<_>>();
        let expected = operator.apply_checked(&rhs).unwrap();
        for (row, value) in expected.into_iter().enumerate() {
            assert_eq!(forward[[row, column]], value);
        }
    }
    for column in 0..transpose_block.ncols() {
        let rhs = (0..transpose_block.nrows())
            .map(|row| transpose_block[[row, column]])
            .collect::<Vec<_>>();
        let expected_t = operator.apply_transpose_checked(&rhs).unwrap();
        let expected_h = operator.apply_hermitian_checked(&rhs).unwrap();
        for row in 0..expected_t.len() {
            assert_eq!(transpose[[row, column]], expected_t[row]);
            assert_eq!(hermitian[[row, column]], expected_h[row]);
        }
    }
}

#[test]
fn checked_direct_actions_reject_dimensions_and_non_finite_inputs() {
    let operator = operator();
    assert!(matches!(
        operator.apply_checked(&[c(1.0, 0.0)]),
        Err(DenseOperatorError::InputLength {
            operation: "forward",
            expected: 3,
            actual: 1
        })
    ));
    assert!(matches!(
        operator.apply_transpose_checked(&[c(1.0, 0.0)]),
        Err(DenseOperatorError::InputLength {
            operation: "transpose",
            expected: 2,
            actual: 1
        })
    ));
    let mut output = vec![c(0.0, 0.0); 1];
    assert!(matches!(
        operator.apply_into_checked(&[c(1.0, 0.0); 3], &mut output),
        Err(DenseOperatorError::OutputLength {
            operation: "forward",
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        operator.apply_checked(&[c(f64::NAN, 0.0), c(0.0, 0.0), c(0.0, 0.0)]),
        Err(DenseOperatorError::NonFiniteInput {
            operation: "forward",
            index: 0
        })
    ));
    let bad_block = DenseMatrix::zeros(4, 1);
    assert!(matches!(
        operator.apply_block_checked(&bad_block),
        Err(DenseOperatorError::BlockInputRows {
            operation: "forward block",
            expected: 3,
            actual: 4
        })
    ));
    let nonfinite_block =
        DenseMatrix::from_row_major_vec(3, 1, vec![c(f64::NAN, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    assert!(matches!(
        operator.apply_block_checked(&nonfinite_block),
        Err(DenseOperatorError::NonFiniteInput {
            operation: "forward block",
            index: 0
        })
    ));
}

#[test]
fn checked_direct_actions_reject_non_finite_coefficients_and_outputs() {
    let bad_matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![c(1.0, 0.0), c(f64::NAN, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
    );
    let bad_operator = DenseOperator::new(bad_matrix);
    assert!(matches!(
        bad_operator.apply_checked(&[c(1.0, 0.0), c(0.0, 0.0)]),
        Err(DenseOperatorError::NonFiniteCoefficient { index: 1 })
    ));

    let overflow_operator =
        DenseOperator::new(DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0e308, 0.0)]));
    assert!(matches!(
        overflow_operator.apply_checked(&[c(1.0e308, 0.0)]),
        Err(DenseOperatorError::NonFiniteOutput {
            operation: "forward",
            index: 0
        })
    ));
}
