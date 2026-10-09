//! Checked direct dense BEM operator actions without launching a simulation.

use num_complex::Complex64;
use sonium_bem::core::solver::{DenseOperator, DenseOperatorError};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;

const CASE_ID: &str = "bem.dense-operator-checked-admission.v1";

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<(usize, usize, usize, f64), String> {
    let matrix = DenseMatrix::from_row_major_vec(
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
    );
    let operator = DenseOperator::new(matrix.clone());
    let x = vec![c(0.25, -0.5), c(-1.0, 0.75), c(0.5, 0.125)];
    let xt = vec![c(-0.5, 0.25), c(1.25, -0.75)];
    let mut parity: f64 = 0.0;
    parity = parity.max(max_error(
        &operator.apply_checked(&x).map_err(|e| e.to_string())?,
        &matrix.matvec(&x),
    ));
    let mut expected = vec![Complex64::default(); 3];
    matrix.matvec_transpose_into(&xt, &mut expected);
    parity = parity.max(max_error(
        &operator
            .apply_transpose_checked(&xt)
            .map_err(|e| e.to_string())?,
        &expected,
    ));
    matrix.matvec_hermitian_into(&xt, &mut expected);
    parity = parity.max(max_error(
        &operator
            .apply_hermitian_checked(&xt)
            .map_err(|e| e.to_string())?,
        &expected,
    ));
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
    parity = parity.max(max_error(
        operator
            .apply_block_checked(&block)
            .map_err(|e| e.to_string())?
            .as_slice(),
        matrix.matmul(&block).as_slice(),
    ));
    let adjoint_input = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![c(0.25, -0.5), c(-1.0, 0.75), c(0.5, 0.125), c(-0.25, 1.0)],
    );
    let transpose_block = operator
        .apply_transpose_block_checked(&adjoint_input)
        .map_err(|e| e.to_string())?;
    let hermitian_block = operator
        .apply_hermitian_block_checked(&adjoint_input)
        .map_err(|e| e.to_string())?;
    for column in 0..2 {
        let mut expected_t = vec![Complex64::default(); 3];
        let mut expected_h = vec![Complex64::default(); 3];
        let input = [adjoint_input[[0, column]], adjoint_input[[1, column]]];
        matrix.matvec_transpose_into(&input, &mut expected_t);
        matrix.matvec_hermitian_into(&input, &mut expected_h);
        for row in 0..3 {
            parity = parity.max((transpose_block[[row, column]] - expected_t[row]).norm());
            parity = parity.max((hermitian_block[[row, column]] - expected_h[row]).norm());
        }
    }
    let mut valid_output = vec![Complex64::default(); 2];
    operator
        .apply_into_checked(&x, &mut valid_output)
        .map_err(|e| e.to_string())?;
    let mut dimension_rejections = 0;
    if matches!(
        operator.apply_checked(&x[..2]),
        Err(DenseOperatorError::InputLength { .. })
    ) {
        dimension_rejections += 1;
    }
    if matches!(
        operator.apply_transpose_checked(&x),
        Err(DenseOperatorError::InputLength { .. })
    ) {
        dimension_rejections += 1;
    }
    if matches!(
        operator.apply_block_checked(&DenseMatrix::zeros(4, 1)),
        Err(DenseOperatorError::BlockInputRows { .. })
    ) {
        dimension_rejections += 1;
    }
    let mut nonfinite_rejections = 0;
    if matches!(
        operator.apply_checked(&[c(f64::NAN, 0.0), c(0.0, 0.0), c(0.0, 0.0)]),
        Err(DenseOperatorError::NonFiniteInput { .. })
    ) {
        nonfinite_rejections += 1;
    }
    let bad_matrix = DenseOperator::new(DenseMatrix::from_row_major_vec(
        1,
        1,
        vec![c(f64::NAN, 0.0)],
    ));
    if matches!(
        bad_matrix.apply_checked(&[c(1.0, 0.0)]),
        Err(DenseOperatorError::NonFiniteCoefficient { .. })
    ) {
        nonfinite_rejections += 1;
    }
    let overflow = DenseOperator::new(DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0e308, 0.0)]));
    if matches!(
        overflow.apply_checked(&[c(1.0e308, 0.0)]),
        Err(DenseOperatorError::NonFiniteOutput { .. })
    ) {
        nonfinite_rejections += 1;
    }
    if parity > 1.0e-12 || dimension_rejections != 3 || nonfinite_rejections != 3 {
        return Err(format!(
            "direct dense admission mismatch: parity={parity}, dimension_rejections={dimension_rejections}, nonfinite_rejections={nonfinite_rejections}"
        ));
    }
    Ok((6, dimension_rejections, nonfinite_rejections, parity))
}

fn main() {
    match run() {
        Ok((checked_actions, dimension_rejections, nonfinite_rejections, parity)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("checked_actions", checked_actions as f64)
                .metric("dimension_rejections", dimension_rejections as f64)
                .metric("nonfinite_rejections", nonfinite_rejections as f64)
                .metric("max_parity_error", parity)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense direct forward/transpose/Hermitian and block actions")
                .dimension("2x3 complex dense direct operator")
                .formulation("checked CPU DenseOperator admission")
                .seed("bem-dense-operator-checked-admission:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("checked direct dense operator admission")
                .dimension("2x3 complex dense direct operator")
                .formulation("checked CPU DenseOperator admission")
                .seed("bem-dense-operator-checked-admission:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("Dense operator checked admission failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
