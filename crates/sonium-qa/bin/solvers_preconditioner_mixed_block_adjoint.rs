//! Independent CPU evidence for mixed-precision block adjoint dispatch.

use num_complex::{Complex32, Complex64};
use sonium_qa::QaResult;
use sonium_solvers::mixed_precision::MixedPrecisionPreconditioner;
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-mixed-block-adjoint.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let low = DiagonalPreconditioner::<Complex32>::from_inverse_diagonal(vec![
        Complex32::new(0.5, 0.25),
        Complex32::new(-0.25, 0.5),
        Complex32::new(0.75, -0.125),
    ]);
    let mixed = MixedPrecisionPreconditioner::<Complex64, Complex32>::new(Box::new(low));
    let input = DenseMatrix::from_row_major_vec(
        3,
        2,
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(0.25, 2.0),
            Complex64::new(-2.0, 1.0),
            Complex64::new(1.5, -0.75),
            Complex64::new(0.25, 0.75),
            Complex64::new(-0.5, 0.2),
        ],
    );
    let mut transpose = DenseMatrix::zeros(3, 2);
    mixed
        .try_apply_transpose_block(&input, &mut transpose)
        .map_err(|error| format!("mixed block transpose failed: {error:?}"))?;
    let mut hermitian = DenseMatrix::zeros(3, 2);
    mixed
        .try_apply_hermitian_block(&input, &mut hermitian)
        .map_err(|error| format!("mixed block Hermitian failed: {error:?}"))?;

    let mut max_transpose = 0.0_f64;
    let mut max_hermitian = 0.0_f64;
    let mut checks = 0.0_f64;
    for column in 0..input.ncols() {
        let rhs = input.column_to_vec(column);
        let expected_transpose = mixed
            .try_apply_transpose(&rhs)
            .map_err(|error| format!("mixed scalar transpose failed: {error:?}"))?;
        let expected_hermitian = mixed
            .try_apply_hermitian(&rhs)
            .map_err(|error| format!("mixed scalar Hermitian failed: {error:?}"))?;
        for row in 0..input.nrows() {
            max_transpose =
                max_transpose.max((transpose[[row, column]] - expected_transpose[row]).norm());
            max_hermitian =
                max_hermitian.max((hermitian[[row, column]] - expected_hermitian[row]).norm());
            checks += 2.0;
        }
    }
    Ok((max_transpose, max_hermitian, checks))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((max_transpose, max_hermitian, checks)) => {
            let record = QaResult::passed(CASE_ID)
                .metric("max_transpose_block_column_error", max_transpose)
                .metric("max_hermitian_block_column_error", max_hermitian)
                .metric("identity_check_count", checks)
                .maximum("max_transpose_block_column_error", 1.0e-5)
                .maximum("max_hermitian_block_column_error", 1.0e-5)
                .minimum("identity_check_count", 12.0)
                .reference("independent scalar-vs-block mixed-precision adjoint parity")
                .input_checksum(
                    "analytic/deterministic-fixture:preconditioner-mixed-block-adjoint-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:preconditioner-mixed-block-adjoint-v1",
                )
                .dimension("3x2 Complex64 block; Complex32 diagonal preconditioner")
                .tolerance("12 transpose/Hermitian column checks <= 1e-5")
                .seed("solvers-preconditioner-mixed-block-adjoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .reference("independent scalar-vs-block mixed-precision adjoint parity")
                .input_checksum(
                    "analytic/deterministic-fixture:preconditioner-mixed-block-adjoint-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:preconditioner-mixed-block-adjoint-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            eprintln!("FAILED: {error}");
            println!("QA_RESULT: {}", record.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
