//! Release-facing scalar-family reference for dense direct LU.

use num_complex::Complex32;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, factorize_dense,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.direct-scalar-execution.v1";

fn lu_config() -> DenseDirectConfig {
    DenseDirectConfig {
        matrix_class: MatrixClass::General,
        method: DenseDirectMethod::Lu,
        ..DenseDirectConfig::default()
    }
}

fn run_f32() -> Result<(f64, f64), String> {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f32, 1.0, 2.0, 3.0]);
    let rhs = [6.0_f32, 8.0];
    let expected = [1.0_f32, 2.0];
    let factorization =
        factorize_dense(&matrix, &lu_config()).map_err(|error| error.to_string())?;
    let solution = factorization
        .solve(&rhs)
        .map_err(|error| error.to_string())?;

    let solution_error = solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs() as f64)
        .fold(0.0, f64::max);
    let residual_error = (4.0 * solution[0] + solution[1] - rhs[0])
        .abs()
        .max((2.0 * solution[0] + 3.0 * solution[1] - rhs[1]).abs())
        as f64;
    Ok((solution_error, residual_error))
}

fn run_complex32() -> Result<(f64, f64), String> {
    let matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex32::new(2.0, 1.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(3.0, -2.0),
        ],
    );
    let rhs = [Complex32::new(4.0, 1.0), Complex32::new(7.0, -4.0)];
    let expected = [Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)];
    let factorization =
        factorize_dense(&matrix, &lu_config()).map_err(|error| error.to_string())?;
    let solution = factorization
        .solve(&rhs)
        .map_err(|error| error.to_string())?;

    let solution_error = solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm() as f64)
        .fold(0.0, f64::max);
    let residual = [
        matrix[(0, 0)] * solution[0] + matrix[(0, 1)] * solution[1] - rhs[0],
        matrix[(1, 0)] * solution[0] + matrix[(1, 1)] * solution[1] - rhs[1],
    ];
    let residual_error = residual
        .iter()
        .map(|value| value.norm() as f64)
        .fold(0.0, f64::max);
    Ok((solution_error, residual_error))
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let (f32_solution_error, f32_residual_error) = run_f32()?;
    let (complex32_solution_error, complex32_residual_error) = run_complex32()?;
    Ok((
        f32_solution_error,
        f32_residual_error,
        complex32_solution_error,
        complex32_residual_error,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            f32_solution_error,
            f32_residual_error,
            complex32_solution_error,
            complex32_residual_error,
        )) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("f32_solution_error", f32_solution_error)
                .metric("f32_residual_error", f32_residual_error)
                .metric("complex32_solution_error", complex32_solution_error)
                .metric("complex32_residual_error", complex32_residual_error)
                .maximum("f32_solution_error", 2.0e-5)
                .maximum("f32_residual_error", 2.0e-5)
                .maximum("complex32_solution_error", 2.0e-5)
                .maximum("complex32_residual_error", 2.0e-5)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line();
            if let Ok(json) = record {
                println!("QA_RESULT: {json}");
            }
            eprintln!("DIRECT SCALAR EXECUTION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
