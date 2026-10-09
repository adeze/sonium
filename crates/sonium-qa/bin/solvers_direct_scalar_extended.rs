//! Release-facing scalar and method differential reference for dense direct solves.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, factorize_dense,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.direct-scalar-method-reference.v1";

fn config(matrix_class: MatrixClass, method: DenseDirectMethod) -> DenseDirectConfig {
    DenseDirectConfig {
        matrix_class,
        method,
        ..DenseDirectConfig::default()
    }
}

fn run_f64_lu() -> Result<(f64, f64), String> {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 2.0, 3.0]);
    let rhs = [6.0_f64, 8.0];
    let expected = [1.0_f64, 2.0];
    let factor = factorize_dense(
        &matrix,
        &config(MatrixClass::General, DenseDirectMethod::Lu),
    )
    .map_err(|error| error.to_string())?;
    let solution = factor.solve(&rhs).map_err(|error| error.to_string())?;
    let solution_error = solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let residual_error = (4.0 * solution[0] + solution[1] - rhs[0])
        .abs()
        .max((2.0 * solution[0] + 3.0 * solution[1] - rhs[1]).abs());
    Ok((solution_error, residual_error))
}

fn run_complex64_lu() -> Result<(f64, f64), String> {
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
    let rhs = [Complex64::new(4.0, 1.0), Complex64::new(7.0, -4.0)];
    let expected = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let factor = factorize_dense(
        &matrix,
        &config(MatrixClass::General, DenseDirectMethod::Lu),
    )
    .map_err(|error| error.to_string())?;
    let solution = factor.solve(&rhs).map_err(|error| error.to_string())?;
    let solution_error = solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let residual_error = (0..2)
        .map(|row| {
            (matrix[(row, 0)] * solution[0] + matrix[(row, 1)] * solution[1] - rhs[row]).norm()
        })
        .fold(0.0, f64::max);
    Ok((solution_error, residual_error))
}

fn run_rectangular_qr() -> Result<(f64, f64), String> {
    let matrix =
        DenseMatrix::from_row_major_vec(4, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]);
    let rhs = [2.0_f64, -1.0, 1.0, 5.0];
    let expected = [2.0_f64, -1.0];
    let factor = factorize_dense(
        &matrix,
        &config(MatrixClass::RectangularFullRank, DenseDirectMethod::Qr),
    )
    .map_err(|error| error.to_string())?;
    let solution = factor.solve(&rhs).map_err(|error| error.to_string())?;
    let solution_error = solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let residual_error = (0..4)
        .map(|row| {
            (matrix[(row, 0)] * solution[0] + matrix[(row, 1)] * solution[1] - rhs[row]).abs()
        })
        .fold(0.0, f64::max);
    Ok((solution_error, residual_error))
}

fn main() {
    let started = Instant::now();
    let result = (|| {
        let (f64_solution_error, f64_residual_error) = run_f64_lu()?;
        let (complex64_solution_error, complex64_residual_error) = run_complex64_lu()?;
        let (qr_solution_error, qr_residual_error) = run_rectangular_qr()?;
        Ok::<_, String>((
            f64_solution_error,
            f64_residual_error,
            complex64_solution_error,
            complex64_residual_error,
            qr_solution_error,
            qr_residual_error,
        ))
    })();

    match result {
        Ok((
            f64_solution_error,
            f64_residual_error,
            complex64_solution_error,
            complex64_residual_error,
            qr_solution_error,
            qr_residual_error,
        )) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("f64_solution_error", f64_solution_error)
                .metric("f64_residual_error", f64_residual_error)
                .metric("complex64_solution_error", complex64_solution_error)
                .metric("complex64_residual_error", complex64_residual_error)
                .metric("qr_solution_error", qr_solution_error)
                .metric("qr_residual_error", qr_residual_error)
                .maximum("f64_solution_error", 1.0e-12)
                .maximum("f64_residual_error", 1.0e-12)
                .maximum("complex64_solution_error", 1.0e-12)
                .maximum("complex64_residual_error", 1.0e-12)
                .maximum("qr_solution_error", 1.0e-12)
                .maximum("qr_residual_error", 1.0e-12)
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
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("DIRECT SCALAR/METHOD REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
