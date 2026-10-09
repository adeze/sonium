//! Independent CPU evidence for checked 32/64-bit CSR index conversion.

use sonium_qa::QaResult;
use sonium_solvers::direct::SparseLuFactorization;
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.csr-index-width.v1";
const TOLERANCE: f64 = 1.0e-12;
const EXPECTED: [f64; 3] = [0.5, -1.25, 2.0];

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn indexed_u32() -> Result<CsrMatrix<f64>, String> {
    CsrMatrix::try_from_raw_parts_indexed(
        3,
        3,
        [0_u32, 2, 4, 5],
        [0_u32, 1, 0, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, 3.0],
    )
    .map_err(|error| format!("u32 CSR conversion failed: {error:?}"))
}

fn indexed_u64() -> Result<CsrMatrix<f64>, String> {
    CsrMatrix::try_from_raw_parts_indexed(
        3,
        3,
        [0_u64, 2, 4, 5],
        [0_u64, 1, 0, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, 3.0],
    )
    .map_err(|error| format!("u64 CSR conversion failed: {error:?}"))
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64, usize), String> {
    let u32_matrix = indexed_u32()?;
    let u64_matrix = indexed_u64()?;
    if u32_matrix.row_ptrs != u64_matrix.row_ptrs
        || u32_matrix.col_indices != u64_matrix.col_indices
        || u32_matrix.values != u64_matrix.values
    {
        return Err("u32 and u64 CSR representations differ".into());
    }
    let mixed_matrix = CsrMatrix::try_from_raw_parts_indexed(
        3,
        3,
        [0_u32, 2, 4, 5],
        [0_u64, 1, 0, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, 3.0],
    )
    .map_err(|error| format!("mixed CSR conversion failed: {error:?}"))?;
    if mixed_matrix.row_ptrs != u32_matrix.row_ptrs
        || mixed_matrix.col_indices != u32_matrix.col_indices
        || mixed_matrix.values != u32_matrix.values
    {
        return Err("mixed CSR representation differs".into());
    }

    let rhs = u32_matrix.matvec(&EXPECTED);
    let mut u32_solution_error = 0.0;
    let mut u32_residual = 0.0;
    let mut u64_solution_error = 0.0;
    let mut u64_residual = 0.0;
    for (is_u32, matrix) in [(true, u32_matrix), (false, u64_matrix)] {
        let factor = SparseLuFactorization::factorize(&matrix)
            .map_err(|error| format!("{is_u32} CSR sparse-LU factorization failed: {error}"))?;
        let solution = factor
            .solve(&rhs)
            .map_err(|error| format!("{is_u32} CSR sparse-LU solve failed: {error}"))?;
        let solution_error = max_error(&solution, &EXPECTED);
        let residual = max_error(&matrix.matvec(&solution), &rhs);
        if is_u32 {
            u32_solution_error = solution_error;
            u32_residual = residual;
        } else {
            u64_solution_error = solution_error;
            u64_residual = residual;
        }
    }

    let invalid_shape = CsrMatrix::<f64>::try_from_raw_parts_indexed(
        2,
        2,
        [0_u32, 1, 1],
        [0_u32, 1],
        vec![1.0, 2.0],
    )
    .is_err();
    let invalid_column =
        CsrMatrix::<f64>::try_from_raw_parts_indexed(1, 2, [0_u64, 1], [2_u64], vec![1.0]).is_err();
    let rejection_checks = usize::from(invalid_shape) + usize::from(invalid_column);
    if rejection_checks != 2 {
        return Err("indexed CSR malformed-input checks did not reject".into());
    }
    Ok((
        u32_solution_error,
        u32_residual,
        u64_solution_error,
        u64_residual,
        1.0,
        1.0,
        1.0,
        rejection_checks,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            u32_solution_error,
            u32_residual,
            u64_solution_error,
            u64_residual,
            u32_ok,
            u64_ok,
            mixed_ok,
            rejection_checks,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:csr-index-width-v1")
                .reference_checksum("analytic/deterministic-fixture:csr-index-width-v1")
                .metric("u32_conversion_checks", u32_ok)
                .metric("u64_conversion_checks", u64_ok)
                .metric("mixed_conversion_checks", mixed_ok)
                .metric("rejection_checks", rejection_checks as f64)
                .metric("u32_solution_error", u32_solution_error)
                .metric("u32_residual", u32_residual)
                .metric("u64_solution_error", u64_solution_error)
                .metric("u64_residual", u64_residual)
                .maximum("u32_solution_error", TOLERANCE)
                .maximum("u32_residual", TOLERANCE)
                .maximum("u64_solution_error", TOLERANCE)
                .maximum("u64_residual", TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .input_checksum("analytic/deterministic-fixture:csr-index-width-v1")
                .reference_checksum("analytic/deterministic-fixture:csr-index-width-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
