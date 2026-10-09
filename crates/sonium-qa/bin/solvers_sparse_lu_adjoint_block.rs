//! Independent CPU evidence for sparse-LU transpose/Hermitian block solves.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-adjoint-block.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:sparse-lu-adjoint-block-v1";
const TOLERANCE: f64 = 1.0e-10;
type Entry = (usize, usize, Complex64);

fn fixture() -> (
    CsrMatrix<Complex64>,
    Vec<Entry>,
    Vec<Complex64>,
    Vec<Complex64>,
) {
    let entries = vec![
        (0, 0, Complex64::new(4.0, 1.0)),
        (0, 1, Complex64::new(1.0, -2.0)),
        (0, 2, Complex64::new(0.5, 0.25)),
        (1, 0, Complex64::new(2.0, 0.5)),
        (1, 1, Complex64::new(5.0, -1.0)),
        (1, 2, Complex64::new(-1.0, 0.75)),
        (2, 0, Complex64::new(0.25, -0.5)),
        (2, 1, Complex64::new(-1.0, 2.0)),
        (2, 2, Complex64::new(3.0, 0.25)),
    ];
    let matrix = CsrMatrix::from_triplets(3, 3, entries.clone());
    let first = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.75, 2.0),
    ];
    let second = vec![
        Complex64::new(-0.25, 0.5),
        Complex64::new(0.5, -0.75),
        Complex64::new(1.25, 0.0),
    ];
    (matrix, entries, first, second)
}

fn adjoint_matvec(entries: &[Entry], x: &[Complex64], hermitian: bool) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); x.len()];
    for &(row, col, value) in entries {
        result[col] += if hermitian { value.conj() } else { value } * x[row];
    }
    result
}

fn block_columns(first: &[Complex64], second: &[Complex64]) -> Vec<Complex64> {
    first
        .iter()
        .zip(second)
        .flat_map(|(&a, &b)| [a, b])
        .collect()
}

fn max_solution_error(actual: &[Complex64], first: &[Complex64], second: &[Complex64]) -> f64 {
    actual
        .chunks_exact(2)
        .zip(first.iter().zip(second))
        .flat_map(|(pair, (&a, &b))| [(pair[0] - a).norm(), (pair[1] - b).norm()])
        .fold(0.0, f64::max)
}

fn max_residual(
    entries: &[Entry],
    actual: &[Complex64],
    rhs_first: &[Complex64],
    rhs_second: &[Complex64],
    hermitian: bool,
) -> f64 {
    let actual_first = actual_column(actual, 0);
    let actual_second = actual_column(actual, 1);
    let first_action = adjoint_matvec(entries, &actual_first, hermitian);
    let second_action = adjoint_matvec(entries, &actual_second, hermitian);
    let mut error = 0.0_f64;
    for row in 0..rhs_first.len() {
        error = error.max((first_action[row] - rhs_first[row]).norm());
        error = error.max((second_action[row] - rhs_second[row]).norm());
    }
    error
}

fn actual_column(block: &[Complex64], column: usize) -> Vec<Complex64> {
    block.iter().skip(column).step_by(2).copied().collect()
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let (matrix, entries, first, second) = fixture();
    let transpose_first = adjoint_matvec(&entries, &first, false);
    let transpose_second = adjoint_matvec(&entries, &second, false);
    let hermitian_first = adjoint_matvec(&entries, &first, true);
    let hermitian_second = adjoint_matvec(&entries, &second, true);
    let transpose_rhs = block_columns(&transpose_first, &transpose_second);
    let hermitian_rhs = block_columns(&hermitian_first, &hermitian_second);

    let factors = vec![
        SparseLuFactorization::factorize(&matrix).map_err(|error| error.to_string())?,
        SparseLuFactorization::factorize_supernodal(&matrix).map_err(|error| error.to_string())?,
    ];
    let mut max_solution = 0.0_f64;
    let mut max_residual_error = 0.0_f64;
    let mut nonfinite_rejections = 0.0_f64;
    let mut preserved_destinations = 0.0_f64;
    for factor in &factors {
        let mut transpose = vec![Complex64::new(9.0, -3.0); 6];
        factor
            .solve_transpose_many_into(&transpose_rhs, &mut transpose, 2)
            .map_err(|error| format!("transpose block solve failed: {error}"))?;
        let mut hermitian = vec![Complex64::new(9.0, -3.0); 6];
        factor
            .solve_hermitian_many_into(&hermitian_rhs, &mut hermitian, 2)
            .map_err(|error| format!("Hermitian block solve failed: {error}"))?;
        max_solution = max_solution
            .max(max_solution_error(&transpose, &first, &second))
            .max(max_solution_error(&hermitian, &first, &second));
        max_residual_error = max_residual_error
            .max(max_residual(
                &entries,
                &transpose,
                &transpose_first,
                &transpose_second,
                false,
            ))
            .max(max_residual(
                &entries,
                &hermitian,
                &hermitian_first,
                &hermitian_second,
                true,
            ));

        let mut non_finite_rhs = transpose_rhs.clone();
        non_finite_rhs[4] = Complex64::new(f64::NAN, 0.0);
        let destination = vec![Complex64::new(7.0, 2.0); 6];
        let mut rejected = destination.clone();
        if factor
            .solve_transpose_many_into(&non_finite_rhs, &mut rejected, 2)
            .is_err()
            && rejected == destination
        {
            nonfinite_rejections += 1.0;
            preserved_destinations += 1.0;
        }
    }
    if max_solution > TOLERANCE
        || max_residual_error > TOLERANCE
        || nonfinite_rejections != 2.0
        || preserved_destinations != 2.0
    {
        return Err(format!(
            "sparse-LU block adjoint metrics failed: solution={max_solution:e}, residual={max_residual_error:e}, rejections={nonfinite_rejections}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("factor_classes", factors.len() as f64)
        .minimum("factor_classes", 2.0)
        .metric("block_columns", 2.0)
        .minimum("block_columns", 2.0)
        .metric("max_solution_error", max_solution)
        .maximum("max_solution_error", TOLERANCE)
        .metric("max_residual", max_residual_error)
        .maximum("max_residual", TOLERANCE)
        .metric("nonfinite_rejections", nonfinite_rejections)
        .minimum("nonfinite_rejections", 2.0)
        .metric("preserved_destinations", preserved_destinations)
        .minimum("preserved_destinations", 2.0)
        .input_checksum(CHECKSUM)
        .reference("independent Complex64 transpose/Hermitian block residual oracle")
        .reference_checksum(CHECKSUM)
        .dimension("3x3 Complex64; two row-major RHS columns; simplicial and supernodal sparse LU")
        .formulation("sparse-LU transpose/Hermitian block solves, residual parity, and transactional non-finite admission")
        .tolerance("maximum block solution/residual error <= 1e-10; two factor classes and two transactional rejections")
        .seed("sparse-lu-adjoint-block:analytic-fixture:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error).elapsed_ms(0);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
