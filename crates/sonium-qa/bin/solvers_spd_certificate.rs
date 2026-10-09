//! Independent CPU reference for the scalable SPD certificate path.

use sonium_qa::QaResult;
use sonium_solvers::{
    AssumptionProvenance, CsrMatrix, LinearSolveConfig, LinearSolverType, LinearSystemAssumption,
    solve_real_with_assumption,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.spd-scalable-certificate.v1";
const SIZE: usize = 1_025;

fn scaled_spd_matrix() -> CsrMatrix<f64> {
    let mut row_ptrs = Vec::with_capacity(SIZE + 1);
    let mut col_indices = Vec::with_capacity(3 * SIZE - 2);
    let mut values = Vec::with_capacity(3 * SIZE - 2);
    row_ptrs.push(0);
    for row in 0..SIZE {
        let row_scale = if row.is_multiple_of(2) { 0.5 } else { 2.0 };
        if row > 0 {
            let column_scale = if (row - 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row - 1);
            values.push(1.0 / (row_scale * column_scale));
        }
        col_indices.push(row);
        values.push(3.0 / (row_scale * row_scale));
        if row + 1 < SIZE {
            let column_scale = if (row + 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row + 1);
            values.push(1.0 / (row_scale * column_scale));
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(SIZE, SIZE, row_ptrs, col_indices, values)
}

fn unscaled_spd_without_strict_dominance() -> CsrMatrix<f64> {
    let mut row_ptrs = Vec::with_capacity(SIZE + 1);
    let mut col_indices = Vec::with_capacity(3 * SIZE - 2);
    let mut values = Vec::with_capacity(3 * SIZE - 2);
    row_ptrs.push(0);
    for row in 0..SIZE {
        if row > 0 {
            col_indices.push(row - 1);
            values.push(-1.0);
        }
        col_indices.push(row);
        values.push(2.0);
        if row + 1 < SIZE {
            col_indices.push(row + 1);
            values.push(-1.0);
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(SIZE, SIZE, row_ptrs, col_indices, values)
}

fn max_residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, bool), String> {
    let matrix = scaled_spd_matrix();
    let expected = vec![1.0; SIZE];
    let rhs = matrix.matvec(&expected);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 128,
        tolerance: 1.0e-10,
        ..Default::default()
    };
    let result = solve_real_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .map_err(|error| format!("scaled SPD solve failed: {error}"))?;
    if !result.converged
        || result.assumption_provenance != AssumptionProvenance::ScalableCertificate
    {
        return Err(format!(
            "unexpected certificate result: converged={} provenance={:?}",
            result.converged, result.assumption_provenance
        ));
    }
    let residual = max_residual(&matrix, &result.x, &rhs);

    let unscaled = unscaled_spd_without_strict_dominance();
    let unscaled_rhs = vec![1.0; SIZE];
    let rejected = matches!(
        solve_real_with_assumption(
            &unscaled,
            &unscaled_rhs,
            &config,
            LinearSystemAssumption::PositiveDefiniteCertified,
        ),
        Err(sonium_solvers::LinearSolveError::InvalidConfig(_))
    );
    Ok((residual, rejected))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((residual, rejected)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference(
                    "independent Jacobi-scaled Gershgorin SPD certificate and rejection fixture",
                )
                .input_checksum("analytic/deterministic-fixture:spd-scalable-certificate-v1")
                .reference_checksum("analytic/deterministic-fixture:spd-scalable-certificate-v1")
                .dimension("1025x1025 symmetric tridiagonal CSR; alternating diagonal scaling")
                .tolerance(
                    "certified CG provenance is ScalableCertificate; independent residual <= 1e-7",
                )
                .metric("scaled_spd_max_true_residual", residual)
                .metric("uncertified_spd_rejected", if rejected { 1.0 } else { 0.0 })
                .maximum("scaled_spd_max_true_residual", 1.0e-7)
                .minimum("uncertified_spd_rejected", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference(
                    "independent Jacobi-scaled Gershgorin SPD certificate and rejection fixture",
                )
                .input_checksum("analytic/deterministic-fixture:spd-scalable-certificate-v1")
                .reference_checksum("analytic/deterministic-fixture:spd-scalable-certificate-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
