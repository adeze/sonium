//! Independent public saddle-point solver contract and reference matrix.

use sonium_qa::QaResult;
use sonium_solvers::{
    BlockPreconditionerConfig, CsrBuilder, CsrMatrix, LinearSolveConfig, LinearSolveError,
    LinearSolverType, SaddlePointConfig, SchurStrategy, solve_saddle_point,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.saddle-point-contract.v1";
const TOLERANCE: f64 = 1.0e-8;
const MATRIX: [[f64; 4]; 4] = [
    [4.0, 1.0, 1.0, 0.0],
    [1.0, 3.0, 0.0, 1.0],
    [1.0, 0.0, 2.0, 0.0],
    [0.0, 1.0, 0.0, 1.5],
];
const KNOWN_SOLUTION: [f64; 4] = [1.0, -0.5, 0.75, 1.5];

fn csr_matrix() -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(4, 4);
    for row in MATRIX {
        builder.add_row_entries(
            row.into_iter()
                .enumerate()
                .filter(|(_, value)| value.abs() > f64::EPSILON),
        );
    }
    builder.finish()
}

fn schur_approximation() -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(2, 2);
    builder.add_row_entries([(0, 2.0)].into_iter());
    builder.add_row_entries([(1, 1.5)].into_iter());
    builder.finish()
}

fn rhs() -> [f64; 4] {
    std::array::from_fn(|row| {
        MATRIX[row]
            .iter()
            .zip(KNOWN_SOLUTION)
            .map(|(value, solution)| value * solution)
            .sum()
    })
}

fn dense_reference(mut matrix: [[f64; 4]; 4], mut rhs: [f64; 4]) -> [f64; 4] {
    for pivot in 0..4 {
        let pivot_row = (pivot..4)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .total_cmp(&matrix[right][pivot].abs())
            })
            .expect("non-empty pivot range");
        assert!(matrix[pivot_row][pivot].abs() > 1.0e-14);
        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);
        for row in (pivot + 1)..4 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for column in pivot..4 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut solution = [0.0; 4];
    for row in (0..4).rev() {
        let tail: f64 = ((row + 1)..4)
            .map(|column| matrix[row][column] * solution[column])
            .sum();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    solution
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn relative_residual(actual: &[f64], expected: &[f64]) -> f64 {
    let scale = expected
        .iter()
        .map(|value| value.abs())
        .sum::<f64>()
        .max(1.0);
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f64>()
        / scale
}

fn config() -> LinearSolveConfig {
    LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        max_iterations: 80,
        tolerance: 1.0e-10,
        ..Default::default()
    }
}

fn saddle_config() -> SaddlePointConfig {
    SaddlePointConfig {
        block1_size: 2,
        schur_strategy: SchurStrategy::External,
        block2_config: BlockPreconditionerConfig::Diagonal,
        ..Default::default()
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = csr_matrix();
    let rhs = rhs();
    let reference = dense_reference(MATRIX, rhs);
    let schur = schur_approximation();
    let solution = solve_saddle_point(&matrix, &rhs, &config(), saddle_config(), Some(&schur))
        .map_err(|error| format!("valid saddle-point solve failed: {error}"))?;

    if !solution.converged {
        return Err(format!(
            "valid saddle-point solve did not converge: {solution:?}"
        ));
    }
    let computed_rhs = matrix.matvec(&solution.x);
    let independent_residual = relative_residual(&computed_rhs, &rhs);
    let reference_error = max_error(&solution.x, &reference);
    if independent_residual > TOLERANCE || reference_error > TOLERANCE {
        return Err(format!(
            "valid saddle-point checks failed: residual={independent_residual:e}, reference_error={reference_error:e}"
        ));
    }
    let assumptions = solution
        .saddle_assumptions
        .ok_or_else(|| "saddle result omitted assumption evidence".to_string())?;
    if assumptions.block1_provenance != sonium_solvers::AssumptionProvenance::IndependentlyValidated
        || assumptions.block2_provenance != sonium_solvers::AssumptionProvenance::NotRequired
    {
        return Err(format!(
            "unexpected saddle assumption evidence: {assumptions:?}"
        ));
    }

    let short_rhs =
        solve_saddle_point(&matrix, &rhs[..3], &config(), saddle_config(), Some(&schur))
            .expect_err("short RHS must be rejected");
    if !matches!(short_rhs, LinearSolveError::DimensionMismatch { .. }) {
        return Err(format!("unexpected short-RHS error: {short_rhs:?}"));
    }

    let unsupported_solver = solve_saddle_point(
        &matrix,
        &rhs,
        &LinearSolveConfig {
            solver: LinearSolverType::Cg,
            ..config()
        },
        saddle_config(),
        Some(&schur),
    )
    .expect_err("CG must be rejected for saddle-point dispatch");
    if !matches!(unsupported_solver, LinearSolveError::InvalidConfig(_)) {
        return Err(format!(
            "unexpected unsupported-solver error: {unsupported_solver:?}"
        ));
    }

    let missing_schur = solve_saddle_point(&matrix, &rhs, &config(), saddle_config(), None)
        .expect_err("external Schur strategy must require an approximation");
    if !matches!(missing_schur, LinearSolveError::InvalidConfig(_)) {
        return Err(format!("unexpected missing-Schur error: {missing_schur:?}"));
    }

    Ok(vec![
        ("valid_reference_error", reference_error),
        ("valid_independent_residual", independent_residual),
        ("contract_checks", 5.0),
    ])
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum("analytic/deterministic-fixture:solvers-saddle-point-contract-v1")
                .reference(
                    "independent dense partial-pivot solution, residual, assumption metadata, and failure classification",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-saddle-point-contract-v1",
                )
                .dimension(
                    "4x4 two-block saddle-point matrix; 2x2 external Schur approximation",
                )
                .formulation(
                    "two-block GMRES saddle-point solve with external Schur approximation and independent dense reference",
                )
                .tolerance("reference and residual <= 1e-8; five contract checks")
                .seed("solvers-saddle-point-contract:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            result
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .seed("solvers-saddle-point-contract:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
