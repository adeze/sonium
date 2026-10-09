//! Independent CPU contract for zero and linearly dependent block-GMRES RHS columns.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{BlockGmresConfig, CsrBuilder, try_block_gmres};
use std::time::Instant;

const CASE_ID: &str = "solvers.block-gmres-rank-deficiency.v1";
const INPUT_CHECKSUM: &str =
    "analytic/deterministic-fixture:solvers-block-gmres-rank-deficiency-v1";
const TOLERANCE: f64 = 1.0e-9;

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let n = 32;
    let mut builder = CsrBuilder::new(n, n);
    for row in 0..n {
        let mut entries = Vec::with_capacity(3);
        if row > 0 {
            entries.push((row - 1, -1.0));
        }
        entries.push((row, 4.0));
        if row + 1 < n {
            entries.push((row + 1, -1.0));
        }
        builder.add_row_entries(entries.into_iter());
    }
    let matrix = builder.finish();

    // Column 0 is exactly zero; columns 1 and 2 are linearly dependent;
    // column 3 is independent. This exercises both initial-block deflation
    // and the zero-RHS convergence path in one deterministic solve.
    let rhs = DenseMatrix::from_fn(n, 4, |i, column| {
        let primary = (i as f64 * 0.31).sin() + 0.25;
        match column {
            0 => 0.0,
            1 => primary,
            2 => 2.0 * primary,
            _ => (i as f64 * 0.47).cos() - 0.1,
        }
    });
    let config = BlockGmresConfig {
        max_iterations: 50,
        restart: 16,
        tolerance: 1.0e-10,
        true_residual_check: true,
        exact_residual_check_interval: 0,
    };
    let solution = try_block_gmres(&matrix, &rhs, &config)
        .map_err(|error| format!("block GMRES rank-deficiency solve failed: {error}"))?;

    if !solution.converged {
        return Err(format!(
            "rank-deficient block did not converge: {:?}",
            solution.true_residuals
        ));
    }

    let mut zero_solution_max: f64 = 0.0;
    let mut scaled_solution_error: f64 = 0.0;
    for i in 0..n {
        zero_solution_max = zero_solution_max.max(solution.x[[i, 0]].abs());
        scaled_solution_error =
            scaled_solution_error.max((solution.x[[i, 2]] - 2.0 * solution.x[[i, 1]]).abs());
    }
    let max_true_residual = solution
        .true_residuals
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    if zero_solution_max > TOLERANCE {
        return Err(format!(
            "zero RHS solution is nonzero: {zero_solution_max:.3e}"
        ));
    }
    if scaled_solution_error > TOLERANCE {
        return Err(format!(
            "scaled dependent RHS solution mismatch: {scaled_solution_error:.3e}"
        ));
    }
    if max_true_residual > TOLERANCE {
        return Err(format!("true residual too large: {max_true_residual:.3e}"));
    }

    Ok(vec![
        ("converged", 1.0),
        ("zero_solution_max", zero_solution_max),
        ("scaled_solution_error", scaled_solution_error),
        ("max_true_residual", max_true_residual),
        ("iterations", solution.iterations as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM);
            for (name, value) in metrics {
                result = result.metric(name, value);
                if name != "converged" && name != "iterations" {
                    result = result.maximum(name, TOLERANCE);
                }
            }
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
