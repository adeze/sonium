//! Independent CPU evidence for portable ILU(0) factor checkpoints.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{IluPreconditioner, IluReordering};
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.ilu-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-12;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0],
    )
}

fn relative_residual(matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64; 3]) -> f64 {
    let ax = matrix.matvec(x);
    ax.iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
        / rhs.iter().map(|value| value.abs()).sum::<f64>().max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let source = IluPreconditioner::from_csr_with_reordering(&matrix, IluReordering::Colamd);
    let checkpoint = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("ILU checkpoint serialization failed: {error}"))?;
    let restored = IluPreconditioner::<f64>::from_checkpoint_bytes(&checkpoint)
        .map_err(|error| format!("ILU checkpoint restore failed: {error}"))?;
    let rhs = [7.0, 13.0, 11.0];
    let source_solution = source.apply(&rhs);
    let restored_solution = restored.apply(&rhs);
    let solution_error = source_solution
        .iter()
        .zip(&restored_solution)
        .map(|(source, restored)| (source - restored).abs())
        .fold(0.0, f64::max);
    let residual = relative_residual(&matrix, &restored_solution, &rhs);
    if solution_error > TOLERANCE || residual > TOLERANCE {
        return Err(format!(
            "restored ILU failed: solution_error={solution_error:.3e}, residual={residual:.3e}"
        ));
    }
    Ok(vec![
        ("restored_solution_error", solution_error),
        ("restored_independent_residual", residual),
        ("checkpoint_bytes", checkpoint.len() as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic 3x3 CSR ILU(0) factor round-trip with COLAMD permutation")
                .input_checksum("analytic/deterministic-fixture:ilu-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:ilu-checkpoint-v1")
                .dimension("3x3 CSR; sequential ILU(0) with optional permutation")
                .tolerance("restored solution error and independent residual <= 1e-12")
                .seed("solvers-ilu-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name.contains("error") || name.contains("residual") {
                    result.maximum(name, TOLERANCE).metric(name, value)
                } else {
                    result.metric(name, value)
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic 3x3 CSR ILU(0) factor round-trip with COLAMD permutation")
                .input_checksum("analytic/deterministic-fixture:ilu-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:ilu-checkpoint-v1")
                .tolerance("restored solution error and independent residual <= 1e-12")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
