//! Independent CPU evidence for portable ILUT/ILUTP factor checkpoints.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{IluReordering, IlutConfig, IlutPreconditioner};
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.ilut-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-11;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        5,
        5,
        vec![0, 3, 6, 10, 13, 15],
        vec![0, 1, 2, 0, 1, 3, 0, 1, 2, 4, 0, 2, 3, 2, 4],
        vec![
            1e-8, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1e-8, 1.0, 2.0, 1.0, 1.0, 1.0, 1e-8,
        ],
    )
}

fn relative_residual(matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64; 5]) -> f64 {
    let ax = matrix.matvec(x);
    ax.iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
        / rhs.iter().map(|value| value.abs()).sum::<f64>().max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let config = IlutConfig {
        drop_tolerance: 0.0,
        fill_factor: 100,
        pivot_tolerance: 1e-15,
        pivoting: true,
    };
    let source =
        IlutPreconditioner::from_csr_with_reordering(&matrix, &config, IluReordering::Colamd);
    let checkpoint = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("ILUT checkpoint serialization failed: {error}"))?;
    let restored = IlutPreconditioner::<f64>::from_checkpoint_bytes(&checkpoint)
        .map_err(|error| format!("ILUT checkpoint restore failed: {error}"))?;
    let rhs = [7.0, 13.0, 11.0, 5.0, 3.0];
    let source_solution = source.apply(&rhs);
    let restored_solution = restored.apply(&rhs);
    let solution_error = source_solution
        .iter()
        .zip(&restored_solution)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f64::max);
    let residual = relative_residual(&matrix, &restored_solution, &rhs);

    let mut corrupted = checkpoint.clone();
    corrupted[0] ^= 0xff;
    let corruption_rejected = IlutPreconditioner::<f64>::from_checkpoint_bytes(&corrupted).is_err();
    let mut trailing = checkpoint.clone();
    trailing.push(0);
    let trailing_rejected = IlutPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err();
    if solution_error > TOLERANCE
        || residual > TOLERANCE
        || !corruption_rejected
        || !trailing_rejected
    {
        return Err(format!(
            "ILUT checkpoint contract failed: solution_error={solution_error}, residual={residual}, corruption_rejected={corruption_rejected}, trailing_rejected={trailing_rejected}"
        ));
    }

    Ok(vec![
        ("checkpoint_bytes", checkpoint.len() as f64),
        ("solution_error", solution_error),
        ("independent_residual", residual),
        (
            "corruption_rejections",
            f64::from(corruption_rejected as u8),
        ),
        (
            "trailing_byte_rejections",
            f64::from(trailing_rejected as u8),
        ),
        (
            "regularized_pivots",
            source.diagnostics().regularized_pivots as f64,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
        .reference("ILUT/ILUTP factor checkpoint round-trip with independent CSR residual")
                .input_checksum("analytic/deterministic-fixture:ilut-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:ilut-checkpoint-v1")
                .tolerance("restored ILUT solution/residual <= 1e-11; corruption and trailing bytes rejected")
                .seed("solvers-ilut-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
                if name.contains("error") || name.contains("residual") {
                    result = result.maximum(name, TOLERANCE);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
        .reference("ILUT/ILUTP factor checkpoint round-trip with independent CSR residual")
                .input_checksum("analytic/deterministic-fixture:ilut-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:ilut-checkpoint-v1")
                .tolerance("restored ILUT solution/residual <= 1e-11; corruption and trailing bytes rejected")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
