//! Independent CPU evidence for Block-Jacobi ILUT checkpoint persistence.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{BlockJacobiPreconditioner, IlutConfig};
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.block-jacobi-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-11;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 3, 6, 9, 11],
        vec![0, 1, 2, 0, 1, 3, 0, 1, 2, 1, 3],
        vec![4.0, -1.0, 0.5, -0.5, 3.5, -1.0, 0.25, -0.75, 4.0, -1.0, 3.0],
    )
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let source = BlockJacobiPreconditioner::try_from_csr(
        &matrix(),
        2,
        IlutConfig {
            fill_factor: 4,
            ..Default::default()
        },
    )
    .map_err(|error| format!("Block-Jacobi construction failed: {error:?}"))?;
    let checkpoint = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("Block-Jacobi checkpoint serialization failed: {error:?}"))?;
    let restored = BlockJacobiPreconditioner::<f64>::from_checkpoint_bytes(&checkpoint)
        .map_err(|error| format!("Block-Jacobi checkpoint restore failed: {error:?}"))?;
    let rhs = [1.0, -2.0, 0.5, 3.0];
    let source_solution = source
        .try_apply(&rhs)
        .map_err(|error| format!("source apply failed: {error:?}"))?;
    let restored_solution = restored
        .try_apply(&rhs)
        .map_err(|error| format!("restored apply failed: {error:?}"))?;
    let action_error = source_solution
        .iter()
        .zip(&restored_solution)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f64::max);

    let mut corrupted = checkpoint.clone();
    corrupted[0] ^= 0xff;
    let corruption_rejected =
        BlockJacobiPreconditioner::<f64>::from_checkpoint_bytes(&corrupted).is_err();
    let mut trailing = checkpoint.clone();
    trailing.push(0);
    let trailing_rejected =
        BlockJacobiPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err();
    if action_error > TOLERANCE || !corruption_rejected || !trailing_rejected {
        return Err(format!(
            "Block-Jacobi checkpoint contract failed: action_error={action_error}, corruption_rejected={corruption_rejected}, trailing_rejected={trailing_rejected}"
        ));
    }

    Ok(vec![
        ("checkpoint_bytes", checkpoint.len() as f64),
        ("action_error", action_error),
        ("corruption_rejection", corruption_rejected as u8 as f64),
        ("trailing_byte_rejection", trailing_rejected as u8 as f64),
        ("block_count", source.stats().0 as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("Block-Jacobi ILUT checkpoint round-trip and corruption rejection")
                .input_checksum("analytic/deterministic-fixture:block-jacobi-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:block-jacobi-checkpoint-v1")
                .tolerance("restored action error <= 1e-11; corruption and trailing bytes rejected")
                .seed("solvers-block-jacobi-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
                if name.contains("error") {
                    result = result.maximum(name, TOLERANCE);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("Block-Jacobi ILUT checkpoint round-trip and corruption rejection")
                .input_checksum("analytic/deterministic-fixture:block-jacobi-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:block-jacobi-checkpoint-v1")
                .tolerance("restored action error <= 1e-11; corruption and trailing bytes rejected")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
