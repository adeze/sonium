//! Independent CPU evidence for portable ParILUT factor checkpoints.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{ParIlutConfig, ParIlutPreconditioner};
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.parilut-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-12;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        5,
        5,
        vec![0, 3, 6, 10, 13, 15],
        vec![0, 1, 2, 0, 1, 3, 0, 1, 2, 4, 0, 2, 3, 2, 4],
        vec![
            4.0, 0.2, 0.1, 0.2, 5.0, 0.2, 0.1, 0.2, 6.0, 0.2, 0.1, 0.2, 5.0, 0.2, 4.0,
        ],
    )
}

fn max_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let config = ParIlutConfig {
        sweeps: 3,
        add_candidates: true,
        max_candidates_per_row: 3,
        drop_tolerance: 1e-4,
        track_residual: true,
    };
    let source = ParIlutPreconditioner::try_from_csr_with_config(&matrix, &config)
        .map_err(|error| format!("ParILUT setup failed: {error}"))?;
    let checkpoint = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("ParILUT checkpoint serialization failed: {error}"))?;
    let restored = ParIlutPreconditioner::<f64>::from_checkpoint_bytes(&checkpoint)
        .map_err(|error| format!("ParILUT checkpoint restore failed: {error}"))?;

    let rhs = [7.0, 13.0, 11.0, 5.0, 3.0];
    let source_action = source.apply(&rhs);
    let restored_action = restored.apply(&rhs);
    let action_error = max_difference(&source_action, &restored_action);
    if source.residual_history().len() != restored.residual_history().len() {
        return Err("source/restored residual-history lengths differ".into());
    }
    let history_error = max_difference(source.residual_history(), restored.residual_history());
    if action_error > TOLERANCE {
        return Err(format!("source/restored action error {action_error:e}"));
    }
    if history_error > TOLERANCE {
        return Err(format!(
            "source/restored residual-history error {history_error:e}"
        ));
    }
    if source.factor_nnz() != restored.factor_nnz() {
        return Err("source/restored factor nnz metadata differs".into());
    }

    let mut corrupted = checkpoint.clone();
    corrupted[0] ^= 0xff;
    let corruption_rejected =
        ParIlutPreconditioner::<f64>::from_checkpoint_bytes(&corrupted).is_err();
    if !corruption_rejected {
        return Err("corrupted ParILUT checkpoint was accepted".into());
    }
    let mut trailing = checkpoint;
    trailing.push(0);
    let trailing_rejected = ParIlutPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err();
    if !trailing_rejected {
        return Err("ParILUT checkpoint with trailing bytes was accepted".into());
    }

    Ok(vec![
        ("checkpoint_bytes", trailing.len() as f64 - 1.0),
        ("action_error", action_error),
        ("history_error", history_error),
        ("history_entries", source.residual_history().len() as f64),
        ("factor_l_nnz", source.factor_nnz().0 as f64),
        ("factor_u_nnz", source.factor_nnz().1 as f64),
        (
            "corruption_rejections",
            if corruption_rejected { 1.0 } else { 0.0 },
        ),
        (
            "trailing_byte_rejections",
            if trailing_rejected { 1.0 } else { 0.0 },
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("ParILUT factor checkpoint round-trip with independent metadata parity")
                .input_checksum("analytic/deterministic-fixture:parilut-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:parilut-checkpoint-v1")
                .tolerance("source/restored action and residual-history errors <= 1e-12; corruption and trailing bytes rejected")
                .seed("solvers-parilut-checkpoint:cpu:v1")
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
                .reference("ParILUT factor checkpoint round-trip with independent metadata parity")
                .input_checksum("analytic/deterministic-fixture:parilut-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:parilut-checkpoint-v1")
                .tolerance("source/restored action and residual-history errors <= 1e-12; corruption and trailing bytes rejected")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
