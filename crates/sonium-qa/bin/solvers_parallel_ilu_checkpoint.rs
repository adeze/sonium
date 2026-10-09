//! Independent CPU evidence for graph-coloring and fixed-point ILU checkpoints.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{IluColoringPreconditioner, IluFixedPointPreconditioner};
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.parallel-ilu-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-12;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        5,
        5,
        vec![0, 3, 6, 10, 13, 15],
        vec![0, 1, 2, 0, 1, 3, 0, 1, 2, 4, 0, 2, 3, 2, 4],
        vec![
            16.0, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 16.0, 1.0, 2.0, 1.0, 1.0, 1.0, 16.0,
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
    let rhs = [7.0, 13.0, 11.0, 5.0, 3.0];
    let coloring = IluColoringPreconditioner::try_from_csr(&matrix)
        .map_err(|error| format!("colored ILU setup failed: {error}"))?;
    let coloring_bytes = coloring
        .to_checkpoint_bytes()
        .map_err(|error| format!("colored ILU serialization failed: {error}"))?;
    let coloring_restored =
        IluColoringPreconditioner::<f64>::from_checkpoint_bytes(&coloring_bytes)
            .map_err(|error| format!("colored ILU restore failed: {error}"))?;
    let coloring_error = max_difference(&coloring.apply(&rhs), &coloring_restored.apply(&rhs));
    if coloring_error > TOLERANCE || coloring.level_stats() != coloring_restored.level_stats() {
        return Err(format!(
            "colored ILU parity failed: error {coloring_error:e}"
        ));
    }

    let fixed = IluFixedPointPreconditioner::try_from_csr_with_tolerance(&matrix, 3, 1e-12)
        .map_err(|error| format!("fixed-point ILU setup failed: {error}"))?;
    let fixed_bytes = fixed
        .to_checkpoint_bytes()
        .map_err(|error| format!("fixed-point ILU serialization failed: {error}"))?;
    let fixed_restored = IluFixedPointPreconditioner::<f64>::from_checkpoint_bytes(&fixed_bytes)
        .map_err(|error| format!("fixed-point ILU restore failed: {error}"))?;
    let fixed_error = max_difference(&fixed.apply(&rhs), &fixed_restored.apply(&rhs));
    if fixed_error > TOLERANCE || fixed.stats() != fixed_restored.stats() {
        return Err(format!(
            "fixed-point ILU parity failed: error {fixed_error:e}"
        ));
    }

    let mut corrupted = coloring_bytes.clone();
    corrupted[0] ^= 0xff;
    let coloring_corruption_rejected =
        IluColoringPreconditioner::<f64>::from_checkpoint_bytes(&corrupted).is_err();
    let mut trailing = fixed_bytes.clone();
    trailing.push(0);
    let fixed_trailing_rejected =
        IluFixedPointPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err();
    if !coloring_corruption_rejected || !fixed_trailing_rejected {
        return Err("parallel ILU corruption boundary was accepted".into());
    }

    Ok(vec![
        ("coloring_checkpoint_bytes", coloring_bytes.len() as f64),
        ("fixed_point_checkpoint_bytes", fixed_bytes.len() as f64),
        ("coloring_action_error", coloring_error),
        ("fixed_point_action_error", fixed_error),
        (
            "coloring_corruption_rejections",
            if coloring_corruption_rejected {
                1.0
            } else {
                0.0
            },
        ),
        (
            "fixed_point_trailing_rejections",
            if fixed_trailing_rejected { 1.0 } else { 0.0 },
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("colored and fixed-point ILU checkpoint action and metadata parity")
                .input_checksum("analytic/deterministic-fixture:parallel-ilu-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:parallel-ilu-checkpoint-v1")
                .tolerance(
                    "parallel ILU action errors <= 1e-12; corruption and trailing bytes rejected",
                )
                .seed("solvers-parallel-ilu-checkpoint:cpu:v1")
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
                .reference("colored and fixed-point ILU checkpoint action and metadata parity")
                .input_checksum("analytic/deterministic-fixture:parallel-ilu-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:parallel-ilu-checkpoint-v1")
                .tolerance(
                    "parallel ILU action errors <= 1e-12; corruption and trailing bytes rejected",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
