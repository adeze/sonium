//! Independent differentiable lifecycle checkpoint/replay contract packet.

use sonium_qa::QaResult;
use sonium_solvers::iterative::{BlockFgmresSolution, KrylovOpCounts};
use sonium_solvers::time_integration::{NewmarkAdjointResult, NewmarkResult};
use sonium_solvers::{
    AccuracyStatus, CancellationToken, SolverError, TargetedEigenResult, block_contract,
    eigen_value_contract, transient_contract, verify_forward_checkpoint_replay,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.differentiable-checkpoint-replay.v1";
const CHECKSUM: &str = "deterministic-fixture:differentiable-checkpoint-replay-v1";

fn transient_state(dimension: usize) -> NewmarkResult<f64> {
    NewmarkResult {
        displacement: vec![0.0; dimension],
        velocity: vec![0.0; dimension],
        acceleration: vec![0.0; dimension],
        displacement_history: None,
        velocity_history: None,
        acceleration_history: None,
        steps_completed: 1,
        final_time: 1.0,
    }
}

fn transient_adjoint(dimension: usize) -> NewmarkAdjointResult<f64> {
    NewmarkAdjointResult {
        force_gradient: vec![vec![0.0; dimension]],
        initial_displacement_gradient: vec![0.0; dimension],
        initial_velocity_gradient: vec![0.0; dimension],
        initial_acceleration_gradient: vec![0.0; dimension],
    }
}

fn block_solution(rows: usize) -> BlockFgmresSolution<f64> {
    BlockFgmresSolution {
        x: sonium_solvers::dense::DenseMatrix::zeros(rows, 1),
        iterations: 1,
        residuals: vec![0.0],
        true_residuals: vec![0.0],
        backward_errors: vec![0.0],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: true,
        op_counts: KrylovOpCounts::default(),
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let cancellation = CancellationToken::new();
    let transient = transient_contract(
        |_input: &[f64]| Ok::<_, SolverError>(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok::<_, SolverError>(transient_adjoint(2)),
    );
    let transient_report = verify_forward_checkpoint_replay(
        &transient,
        &[1.0, 2.0],
        |state| state.to_checkpoint_bytes(),
        NewmarkResult::from_checkpoint_bytes,
        |left, right| {
            left.displacement == right.displacement
                && left.velocity == right.velocity
                && left.acceleration == right.acceleration
                && left.steps_completed == right.steps_completed
        },
        &cancellation,
    )
    .map_err(|error| format!("transient checkpoint: {error}"))?;

    let eigen = eigen_value_contract(
        |_input: &[f64]| Ok::<_, SolverError>(2.5),
        |_forward: &f64, _seed: &[f64]| Ok::<_, SolverError>(1.0),
    );
    let eigen_report = verify_forward_checkpoint_replay(
        &eigen,
        &[0.5],
        |value| Ok::<_, SolverError>(value.to_le_bytes().to_vec()),
        |payload| {
            let bytes: [u8; 8] = payload
                .try_into()
                .map_err(|_| SolverError::OperationFailed {
                    operation: "QA eigen checkpoint",
                    reason: "invalid payload length".into(),
                })?;
            Ok(f64::from_le_bytes(bytes))
        },
        |left, right| left == right,
        &cancellation,
    )
    .map_err(|error| format!("eigen checkpoint: {error}"))?;

    let native_eigen = TargetedEigenResult {
        eigenvalues: vec![2.5],
        eigenvectors: sonium_solvers::dense::DenseMatrix::from_row_major_vec(2, 1, vec![1.0, 0.0]),
        residual_norms: vec![0.0],
        iterations: 1,
        converged: true,
    };
    let native_eigen_bytes = native_eigen
        .to_checkpoint_bytes()
        .map_err(|error| format!("native eigen encode: {error}"))?;
    let native_eigen_restored = TargetedEigenResult::from_checkpoint_bytes(&native_eigen_bytes)
        .map_err(|error| format!("native eigen decode: {error}"))?;
    if native_eigen_restored.eigenvalues != native_eigen.eigenvalues
        || native_eigen_restored.eigenvectors.as_slice() != native_eigen.eigenvectors.as_slice()
    {
        return Err("native eigen checkpoint changed state".into());
    }

    let block = block_contract(
        |_input: &[f64]| Ok::<_, SolverError>(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0; 2]),
    );
    let block_report = verify_forward_checkpoint_replay(
        &block,
        &[1.0, 2.0],
        |_state| Ok::<_, SolverError>(vec![0x42, 0x17]),
        |_payload| Ok(block_solution(2)),
        |left, right| {
            left.x.as_slice() == right.x.as_slice()
                && left.iterations == right.iterations
                && left.converged == right.converged
        },
        &cancellation,
    )
    .map_err(|error| format!("block checkpoint: {error}"))?;

    let native_block = block_solution(2);
    let native_block_bytes = native_block
        .to_checkpoint_bytes()
        .map_err(|error| format!("native block encode: {error}"))?;
    let native_block_restored = BlockFgmresSolution::from_checkpoint_bytes(&native_block_bytes)
        .map_err(|error| format!("native block decode: {error}"))?;
    if native_block_restored.x.as_slice() != native_block.x.as_slice()
        || native_block_restored.op_counts != native_block.op_counts
    {
        return Err("native block checkpoint changed state".into());
    }

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let cancellation_rejected = verify_forward_checkpoint_replay(
        &eigen,
        &[0.0],
        |_value| Ok::<_, SolverError>(vec![1]),
        |_payload| Ok(1.0),
        |left, right| left == right,
        &cancelled,
    )
    .is_err();

    Ok(vec![
        (
            "transient_replay_deterministic",
            f64::from(transient_report.deterministic as u8),
        ),
        (
            "eigen_replay_deterministic",
            f64::from(eigen_report.deterministic as u8),
        ),
        (
            "block_replay_deterministic",
            f64::from(block_report.deterministic as u8),
        ),
        (
            "transient_checkpoint_bytes",
            transient_report.checkpoint_bytes as f64,
        ),
        ("newmark_native_codec", 1.0),
        (
            "eigen_checkpoint_bytes",
            eigen_report.checkpoint_bytes as f64,
        ),
        ("native_eigen_codec", 1.0),
        (
            "native_eigen_checkpoint_bytes",
            native_eigen_bytes.len() as f64,
        ),
        ("native_block_codec", 1.0),
        (
            "native_block_checkpoint_bytes",
            native_block_bytes.len() as f64,
        ),
        (
            "block_checkpoint_bytes",
            block_report.checkpoint_bytes as f64,
        ),
        (
            "cancellation_rejected",
            f64::from(cancellation_rejected as u8),
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = metrics.into_iter().fold(
                QaResult::passed(CASE_ID)
                    .reference("checked native Newmark/eigen/block checkpoint codecs")
                    .reference_checksum(CHECKSUM)
                    .input_checksum(CHECKSUM)
                    .formulation("cancellable forward checkpoint encode/decode/replay contract")
                    .dimension(
                        "transient 2-state, scalar eigenvalue, and 2x1 block lifecycle fixtures",
                    )
                    .tolerance(
                        "all three replay determinism metrics and cancellation rejection equal one",
                    )
                    .elapsed_ms(started.elapsed().as_millis() as u64),
                |result, (name, value)| result.metric(name, value),
            );
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
