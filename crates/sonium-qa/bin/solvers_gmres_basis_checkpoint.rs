//! Independent CPU QA for exact active-GMRES Krylov-basis replay.

use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, CsrMatrix, GmresBasisCheckpoint, GmresBasisRun, GmresConfig,
    IdentityPreconditioner, SolverResultContract, SolverStatus, try_gmres,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.gmres-basis-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-9;
const MATRIX: [(usize, usize, f64); 10] = [
    (0, 0, 5.0),
    (0, 1, -1.0),
    (1, 0, -1.0),
    (1, 1, 5.0),
    (1, 2, -1.0),
    (2, 1, -1.0),
    (2, 2, 5.0),
    (2, 3, -1.0),
    (3, 2, -1.0),
    (3, 3, 5.0),
];
const RHS: [f64; 4] = [1.0, 2.0, -1.0, 3.0];

fn matvec(x: &[f64; 4]) -> [f64; 4] {
    [
        5.0 * x[0] - x[1],
        -x[0] + 5.0 * x[1] - x[2],
        -x[1] + 5.0 * x[2] - x[3],
        -x[2] + 5.0 * x[3],
    ]
}

fn relative_residual(x: &[f64], rhs: &[f64; 4]) -> f64 {
    let candidate = [x[0], x[1], x[2], x[3]];
    let applied = matvec(&candidate);
    let residual = applied
        .iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    residual / rhs.iter().map(|value| value.abs()).sum::<f64>().max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let started = Instant::now();
    let matrix = CsrMatrix::from_triplets(4, 4, MATRIX.to_vec());
    let config = GmresConfig {
        max_iterations: 30,
        restart: 4,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    };
    let preconditioner = IdentityPreconditioner;
    let fresh = try_gmres(&matrix, &RHS, &config)
        .map_err(|error| format!("fresh GMRES failed: {error:?}"))?;
    let capture_run = GmresBasisCheckpoint::capture(&matrix, &preconditioner, &RHS, &config, 2)
        .map_err(|error| format!("basis capture failed: {error:?}"))?;
    let checkpoint = match &capture_run {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint.clone(),
        GmresBasisRun::Solution(_) => return Err("fixture converged before checkpoint".into()),
    };
    let result_contract = if capture_run.solver_status() == SolverStatus::MaxIterationsReached
        && capture_run.accuracy_status() == AccuracyStatus::NotAttained
        && capture_run.validate_contract().is_ok()
    {
        1.0
    } else {
        0.0
    };
    let serialized = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("basis serialization failed: {error:?}"))?;
    let restored = sonium_solvers::GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&serialized)
        .map_err(|error| format!("basis restore failed: {error:?}"))?;
    let roundtrip_shape_error = if restored.iterations() == checkpoint.iterations()
        && restored.basis_vectors() == checkpoint.basis_vectors()
    {
        0.0
    } else {
        1.0
    };
    let resumed = restored
        .resume(&matrix, &preconditioner, &RHS, &config)
        .map_err(|error| format!("basis resume failed: {error:?}"))?;
    if !fresh.converged || !resumed.converged {
        return Err(format!(
            "basis replay did not converge: fresh={:?}, resumed={:?}",
            fresh.status, resumed.status
        ));
    }
    let solution_error = resumed
        .x
        .iter()
        .zip(&fresh.x)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let residual = relative_residual(&resumed.x, &RHS);
    if solution_error > TOLERANCE || residual > TOLERANCE || roundtrip_shape_error > 0.0 {
        return Err(format!(
            "basis replay error: solution={solution_error:e}, residual={residual:e}"
        ));
    }
    let _ = started;
    Ok(vec![
        ("basis_solution_error", solution_error),
        ("basis_independent_residual", residual),
        (
            "basis_checkpoint_iterations",
            checkpoint.iterations() as f64,
        ),
        (
            "basis_checkpoint_vectors",
            checkpoint.basis_vectors() as f64,
        ),
        (
            "basis_checkpoint_storage_bytes",
            checkpoint.storage_bytes() as f64,
        ),
        ("basis_checkpoint_serialized_bytes", serialized.len() as f64),
        (
            "basis_checkpoint_roundtrip_shape_error",
            roundtrip_shape_error,
        ),
        ("basis_resume_iterations", resumed.iterations as f64),
        ("basis_result_contract", result_contract),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic 4x4 tridiagonal SPD matrix and independent matvec")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-v1")
                .dimension("4x4 CSR; fixed-restart GMRES checkpoint after 2 Arnoldi steps")
                .tolerance("solution and independent residual <= 1e-9")
                .seed("solvers-gmres-basis-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("error") || name.contains("residual") {
                    result = result.maximum(name, TOLERANCE).metric(name, value);
                } else {
                    result = result.metric(name, value);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic 4x4 tridiagonal SPD matrix and independent matvec")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-v1")
                .tolerance("solution and independent residual <= 1e-9")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
