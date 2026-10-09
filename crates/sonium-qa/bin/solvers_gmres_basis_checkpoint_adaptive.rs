//! Independent CPU evidence for adaptive-restart GMRES basis replay.

use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, CsrMatrix, GmresBasisCheckpoint, GmresBasisRun, GmresConfig,
    IdentityPreconditioner, SolverResultContract, SolverStatus, try_gmres,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.gmres-basis-checkpoint-adaptive.v1";

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![5.0, -1.0, -1.0, 5.0, -1.0, -1.0, 5.0, -1.0, -1.0, 5.0],
    )
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64), String> {
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let rhs = [1.0, 2.0, -1.0, 3.0];
    let config = GmresConfig {
        max_iterations: 30,
        restart: 2,
        tolerance: 1.0e-12,
        min_restart: Some(1),
        max_restart: Some(4),
        ..GmresConfig::default()
    };
    let fresh = try_gmres(&operator, &rhs, &config)
        .map_err(|error| format!("fresh adaptive GMRES failed: {error:?}"))?;
    let captured = GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config, 3)
        .map_err(|error| format!("adaptive capture failed: {error:?}"))?;
    let result_contract = if captured.solver_status() == SolverStatus::MaxIterationsReached
        && captured.accuracy_status() == AccuracyStatus::NotAttained
        && captured.validate_contract().is_ok()
    {
        1.0
    } else {
        0.0
    };
    let checkpoint = match &captured {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint.clone(),
        GmresBasisRun::Solution(_) => {
            return Err("adaptive fixture converged before checkpoint".into());
        }
    };
    let restart_before_resume = checkpoint.restart_parameter() as f64;
    if !(1.0..=4.0).contains(&restart_before_resume) {
        return Err(format!(
            "restart parameter escaped adaptive bounds: {restart_before_resume}"
        ));
    }
    let serialized = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("adaptive serialization failed: {error:?}"))?;
    let restored = GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&serialized)
        .map_err(|error| format!("adaptive restore failed: {error:?}"))?;
    let resumed = restored
        .resume(&operator, &preconditioner, &rhs, &config)
        .map_err(|error| format!("adaptive resume failed: {error:?}"))?;
    if !resumed.converged {
        return Err(format!(
            "adaptive replay did not converge: {:?}",
            resumed.residual
        ));
    }
    if !fresh.converged {
        return Err(format!(
            "fresh adaptive GMRES did not converge: {:?}",
            fresh.residual
        ));
    }
    let solution_error = resumed
        .x
        .iter()
        .zip(&fresh.x)
        .map(|(resumed, fresh)| (resumed - fresh).abs())
        .fold(0.0_f64, f64::max);
    Ok((
        restart_before_resume,
        resumed.residual,
        serialized.len() as f64,
        1.0,
        solution_error,
        result_contract,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            restart,
            residual,
            serialized_bytes,
            bounds_check,
            solution_error,
            result_contract,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .metric("checkpoint_restart_parameter", restart)
                .metric("adaptive_replay_residual", residual)
                .metric("serialized_bytes", serialized_bytes)
                .metric("adaptive_bounds_check", bounds_check)
                .metric("adaptive_solution_error", solution_error)
                .metric("adaptive_result_contract", result_contract)
                .minimum("checkpoint_restart_parameter", 1.0)
                .maximum("adaptive_replay_residual", 1.0e-10)
                .minimum("serialized_bytes", 1.0)
                .minimum("adaptive_bounds_check", 1.0)
                .maximum("adaptive_solution_error", 1.0e-9)
                .reference("analytic 4x4 tridiagonal SPD matrix; adaptive restart bounds")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-adaptive-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-basis-checkpoint-adaptive-v1",
                )
                .dimension("4x4 CSR; restart=2; adaptive bounds [1,4]")
                .tolerance("adaptive replay residual <= 1e-10; restart remains in bounds")
                .seed("solvers-gmres-basis-checkpoint-adaptive:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic adaptive GMRES basis replay")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-adaptive-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-basis-checkpoint-adaptive-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            eprintln!("FAILED: {error}");
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
