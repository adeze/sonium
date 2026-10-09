//! Independent CPU QA for scalar differentiable forward/adjoint continuation.

use sonium_qa::QaResult;
use sonium_solvers::{
    AdjointSolve, CsrMatrix, DifferentiableLinearSolve, GmresConfig, IdentityPreconditioner,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.differentiable-continuation.v1";
const TOLERANCE: f64 = 1.0e-9;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ],
    )
}

fn max_delta(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let operator = matrix();
    let handle = DifferentiableLinearSolve::new(
        &operator,
        &IdentityPreconditioner,
        GmresConfig {
            max_iterations: 30,
            restart: 3,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        },
    )
    .map_err(|error| format!("differentiable handle setup failed: {error}"))?;

    let forward_rhs = [1.0, 2.0, -1.0];
    let adjoint_rhs = [3.0, -2.0, 1.0];
    let forward = handle
        .solve_forward(&forward_rhs)
        .map_err(|error| format!("scalar forward solve failed: {error}"))?;
    let resumed_forward = handle
        .solve_forward_with_guess(&forward_rhs, Some(&forward.x))
        .map_err(|error| format!("scalar forward continuation failed: {error}"))?;
    let adjoint = handle
        .solve_adjoint(&adjoint_rhs, AdjointSolve::Hermitian)
        .map_err(|error| format!("scalar adjoint solve failed: {error}"))?;
    let resumed_adjoint = handle
        .solve_adjoint_with_guess(&adjoint_rhs, AdjointSolve::Hermitian, Some(&adjoint.x))
        .map_err(|error| format!("scalar adjoint continuation failed: {error}"))?;

    let forward_delta = max_delta(&forward.x, &resumed_forward.x);
    let adjoint_delta = max_delta(&adjoint.x, &resumed_adjoint.x);
    if !forward.converged
        || !adjoint.converged
        || !resumed_forward.converged
        || !resumed_adjoint.converged
        || resumed_forward.iterations != 0
        || resumed_adjoint.iterations != 0
        || resumed_forward.residual > TOLERANCE
        || resumed_adjoint.residual > TOLERANCE
        || forward_delta > TOLERANCE
        || adjoint_delta > TOLERANCE
    {
        return Err(format!(
            "scalar continuation contract failed: forward iterations={}, adjoint iterations={}, forward residual={:e}, adjoint residual={:e}, deltas=({forward_delta:e}, {adjoint_delta:e})",
            resumed_forward.iterations,
            resumed_adjoint.iterations,
            resumed_forward.residual,
            resumed_adjoint.residual,
        ));
    }

    Ok(vec![
        ("forward_residual", resumed_forward.residual),
        ("adjoint_residual", resumed_adjoint.residual),
        (
            "forward_continuation_iterations",
            resumed_forward.iterations as f64,
        ),
        (
            "adjoint_continuation_iterations",
            resumed_adjoint.iterations as f64,
        ),
        ("forward_solution_delta", forward_delta),
        ("adjoint_solution_delta", adjoint_delta),
        ("continuation_contract", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic symmetric 3x3 scalar differentiable forward/adjoint continuation")
                .input_checksum("analytic/deterministic-fixture:differentiable-continuation-v1")
                .reference_checksum("analytic/deterministic-fixture:differentiable-continuation-v1")
                .dimension("3x3 CSR; scalar forward and Hermitian-adjoint RHS")
                .tolerance("converged warm starts resume in zero iterations with residual and solution delta <= 1e-9")
                .seed("solvers-differentiable-continuation:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("residual") || name.contains("delta") {
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
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("DIFFERENTIABLE CONTINUATION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
