//! Release-backed result contracts for solver families outside the Krylov core.

use sonium_qa::QaResult;
use sonium_solvers::direct::{
    DenseDirectError, IrResult, LuError, SparseRefinementProvenance, SparseRefinementResult,
    SparseRefinementStopReason,
};
use sonium_solvers::nonlinear::AndersonError;
use sonium_solvers::optimization::OptimError;
use sonium_solvers::preconditioners::ChebyshevError;
use sonium_solvers::sparse::NumaSpmvError;
use sonium_solvers::time_integration::RevolveError;
use sonium_solvers::{
    AccuracyStatus, AndersonSolution, DifferentiableSolveResult, GmresSolution, KrylovOpCounts,
    LinearSolveError, NewmarkAdjointResult, NewtonSolution, OptimResult, OptimStatus, SolverError,
    SolverErrorKind, SolverFailure, SolverResultContract, SolverStatus,
};
use std::time::Instant;

fn gmres_result() -> GmresSolution<f64> {
    GmresSolution {
        x: vec![1.0],
        iterations: 1,
        restarts: 0,
        residual: 0.0,
        estimated_residual: 0.0,
        backward_error: 0.0,
        attainable_accuracy: AccuracyStatus::Attained,
        converged: true,
        status: SolverStatus::Converged,
        op_counts: KrylovOpCounts::default(),
    }
}

fn check_result<T: SolverResultContract>(
    name: &'static str,
    result: &T,
    expected_status: SolverStatus,
    expected_accuracy: AccuracyStatus,
) -> Result<Vec<(&'static str, f64)>, String> {
    result
        .validate_contract()
        .map_err(|error| format!("{name} validation failed: {error}"))?;
    if result.solver_status() != expected_status {
        return Err(format!(
            "{name} status mismatch: {:?}",
            result.solver_status()
        ));
    }
    if result.accuracy_status() != expected_accuracy {
        return Err(format!(
            "{name} accuracy mismatch: {:?}",
            result.accuracy_status()
        ));
    }
    Ok(vec![
        (name, 1.0),
        (
            match name {
                "differentiable" => "differentiable_status",
                "anderson" => "anderson_status",
                "newton" => "newton_status",
                "optimization" => "optimization_status",
                "newmark_adjoint" => "newmark_adjoint_status",
                "ir" => "ir_status",
                "sparse_refinement" => "sparse_refinement_status",
                _ => "unknown_status",
            },
            1.0,
        ),
    ])
}

fn check_failure(
    name: &'static str,
    error: &impl SolverFailure,
    expected_kind: SolverErrorKind,
) -> Result<(&'static str, f64), String> {
    if error.error_kind() != expected_kind {
        return Err(format!(
            "{name} failure kind mismatch: expected {expected_kind:?}, got {:?}",
            error.error_kind()
        ));
    }
    Ok((name, 1.0))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let mut metrics = Vec::new();
    metrics.push(check_failure(
        "core_dimension_failure_kind",
        &SolverError::DimensionMismatch {
            expected: 2,
            got: 1,
        },
        SolverErrorKind::DimensionMismatch,
    )?);
    metrics.push(check_failure(
        "direct_nested_breakdown_kind",
        &DenseDirectError::Lu(LuError::SingularMatrix),
        SolverErrorKind::Breakdown,
    )?);
    metrics.push(check_failure(
        "nonlinear_dimension_failure_kind",
        &AndersonError::DimensionMismatch {
            expected: 2,
            got: 1,
            stage: "QA fixture",
        },
        SolverErrorKind::DimensionMismatch,
    )?);
    metrics.push(check_failure(
        "optimization_convergence_failure_kind",
        &OptimError::NotConverged {
            iterations: 1,
            grad_norm: 1.0,
        },
        SolverErrorKind::Convergence,
    )?);
    metrics.push(check_failure(
        "time_integration_operation_failure_kind",
        &RevolveError::NoCheckpoint { num_steps: 1 },
        SolverErrorKind::OperationFailed,
    )?);
    metrics.push(check_failure(
        "sparse_dimension_failure_kind",
        &NumaSpmvError::DimensionMismatch {
            expected: 2,
            got: 1,
        },
        SolverErrorKind::DimensionMismatch,
    )?);
    metrics.push(check_failure(
        "preconditioner_operation_failure_kind",
        &ChebyshevError::InvalidCsrStructure,
        SolverErrorKind::OperationFailed,
    )?);
    metrics.push(check_failure(
        "facade_convergence_failure_kind",
        &LinearSolveError::NotConverged {
            iterations: 1,
            residual: 1.0,
        },
        SolverErrorKind::Convergence,
    )?);
    metrics.extend(check_result(
        "differentiable",
        &DifferentiableSolveResult {
            forward: gmres_result(),
            adjoint: Some(gmres_result()),
        },
        SolverStatus::Converged,
        AccuracyStatus::Attained,
    )?);
    metrics.extend(check_result(
        "anderson",
        &AndersonSolution {
            x: vec![1.0_f64],
            iterations: 1,
            residual: 0.0,
            converged: true,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::Attained,
        },
        SolverStatus::Converged,
        AccuracyStatus::Attained,
    )?);
    metrics.extend(check_result(
        "newton",
        &NewtonSolution {
            x: vec![1.0_f64],
            iterations: 1,
            residual: 0.0,
            converged: true,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::Attained,
            residual_evaluations: 2,
            jacobian_evaluations: 1,
            inner_iterations: 0,
        },
        SolverStatus::Converged,
        AccuracyStatus::Attained,
    )?);
    metrics.extend(check_result(
        "optimization",
        &OptimResult {
            x: vec![0.0_f64],
            fun: 0.0,
            grad: Some(vec![0.0]),
            iterations: 1,
            fun_evals: 1,
            grad_evals: 1,
            hess_evals: 0,
            grad_norm: 0.0,
            converged: true,
            status: OptimStatus::FunctionTolerance,
            kkt: None,
        },
        SolverStatus::Converged,
        AccuracyStatus::Attained,
    )?);
    metrics.extend(check_result(
        "newmark_adjoint",
        &NewmarkAdjointResult {
            force_gradient: vec![vec![0.0_f64]],
            initial_displacement_gradient: vec![0.0],
            initial_velocity_gradient: vec![0.0],
            initial_acceleration_gradient: vec![0.0],
        },
        SolverStatus::Converged,
        AccuracyStatus::NotChecked,
    )?);
    metrics.extend(check_result(
        "ir",
        &IrResult {
            x: vec![1.0_f64],
            iterations: 1,
            final_residual: 1.0e-12,
            converged: true,
        },
        SolverStatus::Converged,
        AccuracyStatus::Attained,
    )?);
    metrics.extend(check_result(
        "sparse_refinement",
        &SparseRefinementResult {
            x: vec![1.0_f64],
            iterations: 1,
            initial_relative_residual: 1.0,
            relative_residual: 1.0e-12,
            tolerance: 1.0e-10,
            converged: true,
            stop_reason: SparseRefinementStopReason::ToleranceReached,
            provenance: SparseRefinementProvenance::ReusedValueOnlyPattern {
                structure_fingerprint: 7,
            },
        },
        SolverStatus::Converged,
        AccuracyStatus::Attained,
    )?);
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("checked cross-family solver result and failure-taxonomy contract")
                .reference_checksum("analytic/deterministic-fixture:extended-result-contract-v2")
                .input_checksum("analytic/deterministic-fixture:extended-result-contract-v2")
                .tolerance("all extended result-contract metrics equal one")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error).to_json_line().unwrap()
            );
            eprintln!("EXTENDED RESULT CONTRACT FAILED");
            std::process::exit(1);
        }
    }
}

const CASE_ID: &str = "solvers.extended-result-contract.v1";
