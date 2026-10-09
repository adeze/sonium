//! CPU scalar-family coverage for the public iterative result contract.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, Zero};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{
    AccuracyStatus, BlockGmresSolution, ComplexField, GmresSolution, KrylovOpCounts,
    SolverResultContract, SolverStatus,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-scalar-contract.v1";

fn check_valid_result<T: ComplexField>() -> Result<(), String> {
    let scalar = GmresSolution {
        x: vec![T::one()],
        iterations: 1,
        restarts: 0,
        residual: T::Real::zero(),
        estimated_residual: T::Real::zero(),
        backward_error: T::Real::zero(),
        attainable_accuracy: AccuracyStatus::Attained,
        converged: true,
        status: SolverStatus::Converged,
        op_counts: KrylovOpCounts::default(),
    };
    if scalar.solver_status() != SolverStatus::Converged
        || scalar.accuracy_status() != AccuracyStatus::Attained
    {
        return Err("scalar status/accuracy mapping failed".into());
    }
    scalar
        .validate_contract()
        .map_err(|error| format!("scalar validation failed: {error}"))?;

    let block = BlockGmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![T::one()]),
        iterations: 1,
        residuals: vec![T::Real::zero()],
        true_residuals: vec![T::Real::zero()],
        backward_errors: vec![T::Real::zero()],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: true,
        op_counts: KrylovOpCounts::default(),
    };
    if block.solver_status() != SolverStatus::Converged
        || block.accuracy_status() != AccuracyStatus::Attained
    {
        return Err("block status/accuracy mapping failed".into());
    }
    block
        .validate_contract()
        .map_err(|error| format!("block validation failed: {error}"))
}

fn check_numerical_error<T: ComplexField>() -> Result<(), String> {
    let result = GmresSolution {
        x: vec![T::zero()],
        iterations: 0,
        restarts: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
    };
    if result.solver_status() != SolverStatus::NumericalError
        || result.accuracy_status() != AccuracyStatus::NotChecked
    {
        return Err("numerical-error status/accuracy mapping failed".into());
    }
    result
        .validate_contract()
        .map_err(|error| format!("numerical-error validation failed: {error}"))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    check_valid_result::<f32>()?;
    check_valid_result::<f64>()?;
    check_valid_result::<Complex32>()?;
    check_valid_result::<Complex64>()?;
    check_numerical_error::<f32>()?;
    check_numerical_error::<f64>()?;
    check_numerical_error::<Complex32>()?;
    check_numerical_error::<Complex64>()?;

    Ok(vec![
        ("f32_valid_contract", 1.0),
        ("f64_valid_contract", 1.0),
        ("complex32_valid_contract", 1.0),
        ("complex64_valid_contract", 1.0),
        ("f32_numerical_error_contract", 1.0),
        ("f64_numerical_error_contract", 1.0),
        ("complex32_numerical_error_contract", 1.0),
        ("complex64_numerical_error_contract", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent scalar-family iterative result-contract fixtures")
                .reference_checksum("analytic/deterministic-fixture:iterative-scalar-contract-v1")
                .input_checksum("analytic/deterministic-fixture:iterative-scalar-contract-v1")
                .dimension("scalar and block GMRES result objects over f32/f64/Complex32/Complex64")
                .tolerance(
                    "all scalar-family status, accuracy, and numerical-error metrics equal one",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("ITERATIVE SCALAR RESULT CONTRACT FAILED");
            std::process::exit(1);
        }
    }
}
