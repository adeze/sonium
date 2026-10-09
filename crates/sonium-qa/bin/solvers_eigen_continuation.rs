//! Independent CPU QA for the generalized-eigenpair tangent predictor.

use sonium_qa::QaResult;
use sonium_solvers::{
    CsrMatrix, EigenContinuationConfig, EigenContinuationPoint, continue_generalized_eigenpair,
    predict_generalized_eigenpair,
};
use std::cell::Cell;
use std::time::Instant;

const CASE_ID: &str = "solvers.eigen-continuation-predictor.v1";
const TOLERANCE: f64 = 1.0e-12;

fn diagonal(first: f64, second: f64) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(2, 2, vec![(0, 0, first), (1, 1, second)])
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    // A(p) = diag(6 + 2(p - 2), 8), B = diag(2, 3), so the first branch is
    // lambda(p) = 1 + p and has an exactly zero eigenvector tangent.
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let point = EigenContinuationPoint {
        parameter: 2.0,
        eigenvalue: 3.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    };
    let prediction = predict_generalized_eigenpair(
        &point,
        0.125,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |rhs| {
            if rhs.iter().any(|value| value.abs() > 1.0e-12) {
                return Err(sonium_solvers::SolverError::OperationFailed {
                    operation: "eigen continuation QA singular solve",
                    reason: format!("unexpected manufactured tangent RHS: {rhs:?}"),
                });
            }
            Ok(vec![0.0; rhs.len()])
        },
    )
    .map_err(|error| format!("eigen continuation predictor failed: {error}"))?;
    // A second analytic tangent exercises the generalized-pencil dB term:
    // with dB = diag(1, 0), dA = diag(2, 0), d lambda / dp = -1/2.
    let mass_derivative = diagonal(1.0, 0.0);
    let mass_prediction = predict_generalized_eigenpair(
        &point,
        0.125,
        &mass,
        Some(&mass_derivative),
        &operator_derivative,
        |rhs| {
            if rhs.iter().any(|value| value.abs() > 1.0e-12) {
                return Err(sonium_solvers::SolverError::OperationFailed {
                    operation: "eigen continuation QA mass singular solve",
                    reason: format!("unexpected manufactured mass tangent RHS: {rhs:?}"),
                });
            }
            Ok(vec![0.0; rhs.len()])
        },
    )
    .map_err(|error| format!("mass-derivative predictor failed: {error}"))?;

    let adaptive_attempts = Cell::new(0_usize);
    let adaptive = continue_generalized_eigenpair(
        &point,
        0.5,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| Ok(vec![0.0; 2]),
        &EigenContinuationConfig {
            min_step_size: 0.125,
            max_step_size: 0.5,
            shrink_factor: 0.5,
            max_retries: 2,
        },
        |prediction| {
            adaptive_attempts.set(adaptive_attempts.get() + 1);
            if prediction.step_size.abs() > 0.26 {
                return Err(sonium_solvers::SolverError::OperationFailed {
                    operation: "eigen continuation QA adaptive corrector",
                    reason: "synthetic large-step rejection".into(),
                });
            }
            Ok(EigenContinuationPoint {
                parameter: prediction.parameter,
                eigenvalue: 1.0 + prediction.parameter,
                right_eigenvector: vec![1.0, 0.0],
                left_eigenvector: vec![1.0, 0.0],
            })
        },
    )
    .map_err(|error| format!("adaptive eigen continuation failed: {error}"))?;

    let parameter_error = (prediction.parameter - 2.125).abs();
    let eigenvalue_error = (prediction.eigenvalue - 3.125).abs();
    let sensitivity_error = (prediction.eigenvalue_sensitivity - 1.0).abs();
    let mass_eigenvalue_error = (mass_prediction.eigenvalue - 2.9375).abs();
    let mass_sensitivity_error = (mass_prediction.eigenvalue_sensitivity + 0.5).abs();
    let adaptive_step_error = (adaptive.prediction.step_size - 0.25).abs();
    let adaptive_parameter_error = (adaptive.corrected_point.parameter - 2.25).abs();
    let adaptive_eigenvalue_error = (adaptive.corrected_point.eigenvalue - 3.25).abs();
    let vector_error = prediction
        .right_eigenvector
        .iter()
        .zip([1.0, 0.0])
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if parameter_error > TOLERANCE
        || eigenvalue_error > TOLERANCE
        || sensitivity_error > TOLERANCE
        || mass_eigenvalue_error > TOLERANCE
        || mass_sensitivity_error > TOLERANCE
        || adaptive_step_error > TOLERANCE
        || adaptive_parameter_error > TOLERANCE
        || adaptive_eigenvalue_error > TOLERANCE
        || adaptive.retries != 1
        || adaptive_attempts.get() != 2
        || vector_error > TOLERANCE
    {
        return Err(format!(
            "analytic predictor mismatch: parameter={parameter_error:.3e}, eigenvalue={eigenvalue_error:.3e}, sensitivity={sensitivity_error:.3e}, mass_eigenvalue={mass_eigenvalue_error:.3e}, mass_sensitivity={mass_sensitivity_error:.3e}, adaptive_step={adaptive_step_error:.3e}, adaptive_parameter={adaptive_parameter_error:.3e}, adaptive_eigenvalue={adaptive_eigenvalue_error:.3e}, vector={vector_error:.3e}, retries={}, attempts={}",
            adaptive.retries,
            adaptive_attempts.get()
        ));
    }

    Ok(vec![
        ("parameter_error", parameter_error),
        ("eigenvalue_error", eigenvalue_error),
        ("sensitivity_error", sensitivity_error),
        ("mass_eigenvalue_error", mass_eigenvalue_error),
        ("mass_sensitivity_error", mass_sensitivity_error),
        ("adaptive_step_error", adaptive_step_error),
        ("adaptive_parameter_error", adaptive_parameter_error),
        ("adaptive_eigenvalue_error", adaptive_eigenvalue_error),
        ("adaptive_retries", adaptive.retries as f64),
        ("adaptive_attempts", adaptive_attempts.get() as f64),
        ("vector_error", vector_error),
        ("step_size", prediction.step_size),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic affine generalized-eigenvalue branch")
                .input_checksum("analytic/deterministic-fixture:eigen-continuation-v1")
                .reference_checksum("analytic/deterministic-fixture:eigen-continuation-v1")
                .dimension("2x2 diagonal generalized pencil; scalar predictor step")
                .tolerance("analytic parameter/eigenvalue/sensitivity/vector errors <= 1e-12")
                .seed("solvers-eigen-continuation:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.ends_with("error") {
                    result = result.maximum(name, TOLERANCE);
                }
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic affine generalized-eigenvalue branch")
                .input_checksum("analytic/deterministic-fixture:eigen-continuation-v1")
                .reference_checksum("analytic/deterministic-fixture:eigen-continuation-v1")
                .tolerance("analytic parameter/eigenvalue/sensitivity/vector errors <= 1e-12")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
