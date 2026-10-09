//! Independent CPU QA for checked pseudo-arclength eigen continuation.

use sonium_qa::QaResult;
use sonium_solvers::{
    CsrMatrix, EigenContinuationConfig, EigenContinuationEvent, EigenContinuationPoint,
    continue_generalized_eigenpair_pseudo_arclength,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.eigen-pseudo-arclength.v1";
const TOLERANCE: f64 = 1.0e-12;

fn diagonal(first: f64, second: f64) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(2, 2, vec![(0, 0, first), (1, 1, second)])
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    // A branch that folds in the continuation parameter while its eigenvalue
    // keeps increasing: (-1, 1) -> (0, 2) -> (-0.5, 3).
    let previous = EigenContinuationPoint {
        parameter: -1.0,
        eigenvalue: 1.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    };
    let current = EigenContinuationPoint {
        parameter: 0.0,
        eigenvalue: 2.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    };
    let mass = diagonal(2.0, 3.0);
    let config = EigenContinuationConfig {
        min_step_size: 0.1,
        max_step_size: 2.0,
        shrink_factor: 0.5,
        max_retries: 2,
    };
    let step = continue_generalized_eigenpair_pseudo_arclength(
        &previous,
        &current,
        2.0_f64.sqrt(),
        &mass,
        &config,
        |prediction| {
            Ok(EigenContinuationPoint {
                parameter: -0.5,
                eigenvalue: prediction.eigenvalue,
                right_eigenvector: prediction.right_eigenvector.clone(),
                left_eigenvector: vec![1.0, 0.0],
            })
        },
    )
    .map_err(|error| format!("pseudo-arclength continuation failed: {error:?}"))?;
    let prediction_parameter_error = (step.prediction.parameter - 1.0).abs();
    let prediction_eigenvalue_error = (step.prediction.eigenvalue - 3.0).abs();
    let tangent_parameter_error = (step.prediction.tangent_parameter - 2.0_f64.sqrt() / 2.0).abs();
    let tangent_eigenvalue_error =
        (step.prediction.tangent_eigenvalue - 2.0_f64.sqrt() / 2.0).abs();
    let event_error = if step.event == EigenContinuationEvent::TurningPoint {
        0.0
    } else {
        1.0
    };
    let metrics = [
        ("prediction_parameter_error", prediction_parameter_error),
        ("prediction_eigenvalue_error", prediction_eigenvalue_error),
        ("tangent_parameter_error", tangent_parameter_error),
        ("tangent_eigenvalue_error", tangent_eigenvalue_error),
        ("turning_point_event_error", event_error),
    ];
    if metrics.iter().any(|(_, value)| *value > TOLERANCE) {
        return Err(format!("pseudo-arclength analytic mismatch: {metrics:?}"));
    }
    Ok(metrics.to_vec())
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent analytic fold in a generalized eigenpair branch")
                .input_checksum("analytic/deterministic-fixture:eigen-pseudo-arclength-v1")
                .reference_checksum("analytic/deterministic-fixture:eigen-pseudo-arclength-v1")
                .formulation("secant pseudo-arclength predictor with turning-point classification")
                .dimension("2x2 real generalized eigenpair; one fold")
                .tolerance("all predictor/tangent/event errors <= 1e-12");
            for (name, value) in metrics {
                result = result.metric(name, value).maximum(name, TOLERANCE);
            }
            result = result
                .seed("solvers-eigen-pseudo-arclength:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA JSON must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA JSON must serialize")
            );
            eprintln!("FAILED");
            std::process::exit(1);
        }
    }
}
