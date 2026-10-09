//! Compile-only QA wrapper for measured-calibration uncertainty propagation.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    MeasuredUncertaintyOptions, MeasuredUncertaintySample, evaluate_measured_uncertainty,
};

const CASE_ID: &str = "propeller.measured-uncertainty-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let samples = vec![
        MeasuredUncertaintySample {
            operating_point: 0.1,
            predicted_value: 1.0,
            measured_value: 2.0,
            predicted_standard_deviation: 0.1,
            measured_standard_deviation: 0.2,
        },
        MeasuredUncertaintySample {
            operating_point: 0.2,
            predicted_value: 2.0,
            measured_value: 4.0,
            predicted_standard_deviation: 0.1,
            measured_standard_deviation: 0.2,
        },
        MeasuredUncertaintySample {
            operating_point: 0.3,
            predicted_value: 3.0,
            measured_value: 6.0,
            predicted_standard_deviation: 0.1,
            measured_standard_deviation: 0.2,
        },
    ];
    let result = evaluate_measured_uncertainty(&samples, MeasuredUncertaintyOptions::default())
        .map_err(|error| error.to_string())?;
    if !result.is_finite() || !result.passed || result.points.len() != samples.len() {
        return Err("uncertainty envelope is not finite, complete, or admitted".into());
    }
    Ok((
        result.calibration.gain,
        result.maximum_z_score,
        result.coverage_fraction,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((gain, maximum_z_score, coverage_fraction)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("calibration_gain", gain)
                .metric("maximum_z_score", maximum_z_score)
                .metric("coverage_fraction", coverage_fraction)
                .input_checksum("analytic/deterministic-fixture:propeller-measured-uncertainty-v1")
                .reference("independent first-order Gaussian uncertainty propagation")
                .reference_checksum("analytic/invariant:propeller-measured-uncertainty-v1")
                .dimension("three scalar calibration samples with model and measurement sigma")
                .formulation("affine calibration plus combined uncertainty/z-score admission")
                .tolerance("finite envelope; maximum z-score <= 3; coverage >= 0.95")
                .seed("propeller-measured-uncertainty-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
