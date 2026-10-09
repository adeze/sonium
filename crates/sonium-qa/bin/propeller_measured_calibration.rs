//! Compile-only release-contract wrapper for propeller measured calibration.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    MeasuredCalibrationOptions, MeasuredCalibrationSample, calibrate_measured_response,
};

const CASE_ID: &str = "propeller.measured-calibration-reference.v1";

fn run() -> Result<(f64, f64), String> {
    let samples = [
        MeasuredCalibrationSample {
            operating_point: 0.2,
            predicted_value: 1.0,
            measured_value: 2.5,
        },
        MeasuredCalibrationSample {
            operating_point: 0.4,
            predicted_value: 2.0,
            measured_value: 4.5,
        },
        MeasuredCalibrationSample {
            operating_point: 0.6,
            predicted_value: 3.0,
            measured_value: 6.5,
        },
        MeasuredCalibrationSample {
            operating_point: 0.8,
            predicted_value: 4.0,
            measured_value: 8.5,
        },
    ];
    let result = calibrate_measured_response(&samples, MeasuredCalibrationOptions::default())
        .map_err(|error| error.to_string())?;
    if !result.is_finite() || !result.passed {
        return Err("deterministic measured calibration fixture did not pass".into());
    }
    if (result.gain - 2.0).abs() > 1.0e-12 || (result.offset - 0.5).abs() > 1.0e-12 {
        return Err(format!(
            "unexpected affine coefficients gain={} offset={}",
            result.gain, result.offset
        ));
    }
    Ok((result.gain, result.offset))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((gain, offset)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("gain", gain)
                .metric("offset", offset)
                .metric("sample_count", 4.0)
                .input_checksum("analytic/deterministic-fixture:propeller-measured-calibration-v1")
                .reference_checksum("analytic/affine-fit:propeller-measured-calibration-v1")
                .reference(
                    "caller-supplied affine calibration contract; no bundled measured dataset",
                )
                .dimension("four scalar operating-point samples")
                .formulation("bounded least-squares affine measured-response calibration")
                .tolerance("coefficient recovery <= 1e-12; finite residual and admission gates")
                .seed("propeller-measured-calibration-v1:cpu")
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
