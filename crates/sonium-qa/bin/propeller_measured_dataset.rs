//! Compile-only release-contract wrapper for propeller measured-data provenance.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    MeasuredCalibrationDataset, MeasuredCalibrationDatasetMetadata, MeasuredCalibrationOptions,
    MeasuredCalibrationSample,
};

const CASE_ID: &str = "propeller.measured-dataset-provenance.v1";

fn run() -> Result<(f64, f64), String> {
    let samples = vec![
        MeasuredCalibrationSample {
            operating_point: 0.2,
            predicted_value: 1.0,
            measured_value: 1.2,
        },
        MeasuredCalibrationSample {
            operating_point: 0.4,
            predicted_value: 2.0,
            measured_value: 2.4,
        },
        MeasuredCalibrationSample {
            operating_point: 0.6,
            predicted_value: 3.0,
            measured_value: 3.6,
        },
    ];
    let metadata = MeasuredCalibrationDatasetMetadata {
        dataset_id: "public-fixture-v1".into(),
        source_uri: "https://example.invalid/propeller/public-fixture-v1".into(),
        license: "CC-BY-4.0".into(),
        instrument_id: "rig-fixture-01".into(),
        calibration_reference: "certificate-fixture-v1".into(),
        channel: "thrust".into(),
        unit: "N".into(),
        content_sha256: String::new(),
    };
    let dataset = MeasuredCalibrationDataset::from_parts(samples, metadata)
        .map_err(|error| error.to_string())?;
    if dataset.metadata.content_sha256.len() != 64 {
        return Err("dataset digest is not canonical SHA-256".into());
    }
    let restored = MeasuredCalibrationDataset::from_json(
        &dataset.to_json().map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if restored != dataset {
        return Err("dataset JSON round trip changed provenance or samples".into());
    }
    let calibration = restored
        .calibrate(MeasuredCalibrationOptions::default())
        .map_err(|error| error.to_string())?;
    if !calibration.is_finite() || !calibration.passed {
        return Err("bound measured dataset calibration did not pass".into());
    }
    Ok((calibration.gain, calibration.offset))
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
                .metric("sample_count", 3.0)
                .metric("digest_hex_length", 64.0)
                .input_checksum("analytic/deterministic-fixture:propeller-measured-dataset-v1")
                .reference_checksum("canonical/sha256:propeller-measured-dataset-v1")
                .reference("caller-supplied dataset envelope; content integrity only, no instrument authentication")
                .dimension("three scalar samples with source, license, instrument, and calibration metadata")
                .formulation("versioned JSON provenance envelope with canonical SHA-256 binding")
                .tolerance("digest round trip exact; finite calibration and metadata admission")
                .seed("propeller-measured-dataset-v1:cpu")
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
