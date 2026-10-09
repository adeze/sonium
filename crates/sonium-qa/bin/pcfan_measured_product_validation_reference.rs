//! Compile-only measured-product prediction-validation contract.
//!
//! The fixture is deliberately deterministic and self-contained: it checks
//! the independent comparison API and provenance gate, but it is not external
//! product evidence and does not run a PC-fan simulation.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanMeasuredProductDataset, PcFanMeasuredProductEvidence, PcFanMeasuredProductObservation,
    PcFanMeasuredProductValidationConfig, validate_pc_fan_measured_product_predictions,
};

const CASE_ID: &str = "pcfan.measured-product-validation-reference.v1";

fn dataset() -> PcFanMeasuredProductDataset {
    PcFanMeasuredProductDataset {
        evidence: PcFanMeasuredProductEvidence {
            product_id: "fixture-fan-120mm".into(),
            serial_number: "fixture-SN-001".into(),
            instrument_id: "fixture-daq-001".into(),
            calibration_certificate: "fixture-cal-001".into(),
            acquisition_utc: "2026-08-24T12:00:00Z".into(),
            source_uri: "fixture://pcfan/measured-product-v1.csv".into(),
            raw_data_sha256: "0".repeat(64),
        },
        observations: vec![
            observation(600.0, 42.0, 18.0, 31.0),
            observation(900.0, 48.0, 25.0, 36.0),
            observation(1200.0, 52.0, 31.0, 42.0),
        ],
    }
}

fn observation(
    rpm: f64,
    airflow_cfm: f64,
    pressure_drop_pa: f64,
    acoustic_level_db: f64,
) -> PcFanMeasuredProductObservation {
    PcFanMeasuredProductObservation {
        rpm,
        airflow_cfm,
        pressure_drop_pa,
        acoustic_level_db: Some(acoustic_level_db),
        winding_temperature_c: 37.0,
        maximum_stress_pa: 1.1e6,
        thermal_expansion_m: 2.0e-5,
        fatigue_alternating_stress_pa: 450_000.0,
    }
}

fn run() -> Result<(usize, f64), String> {
    let measured = dataset();
    let predictions = measured.observations.clone();
    let report = validate_pc_fan_measured_product_predictions(
        &measured,
        &predictions,
        "resolved-cfd:cpu:fixture-v1",
        &PcFanMeasuredProductValidationConfig::default(),
    )
    .map_err(|error| error.to_string())?;
    if !report.is_valid() || report.sample_count != 3 || report.airflow_rmse_cfm != 0.0 {
        return Err("independent measured-product comparison did not pass".into());
    }
    Ok((report.sample_count, report.airflow_rmse_cfm))
}

fn main() {
    match run() {
        Ok((sample_count, airflow_rmse)) => {
            let mut result = QaResult::passed(CASE_ID)
                .metric("sample_count", sample_count as f64)
                .metric("airflow_rmse_cfm", airflow_rmse)
                .reference_checksum("synthetic:measured-product-validation-reference:v1")
                .input_checksum("synthetic:independent-measured-and-predicted-fixture:v1");
            result.message = Some("independent measured-product comparison contract".into());
            let result = result.to_json_line().expect("QA result must serialize");
            println!("QA_RESULT: {result}");
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .to_json_line()
                .expect("QA failure result must serialize");
            println!("QA_RESULT: {result}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
