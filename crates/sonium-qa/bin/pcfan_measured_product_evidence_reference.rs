//! Compile-only contract for independent product-measurement ingestion.
//!
//! The record is a deterministic schema fixture. It proves metadata/schema
//! validation and adapter wiring only; it is not a measured product record.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanMeasuredProductEvidence, parse_pc_fan_measured_product_csv,
    pc_fan_measured_product_csv_sha256,
};

const CASE_ID: &str = "pcfan.measured-product-evidence-reference.v1";
const CSV: &str = "rpm,airflow_cfm,pressure_drop_pa,acoustic_level_db,winding_temperature_c,maximum_stress_pa,thermal_expansion_m,fatigue_alternating_stress_pa\n600,42.0,18.0,31.0,35.0,1000000,0.00001,400000\n900,48.0,25.0,36.0,40.0,1100000,0.00002,450000\n1200,52.0,31.0,42.0,46.0,1200000,0.00003,500000\n";

fn run() -> Result<String, String> {
    let evidence = PcFanMeasuredProductEvidence {
        product_id: "fixture-fan-120mm".into(),
        serial_number: "fixture-SN-001".into(),
        instrument_id: "fixture-daq-001".into(),
        calibration_certificate: "fixture-cal-001".into(),
        acquisition_utc: "2026-08-24T12:00:00Z".into(),
        source_uri: "fixture://pcfan/measured-product-v1.csv".into(),
        raw_data_sha256: pc_fan_measured_product_csv_sha256(CSV),
    };
    let dataset =
        parse_pc_fan_measured_product_csv(CSV, evidence).map_err(|error| error.to_string())?;
    let correlation = dataset
        .to_correlation_dataset(true)
        .map_err(|error| error.to_string())?;
    let structural = dataset
        .to_structural_calibration_dataset()
        .map_err(|error| error.to_string())?;
    if correlation.observations.len() != 3 || structural.observations.len() != 3 {
        return Err("measured-product fixture adapter lost observations".into());
    }
    Ok("synthetic measured-product schema fixture passed; no product evidence claimed".into())
}

fn main() {
    match run() {
        Ok(message) => {
            let mut result = QaResult::passed(CASE_ID);
            result.message = Some(message);
            println!(
                "{}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "{}",
                QaResult::failed(CASE_ID, error)
                    .to_json_line()
                    .expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
