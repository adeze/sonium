//! Compile-time QA contract for PC-fan measured-data correlation.
//!
//! The fixture is deterministic and synthetic. It verifies provenance,
//! topology, RMSE, and fail-closed correlation budgets; it is not a claim of
//! measured-product validation.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCorrelationDataset, PcFanCorrelationObservation, PcFanMeasuredCorrelationConfig,
    evaluate_pc_fan_measured_correlation,
};
use std::time::Instant;

const CASE_ID: &str = "pcfan.measured-correlation-contract.v1";

fn dataset(provenance: &str, predicted: bool) -> PcFanCorrelationDataset {
    PcFanCorrelationDataset {
        provenance: provenance.into(),
        observations: [600.0, 900.0, 1200.0]
            .into_iter()
            .enumerate()
            .map(|(index, rpm)| PcFanCorrelationObservation {
                rpm,
                airflow_cfm: 20.0 + index as f64 * 5.0 + if predicted { 0.25 } else { 0.0 },
                pressure_rise_pa: 100.0 + index as f64 * 20.0 + if predicted { 1.0 } else { 0.0 },
                acoustic_level_db: Some(40.0 + index as f64 + if predicted { 0.2 } else { 0.0 }),
            })
            .collect(),
    }
}

fn run() -> Result<(f64, f64, f64), String> {
    let report = evaluate_pc_fan_measured_correlation(
        &PcFanMeasuredCorrelationConfig::default(),
        &dataset("synthetic-measured-fixture:pcfan-v1", false),
        &dataset("solver-prediction-fixture:pcfan-v1", true),
    )
    .map_err(|error| error.to_string())?;
    if !report.is_valid() {
        return Err("synthetic correlation fixture did not pass".into());
    }
    Ok((
        report.sample_count as f64,
        report.airflow_rmse_cfm,
        report.acoustic_rmse_db.unwrap_or(f64::NAN),
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((samples, airflow_rmse, acoustic_rmse)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("sample_count", samples)
                .metric("airflow_rmse_cfm", airflow_rmse)
                .metric("acoustic_rmse_db", acoustic_rmse)
                .minimum("sample_count", 3.0)
                .maximum("airflow_rmse_cfm", 2.0)
                .maximum("acoustic_rmse_db", 3.0)
                .reference_checksum("synthetic/deterministic-fixture:pcfan-measured-correlation-v1")
                .input_checksum("synthetic-measured-and-solver-observations:pcfan-v1")
                .seed("pcfan-measured-correlation-contract:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            record
                .to_json_line()
                .expect("serialize PC-fan correlation QA result");
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error);
            record
                .to_json_line()
                .expect("serialize PC-fan correlation QA failure");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
