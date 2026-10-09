//! Compile-time contract for the PC-fan prediction-to-correlation pipeline.
//!
//! The measured side remains a deterministic fixture. This case checks that
//! solver predictions carry provenance and can enter the existing correlation
//! admission path; it is not measured-product validation.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanConfig, PcFanMeasuredCorrelationConfig, PcFanMeshResolution,
    evaluate_pc_fan_measured_correlation, predict_pc_fan_correlation_dataset,
};

const CASE_ID: &str = "pcfan.predicted-correlation-pipeline-reference.v1";

fn run() -> Result<[f64; 3], String> {
    let mut config = PcFanConfig::default_120mm();
    config.mesh = PcFanMeshResolution {
        radial_segments: 3,
        chord_segments: 2,
        circumferential_segments: 8,
        harmonic_count: 2,
    };
    let design = config.default_design_variables();
    let predicted = predict_pc_fan_correlation_dataset(
        &config,
        &design,
        &[600.0, 900.0, 1200.0],
        "solver:pcfan:reduced-order:compile-contract",
    )
    .map_err(|error| error.to_string())?;
    let mut measured = predicted.clone();
    measured.provenance = "measurement:synthetic:pcfan-compile-contract".into();
    for observation in &mut measured.observations {
        observation.airflow_cfm += 0.1;
        observation.pressure_rise_pa += 1.0;
        observation.acoustic_level_db = observation.acoustic_level_db.map(|value| value + 0.1);
    }
    let report = evaluate_pc_fan_measured_correlation(
        &PcFanMeasuredCorrelationConfig::default(),
        &measured,
        &predicted,
    )
    .map_err(|error| error.to_string())?;
    if !report.is_valid() {
        return Err("prediction-to-correlation pipeline did not pass".into());
    }
    Ok([
        report.sample_count as f64,
        report.airflow_rmse_cfm,
        report.pressure_rmse_pa,
    ])
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .metric("sample_count", metrics[0])
                .metric("airflow_rmse_cfm", metrics[1])
                .metric("pressure_rmse_pa", metrics[2])
                .reference_checksum(
                    "synthetic/deterministic-fixture:pcfan-predicted-correlation-pipeline-v1",
                )
                .input_checksum("pcfan-120mm:reduced-order:rpm=600,900,1200")
                .seed("pcfan-predicted-correlation-pipeline-reference:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .to_json_line()
                    .expect("QA failure serialization")
            );
            std::process::exit(1);
        }
    }
}
