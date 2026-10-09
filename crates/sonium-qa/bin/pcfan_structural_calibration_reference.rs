//! Compile-only contract for the PC-fan structural-calibration admission path.
//!
//! The measured side is a deterministic fixture derived from the reduced-order
//! prediction. It checks that the public comparison contract is wired into the
//! QA crate; it is not product-scale measured evidence.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanConfig, PcFanStructuralCalibrationConfig, PcFanStructuralCalibrationDataset,
    PcFanStructuralCalibrationObservation, evaluate_pc_fan_structural_calibration,
    predict_pc_fan_structural_calibration_dataset,
};

const CASE_ID: &str = "pcfan.structural-calibration-reference.v1";

fn run() -> Result<String, String> {
    let config = PcFanConfig::bench_120mm();
    let design = config.default_design_variables();
    let predicted = predict_pc_fan_structural_calibration_dataset(
        &config,
        &design,
        &[600.0, 900.0, 1200.0],
        "solver:pcfan:reduced-order:compile-contract",
    )
    .map_err(|error| error.to_string())?;

    // This deterministic fixture exercises the contract only. It must not be
    // described or reported as an independent product measurement.
    let measured = PcFanStructuralCalibrationDataset {
        provenance: "fixture:pcfan:synthetic-structural-calibration:v1".into(),
        observations: predicted
            .observations
            .iter()
            .map(|observation| PcFanStructuralCalibrationObservation {
                rpm: observation.rpm,
                winding_temperature_c: observation.winding_temperature_c + 0.2,
                maximum_stress_pa: observation.maximum_stress_pa * 1.01,
                thermal_expansion_m: observation.thermal_expansion_m * 1.01,
                fatigue_alternating_stress_pa: observation.fatigue_alternating_stress_pa * 1.01,
            })
            .collect(),
    };

    let report = evaluate_pc_fan_structural_calibration(
        &PcFanStructuralCalibrationConfig::default(),
        &measured,
        &predicted,
    )
    .map_err(|error| error.to_string())?;
    if !report.passed {
        return Err("deterministic structural-calibration fixture was rejected".into());
    }

    Ok(format!(
        "synthetic fixture admission passed for {} RPM points; not product validation",
        report.sample_count
    ))
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
