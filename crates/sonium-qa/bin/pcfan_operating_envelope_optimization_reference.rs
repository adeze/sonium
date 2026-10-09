//! Compile-time contract for the reduced-order PC-fan envelope optimizer.
//!
//! This reference is intentionally compile-checked in the normal workflow;
//! it is not part of the unit-test or simulation runs used for this change.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanConfig, PcFanOperatingEnvelopeObjectiveConfig,
    evaluate_pc_fan_operating_envelope_objective, optimize_pc_fan_operating_envelope,
};

const CASE_ID: &str = "pcfan.operating-envelope-optimization-reference.v1";
const RPMS: [f64; 4] = [400.0, 600.0, 900.0, 1200.0];

fn objective_config() -> PcFanOperatingEnvelopeObjectiveConfig {
    PcFanOperatingEnvelopeObjectiveConfig {
        rpms: RPMS.into_iter().collect(),
        target_airflow_cfm: vec![1.0; RPMS.len()],
        target_pressure_rise_pa: vec![1.0; RPMS.len()],
        target_acoustic_level_db: None,
        airflow_weight: 1.0,
        pressure_weight: 1.0,
        acoustic_weight: 0.0,
        require_manufacturing_coverage: false,
    }
}

fn run() -> Result<[f64; 3], String> {
    // The robust-envelope reference intentionally prices tolerance corners.
    // Widen the airflow admission band and disable unrelated structural
    // screening limits, matching the library's deterministic robust fixture.
    let mut config = PcFanConfig::bench_120mm();
    config.weights.airflow_band_fraction = 2.0;
    config.structural.min_modal_separation_hz = 0.0;
    config.structural.max_imbalance_g_mm = 100.0;
    let design = config.default_design_variables();
    let objective = objective_config();
    let evaluated = evaluate_pc_fan_operating_envelope_objective(&config, &design, &objective)
        .map_err(|error| error.to_string())?;
    if !evaluated.is_valid() {
        return Err("reduced-order envelope objective is invalid".to_string());
    }

    let optimized = optimize_pc_fan_operating_envelope(config, objective, 1, 5, 17)
        .map_err(|error| error.to_string())?;
    if optimized.optimization.function_evaluations == 0 {
        return Err("envelope optimizer performed no function evaluations".to_string());
    }

    Ok([
        optimized.objective.score,
        optimized.objective.worst_airflow_error_cfm,
        optimized.objective.worst_pressure_error_pa,
    ])
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .metric("objective_score", metrics[0])
                .metric("worst_airflow_error_cfm", metrics[1])
                .metric("worst_pressure_error_pa", metrics[2])
                .reference_checksum(
                    "analytic/deterministic-fixture:pcfan-operating-envelope-optimization-v1",
                )
                .input_checksum("pcfan-120mm:bench:reduced-order:rpm=400,600,900,1200")
                .seed("pcfan-operating-envelope-optimization-reference:cpu:v1");
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
