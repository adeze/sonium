//! Release-backed CPU check for the PC-fan reduced-order operating envelope.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{PcFanConfig, PcFanOperatingPoint, evaluate_pc_fan};

const CASE_ID: &str = "pcfan.operating-envelope-reference.v1";
const RPMS: [f64; 4] = [400.0, 600.0, 900.0, 1200.0];

fn run() -> Result<[f64; 6], String> {
    let mut airflow = [0.0; 4];
    let mut pressure = [0.0; 4];
    let mut acoustic = [0.0; 4];
    let mut max_temperature_c = f64::NEG_INFINITY;
    let mut min_stress_margin_pa = f64::INFINITY;
    let mut min_modal_margin_hz = f64::INFINITY;
    for (index, rpm) in RPMS.into_iter().enumerate() {
        let mut config = PcFanConfig::bench_120mm();
        config.operating_point = PcFanOperatingPoint::bench(rpm);
        // This binary validates the physical RPM trend, not target-flow
        // optimization. Use a broad acceptance band so a baseline prior
        // mismatch cannot mask the envelope evidence.
        config.weights.airflow_band_fraction = 1.0;
        config.operating_point_spread.rpms.clear();
        config.weights.operating_spread_weight = 0.0;
        let evaluation = evaluate_pc_fan(&config, &config.default_design_variables());
        if !evaluation.is_feasible(&config) {
            return Err(format!("baseline is infeasible at {rpm} RPM"));
        }
        for value in [
            evaluation.airflow_cfm,
            evaluation.flow.pressure_rise_pa,
            evaluation.acoustics.overall_spl_db,
            evaluation.thermal.winding_temperature_c,
            evaluation.structural.maximum_stress_pa,
            evaluation.structural.first_mode_hz,
        ] {
            if !value.is_finite() {
                return Err(format!("non-finite operating-envelope value at {rpm} RPM"));
            }
        }
        airflow[index] = evaluation.airflow_cfm;
        pressure[index] = evaluation.flow.pressure_rise_pa;
        acoustic[index] = evaluation.acoustics.overall_spl_db;
        max_temperature_c = max_temperature_c.max(evaluation.thermal.winding_temperature_c);
        min_stress_margin_pa = min_stress_margin_pa.min(evaluation.structural.stress_margin_pa);
        min_modal_margin_hz = min_modal_margin_hz.min(evaluation.structural.modal_margin_hz);
    }
    for pair in airflow.windows(2) {
        if pair[1] <= pair[0] {
            return Err("airflow is not strictly increasing over RPM envelope".into());
        }
    }
    for pair in pressure.windows(2) {
        if pair[1] <= pair[0] {
            return Err("pressure rise is not strictly increasing over RPM envelope".into());
        }
    }
    Ok([
        airflow[3] / airflow[0],
        pressure[3] / pressure[0],
        acoustic[3] - acoustic[0],
        max_temperature_c,
        min_stress_margin_pa,
        min_modal_margin_hz,
    ])
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("airflow_ratio_1200_to_400", metrics[0])
                .metric("pressure_ratio_1200_to_400", metrics[1])
                .metric("acoustic_delta_db", metrics[2])
                .metric("maximum_winding_temperature_c", metrics[3])
                .metric("minimum_stress_margin_pa", metrics[4])
                .metric("minimum_modal_margin_hz", metrics[5])
                .minimum("airflow_ratio_1200_to_400", 1.0)
                .minimum("pressure_ratio_1200_to_400", 1.0)
                .minimum("minimum_stress_margin_pa", 0.0)
                .minimum("minimum_modal_margin_hz", 0.0)
                .reference_checksum("analytic/invariant:pcfan-operating-envelope-v1")
                .input_checksum("pcfan-120mm:bench:reduced-order:rpm=400,600,900,1200")
                .formulation("production PC-fan reduced-order blade-element operating envelope")
                .tolerance("strict airflow/pressure trends; finite thermal and structural margins")
                .seed("pcfan-operating-envelope-reference:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serializes")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("QA failure serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
