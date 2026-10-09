//! Compile-time contract for the reduced-order full-state PC-fan adjoint.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanConfig, PcFanMeshResolution, solve_pc_fan_shape_gradient_reduced_adjoint,
};

const CASE_ID: &str = "pcfan.reduced-adjoint-reference.v1";

fn run() -> Result<[f64; 3], String> {
    let mut config = PcFanConfig::default_120mm();
    config.mesh = PcFanMeshResolution {
        radial_segments: 3,
        chord_segments: 2,
        circumferential_segments: 8,
        harmonic_count: 2,
    };
    let design = config.default_design_variables();
    let gradient = solve_pc_fan_shape_gradient_reduced_adjoint(&config, &design)
        .map_err(|error| error.to_string())?;
    if gradient.residual.max_abs > 1.0e-10
        || !gradient.adjoint.airflow_multiplier.is_finite()
        || gradient.adjoint.acoustic_pressure_multipliers.is_empty()
        || !gradient
            .adjoint
            .acoustic_pressure_multipliers
            .iter()
            .all(|value| value.is_finite())
        || !gradient
            .adjoint
            .acoustic_pressure_multipliers
            .iter()
            .any(|value| value.abs() > 0.0)
    {
        return Err("reduced full-state adjoint contract is invalid".to_string());
    }
    Ok([
        gradient.adjoint.airflow_multiplier,
        gradient.adjoint.acoustic_pressure_multipliers.len() as f64,
        gradient.residual.max_abs,
    ])
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .metric("airflow_multiplier", metrics[0])
                .metric("harmonic_state_count", metrics[1])
                .metric("maximum_primal_residual", metrics[2])
                .reference_checksum("analytic/deterministic-fixture:pcfan-reduced-adjoint-v1")
                .input_checksum("pcfan-120mm:reduced-order:harmonics=2")
                .seed("pcfan-reduced-adjoint-reference:cpu:v1");
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
