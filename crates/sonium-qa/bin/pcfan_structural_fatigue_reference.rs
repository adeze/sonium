//! Compile-only QA contract for the reduced PC-fan fatigue screen.
//!
//! This deterministic case checks the Basquin scaling and fail-closed
//! configuration admission. It is not product-scale fatigue validation.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCfdEvaluationError, PcFanConfig, evaluate_pc_fan, try_evaluate_pc_fan,
};

const CASE_ID: &str = "pcfan.structural-fatigue-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let mut config = PcFanConfig::bench_120mm();
    config.weights.airflow_band_fraction = 1.0;
    let design = config.default_design_variables();

    let nominal = evaluate_pc_fan(&config, &design);
    if !nominal.is_feasible(&config) {
        return Err(format!(
            "nominal reduced-order design is infeasible: {:?}",
            nominal.structural
        ));
    }

    let mut long_life_config = config.clone();
    long_life_config.structural.fatigue_design_cycles = 1.0e10;
    let long_life = evaluate_pc_fan(&long_life_config, &design);
    if long_life.structural.fatigue_allowable_stress_pa
        >= nominal.structural.fatigue_allowable_stress_pa
    {
        return Err("fatigue allowable did not decrease for longer design life".into());
    }

    let mut invalid_config = config.clone();
    invalid_config.structural.fatigue_exponent = 0.0;
    if !matches!(
        try_evaluate_pc_fan(&invalid_config, &design),
        Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
            envelope: "structural",
            parameter: "fatigue_exponent",
            ..
        })
    ) {
        return Err("invalid fatigue exponent was not rejected".into());
    }

    Ok((
        nominal.structural.fatigue_alternating_stress_pa,
        nominal.structural.fatigue_allowable_stress_pa,
        nominal.structural.fatigue_margin_pa,
    ))
}

fn main() {
    match run() {
        Ok((alternating_stress_pa, allowable_stress_pa, margin_pa)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("fatigue_alternating_stress_pa", alternating_stress_pa)
                .metric("fatigue_allowable_stress_pa", allowable_stress_pa)
                .metric("fatigue_margin_pa", margin_pa)
                .maximum("fatigue_alternating_stress_pa", allowable_stress_pa)
                .input_checksum("analytic/deterministic-fixture:pcfan-fatigue-v1")
                .reference(
                    "reduced-order Basquin scaling and typed invalid-configuration rejection",
                )
                .reference_checksum("analytic/deterministic-fixture:pcfan-fatigue-v1")
                .dimension("reduced-order 120 mm PC-fan structural envelope")
                .formulation("temperature-coupled reduced-order structural fatigue screen")
                .tolerance("finite positive allowable stress and non-negative fatigue margin")
                .seed("pcfan-structural-fatigue-reference:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("PC-fan fatigue result must serialize")
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
                    .expect("PC-fan fatigue failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
