//! Compile-only QA wrapper for the turbulence-enabled body-fitted flow lane.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    BodyFittedDuctStation, BodyFittedFlowOptions, BodyFittedTurbulenceModel,
    solve_body_fitted_duct_flow,
};

const CASE_ID: &str = "propeller.body-fitted-turbulence-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let model = BodyFittedTurbulenceModel::KOmegaSstAlgebraic {
        c1: 0.09,
        characteristic_length_m: 0.03,
        nu_t_max_ratio: 20.0,
    };
    let profile = [
        BodyFittedDuctStation {
            axial_position_m: 0.0,
            inner_radius_m: 0.02,
            outer_radius_m: 5.0e-2,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.05,
            inner_radius_m: 0.021,
            outer_radius_m: 0.042,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.1,
            inner_radius_m: 0.02,
            outer_radius_m: 5.0e-2,
        },
    ];
    let options = BodyFittedFlowOptions {
        density_kg_m3: 1.0,
        dynamic_viscosity_pa_s: 1.0,
        turbulence_model: model,
        inlet_axial_velocity_m_s: 0.1,
        axial_elements: 4,
        radial_elements: 3,
        max_picard_iterations: 20,
        min_picard_iterations: 1,
        picard_tolerance: 1.0e-6,
        picard_relaxation: 0.8,
        max_linear_solver_iterations: 100,
        linear_solver_tolerance: 1.0e-8,
        ..BodyFittedFlowOptions::default()
    };
    let result =
        solve_body_fitted_duct_flow(&profile, options).map_err(|error| error.to_string())?;
    let effective_viscosity = model.effective_dynamic_viscosity_pa_s(
        options.dynamic_viscosity_pa_s,
        options.density_kg_m3,
        result.maximum_speed_m_s,
    );
    let viscosity_ratio = effective_viscosity / options.dynamic_viscosity_pa_s;
    if !result.is_finite() || !effective_viscosity.is_finite() || viscosity_ratio <= 1.0 {
        return Err("turbulence-enabled body-fitted result failed admission".into());
    }
    Ok((result.residual, result.maximum_speed_m_s, viscosity_ratio))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((residual, maximum_speed, viscosity_ratio)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum(
                    "analytic/deterministic-fixture:propeller-body-fitted-turbulence-v1",
                )
                .reference_checksum("fem/invariant:propeller-body-fitted-turbulence-v1")
                .dimension("three-station body-fitted CPU FEM with algebraic k-omega SST")
                .metric("picard_residual", residual)
                .metric("maximum_speed_m_s", maximum_speed)
                .metric("effective_to_molecular_viscosity_ratio", viscosity_ratio)
                .tolerance("finite/converged flow; effective viscosity ratio > 1")
                .seed("propeller-body-fitted-turbulence-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
