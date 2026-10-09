//! Compile-only QA wrapper for the body-fitted meridional FEM flow reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    BodyFittedDuctStation, BodyFittedFlowOptions, BodyFittedTurbulenceModel,
    solve_body_fitted_duct_flow,
};

const CASE_ID: &str = "propeller.body-fitted-meridional-flow-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let profile = [
        BodyFittedDuctStation {
            axial_position_m: 0.0,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.05,
            inner_radius_m: 0.021,
            outer_radius_m: 0.042,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.1,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
    ];
    let result = solve_body_fitted_duct_flow(
        &profile,
        BodyFittedFlowOptions {
            density_kg_m3: 1.0,
            dynamic_viscosity_pa_s: 1.0,
            turbulence_model: BodyFittedTurbulenceModel::KOmegaSstAlgebraic {
                c1: 0.09,
                characteristic_length_m: 0.03,
                nu_t_max_ratio: 20.0,
            },
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
        },
    )
    .map_err(|error| error.to_string())?;
    if !result.is_finite() || !result.converged {
        return Err("body-fitted flow result did not pass finite/converged admission".into());
    }
    Ok((
        result.residual,
        result.maximum_speed_m_s,
        result.relative_mean_velocity_mismatch,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((residual, maximum_speed, velocity_mismatch)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:propeller-body-fitted-flow-v1")
                .reference_checksum("fem/invariant:propeller-body-fitted-flow-v1")
                .dimension("three-station body-fitted meridional duct, P1/P1 CPU FEM")
                .formulation("laminar incompressible Navier--Stokes Picard flow state")
                .tolerance("finite converged state; residual <= 1e-6")
                .metric("picard_residual", residual)
                .metric("maximum_speed_m_s", maximum_speed)
                .metric("relative_mean_velocity_mismatch", velocity_mismatch)
                .seed("propeller-body-fitted-flow-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serializes")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
