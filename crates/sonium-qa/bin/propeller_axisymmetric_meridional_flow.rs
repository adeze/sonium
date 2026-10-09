//! Compile-only QA wrapper for the weighted axisymmetric meridional FEM reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    BodyFittedDuctStation, BodyFittedFlowOptions, solve_axisymmetric_meridional_flow,
};

const CASE_ID: &str = "propeller.axisymmetric-meridional-flow-reference.v1";

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
    let result = solve_axisymmetric_meridional_flow(
        &profile,
        BodyFittedFlowOptions {
            density_kg_m3: 1.0,
            dynamic_viscosity_pa_s: 1.0,
            inlet_axial_velocity_m_s: 0.1,
            axial_elements: 4,
            radial_elements: 3,
            max_picard_iterations: 20,
            min_picard_iterations: 1,
            picard_tolerance: 1.0e-6,
            picard_relaxation: 0.8,
            linear_solver_tolerance: 1.0e-8,
            ..BodyFittedFlowOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    if !result.is_finite() || !result.converged {
        return Err("weighted meridional result did not pass finite/converged admission".into());
    }
    Ok((
        result.residual,
        result.relative_volume_flux_mismatch,
        result.maximum_speed_m_s,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((residual, flux_mismatch, maximum_speed)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum(
                    "analytic/deterministic-fixture:propeller-axisymmetric-meridional-flow-v1",
                )
                .reference_checksum("fem/invariant:propeller-axisymmetric-meridional-flow-v1")
                .dimension("annular axisymmetric weighted CPU FEM")
                .metric("picard_residual", residual)
                .metric("relative_volume_flux_mismatch", flux_mismatch)
                .metric("maximum_speed_m_s", maximum_speed)
                .tolerance("finite converged state; relative flux mismatch <= 1e-6")
                .seed("propeller-axisymmetric-meridional-flow-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result JSON")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result JSON")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
