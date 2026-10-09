//! Compile-only QA wrapper for the rotating-wall centrifugal source path.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    BodyFittedDuctStation, BodyFittedFlowOptions, BodyFittedRotatingWallModel,
    solve_body_fitted_duct_flow,
};

const CASE_ID: &str = "propeller.body-fitted-rotating-wall-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let rotating_wall = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: 25.0,
        outer_angular_velocity_rad_s: 25.0,
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
        rotating_wall_model: rotating_wall,
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
    let mid_acceleration = rotating_wall.centrifugal_acceleration_m_s2(0.035, 0.02, 0.05);
    if !result.is_finite() || !mid_acceleration.is_finite() || mid_acceleration <= 0.0 {
        return Err("rotating-wall body-fitted result failed admission".into());
    }
    Ok((result.residual, result.maximum_speed_m_s, mid_acceleration))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((residual, maximum_speed, mid_acceleration)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum(
                    "analytic/deterministic-fixture:propeller-body-fitted-rotating-wall-v1",
                )
                .reference_checksum("fem/invariant:propeller-body-fitted-rotating-wall-v1")
                .dimension(
                    "three-station body-fitted CPU FEM with annular-Couette centrifugal source",
                )
                .metric("picard_residual", residual)
                .metric("maximum_speed_m_s", maximum_speed)
                .metric(
                    "mid_annulus_centrifugal_acceleration_m_s2",
                    mid_acceleration,
                )
                .tolerance("finite converged flow; positive centrifugal source")
                .seed("propeller-body-fitted-rotating-wall-v1")
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
