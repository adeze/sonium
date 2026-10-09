//! Compile-only QA wrapper for the coupled body-fitted swirl transport path.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    BodyFittedDuctStation, BodyFittedFlowOptions, BodyFittedRotatingWallModel,
    BodyFittedSwirlOptions, solve_body_fitted_swirl_transport,
};

const CASE_ID: &str = "propeller.body-fitted-swirl-transport-reference.v1";

fn run() -> Result<(f64, f64), String> {
    let profile = vec![
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
    let flow_options = BodyFittedFlowOptions {
        density_kg_m3: 1.0,
        dynamic_viscosity_pa_s: 1.0,
        inlet_axial_velocity_m_s: 0.0,
        rotating_wall_model: BodyFittedRotatingWallModel::AnnularCouette {
            inner_angular_velocity_rad_s: 10.0,
            outer_angular_velocity_rad_s: 10.0,
        },
        ..BodyFittedFlowOptions::default()
    };
    let result = solve_body_fitted_swirl_transport(
        &profile,
        flow_options,
        BodyFittedSwirlOptions {
            inlet_angular_velocity_rad_s: 10.0,
            ..BodyFittedSwirlOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    if !result.is_finite() {
        return Err("coupled body-fitted swirl result is not finite".into());
    }
    Ok((result.linear_residual, result.maximum_tangential_speed_m_s))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((linear_residual, maximum_tangential_speed)) => {
            let result = QaResult::passed(CASE_ID)
                .reference_checksum("fem/invariant:propeller-body-fitted-swirl-transport-v1")
                .dimension("three-station axisymmetric body-fitted CPU FEM swirl transport")
                .metric("linear_residual", linear_residual)
                .metric("maximum_tangential_speed_m_s", maximum_tangential_speed)
                .maximum("linear_residual", 1.0e-6)
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
