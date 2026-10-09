//! Compile-only QA wrapper for the two-way body-fitted swirl reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    BodyFittedCoupledSwirlOptions, BodyFittedDuctStation, BodyFittedFlowOptions,
    BodyFittedRotatingWallModel, BodyFittedSwirlOptions, solve_body_fitted_coupled_swirl_transport,
};

const CASE_ID: &str = "propeller.body-fitted-coupled-swirl-transport-reference.v1";

fn run() -> Result<(usize, f64, f64), String> {
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
    let result = solve_body_fitted_coupled_swirl_transport(
        &profile,
        flow_options,
        BodyFittedSwirlOptions {
            inlet_angular_velocity_rad_s: 10.0,
            ..BodyFittedSwirlOptions::default()
        },
        BodyFittedCoupledSwirlOptions::default(),
    )
    .map_err(|error| error.to_string())?;
    if !result.is_finite() || !result.converged {
        return Err("coupled body-fitted swirl result is not converged and finite".into());
    }
    Ok((
        result.outer_iterations,
        result.outer_residual,
        result.maximum_centrifugal_acceleration_m_s2,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((outer_iterations, outer_residual, maximum_centrifugal_acceleration)) => {
            let result = QaResult::passed(CASE_ID)
                .reference_checksum("fem/invariant:propeller-body-fitted-coupled-swirl-v1")
                .dimension("two-way steady axisymmetric CPU FEM meridional/swirl reference")
                .metric("outer_iterations", outer_iterations as f64)
                .metric("outer_residual", outer_residual)
                .metric(
                    "maximum_centrifugal_acceleration_m_s2",
                    maximum_centrifugal_acceleration,
                )
                .maximum("outer_residual", 1.0e-7)
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
