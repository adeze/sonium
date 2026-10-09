//! Compile-only QA wrapper for axisymmetric weighted and rotating-wall references.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    AxisymmetricFlowOptions, AxisymmetricRotatingWallOptions, BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1,
    BodyFittedDuctStation, BodyFittedFlowNode, BodyFittedFlowResult,
    evaluate_axisymmetric_rotating_wall, evaluate_axisymmetric_weighted_flow,
};

const CASE_ID: &str = "propeller.axisymmetric-weighted-rotating-wall-reference.v1";

fn run() -> Result<(f64, f64), String> {
    let mut nodes = Vec::new();
    for axial_position_m in [0.0, 1.0] {
        for radial_position_m in [0.02, 0.035, 0.05] {
            nodes.push(BodyFittedFlowNode {
                radial_position_m,
                axial_position_m,
                radial_velocity_m_s: 0.0,
                axial_velocity_m_s: 0.1,
                pressure_pa: 0.0,
            });
        }
    }
    let meridional = BodyFittedFlowResult {
        schema: BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1.into(),
        nodes,
        mesh_node_count: 6,
        mesh_element_count: 4,
        iterations: 2,
        residual: 1.0e-9,
        converged: true,
        inlet_mean_axial_velocity_m_s: 0.1,
        outlet_mean_axial_velocity_m_s: 0.1,
        relative_mean_velocity_mismatch: 0.0,
        maximum_speed_m_s: 0.1,
        minimum_pressure_pa: 0.0,
        maximum_pressure_pa: 0.0,
    };
    let weighted = evaluate_axisymmetric_weighted_flow(
        &meridional,
        AxisymmetricFlowOptions {
            density_kg_m3: 1.2,
            axial_tolerance_m: 1.0e-9,
        },
    )
    .map_err(|error| error.to_string())?;
    let rotating = evaluate_axisymmetric_rotating_wall(
        &[
            BodyFittedDuctStation {
                axial_position_m: 0.0,
                inner_radius_m: 0.02,
                outer_radius_m: 0.05,
            },
            BodyFittedDuctStation {
                axial_position_m: 1.0,
                inner_radius_m: 0.02,
                outer_radius_m: 0.05,
            },
        ],
        AxisymmetricRotatingWallOptions::default(),
    )
    .map_err(|error| error.to_string())?;
    if !weighted.is_finite() || !rotating.is_finite() {
        return Err("axisymmetric reference result failed finite admission".into());
    }
    Ok((
        weighted.relative_volume_flow_mismatch,
        rotating.maximum_absolute_torque_n_m_per_m,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((flow_mismatch, torque)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:propeller-axisymmetric-flow-v1")
                .reference_checksum("fem/invariant:propeller-axisymmetric-flow-v1")
                .dimension("2-D axisymmetric weighted ledger and annular Couette reference")
                .metric("relative_volume_flow_mismatch", flow_mismatch)
                .metric("maximum_absolute_torque_n_m_per_m", torque)
                .seed("propeller-axisymmetric-flow-v1")
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
