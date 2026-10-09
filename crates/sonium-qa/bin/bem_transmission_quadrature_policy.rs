//! Checked higher-order transmission quadrature-admission contract.

use sonium_bem::core::assembly::{
    TbemQuadraturePolicy, TransmissionAssemblyError,
    build_transmission_tbem_system_with_continuous_quadratic_checked,
    build_transmission_tbem_system_with_continuous_quadratic_checked_from_xem,
};
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use sonium_xem::AcousticMedium;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-quadrature-policy.v1";

fn run() -> Result<(usize, bool), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior = PhysicsParams::new(42.0, 343.0, 1.21, false);
    let interior = PhysicsParams::new(42.0, 360.0, 1.30, true);
    let permissive = TbemQuadraturePolicy {
        absolute_tolerance: 1.0e3,
        relative_tolerance: 1.0e3,
        ..TbemQuadraturePolicy::default()
    };
    let direct = build_transmission_tbem_system_with_continuous_quadratic_checked(
        &mesh.elements,
        &mesh.nodes,
        exterior.clone(),
        interior.clone(),
        permissive,
    )
    .map_err(|error| error.to_string())?;
    let incident = AcousticMedium::try_new(
        exterior.frequency,
        exterior.speed_of_sound,
        exterior.density,
    )
    .map_err(|error| error.to_string())?;
    let transmitted = AcousticMedium::try_new(
        interior.frequency,
        interior.speed_of_sound,
        interior.density,
    )
    .map_err(|error| error.to_string())?;
    let typed = build_transmission_tbem_system_with_continuous_quadratic_checked_from_xem(
        &mesh.elements,
        &mesh.nodes,
        incident,
        transmitted,
        permissive,
    )
    .map_err(|error| error.to_string())?;
    if direct.total_dofs() != typed.total_dofs() {
        return Err("typed-XEM and direct checked dimensions differ".into());
    }
    let invalid = TbemQuadraturePolicy {
        regular_coarse_order: 0,
        ..TbemQuadraturePolicy::default()
    };
    let rejected = build_transmission_tbem_system_with_continuous_quadratic_checked(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
        invalid,
    );
    let rejected_typed = matches!(
        rejected,
        Err(TransmissionAssemblyError::Quadrature {
            medium: "exterior",
            ..
        })
    );
    if !rejected_typed {
        return Err("invalid higher-order quadrature policy was not typed/rejected".into());
    }
    Ok((direct.total_dofs(), rejected_typed))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, rejected)) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("invalid_policy_rejected", rejected as u8 as f64)
            .minimum("invalid_policy_rejected", 1.0)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-0:continuous-quadratic-transmission")
            .reference_checksum("analytic/deterministic-fixture:bem-transmission-quadrature-policy-v1")
            .dimension("3-D CPU continuous-quadratic two-medium admission")
            .formulation("adjacent-order regular/singular quadrature validation before dense assembly")
            .tolerance("valid direct and typed-XEM paths agree in DOF dimension; invalid order fails with typed error")
            .seed("bem-transmission-quadrature-policy:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
