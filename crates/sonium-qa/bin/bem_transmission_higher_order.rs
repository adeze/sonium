//! Bounded CPU evidence for higher-order dense Müller transmission assembly.

use std::time::Instant;

use sonium_bem::core::assembly::build_transmission_tbem_system_with_basis;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::transmission::solve_incident_transmission;
use sonium_bem::core::types::{BasisOrder, PhysicsParams};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.transmission-higher-order.v1";

fn run() -> (usize, usize, f64, bool) {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("exterior physics");
    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).expect("interior physics");
    let linear = build_transmission_tbem_system_with_basis(
        &mesh.elements,
        &mesh.nodes,
        exterior.clone(),
        interior.clone(),
        BasisOrder::Linear,
    )
    .expect("linear transmission assembly");
    let incident = IncidentField::plane_wave_z();
    let solution =
        solve_incident_transmission(&linear, &incident, 1.0e-9).expect("linear transmission solve");
    let field = solution
        .evaluate_exterior_field_checked(&linear, &[[0.0, 0.0, 0.8]], &incident)
        .expect("linear exterior field evaluation");
    let quadratic = build_transmission_tbem_system_with_basis(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
        BasisOrder::Quadratic,
    )
    .expect("quadratic transmission assembly");
    let finite = field
        .iter()
        .all(|point| point.p_total.re.is_finite() && point.p_total.im.is_finite())
        && quadratic
            .matrix()
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite());
    (
        linear.interface_dofs(),
        quadratic.interface_dofs(),
        solution.relative_residual,
        finite,
    )
}

fn main() {
    let started = Instant::now();
    let result = match std::panic::catch_unwind(run) {
        Ok((linear_dofs, quadratic_dofs, residual, finite)) if finite && residual <= 1.0e-9 => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("linear_interface_dofs", linear_dofs as f64)
                .metric("quadratic_interface_dofs", quadratic_dofs as f64)
                .metric("linear_residual", residual)
                .minimum("linear_interface_dofs", 1.0)
                .minimum("quadratic_interface_dofs", 1.0)
                .maximum("linear_residual", 1.0e-9)
                .metric("finite_outputs", 1.0)
                .input_checksum("bem:icosphere:subdivision-1:linear-quadratic-transmission")
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-transmission-higher-order-v1",
                )
                .dimension(
                    "3-D dense CPU Müller transmission with discontinuous linear/quadratic traces",
                )
                .seed("bem-transmission-higher-order:cpu:v1")
        }
        Ok((linear_dofs, quadratic_dofs, residual, finite)) => {
            QaResult::failed(CASE_ID, "higher-order transmission metrics failed")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("linear_interface_dofs", linear_dofs as f64)
                .metric("quadratic_interface_dofs", quadratic_dofs as f64)
                .metric("linear_residual", residual)
                .metric("finite_outputs", f64::from(finite))
                .input_checksum("bem:icosphere:subdivision-1:linear-quadratic-transmission")
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-transmission-higher-order-v1",
                )
                .dimension(
                    "3-D dense CPU Müller transmission with discontinuous linear/quadratic traces",
                )
                .seed("bem-transmission-higher-order:cpu:v1")
        }
        Err(_) => QaResult::failed(CASE_ID, "higher-order transmission QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-1:linear-quadratic-transmission")
            .reference_checksum("analytic/deterministic-fixture:bem-transmission-higher-order-v1")
            .dimension(
                "3-D dense CPU Müller transmission with discontinuous linear/quadratic traces",
            )
            .seed("bem-transmission-higher-order:cpu:v1"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    println!("ELAPSED_MS: {}", started.elapsed().as_millis());
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
