//! Release-facing constant-panel impedance sound-speed derivative contract.

use num_complex::Complex64;
use sonium_bem::core::assembly::{
    build_tbem_system_stabilized, build_tbem_system_stabilized_impedance_speed_derivative,
};
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, Element, PhysicsParams};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.impedance-speed-derivative.v1";
const FREQUENCY_HZ: f64 = 800.0;
const DENSITY: f64 = 1.21;
const SPEED: f64 = 343.0;
const SPEED_STEP: f64 = 1.0e-5;

fn fixture(speed: f64) -> Result<(Vec<Element>, Vec<[f64; 3]>, PhysicsParams), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    let velocity = Complex64::new(0.2, -0.1);
    let admittance = Complex64::new(1.0 / (DENSITY * speed), 0.0);

    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
        element.boundary_condition = if index == 0 {
            BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![velocity],
                admittance,
            }
        } else {
            BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)])
        };
    }

    let physics = PhysicsParams::try_new(FREQUENCY_HZ, speed, DENSITY, false)
        .map_err(|error| error.to_string())?;
    Ok((elements, mesh.nodes, physics))
}

fn max_relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected).norm() / actual.norm().max(expected.norm()).max(1.0e-8)
        })
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64), String> {
    let (elements, nodes, physics) = fixture(SPEED)?;
    let (plus_elements, plus_nodes, plus_physics) = fixture(SPEED + SPEED_STEP)?;
    let (minus_elements, minus_nodes, minus_physics) = fixture(SPEED - SPEED_STEP)?;

    let plus = build_tbem_system_stabilized(&plus_elements, &plus_nodes, &plus_physics, 1.0, 1.0);
    let minus =
        build_tbem_system_stabilized(&minus_elements, &minus_nodes, &minus_physics, 1.0, 1.0);
    let derivative = build_tbem_system_stabilized_impedance_speed_derivative(
        &elements, &nodes, &physics, 1.0, 1.0,
    );

    let matrix_fd: Vec<_> = plus
        .matrix
        .as_slice()
        .iter()
        .zip(minus.matrix.as_slice())
        .map(|(&plus, &minus)| (plus - minus) / (2.0 * SPEED_STEP))
        .collect();
    let rhs_fd: Vec<_> = plus
        .rhs
        .iter()
        .zip(minus.rhs.iter())
        .map(|(&plus, &minus)| (plus - minus) / (2.0 * SPEED_STEP))
        .collect();
    let matrix_error = max_relative_error(derivative.matrix.as_slice(), &matrix_fd);
    let rhs_error = max_relative_error(&derivative.rhs, &rhs_fd);

    if !matrix_error.is_finite() || !rhs_error.is_finite() {
        return Err("non-finite impedance speed derivative metric".into());
    }
    Ok((matrix_error, rhs_error))
}

fn main() {
    let started = Instant::now();
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok((matrix_error, rhs_error))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("matrix_relative_error", matrix_error)
            .metric("rhs_relative_error", rhs_error)
            .maximum("matrix_relative_error", 2.0e-5)
            .maximum("rhs_relative_error", 2.0e-5)
            .input_checksum("analytic/deterministic-fixture:bem-impedance-speed-derivative-v1")
            .reference("central speed perturbation with Y=1/(rho*c)")
            .reference_checksum("analytic/deterministic-fixture:bem-impedance-speed-derivative-v1")
            .dimension("3-D constant-panel stabilized TBEM with one impedance panel")
            .formulation("CPU constant-panel Burton-Miller impedance assembly")
            .tolerance("matrix and RHS relative derivative errors <= 2e-5")
            .seed("bem-impedance-speed-derivative:icosphere-level-0:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Err(error)) => {
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64)
        }
        Err(_) => QaResult::failed(CASE_ID, "panic during QA execution")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
