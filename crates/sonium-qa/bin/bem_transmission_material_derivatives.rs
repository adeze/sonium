//! Release-facing constant-panel transmission material-derivative contract.

use num_complex::Complex64;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::Element;
use sonium_bem::{
    BasisOrder, PhysicsParams, TransmissionMaterialParameter, TransmissionSide,
    build_transmission_tbem_system, build_transmission_tbem_system_with_basis,
    build_transmission_tbem_system_with_continuous_quadratic,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-material-derivatives.v1";
const FREQUENCY_HZ: f64 = 420.0;
const SPEED_STEP: f64 = 1.0e-2;
const DENSITY_STEP: f64 = 1.0e-5;

fn setup_mesh() -> (Vec<Element>, Vec<[f64; 3]>) {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    (elements, mesh.nodes)
}

fn build(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
) -> Result<sonium_bem::TransmissionTbemSystem, String> {
    build_transmission_tbem_system(elements, nodes, exterior, interior)
        .map_err(|error| error.to_string())
}

fn max_matrix_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected).norm() / actual.norm().max(expected.norm()).max(1.0e-3)
        })
        .fold(0.0, f64::max)
}

fn centered_difference(
    plus: &sonium_bem::TransmissionTbemSystem,
    minus: &sonium_bem::TransmissionTbemSystem,
    step: f64,
) -> Vec<Complex64> {
    plus.matrix()
        .as_slice()
        .iter()
        .zip(minus.matrix().as_slice())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect()
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64), String> {
    let (elements, nodes) = setup_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).map_err(|e| e.to_string())?;
    let interior =
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true).map_err(|e| e.to_string())?;
    let system = build(&elements, &nodes, exterior.clone(), interior.clone())?;

    let speed_plus = build(
        &elements,
        &nodes,
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            exterior.speed_of_sound + SPEED_STEP,
            exterior.density,
            false,
        )
        .map_err(|e| e.to_string())?,
        interior.clone(),
    )?;
    let speed_minus = build(
        &elements,
        &nodes,
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            exterior.speed_of_sound - SPEED_STEP,
            exterior.density,
            false,
        )
        .map_err(|e| e.to_string())?,
        interior.clone(),
    )?;
    let speed_derivative = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .map_err(|e| e.to_string())?;
    let speed_fd = centered_difference(&speed_plus, &speed_minus, SPEED_STEP);
    let speed_error = max_matrix_error(speed_derivative.as_slice(), &speed_fd);

    let wavenumber_derivative = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .map_err(|e| e.to_string())?;
    let chain_scale = -exterior.wave_number / exterior.speed_of_sound;
    let chain_expected: Vec<Complex64> = wavenumber_derivative
        .as_slice()
        .iter()
        .map(|value| *value * chain_scale)
        .collect();
    let chain_error = max_matrix_error(speed_derivative.as_slice(), &chain_expected);

    let density_plus = build(
        &elements,
        &nodes,
        exterior.clone(),
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            interior.speed_of_sound,
            interior.density + DENSITY_STEP,
            true,
        )
        .map_err(|e| e.to_string())?,
    )?;
    let density_minus = build(
        &elements,
        &nodes,
        exterior.clone(),
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            interior.speed_of_sound,
            interior.density - DENSITY_STEP,
            true,
        )
        .map_err(|e| e.to_string())?,
    )?;
    let density_derivative = system
        .matrix_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        )
        .map_err(|e| e.to_string())?;
    let density_fd = centered_difference(&density_plus, &density_minus, DENSITY_STEP);
    let density_error = max_matrix_error(density_derivative.as_slice(), &density_fd);

    let mut linear_elements = elements.clone();
    for (index, element) in linear_elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    let linear = build_transmission_tbem_system_with_basis(
        &linear_elements,
        &nodes,
        exterior.clone(),
        interior.clone(),
        BasisOrder::Linear,
    )
    .map_err(|error| error.to_string())?;
    let linear_derivative = linear
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .map_err(|error| error.to_string())?;
    let linear_finite = linear_derivative
        .as_slice()
        .iter()
        .filter(|value| value.re.is_finite() && value.im.is_finite())
        .count() as f64;

    let continuous_exterior = PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false)
        .map_err(|error| error.to_string())?;
    let continuous_interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .map_err(|error| error.to_string())?;
    let continuous = build_transmission_tbem_system_with_continuous_quadratic(
        &elements,
        &nodes,
        continuous_exterior.clone(),
        continuous_interior.clone(),
    )
    .map_err(|error| error.to_string())?;
    let continuous_derivative = continuous
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .map_err(|error| error.to_string())?;
    let continuous_finite = continuous_derivative
        .as_slice()
        .iter()
        .filter(|value| value.re.is_finite() && value.im.is_finite())
        .count() as f64;

    let continuous_plus = build_transmission_tbem_system_with_continuous_quadratic(
        &elements,
        &nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0 + SPEED_STEP, 1.21, false)
            .map_err(|error| error.to_string())?,
        continuous_interior.clone(),
    )
    .map_err(|error| error.to_string())?;
    let continuous_minus = build_transmission_tbem_system_with_continuous_quadratic(
        &elements,
        &nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0 - SPEED_STEP, 1.21, false)
            .map_err(|error| error.to_string())?,
        continuous_interior,
    )
    .map_err(|error| error.to_string())?;
    let continuous_speed_fd = centered_difference(&continuous_plus, &continuous_minus, SPEED_STEP);
    let continuous_speed_error =
        max_matrix_error(continuous_derivative.as_slice(), &continuous_speed_fd);

    if !speed_error.is_finite()
        || !chain_error.is_finite()
        || !density_error.is_finite()
        || linear_finite <= 0.0
        || continuous_finite <= 0.0
        || !continuous_speed_error.is_finite()
    {
        return Err("non-finite transmission derivative metric".into());
    }
    Ok((
        speed_error,
        chain_error,
        density_error,
        linear_finite,
        continuous_finite,
        continuous_speed_error,
    ))
}

fn main() {
    let started = Instant::now();
    match std::panic::catch_unwind(run) {
        Ok(Ok((
            speed_error,
            chain_error,
            density_error,
            linear_finite,
            continuous_finite,
            continuous_speed_error,
        ))) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("speed_fd_relative_error", speed_error)
                .metric("speed_chain_relative_error", chain_error)
                .metric("density_fd_relative_error", density_error)
                .metric("linear_higher_order_finite_entries", linear_finite)
                .metric("continuous_quadratic_finite_entries", continuous_finite)
                .metric(
                    "continuous_quadratic_speed_fd_relative_error",
                    continuous_speed_error,
                )
                .maximum("speed_fd_relative_error", 4.0e-5)
                .maximum("speed_chain_relative_error", 2.0e-12)
                .maximum("density_fd_relative_error", 2.0e-8)
                .maximum("continuous_quadratic_speed_fd_relative_error", 4.0e-5)
                .minimum("linear_higher_order_finite_entries", 1.0)
                .minimum("continuous_quadratic_finite_entries", 1.0)
                .input_checksum(
                    "analytic/deterministic-fixture:bem-transmission-material-derivatives-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-transmission-material-derivatives-v1",
                )
                .dimension("3D closed icosphere, constant-panel two-medium interface")
                .formulation("dense constant-panel Müller material derivatives")
                .tolerance("analytic speed/density derivatives match centered rebuilt systems")
                .seed("bem-transmission-material-derivatives:icosphere-level-0:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let record =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            let record = QaResult::failed(CASE_ID, "panic during QA execution")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
