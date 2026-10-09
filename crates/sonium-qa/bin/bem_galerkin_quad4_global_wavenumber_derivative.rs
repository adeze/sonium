//! Bounded CPU QA for global Quad4 analytic wavenumber derivatives.

use num_complex::Complex64;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, PhysicsParams, TransmissionSide,
    assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-quad4-global-wavenumber-derivative.v1";
const REFERENCE: &str =
    "analytic/deterministic-fixture:bem-galerkin-quad4-global-wavenumber-derivative-v1";

fn quad(connectivity: Vec<usize>, dof_addresses: Vec<usize>) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.5, 0.5, 0.5],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses,
    }
}

fn closed_cube() -> (Vec<Element>, Vec<[f64; 3]>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let connectivity = [
        vec![0, 1, 2, 3],
        vec![4, 7, 6, 5],
        vec![0, 4, 5, 1],
        vec![1, 5, 6, 2],
        vec![2, 6, 7, 3],
        vec![3, 7, 4, 0],
    ];
    let elements = connectivity
        .into_iter()
        .map(|connectivity| {
            let dof_addresses = connectivity.clone();
            quad(connectivity, dof_addresses)
        })
        .collect();
    (elements, nodes)
}

fn with_wavenumber(medium: &PhysicsParams, wave_number: f64) -> PhysicsParams {
    let mut perturbed = medium.clone();
    perturbed.speed_of_sound = perturbed.omega / wave_number;
    perturbed.wave_number = wave_number;
    perturbed.wave_length = perturbed.speed_of_sound / perturbed.frequency;
    perturbed.pressure_factor = perturbed.density * perturbed.omega * perturbed.harmonic_factor;
    perturbed
}

fn run() -> Result<(f64, f64), String> {
    let (elements, nodes) = closed_cube();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false)
        .map_err(|error| format!("exterior physics: {error}"))?;
    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true)
        .map_err(|error| format!("interior physics: {error}"))?;
    let analytic =
        assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            2,
            8,
            TransmissionSide::Exterior,
        )
        .map_err(|error| error.to_string())?;
    let step = (exterior.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number + step),
        &interior,
        BasisOrder::Linear,
        2,
        8,
    )
    .map_err(|error| error.to_string())?;
    let minus = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number - step),
        &interior,
        BasisOrder::Linear,
        2,
        8,
    )
    .map_err(|error| error.to_string())?;

    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for ((analytic, plus), minus) in analytic
        .as_slice()
        .iter()
        .zip(plus.matrix().as_slice())
        .zip(minus.matrix().as_slice())
    {
        let reference = (*plus - *minus) / (2.0 * step);
        max_error = max_error.max((*analytic - reference).norm());
        max_reference = max_reference.max(reference.norm());
    }
    let relative_error = max_error / (1.0 + max_reference);
    if !relative_error.is_finite() || relative_error > 2.0e-3 {
        return Err(format!(
            "global derivative relative error {relative_error:e}"
        ));
    }

    let compressed =
        compress_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            2,
            8,
            TransmissionSide::Exterior,
            2,
            1.0e-6,
            8,
        )
        .map_err(|error| error.to_string())?;
    let trial = vec![Complex64::new(0.03, -0.01); compressed.total_dofs()];
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let dense_action = analytic.dot(&trial);
    let action_error = compressed_action
        .iter()
        .zip(dense_action.iter())
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0_f64, f64::max);
    if !action_error.is_finite() || action_error > 1.0e-6 {
        return Err(format!("compressed action error {action_error:e}"));
    }
    Ok((relative_error, action_error))
}

fn main() {
    let started = Instant::now();
    let result = run();
    let qa = match &result {
        Ok((relative_error, action_error)) => QaResult::passed(CASE_ID)
            .input_checksum(REFERENCE)
            .reference_checksum(REFERENCE)
            .dimension("closed 3-D Quad4 cube; 8 interface DOFs; global derivative + ACA action")
            .metric("max_relative_error", *relative_error)
            .metric("compressed_action_error", *action_error)
            .seed("bem-galerkin-quad4-global-wavenumber-derivative:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error.clone())
            .input_checksum(REFERENCE)
            .reference_checksum(REFERENCE)
            .dimension("closed 3-D Quad4 cube; 8 interface DOFs; global derivative + ACA action")
            .seed("bem-galerkin-quad4-global-wavenumber-derivative:cpu:v1"),
    };
    println!(
        "{}",
        qa.elapsed_ms(started.elapsed().as_millis() as u64)
            .to_json_line()
            .expect("valid QA result should serialize")
    );
}
