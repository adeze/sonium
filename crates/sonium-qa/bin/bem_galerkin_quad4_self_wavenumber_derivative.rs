//! Bounded CPU QA for the analytic self-Quad4 Duffy wavenumber derivative.

use num_complex::Complex64;
use sonium_bem::PhysicsParams;
use sonium_bem::core::assembly::common::get_element_coords;
use sonium_bem::core::integration::{
    galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis,
    galerkin_single_layer_self_quad4_with_basis,
};
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-quad4-self-wavenumber-derivative.v1";
const REFERENCE: &str =
    "analytic/deterministic-fixture:bem-galerkin-quad4-self-wavenumber-derivative-v1";

fn with_wavenumber(medium: &PhysicsParams, wave_number: f64) -> PhysicsParams {
    let mut perturbed = medium.clone();
    perturbed.speed_of_sound = perturbed.omega / wave_number;
    perturbed.wave_number = wave_number;
    perturbed.wave_length = perturbed.speed_of_sound / perturbed.frequency;
    perturbed.pressure_factor = perturbed.density * perturbed.omega * perturbed.harmonic_factor;
    perturbed
}

fn run() -> Result<f64, String> {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.2],
        [0.0, 1.0, 0.0],
    ];
    let element = Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.5, 0.5, 0.05],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses: (0..4).collect(),
    };
    let medium = PhysicsParams::try_new(420.0, 343.0, 1.21, false)
        .map_err(|error| format!("physics: {error}"))?;
    let coords = get_element_coords(&element, &nodes);
    let gauss_order = 2;
    let analytic = galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis(
        &coords,
        &medium,
        4,
        gauss_order,
    );
    let step = (medium.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = galerkin_single_layer_self_quad4_with_basis(
        &coords,
        &with_wavenumber(&medium, medium.wave_number + step),
        4,
        gauss_order,
    );
    let minus = galerkin_single_layer_self_quad4_with_basis(
        &coords,
        &with_wavenumber(&medium, medium.wave_number - step),
        4,
        gauss_order,
    );
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for (analytic_row, (plus_row, minus_row)) in analytic
        .basis_results
        .iter()
        .zip(plus.basis_results.iter().zip(minus.basis_results.iter()))
    {
        for ((analytic, plus), minus) in analytic_row
            .iter()
            .zip(plus_row.iter())
            .zip(minus_row.iter())
        {
            let reference = (plus.g_integral - minus.g_integral) / (2.0 * step);
            max_error = max_error.max((analytic.g_integral - reference).norm());
            max_reference = max_reference.max(reference.norm());
        }
    }
    let relative_error = max_error / (1.0 + max_reference);
    if !relative_error.is_finite() || relative_error > 5.0e-4 {
        return Err(format!(
            "self Quad4 analytic derivative error {relative_error:e}"
        ));
    }
    Ok(relative_error)
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(relative_error) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("maximum_relative_derivative_error", relative_error)
            .maximum("maximum_relative_derivative_error", 5.0e-4)
            .input_checksum(REFERENCE)
            .reference_checksum(REFERENCE)
            .dimension("3-D self isoparametric Quad4 Duffy chart")
            .seed("bem-galerkin-quad4-self-wavenumber-derivative:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum(REFERENCE)
            .reference_checksum(REFERENCE)
            .dimension("3-D self isoparametric Quad4 Duffy chart")
            .seed("bem-galerkin-quad4-self-wavenumber-derivative:cpu:v1"),
    };
    println!(
        "{}",
        result
            .elapsed_ms(started.elapsed().as_millis() as u64)
            .to_json_line()
            .expect("valid QA result should serialize")
    );
}
