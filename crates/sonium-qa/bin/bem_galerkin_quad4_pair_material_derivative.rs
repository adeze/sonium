//! Bounded CPU QA for curved Quad4 Müller pair material derivatives.

use num_complex::Complex64;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, PhysicsParams, TransmissionMaterialParameter, TransmissionSide,
    assemble_galerkin_muller_transmission_quad4_pair_material_derivative,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-quad4-pair-material-derivative.v1";

fn element(connectivity: Vec<usize>) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.5, 0.5, 0.0],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![0, 1, 2, 3],
    }
}

fn run() -> Result<(usize, f64), String> {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.3],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.4],
        [1.0, 0.0, 1.4],
        [1.0, 1.0, 1.7],
        [0.0, 1.0, 1.4],
    ];
    let source = element(vec![0, 1, 2, 3]);
    let field = element(vec![4, 5, 6, 7]);
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false)
        .map_err(|error| format!("invalid exterior physics: {error}"))?;
    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true)
        .map_err(|error| format!("invalid interior physics: {error}"))?;
    let derivative = assemble_galerkin_muller_transmission_quad4_pair_material_derivative(
        &source,
        &field,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        4,
        TransmissionSide::Exterior,
        TransmissionMaterialParameter::SpeedOfSound,
    )
    .map_err(|error| error.to_string())?;
    if derivative.shape() != (8, 8) {
        return Err(format!(
            "unexpected derivative shape {:?}",
            derivative.shape()
        ));
    }
    let max_norm = derivative
        .as_slice()
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    if !max_norm.is_finite() || max_norm <= f64::MIN_POSITIVE {
        return Err(format!(
            "derivative entries are not finite/nonzero: {max_norm}"
        ));
    }
    Ok((derivative.nrows(), max_norm))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, max_norm)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("derivative_dimension", dimension as f64)
            .metric("maximum_derivative_entry_norm", max_norm)
            .minimum("maximum_derivative_entry_norm", f64::MIN_POSITIVE)
            .input_checksum("bem:curved-quad4-separated-pair:ka-0.77")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-quad4-pair-material-derivative-v1",
            )
            .dimension("3-D curved Quad4 separated Müller pair, linear two-trace derivative")
            .seed("bem-galerkin-quad4-pair-material-derivative:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:curved-quad4-separated-pair:ka-0.77")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-quad4-pair-material-derivative-v1",
            )
            .dimension("3-D curved Quad4 separated Müller pair, linear two-trace derivative")
            .seed("bem-galerkin-quad4-pair-material-derivative:cpu:v1"),
    };
    println!(
        "{}",
        result
            .elapsed_ms(started.elapsed().as_millis() as u64)
            .to_json_line()
            .expect("valid QA result should serialize")
    );
}
