//! Bounded CPU QA for self-pair Quad4 Galerkin Müller charts.

use num_complex::Complex64;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, PhysicsParams, assemble_galerkin_muller_transmission_quad4_pair,
    assemble_galerkin_muller_transmission_with_global_dofs,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-quad4-transmission-self.v1";

fn element(dofs: Vec<usize>) -> Element {
    Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.5, 0.5, 0.0],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: dofs,
    }
}

fn finite_nonzero(values: &[Complex64]) -> bool {
    values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
        && values.iter().any(|value| value.norm() > 1.0e-12)
}

fn run() -> Result<(usize, usize), String> {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.15],
        [0.0, 1.0, 0.0],
    ];
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false)
        .map_err(|error| format!("invalid exterior physics: {error}"))?;
    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, false)
        .map_err(|error| format!("invalid interior physics: {error}"))?;

    let linear_element = element((0..4).collect());
    let linear = assemble_galerkin_muller_transmission_quad4_pair(
        &linear_element,
        &linear_element,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .map_err(|error| format!("linear self-pair assembly failed: {error}"))?;
    if linear.shape() != (8, 8) || !finite_nonzero(linear.as_slice()) {
        return Err("linear self-pair block was not finite and nonzero 8x8".into());
    }

    let quadratic_element = element((0..9).collect());
    let quadratic = assemble_galerkin_muller_transmission_quad4_pair(
        &quadratic_element,
        &quadratic_element,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| format!("quadratic self-pair assembly failed: {error}"))?;
    if quadratic.shape() != (18, 18) || !finite_nonzero(quadratic.as_slice()) {
        return Err("quadratic self-pair block was not finite and nonzero 18x18".into());
    }

    let global = assemble_galerkin_muller_transmission_with_global_dofs(
        &[linear_element],
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
        4,
    )
    .map_err(|error| format!("global self-pair assembly failed: {error}"))?;
    if global.matrix().shape() != (8, 8) || !finite_nonzero(global.matrix().as_slice()) {
        return Err("global self-pair matrix was not finite and nonzero 8x8".into());
    }

    Ok((linear.nrows(), quadratic.nrows()))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((linear_dimension, quadratic_dimension)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("linear_matrix_dimension", linear_dimension as f64)
            .metric("quadratic_matrix_dimension", quadratic_dimension as f64)
            .dimension("3-D curved Quad4 self-pair two-trace Müller")
            .formulation("mapped self chart plus global self-block cross-check")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-quad4-transmission-self-v1",
            )
            .seed("bem-galerkin-quad4-transmission-self:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D curved Quad4 self-pair two-trace Müller")
            .seed("bem-galerkin-quad4-transmission-self:cpu:v1"),
    };
    let result = result.elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "{}",
        result
            .to_json_line()
            .expect("serialize self-pair QA_RESULT")
    );
}
