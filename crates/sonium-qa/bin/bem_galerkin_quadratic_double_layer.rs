//! Bounded CPU evidence for quadratic affine-Tri3 Galerkin `K`/`K'` traces.

use std::time::Instant;

use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::{AdjacencyType, Element, detect_adjacency};
use sonium_bem::{
    BasisOrder, GalerkinDoubleLayerOperators, PhysicsParams,
    assemble_galerkin_double_layer_with_basis,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.galerkin-quadratic-double-layer.v1";

fn pair_with_relation(
    elements: &[Element],
    relation: impl Fn(AdjacencyType) -> bool,
) -> Result<(Element, Element), String> {
    let first = elements
        .first()
        .cloned()
        .ok_or_else(|| "icosphere has no panels".to_string())?;
    let second = elements
        .iter()
        .skip(1)
        .find(|element| relation(detect_adjacency(&first, element)))
        .cloned()
        .ok_or_else(|| "icosphere is missing the requested adjacency pair".to_string())?;
    Ok((first, second))
}

fn assemble_pair(
    pair: &(Element, Element),
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gauss_order: usize,
) -> Result<GalerkinDoubleLayerOperators, String> {
    let mut elements = vec![pair.0.clone(), pair.1.clone()];
    for (panel, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = (0..6).map(|local| 6 * panel + local).collect();
    }
    assemble_galerkin_double_layer_with_basis(
        &elements,
        nodes,
        physics,
        BasisOrder::Quadratic,
        gauss_order,
    )
    .map_err(|error| error.to_string())
}

fn finite_maximum(operators: &GalerkinDoubleLayerOperators) -> Result<f64, String> {
    let mut maximum = 0.0_f64;
    for value in operators
        .double_layer
        .as_slice()
        .iter()
        .chain(operators.adjoint_double_layer.as_slice())
    {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err("quadratic double-layer entry is non-finite".to_string());
        }
        maximum = maximum.max(value.norm());
    }
    Ok(maximum)
}

fn run() -> Result<(f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let first = mesh
        .elements
        .first()
        .cloned()
        .ok_or_else(|| "icosphere has no panels".to_string())?;
    let mut self_element = first.clone();
    self_element.dof_addresses = (0..6).collect();
    let self_operators = assemble_galerkin_double_layer_with_basis(
        &[self_element],
        &mesh.nodes,
        &physics,
        BasisOrder::Quadratic,
        8,
    )
    .map_err(|error| error.to_string())?;
    let self_zero = self_operators
        .double_layer
        .as_slice()
        .iter()
        .chain(self_operators.adjoint_double_layer.as_slice())
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    let edge = pair_with_relation(&mesh.elements, |adjacency| {
        matches!(adjacency, AdjacencyType::CommonEdge { .. })
    })?;
    let vertex = pair_with_relation(&mesh.elements, |adjacency| {
        matches!(adjacency, AdjacencyType::CommonVertex { .. })
    })?;
    let separated = pair_with_relation(&mesh.elements, |adjacency| {
        matches!(adjacency, AdjacencyType::Separated)
    })?;
    let edge_operators = assemble_pair(&edge, &mesh.nodes, &physics, 8)?;
    let vertex_operators = assemble_pair(&vertex, &mesh.nodes, &physics, 6)?;
    let separated_operators = assemble_pair(&separated, &mesh.nodes, &physics, 6)?;
    let maximum_entry = finite_maximum(&edge_operators)?
        .max(finite_maximum(&vertex_operators)?)
        .max(finite_maximum(&separated_operators)?);
    let reciprocity_error = (0..12)
        .flat_map(|row| (0..12).map(move |column| (row, column)))
        .map(|(row, column)| {
            (separated_operators.adjoint_double_layer[[row, column]]
                - separated_operators.double_layer[[column, row]])
            .norm()
        })
        .fold(0.0_f64, f64::max);
    if self_zero > 1.0e-14 {
        return Err(format!(
            "quadratic flat self-panel norm {self_zero:.3e} exceeds 1e-14"
        ));
    }
    if reciprocity_error > 2.0e-7 {
        return Err(format!(
            "quadratic separated K/K' reciprocity {reciprocity_error:.3e} exceeds 2e-7"
        ));
    }
    Ok((self_zero, maximum_entry, reciprocity_error))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((self_zero, maximum_entry, reciprocity_error)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:icosphere:subdivision-0:quadratic-tri3-galerkin-double-layer")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-quadratic-double-layer-v1",
            )
            .dimension("3-D CPU 12-DOF quadratic Tri3 self/touching/separated traces")
            .metric("self_panel_max_norm", self_zero)
            .metric("maximum_entry_norm", maximum_entry)
            .metric("separated_reciprocity_error", reciprocity_error)
            .maximum("self_panel_max_norm", 1.0e-14)
            .maximum("separated_reciprocity_error", 2.0e-7)
            .seed("bem-galerkin-quadratic-double-layer:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
