//! Bounded CPU evidence for affine-Tri3 Galerkin double-layer traces.

use std::time::Instant;

use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::{AdjacencyType, Element, detect_adjacency};
use sonium_bem::{
    BasisOrder, GalerkinDoubleLayerOperators, PhysicsParams,
    assemble_galerkin_double_layer_with_basis,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.galerkin-double-layer-principal-value.v1";

fn panel_pair(
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
    elements[0].dof_addresses = vec![0, 1, 2];
    elements[1].dof_addresses = vec![3, 4, 5];
    assemble_galerkin_double_layer_with_basis(
        &elements,
        nodes,
        physics,
        BasisOrder::Linear,
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
            return Err("Galerkin double-layer matrix contains non-finite entry".to_string());
        }
        maximum = maximum.max(value.norm());
    }
    Ok(maximum)
}

fn pair_difference(
    coarse: &GalerkinDoubleLayerOperators,
    fine: &GalerkinDoubleLayerOperators,
) -> f64 {
    (0..fine.double_layer.nrows())
        .flat_map(|row| (0..fine.double_layer.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| {
            (fine.double_layer[[row, column]] - coarse.double_layer[[row, column]])
                .norm()
                .max(
                    (fine.adjoint_double_layer[[row, column]]
                        - coarse.adjoint_double_layer[[row, column]])
                    .norm(),
                )
        })
        .fold(0.0_f64, f64::max)
}

fn transpose_error(operators: &GalerkinDoubleLayerOperators) -> f64 {
    (0..operators.double_layer.nrows())
        .flat_map(|row| (0..operators.double_layer.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.adjoint_double_layer[[row, column]] - operators.double_layer[[column, row]])
                .norm()
        })
        .fold(0.0_f64, f64::max)
}

fn run() -> Result<(usize, f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let edge_pair = panel_pair(&mesh.elements, |adjacency| {
        matches!(adjacency, AdjacencyType::CommonEdge { .. })
    })?;
    let vertex_pair = panel_pair(&mesh.elements, |adjacency| {
        matches!(adjacency, AdjacencyType::CommonVertex { .. })
    })?;
    let separated_pair = panel_pair(&mesh.elements, |adjacency| {
        matches!(adjacency, AdjacencyType::Separated)
    })?;
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;

    let mut maximum_entry = 0.0_f64;
    let mut maximum_convergence_error = 0.0_f64;
    for pair in [&edge_pair, &vertex_pair] {
        let coarse = assemble_pair(pair, &mesh.nodes, &physics, 4)?;
        let fine = assemble_pair(pair, &mesh.nodes, &physics, 8)?;
        maximum_entry = maximum_entry.max(finite_maximum(&fine)?);
        maximum_convergence_error = maximum_convergence_error.max(pair_difference(&coarse, &fine));
    }

    let separated = assemble_pair(&separated_pair, &mesh.nodes, &physics, 8)?;
    maximum_entry = maximum_entry.max(finite_maximum(&separated)?);
    let separated_transpose_error = transpose_error(&separated);
    if maximum_convergence_error > 5.0e-4 {
        return Err(format!(
            "touching-pair chart convergence error {maximum_convergence_error:.3e} exceeds 5e-4"
        ));
    }
    if separated_transpose_error > 2.0e-8 {
        return Err(format!(
            "separated K/K' reciprocity error {separated_transpose_error:.3e} exceeds 2e-8"
        ));
    }
    Ok((
        6,
        maximum_entry,
        maximum_convergence_error,
        separated_transpose_error,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, maximum_entry, maximum_convergence_error, separated_transpose_error)) => {
            QaResult::passed(CASE_ID)
                .input_checksum("bem:icosphere:subdivision-0:edge-vertex-separated-linear-tri3")
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-galerkin-double-layer-principal-value-v1",
                )
                .dimension(format!(
                    "3-D CPU {dimension}-DOF touching-chart and separated reciprocity"
                ))
                .metric("maximum_entry_norm", maximum_entry)
                .metric(
                    "touching_chart_convergence_error",
                    maximum_convergence_error,
                )
                .metric("separated_transpose_error", separated_transpose_error)
                .maximum("touching_chart_convergence_error", 5.0e-4)
                .maximum("separated_transpose_error", 2.0e-8)
                .seed("bem-galerkin-double-layer-principal-value:cpu:v2")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
        }
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
