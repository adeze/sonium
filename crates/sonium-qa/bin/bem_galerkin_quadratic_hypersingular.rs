//! Bounded CPU evidence for conforming affine-Tri3 quadratic Galerkin W.

use std::time::Instant;

use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{BasisOrder, PhysicsParams, assemble_galerkin_hypersingular_with_global_dofs};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.galerkin-quadratic-hypersingular-weak-form.v1";

fn run() -> Result<(usize, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    let mut unique_nodes = elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    let mut unique_edges = elements
        .iter()
        .flat_map(|element| {
            [
                (element.connectivity[0], element.connectivity[1]),
                (element.connectivity[1], element.connectivity[2]),
                (element.connectivity[2], element.connectivity[0]),
            ]
            .into_iter()
            .map(|(first, second)| {
                if first < second {
                    (first, second)
                } else {
                    (second, first)
                }
            })
        })
        .collect::<Vec<_>>();
    unique_edges.sort_unstable();
    unique_edges.dedup();
    let edge_offset = unique_nodes.len();
    for element in &mut elements {
        let vertices = element
            .connectivity
            .iter()
            .map(|node| unique_nodes.binary_search(node).expect("node is present"));
        let edges = [
            (element.connectivity[0], element.connectivity[1]),
            (element.connectivity[1], element.connectivity[2]),
            (element.connectivity[2], element.connectivity[0]),
        ]
        .into_iter()
        .map(|(first, second)| {
            let key = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            edge_offset + unique_edges.binary_search(&key).expect("edge is present")
        });
        element.dof_addresses = vertices.chain(edges).collect();
    }
    let global_dof_count = edge_offset + unique_edges.len();
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let operators = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Quadratic,
        3,
        global_dof_count,
    )
    .map_err(|error| error.to_string())?;
    if operators
        .hypersingular
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(
            "quadratic Galerkin hypersingular matrix contains a non-finite entry".to_string(),
        );
    }
    let maximum_asymmetry = (0..global_dof_count)
        .flat_map(|row| (0..global_dof_count).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.hypersingular[[row, column]] - operators.hypersingular[[column, row]]).norm()
        })
        .fold(0.0_f64, f64::max);
    if maximum_asymmetry > 1.0e-12 {
        return Err(format!(
            "quadratic Galerkin hypersingular reciprocity error {maximum_asymmetry:.3e} exceeds 1e-12"
        ));
    }
    Ok((global_dof_count, maximum_asymmetry))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, maximum_asymmetry)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:icosphere:subdivision-0:closed-conforming-quadratic-tri3")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-quadratic-hypersingular-weak-form-v1",
            )
            .dimension(format!(
                "3-D CPU {dimension}-DOF closed conforming P2 Tri3 weak W"
            ))
            .metric("maximum_asymmetry", maximum_asymmetry)
            .maximum("maximum_asymmetry", 1.0e-12)
            .seed("bem-galerkin-quadratic-hypersingular-weak-form:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
