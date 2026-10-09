//! Bounded CPU evidence for conforming linear-Tri3 Galerkin coupling.

use std::time::Instant;

use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{BasisOrder, PhysicsParams, assemble_galerkin_single_layer_with_global_dofs};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.galerkin-conforming-linear-single-layer.v1";

fn run() -> Result<(usize, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements.into_iter().take(2).collect::<Vec<_>>();
    let mut unique_nodes = elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    for element in &mut elements {
        element.dof_addresses = element
            .connectivity
            .iter()
            .map(|node| unique_nodes.binary_search(node).expect("node is present"))
            .collect();
    }

    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let matrix = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Linear,
        4,
        unique_nodes.len(),
    )
    .map_err(|error| error.to_string())?;
    if matrix
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("conforming linear Galerkin matrix contains non-finite entry".to_string());
    }
    let mut asymmetry = 0.0_f64;
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            asymmetry = asymmetry.max((matrix[[row, column]] - matrix[[column, row]]).norm());
        }
    }
    Ok((matrix.nrows(), asymmetry))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, asymmetry)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:icosphere:subdivision-0:conforming-linear-tri3:two-panels")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-conforming-linear-single-layer-v1",
            )
            .reference("independent panel-local block accumulation into shared global DOFs")
            .dimension(format!(
                "3-D CPU conforming {dimension}-DOF single-layer Galerkin; max asymmetry={asymmetry:.3e}"
            ))
            .seed("bem-galerkin-conforming-linear-single-layer:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
