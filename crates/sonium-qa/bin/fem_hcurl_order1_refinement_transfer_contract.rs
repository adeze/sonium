//! CPU-only affine H(curl) order-1 refinement state-transfer contract.

use sonium_fem::mesh::{InterpolationError, transfer_hcurl_order1_state_checked};
use sonium_fem::mesh::{MeshTopology, unit_square_quads, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order1-refinement-transfer.v1";

fn main() {
    let result = match run() {
        Ok((edge_count, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("transferred_target_edges", edge_count as f64)
            .minimum("transferred_target_edges", 1.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 3.0)
            .dimension("affine 2-D triangular order-1 Nedelec refinement transfer")
            .formulation("three-point edge-moment transfer with checked ownership admission")
            .input_checksum("unit-square-triangles-1-to-2:affine-nedelec-field-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hcurl-order1-refinement-transfer-v1",
            )
            .seed("fem-hcurl-order1-refinement-transfer:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}

fn edge_moment(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x))
}

fn run() -> Result<(usize, usize), String> {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let source_topology = MeshTopology::build(&source);
    let source_values: Vec<_> = source_topology
        .edges
        .iter()
        .map(|&edge| edge_moment(&source, edge))
        .collect();
    let transferred = transfer_hcurl_order1_state_checked(
        &source,
        &target,
        &HcurlSpace::new(),
        &HcurlSpace::new(),
        &source_values,
    )
    .map_err(|error| error.to_string())?;
    let target_topology = MeshTopology::build(&target);
    for (edge_index, &edge) in target_topology.edges.iter().enumerate() {
        if (transferred[edge_index] - edge_moment(&target, edge)).abs() >= 1.0e-10 {
            return Err(format!("edge {edge_index} transfer mismatch"));
        }
    }

    let mut rejection_checks = 0;
    let source_dofs = HcurlSpace::new().num_global_dofs(&source);
    if !matches!(
        transfer_hcurl_order1_state_checked(
            &source,
            &target,
            &HcurlSpace::order2(),
            &HcurlSpace::new(),
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        return Err("order-2 source space was accepted".into());
    }
    rejection_checks += 1;
    if !matches!(
        transfer_hcurl_order1_state_checked(
            &source,
            &target,
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &vec![0.0; source_dofs - 1],
        ),
        Err(InterpolationError::SourceValueCount { .. })
    ) {
        return Err("short source state was accepted".into());
    }
    rejection_checks += 1;
    if !matches!(
        transfer_hcurl_order1_state_checked(
            &source,
            &unit_square_quads(1),
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &source_values,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        return Err("non-simplex target geometry was accepted".into());
    }
    rejection_checks += 1;

    Ok((target_topology.edges.len(), rejection_checks))
}
