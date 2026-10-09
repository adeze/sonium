//! CPU-only curved P2 order-1 H(curl) state-transfer contract.

use sonium_fem::mesh::{InterpolationError, transfer_hcurl_order1_state_checked};
use sonium_fem::mesh::{MeshTopology, unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order1-curved-transfer.v1";

fn edge_moment(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let dz = p1.z - p0.z;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x)) + 3.0 * dz
}

fn run() -> Result<(usize, usize, usize), String> {
    let mut curved_2d = unit_square_triangles(1).to_p2();
    let midpoint_2d = curved_2d.elements[0].nodes[3];
    curved_2d.nodes[midpoint_2d].y += 0.08;
    let space = HcurlSpace::new();
    let topology_2d = MeshTopology::build(&curved_2d);
    let values_2d: Vec<_> = topology_2d
        .edges
        .iter()
        .enumerate()
        .map(|(edge, &nodes)| 0.17 + 0.23 * edge as f64 + edge_moment(&curved_2d, nodes))
        .collect();
    let transferred_2d =
        transfer_hcurl_order1_state_checked(&curved_2d, &curved_2d, &space, &space, &values_2d)
            .map_err(|error| error.to_string())?;
    for (actual, expected) in transferred_2d.iter().zip(&values_2d) {
        if (actual - expected).abs() > 2.0e-9 {
            return Err(format!(
                "curved 2-D recovery mismatch: {actual} vs {expected}"
            ));
        }
    }

    let mut curved_3d = unit_cube_tetrahedra(1).to_p2();
    let midpoint_3d = curved_3d.elements[0].nodes[4];
    curved_3d.nodes[midpoint_3d].z += 0.06;
    let topology_3d = MeshTopology::build(&curved_3d);
    let values_3d: Vec<_> = topology_3d
        .edges
        .iter()
        .enumerate()
        .map(|(edge, &nodes)| -0.31 + 0.11 * edge as f64 + edge_moment(&curved_3d, nodes))
        .collect();
    let transferred_3d =
        transfer_hcurl_order1_state_checked(&curved_3d, &curved_3d, &space, &space, &values_3d)
            .map_err(|error| error.to_string())?;
    for (actual, expected) in transferred_3d.iter().zip(&values_3d) {
        if (actual - expected).abs() > 3.0e-9 {
            return Err(format!(
                "curved 3-D recovery mismatch: {actual} vs {expected}"
            ));
        }
    }

    let mut malformed = unit_square_triangles(1);
    malformed.elements[0].nodes.push(malformed.nodes.len());
    malformed.nodes.push((0.25, 0.25).into());
    // Do not ask the infallible topology-backed DOF counter to traverse an
    // intentionally malformed element; the checked transfer owns rejection.
    let rejection_values = vec![0.0; values_2d.len()];
    let mut rejection_checks = 0;
    if matches!(
        transfer_hcurl_order1_state_checked(
            &malformed,
            &malformed,
            &space,
            &space,
            &rejection_values,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        rejection_checks += 1;
    }
    if matches!(
        transfer_hcurl_order1_state_checked(
            &curved_2d,
            &curved_2d,
            &HcurlSpace::order2(),
            &space,
            &values_2d,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        rejection_checks += 1;
    }
    if matches!(
        transfer_hcurl_order1_state_checked(
            &curved_2d,
            &curved_2d,
            &space,
            &space,
            &vec![f64::NAN; values_2d.len()],
        ),
        Err(InterpolationError::NonFiniteSource { .. })
    ) {
        rejection_checks += 1;
    }
    if rejection_checks != 3 {
        return Err(format!(
            "expected three rejection checks, got {rejection_checks}"
        ));
    }

    Ok((transferred_2d.len(), transferred_3d.len(), rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((curved_2d_edges, curved_3d_edges, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("curved_2d_edges", curved_2d_edges as f64)
            .minimum("curved_2d_edges", 1.0)
            .metric("curved_3d_edges", curved_3d_edges as f64)
            .minimum("curved_3d_edges", 1.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 3.0)
            .dimension("curved P2 triangles and tetrahedra")
            .formulation("order-1 Nedelec state transfer over quadratic isoparametric edges")
            .input_checksum("unit-square/unit-cube-p2-curved-edge-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hcurl-order1-curved-transfer-v1",
            )
            .seed("fem-hcurl-order1-curved-transfer:cpu:v1"),
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
