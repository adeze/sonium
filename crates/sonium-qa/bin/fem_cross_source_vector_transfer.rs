//! CPU-only cross-source FEM vector-transfer contract.

use sonium_fem::mesh::{
    ElementType, Mesh, MeshTopology, Point, transfer_hcurl_order1_state_checked,
    transfer_hdiv_rt0_state_checked, transfer_hdiv_rt1_state_checked,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.cross-source-vector-transfer.v1";

fn split_source_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ] {
        mesh.add_node(Point::new_3d(x, y, 0.0));
    }
    for nodes in [[0, 1, 3], [0, 3, 2], [1, 4, 5], [1, 5, 3]] {
        mesh.add_element(ElementType::Triangle, nodes.to_vec());
    }
    mesh
}

fn crossing_target_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75)] {
        mesh.add_node(Point::new_3d(x, y, 0.0));
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh
}

fn run() -> Result<(usize, f64, usize, usize, usize), String> {
    let source = split_source_mesh();
    let target = crossing_target_mesh();

    let constant = [1.25, -0.75, 2.0];
    let source_topology = MeshTopology::build(&source);
    let target_topology = MeshTopology::build(&target);
    let hcurl_source = source_topology
        .edges
        .iter()
        .map(|&(a, b)| {
            let p0 = source.nodes[a];
            let p1 = source.nodes[b];
            constant[0] * (p1.x - p0.x) + constant[1] * (p1.y - p0.y) + constant[2] * (p1.z - p0.z)
        })
        .collect::<Vec<_>>();
    let hcurl_target = transfer_hcurl_order1_state_checked(
        &source,
        &target,
        &HcurlSpace::new(),
        &HcurlSpace::new(),
        &hcurl_source,
    )
    .map_err(|error| error.to_string())?;
    let max_hcurl_error = target_topology
        .edges
        .iter()
        .enumerate()
        .map(|(index, &(a, b))| {
            let p0 = target.nodes[a];
            let p1 = target.nodes[b];
            let expected = constant[0] * (p1.x - p0.x)
                + constant[1] * (p1.y - p0.y)
                + constant[2] * (p1.z - p0.z);
            (hcurl_target[index] - expected).abs()
        })
        .fold(0.0, f64::max);

    let rt0 = HdivSpace::with_order(HdivOrder::Rt0);
    let rt0_source = vec![0.0; rt0.num_global_dofs(&source)];
    let rt0_target = transfer_hdiv_rt0_state_checked(&source, &target, &rt0, &rt0, &rt0_source)
        .map_err(|error| error.to_string())?;
    if rt0_target.iter().any(|value| !value.is_finite()) {
        return Err("RT0 cross-source transfer produced a non-finite value".into());
    }

    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let rt1_source = (0..rt1.num_global_dofs(&source))
        .map(|index| 0.01 * (index as f64 + 1.0))
        .collect::<Vec<_>>();
    let rt1_target = transfer_hdiv_rt1_state_checked(&source, &target, &rt1, &rt1, &rt1_source)
        .map_err(|error| error.to_string())?;
    if rt1_target.iter().any(|value| !value.is_finite()) {
        return Err("RT1 cross-source transfer produced a non-finite value".into());
    }

    let mut outside = crossing_target_mesh();
    outside.nodes[1].x = 1.25;
    let outside_error =
        transfer_hdiv_rt1_state_checked(&source, &outside, &rt1, &rt1, &rt1_source).is_err();
    if !outside_error {
        return Err("outside target admission did not fail closed".into());
    }

    Ok((
        target_topology.edges.len(),
        max_hcurl_error,
        rt0_target.len(),
        rt1_target.len(),
        usize::from(outside_error),
    ))
}

fn main() {
    let result = match run() {
        Ok((target_edges, max_hcurl_error, rt0_dofs, rt1_dofs, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("target_edges", target_edges as f64)
                .minimum("target_edges", 1.0)
                .metric("max_hcurl_constant_field_error", max_hcurl_error)
                .maximum("max_hcurl_constant_field_error", 2.0e-10)
                .metric("rt0_target_dofs", rt0_dofs as f64)
                .minimum("rt0_target_dofs", 1.0)
                .metric("rt1_target_dofs", rt1_dofs as f64)
                .minimum("rt1_target_dofs", 1.0)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 1.0)
                .dimension("2-D affine triangular target crossing a conforming source partition")
                .formulation(
                    "per-quadrature-point source ownership for H(curl) and H(div) Piola transfer",
                )
                .input_checksum("two-triangle-strip-source-to-crossing-target:v1")
                .reference_checksum("analytic-constant-hcurl-and-zero-hdiv-cross-source:v1")
                .seed("fem-cross-source-vector-transfer:cpu:v1")
        }
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
