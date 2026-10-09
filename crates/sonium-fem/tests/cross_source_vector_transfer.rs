use sonium_fem::mesh::{
    ElementType, Mesh, MeshTopology, Point, transfer_hcurl_order1_state_checked,
    transfer_hdiv_rt0_state_checked, transfer_hdiv_rt1_state_checked,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};

/// Two conforming triangular strips separated by x = 1/2.  The target
/// triangle below deliberately crosses that interior source boundary.
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

#[test]
fn hcurl_constant_field_crosses_source_edge_boundary() {
    let source = split_source_mesh();
    let target = crossing_target_mesh();
    let source_topology = MeshTopology::build(&source);
    let target_topology = MeshTopology::build(&target);
    let constant = [1.25, -0.75, 2.0];
    let source_values = source_topology
        .edges
        .iter()
        .map(|&(a, b)| {
            let p0 = source.nodes[a];
            let p1 = source.nodes[b];
            constant[0] * (p1.x - p0.x) + constant[1] * (p1.y - p0.y) + constant[2] * (p1.z - p0.z)
        })
        .collect::<Vec<_>>();

    let transferred = transfer_hcurl_order1_state_checked(
        &source,
        &target,
        &HcurlSpace::new(),
        &HcurlSpace::new(),
        &source_values,
    )
    .expect("a target edge crossing a conforming source boundary is valid");

    for (index, &(a, b)) in target_topology.edges.iter().enumerate() {
        let p0 = target.nodes[a];
        let p1 = target.nodes[b];
        let expected =
            constant[0] * (p1.x - p0.x) + constant[1] * (p1.y - p0.y) + constant[2] * (p1.z - p0.z);
        assert!((transferred[index] - expected).abs() < 2.0e-10);
    }
}

#[test]
fn rt0_cross_source_edge_boundary_is_checked_and_finite() {
    let source = split_source_mesh();
    let target = crossing_target_mesh();
    let space = HdivSpace::with_order(HdivOrder::Rt0);
    let source_values = vec![0.0; space.num_global_dofs(&source)];

    let transferred =
        transfer_hdiv_rt0_state_checked(&source, &target, &space, &space, &source_values)
            .expect("RT0 transfer should resolve source ownership per quadrature point");

    assert_eq!(transferred.len(), space.num_global_dofs(&target));
    assert!(transferred.iter().all(|value| value.is_finite()));
    assert!(transferred.iter().all(|value| value.abs() < 1.0e-12));
}

#[test]
fn rt1_cross_source_cell_boundary_is_checked_and_finite() {
    let source = split_source_mesh();
    let target = crossing_target_mesh();
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values = (0..space.num_global_dofs(&source))
        .map(|index| 0.01 * (index as f64 + 1.0))
        .collect::<Vec<_>>();

    let transferred =
        transfer_hdiv_rt1_state_checked(&source, &target, &space, &space, &source_values)
            .expect("RT1 cell moments should resolve source ownership per quadrature point");

    assert_eq!(transferred.len(), space.num_global_dofs(&target));
    assert!(transferred.iter().all(|value| value.is_finite()));
}

#[test]
fn cross_source_transfer_still_rejects_target_outside_domain() {
    let source = split_source_mesh();
    let mut target = crossing_target_mesh();
    target.nodes[1].x = 1.25;
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let values = vec![0.0; space.num_global_dofs(&source)];

    let error = transfer_hdiv_rt1_state_checked(&source, &target, &space, &space, &values)
        .expect_err("a quadrature point outside the source domain must fail closed");
    assert!(matches!(
        error,
        sonium_fem::mesh::InterpolationError::TargetElementOutside { .. }
    ));
}
