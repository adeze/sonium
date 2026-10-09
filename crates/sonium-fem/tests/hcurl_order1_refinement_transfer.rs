use sonium_fem::mesh::{InterpolationError, transfer_hcurl_order1_state_checked};
use sonium_fem::mesh::{
    MeshTopology, unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn edge_moment_2d(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x))
}

fn edge_moment_3d(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let dz = p1.z - p0.z;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x)) + 3.0 * dz
}

#[test]
fn order1_hcurl_transfer_preserves_nedelec_edge_moments_on_2d_refinement() {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let topology = MeshTopology::build(&source);
    let source_values: Vec<_> = topology
        .edges
        .iter()
        .map(|&edge| edge_moment_2d(&source, edge))
        .collect();

    let transferred = transfer_hcurl_order1_state_checked(
        &source,
        &target,
        &HcurlSpace::new(),
        &HcurlSpace::new(),
        &source_values,
    )
    .unwrap();
    let target_topology = MeshTopology::build(&target);
    assert_eq!(transferred.len(), target_topology.edges.len());
    for (edge_index, &edge) in target_topology.edges.iter().enumerate() {
        assert!((transferred[edge_index] - edge_moment_2d(&target, edge)).abs() < 1.0e-10);
    }
}

#[test]
fn order1_hcurl_transfer_preserves_nedelec_edge_moments_on_3d_mesh() {
    let source = unit_cube_tetrahedra(1);
    let target = unit_cube_tetrahedra(1);
    let topology = MeshTopology::build(&source);
    let source_values: Vec<_> = topology
        .edges
        .iter()
        .map(|&edge| edge_moment_3d(&source, edge))
        .collect();
    let transferred = transfer_hcurl_order1_state_checked(
        &source,
        &target,
        &HcurlSpace::new(),
        &HcurlSpace::new(),
        &source_values,
    )
    .unwrap();
    for (edge_index, &edge) in topology.edges.iter().enumerate() {
        assert!((transferred[edge_index] - edge_moment_3d(&target, edge)).abs() < 1.0e-10);
    }
}

#[test]
fn order1_hcurl_transfer_rejects_orders_geometry_dimensions_and_state_errors() {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let source_dofs = HcurlSpace::new().num_global_dofs(&source);
    assert!(matches!(
        transfer_hcurl_order1_state_checked(
            &source,
            &target,
            &HcurlSpace::order2(),
            &HcurlSpace::new(),
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));
    assert!(matches!(
        transfer_hcurl_order1_state_checked(
            &source,
            &target,
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &vec![0.0; source_dofs - 1],
        ),
        Err(InterpolationError::SourceValueCount { .. })
    ));
    let mut non_finite = vec![0.0; source_dofs];
    non_finite[0] = f64::NAN;
    assert!(matches!(
        transfer_hcurl_order1_state_checked(
            &source,
            &target,
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &non_finite,
        ),
        Err(InterpolationError::NonFiniteSource { .. })
    ));
    assert!(matches!(
        transfer_hcurl_order1_state_checked(
            &source,
            &unit_square_quads(1),
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));
}
