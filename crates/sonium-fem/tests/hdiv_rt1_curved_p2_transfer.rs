use sonium_fem::mesh::{
    InterpolationError, MeshTopology, Point, transfer_hdiv_rt1_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn curved_triangle_mesh() -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += 0.08;
    mesh
}

fn state(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.17 + 0.13 * index as f64)
        .collect()
}

#[test]
fn rt1_transfer_reproduces_complete_state_on_affine_triangles() {
    let mesh = unit_square_triangles(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values = state(&space, &mesh);
    let transferred = transfer_hdiv_rt1_state_checked(&mesh, &mesh, &space, &space, &source_values)
        .expect("affine RT1 transfer must succeed");
    for (index, (actual, expected)) in transferred.iter().zip(source_values).enumerate() {
        assert!(
            (actual - expected).abs() < 2.0e-8,
            "dof {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn rt1_transfer_reproduces_complete_state_on_curved_p2_triangles() {
    let mesh = curved_triangle_mesh();
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values = state(&space, &mesh);

    let transferred = transfer_hdiv_rt1_state_checked(&mesh, &mesh, &space, &space, &source_values)
        .expect("curved P2 RT1 transfer must succeed");

    assert_eq!(transferred.len(), source_values.len());
    for (index, (actual, expected)) in transferred.iter().zip(source_values).enumerate() {
        assert!(
            (actual - expected).abs() < 2.0e-8,
            "dof {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn rt1_transfer_integrates_curved_target_edge_and_preserves_finiteness() {
    let source = unit_square_triangles(1);
    let target = curved_triangle_mesh();
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values = state(&source_space, &source);

    let transferred = transfer_hdiv_rt1_state_checked(
        &source,
        &target,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("affine-to-curved RT1 transfer must succeed");

    assert_eq!(transferred.len(), target_space.num_global_dofs(&target));
    assert!(transferred.iter().all(|value| value.is_finite()));

    let topology = MeshTopology::build(&target);
    assert_eq!(topology.edges.len(), 5);
    assert_eq!(
        transferred.len(),
        topology.edges.len() * 2 + target.num_elements() * 2
    );
}

#[test]
fn rt1_curved_transfer_rejects_nonfinite_geometry_and_state() {
    let mut nonfinite = curved_triangle_mesh();
    let midpoint = nonfinite.elements[0].nodes[3];
    nonfinite.nodes[midpoint] = Point::new_3d(f64::NAN, 0.0, 0.0);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let values = state(&space, &curved_triangle_mesh());

    assert!(matches!(
        transfer_hdiv_rt1_state_checked(&nonfinite, &nonfinite, &space, &space, &values),
        Err(InterpolationError::InvalidMesh { .. })
    ));

    let mesh = curved_triangle_mesh();
    let mut nonfinite_values = state(&space, &mesh);
    nonfinite_values[0] = f64::NAN;
    assert!(matches!(
        transfer_hdiv_rt1_state_checked(&mesh, &mesh, &space, &space, &nonfinite_values,),
        Err(InterpolationError::NonFiniteSource { .. })
    ));
}
