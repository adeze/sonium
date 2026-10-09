use sonium_fem::mesh::{
    ElementType, InterpolationError, Point, transfer_hcurl_order2_triangle_state_checked,
    unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn curved_triangle_mesh() -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += 0.08;
    mesh
}

fn state(space: &HcurlSpace, mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.21 + 0.07 * index as f64 - 0.002 * (index * index) as f64)
        .collect()
}

fn max_abs_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

#[test]
fn order2_triangle_transfer_reproduces_affine_state() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::order2();
    let source_values = state(&space, &mesh);
    let transferred =
        transfer_hcurl_order2_triangle_state_checked(&mesh, &mesh, &space, &space, &source_values)
            .expect("affine order-2 H(curl) transfer must succeed");

    assert_eq!(transferred.len(), source_values.len());
    assert!(max_abs_error(&transferred, &source_values) < 2.0e-11);
}

#[test]
fn order2_triangle_transfer_reproduces_curved_p2_state() {
    let mesh = curved_triangle_mesh();
    let space = HcurlSpace::order2();
    let source_values = state(&space, &mesh);
    let transferred =
        transfer_hcurl_order2_triangle_state_checked(&mesh, &mesh, &space, &space, &source_values)
            .expect("curved P2 order-2 H(curl) transfer must succeed");

    assert_eq!(transferred.len(), source_values.len());
    assert!(
        max_abs_error(&transferred, &source_values) < 2.0e-9,
        "curved order-2 transfer max error: {:e}",
        max_abs_error(&transferred, &source_values)
    );
}

#[test]
fn order2_triangle_transfer_accepts_affine_source_and_curved_target() {
    let source = unit_square_triangles(1);
    let target = curved_triangle_mesh();
    let source_space = HcurlSpace::order2();
    let target_space = HcurlSpace::order2();
    let source_values = state(&source_space, &source);
    let transferred = transfer_hcurl_order2_triangle_state_checked(
        &source,
        &target,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("affine-to-curved order-2 transfer must succeed");

    assert_eq!(transferred.len(), target_space.num_global_dofs(&target));
    assert!(transferred.iter().all(|value| value.is_finite()));
}

#[test]
fn order2_triangle_transfer_rejects_invalid_inputs() {
    let curved = curved_triangle_mesh();
    let space = HcurlSpace::order2();
    let values = state(&space, &curved);

    let mut nonfinite_geometry = curved.clone();
    nonfinite_geometry.nodes[nonfinite_geometry.elements[0].nodes[3]] =
        Point::new_3d(f64::NAN, 0.0, 0.0);
    assert!(matches!(
        transfer_hcurl_order2_triangle_state_checked(
            &nonfinite_geometry,
            &curved,
            &space,
            &space,
            &values,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));

    let order1 = HcurlSpace::new();
    assert!(matches!(
        transfer_hcurl_order2_triangle_state_checked(
            &curved,
            &curved,
            &order1,
            &space,
            &vec![0.0; order1.num_global_dofs(&curved)],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));

    let mut nonfinite_values = values.clone();
    nonfinite_values[0] = f64::NAN;
    assert!(matches!(
        transfer_hcurl_order2_triangle_state_checked(
            &curved,
            &curved,
            &space,
            &space,
            &nonfinite_values,
        ),
        Err(InterpolationError::NonFiniteSource { .. })
    ));

    let mut quadrilateral = unit_square_triangles(1);
    quadrilateral.elements[0].element_type = ElementType::Quadrilateral;
    assert!(matches!(
        transfer_hcurl_order2_triangle_state_checked(
            &unit_square_triangles(1),
            &quadrilateral,
            &HcurlSpace::order2(),
            &HcurlSpace::order2(),
            &vec![0.0; HcurlSpace::order2().num_global_dofs(&unit_square_triangles(1))],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));
}
