use sonium_fem::mesh::{
    InterpolationError, transfer_hcurl_order2_triangle_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

#[test]
fn order2_triangle_transfer_reproduces_same_mesh_state() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::order2();
    let dof_count = space.num_global_dofs(&mesh);
    let source_values: Vec<f64> = (0..dof_count)
        .map(|index| 0.25 + 0.17 * index as f64 - 0.03 * (index * index) as f64)
        .collect();

    let transferred =
        transfer_hcurl_order2_triangle_state_checked(&mesh, &mesh, &space, &space, &source_values)
            .expect("same affine mesh must transfer an order-2 state");

    assert_eq!(transferred.len(), source_values.len());
    let max_error = transferred
        .iter()
        .zip(source_values.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        max_error < 1.0e-11,
        "max order-2 transfer error: {max_error:e}"
    );
}

#[test]
fn order2_triangle_transfer_rejects_non_affine_or_wrong_order_inputs() {
    let mesh = unit_square_triangles(1);
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();
    let values = vec![0.0; order2.num_global_dofs(&mesh)];

    let error =
        transfer_hcurl_order2_triangle_state_checked(&mesh, &mesh, &order1, &order2, &values)
            .expect_err("order-1 source must be rejected before numerical transfer");
    assert!(matches!(error, InterpolationError::InvalidMesh { .. }));
}
