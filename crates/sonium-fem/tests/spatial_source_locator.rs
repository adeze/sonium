use sonium_fem::mesh::{
    InterpolationError, create_interpolation_data_checked, interpolate_field_checked,
    interpolate_scalar_at_point_checked, transfer_design_field_checked, unit_square_triangles,
};

#[test]
fn bvh_backed_scalar_transfer_is_exact_on_a_refined_grid() {
    let source = unit_square_triangles(8);
    let target = unit_square_triangles(16);
    let source_values: Vec<f64> = source
        .nodes
        .iter()
        .map(|node| 0.5 + 1.5 * node.x - 2.25 * node.y)
        .collect();
    let data = create_interpolation_data_checked(&target, &source).unwrap();
    let target_values = interpolate_field_checked(&data, &source, &target, &source_values).unwrap();

    for (node, value) in target.nodes.iter().zip(target_values) {
        let expected = 0.5 + 1.5 * node.x - 2.25 * node.y;
        assert!((value - expected).abs() < 1.0e-12);
    }
}

#[test]
fn bvh_backed_design_transfer_is_finite_and_deterministic() {
    let source = unit_square_triangles(8);
    let target = unit_square_triangles(16);
    let source_values: Vec<f64> = (0..source.num_elements())
        .map(|element| 10.0 + element as f64)
        .collect();
    let first = transfer_design_field_checked(&source, &target, &source_values).unwrap();
    let second = transfer_design_field_checked(&source, &target, &source_values).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), target.num_elements());
    assert!(first.iter().all(|value| value.is_finite()));
}

#[test]
fn bvh_backed_point_lookup_keeps_fail_closed_errors() {
    let mesh = unit_square_triangles(8);
    let values: Vec<f64> = mesh.nodes.iter().map(|node| node.x + node.y).collect();
    let inside = interpolate_scalar_at_point_checked(&mesh, &values, [0.37, 0.61, 0.0])
        .unwrap()
        .unwrap();
    assert!((inside - 0.98).abs() < 1.0e-12);
    assert_eq!(
        interpolate_scalar_at_point_checked(&mesh, &values, [1.1, 0.5, 0.0]).unwrap(),
        None
    );
    assert!(matches!(
        interpolate_scalar_at_point_checked(&mesh, &values, [f64::NAN, 0.5, 0.0]),
        Err(InterpolationError::NonFinitePoint)
    ));
}
