//! Affine higher-order and mixed-order scalar H1 state transfer.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::transfer_h1_state_checked;
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn field_p2(point: [f64; 3]) -> f64 {
    1.0 + 2.0 * point[0] - 3.0 * point[1] + 4.0 * point[0] * point[1] + point[0] * point[0]
}

fn field_p3(point: [f64; 3]) -> f64 {
    0.5 - point[0] + 2.0 * point[1] + point[0] * point[0] * point[1]
        - 0.75 * point[1] * point[1] * point[1]
}

fn coordinates(mesh: &Mesh, space: &H1Space) -> Vec<[f64; 3]> {
    space
        .dof_coordinates(mesh)
        .expect("validated H1 spaces provide complete DOF coordinates")
}

fn state(mesh: &Mesh, space: &H1Space, field: fn([f64; 3]) -> f64) -> Vec<f64> {
    coordinates(mesh, space).into_iter().map(field).collect()
}

fn assert_transferred_field(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    field: fn([f64; 3]) -> f64,
) {
    let transferred = transfer_h1_state_checked(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        source_values,
    )
    .expect("affine mixed-order H1 transfer should succeed");
    let expected = state(target_mesh, target_space, field);
    assert_eq!(transferred.len(), expected.len());
    let max_error = transferred
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(max_error < 2.0e-12, "mixed-order H1 error {max_error}");
}

#[test]
fn p2_source_transfers_exactly_into_p3_target() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2);
    let source_space = H1Space::new(PolynomialDegree::P2);
    let target_space = H1Space::new(PolynomialDegree::P3);
    let source_values = state(&source_mesh, &source_space, field_p2);
    assert_transferred_field(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        field_p2,
    );
}

#[test]
fn p3_source_restricts_exactly_into_p2_target() {
    let source_mesh = unit_square_triangles(2);
    let target_mesh = unit_square_triangles(1);
    let source_space = H1Space::new(PolynomialDegree::P3);
    let target_space = H1Space::new(PolynomialDegree::P2);
    let source_values = state(&source_mesh, &source_space, field_p3);
    assert_transferred_field(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        field_p3,
    );
}

#[test]
fn h1_transfer_rejects_non_affine_geometry_and_bad_state() {
    let source_mesh = unit_square_triangles(1);
    let mut higher_order_geometry = source_mesh.clone();
    for element in &mut higher_order_geometry.elements {
        element.nodes.extend([0; 7]);
    }
    let source_space = H1Space::new(PolynomialDegree::P2);
    let target_space = H1Space::new(PolynomialDegree::P2);
    let target_mesh = unit_square_triangles(1);
    let source_values = state(&source_mesh, &source_space, field_p2);

    assert!(
        transfer_h1_state_checked(
            &higher_order_geometry,
            &target_mesh,
            &source_space,
            &target_space,
            &source_values,
        )
        .is_err()
    );

    let mut non_finite = source_values;
    non_finite[0] = f64::NAN;
    assert!(
        transfer_h1_state_checked(
            &source_mesh,
            &target_mesh,
            &source_space,
            &target_space,
            &non_finite,
        )
        .is_err()
    );
}
