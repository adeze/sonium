//! Independent Q3 hexahedral H1 reproduction and topology reference.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_cube_hexahedra;
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn reference_field(point: [f64; 3]) -> (f64, [f64; 3]) {
    let [x, y, z] = point;
    let value =
        0.7 + 1.1 * x - 0.8 * y + 0.4 * z + 0.3 * x * y - 0.2 * y * z + 0.15 * x * z + 0.25 * x * x
            - 0.1 * y * y
            + 0.2 * z * z
            + 0.05 * x * x * y
            - 0.04 * x * y * z
            + 0.03 * y * z * z
            + 0.02 * x * x * x
            - 0.01 * y * y * y
            + 0.015 * z * z * z;
    let gradient = [
        1.1 + 0.3 * y + 0.15 * z + 0.5 * x + 0.1 * x * y - 0.04 * y * z + 0.06 * x * x,
        -0.8 + 0.3 * x - 0.2 * z - 0.2 * y + 0.05 * x * x - 0.04 * x * z + 0.03 * z * z
            - 0.03 * y * y,
        0.4 - 0.2 * y + 0.15 * x + 0.4 * z - 0.04 * x * y + 0.06 * y * z + 0.045 * z * z,
    ];
    (value, gradient)
}

#[test]
fn p3_hexahedron_reproduces_independent_tensor_cubic_reference() {
    let mesh = unit_cube_hexahedra(1);
    let space = H1Space::new(PolynomialDegree::P3);

    space
        .validate(&mesh)
        .expect("affine Q3 hexahedron must validate");
    assert_eq!(space.num_local_dofs(mesh.elements[0].element_type), 64);
    assert_eq!(space.num_global_dofs(&mesh), 64);

    let coordinates = space
        .dof_coordinates(&mesh)
        .expect("Q3 H1 DOFs expose physical coordinates");
    assert_eq!(coordinates.len(), 64);
    let coefficients: Vec<_> = coordinates
        .iter()
        .map(|&point| reference_field(point).0)
        .collect();

    let local_dofs = space.local_to_global(&mesh, 0);
    assert_eq!(local_dofs.len(), 64);
    let mut unique_dofs = local_dofs.clone();
    unique_dofs.sort_unstable();
    unique_dofs.dedup();
    assert_eq!(unique_dofs.len(), 64);
    assert_eq!(unique_dofs, (0..64).collect::<Vec<_>>());
    for (local, &global) in local_dofs.iter().enumerate() {
        const NODE_INDICES: [(usize, usize, usize); 64] = [
            (0, 0, 0),
            (3, 0, 0),
            (3, 3, 0),
            (0, 3, 0),
            (0, 0, 3),
            (3, 0, 3),
            (3, 3, 3),
            (0, 3, 3),
            (1, 0, 0),
            (2, 0, 0),
            (3, 1, 0),
            (3, 2, 0),
            (2, 3, 0),
            (1, 3, 0),
            (0, 2, 0),
            (0, 1, 0),
            (1, 0, 3),
            (2, 0, 3),
            (3, 1, 3),
            (3, 2, 3),
            (2, 3, 3),
            (1, 3, 3),
            (0, 2, 3),
            (0, 1, 3),
            (0, 0, 1),
            (0, 0, 2),
            (3, 0, 1),
            (3, 0, 2),
            (3, 3, 1),
            (3, 3, 2),
            (0, 3, 1),
            (0, 3, 2),
            (1, 1, 0),
            (2, 1, 0),
            (2, 2, 0),
            (1, 2, 0),
            (1, 1, 3),
            (2, 1, 3),
            (2, 2, 3),
            (1, 2, 3),
            (1, 0, 1),
            (2, 0, 1),
            (2, 0, 2),
            (1, 0, 2),
            (1, 3, 1),
            (2, 3, 1),
            (2, 3, 2),
            (1, 3, 2),
            (0, 1, 1),
            (0, 2, 1),
            (0, 2, 2),
            (0, 1, 2),
            (3, 1, 1),
            (3, 2, 1),
            (3, 2, 2),
            (3, 1, 2),
            (1, 1, 1),
            (2, 1, 1),
            (2, 2, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 1, 2),
            (2, 2, 2),
            (1, 2, 2),
        ];
        let (i, j, k) = NODE_INDICES[local];
        let expected = [i as f64 / 3.0, j as f64 / 3.0, k as f64 / 3.0];
        for (actual, expected) in coordinates[global].iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 2.0e-12,
                "local {local}: {actual} != {expected}"
            );
        }
    }

    for &(xi, eta, zeta) in &[
        (-0.71, -0.33, 0.28),
        (-0.19, 0.62, -0.44),
        (0.37, -0.21, 0.81),
        (0.76, 0.48, -0.13),
    ] {
        let tabulation = space.tabulate(&mesh, 0, xi, eta, zeta);
        let mut value = 0.0;
        let mut gradient = [0.0; 3];
        for (local, &global) in local_dofs.iter().enumerate() {
            value += coefficients[global] * tabulation.value(local);
            for (axis, component) in gradient.iter_mut().enumerate() {
                *component += coefficients[global] * tabulation.grad(local, axis);
            }
        }

        let (expected_value, expected_gradient) = reference_field(tabulation.physical_coords);
        assert!(
            (value - expected_value).abs() < 2.0e-12,
            "value {value:.16e}, expected {expected_value:.16e}, physical {:?}",
            tabulation.physical_coords
        );
        for (actual, expected) in gradient.iter().zip(expected_gradient) {
            assert!(
                (actual - expected).abs() < 2.0e-11,
                "gradient component {actual:.16e}, expected {expected:.16e}, physical {:?}",
                tabulation.physical_coords
            );
        }
        assert!((tabulation.values.iter().sum::<f64>() - 1.0).abs() < 2.0e-12);
    }
}

#[test]
fn p3_hexahedron_boundary_contract_excludes_cell_bubbles() {
    let mesh = unit_cube_hexahedra(1);
    let space = H1Space::new(PolynomialDegree::P3);
    let boundary = space.boundary_dofs(&mesh, 0);

    assert_eq!(boundary.len(), 56);
    assert!(boundary.iter().all(|&dof| dof < 56));
    assert_eq!(
        boundary,
        (0..56).collect::<Vec<_>>(),
        "all vertex, edge, and face Q3 DOFs lie on the cube boundary"
    );
}
