//! Independent reference checks for explicit P3 isoparametric H1 geometry.
//!
//! The mesh carries the complete cubic nodal geometry, while the reference
//! checks independently construct the nodal coordinates and verify the
//! production H1 map, numbering, partition of unity, and assembled operators.

use sonium_fem::assembly::{assemble_mass, assemble_stiffness};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn curved_p3_triangle() -> Mesh {
    let mut mesh = Mesh::new(2);
    let mut points = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (0.0, 2.0 / 3.0),
        (0.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0),
    ];
    // A nonzero cubic bubble makes the geometry genuinely non-affine without
    // changing the boundary curve.
    points[9].1 += 0.08;
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..10).collect());
    mesh
}

fn curved_p3_hexahedron() -> Mesh {
    const NODE_INDEXES: [(usize, usize, usize); 64] = [
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
    let coordinates = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let mut mesh = Mesh::new(3);
    for &(i, j, k) in &NODE_INDEXES {
        let mut point = Point::new_3d(
            0.5 * (coordinates[i] + 1.0),
            0.5 * (coordinates[j] + 1.0),
            0.5 * (coordinates[k] + 1.0),
        );
        if i == 1 && j == 1 && k == 1 {
            point.y += 0.015;
        }
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Hexahedron, (0..64).collect());
    mesh
}

fn dense(rows: &[usize], cols: &[usize], values: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; n]; n];
    for ((&row, &column), &value) in rows.iter().zip(cols).zip(values) {
        result[row][column] += value;
    }
    result
}

/// Entry point used by the release QA wrapper.
pub fn explicit_p3_geometry_reference() {
    let triangle = curved_p3_triangle();
    let space = H1Space::new(PolynomialDegree::P3);
    let validation = space.validate(&triangle);
    assert!(
        validation.is_ok(),
        "explicit P3 triangle validation: {validation:?}"
    );
    assert_eq!(space.num_global_dofs(&triangle), 10);
    assert_eq!(
        space.local_to_global(&triangle, 0),
        (0..10).collect::<Vec<_>>()
    );

    let tabulation = space.tabulate(&triangle, 0, 0.21, 0.27, 0.0);
    assert!((tabulation.values.iter().sum::<f64>() - 1.0).abs() < 2e-12);
    assert!(tabulation.det_j.is_finite() && tabulation.det_j > 0.0);
    assert!(tabulation.derivatives.iter().all(|value| value.is_finite()));
    assert!(tabulation.physical_coords[1] > 0.27);
    assert_eq!(space.dof_coordinates(&triangle).unwrap().len(), 10);

    let stiffness = assemble_stiffness(&triangle, PolynomialDegree::P3);
    let mass = assemble_mass(&triangle, PolynomialDegree::P3);
    assert_eq!(stiffness.dim, 10);
    assert_eq!(mass.dim, 10);
    for matrix in [
        dense(&stiffness.rows, &stiffness.cols, &stiffness.values, 10),
        dense(&mass.rows, &mass.cols, &mass.values, 10),
    ] {
        for row in 0..10 {
            for column in 0..10 {
                assert!(matrix[row][column].is_finite());
                assert!((matrix[row][column] - matrix[column][row]).abs() < 2e-10);
            }
        }
    }

    let hexahedron = curved_p3_hexahedron();
    let hex_space = H1Space::new(PolynomialDegree::P3);
    assert!(hex_space.validate(&hexahedron).is_ok());
    assert_eq!(hex_space.num_global_dofs(&hexahedron), 64);
    assert_eq!(hex_space.local_to_global(&hexahedron, 0).len(), 64);
    let hex_tabulation = hex_space.tabulate(&hexahedron, 0, 0.17, -0.29, 0.23);
    assert!((hex_tabulation.values.iter().sum::<f64>() - 1.0).abs() < 2e-12);
    assert!(hex_tabulation.det_j.is_finite() && hex_tabulation.det_j > 0.0);
    assert!(
        hex_tabulation
            .derivatives
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn explicit_p3_geometry_reference_test() {
    explicit_p3_geometry_reference();
}
