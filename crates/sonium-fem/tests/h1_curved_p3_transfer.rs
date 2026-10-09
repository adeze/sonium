//! Explicit P3 isoparametric scalar H1 transfer.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, Point, transfer_h1_state_checked};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn explicit_p3_triangle_mesh(curved: bool) -> Mesh {
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
    if curved {
        points[9].1 += 0.08;
    }
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..10).collect());
    mesh
}

#[test]
fn curved_p3_h1_state_reproduces_complete_global_state() {
    let mesh = explicit_p3_triangle_mesh(true);
    let space = H1Space::new(PolynomialDegree::P3);
    let source_values: Vec<f64> = (0..space.num_global_dofs(&mesh))
        .map(|dof| 0.23 + 0.17 * dof as f64 - 0.011 * (dof * dof) as f64)
        .collect();

    let transferred = transfer_h1_state_checked(&mesh, &mesh, &space, &space, &source_values)
        .expect("curved P3 transfer should accept its checked geometry");

    assert_eq!(transferred.len(), source_values.len());
    for (actual, expected) in transferred.iter().zip(&source_values) {
        assert!(
            (actual - expected).abs() < 2.0e-10,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn curved_p3_h1_transfer_rejects_nonfinite_and_unsupported_geometry() {
    let mesh = explicit_p3_triangle_mesh(true);
    let space = H1Space::new(PolynomialDegree::P3);
    let mut nonfinite = vec![0.0; space.num_global_dofs(&mesh)];
    nonfinite[3] = f64::NAN;
    assert!(transfer_h1_state_checked(&mesh, &mesh, &space, &space, &nonfinite).is_err());

    let mut unsupported = explicit_p3_triangle_mesh(false);
    unsupported.elements[0].nodes.push(unsupported.nodes.len());
    unsupported.add_node(Point::new_2d(0.25, 0.25));
    assert!(
        transfer_h1_state_checked(&unsupported, &unsupported, &space, &space, &[0.0; 10]).is_err()
    );
}
