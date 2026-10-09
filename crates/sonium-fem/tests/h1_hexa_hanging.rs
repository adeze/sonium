use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{
    H1HangingQuadFaceConstraint, HangingNodeError, h1_hanging_quad_face_constraints_3d,
    h1_hanging_quad_face_mpc_system_3d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn hexa_hanging_face_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(1.0, 1.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(1.0, 0.0, 1.0),
        Point::new_3d(1.0, 1.0, 1.0),
        Point::new_3d(0.0, 1.0, 1.0),
        Point::new_3d(0.5, 0.0, 0.0),
        Point::new_3d(1.0, 0.5, 0.0),
        Point::new_3d(0.5, 1.0, 0.0),
        Point::new_3d(0.0, 0.5, 0.0),
        Point::new_3d(0.5, 0.5, 0.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Hexahedron, (0..8).collect());

    let patches = [
        [0, 8, 12, 11],
        [8, 1, 9, 12],
        [12, 9, 2, 10],
        [11, 12, 10, 3],
    ];
    for patch in patches {
        let mut bottom = [0usize; 4];
        for (slot, &top) in patch.iter().enumerate() {
            let point = mesh.nodes[top];
            bottom[slot] = mesh.nodes.len();
            mesh.add_node(Point::new_3d(point.x, point.y, -1.0));
        }
        mesh.add_element(
            ElementType::Hexahedron,
            vec![
                bottom[0], bottom[1], bottom[2], bottom[3], patch[0], patch[1], patch[2], patch[3],
            ],
        );
    }
    mesh
}

fn polynomial(point: [f64; 3]) -> f64 {
    let [x, y, _] = point;
    1.0 + x - 2.0 * y + x * y + x.powi(2) * y + x.powi(3) - y.powi(3)
}

fn polynomial_for_degree(point: [f64; 3], degree: PolynomialDegree) -> f64 {
    let [x, y, _] = point;
    match degree {
        PolynomialDegree::P1 => 1.0 + x - 2.0 * y + x * y,
        PolynomialDegree::P2 => 1.0 + x - 2.0 * y + x * y + x.powi(2) * y,
        PolynomialDegree::P3 => polynomial(point),
    }
}

fn assert_constraint_contract(
    constraint: &H1HangingQuadFaceConstraint,
    degree: PolynomialDegree,
    coordinates: &[[f64; 3]],
) {
    let expected = match degree {
        PolynomialDegree::P1 => 4,
        PolynomialDegree::P2 => 9,
        PolynomialDegree::P3 => 16,
    };
    let mut face = constraint.face;
    face.sort_unstable();
    assert_eq!(face, [0, 1, 2, 3]);
    assert_eq!(constraint.masters.len(), expected);
    let sum: f64 = constraint
        .masters
        .iter()
        .map(|term| term.coefficient.re)
        .sum();
    assert!((sum - 1.0).abs() < 1.0e-11);
    let value = polynomial_for_degree(coordinates[constraint.slave], degree);
    let interpolated: f64 = constraint
        .masters
        .iter()
        .map(|term| term.coefficient.re * polynomial_for_degree(coordinates[term.dof], degree))
        .sum();
    assert!((value - interpolated).abs() < 1.0e-10);
}

#[test]
fn affine_hexahedral_q1_q2_q3_face_constraints_cover_fine_patch() {
    let mesh = hexa_hanging_face_mesh();
    for (degree, expected_constraints) in [
        (PolynomialDegree::P1, 5),
        (PolynomialDegree::P2, 16),
        (PolynomialDegree::P3, 33),
    ] {
        let space = H1Space::new(degree);
        let coordinates = space.dof_coordinates(&mesh).expect("DOF coordinates");
        let constraints = h1_hanging_quad_face_constraints_3d(&mesh, &space, 1.0e-10)
            .expect("affine hexahedral constraints");
        assert_eq!(constraints.len(), expected_constraints, "degree {degree:?}");
        for constraint in &constraints {
            assert_constraint_contract(constraint, degree, &coordinates);
        }
    }
}

#[test]
fn affine_hexahedral_mpc_recovers_tensor_product_field_and_keeps_conforming_mesh() {
    let mesh = hexa_hanging_face_mesh();
    let space = H1Space::new(PolynomialDegree::P3);
    let coordinates = space.dof_coordinates(&mesh).expect("DOF coordinates");
    let mpc = h1_hanging_quad_face_mpc_system_3d(&mesh, &space, 1.0e-10).expect("hexahedral MPC");
    let reduced = mpc
        .reduced_full_dofs()
        .iter()
        .map(|&dof| Complex64::new(polynomial(coordinates[dof]), 0.0))
        .collect::<Vec<_>>();
    let recovered = mpc.recover(&reduced).expect("full recovery");
    for (dof, value) in recovered.iter().enumerate() {
        assert!((value.re - polynomial(coordinates[dof])).abs() < 1.0e-10);
        assert!(value.im.abs() < 1.0e-14);
    }

    let mut conforming = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(1.0, 1.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(1.0, 0.0, 1.0),
        Point::new_3d(1.0, 1.0, 1.0),
        Point::new_3d(0.0, 1.0, 1.0),
    ] {
        conforming.add_node(point);
    }
    conforming.add_element(ElementType::Hexahedron, (0..8).collect());
    let conforming_space = H1Space::new(PolynomialDegree::P2);
    let conforming_mpc =
        h1_hanging_quad_face_mpc_system_3d(&conforming, &conforming_space, 1.0e-10)
            .expect("conforming mesh is a valid empty constraint system");
    assert_eq!(
        conforming_mpc.num_full_dofs(),
        conforming_mpc.num_reduced_dofs()
    );
}

#[test]
fn hexahedral_face_constraints_fail_closed_for_non_hex_and_curved_geometry() {
    let mut tetra = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        tetra.add_node(point);
    }
    tetra.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    let error =
        h1_hanging_quad_face_constraints_3d(&tetra, &H1Space::new(PolynomialDegree::P1), 1.0e-10)
            .expect_err("tetrahedral input must be rejected");
    assert!(matches!(
        error,
        HangingNodeError::UnsupportedH1Element { .. }
    ));

    let mut curved = hexa_hanging_face_mesh();
    curved.elements[0].nodes.extend([0; 19]);
    let error =
        h1_hanging_quad_face_constraints_3d(&curved, &H1Space::new(PolynomialDegree::P2), 1.0e-10)
            .expect_err("explicit geometry must be rejected");
    assert!(matches!(
        error,
        HangingNodeError::UnsupportedGeometry { .. }
    ));

    let error = h1_hanging_quad_face_constraints_3d(
        &hexa_hanging_face_mesh(),
        &H1Space::new(PolynomialDegree::P1),
        0.0,
    )
    .expect_err("invalid tolerance must be rejected");
    assert!(matches!(error, HangingNodeError::InvalidTolerance(0.0)));
}
