use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{
    HangingNodeError, MpcError, h1_hanging_face_mpc_system_3d, h1_hanging_node_mpc_system_2d,
    vector_h1_hanging_face_mpc_system_3d, vector_h1_hanging_node_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

fn hanging_edge_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(0.5, 0.0),
        Point::new_2d(0.5, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![0, 3, 4]);
    mesh.add_element(ElementType::Triangle, vec![3, 1, 4]);
    mesh
}

fn hanging_face_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(0.5, 0.0, 0.0),
        Point::new_3d(0.5, 0.5, 0.0),
        Point::new_3d(0.0, 0.5, 0.0),
        Point::new_3d(0.1, 0.1, -1.0),
        Point::new_3d(0.6, 0.1, -1.0),
        Point::new_3d(0.1, 0.6, -1.0),
        Point::new_3d(0.3, 0.3, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 4, 6, 7]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 1, 5, 8]);
    mesh.add_element(ElementType::Tetrahedron, vec![6, 5, 2, 9]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 5, 6, 10]);
    mesh
}

fn vector_value(point: [f64; 3], component: usize) -> f64 {
    let x = point[0];
    let y = point[1];
    match component {
        0 => 1.0 + 2.0 * x + 0.5 * x * x,
        1 => -2.0 + 0.75 * x - y + 0.25 * x * x,
        _ => 3.0 - 0.5 * x + 1.25 * y + 0.125 * x * x,
    }
}

fn assert_vector_recovery(
    space: &VectorH1Space,
    mesh: &Mesh,
    mpc: &sonium_fem::boundary::MpcSystem,
) {
    let coordinates = space
        .dof_coordinates(mesh)
        .expect("vector H1 DOF coordinates");
    let reduced = mpc
        .reduced_full_dofs()
        .iter()
        .map(|&dof| {
            Complex64::new(
                vector_value(coordinates[dof], dof % space.num_components()),
                0.0,
            )
        })
        .collect::<Vec<_>>();
    let recovered = mpc.recover(&reduced).expect("vector H1 MPC recovery");
    assert_eq!(recovered.len(), coordinates.len());
    for (dof, value) in recovered.iter().enumerate() {
        let expected = vector_value(coordinates[dof], dof % space.num_components());
        assert!((value.re - expected).abs() < 1.0e-10, "DOF {dof}");
        assert!(value.im.abs() < 1.0e-14);
    }
}

#[test]
fn vector_h1_edge_constraints_preserve_interleaved_components() {
    let mesh = hanging_edge_mesh();
    let scalar = H1Space::new(PolynomialDegree::P2);
    let scalar_mpc = h1_hanging_node_mpc_system_2d(&mesh, &scalar, 1.0e-10).unwrap();
    let vector = VectorH1Space::new(PolynomialDegree::P2, 2);
    let vector_mpc = vector_h1_hanging_node_mpc_system_2d(&mesh, &vector, 1.0e-10).unwrap();

    let scalar_constraints = scalar_mpc.num_full_dofs() - scalar_mpc.num_reduced_dofs();
    let vector_constraints = vector_mpc.num_full_dofs() - vector_mpc.num_reduced_dofs();
    assert!(scalar_constraints > 0);
    assert_eq!(vector_constraints, 2 * scalar_constraints);
    assert_vector_recovery(&vector, &mesh, &vector_mpc);
}

#[test]
fn vector_h1_tetrahedral_face_constraints_preserve_three_components() {
    let mesh = hanging_face_mesh();
    let scalar = H1Space::new(PolynomialDegree::P2);
    let scalar_mpc = h1_hanging_face_mpc_system_3d(&mesh, &scalar, 1.0e-10).unwrap();
    let vector = VectorH1Space::new(PolynomialDegree::P2, 3);
    let vector_mpc = vector_h1_hanging_face_mpc_system_3d(&mesh, &vector, 1.0e-10).unwrap();

    let scalar_constraints = scalar_mpc.num_full_dofs() - scalar_mpc.num_reduced_dofs();
    let vector_constraints = vector_mpc.num_full_dofs() - vector_mpc.num_reduced_dofs();
    assert!(scalar_constraints > 0);
    assert_eq!(vector_constraints, 3 * scalar_constraints);
    assert_vector_recovery(&vector, &mesh, &vector_mpc);
}

#[test]
fn vector_h1_zero_components_fail_closed() {
    let error = vector_h1_hanging_node_mpc_system_2d(
        &hanging_edge_mesh(),
        &VectorH1Space::new(PolynomialDegree::P1, 0),
        1.0e-10,
    )
    .expect_err("zero-component vector space must be rejected");
    assert!(matches!(
        error,
        HangingNodeError::Mpc(MpcError::InvalidComponents(0))
    ));
}
