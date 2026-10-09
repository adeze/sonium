use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt0_hanging_tri_face_constraints_3d,
    hdiv_rt0_hanging_tri_face_mpc_system_3d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point, unit_cube_hexahedra};
use sonium_fem::spaces::{HdivOrder, HdivSpace};

fn hanging_face_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(1.0 / 3.0, 1.0 / 3.0, -1.0),
        Point::new_3d(1.0 / 3.0, 1.0 / 3.0, 0.0),
        Point::new_3d(1.0 / 3.0, 1.0 / 3.0, 1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 2, 1, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 4, 5]);
    mesh.add_element(ElementType::Tetrahedron, vec![1, 2, 4, 5]);
    mesh.add_element(ElementType::Tetrahedron, vec![2, 0, 4, 5]);
    mesh
}

#[test]
fn rt0_hdiv_hanging_faces_preserve_signed_area_fractions() {
    let mesh = hanging_face_mesh();
    let space = HdivSpace::new();
    let constraints = hdiv_rt0_hanging_tri_face_constraints_3d(&mesh, &space, 1.0e-10).unwrap();

    assert_eq!(constraints.len(), 3);
    assert!(
        constraints
            .iter()
            .all(|constraint| (constraint.signed_fraction.abs() - 1.0 / 3.0).abs() < 1.0e-12)
    );
    assert!(
        constraints
            .iter()
            .any(|constraint| constraint.signed_fraction < 0.0),
        "the fixture must exercise opposite canonical face orientation"
    );
    assert!(
        constraints
            .iter()
            .all(|constraint| constraint.coarse_face == constraints[0].coarse_face)
    );

    let mpc = hdiv_rt0_hanging_tri_face_mpc_system_3d(&mesh, &space, 1.0e-10).unwrap();
    let coarse_reduced = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == constraints[0].coarse_face)
        .expect("coarse face must remain a reduced DOF");
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    reduced[coarse_reduced] = Complex64::new(6.0, 0.0);
    let recovered = mpc.recover(&reduced).unwrap();
    for constraint in constraints {
        let value = recovered[constraint.slave_face];
        assert!((value.re - 6.0 * constraint.signed_fraction).abs() < 1.0e-12);
        assert!(value.im.abs() < 1.0e-14);
    }
}

#[test]
fn rt0_hdiv_hanging_faces_reject_unsupported_inputs() {
    let mesh = hanging_face_mesh();
    assert!(matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(
            &mesh,
            &HdivSpace::with_order(HdivOrder::Rt1),
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedOrder(2))
    ));
    assert!(matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&mesh, &HdivSpace::new(), 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ));

    let mut two_d = Mesh::new(2);
    two_d.add_node(Point::new_2d(0.0, 0.0));
    two_d.add_node(Point::new_2d(1.0, 0.0));
    two_d.add_node(Point::new_2d(0.0, 1.0));
    two_d.add_element(ElementType::Triangle, vec![0, 1, 2]);
    assert!(matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&two_d, &HdivSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(2))
    ));

    let mut curved = mesh.clone();
    curved.add_node(Point::new_3d(0.2, 0.2, -0.5));
    curved.elements[0].nodes.push(6);
    assert!(matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&curved, &HdivSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ));

    let hexahedra = unit_cube_hexahedra(1);
    assert!(matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&hexahedra, &HdivSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedFaceTopology)
    ));

    let conforming = sonium_fem::mesh::unit_cube_tetrahedra(1);
    assert!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&conforming, &HdivSpace::new(), 1.0e-10)
            .unwrap()
            .is_empty(),
        "conforming tetrahedral faces must not create hanging constraints"
    );
}
