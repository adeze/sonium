use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt0_hanging_edge_constraints_2d, hdiv_rt0_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

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

#[test]
fn rt0_hdiv_hanging_edges_preserve_signed_normal_fluxes() {
    let mesh = hanging_edge_mesh();
    let space = HdivSpace::new();
    let constraints = hdiv_rt0_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10).unwrap();

    assert_eq!(constraints.len(), 2);
    assert!(
        constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction - 0.5).abs() < 1.0e-12)
    );
    assert!(
        constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction + 0.5).abs() < 1.0e-12)
    );

    let mpc = hdiv_rt0_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10).unwrap();
    assert_eq!(mpc.num_full_dofs(), space.num_global_dofs(&mesh));

    let coarse_edge = constraints[0].coarse_edge;
    let coarse_reduced = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_edge)
        .expect("coarse edge must remain a reduced DOF");
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    reduced[coarse_reduced] = Complex64::new(4.0, 0.0);
    let recovered = mpc.recover(&reduced).unwrap();

    for constraint in constraints {
        let value = recovered[constraint.slave_edge];
        assert!((value.re - 4.0 * constraint.signed_fraction).abs() < 1.0e-12);
        assert!(value.im.abs() < 1.0e-14);
    }
}

#[test]
fn rt0_hdiv_hanging_edges_reject_unsupported_inputs() {
    let mesh = hanging_edge_mesh();
    assert!(matches!(
        hdiv_rt0_hanging_edge_constraints_2d(
            &mesh,
            &HdivSpace::with_order(HdivOrder::Rt1),
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedOrder(2))
    ));
    assert!(matches!(
        hdiv_rt0_hanging_edge_constraints_2d(&mesh, &HdivSpace::new(), 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ));

    let mut three_d = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        three_d.add_node(point);
    }
    three_d.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    assert!(matches!(
        hdiv_rt0_hanging_edge_constraints_2d(&three_d, &HdivSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(3))
    ));
}
