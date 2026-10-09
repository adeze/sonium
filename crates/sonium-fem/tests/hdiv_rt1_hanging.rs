//! Independent affine triangular RT1 H(div) hanging-edge contract checks.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt1_hanging_edge_constraints_2d, hdiv_rt1_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point, unit_square_quads, unit_square_triangles};
use sonium_fem::spaces::{HdivOrder, HdivSpace};

fn hanging_edge_mesh() -> Mesh {
    hanging_edge_mesh_with_split(0.5)
}

fn hanging_edge_mesh_with_split(split: f64) -> Mesh {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(split, 0.0),
        Point::new_2d(split, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![0, 3, 4]);
    mesh.add_element(ElementType::Triangle, vec![3, 1, 4]);
    mesh
}

#[test]
fn rt1_hdiv_hanging_edges_support_nonuniform_subedges() {
    let mesh = hanging_edge_mesh_with_split(0.25);
    let constraints = hdiv_rt1_hanging_edge_constraints_2d(
        &mesh,
        &HdivSpace::with_order(HdivOrder::Rt1),
        1.0e-10,
    )
    .unwrap();
    assert_eq!(constraints.len(), 2);
    assert!(
        constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction.abs() - 0.25).abs() < 1.0e-12)
    );
    assert!(
        constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction.abs() - 0.75).abs() < 1.0e-12)
    );
}

#[test]
fn rt1_hdiv_hanging_edges_have_signed_two_mode_restrictions() {
    let mesh = hanging_edge_mesh();
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let constraints =
        hdiv_rt1_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10).expect("RT1 constraints");

    assert_eq!(constraints.len(), 2);
    for constraint in &constraints {
        let b = constraint.signed_fraction;
        let t0 = 2.0 * constraint.fine_start_parameter - 1.0;
        assert!((constraint.coefficients[0][0] - b).abs() < 1.0e-12);
        assert!((constraint.coefficients[0][1] - 3.0 * b * t0).abs() < 1.0e-12);
        assert!(constraint.coefficients[1][0].abs() < 1.0e-12);
        assert!((constraint.coefficients[1][1] - b * b).abs() < 1.0e-12);
    }
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
}

#[test]
fn rt1_hdiv_hanging_mpcs_recover_constant_and_linear_moments() {
    let mesh = hanging_edge_mesh();
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let constraints = hdiv_rt1_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10).unwrap();
    let mpc = hdiv_rt1_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10).unwrap();

    let coarse_edge = constraints[0].coarse_edge;
    let coarse_mode_0 = coarse_edge * 2;
    let coarse_mode_1 = coarse_mode_0 + 1;
    let reduced_mode_0 = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_mode_0)
        .expect("coarse constant moment remains free");
    let reduced_mode_1 = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_mode_1)
        .expect("coarse linear moment remains free");
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    let coarse_constant = Complex64::new(2.0, -1.0);
    let coarse_linear = Complex64::new(-3.0, 0.5);
    reduced[reduced_mode_0] = coarse_constant;
    reduced[reduced_mode_1] = coarse_linear;

    let recovered = mpc.recover(&reduced).unwrap();
    for constraint in constraints {
        let b = constraint.signed_fraction;
        let t0 = 2.0 * constraint.fine_start_parameter - 1.0;
        let expected_0 = b * (coarse_constant + 3.0 * t0 * coarse_linear);
        let expected_1 = b * b * coarse_linear;
        assert!((recovered[constraint.slave_edge * 2] - expected_0).norm() < 1.0e-12);
        assert!((recovered[constraint.slave_edge * 2 + 1] - expected_1).norm() < 1.0e-12);
    }
}

#[test]
fn rt1_hdiv_hanging_rejects_non_triangles_and_other_orders() {
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let quads = unit_square_quads(1);
    assert!(matches!(
        hdiv_rt1_hanging_edge_constraints_2d(&quads, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedEdgeTopology)
    ));

    let triangles = unit_square_triangles(1);
    assert!(matches!(
        hdiv_rt1_hanging_edge_constraints_2d(
            &triangles,
            &HdivSpace::with_order(HdivOrder::Rt0),
            1.0e-10
        ),
        Err(MixedHangingError::UnsupportedOrder(1))
    ));
    assert!(matches!(
        hdiv_rt1_hanging_edge_constraints_2d(&triangles, &space, 0.0),
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
        hdiv_rt1_hanging_edge_constraints_2d(&three_d, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(3))
    ));
}

#[test]
fn rt1_hdiv_conforming_mesh_has_no_hanging_constraints() {
    let mesh = unit_square_triangles(2);
    let constraints = hdiv_rt1_hanging_edge_constraints_2d(
        &mesh,
        &HdivSpace::with_order(HdivOrder::Rt1),
        1.0e-10,
    )
    .unwrap();
    assert!(constraints.is_empty());
}
