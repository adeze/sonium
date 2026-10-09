use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order2_hanging_edge_constraints_2d,
    hcurl_order2_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, Point, unit_square_triangles};
use sonium_fem::spaces::HcurlSpace;

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

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual} expected={expected}"
    );
}

#[test]
fn order2_hcurl_hanging_edges_have_signed_two_mode_trace_matrices() {
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::order2();
    let topology = MeshTopology::build(&mesh);
    let constraints = hcurl_order2_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10).unwrap();

    assert_eq!(constraints.len(), 2);
    for constraint in &constraints {
        assert_close(constraint.signed_fraction.abs(), 0.5);
        assert_close(constraint.coefficients[0][0], constraint.signed_fraction);
        assert_close(constraint.coefficients[1][0], 0.0);
        assert_close(constraint.coefficients[1][1], 0.25);
        if constraint.signed_fraction > 0.0 {
            assert_close(constraint.fine_start_parameter, 0.0);
            assert_close(constraint.coefficients[0][1], 0.25);
        } else {
            assert_close(constraint.fine_start_parameter, 1.0);
            assert_close(constraint.coefficients[0][1], 0.25);
        }
        for fine_parameter in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let coarse_parameter =
                constraint.fine_start_parameter + constraint.signed_fraction * fine_parameter;
            let expected_trace =
                constraint.signed_fraction * (1.0 + 0.7 * (1.0 - 2.0 * coarse_parameter));
            let fine_mode_0 = constraint.coefficients[0][0] + 0.7 * constraint.coefficients[0][1];
            let fine_mode_1 = constraint.coefficients[1][0] + 0.7 * constraint.coefficients[1][1];
            let mapped_trace = fine_mode_0 + fine_mode_1 * (1.0 - 2.0 * fine_parameter);
            assert_close(mapped_trace, expected_trace);
        }
        assert!(constraint.slave_edge != constraint.coarse_edge);
        assert!(topology.edges[constraint.coarse_edge] == (0, 1));
    }
    assert!(
        constraints
            .iter()
            .any(|constraint| constraint.signed_fraction < 0.0)
    );
}

#[test]
fn order2_hcurl_hanging_edges_recover_both_trace_modes() {
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::order2();
    let constraints = hcurl_order2_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10).unwrap();
    let mpc = hcurl_order2_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10).unwrap();
    let coarse_edge = constraints[0].coarse_edge;
    let coarse_values = [Complex64::new(1.25, -0.5), Complex64::new(-0.75, 0.25)];
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (reduced_dof, &full_dof) in mpc.reduced_full_dofs().iter().enumerate() {
        if full_dof == coarse_edge * 2 {
            reduced[reduced_dof] = coarse_values[0];
        } else if full_dof == coarse_edge * 2 + 1 {
            reduced[reduced_dof] = coarse_values[1];
        }
    }

    let recovered = mpc.recover(&reduced).unwrap();
    for constraint in constraints {
        let expected = [
            constraint.coefficients[0][0] * coarse_values[0]
                + constraint.coefficients[0][1] * coarse_values[1],
            constraint.coefficients[1][0] * coarse_values[0]
                + constraint.coefficients[1][1] * coarse_values[1],
        ];
        for mode in 0..2 {
            let value = recovered[constraint.slave_edge * 2 + mode];
            assert!((value - expected[mode]).norm() < 1.0e-12);
        }
    }
}

#[test]
fn order2_hcurl_hanging_edges_handle_nonuniform_fraction() {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(0.25, 0.0),
        Point::new_2d(0.25, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![0, 3, 4]);

    let constraints =
        hcurl_order2_hanging_edge_constraints_2d(&mesh, &HcurlSpace::order2(), 1.0e-10).unwrap();
    assert_eq!(constraints.len(), 1);
    let constraint = constraints[0];
    assert_close(constraint.signed_fraction, 0.25);
    assert_close(constraint.fine_start_parameter, 0.0);
    assert_close(constraint.coefficients[0][0], 0.25);
    assert_close(constraint.coefficients[0][1], 0.1875);
    assert_close(constraint.coefficients[1][1], 0.0625);
}

#[test]
fn order2_hcurl_hanging_edges_reject_invalid_inputs_and_conforming_meshes() {
    let mesh = hanging_edge_mesh();
    assert!(matches!(
        hcurl_order2_hanging_edge_constraints_2d(&mesh, &HcurlSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(1))
    ));
    assert!(matches!(
        hcurl_order2_hanging_edge_constraints_2d(&mesh, &HcurlSpace::order2(), 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ));

    let mut mesh_3d = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        mesh_3d.add_node(point);
    }
    mesh_3d.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    assert!(matches!(
        hcurl_order2_hanging_edge_constraints_2d(&mesh_3d, &HcurlSpace::order2(), 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(3))
    ));

    let conforming = unit_square_triangles(1);
    assert!(
        hcurl_order2_hanging_edge_constraints_2d(&conforming, &HcurlSpace::order2(), 1.0e-10)
            .unwrap()
            .is_empty()
    );

    let mut curved = hanging_edge_mesh();
    curved.add_node(Point::new_2d(0.25, 0.0));
    curved.add_node(Point::new_2d(0.75, 0.0));
    curved.elements[0].nodes = vec![0, 1, 2, 5, 6, 3];
    assert!(matches!(
        hcurl_order2_hanging_edge_constraints_2d(&curved, &HcurlSpace::order2(), 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ));
}
