use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order1_hanging_edge_constraints_2d,
    hcurl_order1_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, Point};
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

fn constant_tangential_integral(mesh: &Mesh, edge: (usize, usize)) -> f64 {
    mesh.nodes[edge.1].x - mesh.nodes[edge.0].x
}

#[test]
fn order1_hcurl_hanging_edges_preserve_signed_line_moments() {
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::new();
    let constraints = hcurl_order1_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10).unwrap();
    assert_eq!(constraints.len(), 2);
    assert!(
        constraints
            .iter()
            .all(|constraint| constraint.signed_fraction.abs() - 0.5 < 1.0e-12)
    );

    let mpc = hcurl_order1_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10).unwrap();
    let topology = MeshTopology::build(&mesh);
    let reduced = mpc
        .reduced_full_dofs()
        .iter()
        .map(|&dof| {
            Complex64::new(
                constant_tangential_integral(&mesh, topology.edges[dof]),
                0.0,
            )
        })
        .collect::<Vec<_>>();
    let recovered = mpc.recover(&reduced).unwrap();
    for (edge, value) in topology.edges.iter().zip(recovered.iter()) {
        assert!((value.re - constant_tangential_integral(&mesh, *edge)).abs() < 1.0e-12);
        assert!(value.im.abs() < 1.0e-14);
    }
}

#[test]
fn hcurl_hanging_edges_reject_unsupported_order_and_dimension() {
    let mesh = hanging_edge_mesh();
    assert!(matches!(
        hcurl_order1_hanging_edge_constraints_2d(&mesh, &HcurlSpace::order2(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(2))
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
        hcurl_order1_hanging_edge_constraints_2d(&three_d, &HcurlSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(3))
    ));
}
