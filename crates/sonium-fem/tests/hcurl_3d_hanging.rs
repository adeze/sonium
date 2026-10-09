//! Independent affine tetrahedral order-1 H(curl) hanging-edge checks.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order1_hanging_edge_constraints_3d,
    hcurl_order1_hanging_edge_mpc_system_3d,
};
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, Point, unit_cube_tetrahedra};
use sonium_fem::spaces::HcurlSpace;

fn hanging_edge_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(0.5, 0.0, 0.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 4, 2, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 1, 2, 3]);
    mesh
}

fn constant_line_integral(mesh: &Mesh, edge: (usize, usize), vector: [f64; 3]) -> f64 {
    let first = mesh.nodes[edge.0];
    let second = mesh.nodes[edge.1];
    vector[0] * (second.x - first.x)
        + vector[1] * (second.y - first.y)
        + vector[2] * (second.z - first.z)
}

#[test]
fn order1_hcurl_3d_hanging_edges_preserve_signed_line_moments() {
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::new();
    let constraints = hcurl_order1_hanging_edge_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    let topology = MeshTopology::build(&mesh);
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
    let vector = [2.0, -1.0, 0.5];
    for constraint in constraints {
        let coarse_integral =
            constant_line_integral(&mesh, topology.edges[constraint.coarse_edge], vector);
        let fine_integral =
            constant_line_integral(&mesh, topology.edges[constraint.slave_edge], vector);
        assert!((fine_integral - constraint.signed_fraction * coarse_integral).abs() < 1.0e-12);
    }
}

#[test]
fn order1_hcurl_3d_hanging_mpcs_recover_signed_edges() {
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::new();
    let constraints = hcurl_order1_hanging_edge_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    let mpc = hcurl_order1_hanging_edge_mpc_system_3d(&mesh, &space, 1.0e-10).unwrap();
    let coarse_edge = constraints[0].coarse_edge;
    let reduced_dof = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_edge)
        .expect("coarse edge remains free");
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    let coarse_value = Complex64::new(4.0, -0.75);
    reduced[reduced_dof] = coarse_value;
    let recovered = mpc.recover(&reduced).unwrap();
    for constraint in constraints {
        let expected = coarse_value * constraint.signed_fraction;
        assert!((recovered[constraint.slave_edge] - expected).norm() < 1.0e-12);
    }
}

#[test]
fn order1_hcurl_3d_hanging_rejects_unsupported_inputs_and_conforming_noop() {
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::new();
    assert!(matches!(
        hcurl_order1_hanging_edge_constraints_3d(&mesh, &HcurlSpace::order2(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(2))
    ));
    assert!(matches!(
        hcurl_order1_hanging_edge_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ));
    assert!(matches!(
        hcurl_order1_hanging_edge_constraints_3d(
            &sonium_fem::mesh::unit_square_triangles(1),
            &space,
            1.0e-10
        ),
        Err(MixedHangingError::UnsupportedDimension(2))
    ));
    let mut hexa = sonium_fem::mesh::unit_cube_hexahedra(1);
    assert!(matches!(
        hcurl_order1_hanging_edge_constraints_3d(&hexa, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedHcurlTopology)
    ));
    hexa.elements[0].nodes.push(hexa.nodes.len());
    hexa.add_node(Point::new_3d(0.25, 0.25, 0.25));
    assert!(matches!(
        hcurl_order1_hanging_edge_constraints_3d(&hexa, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ));
    assert!(
        hcurl_order1_hanging_edge_constraints_3d(&unit_cube_tetrahedra(1), &space, 1.0e-10)
            .unwrap()
            .is_empty()
    );
}
