//! Checked P3 tetrahedral H1 admission and reference-shape invariants.

use sonium_fem::basis::{
    PolynomialDegree, p3_tetrahedron, p3_tetrahedron_grad, p3_tetrahedron_hessian,
    try_evaluate_shape,
};
use sonium_fem::mesh::MeshTopology;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::spaces::{FunctionSpace, H1Space};

#[test]
fn p3_tetrahedron_shape_is_partition_of_unity() {
    let shape = try_evaluate_shape(
        sonium_fem::mesh::ElementType::Tetrahedron,
        PolynomialDegree::P3,
        0.17,
        0.23,
        0.19,
    )
    .expect("P3 tetrahedron should be supported");
    assert_eq!(shape.values.len(), 20);
    assert_eq!(shape.gradients.len(), 60);
    assert!((shape.values.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    for direction in 0..3 {
        let sum = shape
            .gradients
            .chunks_exact(3)
            .map(|gradient| gradient[direction])
            .sum::<f64>();
        assert!(sum.abs() < 1.0e-12, "gradient partition error: {sum:e}");
    }
    assert!(
        shape
            .values
            .iter()
            .chain(shape.gradients.iter())
            .all(|value| value.is_finite())
    );
}

#[test]
fn p3_tetrahedron_gradients_match_finite_difference() {
    let (xi, eta, zeta) = (0.17, 0.23, 0.19);
    let h = 1.0e-6;
    let analytic = p3_tetrahedron_grad(xi, eta, zeta);
    let samples = [
        (
            p3_tetrahedron(xi + h, eta, zeta),
            p3_tetrahedron(xi - h, eta, zeta),
        ),
        (
            p3_tetrahedron(xi, eta + h, zeta),
            p3_tetrahedron(xi, eta - h, zeta),
        ),
        (
            p3_tetrahedron(xi, eta, zeta + h),
            p3_tetrahedron(xi, eta, zeta - h),
        ),
    ];
    for (basis, row) in analytic.iter().enumerate() {
        for (direction, &expected) in row.iter().enumerate() {
            let numerical = (samples[direction].0[basis] - samples[direction].1[basis]) / (2.0 * h);
            assert!(
                (expected - numerical).abs() < 1.0e-6,
                "P3 tet dN_{basis}/d[{direction}]: analytic={expected}, fd={numerical}"
            );
        }
    }
}

#[test]
fn p3_tetrahedron_hessians_match_gradient_finite_difference() {
    let (xi, eta, zeta) = (0.17, 0.23, 0.19);
    let h = 1.0e-6;
    let analytic = p3_tetrahedron_hessian(xi, eta, zeta);
    let samples = [
        p3_tetrahedron_grad(xi + h, eta, zeta),
        p3_tetrahedron_grad(xi - h, eta, zeta),
        p3_tetrahedron_grad(xi, eta + h, zeta),
        p3_tetrahedron_grad(xi, eta - h, zeta),
        p3_tetrahedron_grad(xi, eta, zeta + h),
        p3_tetrahedron_grad(xi, eta, zeta - h),
    ];
    for (basis, hessian) in analytic.iter().enumerate() {
        for direction in 0..3 {
            for component in 0..3 {
                let (plus, minus) = match direction {
                    0 => (samples[0][basis][component], samples[1][basis][component]),
                    1 => (samples[2][basis][component], samples[3][basis][component]),
                    _ => (samples[4][basis][component], samples[5][basis][component]),
                };
                let finite_difference = (plus - minus) / (2.0 * h);
                assert!(
                    (hessian[direction][component] - finite_difference).abs() < 1.0e-6,
                    "P3 Hessian mismatch basis={basis} direction={direction} component={component} analytic={} finite_difference={}",
                    hessian[direction][component],
                    finite_difference
                );
            }
        }
    }
}

#[test]
fn h1_p3_tetrahedron_has_shared_edge_and_face_dofs() {
    let mesh = unit_cube_tetrahedra(1);
    let space = H1Space::new(PolynomialDegree::P3);
    space.validate(&mesh).expect("P3 tetrahedral H1 admission");
    let topology = MeshTopology::build(&mesh);
    assert_eq!(
        space.num_local_dofs(sonium_fem::mesh::ElementType::Tetrahedron),
        20
    );
    assert_eq!(
        space.num_global_dofs(&mesh),
        mesh.num_nodes() + 2 * topology.num_edges() + topology.num_faces()
    );
    let coordinates = space
        .dof_coordinates(&mesh)
        .expect("P3 tetrahedral coordinates should be available");
    assert_eq!(coordinates.len(), space.num_global_dofs(&mesh));
    let (a, b) = topology.edges[0];
    let pa = mesh.nodes[a];
    let pb = mesh.nodes[b];
    for (coordinate, fraction) in [
        (coordinates[mesh.num_nodes()], 1.0 / 3.0),
        (coordinates[mesh.num_nodes() + 1], 2.0 / 3.0),
    ] {
        assert!((coordinate[0] - ((1.0 - fraction) * pa.x + fraction * pb.x)).abs() < 1.0e-12);
        assert!((coordinate[1] - ((1.0 - fraction) * pa.y + fraction * pb.y)).abs() < 1.0e-12);
        assert!((coordinate[2] - ((1.0 - fraction) * pa.z + fraction * pb.z)).abs() < 1.0e-12);
    }
    let face_nodes = topology.faces[0].nodes();
    let face_offset = mesh.num_nodes() + 2 * topology.num_edges();
    let count = face_nodes.len() as f64;
    let face_coordinate = coordinates[face_offset];
    for direction in 0..3 {
        let expected = face_nodes
            .iter()
            .map(|&node| [mesh.nodes[node].x, mesh.nodes[node].y, mesh.nodes[node].z][direction])
            .sum::<f64>()
            / count;
        assert!((face_coordinate[direction] - expected).abs() < 1.0e-12);
    }
    let local = space.local_to_global(&mesh, 0);
    assert_eq!(local.len(), 20);
    assert_eq!(
        local
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        20
    );
    let tabulation = space.tabulate(&mesh, 0, 0.2, 0.2, 0.2);
    assert_eq!(tabulation.values.len(), 20);
    assert!(tabulation.values.iter().all(|value| value.is_finite()));
    assert_eq!(
        tabulation.hessians.len(),
        20 * tabulation.spatial_dim * tabulation.spatial_dim
    );
    for row in 0..tabulation.spatial_dim {
        for column in 0..tabulation.spatial_dim {
            let sum = (0..tabulation.n_dofs)
                .map(|dof| tabulation.hessian(dof, row, column))
                .sum::<f64>();
            assert!(sum.abs() < 1.0e-10, "Hessian partition error: {sum:e}");
        }
    }
}
