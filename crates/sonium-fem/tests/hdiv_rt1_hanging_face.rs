//! Independent affine tetrahedral RT1 H(div) hanging-face contract checks.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt1_hanging_tri_face_constraints_3d,
    hdiv_rt1_hanging_tri_face_mpc_system_3d,
};
use sonium_fem::mesh::{
    ElementType, Face, Mesh, MeshTopology, Point, unit_cube_hexahedra, unit_cube_tetrahedra,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn hanging_face_mesh(center_x: f64, center_y: f64) -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(center_x, center_y, -1.0),
        Point::new_3d(center_x, center_y, 0.0),
        Point::new_3d(center_x, center_y, 1.0),
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
fn rt1_hdiv_hanging_faces_have_three_mode_restrictions() {
    let mesh = hanging_face_mesh(1.0 / 3.0, 1.0 / 3.0);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let constraints = hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &space, 1.0e-10).unwrap();

    assert_eq!(constraints.len(), 3);
    let coarse_face = constraints[0].coarse_face;
    assert!(
        constraints
            .iter()
            .all(|constraint| constraint.coarse_face == coarse_face)
    );
    assert!(
        constraints
            .iter()
            .all(|constraint| (constraint.signed_fraction.abs() - 1.0 / 3.0).abs() < 1.0e-12)
    );
    assert!(
        constraints
            .iter()
            .flat_map(|constraint| constraint.coefficients.into_iter().flatten())
            .all(f64::is_finite)
    );
}

#[test]
fn rt1_hdiv_hanging_face_mpcs_recover_all_three_moments() {
    let mesh = hanging_face_mesh(1.0 / 3.0, 1.0 / 3.0);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let constraints = hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    let mpc = hdiv_rt1_hanging_tri_face_mpc_system_3d(&mesh, &space, 1.0e-10).unwrap();
    let coarse_face = constraints[0].coarse_face;
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    let coarse_values = [
        Complex64::new(2.0, -1.0),
        Complex64::new(-3.0, 0.5),
        Complex64::new(1.25, 2.0),
    ];
    for (mode, value) in coarse_values.into_iter().enumerate() {
        let full_dof = coarse_face * 3 + mode;
        let reduced_dof = mpc
            .reduced_full_dofs()
            .iter()
            .position(|&dof| dof == full_dof)
            .expect("coarse face mode remains free");
        reduced[reduced_dof] = value;
    }
    let recovered = mpc.recover(&reduced).unwrap();
    let mut recovery_checks = 0;
    for constraint in constraints {
        for fine_mode in 0..3 {
            let expected = (0..3)
                .map(|coarse_mode| {
                    constraint.coefficients[fine_mode][coarse_mode] * coarse_values[coarse_mode]
                })
                .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
            assert!((recovered[constraint.slave_face * 3 + fine_mode] - expected).norm() < 1.0e-11);
            recovery_checks += 1;
        }
    }
    assert_eq!(recovery_checks, 9);
}

#[test]
fn rt1_hdiv_hanging_face_matrices_preserve_pointwise_normal_traces() {
    let mesh = hanging_face_mesh(1.0 / 3.0, 1.0 / 3.0);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let constraints = hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    let topology = MeshTopology::build(&mesh);
    let coarse_face = constraints[0].coarse_face;
    let coarse_element = topology
        .element_face_indices(0)
        .iter()
        .position(|&face| face == coarse_face)
        .map(|_| 0)
        .expect("coarse face belongs to the coarse tetrahedron");
    let coarse_local_face = topology
        .element_face_indices(coarse_element)
        .iter()
        .position(|&face| face == coarse_face)
        .unwrap();
    assert_eq!(coarse_local_face, 0);

    for constraint in constraints {
        let (fine_element, fine_local_face) = topology
            .element_faces
            .iter()
            .enumerate()
            .find_map(|(element, faces)| {
                faces
                    .iter()
                    .position(|&face| face == constraint.slave_face)
                    .map(|local_face| (element, local_face))
            })
            .expect("fine face owner");
        assert_eq!(fine_local_face, 0);
        let Face::Triangle(_) = topology.face(constraint.slave_face) else {
            panic!("RT1 face contract requires triangular faces");
        };
        let element = &mesh.elements[fine_element];
        let p0 = mesh.nodes[element.nodes[0]];
        let p1 = mesh.nodes[element.nodes[1]];
        let p2 = mesh.nodes[element.nodes[2]];
        let sample_a = 0.2;
        let sample_b = 0.3;
        let physical = [
            p0.x + sample_a * (p1.x - p0.x) + sample_b * (p2.x - p0.x),
            p0.y + sample_a * (p1.y - p0.y) + sample_b * (p2.y - p0.y),
            p0.z + sample_a * (p1.z - p0.z) + sample_b * (p2.z - p0.z),
        ];
        let fine_tabulation = space.tabulate(&mesh, fine_element, sample_a, sample_b, 0.0);
        let coarse_tabulation =
            space.tabulate(&mesh, coarse_element, physical[1], physical[0], physical[2]);
        let q0 = mesh.nodes[element.nodes[0]];
        let q1 = mesh.nodes[element.nodes[1]];
        let q2 = mesh.nodes[element.nodes[2]];
        let opposite = mesh.nodes[element.nodes[3]];
        let edge_a = [q1.x - q0.x, q1.y - q0.y, q1.z - q0.z];
        let edge_b = [q2.x - q0.x, q2.y - q0.y, q2.z - q0.z];
        let mut normal = [
            edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
            edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
            edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
        ];
        let toward_opposite = [opposite.x - q0.x, opposite.y - q0.y, opposite.z - q0.z];
        if normal[0] * toward_opposite[0]
            + normal[1] * toward_opposite[1]
            + normal[2] * toward_opposite[2]
            > 0.0
        {
            normal = [-normal[0], -normal[1], -normal[2]];
        }
        let normal_length =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        for coarse_mode in 0..3 {
            let coarse_value = (0..3)
                .map(|component| {
                    coarse_tabulation.vector_value(3 * coarse_local_face + coarse_mode, component)
                        * normal[component]
                })
                .sum::<f64>()
                / normal_length;
            let fine_value = (0..3)
                .map(|component| {
                    (0..3)
                        .map(|fine_mode| {
                            constraint.coefficients[fine_mode][coarse_mode]
                                * fine_tabulation
                                    .vector_value(3 * fine_local_face + fine_mode, component)
                        })
                        .sum::<f64>()
                        * normal[component]
                })
                .sum::<f64>()
                / normal_length;
            assert!(
                (coarse_value - fine_value).abs() < 1.0e-10,
                "coarse mode {coarse_mode} fine face {} normal trace mismatch: {coarse_value} != {fine_value}",
                constraint.slave_face
            );
        }
    }
}

#[test]
fn rt1_hdiv_hanging_faces_support_nonuniform_subfaces() {
    let mesh = hanging_face_mesh(0.2, 0.3);
    let constraints = hdiv_rt1_hanging_tri_face_constraints_3d(
        &mesh,
        &HdivSpace::with_order(HdivOrder::Rt1),
        1.0e-10,
    )
    .unwrap();
    assert_eq!(constraints.len(), 3);
    let fractions = constraints
        .iter()
        .map(|constraint| constraint.signed_fraction.abs())
        .collect::<Vec<_>>();
    assert!(
        fractions
            .iter()
            .any(|fraction| (*fraction - 0.2).abs() < 1.0e-12)
    );
    assert!(
        fractions
            .iter()
            .any(|fraction| (*fraction - 0.3).abs() < 1.0e-12)
    );
    assert!(
        fractions
            .iter()
            .any(|fraction| (*fraction - 0.5).abs() < 1.0e-12)
    );
}

#[test]
fn rt1_hdiv_hanging_faces_reject_unsupported_inputs_and_noop_conforming_mesh() {
    let mesh = hanging_face_mesh(1.0 / 3.0, 1.0 / 3.0);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    assert!(matches!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &HdivSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(1))
    ));
    assert!(matches!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ));
    let two_d = {
        let mut two_d = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
        ] {
            two_d.add_node(point);
        }
        two_d.add_element(ElementType::Triangle, vec![0, 1, 2]);
        two_d
    };
    assert!(matches!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&two_d, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(2))
    ));
    assert!(matches!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&unit_cube_hexahedra(1), &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedFaceTopology)
    ));
    assert!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&unit_cube_tetrahedra(1), &space, 1.0e-10)
            .unwrap()
            .is_empty()
    );
}
