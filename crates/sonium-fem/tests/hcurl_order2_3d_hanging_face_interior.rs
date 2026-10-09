//! Independent affine tetrahedral order-2 H(curl) face-interior checks.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order2_hanging_tri_face_edge_constraints_3d,
    hcurl_order2_hanging_tri_face_interior_constraints_3d,
    hcurl_order2_hanging_tri_face_interior_mpc_system_3d,
};
use sonium_fem::mesh::{
    ElementType, Face, Mesh, MeshTopology, Point, unit_cube_hexahedra, unit_cube_tetrahedra,
    unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

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

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn reference_from_physical(mesh: &Mesh, element: usize, physical: [f64; 3]) -> [f64; 3] {
    let nodes = &mesh.elements[element].nodes;
    let origin = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let p3 = mesh.nodes[nodes[3]];
    let jacobian = [
        [p1.x - origin.x, p2.x - origin.x, p3.x - origin.x],
        [p1.y - origin.y, p2.y - origin.y, p3.y - origin.y],
        [p1.z - origin.z, p2.z - origin.z, p3.z - origin.z],
    ];
    let rhs = [
        physical[0] - origin.x,
        physical[1] - origin.y,
        physical[2] - origin.z,
    ];
    let determinant = jacobian[0][0]
        * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1])
        - jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0])
        + jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]);
    let c0 = [jacobian[0][0], jacobian[1][0], jacobian[2][0]];
    let c1 = [jacobian[0][1], jacobian[1][1], jacobian[2][1]];
    let c2 = [jacobian[0][2], jacobian[1][2], jacobian[2][2]];
    [
        dot(rhs, cross(c1, c2)) / determinant,
        dot(c0, cross(rhs, c2)) / determinant,
        dot(c0, cross(c1, rhs)) / determinant,
    ]
}

fn physical_from_reference(mesh: &Mesh, element: usize, reference: [f64; 3]) -> [f64; 3] {
    let nodes = &mesh.elements[element].nodes;
    let origin = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let p3 = mesh.nodes[nodes[3]];
    let mapped = [
        (p1.x - origin.x) * reference[0]
            + (p2.x - origin.x) * reference[1]
            + (p3.x - origin.x) * reference[2],
        (p1.y - origin.y) * reference[0]
            + (p2.y - origin.y) * reference[1]
            + (p3.y - origin.y) * reference[2],
        (p1.z - origin.z) * reference[0]
            + (p2.z - origin.z) * reference[1]
            + (p3.z - origin.z) * reference[2],
    ];
    [
        origin.x + mapped[0],
        origin.y + mapped[1],
        origin.z + mapped[2],
    ]
}

fn face_reference_point(face: usize, a: f64, b: f64) -> [f64; 3] {
    match face {
        0 => [a, b, 0.0],
        1 => [a, 0.0, b],
        2 => [0.0, a, b],
        3 => [a, b, 1.0 - a - b],
        _ => unreachable!(),
    }
}

fn face_reference_tangents(face: usize) -> ([f64; 3], [f64; 3]) {
    match face {
        0 => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        1 => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        2 => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        3 => ([1.0, 0.0, -1.0], [0.0, 1.0, -1.0]),
        _ => unreachable!(),
    }
}

fn face_edge_indices(topology: &MeshTopology, nodes: [usize; 3]) -> [usize; 3] {
    let pairs = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[0]),
    ];
    let mut edges = pairs.map(|(first, second)| {
        let key = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        topology.edges.iter().position(|&edge| edge == key).unwrap()
    });
    edges.sort_unstable();
    edges
}

fn owner_for_face(topology: &MeshTopology, face: usize) -> (usize, usize) {
    topology
        .element_faces
        .iter()
        .enumerate()
        .find_map(|(element, faces)| {
            faces
                .iter()
                .position(|&candidate| candidate == face)
                .map(|local_face| (element, local_face))
        })
        .expect("face owner")
}

fn coarse_edge_mode_moment(
    mesh: &Mesh,
    space: &HcurlSpace,
    element: usize,
    local_dof: usize,
    fine_edge: (usize, usize),
    fine_mode: usize,
) -> f64 {
    let first = mesh.nodes[fine_edge.0];
    let second = mesh.nodes[fine_edge.1];
    let direction = [second.x - first.x, second.y - first.y, second.z - first.z];
    let offset = 0.5 / 3.0_f64.sqrt();
    [(0.5 - offset, 0.5), (0.5 + offset, 0.5)]
        .into_iter()
        .map(|(parameter, weight)| {
            let physical = [
                first.x + parameter * direction[0],
                first.y + parameter * direction[1],
                first.z + parameter * direction[2],
            ];
            let reference = reference_from_physical(mesh, element, physical);
            let tabulation = space
                .try_tabulate(mesh, element, reference[0], reference[1], reference[2])
                .unwrap();
            let value = [
                tabulation.vector_value(local_dof, 0),
                tabulation.vector_value(local_dof, 1),
                tabulation.vector_value(local_dof, 2),
            ];
            let line_value = dot(value, direction);
            match fine_mode {
                0 => weight * line_value,
                1 => weight * 3.0 * line_value * (1.0 - 2.0 * parameter),
                _ => unreachable!(),
            }
        })
        .sum()
}

fn fine_edge_coefficients(
    mesh: &Mesh,
    space: &HcurlSpace,
    topology: &MeshTopology,
    coarse_element: usize,
    coarse_local_dofs: [usize; 8],
    fine_edges: [usize; 3],
) -> [[[f64; 8]; 2]; 3] {
    std::array::from_fn(|edge_index| {
        std::array::from_fn(|fine_mode| {
            std::array::from_fn(|coarse_mode| {
                coarse_edge_mode_moment(
                    mesh,
                    space,
                    coarse_element,
                    coarse_local_dofs[coarse_mode],
                    topology.edges[fine_edges[edge_index]],
                    fine_mode,
                )
            })
        })
    })
}

#[test]
fn order2_hcurl_hanging_face_interior_reconstructs_tangential_trace() {
    let mesh = hanging_face_mesh(0.2, 0.3);
    let space = HcurlSpace::order2();
    let constraints =
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    assert_eq!(constraints.len(), 3);
    let topology = MeshTopology::build(&mesh);
    let (coarse_element, coarse_local_face) = owner_for_face(&topology, constraints[0].coarse_face);
    let coarse_element_dofs = space.local_to_global(&mesh, coarse_element);
    let coarse_local_dofs = constraints[0].coarse_dofs.map(|dof| {
        coarse_element_dofs
            .iter()
            .position(|&candidate| candidate == dof)
            .unwrap()
    });
    let coarse_values = [2.0, -0.75, 1.25, -1.5, 0.5, 0.25, 1.0, -0.25];

    for constraint in &constraints {
        let (fine_element, fine_local_face) = owner_for_face(&topology, constraint.slave_face);
        let fine_face_nodes = match topology.face(constraint.slave_face) {
            Face::Triangle(nodes) => *nodes,
            Face::Quad(_) => panic!("expected triangular fine face"),
        };
        let fine_edges = face_edge_indices(&topology, fine_face_nodes);
        let fine_local_edges = topology.element_edge_indices(fine_element);
        let fine_edge_local_dofs = fine_edges.map(|edge| {
            let local_edge = fine_local_edges
                .iter()
                .position(|&candidate| candidate == edge)
                .unwrap();
            [2 * local_edge, 2 * local_edge + 1]
        });
        let edge_coefficients = fine_edge_coefficients(
            &mesh,
            &space,
            &topology,
            coarse_element,
            coarse_local_dofs,
            fine_edges,
        );
        let face_local_dofs = [12 + 2 * fine_local_face, 12 + 2 * fine_local_face + 1];
        let face_values = constraint.coefficients.map(|row| {
            row.into_iter()
                .zip(coarse_values)
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>()
        });
        let (tangent_a_ref, tangent_b_ref) = face_reference_tangents(fine_local_face);
        let fine_nodes = &mesh.elements[fine_element].nodes;
        let origin = mesh.nodes[fine_nodes[0]];
        let p1 = mesh.nodes[fine_nodes[1]];
        let p2 = mesh.nodes[fine_nodes[2]];
        let p3 = mesh.nodes[fine_nodes[3]];
        let tangent_a = [
            (p1.x - origin.x) * tangent_a_ref[0]
                + (p2.x - origin.x) * tangent_a_ref[1]
                + (p3.x - origin.x) * tangent_a_ref[2],
            (p1.y - origin.y) * tangent_a_ref[0]
                + (p2.y - origin.y) * tangent_a_ref[1]
                + (p3.y - origin.y) * tangent_a_ref[2],
            (p1.z - origin.z) * tangent_a_ref[0]
                + (p2.z - origin.z) * tangent_a_ref[1]
                + (p3.z - origin.z) * tangent_a_ref[2],
        ];
        let tangent_b = [
            (p1.x - origin.x) * tangent_b_ref[0]
                + (p2.x - origin.x) * tangent_b_ref[1]
                + (p3.x - origin.x) * tangent_b_ref[2],
            (p1.y - origin.y) * tangent_b_ref[0]
                + (p2.y - origin.y) * tangent_b_ref[1]
                + (p3.y - origin.y) * tangent_b_ref[2],
            (p1.z - origin.z) * tangent_b_ref[0]
                + (p2.z - origin.z) * tangent_b_ref[1]
                + (p3.z - origin.z) * tangent_b_ref[2],
        ];
        for (a, b) in [(0.2, 0.2), (0.6, 0.2), (0.2, 0.6)] {
            let fine_reference = face_reference_point(fine_local_face, a, b);
            let physical = physical_from_reference(&mesh, fine_element, fine_reference);
            let coarse_reference = reference_from_physical(&mesh, coarse_element, physical);
            let coarse_tabulation = space
                .try_tabulate(
                    &mesh,
                    coarse_element,
                    coarse_reference[0],
                    coarse_reference[1],
                    coarse_reference[2],
                )
                .unwrap();
            let fine_tabulation = space
                .try_tabulate(
                    &mesh,
                    fine_element,
                    fine_reference[0],
                    fine_reference[1],
                    fine_reference[2],
                )
                .unwrap();
            let mut coarse_value = [0.0; 3];
            for (local_dof, value) in coarse_local_dofs.into_iter().zip(coarse_values) {
                for (component, result) in coarse_value.iter_mut().enumerate() {
                    *result += value * coarse_tabulation.vector_value(local_dof, component);
                }
            }
            let mut fine_value = [0.0; 3];
            for edge_index in 0..3 {
                for fine_mode in 0..2 {
                    let value = edge_coefficients[edge_index][fine_mode]
                        .into_iter()
                        .zip(coarse_values)
                        .map(|(coefficient, coarse_value)| coefficient * coarse_value)
                        .sum::<f64>();
                    for (component, result) in fine_value.iter_mut().enumerate() {
                        *result += value
                            * fine_tabulation.vector_value(
                                fine_edge_local_dofs[edge_index][fine_mode],
                                component,
                            );
                    }
                }
            }
            for (local_dof, value) in face_local_dofs.into_iter().zip(face_values) {
                for (component, result) in fine_value.iter_mut().enumerate() {
                    *result += value * fine_tabulation.vector_value(local_dof, component);
                }
            }
            assert!((dot(coarse_value, tangent_a) - dot(fine_value, tangent_a)).abs() < 1.0e-8);
            assert!((dot(coarse_value, tangent_b) - dot(fine_value, tangent_b)).abs() < 1.0e-8);
        }
    }
    assert_eq!(
        coarse_local_face,
        owner_for_face(&topology, constraints[0].coarse_face).1
    );
}

#[test]
fn order2_hcurl_hanging_face_interior_mpcs_recover_complex_modes() {
    let mesh = hanging_face_mesh(1.0 / 3.0, 1.0 / 3.0);
    let space = HcurlSpace::order2();
    let constraints =
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    let mpc = hcurl_order2_hanging_tri_face_interior_mpc_system_3d(&mesh, &space, 1.0e-10).unwrap();
    let coarse_values = [
        Complex64::new(2.0, -1.0),
        Complex64::new(-0.75, 0.5),
        Complex64::new(1.25, 2.0),
        Complex64::new(-1.5, 0.25),
        Complex64::new(0.5, -2.0),
        Complex64::new(0.25, 0.75),
        Complex64::new(1.0, -0.5),
        Complex64::new(-0.25, 1.5),
    ];
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (dof, value) in constraints[0].coarse_dofs.into_iter().zip(coarse_values) {
        let reduced_dof = mpc
            .reduced_full_dofs()
            .iter()
            .position(|&candidate| candidate == dof)
            .expect("coarse master remains free");
        reduced[reduced_dof] = value;
    }
    let recovered = mpc.recover(&reduced).unwrap();
    for constraint in &constraints {
        for fine_mode in 0..2 {
            let expected = constraint.coefficients[fine_mode]
                .into_iter()
                .zip(coarse_values)
                .map(|(coefficient, value)| value * coefficient)
                .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
            assert!((recovered[constraint.slave_dofs[fine_mode]] - expected).norm() < 1.0e-8);
        }
    }
}

#[test]
fn order2_hcurl_hanging_face_interior_rejects_unsupported_inputs_and_conforming_meshes_are_noop() {
    let mesh = hanging_face_mesh(0.25, 0.25);
    let space = HcurlSpace::order2();
    assert!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(
            &unit_cube_tetrahedra(1),
            &space,
            1.0e-10,
        )
        .unwrap()
        .is_empty()
    );
    assert!(matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&mesh, &HcurlSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(1))
    ));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(
            &unit_square_triangles(1),
            &space,
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedDimension(2))
    ));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(
            &unit_cube_hexahedra(1),
            &space,
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedHcurlTopology)
    ));
    let mut malformed = mesh.clone();
    malformed.elements[0].nodes.push(malformed.nodes.len());
    malformed.add_node(Point::new_3d(0.1, 0.1, 0.1));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&malformed, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ));

    // Keep the order-2 edge packet in the same fixture independently covered.
    assert_eq!(
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&mesh, &space, 1.0e-10)
            .unwrap()
            .len(),
        3
    );
}
