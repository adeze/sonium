//! Independent affine tetrahedral order-2 H(curl) hanging-face checks.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order2_hanging_tri_face_edge_constraints_3d,
    hcurl_order2_hanging_tri_face_edge_mpc_system_3d,
};
use sonium_fem::mesh::{
    ElementType, Mesh, MeshTopology, Point, unit_cube_hexahedra, unit_cube_tetrahedra,
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

fn owner_for_face(topology: &MeshTopology, face: usize) -> usize {
    topology
        .element_faces
        .iter()
        .enumerate()
        .find_map(|(element, faces)| faces.contains(&face).then_some(element))
        .expect("face owner")
}

fn owner_for_edge(topology: &MeshTopology, edge: usize) -> (usize, usize) {
    topology
        .element_edge_indices(0)
        .iter()
        .position(|&candidate| candidate == edge)
        .map(|local_edge| (0, local_edge))
        .or_else(|| {
            (1..topology.element_edges.len()).find_map(|element| {
                topology
                    .element_edge_indices(element)
                    .iter()
                    .position(|&candidate| candidate == edge)
                    .map(|local_edge| (element, local_edge))
            })
        })
        .expect("edge owner")
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

#[test]
fn order2_hcurl_hanging_face_restrictions_match_independent_edge_moments() {
    let mesh = hanging_face_mesh(0.2, 0.3);
    let space = HcurlSpace::order2();
    let constraints =
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    assert_eq!(constraints.len(), 3);

    let topology = MeshTopology::build(&mesh);
    let owner = owner_for_face(&topology, constraints[0].coarse_face);
    let element_dofs = space.local_to_global(&mesh, owner);
    for constraint in &constraints {
        assert_eq!(constraint.coarse_dofs.len(), 8);
        for fine_mode in 0..2 {
            for (coarse_mode, &coarse_dof) in constraint.coarse_dofs.iter().enumerate() {
                let local_dof = element_dofs
                    .iter()
                    .position(|&dof| dof == coarse_dof)
                    .expect("coarse face master is owned by face owner");
                assert!(local_dof < 20);
                let expected = coarse_edge_mode_moment(
                    &mesh,
                    &space,
                    owner,
                    local_dof,
                    topology.edges[constraint.slave_edge],
                    fine_mode,
                );
                assert!(
                    (constraint.coefficients[fine_mode][coarse_mode] - expected).abs() < 1.0e-10,
                    "fine mode {fine_mode} coarse mode {coarse_mode}: {} != {expected}",
                    constraint.coefficients[fine_mode][coarse_mode]
                );
            }
        }
    }
}

#[test]
fn order2_hcurl_hanging_face_mpcs_recover_complex_edge_modes() {
    let mesh = hanging_face_mesh(1.0 / 3.0, 1.0 / 3.0);
    let space = HcurlSpace::order2();
    let constraints =
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    let mpc = hcurl_order2_hanging_tri_face_edge_mpc_system_3d(&mesh, &space, 1.0e-10).unwrap();
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
            .expect("coarse face master remains free");
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
            assert!((recovered[constraint.slave_edge * 2 + fine_mode] - expected).norm() < 1e-10);
        }
    }
}

#[test]
fn order2_hcurl_hanging_face_preserves_pointwise_tangential_trace() {
    let mesh = hanging_face_mesh(0.2, 0.3);
    let space = HcurlSpace::order2();
    let constraints =
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&mesh, &space, 1.0e-10).unwrap();
    let topology = MeshTopology::build(&mesh);
    let coarse_element = owner_for_face(&topology, constraints[0].coarse_face);
    let coarse_element_dofs = space.local_to_global(&mesh, coarse_element);
    let coarse_values = [2.0, -0.75, 1.25, -1.5, 0.5, 0.25, 1.0, -0.25];
    for constraint in &constraints {
        let (fine_element, fine_local_edge) = owner_for_edge(&topology, constraint.slave_edge);
        let fine_edge = topology.edges[constraint.slave_edge];
        let first = mesh.nodes[fine_edge.0];
        let second = mesh.nodes[fine_edge.1];
        let direction = [second.x - first.x, second.y - first.y, second.z - first.z];
        let coarse_local_dofs = constraint.coarse_dofs.map(|dof| {
            coarse_element_dofs
                .iter()
                .position(|&candidate| candidate == dof)
                .unwrap()
        });
        for parameter in [0.17, 0.43, 0.81] {
            let physical = [
                first.x + parameter * direction[0],
                first.y + parameter * direction[1],
                first.z + parameter * direction[2],
            ];
            let coarse_reference = reference_from_physical(&mesh, coarse_element, physical);
            let fine_reference = reference_from_physical(&mesh, fine_element, physical);
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
            let coarse_tangent = coarse_local_dofs
                .into_iter()
                .zip(coarse_values)
                .map(|(local_dof, value)| {
                    value
                        * (0..3)
                            .map(|component| {
                                coarse_tabulation.vector_value(local_dof, component)
                                    * direction[component]
                            })
                            .sum::<f64>()
                })
                .sum::<f64>();
            let fine_tangent = (0..2)
                .map(|fine_mode| {
                    let fine_value = constraint.coefficients[fine_mode]
                        .into_iter()
                        .zip(coarse_values)
                        .map(|(coefficient, value)| coefficient * value)
                        .sum::<f64>();
                    fine_value
                        * (0..3)
                            .map(|component| {
                                fine_tabulation
                                    .vector_value(2 * fine_local_edge + fine_mode, component)
                                    * direction[component]
                            })
                            .sum::<f64>()
                })
                .sum::<f64>();
            assert!((coarse_tangent - fine_tangent).abs() < 1.0e-9);
        }
    }
}

#[test]
fn order2_hcurl_hanging_face_rejects_unsupported_inputs_and_conforming_meshes_are_noop() {
    let mesh = hanging_face_mesh(0.25, 0.25);
    let space = HcurlSpace::order2();
    assert!(
        hcurl_order2_hanging_tri_face_edge_constraints_3d(
            &unit_cube_tetrahedra(1),
            &space,
            1.0e-10,
        )
        .unwrap()
        .is_empty()
    );
    assert!(matches!(
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&mesh, &HcurlSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(1))
    ));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_edge_constraints_3d(
            &unit_square_triangles(1),
            &space,
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedDimension(2))
    ));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&unit_cube_hexahedra(1), &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedHcurlTopology)
    ));

    let mut malformed = mesh.clone();
    malformed.elements[0].nodes.push(malformed.nodes.len());
    malformed.add_node(Point::new_3d(0.1, 0.1, 0.1));
    assert!(matches!(
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&malformed, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ));
}
