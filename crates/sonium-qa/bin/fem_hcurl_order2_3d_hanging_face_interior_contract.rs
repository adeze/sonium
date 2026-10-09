//! Release-facing affine tetrahedral order-2 H(curl) face-interior contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order2_hanging_tri_face_edge_constraints_3d,
    hcurl_order2_hanging_tri_face_interior_constraints_3d,
    hcurl_order2_hanging_tri_face_interior_mpc_system_3d,
};
use sonium_fem::mesh::{
    ElementType, Face, Mesh, MeshTopology, Point, unit_cube_hexahedra, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order2-3d-hanging-face-interior-contract.v1";

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
    [
        origin.x
            + (p1.x - origin.x) * reference[0]
            + (p2.x - origin.x) * reference[1]
            + (p3.x - origin.x) * reference[2],
        origin.y
            + (p1.y - origin.y) * reference[0]
            + (p2.y - origin.y) * reference[1]
            + (p3.y - origin.y) * reference[2],
        origin.z
            + (p1.z - origin.z) * reference[0]
            + (p2.z - origin.z) * reference[1]
            + (p3.z - origin.z) * reference[2],
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

fn run() -> Result<(usize, usize, usize), String> {
    let mesh = hanging_face_mesh(0.2, 0.3);
    let space = HcurlSpace::order2();
    let constraints = hcurl_order2_hanging_tri_face_interior_constraints_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 3 {
        return Err(format!(
            "expected three fine face constraints, got {}",
            constraints.len()
        ));
    }

    let edge_constraints =
        hcurl_order2_hanging_tri_face_edge_constraints_3d(&mesh, &space, 1.0e-10)
            .map_err(|error| error.to_string())?;
    let topology = MeshTopology::build(&mesh);
    let coarse_owner = owner_for_face(&topology, constraints[0].coarse_face).0;
    let coarse_face_nodes = match topology.face(constraints[0].coarse_face) {
        Face::Triangle(nodes) => *nodes,
        Face::Quad(_) => return Err("coarse face is not triangular".to_owned()),
    };
    let coarse_face_edges = face_edge_indices(&topology, coarse_face_nodes);
    let coarse_element_dofs = space.local_to_global(&mesh, coarse_owner);
    let coarse_values = [2.0, -0.75, 1.25, -1.5, 0.5, 0.25, 1.0, -0.25];
    let mut pointwise_checks = 0;
    for constraint in &constraints {
        let (fine_element, fine_local_face) = owner_for_face(&topology, constraint.slave_face);
        let fine_element_dofs = space.local_to_global(&mesh, fine_element);
        let fine_nodes = match topology.face(constraint.slave_face) {
            Face::Triangle(nodes) => *nodes,
            Face::Quad(_) => return Err("fine face is not triangular".to_owned()),
        };
        let fine_edges = face_edge_indices(&topology, fine_nodes);
        let fine_local_edges = topology.element_edge_indices(fine_element);
        let (tangent_a_ref, tangent_b_ref) = face_reference_tangents(fine_local_face);
        let element_nodes = &mesh.elements[fine_element].nodes;
        let origin = mesh.nodes[element_nodes[0]];
        let p1 = mesh.nodes[element_nodes[1]];
        let p2 = mesh.nodes[element_nodes[2]];
        let p3 = mesh.nodes[element_nodes[3]];
        let map_tangent = |reference: [f64; 3]| {
            [
                (p1.x - origin.x) * reference[0]
                    + (p2.x - origin.x) * reference[1]
                    + (p3.x - origin.x) * reference[2],
                (p1.y - origin.y) * reference[0]
                    + (p2.y - origin.y) * reference[1]
                    + (p3.y - origin.y) * reference[2],
                (p1.z - origin.z) * reference[0]
                    + (p2.z - origin.z) * reference[1]
                    + (p3.z - origin.z) * reference[2],
            ]
        };
        let tangent_a = map_tangent(tangent_a_ref);
        let tangent_b = map_tangent(tangent_b_ref);
        let coarse_local_dofs = constraint.coarse_dofs.map(|dof| {
            coarse_element_dofs
                .iter()
                .position(|&candidate| candidate == dof)
                .expect("coarse master owner")
        });
        let face_values = constraint.coefficients.map(|row| {
            row.into_iter()
                .zip(coarse_values)
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>()
        });
        for (a, b) in [(0.2, 0.2), (0.6, 0.2), (0.2, 0.6)] {
            let fine_reference = face_reference_point(fine_local_face, a, b);
            let physical = physical_from_reference(&mesh, fine_element, fine_reference);
            let coarse_reference = reference_from_physical(&mesh, coarse_owner, physical);
            let coarse_tabulation = space
                .try_tabulate(
                    &mesh,
                    coarse_owner,
                    coarse_reference[0],
                    coarse_reference[1],
                    coarse_reference[2],
                )
                .map_err(|error| error.to_string())?;
            let fine_tabulation = space
                .try_tabulate(
                    &mesh,
                    fine_element,
                    fine_reference[0],
                    fine_reference[1],
                    fine_reference[2],
                )
                .map_err(|error| error.to_string())?;
            let mut coarse_value = [0.0; 3];
            for (local_dof, value) in coarse_local_dofs.into_iter().zip(coarse_values) {
                for (component, result) in coarse_value.iter_mut().enumerate() {
                    *result += value * coarse_tabulation.vector_value(local_dof, component);
                }
            }
            let mut fine_value = [0.0; 3];
            for edge_index in 0..3 {
                let local_edge = fine_local_edges
                    .iter()
                    .position(|&candidate| candidate == fine_edges[edge_index])
                    .ok_or_else(|| "fine face edge owner mismatch".to_owned())?;
                for fine_mode in 0..2 {
                    let value = if let Some(coarse_edge_mode) = coarse_face_edges
                        .iter()
                        .position(|&candidate| candidate == fine_edges[edge_index])
                    {
                        coarse_values[2 * coarse_edge_mode + fine_mode]
                    } else {
                        let edge = edge_constraints
                            .iter()
                            .find(|candidate| candidate.slave_edge == fine_edges[edge_index])
                            .ok_or_else(|| "missing fine-edge trace constraint".to_owned())?;
                        edge.coefficients[fine_mode]
                            .into_iter()
                            .zip(coarse_values)
                            .map(|(coefficient, coarse_value)| coefficient * coarse_value)
                            .sum::<f64>()
                    };
                    for (component, result) in fine_value.iter_mut().enumerate() {
                        *result += value
                            * fine_tabulation.vector_value(2 * local_edge + fine_mode, component);
                    }
                }
            }
            for (local_dof, value) in [12 + 2 * fine_local_face, 13 + 2 * fine_local_face]
                .into_iter()
                .zip(face_values)
            {
                for (component, result) in fine_value.iter_mut().enumerate() {
                    *result += value * fine_tabulation.vector_value(local_dof, component);
                }
            }
            if (dot(coarse_value, tangent_a) - dot(fine_value, tangent_a)).abs() > 1.0e-8
                || (dot(coarse_value, tangent_b) - dot(fine_value, tangent_b)).abs() > 1.0e-8
            {
                return Err("pointwise tangential face trace mismatch".to_owned());
            }
            pointwise_checks += 2;
        }
        if !fine_element_dofs.contains(&constraint.slave_dofs[0]) {
            return Err("fine face slave DOF is not owned by face owner".to_owned());
        }
    }

    let mpc = hcurl_order2_hanging_tri_face_interior_mpc_system_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_values_complex = [
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
    for (dof, value) in constraints[0]
        .coarse_dofs
        .into_iter()
        .zip(coarse_values_complex)
    {
        let reduced_dof = mpc
            .reduced_full_dofs()
            .iter()
            .position(|&candidate| candidate == dof)
            .ok_or_else(|| "coarse master unexpectedly eliminated".to_owned())?;
        reduced[reduced_dof] = value;
    }
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let mut recovery_checks = 0;
    for constraint in &constraints {
        for fine_mode in 0..2 {
            let expected = constraint.coefficients[fine_mode]
                .into_iter()
                .zip(coarse_values_complex)
                .map(|(coefficient, value)| value * coefficient)
                .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
            if (recovered[constraint.slave_dofs[fine_mode]] - expected).norm() > 1.0e-8 {
                return Err("complex face-interior MPC recovery mismatch".to_owned());
            }
            recovery_checks += 1;
        }
    }

    let mut rejection_checks = 0;
    if !matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&mesh, &HcurlSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(1))
    ) {
        return Err("order-1 input was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ) {
        return Err("zero tolerance was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(
            &unit_square_triangles(1),
            &space,
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedDimension(2))
    ) {
        return Err("2-D input was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(
            &unit_cube_hexahedra(1),
            &space,
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedHcurlTopology)
    ) {
        return Err("hexahedral input was not rejected".to_owned());
    }
    rejection_checks += 1;
    let mut malformed = mesh.clone();
    malformed.elements[0].nodes.push(malformed.nodes.len());
    malformed.add_node(Point::new_3d(0.1, 0.1, 0.1));
    if !matches!(
        hcurl_order2_hanging_tri_face_interior_constraints_3d(&malformed, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ) {
        return Err("malformed tetrahedral connectivity was not rejected".to_owned());
    }
    rejection_checks += 1;

    Ok((pointwise_checks, recovery_checks, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((pointwise_checks, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .metric("pointwise_tangential_checks", pointwise_checks as f64)
            .metric("recovery_checks", recovery_checks as f64)
            .metric("rejection_checks", rejection_checks as f64)
            .dimension("3-D affine tetrahedral order-2 H(curl) triangular-face interior modes")
            .input_checksum("mesh:coarse-triangle-with-three-fine-subfaces:hcurl-order2-face-interior-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hcurl-order2-3d-hanging-face-interior-contract-v1",
            )
            .seed("fem-hcurl-order2-3d-hanging-face-interior-contract:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
