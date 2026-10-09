//! Release-facing affine tetrahedral order-1 H(curl) hanging-face contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order1_hanging_tri_face_constraints_3d,
    hcurl_order1_hanging_tri_face_mpc_system_3d,
};
use sonium_fem::mesh::{
    ElementType, Mesh, MeshTopology, Point, unit_cube_hexahedra, unit_cube_tetrahedra,
    unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order1-3d-hanging-face-contract.v1";

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

fn coarse_edge_line_moment(
    mesh: &Mesh,
    space: &HcurlSpace,
    element: usize,
    local_dof: usize,
    edge: (usize, usize),
) -> f64 {
    let first = mesh.nodes[edge.0];
    let second = mesh.nodes[edge.1];
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
                .map_err(|error| error.to_string())?;
            let value = [
                tabulation.vector_value(local_dof, 0),
                tabulation.vector_value(local_dof, 1),
                tabulation.vector_value(local_dof, 2),
            ];
            Ok(weight * dot(value, direction))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(|values| values.into_iter().sum())
        .unwrap_or(f64::NAN)
}

fn run() -> Result<(usize, usize, usize), String> {
    let mesh = hanging_face_mesh(0.2, 0.3);
    let space = HcurlSpace::new();
    let constraints = hcurl_order1_hanging_tri_face_constraints_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 3 {
        return Err(format!(
            "expected 3 interior face edges, got {}",
            constraints.len()
        ));
    }

    let topology = MeshTopology::build(&mesh);
    let coarse_face = constraints[0].coarse_face;
    let owner = owner_for_face(&topology, coarse_face);
    let local_edges = topology.element_edge_indices(owner);
    let mut moment_checks = 0;
    for constraint in &constraints {
        for (mode, &coarse_edge) in constraint.coarse_edges.iter().enumerate() {
            let local_dof = local_edges
                .iter()
                .position(|&edge| edge == coarse_edge)
                .ok_or_else(|| "coarse face edge is not owned by face owner".to_owned())?;
            let expected = coarse_edge_line_moment(
                &mesh,
                &space,
                owner,
                local_dof,
                topology.edges[constraint.slave_edge],
            );
            if !expected.is_finite() || (constraint.coefficients[mode] - expected).abs() > 1.0e-11 {
                return Err("independent tangential line-moment mismatch".to_owned());
            }
            moment_checks += 1;
        }
    }

    let mpc = hcurl_order1_hanging_tri_face_mpc_system_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_edges = constraints[0].coarse_edges;
    let coarse_values = [
        Complex64::new(2.0, -1.0),
        Complex64::new(-0.75, 0.5),
        Complex64::new(1.25, 2.0),
    ];
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (edge, value) in coarse_edges.into_iter().zip(coarse_values) {
        let reduced_dof = mpc
            .reduced_full_dofs()
            .iter()
            .position(|&dof| dof == edge)
            .ok_or_else(|| "coarse edge was unexpectedly eliminated".to_owned())?;
        reduced[reduced_dof] = value;
    }
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let mut recovery_checks = 0;
    for constraint in &constraints {
        let expected = constraint
            .coefficients
            .into_iter()
            .zip(coarse_values)
            .map(|(coefficient, value)| value * coefficient)
            .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
        if (recovered[constraint.slave_edge] - expected).norm() > 1.0e-11 {
            return Err("complex MPC recovery mismatch".to_owned());
        }
        recovery_checks += 1;
    }

    let mut rejection_checks = 0;
    if !hcurl_order1_hanging_tri_face_constraints_3d(&mesh, &HcurlSpace::order2(), 1.0e-10)
        .is_err_and(|error| matches!(error, MixedHangingError::UnsupportedOrder(2)))
    {
        return Err("order-2 input was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order1_hanging_tri_face_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ) {
        return Err("zero tolerance was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order1_hanging_tri_face_constraints_3d(&unit_square_triangles(1), &space, 1.0e-10,),
        Err(MixedHangingError::UnsupportedDimension(2))
    ) {
        return Err("2-D input was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order1_hanging_tri_face_constraints_3d(&unit_cube_hexahedra(1), &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedHcurlTopology)
    ) {
        return Err("hexahedral input was not rejected".to_owned());
    }
    rejection_checks += 1;
    let mut malformed = mesh.clone();
    malformed.elements[0].nodes.push(malformed.nodes.len());
    malformed.add_node(Point::new_3d(0.1, 0.1, 0.1));
    if !matches!(
        hcurl_order1_hanging_tri_face_constraints_3d(&malformed, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ) {
        return Err("malformed tetrahedral connectivity was not rejected".to_owned());
    }
    rejection_checks += 1;

    Ok((moment_checks, recovery_checks, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((moment_checks, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .metric("signed_line_moment_checks", moment_checks as f64)
            .metric("recovery_checks", recovery_checks as f64)
            .metric("rejection_checks", rejection_checks as f64)
            .dimension("3-D affine tetrahedral order-1 H(curl) triangular-face edge moments")
            .input_checksum("mesh:coarse-triangle-with-three-fine-subfaces:hcurl-order1-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hcurl-order1-3d-hanging-face-contract-v1",
            )
            .seed("fem-hcurl-order1-3d-hanging-face-contract:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
