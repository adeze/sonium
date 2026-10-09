//! Independent shared-trace and orientation evidence for tetrahedral H1/P3.
//!
//! The release-facing P3 case checks cubic reproduction at interior points.  It
//! does not make the entity contract explicit: the two edge nodes must follow
//! the global edge direction, and the one face-interior node must be shared by
//! both tetrahedra regardless of the local face ordering.  This test derives
//! the physical locations and the face sample points from mesh geometry rather
//! than from the P3 basis implementation, then evaluates the production space
//! on both sides of every interior face.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space};

const TRACE_TOLERANCE: f64 = 2.0e-11;
const GEOMETRY_TOLERANCE: f64 = 2.0e-13;

fn cubic([x, y, z]: [f64; 3]) -> f64 {
    0.23 + 0.41 * x - 0.37 * y + 0.29 * z + 0.62 * x * x - 0.53 * x * y
        + 0.47 * x * z
        + 0.71 * y * y
        - 0.31 * y * z
        + 0.38 * z * z
        + 0.19 * x * x * x
        - 0.17 * x * x * y
        + 0.13 * x * y * y
        + 0.11 * y * y * y
        + 0.07 * x * x * z
        - 0.09 * x * y * z
        + 0.05 * y * y * z
        + 0.03 * x * z * z
        - 0.04 * y * z * z
        + 0.02 * z * z * z
}

fn point_coordinates(mesh: &Mesh, node: usize) -> [f64; 3] {
    let point = mesh.nodes[node];
    [point.x, point.y, point.z]
}

fn affine_combination(points: &[[f64; 3]], weights: &[f64]) -> [f64; 3] {
    assert_eq!(points.len(), weights.len());
    let mut result = [0.0; 3];
    for (point, &weight) in points.iter().zip(weights) {
        for direction in 0..3 {
            result[direction] += weight * point[direction];
        }
    }
    result
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn replace_column(matrix: [[f64; 3]; 3], column: usize, values: [f64; 3]) -> [[f64; 3]; 3] {
    let mut replaced = matrix;
    for row in 0..3 {
        replaced[row][column] = values[row];
    }
    replaced
}

/// Compute barycentric coordinates from physical coordinates independently of
/// the FEM Jacobian/tabulation path.
fn barycentric_coordinates(mesh: &Mesh, element: usize, point: [f64; 3]) -> [f64; 4] {
    let vertices = mesh.elements[element].vertices();
    assert_eq!(vertices.len(), 4);
    let p0 = point_coordinates(mesh, vertices[0]);
    let p1 = point_coordinates(mesh, vertices[1]);
    let p2 = point_coordinates(mesh, vertices[2]);
    let p3 = point_coordinates(mesh, vertices[3]);
    let matrix = [
        [p1[0] - p0[0], p2[0] - p0[0], p3[0] - p0[0]],
        [p1[1] - p0[1], p2[1] - p0[1], p3[1] - p0[1]],
        [p1[2] - p0[2], p2[2] - p0[2], p3[2] - p0[2]],
    ];
    let rhs = [point[0] - p0[0], point[1] - p0[1], point[2] - p0[2]];
    let det = determinant(matrix);
    assert!(
        det.abs() > 1.0e-14,
        "degenerate tetrahedron in reference mesh"
    );
    let lambda = [
        determinant(replace_column(matrix, 0, rhs)) / det,
        determinant(replace_column(matrix, 1, rhs)) / det,
        determinant(replace_column(matrix, 2, rhs)) / det,
    ];
    [
        1.0 - lambda.iter().sum::<f64>(),
        lambda[0],
        lambda[1],
        lambda[2],
    ]
}

fn evaluate_global_field(
    mesh: &Mesh,
    space: &H1Space,
    coefficients: &[f64],
    element: usize,
    point: [f64; 3],
) -> f64 {
    let barycentric = barycentric_coordinates(mesh, element, point);
    assert!(barycentric.iter().all(|&coordinate| coordinate >= -1.0e-12));
    assert!((barycentric.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    let tabulation = space.tabulate(
        mesh,
        element,
        barycentric[1],
        barycentric[2],
        barycentric[3],
    );
    space
        .local_to_global(mesh, element)
        .iter()
        .enumerate()
        .map(|(local, &global)| tabulation.value(local) * coefficients[global])
        .sum()
}

/// Keep the physical mesh fixed while deliberately reversing one tetrahedron's
/// local vertex order.  This forces shared edges/faces to exercise the
/// orientation adapters instead of relying on the generator's canonical
/// ordering.
fn reversed_orientation_reference_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(0.0, 0.0, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    // The second tet has positive volume and shares the [0, 1, 2] face, but
    // traverses its shared edge (0, 1) in the reverse local direction.
    mesh.add_element(ElementType::Tetrahedron, vec![1, 0, 2, 4]);
    mesh.detect_boundaries();
    mesh
}

#[cfg_attr(test, test)]
pub fn p3_tetra_edge_modes_follow_global_direction_on_every_element() {
    let mesh = reversed_orientation_reference_mesh();
    let topology = MeshTopology::build(&mesh);
    let space = H1Space::new(PolynomialDegree::P3);
    space.validate(&mesh).expect("P3 tetrahedral H1 admission");
    let coordinates = space
        .dof_coordinates(&mesh)
        .expect("P3 global DOF coordinates are required");

    let mut edge_use_count = vec![0usize; topology.num_edges()];
    for element in 0..mesh.num_elements() {
        for &global_edge in topology.element_edge_indices(element) {
            edge_use_count[global_edge] += 1;
        }
    }
    let mut saw_reversed_shared_edge = false;
    for element in 0..mesh.num_elements() {
        let local_dofs = space.local_to_global(&mesh, element);
        let edge_indices = topology.element_edge_indices(element);
        let edge_signs = topology.element_edge_signs(element);
        assert_eq!(edge_indices.len(), 6);
        assert_eq!(edge_signs.len(), 6);

        for local_edge in 0..6 {
            let global_edge = edge_indices[local_edge];
            edge_use_count[global_edge] += 1;
            let (start, end) = topology.edges[global_edge];
            let start_point = point_coordinates(&mesh, start);
            let end_point = point_coordinates(&mesh, end);
            let expected_global_positions = if edge_signs[local_edge] > 0 {
                [
                    affine_combination(&[start_point, end_point], &[2.0 / 3.0, 1.0 / 3.0]),
                    affine_combination(&[start_point, end_point], &[1.0 / 3.0, 2.0 / 3.0]),
                ]
            } else {
                saw_reversed_shared_edge |= edge_use_count[global_edge] > 1;
                [
                    affine_combination(&[start_point, end_point], &[1.0 / 3.0, 2.0 / 3.0]),
                    affine_combination(&[start_point, end_point], &[2.0 / 3.0, 1.0 / 3.0]),
                ]
            };
            let local_first = local_dofs[4 + 2 * local_edge];
            let local_second = local_dofs[4 + 2 * local_edge + 1];
            for (dof, expected) in [
                (local_first, expected_global_positions[0]),
                (local_second, expected_global_positions[1]),
            ] {
                let actual = coordinates[dof];
                let residual = actual
                    .iter()
                    .zip(expected)
                    .map(|(&actual, expected)| (actual - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    residual < GEOMETRY_TOLERANCE,
                    "element {element} local edge {local_edge} maps DOF {dof} to {actual:?}, expected {expected:?}, residual {residual:e}"
                );
            }
        }
    }

    assert!(
        edge_use_count.iter().any(|&count| count > 1),
        "reference mesh must contain shared tetrahedral edges"
    );
    assert!(
        saw_reversed_shared_edge,
        "reference mesh must exercise a reversed shared-edge orientation"
    );
}

#[cfg_attr(test, test)]
pub fn p3_tetra_shared_face_traces_match_across_local_orientations() {
    let mesh = reversed_orientation_reference_mesh();
    let topology = MeshTopology::build(&mesh);
    let space = H1Space::new(PolynomialDegree::P3);
    space.validate(&mesh).expect("P3 tetrahedral H1 admission");
    let coordinates = space
        .dof_coordinates(&mesh)
        .expect("P3 global DOF coordinates are required");
    let coefficients: Vec<f64> = coordinates.iter().copied().map(cubic).collect();

    let mut face_elements = vec![Vec::new(); topology.num_faces()];
    for element in 0..mesh.num_elements() {
        for &face in topology.element_face_indices(element) {
            face_elements[face].push(element);
        }
    }

    let samples = [[0.17, 0.29, 0.54], [0.11, 0.37, 0.52], [0.23, 0.31, 0.46]];
    let mut shared_faces = 0;
    let mut maximum_trace_residual: f64 = 0.0;
    for (face, elements) in face_elements.iter().enumerate() {
        if elements.len() != 2 {
            continue;
        }
        shared_faces += 1;
        let first = elements[0];
        let second = elements[1];
        let face_nodes = topology.face(face).nodes();
        assert_eq!(face_nodes.len(), 3);
        let face_points: Vec<[f64; 3]> = face_nodes
            .iter()
            .map(|&node| point_coordinates(&mesh, node))
            .collect();
        for weights in samples {
            let point = affine_combination(&face_points, &weights);
            let first_value = evaluate_global_field(&mesh, &space, &coefficients, first, point);
            let second_value = evaluate_global_field(&mesh, &space, &coefficients, second, point);
            let exact = cubic(point);
            maximum_trace_residual = maximum_trace_residual
                .max((first_value - second_value).abs())
                .max((first_value - exact).abs())
                .max((second_value - exact).abs());
        }
    }

    assert!(
        shared_faces > 0,
        "reference mesh must contain an interior face"
    );
    assert!(
        maximum_trace_residual < TRACE_TOLERANCE,
        "P3 shared-face trace residual was {maximum_trace_residual:e}"
    );
}

/// Run the complete independent P3 trace/orientation reference packet.
#[test]
fn test_p3_tetra_edge_modes_follow_global_direction_on_every_element() {
    p3_tetra_edge_modes_follow_global_direction_on_every_element();
}

#[test]
fn test_p3_tetra_shared_face_traces_match_across_local_orientations() {
    p3_tetra_shared_face_traces_match_across_local_orientations();
}

pub fn p3_tetra_trace_orientation_reference() {
    p3_tetra_edge_modes_follow_global_direction_on_every_element();
    p3_tetra_shared_face_traces_match_across_local_orientations();
}
