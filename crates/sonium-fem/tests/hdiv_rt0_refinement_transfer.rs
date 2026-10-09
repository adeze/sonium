use sonium_fem::mesh::{
    ElementType, InterpolationError, Mesh, MeshTopology, Point, transfer_hdiv_rt0_state_checked,
    unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
};
use sonium_fem::quadrature::{gauss_legendre_1d, gauss_triangle};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn affine_hdiv_field(point: Point) -> [f64; 3] {
    [
        1.0 + 0.25 * point.x,
        2.0 + 0.25 * point.y,
        3.0 + 0.25 * point.z,
    ]
}

fn same_edge(left: (usize, usize), right: (usize, usize)) -> bool {
    (left.0 == right.0 && left.1 == right.1) || (left.0 == right.1 && left.1 == right.0)
}

fn edge_owner(mesh: &Mesh, edge: (usize, usize)) -> (usize, usize) {
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        for (local_edge, &(a, b)) in element.element_type.local_edges().iter().enumerate() {
            if same_edge((vertices[a], vertices[b]), edge) {
                return (element_index, local_edge);
            }
        }
    }
    panic!("edge has no owner: {edge:?}");
}

fn edge_normal_jacobian(p0: Point, p1: Point, centroid: Point) -> [f64; 3] {
    let tangent = [p1.x - p0.x, p1.y - p0.y];
    let midpoint = [(p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5];
    let to_centroid = [centroid.x - midpoint[0], centroid.y - midpoint[1]];
    let mut normal = [tangent[1] * 0.5, -tangent[0] * 0.5, 0.0];
    if normal[0] * to_centroid[0] + normal[1] * to_centroid[1] > 0.0 {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
    }
    normal
}

fn edge_point(p0: Point, p1: Point, coordinate: f64) -> Point {
    let left = 0.5 * (1.0 - coordinate);
    let right = 0.5 * (1.0 + coordinate);
    Point::new_3d(
        left * p0.x + right * p1.x,
        left * p0.y + right * p1.y,
        left * p0.z + right * p1.z,
    )
}

fn simplex_centroid(mesh: &Mesh, element_index: usize) -> Point {
    let vertices = mesh.elements[element_index].vertices();
    let scale = 1.0 / vertices.len() as f64;
    let mut centroid = Point::new_3d(0.0, 0.0, 0.0);
    for &node in vertices {
        let point = mesh.nodes[node];
        centroid.x += scale * point.x;
        centroid.y += scale * point.y;
        centroid.z += scale * point.z;
    }
    centroid
}

fn analytic_edge_state(mesh: &Mesh, space: &HdivSpace) -> Vec<f64> {
    let topology = MeshTopology::build(mesh);
    let quadrature = gauss_legendre_1d(8);
    topology
        .edges
        .iter()
        .enumerate()
        .map(|(edge_index, &edge)| {
            let (element_index, local_edge) = edge_owner(mesh, edge);
            let vertices = mesh.elements[element_index].vertices();
            let &(a, b) = &mesh.elements[element_index].element_type.local_edges()[local_edge];
            let p0 = mesh.nodes[vertices[a]];
            let p1 = mesh.nodes[vertices[b]];
            let normal = edge_normal_jacobian(p0, p1, simplex_centroid(mesh, element_index));
            let local_dofs = space.local_to_global(mesh, element_index);
            let local_dof = local_dofs
                .iter()
                .position(|&dof| dof == edge_index)
                .unwrap();
            let mut flux = 0.0;
            let mut normalization = 0.0;
            for point in &quadrature {
                let coordinate = point.xi();
                let lambda_a = 0.5 * (1.0 - coordinate);
                let lambda_b = 0.5 * (1.0 + coordinate);
                let mut barycentric = [0.0; 3];
                barycentric[a] = lambda_a;
                barycentric[b] = lambda_b;
                let tabulation = space
                    .try_tabulate(mesh, element_index, barycentric[1], barycentric[2], 0.0)
                    .unwrap();
                let basis = [
                    tabulation.vector_value(local_dof, 0),
                    tabulation.vector_value(local_dof, 1),
                    0.0,
                ];
                let field = affine_hdiv_field(edge_point(p0, p1, coordinate));
                flux += point.weight
                    * (field[0] * normal[0] + field[1] * normal[1] + field[2] * normal[2]);
                normalization += point.weight
                    * (basis[0] * normal[0] + basis[1] * normal[1] + basis[2] * normal[2]);
            }
            assert!(
                normalization.abs() > 1.0e-12,
                "edge {edge_index} normalization"
            );
            flux / normalization
        })
        .collect()
}

fn same_face(left: &[usize], right: &[usize]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn face_owner(mesh: &Mesh, face_index: usize) -> (usize, usize) {
    let topology = MeshTopology::build(mesh);
    let face_nodes = topology.faces[face_index].nodes();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        for (local_face, local_vertices) in element.element_type.local_faces().iter().enumerate() {
            let candidate: Vec<_> = local_vertices
                .iter()
                .map(|&index| vertices[index])
                .collect();
            if same_face(&candidate, &face_nodes) {
                return (element_index, local_face);
            }
        }
    }
    panic!("face has no owner: {face_index}");
}

fn face_surface_jacobian(face: [Point; 3], centroid: Point) -> [f64; 3] {
    let ab = [
        face[1].x - face[0].x,
        face[1].y - face[0].y,
        face[1].z - face[0].z,
    ];
    let ac = [
        face[2].x - face[0].x,
        face[2].y - face[0].y,
        face[2].z - face[0].z,
    ];
    let mut surface = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let face_centroid = Point::new_3d(
        (face[0].x + face[1].x + face[2].x) / 3.0,
        (face[0].y + face[1].y + face[2].y) / 3.0,
        (face[0].z + face[1].z + face[2].z) / 3.0,
    );
    let to_centroid = [
        centroid.x - face_centroid.x,
        centroid.y - face_centroid.y,
        centroid.z - face_centroid.z,
    ];
    if surface[0] * to_centroid[0] + surface[1] * to_centroid[1] + surface[2] * to_centroid[2] > 0.0
    {
        surface.iter_mut().for_each(|value| *value = -*value);
    }
    surface
}

fn face_point(face: [Point; 3], xi: f64, eta: f64) -> Point {
    Point::new_3d(
        face[0].x + xi * (face[1].x - face[0].x) + eta * (face[2].x - face[0].x),
        face[0].y + xi * (face[1].y - face[0].y) + eta * (face[2].y - face[0].y),
        face[0].z + xi * (face[1].z - face[0].z) + eta * (face[2].z - face[0].z),
    )
}

fn analytic_face_state(mesh: &Mesh, space: &HdivSpace) -> Vec<f64> {
    let topology = MeshTopology::build(mesh);
    let quadrature = gauss_triangle(4);
    topology
        .faces
        .iter()
        .enumerate()
        .map(|(face_index, face_key)| {
            let (element_index, local_face) = face_owner(mesh, face_index);
            let vertices = mesh.elements[element_index].vertices();
            let local_vertices =
                mesh.elements[element_index].element_type.local_faces()[local_face];
            let face = [
                mesh.nodes[vertices[local_vertices[0]]],
                mesh.nodes[vertices[local_vertices[1]]],
                mesh.nodes[vertices[local_vertices[2]]],
            ];
            let surface = face_surface_jacobian(face, simplex_centroid(mesh, element_index));
            let local_dofs = space.local_to_global(mesh, element_index);
            let local_dof = local_dofs
                .iter()
                .position(|&dof| dof == face_index)
                .unwrap();
            let mut flux = 0.0;
            let mut normalization = 0.0;
            for point in &quadrature {
                let xi = point.xi();
                let eta = point.eta();
                let mut barycentric = [0.0; 4];
                barycentric[local_vertices[0]] = 1.0 - xi - eta;
                barycentric[local_vertices[1]] = xi;
                barycentric[local_vertices[2]] = eta;
                let tabulation = space
                    .try_tabulate(
                        mesh,
                        element_index,
                        barycentric[1],
                        barycentric[2],
                        barycentric[3],
                    )
                    .unwrap();
                let field = affine_hdiv_field(face_point(face, xi, eta));
                let basis = [
                    tabulation.vector_value(local_dof, 0),
                    tabulation.vector_value(local_dof, 1),
                    tabulation.vector_value(local_dof, 2),
                ];
                flux += point.weight
                    * (field[0] * surface[0] + field[1] * surface[1] + field[2] * surface[2]);
                normalization += point.weight
                    * (basis[0] * surface[0] + basis[1] * surface[1] + basis[2] * surface[2]);
            }
            assert!(
                normalization.abs() > 1.0e-12,
                "face {face_index} normalization"
            );
            let _ = face_key;
            flux / normalization
        })
        .collect()
}

#[test]
fn rt0_transfer_preserves_affine_flux_moments_on_2d_refinement() {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_values = analytic_edge_state(&source, &source_space);
    let expected = analytic_edge_state(&target, &target_space);

    let transferred = transfer_hdiv_rt0_state_checked(
        &source,
        &target,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("RT0 transfer must preserve affine 2-D flux moments");
    assert_eq!(transferred.len(), expected.len());
    for (actual, expected) in transferred.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 2.0e-10,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn rt0_transfer_preserves_affine_flux_moments_on_3d_refinement() {
    let source = unit_cube_tetrahedra(1);
    let target = unit_cube_tetrahedra(2);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_values = analytic_face_state(&source, &source_space);
    let expected = analytic_face_state(&target, &target_space);

    let transferred = transfer_hdiv_rt0_state_checked(
        &source,
        &target,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("RT0 transfer must preserve affine 3-D flux moments");
    assert_eq!(transferred.len(), expected.len());
    for (face, (actual, expected)) in transferred.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 5.0e-9,
            "face {face}: {actual} != {expected}"
        );
    }
}

#[test]
fn rt0_transfer_is_identity_under_reversed_local_orientation() {
    let source = unit_square_triangles(1);
    let mut target = source.clone();
    target.elements[0].nodes.swap(1, 2);
    target.detect_boundaries();
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_values = analytic_edge_state(&source, &source_space);
    let expected = analytic_edge_state(&target, &target_space);
    let transferred = transfer_hdiv_rt0_state_checked(
        &source,
        &target,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("reversed local orientation must be admitted");
    for (actual, expected) in transferred.iter().zip(expected) {
        assert!((actual - expected).abs() < 2.0e-10);
    }
}

#[test]
fn rt0_transfer_rejects_unsupported_orders_geometry_and_state() {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let source_dofs = HdivSpace::new().num_global_dofs(&source);
    assert!(matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &target,
            &HdivSpace::with_order(HdivOrder::Rt1),
            &HdivSpace::new(),
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));
    assert!(matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &target,
            &HdivSpace::new(),
            &HdivSpace::new(),
            &vec![0.0; source_dofs - 1],
        ),
        Err(InterpolationError::SourceValueCount { .. })
    ));
    let mut non_finite = vec![0.0; source_dofs];
    non_finite[0] = f64::NAN;
    assert!(matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &target,
            &HdivSpace::new(),
            &HdivSpace::new(),
            &non_finite,
        ),
        Err(InterpolationError::NonFiniteSource { .. })
    ));
    assert!(matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &unit_square_quads(1),
            &HdivSpace::new(),
            &HdivSpace::new(),
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));
    assert!(matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &source,
            &HdivSpace::new(),
            &HdivSpace::new(),
            &vec![0.0; source_dofs],
        ),
        Ok(_)
    ));
    let mut outside = target.clone();
    for point in &mut outside.nodes {
        point.x += 2.0;
    }
    assert!(matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &outside,
            &HdivSpace::new(),
            &HdivSpace::new(),
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::TargetElementOutside { .. })
    ));
    assert_eq!(source.elements[0].element_type, ElementType::Triangle);
}
