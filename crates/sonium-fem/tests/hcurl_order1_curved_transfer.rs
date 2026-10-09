use sonium_fem::mesh::{ElementType, MeshTopology, unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::mesh::{InterpolationError, transfer_hcurl_order1_state_checked};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn edge_moment(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let dz = p1.z - p0.z;
    dx * (1.0 - 0.25 * (p0.y + p1.y)) + dy * (2.0 + 0.25 * (p0.x + p1.x)) + 3.0 * dz
}

fn edge_midpoint_node(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> Option<usize> {
    for element in &mesh.elements {
        let vertices = element.vertices();
        for (local_edge, &(a, b)) in element.element_type.local_edges().iter().enumerate() {
            let endpoints = (vertices[a], vertices[b]);
            if (endpoints.0 == edge.0 && endpoints.1 == edge.1)
                || (endpoints.0 == edge.1 && endpoints.1 == edge.0)
            {
                return element
                    .nodes
                    .get(element.element_type.num_vertices() + local_edge)
                    .copied();
            }
        }
    }
    None
}

fn quadratic_edge_moment(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let pm = edge_midpoint_node(mesh, edge)
        .map(|node| mesh.nodes[node])
        .unwrap_or_else(|| p0.midpoint(&p1));
    let quadrature = sonium_fem::quadrature::gauss_legendre_1d(8);
    quadrature
        .iter()
        .map(|point| {
            let s = point.xi();
            let shape = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)];
            let derivative = [s - 0.5, -2.0 * s, s + 0.5];
            let x = shape[0] * p0.x + shape[1] * pm.x + shape[2] * p1.x;
            let y = shape[0] * p0.y + shape[1] * pm.y + shape[2] * p1.y;
            let dx = derivative[0] * p0.x + derivative[1] * pm.x + derivative[2] * p1.x;
            let dy = derivative[0] * p0.y + derivative[1] * pm.y + derivative[2] * p1.y;
            point.weight * ((1.0 - 0.5 * y) * dx + (2.0 + 0.5 * x) * dy)
        })
        .sum()
}

#[test]
fn order1_hcurl_transfer_recovers_curved_p2_triangle_state() {
    let mut mesh = unit_square_triangles(1).to_p2();
    let curved_edge_midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[curved_edge_midpoint].y += 0.08;
    let space = HcurlSpace::new();
    let topology = MeshTopology::build(&mesh);
    let source_values: Vec<_> = topology
        .edges
        .iter()
        .enumerate()
        .map(|(edge, &nodes)| 0.17 + 0.23 * edge as f64 + edge_moment(&mesh, nodes))
        .collect();

    let transferred =
        transfer_hcurl_order1_state_checked(&mesh, &mesh, &space, &space, &source_values)
            .expect("curved P2 triangle transfer must succeed");
    assert_eq!(transferred.len(), source_values.len());
    for (actual, expected) in transferred.iter().zip(source_values) {
        assert!((actual - expected).abs() < 2.0e-9, "{actual} != {expected}");
    }
}

#[test]
fn order1_hcurl_transfer_integrates_curved_target_edge() {
    let source = unit_square_triangles(1);
    let mut target = source.clone().to_p2();
    let curved_edge_midpoint = target.elements[0].nodes[3];
    target.nodes[curved_edge_midpoint].y += 0.08;

    let source_topology = MeshTopology::build(&source);
    let source_values: Vec<_> = source_topology
        .edges
        .iter()
        .map(|&edge| edge_moment(&source, edge))
        .collect();
    let transferred = transfer_hcurl_order1_state_checked(
        &source,
        &target,
        &HcurlSpace::new(),
        &HcurlSpace::new(),
        &source_values,
    )
    .expect("P1-to-curved-P2 transfer must succeed");

    let target_topology = MeshTopology::build(&target);
    for (index, &edge) in target_topology.edges.iter().enumerate() {
        let expected = quadratic_edge_moment(&target, edge);
        assert!(
            (transferred[index] - expected).abs() < 2.0e-9,
            "edge {edge:?}: transferred={}, expected={expected}",
            transferred[index]
        );
    }
}

#[test]
fn order1_hcurl_transfer_recovers_curved_p2_tetrahedron_state() {
    let mut mesh = unit_cube_tetrahedra(1).to_p2();
    let curved_edge_midpoint = mesh.elements[0].nodes[4];
    mesh.nodes[curved_edge_midpoint].z += 0.06;
    let space = HcurlSpace::new();
    let topology = MeshTopology::build(&mesh);
    let source_values: Vec<_> = topology
        .edges
        .iter()
        .enumerate()
        .map(|(edge, &nodes)| -0.31 + 0.11 * edge as f64 + edge_moment(&mesh, nodes))
        .collect();

    let transferred =
        transfer_hcurl_order1_state_checked(&mesh, &mesh, &space, &space, &source_values)
            .expect("curved P2 tetrahedral transfer must succeed");
    for (actual, expected) in transferred.iter().zip(source_values) {
        assert!((actual - expected).abs() < 3.0e-9, "{actual} != {expected}");
    }
}

#[test]
fn order1_hcurl_curved_transfer_rejects_invalid_layout_and_geometry() {
    let mut p3_like = unit_square_triangles(1);
    p3_like.elements[0].nodes.push(p3_like.nodes.len());
    p3_like.nodes.push((0.25, 0.25).into());
    assert!(matches!(
        transfer_hcurl_order1_state_checked(
            &p3_like,
            &p3_like,
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &[],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));

    let mut nonfinite = unit_square_triangles(1).to_p2();
    nonfinite.nodes[nonfinite.elements[0].nodes[3]].x = f64::NAN;
    assert!(matches!(
        transfer_hcurl_order1_state_checked(
            &nonfinite,
            &nonfinite,
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &[],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));

    let quadrilateral = unit_square_triangles(1);
    let mut invalid_target = quadrilateral.clone();
    invalid_target.elements[0].element_type = ElementType::Quadrilateral;
    assert!(matches!(
        transfer_hcurl_order1_state_checked(
            &quadrilateral,
            &invalid_target,
            &HcurlSpace::new(),
            &HcurlSpace::new(),
            &vec![0.0; HcurlSpace::new().num_global_dofs(&quadrilateral)],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));
}
