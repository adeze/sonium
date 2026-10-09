//! Independent reference for the curved-P2 geometry used by H(curl).
//!
//! The production tabulator must use all six quadratic triangle geometry
//! nodes for both physical coordinates and the covariant-Piola Jacobian. The
//! checks below reconstruct that map independently instead of reusing FEM
//! shape helpers.

use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn p2_triangle_map(mesh: &Mesh, element: usize, xi: f64, eta: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    let element = &mesh.elements[element];
    assert_eq!(element.element_type, ElementType::Triangle);
    assert_eq!(element.nodes.len(), 6);

    let x = xi;
    let y = eta;
    let l = 1.0 - x - y;
    let values = [
        l * (2.0 * l - 1.0),
        x * (2.0 * x - 1.0),
        y * (2.0 * y - 1.0),
        4.0 * x * l,
        4.0 * x * y,
        4.0 * y * l,
    ];
    let gradients = [
        [-(4.0 * l - 1.0), -(4.0 * l - 1.0)],
        [4.0 * x - 1.0, 0.0],
        [0.0, 4.0 * y - 1.0],
        [4.0 * (l - x), -4.0 * x],
        [4.0 * y, 4.0 * x],
        [-4.0 * y, 4.0 * (l - y)],
    ];

    let mut point = [0.0; 2];
    let mut jacobian = [[0.0; 2]; 2];
    for (local, &node) in element.nodes.iter().enumerate() {
        let p = mesh.nodes[node];
        point[0] += values[local] * p.x;
        point[1] += values[local] * p.y;
        jacobian[0][0] += gradients[local][0] * p.x;
        jacobian[0][1] += gradients[local][1] * p.x;
        jacobian[1][0] += gradients[local][0] * p.y;
        jacobian[1][1] += gradients[local][1] * p.y;
    }
    (point, jacobian)
}

fn curved_triangle_mesh() -> Mesh {
    let mut mesh = sonium_fem::mesh::unit_square_triangles(1).to_p2();
    let first = &mesh.elements[0];
    assert_eq!(first.element_type, ElementType::Triangle);
    let curved_edge = first.nodes[3];
    mesh.nodes[curved_edge] = Point::new_2d(0.5, 0.12);
    mesh
}

fn p2_tetrahedron_map(
    mesh: &Mesh,
    element: usize,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> ([f64; 3], [[f64; 3]; 3]) {
    let element = &mesh.elements[element];
    assert_eq!(element.element_type, ElementType::Tetrahedron);
    assert_eq!(element.nodes.len(), 10);

    let lambdas = [1.0 - xi - eta - zeta, xi, eta, zeta];
    let lambda_gradients = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut values = [0.0; 10];
    let mut gradients = [[0.0; 3]; 10];
    for vertex in 0..4 {
        values[vertex] = lambdas[vertex] * (2.0 * lambdas[vertex] - 1.0);
        for component in 0..3 {
            gradients[vertex][component] =
                (4.0 * lambdas[vertex] - 1.0) * lambda_gradients[vertex][component];
        }
    }
    for (edge_index, &(a, b)) in edges.iter().enumerate() {
        let local = edge_index + 4;
        values[local] = 4.0 * lambdas[a] * lambdas[b];
        for component in 0..3 {
            gradients[local][component] = 4.0
                * (lambdas[b] * lambda_gradients[a][component]
                    + lambdas[a] * lambda_gradients[b][component]);
        }
    }

    let mut point = [0.0; 3];
    let mut jacobian = [[0.0; 3]; 3];
    for (local, &node) in element.nodes.iter().enumerate() {
        let p = mesh.nodes[node];
        for component in 0..3 {
            point[component] += values[local] * [p.x, p.y, p.z][component];
            for reference_component in 0..3 {
                jacobian[component][reference_component] +=
                    gradients[local][reference_component] * [p.x, p.y, p.z][component];
            }
        }
    }
    (point, jacobian)
}

fn curved_tetrahedron_mesh() -> Mesh {
    let mut mesh = sonium_fem::mesh::unit_cube_tetrahedra(1).to_p2();
    let first = &mesh.elements[0];
    assert_eq!(first.element_type, ElementType::Tetrahedron);
    let curved_edge = first.nodes[4];
    let midpoint = mesh.nodes[curved_edge];
    mesh.nodes[curved_edge] = Point::new_3d(midpoint.x, midpoint.y, midpoint.z + 0.08);
    mesh
}

#[test]
fn hcurl_rejects_malformed_element_node_layout_without_panicking() {
    let mut mesh = sonium_fem::mesh::unit_square_triangles(1);
    let duplicate = mesh.elements[0].nodes[0];
    mesh.elements[0].nodes.push(duplicate);
    let error = HcurlSpace::new()
        .validate(&mesh)
        .expect_err("malformed element layout must be rejected");
    assert!(matches!(
        error,
        sonium_fem::spaces::SpaceValidationError::InvalidMesh { element: 0, .. }
    ));
}

#[test]
fn hdiv_rejects_malformed_element_node_layout_without_panicking() {
    let mut mesh = sonium_fem::mesh::unit_square_triangles(1);
    let duplicate = mesh.elements[0].nodes[0];
    mesh.elements[0].nodes.push(duplicate);
    let error = sonium_fem::spaces::HdivSpace::new()
        .validate(&mesh)
        .expect_err("malformed element layout must be rejected");
    assert!(matches!(
        error,
        sonium_fem::spaces::SpaceValidationError::InvalidMesh { element: 0, .. }
    ));
}

pub fn curved_p2_hcurl_uses_physical_map_and_jacobian() {
    let mesh = curved_triangle_mesh();
    let xi = 0.23;
    let eta = 0.31;
    let (expected_point, expected_jacobian) = p2_triangle_map(&mesh, 0, xi, eta);
    let expected_det = (expected_jacobian[0][0] * expected_jacobian[1][1]
        - expected_jacobian[0][1] * expected_jacobian[1][0])
        .abs();

    let tabulation = HcurlSpace::new().tabulate(&mesh, 0, xi, eta, 0.0);
    assert_eq!(tabulation.n_dofs, 3);
    assert!((tabulation.physical_coords[0] - expected_point[0]).abs() < 2.0e-13);
    assert!((tabulation.physical_coords[1] - expected_point[1]).abs() < 2.0e-13);
    assert!((tabulation.det_j - expected_det).abs() < 2.0e-12);
    assert!(tabulation.det_j > 0.0);

    let order2 = HcurlSpace::order2().tabulate(&mesh, 0, xi, eta, 0.0);
    assert_eq!(order2.n_dofs, 8);
    assert!((order2.physical_coords[0] - expected_point[0]).abs() < 2.0e-13);
    assert!((order2.physical_coords[1] - expected_point[1]).abs() < 2.0e-13);
    assert!((order2.det_j - expected_det).abs() < 2.0e-12);
}

#[test]
fn hcurl_uses_curved_p2_physical_map_and_jacobian() {
    curved_p2_hcurl_uses_physical_map_and_jacobian();
}

pub fn curved_p2_hcurl_tetrahedral_map_for_both_orders() {
    let mesh = curved_tetrahedron_mesh();
    let xi = 0.17;
    let eta = 0.19;
    let zeta = 0.13;
    let (expected_point, expected_jacobian) = p2_tetrahedron_map(&mesh, 0, xi, eta, zeta);
    let a = expected_jacobian;
    let expected_det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        .abs();

    for space in [HcurlSpace::new(), HcurlSpace::order2()] {
        let tabulation = space.tabulate(&mesh, 0, xi, eta, zeta);
        assert!(tabulation.n_dofs == 6 || tabulation.n_dofs == 20);
        for component in 0..3 {
            assert!(
                (tabulation.physical_coords[component] - expected_point[component]).abs() < 2.0e-12
            );
        }
        assert!((tabulation.det_j - expected_det).abs() < 2.0e-11);
        assert!(tabulation.det_j > 0.0);
    }
}

#[test]
fn hcurl_uses_curved_p2_tetrahedral_map_for_both_orders() {
    curved_p2_hcurl_tetrahedral_map_for_both_orders();
}
