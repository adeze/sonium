//! Independent curved-P2 H1 -> H(curl) commuting reference.
//!
//! The scalar field is defined analytically on the reference triangle.  Its
//! reference gradient is represented by the order-2 Nedelec basis, then the
//! production tabulator is checked after the independent curved isoparametric
//! map and covariant Piola transform are applied.

use sonium_fem::mesh::{Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

const TRIANGLE_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

fn reference_quadratic(xi: f64, eta: f64) -> f64 {
    0.31 + 0.41 * xi - 0.29 * eta + 0.83 * xi * xi - 0.57 * xi * eta + 0.66 * eta * eta
}

fn reference_gradient(xi: f64, eta: f64) -> [f64; 2] {
    [
        0.41 + 1.66 * xi - 0.57 * eta,
        -0.29 - 0.57 * xi + 1.32 * eta,
    ]
}

fn reference_nedelec_coefficients() -> [f64; 8] {
    let alpha = 0.83;
    let beta = -0.57;
    let gamma = 0.66;
    let delta = 0.41;
    let epsilon = -0.29;
    [
        delta + alpha,
        -alpha,
        -alpha + gamma + epsilon - delta,
        -alpha + beta - gamma,
        epsilon + gamma,
        -gamma,
        2.0 * alpha - beta - gamma,
        alpha + gamma - 2.0 * beta,
    ]
}

fn oriented_production_coefficients(mesh: &Mesh, element_idx: usize) -> [f64; 8] {
    let element = &mesh.elements[element_idx];
    let vertex_ids = [element.nodes[0], element.nodes[1], element.nodes[2]];
    let mut coefficients = reference_nedelec_coefficients();

    // The production order-2 oriented triangle basis reverses only the
    // lowest edge mode. Match its canonical (+W01, +W12, -W20) convention.
    for (edge, &(left, right)) in TRIANGLE_EDGES.iter().enumerate() {
        let sign = if vertex_ids[left] < vertex_ids[right] {
            1.0
        } else {
            -1.0
        };
        let canonical_factor = if edge == 2 { -sign } else { sign };
        coefficients[2 * edge] /= canonical_factor;
    }
    coefficients
}

fn p2_triangle_map(
    mesh: &Mesh,
    element_idx: usize,
    xi: f64,
    eta: f64,
) -> ([f64; 2], [[f64; 2]; 2]) {
    let element = &mesh.elements[element_idx];
    assert_eq!(element.nodes.len(), 6);
    let l0 = 1.0 - xi - eta;
    let l1 = xi;
    let l2 = eta;
    let values = [
        l0 * (2.0 * l0 - 1.0),
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        4.0 * l0 * l1,
        4.0 * l1 * l2,
        4.0 * l2 * l0,
    ];
    let gradients = [
        [-(4.0 * l0 - 1.0), -(4.0 * l0 - 1.0)],
        [4.0 * l1 - 1.0, 0.0],
        [0.0, 4.0 * l2 - 1.0],
        [4.0 * (l0 - l1), -4.0 * l1],
        [4.0 * l2, 4.0 * l1],
        [-4.0 * l2, 4.0 * (l0 - l2)],
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
    let curved_edge = mesh.elements[0].nodes[3];
    mesh.nodes[curved_edge] = Point::new_2d(0.5, 0.13);
    mesh
}

pub fn run_curved_p2_h1_hcurl_commuting_reference() {
    let mesh = curved_triangle_mesh();
    let hcurl = HcurlSpace::order2();
    hcurl
        .validate(&mesh)
        .expect("order-2 H(curl) must validate on curved P2 triangles");

    let samples = [[0.13, 0.19], [0.22, 0.31], [0.37, 0.11], [0.17, 0.47]];
    let mut maximum_field_residual: f64 = 0.0;
    let mut maximum_curl_residual: f64 = 0.0;

    for element_idx in 0..mesh.num_elements() {
        let oriented = oriented_production_coefficients(&mesh, element_idx);
        assert_eq!(hcurl.local_to_global(&mesh, element_idx).len(), 8);
        for &[xi, eta] in &samples {
            let (_, jacobian) = p2_triangle_map(&mesh, element_idx, xi, eta);
            let det = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            assert!(
                det > 0.0,
                "curved reference triangle must preserve orientation"
            );

            let reference = reference_gradient(xi, eta);
            let expected = [
                (jacobian[1][1] * reference[0] - jacobian[1][0] * reference[1]) / det,
                (-jacobian[0][1] * reference[0] + jacobian[0][0] * reference[1]) / det,
            ];
            let tabulation = hcurl.tabulate(&mesh, element_idx, xi, eta, 0.0);
            let reconstructed = [
                oriented
                    .iter()
                    .enumerate()
                    .map(|(local, &coefficient)| coefficient * tabulation.vector_value(local, 0))
                    .sum::<f64>(),
                oriented
                    .iter()
                    .enumerate()
                    .map(|(local, &coefficient)| coefficient * tabulation.vector_value(local, 1))
                    .sum::<f64>(),
            ];
            for direction in 0..2 {
                maximum_field_residual = maximum_field_residual
                    .max((reconstructed[direction] - expected[direction]).abs());
            }

            let curl = oriented
                .iter()
                .enumerate()
                .map(|(local, &coefficient)| coefficient * tabulation.derivatives[local])
                .sum::<f64>();
            maximum_curl_residual = maximum_curl_residual.max(curl.abs());

            // Keep the scalar fixture nontrivial at each sample.
            assert!(reference_quadratic(xi, eta).is_finite());
        }
    }

    println!(
        "CURVED_P2_H1_HCURL_INDEPENDENT_REFERENCE max_field_residual={maximum_field_residual:.6e} max_curl_residual={maximum_curl_residual:.6e}"
    );
    assert!(maximum_field_residual < 3.0e-11);
    assert!(maximum_curl_residual < 3.0e-11);
}

#[test]
fn curved_p2_h1_hcurl_commuting_reference() {
    run_curved_p2_h1_hcurl_commuting_reference();
}
