// Independent P2 H1 -> order-2 H(curl) commuting reference.
//
// The existing order-2 exact-sequence tests recover Nedelec coefficients by
// fitting the production tabulation. This packet derives the coefficients
// from an analytical quadratic in the reference triangle instead, applies
// the global edge-orientation signs independently, and then exercises the
// production physical tabulation on an oppositely oriented two-triangle
// patch.

use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

const TRIANGLE_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

/// A quadratic with nonzero edge and interior reference coefficients.
fn manufactured_quadratic(x: f64, y: f64) -> f64 {
    0.3 + 0.7 * x - 0.4 * y + 1.2 * x * x - 0.8 * x * y + 0.9 * y * y
}

fn manufactured_gradient(x: f64, y: f64) -> [f64; 2] {
    [0.7 + 2.4 * x - 0.8 * y, -0.4 - 0.8 * x + 1.8 * y]
}

/// Recover the coefficients of a quadratic in local `(xi, eta)` coordinates
/// from its six P2 nodal samples.  This is an analytical P2 reconstruction,
/// independent of the FEM H1 tabulation.
fn reference_quadratic_coefficients(samples: [f64; 6]) -> [f64; 5] {
    let [v0, v1, v2, m01, m12, m20] = samples;
    let alpha = 2.0 * (v0 - 2.0 * m01 + v1);
    let delta = v1 - v0 - alpha;
    let gamma = 2.0 * (v0 - 2.0 * m20 + v2);
    let epsilon = v2 - v0 - gamma;
    let beta = 4.0 * (m12 - v0 - 0.5 * delta - 0.5 * epsilon - 0.25 * (alpha + gamma));
    [alpha, beta, gamma, delta, epsilon]
}

/// Coefficients of the exact gradient in the explicitly defined order-2
/// reference basis.  The basis is ordered as two modes on each edge followed
/// by the two interior bubbles used by `nedelec2_triangle_oriented`.
fn reference_nedelec_coefficients([alpha, beta, gamma, delta, epsilon]: [f64; 5]) -> [f64; 8] {
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

fn local_p2_samples(mesh: &sonium_fem::mesh::Mesh, element_idx: usize) -> [f64; 6] {
    let element = &mesh.elements[element_idx];
    let vertex = |local: usize| {
        let point = mesh.nodes[element.nodes[local]];
        [point.x, point.y]
    };
    let midpoint = |left: usize, right: usize| {
        let first = vertex(left);
        let second = vertex(right);
        [0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1])]
    };

    let [p0, p1, p2] = [vertex(0), vertex(1), vertex(2)];
    let [m01, m12, m20] = [midpoint(0, 1), midpoint(1, 2), midpoint(2, 0)];
    [
        manufactured_quadratic(p0[0], p0[1]),
        manufactured_quadratic(p1[0], p1[1]),
        manufactured_quadratic(p2[0], p2[1]),
        manufactured_quadratic(m01[0], m01[1]),
        manufactured_quadratic(m12[0], m12[1]),
        manufactured_quadratic(m20[0], m20[1]),
    ]
}

fn physical_point(
    mesh: &sonium_fem::mesh::Mesh,
    element_idx: usize,
    xi: f64,
    eta: f64,
) -> [f64; 2] {
    let element = &mesh.elements[element_idx];
    let p0 = mesh.nodes[element.nodes[0]];
    let p1 = mesh.nodes[element.nodes[1]];
    let p2 = mesh.nodes[element.nodes[2]];
    [
        p0.x + xi * (p1.x - p0.x) + eta * (p2.x - p0.x),
        p0.y + xi * (p1.y - p0.y) + eta * (p2.y - p0.y),
    ]
}

fn oriented_production_coefficients(mesh: &sonium_fem::mesh::Mesh, element_idx: usize) -> [f64; 8] {
    let element = &mesh.elements[element_idx];
    let vertex_ids = [element.nodes[0], element.nodes[1], element.nodes[2]];
    let reference = reference_nedelec_coefficients(reference_quadratic_coefficients(
        local_p2_samples(mesh, element_idx),
    ));
    let mut coefficients = reference;

    // The production order-2 edge basis uses an orientation sign only for
    // the lowest edge mode.  The higher edge mode is even under reversal.
    // Relative to the canonical local basis (+W01, +W12, -W20), the factors
    // are sign(01), sign(12), and -sign(20), respectively.
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

pub fn run_p2_gradient_commutes_against_independent_reference_on_oriented_patch() {
    let mesh = unit_square_triangles(1);
    let hcurl = HcurlSpace::order2();
    hcurl
        .validate(&mesh)
        .expect("order-2 H(curl) must validate on triangles");

    let sample_points = [[0.13, 0.19], [0.22, 0.31], [0.37, 0.11], [0.17, 0.47]];
    let mut local_coefficients = Vec::new();
    let mut maximum_field_residual: f64 = 0.0;
    let mut maximum_curl_residual: f64 = 0.0;

    for element_idx in 0..mesh.num_elements() {
        let coefficients = oriented_production_coefficients(&mesh, element_idx);
        let dofs = hcurl.local_to_global(&mesh, element_idx);
        assert_eq!(dofs.len(), coefficients.len());

        for &[xi, eta] in &sample_points {
            let tabulation = hcurl.tabulate(&mesh, element_idx, xi, eta, 0.0);
            let point = physical_point(&mesh, element_idx, xi, eta);
            let expected = manufactured_gradient(point[0], point[1]);
            let reconstructed = [
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(local, &coefficient)| coefficient * tabulation.vector_value(local, 0))
                    .sum::<f64>(),
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(local, &coefficient)| coefficient * tabulation.vector_value(local, 1))
                    .sum::<f64>(),
            ];
            for direction in 0..2 {
                maximum_field_residual = maximum_field_residual
                    .max((reconstructed[direction] - expected[direction]).abs());
            }
            let curl = coefficients
                .iter()
                .enumerate()
                .map(|(local, &coefficient)| coefficient * tabulation.derivatives[local])
                .sum::<f64>();
            maximum_curl_residual = maximum_curl_residual.max(curl.abs());
        }
        local_coefficients.push((dofs, coefficients));
    }

    // The two elements share one edge with opposite local orientation.  The
    // independently derived coefficients must therefore agree in both edge
    // modes after production global numbering is applied.
    let mut global_coefficients: Vec<Option<f64>> = vec![None; hcurl.num_global_dofs(&mesh)];
    for (dofs, coefficients) in &local_coefficients {
        for (&global, &coefficient) in dofs.iter().zip(coefficients) {
            if let Some(previous) = global_coefficients[global] {
                assert!(
                    (previous - coefficient).abs() < 2.0e-11,
                    "order-2 H(curl) global DOF {global} is not orientation-consistent"
                );
            } else {
                global_coefficients[global] = Some(coefficient);
            }
        }
    }

    println!(
        "P2_H1_HCURL_INDEPENDENT_REFERENCE max_field_residual={maximum_field_residual:.6e} \
         max_curl_residual={maximum_curl_residual:.6e}"
    );
    assert!(maximum_field_residual < 2.0e-11);
    assert!(maximum_curl_residual < 2.0e-11);
}

#[test]
fn p2_gradient_commutes_against_independent_reference_on_oriented_patch() {
    run_p2_gradient_commutes_against_independent_reference_on_oriented_patch();
}
