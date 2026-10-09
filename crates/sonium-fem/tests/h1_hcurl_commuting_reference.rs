// Independent affine exact-sequence evidence for the CPU H¹/H(curl) slice.
//
// The reference calculation uses the analytical edge line integrals of the
// gradient of a P1 scalar field. It therefore does not derive the expected
// coefficients from the Nédélec basis implementation itself.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::basis::{
    nedelec::TRIANGLE_EDGES, nedelec::nedelec1_triangle, p1_triangle, p1_triangle_grad,
};
use sonium_fem::mesh::{MeshTopology, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space, HcurlSpace};

const REFERENCE_VERTICES: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

#[test]
fn test_p1_gradient_commutes_with_affine_nedelec_interpolation() {
    p1_gradient_commutes_with_affine_nedelec_interpolation();
}

#[test]
fn test_affine_nedelec_commuting_reference_is_invariant_under_edge_reversal() {
    affine_nedelec_commuting_reference_is_invariant_under_edge_reversal();
}

#[test]
fn test_production_hcurl_p1_gradient_commutes_across_oriented_patch() {
    production_hcurl_p1_gradient_commutes_across_oriented_patch();
}

fn manufactured_scalar([x, y]: [f64; 2]) -> f64 {
    0.35 + 1.2 * x - 0.8 * y
}

fn p1_gradient_from_nodal_values(nodal_values: [f64; 3]) -> [f64; 2] {
    let gradients = p1_triangle_grad();
    [
        nodal_values
            .iter()
            .zip(gradients)
            .map(|(&value, gradient)| value * gradient[0])
            .sum(),
        nodal_values
            .iter()
            .zip(gradients)
            .map(|(&value, gradient)| value * gradient[1])
            .sum(),
    ]
}

/// Analytical line integrals of a constant vector field on the canonical
/// reference-triangle edges. The edge DOF is `∫ g · dx`, so no unit-tangent
/// or edge-length convention is hidden in this reference.
fn gradient_edge_moments(gradient: [f64; 2]) -> [f64; 3] {
    TRIANGLE_EDGES.map(|(start, end)| {
        let displacement = [
            REFERENCE_VERTICES[end][0] - REFERENCE_VERTICES[start][0],
            REFERENCE_VERTICES[end][1] - REFERENCE_VERTICES[start][1],
        ];
        gradient[0] * displacement[0] + gradient[1] * displacement[1]
    })
}

fn reconstruct_nedelec_gradient(
    xi: f64,
    eta: f64,
    moments: [f64; 3],
    edge_orientation: [f64; 3],
) -> ([f64; 2], f64) {
    let (values, curls) = nedelec1_triangle(xi, eta);
    let mut field = [0.0; 2];
    let mut curl = 0.0;
    for edge in 0..3 {
        // Reversing an edge reverses both its basis function and its line
        // integral. Their product must therefore be orientation-invariant.
        let coefficient = edge_orientation[edge] * moments[edge];
        for component in 0..2 {
            field[component] += coefficient * edge_orientation[edge] * values[edge][component];
        }
        curl += coefficient * edge_orientation[edge] * curls[edge];
    }
    (field, curl)
}

pub fn p1_gradient_commutes_with_affine_nedelec_interpolation() {
    let nodal_values = REFERENCE_VERTICES.map(manufactured_scalar);
    let gradient = p1_gradient_from_nodal_values(nodal_values);
    assert!((gradient[0] - 1.2).abs() < 2.0e-15);
    assert!((gradient[1] + 0.8).abs() < 2.0e-15);

    for [xi, eta] in [[0.13, 0.19], [0.22, 0.43], [0.37, 0.11]] {
        let shape = p1_triangle(xi, eta);
        let scalar = shape
            .iter()
            .zip(nodal_values)
            .map(|(&basis, value)| basis * value)
            .sum::<f64>();
        assert!((scalar - manufactured_scalar([xi, eta])).abs() < 2.0e-15);

        let (field, curl) =
            reconstruct_nedelec_gradient(xi, eta, gradient_edge_moments(gradient), [1.0; 3]);
        for component in 0..2 {
            assert!(
                (field[component] - gradient[component]).abs() < 2.0e-14,
                "commuting gradient residual at ({xi}, {eta}) component {component}: {}",
                (field[component] - gradient[component]).abs()
            );
        }
        assert!(curl.abs() < 2.0e-14, "curl(grad u) residual: {curl:e}");
    }
}

pub fn affine_nedelec_commuting_reference_is_invariant_under_edge_reversal() {
    let nodal_values = REFERENCE_VERTICES.map(manufactured_scalar);
    let gradient = p1_gradient_from_nodal_values(nodal_values);
    let moments = gradient_edge_moments(gradient);
    let point = [0.23, 0.31];
    let (canonical, canonical_curl) =
        reconstruct_nedelec_gradient(point[0], point[1], moments, [1.0; 3]);

    let mut maximum_field_residual: f64 = 0.0;
    let mut maximum_curl_residual: f64 = canonical_curl.abs();
    for reversed_edge in 0..3 {
        let mut orientation = [1.0; 3];
        orientation[reversed_edge] = -1.0;
        let (field, curl) = reconstruct_nedelec_gradient(point[0], point[1], moments, orientation);
        for component in 0..2 {
            maximum_field_residual =
                maximum_field_residual.max((field[component] - canonical[component]).abs());
        }
        maximum_curl_residual = maximum_curl_residual.max((curl - canonical_curl).abs());
    }

    println!(
        "H1_HCURL_COMMUTING_REFERENCE max_field_residual={maximum_field_residual:.6e} \
         max_curl_residual={maximum_curl_residual:.6e}"
    );
    assert!(maximum_field_residual < 2.0e-14);
    assert!(maximum_curl_residual < 2.0e-14);
}

pub fn production_hcurl_p1_gradient_commutes_across_oriented_patch() {
    let mesh = unit_square_triangles(1);
    let h1 = H1Space::new(PolynomialDegree::P1);
    let hcurl = HcurlSpace::new();
    h1.validate(&mesh)
        .expect("P1 H1 must validate on triangles");
    hcurl
        .validate(&mesh)
        .expect("first-kind Nedelec H(curl) must validate on triangles");

    let topology = MeshTopology::build(&mesh);
    let scalar_values: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|point| manufactured_scalar([point.x, point.y]))
        .collect();
    let edge_moments: Vec<f64> = topology
        .edges
        .iter()
        .map(|&(start, end)| {
            let displacement = [
                mesh.nodes[end].x - mesh.nodes[start].x,
                mesh.nodes[end].y - mesh.nodes[start].y,
            ];
            1.2 * displacement[0] - 0.8 * displacement[1]
        })
        .collect();

    let mut maximum_field_residual: f64 = 0.0;
    let mut maximum_curl_residual: f64 = 0.0;
    let sample_points = [[0.13, 0.19], [0.22, 0.31], [0.37, 0.11]];
    for element in 0..mesh.num_elements() {
        let h1_dofs = h1.local_to_global(&mesh, element);
        let hcurl_dofs = hcurl.local_to_global(&mesh, element);
        for &[xi, eta] in &sample_points {
            let h1_tabulation = h1.tabulate(&mesh, element, xi, eta, 0.0);
            let hcurl_tabulation = hcurl.tabulate(&mesh, element, xi, eta, 0.0);
            let expected_gradient = [
                h1_dofs
                    .iter()
                    .enumerate()
                    .map(|(local, &global)| scalar_values[global] * h1_tabulation.grad(local, 0))
                    .sum::<f64>(),
                h1_dofs
                    .iter()
                    .enumerate()
                    .map(|(local, &global)| scalar_values[global] * h1_tabulation.grad(local, 1))
                    .sum::<f64>(),
            ];

            let reconstructed = [
                hcurl_dofs
                    .iter()
                    .enumerate()
                    .map(|(local, &global)| {
                        edge_moments[global] * hcurl_tabulation.vector_value(local, 0)
                    })
                    .sum::<f64>(),
                hcurl_dofs
                    .iter()
                    .enumerate()
                    .map(|(local, &global)| {
                        edge_moments[global] * hcurl_tabulation.vector_value(local, 1)
                    })
                    .sum::<f64>(),
            ];
            for component in 0..2 {
                maximum_field_residual = maximum_field_residual
                    .max((reconstructed[component] - expected_gradient[component]).abs());
            }
            let curl = hcurl_dofs
                .iter()
                .enumerate()
                .map(|(local, &global)| edge_moments[global] * hcurl_tabulation.derivatives[local])
                .sum::<f64>();
            maximum_curl_residual = maximum_curl_residual.max(curl.abs());
        }
    }

    println!(
        "PRODUCTION_H1_HCURL_PATCH max_field_residual={maximum_field_residual:.6e} \
         max_curl_residual={maximum_curl_residual:.6e}"
    );
    assert!(maximum_field_residual < 2.0e-13);
    assert!(maximum_curl_residual < 2.0e-13);
}
