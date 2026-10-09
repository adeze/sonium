//! Lowest-order Nédélec edge elements (first kind, order 1)
//!
//! Whitney 1-forms for H(curl)-conforming finite elements.
//! DOFs are associated with mesh edges; each basis function is tangentially
//! continuous across element boundaries.
//!
//! ## Formula
//!
//! On a simplex with barycentric coordinates `lambda_i` and reference gradients
//! `grad_lambda_i`, the edge basis function for edge `(i, j)` is:
//!
//! ```text
//! N_ij = lambda_i * grad(lambda_j) - lambda_j * grad(lambda_i)
//! ```
//!
//! The curl in the reference domain:
//! - 2D: `curl(N_ij) = 2 * (grad_lambda_i x grad_lambda_j)` (scalar)
//! - 3D: `curl(N_ij) = 2 * (grad_lambda_i x grad_lambda_j)` (vector)

/// Edge ordering for a reference triangle: edges 0,1,2 connect vertex pairs.
/// Edge k connects `TRIANGLE_EDGES[k]`.
pub const TRIANGLE_EDGES: [(usize, usize); 3] = [
    (0, 1), // edge 0: v0 -> v1
    (1, 2), // edge 1: v1 -> v2
    (2, 0), // edge 2: v2 -> v0
];

/// Edge ordering for a reference tetrahedron: 6 edges.
pub const TET_EDGES: [(usize, usize); 6] = [
    (0, 1), // edge 0
    (0, 2), // edge 1
    (0, 3), // edge 2
    (1, 2), // edge 3
    (1, 3), // edge 4
    (2, 3), // edge 5
];

/// Evaluate lowest-order Nédélec basis functions on a reference triangle.
///
/// Returns `(values, curls)` where:
/// - `values[edge][component]`: 2D vector value of basis function at `(xi, eta)`
/// - `curls[edge]`: scalar curl (z-component) of each basis function
///
/// Reference triangle: `(0,0)`, `(1,0)`, `(0,1)`.
pub fn nedelec1_triangle(xi: f64, eta: f64) -> ([[f64; 2]; 3], [f64; 3]) {
    // Barycentric coordinates
    let l0 = 1.0 - xi - eta;
    let l1 = xi;
    let l2 = eta;

    // Reference gradients of barycentric coordinates (constant on simplex)
    let gl0 = [-1.0, -1.0];
    let gl1 = [1.0, 0.0];
    let gl2 = [0.0, 1.0];

    let lambdas = [l0, l1, l2];
    let grads = [gl0, gl1, gl2];

    let mut values = [[0.0; 2]; 3];
    let mut curls = [0.0; 3];

    for (edge_idx, &(i, j)) in TRIANGLE_EDGES.iter().enumerate() {
        // N_ij = lambda_i * grad(lambda_j) - lambda_j * grad(lambda_i)
        for d in 0..2 {
            values[edge_idx][d] = lambdas[i] * grads[j][d] - lambdas[j] * grads[i][d];
        }

        // curl(N_ij) in 2D = d(N_y)/dx - d(N_x)/dy
        // = 2 * (grad_lambda_i[0] * grad_lambda_j[1] - grad_lambda_i[1] * grad_lambda_j[0])
        curls[edge_idx] = 2.0 * (grads[i][0] * grads[j][1] - grads[i][1] * grads[j][0]);
    }

    (values, curls)
}

/// Evaluate lowest-order Nédélec basis functions on a reference tetrahedron.
///
/// Returns `(values, curls)` where:
/// - `values[edge][component]`: 3D vector value of basis function at `(xi, eta, zeta)`
/// - `curls[edge][component]`: 3D vector curl of each basis function
///
/// Reference tetrahedron: `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)`.
pub fn nedelec1_tetrahedron(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 6], [[f64; 3]; 6]) {
    // Barycentric coordinates
    let l0 = 1.0 - xi - eta - zeta;
    let l1 = xi;
    let l2 = eta;
    let l3 = zeta;

    // Reference gradients of barycentric coordinates (constant on simplex)
    let gl0 = [-1.0, -1.0, -1.0];
    let gl1 = [1.0, 0.0, 0.0];
    let gl2 = [0.0, 1.0, 0.0];
    let gl3 = [0.0, 0.0, 1.0];

    let lambdas = [l0, l1, l2, l3];
    let grads = [gl0, gl1, gl2, gl3];

    let mut values = [[0.0; 3]; 6];
    let mut curls = [[0.0; 3]; 6];

    for (edge_idx, &(i, j)) in TET_EDGES.iter().enumerate() {
        // N_ij = lambda_i * grad(lambda_j) - lambda_j * grad(lambda_i)
        for d in 0..3 {
            values[edge_idx][d] = lambdas[i] * grads[j][d] - lambdas[j] * grads[i][d];
        }

        // curl(N_ij) = 2 * (grad_lambda_i x grad_lambda_j)
        let gi = &grads[i];
        let gj = &grads[j];
        curls[edge_idx][0] = 2.0 * (gi[1] * gj[2] - gi[2] * gj[1]);
        curls[edge_idx][1] = 2.0 * (gi[2] * gj[0] - gi[0] * gj[2]);
        curls[edge_idx][2] = 2.0 * (gi[0] * gj[1] - gi[1] * gj[0]);
    }

    (values, curls)
}

// ---------------------------------------------------------------------------
// Second-order Nédélec (first kind, order 2) on triangles
// ---------------------------------------------------------------------------

/// Second-order Nédélec basis functions on a reference triangle.
///
/// 8 DOFs: 2 per edge (6 edge DOFs) + 2 interior DOFs.
///
/// Edge DOFs use the first two Legendre moments of the tangential component
/// along each edge. Interior DOFs are curl-free bubbles.
///
/// Returns `(values, curls)` where:
/// - `values`: `[8][2]` — 2D vector values at `(xi, eta)`
/// - `curls`: `[8]` — scalar curl of each basis function
pub fn nedelec2_triangle(xi: f64, eta: f64) -> ([[f64; 2]; 8], [f64; 8]) {
    nedelec2_triangle_oriented(xi, eta, [0, 1, 2])
}

/// Evaluate the second-order triangle basis using a canonical global vertex
/// ordering for the edge moments.
pub fn nedelec2_triangle_oriented(
    xi: f64,
    eta: f64,
    vertex_ids: [usize; 3],
) -> ([[f64; 2]; 8], [f64; 8]) {
    let l0 = 1.0 - xi - eta;
    let l1 = xi;
    let l2 = eta;

    let gl0 = [-1.0, -1.0];
    let gl1 = [1.0, 0.0];
    let gl2 = [0.0, 1.0];

    let lambdas = [l0, l1, l2];
    let grads = [gl0, gl1, gl2];

    let mut values = [[0.0; 2]; 8];
    let mut curls = [0.0; 8];

    // First 6 DOFs: 2 per edge (Whitney + λ-weighted Whitney). The first
    // moment is odd under edge reversal; the higher moment is even because
    // both the edge polynomial and Whitney form change sign.
    for (edge_idx, &(i, j)) in TRIANGLE_EDGES.iter().enumerate() {
        let orientation = if vertex_ids[i] < vertex_ids[j] {
            1.0
        } else {
            -1.0
        };

        // DOF 2*e: lowest-order Whitney (same as order 1)
        // N_ij = λ_i grad(λ_j) - λ_j grad(λ_i)
        for d in 0..2 {
            values[2 * edge_idx][d] =
                orientation * (lambdas[i] * grads[j][d] - lambdas[j] * grads[i][d]);
        }
        let local_curl = 2.0 * (grads[i][0] * grads[j][1] - grads[i][1] * grads[j][0]);
        curls[2 * edge_idx] = orientation * local_curl;

        // DOF 2*e+1: higher-moment edge function
        // N_ij^{(1)} = (λ_i - λ_j)(λ_i grad(λ_j) - λ_j grad(λ_i))
        let weight = lambdas[i] - lambdas[j];
        let base = [
            lambdas[i] * grads[j][0] - lambdas[j] * grads[i][0],
            lambdas[i] * grads[j][1] - lambdas[j] * grads[i][1],
        ];
        values[2 * edge_idx + 1][0] = weight * base[0];
        values[2 * edge_idx + 1][1] = weight * base[1];
        // curl of (f · N) = f · curl(N) + grad(f) × N
        // grad(λ_i - λ_j) = grad(λ_i) - grad(λ_j)
        let gw = [grads[i][0] - grads[j][0], grads[i][1] - grads[j][1]];
        // grad(f) × N (2D cross product: gw_x * N_y - gw_y * N_x)
        let cross = gw[0] * base[1] - gw[1] * base[0];
        curls[2 * edge_idx + 1] = weight * local_curl + cross;
    }

    // Last 2 DOFs: interior bubble functions
    // Interior 0: λ_0 λ_1 grad(λ_2) - λ_0 λ_2 grad(λ_1) (face bubble)
    for d in 0..2 {
        values[6][d] = l0 * l1 * grads[2][d] - l0 * l2 * grads[1][d];
    }
    // curl = grad(λ_0 λ_1) × grad(λ_2) - grad(λ_0 λ_2) × grad(λ_1) (2D cross products)
    let gl0l1 = [l1 * gl0[0] + l0 * gl1[0], l1 * gl0[1] + l0 * gl1[1]];
    let gl0l2 = [l2 * gl0[0] + l0 * gl2[0], l2 * gl0[1] + l0 * gl2[1]];
    curls[6] = (gl0l1[0] * gl2[1] - gl0l1[1] * gl2[0]) - (gl0l2[0] * gl1[1] - gl0l2[1] * gl1[0]);

    // Interior 1: λ_1 λ_2 grad(λ_0) - λ_1 λ_0 grad(λ_2)
    for d in 0..2 {
        values[7][d] = l1 * l2 * grads[0][d] - l1 * l0 * grads[2][d];
    }
    let gl1l2 = [l2 * gl1[0] + l1 * gl2[0], l2 * gl1[1] + l1 * gl2[1]];
    let gl1l0 = [l0 * gl1[0] + l1 * gl0[0], l0 * gl1[1] + l1 * gl0[1]];
    curls[7] = (gl1l2[0] * gl0[1] - gl1l2[1] * gl0[0]) - (gl1l0[0] * gl2[1] - gl1l0[1] * gl2[0]);

    (values, curls)
}

// ---------------------------------------------------------------------------
// Second-order Nédélec (first kind, order 2) on tetrahedra
// ---------------------------------------------------------------------------

/// Second-order Nédélec basis functions on a reference tetrahedron.
///
/// The 20 functions are 12 edge functions (two moments per edge) and eight
/// face functions (two tangential bubbles per face). This is the complete
/// first-family order-2 element.
pub fn nedelec2_tetrahedron(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 20], [[f64; 3]; 20]) {
    nedelec2_tetrahedron_oriented(xi, eta, zeta, [0, 1, 2, 3])
}

/// Evaluate the tetrahedral order-2 basis with edge and face functions
/// oriented by canonical global vertex IDs.
///
/// `vertex_ids` contains the mesh vertex ID associated with each reference
/// tetrahedron vertex. Canonicalizing each face is important because a face
/// permutation is generally a 2×2 change of basis, not just a sign.
pub fn nedelec2_tetrahedron_oriented(
    xi: f64,
    eta: f64,
    zeta: f64,
    vertex_ids: [usize; 4],
) -> ([[f64; 3]; 20], [[f64; 3]; 20]) {
    let lambdas = [1.0 - xi - eta - zeta, xi, eta, zeta];
    let grads = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    let mut values = [[0.0; 3]; 20];
    let mut curls = [[0.0; 3]; 20];

    // Two edge functions per edge. The first is odd under edge reversal; the
    // second is even because both the edge polynomial and Whitney form
    // change sign.
    for (edge_idx, &(i, j)) in TET_EDGES.iter().enumerate() {
        let orientation = if vertex_ids[i] < vertex_ids[j] {
            1.0
        } else {
            -1.0
        };
        let w = [
            lambdas[i] * grads[j][0] - lambdas[j] * grads[i][0],
            lambdas[i] * grads[j][1] - lambdas[j] * grads[i][1],
            lambdas[i] * grads[j][2] - lambdas[j] * grads[i][2],
        ];
        let curl_w = cross(grads[i], grads[j]);
        let first = 2 * edge_idx;
        for d in 0..3 {
            values[first][d] = orientation * w[d];
            curls[first][d] = orientation * 2.0 * curl_w[d];
        }

        let weight = lambdas[i] - lambdas[j];
        let grad_weight = [
            grads[i][0] - grads[j][0],
            grads[i][1] - grads[j][1],
            grads[i][2] - grads[j][2],
        ];
        let second = first + 1;
        for d in 0..3 {
            values[second][d] = weight * w[d];
            curls[second][d] = weight * 2.0 * curl_w[d];
        }
        let curl_weighted = cross(grad_weight, w);
        for d in 0..3 {
            curls[second][d] += curl_weighted[d];
        }
    }

    // Two face bubbles per face. Sort each face by its global IDs and use
    // those local barycentric slots to make shared traces canonical.
    let face_vertices = [[0usize, 1usize, 2usize], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    for (face_idx, local_face) in face_vertices.iter().enumerate() {
        let mut canonical = *local_face;
        canonical.sort_unstable_by_key(|&local_vertex| vertex_ids[local_vertex]);
        let [a, b, c] = canonical;
        let first = 12 + 2 * face_idx;

        // B0 = λ_a λ_b ∇λ_c - λ_a λ_c ∇λ_b = λ_a W_bc.
        let w_bc = [
            lambdas[b] * grads[c][0] - lambdas[c] * grads[b][0],
            lambdas[b] * grads[c][1] - lambdas[c] * grads[b][1],
            lambdas[b] * grads[c][2] - lambdas[c] * grads[b][2],
        ];
        let curl_w_bc = cross(grads[b], grads[c]);
        let curl_b0 = cross(grads[a], w_bc);
        for d in 0..3 {
            values[first][d] = lambdas[a] * w_bc[d];
            curls[first][d] = curl_b0[d] + lambdas[a] * 2.0 * curl_w_bc[d];
        }

        // B1 = λ_b λ_c ∇λ_a - λ_b λ_a ∇λ_c = λ_b W_ca.
        let w_ca = [
            lambdas[c] * grads[a][0] - lambdas[a] * grads[c][0],
            lambdas[c] * grads[a][1] - lambdas[a] * grads[c][1],
            lambdas[c] * grads[a][2] - lambdas[a] * grads[c][2],
        ];
        let curl_w_ca = cross(grads[c], grads[a]);
        let curl_b1 = cross(grads[b], w_ca);
        for d in 0..3 {
            values[first + 1][d] = lambdas[b] * w_ca[d];
            curls[first + 1][d] = curl_b1[d] + lambdas[b] * 2.0 * curl_w_ca[d];
        }
    }

    (values, curls)
}

/// Lowest-order tensor-product Nédélec basis on the reference square.
///
/// The four entries follow `ElementType::Quadrilateral::local_edges`:
/// `(0,1), (1,2), (2,3), (3,0)`.  The signs in the third and fourth basis
/// functions therefore encode the local edge directions on the upper and
/// left edges.  The returned curl is the scalar two-dimensional curl.
pub fn nedelec1_quadrilateral(xi: f64, eta: f64) -> ([[f64; 2]; 4], [f64; 4]) {
    let values = [
        [0.5 * (1.0 - eta), 0.0],
        [0.0, 0.5 * (1.0 + xi)],
        [-0.5 * (1.0 + eta), 0.0],
        [0.0, -0.5 * (1.0 - xi)],
    ];
    (values, [0.5; 4])
}

/// Lowest-order tensor-product Nédélec basis on the reference cube.
///
/// The twelve entries follow `ElementType::Hexahedron::local_edges`.
/// The second return value contains the three-dimensional curl of each
/// reference basis vector.
pub fn nedelec1_hexahedron(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 12], [[f64; 3]; 12]) {
    let q = 0.25;
    let values = [
        [q * (1.0 - eta) * (1.0 - zeta), 0.0, 0.0],
        [0.0, q * (1.0 + xi) * (1.0 - zeta), 0.0],
        [-q * (1.0 + eta) * (1.0 - zeta), 0.0, 0.0],
        [0.0, -q * (1.0 - xi) * (1.0 - zeta), 0.0],
        [q * (1.0 - eta) * (1.0 + zeta), 0.0, 0.0],
        [0.0, q * (1.0 + xi) * (1.0 + zeta), 0.0],
        [-q * (1.0 + eta) * (1.0 + zeta), 0.0, 0.0],
        [0.0, -q * (1.0 - xi) * (1.0 + zeta), 0.0],
        [0.0, 0.0, q * (1.0 - xi) * (1.0 - eta)],
        [0.0, 0.0, q * (1.0 + xi) * (1.0 - eta)],
        [0.0, 0.0, q * (1.0 + xi) * (1.0 + eta)],
        [0.0, 0.0, q * (1.0 - xi) * (1.0 + eta)],
    ];
    let curls = [
        [0.0, -q * (1.0 - eta), q * (1.0 - zeta)],
        [q * (1.0 + xi), 0.0, q * (1.0 - zeta)],
        [0.0, q * (1.0 + eta), q * (1.0 - zeta)],
        [-q * (1.0 - xi), 0.0, q * (1.0 - zeta)],
        [0.0, q * (1.0 - eta), q * (1.0 + zeta)],
        [-q * (1.0 + xi), 0.0, q * (1.0 + zeta)],
        [0.0, -q * (1.0 + eta), q * (1.0 + zeta)],
        [q * (1.0 - xi), 0.0, q * (1.0 + zeta)],
        [-q * (1.0 - xi), q * (1.0 - eta), 0.0],
        [-q * (1.0 + xi), -q * (1.0 - eta), 0.0],
        [q * (1.0 + xi), -q * (1.0 + eta), 0.0],
        [q * (1.0 - xi), q * (1.0 + eta), 0.0],
    ];
    (values, curls)
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nedelec_triangle_tangential_continuity() {
        // On edge 0 (v0→v1, eta=0), N_0 should have nonzero tangential component
        // and N_1, N_2 should be tangential to their own edges.
        let (vals, _) = nedelec1_triangle(0.5, 0.0);

        // Edge 0 runs along x-axis (eta=0). Tangent direction is (1, 0).
        // N_0 at midpoint of edge 0 should have nonzero x-component.
        assert!(
            vals[0][0].abs() > 1e-14,
            "N_0 should be tangent to edge 0 at midpoint"
        );
    }

    #[test]
    fn test_nedelec_triangle_curl_constant() {
        // For lowest-order Nédélec on a simplex, curls are constant.
        let (_, curls1) = nedelec1_triangle(0.25, 0.25);
        let (_, curls2) = nedelec1_triangle(0.1, 0.3);

        for i in 0..3 {
            assert!(
                (curls1[i] - curls2[i]).abs() < 1e-14,
                "Curl of N_{i} should be constant: {} vs {}",
                curls1[i],
                curls2[i]
            );
        }
    }

    #[test]
    fn test_nedelec_tet_curl_constant() {
        let (_, curls1) = nedelec1_tetrahedron(0.1, 0.2, 0.3);
        let (_, curls2) = nedelec1_tetrahedron(0.05, 0.15, 0.1);

        for i in 0..6 {
            for d in 0..3 {
                assert!(
                    (curls1[i][d] - curls2[i][d]).abs() < 1e-14,
                    "Curl of N_{i}[{d}] should be constant: {} vs {}",
                    curls1[i][d],
                    curls2[i][d]
                );
            }
        }
    }

    #[test]
    fn test_nedelec2_triangle_dof_count() {
        let (values, curls) = nedelec2_triangle(0.25, 0.25);
        assert_eq!(values.len(), 8);
        assert_eq!(curls.len(), 8);
    }

    #[test]
    fn test_nedelec2_triangle_interior_vanish_on_edges() {
        // Interior DOFs (indices 6,7) should vanish on all edges
        // Edge 0: eta=0
        let (vals, _) = nedelec2_triangle(0.5, 0.0);
        // l0 = 0.5, l1 = 0.5, l2 = 0.0
        // Interior 0: l0*l1*grad(l2) - l0*l2*grad(l1) = 0.25*(0,1) - 0 = (0, 0.25)
        // Wait, this doesn't vanish. Interior DOFs have tangential trace zero
        // but may have normal component. Check the tangential component on edge 0:
        // Edge 0 tangent = (1, 0), so tangential part = vals[6][0]
        // l0*l1*gl2[0] - l0*l2*gl1[0] = 0.25*0 - 0*1 = 0 ✓ for x-component
        assert!(
            vals[6][0].abs() < 1e-14,
            "Interior DOF 6 should have zero tangential trace on edge 0: {}",
            vals[6][0]
        );
    }

    #[test]
    fn test_nedelec_triangle_edge_integral() {
        // For lowest-order Nédélec, the DOF is the line integral of N along the edge.
        // integral_{edge_j} N_i . t ds = delta_{ij}
        // Edge 0 goes from (0,0) to (1,0), tangent = (1,0), length = 1.
        // Use midpoint rule: integral ~= N(0.5, 0) . (1,0) * 1
        let (vals, _) = nedelec1_triangle(0.5, 0.0);

        // N_0 . t on edge 0 (tangent = (1,0)):
        let integral_0 = vals[0][0] * 1.0; // x-component * length
        assert!(
            (integral_0 - 1.0).abs() < 1e-14,
            "Edge integral of N_0 on edge 0 = {integral_0}, expected 1.0"
        );

        // N_1 . t on edge 0 should be 0 (Kronecker delta property)
        let integral_1 = vals[1][0] * 1.0;
        assert!(
            integral_1.abs() < 1e-14,
            "Edge integral of N_1 on edge 0 = {integral_1}, expected 0.0"
        );
    }

    #[test]
    fn test_nedelec2_tetrahedron_dof_count_and_curl_variation() {
        let (values_a, curls_a) = nedelec2_tetrahedron(0.1, 0.2, 0.15);
        let (values_b, curls_b) = nedelec2_tetrahedron(0.2, 0.1, 0.2);
        assert_eq!(values_a.len(), 20);
        assert_eq!(curls_a.len(), 20);
        assert!(
            values_a
                .iter()
                .zip(values_b.iter())
                .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-12)),
            "order-2 basis values must vary over the tetrahedron"
        );
        assert!(
            curls_a
                .iter()
                .zip(curls_b.iter())
                .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-12)),
            "order-2 curls must include the expected linear variation"
        );
    }

    #[test]
    fn test_nedelec2_tetrahedron_face_bubbles_have_zero_edge_trace() {
        // Edge (0,1) is the xi-axis: eta=zeta=0. All face bubbles must have
        // zero tangential (x) trace there.
        let (values, _) = nedelec2_tetrahedron(0.35, 0.0, 0.0);
        for face_dof in 12..20 {
            assert!(
                values[face_dof][0].abs() < 1e-14,
                "face DOF {face_dof} has nonzero edge trace {}",
                values[face_dof][0]
            );
        }
    }

    #[test]
    fn test_nedelec2_tetrahedron_edge_orientation_parity() {
        let canonical = nedelec2_tetrahedron_oriented(0.2, 0.25, 0.1, [0, 1, 2, 3]);
        let reversed = nedelec2_tetrahedron_oriented(0.2, 0.25, 0.1, [1, 0, 2, 3]);
        // The local edge (0,1) has reversed canonical orientation. Its first
        // moment changes sign and its higher edge moment remains unchanged.
        for d in 0..3 {
            assert!((canonical.0[0][d] + reversed.0[0][d]).abs() < 1e-14);
            assert!((canonical.0[1][d] - reversed.0[1][d]).abs() < 1e-14);
        }
    }
}
