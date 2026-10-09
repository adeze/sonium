//! Lagrange basis functions for finite elements
//!
//! Provides P1, P2, P3 (simplex) and Q1, Q2 (tensor-product) basis functions.

use crate::mesh::ElementType;

/// Polynomial degree for Lagrange elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolynomialDegree {
    /// Linear (P1/Q1)
    P1,
    /// Quadratic (P2/Q2)
    P2,
    /// Cubic (P3/Q3)
    P3,
}

impl PolynomialDegree {
    pub fn degree(&self) -> usize {
        match self {
            PolynomialDegree::P1 => 1,
            PolynomialDegree::P2 => 2,
            PolynomialDegree::P3 => 3,
        }
    }
}

/// Number of nodes for a given element type and polynomial degree
pub fn num_nodes(element_type: ElementType, degree: PolynomialDegree) -> usize {
    match (element_type, degree) {
        // Triangles: (p+1)(p+2)/2
        (ElementType::Triangle, PolynomialDegree::P1) => 3,
        (ElementType::Triangle, PolynomialDegree::P2) => 6,
        (ElementType::Triangle, PolynomialDegree::P3) => 10,

        // Quadrilaterals: (p+1)^2
        (ElementType::Quadrilateral, PolynomialDegree::P1) => 4,
        (ElementType::Quadrilateral, PolynomialDegree::P2) => 9,
        (ElementType::Quadrilateral, PolynomialDegree::P3) => 16,

        // Tetrahedra: (p+1)(p+2)(p+3)/6
        (ElementType::Tetrahedron, PolynomialDegree::P1) => 4,
        (ElementType::Tetrahedron, PolynomialDegree::P2) => 10,
        (ElementType::Tetrahedron, PolynomialDegree::P3) => 20,

        // Hexahedra: (p+1)^3
        (ElementType::Hexahedron, PolynomialDegree::P1) => 8,
        (ElementType::Hexahedron, PolynomialDegree::P2) => 27,
        (ElementType::Hexahedron, PolynomialDegree::P3) => 64,
    }
}

/// Evaluate P1 triangle basis functions at (xi, eta)
/// Reference triangle: (0,0), (1,0), (0,1)
/// Returns [N0, N1, N2]
pub fn p1_triangle(xi: f64, eta: f64) -> [f64; 3] {
    [1.0 - xi - eta, xi, eta]
}

/// Evaluate P1 triangle basis function gradients (constant)
/// Returns [[dN0/dxi, dN0/deta], [dN1/dxi, dN1/deta], [dN2/dxi, dN2/deta]]
pub fn p1_triangle_grad() -> [[f64; 2]; 3] {
    [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]
}

/// Evaluate P2 triangle basis functions at (xi, eta)
/// Nodes: 3 vertices + 3 edge midpoints
/// Returns [N0, N1, N2, N01, N12, N20]
pub fn p2_triangle(xi: f64, eta: f64) -> [f64; 6] {
    let l0 = 1.0 - xi - eta;
    let l1 = xi;
    let l2 = eta;

    [
        l0 * (2.0 * l0 - 1.0), // vertex 0
        l1 * (2.0 * l1 - 1.0), // vertex 1
        l2 * (2.0 * l2 - 1.0), // vertex 2
        4.0 * l0 * l1,         // edge 0-1
        4.0 * l1 * l2,         // edge 1-2
        4.0 * l2 * l0,         // edge 2-0
    ]
}

/// Evaluate P2 triangle basis function gradients at (xi, eta)
pub fn p2_triangle_grad(xi: f64, eta: f64) -> [[f64; 2]; 6] {
    let l0 = 1.0 - xi - eta;
    let l1 = xi;
    let l2 = eta;

    // dl0/dxi = -1, dl0/deta = -1
    // dl1/dxi = 1,  dl1/deta = 0
    // dl2/dxi = 0,  dl2/deta = 1
    [
        // dN0/d(xi,eta) where N0 = l0*(2*l0-1)
        // dN0/dxi = (2*l0-1)*(-1) + l0*2*(-1) = 1 - 4*l0
        [1.0 - 4.0 * l0, 1.0 - 4.0 * l0],
        // dN1/d(xi,eta) where N1 = l1*(2*l1-1)
        [4.0 * l1 - 1.0, 0.0],
        // dN2/d(xi,eta) where N2 = l2*(2*l2-1)
        [0.0, 4.0 * l2 - 1.0],
        // dN01/d(xi,eta) where N01 = 4*l0*l1
        [4.0 * (l0 - l1), -4.0 * l1],
        // dN12/d(xi,eta) where N12 = 4*l1*l2
        [4.0 * l2, 4.0 * l1],
        // dN20/d(xi,eta) where N20 = 4*l2*l0
        [-4.0 * l2, 4.0 * (l0 - l2)],
    ]
}

const P3_TRIANGLE_EXPONENTS: [[usize; 4]; 10] = [
    [3, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 3, 0],
    [2, 1, 0, 0],
    [1, 2, 0, 0],
    [0, 2, 1, 0],
    [0, 1, 2, 0],
    [1, 0, 2, 0],
    [2, 0, 1, 0],
    [1, 1, 1, 0],
];

/// Evaluate cubic triangular Lagrange basis functions at `(xi, eta)`.
///
/// The ordering is three vertices, two nodes on each local edge, then the
/// interior node. Edge-node ordering follows `ElementType::local_edges`.
pub fn p3_triangle(xi: f64, eta: f64) -> [f64; 10] {
    let barycentric = [1.0 - xi - eta, xi, eta, 0.0];
    let gradients = [
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    P3_TRIANGLE_EXPONENTS
        .map(|exponents| cubic_barycentric_value(exponents, &barycentric, &gradients).0)
}

/// Evaluate cubic triangular Lagrange basis gradients at `(xi, eta)`.
pub fn p3_triangle_grad(xi: f64, eta: f64) -> [[f64; 2]; 10] {
    let barycentric = [1.0 - xi - eta, xi, eta, 0.0];
    let gradients = [
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    P3_TRIANGLE_EXPONENTS.map(|exponents| {
        let gradient = cubic_barycentric_value(exponents, &barycentric, &gradients).1;
        [gradient[0], gradient[1]]
    })
}

/// Evaluate Q1 quadrilateral basis functions at (xi, eta)
/// Reference quad: [-1,1] x [-1,1]
/// Nodes: (−1,−1), (1,−1), (1,1), (−1,1)
pub fn q1_quadrilateral(xi: f64, eta: f64) -> [f64; 4] {
    [
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    ]
}

/// Evaluate Q1 quadrilateral basis function gradients at (xi, eta)
pub fn q1_quadrilateral_grad(xi: f64, eta: f64) -> [[f64; 2]; 4] {
    [
        [-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)],
        [0.25 * (1.0 - eta), -0.25 * (1.0 + xi)],
        [0.25 * (1.0 + eta), 0.25 * (1.0 + xi)],
        [-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)],
    ]
}

/// Evaluate P1 tetrahedron basis functions at (xi, eta, zeta)
/// Reference tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
pub fn p1_tetrahedron(xi: f64, eta: f64, zeta: f64) -> [f64; 4] {
    [1.0 - xi - eta - zeta, xi, eta, zeta]
}

/// Evaluate P1 tetrahedron basis function gradients (constant)
pub fn p1_tetrahedron_grad() -> [[f64; 3]; 4] {
    [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

/// Evaluate Q1 hexahedron basis functions at (xi, eta, zeta)
/// Reference hex: [-1,1]^3
pub fn q1_hexahedron(xi: f64, eta: f64, zeta: f64) -> [f64; 8] {
    let xm = 1.0 - xi;
    let xp = 1.0 + xi;
    let em = 1.0 - eta;
    let ep = 1.0 + eta;
    let zm = 1.0 - zeta;
    let zp = 1.0 + zeta;

    [
        0.125 * xm * em * zm,
        0.125 * xp * em * zm,
        0.125 * xp * ep * zm,
        0.125 * xm * ep * zm,
        0.125 * xm * em * zp,
        0.125 * xp * em * zp,
        0.125 * xp * ep * zp,
        0.125 * xm * ep * zp,
    ]
}

/// Evaluate Q1 hexahedron basis function gradients at (xi, eta, zeta)
pub fn q1_hexahedron_grad(xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 8] {
    let xm = 1.0 - xi;
    let xp = 1.0 + xi;
    let em = 1.0 - eta;
    let ep = 1.0 + eta;
    let zm = 1.0 - zeta;
    let zp = 1.0 + zeta;

    [
        [-0.125 * em * zm, -0.125 * xm * zm, -0.125 * xm * em],
        [0.125 * em * zm, -0.125 * xp * zm, -0.125 * xp * em],
        [0.125 * ep * zm, 0.125 * xp * zm, -0.125 * xp * ep],
        [-0.125 * ep * zm, 0.125 * xm * zm, -0.125 * xm * ep],
        [-0.125 * em * zp, -0.125 * xm * zp, 0.125 * xm * em],
        [0.125 * em * zp, -0.125 * xp * zp, 0.125 * xp * em],
        [0.125 * ep * zp, 0.125 * xp * zp, 0.125 * xp * ep],
        [-0.125 * ep * zp, 0.125 * xm * zp, 0.125 * xm * ep],
    ]
}

/// Evaluate P2 tetrahedron basis functions at (xi, eta, zeta).
///
/// Reference tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
/// Node ordering: 4 vertices, then 6 edge midpoints following the
/// canonical edge order `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`.
pub fn p2_tetrahedron(xi: f64, eta: f64, zeta: f64) -> [f64; 10] {
    let l0 = 1.0 - xi - eta - zeta;
    let l1 = xi;
    let l2 = eta;
    let l3 = zeta;

    [
        l0 * (2.0 * l0 - 1.0),
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l0 * l1,
        4.0 * l0 * l2,
        4.0 * l0 * l3,
        4.0 * l1 * l2,
        4.0 * l1 * l3,
        4.0 * l2 * l3,
    ]
}

/// Evaluate P2 tetrahedron basis function gradients at (xi, eta, zeta).
pub fn p2_tetrahedron_grad(xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 10] {
    let l = [1.0 - xi - eta - zeta, xi, eta, zeta];
    let dl = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    let mut grads = [[0.0; 3]; 10];
    for a in 0..4 {
        let s = 4.0 * l[a] - 1.0;
        for d in 0..3 {
            grads[a][d] = s * dl[a][d];
        }
    }
    for (k, (a, b)) in edges.iter().enumerate() {
        for d in 0..3 {
            grads[4 + k][d] = 4.0 * (dl[*a][d] * l[*b] + l[*a] * dl[*b][d]);
        }
    }
    grads
}

/// 1D quadratic Lagrange shape functions at nodes `{-1, 0, +1}`.
/// Returns `[L_{-1}(s), L_0(s), L_{+1}(s)]`.
#[inline]
fn q2_1d(s: f64) -> [f64; 3] {
    [
        0.5 * s * (s - 1.0),
        (1.0 - s) * (1.0 + s),
        0.5 * s * (s + 1.0),
    ]
}

/// Derivatives of the 1D quadratic Lagrange shape functions at nodes `{-1, 0, +1}`.
#[inline]
fn q2_1d_grad(s: f64) -> [f64; 3] {
    [s - 0.5, -2.0 * s, s + 0.5]
}

/// Evaluate Q2 (biquadratic) quadrilateral basis functions at (xi, eta).
///
/// Reference quad: `[-1, 1] × [-1, 1]`. Node ordering matches the canonical
/// FEM convention used elsewhere in the workspace:
/// 4 vertices, 4 edge midpoints (in `local_edges` order), then the interior
/// center node.
///
/// ```text
///        v3 ---- e2 ---- v2
///         |              |
///        e3      c       e1
///         |              |
///        v0 ---- e0 ---- v1
/// ```
pub fn q2_quadrilateral(xi: f64, eta: f64) -> [f64; 9] {
    let lx = q2_1d(xi);
    let ly = q2_1d(eta);

    // Map index 0/1/2 -> position xi/eta in {-1, 0, +1}.
    [
        // Vertices
        lx[0] * ly[0], // v0 (-1,-1)
        lx[2] * ly[0], // v1 ( 1,-1)
        lx[2] * ly[2], // v2 ( 1, 1)
        lx[0] * ly[2], // v3 (-1, 1)
        // Edge midpoints (following local_edges: (0,1),(1,2),(2,3),(3,0))
        lx[1] * ly[0], // e0 ( 0,-1)
        lx[2] * ly[1], // e1 ( 1, 0)
        lx[1] * ly[2], // e2 ( 0, 1)
        lx[0] * ly[1], // e3 (-1, 0)
        // Center
        lx[1] * ly[1], // c  ( 0, 0)
    ]
}

/// Evaluate Q2 quadrilateral basis function gradients at (xi, eta).
pub fn q2_quadrilateral_grad(xi: f64, eta: f64) -> [[f64; 2]; 9] {
    let lx = q2_1d(xi);
    let ly = q2_1d(eta);
    let dx = q2_1d_grad(xi);
    let dy = q2_1d_grad(eta);

    // Each row: [d/dxi, d/deta]
    [
        [dx[0] * ly[0], lx[0] * dy[0]],
        [dx[2] * ly[0], lx[2] * dy[0]],
        [dx[2] * ly[2], lx[2] * dy[2]],
        [dx[0] * ly[2], lx[0] * dy[2]],
        [dx[1] * ly[0], lx[1] * dy[0]],
        [dx[2] * ly[1], lx[2] * dy[1]],
        [dx[1] * ly[2], lx[1] * dy[2]],
        [dx[0] * ly[1], lx[0] * dy[1]],
        [dx[1] * ly[1], lx[1] * dy[1]],
    ]
}

/// Cubic one-dimensional Lagrange values and derivatives on `[-1, 1]`.
/// The nodes are ordered `-1, -1/3, 1/3, 1`.
fn q3_1d(s: f64) -> ([f64; 4], [f64; 4]) {
    let nodes = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let mut values = [0.0; 4];
    let mut gradients = [0.0; 4];
    for i in 0..4 {
        let mut denominator = 1.0;
        for j in 0..4 {
            if i != j {
                denominator *= nodes[i] - nodes[j];
            }
        }
        let mut numerator = 1.0;
        for (j, &node) in nodes.iter().enumerate() {
            if i != j {
                numerator *= s - node;
            }
        }
        values[i] = numerator / denominator;
        let mut derivative = 0.0;
        for omitted in 0..4 {
            if omitted == i {
                continue;
            }
            let mut term = 1.0;
            for (j, &node) in nodes.iter().enumerate() {
                if j != i && j != omitted {
                    term *= s - node;
                }
            }
            derivative += term;
        }
        gradients[i] = derivative / denominator;
    }
    (values, gradients)
}

const Q3_QUAD_NODE_INDICES: [(usize, usize); 16] = [
    (0, 0),
    (3, 0),
    (3, 3),
    (0, 3),
    (1, 0),
    (2, 0),
    (3, 1),
    (3, 2),
    (2, 3),
    (1, 3),
    (0, 2),
    (0, 1),
    (1, 1),
    (2, 1),
    (2, 2),
    (1, 2),
];

/// Evaluate cubic quadrilateral Lagrange basis functions at `(xi, eta)`.
pub fn q3_quadrilateral(xi: f64, eta: f64) -> [f64; 16] {
    let (lx, _) = q3_1d(xi);
    let (ly, _) = q3_1d(eta);
    Q3_QUAD_NODE_INDICES.map(|(i, j)| lx[i] * ly[j])
}

/// Evaluate cubic quadrilateral Lagrange basis gradients at `(xi, eta)`.
pub fn q3_quadrilateral_grad(xi: f64, eta: f64) -> [[f64; 2]; 16] {
    let (lx, dx) = q3_1d(xi);
    let (ly, dy) = q3_1d(eta);
    Q3_QUAD_NODE_INDICES.map(|(i, j)| [dx[i] * ly[j], lx[i] * dy[j]])
}

/// Local (xi, eta, zeta) coordinates of the 27 Q2-hex Lagrange nodes, in the
/// canonical workspace order used by [`q2_hexahedron`].
const Q2_HEX_NODE_INDICES: [(usize, usize, usize); 27] = [
    // Vertices (z = -1 bottom, then z = +1 top)
    (0, 0, 0), // v0 (-1,-1,-1)
    (2, 0, 0), // v1 ( 1,-1,-1)
    (2, 2, 0), // v2 ( 1, 1,-1)
    (0, 2, 0), // v3 (-1, 1,-1)
    (0, 0, 2), // v4 (-1,-1, 1)
    (2, 0, 2), // v5 ( 1,-1, 1)
    (2, 2, 2), // v6 ( 1, 1, 1)
    (0, 2, 2), // v7 (-1, 1, 1)
    // Edges (in `local_edges` order: bottom 4, top 4, vertical 4)
    (1, 0, 0), // e0  (0,1)   ( 0,-1,-1)
    (2, 1, 0), // e1  (1,2)   ( 1, 0,-1)
    (1, 2, 0), // e2  (2,3)   ( 0, 1,-1)
    (0, 1, 0), // e3  (3,0)   (-1, 0,-1)
    (1, 0, 2), // e4  (4,5)   ( 0,-1, 1)
    (2, 1, 2), // e5  (5,6)   ( 1, 0, 1)
    (1, 2, 2), // e6  (6,7)   ( 0, 1, 1)
    (0, 1, 2), // e7  (7,4)   (-1, 0, 1)
    (0, 0, 1), // e8  (0,4)   (-1,-1, 0)
    (2, 0, 1), // e9  (1,5)   ( 1,-1, 0)
    (2, 2, 1), // e10 (2,6)   ( 1, 1, 0)
    (0, 2, 1), // e11 (3,7)   (-1, 1, 0)
    // Face centres (in `local_faces` order)
    (1, 1, 0), // f0 bottom (z-)
    (1, 1, 2), // f1 top    (z+)
    (1, 0, 1), // f2 front  (y-)
    (1, 2, 1), // f3 back   (y+)
    (0, 1, 1), // f4 left   (x-)
    (2, 1, 1), // f5 right  (x+)
    // Volume centre
    (1, 1, 1),
];

/// Evaluate Q2 (triquadratic) hexahedron basis functions at (xi, eta, zeta).
///
/// Reference hex: `[-1, 1]^3`. Node ordering: 8 vertices, 12 edge midpoints
/// (in `ElementType::Hexahedron.local_edges` order), 6 face centres (in
/// `local_faces` order), then the volume centre.
pub fn q2_hexahedron(xi: f64, eta: f64, zeta: f64) -> [f64; 27] {
    let lx = q2_1d(xi);
    let ly = q2_1d(eta);
    let lz = q2_1d(zeta);

    let mut n = [0.0; 27];
    for (k, (i, j, m)) in Q2_HEX_NODE_INDICES.iter().enumerate() {
        n[k] = lx[*i] * ly[*j] * lz[*m];
    }
    n
}

/// Evaluate Q2 hexahedron basis function gradients at (xi, eta, zeta).
pub fn q2_hexahedron_grad(xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 27] {
    let lx = q2_1d(xi);
    let ly = q2_1d(eta);
    let lz = q2_1d(zeta);
    let dx = q2_1d_grad(xi);
    let dy = q2_1d_grad(eta);
    let dz = q2_1d_grad(zeta);

    let mut g = [[0.0; 3]; 27];
    for (k, (i, j, m)) in Q2_HEX_NODE_INDICES.iter().enumerate() {
        g[k][0] = dx[*i] * ly[*j] * lz[*m];
        g[k][1] = lx[*i] * dy[*j] * lz[*m];
        g[k][2] = lx[*i] * ly[*j] * dz[*m];
    }
    g
}

const Q3_HEX_NODE_INDICES: [(usize, usize, usize); 64] = [
    // Vertices.
    (0, 0, 0),
    (3, 0, 0),
    (3, 3, 0),
    (0, 3, 0),
    (0, 0, 3),
    (3, 0, 3),
    (3, 3, 3),
    (0, 3, 3),
    // Two nodes on each local edge.
    (1, 0, 0),
    (2, 0, 0),
    (3, 1, 0),
    (3, 2, 0),
    (2, 3, 0),
    (1, 3, 0),
    (0, 2, 0),
    (0, 1, 0),
    (1, 0, 3),
    (2, 0, 3),
    (3, 1, 3),
    (3, 2, 3),
    (2, 3, 3),
    (1, 3, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 0, 1),
    (0, 0, 2),
    (3, 0, 1),
    (3, 0, 2),
    (3, 3, 1),
    (3, 3, 2),
    (0, 3, 1),
    (0, 3, 2),
    // Four interior nodes on each local face.
    (1, 1, 0),
    (2, 1, 0),
    (2, 2, 0),
    (1, 2, 0),
    (1, 1, 3),
    (2, 1, 3),
    (2, 2, 3),
    (1, 2, 3),
    (1, 0, 1),
    (2, 0, 1),
    (2, 0, 2),
    (1, 0, 2),
    (1, 3, 1),
    (2, 3, 1),
    (2, 3, 2),
    (1, 3, 2),
    (0, 1, 1),
    (0, 2, 1),
    (0, 2, 2),
    (0, 1, 2),
    (3, 1, 1),
    (3, 2, 1),
    (3, 2, 2),
    (3, 1, 2),
    // Eight volume-interior nodes.
    (1, 1, 1),
    (2, 1, 1),
    (2, 2, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 1, 2),
    (2, 2, 2),
    (1, 2, 2),
];

/// Evaluate cubic hexahedral Lagrange basis functions at `(xi, eta, zeta)`.
pub fn q3_hexahedron(xi: f64, eta: f64, zeta: f64) -> [f64; 64] {
    let (lx, _) = q3_1d(xi);
    let (ly, _) = q3_1d(eta);
    let (lz, _) = q3_1d(zeta);
    Q3_HEX_NODE_INDICES.map(|(i, j, k)| lx[i] * ly[j] * lz[k])
}

/// Evaluate cubic hexahedral Lagrange basis gradients at `(xi, eta, zeta)`.
pub fn q3_hexahedron_grad(xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 64] {
    let (lx, dx) = q3_1d(xi);
    let (ly, dy) = q3_1d(eta);
    let (lz, dz) = q3_1d(zeta);
    Q3_HEX_NODE_INDICES.map(|(i, j, k)| {
        [
            dx[i] * ly[j] * lz[k],
            lx[i] * dy[j] * lz[k],
            lx[i] * ly[j] * dz[k],
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_nodes() {
        assert_eq!(num_nodes(ElementType::Triangle, PolynomialDegree::P1), 3);
        assert_eq!(num_nodes(ElementType::Triangle, PolynomialDegree::P2), 6);
        assert_eq!(
            num_nodes(ElementType::Quadrilateral, PolynomialDegree::P1),
            4
        );
        assert_eq!(num_nodes(ElementType::Tetrahedron, PolynomialDegree::P1), 4);
        assert_eq!(num_nodes(ElementType::Hexahedron, PolynomialDegree::P1), 8);
    }

    #[test]
    fn test_p1_triangle_partition_of_unity() {
        // Shape functions should sum to 1 at any point
        let (xi, eta) = (0.3, 0.2);
        let n = p1_triangle(xi, eta);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_p1_triangle_kronecker() {
        // N_i(vertex_j) = delta_ij
        let vertices = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        for (i, &(xi, eta)) in vertices.iter().enumerate() {
            let n = p1_triangle(xi, eta);
            for (j, &nj) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((nj - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_q1_quad_partition_of_unity() {
        let (xi, eta) = (0.5, -0.3);
        let n = q1_quadrilateral(xi, eta);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_q1_quad_kronecker() {
        let vertices = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        for (i, &(xi, eta)) in vertices.iter().enumerate() {
            let n = q1_quadrilateral(xi, eta);
            for (j, &nj) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((nj - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_p1_tet_partition_of_unity() {
        let (xi, eta, zeta) = (0.1, 0.2, 0.3);
        let n = p1_tetrahedron(xi, eta, zeta);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_q1_hex_partition_of_unity() {
        let (xi, eta, zeta) = (0.2, -0.4, 0.6);
        let n = q1_hexahedron(xi, eta, zeta);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_p2_tetrahedron_partition_of_unity() {
        let (xi, eta, zeta) = (0.1, 0.2, 0.3);
        let n = p2_tetrahedron(xi, eta, zeta);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "P2 tet sum = {sum}");
    }

    #[test]
    fn test_p2_tetrahedron_kronecker() {
        let nodes = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ];
        for (i, &(xi, eta, zeta)) in nodes.iter().enumerate() {
            let n = p2_tetrahedron(xi, eta, zeta);
            for (j, &nj) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (nj - expected).abs() < 1e-14,
                    "P2 tet N_{j} at node {i} = {nj}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_p2_tetrahedron_grad_partition() {
        // Sum of gradients must vanish (partition of unity)
        let g = p2_tetrahedron_grad(0.13, 0.21, 0.37);
        let mut s = [0.0; 3];
        for row in &g {
            for (d, sd) in s.iter_mut().enumerate() {
                *sd += row[d];
            }
        }
        for (d, sd) in s.iter().enumerate() {
            assert!(sd.abs() < 1e-14, "grad sum[{d}] = {sd}");
        }
    }

    #[test]
    fn test_q2_quad_partition_of_unity() {
        for &(xi, eta) in &[(0.0, 0.0), (0.3, -0.5), (-0.7, 0.4), (1.0, -1.0)] {
            let n = q2_quadrilateral(xi, eta);
            let sum: f64 = n.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "Q2 quad sum at ({xi},{eta}) = {sum}"
            );
        }
    }

    #[test]
    fn test_q2_quad_kronecker() {
        // Node positions matching q2_quadrilateral ordering: 4 vertices, 4 edges, center.
        let nodes = [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, 0.0),
        ];
        for (i, &(xi, eta)) in nodes.iter().enumerate() {
            let n = q2_quadrilateral(xi, eta);
            for (j, &nj) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (nj - expected).abs() < 1e-14,
                    "Q2 quad N_{j} at node {i} = {nj}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_q2_quad_grad_partition() {
        let g = q2_quadrilateral_grad(0.21, -0.34);
        let mut s = [0.0; 2];
        for row in &g {
            s[0] += row[0];
            s[1] += row[1];
        }
        assert!(s[0].abs() < 1e-13);
        assert!(s[1].abs() < 1e-13);
    }

    #[test]
    fn test_q2_hex_partition_of_unity() {
        for &(xi, eta, zeta) in &[
            (0.0, 0.0, 0.0),
            (0.3, -0.5, 0.7),
            (-0.9, 0.4, -0.1),
            (1.0, 1.0, 1.0),
        ] {
            let n = q2_hexahedron(xi, eta, zeta);
            let sum: f64 = n.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Q2 hex sum at ({xi},{eta},{zeta}) = {sum}"
            );
        }
    }

    #[test]
    fn test_q2_hex_kronecker() {
        // Build node positions in the same order as q2_hexahedron.
        let coord = [-1.0, 0.0, 1.0];
        let mut nodes = [(0.0, 0.0, 0.0); 27];
        for (k, (i, j, m)) in Q2_HEX_NODE_INDICES.iter().enumerate() {
            nodes[k] = (coord[*i], coord[*j], coord[*m]);
        }
        for (i, &(xi, eta, zeta)) in nodes.iter().enumerate() {
            let n = q2_hexahedron(xi, eta, zeta);
            for (j, &nj) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (nj - expected).abs() < 1e-13,
                    "Q2 hex N_{j} at node {i} = {nj}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_q2_hex_grad_partition() {
        let g = q2_hexahedron_grad(0.21, -0.34, 0.57);
        let mut s = [0.0; 3];
        for row in &g {
            for (d, sd) in s.iter_mut().enumerate() {
                *sd += row[d];
            }
        }
        for (d, sd) in s.iter().enumerate() {
            assert!(sd.abs() < 1e-12, "grad sum[{d}] = {sd}");
        }
    }

    /// Central-difference verification: analytic gradient must match the
    /// numerical derivative of the basis values to high precision. Catches
    /// sign/index/tensor-product errors that partition-sum tests would miss.
    #[test]
    fn test_q2_quad_grad_finite_difference() {
        let (xi0, eta0) = (0.31, -0.47);
        let h = 1e-5;
        let analytic = q2_quadrilateral_grad(xi0, eta0);
        let plus_xi = q2_quadrilateral(xi0 + h, eta0);
        let minus_xi = q2_quadrilateral(xi0 - h, eta0);
        let plus_eta = q2_quadrilateral(xi0, eta0 + h);
        let minus_eta = q2_quadrilateral(xi0, eta0 - h);
        for i in 0..9 {
            let num_xi = (plus_xi[i] - minus_xi[i]) / (2.0 * h);
            let num_eta = (plus_eta[i] - minus_eta[i]) / (2.0 * h);
            assert!(
                (analytic[i][0] - num_xi).abs() < 1e-7,
                "Q2 quad dN_{i}/dxi: analytic={}, fd={}",
                analytic[i][0],
                num_xi
            );
            assert!(
                (analytic[i][1] - num_eta).abs() < 1e-7,
                "Q2 quad dN_{i}/deta: analytic={}, fd={}",
                analytic[i][1],
                num_eta
            );
        }
    }

    #[test]
    fn test_q2_hex_grad_finite_difference() {
        let (xi0, eta0, zeta0) = (0.21, -0.34, 0.57);
        let h = 1e-5;
        let analytic = q2_hexahedron_grad(xi0, eta0, zeta0);
        let pts = [
            (
                q2_hexahedron(xi0 + h, eta0, zeta0),
                q2_hexahedron(xi0 - h, eta0, zeta0),
            ),
            (
                q2_hexahedron(xi0, eta0 + h, zeta0),
                q2_hexahedron(xi0, eta0 - h, zeta0),
            ),
            (
                q2_hexahedron(xi0, eta0, zeta0 + h),
                q2_hexahedron(xi0, eta0, zeta0 - h),
            ),
        ];
        for (i, row) in analytic.iter().enumerate() {
            for (d, &expected) in row.iter().enumerate() {
                let num = (pts[d].0[i] - pts[d].1[i]) / (2.0 * h);
                assert!(
                    (expected - num).abs() < 1e-7,
                    "Q2 hex dN_{i}/d[{d}]: analytic={expected}, fd={num}"
                );
            }
        }
    }

    #[test]
    fn test_p2_tetrahedron_grad_finite_difference() {
        let (xi0, eta0, zeta0) = (0.13, 0.21, 0.37);
        let h = 1e-6;
        let analytic = p2_tetrahedron_grad(xi0, eta0, zeta0);
        let pts = [
            (
                p2_tetrahedron(xi0 + h, eta0, zeta0),
                p2_tetrahedron(xi0 - h, eta0, zeta0),
            ),
            (
                p2_tetrahedron(xi0, eta0 + h, zeta0),
                p2_tetrahedron(xi0, eta0 - h, zeta0),
            ),
            (
                p2_tetrahedron(xi0, eta0, zeta0 + h),
                p2_tetrahedron(xi0, eta0, zeta0 - h),
            ),
        ];
        for (i, row) in analytic.iter().enumerate() {
            for (d, &expected) in row.iter().enumerate() {
                let num = (pts[d].0[i] - pts[d].1[i]) / (2.0 * h);
                assert!(
                    (expected - num).abs() < 1e-6,
                    "P2 tet dN_{i}/d[{d}]: analytic={expected}, fd={num}"
                );
            }
        }
    }
}

/// Evaluate the cubic Lagrange basis on the reference tetrahedron.
///
/// The ordering is vertices, the two interior nodes on each canonical edge,
/// then one interior node on each canonical face. This matches
/// [`ElementType::local_edges`](crate::mesh::ElementType::local_edges) and
/// [`ElementType::local_faces`](crate::mesh::ElementType::local_faces).
pub fn p3_tetrahedron(xi: f64, eta: f64, zeta: f64) -> [f64; 20] {
    let l = [1.0 - xi - eta - zeta, xi, eta, zeta];
    let dl = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let alpha = [
        [3, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [2, 1, 0, 0],
        [1, 2, 0, 0],
        [2, 0, 1, 0],
        [1, 0, 2, 0],
        [2, 0, 0, 1],
        [1, 0, 0, 2],
        [0, 2, 1, 0],
        [0, 1, 2, 0],
        [0, 2, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 2, 1],
        [0, 0, 1, 2],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ];
    alpha.map(|exponents| cubic_barycentric_value(exponents, &l, &dl).0)
}

/// Evaluate cubic tetrahedral Lagrange basis gradients in reference space.
pub fn p3_tetrahedron_grad(xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 20] {
    let l = [1.0 - xi - eta - zeta, xi, eta, zeta];
    let dl = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let alpha = [
        [3, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [2, 1, 0, 0],
        [1, 2, 0, 0],
        [2, 0, 1, 0],
        [1, 0, 2, 0],
        [2, 0, 0, 1],
        [1, 0, 0, 2],
        [0, 2, 1, 0],
        [0, 1, 2, 0],
        [0, 2, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 2, 1],
        [0, 0, 1, 2],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ];
    alpha.map(|exponents| cubic_barycentric_value(exponents, &l, &dl).1)
}

/// Evaluate cubic tetrahedral Lagrange basis Hessians in reference space.
pub fn p3_tetrahedron_hessian(xi: f64, eta: f64, zeta: f64) -> [[[f64; 3]; 3]; 20] {
    let barycentric = [1.0 - xi - eta - zeta, xi, eta, zeta];
    let gradients = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let exponents = [
        [3, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [2, 1, 0, 0],
        [1, 2, 0, 0],
        [2, 0, 1, 0],
        [1, 0, 2, 0],
        [2, 0, 0, 1],
        [1, 0, 0, 2],
        [0, 2, 1, 0],
        [0, 1, 2, 0],
        [0, 2, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 2, 1],
        [0, 0, 1, 2],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ];
    exponents.map(|alpha| cubic_barycentric_hessian(alpha, &barycentric, &gradients))
}

fn cubic_barycentric_hessian(
    exponents: [usize; 4],
    barycentric: &[f64; 4],
    gradients: &[[f64; 3]; 4],
) -> [[f64; 3]; 3] {
    let coefficient = exponents.iter().fold(1.0, |coefficient, &exponent| {
        coefficient / (1..=exponent).map(|value| value as f64).product::<f64>()
    });
    let mut hessian = [[0.0; 3]; 3];
    for active in 0..4 {
        for active_factor in 0..exponents[active] {
            for row in 0..3 {
                for second in 0..4 {
                    for second_factor in 0..exponents[second] {
                        if active == second && active_factor == second_factor {
                            continue;
                        }
                        for column in 0..3 {
                            let mut term = coefficient
                                * 9.0
                                * gradients[active][row]
                                * gradients[second][column];
                            for coordinate in 0..4 {
                                for factor in 0..exponents[coordinate] {
                                    if (coordinate == active && factor == active_factor)
                                        || (coordinate == second && factor == second_factor)
                                    {
                                        continue;
                                    }
                                    term *= 3.0 * barycentric[coordinate] - factor as f64;
                                }
                            }
                            hessian[row][column] += term;
                        }
                    }
                }
            }
        }
    }
    hessian
}

fn cubic_barycentric_value(
    exponents: [usize; 4],
    barycentric: &[f64; 4],
    gradients: &[[f64; 3]; 4],
) -> (f64, [f64; 3]) {
    let coefficient = exponents.iter().fold(1.0, |coefficient, &exponent| {
        coefficient / (1..=exponent).map(|value| value as f64).product::<f64>()
    });
    let mut value = coefficient;
    for (coordinate, &exponent) in exponents.iter().enumerate() {
        for factor in 0..exponent {
            value *= 3.0 * barycentric[coordinate] - factor as f64;
        }
    }

    let mut gradient = [0.0; 3];
    for active in 0..4 {
        let exponent = exponents[active];
        if exponent == 0 {
            continue;
        }
        for factor in 0..exponent {
            for direction in 0..3 {
                let mut term = coefficient * 3.0 * gradients[active][direction];
                for coordinate in 0..4 {
                    for other_factor in 0..exponents[coordinate] {
                        if coordinate == active && other_factor == factor {
                            continue;
                        }
                        term *= 3.0 * barycentric[coordinate] - other_factor as f64;
                    }
                }
                gradient[direction] += term;
            }
        }
    }
    (value, gradient)
}
