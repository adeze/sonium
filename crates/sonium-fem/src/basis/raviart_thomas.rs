//! Raviart-Thomas (RT) basis functions for H(div)-conforming elements
//!
//! RT elements are face elements where DOFs represent flux through element faces.
//! They ensure normal continuity across element boundaries, which is required
//! for div-conforming spaces used in mixed formulations (Darcy, Stokes, E-div).
//!
//! ## RT0 (Lowest Order)
//!
//! **Triangle (2D):** 3 DOFs, one per edge
//! - Each basis function is constant on the edge in the outward normal direction
//! - Basis `N_i` has unit normal flux through face `i` and zero flux through
//!   every other face
//! - Formula: `N_i(x) = (x - v_i)/(d |K̂|)`, ordered by the local mesh-face
//!   convention (the face opposite `v_i`)

//! ## RT1 (Triangle)
//!
//! The first high-order slice implemented here is the complete triangular
//! `RT1` space.  It has eight local DOFs: two normal-flux moments on each of
//! the three edges, followed by two element-interior vector moments.  Edge
//! moment `0` is the constant moment and edge moment `1` is the moment against
//! the centered edge coordinate `t ∈ [-1, 1]`.  The two interior moments are
//! `∫_K v_x` and `∫_K v_y`.
//!
//! **Tetrahedron (3D):** 4 DOFs, one per face
//! - Each basis function is constant on the face in the outward normal direction
//! - Basis `N_i` has unit normal flux through face `i` and zero flux through
//!   every other face
//! - Formula: `N_i(x) = (x - v_i)/(d |K̂|)`, ordered by the local mesh-face
//!   convention (the face opposite `v_i`)

/// Face ordering for reference triangle: faces 0,1,2 are edges.
/// Edge k connects `TRIANGLE_EDGES[k]`.
pub const TRIANGLE_EDGES: [(usize, usize); 3] = [
    (0, 1), // edge 0: v0 -> v1
    (1, 2), // edge 1: v1 -> v2
    (2, 0), // edge 2: v2 -> v0
];

/// Face ordering for reference tetrahedron: 4 faces (triangular).
/// Face k has vertices `TET_FACES[k]`.
pub const TET_FACES: [[usize; 3]; 4] = [
    [0, 1, 2], // face 0: opposite vertex 3
    [0, 1, 3], // face 1: opposite vertex 2
    [0, 2, 3], // face 2: opposite vertex 1
    [1, 2, 3], // face 3: opposite vertex 0
];

/// Quadrilateral face (for hexahedra): 6 faces
pub const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3], // bottom
    [4, 5, 6, 7], // top
    [0, 1, 4, 5], // front
    [2, 3, 6, 7], // back
    [0, 2, 4, 6], // left
    [1, 3, 5, 7], // right
];

/// Evaluate lowest-order Raviart-Thomas (RT0) basis on a reference triangle.
///
/// Returns `(values, divs)` where:
/// - `values[edge][component]`: 2D vector value of basis function at `(xi, eta)`
/// - `divs[edge]`: divergence of each basis function (constant)
///
/// Reference triangle: `(0,0)`, `(1,0)`, `(0,1)`.
///
/// The RT0 basis functions are ordered like
/// [`crate::mesh::ElementType::local_edges`]: `(v0,v1)`, `(v1,v2)`,
/// `(v2,v0)`. Thus basis `i` is associated with the opposite vertex
/// `(v2,v0,v1)` respectively. They are:
/// ```text
/// N_i(x) = x - v_opposite(i)
/// ```
/// on the unit reference triangle. Each has one unit outward normal flux and
/// zero normal flux on the other two edges.
pub fn rt0_triangle(xi: f64, eta: f64) -> ([[f64; 2]; 3], [f64; 3]) {
    // The unit reference triangle has 2|K̂| = 1.  The local mesh edge order
    // is (v0,v1), (v1,v2), (v2,v0), whose opposite vertices are v2, v0, v1.
    ([[xi, eta - 1.0], [xi, eta], [xi - 1.0, eta]], [2.0; 3])
}

/// Evaluate the complete first-order Raviart--Thomas basis on a reference
/// triangle.
///
/// The reference triangle is `(0, 0)`, `(1, 0)`, `(0, 1)`, with the local
/// edge directions `(v0 -> v1)`, `(v1 -> v2)`, and `(v2 -> v0)`.  The returned
/// basis is ordered as
///
/// ```text
/// edge 0: moment 0, moment 1
/// edge 1: moment 0, moment 1
/// edge 2: moment 0, moment 1
/// cell:   x moment, y moment
/// ```
///
/// For an edge with parameter `t ∈ [-1, 1]`, the two edge functionals are
/// `∫_e (v · n) ds` and `∫_e t (v · n) ds`.  The cell functionals are the
/// unscaled moments `∫_K v_x dK` and `∫_K v_y dK`.  Consequently, the basis
/// is unisolvent and its divergence is affine, as required by `RT1`.
///
/// The coefficients below are the inverse of these eight reference
/// functionals applied to the polynomial spanning set
///
/// ```text
/// (1, 0), (0, 1), (x, 0), (0, x), (y, 0), (0, y),
/// (x², x y), (x y, y²).
/// ```
pub fn rt1_triangle(xi: f64, eta: f64) -> ([[f64; 2]; 8], [f64; 8]) {
    // Each row is one RT1 basis function; columns are the polynomial
    // spanning functions documented above.  Keeping the dual basis as a
    // small constant table makes tabulation allocation-free and keeps the
    // reference convention independently inspectable.
    const COEFFICIENTS: [[f64; 8]; 8] = [
        [0.0, -1.0, 1.0, 0.0, 0.0, 5.0, 0.0, -4.0],
        [0.0, 3.0, -5.0, -6.0, 0.0, -7.0, 8.0, 4.0],
        [0.0, 0.0, -3.0, 0.0, 0.0, -3.0, 4.0, 4.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -4.0, 4.0],
        [-1.0, 0.0, 5.0, 0.0, 0.0, 1.0, -4.0, 0.0],
        [-3.0, 0.0, 7.0, 0.0, 6.0, 5.0, -4.0, -8.0],
        [0.0, 0.0, 16.0, 0.0, 0.0, 8.0, -16.0, -8.0],
        [0.0, 0.0, 8.0, 0.0, 0.0, 16.0, -8.0, -16.0],
    ];

    let values = [
        [1.0, 0.0],
        [0.0, 1.0],
        [xi, 0.0],
        [0.0, xi],
        [eta, 0.0],
        [0.0, eta],
        [xi * xi, xi * eta],
        [xi * eta, eta * eta],
    ];
    let divergences = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0 * xi, 3.0 * eta];

    let mut basis = [[0.0; 2]; 8];
    let mut divs = [0.0; 8];
    for basis_index in 0..8 {
        for polynomial in 0..8 {
            basis[basis_index][0] += COEFFICIENTS[basis_index][polynomial] * values[polynomial][0];
            basis[basis_index][1] += COEFFICIENTS[basis_index][polynomial] * values[polynomial][1];
            divs[basis_index] += COEFFICIENTS[basis_index][polynomial] * divergences[polynomial];
        }
    }
    (basis, divs)
}

/// Evaluate the first-order Raviart--Thomas basis on the reference square
/// `[-1, 1]²`.
///
/// The basis follows [`crate::mesh::ElementType::local_edges`]: bottom,
/// right, top, left. Each edge has the constant and centered first normal
/// moments, followed by four cell moments: `v_x` against `1` and `eta`, and
/// `v_y` against `1` and `xi`. The reference vector space is
/// `Q₂,₁ × Q₁,₂`, so the returned divergence is bilinear.
pub fn rt1_quadrilateral(xi: f64, eta: f64) -> ([[f64; 2]; 12], [f64; 12]) {
    // The polynomial columns are
    //
    //   (1, eta, xi, xi*eta, xi², xi²*eta) for v_x,
    //   (1, eta, eta², xi, xi*eta, xi*eta²) for v_y.
    //
    // These rows are the inverse of the twelve reference edge/cell
    // functionals documented above. Keeping the dual table explicit makes
    // the reference convention auditable and the tabulation allocation-free.
    const COEFFICIENTS: [[f64; 12]; 12] = [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / 8.0,
            1.0 / 4.0,
            -3.0 / 8.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 / 8.0,
            3.0 / 4.0,
            -9.0 / 8.0,
        ],
        [
            -1.0 / 8.0,
            0.0,
            1.0 / 4.0,
            0.0,
            3.0 / 8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            -3.0 / 8.0,
            0.0,
            3.0 / 4.0,
            0.0,
            9.0 / 8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0 / 8.0,
            1.0 / 4.0,
            3.0 / 8.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 / 8.0,
            -3.0 / 4.0,
            -9.0 / 8.0,
        ],
        [
            1.0 / 8.0,
            0.0,
            1.0 / 4.0,
            0.0,
            -3.0 / 8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            -3.0 / 8.0,
            0.0,
            -3.0 / 4.0,
            0.0,
            9.0 / 8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            3.0 / 8.0,
            0.0,
            0.0,
            0.0,
            -3.0 / 8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            9.0 / 8.0,
            0.0,
            0.0,
            0.0,
            -9.0 / 8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 / 8.0,
            0.0,
            -3.0 / 8.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            9.0 / 8.0,
            0.0,
            -9.0 / 8.0,
        ],
    ];
    let values = [
        [1.0, 0.0],
        [eta, 0.0],
        [xi, 0.0],
        [xi * eta, 0.0],
        [xi * xi, 0.0],
        [xi * xi * eta, 0.0],
        [0.0, 1.0],
        [0.0, eta],
        [0.0, eta * eta],
        [0.0, xi],
        [0.0, xi * eta],
        [0.0, xi * eta * eta],
    ];
    let divergences = [
        0.0,
        0.0,
        1.0,
        eta,
        2.0 * xi,
        2.0 * xi * eta,
        0.0,
        1.0,
        2.0 * eta,
        0.0,
        xi,
        2.0 * xi * eta,
    ];

    let mut basis = [[0.0; 2]; 12];
    let mut divs = [0.0; 12];
    for basis_index in 0..12 {
        for polynomial in 0..12 {
            basis[basis_index][0] += COEFFICIENTS[basis_index][polynomial] * values[polynomial][0];
            basis[basis_index][1] += COEFFICIENTS[basis_index][polynomial] * values[polynomial][1];
            divs[basis_index] += COEFFICIENTS[basis_index][polynomial] * divergences[polynomial];
        }
    }
    (basis, divs)
}

/// Evaluate lowest-order Raviart-Thomas (RT0) basis on the reference square
/// `[-1, 1]²`.
///
/// The basis follows [`crate::mesh::ElementType::local_edges`]: bottom,
/// right, top, left. Each vector has unit outward normal flux through its
/// associated edge, zero flux through the other edges, and divergence `1/4`.
pub fn rt0_quadrilateral(xi: f64, eta: f64) -> ([[f64; 2]; 4], [f64; 4]) {
    (
        [
            [0.0, (eta - 1.0) / 4.0],
            [(xi + 1.0) / 4.0, 0.0],
            [0.0, (eta + 1.0) / 4.0],
            [(xi - 1.0) / 4.0, 0.0],
        ],
        [0.25; 4],
    )
}

/// Evaluate RT0 basis on a reference tetrahedron.
///
/// Returns `(values, divs)` where:
/// - `values[face][component]`: 3D vector value at `(xi, eta, zeta)`
/// - `divs[face]`: divergence (constant)
///
/// Reference tetrahedron: `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)`.
///
/// The basis is ordered like
/// [`crate::mesh::ElementType::local_faces`]: faces opposite v3, v2, v1, v0.
/// On the unit reference tetrahedron, `3|K̂| = 1/2`, so the corresponding
/// RT0 functions are `2 (x - v_opposite)` and each has divergence six.
pub fn rt0_tetrahedron(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 4], [f64; 4]) {
    (
        [
            [2.0 * xi, 2.0 * eta, 2.0 * (zeta - 1.0)],
            [2.0 * xi, 2.0 * (eta - 1.0), 2.0 * zeta],
            [2.0 * (xi - 1.0), 2.0 * eta, 2.0 * zeta],
            [2.0 * xi, 2.0 * eta, 2.0 * zeta],
        ],
        [6.0; 4],
    )
}

/// Evaluate first-order Raviart--Thomas basis on the reference tetrahedron.
///
/// The fifteen functions use three moments on each triangular face (`1` and
/// two affine face coordinates), followed by three Cartesian cell moments.
/// The coefficient table is the exact dual of those reference functionals;
/// the returned divergence is affine.
pub fn rt1_tetrahedron(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 15], [f64; 15]) {
    const COEFFICIENTS: [[f64; 15]; 15] = [
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -18.0, 24.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            24.0, -36.0, -30.0, 24.0, -36.0, -30.0, 48.0, -54.0, -54.0, -12.0, 6.0, 6.0, 60.0,
            30.0, 30.0,
        ],
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, -48.0, -24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, -24.0, -48.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 0.0, -18.0, 24.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 0.0, 24.0, -48.0, -24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            24.0, -30.0, -36.0, 48.0, -54.0, -54.0, 24.0, -36.0, -30.0, -6.0, -6.0, 0.0, 30.0,
            60.0, 30.0,
        ],
        [
            0.0, 0.0, 0.0, 24.0, -24.0, -48.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            -18.0, 24.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            24.0, -48.0, -24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            24.0, -24.0, -48.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            48.0, -54.0, -54.0, 24.0, -30.0, -36.0, 24.0, -30.0, -36.0, -6.0, 0.0, -6.0, 30.0,
            30.0, 60.0,
        ],
        [
            -30.0, 60.0, 30.0, -30.0, 60.0, 30.0, -30.0, 30.0, 30.0, 30.0, -30.0, -30.0, -60.0,
            -30.0, -30.0,
        ],
        [
            -30.0, 30.0, 60.0, -30.0, 30.0, 30.0, -30.0, 60.0, 30.0, 0.0, 30.0, 0.0, -30.0, -60.0,
            -30.0,
        ],
        [
            -30.0, 30.0, 30.0, -30.0, 30.0, 60.0, -30.0, 30.0, 60.0, 0.0, 0.0, 30.0, -30.0, -30.0,
            -60.0,
        ],
    ];
    let raw = [
        [1.0, 0.0, 0.0],
        [xi, 0.0, 0.0],
        [eta, 0.0, 0.0],
        [zeta, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, xi, 0.0],
        [0.0, eta, 0.0],
        [0.0, zeta, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, xi],
        [0.0, 0.0, eta],
        [0.0, 0.0, zeta],
        [xi * xi, xi * eta, xi * zeta],
        [xi * eta, eta * eta, eta * zeta],
        [xi * zeta, eta * zeta, zeta * zeta],
    ];
    let raw_divergence = [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        4.0 * xi,
        4.0 * eta,
        4.0 * zeta,
    ];
    let mut values = [[0.0; 3]; 15];
    let mut divs = [0.0; 15];
    for basis in 0..15 {
        for raw_index in 0..15 {
            let coefficient = COEFFICIENTS[raw_index][basis];
            for component in 0..3 {
                values[basis][component] += coefficient * raw[raw_index][component];
            }
            divs[basis] += coefficient * raw_divergence[raw_index];
        }
    }
    (values, divs)
}

/// Evaluate lowest-order Raviart--Thomas (RT0) basis on the reference cube
/// `[-1, 1]³`.
///
/// The basis follows [`crate::mesh::ElementType::local_faces`]: bottom, top,
/// front, back, left, right. Each vector has unit outward normal flux through
/// its associated face, zero flux through the other faces, and divergence
/// `1/8`.
pub fn rt0_hexahedron(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 6], [f64; 6]) {
    (
        [
            [0.0, 0.0, (zeta - 1.0) / 8.0],
            [0.0, 0.0, (zeta + 1.0) / 8.0],
            [0.0, (eta - 1.0) / 8.0, 0.0],
            [0.0, (eta + 1.0) / 8.0, 0.0],
            [(xi - 1.0) / 8.0, 0.0, 0.0],
            [(xi + 1.0) / 8.0, 0.0, 0.0],
        ],
        [0.125; 6],
    )
}

/// Compute outward normal for a triangular face in 3D.
///
/// Given 3 vertices of a triangle, returns the unit outward normal.
/// The normal is computed as normalized cross product of edge vectors.
/// Evaluate the affine first-order Raviart--Thomas basis on the reference
/// cube `[-1, 1]³`.
///
/// The tensor-product space is `Q₂,₁,₁ × Q₁,₂,₁ × Q₁,₁,₂` and has 36
/// functions: four moments on each quadrilateral face followed by four cell
/// moments for each Cartesian component. Face modes are ordered as `1`,
/// `t₀`, `t₁`, and `t₀t₁`, where the moment tests use the corresponding
/// unscaled monomials. The returned functions are dual to those moments and
/// have unit outward normal flux.
pub fn rt1_hexahedron(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 36], [f64; 36]) {
    rt1_hexahedron_oriented(xi, eta, zeta, [[1, 0, 0, 1]; 6])
}

/// Evaluate RT1 on a reference cube using per-face canonical tangential
/// coordinates.
///
/// Each face transform is `[u_t0, u_t1, v_t0, v_t1]`, so that the canonical
/// face coordinates are `u = u_t0*t₀ + u_t1*t₁` and
/// `v = v_t0*t₀ + v_t1*t₁`. The transforms are signed axis permutations;
/// they keep the four face moments coherent when a shared quadrilateral is
/// seen with a different local rotation or reversal.
pub fn rt1_hexahedron_oriented(
    xi: f64,
    eta: f64,
    zeta: f64,
    face_axes: [[i8; 4]; 6],
) -> ([[f64; 3]; 36], [f64; 36]) {
    let q0 = |_: f64| 0.5;
    let q1 = |value: f64| 1.5 * value;
    let face_mode = |u: f64, v: f64, mode: usize| {
        let first = if matches!(mode, 1 | 3) { q1(u) } else { q0(u) };
        let second = if matches!(mode, 2 | 3) { q1(v) } else { q0(v) };
        first * second
    };

    let mut values = [[0.0; 3]; 36];
    let mut divs = [0.0; 36];
    let bubble = |coordinate: f64| 0.75 * (1.0 - coordinate * coordinate);
    let bubble_derivative = |coordinate: f64| -1.5 * coordinate;

    // local face order: z-, z+, y-, y+, x-, x+.
    for (face, &axes) in face_axes.iter().enumerate() {
        let (component, normal_coordinate, outward_sign, t0, t1) = match face {
            0 => (2, zeta, -1.0, xi, eta),
            1 => (2, zeta, 1.0, xi, eta),
            2 => (1, eta, -1.0, xi, zeta),
            3 => (1, eta, 1.0, xi, zeta),
            4 => (0, xi, -1.0, eta, zeta),
            5 => (0, xi, 1.0, eta, zeta),
            _ => unreachable!(),
        };
        let u = f64::from(axes[0]) * t0 + f64::from(axes[1]) * t1;
        let v = f64::from(axes[2]) * t0 + f64::from(axes[3]) * t1;
        let endpoint = if face % 2 == 0 {
            0.5 * (1.0 - normal_coordinate)
        } else {
            0.5 * (1.0 + normal_coordinate)
        };
        for mode in 0..4 {
            let index = 4 * face + mode;
            let tangential = face_mode(u, v, mode);
            // Subtract the normal bubble so face modes have zero cell
            // moments while retaining their prescribed boundary traces.
            values[index][component] =
                outward_sign * (endpoint - bubble(normal_coordinate)) * tangential;
            divs[index] = (0.5 - outward_sign * bubble_derivative(normal_coordinate)) * tangential;
        }
    }

    // Cell moments use the normalized quadratic bubble in the normal
    // coordinate and the one-dimensional dual polynomials in the other two.
    for mode in 0..4 {
        let q_eta_x = if matches!(mode, 1 | 3) {
            q1(eta)
        } else {
            q0(eta)
        };
        let q_zeta = if matches!(mode, 2 | 3) {
            q1(zeta)
        } else {
            q0(zeta)
        };
        let x_index = 24 + mode;
        values[x_index][0] = bubble(xi) * q_eta_x * q_zeta;
        divs[x_index] = bubble_derivative(xi) * q_eta_x * q_zeta;

        let q_xi = if matches!(mode, 1 | 3) {
            q1(xi)
        } else {
            q0(xi)
        };
        let y_index = 28 + mode;
        values[y_index][1] = bubble(eta) * q_xi * q_zeta;
        divs[y_index] = bubble_derivative(eta) * q_xi * q_zeta;

        let q_eta_z = if matches!(mode, 2 | 3) {
            q1(eta)
        } else {
            q0(eta)
        };
        let z_index = 32 + mode;
        values[z_index][2] = bubble(zeta) * q_xi * q_eta_z;
        divs[z_index] = bubble_derivative(zeta) * q_xi * q_eta_z;
    }

    (values, divs)
}

pub fn triangle_outward_normal_3d(
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    _element_center: [f64; 3],
) -> [f64; 3] {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let mut n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];

    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if norm > 1e-12 {
        n[0] /= norm;
        n[1] /= norm;
        n[2] /= norm;
    }

    n
}

/// Compute outward normal for a quad face in 3D (bilinear mapping, approximated as linear).
pub fn quad_outward_normal_3d(
    v0: [f64; 3],
    v1: [f64; 3],
    _v2: [f64; 3],
    v3: [f64; 3],
    _element_center: [f64; 3],
) -> [f64; 3] {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]];

    let mut n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];

    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if norm > 1e-12 {
        n[0] /= norm;
        n[1] /= norm;
        n[2] /= norm;
    }

    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rt0_triangle_basis() {
        let (values, divs) = rt0_triangle(0.5, 0.25);

        assert_eq!(values.len(), 3);
        assert_eq!(divs.len(), 3);

        // Divergence should be constant = 2
        for div in &divs[..3] {
            assert!((div - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rt0_tetrahedron_basis() {
        let (values, divs) = rt0_tetrahedron(0.25, 0.25, 0.25);

        assert_eq!(values.len(), 4);
        assert_eq!(divs.len(), 4);

        // Divergence should be constant = 6 on the unit tetrahedron.
        for div in &divs[..4] {
            assert!((div - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    fn rt0_quadrilateral_face_fluxes_are_unisolvent_and_conservative() {
        let faces = [
            ([0.0, -1.0], [0.0, -1.0]),
            ([1.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([-1.0, 0.0], [-1.0, 0.0]),
        ];
        for (face, (point, normal)) in faces.iter().enumerate() {
            let (values, divs) = rt0_quadrilateral(point[0], point[1]);
            for (basis, value) in values.iter().enumerate() {
                let flux = (value[0] * normal[0] + value[1] * normal[1]) * 2.0;
                let expected = if basis == face { 1.0 } else { 0.0 };
                assert!(
                    (flux - expected).abs() < 1e-12,
                    "face={face}, basis={basis}"
                );
            }
            assert!(divs.iter().all(|div| (*div * 4.0 - 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn rt0_hexahedron_face_fluxes_are_unisolvent_and_conservative() {
        let faces = [
            ([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]),
            ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            ([0.0, -1.0, 0.0], [0.0, -1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            ([-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        for (face, (point, normal)) in faces.iter().enumerate() {
            let (values, divs) = rt0_hexahedron(point[0], point[1], point[2]);
            for (basis, value) in values.iter().enumerate() {
                let flux =
                    (value[0] * normal[0] + value[1] * normal[1] + value[2] * normal[2]) * 4.0;
                let expected = if basis == face { 1.0 } else { 0.0 };
                assert!(
                    (flux - expected).abs() < 1e-12,
                    "face={face}, basis={basis}"
                );
            }
            assert!(divs.iter().all(|div| (*div * 8.0 - 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn rt0_triangle_face_fluxes_are_unisolvent_and_conservative() {
        // Midpoint evaluation is exact because RT0 normal traces are constant
        // on every affine reference edge.
        let faces = [
            ([0.5, 0.0], [0.0, -1.0], 1.0),
            (
                [0.5, 0.5],
                [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()],
                2.0_f64.sqrt(),
            ),
            ([0.0, 0.5], [-1.0, 0.0], 1.0),
        ];
        for (face, (point, normal, measure)) in faces.iter().enumerate() {
            let (values, divs) = rt0_triangle(point[0], point[1]);
            for (basis, value) in values.iter().enumerate() {
                let flux = (value[0] * normal[0] + value[1] * normal[1]) * measure;
                let expected = if basis == face { 1.0 } else { 0.0 };
                assert!(
                    (flux - expected).abs() < 1e-12,
                    "face={face}, basis={basis}"
                );
            }
            // ∫_K div(N_i) = 2 × 1/2 = 1 = Σ_faces flux(N_i).
            assert!(divs.iter().all(|div| (*div * 0.5 - 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn rt0_tetrahedron_face_fluxes_are_unisolvent_and_conservative() {
        // Again, RT0 normal traces are constant on affine faces. The face
        // ordering matches ElementType::local_faces for tetrahedra.
        let faces = [
            ([1.0 / 3.0, 1.0 / 3.0, 0.0], [0.0, 0.0, -1.0], 0.5),
            ([1.0 / 3.0, 0.0, 1.0 / 3.0], [0.0, -1.0, 0.0], 0.5),
            ([0.0, 1.0 / 3.0, 1.0 / 3.0], [-1.0, 0.0, 0.0], 0.5),
            (
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [
                    1.0 / 3.0_f64.sqrt(),
                    1.0 / 3.0_f64.sqrt(),
                    1.0 / 3.0_f64.sqrt(),
                ],
                3.0_f64.sqrt() / 2.0,
            ),
        ];
        for (face, (point, normal, measure)) in faces.iter().enumerate() {
            let (values, divs) = rt0_tetrahedron(point[0], point[1], point[2]);
            for (basis, value) in values.iter().enumerate() {
                let flux =
                    (value[0] * normal[0] + value[1] * normal[1] + value[2] * normal[2]) * measure;
                let expected = if basis == face { 1.0 } else { 0.0 };
                assert!(
                    (flux - expected).abs() < 1e-12,
                    "face={face}, basis={basis}"
                );
            }
            // ∫_K div(N_i) = 6 × 1/6 = 1 = Σ_faces flux(N_i).
            assert!(divs.iter().all(|div| (*div / 6.0 - 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn test_triangle_normal() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let center = [0.0, 0.0, 0.0];

        let n = triangle_outward_normal_3d(v0, v1, v2, center);

        // Should point in +z direction (up)
        assert!(n[2] > 0.9);
    }
}
