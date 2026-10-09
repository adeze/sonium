//! Independent physical-coordinate reference for curved-Q2 RT1 mixed assembly.
//!
//! The existing RT1 mixed-conservation packet uses affine cells, while the
//! existing curved RT1 packet checks tabulation and a finite-difference
//! divergence.  This packet checks their combined contract: every entry of the
//! 3-D curved-Q2 RT1/P1 mixed divergence block is compared with an independently
//! written Q2 map, RT1 reference basis, and tensor-product Gauss rule.

use sonium_fem::assembly::assemble_mixed_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{HdivDivergenceForm, MassForm};
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, unit_cube_hexahedra};
use sonium_fem::spaces::{FunctionSpace, H1Space, HdivOrder, HdivSpace};

const QUADRATURE: [(f64, f64); 5] = [
    (-0.906_179_845_938_664, 0.236_926_885_056_189_08),
    (-0.538_469_310_105_683_1, 0.478_628_670_499_366_47),
    (0.0, 0.568_888_888_888_888_9),
    (0.538_469_310_105_683_1, 0.478_628_670_499_366_47),
    (0.906_179_845_938_664, 0.236_926_885_056_189_08),
];

/// Error and geometry metrics exported to the release-facing QA binary.
#[derive(Debug, Clone, Copy)]
pub struct CurvedRt1MixedMetrics {
    pub max_block_error: f64,
    pub max_constant_conservation_error: f64,
    pub min_jacobian_det: f64,
    pub checked_entries: usize,
}

fn curved_q2_hexahedron() -> Mesh {
    let mut mesh = unit_cube_hexahedra(1).to_p2();
    let element_nodes = mesh.elements[0].nodes.clone();

    // Move edge, face, and interior nodes.  The perturbations are small enough
    // to keep the map orientation positive, but make the Jacobian genuinely
    // variable in all three physical directions.
    for &(local, dx, dy, dz) in &[
        (8, 0.031, 0.009, -0.006),
        (9, -0.012, 0.024, 0.008),
        (14, 0.010, -0.015, 0.021),
        (19, -0.017, 0.013, 0.011),
        (20, 0.014, -0.011, 0.018),
        (26, 0.022, -0.019, 0.016),
    ] {
        let point = &mut mesh.nodes[element_nodes[local]];
        point.x += dx;
        point.y += dy;
        point.z += dz;
    }
    mesh
}

fn q2_1d(x: f64) -> ([f64; 3], [f64; 3]) {
    (
        [0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)],
        [x - 0.5, -2.0 * x, x + 0.5],
    )
}

fn q2_map(mesh: &Mesh, xi: f64, eta: f64, zeta: f64) -> ([f64; 3], [[f64; 3]; 3]) {
    let (lx, dx) = q2_1d(xi);
    let (ly, dy) = q2_1d(eta);
    let (lz, dz) = q2_1d(zeta);
    let indices = [
        (0, 0, 0),
        (2, 0, 0),
        (2, 2, 0),
        (0, 2, 0),
        (0, 0, 2),
        (2, 0, 2),
        (2, 2, 2),
        (0, 2, 2),
        (1, 0, 0),
        (2, 1, 0),
        (1, 2, 0),
        (0, 1, 0),
        (1, 0, 2),
        (2, 1, 2),
        (1, 2, 2),
        (0, 1, 2),
        (0, 0, 1),
        (2, 0, 1),
        (2, 2, 1),
        (0, 2, 1),
        (1, 1, 0),
        (1, 1, 2),
        (1, 0, 1),
        (1, 2, 1),
        (0, 1, 1),
        (2, 1, 1),
        (1, 1, 1),
    ];
    let element = &mesh.elements[0];
    let mut point = [0.0; 3];
    let mut jacobian = [[0.0; 3]; 3];

    for (local, &(i, j, k)) in indices.iter().enumerate() {
        let value = lx[i] * ly[j] * lz[k];
        let gradient = [
            dx[i] * ly[j] * lz[k],
            lx[i] * dy[j] * lz[k],
            lx[i] * ly[j] * dz[k],
        ];
        let node = &mesh.nodes[element.nodes[local]];
        let coordinates = [node.x, node.y, node.z];
        for component in 0..3 {
            point[component] += value * coordinates[component];
            for reference_component in 0..3 {
                jacobian[component][reference_component] +=
                    gradient[reference_component] * coordinates[component];
            }
        }
    }
    (point, jacobian)
}

fn determinant(a: [[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn q1_hex(xi: f64, eta: f64, zeta: f64) -> [f64; 8] {
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

fn q0(_: f64) -> f64 {
    0.5
}

fn q1(x: f64) -> f64 {
    1.5 * x
}

fn face_mode(u: f64, v: f64, mode: usize) -> f64 {
    let first = if matches!(mode, 1 | 3) { q1(u) } else { q0(u) };
    let second = if matches!(mode, 2 | 3) { q1(v) } else { q0(v) };
    first * second
}

// Reconstruct the tangential signed axis maps from public mesh topology.  RT1
// face modes are moments in the canonical global-face coordinates, so a local
// face can require a signed permutation even when its normal sign is positive.
fn hexa_face_axes(mesh: &Mesh, topology: &MeshTopology) -> [[i8; 4]; 6] {
    let vertices = mesh.elements[0].vertices();
    let local_faces = ElementType::Hexahedron.local_faces();
    let reference_vertices = [
        [-1_i8, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ];
    let canonical_face_coordinates = [[-1_i8, -1_i8], [1, -1], [1, 1], [-1, 1]];
    let mut result = [[1_i8, 0, 0, 1]; 6];

    for (local_face, &face_vertices) in local_faces.iter().enumerate() {
        let global_face = topology.element_face_indices(0)[local_face];
        let canonical_nodes = topology.face(global_face).nodes();
        let mut sums = [0_i16; 4];
        for &local_vertex in face_vertices {
            let node = vertices[local_vertex];
            let canonical_corner = canonical_nodes
                .iter()
                .position(|&candidate| candidate == node)
                .expect("each hexahedron face vertex must be in its topology face");
            let [u, v] = canonical_face_coordinates[canonical_corner];
            let reference = reference_vertices[local_vertex];
            let [t0, t1] = match local_face {
                0 | 1 => [reference[0], reference[1]],
                2 | 3 => [reference[0], reference[2]],
                4 | 5 => [reference[1], reference[2]],
                _ => unreachable!(),
            };
            sums[0] += i16::from(u) * i16::from(t0);
            sums[1] += i16::from(u) * i16::from(t1);
            sums[2] += i16::from(v) * i16::from(t0);
            sums[3] += i16::from(v) * i16::from(t1);
        }
        result[local_face] = sums.map(|sum| (sum / 4) as i8);
    }
    result
}

/// Explicit canonical RT1 basis on `[-1,1]^3`, including reference divergence.
fn reference_rt1_hex(
    xi: f64,
    eta: f64,
    zeta: f64,
    face_axes: [[i8; 4]; 6],
) -> ([[f64; 3]; 36], [f64; 36]) {
    let mut values = [[0.0; 3]; 36];
    let mut divergences = [0.0; 36];
    let faces = [
        (2usize, zeta, -1.0, xi, eta),
        (2, zeta, 1.0, xi, eta),
        (1, eta, -1.0, xi, zeta),
        (1, eta, 1.0, xi, zeta),
        (0, xi, -1.0, eta, zeta),
        (0, xi, 1.0, eta, zeta),
    ];

    for (face, &(component, normal, outward, tangent0, tangent1)) in faces.iter().enumerate() {
        let endpoint = 0.5 * (1.0 + outward * normal);
        let bubble = 0.75 * (1.0 - normal * normal);
        let bubble_derivative = -1.5 * normal;
        let axes = face_axes[face];
        let u = f64::from(axes[0]) * tangent0 + f64::from(axes[1]) * tangent1;
        let v = f64::from(axes[2]) * tangent0 + f64::from(axes[3]) * tangent1;
        for mode in 0..4 {
            let local = 4 * face + mode;
            let tangential = face_mode(u, v, mode);
            values[local][component] = outward * (endpoint - bubble) * tangential;
            divergences[local] = (0.5 - outward * bubble_derivative) * tangential;
        }
    }

    let bubble = |x: f64| 0.75 * (1.0 - x * x);
    let bubble_derivative = |x: f64| -1.5 * x;
    for mode in 0..4 {
        let eta_factor = if matches!(mode, 1 | 3) {
            q1(eta)
        } else {
            q0(eta)
        };
        let zeta_factor = if matches!(mode, 2 | 3) {
            q1(zeta)
        } else {
            q0(zeta)
        };
        values[24 + mode][0] = bubble(xi) * eta_factor * zeta_factor;
        divergences[24 + mode] = bubble_derivative(xi) * eta_factor * zeta_factor;

        let xi_factor = if matches!(mode, 1 | 3) {
            q1(xi)
        } else {
            q0(xi)
        };
        values[28 + mode][1] = bubble(eta) * xi_factor * zeta_factor;
        divergences[28 + mode] = bubble_derivative(eta) * xi_factor * zeta_factor;

        let eta_factor = if matches!(mode, 2 | 3) {
            q1(eta)
        } else {
            q0(eta)
        };
        values[32 + mode][2] = bubble(zeta) * xi_factor * eta_factor;
        divergences[32 + mode] = bubble_derivative(zeta) * xi_factor * eta_factor;
    }
    (values, divergences)
}

fn independent_reference_block(
    mesh: &Mesh,
    hdiv_dofs: &[usize],
    scalar_dofs: &[usize],
    face_signs: &[i8],
    face_axes: [[i8; 4]; 6],
    n_hdiv: usize,
    n_scalar: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut block = vec![0.0; n_hdiv * n_scalar];
    let mut constant = vec![0.0; n_hdiv];
    let mut min_det = f64::INFINITY;

    for &(xi, wx) in &QUADRATURE {
        for &(eta, wy) in &QUADRATURE {
            for &(zeta, wz) in &QUADRATURE {
                let (physical_point, jacobian) = q2_map(mesh, xi, eta, zeta);
                let det = determinant(jacobian);
                assert!(
                    physical_point.iter().all(|value| value.is_finite()),
                    "curved Q2 map produced a non-finite physical point"
                );
                assert!(det > 0.0, "curved Q2 map changed orientation: det={det}");
                min_det = min_det.min(det);

                let (_, reference_divergences) = reference_rt1_hex(xi, eta, zeta, face_axes);
                let scalar_values = q1_hex(xi, eta, zeta);
                let weight = wx * wy * wz;
                for local in 0..36 {
                    let face_sign = if local < 24 {
                        f64::from(face_signs[local / 4])
                    } else {
                        1.0
                    };
                    // The physical divergence is (signed div_ref)/det.  Keep
                    // both factors explicit here: their cancellation is the
                    // curved-coordinate conservation identity under test.
                    let physical_divergence = face_sign * reference_divergences[local] / det;
                    let row = hdiv_dofs[local];
                    constant[row] += -physical_divergence * weight * det;
                    for scalar in 0..8 {
                        let col = scalar_dofs[scalar];
                        block[row * n_scalar + col] +=
                            -physical_divergence * scalar_values[scalar] * weight * det;
                    }
                }
            }
        }
    }
    (block, constant, min_det)
}

pub fn run_curved_rt1_mixed_physical_reference() -> CurvedRt1MixedMetrics {
    let mesh = curved_q2_hexahedron();
    let hdiv = HdivSpace::with_order(HdivOrder::Rt1);
    let scalar = H1Space::new(PolynomialDegree::P1);
    hdiv.validate(&mesh)
        .expect("curved RT1 H(div) mesh must validate");
    scalar
        .validate(&mesh)
        .expect("curved P1 scalar mesh must validate");

    let hdiv_dofs = hdiv.local_to_global(&mesh, 0);
    let scalar_dofs = scalar.local_to_global(&mesh, 0);
    assert_eq!(hdiv_dofs.len(), 36);
    assert_eq!(scalar_dofs.len(), 8);
    let n_hdiv = hdiv.num_global_dofs(&mesh);
    let n_scalar = scalar.num_global_dofs(&mesh);
    assert_eq!(n_hdiv, 36);
    assert_eq!(n_scalar, 8);

    let topology = MeshTopology::build(&mesh);
    let face_signs = topology.element_face_signs(0);
    assert_eq!(face_signs.len(), 6);
    let face_axes = hexa_face_axes(&mesh, &topology);

    let mixed = assemble_mixed_checked(&MassForm, &HdivDivergenceForm, &hdiv, &scalar, &mesh, 8)
        .expect("checked curved RT1 mixed assembly must succeed");
    assert_eq!(mixed.block_01.num_rows, n_hdiv);
    assert_eq!(mixed.block_01.num_cols, n_scalar);

    let (reference, reference_constant, min_det) = independent_reference_block(
        &mesh,
        &hdiv_dofs,
        &scalar_dofs,
        &face_signs,
        face_axes,
        n_hdiv,
        n_scalar,
    );
    let mut actual = vec![0.0; n_hdiv * n_scalar];
    for ((&row, &col), &value) in mixed
        .block_01
        .rows
        .iter()
        .zip(&mixed.block_01.cols)
        .zip(&mixed.block_01.values)
    {
        actual[row * n_scalar + col] += value;
    }

    let max_block_error = actual
        .iter()
        .zip(&reference)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0, f64::max);
    let mut actual_constant = vec![0.0; n_hdiv];
    for row in 0..n_hdiv {
        actual_constant[row] = actual[row * n_scalar..(row + 1) * n_scalar].iter().sum();
    }
    let max_constant_conservation_error = actual_constant
        .iter()
        .zip(reference_constant)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0, f64::max);

    assert!(
        max_block_error < 2.0e-10,
        "curved RT1/P1 mixed block error: {max_block_error:.3e}"
    );
    assert!(
        max_constant_conservation_error < 2.0e-10,
        "curved RT1 constant-field conservation error: {max_constant_conservation_error:.3e}"
    );
    assert!(
        min_det > 0.05,
        "curved Q2 Jacobian became too small: {min_det:.3e}"
    );

    CurvedRt1MixedMetrics {
        max_block_error,
        max_constant_conservation_error,
        min_jacobian_det: min_det,
        checked_entries: n_hdiv * n_scalar,
    }
}

#[test]
fn curved_rt1_mixed_block_matches_independent_physical_reference() {
    let metrics = run_curved_rt1_mixed_physical_reference();
    assert_eq!(metrics.checked_entries, 288);
    assert!(metrics.max_block_error < 2.0e-10);
    assert!(metrics.max_constant_conservation_error < 2.0e-10);
}
