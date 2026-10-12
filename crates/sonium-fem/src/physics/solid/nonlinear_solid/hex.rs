//! Eight-node hexahedron machinery for locking-free bricks.
//!
//! Isoparametric H8 shape functions, 2x2x2 Gauss quadrature, Jacobian maps,
//! and small mesh builders shared by the F-bar and EAS formulations. Node
//! ordering is the standard `xi`-fastest cube below.

// Rust guideline compliant 2026-02-21

/// Natural coordinates of the 8 hex nodes.
pub const HEX_NODES: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// H8 shape functions at a natural point.
#[must_use]
pub fn hex_shape(natural: &[f64; 3]) -> [f64; 8] {
    let mut n = [0.0; 8];
    for (a, node) in HEX_NODES.iter().enumerate() {
        n[a] = (1.0 + node[0] * natural[0])
            * (1.0 + node[1] * natural[1])
            * (1.0 + node[2] * natural[2])
            / 8.0;
    }
    n
}

/// H8 shape derivatives `dN_a / dxi_j` at a natural point.
#[must_use]
pub fn hex_shape_derivatives(natural: &[f64; 3]) -> [[f64; 3]; 8] {
    let mut d = [[0.0; 3]; 8];
    for (a, node) in HEX_NODES.iter().enumerate() {
        let (sx, sy, sz) = (node[0], node[1], node[2]);
        let (xi, eta, zeta) = (natural[0], natural[1], natural[2]);
        d[a][0] = sx * (1.0 + sy * eta) * (1.0 + sz * zeta) / 8.0;
        d[a][1] = sy * (1.0 + sx * xi) * (1.0 + sz * zeta) / 8.0;
        d[a][2] = sz * (1.0 + sx * xi) * (1.0 + sy * eta) / 8.0;
    }
    d
}

/// Two-point Gauss rule on `[-1, 1]`.
#[must_use]
pub fn gauss_2() -> ([f64; 2], [f64; 2]) {
    let point = 1.0 / 3.0_f64.sqrt();
    ([-point, point], [1.0, 1.0])
}

/// Eight Gauss points and weights for the reference cube.
#[must_use]
pub fn hex_quadrature() -> [([f64; 3], f64); 8] {
    let (points, weights) = gauss_2();
    let mut out = [([0.0; 3], 0.0); 8];
    let mut qp = 0;
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                out[qp] = (
                    [points[i], points[j], points[k]],
                    weights[i] * weights[j] * weights[k],
                );
                qp += 1;
            }
        }
    }
    out
}

/// Jacobian, determinant, and inverse at shape derivatives.
#[must_use]
pub fn hex_jacobian(
    coords: &[[f64; 3]; 8],
    derivatives: &[[f64; 3]; 8],
) -> ([f64; 9], f64, [f64; 9]) {
    let mut j = [0.0; 9];
    for a in 0..8 {
        for i in 0..3 {
            for col in 0..3 {
                j[i * 3 + col] += coords[a][i] * derivatives[a][col];
            }
        }
    }
    let det = j[0] * (j[4] * j[8] - j[5] * j[7]) - j[1] * (j[3] * j[8] - j[5] * j[6])
        + j[2] * (j[3] * j[7] - j[4] * j[6]);
    let inv_det = 1.0 / det;
    let inv = [
        (j[4] * j[8] - j[5] * j[7]) * inv_det,
        (j[2] * j[7] - j[1] * j[8]) * inv_det,
        (j[1] * j[5] - j[2] * j[4]) * inv_det,
        (j[5] * j[6] - j[3] * j[8]) * inv_det,
        (j[0] * j[8] - j[2] * j[6]) * inv_det,
        (j[2] * j[3] - j[0] * j[5]) * inv_det,
        (j[3] * j[7] - j[4] * j[6]) * inv_det,
        (j[1] * j[6] - j[0] * j[7]) * inv_det,
        (j[0] * j[4] - j[1] * j[3]) * inv_det,
    ];
    (j, det, inv)
}

/// Physical gradients `dN_a / dX` from natural derivatives.
///
/// Uses the inverse transpose: `dN/dX = J^{-T} dN/dxi` with the Jacobian
/// holding physical rows and natural columns.
#[must_use]
pub fn physical_gradients(jacobian_inv: &[f64; 9], derivatives: &[[f64; 3]; 8]) -> [[f64; 3]; 8] {
    let mut grad = [[0.0; 3]; 8];
    for a in 0..8 {
        for i in 0..3 {
            grad[a][i] = jacobian_inv[i] * derivatives[a][0]
                + jacobian_inv[3 + i] * derivatives[a][1]
                + jacobian_inv[6 + i] * derivatives[a][2];
        }
    }
    grad
}

/// Determinant of a row-major 3x3 matrix.
#[must_use]
pub fn det3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Solve a dense `n x n` system in place with partial pivoting.
///
/// # Panics
///
/// Panics for a numerically singular pivot.
pub fn dense_solve(matrix: &mut [f64], rhs: &mut [f64], n: usize) {
    for col in 0..n {
        let mut pivot = col;
        let mut best = matrix[col * n + col].abs();
        for row in col + 1..n {
            let candidate = matrix[row * n + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "dense solve hit a singular pivot");
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diag = matrix[col * n + col];
        for row in col + 1..n {
            let factor = matrix[row * n + col] / diag;
            matrix[row * n + col] = 0.0;
            for k in col + 1..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in row + 1..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
}

/// Standard Q1 internal forces for one hex element.
///
/// Evaluates the base hyperelastic response without locking cures, serving
/// as the locked reference for F-bar and EAS comparisons. Nodal
/// displacements use interleaved `[ux, uy, uz]` order.
#[must_use]
pub fn q1_element_residual(
    material: &impl crate::physics::solid::nonlinear_solid::HyperelasticMaterial,
    mesh: &HexMesh,
    elem_idx: usize,
    u: &[f64],
) -> [f64; 24] {
    // The 3x3 deformation gradient below indexes nine stress entries; a
    // 2D material would previously panic on the short heap vector, so
    // fail closed with the same strictness on the stack spelling.
    assert_eq!(
        material.dim(),
        3,
        "q1_element_residual needs a 3D material"
    );
    let coords = mesh.element_coords(elem_idx);
    let mut residual = [0.0; 24];
    for (natural, weight) in hex_quadrature() {
        let derivatives = hex_shape_derivatives(&natural);
        let (_, det, inv) = hex_jacobian(&coords, &derivatives);
        let grad = physical_gradients(&inv, &derivatives);
        let mut f = [0.0; 9];
        for i in 0..3 {
            f[i * 3 + i] = 1.0;
        }
        for (a, grad_a) in grad.iter().enumerate() {
            let node = mesh.conn[elem_idx][a];
            for i in 0..3 {
                let u_a = u[node * 3 + i];
                for col in 0..3 {
                    f[i * 3 + col] += u_a * grad_a[col];
                }
            }
        }
        // Stack stress: bit-identical to `stress` (same trait contract),
        // minus one heap allocation per quadrature point on this hot path.
        let mut p = [0.0; 9];
        material.stress_into(&f, &mut p);
        let w = weight * det;
        for a in 0..8 {
            for i in 0..3 {
                let mut r = 0.0;
                for col in 0..3 {
                    r += p[i * 3 + col] * grad[a][col];
                }
                residual[a * 3 + i] += r * w;
            }
        }
    }
    residual
}

/// Minimal hexahedral mesh with explicit coordinates.
#[derive(Debug, Clone)]
pub struct HexMesh {
    /// Node coordinates indexed by node id.
    pub coords: Vec<[f64; 3]>,
    /// Element connectivity in [`HEX_NODES`] order.
    pub conn: Vec<[usize; 8]>,
}

impl HexMesh {
    /// Build a mesh from coordinates and connectivity.
    #[must_use]
    pub fn new(coords: Vec<[f64; 3]>, conn: Vec<[usize; 8]>) -> Self {
        Self { coords, conn }
    }

    /// Number of nodes in the mesh.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }

    /// Single undistorted cube spanning `[0, h] x [0, h] x [0, h]`.
    #[must_use]
    pub fn unit_cube(h: f64) -> Self {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [h, 0.0, 0.0],
            [h, h, 0.0],
            [0.0, h, 0.0],
            [0.0, 0.0, h],
            [h, 0.0, h],
            [h, h, h],
            [0.0, h, h],
        ];
        Self::new(coords, vec![[0, 1, 2, 3, 4, 5, 6, 7]])
    }

    /// Structured brick of `nx * ny * nz` cubes over a box.
    #[must_use]
    pub fn brick(nx: usize, ny: usize, nz: usize, min: [f64; 3], max: [f64; 3]) -> Self {
        let mut coords = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    coords.push([
                        min[0] + (max[0] - min[0]) * i as f64 / nx as f64,
                        min[1] + (max[1] - min[1]) * j as f64 / ny as f64,
                        min[2] + (max[2] - min[2]) * k as f64 / nz as f64,
                    ]);
                }
            }
        }
        let node = |i: usize, j: usize, k: usize| k * (nx + 1) * (ny + 1) + j * (nx + 1) + i;
        let mut conn = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    conn.push([
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ]);
                }
            }
        }
        Self::new(coords, conn)
    }

    /// Coordinates of one element's nodes.
    #[must_use]
    pub fn element_coords(&self, elem_idx: usize) -> [[f64; 3]; 8] {
        let mut out = [[0.0; 3]; 8];
        for (slot, node) in out.iter_mut().zip(self.conn[elem_idx].iter()) {
            *slot = self.coords[*node];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_functions_partition_unity() {
        for qp in hex_quadrature() {
            let n = hex_shape(&qp.0);
            let sum: f64 = n.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-15);
            let d = hex_shape_derivatives(&qp.0);
            for j in 0..3 {
                let grad_sum: f64 = d.iter().map(|row| row[j]).sum();
                assert!(grad_sum.abs() < 1.0e-15);
            }
        }
    }

    #[test]
    fn distorted_gradients_reproduce_identity() {
        // Nonsymmetric Jacobian: physical gradients need the inverse transpose.
        let coords = [
            [0.0, 0.0, 0.0],
            [1.1, 0.05, 0.0],
            [1.05, 1.0, 0.02],
            [0.0, 1.05, 0.0],
            [0.02, 0.0, 1.0],
            [1.0, -0.03, 1.08],
            [1.02, 1.03, 0.97],
            [0.0, 0.98, 1.0],
        ];
        for qp in hex_quadrature() {
            let d = hex_shape_derivatives(&qp.0);
            let (_, det, inv) = hex_jacobian(&coords, &d);
            assert!(det > 0.0);
            let g = physical_gradients(&inv, &d);
            for i in 0..3 {
                let sum: f64 = g.iter().map(|row| row[i]).sum();
                assert!(sum.abs() < 1.0e-14, "gradient sum {sum:.3e}");
                for j in 0..3 {
                    let mut ident = 0.0;
                    for a in 0..8 {
                        ident += g[a][i] * coords[a][j];
                    }
                    let target = f64::from(i == j);
                    assert!((ident - target).abs() < 1.0e-12, "identity {ident:.6e}");
                }
            }
        }
    }

    #[test]
    fn cube_jacobian_is_constant() {
        let mesh = HexMesh::unit_cube(2.0);
        let coords = mesh.element_coords(0);
        for qp in hex_quadrature() {
            let d = hex_shape_derivatives(&qp.0);
            let (_, det, _) = hex_jacobian(&coords, &d);
            assert!((det - 1.0).abs() < 1.0e-14);
        }
    }

    #[test]
    fn q1_residual_is_self_equilibrated() {
        let material =
            crate::physics::solid::nonlinear_solid::NeoHookean::from_young(1.0e6, 0.3, 3);
        let mesh = HexMesh::unit_cube(1.0);
        // Irrotational stretch keeps P symmetric, so moments also balance.
        let mut u = vec![0.0; 24];
        for node in 0..8 {
            let [x, y, z] = mesh.coords[node];
            u[node * 3] = 0.02 * x + 0.01 * y + 0.005 * z;
            u[node * 3 + 1] = 0.01 * x + 0.03 * y - 0.004 * z;
            u[node * 3 + 2] = 0.005 * x - 0.004 * y + 0.025 * z;
        }
        let residual = q1_element_residual(&material, &mesh, 0, &u);
        // Internal forces sum to zero force and zero moment about the origin.
        let mut force = [0.0; 3];
        let mut moment = [0.0; 3];
        for a in 0..8 {
            let x = mesh.coords[a];
            for i in 0..3 {
                force[i] += residual[a * 3 + i];
            }
            let f = [residual[a * 3], residual[a * 3 + 1], residual[a * 3 + 2]];
            moment[0] += x[1] * f[2] - x[2] * f[1];
            moment[1] += x[2] * f[0] - x[0] * f[2];
            moment[2] += x[0] * f[1] - x[1] * f[0];
        }
        let scale: f64 = residual.iter().map(|r| r.abs()).sum::<f64>().max(1.0e-30);
        for i in 0..3 {
            assert!(force[i].abs() / scale < 1.0e-12);
            assert!(moment[i].abs() / scale < 1.0e-12);
        }
    }

    #[test]
    fn dense_solve_matches_known_system() {
        let mut matrix = vec![2.0, 1.0, 1.0, 3.0];
        let mut rhs = vec![5.0, 7.0];
        dense_solve(&mut matrix, &mut rhs, 2);
        assert!((rhs[0] - 1.6).abs() < 1.0e-14);
        assert!((rhs[1] - 1.8).abs() < 1.0e-14);
    }
}
