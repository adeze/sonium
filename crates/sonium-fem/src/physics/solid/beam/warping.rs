//! Thin-walled open-section beam element with warping torsion.
//!
//! Design choice: a 7th DOF per node. Each node carries the standard
//! `[ux, uy, uz, rx, ry, rz]` plus the rate of twist `psi = d(rx)/dx`, whose
//! work conjugate is the Vlasov bimoment. A built-in (warping-restrained)
//! end fixes `psi = 0`; a free end leaves it unconstrained. Twist is
//! interpolated with cubic Hermite polynomials in the nodal `(rx, psi)`
//! values, and the 4x4 torsion-warping block integrates Saint-Venant energy
//! `GJ (theta')^2` plus warping energy `E I_omega (theta'')^2` with exact
//! 4-point Gauss quadrature. The block converges to the Vlasov solution with
//! mesh refinement and reproduces uniform (warping-free) torsion exactly.
//! Bending, axial, and shear blocks are the straight prismatic ones.

// Rust guideline compliant 2026-02-21

use super::element3d::BeamElement3D;
use super::section3d::Section3D;

/// Warping beam element with 14 local DOFs.
///
/// Per-node layout is `[ux, uy, uz, rx, ry, rz, psi]` with `psi` the rate of
/// twist about local `x`. Assembly uses 7 DOFs per node through
/// [`WarpingBeamElement3D::global_dofs`]; do not mix warping and plain
/// 6-DOF elements in one global numbering.
#[derive(Debug, Clone)]
pub struct WarpingBeamElement3D {
    /// Node indices (2 nodes).
    pub nodes: [usize; 2],
    /// Element length in m.
    pub length: f64,
    /// Local axes in global coordinates, row-major (3 local axis rows).
    pub axes: [f64; 9],
    /// Cross-section properties (reads `warping_constant`).
    pub section: Section3D,
}

/// 4-point Gauss-Legendre nodes and weights on `[-1, 1]`.
///
/// Standard values (Abramowitz and Stegun, Table 25.4); exact through
/// degree 7, which covers the Hermite twist integrands.
const GAUSS4_XI: [f64; 4] = [
    -0.861_136_311_594_052_6,
    -0.339_981_043_584_856_3,
    0.339_981_043_584_856_3,
    0.861_136_311_594_052_6,
];
const GAUSS4_W: [f64; 4] = [
    0.347_854_845_137_453_8,
    0.652_145_154_862_546_1,
    0.652_145_154_862_546_1,
    0.347_854_845_137_453_8,
];

impl WarpingBeamElement3D {
    /// Create an element between global points `p1` and `p2`.
    ///
    /// The frame follows [`BeamElement3D`]: local `x` runs from node 1 to
    /// node 2, local `y` is the normalized component of `up` orthogonal to
    /// the axis, and `z = x cross y`.
    ///
    /// # Panics
    ///
    /// Panics for coincident nodes or an `up` hint parallel to the beam axis.
    #[must_use]
    pub fn new(
        nodes: [usize; 2],
        p1: [f64; 3],
        p2: [f64; 3],
        section: Section3D,
        up: [f64; 3],
    ) -> Self {
        let straight = BeamElement3D::new(nodes, p1, p2, section, up);
        Self {
            nodes,
            length: straight.length,
            axes: straight.axes,
            section,
        }
    }

    /// Torsion decay rate `mu = sqrt(GJ / E I_omega)` in rad/m.
    ///
    /// Returns `None` for sections without warping stiffness (pure
    /// Saint-Venant torsion, infinite decay rate).
    #[must_use]
    pub fn torsion_decay_rate(&self) -> Option<f64> {
        let warping = self.section.warping_stiffness();
        if warping <= 0.0 {
            return None;
        }
        Some((self.section.gj() / warping).sqrt())
    }

    /// 4x4 torsion-warping block on `[rx1, psi1, rx2, psi2]`, row-major.
    ///
    /// Cubic Hermite twist interpolation with exact Gauss quadrature of the
    /// Saint-Venant and warping strain energies.
    #[must_use]
    pub fn torsion_warping_block(&self) -> [f64; 16] {
        let l = self.length;
        let gj = self.section.gj();
        let eg = self.section.warping_stiffness();
        let mut k = [0.0; 16];
        for (xi, w) in GAUSS4_XI.iter().zip(GAUSS4_W.iter()) {
            // Map [-1, 1] to xi in [0, 1] with dx = (L/2) d(xi).
            let s = 0.5 * (xi + 1.0);
            // First derivatives of the Hermite twist shapes wrt x.
            let hp = [
                (-6.0 * s + 6.0 * s * s) / l,
                1.0 - 4.0 * s + 3.0 * s * s,
                (6.0 * s - 6.0 * s * s) / l,
                -2.0 * s + 3.0 * s * s,
            ];
            // Second derivatives wrt x.
            let hpp = [
                (-6.0 + 12.0 * s) / (l * l),
                (-4.0 + 6.0 * s) / l,
                (6.0 - 12.0 * s) / (l * l),
                (-2.0 + 6.0 * s) / l,
            ];
            let weight = 0.5 * w * l;
            for i in 0..4 {
                for j in 0..4 {
                    k[i * 4 + j] +=
                        weight * (gj * hp[i] * hp[j] + eg * hpp[i] * hpp[j]);
                }
            }
        }
        k
    }

    /// Local stiffness matrix (14x14, row-major).
    ///
    /// Axial, shear, and bending entries come from the straight prismatic
    /// element; the `rx` rows and columns are replaced by the
    /// torsion-warping block over `[rx1, psi1, rx2, psi2]`.
    #[must_use]
    pub fn stiffness_local(&self) -> [f64; 196] {
        let straight = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [self.length, 0.0, 0.0],
            self.section,
            [0.0, 1.0, 0.0],
        );
        let ks = straight.stiffness_local();
        // Straight 12-DOF indices to warping 14-DOF indices.
        let map = [0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12];
        let mut k = [0.0; 196];
        for i in 0..12 {
            if i == 3 || i == 9 {
                continue;
            }
            for j in 0..12 {
                if j == 3 || j == 9 {
                    continue;
                }
                k[map[i] * 14 + map[j]] = ks[i * 12 + j];
            }
        }
        let kw = self.torsion_warping_block();
        let twist = [3_usize, 6, 10, 13];
        for i in 0..4 {
            for j in 0..4 {
                k[twist[i] * 14 + twist[j]] = kw[i * 4 + j];
            }
        }
        k
    }

    /// Global stiffness matrix (14x14, row-major).
    ///
    /// `T` is block diagonal with 3x3 rotation blocks on the displacement
    /// and rotation triplets and ones on the scalar warping DOFs; the global
    /// matrix is `T' K_local T`.
    #[must_use]
    pub fn stiffness_global(&self) -> [f64; 196] {
        // Column block of each of the 14 local DOFs: 0..2 vector, 3 scalar.
        let block_of = |dof: usize| match dof {
            0..=2 => 0,
            3..=5 => 1,
            6 => 2,
            7..=9 => 3,
            10..=12 => 4,
            _ => 5,
        };
        let local = self.stiffness_local();
        let mut temp = [0.0; 196];
        for row in 0..14 {
            for col in 0..14 {
                let block = block_of(col);
                if block == 2 || block == 5 {
                    temp[row * 14 + col] = local[row * 14 + col];
                    continue;
                }
                let base = [0_usize, 3, 6, 7, 10, 13][block];
                let col_in_block = col - base;
                let mut sum = 0.0;
                for p in 0..3 {
                    sum += local[row * 14 + base + p] * self.axes[p * 3 + col_in_block];
                }
                temp[row * 14 + col] = sum;
            }
        }
        let mut global = [0.0; 196];
        for row in 0..14 {
            for col in 0..14 {
                let block = block_of(row);
                if block == 2 || block == 5 {
                    global[row * 14 + col] = temp[row * 14 + col];
                    continue;
                }
                let base = [0_usize, 3, 6, 7, 10, 13][block];
                let row_in_block = row - base;
                let mut sum = 0.0;
                for p in 0..3 {
                    sum += self.axes[p * 3 + row_in_block] * temp[(base + p) * 14 + col];
                }
                global[row * 14 + col] = sum;
            }
        }
        global
    }

    /// Global DOF indices with 7 DOFs per node.
    ///
    /// Node `i` owns `[7*i, ..., 7*i + 6]` for
    /// `[ux, uy, uz, rx, ry, rz, psi]`.
    #[must_use]
    pub fn global_dofs(&self) -> [usize; 14] {
        let n1 = self.nodes[0];
        let n2 = self.nodes[1];
        let mut dofs = [0_usize; 14];
        for component in 0..7 {
            dofs[component] = n1 * 7 + component;
            dofs[7 + component] = n2 * 7 + component;
        }
        dofs
    }

    /// Twist angle at station `xi` in `[0, 1]` for nodal twist DOFs.
    ///
    /// `twist` holds `[rx1, psi1, rx2, psi2]` in the element frame.
    #[must_use]
    pub fn twist_at(&self, twist: &[f64; 4], xi: f64) -> f64 {
        let s = xi;
        let l = self.length;
        let h = [
            1.0 - 3.0 * s * s + 2.0 * s * s * s,
            l * (s - 2.0 * s * s + s * s * s),
            3.0 * s * s - 2.0 * s * s * s,
            l * (-s * s + s * s * s),
        ];
        h[0] * twist[0] + h[1] * twist[1] + h[2] * twist[2] + h[3] * twist[3]
    }

    /// Internal bimoment at station `xi` in `[0, 1]` for nodal twist DOFs.
    ///
    /// Uses the Vlasov relation `B = -E I_omega theta''`; positive `B`
    /// pairs with positive warping curvature in the element frame.
    #[must_use]
    pub fn bimoment_at(&self, twist: &[f64; 4], xi: f64) -> f64 {
        let s = xi;
        let l = self.length;
        let hpp = [
            (-6.0 + 12.0 * s) / (l * l),
            (-4.0 + 6.0 * s) / l,
            (6.0 - 12.0 * s) / (l * l),
            (-2.0 + 6.0 * s) / l,
        ];
        let curvature =
            hpp[0] * twist[0] + hpp[1] * twist[1] + hpp[2] * twist[2] + hpp[3] * twist[3];
        -self.section.warping_stiffness() * curvature
    }
}

/// Assemble warping-beam stiffness into a global triplet matrix.
///
/// The global numbering carries 7 DOFs per node.
pub fn assemble_warping_beam_stiffness(
    elements: &[WarpingBeamElement3D],
    n_dofs: usize,
) -> crate::assembly::TripletMatrix<f64> {
    let mut triplet = crate::assembly::TripletMatrix::new(n_dofs, n_dofs);
    for elem in elements {
        let k = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let val = k[i * 14 + j];
                if val.abs() > 1.0e-15 {
                    triplet.push(gi, gj, val);
                }
            }
        }
    }
    triplet
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i_section() -> Section3D {
        Section3D::i_beam(200.0e9, 0.3, 0.1, 0.2, 0.01, 7800.0, false)
    }

    fn symmetry_error(k: &[f64; 196]) -> f64 {
        let mut worst: f64 = 0.0;
        for i in 0..14 {
            for j in 0..14 {
                worst = worst.max((k[i * 14 + j] - k[j * 14 + i]).abs());
            }
        }
        worst / k.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
    }

    /// Dense solve with partial pivoting for unit tests.
    fn dense_solve(matrix: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
        let mut m = matrix.to_vec();
        let mut x = rhs.to_vec();
        for col in 0..n {
            let mut pivot = col;
            let mut best = m[col * n + col].abs();
            for row in col + 1..n {
                let candidate = m[row * n + col].abs();
                if candidate > best {
                    best = candidate;
                    pivot = row;
                }
            }
            assert!(best > 1.0e-300, "singular test system");
            if pivot != col {
                for k in 0..n {
                    m.swap(col * n + k, pivot * n + k);
                }
                x.swap(col, pivot);
            }
            let diag = m[col * n + col];
            for row in col + 1..n {
                let factor = m[row * n + col] / diag;
                for k in col..n {
                    m[row * n + k] -= factor * m[col * n + k];
                }
                x[row] -= factor * x[col];
            }
        }
        for row in (0..n).rev() {
            let mut sum = x[row];
            for k in row + 1..n {
                sum -= m[row * n + k] * x[k];
            }
            x[row] = sum / m[row * n + row];
        }
        x
    }

    #[test]
    fn stiffness_is_symmetric() {
        let elem = WarpingBeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            i_section(),
            [0.0, 1.0, 0.0],
        );
        assert!(symmetry_error(&elem.stiffness_local()) < 1.0e-12);
        assert!(symmetry_error(&elem.stiffness_global()) < 1.0e-12);
    }

    #[test]
    fn saint_venant_limit_is_exact_with_one_element() {
        // Zero warping constant with warping unrestrained (psi free): one
        // element must give TL/GJ exactly.
        let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
        assert_eq!(section.warping_constant, 0.0);
        let elem = WarpingBeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        );
        let k = elem.stiffness_local();
        // Node-1 structural DOFs fixed, psi1 (index 6) free.
        let free: Vec<usize> = [6].into_iter().chain(7..14).collect();
        let n = free.len();
        let mut matrix = vec![0.0; n * n];
        for (i, &gi) in free.iter().enumerate() {
            for (j, &gj) in free.iter().enumerate() {
                matrix[i * n + j] = k[gi * 14 + gj];
            }
        }
        let mut rhs = vec![0.0; n];
        rhs[free.iter().position(|&d| d == 10).unwrap()] = 1000.0;
        let u = dense_solve(&matrix, &rhs, n);
        let tip = u[free.iter().position(|&d| d == 10).unwrap()];
        let expected = 1000.0 * 2.0 / section.gj();
        assert!((tip - expected).abs() / expected < 1.0e-12);
    }

    #[test]
    fn free_warping_recovers_uniform_torsion() {
        // Warping stiffness present but psi free at both ends: the minimizer
        // is linear twist, represented exactly by the Hermite interpolation.
        let section = i_section();
        let elem = WarpingBeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        );
        let k = elem.stiffness_local();
        // Fix node-1 translations and rotations but leave psi1 free.
        let fixed = [0, 1, 2, 3, 4, 5];
        let free: Vec<usize> = (0..14).filter(|d| !fixed.contains(d)).collect();
        let n = free.len();
        let mut matrix = vec![0.0; n * n];
        for (i, &gi) in free.iter().enumerate() {
            for (j, &gj) in free.iter().enumerate() {
                matrix[i * n + j] = k[gi * 14 + gj];
            }
        }
        let mut rhs = vec![0.0; n];
        // Torque at node-2 rx (local index 10).
        rhs[free.iter().position(|&d| d == 10).unwrap()] = 1000.0;
        let u = dense_solve(&matrix, &rhs, n);
        let tip = u[free.iter().position(|&d| d == 10).unwrap()];
        let expected = 1000.0 * 2.0 / section.gj();
        assert!((tip - expected).abs() / expected < 1.0e-9);
    }

    #[test]
    fn restrained_cantilever_matches_vlasov() {
        // I-beam cantilever, tip torque, warping fixed at the root.
        let section = i_section();
        let length = 2.0;
        let n_elements = 8;
        let dx = length / n_elements as f64;
        let elements: Vec<WarpingBeamElement3D> = (0..n_elements)
            .map(|i| {
                WarpingBeamElement3D::new(
                    [i, i + 1],
                    [i as f64 * dx, 0.0, 0.0],
                    [(i + 1) as f64 * dx, 0.0, 0.0],
                    section,
                    [0.0, 1.0, 0.0],
                )
            })
            .collect();
        let n = (n_elements + 1) * 7;
        let mut k = vec![0.0; n * n];
        for elem in &elements {
            let ke = elem.stiffness_global();
            let dofs = elem.global_dofs();
            for (i, &gi) in dofs.iter().enumerate() {
                for (j, &gj) in dofs.iter().enumerate() {
                    k[gi * n + gj] += ke[i * 14 + j];
                }
            }
        }
        let free: Vec<usize> = (7..n).collect();
        let m = free.len();
        let mut matrix = vec![0.0; m * m];
        for (i, &gi) in free.iter().enumerate() {
            for (j, &gj) in free.iter().enumerate() {
                matrix[i * m + j] = k[gi * n + gj];
            }
        }
        let mut rhs = vec![0.0; m];
        rhs[m - 4] = 1000.0;
        let u = dense_solve(&matrix, &rhs, m);
        let mu = section.gj() / section.warping_stiffness();
        let mu = mu.sqrt();
        let expected =
            1000.0 / section.gj() * (length - (mu * length).tanh() / mu);
        let error = (u[m - 4] - expected).abs() / expected;
        assert!(error < 1.0e-3, "tip twist error {error:.3e}");
    }
}
