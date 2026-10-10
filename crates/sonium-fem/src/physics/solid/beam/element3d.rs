//! Timoshenko space-frame beam element.
//!
//! A 2-node, 12-DOF prismatic beam with axial stretch, Saint-Venant torsion,
//! and biaxial bending with per-plane Timoshenko shear. Local node order is
//! `[ux, uy, uz, rx, ry, rz]`; the element matrix concatenates node 1 and
//! node 2 blocks. Consistent and lumped mass matrices support dynamics.

// Rust guideline compliant 2026-02-21

use super::section3d::Section3D;

/// Space-frame beam element with 12 local DOFs.
///
/// The local `x` axis runs from node 1 to node 2. Transverse axes follow the
/// `up` hint passed to [`BeamElement3D::new`]: local `y` is the normalized
/// component of `up` orthogonal to the beam axis, and `z = x cross y`.
#[derive(Debug, Clone)]
pub struct BeamElement3D {
    /// Node indices (2 nodes).
    pub nodes: [usize; 2],
    /// Element length in m.
    pub length: f64,
    /// Local axes in global coordinates, row-major (3 local axis rows).
    pub axes: [f64; 9],
    /// Cross-section properties.
    pub section: Section3D,
}

impl BeamElement3D {
    /// Create an element between global points `p1` and `p2`.
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
        let axis = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        assert!(length > 0.0, "beam element needs distinct nodes");
        let ex = [axis[0] / length, axis[1] / length, axis[2] / length];
        // Remove the axial component of the up hint, then normalize.
        let dot = up[0] * ex[0] + up[1] * ex[1] + up[2] * ex[2];
        let mut ey = [
            up[0] - dot * ex[0],
            up[1] - dot * ex[1],
            up[2] - dot * ex[2],
        ];
        let ey_norm = (ey[0] * ey[0] + ey[1] * ey[1] + ey[2] * ey[2]).sqrt();
        assert!(ey_norm > 1.0e-12, "up hint is parallel to the beam axis");
        ey[0] /= ey_norm;
        ey[1] /= ey_norm;
        ey[2] /= ey_norm;
        // Right-handed frame: ez = ex cross ey.
        let ez = [
            ex[1] * ey[2] - ex[2] * ey[1],
            ex[2] * ey[0] - ex[0] * ey[2],
            ex[0] * ey[1] - ex[1] * ey[0],
        ];
        Self {
            nodes,
            length,
            axes: [
                ex[0], ex[1], ex[2], ey[0], ey[1], ey[2], ez[0], ez[1], ez[2],
            ],
            section,
        }
    }

    /// Timoshenko bending block for one plane.
    ///
    /// Returns `[a, b, c, d]` with `K = [[a, b, -a, b], [b, c, -b, d], ...]`
    /// for `EI`, shear parameter `phi`, and length `l`.
    fn bending_block(ei: f64, phi: f64, l: f64) -> [f64; 4] {
        let l2 = l * l;
        let denom = (1.0 + phi) * l2 * l;
        [
            12.0 * ei / denom,
            6.0 * ei * l / denom,
            (4.0 + phi) * ei * l2 / denom,
            (2.0 - phi) * ei * l2 / denom,
        ]
    }

    /// Local stiffness matrix (12x12, row-major).
    ///
    /// Bending about `z` uses DOFs `[uy1, rz1, uy2, rz2]`; bending about `y`
    /// uses `[uz1, ry1, uz2, ry2]` with the sign pattern for `ry = -dw/dx`.
    #[must_use]
    pub fn stiffness_local(&self) -> [f64; 144] {
        let l = self.length;
        let mut k = [0.0; 144];

        // Axial block on ux DOFs 0 and 6.
        let axial = self.section.ea() / l;
        k[0] = axial;
        k[6] = -axial;
        k[6 * 12] = -axial;
        k[6 * 12 + 6] = axial;

        // Torsion block on rx DOFs 3 and 9.
        let torsion = self.section.gj() / l;
        k[3 * 12 + 3] = torsion;
        k[3 * 12 + 9] = -torsion;
        k[9 * 12 + 3] = -torsion;
        k[9 * 12 + 9] = torsion;

        // Bending about z (x-y plane) on [1, 5, 7, 11].
        let [a, b, c, d] = Self::bending_block(self.section.eiz(), self.section.phi_z(l), l);
        let plane_z = [1_usize, 5, 7, 11];
        let block_z = [[a, b, -a, b], [b, c, -b, d], [-a, -b, a, -b], [b, d, -b, c]];
        for i in 0..4 {
            for j in 0..4 {
                k[plane_z[i] * 12 + plane_z[j]] = block_z[i][j];
            }
        }

        // Bending about y (x-z plane) on [2, 4, 8, 10] with ry sign flips.
        let [a, b, c, d] = Self::bending_block(self.section.eiy(), self.section.phi_y(l), l);
        let plane_y = [2_usize, 4, 8, 10];
        let block_y = [[a, -b, -a, -b], [-b, c, b, d], [-a, b, a, b], [-b, d, b, c]];
        for i in 0..4 {
            for j in 0..4 {
                k[plane_y[i] * 12 + plane_y[j]] = block_y[i][j];
            }
        }

        k
    }

    /// Consistent mass matrix (12x12, row-major).
    ///
    /// Axial and torsional blocks use the linear `[[2, 1], [1, 2]]` form;
    /// bending planes reuse the Euler-Bernoulli cubic block. Torsional rotary
    /// inertia uses the polar second moment `iy + iz`.
    #[must_use]
    pub fn mass_local_consistent(&self) -> [f64; 144] {
        let rho_a = self.section.mass_per_length();
        let l = self.length;
        let l2 = l * l;
        let mut m = [0.0; 144];

        // Axial block.
        let axial = rho_a * l / 6.0;
        m[0] = 2.0 * axial;
        m[6] = axial;
        m[6 * 12] = axial;
        m[6 * 12 + 6] = 2.0 * axial;

        // Torsional rotary inertia block on rx DOFs 3 and 9.
        let rotary = self.section.density * (self.section.iy + self.section.iz) * l / 6.0;
        m[3 * 12 + 3] = 2.0 * rotary;
        m[3 * 12 + 9] = rotary;
        m[9 * 12 + 3] = rotary;
        m[9 * 12 + 9] = 2.0 * rotary;

        // Bending blocks share the cubic translational coefficients.
        let mb = rho_a * l / 420.0;
        let coeffs = [
            [156.0, 22.0 * l, 54.0, -13.0 * l],
            [22.0 * l, 4.0 * l2, 13.0 * l, -3.0 * l2],
            [54.0, 13.0 * l, 156.0, -22.0 * l],
            [-13.0 * l, -3.0 * l2, -22.0 * l, 4.0 * l2],
        ];
        // About z on [1, 5, 7, 11] with signs as tabulated.
        let plane_z = [1_usize, 5, 7, 11];
        for i in 0..4 {
            for j in 0..4 {
                m[plane_z[i] * 12 + plane_z[j]] = mb * coeffs[i][j];
            }
        }
        // About y on [2, 4, 8, 10]: ry rows and columns flip sign.
        let plane_y = [2_usize, 4, 8, 10];
        let signs = [1.0, -1.0, 1.0, -1.0];
        for i in 0..4 {
            for j in 0..4 {
                m[plane_y[i] * 12 + plane_y[j]] = signs[i] * signs[j] * mb * coeffs[i][j];
            }
        }

        m
    }

    /// Lumped mass matrix (12x12, diagonal row-major).
    ///
    /// Half the element mass sits on each node translation; rotations carry
    /// no lumped inertia, the standard explicit-dynamics choice.
    #[must_use]
    pub fn mass_local_lumped(&self) -> [f64; 144] {
        let half = self.section.mass_per_length() * self.length / 2.0;
        let mut m = [0.0; 144];
        for node_base in [0_usize, 6] {
            for component in 0..3 {
                let dof = node_base + component;
                m[dof * 12 + dof] = half;
            }
        }
        m
    }

    /// Transform a local 12x12 matrix to global coordinates.
    ///
    /// `T` is block diagonal with four copies of the 3x3 axis matrix, and the
    /// global matrix is `T' K_local T`.
    fn transform_12x12(&self, local: &[f64; 144]) -> [f64; 144] {
        transform_12x12(&self.axes, local)
    }

    /// Global stiffness matrix (12x12, row-major).
    #[must_use]
    pub fn stiffness_global(&self) -> [f64; 144] {
        self.transform_12x12(&self.stiffness_local())
    }

    /// Global consistent mass matrix (12x12, row-major).
    #[must_use]
    pub fn mass_global_consistent(&self) -> [f64; 144] {
        self.transform_12x12(&self.mass_local_consistent())
    }

    /// Global lumped mass matrix (12x12, row-major).
    #[must_use]
    pub fn mass_global_lumped(&self) -> [f64; 144] {
        self.transform_12x12(&self.mass_local_lumped())
    }

    /// Global DOF indices with 6 DOFs per node.
    ///
    /// Node `i` owns `[6*i, ..., 6*i + 5]` for `[ux, uy, uz, rx, ry, rz]`.
    #[must_use]
    pub fn global_dofs(&self) -> [usize; 12] {
        let n1 = self.nodes[0];
        let n2 = self.nodes[1];
        let mut dofs = [0_usize; 12];
        for component in 0..6 {
            dofs[component] = n1 * 6 + component;
            dofs[6 + component] = n2 * 6 + component;
        }
        dofs
    }
}

/// Transform a local 12x12 beam matrix to global coordinates.
///
/// `T` is block diagonal with four copies of the 3x3 axis matrix (rows are
/// local axes in global coordinates), and the global matrix is `T' K T`.
/// Shared by the straight, curved, and tapered 3-D beam elements.
pub(crate) fn transform_12x12(axes: &[f64; 9], local: &[f64; 144]) -> [f64; 144] {
    // temp = K_local * T; only the column block of T contributes.
    let mut temp = [0.0; 144];
    for row in 0..12 {
        for col in 0..12 {
            let block = col / 3;
            let col_in_block = col % 3;
            let mut sum = 0.0;
            for p in 0..3 {
                sum += local[row * 12 + block * 3 + p] * axes[p * 3 + col_in_block];
            }
            temp[row * 12 + col] = sum;
        }
    }
    // global = T' * temp; only the row block of T' contributes.
    let mut global = [0.0; 144];
    for row in 0..12 {
        for col in 0..12 {
            let block = row / 3;
            let row_in_block = row % 3;
            let mut sum = 0.0;
            for p in 0..3 {
                sum += axes[p * 3 + row_in_block] * temp[(block * 3 + p) * 12 + col];
            }
            global[row * 12 + col] = sum;
        }
    }
    global
}

/// Assemble space-frame stiffness into a global triplet matrix.
pub fn assemble_beam3d_stiffness(
    elements: &[BeamElement3D],
    n_dofs: usize,
) -> crate::assembly::TripletMatrix<f64> {
    let mut triplet = crate::assembly::TripletMatrix::new(n_dofs, n_dofs);
    for elem in elements {
        let k = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let val = k[i * 12 + j];
                if val.abs() > 1.0e-15 {
                    triplet.push(gi, gj, val);
                }
            }
        }
    }
    triplet
}

/// Assemble space-frame mass into a global triplet matrix.
///
/// Set `lumped` for the diagonal mass; otherwise the consistent mass applies.
pub fn assemble_beam3d_mass(
    elements: &[BeamElement3D],
    n_dofs: usize,
    lumped: bool,
) -> crate::assembly::TripletMatrix<f64> {
    let mut triplet = crate::assembly::TripletMatrix::new(n_dofs, n_dofs);
    for elem in elements {
        let m = if lumped {
            elem.mass_global_lumped()
        } else {
            elem.mass_global_consistent()
        };
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let val = m[i * 12 + j];
                if val.abs() > 1.0e-15 {
                    triplet.push(gi, gj, val);
                }
            }
        }
    }
    triplet
}

/// Beam model wrapper implementing [`ElementForm`](crate::forms::ElementForm).
#[derive(Debug)]
pub struct BeamModel3D {
    /// Space-frame elements.
    pub elements: Vec<BeamElement3D>,
    /// Total number of global DOFs.
    pub n_dofs: usize,
}

impl BeamModel3D {
    /// Build a model from elements and a global DOF count.
    #[must_use]
    pub fn new(elements: Vec<BeamElement3D>, n_dofs: usize) -> Self {
        Self { elements, n_dofs }
    }
}

impl crate::forms::ElementForm<f64> for BeamModel3D {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        12
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.elements[elem_idx].global_dofs().to_vec()
    }

    fn element_matrix(&self, elem_idx: usize) -> Vec<f64> {
        self.elements[elem_idx].stiffness_global().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::beam::element::BeamElement2D;

    fn test_section() -> Section3D {
        Section3D::rectangular(200.0e9, 0.3, 0.02, 0.01, 7800.0, true)
    }

    fn symmetry_error(k: &[f64; 144]) -> f64 {
        let mut worst: f64 = 0.0;
        for i in 0..12 {
            for j in 0..12 {
                worst = worst.max((k[i * 12 + j] - k[j * 12 + i]).abs());
            }
        }
        worst
    }

    #[test]
    fn stiffness_is_symmetric_with_six_rigid_modes() {
        let elem = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            test_section(),
            [0.0, 1.0, 0.0],
        );
        let k = elem.stiffness_local();
        assert!(symmetry_error(&k) < 1.0e-9);
        // Rigid translations and small rigid rotations carry no strain energy.
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let mut modes = [[0.0; 12]; 6];
        for mode in 0..3 {
            modes[mode][mode] = 1.0;
            modes[mode][6 + mode] = 1.0;
        }
        for (mode, axis) in (3..6).zip([0, 1, 2]) {
            for (node, p) in [p1, p2].iter().enumerate() {
                // Rotation about `axis`: u = axis cross p, theta = axis.
                let u = [
                    (axis == 1) as u8 as f64 * p[2] - (axis == 2) as u8 as f64 * p[1],
                    (axis == 2) as u8 as f64 * p[0] - (axis == 0) as u8 as f64 * p[2],
                    (axis == 0) as u8 as f64 * p[1] - (axis == 1) as u8 as f64 * p[0],
                ];
                for component in 0..3 {
                    modes[mode][node * 6 + component] = u[component];
                    modes[mode][node * 6 + 3 + component] = (component == axis) as u8 as f64;
                }
            }
        }
        for mode in &modes {
            let mut energy = 0.0;
            for i in 0..12 {
                let mut row = 0.0;
                for j in 0..12 {
                    row += k[i * 12 + j] * mode[j];
                }
                energy += mode[i] * row;
            }
            assert!(energy.abs() < 1.0e-6, "rigid mode energy {energy}");
        }
    }

    #[test]
    fn planar_bending_matches_2d_element() {
        let section3d = test_section();
        let elem3d = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            section3d,
            [0.0, 1.0, 0.0],
        );
        let elem2d = BeamElement2D::new([0, 1], 0.0, 0.0, 2.0, 0.0, section3d.to_planar_xy());
        let k3 = elem3d.stiffness_global();
        let k2 = elem2d.stiffness_global();
        // 2D DOFs [u1, v1, t1, u2, v2, t2] map to 3D [ux, uy, rz] per node.
        let map = [0_usize, 1, 5, 6, 7, 11];
        for i in 0..6 {
            for j in 0..6 {
                let diff = (k3[map[i] * 12 + map[j]] - k2[i * 6 + j]).abs();
                assert!(diff < 1.0e-9, "entry ({i},{j}) differs by {diff}");
            }
        }
    }

    #[test]
    fn torsion_matches_twist_formula() {
        let section = Section3D::circular(70.0e9, 0.33, 0.05, 2700.0, false);
        let elem = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            section,
            [0.0, 0.0, 1.0],
        );
        let k = elem.stiffness_local();
        let expected = section.gj() / 3.0;
        assert!((k[3 * 12 + 3] - expected).abs() / expected < 1.0e-15);
        assert!((k[3 * 12 + 9] + expected).abs() / expected < 1.0e-15);
    }

    #[test]
    fn mass_matrices_conserve_total_mass() {
        let section = test_section();
        let elem = BeamElement3D::new(
            [0, 1],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 5.0],
            section,
            [0.0, 1.0, 0.0],
        );
        let total = section.mass_per_length() * elem.length;
        for m in [elem.mass_local_consistent(), elem.mass_local_lumped()] {
            // Rigid translation in x: 1' M 1 over ux DOFs equals total mass.
            let sum = m[0] + m[6] + m[6 * 12] + m[6 * 12 + 6];
            assert!((sum - total).abs() / total < 1.0e-12);
        }
        let lumped = elem.mass_local_lumped();
        for i in 0..12 {
            for j in 0..12 {
                if i != j {
                    assert_eq!(lumped[i * 12 + j], 0.0);
                }
            }
        }
        assert!(symmetry_error(&elem.mass_global_consistent()) / total < 1.0e-12);
    }
}
