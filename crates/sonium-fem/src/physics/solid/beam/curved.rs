//! Circular-arch beam element with constant curvature.
//!
//! A 2-node, 12-DOF curved beam for in-plane arches. Both the in-plane and
//! the out-of-plane blocks are flexibility-exact: end loads at node 2
//! produce section resultants along the arc, complementary energy is
//! integrated with 8-point Gauss-Legendre quadrature, and each 3x3
//! flexibility inverts to stiffness through exact rigid transport from
//! node 1. A single element therefore reproduces thin-arch theory (bending
//! plus axial, Timoshenko shear when the section enables it) to quadrature
//! precision, with exact rigid-body modes. Torsion/out-of-plane-bending
//! coupling of curved members is neglected (thin-arch approximation, valid
//! for radius-to-depth ratios above roughly 10).

// Rust guideline compliant 2026-02-21

use super::element3d::transform_12x12;
use super::section3d::Section3D;

/// Circular-arch beam element with 12 local DOFs.
///
/// The arch plane is fixed by the two nodes and the center passed to
/// [`CurvedBeamElement3D::from_arc`]. Local `x` is tangent to the arc at
/// midspan (pointing from node 1 toward node 2), local `y` is the outward
/// radial direction, and `z` completes the right-handed frame. Node order
/// and DOF layout match [`BeamElement3D`]: `[ux, uy, uz, rx, ry, rz]` per
/// node.
#[derive(Debug, Clone)]
pub struct CurvedBeamElement3D {
    /// Node indices (2 nodes).
    pub nodes: [usize; 2],
    /// Arch radius in m.
    pub radius: f64,
    /// Arc angle of node 1 measured from midspan in rad.
    pub beta1: f64,
    /// Arc angle of node 2 measured from midspan in rad.
    pub beta2: f64,
    /// Arc length in m.
    pub arc_length: f64,
    /// Local axes in global coordinates, row-major (3 local axis rows).
    pub axes: [f64; 9],
    /// Cross-section properties.
    pub section: Section3D,
}

/// 8-point Gauss-Legendre nodes and weights on `[-1, 1]`.
///
/// Values follow the standard tables (for example Abramowitz and Stegun,
/// Table 25.4); the weights sum to exactly 2.
const GAUSS8_XI: [f64; 8] = [
    -0.960_289_856_497_536_2,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_2,
];
const GAUSS8_W: [f64; 8] = [
    0.101_228_536_290_376_3,
    0.222_381_034_453_374_5,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

impl CurvedBeamElement3D {
    /// Build an arch element through global points `p1` and `p2`.
    ///
    /// Both points must sit on a circle of common radius about `center`, and
    /// the subtended angle must lie strictly between 0 and pi. Model larger
    /// arcs with several elements.
    ///
    /// # Panics
    ///
    /// Panics when the points are not co-circular about `center`, when they
    /// coincide with the center or each other, or when the subtended angle
    /// reaches pi (antipodal nodes leave the midspan frame undefined).
    #[must_use]
    pub fn from_arc(
        nodes: [usize; 2],
        p1: [f64; 3],
        p2: [f64; 3],
        center: [f64; 3],
        section: Section3D,
    ) -> Self {
        let r1 = [p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]];
        let r2 = [p2[0] - center[0], p2[1] - center[1], p2[2] - center[2]];
        let norm1 = (r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]).sqrt();
        let norm2 = (r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]).sqrt();
        assert!(norm1 > 0.0, "arch node 1 coincides with the center");
        assert!(norm2 > 0.0, "arch node 2 coincides with the center");
        assert!(
            (norm1 - norm2).abs() <= 1.0e-9 * norm1,
            "arch nodes are not co-circular about the center"
        );
        let radius = 0.5 * (norm1 + norm2);
        let u1 = [r1[0] / norm1, r1[1] / norm1, r1[2] / norm1];
        let u2 = [r2[0] / norm2, r2[1] / norm2, r2[2] / norm2];
        // Binormal from spoke 1 to spoke 2; vanishes for coincident nodes.
        let mut ez = [
            u1[1] * u2[2] - u1[2] * u2[1],
            u1[2] * u2[0] - u1[0] * u2[2],
            u1[0] * u2[1] - u1[1] * u2[0],
        ];
        let ez_norm = (ez[0] * ez[0] + ez[1] * ez[1] + ez[2] * ez[2]).sqrt();
        assert!(
            ez_norm > 1.0e-9,
            "arch nodes coincide or share a spoke with the center"
        );
        ez[0] /= ez_norm;
        ez[1] /= ez_norm;
        ez[2] /= ez_norm;
        // Midspan radial direction; vanishes for antipodal nodes.
        let mut ey = [u1[0] + u2[0], u1[1] + u2[1], u1[2] + u2[2]];
        let ey_norm = (ey[0] * ey[0] + ey[1] * ey[1] + ey[2] * ey[2]).sqrt();
        assert!(
            ey_norm > 1.0e-9,
            "arch subtends pi or more; split it into smaller elements"
        );
        ey[0] /= ey_norm;
        ey[1] /= ey_norm;
        ey[2] /= ey_norm;
        // Tangent keeps the frame right-handed: ex x ey = ez.
        let mut ex = [
            ey[1] * ez[2] - ey[2] * ez[1],
            ey[2] * ez[0] - ey[0] * ez[2],
            ey[0] * ez[1] - ey[1] * ez[0],
        ];
        let chord = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        if ex[0] * chord[0] + ex[1] * chord[1] + ex[2] * chord[2] < 0.0 {
            // Flip the tangent toward node 2 and the binormal with it so the
            // frame stays right-handed.
            ex = [-ex[0], -ex[1], -ex[2]];
            ez = [-ez[0], -ez[1], -ez[2]];
        }
        let beta1 = (u1[0] * ex[0] + u1[1] * ex[1] + u1[2] * ex[2])
            .atan2(u1[0] * ey[0] + u1[1] * ey[1] + u1[2] * ey[2]);
        let beta2 = (u2[0] * ex[0] + u2[1] * ex[1] + u2[2] * ex[2])
            .atan2(u2[0] * ey[0] + u2[1] * ey[1] + u2[2] * ey[2]);
        let arc_length = radius * (beta2 - beta1).abs();
        Self {
            nodes,
            radius,
            beta1,
            beta2,
            arc_length,
            axes: [
                ex[0], ex[1], ex[2], ey[0], ey[1], ey[2], ez[0], ez[1], ez[2],
            ],
            section,
        }
    }

    /// Subtended arc angle in rad.
    #[must_use]
    pub fn arc_angle(&self) -> f64 {
        (self.beta2 - self.beta1).abs()
    }

    /// Local `(x, y)` of the arc point at angle `beta` from midspan.
    ///
    /// The origin sits at the midspan arc point, so the center is at
    /// `(0, -R)` and the point is `(R sin beta, R (cos beta - 1))`.
    fn local_point(&self, beta: f64) -> [f64; 2] {
        [self.radius * beta.sin(), self.radius * (beta.cos() - 1.0)]
    }

    /// In-plane 3x3 flexibility at node 2 with node 1 fixed.
    ///
    /// Columns hold the end displacements for unit `[Fx2, Fy2, M2]` loads in
    /// the local midspan frame. Bending uses `Iz`, axial uses `EA`, and
    /// shear uses the `y` shear stiffness for Timoshenko sections; EB
    /// sections carry no shear flexibility.
    fn flexibility_inplane(&self) -> [f64; 9] {
        let mid = 0.5 * (self.beta1 + self.beta2);
        let half = 0.5 * (self.beta2 - self.beta1).abs();
        let end = self.local_point(self.beta2);
        let ei = self.section.eiz();
        let ea = self.section.ea();
        let ga = self.section.shear_stiffness_y();
        let mut flex = [0.0; 9];
        for (xi, w) in GAUSS8_XI.iter().zip(GAUSS8_W.iter()) {
            let beta = mid + xi * half * (self.beta2 - self.beta1).signum();
            let point = self.local_point(beta);
            // Lever arms from the section to the loaded end.
            let rx = end[0] - point[0];
            let ry = end[1] - point[1];
            // Tangent and outward normal of increasing beta.
            let tx = beta.cos();
            let ty = -beta.sin();
            let nx = beta.sin();
            let ny = beta.cos();
            // Resultants for unit Fx2, Fy2, M2 in that column order.
            let moment = [-ry, rx, 1.0];
            let axial = [tx, ty, 0.0];
            let shear = [nx, ny, 0.0];
            let weight = w * half * self.radius;
            for j in 0..3 {
                for k in 0..3 {
                    flex[j * 3 + k] +=
                        weight * (moment[j] * moment[k] / ei + axial[j] * axial[k] / ea);
                    if ga > 0.0 {
                        flex[j * 3 + k] += weight * shear[j] * shear[k] / ga;
                    }
                }
            }
        }
        flex
    }

    /// Out-of-plane 3x3 flexibility at node 2 with node 1 fixed.
    ///
    /// Columns hold the end responses for unit `[Fz2, Mx2, My2]` loads in
    /// the local midspan frame. Torque integrates over `GJ`, out-of-plane
    /// bending over `EIy`, and transverse shear over the `z` shear stiffness
    /// for Timoshenko sections; EB sections carry no shear flexibility.
    /// Torsion/bending coupling is neglected (thin-arch approximation).
    fn flexibility_out_of_plane(&self) -> [f64; 9] {
        let mid = 0.5 * (self.beta1 + self.beta2);
        let half = 0.5 * (self.beta2 - self.beta1).abs();
        let sign = (self.beta2 - self.beta1).signum();
        let end = self.local_point(self.beta2);
        let gj = self.section.gj();
        let eiy = self.section.eiy();
        let ga = self.section.shear_stiffness_z();
        let mut flex = [0.0; 9];
        for (xi, w) in GAUSS8_XI.iter().zip(GAUSS8_W.iter()) {
            let beta = mid + xi * half * sign;
            let point = self.local_point(beta);
            // Lever arms from the section to the loaded end.
            let rx = end[0] - point[0];
            let ry = end[1] - point[1];
            // Tangent and outward normal of increasing beta.
            let tx = beta.cos();
            let ty = -beta.sin();
            let nx = beta.sin();
            let ny = beta.cos();
            // Moment of unit Fz2 about the section: (ry, -rx, 0).
            // Torque about the tangent for unit [Fz2, Mx2, My2].
            let torque = [ry * tx - rx * ty, tx, ty];
            // Bending about the normal for unit [Fz2, Mx2, My2].
            let bending = [ry * nx - rx * ny, nx, ny];
            let shear = [1.0, 0.0, 0.0];
            let weight = w * half * self.radius;
            for j in 0..3 {
                for k in 0..3 {
                    flex[j * 3 + k] +=
                        weight * (torque[j] * torque[k] / gj + bending[j] * bending[k] / eiy);
                    if ga > 0.0 {
                        flex[j * 3 + k] += weight * shear[j] * shear[k] / ga;
                    }
                }
            }
        }
        flex
    }

    /// Expand a 3x3 end stiffness to a 6x6 nodal block via `B' K33 B`.
    ///
    /// `b` maps the 6 nodal DOFs of one plane to 3 elastic end deformations.
    fn expand_block(stiff33: &[f64; 9], b: &[f64; 18]) -> [f64; 36] {
        let mut k6 = [0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut sum = 0.0;
                for a in 0..3 {
                    for c in 0..3 {
                        sum += b[a * 6 + i] * stiff33[a * 3 + c] * b[c * 6 + j];
                    }
                }
                k6[i * 6 + j] = sum;
            }
        }
        k6
    }

    /// Local stiffness matrix (12x12, row-major).
    ///
    /// The in-plane `[ux, uy, rz]` block and the out-of-plane
    /// `[uz, rx, ry]` block are inverse flexibilities transported with exact
    /// rigid maps, so all six rigid-body modes carry zero energy.
    #[must_use]
    pub fn stiffness_local(&self) -> [f64; 144] {
        // Rigid transport from node 1 to node 2 in local coordinates.
        let p1 = self.local_point(self.beta1);
        let p2 = self.local_point(self.beta2);
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let mut k = [0.0; 144];
        // In-plane block: e = d2 - R d1 with R the planar transport.
        let stiff_in = invert_3x3(&self.flexibility_inplane());
        let b_in = [
            -1.0, 0.0, dy, 1.0, 0.0, 0.0, 0.0, -1.0, -dx, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            1.0,
        ];
        let k_in = Self::expand_block(&stiff_in, &b_in);
        let plane = [0_usize, 1, 5, 6, 7, 11];
        for i in 0..6 {
            for j in 0..6 {
                k[plane[i] * 12 + plane[j]] = k_in[i * 6 + j];
            }
        }
        // Out-of-plane block: uz2 = uz1 + rx1 dy - ry1 dx, rotations carry.
        let stiff_out = invert_3x3(&self.flexibility_out_of_plane());
        let b_out = [
            -1.0, -dy, dx, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            1.0,
        ];
        let k_out = Self::expand_block(&stiff_out, &b_out);
        let out_of_plane = [2_usize, 3, 4, 8, 9, 10];
        for i in 0..6 {
            for j in 0..6 {
                k[out_of_plane[i] * 12 + out_of_plane[j]] = k_out[i * 6 + j];
            }
        }
        // Roundoff symmetrization: the B' K B products are symmetric to
        // ~1e-16, and assemblies amplify any non-passive remainder.
        for i in 0..12 {
            for j in i + 1..12 {
                let avg = 0.5 * (k[i * 12 + j] + k[j * 12 + i]);
                k[i * 12 + j] = avg;
                k[j * 12 + i] = avg;
            }
        }
        k
    }

    /// Global stiffness matrix (12x12, row-major).
    #[must_use]
    pub fn stiffness_global(&self) -> [f64; 144] {
        transform_12x12(&self.axes, &self.stiffness_local())
    }

    /// Global DOF indices with 6 DOFs per node.
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

/// Invert a 3x3 row-major matrix with partial pivoting.
///
/// # Panics
///
/// Panics for a singular matrix, which signals a degenerate element.
fn invert_3x3(a: &[f64; 9]) -> [f64; 9] {
    let mut work = [0.0; 18];
    for i in 0..3 {
        for j in 0..3 {
            work[i * 6 + j] = a[i * 3 + j];
        }
        work[i * 6 + 3 + i] = 1.0;
    }
    for col in 0..3 {
        let mut pivot = col;
        let mut best = work[col * 6 + col].abs();
        for row in col + 1..3 {
            let candidate = work[row * 6 + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "singular 3x3 flexibility matrix");
        if pivot != col {
            for k in 0..6 {
                work.swap(col * 6 + k, pivot * 6 + k);
            }
        }
        let diag = work[col * 6 + col];
        for k in 0..6 {
            work[col * 6 + k] /= diag;
        }
        for row in 0..3 {
            if row != col {
                let factor = work[row * 6 + col];
                for k in 0..6 {
                    work[row * 6 + k] -= factor * work[col * 6 + k];
                }
            }
        }
    }
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = work[i * 6 + 3 + j];
        }
    }
    out
}

/// Assemble curved-beam stiffness into a global triplet matrix.
pub fn assemble_curved_beam3d_stiffness(
    elements: &[CurvedBeamElement3D],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::beam::BeamElement3D;

    fn test_section() -> Section3D {
        Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false)
    }

    fn symmetry_error(k: &[f64; 144]) -> f64 {
        let mut worst: f64 = 0.0;
        for i in 0..12 {
            for j in 0..12 {
                worst = worst.max((k[i * 12 + j] - k[j * 12 + i]).abs());
            }
        }
        let scale = k.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        worst / scale
    }

    #[test]
    fn shallow_arch_recovers_straight_stiffness() {
        // Radius 1000 with unit chord: curvature corrections are O(1e-6).
        let radius = 1000.0;
        let half_angle = 0.0005_f64.asin();
        let center = [0.0, -radius, 0.0];
        let p1 = [
            radius * (-half_angle).sin(),
            radius * (-half_angle).cos() - radius,
            0.0,
        ];
        let p2 = [
            radius * half_angle.sin(),
            radius * half_angle.cos() - radius,
            0.0,
        ];
        let section = test_section();
        let curved = CurvedBeamElement3D::from_arc([0, 1], p1, p2, center, section);
        let chord = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        let straight = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [chord, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        );
        let kc = curved.stiffness_local();
        let ks = straight.stiffness_local();
        // Compare the in-plane block entry by entry against the straight
        // scale. Diagonal entries match to curvature-squared; the
        // axial-bending coupling entries vanish only linearly with the
        // 1e-3 curvature, so they carry a looser bound.
        let plane = [0_usize, 1, 5, 6, 7, 11];
        let scale = ks.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        for &i in &plane {
            for &j in &plane {
                let tol = if i == j { 1.0e-5 } else { 2.0e-4 };
                let diff = (kc[i * 12 + j] - ks[i * 12 + j]).abs() / scale;
                assert!(diff < tol, "entry ({i},{j}) differs by {diff:.3e}");
            }
        }
    }

    #[test]
    fn stiffness_is_symmetric_with_six_rigid_modes() {
        let radius = 2.0;
        let center = [0.0, 0.0, 0.0];
        let p1 = [radius, 0.0, 0.0];
        let p2 = [radius * 0.5, radius * 0.866_025_403_784_438_6, 0.0];
        let elem = CurvedBeamElement3D::from_arc([0, 1], p1, p2, center, test_section());
        let k = elem.stiffness_global();
        assert!(symmetry_error(&k) < 1.0e-12);
        // Rigid modes about the global origin in global coordinates.
        let mut modes = [[0.0; 12]; 6];
        for mode in 0..3 {
            modes[mode][mode] = 1.0;
            modes[mode][6 + mode] = 1.0;
        }
        for (mode, axis) in (3..6).zip([0, 1, 2]) {
            for (node, p) in [p1, p2].iter().enumerate() {
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
        let scale = k.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        for mode in &modes {
            let mut energy = 0.0;
            for i in 0..12 {
                let mut row = 0.0;
                for j in 0..12 {
                    row += k[i * 12 + j] * mode[j];
                }
                energy += mode[i] * row;
            }
            assert!(energy.abs() / scale < 1.0e-9, "rigid mode energy {energy}");
        }
    }

    #[test]
    fn single_quarter_element_matches_castigliano() {
        // Quarter-circular cantilever, tip tangential load: one exact element.
        let radius = 1.0;
        let center = [0.0, 0.0, 0.0];
        let p1 = [radius, 0.0, 0.0];
        let p2 = [0.0, radius, 0.0];
        let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
        let elem = CurvedBeamElement3D::from_arc([0, 1], p1, p2, center, section);
        // Clamp node 1, load node 2 along global -x (tip tangent direction).
        let k = elem.stiffness_global();
        let free: Vec<usize> = (6..12).collect();
        let mut matrix = [0.0; 36];
        for (i, &gi) in free.iter().enumerate() {
            for (j, &gj) in free.iter().enumerate() {
                matrix[i * 6 + j] = k[gi * 12 + gj];
            }
        }
        let mut rhs = [0.0; 6];
        rhs[0] = -1000.0;
        let u = solve_6(&matrix, &rhs);
        // Castigliano: bending PR^3(3pi/4 - 2)/EI plus axial PR pi/(4 EA).
        let bend =
            1000.0 * radius.powi(3) / section.eiz() * (3.0 * std::f64::consts::PI / 4.0 - 2.0);
        let axial = 1000.0 * radius * std::f64::consts::PI / (4.0 * section.ea());
        let expected = bend + axial;
        let error = (u[0].abs() - expected).abs() / expected;
        assert!(error < 1.0e-9, "tip error {error:.3e}");
    }

    /// Dense 6x6 solve with partial pivoting for unit tests.
    fn solve_6(a: &[f64; 36], b: &[f64; 6]) -> [f64; 6] {
        let mut m = *a;
        let mut x = *b;
        for col in 0..6 {
            let mut pivot = col;
            let mut best = m[col * 6 + col].abs();
            for row in col + 1..6 {
                let candidate = m[row * 6 + col].abs();
                if candidate > best {
                    best = candidate;
                    pivot = row;
                }
            }
            assert!(best > 1.0e-300, "singular test system");
            if pivot != col {
                for k in 0..6 {
                    m.swap(col * 6 + k, pivot * 6 + k);
                }
                x.swap(col, pivot);
            }
            let diag = m[col * 6 + col];
            for row in col + 1..6 {
                let factor = m[row * 6 + col] / diag;
                for k in col..6 {
                    m[row * 6 + k] -= factor * m[col * 6 + k];
                }
                x[row] -= factor * x[col];
            }
        }
        for row in (0..6).rev() {
            let mut sum = x[row];
            for k in row + 1..6 {
                sum -= m[row * 6 + k] * x[k];
            }
            x[row] = sum / m[row * 6 + row];
        }
        x
    }
}
