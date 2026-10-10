//! Tapered 3-D beam element with interpolated section.
//!
//! A 2-node, 12-DOF straight beam whose section varies along the axis.
//! [`TaperedBeamElement3D::section_at`] interpolates area, inertias, torsion
//! constant, and shear factors linearly between the end sections; stiffness
//! evaluation integrates the beam strain energy with Gauss quadrature over
//! that interpolated section. Bending uses Euler-Bernoulli Hermite cubics
//! or Timoshenko cubics (derived at runtime from the homogeneous solution,
//! exact for the midpoint shear parameter), axial and torsion use linear
//! interpolation. Quadrature is exact for the linearly interpolated
//! section, so a prismatic element (equal end sections) reproduces
//! [`BeamElement3D`](super::element3d::BeamElement3D) to machine precision,
//! and taper accuracy improves quadratically with mesh refinement.

// Rust guideline compliant 2026-02-21

use super::element3d::transform_12x12;
use super::section3d::Section3D;

/// Tapered space-frame beam element with 12 local DOFs.
///
/// Local axes, node order, and DOF layout match [`BeamElement3D`](super::element3d::BeamElement3D).
/// Both end sections must share Young's modulus, shear modulus, density, and
/// the Euler-Bernoulli versus Timoshenko regime in each bending plane; only
/// geometric properties vary.
#[derive(Debug, Clone)]
pub struct TaperedBeamElement3D {
    /// Node indices (2 nodes).
    pub nodes: [usize; 2],
    /// First endpoint in global coordinates.
    pub p1: [f64; 3],
    /// Second endpoint in global coordinates.
    pub p2: [f64; 3],
    /// Element length in m.
    pub length: f64,
    /// Local axes in global coordinates, row-major (3 local axis rows).
    pub axes: [f64; 9],
    /// Cross-section at node 1.
    pub section1: Section3D,
    /// Cross-section at node 2.
    pub section2: Section3D,
}

impl TaperedBeamElement3D {
    /// Build a tapered element between global points `p1` and `p2`.
    ///
    /// The `up` hint orients the section exactly as in
    /// [`BeamElement3D::new`](super::element3d::BeamElement3D::new).
    ///
    /// # Panics
    ///
    /// Panics for coincident nodes, an `up` hint parallel to the beam axis,
    /// or end sections with mismatched material or bending regime.
    #[must_use]
    pub fn new(
        nodes: [usize; 2],
        p1: [f64; 3],
        p2: [f64; 3],
        section1: Section3D,
        section2: Section3D,
        up: [f64; 3],
    ) -> Self {
        assert_material_match(&section1, &section2);
        let axis = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        assert!(length > 0.0, "beam element needs distinct nodes");
        let ex = [axis[0] / length, axis[1] / length, axis[2] / length];
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
        let ez = [
            ex[1] * ey[2] - ex[2] * ey[1],
            ex[2] * ey[0] - ex[0] * ey[2],
            ex[0] * ey[1] - ex[1] * ey[0],
        ];
        Self {
            nodes,
            p1,
            p2,
            length,
            axes: [
                ex[0], ex[1], ex[2], ey[0], ey[1], ey[2], ez[0], ez[1], ez[2],
            ],
            section1,
            section2,
        }
    }

    /// Linearly interpolated section at axial station `xi` in `[0, 1]`.
    ///
    /// `xi = 0` returns the node-1 section and `xi = 1` the node-2 section.
    /// Area, both inertias, torsion constant, and shear factors interpolate
    /// linearly; material moduli stay constant by construction.
    #[must_use]
    pub fn section_at(&self, xi: f64) -> Section3D {
        let a = &self.section1;
        let b = &self.section2;
        let lerp = |x: f64, y: f64| x + (y - x) * xi;
        Section3D {
            youngs_modulus: a.youngs_modulus,
            shear_modulus: a.shear_modulus,
            area: lerp(a.area, b.area),
            iy: lerp(a.iy, b.iy),
            iz: lerp(a.iz, b.iz),
            torsion_constant: lerp(a.torsion_constant, b.torsion_constant),
            warping_constant: lerp(a.warping_constant, b.warping_constant),
            kappa_y: lerp(a.kappa_y, b.kappa_y),
            kappa_z: lerp(a.kappa_z, b.kappa_z),
            density: a.density,
        }
    }

    /// Local stiffness matrix (12x12, row-major).
    ///
    /// Integrates axial, torsional, bending, and shear strain energies with
    /// Gauss quadrature over the interpolated section. Two-point quadrature
    /// is exact for the linear axial/torsion resultants times the linear
    /// section, and for the bending curvature (linear in `x`) squared times
    /// the linear bending stiffness.
    #[must_use]
    pub fn stiffness_local(&self) -> [f64; 144] {
        let l = self.length;
        let mut k = [0.0; 144];
        // Axial block on ux DOFs 0 and 6; torsion on rx DOFs 3 and 9.
        for (xi, w) in GAUSS2_XI.iter().zip(GAUSS2_W.iter()) {
            let section = self.section_at(*xi);
            let f = w * l / (l * l);
            let ea = section.ea() * f;
            k[0] += ea;
            k[6] -= ea;
            k[6 * 12] -= ea;
            k[6 * 12 + 6] += ea;
            let gj = section.gj() * f;
            k[3 * 12 + 3] += gj;
            k[3 * 12 + 9] -= gj;
            k[9 * 12 + 3] -= gj;
            k[9 * 12 + 9] += gj;
        }
        self.add_bending_plane(&mut k, false);
        self.add_bending_plane(&mut k, true);
        // Roundoff symmetrization: quadrature sums are symmetric to ~1e-16,
        // and long assemblies amplify any non-passive remainder.
        for i in 0..12 {
            for j in i + 1..12 {
                let avg = 0.5 * (k[i * 12 + j] + k[j * 12 + i]);
                k[i * 12 + j] = avg;
                k[j * 12 + i] = avg;
            }
        }
        k
    }

    /// Add one bending plane to the local stiffness matrix.
    ///
    /// `about_y` selects the `x-z` plane (DOFs `[uz1, ry1, uz2, ry2]`,
    /// `ry = -dw/dx`) over the `x-y` plane (DOFs `[uy1, rz1, uy2, rz2]`).
    /// Euler-Bernoulli planes use Hermite cubics; Timoshenko planes use the
    /// exact homogeneous cubics for the midpoint shear parameter with the
    /// constant shear strain integrated at the midpoint.
    fn add_bending_plane(&self, k: &mut [f64; 144], about_y: bool) {
        let l = self.length;
        let dofs = if about_y {
            [2_usize, 4, 8, 10]
        } else {
            [1_usize, 5, 7, 11]
        };
        let sign = if about_y {
            [1.0, -1.0, 1.0, -1.0]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        };
        let mid = self.section_at(0.5);
        let phi_mid = if about_y {
            mid.phi_y(l)
        } else {
            mid.phi_z(l)
        };
        // Cubic coefficients [a, b, c, d] in x per unit nodal value.
        let shapes = if phi_mid > 0.0 {
            timoshenko_shapes(l, phi_mid)
        } else {
            hermite_shapes(l)
        };
        // Curvature rows Bk_j(x) = 2 c_j + 6 d_j x.
        for (xi, w) in GAUSS2_XI.iter().zip(GAUSS2_W.iter()) {
            let x = xi * l;
            let section = self.section_at(*xi);
            let ei = if about_y {
                section.eiy()
            } else {
                section.eiz()
            };
            let f = w * l * ei;
            let mut bk = [0.0; 4];
            for j in 0..4 {
                bk[j] = 2.0 * shapes[j][2] + 6.0 * shapes[j][3] * x;
            }
            for i in 0..4 {
                for j in 0..4 {
                    k[dofs[i] * 12 + dofs[j]] += f * sign[i] * bk[i] * sign[j] * bk[j];
                }
            }
        }
        if phi_mid > 0.0 {
            // Constant shear strain of magnitude |d_j| phi L^2 / 2; only the
            // square enters the energy, so the moment-convention sign drops.
            let ga = if about_y {
                mid.shear_stiffness_z()
            } else {
                mid.shear_stiffness_y()
            };
            // Shear uses the midpoint stiffness: the 1-point rule is exact
            // for the constant strain times the linear shear stiffness only
            // at the midpoint value, which is the standard tapered
            // Timoshenko treatment (exact for prismatic members).
            let f = l * ga;
            let mut bs = [0.0; 4];
            for j in 0..4 {
                bs[j] = shapes[j][3] * phi_mid * l * l / 2.0;
            }
            for i in 0..4 {
                for j in 0..4 {
                    k[dofs[i] * 12 + dofs[j]] += f * sign[i] * bs[i] * sign[j] * bs[j];
                }
            }
        }
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

/// Check that two end sections share material and bending regime.
///
/// # Panics
///
/// Panics when moduli, density, or the Euler-Bernoulli versus Timoshenko
/// regime differ between the sections.
fn assert_material_match(a: &Section3D, b: &Section3D) {
    for (name, x, y) in [
        ("Young's modulus", a.youngs_modulus, b.youngs_modulus),
        ("shear modulus", a.shear_modulus, b.shear_modulus),
        ("density", a.density, b.density),
    ] {
        assert!(
            (x - y).abs() <= 1.0e-12 * x.abs().max(1.0e-300),
            "tapered element needs matching {name} at both ends"
        );
    }
    assert!(
        (a.kappa_y > 0.0) == (b.kappa_y > 0.0),
        "tapered element needs matching y-plane bending regime at both ends"
    );
    assert!(
        (a.kappa_z > 0.0) == (b.kappa_z > 0.0),
        "tapered element needs matching z-plane bending regime at both ends"
    );
}

/// 2-point Gauss-Legendre nodes and weights on `[0, 1]`.
///
/// Exact through degree 3, which covers the linear-axial and linear-curvature
/// integrands times the linearly interpolated section.
const GAUSS2_XI: [f64; 2] = [0.211_324_865_405_187_1, 0.788_675_134_594_812_9];
const GAUSS2_W: [f64; 2] = [0.5, 0.5];

/// Hermite cubic coefficients `[a, b, c, d]` in `x` per unit nodal value.
///
/// Nodal order is `[w1, t1, w2, t2]` with `t = dw/dx`; the standard forms
/// in `xi = x/L` convert directly (`H1 = 1 - 3 xi^2 + 2 xi^3`, and so on).
fn hermite_shapes(l: f64) -> [[f64; 4]; 4] {
    let l2 = l * l;
    let l3 = l2 * l;
    [
        [1.0, 0.0, -3.0 / l2, 2.0 / l3],
        [0.0, 1.0, -2.0 / l, 1.0 / l2],
        [0.0, 0.0, 3.0 / l2, -2.0 / l3],
        [0.0, 0.0, -1.0 / l, 1.0 / l2],
    ]
}

/// Timoshenko cubic coefficients from the homogeneous solution.
///
/// Each unit nodal value `[w1, t1, w2, t2]` determines a cubic
/// `w(x) = a + b x + c x^2 + d x^3` with rotation
/// `theta(x) = w'(x) + 6 d EI/(GA kappa)` and
/// `6 EI/(GA kappa) = phi L^2/2`, solved at runtime so no transcribed shape
/// table can drift. The shear parameter `phi` is the midpoint value of the
/// tapered element. The plus sign follows the element moment convention
/// (`V = -EI w'''`); the minus sign would produce `(1 - phi)` stiffness
/// harder than Euler-Bernoulli, which is unphysical.
fn timoshenko_shapes(l: f64, phi: f64) -> [[f64; 4]; 4] {
    let l2 = l * l;
    let l3 = l2 * l;
    let shear = phi * l2 / 2.0;
    // Nodal conditions [w(0), t(0), w(L), t(L)] on [a, b, c, d].
    let a = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, shear, 1.0, l, l2, l3, 0.0, 1.0,
        2.0 * l, 3.0 * l2 + shear,
    ];
    let mut shapes = [[0.0; 4]; 4];
    for (j, shape) in shapes.iter_mut().enumerate() {
        let mut rhs = [0.0; 4];
        rhs[j] = 1.0;
        *shape = solve_4(&a, &rhs);
    }
    shapes
}

/// Solve a dense 4x4 system with partial pivoting.
///
/// # Panics
///
/// Panics for a singular matrix.
fn solve_4(a: &[f64; 16], b: &[f64; 4]) -> [f64; 4] {
    let mut m = *a;
    let mut x = *b;
    for col in 0..4 {
        let mut pivot = col;
        let mut best = m[col * 4 + col].abs();
        for row in col + 1..4 {
            let candidate = m[row * 4 + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "singular shape system");
        if pivot != col {
            for k in 0..4 {
                m.swap(col * 4 + k, pivot * 4 + k);
            }
            x.swap(col, pivot);
        }
        let diag = m[col * 4 + col];
        for k in 0..4 {
            m[col * 4 + k] /= diag;
        }
        x[col] /= diag;
        for row in 0..4 {
            if row != col {
                let factor = m[row * 4 + col];
                for k in 0..4 {
                    m[row * 4 + k] -= factor * m[col * 4 + k];
                }
                x[row] -= factor * x[col];
            }
        }
    }
    x
}

/// Assemble tapered-beam stiffness into a global triplet matrix.
pub fn assemble_tapered_beam3d_stiffness(
    elements: &[TaperedBeamElement3D],
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

    fn root_section() -> Section3D {
        Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false)
    }

    fn tip_section() -> Section3D {
        Section3D::rectangular(200.0e9, 0.3, 0.05, 0.025, 7800.0, false)
    }

    fn symmetry_error(k: &[f64; 144]) -> f64 {
        let mut worst: f64 = 0.0;
        for i in 0..12 {
            for j in 0..12 {
                worst = worst.max((k[i * 12 + j] - k[j * 12 + i]).abs());
            }
        }
        worst / k.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
    }

    #[test]
    fn prismatic_taper_matches_straight_element() {
        for timoshenko in [false, true] {
            let section =
                Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, timoshenko);
            let tapered = TaperedBeamElement3D::new(
                [0, 1],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                section,
                section,
                [0.0, 1.0, 0.0],
            );
            let straight = BeamElement3D::new(
                [0, 1],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                section,
                [0.0, 1.0, 0.0],
            );
            let kt = tapered.stiffness_local();
            let ks = straight.stiffness_local();
            let scale = ks.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            for i in 0..144 {
                assert!(
                    (kt[i] - ks[i]).abs() / scale < 1.0e-12,
                    "entry {i} differs (timoshenko={timoshenko})"
                );
            }
        }
    }

    #[test]
    fn section_interpolation_is_linear() {
        let elem = TaperedBeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            root_section(),
            tip_section(),
            [0.0, 1.0, 0.0],
        );
        let at0 = elem.section_at(0.0);
        let at1 = elem.section_at(1.0);
        assert!((at0.area - root_section().area).abs() < 1.0e-18);
        assert!((at1.area - tip_section().area).abs() < 1.0e-18);
        let mid = elem.section_at(0.5);
        let expected = 0.5 * (root_section().iz + tip_section().iz);
        assert!((mid.iz - expected).abs() / expected < 1.0e-15);
    }

    #[test]
    fn tapered_stiffness_is_symmetric_and_softer_than_root_prismatic() {
        let elem = TaperedBeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            root_section(),
            tip_section(),
            [0.0, 1.0, 0.0],
        );
        let k = elem.stiffness_local();
        assert!(symmetry_error(&k) < 1.0e-12);
        // Tapering toward a smaller tip softens transverse response below
        // the root-prismatic stiffness but stays above the tip-prismatic one.
        let root = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            root_section(),
            [0.0, 1.0, 0.0],
        )
        .stiffness_local();
        let tip = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            tip_section(),
            [0.0, 1.0, 0.0],
        )
        .stiffness_local();
        assert!(k[1 * 12 + 1] < root[1 * 12 + 1]);
        assert!(k[1 * 12 + 1] > tip[1 * 12 + 1]);
    }

    #[test]
    fn tapered_cantilever_matches_stepped_reference() {
        // Two prismatic spans modeled with degenerate tapered elements must
        // reproduce the stepped-beam closed form exactly.
        let l1 = 0.6;
        let l2 = 0.4;
        let e1 = TaperedBeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [l1, 0.0, 0.0],
            root_section(),
            root_section(),
            [0.0, 1.0, 0.0],
        );
        let e2 = TaperedBeamElement3D::new(
            [1, 2],
            [l1, 0.0, 0.0],
            [l1 + l2, 0.0, 0.0],
            tip_section(),
            tip_section(),
            [0.0, 1.0, 0.0],
        );
        let n = 18;
        let mut k = vec![0.0; n * n];
        for elem in [&e1, &e2] {
            let ke = elem.stiffness_global();
            let dofs = elem.global_dofs();
            for (i, &gi) in dofs.iter().enumerate() {
                for (j, &gj) in dofs.iter().enumerate() {
                    k[gi * n + gj] += ke[i * 12 + j];
                }
            }
        }
        // Clamp node 0, keep the free 12x12 block.
        let m = 12;
        let mut matrix = vec![0.0; m * m];
        let mut rhs = vec![0.0; m];
        for i in 0..m {
            for j in 0..m {
                matrix[i * m + j] = k[(i + 6) * n + (j + 6)];
            }
        }
        rhs[6 + 1] = 1000.0;
        let tip = solve_n(&matrix, &rhs, m);
        let p = 1000.0;
        let i1 = root_section().iz;
        let i2 = tip_section().iz;
        let e = 200.0e9;
        // Junction moment M = P l2 bends and rotates the root span too.
        let expected = p * l2.powi(3) / (3.0 * e * i2) + p * l1.powi(3) / (3.0 * e * i1)
            + p * l1 * l1 * l2 / (e * i1)
            + p * l1 * l2 * l2 / (e * i1);
        let error = (tip[6 + 1] - expected).abs() / expected;
        assert!(error < 1.0e-9, "stepped tip error {error:.3e}");
    }

    /// Dense solve with partial pivoting for unit tests.
    fn solve_n(matrix: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
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
}
