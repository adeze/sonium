//! Axisymmetric shells of revolution (meridional formulation).
//!
//! Two-node conical frustum element on the meridian with 3 DOFs per node
//! (`u_r, u_z, beta`): radial and axial displacement plus the meridional
//! rotation about the circumferential axis. Membrane, hoop, bending, and
//! transverse shear terms are integrated with 2*pi*r weighting; shear uses
//! single-point selective reduced integration to avoid locking.
//!
//! The hoop strain-displacement rows reuse the shared building blocks in
//! [`axisymmetric`](crate::physics::solid::axisymmetric). Closed-form
//! membrane solutions for spheres and cylinders are provided for QA.
//!
//! Reference: Zienkiewicz, Taylor, and Zhu, "The Finite Element Method",
//! 7th ed., Chapter on axisymmetric shells; Timoshenko and Woinowsky-Krieger,
//! "Theory of Plates and Shells", membrane theory of shells of revolution.

// Rust guideline compliant 2026-02-21

use super::section::ShellSection;
use crate::physics::solid::axisymmetric::{
    axisymmetric_shell_hoop_bending_factor, axisymmetric_shell_hoop_membrane_factors,
    meridional_circumference,
};

/// Transverse shear correction factor.
const SHEAR_CORRECTION: f64 = 5.0 / 6.0;

/// 2-point Gauss rule on [-1, 1].
const GAUSS_2: [(f64, f64); 2] = {
    let a = 0.577_350_269_189_625_8;
    [(-a, 1.0), (a, 1.0)]
};

/// Two-node conical shell-of-revolution element.
///
/// Nodes carry global `(u_r, u_z, beta)`. Local `(u_s, w)` are the
/// meridional and outward-normal displacements.
#[derive(Debug, Clone)]
pub struct RevolutionShellElement {
    /// Node indices along the meridian, ordered with the shell interior
    /// on the left of the traversal direction (outward = clockwise
    /// normal): bottom to top for cylinders, equator to pole for spheres.
    pub nodes: [usize; 2],
    /// Nodal radii in m.
    pub r: [f64; 2],
    /// Nodal axial coordinates in m.
    pub z: [f64; 2],
    /// Shell section properties.
    pub section: ShellSection,
}

impl RevolutionShellElement {
    /// Create from nodal coordinates and section.
    pub fn new(nodes: [usize; 2], r: [f64; 2], z: [f64; 2], section: ShellSection) -> Self {
        assert!(r[0] > 0.0 && r[1] > 0.0, "meridian must stay off the axis");
        Self {
            nodes,
            r,
            z,
            section,
        }
    }

    /// Slant length in m.
    pub fn slant_length(&self) -> f64 {
        let dr = self.r[1] - self.r[0];
        let dz = self.z[1] - self.z[0];
        (dr * dr + dz * dz).sqrt()
    }

    /// Meridional tangent (cos_phi, sin_phi) in (r, z).
    pub fn tangent(&self) -> (f64, f64) {
        let length = self.slant_length();
        (
            (self.r[1] - self.r[0]) / length,
            (self.z[1] - self.z[0]) / length,
        )
    }

    /// Radius at natural coordinate xi.
    fn radius_at(&self, xi: f64) -> f64 {
        0.5 * (1.0 - xi) * self.r[0] + 0.5 * (1.0 + xi) * self.r[1]
    }

    /// 6x6 local stiffness over [us1, w1, b1, us2, w2, b2].
    pub fn stiffness_local(&self) -> [f64; 36] {
        let length = self.slant_length();
        let det_j = length / 2.0;
        let (cos_phi, sin_phi) = self.tangent();
        let e = self.section.youngs_modulus;
        let nu = self.section.poisson_ratio;
        let t = self.section.thickness;
        let dm = e * t / (1.0 - nu * nu);
        let db = e * t.powi(3) / (12.0 * (1.0 - nu * nu));
        let ds = SHEAR_CORRECTION * e / (2.0 * (1.0 + nu)) * t;
        // Shape derivatives dN/ds are constant for the linear meridian.
        let dnds = [-1.0 / length, 1.0 / length];

        let mut k = [0.0; 36];
        for &(xi, w) in &GAUSS_2 {
            let n = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let r = self.radius_at(xi).max(1.0e-12);
            let weight = w * det_j * meridional_circumference(r);
            // Membrane rows over [us1, us2], [w1, w2].
            let hoop = axisymmetric_shell_hoop_membrane_factors(cos_phi, sin_phi, r);
            // Bm rows: [eps_s, eps_theta] x [us1, w1, b1, us2, w2, b2].
            let mut bm = [0.0; 12];
            bm[0] = dnds[0];
            bm[3] = dnds[1];
            bm[6] = hoop[0] * n[0];
            bm[7] = hoop[1] * n[0];
            bm[9] = hoop[0] * n[1];
            bm[10] = hoop[1] * n[1];
            // Bb rows: [kap_s, kap_theta] with hoop factor * N for beta.
            let bend_hoop = axisymmetric_shell_hoop_bending_factor(cos_phi, r);
            let mut bb = [0.0; 12];
            bb[2] = dnds[0];
            bb[5] = dnds[1];
            bb[8] = bend_hoop * n[0];
            bb[11] = bend_hoop * n[1];
            add_2x6_block(&mut k, &bm, dm, dm * nu, weight);
            add_2x6_block(&mut k, &bb, db, db * nu, weight);
        }
        // Transverse shear at the midpoint (selective reduced integration).
        {
            let n = [0.5, 0.5];
            let r = self.radius_at(0.0).max(1.0e-12);
            let weight = 2.0 * det_j * meridional_circumference(r);
            let mut bs = [0.0; 6];
            bs[1] = dnds[0];
            bs[2] = -n[0];
            bs[4] = dnds[1];
            bs[5] = -n[1];
            for i in 0..6 {
                for j in 0..6 {
                    k[i * 6 + j] += weight * ds * bs[i] * bs[j];
                }
            }
        }
        k
    }

    /// 6x6 global stiffness over [ur1, uz1, b1, ur2, uz2, b2].
    pub fn stiffness_global(&self) -> [f64; 36] {
        let k_local = self.stiffness_local();
        let (cos_phi, sin_phi) = self.tangent();
        // Global -> local rotation per node (orthogonal).
        let mut t = [0.0; 36];
        for node in 0..2 {
            let o = node * 3;
            t[(o) * 6 + o] = cos_phi;
            t[(o) * 6 + o + 1] = sin_phi;
            t[(o + 1) * 6 + o] = sin_phi;
            t[(o + 1) * 6 + o + 1] = -cos_phi;
            t[(o + 2) * 6 + o + 2] = 1.0;
        }
        // K_g = T^T K_l T.
        let mut kt = [0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut val = 0.0;
                for p in 0..6 {
                    val += k_local[i * 6 + p] * t[p * 6 + j];
                }
                kt[i * 6 + j] = val;
            }
        }
        let mut kg = [0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut val = 0.0;
                for p in 0..6 {
                    val += t[p * 6 + i] * kt[p * 6 + j];
                }
                kg[i * 6 + j] = val;
            }
        }
        kg
    }

    /// Consistent pressure load in global DOFs (outward normal, Pa).
    pub fn pressure_load(&self, pressure: f64) -> [f64; 6] {
        let length = self.slant_length();
        let det_j = length / 2.0;
        let (cos_phi, sin_phi) = self.tangent();
        let mut local = [0.0; 6];
        for &(xi, w) in &GAUSS_2 {
            let n = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let r = self.radius_at(xi).max(1.0e-12);
            let weight = w * det_j * meridional_circumference(r) * pressure;
            local[1] += weight * n[0];
            local[4] += weight * n[1];
        }
        // Local (us, w) -> global (ur, uz): [ur; uz] = R^T [us; w].
        let mut global = [0.0; 6];
        for node in 0..2 {
            let us = local[node * 3];
            let w = local[node * 3 + 1];
            global[node * 3] = cos_phi * us + sin_phi * w;
            global[node * 3 + 1] = sin_phi * us - cos_phi * w;
            global[node * 3 + 2] = local[node * 3 + 2];
        }
        global
    }

    /// Lumped mass in global DOFs (translational + rotary diagonal).
    pub fn mass_diagonal(&self) -> [f64; 6] {
        let slant = self.slant_length();
        let r_mid = 0.5 * (self.r[0] + self.r[1]);
        let area = meridional_circumference(r_mid) * slant;
        let nodal = self.section.mass_per_area() * area / 2.0;
        let rotary = nodal * self.section.thickness.powi(2) / 12.0;
        [nodal, nodal, rotary, nodal, nodal, rotary]
    }

    /// Global DOF indices: node i -> [3*node+0..3*node+2].
    pub fn global_dofs(&self) -> [usize; 6] {
        let mut dofs = [0; 6];
        for (i, &node) in self.nodes.iter().enumerate() {
            for d in 0..3 {
                dofs[i * 3 + d] = node * 3 + d;
            }
        }
        dofs
    }
}

/// Add `weight * B^T D B` for a 2-row B with D = [[d, dnu], [dnu, d]].
fn add_2x6_block(k: &mut [f64; 36], b: &[f64; 12], d: f64, dnu: f64, weight: f64) {
    for i in 0..6 {
        for j in 0..6 {
            let val =
                d * (b[i] * b[j] + b[6 + i] * b[6 + j]) + dnu * (b[i] * b[6 + j] + b[6 + i] * b[j]);
            k[i * 6 + j] += weight * val;
        }
    }
}

/// Assemble revolution elements into a global stiffness matrix.
pub fn assemble_revolution_stiffness(
    elements: &[RevolutionShellElement],
    n_dofs: usize,
) -> crate::assembly::TripletMatrix<f64> {
    let mut triplet = crate::assembly::TripletMatrix::new(n_dofs, n_dofs);
    for elem in elements {
        let k = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let val = k[i * 6 + j];
                if val.abs() > 1e-15 {
                    triplet.push(gi, gj, val);
                }
            }
        }
    }
    triplet
}

/// Assemble uniform pressure on revolution elements (outward normal).
pub fn assemble_revolution_pressure(
    elements: &[RevolutionShellElement],
    n_dofs: usize,
    pressure: f64,
) -> Vec<f64> {
    let mut rhs = vec![0.0; n_dofs];
    for elem in elements {
        let f = elem.pressure_load(pressure);
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            rhs[gi] += f[i];
        }
    }
    rhs
}

/// Spherical pressure-vessel membrane stress pr/2t in Pa.
pub fn sphere_membrane_stress(pressure: f64, radius: f64, thickness: f64) -> f64 {
    pressure * radius / (2.0 * thickness)
}

/// Spherical pressure-vessel radial expansion in m.
pub fn sphere_membrane_displacement(
    pressure: f64,
    radius: f64,
    thickness: f64,
    youngs_modulus: f64,
    poisson_ratio: f64,
) -> f64 {
    pressure * radius * radius * (1.0 - poisson_ratio) / (2.0 * youngs_modulus * thickness)
}

/// Cylindrical pressure-vessel hoop stress pr/t in Pa.
pub fn cylinder_hoop_stress(pressure: f64, radius: f64, thickness: f64) -> f64 {
    pressure * radius / thickness
}

/// Cylindrical pressure-vessel radial expansion in m (open ends, free).
pub fn cylinder_membrane_displacement(
    pressure: f64,
    radius: f64,
    thickness: f64,
    youngs_modulus: f64,
) -> f64 {
    pressure * radius * radius / (youngs_modulus * thickness)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cylinder_element() -> RevolutionShellElement {
        RevolutionShellElement::new(
            [0, 1],
            [1.0, 1.0],
            [0.0, 0.5],
            ShellSection::new(200e9, 0.3, 0.01, 7800.0),
        )
    }

    #[test]
    fn revolution_stiffness_is_symmetric() {
        let elem = RevolutionShellElement::new(
            [0, 1],
            [1.0, 1.2],
            [0.0, 0.5],
            ShellSection::new(200e9, 0.3, 0.01, 7800.0),
        );
        let k = elem.stiffness_global();
        for i in 0..6 {
            for j in (i + 1)..6 {
                let diff = (k[i * 6 + j] - k[j * 6 + i]).abs();
                let scale = k[i * 6 + i].abs().max(1.0);
                assert!(diff / scale < 1e-12, "asymmetry at ({i},{j})");
            }
        }
    }

    #[test]
    fn cylinder_hoop_row_matches_analytical() {
        // Uniform radial expansion u_r = 1e-3 on a cylinder: hoop strain is
        // u_r / r everywhere, meridional strain is zero.
        let elem = cylinder_element();
        let (cos_phi, sin_phi) = elem.tangent();
        assert!(cos_phi.abs() < 1e-12);
        assert!((sin_phi - 1.0).abs() < 1e-12);
        let hoop = axisymmetric_shell_hoop_membrane_factors(cos_phi, sin_phi, 1.0);
        // w = u_r on a cylinder, so the w factor must be 1/r.
        assert!((hoop[1] - 1.0).abs() < 1e-12);
        assert!(hoop[0].abs() < 1e-12);
        let k = elem.stiffness_local();
        // Axial rigid translation: us = 1 at both nodes costs no energy.
        let d = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut energy = 0.0;
        for i in 0..6 {
            let mut row = 0.0;
            for j in 0..6 {
                row += k[i * 6 + j] * d[j];
            }
            energy += d[i] * row;
        }
        let norm = k.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(energy / norm < 1e-12, "axial rigid energy {energy:e}");
    }

    #[test]
    fn closed_forms_scale_correctly() {
        let s = sphere_membrane_stress(1e6, 2.0, 0.02);
        assert!((s - 50e6).abs() < 1.0);
        let u = sphere_membrane_displacement(1e6, 2.0, 0.02, 200e9, 0.3);
        assert!(u > 0.0 && u < 1e-2);
        // Doubling pressure doubles displacement; doubling E halves it.
        let u2 = sphere_membrane_displacement(2e6, 2.0, 0.02, 200e9, 0.3);
        assert!((u2 - 2.0 * u).abs() / u < 1e-12);
        let hoop = cylinder_hoop_stress(1e6, 1.0, 0.01);
        assert!((hoop - 100e6).abs() < 1.0);
    }

    #[test]
    fn pressure_load_points_outward() {
        let elem = cylinder_element();
        let f = elem.pressure_load(1000.0);
        // Cylinder: outward normal is +r, symmetric halves.
        assert!(f[0] > 0.0 && f[3] > 0.0);
        assert!((f[0] - f[3]).abs() / f[0] < 1e-12);
        assert!(f[1].abs() / f[0] < 1e-12);
        assert!(f[2].abs() < 1e-9 && f[5].abs() < 1e-9);
    }
}
