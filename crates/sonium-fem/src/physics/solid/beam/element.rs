//! Beam element stiffness and mass matrices
//!
//! 2-node beam element with DOFs `[w1, theta1, w2, theta2]` (transverse displacement + rotation).
//! For frame analysis, axial DOFs `[u1, u2]` are added, giving 6 DOFs per element in 2D
//! or 12 in 3D.
//!
//! The bending stiffness uses Hermite cubic interpolation (Euler-Bernoulli) or
//! includes shear deformation (Timoshenko).

use super::section::BeamSection;

/// 2D beam element with bending + axial DOFs.
///
/// Local DOFs: `[u1, w1, theta1, u2, w2, theta2]` (6 per element, 3 per node).
/// - `u`: axial displacement (along beam axis)
/// - `w`: transverse displacement (perpendicular to beam)
/// - `theta`: rotation (= dw/dx for Euler-Bernoulli)
#[derive(Debug, Clone)]
pub struct BeamElement2D {
    /// Node indices (2 nodes).
    pub nodes: [usize; 2],
    /// Element length.
    pub length: f64,
    /// Direction cosines (cos, sin) of beam axis relative to global x-axis.
    pub cos_alpha: f64,
    pub sin_alpha: f64,
    /// Cross-section properties.
    pub section: BeamSection,
}

impl BeamElement2D {
    /// Create from two node positions and section properties.
    pub fn new(
        nodes: [usize; 2],
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        section: BeamSection,
    ) -> Self {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let length = (dx * dx + dy * dy).sqrt();
        Self {
            nodes,
            length,
            cos_alpha: dx / length,
            sin_alpha: dy / length,
            section,
        }
    }

    /// Local bending stiffness matrix (4x4) for `[w1, theta1, w2, theta2]`.
    ///
    /// Euler-Bernoulli: `K_b = (EI/L^3) * [[12, 6L, -12, 6L], [6L, 4L^2, -6L, 2L^2], ...]`
    pub fn bending_stiffness_local(&self) -> [f64; 16] {
        let ei = self.section.ei();
        let l = self.length;
        let l2 = l * l;
        let l3 = l2 * l;

        if self.section.is_timoshenko() {
            // Timoshenko beam: includes shear parameter phi = 12EI/(kappa*G*A*L^2)
            let kga =
                self.section.shear_correction * self.section.shear_modulus * self.section.area;
            let phi = 12.0 * ei / (kga * l2);
            let denom = (1.0 + phi) * l3;
            let a = 12.0 * ei / denom;
            let b = 6.0 * ei * l / denom;
            let c = (4.0 + phi) * ei * l2 / denom;
            let d = (2.0 - phi) * ei * l2 / denom;

            [a, b, -a, b, b, c, -b, d, -a, -b, a, -b, b, d, -b, c]
        } else {
            // Euler-Bernoulli
            let c = ei / l3;
            [
                12.0 * c,
                6.0 * c * l,
                -12.0 * c,
                6.0 * c * l,
                6.0 * c * l,
                4.0 * c * l2,
                -6.0 * c * l,
                2.0 * c * l2,
                -12.0 * c,
                -6.0 * c * l,
                12.0 * c,
                -6.0 * c * l,
                6.0 * c * l,
                2.0 * c * l2,
                -6.0 * c * l,
                4.0 * c * l2,
            ]
        }
    }

    /// Local axial stiffness matrix (2x2) for `[u1, u2]`.
    ///
    /// `K_a = (EA/L) * [[1, -1], [-1, 1]]`
    pub fn axial_stiffness_local(&self) -> [f64; 4] {
        let c = self.section.ea() / self.length;
        [c, -c, -c, c]
    }

    /// Full local stiffness matrix (6x6) for `[u1, w1, theta1, u2, w2, theta2]`.
    pub fn stiffness_local(&self) -> [f64; 36] {
        let k_a = self.axial_stiffness_local();
        let k_b = self.bending_stiffness_local();

        // Assemble into 6x6: axial DOFs at positions 0,3; bending at 1,2,4,5
        let mut k = [0.0; 36];
        // Axial block: rows/cols 0 and 3
        k[0] = k_a[0]; // [0,0]
        k[3] = k_a[1]; // [0,3]
        k[18] = k_a[2]; // [3,0]
        k[21] = k_a[3]; // [3,3]

        // Bending block: map [w1,t1,w2,t2] -> [1,2,4,5]
        let bending_dofs = [1, 2, 4, 5];
        for (i, &gi) in bending_dofs.iter().enumerate() {
            for (j, &gj) in bending_dofs.iter().enumerate() {
                k[gi * 6 + gj] = k_b[i * 4 + j];
            }
        }

        k
    }

    /// Transform local stiffness to global coordinates.
    ///
    /// `K_global = T^T K_local T` where T is the rotation matrix.
    pub fn stiffness_global(&self) -> [f64; 36] {
        let k_local = self.stiffness_local();
        let t = self.rotation_matrix();
        transform_6x6(&k_local, &t)
    }

    /// Consistent mass matrix (6x6 local) for `[u1, w1, theta1, u2, w2, theta2]`.
    pub fn mass_local(&self) -> [f64; 36] {
        let rho_a = self.section.mass_per_length();
        let l = self.length;
        let l2 = l * l;

        // Consistent mass for axial: (rhoA*L/6) * [[2,1],[1,2]]
        let ma = rho_a * l / 6.0;

        // Consistent mass for bending (Euler-Bernoulli):
        // (rhoA*L/420) * [[156, 22L, 54, -13L], [22L, 4L^2, 13L, -3L^2], ...]
        let mb = rho_a * l / 420.0;

        let mut m = [0.0; 36];

        // Axial: rows/cols 0 and 3
        m[0] = 2.0 * ma; // [0,0]
        m[3] = ma; // [0,3]
        m[18] = ma; // [3,0]
        m[21] = 2.0 * ma; // [3,3]

        // Bending: [w1,t1,w2,t2] -> [1,2,4,5]
        let bending_mass = [
            156.0 * mb,
            22.0 * mb * l,
            54.0 * mb,
            -13.0 * mb * l,
            22.0 * mb * l,
            4.0 * mb * l2,
            13.0 * mb * l,
            -3.0 * mb * l2,
            54.0 * mb,
            13.0 * mb * l,
            156.0 * mb,
            -22.0 * mb * l,
            -13.0 * mb * l,
            -3.0 * mb * l2,
            -22.0 * mb * l,
            4.0 * mb * l2,
        ];

        let bending_dofs = [1, 2, 4, 5];
        for (i, &gi) in bending_dofs.iter().enumerate() {
            for (j, &gj) in bending_dofs.iter().enumerate() {
                m[gi * 6 + gj] = bending_mass[i * 4 + j];
            }
        }

        m
    }

    /// Transform local mass matrix to global coordinates.
    pub fn mass_global(&self) -> [f64; 36] {
        let m_local = self.mass_local();
        let t = self.rotation_matrix();
        transform_6x6(&m_local, &t)
    }

    /// 6x6 rotation matrix from local to global coordinates.
    /// Rotates each (u, w) pair by the beam angle.
    fn rotation_matrix(&self) -> [f64; 36] {
        let c = self.cos_alpha;
        let s = self.sin_alpha;
        let mut t = [0.0; 36];

        // Node 1: [u_local, w_local, theta_local] -> [u_global, v_global, theta_global]
        // [u_local]   [c  s  0] [u_global]
        // [w_local] = [-s c  0] [v_global]
        // [theta]     [0  0  1] [theta]
        t[0] = c; // [0,0]
        t[1] = s; // [0,1]
        t[6] = -s; // [1,0]
        t[7] = c; // [1,1]
        t[14] = 1.0; // [2,2]

        // Node 2: same rotation
        t[3 * 6 + 3] = c;
        t[3 * 6 + 4] = s;
        t[4 * 6 + 3] = -s;
        t[4 * 6 + 4] = c;
        t[5 * 6 + 5] = 1.0;

        t
    }

    /// Global DOF indices for this element.
    /// Node `i` has DOFs `[3*node + 0, 3*node + 1, 3*node + 2]` = `[u, v, theta]`.
    pub fn global_dofs(&self) -> [usize; 6] {
        let n1 = self.nodes[0];
        let n2 = self.nodes[1];
        [
            n1 * 3,
            n1 * 3 + 1,
            n1 * 3 + 2,
            n2 * 3,
            n2 * 3 + 1,
            n2 * 3 + 2,
        ]
    }
}

/// Transform a 6x6 local matrix to global: `K_g = T^T K_l T`.
fn transform_6x6(k_local: &[f64; 36], t: &[f64; 36]) -> [f64; 36] {
    // First: temp = K_l * T
    let mut temp = [0.0; 36];
    for i in 0..6 {
        for j in 0..6 {
            for p in 0..6 {
                temp[i * 6 + j] += k_local[i * 6 + p] * t[p * 6 + j];
            }
        }
    }

    // Then: K_g = T^T * temp
    let mut k_global = [0.0; 36];
    for i in 0..6 {
        for j in 0..6 {
            for p in 0..6 {
                k_global[i * 6 + j] += t[p * 6 + i] * temp[p * 6 + j];
            }
        }
    }

    k_global
}

/// Beam model wrapper implementing [`ElementForm`](crate::forms::ElementForm).
///
/// Enables beam elements to participate in the generic assembly pipeline
/// alongside FEM forms. Delegates to the existing analytical stiffness matrices.
pub struct BeamModel {
    pub elements: Vec<BeamElement2D>,
    pub n_dofs: usize,
}

impl BeamModel {
    pub fn new(elements: Vec<BeamElement2D>, n_dofs: usize) -> Self {
        Self { elements, n_dofs }
    }
}

impl crate::forms::ElementForm<f64> for BeamModel {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        6
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.elements[elem_idx].global_dofs().to_vec()
    }

    fn element_matrix(&self, elem_idx: usize) -> Vec<f64> {
        self.elements[elem_idx].stiffness_global().to_vec()
    }
}

/// Corotational beam form for geometrically nonlinear analysis.
///
/// Uses the corotational approach: at each element, extract the rigid rotation
/// from the deformed configuration, compute local deformations in the corotated
/// frame, then apply the analytical linear stiffness locally and transform back.
/// A geometric stiffness term accounts for the change of orientation under load.
///
/// Implements [`NonlinearElementForm<f64>`](crate::forms::NonlinearElementForm).
pub struct CorotationalBeamForm {
    /// Reference (undeformed) elements.
    pub elements: Vec<BeamElement2D>,
    /// Total number of global DOFs.
    pub n_dofs: usize,
}

impl CorotationalBeamForm {
    pub fn new(elements: Vec<BeamElement2D>, n_dofs: usize) -> Self {
        Self { elements, n_dofs }
    }

    /// Extract deformed node positions from global solution.
    fn deformed_coords(&self, elem_idx: usize, u: &[f64]) -> ([f64; 2], [f64; 2]) {
        let elem = &self.elements[elem_idx];
        let n1 = elem.nodes[0];
        let n2 = elem.nodes[1];

        // Reference positions (reconstructed from element geometry)
        let x1_ref = 0.0_f64; // element stores relative geometry
        let y1_ref = 0.0_f64;
        // p2 relative to p1 in global frame:
        let x2_ref = elem.length * elem.cos_alpha;
        let y2_ref = elem.length * elem.sin_alpha;

        let x1 = x1_ref + u[n1 * 3];
        let y1 = y1_ref + u[n1 * 3 + 1];
        let x2 = x2_ref + u[n2 * 3];
        let y2 = y2_ref + u[n2 * 3 + 1];

        ([x1, y1], [x2, y2])
    }
}

impl crate::forms::NonlinearElementForm<f64> for CorotationalBeamForm {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        6
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.elements[elem_idx].global_dofs().to_vec()
    }

    fn num_global_dofs(&self) -> usize {
        self.n_dofs
    }

    fn element_residual(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let elem = &self.elements[elem_idx];
        let dofs = elem.global_dofs();

        // Extract global displacements for this element
        let mut u_global = [0.0; 6];
        for (i, &d) in dofs.iter().enumerate() {
            u_global[i] = global_solution[d];
        }

        // Deformed positions
        let (p1, p2) = self.deformed_coords(elem_idx, global_solution);
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let l_def = (dx * dx + dy * dy).sqrt();
        let cos_def = dx / l_def;
        let sin_def = dy / l_def;

        // Current rotation matrix
        let mut t_def = [0.0; 36];
        t_def[0] = cos_def;
        t_def[1] = sin_def;
        t_def[6] = -sin_def;
        t_def[7] = cos_def;
        t_def[14] = 1.0;
        t_def[3 * 6 + 3] = cos_def;
        t_def[3 * 6 + 4] = sin_def;
        t_def[4 * 6 + 3] = -sin_def;
        t_def[4 * 6 + 4] = cos_def;
        t_def[5 * 6 + 5] = 1.0;

        // Local deformational DOFs in the corotated frame. Both nodes lie
        // on the local x-axis by construction (origin at deformed node 1,
        // x-axis along the deformed chord), so transverse deformational
        // displacements are exactly zero — rotating the global displacement
        // vector (the old `T_def * u_global` form) instead leaves rigid
        // chord-rotation contamination that accumulates to percent-level
        // tip errors on elastica benchmarks. Likewise the local rotations
        // subtract the rigid chord rotation Δβ, recovered exactly in
        // (-π, π] from the reference/deformed direction cosines.
        let delta_beta = (sin_def * elem.cos_alpha - cos_def * elem.sin_alpha)
            .atan2(cos_def * elem.cos_alpha + sin_def * elem.sin_alpha);
        let mut d_local = [0.0; 6];
        d_local[2] = u_global[2] - delta_beta;
        d_local[3] = l_def - elem.length; // node 2 axial: extension
        d_local[5] = u_global[5] - delta_beta;

        // Local internal forces: f_local = K_local * d_local
        let k_local = elem.stiffness_local();
        let mut f_local = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                f_local[i] += k_local[i * 6 + j] * d_local[j];
            }
        }

        // Transform to global: f_global = T_def^T * f_local
        let mut f_global = vec![0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                f_global[i] += t_def[j * 6 + i] * f_local[j];
            }
        }

        f_global
    }

    fn element_tangent(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let elem = &self.elements[elem_idx];

        // Deformed geometry
        let (p1, p2) = self.deformed_coords(elem_idx, global_solution);
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let l_def = (dx * dx + dy * dy).sqrt();
        let cos_def = dx / l_def;
        let sin_def = dy / l_def;

        // Current rotation matrix
        let mut t_def = [0.0; 36];
        t_def[0] = cos_def;
        t_def[1] = sin_def;
        t_def[6] = -sin_def;
        t_def[7] = cos_def;
        t_def[14] = 1.0;
        t_def[3 * 6 + 3] = cos_def;
        t_def[3 * 6 + 4] = sin_def;
        t_def[4 * 6 + 3] = -sin_def;
        t_def[4 * 6 + 4] = cos_def;
        t_def[5 * 6 + 5] = 1.0;

        // Material tangent: K_mat = T^T * K_local * T
        let k_local = elem.stiffness_local();
        let k_mat = transform_6x6(&k_local, &t_def);

        // Geometric stiffness from axial force
        let dofs = elem.global_dofs();
        let mut u_global = [0.0; 6];
        for (i, &d) in dofs.iter().enumerate() {
            u_global[i] = global_solution[d];
        }

        // Local deformations to get axial force
        let mut _d_local = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                _d_local[i] += t_def[i * 6 + j] * u_global[j];
            }
        }
        _d_local[0] = 0.0;
        _d_local[3] = l_def - elem.length;

        let n_axial = elem.section.ea() * (l_def - elem.length) / elem.length;

        // Geometric stiffness in global coordinates
        // K_sigma accounts for transverse stiffening from axial force
        let s2 = sin_def * sin_def;
        let c2 = cos_def * cos_def;
        let sc = sin_def * cos_def;
        let factor = n_axial / l_def;

        // DOF ordering: [u1, v1, theta1, u2, v2, theta2]
        // Geometric stiffness from axial load:
        // [ s² -sc 0 -s²  sc 0]
        // [-sc  c² 0  sc -c² 0]
        // [ 0   0  0  0   0  0]
        // [-s²  sc 0  s² -sc 0]
        // [ sc -c² 0 -sc  c² 0]
        // [ 0   0  0  0   0  0]
        let mut k_geo = [0.0; 36];
        k_geo[0] = s2 * factor; // [0,0]
        k_geo[1] = -sc * factor; // [0,1]
        k_geo[3] = -s2 * factor; // [0,3]
        k_geo[4] = sc * factor; // [0,4]

        k_geo[6] = -sc * factor; // [1,0]
        k_geo[7] = c2 * factor; // [1,1]
        k_geo[9] = sc * factor; // [1,3]
        k_geo[10] = -c2 * factor; // [1,4]

        k_geo[18] = -s2 * factor; // [3,0]
        k_geo[19] = sc * factor; // [3,1]
        k_geo[21] = s2 * factor; // [3,3]
        k_geo[22] = -sc * factor; // [3,4]

        k_geo[24] = sc * factor; // [4,0]
        k_geo[25] = -c2 * factor; // [4,1]
        k_geo[27] = -sc * factor; // [4,3]
        k_geo[28] = c2 * factor; // [4,4]

        // Total tangent: K_mat + K_geo
        let mut k_total = vec![0.0; 36];
        for i in 0..36 {
            k_total[i] = k_mat[i] + k_geo[i];
        }

        k_total
    }
}

/// Assemble beam elements into a global stiffness matrix.
pub fn assemble_beam_stiffness(
    elements: &[BeamElement2D],
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

/// Assemble beam elements into a global mass matrix.
pub fn assemble_beam_mass(
    elements: &[BeamElement2D],
    n_dofs: usize,
) -> crate::assembly::TripletMatrix<f64> {
    let mut triplet = crate::assembly::TripletMatrix::new(n_dofs, n_dofs);

    for elem in elements {
        let m = elem.mass_global();
        let dofs = elem.global_dofs();

        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let val = m[i * 6 + j];
                if val.abs() > 1e-15 {
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
    use crate::forms::NonlinearElementForm;

    fn cantilever_beam(n_elements: usize) -> (Vec<BeamElement2D>, usize) {
        let l_total = 1.0;
        let section = BeamSection::rectangular(200e9, 0.01, 0.02, 7800.0);
        let dx = l_total / n_elements as f64;

        let elements: Vec<BeamElement2D> = (0..n_elements)
            .map(|i| {
                let x1 = i as f64 * dx;
                let x2 = (i + 1) as f64 * dx;
                BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section)
            })
            .collect();

        let n_nodes = n_elements + 1;
        let n_dofs = n_nodes * 3;
        (elements, n_dofs)
    }

    #[test]
    fn test_bending_stiffness_symmetry() {
        let section = BeamSection::rectangular(200e9, 0.01, 0.02, 7800.0);
        let elem = BeamElement2D::new([0, 1], 0.0, 0.0, 1.0, 0.0, section);
        let k = elem.bending_stiffness_local();

        for i in 0..4 {
            for j in i + 1..4 {
                assert!(
                    (k[i * 4 + j] - k[j * 4 + i]).abs() < 1e-6,
                    "K_b not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_stiffness_global_symmetry() {
        let section = BeamSection::rectangular(200e9, 0.01, 0.02, 7800.0);
        // Angled beam
        let elem = BeamElement2D::new([0, 1], 0.0, 0.0, 1.0, 0.5, section);
        let k = elem.stiffness_global();

        for i in 0..6 {
            for j in i + 1..6 {
                let diff = (k[i * 6 + j] - k[j * 6 + i]).abs();
                let scale = k[i * 6 + i].abs().max(1.0);
                assert!(
                    diff / scale < 1e-10,
                    "K_global not symmetric at ({i},{j}): {} vs {}",
                    k[i * 6 + j],
                    k[j * 6 + i]
                );
            }
        }
    }

    #[test]
    fn test_cantilever_tip_deflection() {
        // Cantilever beam with tip load P: delta = PL^3/(3EI)
        let (elements, n_dofs) = cantilever_beam(10);

        let stiffness = assemble_beam_stiffness(&elements, n_dofs);

        let p = 1000.0; // 1 kN tip load
        let mut rhs = vec![0.0; n_dofs];
        let tip_w_dof = (n_dofs / 3 - 1) * 3 + 1; // w at last node
        rhs[tip_w_dof] = -p; // downward load

        // Apply clamped BC at node 0: u=0, w=0, theta=0
        let clamped_dofs: std::collections::HashSet<usize> = [0, 1, 2].into();

        // Manual Dirichlet: zero rows/cols, identity diagonal
        // Use TripletMatrix approach
        let mut sys = crate::boundary::generic::FemSystem::new(stiffness, rhs);
        {
            let mut new_r = Vec::new();
            let mut new_c = Vec::new();
            let mut new_v = Vec::new();
            for i in 0..sys.matrix.nnz() {
                let r = sys.matrix.rows[i];
                let c = sys.matrix.cols[i];
                if !clamped_dofs.contains(&r) && !clamped_dofs.contains(&c) {
                    new_r.push(r);
                    new_c.push(c);
                    new_v.push(sys.matrix.values[i]);
                }
            }
            for &d in &clamped_dofs {
                new_r.push(d);
                new_c.push(d);
                new_v.push(1.0);
                sys.rhs[d] = 0.0;
            }
            sys.matrix.rows = new_r;
            sys.matrix.cols = new_c;
            sys.matrix.values = new_v;
        }

        let csr = sys.matrix.to_csr();
        let rhs = sys.rhs;
        let result = sonium_solvers::gmres(
            &csr,
            &rhs,
            &sonium_solvers::GmresConfig {
                max_iterations: 1000,
                tolerance: 1e-12,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
                ..Default::default()
            },
        );
        assert!(result.converged);

        let section = elements[0].section;
        let l = 1.0_f64;
        let exact_delta = p * l.powi(3) / (3.0 * section.ei());
        let computed_delta = -result.x[tip_w_dof]; // sign: load is negative, deflection is negative

        let rel_error = (computed_delta - exact_delta).abs() / exact_delta;
        assert!(
            rel_error < 0.01,
            "Tip deflection: computed={computed_delta:.6e}, exact={exact_delta:.6e}, error={rel_error:.2e}"
        );
    }

    #[test]
    fn test_beam_model_element_form_equivalence() {
        let (elements, n_dofs) = cantilever_beam(5);
        let model = BeamModel::new(elements.clone(), n_dofs);

        let k_legacy = assemble_beam_stiffness(&elements, n_dofs).to_csr();
        let k_form = crate::assembly::assemble_element_form(&model, n_dofs).to_csr();

        assert_eq!(k_legacy.num_rows, k_form.num_rows);
        // Compare all entries
        for i in 0..k_legacy.num_rows {
            let legacy_entries: std::collections::HashMap<usize, f64> =
                k_legacy.row_entries(i).collect();
            let form_entries: std::collections::HashMap<usize, f64> =
                k_form.row_entries(i).collect();

            for (&col, &val) in &legacy_entries {
                let form_val = form_entries.get(&col).copied().unwrap_or(0.0);
                assert!(
                    (val - form_val).abs() < 1e-10,
                    "Mismatch at ({i},{col}): legacy={val}, form={form_val}"
                );
            }
        }
    }

    #[test]
    fn test_corotational_zero_displacement_zero_residual() {
        let (elements, n_dofs) = cantilever_beam(5);
        let form = CorotationalBeamForm::new(elements, n_dofs);
        let u = vec![0.0; n_dofs];

        for e in 0..form.num_elements() {
            let r = form.element_residual(e, &u);
            for (i, &ri) in r.iter().enumerate() {
                assert!(ri.abs() < 1e-10, "Residual[{e}][{i}] = {ri}, expected 0");
            }
        }
    }

    #[test]
    fn test_corotational_tangent_fd_check() {
        let (elements, n_dofs) = cantilever_beam(3);
        let form = CorotationalBeamForm::new(elements, n_dofs);

        // Small displacement state
        let mut u = vec![0.0; n_dofs];
        u[4] = 0.001; // small transverse displacement at node 1

        let eps = 1e-7;
        let elem_idx = 0;
        let dofs = form.local_to_global(elem_idx);
        let n_local = dofs.len();

        let k_analytical = form.element_tangent(elem_idx, &u);
        let r0 = form.element_residual(elem_idx, &u);

        // Finite difference tangent
        for j in 0..n_local {
            let gj = dofs[j];
            let u_old = u[gj];

            u[gj] = u_old + eps;
            let r_plus = form.element_residual(elem_idx, &u);
            u[gj] = u_old;

            for i in 0..n_local {
                let k_fd = (r_plus[i] - r0[i]) / eps;
                let k_an = k_analytical[i * n_local + j];
                // Corotational tangent is approximate (simplified geometric stiffness).
                // 5% agreement with FD is acceptable.
                let tol = 0.05 * k_an.abs().max(1.0);
                assert!(
                    (k_fd - k_an).abs() < tol,
                    "FD tangent mismatch at ({i},{j}): FD={k_fd:.6e}, analytical={k_an:.6e}, diff={:.2}%",
                    100.0 * (k_fd - k_an).abs() / k_an.abs().max(1e-10)
                );
            }
        }
    }

    #[test]
    fn test_corotational_rigid_rotation_zero_residual() {
        // A rigidly rotated (but undeformed) element must produce zero
        // internal forces: the local frame follows the chord exactly.
        let section = BeamSection::rectangular(200e9, 0.01, 0.02, 7800.0);
        let length = 1.0;
        let elem = BeamElement2D::new([0, 1], 0.0, 0.0, length, 0.0, section);
        let form = CorotationalBeamForm::new(vec![elem], 6);
        let gamma = 0.3f64;
        let u = vec![
            0.0,
            0.0,
            gamma,
            length * (gamma.cos() - 1.0),
            length * gamma.sin(),
            gamma,
        ];
        let residual = form.element_residual(0, &u);
        for (i, &value) in residual.iter().enumerate() {
            assert!(
                value.abs() < 1e-9,
                "rigid rotation residual[{i}] = {value}, expected 0"
            );
        }
    }

    #[test]
    fn test_corotational_tangent_at_zero_is_linear_stiffness() {
        // At zero displacement the deformed frame coincides with the
        // reference frame and the axial force vanishes, so the tangent
        // must reproduce the linear element stiffness exactly.
        let (elements, n_dofs) = cantilever_beam(3);
        let form = CorotationalBeamForm::new(elements.clone(), n_dofs);
        let u = vec![0.0; n_dofs];
        for (e, elem) in elements.iter().enumerate() {
            let tangent = form.element_tangent(e, &u);
            let linear = elem.stiffness_global();
            let scale = linear.iter().map(|value| value.abs()).fold(1.0, f64::max);
            for i in 0..36 {
                assert!(
                    (tangent[i] - linear[i]).abs() / scale < 1e-12,
                    "tangent/linear mismatch at element {e} entry {i}"
                );
            }
        }
    }

    #[test]
    fn test_timoshenko_vs_euler_bernoulli() {
        let section_eb = BeamSection::rectangular(200e9, 0.01, 0.02, 7800.0);
        let section_ts = BeamSection::timoshenko(
            200e9,
            0.01 * 0.02_f64.powi(3) / 12.0, // I
            0.01 * 0.02,                    // A
            7800.0,
            0.3,
            5.0 / 6.0,
        );

        let elem_eb = BeamElement2D::new([0, 1], 0.0, 0.0, 1.0, 0.0, section_eb);
        let elem_ts = BeamElement2D::new([0, 1], 0.0, 0.0, 1.0, 0.0, section_ts);

        let k_eb = elem_eb.bending_stiffness_local();
        let k_ts = elem_ts.bending_stiffness_local();

        // Timoshenko should be softer (smaller stiffness) than Euler-Bernoulli
        // For slender beams the difference is small
        assert!(
            k_ts[0] <= k_eb[0],
            "Timoshenko K[0,0]={} should be <= EB K[0,0]={}",
            k_ts[0],
            k_eb[0]
        );
    }
}
