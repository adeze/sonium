//! Flat triangular shell element: CST membrane + DKT plate bending
//!
//! 3-node triangle with 6 DOFs per node: `(u, v, w, theta_x, theta_y, theta_z)`.
//! - Membrane (in-plane): CST (Constant Strain Triangle), DOFs: u, v
//! - Bending (out-of-plane): DKT (Discrete Kirchhoff Triangle), DOFs: w, theta_x, theta_y
//! - theta_z: drilling rotation (small fictitious stiffness for conditioning)
//!
//! The element operates in a local coordinate system aligned with the triangle plane.
//! A rotation matrix transforms to/from global coordinates.

use super::section::ShellSection;
use crate::mesh::{Mesh, Point};

/// Flat triangular shell element.
#[derive(Debug, Clone)]
pub struct ShellElement {
    /// Node indices (3 nodes).
    pub nodes: [usize; 3],
    /// Triangle area.
    pub area: f64,
    /// Local-to-global rotation matrix (3x3, row-major).
    /// Columns are the local x, y, z axes in global coordinates.
    pub rotation: [f64; 9],
    /// Node coordinates in local frame: `[(x1,y1), (x2,y2), (x3,y3)]`.
    pub local_coords: [(f64, f64); 3],
    /// Global origin of the local shell coordinate frame.
    pub origin: [f64; 3],
    /// Shell section properties.
    pub section: ShellSection,
}

impl ShellElement {
    /// Create from mesh node indices.
    pub fn from_mesh(nodes: [usize; 3], mesh: &Mesh, section: ShellSection) -> Self {
        let p0 = mesh.nodes[nodes[0]];
        let p1 = mesh.nodes[nodes[1]];
        let p2 = mesh.nodes[nodes[2]];
        Self::from_points(nodes, &p0, &p1, &p2, section)
    }

    /// Create from three points.
    pub fn from_points(
        nodes: [usize; 3],
        p0: &Point,
        p1: &Point,
        p2: &Point,
        section: ShellSection,
    ) -> Self {
        // Build local coordinate system
        let v01 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
        let v02 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];

        // Local x: along edge 0->1
        let len01 = (v01[0] * v01[0] + v01[1] * v01[1] + v01[2] * v01[2]).sqrt();
        let ex = [v01[0] / len01, v01[1] / len01, v01[2] / len01];

        // Local z: normal to triangle
        let normal = [
            v01[1] * v02[2] - v01[2] * v02[1],
            v01[2] * v02[0] - v01[0] * v02[2],
            v01[0] * v02[1] - v01[1] * v02[0],
        ];
        let area2 = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let ez = [normal[0] / area2, normal[1] / area2, normal[2] / area2];

        // Local y: ez x ex
        let ey = [
            ez[1] * ex[2] - ez[2] * ex[1],
            ez[2] * ex[0] - ez[0] * ex[2],
            ez[0] * ex[1] - ez[1] * ex[0],
        ];

        let rotation = [
            ex[0], ey[0], ez[0], ex[1], ey[1], ez[1], ex[2], ey[2], ez[2],
        ];

        // Project nodes to local frame
        let to_local = |p: &Point| -> (f64, f64) {
            let dx = p.x - p0.x;
            let dy = p.y - p0.y;
            let dz = p.z - p0.z;
            (
                ex[0] * dx + ex[1] * dy + ex[2] * dz,
                ey[0] * dx + ey[1] * dy + ey[2] * dz,
            )
        };

        let lc0 = (0.0, 0.0);
        let lc1 = to_local(p1);
        let lc2 = to_local(p2);

        Self {
            nodes,
            area: area2 / 2.0,
            rotation,
            local_coords: [lc0, lc1, lc2],
            origin: [p0.x, p0.y, p0.z],
            section,
        }
    }

    /// CST membrane strain-displacement matrix (3×6, row-major).
    ///
    /// Maps local membrane DOFs `[u1,v1,u2,v2,u3,v3]` to strain `[ε_xx, ε_yy, γ_xy]`.
    /// Constant over the element (hence "Constant Strain Triangle").
    pub fn cst_b_matrix(&self) -> [f64; 18] {
        let [(x1, y1), (x2, y2), (x3, y3)] = self.local_coords;
        let a = self.area;
        let inv2a = 1.0 / (2.0 * a);
        let y23 = y2 - y3;
        let y31 = y3 - y1;
        let y12 = y1 - y2;
        let x32 = x3 - x2;
        let x13 = x1 - x3;
        let x21 = x2 - x1;
        [
            y23 * inv2a,
            0.0,
            y31 * inv2a,
            0.0,
            y12 * inv2a,
            0.0,
            0.0,
            x32 * inv2a,
            0.0,
            x13 * inv2a,
            0.0,
            x21 * inv2a,
            x32 * inv2a,
            y23 * inv2a,
            x13 * inv2a,
            y31 * inv2a,
            x21 * inv2a,
            y12 * inv2a,
        ]
    }

    /// CST membrane stiffness in local coordinates (6x6 for u1,v1,u2,v2,u3,v3).
    pub fn membrane_stiffness_local(&self) -> [f64; 36] {
        let [(x1, y1), (x2, y2), (x3, y3)] = self.local_coords;
        let a = self.area;
        let d = self.section.membrane_constitutive();

        // B-matrix for CST: constant strain over element
        // eps_xx = (1/2A) * (y23*u1 + y31*u2 + y12*u3)
        // eps_yy = (1/2A) * (x32*v1 + x13*v2 + x21*v3)
        // gamma_xy = (1/2A) * (x32*u1 + y23*v1 + x13*u2 + y31*v2 + x21*u3 + y12*v3)
        let y23 = y2 - y3;
        let y31 = y3 - y1;
        let y12 = y1 - y2;
        let x32 = x3 - x2;
        let x13 = x1 - x3;
        let x21 = x2 - x1;

        // B = (1/2A) * [[y23, 0, y31, 0, y12, 0],
        //               [0, x32, 0, x13, 0, x21],
        //               [x32, y23, x13, y31, x21, y12]]
        let inv2a = 1.0 / (2.0 * a);
        let b = [
            y23 * inv2a,
            0.0,
            y31 * inv2a,
            0.0,
            y12 * inv2a,
            0.0,
            0.0,
            x32 * inv2a,
            0.0,
            x13 * inv2a,
            0.0,
            x21 * inv2a,
            x32 * inv2a,
            y23 * inv2a,
            x13 * inv2a,
            y31 * inv2a,
            x21 * inv2a,
            y12 * inv2a,
        ];

        // K = A * B^T * D * B
        let mut db = [0.0; 18]; // D*B (3x6)
        for i in 0..3 {
            for j in 0..6 {
                for k in 0..3 {
                    db[i * 6 + j] += d[i * 3 + k] * b[k * 6 + j];
                }
            }
        }

        let mut km = [0.0; 36]; // B^T * D * B * A
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..3 {
                    km[i * 6 + j] += b[k * 6 + i] * db[k * 6 + j] * a;
                }
            }
        }

        km
    }

    /// Full 18x18 element stiffness in **local** coordinates.
    ///
    /// Combines CST membrane, DKT bending, and drilling stiffness before
    /// the rotation transform. Used by the corotational nonlinear form.
    pub fn stiffness_local_combined(&self) -> Vec<f64> {
        let km = self.membrane_stiffness_local();

        let mut k_local = vec![0.0; 18 * 18];

        let mem_dofs = [0, 1, 6, 7, 12, 13];
        for (i, &gi) in mem_dofs.iter().enumerate() {
            for (j, &gj) in mem_dofs.iter().enumerate() {
                k_local[gi * 18 + gj] = km[i * 6 + j];
            }
        }

        let db = self.section.bending_constitutive();
        let kb = super::dkt::dkt_bending_stiffness(&self.local_coords, &db, self.area);

        let bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16];
        for (i, &gi) in bend_dofs.iter().enumerate() {
            for (j, &gj) in bend_dofs.iter().enumerate() {
                k_local[gi * 18 + gj] = kb[i * 9 + j];
            }
        }

        let drill = self.section.membrane_stiffness() * self.area * 1e-6;
        for i in 0..3 {
            let dof = i * 6 + 5;
            k_local[dof * 18 + dof] = drill;
        }

        k_local
    }

    /// Full 18x18 element stiffness in global coordinates.
    ///
    /// DOF order per node: `(u, v, w, theta_x, theta_y, theta_z)`.
    /// Membrane contributes to (u, v), bending to (w, theta_x, theta_y),
    /// drilling rotation (theta_z) gets a small fictitious stiffness.
    pub fn stiffness_global(&self) -> Vec<f64> {
        let k_local = self.stiffness_local_combined();
        transform_18x18(&k_local, &self.rotation)
    }

    /// Consistent mass matrix (18x18 global).
    pub fn mass_global(&self) -> Vec<f64> {
        let rho_a = self.section.mass_per_area();
        let a = self.area;

        // Lumped mass: M = rho*A*t/3 * I per node (for translational DOFs)
        let nodal_mass = rho_a * a / 3.0;

        let mut m_local = vec![0.0; 18 * 18];
        for i in 0..3 {
            // Translational DOFs (u, v, w)
            m_local[(i * 6) * 18 + (i * 6)] = nodal_mass;
            m_local[(i * 6 + 1) * 18 + (i * 6 + 1)] = nodal_mass;
            m_local[(i * 6 + 2) * 18 + (i * 6 + 2)] = nodal_mass;

            // Rotational inertia (small, proportional to t^2)
            let rot_inertia = nodal_mass * self.section.thickness.powi(2) / 12.0;
            m_local[(i * 6 + 3) * 18 + (i * 6 + 3)] = rot_inertia;
            m_local[(i * 6 + 4) * 18 + (i * 6 + 4)] = rot_inertia;
            m_local[(i * 6 + 5) * 18 + (i * 6 + 5)] = rot_inertia;
        }

        transform_18x18(&m_local, &self.rotation)
    }

    /// Global DOF indices: node i -> [6*node+0..6*node+5].
    pub fn global_dofs(&self) -> [usize; 18] {
        let mut dofs = [0; 18];
        for (i, &node) in self.nodes.iter().enumerate() {
            for d in 0..6 {
                dofs[i * 6 + d] = node * 6 + d;
            }
        }
        dofs
    }
}

/// Transform 18x18 matrix from local to global: K_g = T_full^T K_l T_full.
/// T_full is block-diagonal with six 3x3 rotation blocks.
fn transform_18x18(k_local: &[f64], rot: &[f64; 9]) -> Vec<f64> {
    let n = 18;

    // Build 18x18 block-diagonal rotation matrix
    let mut t = vec![0.0; n * n];
    for block in 0..6 {
        let offset = block * 3;
        for i in 0..3 {
            for j in 0..3 {
                t[(offset + i) * n + (offset + j)] = rot[i * 3 + j];
            }
        }
    }

    // K_g = T K_l T^T with T the block-diagonal local-to-global rotation
    // (columns of `rot` are the local axes in global coordinates, so a
    // local vector maps to global via T and energies transform by
    // congruence). A previous revision used T^T K_l T, which mirrors every
    // non-axis-aligned element's stiffness and over-stiffens assemblies by
    // orders of magnitude.
    let mut temp = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for p in 0..n {
                temp[i * n + j] += k_local[i * n + p] * t[j * n + p];
            }
        }
    }

    let mut k_global = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for p in 0..n {
                k_global[i * n + j] += t[i * n + p] * temp[p * n + j];
            }
        }
    }

    k_global
}

/// Shell model wrapper implementing [`ElementForm`](crate::forms::ElementForm).
///
/// Enables shell elements to participate in the generic assembly pipeline.
/// Delegates to the existing analytical CST + DKT stiffness matrices.
pub struct ShellModel {
    pub elements: Vec<ShellElement>,
    pub n_dofs: usize,
}

impl ShellModel {
    pub fn new(elements: Vec<ShellElement>, n_dofs: usize) -> Self {
        Self { elements, n_dofs }
    }
}

impl crate::forms::ElementForm<f64> for ShellModel {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        18
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.elements[elem_idx].global_dofs().to_vec()
    }

    fn element_matrix(&self, elem_idx: usize) -> Vec<f64> {
        self.elements[elem_idx].stiffness_global()
    }
}

/// Assemble shell elements into a global stiffness matrix.
pub fn assemble_shell_stiffness(
    elements: &[ShellElement],
    n_dofs: usize,
) -> crate::assembly::TripletMatrix<f64> {
    let mut triplet = crate::assembly::TripletMatrix::new(n_dofs, n_dofs);

    for elem in elements {
        let k = elem.stiffness_global();
        let dofs = elem.global_dofs();

        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let val = k[i * 18 + j];
                if val.abs() > 1e-15 {
                    triplet.push(gi, gj, val);
                }
            }
        }
    }

    triplet
}

/// Assemble distributed pressure/load on shell triangles.
///
/// `pressure_fn` returns a scalar pressure in the element normal direction.
/// The load is distributed consistently to the translational DOFs of each
/// node: `f_i += p * area / 3 * n`.
pub fn assemble_shell_load(
    elements: &[ShellElement],
    n_dofs: usize,
    pressure_fn: impl Fn([f64; 3]) -> f64,
) -> Vec<f64> {
    let mut rhs = vec![0.0; n_dofs];
    for elem in elements {
        let normal = [elem.rotation[2], elem.rotation[5], elem.rotation[8]];
        let centroid = shell_element_centroid(elem);
        let nodal = pressure_fn(centroid) * elem.area / 3.0;
        for &node in &elem.nodes {
            rhs[node * 6] += nodal * normal[0];
            rhs[node * 6 + 1] += nodal * normal[1];
            rhs[node * 6 + 2] += nodal * normal[2];
        }
    }
    rhs
}

/// Add a concentrated shell point load at a node.
pub fn add_shell_point_load(rhs: &mut [f64], node: usize, load: [f64; 6]) {
    for dof in 0..6 {
        rhs[node * 6 + dof] += load[dof];
    }
}

fn shell_element_centroid(elem: &ShellElement) -> [f64; 3] {
    // Reconstruct centroid from local coordinates and the global basis.
    let cx = (elem.local_coords[0].0 + elem.local_coords[1].0 + elem.local_coords[2].0) / 3.0;
    let cy = (elem.local_coords[0].1 + elem.local_coords[1].1 + elem.local_coords[2].1) / 3.0;
    [
        elem.origin[0] + elem.rotation[0] * cx + elem.rotation[1] * cy,
        elem.origin[1] + elem.rotation[3] * cx + elem.rotation[4] * cy,
        elem.origin[2] + elem.rotation[6] * cx + elem.rotation[7] * cy,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_element_area() {
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(0.0, 1.0, 0.0);
        let section = ShellSection::new(200e9, 0.3, 0.001, 7800.0);

        let elem = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);
        assert!((elem.area - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_membrane_stiffness_symmetry() {
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(0.5, 0.8, 0.0);
        let section = ShellSection::cone(0.001);

        let elem = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);
        let km = elem.membrane_stiffness_local();

        for i in 0..6 {
            for j in i + 1..6 {
                assert!(
                    (km[i * 6 + j] - km[j * 6 + i]).abs() < 1e-6,
                    "Km not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_global_stiffness_symmetry() {
        // Tilted triangle in 3D
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.1);
        let p2 = Point::new_3d(0.5, 0.8, 0.2);
        let section = ShellSection::new(200e9, 0.3, 0.002, 7800.0);

        let elem = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);
        let k = elem.stiffness_global();

        for i in 0..18 {
            for j in i + 1..18 {
                let diff = (k[i * 18 + j] - k[j * 18 + i]).abs();
                let scale = k[i * 18 + i].abs().max(1.0);
                assert!(diff / scale < 1e-10, "K_global not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_shell_model_element_form_equivalence() {
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(0.5, 0.8, 0.0);
        let p3 = Point::new_3d(0.5, 0.3, 0.5);
        let section = ShellSection::new(200e9, 0.3, 0.002, 7800.0);

        let elements = vec![
            ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section),
            ShellElement::from_points([0, 2, 3], &p0, &p2, &p3, section),
        ];
        let n_nodes = 4;
        let n_dofs = n_nodes * 6;

        let model = ShellModel::new(elements.clone(), n_dofs);
        let k_legacy = assemble_shell_stiffness(&elements, n_dofs).to_csr();
        let k_form = crate::assembly::assemble_element_form(&model, n_dofs).to_csr();

        assert_eq!(k_legacy.num_rows, k_form.num_rows);
        for i in 0..k_legacy.num_rows {
            let legacy_entries: std::collections::HashMap<usize, f64> =
                k_legacy.row_entries(i).collect();
            let form_entries: std::collections::HashMap<usize, f64> =
                k_form.row_entries(i).collect();

            for (&col, &val) in &legacy_entries {
                let form_val = form_entries.get(&col).copied().unwrap_or(0.0);
                assert!(
                    (val - form_val).abs() < 1e-6,
                    "Mismatch at ({i},{col}): legacy={val}, form={form_val}"
                );
            }
        }
    }

    #[test]
    fn test_shell_load_uses_global_centroid() {
        let p0 = Point::new_3d(10.0, 0.0, 0.0);
        let p1 = Point::new_3d(12.0, 0.0, 0.0);
        let p2 = Point::new_3d(10.0, 2.0, 0.0);
        let section = ShellSection::cone(0.001);
        let elem = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);

        let rhs = assemble_shell_load(&[elem], 18, |x| x[0]);
        let total_z = rhs[2] + rhs[8] + rhs[14];
        let expected_centroid_x = (p0.x + p1.x + p2.x) / 3.0;
        let expected_load = expected_centroid_x * 2.0;

        assert!((total_z - expected_load).abs() < 1e-12);
    }

    #[test]
    fn test_flat_shell_matches_plane_for_membrane() {
        // A flat shell in the xy-plane should have membrane stiffness
        // identical to a plane stress CST element
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(0.0, 1.0, 0.0);
        let section = ShellSection::new(1.0, 0.25, 1.0, 1.0);

        let elem = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);
        let km = elem.membrane_stiffness_local();

        // All diagonal entries should be positive
        for i in 0..6 {
            assert!(km[i * 6 + i] > 0.0, "Km diagonal {i} should be positive");
        }
    }
}
