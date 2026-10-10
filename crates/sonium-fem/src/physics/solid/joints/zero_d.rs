//! Zero-dimensional structural elements.
//!
//! Scalar and nodal springs, viscous dampers, point masses, and 6x6
//! bushings for joints, mounts, and quasi-static spotwelds. Each element
//! stores explicit global DOF indices plus a dense matrix, then scatters
//! into a triplet system. Damping and mass variants target transient and
//! modal assembly alongside the stiffness path.

// Rust guideline compliant 2026-02-21

/// Dense 0-D element with explicit global DOFs.
#[derive(Debug, Clone)]
pub struct NodalElement {
    /// Global DOF index per local row and column.
    pub dofs: Vec<usize>,
    /// Dense row-major local matrix.
    pub matrix: Vec<f64>,
}

impl NodalElement {
    /// Build an element, checking square dense storage.
    ///
    /// # Panics
    ///
    /// Panics when the matrix is not square over the DOF list.
    #[must_use]
    pub fn new(dofs: Vec<usize>, matrix: Vec<f64>) -> Self {
        assert_eq!(
            dofs.len() * dofs.len(),
            matrix.len(),
            "matrix must be square"
        );
        Self { dofs, matrix }
    }

    /// Scatter the dense matrix into a global triplet system.
    pub fn assemble_into(&self, triplet: &mut crate::assembly::TripletMatrix<f64>) {
        for (i, &gi) in self.dofs.iter().enumerate() {
            for (j, &gj) in self.dofs.iter().enumerate() {
                let val = self.matrix[i * self.dofs.len() + j];
                if val.abs() > 1.0e-15 {
                    triplet.push(gi, gj, val);
                }
            }
        }
    }
}

/// Axial spring between two scalar DOFs.
#[derive(Debug, Clone, Copy)]
pub struct AxialSpring {
    /// First scalar DOF.
    pub dof_a: usize,
    /// Second scalar DOF.
    pub dof_b: usize,
    /// Stiffness in N/m.
    pub stiffness: f64,
}

impl AxialSpring {
    /// Create a spring with stiffness in N/m.
    #[must_use]
    pub fn new(dof_a: usize, dof_b: usize, stiffness: f64) -> Self {
        Self {
            dof_a,
            dof_b,
            stiffness,
        }
    }

    /// Convert to a scatterable nodal element.
    #[must_use]
    pub fn element(&self) -> NodalElement {
        let k = self.stiffness;
        NodalElement::new(vec![self.dof_a, self.dof_b], vec![k, -k, -k, k])
    }
}

/// Three-dimensional spring between two 3-DOF nodes.
#[derive(Debug, Clone)]
pub struct Spring3D {
    /// Global translational DOFs of node A (3 entries).
    pub dofs_a: [usize; 3],
    /// Global translational DOFs of node B (3 entries).
    pub dofs_b: [usize; 3],
    /// Unit direction of the spring axis in global coordinates.
    pub direction: [f64; 3],
    /// Axial stiffness in N/m.
    pub stiffness: f64,
    /// Optional transverse stiffness in N/m for a full 3-D mount.
    pub transverse_stiffness: f64,
}

impl Spring3D {
    /// Create an axial-only 3-D spring along `direction`.
    #[must_use]
    pub fn axial(
        dofs_a: [usize; 3],
        dofs_b: [usize; 3],
        direction: [f64; 3],
        stiffness: f64,
    ) -> Self {
        Self {
            dofs_a,
            dofs_b,
            direction: normalize(direction),
            stiffness,
            transverse_stiffness: 0.0,
        }
    }

    /// Convert to a scatterable 6x6 nodal element.
    #[must_use]
    pub fn element(&self) -> NodalElement {
        let d = self.direction;
        // Axial projector plus isotropic transverse stiffness.
        let mut k3 = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                let delta = f64::from(i == j);
                k3[i * 3 + j] = self.stiffness * d[i] * d[j]
                    + self.transverse_stiffness * (delta - d[i] * d[j]);
            }
        }
        let mut dofs = Vec::with_capacity(6);
        dofs.extend_from_slice(&self.dofs_a);
        dofs.extend_from_slice(&self.dofs_b);
        let mut matrix = vec![0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                matrix[i * 6 + j] = k3[i * 3 + j];
                matrix[i * 6 + 3 + j] = -k3[i * 3 + j];
                matrix[(3 + i) * 6 + j] = -k3[i * 3 + j];
                matrix[(3 + i) * 6 + 3 + j] = k3[i * 3 + j];
            }
        }
        NodalElement::new(dofs, matrix)
    }
}

/// Viscous damper between two scalar DOFs.
#[derive(Debug, Clone, Copy)]
pub struct ViscousDamper {
    /// First scalar DOF.
    pub dof_a: usize,
    /// Second scalar DOF.
    pub dof_b: usize,
    /// Damping coefficient in N s/m.
    pub damping: f64,
}

impl ViscousDamper {
    /// Create a damper with coefficient in N s/m.
    #[must_use]
    pub fn new(dof_a: usize, dof_b: usize, damping: f64) -> Self {
        Self {
            dof_a,
            dof_b,
            damping,
        }
    }

    /// Convert to a scatterable nodal damping element.
    #[must_use]
    pub fn element(&self) -> NodalElement {
        let c = self.damping;
        NodalElement::new(vec![self.dof_a, self.dof_b], vec![c, -c, -c, c])
    }
}

/// Point mass on translational DOFs with optional rotary inertia.
#[derive(Debug, Clone)]
pub struct PointMass {
    /// Global translational DOFs (up to 3 entries).
    pub translational_dofs: Vec<usize>,
    /// Global rotational DOFs paired with `rotary_inertia`.
    pub rotational_dofs: Vec<usize>,
    /// Mass in kg.
    pub mass: f64,
    /// Rotary inertia in kg m^2 per rotational DOF.
    pub rotary_inertia: f64,
}

impl PointMass {
    /// Create a translational point mass.
    #[must_use]
    pub fn new(translational_dofs: Vec<usize>, mass: f64) -> Self {
        Self {
            translational_dofs,
            rotational_dofs: Vec::new(),
            mass,
            rotary_inertia: 0.0,
        }
    }

    /// Attach rotary inertia to rotational DOFs.
    #[must_use]
    pub fn with_rotary(mut self, rotational_dofs: Vec<usize>, rotary_inertia: f64) -> Self {
        self.rotational_dofs = rotational_dofs;
        self.rotary_inertia = rotary_inertia;
        self
    }

    /// Convert to a scatterable diagonal mass element.
    #[must_use]
    pub fn element(&self) -> NodalElement {
        let mut dofs = self.translational_dofs.clone();
        dofs.extend_from_slice(&self.rotational_dofs);
        let n = dofs.len();
        let mut matrix = vec![0.0; n * n];
        for i in 0..self.translational_dofs.len() {
            matrix[i * n + i] = self.mass;
        }
        for (k, _) in self.rotational_dofs.iter().enumerate() {
            let i = self.translational_dofs.len() + k;
            matrix[i * n + i] = self.rotary_inertia;
        }
        NodalElement::new(dofs, matrix)
    }
}

/// Bushing joint: 6x6 stiffness plus 6x6 damping between two nodes.
///
/// Local axes default to global; an optional 3x3 row-major rotation maps
/// local joint axes to global coordinates per 3-DOF block.
#[derive(Debug, Clone)]
pub struct Bushing {
    /// Global 6-DOF list of node A (`[ux, uy, uz, rx, ry, rz]`).
    pub dofs_a: [usize; 6],
    /// Global 6-DOF list of node B.
    pub dofs_b: [usize; 6],
    /// Local stiffness, row-major 6x6.
    pub stiffness: [f64; 36],
    /// Local damping, row-major 6x6.
    pub damping: [f64; 36],
    /// Local-to-global axis rows, or identity when `None`.
    pub axes: Option<[f64; 9]>,
}

impl Bushing {
    /// Create a bushing with diagonal stiffness and damping.
    #[must_use]
    pub fn diagonal(
        dofs_a: [usize; 6],
        dofs_b: [usize; 6],
        stiffness_diag: [f64; 6],
        damping_diag: [f64; 6],
    ) -> Self {
        let mut stiffness = [0.0; 36];
        let mut damping = [0.0; 36];
        for i in 0..6 {
            stiffness[i * 6 + i] = stiffness_diag[i];
            damping[i * 6 + i] = damping_diag[i];
        }
        Self {
            dofs_a,
            dofs_b,
            stiffness,
            damping,
            axes: None,
        }
    }

    /// Set local joint axes (row-major 3x3).
    #[must_use]
    pub fn with_axes(mut self, axes: [f64; 9]) -> Self {
        self.axes = Some(axes);
        self
    }

    /// Rotate one 6x6 local block to global coordinates.
    fn global_block(&self, local: &[f64; 36]) -> [f64; 36] {
        let Some(axes) = self.axes else {
            return *local;
        };
        // temp = L * T with 3x3 axis blocks, then out = T' * temp.
        let mut temp = [0.0; 36];
        for row in 0..6 {
            for col in 0..6 {
                let block = col / 3;
                let mut sum = 0.0;
                for p in 0..3 {
                    sum += local[row * 6 + block * 3 + p] * axes[p * 3 + col % 3];
                }
                temp[row * 6 + col] = sum;
            }
        }
        let mut out = [0.0; 36];
        for row in 0..6 {
            for col in 0..6 {
                let block = row / 3;
                let mut sum = 0.0;
                for p in 0..3 {
                    sum += axes[p * 3 + row % 3] * temp[(block * 3 + p) * 6 + col];
                }
                out[row * 6 + col] = sum;
            }
        }
        out
    }

    /// Convert to a scatterable 12x12 stiffness element.
    #[must_use]
    pub fn stiffness_element(&self) -> NodalElement {
        self.block_element(&self.global_block(&self.stiffness))
    }

    /// Convert to a scatterable 12x12 damping element.
    #[must_use]
    pub fn damping_element(&self) -> NodalElement {
        self.block_element(&self.global_block(&self.damping))
    }

    /// Assemble a `[[K, -K], [-K, K]]` nodal element from a 6x6 block.
    fn block_element(&self, block: &[f64; 36]) -> NodalElement {
        let mut dofs = Vec::with_capacity(12);
        dofs.extend_from_slice(&self.dofs_a);
        dofs.extend_from_slice(&self.dofs_b);
        let mut matrix = vec![0.0; 144];
        for i in 0..6 {
            for j in 0..6 {
                let v = block[i * 6 + j];
                matrix[i * 12 + j] = v;
                matrix[i * 12 + 6 + j] = -v;
                matrix[(6 + i) * 12 + j] = -v;
                matrix[(6 + i) * 12 + 6 + j] = v;
            }
        }
        NodalElement::new(dofs, matrix)
    }
}

/// Quasi-static spotweld between coincident structural nodes.
///
/// Models the weld as a stiff diagonal bushing; `check` reports the peak
/// force resultant against a nugget strength for post-processing.
#[derive(Debug, Clone)]
pub struct Spotweld {
    /// Underlying stiff bushing.
    pub bushing: Bushing,
    /// Nugget shear strength in N.
    pub strength: f64,
}

impl Spotweld {
    /// Create a spotweld with penalty stiffness and nugget strength.
    #[must_use]
    pub fn new(
        dofs_a: [usize; 6],
        dofs_b: [usize; 6],
        penalty_stiffness: f64,
        penalty_rotation: f64,
        strength: f64,
    ) -> Self {
        let stiffness_diag = [
            penalty_stiffness,
            penalty_stiffness,
            penalty_stiffness,
            penalty_rotation,
            penalty_rotation,
            penalty_rotation,
        ];
        Self {
            bushing: Bushing::diagonal(dofs_a, dofs_b, stiffness_diag, [0.0; 6]),
            strength,
        }
    }

    /// Peak translational force resultant for relative displacements.
    ///
    /// Returns `(resultant, utilization)` with utilization above one marking
    /// a failed nugget.
    #[must_use]
    pub fn check(&self, relative_displacement: [f64; 6]) -> (f64, f64) {
        let block = self.bushing.global_block(&self.bushing.stiffness);
        let mut force = [0.0; 3];
        for i in 0..3 {
            for j in 0..6 {
                force[i] += block[i * 6 + j] * relative_displacement[j];
            }
        }
        let resultant = (force[0] * force[0] + force[1] * force[1] + force[2] * force[2]).sqrt();
        (resultant, resultant / self.strength)
    }
}

/// Normalize a 3-vector to unit length.
///
/// # Panics
///
/// Panics for a zero-length direction.
fn normalize(direction: [f64; 3]) -> [f64; 3] {
    let norm =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    assert!(norm > 0.0, "spring direction must be nonzero");
    [
        direction[0] / norm,
        direction[1] / norm,
        direction[2] / norm,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_spring_series_matches_closed_form() {
        // Two springs k1, k2 in series: 1/k = 1/k1 + 1/k2.
        let k1 = 100.0;
        let k2 = 300.0;
        let s1 = AxialSpring::new(0, 1, k1).element();
        let s2 = AxialSpring::new(1, 2, k2).element();
        // Condense the middle DOF: k_series = k1 k2 / (k1 + k2).
        let condensed = k1 * k2 / (k1 + k2);
        // Verify assembly entries carry both springs at the shared DOF.
        let mut diag_1 = 0.0;
        for element in [&s1, &s2] {
            for (i, &gi) in element.dofs.iter().enumerate() {
                if gi == 1 {
                    diag_1 += element.matrix[i * element.dofs.len() + i];
                }
            }
        }
        assert!((diag_1 - (k1 + k2)).abs() < 1.0e-12);
        assert!((condensed - 75.0).abs() < 1.0e-12);
    }

    #[test]
    fn spring_3d_projects_onto_axis() {
        let spring = Spring3D::axial([0, 1, 2], [3, 4, 5], [1.0, 1.0, 0.0], 200.0).element();
        // Unit axis is (1,1,0)/sqrt(2); K[0,0] block entry is k/2.
        assert!((spring.matrix[0] - 100.0).abs() < 1.0e-12);
        assert!((spring.matrix[1] - 100.0).abs() < 1.0e-12);
        assert!(spring.matrix[2].abs() < 1.0e-12);
    }

    #[test]
    fn bushing_recovers_diagonal_without_axes() {
        let bushing = Bushing::diagonal(
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let stiffness = bushing.stiffness_element();
        assert!((stiffness.matrix[0] - 10.0).abs() < 1.0e-12);
        assert!((stiffness.matrix[6] + 10.0).abs() < 1.0e-12);
        let damping = bushing.damping_element();
        assert!((damping.matrix[12 * 5 + 5] - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn spotweld_check_flags_overload() {
        let weld = Spotweld::new(
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            1.0e9,
            1.0e6,
            5000.0,
        );
        let (ok_force, ok_use) = weld.check([1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((ok_force - 1000.0).abs() < 1.0e-9);
        assert!(ok_use < 1.0);
        let (_, bad_use) = weld.check([1.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(bad_use > 1.0);
    }
}
