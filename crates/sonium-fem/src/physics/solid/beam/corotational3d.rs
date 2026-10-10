//! Corotational 3-D beam form for large rotations.
//!
//! Each element tracks a corotated frame built from the deformed chord plus
//! the reference transverse axes. Deformational rotations come from the local
//! rotation `E' R(theta) E0`, so rigid motions produce zero local strain by
//! construction. The tangent is a central-difference Jacobian of the residual,
//! which captures geometric stiffness without an analytic projector.

// Rust guideline compliant 2026-02-21

use super::element3d::BeamElement3D;

/// Corotational space-frame form for geometrically nonlinear analysis.
///
/// Implements [`NonlinearElementForm<f64>`](crate::forms::NonlinearElementForm)
/// with 6 DOFs per node (`[ux, uy, uz, rx, ry, rz]`). Rotations use the
/// rotation-vector parametrization.
#[derive(Debug)]
pub struct CorotationalBeamForm3D {
    /// Reference (undeformed) elements.
    pub elements: Vec<BeamElement3D>,
    /// Reference node positions indexed by node id.
    pub reference_positions: Vec<[f64; 3]>,
    /// Total number of global DOFs.
    pub n_dofs: usize,
    /// Finite-difference step for the tangent stiffness.
    pub fd_step: f64,
}

impl CorotationalBeamForm3D {
    /// Build a form from reference elements, positions, and a DOF count.
    #[must_use]
    pub fn new(
        elements: Vec<BeamElement3D>,
        reference_positions: Vec<[f64; 3]>,
        n_dofs: usize,
    ) -> Self {
        Self {
            elements,
            reference_positions,
            n_dofs,
            fd_step: 1.0e-8,
        }
    }

    /// Deformed node positions for one element.
    fn deformed_positions(&self, elem_idx: usize, u: &[f64]) -> [[f64; 3]; 2] {
        let elem = &self.elements[elem_idx];
        let mut out = [[0.0; 3]; 2];
        for (slot, node) in elem.nodes.iter().enumerate() {
            let base = node * 6;
            for component in 0..3 {
                out[slot][component] =
                    self.reference_positions[*node][component] + u[base + component];
            }
        }
        out
    }

    /// Current corotated frame: chord plus transported reference axes.
    fn current_frame(&self, elem_idx: usize, u: &[f64]) -> [f64; 9] {
        let elem = &self.elements[elem_idx];
        let [p1, p2] = self.deformed_positions(elem_idx, u);
        let chord = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let length = (chord[0] * chord[0] + chord[1] * chord[1] + chord[2] * chord[2]).sqrt();
        let e1 = [chord[0] / length, chord[1] / length, chord[2] / length];
        // Reference transverse axis with the chord component removed.
        let r2 = [elem.axes[3], elem.axes[4], elem.axes[5]];
        let dot = r2[0] * e1[0] + r2[1] * e1[1] + r2[2] * e1[2];
        let mut e2 = [
            r2[0] - dot * e1[0],
            r2[1] - dot * e1[1],
            r2[2] - dot * e1[2],
        ];
        let norm = (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]).sqrt();
        e2[0] /= norm;
        e2[1] /= norm;
        e2[2] /= norm;
        let e3 = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        [
            e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
        ]
    }

    /// Local deformational DOFs (12) in the corotated frame.
    fn local_deformations(&self, elem_idx: usize, u: &[f64]) -> [f64; 12] {
        let elem = &self.elements[elem_idx];
        let frame = self.current_frame(elem_idx, u);
        let [p1, p2] = self.deformed_positions(elem_idx, u);
        let chord = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let length = (chord[0] * chord[0] + chord[1] * chord[1] + chord[2] * chord[2]).sqrt();

        let mut d = [0.0; 12];
        d[6] = length - elem.length;
        for (node, base) in elem.nodes.iter().zip([0_usize, 6]) {
            let theta = [u[node * 6 + 3], u[node * 6 + 4], u[node * 6 + 5]];
            let global = rotation_matrix(&theta);
            // Local rotation Rl = A R(theta) A0' with row-frames A, A0.
            let e0 = &elem.axes;
            let e0_transpose = mat3_transpose(e0);
            let tmp = mat3_mul(&global, &e0_transpose);
            let local = mat3_mul(&frame, &tmp);
            let spin = rotation_vector(&local);
            d[base + 3] = spin[0];
            d[base + 4] = spin[1];
            d[base + 5] = spin[2];
        }
        d
    }

    /// Internal force vector for one element in global coordinates.
    fn internal_forces(&self, elem_idx: usize, u: &[f64]) -> [f64; 12] {
        let elem = &self.elements[elem_idx];
        let d = self.local_deformations(elem_idx, u);
        let k = elem.stiffness_local();
        let mut f_local = [0.0; 12];
        for i in 0..12 {
            for j in 0..12 {
                f_local[i] += k[i * 12 + j] * d[j];
            }
        }
        // Rotate local forces and moments back with the current frame.
        let frame = self.current_frame(elem_idx, u);
        let mut f_global = [0.0; 12];
        for block in 0..4 {
            for row in 0..3 {
                let mut sum = 0.0;
                for p in 0..3 {
                    // Global component row of frame row p.
                    sum += frame[p * 3 + row] * f_local[block * 3 + p];
                }
                f_global[block * 3 + row] = sum;
            }
        }
        f_global
    }
}

impl crate::forms::NonlinearElementForm<f64> for CorotationalBeamForm3D {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        12
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.elements[elem_idx].global_dofs().to_vec()
    }

    fn num_global_dofs(&self) -> usize {
        self.n_dofs
    }

    fn element_residual(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        self.internal_forces(elem_idx, global_solution).to_vec()
    }

    fn element_tangent(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let dofs = self.elements[elem_idx].global_dofs();
        let step = self.fd_step;
        let mut tangent = vec![0.0; 144];
        let mut perturbed = global_solution.to_vec();
        for (col, &dof) in dofs.iter().enumerate() {
            let base = global_solution[dof];
            let h = step * (1.0 + base.abs());
            perturbed[dof] = base + h;
            let plus = self.internal_forces(elem_idx, &perturbed);
            perturbed[dof] = base - h;
            let minus = self.internal_forces(elem_idx, &perturbed);
            perturbed[dof] = base;
            for row in 0..12 {
                tangent[row * 12 + col] = (plus[row] - minus[row]) / (2.0 * h);
            }
        }
        tangent
    }
}

/// Rotation matrix from a rotation vector via Rodrigues formula.
fn rotation_matrix(theta: &[f64; 3]) -> [f64; 9] {
    let angle = (theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2]).sqrt();
    if angle < 1.0e-12 {
        // First-order expansion avoids the division by a tiny angle.
        return [
            1.0, -theta[2], theta[1], theta[2], 1.0, -theta[0], -theta[1], theta[0], 1.0,
        ];
    }
    let (x, y, z) = (theta[0] / angle, theta[1] / angle, theta[2] / angle);
    let (s, c) = (angle.sin(), angle.cos());
    let v = 1.0 - c;
    [
        x * x * v + c,
        x * y * v - z * s,
        x * z * v + y * s,
        y * x * v + z * s,
        y * y * v + c,
        y * z * v - x * s,
        z * x * v - y * s,
        z * y * v + x * s,
        z * z * v + c,
    ]
}

/// Rotation vector from a rotation matrix via the matrix logarithm.
///
/// The angle uses `atan2` of the skew magnitude so small rotations stay
/// accurate to machine precision (`acos` collapses below 1e-8 rad).
fn rotation_vector(r: &[f64; 9]) -> [f64; 3] {
    let trace = r[0] + r[4] + r[8];
    let skew = [r[7] - r[5], r[2] - r[6], r[3] - r[1]];
    let sin_angle = 0.5 * (skew[0] * skew[0] + skew[1] * skew[1] + skew[2] * skew[2]).sqrt();
    let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let angle = sin_angle.atan2(cos_angle);
    if angle < 1.0e-12 {
        // First-order limit: spin components equal half the skew part.
        return [skew[0] / 2.0, skew[1] / 2.0, skew[2] / 2.0];
    }
    if (angle - std::f64::consts::PI).abs() < 1.0e-9 {
        // Stable pi-rotation extraction from the diagonal.
        let x = ((r[0] + 1.0) / 2.0).max(0.0).sqrt();
        let y = ((r[4] + 1.0) / 2.0).max(0.0).sqrt();
        let z = ((r[8] + 1.0) / 2.0).max(0.0).sqrt();
        let mut axis = [x, y, z];
        // Recover off-diagonal signs from the symmetric part.
        if r[1] + r[3] < 0.0 {
            axis[1] = -axis[1];
        }
        if r[2] + r[6] < 0.0 {
            axis[2] = -axis[2];
        }
        return [axis[0] * angle, axis[1] * angle, axis[2] * angle];
    }
    let factor = angle / (2.0 * sin_angle);
    [skew[0] * factor, skew[1] * factor, skew[2] * factor]
}

/// Row-major 3x3 matrix product.
fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    out
}

/// Row-major 3x3 transpose.
fn mat3_transpose(a: &[f64; 9]) -> [f64; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms::NonlinearElementForm;
    use crate::physics::solid::beam::section3d::Section3D;

    fn single_element_form() -> CorotationalBeamForm3D {
        let section = Section3D::rectangular(200.0e9, 0.3, 0.02, 0.01, 7800.0, false);
        let elem = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        );
        CorotationalBeamForm3D::new(vec![elem], vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 12)
    }

    #[test]
    fn rigid_rotation_produces_zero_residual() {
        let form = single_element_form();
        // Rotate both nodes by 30 degrees about global z, moving node 2 along.
        let angle = std::f64::consts::FRAC_PI_6;
        let (c, s) = (angle.cos(), angle.sin());
        let mut u = vec![0.0; 12];
        u[3] = 0.0;
        u[4] = 0.0;
        u[5] = angle;
        u[6] = c - 1.0;
        u[7] = s;
        u[11] = angle;
        let residual = form.element_residual(0, &u);
        let scale = 200.0e9 * 2.0e-4;
        for (i, &r) in residual.iter().enumerate() {
            assert!(r.abs() / scale < 1.0e-9, "residual[{i}] = {r}");
        }
    }

    #[test]
    fn small_displacement_matches_linear_stiffness() {
        let form = single_element_form();
        // Tiny displacements keep second-order chord-length effects in the noise.
        let u = vec![
            1.0e-7, 2.0e-7, -1.0e-7, 3.0e-8, -2.0e-8, 4.0e-8, -1.0e-7, 1.5e-7, 2.0e-7, -3.0e-8,
            1.0e-8, -2.0e-8,
        ];
        let residual = form.element_residual(0, &u);
        let k = form.elements[0].stiffness_local();
        let mut expected = [0.0; 12];
        for i in 0..12 {
            for j in 0..12 {
                expected[i] += k[i * 12 + j] * u[j];
            }
        }
        let scale = expected
            .iter()
            .fold(0.0_f64, |worst, &v| worst.max(v.abs()));
        for i in 0..12 {
            assert!(
                (residual[i] - expected[i]).abs() / scale < 1.0e-6,
                "dof {i}: {} vs {}",
                residual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn tangent_is_consistent_with_residual() {
        let form = single_element_form();
        let state = vec![
            0.05, -0.03, 0.02, 0.1, -0.08, 0.12, -0.04, 0.06, -0.02, 0.05, 0.09, -0.07,
        ];
        let direction = vec![
            0.2, -0.1, 0.05, 0.03, -0.07, 0.11, 0.09, -0.04, 0.06, -0.08, 0.02, 0.05,
        ];
        let report = crate::validation::check_tangent_consistency(
            |current| form.element_residual(0, current),
            |current, tangent_direction| {
                let tangent = form.element_tangent(0, current);
                (0..current.len())
                    .map(|row| {
                        (0..current.len())
                            .map(|column| tangent[row * 12 + column] * tangent_direction[column])
                            .sum()
                    })
                    .collect()
            },
            &state,
            &direction,
            &[1.0e-3, 5.0e-4, 2.5e-4],
        )
        .expect("corotational residual and tangent dimensions must agree");
        assert!(report.observed_order > 1.8, "{report:?}");
    }
}
