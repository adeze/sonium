//! Energy-consistent corotational MITC4 shell elements.
//!
//! Retains the existing reference membrane, bending, shear and drilling
//! matrix, including reference corner warp. Forces differentiate its elastic
//! potential through the position-dependent frame. This small-strain elastic
//! model uses the same incremental rotation measure as the triangular form.

// Rust guideline compliant 2026-02-21

use super::{CorotationalShellForm, QuadShellElement};
use crate::forms::NonlinearElementForm;

struct PreparedQuad {
    element: QuadShellElement,
    reference: [[f64; 3]; 4],
    stiffness: Vec<f64>,
}

/// Corotational quadrilateral shell forces derived from the MITC4 elastic potential.
///
/// Reference matrices are cached and immutable. The complete force includes
/// derivatives of the current frame; its numerical tangent preserves those
/// terms. Physical mass and boundary conditions remain the caller's responsibility.
pub struct CorotationalQuadShellForm {
    elements: Vec<PreparedQuad>,
    n_dofs: usize,
}

impl CorotationalQuadShellForm {
    /// Prepare reference quadrilateral geometry and elastic stiffness.
    ///
    /// # Panics
    /// Panics for nonfinite/degenerate reference geometry or stiffness, or
    /// element DOFs outside the declared global vector.
    pub fn new(elements: Vec<QuadShellElement>, n_dofs: usize) -> Self {
        let elements = elements
            .into_iter()
            .map(|element| {
                assert!(element.global_dofs().iter().all(|&dof| dof < n_dofs));
                assert!(element.area.is_finite() && element.area > 0.0);
                let reference = std::array::from_fn(|node| {
                    let (x, y) = element.local_coords[node];
                    let z = element.warp[node];
                    std::array::from_fn(|i| {
                        element.rotation[3 * i] * x
                            + element.rotation[3 * i + 1] * y
                            + element.rotation[3 * i + 2] * z
                    })
                });
                assert!(reference.iter().flatten().all(|v| v.is_finite()));
                let mut stiffness = element.stiffness_local_combined();
                assert!(stiffness.iter().all(|v| v.is_finite()));
                // The potential depends only on the symmetric material matrix.
                for i in 0..24 {
                    for j in 0..i {
                        let v = 0.5 * (stiffness[24 * i + j] + stiffness[24 * j + i]);
                        stiffness[24 * i + j] = v;
                        stiffness[24 * j + i] = v;
                    }
                }
                PreparedQuad {
                    element,
                    reference,
                    stiffness,
                }
            })
            .collect();
        Self { elements, n_dofs }
    }

    fn current_points(data: &PreparedQuad, u: &[f64; 24]) -> [[f64; 3]; 4] {
        std::array::from_fn(|node| {
            std::array::from_fn(|i| data.reference[node][i] + (u[6 * node + i] - u[i]))
        })
    }

    // Opposite-edge midpoint differences match QuadShellElement's frame.
    fn frame_triangle(points: &[[f64; 3]; 4]) -> [[f64; 3]; 3] {
        [
            [0.0; 3],
            std::array::from_fn(|i| {
                0.5 * (points[1][i] + points[2][i] - points[0][i] - points[3][i])
            }),
            std::array::from_fn(|i| {
                0.5 * (points[2][i] + points[3][i] - points[0][i] - points[1][i])
            }),
        ]
    }

    fn coordinates(data: &PreparedQuad, u: &[f64; 24]) -> ([f64; 24], [[f64; 3]; 4], [f64; 9]) {
        let points = Self::current_points(data, u);
        let rotation = CorotationalShellForm::deformed_rotation(&Self::frame_triangle(&points));
        let center: [f64; 3] =
            std::array::from_fn(|i| points.iter().map(|p| p[i]).sum::<f64>() / 4.0);
        let reference_center: [f64; 3] =
            std::array::from_fn(|i| data.reference.iter().map(|p| p[i]).sum::<f64>() / 4.0);
        let rigid =
            CorotationalShellForm::relative_rotation_vector(&data.element.rotation, &rotation);
        let relative =
            std::array::from_fn(|node| std::array::from_fn(|i| points[node][i] - center[i]));
        let mut q = [0.0; 24];
        for node in 0..4 {
            let current = CorotationalShellForm::global_to_local(&rotation, relative[node]);
            let reference = CorotationalShellForm::global_to_local(
                &data.element.rotation,
                std::array::from_fn(|i| data.reference[node][i] - reference_center[i]),
            );
            let theta = CorotationalShellForm::global_to_local(
                &rotation,
                [u[6 * node + 3], u[6 * node + 4], u[6 * node + 5]],
            );
            for i in 0..3 {
                q[6 * node + i] = current[i] - reference[i];
                q[6 * node + 3 + i] = theta[i] - rigid[i];
            }
        }
        (q, relative, rotation)
    }

    fn force(data: &PreparedQuad, u: &[f64; 24]) -> Vec<f64> {
        let (q, relative, rotation) = Self::coordinates(data, u);
        let local: [f64; 24] =
            std::array::from_fn(|i| (0..24).map(|j| data.stiffness[24 * i + j] * q[j]).sum());
        let frame = Self::frame_triangle(&Self::current_points(data, u));
        let mut result = vec![0.0; 24];
        for component in 0..3 {
            let a =
                CorotationalShellForm::deformed_rotation_variation(&frame, &rotation, 1, component);
            let b =
                CorotationalShellForm::deformed_rotation_variation(&frame, &rotation, 2, component);
            for node in 0..4 {
                let ca = [-0.5, 0.5, 0.5, -0.5][node];
                let cb = [-0.5, -0.5, 0.5, 0.5][node];
                let variation = std::array::from_fn(|i| ca * a[i] + cb * b[i]);
                let rigid = CorotationalShellForm::relative_rotation_vector(
                    &data.element.rotation,
                    &variation,
                );
                for other in 0..4 {
                    let frame_motion =
                        CorotationalShellForm::global_to_local(&variation, relative[other]);
                    let mut motion = [0.0; 3];
                    motion[component] = f64::from(node == other) - 0.25;
                    let direct = CorotationalShellForm::global_to_local(&rotation, motion);
                    let theta = CorotationalShellForm::global_to_local(
                        &variation,
                        [u[6 * other + 3], u[6 * other + 4], u[6 * other + 5]],
                    );
                    for i in 0..3 {
                        result[6 * node + component] += local[6 * other + i]
                            * (frame_motion[i] + direct[i])
                            + local[6 * other + 3 + i] * (theta[i] - rigid[i]);
                    }
                }
                for i in 0..3 {
                    result[6 * node + 3 + component] +=
                        rotation[3 * component + i] * local[6 * node + 3 + i];
                }
            }
        }
        result
    }

    /// Evaluate the corotated reference elastic potential of one element.
    ///
    /// # Panics
    /// Panics for an invalid element index or an insufficient global vector.
    pub fn element_strain_energy(&self, index: usize, state: &[f64]) -> f64 {
        let data = &self.elements[index];
        let dofs = data.element.global_dofs();
        let u = std::array::from_fn(|i| state[dofs[i]]);
        let (q, _, _) = Self::coordinates(data, &u);
        (0..24)
            .map(|i| {
                (0..24)
                    .map(|j| 0.5 * q[i] * data.stiffness[24 * i + j] * q[j])
                    .sum::<f64>()
            })
            .sum()
    }
}

impl NonlinearElementForm<f64> for CorotationalQuadShellForm {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }
    fn num_local_dofs(&self, _: usize) -> usize {
        24
    }
    fn num_global_dofs(&self) -> usize {
        self.n_dofs
    }
    fn local_to_global(&self, index: usize) -> Vec<usize> {
        self.elements[index].element.global_dofs().to_vec()
    }
    fn element_residual(&self, index: usize, state: &[f64]) -> Vec<f64> {
        let data = &self.elements[index];
        let dofs = data.element.global_dofs();
        Self::force(data, &std::array::from_fn(|i| state[dofs[i]]))
    }
    fn element_tangent(&self, index: usize, state: &[f64]) -> Vec<f64> {
        let data = &self.elements[index];
        let dofs = data.element.global_dofs();
        let u: [f64; 24] = std::array::from_fn(|i| state[dofs[i]]);
        let mut result = vec![0.0; 24 * 24];
        for column in 0..24 {
            let h = 1e-7 * (1.0 + u[column].abs());
            let mut plus = u;
            let mut minus = u;
            plus[column] += h;
            minus[column] -= h;
            let a = Self::force(data, &plus);
            let b = Self::force(data, &minus);
            for row in 0..24 {
                result[24 * row + column] = (a[row] - b[row]) / (2.0 * h);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::ShellSection;
    use super::*;
    use crate::mesh::Point;

    fn element(warp: bool) -> QuadShellElement {
        let p = [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.8, if warp { 0.08 } else { 0.0 }),
            Point::new_3d(0.0, 0.8, 0.0),
        ];
        QuadShellElement::from_points_with_section(
            [0, 1, 2, 3],
            &p[0],
            &p[1],
            &p[2],
            &p[3],
            ShellSection::new(200e9, 0.3, 0.002, 7800.0),
        )
    }

    #[test]
    fn zero_state_and_linear_tangent_match_native_mitc4() {
        let elem = element(false);
        let linear = elem.stiffness_global();
        let form = CorotationalQuadShellForm::new(vec![elem], 24);
        let u = [0.0; 24];
        assert!(form.element_residual(0, &u).iter().all(|v| v.abs() < 1e-6));
        let tangent = form.element_tangent(0, &u);
        let error = tangent
            .iter()
            .zip(&linear)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / linear.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(error < 2e-7, "linear-limit relative error {error:e}");
    }

    #[test]
    fn warped_quad_force_matches_energy_gradient_and_symmetric_tangent() {
        let form = CorotationalQuadShellForm::new(vec![element(true)], 24);
        let mut u = [0.0; 24];
        u[6] = 0.015;
        u[13] = -0.01;
        u[2] = 0.05;
        u[20] = -0.02;
        u[3] = 0.1;
        u[10] = -0.08;
        let force = form.element_residual(0, &u);
        let derivative = (0..24)
            .map(|i| {
                let mut a = u;
                let mut b = u;
                let h = 1e-7;
                a[i] += h;
                b[i] -= h;
                (form.element_strain_energy(0, &a) - form.element_strain_energy(0, &b)) / (2.0 * h)
            })
            .collect::<Vec<_>>();
        let error = force
            .iter()
            .zip(&derivative)
            .map(|(f, d)| (f - d).powi(2))
            .sum::<f64>()
            .sqrt()
            / derivative.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(error < 2e-7, "virtual-work error {error:e}");
        let tangent = form.element_tangent(0, &u);
        let skew = (0..24)
            .flat_map(|i| (0..24).map(move |j| (i, j)))
            .map(|(i, j)| (tangent[24 * i + j] - tangent[24 * j + i]).powi(2))
            .sum::<f64>()
            .sqrt()
            / tangent.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(skew < 2e-7, "tangent asymmetry {skew:e}");
    }

    #[test]
    fn world_origin_and_rigid_translation_preserve_warped_quad_forces() {
        let original = element(true);
        let mut distant = original.clone();
        distant.origin = [1e12, -1e12, 1e12];
        let reference = CorotationalQuadShellForm::new(vec![original], 24);
        let distant = CorotationalQuadShellForm::new(vec![distant], 24);
        let mut u = [0.0; 24];
        u[6] = 1.0 / 128.0;
        u[13] = -1.0 / 64.0;
        u[20] = 1.0 / 256.0;
        u[10] = 1.0 / 128.0;
        let force = reference.element_residual(0, &u);
        assert!(force.iter().any(|v| v.abs() > 1.0));
        assert_eq!(distant.element_residual(0, &u), force);
        for node in 0..4 {
            for component in 0..3 {
                u[6 * node + component] += 1e8;
            }
        }
        assert_eq!(reference.element_residual(0, &u), force);
        assert_eq!(distant.element_residual(0, &u), force);
    }
}
