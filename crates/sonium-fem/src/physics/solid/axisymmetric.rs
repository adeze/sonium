//! Shared axisymmetric weak forms for meridional `(r,z)` FEM models.
//!
//! The domain integral is `2*pi*int r (...) dr dz`. Existing acoustics and
//! magnetostatics modules have specialised axisymmetric forms; this module
//! provides reusable multiphysics pieces for coupled speaker and motor studies.

use std::f64::consts::PI;

use num_complex::Complex64;

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

use super::elasticity::IsotropicMaterial;

/// Return the axisymmetric quadrature factor `2*pi*r*detJ*weight`.
pub fn axisymmetric_weight(tab: &BasisTabulation, weight: f64, axis_epsilon: f64) -> f64 {
    let r = tab.physical_coords[0].abs().max(axis_epsilon);
    2.0 * PI * r * tab.det_j * weight
}

/// Scalar mass form with full `2*pi*r` meridional weighting.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricScalarMassForm {
    pub coefficient: f64,
    pub axis_epsilon: f64,
}

/// Vector mass form with full `2*pi*r` meridional weighting.
///
/// Component gating is essential for `(u_r, u_z)` spaces: inertia couples
/// equal components through the scalar shape functions, never radial and
/// axial components directly.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricVectorMassForm {
    pub coefficient: f64,
    pub axis_epsilon: f64,
}

impl AxisymmetricVectorMassForm {
    pub fn new(coefficient: f64) -> Self {
        Self {
            coefficient,
            axis_epsilon: 1.0e-12,
        }
    }
}

impl BilinearForm<f64> for AxisymmetricVectorMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = axisymmetric_weight(test, weight, self.axis_epsilon) * self.coefficient;
        for i in 0..test.n_dofs {
            let test_component = i % 2;
            for j in 0..trial.n_dofs {
                if test_component == j % 2 {
                    local_matrix.add(
                        i,
                        j,
                        test.vector_value(i, test_component) * trial.vector_value(j, j % 2) * w,
                    );
                }
            }
        }
    }
}

impl AxisymmetricScalarMassForm {
    pub fn new(coefficient: f64) -> Self {
        Self {
            coefficient,
            axis_epsilon: 1.0e-12,
        }
    }
}

impl BilinearForm<f64> for AxisymmetricScalarMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = axisymmetric_weight(test, weight, self.axis_epsilon) * self.coefficient;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Scalar gradient form with full `2*pi*r` meridional weighting.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricScalarGradientForm {
    pub coefficient: f64,
    pub axis_epsilon: f64,
}

impl AxisymmetricScalarGradientForm {
    pub fn new(coefficient: f64) -> Self {
        Self {
            coefficient,
            axis_epsilon: 1.0e-12,
        }
    }
}

impl BilinearForm<f64> for AxisymmetricScalarGradientForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = axisymmetric_weight(test, weight, self.axis_epsilon) * self.coefficient;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let grad_dot = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

/// Axisymmetric Helmholtz form `2*pi*r*(grad p.grad q - k^2 p q)`.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricPressureHelmholtzForm {
    pub wavenumber: Complex64,
    pub axis_epsilon: f64,
}

impl AxisymmetricPressureHelmholtzForm {
    pub fn new(wavenumber: Complex64) -> Self {
        Self {
            wavenumber,
            axis_epsilon: 1.0e-12,
        }
    }
}

impl BilinearForm<Complex64> for AxisymmetricPressureHelmholtzForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let w = axisymmetric_weight(test, weight, self.axis_epsilon);
        let k2 = self.wavenumber * self.wavenumber;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let grad_dot = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum::<f64>();
                let mass = test.value(i) * trial.value(j);
                local_matrix.add(i, j, Complex64::new(grad_dot * w, 0.0) - k2 * mass * w);
            }
        }
    }
}

/// Axisymmetric `(u_r,u_z)` elasticity including hoop strain `epsilon_tt = u_r/r`.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricElasticityForm {
    pub material: IsotropicMaterial,
    pub axis_epsilon: f64,
}

impl AxisymmetricElasticityForm {
    pub fn new(material: IsotropicMaterial) -> Self {
        Self {
            material,
            axis_epsilon: 1.0e-12,
        }
    }
}

impl BilinearForm<f64> for AxisymmetricElasticityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let r = test.physical_coords[0].abs().max(self.axis_epsilon);
        let w = axisymmetric_weight(test, weight, self.axis_epsilon);
        let lambda = self.material.lambda();
        let mu = self.material.mu();
        let c = [
            [lambda + 2.0 * mu, lambda, lambda, 0.0],
            [lambda, lambda + 2.0 * mu, lambda, 0.0],
            [lambda, lambda, lambda + 2.0 * mu, 0.0],
            [0.0, 0.0, 0.0, mu],
        ];

        for a in 0..test.n_dofs {
            let ba = axisymmetric_strain_column(test, a, r);
            for b in 0..trial.n_dofs {
                let bb = axisymmetric_strain_column(trial, b, r);
                let mut value = 0.0;
                for p in 0..4 {
                    for q in 0..4 {
                        value += ba[p] * c[p][q] * bb[q];
                    }
                }
                local_matrix.add(a, b, value * w);
            }
        }
    }
}

fn axisymmetric_strain_column(tab: &BasisTabulation, dof: usize, r: f64) -> [f64; 4] {
    let component = dof % 2;
    let phi = tab.vector_value(dof, component);
    let dr = tab.grad(dof, 0);
    let dz = tab.grad(dof, 1);
    match component {
        0 => [dr, 0.0, phi / r, dz],
        1 => [0.0, dz, 0.0, dr],
        _ => unreachable!(),
    }
}

/// Thin shell hoop membrane stiffness contribution for a meridional shell line.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricShellHoopForm {
    pub membrane_stiffness: f64,
    pub thickness: f64,
    pub axis_epsilon: f64,
}

impl AxisymmetricShellHoopForm {
    pub fn new(membrane_stiffness: f64, thickness: f64) -> Self {
        Self {
            membrane_stiffness,
            thickness,
            axis_epsilon: 1.0e-12,
        }
    }
}

impl BilinearForm<f64> for AxisymmetricShellHoopForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let r = test.physical_coords[0].abs().max(self.axis_epsilon);
        let w = axisymmetric_weight(test, weight, self.axis_epsilon)
            * self.membrane_stiffness
            * self.thickness
            / (r * r);
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Pressure-to-normal-displacement surface coupling with `2*pi*r` weighting.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricPressureNormalCouplingForm {
    pub normal: [f64; 2],
    pub coefficient: f64,
    pub axis_epsilon: f64,
}

impl AxisymmetricPressureNormalCouplingForm {
    pub fn new(normal: [f64; 2], coefficient: f64) -> Self {
        Self {
            normal,
            coefficient,
            axis_epsilon: 1.0e-12,
        }
    }
}

impl BilinearForm<f64> for AxisymmetricPressureNormalCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = axisymmetric_weight(test, weight, self.axis_epsilon) * self.coefficient;
        for p in 0..test.n_dofs {
            for u in 0..trial.n_dofs {
                let component = u % 2;
                local_matrix.add(
                    p,
                    u,
                    test.value(p) * trial.vector_value(u, component) * self.normal[component] * w,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 2.0, -1.0, 0.5],
            hessians: vec![],
            det_j: 0.5,
            physical_coords: [0.2, 0.4, 0.0],
        }
    }

    fn vector_tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 4,
            spatial_dim: 2,
            values: vec![0.25, 0.0, 0.0, 0.25, 0.75, 0.0, 0.0, 0.75],
            derivatives: vec![1.0, 2.0, 1.0, 2.0, -1.0, 0.5, -1.0, 0.5],
            hessians: vec![],
            det_j: 0.5,
            physical_coords: [0.2, 0.4, 0.0],
        }
    }

    #[test]
    fn scalar_mass_uses_full_two_pi_r_weight() {
        let tab = scalar_tab();
        let mut local = LocalMatrix::zeros(2);
        AxisymmetricScalarMassForm::new(3.0).integrate_point(&tab, &tab, 0.5, &mut local);
        let expected = 2.0 * PI * 0.2 * 0.5 * 0.5 * 3.0 * 0.25 * 0.75;
        assert!((local.get(0, 1) - expected).abs() < 1.0e-12);
    }

    #[test]
    fn vector_mass_is_axisymmetric_and_component_gated() {
        let tab = vector_tab();
        let mut local = LocalMatrix::zeros(4);
        AxisymmetricVectorMassForm::new(3.0).integrate_point(&tab, &tab, 0.5, &mut local);
        assert!(local.get(0, 2) > 0.0);
        assert!(local.get(1, 3) > 0.0);
        assert_eq!(local.get(0, 1), 0.0);
        assert_eq!(local.get(2, 3), 0.0);
    }

    #[test]
    fn pressure_normal_coupling_selects_vector_component() {
        let p = scalar_tab();
        let u = vector_tab();
        let mut local = LocalMatrix::zeros(4);
        AxisymmetricPressureNormalCouplingForm::new([0.0, 1.0], 2.0)
            .integrate_point(&p, &u, 0.5, &mut local);
        assert_eq!(local.get(0, 0), 0.0);
        assert!(local.get(0, 1) > 0.0);
    }

    #[test]
    fn axisymmetric_elasticity_includes_hoop_strain() {
        let tab = vector_tab();
        let mut local = LocalMatrix::zeros(4);
        let form = AxisymmetricElasticityForm::new(IsotropicMaterial::new(10.0, 0.25));
        form.integrate_point(&tab, &tab, 1.0, &mut local);
        assert!(local.get(0, 0) > 0.0);
        assert!(local.get(1, 1) > 0.0);
        assert!(local.get(0, 2).is_finite());
    }

    #[test]
    fn vector_basis_hoop_values_follow_interleaved_components() {
        let tab = vector_tab();
        assert!((axisymmetric_strain_column(&tab, 0, 0.2)[2] - 1.25).abs() < 1.0e-12);
        assert_eq!(axisymmetric_strain_column(&tab, 1, 0.2)[2], 0.0);
        assert!((axisymmetric_strain_column(&tab, 2, 0.2)[2] - 3.75).abs() < 1.0e-12);
        assert_eq!(axisymmetric_strain_column(&tab, 3, 0.2)[2], 0.0);
    }
}
