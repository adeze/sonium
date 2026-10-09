//! Mixed u-p weak forms.
//!
//! - [`DeviatoricElasticityForm`]: `a(u,v) = 2μ ∫ε(u):ε(v)` (VectorH1 × VectorH1)
//! - [`UpCouplingForm`]: `b(v,p) = +∫p div(v)` (VectorH1 test × H1 trial)
//! - [`UpCouplingTransposeForm`]: `c(q,u) = +∫q div(u)` (H1 test × VectorH1 trial)
//! - [`PressureComplianceForm`]: `d(p,q) = −c ∫p q` (H1 × H1)

use crate::forms::{BilinearForm, LocalMatrix, WeightedMassForm};
use crate::physics::solid::elasticity::{strain_col_2d, strain_col_3d, vector_component};
use crate::spaces::BasisTabulation;

// Rust guideline compliant 2026-02-21

/// Deviatoric elasticity: `a(u,v) = 2μ ∫ε(u):ε(v)`.
///
/// Uses the same Voigt B-matrix convention as [`ElasticityForm`](crate::physics::solid::elasticity::ElasticityForm)
/// with the deviatoric constitutive matrix `D = 2μ·diag(1,…,1/2,…)`.
#[derive(Debug, Clone, Copy)]
pub struct DeviatoricElasticityForm {
    /// Shear modulus in Pa.
    pub mu: f64,
}

impl DeviatoricElasticityForm {
    /// Build the deviatoric form for shear modulus `mu`.
    pub fn new(mu: f64) -> Self {
        Self { mu }
    }
}

impl BilinearForm<f64> for DeviatoricElasticityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        match dim {
            2 => {
                let n_dofs = test.n_dofs;
                for a in 0..n_dofs {
                    let b_a =
                        strain_col_2d(vector_component(a, 2), test.grad(a, 0), test.grad(a, 1));
                    for b in 0..n_dofs {
                        let b_b = strain_col_2d(
                            vector_component(b, 2),
                            trial.grad(b, 0),
                            trial.grad(b, 1),
                        );
                        // diag(1, 1, 1/2): engineering shear carries γ²/2.
                        let k = b_a[0] * b_b[0] + b_a[1] * b_b[1] + 0.5 * b_a[2] * b_b[2];
                        local_matrix.add(a, b, 2.0 * self.mu * k * w);
                    }
                }
            }
            3 => {
                let n_dofs = test.n_dofs;
                for a in 0..n_dofs {
                    let b_a = strain_col_3d(
                        vector_component(a, 3),
                        [test.grad(a, 0), test.grad(a, 1), test.grad(a, 2)],
                    );
                    for b in 0..n_dofs {
                        let b_b = strain_col_3d(
                            vector_component(b, 3),
                            [trial.grad(b, 0), trial.grad(b, 1), trial.grad(b, 2)],
                        );
                        let mut k = b_a[0] * b_b[0] + b_a[1] * b_b[1] + b_a[2] * b_b[2];
                        for component in 3..6 {
                            k += 0.5 * b_a[component] * b_b[component];
                        }
                        local_matrix.add(a, b, 2.0 * self.mu * k * w);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Velocity-pressure coupling: `b(v,p) = +∫p div(v)`.
///
/// Test space is vector (displacement), trial space is scalar (pressure).
#[derive(Debug, Clone, Copy)]
pub struct UpCouplingForm;

impl BilinearForm<f64> for UpCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            let divergence = test.grad(i, vector_component(i, dim));
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, trial.value(j) * divergence * w);
            }
        }
    }
}

/// Pressure-velocity coupling: `c(q,u) = +∫q div(u)`.
///
/// Test space is scalar (pressure), trial space is vector (displacement).
/// This is the transpose orientation of [`UpCouplingForm`], assembled
/// explicitly (not by transposition) so each block states its own
/// bilinearity.
#[derive(Debug, Clone, Copy)]
pub struct UpCouplingTransposeForm;

impl BilinearForm<f64> for UpCouplingTransposeForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = trial.spatial_dim;
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let divergence = trial.grad(j, vector_component(j, dim));
                local_matrix.add(i, j, test.value(i) * divergence * w);
            }
        }
    }
}

/// Pressure compliance: `d(p,q) = −c ∫p q` with `c = 1/λ`.
#[derive(Debug, Clone, Copy)]
pub struct PressureComplianceForm(WeightedMassForm);

impl PressureComplianceForm {
    /// Build the compliance form for volumetric compliance `c = 1/λ`.
    pub fn new(compliance: f64) -> Self {
        Self(WeightedMassForm {
            coefficient: -compliance,
        })
    }
}

impl BilinearForm<f64> for PressureComplianceForm {
    #[inline]
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        self.0.integrate_point(test, trial, weight, local_matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly;
    use crate::basis::PolynomialDegree;
    use crate::mesh::{box_mesh_tetrahedra, unit_square_triangles};
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

    fn max_asymmetry(csr: &sonium_solvers::CsrMatrix<f64>) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..csr.num_rows {
            for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
                let j = csr.col_indices[k];
                let mirror = csr.col_indices[csr.row_ptrs[j]..csr.row_ptrs[j + 1]]
                    .iter()
                    .zip(&csr.values[csr.row_ptrs[j]..csr.row_ptrs[j + 1]])
                    .find(|&(&col, _)| col == i)
                    .map(|(_, &v)| v)
                    .unwrap_or(0.0);
                worst = worst.max((csr.values[k] - mirror).abs());
            }
        }
        worst
    }

    #[test]
    fn deviatoric_block_is_symmetric_2d_and_3d() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let form = DeviatoricElasticityForm::new(1.0e6);
        let csr = assembly::assemble_bilinear(&form, &space, &space, &mesh, 4).to_csr();
        assert!(max_asymmetry(&csr) < 1.0e-12);

        let mesh = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
        let space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let csr = assembly::assemble_bilinear(&form, &space, &space, &mesh, 2).to_csr();
        assert!(max_asymmetry(&csr) < 1.0e-12);
    }

    #[test]
    fn coupling_blocks_have_expected_dimensions() {
        let mesh = unit_square_triangles(2);
        let u_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let system = assembly::assemble_mixed_full_checked(
            &DeviatoricElasticityForm::new(1.0e6),
            &UpCouplingForm,
            &UpCouplingTransposeForm,
            &PressureComplianceForm::new(1.0e-6),
            &u_space,
            &p_space,
            &mesh,
            4,
        )
        .expect("valid Taylor-Hood topology");
        let n_u = u_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        assert!(n_u > n_p);
        assert_eq!(
            (system.block_00.num_rows, system.block_00.num_cols),
            (n_u, n_u)
        );
        assert_eq!(
            (system.block_01.num_rows, system.block_01.num_cols),
            (n_u, n_p)
        );
        assert_eq!(
            (system.block_10.num_rows, system.block_10.num_cols),
            (n_p, n_u)
        );
        assert_eq!(
            (system.block_11.num_rows, system.block_11.num_cols),
            (n_p, n_p)
        );
        // Coupling blocks are transposes of each other.
        let b01 = system.block_01.to_csr();
        let b10 = system.block_10.to_csr();
        for i in 0..n_u {
            for k in b01.row_ptrs[i]..b01.row_ptrs[i + 1] {
                let j = b01.col_indices[k];
                let mirror = b10.col_indices[b10.row_ptrs[j]..b10.row_ptrs[j + 1]]
                    .iter()
                    .zip(&b10.values[b10.row_ptrs[j]..b10.row_ptrs[j + 1]])
                    .find(|&(&col, _)| col == i)
                    .map(|(_, &v)| v)
                    .unwrap_or(0.0);
                assert!((b01.values[k] - mirror).abs() < 1.0e-12);
            }
        }
        // Compliance block is negative semi-definite diagonal-mass-like.
        assert!(system.block_11.values.iter().all(|&v| v <= 0.0));
    }
}
