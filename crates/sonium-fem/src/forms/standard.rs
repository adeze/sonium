//! Standard bilinear and linear forms
//!
//! Reusable operator building blocks shared across physics modules.
//! Physics-specific forms delegate to these via thin wrappers that
//! preserve domain-specific constructors.

use crate::spaces::BasisTabulation;

use super::traits::{BilinearForm, LinearForm, LocalMatrix, LocalVector};

// ---------------------------------------------------------------------------
// Scalar H1 forms
// ---------------------------------------------------------------------------

/// Laplacian (stiffness) bilinear form: `a(u,v) = integral(grad_u . grad_v) dOmega`
///
/// Assembles the standard stiffness matrix for any elliptic PDE.
/// Works with scalar H1 function spaces.
#[derive(Debug, Clone, Copy)]
pub struct LaplacianForm;

impl BilinearForm<f64> for LaplacianForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n_test {
            for j in 0..n_trial {
                let mut grad_dot = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                }
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

/// Coefficient-weighted Laplacian: `a(u,v) = integral(c * grad_u . grad_v) dOmega`
///
/// Generalises `LaplacianForm` with a constant scalar coefficient.
/// Use this for thermal conduction (`c = k`), magnetic permeability (`c = mu`), etc.
#[derive(Debug, Clone, Copy)]
pub struct WeightedLaplacianForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for WeightedLaplacianForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.coefficient;

        for i in 0..n_test {
            for j in 0..n_trial {
                let mut grad_dot = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                }
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

/// Spatially varying Laplacian:
/// a(u,v) = integral(c(x,y,z) * grad_u . grad_v) dOmega.
///
/// This is the scalar diffusion/Sturm--Liouville building block used when a
/// material coefficient cannot be represented by one constant per domain.
pub struct VariableLaplacianForm<F> {
    /// Coefficient evaluated at every physical quadrature point.
    pub coefficient_fn: F,
}

impl<F> VariableLaplacianForm<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Construct a variable-coefficient Laplacian form.
    pub fn new(coefficient_fn: F) -> Self {
        Self { coefficient_fn }
    }
}

impl<F> BilinearForm<f64> for VariableLaplacianForm<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let coefficient = (self.coefficient_fn)(x, y, z);
        let w = weight * test.det_j * coefficient;

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

/// Mass bilinear form: `a(u,v) = integral(u * v) dOmega`
///
/// Assembles the standard mass matrix. Works with scalar H1 function spaces.
#[derive(Debug, Clone, Copy)]
pub struct MassForm;

impl BilinearForm<f64> for MassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let w = weight * test.det_j;

        for i in 0..n_test {
            for j in 0..n_trial {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Coefficient-weighted mass: `a(u,v) = integral(c * u * v) dOmega`
///
/// Generalises `MassForm` with a constant scalar coefficient.
/// Use for heat capacity (`c = rho*cp`), convection BC (`c = h`), etc.
#[derive(Debug, Clone, Copy)]
pub struct WeightedMassForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for WeightedMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let w = weight * test.det_j * self.coefficient;

        for i in 0..n_test {
            for j in 0..n_trial {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Spatially varying mass/reaction form:
/// a(u,v) = integral(c(x,y,z) * u * v) dOmega.
///
/// The same form represents a potential on the stiffness side or a weight on
/// the mass side of a generalized eigenproblem.
pub struct VariableMassForm<F> {
    /// Coefficient evaluated at every physical quadrature point.
    pub coefficient_fn: F,
}

impl<F> VariableMassForm<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Construct a variable-coefficient mass form.
    pub fn new(coefficient_fn: F) -> Self {
        Self { coefficient_fn }
    }
}

impl<F> BilinearForm<f64> for VariableMassForm<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let coefficient = (self.coefficient_fn)(x, y, z);
        let w = weight * test.det_j * coefficient;

        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Source term linear form: `l(v) = integral(f * v) dOmega`
///
/// The source function `f(x, y, z)` is provided as a closure.
pub struct SourceForm<F: Fn(f64, f64, f64) -> f64 + Send + Sync> {
    pub source_fn: F,
}

impl<F: Fn(f64, f64, f64) -> f64 + Send + Sync> SourceForm<F> {
    pub fn new(source_fn: F) -> Self {
        Self { source_fn }
    }
}

impl<F: Fn(f64, f64, f64) -> f64 + Send + Sync> LinearForm<f64> for SourceForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let f_val = (self.source_fn)(x, y, z);
        let w = weight * test.det_j;

        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * f_val * w);
        }
    }
}

// ---------------------------------------------------------------------------
// Vector H1 forms
// ---------------------------------------------------------------------------

/// Vector Laplacian: `a(u,v) = integral(c * grad(u) : grad(v)) dOmega`
///
/// Component-gated gradient-dot-gradient on `VectorH1Space`.
/// Only accumulates when test and trial DOFs belong to the same component.
/// Use for Stokes viscous term (`c = mu`), vector diffusion, etc.
#[derive(Debug, Clone, Copy)]
pub struct VectorLaplacianForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for VectorLaplacianForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let w = self.coefficient * weight * test.det_j;

        for i in 0..n_test {
            let ci = i % dim;
            for j in 0..n_trial {
                let cj = j % dim;
                if ci != cj {
                    continue;
                }
                let mut grad_dot = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                }
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

/// Vector mass: `a(u,v) = integral(c * u . v) dOmega`
///
/// Component-gated mass on `VectorH1Space`.
/// Only accumulates when test and trial DOFs belong to the same component.
/// Use for structural mass (`c = rho`), inertia terms, etc.
#[derive(Debug, Clone, Copy)]
pub struct VectorMassForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for VectorMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.coefficient;

        for a in 0..test.n_dofs {
            for b in 0..trial.n_dofs {
                let dot_product = (0..dim)
                    .map(|component| {
                        test.values[a * dim + component] * trial.values[b * dim + component]
                    })
                    .sum::<f64>();
                local_matrix.add(a, b, dot_product * w);
            }
        }
    }
}

/// H(div) divergence-divergence form: `a(u,v) = integral(div(u) * div(v)) dOmega`.
///
/// RT0/RT1 tabulations store one physical divergence per basis function. Keeping
/// this form separate from [`DivergenceForm`] avoids interpreting a scalar
/// divergence as a vector H1 gradient component.
#[derive(Debug, Clone, Copy)]
pub struct DivergenceDivergenceForm;

impl BilinearForm<f64> for DivergenceDivergenceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.divergence(i) * trial.divergence(j) * w);
            }
        }
    }
}

/// Mixed H(div)-scalar divergence coupling:
/// `b(u,p) = -integral(p * div(u)) dOmega`.
///
/// This H(div)-specific counterpart to [`DivergenceForm`] reads the scalar
/// divergence stored by [`BasisTabulation::divergence`].
#[derive(Debug, Clone, Copy)]
pub struct HdivDivergenceForm;

impl BilinearForm<f64> for HdivDivergenceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, -test.divergence(i) * trial.value(j) * w);
            }
        }
    }
}

/// Divergence coupling form: `b(v,p) = -integral(p * div(v)) dOmega`
///
/// Rectangular form: test = `VectorH1Space` (velocity), trial = scalar `H1Space` (pressure).
/// For vector DOF `i` with component `c_i` and scalar DOF `j`:
/// `B[i,j] = -phi_j * (dphi_i / dx_{c_i}) * w * |J|`
#[derive(Debug, Clone, Copy)]
pub struct DivergenceForm;

impl BilinearForm<f64> for DivergenceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let w = weight * test.det_j;

        for i in 0..n_test {
            let ci = i % dim;
            let div_vi = test.grad(i, ci);
            for j in 0..n_trial {
                local_matrix.add(i, j, -trial.value(j) * div_vi * w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// H(curl) forms
// ---------------------------------------------------------------------------

/// Curl-curl bilinear form: `a(E,F) = integral(curl(E) . curl(F)) dOmega`
///
/// Works with `HcurlSpace` (Nedelec edge elements).
/// In 2D the curl is a scalar; in 3D it is a 3-vector.
#[derive(Debug, Clone, Copy)]
pub struct CurlCurlForm;

impl BilinearForm<f64> for CurlCurlForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        match dim {
            2 => {
                for i in 0..n_test {
                    for j in 0..n_trial {
                        let val = test.derivatives[i] * trial.derivatives[j] * w;
                        local_matrix.add(i, j, val);
                    }
                }
            }
            3 => {
                for i in 0..n_test {
                    for j in 0..n_trial {
                        let dot: f64 = (0..3)
                            .map(|d| test.derivatives[i * 3 + d] * trial.derivatives[j * 3 + d])
                            .sum();
                        local_matrix.add(i, j, dot * w);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Coefficient-weighted curl-curl: `a(E,F) = integral(c * curl(E) . curl(F)) dOmega`
///
/// Use for `1/mu` coefficient in Maxwell/eddy-current formulations.
#[derive(Debug, Clone, Copy)]
pub struct WeightedCurlCurlForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for WeightedCurlCurlForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.coefficient;

        match dim {
            2 => {
                for i in 0..n_test {
                    for j in 0..n_trial {
                        let val = test.derivatives[i] * trial.derivatives[j] * w;
                        local_matrix.add(i, j, val);
                    }
                }
            }
            3 => {
                for i in 0..n_test {
                    for j in 0..n_trial {
                        let dot: f64 = (0..3)
                            .map(|d| test.derivatives[i * 3 + d] * trial.derivatives[j * 3 + d])
                            .sum();
                        local_matrix.add(i, j, dot * w);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Edge-element mass form: `a(E,F) = integral(E . F) dOmega`
///
/// Mass matrix for `HcurlSpace` (vector dot product of edge basis functions).
#[derive(Debug, Clone, Copy)]
pub struct EdgeMassForm;

impl BilinearForm<f64> for EdgeMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n_test {
            for j in 0..n_trial {
                let dot: f64 = (0..dim)
                    .map(|d| test.values[i * dim + d] * trial.values[j * dim + d])
                    .sum();
                local_matrix.add(i, j, dot * w);
            }
        }
    }
}

/// Coefficient-weighted edge mass: `a(E,F) = integral(c * E . F) dOmega`
///
/// Use for conductivity-weighted mass (`c = sigma`) in eddy-current formulations.
#[derive(Debug, Clone, Copy)]
pub struct WeightedEdgeMassForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for WeightedEdgeMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.coefficient;

        for i in 0..n_test {
            for j in 0..n_trial {
                let dot: f64 = (0..dim)
                    .map(|d| test.values[i * dim + d] * trial.values[j * dim + d])
                    .sum();
                local_matrix.add(i, j, dot * w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space, HcurlSpace, VectorH1Space};

    fn assert_rectangular_form_nonzero<F: BilinearForm<f64>>(
        form: &F,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        label: &str,
    ) {
        assert_ne!(
            test.n_dofs, trial.n_dofs,
            "{label} fixture must be rectangular"
        );
        let mut matrix = LocalMatrix::zeros(test.n_dofs.max(trial.n_dofs));
        form.integrate_point(test, trial, 1.0, &mut matrix);

        let mut nonzero = 0;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let value = matrix.get(i, j);
                assert!(
                    value.is_finite(),
                    "{label} produced a non-finite entry at ({i},{j})"
                );
                nonzero += usize::from(value.abs() > 1.0e-14);
            }
        }
        assert!(nonzero > 0, "{label} rectangular block must be nonzero");
    }

    #[test]
    fn scalar_standard_forms_support_rectangular_tabulations() {
        let mesh = unit_square_triangles(1);
        let p1 = H1Space::new(PolynomialDegree::P1).tabulate(&mesh, 0, 0.2, 0.3, 0.0);
        let p2 = H1Space::new(PolynomialDegree::P2).tabulate(&mesh, 0, 0.2, 0.3, 0.0);

        assert_rectangular_form_nonzero(&LaplacianForm, &p1, &p2, "LaplacianForm");
        assert_rectangular_form_nonzero(
            &WeightedLaplacianForm { coefficient: 2.0 },
            &p1,
            &p2,
            "WeightedLaplacianForm",
        );
        assert_rectangular_form_nonzero(&MassForm, &p1, &p2, "MassForm");
        assert_rectangular_form_nonzero(
            &WeightedMassForm { coefficient: 2.0 },
            &p1,
            &p2,
            "WeightedMassForm",
        );
    }

    #[test]
    fn vector_standard_forms_support_rectangular_tabulations() {
        let mesh = unit_square_triangles(1);
        let p1 = VectorH1Space::new(PolynomialDegree::P1, 2).tabulate(&mesh, 0, 0.2, 0.3, 0.0);
        let p2 = VectorH1Space::new(PolynomialDegree::P2, 2).tabulate(&mesh, 0, 0.2, 0.3, 0.0);

        assert_rectangular_form_nonzero(
            &VectorLaplacianForm { coefficient: 1.0 },
            &p1,
            &p2,
            "VectorLaplacianForm",
        );
        assert_rectangular_form_nonzero(
            &VectorMassForm { coefficient: 1.0 },
            &p1,
            &p2,
            "VectorMassForm",
        );
    }

    #[test]
    fn hcurl_standard_forms_support_rectangular_tabulations() {
        let mesh = unit_square_triangles(1);
        let order1 = HcurlSpace::new().tabulate(&mesh, 0, 0.2, 0.3, 0.0);
        let order2 = HcurlSpace::order2().tabulate(&mesh, 0, 0.2, 0.3, 0.0);

        assert_rectangular_form_nonzero(&CurlCurlForm, &order1, &order2, "CurlCurlForm");
        assert_rectangular_form_nonzero(
            &WeightedCurlCurlForm { coefficient: 2.0 },
            &order1,
            &order2,
            "WeightedCurlCurlForm",
        );
        assert_rectangular_form_nonzero(&EdgeMassForm, &order1, &order2, "EdgeMassForm");
        assert_rectangular_form_nonzero(
            &WeightedEdgeMassForm { coefficient: 2.0 },
            &order1,
            &order2,
            "WeightedEdgeMassForm",
        );
    }

    #[test]
    fn test_laplacian_form_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);

        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;
        let mut mat = LocalMatrix::zeros(n);

        LaplacianForm.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (mat.get(i, j) - mat.get(j, i)).abs() < 1e-14,
                    "Laplacian not symmetric at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_mass_form_positive() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);

        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;
        let mut mat = LocalMatrix::zeros(n);

        MassForm.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            assert!(mat.get(i, i) > 0.0, "Mass diagonal negative at {i}");
        }
    }

    #[test]
    fn test_weighted_laplacian_scales() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut m1 = LocalMatrix::zeros(n);
        LaplacianForm.integrate_point(&tab, &tab, 1.0, &mut m1);

        let mut m2 = LocalMatrix::zeros(n);
        let form = WeightedLaplacianForm { coefficient: 42.0 };
        form.integrate_point(&tab, &tab, 1.0, &mut m2);

        for i in 0..n {
            for j in 0..n {
                let expected = m1.get(i, j) * 42.0;
                assert!(
                    (m2.get(i, j) - expected).abs() < 1e-12,
                    "WeightedLaplacian scaling wrong at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_weighted_mass_scales() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut m1 = LocalMatrix::zeros(n);
        MassForm.integrate_point(&tab, &tab, 1.0, &mut m1);

        let mut m2 = LocalMatrix::zeros(n);
        let form = WeightedMassForm { coefficient: 7.5 };
        form.integrate_point(&tab, &tab, 1.0, &mut m2);

        for i in 0..n {
            for j in 0..n {
                let expected = m1.get(i, j) * 7.5;
                assert!(
                    (m2.get(i, j) - expected).abs() < 1e-12,
                    "WeightedMass scaling wrong at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn variable_mass_integrates_physical_coordinate() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);
        let form = VariableMassForm::new(|x, _, _| x);
        let matrix = crate::assembly::assemble_bilinear(&form, &space, &space, &mesh, 3).to_csr();
        let integral: f64 = (0..matrix.num_rows)
            .map(|row| matrix.row_entries(row).map(|(_, value)| value).sum::<f64>())
            .sum();
        assert!((integral - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn test_vector_laplacian_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = VectorLaplacianForm { coefficient: 1.0 };
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                assert!(
                    (mat.get(i, j) - mat.get(j, i)).abs() < 1e-14,
                    "VectorLaplacian not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_vector_mass_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = VectorMassForm { coefficient: 1.0 };
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                assert!(
                    (mat.get(i, j) - mat.get(j, i)).abs() < 1e-14,
                    "VectorMass not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_divergence_form_nonzero() {
        let mesh = unit_square_triangles(2);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);

        let v_tab = v_space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let p_tab = p_space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        let n = v_tab.n_dofs.max(p_tab.n_dofs);
        let mut mat = LocalMatrix::zeros(n);
        DivergenceForm.integrate_point(&v_tab, &p_tab, 1.0, &mut mat);

        let any_nonzero =
            (0..v_tab.n_dofs).any(|i| (0..p_tab.n_dofs).any(|j| mat.get(i, j).abs() > 1e-15));
        assert!(
            any_nonzero,
            "DivergenceForm should produce non-zero entries"
        );
    }

    #[test]
    fn test_curl_curl_form_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut mat = LocalMatrix::zeros(n);
        CurlCurlForm.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (mat.get(i, j) - mat.get(j, i)).abs();
                assert!(diff < 1e-14, "CurlCurl not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_weighted_curl_curl_scales() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut m1 = LocalMatrix::zeros(n);
        CurlCurlForm.integrate_point(&tab, &tab, 1.0, &mut m1);

        let mut m2 = LocalMatrix::zeros(n);
        let form = WeightedCurlCurlForm { coefficient: 3.0 };
        form.integrate_point(&tab, &tab, 1.0, &mut m2);

        for i in 0..n {
            for j in 0..n {
                let expected = m1.get(i, j) * 3.0;
                assert!(
                    (m2.get(i, j) - expected).abs() < 1e-12,
                    "WeightedCurlCurl scaling wrong at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_edge_mass_form_positive_diagonal() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut mat = LocalMatrix::zeros(n);
        EdgeMassForm.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            assert!(
                mat.get(i, i) > 0.0,
                "EdgeMass diagonal {i} should be positive"
            );
        }
    }

    #[test]
    fn test_weighted_edge_mass_scales() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut m1 = LocalMatrix::zeros(n);
        EdgeMassForm.integrate_point(&tab, &tab, 1.0, &mut m1);

        let mut m2 = LocalMatrix::zeros(n);
        let form = WeightedEdgeMassForm { coefficient: 5.0 };
        form.integrate_point(&tab, &tab, 1.0, &mut m2);

        for i in 0..n {
            for j in 0..n {
                let expected = m1.get(i, j) * 5.0;
                assert!(
                    (m2.get(i, j) - expected).abs() < 1e-12,
                    "WeightedEdgeMass scaling wrong at ({i},{j})"
                );
            }
        }
    }
}
