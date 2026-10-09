//! Helmholtz bilinear form for acoustic wave propagation
//!
//! The Helmholtz equation: `nabla^2 u + k^2 u = f`
//! Weak form: `a(u,v) = integral(grad_u . grad_v) dOmega - k^2 integral(u v) dOmega`

use crate::boundary::PmlRegion;
use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::mesh::Point;
use crate::spaces::BasisTabulation;
use num_complex::Complex64;

/// Helmholtz bilinear form: `a(u,v) = integral(grad_u . grad_v - k^2 u v) dOmega`
///
/// For frequency sweeps via [`AcousticsAssembler`](super::AcousticsAssembler),
/// the stiffness and mass components are assembled separately using
/// [`LaplacianForm`](crate::forms::LaplacianForm) and [`MassForm`](crate::forms::MassForm),
/// then combined as `K - k^2 M` at each wavenumber. This form is for one-shot assembly
/// at a fixed wavenumber.
#[derive(Debug, Clone)]
pub struct HelmholtzForm {
    /// Wavenumber k (may be complex for lossy media)
    pub wavenumber: Complex64,
}

impl HelmholtzForm {
    pub fn new(wavenumber: Complex64) -> Self {
        Self { wavenumber }
    }

    pub fn from_real(k: f64) -> Self {
        Self {
            wavenumber: Complex64::new(k, 0.0),
        }
    }
}

impl BilinearForm<Complex64> for HelmholtzForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let k_sq = self.wavenumber * self.wavenumber;
        let w = weight * test.det_j;

        for i in 0..n {
            for j in 0..n {
                // Stiffness: grad(phi_i) . grad(phi_j)
                let mut grad_dot = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                }
                // Mass: phi_i * phi_j
                let mass = test.value(i) * trial.value(j);

                let val = Complex64::new(grad_dot * w, 0.0) - k_sq * Complex64::new(mass * w, 0.0);
                local_matrix.add(i, j, val);
            }
        }
    }
}

/// Helmholtz form with element-local PML coordinate stretching.
///
/// In PML regions the scalar Helmholtz weak form is modified as:
///
/// ```text
/// ∫ (sy*sz/sx) ux vx + (sx*sz/sy) uy vy + (sx*sy/sz) uz vz
///   - k^2 (sx*sy*sz) u v dΩ
/// ```
///
/// Outside all supplied regions this reduces to [`HelmholtzForm`]. If multiple
/// regions overlap, the first matching region is used; use a combined
/// `PmlDirection::XY`, `XZ`, `YZ`, or `XYZ` region for corners when possible.
#[derive(Debug, Clone)]
pub struct PmlHelmholtzForm {
    /// Wavenumber k.
    pub wavenumber: Complex64,
    /// PML regions used during assembly.
    pub regions: Vec<PmlRegion>,
}

impl PmlHelmholtzForm {
    pub fn new(wavenumber: Complex64, regions: Vec<PmlRegion>) -> Self {
        Self {
            wavenumber,
            regions,
        }
    }

    pub fn from_real(k: f64, regions: Vec<PmlRegion>) -> Self {
        Self {
            wavenumber: Complex64::new(k, 0.0),
            regions,
        }
    }
}

impl BilinearForm<Complex64> for PmlHelmholtzForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let k_sq = self.wavenumber * self.wavenumber;
        let w = weight * test.det_j;
        let [x, y, z] = test.physical_coords;
        let point = if dim == 2 {
            Point::new_2d(x, y)
        } else {
            Point::new_3d(x, y, z)
        };
        let coeffs = pml_coefficients(&point, &self.regions);

        for i in 0..n {
            for j in 0..n {
                let mut stiffness = Complex64::new(0.0, 0.0);
                for d in 0..dim {
                    stiffness += coeffs.stiffness[d]
                        * Complex64::new(test.grad(i, d) * trial.grad(j, d) * w, 0.0);
                }

                let mass = test.value(i) * trial.value(j);
                let val = stiffness - k_sq * coeffs.mass * Complex64::new(mass * w, 0.0);
                local_matrix.add(i, j, val);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PmlCoefficients {
    stiffness: [Complex64; 3],
    mass: Complex64,
}

fn pml_coefficients(point: &Point, regions: &[PmlRegion]) -> PmlCoefficients {
    for region in regions {
        if region.contains(point) {
            return PmlCoefficients {
                stiffness: [
                    region.stiffness_coefficient_x(point),
                    region.stiffness_coefficient_y(point),
                    region.stiffness_coefficient_z(point),
                ],
                mass: region.mass_coefficient(point),
            };
        }
    }

    PmlCoefficients {
        stiffness: [Complex64::new(1.0, 0.0); 3],
        mass: Complex64::new(1.0, 0.0),
    }
}

/// Complex-valued source term for Helmholtz: `l(v) = integral(f * v) dOmega`
pub struct ComplexSourceForm<F: Fn(f64, f64, f64) -> Complex64 + Send + Sync> {
    pub source_fn: F,
}

impl<F: Fn(f64, f64, f64) -> Complex64 + Send + Sync> ComplexSourceForm<F> {
    pub fn new(source_fn: F) -> Self {
        Self { source_fn }
    }
}

impl<F: Fn(f64, f64, f64) -> Complex64 + Send + Sync> LinearForm<Complex64>
    for ComplexSourceForm<F>
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<Complex64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let f_val = (self.source_fn)(x, y, z);
        let w = weight * test.det_j;

        for i in 0..test.n_dofs {
            local_vector.add(i, Complex64::new(test.value(i) * w, 0.0) * f_val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space};

    #[test]
    fn test_helmholtz_form_reduces_to_laplacian_at_k0() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        // At k=0, Helmholtz = Laplacian
        let form = HelmholtzForm::from_real(0.0);
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        // Compare with LaplacianForm
        let mut lap_mat = LocalMatrix::zeros(n);
        crate::forms::LaplacianForm.integrate_point(&tab, &tab, 1.0, &mut lap_mat);

        for i in 0..n {
            for j in 0..n {
                let h = mat.get(i, j);
                let l = lap_mat.get(i, j);
                assert!(
                    (h.re - l).abs() < 1e-14 && h.im.abs() < 1e-14,
                    "Mismatch at ({i},{j}): helmholtz={h}, laplacian={l}"
                );
            }
        }
    }

    #[test]
    fn test_helmholtz_form_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = HelmholtzForm::from_real(5.0);
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                let a = mat.get(i, j);
                let b = mat.get(j, i);
                assert!(
                    (a - b).norm() < 1e-14,
                    "Not symmetric at ({i},{j}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_pml_helmholtz_reduces_to_standard_outside_regions() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let pml = crate::boundary::PmlRegion::x_positive(10.0, 1.0, 8.0, 2.0);
        let standard = HelmholtzForm::from_real(2.0);
        let pml_form = PmlHelmholtzForm::from_real(2.0, vec![pml]);
        let mut standard_mat = LocalMatrix::zeros(n);
        let mut pml_mat = LocalMatrix::zeros(n);

        standard.integrate_point(&tab, &tab, 1.0, &mut standard_mat);
        pml_form.integrate_point(&tab, &tab, 1.0, &mut pml_mat);

        for i in 0..n {
            for j in 0..n {
                assert!((standard_mat.get(i, j) - pml_mat.get(i, j)).norm() < 1e-14);
            }
        }
    }

    #[test]
    fn test_pml_helmholtz_adds_complex_stretching_inside_region() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let pml = crate::boundary::PmlRegion::x_positive(0.0, 1.0, 8.0, 2.0);
        let pml_form = PmlHelmholtzForm::from_real(2.0, vec![pml]);
        let mut mat = LocalMatrix::zeros(n);

        pml_form.integrate_point(&tab, &tab, 1.0, &mut mat);

        let has_imaginary = (0..n).any(|i| (0..n).any(|j| mat.get(i, j).im.abs() > 1e-14));
        assert!(has_imaginary);
    }
}
