//! Linear elasticity bilinear form
//!
//! Weak form: `a(u,v) = integral C : epsilon(u) : epsilon(v) dOmega`
//! where `epsilon(u) = sym(grad u)` is the symmetric gradient (strain tensor)
//! and `C` is the elasticity tensor.
//!
//! Uses Voigt notation for efficient assembly:
//! - 2D: `[eps_xx, eps_yy, gamma_xy]` (3 components)
//! - 3D: `[eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]` (6 components)

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;
use num_complex::Complex64;

use super::materials::{IsotropicMaterial, ViscoelasticMaterial};

/// 2D elasticity assumption.
#[derive(Debug, Clone, Copy)]
pub enum PlaneAssumption {
    /// Plane stress: sigma_zz = 0 (thin plates)
    PlaneStress,
    /// Plane strain: eps_zz = 0 (long bodies)
    PlaneStrain,
}

/// Linear elasticity bilinear form: `a(u,v) = integral C:eps(u):eps(v) dOmega`
///
/// Works with [`VectorH1Space`](crate::spaces::VectorH1Space).
/// The `BasisTabulation` from `VectorH1Space` provides per-DOF gradients
/// in interleaved order `[node0_x, node0_y, node0_z, node1_x, ...]`.
#[derive(Debug, Clone)]
pub struct ElasticityForm {
    material: IsotropicMaterial,
    plane: Option<PlaneAssumption>,
}

impl ElasticityForm {
    /// Create a 3D elasticity form.
    pub fn new_3d(material: IsotropicMaterial) -> Self {
        Self {
            material,
            plane: None,
        }
    }

    /// Create a 2D elasticity form with the given plane assumption.
    pub fn new_2d(material: IsotropicMaterial, plane: PlaneAssumption) -> Self {
        Self {
            material,
            plane: Some(plane),
        }
    }

    /// Plane assumption for 2D assembly.
    ///
    /// # Panics
    ///
    /// Panics if the form was built with [`new_3d`](Self::new_3d) (no plane
    /// assumption) and is then assembled on a 2D mesh. Build the form with
    /// [`new_2d`](Self::new_2d) when the mesh is 2D.
    fn plane_2d(&self) -> PlaneAssumption {
        self.plane.expect(
            "ElasticityForm assembled on a 2D mesh without a plane assumption; \
             build the form with ElasticityForm::new_2d",
        )
    }
}

impl BilinearForm<f64> for ElasticityForm {
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
            2 => integrate_2d(self, test, trial, w, local_matrix),
            3 => integrate_3d(self, test, trial, w, local_matrix),
            _ => unreachable!(),
        }
    }
}

/// 2D elasticity integration using Voigt B-matrix.
///
/// Voigt strain: `[eps_xx, eps_yy, gamma_xy]`
///
/// For vector DOF `k = (node_i, component_c)` with `c in {0=x, 1=y}`:
/// - B[0, k] = dphi_i/dx  if c == 0, else 0    (eps_xx)
/// - B[1, k] = dphi_i/dy  if c == 1, else 0    (eps_yy)
/// - B[2, k] = dphi_i/dy  if c == 0  (gamma_xy, du_x/dy),
///   or dphi_i/dx  if c == 1  (gamma_xy, du_y/dx)
fn integrate_2d(
    form: &ElasticityForm,
    test: &BasisTabulation,
    trial: &BasisTabulation,
    w: f64,
    local_matrix: &mut LocalMatrix<f64>,
) {
    let c_voigt = match form.plane_2d() {
        PlaneAssumption::PlaneStress => form.material.voigt_2d_plane_stress(),
        PlaneAssumption::PlaneStrain => form.material.voigt_2d_plane_strain(),
    };

    // Vector DOFs are interleaved [node0_x, node0_y, node1_x, ...], matching
    // the `VectorH1Space` tabulation contract documented on `ElasticityForm`.
    debug_assert_eq!(test.n_dofs % 2, 0);
    debug_assert_eq!(test.derivatives.len(), test.n_dofs * 2);
    let n_dofs = test.n_dofs;

    // For each pair of vector DOFs (a, b), compute B_a^T C B_b * w
    for a in 0..n_dofs {
        let comp_a = vector_component(a, 2);
        let dphi_a_dx = test.grad(a, 0);
        let dphi_a_dy = test.grad(a, 1);

        // B_a column (3 Voigt components)
        let b_a = strain_col_2d(comp_a, dphi_a_dx, dphi_a_dy);

        for b in 0..n_dofs {
            let comp_b = vector_component(b, 2);
            let dphi_b_dx = trial.grad(b, 0);
            let dphi_b_dy = trial.grad(b, 1);

            let b_b = strain_col_2d(comp_b, dphi_b_dx, dphi_b_dy);

            // k_ab = B_a^T * C * B_b * w = sum_{p,q} B_a[p] * C[p][q] * B_b[q] * w
            let mut k = 0.0;
            for p in 0..3 {
                for q in 0..3 {
                    k += b_a[p] * c_voigt[p * 3 + q] * b_b[q];
                }
            }

            local_matrix.add(a, b, k * w);
        }
    }
}

/// 3D elasticity integration using Voigt B-matrix.
///
/// Voigt strain: `[eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]`
fn integrate_3d(
    form: &ElasticityForm,
    test: &BasisTabulation,
    trial: &BasisTabulation,
    w: f64,
    local_matrix: &mut LocalMatrix<f64>,
) {
    let c_voigt = form.material.voigt_3d();
    debug_assert_eq!(test.n_dofs % 3, 0);
    debug_assert_eq!(test.derivatives.len(), test.n_dofs * 3);
    let n_dofs = test.n_dofs;

    for a in 0..n_dofs {
        let comp_a = vector_component(a, 3);
        let dphi_a = [test.grad(a, 0), test.grad(a, 1), test.grad(a, 2)];
        let b_a = strain_col_3d(comp_a, dphi_a);

        for b in 0..n_dofs {
            let comp_b = vector_component(b, 3);
            let dphi_b = [trial.grad(b, 0), trial.grad(b, 1), trial.grad(b, 2)];
            let b_b = strain_col_3d(comp_b, dphi_b);

            let mut k = 0.0;
            for p in 0..6 {
                for q in 0..6 {
                    k += b_a[p] * c_voigt[p * 6 + q] * b_b[q];
                }
            }

            local_matrix.add(a, b, k * w);
        }
    }
}

/// Build column of the B-matrix for a 2D vector DOF.
///
/// Given component `c` (0=x, 1=y) and gradient `(dphi/dx, dphi/dy)`,
/// returns the 3-component Voigt strain column `[eps_xx, eps_yy, gamma_xy]`.
#[inline]
pub(crate) fn strain_col_2d(c: usize, dphi_dx: f64, dphi_dy: f64) -> [f64; 3] {
    match c {
        0 => [dphi_dx, 0.0, dphi_dy], // u_x: eps_xx = du_x/dx, gamma_xy = du_x/dy
        1 => [0.0, dphi_dy, dphi_dx], // u_y: eps_yy = du_y/dy, gamma_xy = du_y/dx
        _ => unreachable!(),
    }
}

/// Build column of the B-matrix for a 3D vector DOF.
///
/// Given component `c` (0=x, 1=y, 2=z) and gradient `[dphi/dx, dphi/dy, dphi/dz]`,
/// returns the 6-component Voigt strain column.
#[inline]
pub(crate) fn strain_col_3d(c: usize, dphi: [f64; 3]) -> [f64; 6] {
    //                    eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy
    match c {
        0 => [dphi[0], 0.0, 0.0, 0.0, dphi[2], dphi[1]],
        1 => [0.0, dphi[1], 0.0, dphi[2], 0.0, dphi[0]],
        2 => [0.0, 0.0, dphi[2], dphi[1], dphi[0], 0.0],
        _ => unreachable!(),
    }
}

/// Body force linear form: `l(v) = integral f . v dOmega`
///
/// `force_fn` returns the body force vector `(f_x, f_y)` or `(f_x, f_y, f_z)` at a point.
pub struct BodyForceForm<F: Fn(f64, f64, f64) -> Vec<f64> + Send + Sync> {
    pub force_fn: F,
}

impl<F: Fn(f64, f64, f64) -> Vec<f64> + Send + Sync> LinearForm<f64> for BodyForceForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let f = (self.force_fn)(x, y, z);
        let w = weight * test.det_j;
        let dim = test.spatial_dim;

        // Vector DOFs are interleaved: [node0_x, node0_y, ...]. A short
        // closure return only contributes its available components; require
        // the full vector in debug builds so truncation never goes unnoticed.
        debug_assert!(
            f.len() >= dim,
            "BodyForceForm force closure returned {} values for a {dim}D mesh",
            f.len()
        );
        let n_nodes = test.n_dofs / dim;
        for node in 0..n_nodes {
            for (c, &fc) in f.iter().enumerate().take(dim) {
                let dof = node * dim + c;
                let phi = test.vector_value(dof, c);
                local_vector.add(dof, phi * fc * w);
            }
        }
    }
}

/// Viscoelastic elasticity form: `a(u,v) = integral C*(1+i*eta) : eps(u) : eps(v) dOmega`
///
/// Complex-valued stiffness for frequency-domain analysis with structural damping.
/// The loss factor `eta` introduces energy dissipation proportional to strain energy.
///
/// Used for loudspeaker suspension (surround, spider) and other damped structures.
#[derive(Debug, Clone)]
pub struct ViscoelasticForm {
    material: ViscoelasticMaterial,
    plane: Option<PlaneAssumption>,
}

impl ViscoelasticForm {
    pub fn new_3d(material: ViscoelasticMaterial) -> Self {
        Self {
            material,
            plane: None,
        }
    }

    pub fn new_2d(material: ViscoelasticMaterial, plane: PlaneAssumption) -> Self {
        Self {
            material,
            plane: Some(plane),
        }
    }

    fn plane_2d(&self) -> PlaneAssumption {
        self.plane.expect(
            "ViscoelasticForm assembled on a 2D mesh without a plane assumption; \
             build the form with ViscoelasticForm::new_2d",
        )
    }
}

/// Component index of an interleaved vector DOF.
///
/// Vector DOFs are ordered `[node0_x, node0_y, ..., node1_x, ...]`, so DOF
/// `dof` carries vector component `dof % dim`. Routing every `% dim` through
/// this helper keeps the layout assumption in one documented place: a future
/// blocked/segmented ordering only has to change this function (and the
/// [`BasisTabulation`](crate::spaces::BasisTabulation) accessors it mirrors).
#[inline]
pub(crate) fn vector_component(dof: usize, dim: usize) -> usize {
    dof % dim
}

impl BilinearForm<Complex64> for ViscoelasticForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        let factor = self.material.complex_factor(); // (1 + i*eta)

        match dim {
            2 => {
                let c_voigt = match self.plane_2d() {
                    PlaneAssumption::PlaneStress => self.material.elastic.voigt_2d_plane_stress(),
                    PlaneAssumption::PlaneStrain => self.material.elastic.voigt_2d_plane_strain(),
                };

                debug_assert_eq!(test.n_dofs % 2, 0);
                debug_assert_eq!(test.derivatives.len(), test.n_dofs * 2);
                let n_dofs = test.n_dofs;
                for a in 0..n_dofs {
                    let comp_a = vector_component(a, 2);
                    let dphi_a_dx = test.grad(a, 0);
                    let dphi_a_dy = test.grad(a, 1);
                    let b_a = strain_col_2d(comp_a, dphi_a_dx, dphi_a_dy);

                    for b in 0..n_dofs {
                        let comp_b = vector_component(b, 2);
                        let dphi_b_dx = trial.grad(b, 0);
                        let dphi_b_dy = trial.grad(b, 1);
                        let b_b = strain_col_2d(comp_b, dphi_b_dx, dphi_b_dy);

                        let mut k = 0.0;
                        for p in 0..3 {
                            for q in 0..3 {
                                k += b_a[p] * c_voigt[p * 3 + q] * b_b[q];
                            }
                        }
                        local_matrix.add(a, b, factor * k * w);
                    }
                }
            }
            3 => {
                let c_voigt = self.material.elastic.voigt_3d();
                debug_assert_eq!(test.n_dofs % 3, 0);
                debug_assert_eq!(test.derivatives.len(), test.n_dofs * 3);
                let n_dofs = test.n_dofs;

                for a in 0..n_dofs {
                    let comp_a = vector_component(a, 3);
                    let dphi_a = [test.grad(a, 0), test.grad(a, 1), test.grad(a, 2)];
                    let b_a = strain_col_3d(comp_a, dphi_a);

                    for b in 0..n_dofs {
                        let comp_b = vector_component(b, 3);
                        let dphi_b = [trial.grad(b, 0), trial.grad(b, 1), trial.grad(b, 2)];
                        let b_b = strain_col_3d(comp_b, dphi_b);

                        let mut k = 0.0;
                        for p in 0..6 {
                            for q in 0..6 {
                                k += b_a[p] * c_voigt[p * 6 + q] * b_b[q];
                            }
                        }
                        local_matrix.add(a, b, factor * k * w);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Structural mass form for dynamic analysis: `m(u,v) = integral rho * u . v dOmega`
///
/// Delegates to [`VectorMassForm`](crate::forms::VectorMassForm) with `coefficient = rho`.
#[derive(Debug, Clone, Copy)]
pub struct StructuralMassForm(crate::forms::VectorMassForm);

impl StructuralMassForm {
    pub fn new(density: f64) -> Self {
        Self(crate::forms::VectorMassForm {
            coefficient: density,
        })
    }
}

impl BilinearForm<f64> for StructuralMassForm {
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
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, VectorH1Space};

    #[test]
    fn test_elasticity_form_symmetry_2d() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = ElasticityForm::new_2d(
            IsotropicMaterial::new(1.0, 0.25),
            PlaneAssumption::PlaneStrain,
        );
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (mat.get(i, j) - mat.get(j, i)).abs();
                assert!(
                    diff < 1e-14,
                    "Not symmetric at ({i},{j}): {} vs {}",
                    mat.get(i, j),
                    mat.get(j, i)
                );
            }
        }
    }

    #[test]
    fn test_elasticity_form_positive_diagonal_2d() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = ElasticityForm::new_2d(IsotropicMaterial::steel(), PlaneAssumption::PlaneStress);
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            assert!(
                mat.get(i, i) > 0.0,
                "Diagonal {i} = {} should be positive",
                mat.get(i, i)
            );
        }
    }

    #[test]
    fn test_elasticity_form_symmetry_3d() {
        let mesh = crate::mesh::unit_cube_tetrahedra(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let tab = space.tabulate(&mesh, 0, 0.25, 0.25, 0.25);
        let n = tab.n_dofs;

        let form = ElasticityForm::new_3d(IsotropicMaterial::new(1.0, 0.3));
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (mat.get(i, j) - mat.get(j, i)).abs();
                assert!(
                    diff < 1e-14,
                    "Not symmetric at ({i},{j}): {} vs {}",
                    mat.get(i, j),
                    mat.get(j, i)
                );
            }
        }
    }

    #[test]
    fn test_viscoelastic_form_matches_elastic_real_part() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let elastic_mat = IsotropicMaterial::new(1.0, 0.25);
        let visco_mat = ViscoelasticMaterial::new(elastic_mat, 0.3);

        let elastic_form = ElasticityForm::new_2d(elastic_mat, PlaneAssumption::PlaneStrain);
        let visco_form = ViscoelasticForm::new_2d(visco_mat, PlaneAssumption::PlaneStrain);

        let mut k_real = LocalMatrix::<f64>::zeros(n);
        let mut k_complex = LocalMatrix::<Complex64>::zeros(n);

        elastic_form.integrate_point(&tab, &tab, 1.0, &mut k_real);
        visco_form.integrate_point(&tab, &tab, 1.0, &mut k_complex);

        // Real part of viscoelastic should equal real elastic
        for i in 0..n {
            for j in 0..n {
                let re = k_complex.get(i, j).re;
                let el = k_real.get(i, j);
                assert!(
                    (re - el).abs() < 1e-14,
                    "Real part mismatch at ({i},{j}): {re} vs {el}"
                );
            }
        }

        // Imaginary part should be eta * real part
        let eta = 0.3;
        for i in 0..n {
            for j in 0..n {
                let im = k_complex.get(i, j).im;
                let expected = eta * k_real.get(i, j);
                assert!(
                    (im - expected).abs() < 1e-14,
                    "Imag part at ({i},{j}): {im} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn test_viscoelastic_form_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mat = ViscoelasticMaterial::surround();
        let form = ViscoelasticForm::new_2d(mat, PlaneAssumption::PlaneStress);
        let mut k = LocalMatrix::<Complex64>::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (k.get(i, j) - k.get(j, i)).norm();
                assert!(diff < 1e-14, "Not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    #[should_panic(
        expected = "without a plane assumption; build the form with ElasticityForm::new_2d"
    )]
    fn test_elasticity_3d_form_on_2d_mesh_panics_with_contract_message() {
        // A form built with `new_3d` (plane: None) reused on a 2D mesh must
        // fail loudly at the dimensional precondition, not silently assemble.
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = ElasticityForm::new_3d(IsotropicMaterial::new(1.0, 0.25));
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);
    }

    #[test]
    fn test_vector_dof_layout_matches_tabulation_accessors() {
        // The interleaved-layout contract: `vector_component` and `grad()`
        // must agree with the raw `derivatives` storage for VectorH1Space.
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        assert_eq!(tab.n_dofs % 2, 0);
        assert_eq!(tab.derivatives.len(), tab.n_dofs * 2);
        for dof in 0..tab.n_dofs {
            assert_eq!(super::vector_component(dof, 2), dof % 2);
            for d in 0..2 {
                assert_eq!(tab.grad(dof, d), tab.derivatives[dof * 2 + d]);
            }
        }
    }

    #[test]
    fn test_structural_mass_form_spd() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = StructuralMassForm::new(1000.0); // 1000 kg/m^3
        let mut m = LocalMatrix::<f64>::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut m);

        // Diagonal should be positive
        for i in 0..n {
            assert!(m.get(i, i) > 0.0, "M[{i},{i}] = {}", m.get(i, i));
        }

        // Symmetric
        for i in 0..n {
            for j in i + 1..n {
                let diff = (m.get(i, j) - m.get(j, i)).abs();
                assert!(diff < 1e-14, "Not symmetric at ({i},{j})");
            }
        }
    }
}
