//! Magnetostatic bilinear and linear forms
//!
//! Reduced scalar magnetic potential formulation:
//!
//! ```text
//! div(mu * grad(phi)) = div(B_r)
//! ```
//!
//! Weak form:
//! - Bilinear: `a(phi, psi) = integral mu(x) * grad(phi) . grad(psi) dOmega`
//! - Linear:   `l(psi) = integral B_r(x) . grad(psi) dOmega`
//!
//! After solving for phi, the B-field is recovered as:
//! ```text
//! B = -mu * grad(phi) + B_r   (in magnet regions)
//! B = -mu * grad(phi)          (elsewhere)
//! ```

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector, WeightedLaplacianForm};
use crate::mesh::Mesh;
use crate::spaces::{BasisTabulation, FunctionSpace};

use super::materials::MagneticMaterial;

/// Permeability-weighted Laplacian: `a(phi, psi) = integral mu * grad(phi) . grad(psi) dOmega`
///
/// Delegates to [`WeightedLaplacianForm`] with `coefficient = mu`.
#[derive(Debug, Clone)]
pub struct PermeabilityForm(WeightedLaplacianForm);

impl PermeabilityForm {
    pub fn new(material: &MagneticMaterial) -> Self {
        Self(WeightedLaplacianForm {
            coefficient: material.permeability(),
        })
    }

    pub fn from_mu(mu: f64) -> Self {
        Self(WeightedLaplacianForm { coefficient: mu })
    }
}

impl BilinearForm<f64> for PermeabilityForm {
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

/// Permanent magnet source form: `l(psi) = integral B_r . grad(psi) dOmega`
///
/// This is the RHS contribution from permanent magnets. Only contributes in
/// elements that contain magnet material.
pub struct MagnetSourceForm {
    /// Remanent flux density vector [B_rx, B_ry, B_rz].
    pub br: [f64; 3],
}

impl MagnetSourceForm {
    pub fn new(material: &MagneticMaterial) -> Self {
        Self {
            br: material.remanent_b(),
        }
    }

    /// Check if this source has any contribution (non-zero B_r).
    pub fn is_active(&self) -> bool {
        self.br[0].abs() > 1e-15 || self.br[1].abs() > 1e-15 || self.br[2].abs() > 1e-15
    }
}

impl LinearForm<f64> for MagnetSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n {
            let mut br_dot_grad = 0.0;
            for d in 0..dim {
                br_dot_grad += self.br[d] * test.grad(i, d);
            }
            local_vector.add(i, br_dot_grad * w);
        }
    }
}

/// Axisymmetric magnetic vector-potential form for solenoids and coils.
///
/// Solves for the azimuthal component `A_theta(r,z)` in a 2D `(r,z)` mesh:
///
/// ```text
/// ∫ (1/mu) r curl(A_theta e_theta).curl(v e_theta) dr dz
///     = ∫ J_theta v r dr dz
/// ```
///
/// With `B_r = -∂A_theta/∂z` and
/// `B_z = ∂A_theta/∂r + A_theta/r`, the bilinear integrand is
/// `(1/mu) r [∂z A ∂z v + (∂r A + A/r)(∂r v + v/r)]`.
///
/// This scalar formulation is the natural FEM model for axisymmetric coils
/// carrying azimuthal current. Use [`AzimuthalCurrentDensityForm`] for the RHS.
#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricAThetaForm {
    /// Inverse permeability `1/mu`.
    pub inv_mu: f64,
    /// Small radius used to regularize quadrature points on the axis.
    pub axis_epsilon: f64,
}

impl AxisymmetricAThetaForm {
    pub fn new(mu: f64) -> Self {
        Self {
            inv_mu: 1.0 / mu,
            axis_epsilon: 1e-12,
        }
    }
}

impl BilinearForm<f64> for AxisymmetricAThetaForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let r = test.physical_coords[0].abs().max(self.axis_epsilon);
        let w = weight * test.det_j * self.inv_mu;

        for i in 0..test.n_dofs {
            let test_curl = [-test.grad(i, 1), test.grad(i, 0) + test.value(i) / r];
            for j in 0..trial.n_dofs {
                let trial_curl = [-trial.grad(j, 1), trial.grad(j, 0) + trial.value(j) / r];
                local_matrix.add(
                    i,
                    j,
                    r * (test_curl[0] * trial_curl[0] + test_curl[1] * trial_curl[1]) * w,
                );
            }
        }
    }
}

/// Azimuthal current source for the axisymmetric `A_theta` formulation.
///
/// The closure receives `(r, z)` and returns `J_theta` in A/m².
pub struct AzimuthalCurrentDensityForm<F: Fn(f64, f64) -> f64 + Send + Sync> {
    pub current_density_fn: F,
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> AzimuthalCurrentDensityForm<F> {
    pub fn new(current_density_fn: F) -> Self {
        Self { current_density_fn }
    }
}

/// Uniform azimuthal current density constructor.
pub fn uniform_azimuthal_current_density(
    j_theta: f64,
) -> AzimuthalCurrentDensityForm<impl Fn(f64, f64) -> f64 + Send + Sync> {
    AzimuthalCurrentDensityForm {
        current_density_fn: move |_, _| j_theta,
    }
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> LinearForm<f64> for AzimuthalCurrentDensityForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let r = test.physical_coords[0].abs();
        let z = test.physical_coords[1];
        let j_theta = (self.current_density_fn)(r, z);
        let w = weight * test.det_j * r;

        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * j_theta * w);
        }
    }
}

/// Distributional point magnetic dipole for scalar magnetic potential.
///
/// For a dipole at `x0` with moment `m`, the weak RHS contribution is
/// `-m · grad(v)(x0)`. This avoids replacing the dipole with a finite magnet
/// volume and gives the correct singular source in the finite-element sense.
#[derive(Debug, Clone, Copy)]
pub struct PointDipoleSource {
    pub position: [f64; 3],
    pub moment: [f64; 3],
}

impl PointDipoleSource {
    pub fn new(position: [f64; 3], moment: [f64; 3]) -> Self {
        Self { position, moment }
    }
}

/// Assemble a point-dipole RHS over the element containing the source.
///
/// Currently supports affine P1 triangle meshes, which covers the 2D scalar
/// potential QA cases. Returns a zero vector when the point lies outside the
/// mesh.
pub fn assemble_point_dipole_source<S>(
    source: &PointDipoleSource,
    mesh: &Mesh,
    space: &S,
) -> Vec<f64>
where
    S: FunctionSpace,
{
    let mut rhs = vec![0.0; space.num_global_dofs(mesh)];

    for elem_idx in 0..mesh.elements.len() {
        let elem = &mesh.elements[elem_idx];
        if elem.element_type != crate::mesh::ElementType::Triangle {
            continue;
        }

        let nodes = elem.vertices();
        let p0 = mesh.nodes[nodes[0]];
        let p1 = mesh.nodes[nodes[1]];
        let p2 = mesh.nodes[nodes[2]];
        if let Some((l1, l2)) = reference_coords_in_triangle(source.position, p0, p1, p2) {
            let tab = space.tabulate(mesh, elem_idx, l1, l2, 0.0);
            let dofs = space.local_to_global(mesh, elem_idx);
            for (local, &dof) in dofs.iter().enumerate() {
                let mut contribution = 0.0;
                for d in 0..tab.spatial_dim {
                    contribution -= source.moment[d] * tab.grad(local, d);
                }
                rhs[dof] += contribution;
            }
            break;
        }
    }

    rhs
}

fn reference_coords_in_triangle(
    position: [f64; 3],
    p0: crate::mesh::Point,
    p1: crate::mesh::Point,
    p2: crate::mesh::Point,
) -> Option<(f64, f64)> {
    let v0 = [p1.x - p0.x, p1.y - p0.y];
    let v1 = [p2.x - p0.x, p2.y - p0.y];
    let v2 = [position[0] - p0.x, position[1] - p0.y];
    let det = v0[0] * v1[1] - v0[1] * v1[0];
    if det.abs() < 1e-30 {
        return None;
    }
    let xi = (v2[0] * v1[1] - v2[1] * v1[0]) / det;
    let eta = (v0[0] * v2[1] - v0[1] * v2[0]) / det;
    let l0 = 1.0 - xi - eta;
    let tol = 1e-12;
    if xi >= -tol && eta >= -tol && l0 >= -tol {
        Some((xi, eta))
    } else {
        None
    }
}

/// Compute B-field from the scalar potential solution at a point.
///
/// `B = -mu * grad(phi) + B_r`
///
/// `potential_values`: solution phi at the element's DOFs
/// `tab`: basis tabulation at the evaluation point
pub fn compute_b_field(
    potential_values: &[f64],
    tab: &BasisTabulation,
    mu: f64,
    br: &[f64; 3],
) -> Vec<f64> {
    let dim = tab.spatial_dim;

    // grad(phi) = sum_i phi_i * grad(N_i)
    let mut grad_phi = vec![0.0; dim];
    for (i, &phi_i) in potential_values.iter().enumerate().take(tab.n_dofs) {
        for (d, gp) in grad_phi.iter_mut().enumerate() {
            *gp += phi_i * tab.grad(i, d);
        }
    }

    // B = -mu * grad(phi) + B_r
    (0..dim).map(|d| -mu * grad_phi[d] + br[d]).collect()
}

/// Extract the BL product (force factor) by integrating B along a coil path.
///
/// `BL = integral (B x dl) . e_motion` along the coil
///
/// For a simple axisymmetric motor, this reduces to `BL = B_r * 2 * pi * r * N_turns`
/// where `B_r` is the radial component of B at the coil position.
///
/// `coil_points`: points along the coil path
/// `coil_tangents`: tangent vectors at each point (dl direction)
/// `b_at_points`: B-field evaluated at each coil point
/// `motion_direction`: direction of voice coil motion (typically axial [0,0,1])
pub fn compute_bl(
    coil_points: &[[f64; 3]],
    coil_tangents: &[[f64; 3]],
    b_at_points: &[[f64; 3]],
    motion_direction: &[f64; 3],
) -> f64 {
    assert_eq!(coil_points.len(), coil_tangents.len());
    assert_eq!(coil_points.len(), b_at_points.len());

    let mut bl = 0.0;
    for i in 0..coil_points.len() {
        let b = &b_at_points[i];
        let dl = &coil_tangents[i];

        // F = I * (dl x B), projected onto motion direction
        // BL contribution = (dl x B) . e_motion
        let cross = [
            dl[1] * b[2] - dl[2] * b[1],
            dl[2] * b[0] - dl[0] * b[2],
            dl[0] * b[1] - dl[1] * b[0],
        ];
        bl += cross[0] * motion_direction[0]
            + cross[1] * motion_direction[1]
            + cross[2] * motion_direction[2];
    }
    bl
}

/// Compute BL(x) curve: force factor as a function of voice coil displacement.
///
/// Returns a vector of `(displacement, BL)` pairs.
///
/// `displacements`: list of axial displacements to evaluate
/// `coil_points_at_rest`: coil path points at rest position
/// `coil_tangents`: tangent vectors (constant, coil shape doesn't change)
/// `motion_direction`: direction of motion [0, 0, 1] for axial
/// `b_field_fn`: function that evaluates B at any 3D point (from the magnetostatic FEM solution)
pub fn compute_bl_curve<F>(
    displacements: &[f64],
    coil_points_at_rest: &[[f64; 3]],
    coil_tangents: &[[f64; 3]],
    motion_direction: &[f64; 3],
    b_field_fn: F,
) -> Vec<(f64, f64)>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    displacements
        .iter()
        .map(|&dx| {
            // Shift coil points by displacement along motion direction
            let shifted_points: Vec<[f64; 3]> = coil_points_at_rest
                .iter()
                .map(|p| {
                    [
                        p[0] + dx * motion_direction[0],
                        p[1] + dx * motion_direction[1],
                        p[2] + dx * motion_direction[2],
                    ]
                })
                .collect();

            // Evaluate B at shifted positions
            let b_values: Vec<[f64; 3]> = shifted_points.iter().map(&b_field_fn).collect();

            let bl = compute_bl(&shifted_points, coil_tangents, &b_values, motion_direction);
            (dx, bl)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space};

    #[test]
    fn test_permeability_form_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = PermeabilityForm::from_mu(1000.0 * super::super::materials::MU_0);
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (mat.get(i, j) - mat.get(j, i)).abs();
                assert!(diff < 1e-14, "Not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_permeability_form_scales_with_mu() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form1 = PermeabilityForm::from_mu(1.0);
        let form2 = PermeabilityForm::from_mu(100.0);
        let mut mat1 = LocalMatrix::zeros(n);
        let mut mat2 = LocalMatrix::zeros(n);
        form1.integrate_point(&tab, &tab, 1.0, &mut mat1);
        form2.integrate_point(&tab, &tab, 1.0, &mut mat2);

        for i in 0..n {
            for j in 0..n {
                let ratio = if mat1.get(i, j).abs() > 1e-15 {
                    mat2.get(i, j) / mat1.get(i, j)
                } else {
                    100.0 // both zero
                };
                assert!(
                    (ratio - 100.0).abs() < 1e-10,
                    "Scaling mismatch at ({i},{j}): ratio = {ratio}"
                );
            }
        }
    }

    #[test]
    fn test_magnet_source_inactive_for_air() {
        let mat = super::super::materials::MagneticMaterial::air();
        let form = MagnetSourceForm::new(&mat);
        assert!(!form.is_active());
    }

    #[test]
    fn test_b_field_recovery() {
        // Uniform potential phi = x → grad(phi) = (1, 0)
        // B = -mu * (1, 0) + B_r
        let tab = BasisTabulation {
            n_dofs: 3,
            spatial_dim: 2,
            values: vec![0.0, 0.5, 0.5], // doesn't matter for grad
            derivatives: vec![
                -1.0, -1.0, // grad N0
                1.0, 0.0, // grad N1
                0.0, 1.0, // grad N2
            ],
            hessians: Vec::new(),
            det_j: 1.0,
            physical_coords: [0.5, 0.5, 0.0],
        };
        // phi at nodes: node0=0, node1=1, node2=0 (so phi=x on reference triangle)
        let phi = [0.0, 1.0, 0.0];
        let mu = 1.0;
        let br = [0.0, 0.0, 0.0];

        let b = compute_b_field(&phi, &tab, mu, &br);
        // grad(phi) = 0*(-1,-1) + 1*(1,0) + 0*(0,1) = (1, 0)
        // B = -1 * (1, 0) = (-1, 0)
        assert!((b[0] - (-1.0)).abs() < 1e-14);
        assert!(b[1].abs() < 1e-14);
    }

    #[test]
    fn test_bl_uniform_field() {
        // Uniform B = (1, 0, 0), coil along y-axis, motion in z
        let coil_points = vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let coil_tangents = vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]; // dl = dy
        let b_values = vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let motion = [0.0, 0.0, 1.0];

        let bl = compute_bl(&coil_points, &coil_tangents, &b_values, &motion);
        // dl x B = (0,1,0) x (1,0,0) = (0,0,-1)
        // (dl x B) . e_z = -1 per point, 2 points → BL = -2
        assert!((bl - (-2.0)).abs() < 1e-14, "BL = {bl}, expected -2.0");
    }

    #[test]
    fn test_axisymmetric_a_theta_has_positive_diagonal() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 1, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let form = AxisymmetricAThetaForm::new(super::super::materials::MU_0);
        let mut mat = LocalMatrix::zeros(tab.n_dofs);

        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..tab.n_dofs {
            assert!(mat.get(i, i) >= 0.0);
        }
    }

    #[test]
    fn axisymmetric_a_theta_matches_full_curl_curl_integrand() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 1, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let form = AxisymmetricAThetaForm::new(2.0);
        let mut matrix = LocalMatrix::zeros(tab.n_dofs);
        form.integrate_point(&tab, &tab, 1.0, &mut matrix);

        let radius = tab.physical_coords[0];
        for i in 0..tab.n_dofs {
            let curl_i = [-tab.grad(i, 1), tab.grad(i, 0) + tab.value(i) / radius];
            for j in 0..tab.n_dofs {
                let curl_j = [-tab.grad(j, 1), tab.grad(j, 0) + tab.value(j) / radius];
                let expected =
                    radius * (curl_i[0] * curl_j[0] + curl_i[1] * curl_j[1]) * tab.det_j / 2.0;
                assert!((matrix.get(i, j) - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn test_azimuthal_current_density_rhs_nonzero() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 1, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let form = AzimuthalCurrentDensityForm::new(|_, _| 2.0);
        let mut rhs = LocalVector::zeros(tab.n_dofs);

        form.integrate_point(&tab, 1.0, &mut rhs);

        assert!(rhs.values.iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn test_point_dipole_source_is_distributional_gradient_load() {
        let mesh = unit_square_triangles(1);
        let space = H1Space::new(PolynomialDegree::P1);
        let source = PointDipoleSource::new([0.25, 0.25, 0.0], [1.0, 0.0, 0.0]);
        let rhs = assemble_point_dipole_source(&source, &mesh, &space);

        assert_eq!(rhs.len(), space.num_global_dofs(&mesh));
        assert!(rhs.iter().any(|v| v.abs() > 0.0));
        let sum: f64 = rhs.iter().sum();
        assert!(
            sum.abs() < 1e-12,
            "dipole source should have zero net charge"
        );
    }
}
