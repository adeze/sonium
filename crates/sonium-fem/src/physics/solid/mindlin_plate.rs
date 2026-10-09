//! Mindlin (Reissner-Mindlin) plate bending element.
//!
//! Three-field formulation with DOFs `[w, θx, θy]` per node:
//! - `w`: transverse displacement
//! - `θx`: rotation about y-axis
//! - `θy`: rotation about x-axis
//!
//! Works with 2D triangular or quadrilateral meshes.
//!
//! The default form uses a low-order assumed-shear projection for P1 triangles,
//! equivalent to an element-average DSG/MITC-style shear strain. This removes the
//! worst shear-locking behaviour of the fully compatible `theta - grad(w)` field
//! while preserving the original compatible form through `new_consistent`.

use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::mesh::{ElementType, Mesh};
use crate::spaces::{BasisTabulation, FieldType, FunctionSpace, H1Space};

/// Mindlin plate function space: 3 components (w, θx, θy) per node.
///
/// DOF ordering is interleaved: `[node0_w, node0_θx, node0_θy, node1_w, ...]`
pub struct MindlinPlateSpace {
    scalar_space: H1Space,
}

impl MindlinPlateSpace {
    pub fn new(degree: PolynomialDegree) -> Self {
        Self {
            scalar_space: H1Space::new(degree),
        }
    }
}

impl FunctionSpace for MindlinPlateSpace {
    fn field_type(&self) -> FieldType {
        FieldType::Vector
    }

    fn num_components(&self) -> usize {
        3
    }

    fn num_global_dofs(&self, mesh: &Mesh) -> usize {
        self.scalar_space.num_global_dofs(mesh) * 3
    }

    fn num_local_dofs(&self, elem_type: ElementType) -> usize {
        self.scalar_space.num_local_dofs(elem_type) * 3
    }

    fn local_to_global(&self, mesh: &Mesh, elem_idx: usize) -> Vec<usize> {
        let scalar_dofs = self.scalar_space.local_to_global(mesh, elem_idx);
        let mut dofs = Vec::with_capacity(scalar_dofs.len() * 3);
        for &scalar_dof in &scalar_dofs {
            dofs.push(scalar_dof * 3);
            dofs.push(scalar_dof * 3 + 1);
            dofs.push(scalar_dof * 3 + 2);
        }
        dofs
    }

    fn tabulate(
        &self,
        mesh: &Mesh,
        elem_idx: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> BasisTabulation {
        let scalar_tab = self.scalar_space.tabulate(mesh, elem_idx, xi, eta, zeta);
        let n_scalar = scalar_tab.n_dofs;
        let dim = scalar_tab.spatial_dim; // 2 for 2D mesh
        let n_dofs = n_scalar * 3;

        // Values: each DOF gets the scalar basis value of its associated node.
        // values[3*i + c] = phi_i for all components c.
        let mut values = vec![0.0; n_dofs];
        for i in 0..n_scalar {
            let phi = scalar_tab.values[i];
            values[3 * i] = phi;
            values[3 * i + 1] = phi;
            values[3 * i + 2] = phi;
        }

        // Derivatives: gradient of the scalar basis for each DOF.
        // derivatives[(3*i + c) * dim + d] = dphi_i/dx_d
        let mut derivatives = vec![0.0; n_dofs * dim];
        for i in 0..n_scalar {
            for d in 0..dim {
                let grad_d = scalar_tab.grad(i, d);
                derivatives[(3 * i) * dim + d] = grad_d;
                derivatives[(3 * i + 1) * dim + d] = grad_d;
                derivatives[(3 * i + 2) * dim + d] = grad_d;
            }
        }

        BasisTabulation {
            n_dofs,
            spatial_dim: dim,
            values,
            derivatives,
            hessians: Vec::new(),
            det_j: scalar_tab.det_j,
            physical_coords: scalar_tab.physical_coords,
        }
    }

    fn boundary_dofs(&self, mesh: &Mesh, boundary_marker: i32) -> Vec<usize> {
        let scalar_dofs = self.scalar_space.boundary_dofs(mesh, boundary_marker);
        let mut dofs = Vec::with_capacity(scalar_dofs.len() * 3);
        for &scalar_dof in &scalar_dofs {
            dofs.push(scalar_dof * 3);
            dofs.push(scalar_dof * 3 + 1);
            dofs.push(scalar_dof * 3 + 2);
        }
        dofs
    }
}

/// Shear-strain treatment for low-order Mindlin plate elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MindlinPlateShearStabilization {
    /// Fully compatible shear strain `gamma = theta - grad(w)`.
    ///
    /// This is retained for regression tests and thick-plate comparisons, but
    /// P1 triangles lock badly as `h/L -> 0`.
    Consistent,
    /// Element-average discrete-shear-gap/MITC-style assumed shear strain.
    ///
    /// For P1 triangles this replaces the rotational shape functions in the
    /// shear energy by their element average while keeping the constant `grad(w)`
    /// exactly. For non-triangular or higher-order elements it falls back to the
    /// compatible field rather than applying the wrong projection.
    Dsg,
}

/// Mindlin plate stiffness form: bending + shear energy.
///
/// Bending: `D_b * ∫ ε(θ)^T D_matrix ε(φ) dΩ`
/// Shear:  `κGh * ∫ (θ - ∇w) · (φ - ∇v) dΩ`
///
/// where `D_b = E*h³ / (12*(1-ν²))` and `D_matrix = [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]`
pub struct MindlinPlateForm {
    /// Bending rigidity D = E*h³ / (12*(1-ν²))
    pub d_bending: f64,
    /// Poisson ratio
    pub nu: f64,
    /// Shear stiffness κ*G*h
    pub kappa_gh: f64,
    /// Shear-strain projection used to avoid locking in P1 triangles.
    pub shear_stabilization: MindlinPlateShearStabilization,
}

impl MindlinPlateForm {
    /// Create the stabilized Mindlin plate form.
    pub fn new(e: f64, nu: f64, h: f64, kappa: f64) -> Self {
        Self::with_shear_stabilization(e, nu, h, kappa, MindlinPlateShearStabilization::Dsg)
    }

    /// Create the original fully compatible Mindlin plate form.
    pub fn new_consistent(e: f64, nu: f64, h: f64, kappa: f64) -> Self {
        Self::with_shear_stabilization(e, nu, h, kappa, MindlinPlateShearStabilization::Consistent)
    }

    pub fn with_shear_stabilization(
        e: f64,
        nu: f64,
        h: f64,
        kappa: f64,
        shear_stabilization: MindlinPlateShearStabilization,
    ) -> Self {
        let d_bending = e * h.powi(3) / (12.0 * (1.0 - nu * nu));
        let g = e / (2.0 * (1.0 + nu));
        let kappa_gh = kappa * g * h;
        Self {
            d_bending,
            nu,
            kappa_gh,
            shear_stabilization,
        }
    }
}

impl BilinearForm<f64> for MindlinPlateForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        let d11 = self.d_bending;
        let d12 = self.nu * self.d_bending;
        let d33 = (1.0 - self.nu) / 2.0 * self.d_bending;
        let kg = self.kappa_gh;

        for a in 0..test.n_dofs {
            let comp_a = a % 3;
            let phi_a = test.value(a);
            let grad_ax = test.grad(a, 0);
            let grad_ay = test.grad(a, 1);

            for b in 0..trial.n_dofs {
                let comp_b = b % 3;
                let phi_b = trial.value(b);
                let grad_bx = trial.grad(b, 0);
                let grad_by = trial.grad(b, 1);

                // Bending stiffness
                let mut k_b = 0.0;
                match (comp_a, comp_b) {
                    (1, 1) => {
                        // θx-θx
                        k_b += d11 * grad_ax * grad_bx;
                        k_b += d33 * grad_ay * grad_by;
                    }
                    (2, 2) => {
                        // θy-θy
                        k_b += d11 * grad_ay * grad_by;
                        k_b += d33 * grad_ax * grad_bx;
                    }
                    (1, 2) => {
                        // θx-θy
                        k_b += d12 * grad_ax * grad_by;
                        k_b += d33 * grad_ay * grad_bx;
                    }
                    (2, 1) => {
                        // θy-θx
                        k_b += d12 * grad_ay * grad_bx;
                        k_b += d33 * grad_ax * grad_by;
                    }
                    _ => {}
                }

                let k_s = kg
                    * shear_entry(
                        self.shear_stabilization,
                        test,
                        trial,
                        ShearDof::new(a, phi_a, grad_ax, grad_ay),
                        ShearDof::new(b, phi_b, grad_bx, grad_by),
                    );

                local_matrix.add(a, b, (k_b + k_s) * w);
            }
        }
    }
}

/// Mindlin plate load form for distributed pressure `q(x,y)`.
pub struct MindlinPlateLoadForm<F: Fn(f64, f64) -> f64 + Send + Sync> {
    pub load_fn: F,
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> LinearForm<f64> for MindlinPlateLoadForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j;
        let x = test.physical_coords[0];
        let y = test.physical_coords[1];
        let q = (self.load_fn)(x, y);

        for a in 0..test.n_dofs {
            let comp_a = a % 3;
            if comp_a == 0 {
                // Load only acts on w-DOF
                let phi_a = test.value(a);
                local_vector.add(a, q * phi_a * w);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ShearDof {
    index: usize,
    phi: f64,
    grad: [f64; 2],
}

impl ShearDof {
    fn new(index: usize, phi: f64, grad_x: f64, grad_y: f64) -> Self {
        Self {
            index,
            phi,
            grad: [grad_x, grad_y],
        }
    }
}

fn shear_entry(
    stabilization: MindlinPlateShearStabilization,
    test: &BasisTabulation,
    trial: &BasisTabulation,
    a: ShearDof,
    b: ShearDof,
) -> f64 {
    match stabilization {
        MindlinPlateShearStabilization::Consistent => consistent_shear_entry(a, b),
        MindlinPlateShearStabilization::Dsg if is_p1_triangle_plate_tab(test, trial) => {
            projected_shear_entry(a, b)
        }
        MindlinPlateShearStabilization::Dsg => consistent_shear_entry(a, b),
    }
}

fn consistent_shear_entry(a: ShearDof, b: ShearDof) -> f64 {
    match (a.index % 3, b.index % 3) {
        (0, 0) => a.grad[0] * b.grad[0] + a.grad[1] * b.grad[1],
        (0, 1) => -a.grad[0] * b.phi,
        (0, 2) => -a.grad[1] * b.phi,
        (1, 0) => -a.phi * b.grad[0],
        (2, 0) => -a.phi * b.grad[1],
        (1, 1) | (2, 2) => a.phi * b.phi,
        _ => 0.0,
    }
}

fn projected_shear_entry(a: ShearDof, b: ShearDof) -> f64 {
    let ba = projected_shear_coefficients(a);
    let bb = projected_shear_coefficients(b);
    ba[0] * bb[0] + ba[1] * bb[1]
}

fn projected_shear_coefficients(dof: ShearDof) -> [f64; 2] {
    match dof.index % 3 {
        0 => [-dof.grad[0], -dof.grad[1]],
        1 => [1.0 / 3.0, 0.0],
        2 => [0.0, 1.0 / 3.0],
        _ => unreachable!(),
    }
}

fn is_p1_triangle_plate_tab(test: &BasisTabulation, trial: &BasisTabulation) -> bool {
    test.spatial_dim == 2
        && trial.spatial_dim == 2
        && test.n_dofs == 9
        && trial.n_dofs == 9
        && test.det_j > 0.0
        && trial.det_j > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::rectangular_mesh_triangles;

    #[test]
    fn test_mindlin_plate_space_dofs() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1);
        let space = MindlinPlateSpace::new(PolynomialDegree::P1);
        assert_eq!(space.num_global_dofs(&mesh), mesh.nodes.len() * 3);

        let dofs = space.local_to_global(&mesh, 0);
        assert_eq!(dofs.len(), 9);
        assert_eq!(&dofs[0..3], &[0, 1, 2]);
    }

    #[test]
    fn projected_shear_uses_element_average_rotations() {
        let k = projected_shear_entry(
            ShearDof::new(0, 0.0, 2.0, 3.0),
            ShearDof::new(1, 0.0, 0.0, 0.0),
        );
        assert!((k + 2.0 / 3.0).abs() < 1.0e-12);

        let rr = projected_shear_entry(
            ShearDof::new(1, 0.0, 0.0, 0.0),
            ShearDof::new(1, 0.0, 0.0, 0.0),
        );
        assert!((rr - 1.0 / 9.0).abs() < 1.0e-12);
    }

    #[test]
    fn default_form_uses_dsg_shear_projection() {
        let form = MindlinPlateForm::new(1.0, 0.3, 0.1, 5.0 / 6.0);
        assert_eq!(
            form.shear_stabilization,
            MindlinPlateShearStabilization::Dsg
        );

        let consistent = MindlinPlateForm::new_consistent(1.0, 0.3, 0.1, 5.0 / 6.0);
        assert_eq!(
            consistent.shear_stabilization,
            MindlinPlateShearStabilization::Consistent
        );
    }
}

/// Bending-only Mindlin plate form (for selective reduced integration).
pub struct MindlinPlateBendingForm {
    pub d_bending: f64,
    pub nu: f64,
}

impl MindlinPlateBendingForm {
    pub fn new(e: f64, nu: f64, h: f64) -> Self {
        let d_bending = e * h.powi(3) / (12.0 * (1.0 - nu * nu));
        Self { d_bending, nu }
    }
}

impl BilinearForm<f64> for MindlinPlateBendingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        let d11 = self.d_bending;
        let d12 = self.nu * self.d_bending;
        let d33 = (1.0 - self.nu) / 2.0 * self.d_bending;

        for a in 0..test.n_dofs {
            let comp_a = a % 3;
            let grad_ax = test.grad(a, 0);
            let grad_ay = test.grad(a, 1);

            for b in 0..trial.n_dofs {
                let comp_b = b % 3;
                let grad_bx = trial.grad(b, 0);
                let grad_by = trial.grad(b, 1);

                let mut k_b = 0.0;
                match (comp_a, comp_b) {
                    (1, 1) => {
                        k_b += d11 * grad_ax * grad_bx;
                        k_b += d33 * grad_ay * grad_by;
                    }
                    (2, 2) => {
                        k_b += d11 * grad_ay * grad_by;
                        k_b += d33 * grad_ax * grad_bx;
                    }
                    (1, 2) => {
                        k_b += d12 * grad_ax * grad_by;
                        k_b += d33 * grad_ay * grad_bx;
                    }
                    (2, 1) => {
                        k_b += d12 * grad_ay * grad_bx;
                        k_b += d33 * grad_ax * grad_by;
                    }
                    _ => {}
                }
                local_matrix.add(a, b, k_b * w);
            }
        }
    }
}

/// Shear-only Mindlin plate form (for selective reduced integration).
pub struct MindlinPlateShearForm {
    pub kappa_gh: f64,
    pub shear_stabilization: MindlinPlateShearStabilization,
}

impl MindlinPlateShearForm {
    pub fn new(e: f64, nu: f64, h: f64, kappa: f64) -> Self {
        Self::with_shear_stabilization(e, nu, h, kappa, MindlinPlateShearStabilization::Dsg)
    }

    pub fn new_consistent(e: f64, nu: f64, h: f64, kappa: f64) -> Self {
        Self::with_shear_stabilization(e, nu, h, kappa, MindlinPlateShearStabilization::Consistent)
    }

    pub fn with_shear_stabilization(
        e: f64,
        nu: f64,
        h: f64,
        kappa: f64,
        shear_stabilization: MindlinPlateShearStabilization,
    ) -> Self {
        let g = e / (2.0 * (1.0 + nu));
        let kappa_gh = kappa * g * h;
        Self {
            kappa_gh,
            shear_stabilization,
        }
    }
}

impl BilinearForm<f64> for MindlinPlateShearForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        let kg = self.kappa_gh;

        for a in 0..test.n_dofs {
            let phi_a = test.value(a);
            let grad_ax = test.grad(a, 0);
            let grad_ay = test.grad(a, 1);

            for b in 0..trial.n_dofs {
                let phi_b = trial.value(b);
                let grad_bx = trial.grad(b, 0);
                let grad_by = trial.grad(b, 1);

                let k_s = kg
                    * shear_entry(
                        self.shear_stabilization,
                        test,
                        trial,
                        ShearDof::new(a, phi_a, grad_ax, grad_ay),
                        ShearDof::new(b, phi_b, grad_bx, grad_by),
                    );
                local_matrix.add(a, b, k_s * w);
            }
        }
    }
}
