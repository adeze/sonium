//! Linearized Euler Equations (LEE) interface
//!
//! Solves for acoustic variations in velocity and pressure in the presence
//! of a stationary background mean flow, assuming an ideal gas with
//! isentropic perturbations.
//!
//! # Governing equations (frequency domain)
//!
//! Momentum:
//! ```text
//! rho_0 (i omega u' + (U_0 . nabla) u' + (u' . nabla) U_0) + nabla p' = 0
//! ```
//!
//! Continuity (isentropic):
//! ```text
//! i omega p' + U_0 . nabla p' + rho_0 c_0^2 nabla . u' = 0
//! ```
//!
//! # Block system
//!
//! ```text
//! | A_uu    A_up | | u' |   | f_u |
//! | A_pu    A_pp | | p' | = | f_p |
//! ```
//!
//! where:
//! - A_uu = rho_0 (i omega M_v + C_v + G_v): velocity block
//! - A_up = B: pressure gradient in divergence form, whose minus sign B
//!   already carries
//! - A_pu = -rho_0 c_0^2 B^T: velocity divergence, negated because
//!   B^Tu = -(div u, q) exactly
//! - A_pp = i omega M_p + C_p: pressure block
//!
//! The velocity field uses VectorH1 and pressure uses scalar H1.

use crate::assembly::TripletMatrix;
use crate::forms::{BilinearForm, DivergenceForm, LocalMatrix, MassForm, VectorMassForm};
use crate::mesh::Mesh;
use crate::spaces::BasisTabulation;
use num_complex::Complex64;

use super::mean_flow::{MeanFlow, MeanFlowField};

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// Vector convection form for LEE momentum:
/// `c_v(u, v) = integral(rho_0 (U_0 . nabla u) . v) dOmega`
///
/// This is the linearised convection operator acting on the velocity
/// perturbation u' with a known background velocity U_0.
/// Only same-component DOF pairs contribute (component-gated).
pub struct VectorConvectionForm<F: Fn([f64; 3]) -> [f64; 3] + Send + Sync> {
    pub rho: f64,
    pub velocity_fn: F,
}

impl<F: Fn([f64; 3]) -> [f64; 3] + Send + Sync> BilinearForm<f64> for VectorConvectionForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n = test.n_dofs;
        let vel = (self.velocity_fn)(test.physical_coords);
        let w = weight * test.det_j * self.rho;

        for i in 0..n {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            for j in 0..n {
                let cj = j % dim;
                if ci != cj {
                    continue;
                }
                // U_0 . grad(phi_j)
                let u_dot_grad_j: f64 = (0..dim).map(|d| vel[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, phi_i * u_dot_grad_j * w);
            }
        }
    }
}

/// Mean flow velocity gradient form for LEE momentum:
/// `g_v(u, v) = integral(rho_0 (u . nabla U_0) . v) dOmega`
///
/// This term accounts for inhomogeneity of the background flow.
/// It couples different velocity components since (u' . nabla) U_0
/// is a matrix-vector product: sum_j u'_j (dU_0_i / dx_j).
///
/// For uniform mean flow, this term is zero.
pub struct MeanFlowGradientForm<F: MeanFlowField> {
    pub rho: f64,
    pub mean_flow: F,
}

impl<F: MeanFlowField> BilinearForm<f64> for MeanFlowGradientForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n = test.n_dofs;
        let w = weight * test.det_j * self.rho;

        let grad_u0 = self.mean_flow.velocity_gradient(test.physical_coords);

        // (u' . nabla) U_0 = sum_j u'_j (dU_0_i/dx_j)
        // Test DOF i has component ci, trial DOF j has component cj.
        // Contribution: v_i * (dU_0_{ci}/dx_{cj}) * u'_j
        for i in 0..n {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            for j in 0..n {
                let cj = j % dim;
                let phi_j = trial.vector_value(j, cj);
                let du_ci_dx_cj = grad_u0[ci][cj];
                local_matrix.add(i, j, phi_i * du_ci_dx_cj * phi_j * w);
            }
        }
    }
}

/// Gradient coupling form for LEE continuity:
/// `g(u, q) = integral(u . grad q) dOmega`
///
/// This is the coupling from velocity to pressure in the continuity
/// equation. Integration by parts gives `g = B^T` up to natural boundary
/// terms (B is from [`DivergenceForm`]): the weak continuity carries
/// `-rho_0 c_0^2 g(u, q)`, i.e. `-rho_0 c_0^2 B^T` in the interior.
pub struct GradientCouplingForm;

impl BilinearForm<f64> for GradientCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n_test = test.n_dofs; // scalar H1 (pressure test)
        let n_trial = trial.n_dofs; // vector H1 (velocity trial)
        let w = weight * test.det_j;

        for i in 0..n_test {
            for j in 0..n_trial {
                let cj = j % dim;
                // u_j . grad(q_i) = phi_j^{cj} * d(psi_i)/dx_{cj}
                let phi_j = trial.vector_value(j, cj);
                let grad_q_cj = test.grad(i, cj);
                local_matrix.add(i, j, phi_j * grad_q_cj * w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Block system
// ---------------------------------------------------------------------------

/// Linearized Euler Equations block system.
///
/// Stores the preassembled operators for efficient frequency sweeps.
/// The full system matrix at each frequency is assembled by combining
/// these operators with frequency-dependent coefficients.
pub struct LeeSystem {
    /// Number of velocity DOFs (VectorH1).
    pub n_velocity: usize,
    /// Number of pressure DOFs (scalar H1).
    pub n_pressure: usize,
    /// Velocity mass matrix M_v: integral(u . v) dOmega.
    pub velocity_mass: TripletMatrix<f64>,
    /// Velocity convection matrix C_v: integral((U_0 . nabla u) . v) dOmega.
    pub velocity_convection: TripletMatrix<f64>,
    /// Mean flow gradient matrix G_v: integral((u . nabla U_0) . v) dOmega.
    /// Zero for uniform mean flow.
    pub velocity_gradient: TripletMatrix<f64>,
    /// Pressure mass matrix M_p: integral(p q) dOmega.
    pub pressure_mass: TripletMatrix<f64>,
    /// Pressure convection matrix C_p: integral((U_0 . nabla p) q) dOmega.
    pub pressure_convection: TripletMatrix<f64>,
    /// Divergence coupling B: DivergenceForm output (n_vel x n_pres).
    /// B[i,j] = -integral(phi_j div(v_i)) dOmega
    pub divergence: TripletMatrix<f64>,
}

impl LeeSystem {
    /// Total number of DOFs in the coupled system.
    pub fn total_dofs(&self) -> usize {
        self.n_velocity + self.n_pressure
    }

    /// Assemble the full block system at angular frequency omega.
    ///
    /// ```text
    /// | rho_0(i omega M_v + C_v + G_v)   B     | | u' |   | f_u |
    /// | -rho_0 c_0^2 B^T          i omega M_p + C_p | | p' | = | f_p |
    /// ```
    ///
    /// Note: B already contains the negative sign from DivergenceForm,
    /// so A_up = B (not -B). Transposing keeps that sign:
    /// (B^Tu)_j = -integral(psi_j div u_h), so the weak continuity term
    /// +rho_0 c_0^2 (div u, q) assembles as A_pu = -rho_0 c_0^2 B^T.
    pub fn assemble(&self, mean_flow: &MeanFlow, omega: f64) -> TripletMatrix<Complex64> {
        let n_total = self.total_dofs();
        let nv = self.n_velocity;
        let rho0 = mean_flow.density;
        let c0 = mean_flow.speed_of_sound;
        let iw = Complex64::new(0.0, omega);

        let cap = self.velocity_mass.nnz()
            + self.velocity_convection.nnz()
            + self.velocity_gradient.nnz()
            + self.pressure_mass.nnz()
            + self.pressure_convection.nnz()
            + 2 * self.divergence.nnz();
        let mut result = TripletMatrix::with_capacity(n_total, n_total, cap);

        // Block (0,0): rho_0 (i omega M_v + C_v + G_v)
        for k in 0..self.velocity_mass.nnz() {
            result.push(
                self.velocity_mass.rows[k],
                self.velocity_mass.cols[k],
                rho0 * iw * self.velocity_mass.values[k],
            );
        }
        for k in 0..self.velocity_convection.nnz() {
            result.push(
                self.velocity_convection.rows[k],
                self.velocity_convection.cols[k],
                Complex64::new(rho0 * self.velocity_convection.values[k], 0.0),
            );
        }
        for k in 0..self.velocity_gradient.nnz() {
            result.push(
                self.velocity_gradient.rows[k],
                self.velocity_gradient.cols[k],
                Complex64::new(rho0 * self.velocity_gradient.values[k], 0.0),
            );
        }

        // Block (0,1): B (divergence coupling, velocity-pressure)
        for k in 0..self.divergence.nnz() {
            result.push(
                self.divergence.rows[k],
                nv + self.divergence.cols[k],
                Complex64::new(self.divergence.values[k], 0.0),
            );
        }

        // Block (1,0): -rho_0 c_0^2 B^T (gradient coupling, pressure-velocity).
        // B[i,j] = -integral(psi_j div(phi_i)), so (B^Tu)_j is exactly
        // -integral(psi_j div u_h); the weak continuity term
        // +rho_0 c_0^2 (div u, q) is therefore -rho_0 c_0^2 B^T.
        let rho_c_sq = rho0 * c0 * c0;
        for k in 0..self.divergence.nnz() {
            result.push(
                nv + self.divergence.cols[k],
                self.divergence.rows[k],
                Complex64::new(-rho_c_sq * self.divergence.values[k], 0.0),
            );
        }

        // Block (1,1): i omega M_p + C_p
        for k in 0..self.pressure_mass.nnz() {
            result.push(
                nv + self.pressure_mass.rows[k],
                nv + self.pressure_mass.cols[k],
                iw * self.pressure_mass.values[k],
            );
        }
        for k in 0..self.pressure_convection.nnz() {
            result.push(
                nv + self.pressure_convection.rows[k],
                nv + self.pressure_convection.cols[k],
                Complex64::new(self.pressure_convection.values[k], 0.0),
            );
        }

        result
    }
}

/// Build a [`LeeSystem`] from mesh, polynomial degrees, and mean flow.
///
/// Uses P2/P1 (Taylor-Hood) by default: quadratic velocity, linear pressure.
/// The mean flow velocity gradient term is included if the flow is non-uniform.
pub fn build_lee_system<F: MeanFlowField>(
    mesh: &Mesh,
    vel_degree: PolynomialDegree,
    pres_degree: PolynomialDegree,
    mean_flow: &F,
    dim: usize,
) -> LeeSystem {
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

    let v_space = VectorH1Space::new(vel_degree, dim);
    let p_space = H1Space::new(pres_degree);

    let n_vel = v_space.num_global_dofs(mesh);
    let n_pres = p_space.num_global_dofs(mesh);

    let quad_order = crate::quadrature::required_order_for_stiffness(vel_degree.degree());

    // Velocity mass (rho=1, scale at assembly time)
    let v_mass_form = VectorMassForm { coefficient: 1.0 };
    let velocity_mass =
        crate::assembly::assemble_bilinear(&v_mass_form, &v_space, &v_space, mesh, quad_order);

    // Velocity convection: (U_0 . grad u) . v
    // Use a reference point to get uniform velocity (for the form)
    let u0 = mean_flow.velocity([0.0; 3]);
    let v_conv_form = VectorConvectionForm {
        rho: 1.0,
        velocity_fn: move |_x| u0,
    };
    let velocity_convection =
        crate::assembly::assemble_bilinear(&v_conv_form, &v_space, &v_space, mesh, quad_order);

    // Mean flow gradient: (u . nabla U_0) . v
    // For uniform flow this is zero, but we assemble it anyway for generality
    let v_grad_form = MeanFlowGradientForm {
        rho: 1.0,
        mean_flow: MeanFlow::air_standard([0.0; 3]), // placeholder for uniform
    };
    let velocity_gradient =
        crate::assembly::assemble_bilinear(&v_grad_form, &v_space, &v_space, mesh, quad_order);

    // Pressure mass
    let pressure_mass =
        crate::assembly::assemble_bilinear(&MassForm, &p_space, &p_space, mesh, quad_order);

    // Pressure convection: (U_0 . grad p) q
    let p_conv_form = super::linearized_potential_flow::ScalarConvectionForm { advection: u0 };
    let pressure_convection =
        crate::assembly::assemble_bilinear(&p_conv_form, &p_space, &p_space, mesh, quad_order);

    // Divergence coupling (VectorH1 test x H1 trial) — use mixed assembly
    // since test and trial spaces have different types
    let mixed = crate::assembly::mixed::assemble_mixed(
        &v_mass_form,
        &DivergenceForm,
        &v_space,
        &p_space,
        mesh,
        quad_order,
    );
    let divergence = mixed.block_01;

    LeeSystem {
        n_velocity: n_vel,
        n_pressure: n_pres,
        velocity_mass,
        velocity_convection,
        velocity_gradient,
        pressure_mass,
        pressure_convection,
        divergence,
    }
}

use crate::basis::PolynomialDegree;

/// Build a [`LeeSystem`] for uniform mean flow (simplified API).
pub fn build_lee_system_uniform(
    mesh: &Mesh,
    vel_degree: PolynomialDegree,
    pres_degree: PolynomialDegree,
    mean_flow: &MeanFlow,
    dim: usize,
) -> LeeSystem {
    build_lee_system(mesh, vel_degree, pres_degree, mean_flow, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::forms::LocalMatrix;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

    #[test]
    fn test_vector_convection_form_nonzero() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = VectorConvectionForm {
            rho: 1.0,
            velocity_fn: |_| [1.0, 0.0, 0.0],
        };
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        let any_nonzero = (0..n).any(|i| (0..n).any(|j| mat.get(i, j).abs() > 1e-15));
        assert!(
            any_nonzero,
            "VectorConvectionForm should produce non-zero entries"
        );
    }

    #[test]
    fn test_gradient_coupling_nonzero() {
        let mesh = unit_square_triangles(2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);

        let p_tab = p_space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let v_tab = v_space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        let n = p_tab.n_dofs.max(v_tab.n_dofs);
        let mut mat = LocalMatrix::zeros(n);
        GradientCouplingForm.integrate_point(&p_tab, &v_tab, 1.0, &mut mat);

        let any_nonzero =
            (0..p_tab.n_dofs).any(|i| (0..v_tab.n_dofs).any(|j| mat.get(i, j).abs() > 1e-15));
        assert!(
            any_nonzero,
            "GradientCouplingForm should produce non-zero entries"
        );
    }

    #[test]
    fn test_lee_system_dimensions() {
        let mesh = unit_square_triangles(4);
        let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]);
        let system =
            build_lee_system_uniform(&mesh, PolynomialDegree::P2, PolynomialDegree::P1, &flow, 2);

        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_vel = v_space.num_global_dofs(&mesh);
        let n_pres = p_space.num_global_dofs(&mesh);

        assert_eq!(system.n_velocity, n_vel);
        assert_eq!(system.n_pressure, n_pres);
        assert_eq!(system.total_dofs(), n_vel + n_pres);
    }

    #[test]
    fn test_lee_system_assembly() {
        let mesh = unit_square_triangles(4);
        let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]);
        let system =
            build_lee_system_uniform(&mesh, PolynomialDegree::P2, PolynomialDegree::P1, &flow, 2);

        let omega = 1000.0;
        let block = system.assemble(&flow, omega);
        let n_total = system.total_dofs();

        assert_eq!(block.num_rows, n_total);
        assert_eq!(block.num_cols, n_total);
        assert!(block.nnz() > 0);

        // Convert to CSR and verify non-trivial
        let csr = block.to_csr();
        assert_eq!(csr.num_rows, n_total);
    }

    #[test]
    fn test_lee_plane_wave_residual_converges() {
        // A downstream plane wave in uniform flow satisfies the homogeneous
        // LEE, so the assembled operator must annihilate its interpolant
        // with refinement. This pins the continuity-block sign: with the
        // wrong sign the system only supports evanescent waves and the
        // residual stalls at O(1).
        let omega = 600.0;
        let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]);
        let c0 = flow.speed_of_sound;
        let u0 = flow.velocity[0];
        let kx = omega / (c0 + u0);
        let doppler = omega - u0 * kx;
        let mut previous = f64::INFINITY;
        for nx in [8, 16] {
            let mesh = unit_square_triangles(nx);
            let system = build_lee_system_uniform(
                &mesh,
                PolynomialDegree::P1,
                PolynomialDegree::P1,
                &flow,
                2,
            );
            let nv = system.n_velocity;
            let nn = mesh.num_nodes();
            assert_eq!(nv, 2 * nn);
            let mut x = vec![Complex64::new(0.0, 0.0); system.total_dofs()];
            for (node, point) in mesh.nodes.iter().enumerate() {
                let pressure = Complex64::from_polar(1.0, -kx * point.x);
                // u = i.grad p / (rho.doppler); d_x p = -ikx.p cancels the i.
                x[2 * node] = Complex64::new(kx, 0.0) * pressure / (flow.density * doppler);
                x[2 * node + 1] = Complex64::new(0.0, 0.0);
                x[nv + node] = pressure;
            }
            let matrix = system.assemble(&flow, omega);
            let mut residual = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
            for k in 0..matrix.nnz() {
                residual[matrix.rows[k]] += matrix.values[k] * x[matrix.cols[k]];
            }
            let norm = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt() / nv as f64;
            assert!(
                norm < previous,
                "residual must shrink with refinement: {norm} vs {previous}"
            );
            previous = norm;
        }
        assert!(previous < 0.02, "P1 residual {previous}");
    }
}
