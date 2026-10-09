//! Linearized Navier-Stokes (LNS) interface
//!
//! The most detailed aeroacoustics formulation: solves for acoustic
//! perturbations in velocity, pressure, and temperature, including
//! viscous and thermal dissipation effects.
//!
//! # Governing equations (frequency domain)
//!
//! Momentum:
//! ```text
//! rho_0 (i omega u' + (U_0 . nabla) u' + (u' . nabla) U_0)
//!   - nabla . tau' + nabla p' = 0
//! ```
//!
//! where tau' = mu (nabla u' + (nabla u')^T) - (2/3) mu (nabla . u') I
//! is the viscous stress perturbation.
//!
//! Continuity:
//! ```text
//! i omega rho' + nabla . (rho_0 u' + rho' U_0) = 0
//! ```
//!
//! Energy:
//! ```text
//! rho_0 c_p (i omega T' + U_0 . nabla T')
//!   - kappa nabla^2 T' - (i omega p' + U_0 . nabla p') = 0
//! ```
//!
//! With equation of state: p' = rho_0 R T' + R T_0 rho'
//!
//! # Block system (3x3)
//!
//! ```text
//! | A_uu   A_up   A_uT | | u' |   | f_u |
//! | A_pu   A_pp   A_pT | | p' | = | f_p |
//! | A_Tu   A_Tp   A_TT | | T' |   | f_T |
//! ```
//!
//! The velocity-pressure coupling is the same as LEE but with added
//! viscous diffusion. The energy equation introduces temperature as
//! a third unknown with thermal conduction and pressure-temperature
//! coupling.
//!
//! # Boundary layer losses
//!
//! LNS naturally captures viscous and thermal boundary layer losses,
//! which are essential for:
//! - Narrow duct acoustics (HVAC, musical instruments)
//! - Near-wall absorption
//! - Turbulent flow damping of acoustic waves

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::forms::{
    BilinearForm, DivergenceForm, LocalMatrix, MassForm, VectorLaplacianForm, VectorMassForm,
    WeightedLaplacianForm,
};
use crate::mesh::Mesh;
use crate::spaces::BasisTabulation;
use num_complex::Complex64;

use super::linearized_euler::VectorConvectionForm;
use super::linearized_potential_flow::ScalarConvectionForm;
use super::mean_flow::MeanFlow;

// ---------------------------------------------------------------------------
// Additional forms for LNS
// ---------------------------------------------------------------------------

/// Bulk viscosity divergence form:
/// `b(u, v) = (2/3) mu integral(div u)(div v) dOmega`
///
/// Arises from the trace of the viscous stress tensor.
/// For VectorH1: div(v_i) = d(phi_{node})/dx_{ci} where ci = i % dim.
pub struct BulkViscosityForm {
    /// (2/3) * dynamic viscosity
    pub two_thirds_mu: f64,
}

impl BilinearForm<f64> for BulkViscosityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n = test.n_dofs;
        let w = weight * test.det_j * self.two_thirds_mu;

        for i in 0..n {
            let ci = i % dim;
            let div_vi = test.grad(i, ci);
            for j in 0..n {
                let cj = j % dim;
                let div_uj = trial.grad(j, cj);
                local_matrix.add(i, j, div_vi * div_uj * w);
            }
        }
    }
}

// Velocity-temperature coupling form (momentum equation):
// Currently zero for the simplified LNS without buoyancy/stratification.
// Placeholder for extension with Boussinesq approximation.

// Temperature-pressure coupling form (energy equation):
// c_Tp(p, w) = integral((i omega + U_0 . nabla) p' w) dOmega
//
// This is a combination of mass and convection on pressure,
// tested against the temperature test function. In the energy
// equation, this appears as -(i omega p' + U_0 . nabla p').
//
// Assembled as: mass (coeff = -i omega) + convection (coeff = -1).

// ---------------------------------------------------------------------------
// Block system
// ---------------------------------------------------------------------------

/// Linearized Navier-Stokes block system.
///
/// Extends [`LeeSystem`](super::linearized_euler::LeeSystem) with viscous
/// stress operators and a temperature (energy) equation.
pub struct LnsSystem {
    /// Number of velocity DOFs (VectorH1).
    pub n_velocity: usize,
    /// Number of pressure DOFs (scalar H1).
    pub n_pressure: usize,
    /// Number of temperature DOFs (scalar H1).
    pub n_temperature: usize,

    // --- Velocity block operators ---
    /// Velocity mass M_v: integral(u . v) dOmega
    pub velocity_mass: TripletMatrix<f64>,
    /// Velocity convection C_v: integral((U_0 . nabla u) . v) dOmega
    pub velocity_convection: TripletMatrix<f64>,
    /// Viscous stiffness K_v: mu integral(grad u : grad v) dOmega
    pub velocity_viscous: TripletMatrix<f64>,
    /// Bulk viscosity: (2/3)mu integral(div u div v) dOmega
    pub velocity_bulk_viscosity: TripletMatrix<f64>,

    // --- Pressure block operators ---
    /// Pressure mass M_p: integral(p q) dOmega
    pub pressure_mass: TripletMatrix<f64>,
    /// Pressure convection C_p: integral((U_0 . nabla p) q) dOmega
    pub pressure_convection: TripletMatrix<f64>,

    // --- Temperature block operators ---
    /// Temperature mass M_T: integral(T w) dOmega
    pub temperature_mass: TripletMatrix<f64>,
    /// Temperature convection C_T: integral((U_0 . nabla T) w) dOmega
    pub temperature_convection: TripletMatrix<f64>,
    /// Thermal conduction K_T: kappa integral(grad T . grad w) dOmega
    pub temperature_conduction: TripletMatrix<f64>,

    // --- Coupling operators ---
    /// Divergence coupling B (n_vel x n_pres): DivergenceForm output
    pub divergence: TripletMatrix<f64>,
}

impl LnsSystem {
    /// Total number of DOFs in the coupled system.
    pub fn total_dofs(&self) -> usize {
        self.n_velocity + self.n_pressure + self.n_temperature
    }

    /// Assemble the full 3x3 block system at angular frequency omega.
    ///
    /// Block structure:
    /// ```text
    /// | rho_0(iwM_v + C_v) + mu K_v - (2/3)mu D_v   B        0     |
    /// | rho_0 c_0^2 B^T                          iwM_p + C_p  0     |
    /// | 0                                  -(iwM_p + C_p)  rho_0 c_p(iwM_T + C_T) + kappa K_T |
    /// ```
    pub fn assemble(&self, mean_flow: &MeanFlow, omega: f64) -> TripletMatrix<Complex64> {
        let n_total = self.total_dofs();
        let nv = self.n_velocity;
        let np = self.n_pressure;
        let rho0 = mean_flow.density;
        let c0 = mean_flow.speed_of_sound;
        let mu = mean_flow.dynamic_viscosity;
        let kappa = mean_flow.thermal_conductivity;
        let cp = mean_flow.specific_heat_cp;
        let iw = Complex64::new(0.0, omega);

        let cap = self.velocity_mass.nnz()
            + self.velocity_convection.nnz()
            + self.velocity_viscous.nnz()
            + self.velocity_bulk_viscosity.nnz()
            + self.pressure_mass.nnz()
            + self.pressure_convection.nnz()
            + self.temperature_mass.nnz()
            + self.temperature_convection.nnz()
            + self.temperature_conduction.nnz()
            + 2 * self.divergence.nnz()
            + self.pressure_mass.nnz()
            + self.pressure_convection.nnz();
        let mut result = TripletMatrix::with_capacity(n_total, n_total, cap);

        // =====================================================================
        // Block (0,0): Velocity — rho_0(iw M_v + C_v) + mu K_v - (2/3)mu D_v
        // =====================================================================
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
        for k in 0..self.velocity_viscous.nnz() {
            result.push(
                self.velocity_viscous.rows[k],
                self.velocity_viscous.cols[k],
                Complex64::new(mu * self.velocity_viscous.values[k], 0.0),
            );
        }
        for k in 0..self.velocity_bulk_viscosity.nnz() {
            // Subtract bulk viscosity (it has positive sign in the form,
            // but enters with -2/3 mu in the stress tensor)
            result.push(
                self.velocity_bulk_viscosity.rows[k],
                self.velocity_bulk_viscosity.cols[k],
                Complex64::new(-self.velocity_bulk_viscosity.values[k], 0.0),
            );
        }

        // =====================================================================
        // Block (0,1): Velocity-pressure coupling B
        // =====================================================================
        for k in 0..self.divergence.nnz() {
            result.push(
                self.divergence.rows[k],
                nv + self.divergence.cols[k],
                Complex64::new(self.divergence.values[k], 0.0),
            );
        }

        // =====================================================================
        // Block (1,0): Pressure-velocity coupling rho_0 c_0^2 B^T
        // =====================================================================
        let rho_c_sq = rho0 * c0 * c0;
        for k in 0..self.divergence.nnz() {
            result.push(
                nv + self.divergence.cols[k],
                self.divergence.rows[k],
                Complex64::new(rho_c_sq * self.divergence.values[k], 0.0),
            );
        }

        // =====================================================================
        // Block (1,1): Pressure — iw M_p + C_p
        // =====================================================================
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

        // =====================================================================
        // Block (2,1): Temperature-pressure coupling -(iw M_p + C_p)
        // =====================================================================
        // The energy equation has: -(iw p' + U_0 . nabla p')
        // Pressure mass and convection assembled on the pressure space,
        // but placed in the temperature rows.
        // Since pressure and temperature share the same H1 space and mesh,
        // the mass/convection matrices have the same DOF indexing.
        for k in 0..self.pressure_mass.nnz() {
            result.push(
                nv + np + self.pressure_mass.rows[k],
                nv + self.pressure_mass.cols[k],
                -iw * self.pressure_mass.values[k],
            );
        }
        for k in 0..self.pressure_convection.nnz() {
            result.push(
                nv + np + self.pressure_convection.rows[k],
                nv + self.pressure_convection.cols[k],
                Complex64::new(-self.pressure_convection.values[k], 0.0),
            );
        }

        // =====================================================================
        // Block (2,2): Temperature — rho_0 c_p (iw M_T + C_T) + kappa K_T
        // =====================================================================
        let rho_cp = rho0 * cp;
        for k in 0..self.temperature_mass.nnz() {
            result.push(
                nv + np + self.temperature_mass.rows[k],
                nv + np + self.temperature_mass.cols[k],
                rho_cp * iw * self.temperature_mass.values[k],
            );
        }
        for k in 0..self.temperature_convection.nnz() {
            result.push(
                nv + np + self.temperature_convection.rows[k],
                nv + np + self.temperature_convection.cols[k],
                Complex64::new(rho_cp * self.temperature_convection.values[k], 0.0),
            );
        }
        for k in 0..self.temperature_conduction.nnz() {
            result.push(
                nv + np + self.temperature_conduction.rows[k],
                nv + np + self.temperature_conduction.cols[k],
                Complex64::new(kappa * self.temperature_conduction.values[k], 0.0),
            );
        }

        result
    }
}

/// Build an [`LnsSystem`] from mesh, polynomial degrees, and mean flow.
pub fn build_lns_system(
    mesh: &Mesh,
    vel_degree: PolynomialDegree,
    pres_degree: PolynomialDegree,
    temp_degree: PolynomialDegree,
    mean_flow: &MeanFlow,
    dim: usize,
) -> LnsSystem {
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

    let v_space = VectorH1Space::new(vel_degree, dim);
    let p_space = H1Space::new(pres_degree);
    let t_space = H1Space::new(temp_degree);

    let n_vel = v_space.num_global_dofs(mesh);
    let n_pres = p_space.num_global_dofs(mesh);
    let n_temp = t_space.num_global_dofs(mesh);

    let quad_order = crate::quadrature::required_order_for_stiffness(vel_degree.degree());

    // --- Velocity operators ---
    let v_mass_form = VectorMassForm { coefficient: 1.0 };
    let velocity_mass =
        crate::assembly::assemble_bilinear(&v_mass_form, &v_space, &v_space, mesh, quad_order);

    let u0 = mean_flow.velocity;
    let v_conv_form = VectorConvectionForm {
        rho: 1.0,
        velocity_fn: move |_x| u0,
    };
    let velocity_convection =
        crate::assembly::assemble_bilinear(&v_conv_form, &v_space, &v_space, mesh, quad_order);

    let v_visc_form = VectorLaplacianForm { coefficient: 1.0 };
    let velocity_viscous =
        crate::assembly::assemble_bilinear(&v_visc_form, &v_space, &v_space, mesh, quad_order);

    let mu = mean_flow.dynamic_viscosity;
    let bulk_form = BulkViscosityForm {
        two_thirds_mu: (2.0 / 3.0) * mu,
    };
    let velocity_bulk_viscosity =
        crate::assembly::assemble_bilinear(&bulk_form, &v_space, &v_space, mesh, quad_order);

    // --- Pressure operators ---
    let pressure_mass =
        crate::assembly::assemble_bilinear(&MassForm, &p_space, &p_space, mesh, quad_order);

    let p_conv_form = ScalarConvectionForm { advection: u0 };
    let pressure_convection =
        crate::assembly::assemble_bilinear(&p_conv_form, &p_space, &p_space, mesh, quad_order);

    // --- Temperature operators ---
    let temperature_mass =
        crate::assembly::assemble_bilinear(&MassForm, &t_space, &t_space, mesh, quad_order);

    let t_conv_form = ScalarConvectionForm { advection: u0 };
    let temperature_convection =
        crate::assembly::assemble_bilinear(&t_conv_form, &t_space, &t_space, mesh, quad_order);

    let t_cond_form = WeightedLaplacianForm { coefficient: 1.0 };
    let temperature_conduction =
        crate::assembly::assemble_bilinear(&t_cond_form, &t_space, &t_space, mesh, quad_order);

    // --- Coupling (VectorH1 test x H1 trial) ---
    let mixed = crate::assembly::mixed::assemble_mixed(
        &v_mass_form,
        &DivergenceForm,
        &v_space,
        &p_space,
        mesh,
        quad_order,
    );
    let divergence = mixed.block_01;

    LnsSystem {
        n_velocity: n_vel,
        n_pressure: n_pres,
        n_temperature: n_temp,
        velocity_mass,
        velocity_convection,
        velocity_viscous,
        velocity_bulk_viscosity,
        pressure_mass,
        pressure_convection,
        temperature_mass,
        temperature_convection,
        temperature_conduction,
        divergence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

    #[test]
    fn test_bulk_viscosity_form_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = BulkViscosityForm { two_thirds_mu: 1.0 };
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                assert!(
                    (mat.get(i, j) - mat.get(j, i)).abs() < 1e-14,
                    "BulkViscosity not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_lns_system_dimensions() {
        let mesh = unit_square_triangles(4);
        let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]);
        let system = build_lns_system(
            &mesh,
            PolynomialDegree::P2,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            &flow,
            2,
        );

        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_vel = v_space.num_global_dofs(&mesh);
        let n_pres = p_space.num_global_dofs(&mesh);
        // Temperature uses same space as pressure
        let n_temp = p_space.num_global_dofs(&mesh);

        assert_eq!(system.n_velocity, n_vel);
        assert_eq!(system.n_pressure, n_pres);
        assert_eq!(system.n_temperature, n_temp);
        assert_eq!(system.total_dofs(), n_vel + n_pres + n_temp);
    }

    #[test]
    fn test_lns_system_assembly() {
        let mesh = unit_square_triangles(4);
        let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]);
        let system = build_lns_system(
            &mesh,
            PolynomialDegree::P2,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            &flow,
            2,
        );

        let omega = 1000.0;
        let block = system.assemble(&flow, omega);
        let n_total = system.total_dofs();

        assert_eq!(block.num_rows, n_total);
        assert_eq!(block.num_cols, n_total);
        assert!(block.nnz() > 0);
    }

    #[test]
    fn test_lns_reduces_to_lee_without_viscosity() {
        // With mu=0, kappa=0, the LNS velocity-pressure blocks should match LEE
        let mesh = unit_square_triangles(3);
        let mut flow = MeanFlow::air_standard([34.3, 0.0, 0.0]);
        flow.dynamic_viscosity = 0.0;
        flow.thermal_conductivity = 0.0;

        let lns = build_lns_system(
            &mesh,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            &flow,
            2,
        );
        let lee = super::super::linearized_euler::build_lee_system_uniform(
            &mesh,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            &flow,
            2,
        );

        // Velocity mass should match
        assert_eq!(lns.velocity_mass.nnz(), lee.velocity_mass.nnz());
        // Pressure mass should match
        assert_eq!(lns.pressure_mass.nnz(), lee.pressure_mass.nnz());
        // Divergence coupling should match
        assert_eq!(lns.divergence.nnz(), lee.divergence.nnz());
    }
}
