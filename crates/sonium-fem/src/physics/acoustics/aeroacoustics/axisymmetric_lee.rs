//! Axisymmetric Linearized Euler Equations with azimuthal mode order.
//!
//! Frequency-domain LEE on a meridional `(r, z)` mesh (mesh-x is the
//! radius) with spinning ansatz `exp(imφ)`. The velocity splits into a
//! meridional `(u_r, u_z)` VectorH1 block and an azimuthal `u_φ` scalar
//! H1 block; with non-swirling mean flow the two decouple except through
//! pressure. Only the pressure coupling carries the mode order, as `im`
//! times a plain mass matrix: the r-weighting cancels the cylindrical
//! `1/r`, so no singular quadrature is ever needed.
//!
//! Mean velocity is MESH frame (`[U_r, U_z, _]`, following the
//! axisymmetric LPF convention): uniform axial flow is `[0, U_z, 0]`.
//! Radial or swirling mean components are rejected.
//!
//! # Block system
//!
//! ```text
//! | A_vmvm    0       D_mer   | | u_mer |   | f_vm |
//! | 0         A_vava  im·Eᵀ   | | u_φ   | = | f_va |
//! | −ρ₀c₀²Dᵀ  imρ₀c₀²E A_pp   | | p     |   | f_p  |
//! ```
//!
//! Sign convention: the meridional form `D` carries the divergence-form
//! minus sign (`−∫r·p·∇·v`), so `(Dᵀu) = −(∇·u, q)_r` exactly and the
//! continuity block is `−ρ₀c₀²Dᵀ`. The azimuthal coupling `E` is a plain
//! (unsigned) mass matrix, so both its blocks keep the `+im` factor.
//!
//! Axis regularity is enforced by constraining nodal DOFs (see
//! [`axi_lee_axis_dofs`]), driven by the [`AzimuthalMode`] power table:
//! `m = 0` pins `u_r` and `u_φ`; `|m| = 1` pins `u_z` and `p` while the
//! transverse pair stays finite; `|m| ≥ 2` pins everything.

use num_complex::Complex64;
use sonium_wave::analytical::azimuthal::AzimuthalMode;

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, LocalMatrix, MassForm, VectorMassForm};
use crate::mesh::Mesh;
use crate::physics::acoustics::pressure::axisymmetric::AxiMassForm;
use crate::spaces::{BasisTabulation, FunctionSpace, H1Space, VectorH1Space};

use super::axisymmetric::{AxiScalarConvectionForm, axi_axis_scalar_dofs};
use super::error::AeroacousticsError;
use super::mean_flow::MeanFlow;

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// r-weighted vector mass form for the meridional velocity block.
pub struct AxiVectorMassForm {
    /// Scalar coefficient folded into the quadrature weight.
    pub coefficient: f64,
}

impl BilinearForm<f64> for AxiVectorMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n = test.n_dofs;
        let r = test.physical_coords[0];
        let w = weight * test.det_j * r * self.coefficient;
        for i in 0..n {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            for j in 0..n {
                if j % dim != ci {
                    continue;
                }
                local_matrix.add(i, j, phi_i * trial.vector_value(j, ci) * w);
            }
        }
    }
}

/// r-weighted uniform-axial convection form for vector blocks.
///
/// Integrates `∫ r·U_z·∂_z u·v`; mesh-y is the axis, so the axial
/// derivative is gradient component 1.
pub struct AxiVectorAxialConvectionForm {
    /// Uniform axial velocity `[m/s]` (mesh-frame component 1).
    pub axial_velocity: f64,
}

impl BilinearForm<f64> for AxiVectorAxialConvectionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n = test.n_dofs;
        let r = test.physical_coords[0];
        let w = weight * test.det_j * r * self.axial_velocity;
        for i in 0..n {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            for j in 0..n {
                if j % dim != ci {
                    continue;
                }
                local_matrix.add(i, j, phi_i * trial.grad(j, 1) * w);
            }
        }
    }
}

/// r-weighted meridional divergence form (vector test, scalar trial).
///
/// `D[i,j] = −∫ r·ψ_j·(∂_r v_{i,r} + v_{i,r}/r + ∂_z v_{i,z})`: the
/// `r·(v_r/r)` term cancels analytically, so nothing singular is ever
/// evaluated.
pub struct AxiMeridionalDivergenceForm;

impl BilinearForm<f64> for AxiMeridionalDivergenceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        debug_assert_eq!(dim, 2, "meridional divergence needs a 2-D mesh");
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let r = test.physical_coords[0];
        let w = weight * test.det_j;
        for i in 0..n_test {
            let ci = i % dim;
            for j in 0..n_trial {
                let psi_j = trial.value(j);
                let divergence = if ci == 0 {
                    r * test.grad(i, 0) + test.vector_value(i, 0)
                } else {
                    r * test.grad(i, 1)
                };
                local_matrix.add(i, j, -psi_j * divergence * w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Block system
// ---------------------------------------------------------------------------

/// Axisymmetric LEE block system for one azimuthal order.
///
/// All stored operators are real; the mode order enters only through the
/// `im` coefficients at assembly time.
pub struct AxiLeeSystem {
    /// Azimuthal order this system was built for.
    pub azimuthal: AzimuthalMode,
    /// Meridional velocity DOFs (VectorH1, `[u_r, u_z]` interleaved).
    pub n_meridional: usize,
    /// Azimuthal velocity DOFs (scalar H1, `u_φ`).
    pub n_azimuthal: usize,
    /// Pressure DOFs (scalar H1).
    pub n_pressure: usize,
    /// Meridional mass `∫ r·u·v`.
    pub meridional_mass: TripletMatrix<f64>,
    /// Meridional axial convection `∫ r·U_z·∂_z u·v`.
    pub meridional_convection: TripletMatrix<f64>,
    /// Azimuthal mass `∫ r·u_φ·v_φ`.
    pub azimuthal_mass: TripletMatrix<f64>,
    /// Azimuthal axial convection `∫ r·U_z·∂_z u_φ·v_φ`.
    pub azimuthal_convection: TripletMatrix<f64>,
    /// Pressure mass `∫ r·p·q`.
    pub pressure_mass: TripletMatrix<f64>,
    /// Pressure axial convection `∫ r·U_z·∂_z p·q`.
    pub pressure_convection: TripletMatrix<f64>,
    /// Meridional divergence (meridional-test × pressure-trial).
    pub divergence_meridional: TripletMatrix<f64>,
    /// Azimuthal coupling (pressure-test × azimuthal-trial, plain mass).
    pub coupling_azimuthal: TripletMatrix<f64>,
}

impl AxiLeeSystem {
    /// Total number of DOFs in the coupled system.
    pub fn total_dofs(&self) -> usize {
        self.n_meridional + self.n_azimuthal + self.n_pressure
    }

    /// Assemble the full block system at angular frequency omega.
    ///
    /// The azimuthal pressure coupling applies the stored mode order as
    /// `im` times the plain mass coupling, in both off-diagonal blocks.
    pub fn assemble(&self, mean_flow: &MeanFlow, omega: f64) -> TripletMatrix<Complex64> {
        let n_total = self.total_dofs();
        let n_vm = self.n_meridional;
        let n_va = self.n_azimuthal;
        let va_off = n_vm;
        let p_off = n_vm + n_va;
        let rho0 = mean_flow.density;
        let c0 = mean_flow.speed_of_sound;
        let iw = Complex64::new(0.0, omega);
        let azimuthal_factor = Complex64::new(0.0, f64::from(self.azimuthal.order()));

        let cap = self.meridional_mass.nnz()
            + self.meridional_convection.nnz()
            + self.azimuthal_mass.nnz()
            + self.azimuthal_convection.nnz()
            + self.pressure_mass.nnz()
            + self.pressure_convection.nnz()
            + 2 * self.divergence_meridional.nnz()
            + 2 * self.coupling_azimuthal.nnz();
        let mut result = TripletMatrix::with_capacity(n_total, n_total, cap);

        let mut push_scaled =
            |matrix: &TripletMatrix<f64>, row_off: usize, col_off: usize, scale: Complex64| {
                for k in 0..matrix.nnz() {
                    result.push(
                        row_off + matrix.rows[k],
                        col_off + matrix.cols[k],
                        scale * matrix.values[k],
                    );
                }
            };

        // Diagonal blocks.
        push_scaled(&self.meridional_mass, 0, 0, rho0 * iw);
        push_scaled(&self.meridional_convection, 0, 0, Complex64::new(rho0, 0.0));
        push_scaled(&self.azimuthal_mass, va_off, va_off, rho0 * iw);
        push_scaled(
            &self.azimuthal_convection,
            va_off,
            va_off,
            Complex64::new(rho0, 0.0),
        );
        push_scaled(&self.pressure_mass, p_off, p_off, iw);
        push_scaled(
            &self.pressure_convection,
            p_off,
            p_off,
            Complex64::new(1.0, 0.0),
        );

        // Meridional pressure coupling (divergence form + transpose).
        push_scaled(
            &self.divergence_meridional,
            0,
            p_off,
            Complex64::new(1.0, 0.0),
        );
        // Continuity needs +ρ₀c₀²(∇·u, q)_r, and Dᵀu = −(∇·u, q)_r
        // exactly (D carries the divergence-form minus sign), hence the
        // negation: −ρ₀c₀²Dᵀ.
        for k in 0..self.divergence_meridional.nnz() {
            result.push(
                p_off + self.divergence_meridional.cols[k],
                self.divergence_meridional.rows[k],
                Complex64::new(-rho0 * c0 * c0 * self.divergence_meridional.values[k], 0.0),
            );
        }

        // Azimuthal pressure coupling: im·Eᵀ and im·ρ₀c₀²·E.
        let rho_c_sq = rho0 * c0 * c0;
        for k in 0..self.coupling_azimuthal.nnz() {
            let prow = self.coupling_azimuthal.rows[k];
            let vacol = self.coupling_azimuthal.cols[k];
            let value = self.coupling_azimuthal.values[k];
            result.push(va_off + vacol, p_off + prow, azimuthal_factor * value);
            result.push(
                p_off + prow,
                va_off + vacol,
                azimuthal_factor * rho_c_sq * value,
            );
        }

        result
    }
}

/// Build an [`AxiLeeSystem`] on a meridional mesh.
///
/// The mesh must be 2-D (`x = r`, `y = z`) and the mean flow uniform
/// axial in MESH frame (`[0, U_z, 0]`); radial, swirling, or
/// out-of-plane components are rejected.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for a non-meridional mesh, a
/// non-uniform-axial mean flow, or non-physical flow parameters.
pub fn build_axi_lee_system(
    mesh: &Mesh,
    vel_degree: PolynomialDegree,
    pres_degree: PolynomialDegree,
    mean_flow: &MeanFlow,
    azimuthal: AzimuthalMode,
) -> Result<AxiLeeSystem, AeroacousticsError> {
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type.dimension() != 2)
    {
        return Err(AeroacousticsError::UnsupportedDiscretization {
            detail: "axisymmetric LEE needs a 2-D meridional mesh",
        });
    }
    if mean_flow.velocity[0] != 0.0 || mean_flow.velocity[2] != 0.0 {
        return Err(AeroacousticsError::UnsupportedDiscretization {
            detail: "axisymmetric LEE needs uniform axial mesh-frame flow [0, U_z, 0]",
        });
    }
    if !mean_flow.velocity[1].is_finite()
        || !mean_flow.density.is_finite()
        || !mean_flow.speed_of_sound.is_finite()
        || mean_flow.density <= 0.0
        || mean_flow.speed_of_sound <= 0.0
    {
        return Err(AeroacousticsError::NonFiniteField { name: "mean_flow" });
    }
    let axial_velocity = mean_flow.velocity[1];

    let vm_space = VectorH1Space::new(vel_degree, 2);
    let va_space = H1Space::new(vel_degree);
    let p_space = H1Space::new(pres_degree);

    let n_vm = vm_space.num_global_dofs(mesh);
    let n_va = va_space.num_global_dofs(mesh);
    let n_p = p_space.num_global_dofs(mesh);

    let (quad_stiffness, quad_mass) =
        crate::physics::acoustics::pressure::axisymmetric::axisymmetric_quadrature_orders(
            vel_degree,
        );

    let meridional_mass = crate::assembly::assemble_bilinear(
        &AxiVectorMassForm { coefficient: 1.0 },
        &vm_space,
        &vm_space,
        mesh,
        quad_mass,
    );
    let meridional_convection = crate::assembly::assemble_bilinear(
        &AxiVectorAxialConvectionForm { axial_velocity },
        &vm_space,
        &vm_space,
        mesh,
        quad_stiffness,
    );
    let azimuthal_mass =
        crate::assembly::assemble_bilinear(&AxiMassForm, &va_space, &va_space, mesh, quad_mass);
    let azimuthal_convection = crate::assembly::assemble_bilinear(
        &AxiScalarConvectionForm {
            advection: [0.0, axial_velocity, 0.0],
        },
        &va_space,
        &va_space,
        mesh,
        quad_stiffness,
    );
    let pressure_mass =
        crate::assembly::assemble_bilinear(&AxiMassForm, &p_space, &p_space, mesh, quad_mass);
    let pressure_convection = crate::assembly::assemble_bilinear(
        &AxiScalarConvectionForm {
            advection: [0.0, axial_velocity, 0.0],
        },
        &p_space,
        &p_space,
        mesh,
        quad_stiffness,
    );
    let mixed_meridional = crate::assembly::mixed::assemble_mixed(
        &VectorMassForm { coefficient: 1.0 },
        &AxiMeridionalDivergenceForm,
        &vm_space,
        &p_space,
        mesh,
        quad_stiffness,
    );
    let mixed_azimuthal = crate::assembly::mixed::assemble_mixed(
        &MassForm, &MassForm, &p_space, &va_space, mesh, quad_mass,
    );

    Ok(AxiLeeSystem {
        azimuthal,
        n_meridional: n_vm,
        n_azimuthal: n_va,
        n_pressure: n_p,
        meridional_mass,
        meridional_convection,
        azimuthal_mass,
        azimuthal_convection,
        pressure_mass,
        pressure_convection,
        divergence_meridional: mixed_meridional.block_01,
        coupling_azimuthal: mixed_azimuthal.block_01,
    })
}

/// Global block-system DOFs pinned by axis regularity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxiLeeAxisDofs {
    /// Constrained meridional block indices.
    pub meridional: Vec<usize>,
    /// Constrained azimuthal block indices.
    pub azimuthal: Vec<usize>,
    /// Constrained pressure block indices.
    pub pressure: Vec<usize>,
}

/// Collects the axis-regularity DOFs for an assembled system.
///
/// A velocity component is pinned exactly when its
/// [`AzimuthalMode::velocity_axis_powers`] entry is positive; pressure
/// is pinned exactly for spinning modes. Indices are global block
/// positions matching [`AxiLeeSystem::assemble`].
pub fn axi_lee_axis_dofs(
    system: &AxiLeeSystem,
    mesh: &Mesh,
    vel_degree: PolynomialDegree,
    pres_degree: PolynomialDegree,
) -> AxiLeeAxisDofs {
    let powers = system.azimuthal.velocity_axis_powers();
    let vm_axis = axi_axis_scalar_dofs(mesh, vel_degree);
    let p_axis = axi_axis_scalar_dofs(mesh, pres_degree);

    let mut meridional = Vec::new();
    for scalar in &vm_axis {
        if powers[0] > 0 {
            meridional.push(scalar * 2);
        }
        if powers[2] > 0 {
            meridional.push(scalar * 2 + 1);
        }
    }
    let mut azimuthal = Vec::new();
    if powers[1] > 0 {
        azimuthal.extend(vm_axis.iter().map(|scalar| system.n_meridional + scalar));
    }
    let mut pressure = Vec::new();
    if system.azimuthal.pressure_vanishes_on_axis() {
        pressure.extend(
            p_axis
                .iter()
                .map(|scalar| system.n_meridional + system.n_azimuthal + scalar),
        );
    }
    meridional.sort_unstable();
    azimuthal.sort_unstable();
    pressure.sort_unstable();
    AxiLeeAxisDofs {
        meridional,
        azimuthal,
        pressure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meridional_rectangle(nx: usize, r_in: f64, r_out: f64, length: f64) -> Mesh {
        let mut mesh = crate::mesh::unit_square_triangles(nx);
        for node in mesh.nodes.iter_mut() {
            node.x = r_in + (r_out - r_in) * node.x;
            node.y = length * node.y;
        }
        mesh
    }

    fn triplet_matvec(matrix: &TripletMatrix<Complex64>, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
        for k in 0..matrix.nnz() {
            y[matrix.rows[k]] += matrix.values[k] * x[matrix.cols[k]];
        }
        y
    }

    #[test]
    fn stored_operators_are_mode_independent() {
        let mesh = meridional_rectangle(2, 0.0, 0.5, 1.0);
        let flow = MeanFlow::air_standard([0.0, 34.3, 0.0]);
        let plain = build_axi_lee_system(
            &mesh,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            &flow,
            AzimuthalMode::axisymmetric(),
        )
        .unwrap();
        let spinning = build_axi_lee_system(
            &mesh,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            &flow,
            AzimuthalMode::new(3),
        )
        .unwrap();
        for (left, right) in [
            (&plain.meridional_mass, &spinning.meridional_mass),
            (
                &plain.meridional_convection,
                &spinning.meridional_convection,
            ),
            (&plain.azimuthal_mass, &spinning.azimuthal_mass),
            (&plain.pressure_mass, &spinning.pressure_mass),
            (
                &plain.divergence_meridional,
                &spinning.divergence_meridional,
            ),
            (&plain.coupling_azimuthal, &spinning.coupling_azimuthal),
        ] {
            assert_eq!(left.rows, right.rows);
            assert_eq!(left.cols, right.cols);
            assert_eq!(left.values, right.values);
        }
    }

    #[test]
    fn azimuthal_coupling_sums_to_domain_area() {
        // Partition of unity: Σ E[i,j] = ∫ 1·1 = area.
        let mesh = meridional_rectangle(3, 0.2, 0.7, 2.0);
        let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let system = build_axi_lee_system(
            &mesh,
            PolynomialDegree::P1,
            PolynomialDegree::P1,
            &flow,
            AzimuthalMode::new(1),
        )
        .unwrap();
        let total: f64 = system.coupling_azimuthal.values.iter().sum();
        assert!((total - 0.5 * 2.0).abs() < 1e-12, "coupling sum {total}");
        assert!(
            system
                .coupling_azimuthal
                .values
                .iter()
                .all(|value| *value >= 0.0),
            "mass coupling must be non-negative"
        );
    }

    #[test]
    fn axis_dofs_follow_regularity_table() {
        let mesh = meridional_rectangle(2, 0.0, 1.0, 1.0);
        let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        // unit_square_triangles(2) has 3 axis vertices (x = 0).
        for (order, n_vm, n_va, n_p) in [(0, 3, 3, 0), (1, 3, 0, 3), (2, 6, 3, 3)] {
            let system = build_axi_lee_system(
                &mesh,
                PolynomialDegree::P1,
                PolynomialDegree::P1,
                &flow,
                AzimuthalMode::new(order),
            )
            .unwrap();
            let pinned =
                axi_lee_axis_dofs(&system, &mesh, PolynomialDegree::P1, PolynomialDegree::P1);
            assert_eq!(pinned.meridional.len(), n_vm, "m = {order}");
            assert_eq!(pinned.azimuthal.len(), n_va, "m = {order}");
            assert_eq!(pinned.pressure.len(), n_p, "m = {order}");
            // m = 0 pins the radial component (even interleaved slots).
            if order == 0 {
                assert!(pinned.meridional.iter().all(|dof| dof % 2 == 0));
            }
            // m = 1 pins the axial component (odd interleaved slots).
            if order == 1 {
                assert!(pinned.meridional.iter().all(|dof| dof % 2 == 1));
            }
        }
    }

    #[test]
    fn plane_wave_residual_converges_with_refinement() {
        // Downstream plane wave in uniform axial flow satisfies the
        // homogeneous LEE, so the assembled operator must annihilate its
        // interpolant with refinement (P1/P1, axis included).
        let omega = 800.0;
        let flow = MeanFlow::air_standard([0.0, 34.3, 0.0]);
        let c0 = flow.speed_of_sound;
        let axial = flow.velocity[1];
        let kz = omega / (c0 + axial);
        let doppler = omega - axial * kz;
        let mut previous = f64::INFINITY;
        for nx in [8, 16] {
            let mesh = meridional_rectangle(nx, 0.0, 0.5, 2.0);
            let system = build_axi_lee_system(
                &mesh,
                PolynomialDegree::P1,
                PolynomialDegree::P1,
                &flow,
                AzimuthalMode::axisymmetric(),
            )
            .unwrap();
            let n_vm = system.n_meridional;
            let n_va = system.n_azimuthal;
            let nn = mesh.num_nodes();
            assert_eq!(n_vm, 2 * nn);
            let mut x = vec![Complex64::new(0.0, 0.0); system.total_dofs()];
            for (node, point) in mesh.nodes.iter().enumerate() {
                let pressure = Complex64::from_polar(1.0, -kz * point.y);
                // u = i·∇p/(ρ·ω̃); ∂_z p = −ikz·p cancels the i.
                let axial_velocity = Complex64::new(kz, 0.0) * pressure / (flow.density * doppler);
                x[2 * node] = Complex64::new(0.0, 0.0);
                x[2 * node + 1] = axial_velocity;
                x[n_vm + node] = Complex64::new(0.0, 0.0);
                x[n_vm + n_va + node] = pressure;
            }
            let matrix = system.assemble(&flow, omega);
            let residual = triplet_matvec(&matrix, &x);
            let norm = residual
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt()
                / (n_vm + n_va + nn) as f64;
            assert!(
                norm < previous,
                "residual must shrink: {norm} vs {previous}"
            );
            previous = norm;
        }
        assert!(previous < 0.02, "P1 residual {previous}");
    }

    #[test]
    fn spinning_bessel_mode_residual_converges() {
        use spec_math::Bessel as _;
        // m = 1 Bessel duct mode with uniform axial flow (regular data:
        // J₁(0) = 0 pins pressure on the axis exactly).
        let chi = 1.8411837813406595; // first J₁' zero
        let radius = 0.5;
        let kappa = chi / radius;
        let omega = 3000.0;
        let flow = MeanFlow::air_standard([0.0, 34.3, 0.0]);
        let c0 = flow.speed_of_sound;
        let axial = flow.velocity[1];
        let beta_sq = 1.0 - (axial / c0).powi(2);
        let k = omega / c0;
        let kz = (-k * axial / c0 + (k * k - beta_sq * kappa * kappa).sqrt()) / beta_sq;
        let doppler = omega - axial * kz;
        let mut previous = f64::INFINITY;
        for nx in [8, 16] {
            let mesh = meridional_rectangle(nx, 0.0, radius, 1.5);
            let system = build_axi_lee_system(
                &mesh,
                PolynomialDegree::P1,
                PolynomialDegree::P1,
                &flow,
                AzimuthalMode::new(1),
            )
            .unwrap();
            let n_vm = system.n_meridional;
            let n_va = system.n_azimuthal;
            let nn = mesh.num_nodes();
            let mut x = vec![Complex64::new(0.0, 0.0); system.total_dofs()];
            for (node, point) in mesh.nodes.iter().enumerate() {
                let r = point.x;
                let axial_phase = Complex64::from_polar(1.0, -kz * point.y);
                let (bessel, slope) = if r < 1e-12 {
                    (0.0, 0.5 * kappa)
                } else {
                    let argument = kappa * r;
                    let slope = 0.5 * kappa * (argument.bessel_jv(0.0) - argument.bessel_jv(2.0));
                    (argument.bessel_jv(1.0), slope)
                };
                let pressure = Complex64::new(bessel, 0.0) * axial_phase;
                // u = i∇_cyl p / (ρ·doppler), ∇_cyl = (∂_r, i/r, ∂_z).
                let scale = Complex64::new(0.0, 1.0) / (flow.density * doppler);
                let ur = scale * slope * axial_phase;
                let hoop = if r < 1e-12 {
                    Complex64::new(0.5 * kappa, 0.0) * axial_phase
                } else {
                    Complex64::new(bessel / r, 0.0) * axial_phase
                };
                // u_φ = i·(i·p/r)/(ρ·ω̃) = −(p/r)/(ρ·ω̃): both i cancel.
                let uphi = -hoop / (flow.density * doppler);
                let uz = scale * Complex64::new(0.0, -kz) * pressure;
                x[2 * node] = ur;
                x[2 * node + 1] = uz;
                x[n_vm + node] = uphi;
                x[n_vm + n_va + node] = pressure;
            }
            let matrix = system.assemble(&flow, omega);
            let residual = triplet_matvec(&matrix, &x);
            let norm = residual
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt()
                / (n_vm + n_va + nn) as f64;
            assert!(
                norm < previous,
                "residual must shrink: {norm} vs {previous}"
            );
            previous = norm;
        }
        assert!(previous < 0.05, "m = 1 P1 residual {previous}");
    }

    #[test]
    fn builder_rejects_non_axial_mean_flow() {
        let mesh = meridional_rectangle(1, 0.0, 0.5, 1.0);
        let swirl = MeanFlow::air_standard([10.0, 34.3, 0.0]);
        assert!(matches!(
            build_axi_lee_system(
                &mesh,
                PolynomialDegree::P1,
                PolynomialDegree::P1,
                &swirl,
                AzimuthalMode::axisymmetric(),
            ),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
    }
}
