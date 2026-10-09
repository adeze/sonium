//! Coupled interior-FEM/exterior-BEM Helmholtz solve.
//!
//! The interior acoustic volume is solved with the P1 Helmholtz FEM
//! ([`AcousticsAssembler`], `A = K − k²M`), the exterior radiation load
//! enters through the condensed DtN map `D(ω)` (see [`dtn`]). The aperture
//! couples as a dense low-rank block materialized into the CSR system:
//!
//! ```text
//! ( A_fem − iωρ · B · D⁻¹ · N · Bᵀ ) p = f
//! ```
//!
//! - `B_{i,j} = ∫_{face_j} φ_i dΓ = area_j / 3` for each P1 vertex `i` of
//!   aperture face `j`,
//! - `N = diag(1 / area_j)` maps nodal pressures to element averages,
//! - `f` collects the driver Neumann data `h = iωρ v_drive` on the
//!   driver-marker faces (same convention as
//!   `qa-fem-acoustics-sphere-scattering`).
//!
//! The combined matrix is solved with the existing FGMRES+ILU stack
//! ([`solve_complex`]). A sealed cavity (`exterior = None`) drops the dense
//! block entirely, which is the rigid-termination limit `D⁻¹ → 0`.

use super::dtn::{DtnBuilder, DtnError};
use super::fixtures::CouplingFixture;
use super::mesh_bridge::{MeshBridgeError, build_exterior_bem_mesh};
use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_neumann_complex};
use sonium_fem::mesh::Mesh as FemMesh;
use sonium_fem::physics::acoustics::pressure::{AcousticsAssemblerExt, build_acoustics_assembler};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::sparse::TripletMatrix;
use sonium_solvers::{LinearSolveConfig, solve_complex};
use std::collections::HashMap;

/// Maximum FEM pressure DOFs admitted by the dense condensed reference path.
pub const DEFAULT_DIRECT_DENSE_MAX_DOFS: usize = 4_096;

/// Errors raised by the coupled solve.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CouplingError {
    #[error("mesh bridge failed: {0}")]
    MeshBridge(#[from] MeshBridgeError),
    #[error("DtN condensation failed: {0}")]
    Dtn(#[from] DtnError),
    #[error("linear solve failed: {0}")]
    LinearSolve(String),
    #[error("iterative solver did not converge (residual {residual:.3e})")]
    NotConverged { residual: f64 },
    #[error(
        "dense direct condensed system has {requested} DOFs, exceeding policy maximum {maximum}"
    )]
    DenseDirectSystemTooLarge { requested: usize, maximum: usize },
    #[error("driver drive requested but the fixture has no driver marker")]
    MissingDriverMarker,
    #[error("at least one distributed driver velocity profile is required")]
    EmptyDriverVelocityProfiles,
    #[error(
        "distributed driver velocity profile must have finite value, centre, and positive radius"
    )]
    InvalidDriverVelocityProfile,
}

/// Interior drive specification for one frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriverVelocityProfile {
    /// RMS velocity amplitude on one spatially localized driver patch.
    pub value: Complex64,
    pub center_x_m: f64,
    pub center_y_m: f64,
    pub radius_m: f64,
}

impl DriverVelocityProfile {
    fn validate(&self) -> Result<(), CouplingError> {
        if self.value.re.is_finite()
            && self.value.im.is_finite()
            && self.center_x_m.is_finite()
            && self.center_y_m.is_finite()
            && self.radius_m.is_finite()
            && self.radius_m > 0.0
        {
            Ok(())
        } else {
            Err(CouplingError::InvalidDriverVelocityProfile)
        }
    }
}

fn validate_driver_velocity_profiles(
    profiles: &[DriverVelocityProfile],
) -> Result<(), CouplingError> {
    if profiles.is_empty() {
        return Err(CouplingError::EmptyDriverVelocityProfiles);
    }
    for profile in profiles {
        profile.validate()?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum CouplingDrive {
    /// No interior source (`f = 0`).
    None,
    /// Uniform outward normal velocity (m/s, complex amplitude) on the
    /// driver-marker faces — a rigid piston drive.
    DriverVelocity(Complex64),
    /// Sum of localized driver patches on the FEM driver boundary.  The
    /// spatial profiles are assembled into one distributed RHS, preserving
    /// mutual cavity loading before the exterior DtN exchange.
    DriverVelocityProfiles(Vec<DriverVelocityProfile>),
    /// Outward normal velocity `value` on the part of the driver-marker
    /// faces with `x ≤ x_split` and `y ≤ y_split`, zero elsewhere. A partial
    /// piston breaks the symmetry of a uniform drive, which has zero modal
    /// projection onto all odd cavity modes (e.g. a uniform face piston
    /// never excites the (1,0,0) box mode).
    DriverPartialPiston {
        value: Complex64,
        x_split: f64,
        y_split: f64,
    },
}

/// Aperture interface data extracted from the FEM boundary faces.
#[derive(Debug, Clone, Default)]
pub(super) struct ApertureInterface {
    /// `(fem node triple, area)` per aperture face, aligned with the BEM
    /// aperture element ordering produced by
    /// [`build_exterior_bem_mesh`] (both follow the FEM boundary order).
    pub(super) faces: Vec<([usize; 3], f64)>,
}

impl ApertureInterface {
    fn from_mesh(mesh: &FemMesh, aperture_marker: i32) -> Self {
        let faces = mesh
            .boundaries
            .iter()
            .filter(|b| b.marker == aperture_marker)
            .map(|b| {
                let [a, b_, c] = [b.nodes[0], b.nodes[1], b.nodes[2]];
                let pa = &mesh.nodes[a];
                let pb = &mesh.nodes[b_];
                let pc = &mesh.nodes[c];
                let v1 = [pb.x - pa.x, pb.y - pa.y, pb.z - pa.z];
                let v2 = [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z];
                let cx = v1[1] * v2[2] - v1[2] * v2[1];
                let cy = v1[2] * v2[0] - v1[0] * v2[2];
                let cz = v1[0] * v2[1] - v1[1] * v2[0];
                let area = 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
                ([a, b_, c], area)
            })
            .collect();
        Self { faces }
    }

    fn areas(&self) -> Vec<f64> {
        self.faces.iter().map(|f| f.1).collect()
    }

    /// Element-average pressures `p̄_k = (1/3) Σ_{l ∈ face_k} p_l`.
    pub(super) fn element_average_pressures(&self, p: &[Complex64]) -> Vec<Complex64> {
        self.faces
            .iter()
            .map(|(nodes, _)| (p[nodes[0]] + p[nodes[1]] + p[nodes[2]]) / 3.0)
            .collect()
    }
}

/// Coupled interior-FEM/exterior-BEM model for one geometry.
pub struct FemBemCoupling {
    pub(super) mesh: FemMesh,
    pub(super) assembler: sonium_fem::physics::acoustics::pressure::AcousticsAssembler,
    pub(super) aperture: ApertureInterface,
    pub(super) exterior: Option<DtnBuilder>,
    pub(super) driver_marker: Option<i32>,
    pub(super) speed_of_sound: f64,
    pub(super) density: f64,
}

impl FemBemCoupling {
    /// Build the coupled model from a fixture: the exterior BEM surface is
    /// derived from the FEM boundary faces (aperture + wall + driver
    /// markers), keeping the 1:1 element alignment.
    pub fn from_fixture(
        fixture: &CouplingFixture,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<Self, CouplingError> {
        let mut wall_markers = vec![fixture.wall_marker];
        if let Some(driver) = fixture.driver_marker {
            wall_markers.push(driver);
        }
        let bem =
            build_exterior_bem_mesh(&fixture.fem_mesh, fixture.aperture_marker, &wall_markers)?;
        // Sanity: the DtN interface ordering must match the FEM aperture
        // face ordering (both follow the FEM boundary-face order).
        debug_assert_eq!(bem.aperture_elements.len(), {
            fixture
                .fem_mesh
                .boundaries
                .iter()
                .filter(|b| b.marker == fixture.aperture_marker)
                .count()
        });
        let exterior = DtnBuilder::new(bem.mesh, bem.aperture_elements, speed_of_sound, density)?;
        Ok(Self::assemble_parts(
            fixture.fem_mesh.clone(),
            fixture.aperture_marker,
            fixture.driver_marker,
            Some(exterior),
            speed_of_sound,
            density,
        ))
    }

    /// Build a sealed-cavity model (no exterior, rigid walls everywhere).
    pub fn sealed(
        fem_mesh: FemMesh,
        aperture_marker: i32,
        driver_marker: Option<i32>,
        speed_of_sound: f64,
        density: f64,
    ) -> Self {
        Self::assemble_parts(
            fem_mesh,
            aperture_marker,
            driver_marker,
            None,
            speed_of_sound,
            density,
        )
    }

    pub(super) fn assemble_parts(
        mesh: FemMesh,
        aperture_marker: i32,
        driver_marker: Option<i32>,
        exterior: Option<DtnBuilder>,
        speed_of_sound: f64,
        density: f64,
    ) -> Self {
        let assembler = build_acoustics_assembler(&mesh, PolynomialDegree::P1);
        let aperture = ApertureInterface::from_mesh(&mesh, aperture_marker);
        Self {
            mesh,
            assembler,
            aperture,
            exterior,
            driver_marker,
            speed_of_sound,
            density,
        }
    }

    /// Number of interior pressure DOFs.
    pub fn num_dofs(&self) -> usize {
        self.mesh.num_nodes()
    }

    /// Number of aperture interface elements.
    pub fn num_aperture_elements(&self) -> usize {
        self.aperture.faces.len()
    }

    /// Aperture element areas.
    pub fn aperture_element_areas(&self) -> Vec<f64> {
        self.aperture.areas()
    }

    /// The exterior DtN builder, if this model has an exterior.
    pub fn exterior(&self) -> Option<&DtnBuilder> {
        self.exterior.as_ref()
    }

    /// Driver Neumann RHS at one frequency: `h = iωρ v_drive` (outward
    /// normal velocity) on the driver-marker faces. Shared by the full
    /// coupled solve and the reduced-order path in [`super::rom`].
    pub(super) fn drive_rhs(
        &self,
        omega: f64,
        drive: CouplingDrive,
    ) -> Result<Vec<Complex64>, CouplingError> {
        let n = self.num_dofs();
        let mut system = FemSystem::new(
            TripletMatrix::<Complex64>::new(n, n),
            vec![Complex64::new(0.0, 0.0); n],
        );
        let i_omega_rho = Complex64::new(0.0, omega * self.density);
        match drive {
            CouplingDrive::None => {}
            CouplingDrive::DriverVelocity(v_drive) => {
                let driver_marker = self
                    .driver_marker
                    .ok_or(CouplingError::MissingDriverMarker)?;
                let h = i_omega_rho * v_drive;
                apply_neumann_complex(&mut system, &self.mesh, driver_marker, &move |_, _, _| h);
            }
            CouplingDrive::DriverVelocityProfiles(profiles) => {
                let driver_marker = self
                    .driver_marker
                    .ok_or(CouplingError::MissingDriverMarker)?;
                validate_driver_velocity_profiles(&profiles)?;
                for profile in profiles {
                    let h = i_omega_rho * profile.value;
                    apply_neumann_complex(
                        &mut system,
                        &self.mesh,
                        driver_marker,
                        &move |x, y, _| {
                            let dx = x - profile.center_x_m;
                            let dy = y - profile.center_y_m;
                            if dx * dx + dy * dy <= profile.radius_m * profile.radius_m {
                                h
                            } else {
                                Complex64::new(0.0, 0.0)
                            }
                        },
                    );
                }
            }
            CouplingDrive::DriverPartialPiston {
                value,
                x_split,
                y_split,
            } => {
                let driver_marker = self
                    .driver_marker
                    .ok_or(CouplingError::MissingDriverMarker)?;
                let h = i_omega_rho * value;
                apply_neumann_complex(&mut system, &self.mesh, driver_marker, &move |x, y, _| {
                    if x <= x_split && y <= y_split {
                        h
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                });
            }
        }
        Ok(system.rhs)
    }

    /// Solve the coupled system at one frequency.
    pub fn solve_frequency(
        &self,
        frequency_hz: f64,
        drive: CouplingDrive,
        config: &LinearSolveConfig,
    ) -> Result<CoupledSolution, CouplingError> {
        let n = self.num_dofs();
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let k = omega / self.speed_of_sound;

        // FEM Helmholtz block.
        let csr_fem = self
            .assembler
            .assemble_helmholtz(Complex64::new(k, 0.0), &HashMap::new());

        // DtN dense block: W = −iωρ · B · D⁻¹ · N · Bᵀ over the aperture
        // DOFs. Per face pair (j, k) each vertex pair (i ∈ j, l ∈ k) gets
        // −iωρ · area_j · G_{j,k} / 9.
        let mut dtn_inverse = None;
        let mut triplets = Vec::with_capacity(csr_fem.nnz());
        for i in 0..n {
            for (j, v) in csr_fem.row_entries(i) {
                triplets.push((i, j, v));
            }
        }
        if let Some(exterior) = &self.exterior {
            let factors = exterior.factor(frequency_hz)?;
            let dtn = factors.build_dtn()?;
            let g = dtn.inverse()?;
            let m = self.aperture.faces.len();
            let minus_i_omega_rho = Complex64::new(0.0, -omega * self.density);
            for j in 0..m {
                let (nodes_j, area_j) = self.aperture.faces[j];
                for k_ in 0..m {
                    let g_jk = g[j * m + k_];
                    if g_jk.norm_sqr() == 0.0 {
                        continue;
                    }
                    let coeff = minus_i_omega_rho * (area_j / 9.0) * g_jk;
                    let (nodes_k, _) = self.aperture.faces[k_];
                    for &row in &nodes_j {
                        for &col in &nodes_k {
                            triplets.push((row, col, coeff));
                        }
                    }
                }
            }
            dtn_inverse = Some(g);
        }
        let matrix = CsrMatrix::from_triplets(n, n, triplets);

        let rhs = self.drive_rhs(omega, drive)?;
        let result = solve_complex(&matrix, &rhs, config)
            .map_err(|e| CouplingError::LinearSolve(e.to_string()))?;
        if !result.converged {
            return Err(CouplingError::NotConverged {
                residual: result.residual,
            });
        }

        // Recover aperture element velocities: v = D⁻¹ · p̄.
        let aperture_velocity = match &dtn_inverse {
            Some(g) => {
                let p_avg = self.aperture.element_average_pressures(&result.x);
                let m = p_avg.len();
                let mut v = vec![Complex64::new(0.0, 0.0); m];
                for j in 0..m {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for k_ in 0..m {
                        acc += g[j * m + k_] * p_avg[k_];
                    }
                    v[j] = acc;
                }
                v
            }
            None => Vec::new(),
        };

        Ok(CoupledSolution {
            frequency_hz,
            pressure: result.x,
            aperture_velocity,
            solver_iterations: result.iterations,
            relative_residual: Some(result.residual),
        })
    }

    /// Solve the same condensed FEM/DtN system with a dense LU reference.
    ///
    /// This path is intentionally separate from the production iterative
    /// solver: it provides a small, deterministic algebraic oracle for
    /// monolithic FEM/BEM elimination and must only be used within the dense
    /// reference-size policy. The assembled operator and DtN inverse are the
    /// same as [`Self::solve_frequency`].
    pub fn solve_frequency_direct_dense(
        &self,
        frequency_hz: f64,
        drive: CouplingDrive,
    ) -> Result<CoupledSolution, CouplingError> {
        let n = self.num_dofs();
        if n > DEFAULT_DIRECT_DENSE_MAX_DOFS {
            return Err(CouplingError::DenseDirectSystemTooLarge {
                requested: n,
                maximum: DEFAULT_DIRECT_DENSE_MAX_DOFS,
            });
        }
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let k = omega / self.speed_of_sound;
        let csr_fem = self
            .assembler
            .assemble_helmholtz(Complex64::new(k, 0.0), &HashMap::new());
        let mut matrix = DenseMatrix::zeros(n, n);
        for row in 0..n {
            for (column, value) in csr_fem.row_entries(row) {
                matrix[[row, column]] += value;
            }
        }

        let mut dtn_inverse = None;
        if let Some(exterior) = &self.exterior {
            let factors = exterior.factor(frequency_hz)?;
            let dtn = factors.build_dtn()?;
            let inverse = dtn.inverse()?;
            let m = self.aperture.faces.len();
            let minus_i_omega_rho = Complex64::new(0.0, -omega * self.density);
            for j in 0..m {
                let (nodes_j, area_j) = self.aperture.faces[j];
                for k_ in 0..m {
                    let coefficient = minus_i_omega_rho * (area_j / 9.0) * inverse[j * m + k_];
                    if coefficient.norm_sqr() == 0.0 {
                        continue;
                    }
                    let (nodes_k, _) = self.aperture.faces[k_];
                    for &row in &nodes_j {
                        for &column in &nodes_k {
                            matrix[[row, column]] += coefficient;
                        }
                    }
                }
            }
            dtn_inverse = Some(inverse);
        }

        let rhs = self.drive_rhs(omega, drive)?;
        let factor =
            lu_factorize(&matrix).map_err(|error| CouplingError::LinearSolve(error.to_string()))?;
        let state = factor
            .solve(&rhs)
            .map_err(|error| CouplingError::LinearSolve(error.to_string()))?;
        if state.iter().any(|value| !value.is_finite()) {
            return Err(CouplingError::LinearSolve(
                "dense direct solve produced a non-finite state".into(),
            ));
        }

        let mut residual_norm_sq = 0.0;
        let mut rhs_norm_sq = 0.0;
        for row in 0..n {
            let mut product = Complex64::default();
            for column in 0..n {
                product += matrix[[row, column]] * state[column];
            }
            residual_norm_sq += (product - rhs[row]).norm_sqr();
            rhs_norm_sq += rhs[row].norm_sqr();
        }
        let relative_residual = if rhs_norm_sq > 0.0 {
            residual_norm_sq.sqrt() / rhs_norm_sq.sqrt()
        } else {
            residual_norm_sq.sqrt()
        };
        if !relative_residual.is_finite() {
            return Err(CouplingError::LinearSolve(
                "dense direct solve produced a non-finite residual".into(),
            ));
        }

        let aperture_velocity = match dtn_inverse {
            Some(inverse) => {
                let pressure = self.aperture.element_average_pressures(&state);
                let m = pressure.len();
                (0..m)
                    .map(|row| {
                        (0..m)
                            .map(|column| inverse[row * m + column] * pressure[column])
                            .sum()
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        Ok(CoupledSolution {
            frequency_hz,
            pressure: state,
            aperture_velocity,
            solver_iterations: 1,
            relative_residual: Some(relative_residual),
        })
    }
}

/// Solution of the coupled system at one frequency.
#[derive(Debug, Clone)]
pub struct CoupledSolution {
    pub frequency_hz: f64,
    /// Interior nodal pressures (Pa, complex amplitude).
    pub pressure: Vec<Complex64>,
    /// Aperture element outward velocities (m/s); empty for sealed models.
    pub aperture_velocity: Vec<Complex64>,
    pub solver_iterations: usize,
    /// Relative linear-system residual when the solve retained one. Reduced
    /// models may omit this evidence when their residual check is disabled;
    /// product reference promotion must reject that state.
    pub relative_residual: Option<f64>,
}

impl CoupledSolution {
    /// Aperture volume velocity `U = Σ_j v_j · area_j` (m³/s).
    pub fn aperture_volume_velocity(&self, element_areas: &[f64]) -> Complex64 {
        self.aperture_velocity
            .iter()
            .zip(element_areas.iter())
            .map(|(v, a)| v * *a)
            .sum()
    }

    /// Mean absolute interior pressure (a response scalar for sweeps).
    pub fn mean_abs_pressure(&self) -> f64 {
        if self.pressure.is_empty() {
            return 0.0;
        }
        self.pressure.iter().map(|p| p.norm()).sum::<f64>() / self.pressure.len() as f64
    }

    /// Complex mean interior pressure, retaining phase for downstream
    /// structural loading.  The absolute-pressure observable above remains
    /// useful for conservative scalar reports.
    pub fn mean_pressure(&self) -> Complex64 {
        if self.pressure.is_empty() {
            return Complex64::default();
        }
        self.pressure.iter().copied().sum::<Complex64>() / self.pressure.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_driver_profiles_fail_closed_when_malformed() {
        assert_eq!(
            validate_driver_velocity_profiles(&[]),
            Err(CouplingError::EmptyDriverVelocityProfiles)
        );
        assert_eq!(
            validate_driver_velocity_profiles(&[DriverVelocityProfile {
                value: Complex64::new(f64::NAN, 0.0),
                center_x_m: 0.0,
                center_y_m: 0.0,
                radius_m: 0.01,
            }]),
            Err(CouplingError::InvalidDriverVelocityProfile)
        );
        assert_eq!(
            validate_driver_velocity_profiles(&[DriverVelocityProfile {
                value: Complex64::new(1.0e-3, 0.0),
                center_x_m: 0.0,
                center_y_m: f64::INFINITY,
                radius_m: 0.01,
            }]),
            Err(CouplingError::InvalidDriverVelocityProfile)
        );
    }

    #[test]
    fn coupled_solution_observables_handle_sealed_and_ported_cases() {
        let sealed = CoupledSolution {
            frequency_hz: 100.0,
            pressure: vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 2.0)],
            aperture_velocity: Vec::new(),
            solver_iterations: 7,
            relative_residual: None,
        };
        assert_eq!(sealed.aperture_volume_velocity(&[]), Complex64::default());
        assert!((sealed.mean_abs_pressure() - 3.5).abs() < 1.0e-12);

        let ported = CoupledSolution {
            frequency_hz: 100.0,
            pressure: vec![Complex64::new(1.0, 0.0)],
            aperture_velocity: vec![Complex64::new(2.0, -1.0), Complex64::new(-0.5, 0.25)],
            solver_iterations: 3,
            relative_residual: Some(1.0e-9),
        };
        assert_eq!(
            ported.aperture_volume_velocity(&[0.1, 0.2]),
            Complex64::new(0.1, -0.05)
        );
    }
}
