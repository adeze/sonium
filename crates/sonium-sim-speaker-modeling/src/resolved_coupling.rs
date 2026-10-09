//! Two-way coupling between the distributed driver solve and the resolved
//! enclosure/port/radiation acoustics.
//!
//! The whole-speaker pipeline hands the area-averaged diaphragm velocity
//! one-way to the resolved acoustic stage. This module closes the loop with
//! a frequency-local load condensation plus a fixed point:
//!
//! 1. The resolved acoustic model is linear in its diaphragm drive, so one
//!    unit-velocity solve per frequency condenses it to the exact piston
//!    load impedance `Z_resolved(ω) = F_opposing / v̄` (see
//!    [`ResolvedAcousticModel::piston_load_impedance`]).
//! 2. The distributed stage-3 system already loads the diaphragm with its
//!    compact front fluid domain, `Z_compact = −(Σ Cᵀp) / v̄`. The feedback
//!    injects only the difference `ΔZ = Z_resolved − Z_compact` so the
//!    converged diaphragm sees exactly the resolved load, never both.
//! 3. `ΔZ` enters the monolithic system through the normalized interface
//!    load distribution (`iω·ΔZ·M_Γ` on the solid block, installed by
//!    [`AxisymmetricLinearModel::install_interface_load`]), and a fixed
//!    point iterates the scalar, area-averaged velocity to a tight tolerance.
//!    This rank-1 closure intentionally omits higher diaphragm/aperture modes;
//!    the fidelity limit is explicit so callers can select the distributed
//!    FEM/BEM path when those modes are important. All phasors
//!    use `exp(+iωt)` on both sides, so no conjugation is needed; passivity
//!    of the total load is `Re(Z_resolved) ≥ 0`.

use num_complex::Complex64;
use rayon::prelude::*;
use sonium_fem::physics::acoustics::electroacoustics::{
    CoupledFrequencyContinuation, CoupledIterativeError,
};

use crate::axisymmetric_linear::{
    AxisymmetricLinearError, AxisymmetricLinearModel, AxisymmetricLinearSolution,
};
use crate::resolved_acoustics::{
    ResolvedAcousticModel, ResolvedAcousticSolution, ResolvedFrequencyContinuation,
    ResolvedSpeakerAcousticError,
};

// Discretization/PML roundoff can produce small negative real loads on a
// reactive solution; reject only materially active loads.
const PASSIVITY_REL_TOLERANCE: f64 = 1.0e-4;
const PASSIVITY_ABS_TOLERANCE: f64 = 1.0e-3;

fn validate_resolved_load(
    frequency_hz: f64,
    z_resolved: Complex64,
) -> Result<(), ResolvedCouplingError> {
    let scale = z_resolved.norm().max(f64::MIN_POSITIVE);
    if !frequency_hz.is_finite()
        || !z_resolved.re.is_finite()
        || !z_resolved.im.is_finite()
        || z_resolved.re < -PASSIVITY_ABS_TOLERANCE.max(PASSIVITY_REL_TOLERANCE * scale)
    {
        return Err(ResolvedCouplingError::NonPassiveResolvedLoad {
            frequency_hz,
            real: z_resolved.re,
            imaginary: z_resolved.im,
        });
    }
    Ok(())
}

/// Fixed-point controls for the resolved-load coupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedCouplingConfig {
    /// Relative velocity change at which the fixed point is accepted.
    pub velocity_tolerance: f64,
    /// Maximum load updates per frequency.
    pub max_iterations: usize,
    /// Fixed-point under-relaxation factor in (0, 1]. The condensed load is
    /// exact and the iteration map is affine, so full steps (1.0) converge
    /// in two to three iterations; smaller values damp oscillation on
    /// near-resonant points.
    pub under_relaxation: f64,
    /// Maximum observed ratio between consecutive fixed-point updates.
    /// Values at or above one are not contractive and are rejected even if a
    /// finite iteration happens to satisfy the velocity tolerance.
    pub max_contraction_ratio: f64,
    /// Relative velocity agreement required against the direct condensed
    /// (monolithic interface) reference solve.
    pub monolithic_agreement_tolerance: f64,
}

impl Default for ResolvedCouplingConfig {
    fn default() -> Self {
        Self {
            velocity_tolerance: 1.0e-6,
            max_iterations: 8,
            under_relaxation: 1.0,
            max_contraction_ratio: 0.99,
            monolithic_agreement_tolerance: 1.0e-5,
        }
    }
}

impl ResolvedCouplingConfig {
    pub fn validate(&self) -> Result<(), ResolvedCouplingError> {
        if !self.velocity_tolerance.is_finite()
            || self.velocity_tolerance <= 0.0
            || self.max_iterations == 0
            || !self.under_relaxation.is_finite()
            || !(0.0..=1.0).contains(&self.under_relaxation)
            || self.under_relaxation <= 0.0
            || !self.max_contraction_ratio.is_finite()
            || !(0.0..1.0).contains(&self.max_contraction_ratio)
            || !self.monolithic_agreement_tolerance.is_finite()
            || self.monolithic_agreement_tolerance <= 0.0
        {
            return Err(ResolvedCouplingError::InvalidConfiguration(
                "velocity tolerance, iteration budget, and relaxation must be finite and in range",
            ));
        }
        Ok(())
    }
}

/// Converged two-way coupled result for one frequency.
#[derive(Debug, Clone)]
pub struct ResolvedCouplingPoint {
    pub frequency_hz: f64,
    /// Exact condensed load of the resolved acoustic stage, `F / v̄`.
    pub z_resolved_n_s_per_m: Complex64,
    /// Load the stage-3 compact fluid domain applies to the diaphragm,
    /// extracted from the converged monolithic solution as `−(Σ Cᵀp) / v̄`.
    pub z_compact_n_s_per_m: Complex64,
    /// Feedback load injected into the solid block, `Z_resolved − Z_compact`.
    pub delta_z_n_s_per_m: Complex64,
    /// Number of load updates performed (one stage-3 re-solve each).
    pub iterations: usize,
    /// Relative velocity change of the final fixed-point step.
    pub relative_velocity_change: f64,
    /// Relative velocity change after every update, for convergence audits.
    pub velocity_change_history: Vec<f64>,
    /// Largest ratio of consecutive fixed-point update changes.
    pub maximum_contraction_ratio: f64,
    pub converged: bool,
    /// Velocity from the direct condensed interface reference solve.
    pub monolithic_reference_velocity_rms_m_per_s: Complex64,
    /// Relative difference between the partitioned fixed point and the
    /// direct reference solve.
    pub monolithic_agreement_relative_error: f64,
    /// Converged area-averaged diaphragm velocity (the resolved-stage drive).
    pub diaphragm_velocity_rms_m_per_s: Complex64,
    /// Converged stage-3 solution including the injected load.
    pub linear_solution: AxisymmetricLinearSolution,
    /// Resolved acoustic fields driven by the converged velocity.
    pub acoustic_solution: ResolvedAcousticSolution,
}

impl ResolvedCouplingPoint {
    /// Total acoustic load the diaphragm reacts against at the fixed point,
    /// `Z_compact + ΔZ`; equals `Z_resolved` up to the velocity tolerance.
    pub fn total_acoustic_load_n_s_per_m(&self) -> Complex64 {
        self.z_compact_n_s_per_m + self.delta_z_n_s_per_m
    }

    /// Time-averaged power absorbed by the total acoustic load,
    /// `Re(Z_total·|v̄|²)`; non-negative for a passive acoustic domain.
    pub fn acoustic_load_absorbed_power_w(&self) -> f64 {
        (self.total_acoustic_load_n_s_per_m() * self.diaphragm_velocity_rms_m_per_s.norm_sqr()).re
    }

    /// Enforce the coupling acceptance contract at a typed boundary.
    ///
    /// Callers that promote a resolved point into a driver model or optimizer
    /// state must not merely inspect the diagnostic boolean and continue with
    /// a non-converged load.
    pub fn require_converged(self) -> Result<Self, ResolvedCouplingError> {
        if self.converged {
            Ok(self)
        } else {
            Err(ResolvedCouplingError::NotConverged {
                frequency_hz: self.frequency_hz,
                iterations: self.iterations,
                relative_velocity_change: self.relative_velocity_change,
            })
        }
    }
}

/// Drives the per-frequency load condensation and fixed point, reusing the
/// stage-3 and stage-4 continuation caches (warm starts and in-band ILU
/// factors) across iterations and frequencies.
pub struct ResolvedLoadCoupler<'a> {
    linear: &'a AxisymmetricLinearModel,
    acoustic: &'a ResolvedAcousticModel,
    config: ResolvedCouplingConfig,
    linear_continuation: CoupledFrequencyContinuation<'a>,
    /// Independent continuation cache for the direct reference re-solve.
    reference_continuation: CoupledFrequencyContinuation<'a>,
    acoustic_continuation: ResolvedFrequencyContinuation<'a>,
    /// Last converged `ΔZ`, used as the fixed-point seed at the next
    /// frequency of a sweep.
    previous_delta_z: Complex64,
}

impl<'a> ResolvedLoadCoupler<'a> {
    pub fn new(
        linear: &'a AxisymmetricLinearModel,
        acoustic: &'a ResolvedAcousticModel,
        config: ResolvedCouplingConfig,
    ) -> Result<Self, ResolvedCouplingError> {
        config.validate()?;
        if linear.coupled_system.interface_load.is_none() {
            return Err(ResolvedCouplingError::MissingInterfaceLoad);
        }
        let linear_continuation = CoupledFrequencyContinuation::new(
            &linear.coupled_system,
            linear.config.iterative_solver.clone(),
        )?;
        let reference_continuation = CoupledFrequencyContinuation::new(
            &linear.coupled_system,
            linear.config.iterative_solver.clone(),
        )?;
        Ok(Self {
            linear,
            acoustic,
            config,
            linear_continuation,
            reference_continuation,
            acoustic_continuation: ResolvedFrequencyContinuation::new(acoustic),
            previous_delta_z: Complex64::default(),
        })
    }

    /// Solve contiguous frequency lanes in parallel. Every lane constructs
    /// its own fixed-point and acoustic continuation caches.
    pub fn solve_sweep_parallel(
        linear: &'a AxisymmetricLinearModel,
        acoustic: &'a ResolvedAcousticModel,
        config: ResolvedCouplingConfig,
        frequencies_hz: &[f64],
        voltage_rms_v: Complex64,
    ) -> Result<Vec<ResolvedCouplingPoint>, ResolvedCouplingError> {
        if frequencies_hz.is_empty() {
            return Ok(Vec::new());
        }
        let state_dofs = linear
            .structural_mesh
            .num_nodes()
            .saturating_mul(2)
            .saturating_add(linear.acoustic_mesh.num_nodes())
            .saturating_add(acoustic.mesh.num_nodes());
        let estimated_bytes_per_lane = state_dofs.saturating_mul(16).saturating_mul(96);
        let lane_count =
            crate::parallelism::resolved_lane_count(frequencies_hz.len(), estimated_bytes_per_lane);
        let lanes = crate::parallelism::contiguous_lane_ranges(frequencies_hz.len(), lane_count);
        let lane_points = lanes
            .into_par_iter()
            .map(|range| {
                let mut coupler = Self::new(linear, acoustic, config)?;
                coupler.solve_sweep(&frequencies_hz[range], voltage_rms_v)
            })
            .collect::<Result<Vec<_>, ResolvedCouplingError>>()?;
        Ok(lane_points.into_iter().flatten().collect())
    }

    /// Run the condensed-load fixed point at one frequency. `initial_delta_z`
    /// seeds the fixed point (sweep warm start); the converged `ΔZ` is
    /// remembered for the next call.
    pub fn solve_frequency(
        &mut self,
        frequency_hz: f64,
        voltage_rms_v: Complex64,
    ) -> Result<ResolvedCouplingPoint, ResolvedCouplingError> {
        // Exact per-frequency condensation: the resolved model is linear in
        // the diaphragm drive, so one unit-velocity solve fixes Z_resolved.
        let unit_solution = self
            .acoustic_continuation
            .solve_frequency_with_drive(frequency_hz, Complex64::new(1.0, 0.0))?;
        let z_resolved = self.acoustic.piston_load_impedance(&unit_solution);
        // The resolved load is the quantity injected back into the stage-3
        // system. Reject non-finite or materially active loads before they
        // can poison the fixed-point iteration. The correction ΔZ itself may
        // have either sign in its real part; passivity is enforced on the
        // resulting total load below.
        validate_resolved_load(frequency_hz, z_resolved)?;

        let mut delta_z = self.previous_delta_z;
        let mut solution = self.linear.solve_with_continuation_and_load(
            frequency_hz,
            voltage_rms_v,
            &mut self.linear_continuation,
            delta_z,
        )?;
        let mut velocity = self.linear.area_averaged_diaphragm_velocity(&solution)?;
        let mut history = Vec::new();
        let mut iterations = 0;
        loop {
            let force = self.linear.compact_fluid_interface_force_n(&solution);
            if !velocity.re.is_finite()
                || !velocity.im.is_finite()
                || velocity.norm() <= f64::MIN_POSITIVE
            {
                return Err(ResolvedCouplingError::InvalidLoadState(
                    "compact-load extraction produced a zero or non-finite diaphragm velocity",
                ));
            }
            let z_compact = -force / velocity;
            if !z_compact.re.is_finite() || !z_compact.im.is_finite() {
                return Err(ResolvedCouplingError::InvalidLoadState(
                    "compact-load extraction produced a non-finite impedance",
                ));
            }
            let updated =
                delta_z + self.config.under_relaxation * (z_resolved - z_compact - delta_z);
            if !updated.re.is_finite() || !updated.im.is_finite() {
                return Err(ResolvedCouplingError::InvalidLoadState(
                    "injected impedance correction is non-finite",
                ));
            }
            let new_solution = self.linear.solve_with_continuation_and_load(
                frequency_hz,
                voltage_rms_v,
                &mut self.linear_continuation,
                updated,
            )?;
            let new_velocity = self
                .linear
                .area_averaged_diaphragm_velocity(&new_solution)?;
            let change =
                (new_velocity - velocity).norm() / new_velocity.norm().max(f64::MIN_POSITIVE);
            history.push(change);
            iterations += 1;
            delta_z = updated;
            solution = new_solution;
            velocity = new_velocity;
            if change < self.config.velocity_tolerance || iterations >= self.config.max_iterations {
                break;
            }
        }
        let converged = *history.last().unwrap_or(&f64::INFINITY) < self.config.velocity_tolerance;

        let maximum_contraction_ratio = history
            .windows(2)
            .map(|pair| {
                if pair[0] > f64::MIN_POSITIVE {
                    pair[1] / pair[0]
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max);
        if !maximum_contraction_ratio.is_finite()
            || maximum_contraction_ratio > self.config.max_contraction_ratio
        {
            return Err(ResolvedCouplingError::NonContractive {
                frequency_hz,
                maximum_contraction_ratio,
            });
        }

        // Independent direct reference: re-solve the full-order FEM block
        // with the converged resolved-load correction. The compact fluid block
        // is already part of this monolithic system, so the injected quantity
        // is ΔZ rather than Z_resolved (otherwise the compact load would be
        // counted twice). Keeping a separate continuation cache prevents the
        // partitioned history from masking a linear-solve disagreement.
        let reference_solution = self.linear.solve_with_continuation_and_load(
            frequency_hz,
            voltage_rms_v,
            &mut self.reference_continuation,
            delta_z,
        )?;
        let monolithic_reference_velocity = self
            .linear
            .area_averaged_diaphragm_velocity(&reference_solution)?;
        let monolithic_agreement_relative_error = (velocity - monolithic_reference_velocity).norm()
            / monolithic_reference_velocity.norm().max(f64::MIN_POSITIVE);
        if !monolithic_agreement_relative_error.is_finite()
            || monolithic_agreement_relative_error > self.config.monolithic_agreement_tolerance
        {
            return Err(ResolvedCouplingError::MonolithicDisagreement {
                frequency_hz,
                relative_velocity_error: monolithic_agreement_relative_error,
            });
        }

        // Resolved fields at the converged drive (the unit solve above only
        // serves the condensation; power quantities are quadratic in v̄).
        let acoustic_solution = self
            .acoustic_continuation
            .solve_frequency_with_drive(frequency_hz, velocity)?;
        let z_compact = -self.linear.compact_fluid_interface_force_n(&solution) / velocity;
        let total_load = z_compact + delta_z;
        validate_resolved_load(frequency_hz, total_load)?;
        let point = ResolvedCouplingPoint {
            frequency_hz,
            z_resolved_n_s_per_m: z_resolved,
            z_compact_n_s_per_m: z_compact,
            delta_z_n_s_per_m: delta_z,
            iterations,
            relative_velocity_change: *history.last().unwrap_or(&f64::INFINITY),
            velocity_change_history: history,
            maximum_contraction_ratio,
            converged,
            monolithic_reference_velocity_rms_m_per_s: monolithic_reference_velocity,
            monolithic_agreement_relative_error,
            diaphragm_velocity_rms_m_per_s: velocity,
            linear_solution: solution,
            acoustic_solution,
        };
        let point = point.require_converged()?;
        self.previous_delta_z = delta_z;
        Ok(point)
    }

    /// Coupled sweep in ascending frequency order, warm-starting each fixed
    /// point from the previous frequency's converged `ΔZ`.
    pub fn solve_sweep(
        &mut self,
        frequencies_hz: &[f64],
        voltage_rms_v: Complex64,
    ) -> Result<Vec<ResolvedCouplingPoint>, ResolvedCouplingError> {
        self.solve_sweep_with_progress(frequencies_hz, voltage_rms_v, |_, _| {})
    }

    /// Solve an ordered coupled sweep and report each completed fixed point.
    pub fn solve_sweep_with_progress<F>(
        &mut self,
        frequencies_hz: &[f64],
        voltage_rms_v: Complex64,
        mut on_frequency_complete: F,
    ) -> Result<Vec<ResolvedCouplingPoint>, ResolvedCouplingError>
    where
        F: FnMut(usize, f64),
    {
        let mut points = Vec::with_capacity(frequencies_hz.len());
        for (index, &frequency_hz) in frequencies_hz.iter().enumerate() {
            let point = self.solve_frequency(frequency_hz, voltage_rms_v)?;
            on_frequency_complete(index + 1, frequency_hz);
            points.push(point);
        }
        Ok(points)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResolvedCouplingError {
    #[error("invalid resolved coupling configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("resolved load coupling requires install_interface_load on the linear model")]
    MissingInterfaceLoad,
    #[error(
        "resolved acoustic load is non-finite or active at {frequency_hz} Hz: {real}+i{imaginary}"
    )]
    NonPassiveResolvedLoad {
        frequency_hz: f64,
        real: f64,
        imaginary: f64,
    },
    #[error("invalid resolved coupling load state: {0}")]
    InvalidLoadState(&'static str),
    #[error(
        "resolved acoustic coupling did not converge at {frequency_hz} Hz after {iterations} iterations (relative velocity change {relative_velocity_change:.3e})"
    )]
    NotConverged {
        frequency_hz: f64,
        iterations: usize,
        relative_velocity_change: f64,
    },
    #[error(
        "resolved acoustic coupling is non-contractive at {frequency_hz} Hz (maximum update ratio {maximum_contraction_ratio:.3e})"
    )]
    NonContractive {
        frequency_hz: f64,
        maximum_contraction_ratio: f64,
    },
    #[error(
        "partitioned coupling disagrees with the monolithic reference at {frequency_hz} Hz (relative velocity error {relative_velocity_error:.3e})"
    )]
    MonolithicDisagreement {
        frequency_hz: f64,
        relative_velocity_error: f64,
    },
    #[error(transparent)]
    Linear(#[from] AxisymmetricLinearError),
    #[error(transparent)]
    Acoustic(#[from] ResolvedSpeakerAcousticError),
    #[error(transparent)]
    Iterative(#[from] CoupledIterativeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axisymmetric_linear::AxisymmetricLinearConfig;
    use crate::resolved_acoustics::ResolvedAcousticConfig;
    use sonium_fem::physics::acoustics::electroacoustics::{
        CoupledIterativeConfig, CoupledPreconditioner,
    };

    fn coarse_linear_model() -> AxisymmetricLinearModel {
        let mut model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 4,
            iterative_solver: CoupledIterativeConfig {
                preconditioner: CoupledPreconditioner::Monolithic,
                ..CoupledIterativeConfig::default()
            },
            ..AxisymmetricLinearConfig::default()
        })
        .unwrap();
        model.install_interface_load().unwrap();
        model
    }

    fn coarse_acoustic_model() -> ResolvedAcousticModel {
        ResolvedAcousticModel::build(ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            ..ResolvedAcousticConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn interface_load_installation_is_normalized_and_area_consistent() {
        let model = coarse_linear_model();
        let load = model.coupled_system.interface_load.as_ref().unwrap();
        let sum: f64 = load.values.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1.0e-12,
            "normalized load distribution sums to {sum}"
        );
        let area = model.interface_area_m2().unwrap();
        let expected = std::f64::consts::PI * model.config.diaphragm_radius_m.powi(2);
        assert!(
            (area - expected) / expected < 0.05,
            "interface area {area} vs piston area {expected}"
        );
    }

    #[test]
    fn condensed_resolved_load_is_passive_and_closes_power_balance() {
        let model = coarse_acoustic_model();
        let mut continuation = ResolvedFrequencyContinuation::new(&model);
        for frequency_hz in [100.0, 250.0, 500.0] {
            let solution = continuation
                .solve_frequency_with_drive(frequency_hz, Complex64::new(0.01, 0.0))
                .unwrap();
            let z = model.piston_load_impedance(&solution);
            let absorbed = (z * solution.diaphragm_velocity_rms_m_per_s.norm_sqr()).re;
            assert!(
                z.re >= 0.0,
                "resolved load is not passive at {frequency_hz} Hz: {z}"
            );
            let scale = solution.driver_input_power_w.abs().max(1.0e-12);
            assert!(
                (absorbed - solution.driver_input_power_w).abs() / scale < 1.0e-9,
                "Z·|v̄|² = {absorbed} vs driver input {} at {frequency_hz} Hz",
                solution.driver_input_power_w
            );
        }
    }

    #[test]
    fn fixed_point_converges_fast_and_is_seed_invariant() {
        let linear = coarse_linear_model();
        let acoustic = coarse_acoustic_model();
        let voltage = Complex64::new(2.83, 0.0);
        let frequency_hz = 100.0;

        let mut coupler =
            ResolvedLoadCoupler::new(&linear, &acoustic, ResolvedCouplingConfig::default())
                .unwrap();
        let point = coupler.solve_frequency(frequency_hz, voltage).unwrap();
        assert!(
            point.converged,
            "history: {:?}",
            point.velocity_change_history
        );
        assert!(point.iterations <= ResolvedCouplingConfig::default().max_iterations);
        assert!(point.maximum_contraction_ratio < 1.0);
        assert!(
            point.monolithic_agreement_relative_error
                <= ResolvedCouplingConfig::default().monolithic_agreement_tolerance
        );
        // The condensed load is exact and the map is affine: expect two to
        // three updates, each decaying the velocity change sharply.
        assert!(
            point.iterations <= 3,
            "fixed point took {} iterations: {:?}",
            point.iterations,
            point.velocity_change_history
        );
        assert!(point.acoustic_load_absorbed_power_w() >= 0.0);
        // At the fixed point the diaphragm sees exactly the resolved load.
        let defect = (point.total_acoustic_load_n_s_per_m() - point.z_resolved_n_s_per_m).norm()
            / point.z_resolved_n_s_per_m.norm().max(f64::MIN_POSITIVE);
        assert!(defect < 1.0e-6, "total/resolved load defect = {defect}");

        // Seed invariance: a deliberately wrong seed converges to the same
        // piston velocity (the coupled system is linear in v̄).
        let mut seeded =
            ResolvedLoadCoupler::new(&linear, &acoustic, ResolvedCouplingConfig::default())
                .unwrap();
        seeded.previous_delta_z = 4.0 * point.z_resolved_n_s_per_m;
        let reseeded = seeded.solve_frequency(frequency_hz, voltage).unwrap();
        assert!(reseeded.converged);
        let velocity_defect =
            (reseeded.diaphragm_velocity_rms_m_per_s - point.diaphragm_velocity_rms_m_per_s).norm()
                / point.diaphragm_velocity_rms_m_per_s.norm();
        assert!(
            velocity_defect < 1.0e-6,
            "seed-invariance velocity defect = {velocity_defect}"
        );
    }

    #[test]
    fn coupling_config_rejects_noncontractive_policy() {
        let config = ResolvedCouplingConfig {
            max_contraction_ratio: 1.0,
            ..ResolvedCouplingConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ResolvedCouplingError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn sealed_box_feedback_matches_lumped_compliance_prediction() {
        // Sealed-box limit: `seal_enclosure_walls` closes the rear cavity and
        // the annular port is narrowed to a ~10 µm slit whose acoustic
        // inertance effectively blocks the vent, leaving a sealed cavity of
        // the configured volume behind the diaphragm. The lumped reference
        // therefore uses a vented enclosure whose port has the slit's
        // equivalent area and the baffle thickness as its length, so both
        // sides model the same compliance || slit-inertance admittance.
        let linear = coarse_linear_model();
        let acoustic_config = ResolvedAcousticConfig {
            diaphragm_radius_m: linear.config.diaphragm_radius_m,
            port_inner_radius_m: 79.99e-3,
            port_outer_radius_m: 80.0e-3 - 1.0e-9,
            seal_enclosure_walls: true,
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            ..ResolvedAcousticConfig::default()
        };
        let acoustic = ResolvedAcousticModel::build(acoustic_config.clone()).unwrap();
        let mut coupler =
            ResolvedLoadCoupler::new(&linear, &acoustic, ResolvedCouplingConfig::default())
                .unwrap();
        let voltage = Complex64::new(2.83, 0.0);

        let medium = crate::AcousticMedium {
            density_kg_per_m3: acoustic_config.fluid.density,
            sound_speed_m_per_s: acoustic_config.fluid.speed_of_sound,
            ..crate::AcousticMedium::standard_air()
        };
        let slit_width_m =
            acoustic_config.port_outer_radius_m - acoustic_config.port_inner_radius_m;
        let slit_area_m2 = 2.0
            * std::f64::consts::PI
            * 0.5
            * (acoustic_config.port_inner_radius_m + acoustic_config.port_outer_radius_m)
            * slit_width_m;
        let slit = crate::PortSpec {
            radius_m: (slit_area_m2 / std::f64::consts::PI).sqrt(),
            length_m: 2.0 * acoustic_config.baffle_half_thickness_m,
            end_correction_m: 0.0,
            resistance_pa_s_per_m3: 0.0,
        };
        let sealed = crate::EnclosureSpec::Vented {
            volume_m3: acoustic_config.cavity_volume_m3(),
            port: slit,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let piston_area = linear.interface_area_m2().unwrap();

        for frequency_hz in [40.0, 63.0, 100.0] {
            let point = coupler.solve_frequency(frequency_hz, voltage).unwrap();
            assert!(
                point.converged,
                "{frequency_hz} Hz: {:?}",
                point.velocity_change_history
            );

            // Lumped reference for the condensed piston load, in the same
            // opposing-force convention: rear compliance || slit inertance
            // plus the front baffled-piston radiation load. This mechanical
            // load is the well-conditioned observable of the condensation;
            // an electrical-impedance extraction divides by the motional
            // impedance, which is ~1e-4 of the blocked impedance here.
            let z_lumped = sealed.acoustic_impedance(frequency_hz, medium).unwrap()
                * piston_area
                * piston_area
                + crate::baffled_piston_mechanical_radiation_impedance(
                    linear.config.diaphragm_radius_m,
                    frequency_hz,
                    medium,
                )
                .unwrap();
            let z_resolved = point.z_resolved_n_s_per_m;
            let error = (z_resolved - z_lumped).norm() / z_lumped.norm();
            // P1 discretization with one cell across the slit and the cavity,
            // plus an end-correction-free lumped slit length: measured error
            // is 0.03-0.7% across this sweep, so 2% leaves 3x headroom while
            // still an order of magnitude tighter than the 20% bound of the
            // distributed/lumped reduction test in axisymmetric_linear.rs.
            assert!(
                error < 0.02,
                "{frequency_hz} Hz: resolved load {z_resolved} vs lumped {z_lumped} (rel err {error:.3})"
            );
            assert!(
                z_resolved.re >= 0.0,
                "{frequency_hz} Hz: condensed load is not passive: {z_resolved}"
            );
        }
    }
}
