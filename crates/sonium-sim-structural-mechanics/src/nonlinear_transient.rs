//! Bounded nonlinear transient structural reference.
//!
//! This module integrates a single degree of freedom with an implicit
//! Newmark average-acceleration step and a cubic hardening restoring force.
//! It is a checked nonlinear dynamics contract, not a nonlinear transient FEM
//! or transient topology solver.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError};
use serde::{Deserialize, Serialize};

/// Schema version for the nonlinear transient result.
pub const NONLINEAR_TRANSIENT_SCHEMA_VERSION: u32 = 1;
const MAX_NONLINEAR_TRANSIENT_STEPS: usize = 10_000;

/// Cubic-hardening single-degree-of-freedom transient inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearTransientStudy {
    pub mass_kg: f64,
    pub linear_stiffness_n_per_m: f64,
    pub cubic_stiffness_n_per_m3: f64,
    pub damping_ns_per_m: f64,
    pub time_step_s: f64,
    pub force_history_n: Vec<f64>,
    pub initial_displacement_m: f64,
    pub initial_velocity_m_per_s: f64,
    pub max_newton_iterations: usize,
    pub residual_tolerance_n: f64,
}

impl NonlinearTransientStudy {
    /// Compact pulse that exercises the nonlinear restoring force.
    pub fn introductory() -> Self {
        Self {
            mass_kg: 1.0,
            linear_stiffness_n_per_m: 100.0,
            cubic_stiffness_n_per_m3: 250.0,
            damping_ns_per_m: 0.25,
            time_step_s: 0.01,
            force_history_n: vec![0.0, 2.0, 6.0, 10.0, 6.0, 2.0, 0.0, 0.0],
            initial_displacement_m: 0.0,
            initial_velocity_m_per_s: 0.0,
            max_newton_iterations: 20,
            residual_tolerance_n: 1.0e-10,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("nonlinear_transient.mass_kg", self.mass_kg),
            (
                "nonlinear_transient.linear_stiffness_n_per_m",
                self.linear_stiffness_n_per_m,
            ),
            (
                "nonlinear_transient.cubic_stiffness_n_per_m3",
                self.cubic_stiffness_n_per_m3,
            ),
            ("nonlinear_transient.time_step_s", self.time_step_s),
            (
                "nonlinear_transient.residual_tolerance_n",
                self.residual_tolerance_n,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.damping_ns_per_m.is_finite() || self.damping_ns_per_m < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient.damping_ns_per_m",
                value: self.damping_ns_per_m.to_string(),
            });
        }
        if !self.initial_displacement_m.is_finite() || !self.initial_velocity_m_per_s.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient.initial_state",
                value: format!(
                    "({}, {})",
                    self.initial_displacement_m, self.initial_velocity_m_per_s
                ),
            });
        }
        if self.force_history_n.len() < 2
            || self.force_history_n.len() > MAX_NONLINEAR_TRANSIENT_STEPS
            || self.force_history_n.iter().any(|value| !value.is_finite())
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient.force_history_n",
                value: self.force_history_n.len().to_string(),
            });
        }
        if self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_transient.max_newton_iterations",
                value: self.max_newton_iterations.to_string(),
            });
        }
        Ok(())
    }
}

/// Physical idealisation used by this reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NonlinearTransientModelAssumption {
    CubicHardeningSingleDof,
}

/// One converged nonlinear Newmark sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NonlinearTransientStepResult {
    pub step: usize,
    pub time_s: f64,
    pub force_n: f64,
    pub displacement_m: f64,
    pub velocity_m_per_s: f64,
    pub acceleration_m_per_s2: f64,
    pub equilibrium_residual_n: f64,
    pub linear_potential_energy_j: f64,
    pub nonlinear_potential_energy_j: f64,
    pub total_energy_j: f64,
    pub damping_power_w: f64,
    pub newton_iterations: usize,
}

/// Versioned nonlinear transient result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearTransientResult {
    pub schema_version: u32,
    pub study: NonlinearTransientStudy,
    pub steps: Vec<NonlinearTransientStepResult>,
    pub final_displacement_m: f64,
    pub final_velocity_m_per_s: f64,
    pub max_equilibrium_residual_n: f64,
    pub max_total_energy_j: f64,
    pub total_input_work_j: f64,
    pub total_damping_dissipation_j: f64,
    pub model_assumption: NonlinearTransientModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

fn restoring_force(study: &NonlinearTransientStudy, displacement_m: f64) -> f64 {
    study.linear_stiffness_n_per_m * displacement_m
        + study.cubic_stiffness_n_per_m3 * displacement_m.powi(3)
}

fn potential_energy(study: &NonlinearTransientStudy, displacement_m: f64) -> (f64, f64) {
    (
        0.5 * study.linear_stiffness_n_per_m * displacement_m.powi(2),
        0.25 * study.cubic_stiffness_n_per_m3 * displacement_m.powi(4),
    )
}

/// Run the bounded implicit-Newmark nonlinear transient reference.
pub fn run_nonlinear_transient(
    study: &NonlinearTransientStudy,
) -> Result<NonlinearTransientResult, StructuralMechanicsError> {
    study.validate()?;

    let beta = 0.25;
    let gamma = 0.5;
    let dt = study.time_step_s;
    let acceleration_coefficient = 1.0 / (beta * dt * dt);
    let velocity_acceleration_coefficient = gamma * dt * acceleration_coefficient;
    let initial_force = study.force_history_n[0];
    let mut displacement_m = study.initial_displacement_m;
    let mut velocity_m_per_s = study.initial_velocity_m_per_s;
    let mut acceleration_m_per_s2 = (initial_force
        - study.damping_ns_per_m * velocity_m_per_s
        - restoring_force(study, displacement_m))
        / study.mass_kg;
    let mut input_work_j = 0.0;
    let mut damping_dissipation_j = 0.0;
    let (linear_energy, nonlinear_energy) = potential_energy(study, displacement_m);
    let initial_energy_j =
        0.5 * study.mass_kg * velocity_m_per_s.powi(2) + linear_energy + nonlinear_energy;
    let mut max_equilibrium_residual_n: f64 = 0.0;
    let mut max_total_energy_j: f64 = initial_energy_j;
    let mut steps = Vec::with_capacity(study.force_history_n.len());
    steps.push(NonlinearTransientStepResult {
        step: 0,
        time_s: 0.0,
        force_n: initial_force,
        displacement_m,
        velocity_m_per_s,
        acceleration_m_per_s2,
        equilibrium_residual_n: 0.0,
        linear_potential_energy_j: linear_energy,
        nonlinear_potential_energy_j: nonlinear_energy,
        total_energy_j: initial_energy_j,
        damping_power_w: study.damping_ns_per_m * velocity_m_per_s.powi(2),
        newton_iterations: 0,
    });

    for step in 1..study.force_history_n.len() {
        let previous_displacement = displacement_m;
        let previous_velocity = velocity_m_per_s;
        let previous_acceleration = acceleration_m_per_s2;
        let force_n = study.force_history_n[step];
        let predicted_displacement = previous_displacement
            + dt * previous_velocity
            + dt * dt * (0.5 - beta) * previous_acceleration;
        let mut trial_displacement = predicted_displacement;
        let mut residual_n = f64::INFINITY;
        let mut iterations = 0;
        let mut converged = false;

        for iteration in 1..=study.max_newton_iterations {
            let trial_acceleration =
                acceleration_coefficient * (trial_displacement - predicted_displacement);
            let trial_velocity = previous_velocity
                + dt * ((1.0 - gamma) * previous_acceleration + gamma * trial_acceleration);
            residual_n = (study.mass_kg * trial_acceleration
                + study.damping_ns_per_m * trial_velocity
                + restoring_force(study, trial_displacement)
                - force_n)
                .abs();
            iterations = iteration;
            if residual_n <= study.residual_tolerance_n {
                converged = true;
                break;
            }
            let tangent = study.mass_kg * acceleration_coefficient
                + study.damping_ns_per_m * velocity_acceleration_coefficient
                + study.linear_stiffness_n_per_m
                + 3.0 * study.cubic_stiffness_n_per_m3 * trial_displacement.powi(2);
            if !tangent.is_finite() || tangent <= 0.0 {
                break;
            }
            let signed_residual = study.mass_kg * trial_acceleration
                + study.damping_ns_per_m * trial_velocity
                + restoring_force(study, trial_displacement)
                - force_n;
            trial_displacement -= signed_residual / tangent;
            if !trial_displacement.is_finite() {
                break;
            }
        }
        if !converged {
            return Err(StructuralMechanicsError::NonConvergence {
                iterations,
                residual: residual_n,
            });
        }

        displacement_m = trial_displacement;
        acceleration_m_per_s2 =
            acceleration_coefficient * (displacement_m - predicted_displacement);
        velocity_m_per_s = previous_velocity
            + dt * ((1.0 - gamma) * previous_acceleration + gamma * acceleration_m_per_s2);
        let (linear_energy, nonlinear_energy) = potential_energy(study, displacement_m);
        let total_energy_j =
            0.5 * study.mass_kg * velocity_m_per_s.powi(2) + linear_energy + nonlinear_energy;
        let damping_power_w = study.damping_ns_per_m * velocity_m_per_s.powi(2);
        input_work_j += 0.5
            * (study.force_history_n[step - 1] + force_n)
            * (displacement_m - previous_displacement);
        damping_dissipation_j += 0.5
            * study.damping_ns_per_m
            * (previous_velocity.powi(2) + velocity_m_per_s.powi(2))
            * dt;
        max_equilibrium_residual_n = max_equilibrium_residual_n.max(residual_n);
        max_total_energy_j = max_total_energy_j.max(total_energy_j);
        steps.push(NonlinearTransientStepResult {
            step,
            time_s: step as f64 * dt,
            force_n,
            displacement_m,
            velocity_m_per_s,
            acceleration_m_per_s2,
            equilibrium_residual_n: residual_n,
            linear_potential_energy_j: linear_energy,
            nonlinear_potential_energy_j: nonlinear_energy,
            total_energy_j,
            damping_power_w,
            newton_iterations: iterations,
        });
    }

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.nonlinear-transient-sdof.v1".to_string();
    execution.limitations.extend([
        "Cubic hardening single-degree-of-freedom reference only; no nonlinear transient FEM, contact, topology update, or distributed/device execution is represented.".to_string(),
        "The prescribed force history is a deterministic integration fixture, not calibrated measured loading or external-solver parity.".to_string(),
    ]);

    Ok(NonlinearTransientResult {
        schema_version: NONLINEAR_TRANSIENT_SCHEMA_VERSION,
        study: study.clone(),
        final_displacement_m: displacement_m,
        final_velocity_m_per_s: velocity_m_per_s,
        max_equilibrium_residual_n,
        max_total_energy_j,
        total_input_work_j: input_work_j,
        total_damping_dissipation_j: damping_dissipation_j,
        steps,
        model_assumption: NonlinearTransientModelAssumption::CubicHardeningSingleDof,
        execution,
    })
}
