//! Reduced transient and fatigue reference studies.
//!
//! The transient family is a single linear degree of freedom integrated with
//! Newmark's average-acceleration method. The fatigue family is a stress-life
//! Miner sum over supplied cycle blocks. Both are intentionally small enough
//! to validate against identities without claiming general nonlinear FEM,
//! multiaxial fatigue, or fracture capability.

use super::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};

/// Linear single-degree-of-freedom transient structural model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSdoFStudy {
    pub mass_kg: f64,
    pub stiffness_n_per_m: f64,
    pub damping_ns_per_m: f64,
    pub time_step_s: f64,
    pub force_history_n: Vec<f64>,
    pub initial_displacement_m: f64,
    pub initial_velocity_m_per_s: f64,
}

impl TransientSdoFStudy {
    pub fn introductory() -> Self {
        Self {
            mass_kg: 1.0,
            stiffness_n_per_m: 100.0,
            damping_ns_per_m: 1.0,
            time_step_s: 0.01,
            force_history_n: vec![0.0; 501],
            initial_displacement_m: 0.0,
            initial_velocity_m_per_s: 0.0,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        let positive = [
            ("transient_sdof.mass_kg", self.mass_kg),
            ("transient_sdof.stiffness_n_per_m", self.stiffness_n_per_m),
            ("transient_sdof.time_step_s", self.time_step_s),
        ];
        for (field, value) in positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.damping_ns_per_m.is_finite() || self.damping_ns_per_m < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_sdof.damping_ns_per_m",
                value: self.damping_ns_per_m.to_string(),
            });
        }
        if self.force_history_n.len() < 2
            || self.force_history_n.iter().any(|value| !value.is_finite())
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_sdof.force_history_n",
                value: format!("length={}", self.force_history_n.len()),
            });
        }
        if !self.initial_displacement_m.is_finite() || !self.initial_velocity_m_per_s.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_sdof.initial_state",
                value: format!(
                    "u0={}, v0={}",
                    self.initial_displacement_m, self.initial_velocity_m_per_s
                ),
            });
        }
        Ok(())
    }
}

/// A single Newmark time level with balance and energy evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientStepResult {
    pub step: usize,
    pub time_s: f64,
    pub force_n: f64,
    pub displacement_m: f64,
    pub velocity_m_per_s: f64,
    pub acceleration_m_per_s2: f64,
    pub equilibrium_residual_n: f64,
    pub mechanical_energy_j: f64,
    pub cumulative_input_work_j: f64,
    pub cumulative_damping_dissipation_j: f64,
    pub energy_balance_residual_j: f64,
}

/// Result of a reduced transient execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSdoFResult {
    pub study: TransientSdoFStudy,
    pub steps: Vec<TransientStepResult>,
    pub final_displacement_m: f64,
    pub final_velocity_m_per_s: f64,
    pub max_equilibrium_residual_n: f64,
    pub max_energy_balance_residual_j: f64,
    pub total_input_work_j: f64,
    pub total_damping_dissipation_j: f64,
    pub model_assumption: StructuralModelAssumption,
}

/// Integrate the SDOF model with Newmark `beta=1/4`, `gamma=1/2`.
pub fn run_transient_sdof(
    study: &TransientSdoFStudy,
) -> Result<TransientSdoFResult, StructuralMechanicsError> {
    study.validate()?;
    let beta = 0.25;
    let gamma = 0.5;
    let dt = study.time_step_s;
    let a0 = 1.0 / (beta * dt * dt);
    let a1 = gamma / (beta * dt);
    let a2 = 1.0 / (beta * dt);
    let a3 = 1.0 / (2.0 * beta) - 1.0;
    let a4 = gamma / beta - 1.0;
    let a5 = dt * (gamma / (2.0 * beta) - 1.0);
    let effective_stiffness =
        study.stiffness_n_per_m + a1 * study.damping_ns_per_m + a0 * study.mass_kg;
    if !effective_stiffness.is_finite() || effective_stiffness <= 0.0 {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "transient effective stiffness",
            message: effective_stiffness.to_string(),
        });
    }

    let initial_force_n = study.force_history_n[0];
    let mut displacement_m = study.initial_displacement_m;
    let mut velocity_m_per_s = study.initial_velocity_m_per_s;
    let mut acceleration_m_per_s2 = (initial_force_n
        - study.damping_ns_per_m * velocity_m_per_s
        - study.stiffness_n_per_m * displacement_m)
        / study.mass_kg;
    let initial_energy_j = 0.5 * study.mass_kg * velocity_m_per_s * velocity_m_per_s
        + 0.5 * study.stiffness_n_per_m * displacement_m * displacement_m;
    let mut cumulative_input_work_j = 0.0;
    let mut cumulative_damping_dissipation_j = 0.0;
    let mut max_equilibrium_residual_n: f64 = 0.0;
    let mut max_energy_balance_residual_j: f64 = 0.0;
    let mut steps = Vec::with_capacity(study.force_history_n.len());
    steps.push(TransientStepResult {
        step: 0,
        time_s: 0.0,
        force_n: initial_force_n,
        displacement_m,
        velocity_m_per_s,
        acceleration_m_per_s2,
        equilibrium_residual_n: 0.0,
        mechanical_energy_j: initial_energy_j,
        cumulative_input_work_j,
        cumulative_damping_dissipation_j,
        energy_balance_residual_j: 0.0,
    });

    for step in 1..study.force_history_n.len() {
        let force_n = study.force_history_n[step];
        let previous_displacement_m = displacement_m;
        let previous_velocity_m_per_s = velocity_m_per_s;
        let previous_acceleration_m_per_s2 = acceleration_m_per_s2;
        let effective_force_n = force_n
            + study.mass_kg
                * (a0 * previous_displacement_m
                    + a2 * previous_velocity_m_per_s
                    + a3 * previous_acceleration_m_per_s2)
            + study.damping_ns_per_m
                * (a1 * previous_displacement_m
                    + a4 * previous_velocity_m_per_s
                    + a5 * previous_acceleration_m_per_s2);
        displacement_m = effective_force_n / effective_stiffness;
        acceleration_m_per_s2 = a0 * (displacement_m - previous_displacement_m)
            - a2 * previous_velocity_m_per_s
            - a3 * previous_acceleration_m_per_s2;
        velocity_m_per_s = previous_velocity_m_per_s
            + dt * ((1.0 - gamma) * previous_acceleration_m_per_s2 + gamma * acceleration_m_per_s2);

        let equilibrium_residual_n = (study.mass_kg * acceleration_m_per_s2
            + study.damping_ns_per_m * velocity_m_per_s
            + study.stiffness_n_per_m * displacement_m
            - force_n)
            .abs();
        cumulative_input_work_j += 0.5
            * (study.force_history_n[step - 1] + force_n)
            * (displacement_m - previous_displacement_m);
        cumulative_damping_dissipation_j += 0.5
            * study.damping_ns_per_m
            * (previous_velocity_m_per_s * previous_velocity_m_per_s
                + velocity_m_per_s * velocity_m_per_s)
            * dt;
        let mechanical_energy_j = 0.5 * study.mass_kg * velocity_m_per_s * velocity_m_per_s
            + 0.5 * study.stiffness_n_per_m * displacement_m * displacement_m;
        let energy_balance_residual_j = mechanical_energy_j - initial_energy_j
            + cumulative_damping_dissipation_j
            - cumulative_input_work_j;
        max_equilibrium_residual_n = max_equilibrium_residual_n.max(equilibrium_residual_n);
        max_energy_balance_residual_j =
            max_energy_balance_residual_j.max(energy_balance_residual_j.abs());
        steps.push(TransientStepResult {
            step,
            time_s: step as f64 * dt,
            force_n,
            displacement_m,
            velocity_m_per_s,
            acceleration_m_per_s2,
            equilibrium_residual_n,
            mechanical_energy_j,
            cumulative_input_work_j,
            cumulative_damping_dissipation_j,
            energy_balance_residual_j,
        });
    }

    let force_scale = study
        .force_history_n
        .iter()
        .map(|force| force.abs())
        .fold(1.0, f64::max);
    if max_equilibrium_residual_n > 1.0e-8 * force_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "transient_equilibrium",
            value: max_equilibrium_residual_n,
        });
    }
    Ok(TransientSdoFResult {
        study: study.clone(),
        steps,
        final_displacement_m: displacement_m,
        final_velocity_m_per_s: velocity_m_per_s,
        max_equilibrium_residual_n,
        max_energy_balance_residual_j,
        total_input_work_j: cumulative_input_work_j,
        total_damping_dissipation_j: cumulative_damping_dissipation_j,
        model_assumption: StructuralModelAssumption::TransientSingleDof,
    })
}

/// Stress-life fatigue input represented as independent constant-amplitude
/// cycle blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FatigueDamageStudy {
    pub reference_stress_pa: f64,
    pub reference_cycles: f64,
    pub sn_slope: f64,
    pub stress_amplitudes_pa: Vec<f64>,
    pub cycle_blocks: Vec<f64>,
    pub damage_limit: f64,
}

impl FatigueDamageStudy {
    pub fn introductory() -> Self {
        Self {
            reference_stress_pa: 100.0e6,
            reference_cycles: 1.0e6,
            sn_slope: 3.0,
            stress_amplitudes_pa: vec![50.0e6, 100.0e6, 125.0e6],
            cycle_blocks: vec![100_000.0, 50_000.0, 10_000.0],
            damage_limit: 1.0,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("fatigue.reference_stress_pa", self.reference_stress_pa),
            ("fatigue.reference_cycles", self.reference_cycles),
            ("fatigue.sn_slope", self.sn_slope),
            ("fatigue.damage_limit", self.damage_limit),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.stress_amplitudes_pa.len() != self.cycle_blocks.len()
            || self.stress_amplitudes_pa.is_empty()
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fatigue.block_lengths",
                value: format!(
                    "stress_amplitudes={}, cycle_blocks={}",
                    self.stress_amplitudes_pa.len(),
                    self.cycle_blocks.len()
                ),
            });
        }
        if self
            .stress_amplitudes_pa
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .cycle_blocks
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fatigue.blocks",
                value: "negative or non-finite block".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FatigueBlockResult {
    pub stress_amplitude_pa: f64,
    pub applied_cycles: f64,
    /// Finite S-N allowable cycles. `None` denotes the explicit zero-stress,
    /// unbounded-life case rather than serializing `f64::INFINITY`.
    pub allowable_cycles: Option<f64>,
    pub damage_increment: f64,
    pub cumulative_damage: f64,
    pub cumulative_cycles: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FatigueDamageResult {
    pub study: FatigueDamageStudy,
    pub blocks: Vec<FatigueBlockResult>,
    pub total_damage: f64,
    pub failed: bool,
    pub failure_cycle: Option<f64>,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Evaluate Basquin stress-life blocks and Miner cumulative damage.
pub fn run_fatigue_damage(
    study: &FatigueDamageStudy,
) -> Result<FatigueDamageResult, StructuralMechanicsError> {
    study.validate()?;
    let mut blocks = Vec::with_capacity(study.stress_amplitudes_pa.len());
    let mut cumulative_damage = 0.0;
    let mut cumulative_cycles = 0.0;
    let mut failure_cycle = None;
    for (&stress_amplitude_pa, &applied_cycles) in
        study.stress_amplitudes_pa.iter().zip(&study.cycle_blocks)
    {
        let allowable_cycles = if stress_amplitude_pa == 0.0 {
            None
        } else {
            let allowable_cycles = study.reference_cycles
                * (study.reference_stress_pa / stress_amplitude_pa).powf(study.sn_slope);
            if !allowable_cycles.is_finite() || allowable_cycles <= 0.0 {
                return Err(StructuralMechanicsError::SolverFailure {
                    stage: "fatigue S-N allowable cycles",
                    message: format!(
                        "non-finite or non-positive allowable cycles for stress amplitude {stress_amplitude_pa}"
                    ),
                });
            }
            Some(allowable_cycles)
        };
        let damage_increment = allowable_cycles.map_or(0.0, |cycles| applied_cycles / cycles);
        if !damage_increment.is_finite() {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "fatigue Miner damage",
                message: format!(
                    "non-finite damage increment for stress amplitude {stress_amplitude_pa}"
                ),
            });
        }
        let previous_damage = cumulative_damage;
        cumulative_damage += damage_increment;
        if !cumulative_damage.is_finite() {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "fatigue Miner damage",
                message: "cumulative damage overflowed".to_string(),
            });
        }
        cumulative_cycles += applied_cycles;
        if failure_cycle.is_none()
            && previous_damage < study.damage_limit
            && cumulative_damage >= study.damage_limit
            && damage_increment > 0.0
        {
            failure_cycle = Some(
                cumulative_cycles
                    - applied_cycles * (cumulative_damage - study.damage_limit) / damage_increment,
            );
        }
        blocks.push(FatigueBlockResult {
            stress_amplitude_pa,
            applied_cycles,
            allowable_cycles,
            damage_increment,
            cumulative_damage,
            cumulative_cycles,
        });
    }
    Ok(FatigueDamageResult {
        study: study.clone(),
        blocks,
        total_damage: cumulative_damage,
        failed: cumulative_damage >= study.damage_limit,
        failure_cycle,
        model_assumption: StructuralModelAssumption::MinerFatigue,
        execution: StructuralExecutionProvenance::cpu_fatigue_reference(),
    })
}

/// Base-acceleration PSD input for a linear single-degree-of-freedom random
/// vibration reference calculation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomVibrationStudy {
    pub mass_kg: f64,
    pub stiffness_n_per_m: f64,
    pub damping_ns_per_m: f64,
    pub frequencies_hz: Vec<f64>,
    pub acceleration_psd_m2_per_s3: Vec<f64>,
    pub peak_factor: f64,
    pub allowable_displacement_m: f64,
}

impl RandomVibrationStudy {
    pub fn introductory() -> Self {
        let frequencies_hz = (0..=100).map(|index| 0.5 + index as f64 * 0.5).collect();
        Self {
            mass_kg: 1.0,
            stiffness_n_per_m: 100.0,
            damping_ns_per_m: 1.0,
            frequencies_hz,
            acceleration_psd_m2_per_s3: vec![1.0e-4; 101],
            peak_factor: 3.0,
            allowable_displacement_m: 0.5,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("random_vibration.mass_kg", self.mass_kg),
            ("random_vibration.stiffness_n_per_m", self.stiffness_n_per_m),
            ("random_vibration.peak_factor", self.peak_factor),
            (
                "random_vibration.allowable_displacement_m",
                self.allowable_displacement_m,
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
                field: "random_vibration.damping_ns_per_m",
                value: self.damping_ns_per_m.to_string(),
            });
        }
        if self.frequencies_hz.len() < 2
            || self.frequencies_hz.len() != self.acceleration_psd_m2_per_s3.len()
            || self
                .frequencies_hz
                .windows(2)
                .any(|window| !window[0].is_finite() || window[1] <= window[0])
            || self
                .frequencies_hz
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .acceleration_psd_m2_per_s3
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "random_vibration.frequency_psd_grid",
                value: "non-monotone, mismatched, or negative grid".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomVibrationResult {
    pub study: RandomVibrationStudy,
    pub displacement_psd_m2_per_hz: Vec<f64>,
    pub displacement_variance_m2: f64,
    pub rms_displacement_m: f64,
    pub peak_displacement_m: f64,
    pub maximum_dynamic_amplification: f64,
    pub feasible: bool,
    pub model_assumption: StructuralModelAssumption,
}

/// Integrate a one-sided base-acceleration PSD through the exact SDOF transfer
/// function using the trapezoidal rule.
pub fn run_random_vibration(
    study: &RandomVibrationStudy,
) -> Result<RandomVibrationResult, StructuralMechanicsError> {
    study.validate()?;
    let mut displacement_psd_m2_per_hz = Vec::with_capacity(study.frequencies_hz.len());
    let mut maximum_dynamic_amplification: f64 = 0.0;
    for (&frequency_hz, &acceleration_psd) in study
        .frequencies_hz
        .iter()
        .zip(&study.acceleration_psd_m2_per_s3)
    {
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let real = study.stiffness_n_per_m - study.mass_kg * omega * omega;
        let imaginary = study.damping_ns_per_m * omega;
        let denominator = real * real + imaginary * imaginary;
        let transfer_squared = if denominator == 0.0 {
            0.0
        } else {
            study.mass_kg * study.mass_kg / denominator
        };
        maximum_dynamic_amplification = maximum_dynamic_amplification.max(transfer_squared.sqrt());
        displacement_psd_m2_per_hz.push(acceleration_psd * transfer_squared);
    }
    let displacement_variance_m2 =
        trapezoid_integral(&study.frequencies_hz, &displacement_psd_m2_per_hz);
    let rms_displacement_m = displacement_variance_m2.max(0.0).sqrt();
    let peak_displacement_m = study.peak_factor * rms_displacement_m;
    Ok(RandomVibrationResult {
        study: study.clone(),
        displacement_psd_m2_per_hz,
        displacement_variance_m2,
        rms_displacement_m,
        peak_displacement_m,
        maximum_dynamic_amplification,
        feasible: peak_displacement_m <= study.allowable_displacement_m,
        model_assumption: StructuralModelAssumption::RandomVibrationSingleDof,
    })
}

/// Deterministic paired load/resistance samples for empirical reliability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityStudy {
    pub load_samples_n: Vec<f64>,
    pub resistance_samples_n: Vec<f64>,
    pub allowable_failure_probability: f64,
}

impl ReliabilityStudy {
    pub fn introductory() -> Self {
        Self {
            load_samples_n: vec![80.0, 90.0, 100.0, 110.0, 120.0],
            resistance_samples_n: vec![140.0, 130.0, 125.0, 115.0, 100.0],
            allowable_failure_probability: 0.20,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.load_samples_n.is_empty()
            || self.load_samples_n.len() != self.resistance_samples_n.len()
            || !self.allowable_failure_probability.is_finite()
            || self.allowable_failure_probability < 0.0
            || self.allowable_failure_probability > 1.0
            || self.load_samples_n.iter().any(|value| !value.is_finite())
            || self
                .resistance_samples_n
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "reliability.samples",
                value: "empty, mismatched, non-finite, or invalid probability".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityStudyResult {
    pub study: ReliabilityStudy,
    pub margins_n: Vec<f64>,
    pub failure_count: usize,
    pub failure_probability: f64,
    pub mean_margin_n: f64,
    pub margin_standard_deviation_n: f64,
    /// Gaussian-equivalent margin index; empirical probability remains the gate.
    pub reliability_index: f64,
    pub feasible: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Evaluate empirical failure probability and a Gaussian-equivalent margin
/// index. This is a screening constraint, not a certified FORM/SORM result.
pub fn run_reliability_study(
    study: &ReliabilityStudy,
) -> Result<ReliabilityStudyResult, StructuralMechanicsError> {
    study.validate()?;
    let margins_n = study
        .resistance_samples_n
        .iter()
        .zip(&study.load_samples_n)
        .map(|(resistance, load)| resistance - load)
        .collect::<Vec<_>>();
    let failure_count = margins_n.iter().filter(|margin| **margin < 0.0).count();
    let failure_probability = failure_count as f64 / margins_n.len() as f64;
    let mean_margin_n = margins_n.iter().sum::<f64>() / margins_n.len() as f64;
    let margin_standard_deviation_n = (margins_n
        .iter()
        .map(|margin| (margin - mean_margin_n).powi(2))
        .sum::<f64>()
        / margins_n.len() as f64)
        .sqrt();
    let reliability_index = if margin_standard_deviation_n > 0.0 {
        mean_margin_n / margin_standard_deviation_n
    } else if mean_margin_n >= 0.0 {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    Ok(ReliabilityStudyResult {
        study: study.clone(),
        margins_n,
        failure_count,
        failure_probability,
        mean_margin_n,
        margin_standard_deviation_n,
        reliability_index,
        feasible: failure_probability <= study.allowable_failure_probability,
        model_assumption: StructuralModelAssumption::EmpiricalReliability,
        execution: StructuralExecutionProvenance::cpu_reliability_reference(),
    })
}

fn trapezoid_integral(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(x_pair, y_pair)| 0.5 * (x_pair[1] - x_pair[0]) * (y_pair[0] + y_pair[1]))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undamped_newmark_free_vibration_preserves_energy() {
        let study = TransientSdoFStudy {
            mass_kg: 2.0,
            stiffness_n_per_m: 50.0,
            damping_ns_per_m: 0.0,
            time_step_s: 0.01,
            force_history_n: vec![0.0; 1_001],
            initial_displacement_m: 0.1,
            initial_velocity_m_per_s: 0.0,
        };
        let result = run_transient_sdof(&study).unwrap();
        let initial_energy_j = 0.5 * study.stiffness_n_per_m * study.initial_displacement_m.powi(2);
        let final_energy_j = result.steps.last().unwrap().mechanical_energy_j;

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::TransientSingleDof
        );
        assert!((final_energy_j - initial_energy_j).abs() < 1.0e-10);
        assert!(result.max_equilibrium_residual_n < 1.0e-9);
        assert!(result.total_damping_dissipation_j.abs() < 1.0e-12);
    }

    #[test]
    fn fatigue_miner_sum_matches_basquin_blocks() {
        let study = FatigueDamageStudy {
            reference_stress_pa: 100.0,
            reference_cycles: 1.0e6,
            sn_slope: 3.0,
            stress_amplitudes_pa: vec![100.0, 200.0, 0.0],
            cycle_blocks: vec![100_000.0, 10_000.0, 1.0e6],
            damage_limit: 1.0,
        };
        let result = run_fatigue_damage(&study).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::MinerFatigue
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.fatigue-evidence.v1"
        );
        assert!((result.total_damage - 0.18).abs() < 1.0e-12);
        assert!(!result.failed);
        assert!(result.failure_cycle.is_none());
        assert_eq!(result.blocks[2].allowable_cycles, None);
        assert_eq!(result.blocks[2].damage_increment, 0.0);
        assert!(serde_json::to_string(&result).is_ok());
    }

    #[test]
    fn fatigue_rejects_nonfinite_sn_evidence_before_serialization() {
        let study = FatigueDamageStudy {
            reference_stress_pa: 1.0,
            reference_cycles: 1.0,
            sn_slope: 300.0,
            stress_amplitudes_pa: vec![1.0e-300],
            cycle_blocks: vec![1.0],
            damage_limit: 1.0,
        };
        assert!(matches!(
            run_fatigue_damage(&study),
            Err(StructuralMechanicsError::SolverFailure {
                stage: "fatigue S-N allowable cycles",
                ..
            })
        ));
    }

    #[test]
    fn random_vibration_integrates_psd_and_applies_peak_feasibility_gate() {
        let result = run_random_vibration(&RandomVibrationStudy::introductory()).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::RandomVibrationSingleDof
        );
        assert_eq!(result.displacement_psd_m2_per_hz.len(), 101);
        assert!(result.displacement_variance_m2 > 0.0);
        assert!(result.rms_displacement_m > 0.0);
        assert!(result.peak_displacement_m > result.rms_displacement_m);
        assert!(result.maximum_dynamic_amplification > 0.0);
        assert!(result.feasible);
    }

    #[test]
    fn reliability_study_reports_empirical_failure_probability_and_margin_index() {
        let result = run_reliability_study(&ReliabilityStudy::introductory()).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::EmpiricalReliability
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.reliability-evidence.v1"
        );
        assert_eq!(result.failure_count, 1);
        assert!((result.failure_probability - 0.2).abs() < 1.0e-12);
        assert!((result.mean_margin_n - 22.0).abs() < 1.0e-12);
        assert!(result.reliability_index > 0.0);
        assert!(result.feasible);
    }

    #[test]
    fn transient_and_fatigue_inputs_are_checked() {
        let mut transient = TransientSdoFStudy::introductory();
        transient.force_history_n.clear();
        assert!(matches!(
            run_transient_sdof(&transient),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "transient_sdof.force_history_n",
                ..
            })
        ));

        let mut fatigue = FatigueDamageStudy::introductory();
        fatigue.cycle_blocks.pop();
        assert!(matches!(
            run_fatigue_damage(&fatigue),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fatigue.block_lengths",
                ..
            })
        ));

        let mut random = RandomVibrationStudy::introductory();
        random.frequencies_hz[1] = random.frequencies_hz[0];
        assert!(matches!(
            run_random_vibration(&random),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "random_vibration.frequency_psd_grid",
                ..
            })
        ));

        let mut reliability = ReliabilityStudy::introductory();
        reliability.resistance_samples_n.pop();
        assert!(matches!(
            run_reliability_study(&reliability),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "reliability.samples",
                ..
            })
        ));
    }
}
