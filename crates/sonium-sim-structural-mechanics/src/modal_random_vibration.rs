//! Bounded modal spectral random-vibration reference.
//!
//! This module consumes a validated modal result and caller-supplied base
//! acceleration PSD/participation data. It uses independent modal responses
//! and SRSS variance accumulation, so it is an auditable modal screening
//! contract rather than a full spatial PSD matrix, correlated modal, or
//! product-scale random-vibration solver.

use crate::{
    ModalNormalization, Solid3DModalResult, StructuralExecutionProvenance,
    StructuralMechanicsError, StructuralModelAssumption,
};
use serde::{Deserialize, Serialize};

/// Schema version for the modal random-vibration result.
pub const MODAL_RANDOM_VIBRATION_SCHEMA_VERSION: u32 = 1;

/// Caller-supplied spectral input and modal participation controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalRandomVibrationStudy {
    pub name: String,
    pub frequencies_hz: Vec<f64>,
    pub acceleration_psd_m2_per_s3: Vec<f64>,
    pub mode_participation: Vec<f64>,
    pub damping_ratios: Vec<f64>,
    pub peak_factor: f64,
    pub allowable_displacement_m: f64,
}

impl Default for ModalRandomVibrationStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl ModalRandomVibrationStudy {
    /// Small bounded spectral fixture for a caller-supplied modal result.
    pub fn introductory() -> Self {
        let frequencies_hz = (0..=100)
            .map(|index| 0.5 + index as f64 * 0.5)
            .collect::<Vec<_>>();
        Self {
            name: "modal-random-vibration-reference".to_string(),
            acceleration_psd_m2_per_s3: vec![1.0e-4; frequencies_hz.len()],
            frequencies_hz,
            mode_participation: vec![1.0, 0.35, 0.2],
            damping_ratios: vec![0.02, 0.025, 0.03],
            peak_factor: 3.0,
            allowable_displacement_m: 0.05,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.name.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.name",
                value: "empty".to_string(),
            });
        }
        if self.frequencies_hz.len() < 2
            || self.frequencies_hz.len() != self.acceleration_psd_m2_per_s3.len()
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.frequency_psd_grid",
                value: "matching grid with at least two samples required".to_string(),
            });
        }
        if self.frequencies_hz.len() > 1_000_001 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.frequency_psd_grid",
                value: "grid exceeds one-million-point bound".to_string(),
            });
        }
        if self
            .frequencies_hz
            .windows(2)
            .any(|window| !window[0].is_finite() || window[1] <= window[0])
            || self.frequencies_hz[0] < 0.0
            || !self
                .frequencies_hz
                .last()
                .is_some_and(|value| value.is_finite())
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.frequency_psd_grid",
                value: "finite, non-negative, strictly increasing grid required".to_string(),
            });
        }
        if self
            .acceleration_psd_m2_per_s3
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.acceleration_psd_m2_per_s3",
                value: "finite non-negative PSD required".to_string(),
            });
        }
        for (field, value) in [
            ("modal_random_vibration.peak_factor", self.peak_factor),
            (
                "modal_random_vibration.allowable_displacement_m",
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
        Ok(())
    }

    fn validate_against_modal(
        &self,
        modal: &Solid3DModalResult,
    ) -> Result<(), StructuralMechanicsError> {
        self.validate()?;
        if !modal.converged {
            return Err(StructuralMechanicsError::ModalNonConvergence {
                iterations: modal.solver_iterations,
                residual: modal.residual_norms.iter().copied().fold(0.0_f64, f64::max),
            });
        }
        if modal.normalization != ModalNormalization::MassNormalized {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.modal_normalization",
                value: "mass-normalized modes required".to_string(),
            });
        }
        if modal.frequencies_hz.is_empty()
            || modal.frequencies_hz.len() != self.mode_participation.len()
            || modal.frequencies_hz.len() != self.damping_ratios.len()
            || modal.frequencies_hz.len() != modal.residual_norms.len()
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.modal_arrays",
                value: "modal frequencies, residuals, participation, and damping must align"
                    .to_string(),
            });
        }
        if modal
            .frequencies_hz
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || modal
                .residual_norms
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.modal_result",
                value: "finite positive frequencies and residuals required".to_string(),
            });
        }
        if self
            .mode_participation
            .iter()
            .any(|value| !value.is_finite())
            || self
                .damping_ratios
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_random_vibration.modal_controls",
                value: "finite participation and non-negative damping required".to_string(),
            });
        }
        Ok(())
    }
}

/// Per-mode spectral response and feasibility diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalRandomVibrationModeResult {
    pub mode_index: usize,
    pub frequency_hz: f64,
    pub participation: f64,
    pub damping_ratio: f64,
    pub displacement_psd_m2_per_hz: Vec<f64>,
    pub displacement_variance_m2: f64,
    pub rms_displacement_m: f64,
    pub peak_displacement_m: f64,
    pub maximum_dynamic_amplification_s2: f64,
    pub feasible: bool,
}

/// Versioned modal spectral random-vibration result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalRandomVibrationResult {
    pub schema_version: u32,
    pub study: ModalRandomVibrationStudy,
    pub modal_frequencies_hz: Vec<f64>,
    pub modes: Vec<ModalRandomVibrationModeResult>,
    pub total_displacement_variance_m2: f64,
    pub rms_displacement_m: f64,
    pub peak_displacement_m: f64,
    pub feasible: bool,
    pub normalization: ModalNormalization,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Evaluate independent modal responses to a base-acceleration PSD.
pub fn run_modal_random_vibration(
    modal: &Solid3DModalResult,
    study: &ModalRandomVibrationStudy,
) -> Result<ModalRandomVibrationResult, StructuralMechanicsError> {
    study.validate_against_modal(modal)?;

    let mut modes = Vec::with_capacity(modal.frequencies_hz.len());
    let mut total_displacement_variance_m2 = 0.0;
    for (mode_index, ((frequency_hz, participation), damping_ratio)) in modal
        .frequencies_hz
        .iter()
        .zip(&study.mode_participation)
        .zip(&study.damping_ratios)
        .enumerate()
    {
        let natural_omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let mut displacement_psd_m2_per_hz = Vec::with_capacity(study.frequencies_hz.len());
        let mut maximum_dynamic_amplification_s2: f64 = 0.0;
        for (frequency, acceleration_psd) in study
            .frequencies_hz
            .iter()
            .zip(&study.acceleration_psd_m2_per_s3)
        {
            let omega = 2.0 * std::f64::consts::PI * frequency;
            let real = natural_omega * natural_omega - omega * omega;
            let imaginary = 2.0 * damping_ratio * natural_omega * omega;
            let denominator = real * real + imaginary * imaginary;
            if !denominator.is_finite() || denominator <= f64::MIN_POSITIVE {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "modal_random_vibration.resonance",
                    value: format!("mode {mode_index} has an unresolved undamped resonance"),
                });
            }
            let transfer_squared = participation * participation / denominator;
            let transfer = transfer_squared.sqrt();
            maximum_dynamic_amplification_s2 = maximum_dynamic_amplification_s2.max(transfer);
            displacement_psd_m2_per_hz.push(acceleration_psd * transfer_squared);
        }
        let displacement_variance_m2 =
            trapezoid_integral(&study.frequencies_hz, &displacement_psd_m2_per_hz);
        if !displacement_variance_m2.is_finite() || displacement_variance_m2 < 0.0 {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "modal random-vibration PSD integration",
                message: format!("non-finite variance for mode {mode_index}"),
            });
        }
        let rms_displacement_m = displacement_variance_m2.sqrt();
        let peak_displacement_m = study.peak_factor * rms_displacement_m;
        total_displacement_variance_m2 += displacement_variance_m2;
        modes.push(ModalRandomVibrationModeResult {
            mode_index,
            frequency_hz: *frequency_hz,
            participation: *participation,
            damping_ratio: *damping_ratio,
            displacement_psd_m2_per_hz,
            displacement_variance_m2,
            rms_displacement_m,
            peak_displacement_m,
            maximum_dynamic_amplification_s2,
            feasible: peak_displacement_m <= study.allowable_displacement_m,
        });
    }

    let rms_displacement_m = total_displacement_variance_m2.sqrt();
    let peak_displacement_m = study.peak_factor * rms_displacement_m;
    let feasible = peak_displacement_m <= study.allowable_displacement_m
        && modes.iter().all(|mode| mode.feasible);
    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.modal-random-vibration.v1".to_string();
    execution.limitations.extend([
        "Response uses caller-supplied base-acceleration PSD, modal participation, and damping."
            .to_string(),
        "Modal variances are accumulated by independent-mode SRSS; cross-modal correlation and a full spatial PSD matrix are not represented.".to_string(),
        "This is a bounded modal screening contract, not product-scale random-vibration FEM, fatigue, measured correlation, or accelerator evidence.".to_string(),
    ]);

    Ok(ModalRandomVibrationResult {
        schema_version: MODAL_RANDOM_VIBRATION_SCHEMA_VERSION,
        study: study.clone(),
        modal_frequencies_hz: modal.frequencies_hz.clone(),
        modes,
        total_displacement_variance_m2,
        rms_displacement_m,
        peak_displacement_m,
        feasible,
        normalization: modal.normalization,
        model_assumption: StructuralModelAssumption::ModalRandomVibration,
        execution,
    })
}

fn trapezoid_integral(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(x_pair, y_pair)| 0.5 * (x_pair[1] - x_pair[0]) * (y_pair[0] + y_pair[1]))
        .sum()
}
