//! Bounded mass-normalized 3-D modal harmonic-response evidence.
//!
//! This module consumes a converged `Solid3DModalResult` and caller-supplied
//! complex modal-force phasors. It reconstructs one observed displacement DOF
//! by linear modal superposition, retaining the complex response and phase
//! convention for every requested frequency. It is a checked reduced modal
//! reference, not a nonlinear, coupled-mode, or product-scale harmonic FEM
//! solver.

use crate::{
    ModalNormalization, Solid3DModalResult, StructuralExecutionProvenance,
    StructuralMechanicsError, StructuralModelAssumption,
};
use serde::{Deserialize, Serialize};

/// Schema version for the modal harmonic-response result.
pub const MODAL_HARMONIC_RESPONSE_SCHEMA_VERSION: u32 = 1;

/// Caller-supplied modal harmonic excitation and observation controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalHarmonicResponseStudy {
    pub name: String,
    pub frequencies_hz: Vec<f64>,
    /// Real/imaginary modal-force phasors, one pair per mode.
    pub modal_force_phasors: Vec<[f64; 2]>,
    pub damping_ratios: Vec<f64>,
    pub observation_node: usize,
    pub observation_component: usize,
    pub allowable_displacement_m: f64,
}

impl Default for ModalHarmonicResponseStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl ModalHarmonicResponseStudy {
    /// Small bounded frequency-response fixture for a caller-supplied modal result.
    pub fn introductory() -> Self {
        Self {
            name: "modal-harmonic-response-reference".to_string(),
            frequencies_hz: vec![5.0, 10.0, 15.0, 30.0, 60.0],
            modal_force_phasors: vec![[1.0, 0.0], [0.35, 0.0], [0.2, 0.0]],
            damping_ratios: vec![0.02, 0.025, 0.03],
            observation_node: 0,
            observation_component: 0,
            allowable_displacement_m: 0.05,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.name.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.name",
                value: "empty".to_string(),
            });
        }
        if self.frequencies_hz.is_empty()
            || self.frequencies_hz.len() > 1_000_001
            || self
                .frequencies_hz
                .windows(2)
                .any(|window| !window[0].is_finite() || window[1] <= window[0])
            || self
                .frequencies_hz
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.frequency_grid",
                value: "finite, positive, strictly increasing bounded grid required".to_string(),
            });
        }
        if self.modal_force_phasors.is_empty()
            || self
                .modal_force_phasors
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.modal_force_phasors",
                value: "finite modal-force phasors required".to_string(),
            });
        }
        if self
            .damping_ratios
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.damping_ratios",
                value: "finite non-negative damping ratios required".to_string(),
            });
        }
        if !self.allowable_displacement_m.is_finite() || self.allowable_displacement_m <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.allowable_displacement_m",
                value: self.allowable_displacement_m.to_string(),
            });
        }
        if self.observation_component >= 3 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.observation_component",
                value: self.observation_component.to_string(),
            });
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
                field: "modal_harmonic_response.modal_normalization",
                value: "mass-normalized modes required".to_string(),
            });
        }
        if modal.model_assumption != StructuralModelAssumption::Solid3D {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.model_assumption",
                value: "3-D solid modal result required".to_string(),
            });
        }
        if modal.frequencies_hz.is_empty()
            || modal.frequencies_hz.len() != modal.residual_norms.len()
            || modal.frequencies_hz.len() != modal.modes.len()
            || modal.frequencies_hz.len() != self.modal_force_phasors.len()
            || modal.frequencies_hz.len() != self.damping_ratios.len()
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.modal_arrays",
                value: "modal frequencies, residuals, shapes, forces, and damping must align"
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
                field: "modal_harmonic_response.modal_result",
                value: "finite positive frequencies and residuals required".to_string(),
            });
        }
        if modal.modes.iter().any(|mode| {
            self.observation_node >= mode.len()
                || mode.iter().flatten().any(|value| !value.is_finite())
        }) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "modal_harmonic_response.observation_node",
                value: "observation node must exist in every finite mode shape".to_string(),
            });
        }
        Ok(())
    }
}

/// Complex modal contributions and observed displacement at one frequency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalHarmonicResponsePoint {
    pub frequency_hz: f64,
    /// Real/imaginary observed displacement phasor.
    pub displacement_phasor_m: [f64; 2],
    pub displacement_abs_m: f64,
    /// Principal-value lag for the `exp(iωt)` convention, in `[-π, π]`.
    pub phase_lag_rad: f64,
    /// Real/imaginary contribution from each mode at the observation DOF.
    pub modal_displacement_phasors_m: Vec<[f64; 2]>,
}

/// Versioned modal harmonic-response result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalHarmonicResponseResult {
    pub schema_version: u32,
    pub study: ModalHarmonicResponseStudy,
    pub modal_frequencies_hz: Vec<f64>,
    pub points: Vec<ModalHarmonicResponsePoint>,
    pub peak_displacement_abs_m: f64,
    pub peak_frequency_hz: f64,
    pub converged: bool,
    pub feasible: bool,
    pub normalization: ModalNormalization,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Evaluate a mass-normalized 3-D modal harmonic response at one observation DOF.
pub fn run_modal_harmonic_response(
    modal: &Solid3DModalResult,
    study: &ModalHarmonicResponseStudy,
) -> Result<ModalHarmonicResponseResult, StructuralMechanicsError> {
    study.validate_against_modal(modal)?;

    let mut points = Vec::with_capacity(study.frequencies_hz.len());
    let mut peak_displacement_abs_m = 0.0_f64;
    let mut peak_frequency_hz = study.frequencies_hz[0];

    for &frequency_hz in &study.frequencies_hz {
        let omega = std::f64::consts::TAU * frequency_hz;
        let mut observed = [0.0, 0.0];
        let mut modal_displacement_phasors_m = Vec::with_capacity(modal.frequencies_hz.len());

        for (mode_index, (((natural_frequency_hz, modal_force), damping_ratio), mode_shape)) in
            modal
                .frequencies_hz
                .iter()
                .zip(&study.modal_force_phasors)
                .zip(&study.damping_ratios)
                .zip(&modal.modes)
                .enumerate()
        {
            let natural_omega = std::f64::consts::TAU * natural_frequency_hz;
            let denominator_real = natural_omega * natural_omega - omega * omega;
            let denominator_imag = 2.0 * damping_ratio * natural_omega * omega;
            let denominator_norm_squared =
                denominator_real * denominator_real + denominator_imag * denominator_imag;
            if !denominator_norm_squared.is_finite()
                || denominator_norm_squared <= f64::MIN_POSITIVE
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "modal_harmonic_response.resonance",
                    value: format!(
                        "mode {mode_index} has an unresolved undamped resonance at {frequency_hz} Hz"
                    ),
                });
            }

            let modal_coordinate =
                divide_complex(*modal_force, [denominator_real, denominator_imag]);
            let observation_shape = mode_shape[study.observation_node][study.observation_component];
            let contribution = [
                observation_shape * modal_coordinate[0],
                observation_shape * modal_coordinate[1],
            ];
            observed[0] += contribution[0];
            observed[1] += contribution[1];
            modal_displacement_phasors_m.push(contribution);
        }

        let displacement_abs_m = norm_complex(observed);
        if !displacement_abs_m.is_finite() {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "modal harmonic response reconstruction",
                message: format!("non-finite displacement at {frequency_hz} Hz"),
            });
        }
        if displacement_abs_m > peak_displacement_abs_m {
            peak_displacement_abs_m = displacement_abs_m;
            peak_frequency_hz = frequency_hz;
        }
        points.push(ModalHarmonicResponsePoint {
            frequency_hz,
            displacement_phasor_m: observed,
            displacement_abs_m,
            phase_lag_rad: (-observed[1]).atan2(observed[0]),
            modal_displacement_phasors_m,
        });
    }

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.modal-harmonic-response.v1".to_string();
    execution.limitations.extend([
        "Response uses caller-supplied mass-normalized modes, modal-force phasors, damping, and observation DOF.".to_string(),
        "The response is linear independent modal superposition; modal coupling, nonlinearities, and direct complex FEM assembly are not represented.".to_string(),
        "This is bounded modal harmonic evidence, not product-scale frequency-response FEM, measured correlation, external parity, or accelerator evidence.".to_string(),
    ]);

    Ok(ModalHarmonicResponseResult {
        schema_version: MODAL_HARMONIC_RESPONSE_SCHEMA_VERSION,
        study: study.clone(),
        modal_frequencies_hz: modal.frequencies_hz.clone(),
        points,
        peak_displacement_abs_m,
        peak_frequency_hz,
        converged: true,
        feasible: peak_displacement_abs_m <= study.allowable_displacement_m,
        normalization: modal.normalization,
        model_assumption: StructuralModelAssumption::ModalHarmonicResponse,
        execution,
    })
}

fn divide_complex(numerator: [f64; 2], denominator: [f64; 2]) -> [f64; 2] {
    let norm_squared = denominator[0] * denominator[0] + denominator[1] * denominator[1];
    [
        (numerator[0] * denominator[0] + numerator[1] * denominator[1]) / norm_squared,
        (numerator[1] * denominator[0] - numerator[0] * denominator[1]) / norm_squared,
    ]
}

fn norm_complex(value: [f64; 2]) -> f64 {
    value[0].hypot(value[1])
}
