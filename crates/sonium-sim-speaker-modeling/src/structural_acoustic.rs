//! Frequency-domain structural/acoustic boundary feedback.
//!
//! The product BEM uses peak phasors with an `e^{-iωt}` convention, while the
//! modeling crate's electromechanical contracts use RMS phasors with
//! `e^{+iωt}`.  This module owns the conversion at the product boundary and
//! exposes a small modal structural model that can be used as a partitioned
//! acoustic boundary/load operator.  A positive modal damping ratio is
//! passive; the returned response therefore includes an explicit absorbed
//! power check for every frequency.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{ModelingError, require_positive};

const RMS_TO_PEAK: f64 = std::f64::consts::SQRT_2;
const MIN_PHASOR: f64 = 1.0e-15;

/// One mass-normalized panel/cone mode used by the product feedback model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralAcousticMode {
    /// Undamped natural frequency of the mode (Hz).
    pub natural_frequency_hz: f64,
    /// Generalized modal mass (kg).
    pub effective_mass_kg: f64,
    /// Modal viscous damping ratio.
    pub damping_ratio: f64,
    /// Pressure-to-generalized-force participation factor.
    pub pressure_participation: f64,
    /// Recovered stress amplitude per modal displacement (Pa/m).
    pub stress_per_displacement_pa_per_m: f64,
}

impl StructuralAcousticMode {
    pub fn validate(self, index: usize) -> Result<(), ModelingError> {
        for (field, value) in [
            ("natural_frequency_hz", self.natural_frequency_hz),
            ("effective_mass_kg", self.effective_mass_kg),
            (
                "stress_per_displacement_pa_per_m",
                self.stress_per_displacement_pa_per_m,
            ),
        ] {
            if !value.is_finite() {
                return Err(ModelingError::NonFinite { field });
            }
        }
        if self.natural_frequency_hz <= 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "structural_acoustic_mode.natural_frequency_hz",
                constraint: "positive",
            });
        }
        if self.effective_mass_kg <= 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "structural_acoustic_mode.effective_mass_kg",
                constraint: "positive",
            });
        }
        if !self.damping_ratio.is_finite() || self.damping_ratio < 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "structural_acoustic_mode.damping_ratio",
                constraint: "finite and non-negative",
            });
        }
        if !self.pressure_participation.is_finite() {
            return Err(ModelingError::NonFinite {
                field: "structural_acoustic_mode.pressure_participation",
            });
        }
        if self.stress_per_displacement_pa_per_m < 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "structural_acoustic_mode.stress_per_displacement_pa_per_m",
                constraint: "non-negative",
            });
        }
        let _ = index;
        Ok(())
    }
}

/// Product-level structural boundary model.
///
/// The modes represent a panel, cone, or moving-assembly attachment reduced
/// to its pressure-coupled generalized coordinates.  The response is linear
/// at each frequency; the speaker BEM performs the outer partitioned iteration
/// when the returned velocity is fed back into the acoustic boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralAcousticFeedback {
    /// Effective pressure-loaded area of the panel/cone boundary (m²).
    pub area_m2: f64,
    /// Modal basis used for pressure/displacement/stress recovery.
    pub modes: Vec<StructuralAcousticMode>,
    /// Maximum partitioned acoustic/structural feedback iterations.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// Relative boundary-velocity convergence tolerance.
    #[serde(default = "default_velocity_tolerance")]
    pub velocity_tolerance: f64,
    /// Under-relaxation for the partitioned load exchange.
    #[serde(default = "default_under_relaxation")]
    pub under_relaxation: f64,
}

fn default_max_iterations() -> usize {
    12
}

fn default_velocity_tolerance() -> f64 {
    1.0e-6
}

fn default_under_relaxation() -> f64 {
    0.8
}

impl StructuralAcousticFeedback {
    pub fn validate(&self) -> Result<(), ModelingError> {
        require_positive("structural_acoustic_feedback.area_m2", self.area_m2)?;
        if self.modes.is_empty() {
            return Err(ModelingError::OutOfRange {
                field: "structural_acoustic_feedback.modes",
                constraint: "non-empty",
            });
        }
        if self.max_iterations == 0
            || !self.velocity_tolerance.is_finite()
            || self.velocity_tolerance <= 0.0
            || !self.under_relaxation.is_finite()
            || !(0.0..=1.0).contains(&self.under_relaxation)
            || self.under_relaxation <= 0.0
        {
            return Err(ModelingError::OutOfRange {
                field: "structural_acoustic_feedback.iteration_policy",
                constraint: "positive finite values with relaxation in (0, 1]",
            });
        }
        for (index, mode) in self.modes.iter().copied().enumerate() {
            mode.validate(index)?;
        }
        Ok(())
    }

    /// Solve the modal structural response to a complex RMS cavity/exterior
    /// pressure in the modeling `e^{+iωt}` convention.
    pub fn response_at(
        &self,
        frequency_hz: f64,
        pressure_rms_pa: Complex64,
    ) -> Result<StructuralAcousticResponse, ModelingError> {
        let block = StructuralAcousticModalBlock::from_feedback(self, frequency_hz)?;
        if !pressure_rms_pa.re.is_finite() || !pressure_rms_pa.im.is_finite() {
            return Err(ModelingError::NonFinite {
                field: "structural_acoustic_feedback.pressure_rms_pa",
            });
        }
        let generalized_displacements = block.solve_generalized_displacements(pressure_rms_pa)?;
        let mut response = StructuralAcousticResponse::default();
        for (mode, generalized_displacement) in
            self.modes.iter().copied().zip(generalized_displacements)
        {
            let displacement = generalized_displacement * mode.pressure_participation;
            let omega = std::f64::consts::TAU * frequency_hz;
            let velocity = Complex64::new(0.0, omega) * displacement;
            let generalized_force = pressure_rms_pa * self.area_m2 * mode.pressure_participation;
            response.displacement_rms_m += displacement;
            response.velocity_rms_m_per_s += velocity;
            response.stress_rms_pa += displacement * mode.stress_per_displacement_pa_per_m;
            response.volume_velocity_rms_m3_per_s += velocity * self.area_m2;
            response.absorbed_power_w += (generalized_force * velocity.conj()).re;
        }
        response.frequency_hz = frequency_hz;
        response.pressure_rms_pa = pressure_rms_pa;
        if !response.absorbed_power_w.is_finite() || response.absorbed_power_w < -1.0e-10 {
            return Err(ModelingError::InvalidRadiationImpedance);
        }
        Ok(response)
    }
}

/// Explicit frequency-domain modal block for structural/acoustic coupling.
///
/// The unknown for each mode is its generalized displacement `q`. For an RMS
/// pressure excitation `p`, the block equations are
///
/// ```text
/// D q - (A * participation) p = 0
/// v_surface = iω * participation * q
/// ```
///
/// where `D` is the damped modal dynamic stiffness. This is the reusable
/// structural block that a future product monolithic matrix can couple to
/// directly; `StructuralAcousticFeedback::response_at` uses its solve path,
/// while existing FEM callers may still consume the equivalent condensed
/// Robin coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralAcousticModalBlock {
    frequency_hz: f64,
    dynamic_stiffness: Vec<Complex64>,
    pressure_to_generalized_force: Vec<f64>,
    generalized_to_surface_velocity: Vec<Complex64>,
}

impl StructuralAcousticModalBlock {
    /// Create the explicit modal block for one positive frequency.
    pub fn from_feedback(
        feedback: &StructuralAcousticFeedback,
        frequency_hz: f64,
    ) -> Result<Self, ModelingError> {
        feedback.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(ModelingError::InvalidFrequencyGrid);
        }

        let omega = std::f64::consts::TAU * frequency_hz;
        let mut dynamic_stiffness = Vec::with_capacity(feedback.modes.len());
        let mut pressure_to_generalized_force = Vec::with_capacity(feedback.modes.len());
        let mut generalized_to_surface_velocity = Vec::with_capacity(feedback.modes.len());
        for mode in feedback.modes.iter().copied() {
            let mass = mode.effective_mass_kg;
            let natural_omega = std::f64::consts::TAU * mode.natural_frequency_hz;
            let stiffness = mass * natural_omega.powi(2);
            let damping = 2.0 * mode.damping_ratio * mass * natural_omega;
            let dynamic = Complex64::new(stiffness - mass * omega * omega, damping * omega);
            let force = feedback.area_m2 * mode.pressure_participation;
            let velocity = Complex64::new(0.0, omega * mode.pressure_participation);
            if !dynamic.is_finite() || !force.is_finite() || !velocity.is_finite() {
                return Err(ModelingError::InvalidRadiationImpedance);
            }
            dynamic_stiffness.push(dynamic);
            pressure_to_generalized_force.push(force);
            generalized_to_surface_velocity.push(velocity);
        }

        Ok(Self {
            frequency_hz,
            dynamic_stiffness,
            pressure_to_generalized_force,
            generalized_to_surface_velocity,
        })
    }

    /// Frequency represented by this block in Hz.
    pub fn frequency_hz(&self) -> f64 {
        self.frequency_hz
    }

    /// Number of explicit generalized structural unknowns.
    pub fn mode_count(&self) -> usize {
        self.dynamic_stiffness.len()
    }

    /// Damped modal dynamic-stiffness diagonal, one entry per mode.
    pub fn dynamic_stiffness(&self) -> &[Complex64] {
        &self.dynamic_stiffness
    }

    /// Generalized force coefficient multiplying RMS pressure.
    pub fn pressure_to_generalized_force(&self) -> &[f64] {
        &self.pressure_to_generalized_force
    }

    /// Surface-velocity coefficient multiplying generalized displacement.
    pub fn generalized_to_surface_velocity(&self) -> &[Complex64] {
        &self.generalized_to_surface_velocity
    }

    fn validate_layout(&self) -> Result<(), ModelingError> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(ModelingError::InvalidFrequencyGrid);
        }
        let count = self.dynamic_stiffness.len();
        if count == 0
            || self.pressure_to_generalized_force.len() != count
            || self.generalized_to_surface_velocity.len() != count
            || self
                .dynamic_stiffness
                .iter()
                .any(|value| !value.is_finite())
            || self
                .pressure_to_generalized_force
                .iter()
                .any(|value| !value.is_finite())
            || self
                .generalized_to_surface_velocity
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(ModelingError::InvalidRadiationImpedance);
        }
        Ok(())
    }

    /// Solve the explicit modal equations for one RMS pressure value.
    pub fn solve_generalized_displacements(
        &self,
        pressure_rms_pa: Complex64,
    ) -> Result<Vec<Complex64>, ModelingError> {
        self.validate_layout()?;
        if !pressure_rms_pa.is_finite() {
            return Err(ModelingError::NonFinite {
                field: "structural_acoustic_modal_block.pressure_rms_pa",
            });
        }

        let mut displacements = Vec::with_capacity(self.mode_count());
        for (&dynamic, &force) in self
            .dynamic_stiffness
            .iter()
            .zip(&self.pressure_to_generalized_force)
        {
            if dynamic.norm_sqr() <= MIN_PHASOR * MIN_PHASOR {
                return Err(ModelingError::InvalidRadiationImpedance);
            }
            let displacement = pressure_rms_pa * force / dynamic;
            if !displacement.is_finite() {
                return Err(ModelingError::InvalidRadiationImpedance);
            }
            displacements.push(displacement);
        }
        Ok(displacements)
    }

    /// Recover surface velocity from explicit generalized displacements.
    pub fn surface_velocity(
        &self,
        generalized_displacements: &[Complex64],
    ) -> Result<Complex64, ModelingError> {
        self.validate_layout()?;
        if generalized_displacements.len() != self.mode_count() {
            return Err(ModelingError::InvalidRadiationImpedance);
        }
        if generalized_displacements
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(ModelingError::NonFinite {
                field: "structural_acoustic_modal_block.generalized_displacements",
            });
        }
        let velocity: Complex64 = self
            .generalized_to_surface_velocity
            .iter()
            .zip(generalized_displacements)
            .map(|(coefficient, displacement)| *coefficient * *displacement)
            .sum();
        velocity
            .is_finite()
            .then_some(velocity)
            .ok_or(ModelingError::InvalidRadiationImpedance)
    }

    /// Relative residual of the explicit structural equations.
    pub fn relative_residual(
        &self,
        pressure_rms_pa: Complex64,
        generalized_displacements: &[Complex64],
    ) -> Result<f64, ModelingError> {
        self.validate_layout()?;
        if generalized_displacements.len() != self.mode_count()
            || !pressure_rms_pa.is_finite()
            || generalized_displacements
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(ModelingError::InvalidRadiationImpedance);
        }
        let mut residual_norm_sq = 0.0;
        let mut rhs_norm_sq = 0.0;
        for ((&dynamic, &force), &displacement) in self
            .dynamic_stiffness
            .iter()
            .zip(&self.pressure_to_generalized_force)
            .zip(generalized_displacements)
        {
            let rhs = pressure_rms_pa * force;
            let residual = dynamic * displacement - rhs;
            residual_norm_sq += residual.norm_sqr();
            rhs_norm_sq += rhs.norm_sqr();
        }
        Ok(residual_norm_sq.sqrt() / rhs_norm_sq.sqrt().max(MIN_PHASOR))
    }
}

/// Validated frequency-domain structural response and conversion helpers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralAcousticResponse {
    #[serde(default)]
    pub frequency_hz: f64,
    #[serde(default)]
    pub pressure_rms_pa: Complex64,
    #[serde(default)]
    pub displacement_rms_m: Complex64,
    #[serde(default)]
    pub velocity_rms_m_per_s: Complex64,
    #[serde(default)]
    pub stress_rms_pa: Complex64,
    #[serde(default)]
    pub volume_velocity_rms_m3_per_s: Complex64,
    #[serde(default)]
    pub absorbed_power_w: f64,
}

impl Default for StructuralAcousticResponse {
    fn default() -> Self {
        Self {
            frequency_hz: 0.0,
            pressure_rms_pa: Complex64::default(),
            displacement_rms_m: Complex64::default(),
            velocity_rms_m_per_s: Complex64::default(),
            stress_rms_pa: Complex64::default(),
            volume_velocity_rms_m3_per_s: Complex64::default(),
            absorbed_power_w: 0.0,
        }
    }
}

impl StructuralAcousticResponse {
    /// Convert the modeling RMS `e^{+iωt}` normal velocity to the speaker
    /// BEM peak `e^{-iωt}` boundary convention.
    pub fn velocity_bem_peak(&self) -> Complex64 {
        RMS_TO_PEAK * self.velocity_rms_m_per_s.conj()
    }

    /// Convert the modeling RMS displacement to a BEM-compatible peak phasor.
    pub fn displacement_peak(&self) -> Complex64 {
        RMS_TO_PEAK * self.displacement_rms_m.conj()
    }

    /// Relative response error used by the outer partitioned fixed point.
    pub fn relative_velocity_error(&self, previous_bem_peak: Complex64) -> f64 {
        let current = self.velocity_bem_peak();
        (current - previous_bem_peak).norm() / current.norm().max(MIN_PHASOR)
    }
}

/// One frequency-domain structural/acoustic response supplied by a reference
/// measurement or an independently evaluated analytical/FEM fixture.
///
/// Pressure is the caller-supplied excitation. The remaining fields are
/// compared against the production modal response returned by
/// [`StructuralAcousticFeedback::response_at`]. Keeping the excitation in the
/// point makes the reference replayable without embedding a hidden drive
/// convention in the correlator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredStructuralAcousticPoint {
    pub frequency_hz: f64,
    pub pressure_rms_pa: Complex64,
    pub displacement_rms_m: Complex64,
    pub velocity_rms_m_per_s: Complex64,
    pub stress_rms_pa: Complex64,
    pub volume_velocity_rms_m3_per_s: Complex64,
    pub absorbed_power_w: f64,
}

/// Error budgets for a structural/acoustic reference correlation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralAcousticReferenceTolerances {
    pub displacement_m: f64,
    pub velocity_m_per_s: f64,
    pub stress_pa: f64,
    pub volume_velocity_m3_per_s: f64,
    pub absorbed_power_w: f64,
}

impl Default for StructuralAcousticReferenceTolerances {
    fn default() -> Self {
        Self {
            displacement_m: 1.0e-9,
            velocity_m_per_s: 1.0e-6,
            stress_pa: 1.0e-3,
            volume_velocity_m3_per_s: 1.0e-9,
            absorbed_power_w: 1.0e-9,
        }
    }
}

impl StructuralAcousticReferenceTolerances {
    fn validate(self) -> Result<(), ModelingError> {
        for (field, value) in [
            ("modal_reference.displacement_m", self.displacement_m),
            ("modal_reference.velocity_m_per_s", self.velocity_m_per_s),
            ("modal_reference.stress_pa", self.stress_pa),
            (
                "modal_reference.volume_velocity_m3_per_s",
                self.volume_velocity_m3_per_s,
            ),
            ("modal_reference.absorbed_power_w", self.absorbed_power_w),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ModelingError::OutOfRange {
                    field,
                    constraint: "finite and non-negative",
                });
            }
        }
        Ok(())
    }
}

/// RMSE and worst-case error for one modal response observable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralAcousticMetricReport {
    pub samples: usize,
    pub rmse: f64,
    pub maximum_absolute_error: f64,
    pub normalized_rmse: f64,
    pub tolerance: f64,
    pub passed: bool,
}

/// Complete modal reference correlation report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralAcousticReferenceReport {
    pub displacement: StructuralAcousticMetricReport,
    pub velocity: StructuralAcousticMetricReport,
    pub stress: StructuralAcousticMetricReport,
    pub volume_velocity: StructuralAcousticMetricReport,
    pub absorbed_power: StructuralAcousticMetricReport,
    pub passed: bool,
}

/// Correlate the production modal structural response against an independent
/// reference point set.
///
/// The point grid must be strictly increasing in frequency. A report with
/// `passed == false` is valid evidence that failed its caller-selected gate;
/// malformed input and invalid model responses are returned as errors.
pub fn correlate_structural_acoustic_reference(
    feedback: &StructuralAcousticFeedback,
    measurements: &[MeasuredStructuralAcousticPoint],
    tolerances: StructuralAcousticReferenceTolerances,
) -> Result<StructuralAcousticReferenceReport, ModelingError> {
    if measurements.is_empty() {
        return Err(ModelingError::InvalidReferenceEvidence(
            "modal reference point set must not be empty".to_owned(),
        ));
    }
    tolerances.validate()?;
    feedback.validate()?;

    let frequencies: Vec<_> = measurements
        .iter()
        .map(|measurement| measurement.frequency_hz)
        .collect();
    crate::validate_frequency_grid(&frequencies)?;

    let mut displacement_errors = Vec::with_capacity(measurements.len());
    let mut velocity_errors = Vec::with_capacity(measurements.len());
    let mut stress_errors = Vec::with_capacity(measurements.len());
    let mut volume_velocity_errors = Vec::with_capacity(measurements.len());
    let mut absorbed_power_errors = Vec::with_capacity(measurements.len());

    for measurement in measurements {
        for (field, value) in [
            (
                "modal_reference.pressure_rms_pa",
                measurement.pressure_rms_pa,
            ),
            (
                "modal_reference.displacement_rms_m",
                measurement.displacement_rms_m,
            ),
            (
                "modal_reference.velocity_rms_m_per_s",
                measurement.velocity_rms_m_per_s,
            ),
            ("modal_reference.stress_rms_pa", measurement.stress_rms_pa),
            (
                "modal_reference.volume_velocity_rms_m3_per_s",
                measurement.volume_velocity_rms_m3_per_s,
            ),
        ] {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(ModelingError::NonFinite { field });
            }
        }
        if !measurement.absorbed_power_w.is_finite() || measurement.absorbed_power_w < 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "modal_reference.absorbed_power_w",
                constraint: "finite and non-negative",
            });
        }

        let predicted =
            feedback.response_at(measurement.frequency_hz, measurement.pressure_rms_pa)?;
        displacement_errors.push(complex_error(
            predicted.displacement_rms_m,
            measurement.displacement_rms_m,
        ));
        velocity_errors.push(complex_error(
            predicted.velocity_rms_m_per_s,
            measurement.velocity_rms_m_per_s,
        ));
        stress_errors.push(complex_error(
            predicted.stress_rms_pa,
            measurement.stress_rms_pa,
        ));
        volume_velocity_errors.push(complex_error(
            predicted.volume_velocity_rms_m3_per_s,
            measurement.volume_velocity_rms_m3_per_s,
        ));
        absorbed_power_errors.push((
            (predicted.absorbed_power_w - measurement.absorbed_power_w).abs(),
            measurement.absorbed_power_w.abs(),
        ));
    }

    let displacement = metric_report(&displacement_errors, tolerances.displacement_m);
    let velocity = metric_report(&velocity_errors, tolerances.velocity_m_per_s);
    let stress = metric_report(&stress_errors, tolerances.stress_pa);
    let volume_velocity =
        metric_report(&volume_velocity_errors, tolerances.volume_velocity_m3_per_s);
    let absorbed_power = metric_report(&absorbed_power_errors, tolerances.absorbed_power_w);
    let passed = displacement.passed
        && velocity.passed
        && stress.passed
        && volume_velocity.passed
        && absorbed_power.passed;

    Ok(StructuralAcousticReferenceReport {
        displacement,
        velocity,
        stress,
        volume_velocity,
        absorbed_power,
        passed,
    })
}

fn complex_error(actual: Complex64, reference: Complex64) -> (f64, f64) {
    ((actual - reference).norm(), reference.norm())
}

fn metric_report(errors: &[(f64, f64)], tolerance: f64) -> StructuralAcousticMetricReport {
    let squared_error_sum: f64 = errors.iter().map(|(error, _)| error * error).sum();
    let squared_reference_sum: f64 = errors
        .iter()
        .map(|(_, reference)| reference * reference)
        .sum();
    let rmse = (squared_error_sum / errors.len() as f64).sqrt();
    let maximum_absolute_error = errors.iter().map(|(error, _)| *error).fold(0.0, f64::max);
    let normalized_rmse = rmse
        / (squared_reference_sum / errors.len() as f64)
            .sqrt()
            .max(MIN_PHASOR);
    StructuralAcousticMetricReport {
        samples: errors.len(),
        rmse,
        maximum_absolute_error,
        normalized_rmse,
        tolerance,
        passed: rmse <= tolerance && maximum_absolute_error <= tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feedback() -> StructuralAcousticFeedback {
        StructuralAcousticFeedback {
            area_m2: 0.02,
            modes: vec![StructuralAcousticMode {
                natural_frequency_hz: 400.0,
                effective_mass_kg: 0.2,
                damping_ratio: 0.03,
                pressure_participation: 1.0,
                stress_per_displacement_pa_per_m: 2.0e6,
            }],
            ..StructuralAcousticFeedback {
                max_iterations: 12,
                velocity_tolerance: 1.0e-6,
                under_relaxation: 0.8,
                area_m2: 0.02,
                modes: Vec::new(),
            }
        }
    }

    #[test]
    fn modal_response_is_finite_and_damping_absorbs_power() {
        let response = feedback()
            .response_at(250.0, Complex64::new(10.0, 3.0))
            .unwrap();
        assert!(response.velocity_rms_m_per_s.norm() > 0.0);
        assert!(response.stress_rms_pa.norm() > 0.0);
        assert!(response.absorbed_power_w >= 0.0);
        assert_eq!(
            response.velocity_bem_peak(),
            std::f64::consts::SQRT_2 * response.velocity_rms_m_per_s.conj()
        );
    }

    #[test]
    fn explicit_modal_block_matches_condensed_response() {
        let model = feedback();
        let pressure = Complex64::new(10.0, 3.0);
        let block = StructuralAcousticModalBlock::from_feedback(&model, 250.0).unwrap();
        let generalized = block.solve_generalized_displacements(pressure).unwrap();
        let response = model.response_at(250.0, pressure).unwrap();

        assert_eq!(block.mode_count(), model.modes.len());
        assert_eq!(block.dynamic_stiffness().len(), 1);
        assert_eq!(block.pressure_to_generalized_force(), &[model.area_m2]);
        assert!(block.relative_residual(pressure, &generalized).unwrap() < 1.0e-12);
        assert!(
            (block.surface_velocity(&generalized).unwrap() - response.velocity_rms_m_per_s).norm()
                < 1.0e-12
        );
        assert!(
            (generalized[0] * model.modes[0].pressure_participation - response.displacement_rms_m)
                .norm()
                < 1.0e-12
        );
    }

    #[test]
    fn explicit_modal_block_rejects_malformed_and_singular_solves() {
        let model = feedback();
        let block = StructuralAcousticModalBlock::from_feedback(&model, 250.0).unwrap();
        assert!(matches!(
            block.solve_generalized_displacements(Complex64::new(f64::NAN, 0.0)),
            Err(ModelingError::NonFinite {
                field: "structural_acoustic_modal_block.pressure_rms_pa"
            })
        ));
        assert!(matches!(
            block.surface_velocity(&[]),
            Err(ModelingError::InvalidRadiationImpedance)
        ));

        let mut singular_model = model;
        singular_model.modes[0].damping_ratio = 0.0;
        let singular = StructuralAcousticModalBlock::from_feedback(&singular_model, 400.0).unwrap();
        assert!(matches!(
            singular.solve_generalized_displacements(Complex64::new(1.0, 0.0)),
            Err(ModelingError::InvalidRadiationImpedance)
        ));
    }

    #[test]
    fn zero_damping_is_rejected_only_by_validation_policy_when_negative() {
        let mut model = feedback();
        model.modes[0].damping_ratio = -0.1;
        assert!(model.validate().is_err());
    }

    #[test]
    fn modal_reference_correlation_matches_independent_closed_form() {
        let model = feedback();
        let frequencies = [100.0, 250.0, 800.0];
        let measurements: Vec<_> = frequencies
            .into_iter()
            .map(|frequency_hz| {
                let pressure = Complex64::new(10.0, 3.0);
                let response = model.response_at(frequency_hz, pressure).unwrap();
                let omega = std::f64::consts::TAU * frequency_hz;
                let mode = model.modes[0];
                let mass = mode.effective_mass_kg;
                let stiffness = mass * (std::f64::consts::TAU * mode.natural_frequency_hz).powi(2);
                let damping = 2.0
                    * mode.damping_ratio
                    * mass
                    * std::f64::consts::TAU
                    * mode.natural_frequency_hz;
                let denominator = Complex64::new(stiffness - mass * omega * omega, damping * omega);
                let expected_displacement =
                    pressure * model.area_m2 * mode.pressure_participation.powi(2) / denominator;
                assert!((response.displacement_rms_m - expected_displacement).norm() < 1.0e-14);
                MeasuredStructuralAcousticPoint {
                    frequency_hz,
                    pressure_rms_pa: pressure,
                    displacement_rms_m: response.displacement_rms_m,
                    velocity_rms_m_per_s: response.velocity_rms_m_per_s,
                    stress_rms_pa: response.stress_rms_pa,
                    volume_velocity_rms_m3_per_s: response.volume_velocity_rms_m3_per_s,
                    absorbed_power_w: response.absorbed_power_w,
                }
            })
            .collect();
        let report = correlate_structural_acoustic_reference(
            &model,
            &measurements,
            StructuralAcousticReferenceTolerances::default(),
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.displacement.samples, 3);
        let encoded = serde_json::to_string(&measurements).unwrap();
        let decoded: Vec<MeasuredStructuralAcousticPoint> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.len(), measurements.len());
        for (decoded, original) in decoded.iter().zip(&measurements) {
            assert_eq!(decoded.frequency_hz, original.frequency_hz);
            assert!((decoded.pressure_rms_pa - original.pressure_rms_pa).norm() < 1.0e-15);
            assert!((decoded.displacement_rms_m - original.displacement_rms_m).norm() < 1.0e-15);
            assert!(
                (decoded.velocity_rms_m_per_s - original.velocity_rms_m_per_s).norm() < 1.0e-15
            );
            assert!((decoded.stress_rms_pa - original.stress_rms_pa).norm() < 1.0e-15);
            assert!(
                (decoded.volume_velocity_rms_m3_per_s - original.volume_velocity_rms_m3_per_s)
                    .norm()
                    < 1.0e-15
            );
            assert!((decoded.absorbed_power_w - original.absorbed_power_w).abs() < 1.0e-15);
        }
    }

    #[test]
    fn modal_reference_correlation_reports_mismatch_and_rejects_bad_input() {
        let model = feedback();
        let response = model.response_at(250.0, Complex64::new(10.0, 3.0)).unwrap();
        let mut measurement = MeasuredStructuralAcousticPoint {
            frequency_hz: 250.0,
            pressure_rms_pa: response.pressure_rms_pa,
            displacement_rms_m: response.displacement_rms_m,
            velocity_rms_m_per_s: response.velocity_rms_m_per_s + Complex64::new(1.0e-3, 0.0),
            stress_rms_pa: response.stress_rms_pa,
            volume_velocity_rms_m3_per_s: response.volume_velocity_rms_m3_per_s,
            absorbed_power_w: response.absorbed_power_w,
        };
        let report = correlate_structural_acoustic_reference(
            &model,
            &[measurement],
            StructuralAcousticReferenceTolerances::default(),
        )
        .unwrap();
        assert!(!report.passed);
        assert!(!report.velocity.passed);

        measurement.frequency_hz = 251.0;
        assert!(matches!(
            correlate_structural_acoustic_reference(
                &model,
                &[
                    measurement,
                    MeasuredStructuralAcousticPoint {
                        frequency_hz: 250.0,
                        ..measurement
                    },
                ],
                StructuralAcousticReferenceTolerances::default(),
            ),
            Err(ModelingError::InvalidFrequencyGrid)
        ));
        assert!(matches!(
            correlate_structural_acoustic_reference(
                &model,
                &[MeasuredStructuralAcousticPoint {
                    absorbed_power_w: -1.0,
                    ..measurement
                }],
                StructuralAcousticReferenceTolerances::default(),
            ),
            Err(ModelingError::OutOfRange {
                field: "modal_reference.absorbed_power_w",
                ..
            })
        ));
    }
}
