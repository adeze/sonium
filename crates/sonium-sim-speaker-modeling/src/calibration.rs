//! Measurement correlation for the typed driver-model handoff.
//!
//! Calibration is deliberately report-only at this boundary.  A measured
//! fixture can establish whether a golden model is trustworthy, but it must
//! not silently mutate the model consumed by product simulation.  The report
//! covers the curves available in [`DriverModel`] plus paired magnetic-scan
//! samples supplied by a magnetic solver or fixture pipeline.

use serde::{Deserialize, Serialize};

use crate::ModelingError;
use crate::driver_model::{DriverModel, DriverModelError};

/// One measured complex electrical-impedance sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredImpedancePoint {
    pub frequency_hz: f64,
    pub real_ohm: f64,
    pub imaginary_ohm: f64,
}

/// One measured displacement-dependent BL(x) sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredBlPoint {
    pub displacement_m: f64,
    pub force_factor_tm: f64,
}

/// One measured frequency-dependent voice-coil sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredInductancePoint {
    pub frequency_hz: f64,
    pub inductance_h: f64,
    pub eddy_resistance_ohm: f64,
}

/// One measured skin-depth/proximity observation at a motor operating point.
///
/// The eddy-current model is still the source of the calibrated electrical
/// resistance.  These two diagnostics provide an independent check that the
/// retained surface has material/geometry meaning rather than only matching a
/// lumped Le(f) resistance curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredEddyCurrentPoint {
    pub frequency_hz: f64,
    pub flux_density_t: f64,
    pub skin_depth_m: f64,
    pub proximity_factor: f64,
}

/// One measured steady-state thermal operating point.
///
/// The loss inputs identify the same source split used by the driver's
/// conjugate thermal network. Temperatures are retained in the stable
/// [`crate::ThermalNode`] ordering so magnet/gap/cabinet heating cannot be
/// hidden behind a single coil-only temperature.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredThermalPoint {
    pub copper_loss_w: f64,
    pub eddy_loss_w: f64,
    pub temperatures_k: [f64; crate::THERMAL_NODE_COUNT],
}

/// One measured first-quadrant material B-H sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredMagneticBhPoint {
    pub flux_density_t: f64,
    pub field_strength_a_per_m: f64,
}

/// One laser-vibrometry velocity-transfer sample at the model reference drive.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredVelocityPoint {
    pub frequency_hz: f64,
    pub velocity_real_m_per_s: f64,
    pub velocity_imag_m_per_s: f64,
}

/// One measured complex displacement-transfer sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredDisplacementPoint {
    pub frequency_hz: f64,
    pub displacement_real_m: f64,
    pub displacement_imag_m: f64,
}

/// One measured electromechanical response under an independently supplied
/// complex exterior radiation load.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredRadiationResponsePoint {
    pub frequency_hz: f64,
    pub radiation_real_n_s_per_m: f64,
    pub radiation_imag_n_s_per_m: f64,
    pub velocity_real_m_per_s: f64,
    pub velocity_imag_m_per_s: f64,
    pub electrical_real_ohm: f64,
    pub electrical_imag_ohm: f64,
}

/// One paired magnetic scan sample.
///
/// The predicted value is supplied by the magnetic-field solver that owns the
/// scan geometry; keeping it paired here makes the correlation report usable
/// for axisymmetric, sector, and full-3D scans without inventing a field model
/// inside the product handoff.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MagneticScanPoint {
    pub position_m: [f64; 3],
    pub measured_flux_density_t: f64,
    pub predicted_flux_density_t: f64,
}

/// Origin of the data used for a calibration report.
///
/// The origin is part of the serialized evidence so fixture correlation cannot
/// be presented as independent speaker-measurement validation by accident.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationDataOrigin {
    #[default]
    Fixture,
    IndependentMeasurement,
}

/// Identity and traceability for a measurement dataset that was not produced
/// by the model being evaluated.  Requiring this metadata keeps an
/// `IndependentMeasurement` report from being created by merely changing an
/// enum field on a synthetic fixture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentMeasurementMetadata {
    pub dataset_id: String,
    pub source: String,
    pub acquisition_method: String,
}

impl IndependentMeasurementMetadata {
    pub fn validate(&self) -> Result<(), CalibrationError> {
        for (field, value) in [
            ("dataset_id", &self.dataset_id),
            ("source", &self.source),
            ("acquisition_method", &self.acquisition_method),
        ] {
            if value.trim().is_empty() {
                return Err(CalibrationError::InvalidIndependentMetadata { field });
            }
        }
        Ok(())
    }
}

/// Measurement fixture data used to correlate one [`DriverModel`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DriverCalibrationData {
    #[serde(default)]
    pub origin: CalibrationDataOrigin,
    /// Required when `origin` is `IndependentMeasurement`; absent for model
    /// fixtures and legacy fixture JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub independent_measurement: Option<IndependentMeasurementMetadata>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub impedance: Vec<MeasuredImpedancePoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bl_curve: Vec<MeasuredBlPoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inductance_curve: Vec<MeasuredInductancePoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub eddy_current: Vec<MeasuredEddyCurrentPoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thermal_response: Vec<MeasuredThermalPoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub radiation_response: Vec<MeasuredRadiationResponsePoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub magnetic_bh: Vec<MeasuredMagneticBhPoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub displacement_transfer: Vec<MeasuredDisplacementPoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub velocity_transfer: Vec<MeasuredVelocityPoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub magnetic_scan: Vec<MagneticScanPoint>,
}

impl DriverCalibrationData {
    /// Validate all fixture grids and values before correlation.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        match (self.origin, &self.independent_measurement) {
            (CalibrationDataOrigin::Fixture, Some(_)) => {
                return Err(CalibrationError::UnexpectedIndependentMetadata);
            }
            (CalibrationDataOrigin::IndependentMeasurement, None) => {
                return Err(CalibrationError::IndependentMetadataRequired);
            }
            (CalibrationDataOrigin::IndependentMeasurement, Some(metadata)) => {
                metadata.validate()?
            }
            (CalibrationDataOrigin::Fixture, None) => {}
        }
        validate_frequency_grid(
            "impedance",
            self.impedance.iter().map(|point| point.frequency_hz),
        )?;
        for (index, point) in self.impedance.iter().enumerate() {
            if !point.real_ohm.is_finite() || !point.imaginary_ohm.is_finite() {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "impedance",
                    index,
                });
            }
        }

        validate_grid(
            "BL(x)",
            self.bl_curve.iter().map(|point| point.displacement_m),
        )?;
        for (index, point) in self.bl_curve.iter().enumerate() {
            if !point.force_factor_tm.is_finite() {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "BL(x)",
                    index,
                });
            }
        }

        validate_frequency_grid(
            "Le(f)",
            self.inductance_curve.iter().map(|point| point.frequency_hz),
        )?;
        for (index, point) in self.inductance_curve.iter().enumerate() {
            if !point.inductance_h.is_finite()
                || point.inductance_h < 0.0
                || !point.eddy_resistance_ohm.is_finite()
                || point.eddy_resistance_ohm < 0.0
            {
                return Err(CalibrationError::InvalidSample {
                    curve: "Le(f)",
                    index,
                });
            }
        }

        for (index, point) in self.eddy_current.iter().enumerate() {
            if !point.frequency_hz.is_finite()
                || !point.flux_density_t.is_finite()
                || !point.skin_depth_m.is_finite()
                || !point.proximity_factor.is_finite()
            {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "eddy current",
                    index,
                });
            }
            if point.frequency_hz <= 0.0 {
                return Err(CalibrationError::InvalidFrequency {
                    curve: "eddy current",
                    index,
                });
            }
            if point.skin_depth_m <= 0.0 || point.proximity_factor <= 0.0 {
                return Err(CalibrationError::InvalidSample {
                    curve: "eddy current",
                    index,
                });
            }
        }

        for (index, point) in self.thermal_response.iter().enumerate() {
            if !point.copper_loss_w.is_finite()
                || !point.eddy_loss_w.is_finite()
                || point.temperatures_k.iter().any(|value| !value.is_finite())
            {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "thermal response",
                    index,
                });
            }
            if point.copper_loss_w < 0.0
                || point.eddy_loss_w < 0.0
                || point.temperatures_k.iter().any(|value| *value <= 0.0)
            {
                return Err(CalibrationError::InvalidSample {
                    curve: "thermal response",
                    index,
                });
            }
        }

        validate_frequency_grid(
            "radiation response",
            self.radiation_response
                .iter()
                .map(|point| point.frequency_hz),
        )?;
        for (index, point) in self.radiation_response.iter().enumerate() {
            if [
                point.radiation_real_n_s_per_m,
                point.radiation_imag_n_s_per_m,
                point.velocity_real_m_per_s,
                point.velocity_imag_m_per_s,
                point.electrical_real_ohm,
                point.electrical_imag_ohm,
            ]
            .iter()
            .any(|value| !value.is_finite())
            {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "radiation response",
                    index,
                });
            }
            if point.radiation_real_n_s_per_m < 0.0 {
                return Err(CalibrationError::InvalidSample {
                    curve: "radiation response",
                    index,
                });
            }
        }

        validate_grid(
            "B-H",
            self.magnetic_bh.iter().map(|point| point.flux_density_t),
        )?;
        validate_grid(
            "B-H field strength",
            self.magnetic_bh
                .iter()
                .map(|point| point.field_strength_a_per_m),
        )?;
        for (index, point) in self.magnetic_bh.iter().enumerate() {
            if !point.flux_density_t.is_finite() || !point.field_strength_a_per_m.is_finite() {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "B-H",
                    index,
                });
            }
            if point.flux_density_t < 0.0 || point.field_strength_a_per_m < 0.0 {
                return Err(CalibrationError::InvalidSample {
                    curve: "B-H",
                    index,
                });
            }
        }
        if let Some(first) = self.magnetic_bh.first()
            && (first.flux_density_t != 0.0 || first.field_strength_a_per_m != 0.0)
        {
            return Err(CalibrationError::InvalidSample {
                curve: "B-H",
                index: 0,
            });
        }

        validate_frequency_grid(
            "displacement",
            self.displacement_transfer
                .iter()
                .map(|point| point.frequency_hz),
        )?;
        for (index, point) in self.displacement_transfer.iter().enumerate() {
            if !point.displacement_real_m.is_finite() || !point.displacement_imag_m.is_finite() {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "displacement",
                    index,
                });
            }
        }

        validate_frequency_grid(
            "laser velocity",
            self.velocity_transfer
                .iter()
                .map(|point| point.frequency_hz),
        )?;
        for (index, point) in self.velocity_transfer.iter().enumerate() {
            if !point.velocity_real_m_per_s.is_finite() || !point.velocity_imag_m_per_s.is_finite()
            {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "laser velocity",
                    index,
                });
            }
        }

        for (index, point) in self.magnetic_scan.iter().enumerate() {
            if point.position_m.iter().any(|value| !value.is_finite())
                || !point.measured_flux_density_t.is_finite()
                || !point.predicted_flux_density_t.is_finite()
            {
                return Err(CalibrationError::NonFiniteSample {
                    curve: "magnetic scan",
                    index,
                });
            }
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.impedance.is_empty()
            && self.bl_curve.is_empty()
            && self.inductance_curve.is_empty()
            && self.eddy_current.is_empty()
            && self.thermal_response.is_empty()
            && self.radiation_response.is_empty()
            && self.magnetic_bh.is_empty()
            && self.displacement_transfer.is_empty()
            && self.velocity_transfer.is_empty()
            && self.magnetic_scan.is_empty()
    }
}

fn default_skin_depth_tolerance() -> f64 {
    1.0e-6
}

fn default_proximity_factor_tolerance() -> f64 {
    0.05
}

fn default_magnetic_field_tolerance() -> f64 {
    1.0
}

fn default_displacement_tolerance() -> f64 {
    1.0e-6
}

fn default_temperature_tolerance() -> f64 {
    0.25
}

fn default_radiation_velocity_tolerance() -> f64 {
    1.0e-4
}

fn default_radiation_impedance_tolerance() -> f64 {
    0.05
}

/// Absolute RMSE gates for the individual calibration observables.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationTolerances {
    pub impedance_ohm: f64,
    pub bl_tm: f64,
    pub inductance_h: f64,
    pub eddy_resistance_ohm: f64,
    #[serde(default = "default_skin_depth_tolerance")]
    pub skin_depth_m: f64,
    #[serde(default = "default_proximity_factor_tolerance")]
    pub proximity_factor: f64,
    #[serde(default = "default_temperature_tolerance")]
    pub temperature_k: f64,
    #[serde(default = "default_radiation_velocity_tolerance")]
    pub radiation_velocity_m_per_s: f64,
    #[serde(default = "default_radiation_impedance_tolerance")]
    pub radiation_impedance_ohm: f64,
    #[serde(default = "default_magnetic_field_tolerance")]
    pub magnetic_field_a_per_m: f64,
    #[serde(default = "default_displacement_tolerance")]
    pub displacement_m: f64,
    pub velocity_m_per_s: f64,
    pub magnetic_flux_density_t: f64,
}

impl Default for CalibrationTolerances {
    fn default() -> Self {
        Self {
            // These are engineering defaults rather than solver residual
            // tolerances. Production fixtures should provide an explicit
            // tolerance file when instrumentation uncertainty is known.
            impedance_ohm: 0.05,
            bl_tm: 0.05,
            inductance_h: 1.0e-6,
            eddy_resistance_ohm: 0.05,
            skin_depth_m: default_skin_depth_tolerance(),
            proximity_factor: default_proximity_factor_tolerance(),
            temperature_k: default_temperature_tolerance(),
            radiation_velocity_m_per_s: default_radiation_velocity_tolerance(),
            radiation_impedance_ohm: default_radiation_impedance_tolerance(),
            magnetic_field_a_per_m: default_magnetic_field_tolerance(),
            displacement_m: default_displacement_tolerance(),
            velocity_m_per_s: 1.0e-4,
            magnetic_flux_density_t: 0.01,
        }
    }
}

impl CalibrationTolerances {
    pub fn validate(&self) -> Result<(), CalibrationError> {
        for (curve, tolerance) in [
            ("impedance", self.impedance_ohm),
            ("BL(x)", self.bl_tm),
            ("Le(f)", self.inductance_h),
            ("eddy resistance", self.eddy_resistance_ohm),
            ("skin depth", self.skin_depth_m),
            ("proximity factor", self.proximity_factor),
            ("temperature", self.temperature_k),
            ("radiation velocity", self.radiation_velocity_m_per_s),
            ("radiation impedance", self.radiation_impedance_ohm),
            ("B-H field strength", self.magnetic_field_a_per_m),
            ("displacement", self.displacement_m),
            ("laser velocity", self.velocity_m_per_s),
            ("magnetic scan", self.magnetic_flux_density_t),
        ] {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(CalibrationError::InvalidTolerance { curve });
            }
        }
        Ok(())
    }
}

/// Correlation statistics and pass/fail gate for one observable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetricReport {
    pub samples: usize,
    pub rmse: f64,
    pub maximum_absolute_error: f64,
    pub normalized_rmse: f64,
    pub tolerance: f64,
    pub passed: bool,
}

/// One physical observable that may be required before a calibration report
/// is admitted as product-model evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationObservable {
    Impedance,
    BlCurve,
    Inductance,
    EddyResistance,
    SkinDepth,
    ProximityFactor,
    ThermalResponse,
    RadiationVelocity,
    RadiationImpedance,
    MagneticBh,
    Displacement,
    VelocityTransfer,
    MagneticFluxDensity,
}

impl CalibrationObservable {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Impedance => "impedance",
            Self::BlCurve => "BL(x)",
            Self::Inductance => "Le(f)",
            Self::EddyResistance => "eddy resistance",
            Self::SkinDepth => "skin depth",
            Self::ProximityFactor => "proximity factor",
            Self::ThermalResponse => "thermal response",
            Self::RadiationVelocity => "radiation velocity",
            Self::RadiationImpedance => "radiation impedance",
            Self::MagneticBh => "B-H",
            Self::Displacement => "displacement",
            Self::VelocityTransfer => "laser velocity",
            Self::MagneticFluxDensity => "magnetic scan",
        }
    }
}

/// Explicit coverage contract for promoting a calibration report.
///
/// Correlation intentionally permits partial fixture datasets for development.
/// Product promotion must opt into this policy to prevent a passing one-curve
/// fixture from being presented as a complete driver calibration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationCoveragePolicy {
    /// Every observable that must be present and pass its RMSE gate.
    pub required_observables: Vec<CalibrationObservable>,
    /// Require traceable independent-measurement metadata rather than a model
    /// fixture or synthetic reference curve.
    pub require_independent_measurement: bool,
}

impl CalibrationCoveragePolicy {
    /// Full driver-correlation policy for product promotion.
    pub fn full_independent() -> Self {
        Self {
            required_observables: vec![
                CalibrationObservable::Impedance,
                CalibrationObservable::BlCurve,
                CalibrationObservable::Inductance,
                CalibrationObservable::EddyResistance,
                CalibrationObservable::SkinDepth,
                CalibrationObservable::ProximityFactor,
                CalibrationObservable::ThermalResponse,
                CalibrationObservable::RadiationVelocity,
                CalibrationObservable::RadiationImpedance,
                CalibrationObservable::MagneticBh,
                CalibrationObservable::Displacement,
                CalibrationObservable::VelocityTransfer,
                CalibrationObservable::MagneticFluxDensity,
            ],
            require_independent_measurement: true,
        }
    }

    /// Validate the policy itself before it is used to admit evidence.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        let mut seen = std::collections::HashSet::new();
        for observable in &self.required_observables {
            if !seen.insert(*observable) {
                return Err(CalibrationError::DuplicateRequiredObservable {
                    observable: observable.as_str(),
                });
            }
        }
        Ok(())
    }
}

/// Complete correlation report for one driver fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// Origin copied from the input data and retained in calibration artifacts.
    #[serde(default)]
    pub origin: CalibrationDataOrigin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub independent_measurement: Option<IndependentMeasurementMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impedance: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bl_curve: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inductance: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eddy_resistance: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skin_depth: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proximity_factor: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_response: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radiation_velocity: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radiation_impedance: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_bh: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub displacement: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub velocity_transfer: Option<CalibrationMetricReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_flux_density: Option<CalibrationMetricReport>,
    pub passed: bool,
}

impl CalibrationReport {
    /// Validate serialized report integrity independently of a particular
    /// consumer's coverage policy.
    pub fn validate_integrity(&self) -> Result<(), CalibrationError> {
        match (self.origin, &self.independent_measurement) {
            (CalibrationDataOrigin::Fixture, Some(_)) => {
                return Err(CalibrationError::UnexpectedIndependentMetadata);
            }
            (CalibrationDataOrigin::IndependentMeasurement, None) => {
                return Err(CalibrationError::IndependentMetadataRequired);
            }
            (CalibrationDataOrigin::IndependentMeasurement, Some(metadata)) => {
                metadata.validate()?;
            }
            (CalibrationDataOrigin::Fixture, None) => {}
        }
        for (observable, metric) in [
            (CalibrationObservable::Impedance, self.impedance.as_ref()),
            (CalibrationObservable::BlCurve, self.bl_curve.as_ref()),
            (CalibrationObservable::Inductance, self.inductance.as_ref()),
            (
                CalibrationObservable::EddyResistance,
                self.eddy_resistance.as_ref(),
            ),
            (CalibrationObservable::SkinDepth, self.skin_depth.as_ref()),
            (
                CalibrationObservable::ProximityFactor,
                self.proximity_factor.as_ref(),
            ),
            (
                CalibrationObservable::ThermalResponse,
                self.thermal_response.as_ref(),
            ),
            (
                CalibrationObservable::RadiationVelocity,
                self.radiation_velocity.as_ref(),
            ),
            (
                CalibrationObservable::RadiationImpedance,
                self.radiation_impedance.as_ref(),
            ),
            (CalibrationObservable::MagneticBh, self.magnetic_bh.as_ref()),
            (
                CalibrationObservable::Displacement,
                self.displacement.as_ref(),
            ),
            (
                CalibrationObservable::VelocityTransfer,
                self.velocity_transfer.as_ref(),
            ),
            (
                CalibrationObservable::MagneticFluxDensity,
                self.magnetic_flux_density.as_ref(),
            ),
        ] {
            if let Some(metric) = metric {
                validate_metric_report(metric, observable.as_str())?;
            }
        }
        let derived_passed = report_passed(&self.impedance)
            && report_passed(&self.bl_curve)
            && report_passed(&self.inductance)
            && report_passed(&self.eddy_resistance)
            && report_passed(&self.skin_depth)
            && report_passed(&self.proximity_factor)
            && report_passed(&self.thermal_response)
            && report_passed(&self.radiation_velocity)
            && report_passed(&self.radiation_impedance)
            && report_passed(&self.magnetic_bh)
            && report_passed(&self.displacement)
            && report_passed(&self.velocity_transfer)
            && report_passed(&self.magnetic_flux_density);
        if self.passed != derived_passed {
            return Err(CalibrationError::InconsistentReportStatus);
        }
        Ok(())
    }

    /// Require that this report is complete, traceable, and passing for a
    /// caller-selected promotion policy.
    pub fn require_promotion_coverage(
        &self,
        policy: &CalibrationCoveragePolicy,
    ) -> Result<(), CalibrationError> {
        policy.validate()?;
        self.validate_integrity()?;
        if policy.require_independent_measurement
            && (self.origin != CalibrationDataOrigin::IndependentMeasurement
                || self.independent_measurement.is_none())
        {
            return Err(CalibrationError::IndependentReportRequired);
        }
        for observable in &policy.required_observables {
            let metric = match observable {
                CalibrationObservable::Impedance => self.impedance.as_ref(),
                CalibrationObservable::BlCurve => self.bl_curve.as_ref(),
                CalibrationObservable::Inductance => self.inductance.as_ref(),
                CalibrationObservable::EddyResistance => self.eddy_resistance.as_ref(),
                CalibrationObservable::SkinDepth => self.skin_depth.as_ref(),
                CalibrationObservable::ProximityFactor => self.proximity_factor.as_ref(),
                CalibrationObservable::ThermalResponse => self.thermal_response.as_ref(),
                CalibrationObservable::RadiationVelocity => self.radiation_velocity.as_ref(),
                CalibrationObservable::RadiationImpedance => self.radiation_impedance.as_ref(),
                CalibrationObservable::MagneticBh => self.magnetic_bh.as_ref(),
                CalibrationObservable::Displacement => self.displacement.as_ref(),
                CalibrationObservable::VelocityTransfer => self.velocity_transfer.as_ref(),
                CalibrationObservable::MagneticFluxDensity => self.magnetic_flux_density.as_ref(),
            }
            .ok_or(CalibrationError::MissingRequiredObservable {
                observable: observable.as_str(),
            })?;
            validate_metric_report(metric, observable.as_str())?;
            if !metric.passed {
                return Err(CalibrationError::RequiredObservableFailed {
                    observable: observable.as_str(),
                });
            }
        }
        if !self.passed {
            return Err(CalibrationError::ReportFailed);
        }
        Ok(())
    }
}

/// Correlate measured fixture data against one validated typed driver model.
///
/// Measurements outside the model's tabulated domain are rejected instead of
/// being silently clamped.  A returned report with `passed == false` is valid
/// calibration evidence and is intentionally not converted into an error.
pub fn correlate_driver_model(
    model: &DriverModel,
    data: &DriverCalibrationData,
    tolerances: &CalibrationTolerances,
) -> Result<CalibrationReport, CalibrationError> {
    model.validate()?;
    data.validate()?;
    tolerances.validate()?;
    if data.is_empty() {
        return Err(CalibrationError::EmptyDataset);
    }

    let impedance = if data.impedance.is_empty() {
        None
    } else {
        let mut errors = Vec::with_capacity(data.impedance.len());
        for (index, measured) in data.impedance.iter().enumerate() {
            ensure_in_domain(
                "impedance",
                index,
                measured.frequency_hz,
                model.electrical_impedance[0].frequency_hz,
                model.electrical_impedance[model.electrical_impedance.len() - 1].frequency_hz,
            )?;
            let predicted = model.electrical_impedance_at(measured.frequency_hz);
            let measured = num_complex::Complex64::new(measured.real_ohm, measured.imaginary_ohm);
            errors.push(((predicted - measured).norm(), measured.norm()));
        }
        Some(metric_report(&errors, tolerances.impedance_ohm))
    };

    let bl_curve = if data.bl_curve.is_empty() {
        None
    } else {
        let mut errors = Vec::with_capacity(data.bl_curve.len());
        let first = model.bl_curve[0].displacement_m;
        let last = model.bl_curve[model.bl_curve.len() - 1].displacement_m;
        for (index, measured) in data.bl_curve.iter().enumerate() {
            ensure_in_domain("BL(x)", index, measured.displacement_m, first, last)?;
            let predicted = interpolate_bl(&model.bl_curve, measured.displacement_m);
            errors.push((
                predicted - measured.force_factor_tm,
                measured.force_factor_tm,
            ));
        }
        Some(metric_report(&errors, tolerances.bl_tm))
    };

    let (inductance, eddy_resistance) = if data.inductance_curve.is_empty() {
        (None, None)
    } else {
        let mut inductance_errors = Vec::with_capacity(data.inductance_curve.len());
        let mut resistance_errors = Vec::with_capacity(data.inductance_curve.len());
        let first = model.inductance_curve[0].frequency_hz;
        let last = model.inductance_curve[model.inductance_curve.len() - 1].frequency_hz;
        for (index, measured) in data.inductance_curve.iter().enumerate() {
            ensure_in_domain("Le(f)", index, measured.frequency_hz, first, last)?;
            let (predicted_inductance, predicted_resistance) =
                interpolate_inductance(&model.inductance_curve, measured.frequency_hz);
            inductance_errors.push((
                predicted_inductance - measured.inductance_h,
                measured.inductance_h,
            ));
            resistance_errors.push((
                predicted_resistance - measured.eddy_resistance_ohm,
                measured.eddy_resistance_ohm,
            ));
        }
        (
            Some(metric_report(&inductance_errors, tolerances.inductance_h)),
            Some(metric_report(
                &resistance_errors,
                tolerances.eddy_resistance_ohm,
            )),
        )
    };

    let (skin_depth, proximity_factor) = if data.eddy_current.is_empty() {
        (None, None)
    } else {
        let surface = calibrated_eddy_current_model(model)?;
        let mut skin_depth_errors = Vec::with_capacity(data.eddy_current.len());
        let mut proximity_errors = Vec::with_capacity(data.eddy_current.len());
        for (index, measured) in data.eddy_current.iter().enumerate() {
            ensure_eddy_current_in_domain(surface, index, measured)?;
            let predicted = surface.evaluate(measured.frequency_hz, measured.flux_density_t)?;
            skin_depth_errors.push((
                predicted.skin_depth_m - measured.skin_depth_m,
                measured.skin_depth_m,
            ));
            proximity_errors.push((
                predicted.proximity_factor - measured.proximity_factor,
                measured.proximity_factor,
            ));
        }
        (
            Some(metric_report(&skin_depth_errors, tolerances.skin_depth_m)),
            Some(metric_report(
                &proximity_errors,
                tolerances.proximity_factor,
            )),
        )
    };

    let thermal_response = if data.thermal_response.is_empty() {
        None
    } else {
        let thermal = model
            .thermal_network
            .as_ref()
            .ok_or(CalibrationError::MissingThermalNetwork)?;
        let mut errors =
            Vec::with_capacity(data.thermal_response.len() * crate::THERMAL_NODE_COUNT);
        for measured in &data.thermal_response {
            let power = thermal.power_from_losses(measured.copper_loss_w, measured.eddy_loss_w)?;
            let predicted = thermal.specification.solve_steady_state(power)?;
            for (predicted_temperature, measured_temperature) in predicted
                .temperatures_k
                .iter()
                .zip(measured.temperatures_k.iter())
            {
                errors.push((
                    predicted_temperature - measured_temperature,
                    *measured_temperature,
                ));
            }
        }
        Some(metric_report(&errors, tolerances.temperature_k))
    };

    let (radiation_velocity, radiation_impedance) = if data.radiation_response.is_empty() {
        (None, None)
    } else {
        let mut velocity_errors = Vec::with_capacity(data.radiation_response.len());
        let mut impedance_errors = Vec::with_capacity(data.radiation_response.len());
        let first = model.velocity_transfer[0].frequency_hz;
        let last = model.velocity_transfer[model.velocity_transfer.len() - 1].frequency_hz;
        for (index, measured) in data.radiation_response.iter().enumerate() {
            ensure_in_domain(
                "radiation response",
                index,
                measured.frequency_hz,
                first,
                last,
            )?;
            let radiation = num_complex::Complex64::new(
                measured.radiation_real_n_s_per_m,
                measured.radiation_imag_n_s_per_m,
            );
            let (predicted_velocity, predicted_impedance) =
                model.response_with_radiation_impedance(measured.frequency_hz, radiation)?;
            let measured_velocity = num_complex::Complex64::new(
                measured.velocity_real_m_per_s,
                measured.velocity_imag_m_per_s,
            );
            let measured_impedance = num_complex::Complex64::new(
                measured.electrical_real_ohm,
                measured.electrical_imag_ohm,
            );
            velocity_errors.push((
                (predicted_velocity - measured_velocity).norm(),
                measured_velocity.norm(),
            ));
            impedance_errors.push((
                (predicted_impedance - measured_impedance).norm(),
                measured_impedance.norm(),
            ));
        }
        (
            Some(metric_report(
                &velocity_errors,
                tolerances.radiation_velocity_m_per_s,
            )),
            Some(metric_report(
                &impedance_errors,
                tolerances.radiation_impedance_ohm,
            )),
        )
    };

    let magnetic_bh = if data.magnetic_bh.is_empty() {
        None
    } else {
        let law = calibrated_magnetic_law(model)?;
        let first = law.bh_curve[0].flux_density_t;
        let last = law.bh_curve[law.bh_curve.len() - 1].flux_density_t;
        let mut errors = Vec::with_capacity(data.magnetic_bh.len());
        for (index, measured) in data.magnetic_bh.iter().enumerate() {
            ensure_in_domain("B-H", index, measured.flux_density_t, first, last)?;
            let predicted = law.evaluate(0.0, measured.flux_density_t)?;
            errors.push((
                predicted.field_strength_a_per_m - measured.field_strength_a_per_m,
                measured.field_strength_a_per_m,
            ));
        }
        Some(metric_report(&errors, tolerances.magnetic_field_a_per_m))
    };

    let displacement = if data.displacement_transfer.is_empty() {
        None
    } else {
        let mut errors = Vec::with_capacity(data.displacement_transfer.len());
        let first = model.velocity_transfer[0].frequency_hz;
        let last = model.velocity_transfer[model.velocity_transfer.len() - 1].frequency_hz;
        for (index, measured) in data.displacement_transfer.iter().enumerate() {
            ensure_in_domain("displacement", index, measured.frequency_hz, first, last)?;
            let omega = std::f64::consts::TAU * measured.frequency_hz;
            let predicted =
                model.velocity_at(measured.frequency_hz) / num_complex::Complex64::new(0.0, omega);
            let measured = num_complex::Complex64::new(
                measured.displacement_real_m,
                measured.displacement_imag_m,
            );
            errors.push(((predicted - measured).norm(), measured.norm()));
        }
        Some(metric_report(&errors, tolerances.displacement_m))
    };

    let velocity_transfer = if data.velocity_transfer.is_empty() {
        None
    } else {
        let mut errors = Vec::with_capacity(data.velocity_transfer.len());
        let first = model.velocity_transfer[0].frequency_hz;
        let last = model.velocity_transfer[model.velocity_transfer.len() - 1].frequency_hz;
        for (index, measured) in data.velocity_transfer.iter().enumerate() {
            ensure_in_domain("laser velocity", index, measured.frequency_hz, first, last)?;
            let predicted = model.velocity_at(measured.frequency_hz);
            let measured = num_complex::Complex64::new(
                measured.velocity_real_m_per_s,
                measured.velocity_imag_m_per_s,
            );
            errors.push(((predicted - measured).norm(), measured.norm()));
        }
        Some(metric_report(&errors, tolerances.velocity_m_per_s))
    };

    let magnetic_flux_density = if data.magnetic_scan.is_empty() {
        None
    } else {
        let errors = data
            .magnetic_scan
            .iter()
            .map(|point| {
                (
                    point.predicted_flux_density_t - point.measured_flux_density_t,
                    point.measured_flux_density_t,
                )
            })
            .collect::<Vec<_>>();
        Some(metric_report(&errors, tolerances.magnetic_flux_density_t))
    };

    let passed = report_passed(&impedance)
        && report_passed(&bl_curve)
        && report_passed(&inductance)
        && report_passed(&eddy_resistance)
        && report_passed(&skin_depth)
        && report_passed(&proximity_factor)
        && report_passed(&thermal_response)
        && report_passed(&radiation_velocity)
        && report_passed(&radiation_impedance)
        && report_passed(&magnetic_bh)
        && report_passed(&displacement)
        && report_passed(&velocity_transfer)
        && report_passed(&magnetic_flux_density);
    Ok(CalibrationReport {
        origin: data.origin,
        independent_measurement: data.independent_measurement.clone(),
        impedance,
        bl_curve,
        inductance,
        eddy_resistance,
        skin_depth,
        proximity_factor,
        thermal_response,
        radiation_velocity,
        radiation_impedance,
        magnetic_bh,
        displacement,
        velocity_transfer,
        magnetic_flux_density,
        passed,
    })
}

fn calibrated_eddy_current_model(
    model: &DriverModel,
) -> Result<&crate::nonlinear_motor::EddyCurrentLossModel, CalibrationError> {
    model
        .nonlinear_motor
        .as_ref()
        .and_then(|motor| motor.eddy_current.as_ref())
        .ok_or(CalibrationError::MissingEddyCurrentModel)
}

fn calibrated_magnetic_law(
    model: &DriverModel,
) -> Result<&crate::nonlinear_motor::MotorMagneticLaw, CalibrationError> {
    model
        .nonlinear_motor
        .as_ref()
        .and_then(|motor| motor.magnetic.as_ref())
        .ok_or(CalibrationError::MissingMagneticModel)
}

fn ensure_eddy_current_in_domain(
    model: &crate::nonlinear_motor::EddyCurrentLossModel,
    index: usize,
    measured: &MeasuredEddyCurrentPoint,
) -> Result<(), CalibrationError> {
    let (minimum_frequency, maximum_frequency) =
        model.samples.iter().map(|sample| sample.frequency_hz).fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
    ensure_in_domain(
        "eddy current frequency",
        index,
        measured.frequency_hz,
        minimum_frequency,
        maximum_frequency,
    )?;

    let (minimum_flux, maximum_flux) = model
        .samples
        .iter()
        .map(|sample| sample.flux_density_t.abs())
        .fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
    if minimum_flux != maximum_flux {
        ensure_in_domain(
            "eddy current flux density",
            index,
            measured.flux_density_t.abs(),
            minimum_flux,
            maximum_flux,
        )?;
    }
    Ok(())
}

fn metric_report(errors: &[(f64, f64)], tolerance: f64) -> CalibrationMetricReport {
    debug_assert!(!errors.is_empty());
    let sum_squared_error = errors.iter().map(|(error, _)| error * error).sum::<f64>();
    let reference_sum_squared = errors
        .iter()
        .map(|(_, reference)| reference * reference)
        .sum::<f64>();
    let samples = errors.len();
    let rmse = (sum_squared_error / samples as f64).sqrt();
    let reference_rms = (reference_sum_squared / samples as f64).sqrt();
    let normalized_rmse = if reference_rms > f64::MIN_POSITIVE {
        rmse / reference_rms
    } else {
        rmse
    };
    let maximum_absolute_error = errors
        .iter()
        .map(|(error, _)| error.abs())
        .fold(0.0, f64::max);
    CalibrationMetricReport {
        samples,
        rmse,
        maximum_absolute_error,
        normalized_rmse,
        tolerance,
        passed: rmse <= tolerance,
    }
}

fn report_passed(report: &Option<CalibrationMetricReport>) -> bool {
    report.as_ref().is_none_or(|report| report.passed)
}

fn validate_metric_report(
    metric: &CalibrationMetricReport,
    observable: &'static str,
) -> Result<(), CalibrationError> {
    if metric.samples == 0
        || !metric.rmse.is_finite()
        || !metric.maximum_absolute_error.is_finite()
        || !metric.normalized_rmse.is_finite()
        || metric.normalized_rmse < 0.0
        || !metric.tolerance.is_finite()
        || metric.tolerance < 0.0
        || metric.maximum_absolute_error < 0.0
        || metric.rmse < 0.0
        || metric.passed != (metric.rmse <= metric.tolerance)
    {
        return Err(CalibrationError::InvalidMetricReport { observable });
    }
    Ok(())
}

fn validate_grid<I>(curve: &'static str, values: I) -> Result<(), CalibrationError>
where
    I: IntoIterator<Item = f64>,
{
    let mut previous = None;
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(CalibrationError::NonFiniteSample { curve, index });
        }
        if previous.is_some_and(|previous| value <= previous) {
            return Err(CalibrationError::UnorderedGrid { curve, index });
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_frequency_grid<I>(curve: &'static str, values: I) -> Result<(), CalibrationError>
where
    I: IntoIterator<Item = f64>,
{
    let mut previous = None;
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(CalibrationError::InvalidFrequency { curve, index });
        }
        if previous.is_some_and(|previous| value <= previous) {
            return Err(CalibrationError::UnorderedGrid { curve, index });
        }
        previous = Some(value);
    }
    Ok(())
}

fn ensure_in_domain(
    curve: &'static str,
    index: usize,
    value: f64,
    minimum: f64,
    maximum: f64,
) -> Result<(), CalibrationError> {
    if value < minimum || value > maximum {
        Err(CalibrationError::OutsideModelGrid {
            curve,
            index,
            value,
        })
    } else {
        Ok(())
    }
}

fn interpolate_bl(points: &[crate::driver_model::BlPoint], displacement_m: f64) -> f64 {
    if displacement_m <= points[0].displacement_m {
        return points[0].force_factor_tm;
    }
    if displacement_m >= points[points.len() - 1].displacement_m {
        return points[points.len() - 1].force_factor_tm;
    }
    let upper = points.partition_point(|point| point.displacement_m < displacement_m);
    let lower = upper - 1;
    let fraction = (displacement_m - points[lower].displacement_m)
        / (points[upper].displacement_m - points[lower].displacement_m);
    points[lower].force_factor_tm
        + fraction * (points[upper].force_factor_tm - points[lower].force_factor_tm)
}

fn interpolate_inductance(
    points: &[crate::driver_model::InductancePoint],
    frequency_hz: f64,
) -> (f64, f64) {
    if frequency_hz <= points[0].frequency_hz {
        return (points[0].inductance_h, points[0].eddy_resistance_ohm);
    }
    if frequency_hz >= points[points.len() - 1].frequency_hz {
        let point = points[points.len() - 1];
        return (point.inductance_h, point.eddy_resistance_ohm);
    }
    let upper = points.partition_point(|point| point.frequency_hz < frequency_hz);
    let lower = upper - 1;
    let fraction = (frequency_hz - points[lower].frequency_hz)
        / (points[upper].frequency_hz - points[lower].frequency_hz);
    (
        points[lower].inductance_h
            + fraction * (points[upper].inductance_h - points[lower].inductance_h),
        points[lower].eddy_resistance_ohm
            + fraction * (points[upper].eddy_resistance_ohm - points[lower].eddy_resistance_ohm),
    )
}

/// Errors that prevent a meaningful calibration report from being produced.
#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error(transparent)]
    DriverModel(#[from] DriverModelError),
    #[error(transparent)]
    Modeling(#[from] ModelingError),
    #[error("calibration dataset must contain at least one observable")]
    EmptyDataset,
    #[error("skin-depth/proximity measurements require a calibrated eddy-current surface")]
    MissingEddyCurrentModel,
    #[error("B-H measurements require a calibrated magnetic law")]
    MissingMagneticModel,
    #[error("thermal measurements require an attached driver thermal network")]
    MissingThermalNetwork,
    #[error("{curve} sample {index} contains a non-finite value")]
    NonFiniteSample { curve: &'static str, index: usize },
    #[error("{curve} sample {index} is physically invalid")]
    InvalidSample { curve: &'static str, index: usize },
    #[error("{curve} sample {index} has a non-positive frequency")]
    InvalidFrequency { curve: &'static str, index: usize },
    #[error("{curve} sample grid is not strictly ordered at index {index}")]
    UnorderedGrid { curve: &'static str, index: usize },
    #[error("{curve} sample {index} at {value} is outside the model grid")]
    OutsideModelGrid {
        curve: &'static str,
        index: usize,
        value: f64,
    },
    #[error("{curve} tolerance must be finite and non-negative")]
    InvalidTolerance { curve: &'static str },
    #[error("independent measurement metadata is required for independent calibration")]
    IndependentMetadataRequired,
    #[error("fixture calibration data must not carry independent measurement metadata")]
    UnexpectedIndependentMetadata,
    #[error("independent measurement metadata field {field} must not be blank")]
    InvalidIndependentMetadata { field: &'static str },
    #[error("independent measurement calibration report is required for this promotion")]
    IndependentReportRequired,
    #[error("promotion policy lists required observable {observable} more than once")]
    DuplicateRequiredObservable { observable: &'static str },
    #[error("calibration report is missing required observable {observable}")]
    MissingRequiredObservable { observable: &'static str },
    #[error("calibration report metric for {observable} is malformed")]
    InvalidMetricReport { observable: &'static str },
    #[error("calibration report required observable {observable} did not pass")]
    RequiredObservableFailed { observable: &'static str },
    #[error("calibration report did not pass all correlated observables")]
    ReportFailed,
    #[error("calibration report passed flag disagrees with its metric gates")]
    InconsistentReportStatus,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DriverThermalNetworkConfig, EddyCurrentLossModel, EddyCurrentLossSample,
        HysteresisLossModel, MotorConstitutiveModel, MotorMagneticBhPoint, MotorMagneticLaw,
        MotorTemperatureLaw, NonlinearBlPoint, NonlinearLePoint, SpeakerDriverStudy,
    };

    fn model() -> DriverModel {
        let mut model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let eddy_current = EddyCurrentLossModel::from_samples(&[
            EddyCurrentLossSample {
                frequency_hz: 100.0,
                flux_density_t: 0.5,
                eddy_resistance_ohm: 0.1,
                skin_depth_m: 0.01,
                proximity_factor: 1.1,
            },
            EddyCurrentLossSample {
                frequency_hz: 1_000.0,
                flux_density_t: 0.5,
                eddy_resistance_ohm: 0.2,
                skin_depth_m: 0.005,
                proximity_factor: 1.2,
            },
        ])
        .unwrap();
        let magnetic = MotorMagneticLaw {
            bh_curve: vec![
                MotorMagneticBhPoint {
                    flux_density_t: 0.0,
                    field_strength_a_per_m: 0.0,
                },
                MotorMagneticBhPoint {
                    flux_density_t: 0.5,
                    field_strength_a_per_m: 400.0,
                },
                MotorMagneticBhPoint {
                    flux_density_t: 1.0,
                    field_strength_a_per_m: 1_200.0,
                },
            ],
            active_volume_m3: 1.0e-5,
            flux_modulation_t_per_a: 0.0,
            flux_modulation_curve: Vec::new(),
            reluctance: None,
        };
        magnetic.validate().unwrap();
        model.nonlinear_motor = Some(MotorConstitutiveModel {
            bl_curve: model
                .bl_curve
                .iter()
                .map(|point| NonlinearBlPoint {
                    displacement_m: point.displacement_m,
                    force_factor_tm: point.force_factor_tm,
                })
                .collect(),
            le_surface: model
                .inductance_curve
                .iter()
                .map(|point| NonlinearLePoint {
                    displacement_m: 0.0,
                    current_a: 0.0,
                    frequency_hz: point.frequency_hz,
                    inductance_h: point.inductance_h,
                    eddy_resistance_ohm: point.eddy_resistance_ohm,
                })
                .collect(),
            temperature: MotorTemperatureLaw {
                copper_resistance_ohm: model.thiele_small.voice_coil_resistance_ohm,
                ..MotorTemperatureLaw::default()
            },
            hysteresis: HysteresisLossModel::default(),
            magnetic: Some(magnetic),
            eddy_current: Some(eddy_current),
        });
        model.thermal_network = Some(DriverThermalNetworkConfig::default());
        model.validate().unwrap();
        model
    }

    fn tolerances() -> CalibrationTolerances {
        CalibrationTolerances {
            impedance_ohm: 1.0e-9,
            bl_tm: 1.0e-9,
            inductance_h: 1.0e-12,
            eddy_resistance_ohm: 1.0e-9,
            skin_depth_m: 1.0e-12,
            proximity_factor: 1.0e-12,
            temperature_k: 1.0e-12,
            radiation_velocity_m_per_s: 1.0e-12,
            radiation_impedance_ohm: 1.0e-12,
            magnetic_field_a_per_m: 1.0e-9,
            displacement_m: 1.0e-12,
            velocity_m_per_s: 1.0e-12,
            magnetic_flux_density_t: 1.0e-9,
        }
    }

    #[test]
    fn exact_fixture_curves_pass_calibration() {
        let model = model();
        let radiation = num_complex::Complex64::new(1.0, 0.5);
        let (radiation_velocity, radiation_impedance) = model
            .response_with_radiation_impedance(model.velocity_transfer[0].frequency_hz, radiation)
            .unwrap();
        let data = DriverCalibrationData {
            origin: CalibrationDataOrigin::Fixture,
            independent_measurement: None,
            impedance: model
                .electrical_impedance
                .iter()
                .map(|point| MeasuredImpedancePoint {
                    frequency_hz: point.frequency_hz,
                    real_ohm: point.real,
                    imaginary_ohm: point.imaginary,
                })
                .collect(),
            bl_curve: model
                .bl_curve
                .iter()
                .map(|point| MeasuredBlPoint {
                    displacement_m: point.displacement_m,
                    force_factor_tm: point.force_factor_tm,
                })
                .collect(),
            inductance_curve: model
                .inductance_curve
                .iter()
                .map(|point| MeasuredInductancePoint {
                    frequency_hz: point.frequency_hz,
                    inductance_h: point.inductance_h,
                    eddy_resistance_ohm: point.eddy_resistance_ohm,
                })
                .collect(),
            eddy_current: model
                .nonlinear_motor
                .as_ref()
                .unwrap()
                .eddy_current
                .as_ref()
                .unwrap()
                .samples
                .iter()
                .map(|point| MeasuredEddyCurrentPoint {
                    frequency_hz: point.frequency_hz,
                    flux_density_t: point.flux_density_t,
                    skin_depth_m: point.skin_depth_m,
                    proximity_factor: point.proximity_factor,
                })
                .collect(),
            thermal_response: vec![MeasuredThermalPoint {
                copper_loss_w: 0.0,
                eddy_loss_w: 0.0,
                temperatures_k: [293.15; crate::THERMAL_NODE_COUNT],
            }],
            radiation_response: vec![MeasuredRadiationResponsePoint {
                frequency_hz: model.velocity_transfer[0].frequency_hz,
                radiation_real_n_s_per_m: radiation.re,
                radiation_imag_n_s_per_m: radiation.im,
                velocity_real_m_per_s: radiation_velocity.re,
                velocity_imag_m_per_s: radiation_velocity.im,
                electrical_real_ohm: radiation_impedance.re,
                electrical_imag_ohm: radiation_impedance.im,
            }],
            magnetic_bh: model
                .nonlinear_motor
                .as_ref()
                .unwrap()
                .magnetic
                .as_ref()
                .unwrap()
                .bh_curve
                .iter()
                .map(|point| MeasuredMagneticBhPoint {
                    flux_density_t: point.flux_density_t,
                    field_strength_a_per_m: point.field_strength_a_per_m,
                })
                .collect(),
            displacement_transfer: model
                .velocity_transfer
                .iter()
                .map(|point| {
                    let displacement = num_complex::Complex64::new(
                        point.velocity_real_m_per_s,
                        point.velocity_imag_m_per_s,
                    ) / num_complex::Complex64::new(
                        0.0,
                        std::f64::consts::TAU * point.frequency_hz,
                    );
                    MeasuredDisplacementPoint {
                        frequency_hz: point.frequency_hz,
                        displacement_real_m: displacement.re,
                        displacement_imag_m: displacement.im,
                    }
                })
                .collect(),
            velocity_transfer: model
                .velocity_transfer
                .iter()
                .map(|point| MeasuredVelocityPoint {
                    frequency_hz: point.frequency_hz,
                    velocity_real_m_per_s: point.velocity_real_m_per_s,
                    velocity_imag_m_per_s: point.velocity_imag_m_per_s,
                })
                .collect(),
            magnetic_scan: vec![MagneticScanPoint {
                position_m: [0.0, 0.01, 0.02],
                measured_flux_density_t: 0.8,
                predicted_flux_density_t: 0.8,
            }],
        };
        let report = correlate_driver_model(&model, &data, &tolerances()).unwrap();
        assert!(report.passed);
        assert_eq!(report.origin, CalibrationDataOrigin::Fixture);
        assert_eq!(
            report.impedance.unwrap().samples,
            model.electrical_impedance.len()
        );
        assert_eq!(
            report.inductance.unwrap().samples,
            model.inductance_curve.len()
        );
        assert_eq!(report.radiation_velocity.unwrap().samples, 1);
        assert_eq!(report.radiation_impedance.unwrap().samples, 1);
        assert_eq!(report.magnetic_bh.unwrap().samples, 3);
        assert_eq!(
            report.displacement.unwrap().samples,
            model.velocity_transfer.len()
        );
        assert_eq!(report.magnetic_flux_density.unwrap().rmse, 0.0);

        let mut independent_data = data.clone();
        independent_data.origin = CalibrationDataOrigin::IndependentMeasurement;
        independent_data.independent_measurement = Some(IndependentMeasurementMetadata {
            dataset_id: "qa-independent-driver-001".to_owned(),
            source: "analytical reference implementation".to_owned(),
            acquisition_method: "independent frequency-domain evaluation".to_owned(),
        });
        let independent_report =
            correlate_driver_model(&model, &independent_data, &tolerances()).unwrap();
        assert_eq!(
            independent_report.origin,
            CalibrationDataOrigin::IndependentMeasurement
        );
        assert_eq!(
            independent_report
                .independent_measurement
                .as_ref()
                .unwrap()
                .dataset_id,
            "qa-independent-driver-001"
        );
        independent_report
            .require_promotion_coverage(&CalibrationCoveragePolicy::full_independent())
            .unwrap();

        let mut relabeled_fixture = data;
        relabeled_fixture.origin = CalibrationDataOrigin::IndependentMeasurement;
        assert!(matches!(
            correlate_driver_model(&model, &relabeled_fixture, &tolerances()),
            Err(CalibrationError::IndependentMetadataRequired)
        ));
    }

    #[test]
    fn report_fails_for_measurement_mismatch_without_mutating_model() {
        let model = model();
        let original = model.clone();
        let mut data = DriverCalibrationData {
            impedance: vec![MeasuredImpedancePoint {
                frequency_hz: model.electrical_impedance[1].frequency_hz,
                real_ohm: model.electrical_impedance[1].real + 1.0,
                imaginary_ohm: model.electrical_impedance[1].imaginary,
            }],
            ..DriverCalibrationData::default()
        };
        let report = correlate_driver_model(&model, &data, &tolerances()).unwrap();
        assert!(!report.passed);
        assert!(!report.impedance.unwrap().passed);
        assert_eq!(model, original);

        data.impedance[0].frequency_hz = 0.5 * model.electrical_impedance[0].frequency_hz;
        assert!(matches!(
            correlate_driver_model(&model, &data, &tolerances()),
            Err(CalibrationError::OutsideModelGrid { .. })
        ));
    }

    #[test]
    fn invalid_fixture_grid_and_tolerance_are_rejected() {
        let data = DriverCalibrationData {
            impedance: vec![
                MeasuredImpedancePoint {
                    frequency_hz: 200.0,
                    real_ohm: 1.0,
                    imaginary_ohm: 0.0,
                },
                MeasuredImpedancePoint {
                    frequency_hz: 100.0,
                    real_ohm: 1.0,
                    imaginary_ohm: 0.0,
                },
            ],
            ..DriverCalibrationData::default()
        };
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::UnorderedGrid { .. })
        ));

        let mut tolerance = tolerances();
        tolerance.velocity_m_per_s = f64::NAN;
        assert!(matches!(
            tolerance.validate(),
            Err(CalibrationError::InvalidTolerance {
                curve: "laser velocity"
            })
        ));
    }

    #[test]
    fn eddy_current_diagnostics_correlate_and_respect_surface_support() {
        let model = model();
        let measured = MeasuredEddyCurrentPoint {
            frequency_hz: 100.0,
            flux_density_t: 0.5,
            skin_depth_m: 0.01,
            proximity_factor: 1.1,
        };
        let data = DriverCalibrationData {
            eddy_current: vec![measured],
            ..DriverCalibrationData::default()
        };
        let report = correlate_driver_model(&model, &data, &tolerances()).unwrap();
        assert!(report.passed);
        assert_eq!(report.skin_depth.unwrap().samples, 1);
        assert_eq!(report.proximity_factor.unwrap().samples, 1);

        let mut outside = data.clone();
        outside.eddy_current[0].frequency_hz = 50.0;
        assert!(matches!(
            correlate_driver_model(&model, &outside, &tolerances()),
            Err(CalibrationError::OutsideModelGrid {
                curve: "eddy current frequency",
                ..
            })
        ));

        let mut without_surface = model.clone();
        without_surface
            .nonlinear_motor
            .as_mut()
            .unwrap()
            .eddy_current = None;
        assert!(matches!(
            correlate_driver_model(&without_surface, &data, &tolerances()),
            Err(CalibrationError::MissingEddyCurrentModel)
        ));
    }

    #[test]
    fn eddy_current_measurements_validate_physical_values() {
        let mut data = DriverCalibrationData {
            eddy_current: vec![MeasuredEddyCurrentPoint {
                frequency_hz: 100.0,
                flux_density_t: 0.5,
                skin_depth_m: 0.01,
                proximity_factor: 1.1,
            }],
            ..DriverCalibrationData::default()
        };
        data.eddy_current[0].skin_depth_m = f64::NAN;
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::NonFiniteSample {
                curve: "eddy current",
                ..
            })
        ));
        data.eddy_current[0].skin_depth_m = 0.0;
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::InvalidSample {
                curve: "eddy current",
                ..
            })
        ));
    }

    #[test]
    fn thermal_response_correlation_uses_all_four_network_nodes() {
        let mut model = model();
        let thermal = DriverThermalNetworkConfig::default();
        let power = thermal.power_from_losses(2.0, 1.0).unwrap();
        let expected = thermal.specification.solve_steady_state(power).unwrap();
        model.thermal_network = Some(thermal);

        let data = DriverCalibrationData {
            thermal_response: vec![MeasuredThermalPoint {
                copper_loss_w: 2.0,
                eddy_loss_w: 1.0,
                temperatures_k: expected.temperatures_k,
            }],
            ..DriverCalibrationData::default()
        };
        let report = correlate_driver_model(&model, &data, &tolerances()).unwrap();
        assert!(report.passed);
        assert_eq!(
            report.thermal_response.unwrap().samples,
            crate::THERMAL_NODE_COUNT
        );

        let mut mismatch = data.clone();
        mismatch.thermal_response[0].temperatures_k[crate::ThermalNode::Magnet as usize] += 1.0;
        let report = correlate_driver_model(&model, &mismatch, &tolerances()).unwrap();
        assert!(!report.thermal_response.unwrap().passed);

        model.thermal_network = None;
        assert!(matches!(
            correlate_driver_model(&model, &data, &tolerances()),
            Err(CalibrationError::MissingThermalNetwork)
        ));
    }

    #[test]
    fn thermal_response_measurements_reject_nonphysical_values() {
        let mut data = DriverCalibrationData {
            thermal_response: vec![MeasuredThermalPoint {
                copper_loss_w: 1.0,
                eddy_loss_w: 0.5,
                temperatures_k: [293.15; crate::THERMAL_NODE_COUNT],
            }],
            ..DriverCalibrationData::default()
        };
        data.thermal_response[0].temperatures_k[0] = 0.0;
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::InvalidSample {
                curve: "thermal response",
                ..
            })
        ));

        data.thermal_response[0].temperatures_k[0] = f64::NAN;
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::NonFiniteSample {
                curve: "thermal response",
                ..
            })
        ));
    }

    #[test]
    fn radiation_response_correlation_checks_velocity_impedance_and_support() {
        let model = model();
        let frequency_hz = model.velocity_transfer[1].frequency_hz;
        let radiation = num_complex::Complex64::new(1.0, -0.25);
        let (velocity, impedance) = model
            .response_with_radiation_impedance(frequency_hz, radiation)
            .unwrap();
        let data = DriverCalibrationData {
            radiation_response: vec![MeasuredRadiationResponsePoint {
                frequency_hz,
                radiation_real_n_s_per_m: radiation.re,
                radiation_imag_n_s_per_m: radiation.im,
                velocity_real_m_per_s: velocity.re,
                velocity_imag_m_per_s: velocity.im,
                electrical_real_ohm: impedance.re,
                electrical_imag_ohm: impedance.im,
            }],
            ..DriverCalibrationData::default()
        };
        let report = correlate_driver_model(&model, &data, &tolerances()).unwrap();
        assert!(report.passed);
        assert_eq!(report.radiation_velocity.unwrap().samples, 1);
        assert_eq!(report.radiation_impedance.unwrap().samples, 1);

        let mut mismatch = data.clone();
        mismatch.radiation_response[0].velocity_real_m_per_s += 1.0e-2;
        let report = correlate_driver_model(&model, &mismatch, &tolerances()).unwrap();
        assert!(!report.radiation_velocity.unwrap().passed);
        assert!(report.radiation_impedance.unwrap().passed);

        mismatch.radiation_response[0].frequency_hz = 0.5;
        assert!(matches!(
            correlate_driver_model(&model, &mismatch, &tolerances()),
            Err(CalibrationError::OutsideModelGrid {
                curve: "radiation response",
                ..
            })
        ));

        mismatch.radiation_response[0].frequency_hz = frequency_hz;
        mismatch.radiation_response[0].radiation_real_n_s_per_m = -1.0;
        assert!(matches!(
            mismatch.validate(),
            Err(CalibrationError::InvalidSample {
                curve: "radiation response",
                ..
            })
        ));
    }

    #[test]
    fn magnetic_bh_correlation_requires_law_and_support() {
        let model = model();
        let data = DriverCalibrationData {
            magnetic_bh: vec![
                MeasuredMagneticBhPoint {
                    flux_density_t: 0.0,
                    field_strength_a_per_m: 0.0,
                },
                MeasuredMagneticBhPoint {
                    flux_density_t: 0.5,
                    field_strength_a_per_m: 400.0,
                },
                MeasuredMagneticBhPoint {
                    flux_density_t: 1.0,
                    field_strength_a_per_m: 1_200.0,
                },
            ],
            ..DriverCalibrationData::default()
        };
        let report = correlate_driver_model(&model, &data, &tolerances()).unwrap();
        assert!(report.passed);
        assert_eq!(report.magnetic_bh.unwrap().samples, 3);

        let mut outside = data.clone();
        outside.magnetic_bh[2].flux_density_t = 1.1;
        assert!(matches!(
            correlate_driver_model(&model, &outside, &tolerances()),
            Err(CalibrationError::OutsideModelGrid { curve: "B-H", .. })
        ));

        let mut without_law = model.clone();
        without_law.nonlinear_motor.as_mut().unwrap().magnetic = None;
        assert!(matches!(
            correlate_driver_model(&without_law, &data, &tolerances()),
            Err(CalibrationError::MissingMagneticModel)
        ));

        let invalid = DriverCalibrationData {
            magnetic_bh: vec![MeasuredMagneticBhPoint {
                flux_density_t: 0.5,
                field_strength_a_per_m: 400.0,
            }],
            ..DriverCalibrationData::default()
        };
        assert!(matches!(
            invalid.validate(),
            Err(CalibrationError::InvalidSample { curve: "B-H", .. })
        ));
    }

    #[test]
    fn displacement_correlation_uses_velocity_transfer_convention() {
        let model = model();
        let point = model.velocity_transfer[1];
        let velocity =
            num_complex::Complex64::new(point.velocity_real_m_per_s, point.velocity_imag_m_per_s);
        let displacement =
            velocity / num_complex::Complex64::new(0.0, std::f64::consts::TAU * point.frequency_hz);
        let data = DriverCalibrationData {
            displacement_transfer: vec![MeasuredDisplacementPoint {
                frequency_hz: point.frequency_hz,
                displacement_real_m: displacement.re,
                displacement_imag_m: displacement.im,
            }],
            ..DriverCalibrationData::default()
        };
        let report = correlate_driver_model(&model, &data, &tolerances()).unwrap();
        assert!(report.passed);

        let mut mismatch = data.clone();
        mismatch.displacement_transfer[0].displacement_real_m += 1.0e-3;
        let report = correlate_driver_model(&model, &mismatch, &tolerances()).unwrap();
        assert!(!report.displacement.unwrap().passed);

        mismatch.displacement_transfer[0].frequency_hz = 0.5;
        assert!(matches!(
            correlate_driver_model(&model, &mismatch, &tolerances()),
            Err(CalibrationError::OutsideModelGrid {
                curve: "displacement",
                ..
            })
        ));
    }

    #[test]
    fn legacy_calibration_json_defaults_to_fixture_origin() {
        let data: DriverCalibrationData = serde_json::from_str("{}").unwrap();
        assert_eq!(data.origin, CalibrationDataOrigin::Fixture);

        let report: CalibrationReport = serde_json::from_str(r#"{"passed":false}"#).unwrap();
        assert_eq!(report.origin, CalibrationDataOrigin::Fixture);

        let legacy_tolerances: CalibrationTolerances = serde_json::from_str(
            r#"{
                "impedance_ohm": 0.1,
                "bl_tm": 0.1,
                "inductance_h": 0.000001,
                "eddy_resistance_ohm": 0.1,
                "velocity_m_per_s": 0.001,
                "magnetic_flux_density_t": 0.01
            }"#,
        )
        .unwrap();
        assert_eq!(
            legacy_tolerances.magnetic_field_a_per_m,
            default_magnetic_field_tolerance()
        );
        assert_eq!(
            legacy_tolerances.temperature_k,
            default_temperature_tolerance()
        );
        assert_eq!(
            legacy_tolerances.radiation_velocity_m_per_s,
            default_radiation_velocity_tolerance()
        );
    }

    #[test]
    fn fixture_and_independent_metadata_cannot_be_mixed() {
        let mut data = DriverCalibrationData {
            impedance: vec![MeasuredImpedancePoint {
                frequency_hz: 100.0,
                real_ohm: 1.0,
                imaginary_ohm: 0.0,
            }],
            independent_measurement: Some(IndependentMeasurementMetadata {
                dataset_id: "independent".to_owned(),
                source: "test".to_owned(),
                acquisition_method: "test".to_owned(),
            }),
            ..DriverCalibrationData::default()
        };
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::UnexpectedIndependentMetadata)
        ));

        data.origin = CalibrationDataOrigin::IndependentMeasurement;
        data.independent_measurement
            .as_mut()
            .unwrap()
            .source
            .clear();
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::InvalidIndependentMetadata { field: "source" })
        ));
    }

    #[test]
    fn promotion_coverage_rejects_partial_failed_and_malformed_reports() {
        let passing_metric = CalibrationMetricReport {
            samples: 2,
            rmse: 0.01,
            maximum_absolute_error: 0.02,
            normalized_rmse: 0.01,
            tolerance: 0.02,
            passed: true,
        };
        let partial_fixture = CalibrationReport {
            origin: CalibrationDataOrigin::Fixture,
            independent_measurement: None,
            impedance: Some(passing_metric),
            bl_curve: None,
            inductance: None,
            eddy_resistance: None,
            skin_depth: None,
            proximity_factor: None,
            thermal_response: None,
            radiation_velocity: None,
            radiation_impedance: None,
            magnetic_bh: None,
            displacement: None,
            velocity_transfer: None,
            magnetic_flux_density: None,
            passed: true,
        };
        let fixture_impedance_only = CalibrationCoveragePolicy {
            required_observables: vec![CalibrationObservable::Impedance],
            require_independent_measurement: false,
        };
        partial_fixture
            .require_promotion_coverage(&fixture_impedance_only)
            .unwrap();
        assert!(matches!(
            partial_fixture
                .require_promotion_coverage(&CalibrationCoveragePolicy::full_independent()),
            Err(CalibrationError::IndependentReportRequired)
        ));

        let failed = CalibrationReport {
            impedance: Some(CalibrationMetricReport {
                rmse: 0.03,
                passed: false,
                ..passing_metric
            }),
            passed: false,
            ..partial_fixture.clone()
        };
        assert!(matches!(
            failed.require_promotion_coverage(&fixture_impedance_only),
            Err(CalibrationError::RequiredObservableFailed {
                observable: "impedance"
            })
        ));

        let malformed = CalibrationReport {
            impedance: Some(CalibrationMetricReport {
                samples: 0,
                ..passing_metric
            }),
            ..partial_fixture.clone()
        };
        assert!(matches!(
            malformed.require_promotion_coverage(&fixture_impedance_only),
            Err(CalibrationError::InvalidMetricReport {
                observable: "impedance"
            })
        ));

        let negative_normalized_rmse = CalibrationReport {
            impedance: Some(CalibrationMetricReport {
                normalized_rmse: -0.01,
                ..passing_metric
            }),
            ..partial_fixture.clone()
        };
        assert!(matches!(
            negative_normalized_rmse.require_promotion_coverage(&fixture_impedance_only),
            Err(CalibrationError::InvalidMetricReport {
                observable: "impedance"
            })
        ));

        let inconsistent = CalibrationReport {
            impedance: Some(CalibrationMetricReport {
                rmse: 0.03,
                passed: false,
                ..passing_metric
            }),
            passed: true,
            ..partial_fixture.clone()
        };
        assert!(matches!(
            inconsistent.require_promotion_coverage(&fixture_impedance_only),
            Err(CalibrationError::InconsistentReportStatus)
        ));

        let duplicate = CalibrationCoveragePolicy {
            required_observables: vec![
                CalibrationObservable::Impedance,
                CalibrationObservable::Impedance,
            ],
            require_independent_measurement: false,
        };
        assert!(matches!(
            partial_fixture.require_promotion_coverage(&duplicate),
            Err(CalibrationError::DuplicateRequiredObservable {
                observable: "impedance"
            })
        ));
    }
}
