//! Bias- and operating-point-dependent motor constitutive laws.
//!
//! The ordinary [`DriverModel`](crate::DriverModel) is intentionally a linear
//! handoff.  This module is the explicit nonlinear extension used by large
//! signal and calibration workflows.  It keeps the three independent inputs
//! to a motor model visible: BL(x), Le(x, i, f), and temperature-dependent
//! material laws.  Loss is reported separately from reactive inductance so it
//! cannot accidentally be counted twice in an electrical impedance.

use serde::{Deserialize, Serialize};

use crate::{ModelingError, require_finite, require_nonnegative, require_positive};

/// One measured or extracted force-factor sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NonlinearBlPoint {
    pub displacement_m: f64,
    pub force_factor_tm: f64,
}

/// One sample of the position/current/frequency-dependent coil model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NonlinearLePoint {
    pub displacement_m: f64,
    pub current_a: f64,
    pub frequency_hz: f64,
    pub inductance_h: f64,
    pub eddy_resistance_ohm: f64,
}

/// Operating point at which the nonlinear motor is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotorOperatingPoint {
    pub displacement_m: f64,
    pub current_a: f64,
    pub frequency_hz: f64,
    pub temperature_k: f64,
    pub flux_density_t: f64,
}

/// One first-quadrant sample of the motor's retained magnetic B-H law.
///
/// The curve is intentionally speaker-owned and serializable.  The FEM
/// solver's `BhCurve` remains an execution primitive, while this type is the
/// stable constitutive handoff that can be checked when a driver artifact is
/// loaded again.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotorMagneticBhPoint {
    pub flux_density_t: f64,
    pub field_strength_a_per_m: f64,
}

/// One signed current-sweep sample of the flux change relative to zero
/// current. These samples are retained as identification evidence and are
/// also used by [`MotorMagneticLaw::evaluate`] when present.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotorMagneticFluxModulationPoint {
    pub current_a: f64,
    pub delta_flux_density_t: f64,
}

/// Geometry needed for a signed reluctance-force estimate from gap energy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotorReluctanceGeometry {
    pub effective_gap_area_m2: f64,
    pub magnetic_gap_length_m: f64,
    pub gap_length_gradient_m_per_m: f64,
}

impl MotorReluctanceGeometry {
    fn validate(self) -> Result<(), ModelingError> {
        require_positive(
            "motor_magnetic.reluctance.effective_gap_area_m2",
            self.effective_gap_area_m2,
        )?;
        require_positive(
            "motor_magnetic.reluctance.magnetic_gap_length_m",
            self.magnetic_gap_length_m,
        )?;
        require_finite(
            "motor_magnetic.reluctance.gap_length_gradient_m_per_m",
            self.gap_length_gradient_m_per_m,
        )
    }
}

/// Serializable nonlinear magnetic law used by operating-point evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotorMagneticLaw {
    pub bh_curve: Vec<MotorMagneticBhPoint>,
    pub active_volume_m3: f64,
    pub flux_modulation_t_per_a: f64,
    /// Optional FEM current-sweep samples relative to the zero-current
    /// operating point. Empty means the scalar modulation fallback is used.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flux_modulation_curve: Vec<MotorMagneticFluxModulationPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reluctance: Option<MotorReluctanceGeometry>,
}

/// Magnetic quantities retained alongside the electrical operating values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotorMagneticOperatingValues {
    pub effective_flux_density_t: f64,
    pub field_strength_a_per_m: f64,
    pub permeability_h_per_m: f64,
    pub differential_reluctivity_m_per_h: f64,
    pub differential_permeability_h_per_m: f64,
    pub energy_density_j_per_m3: f64,
    pub coenergy_density_j_per_m3: f64,
    pub magnetic_energy_j: f64,
    pub magnetic_coenergy_j: f64,
    pub reluctance_force_n: f64,
}

impl MotorMagneticLaw {
    /// Build the serializable handoff from validated FEM-style `(B, H)`
    /// samples.  Keeping this conversion here prevents callers from
    /// bypassing the same monotonicity and passivity checks used at load time.
    pub fn from_bh_samples(
        samples: &[(f64, f64)],
        active_volume_m3: f64,
        flux_modulation_t_per_a: f64,
        reluctance: Option<MotorReluctanceGeometry>,
    ) -> Result<Self, ModelingError> {
        let law = Self {
            bh_curve: samples
                .iter()
                .map(
                    |&(flux_density_t, field_strength_a_per_m)| MotorMagneticBhPoint {
                        flux_density_t,
                        field_strength_a_per_m,
                    },
                )
                .collect(),
            active_volume_m3,
            flux_modulation_t_per_a,
            flux_modulation_curve: Vec::new(),
            reluctance,
        };
        law.validate()?;
        Ok(law)
    }

    /// Build a magnetic handoff with validated signed FEM current-sweep
    /// evidence. The scalar modulation remains as a compact fitted summary,
    /// while evaluation uses the retained curve for in-range currents.
    pub fn from_bh_samples_with_flux_modulation_curve(
        samples: &[(f64, f64)],
        active_volume_m3: f64,
        flux_modulation_t_per_a: f64,
        flux_modulation_curve: Vec<MotorMagneticFluxModulationPoint>,
        reluctance: Option<MotorReluctanceGeometry>,
    ) -> Result<Self, ModelingError> {
        let law = Self {
            bh_curve: samples
                .iter()
                .map(
                    |&(flux_density_t, field_strength_a_per_m)| MotorMagneticBhPoint {
                        flux_density_t,
                        field_strength_a_per_m,
                    },
                )
                .collect(),
            active_volume_m3,
            flux_modulation_t_per_a,
            flux_modulation_curve,
            reluctance,
        };
        law.validate()?;
        Ok(law)
    }

    /// Validate the monotone B-H samples and bounded passive parameters.
    pub fn validate(&self) -> Result<(), ModelingError> {
        if self.bh_curve.len() < 2 {
            return Err(ModelingError::OutOfRange {
                field: "motor_magnetic.bh_curve",
                constraint: "at least two samples",
            });
        }
        if self.bh_curve[0].flux_density_t != 0.0 || self.bh_curve[0].field_strength_a_per_m != 0.0
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_magnetic.bh_curve",
                constraint: "must start at the origin",
            });
        }
        for (index, point) in self.bh_curve.iter().enumerate() {
            if !point.flux_density_t.is_finite()
                || !point.field_strength_a_per_m.is_finite()
                || point.flux_density_t < 0.0
                || point.field_strength_a_per_m < 0.0
            {
                return Err(ModelingError::OutOfRange {
                    field: "motor_magnetic.bh_curve",
                    constraint: "finite first-quadrant samples",
                });
            }
            if index > 0 {
                let previous = self.bh_curve[index - 1];
                if point.flux_density_t <= previous.flux_density_t
                    || point.field_strength_a_per_m <= previous.field_strength_a_per_m
                {
                    return Err(ModelingError::OutOfRange {
                        field: "motor_magnetic.bh_curve",
                        constraint: "strictly increasing B and H",
                    });
                }
            }
        }
        if (0..self.bh_curve.len() - 1).any(|segment| {
            let lower = self.bh_curve[segment];
            let upper = self.bh_curve[segment + 1];
            let slope = (upper.field_strength_a_per_m - lower.field_strength_a_per_m)
                / (upper.flux_density_t - lower.flux_density_t);
            !slope.is_finite() || slope <= 0.0
        }) {
            return Err(ModelingError::OutOfRange {
                field: "motor_magnetic.bh_curve",
                constraint: "finite positive differential reluctivity",
            });
        }
        require_positive("motor_magnetic.active_volume_m3", self.active_volume_m3)?;
        require_finite(
            "motor_magnetic.flux_modulation_t_per_a",
            self.flux_modulation_t_per_a,
        )?;
        if !self.flux_modulation_curve.is_empty()
            && !self
                .flux_modulation_curve
                .iter()
                .any(|point| point.current_a == 0.0)
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_magnetic.flux_modulation_curve",
                constraint: "must include zero current",
            });
        }
        for (index, point) in self.flux_modulation_curve.iter().enumerate() {
            if !point.current_a.is_finite() || !point.delta_flux_density_t.is_finite() {
                return Err(ModelingError::NonFinite {
                    field: "motor_magnetic.flux_modulation_curve",
                });
            }
            if index > 0 {
                let previous = self.flux_modulation_curve[index - 1].current_a;
                if point.current_a <= previous {
                    return Err(ModelingError::OutOfRange {
                        field: "motor_magnetic.flux_modulation_curve",
                        constraint: "currents must be strictly increasing",
                    });
                }
            }
            if point.current_a == 0.0 && point.delta_flux_density_t != 0.0 {
                return Err(ModelingError::OutOfRange {
                    field: "motor_magnetic.flux_modulation_curve",
                    constraint: "zero-current delta must be zero",
                });
            }
        }
        if let Some(reluctance) = self.reluctance {
            reluctance.validate()?;
        }
        Ok(())
    }

    /// Evaluate the B-H law, current-driven flux modulation, energy, and
    /// optional gap reluctance force at one operating point.
    pub fn evaluate(
        &self,
        current_a: f64,
        flux_density_t: f64,
    ) -> Result<MotorMagneticOperatingValues, ModelingError> {
        self.validate()?;
        require_finite("motor_magnetic.current_a", current_a)?;
        require_finite("motor_magnetic.flux_density_t", flux_density_t)?;
        let effective_flux_density_t = flux_density_t + self.current_flux_modulation(current_a);
        let magnitude = effective_flux_density_t.abs();
        let segment = self.segment(magnitude);
        let lower = self.bh_curve[segment];
        let slope = self.segment_slope(segment);
        let delta_b = magnitude - lower.flux_density_t;
        let field_strength_magnitude = lower.field_strength_a_per_m + slope * delta_b;
        let sign = if effective_flux_density_t.is_sign_negative() {
            -1.0
        } else {
            1.0
        };
        let field_strength_a_per_m = sign * field_strength_magnitude;
        let differential_reluctivity_m_per_h = slope;
        let differential_permeability_h_per_m = 1.0 / slope;
        let permeability_h_per_m = if magnitude > 1.0e-15 {
            magnitude / field_strength_magnitude
        } else {
            differential_permeability_h_per_m
        };
        let energy_density_j_per_m3 = self.energy_density(segment, delta_b);
        let coenergy_density_j_per_m3 =
            magnitude * field_strength_magnitude - energy_density_j_per_m3;
        let magnetic_energy_j = energy_density_j_per_m3 * self.active_volume_m3;
        let magnetic_coenergy_j = coenergy_density_j_per_m3 * self.active_volume_m3;
        let reluctance_force_n = self
            .reluctance
            .map(|geometry| {
                -0.5 * magnitude.powi(2) * geometry.effective_gap_area_m2
                    / VACUUM_PERMEABILITY_H_PER_M
                    * geometry.gap_length_gradient_m_per_m
            })
            .unwrap_or(0.0);
        let values = MotorMagneticOperatingValues {
            effective_flux_density_t,
            field_strength_a_per_m,
            permeability_h_per_m,
            differential_reluctivity_m_per_h,
            differential_permeability_h_per_m,
            energy_density_j_per_m3,
            coenergy_density_j_per_m3,
            magnetic_energy_j,
            magnetic_coenergy_j,
            reluctance_force_n,
        };
        values.validate()?;
        Ok(values)
    }

    fn current_flux_modulation(&self, current_a: f64) -> f64 {
        if self.flux_modulation_curve.is_empty() {
            return current_a * self.flux_modulation_t_per_a;
        }
        let segment = self
            .flux_modulation_curve
            .partition_point(|point| point.current_a <= current_a)
            .saturating_sub(1)
            .min(self.flux_modulation_curve.len().saturating_sub(2));
        if self.flux_modulation_curve.len() == 1 {
            return self.flux_modulation_curve[0].delta_flux_density_t;
        }
        let lower = self.flux_modulation_curve[segment];
        let upper = self.flux_modulation_curve[segment + 1];
        let span = upper.current_a - lower.current_a;
        lower.delta_flux_density_t
            + (current_a - lower.current_a) / span
                * (upper.delta_flux_density_t - lower.delta_flux_density_t)
    }

    fn segment(&self, flux_density_t: f64) -> usize {
        self.bh_curve
            .partition_point(|point| point.flux_density_t <= flux_density_t)
            .saturating_sub(1)
            .min(self.bh_curve.len() - 2)
    }

    fn segment_slope(&self, segment: usize) -> f64 {
        let lower = self.bh_curve[segment];
        let upper = self.bh_curve[segment + 1];
        (upper.field_strength_a_per_m - lower.field_strength_a_per_m)
            / (upper.flux_density_t - lower.flux_density_t)
    }

    fn energy_density(&self, segment: usize, delta_b: f64) -> f64 {
        let mut energy = 0.0;
        for index in 0..segment {
            let lower = self.bh_curve[index];
            let upper = self.bh_curve[index + 1];
            energy += 0.5
                * (lower.field_strength_a_per_m + upper.field_strength_a_per_m)
                * (upper.flux_density_t - lower.flux_density_t);
        }
        let lower = self.bh_curve[segment];
        energy
            + lower.field_strength_a_per_m * delta_b
            + 0.5 * self.segment_slope(segment) * delta_b.powi(2)
    }
}

impl MotorMagneticOperatingValues {
    fn validate(self) -> Result<(), ModelingError> {
        if !self.effective_flux_density_t.is_finite()
            || !self.field_strength_a_per_m.is_finite()
            || !self.permeability_h_per_m.is_finite()
            || self.permeability_h_per_m <= 0.0
            || !self.differential_reluctivity_m_per_h.is_finite()
            || self.differential_reluctivity_m_per_h <= 0.0
            || !self.differential_permeability_h_per_m.is_finite()
            || self.differential_permeability_h_per_m <= 0.0
            || !self.energy_density_j_per_m3.is_finite()
            || self.energy_density_j_per_m3 < 0.0
            || !self.coenergy_density_j_per_m3.is_finite()
            || self.coenergy_density_j_per_m3 < 0.0
            || !self.magnetic_energy_j.is_finite()
            || self.magnetic_energy_j < 0.0
            || !self.magnetic_coenergy_j.is_finite()
            || self.magnetic_coenergy_j < 0.0
            || !self.reluctance_force_n.is_finite()
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_magnetic.operating_values",
                constraint: "finite passive magnetic quantities",
            });
        }
        Ok(())
    }
}

const VACUUM_PERMEABILITY_H_PER_M: f64 = 4.0e-7 * std::f64::consts::PI;

/// Current schema version for retained nonlinear operating-domain evidence.
pub const MOTOR_OPERATING_DOMAIN_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Maximum number of Cartesian operating points accepted by one validation
/// request.  This keeps a malformed serialized request from allocating an
/// unbounded evidence artifact while still allowing a useful five-axis
/// engineering grid (for example 3×3×3×3×3 = 243 points).
const MAX_MOTOR_OPERATING_DOMAIN_POINTS: usize = 4096;

/// Cartesian operating-domain request for nonlinear motor validation.
///
/// Every axis must be non-empty, finite, and strictly increasing. Frequency
/// and temperature are positive; displacement, current, and flux density may
/// be signed. The resulting evidence uses the deterministic axis order
/// displacement → current → frequency → temperature → flux density.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotorOperatingDomain {
    pub displacement_m: Vec<f64>,
    pub current_a: Vec<f64>,
    pub frequency_hz: Vec<f64>,
    pub temperature_k: Vec<f64>,
    pub flux_density_t: Vec<f64>,
}

impl MotorOperatingDomain {
    /// Validate the axes and their bounded Cartesian cardinality.
    pub fn validate(&self) -> Result<(), ModelingError> {
        validate_domain_axis(
            "motor_operating_domain.displacement_m",
            &self.displacement_m,
            false,
        )?;
        validate_domain_axis("motor_operating_domain.current_a", &self.current_a, false)?;
        validate_domain_axis(
            "motor_operating_domain.frequency_hz",
            &self.frequency_hz,
            true,
        )?;
        validate_domain_axis(
            "motor_operating_domain.temperature_k",
            &self.temperature_k,
            true,
        )?;
        validate_domain_axis(
            "motor_operating_domain.flux_density_t",
            &self.flux_density_t,
            false,
        )?;
        if self
            .point_count_checked()
            .is_none_or(|count| count > MAX_MOTOR_OPERATING_DOMAIN_POINTS)
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_domain",
                constraint: "Cartesian grid must contain at most 4096 points",
            });
        }
        Ok(())
    }

    /// Return the number of Cartesian operating points after validation.
    pub fn point_count(&self) -> Result<usize, ModelingError> {
        self.validate()?;
        Ok(self.point_count_unchecked())
    }

    /// Expand the request into its stable, serializable point order.
    pub fn points(&self) -> Result<Vec<MotorOperatingPoint>, ModelingError> {
        self.validate()?;
        let mut points = Vec::with_capacity(self.point_count_unchecked());
        for &displacement_m in &self.displacement_m {
            for &current_a in &self.current_a {
                for &frequency_hz in &self.frequency_hz {
                    for &temperature_k in &self.temperature_k {
                        for &flux_density_t in &self.flux_density_t {
                            points.push(MotorOperatingPoint {
                                displacement_m,
                                current_a,
                                frequency_hz,
                                temperature_k,
                                flux_density_t,
                            });
                        }
                    }
                }
            }
        }
        Ok(points)
    }

    fn point_count_unchecked(&self) -> usize {
        self.displacement_m.len()
            * self.current_a.len()
            * self.frequency_hz.len()
            * self.temperature_k.len()
            * self.flux_density_t.len()
    }

    fn point_count_checked(&self) -> Option<usize> {
        [
            self.displacement_m.len(),
            self.current_a.len(),
            self.frequency_hz.len(),
            self.temperature_k.len(),
            self.flux_density_t.len(),
        ]
        .into_iter()
        .try_fold(1usize, |count, axis| count.checked_mul(axis))
    }
}

fn validate_domain_axis(
    field: &'static str,
    values: &[f64],
    require_positive: bool,
) -> Result<(), ModelingError> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || (require_positive && *value <= 0.0))
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(ModelingError::OutOfRange {
            field,
            constraint: "non-empty finite strictly increasing physical axis",
        });
    }
    Ok(())
}

/// Retained, fail-closed evidence for one nonlinear operating-domain request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotorOperatingDomainEvidence {
    pub schema_version: u32,
    pub domain: MotorOperatingDomain,
    pub points: Vec<MotorOperatingPoint>,
    pub values: Vec<MotorOperatingValues>,
}

impl MotorOperatingDomainEvidence {
    /// Validate shape, schema, point ordering, and passive finite values.
    pub fn validate(&self) -> Result<(), ModelingError> {
        if self.schema_version != MOTOR_OPERATING_DOMAIN_EVIDENCE_SCHEMA_VERSION {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_domain_evidence.schema_version",
                constraint: "supported schema version",
            });
        }
        self.domain.validate()?;
        let expected_points = self.domain.points()?;
        if self.points != expected_points || self.values.len() != expected_points.len() {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_domain_evidence",
                constraint: "domain points and evaluated values must match",
            });
        }
        for (point, values) in self.points.iter().zip(&self.values) {
            values.validate()?;
            let expected_eddy_loss = values.eddy_resistance_ohm * point.current_a.powi(2);
            let scale = expected_eddy_loss
                .abs()
                .max(values.eddy_loss_w.abs())
                .max(1.0);
            if !expected_eddy_loss.is_finite()
                || (expected_eddy_loss - values.eddy_loss_w).abs() > 1.0e-11 * scale
            {
                return Err(ModelingError::OutOfRange {
                    field: "motor_operating_domain_evidence.eddy_loss_w",
                    constraint: "must match eddy resistance and current",
                });
            }
        }
        Ok(())
    }
}

/// Temperature coefficients for the small-signal motor quantities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotorTemperatureLaw {
    pub reference_temperature_k: f64,
    pub copper_resistance_ohm: f64,
    pub copper_resistance_per_k: f64,
    pub bl_per_k: f64,
    pub inductance_per_k: f64,
    pub eddy_resistance_per_k: f64,
}

impl Default for MotorTemperatureLaw {
    fn default() -> Self {
        Self {
            reference_temperature_k: 293.15,
            copper_resistance_ohm: 5.8,
            copper_resistance_per_k: 0.00393,
            bl_per_k: 0.0,
            inductance_per_k: 0.0,
            eddy_resistance_per_k: 0.0,
        }
    }
}

impl MotorTemperatureLaw {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive(
            "motor_temperature.reference_temperature_k",
            self.reference_temperature_k,
        )?;
        require_positive(
            "motor_temperature.copper_resistance_ohm",
            self.copper_resistance_ohm,
        )?;
        for (field, value) in [
            (
                "motor_temperature.copper_resistance_per_k",
                self.copper_resistance_per_k,
            ),
            ("motor_temperature.bl_per_k", self.bl_per_k),
            ("motor_temperature.inductance_per_k", self.inductance_per_k),
            (
                "motor_temperature.eddy_resistance_per_k",
                self.eddy_resistance_per_k,
            ),
        ] {
            require_finite(field, value)?;
        }
        Ok(())
    }

    fn factor(value_per_k: f64, temperature_k: f64, reference_k: f64) -> f64 {
        1.0 + value_per_k * (temperature_k - reference_k)
    }
}

/// A simple Steinmetz-style magnetic hysteresis/core-loss law.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HysteresisLossModel {
    pub coefficient_w_per_m3: f64,
    pub frequency_exponent: f64,
    pub flux_density_exponent: f64,
    pub active_volume_m3: f64,
}

/// One positive magnetic-loss observation used to fit a hysteresis law.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HysteresisLossSample {
    pub frequency_hz: f64,
    pub flux_density_t: f64,
    pub loss_w: f64,
}

/// Residual evidence for a fitted hysteresis/core-loss law.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HysteresisLossFitReport {
    pub sample_count: usize,
    pub rms_relative_error: f64,
    pub maximum_relative_error: f64,
    pub acceptance_tolerance: f64,
    pub accepted: bool,
}

impl Default for HysteresisLossModel {
    fn default() -> Self {
        Self {
            coefficient_w_per_m3: 0.0,
            frequency_exponent: 1.5,
            flux_density_exponent: 2.0,
            active_volume_m3: 1.0,
        }
    }
}

impl HysteresisLossModel {
    /// Fit `P = C V f^a |B|^b` in log space from positive measured samples.
    /// The fitted model is returned even when the residual exceeds the
    /// requested tolerance; callers must inspect `report.accepted` before
    /// promoting it into a calibrated product artifact.
    pub fn fit_from_samples(
        samples: &[HysteresisLossSample],
        active_volume_m3: f64,
        acceptance_tolerance: f64,
    ) -> Result<(Self, HysteresisLossFitReport), ModelingError> {
        require_positive("hysteresis_loss.active_volume_m3", active_volume_m3)?;
        require_nonnegative("hysteresis_loss.acceptance_tolerance", acceptance_tolerance)?;
        if samples.len() < 3 {
            return Err(ModelingError::OutOfRange {
                field: "hysteresis_loss.samples",
                constraint: "at least three positive observations",
            });
        }

        let mut normal = [[0.0_f64; 3]; 3];
        let mut rhs = [0.0_f64; 3];
        for sample in samples {
            require_positive("hysteresis_loss.sample.frequency_hz", sample.frequency_hz)?;
            require_positive(
                "hysteresis_loss.sample.flux_density_t",
                sample.flux_density_t.abs(),
            )?;
            require_positive("hysteresis_loss.sample.loss_w", sample.loss_w)?;
            let row = [
                1.0,
                sample.frequency_hz.ln(),
                sample.flux_density_t.abs().ln(),
            ];
            let target = (sample.loss_w / active_volume_m3).ln();
            for row_index in 0..3 {
                rhs[row_index] += row[row_index] * target;
                for column_index in 0..3 {
                    normal[row_index][column_index] += row[row_index] * row[column_index];
                }
            }
        }
        let [log_coefficient, frequency_exponent, flux_density_exponent] =
            solve_three_by_three(normal, rhs).ok_or(ModelingError::OutOfRange {
                field: "hysteresis_loss.samples",
                constraint: "independent frequency and flux-density axes",
            })?;
        let model = Self {
            coefficient_w_per_m3: log_coefficient.exp(),
            frequency_exponent,
            flux_density_exponent,
            active_volume_m3,
        };
        model.validate()?;

        let mut squared_relative_error = 0.0;
        let mut maximum_relative_error: f64 = 0.0;
        for sample in samples {
            let predicted = model.loss_w(sample.frequency_hz, sample.flux_density_t)?;
            let relative_error = (predicted - sample.loss_w).abs() / sample.loss_w;
            squared_relative_error += relative_error.powi(2);
            maximum_relative_error = maximum_relative_error.max(relative_error);
        }
        let report = HysteresisLossFitReport {
            sample_count: samples.len(),
            rms_relative_error: (squared_relative_error / samples.len() as f64).sqrt(),
            maximum_relative_error,
            acceptance_tolerance,
            accepted: maximum_relative_error <= acceptance_tolerance,
        };
        Ok((model, report))
    }

    pub fn validate(self) -> Result<(), ModelingError> {
        require_nonnegative(
            "hysteresis_loss.coefficient_w_per_m3",
            self.coefficient_w_per_m3,
        )?;
        require_positive(
            "hysteresis_loss.frequency_exponent",
            self.frequency_exponent,
        )?;
        require_positive(
            "hysteresis_loss.flux_density_exponent",
            self.flux_density_exponent,
        )?;
        require_positive("hysteresis_loss.active_volume_m3", self.active_volume_m3)
    }

    pub fn loss_w(self, frequency_hz: f64, flux_density_t: f64) -> Result<f64, ModelingError> {
        require_positive("frequency_hz", frequency_hz)?;
        require_finite("flux_density_t", flux_density_t)?;
        let loss = self.coefficient_w_per_m3
            * self.active_volume_m3
            * frequency_hz.powf(self.frequency_exponent)
            * flux_density_t.abs().powf(self.flux_density_exponent);
        require_nonnegative("hysteresis_loss_w", loss)?;
        Ok(loss)
    }
}

/// One calibrated eddy-current observation at a frequency and flux-density
/// operating point.
///
/// The resistance is the equivalent series resistance seen by the voice coil
/// for the retained geometry. `skin_depth_m` and `proximity_factor` remain
/// explicit diagnostics so a fitted surface cannot be mistaken for a purely
/// lumped resistance with no material-validity information.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EddyCurrentLossSample {
    pub frequency_hz: f64,
    pub flux_density_t: f64,
    pub eddy_resistance_ohm: f64,
    pub skin_depth_m: f64,
    pub proximity_factor: f64,
}

/// Eddy-current diagnostics evaluated from a calibrated surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EddyCurrentOperatingValues {
    pub eddy_resistance_ohm: f64,
    pub skin_depth_m: f64,
    pub proximity_factor: f64,
}

impl EddyCurrentOperatingValues {
    fn validate(self) -> Result<(), ModelingError> {
        require_nonnegative("eddy_current.eddy_resistance_ohm", self.eddy_resistance_ohm)?;
        require_positive("eddy_current.skin_depth_m", self.skin_depth_m)?;
        require_positive("eddy_current.proximity_factor", self.proximity_factor)
    }
}

/// Validated frequency/flux surface for calibrated eddy-current, skin, and
/// proximity behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EddyCurrentLossModel {
    pub samples: Vec<EddyCurrentLossSample>,
}

impl EddyCurrentLossModel {
    /// Construct a complete rectangular calibration surface.
    pub fn from_samples(samples: &[EddyCurrentLossSample]) -> Result<Self, ModelingError> {
        let model = Self {
            samples: samples.to_vec(),
        };
        model.validate()?;
        Ok(model)
    }

    /// Require at least a 2×2 rectangular surface with finite passive values.
    pub fn validate(&self) -> Result<(), ModelingError> {
        if self.samples.len() < 2 {
            return Err(ModelingError::OutOfRange {
                field: "eddy_current.samples",
                constraint: "at least two samples",
            });
        }

        let mut frequencies = Vec::new();
        let mut flux_densities = Vec::new();
        for sample in &self.samples {
            require_positive("eddy_current.sample.frequency_hz", sample.frequency_hz)?;
            require_positive(
                "eddy_current.sample.flux_density_t",
                sample.flux_density_t.abs(),
            )?;
            require_nonnegative(
                "eddy_current.sample.eddy_resistance_ohm",
                sample.eddy_resistance_ohm,
            )?;
            require_positive("eddy_current.sample.skin_depth_m", sample.skin_depth_m)?;
            require_positive(
                "eddy_current.sample.proximity_factor",
                sample.proximity_factor,
            )?;
            if !frequencies.contains(&sample.frequency_hz) {
                frequencies.push(sample.frequency_hz);
            }
            let flux_density_t = sample.flux_density_t.abs();
            if !flux_densities.contains(&flux_density_t) {
                flux_densities.push(flux_density_t);
            }
        }
        frequencies.sort_by(f64::total_cmp);
        flux_densities.sort_by(f64::total_cmp);
        if frequencies.len() < 2 || flux_densities.is_empty() {
            return Err(ModelingError::OutOfRange {
                field: "eddy_current.samples",
                constraint: "at least two frequency and one flux-density level",
            });
        }
        if self.samples.len() != frequencies.len() * flux_densities.len() {
            return Err(ModelingError::OutOfRange {
                field: "eddy_current.samples",
                constraint: "complete rectangular frequency/flux grid without duplicates",
            });
        }
        for &frequency_hz in &frequencies {
            for &flux_density_t in &flux_densities {
                let matches = self
                    .samples
                    .iter()
                    .filter(|sample| {
                        sample.frequency_hz == frequency_hz
                            && sample.flux_density_t.abs() == flux_density_t
                    })
                    .count();
                if matches != 1 {
                    return Err(ModelingError::OutOfRange {
                        field: "eddy_current.samples",
                        constraint: "one sample per frequency/flux grid cell",
                    });
                }
            }
        }
        Ok(())
    }

    /// Evaluate the calibrated surface without extrapolating beyond its
    /// measured/FEM support.
    pub fn evaluate(
        &self,
        frequency_hz: f64,
        flux_density_t: f64,
    ) -> Result<EddyCurrentOperatingValues, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        require_finite("flux_density_t", flux_density_t)?;

        let mut frequencies: Vec<_> = self
            .samples
            .iter()
            .map(|sample| sample.frequency_hz)
            .collect();
        frequencies.sort_by(f64::total_cmp);
        frequencies.dedup();
        let mut flux_densities: Vec<_> = self
            .samples
            .iter()
            .map(|sample| sample.flux_density_t.abs())
            .collect();
        flux_densities.sort_by(f64::total_cmp);
        flux_densities.dedup();

        let (frequency_lower, frequency_upper, frequency_weight) =
            interpolation_bracket(&frequencies, frequency_hz)?;
        let (flux_lower, flux_upper, flux_weight) = if flux_densities.len() == 1 {
            // A one-flux FEM sweep is a frequency-only calibration slice;
            // do not pretend it identifies flux dependence, but allow it to
            // serve operating points whose caller does not retain B.
            (flux_densities[0], flux_densities[0], 0.0)
        } else {
            require_positive("flux_density_t", flux_density_t.abs())?;
            interpolation_bracket(&flux_densities, flux_density_t.abs())?
        };
        let sample = |frequency: f64, flux_density: f64| {
            self.samples
                .iter()
                .find(|candidate| {
                    candidate.frequency_hz == frequency
                        && candidate.flux_density_t.abs() == flux_density
                })
                .copied()
                .ok_or(ModelingError::OutOfRange {
                    field: "eddy_current.samples",
                    constraint: "complete rectangular frequency/flux grid",
                })
        };
        let lower_lower = sample(frequency_lower, flux_lower)?;
        let upper_lower = sample(frequency_upper, flux_lower)?;
        let lower_upper = sample(frequency_lower, flux_upper)?;
        let upper_upper = sample(frequency_upper, flux_upper)?;
        let blend = |lower_lower: f64, upper_lower: f64, lower_upper: f64, upper_upper: f64| {
            let lower = lower_lower + frequency_weight * (upper_lower - lower_lower);
            let upper = lower_upper + frequency_weight * (upper_upper - lower_upper);
            lower + flux_weight * (upper - lower)
        };
        let values = EddyCurrentOperatingValues {
            eddy_resistance_ohm: blend(
                lower_lower.eddy_resistance_ohm,
                upper_lower.eddy_resistance_ohm,
                lower_upper.eddy_resistance_ohm,
                upper_upper.eddy_resistance_ohm,
            ),
            skin_depth_m: blend(
                lower_lower.skin_depth_m,
                upper_lower.skin_depth_m,
                lower_upper.skin_depth_m,
                upper_upper.skin_depth_m,
            ),
            proximity_factor: blend(
                lower_lower.proximity_factor,
                upper_lower.proximity_factor,
                lower_upper.proximity_factor,
                upper_upper.proximity_factor,
            ),
        };
        values.validate()?;
        Ok(values)
    }
}

fn interpolation_bracket(axis: &[f64], target: f64) -> Result<(f64, f64, f64), ModelingError> {
    if target < axis[0] || target > axis[axis.len() - 1] {
        return Err(ModelingError::OutOfRange {
            field: "eddy_current.operating_point",
            constraint: "inside calibrated frequency/flux support",
        });
    }
    if let Some(index) = axis.iter().position(|value| *value == target) {
        return Ok((axis[index], axis[index], 0.0));
    }
    let upper = axis.partition_point(|value| *value < target);
    let lower = upper - 1;
    let weight = (target - axis[lower]) / (axis[upper] - axis[lower]);
    Ok((axis[lower], axis[upper], weight))
}

fn solve_three_by_three(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for pivot_column in 0..3 {
        let mut pivot_row = pivot_column;
        for row in (pivot_column + 1)..3 {
            if matrix[row][pivot_column].abs() > matrix[pivot_row][pivot_column].abs() {
                pivot_row = row;
            }
        }
        let pivot = matrix[pivot_row][pivot_column];
        if !pivot.is_finite() || pivot.abs() <= f64::EPSILON {
            return None;
        }
        if pivot_row != pivot_column {
            matrix.swap(pivot_row, pivot_column);
            rhs.swap(pivot_row, pivot_column);
        }
        for row in (pivot_column + 1)..3 {
            let factor = matrix[row][pivot_column] / matrix[pivot_column][pivot_column];
            let pivot_row = matrix[pivot_column];
            for (off, slot) in matrix[row][pivot_column..3].iter_mut().enumerate() {
                *slot -= factor * pivot_row[pivot_column + off];
            }
            rhs[row] -= factor * rhs[pivot_column];
        }
    }

    let mut solution = [0.0; 3];
    for row in (0..3).rev() {
        let tail = ((row + 1)..3)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
        if !solution[row].is_finite() {
            return None;
        }
    }
    Some(solution)
}

/// Evaluated motor quantities at one nonlinear operating point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotorOperatingValues {
    pub force_factor_tm: f64,
    pub inductance_h: f64,
    pub eddy_resistance_ohm: f64,
    pub copper_resistance_ohm: f64,
    pub hysteresis_loss_w: f64,
    pub eddy_loss_w: f64,
    pub total_magnetic_loss_w: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic: Option<MotorMagneticOperatingValues>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eddy_current: Option<EddyCurrentOperatingValues>,
}

impl MotorOperatingValues {
    /// Validate the passive, finite result retained in domain evidence.
    pub fn validate(&self) -> Result<(), ModelingError> {
        if !self.force_factor_tm.is_finite()
            || !self.inductance_h.is_finite()
            || self.inductance_h < 0.0
            || !self.eddy_resistance_ohm.is_finite()
            || self.eddy_resistance_ohm < 0.0
            || !self.copper_resistance_ohm.is_finite()
            || self.copper_resistance_ohm <= 0.0
            || !self.hysteresis_loss_w.is_finite()
            || self.hysteresis_loss_w < 0.0
            || !self.eddy_loss_w.is_finite()
            || self.eddy_loss_w < 0.0
            || !self.total_magnetic_loss_w.is_finite()
            || self.total_magnetic_loss_w < 0.0
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_values",
                constraint: "finite passive motor quantities",
            });
        }
        let expected_total = self.hysteresis_loss_w + self.eddy_loss_w;
        let scale = expected_total
            .abs()
            .max(self.total_magnetic_loss_w.abs())
            .max(1.0);
        if !expected_total.is_finite()
            || (expected_total - self.total_magnetic_loss_w).abs() > 1.0e-11 * scale
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_values.total_magnetic_loss_w",
                constraint: "must equal hysteresis plus eddy loss",
            });
        }
        if let Some(magnetic) = self.magnetic {
            magnetic.validate()?;
        }
        if let Some(eddy_current) = self.eddy_current {
            eddy_current.validate()?;
            let scale = self
                .eddy_resistance_ohm
                .abs()
                .max(eddy_current.eddy_resistance_ohm.abs())
                .max(1.0);
            if (self.eddy_resistance_ohm - eddy_current.eddy_resistance_ohm).abs() > 1.0e-11 * scale
            {
                return Err(ModelingError::OutOfRange {
                    field: "motor_operating_values.eddy_current.eddy_resistance_ohm",
                    constraint: "must match the evaluated eddy resistance",
                });
            }
        }
        Ok(())
    }
}

/// Validated nonlinear motor model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotorConstitutiveModel {
    pub bl_curve: Vec<NonlinearBlPoint>,
    pub le_surface: Vec<NonlinearLePoint>,
    pub temperature: MotorTemperatureLaw,
    pub hysteresis: HysteresisLossModel,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic: Option<MotorMagneticLaw>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eddy_current: Option<EddyCurrentLossModel>,
}

impl MotorConstitutiveModel {
    pub fn validate(&self) -> Result<(), ModelingError> {
        if self.bl_curve.is_empty()
            || self.bl_curve.iter().any(|point| {
                !point.displacement_m.is_finite() || !point.force_factor_tm.is_finite()
            })
            || self
                .bl_curve
                .windows(2)
                .any(|pair| pair[0].displacement_m >= pair[1].displacement_m)
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_nonlinearity.bl_curve",
                constraint: "non-empty finite strictly ordered samples",
            });
        }
        if self.le_surface.is_empty()
            || self.le_surface.iter().any(|point| {
                !point.displacement_m.is_finite()
                    || !point.current_a.is_finite()
                    || !point.frequency_hz.is_finite()
                    || point.frequency_hz <= 0.0
                    || !point.inductance_h.is_finite()
                    || point.inductance_h < 0.0
                    || !point.eddy_resistance_ohm.is_finite()
                    || point.eddy_resistance_ohm < 0.0
            })
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_nonlinearity.le_surface",
                constraint: "non-empty finite passive samples",
            });
        }
        self.temperature.validate()?;
        self.hysteresis.validate()?;
        if let Some(magnetic) = &self.magnetic {
            magnetic.validate()?;
        }
        if let Some(eddy_current) = &self.eddy_current {
            eddy_current.validate()?;
        }
        Ok(())
    }

    /// Evaluate BL(x), Le(x,i,f), temperature drift, and both magnetic loss
    /// channels. Surface interpolation is deterministic inverse-distance
    /// interpolation, which also supports irregular FEM extraction grids.
    pub fn evaluate(
        &self,
        point: MotorOperatingPoint,
    ) -> Result<MotorOperatingValues, ModelingError> {
        self.validate()?;
        require_finite("motor_operating_point.displacement_m", point.displacement_m)?;
        require_finite("motor_operating_point.current_a", point.current_a)?;
        require_positive("motor_operating_point.frequency_hz", point.frequency_hz)?;
        require_positive("motor_operating_point.temperature_k", point.temperature_k)?;
        require_finite("motor_operating_point.flux_density_t", point.flux_density_t)?;

        let temperature_bl = MotorTemperatureLaw::factor(
            self.temperature.bl_per_k,
            point.temperature_k,
            self.temperature.reference_temperature_k,
        );
        let temperature_le = MotorTemperatureLaw::factor(
            self.temperature.inductance_per_k,
            point.temperature_k,
            self.temperature.reference_temperature_k,
        );
        let temperature_eddy = MotorTemperatureLaw::factor(
            self.temperature.eddy_resistance_per_k,
            point.temperature_k,
            self.temperature.reference_temperature_k,
        );
        let force_factor_tm = interpolate_bl(&self.bl_curve, point.displacement_m) * temperature_bl;
        let (inductance_h, eddy_resistance_ohm) = interpolate_le(
            &self.le_surface,
            point.displacement_m,
            point.current_a,
            point.frequency_hz,
        );
        let inductance_h = inductance_h * temperature_le;
        let calibrated_eddy_current = self
            .eddy_current
            .as_ref()
            .map(|model| model.evaluate(point.frequency_hz, point.flux_density_t))
            .transpose()?;
        let eddy_resistance_ohm = calibrated_eddy_current
            .map_or(eddy_resistance_ohm, |values| values.eddy_resistance_ohm)
            * temperature_eddy;
        let calibrated_eddy_current = calibrated_eddy_current.map(|mut values| {
            values.eddy_resistance_ohm *= temperature_eddy;
            values
        });
        let copper_resistance_ohm = self.temperature.copper_resistance_ohm
            * MotorTemperatureLaw::factor(
                self.temperature.copper_resistance_per_k,
                point.temperature_k,
                self.temperature.reference_temperature_k,
            );
        for (field, value) in [
            ("motor_operating.force_factor_tm", force_factor_tm),
            ("motor_operating.inductance_h", inductance_h),
            ("motor_operating.eddy_resistance_ohm", eddy_resistance_ohm),
            (
                "motor_operating.copper_resistance_ohm",
                copper_resistance_ohm,
            ),
        ] {
            require_finite(field, value)?;
        }
        if inductance_h < 0.0 || eddy_resistance_ohm < 0.0 || copper_resistance_ohm <= 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating.temperature",
                constraint: "temperature law must preserve passive electrical quantities",
            });
        }
        let magnetic = self
            .magnetic
            .as_ref()
            .map(|law| law.evaluate(point.current_a, point.flux_density_t))
            .transpose()?;
        let hysteresis_flux_density_t = magnetic
            .map(|values| values.effective_flux_density_t)
            .unwrap_or(point.flux_density_t);
        let hysteresis_loss_w = self
            .hysteresis
            .loss_w(point.frequency_hz, hysteresis_flux_density_t)?;
        let eddy_loss_w = eddy_resistance_ohm * point.current_a.powi(2);
        let values = MotorOperatingValues {
            force_factor_tm,
            inductance_h,
            eddy_resistance_ohm,
            copper_resistance_ohm,
            hysteresis_loss_w,
            eddy_loss_w,
            total_magnetic_loss_w: hysteresis_loss_w + eddy_loss_w,
            magnetic,
            eddy_current: calibrated_eddy_current,
        };
        values.validate()?;
        Ok(values)
    }

    /// Evaluate every point in a bounded Cartesian operating-domain request
    /// and retain the result as a promotion artifact.
    pub fn validate_operating_domain(
        &self,
        domain: &MotorOperatingDomain,
    ) -> Result<MotorOperatingDomainEvidence, ModelingError> {
        self.validate()?;
        let points = domain.points()?;
        let mut values = Vec::with_capacity(points.len());
        for point in &points {
            self.validate_point_support(*point)?;
            values.push(self.evaluate(*point)?);
        }
        let evidence = MotorOperatingDomainEvidence {
            schema_version: MOTOR_OPERATING_DOMAIN_EVIDENCE_SCHEMA_VERSION,
            domain: domain.clone(),
            points,
            values,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    /// Re-evaluate retained evidence against this exact constitutive model.
    /// This is used when a serialized `DriverModel` is loaded so stale or
    /// substituted operating-point values cannot be promoted silently.
    pub fn validate_operating_domain_evidence(
        &self,
        evidence: &MotorOperatingDomainEvidence,
    ) -> Result<(), ModelingError> {
        self.validate()?;
        evidence.validate()?;
        for (point, retained) in evidence.points.iter().zip(&evidence.values) {
            self.validate_point_support(*point)?;
            let expected = self.evaluate(*point)?;
            if !operating_values_match(expected, *retained) {
                return Err(ModelingError::OutOfRange {
                    field: "motor_operating_domain_evidence.values",
                    constraint: "values must match the attached constitutive model",
                });
            }
        }
        Ok(())
    }

    fn validate_point_support(&self, point: MotorOperatingPoint) -> Result<(), ModelingError> {
        let (bl_min, bl_max) = curve_bounds(&self.bl_curve, |sample| sample.displacement_m);
        if point.displacement_m < bl_min || point.displacement_m > bl_max {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_point.displacement_m",
                constraint: "inside extracted BL(x) support",
            });
        }

        let (le_x_min, le_x_max) = curve_bounds(&self.le_surface, |sample| sample.displacement_m);
        let (le_i_min, le_i_max) = curve_bounds(&self.le_surface, |sample| sample.current_a);
        let (le_f_min, le_f_max) = curve_bounds(&self.le_surface, |sample| sample.frequency_hz);
        if point.displacement_m < le_x_min
            || point.displacement_m > le_x_max
            || point.current_a < le_i_min
            || point.current_a > le_i_max
            || point.frequency_hz < le_f_min
            || point.frequency_hz > le_f_max
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_point",
                constraint: "inside extracted Le(x,i,f) support",
            });
        }

        let exact_sample = self.le_surface.iter().any(|sample| {
            le_coordinate_matches(sample.displacement_m, point.displacement_m)
                && le_coordinate_matches(sample.current_a, point.current_a)
                && le_coordinate_matches(sample.frequency_hz, point.frequency_hz)
        });
        if !exact_sample
            && interpolate_le_tensor_grid(
                &self.le_surface,
                point.displacement_m,
                point.current_a,
                point.frequency_hz,
            )
            .is_none()
        {
            return Err(ModelingError::OutOfRange {
                field: "motor_operating_point",
                constraint: "Le(x,i,f) must contain an exact sample or complete tensor cell",
            });
        }
        Ok(())
    }
}

fn curve_bounds<T>(points: &[T], coordinate: impl Fn(&T) -> f64) -> (f64, f64) {
    points.iter().map(coordinate).fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    )
}

fn operating_values_match(expected: MotorOperatingValues, retained: MotorOperatingValues) -> bool {
    let scalar_match = [
        (expected.force_factor_tm, retained.force_factor_tm),
        (expected.inductance_h, retained.inductance_h),
        (expected.eddy_resistance_ohm, retained.eddy_resistance_ohm),
        (
            expected.copper_resistance_ohm,
            retained.copper_resistance_ohm,
        ),
        (expected.hysteresis_loss_w, retained.hysteresis_loss_w),
        (expected.eddy_loss_w, retained.eddy_loss_w),
        (
            expected.total_magnetic_loss_w,
            retained.total_magnetic_loss_w,
        ),
    ]
    .into_iter()
    .all(|(expected, retained)| {
        let scale = expected.abs().max(retained.abs()).max(1.0);
        (expected - retained).abs() <= 1.0e-11 * scale
    });
    scalar_match
        && match (expected.magnetic, retained.magnetic) {
            (None, None) => true,
            (Some(expected), Some(retained)) => [
                (
                    expected.effective_flux_density_t,
                    retained.effective_flux_density_t,
                ),
                (
                    expected.field_strength_a_per_m,
                    retained.field_strength_a_per_m,
                ),
                (expected.permeability_h_per_m, retained.permeability_h_per_m),
                (
                    expected.differential_reluctivity_m_per_h,
                    retained.differential_reluctivity_m_per_h,
                ),
                (
                    expected.differential_permeability_h_per_m,
                    retained.differential_permeability_h_per_m,
                ),
                (
                    expected.energy_density_j_per_m3,
                    retained.energy_density_j_per_m3,
                ),
                (
                    expected.coenergy_density_j_per_m3,
                    retained.coenergy_density_j_per_m3,
                ),
                (expected.magnetic_energy_j, retained.magnetic_energy_j),
                (expected.magnetic_coenergy_j, retained.magnetic_coenergy_j),
                (expected.reluctance_force_n, retained.reluctance_force_n),
            ]
            .into_iter()
            .all(|(expected, retained)| {
                let scale = expected.abs().max(retained.abs()).max(1.0);
                (expected - retained).abs() <= 1.0e-11 * scale
            }),
            _ => false,
        }
        && match (expected.eddy_current, retained.eddy_current) {
            (None, None) => true,
            (Some(expected), Some(retained)) => [
                (expected.eddy_resistance_ohm, retained.eddy_resistance_ohm),
                (expected.skin_depth_m, retained.skin_depth_m),
                (expected.proximity_factor, retained.proximity_factor),
            ]
            .into_iter()
            .all(|(expected, retained)| {
                let scale = expected.abs().max(retained.abs()).max(1.0);
                (expected - retained).abs() <= 1.0e-11 * scale
            }),
            _ => false,
        }
}

fn interpolate_bl(points: &[NonlinearBlPoint], x: f64) -> f64 {
    if x <= points[0].displacement_m {
        return points[0].force_factor_tm;
    }
    if x >= points[points.len() - 1].displacement_m {
        return points[points.len() - 1].force_factor_tm;
    }
    let upper = points.partition_point(|point| point.displacement_m < x);
    let lower = upper - 1;
    let fraction = (x - points[lower].displacement_m)
        / (points[upper].displacement_m - points[lower].displacement_m);
    points[lower].force_factor_tm
        + fraction * (points[upper].force_factor_tm - points[lower].force_factor_tm)
}

const LE_COORDINATE_MATCH_TOLERANCE: f64 = 1.0e-12;

fn le_coordinate_matches(actual: f64, target: f64) -> bool {
    if !actual.is_finite() || !target.is_finite() {
        return false;
    }
    let scale = actual.abs().max(target.abs()).max(1.0);
    (actual - target).abs() <= LE_COORDINATE_MATCH_TOLERANCE * scale
}

fn interpolate_le(points: &[NonlinearLePoint], x: f64, current: f64, frequency: f64) -> (f64, f64) {
    if let Some(point) = points.iter().find(|point| {
        le_coordinate_matches(point.displacement_m, x)
            && le_coordinate_matches(point.current_a, current)
            && le_coordinate_matches(point.frequency_hz, frequency)
    }) {
        return (point.inductance_h, point.eddy_resistance_ohm);
    }
    if let Some(interpolated) = interpolate_le_tensor_grid(points, x, current, frequency) {
        return interpolated;
    }
    let x_scale = points
        .iter()
        .map(|point| point.displacement_m.abs())
        .fold(0.0, f64::max)
        .max(1.0e-9);
    let i_scale = points
        .iter()
        .map(|point| point.current_a.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    let f_scale = points
        .iter()
        .map(|point| point.frequency_hz)
        .fold(0.0, f64::max)
        .max(1.0);
    let fallback = points.first().map_or((0.0, 0.0), |point| {
        (point.inductance_h, point.eddy_resistance_ohm)
    });
    let mut denominator = 0.0;
    let mut inductance = 0.0;
    let mut resistance = 0.0;
    for point in points {
        let distance = ((point.displacement_m - x) / x_scale).powi(2)
            + ((point.current_a - current) / i_scale).powi(2)
            + ((point.frequency_hz - frequency) / f_scale).powi(2);
        let weight = 1.0 / (distance + 1.0e-18);
        denominator += weight;
        inductance += weight * point.inductance_h;
        resistance += weight * point.eddy_resistance_ohm;
    }
    if denominator > 0.0 {
        (inductance / denominator, resistance / denominator)
    } else {
        fallback
    }
}

/// Return the bracketing sample coordinates for one tensor-grid axis, clamped
/// to the measured range. `None` means the supplied surface cannot support a
/// finite interpolation.
fn le_axis_bounds(
    points: &[NonlinearLePoint],
    target: f64,
    coordinate: impl Fn(&NonlinearLePoint) -> f64,
) -> Option<(f64, f64)> {
    if !target.is_finite() {
        return None;
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::INFINITY;
    for point in points {
        let value = coordinate(point);
        if !value.is_finite() {
            return None;
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        if value <= target {
            lower = lower.max(value);
        }
        if value >= target {
            upper = upper.min(value);
        }
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        None
    } else if target <= minimum {
        Some((minimum, minimum))
    } else if target >= maximum {
        Some((maximum, maximum))
    } else if lower.is_finite() && upper.is_finite() {
        Some((lower, upper))
    } else {
        None
    }
}

fn le_axis_weights(target: f64, lower: f64, upper: f64) -> [f64; 2] {
    if lower == upper {
        [1.0, 0.0]
    } else {
        let upper_weight = ((target - lower) / (upper - lower)).clamp(0.0, 1.0);
        [1.0 - upper_weight, upper_weight]
    }
}

/// Interpolate a complete local `Le(x, i, f)` tensor cell. The measured
/// extraction surface may be scattered (for example during calibration), so
/// callers fall back to inverse-distance interpolation when a required corner
/// is absent rather than inventing a synthetic corner value.
fn interpolate_le_tensor_grid(
    points: &[NonlinearLePoint],
    x: f64,
    current: f64,
    frequency: f64,
) -> Option<(f64, f64)> {
    let (x0, x1) = le_axis_bounds(points, x, |point| point.displacement_m)?;
    let (i0, i1) = le_axis_bounds(points, current, |point| point.current_a)?;
    let (f0, f1) = le_axis_bounds(points, frequency, |point| point.frequency_hz)?;
    let x_values = [x0, x1];
    let i_values = [i0, i1];
    let f_values = [f0, f1];
    let x_weights = le_axis_weights(x, x0, x1);
    let i_weights = le_axis_weights(current, i0, i1);
    let f_weights = le_axis_weights(frequency, f0, f1);
    let mut weight_sum = 0.0;
    let mut inductance = 0.0;
    let mut resistance = 0.0;

    for xi in 0..2 {
        for ii in 0..2 {
            for fi in 0..2 {
                let weight = x_weights[xi] * i_weights[ii] * f_weights[fi];
                if weight == 0.0 {
                    continue;
                }
                let point = points.iter().find(|point| {
                    le_coordinate_matches(point.displacement_m, x_values[xi])
                        && le_coordinate_matches(point.current_a, i_values[ii])
                        && le_coordinate_matches(point.frequency_hz, f_values[fi])
                })?;
                if !point.inductance_h.is_finite()
                    || point.inductance_h <= 0.0
                    || !point.eddy_resistance_ohm.is_finite()
                    || point.eddy_resistance_ohm < 0.0
                {
                    return None;
                }
                weight_sum += weight;
                inductance += weight * point.inductance_h;
                resistance += weight * point.eddy_resistance_ohm;
            }
        }
    }
    if weight_sum > 0.0 && weight_sum.is_finite() {
        Some((inductance / weight_sum, resistance / weight_sum))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> MotorConstitutiveModel {
        MotorConstitutiveModel {
            bl_curve: vec![
                NonlinearBlPoint {
                    displacement_m: -0.001,
                    force_factor_tm: 5.0,
                },
                NonlinearBlPoint {
                    displacement_m: 0.0,
                    force_factor_tm: 6.0,
                },
                NonlinearBlPoint {
                    displacement_m: 0.001,
                    force_factor_tm: 5.0,
                },
            ],
            le_surface: vec![
                NonlinearLePoint {
                    displacement_m: 0.0,
                    current_a: 0.0,
                    frequency_hz: 100.0,
                    inductance_h: 0.5e-3,
                    eddy_resistance_ohm: 0.1,
                },
                NonlinearLePoint {
                    displacement_m: 0.0,
                    current_a: 2.0,
                    frequency_hz: 100.0,
                    inductance_h: 0.4e-3,
                    eddy_resistance_ohm: 0.3,
                },
                NonlinearLePoint {
                    displacement_m: 0.001,
                    current_a: 0.0,
                    frequency_hz: 1_000.0,
                    inductance_h: 0.3e-3,
                    eddy_resistance_ohm: 0.5,
                },
            ],
            temperature: MotorTemperatureLaw {
                copper_resistance_ohm: 5.8,
                bl_per_k: -0.001,
                ..MotorTemperatureLaw::default()
            },
            hysteresis: HysteresisLossModel {
                coefficient_w_per_m3: 2.0,
                active_volume_m3: 1.0e-5,
                ..HysteresisLossModel::default()
            },
            magnetic: None,
            eddy_current: None,
        }
    }

    fn magnetic_law() -> MotorMagneticLaw {
        MotorMagneticLaw {
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
                MotorMagneticBhPoint {
                    flux_density_t: 1.5,
                    field_strength_a_per_m: 5_000.0,
                },
            ],
            active_volume_m3: 2.0e-5,
            flux_modulation_t_per_a: 0.05,
            flux_modulation_curve: Vec::new(),
            reluctance: Some(MotorReluctanceGeometry {
                effective_gap_area_m2: 0.01,
                magnetic_gap_length_m: 1.0e-3,
                gap_length_gradient_m_per_m: 0.5,
            }),
        }
    }

    fn eddy_current_law() -> EddyCurrentLossModel {
        EddyCurrentLossModel::from_samples(&[
            EddyCurrentLossSample {
                frequency_hz: 100.0,
                flux_density_t: 0.5,
                eddy_resistance_ohm: 0.1,
                skin_depth_m: 0.010,
                proximity_factor: 1.1,
            },
            EddyCurrentLossSample {
                frequency_hz: 1_000.0,
                flux_density_t: 0.5,
                eddy_resistance_ohm: 0.3,
                skin_depth_m: 0.005,
                proximity_factor: 1.3,
            },
            EddyCurrentLossSample {
                frequency_hz: 100.0,
                flux_density_t: 1.0,
                eddy_resistance_ohm: 0.2,
                skin_depth_m: 0.009,
                proximity_factor: 1.2,
            },
            EddyCurrentLossSample {
                frequency_hz: 1_000.0,
                flux_density_t: 1.0,
                eddy_resistance_ohm: 0.4,
                skin_depth_m: 0.004,
                proximity_factor: 1.4,
            },
        ])
        .unwrap()
    }

    #[test]
    fn calibrated_eddy_surface_interpolates_skin_and_proximity_values() {
        let law = eddy_current_law();
        let values = law.evaluate(550.0, 0.75).unwrap();
        assert!((values.eddy_resistance_ohm - 0.25).abs() < 1.0e-12);
        assert!((values.skin_depth_m - 0.007).abs() < 1.0e-12);
        assert!((values.proximity_factor - 1.25).abs() < 1.0e-12);

        let decoded: EddyCurrentLossModel =
            serde_json::from_slice(&serde_json::to_vec(&law).unwrap()).unwrap();
        assert_eq!(decoded, law);
    }

    #[test]
    fn calibrated_eddy_surface_rejects_incomplete_and_out_of_support_data() {
        let mut incomplete = eddy_current_law().samples;
        incomplete.pop();
        assert!(EddyCurrentLossModel::from_samples(&incomplete).is_err());

        let law = eddy_current_law();
        assert!(law.evaluate(50.0, 0.75).is_err());
        assert!(law.evaluate(550.0, 1.5).is_err());

        let mut malformed = law.samples;
        malformed[0].proximity_factor = 0.0;
        assert!(EddyCurrentLossModel::from_samples(&malformed).is_err());
    }

    #[test]
    fn calibrated_eddy_surface_overrides_lumped_resistance_and_retains_diagnostics() {
        let mut motor = model();
        motor.eddy_current = Some(eddy_current_law());
        let values = motor
            .evaluate(MotorOperatingPoint {
                displacement_m: 0.0,
                current_a: 2.0,
                frequency_hz: 550.0,
                temperature_k: 293.15,
                flux_density_t: 0.75,
            })
            .unwrap();
        assert!((values.eddy_resistance_ohm - 0.25).abs() < 1.0e-12);
        assert!((values.eddy_loss_w - 1.0).abs() < 1.0e-12);
        let eddy_current = values.eddy_current.unwrap();
        assert!((eddy_current.eddy_resistance_ohm - 0.25).abs() < 1.0e-12);
        assert!((eddy_current.skin_depth_m - 0.007).abs() < 1.0e-12);
        assert!((eddy_current.proximity_factor - 1.25).abs() < 1.0e-12);
    }

    #[test]
    fn magnetic_law_exposes_saturation_energy_and_reluctance_force() {
        let law = magnetic_law();
        let low = law.evaluate(0.0, 0.25).unwrap();
        let high = law.evaluate(0.0, 1.25).unwrap();

        assert!(high.differential_permeability_h_per_m < low.differential_permeability_h_per_m);
        assert!(high.energy_density_j_per_m3 > low.energy_density_j_per_m3);
        let conjugate = high.effective_flux_density_t.abs() * high.field_strength_a_per_m.abs();
        assert!(
            (high.energy_density_j_per_m3 + high.coenergy_density_j_per_m3 - conjugate).abs()
                < 1.0e-9
        );
        let expected_force = -0.5 * 1.25_f64.powi(2) * 0.01 / VACUUM_PERMEABILITY_H_PER_M * 0.5;
        assert!((high.reluctance_force_n - expected_force).abs() / expected_force.abs() < 1.0e-12);
    }

    #[test]
    fn magnetic_law_applies_current_flux_modulation_to_operating_values() {
        let mut motor = model();
        motor.magnetic = Some(magnetic_law());
        let values = motor
            .evaluate(MotorOperatingPoint {
                displacement_m: 0.0,
                current_a: 2.0,
                frequency_hz: 100.0,
                temperature_k: 293.15,
                flux_density_t: 0.5,
            })
            .unwrap();
        let magnetic = values.magnetic.unwrap();
        assert!((magnetic.effective_flux_density_t - 0.6).abs() < 1.0e-12);
        assert!(
            values.hysteresis_loss_w > HysteresisLossModel::default().loss_w(100.0, 0.5).unwrap()
        );
    }

    #[test]
    fn magnetic_law_uses_and_validates_retained_flux_modulation_curve() {
        let mut law = magnetic_law();
        law.flux_modulation_curve = vec![
            MotorMagneticFluxModulationPoint {
                current_a: -1.0,
                delta_flux_density_t: -0.2,
            },
            MotorMagneticFluxModulationPoint {
                current_a: 0.0,
                delta_flux_density_t: 0.0,
            },
            MotorMagneticFluxModulationPoint {
                current_a: 1.0,
                delta_flux_density_t: 0.1,
            },
        ];
        let values = law.evaluate(0.5, 0.25).unwrap();
        assert!((values.effective_flux_density_t - 0.30).abs() < 1.0e-12);
        let decoded: MotorMagneticLaw =
            serde_json::from_slice(&serde_json::to_vec(&law).unwrap()).unwrap();
        assert_eq!(decoded, law);

        let mut malformed = law;
        malformed.flux_modulation_curve[1].delta_flux_density_t = 1.0e-6;
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn magnetic_law_rejects_malformed_curves_and_nonphysical_geometry() {
        let mut malformed = magnetic_law();
        malformed.bh_curve[2].field_strength_a_per_m = malformed.bh_curve[1].field_strength_a_per_m;
        assert!(malformed.validate().is_err());

        let mut invalid_origin = magnetic_law();
        invalid_origin.bh_curve[0].flux_density_t = 1.0e-6;
        assert!(invalid_origin.validate().is_err());

        let mut invalid_geometry = magnetic_law();
        invalid_geometry
            .reluctance
            .as_mut()
            .unwrap()
            .effective_gap_area_m2 = 0.0;
        assert!(invalid_geometry.validate().is_err());
    }

    #[test]
    fn magnetic_operating_evidence_round_trips_and_rejects_tampering() {
        let mut motor = complete_domain_model();
        motor.magnetic = Some(magnetic_law());
        let mut legacy_model = serde_json::to_value(&motor).unwrap();
        legacy_model.as_object_mut().unwrap().remove("magnetic");
        let decoded_legacy: MotorConstitutiveModel = serde_json::from_value(legacy_model).unwrap();
        assert!(decoded_legacy.magnetic.is_none());

        let evidence = motor.validate_operating_domain(&complete_domain()).unwrap();
        let encoded = serde_json::to_vec(&evidence).unwrap();
        let decoded: MotorOperatingDomainEvidence = serde_json::from_slice(&encoded).unwrap();
        motor.validate_operating_domain_evidence(&decoded).unwrap();
        assert!(
            decoded
                .values
                .iter()
                .all(|values| values.magnetic.is_some())
        );

        let mut tampered = decoded;
        tampered.values[0]
            .magnetic
            .as_mut()
            .unwrap()
            .differential_permeability_h_per_m *= 1.1;
        assert!(motor.validate_operating_domain_evidence(&tampered).is_err());
    }

    #[test]
    fn complete_le_tensor_cell_is_trilinearly_interpolated() {
        let mut points = Vec::new();
        for (x_index, displacement_m) in [-0.01, 0.01].into_iter().enumerate() {
            for (current_index, current_a) in [0.0, 2.0].into_iter().enumerate() {
                for (frequency_index, frequency_hz) in [100.0, 1_000.0].into_iter().enumerate() {
                    points.push(NonlinearLePoint {
                        displacement_m,
                        current_a,
                        frequency_hz,
                        inductance_h: 1.0e-3
                            + x_index as f64 * 1.0e-4
                            + current_index as f64 * 2.0e-4
                            + frequency_index as f64 * 3.0e-4,
                        eddy_resistance_ohm: 0.1
                            + x_index as f64 * 0.1
                            + current_index as f64 * 0.2
                            + frequency_index as f64 * 0.3,
                    });
                }
            }
        }

        let (inductance_h, resistance_ohm) = interpolate_le(&points, 0.0, 1.0, 550.0);
        assert!((inductance_h - 1.3e-3).abs() < 1.0e-15);
        assert!((resistance_ohm - 0.4).abs() < 1.0e-15);
    }

    #[test]
    fn tensor_grid_matches_floating_extraction_coordinates() {
        let mut points = Vec::new();
        for (x_index, displacement_m) in [-0.01, 0.01].into_iter().enumerate() {
            for (current_index, current_a) in [0.0, 2.0].into_iter().enumerate() {
                for (frequency_index, frequency_hz) in [100.0, 1_000.0].into_iter().enumerate() {
                    points.push(NonlinearLePoint {
                        displacement_m: displacement_m
                            + (x_index + current_index + frequency_index) as f64 * 2.0e-13,
                        current_a: current_a
                            + (x_index + current_index + frequency_index) as f64 * 2.0e-13,
                        frequency_hz: frequency_hz
                            + (x_index + current_index + frequency_index) as f64 * 2.0e-10,
                        inductance_h: 1.0e-3
                            + x_index as f64 * 1.0e-4
                            + current_index as f64 * 2.0e-4
                            + frequency_index as f64 * 3.0e-4,
                        eddy_resistance_ohm: 0.1
                            + x_index as f64 * 0.1
                            + current_index as f64 * 0.2
                            + frequency_index as f64 * 0.3,
                    });
                }
            }
        }

        let (inductance_h, resistance_ohm) = interpolate_le(&points, 0.0, 1.0, 550.0);
        assert!((inductance_h - 1.3e-3).abs() < 1.0e-12);
        assert!((resistance_ohm - 0.4).abs() < 1.0e-9);
    }

    #[test]
    fn operating_point_uses_position_current_frequency_and_temperature() {
        let motor = model();
        let cold = motor
            .evaluate(MotorOperatingPoint {
                displacement_m: 0.0,
                current_a: 0.0,
                frequency_hz: 100.0,
                temperature_k: 293.15,
                flux_density_t: 0.8,
            })
            .unwrap();
        let hot = motor
            .evaluate(MotorOperatingPoint {
                displacement_m: 0.001,
                current_a: 2.0,
                frequency_hz: 1_000.0,
                temperature_k: 343.15,
                flux_density_t: 0.8,
            })
            .unwrap();
        assert_ne!(cold.inductance_h, hot.inductance_h);
        assert!(hot.copper_resistance_ohm > cold.copper_resistance_ohm);
        assert!(hot.total_magnetic_loss_w > cold.total_magnetic_loss_w);
    }

    #[test]
    fn hysteresis_loss_is_frequency_and_flux_dependent() {
        let law = HysteresisLossModel {
            coefficient_w_per_m3: 1.0,
            active_volume_m3: 2.0,
            ..HysteresisLossModel::default()
        };
        assert!(law.loss_w(200.0, 1.0).unwrap() > law.loss_w(100.0, 1.0).unwrap());
        assert_eq!(law.loss_w(100.0, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn passive_temperature_law_rejects_negative_inductance() {
        let mut motor = model();
        motor.temperature.inductance_per_k = -0.1;
        assert!(
            motor
                .evaluate(MotorOperatingPoint {
                    displacement_m: 0.0,
                    current_a: 0.0,
                    frequency_hz: 100.0,
                    temperature_k: 400.0,
                    flux_density_t: 0.8
                })
                .is_err()
        );
    }

    fn complete_domain_model() -> MotorConstitutiveModel {
        let mut motor = model();
        motor.bl_curve = vec![
            NonlinearBlPoint {
                displacement_m: -0.001,
                force_factor_tm: 5.0,
            },
            NonlinearBlPoint {
                displacement_m: 0.0,
                force_factor_tm: 6.0,
            },
            NonlinearBlPoint {
                displacement_m: 0.001,
                force_factor_tm: 5.0,
            },
        ];
        motor.le_surface.clear();
        for (x_index, displacement_m) in [-0.001, 0.001].into_iter().enumerate() {
            for (current_index, current_a) in [0.0, 2.0].into_iter().enumerate() {
                for (frequency_index, frequency_hz) in [100.0, 1_000.0].into_iter().enumerate() {
                    motor.le_surface.push(NonlinearLePoint {
                        displacement_m,
                        current_a,
                        frequency_hz,
                        inductance_h: 0.4e-3 + x_index as f64 * 0.05e-3
                            - current_index as f64 * 0.04e-3
                            + frequency_index as f64 * 0.02e-3,
                        eddy_resistance_ohm: 0.1
                            + x_index as f64 * 0.05
                            + current_index as f64 * 0.2
                            + frequency_index as f64 * 0.1,
                    });
                }
            }
        }
        motor
    }

    fn complete_domain() -> MotorOperatingDomain {
        MotorOperatingDomain {
            displacement_m: vec![-0.001, 0.001],
            current_a: vec![0.0, 2.0],
            frequency_hz: vec![100.0, 1_000.0],
            temperature_k: vec![293.15, 343.15],
            flux_density_t: vec![0.5, 0.8],
        }
    }

    #[test]
    fn operating_domain_retains_complete_passive_grid_evidence() {
        let motor = complete_domain_model();
        let domain = complete_domain();
        assert_eq!(domain.point_count().unwrap(), 32);
        let evidence = motor.validate_operating_domain(&domain).unwrap();
        assert_eq!(
            evidence.schema_version,
            MOTOR_OPERATING_DOMAIN_EVIDENCE_SCHEMA_VERSION
        );
        assert_eq!(evidence.points.len(), 32);
        assert_eq!(evidence.values.len(), 32);
        evidence.validate().unwrap();
        motor.validate_operating_domain_evidence(&evidence).unwrap();
        let decoded: MotorOperatingDomainEvidence =
            serde_json::from_slice(&serde_json::to_vec(&evidence).unwrap()).unwrap();
        assert_eq!(decoded.schema_version, evidence.schema_version);
        assert_eq!(decoded.domain, evidence.domain);
        assert_eq!(decoded.points, evidence.points);
        assert_eq!(decoded.values.len(), evidence.values.len());
        assert!(
            decoded
                .values
                .iter()
                .zip(&evidence.values)
                .all(|(decoded, original)| operating_values_match(*decoded, *original))
        );
        motor.validate_operating_domain_evidence(&decoded).unwrap();
        assert!(evidence.values.iter().all(|value| {
            value.inductance_h >= 0.0
                && value.eddy_resistance_ohm >= 0.0
                && value.copper_resistance_ohm > 0.0
                && value.total_magnetic_loss_w >= 0.0
        }));
    }

    #[test]
    fn operating_domain_rejects_out_of_support_and_incomplete_tensor_cells() {
        let motor = complete_domain_model();
        let mut outside = complete_domain();
        outside.current_a = vec![3.0];
        assert!(motor.validate_operating_domain(&outside).is_err());

        let mut incomplete = complete_domain_model();
        incomplete.le_surface.pop();
        let domain = MotorOperatingDomain {
            displacement_m: vec![0.0],
            current_a: vec![1.0],
            frequency_hz: vec![550.0],
            temperature_k: vec![293.15],
            flux_density_t: vec![0.5],
        };
        assert!(incomplete.validate_operating_domain(&domain).is_err());
    }

    #[test]
    fn operating_domain_rejects_malformed_grid_and_tampered_evidence() {
        let motor = complete_domain_model();
        let mut malformed = complete_domain();
        malformed.current_a = vec![0.0, 0.0];
        assert!(malformed.validate().is_err());

        let mut evidence = motor.validate_operating_domain(&complete_domain()).unwrap();
        evidence.values[0].eddy_resistance_ohm = -1.0;
        assert!(evidence.validate().is_err());

        let mut valid = motor.validate_operating_domain(&complete_domain()).unwrap();
        valid.values[0].inductance_h *= 1.1;
        assert!(motor.validate_operating_domain_evidence(&valid).is_err());
    }

    #[test]
    fn operating_domain_cardinality_is_bounded_without_overflow() {
        let domain = MotorOperatingDomain {
            displacement_m: (0..65).map(|index| index as f64).collect(),
            current_a: (0..65).map(|index| index as f64).collect(),
            frequency_hz: (1..=65).map(|index| index as f64).collect(),
            temperature_k: (293..358).map(|index| index as f64).collect(),
            flux_density_t: (0..65).map(|index| index as f64).collect(),
        };
        assert!(domain.validate().is_err());
    }

    #[test]
    fn hysteresis_fit_recovers_power_law_and_reports_acceptance() {
        let truth = HysteresisLossModel {
            coefficient_w_per_m3: 2.5e-3,
            frequency_exponent: 1.4,
            flux_density_exponent: 2.2,
            active_volume_m3: 3.0e-5,
        };
        let samples: Vec<_> = [40.0, 200.0, 1_000.0]
            .into_iter()
            .flat_map(|frequency_hz| {
                [0.15, 0.45, 1.1]
                    .into_iter()
                    .map(move |flux_density_t| HysteresisLossSample {
                        frequency_hz,
                        flux_density_t,
                        loss_w: truth.loss_w(frequency_hz, flux_density_t).unwrap(),
                    })
            })
            .collect();
        let (fit, report) =
            HysteresisLossModel::fit_from_samples(&samples, truth.active_volume_m3, 1.0e-10)
                .unwrap();

        assert_eq!(report.sample_count, samples.len());
        assert!(report.accepted);
        assert!(report.rms_relative_error < 1.0e-10);
        assert!((fit.coefficient_w_per_m3 - truth.coefficient_w_per_m3).abs() < 1.0e-12);
        assert!((fit.frequency_exponent - truth.frequency_exponent).abs() < 1.0e-12);
        assert!((fit.flux_density_exponent - truth.flux_density_exponent).abs() < 1.0e-12);

        let decoded: HysteresisLossFitReport =
            serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        assert_eq!(decoded.sample_count, report.sample_count);
        assert_eq!(decoded.accepted, report.accepted);
        assert!((decoded.rms_relative_error - report.rms_relative_error).abs() < 1.0e-20);
        assert!((decoded.maximum_relative_error - report.maximum_relative_error).abs() < 1.0e-20);
        assert!((decoded.acceptance_tolerance - report.acceptance_tolerance).abs() < 1.0e-20);
    }

    #[test]
    fn hysteresis_fit_rejects_insufficient_degenerate_and_nonpositive_samples() {
        let insufficient = [
            HysteresisLossSample {
                frequency_hz: 100.0,
                flux_density_t: 0.5,
                loss_w: 1.0,
            },
            HysteresisLossSample {
                frequency_hz: 200.0,
                flux_density_t: 0.5,
                loss_w: 2.0,
            },
        ];
        assert!(HysteresisLossModel::fit_from_samples(&insufficient, 1.0, 0.1).is_err());

        let degenerate = [
            HysteresisLossSample {
                frequency_hz: 100.0,
                flux_density_t: 0.5,
                loss_w: 1.0,
            },
            HysteresisLossSample {
                frequency_hz: 100.0,
                flux_density_t: 0.5,
                loss_w: 2.0,
            },
            HysteresisLossSample {
                frequency_hz: 100.0,
                flux_density_t: 0.5,
                loss_w: 3.0,
            },
        ];
        assert!(HysteresisLossModel::fit_from_samples(&degenerate, 1.0, 0.1).is_err());

        let mut invalid = degenerate;
        invalid[0].loss_w = 0.0;
        assert!(HysteresisLossModel::fit_from_samples(&invalid, 1.0, 0.1).is_err());
    }
}
