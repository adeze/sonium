//! 3D blade-shape optimization for low-noise propeller studies.
//!
//! The optimizer uses deterministic coordinate search over a bounded toroidal
//! serrated blade family. It keeps the blade radius at or below the baseline,
//! holds material mass constant through a uniform thickness scale, and rejects
//! designs with more than the configured thrust loss.

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

use crate::{
    Blade3dConfig, BladeMesh, BladeMeshError, COMBINED_REDUCTION_PRIOR_DB, EdgeTreatment,
    EvaluationMode, LOOP_CLOSURE_FRACTION_RANGE, PanelAcousticSummary, PropellerDesign,
    PropellerMultipointOperatingPoint, PropellerOperatingCondition, PropellerOperatingEnvelope,
    PropellerPlanform, PropellerResultProvenance, PropellerSpectrum,
    SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE, SERRATION_SPAN_FRACTION_RANGE,
    SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE, SerrationGeometry, compare_panel_acoustics,
    sha256_hex,
};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeShapeVariables {
    pub radius_scale: f64,
    pub pitch_root_delta_deg: f64,
    pub pitch_tip_delta_deg: f64,
    pub chord_root_scale: f64,
    pub chord_tip_scale: f64,
    pub loop_closure_fraction: f64,
    pub loop_axial_lift_scale: f64,
    pub serration_amplitude: f64,
    pub serration_wavelength: f64,
    pub serration_span: f64,
}

impl Default for BladeShapeVariables {
    fn default() -> Self {
        Self {
            radius_scale: 0.96,
            pitch_root_delta_deg: 0.0,
            pitch_tip_delta_deg: 1.0,
            chord_root_scale: 1.08,
            chord_tip_scale: 1.18,
            loop_closure_fraction: 0.78,
            loop_axial_lift_scale: 1.0,
            serration_amplitude: 0.018,
            serration_wavelength: 0.055,
            serration_span: 0.72,
        }
    }
}

impl BladeShapeVariables {
    /// Number of independent variables in the bounded blade family.
    pub const COUNT: usize = 10;

    /// Return variables in the stable optimizer ordering used by derivatives.
    pub fn as_array(self) -> [f64; Self::COUNT] {
        [
            self.radius_scale,
            self.pitch_root_delta_deg,
            self.pitch_tip_delta_deg,
            self.chord_root_scale,
            self.chord_tip_scale,
            self.loop_closure_fraction,
            self.loop_axial_lift_scale,
            self.serration_amplitude,
            self.serration_wavelength,
            self.serration_span,
        ]
    }

    fn from_array(values: [f64; Self::COUNT]) -> Self {
        Self {
            radius_scale: values[0],
            pitch_root_delta_deg: values[1],
            pitch_tip_delta_deg: values[2],
            chord_root_scale: values[3],
            chord_tip_scale: values[4],
            loop_closure_fraction: values[5],
            loop_axial_lift_scale: values[6],
            serration_amplitude: values[7],
            serration_wavelength: values[8],
            serration_span: values[9],
        }
    }

    fn shifted(self, direction: [f64; Self::COUNT], step: f64) -> Self {
        let mut values = self.as_array();
        for (value, component) in values.iter_mut().zip(direction) {
            *value += step * component;
        }
        Self::from_array(values)
    }

    fn clamped(self) -> Self {
        Self {
            radius_scale: self.radius_scale.clamp(0.82, 1.0),
            pitch_root_delta_deg: self.pitch_root_delta_deg.clamp(-7.0, 5.0),
            pitch_tip_delta_deg: self.pitch_tip_delta_deg.clamp(-5.0, 7.0),
            chord_root_scale: self.chord_root_scale.clamp(0.78, 1.38),
            chord_tip_scale: self.chord_tip_scale.clamp(0.75, 1.55),
            loop_closure_fraction: self.loop_closure_fraction.clamp(
                LOOP_CLOSURE_FRACTION_RANGE[0],
                LOOP_CLOSURE_FRACTION_RANGE[1],
            ),
            loop_axial_lift_scale: self.loop_axial_lift_scale.clamp(0.55, 1.85),
            serration_amplitude: self.serration_amplitude.clamp(
                SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE[0],
                SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE[1],
            ),
            serration_wavelength: self.serration_wavelength.clamp(
                SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE[0],
                SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE[1],
            ),
            serration_span: self.serration_span.clamp(
                SERRATION_SPAN_FRACTION_RANGE[0],
                SERRATION_SPAN_FRACTION_RANGE[1],
            ),
        }
    }

    fn perturb(self, index: usize, delta: f64) -> Self {
        let mut next = self;
        match index {
            0 => next.radius_scale += delta,
            1 => next.pitch_root_delta_deg += delta,
            2 => next.pitch_tip_delta_deg += delta,
            3 => next.chord_root_scale += delta,
            4 => next.chord_tip_scale += delta,
            5 => next.loop_closure_fraction += delta,
            6 => next.loop_axial_lift_scale += delta,
            7 => next.serration_amplitude += delta,
            8 => next.serration_wavelength += delta,
            9 => next.serration_span += delta,
            _ => {}
        }
        next.clamped()
    }
}

#[derive(Debug, Clone)]
pub struct Propeller3dOptConfig {
    pub source: PropellerSource,
    pub mesh: Blade3dConfig,
    pub initial_variables: BladeShapeVariables,
    pub n_harmonics: usize,
    pub max_iterations: usize,
    pub max_thrust_loss: f64,
    /// Upper validity boundary of the empirical combined-attenuation prior.
    /// Candidates beyond it are surrogate extrapolations and are not selected.
    pub max_calibrated_reduction_db: f64,
    pub peak_spread_weight: f64,
    pub thrust_penalty_weight: f64,
    pub thickness_penalty_weight: f64,
    /// Weight for the soft penalty on the worst normalized multipoint power
    /// margin. Hard admission remains fail-closed independently of this term.
    pub operating_point_margin_weight: f64,
    /// Hard lower manufacturing bound on the thickness scale used to preserve
    /// constant blade weight relative to the conventional baseline.
    pub min_thickness_scale_for_constant_weight: f64,
    /// Hard upper manufacturing bound on the thickness scale used to preserve
    /// constant blade weight relative to the conventional baseline.
    pub max_thickness_scale_for_constant_weight: f64,
    /// Optional off-design conditions that every promoted candidate must
    /// admit under the explicit multipoint load surrogate. Empty preserves
    /// the historical single-operating-point optimization path.
    pub operating_conditions: Vec<PropellerOperatingCondition>,
}

/// Invalid public configuration for the deterministic propeller optimizer.
#[derive(Debug, Clone, PartialEq)]
pub enum Propeller3dOptError {
    /// The requested blade mesh is not physically or numerically admissible.
    Mesh(BladeMeshError),
    /// A scalar optimization setting is invalid.
    InvalidConfiguration {
        /// Setting name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A finite-difference derivative probe could not produce a valid result.
    DerivativeEvaluation {
        /// Variable index in the stable `BladeShapeVariables::as_array` order.
        variable: usize,
        /// Required derivative-probe condition.
        requirement: &'static str,
    },
}

impl fmt::Display for Propeller3dOptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mesh(error) => write!(f, "invalid propeller optimizer mesh: {error}"),
            Self::InvalidConfiguration { field, requirement } => {
                write!(f, "propeller optimizer {field} must be {requirement}")
            }
            Self::DerivativeEvaluation {
                variable,
                requirement,
            } => write!(
                f,
                "propeller optimizer derivative variable {variable} must be {requirement}"
            ),
        }
    }
}

impl std::error::Error for Propeller3dOptError {}

impl From<BladeMeshError> for Propeller3dOptError {
    fn from(error: BladeMeshError) -> Self {
        Self::Mesh(error)
    }
}

impl Default for Propeller3dOptConfig {
    fn default() -> Self {
        Self {
            source: PropellerSource::drone_5inch(18_000.0),
            mesh: Blade3dConfig {
                radial_segments: 128,
                chord_segments: 28,
                loop_segments: 64,
                ..Blade3dConfig::default()
            },
            initial_variables: BladeShapeVariables::default(),
            n_harmonics: 20,
            max_iterations: 8,
            max_thrust_loss: 0.05,
            max_calibrated_reduction_db: COMBINED_REDUCTION_PRIOR_DB.max_db,
            peak_spread_weight: 0.45,
            thrust_penalty_weight: 250.0,
            thickness_penalty_weight: 20.0,
            operating_point_margin_weight: 1.0,
            min_thickness_scale_for_constant_weight: 0.55,
            max_thickness_scale_for_constant_weight: 1.85,
            operating_conditions: Vec::new(),
        }
    }
}

impl Propeller3dOptConfig {
    /// Validate externally supplied geometry and objective settings.
    pub fn validate(&self) -> Result<(), Propeller3dOptError> {
        if self.source.n_blades == 0 {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "source.n_blades",
                requirement: "non-zero",
            });
        }
        for (field, value) in [
            ("source.rpm", self.source.rpm),
            ("source.thrust", self.source.thrust),
            ("source.torque", self.source.torque),
            ("source.tip_radius", self.source.tip_radius),
            ("source.hub_radius", self.source.hub_radius),
            ("source.speed_of_sound", self.source.speed_of_sound),
            ("source.density", self.source.density),
        ] {
            if !value.is_finite() {
                return Err(Propeller3dOptError::InvalidConfiguration {
                    field,
                    requirement: "finite",
                });
            }
        }
        if self.source.rpm <= 0.0
            || self.source.thrust < 0.0
            || self.source.torque <= 0.0
            || self.source.hub_radius <= 0.0
            || self.source.tip_radius <= self.source.hub_radius
            || self.source.speed_of_sound <= 0.0
            || self.source.density <= 0.0
        {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "source",
                requirement: "positive RPM/torque/fluid values and 0 < hub_radius < tip_radius with non-negative thrust",
            });
        }
        self.mesh
            .validate(&PropellerDesign::conventional(self.source.clone()))?;
        if !self.operating_conditions.is_empty() {
            let envelope = PropellerOperatingEnvelope::try_from_source(self.source.clone())
                .map_err(|_| Propeller3dOptError::InvalidConfiguration {
                    field: "operating_conditions",
                    requirement: "admitted by the source operating envelope",
                })?;
            for condition in &self.operating_conditions {
                if envelope.evaluate_condition(*condition).is_err() {
                    return Err(Propeller3dOptError::InvalidConfiguration {
                        field: "operating_conditions",
                        requirement: "finite, physical, and within source shaft-power bounds",
                    });
                }
            }
        }
        if self.n_harmonics == 0 {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "n_harmonics",
                requirement: "non-zero",
            });
        }
        if self.max_iterations == 0 {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "max_iterations",
                requirement: "non-zero",
            });
        }
        if !self.max_thrust_loss.is_finite() || !(0.0..1.0).contains(&self.max_thrust_loss) {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "max_thrust_loss",
                requirement: "finite and within [0, 1)",
            });
        }
        for (field, value) in [
            (
                "max_calibrated_reduction_db",
                self.max_calibrated_reduction_db,
            ),
            ("peak_spread_weight", self.peak_spread_weight),
            ("thrust_penalty_weight", self.thrust_penalty_weight),
            ("thickness_penalty_weight", self.thickness_penalty_weight),
            (
                "operating_point_margin_weight",
                self.operating_point_margin_weight,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Propeller3dOptError::InvalidConfiguration {
                    field,
                    requirement: "finite and non-negative",
                });
            }
        }
        if !self.min_thickness_scale_for_constant_weight.is_finite()
            || !self.max_thickness_scale_for_constant_weight.is_finite()
            || self.min_thickness_scale_for_constant_weight <= 0.0
            || self.min_thickness_scale_for_constant_weight
                > self.max_thickness_scale_for_constant_weight
        {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "min_thickness_scale_for_constant_weight/max_thickness_scale_for_constant_weight",
                requirement: "finite, positive, and ordered",
            });
        }
        for (field, value) in [
            ("radius_scale", self.initial_variables.radius_scale),
            (
                "pitch_root_delta_deg",
                self.initial_variables.pitch_root_delta_deg,
            ),
            (
                "pitch_tip_delta_deg",
                self.initial_variables.pitch_tip_delta_deg,
            ),
            ("chord_root_scale", self.initial_variables.chord_root_scale),
            ("chord_tip_scale", self.initial_variables.chord_tip_scale),
            (
                "loop_closure_fraction",
                self.initial_variables.loop_closure_fraction,
            ),
            (
                "loop_axial_lift_scale",
                self.initial_variables.loop_axial_lift_scale,
            ),
            (
                "serration_amplitude",
                self.initial_variables.serration_amplitude,
            ),
            (
                "serration_wavelength",
                self.initial_variables.serration_wavelength,
            ),
            ("serration_span", self.initial_variables.serration_span),
        ] {
            if !value.is_finite() {
                return Err(Propeller3dOptError::InvalidConfiguration {
                    field,
                    requirement: "finite",
                });
            }
        }
        Ok(())
    }

    fn candidate_is_feasible(&self, candidate: &OptimizedBlade) -> bool {
        candidate.objective.is_finite()
            && candidate.thrust_loss.is_finite()
            && candidate.thrust_loss <= self.max_thrust_loss
            && candidate
                .panel_summary
                .sensitive_band_reduction_db
                .is_finite()
            && candidate.panel_summary.sensitive_band_reduction_db
                <= self.max_calibrated_reduction_db
            && candidate.thickness_scale_for_constant_weight.is_finite()
            && (self.min_thickness_scale_for_constant_weight
                ..=self.max_thickness_scale_for_constant_weight)
                .contains(&candidate.thickness_scale_for_constant_weight)
            && candidate.loads.is_physically_admissible()
            && candidate.provenance.is_bounded_surrogate()
            && candidate.minimum_power_margin_ratio.is_finite()
            && candidate.minimum_power_margin_ratio >= 0.0
            && candidate.actuator_disk_reference_is_consistent()
            && (self.operating_conditions.is_empty()
                || (candidate.operating_points.len() == self.operating_conditions.len()
                    && candidate.operating_points.iter().all(|point| {
                        point.power_margin_w.is_finite() && point.power_margin_w >= -1.0e-10
                    })))
    }
}

/// Load, power, and acoustic-power bookkeeping for one surrogate evaluation.
///
/// This is a closure record for the declared source model, not a claim that a
/// panel/free-wake/RANS solution has been resolved. In particular, the ideal
/// hover power is actuator-disk theory and exists to reject impossible source
/// combinations before they are promoted or exported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerLoadPowerSummary {
    /// Source thrust [N].
    pub thrust_n: f64,
    /// Source torque [N m].
    pub torque_nm: f64,
    /// Rotation speed magnitude [rad/s].
    pub angular_speed_rad_s: f64,
    /// Mechanical shaft input, `Q * |omega|` [W].
    pub shaft_power_w: f64,
    /// Annulus area used by the hover reference [m²].
    pub disk_area_m2: f64,
    /// Ideal actuator-disk hover power, `T^(3/2) / sqrt(2 rho A)` [W].
    pub ideal_hover_power_w: f64,
    /// Ideal-hover power divided by shaft power.
    pub ideal_hover_efficiency: f64,
    /// Sum of isotropic harmonic acoustic powers reconstructed from the
    /// modified peak pressures [W].
    pub acoustic_power_w: f64,
    /// Acoustic power divided by shaft power.
    pub acoustic_to_shaft_power_ratio: f64,
    /// Relative disagreement between the pressure-derived power and the
    /// source-power normalization after each harmonic reduction is applied.
    pub acoustic_power_closure_error: f64,
    /// Absolute disagreement [W] behind the relative closure error, so
    /// near-silent spectra can be judged by absolute power.
    pub acoustic_power_abs_diff_w: f64,
}

/// Independent actuator-disk reference for a static/hover propeller load.
///
/// This is intentionally a low-fidelity physical oracle: it uses the annular
/// disk area and prescribed thrust, not the blade-panel surrogate. It is
/// useful for rejecting an internally inconsistent source ledger, but it does
/// not validate a resolved wake or forward-flight operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerActuatorDiskReference {
    /// Prescribed total thrust [N].
    pub thrust_n: f64,
    /// Fluid density [kg/m³].
    pub density_kg_m3: f64,
    /// Annular disk area [m²].
    pub disk_area_m2: f64,
    /// Ideal induced velocity through the disk [m/s].
    pub induced_velocity_m_s: f64,
    /// Ideal pressure jump across the disk [Pa].
    pub pressure_jump_pa: f64,
    /// Ideal actuator-disk power, `T * v_i` [W].
    pub ideal_hover_power_w: f64,
}

/// Failure of the bounded actuator-disk admission contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropellerActuatorDiskReferenceError {
    /// A source or derived reference quantity is not physical.
    InvalidInput {
        /// Source or derived field that failed admission.
        field: &'static str,
        /// Required physical domain.
        requirement: &'static str,
    },
    /// The requested comparison tolerance is invalid.
    InvalidTolerance,
    /// A production load summary disagrees with the independent reference.
    Mismatch {
        /// Quantity that failed the comparison.
        field: &'static str,
        /// Relative disagreement using the larger magnitude as scale.
        relative_error: f64,
        /// Accepted relative tolerance.
        tolerance: f64,
    },
}

impl fmt::Display for PropellerActuatorDiskReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, requirement } => {
                write!(f, "actuator-disk reference {field} must be {requirement}")
            }
            Self::InvalidTolerance => write!(
                f,
                "actuator-disk reference tolerance must be finite and non-negative"
            ),
            Self::Mismatch {
                field,
                relative_error,
                tolerance,
            } => write!(
                f,
                "actuator-disk reference mismatch for {field}: relative error {relative_error:e} exceeds {tolerance:e}"
            ),
        }
    }
}

impl std::error::Error for PropellerActuatorDiskReferenceError {}

impl PropellerActuatorDiskReference {
    /// Relative tolerance used by production candidate promotion.
    pub const PRODUCTION_TOLERANCE: f64 = 1.0e-12;

    /// Construct the independent annular actuator-disk reference.
    pub fn try_from_source(
        source: &PropellerSource,
    ) -> Result<Self, PropellerActuatorDiskReferenceError> {
        for (field, value) in [
            ("source.thrust", source.thrust),
            ("source.density", source.density),
            ("source.hub_radius", source.hub_radius),
            ("source.tip_radius", source.tip_radius),
        ] {
            if !value.is_finite() {
                return Err(PropellerActuatorDiskReferenceError::InvalidInput {
                    field,
                    requirement: "finite",
                });
            }
        }
        if source.thrust < 0.0 {
            return Err(PropellerActuatorDiskReferenceError::InvalidInput {
                field: "source.thrust",
                requirement: "non-negative",
            });
        }
        if source.density <= 0.0 {
            return Err(PropellerActuatorDiskReferenceError::InvalidInput {
                field: "source.density",
                requirement: "positive",
            });
        }
        if source.hub_radius <= 0.0 || source.tip_radius <= source.hub_radius {
            return Err(PropellerActuatorDiskReferenceError::InvalidInput {
                field: "source.hub_radius/source.tip_radius",
                requirement: "0 < hub_radius < tip_radius",
            });
        }

        // Annular actuator-disk area, deliberately kept as a separate
        // derivation from PropellerLoadPowerSummary::from_design_and_spectrum.
        let disk_area_m2 =
            std::f64::consts::PI * (source.tip_radius.powi(2) - source.hub_radius.powi(2));
        if !disk_area_m2.is_finite() || disk_area_m2 <= 0.0 {
            return Err(PropellerActuatorDiskReferenceError::InvalidInput {
                field: "disk_area_m2",
                requirement: "finite and positive",
            });
        }

        // Momentum balance: T = 2 rho A v_i² and Δp = T/A.
        let induced_velocity_m_s = (source.thrust / (2.0 * source.density * disk_area_m2)).sqrt();
        let pressure_jump_pa = source.thrust / disk_area_m2;
        let ideal_hover_power_w = source.thrust * induced_velocity_m_s;
        if !induced_velocity_m_s.is_finite()
            || !pressure_jump_pa.is_finite()
            || !ideal_hover_power_w.is_finite()
        {
            return Err(PropellerActuatorDiskReferenceError::InvalidInput {
                field: "derived actuator-disk quantities",
                requirement: "finite",
            });
        }

        Ok(Self {
            thrust_n: source.thrust,
            density_kg_m3: source.density,
            disk_area_m2,
            induced_velocity_m_s,
            pressure_jump_pa,
            ideal_hover_power_w,
        })
    }

    /// Compare a production load ledger against the independent reference.
    pub fn validate_load_summary(
        self,
        loads: PropellerLoadPowerSummary,
        tolerance: f64,
    ) -> Result<(), PropellerActuatorDiskReferenceError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(PropellerActuatorDiskReferenceError::InvalidTolerance);
        }
        if !loads.shaft_power_w.is_finite() || loads.shaft_power_w <= 0.0 {
            return Err(PropellerActuatorDiskReferenceError::InvalidInput {
                field: "loads.shaft_power_w",
                requirement: "finite and positive",
            });
        }

        for (field, expected, actual) in [
            ("thrust_n", self.thrust_n, loads.thrust_n),
            ("disk_area_m2", self.disk_area_m2, loads.disk_area_m2),
            (
                "ideal_hover_power_w",
                self.ideal_hover_power_w,
                loads.ideal_hover_power_w,
            ),
            (
                "ideal_hover_efficiency",
                self.ideal_hover_power_w / loads.shaft_power_w,
                loads.ideal_hover_efficiency,
            ),
        ] {
            let scale = expected.abs().max(actual.abs()).max(f64::MIN_POSITIVE);
            let relative_error = (actual - expected).abs() / scale;
            if !relative_error.is_finite() || relative_error > tolerance {
                return Err(PropellerActuatorDiskReferenceError::Mismatch {
                    field,
                    relative_error,
                    tolerance,
                });
            }
        }

        // Independent pressure/area/velocity work identity.
        let pressure_power_w =
            self.pressure_jump_pa * self.disk_area_m2 * self.induced_velocity_m_s;
        let scale = self
            .ideal_hover_power_w
            .abs()
            .max(pressure_power_w.abs())
            .max(f64::MIN_POSITIVE);
        let relative_error = (pressure_power_w - self.ideal_hover_power_w).abs() / scale;
        if !relative_error.is_finite() || relative_error > tolerance {
            return Err(PropellerActuatorDiskReferenceError::Mismatch {
                field: "pressure_jump_area_velocity_power",
                relative_error,
                tolerance,
            });
        }
        Ok(())
    }
}

impl PropellerLoadPowerSummary {
    fn from_design_and_spectrum(design: &PropellerDesign, spectrum: &PropellerSpectrum) -> Self {
        let source = &design.source;
        let angular_speed_rad_s = source.omega().abs();
        let thrust_n = source.thrust;
        let torque_nm = source.torque.abs();
        let shaft_power_w = torque_nm * angular_speed_rad_s;
        let disk_area_m2 = std::f64::consts::PI
            * (source.tip_radius * source.tip_radius - source.hub_radius * source.hub_radius);
        let ideal_hover_power_w = if thrust_n >= 0.0 && source.density > 0.0 && disk_area_m2 > 0.0 {
            thrust_n.powf(1.5) / (2.0 * source.density * disk_area_m2).sqrt()
        } else {
            f64::NAN
        };
        let impedance = source.density * source.speed_of_sound;
        let acoustic_power_w = if impedance > 0.0 {
            spectrum
                .harmonics
                .iter()
                .map(|harmonic| {
                    2.0 * std::f64::consts::PI * harmonic.pressure_pa.powi(2) / impedance
                })
                .sum()
        } else {
            f64::NAN
        };
        let source_reduced_power_w: f64 = source
            .harmonic_sound_powers(spectrum.harmonics.len())
            .into_iter()
            .zip(&spectrum.harmonics)
            .map(|(power, harmonic)| power * harmonic.amplitude_reduction.powi(2))
            .sum();
        let acoustic_power_abs_diff = (acoustic_power_w - source_reduced_power_w).abs();
        // Absolute floor so near-silent spectra agree by absolute power
        // instead of failing on an enormous relative error.
        let acoustic_power_closure_error =
            acoustic_power_abs_diff / source_reduced_power_w.max(1e-12);

        Self {
            thrust_n,
            torque_nm,
            angular_speed_rad_s,
            shaft_power_w,
            disk_area_m2,
            ideal_hover_power_w,
            ideal_hover_efficiency: ideal_hover_power_w / shaft_power_w,
            acoustic_power_w,
            acoustic_to_shaft_power_ratio: acoustic_power_w / shaft_power_w,
            acoustic_power_closure_error,
            acoustic_power_abs_diff_w: acoustic_power_abs_diff,
        }
    }

    /// Whether the declared source load/power ledger is internally physical.
    /// This does not imply a resolved aerodynamic-flow validation.
    pub fn is_physically_admissible(self) -> bool {
        self.thrust_n.is_finite()
            && self.thrust_n >= 0.0
            && self.torque_nm.is_finite()
            && self.torque_nm > 0.0
            && self.angular_speed_rad_s.is_finite()
            && self.angular_speed_rad_s > 0.0
            && self.shaft_power_w.is_finite()
            && self.shaft_power_w > 0.0
            && self.disk_area_m2.is_finite()
            && self.disk_area_m2 > 0.0
            && self.ideal_hover_power_w.is_finite()
            && self.ideal_hover_efficiency.is_finite()
            && (0.0..=1.0).contains(&self.ideal_hover_efficiency)
            && self.acoustic_power_w.is_finite()
            && self.acoustic_power_w >= 0.0
            && self.acoustic_to_shaft_power_ratio.is_finite()
            && (0.0..=1.0).contains(&self.acoustic_to_shaft_power_ratio)
            && self.acoustic_power_closure_error.is_finite()
            && (self.acoustic_power_closure_error <= 1.0e-12
                || self.acoustic_power_abs_diff_w <= 1e-12)
    }
}

#[derive(Debug, Clone)]
pub struct OptimizedBlade {
    pub variables: BladeShapeVariables,
    pub design: PropellerDesign,
    pub mesh_config: Blade3dConfig,
    pub mesh: BladeMesh,
    pub spectrum: PropellerSpectrum,
    pub panel_summary: PanelAcousticSummary,
    /// Auditable source load/power and acoustic-power closure record.
    pub loads: PropellerLoadPowerSummary,
    /// Typed model-fidelity and claim flags for the result.
    pub provenance: PropellerResultProvenance,
    /// Optional robust-design operating points evaluated for this candidate.
    /// Empty means the legacy single-point optimization contract was used.
    pub operating_points: Vec<PropellerMultipointOperatingPoint>,
    /// Smallest `power_margin_w / shaft_power_w` across admitted points.
    pub minimum_power_margin_ratio: f64,
    pub objective: f64,
    pub thrust_loss: f64,
    pub thickness_scale_for_constant_weight: f64,
    pub peak_spread_delta_db: f64,
    /// Whether the returned candidate satisfies all hard promotion bounds.
    pub feasible: bool,
    pub iterations: usize,
}

impl OptimizedBlade {
    /// Whether this result satisfies the hard optimization promotion bounds.
    pub fn is_feasible(&self) -> bool {
        self.feasible
    }

    /// Whether this surrogate result may be exported as a checked production
    /// artifact. Gallery/render callers must use this rather than trusting
    /// that the optimizer was the producer of the supplied struct.
    pub fn is_exportable(&self) -> bool {
        self.feasible
            && self.objective.is_finite()
            && self.thrust_loss.is_finite()
            && self.thickness_scale_for_constant_weight.is_finite()
            && self.panel_summary.sensitive_band_reduction_db.is_finite()
            && self.loads.is_physically_admissible()
            && self.provenance.is_bounded_surrogate()
            && self.output_checksum_matches()
            && self.minimum_power_margin_ratio.is_finite()
            && self.minimum_power_margin_ratio >= 0.0
            && self.actuator_disk_reference_is_consistent()
            && self.operating_points.iter().all(|point| {
                point.freestream_m_s.is_finite()
                    && point.axial_freestream_m_s.is_finite()
                    && point.rpm.is_finite()
                    && point.rpm > 0.0
                    && point.density_kg_m3.is_finite()
                    && point.density_kg_m3 > 0.0
                    && point.temperature_k.is_finite()
                    && point.temperature_k > 0.0
                    && point.speed_of_sound_m_s.is_finite()
                    && point.speed_of_sound_m_s > 0.0
                    && point.load_scale.is_finite()
                    && point.load_scale > 0.0
                    && point.thrust_n.is_finite()
                    && point.thrust_n >= 0.0
                    && point.torque_nm.is_finite()
                    && point.torque_nm > 0.0
                    && point.shaft_power_w.is_finite()
                    && point.shaft_power_w > 0.0
                    && point.ideal_actuator_power_w.is_finite()
                    && point.ideal_actuator_power_w > 0.0
                    && point.ideal_actuator_efficiency.is_finite()
                    && (0.0..=1.0).contains(&point.ideal_actuator_efficiency)
                    && point.propulsive_efficiency.is_finite()
                    && (0.0..=1.0).contains(&point.propulsive_efficiency)
                    && point.power_margin_w.is_finite()
                    && point.power_margin_w >= -1.0e-10
            })
            && self.mesh.panels.iter().all(|panel| {
                panel.area.is_finite()
                    && panel.area > 0.0
                    && panel.radius_fraction.is_finite()
                    && panel.chord_fraction.is_finite()
            })
            && self.spectrum.harmonics.iter().all(|harmonic| {
                harmonic.frequency_hz.is_finite()
                    && harmonic.frequency_hz > 0.0
                    && harmonic.pressure_pa.is_finite()
                    && harmonic.pressure_pa >= 0.0
            })
    }

    fn actuator_disk_reference_is_consistent(&self) -> bool {
        PropellerActuatorDiskReference::try_from_source(&self.design.source)
            .and_then(|reference| {
                reference.validate_load_summary(
                    self.loads,
                    PropellerActuatorDiskReference::PRODUCTION_TOLERANCE,
                )
            })
            .is_ok()
    }

    fn output_checksum_matches(&self) -> bool {
        let output = output_descriptor(
            &self.design,
            &self.mesh,
            &self.spectrum,
            self.loads,
            self.panel_summary,
            self.objective,
        );
        sha256_hex(output.as_bytes()) == self.provenance.execution.output_checksum_sha256
    }
}

/// Per-variable trust-region scales in [`BladeShapeVariables::as_array`] order.
///
/// Raw design units mix degrees (pitch deltas, bound width 12) with
/// fractions (serration amplitude, bound width 0.029); an unweighted
/// Euclidean metric would let a 1-degree pitch step dominate a full-range
/// serration step. Dividing each component by its clamp bound width keeps
/// every variable on a unitless ~[0, 1] footing so the radius means the
/// same fractional design move in every direction.
pub const TRUST_REGION_SCALES: [f64; BladeShapeVariables::COUNT] = [
    0.18,  // radius_scale: 1.00 - 0.82
    12.0,  // pitch_root_delta_deg: 5 - (-7)
    12.0,  // pitch_tip_delta_deg: 7 - (-5)
    0.60,  // chord_root_scale: 1.38 - 0.78
    0.80,  // chord_tip_scale: 1.55 - 0.75
    0.60,  // loop_closure_fraction: 0.95 - 0.35
    1.30,  // loop_axial_lift_scale: 1.85 - 0.55
    0.029, // serration_amplitude: 0.035 - 0.006
    0.085, // serration_wavelength: 0.120 - 0.035
    0.55,  // serration_span: 0.95 - 0.40
];

/// Validate a proposed shape step against an explicit Euclidean trust region.
///
/// The check uses unclamped variables so callers cannot mistake a bound-clamped
/// proposal for a smaller accepted step. Components are normalized by
/// [`TRUST_REGION_SCALES`] (bound widths) so mixed units do not bias the
/// radius toward the degree-valued pitch deltas.
pub fn validate_shape_trust_region(
    from: BladeShapeVariables,
    to: BladeShapeVariables,
    max_distance: f64,
) -> Result<f64, Propeller3dOptError> {
    if !max_distance.is_finite() || max_distance < 0.0 {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "trust_region_radius",
            requirement: "finite and non-negative",
        });
    }
    let from_values = from.as_array();
    let to_values = to.as_array();
    if from_values
        .iter()
        .chain(to_values.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "trust_region_variables",
            requirement: "finite",
        });
    }

    let distance = from_values
        .iter()
        .zip(to_values)
        .zip(TRUST_REGION_SCALES)
        .map(|((from, to), scale)| (to - from) / scale)
        .map(|delta| delta * delta)
        .sum::<f64>()
        .sqrt();
    if !distance.is_finite() {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "trust_region_variables",
            requirement: "a finite realizable step",
        });
    }
    let roundoff = 32.0 * f64::EPSILON * (1.0 + max_distance);
    if distance > max_distance + roundoff {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "trust_region_step",
            requirement: "within the configured Euclidean radius",
        });
    }
    Ok(distance)
}

/// Return the deterministic Pareto frontier of exportable surrogate results.
///
/// Lower objective is preferred; a larger minimum power margin is preferred.
/// Input order is retained for non-dominated results. This is a screening
/// utility over the bounded surrogate and does not compare model fidelity.
pub fn pareto_frontier(candidates: &[OptimizedBlade]) -> Result<Vec<usize>, Propeller3dOptError> {
    for result in candidates {
        if !result.is_exportable() {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "pareto_candidates",
                requirement: "every candidate to be exportable",
            });
        }
        if !result.minimum_power_margin_ratio.is_finite() {
            return Err(Propeller3dOptError::InvalidConfiguration {
                field: "pareto_candidates",
                requirement: "finite power margins",
            });
        }
    }

    Ok(candidates
        .iter()
        .enumerate()
        .filter_map(|(candidate_index, candidate)| {
            let dominated = candidates.iter().enumerate().any(|(other_index, other)| {
                other_index != candidate_index
                    && other.objective <= candidate.objective
                    && other.minimum_power_margin_ratio >= candidate.minimum_power_margin_ratio
                    && (other.objective < candidate.objective
                        || other.minimum_power_margin_ratio > candidate.minimum_power_margin_ratio)
            });
            (!dominated).then_some(candidate_index)
        })
        .collect())
}

/// Central finite-difference gradient of the bounded surrogate objective.
///
/// This is an auditable shape-search derivative, not an analytic adjoint of a
/// resolved flow solver. Probes that hit a variable bound are rejected so the
/// result cannot silently mix central and one-sided derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeShapeGradient {
    /// Components in the stable `BladeShapeVariables::as_array` ordering.
    pub components: [f64; BladeShapeVariables::COUNT],
    /// Requested central-difference step in variable units.
    pub step: f64,
}

impl BladeShapeGradient {
    /// Contract the gradient with a direction in optimizer-variable units.
    pub fn directional_derivative(self, direction: [f64; BladeShapeVariables::COUNT]) -> f64 {
        self.components
            .iter()
            .zip(direction)
            .map(|(gradient, component)| gradient * component)
            .sum()
    }
}

/// Result of a first-order Taylor remainder test for the surrogate objective.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeShapeTaylorReport {
    /// Positive step sizes, in the order supplied by the caller.
    pub step_sizes: Vec<f64>,
    /// `|f(x+h d) - f(x) - h grad(f)·d|` at each step.
    pub first_order_remainders: Vec<f64>,
    /// Observed log-log order between consecutive remainders.
    pub observed_orders: Vec<f64>,
    /// Whether the final remainder and observed order satisfy the contract.
    pub converged: bool,
}

/// Estimate the finite-difference objective gradient at an interior point.
pub fn estimate_shape_gradient(
    variables: BladeShapeVariables,
    config: &Propeller3dOptConfig,
    baseline: &OptimizedBlade,
    step: f64,
) -> Result<BladeShapeGradient, Propeller3dOptError> {
    config.validate()?;
    if !step.is_finite() || step <= 0.0 {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "shape_derivative_step",
            requirement: "finite and positive",
        });
    }

    let variables = variables.clamped();
    let mut components = [0.0; BladeShapeVariables::COUNT];
    for (index, component) in components.iter_mut().enumerate() {
        let plus = evaluate_blade_shape(variables.perturb(index, step), config, baseline);
        let minus = evaluate_blade_shape(variables.perturb(index, -step), config, baseline);
        let plus_value = plus.variables.as_array()[index];
        let minus_value = minus.variables.as_array()[index];
        let denominator = plus_value - minus_value;
        if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
            return Err(Propeller3dOptError::DerivativeEvaluation {
                variable: index,
                requirement: "strictly interior to its bounds for a central difference",
            });
        }
        if !plus.objective.is_finite() || !minus.objective.is_finite() {
            return Err(Propeller3dOptError::DerivativeEvaluation {
                variable: index,
                requirement: "finite objective at both probes",
            });
        }
        *component = (plus.objective - minus.objective) / denominator;
        if !component.is_finite() {
            return Err(Propeller3dOptError::DerivativeEvaluation {
                variable: index,
                requirement: "finite gradient component",
            });
        }
    }
    Ok(BladeShapeGradient { components, step })
}

/// Run a first-order Taylor test for a finite-difference shape derivative.
pub fn taylor_test_shape_derivative(
    variables: BladeShapeVariables,
    direction: [f64; BladeShapeVariables::COUNT],
    config: &Propeller3dOptConfig,
    baseline: &OptimizedBlade,
    derivative_step: f64,
    step_sizes: &[f64],
    tolerance: f64,
) -> Result<BladeShapeTaylorReport, Propeller3dOptError> {
    config.validate()?;
    if step_sizes.len() < 3
        || step_sizes
            .iter()
            .any(|step| !step.is_finite() || *step <= 0.0)
        || step_sizes.windows(2).any(|pair| pair[1] >= pair[0])
    {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "shape_taylor_step_sizes",
            requirement: "at least three finite positive strictly decreasing steps",
        });
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "shape_taylor_tolerance",
            requirement: "finite and positive",
        });
    }
    if direction.iter().any(|component| !component.is_finite())
        || direction
            .iter()
            .all(|component| component.abs() <= f64::EPSILON)
    {
        return Err(Propeller3dOptError::InvalidConfiguration {
            field: "shape_taylor_direction",
            requirement: "finite and non-zero",
        });
    }

    let variables = variables.clamped();
    let baseline_value = evaluate_blade_shape(variables, config, baseline).objective;
    if !baseline_value.is_finite() {
        return Err(Propeller3dOptError::DerivativeEvaluation {
            variable: 0,
            requirement: "finite base objective",
        });
    }
    let gradient = estimate_shape_gradient(variables, config, baseline, derivative_step)?;
    let directional_derivative = gradient.directional_derivative(direction);
    let mut first_order_remainders = Vec::with_capacity(step_sizes.len());
    for step in step_sizes {
        let requested = variables.shifted(direction, *step);
        let candidate = evaluate_blade_shape(requested, config, baseline);
        let requested_values = requested.as_array();
        let actual_values = candidate.variables.as_array();
        for index in 0..BladeShapeVariables::COUNT {
            if (requested_values[index] - actual_values[index]).abs()
                > 1.0e-12 * (1.0 + requested_values[index].abs())
            {
                return Err(Propeller3dOptError::DerivativeEvaluation {
                    variable: index,
                    requirement: "unclamped Taylor perturbations",
                });
            }
        }
        if !candidate.objective.is_finite() {
            return Err(Propeller3dOptError::DerivativeEvaluation {
                variable: 0,
                requirement: "finite objective at Taylor probe",
            });
        }
        first_order_remainders
            .push((candidate.objective - baseline_value - *step * directional_derivative).abs());
    }

    let observed_orders = first_order_remainders
        .windows(2)
        .zip(step_sizes.windows(2))
        .map(|(errors, steps)| {
            if errors[1] <= f64::MIN_POSITIVE {
                f64::INFINITY
            } else {
                (errors[0] / errors[1]).ln() / (steps[0] / steps[1]).ln()
            }
        })
        .collect::<Vec<_>>();
    let converged = first_order_remainders
        .last()
        .copied()
        .is_some_and(|remainder| remainder <= tolerance)
        && observed_orders
            .last()
            .copied()
            .is_some_and(|order| order.is_infinite() || order >= 1.25);

    Ok(BladeShapeTaylorReport {
        step_sizes: step_sizes.to_vec(),
        first_order_remainders,
        observed_orders,
        converged,
    })
}

pub fn optimize_blade_shape(config: &Propeller3dOptConfig) -> OptimizedBlade {
    let baseline = baseline_state(config);
    let initial = evaluate_blade_shape(config.initial_variables, config, &baseline);
    let mut current = if initial.feasible {
        initial.clone()
    } else if baseline.feasible {
        baseline.clone()
    } else {
        initial.clone()
    };
    let mut best = current.clone();
    let mut steps = [
        0.035, 1.6, 1.6, 0.08, 0.10, 0.035, 0.16, 0.0035, 0.010, 0.055,
    ];

    for _ in 0..config.max_iterations {
        let mut improved = false;
        for (i, step) in steps.iter().copied().enumerate() {
            let candidates = [
                current.variables.perturb(i, -step),
                current.variables.perturb(i, step),
            ];
            for candidate_vars in candidates {
                let candidate = evaluate_blade_shape(candidate_vars, config, &baseline);
                if candidate.feasible && (!best.feasible || candidate.objective < best.objective) {
                    current = candidate.clone();
                    best = candidate;
                    improved = true;
                }
            }
        }

        for step in &mut steps {
            *step *= if improved { 0.72 } else { 0.48 };
        }
    }

    best.iterations = config.max_iterations;
    best
}

/// Optimize after checked admission of user-provided settings.
pub fn try_optimize_blade_shape(
    config: &Propeller3dOptConfig,
) -> Result<OptimizedBlade, Propeller3dOptError> {
    config.validate()?;
    Ok(optimize_blade_shape(config))
}

pub fn evaluate_blade_shape(
    variables: BladeShapeVariables,
    config: &Propeller3dOptConfig,
    baseline: &OptimizedBlade,
) -> OptimizedBlade {
    let variables = variables.clamped();
    let mut source = config.source.clone();
    source.tip_radius *= variables.radius_scale;

    let mesh_config = Blade3dConfig {
        pitch_deg_root: config.mesh.pitch_deg_root + variables.pitch_root_delta_deg,
        pitch_deg_tip: config.mesh.pitch_deg_tip + variables.pitch_tip_delta_deg,
        chord_root: config.mesh.chord_root * variables.chord_root_scale,
        chord_tip: config.mesh.chord_tip * variables.chord_tip_scale,
        loop_closure_fraction: variables.loop_closure_fraction,
        loop_axial_lift: config.mesh.loop_axial_lift * variables.loop_axial_lift_scale,
        ..config.mesh
    };
    let baseline_thrust_proxy = thrust_proxy(&config.mesh, &config.source).max(f64::EPSILON);
    let candidate_thrust_proxy = thrust_proxy(&mesh_config, &source).max(0.0);
    let loading_scale = candidate_thrust_proxy / baseline_thrust_proxy;
    // The acoustic loading source must follow the candidate aerodynamic
    // loading. Keeping the baseline thrust/torque here made geometry changes
    // acoustically invisible except through tip Mach.
    source.thrust = config.source.thrust * loading_scale;
    source.torque = config.source.torque * loading_scale;
    let design = PropellerDesign {
        source,
        planform: PropellerPlanform::ToroidalLoop,
        edge_treatment: EdgeTreatment::LeadingAndTrailingSerrations,
        serrations: SerrationGeometry {
            amplitude_chord_fraction: variables.serration_amplitude,
            wavelength_chord_fraction: variables.serration_wavelength,
            span_fraction: variables.serration_span,
        },
    };
    let mesh = mesh_config.build_mesh(&design);
    let spectrum = design.spectrum(config.n_harmonics, EvaluationMode::ResearchControl);
    let loads = PropellerLoadPowerSummary::from_design_and_spectrum(&design, &spectrum);
    let operating_points = if config.operating_conditions.is_empty() {
        Vec::new()
    } else {
        PropellerOperatingEnvelope::try_from_source(design.source.clone())
            .and_then(|envelope| envelope.evaluate_conditions(&config.operating_conditions))
            .unwrap_or_default()
    };
    let minimum_power_margin_ratio = operating_points
        .iter()
        .map(|point| point.power_margin_w / point.shaft_power_w)
        .fold(1.0, f64::min);
    let panel_summary =
        compare_panel_acoustics(&baseline.mesh, &mesh, &baseline.spectrum, &spectrum);
    let thrust_loss = 1.0 - loading_scale;
    let thickness_scale_for_constant_weight =
        baseline.mesh.total_area() / mesh.total_area().max(f64::EPSILON);
    let peak_spread_delta_db = spectral_peakiness_db(&spectrum, 1_000.0, 5_000.0)
        - spectral_peakiness_db(&baseline.spectrum, 1_000.0, 5_000.0);

    let thrust_excess = (thrust_loss - config.max_thrust_loss).max(0.0);
    let thickness_excess = (thickness_scale_for_constant_weight
        - config.max_thickness_scale_for_constant_weight)
        .max(0.0)
        + (config.min_thickness_scale_for_constant_weight - thickness_scale_for_constant_weight)
            .max(0.0);
    let objective = -panel_summary.sensitive_band_reduction_db
        + config.peak_spread_weight * peak_spread_delta_db
        + config.thrust_penalty_weight * thrust_excess * thrust_excess
        + config.thickness_penalty_weight * thickness_excess * thickness_excess;
    let operating_point_margin_excess = if config.operating_conditions.is_empty() {
        0.0
    } else {
        (1.0 - minimum_power_margin_ratio).max(0.0)
    };
    let objective =
        objective + config.operating_point_margin_weight * operating_point_margin_excess;
    let provenance = production_provenance(ProductionEvaluation {
        config,
        variables,
        design: &design,
        mesh: &mesh,
        spectrum: &spectrum,
        loads,
        panel_summary,
        objective,
    });

    let mut candidate = OptimizedBlade {
        variables,
        design,
        mesh_config,
        mesh,
        spectrum,
        panel_summary,
        loads,
        provenance,
        operating_points,
        minimum_power_margin_ratio,
        objective,
        thrust_loss,
        thickness_scale_for_constant_weight,
        peak_spread_delta_db,
        feasible: false,
        iterations: 0,
    };
    candidate.feasible = config.candidate_is_feasible(&candidate);
    candidate
}

fn baseline_state(config: &Propeller3dOptConfig) -> OptimizedBlade {
    let design = PropellerDesign::conventional(config.source.clone());
    let mesh = config.mesh.build_mesh(&design);
    let spectrum = design.spectrum(config.n_harmonics, EvaluationMode::ResearchControl);
    let loads = PropellerLoadPowerSummary::from_design_and_spectrum(&design, &spectrum);
    let operating_points = if config.operating_conditions.is_empty() {
        Vec::new()
    } else {
        PropellerOperatingEnvelope::try_from_source(design.source.clone())
            .and_then(|envelope| envelope.evaluate_conditions(&config.operating_conditions))
            .unwrap_or_default()
    };
    let minimum_power_margin_ratio = operating_points
        .iter()
        .map(|point| point.power_margin_w / point.shaft_power_w)
        .fold(1.0, f64::min);
    let summary = compare_panel_acoustics(&mesh, &mesh, &spectrum, &spectrum);
    let variables = BladeShapeVariables {
        radius_scale: 1.0,
        pitch_root_delta_deg: 0.0,
        pitch_tip_delta_deg: 0.0,
        chord_root_scale: 1.0,
        chord_tip_scale: 1.0,
        loop_closure_fraction: config.mesh.loop_closure_fraction,
        loop_axial_lift_scale: 1.0,
        serration_amplitude: SerrationGeometry::default().amplitude_chord_fraction,
        serration_wavelength: SerrationGeometry::default().wavelength_chord_fraction,
        serration_span: SerrationGeometry::default().span_fraction,
    };
    let provenance = production_provenance(ProductionEvaluation {
        config,
        variables,
        design: &design,
        mesh: &mesh,
        spectrum: &spectrum,
        loads,
        panel_summary: summary,
        objective: 0.0,
    });

    let mut baseline = OptimizedBlade {
        variables,
        design,
        mesh_config: config.mesh,
        mesh,
        spectrum,
        panel_summary: summary,
        loads,
        provenance,
        operating_points,
        minimum_power_margin_ratio,
        objective: 0.0,
        thrust_loss: 0.0,
        thickness_scale_for_constant_weight: 1.0,
        peak_spread_delta_db: 0.0,
        feasible: false,
        iterations: 0,
    };
    baseline.feasible = config.candidate_is_feasible(&baseline);
    baseline
}

/// Construct the conventional reference blade used by the bounded optimizer.
/// Gallery and reporting clients use this exact state rather than recreating a
/// visually similar, but numerically disconnected, baseline.
pub fn baseline_blade_shape(config: &Propeller3dOptConfig) -> OptimizedBlade {
    baseline_state(config)
}

struct ProductionEvaluation<'a> {
    config: &'a Propeller3dOptConfig,
    variables: BladeShapeVariables,
    design: &'a PropellerDesign,
    mesh: &'a BladeMesh,
    spectrum: &'a PropellerSpectrum,
    loads: PropellerLoadPowerSummary,
    panel_summary: PanelAcousticSummary,
    objective: f64,
}

fn production_provenance(evaluation: ProductionEvaluation<'_>) -> PropellerResultProvenance {
    let input = format!(
        "source={:?}|mesh={:?}|variables={:?}|harmonics={}",
        evaluation.config.source,
        evaluation.config.mesh,
        evaluation.variables,
        evaluation.config.n_harmonics,
    );
    let output = output_descriptor(
        evaluation.design,
        evaluation.mesh,
        evaluation.spectrum,
        evaluation.loads,
        evaluation.panel_summary,
        evaluation.objective,
    );
    let seed = format!(
        "propeller.cpu.f64.panel-surrogate:mesh={}x{}+{}:harmonics={}",
        evaluation.config.mesh.radial_segments,
        evaluation.config.mesh.chord_segments,
        evaluation.config.mesh.loop_segments,
        evaluation.config.n_harmonics,
    );
    PropellerResultProvenance::panel_geometry_surrogate_for(
        seed,
        input.as_bytes(),
        output.as_bytes(),
    )
}

fn output_descriptor(
    design: &PropellerDesign,
    mesh: &BladeMesh,
    spectrum: &PropellerSpectrum,
    loads: PropellerLoadPowerSummary,
    panel_summary: PanelAcousticSummary,
    objective: f64,
) -> String {
    format!(
        "design={design:?}|mesh={mesh:?}|spectrum={spectrum:?}|loads={loads:?}|panel={panel_summary:?}|objective={objective:.17e}",
    )
}

fn thrust_proxy(mesh: &Blade3dConfig, source: &PropellerSource) -> f64 {
    let n = 512usize;
    let span = source.tip_radius - source.hub_radius;
    let mut thrust = 0.0;
    for i in 0..n {
        let eta = (i as f64 + 0.5) / n as f64;
        let radius = source.hub_radius + eta * span;
        let chord = mesh.chord_root + eta * (mesh.chord_tip - mesh.chord_root);
        let pitch =
            (mesh.pitch_deg_root + eta * (mesh.pitch_deg_tip - mesh.pitch_deg_root)).to_radians();
        let lift_angle = pitch.sin().max(0.04);
        thrust += chord * lift_angle * radius * radius * span / n as f64;
    }
    thrust * source.omega().powi(2) * source.density * source.n_blades as f64
}

fn spectral_peakiness_db(spectrum: &PropellerSpectrum, lo_hz: f64, hi_hz: f64) -> f64 {
    let powers: Vec<f64> = spectrum
        .harmonics
        .iter()
        .filter(|h| h.frequency_hz >= lo_hz && h.frequency_hz <= hi_hz)
        .map(|h| h.pressure_pa * h.pressure_pa)
        .collect();
    if powers.is_empty() {
        return 0.0;
    }
    let peak = powers.iter().copied().fold(0.0_f64, f64::max);
    let mean = powers.iter().sum::<f64>() / powers.len() as f64;
    10.0 * (peak.max(1e-30) / mean.max(1e-30)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_shape_respects_constraints() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 48,
                chord_segments: 12,
                loop_segments: 24,
                ..Blade3dConfig::default()
            },
            n_harmonics: 12,
            max_iterations: 3,
            ..Propeller3dOptConfig::default()
        };
        let result = optimize_blade_shape(&config);

        assert!(result.variables.radius_scale <= 1.0);
        assert!(result.is_feasible());
        assert!(result.thrust_loss <= config.max_thrust_loss + 1e-9);
        assert!(
            result.panel_summary.sensitive_band_reduction_db
                <= config.max_calibrated_reduction_db + 1e-9
        );
        assert!(result.thickness_scale_for_constant_weight.is_finite());
        assert!(result.panel_summary.sensitive_band_reduction_db > 3.0);
    }

    #[test]
    fn source_load_and_acoustic_power_closure_is_recorded_before_promotion() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 4,
                ..Blade3dConfig::default()
            },
            n_harmonics: 5,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);
        let candidate = evaluate_blade_shape(config.initial_variables, &config, &baseline);

        for result in [&baseline, &candidate] {
            let loads = result.loads;
            assert!(loads.is_physically_admissible(), "invalid loads: {loads:?}");
            assert!(
                (loads.shaft_power_w
                    - result.design.source.torque.abs() * result.design.source.omega().abs())
                .abs()
                    <= 1.0e-12 * loads.shaft_power_w
            );
            assert!(loads.acoustic_power_w <= loads.shaft_power_w);
            assert!((0.0..=1.0).contains(&loads.ideal_hover_efficiency));
            assert!(loads.acoustic_power_closure_error <= 1.0e-12);
        }
    }

    #[test]
    fn actuator_disk_reference_closes_hover_load_independently() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 4,
                ..Blade3dConfig::default()
            },
            n_harmonics: 5,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);
        let reference = PropellerActuatorDiskReference::try_from_source(&baseline.design.source)
            .expect("default source must admit the hover reference");

        reference
            .validate_load_summary(
                baseline.loads,
                PropellerActuatorDiskReference::PRODUCTION_TOLERANCE,
            )
            .expect("production load ledger must close against momentum theory");
        assert!(reference.induced_velocity_m_s > 0.0);
        assert!(reference.pressure_jump_pa > 0.0);
        assert!(reference.ideal_hover_power_w > 0.0);
        assert!(
            (reference.pressure_jump_pa * reference.disk_area_m2 * reference.induced_velocity_m_s
                - reference.ideal_hover_power_w)
                .abs()
                <= 1.0e-12 * reference.ideal_hover_power_w
        );
    }

    #[test]
    fn actuator_disk_reference_rejects_invalid_source_and_tampered_ledger() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 4,
                ..Blade3dConfig::default()
            },
            n_harmonics: 3,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);
        let mut invalid_source = baseline.design.source.clone();
        invalid_source.tip_radius = invalid_source.hub_radius;
        assert!(matches!(
            PropellerActuatorDiskReference::try_from_source(&invalid_source),
            Err(PropellerActuatorDiskReferenceError::InvalidInput {
                field: "source.hub_radius/source.tip_radius",
                ..
            })
        ));

        let reference = PropellerActuatorDiskReference::try_from_source(&baseline.design.source)
            .expect("default source must admit the hover reference");
        let mut tampered = baseline.loads;
        tampered.ideal_hover_power_w *= 1.001;
        assert!(matches!(
            reference.validate_load_summary(tampered, 1.0e-12),
            Err(PropellerActuatorDiskReferenceError::Mismatch {
                field: "ideal_hover_power_w",
                ..
            })
        ));
        assert!(matches!(
            reference.validate_load_summary(baseline.loads, f64::NAN),
            Err(PropellerActuatorDiskReferenceError::InvalidTolerance)
        ));
    }

    #[test]
    fn thickness_bounds_are_hard_promotion_constraints() {
        let mut config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 16,
                chord_segments: 5,
                loop_segments: 8,
                ..Blade3dConfig::default()
            },
            n_harmonics: 5,
            max_iterations: 2,
            initial_variables: BladeShapeVariables {
                chord_root_scale: 0.78,
                chord_tip_scale: 0.75,
                ..BladeShapeVariables::default()
            },
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_state(&config);
        let unconstrained_initial =
            evaluate_blade_shape(config.initial_variables, &config, &baseline);
        assert!(
            unconstrained_initial.thickness_scale_for_constant_weight > 1.0,
            "fixture must thin the planform before constant-weight rescaling"
        );
        config.max_thickness_scale_for_constant_weight =
            0.5 * (1.0 + unconstrained_initial.thickness_scale_for_constant_weight);
        let constrained_baseline = baseline_state(&config);
        let constrained_initial =
            evaluate_blade_shape(config.initial_variables, &config, &constrained_baseline);
        assert!(constrained_baseline.is_feasible());
        assert!(!constrained_initial.is_feasible());

        let result = optimize_blade_shape(&config);
        assert!(result.is_feasible());
        assert!(
            result.thickness_scale_for_constant_weight
                <= config.max_thickness_scale_for_constant_weight
        );
    }

    #[test]
    fn configured_thickness_bounds_also_drive_the_objective_penalty() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 16,
                chord_segments: 5,
                loop_segments: 8,
                ..Blade3dConfig::default()
            },
            n_harmonics: 5,
            initial_variables: BladeShapeVariables {
                chord_root_scale: 0.78,
                chord_tip_scale: 0.75,
                ..BladeShapeVariables::default()
            },
            thickness_penalty_weight: 50.0,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_state(&config);
        let unconstrained = evaluate_blade_shape(config.initial_variables, &config, &baseline);
        let excess = 0.1;
        assert!(
            unconstrained.thickness_scale_for_constant_weight > excess,
            "fixture must support a positive customized thickness bound"
        );

        let constrained = Propeller3dOptConfig {
            min_thickness_scale_for_constant_weight: 0.1,
            max_thickness_scale_for_constant_weight: unconstrained
                .thickness_scale_for_constant_weight
                - excess,
            ..config.clone()
        };
        let constrained_baseline = baseline_state(&constrained);
        let constrained_candidate = evaluate_blade_shape(
            constrained.initial_variables,
            &constrained,
            &constrained_baseline,
        );

        assert!(!constrained_candidate.is_feasible());
        assert!(
            (constrained_candidate.objective
                - unconstrained.objective
                - constrained.thickness_penalty_weight * excess * excess)
                .abs()
                < 1.0e-10,
            "custom thickness bounds must change the objective penalty"
        );
    }

    #[test]
    fn candidate_acoustic_loading_tracks_thrust_proxy() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 16,
                chord_segments: 6,
                loop_segments: 8,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 0,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_state(&config);
        let variables = BladeShapeVariables {
            radius_scale: 0.9,
            chord_root_scale: 0.9,
            chord_tip_scale: 0.9,
            ..BladeShapeVariables::default()
        };
        let candidate = evaluate_blade_shape(variables, &config, &baseline);
        let expected_scale = 1.0 - candidate.thrust_loss;
        assert!(
            (candidate.design.source.thrust / config.source.thrust - expected_scale).abs()
                < 1.0e-12
        );
        assert!(
            (candidate.design.source.torque / config.source.torque - expected_scale).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn checked_optimizer_admits_valid_configuration_and_rejects_invalid_inputs() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let result = try_optimize_blade_shape(&config).unwrap();
        assert!(result.objective.is_finite());

        let zero_harmonics = Propeller3dOptConfig {
            n_harmonics: 0,
            ..config.clone()
        };
        assert!(matches!(
            zero_harmonics.validate(),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "n_harmonics",
                ..
            })
        ));

        let nonfinite_variable = Propeller3dOptConfig {
            initial_variables: BladeShapeVariables {
                serration_span: f64::NAN,
                ..config.initial_variables
            },
            ..config.clone()
        };
        assert!(matches!(
            nonfinite_variable.validate(),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "serration_span",
                ..
            })
        ));

        let invalid_thickness_range = Propeller3dOptConfig {
            min_thickness_scale_for_constant_weight: 1.2,
            max_thickness_scale_for_constant_weight: 1.1,
            ..config.clone()
        };
        assert!(matches!(
            invalid_thickness_range.validate(),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "min_thickness_scale_for_constant_weight/max_thickness_scale_for_constant_weight",
                ..
            })
        ));

        let zero_torque = Propeller3dOptConfig {
            source: PropellerSource {
                torque: 0.0,
                ..config.source
            },
            ..config
        };
        assert!(matches!(
            zero_torque.validate(),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "source",
                ..
            })
        ));
    }

    #[test]
    fn multipoint_optimizer_populates_deterministic_baseline_and_candidate_points() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let source_temperature_k =
            source.speed_of_sound.powi(2) / (crate::AIR_GAMMA * crate::AIR_GAS_CONSTANT_J_KG_K);
        let config = Propeller3dOptConfig {
            source: source.clone(),
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            operating_conditions: vec![
                PropellerOperatingCondition {
                    freestream_m_s: 0.0,
                    rpm: source.rpm,
                    inflow_angle_rad: 0.0,
                    density_kg_m3: source.density,
                    temperature_k: source_temperature_k,
                },
                PropellerOperatingCondition {
                    freestream_m_s: 6.0,
                    rpm: source.rpm * 0.9,
                    inflow_angle_rad: 0.25,
                    density_kg_m3: source.density * 1.05,
                    temperature_k: source_temperature_k * 0.95,
                },
            ],
            ..Propeller3dOptConfig::default()
        };

        let baseline = baseline_blade_shape(&config);
        let result = try_optimize_blade_shape(&config).unwrap();

        assert_eq!(
            baseline.operating_points.len(),
            config.operating_conditions.len()
        );
        assert_eq!(
            result.operating_points.len(),
            config.operating_conditions.len()
        );
        assert!(baseline.is_feasible());
        assert!(result.is_feasible());
        assert!(result.is_exportable());
        assert!(result.minimum_power_margin_ratio.is_finite());
        assert!(result.minimum_power_margin_ratio >= 0.0);
        assert_eq!(
            result
                .operating_points
                .iter()
                .map(|point| point.freestream_m_s)
                .collect::<Vec<_>>(),
            config
                .operating_conditions
                .iter()
                .map(|condition| condition.freestream_m_s)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multipoint_optimizer_rejects_invalid_condition_at_checked_boundary() {
        let base = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let invalid = PropellerOperatingCondition {
            freestream_m_s: 0.0,
            rpm: 0.0,
            inflow_angle_rad: 0.0,
            density_kg_m3: base.source.density,
            temperature_k: 288.15,
        };
        let config = Propeller3dOptConfig {
            operating_conditions: vec![invalid],
            ..base
        };

        assert!(matches!(
            config.validate(),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "operating_conditions",
                ..
            })
        ));
        assert!(try_optimize_blade_shape(&config).is_err());
    }

    #[test]
    fn failed_candidate_off_design_admission_is_not_feasible_or_exportable() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let envelope = PropellerOperatingEnvelope::try_from_source(source.clone()).unwrap();
        let source_temperature_k =
            source.speed_of_sound.powi(2) / (crate::AIR_GAMMA * crate::AIR_GAS_CONSTANT_J_KG_K);
        let condition = PropellerOperatingCondition {
            freestream_m_s: envelope.max_freestream_m_s() * 0.95,
            rpm: source.rpm,
            inflow_angle_rad: 0.0,
            density_kg_m3: source.density,
            temperature_k: source_temperature_k,
        };
        let config = Propeller3dOptConfig {
            source: source.clone(),
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            operating_conditions: vec![condition],
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);
        assert!(config.validate().is_ok());
        assert!(baseline.is_feasible());
        assert_eq!(baseline.operating_points.len(), 1);

        let candidate = evaluate_blade_shape(
            BladeShapeVariables {
                radius_scale: 1.0,
                pitch_root_delta_deg: 5.0,
                pitch_tip_delta_deg: 7.0,
                chord_root_scale: 1.38,
                chord_tip_scale: 1.55,
                ..BladeShapeVariables::default()
            },
            &config,
            &baseline,
        );

        assert!(candidate.operating_points.is_empty());
        assert!(!candidate.is_feasible());
        assert!(!candidate.is_exportable());
    }

    #[test]
    fn multipoint_margin_weight_changes_objective_without_changing_admission() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let envelope = PropellerOperatingEnvelope::try_from_source(source.clone()).unwrap();
        let temperature_k =
            source.speed_of_sound.powi(2) / (crate::AIR_GAMMA * crate::AIR_GAS_CONSTANT_J_KG_K);
        let condition = PropellerOperatingCondition {
            freestream_m_s: envelope.max_freestream_m_s() * 0.9,
            rpm: source.rpm,
            inflow_angle_rad: 0.0,
            density_kg_m3: source.density,
            temperature_k,
        };
        let base = Propeller3dOptConfig {
            source,
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            operating_conditions: vec![condition],
            operating_point_margin_weight: 0.0,
            ..Propeller3dOptConfig::default()
        };
        let weighted = Propeller3dOptConfig {
            operating_point_margin_weight: 100.0,
            ..base.clone()
        };
        let variables = BladeShapeVariables {
            radius_scale: 1.0,
            pitch_root_delta_deg: 0.0,
            pitch_tip_delta_deg: 0.0,
            chord_root_scale: 1.0,
            chord_tip_scale: 1.0,
            loop_closure_fraction: base.mesh.loop_closure_fraction,
            loop_axial_lift_scale: 1.0,
            ..BladeShapeVariables::default()
        };
        let baseline = baseline_blade_shape(&base);
        let baseline_weighted = baseline_blade_shape(&weighted);
        let unweighted = evaluate_blade_shape(variables, &base, &baseline);
        let weighted_result = evaluate_blade_shape(variables, &weighted, &baseline_weighted);

        assert_eq!(unweighted.operating_points.len(), 1);
        assert_eq!(weighted_result.operating_points.len(), 1);
        assert!(unweighted.is_feasible());
        assert!(weighted_result.is_feasible());
        let expected_penalty = 100.0 * (1.0 - unweighted.minimum_power_margin_ratio).max(0.0);
        assert!(
            (weighted_result.objective - unweighted.objective - expected_penalty).abs() < 1.0e-10
        );
    }

    #[test]
    fn shape_gradient_is_finite_and_rejects_invalid_probe_steps() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);
        let gradient =
            estimate_shape_gradient(BladeShapeVariables::default(), &config, &baseline, 1.0e-5)
                .unwrap();

        assert_eq!(gradient.components.len(), BladeShapeVariables::COUNT);
        assert!(
            gradient
                .components
                .iter()
                .all(|component| component.is_finite())
        );
        assert!(
            gradient
                .directional_derivative([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .is_finite()
        );
        assert!(matches!(
            estimate_shape_gradient(BladeShapeVariables::default(), &config, &baseline, 0.0),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "shape_derivative_step",
                ..
            })
        ));
    }

    #[test]
    fn shape_derivative_taylor_remainder_has_second_order_decay() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);
        let report = taylor_test_shape_derivative(
            BladeShapeVariables::default(),
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &config,
            &baseline,
            1.0e-5,
            &[1.0e-2, 5.0e-3, 2.5e-3],
            1.0e-7,
        )
        .unwrap();

        assert_eq!(report.first_order_remainders.len(), 3);
        assert_eq!(report.observed_orders.len(), 2);
        assert!(
            report
                .first_order_remainders
                .iter()
                .all(|remainder| remainder.is_finite())
        );
        assert!(report.first_order_remainders[2] < report.first_order_remainders[0]);
        assert!(report.observed_orders[1] >= 1.25, "{report:?}");
        assert!(report.converged, "{report:?}");
    }

    #[test]
    fn shape_derivative_taylor_rejects_clamped_direction_and_bad_steps() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);
        let direction = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            taylor_test_shape_derivative(
                BladeShapeVariables::default(),
                direction,
                &config,
                &baseline,
                1.0e-5,
                &[0.5, 0.25, 0.125],
                1.0e-7,
            ),
            Err(Propeller3dOptError::DerivativeEvaluation { .. })
        ));
        assert!(matches!(
            taylor_test_shape_derivative(
                BladeShapeVariables::default(),
                direction,
                &config,
                &baseline,
                1.0e-5,
                &[1.0e-2, 1.0e-2, 5.0e-3],
                1.0e-7,
            ),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "shape_taylor_step_sizes",
                ..
            })
        ));
    }

    #[test]
    fn optimized_result_has_typed_bounded_surrogate_provenance() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let result = baseline_blade_shape(&config);
        assert_eq!(
            result.provenance.fidelity,
            crate::PropellerModelFidelity::PanelGeometrySurrogate
        );
        assert!(result.provenance.is_bounded_surrogate());
        assert!(result.is_exportable());
        assert_eq!(
            result.provenance.execution.requested_backend,
            crate::PropellerBackend::Cpu
        );
        assert_eq!(
            result.provenance.execution.actual_backend,
            Some(crate::PropellerBackend::Cpu)
        );
        assert_eq!(
            result.provenance.execution.precision,
            crate::PropellerPrecision::F64
        );
        assert_eq!(result.provenance.execution.input_checksum_sha256.len(), 64);
        assert_eq!(result.provenance.execution.output_checksum_sha256.len(), 64);

        let mut claimed_resolved = result;
        claimed_resolved.provenance.resolved_observer_directivity = true;
        assert!(!claimed_resolved.is_exportable());
    }

    #[test]
    fn trust_region_validation_is_unclamped_and_checked() {
        let from = BladeShapeVariables::default();
        let mut to = from;
        to.pitch_tip_delta_deg += 0.3;

        // Normalized by the 12-degree pitch bound width: 0.3 deg -> 0.025.
        let distance = validate_shape_trust_region(from, to, 0.3).unwrap();
        assert!((distance - 0.025).abs() < 1.0e-12);
        assert!(matches!(
            validate_shape_trust_region(from, to, 0.02),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "trust_region_step",
                ..
            })
        ));
        assert!(matches!(
            validate_shape_trust_region(from, to, f64::NAN),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "trust_region_radius",
                ..
            })
        ));
    }

    #[test]
    fn trust_region_metric_is_unitless_across_mixed_units() {
        // A 1-degree pitch step must not dominate a 0.01 serration step:
        // normalized, pitch is 1/12 while serration amplitude is 0.01/0.029.
        let from = BladeShapeVariables::default();
        let mut pitch_step = from;
        pitch_step.pitch_tip_delta_deg += 1.0;
        let mut serration_step = from;
        serration_step.serration_amplitude += 0.01;

        let pitch_distance = validate_shape_trust_region(from, pitch_step, 1.0).unwrap();
        let serration_distance = validate_shape_trust_region(from, serration_step, 1.0).unwrap();
        assert!((pitch_distance - 1.0 / 12.0).abs() < 1.0e-12);
        assert!((serration_distance - 0.01 / 0.029).abs() < 1.0e-12);
        assert!(
            serration_distance > pitch_distance,
            "serration {serration_distance} must outweigh pitch {pitch_distance}"
        );
    }

    #[test]
    fn pareto_frontier_is_deterministic_and_rejects_unexportable_results() {
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 3,
                loop_segments: 6,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 1,
            ..Propeller3dOptConfig::default()
        };
        let baseline = baseline_blade_shape(&config);

        let mut dominated = baseline.clone();
        dominated.objective = 1.0;
        dominated.minimum_power_margin_ratio = 0.5;
        dominated.provenance.execution.output_checksum_sha256 = sha256_hex(
            output_descriptor(
                &dominated.design,
                &dominated.mesh,
                &dominated.spectrum,
                dominated.loads,
                dominated.panel_summary,
                dominated.objective,
            )
            .as_bytes(),
        );

        let mut tradeoff = baseline.clone();
        tradeoff.objective = -1.0;
        tradeoff.minimum_power_margin_ratio = 0.25;
        tradeoff.provenance.execution.output_checksum_sha256 = sha256_hex(
            output_descriptor(
                &tradeoff.design,
                &tradeoff.mesh,
                &tradeoff.spectrum,
                tradeoff.loads,
                tradeoff.panel_summary,
                tradeoff.objective,
            )
            .as_bytes(),
        );

        assert_eq!(
            pareto_frontier(&[baseline.clone(), dominated, tradeoff.clone()]).unwrap(),
            vec![0, 2]
        );
        assert_eq!(
            pareto_frontier(&[tradeoff, baseline.clone()]).unwrap(),
            vec![0, 1]
        );

        let mut invalid = baseline;
        invalid.feasible = false;
        assert!(matches!(
            pareto_frontier(&[invalid]),
            Err(Propeller3dOptError::InvalidConfiguration {
                field: "pareto_candidates",
                ..
            })
        ));
    }
}
