//! Bounded ideal-actuator-disk operating-envelope reference.
//!
//! This module extends the static load ledger with a small, explicit
//! forward-flight reference.  It keeps thrust, torque, and geometry fixed at
//! the supplied source values and solves the annular actuator-disk momentum
//! equation for induced velocity.  It is useful for rejecting impossible
//! operating points and for checking power/efficiency bookkeeping; it is not a
//! blade-element, free-wake, CFD, or observer-directivity model.

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use std::fmt;

const POWER_ADMISSION_TOLERANCE: f64 = 1.0e-10;
/// Ratio of specific heats used by the ideal-gas air speed-of-sound reference.
pub const AIR_GAMMA: f64 = 1.4;
/// Specific gas constant for dry air [J/(kg K)].
pub const AIR_GAS_CONSTANT_J_KG_K: f64 = 287.05;

/// A fixed-load propeller operating envelope admitted by ideal actuator-disk
/// power and source shaft-power bounds.
#[derive(Debug, Clone)]
pub struct PropellerOperatingEnvelope {
    source: PropellerSource,
    disk_area_m2: f64,
    shaft_power_w: f64,
    ideal_hover_power_w: f64,
    max_freestream_m_s: f64,
}

/// An analytically evaluated operating point inside an admitted envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerOperatingPoint {
    /// Freestream velocity [m/s].
    pub freestream_m_s: f64,
    /// Advance ratio `J = V/(nD)`.
    pub advance_ratio: f64,
    /// Ideal induced velocity through the annular disk [m/s].
    pub induced_velocity_m_s: f64,
    /// Total disk-plane axial velocity `V + v_i` [m/s].
    pub disk_velocity_m_s: f64,
    /// Prescribed source thrust [N].
    pub thrust_n: f64,
    /// Prescribed source torque [N m].
    pub torque_nm: f64,
    /// Source mechanical shaft power `Q |omega|` [W].
    pub shaft_power_w: f64,
    /// Useful propulsive power `T V` [W].
    pub useful_power_w: f64,
    /// Ideal actuator-disk power `T (V + v_i)` [W].
    pub ideal_actuator_power_w: f64,
    /// Ideal actuator-disk propulsive efficiency `V/(V + v_i)`.
    pub ideal_actuator_efficiency: f64,
    /// Useful power divided by source shaft power.
    pub propulsive_efficiency: f64,
    /// Source shaft power minus ideal actuator-disk power [W].
    pub power_margin_w: f64,
}

/// One off-design operating condition for the fixed-geometry propeller
/// surrogate.
///
/// Thrust and torque are scaled from the admitted source using constant
/// nondimensional load coefficients: `rho * RPM²`. This is a transparent
/// engineering load model for multipoint screening, not a blade-element,
/// free-wake, or CFD solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerOperatingCondition {
    /// Freestream speed magnitude [m/s].
    pub freestream_m_s: f64,
    /// Positive rotational speed [RPM].
    pub rpm: f64,
    /// Inflow angle from the rotor axis [rad], limited to `[-pi/2, pi/2]`.
    pub inflow_angle_rad: f64,
    /// Operating density [kg/m³].
    pub density_kg_m3: f64,
    /// Operating static temperature [K].
    pub temperature_k: f64,
}

impl PropellerOperatingCondition {
    /// Ideal-gas speed of sound for this condition [m/s].
    pub fn speed_of_sound_m_s(self) -> f64 {
        (AIR_GAMMA * AIR_GAS_CONSTANT_J_KG_K * self.temperature_k).sqrt()
    }

    /// Axial component seen by the one-dimensional disk model [m/s].
    pub fn axial_freestream_m_s(self) -> f64 {
        self.freestream_m_s * self.inflow_angle_rad.cos()
    }
}

/// Multipoint operating result with explicit frame and thermodynamic inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerMultipointOperatingPoint {
    /// Original freestream speed magnitude [m/s].
    pub freestream_m_s: f64,
    /// Axial freestream component used by momentum theory [m/s].
    pub axial_freestream_m_s: f64,
    /// Inflow angle from the rotor axis [rad].
    pub inflow_angle_rad: f64,
    /// Operating rotational speed [RPM].
    pub rpm: f64,
    /// Operating density [kg/m³].
    pub density_kg_m3: f64,
    /// Operating temperature [K].
    pub temperature_k: f64,
    /// Operating speed of sound [m/s].
    pub speed_of_sound_m_s: f64,
    /// Advance ratio based on axial freestream and operating RPM.
    pub advance_ratio: f64,
    /// Ideal induced velocity [m/s].
    pub induced_velocity_m_s: f64,
    /// Total disk-plane axial velocity [m/s].
    pub disk_velocity_m_s: f64,
    /// Density/RPM-squared load scaling relative to the source.
    pub load_scale: f64,
    /// Scaled thrust [N].
    pub thrust_n: f64,
    /// Scaled torque [N m].
    pub torque_nm: f64,
    /// Scaled mechanical shaft power [W].
    pub shaft_power_w: f64,
    /// Useful propulsive power based on axial velocity [W].
    pub useful_power_w: f64,
    /// Ideal actuator-disk power [W].
    pub ideal_actuator_power_w: f64,
    /// Ideal actuator-disk efficiency.
    pub ideal_actuator_efficiency: f64,
    /// Useful power divided by shaft power.
    pub propulsive_efficiency: f64,
    /// Shaft power minus ideal actuator-disk power [W].
    pub power_margin_w: f64,
}

/// Failure-bounded operating-envelope construction/evaluation errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropellerOperatingEnvelopeError {
    /// A source field is outside the domain required by the reference.
    InvalidSource {
        /// Source field that failed admission.
        field: &'static str,
        /// Required physical domain.
        requirement: &'static str,
    },
    /// The source shaft power is below ideal hover power.
    InsufficientHoverPower {
        /// Source shaft power [W].
        shaft_power_w: f64,
        /// Ideal hover power [W].
        ideal_hover_power_w: f64,
    },
    /// A requested freestream velocity is invalid.
    InvalidFreestream,
    /// A requested point needs more ideal power than the source supplies.
    ExceedsShaftPower {
        /// Requested freestream velocity [m/s].
        freestream_m_s: f64,
        /// Required ideal actuator-disk power [W].
        ideal_actuator_power_w: f64,
        /// Available source shaft power [W].
        shaft_power_w: f64,
    },
    /// An off-design condition is outside the multipoint reference domain.
    InvalidCondition {
        /// Condition field that failed validation.
        field: &'static str,
        /// Required physical domain.
        requirement: &'static str,
    },
}

impl fmt::Display for PropellerOperatingEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSource { field, requirement } => {
                write!(
                    f,
                    "propeller operating-envelope {field} must be {requirement}"
                )
            }
            Self::InsufficientHoverPower {
                shaft_power_w,
                ideal_hover_power_w,
            } => write!(
                f,
                "propeller shaft power {shaft_power_w:e} W is below ideal hover power {ideal_hover_power_w:e} W"
            ),
            Self::InvalidFreestream => {
                write!(
                    f,
                    "propeller freestream velocity must be finite and non-negative"
                )
            }
            Self::ExceedsShaftPower {
                freestream_m_s,
                ideal_actuator_power_w,
                shaft_power_w,
            } => write!(
                f,
                "propeller operating point {freestream_m_s:e} m/s needs {ideal_actuator_power_w:e} W, exceeding {shaft_power_w:e} W shaft power"
            ),
            Self::InvalidCondition { field, requirement } => {
                write!(
                    f,
                    "propeller operating condition {field} must be {requirement}"
                )
            }
        }
    }
}

impl std::error::Error for PropellerOperatingEnvelopeError {}

impl PropellerOperatingEnvelope {
    /// Construct a fixed-load ideal-actuator-disk operating envelope.
    pub fn try_from_source(
        source: PropellerSource,
    ) -> Result<Self, PropellerOperatingEnvelopeError> {
        if source.n_blades == 0 {
            return Err(Self::invalid_source("source.n_blades", "non-zero"));
        }
        for (field, value) in [("source.rpm", source.rpm), ("source.thrust", source.thrust)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Self::invalid_source(field, "finite and positive"));
            }
        }
        for (field, value) in [
            ("source.torque", source.torque),
            ("source.speed_of_sound", source.speed_of_sound),
            ("source.density", source.density),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Self::invalid_source(field, "finite and positive"));
            }
        }
        for (field, value) in [
            ("source.tip_radius", source.tip_radius),
            ("source.hub_radius", source.hub_radius),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Self::invalid_source(field, "finite and positive"));
            }
        }
        if source.tip_radius <= source.hub_radius {
            return Err(Self::invalid_source(
                "source.hub_radius/source.tip_radius",
                "0 < hub_radius < tip_radius",
            ));
        }

        let disk_area_m2 =
            std::f64::consts::PI * (source.tip_radius.powi(2) - source.hub_radius.powi(2));
        let shaft_power_w = source.torque * source.omega().abs();
        let ideal_hover_power_w =
            source.thrust * (source.thrust / (2.0 * source.density * disk_area_m2)).sqrt();
        if !disk_area_m2.is_finite()
            || disk_area_m2 <= 0.0
            || !shaft_power_w.is_finite()
            || shaft_power_w <= 0.0
            || !ideal_hover_power_w.is_finite()
            || ideal_hover_power_w <= 0.0
        {
            return Err(Self::invalid_source(
                "derived disk area/shaft power/hover power",
                "finite and positive",
            ));
        }
        if shaft_power_w + POWER_ADMISSION_TOLERANCE < ideal_hover_power_w {
            return Err(Self::insufficient_hover_power(
                shaft_power_w,
                ideal_hover_power_w,
            ));
        }

        // At the shaft-power boundary P/T = V + v_i and momentum theory gives
        // v_i = T²/(2 rho A P).  This is the largest admitted forward speed.
        let power_per_thrust = shaft_power_w / source.thrust;
        let max_freestream_m_s = power_per_thrust
            - source.thrust.powi(2) / (2.0 * source.density * disk_area_m2 * shaft_power_w);
        if !max_freestream_m_s.is_finite() || max_freestream_m_s < 0.0 {
            return Err(Self::invalid_source(
                "derived maximum freestream",
                "finite and non-negative",
            ));
        }

        Ok(Self {
            source,
            disk_area_m2,
            shaft_power_w,
            ideal_hover_power_w,
            max_freestream_m_s,
        })
    }

    /// Source used to construct this envelope.
    pub fn source(&self) -> &PropellerSource {
        &self.source
    }

    /// Annular actuator-disk area [m²].
    pub fn disk_area_m2(&self) -> f64 {
        self.disk_area_m2
    }

    /// Source shaft power [W].
    pub fn shaft_power_w(&self) -> f64 {
        self.shaft_power_w
    }

    /// Ideal hover power [W].
    pub fn ideal_hover_power_w(&self) -> f64 {
        self.ideal_hover_power_w
    }

    /// Largest forward speed admitted by ideal actuator-disk power [m/s].
    pub fn max_freestream_m_s(&self) -> f64 {
        self.max_freestream_m_s
    }

    /// Evaluate one admitted forward-flight operating point.
    pub fn evaluate(
        &self,
        freestream_m_s: f64,
    ) -> Result<PropellerOperatingPoint, PropellerOperatingEnvelopeError> {
        if !freestream_m_s.is_finite() || freestream_m_s < 0.0 {
            return Err(PropellerOperatingEnvelopeError::InvalidFreestream);
        }

        let thrust = self.source.thrust;
        let density = self.source.density;
        let discriminant = freestream_m_s.powi(2) + 2.0 * thrust / (density * self.disk_area_m2);
        let induced_velocity_m_s =
            thrust / (density * self.disk_area_m2 * (discriminant.sqrt() + freestream_m_s));
        let disk_velocity_m_s = freestream_m_s + induced_velocity_m_s;
        let ideal_actuator_power_w = thrust * disk_velocity_m_s;
        let power_margin_w = self.shaft_power_w - ideal_actuator_power_w;
        if power_margin_w < -POWER_ADMISSION_TOLERANCE {
            return Err(PropellerOperatingEnvelopeError::ExceedsShaftPower {
                freestream_m_s,
                ideal_actuator_power_w,
                shaft_power_w: self.shaft_power_w,
            });
        }

        let revolutions_per_second = self.source.rpm / 60.0;
        let diameter_m = 2.0 * self.source.tip_radius;
        let advance_ratio = freestream_m_s / (revolutions_per_second * diameter_m);
        let useful_power_w = thrust * freestream_m_s;
        let ideal_actuator_efficiency = useful_power_w / ideal_actuator_power_w;
        let propulsive_efficiency = useful_power_w / self.shaft_power_w;

        Ok(PropellerOperatingPoint {
            freestream_m_s,
            advance_ratio,
            induced_velocity_m_s,
            disk_velocity_m_s,
            thrust_n: thrust,
            torque_nm: self.source.torque,
            shaft_power_w: self.shaft_power_w,
            useful_power_w,
            ideal_actuator_power_w,
            ideal_actuator_efficiency,
            propulsive_efficiency,
            power_margin_w: power_margin_w.max(0.0),
        })
    }

    /// Evaluate a deterministic list of forward speeds.
    pub fn evaluate_many(
        &self,
        freestream_m_s: &[f64],
    ) -> Result<Vec<PropellerOperatingPoint>, PropellerOperatingEnvelopeError> {
        freestream_m_s
            .iter()
            .copied()
            .map(|speed| self.evaluate(speed))
            .collect()
    }

    /// Evaluate one off-design condition using the constant-load-coefficient
    /// multipoint surrogate.
    pub fn evaluate_condition(
        &self,
        condition: PropellerOperatingCondition,
    ) -> Result<PropellerMultipointOperatingPoint, PropellerOperatingEnvelopeError> {
        self.validate_condition(condition)?;
        let rpm_ratio = condition.rpm / self.source.rpm;
        let load_scale = (condition.density_kg_m3 / self.source.density) * rpm_ratio.powi(2);
        let thrust_n = self.source.thrust * load_scale;
        let torque_nm = self.source.torque * load_scale;
        let shaft_power_w = torque_nm * (2.0 * std::f64::consts::PI * condition.rpm / 60.0);
        let axial_freestream_m_s = condition.axial_freestream_m_s();
        let point = self.evaluate_scaled_load(
            axial_freestream_m_s,
            thrust_n,
            torque_nm,
            shaft_power_w,
            condition.density_kg_m3,
            condition.rpm,
        )?;

        Ok(PropellerMultipointOperatingPoint {
            freestream_m_s: condition.freestream_m_s,
            axial_freestream_m_s,
            inflow_angle_rad: condition.inflow_angle_rad,
            rpm: condition.rpm,
            density_kg_m3: condition.density_kg_m3,
            temperature_k: condition.temperature_k,
            speed_of_sound_m_s: condition.speed_of_sound_m_s(),
            advance_ratio: point.advance_ratio,
            induced_velocity_m_s: point.induced_velocity_m_s,
            disk_velocity_m_s: point.disk_velocity_m_s,
            load_scale,
            thrust_n,
            torque_nm,
            shaft_power_w,
            useful_power_w: point.useful_power_w,
            ideal_actuator_power_w: point.ideal_actuator_power_w,
            ideal_actuator_efficiency: point.ideal_actuator_efficiency,
            propulsive_efficiency: point.propulsive_efficiency,
            power_margin_w: point.power_margin_w,
        })
    }

    /// Evaluate a deterministic multipoint operating matrix in input order.
    pub fn evaluate_conditions(
        &self,
        conditions: &[PropellerOperatingCondition],
    ) -> Result<Vec<PropellerMultipointOperatingPoint>, PropellerOperatingEnvelopeError> {
        conditions
            .iter()
            .copied()
            .map(|condition| self.evaluate_condition(condition))
            .collect()
    }

    fn validate_condition(
        &self,
        condition: PropellerOperatingCondition,
    ) -> Result<(), PropellerOperatingEnvelopeError> {
        for (field, value) in [
            ("freestream_m_s", condition.freestream_m_s),
            ("rpm", condition.rpm),
            ("inflow_angle_rad", condition.inflow_angle_rad),
            ("density_kg_m3", condition.density_kg_m3),
            ("temperature_k", condition.temperature_k),
        ] {
            if !value.is_finite() {
                return Err(Self::invalid_condition(field, "finite"));
            }
        }
        if condition.freestream_m_s < 0.0 {
            return Err(Self::invalid_condition("freestream_m_s", "non-negative"));
        }
        if condition.rpm <= 0.0 {
            return Err(Self::invalid_condition("rpm", "positive"));
        }
        if condition.density_kg_m3 <= 0.0 {
            return Err(Self::invalid_condition("density_kg_m3", "positive"));
        }
        if condition.temperature_k <= 0.0 {
            return Err(Self::invalid_condition("temperature_k", "positive"));
        }
        if condition.inflow_angle_rad.abs() > std::f64::consts::FRAC_PI_2 {
            return Err(Self::invalid_condition(
                "inflow_angle_rad",
                "within [-pi/2, pi/2]",
            ));
        }
        let speed_of_sound_m_s = condition.speed_of_sound_m_s();
        if !speed_of_sound_m_s.is_finite() || speed_of_sound_m_s <= 0.0 {
            return Err(Self::invalid_condition(
                "derived speed_of_sound_m_s",
                "finite and positive",
            ));
        }
        Ok(())
    }

    fn evaluate_scaled_load(
        &self,
        freestream_m_s: f64,
        thrust: f64,
        torque: f64,
        shaft_power_w: f64,
        density: f64,
        rpm: f64,
    ) -> Result<PropellerOperatingPoint, PropellerOperatingEnvelopeError> {
        let discriminant = freestream_m_s.powi(2) + 2.0 * thrust / (density * self.disk_area_m2);
        let induced_velocity_m_s =
            thrust / (density * self.disk_area_m2 * (discriminant.sqrt() + freestream_m_s));
        let disk_velocity_m_s = freestream_m_s + induced_velocity_m_s;
        let ideal_actuator_power_w = thrust * disk_velocity_m_s;
        let power_margin_w = shaft_power_w - ideal_actuator_power_w;
        if power_margin_w < -POWER_ADMISSION_TOLERANCE {
            return Err(PropellerOperatingEnvelopeError::ExceedsShaftPower {
                freestream_m_s,
                ideal_actuator_power_w,
                shaft_power_w,
            });
        }

        let revolutions_per_second = rpm / 60.0;
        let diameter_m = 2.0 * self.source.tip_radius;
        let advance_ratio = freestream_m_s / (revolutions_per_second * diameter_m);
        let useful_power_w = thrust * freestream_m_s;
        let ideal_actuator_efficiency = useful_power_w / ideal_actuator_power_w;
        let propulsive_efficiency = useful_power_w / shaft_power_w;

        Ok(PropellerOperatingPoint {
            freestream_m_s,
            advance_ratio,
            induced_velocity_m_s,
            disk_velocity_m_s,
            thrust_n: thrust,
            torque_nm: torque,
            shaft_power_w,
            useful_power_w,
            ideal_actuator_power_w,
            ideal_actuator_efficiency,
            propulsive_efficiency,
            power_margin_w: power_margin_w.max(0.0),
        })
    }

    fn invalid_condition(
        field: &'static str,
        requirement: &'static str,
    ) -> PropellerOperatingEnvelopeError {
        PropellerOperatingEnvelopeError::InvalidCondition { field, requirement }
    }

    fn invalid_source(
        field: &'static str,
        requirement: &'static str,
    ) -> PropellerOperatingEnvelopeError {
        PropellerOperatingEnvelopeError::InvalidSource { field, requirement }
    }

    fn insufficient_hover_power(
        shaft_power_w: f64,
        ideal_hover_power_w: f64,
    ) -> PropellerOperatingEnvelopeError {
        PropellerOperatingEnvelopeError::InsufficientHoverPower {
            shaft_power_w,
            ideal_hover_power_w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_source_matches_hover_momentum_reference() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let envelope = PropellerOperatingEnvelope::try_from_source(source.clone()).unwrap();
        let hover = envelope.evaluate(0.0).unwrap();
        let expected_induced =
            (source.thrust / (2.0 * source.density * envelope.disk_area_m2())).sqrt();
        let expected_power = source.thrust * expected_induced;

        assert!((hover.induced_velocity_m_s - expected_induced).abs() < 1.0e-12);
        assert!((hover.ideal_actuator_power_w - expected_power).abs() < 1.0e-12);
        assert_eq!(hover.useful_power_w, 0.0);
        assert_eq!(hover.advance_ratio, 0.0);
        assert_eq!(hover.ideal_actuator_efficiency, 0.0);
    }

    #[test]
    fn admitted_envelope_is_monotone_and_bounded() {
        let envelope =
            PropellerOperatingEnvelope::try_from_source(PropellerSource::drone_5inch(18_000.0))
                .unwrap();
        let speeds = [
            0.0,
            envelope.max_freestream_m_s() * 0.25,
            envelope.max_freestream_m_s() * 0.75,
            envelope.max_freestream_m_s(),
        ];
        let points = envelope.evaluate_many(&speeds).unwrap();

        for pair in points.windows(2) {
            assert!(pair[1].useful_power_w >= pair[0].useful_power_w);
            assert!(pair[1].ideal_actuator_power_w >= pair[0].ideal_actuator_power_w);
            assert!(pair[1].induced_velocity_m_s <= pair[0].induced_velocity_m_s);
        }
        for point in points {
            assert!((0.0..=1.0).contains(&point.ideal_actuator_efficiency));
            assert!((0.0..=1.0).contains(&point.propulsive_efficiency));
            assert!(point.power_margin_w.is_finite());
        }
    }

    #[test]
    fn invalid_and_overpowered_points_are_rejected() {
        let envelope =
            PropellerOperatingEnvelope::try_from_source(PropellerSource::drone_5inch(18_000.0))
                .unwrap();
        assert_eq!(
            envelope.evaluate(-1.0),
            Err(PropellerOperatingEnvelopeError::InvalidFreestream)
        );
        assert_eq!(
            envelope.evaluate(f64::NAN),
            Err(PropellerOperatingEnvelopeError::InvalidFreestream)
        );
        assert!(matches!(
            envelope.evaluate(envelope.max_freestream_m_s() * 1.01),
            Err(PropellerOperatingEnvelopeError::ExceedsShaftPower { .. })
        ));

        let mut weak_source = PropellerSource::drone_5inch(18_000.0);
        weak_source.torque = 0.001;
        assert!(matches!(
            PropellerOperatingEnvelope::try_from_source(weak_source),
            Err(PropellerOperatingEnvelopeError::InsufficientHoverPower { .. })
        ));
    }

    #[test]
    fn multipoint_scaling_matches_density_and_rpm_squared_reference() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let envelope = PropellerOperatingEnvelope::try_from_source(source.clone()).unwrap();
        let temperature_k = source.speed_of_sound.powi(2) / (AIR_GAMMA * AIR_GAS_CONSTANT_J_KG_K);
        let baseline = envelope
            .evaluate_condition(PropellerOperatingCondition {
                freestream_m_s: 0.0,
                rpm: source.rpm,
                inflow_angle_rad: 0.0,
                density_kg_m3: source.density,
                temperature_k,
            })
            .unwrap();
        assert!((baseline.load_scale - 1.0).abs() < 1.0e-12);
        assert!((baseline.thrust_n - source.thrust).abs() < 1.0e-12);
        assert!((baseline.torque_nm - source.torque).abs() < 1.0e-12);

        let changed = envelope
            .evaluate_condition(PropellerOperatingCondition {
                freestream_m_s: 0.0,
                rpm: source.rpm * 0.5,
                inflow_angle_rad: 0.0,
                density_kg_m3: source.density * 2.0,
                temperature_k: temperature_k * 1.1,
            })
            .unwrap();
        assert!((changed.load_scale - 0.5).abs() < 1.0e-12);
        assert!((changed.thrust_n / baseline.thrust_n - 0.5).abs() < 1.0e-12);
        assert!((changed.torque_nm / baseline.torque_nm - 0.5).abs() < 1.0e-12);
        assert!((changed.shaft_power_w / baseline.shaft_power_w - 0.25).abs() < 1.0e-12);
        assert!(
            (changed.induced_velocity_m_s / baseline.induced_velocity_m_s - 0.5).abs() < 1.0e-12
        );
        assert!(
            (changed.speed_of_sound_m_s / source.speed_of_sound - 1.1_f64.sqrt()).abs() < 1.0e-12
        );
    }

    #[test]
    fn multipoint_frame_transform_preserves_axial_solution_for_symmetric_inflow() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let envelope = PropellerOperatingEnvelope::try_from_source(source.clone()).unwrap();
        let temperature_k = source.speed_of_sound.powi(2) / (AIR_GAMMA * AIR_GAS_CONSTANT_J_KG_K);
        let positive = envelope
            .evaluate_condition(PropellerOperatingCondition {
                freestream_m_s: 8.0,
                rpm: source.rpm,
                inflow_angle_rad: 0.35,
                density_kg_m3: source.density,
                temperature_k,
            })
            .unwrap();
        let negative = envelope
            .evaluate_condition(PropellerOperatingCondition {
                inflow_angle_rad: -0.35,
                ..PropellerOperatingCondition {
                    freestream_m_s: 8.0,
                    rpm: source.rpm,
                    inflow_angle_rad: 0.35,
                    density_kg_m3: source.density,
                    temperature_k,
                }
            })
            .unwrap();
        assert!((positive.axial_freestream_m_s - 8.0 * 0.35_f64.cos()).abs() < 1.0e-12);
        assert!((positive.axial_freestream_m_s - negative.axial_freestream_m_s).abs() < 1.0e-12);
        assert!((positive.induced_velocity_m_s - negative.induced_velocity_m_s).abs() < 1.0e-12);
        assert!(
            (positive.ideal_actuator_power_w - negative.ideal_actuator_power_w).abs() < 1.0e-12
        );
        assert!((positive.speed_of_sound_m_s - negative.speed_of_sound_m_s).abs() < 1.0e-12);
    }

    #[test]
    fn multipoint_batch_is_deterministic_and_rejects_invalid_conditions() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let envelope = PropellerOperatingEnvelope::try_from_source(source.clone()).unwrap();
        let temperature_k = source.speed_of_sound.powi(2) / (AIR_GAMMA * AIR_GAS_CONSTANT_J_KG_K);
        let conditions = [
            PropellerOperatingCondition {
                freestream_m_s: 0.0,
                rpm: source.rpm,
                inflow_angle_rad: 0.0,
                density_kg_m3: source.density,
                temperature_k,
            },
            PropellerOperatingCondition {
                freestream_m_s: 6.0,
                rpm: source.rpm * 0.9,
                inflow_angle_rad: 0.2,
                density_kg_m3: source.density * 1.05,
                temperature_k: temperature_k * 0.95,
            },
        ];
        let first = envelope.evaluate_conditions(&conditions).unwrap();
        let second = envelope.evaluate_conditions(&conditions).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), conditions.len());
        assert!(matches!(
            envelope.evaluate_condition(PropellerOperatingCondition {
                rpm: 0.0,
                ..conditions[0]
            }),
            Err(PropellerOperatingEnvelopeError::InvalidCondition { field: "rpm", .. })
        ));
        assert!(matches!(
            envelope.evaluate_condition(PropellerOperatingCondition {
                inflow_angle_rad: std::f64::consts::PI,
                ..conditions[0]
            }),
            Err(PropellerOperatingEnvelopeError::InvalidCondition {
                field: "inflow_angle_rad",
                ..
            })
        ));
        assert!(matches!(
            envelope.evaluate_condition(PropellerOperatingCondition {
                temperature_k: f64::NAN,
                ..conditions[0]
            }),
            Err(PropellerOperatingEnvelopeError::InvalidCondition {
                field: "temperature_k",
                ..
            })
        ));
    }
}
