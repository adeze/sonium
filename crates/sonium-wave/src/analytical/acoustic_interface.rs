//! Normal-incidence pressure reflection and transmission at a fluid interface.
//!
//! This lossless one-dimensional reference uses the crate's `exp(+iωt)`
//! convention, so a wave travelling toward increasing `x` is `exp(-ikx)`.
//! It uses specific impedance `Z=ρc`; pressure and positive-`x` particle
//! velocity are continuous at the interface `x=0`.

use num_complex::Complex64;
use std::fmt;

/// Invalid input or unrepresentable output for [`AcousticInterface1d`].
#[derive(Debug, Clone, PartialEq)]
pub enum AcousticInterfaceError {
    /// A named input is non-finite or outside its documented domain.
    InvalidParameter {
        field: &'static str,
        requirement: &'static str,
    },
    /// A derived field or coefficient cannot be represented in binary64.
    EvaluationOverflow,
}

impl fmt::Display for AcousticInterfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => write!(
                f,
                "acoustic interface parameter {field} must be {requirement}"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "acoustic interface result cannot be represented in binary64"
            ),
        }
    }
}
impl std::error::Error for AcousticInterfaceError {}

/// Complex total pressure and particle velocity at one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticInterfacePoint {
    /// Coordinate in metres.
    pub x_m: f64,
    /// Total pressure in Pa.
    pub pressure_pa: Complex64,
    /// Positive-`x` particle velocity in m/s.
    pub particle_velocity_m_per_s: Complex64,
}

/// Exact lossless, normal-incidence interface between two homogeneous fluids.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticInterface1d {
    density_left_kg_per_m3: f64,
    sound_speed_left_m_per_s: f64,
    density_right_kg_per_m3: f64,
    sound_speed_right_m_per_s: f64,
    frequency_hz: f64,
    incident_pressure_pa: Complex64,
}

impl AcousticInterface1d {
    /// Construct a checked interface reference from material properties and incident pressure.
    pub fn try_new(
        density_left_kg_per_m3: f64,
        sound_speed_left_m_per_s: f64,
        density_right_kg_per_m3: f64,
        sound_speed_right_m_per_s: f64,
        frequency_hz: f64,
        incident_pressure_pa: Complex64,
    ) -> Result<Self, AcousticInterfaceError> {
        for (field, value) in [
            ("density_left_kg_per_m3", density_left_kg_per_m3),
            ("sound_speed_left_m_per_s", sound_speed_left_m_per_s),
            ("density_right_kg_per_m3", density_right_kg_per_m3),
            ("sound_speed_right_m_per_s", sound_speed_right_m_per_s),
            ("frequency_hz", frequency_hz),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(AcousticInterfaceError::InvalidParameter {
                    field,
                    requirement: "finite and positive",
                });
            }
        }
        if !incident_pressure_pa.re.is_finite() || !incident_pressure_pa.im.is_finite() {
            return Err(AcousticInterfaceError::InvalidParameter {
                field: "incident_pressure_pa",
                requirement: "finite real and imaginary components",
            });
        }
        if ![
            density_left_kg_per_m3 * sound_speed_left_m_per_s,
            density_right_kg_per_m3 * sound_speed_right_m_per_s,
            std::f64::consts::TAU * frequency_hz / sound_speed_left_m_per_s,
            std::f64::consts::TAU * frequency_hz / sound_speed_right_m_per_s,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return Err(AcousticInterfaceError::EvaluationOverflow);
        }
        Ok(Self {
            density_left_kg_per_m3,
            sound_speed_left_m_per_s,
            density_right_kg_per_m3,
            sound_speed_right_m_per_s,
            frequency_hz,
            incident_pressure_pa,
        })
    }

    /// Left specific impedance `ρ₁c₁` in Pa·s/m.
    pub fn left_specific_impedance_pa_s_per_m(self) -> f64 {
        self.density_left_kg_per_m3 * self.sound_speed_left_m_per_s
    }
    /// Right specific impedance `ρ₂c₂` in Pa·s/m.
    pub fn right_specific_impedance_pa_s_per_m(self) -> f64 {
        self.density_right_kg_per_m3 * self.sound_speed_right_m_per_s
    }
    /// Left wavenumber in rad/m.
    pub fn left_wavenumber_rad_per_m(self) -> f64 {
        std::f64::consts::TAU * self.frequency_hz / self.sound_speed_left_m_per_s
    }
    /// Right wavenumber in rad/m.
    pub fn right_wavenumber_rad_per_m(self) -> f64 {
        std::f64::consts::TAU * self.frequency_hz / self.sound_speed_right_m_per_s
    }
    /// Pressure reflection coefficient `(Z₂-Z₁)/(Z₂+Z₁)`.
    pub fn pressure_reflection_coefficient(self) -> f64 {
        let (a, b) = (
            self.left_specific_impedance_pa_s_per_m(),
            self.right_specific_impedance_pa_s_per_m(),
        );
        (b - a) / (b + a)
    }
    /// Pressure transmission coefficient `2Z₂/(Z₂+Z₁)`.
    pub fn pressure_transmission_coefficient(self) -> f64 {
        let (a, b) = (
            self.left_specific_impedance_pa_s_per_m(),
            self.right_specific_impedance_pa_s_per_m(),
        );
        2.0 * b / (b + a)
    }
    /// Incident, reflected, and transmitted time-averaged intensities in W/m².
    pub fn intensities_w_per_m2(self) -> [f64; 3] {
        let zi = self.left_specific_impedance_pa_s_per_m();
        let zt = self.right_specific_impedance_pa_s_per_m();
        let incident = self.incident_pressure_pa.norm_sqr() / (2.0 * zi);
        [
            incident,
            incident * self.pressure_reflection_coefficient().powi(2),
            self.incident_pressure_pa.norm_sqr() * self.pressure_transmission_coefficient().powi(2)
                / (2.0 * zt),
        ]
    }
    /// Evaluate the incident-plus-reflected field for `x <= 0`.
    pub fn try_evaluate_left(
        self,
        x_m: f64,
    ) -> Result<AcousticInterfacePoint, AcousticInterfaceError> {
        if !x_m.is_finite() || x_m > 0.0 {
            return Err(AcousticInterfaceError::InvalidParameter {
                field: "x_m (left medium)",
                requirement: "finite and <= 0",
            });
        }
        let incident = self.incident_pressure_pa
            * Complex64::from_polar(1.0, -self.left_wavenumber_rad_per_m() * x_m);
        let reflected = self.incident_pressure_pa
            * self.pressure_reflection_coefficient()
            * Complex64::from_polar(1.0, self.left_wavenumber_rad_per_m() * x_m);
        self.checked_point(
            x_m,
            incident + reflected,
            (incident - reflected) / self.left_specific_impedance_pa_s_per_m(),
        )
    }
    /// Evaluate the transmitted field for `x >= 0`.
    pub fn try_evaluate_right(
        self,
        x_m: f64,
    ) -> Result<AcousticInterfacePoint, AcousticInterfaceError> {
        if !x_m.is_finite() || x_m < 0.0 {
            return Err(AcousticInterfaceError::InvalidParameter {
                field: "x_m (right medium)",
                requirement: "finite and >= 0",
            });
        }
        let transmitted = self.incident_pressure_pa
            * self.pressure_transmission_coefficient()
            * Complex64::from_polar(1.0, -self.right_wavenumber_rad_per_m() * x_m);
        self.checked_point(
            x_m,
            transmitted,
            transmitted / self.right_specific_impedance_pa_s_per_m(),
        )
    }
    fn checked_point(
        self,
        x_m: f64,
        pressure_pa: Complex64,
        particle_velocity_m_per_s: Complex64,
    ) -> Result<AcousticInterfacePoint, AcousticInterfaceError> {
        if pressure_pa.re.is_finite()
            && pressure_pa.im.is_finite()
            && particle_velocity_m_per_s.re.is_finite()
            && particle_velocity_m_per_s.im.is_finite()
        {
            Ok(AcousticInterfacePoint {
                x_m,
                pressure_pa,
                particle_velocity_m_per_s,
            })
        } else {
            Err(AcousticInterfaceError::EvaluationOverflow)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn identical_media_are_reflection_free() {
        let reference =
            AcousticInterface1d::try_new(1.2, 343.0, 1.2, 343.0, 400.0, Complex64::new(2.0, -1.0))
                .unwrap();
        assert_abs_diff_eq!(
            reference.pressure_reflection_coefficient(),
            0.0,
            epsilon = 1e-14
        );
        assert_abs_diff_eq!(
            reference.pressure_transmission_coefficient(),
            1.0,
            epsilon = 1e-14
        );
        let (left, right) = (
            reference.try_evaluate_left(0.0).unwrap(),
            reference.try_evaluate_right(0.0).unwrap(),
        );
        assert_abs_diff_eq!(left.pressure_pa.re, right.pressure_pa.re, epsilon = 1e-14);
        assert_abs_diff_eq!(
            left.particle_velocity_m_per_s.im,
            right.particle_velocity_m_per_s.im,
            epsilon = 1e-14
        );
    }
    #[test]
    fn interface_continuity_and_power_balance_hold() {
        let reference = AcousticInterface1d::try_new(
            1.2,
            343.0,
            1000.0,
            1480.0,
            1000.0,
            Complex64::new(1.0, 0.4),
        )
        .unwrap();
        let (left, right) = (
            reference.try_evaluate_left(0.0).unwrap(),
            reference.try_evaluate_right(0.0).unwrap(),
        );
        assert_abs_diff_eq!(left.pressure_pa.re, right.pressure_pa.re, epsilon = 1e-12);
        assert_abs_diff_eq!(
            left.particle_velocity_m_per_s.re,
            right.particle_velocity_m_per_s.re,
            epsilon = 1e-15
        );
        let [incident, reflected, transmitted] = reference.intensities_w_per_m2();
        assert_abs_diff_eq!(incident - reflected, transmitted, epsilon = 1e-15);
    }
    #[test]
    fn checked_admission_rejects_bad_media_and_wrong_side() {
        assert!(matches!(
            AcousticInterface1d::try_new(0.0, 343.0, 1.2, 343.0, 100.0, Complex64::new(1.0, 0.0)),
            Err(AcousticInterfaceError::InvalidParameter { .. })
        ));
        let reference =
            AcousticInterface1d::try_new(1.2, 343.0, 1.0, 1000.0, 100.0, Complex64::new(1.0, 0.0))
                .unwrap();
        assert!(reference.try_evaluate_left(0.1).is_err());
        assert!(reference.try_evaluate_right(-0.1).is_err());
    }
}
