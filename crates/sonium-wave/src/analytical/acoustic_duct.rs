//! Lossless below-cutoff transfer-matrix reference for a uniform acoustic duct.
//!
//! The model uses `exp(+iωt)`. A wave travelling toward increasing `x` is
//! `exp(-ikx)`, and positive volume velocity is toward increasing `x`. It is
//! valid only for the plane-wave regime below the first transverse-mode cutoff;
//! it deliberately does not model discontinuity evanescent modes, radiation,
//! mean flow, or thermoviscous loss.

use num_complex::Complex64;
use std::fmt;

/// Input or evaluation error for [`LosslessDuct1d`].
#[derive(Debug, Clone, PartialEq)]
pub enum AcousticDuctError {
    InvalidParameter {
        field: &'static str,
        requirement: &'static str,
    },
    EvaluationOverflow,
}

impl fmt::Display for AcousticDuctError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(f, "acoustic duct parameter {field} must be {requirement}")
            }
            Self::EvaluationOverflow => {
                write!(f, "acoustic duct result cannot be represented in binary64")
            }
        }
    }
}
impl std::error::Error for AcousticDuctError {}

/// Pressure and positive-`x` volume velocity at a duct cross-section.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuctState {
    pub pressure_pa: Complex64,
    pub volume_velocity_m3_per_s: Complex64,
}

/// A uniform, lossless duct segment with constant cross-sectional area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LosslessDuct1d {
    density_kg_per_m3: f64,
    sound_speed_m_per_s: f64,
    area_m2: f64,
    length_m: f64,
    frequency_hz: f64,
    /// Spatial sign paired with the harmonic convention. The legacy scalar
    /// constructor uses `-1` (`exp(+iωt)` with `exp(-ikx)`).
    spatial_sign: i8,
}

impl LosslessDuct1d {
    /// Construct a checked plane-wave duct. Zero length is permitted and is
    /// the identity transfer; area, density, speed, and frequency are positive.
    pub fn try_new(
        density_kg_per_m3: f64,
        sound_speed_m_per_s: f64,
        area_m2: f64,
        length_m: f64,
        frequency_hz: f64,
    ) -> Result<Self, AcousticDuctError> {
        Self::try_new_with_spatial_sign(
            density_kg_per_m3,
            sound_speed_m_per_s,
            area_m2,
            length_m,
            frequency_hz,
            -1,
        )
    }

    fn try_new_with_spatial_sign(
        density_kg_per_m3: f64,
        sound_speed_m_per_s: f64,
        area_m2: f64,
        length_m: f64,
        frequency_hz: f64,
        spatial_sign: i8,
    ) -> Result<Self, AcousticDuctError> {
        for (field, value, positive) in [
            ("density_kg_per_m3", density_kg_per_m3, true),
            ("sound_speed_m_per_s", sound_speed_m_per_s, true),
            ("area_m2", area_m2, true),
            ("length_m", length_m, false),
            ("frequency_hz", frequency_hz, true),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) || (!positive && value < 0.0) {
                return Err(AcousticDuctError::InvalidParameter {
                    field,
                    requirement: if positive {
                        "finite and positive"
                    } else {
                        "finite and non-negative"
                    },
                });
            }
        }
        let result = Self {
            density_kg_per_m3,
            sound_speed_m_per_s,
            area_m2,
            length_m,
            frequency_hz,
            spatial_sign,
        };
        if !result.volume_velocity_impedance_pa_s_per_m3().is_finite()
            || !result.wavenumber_rad_per_m().is_finite()
            || !(result.wavenumber_rad_per_m() * length_m).is_finite()
        {
            return Err(AcousticDuctError::EvaluationOverflow);
        }
        Ok(result)
    }

    /// Construct a duct from the typed XEM acoustic-medium contract.
    ///
    /// The scalar constructor remains equivalent to `ExpPlusIwt`; this
    /// adapter also supports `ExpMinusIwt` by carrying the XEM spatial sign
    /// into the transfer matrix.
    pub fn try_from_xem(
        medium: sonium_xem::AcousticMedium,
        area_m2: f64,
        length_m: f64,
    ) -> Result<Self, AcousticDuctError> {
        let (_, spatial_sign) = medium.conventions.harmonic_time.signs();
        Self::try_new_with_spatial_sign(
            medium.density.get(),
            medium.speed_of_sound.get(),
            area_m2,
            length_m,
            medium.frequency.get(),
            spatial_sign,
        )
    }

    /// Plane-wave wavenumber in rad/m.
    pub fn wavenumber_rad_per_m(self) -> f64 {
        std::f64::consts::TAU * self.frequency_hz / self.sound_speed_m_per_s
    }
    /// Volume-velocity characteristic impedance `ρc/S` in Pa·s/m³.
    pub fn volume_velocity_impedance_pa_s_per_m3(self) -> f64 {
        self.density_kg_per_m3 * self.sound_speed_m_per_s / self.area_m2
    }

    /// Propagate a state from the inlet (`x=0`) to the outlet (`x=L`).
    pub fn try_transfer(self, inlet: DuctState) -> Result<DuctState, AcousticDuctError> {
        if [
            inlet.pressure_pa.re,
            inlet.pressure_pa.im,
            inlet.volume_velocity_m3_per_s.re,
            inlet.volume_velocity_m3_per_s.im,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err(AcousticDuctError::InvalidParameter {
                field: "inlet state",
                requirement: "finite real and imaginary components",
            });
        }
        let theta = self.wavenumber_rad_per_m() * self.length_m;
        let (sin, cos) = theta.sin_cos();
        let z = self.volume_velocity_impedance_pa_s_per_m3();
        let propagation = Complex64::new(0.0, f64::from(self.spatial_sign) * sin);
        let outlet = DuctState {
            pressure_pa: inlet.pressure_pa * cos + propagation * z * inlet.volume_velocity_m3_per_s,
            volume_velocity_m3_per_s: propagation / z * inlet.pressure_pa
                + inlet.volume_velocity_m3_per_s * cos,
        };
        if [
            outlet.pressure_pa.re,
            outlet.pressure_pa.im,
            outlet.volume_velocity_m3_per_s.re,
            outlet.volume_velocity_m3_per_s.im,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            Ok(outlet)
        } else {
            Err(AcousticDuctError::EvaluationOverflow)
        }
    }

    /// Transform a load impedance at `x=L` to the inlet. A matched load
    /// returns the characteristic impedance; a rigid load produces a purely
    /// reactive impedance where finite.
    pub fn try_input_impedance(
        self,
        load_pa_s_per_m3: Complex64,
    ) -> Result<Complex64, AcousticDuctError> {
        if !load_pa_s_per_m3.re.is_finite() || !load_pa_s_per_m3.im.is_finite() {
            return Err(AcousticDuctError::InvalidParameter {
                field: "load_pa_s_per_m3",
                requirement: "finite real and imaginary components",
            });
        }
        let (sin, cos) = (self.wavenumber_rad_per_m() * self.length_m).sin_cos();
        let z = self.volume_velocity_impedance_pa_s_per_m3();
        let propagation = Complex64::new(0.0, f64::from(self.spatial_sign) * sin);
        let denominator = Complex64::new(cos, 0.0) + propagation * load_pa_s_per_m3 / z;
        if denominator.norm_sqr() == 0.0 {
            return Err(AcousticDuctError::EvaluationOverflow);
        }
        let input = (load_pa_s_per_m3 * cos + propagation * z) / denominator;
        if input.re.is_finite() && input.im.is_finite() {
            Ok(input)
        } else {
            Err(AcousticDuctError::EvaluationOverflow)
        }
    }

    /// Time-averaged acoustic power in W for a pressure/volume-velocity state.
    pub fn time_average_power_w(state: DuctState) -> f64 {
        0.5 * (state.pressure_pa * state.volume_velocity_m3_per_s.conj()).re
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn zero_length_and_matched_load_are_identities() {
        let duct = LosslessDuct1d::try_new(1.2, 343.0, 0.01, 0.0, 500.0).unwrap();
        let inlet = DuctState {
            pressure_pa: Complex64::new(2.0, -1.0),
            volume_velocity_m3_per_s: Complex64::new(0.001, 0.0003),
        };
        assert_eq!(duct.try_transfer(inlet).unwrap(), inlet);
        assert_eq!(
            duct.try_input_impedance(Complex64::new(
                duct.volume_velocity_impedance_pa_s_per_m3(),
                0.0
            ))
            .unwrap(),
            Complex64::new(duct.volume_velocity_impedance_pa_s_per_m3(), 0.0)
        );
    }
    #[test]
    fn lossless_transfer_preserves_power() {
        let duct = LosslessDuct1d::try_new(1.204, 343.0, 0.008, 0.37, 430.0).unwrap();
        let inlet = DuctState {
            pressure_pa: Complex64::new(3.0, -1.0),
            volume_velocity_m3_per_s: Complex64::new(0.001, 0.0002),
        };
        let outlet = duct.try_transfer(inlet).unwrap();
        assert_abs_diff_eq!(
            LosslessDuct1d::time_average_power_w(inlet),
            LosslessDuct1d::time_average_power_w(outlet),
            epsilon = 1.0e-14
        );
    }
    #[test]
    fn checked_admission_rejects_invalid_geometry_and_load() {
        assert!(LosslessDuct1d::try_new(1.2, 343.0, 0.0, 1.0, 100.0).is_err());
        let duct = LosslessDuct1d::try_new(1.2, 343.0, 0.01, 1.0, 100.0).unwrap();
        assert!(
            duct.try_input_impedance(Complex64::new(f64::NAN, 0.0))
                .is_err()
        );
    }

    #[test]
    fn xem_medium_adapter_preserves_harmonic_sign() {
        let mut plus = sonium_xem::ConventionRegistry::default();
        plus.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let plus_medium = sonium_xem::AcousticMedium::try_new_with_conventions(
            sonium_xem::FrequencyHz::new(430.0),
            sonium_xem::SpeedOfSoundMPerS::new(343.0),
            sonium_xem::DensityKgM3::new(1.204),
            plus,
        )
        .unwrap();
        let minus_medium = sonium_xem::AcousticMedium::try_new(430.0, 343.0, 1.204).unwrap();
        let plus_duct = LosslessDuct1d::try_from_xem(plus_medium, 0.008, 0.37).unwrap();
        let minus_duct = LosslessDuct1d::try_from_xem(minus_medium, 0.008, 0.37).unwrap();
        let inlet = DuctState {
            pressure_pa: Complex64::new(3.0, 0.0),
            volume_velocity_m3_per_s: Complex64::new(0.001, 0.0),
        };
        let plus_outlet = plus_duct.try_transfer(inlet).unwrap();
        let minus_outlet = minus_duct.try_transfer(inlet).unwrap();
        assert!((plus_outlet.pressure_pa - minus_outlet.pressure_pa.conj()).norm() < 1.0e-14);
        assert!(
            (plus_outlet.volume_velocity_m3_per_s - minus_outlet.volume_velocity_m3_per_s.conj())
                .norm()
                < 1.0e-14
        );
    }
}
