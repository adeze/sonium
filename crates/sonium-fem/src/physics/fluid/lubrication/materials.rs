//! Lubricant and cavitation state models.

/// Constant-property Newtonian lubricant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lubricant {
    /// Dynamic viscosity in Pa s.
    pub dynamic_viscosity: f64,
    /// Density in kg/m^3.
    pub density: f64,
    /// Ambient/cavitation pressure in Pa.
    pub ambient_pressure: f64,
    /// Strictly positive floor applied to film thicknesses.
    pub minimum_film_thickness: f64,
}

impl Lubricant {
    pub fn validate(self) -> Result<Self, LubricationError> {
        if !self.dynamic_viscosity.is_finite() || self.dynamic_viscosity <= 0.0 {
            return Err(LubricationError::NonPositiveViscosity);
        }
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(LubricationError::NonPositiveDensity);
        }
        if !self.minimum_film_thickness.is_finite() || self.minimum_film_thickness <= 0.0 {
            return Err(LubricationError::NonPositiveFilmThickness);
        }
        if !self.ambient_pressure.is_finite() {
            return Err(LubricationError::NonFinitePressure);
        }
        Ok(self)
    }

    /// Reynolds mobility `h^3 / (12 mu)`.
    #[inline]
    pub fn mobility(&self, film_thickness: f64) -> f64 {
        let h = film_thickness.max(self.minimum_film_thickness);
        h * h * h / (12.0 * self.dynamic_viscosity)
    }
}

/// Elrod-Adams-style pressure/saturation state.
///
/// Full-film nodes have `fraction = 1`; cavitated nodes retain a bounded mass
/// fraction while their pressure is clamped to ambient pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CavitationState {
    pub pressure: f64,
    pub fraction: f64,
}

impl CavitationState {
    pub fn from_trial_pressure(trial_pressure: f64, ambient_pressure: f64) -> Self {
        if trial_pressure >= ambient_pressure {
            Self {
                pressure: trial_pressure,
                fraction: 1.0,
            }
        } else {
            Self {
                pressure: ambient_pressure,
                fraction: (trial_pressure / ambient_pressure.max(f64::MIN_POSITIVE))
                    .clamp(0.0, 1.0),
            }
        }
    }

    pub fn is_cavitated(&self) -> bool {
        self.fraction < 1.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LubricationError {
    NonPositiveViscosity,
    NonPositiveDensity,
    NonPositiveFilmThickness,
    NonFinitePressure,
}
