//! Inductionless-MHD material and imposed-field parameters.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MhdMaterial {
    pub density: f64,
    pub dynamic_viscosity: f64,
    pub electrical_conductivity: f64,
    pub imposed_magnetic_flux_density: [f64; 3],
}

impl MhdMaterial {
    pub fn validate(self) -> Result<Self, MhdError> {
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(MhdError::NonPositiveDensity);
        }
        if !self.dynamic_viscosity.is_finite() || self.dynamic_viscosity <= 0.0 {
            return Err(MhdError::NonPositiveViscosity);
        }
        if !self.electrical_conductivity.is_finite() || self.electrical_conductivity <= 0.0 {
            return Err(MhdError::NonPositiveElectricalConductivity);
        }
        if self
            .imposed_magnetic_flux_density
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(MhdError::NonFiniteMagneticField);
        }
        Ok(self)
    }

    pub fn kinematic_viscosity(&self) -> f64 {
        self.dynamic_viscosity / self.density
    }

    pub fn magnetic_field_magnitude(&self) -> f64 {
        self.imposed_magnetic_flux_density
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    pub fn hartmann_number(&self, length_scale: f64) -> f64 {
        length_scale
            * self.magnetic_field_magnitude()
            * (self.electrical_conductivity / self.dynamic_viscosity).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MhdError {
    NonPositiveDensity,
    NonPositiveViscosity,
    NonPositiveElectricalConductivity,
    NonFiniteMagneticField,
}

#[inline]
pub(crate) fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
