//! Thermoelectric material coefficients.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoelectricMaterial {
    /// Electrical conductivity, S/m.
    pub electrical_conductivity: f64,
    /// Thermal conductivity, W/(m K).
    pub thermal_conductivity: f64,
    /// Seebeck coefficient at `reference_temperature`, V/K.
    pub seebeck_coefficient: f64,
    /// Linear slope `d alpha / dT`, V/K^2.
    pub seebeck_temperature_coefficient: f64,
    pub reference_temperature: f64,
    pub density: f64,
    pub specific_heat: f64,
}

impl ThermoelectricMaterial {
    pub fn validate(self) -> Result<Self, ThermoelectricError> {
        if !self.electrical_conductivity.is_finite() || self.electrical_conductivity <= 0.0 {
            return Err(ThermoelectricError::NonPositiveElectricalConductivity);
        }
        if !self.thermal_conductivity.is_finite() || self.thermal_conductivity <= 0.0 {
            return Err(ThermoelectricError::NonPositiveThermalConductivity);
        }
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(ThermoelectricError::NonPositiveDensity);
        }
        if !self.specific_heat.is_finite() || self.specific_heat <= 0.0 {
            return Err(ThermoelectricError::NonPositiveSpecificHeat);
        }
        if !self.seebeck_coefficient.is_finite()
            || !self.seebeck_temperature_coefficient.is_finite()
            || !self.reference_temperature.is_finite()
        {
            return Err(ThermoelectricError::NonFiniteCoefficient);
        }
        Ok(self)
    }

    #[inline]
    pub fn electrical_resistivity(&self) -> f64 {
        1.0 / self.electrical_conductivity
    }

    #[inline]
    pub fn seebeck(&self, temperature: f64) -> f64 {
        self.seebeck_coefficient
            + self.seebeck_temperature_coefficient * (temperature - self.reference_temperature)
    }

    #[inline]
    pub fn peltier(&self, temperature: f64) -> f64 {
        self.seebeck(temperature) * temperature
    }

    #[inline]
    pub fn thomson(&self, temperature: f64) -> f64 {
        temperature * self.seebeck_temperature_coefficient
    }

    #[inline]
    pub fn figure_of_merit(&self, temperature: f64) -> f64 {
        let alpha = self.seebeck(temperature);
        alpha * alpha * self.electrical_conductivity * temperature / self.thermal_conductivity
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermoelectricError {
    NonPositiveElectricalConductivity,
    NonPositiveThermalConductivity,
    NonPositiveDensity,
    NonPositiveSpecificHeat,
    NonFiniteCoefficient,
}
