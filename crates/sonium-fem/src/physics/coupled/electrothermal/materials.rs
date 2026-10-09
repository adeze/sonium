//! Temperature-dependent electrothermal materials.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectrothermalMaterial {
    pub reference_electrical_conductivity: f64,
    /// Linear resistivity temperature coefficient, 1/K.
    pub resistivity_temperature_coefficient: f64,
    pub reference_temperature: f64,
    pub relative_permeability: f64,
    pub thermal_conductivity: f64,
    pub density: f64,
    pub specific_heat: f64,
}

impl ElectrothermalMaterial {
    pub fn validate(self) -> Result<Self, ElectrothermalError> {
        if !self.reference_electrical_conductivity.is_finite()
            || self.reference_electrical_conductivity <= 0.0
        {
            return Err(ElectrothermalError::NonPositiveElectricalConductivity);
        }
        if !self.relative_permeability.is_finite() || self.relative_permeability <= 0.0 {
            return Err(ElectrothermalError::NonPositivePermeability);
        }
        if !self.thermal_conductivity.is_finite() || self.thermal_conductivity <= 0.0 {
            return Err(ElectrothermalError::NonPositiveThermalConductivity);
        }
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(ElectrothermalError::NonPositiveDensity);
        }
        if !self.specific_heat.is_finite() || self.specific_heat <= 0.0 {
            return Err(ElectrothermalError::NonPositiveSpecificHeat);
        }
        if !self.resistivity_temperature_coefficient.is_finite()
            || !self.reference_temperature.is_finite()
        {
            return Err(ElectrothermalError::NonFiniteCoefficient);
        }
        Ok(self)
    }

    /// Conductivity from a linear resistivity law.
    pub fn electrical_conductivity(&self, temperature: f64) -> f64 {
        let denominator = 1.0
            + self.resistivity_temperature_coefficient * (temperature - self.reference_temperature);
        self.reference_electrical_conductivity / denominator.max(f64::MIN_POSITIVE)
    }

    pub fn magnetic_permeability(&self) -> f64 {
        4.0e-7 * std::f64::consts::PI * self.relative_permeability
    }

    pub fn volumetric_heat_capacity(&self) -> f64 {
        self.density * self.specific_heat
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectrothermalError {
    NonPositiveElectricalConductivity,
    NonPositivePermeability,
    NonPositiveThermalConductivity,
    NonPositiveDensity,
    NonPositiveSpecificHeat,
    NonFiniteCoefficient,
}
