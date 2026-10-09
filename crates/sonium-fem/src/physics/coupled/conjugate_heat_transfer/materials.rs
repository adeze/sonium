//! Fluid and solid region properties for conjugate heat transfer.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjugateRegionKind {
    Solid,
    Fluid,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConjugateRegion {
    pub kind: ConjugateRegionKind,
    pub thermal_conductivity: f64,
    pub density: f64,
    pub specific_heat: f64,
    /// Prescribed/recovered fluid velocity in m/s. Ignored for solid regions.
    pub velocity: [f64; 3],
}

impl ConjugateRegion {
    pub fn solid(thermal_conductivity: f64, density: f64, specific_heat: f64) -> Self {
        Self {
            kind: ConjugateRegionKind::Solid,
            thermal_conductivity,
            density,
            specific_heat,
            velocity: [0.0; 3],
        }
    }

    pub fn fluid(
        thermal_conductivity: f64,
        density: f64,
        specific_heat: f64,
        velocity: [f64; 3],
    ) -> Self {
        Self {
            kind: ConjugateRegionKind::Fluid,
            thermal_conductivity,
            density,
            specific_heat,
            velocity,
        }
    }

    pub fn volumetric_heat_capacity(&self) -> f64 {
        self.density * self.specific_heat
    }

    pub fn advective_heat_flux(&self, temperature: f64) -> [f64; 3] {
        let scale = if self.kind == ConjugateRegionKind::Fluid {
            self.volumetric_heat_capacity() * temperature
        } else {
            0.0
        };
        self.velocity.map(|value| value * scale)
    }

    pub fn validate(self) -> Result<Self, ConjugateHeatTransferError> {
        if !self.thermal_conductivity.is_finite() || self.thermal_conductivity <= 0.0 {
            return Err(ConjugateHeatTransferError::NonPositiveConductivity);
        }
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(ConjugateHeatTransferError::NonPositiveDensity);
        }
        if !self.specific_heat.is_finite() || self.specific_heat <= 0.0 {
            return Err(ConjugateHeatTransferError::NonPositiveSpecificHeat);
        }
        if self.velocity.iter().any(|value| !value.is_finite()) {
            return Err(ConjugateHeatTransferError::NonFiniteVelocity);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjugateHeatTransferError {
    NonPositiveConductivity,
    NonPositiveDensity,
    NonPositiveSpecificHeat,
    NonFiniteVelocity,
}
