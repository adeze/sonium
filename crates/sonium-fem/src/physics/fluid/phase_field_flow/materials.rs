//! Diffuse-interface material properties.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseFieldFluid {
    pub density_minus: f64,
    pub density_plus: f64,
    pub viscosity_minus: f64,
    pub viscosity_plus: f64,
    pub surface_tension: f64,
    pub interface_thickness: f64,
    pub mobility: f64,
}

impl PhaseFieldFluid {
    pub fn validate(self) -> Result<Self, PhaseFieldError> {
        if self.density_minus <= 0.0
            || self.density_plus <= 0.0
            || !self.density_minus.is_finite()
            || !self.density_plus.is_finite()
        {
            return Err(PhaseFieldError::NonPositiveDensity);
        }
        if self.viscosity_minus <= 0.0
            || self.viscosity_plus <= 0.0
            || !self.viscosity_minus.is_finite()
            || !self.viscosity_plus.is_finite()
        {
            return Err(PhaseFieldError::NonPositiveViscosity);
        }
        if self.surface_tension <= 0.0 || !self.surface_tension.is_finite() {
            return Err(PhaseFieldError::NonPositiveSurfaceTension);
        }
        if self.interface_thickness <= 0.0 || !self.interface_thickness.is_finite() {
            return Err(PhaseFieldError::NonPositiveInterfaceThickness);
        }
        if self.mobility <= 0.0 || !self.mobility.is_finite() {
            return Err(PhaseFieldError::NonPositiveMobility);
        }
        Ok(self)
    }

    /// Double-well coefficient yielding the requested surface tension with the
    /// quartic free energy and the matching gradient coefficient below.
    pub fn double_well_coefficient(&self) -> f64 {
        3.0 * self.surface_tension / (2.0 * std::f64::consts::SQRT_2 * self.interface_thickness)
    }

    pub fn gradient_energy_coefficient(&self) -> f64 {
        self.double_well_coefficient() * self.interface_thickness.powi(2)
    }

    pub fn density(&self, phase: f64) -> f64 {
        let fraction = 0.5 * (phase.clamp(-1.0, 1.0) + 1.0);
        self.density_minus + fraction * (self.density_plus - self.density_minus)
    }

    pub fn viscosity(&self, phase: f64) -> f64 {
        let fraction = 0.5 * (phase.clamp(-1.0, 1.0) + 1.0);
        self.viscosity_minus + fraction * (self.viscosity_plus - self.viscosity_minus)
    }

    pub fn bulk_free_energy(&self, phase: f64) -> f64 {
        0.25 * self.double_well_coefficient() * (phase * phase - 1.0).powi(2)
    }

    pub fn chemical_potential_bulk(&self, phase: f64) -> f64 {
        self.double_well_coefficient() * (phase.powi(3) - phase)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseFieldError {
    NonPositiveDensity,
    NonPositiveViscosity,
    NonPositiveSurfaceTension,
    NonPositiveInterfaceThickness,
    NonPositiveMobility,
}
