//! FSI material and algorithm parameters.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsiFluid {
    pub density: f64,
    pub dynamic_viscosity: f64,
}

impl FsiFluid {
    pub fn validate(self) -> Result<Self, FsiError> {
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(FsiError::NonPositiveFluidDensity);
        }
        if !self.dynamic_viscosity.is_finite() || self.dynamic_viscosity <= 0.0 {
            return Err(FsiError::NonPositiveFluidViscosity);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsiCouplingConfig {
    pub initial_relaxation: f64,
    pub minimum_relaxation: f64,
    pub maximum_relaxation: f64,
    pub interface_tolerance: f64,
    pub maximum_iterations: usize,
}

impl Default for FsiCouplingConfig {
    fn default() -> Self {
        Self {
            initial_relaxation: 0.5,
            minimum_relaxation: 0.05,
            maximum_relaxation: 1.0,
            interface_tolerance: 1e-8,
            maximum_iterations: 50,
        }
    }
}

impl FsiCouplingConfig {
    pub fn validate(self) -> Result<Self, FsiError> {
        if !(0.0..=1.0).contains(&self.initial_relaxation)
            || !(0.0..=1.0).contains(&self.minimum_relaxation)
            || !(0.0..=1.0).contains(&self.maximum_relaxation)
            || self.minimum_relaxation > self.maximum_relaxation
        {
            return Err(FsiError::InvalidRelaxation);
        }
        if !self.interface_tolerance.is_finite() || self.interface_tolerance <= 0.0 {
            return Err(FsiError::InvalidTolerance);
        }
        if self.maximum_iterations == 0 {
            return Err(FsiError::ZeroIterations);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsiError {
    NonPositiveFluidDensity,
    NonPositiveFluidViscosity,
    InvalidRelaxation,
    InvalidTolerance,
    ZeroIterations,
    InvalidMortarConfig,
    InvalidMortarSurface,
    MortarProjectionCoverage,
    InterfaceSizeMismatch,
    InterfaceIterationsDidNotConverge,
}
