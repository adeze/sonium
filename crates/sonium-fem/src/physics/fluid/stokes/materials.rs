//! Fluid material properties for Stokes/Navier-Stokes

/// Incompressible Newtonian fluid properties.
#[derive(Debug, Clone, Copy)]
pub struct FluidProperties {
    /// Dynamic viscosity [Pa*s]
    pub mu: f64,
    /// Density [kg/m^3]
    pub rho: f64,
}

impl FluidProperties {
    /// Kinematic viscosity nu = mu / rho [m^2/s]
    pub fn kinematic_viscosity(&self) -> f64 {
        self.mu / self.rho
    }
}
