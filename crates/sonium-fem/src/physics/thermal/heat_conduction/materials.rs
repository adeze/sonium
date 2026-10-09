//! Thermal material properties

/// Stefan-Boltzmann constant (W/(m^2·K^4)).
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Thermal material properties.
#[derive(Debug, Clone, Copy)]
pub struct ThermalMaterial {
    /// Thermal conductivity k (W/(m·K)).
    pub conductivity: f64,
    /// Density rho (kg/m^3).
    pub density: f64,
    /// Specific heat capacity c (J/(kg·K)).
    pub specific_heat: f64,
    /// Surface emissivity for radiation (0-1).
    pub emissivity: f64,
}

impl ThermalMaterial {
    pub fn new(conductivity: f64, density: f64, specific_heat: f64) -> Self {
        Self {
            conductivity,
            density,
            specific_heat,
            emissivity: 0.0,
        }
    }

    /// With radiation emissivity.
    pub fn with_emissivity(mut self, emissivity: f64) -> Self {
        self.emissivity = emissivity;
        self
    }

    /// Copper: k=401, rho=8960, c=385.
    pub fn copper() -> Self {
        Self::new(401.0, 8960.0, 385.0)
    }

    /// Aluminum: k=237, rho=2700, c=897.
    pub fn aluminum() -> Self {
        Self::new(237.0, 2700.0, 897.0)
    }

    /// Steel: k=50, rho=7800, c=500.
    pub fn steel() -> Self {
        Self::new(50.0, 7800.0, 500.0)
    }

    /// Air at 20°C: k=0.026, rho=1.2, c=1005.
    pub fn air() -> Self {
        Self::new(0.026, 1.2, 1005.0)
    }

    /// Voice coil wire (copper with emissivity).
    pub fn voice_coil_wire() -> Self {
        Self::copper().with_emissivity(0.8)
    }

    /// Thermal diffusivity alpha = k / (rho * c) (m^2/s).
    pub fn diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }

    /// Volumetric heat capacity rho * c (J/(m^3·K)).
    pub fn volumetric_heat_capacity(&self) -> f64 {
        self.density * self.specific_heat
    }
}

/// Convection boundary condition parameters.
#[derive(Debug, Clone, Copy)]
pub struct ConvectionCoeff {
    /// Convection heat transfer coefficient h (W/(m^2·K)).
    pub h: f64,
    /// Ambient temperature T_inf (K).
    pub t_ambient: f64,
}

impl ConvectionCoeff {
    pub fn new(h: f64, t_ambient: f64) -> Self {
        Self { h, t_ambient }
    }

    /// Natural convection in air (h ~ 5-25 W/(m^2·K)).
    pub fn natural_convection(t_ambient: f64) -> Self {
        Self::new(10.0, t_ambient)
    }

    /// Forced convection in air (h ~ 25-250 W/(m^2·K)).
    pub fn forced_convection(h: f64, t_ambient: f64) -> Self {
        Self::new(h, t_ambient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copper_diffusivity() {
        let cu = ThermalMaterial::copper();
        let alpha = cu.diffusivity();
        // Copper: ~1.17e-4 m^2/s
        assert!(alpha > 1e-4 && alpha < 1.5e-4, "alpha = {alpha}");
    }
}
