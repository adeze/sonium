//! Electromagnetic material properties for eddy current analysis

/// Electromagnetic conductor properties.
#[derive(Debug, Clone, Copy)]
pub struct ConductorMaterial {
    /// Relative permeability mu_r.
    pub mu_r: f64,
    /// Electrical conductivity sigma (S/m).
    pub conductivity: f64,
}

impl ConductorMaterial {
    pub fn new(mu_r: f64, conductivity: f64) -> Self {
        Self { mu_r, conductivity }
    }

    /// Non-conducting magnetic material (iron without eddy currents, for comparison).
    pub fn non_conducting_iron(mu_r: f64) -> Self {
        Self::new(mu_r, 0.0)
    }

    /// Low-carbon steel pole piece: mu_r ~= 2000, sigma ~= 6.99e6 S/m.
    pub fn steel_pole_piece() -> Self {
        Self::new(2000.0, 6.99e6)
    }

    /// Copper voice coil: mu_r ~= 1, sigma ~= 5.96e7 S/m.
    pub fn copper() -> Self {
        Self::new(1.0, 5.96e7)
    }

    /// Aluminum shorting ring: mu_r ~= 1, sigma ~= 3.77e7 S/m.
    pub fn aluminum() -> Self {
        Self::new(1.0, 3.77e7)
    }

    /// Air / non-conducting region.
    pub fn air() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Absolute permeability mu = mu_r * mu_0.
    pub fn permeability(&self) -> f64 {
        self.mu_r * 4.0 * std::f64::consts::PI * 1e-7
    }

    /// Skin depth at a given frequency: delta = sqrt(2 / (omega * mu * sigma)).
    pub fn skin_depth(&self, frequency: f64) -> f64 {
        if self.conductivity <= 0.0 {
            return f64::INFINITY;
        }
        let omega = 2.0 * std::f64::consts::PI * frequency;
        (2.0 / (omega * self.permeability() * self.conductivity)).sqrt()
    }

    /// Is this material conducting (has eddy currents)?
    pub fn is_conducting(&self) -> bool {
        self.conductivity > 0.0
    }
}

/// Source current density.
#[derive(Debug, Clone)]
pub struct CurrentSource {
    /// Current density J_s (A/m^2) as a function of position.
    /// Returns [Jx, Jy, Jz].
    direction: [f64; 3],
    /// Magnitude (A/m^2).
    magnitude: f64,
}

impl CurrentSource {
    /// Uniform current density in a given direction.
    pub fn uniform(magnitude: f64, direction: [f64; 3]) -> Self {
        let len = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        Self {
            direction: [direction[0] / len, direction[1] / len, direction[2] / len],
            magnitude,
        }
    }

    /// Current density vector at any point.
    pub fn density(&self) -> [f64; 3] {
        [
            self.magnitude * self.direction[0],
            self.magnitude * self.direction[1],
            self.magnitude * self.direction[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skin_depth_steel() {
        let steel = ConductorMaterial::steel_pole_piece();
        let delta = steel.skin_depth(1000.0);
        // At 1 kHz: delta ~= 0.19 mm for steel with mu_r=2000
        assert!(delta > 0.0001 && delta < 0.001, "skin depth = {delta}");
    }

    #[test]
    fn test_skin_depth_copper() {
        let cu = ConductorMaterial::copper();
        let delta = cu.skin_depth(1000.0);
        // At 1 kHz: delta ~= 2.1 mm for copper
        assert!(delta > 0.001 && delta < 0.005, "skin depth = {delta}");
    }

    #[test]
    fn test_skin_depth_air() {
        let air = ConductorMaterial::air();
        assert!(air.skin_depth(1000.0).is_infinite());
    }
}
