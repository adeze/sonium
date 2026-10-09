//! Electrochemical constants and kinetic laws.

pub const FARADAY: f64 = 96_485.332_12;
pub const GAS_CONSTANT: f64 = 8.314_462_618;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonicSpecies {
    pub diffusivity: f64,
    pub valence: f64,
    pub mobility: f64,
}

impl IonicSpecies {
    pub fn from_diffusivity(diffusivity: f64, valence: f64, temperature: f64) -> Self {
        Self {
            diffusivity,
            valence,
            mobility: diffusivity * FARADAY / (GAS_CONSTANT * temperature),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButlerVolmerKinetics {
    pub exchange_current_density: f64,
    pub alpha_anodic: f64,
    pub alpha_cathodic: f64,
    pub temperature: f64,
}

impl ButlerVolmerKinetics {
    pub fn current_density(&self, overpotential: f64) -> f64 {
        let scale = FARADAY * overpotential / (GAS_CONSTANT * self.temperature);
        self.exchange_current_density
            * ((self.alpha_anodic * scale).exp() - (-self.alpha_cathodic * scale).exp())
    }

    pub fn linearized_charge_transfer_resistance(&self) -> f64 {
        GAS_CONSTANT * self.temperature
            / (FARADAY * self.exchange_current_density * (self.alpha_anodic + self.alpha_cathodic))
    }
}
