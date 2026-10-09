//! Grey radiative material and angular quadrature.

use std::f64::consts::PI;

pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiativeMaterial {
    pub absorption_coefficient: f64,
    pub scattering_coefficient: f64,
    /// Henyey-Greenstein asymmetry parameter.
    pub asymmetry_parameter: f64,
    pub refractive_index: f64,
    pub density: f64,
    pub specific_heat: f64,
}

impl RadiativeMaterial {
    pub fn validate(self) -> Result<Self, RadiationTransportError> {
        if self.absorption_coefficient < 0.0
            || self.scattering_coefficient < 0.0
            || !self.absorption_coefficient.is_finite()
            || !self.scattering_coefficient.is_finite()
        {
            return Err(RadiationTransportError::NegativeExtinction);
        }
        if !(-1.0..1.0).contains(&self.asymmetry_parameter) {
            return Err(RadiationTransportError::InvalidAsymmetry);
        }
        if self.refractive_index <= 0.0
            || self.density <= 0.0
            || self.specific_heat <= 0.0
            || !self.refractive_index.is_finite()
            || !self.density.is_finite()
            || !self.specific_heat.is_finite()
        {
            return Err(RadiationTransportError::NonPositiveMaterialCoefficient);
        }
        Ok(self)
    }

    pub fn extinction_coefficient(&self) -> f64 {
        self.absorption_coefficient + self.scattering_coefficient
    }

    pub fn single_scattering_albedo(&self) -> f64 {
        if self.extinction_coefficient() == 0.0 {
            0.0
        } else {
            self.scattering_coefficient / self.extinction_coefficient()
        }
    }

    /// Grey blackbody radiance integrated over wavelength, W/(m^2 sr).
    pub fn blackbody_radiance(&self, temperature: f64) -> f64 {
        assert!(temperature >= 0.0);
        self.refractive_index.powi(2) * STEFAN_BOLTZMANN * temperature.powi(4) / PI
    }

    pub fn phase_function(&self, incoming: [f64; 3], outgoing: [f64; 3]) -> f64 {
        let cosine = incoming
            .iter()
            .zip(outgoing)
            .map(|(left, right)| left * right)
            .sum::<f64>()
            .clamp(-1.0, 1.0);
        let g = self.asymmetry_parameter;
        (1.0 - g * g) / (1.0 + g * g - 2.0 * g * cosine).powf(1.5)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AngularQuadrature {
    pub directions: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
}

impl AngularQuadrature {
    /// Eight-octant S2 quadrature exact for constants and odd first moments.
    pub fn s2_3d() -> Self {
        let coordinate = 1.0 / 3.0_f64.sqrt();
        let mut directions = Vec::with_capacity(8);
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    directions.push([sx * coordinate, sy * coordinate, sz * coordinate]);
                }
            }
        }
        Self {
            directions,
            weights: vec![4.0 * PI / 8.0; 8],
        }
    }

    pub fn validate(&self) -> Result<(), RadiationTransportError> {
        if self.directions.is_empty() || self.directions.len() != self.weights.len() {
            return Err(RadiationTransportError::InvalidAngularQuadrature);
        }
        for (direction, weight) in self.directions.iter().zip(&self.weights) {
            let norm = direction
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if (norm - 1.0).abs() > 1e-12 || !weight.is_finite() || *weight <= 0.0 {
                return Err(RadiationTransportError::InvalidAngularQuadrature);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiationTransportError {
    NegativeExtinction,
    InvalidAsymmetry,
    NonPositiveMaterialCoefficient,
    InvalidAngularQuadrature,
}
