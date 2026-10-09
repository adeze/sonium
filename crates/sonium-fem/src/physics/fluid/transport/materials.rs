//! Coefficients for scalar and multi-species transport.

/// Linearized scalar advection-diffusion-reaction coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransportMaterial {
    pub diffusivity: [[f64; 3]; 3],
    pub velocity: [f64; 3],
    pub reaction_rate: f64,
    pub capacity: f64,
}

impl TransportMaterial {
    pub fn isotropic(diffusivity: f64, velocity: [f64; 3], reaction_rate: f64) -> Self {
        Self {
            diffusivity: [
                [diffusivity, 0.0, 0.0],
                [0.0, diffusivity, 0.0],
                [0.0, 0.0, diffusivity],
            ],
            velocity,
            reaction_rate,
            capacity: 1.0,
        }
    }

    pub fn with_capacity(
        diffusivity: f64,
        velocity: [f64; 3],
        reaction_rate: f64,
        capacity: f64,
    ) -> Self {
        Self {
            diffusivity: [
                [diffusivity, 0.0, 0.0],
                [0.0, diffusivity, 0.0],
                [0.0, 0.0, diffusivity],
            ],
            velocity,
            reaction_rate,
            capacity,
        }
    }

    pub fn peclet(&self, length: f64) -> f64 {
        let speed = self.velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
        let d = self.diffusivity[0][0].abs().max(f64::EPSILON);
        speed * length / (2.0 * d)
    }
}

/// Linear multi-species reaction model `R = J c + s`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearReactionNetwork {
    pub jacobian: Vec<Vec<f64>>,
    pub source: Vec<f64>,
}

impl LinearReactionNetwork {
    pub fn evaluate(&self, concentrations: &[f64]) -> Vec<f64> {
        self.jacobian
            .iter()
            .zip(self.source.iter())
            .map(|(row, &s)| {
                s + row
                    .iter()
                    .zip(concentrations.iter())
                    .map(|(&j, &c)| j * c)
                    .sum::<f64>()
            })
            .collect()
    }
}
