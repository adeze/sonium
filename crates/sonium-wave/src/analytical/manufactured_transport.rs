//! Manufactured steady one-dimensional advection-diffusion-reaction references.
//!
//! The constant-coefficient scalar transport convention is
//!
//! ```text
//! -κ ∂²u/∂x² + v ∂u/∂x + σu = s(x),
//! ```
//!
//! with positive diffusivity `κ`, signed advective velocity `v`, and a
//! non-negative reaction/sink rate `σ`.  It is a real-valued reference for
//! thermal or scalar transport formulations; it is not an acoustic phasor.

use std::fmt;

/// Invalid input to a manufactured one-dimensional transport reference.
#[derive(Debug, Clone, PartialEq)]
pub enum TransportManufacturedError {
    /// A model parameter is non-finite or outside the documented domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain or invariant.
        requirement: &'static str,
    },
    /// A sampled spatial interval is invalid.
    InvalidInterval {
        /// Start coordinate in metres.
        x_min: f64,
        /// End coordinate in metres.
        x_max: f64,
    },
    /// An evaluated phase or source term is not representable.
    EvaluationOverflow,
}

impl fmt::Display for TransportManufacturedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => write!(
                f,
                "manufactured transport parameter {field} must be {requirement}"
            ),
            Self::InvalidInterval { x_min, x_max } => write!(
                f,
                "manufactured transport interval must be finite and strictly increasing, got [{x_min}, {x_max}]"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "manufactured transport evaluation cannot be represented in binary64"
            ),
        }
    }
}

impl std::error::Error for TransportManufacturedError {}

/// Exact state and operator terms at one transport evaluation point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransportPoint1d {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// Exact scalar state `u`.
    pub value: f64,
    /// Exact spatial derivative `∂u/∂x`.
    pub spatial_gradient: f64,
    /// Exact Laplacian `∂²u/∂x²`.
    pub laplacian: f64,
    /// Exact manufactured source `s`.
    pub source: f64,
}

impl TransportPoint1d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.value,
            self.spatial_gradient,
            self.laplacian,
            self.source,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// Uniform exact samples and operator terms for a transport reference.
#[derive(Debug, Clone, PartialEq)]
pub struct TransportSample1d {
    /// Sample coordinates in metres.
    pub positions_m: Vec<f64>,
    /// Exact scalar state `u`.
    pub values: Vec<f64>,
    /// Exact spatial derivative `∂u/∂x`.
    pub spatial_gradients: Vec<f64>,
    /// Exact Laplacian `∂²u/∂x²`.
    pub laplacians: Vec<f64>,
    /// Exact manufactured source `s`.
    pub sources: Vec<f64>,
}

impl TransportSample1d {
    /// Verify a finite, consistently-sized exact field set.
    pub fn validate(&self) -> Result<(), TransportManufacturedError> {
        let count = self.positions_m.len();
        if count < 2 {
            return Err(TransportManufacturedError::InvalidParameter {
                field: "positions_m",
                requirement: "at least 2 samples",
            });
        }
        if [
            self.values.len(),
            self.spatial_gradients.len(),
            self.laplacians.len(),
            self.sources.len(),
        ]
        .iter()
        .any(|&length| length != count)
        {
            return Err(TransportManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the same length as positions_m",
            });
        }
        if self
            .positions_m
            .iter()
            .chain(self.values.iter())
            .chain(self.spatial_gradients.iter())
            .chain(self.laplacians.iter())
            .chain(self.sources.iter())
            .any(|value| !value.is_finite())
        {
            return Err(TransportManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A smooth manufactured solution for steady scalar transport.
///
/// The state is `u(x) = A sin(q(x-x₀))`; the matching source is
/// `s = (κq² + σ)u + v∂u/∂x`.  This makes diffusion, advection, and reaction
/// contributions individually available to a consuming discretization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedTransport1d {
    diffusivity_m2_per_s: f64,
    velocity_m_per_s: f64,
    reaction_rate_per_s: f64,
    spatial_wavenumber_per_m: f64,
    amplitude: f64,
    origin_m: f64,
}

impl ManufacturedTransport1d {
    /// Construct a checked constant-coefficient transport reference.
    pub fn try_new(
        diffusivity_m2_per_s: f64,
        velocity_m_per_s: f64,
        reaction_rate_per_s: f64,
        spatial_wavenumber_per_m: f64,
        amplitude: f64,
        origin_m: f64,
    ) -> Result<Self, TransportManufacturedError> {
        for (field, value, requirement) in [
            (
                "diffusivity_m2_per_s",
                diffusivity_m2_per_s,
                "finite and positive",
            ),
            ("velocity_m_per_s", velocity_m_per_s, "finite"),
            (
                "reaction_rate_per_s",
                reaction_rate_per_s,
                "finite and non-negative",
            ),
            (
                "spatial_wavenumber_per_m",
                spatial_wavenumber_per_m,
                "finite and positive",
            ),
            ("amplitude", amplitude, "finite"),
            ("origin_m", origin_m, "finite"),
        ] {
            let valid = match field {
                "diffusivity_m2_per_s" | "spatial_wavenumber_per_m" => {
                    value.is_finite() && value > 0.0
                }
                "reaction_rate_per_s" => value.is_finite() && value >= 0.0,
                _ => value.is_finite(),
            };
            if !valid {
                return Err(TransportManufacturedError::InvalidParameter { field, requirement });
            }
        }
        let diffusive_reactive_factor =
            diffusivity_m2_per_s * spatial_wavenumber_per_m.powi(2) + reaction_rate_per_s;
        let advective_factor = velocity_m_per_s * spatial_wavenumber_per_m;
        if !diffusive_reactive_factor.is_finite() || !advective_factor.is_finite() {
            return Err(TransportManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            diffusivity_m2_per_s,
            velocity_m_per_s,
            reaction_rate_per_s,
            spatial_wavenumber_per_m,
            amplitude,
            origin_m,
        })
    }

    /// Diffusivity `κ` in m²/s.
    pub fn diffusivity_m2_per_s(self) -> f64 {
        self.diffusivity_m2_per_s
    }

    /// Signed advective velocity `v` in m/s.
    pub fn velocity_m_per_s(self) -> f64 {
        self.velocity_m_per_s
    }

    /// Reaction/sink rate `σ` in 1/s.
    pub fn reaction_rate_per_s(self) -> f64 {
        self.reaction_rate_per_s
    }

    /// Spatial wavenumber `q` in 1/m.
    pub fn spatial_wavenumber_per_m(self) -> f64 {
        self.spatial_wavenumber_per_m
    }

    /// Evaluate the exact state and transport source at `x`.
    pub fn try_evaluate(self, x_m: f64) -> Result<TransportPoint1d, TransportManufacturedError> {
        if !x_m.is_finite() {
            return Err(TransportManufacturedError::InvalidParameter {
                field: "x_m",
                requirement: "finite",
            });
        }
        let phase = self.spatial_wavenumber_per_m * (x_m - self.origin_m);
        let diffusive_reactive_factor = self.diffusivity_m2_per_s
            * self.spatial_wavenumber_per_m.powi(2)
            + self.reaction_rate_per_s;
        if !phase.is_finite() || !diffusive_reactive_factor.is_finite() {
            return Err(TransportManufacturedError::EvaluationOverflow);
        }
        let value = self.amplitude * phase.sin();
        let spatial_gradient = self.amplitude * self.spatial_wavenumber_per_m * phase.cos();
        let laplacian = -self.spatial_wavenumber_per_m.powi(2) * value;
        let point = TransportPoint1d {
            x_m,
            value,
            spatial_gradient,
            laplacian,
            source: diffusive_reactive_factor * value + self.velocity_m_per_s * spatial_gradient,
        };
        if !point.all_finite() {
            return Err(TransportManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample the reference and all exact terms on a uniform spatial grid.
    pub fn try_sample(
        self,
        x_min_m: f64,
        x_max_m: f64,
        num_points: usize,
    ) -> Result<TransportSample1d, TransportManufacturedError> {
        if num_points < 2 {
            return Err(TransportManufacturedError::InvalidParameter {
                field: "num_points",
                requirement: "at least 2",
            });
        }
        if !x_min_m.is_finite() || !x_max_m.is_finite() || x_max_m <= x_min_m {
            return Err(TransportManufacturedError::InvalidInterval {
                x_min: x_min_m,
                x_max: x_max_m,
            });
        }
        let dx = (x_max_m - x_min_m) / (num_points - 1) as f64;
        if !dx.is_finite() {
            return Err(TransportManufacturedError::EvaluationOverflow);
        }
        let mut sample = TransportSample1d {
            positions_m: Vec::with_capacity(num_points),
            values: Vec::with_capacity(num_points),
            spatial_gradients: Vec::with_capacity(num_points),
            laplacians: Vec::with_capacity(num_points),
            sources: Vec::with_capacity(num_points),
        };
        for index in 0..num_points {
            let point = self.try_evaluate(x_min_m + index as f64 * dx)?;
            sample.positions_m.push(point.x_m);
            sample.values.push(point.value);
            sample.spatial_gradients.push(point.spatial_gradient);
            sample.laplacians.push(point.laplacian);
            sample.sources.push(point.source);
        }
        sample.validate()?;
        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn reference() -> ManufacturedTransport1d {
        ManufacturedTransport1d::try_new(0.7, -1.1, 0.4, 1.3, 2.0, -0.2).unwrap()
    }

    #[test]
    fn exact_terms_close_the_transport_equation() {
        let reference = reference();
        let sample = reference.try_sample(-1.0, 1.0, 17).unwrap();
        for index in 0..sample.positions_m.len() {
            assert_abs_diff_eq!(
                -reference.diffusivity_m2_per_s() * sample.laplacians[index]
                    + reference.velocity_m_per_s() * sample.spatial_gradients[index]
                    + reference.reaction_rate_per_s() * sample.values[index],
                sample.sources[index],
                epsilon = 1e-13
            );
        }
    }

    #[test]
    fn dirichlet_endpoints_remain_exact_with_advection_and_reaction() {
        let length = 2.0;
        let reference =
            ManufacturedTransport1d::try_new(0.5, 1.2, 0.8, PI / length, 1.5, 0.0).unwrap();
        let sample = reference.try_sample(0.0, length, 9).unwrap();
        assert_abs_diff_eq!(sample.values[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(*sample.values.last().unwrap(), 0.0, epsilon = 1e-14);
        assert!(sample.sources.iter().any(|source| source.abs() > 1e-6));
    }

    #[test]
    fn derivatives_agree_with_centered_finite_differences() {
        let reference = reference();
        let x = 0.45;
        let h = 1e-5;
        let point = reference.try_evaluate(x).unwrap();
        let left = reference.try_evaluate(x - h).unwrap().value;
        let right = reference.try_evaluate(x + h).unwrap().value;
        assert_abs_diff_eq!(
            (right - left) / (2.0 * h),
            point.spatial_gradient,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            (right - 2.0 * point.value + left) / h.powi(2),
            point.laplacian,
            epsilon = 3e-6
        );
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_samples() {
        assert!(matches!(
            ManufacturedTransport1d::try_new(0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
            Err(TransportManufacturedError::InvalidParameter {
                field: "diffusivity_m2_per_s",
                ..
            })
        ));
        assert!(matches!(
            ManufacturedTransport1d::try_new(1.0, 0.0, -0.1, 1.0, 1.0, 0.0),
            Err(TransportManufacturedError::InvalidParameter {
                field: "reaction_rate_per_s",
                ..
            })
        ));
        let reference = reference();
        assert!(matches!(
            reference.try_evaluate(f64::NAN),
            Err(TransportManufacturedError::InvalidParameter { field: "x_m", .. })
        ));
        assert!(matches!(
            reference.try_sample(1.0, 0.0, 4),
            Err(TransportManufacturedError::InvalidInterval { .. })
        ));
        assert!(matches!(
            reference.try_sample(0.0, 1.0, 1),
            Err(TransportManufacturedError::InvalidParameter {
                field: "num_points",
                ..
            })
        ));
    }
}
