//! Manufactured scalar diffusion references.
//!
//! This module supplies an independently evaluable one-dimensional reference
//! for the constant-coefficient diffusion equation
//!
//! ```text
//! ∂u/∂t - κ ∂²u/∂x² = s(x, t),
//! ```
//!
//! where `κ` is a positive scalar diffusivity in m²/s.  The reference family
//! is useful for heat, concentration, and any scalar diffusion implementation
//! with the same sign convention.  It is deliberately separate from the
//! acoustic phasor references: `u` and `s` are real-valued time-domain
//! quantities, not pressure phasors.

use std::fmt;

/// Invalid input to a manufactured one-dimensional diffusion reference.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffusionManufacturedError {
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
    /// An evaluated phase, decay, or forcing coefficient is not representable.
    EvaluationOverflow,
}

impl fmt::Display for DiffusionManufacturedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(
                    f,
                    "manufactured diffusion parameter {field} must be {requirement}"
                )
            }
            Self::InvalidInterval { x_min, x_max } => write!(
                f,
                "manufactured diffusion interval must be finite and strictly increasing, got [{x_min}, {x_max}]"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "manufactured diffusion evaluation cannot be represented in binary64"
            ),
        }
    }
}

impl std::error::Error for DiffusionManufacturedError {}

/// Samples and exact derivative/source terms for a diffusion reference.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionSample1d {
    /// Evaluation time in seconds.
    pub time_s: f64,
    /// Sample coordinates in metres.
    pub positions_m: Vec<f64>,
    /// Exact scalar state `u`.
    pub values: Vec<f64>,
    /// Exact spatial derivative `∂u/∂x` in inverse metres times `u` units.
    pub spatial_gradients: Vec<f64>,
    /// Exact Laplacian `∂²u/∂x²`.
    pub laplacians: Vec<f64>,
    /// Exact time derivative `∂u/∂t`.
    pub time_derivatives: Vec<f64>,
    /// Exact manufactured forcing `s = ∂u/∂t - κ∂²u/∂x²`.
    pub sources: Vec<f64>,
}

/// Exact state and derivative/source terms at one diffusion evaluation point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffusionPoint1d {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// Evaluation time in seconds.
    pub time_s: f64,
    /// Exact scalar state `u`.
    pub value: f64,
    /// Exact spatial derivative `∂u/∂x`.
    pub spatial_gradient: f64,
    /// Exact Laplacian `∂²u/∂x²`.
    pub laplacian: f64,
    /// Exact time derivative `∂u/∂t`.
    pub time_derivative: f64,
    /// Exact manufactured forcing `s = ∂u/∂t - κ∂²u/∂x²`.
    pub source: f64,
}

impl DiffusionPoint1d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.time_s,
            self.value,
            self.spatial_gradient,
            self.laplacian,
            self.time_derivative,
            self.source,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

impl DiffusionSample1d {
    /// Verify that every sample has a finite, consistently sized field set.
    pub fn validate(&self) -> Result<(), DiffusionManufacturedError> {
        if !self.time_s.is_finite() || self.time_s < 0.0 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "time_s",
                requirement: "finite and non-negative",
            });
        }
        let count = self.positions_m.len();
        if count < 2 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "positions_m",
                requirement: "at least 2 samples",
            });
        }
        if [
            self.values.len(),
            self.spatial_gradients.len(),
            self.laplacians.len(),
            self.time_derivatives.len(),
            self.sources.len(),
        ]
        .iter()
        .any(|&length| length != count)
        {
            return Err(DiffusionManufacturedError::InvalidParameter {
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
            .chain(self.time_derivatives.iter())
            .chain(self.sources.iter())
            .any(|value| !value.is_finite())
        {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A smooth manufactured solution of the scalar diffusion equation.
///
/// The state is
/// `u(x,t) = A exp(-λt) sin(q(x-x₀))`, where `A` is the scalar amplitude,
/// `λ` the non-negative temporal decay rate (1/s), and `q` the positive
/// spatial wavenumber (1/m). The matching source is
/// `s = (κq² - λ)u`. Setting `λ = κq²` gives the unforced homogeneous
/// diffusion mode; other finite `λ` values make the forcing explicit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedDiffusion1d {
    diffusivity_m2_per_s: f64,
    spatial_wavenumber_per_m: f64,
    decay_rate_per_s: f64,
    amplitude: f64,
    origin_m: f64,
}

impl ManufacturedDiffusion1d {
    /// Construct a checked constant-coefficient diffusion reference.
    pub fn try_new(
        diffusivity_m2_per_s: f64,
        spatial_wavenumber_per_m: f64,
        decay_rate_per_s: f64,
        amplitude: f64,
        origin_m: f64,
    ) -> Result<Self, DiffusionManufacturedError> {
        for (field, value, requirement) in [
            (
                "diffusivity_m2_per_s",
                diffusivity_m2_per_s,
                "finite and positive",
            ),
            (
                "spatial_wavenumber_per_m",
                spatial_wavenumber_per_m,
                "finite and positive",
            ),
            (
                "decay_rate_per_s",
                decay_rate_per_s,
                "finite and non-negative",
            ),
            ("amplitude", amplitude, "finite"),
            ("origin_m", origin_m, "finite"),
        ] {
            let valid = match field {
                "diffusivity_m2_per_s" | "spatial_wavenumber_per_m" => {
                    value.is_finite() && value > 0.0
                }
                "decay_rate_per_s" => value.is_finite() && value >= 0.0,
                _ => value.is_finite(),
            };
            if !valid {
                return Err(DiffusionManufacturedError::InvalidParameter { field, requirement });
            }
        }
        let coefficient =
            diffusivity_m2_per_s * spatial_wavenumber_per_m.powi(2) - decay_rate_per_s;
        if !coefficient.is_finite() {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            diffusivity_m2_per_s,
            spatial_wavenumber_per_m,
            decay_rate_per_s,
            amplitude,
            origin_m,
        })
    }

    /// Diffusivity `κ` in m²/s.
    pub fn diffusivity_m2_per_s(self) -> f64 {
        self.diffusivity_m2_per_s
    }

    /// Spatial wavenumber `q` in 1/m.
    pub fn spatial_wavenumber_per_m(self) -> f64 {
        self.spatial_wavenumber_per_m
    }

    /// Evaluate the exact state and derivative/source terms at `(x, t)`.
    pub fn try_evaluate(
        self,
        x_m: f64,
        time_s: f64,
    ) -> Result<DiffusionPoint1d, DiffusionManufacturedError> {
        if !x_m.is_finite() {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "x_m",
                requirement: "finite",
            });
        }
        if !time_s.is_finite() || time_s < 0.0 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "time_s",
                requirement: "finite and non-negative",
            });
        }
        let phase = self.spatial_wavenumber_per_m * (x_m - self.origin_m);
        let exponent = -self.decay_rate_per_s * time_s;
        let forcing_factor = self.diffusivity_m2_per_s * self.spatial_wavenumber_per_m.powi(2)
            - self.decay_rate_per_s;
        if !phase.is_finite() || !exponent.is_finite() || !forcing_factor.is_finite() {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        let scale = self.amplitude * exponent.exp();
        let value = scale * phase.sin();
        let gradient = scale * self.spatial_wavenumber_per_m * phase.cos();
        let laplacian = -self.spatial_wavenumber_per_m.powi(2) * value;
        let time_derivative = -self.decay_rate_per_s * value;
        let source = forcing_factor * value;
        let point = DiffusionPoint1d {
            x_m,
            time_s,
            value,
            spatial_gradient: gradient,
            laplacian,
            time_derivative,
            source,
        };
        if !point.all_finite() {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample the reference and all exact terms on a uniform spatial grid.
    pub fn try_sample(
        self,
        x_min_m: f64,
        x_max_m: f64,
        num_points: usize,
        time_s: f64,
    ) -> Result<DiffusionSample1d, DiffusionManufacturedError> {
        if num_points < 2 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "num_points",
                requirement: "at least 2",
            });
        }
        if !x_min_m.is_finite() || !x_max_m.is_finite() || x_max_m <= x_min_m {
            return Err(DiffusionManufacturedError::InvalidInterval {
                x_min: x_min_m,
                x_max: x_max_m,
            });
        }
        let dx = (x_max_m - x_min_m) / (num_points - 1) as f64;
        if !dx.is_finite() {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        let mut sample = DiffusionSample1d {
            time_s,
            positions_m: Vec::with_capacity(num_points),
            values: Vec::with_capacity(num_points),
            spatial_gradients: Vec::with_capacity(num_points),
            laplacians: Vec::with_capacity(num_points),
            time_derivatives: Vec::with_capacity(num_points),
            sources: Vec::with_capacity(num_points),
        };
        for index in 0..num_points {
            let x_m = x_min_m + index as f64 * dx;
            let point = self.try_evaluate(x_m, time_s)?;
            sample.positions_m.push(x_m);
            sample.values.push(point.value);
            sample.spatial_gradients.push(point.spatial_gradient);
            sample.laplacians.push(point.laplacian);
            sample.time_derivatives.push(point.time_derivative);
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

    #[test]
    fn exact_terms_close_the_diffusion_equation() {
        let reference = ManufacturedDiffusion1d::try_new(0.7, 1.3, 0.4, 2.0, -0.2).unwrap();
        let sample = reference.try_sample(-1.0, 1.0, 17, 0.8).unwrap();

        for index in 0..sample.positions_m.len() {
            assert_abs_diff_eq!(
                sample.time_derivatives[index]
                    - reference.diffusivity_m2_per_s() * sample.laplacians[index],
                sample.sources[index],
                epsilon = 1e-13
            );
        }
    }

    #[test]
    fn homogeneous_mode_has_zero_source_and_dirichlet_endpoints() {
        let length = 2.0;
        let wavenumber = PI / length;
        let diffusivity = 0.5;
        let reference = ManufacturedDiffusion1d::try_new(
            diffusivity,
            wavenumber,
            diffusivity * wavenumber.powi(2),
            1.5,
            0.0,
        )
        .unwrap();
        let sample = reference.try_sample(0.0, length, 9, 0.3).unwrap();

        assert_abs_diff_eq!(sample.values[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(*sample.values.last().unwrap(), 0.0, epsilon = 1e-14);
        for source in sample.sources {
            assert_abs_diff_eq!(source, 0.0, epsilon = 1e-13);
        }
    }

    #[test]
    fn derivatives_agree_with_centered_finite_differences() {
        let reference = ManufacturedDiffusion1d::try_new(0.25, 2.0, 0.8, -0.75, 0.1).unwrap();
        let x = 0.45;
        let time = 0.6;
        let h = 1e-5;
        let point = reference.try_evaluate(x, time).unwrap();
        let left = reference.try_evaluate(x - h, time).unwrap().value;
        let right = reference.try_evaluate(x + h, time).unwrap().value;
        assert_abs_diff_eq!(
            (right - left) / (2.0 * h),
            point.spatial_gradient,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            (right - 2.0 * point.value + left) / h.powi(2),
            point.laplacian,
            epsilon = 2e-6
        );
        // Differentiate the state itself in time, rather than comparing the
        // already exact derivative values above.
        let state_before = reference.try_evaluate(x, time - h).unwrap().value;
        let state_after = reference.try_evaluate(x, time + h).unwrap().value;
        assert_abs_diff_eq!(
            (state_after - state_before) / (2.0 * h),
            point.time_derivative,
            epsilon = 1e-9
        );
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_evaluations() {
        assert!(matches!(
            ManufacturedDiffusion1d::try_new(0.0, 1.0, 0.0, 1.0, 0.0),
            Err(DiffusionManufacturedError::InvalidParameter {
                field: "diffusivity_m2_per_s",
                ..
            })
        ));
        let reference = ManufacturedDiffusion1d::try_new(1.0, 1.0, 0.0, 1.0, 0.0).unwrap();
        assert!(matches!(
            reference.try_evaluate(f64::NAN, 0.0),
            Err(DiffusionManufacturedError::InvalidParameter { field: "x_m", .. })
        ));
        assert!(matches!(
            reference.try_sample(1.0, 0.0, 4, 0.0),
            Err(DiffusionManufacturedError::InvalidInterval { .. })
        ));
        assert!(matches!(
            reference.try_sample(0.0, 1.0, 1, 0.0),
            Err(DiffusionManufacturedError::InvalidParameter {
                field: "num_points",
                ..
            })
        ));

        let empty = DiffusionSample1d {
            time_s: 0.0,
            positions_m: Vec::new(),
            values: Vec::new(),
            spatial_gradients: Vec::new(),
            laplacians: Vec::new(),
            time_derivatives: Vec::new(),
            sources: Vec::new(),
        };
        assert!(matches!(
            empty.validate(),
            Err(DiffusionManufacturedError::InvalidParameter {
                field: "positions_m",
                ..
            })
        ));
    }
}
