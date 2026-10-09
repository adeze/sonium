//! Manufactured two-dimensional scalar-diffusion references.
//!
//! The real-valued state satisfies `∂u/∂t - κ(∂²u/∂x² + ∂²u/∂y²) = s` on a
//! Cartesian domain. It is an exact term-level reference, not a solver.

use super::manufactured::DiffusionManufacturedError;

/// Exact state and derivative/source terms at a two-dimensional point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffusionPoint2d {
    /// Cartesian coordinates in metres.
    pub position_m: [f64; 2],
    /// Evaluation time in seconds.
    pub time_s: f64,
    /// Exact scalar state.
    pub value: f64,
    /// Exact Cartesian gradient.
    pub spatial_gradient: [f64; 2],
    /// Exact Laplacian.
    pub laplacian: f64,
    /// Exact time derivative.
    pub time_derivative: f64,
    /// Manufactured source.
    pub source: f64,
}

impl DiffusionPoint2d {
    fn all_finite(self) -> bool {
        self.position_m
            .into_iter()
            .chain([
                self.time_s,
                self.value,
                self.spatial_gradient[0],
                self.spatial_gradient[1],
                self.laplacian,
                self.time_derivative,
                self.source,
            ])
            .all(f64::is_finite)
    }
}

/// Exact values sampled on a row-major Cartesian grid (`y` rows, `x` columns).
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionSample2d {
    /// Evaluation time in seconds.
    pub time_s: f64,
    /// Strictly increasing x positions in metres.
    pub x_positions_m: Vec<f64>,
    /// Strictly increasing y positions in metres.
    pub y_positions_m: Vec<f64>,
    /// Exact state, indexed as `y_index * x_count + x_index`.
    pub values: Vec<f64>,
    /// Exact Cartesian gradients in matching row-major order.
    pub spatial_gradients: Vec<[f64; 2]>,
    /// Exact Laplacians in matching order.
    pub laplacians: Vec<f64>,
    /// Exact time derivatives in matching order.
    pub time_derivatives: Vec<f64>,
    /// Exact manufactured sources in matching order.
    pub sources: Vec<f64>,
}

impl DiffusionSample2d {
    /// Validate time, axes, cardinality, and finite exact fields.
    pub fn validate(&self) -> Result<(), DiffusionManufacturedError> {
        if !self.time_s.is_finite() || self.time_s < 0.0 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "time_s",
                requirement: "finite and non-negative",
            });
        }
        let axes = [&self.x_positions_m, &self.y_positions_m];
        if axes.iter().any(|axis| axis.len() < 2) {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "sample axes",
                requirement: "at least 2 samples on each axis",
            });
        }
        if axes.iter().any(|axis| {
            axis.iter().any(|value| !value.is_finite())
                || axis.windows(2).any(|pair| pair[1] <= pair[0])
        }) {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "sample axes",
                requirement: "finite and strictly increasing",
            });
        }
        let expected = self
            .x_positions_m
            .len()
            .checked_mul(self.y_positions_m.len())
            .ok_or(DiffusionManufacturedError::EvaluationOverflow)?;
        if [
            self.values.len(),
            self.spatial_gradients.len(),
            self.laplacians.len(),
            self.time_derivatives.len(),
            self.sources.len(),
        ]
        .into_iter()
        .any(|length| length != expected)
        {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the Cartesian grid cardinality",
            });
        }
        if self
            .values
            .iter()
            .chain(&self.laplacians)
            .chain(&self.time_derivatives)
            .chain(&self.sources)
            .chain(self.spatial_gradients.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A separable sine manufactured solution of the 2-D diffusion equation.
///
/// `u = A exp(-λt) sin(q_x(x-x₀)) sin(q_y(y-y₀))`, with
/// `s = (κ(q_x²+q_y²)-λ)u`. Setting `λ = κ(q_x²+q_y²)` gives an unforced
/// homogeneous Dirichlet rectangle mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedDiffusion2d {
    diffusivity_m2_per_s: f64,
    spatial_wavenumbers_per_m: [f64; 2],
    decay_rate_per_s: f64,
    amplitude: f64,
    origin_m: [f64; 2],
}

impl ManufacturedDiffusion2d {
    /// Construct a checked constant-coefficient two-dimensional reference.
    pub fn try_new(
        diffusivity_m2_per_s: f64,
        spatial_wavenumbers_per_m: [f64; 2],
        decay_rate_per_s: f64,
        amplitude: f64,
        origin_m: [f64; 2],
    ) -> Result<Self, DiffusionManufacturedError> {
        if !diffusivity_m2_per_s.is_finite() || diffusivity_m2_per_s <= 0.0 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "diffusivity_m2_per_s",
                requirement: "finite and positive",
            });
        }
        if spatial_wavenumbers_per_m
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "spatial_wavenumbers_per_m",
                requirement: "all finite and positive",
            });
        }
        if !decay_rate_per_s.is_finite() || decay_rate_per_s < 0.0 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "decay_rate_per_s",
                requirement: "finite and non-negative",
            });
        }
        if !amplitude.is_finite() || origin_m.iter().any(|value| !value.is_finite()) {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "amplitude/origin_m",
                requirement: "finite",
            });
        }
        let wave_sq = spatial_wavenumbers_per_m
            .into_iter()
            .map(|value| value.powi(2))
            .sum::<f64>();
        if !(diffusivity_m2_per_s * wave_sq - decay_rate_per_s).is_finite() {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            diffusivity_m2_per_s,
            spatial_wavenumbers_per_m,
            decay_rate_per_s,
            amplitude,
            origin_m,
        })
    }

    /// Diffusivity `κ` in m²/s.
    pub fn diffusivity_m2_per_s(self) -> f64 {
        self.diffusivity_m2_per_s
    }

    /// Evaluate exact state, gradient, Laplacian, time derivative, and source.
    pub fn try_evaluate(
        self,
        position_m: [f64; 2],
        time_s: f64,
    ) -> Result<DiffusionPoint2d, DiffusionManufacturedError> {
        if position_m.iter().any(|value| !value.is_finite()) {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "position_m",
                requirement: "finite",
            });
        }
        if !time_s.is_finite() || time_s < 0.0 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "time_s",
                requirement: "finite and non-negative",
            });
        }
        let phase = [
            self.spatial_wavenumbers_per_m[0] * (position_m[0] - self.origin_m[0]),
            self.spatial_wavenumbers_per_m[1] * (position_m[1] - self.origin_m[1]),
        ];
        let exponent = -self.decay_rate_per_s * time_s;
        let wave_sq = self
            .spatial_wavenumbers_per_m
            .into_iter()
            .map(|value| value.powi(2))
            .sum::<f64>();
        let source_factor = self.diffusivity_m2_per_s * wave_sq - self.decay_rate_per_s;
        if phase.iter().any(|value| !value.is_finite())
            || !exponent.is_finite()
            || !source_factor.is_finite()
        {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        let scale = self.amplitude * exponent.exp();
        let sine = [phase[0].sin(), phase[1].sin()];
        let value = scale * sine[0] * sine[1];
        let point = DiffusionPoint2d {
            position_m,
            time_s,
            value,
            spatial_gradient: [
                scale * self.spatial_wavenumbers_per_m[0] * phase[0].cos() * sine[1],
                scale * self.spatial_wavenumbers_per_m[1] * sine[0] * phase[1].cos(),
            ],
            laplacian: -wave_sq * value,
            time_derivative: -self.decay_rate_per_s * value,
            source: source_factor * value,
        };
        if !point.all_finite() {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample the exact fields on a uniform Cartesian rectangle at one time.
    pub fn try_sample(
        self,
        x_bounds_m: [f64; 2],
        x_count: usize,
        y_bounds_m: [f64; 2],
        y_count: usize,
        time_s: f64,
    ) -> Result<DiffusionSample2d, DiffusionManufacturedError> {
        if x_count < 2 || y_count < 2 {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "sample counts",
                requirement: "at least 2 samples on each axis",
            });
        }
        if [x_bounds_m, y_bounds_m]
            .iter()
            .any(|bounds| bounds.iter().any(|value| !value.is_finite()) || bounds[1] <= bounds[0])
        {
            return Err(DiffusionManufacturedError::InvalidParameter {
                field: "sample bounds",
                requirement: "finite and strictly increasing on both axes",
            });
        }
        let dx = (x_bounds_m[1] - x_bounds_m[0]) / (x_count - 1) as f64;
        let dy = (y_bounds_m[1] - y_bounds_m[0]) / (y_count - 1) as f64;
        let total = x_count
            .checked_mul(y_count)
            .ok_or(DiffusionManufacturedError::EvaluationOverflow)?;
        if !dx.is_finite() || !dy.is_finite() {
            return Err(DiffusionManufacturedError::EvaluationOverflow);
        }
        let mut sample = DiffusionSample2d {
            time_s,
            x_positions_m: (0..x_count)
                .map(|index| x_bounds_m[0] + index as f64 * dx)
                .collect(),
            y_positions_m: (0..y_count)
                .map(|index| y_bounds_m[0] + index as f64 * dy)
                .collect(),
            values: Vec::with_capacity(total),
            spatial_gradients: Vec::with_capacity(total),
            laplacians: Vec::with_capacity(total),
            time_derivatives: Vec::with_capacity(total),
            sources: Vec::with_capacity(total),
        };
        for &y_m in &sample.y_positions_m {
            for &x_m in &sample.x_positions_m {
                let point = self.try_evaluate([x_m, y_m], time_s)?;
                sample.values.push(point.value);
                sample.spatial_gradients.push(point.spatial_gradient);
                sample.laplacians.push(point.laplacian);
                sample.time_derivatives.push(point.time_derivative);
                sample.sources.push(point.source);
            }
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
        let reference =
            ManufacturedDiffusion2d::try_new(0.7, [1.3, 0.9], 0.4, 2.0, [-0.2, 0.4]).unwrap();
        let sample = reference
            .try_sample([-1.0, 1.0], 7, [-0.5, 1.5], 5, 0.8)
            .unwrap();
        for ((time_derivative, laplacian), source) in sample
            .time_derivatives
            .iter()
            .zip(&sample.laplacians)
            .zip(&sample.sources)
        {
            assert_abs_diff_eq!(
                time_derivative - reference.diffusivity_m2_per_s() * laplacian,
                *source,
                epsilon = 1e-13
            );
        }
    }

    #[test]
    fn homogeneous_mode_has_zero_source_and_rectangle_boundary() {
        let lengths = [2.0, 3.0];
        let q = lengths.map(|length| PI / length);
        let diffusivity = 0.5;
        let reference = ManufacturedDiffusion2d::try_new(
            diffusivity,
            q,
            diffusivity * q.into_iter().map(|value| value.powi(2)).sum::<f64>(),
            1.5,
            [0.0; 2],
        )
        .unwrap();
        let sample = reference
            .try_sample([0.0, lengths[0]], 5, [0.0, lengths[1]], 7, 0.3)
            .unwrap();
        assert!(sample.sources.iter().all(|source| source.abs() < 1e-13));
        let nx = sample.x_positions_m.len();
        for (index, value) in sample.values.iter().enumerate() {
            let x = index % nx;
            let y = index / nx;
            if x == 0 || x + 1 == nx || y == 0 || y + 1 == sample.y_positions_m.len() {
                assert!(value.abs() < 1e-13);
            }
        }
    }

    #[test]
    fn derivatives_agree_with_centered_finite_differences() {
        let reference =
            ManufacturedDiffusion2d::try_new(0.25, [2.0, 1.4], 0.8, -0.75, [0.1, -0.2]).unwrap();
        let position = [0.45, 0.2];
        let time = 0.6;
        let h = 1e-5;
        let point = reference.try_evaluate(position, time).unwrap();
        for axis in 0..2 {
            let mut plus = position;
            let mut minus = position;
            plus[axis] += h;
            minus[axis] -= h;
            assert_abs_diff_eq!(
                (reference.try_evaluate(plus, time).unwrap().value
                    - reference.try_evaluate(minus, time).unwrap().value)
                    / (2.0 * h),
                point.spatial_gradient[axis],
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_grids() {
        assert!(ManufacturedDiffusion2d::try_new(0.0, [1.0, 1.0], 0.0, 1.0, [0.0; 2]).is_err());
        let reference =
            ManufacturedDiffusion2d::try_new(1.0, [1.0, 1.0], 0.0, 1.0, [0.0; 2]).unwrap();
        assert!(reference.try_evaluate([f64::NAN, 0.0], 0.0).is_err());
        assert!(
            reference
                .try_sample([1.0, 0.0], 4, [0.0, 1.0], 4, 0.0)
                .is_err()
        );
        assert!(
            reference
                .try_sample([0.0, 1.0], 1, [0.0, 1.0], 4, 0.0)
                .is_err()
        );
    }
}
