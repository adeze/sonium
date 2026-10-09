//! Manufactured two-dimensional Helmholtz references.
//!
//! The exact complex-pressure field satisfies
//! `∂²p/∂x² + ∂²p/∂y² + k²p = f` on a Cartesian domain.  It is an
//! independent reference for 2-D Helmholtz discretizations, not a numerical
//! solve or a substitute for a mesh-convergence gate.

use super::manufactured_helmholtz::HelmholtzManufacturedError;
use num_complex::Complex64;

/// Exact pressure and operator terms at one two-dimensional Helmholtz point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzPoint2d {
    /// Cartesian x coordinate in metres.
    pub x_m: f64,
    /// Cartesian y coordinate in metres.
    pub y_m: f64,
    /// Exact complex pressure amplitude in Pa.
    pub pressure_pa: Complex64,
    /// Exact Cartesian pressure gradient `[∂p/∂x, ∂p/∂y]` in Pa/m.
    pub pressure_gradient_pa_per_m: [Complex64; 2],
    /// Exact Laplacian in Pa/m².
    pub laplacian_pa_per_m2: Complex64,
    /// Manufactured source in Pa/m².
    pub source_pa_per_m2: Complex64,
}

impl HelmholtzPoint2d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.y_m,
            self.pressure_pa.re,
            self.pressure_pa.im,
            self.pressure_gradient_pa_per_m[0].re,
            self.pressure_gradient_pa_per_m[0].im,
            self.pressure_gradient_pa_per_m[1].re,
            self.pressure_gradient_pa_per_m[1].im,
            self.laplacian_pa_per_m2.re,
            self.laplacian_pa_per_m2.im,
            self.source_pa_per_m2.re,
            self.source_pa_per_m2.im,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// Exact values sampled on a row-major Cartesian grid (`y` rows, `x` columns).
#[derive(Debug, Clone, PartialEq)]
pub struct HelmholtzSample2d {
    /// Strictly increasing x coordinates in metres.
    pub x_positions_m: Vec<f64>,
    /// Strictly increasing y coordinates in metres.
    pub y_positions_m: Vec<f64>,
    /// Exact complex pressures, indexed as `y_index * x_count + x_index`.
    pub pressures_pa: Vec<Complex64>,
    /// Exact Cartesian gradients in the same row-major order.
    pub pressure_gradients_pa_per_m: Vec<[Complex64; 2]>,
    /// Exact Laplacians in the same row-major order.
    pub laplacians_pa_per_m2: Vec<Complex64>,
    /// Exact sources in the same row-major order.
    pub sources_pa_per_m2: Vec<Complex64>,
}

impl HelmholtzSample2d {
    /// Validate grid monotonicity, cardinality, and finite exact fields.
    pub fn validate(&self) -> Result<(), HelmholtzManufacturedError> {
        if self.x_positions_m.len() < 2 || self.y_positions_m.len() < 2 {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample axes",
                requirement: "at least 2 samples on each axis",
            });
        }
        if self
            .x_positions_m
            .iter()
            .chain(self.y_positions_m.iter())
            .any(|value| !value.is_finite())
            || self.x_positions_m.windows(2).any(|pair| pair[1] <= pair[0])
            || self.y_positions_m.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample axes",
                requirement: "finite and strictly increasing",
            });
        }
        let expected = self
            .x_positions_m
            .len()
            .checked_mul(self.y_positions_m.len())
            .ok_or(HelmholtzManufacturedError::EvaluationOverflow)?;
        if [
            self.pressures_pa.len(),
            self.pressure_gradients_pa_per_m.len(),
            self.laplacians_pa_per_m2.len(),
            self.sources_pa_per_m2.len(),
        ]
        .iter()
        .any(|&length| length != expected)
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the Cartesian grid cardinality",
            });
        }
        if self
            .pressures_pa
            .iter()
            .chain(self.laplacians_pa_per_m2.iter())
            .chain(self.sources_pa_per_m2.iter())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
            || self.pressure_gradients_pa_per_m.iter().any(|gradient| {
                gradient
                    .iter()
                    .any(|value| !value.re.is_finite() || !value.im.is_finite())
            })
        {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A separable complex-pressure manufactured solution of the 2-D Helmholtz equation.
///
/// `p(x,y) = P sin(q_x(x-x₀)) sin(q_y(y-y₀))`, with source
/// `f = (k²-q_x²-q_y²)p`.  Setting `k² = q_x²+q_y²` gives an unforced
/// homogeneous Dirichlet mode on a rectangular sine-mode domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedHelmholtz2d {
    equation_wavenumber_per_m: f64,
    spatial_wavenumbers_per_m: [f64; 2],
    amplitude_pa: Complex64,
    origin_m: [f64; 2],
}

impl ManufacturedHelmholtz2d {
    /// Construct a checked constant-coefficient two-dimensional reference.
    pub fn try_new(
        equation_wavenumber_per_m: f64,
        spatial_wavenumbers_per_m: [f64; 2],
        amplitude_pa: Complex64,
        origin_m: [f64; 2],
    ) -> Result<Self, HelmholtzManufacturedError> {
        for (field, value, positive) in [
            ("equation_wavenumber_per_m", equation_wavenumber_per_m, true),
            (
                "spatial_wavenumbers_per_m[0]",
                spatial_wavenumbers_per_m[0],
                true,
            ),
            (
                "spatial_wavenumbers_per_m[1]",
                spatial_wavenumbers_per_m[1],
                true,
            ),
            ("amplitude_pa.re", amplitude_pa.re, false),
            ("amplitude_pa.im", amplitude_pa.im, false),
            ("origin_m[0]", origin_m[0], false),
            ("origin_m[1]", origin_m[1], false),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) {
                return Err(HelmholtzManufacturedError::InvalidParameter {
                    field,
                    requirement: if positive {
                        "finite and positive"
                    } else {
                        "finite"
                    },
                });
            }
        }
        let source_factor = equation_wavenumber_per_m.powi(2)
            - spatial_wavenumbers_per_m[0].powi(2)
            - spatial_wavenumbers_per_m[1].powi(2);
        if !source_factor.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            equation_wavenumber_per_m,
            spatial_wavenumbers_per_m,
            amplitude_pa,
            origin_m,
        })
    }

    /// Helmholtz equation wavenumber `k` in rad/m.
    pub fn equation_wavenumber_per_m(self) -> f64 {
        self.equation_wavenumber_per_m
    }

    /// Evaluate exact pressure, gradient, Laplacian, and source at `(x, y)`.
    pub fn try_evaluate(
        self,
        x_m: f64,
        y_m: f64,
    ) -> Result<HelmholtzPoint2d, HelmholtzManufacturedError> {
        if !x_m.is_finite() || !y_m.is_finite() {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "coordinates_m",
                requirement: "finite",
            });
        }
        let phase_x = self.spatial_wavenumbers_per_m[0] * (x_m - self.origin_m[0]);
        let phase_y = self.spatial_wavenumbers_per_m[1] * (y_m - self.origin_m[1]);
        let wave_sq =
            self.spatial_wavenumbers_per_m[0].powi(2) + self.spatial_wavenumbers_per_m[1].powi(2);
        let source_factor = self.equation_wavenumber_per_m.powi(2) - wave_sq;
        if !phase_x.is_finite() || !phase_y.is_finite() || !source_factor.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        let sin_x = phase_x.sin();
        let sin_y = phase_y.sin();
        let pressure = self.amplitude_pa * (sin_x * sin_y);
        let gradient = [
            self.amplitude_pa * (self.spatial_wavenumbers_per_m[0] * phase_x.cos() * sin_y),
            self.amplitude_pa * (self.spatial_wavenumbers_per_m[1] * sin_x * phase_y.cos()),
        ];
        let laplacian = pressure * -wave_sq;
        let source = pressure * source_factor;
        let point = HelmholtzPoint2d {
            x_m,
            y_m,
            pressure_pa: pressure,
            pressure_gradient_pa_per_m: gradient,
            laplacian_pa_per_m2: laplacian,
            source_pa_per_m2: source,
        };
        if !point.all_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample the exact fields on a uniform Cartesian rectangle.
    pub fn try_sample(
        self,
        x_bounds_m: [f64; 2],
        x_count: usize,
        y_bounds_m: [f64; 2],
        y_count: usize,
    ) -> Result<HelmholtzSample2d, HelmholtzManufacturedError> {
        if x_count < 2 || y_count < 2 {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample counts",
                requirement: "at least 2 samples on each axis",
            });
        }
        if x_bounds_m.iter().any(|value| !value.is_finite())
            || y_bounds_m.iter().any(|value| !value.is_finite())
            || x_bounds_m[1] <= x_bounds_m[0]
            || y_bounds_m[1] <= y_bounds_m[0]
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample bounds",
                requirement: "finite and strictly increasing on both axes",
            });
        }
        let dx = (x_bounds_m[1] - x_bounds_m[0]) / (x_count - 1) as f64;
        let dy = (y_bounds_m[1] - y_bounds_m[0]) / (y_count - 1) as f64;
        let total = x_count
            .checked_mul(y_count)
            .ok_or(HelmholtzManufacturedError::EvaluationOverflow)?;
        if !dx.is_finite() || !dy.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        let mut sample = HelmholtzSample2d {
            x_positions_m: (0..x_count)
                .map(|index| x_bounds_m[0] + index as f64 * dx)
                .collect(),
            y_positions_m: (0..y_count)
                .map(|index| y_bounds_m[0] + index as f64 * dy)
                .collect(),
            pressures_pa: Vec::with_capacity(total),
            pressure_gradients_pa_per_m: Vec::with_capacity(total),
            laplacians_pa_per_m2: Vec::with_capacity(total),
            sources_pa_per_m2: Vec::with_capacity(total),
        };
        for &y_m in &sample.y_positions_m {
            for &x_m in &sample.x_positions_m {
                let point = self.try_evaluate(x_m, y_m)?;
                sample.pressures_pa.push(point.pressure_pa);
                sample
                    .pressure_gradients_pa_per_m
                    .push(point.pressure_gradient_pa_per_m);
                sample.laplacians_pa_per_m2.push(point.laplacian_pa_per_m2);
                sample.sources_pa_per_m2.push(point.source_pa_per_m2);
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
    fn exact_terms_close_the_2d_helmholtz_equation() {
        let reference = ManufacturedHelmholtz2d::try_new(
            3.1,
            [1.3, 0.9],
            Complex64::new(2.0, -0.7),
            [-0.2, 0.4],
        )
        .unwrap();
        let sample = reference
            .try_sample([-1.0, 1.0], 7, [-0.5, 1.5], 5)
            .unwrap();
        for index in 0..sample.pressures_pa.len() {
            let residual = sample.laplacians_pa_per_m2[index]
                + sample.pressures_pa[index] * reference.equation_wavenumber_per_m().powi(2)
                - sample.sources_pa_per_m2[index];
            assert_abs_diff_eq!(residual.norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn homogeneous_mode_has_zero_source_and_boundary_trace() {
        let length_x = 2.0;
        let length_y = 3.0;
        let qx = PI / length_x;
        let qy = PI / length_y;
        let reference = ManufacturedHelmholtz2d::try_new(
            (qx.powi(2) + qy.powi(2)).sqrt(),
            [qx, qy],
            Complex64::new(1.5, -0.4),
            [0.0, 0.0],
        )
        .unwrap();
        let sample = reference
            .try_sample([0.0, length_x], 5, [0.0, length_y], 7)
            .unwrap();
        assert!(
            sample
                .sources_pa_per_m2
                .iter()
                .all(|source| source.norm() < 1e-13)
        );
        for y_index in 0..sample.y_positions_m.len() {
            let first = y_index * sample.x_positions_m.len();
            let last = first + sample.x_positions_m.len() - 1;
            assert!(sample.pressures_pa[first].norm() < 1e-14);
            assert!(sample.pressures_pa[last].norm() < 1e-14);
        }
    }

    #[test]
    fn gradients_agree_with_centered_finite_differences() {
        let reference = ManufacturedHelmholtz2d::try_new(
            2.5,
            [1.7, 1.1],
            Complex64::new(-0.75, 0.4),
            [0.1, -0.3],
        )
        .unwrap();
        let (x, y, h) = (0.45, 0.2, 1e-5);
        let point = reference.try_evaluate(x, y).unwrap();
        let x_gradient = (reference.try_evaluate(x + h, y).unwrap().pressure_pa
            - reference.try_evaluate(x - h, y).unwrap().pressure_pa)
            / (2.0 * h);
        let y_gradient = (reference.try_evaluate(x, y + h).unwrap().pressure_pa
            - reference.try_evaluate(x, y - h).unwrap().pressure_pa)
            / (2.0 * h);
        assert_abs_diff_eq!(
            (x_gradient - point.pressure_gradient_pa_per_m[0]).norm(),
            0.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            (y_gradient - point.pressure_gradient_pa_per_m[1]).norm(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_grids() {
        assert!(ManufacturedHelmholtz2d::try_new(
            1.0,
            [0.0, 1.0],
            Complex64::new(1.0, 0.0),
            [0.0, 0.0],
        )
        .is_err());
        let reference =
            ManufacturedHelmholtz2d::try_new(1.0, [1.0, 1.0], Complex64::new(1.0, 0.0), [0.0, 0.0])
                .unwrap();
        assert!(reference.try_evaluate(f64::NAN, 0.0).is_err());
        assert!(reference.try_sample([1.0, 0.0], 4, [0.0, 1.0], 4).is_err());
        assert!(reference.try_sample([0.0, 1.0], 1, [0.0, 1.0], 4).is_err());
    }
}
