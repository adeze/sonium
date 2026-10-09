//! Manufactured three-dimensional Helmholtz references.
//!
//! The exact complex-pressure field satisfies
//! `∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z² + k²p = f` on a Cartesian domain. It is a
//! reference for numerical solvers, not a numerical solve itself.

use super::manufactured_helmholtz::HelmholtzManufacturedError;
use num_complex::Complex64;

/// Exact pressure and Helmholtz operator terms at one Cartesian point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzPoint3d {
    /// Cartesian coordinates in metres.
    pub position_m: [f64; 3],
    /// Exact complex pressure amplitude in Pa.
    pub pressure_pa: Complex64,
    /// Exact Cartesian pressure gradient in Pa/m.
    pub pressure_gradient_pa_per_m: [Complex64; 3],
    /// Exact Laplacian in Pa/m².
    pub laplacian_pa_per_m2: Complex64,
    /// Manufactured source in Pa/m².
    pub source_pa_per_m2: Complex64,
}

impl HelmholtzPoint3d {
    fn all_finite(self) -> bool {
        self.position_m
            .into_iter()
            .chain([
                self.pressure_pa.re,
                self.pressure_pa.im,
                self.laplacian_pa_per_m2.re,
                self.laplacian_pa_per_m2.im,
                self.source_pa_per_m2.re,
                self.source_pa_per_m2.im,
            ])
            .chain(
                self.pressure_gradient_pa_per_m
                    .into_iter()
                    .flat_map(|value| [value.re, value.im]),
            )
            .all(f64::is_finite)
    }
}

/// Exact values on a row-major Cartesian grid (`z`, then `y`, then `x`).
#[derive(Debug, Clone, PartialEq)]
pub struct HelmholtzSample3d {
    /// Strictly increasing Cartesian positions in metres.
    pub x_positions_m: Vec<f64>,
    /// Strictly increasing Cartesian positions in metres.
    pub y_positions_m: Vec<f64>,
    /// Strictly increasing Cartesian positions in metres.
    pub z_positions_m: Vec<f64>,
    /// Exact complex pressures, indexed as `(z * y_count + y) * x_count + x`.
    pub pressures_pa: Vec<Complex64>,
    /// Exact Cartesian gradients in matching row-major order.
    pub pressure_gradients_pa_per_m: Vec<[Complex64; 3]>,
    /// Exact Laplacians in matching row-major order.
    pub laplacians_pa_per_m2: Vec<Complex64>,
    /// Exact sources in matching row-major order.
    pub sources_pa_per_m2: Vec<Complex64>,
}

impl HelmholtzSample3d {
    /// Validate axes, cardinality, and exact-field finiteness.
    pub fn validate(&self) -> Result<(), HelmholtzManufacturedError> {
        let axes = [
            &self.x_positions_m,
            &self.y_positions_m,
            &self.z_positions_m,
        ];
        if axes.iter().any(|axis| axis.len() < 2) {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample axes",
                requirement: "at least 2 samples on each axis",
            });
        }
        if axes.iter().any(|axis| {
            axis.iter().any(|value| !value.is_finite())
                || axis.windows(2).any(|pair| pair[1] <= pair[0])
        }) {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample axes",
                requirement: "finite and strictly increasing",
            });
        }
        let expected = self
            .x_positions_m
            .len()
            .checked_mul(self.y_positions_m.len())
            .and_then(|count| count.checked_mul(self.z_positions_m.len()))
            .ok_or(HelmholtzManufacturedError::EvaluationOverflow)?;
        if [
            self.pressures_pa.len(),
            self.pressure_gradients_pa_per_m.len(),
            self.laplacians_pa_per_m2.len(),
            self.sources_pa_per_m2.len(),
        ]
        .into_iter()
        .any(|length| length != expected)
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the Cartesian grid cardinality",
            });
        }
        if self
            .pressures_pa
            .iter()
            .chain(&self.laplacians_pa_per_m2)
            .chain(&self.sources_pa_per_m2)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
            || self
                .pressure_gradients_pa_per_m
                .iter()
                .flatten()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// Separable complex sine manufactured solution of the 3-D Helmholtz equation.
///
/// `p = P sin(q_x(x-x₀)) sin(q_y(y-y₀)) sin(q_z(z-z₀))` and
/// `f = (k²-q_x²-q_y²-q_z²)p`. Choosing `k² = q_x²+q_y²+q_z²` produces a
/// homogeneous Dirichlet box mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedHelmholtz3d {
    equation_wavenumber_per_m: f64,
    spatial_wavenumbers_per_m: [f64; 3],
    amplitude_pa: Complex64,
    origin_m: [f64; 3],
}

impl ManufacturedHelmholtz3d {
    /// Construct a checked constant-coefficient 3-D reference.
    pub fn try_new(
        equation_wavenumber_per_m: f64,
        spatial_wavenumbers_per_m: [f64; 3],
        amplitude_pa: Complex64,
        origin_m: [f64; 3],
    ) -> Result<Self, HelmholtzManufacturedError> {
        if !equation_wavenumber_per_m.is_finite() || equation_wavenumber_per_m <= 0.0 {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "equation_wavenumber_per_m",
                requirement: "finite and positive",
            });
        }
        if spatial_wavenumbers_per_m
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "spatial_wavenumbers_per_m",
                requirement: "all finite and positive",
            });
        }
        if [amplitude_pa.re, amplitude_pa.im]
            .into_iter()
            .chain(origin_m)
            .any(|value| !value.is_finite())
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "amplitude_pa/origin_m",
                requirement: "finite",
            });
        }
        let wave_sq = spatial_wavenumbers_per_m
            .into_iter()
            .map(|value| value.powi(2))
            .sum::<f64>();
        if !(equation_wavenumber_per_m.powi(2) - wave_sq).is_finite() {
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

    /// Evaluate exact pressure, gradient, Laplacian, and source.
    pub fn try_evaluate(
        self,
        position_m: [f64; 3],
    ) -> Result<HelmholtzPoint3d, HelmholtzManufacturedError> {
        if position_m.iter().any(|value| !value.is_finite()) {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "position_m",
                requirement: "finite",
            });
        }
        let phase: [f64; 3] = std::array::from_fn(|axis| {
            self.spatial_wavenumbers_per_m[axis] * (position_m[axis] - self.origin_m[axis])
        });
        let wave_sq = self
            .spatial_wavenumbers_per_m
            .into_iter()
            .map(|value| value.powi(2))
            .sum::<f64>();
        let source_factor = self.equation_wavenumber_per_m.powi(2) - wave_sq;
        if phase.iter().any(|value| !value.is_finite()) || !source_factor.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        let sine = phase.map(f64::sin);
        let pressure = self.amplitude_pa * sine.into_iter().product::<f64>();
        let gradient = std::array::from_fn(|axis| {
            let product = (0..3)
                .map(|other| {
                    if other == axis {
                        phase[other].cos()
                    } else {
                        sine[other]
                    }
                })
                .product::<f64>();
            self.amplitude_pa * (self.spatial_wavenumbers_per_m[axis] * product)
        });
        let point = HelmholtzPoint3d {
            position_m,
            pressure_pa: pressure,
            pressure_gradient_pa_per_m: gradient,
            laplacian_pa_per_m2: pressure * -wave_sq,
            source_pa_per_m2: pressure * source_factor,
        };
        if !point.all_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample the exact fields on a uniform Cartesian box.
    pub fn try_sample(
        self,
        x_bounds_m: [f64; 2],
        x_count: usize,
        y_bounds_m: [f64; 2],
        y_count: usize,
        z_bounds_m: [f64; 2],
        z_count: usize,
    ) -> Result<HelmholtzSample3d, HelmholtzManufacturedError> {
        let bounds = [x_bounds_m, y_bounds_m, z_bounds_m];
        let counts = [x_count, y_count, z_count];
        if counts.iter().any(|count| *count < 2) {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample counts",
                requirement: "at least 2 samples on each axis",
            });
        }
        if bounds
            .iter()
            .any(|bound| bound.iter().any(|value| !value.is_finite()) || bound[1] <= bound[0])
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample bounds",
                requirement: "finite and strictly increasing on all axes",
            });
        }
        let axes: [Vec<f64>; 3] = std::array::from_fn(|axis| {
            let step = (bounds[axis][1] - bounds[axis][0]) / (counts[axis] - 1) as f64;
            (0..counts[axis])
                .map(|index| bounds[axis][0] + index as f64 * step)
                .collect::<Vec<_>>()
        });
        let total = counts[0]
            .checked_mul(counts[1])
            .and_then(|count| count.checked_mul(counts[2]))
            .ok_or(HelmholtzManufacturedError::EvaluationOverflow)?;
        let mut sample = HelmholtzSample3d {
            x_positions_m: axes[0].clone(),
            y_positions_m: axes[1].clone(),
            z_positions_m: axes[2].clone(),
            pressures_pa: Vec::with_capacity(total),
            pressure_gradients_pa_per_m: Vec::with_capacity(total),
            laplacians_pa_per_m2: Vec::with_capacity(total),
            sources_pa_per_m2: Vec::with_capacity(total),
        };
        for &z_m in &sample.z_positions_m {
            for &y_m in &sample.y_positions_m {
                for &x_m in &sample.x_positions_m {
                    let point = self.try_evaluate([x_m, y_m, z_m])?;
                    sample.pressures_pa.push(point.pressure_pa);
                    sample
                        .pressure_gradients_pa_per_m
                        .push(point.pressure_gradient_pa_per_m);
                    sample.laplacians_pa_per_m2.push(point.laplacian_pa_per_m2);
                    sample.sources_pa_per_m2.push(point.source_pa_per_m2);
                }
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
    fn exact_terms_close_the_3d_helmholtz_equation() {
        let reference = ManufacturedHelmholtz3d::try_new(
            3.4,
            [1.3, 0.9, 1.1],
            Complex64::new(2.0, -0.7),
            [-0.2, 0.4, 0.1],
        )
        .unwrap();
        let sample = reference
            .try_sample([-1.0, 1.0], 5, [-0.5, 1.5], 4, [-0.2, 0.8], 3)
            .unwrap();
        for ((laplacian, pressure), source) in sample
            .laplacians_pa_per_m2
            .iter()
            .zip(&sample.pressures_pa)
            .zip(&sample.sources_pa_per_m2)
        {
            assert_abs_diff_eq!(
                (*laplacian + *pressure * reference.equation_wavenumber_per_m().powi(2) - *source)
                    .norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn homogeneous_mode_has_zero_source_and_box_boundary_trace() {
        let lengths = [2.0, 3.0, 1.5];
        let q = lengths.map(|length| PI / length);
        let reference = ManufacturedHelmholtz3d::try_new(
            q.into_iter().map(|value| value.powi(2)).sum::<f64>().sqrt(),
            q,
            Complex64::new(1.5, -0.4),
            [0.0; 3],
        )
        .unwrap();
        let sample = reference
            .try_sample(
                [0.0, lengths[0]],
                4,
                [0.0, lengths[1]],
                5,
                [0.0, lengths[2]],
                3,
            )
            .unwrap();
        assert!(
            sample
                .sources_pa_per_m2
                .iter()
                .all(|source| source.norm() < 1e-13)
        );
        let nx = sample.x_positions_m.len();
        let ny = sample.y_positions_m.len();
        for index in 0..sample.pressures_pa.len() {
            let x = index % nx;
            let y = (index / nx) % ny;
            let z = index / (nx * ny);
            if x == 0
                || x + 1 == nx
                || y == 0
                || y + 1 == ny
                || z == 0
                || z + 1 == sample.z_positions_m.len()
            {
                assert!(sample.pressures_pa[index].norm() < 1e-13);
            }
        }
    }

    #[test]
    fn gradients_agree_with_centered_finite_differences() {
        let reference = ManufacturedHelmholtz3d::try_new(
            3.0,
            [1.7, 1.1, 0.8],
            Complex64::new(-0.75, 0.4),
            [0.1, -0.3, 0.2],
        )
        .unwrap();
        let position = [0.45, 0.2, 0.35];
        let h = 1e-5;
        let point = reference.try_evaluate(position).unwrap();
        for axis in 0..3 {
            let mut plus = position;
            let mut minus = position;
            plus[axis] += h;
            minus[axis] -= h;
            let gradient = (reference.try_evaluate(plus).unwrap().pressure_pa
                - reference.try_evaluate(minus).unwrap().pressure_pa)
                / (2.0 * h);
            assert_abs_diff_eq!(
                (gradient - point.pressure_gradient_pa_per_m[axis]).norm(),
                0.0,
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_grids() {
        assert!(
            ManufacturedHelmholtz3d::try_new(
                1.0,
                [1.0, 0.0, 1.0],
                Complex64::new(1.0, 0.0),
                [0.0; 3],
            )
            .is_err()
        );
        let reference =
            ManufacturedHelmholtz3d::try_new(1.0, [1.0; 3], Complex64::new(1.0, 0.0), [0.0; 3])
                .unwrap();
        assert!(reference.try_evaluate([f64::NAN, 0.0, 0.0]).is_err());
        assert!(
            reference
                .try_sample([0.0, 1.0], 3, [0.0, 1.0], 3, [1.0, 0.0], 3)
                .is_err()
        );
    }
}
