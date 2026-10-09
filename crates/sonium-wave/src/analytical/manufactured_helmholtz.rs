//! Manufactured one-dimensional Helmholtz references.
//!
//! This module supplies an independently evaluable complex-pressure reference
//! for the constant-coefficient equation
//!
//! ```text
//! ∂²p/∂x² + k²p = f(x),
//! ```
//!
//! with `k > 0` in rad/m.  The complex pressure amplitude uses Sonium's
//! `exp(+iωt)` phasor convention; the equation itself is independent of that
//! time-sign choice.  A numerical Helmholtz implementation can use the
//! exposed state, gradient, Laplacian, and source independently, rather than
//! comparing two implementations of the same discretization.

use num_complex::Complex64;
use std::fmt;

/// Invalid input to a manufactured one-dimensional Helmholtz reference.
#[derive(Debug, Clone, PartialEq)]
pub enum HelmholtzManufacturedError {
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
    /// An evaluated phase or exact term is not representable.
    EvaluationOverflow,
}

impl fmt::Display for HelmholtzManufacturedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => write!(
                f,
                "manufactured Helmholtz parameter {field} must be {requirement}"
            ),
            Self::InvalidInterval { x_min, x_max } => write!(
                f,
                "manufactured Helmholtz interval must be finite and strictly increasing, got [{x_min}, {x_max}]"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "manufactured Helmholtz evaluation cannot be represented in binary64"
            ),
        }
    }
}

impl std::error::Error for HelmholtzManufacturedError {}

/// Exact state and operator terms at one Helmholtz evaluation point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzPoint1d {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// Exact complex pressure amplitude `p` in Pa.
    pub pressure_pa: Complex64,
    /// Exact pressure gradient `∂p/∂x` in Pa/m.
    pub pressure_gradient_pa_per_m: Complex64,
    /// Exact pressure Laplacian `∂²p/∂x²` in Pa/m².
    pub laplacian_pa_per_m2: Complex64,
    /// Manufactured source `f = ∂²p/∂x² + k²p` in Pa/m².
    pub source_pa_per_m2: Complex64,
}

impl HelmholtzPoint1d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.pressure_pa.re,
            self.pressure_pa.im,
            self.pressure_gradient_pa_per_m.re,
            self.pressure_gradient_pa_per_m.im,
            self.laplacian_pa_per_m2.re,
            self.laplacian_pa_per_m2.im,
            self.source_pa_per_m2.re,
            self.source_pa_per_m2.im,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// Exact sensitivities of a manufactured Helmholtz point.
///
/// `origin_m` is a rigid translation of the reference field, making its
/// derivative a canonical one-dimensional shape derivative. The equation
/// wavenumber derivative is a material/frequency sensitivity of the forcing:
/// the prescribed pressure field itself is held fixed while the operator
/// `∂²/∂x² + k²` changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzSensitivityPoint1d {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// `∂p/∂x₀` in Pa/m, where `x₀` is the translated-field origin.
    pub pressure_origin_derivative_pa_per_m: Complex64,
    /// `∂(∂p/∂x)/∂x₀` in Pa/m².
    pub pressure_gradient_origin_derivative_pa_per_m2: Complex64,
    /// `∂(∂²p/∂x²)/∂x₀` in Pa/m³.
    pub laplacian_origin_derivative_pa_per_m3: Complex64,
    /// `∂f/∂x₀` in Pa/m³.
    pub source_origin_derivative_pa_per_m3: Complex64,
    /// `∂f/∂k` in Pa/m, with the prescribed pressure held fixed.
    pub source_equation_wavenumber_derivative_pa_per_m: Complex64,
}

impl HelmholtzSensitivityPoint1d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.pressure_origin_derivative_pa_per_m.re,
            self.pressure_origin_derivative_pa_per_m.im,
            self.pressure_gradient_origin_derivative_pa_per_m2.re,
            self.pressure_gradient_origin_derivative_pa_per_m2.im,
            self.laplacian_origin_derivative_pa_per_m3.re,
            self.laplacian_origin_derivative_pa_per_m3.im,
            self.source_origin_derivative_pa_per_m3.re,
            self.source_origin_derivative_pa_per_m3.im,
            self.source_equation_wavenumber_derivative_pa_per_m.re,
            self.source_equation_wavenumber_derivative_pa_per_m.im,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// Uniform exact samples and operator terms for a Helmholtz reference.
#[derive(Debug, Clone, PartialEq)]
pub struct HelmholtzSample1d {
    /// Sample coordinates in metres.
    pub positions_m: Vec<f64>,
    /// Exact complex pressure amplitudes in Pa.
    pub pressures_pa: Vec<Complex64>,
    /// Exact pressure gradients in Pa/m.
    pub pressure_gradients_pa_per_m: Vec<Complex64>,
    /// Exact Laplacians in Pa/m².
    pub laplacians_pa_per_m2: Vec<Complex64>,
    /// Exact manufactured sources in Pa/m².
    pub sources_pa_per_m2: Vec<Complex64>,
}

impl HelmholtzSample1d {
    /// Verify that every sample has a finite, consistently sized field set.
    pub fn validate(&self) -> Result<(), HelmholtzManufacturedError> {
        let count = self.positions_m.len();
        if count < 2 {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "positions_m",
                requirement: "at least 2 samples",
            });
        }
        if [
            self.pressures_pa.len(),
            self.pressure_gradients_pa_per_m.len(),
            self.laplacians_pa_per_m2.len(),
            self.sources_pa_per_m2.len(),
        ]
        .iter()
        .any(|&length| length != count)
        {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the same length as positions_m",
            });
        }
        if self.positions_m.iter().any(|value| !value.is_finite())
            || self
                .pressures_pa
                .iter()
                .chain(self.pressure_gradients_pa_per_m.iter())
                .chain(self.laplacians_pa_per_m2.iter())
                .chain(self.sources_pa_per_m2.iter())
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A smooth manufactured complex-pressure solution of the 1-D Helmholtz equation.
///
/// The pressure amplitude is `p(x) = P sin(q(x-x₀))`, where `P` is a finite
/// complex pressure amplitude and `q > 0` is the spatial wavenumber.  The
/// matching source is `f = (k²-q²)p`.  Choosing `q = k` produces an unforced
/// homogeneous Dirichlet mode on any interval whose endpoints lie at zeros of
/// the sine factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedHelmholtz1d {
    equation_wavenumber_per_m: f64,
    spatial_wavenumber_per_m: f64,
    amplitude_pa: Complex64,
    origin_m: f64,
}

impl ManufacturedHelmholtz1d {
    /// Construct a checked constant-coefficient Helmholtz reference.
    pub fn try_new(
        equation_wavenumber_per_m: f64,
        spatial_wavenumber_per_m: f64,
        amplitude_pa: Complex64,
        origin_m: f64,
    ) -> Result<Self, HelmholtzManufacturedError> {
        for (field, value) in [
            ("equation_wavenumber_per_m", equation_wavenumber_per_m),
            ("spatial_wavenumber_per_m", spatial_wavenumber_per_m),
            ("origin_m", origin_m),
            ("amplitude_pa.re", amplitude_pa.re),
            ("amplitude_pa.im", amplitude_pa.im),
        ] {
            let valid = match field {
                "equation_wavenumber_per_m" | "spatial_wavenumber_per_m" => {
                    value.is_finite() && value > 0.0
                }
                _ => value.is_finite(),
            };
            if !valid {
                let requirement = match field {
                    "equation_wavenumber_per_m" | "spatial_wavenumber_per_m" => {
                        "finite and positive"
                    }
                    _ => "finite",
                };
                return Err(HelmholtzManufacturedError::InvalidParameter { field, requirement });
            }
        }
        let source_factor = equation_wavenumber_per_m.powi(2) - spatial_wavenumber_per_m.powi(2);
        if !source_factor.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            equation_wavenumber_per_m,
            spatial_wavenumber_per_m,
            amplitude_pa,
            origin_m,
        })
    }

    /// Helmholtz equation wavenumber `k` in rad/m.
    pub fn equation_wavenumber_per_m(self) -> f64 {
        self.equation_wavenumber_per_m
    }

    /// Spatial sinusoid wavenumber `q` in rad/m.
    pub fn spatial_wavenumber_per_m(self) -> f64 {
        self.spatial_wavenumber_per_m
    }

    /// Translation origin `x₀` in metres.
    pub fn origin_m(self) -> f64 {
        self.origin_m
    }

    /// Evaluate the exact pressure and Helmholtz operator terms at `x`.
    pub fn try_evaluate(self, x_m: f64) -> Result<HelmholtzPoint1d, HelmholtzManufacturedError> {
        if !x_m.is_finite() {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "x_m",
                requirement: "finite",
            });
        }
        let phase = self.spatial_wavenumber_per_m * (x_m - self.origin_m);
        let source_factor =
            self.equation_wavenumber_per_m.powi(2) - self.spatial_wavenumber_per_m.powi(2);
        if !phase.is_finite() || !source_factor.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        let pressure = self.amplitude_pa * phase.sin();
        let gradient = self.amplitude_pa * (self.spatial_wavenumber_per_m * phase.cos());
        let laplacian = pressure * -self.spatial_wavenumber_per_m.powi(2);
        let source = pressure * source_factor;
        let point = HelmholtzPoint1d {
            x_m,
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

    /// Evaluate exact shape and material sensitivities at `x`.
    ///
    /// For `p = P sin(q(x-x₀))`, rigid translation gives
    /// `∂p/∂x₀ = -∂p/∂x`.  Holding `p` fixed while changing the equation
    /// wavenumber gives `∂f/∂k = 2kp`.  These identities make this reference
    /// suitable for independently Taylor-testing an adjoint or a material
    /// sensitivity implementation.
    pub fn try_evaluate_sensitivities(
        self,
        x_m: f64,
    ) -> Result<HelmholtzSensitivityPoint1d, HelmholtzManufacturedError> {
        let point = self.try_evaluate(x_m)?;
        let spatial_wavenumber_squared = self.spatial_wavenumber_per_m.powi(2);
        let source_factor =
            self.equation_wavenumber_per_m.powi(2) - self.spatial_wavenumber_per_m.powi(2);
        if !spatial_wavenumber_squared.is_finite() || !source_factor.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        let sensitivity = HelmholtzSensitivityPoint1d {
            x_m,
            pressure_origin_derivative_pa_per_m: -point.pressure_gradient_pa_per_m,
            pressure_gradient_origin_derivative_pa_per_m2: -point.laplacian_pa_per_m2,
            laplacian_origin_derivative_pa_per_m3: point.pressure_gradient_pa_per_m
                * spatial_wavenumber_squared,
            source_origin_derivative_pa_per_m3: -point.pressure_gradient_pa_per_m * source_factor,
            source_equation_wavenumber_derivative_pa_per_m: point.pressure_pa
                * (2.0 * self.equation_wavenumber_per_m),
        };
        if !sensitivity.all_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        Ok(sensitivity)
    }

    /// Sample the reference and all exact terms on a uniform spatial grid.
    pub fn try_sample(
        self,
        x_min_m: f64,
        x_max_m: f64,
        num_points: usize,
    ) -> Result<HelmholtzSample1d, HelmholtzManufacturedError> {
        if num_points < 2 {
            return Err(HelmholtzManufacturedError::InvalidParameter {
                field: "num_points",
                requirement: "at least 2",
            });
        }
        if !x_min_m.is_finite() || !x_max_m.is_finite() || x_max_m <= x_min_m {
            return Err(HelmholtzManufacturedError::InvalidInterval {
                x_min: x_min_m,
                x_max: x_max_m,
            });
        }
        let dx = (x_max_m - x_min_m) / (num_points - 1) as f64;
        if !dx.is_finite() {
            return Err(HelmholtzManufacturedError::EvaluationOverflow);
        }
        let mut sample = HelmholtzSample1d {
            positions_m: Vec::with_capacity(num_points),
            pressures_pa: Vec::with_capacity(num_points),
            pressure_gradients_pa_per_m: Vec::with_capacity(num_points),
            laplacians_pa_per_m2: Vec::with_capacity(num_points),
            sources_pa_per_m2: Vec::with_capacity(num_points),
        };
        for index in 0..num_points {
            let point = self.try_evaluate(x_min_m + index as f64 * dx)?;
            sample.positions_m.push(point.x_m);
            sample.pressures_pa.push(point.pressure_pa);
            sample
                .pressure_gradients_pa_per_m
                .push(point.pressure_gradient_pa_per_m);
            sample.laplacians_pa_per_m2.push(point.laplacian_pa_per_m2);
            sample.sources_pa_per_m2.push(point.source_pa_per_m2);
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
    fn exact_terms_close_the_helmholtz_equation() {
        let reference =
            ManufacturedHelmholtz1d::try_new(3.1, 1.3, Complex64::new(2.0, -0.7), -0.2).unwrap();
        let sample = reference.try_sample(-1.0, 1.0, 17).unwrap();
        for index in 0..sample.positions_m.len() {
            let residual = sample.laplacians_pa_per_m2[index]
                + sample.pressures_pa[index] * reference.equation_wavenumber_per_m().powi(2)
                - sample.sources_pa_per_m2[index];
            assert_abs_diff_eq!(residual.norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn homogeneous_mode_has_zero_source_and_dirichlet_endpoints() {
        let length = 2.0;
        let wavenumber = PI / length;
        let reference = ManufacturedHelmholtz1d::try_new(
            wavenumber,
            wavenumber,
            Complex64::new(1.5, -0.4),
            0.0,
        )
        .unwrap();
        let sample = reference.try_sample(0.0, length, 9).unwrap();
        assert_abs_diff_eq!(sample.pressures_pa[0].norm(), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(
            sample.pressures_pa.last().unwrap().norm(),
            0.0,
            epsilon = 1e-14
        );
        for source in sample.sources_pa_per_m2 {
            assert_abs_diff_eq!(source.norm(), 0.0, epsilon = 1e-13);
        }
    }

    #[test]
    fn gradient_agrees_with_centered_finite_difference() {
        let reference =
            ManufacturedHelmholtz1d::try_new(2.5, 1.7, Complex64::new(-0.75, 0.4), 0.1).unwrap();
        let x = 0.45;
        let h = 1e-5;
        let point = reference.try_evaluate(x).unwrap();
        let left = reference.try_evaluate(x - h).unwrap().pressure_pa;
        let right = reference.try_evaluate(x + h).unwrap().pressure_pa;
        assert_abs_diff_eq!(
            ((right - left) / (2.0 * h) - point.pressure_gradient_pa_per_m).norm(),
            0.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            ((right - 2.0 * point.pressure_pa + left) / h.powi(2) - point.laplacian_pa_per_m2)
                .norm(),
            0.0,
            epsilon = 3e-6
        );
    }

    #[test]
    fn shape_and_material_sensitivities_match_centered_differences() {
        let reference =
            ManufacturedHelmholtz1d::try_new(3.1, 1.3, Complex64::new(2.0, -0.7), -0.2).unwrap();
        let x = 0.45;
        let h = 1e-5;
        let sensitivity = reference.try_evaluate_sensitivities(x).unwrap();

        let left_origin = ManufacturedHelmholtz1d::try_new(
            reference.equation_wavenumber_per_m(),
            reference.spatial_wavenumber_per_m(),
            Complex64::new(2.0, -0.7),
            reference.origin_m() - h,
        )
        .unwrap()
        .try_evaluate(x)
        .unwrap();
        let right_origin = ManufacturedHelmholtz1d::try_new(
            reference.equation_wavenumber_per_m(),
            reference.spatial_wavenumber_per_m(),
            Complex64::new(2.0, -0.7),
            reference.origin_m() + h,
        )
        .unwrap()
        .try_evaluate(x)
        .unwrap();
        assert_abs_diff_eq!(
            ((right_origin.pressure_pa - left_origin.pressure_pa) / (2.0 * h)
                - sensitivity.pressure_origin_derivative_pa_per_m)
                .norm(),
            0.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            ((right_origin.source_pa_per_m2 - left_origin.source_pa_per_m2) / (2.0 * h)
                - sensitivity.source_origin_derivative_pa_per_m3)
                .norm(),
            0.0,
            epsilon = 1e-9
        );

        let left_k = ManufacturedHelmholtz1d::try_new(
            reference.equation_wavenumber_per_m() - h,
            reference.spatial_wavenumber_per_m(),
            Complex64::new(2.0, -0.7),
            reference.origin_m(),
        )
        .unwrap()
        .try_evaluate(x)
        .unwrap();
        let right_k = ManufacturedHelmholtz1d::try_new(
            reference.equation_wavenumber_per_m() + h,
            reference.spatial_wavenumber_per_m(),
            Complex64::new(2.0, -0.7),
            reference.origin_m(),
        )
        .unwrap()
        .try_evaluate(x)
        .unwrap();
        assert_abs_diff_eq!(
            ((right_k.source_pa_per_m2 - left_k.source_pa_per_m2) / (2.0 * h)
                - sensitivity.source_equation_wavenumber_derivative_pa_per_m)
                .norm(),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn origin_shape_derivative_has_second_order_taylor_remainder() {
        let reference =
            ManufacturedHelmholtz1d::try_new(3.1, 1.3, Complex64::new(2.0, -0.7), -0.2).unwrap();
        let x = 0.45;
        let baseline = reference.try_evaluate(x).unwrap().pressure_pa;
        let derivative = reference
            .try_evaluate_sensitivities(x)
            .unwrap()
            .pressure_origin_derivative_pa_per_m;
        let remainder = |h: f64| {
            let perturbed = ManufacturedHelmholtz1d::try_new(
                reference.equation_wavenumber_per_m(),
                reference.spatial_wavenumber_per_m(),
                Complex64::new(2.0, -0.7),
                reference.origin_m() + h,
            )
            .unwrap()
            .try_evaluate(x)
            .unwrap()
            .pressure_pa;
            (perturbed - baseline - derivative * h).norm()
        };
        let coarse = remainder(1e-3);
        let fine = remainder(5e-4);
        assert!(coarse.is_finite() && fine.is_finite());
        assert!(fine < coarse * 0.251, "coarse={coarse:e}, fine={fine:e}");
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_evaluations() {
        assert!(matches!(
            ManufacturedHelmholtz1d::try_new(0.0, 1.0, Complex64::new(1.0, 0.0), 0.0),
            Err(HelmholtzManufacturedError::InvalidParameter {
                field: "equation_wavenumber_per_m",
                ..
            })
        ));
        let reference =
            ManufacturedHelmholtz1d::try_new(1.0, 1.0, Complex64::new(1.0, 0.0), 0.0).unwrap();
        assert!(matches!(
            reference.try_evaluate(f64::NAN),
            Err(HelmholtzManufacturedError::InvalidParameter { field: "x_m", .. })
        ));
        assert!(matches!(
            reference.try_sample(1.0, 0.0, 4),
            Err(HelmholtzManufacturedError::InvalidInterval { .. })
        ));
        assert!(matches!(
            reference.try_sample(0.0, 1.0, 1),
            Err(HelmholtzManufacturedError::InvalidParameter {
                field: "num_points",
                ..
            })
        ));
    }
}
