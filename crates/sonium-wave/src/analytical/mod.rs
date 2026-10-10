//! Analytical solutions for wave equation validation
//!
//! This module provides exact solutions to acoustic scattering problems
//! used to validate numerical methods (BEM, FEM).
//!
//! ## Available Solutions
//!
//! - **1D**: Plane wave, standing wave, damped wave
//! - **2D**: Cylinder scattering (Bessel/Hankel series)
//! - **3D**: Sphere scattering (Mie theory)

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt;

pub mod acoustic_duct;
pub mod acoustic_interface;
pub mod annular_duct_modes;
pub mod axisymmetric_duct;
pub mod azimuthal;
pub mod manufactured;
pub mod manufactured_diffusion_2d;
pub mod manufactured_diffusion_3d;
pub mod manufactured_elasticity;
pub mod manufactured_helmholtz;
pub mod manufactured_helmholtz_2d;
pub mod manufactured_helmholtz_3d;
pub mod manufactured_incompressible_flow;
pub mod manufactured_maxwell;
pub mod manufactured_transport;
pub mod solutions_1d;
pub mod solutions_2d;
pub mod solutions_3d;

pub use acoustic_duct::*;
pub use acoustic_interface::*;
pub use annular_duct_modes::*;
pub use axisymmetric_duct::*;
pub use azimuthal::*;
pub use manufactured::*;
pub use manufactured_diffusion_2d::*;
pub use manufactured_diffusion_3d::*;
pub use manufactured_elasticity::*;
pub use manufactured_helmholtz::*;
pub use manufactured_helmholtz_2d::*;
pub use manufactured_helmholtz_3d::*;
pub use manufactured_incompressible_flow::*;
pub use manufactured_maxwell::*;
pub use manufactured_transport::*;
pub use solutions_1d::*;
pub use solutions_2d::*;
pub use solutions_3d::*;

/// Validation and comparison errors for analytical reference data.
///
/// The legacy scalar comparison helpers retain their historical infallible
/// signatures for compatibility.  New callers should use the checked
/// methods, which reject malformed metadata and mismatched sample grids
/// before doing any arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum AnalyticalError {
    /// The dimensionality is not one of Sonium's supported spatial dimensions.
    InvalidDimensions { dimensions: usize },
    /// A metadata scalar or coordinate contains NaN or infinity.
    NonFinite {
        field: &'static str,
        index: Option<usize>,
    },
    /// The two solutions contain a different number of pressure samples.
    SampleCountMismatch { lhs: usize, rhs: usize },
    /// A pressure vector does not have one value for each position.
    PositionPressureMismatch { positions: usize, pressure: usize },
}

impl fmt::Display for AnalyticalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { dimensions } => {
                write!(
                    f,
                    "analytical solution dimensions {dimensions} must be 1, 2, or 3"
                )
            }
            Self::NonFinite { field, index } => match index {
                Some(index) => write!(
                    f,
                    "analytical solution field {field}[{index}] is non-finite"
                ),
                None => write!(f, "analytical solution field {field} is non-finite"),
            },
            Self::SampleCountMismatch { lhs, rhs } => {
                write!(f, "analytical sample counts differ: lhs={lhs}, rhs={rhs}")
            }
            Self::PositionPressureMismatch {
                positions,
                pressure,
            } => {
                write!(
                    f,
                    "positions ({positions}) and pressure ({pressure}) lengths differ"
                )
            }
        }
    }
}

impl std::error::Error for AnalyticalError {}

/// Point in space (1D, 2D, or 3D)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Point {
    /// x-coordinate
    pub x: f64,
    /// y-coordinate (0 for 1D)
    pub y: f64,
    /// z-coordinate (0 for 1D/2D)
    pub z: f64,
}

impl Point {
    /// Create 1D point
    pub fn new_1d(x: f64) -> Self {
        Self { x, y: 0.0, z: 0.0 }
    }

    /// Create 2D point
    pub fn new_2d(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    /// Create 3D point
    pub fn new_3d(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Polar coordinates (r, θ) for 2D
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new_2d(r * theta.cos(), r * theta.sin())
    }

    /// Spherical coordinates (r, θ, φ) for 3D
    /// θ = polar angle from z-axis
    /// φ = azimuthal angle in xy-plane
    pub fn from_spherical(r: f64, theta: f64, phi: f64) -> Self {
        Self::new_3d(
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
            r * theta.cos(),
        )
    }

    /// Distance from origin
    pub fn radius(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Polar angle (2D) - angle from positive x-axis
    pub fn theta_2d(&self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Spherical polar angle (3D, from z-axis)
    pub fn theta_3d(&self) -> f64 {
        let r = self.radius();
        if r < 1e-15 {
            0.0
        } else {
            (self.z / r).clamp(-1.0, 1.0).acos()
        }
    }

    /// Spherical azimuthal angle (3D)
    pub fn phi_3d(&self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Distance to another point
    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl Default for Point {
    fn default() -> Self {
        Self::new_3d(0.0, 0.0, 0.0)
    }
}

/// Analytical solution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticalSolution {
    /// Test name
    pub name: String,
    /// Dimensionality (1, 2, or 3)
    pub dimensions: usize,
    /// Evaluation points
    pub positions: Vec<Point>,
    /// Complex pressure values
    pub pressure: Vec<Complex64>,
    /// Wave number k = ω/c = 2πf/c
    pub wave_number: f64,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl AnalyticalSolution {
    /// Create a new analytical solution
    pub fn new(
        name: impl Into<String>,
        dimensions: usize,
        positions: Vec<Point>,
        pressure: Vec<Complex64>,
        wave_number: f64,
        frequency: f64,
    ) -> Self {
        Self {
            name: name.into(),
            dimensions,
            positions,
            pressure,
            wave_number,
            frequency,
            metadata: serde_json::json!({}),
        }
    }

    /// Validate dimensions, metadata, coordinates, and sample cardinality.
    pub fn validate(&self) -> Result<(), AnalyticalError> {
        if !(1..=3).contains(&self.dimensions) {
            return Err(AnalyticalError::InvalidDimensions {
                dimensions: self.dimensions,
            });
        }
        if !self.wave_number.is_finite() {
            return Err(AnalyticalError::NonFinite {
                field: "wave_number",
                index: None,
            });
        }
        if !self.frequency.is_finite() {
            return Err(AnalyticalError::NonFinite {
                field: "frequency",
                index: None,
            });
        }
        if self.positions.len() != self.pressure.len() {
            return Err(AnalyticalError::PositionPressureMismatch {
                positions: self.positions.len(),
                pressure: self.pressure.len(),
            });
        }
        for (index, point) in self.positions.iter().enumerate() {
            if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
                return Err(AnalyticalError::NonFinite {
                    field: "positions",
                    index: Some(index),
                });
            }
        }
        for (index, pressure) in self.pressure.iter().enumerate() {
            if !pressure.re.is_finite() || !pressure.im.is_finite() {
                return Err(AnalyticalError::NonFinite {
                    field: "pressure",
                    index: Some(index),
                });
            }
        }
        Ok(())
    }

    /// Compute an L2 error after validating both reference solutions.
    pub fn try_l2_error(&self, other: &AnalyticalSolution) -> Result<f64, AnalyticalError> {
        self.validate()?;
        other.validate()?;
        if self.pressure.len() != other.pressure.len() {
            return Err(AnalyticalError::SampleCountMismatch {
                lhs: self.pressure.len(),
                rhs: other.pressure.len(),
            });
        }
        Ok(self
            .pressure
            .iter()
            .zip(other.pressure.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt())
    }

    /// Compute a relative L2 error after validating both reference solutions.
    pub fn try_relative_l2_error(
        &self,
        other: &AnalyticalSolution,
    ) -> Result<f64, AnalyticalError> {
        let l2_err = self.try_l2_error(other)?;
        let norm = other
            .pressure
            .iter()
            .map(|pressure| pressure.norm_sqr())
            .sum::<f64>()
            .sqrt();
        Ok(if norm < 1e-15 { l2_err } else { l2_err / norm })
    }

    /// Compute a max-norm error after validating both reference solutions.
    pub fn try_linf_error(&self, other: &AnalyticalSolution) -> Result<f64, AnalyticalError> {
        self.validate()?;
        other.validate()?;
        if self.pressure.len() != other.pressure.len() {
            return Err(AnalyticalError::SampleCountMismatch {
                lhs: self.pressure.len(),
                rhs: other.pressure.len(),
            });
        }
        Ok(self
            .pressure
            .iter()
            .zip(other.pressure.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max))
    }

    /// Compute pressure magnitude |p|
    pub fn magnitude(&self) -> Vec<f64> {
        self.pressure.iter().map(|p| p.norm()).collect()
    }

    /// Compute pressure phase (radians)
    pub fn phase(&self) -> Vec<f64> {
        self.pressure.iter().map(|p| p.arg()).collect()
    }

    /// Real part of pressure
    pub fn real(&self) -> Vec<f64> {
        self.pressure.iter().map(|p| p.re).collect()
    }

    /// Imaginary part of pressure
    pub fn imag(&self) -> Vec<f64> {
        self.pressure.iter().map(|p| p.im).collect()
    }

    /// Compute L2 error compared to another solution
    ///
    /// # Panics
    ///
    /// Panics if the two solutions have different lengths. Prefer the
    /// checked `try_l2_error` variant where inputs are user-controlled.
    pub fn l2_error(&self, other: &AnalyticalSolution) -> f64 {
        assert_eq!(self.pressure.len(), other.pressure.len());

        let sum_sq: f64 = self
            .pressure
            .iter()
            .zip(other.pressure.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum();

        sum_sq.sqrt()
    }

    /// Compute relative L2 error
    pub fn relative_l2_error(&self, other: &AnalyticalSolution) -> f64 {
        let l2_err = self.l2_error(other);
        let norm: f64 = other
            .pressure
            .iter()
            .map(|p| p.norm_sqr())
            .sum::<f64>()
            .sqrt();

        if norm < 1e-15 { l2_err } else { l2_err / norm }
    }

    /// Compute max (L∞) error
    ///
    /// # Panics
    ///
    /// Panics if the two solutions have different lengths. Prefer the
    /// checked `try_linf_error` variant where inputs are user-controlled.
    pub fn linf_error(&self, other: &AnalyticalSolution) -> f64 {
        assert_eq!(self.pressure.len(), other.pressure.len());

        self.pressure
            .iter()
            .zip(other.pressure.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_point_creation() {
        let p1d = Point::new_1d(1.0);
        assert_eq!(p1d.x, 1.0);
        assert_eq!(p1d.y, 0.0);
        assert_eq!(p1d.z, 0.0);

        let p2d = Point::from_polar(1.0, PI / 4.0);
        assert!((p2d.radius() - 1.0).abs() < 1e-10);

        let p3d = Point::from_spherical(1.0, PI / 2.0, 0.0);
        assert!((p3d.radius() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_distance() {
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(4.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solution_error() {
        let sol1 = AnalyticalSolution::new(
            "test1",
            1,
            vec![Point::new_1d(0.0)],
            vec![Complex64::new(1.0, 0.0)],
            1.0,
            1.0,
        );

        let sol2 = AnalyticalSolution::new(
            "test2",
            1,
            vec![Point::new_1d(0.0)],
            vec![Complex64::new(1.1, 0.0)],
            1.0,
            1.0,
        );

        assert!((sol1.l2_error(&sol2) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn checked_solution_comparison_rejects_malformed_metadata() {
        let mut solution = AnalyticalSolution::new(
            "bad",
            4,
            vec![Point::new_1d(0.0)],
            vec![Complex64::new(1.0, 0.0)],
            1.0,
            1.0,
        );
        let valid = solution.clone();
        assert!(matches!(
            solution.try_l2_error(&valid),
            Err(AnalyticalError::InvalidDimensions { dimensions: 4 })
        ));

        solution.dimensions = 1;
        solution.positions.clear();
        assert!(matches!(
            solution.try_linf_error(&valid),
            Err(AnalyticalError::PositionPressureMismatch { .. })
        ));
    }

    #[test]
    fn checked_solution_comparison_rejects_mismatched_sample_counts() {
        let lhs = AnalyticalSolution::new(
            "lhs",
            1,
            vec![Point::new_1d(0.0)],
            vec![Complex64::new(1.0, 0.0)],
            1.0,
            1.0,
        );
        let rhs = AnalyticalSolution::new(
            "rhs",
            1,
            vec![Point::new_1d(0.0), Point::new_1d(1.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            1.0,
            1.0,
        );
        assert!(matches!(
            lhs.try_relative_l2_error(&rhs),
            Err(AnalyticalError::SampleCountMismatch { lhs: 1, rhs: 2 })
        ));
    }

    #[test]
    fn checked_solution_validation_rejects_nonfinite_values() {
        let solution = AnalyticalSolution::new(
            "nan",
            1,
            vec![Point::new_1d(f64::NAN)],
            vec![Complex64::new(1.0, 0.0)],
            1.0,
            1.0,
        );
        assert!(matches!(
            solution.validate(),
            Err(AnalyticalError::NonFinite {
                field: "positions",
                index: Some(0)
            })
        ));
    }
}
