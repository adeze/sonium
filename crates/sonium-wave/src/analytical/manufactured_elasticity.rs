//! Manufactured one-dimensional linear-elasticity references.
//!
//! This module defines a static, constant-section axial bar under the
//! equilibrium convention
//!
//! ```text
//! -d(E A du/dx)/dx = b(x),
//! ```
//!
//! where `E` is Young's modulus, `A` is cross-sectional area, `u` is axial
//! displacement, and `b` is the distributed axial load in N/m.  It exposes
//! the exact displacement, strain, stress, axial force, and load so a
//! consuming discretization can check each physical term independently.

use std::fmt;

/// Invalid input to a manufactured one-dimensional elasticity reference.
#[derive(Debug, Clone, PartialEq)]
pub enum ElasticityManufacturedError {
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

impl fmt::Display for ElasticityManufacturedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(
                    f,
                    "manufactured elasticity parameter {field} must be {requirement}"
                )
            }
            Self::InvalidInterval { x_min, x_max } => write!(
                f,
                "manufactured elasticity interval must be finite and strictly increasing, got [{x_min}, {x_max}]"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "manufactured elasticity evaluation cannot be represented in binary64"
            ),
        }
    }
}

impl std::error::Error for ElasticityManufacturedError {}

/// Exact state and equilibrium terms at one axial-bar evaluation point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticityPoint1d {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// Exact axial displacement `u` in metres.
    pub displacement_m: f64,
    /// Exact axial strain `du/dx`, dimensionless.
    pub axial_strain: f64,
    /// Exact axial stress `E du/dx` in Pa.
    pub axial_stress_pa: f64,
    /// Exact axial force `E A du/dx` in N.
    pub axial_force_n: f64,
    /// Exact distributed load `b = -d(E A du/dx)/dx` in N/m.
    pub distributed_load_n_per_m: f64,
}

impl ElasticityPoint1d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.displacement_m,
            self.axial_strain,
            self.axial_stress_pa,
            self.axial_force_n,
            self.distributed_load_n_per_m,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// Uniform exact samples and equilibrium terms for an axial-bar reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticitySample1d {
    /// Sample coordinates in metres.
    pub positions_m: Vec<f64>,
    /// Exact axial displacements in metres.
    pub displacements_m: Vec<f64>,
    /// Exact axial strains, dimensionless.
    pub axial_strains: Vec<f64>,
    /// Exact axial stresses in Pa.
    pub axial_stresses_pa: Vec<f64>,
    /// Exact axial forces in N.
    pub axial_forces_n: Vec<f64>,
    /// Exact distributed loads in N/m.
    pub distributed_loads_n_per_m: Vec<f64>,
}

impl ElasticitySample1d {
    /// Verify a finite, consistently-sized exact field set.
    pub fn validate(&self) -> Result<(), ElasticityManufacturedError> {
        let count = self.positions_m.len();
        if count < 2 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "positions_m",
                requirement: "at least 2 samples",
            });
        }
        if [
            self.displacements_m.len(),
            self.axial_strains.len(),
            self.axial_stresses_pa.len(),
            self.axial_forces_n.len(),
            self.distributed_loads_n_per_m.len(),
        ]
        .iter()
        .any(|&length| length != count)
        {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the same length as positions_m",
            });
        }
        if self
            .positions_m
            .iter()
            .chain(self.displacements_m.iter())
            .chain(self.axial_strains.iter())
            .chain(self.axial_stresses_pa.iter())
            .chain(self.axial_forces_n.iter())
            .chain(self.distributed_loads_n_per_m.iter())
            .any(|value| !value.is_finite())
        {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A smooth manufactured solution for a constant-section static axial bar.
///
/// The exact displacement is `u(x) = U sin(q(x-x₀))`, where `U` is a finite
/// signed displacement amplitude and `q > 0` is the spatial wavenumber.  The
/// reference assumes small-strain, homogeneous, isotropic, one-dimensional
/// linear elasticity with constant positive `E` and `A`; it is not a bending,
/// nonlinear, or dynamic structural model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedElasticity1d {
    youngs_modulus_pa: f64,
    area_m2: f64,
    spatial_wavenumber_per_m: f64,
    displacement_amplitude_m: f64,
    origin_m: f64,
}

impl ManufacturedElasticity1d {
    /// Construct a checked constant-section axial-bar reference.
    pub fn try_new(
        youngs_modulus_pa: f64,
        area_m2: f64,
        spatial_wavenumber_per_m: f64,
        displacement_amplitude_m: f64,
        origin_m: f64,
    ) -> Result<Self, ElasticityManufacturedError> {
        for (field, value, positive) in [
            ("youngs_modulus_pa", youngs_modulus_pa, true),
            ("area_m2", area_m2, true),
            ("spatial_wavenumber_per_m", spatial_wavenumber_per_m, true),
            ("displacement_amplitude_m", displacement_amplitude_m, false),
            ("origin_m", origin_m, false),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) {
                return Err(ElasticityManufacturedError::InvalidParameter {
                    field,
                    requirement: if positive {
                        "finite and positive"
                    } else {
                        "finite"
                    },
                });
            }
        }
        let stiffness_n = youngs_modulus_pa * area_m2;
        let load_scale = stiffness_n * spatial_wavenumber_per_m.powi(2);
        if !stiffness_n.is_finite() || !load_scale.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            youngs_modulus_pa,
            area_m2,
            spatial_wavenumber_per_m,
            displacement_amplitude_m,
            origin_m,
        })
    }

    /// Young's modulus `E` in Pa.
    pub fn youngs_modulus_pa(self) -> f64 {
        self.youngs_modulus_pa
    }

    /// Cross-sectional area `A` in m².
    pub fn area_m2(self) -> f64 {
        self.area_m2
    }

    /// Spatial sinusoid wavenumber `q` in rad/m.
    pub fn spatial_wavenumber_per_m(self) -> f64 {
        self.spatial_wavenumber_per_m
    }

    /// Evaluate the exact displacement and equilibrium terms at `x`.
    pub fn try_evaluate(self, x_m: f64) -> Result<ElasticityPoint1d, ElasticityManufacturedError> {
        if !x_m.is_finite() {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "x_m",
                requirement: "finite",
            });
        }
        let phase = self.spatial_wavenumber_per_m * (x_m - self.origin_m);
        let stiffness_n = self.youngs_modulus_pa * self.area_m2;
        let load_scale = stiffness_n * self.spatial_wavenumber_per_m.powi(2);
        if !phase.is_finite() || !stiffness_n.is_finite() || !load_scale.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let displacement_m = self.displacement_amplitude_m * phase.sin();
        let axial_strain =
            self.displacement_amplitude_m * self.spatial_wavenumber_per_m * phase.cos();
        let point = ElasticityPoint1d {
            x_m,
            displacement_m,
            axial_strain,
            axial_stress_pa: self.youngs_modulus_pa * axial_strain,
            axial_force_n: stiffness_n * axial_strain,
            distributed_load_n_per_m: load_scale * displacement_m,
        };
        if !point.all_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample the reference and exact equilibrium terms on a uniform grid.
    pub fn try_sample(
        self,
        x_min_m: f64,
        x_max_m: f64,
        num_points: usize,
    ) -> Result<ElasticitySample1d, ElasticityManufacturedError> {
        if num_points < 2 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "num_points",
                requirement: "at least 2",
            });
        }
        if !x_min_m.is_finite() || !x_max_m.is_finite() || x_max_m <= x_min_m {
            return Err(ElasticityManufacturedError::InvalidInterval {
                x_min: x_min_m,
                x_max: x_max_m,
            });
        }
        let dx = (x_max_m - x_min_m) / (num_points - 1) as f64;
        if !dx.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let mut sample = ElasticitySample1d {
            positions_m: Vec::with_capacity(num_points),
            displacements_m: Vec::with_capacity(num_points),
            axial_strains: Vec::with_capacity(num_points),
            axial_stresses_pa: Vec::with_capacity(num_points),
            axial_forces_n: Vec::with_capacity(num_points),
            distributed_loads_n_per_m: Vec::with_capacity(num_points),
        };
        for index in 0..num_points {
            let point = self.try_evaluate(x_min_m + index as f64 * dx)?;
            sample.positions_m.push(point.x_m);
            sample.displacements_m.push(point.displacement_m);
            sample.axial_strains.push(point.axial_strain);
            sample.axial_stresses_pa.push(point.axial_stress_pa);
            sample.axial_forces_n.push(point.axial_force_n);
            sample
                .distributed_loads_n_per_m
                .push(point.distributed_load_n_per_m);
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

    fn reference() -> ManufacturedElasticity1d {
        ManufacturedElasticity1d::try_new(70.0e9, 2.5e-4, 2.3, 1.2e-3, -0.15).unwrap()
    }

    #[test]
    fn exact_terms_close_axial_equilibrium() {
        let reference = reference();
        let x = 0.45;
        let h = 1e-5;
        let point = reference.try_evaluate(x).unwrap();
        let left = reference.try_evaluate(x - h).unwrap().axial_force_n;
        let right = reference.try_evaluate(x + h).unwrap().axial_force_n;
        assert_abs_diff_eq!(
            -(right - left) / (2.0 * h),
            point.distributed_load_n_per_m,
            epsilon = 2e-4
        );
        assert_abs_diff_eq!(
            point.axial_stress_pa * reference.area_m2(),
            point.axial_force_n,
            epsilon = 1e-9
        );
    }

    #[test]
    fn dirichlet_endpoints_and_strain_match_exact_formula() {
        let length = 2.0;
        let reference =
            ManufacturedElasticity1d::try_new(210.0e9, 1.0e-4, PI / length, 2.0e-3, 0.0).unwrap();
        let sample = reference.try_sample(0.0, length, 9).unwrap();
        assert_abs_diff_eq!(sample.displacements_m[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(
            sample.displacements_m.last().copied().unwrap(),
            0.0,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(
            sample.axial_strains[0],
            2.0e-3 * PI / length,
            epsilon = 1e-15
        );
    }

    #[test]
    fn displacement_and_strain_match_centered_difference() {
        let reference = reference();
        let x = 0.45;
        let h = 1e-5;
        let point = reference.try_evaluate(x).unwrap();
        let left = reference.try_evaluate(x - h).unwrap().displacement_m;
        let right = reference.try_evaluate(x + h).unwrap().displacement_m;
        assert_abs_diff_eq!(
            (right - left) / (2.0 * h),
            point.axial_strain,
            epsilon = 1e-12
        );
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_samples() {
        assert!(matches!(
            ManufacturedElasticity1d::try_new(0.0, 1.0, 1.0, 1.0, 0.0),
            Err(ElasticityManufacturedError::InvalidParameter {
                field: "youngs_modulus_pa",
                ..
            })
        ));
        let reference = reference();
        assert!(matches!(
            reference.try_evaluate(f64::NAN),
            Err(ElasticityManufacturedError::InvalidParameter { field: "x_m", .. })
        ));
        assert!(matches!(
            reference.try_sample(1.0, 0.0, 4),
            Err(ElasticityManufacturedError::InvalidInterval { .. })
        ));
        assert!(matches!(
            reference.try_sample(0.0, 1.0, 1),
            Err(ElasticityManufacturedError::InvalidParameter {
                field: "num_points",
                ..
            })
        ));
    }
}
