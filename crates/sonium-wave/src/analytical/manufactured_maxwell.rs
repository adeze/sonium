//! Manufactured one-dimensional transverse-electromagnetic Maxwell references.
//!
//! The reference uses a one-dimensional TE pair, with electric field `E_y`
//! and magnetic field `H_z`, in a homogeneous medium:
//!
//! ```text
//! ε ∂E_y/∂t + ∂H_z/∂x = -J_y,
//! μ ∂H_z/∂t + ∂E_y/∂x = -M_z.
//! ```
//!
//! `J_y` and `M_z` are manufactured electric and magnetic current densities.
//! The exact fields are `E_y = E₀ sin(q(x-x₀)) cos(ωt)` and
//! `H_z = -H₀ cos(q(x-x₀)) sin(ωt)`.  Choosing
//! `q² = ε μ ω²` and `H₀ = εωE₀/q` makes both source terms zero.  This is a
//! reference-only checked field/source contract, not a numerical Maxwell solve.

use std::fmt;

/// Invalid input to a manufactured one-dimensional Maxwell reference.
#[derive(Debug, Clone, PartialEq)]
pub enum MaxwellManufacturedError {
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
    /// An evaluated phase, derivative, or source term is not representable.
    EvaluationOverflow,
}

impl fmt::Display for MaxwellManufacturedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => write!(
                f,
                "manufactured Maxwell parameter {field} must be {requirement}"
            ),
            Self::InvalidInterval { x_min, x_max } => write!(
                f,
                "manufactured Maxwell interval must be finite and strictly increasing, got [{x_min}, {x_max}]"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "manufactured Maxwell evaluation cannot be represented in binary64"
            ),
        }
    }
}

impl std::error::Error for MaxwellManufacturedError {}

/// Exact fields, derivatives, and source terms at one TE-Maxwell point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxwellPoint1d {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// Evaluation time in seconds.
    pub time_s: f64,
    /// Exact transverse electric field `E_y` in V/m.
    pub electric_field_v_per_m: f64,
    /// Exact transverse magnetic field `H_z` in A/m.
    pub magnetic_field_a_per_m: f64,
    /// Exact `∂E_y/∂t` in V/(m·s).
    pub electric_time_derivative_v_per_m_s: f64,
    /// Exact `∂H_z/∂t` in A/(m·s).
    pub magnetic_time_derivative_a_per_m_s: f64,
    /// Exact `∂E_y/∂x` in V/m².
    pub electric_spatial_derivative_v_per_m2: f64,
    /// Exact `∂H_z/∂x` in A/m².
    pub magnetic_spatial_derivative_a_per_m2: f64,
    /// Exact manufactured electric-current density `J_y` in A/m².
    pub electric_current_density_a_per_m2: f64,
    /// Exact manufactured magnetic-current density `M_z` in V/m².
    pub magnetic_current_density_v_per_m2: f64,
}

impl MaxwellPoint1d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.time_s,
            self.electric_field_v_per_m,
            self.magnetic_field_a_per_m,
            self.electric_time_derivative_v_per_m_s,
            self.magnetic_time_derivative_a_per_m_s,
            self.electric_spatial_derivative_v_per_m2,
            self.magnetic_spatial_derivative_a_per_m2,
            self.electric_current_density_a_per_m2,
            self.magnetic_current_density_v_per_m2,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// Uniform exact TE-Maxwell samples at one time.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxwellSample1d {
    /// Sample coordinates in metres.
    pub positions_m: Vec<f64>,
    /// Exact transverse electric field `E_y` in V/m.
    pub electric_fields_v_per_m: Vec<f64>,
    /// Exact transverse magnetic field `H_z` in A/m.
    pub magnetic_fields_a_per_m: Vec<f64>,
    /// Exact `∂E_y/∂t` in V/(m·s).
    pub electric_time_derivatives_v_per_m_s: Vec<f64>,
    /// Exact `∂H_z/∂t` in A/(m·s).
    pub magnetic_time_derivatives_a_per_m_s: Vec<f64>,
    /// Exact `∂E_y/∂x` in V/m².
    pub electric_spatial_derivatives_v_per_m2: Vec<f64>,
    /// Exact `∂H_z/∂x` in A/m².
    pub magnetic_spatial_derivatives_a_per_m2: Vec<f64>,
    /// Exact manufactured `J_y` in A/m².
    pub electric_current_densities_a_per_m2: Vec<f64>,
    /// Exact manufactured `M_z` in V/m².
    pub magnetic_current_densities_v_per_m2: Vec<f64>,
}

impl MaxwellSample1d {
    /// Verify a finite, consistently-sized exact field set.
    pub fn validate(&self) -> Result<(), MaxwellManufacturedError> {
        let count = self.positions_m.len();
        if count < 2 {
            return Err(MaxwellManufacturedError::InvalidParameter {
                field: "positions_m",
                requirement: "at least 2 samples",
            });
        }
        if [
            self.electric_fields_v_per_m.len(),
            self.magnetic_fields_a_per_m.len(),
            self.electric_time_derivatives_v_per_m_s.len(),
            self.magnetic_time_derivatives_a_per_m_s.len(),
            self.electric_spatial_derivatives_v_per_m2.len(),
            self.magnetic_spatial_derivatives_a_per_m2.len(),
            self.electric_current_densities_a_per_m2.len(),
            self.magnetic_current_densities_v_per_m2.len(),
        ]
        .iter()
        .any(|&length| length != count)
        {
            return Err(MaxwellManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the same length as positions_m",
            });
        }
        if self
            .positions_m
            .iter()
            .chain(self.electric_fields_v_per_m.iter())
            .chain(self.magnetic_fields_a_per_m.iter())
            .chain(self.electric_time_derivatives_v_per_m_s.iter())
            .chain(self.magnetic_time_derivatives_a_per_m_s.iter())
            .chain(self.electric_spatial_derivatives_v_per_m2.iter())
            .chain(self.magnetic_spatial_derivatives_a_per_m2.iter())
            .chain(self.electric_current_densities_a_per_m2.iter())
            .chain(self.magnetic_current_densities_v_per_m2.iter())
            .any(|value| !value.is_finite())
        {
            return Err(MaxwellManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A smooth manufactured solution for one-dimensional TE Maxwell equations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedMaxwell1d {
    permittivity_f_per_m: f64,
    permeability_h_per_m: f64,
    angular_frequency_rad_per_s: f64,
    spatial_wavenumber_per_m: f64,
    electric_amplitude_v_per_m: f64,
    magnetic_amplitude_a_per_m: f64,
    origin_m: f64,
}

impl ManufacturedMaxwell1d {
    /// Construct a checked homogeneous TE-Maxwell manufactured reference.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        permittivity_f_per_m: f64,
        permeability_h_per_m: f64,
        angular_frequency_rad_per_s: f64,
        spatial_wavenumber_per_m: f64,
        electric_amplitude_v_per_m: f64,
        magnetic_amplitude_a_per_m: f64,
        origin_m: f64,
    ) -> Result<Self, MaxwellManufacturedError> {
        for (field, value, requirement) in [
            (
                "permittivity_f_per_m",
                permittivity_f_per_m,
                "finite and positive",
            ),
            (
                "permeability_h_per_m",
                permeability_h_per_m,
                "finite and positive",
            ),
            (
                "angular_frequency_rad_per_s",
                angular_frequency_rad_per_s,
                "finite and non-negative",
            ),
            (
                "spatial_wavenumber_per_m",
                spatial_wavenumber_per_m,
                "finite and positive",
            ),
            (
                "electric_amplitude_v_per_m",
                electric_amplitude_v_per_m,
                "finite",
            ),
            (
                "magnetic_amplitude_a_per_m",
                magnetic_amplitude_a_per_m,
                "finite",
            ),
            ("origin_m", origin_m, "finite"),
        ] {
            let valid = match field {
                "permittivity_f_per_m" | "permeability_h_per_m" | "spatial_wavenumber_per_m" => {
                    value.is_finite() && value > 0.0
                }
                "angular_frequency_rad_per_s" => value.is_finite() && value >= 0.0,
                _ => value.is_finite(),
            };
            if !valid {
                return Err(MaxwellManufacturedError::InvalidParameter { field, requirement });
            }
        }
        if ![
            permittivity_f_per_m * electric_amplitude_v_per_m * angular_frequency_rad_per_s,
            magnetic_amplitude_a_per_m * spatial_wavenumber_per_m,
            permeability_h_per_m * magnetic_amplitude_a_per_m * angular_frequency_rad_per_s,
            electric_amplitude_v_per_m * spatial_wavenumber_per_m,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return Err(MaxwellManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            permittivity_f_per_m,
            permeability_h_per_m,
            angular_frequency_rad_per_s,
            spatial_wavenumber_per_m,
            electric_amplitude_v_per_m,
            magnetic_amplitude_a_per_m,
            origin_m,
        })
    }

    /// Permittivity `ε` in F/m.
    pub fn permittivity_f_per_m(self) -> f64 {
        self.permittivity_f_per_m
    }

    /// Permeability `μ` in H/m.
    pub fn permeability_h_per_m(self) -> f64 {
        self.permeability_h_per_m
    }

    /// Angular frequency `ω` in rad/s.
    pub fn angular_frequency_rad_per_s(self) -> f64 {
        self.angular_frequency_rad_per_s
    }

    /// Spatial wavenumber `q` in 1/m.
    pub fn spatial_wavenumber_per_m(self) -> f64 {
        self.spatial_wavenumber_per_m
    }

    /// Evaluate exact fields, derivatives, and source terms at `(x, t)`.
    pub fn try_evaluate(
        self,
        x_m: f64,
        time_s: f64,
    ) -> Result<MaxwellPoint1d, MaxwellManufacturedError> {
        if !x_m.is_finite() || !time_s.is_finite() {
            return Err(MaxwellManufacturedError::InvalidParameter {
                field: if !x_m.is_finite() { "x_m" } else { "time_s" },
                requirement: "finite",
            });
        }
        let space_phase = self.spatial_wavenumber_per_m * (x_m - self.origin_m);
        let time_phase = self.angular_frequency_rad_per_s * time_s;
        if !space_phase.is_finite() || !time_phase.is_finite() {
            return Err(MaxwellManufacturedError::EvaluationOverflow);
        }
        let (sin_space, cos_space) = space_phase.sin_cos();
        let (sin_time, cos_time) = time_phase.sin_cos();
        let electric_field_v_per_m = self.electric_amplitude_v_per_m * sin_space * cos_time;
        let magnetic_field_a_per_m = -self.magnetic_amplitude_a_per_m * cos_space * sin_time;
        let electric_time_derivative_v_per_m_s = -self.electric_amplitude_v_per_m
            * self.angular_frequency_rad_per_s
            * sin_space
            * sin_time;
        let magnetic_time_derivative_a_per_m_s = -self.magnetic_amplitude_a_per_m
            * self.angular_frequency_rad_per_s
            * cos_space
            * cos_time;
        let electric_spatial_derivative_v_per_m2 =
            self.electric_amplitude_v_per_m * self.spatial_wavenumber_per_m * cos_space * cos_time;
        let magnetic_spatial_derivative_a_per_m2 =
            self.magnetic_amplitude_a_per_m * self.spatial_wavenumber_per_m * sin_space * sin_time;
        let point = MaxwellPoint1d {
            x_m,
            time_s,
            electric_field_v_per_m,
            magnetic_field_a_per_m,
            electric_time_derivative_v_per_m_s,
            magnetic_time_derivative_a_per_m_s,
            electric_spatial_derivative_v_per_m2,
            magnetic_spatial_derivative_a_per_m2,
            electric_current_density_a_per_m2: -(self.permittivity_f_per_m
                * electric_time_derivative_v_per_m_s
                + magnetic_spatial_derivative_a_per_m2),
            magnetic_current_density_v_per_m2: -(self.permeability_h_per_m
                * magnetic_time_derivative_a_per_m_s
                + electric_spatial_derivative_v_per_m2),
        };
        if !point.all_finite() {
            return Err(MaxwellManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample exact fields and sources on a uniform spatial grid at one time.
    pub fn try_sample(
        self,
        x_min_m: f64,
        x_max_m: f64,
        num_points: usize,
        time_s: f64,
    ) -> Result<MaxwellSample1d, MaxwellManufacturedError> {
        if num_points < 2 {
            return Err(MaxwellManufacturedError::InvalidParameter {
                field: "num_points",
                requirement: "at least 2",
            });
        }
        if !x_min_m.is_finite() || !x_max_m.is_finite() || x_max_m <= x_min_m {
            return Err(MaxwellManufacturedError::InvalidInterval {
                x_min: x_min_m,
                x_max: x_max_m,
            });
        }
        let dx = (x_max_m - x_min_m) / (num_points - 1) as f64;
        if !dx.is_finite() {
            return Err(MaxwellManufacturedError::EvaluationOverflow);
        }
        let mut sample = MaxwellSample1d {
            positions_m: Vec::with_capacity(num_points),
            electric_fields_v_per_m: Vec::with_capacity(num_points),
            magnetic_fields_a_per_m: Vec::with_capacity(num_points),
            electric_time_derivatives_v_per_m_s: Vec::with_capacity(num_points),
            magnetic_time_derivatives_a_per_m_s: Vec::with_capacity(num_points),
            electric_spatial_derivatives_v_per_m2: Vec::with_capacity(num_points),
            magnetic_spatial_derivatives_a_per_m2: Vec::with_capacity(num_points),
            electric_current_densities_a_per_m2: Vec::with_capacity(num_points),
            magnetic_current_densities_v_per_m2: Vec::with_capacity(num_points),
        };
        for index in 0..num_points {
            let point = self.try_evaluate(x_min_m + index as f64 * dx, time_s)?;
            sample.positions_m.push(point.x_m);
            sample
                .electric_fields_v_per_m
                .push(point.electric_field_v_per_m);
            sample
                .magnetic_fields_a_per_m
                .push(point.magnetic_field_a_per_m);
            sample
                .electric_time_derivatives_v_per_m_s
                .push(point.electric_time_derivative_v_per_m_s);
            sample
                .magnetic_time_derivatives_a_per_m_s
                .push(point.magnetic_time_derivative_a_per_m_s);
            sample
                .electric_spatial_derivatives_v_per_m2
                .push(point.electric_spatial_derivative_v_per_m2);
            sample
                .magnetic_spatial_derivatives_a_per_m2
                .push(point.magnetic_spatial_derivative_a_per_m2);
            sample
                .electric_current_densities_a_per_m2
                .push(point.electric_current_density_a_per_m2);
            sample
                .magnetic_current_densities_v_per_m2
                .push(point.magnetic_current_density_v_per_m2);
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

    fn reference() -> ManufacturedMaxwell1d {
        ManufacturedMaxwell1d::try_new(2.0, 3.0, 1.7, 1.3, 2.4, 0.9, -0.2).unwrap()
    }

    #[test]
    fn exact_terms_close_both_maxwell_equations() {
        let reference = reference();
        let sample = reference.try_sample(-1.0, 1.0, 19, 0.37).unwrap();
        for index in 0..sample.positions_m.len() {
            assert_abs_diff_eq!(
                reference.permittivity_f_per_m()
                    * sample.electric_time_derivatives_v_per_m_s[index]
                    + sample.magnetic_spatial_derivatives_a_per_m2[index]
                    + sample.electric_current_densities_a_per_m2[index],
                0.0,
                epsilon = 1e-13
            );
            assert_abs_diff_eq!(
                reference.permeability_h_per_m()
                    * sample.magnetic_time_derivatives_a_per_m_s[index]
                    + sample.electric_spatial_derivatives_v_per_m2[index]
                    + sample.magnetic_current_densities_v_per_m2[index],
                0.0,
                epsilon = 1e-13
            );
        }
    }

    #[test]
    fn matched_mode_is_source_free_and_respects_pec_endpoints() {
        let length_m = 2.0;
        let permittivity = 2.0_f64;
        let permeability = 3.0_f64;
        let q = PI / length_m;
        let angular_frequency = q / (permittivity * permeability).sqrt();
        let electric_amplitude = 2.3;
        let magnetic_amplitude = permittivity * angular_frequency * electric_amplitude / q;
        let reference = ManufacturedMaxwell1d::try_new(
            permittivity,
            permeability,
            angular_frequency,
            q,
            electric_amplitude,
            magnetic_amplitude,
            0.0,
        )
        .unwrap();
        let time_s = PI / (5.0 * angular_frequency);
        let sample = reference.try_sample(0.0, length_m, 17, time_s).unwrap();
        assert_abs_diff_eq!(sample.electric_fields_v_per_m[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(
            *sample.electric_fields_v_per_m.last().unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert!(
            sample
                .electric_current_densities_a_per_m2
                .iter()
                .all(|value| value.abs() < 1e-13)
        );
        assert!(
            sample
                .magnetic_current_densities_v_per_m2
                .iter()
                .all(|value| value.abs() < 1e-13)
        );
    }

    #[test]
    fn derivatives_agree_with_centered_finite_differences() {
        let reference = reference();
        let x = 0.45;
        let time = 0.31;
        let h_x = 1e-5;
        let h_t = 1e-5;
        let point = reference.try_evaluate(x, time).unwrap();
        let x_left = reference.try_evaluate(x - h_x, time).unwrap();
        let x_right = reference.try_evaluate(x + h_x, time).unwrap();
        let t_left = reference.try_evaluate(x, time - h_t).unwrap();
        let t_right = reference.try_evaluate(x, time + h_t).unwrap();
        assert_abs_diff_eq!(
            (x_right.electric_field_v_per_m - x_left.electric_field_v_per_m) / (2.0 * h_x),
            point.electric_spatial_derivative_v_per_m2,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            (x_right.magnetic_field_a_per_m - x_left.magnetic_field_a_per_m) / (2.0 * h_x),
            point.magnetic_spatial_derivative_a_per_m2,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            (t_right.electric_field_v_per_m - t_left.electric_field_v_per_m) / (2.0 * h_t),
            point.electric_time_derivative_v_per_m_s,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            (t_right.magnetic_field_a_per_m - t_left.magnetic_field_a_per_m) / (2.0 * h_t),
            point.magnetic_time_derivative_a_per_m_s,
            epsilon = 1e-9
        );
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_samples() {
        assert!(matches!(
            ManufacturedMaxwell1d::try_new(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
            Err(MaxwellManufacturedError::InvalidParameter {
                field: "permittivity_f_per_m",
                ..
            })
        ));
        assert!(matches!(
            ManufacturedMaxwell1d::try_new(1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0),
            Err(MaxwellManufacturedError::InvalidParameter {
                field: "angular_frequency_rad_per_s",
                ..
            })
        ));
        let reference = reference();
        assert!(matches!(
            reference.try_evaluate(f64::NAN, 0.0),
            Err(MaxwellManufacturedError::InvalidParameter { field: "x_m", .. })
        ));
        assert!(matches!(
            reference.try_sample(1.0, 0.0, 4, 0.0),
            Err(MaxwellManufacturedError::InvalidInterval { .. })
        ));
        assert!(matches!(
            reference.try_sample(0.0, 1.0, 1, 0.0),
            Err(MaxwellManufacturedError::InvalidParameter {
                field: "num_points",
                ..
            })
        ));
    }
}
