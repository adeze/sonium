//! Exact two-dimensional incompressible Taylor-Green vortex reference.
//!
//! The reference satisfies the constant-density incompressible
//! Navier-Stokes momentum equation
//! `∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u = 0` and `∇·u = 0` exactly. It provides
//! a checked analytical field/residual contract for consuming solvers; it is
//! not itself a CFD solve.

use std::fmt;

/// Invalid input to a manufactured incompressible-flow reference.
#[derive(Debug, Clone, PartialEq)]
pub enum IncompressibleFlowManufacturedError {
    /// A model parameter is non-finite or outside its documented domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// An exact term cannot be represented in binary64.
    EvaluationOverflow,
}

impl fmt::Display for IncompressibleFlowManufacturedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => write!(
                f,
                "manufactured incompressible-flow parameter {field} must be {requirement}"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "manufactured incompressible-flow evaluation cannot be represented in binary64"
            ),
        }
    }
}

impl std::error::Error for IncompressibleFlowManufacturedError {}

/// Exact Taylor-Green velocity, pressure, derivative, and residual terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleFlowPoint2d {
    /// Evaluation coordinate in metres.
    pub position_m: [f64; 2],
    /// Evaluation time in seconds.
    pub time_s: f64,
    /// Velocity `[u, v]` in m/s.
    pub velocity_m_per_s: [f64; 2],
    /// Kinematic pressure `p` in Pa.
    pub pressure_pa: f64,
    /// Time derivative `∂uᵢ/∂t` in m/s².
    pub velocity_time_derivative_m_per_s2: [f64; 2],
    /// Velocity gradient `∂uᵢ/∂xⱼ` in 1/s.
    pub velocity_gradient_per_s: [[f64; 2]; 2],
    /// Pressure gradient in Pa/m.
    pub pressure_gradient_pa_per_m: [f64; 2],
    /// Velocity Laplacian in 1/(m·s).
    pub velocity_laplacian_per_m_s: [f64; 2],
    /// Convective acceleration `(u·∇)u` in m/s².
    pub convective_acceleration_m_per_s2: [f64; 2],
    /// Incompressible momentum residual in m/s².
    pub momentum_residual_m_per_s2: [f64; 2],
    /// Velocity divergence in 1/s.
    pub divergence_per_s: f64,
}

impl IncompressibleFlowPoint2d {
    fn all_finite(self) -> bool {
        self.position_m
            .into_iter()
            .chain([self.time_s, self.pressure_pa, self.divergence_per_s])
            .chain(self.velocity_m_per_s)
            .chain(self.velocity_time_derivative_m_per_s2)
            .chain(self.velocity_gradient_per_s.into_iter().flatten())
            .chain(self.pressure_gradient_pa_per_m)
            .chain(self.velocity_laplacian_per_m_s)
            .chain(self.convective_acceleration_m_per_s2)
            .chain(self.momentum_residual_m_per_s2)
            .all(f64::is_finite)
    }
}

/// Checked exact two-dimensional Taylor-Green vortex model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedIncompressibleFlow2d {
    density_kg_per_m3: f64,
    kinematic_viscosity_m2_per_s: f64,
    wavenumber_per_m: f64,
    velocity_amplitude_m_per_s: f64,
    origin_m: [f64; 2],
}

impl ManufacturedIncompressibleFlow2d {
    /// Construct a checked constant-density Taylor-Green vortex reference.
    pub fn try_new(
        density_kg_per_m3: f64,
        kinematic_viscosity_m2_per_s: f64,
        wavenumber_per_m: f64,
        velocity_amplitude_m_per_s: f64,
        origin_m: [f64; 2],
    ) -> Result<Self, IncompressibleFlowManufacturedError> {
        for (field, value, positive) in [
            ("density_kg_per_m3", density_kg_per_m3, true),
            (
                "kinematic_viscosity_m2_per_s",
                kinematic_viscosity_m2_per_s,
                true,
            ),
            ("wavenumber_per_m", wavenumber_per_m, true),
            (
                "velocity_amplitude_m_per_s",
                velocity_amplitude_m_per_s,
                false,
            ),
            ("origin_m[0]", origin_m[0], false),
            ("origin_m[1]", origin_m[1], false),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) {
                return Err(IncompressibleFlowManufacturedError::InvalidParameter {
                    field,
                    requirement: if positive {
                        "finite and positive"
                    } else {
                        "finite"
                    },
                });
            }
        }
        if ![
            2.0 * kinematic_viscosity_m2_per_s * wavenumber_per_m.powi(2),
            density_kg_per_m3 * velocity_amplitude_m_per_s.powi(2),
            velocity_amplitude_m_per_s * wavenumber_per_m,
        ]
        .iter()
        .all(|term| term.is_finite())
        {
            return Err(IncompressibleFlowManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            density_kg_per_m3,
            kinematic_viscosity_m2_per_s,
            wavenumber_per_m,
            velocity_amplitude_m_per_s,
            origin_m,
        })
    }

    /// Constant density in kg/m³.
    pub fn density_kg_per_m3(self) -> f64 {
        self.density_kg_per_m3
    }

    /// Kinematic viscosity in m²/s.
    pub fn kinematic_viscosity_m2_per_s(self) -> f64 {
        self.kinematic_viscosity_m2_per_s
    }

    /// Spatial wavenumber in 1/m.
    pub fn wavenumber_per_m(self) -> f64 {
        self.wavenumber_per_m
    }

    /// Evaluate exact Taylor-Green fields and every momentum term at `(x, y, t)`.
    pub fn try_evaluate(
        self,
        position_m: [f64; 2],
        time_s: f64,
    ) -> Result<IncompressibleFlowPoint2d, IncompressibleFlowManufacturedError> {
        if !position_m.into_iter().all(f64::is_finite) || !time_s.is_finite() {
            return Err(IncompressibleFlowManufacturedError::InvalidParameter {
                field: "position_m/time_s",
                requirement: "finite",
            });
        }
        let x_phase = self.wavenumber_per_m * (position_m[0] - self.origin_m[0]);
        let y_phase = self.wavenumber_per_m * (position_m[1] - self.origin_m[1]);
        let decay_exponent =
            -2.0 * self.kinematic_viscosity_m2_per_s * self.wavenumber_per_m.powi(2) * time_s;
        if !x_phase.is_finite() || !y_phase.is_finite() || !decay_exponent.is_finite() {
            return Err(IncompressibleFlowManufacturedError::EvaluationOverflow);
        }
        let decay = decay_exponent.exp();
        let decay_squared = decay * decay;
        let (sin_x, cos_x) = x_phase.sin_cos();
        let (sin_y, cos_y) = y_phase.sin_cos();
        let amplitude = self.velocity_amplitude_m_per_s;
        let wave = self.wavenumber_per_m;
        let velocity = [
            amplitude * sin_x * cos_y * decay,
            -amplitude * cos_x * sin_y * decay,
        ];
        let gradient = [
            [
                amplitude * wave * cos_x * cos_y * decay,
                -amplitude * wave * sin_x * sin_y * decay,
            ],
            [
                amplitude * wave * sin_x * sin_y * decay,
                -amplitude * wave * cos_x * cos_y * decay,
            ],
        ];
        let velocity_time_derivative = velocity
            .map(|component| -2.0 * self.kinematic_viscosity_m2_per_s * wave.powi(2) * component);
        let velocity_laplacian = velocity.map(|component| -2.0 * wave.powi(2) * component);
        let pressure_pa = self.density_kg_per_m3
            * amplitude.powi(2)
            * 0.25
            * ((2.0 * x_phase).cos() + (2.0 * y_phase).cos())
            * decay_squared;
        let pressure_gradient = [
            -self.density_kg_per_m3 * amplitude.powi(2) * wave * sin_x * cos_x * decay_squared,
            -self.density_kg_per_m3 * amplitude.powi(2) * wave * sin_y * cos_y * decay_squared,
        ];
        let convective_acceleration = [
            amplitude.powi(2) * wave * sin_x * cos_x * decay_squared,
            amplitude.powi(2) * wave * sin_y * cos_y * decay_squared,
        ];
        let momentum_residual = [0, 1].map(|component| {
            velocity_time_derivative[component]
                + convective_acceleration[component]
                + pressure_gradient[component] / self.density_kg_per_m3
                - self.kinematic_viscosity_m2_per_s * velocity_laplacian[component]
        });
        let point = IncompressibleFlowPoint2d {
            position_m,
            time_s,
            velocity_m_per_s: velocity,
            pressure_pa,
            velocity_time_derivative_m_per_s2: velocity_time_derivative,
            velocity_gradient_per_s: gradient,
            pressure_gradient_pa_per_m: pressure_gradient,
            velocity_laplacian_per_m_s: velocity_laplacian,
            convective_acceleration_m_per_s2: convective_acceleration,
            momentum_residual_m_per_s2: momentum_residual,
            divergence_per_s: gradient[0][0] + gradient[1][1],
        };
        if !point.all_finite() {
            return Err(IncompressibleFlowManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn reference() -> ManufacturedIncompressibleFlow2d {
        ManufacturedIncompressibleFlow2d::try_new(1.2, 0.03, 1.4, 2.0, [-0.2, 0.3]).unwrap()
    }

    #[test]
    fn exact_flow_closes_momentum_and_incompressibility() {
        let reference = reference();
        for position in [[-0.4, 0.1], [0.2, 0.6], [1.0, -0.7]] {
            let point = reference.try_evaluate(position, 0.37).unwrap();
            assert_abs_diff_eq!(point.momentum_residual_m_per_s2[0], 0.0, epsilon = 1e-13);
            assert_abs_diff_eq!(point.momentum_residual_m_per_s2[1], 0.0, epsilon = 1e-13);
            assert_abs_diff_eq!(point.divergence_per_s, 0.0, epsilon = 1e-13);
        }
    }

    #[test]
    fn derivatives_agree_with_centered_differences() {
        let reference = reference();
        let position = [0.45, -0.23];
        let time = 0.31;
        let h = 1e-5;
        let point = reference.try_evaluate(position, time).unwrap();
        for direction in 0..2 {
            let mut left_position = position;
            let mut right_position = position;
            left_position[direction] -= h;
            right_position[direction] += h;
            let left = reference.try_evaluate(left_position, time).unwrap();
            let right = reference.try_evaluate(right_position, time).unwrap();
            for component in 0..2 {
                assert_abs_diff_eq!(
                    (right.velocity_m_per_s[component] - left.velocity_m_per_s[component])
                        / (2.0 * h),
                    point.velocity_gradient_per_s[component][direction],
                    epsilon = 1e-9
                );
            }
        }
        let left = reference.try_evaluate(position, time - h).unwrap();
        let right = reference.try_evaluate(position, time + h).unwrap();
        for component in 0..2 {
            assert_abs_diff_eq!(
                (right.velocity_m_per_s[component] - left.velocity_m_per_s[component]) / (2.0 * h),
                point.velocity_time_derivative_m_per_s2[component],
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn checked_admission_rejects_invalid_models_and_coordinates() {
        assert!(matches!(
            ManufacturedIncompressibleFlow2d::try_new(0.0, 0.1, 1.0, 1.0, [0.0, 0.0]),
            Err(IncompressibleFlowManufacturedError::InvalidParameter {
                field: "density_kg_per_m3",
                ..
            })
        ));
        assert!(matches!(
            ManufacturedIncompressibleFlow2d::try_new(1.0, 0.1, 0.0, 1.0, [0.0, 0.0]),
            Err(IncompressibleFlowManufacturedError::InvalidParameter {
                field: "wavenumber_per_m",
                ..
            })
        ));
        assert!(matches!(
            reference().try_evaluate([f64::NAN, 0.0], 0.0),
            Err(IncompressibleFlowManufacturedError::InvalidParameter { .. })
        ));
        assert!(matches!(
            reference().try_evaluate([0.0, 0.0], f64::NAN),
            Err(IncompressibleFlowManufacturedError::InvalidParameter { .. })
        ));
    }
}
