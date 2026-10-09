//! Axisymmetric rigid-wall modes of a circular cylindrical acoustic cavity.
//!
//! The reference uses the `exp(+iωt)` convention in a homogeneous, lossless
//! fluid.  It represents only azimuthal order zero in a circular cavity of
//! radius `a` and length `L`; it does not model losses, radiation, mean flow,
//! non-circular sections, or non-axisymmetric modes.  The unit-peak pressure
//! mode is
//!
//! ```text
//! p(r, z) = J₀(χₘ r / a) cos(nπz / L),
//! ```
//!
//! where `χₘ` is the `m`th positive zero of `J₁`, so every wall has zero
//! normal pressure gradient.  This is a reusable `(r, z)` Helmholtz reference
//! for axisymmetric FEM/BEM convergence work, not itself a numerical solve.

use spec_math::Bessel;
use std::fmt;

/// Input or evaluation error for [`AxisymmetricRigidCavityMode`].
#[derive(Debug, Clone, PartialEq)]
pub enum AxisymmetricCavityError {
    InvalidParameter {
        field: &'static str,
        requirement: &'static str,
    },
    /// The constant-pressure `(m, n) = (0, 0)` mode has no acoustic frequency.
    ZeroFrequencyMode,
    EvaluationOutsideCavity {
        coordinate: &'static str,
        value: f64,
        upper_bound: f64,
    },
    RootConvergenceFailure {
        radial_mode: usize,
    },
    NonFiniteResult,
}

impl fmt::Display for AxisymmetricCavityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(
                    f,
                    "axisymmetric cavity parameter {field} must be {requirement}"
                )
            }
            Self::ZeroFrequencyMode => write!(
                f,
                "axisymmetric cavity mode (radial=0, axial=0) has zero acoustic frequency"
            ),
            Self::EvaluationOutsideCavity {
                coordinate,
                value,
                upper_bound,
            } => write!(
                f,
                "axisymmetric cavity {coordinate}={value} is outside [0, {upper_bound}]"
            ),
            Self::RootConvergenceFailure { radial_mode } => write!(
                f,
                "could not converge the J1 zero for radial mode {radial_mode}"
            ),
            Self::NonFiniteResult => write!(
                f,
                "axisymmetric cavity mode produced a non-finite binary64 result"
            ),
        }
    }
}

impl std::error::Error for AxisymmetricCavityError {}

/// Exact scalar fields at one `(r, z)` point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisymmetricCavitySample {
    /// Unit-peak acoustic pressure.
    pub pressure: f64,
    /// `∂p/∂r` in inverse metres.
    pub radial_gradient: f64,
    /// `∂p/∂z` in inverse metres.
    pub axial_gradient: f64,
    /// Axisymmetric Laplacian `∂²p/∂r² + r⁻¹∂p/∂r + ∂²p/∂z²`.
    pub laplacian: f64,
}

/// Checked unit-peak axisymmetric mode of a rigid circular cylindrical cavity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisymmetricRigidCavityMode {
    radius_m: f64,
    length_m: f64,
    sound_speed_m_per_s: f64,
    radial_mode: usize,
    axial_mode: usize,
    radial_root: f64,
    radial_wavenumber_rad_per_m: f64,
    axial_wavenumber_rad_per_m: f64,
    wavenumber_rad_per_m: f64,
}

impl AxisymmetricRigidCavityMode {
    /// Construct an `m=0` azimuthal mode. `radial_mode = 0` is uniform in
    /// radius; positive radial modes use the corresponding positive `J₁` root.
    /// At least one of the radial and axial mode numbers must be non-zero.
    pub fn try_new(
        radius_m: f64,
        length_m: f64,
        sound_speed_m_per_s: f64,
        radial_mode: usize,
        axial_mode: usize,
    ) -> Result<Self, AxisymmetricCavityError> {
        for (field, value) in [
            ("radius_m", radius_m),
            ("length_m", length_m),
            ("sound_speed_m_per_s", sound_speed_m_per_s),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(AxisymmetricCavityError::InvalidParameter {
                    field,
                    requirement: "finite and positive",
                });
            }
        }
        if radial_mode == 0 && axial_mode == 0 {
            return Err(AxisymmetricCavityError::ZeroFrequencyMode);
        }

        let radial_root = j1_zero(radial_mode)?;
        let radial_wavenumber_rad_per_m = radial_root / radius_m;
        let axial_wavenumber_rad_per_m = axial_mode as f64 * std::f64::consts::PI / length_m;
        let wavenumber_rad_per_m =
            (radial_wavenumber_rad_per_m.powi(2) + axial_wavenumber_rad_per_m.powi(2)).sqrt();
        if ![
            radial_root,
            radial_wavenumber_rad_per_m,
            axial_wavenumber_rad_per_m,
            wavenumber_rad_per_m,
            sound_speed_m_per_s * wavenumber_rad_per_m,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return Err(AxisymmetricCavityError::NonFiniteResult);
        }

        Ok(Self {
            radius_m,
            length_m,
            sound_speed_m_per_s,
            radial_mode,
            axial_mode,
            radial_root,
            radial_wavenumber_rad_per_m,
            axial_wavenumber_rad_per_m,
            wavenumber_rad_per_m,
        })
    }

    pub fn radial_mode(self) -> usize {
        self.radial_mode
    }

    pub fn axial_mode(self) -> usize {
        self.axial_mode
    }

    /// The `J₁` root `χₘ`; zero for the radially uniform mode.
    pub fn radial_root(self) -> f64 {
        self.radial_root
    }

    pub fn wavenumber_rad_per_m(self) -> f64 {
        self.wavenumber_rad_per_m
    }

    pub fn frequency_hz(self) -> f64 {
        self.sound_speed_m_per_s * self.wavenumber_rad_per_m / std::f64::consts::TAU
    }

    /// Evaluate the exact mode inside the closed cavity.
    pub fn try_evaluate(
        self,
        radius_m: f64,
        axial_position_m: f64,
    ) -> Result<AxisymmetricCavitySample, AxisymmetricCavityError> {
        for (coordinate, value, upper_bound) in [
            ("radius_m", radius_m, self.radius_m),
            ("axial_position_m", axial_position_m, self.length_m),
        ] {
            if !value.is_finite() || value < 0.0 || value > upper_bound {
                return Err(AxisymmetricCavityError::EvaluationOutsideCavity {
                    coordinate,
                    value,
                    upper_bound,
                });
            }
        }
        let radial_argument = self.radial_wavenumber_rad_per_m * radius_m;
        let axial_argument = self.axial_wavenumber_rad_per_m * axial_position_m;
        let radial = radial_argument.bessel_jv(0.0);
        let radial_derivative_basis = -radial_argument.bessel_jv(1.0);
        let (sin_z, cos_z) = axial_argument.sin_cos();
        let pressure = radial * cos_z;
        let radial_gradient = self.radial_wavenumber_rad_per_m * radial_derivative_basis * cos_z;
        let axial_gradient = -self.axial_wavenumber_rad_per_m * radial * sin_z;
        let laplacian = -self.wavenumber_rad_per_m.powi(2) * pressure;
        let sample = AxisymmetricCavitySample {
            pressure,
            radial_gradient,
            axial_gradient,
            laplacian,
        };
        if [
            sample.pressure,
            sample.radial_gradient,
            sample.axial_gradient,
            sample.laplacian,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            Ok(sample)
        } else {
            Err(AxisymmetricCavityError::NonFiniteResult)
        }
    }

    /// Evaluate `∇²p + k²p`, which is zero for this exact Helmholtz mode.
    pub fn try_helmholtz_residual(
        self,
        radius_m: f64,
        axial_position_m: f64,
    ) -> Result<f64, AxisymmetricCavityError> {
        let sample = self.try_evaluate(radius_m, axial_position_m)?;
        Ok(sample.laplacian + self.wavenumber_rad_per_m.powi(2) * sample.pressure)
    }
}

fn j1_zero(radial_mode: usize) -> Result<f64, AxisymmetricCavityError> {
    if radial_mode == 0 {
        return Ok(0.0);
    }
    // McMahon's asymptotic initial value for the nth J₁ zero, followed by
    // Newton iteration using J₁'(x) = J₀(x) - J₁(x)/x.
    let mut root = (radial_mode as f64 + 0.25) * std::f64::consts::PI;
    for _ in 0..32 {
        let j0 = root.bessel_jv(0.0);
        let j1 = root.bessel_jv(1.0);
        let derivative = j0 - j1 / root;
        if !j0.is_finite() || !j1.is_finite() || !derivative.is_finite() || derivative == 0.0 {
            return Err(AxisymmetricCavityError::RootConvergenceFailure { radial_mode });
        }
        let next = root - j1 / derivative;
        if !next.is_finite() || next <= 0.0 {
            return Err(AxisymmetricCavityError::RootConvergenceFailure { radial_mode });
        }
        if (next - root).abs() <= 4.0 * f64::EPSILON * next.abs().max(1.0) {
            return Ok(next);
        }
        root = next;
    }
    Err(AxisymmetricCavityError::RootConvergenceFailure { radial_mode })
}

/// Input or evaluation error for [`AzimuthalRigidCavityMode`].
#[derive(Debug, Clone, PartialEq)]
pub enum AzimuthalCavityError {
    InvalidParameter {
        field: &'static str,
        requirement: &'static str,
    },
    ZeroFrequencyMode,
    EvaluationOutsideCavity {
        coordinate: &'static str,
        value: f64,
        upper_bound: f64,
    },
    RootConvergenceFailure {
        azimuthal_order: usize,
        radial_mode: usize,
    },
    NonFiniteResult,
}

impl fmt::Display for AzimuthalCavityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(
                    f,
                    "azimuthal cavity parameter {field} must be {requirement}"
                )
            }
            Self::ZeroFrequencyMode => {
                write!(
                    f,
                    "the constant cylindrical-cavity mode has zero acoustic frequency"
                )
            }
            Self::EvaluationOutsideCavity {
                coordinate,
                value,
                upper_bound,
            } => write!(
                f,
                "azimuthal cavity {coordinate}={value} is outside [0, {upper_bound}]"
            ),
            Self::RootConvergenceFailure {
                azimuthal_order,
                radial_mode,
            } => write!(
                f,
                "could not converge the J_{azimuthal_order} derivative root for radial mode {radial_mode}"
            ),
            Self::NonFiniteResult => {
                write!(
                    f,
                    "azimuthal cavity mode produced a non-finite binary64 result"
                )
            }
        }
    }
}

impl std::error::Error for AzimuthalCavityError {}

/// Exact fields at one cylindrical `(r, θ, z)` point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AzimuthalCavitySample {
    /// Unit-peak acoustic pressure.
    pub pressure: f64,
    /// `∂p/∂r` in inverse metres.
    pub radial_gradient: f64,
    /// `∂p/∂θ` (the angle derivative; divide by `r` away from the axis for
    /// the physical azimuthal gradient component).
    pub azimuthal_angle_derivative: f64,
    /// `∂p/∂z` in inverse metres.
    pub axial_gradient: f64,
    /// Cylindrical-coordinate Helmholtz Laplacian.
    pub laplacian: f64,
}

/// A real cosine-phase azimuthal mode of a rigid circular cylindrical cavity.
///
/// With azimuthal order `ℓ`, radial mode `m`, and axial mode `n`, this is
/// `J_ℓ(χ_{ℓm} r/a) cos(ℓθ) cos(nπz/L)`.  The radial root is a zero of
/// `J_ℓ'`, which enforces a rigid radial wall.  For `ℓ = 0`, radial mode zero
/// is the radially uniform branch; for `ℓ > 0`, radial mode zero denotes the
/// first positive derivative root.  Sine-phase partners are rotations of this
/// cosine reference and therefore need no separate numerical contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AzimuthalRigidCavityMode {
    radius_m: f64,
    length_m: f64,
    sound_speed_m_per_s: f64,
    azimuthal_order: usize,
    radial_mode: usize,
    axial_mode: usize,
    radial_root: f64,
    radial_wavenumber_rad_per_m: f64,
    axial_wavenumber_rad_per_m: f64,
    wavenumber_rad_per_m: f64,
}

impl AzimuthalRigidCavityMode {
    /// Construct a checked non-axisymmetric-capable rigid cavity mode.
    /// Angles passed to [`Self::try_evaluate`] are in radians around the
    /// positive `z` axis. The model uses unit-peak pressure and `exp(+iωt)`.
    pub fn try_new(
        radius_m: f64,
        length_m: f64,
        sound_speed_m_per_s: f64,
        azimuthal_order: usize,
        radial_mode: usize,
        axial_mode: usize,
    ) -> Result<Self, AzimuthalCavityError> {
        for (field, value) in [
            ("radius_m", radius_m),
            ("length_m", length_m),
            ("sound_speed_m_per_s", sound_speed_m_per_s),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(AzimuthalCavityError::InvalidParameter {
                    field,
                    requirement: "finite and positive",
                });
            }
        }
        if azimuthal_order == 0 && radial_mode == 0 && axial_mode == 0 {
            return Err(AzimuthalCavityError::ZeroFrequencyMode);
        }
        let radial_root = bessel_derivative_root(azimuthal_order, radial_mode)?;
        let radial_wavenumber_rad_per_m = radial_root / radius_m;
        let axial_wavenumber_rad_per_m = axial_mode as f64 * std::f64::consts::PI / length_m;
        let wavenumber_rad_per_m =
            (radial_wavenumber_rad_per_m.powi(2) + axial_wavenumber_rad_per_m.powi(2)).sqrt();
        if [
            radial_root,
            radial_wavenumber_rad_per_m,
            axial_wavenumber_rad_per_m,
            wavenumber_rad_per_m,
            sound_speed_m_per_s * wavenumber_rad_per_m,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            Ok(Self {
                radius_m,
                length_m,
                sound_speed_m_per_s,
                azimuthal_order,
                radial_mode,
                axial_mode,
                radial_root,
                radial_wavenumber_rad_per_m,
                axial_wavenumber_rad_per_m,
                wavenumber_rad_per_m,
            })
        } else {
            Err(AzimuthalCavityError::NonFiniteResult)
        }
    }

    pub fn azimuthal_order(self) -> usize {
        self.azimuthal_order
    }

    pub fn radial_mode(self) -> usize {
        self.radial_mode
    }

    pub fn axial_mode(self) -> usize {
        self.axial_mode
    }

    /// The derivative root `χ_{ℓm}`; zero only for the axisymmetric uniform
    /// radial branch.
    pub fn radial_root(self) -> f64 {
        self.radial_root
    }

    pub fn wavenumber_rad_per_m(self) -> f64 {
        self.wavenumber_rad_per_m
    }

    pub fn frequency_hz(self) -> f64 {
        self.sound_speed_m_per_s * self.wavenumber_rad_per_m / std::f64::consts::TAU
    }

    /// Evaluate the exact cosine-phase mode inside the closed cylindrical
    /// cavity. `azimuth_rad` is periodic but must be finite.
    pub fn try_evaluate(
        self,
        radius_m: f64,
        azimuth_rad: f64,
        axial_position_m: f64,
    ) -> Result<AzimuthalCavitySample, AzimuthalCavityError> {
        for (coordinate, value, upper_bound) in [
            ("radius_m", radius_m, self.radius_m),
            ("axial_position_m", axial_position_m, self.length_m),
        ] {
            if !value.is_finite() || value < 0.0 || value > upper_bound {
                return Err(AzimuthalCavityError::EvaluationOutsideCavity {
                    coordinate,
                    value,
                    upper_bound,
                });
            }
        }
        if !azimuth_rad.is_finite() {
            return Err(AzimuthalCavityError::InvalidParameter {
                field: "azimuth_rad",
                requirement: "finite",
            });
        }
        let radial_argument = self.radial_wavenumber_rad_per_m * radius_m;
        let radial = bessel_j(self.azimuthal_order, radial_argument);
        let radial_derivative = self.radial_wavenumber_rad_per_m
            * bessel_j_derivative(self.azimuthal_order, radial_argument);
        let azimuth_argument = self.azimuthal_order as f64 * azimuth_rad;
        let (sin_theta, cos_theta) = azimuth_argument.sin_cos();
        let axial_argument = self.axial_wavenumber_rad_per_m * axial_position_m;
        let (sin_z, cos_z) = axial_argument.sin_cos();
        let pressure = radial * cos_theta * cos_z;
        let sample = AzimuthalCavitySample {
            pressure,
            radial_gradient: radial_derivative * cos_theta * cos_z,
            azimuthal_angle_derivative: -(self.azimuthal_order as f64) * radial * sin_theta * cos_z,
            axial_gradient: -self.axial_wavenumber_rad_per_m * radial * cos_theta * sin_z,
            laplacian: -self.wavenumber_rad_per_m.powi(2) * pressure,
        };
        if [
            sample.pressure,
            sample.radial_gradient,
            sample.azimuthal_angle_derivative,
            sample.axial_gradient,
            sample.laplacian,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            Ok(sample)
        } else {
            Err(AzimuthalCavityError::NonFiniteResult)
        }
    }

    /// Evaluate `∇²p + k²p`, zero for the exact mode.
    pub fn try_helmholtz_residual(
        self,
        radius_m: f64,
        azimuth_rad: f64,
        axial_position_m: f64,
    ) -> Result<f64, AzimuthalCavityError> {
        let sample = self.try_evaluate(radius_m, azimuth_rad, axial_position_m)?;
        Ok(sample.laplacian + self.wavenumber_rad_per_m.powi(2) * sample.pressure)
    }
}

fn bessel_j(order: usize, argument: f64) -> f64 {
    argument.bessel_jv(order as f64)
}

fn bessel_j_derivative(order: usize, argument: f64) -> f64 {
    if order == 0 {
        -bessel_j(1, argument)
    } else {
        0.5 * (bessel_j(order - 1, argument) - bessel_j(order + 1, argument))
    }
}

fn bessel_derivative_root(
    azimuthal_order: usize,
    radial_mode: usize,
) -> Result<f64, AzimuthalCavityError> {
    if azimuthal_order == 0 {
        return j1_zero(radial_mode).map_err(|_| AzimuthalCavityError::RootConvergenceFailure {
            azimuthal_order,
            radial_mode,
        });
    }

    // Positive J_l' roots are simple. A short fixed-step scan gives a robust
    // bracket without trusting a large-order asymptotic guess, then bisection
    // establishes a deterministic binary64 value. The first non-axisymmetric
    // radial mode is index zero by definition.
    let target_root = radial_mode + 1;
    let step = std::f64::consts::PI / 16.0;
    let mut left = f64::EPSILON;
    let mut left_value = bessel_j_derivative(azimuthal_order, left);
    let max_steps = (azimuthal_order + target_root + 8).saturating_mul(32);
    let mut found = 0usize;
    for _ in 0..max_steps {
        let right = left + step;
        let right_value = bessel_j_derivative(azimuthal_order, right);
        if !left_value.is_finite() || !right_value.is_finite() {
            break;
        }
        if left_value.signum() != right_value.signum() {
            found += 1;
            if found == target_root {
                let mut lo = left;
                let mut hi = right;
                let mut lo_value = left_value;
                for _ in 0..80 {
                    let mid = 0.5 * (lo + hi);
                    let mid_value = bessel_j_derivative(azimuthal_order, mid);
                    if !mid_value.is_finite() {
                        break;
                    }
                    if mid_value == 0.0 {
                        return Ok(mid);
                    }
                    if lo_value.signum() == mid_value.signum() {
                        lo = mid;
                        lo_value = mid_value;
                    } else {
                        hi = mid;
                    }
                }
                return Ok(0.5 * (lo + hi));
            }
        }
        left = right;
        left_value = right_value;
    }
    Err(AzimuthalCavityError::RootConvergenceFailure {
        azimuthal_order,
        radial_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn first_radial_root_and_frequency_match_the_closed_cavity_mode() {
        let mode = AxisymmetricRigidCavityMode::try_new(0.2, 0.8, 343.0, 1, 2).unwrap();
        assert_abs_diff_eq!(
            mode.radial_root(),
            3.831_705_970_207_512_5,
            epsilon = 2.0e-14
        );
        let expected_k = ((3.831_705_970_207_512_5_f64 / 0.2).powi(2)
            + (2.0 * std::f64::consts::PI / 0.8).powi(2))
        .sqrt();
        assert_abs_diff_eq!(mode.wavenumber_rad_per_m(), expected_k, epsilon = 2.0e-13);
        assert_abs_diff_eq!(
            mode.frequency_hz(),
            343.0 * expected_k / std::f64::consts::TAU,
            epsilon = 2.0e-11
        );
    }

    #[test]
    fn rigid_walls_and_helmholtz_identity_hold() {
        let mode = AxisymmetricRigidCavityMode::try_new(0.2, 0.8, 343.0, 1, 2).unwrap();
        for z in [0.0, 0.23, 0.8] {
            assert!(
                mode.try_evaluate(0.2, z).unwrap().radial_gradient.abs() < 5.0e-14,
                "radial wall derivative at z={z}"
            );
        }
        for r in [0.0, 0.07, 0.2] {
            for z in [0.0, 0.8] {
                assert!(
                    mode.try_evaluate(r, z).unwrap().axial_gradient.abs() < 5.0e-14,
                    "end-wall derivative at r={r}, z={z}"
                );
            }
        }
        for (r, z) in [(0.0, 0.13), (0.06, 0.37), (0.19, 0.71)] {
            assert_abs_diff_eq!(
                mode.try_helmholtz_residual(r, z).unwrap(),
                0.0,
                epsilon = 2e-12
            );
        }
    }

    #[test]
    fn checked_admission_rejects_static_and_outside_modes() {
        assert!(matches!(
            AxisymmetricRigidCavityMode::try_new(0.2, 0.8, 343.0, 0, 0),
            Err(AxisymmetricCavityError::ZeroFrequencyMode)
        ));
        assert!(AxisymmetricRigidCavityMode::try_new(0.0, 0.8, 343.0, 1, 0).is_err());
        let mode = AxisymmetricRigidCavityMode::try_new(0.2, 0.8, 343.0, 1, 0).unwrap();
        assert!(matches!(
            mode.try_evaluate(0.200_000_1, 0.2),
            Err(AxisymmetricCavityError::EvaluationOutsideCavity {
                coordinate: "radius_m",
                ..
            })
        ));
        assert!(matches!(
            mode.try_evaluate(0.1, f64::NAN),
            Err(AxisymmetricCavityError::EvaluationOutsideCavity {
                coordinate: "axial_position_m",
                ..
            })
        ));
    }

    #[test]
    fn first_azimuthal_derivative_root_and_modal_frequency_are_pinned() {
        let mode = AzimuthalRigidCavityMode::try_new(0.2, 0.8, 343.0, 1, 0, 2).unwrap();
        assert_abs_diff_eq!(
            mode.radial_root(),
            1.841_183_781_340_659_3,
            epsilon = 2.0e-14
        );
        let expected_k = ((1.841_183_781_340_659_3_f64 / 0.2).powi(2)
            + (2.0 * std::f64::consts::PI / 0.8).powi(2))
        .sqrt();
        assert_abs_diff_eq!(mode.wavenumber_rad_per_m(), expected_k, epsilon = 2.0e-13);
    }

    #[test]
    fn azimuthal_mode_satisfies_rigid_walls_periodicity_and_helmholtz() {
        let mode = AzimuthalRigidCavityMode::try_new(0.2, 0.8, 343.0, 1, 0, 2).unwrap();
        for azimuth in [0.0, 0.41, 1.9] {
            for z in [0.0, 0.23, 0.8] {
                assert!(
                    mode.try_evaluate(0.2, azimuth, z)
                        .unwrap()
                        .radial_gradient
                        .abs()
                        < 5e-13
                );
            }
            for r in [0.0, 0.07, 0.2] {
                for z in [0.0, 0.8] {
                    assert!(
                        mode.try_evaluate(r, azimuth, z)
                            .unwrap()
                            .axial_gradient
                            .abs()
                            < 5e-13
                    );
                }
            }
        }
        let sample = mode.try_evaluate(0.11, 0.31, 0.29).unwrap();
        let wrapped = mode
            .try_evaluate(0.11, 0.31 + std::f64::consts::TAU, 0.29)
            .unwrap();
        assert_abs_diff_eq!(sample.pressure, wrapped.pressure, epsilon = 2e-14);
        assert_abs_diff_eq!(
            mode.try_evaluate(0.0, 0.31, 0.29).unwrap().pressure,
            0.0,
            epsilon = 2e-14
        );
        assert_abs_diff_eq!(
            mode.try_helmholtz_residual(0.11, 0.31, 0.29).unwrap(),
            0.0,
            epsilon = 2e-12
        );
    }

    #[test]
    fn azimuthal_mode_checked_admission_rejects_invalid_inputs() {
        assert!(matches!(
            AzimuthalRigidCavityMode::try_new(0.2, 0.8, 343.0, 0, 0, 0),
            Err(AzimuthalCavityError::ZeroFrequencyMode)
        ));
        let mode = AzimuthalRigidCavityMode::try_new(0.2, 0.8, 343.0, 1, 0, 0).unwrap();
        assert!(matches!(
            mode.try_evaluate(0.21, 0.0, 0.4),
            Err(AzimuthalCavityError::EvaluationOutsideCavity {
                coordinate: "radius_m",
                ..
            })
        ));
        assert!(matches!(
            mode.try_evaluate(0.1, f64::NAN, 0.4),
            Err(AzimuthalCavityError::InvalidParameter {
                field: "azimuth_rad",
                ..
            })
        ));
    }
}
