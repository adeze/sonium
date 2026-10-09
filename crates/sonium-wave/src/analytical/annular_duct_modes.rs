//! Annular and circular hard-wall duct modes with azimuthal order.
//!
//! Spinning duct modes `J_m(κr) cos(mφ - k_z z)` (circular) and their
//! annular `J_m/Y_m` counterparts are the reference for lined-duct,
//! rotor-stator, and axisymmetric-assembly validation. This module provides
//! radial roots, cut-on/cut-off data, axial wavenumbers with uniform axial
//! mean flow, modal group velocity, lined-wall (impedance) roots for the
//! circular duct, and Tyler–Sofrin interaction orders.
//!
//! # Radial-mode indexing
//!
//! The plane-wave branch `(m, n) = (0, 0)` has `κ = 0`. Otherwise `n`
//! counts positive roots of the wall discriminant: for `m = 0`, `n ≥ 1`
//! selects the `n`-th positive `J₁` zero; for `m ≥ 1`, `n = 0` selects the
//! first positive root. This matches [`AzimuthalRigidCavityMode`].
//!
//! Axial dependence is `exp(-ik_z z)` under `exp(+iωt)`, so downstream
//! modes carry `Re(k_z) > 0` and decaying downstream evanescent modes
//! carry `Im(k_z) < 0`, matching the rectangular-duct convention in
//! `sonium-fem` aeroacoustics.
//!
//! [`AzimuthalRigidCavityMode`]: super::axisymmetric_duct::AzimuthalRigidCavityMode

use num_complex::Complex64;
use spec_math::Bessel as _;
use std::f64::consts::PI;

use super::azimuthal::AzimuthalMode;

/// Failures from duct-mode construction or evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum DuctModeError {
    /// A geometric or physical parameter violated its contract.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Human-readable requirement.
        requirement: &'static str,
    },
    /// Root bracketing or Newton refinement missed the requested root.
    RootConvergenceFailure {
        /// Requested azimuthal order.
        azimuthal_order: u32,
        /// Requested radial mode.
        radial_mode: usize,
        /// Hub-to-tip ratio in effect.
        hub_tip_ratio: f64,
    },
    /// An evaluation produced a non-finite value.
    NonFiniteResult,
}

impl std::fmt::Display for DuctModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(f, "duct mode parameter {field} must be {requirement}")
            }
            Self::RootConvergenceFailure {
                azimuthal_order,
                radial_mode,
                hub_tip_ratio,
            } => write!(
                f,
                "could not converge the (m={azimuthal_order}, n={radial_mode}) wall root at hub ratio {hub_tip_ratio}"
            ),
            Self::NonFiniteResult => {
                write!(f, "duct mode evaluation produced a non-finite value")
            }
        }
    }
}

impl std::error::Error for DuctModeError {}

/// Downstream or upstream axial branch of one duct mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxialBranch {
    /// Energy flows toward increasing `z`.
    Downstream,
    /// Energy flows toward decreasing `z`.
    Upstream,
}

fn bessel_j(order: u32, argument: f64) -> f64 {
    argument.bessel_jv(f64::from(order))
}

fn bessel_y(order: u32, argument: f64) -> f64 {
    argument.bessel_yv(f64::from(order))
}

fn bessel_j_prime(order: u32, argument: f64) -> f64 {
    if order == 0 {
        -bessel_j(1, argument)
    } else {
        0.5 * (bessel_j(order - 1, argument) - bessel_j(order + 1, argument))
    }
}

fn bessel_y_prime(order: u32, argument: f64) -> f64 {
    if order == 0 {
        -bessel_y(1, argument)
    } else {
        0.5 * (bessel_y(order - 1, argument) - bessel_y(order + 1, argument))
    }
}

/// Collects the first `wanted` positive roots of a smooth discriminant.
///
/// Scans from just above the origin with a fixed step, then bisects each
/// sign bracket. Returns `None` when the scan budget or the upper bound is
/// exhausted first, or when the discriminant goes non-finite.
fn scan_positive_roots(
    discriminant: impl Fn(f64) -> f64,
    wanted: usize,
    step: f64,
    upper_bound: f64,
) -> Option<Vec<f64>> {
    let mut roots = Vec::with_capacity(wanted);
    let mut left = step;
    let mut left_value = discriminant(left);
    if !left_value.is_finite() {
        return None;
    }
    let mut x = left;
    while roots.len() < wanted {
        x += step;
        if x > upper_bound {
            return None;
        }
        let right_value = discriminant(x);
        if !right_value.is_finite() {
            return None;
        }
        if left_value.signum() != right_value.signum() {
            let mut lo = left;
            let mut hi = x;
            let mut lo_value = left_value;
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                let mid_value = discriminant(mid);
                if !mid_value.is_finite() {
                    return None;
                }
                if mid_value == 0.0 {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if lo_value.signum() == mid_value.signum() {
                    lo = mid;
                    lo_value = mid_value;
                } else {
                    hi = mid;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
        left = x;
        left_value = right_value;
    }
    Some(roots)
}

/// Radial root `χ = κa` of a hard-wall circular duct.
///
/// # Errors
///
/// Returns [`DuctModeError::RootConvergenceFailure`] when bracketing misses
/// the requested root.
pub fn circular_duct_radial_root(
    azimuthal_order: u32,
    radial_mode: usize,
) -> Result<f64, DuctModeError> {
    if azimuthal_order == 0 && radial_mode == 0 {
        return Ok(0.0);
    }
    // Positive-root rank: m = 0 counts J₁ zeros from 1, m ≥ 1 from 0.
    let rank = if azimuthal_order == 0 {
        radial_mode
    } else {
        radial_mode + 1
    };
    let upper = (f64::from(azimuthal_order) + rank as f64 + 4.0) * PI + 8.0;
    let roots = scan_positive_roots(
        |x| bessel_j_prime(azimuthal_order, x),
        rank,
        PI / 32.0,
        upper,
    )
    .ok_or(DuctModeError::RootConvergenceFailure {
        azimuthal_order,
        radial_mode,
        hub_tip_ratio: 0.0,
    })?;
    Ok(roots[rank - 1])
}

/// Radial root `χ = κa` of a hard-wall annular duct.
///
/// The discriminant `J_m'(σχ)Y_m'(χ) − J_m'(χ)Y_m'(σχ)` enforces rigid
/// walls at hub (`σa`) and tip (`a`). A zero hub ratio delegates to the
/// circular duct.
///
/// # Errors
///
/// Returns [`DuctModeError`] for a hub ratio outside `[0, 1)` or a missed
/// root bracket.
pub fn annular_duct_radial_root(
    azimuthal_order: u32,
    radial_mode: usize,
    hub_tip_ratio: f64,
) -> Result<f64, DuctModeError> {
    if !hub_tip_ratio.is_finite() || !(0.0..1.0).contains(&hub_tip_ratio) {
        return Err(DuctModeError::InvalidParameter {
            field: "hub_tip_ratio",
            requirement: "finite and within [0, 1)",
        });
    }
    if hub_tip_ratio == 0.0 {
        return circular_duct_radial_root(azimuthal_order, radial_mode);
    }
    if azimuthal_order == 0 && radial_mode == 0 {
        return Ok(0.0);
    }
    let rank = if azimuthal_order == 0 {
        radial_mode
    } else {
        radial_mode + 1
    };
    // Narrow gaps push the first root toward π/(1-σ); widen the scan.
    let upper = ((f64::from(azimuthal_order) + rank as f64 + 4.0) * PI)
        .max(8.0 / (1.0 - hub_tip_ratio))
        + 8.0;
    let roots = scan_positive_roots(
        |x| {
            bessel_j_prime(azimuthal_order, hub_tip_ratio * x) * bessel_y_prime(azimuthal_order, x)
                - bessel_j_prime(azimuthal_order, x)
                    * bessel_y_prime(azimuthal_order, hub_tip_ratio * x)
        },
        rank,
        PI / 32.0,
        upper,
    )
    .ok_or(DuctModeError::RootConvergenceFailure {
        azimuthal_order,
        radial_mode,
        hub_tip_ratio,
    })?;
    Ok(roots[rank - 1])
}

/// Checked hard-wall annular (or circular) duct mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnularDuctMode {
    hub_radius_m: f64,
    tip_radius_m: f64,
    sound_speed_m_per_s: f64,
    azimuthal: AzimuthalMode,
    radial_mode: usize,
    radial_root: f64,
    transverse_wavenumber_rad_per_m: f64,
}

impl AnnularDuctMode {
    /// Builds the `(m, n)` hard-wall mode of an annular duct.
    ///
    /// A zero hub radius selects the circular duct. The radial root is the
    /// `n`-th positive wall-discriminant zero per the module indexing.
    ///
    /// # Errors
    ///
    /// Returns [`DuctModeError`] for invalid geometry, a non-positive sound
    /// speed, or a missed root bracket.
    pub fn try_new(
        hub_radius_m: f64,
        tip_radius_m: f64,
        sound_speed_m_per_s: f64,
        azimuthal: AzimuthalMode,
        radial_mode: usize,
    ) -> Result<Self, DuctModeError> {
        if !hub_radius_m.is_finite() || hub_radius_m < 0.0 {
            return Err(DuctModeError::InvalidParameter {
                field: "hub_radius_m",
                requirement: "finite and non-negative",
            });
        }
        if !tip_radius_m.is_finite() || tip_radius_m <= hub_radius_m {
            return Err(DuctModeError::InvalidParameter {
                field: "tip_radius_m",
                requirement: "finite and larger than the hub radius",
            });
        }
        if !sound_speed_m_per_s.is_finite() || sound_speed_m_per_s <= 0.0 {
            return Err(DuctModeError::InvalidParameter {
                field: "sound_speed_m_per_s",
                requirement: "finite and positive",
            });
        }
        let hub_tip_ratio = hub_radius_m / tip_radius_m;
        let radial_root =
            annular_duct_radial_root(azimuthal.abs_order(), radial_mode, hub_tip_ratio)?;
        let transverse_wavenumber_rad_per_m = radial_root / tip_radius_m;
        if !transverse_wavenumber_rad_per_m.is_finite() {
            return Err(DuctModeError::NonFiniteResult);
        }
        Ok(Self {
            hub_radius_m,
            tip_radius_m,
            sound_speed_m_per_s,
            azimuthal,
            radial_mode,
            radial_root,
            transverse_wavenumber_rad_per_m,
        })
    }

    /// Returns the azimuthal mode.
    pub fn azimuthal(self) -> AzimuthalMode {
        self.azimuthal
    }

    /// Returns the radial-mode index.
    pub fn radial_mode(self) -> usize {
        self.radial_mode
    }

    /// Returns the dimensionless wall root `χ = κa`.
    pub fn radial_root(self) -> f64 {
        self.radial_root
    }

    /// Returns the transverse wavenumber `κ` in rad/m.
    pub fn transverse_wavenumber(self) -> f64 {
        self.transverse_wavenumber_rad_per_m
    }

    /// Cut-on frequency in Hz at one axial Mach number.
    ///
    /// With uniform axial flow the mode propagates for `ω > βκc`, so flow
    /// lowers the cut-on frequency by the Prandtl–Glauert factor.
    ///
    /// # Errors
    ///
    /// Returns [`DuctModeError`] for a non-subsonic Mach number.
    pub fn cutoff_frequency_hz(self, mach_axial: f64) -> Result<f64, DuctModeError> {
        let beta = subsonic_beta(mach_axial)?;
        Ok(beta * self.transverse_wavenumber_rad_per_m * self.sound_speed_m_per_s / (2.0 * PI))
    }

    /// Reports whether the mode propagates at one frequency and Mach number.
    ///
    /// # Errors
    ///
    /// Returns [`DuctModeError`] for a non-finite frequency or a
    /// non-subsonic Mach number.
    pub fn is_cut_on(self, angular_frequency: f64, mach_axial: f64) -> Result<bool, DuctModeError> {
        if !angular_frequency.is_finite() {
            return Err(DuctModeError::InvalidParameter {
                field: "angular_frequency",
                requirement: "finite",
            });
        }
        let beta = subsonic_beta(mach_axial)?;
        let k = angular_frequency / self.sound_speed_m_per_s;
        Ok(k * k > beta * beta * self.transverse_wavenumber_rad_per_m.powi(2))
    }

    /// Axial wavenumbers `(downstream, upstream)` with uniform axial flow.
    ///
    /// Propagating branches are real; cut-off branches carry the decaying
    /// imaginary part per the module axial convention.
    ///
    /// # Errors
    ///
    /// Returns [`DuctModeError`] for a non-finite frequency or a
    /// non-subsonic Mach number.
    pub fn axial_wavenumbers(
        self,
        angular_frequency: f64,
        mach_axial: f64,
    ) -> Result<(Complex64, Complex64), DuctModeError> {
        if !angular_frequency.is_finite() {
            return Err(DuctModeError::InvalidParameter {
                field: "angular_frequency",
                requirement: "finite",
            });
        }
        let beta = subsonic_beta(mach_axial)?;
        let beta_sq = beta * beta;
        let k = angular_frequency / self.sound_speed_m_per_s;
        let discriminant = k * k - beta_sq * self.transverse_wavenumber_rad_per_m.powi(2);
        let shift = -k * mach_axial / beta_sq;
        if discriminant >= 0.0 {
            let axial = discriminant.sqrt() / beta_sq;
            Ok((
                Complex64::new(shift + axial, 0.0),
                Complex64::new(shift - axial, 0.0),
            ))
        } else {
            let axial = (-discriminant).sqrt() / beta_sq;
            Ok((Complex64::new(shift, -axial), Complex64::new(shift, axial)))
        }
    }

    /// Modal group velocity `dω/dk_z` in m/s, if the mode propagates.
    ///
    /// Returns `None` for cut-off (evanescent) modes and `Some(0.0)` exactly
    /// at cut-on, where energy stops propagating axially.
    ///
    /// # Errors
    ///
    /// Returns [`DuctModeError`] for a non-finite frequency or a
    /// non-subsonic Mach number.
    pub fn group_velocity(
        self,
        angular_frequency: f64,
        mach_axial: f64,
        branch: AxialBranch,
    ) -> Result<Option<f64>, DuctModeError> {
        let (downstream, upstream) = self.axial_wavenumbers(angular_frequency, mach_axial)?;
        let axial = match branch {
            AxialBranch::Downstream => downstream,
            AxialBranch::Upstream => upstream,
        };
        if axial.im != 0.0 {
            return Ok(None);
        }
        let beta = subsonic_beta(mach_axial)?;
        let k = angular_frequency / self.sound_speed_m_per_s;
        let numerator = beta * beta * axial.re + mach_axial * k;
        let denominator = k - mach_axial * axial.re;
        if denominator == 0.0 {
            return Ok(None);
        }
        let velocity = self.sound_speed_m_per_s * numerator / denominator;
        if !velocity.is_finite() {
            return Err(DuctModeError::NonFiniteResult);
        }
        Ok(Some(velocity))
    }

    /// Radial eigenfunction normalized to unit tip value.
    ///
    /// The circular shape is `J_m(κr)`; the annular shape combines `J_m` and
    /// `Y_m` to satisfy the tip Neumann condition by construction. The
    /// plane-wave branch returns the uniform shape.
    ///
    /// # Errors
    ///
    /// Returns [`DuctModeError`] for a radius outside the annulus or a
    /// non-finite shape value.
    pub fn radial_shape(self, radius_m: f64) -> Result<f64, DuctModeError> {
        if !radius_m.is_finite() || radius_m < self.hub_radius_m || radius_m > self.tip_radius_m {
            return Err(DuctModeError::InvalidParameter {
                field: "radius_m",
                requirement: "finite and inside the annulus",
            });
        }
        if self.radial_root == 0.0 {
            return Ok(1.0);
        }
        let order = self.azimuthal.abs_order();
        let kappa = self.transverse_wavenumber_rad_per_m;
        let raw = |r: f64| {
            if self.hub_radius_m == 0.0 {
                bessel_j(order, kappa * r)
            } else {
                let tip = kappa * self.tip_radius_m;
                bessel_j(order, kappa * r) * bessel_y_prime(order, tip)
                    - bessel_y(order, kappa * r) * bessel_j_prime(order, tip)
            }
        };
        let scale = raw(self.tip_radius_m);
        let normalizer =
            if scale.abs() > 1e-12 * raw(self.hub_radius_m.max(kappa.recip())).abs().max(1e-300) {
                scale
            } else {
                // A tip node is non-generic but possible; fall back to the peak.
                let mut peak = 0.0f64;
                for sample in 0..=64 {
                    let r = self.hub_radius_m
                        + (self.tip_radius_m - self.hub_radius_m) * f64::from(sample) / 64.0;
                    peak = peak.max(raw(r).abs());
                }
                if peak == 0.0 {
                    return Err(DuctModeError::NonFiniteResult);
                }
                peak
            };
        let value = raw(radius_m) / normalizer;
        if !value.is_finite() {
            return Err(DuctModeError::NonFiniteResult);
        }
        Ok(value)
    }
}

fn subsonic_beta(mach_axial: f64) -> Result<f64, DuctModeError> {
    if !mach_axial.is_finite() || mach_axial.abs() >= 1.0 {
        return Err(DuctModeError::InvalidParameter {
            field: "mach_axial",
            requirement: "finite with magnitude below 1",
        });
    }
    Ok((1.0 - mach_axial * mach_axial).sqrt())
}

/// Integer Bessel `J_m` of a complex argument via series.
///
/// The ascending power series converges for every finite `z`; lined-wall
/// roots sit near real hard-wall roots where a few dozen terms reach
/// binary64 precision. Validated against real-argument `spec_math` values.
fn complex_bessel_j(order: u32, z: Complex64) -> Complex64 {
    let half = z * 0.5;
    let mut term = Complex64::new(1.0, 0.0);
    for factor in 1..=order {
        term *= half / f64::from(factor);
    }
    let step = -half * half;
    let mut sum = term;
    for index in 1..=400usize {
        let step_index = index as f64;
        term *= step / (step_index * (step_index + f64::from(order)));
        sum += term;
        if term.norm() <= 1e-16 * sum.norm().max(1e-300) {
            break;
        }
    }
    sum
}

fn complex_bessel_j_prime(order: u32, z: Complex64) -> Complex64 {
    if order == 0 {
        -complex_bessel_j(1, z)
    } else {
        (complex_bessel_j(order - 1, z) - complex_bessel_j(order + 1, z)) * 0.5
    }
}

/// Lined-wall radial root `κa` of a circular duct branch.
///
/// Solves the Ingard–Myers dispersion relation `κa J_m'(κa)/J_m(κa) =
/// −iρa(ω − Uk_z)²/(ωZ)` by homotopy: the wall admittance ramps from the
/// rigid wall toward `1/Z` in stages, with damped Newton tracking the
/// acoustic branch at each stage. Direct Newton from the hard-wall root
/// overshoots across `J_m` poles for resistive liners, hence the ramp. A
/// zero impedance selects the pressure-release (Dirichlet) real roots.
/// Annular lined walls need complex `Y_m` and stay a documented follow-up.
///
/// # Errors
///
/// Returns [`DuctModeError`] for invalid parameters or a missed Newton
/// convergence.
#[allow(clippy::too_many_arguments)]
pub fn lined_circular_root(
    azimuthal_order: u32,
    radial_mode: usize,
    angular_frequency: f64,
    density: f64,
    sound_speed: f64,
    axial_velocity: f64,
    duct_radius_m: f64,
    impedance: Complex64,
    branch: AxialBranch,
) -> Result<Complex64, DuctModeError> {
    if !angular_frequency.is_finite() || angular_frequency == 0.0 {
        return Err(DuctModeError::InvalidParameter {
            field: "angular_frequency",
            requirement: "finite and nonzero",
        });
    }
    if !density.is_finite() || density <= 0.0 {
        return Err(DuctModeError::InvalidParameter {
            field: "density",
            requirement: "finite and positive",
        });
    }
    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(DuctModeError::InvalidParameter {
            field: "sound_speed",
            requirement: "finite and positive",
        });
    }
    if !duct_radius_m.is_finite() || duct_radius_m <= 0.0 {
        return Err(DuctModeError::InvalidParameter {
            field: "duct_radius_m",
            requirement: "finite and positive",
        });
    }
    if !axial_velocity.is_finite() || (axial_velocity / sound_speed).abs() >= 1.0 {
        return Err(DuctModeError::InvalidParameter {
            field: "axial_velocity",
            requirement: "finite and subsonic",
        });
    }
    if !impedance.re.is_finite() || !impedance.im.is_finite() {
        return Err(DuctModeError::InvalidParameter {
            field: "impedance",
            requirement: "finite components",
        });
    }
    if impedance.norm() == 0.0 {
        // Pressure-release wall: Dirichlet zeros of J_m.
        if azimuthal_order == 0 && radial_mode == 0 {
            // No uniform pressure-release branch exists; report the miss.
            return Err(DuctModeError::RootConvergenceFailure {
                azimuthal_order,
                radial_mode,
                hub_tip_ratio: 0.0,
            });
        }
        let rank = radial_mode + 1;
        let upper = (f64::from(azimuthal_order) + rank as f64 + 4.0) * PI + 8.0;
        let roots = scan_positive_roots(|x| bessel_j(azimuthal_order, x), rank, PI / 32.0, upper)
            .ok_or(DuctModeError::RootConvergenceFailure {
            azimuthal_order,
            radial_mode,
            hub_tip_ratio: 0.0,
        })?;
        return Ok(Complex64::new(roots[rank - 1], 0.0));
    }
    let start = circular_duct_radial_root(azimuthal_order, radial_mode)?;
    // The plane-wave start needs a nudge off the removable κ = 0 point.
    let mut root = Complex64::new(start.max(1e-6), 0.0);
    let mach = axial_velocity / sound_speed;
    let beta_sq = 1.0 - mach * mach;
    let k = angular_frequency / sound_speed;
    let admittance_target = Complex64::new(1.0, 0.0) / impedance;
    let dispersion = |kappa: Complex64, admittance: Complex64| -> Complex64 {
        let ratio = kappa * complex_bessel_j_prime(azimuthal_order, kappa)
            / complex_bessel_j(azimuthal_order, kappa);
        let kz_sq_term = beta_sq * kappa * kappa / (duct_radius_m * duct_radius_m);
        let axial = match branch {
            AxialBranch::Downstream => (-k * mach + (k * k - kz_sq_term).sqrt()) / beta_sq,
            AxialBranch::Upstream => (-k * mach - (k * k - kz_sq_term).sqrt()) / beta_sq,
        };
        let doppler = angular_frequency - axial_velocity * axial;
        // Outward-normal Myers condition: ∂p/∂r = −iρ(ω−Uk_z)²p/(ωZ),
        // hence the plus sign against the positive wall ratio.
        ratio
            + Complex64::new(0.0, 1.0) * density * duct_radius_m * doppler * doppler
                / angular_frequency
                * admittance
    };
    // Admittance homotopy: 16 stages from the rigid wall keep every Newton
    // solve inside the basin of the tracked acoustic branch.
    for stage in 1..=16usize {
        let admittance = admittance_target * (f64::from(stage as u32) / 16.0);
        root = damped_newton_stage(root, &dispersion, admittance).ok_or(
            DuctModeError::RootConvergenceFailure {
                azimuthal_order,
                radial_mode,
                hub_tip_ratio: 0.0,
            },
        )?;
    }
    Ok(root)
}

/// Damped Newton solve of one homotopy stage.
fn damped_newton_stage(
    start: Complex64,
    dispersion: &impl Fn(Complex64, Complex64) -> Complex64,
    admittance: Complex64,
) -> Option<Complex64> {
    let mut root = start;
    for _ in 0..64 {
        let value = dispersion(root, admittance);
        if !value.re.is_finite() || !value.im.is_finite() {
            return None;
        }
        if value.norm() <= 1e-12 * root.norm().max(1.0) {
            return Some(root);
        }
        let h = Complex64::new(1e-8 * root.norm().max(1.0), 0.0);
        let slope = (dispersion(root + h, admittance) - value) / h;
        if slope.norm() == 0.0 || !slope.re.is_finite() || !slope.im.is_finite() {
            return None;
        }
        let mut step = value / slope;
        let mut accepted = false;
        for _ in 0..24 {
            let trial = root - step;
            let trial_value = dispersion(trial, admittance);
            if trial_value.re.is_finite()
                && trial_value.im.is_finite()
                && trial_value.norm() < value.norm()
            {
                root = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            return None;
        }
    }
    None
}

/// Tyler–Sofrin rotor-stator interaction orders.
///
/// Returns `m = nB − kV` for stator lobe counts `k = 0..=max_stator_lobes`
/// with `B` rotor blades, `V` stator vanes, and `n`-th blade-passing
/// harmonic. A vaneless (`V = 0`) stage keeps the rotor-alone order `nB`.
///
/// # Errors
///
/// Returns [`DuctModeError`] for a zero blade count or harmonic.
pub fn tyler_sofrin_orders(
    blades: usize,
    vanes: usize,
    harmonic: usize,
    max_stator_lobes: usize,
) -> Result<Vec<i32>, DuctModeError> {
    if blades == 0 {
        return Err(DuctModeError::InvalidParameter {
            field: "blades",
            requirement: "at least one rotor blade",
        });
    }
    if harmonic == 0 {
        return Err(DuctModeError::InvalidParameter {
            field: "harmonic",
            requirement: "a positive blade-passing harmonic",
        });
    }
    let rotor = (harmonic * blades) as i32;
    if vanes == 0 {
        return Ok(vec![rotor]);
    }
    Ok((0..=max_stator_lobes)
        .map(|lobes| rotor - (lobes * vanes) as i32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn circular_roots_match_classical_bessel_zeros() {
        // First two J₁ zeros (Neumann m = 0 radial roots).
        assert!((circular_duct_radial_root(0, 1).unwrap() - 3.8317059702075125).abs() < TOL);
        assert!((circular_duct_radial_root(0, 2).unwrap() - 7.015586669815618).abs() < TOL);
        // First J₂' zero.
        assert!((circular_duct_radial_root(2, 0).unwrap() - 3.0542369282271408).abs() < 1e-8);
        assert_eq!(circular_duct_radial_root(0, 0).unwrap(), 0.0);
    }

    #[test]
    fn annular_roots_satisfy_both_wall_conditions() {
        for (m, n, sigma) in [(0, 1, 0.5), (3, 0, 0.3), (1, 2, 0.7)] {
            let root = annular_duct_radial_root(m, n, sigma).unwrap();
            let hub_residual = annular_wall_residual(m, root, sigma, true);
            let tip_residual = annular_wall_residual(m, root, sigma, false);
            assert!(
                hub_residual < 1e-8 && tip_residual < 1e-8,
                "(m={m}, n={n}, σ={sigma}): hub {hub_residual}, tip {tip_residual}"
            );
        }
    }

    fn annular_wall_residual(order: u32, root: f64, sigma: f64, hub: bool) -> f64 {
        let r = if hub { sigma } else { 1.0 };
        // Shape derivative at the wall over the shape peak scale.
        let shape_prime = bessel_j_prime(order, root * r) * bessel_y_prime(order, root)
            - bessel_y_prime(order, root * r) * bessel_j_prime(order, root);
        let scale = bessel_j(order, root).abs() + bessel_y(order, root).abs() + 1e-300;
        (root * shape_prime / scale).abs()
    }

    #[test]
    fn narrow_gap_annulus_recovers_transverse_duct_modes() {
        // A thin annulus behaves like a flat duct of width a(1-σ).
        let sigma = 0.9;
        let root = annular_duct_radial_root(0, 1, sigma).unwrap();
        let gap_modes = root * (1.0 - sigma) / PI;
        assert!((gap_modes - 1.0).abs() < 0.05, "κd/π = {gap_modes}");
    }

    #[test]
    fn axial_wavenumbers_match_closed_form_without_flow() {
        let mode = AnnularDuctMode::try_new(0.0, 0.5, 343.0, AzimuthalMode::new(0), 1).unwrap();
        let omega = 2.0 * PI * 2000.0;
        let (down, up) = mode.axial_wavenumbers(omega, 0.0).unwrap();
        let k = omega / 343.0;
        let expected = (k * k - mode.transverse_wavenumber().powi(2)).sqrt();
        assert!((down.re - expected).abs() < TOL);
        assert!((up.re + expected).abs() < TOL);
        assert!(mode.is_cut_on(omega, 0.0).unwrap());
        assert!(!mode.is_cut_on(2.0 * PI * 100.0, 0.0).unwrap());
    }

    #[test]
    fn evanescent_branches_decay_away_from_the_source() {
        let mode = AnnularDuctMode::try_new(0.0, 0.5, 343.0, AzimuthalMode::new(0), 1).unwrap();
        let (down, up) = mode.axial_wavenumbers(2.0 * PI * 50.0, 0.1).unwrap();
        assert!(down.im < 0.0 && up.im > 0.0);
        assert_eq!(
            mode.group_velocity(2.0 * PI * 50.0, 0.1, AxialBranch::Downstream)
                .unwrap(),
            None
        );
    }

    #[test]
    fn group_velocity_matches_energy_velocity_without_flow() {
        let mode = AnnularDuctMode::try_new(0.2, 0.5, 343.0, AzimuthalMode::new(2), 0).unwrap();
        let omega = 2.0 * PI * 3000.0;
        assert!(mode.is_cut_on(omega, 0.0).unwrap());
        let (down, _) = mode.axial_wavenumbers(omega, 0.0).unwrap();
        let group = mode
            .group_velocity(omega, 0.0, AxialBranch::Downstream)
            .unwrap()
            .unwrap();
        assert!((group - 343.0 * 343.0 * down.re / omega).abs() < 1e-9);
        assert!(group > 0.0 && group < 343.0);
    }

    #[test]
    fn mode_shapes_are_unit_at_tip_and_neumann_at_walls() {
        let mode = AnnularDuctMode::try_new(0.15, 0.5, 340.0, AzimuthalMode::new(1), 1).unwrap();
        assert!((mode.radial_shape(0.5).unwrap() - 1.0).abs() < 1e-12);
        // Neumann walls: central-difference the shape at both radii.
        for wall in [0.15, 0.5] {
            let h = 1e-6;
            let slope = (mode
                .radial_shape(wall + h)
                .unwrap_or(mode.radial_shape(wall).unwrap())
                - mode
                    .radial_shape(wall - h)
                    .unwrap_or(mode.radial_shape(wall).unwrap()))
                / (2.0 * h);
            assert!(slope.abs() < 1e-4, "wall {wall}: slope {slope}");
        }
    }

    #[test]
    fn complex_bessel_matches_real_bessel_on_the_axis() {
        for (order, x) in [(0, 2.5), (1, 2.5), (3, 5.0), (5, 12.0)] {
            let complex = complex_bessel_j(order, Complex64::new(x, 0.0));
            assert!((complex.re - bessel_j(order, x)).abs() < 1e-12);
            assert!(complex.im.abs() < 1e-12);
        }
    }

    #[test]
    fn rigid_liner_recovers_hard_wall_root() {
        let hard = circular_duct_radial_root(0, 1).unwrap();
        let lined = lined_circular_root(
            0,
            1,
            2.0 * PI * 2000.0,
            1.2,
            343.0,
            0.0,
            0.5,
            Complex64::new(1e12, 0.0),
            AxialBranch::Downstream,
        )
        .unwrap();
        assert!((lined.re - hard).abs() < 1e-6);
        assert!(lined.im.abs() < 1e-6);
    }

    #[test]
    fn resistive_liner_attenuates_the_downstream_mode() {
        let mode = AnnularDuctMode::try_new(0.0, 0.5, 343.0, AzimuthalMode::new(0), 1).unwrap();
        let omega = 2.0 * PI * 2000.0;
        assert!(mode.is_cut_on(omega, 0.0).unwrap());
        let root = lined_circular_root(
            0,
            1,
            omega,
            1.2,
            343.0,
            0.0,
            0.5,
            Complex64::new(1.2 * 343.0, 0.0),
            AxialBranch::Downstream,
        )
        .unwrap();
        // Lined axial wavenumber inherits Im(k_z) < 0: downstream decay.
        let kappa = root / 0.5;
        let k = omega / 343.0;
        let kz = (k * k - kappa * kappa).sqrt();
        assert!(kz.im < 0.0, "lined k_z = {kz}");
    }

    #[test]
    fn tyler_sofrin_counts_rotor_minus_stator_lobes() {
        assert_eq!(tyler_sofrin_orders(9, 10, 1, 2).unwrap(), vec![9, -1, -11]);
        assert_eq!(tyler_sofrin_orders(2, 0, 3, 5).unwrap(), vec![6]);
        assert!(tyler_sofrin_orders(0, 10, 1, 1).is_err());
    }
}
