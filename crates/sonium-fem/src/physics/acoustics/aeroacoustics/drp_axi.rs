//! Axisymmetric DRP propagation of spinning linearized Euler modes.
//!
//! This solver advances one azimuthal Fourier mode of the linearized
//! Euler equations about uniform axial flow on a meridional `(r, z)` grid.
//! The spinning ansatz is carried as a real cosine/sine modal pair,
//! `q(r, φ, z, t) = Qc(r, z, t)·cos(mφ) + Qs(r, z, t)·sin(mφ)`, so the
//! `im/r` azimuthal coupling becomes a real skew-symmetric exchange
//! between the pair members:
//!
//! ```text
//! ∂t Pc + U·∂z Pc + ρc²(∂r Urc + Urc/r + ∂z Uzc + (m/r)·Uhs) = 0,
//! ∂t Uhc + U·∂z Uhc + (m/(ρr))·Ps = 0,
//! ```
//!
//! with mirrored signs in the sine member. Meridional derivatives use the
//! seven-point Tam--Webb DRP stencil on a cell-centered grid: cell centers
//! never touch `r = 0`, so no singular `1/r` evaluation occurs, and the
//! axis enters through regularity-aware ghost mirrors derived from
//! [`AzimuthalMode`] (symmetric for even powers, antisymmetric for the
//! linear power). Time integration is classical RK4 with preallocated
//! scratch, matching the Cartesian solver in [`super::drp_structured`].
//!
//! The CFL rate adds the modal coupling scale `|m|·c/r_min` to the
//! stencil advection bound, keeping `dt·rate ≤ 1` stable by construction.
//!
//! # Reference
//!
//! C. K. W. Tam and J. C. Webb, "Dispersion-Relation-Preserving Finite
//! Difference Schemes for Computational Acoustics", *Journal of Computational
//! Physics* 107 (1993), 262--281.

// Rust guideline compliant 2026-02-21

use sonium_wave::analytical::annular_duct_modes::DuctModeError;
use sonium_wave::analytical::azimuthal::AzimuthalMode;
use spec_math::Bessel as _;

use super::drp_euler::{AcousticBranch, DrpStencil, TAM_WEBB_DRP_7};
use super::error::{AeroacousticsError, require_finite, require_positive};

/// Boundary treatment applied at a radial face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxiRadialFace {
    /// Symmetry axis (`r = 0`): regularity-aware ghost mirrors.
    Axis,
    /// Slip wall: radial velocity mirrors antisymmetrically.
    Wall,
    /// Zero-gradient exterior with sponge absorption ahead of the face.
    Outflow {
        /// Sponge thickness in grid cells from the face.
        sponge_points: usize,
        /// Peak sponge rate `[1/s]` at the face.
        sponge_strength: f64,
    },
}

/// Boundary treatment applied at an axial face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxiAxialFace {
    /// Periodic wrap into the ghost cells.
    Periodic,
    /// Zero-gradient exterior with sponge absorption ahead of the face.
    Outflow {
        /// Sponge thickness in grid cells from the face.
        sponge_points: usize,
        /// Peak sponge rate `[1/s]` at the face.
        sponge_strength: f64,
    },
}

/// Configuration of an axisymmetric modal DRP grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxiDrpConfig {
    /// Radial cell count.
    pub n_radial: usize,
    /// Inner radius `[m]` (zero for a full disk, positive for an annulus).
    pub inner_radius: f64,
    /// Outer radius `[m]`.
    pub outer_radius: f64,
    /// Axial cell count.
    pub n_axial: usize,
    /// Axial extent `[m]`.
    pub length: f64,
    /// Azimuthal Fourier order of the modal pair.
    pub azimuthal: AzimuthalMode,
    /// Uniform mean density `[kg/m^3]`.
    pub density: f64,
    /// Uniform sound speed `[m/s]`.
    pub speed_of_sound: f64,
    /// Uniform axial mean-flow velocity `[m/s]`.
    pub axial_velocity: f64,
    /// Explicit RK4 step `[s]`.
    pub time_step: f64,
    /// Treatment at the inner radial face.
    pub inner_face: AxiRadialFace,
    /// Treatment at the outer radial face.
    pub outer_face: AxiRadialFace,
    /// Treatment at the lower axial face.
    pub lower_axial: AxiAxialFace,
    /// Treatment at the upper axial face.
    pub upper_axial: AxiAxialFace,
}

/// Rigorous ceiling of the stencil modified wavenumber `k*·dx`.
///
/// The triangle inequality bounds `|2·Σ a_j sin(j·θ)|` by `2·Σ|a_j|`,
/// evaluated here from the stencil coefficients so the stability bound
/// can never drift from the stencil it guards.
fn modified_wavenumber_bound() -> f64 {
    2.0 * TAM_WEBB_DRP_7
        .coefficients
        .iter()
        .map(|coefficient| coefficient.abs())
        .sum::<f64>()
}

impl AxiDrpConfig {
    /// Construct a grid and select `dt` from a CFL number.
    ///
    /// The step is `dt = CFL / rate` with the modal characteristic
    /// rate, so `CFL ≤ 1` is stable by construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::acoustics::aeroacoustics::drp_axi::{
    ///     AxiAxialFace, AxiDrpConfig, AxiRadialFace,
    /// };
    /// use sonium_wave::analytical::azimuthal::AzimuthalMode;
    ///
    /// let config = AxiDrpConfig::with_cfl(
    ///     16,
    ///     0.0,
    ///     0.5,
    ///     32,
    ///     2.0,
    ///     AzimuthalMode::axisymmetric(),
    ///     1.2,
    ///     343.0,
    ///     30.0,
    ///     0.5,
    ///     AxiRadialFace::Axis,
    ///     AxiRadialFace::Wall,
    ///     AxiAxialFace::Periodic,
    ///     AxiAxialFace::Periodic,
    /// )
    /// .unwrap();
    /// assert!(config.time_step > 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for a short grid, a non-positive
    /// parameter, a CFL number outside `(0, 1]`, an axis/wall face that
    /// contradicts the inner radius, or a sponge covering the grid.
    #[allow(clippy::too_many_arguments)]
    pub fn with_cfl(
        n_radial: usize,
        inner_radius: f64,
        outer_radius: f64,
        n_axial: usize,
        length: f64,
        azimuthal: AzimuthalMode,
        density: f64,
        speed_of_sound: f64,
        axial_velocity: f64,
        cfl: f64,
        inner_face: AxiRadialFace,
        outer_face: AxiRadialFace,
        lower_axial: AxiAxialFace,
        upper_axial: AxiAxialFace,
    ) -> Result<Self, AeroacousticsError> {
        let candidate = Self {
            n_radial,
            inner_radius,
            outer_radius,
            n_axial,
            length,
            azimuthal,
            density,
            speed_of_sound,
            axial_velocity,
            time_step: f64::NAN,
            inner_face,
            outer_face,
            lower_axial,
            upper_axial,
        };
        candidate.validate_shape()?;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        let rate = candidate.characteristic_rate();
        if !rate.is_finite() || rate <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "characteristic rate",
                value: rate,
            });
        }
        let mut configured = candidate;
        configured.time_step = cfl / rate;
        configured.validate()?;
        Ok(configured)
    }

    /// Radial cell width `[m]`.
    pub fn dr(&self) -> f64 {
        (self.outer_radius - self.inner_radius) / self.n_radial as f64
    }

    /// Axial cell width `[m]`.
    pub fn dz(&self) -> f64 {
        self.length / self.n_axial as f64
    }

    /// Cell-center radius of one radial index `[m]`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not a radial cell, which signals a
    /// caller-side indexing bug.
    pub fn radius_at(&self, index: usize) -> f64 {
        assert!(index < self.n_radial, "radial index out of range");
        self.inner_radius + (index as f64 + 0.5) * self.dr()
    }

    /// Cell-center axial position of one axial index `[m]`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not an axial cell, which signals a
    /// caller-side indexing bug.
    pub fn axial_at(&self, index: usize) -> f64 {
        assert!(index < self.n_axial, "axial index out of range");
        (index as f64 + 0.5) * self.dz()
    }

    /// Interior cell count over the meridional grid.
    pub fn n_nodes(&self) -> usize {
        self.n_radial * self.n_axial
    }

    /// Unsigned azimuthal order as a coupling coefficient.
    fn modal_order(&self) -> f64 {
        f64::from(self.azimuthal.abs_order())
    }

    /// Spectral-radius bound of the modal operator `[1/s]`.
    ///
    /// Adds the azimuthal coupling scale `|m|·c/r_min` (plus the
    /// `u_r/r` geometric scale `c/r_min`) to the stencil advection
    /// bound, so `dt·rate ≤ 1` keeps every mode RK4-stable.
    pub fn characteristic_rate(&self) -> f64 {
        let bound = modified_wavenumber_bound();
        let advection = bound
            * (self.axial_velocity.abs() / self.dz()
                + self.speed_of_sound / self.dz()
                + self.speed_of_sound / self.dr());
        let r_min = self.inner_radius + 0.5 * self.dr();
        advection + (self.modal_order() * self.speed_of_sound + self.speed_of_sound) / r_min
    }

    /// Validate grid shape, flow parameters, and the CFL contract.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for the same conditions as
    /// [`Self::with_cfl`], plus a time step violating `dt·rate ≤ 1`.
    pub fn validate(&self) -> Result<(), AeroacousticsError> {
        self.validate_shape()?;
        require_positive("time_step", self.time_step)?;
        let rate = self.characteristic_rate();
        if !rate.is_finite() || rate <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "characteristic rate",
                value: rate,
            });
        }
        let cfl = self.time_step * rate;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), AeroacousticsError> {
        if self.n_radial < 7 {
            return Err(AeroacousticsError::GridTooSmall {
                minimum: 7,
                actual: self.n_radial,
            });
        }
        if self.n_axial < 7 {
            return Err(AeroacousticsError::GridTooSmall {
                minimum: 7,
                actual: self.n_axial,
            });
        }
        require_finite("inner_radius", self.inner_radius)?;
        if self.inner_radius < 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "inner_radius",
                value: self.inner_radius,
            });
        }
        require_positive("outer_radius", self.outer_radius)?;
        // Both radii are finite past the guards above, so `>=` is exact.
        if self.inner_radius >= self.outer_radius {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "inner radius must sit strictly inside the outer radius",
            });
        }
        require_positive("length", self.length)?;
        require_positive("density", self.density)?;
        require_positive("speed_of_sound", self.speed_of_sound)?;
        require_finite("axial_velocity", self.axial_velocity)?;
        if self.inner_radius == 0.0 && self.inner_face != AxiRadialFace::Axis {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "a disk grid needs the Axis treatment at r = 0",
            });
        }
        if self.inner_radius > 0.0 && self.inner_face == AxiRadialFace::Axis {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "an annular grid cannot carry the Axis treatment",
            });
        }
        validate_radial_sponge(self.inner_face, self.n_radial)?;
        validate_radial_sponge(self.outer_face, self.n_radial)?;
        validate_axial_sponge(self.lower_axial, self.n_axial)?;
        validate_axial_sponge(self.upper_axial, self.n_axial)?;
        Ok(())
    }
}

fn validate_radial_sponge(face: AxiRadialFace, n_cells: usize) -> Result<(), AeroacousticsError> {
    if let AxiRadialFace::Outflow {
        sponge_points,
        sponge_strength,
    } = face
    {
        validate_sponge(sponge_points, sponge_strength, n_cells)?;
    }
    Ok(())
}

fn validate_axial_sponge(face: AxiAxialFace, n_cells: usize) -> Result<(), AeroacousticsError> {
    if let AxiAxialFace::Outflow {
        sponge_points,
        sponge_strength,
    } = face
    {
        validate_sponge(sponge_points, sponge_strength, n_cells)?;
    }
    Ok(())
}

fn validate_sponge(
    sponge_points: usize,
    sponge_strength: f64,
    n_cells: usize,
) -> Result<(), AeroacousticsError> {
    if sponge_points == 0 || sponge_points >= n_cells {
        return Err(AeroacousticsError::UnsupportedDiscretization {
            detail: "sponge layer must cover 1..n_cells-1 cells",
        });
    }
    require_finite("sponge_strength", sponge_strength)?;
    if sponge_strength < 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "sponge_strength",
            value: sponge_strength,
        });
    }
    Ok(())
}

/// One member of the cosine/sine modal pair.
///
/// Fields are row-major over `(radial, axial)` interior cells with the
/// radial index fastest.
#[derive(Debug, Clone, PartialEq)]
pub struct AxiModalState {
    /// Pressure perturbation `[Pa]`.
    pub pressure: Vec<f64>,
    /// Radial-velocity perturbation `[m/s]`.
    pub radial_velocity: Vec<f64>,
    /// Axial-velocity perturbation `[m/s]`.
    pub axial_velocity: Vec<f64>,
    /// Azimuthal-velocity perturbation `[m/s]`.
    pub azimuthal_velocity: Vec<f64>,
}

impl AxiModalState {
    /// Construct a zero modal member.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid, which signals a
    /// caller-side setup bug: admitted configs pass validation.
    pub fn zeros(config: &AxiDrpConfig) -> Self {
        config.validate_shape().expect("valid axi-DRP grid");
        Self {
            pressure: vec![0.0; config.n_nodes()],
            radial_velocity: vec![0.0; config.n_nodes()],
            axial_velocity: vec![0.0; config.n_nodes()],
            azimuthal_velocity: vec![0.0; config.n_nodes()],
        }
    }
}

/// Cosine/sine modal-pair perturbation state.
#[derive(Debug, Clone, PartialEq)]
pub struct AxiDrpState {
    /// Cosine (`cos(mφ)`) member.
    pub cosine: AxiModalState,
    /// Sine (`sin(mφ)`) member.
    pub sine: AxiModalState,
}

impl AxiDrpState {
    /// Construct a zero modal pair.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid, which signals a
    /// caller-side setup bug: admitted configs pass validation.
    pub fn zeros(config: &AxiDrpConfig) -> Self {
        Self {
            cosine: AxiModalState::zeros(config),
            sine: AxiModalState::zeros(config),
        }
    }

    /// Construct a φ-independent axial packet in the cosine member.
    ///
    /// The packet is uniform in radius with an axial Gaussian envelope
    /// and carrier, polarized as `uz = ±p / (rho·c)`; the sine member
    /// stays zero. Residual radial/azimuthal velocity must remain at
    /// roundoff, which gates the axis treatment for `m = 0`.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an invalid configuration or a
    /// non-finite or non-positive shape parameter.
    pub fn axial_packet(
        config: &AxiDrpConfig,
        center: f64,
        standard_deviation: f64,
        axial_wavenumber: f64,
        pressure_amplitude: f64,
        branch: AcousticBranch,
    ) -> Result<Self, AeroacousticsError> {
        config.validate()?;
        require_finite("packet center", center)?;
        require_positive("packet standard_deviation", standard_deviation)?;
        require_finite("packet wavenumber", axial_wavenumber)?;
        require_finite("pressure amplitude", pressure_amplitude)?;
        let sign = match branch {
            AcousticBranch::Downstream => 1.0,
            AcousticBranch::Upstream => -1.0,
        };
        let mut state = Self::zeros(config);
        let periodic = matches!(config.lower_axial, AxiAxialFace::Periodic)
            && matches!(config.upper_axial, AxiAxialFace::Periodic);
        for j in 0..config.n_axial {
            let z = config.axial_at(j);
            let delta = if periodic {
                periodic_distance(z, center, config.length)
            } else {
                z - center
            };
            let envelope = (-0.5 * (delta / standard_deviation).powi(2)).exp();
            let pressure = pressure_amplitude * envelope * (axial_wavenumber * delta).cos();
            let axial = if axial_wavenumber == 0.0 {
                0.0
            } else {
                sign * pressure / (config.density * config.speed_of_sound)
            };
            for i in 0..config.n_radial {
                let node = i + config.n_radial * j;
                state.cosine.pressure[node] = pressure;
                state.cosine.axial_velocity[node] = axial;
            }
        }
        Ok(state)
    }
}

/// Initialized spinning duct mode with its analytic descriptors.
#[derive(Debug, Clone)]
pub struct DuctModeInit {
    /// Modal-pair state of the downstream mode at `t = 0`.
    pub state: AxiDrpState,
    /// Analytic angular frequency `[rad/s]`.
    pub angular_frequency: f64,
    /// Transverse wavenumber `κ = χ/R` `[rad/m]`.
    pub transverse_wavenumber: f64,
}

/// Bessel `J_m` profile of a hard-wall circular-duct mode.
fn duct_radial_shape(order: u32, kappa: f64, radius: f64) -> f64 {
    (kappa * radius).bessel_jv(f64::from(order))
}

/// Radial derivative `dJ_m(κr)/dr` via the Bessel recurrence.
///
/// Below `x = κr = 0` the recurrence is singular; the only finite
/// limits are `J_0' = 0` and `J_1' = κ/2`, used on that branch.
fn duct_radial_slope(order: u32, kappa: f64, radius: f64) -> f64 {
    let x = kappa * radius;
    if x == 0.0 {
        return if order == 1 { 0.5 * kappa } else { 0.0 };
    }
    let down = if order == 0 {
        -x.bessel_jv(1.0)
    } else {
        x.bessel_jv(f64::from(order) - 1.0)
    };
    0.5 * kappa * (down - x.bessel_jv(f64::from(order) + 1.0))
}

/// Initialize a hard-wall circular-duct spinning mode.
///
/// Builds the downstream mode `(m, n)` with transverse root `χ` from
/// [`circular_duct_radial_root`](sonium_wave::analytical::annular_duct_modes::circular_duct_radial_root)
/// and axial wavenumber `kz`: the transverse profile is `J_m(κr)` with
/// `κ = χ/R`, the frequency follows the convected dispersion relation
/// `(ω − U·kz)² = c²(kz² + κ²)`, and velocities carry the exact LEE
/// polarization. The outer face must be a wall (the root enforces the
/// `J_m' = 0` wall condition) and `kz` must be a periodic lattice
/// harmonic of the axial extent.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for an invalid configuration, a
/// non-wall outer face, a non-harmonic axial wavenumber, or a duct-mode
/// evaluation failure.
pub fn duct_mode_init(
    config: &AxiDrpConfig,
    radial_mode: usize,
    axial_wavenumber: f64,
    pressure_amplitude: f64,
) -> Result<DuctModeInit, AeroacousticsError> {
    config.validate()?;
    require_finite("axial wavenumber", axial_wavenumber)?;
    require_finite("pressure amplitude", pressure_amplitude)?;
    if config.outer_face != AxiRadialFace::Wall {
        return Err(AeroacousticsError::UnsupportedDiscretization {
            detail: "duct modes need a slip wall at the outer radius",
        });
    }
    let harmonic = axial_wavenumber * config.length / (2.0 * std::f64::consts::PI);
    if (harmonic - harmonic.round()).abs() > 1e-9 {
        return Err(AeroacousticsError::DuctMode {
            detail: format!("axial wavenumber {axial_wavenumber} is not a lattice harmonic"),
        });
    }
    let order = config.azimuthal.abs_order();
    let root =
        sonium_wave::analytical::annular_duct_modes::circular_duct_radial_root(order, radial_mode)?;
    let kappa = root / config.outer_radius;
    let doppler = config.speed_of_sound * (axial_wavenumber.powi(2) + kappa.powi(2)).sqrt();
    let angular_frequency = doppler + config.axial_velocity * axial_wavenumber;
    if !angular_frequency.is_finite() || angular_frequency <= 0.0 {
        return Err(DuctModeError::InvalidParameter {
            field: "angular_frequency",
            requirement: "finite and positive",
        }
        .into());
    }
    let rho = config.density;
    let m = config.modal_order();
    let mut state = AxiDrpState::zeros(config);
    for j in 0..config.n_axial {
        let phase = axial_wavenumber * config.axial_at(j);
        let (phase_sin, phase_cos) = phase.sin_cos();
        for i in 0..config.n_radial {
            let r = config.radius_at(i);
            let shape = duct_radial_shape(order, kappa, r);
            let slope = duct_radial_slope(order, kappa, r);
            let node = i + config.n_radial * j;
            // Modal pair of Re[Q̂·e^{i(mφ+kz·z)}]: the cosine member takes
            // Re[Q̂·e^{ikz·z}], the sine member −Im[Q̂·e^{ikz·z}].
            let p_cos = pressure_amplitude * shape * phase_cos;
            let p_sin = -pressure_amplitude * shape * phase_sin;
            // u_r = −i·κJ'/(ρ·ω̃), u_φ = m·J/(r·ρ·ω̃), u_z = kz·J/(ρ·ω̃).
            let ur_hat = pressure_amplitude * slope / (rho * doppler);
            let uph_hat = pressure_amplitude * m * shape / (r * rho * doppler);
            let uz_hat = pressure_amplitude * axial_wavenumber * shape / (rho * doppler);
            state.cosine.pressure[node] = p_cos;
            state.sine.pressure[node] = p_sin;
            state.cosine.radial_velocity[node] = ur_hat * phase_sin;
            state.sine.radial_velocity[node] = ur_hat * phase_cos;
            state.cosine.azimuthal_velocity[node] = uph_hat * phase_cos;
            state.sine.azimuthal_velocity[node] = -uph_hat * phase_sin;
            state.cosine.axial_velocity[node] = uz_hat * phase_cos;
            state.sine.axial_velocity[node] = -uz_hat * phase_sin;
        }
    }
    Ok(DuctModeInit {
        state,
        angular_frequency,
        transverse_wavenumber: kappa,
    })
}

/// Axisymmetric modal DRP/RK4 solver.
#[derive(Debug, Clone)]
pub struct AxiDrpEuler {
    /// Grid and mean-flow parameters.
    pub config: AxiDrpConfig,
    /// Current modal-pair state.
    pub state: AxiDrpState,
    /// Simulated time `[s]`.
    pub time: f64,
    scratch: AxiDrpScratch,
}

impl AxiDrpEuler {
    /// Construct a solver from a validated state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::acoustics::aeroacoustics::drp_axi::{
    ///     AxiAxialFace, AxiDrpConfig, AxiDrpEuler, AxiDrpState, AxiRadialFace,
    /// };
    /// use sonium_wave::analytical::azimuthal::AzimuthalMode;
    ///
    /// let config = AxiDrpConfig::with_cfl(
    ///     8, 0.0, 0.5, 16, 2.0, AzimuthalMode::axisymmetric(), 1.2, 343.0, 0.0, 0.5,
    ///     AxiRadialFace::Axis, AxiRadialFace::Wall, AxiAxialFace::Periodic,
    ///     AxiAxialFace::Periodic,
    /// )
    /// .unwrap();
    /// let state = AxiDrpState::zeros(&config);
    /// let mut solver = AxiDrpEuler::new(config, state).unwrap();
    /// solver.advance(4);
    /// assert!(solver.time > 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an invalid configuration, a
    /// state whose arrays miss the grid length, or a non-finite state.
    pub fn new(config: AxiDrpConfig, state: AxiDrpState) -> Result<Self, AeroacousticsError> {
        config.validate()?;
        let n = config.n_nodes();
        for member in [&state.cosine, &state.sine] {
            for field in [
                &member.pressure,
                &member.radial_velocity,
                &member.axial_velocity,
                &member.azimuthal_velocity,
            ] {
                if field.len() != n {
                    return Err(AeroacousticsError::StateLengthMismatch {
                        expected: n,
                        pressure: state.cosine.pressure.len(),
                        velocity: field.len(),
                    });
                }
                if field.iter().any(|x| !x.is_finite()) {
                    return Err(AeroacousticsError::NonFiniteField { name: "LEE state" });
                }
            }
        }
        let scratch = AxiDrpScratch::zeros(&config);
        Ok(Self {
            config,
            state,
            time: 0.0,
            scratch,
        })
    }

    /// Advance one classical RK4 step.
    pub fn step(&mut self) {
        let dt = self.config.time_step;
        let scratch = &mut self.scratch;
        rhs(
            &self.config,
            &self.state,
            &mut scratch.k1,
            &mut scratch.work,
        );
        combine(
            &self.state,
            &scratch.k1,
            0.5 * dt,
            &mut scratch.intermediate,
        );
        rhs(
            &self.config,
            &scratch.intermediate,
            &mut scratch.k2,
            &mut scratch.work,
        );
        combine(
            &self.state,
            &scratch.k2,
            0.5 * dt,
            &mut scratch.intermediate,
        );
        rhs(
            &self.config,
            &scratch.intermediate,
            &mut scratch.k3,
            &mut scratch.work,
        );
        combine(&self.state, &scratch.k3, dt, &mut scratch.intermediate);
        rhs(
            &self.config,
            &scratch.intermediate,
            &mut scratch.k4,
            &mut scratch.work,
        );
        accumulate(
            &mut self.state.cosine,
            &scratch.k1.cosine,
            &scratch.k2.cosine,
            &scratch.k3.cosine,
            &scratch.k4.cosine,
            dt,
        );
        accumulate(
            &mut self.state.sine,
            &scratch.k1.sine,
            &scratch.k2.sine,
            &scratch.k3.sine,
            &scratch.k4.sine,
            dt,
        );
        self.time += dt;
    }

    /// Advance an integer number of RK4 steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

/// Total acoustic energy of a modal pair `[J]`.
///
/// Integrates `π·r·(e_cosine + e_sine)` over the meridional cells with
/// the plain-wave energy density per member.
pub fn acoustic_energy_axi(config: &AxiDrpConfig, state: &AxiDrpState) -> f64 {
    let rho = config.density;
    let c = config.speed_of_sound;
    let cell = config.dr() * config.dz() * std::f64::consts::PI;
    let mut total = 0.0;
    for j in 0..config.n_axial {
        for i in 0..config.n_radial {
            let node = i + config.n_radial * j;
            let r = config.radius_at(i);
            for member in [&state.cosine, &state.sine] {
                total += r
                    * (member.pressure[node].powi(2) / (2.0 * rho * c * c)
                        + rho * member.radial_velocity[node].powi(2) / 2.0
                        + rho * member.axial_velocity[node].powi(2) / 2.0
                        + rho * member.azimuthal_velocity[node].powi(2) / 2.0);
            }
        }
    }
    total * cell
}

/// r-weighted pair pressure energy along the axis.
///
/// Returns, per axial cell, `Σ_i r_i·(Pc² + Ps²)`: the weight profile
/// feeding [`centroid_1d`](super::wave_diagnostics::centroid_1d) in
/// axial group-velocity gates.
pub fn axial_energy_profile(config: &AxiDrpConfig, state: &AxiDrpState) -> Vec<f64> {
    let mut profile = vec![0.0; config.n_axial];
    for (j, slot) in profile.iter_mut().enumerate() {
        let mut sum = 0.0;
        for i in 0..config.n_radial {
            let node = i + config.n_radial * j;
            sum += config.radius_at(i)
                * (state.cosine.pressure[node].powi(2) + state.sine.pressure[node].powi(2));
        }
        *slot = sum;
    }
    profile
}

/// Ghost-mirror parities `[p, u_r, u_z, u_φ]` at the symmetry axis.
///
/// Fields with linear on-axis power mirror antisymmetrically (odd
/// extension of a function vanishing at the axis); all other powers
/// mirror symmetrically, which is exact for even powers and carries
/// only roundoff-scale error for the tiny higher odd powers.
fn axis_parities(mode: AzimuthalMode) -> [f64; 4] {
    let parity = |power: u32| {
        if power == 1 { -1.0 } else { 1.0 }
    };
    let transverse = mode.velocity_axis_powers();
    [
        parity(mode.scalar_axis_power()),
        parity(transverse[0]),
        parity(transverse[2]),
        parity(transverse[1]),
    ]
}

#[derive(Debug, Clone)]
struct AxiDrpScratch {
    k1: AxiDrpState,
    k2: AxiDrpState,
    k3: AxiDrpState,
    k4: AxiDrpState,
    intermediate: AxiDrpState,
    work: AxiDrpWork,
}

#[derive(Debug, Clone)]
struct AxiDrpWork {
    ghosted: Vec<f64>,
    line: Vec<f64>,
    pressure_radial: Vec<f64>,
    pressure_axial: Vec<f64>,
    radial_radial: Vec<f64>,
    radial_axial: Vec<f64>,
    axial_radial: Vec<f64>,
    axial_axial: Vec<f64>,
    azimuthal_axial: Vec<f64>,
    sponge: Vec<f64>,
    radii: Vec<f64>,
}

impl AxiDrpScratch {
    fn zeros(config: &AxiDrpConfig) -> Self {
        Self {
            k1: AxiDrpState::zeros(config),
            k2: AxiDrpState::zeros(config),
            k3: AxiDrpState::zeros(config),
            k4: AxiDrpState::zeros(config),
            intermediate: AxiDrpState::zeros(config),
            work: AxiDrpWork::zeros(config),
        }
    }
}

impl AxiDrpWork {
    fn zeros(config: &AxiDrpConfig) -> Self {
        let n = config.n_nodes();
        let radius = DrpStencil::RADIUS;
        let ghosted = (config.n_radial + 2 * radius) * (config.n_axial + 2 * radius);
        let mut radii = Vec::with_capacity(config.n_radial);
        for i in 0..config.n_radial {
            radii.push(config.radius_at(i));
        }
        Self {
            ghosted: vec![0.0; ghosted],
            line: vec![0.0; config.n_radial.max(config.n_axial)],
            pressure_radial: vec![0.0; n],
            pressure_axial: vec![0.0; n],
            radial_radial: vec![0.0; n],
            radial_axial: vec![0.0; n],
            axial_radial: vec![0.0; n],
            axial_axial: vec![0.0; n],
            azimuthal_axial: vec![0.0; n],
            sponge: sponge_profile(config),
            radii,
        }
    }
}

/// Quadratic sponge profile summed over all outflow faces.
fn sponge_profile(config: &AxiDrpConfig) -> Vec<f64> {
    let mut sponge = vec![0.0; config.n_nodes()];
    for j in 0..config.n_axial {
        for i in 0..config.n_radial {
            let node = i + config.n_radial * j;
            let mut rate = 0.0;
            for (face, depth) in [
                (config.inner_face, i),
                (config.outer_face, config.n_radial - 1 - i),
            ] {
                if let AxiRadialFace::Outflow {
                    sponge_points,
                    sponge_strength,
                } = face
                {
                    rate += sponge_rate(depth, sponge_points, sponge_strength);
                }
            }
            for (face, depth) in [
                (config.lower_axial, j),
                (config.upper_axial, config.n_axial - 1 - j),
            ] {
                if let AxiAxialFace::Outflow {
                    sponge_points,
                    sponge_strength,
                } = face
                {
                    rate += sponge_rate(depth, sponge_points, sponge_strength);
                }
            }
            sponge[node] = rate;
        }
    }
    sponge
}

fn sponge_rate(depth: usize, sponge_points: usize, sponge_strength: f64) -> f64 {
    // Depth counts cells from the face inward; the profile peaks at the
    // face and vanishes at the sponge edge.
    let inside = sponge_points - 1 - depth.min(sponge_points - 1);
    let fraction = inside as f64 / sponge_points as f64;
    sponge_strength * fraction * fraction
}

fn pack_interior(config: &AxiDrpConfig, interior: &[f64], ghosted: &mut [f64]) {
    let radius = DrpStencil::RADIUS;
    let stride = config.n_radial + 2 * radius;
    for j in 0..config.n_axial {
        for i in 0..config.n_radial {
            ghosted[(j + radius) * stride + i + radius] = interior[i + config.n_radial * j];
        }
    }
}

/// Mirror parity of one field at a radial face.
///
/// `field` indexes `[p, u_r, u_z, u_φ]`; walls pin the radial velocity
/// while the axis follows the modal regularity table.
fn radial_parity(config: &AxiDrpConfig, face: AxiRadialFace, field: usize) -> f64 {
    match face {
        AxiRadialFace::Axis => axis_parities(config.azimuthal)[field],
        AxiRadialFace::Wall => {
            if field == 1 {
                -1.0
            } else {
                1.0
            }
        }
        AxiRadialFace::Outflow { .. } => 1.0,
    }
}

fn fill_ghosts_radial(config: &AxiDrpConfig, field: usize, ghosted: &mut [f64]) {
    let radius = DrpStencil::RADIUS;
    let stride = config.n_radial + 2 * radius;
    for j in 0..config.n_axial {
        let row = (j + radius) * stride;
        for layer in 0..radius {
            ghosted[row + radius - 1 - layer] = match config.inner_face {
                AxiRadialFace::Outflow { .. } => ghosted[row + radius],
                face => radial_parity(config, face, field) * ghosted[row + radius + layer],
            };
            ghosted[row + radius + config.n_radial + layer] = match config.outer_face {
                AxiRadialFace::Outflow { .. } => ghosted[row + radius + config.n_radial - 1],
                face => {
                    radial_parity(config, face, field)
                        * ghosted[row + radius + config.n_radial - 1 - layer]
                }
            };
        }
    }
}

fn fill_ghosts_axial(config: &AxiDrpConfig, ghosted: &mut [f64]) {
    let radius = DrpStencil::RADIUS;
    let stride = config.n_radial + 2 * radius;
    for layer in 0..radius {
        for column in 0..stride {
            ghosted[(radius - 1 - layer) * stride + column] = match config.lower_axial {
                // Layer 0 is adjacent to interior[0], so it wraps from
                // interior[n-1] (slot radius + n - 1 - layer).
                AxiAxialFace::Periodic => {
                    ghosted[(radius + config.n_axial - 1 - layer) * stride + column]
                }
                AxiAxialFace::Outflow { .. } => ghosted[radius * stride + column],
            };
            ghosted[(radius + config.n_axial + layer) * stride + column] = match config.upper_axial
            {
                AxiAxialFace::Periodic => ghosted[(radius + layer) * stride + column],
                AxiAxialFace::Outflow { .. } => {
                    ghosted[(radius + config.n_axial - 1) * stride + column]
                }
            };
        }
    }
}

/// Differentiate all radial lines of a packed field.
fn differentiate_radial(
    config: &AxiDrpConfig,
    ghosted: &[f64],
    line: &mut [f64],
    output: &mut [f64],
) {
    let radius = DrpStencil::RADIUS;
    let stride = config.n_radial + 2 * radius;
    for j in 0..config.n_axial {
        let base = (j + radius) * stride;
        TAM_WEBB_DRP_7.derivative_ghosted_line(
            ghosted,
            base,
            1,
            config.n_radial,
            config.dr(),
            line,
        );
        for i in 0..config.n_radial {
            output[i + config.n_radial * j] = line[i];
        }
    }
}

/// Differentiate all axial lines of a packed field.
fn differentiate_axial(
    config: &AxiDrpConfig,
    ghosted: &[f64],
    line: &mut [f64],
    output: &mut [f64],
) {
    let radius = DrpStencil::RADIUS;
    let stride = config.n_radial + 2 * radius;
    for i in 0..config.n_radial {
        let base = i + radius;
        TAM_WEBB_DRP_7.derivative_ghosted_line(
            ghosted,
            base,
            stride,
            config.n_axial,
            config.dz(),
            line,
        );
        for j in 0..config.n_axial {
            output[i + config.n_radial * j] = line[j];
        }
    }
}

fn differentiate_member(config: &AxiDrpConfig, member: &AxiModalState, work: &mut AxiDrpWork) {
    differentiate_member_radial(config, member, work);
    differentiate_member_axial(config, member, work);
}

fn differentiate_member_radial(
    config: &AxiDrpConfig,
    member: &AxiModalState,
    work: &mut AxiDrpWork,
) {
    // Field order [p, u_r, u_z] matches radial_parity indexing; the
    // azimuthal velocity needs no radial derivative in the φ-momentum.
    let fields = [
        (&member.pressure, 0),
        (&member.radial_velocity, 1),
        (&member.axial_velocity, 2),
    ];
    for (field, field_index) in fields {
        pack_interior(config, field, &mut work.ghosted);
        fill_ghosts_radial(config, field_index, &mut work.ghosted);
        let output: &mut Vec<f64> = match field_index {
            0 => &mut work.pressure_radial,
            1 => &mut work.radial_radial,
            _ => &mut work.axial_radial,
        };
        differentiate_radial(config, &work.ghosted, &mut work.line, output);
    }
}

fn differentiate_member_axial(
    config: &AxiDrpConfig,
    member: &AxiModalState,
    work: &mut AxiDrpWork,
) {
    let fields = [
        &member.pressure,
        &member.radial_velocity,
        &member.axial_velocity,
        &member.azimuthal_velocity,
    ];
    for (field_index, field) in fields.iter().enumerate() {
        pack_interior(config, field, &mut work.ghosted);
        fill_ghosts_axial(config, &mut work.ghosted);
        let output: &mut Vec<f64> = match field_index {
            0 => &mut work.pressure_axial,
            1 => &mut work.radial_axial,
            2 => &mut work.axial_axial,
            _ => &mut work.azimuthal_axial,
        };
        differentiate_axial(config, &work.ghosted, &mut work.line, output);
    }
}

fn rhs(
    config: &AxiDrpConfig,
    state: &AxiDrpState,
    output: &mut AxiDrpState,
    work: &mut AxiDrpWork,
) {
    rhs_member(
        config,
        &state.cosine,
        &state.sine,
        &mut output.cosine,
        1.0,
        work,
    );
    rhs_member(
        config,
        &state.sine,
        &state.cosine,
        &mut output.sine,
        -1.0,
        work,
    );
}

/// Right-hand side of one pair member.
///
/// `other` is the opposite member feeding the skew-symmetric modal
/// coupling; `coupling_sign` is `+1` for the cosine member and `−1`
/// for the sine member.
fn rhs_member(
    config: &AxiDrpConfig,
    member: &AxiModalState,
    other: &AxiModalState,
    output: &mut AxiModalState,
    coupling_sign: f64,
    work: &mut AxiDrpWork,
) {
    differentiate_member(config, member, work);
    let rho = config.density;
    let c = config.speed_of_sound;
    let flow = config.axial_velocity;
    let m = config.modal_order();
    for j in 0..config.n_axial {
        for i in 0..config.n_radial {
            let node = i + config.n_radial * j;
            let r = work.radii[i];
            let sponge = work.sponge[node];
            output.pressure[node] = -sponge * member.pressure[node]
                - flow * work.pressure_axial[node]
                - rho
                    * c
                    * c
                    * (work.radial_radial[node]
                        + member.radial_velocity[node] / r
                        + work.axial_axial[node]
                        + coupling_sign * m * other.azimuthal_velocity[node] / r);
            output.radial_velocity[node] = -sponge * member.radial_velocity[node]
                - flow * work.radial_axial[node]
                - work.pressure_radial[node] / rho;
            output.axial_velocity[node] = -sponge * member.axial_velocity[node]
                - flow * work.axial_axial[node]
                - work.pressure_axial[node] / rho;
            output.azimuthal_velocity[node] = -sponge * member.azimuthal_velocity[node]
                - flow * work.azimuthal_axial[node]
                - coupling_sign * m * other.pressure[node] / (rho * r);
        }
    }
}

fn combine(base: &AxiDrpState, increment: &AxiDrpState, scale: f64, out: &mut AxiDrpState) {
    combine_member(&base.cosine, &increment.cosine, scale, &mut out.cosine);
    combine_member(&base.sine, &increment.sine, scale, &mut out.sine);
}

fn combine_member(
    base: &AxiModalState,
    increment: &AxiModalState,
    scale: f64,
    out: &mut AxiModalState,
) {
    for (out_field, (base_field, increment_field)) in [
        (&mut out.pressure, (&base.pressure, &increment.pressure)),
        (
            &mut out.radial_velocity,
            (&base.radial_velocity, &increment.radial_velocity),
        ),
        (
            &mut out.axial_velocity,
            (&base.axial_velocity, &increment.axial_velocity),
        ),
        (
            &mut out.azimuthal_velocity,
            (&base.azimuthal_velocity, &increment.azimuthal_velocity),
        ),
    ] {
        for i in 0..base_field.len() {
            out_field[i] = base_field[i] + scale * increment_field[i];
        }
    }
}

fn accumulate(
    state: &mut AxiModalState,
    k1: &AxiModalState,
    k2: &AxiModalState,
    k3: &AxiModalState,
    k4: &AxiModalState,
    dt: f64,
) {
    for (state_field, stages) in [
        (
            &mut state.pressure,
            (&k1.pressure, &k2.pressure, &k3.pressure, &k4.pressure),
        ),
        (
            &mut state.radial_velocity,
            (
                &k1.radial_velocity,
                &k2.radial_velocity,
                &k3.radial_velocity,
                &k4.radial_velocity,
            ),
        ),
        (
            &mut state.axial_velocity,
            (
                &k1.axial_velocity,
                &k2.axial_velocity,
                &k3.axial_velocity,
                &k4.axial_velocity,
            ),
        ),
        (
            &mut state.azimuthal_velocity,
            (
                &k1.azimuthal_velocity,
                &k2.azimuthal_velocity,
                &k3.azimuthal_velocity,
                &k4.azimuthal_velocity,
            ),
        ),
    ] {
        for (i, slot) in state_field.iter_mut().enumerate() {
            *slot += dt * (stages.0[i] + 2.0 * stages.1[i] + 2.0 * stages.2[i] + stages.3[i]) / 6.0;
        }
    }
}

fn periodic_distance(x: f64, center: f64, length: f64) -> f64 {
    (x - center + 0.5 * length).rem_euclid(length) - 0.5 * length
}

#[cfg(test)]
mod tests {
    use super::super::wave_diagnostics::{centroid_1d, group_velocity_from_track};
    use super::*;
    use std::f64::consts::PI;

    fn duct_config(order: i32) -> AxiDrpConfig {
        AxiDrpConfig::with_cfl(
            24,
            0.0,
            0.5,
            72,
            1.5,
            AzimuthalMode::new(order),
            1.2,
            343.0,
            34.3,
            0.9,
            AxiRadialFace::Axis,
            AxiRadialFace::Wall,
            AxiAxialFace::Periodic,
            AxiAxialFace::Periodic,
        )
        .unwrap()
    }

    #[test]
    fn config_validation_rejects_bad_inputs() {
        assert!(matches!(
            AxiDrpConfig::with_cfl(
                6,
                0.0,
                0.5,
                72,
                1.5,
                AzimuthalMode::axisymmetric(),
                1.2,
                343.0,
                0.0,
                0.5,
                AxiRadialFace::Axis,
                AxiRadialFace::Wall,
                AxiAxialFace::Periodic,
                AxiAxialFace::Periodic,
            ),
            Err(AeroacousticsError::GridTooSmall { .. })
        ));
        // A disk grid with a wall at r = 0 contradicts the axis.
        assert!(matches!(
            AxiDrpConfig::with_cfl(
                24,
                0.0,
                0.5,
                72,
                1.5,
                AzimuthalMode::axisymmetric(),
                1.2,
                343.0,
                0.0,
                0.5,
                AxiRadialFace::Wall,
                AxiRadialFace::Wall,
                AxiAxialFace::Periodic,
                AxiAxialFace::Periodic,
            ),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
        // An annular grid cannot carry the axis treatment.
        assert!(matches!(
            AxiDrpConfig::with_cfl(
                24,
                0.2,
                0.5,
                72,
                1.5,
                AzimuthalMode::axisymmetric(),
                1.2,
                343.0,
                0.0,
                0.5,
                AxiRadialFace::Axis,
                AxiRadialFace::Wall,
                AxiAxialFace::Periodic,
                AxiAxialFace::Periodic,
            ),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
        let config = duct_config(0);
        let mut over_cfl = config;
        over_cfl.time_step *= 1.2;
        assert!(matches!(
            over_cfl.validate(),
            Err(AeroacousticsError::InvalidCfl(_))
        ));
        // Duct modes need a lattice-harmonic axial wavenumber.
        assert!(matches!(
            duct_mode_init(&config, 1, 1.0, 1.0),
            Err(AeroacousticsError::DuctMode { .. })
        ));
    }

    #[test]
    fn axis_parities_follow_regularity_table() {
        // [p, u_r, u_z, u_φ]: antisymmetric mirrors carry −1.
        assert_eq!(axis_parities(AzimuthalMode::new(0)), [1.0, -1.0, 1.0, -1.0]);
        assert_eq!(axis_parities(AzimuthalMode::new(1)), [-1.0, 1.0, -1.0, 1.0]);
        assert_eq!(axis_parities(AzimuthalMode::new(2)), [1.0, -1.0, 1.0, -1.0]);
        // |m| = 3: transverse power 2 mirrors symmetrically.
        assert_eq!(axis_parities(AzimuthalMode::new(-3)), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn axisymmetric_packet_advects_at_downstream_speed() {
        let config = duct_config(0);
        let kz = 2.0 * PI / config.length * 2.0;
        let state =
            AxiDrpState::axial_packet(&config, 0.5, 0.12, kz, 1.0, AcousticBranch::Downstream)
                .unwrap();
        let mut solver = AxiDrpEuler::new(config.clone(), state).unwrap();
        let coordinates: Vec<f64> = (0..config.n_axial).map(|j| config.axial_at(j)).collect();
        let mut times = Vec::new();
        let mut centroids = Vec::new();
        for _ in 0..10 {
            solver.advance(20);
            let profile = axial_energy_profile(&config, &solver.state);
            let centroid = centroid_1d(&coordinates, &profile).unwrap();
            if centroid > 1.2 {
                break;
            }
            times.push(solver.time);
            centroids.push(centroid);
        }
        assert!(times.len() >= 6, "track too short: {}", times.len());
        let measured = group_velocity_from_track(&times, &centroids).unwrap();
        let analytic = config.axial_velocity + config.speed_of_sound;
        assert!(
            (measured - analytic).abs() / analytic < 5e-3,
            "measured={measured}, analytic={analytic}"
        );
        // Radial uniformity must survive: no spurious transverse flow.
        let transverse: f64 = solver
            .state
            .cosine
            .radial_velocity
            .iter()
            .chain(solver.state.cosine.azimuthal_velocity.iter())
            .chain(solver.state.sine.pressure.iter())
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        let axial_peak: f64 = solver
            .state
            .cosine
            .axial_velocity
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        assert!(
            transverse / axial_peak < 1e-10,
            "transverse leakage={transverse:e}"
        );
    }

    /// Rising zero crossings of a probe track by linear interpolation.
    fn crossing_times(times: &[f64], samples: &[f64]) -> Vec<f64> {
        let mut crossings = Vec::new();
        for window in samples.windows(2).enumerate() {
            let (i, pair) = window;
            if pair[0] <= 0.0 && pair[1] > 0.0 {
                let fraction = -pair[0] / (pair[1] - pair[0]);
                crossings.push(times[i] + fraction * (times[i + 1] - times[i]));
            }
        }
        crossings
    }

    #[test]
    fn spinning_duct_mode_matches_analytic_frequency_and_profile() {
        let config = duct_config(1);
        let kz = 2.0 * PI / config.length;
        let init = duct_mode_init(&config, 0, kz, 1.0).unwrap();
        // Analytic check: χ = 1.841... is the first J_1' zero.
        assert!((init.transverse_wavenumber * config.outer_radius - 1.841_183_781_34).abs() < 1e-6);
        let mut solver = AxiDrpEuler::new(config.clone(), init.state).unwrap();
        let probe_radial = config.n_radial / 2;
        let probe_axial = config.n_axial / 2;
        let probe = probe_radial + config.n_radial * probe_axial;
        let mut times = Vec::new();
        let mut samples = Vec::new();
        let steps = (0.010 / config.time_step).ceil() as usize;
        for _ in 0..steps {
            solver.step();
            times.push(solver.time);
            samples.push(solver.state.cosine.pressure[probe]);
        }
        let crossings = crossing_times(&times, &samples);
        assert!(crossings.len() >= 3, "crossings={crossings:?}");
        let periods = crossings.len() as f64 - 1.0;
        let measured = 2.0 * PI * periods / (crossings.last().unwrap() - crossings[0]);
        assert!(
            (measured - init.angular_frequency).abs() / init.angular_frequency < 0.02,
            "measured={measured}, analytic={}",
            init.angular_frequency
        );
        // Transverse profile keeps the J_1 shape at fixed (z, t).
        let mut dot = 0.0;
        let mut state_norm = 0.0;
        let mut bessel_norm = 0.0;
        for i in 0..config.n_radial {
            let node = i + config.n_radial * probe_axial;
            let shape = duct_radial_shape(1, init.transverse_wavenumber, config.radius_at(i));
            let value = solver.state.cosine.pressure[node];
            dot += value * shape;
            state_norm += value * value;
            bessel_norm += shape * shape;
        }
        let correlation = dot / (state_norm * bessel_norm).sqrt();
        assert!(correlation.abs() > 0.999, "correlation={correlation}");
        // On-axis pressure node: J_1(κr)/max(J_1) at the first center.
        let first = solver.state.cosine.pressure[probe_axial * config.n_radial];
        let first_sine = solver.state.sine.pressure[probe_axial * config.n_radial];
        let amplitude = (first * first + first_sine * first_sine).sqrt();
        let mut peak = 0.0f64;
        for i in 0..config.n_radial {
            let node = i + config.n_radial * probe_axial;
            let pair = (solver.state.cosine.pressure[node].powi(2)
                + solver.state.sine.pressure[node].powi(2))
            .sqrt();
            peak = peak.max(pair);
        }
        let expected =
            duct_radial_shape(1, init.transverse_wavenumber, config.radius_at(0)) / 0.581_865_53;
        assert!(
            (amplitude / peak - expected).abs() / expected < 0.05,
            "axis ratio={} vs {expected}",
            amplitude / peak
        );
    }

    #[test]
    fn periodic_axial_ghosts_wrap_in_exact_order() {
        let config = duct_config(0);
        let radius = DrpStencil::RADIUS;
        let stride = config.n_radial + 2 * radius;
        let mut ghosted = vec![0.0; stride * (config.n_axial + 2 * radius)];
        // Axial ramp: interior row j holds its own index.
        for j in 0..config.n_axial {
            for i in 0..config.n_radial {
                ghosted[(j + radius) * stride + i + radius] = j as f64;
            }
        }
        fill_ghosts_axial(&config, &mut ghosted);
        let n = config.n_axial;
        for layer in 0..radius {
            // Lower ghosts (adjacent-first) read the tail in reverse.
            assert_eq!(
                ghosted[(radius - 1 - layer) * stride + radius],
                (n - 1 - layer) as f64
            );
            // Upper ghosts (adjacent-first) read the head in order.
            assert_eq!(
                ghosted[(radius + n + layer) * stride + radius],
                layer as f64
            );
        }
    }

    #[test]
    fn second_spinning_mode_stays_bounded() {
        let config = duct_config(2);
        let kz = 2.0 * PI / config.length;
        let init = duct_mode_init(&config, 0, kz, 1.0).unwrap();
        let mut solver = AxiDrpEuler::new(config.clone(), init.state).unwrap();
        let initial = acoustic_energy_axi(&config, &solver.state);
        solver.advance((0.004 / config.time_step).ceil() as usize);
        for member in [&solver.state.cosine, &solver.state.sine] {
            for field in [
                &member.pressure,
                &member.radial_velocity,
                &member.axial_velocity,
                &member.azimuthal_velocity,
            ] {
                assert!(field.iter().all(|x| x.is_finite()));
            }
        }
        let drift = (acoustic_energy_axi(&config, &solver.state) - initial).abs() / initial;
        assert!(drift < 0.05, "energy drift={drift:e}");
    }

    #[test]
    fn axial_sponge_absorbs_an_outgoing_packet() {
        let mut config = duct_config(0);
        config.upper_axial = AxiAxialFace::Outflow {
            sponge_points: 16,
            sponge_strength: 8000.0,
        };
        config.time_step = 0.9 / config.characteristic_rate();
        config.validate().unwrap();
        let kz = 2.0 * PI / config.length * 2.0;
        let state =
            AxiDrpState::axial_packet(&config, 0.4, 0.1, kz, 1.0, AcousticBranch::Downstream)
                .unwrap();
        let mut solver = AxiDrpEuler::new(config.clone(), state).unwrap();
        let initial = acoustic_energy_axi(&config, &solver.state);
        solver.advance((0.008 / config.time_step).ceil() as usize);
        let remaining = acoustic_energy_axi(&config, &solver.state) / initial;
        assert!(remaining < 0.02, "reflected fraction={remaining:e}");
    }
}
