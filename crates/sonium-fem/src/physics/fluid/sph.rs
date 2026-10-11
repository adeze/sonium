//! Smoothed particle hydrodynamics core (Phase-5 lane 5B).
//!
//! Decision spike (P5C3 verdict: BUILD, greenfield):
//! the WavesCompression deck (`SPH/WavesCompression`, rootname `CUBC`) models
//! a copper bar (`/MAT/HYDPLA` plus `/EOS/POLYNOMIAL` with `C1 = 1.3`) as SPH
//! particles (`/PROP/TYPE34`: particle mass `mp = 0.58936189`, quadratic bulk
//! viscosity `qa = 2`, linear bulk viscosity `qb = 1`, `order = -1` meaning no
//! kernel correction, automatic smoothing length) with an initial downward
//! velocity (`/INIVEL/TRA`, `Vz = -0.03815`), ghost-particle symmetry planes
//! (`/SPHBCS`: one `TIED` plane with rebound level `Ilev = 1`, four `SLIDE`
//! planes with `Ilev = 0`), global neighbor-search security `0.1` (`/SPHGLO`),
//! and a 30.1 ms explicit run (`/RUN/CUBC/1`). Theory Manual pages 440-449
//! specify a Lagrangian SPH formulation with the Monaghan cubic B-spline
//! kernel of support `2h`, variable smoothing length, symmetric pressure
//! forces, Monaghan artificial viscosity with the Balsara vorticity switch,
//! and CFL-style time step control. No SPH seed exists in `sonium-fem`
//! (whole-word `sph` search over `src/` is empty apart from this module), so a
//! greenfield module behind existing APIs is justified; nothing here edits or
//! wraps existing physics code (P5C1).
//!
//! This module implements that core: cubic B-spline kernels, a uniform-grid
//! neighbor search, summation/differential continuity, symmetric momentum with
//! artificial viscosity, compatible energy update, the polynomial equation of
//! state used by the deck, smoothing-length adaptation, ghost-particle
//! symmetry boundaries, and explicit time step control. Engine time-history
//! replay stays deferred (see `qa-fem-sph`): only the checked-in golden and
//! `SCHEMA.md` are visible offline, and vendor TH channels cannot be
//! reproduced to vendor tolerance by an independent integrator.
//!
//! W-tier fixtures: `fixtures/sph_kernel_consistency` (kernel values,
//! partition of unity, gradient accuracy) and `fixtures/sph_waves`
//! (one-step compression-wave operators with deck viscosity coefficients).

// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use thiserror::Error;

/// Support radius as a multiple of smoothing length.
///
/// The Theory Manual (p. 447) states weight functions vanish at distance
/// `2h`; the cubic B-spline below is exactly zero for `q >= 2`.
pub const SUPPORT_RADIUS_FACTOR: f64 = 2.0;

/// Default CFL coefficient for the SPH stable step.
///
/// Theory Manual p. 446 recommends `0.3` for the user-defined coefficient.
pub const DEFAULT_CFL: f64 = 0.3;

/// Softening factor in the artificial-viscosity denominator.
///
/// Standard Monaghan value; avoids a singularity when two particles coincide
/// while leaving well-separated pairs unaffected.
pub const VISCOSITY_SOFTENING: f64 = 0.01;

/// Errors from SPH configuration and operator evaluation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SphError {
    /// Smoothing length is not positive or not finite.
    #[error("smoothing length must be positive and finite, got {0}")]
    BadSmoothingLength(f64),
    /// Requested kernel dimension is not 1, 2, or 3.
    #[error("kernel dimension must be 1, 2, or 3, got {0}")]
    BadDimension(usize),
    /// A particle mass, density, or equation-of-state input is invalid.
    #[error("invalid particle state: {0}")]
    BadState(&'static str),
    /// A symmetry plane normal is zero or not finite.
    #[error("symmetry plane normal must be a finite unit axis")]
    BadPlane,
    /// CFL number is not in the open interval `(0, 1]`.
    #[error("CFL number must lie in (0, 1], got {0}")]
    BadCfl(f64),
}

/// Kernel normalization for the cubic B-spline in `dim` dimensions.
///
/// Returns the factor `sigma` such that `W = sigma / h^dim * f(q)` with the
/// Monaghan shape below integrates to one over free space.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::kernel_normalization;
/// use std::f64::consts::PI;
///
/// assert!((kernel_normalization(3).unwrap() - 1.0 / PI).abs() < 1e-15);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadDimension`] unless `dim` is 1, 2, or 3.
pub fn kernel_normalization(dim: usize) -> Result<f64, SphError> {
    match dim {
        1 => Ok(2.0 / 3.0),
        2 => Ok(10.0 / (7.0 * std::f64::consts::PI)),
        3 => Ok(1.0 / std::f64::consts::PI),
        other => Err(SphError::BadDimension(other)),
    }
}

/// Cubic B-spline shape at normalized radius `q = r / h`.
///
/// Piecewise polynomial from Monaghan (Theory Manual eq. 1233-1235):
/// `1 - 3q^2/2 + 3q^3/4` for `q < 1`, `(2 - q)^3 / 4` for `q < 2`, else `0`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::cubic_spline_shape;
///
/// assert_eq!(cubic_spline_shape(0.0), 1.0);
/// assert_eq!(cubic_spline_shape(2.5), 0.0);
/// ```
pub fn cubic_spline_shape(q: f64) -> f64 {
    if q < 0.0 {
        return 0.0;
    }
    if q < 1.0 {
        1.0 - 1.5 * q * q + 0.75 * q * q * q
    } else if q < 2.0 {
        let t = 2.0 - q;
        0.25 * t * t * t
    } else {
        0.0
    }
}

/// Shape derivative `df/dq` of the cubic B-spline.
///
/// Exact derivatives of the branches in [`cubic_spline_shape`]:
/// `-3q + 9q^2/4` for `q < 1`, `-3(2 - q)^2/4` for `q < 2`, else `0`.
fn shape_derivative(q: f64) -> f64 {
    if q <= 0.0 {
        return 0.0;
    }
    if q < 1.0 {
        -3.0 * q + 2.25 * q * q
    } else if q < 2.0 {
        let t = 2.0 - q;
        -0.75 * t * t
    } else {
        0.0
    }
}

/// Kernel value `W(r, h)` in `dim` dimensions.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::kernel_value;
///
/// let center = kernel_value(0.0, 1.0, 3).unwrap();
/// assert!(center > 0.0);
/// assert_eq!(kernel_value(2.0, 1.0, 3).unwrap(), 0.0);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`] for non-positive `h` and
/// [`SphError::BadDimension`] unless `dim` is 1, 2, or 3.
pub fn kernel_value(radius: f64, smoothing_length: f64, dim: usize) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    let sigma = kernel_normalization(dim)?;
    let q = radius / smoothing_length;
    Ok(sigma / smoothing_length.powi(dim as i32) * cubic_spline_shape(q))
}

/// Radial kernel-gradient factor `(dW/dr) / r`.
///
/// Writing the gradient as `(dW/dr) r_vec / r` needs this factor; it stays
/// finite at `r = 0` (limit `-3 sigma / h^(dim+2)`), so callers never divide
/// by the radius.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::kernel_gradient_factor;
///
/// let at_origin = kernel_gradient_factor(0.0, 1.0, 3).unwrap();
/// assert!(at_origin.is_finite() && at_origin < 0.0);
/// assert_eq!(kernel_gradient_factor(3.0, 1.0, 3).unwrap(), 0.0);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`] for non-positive `h` and
/// [`SphError::BadDimension`] unless `dim` is 1, 2, or 3.
pub fn kernel_gradient_factor(
    radius: f64,
    smoothing_length: f64,
    dim: usize,
) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    let sigma = kernel_normalization(dim)?;
    let q = radius / smoothing_length;
    if q >= 2.0 || q < 0.0 {
        return Ok(0.0);
    }
    // (dW/dr)/r = sigma / h^(d+2) * (df/dq)/q; the quotient tends to -3
    // at the origin, matching the linear term of the inner branch.
    let quotient = if q == 0.0 {
        -3.0
    } else {
        shape_derivative(q) / q
    };
    Ok(sigma / smoothing_length.powi(dim as i32 + 2) * quotient)
}

/// Kernel gradient vector at `offset`.
///
/// The gradient is taken with respect to the offset vector itself; SPH
/// operators pass the home-minus-neighbor offset so the result equals
/// `grad_i W_ij`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::kernel_gradient;
///
/// let grad = kernel_gradient([0.5, 0.0, 0.0], 1.0, 3).unwrap();
/// assert!(grad[0] < 0.0 && grad[1] == 0.0 && grad[2] == 0.0);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`] for non-positive `h` and
/// [`SphError::BadDimension`] unless `dim` is 1, 2, or 3.
pub fn kernel_gradient(
    offset: [f64; 3],
    smoothing_length: f64,
    dim: usize,
) -> Result<[f64; 3], SphError> {
    let radius = offset[0].hypot(offset[1]).hypot(offset[2]);
    let factor = kernel_gradient_factor(radius, smoothing_length, dim)?;
    Ok([offset[0] * factor, offset[1] * factor, offset[2] * factor])
}

fn check_smoothing_length(h: f64) -> Result<(), SphError> {
    if h.is_finite() && h > 0.0 {
        Ok(())
    } else {
        Err(SphError::BadSmoothingLength(h))
    }
}

/// One SPH particle (Lagrangian state).
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    /// Position in model length units.
    pub position: [f64; 3],
    /// Velocity in model length/time units.
    pub velocity: [f64; 3],
    /// Particle mass (constant along a trajectory).
    pub mass: f64,
    /// Current density.
    pub density: f64,
    /// Current pressure from the equation of state.
    pub pressure: f64,
    /// Current sound speed from the equation of state.
    pub sound_speed: f64,
    /// Specific internal energy.
    pub energy: f64,
}

impl Particle {
    /// Build a particle with unit density and zero pressure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::Particle;
    ///
    /// let p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5).unwrap();
    /// assert_eq!(p.mass, 0.5);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SphError::BadState`] when `mass` is not positive and finite.
    pub fn new(position: [f64; 3], velocity: [f64; 3], mass: f64) -> Result<Self, SphError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(SphError::BadState("mass must be positive and finite"));
        }
        Ok(Self {
            position,
            velocity,
            mass,
            density: 1.0,
            pressure: 0.0,
            sound_speed: 0.0,
            energy: 0.0,
        })
    }
}

/// Uniform-grid neighbor search with cell size equal to support.
///
/// Particles are binned into cubic cells of edge `support_radius`, so all
/// neighbors within the kernel support sit in the 27 cells around the home
/// cell. This mirrors the engine bucket sort behind `/SPHGLO`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::NeighborSearch;
///
/// let positions = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]];
/// let search = NeighborSearch::build(&positions, 2.0).unwrap();
/// assert_eq!(search.neighbors(0), vec![1]);
/// assert!(search.neighbors(2).is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct NeighborSearch {
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
    positions: Vec<[f64; 3]>,
    support_radius: f64,
}

impl NeighborSearch {
    /// Bin `positions` for neighbor queries within `support_radius`.
    ///
    /// # Errors
    ///
    /// Returns [`SphError::BadSmoothingLength`] when the support radius is
    /// not positive and finite (same validity rule as `h`).
    pub fn build(positions: &[[f64; 3]], support_radius: f64) -> Result<Self, SphError> {
        check_smoothing_length(support_radius)?;
        let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (index, position) in positions.iter().enumerate() {
            cells
                .entry(cell_key(position, support_radius))
                .or_default()
                .push(index);
        }
        Ok(Self {
            cells,
            positions: positions.to_vec(),
            support_radius,
        })
    }

    /// Sorted neighbor indices within support, excluding `index` itself.
    ///
    /// # Panics
    ///
    /// Panics when `index` is out of range for the built positions.
    pub fn neighbors(&self, index: usize) -> Vec<usize> {
        let home = self.positions[index];
        let (cx, cy, cz) = cell_key(&home, self.support_radius);
        let mut found = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &candidate in bucket {
                            if candidate == index {
                                continue;
                            }
                            let other = self.positions[candidate];
                            let dist = (other[0] - home[0])
                                .hypot(other[1] - home[1])
                                .hypot(other[2] - home[2]);
                            if dist < self.support_radius {
                                found.push(candidate);
                            }
                        }
                    }
                }
            }
        }
        found.sort_unstable();
        found
    }
}

fn cell_key(position: &[f64; 3], cell: f64) -> (i64, i64, i64) {
    (
        (position[0] / cell).floor() as i64,
        (position[1] / cell).floor() as i64,
        (position[2] / cell).floor() as i64,
    )
}

/// Brute-force neighbor lists for cross-checking the grid search.
///
/// Returns, per particle, the sorted indices within `support_radius`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::brute_force_neighbors;
///
/// let positions = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
/// assert_eq!(brute_force_neighbors(&positions, 2.0), vec![vec![1], vec![0]]);
/// ```
///
/// # Panics
///
/// Panics when `support_radius` is not positive and finite.
pub fn brute_force_neighbors(positions: &[[f64; 3]], support_radius: f64) -> Vec<Vec<usize>> {
    assert!(
        support_radius.is_finite() && support_radius > 0.0,
        "support radius must be positive and finite"
    );
    let mut lists = vec![Vec::new(); positions.len()];
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dist = (positions[j][0] - positions[i][0])
                .hypot(positions[j][1] - positions[i][1])
                .hypot(positions[j][2] - positions[i][2]);
            if dist < support_radius {
                lists[i].push(j);
                lists[j].push(i);
            }
        }
    }
    lists
}

/// Summation density at particle `index` (Theory Manual eq. 1248).
///
/// `rho_i = sum_j m_j W_ij`, including the self contribution `j = i`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::{Particle, summation_density};
///
/// let particles = vec![Particle::new([0.0; 3], [0.0; 3], 1.0).unwrap()];
/// let rho = summation_density(0, &particles, &[], 1.0, 3).unwrap();
/// assert!(rho > 0.0);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`] or [`SphError::BadDimension`]
/// from the kernel evaluation.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn summation_density(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    let home = particles[index].position;
    let mut rho = particles[index].mass * kernel_value(0.0, smoothing_length, dim)?;
    for &j in neighbors {
        let offset = [
            particles[j].position[0] - home[0],
            particles[j].position[1] - home[1],
            particles[j].position[2] - home[2],
        ];
        let radius = offset[0].hypot(offset[1]).hypot(offset[2]);
        rho += particles[j].mass * kernel_value(radius, smoothing_length, dim)?;
    }
    Ok(rho)
}

/// Continuity rate `d rho/dt` at particle `index`.
///
/// `drho_i/dt = sum_j m_j (v_i - v_j) . grad_i W_ij`, the differential form
/// evolved alongside (or instead of) the summation density.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::{Particle, continuity_rate};
///
/// let a = Particle::new([0.0; 3], [0.0; 3], 1.0).unwrap();
/// let b = Particle::new([0.5, 0.0, 0.0], [0.0; 3], 1.0).unwrap();
/// let particles = vec![a, b];
/// // Uniform velocity field: no compression.
/// assert_eq!(continuity_rate(0, &particles, &[1], 1.0, 3).unwrap(), 0.0);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`] or [`SphError::BadDimension`]
/// from the kernel evaluation.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn continuity_rate(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    let home = &particles[index];
    let mut rate = 0.0;
    for &j in neighbors {
        let other = &particles[j];
        // Offset points from neighbor to home, so `grad` is grad_i W_ij.
        let offset = [
            home.position[0] - other.position[0],
            home.position[1] - other.position[1],
            home.position[2] - other.position[2],
        ];
        let grad = kernel_gradient(offset, smoothing_length, dim)?;
        let dv = [
            home.velocity[0] - other.velocity[0],
            home.velocity[1] - other.velocity[1],
            home.velocity[2] - other.velocity[2],
        ];
        rate += other.mass * (dv[0] * grad[0] + dv[1] * grad[1] + dv[2] * grad[2]);
    }
    Ok(rate)
}

/// Monaghan pairwise artificial viscosity (Theory Manual eq. 1254-1255).
///
/// With `qa` the quadratic and `qb` the linear coefficient (deck values
/// `qa = 2`, `qb = 1` equal the manual defaults), approaching pairs gain the
/// inter-particle pressure `Pi = (-qb c_bar mu + qa mu^2) / rho_bar` where
/// `mu = h_bar (dv . dx) / (|dx|^2 + eps h_bar^2)`; receding pairs return 0.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::{Particle, artificial_viscosity};
///
/// let mut a = Particle::new([0.0; 3], [-1.0, 0.0, 0.0], 1.0).unwrap();
/// let mut b = Particle::new([0.5, 0.0, 0.0], [0.0; 3], 1.0).unwrap();
/// a.sound_speed = 1.0;
/// b.sound_speed = 1.0;
/// // Receding pair: no viscosity.
/// assert_eq!(artificial_viscosity(&a, &b, 1.0, 2.0, 1.0), 0.0);
/// ```
pub fn artificial_viscosity(
    home: &Particle,
    other: &Particle,
    smoothing_length: f64,
    qa: f64,
    qb: f64,
) -> f64 {
    let dx = [
        home.position[0] - other.position[0],
        home.position[1] - other.position[1],
        home.position[2] - other.position[2],
    ];
    let dv = [
        home.velocity[0] - other.velocity[0],
        home.velocity[1] - other.velocity[1],
        home.velocity[2] - other.velocity[2],
    ];
    let approach = dv[0] * dx[0] + dv[1] * dx[1] + dv[2] * dx[2];
    if approach >= 0.0 {
        return 0.0;
    }
    let dist_sq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
    let mu = smoothing_length * approach
        / (dist_sq + VISCOSITY_SOFTENING * smoothing_length * smoothing_length);
    let sound = 0.5 * (home.sound_speed + other.sound_speed);
    let density = 0.5 * (home.density + other.density);
    if density <= 0.0 {
        return 0.0;
    }
    (-qb * sound * mu + qa * mu * mu) / density
}

/// Momentum acceleration at particle `index` (Theory Manual eq. 1252).
///
/// Symmetric pressure form without kernel correction (deck `order = -1`):
/// `a_i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij) grad_i W_ij`.
/// Pass `qa = qb = 0` to disable viscosity, or scale it with the Balsara
/// factor from [`balsara_factor`] per pair.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::{Particle, momentum_acceleration};
///
/// let a = Particle::new([0.0; 3], [0.0; 3], 1.0).unwrap();
/// let b = Particle::new([0.5, 0.0, 0.0], [0.0; 3], 1.0).unwrap();
/// let particles = vec![a, b];
/// // Zero pressure and zero viscosity: no acceleration.
/// assert_eq!(
///     momentum_acceleration(0, &particles, &[1], 1.0, 3, 0.0, 0.0).unwrap(),
///     [0.0; 3]
/// );
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`], [`SphError::BadDimension`], or
/// [`SphError::BadState`] when a density is not positive.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn momentum_acceleration(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
    qa: f64,
    qb: f64,
) -> Result<[f64; 3], SphError> {
    momentum_acceleration_with_limiter(
        index,
        particles,
        neighbors,
        smoothing_length,
        dim,
        qa,
        qb,
        &[],
    )
}

/// Momentum acceleration with per-pair viscosity limiters.
///
/// `limiters` holds one Balsara-style factor per entry of `neighbors`
/// (empty means no limiting); each artificial-viscosity term is scaled by
/// its factor. See [`momentum_acceleration`] for the unscaled form.
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`], [`SphError::BadDimension`], or
/// [`SphError::BadState`] when a density is not positive or the limiter
/// count mismatches the neighbor count.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn momentum_acceleration_with_limiter(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
    qa: f64,
    qb: f64,
    limiters: &[f64],
) -> Result<[f64; 3], SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    if !limiters.is_empty() && limiters.len() != neighbors.len() {
        return Err(SphError::BadState(
            "limiter count must match neighbor count",
        ));
    }
    let home = &particles[index];
    if home.density <= 0.0 {
        return Err(SphError::BadState("density must be positive"));
    }
    let home_term = home.pressure / (home.density * home.density);
    let mut acceleration = [0.0; 3];
    for (slot, &j) in neighbors.iter().enumerate() {
        let other = &particles[j];
        if other.density <= 0.0 {
            return Err(SphError::BadState("density must be positive"));
        }
        // Offset points from neighbor to home, so `grad` is grad_i W_ij.
        let offset = [
            home.position[0] - other.position[0],
            home.position[1] - other.position[1],
            home.position[2] - other.position[2],
        ];
        let grad = kernel_gradient(offset, smoothing_length, dim)?;
        let limiter = if limiters.is_empty() {
            1.0
        } else {
            limiters[slot]
        };
        let viscosity = limiter * artificial_viscosity(home, other, smoothing_length, qa, qb);
        let scale = -other.mass
            * (home_term + other.pressure / (other.density * other.density) + viscosity);
        acceleration[0] += scale * grad[0];
        acceleration[1] += scale * grad[1];
        acceleration[2] += scale * grad[2];
    }
    Ok(acceleration)
}

/// Specific-energy rate at particle `index`.
///
/// Form compatible with [`momentum_acceleration`]:
/// `de_i/dt = sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij)/2
/// (v_i - v_j) . grad_i W_ij`, so pressure work plus viscous heating.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::{Particle, energy_rate};
///
/// let a = Particle::new([0.0; 3], [0.0; 3], 1.0).unwrap();
/// let b = Particle::new([0.5, 0.0, 0.0], [0.0; 3], 1.0).unwrap();
/// let particles = vec![a, b];
/// assert_eq!(energy_rate(0, &particles, &[1], 1.0, 3, 0.0, 0.0).unwrap(), 0.0);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`], [`SphError::BadDimension`], or
/// [`SphError::BadState`] when a density is not positive.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn energy_rate(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
    qa: f64,
    qb: f64,
) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    let home = &particles[index];
    if home.density <= 0.0 {
        return Err(SphError::BadState("density must be positive"));
    }
    let home_term = home.pressure / (home.density * home.density);
    let mut rate = 0.0;
    for &j in neighbors {
        let other = &particles[j];
        if other.density <= 0.0 {
            return Err(SphError::BadState("density must be positive"));
        }
        // Offset points from neighbor to home, so `grad` is grad_i W_ij.
        let offset = [
            home.position[0] - other.position[0],
            home.position[1] - other.position[1],
            home.position[2] - other.position[2],
        ];
        let grad = kernel_gradient(offset, smoothing_length, dim)?;
        let dv = [
            home.velocity[0] - other.velocity[0],
            home.velocity[1] - other.velocity[1],
            home.velocity[2] - other.velocity[2],
        ];
        let viscosity = artificial_viscosity(home, other, smoothing_length, qa, qb);
        let scale = 0.5
            * other.mass
            * (home_term + other.pressure / (other.density * other.density) + viscosity);
        rate += scale * (dv[0] * grad[0] + dv[1] * grad[1] + dv[2] * grad[2]);
    }
    Ok(rate)
}

/// Velocity divergence estimate at particle `index`.
///
/// `(div v)_i = sum_j (m_j / rho_i) (v_j - v_i) . grad_i W_ij`, the SPH
/// divergence used by the Balsara switch.
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`], [`SphError::BadDimension`], or
/// [`SphError::BadState`] when the home density is not positive.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn divergence_estimate(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    let home = &particles[index];
    if home.density <= 0.0 {
        return Err(SphError::BadState("density must be positive"));
    }
    let mut divergence = 0.0;
    for &j in neighbors {
        let other = &particles[j];
        // Offset points from neighbor to home, so `grad` is grad_i W_ij.
        let offset = [
            home.position[0] - other.position[0],
            home.position[1] - other.position[1],
            home.position[2] - other.position[2],
        ];
        let grad = kernel_gradient(offset, smoothing_length, dim)?;
        let dv = [
            other.velocity[0] - home.velocity[0],
            other.velocity[1] - home.velocity[1],
            other.velocity[2] - home.velocity[2],
        ];
        divergence += other.mass * (dv[0] * grad[0] + dv[1] * grad[1] + dv[2] * grad[2]);
    }
    Ok(divergence / home.density)
}

/// Velocity curl estimate at particle `index`.
///
/// `(curl v)_i = sum_j (m_j / rho_i) (v_i - v_j) x grad_i W_ij`.
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`], [`SphError::BadDimension`], or
/// [`SphError::BadState`] when the home density is not positive.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn curl_estimate(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
) -> Result<[f64; 3], SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    let home = &particles[index];
    if home.density <= 0.0 {
        return Err(SphError::BadState("density must be positive"));
    }
    let mut curl = [0.0; 3];
    for &j in neighbors {
        let other = &particles[j];
        // Offset points from neighbor to home, so `grad` is grad_i W_ij.
        let offset = [
            home.position[0] - other.position[0],
            home.position[1] - other.position[1],
            home.position[2] - other.position[2],
        ];
        let grad = kernel_gradient(offset, smoothing_length, dim)?;
        let dv = [
            home.velocity[0] - other.velocity[0],
            home.velocity[1] - other.velocity[1],
            home.velocity[2] - other.velocity[2],
        ];
        curl[0] += other.mass * (dv[1] * grad[2] - dv[2] * grad[1]);
        curl[1] += other.mass * (dv[2] * grad[0] - dv[0] * grad[2]);
        curl[2] += other.mass * (dv[0] * grad[1] - dv[1] * grad[0]);
    }
    Ok([
        curl[0] / home.density,
        curl[1] / home.density,
        curl[2] / home.density,
    ])
}

/// Balsara vorticity switch at one particle (Theory Manual eq. 1256-1257).
///
/// `f = |div v| / (|div v| + |curl v| + 1e-4 c / h)`: near one in shocks,
/// near zero in pure shear. Callers average the two pair values and scale
/// the artificial viscosity through [`momentum_acceleration_with_limiter`].
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::balsara_factor;
///
/// // Pure compression keeps full viscosity; pure shear suppresses it.
/// assert!(balsara_factor(2.0, 0.0, 1.0, 1.0).unwrap() > 0.99);
/// assert!(balsara_factor(0.0, 2.0, 1.0, 1.0).unwrap() < 0.01);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`] for non-positive `h` and
/// [`SphError::BadState`] for negative sound speed.
pub fn balsara_factor(
    divergence: f64,
    curl_norm: f64,
    sound_speed: f64,
    smoothing_length: f64,
) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    if sound_speed < 0.0 || !sound_speed.is_finite() {
        return Err(SphError::BadState(
            "sound speed must be finite and non-negative",
        ));
    }
    // Floor from Balsara (1995): keeps the switch well-defined in uniform
    // flow and scales with the local signal speed over h.
    let floor = 1e-4 * sound_speed / smoothing_length;
    Ok(divergence.abs() / (divergence.abs() + curl_norm.abs() + floor))
}

/// Polynomial equation of state (`/EOS/POLYNOMIAL`).
///
/// Pressure follows the volumetric strain `mu = rho / rho0 - 1`:
/// `p = C1 mu + C2 mu^2 + C3 mu^3`, with bulk sound speed squared
/// `c^2 = (C1 + 2 C2 mu + 3 C3 mu^2) / rho0`. The WavesCompression deck sets
/// `C1 = 1.3` and all other coefficients to zero (linear bulk response).
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialEos {
    /// Linear bulk coefficient `C1`.
    pub c1: f64,
    /// Quadratic bulk coefficient `C2`.
    pub c2: f64,
    /// Cubic bulk coefficient `C3`.
    pub c3: f64,
    /// Reference density `rho0`.
    pub rho0: f64,
}

impl PolynomialEos {
    /// Build the deck equation of state from its card values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::PolynomialEos;
    ///
    /// let eos = PolynomialEos::new(1.3, 0.0, 0.0, 8.93).unwrap();
    /// assert_eq!(eos.pressure(8.93), 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SphError::BadState`] when `rho0` is not positive and finite.
    pub fn new(c1: f64, c2: f64, c3: f64, rho0: f64) -> Result<Self, SphError> {
        if !rho0.is_finite() || rho0 <= 0.0 {
            return Err(SphError::BadState("reference density must be positive"));
        }
        Ok(Self { c1, c2, c3, rho0 })
    }

    /// Pressure at `density`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::PolynomialEos;
    ///
    /// let eos = PolynomialEos::new(1.3, 0.0, 0.0, 8.93).unwrap();
    /// assert!(eos.pressure(9.5) > 0.0);
    /// ```
    pub fn pressure(&self, density: f64) -> f64 {
        let mu = density / self.rho0 - 1.0;
        self.c1 * mu + self.c2 * mu * mu + self.c3 * mu * mu * mu
    }

    /// Bulk sound speed at `density` (zero when the slope is negative).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::PolynomialEos;
    ///
    /// let eos = PolynomialEos::new(1.3, 0.0, 0.0, 8.93).unwrap();
    /// assert!((eos.sound_speed(8.93) - (1.3f64 / 8.93).sqrt()).abs() < 1e-15);
    /// ```
    pub fn sound_speed(&self, density: f64) -> f64 {
        let mu = density / self.rho0 - 1.0;
        let slope = self.c1 + 2.0 * self.c2 * mu + 3.0 * self.c3 * mu * mu;
        (slope / self.rho0).max(0.0).sqrt()
    }
}

/// Adapted smoothing length keeping neighbor count stable (eq. 1253).
///
/// `h = h0 (rho0 / rho)^(1/dim)`: compression shrinks the support so each
/// particle keeps roughly the same neighbor count over time.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::updated_smoothing_length;
///
/// let h = updated_smoothing_length(1.2, 8.93, 9.5, 3).unwrap();
/// assert!(h < 1.2);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`] for non-positive `h0`,
/// [`SphError::BadDimension`] unless `dim` is 1, 2, or 3, and
/// [`SphError::BadState`] for non-positive densities.
pub fn updated_smoothing_length(
    h0: f64,
    rho0: f64,
    density: f64,
    dim: usize,
) -> Result<f64, SphError> {
    check_smoothing_length(h0)?;
    kernel_normalization(dim)?;
    if rho0 <= 0.0 || density <= 0.0 || !rho0.is_finite() || !density.is_finite() {
        return Err(SphError::BadState("densities must be positive and finite"));
    }
    Ok(h0 * (rho0 / density).powf(1.0 / dim as f64))
}

/// Ghost-particle symmetry condition (`/SPHBCS` types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryCondition {
    /// Material slides along the plane (normal velocity mirrors).
    Slide,
    /// Material cannot slide (ghost velocity opposes the particle).
    Tied,
}

/// Symmetry plane generating ghost particles (`/SPHBCS`).
///
/// Ghost particles mirror real particles across the plane (Reference Guide
/// `/SPHBCS` comment 1); particles must lie in the positive half-space.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::{SymmetryCondition, SymmetryPlane};
///
/// let plane = SymmetryPlane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
/// assert_eq!(plane.reflect_position([1.0, 2.0, 3.0]), [1.0, 2.0, -3.0]);
/// assert_eq!(
///     plane.ghost_velocity([1.0, 0.0, 2.0], SymmetryCondition::Slide),
///     [1.0, 0.0, -2.0]
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetryPlane {
    point: [f64; 3],
    normal: [f64; 3],
}

impl SymmetryPlane {
    /// Build a plane through `point` with unit direction `normal`.
    ///
    /// # Errors
    ///
    /// Returns [`SphError::BadPlane`] when `normal` is zero or not finite.
    pub fn new(point: [f64; 3], normal: [f64; 3]) -> Result<Self, SphError> {
        let norm = normal[0].hypot(normal[1]).hypot(normal[2]);
        if !norm.is_finite() || norm == 0.0 {
            return Err(SphError::BadPlane);
        }
        Ok(Self {
            point,
            normal: [normal[0] / norm, normal[1] / norm, normal[2] / norm],
        })
    }

    /// Mirror a position across the plane.
    pub fn reflect_position(&self, position: [f64; 3]) -> [f64; 3] {
        let d = (position[0] - self.point[0]) * self.normal[0]
            + (position[1] - self.point[1]) * self.normal[1]
            + (position[2] - self.point[2]) * self.normal[2];
        [
            position[0] - 2.0 * d * self.normal[0],
            position[1] - 2.0 * d * self.normal[1],
            position[2] - 2.0 * d * self.normal[2],
        ]
    }

    /// Ghost velocity for `condition` (`SLIDE` mirrors, `TIED` opposes).
    pub fn ghost_velocity(&self, velocity: [f64; 3], condition: SymmetryCondition) -> [f64; 3] {
        match condition {
            SymmetryCondition::Slide => {
                let vn = velocity[0] * self.normal[0]
                    + velocity[1] * self.normal[1]
                    + velocity[2] * self.normal[2];
                [
                    velocity[0] - 2.0 * vn * self.normal[0],
                    velocity[1] - 2.0 * vn * self.normal[1],
                    velocity[2] - 2.0 * vn * self.normal[2],
                ]
            }
            SymmetryCondition::Tied => [-velocity[0], -velocity[1], -velocity[2]],
        }
    }
}

/// Stable explicit step (Theory Manual eq. 1258 narrative).
///
/// `dt = cfl * h / (c + signal)` with sound speed `c` and extra signal speed
/// `signal` (viscous or relative-velocity contribution, non-negative).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::sph::{DEFAULT_CFL, stable_time_step};
///
/// let dt = stable_time_step(DEFAULT_CFL, 1.0, 1.3, 0.0).unwrap();
/// assert!((dt - 0.3 / 1.3).abs() < 1e-15);
/// ```
///
/// # Errors
///
/// Returns [`SphError::BadCfl`] unless `cfl` lies in `(0, 1]`,
/// [`SphError::BadSmoothingLength`] for non-positive `h`, and
/// [`SphError::BadState`] for negative speeds or a non-positive total.
pub fn stable_time_step(cfl: f64, h: f64, sound_speed: f64, signal: f64) -> Result<f64, SphError> {
    if !cfl.is_finite() || cfl <= 0.0 || cfl > 1.0 {
        return Err(SphError::BadCfl(cfl));
    }
    check_smoothing_length(h)?;
    if sound_speed < 0.0 || signal < 0.0 || !sound_speed.is_finite() || !signal.is_finite() {
        return Err(SphError::BadState("speeds must be finite and non-negative"));
    }
    if sound_speed + signal <= 0.0 {
        return Err(SphError::BadState("total signal speed must be positive"));
    }
    Ok(cfl * h / (sound_speed + signal))
}

/// Conservatively smoothed velocity at particle `index` (eq. 1270).
///
/// `v_hat_i = v_i + eps sum_j (m_j / rho_bar_ij) (v_j - v_i) W_ij`: the
/// Balsara velocity filter against tensile instability. The deck disables it
/// (`alpha_cs = 0`); pass `eps = 0` to reproduce the deck exactly.
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`], [`SphError::BadDimension`], or
/// [`SphError::BadState`] when a density is not positive.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range.
pub fn smoothed_velocity(
    index: usize,
    particles: &[Particle],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
    eps: f64,
) -> Result<[f64; 3], SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    let home = &particles[index];
    if home.density <= 0.0 {
        return Err(SphError::BadState("density must be positive"));
    }
    let mut smoothed = home.velocity;
    for &j in neighbors {
        let other = &particles[j];
        if other.density <= 0.0 {
            return Err(SphError::BadState("density must be positive"));
        }
        let offset = [
            other.position[0] - home.position[0],
            other.position[1] - home.position[1],
            other.position[2] - home.position[2],
        ];
        let radius = offset[0].hypot(offset[1]).hypot(offset[2]);
        let weight = eps * other.mass / (0.5 * (home.density + other.density))
            * kernel_value(radius, smoothing_length, dim)?;
        smoothed[0] += weight * (other.velocity[0] - home.velocity[0]);
        smoothed[1] += weight * (other.velocity[1] - home.velocity[1]);
        smoothed[2] += weight * (other.velocity[2] - home.velocity[2]);
    }
    Ok(smoothed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(dx: f64, vx: f64) -> Vec<Particle> {
        vec![
            Particle::new([0.0, 0.0, 0.0], [vx, 0.0, 0.0], 1.0).unwrap(),
            Particle::new([dx, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0).unwrap(),
        ]
    }

    #[test]
    fn kernel_shape_matches_branches() {
        assert_eq!(cubic_spline_shape(0.0), 1.0);
        assert!((cubic_spline_shape(1.0) - 0.25).abs() < 1e-15);
        assert_eq!(cubic_spline_shape(2.0), 0.0);
        assert_eq!(cubic_spline_shape(-1.0), 0.0);
        // Branch continuity at q = 1 from both sides.
        let left = 1.0 - 1.5 * 0.9999f64.powi(2) + 0.75 * 0.9999f64.powi(3);
        assert!((cubic_spline_shape(0.9999) - left).abs() < 1e-15);
        assert!((cubic_spline_shape(1.0001) - 0.25 * 0.9999f64.powi(3)).abs() < 1e-15);
    }

    #[test]
    fn kernel_gradient_is_antisymmetric_and_compact() {
        let forward = kernel_gradient([0.7, 0.0, 0.0], 1.0, 3).unwrap();
        let backward = kernel_gradient([-0.7, 0.0, 0.0], 1.0, 3).unwrap();
        assert!((forward[0] + backward[0]).abs() < 1e-15);
        assert_eq!(kernel_gradient([2.5, 0.0, 0.0], 1.0, 3).unwrap(), [0.0; 3]);
        assert_eq!(kernel_gradient([0.0; 3], 1.0, 3).unwrap(), [0.0; 3]);
    }

    #[test]
    fn kernel_normalizes_over_ball() {
        // Midpoint quadrature of W over |x| < 2h in 3D must approach 1.
        let h = 1.0f64;
        let n = 160usize;
        let edge = 4.0 * h;
        let dx = edge / n as f64;
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = -2.0 * h + (i as f64 + 0.5) * dx;
                    let y = -2.0 * h + (j as f64 + 0.5) * dx;
                    let z = -2.0 * h + (k as f64 + 0.5) * dx;
                    let r = x.hypot(y).hypot(z);
                    sum += kernel_value(r, h, 3).unwrap();
                }
            }
        }
        assert!(
            (sum * dx * dx * dx - 1.0).abs() < 1e-3,
            "integral = {}",
            sum * dx * dx * dx
        );
    }

    #[test]
    fn grid_search_matches_brute_force() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.2, -0.1],
            [1.9, 0.0, 0.0],
            [-1.5, 1.5, 0.0],
            [3.0, 0.0, 0.0],
            [0.1, -0.1, 0.15],
        ];
        let search = NeighborSearch::build(&positions, 2.0).unwrap();
        let brute = brute_force_neighbors(&positions, 2.0);
        for (i, expected) in brute.iter().enumerate() {
            assert_eq!(search.neighbors(i), *expected, "particle {i}");
        }
    }

    #[test]
    fn summation_density_recovers_uniform_lattice() {
        // Cubic lattice, spacing 1, mass 1: interior density approaches 1.
        let mut particles = Vec::new();
        for i in -3..=3 {
            for j in -3..=3 {
                for k in -3..=3 {
                    particles.push(
                        Particle::new([i as f64, j as f64, k as f64], [0.0; 3], 1.0).unwrap(),
                    );
                }
            }
        }
        let positions: Vec<[f64; 3]> = particles.iter().map(|p| p.position).collect();
        let search = NeighborSearch::build(&positions, 2.0 * 1.2).unwrap();
        let center = particles.len() / 2;
        let rho = summation_density(center, &particles, &search.neighbors(center), 1.2, 3).unwrap();
        assert!((rho - 1.0).abs() < 0.05, "rho = {rho}");
    }

    #[test]
    fn uniform_pressure_exerts_no_force_interior() {
        // Symmetric neighbor pair with equal pressure: forces cancel exactly.
        let mut particles = vec![
            Particle::new([0.0, 0.0, 0.0], [0.0; 3], 1.0).unwrap(),
            Particle::new([0.5, 0.0, 0.0], [0.0; 3], 1.0).unwrap(),
            Particle::new([-0.5, 0.0, 0.0], [0.0; 3], 1.0).unwrap(),
        ];
        for p in &mut particles {
            p.pressure = 2.0;
        }
        let accel = momentum_acceleration(0, &particles, &[1, 2], 1.0, 3, 0.0, 0.0).unwrap();
        assert!(accel[0].abs() < 1e-15 && accel[1] == 0.0 && accel[2] == 0.0);
    }

    #[test]
    fn pair_forces_conserve_momentum() {
        let mut particles = pair(0.6, 0.3);
        particles[0].pressure = 1.5;
        particles[1].pressure = 0.5;
        particles[0].sound_speed = 1.0;
        particles[1].sound_speed = 1.0;
        let a0 = momentum_acceleration(0, &particles, &[1], 1.0, 3, 2.0, 1.0).unwrap();
        let a1 = momentum_acceleration(1, &particles, &[0], 1.0, 3, 2.0, 1.0).unwrap();
        // Equal masses: m*a sums to zero across the antisymmetric pair.
        assert!((a0[0] + a1[0]).abs() < 1e-14, "{a0:?} {a1:?}");
        assert_eq!(a0[1] + a1[1], 0.0);
        assert_eq!(a0[2] + a1[2], 0.0);
    }

    #[test]
    fn viscosity_heats_approaching_pairs_only() {
        // Home particle moves toward the neighbor at +x: approaching.
        let mut approaching = pair(0.6, 0.5);
        for p in &mut approaching {
            p.sound_speed = 1.0;
        }
        let heat = energy_rate(0, &approaching, &[1], 1.0, 3, 2.0, 1.0).unwrap();
        assert!(heat > 0.0, "heat = {heat}");

        let mut receding = pair(0.6, -0.5);
        for p in &mut receding {
            p.sound_speed = 1.0;
        }
        assert_eq!(
            energy_rate(0, &receding, &[1], 1.0, 3, 2.0, 1.0).unwrap(),
            0.0
        );
    }

    #[test]
    fn limiter_mismatch_is_rejected() {
        let particles = pair(0.5, 0.0);
        let err =
            momentum_acceleration_with_limiter(0, &particles, &[1], 1.0, 3, 0.0, 0.0, &[1.0, 0.5])
                .unwrap_err();
        assert_eq!(
            err,
            SphError::BadState("limiter count must match neighbor count")
        );
    }

    #[test]
    fn polynomial_eos_matches_deck_linear_response() {
        let eos = PolynomialEos::new(1.3, 0.0, 0.0, 8.93).unwrap();
        assert_eq!(eos.pressure(8.93), 0.0);
        assert!((eos.pressure(9.5) - 1.3 * (9.5 / 8.93 - 1.0)).abs() < 1e-15);
        assert!(eos.pressure(8.0) < 0.0);
    }

    #[test]
    fn smoothing_length_shrinks_under_compression() {
        let h = updated_smoothing_length(1.2, 8.93, 9.5, 3).unwrap();
        assert!((h - 1.2 * (8.93f64 / 9.5).powf(1.0 / 3.0)).abs() < 1e-15);
        let grown = updated_smoothing_length(1.2, 8.93, 8.0, 3).unwrap();
        assert!(grown > 1.2);
    }

    #[test]
    fn reflection_is_an_involution() {
        let plane = SymmetryPlane::new([0.0, 0.0, 1.0], [0.0, 0.0, 2.0]).unwrap();
        let pointed = [1.0, -2.0, 4.0];
        let mirrored = plane.reflect_position(pointed);
        assert_eq!(mirrored, [1.0, -2.0, -2.0]);
        assert_eq!(plane.reflect_position(mirrored), pointed);
        assert_eq!(
            plane.ghost_velocity([1.0, 2.0, 3.0], SymmetryCondition::Tied),
            [-1.0, -2.0, -3.0]
        );
    }

    #[test]
    fn step_validation_rejects_bad_inputs() {
        assert_eq!(
            stable_time_step(0.0, 1.0, 1.0, 0.0),
            Err(SphError::BadCfl(0.0))
        );
        assert_eq!(
            stable_time_step(0.3, -1.0, 1.0, 0.0),
            Err(SphError::BadSmoothingLength(-1.0))
        );
        assert!(stable_time_step(0.3, 1.0, 0.0, 0.0).is_err());
        assert!(kernel_value(0.0, 1.0, 4).is_err());
        assert!(SymmetryPlane::new([0.0; 3], [0.0; 3]).is_err());
        assert!(Particle::new([0.0; 3], [0.0; 3], 0.0).is_err());
    }

    #[test]
    fn pressure_pushes_pair_apart() {
        let mut particles = pair(0.6, 0.0);
        particles[0].pressure = 1.0;
        particles[1].pressure = 1.0;
        let a0 = momentum_acceleration(0, &particles, &[1], 1.0, 3, 0.0, 0.0).unwrap();
        // Home particle sits left of its neighbor: pushed further left.
        assert!(a0[0] < 0.0, "{a0:?}");
    }

    #[test]
    fn compression_raises_density_rate() {
        let approaching = pair(0.6, 0.5);
        let rate = continuity_rate(0, &approaching, &[1], 1.0, 3).unwrap();
        assert!(rate > 0.0, "rate = {rate}");
    }

    #[test]
    fn rigid_rotation_curl_matches_analytic() {
        // Rigid rotation v = omega x r has curl exactly 2 omega.
        let mut particles = Vec::new();
        for i in -3..=3 {
            for j in -3..=3 {
                for k in -3..=3 {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    particles.push(Particle::new([x, y, z], [-y, x, 0.0], 1.0).unwrap());
                }
            }
        }
        let positions: Vec<[f64; 3]> = particles.iter().map(|p| p.position).collect();
        let search = NeighborSearch::build(&positions, 2.0 * 1.2).unwrap();
        let center = particles.len() / 2;
        let curl = curl_estimate(center, &particles, &search.neighbors(center), 1.2, 3).unwrap();
        assert!((curl[2] - 2.0).abs() < 0.15, "{curl:?}");
        assert!(curl[0].abs() < 0.15 && curl[1].abs() < 0.15);
        let div =
            divergence_estimate(center, &particles, &search.neighbors(center), 1.2, 3).unwrap();
        assert!(div.abs() < 0.15, "div = {div}");
    }

    #[test]
    fn smoothing_with_zero_eps_is_identity() {
        let particles = pair(0.5, 1.0);
        let smoothed = smoothed_velocity(0, &particles, &[1], 1.0, 3, 0.0).unwrap();
        assert_eq!(smoothed, particles[0].velocity);
    }
}
