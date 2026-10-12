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
//! symmetry boundaries, and explicit time step control. Phase 7 lane 7D
//! adds the native [`CompressionRun`] driver plus [`Series`] output in
//! TH-CSV shape: `qa-fem-sph --series` emits a native compression run,
//! while `qa-fem-sph --replay` compares a provided engine series against
//! the harvested golden at vendor tolerance. The native series comes
//! from an independent integrator with its own documented invariant
//! bands; it is not claimed at vendor tolerance.
//!
//! W-tier fixtures: `fixtures/sph_kernel_consistency` (kernel values,
//! partition of unity, gradient accuracy) and `fixtures/sph_waves`
//! (one-step compression-wave operators with deck viscosity coefficients).

// Rust guideline compliant 2026-02-21

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

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

/// Softening in the Brookshaw denominator as a fraction of `h^2`.
///
/// Standard Monaghan value; keeps the quotient finite when two
/// particles coincide while leaving well-separated pairs unaffected.
pub const BROOKSHAW_SOFTENING: f64 = 0.01;

/// Brookshaw Laplacian estimate of a scalar field at particle `index`.
///
/// `lap_i = 2 sum_j (m_j / rho_j) (A_i - A_j) F_ij` with
/// `F_ij = (r_ij . grad_i W_ij) / (r_ij^2 + eta^2)` and
/// `eta^2 = 0.01 h^2`, built on [`kernel_gradient_factor`] via
/// `r . grad W = r^2 (dW/dr)/r`. The factor 2 is the Cleary-Monaghan
/// normalization: the raw sum converges to half the Laplacian (the
/// symmetric second moment of `(1/r) dW/dr` is `-delta`). Brookshaw,
/// "A method of calculating radiative heat diffusion in particle
/// simulations", MNRAS 207 (1985), 191-214; Cleary and Monaghan,
/// "Conduction modelling using smoothed particle hydrodynamics",
/// JCP 148 (1999), 227-264.
///
/// # Errors
///
/// Returns [`SphError::BadSmoothingLength`], [`SphError::BadDimension`],
/// or [`SphError::BadState`] when the home or a neighbor density is
/// not positive.
///
/// # Panics
///
/// Panics when `index` or a neighbor entry is out of range, or when
/// `field` does not hold one value per particle.
pub fn brookshaw_laplacian(
    index: usize,
    particles: &[Particle],
    field: &[f64],
    neighbors: &[usize],
    smoothing_length: f64,
    dim: usize,
) -> Result<f64, SphError> {
    check_smoothing_length(smoothing_length)?;
    kernel_normalization(dim)?;
    assert_eq!(
        field.len(),
        particles.len(),
        "field must hold one value per particle"
    );
    let home = &particles[index];
    if home.density <= 0.0 {
        return Err(SphError::BadState("density must be positive"));
    }
    let softening = BROOKSHAW_SOFTENING * smoothing_length * smoothing_length;
    let mut laplacian = 0.0;
    for &j in neighbors {
        let other = &particles[j];
        if other.density <= 0.0 {
            return Err(SphError::BadState("density must be positive"));
        }
        let dx = home.position[0] - other.position[0];
        let dy = home.position[1] - other.position[1];
        let dz = home.position[2] - other.position[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let radius = r2.sqrt();
        // Dot product via the radial factor, finite at r = 0.
        let dot = r2 * kernel_gradient_factor(radius, smoothing_length, dim)?;
        let weight = other.mass / other.density * (field[index] - field[j]);
        laplacian += weight * dot / (r2 + softening);
    }
    Ok(2.0 * laplacian)
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

    /// Returns the plane reference point.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::SymmetryPlane;
    ///
    /// let plane = SymmetryPlane::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]).unwrap();
    /// assert_eq!(plane.point(), [1.0, 2.0, 3.0]);
    /// ```
    pub fn point(&self) -> [f64; 3] {
        self.point
    }

    /// Returns the unit plane normal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::SymmetryPlane;
    ///
    /// let plane = SymmetryPlane::new([0.0; 3], [0.0, 0.0, 2.0]).unwrap();
    /// assert_eq!(plane.normal(), [0.0, 0.0, 1.0]);
    /// ```
    pub fn normal(&self) -> [f64; 3] {
        self.normal
    }

    /// Returns the signed distance from the plane along its normal.
    ///
    /// Positive on the normal side, negative behind it; the deck
    /// symmetry planes face into the material, so contained particles
    /// read non-negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::SymmetryPlane;
    ///
    /// let plane = SymmetryPlane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
    /// assert_eq!(plane.signed_distance([0.0, 0.0, 2.0]), 2.0);
    /// assert_eq!(plane.signed_distance([0.0, 0.0, -1.0]), -1.0);
    /// ```
    pub fn signed_distance(&self, position: [f64; 3]) -> f64 {
        (position[0] - self.point[0]) * self.normal[0]
            + (position[1] - self.point[1]) * self.normal[1]
            + (position[2] - self.point[2]) * self.normal[2]
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

/// Copper reference density of the deck (`/MAT/HYDPLA` `RHO_0`).
///
/// Deck value in kg-mm-ms units; the native run derives particle mass
/// from this density times the lattice cell volume.
pub const WAVES_RHO0: f64 = 8.93;

/// Linear bulk coefficient of the deck (`/EOS/POLYNOMIAL` `C1`).
///
/// Deck literal (float32 spelling of 1.3); all other polynomial
/// coefficients are zero.
pub const WAVES_EOS_C1: f64 = 1.29999995;

/// Minimum pressure (`/MAT/HYDPLA` `Pmin`, deck literal).
///
/// Equation-of-state pressures clip at this floor, so tension never
/// pulls: free-surface and edge particles with under-supported
/// summation densities rest at zero pressure instead of collapsing
/// inward. Sound speeds keep the unclipped bulk slope.
pub const WAVES_PMIN: f64 = 0.0;

/// Quadratic artificial-viscosity coefficient (`/PROP/TYPE34` `qa`).
pub const WAVES_QA: f64 = 2.0;

/// Linear artificial-viscosity coefficient (`/PROP/TYPE34` `qb`).
pub const WAVES_QB: f64 = 1.0;

/// Initial downward velocity (`/INIVEL/TRA` `Vz`, deck literal).
pub const WAVES_VZ: f64 = -0.0381500006;

/// Explicit run length (`/RUN/CUBC/1`, deck time units).
pub const WAVES_END_TIME: f64 = 30.1;

/// Cell count of the native impact column.
///
/// The deck resolves the bar height with 32 planes; the native
/// column doubles that axial resolution (64 cells over the same
/// span) so the compression front spans several particles.
pub const WAVES_COLUMN_CELLS: usize = 64;

/// Minimum corner of the deck particle box (kg-mm-ms length units).
///
/// The native column spans the box `z` extent; the `x`/`y` extents
/// set the column cross-section for extensive probes.
pub const WAVES_BOX_MIN: [f64; 3] = [-1.283001, -1.283001, 0.0];

/// Maximum corner of the deck particle box (kg-mm-ms length units).
pub const WAVES_BOX_MAX: [f64; 3] = [1.283001, 1.283001, 9.943255];

/// Smoothing length as a multiple of the column spacing.
///
/// At `1.3` spacings the one-dimensional kernel sum over the column
/// lattice reads unity within 1%, with about five neighbors in
/// support. Raising it damps the impact pulse; lowering it toward
/// 1.0 noisifies the summation density.
pub const WAVES_SMOOTHING_FACTOR: f64 = 1.3;

/// Stamped channel set in TH-CSV shape.
///
/// A [`CompressionRun`] fills one of these per run; [`Series::to_csv`]
/// renders the vendor time-history layout (a `Time` column plus one
/// column per channel) at full roundtrip precision. The text parses
/// through the golden replay CSV reader, so native series overlay the
/// harvested goldens column by column.
#[derive(Debug, Clone, Default)]
pub struct Series {
    /// Stamp times in deck time units, strictly increasing.
    pub times: Vec<f64>,
    /// Channel name to one sample per stamp.
    pub channels: BTreeMap<String, Vec<f64>>,
    order: Vec<String>,
}

impl Series {
    /// Builds an empty series with fixed channel order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::Series;
    ///
    /// let mut series = Series::with_channels(&["MASS", "Z-MOMENTUM"]);
    /// series.push(0.0, &[1.0, 2.0]);
    /// assert_eq!(series.len(), 1);
    /// assert_eq!(series.channel("MASS"), Some(&[1.0][..]));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when a channel name repeats; callers pass distinct
    /// probe names by construction.
    pub fn with_channels(names: &[&str]) -> Self {
        let channels: BTreeMap<String, Vec<f64>> = names
            .iter()
            .map(|name| ((*name).to_string(), Vec::new()))
            .collect();
        assert_eq!(
            channels.len(),
            names.len(),
            "series channels must be distinct"
        );
        Self {
            times: Vec::new(),
            channels,
            order: names.iter().map(|name| (*name).to_string()).collect(),
        }
    }

    /// Appends one stamp with a value per channel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::Series;
    ///
    /// let mut series = Series::with_channels(&["MASS"]);
    /// series.push(0.5, &[3.0]);
    /// assert_eq!(series.times, vec![0.5]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `values` does not hold exactly one entry per
    /// channel; callers pass full probe rows by construction.
    pub fn push(&mut self, time: f64, values: &[f64]) {
        assert_eq!(
            values.len(),
            self.order.len(),
            "series stamp needs one value per channel"
        );
        self.times.push(time);
        for (name, value) in self.order.iter().zip(values.iter()) {
            self.channels
                .get_mut(name)
                .expect("series channel registered at build")
                .push(*value);
        }
    }

    /// Returns the stamp count.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Reports whether the series holds no stamps.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Looks up a channel by name.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::Series;
    ///
    /// let series = Series::with_channels(&["MASS"]);
    /// assert!(series.channel("Z-MOMENTUM").is_none());
    /// ```
    pub fn channel(&self, name: &str) -> Option<&[f64]> {
        self.channels.get(name).map(Vec::as_slice)
    }

    /// Renders the series as TH-CSV text.
    ///
    /// The header starts with a `Time` column followed by the channels
    /// in build order; every value prints at full roundtrip precision
    /// with no trailing comma, matching the replay reader's shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::Series;
    ///
    /// let mut series = Series::with_channels(&["MASS"]);
    /// series.push(0.0, &[1.5]);
    /// let text = series.to_csv();
    /// assert!(text.starts_with("Time,MASS\n"));
    /// assert_eq!(text.lines().count(), 2);
    /// ```
    pub fn to_csv(&self) -> String {
        let mut text = String::from("Time");
        for name in &self.order {
            text.push(',');
            text.push_str(name);
        }
        text.push('\n');
        for (stamp, time) in self.times.iter().enumerate() {
            text.push_str(&format!("{time}"));
            for name in &self.order {
                let value = self.channels[name][stamp];
                text.push_str(&format!(",{value}"));
            }
            text.push('\n');
        }
        text
    }

    /// Writes [`Series::to_csv`] output to `path`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::Series;
    ///
    /// let mut series = Series::with_channels(&["MASS"]);
    /// series.push(0.0, &[1.5]);
    /// let path = std::env::temp_dir().join("sph-series-doctest.csv");
    /// series.write_csv(&path).unwrap();
    /// assert!(std::fs::read_to_string(&path).unwrap().starts_with("Time,MASS\n"));
    /// std::fs::remove_file(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the underlying I/O error when the file cannot be
    /// written.
    pub fn write_csv(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_csv())
    }
}

/// Native SPH compression run mirroring WavesCompression.
///
/// A kick-drift-kick leapfrog integrator over a one-dimensional
/// column with the deck `/PROP/TYPE34` viscosity, the deck
/// polynomial equation of state with the deck `Pmin` pressure floor,
/// and one `TIED` ghost-particle floor at `z = 0` matching the deck
/// `/SPHBCS` card. The reduction is the laterally uniform impact the
/// deck's own four `SLIDE` symmetry planes enforce: transverse
/// motion stays exactly zero, so the column carries the full
/// compression-wave physics (shock formation, wall rebound, kinetic
/// to internal conversion) without lateral boundary artifacts.
/// Ghosts closer than the kernel support mirror live particles every
/// evaluation; ghosts exactly coincident with their source are
/// skipped because they carry no gradient and would double-count the
/// self density. The smoothing length stays fixed (the impact
/// compression is only a few percent); energy advances by the
/// trapezoid of the compatible [`energy_rate`].
///
/// Mass per unit area convention: with the one-dimensional kernel,
/// particle `mass` stores reference density times column spacing, so
/// summation densities come out as true volumetric densities and
/// every extensive probe scales by the deck cross-section. Viscosity
/// runs raw (`qa`/`qb` unscaled): the pair limiter would need an
/// energy-rate twin to stay adjoint-consistent, and the deck impact
/// is compression-dominated, where the switch reads near one anyway.
/// See [`balsara_factor`] for the available switch.
///
/// All quantities use the deck unit system (kg-mm-ms).
///
/// Boundary containment runs after every drift: particles behind the
/// floor reflect back inside with the `TIED` condition velocity (the
/// deck `Ilev = 1` elastic rebound). The deck-side mass accounting
/// then matches to 2.2%: the native total is exactly reference
/// density times deck box volume, while the golden MASS reads
/// 597.9077 (the starter log evidences an undocumented engine-side
/// renormalization of the nominal particle mass).
#[derive(Debug, Clone)]
pub struct CompressionRun {
    particles: Vec<Particle>,
    eos: PolynomialEos,
    planes: Vec<(SymmetryPlane, SymmetryCondition)>,
    qa: f64,
    qb: f64,
    cfl: f64,
    h: f64,
    area: f64,
    time: f64,
    center: usize,
}

/// One operator evaluation over the ghost-completed system.
struct Evaluation {
    accelerations: Vec<[f64; 3]>,
    rates: Vec<f64>,
    densities: Vec<f64>,
    sounds: Vec<f64>,
}

impl CompressionRun {
    /// Builds the deck-mirror run at `t = 0`.
    ///
    /// Particles start cell-centered on the deck-height column with
    /// uniform downward velocity [`WAVES_VZ`], zero internal energy,
    /// and summation-density thermodynamic state; the center probe
    /// tracks the particle nearest mid-height.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::{CompressionRun, WAVES_COLUMN_CELLS};
    ///
    /// let run = CompressionRun::waves_compression().unwrap();
    /// assert_eq!(run.particles().len(), WAVES_COLUMN_CELLS);
    /// assert_eq!(run.time(), 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SphError`] when the deck constants violate a kernel
    /// contract (unreachable for the checked-in values).
    pub fn waves_compression() -> Result<Self, SphError> {
        let eos = PolynomialEos::new(WAVES_EOS_C1, 0.0, 0.0, WAVES_RHO0)?;
        let height = WAVES_BOX_MAX[2] - WAVES_BOX_MIN[2];
        let spacing = height / WAVES_COLUMN_CELLS as f64;
        let area = (WAVES_BOX_MAX[0] - WAVES_BOX_MIN[0]) * (WAVES_BOX_MAX[1] - WAVES_BOX_MIN[1]);
        let mut particles = Vec::with_capacity(WAVES_COLUMN_CELLS);
        for cell in 0..WAVES_COLUMN_CELLS {
            particles.push(Particle::new(
                [0.0, 0.0, WAVES_BOX_MIN[2] + (cell as f64 + 0.5) * spacing],
                [0.0, 0.0, WAVES_VZ],
                WAVES_RHO0 * spacing,
            )?);
        }
        let h = WAVES_SMOOTHING_FACTOR * spacing;
        let planes = vec![(
            SymmetryPlane::new([0.0, 0.0, WAVES_BOX_MIN[2]], [0.0, 0.0, 1.0])?,
            SymmetryCondition::Tied,
        )];
        let mid = 0.5 * (WAVES_BOX_MIN[2] + WAVES_BOX_MAX[2]);
        let mut center = 0;
        let mut best = f64::INFINITY;
        for (index, particle) in particles.iter().enumerate() {
            let gap = (particle.position[2] - mid).abs();
            if gap < best {
                best = gap;
                center = index;
            }
        }
        let mut run = Self {
            particles,
            eos,
            planes,
            qa: WAVES_QA,
            qb: WAVES_QB,
            cfl: DEFAULT_CFL,
            h,
            area,
            time: 0.0,
            center,
        };
        run.refresh()?;
        Ok(run)
    }

    /// Returns the live particles.
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Returns the current time in deck time units.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns the current smoothing length.
    pub fn smoothing_length(&self) -> f64 {
        self.h
    }

    /// Returns the center-probe particle index.
    pub fn center_index(&self) -> usize {
        self.center
    }

    /// Returns the column cross-section in deck area units.
    pub fn cross_section(&self) -> f64 {
        self.area
    }

    /// Returns the total mass (exactly conserved: no mass update).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::CompressionRun;
    ///
    /// let run = CompressionRun::waves_compression().unwrap();
    /// assert!(run.total_mass() > 500.0);
    /// ```
    pub fn total_mass(&self) -> f64 {
        self.area * self.particles.iter().map(|p| p.mass).sum::<f64>()
    }

    /// Returns the total internal energy `area * sum m e`.
    pub fn internal_energy(&self) -> f64 {
        self.area
            * self
                .particles
                .iter()
                .map(|p| p.mass * p.energy)
                .sum::<f64>()
    }

    /// Returns the total kinetic energy `area * sum m v^2 / 2`.
    pub fn kinetic_energy(&self) -> f64 {
        self.area
            * self
                .particles
                .iter()
                .map(|p| {
                    0.5 * p.mass
                        * (p.velocity[0] * p.velocity[0]
                            + p.velocity[1] * p.velocity[1]
                            + p.velocity[2] * p.velocity[2])
                })
                .sum::<f64>()
    }

    /// Returns the total momentum `area * sum m v`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::{CompressionRun, WAVES_VZ};
    ///
    /// let run = CompressionRun::waves_compression().unwrap();
    /// let momentum = run.momentum();
    /// assert_eq!(momentum[0], 0.0);
    /// assert_eq!(momentum[1], 0.0);
    /// assert!((momentum[2] - run.total_mass() * WAVES_VZ).abs() < 1.0e-9);
    /// ```
    pub fn momentum(&self) -> [f64; 3] {
        let mut total = [0.0; 3];
        for particle in &self.particles {
            total[0] += particle.mass * particle.velocity[0];
            total[1] += particle.mass * particle.velocity[1];
            total[2] += particle.mass * particle.velocity[2];
        }
        [
            self.area * total[0],
            self.area * total[1],
            self.area * total[2],
        ]
    }

    /// Returns the center-probe density and pressure.
    pub fn center_probe(&self) -> (f64, f64) {
        let particle = &self.particles[self.center];
        (particle.density, particle.pressure)
    }

    /// Evaluates the deck equation of state with the `Pmin` floor.
    fn clipped_pressure(&self, density: f64) -> f64 {
        self.eos.pressure(density).max(WAVES_PMIN)
    }

    /// Advances one CFL-limited leapfrog step.
    ///
    /// Returns the step taken. Densities refresh from summation every
    /// evaluation, so no continuity drift accumulates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::CompressionRun;
    ///
    /// let mut run = CompressionRun::waves_compression().unwrap();
    /// let dt = run.step().unwrap();
    /// assert!(dt > 0.0 && run.time() == dt);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SphError`] when the signal speed collapses (all sound
    /// speeds plus bulk motion reach zero) or a kernel contract fails.
    pub fn step(&mut self) -> Result<f64, SphError> {
        let first = self.evaluate()?;
        let sound = first.sounds.iter().cloned().fold(0.0, f64::max);
        let signal = self
            .particles
            .iter()
            .map(|p| {
                (p.velocity[0] * p.velocity[0]
                    + p.velocity[1] * p.velocity[1]
                    + p.velocity[2] * p.velocity[2])
                    .sqrt()
            })
            .fold(0.0, f64::max);
        let dt = stable_time_step(self.cfl, self.h, sound, signal)?;
        self.advance(dt, &first)?;
        Ok(dt)
    }

    /// Runs to `end_time`, stamping `stamps` rows into a series.
    ///
    /// Stamps land on an approximately uniform grid (steps may
    /// overshoot a target; the final stamp sits at or just past
    /// `end_time`). Channels are `MASS`, `INTERNAL ENERGY`, `KINETIC
    /// ENERGY`, `TOTAL ENERGY`, `X-MOMENTUM`, `Y-MOMENTUM`,
    /// `Z-MOMENTUM`, `TIME STEP`, `CENTER DENSITY`, and `CENTER
    /// PRESSURE`; the first stamp carries a zero step (no step taken
    /// yet). `TOTAL ENERGY` maps to the vendor `NO NAME` channel and
    /// the `CENTER` probes are native analogs of the deck
    /// `/TH/SPHCEL` column, not vendor samples.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::sph::CompressionRun;
    ///
    /// let mut run = CompressionRun::waves_compression().unwrap();
    /// let series = run.run(0.5, 3).unwrap();
    /// assert_eq!(series.len(), 3);
    /// assert_eq!(series.times[0], 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SphError::BadState`] when `end_time` does not exceed
    /// the current time or fewer than two stamps are requested, plus
    /// any [`CompressionRun::step`] failure.
    pub fn run(&mut self, end_time: f64, stamps: usize) -> Result<Series, SphError> {
        if !end_time.is_finite() || end_time <= self.time {
            return Err(SphError::BadState("end time must exceed current time"));
        }
        if stamps < 2 {
            return Err(SphError::BadState("series needs at least two stamps"));
        }
        let mut series = Series::with_channels(&[
            "MASS",
            "INTERNAL ENERGY",
            "KINETIC ENERGY",
            "TOTAL ENERGY",
            "X-MOMENTUM",
            "Y-MOMENTUM",
            "Z-MOMENTUM",
            "TIME STEP",
            "CENTER DENSITY",
            "CENTER PRESSURE",
        ]);
        let start = self.time;
        series.push(start, &self.stamp(0.0));
        let mut next = 1;
        while self.time < end_time {
            let dt = self.step()?;
            while next < stamps
                && self.time >= start + (end_time - start) * next as f64 / (stamps - 1) as f64
            {
                series.push(self.time, &self.stamp(dt));
                next += 1;
            }
        }
        Ok(series)
    }

    /// Samples the current probe row with the given last step.
    fn stamp(&self, dt: f64) -> [f64; 10] {
        let internal = self.internal_energy();
        let kinetic = self.kinetic_energy();
        let momentum = self.momentum();
        let (density, pressure) = self.center_probe();
        [
            self.total_mass(),
            internal,
            kinetic,
            internal + kinetic,
            momentum[0],
            momentum[1],
            momentum[2],
            dt,
            density,
            pressure,
        ]
    }

    /// Applies one kick-drift-kick advance of `dt` from `first`.
    fn advance(&mut self, dt: f64, first: &Evaluation) -> Result<(), SphError> {
        for (particle, accel) in self.particles.iter_mut().zip(first.accelerations.iter()) {
            for (component, pull) in particle.velocity.iter_mut().zip(accel.iter()) {
                *component += 0.5 * dt * pull;
            }
        }
        for particle in self.particles.iter_mut() {
            for (coordinate, speed) in particle.position.iter_mut().zip(particle.velocity.iter()) {
                *coordinate += dt * speed;
            }
        }
        for particle in self.particles.iter_mut() {
            for (plane, condition) in &self.planes {
                // Axis-aligned planes: sequential reflection lands
                // corner crossers inside every plane at once.
                if plane.signed_distance(particle.position) < 0.0 {
                    particle.position = plane.reflect_position(particle.position);
                    particle.velocity = plane.ghost_velocity(particle.velocity, *condition);
                }
            }
        }
        // Fixed smoothing length (no per-step adaptation): snapping
        // h to the live density feeds back through the kernel self
        // term, and the impact compression is only a few percent, so
        // the support needs no tracking. See
        // updated_smoothing_length for the available rule.
        let second = self.evaluate()?;
        for (index, particle) in self.particles.iter_mut().enumerate() {
            for (component, pull) in particle
                .velocity
                .iter_mut()
                .zip(second.accelerations[index].iter())
            {
                *component += 0.5 * dt * pull;
            }
            particle.energy += 0.5 * dt * (first.rates[index] + second.rates[index]);
        }
        let pressures: Vec<f64> = second
            .densities
            .iter()
            .map(|density| self.clipped_pressure(*density))
            .collect();
        for (index, particle) in self.particles.iter_mut().enumerate() {
            particle.density = second.densities[index];
            particle.pressure = pressures[index];
            particle.sound_speed = self.eos.sound_speed(second.densities[index]);
        }
        self.time += dt;
        Ok(())
    }

    /// Refreshes stored thermodynamic state from one evaluation.
    fn refresh(&mut self) -> Result<(), SphError> {
        let evaluation = self.evaluate()?;
        let pressures: Vec<f64> = evaluation
            .densities
            .iter()
            .map(|density| self.clipped_pressure(*density))
            .collect();
        for (index, particle) in self.particles.iter_mut().enumerate() {
            particle.density = evaluation.densities[index];
            particle.pressure = pressures[index];
            particle.sound_speed = self.eos.sound_speed(evaluation.densities[index]);
        }
        Ok(())
    }

    /// Evaluates accelerations and energy rates with ghost support.
    fn evaluate(&self) -> Result<Evaluation, SphError> {
        let support = SUPPORT_RADIUS_FACTOR * self.h;
        let mut system = self.extended_system(support);
        let positions: Vec<[f64; 3]> = system.iter().map(|p| p.position).collect();
        let search = NeighborSearch::build(&positions, support)?;
        let count = self.particles.len();
        let mut densities = vec![0.0; count];
        for (index, slot) in densities.iter_mut().enumerate().take(count) {
            let neighbors = search.neighbors(index);
            *slot = summation_density(index, &system, &neighbors, self.h, 1)?;
        }
        let mut sounds = vec![0.0; count];
        for index in 0..count {
            system[index].density = densities[index];
            system[index].pressure = self.clipped_pressure(densities[index]);
            system[index].sound_speed = self.eos.sound_speed(densities[index]);
            sounds[index] = system[index].sound_speed;
        }
        let mut accelerations = vec![[0.0; 3]; count];
        let mut rates = vec![0.0; count];
        for index in 0..count {
            let neighbors = search.neighbors(index);
            accelerations[index] =
                momentum_acceleration(index, &system, &neighbors, self.h, 1, self.qa, self.qb)?;
            rates[index] = energy_rate(index, &system, &neighbors, self.h, 1, self.qa, self.qb)?;
        }
        Ok(Evaluation {
            accelerations,
            rates,
            densities,
            sounds,
        })
    }

    /// Mirrors live particles across symmetry planes within support.
    ///
    /// Ghosts copy their source mass and stored thermodynamic state;
    /// mirrors exactly coincident with their source (particles sitting
    /// on a plane) are skipped: they carry no gradient and would
    /// double-count the self density.
    fn extended_system(&self, support: f64) -> Vec<Particle> {
        let mut system = self.particles.clone();
        for (plane, condition) in &self.planes {
            for particle in &self.particles {
                let mirrored = plane.reflect_position(particle.position);
                let gap = (mirrored[0] - particle.position[0])
                    .hypot(mirrored[1] - particle.position[1])
                    .hypot(mirrored[2] - particle.position[2]);
                // Exact coincidence only: near-plane mirrors are real
                // symmetric counterparts and must stay in the sum.
                // Mirrors at or beyond the kernel support carry zero
                // weight, so the filter is exact, not an approximation.
                if gap == 0.0 || gap >= support {
                    continue;
                }
                let mut ghost = particle.clone();
                ghost.position = mirrored;
                ghost.velocity = plane.ghost_velocity(particle.velocity, *condition);
                system.push(ghost);
            }
        }
        system
    }
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

    fn uniform_lattice(n: usize, spacing: f64) -> Vec<Particle> {
        let mut particles = Vec::with_capacity(n * n * n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    particles.push(
                        Particle::new(
                            [i as f64 * spacing, j as f64 * spacing, k as f64 * spacing],
                            [0.0; 3],
                            spacing.powi(3),
                        )
                        .unwrap(),
                    );
                }
            }
        }
        particles
    }

    #[test]
    fn brookshaw_laplacian_recovers_quadratic_field() {
        // 5x5x5 lattice, unit volumes; quadratic bowl has Laplacian 6.
        let particles = uniform_lattice(5, 1.0);
        let field: Vec<f64> = particles
            .iter()
            .map(|p| {
                p.position[0] * p.position[0]
                    + p.position[1] * p.position[1]
                    + p.position[2] * p.position[2]
            })
            .collect();
        let positions: Vec<[f64; 3]> = particles.iter().map(|p| p.position).collect();
        let center = 2 * 25 + 2 * 5 + 2;
        let lists = brute_force_neighbors(&positions, 2.0 * 1.2);
        let lap = brookshaw_laplacian(center, &particles, &field, &lists[center], 1.2, 3).unwrap();
        assert!((lap - 6.0).abs() / 6.0 < 0.05, "laplacian = {lap}");
        // A linear field has zero Laplacian by antisymmetry.
        let linear: Vec<f64> = particles.iter().map(|p| p.position[0]).collect();
        let flat =
            brookshaw_laplacian(center, &particles, &linear, &lists[center], 1.2, 3).unwrap();
        assert!(flat.abs() < 1.0e-9, "linear laplacian = {flat}");
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

    #[test]
    fn column_matches_deck_footprint() {
        let run = CompressionRun::waves_compression().unwrap();
        assert_eq!(run.particles().len(), WAVES_COLUMN_CELLS);
        let height = WAVES_BOX_MAX[2] - WAVES_BOX_MIN[2];
        let spacing = height / WAVES_COLUMN_CELLS as f64;
        for (cell, particle) in run.particles().iter().enumerate() {
            assert_eq!(particle.position[0], 0.0);
            assert_eq!(particle.position[1], 0.0);
            let expected = WAVES_BOX_MIN[2] + (cell as f64 + 0.5) * spacing;
            assert_eq!(particle.position[2], expected);
            assert_eq!(particle.velocity, [0.0, 0.0, WAVES_VZ]);
            assert_eq!(particle.mass, WAVES_RHO0 * spacing);
        }
        // Extensive total is exactly reference density times box volume.
        let volume = height * run.cross_section();
        let expected = WAVES_RHO0 * volume;
        assert!(
            (run.total_mass() - expected).abs() < 1.0e-9 * expected,
            "mass {} vs {expected}",
            run.total_mass()
        );
        let center = &run.particles()[run.center_index()];
        assert!((center.position[2] - 0.5 * height).abs() < spacing);
    }

    #[test]
    fn initial_state_matches_deck_drop() {
        let run = CompressionRun::waves_compression().unwrap();
        let momentum = run.momentum();
        assert_eq!(momentum[0], 0.0);
        assert_eq!(momentum[1], 0.0);
        assert!((momentum[2] - run.total_mass() * WAVES_VZ).abs() < 1.0e-9);
        assert_eq!(run.internal_energy(), 0.0);
        let expected = 0.5 * run.total_mass() * WAVES_VZ * WAVES_VZ;
        assert!((run.kinetic_energy() - expected).abs() < 1.0e-9 * expected);
    }

    #[test]
    fn initial_densities_recover_reference() {
        let run = CompressionRun::waves_compression().unwrap();
        // Bulk and floor-adjacent particles sit within 2% (the floor
        // ghosts complete the support); only the free top face dips,
        // clipped to zero pressure by WAVES_PMIN.
        for (index, particle) in run.particles().iter().enumerate() {
            let last = index + 1 == run.particles().len();
            let deviation = ((particle.density - WAVES_RHO0) / WAVES_RHO0).abs();
            if last {
                assert!(deviation < 0.30, "top face dip {deviation}");
            } else {
                assert!(deviation < 0.02, "particle {index}: {deviation}");
            }
            assert!(particle.pressure >= 0.0, "Pmin floor holds");
        }
        let (density, _) = run.center_probe();
        assert!(((density - WAVES_RHO0) / WAVES_RHO0).abs() < 0.02);
    }

    #[test]
    fn floor_ghosts_complete_near_support() {
        let run = CompressionRun::waves_compression().unwrap();
        let support = SUPPORT_RADIUS_FACTOR * run.smoothing_length();
        let system = run.extended_system(support);
        let ghosts = &system[run.particles().len()..];
        assert!(!ghosts.is_empty(), "floor particles need mirrors");
        // Every ghost sits below the floor within kernel support of
        // its source, and cell-centered sources never coincide.
        for ghost in ghosts {
            assert!(ghost.position[2] < 0.0);
            // Floor mirror of a source at -z: gap is twice the depth.
            let gap = ghost.position[2].abs() * 2.0;
            assert!(gap > 0.0 && gap < support, "ghost gap {gap}");
        }
        let near = run
            .particles()
            .iter()
            .filter(|p| 2.0 * p.position[2] < support)
            .count();
        assert_eq!(ghosts.len(), near, "one mirror per near-floor particle");
    }

    #[test]
    fn run_rejects_bad_inputs() {
        let mut run = CompressionRun::waves_compression().unwrap();
        assert!(run.run(0.0, 5).is_err());
        assert!(run.run(-1.0, 5).is_err());
        assert!(run.run(1.0, 1).is_err());
        // A step advances time with a positive CFL step.
        let dt = run.step().unwrap();
        assert!(dt > 0.0 && run.time() == dt);
        assert!(run.run(run.time(), 5).is_err());
    }

    #[test]
    fn short_run_conserves_mass_and_stays_finite() {
        let mut run = CompressionRun::waves_compression().unwrap();
        let series = run.run(2.0, 5).unwrap();
        assert_eq!(series.len(), 5);
        assert_eq!(series.times[0], 0.0);
        assert!(series.times.windows(2).all(|w| w[1] > w[0]));
        // No mass update exists: bit-identical across stamps.
        let mass = series.channel("MASS").unwrap();
        assert!(mass.iter().all(|m| *m == mass[0]));
        for channel in series.channels.values() {
            assert_eq!(channel.len(), 5);
            assert!(channel.iter().all(|v| v.is_finite()));
        }
        // Transverse momenta stay bit-exact zero (no lateral forces).
        for name in ["X-MOMENTUM", "Y-MOMENTUM"] {
            assert!(
                series.channel(name).unwrap().iter().all(|v| *v == 0.0),
                "{name} must stay zero"
            );
        }
        // Energy drifts only at integration accuracy over 2 ms.
        let total = series.channel("TOTAL ENERGY").unwrap();
        assert!(((total[4] - total[0]) / total[0]).abs() < 0.02);
    }

    #[test]
    fn full_run_rebounds_and_converts_energy() {
        let mut run = CompressionRun::waves_compression().unwrap();
        let h0 = run.smoothing_length();
        let series = run.run(WAVES_END_TIME, 45).unwrap();
        assert_eq!(series.len(), 45);
        assert_eq!(run.smoothing_length(), h0, "fixed support");
        let internal = series.channel("INTERNAL ENERGY").unwrap();
        let kinetic = series.channel("KINETIC ENERGY").unwrap();
        let zmomentum = series.channel("Z-MOMENTUM").unwrap();
        // Impact signature: downward launch, upward rebound, and
        // kinetic energy converted into internal energy.
        assert!(zmomentum[0] < 0.0);
        assert!(
            *zmomentum.last().unwrap() > 0.0,
            "rebound, got {}",
            zmomentum.last().unwrap()
        );
        assert!(
            internal.last().unwrap() > kinetic.last().unwrap(),
            "conversion, ie={} ke={}",
            internal.last().unwrap(),
            kinetic.last().unwrap()
        );
        // Total energy (wall work plus clipping plus leapfrog error)
        // stays within its documented qualitative band.
        let total = series.channel("TOTAL ENERGY").unwrap();
        let drift = total
            .iter()
            .map(|e| ((e - total[0]) / total[0]).abs())
            .fold(0.0f64, f64::max);
        assert!(drift < 0.10, "drift {drift}");
        // Floor containment held every stamp of the probe column.
        assert!(run.particles().iter().all(|p| p.position[2] >= 0.0));
    }

    #[test]
    fn series_csv_renders_th_shape() {
        let mut series = Series::with_channels(&["MASS", "Z-MOMENTUM"]);
        series.push(0.0, &[584.5, -22.3]);
        series.push(1.5, &[584.5, 5.1]);
        let text = series.to_csv();
        let mut lines = text.lines();
        assert_eq!(lines.next().unwrap(), "Time,MASS,Z-MOMENTUM");
        assert_eq!(lines.next().unwrap(), "0,584.5,-22.3");
        assert_eq!(lines.next().unwrap(), "1.5,584.5,5.1");
        assert_eq!(lines.next(), None);
    }

    #[test]
    #[should_panic(expected = "distinct")]
    fn series_rejects_duplicate_channels() {
        let _ = Series::with_channels(&["MASS", "MASS"]);
    }

    #[test]
    #[should_panic(expected = "one value per channel")]
    fn series_rejects_short_stamp() {
        let mut series = Series::with_channels(&["MASS", "Z-MOMENTUM"]);
        series.push(0.0, &[1.0]);
    }
}
