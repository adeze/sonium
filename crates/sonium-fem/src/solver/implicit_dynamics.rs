//! Newmark-beta and HHT-alpha implicit structural dynamics.
//!
//! Integrates linear structural transients `M u'' + C u' + K u = F(t)`
//! with the unconditionally stable Newmark family. The default
//! trapezoidal rule (beta 1/4, gamma 1/2) preserves energy; the
//! Hilber-Hughes-Taylor variant (alpha in [-1/3, 0]) adds controllable
//! high-frequency damping while keeping second-order accuracy.
//!
//! Mass and stiffness matrices come from the existing structural
//! assembly (beam, shell, or continuum forms with consistent or lumped
//! mass); damping is an optional explicit matrix plus Rayleigh terms.
//! The effective matrix is factored once per step size with the sparse
//! direct solver, so each step costs one backsolve plus matvecs.
//!
//! Nonlinear (C1-material) dynamics reuses these coefficients with a
//! caller-supplied tangent and residual; this module covers the linear
//! mass-plus-stiffness-driven path the Phase-B QA validates.

// Rust guideline compliant 2026-02-21

use sonium_solvers::LinearOperator;
use sonium_solvers::TripletMatrix;
use sonium_solvers::direct::SparseLuFactorization;
use sonium_solvers::{CsrMatrix, SolverError};
use thiserror::Error;

/// Configuration of the Newmark/HHT integrator.
#[derive(Debug, Clone, Copy)]
pub struct ImplicitDynamicsConfig {
    /// Newmark beta (1/4 = trapezoidal average acceleration).
    pub beta: f64,
    /// Newmark gamma (1/2 = no algorithmic damping).
    pub gamma: f64,
    /// HHT alpha in [-1/3, 0] (0 = pure Newmark).
    pub alpha: f64,
    /// Mass-proportional Rayleigh damping coefficient.
    pub rayleigh_mass: f64,
    /// Stiffness-proportional Rayleigh damping coefficient.
    pub stiffness_damping: f64,
}

impl Default for ImplicitDynamicsConfig {
    /// Trapezoidal rule without damping.
    fn default() -> Self {
        Self {
            beta: 0.25,
            gamma: 0.5,
            alpha: 0.0,
            rayleigh_mass: 0.0,
            stiffness_damping: 0.0,
        }
    }
}

impl ImplicitDynamicsConfig {
    /// Newmark scheme with explicit beta and gamma.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::solver::implicit_dynamics::ImplicitDynamicsConfig;
    ///
    /// let config = ImplicitDynamicsConfig::newmark(0.25, 0.5);
    /// config.validate().unwrap();
    /// ```
    pub fn newmark(beta: f64, gamma: f64) -> Self {
        Self {
            beta,
            gamma,
            alpha: 0.0,
            rayleigh_mass: 0.0,
            stiffness_damping: 0.0,
        }
    }

    /// HHT-alpha scheme with second-order parameters.
    ///
    /// Sets beta to `(1 - alpha)^2 / 4` and gamma to `1/2 - alpha`,
    /// which keeps second-order accuracy with algorithmic damping
    /// growing as alpha goes negative (Hilber, Hughes, and Taylor).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::solver::implicit_dynamics::ImplicitDynamicsConfig;
    ///
    /// let config = ImplicitDynamicsConfig::hht(-0.1);
    /// assert!((config.gamma - 0.6).abs() < 1e-15);
    /// config.validate().unwrap();
    /// ```
    pub fn hht(alpha: f64) -> Self {
        Self {
            beta: (1.0 - alpha).powi(2) / 4.0,
            gamma: 0.5 - alpha,
            alpha,
            rayleigh_mass: 0.0,
            stiffness_damping: 0.0,
        }
    }

    /// Check parameter ranges and stability relations.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError::InvalidConfig`] for non-finite
    /// values, non-positive beta, gamma below 1/2, HHT alpha outside
    /// [-1/3, 0], negative damping, or a conditionally stable pair.
    pub fn validate(&self) -> Result<(), ImplicitDynamicsError> {
        for (name, value) in [
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("alpha", self.alpha),
            ("rayleigh_mass", self.rayleigh_mass),
            ("stiffness_damping", self.stiffness_damping),
        ] {
            if !value.is_finite() {
                return Err(ImplicitDynamicsError::InvalidConfig(format!(
                    "{name} must be finite"
                )));
            }
        }
        if self.beta <= 0.0 {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "beta must be positive".to_string(),
            ));
        }
        if self.gamma < 0.5 {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "gamma below 1/2 loses unconditional stability".to_string(),
            ));
        }
        if !(-1.0 / 3.0..=0.0).contains(&self.alpha) {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "HHT alpha must lie in [-1/3, 0]".to_string(),
            ));
        }
        if self.rayleigh_mass < 0.0 || self.stiffness_damping < 0.0 {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "Rayleigh coefficients must be non-negative".to_string(),
            ));
        }
        // Newmark unconditional-stability boundary 2 beta >= gamma.
        if 2.0 * self.beta < self.gamma {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "2 beta >= gamma is required for stability".to_string(),
            ));
        }
        Ok(())
    }
}

/// Mass, stiffness, and damping of a linear structural system.
#[derive(Debug, Clone)]
pub struct ImplicitDynamicsSystem {
    /// Consistent or lumped mass matrix (symmetric positive definite).
    pub mass: CsrMatrix<f64>,
    /// Linear stiffness matrix (symmetric positive semi-definite).
    pub stiffness: CsrMatrix<f64>,
    /// Explicit viscous damping matrix (added to Rayleigh terms).
    pub damping: Option<CsrMatrix<f64>>,
}

impl ImplicitDynamicsSystem {
    /// Count system degrees of freedom.
    pub fn num_dofs(&self) -> usize {
        self.mass.num_rows
    }

    /// Check squareness and matching dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError::DimensionMismatch`] for
    /// non-square or inconsistently sized matrices.
    pub fn validate(&self) -> Result<(), ImplicitDynamicsError> {
        let n = self.mass.num_rows;
        if n == 0 || self.mass.num_cols != n {
            return Err(ImplicitDynamicsError::DimensionMismatch {
                expected: n,
                actual: self.mass.num_cols,
            });
        }
        if self.stiffness.num_rows != n || self.stiffness.num_cols != n {
            return Err(ImplicitDynamicsError::DimensionMismatch {
                expected: n,
                actual: self.stiffness.num_rows.max(self.stiffness.num_cols),
            });
        }
        if let Some(damping) = &self.damping
            && (damping.num_rows != n || damping.num_cols != n)
        {
            return Err(ImplicitDynamicsError::DimensionMismatch {
                expected: n,
                actual: damping.num_rows.max(damping.num_cols),
            });
        }
        Ok(())
    }
}

/// Displacement, velocity, and acceleration at one time station.
#[derive(Debug, Clone)]
pub struct ImplicitDynamicsState {
    /// Nodal displacements.
    pub displacement: Vec<f64>,
    /// Nodal velocities.
    pub velocity: Vec<f64>,
    /// Nodal accelerations.
    pub acceleration: Vec<f64>,
    /// Simulated time in seconds.
    pub time: f64,
}

/// Failure of an implicit dynamics operation.
#[derive(Debug, Error)]
pub enum ImplicitDynamicsError {
    /// Matrix or vector dimensions disagree.
    #[error("dynamics dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected length or order.
        expected: usize,
        /// Observed length or order.
        actual: usize,
    },
    /// A state, force, or matrix entry is NaN or infinite.
    #[error("dynamics input {name} must be finite")]
    NonFinite {
        /// Offending input name.
        name: String,
    },
    /// Integrator parameters are out of range.
    #[error("invalid dynamics configuration: {0}")]
    InvalidConfig(String),
    /// Effective-matrix factorization or backsolve failed.
    #[error("dynamics linear solve failed: {0}")]
    LinearSolve(String),
    /// A step ran before initial conditions were set.
    #[error("dynamics integrator needs initialize() before stepping")]
    NotInitialized,
}

impl From<SolverError> for ImplicitDynamicsError {
    fn from(error: SolverError) -> Self {
        Self::LinearSolve(error.to_string())
    }
}

/// Newmark effective-load coefficients `[a0..a5]`.
///
/// With step `dt`: `a0 = 1/(beta dt^2)`, `a1 = gamma/(beta dt)`,
/// `a2 = 1/(beta dt)`, `a3 = 1/(2 beta) - 1`, `a4 = gamma/beta - 1`,
/// and `a5 = dt (gamma/(2 beta) - 1)` (Bathe textbook notation).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::implicit_dynamics::newmark_coefficients;
///
/// let a = newmark_coefficients(0.25, 0.5, 0.1);
/// assert!((a[0] - 400.0).abs() < 1e-9);
/// assert!((a[1] - 20.0).abs() < 1e-9);
/// ```
pub fn newmark_coefficients(beta: f64, gamma: f64, dt: f64) -> [f64; 6] {
    [
        1.0 / (beta * dt * dt),
        gamma / (beta * dt),
        1.0 / (beta * dt),
        1.0 / (2.0 * beta) - 1.0,
        gamma / beta - 1.0,
        dt * (gamma / (2.0 * beta) - 1.0),
    ]
}

/// Amplification matrix of a damped SDOF step.
///
/// Returns the 3x3 matrix mapping `[u, v, a]` at step `n` to step
/// `n + 1` for mass `m`, damping `c`, stiffness `k`, and the given
/// Newmark/HHT parameters (force-free part of the update).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::implicit_dynamics::amplification_matrix_sdof;
///
/// let a = amplification_matrix_sdof(1.0, 0.0, 100.0, 0.01, 0.25, 0.5, 0.0);
/// assert!(a[0][0] < 1.0 && a[0][0] > 0.9);
/// ```
pub fn amplification_matrix_sdof(
    m: f64,
    c: f64,
    k: f64,
    dt: f64,
    beta: f64,
    gamma: f64,
    alpha: f64,
) -> [[f64; 3]; 3] {
    let a = newmark_coefficients(beta, gamma, dt);
    let h = 1.0 + alpha;
    let khat = a[0] * m + a[1] * h * c + h * k;
    // Displacement row from the HHT equilibrium substitution.
    let row0 = [
        (m * a[0] + c * h * a[1] + k * alpha) / khat,
        (m * a[2] + c * (h * a[4] + alpha)) / khat,
        (m * a[3] + c * h * a[5]) / khat,
    ];
    // Velocity and acceleration rows from the Newmark updates.
    let updates = [[a[1], a[4], a[5]], [a[0], a[2], a[3]]];
    let mut out = [row0, [0.0; 3], [0.0; 3]];
    for (row, weights) in out.iter_mut().skip(1).zip(updates.iter()) {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = weights[0] * row0[j] - weights[j];
        }
    }
    out
}

/// Spectral radius (largest eigenvalue modulus) of a 3x3 matrix.
///
/// Solves the characteristic cubic in closed form (Cardano with the
/// trigonometric three-real-root branch), so conjugate pairs report
/// their shared modulus. Used to audit integrator damping.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::implicit_dynamics::spectral_radius_3x3;
///
/// let rho = spectral_radius_3x3([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -0.8]]);
/// assert!((rho - 1.0).abs() < 1e-15);
/// ```
pub fn spectral_radius_3x3(matrix: [[f64; 3]; 3]) -> f64 {
    let trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    // Sum of principal 2x2 minors.
    let minor = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        + matrix[0][0] * matrix[2][2]
        - matrix[0][2] * matrix[2][0]
        + matrix[1][1] * matrix[2][2]
        - matrix[1][2] * matrix[2][1];
    let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    // Depressed cubic t^3 + p t + q with lambda = t + trace/3.
    let shift = trace / 3.0;
    let p = minor - trace * trace / 3.0;
    let q = -2.0 * trace.powi(3) / 27.0 + minor * trace / 3.0 - det;
    if p.abs() < 1e-300 && q.abs() < 1e-300 {
        return shift.abs();
    }
    let discriminant = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    if discriminant > 0.0 {
        let root = discriminant.sqrt();
        let u = (-q / 2.0 + root).cbrt();
        let v = (-q / 2.0 - root).cbrt();
        let real = (u + v + shift).abs();
        // Conjugate pair shares one modulus.
        let pair_re = -(u + v) / 2.0 + shift;
        let pair_im = (u - v) * 0.5 * 3.0_f64.sqrt();
        real.max((pair_re * pair_re + pair_im * pair_im).sqrt())
    } else {
        // Three real roots via the trigonometric branch.
        let radius = 2.0 * (-p / 3.0).sqrt();
        let cos_arg = ((-q / 2.0) / (-p / 3.0).powf(1.5)).clamp(-1.0, 1.0);
        let angle = cos_arg.acos() / 3.0;
        let mut max = 0.0_f64;
        for k in 0..3 {
            let root =
                radius * (angle - 2.0 * std::f64::consts::PI * f64::from(k) / 3.0).cos() + shift;
            max = max.max(root.abs());
        }
        max
    }
}

fn check_finite(values: &[f64], name: &str) -> Result<(), ImplicitDynamicsError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ImplicitDynamicsError::NonFinite {
            name: name.to_string(),
        })
    }
}

/// Newmark/HHT integrator with a factored effective matrix.
///
/// The effective operator `Khat = a0 M + a1 (1+alpha) C + (1+alpha) K`
/// is assembled once per step size; [`step`](Self::step) then applies
/// one backsolve plus sparse matvecs per station.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::implicit_dynamics::{
///     ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
/// };
/// use sonium_solvers::{CsrMatrix, TripletMatrix};
///
/// let mut mass = TripletMatrix::new(1, 1);
/// mass.push(0, 0, 1.0);
/// let mut stiff = TripletMatrix::new(1, 1);
/// stiff.push(0, 0, 4.0);
/// let system = ImplicitDynamicsSystem {
///     mass: mass.to_csr(),
///     stiffness: stiff.to_csr(),
///     damping: None,
/// };
/// let mut integrator = ImplicitDynamicsIntegrator::new(
///     system,
///     ImplicitDynamicsConfig::default(),
///     0.01,
/// )
/// .unwrap();
/// integrator.initialize(&[1.0], &[0.0], &[0.0]).unwrap();
/// for _ in 0..10 {
///     integrator.step(&[0.0]).unwrap();
/// }
/// assert!(integrator.state().displacement[0].abs() <= 1.0 + 1e-9);
/// ```
#[derive(Debug)]
pub struct ImplicitDynamicsIntegrator {
    config: ImplicitDynamicsConfig,
    dt: f64,
    num_dofs: usize,
    mass: CsrMatrix<f64>,
    stiffness: CsrMatrix<f64>,
    damping: Option<CsrMatrix<f64>>,
    coeffs: [f64; 6],
    effective: SparseLuFactorization<f64>,
    mass_solver: SparseLuFactorization<f64>,
    state: ImplicitDynamicsState,
    prev_force: Vec<f64>,
    initialized: bool,
}

impl ImplicitDynamicsIntegrator {
    /// Build an integrator and factor the effective matrix.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError`] for invalid parameters,
    /// inconsistent dimensions, a non-positive step, or a factorization
    /// failure.
    pub fn new(
        system: ImplicitDynamicsSystem,
        config: ImplicitDynamicsConfig,
        dt: f64,
    ) -> Result<Self, ImplicitDynamicsError> {
        config.validate()?;
        system.validate()?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "time step must be positive and finite".to_string(),
            ));
        }
        let num_dofs = system.num_dofs();
        let coeffs = newmark_coefficients(config.beta, config.gamma, dt);
        let effective = factor_effective(&system, &config, &coeffs)?;
        let mass_solver = SparseLuFactorization::factorize(&system.mass)
            .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string()))?;
        Ok(Self {
            config,
            dt,
            num_dofs,
            mass: system.mass,
            stiffness: system.stiffness,
            damping: system.damping,
            coeffs,
            effective,
            mass_solver,
            state: ImplicitDynamicsState {
                displacement: vec![0.0; num_dofs],
                velocity: vec![0.0; num_dofs],
                acceleration: vec![0.0; num_dofs],
                time: 0.0,
            },
            prev_force: vec![0.0; num_dofs],
            initialized: false,
        })
    }

    /// Change the step size, refactoring the effective matrix.
    ///
    /// State is preserved; the next step advances by the new `dt`.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError`] for a non-positive step or a
    /// refactorization failure.
    pub fn set_timestep(&mut self, dt: f64) -> Result<(), ImplicitDynamicsError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "time step must be positive and finite".to_string(),
            ));
        }
        let coeffs = newmark_coefficients(self.config.beta, self.config.gamma, dt);
        let system = ImplicitDynamicsSystem {
            mass: self.mass.clone(),
            stiffness: self.stiffness.clone(),
            damping: self.damping.clone(),
        };
        self.effective = factor_effective(&system, &self.config, &coeffs)?;
        self.dt = dt;
        self.coeffs = coeffs;
        Ok(())
    }

    /// Set initial displacement, velocity, and load.
    ///
    /// Computes the consistent initial acceleration from
    /// `M a0 = F0 - C v0 - K u0`.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError`] for wrong lengths, non-finite
    /// inputs, or a mass-solve failure.
    pub fn initialize(
        &mut self,
        displacement: &[f64],
        velocity: &[f64],
        force: &[f64],
    ) -> Result<(), ImplicitDynamicsError> {
        for (name, values) in [
            ("displacement", displacement),
            ("velocity", velocity),
            ("force", force),
        ] {
            if values.len() != self.num_dofs {
                return Err(ImplicitDynamicsError::DimensionMismatch {
                    expected: self.num_dofs,
                    actual: values.len(),
                });
            }
            check_finite(values, name)?;
        }
        let damping_force = self.apply_damping(velocity);
        let elastic_force = self.stiffness.apply(displacement);
        let residual: Vec<f64> = force
            .iter()
            .zip(damping_force.iter())
            .zip(elastic_force.iter())
            .map(|((f, c), k)| f - c - k)
            .collect();
        let acceleration = self
            .mass_solver
            .solve(&residual)
            .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string()))?;
        check_finite(&acceleration, "initial acceleration")?;
        self.state = ImplicitDynamicsState {
            displacement: displacement.to_vec(),
            velocity: velocity.to_vec(),
            acceleration,
            time: 0.0,
        };
        self.prev_force = force.to_vec();
        self.initialized = true;
        Ok(())
    }

    /// Advance one step under the next-station load.
    ///
    /// For HHT the equilibrium load is the alpha-weighted blend
    /// `(1+alpha) F+ - alpha F` of the given and previous forces.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError`] before initialization, for a
    /// wrongly sized or non-finite load, or when the backsolve fails.
    pub fn step(
        &mut self,
        force_next: &[f64],
    ) -> Result<&ImplicitDynamicsState, ImplicitDynamicsError> {
        if !self.initialized {
            return Err(ImplicitDynamicsError::NotInitialized);
        }
        if force_next.len() != self.num_dofs {
            return Err(ImplicitDynamicsError::DimensionMismatch {
                expected: self.num_dofs,
                actual: force_next.len(),
            });
        }
        check_finite(force_next, "force")?;
        let a = self.coeffs;
        let h = 1.0 + self.config.alpha;
        let state = &self.state;
        // Mass-weighted history: a0 u + a2 v + a3 a.
        let history_m: Vec<f64> = state
            .displacement
            .iter()
            .zip(state.velocity.iter())
            .zip(state.acceleration.iter())
            .map(|((u, v), acc)| a[0] * u + a[2] * v + a[3] * acc)
            .collect();
        // Damping-weighted history plus the HHT velocity remnant.
        let history_c: Vec<f64> = state
            .displacement
            .iter()
            .zip(state.velocity.iter())
            .zip(state.acceleration.iter())
            .map(|((u, v), acc)| h * (a[1] * u + a[4] * v + a[5] * acc) + self.config.alpha * v)
            .collect();
        let mass_part = self.mass.apply(&history_m);
        let damp_part = self.apply_damping(&history_c);
        let stiff_history: Vec<f64> = state
            .displacement
            .iter()
            .map(|u| self.config.alpha * u)
            .collect();
        let stiff_part = self.stiffness.apply(&stiff_history);
        let rhs: Vec<f64> = force_next
            .iter()
            .zip(self.prev_force.iter())
            .zip(mass_part.iter())
            .zip(damp_part.iter())
            .zip(stiff_part.iter())
            .map(|((((next, prev), m), c), k)| h * next - self.config.alpha * prev + m + c + k)
            .collect();
        let next_u = self
            .effective
            .solve(&rhs)
            .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string()))?;
        check_finite(&next_u, "displacement")?;
        let next_a: Vec<f64> = next_u
            .iter()
            .zip(state.displacement.iter())
            .zip(state.velocity.iter())
            .zip(state.acceleration.iter())
            .map(|(((plus, u), v), acc)| a[0] * (plus - u) - a[2] * v - a[3] * acc)
            .collect();
        let gamma = self.config.gamma;
        let next_v: Vec<f64> = state
            .velocity
            .iter()
            .zip(state.acceleration.iter())
            .zip(next_a.iter())
            .map(|((v, acc), plus)| v + self.dt * ((1.0 - gamma) * acc + gamma * plus))
            .collect();
        self.state = ImplicitDynamicsState {
            displacement: next_u,
            velocity: next_v,
            acceleration: next_a,
            time: state.time + self.dt,
        };
        self.prev_force = force_next.to_vec();
        Ok(&self.state)
    }

    /// Borrow the current state.
    pub fn state(&self) -> &ImplicitDynamicsState {
        &self.state
    }

    /// Borrow the integrator configuration.
    pub fn config(&self) -> &ImplicitDynamicsConfig {
        &self.config
    }

    /// Current step size in seconds.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Count system degrees of freedom.
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }

    fn apply_damping(&self, values: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.num_dofs];
        if let Some(damping) = &self.damping {
            let part = damping.apply(values);
            for (slot, value) in out.iter_mut().zip(part.iter()) {
                *slot += value;
            }
        }
        if self.config.rayleigh_mass != 0.0 {
            let part = self.mass.apply(values);
            for (slot, value) in out.iter_mut().zip(part.iter()) {
                *slot += self.config.rayleigh_mass * value;
            }
        }
        if self.config.stiffness_damping != 0.0 {
            let part = self.stiffness.apply(values);
            for (slot, value) in out.iter_mut().zip(part.iter()) {
                *slot += self.config.stiffness_damping * value;
            }
        }
        out
    }
}

fn factor_effective(
    system: &ImplicitDynamicsSystem,
    config: &ImplicitDynamicsConfig,
    coeffs: &[f64; 6],
) -> Result<SparseLuFactorization<f64>, ImplicitDynamicsError> {
    let h = 1.0 + config.alpha;
    let scale_m = coeffs[0] + coeffs[1] * h * config.rayleigh_mass;
    let scale_k = h + coeffs[1] * h * config.stiffness_damping;
    let scale_c = coeffs[1] * h;
    let n = system.num_dofs();
    let mut triplet = TripletMatrix::new(n, n);
    for row in 0..n {
        for (col, value) in system.mass.row_entries(row) {
            if value != 0.0 {
                triplet.push(row, col, scale_m * value);
            }
        }
        for (col, value) in system.stiffness.row_entries(row) {
            if value != 0.0 {
                triplet.push(row, col, scale_k * value);
            }
        }
        if let Some(damping) = &system.damping {
            for (col, value) in damping.row_entries(row) {
                if value != 0.0 {
                    triplet.push(row, col, scale_c * value);
                }
            }
        }
    }
    // The triplet buffers are constructed valid; conversion sums the
    // overlapping mass, stiffness, and damping patterns.
    let matrix = triplet.to_csr();
    SparseLuFactorization::factorize(&matrix)
        .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sdof_system(m: f64, c: f64, k: f64) -> ImplicitDynamicsSystem {
        let mut mass = TripletMatrix::new(1, 1);
        mass.push(0, 0, m);
        let mut stiffness = TripletMatrix::new(1, 1);
        stiffness.push(0, 0, k);
        let damping = if c == 0.0 {
            None
        } else {
            let mut triplet = TripletMatrix::new(1, 1);
            triplet.push(0, 0, c);
            Some(triplet.to_csr())
        };
        ImplicitDynamicsSystem {
            mass: mass.to_csr(),
            stiffness: stiffness.to_csr(),
            damping,
        }
    }

    #[test]
    fn hht_parameters_follow_second_order_relations() {
        let config = ImplicitDynamicsConfig::hht(-0.1);
        assert!((config.beta - 0.3025).abs() < 1e-15);
        assert!((config.gamma - 0.6).abs() < 1e-15);
        config.validate().unwrap();
        let bad = ImplicitDynamicsConfig::hht(-0.5);
        assert!(bad.validate().is_err());
        let unstable = ImplicitDynamicsConfig::newmark(0.2, 0.5);
        assert!(unstable.validate().is_err());
    }

    #[test]
    fn trapezoidal_free_vibration_holds_amplitude() {
        let omega = 2.0 * std::f64::consts::PI;
        let system = sdof_system(1.0, 0.0, omega * omega);
        let mut integrator =
            ImplicitDynamicsIntegrator::new(system, ImplicitDynamicsConfig::default(), 0.01)
                .unwrap();
        integrator.initialize(&[1.0], &[0.0], &[0.0]).unwrap();
        for _ in 0..100 {
            integrator.step(&[0.0]).unwrap();
        }
        let state = integrator.state();
        assert!((state.time - 1.0).abs() < 1e-12);
        // Exact return to 1 after one period; trapezoidal phase lags slightly.
        assert!((state.displacement[0] - 1.0).abs() < 1e-2);
        assert!(state.velocity[0].abs() < 5e-2);
    }

    #[test]
    fn step_load_matches_closed_form() {
        let omega = 10.0;
        let system = sdof_system(1.0, 0.0, omega * omega);
        let period = 2.0 * std::f64::consts::PI / omega;
        let mut integrator = ImplicitDynamicsIntegrator::new(
            system,
            ImplicitDynamicsConfig::default(),
            period / 200.0,
        )
        .unwrap();
        integrator.initialize(&[0.0], &[0.0], &[1.0]).unwrap();
        for _ in 0..100 {
            integrator.step(&[1.0]).unwrap();
        }
        // Half-period peak of 1 - cos: twice the static deflection.
        let peak = integrator.state().displacement[0];
        assert!((peak - 0.02).abs() / 0.02 < 0.02, "peak is {peak}");
    }

    #[test]
    fn rayleigh_damping_matches_decay_envelope() {
        let omega = 2.0 * std::f64::consts::PI;
        let zeta = 0.05;
        let mut config = ImplicitDynamicsConfig::default();
        config.rayleigh_mass = 2.0 * zeta * omega;
        let system = sdof_system(1.0, 0.0, omega * omega);
        let mut integrator = ImplicitDynamicsIntegrator::new(system, config, 0.005).unwrap();
        integrator.initialize(&[1.0], &[0.0], &[0.0]).unwrap();
        for _ in 0..200 {
            integrator.step(&[0.0]).unwrap();
        }
        let expected = (-2.0 * std::f64::consts::PI * zeta).exp();
        let amplitude = integrator.state().displacement[0];
        assert!(
            (amplitude - expected).abs() / expected < 0.02,
            "amplitude is {amplitude}"
        );
    }

    #[test]
    fn hht_damps_above_trapezoidal() {
        // Large step: trapezoidal holds unit radius, HHT contracts.
        let trap = amplification_matrix_sdof(1.0, 0.0, 1.0, 1.0, 0.25, 0.5, 0.0);
        let hht = amplification_matrix_sdof(1.0, 0.0, 1.0, 1.0, 0.4225, 0.8, -0.3);
        assert!((spectral_radius_3x3(trap) - 1.0).abs() < 1e-12);
        assert!(spectral_radius_3x3(hht) < 1.0);
    }

    #[test]
    fn spectral_radius_covers_real_and_complex_spectra() {
        let diag = spectral_radius_3x3([[0.5, 0.0, 0.0], [0.0, -0.8, 0.0], [0.0, 0.0, 0.3]]);
        assert!((diag - 0.8).abs() < 1e-15);
        // Planar rotation through 90 degrees has unit modulus.
        let rotation = spectral_radius_3x3([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]]);
        assert!((rotation - 1.0).abs() < 1e-12);
        // Nilpotent Jordan block has zero radius.
        let nilpotent = spectral_radius_3x3([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]);
        assert!(nilpotent < 1e-12);
    }

    #[test]
    fn integrator_rejects_bad_inputs() {
        let system = sdof_system(1.0, 0.0, 4.0);
        let mut integrator =
            ImplicitDynamicsIntegrator::new(system, ImplicitDynamicsConfig::default(), 0.01)
                .unwrap();
        assert!(integrator.step(&[0.0]).is_err());
        assert!(integrator.initialize(&[1.0, 2.0], &[0.0], &[0.0]).is_err());
        integrator.initialize(&[1.0], &[0.0], &[0.0]).unwrap();
        assert!(integrator.step(&[0.0, 1.0]).is_err());
        assert!(integrator.set_timestep(0.0).is_err());
        integrator.set_timestep(0.02).unwrap();
        assert!((integrator.dt() - 0.02).abs() < 1e-15);
    }
}
