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
use sonium_solvers::direct::{SparseCholeskyFactorization, SparseLuFactorization};
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

#[derive(Debug)]
enum DynamicsFactor {
    Lu(Box<SparseLuFactorization<f64>>),
    Cholesky(Box<SparseCholeskyFactorization<f64>>),
}

impl DynamicsFactor {
    fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, ImplicitDynamicsError> {
        match self {
            Self::Lu(factor) => factor
                .solve(rhs)
                .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string())),
            Self::Cholesky(factor) => factor
                .solve(rhs)
                .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string())),
        }
    }
}

fn factor_matrix(
    matrix: &CsrMatrix<f64>,
    cholesky: bool,
) -> Result<DynamicsFactor, ImplicitDynamicsError> {
    if !cholesky {
        return SparseLuFactorization::factorize(matrix)
            .map(|factor| DynamicsFactor::Lu(Box::new(factor)))
            .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string()));
    }
    check_finite(&matrix.values, "Cholesky matrix")?;
    let scale = matrix
        .values
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    // Permit assembly roundoff relative to the operator scale, not a missing asymmetric block.
    let tolerance = 1e-12 * scale;
    for row in 0..matrix.num_rows {
        for (column, value) in matrix.row_entries(row) {
            if (value - matrix.get(column, row)).abs() > tolerance {
                return Err(ImplicitDynamicsError::InvalidConfig(
                    "Cholesky requires symmetric mass and effective matrices".into(),
                ));
            }
        }
    }
    SparseCholeskyFactorization::factorize_supernodal(matrix)
        .map(|factor| DynamicsFactor::Cholesky(Box::new(factor)))
        .map_err(|error| ImplicitDynamicsError::LinearSolve(error.to_string()))
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
    effective: DynamicsFactor,
    mass_solver: DynamicsFactor,
    cholesky: bool,
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
        Self::new_with_factorization(system, config, dt, false)
    }

    /// Build an integrator using Cholesky for positive-definite systems.
    ///
    /// Uses nested-dissection ordering and supernodal factorization for both
    /// mass and effective operators. Symmetry is checked to `1e-12` of each
    /// operator's largest entry; positive-definiteness is checked by factorization.
    /// The Newmark/HHT equations and checked-step residual contract are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError`] for invalid configuration or dimensions,
    /// non-finite or asymmetric matrices, or a non-positive Cholesky pivot.
    pub fn new_spd(
        system: ImplicitDynamicsSystem,
        config: ImplicitDynamicsConfig,
        dt: f64,
    ) -> Result<Self, ImplicitDynamicsError> {
        Self::new_with_factorization(system, config, dt, true)
    }

    fn new_with_factorization(
        system: ImplicitDynamicsSystem,
        config: ImplicitDynamicsConfig,
        dt: f64,
        cholesky: bool,
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
        let effective = factor_effective(&system, &config, &coeffs, cholesky)?;
        let mass_solver = factor_matrix(&system.mass, cholesky)?;
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
            cholesky,
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
        self.effective = factor_effective(&system, &self.config, &coeffs, self.cholesky)?;
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
        let acceleration = self.mass_solver.solve(&residual)?;
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
        self.step_impl(force_next, None, false)
    }

    /// Advance while verifying the original HHT or Newmark force balance.
    ///
    /// `absolute_residual_tolerance` bounds the Euclidean norm of the weighted
    /// equilibrium equation in force units. Up to three corrections reuse the
    /// effective factorization, updating displacement, velocity, and acceleration
    /// together through the Newmark relations. A failed check leaves the previous
    /// state and force history unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError`] for the ordinary step errors, a
    /// non-finite or negative tolerance, or a force-balance residual that cannot
    /// meet the tolerance within three refinement corrections.
    pub fn step_checked(
        &mut self,
        force_next: &[f64],
        absolute_residual_tolerance: f64,
    ) -> Result<&ImplicitDynamicsState, ImplicitDynamicsError> {
        self.checked_step_impl(force_next, absolute_residual_tolerance, false)
    }

    /// Advance with compensated matrix products and verified physical force balance.
    ///
    /// Uses error-free product residuals and compensated additions for mass,
    /// stiffness, and damping products in the history and equilibrium checks.
    /// This reduces cancellation error in mixed-unit thin structural systems.
    /// The factorization, Newmark/HHT equations, and absolute force-norm
    /// tolerance are the same as [`Self::step_checked`]. Up to eight corrections
    /// allow cancellation-sensitive systems to finish iterative refinement.
    /// Failure leaves the previous state and force history unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ImplicitDynamicsError`] for invalid loads or tolerance,
    /// factorization solve failures, or failure to meet the force-balance bound.
    pub fn step_checked_accurate(
        &mut self,
        force_next: &[f64],
        absolute_residual_tolerance: f64,
    ) -> Result<&ImplicitDynamicsState, ImplicitDynamicsError> {
        self.checked_step_impl(force_next, absolute_residual_tolerance, true)
    }

    fn checked_step_impl(
        &mut self,
        force_next: &[f64],
        absolute_residual_tolerance: f64,
        compensated: bool,
    ) -> Result<&ImplicitDynamicsState, ImplicitDynamicsError> {
        if !absolute_residual_tolerance.is_finite() || absolute_residual_tolerance < 0.0 {
            return Err(ImplicitDynamicsError::InvalidConfig(
                "equilibrium tolerance must be finite and non-negative".into(),
            ));
        }
        self.step_impl(force_next, Some(absolute_residual_tolerance), compensated)
    }

    fn step_impl(
        &mut self,
        force_next: &[f64],
        equilibrium_tolerance: Option<f64>,
        compensated: bool,
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
        let mass_part = force_matvec(&self.mass, &history_m, compensated);
        let damp_part = self.apply_damping_impl(&history_c, compensated);
        let stiff_history: Vec<f64> = state
            .displacement
            .iter()
            .map(|u| self.config.alpha * u)
            .collect();
        let stiff_part = force_matvec(&self.stiffness, &stiff_history, compensated);
        let rhs: Vec<f64> = force_next
            .iter()
            .zip(self.prev_force.iter())
            .zip(mass_part.iter())
            .zip(damp_part.iter())
            .zip(stiff_part.iter())
            .map(|((((next, prev), m), c), k)| h * next - self.config.alpha * prev + m + c + k)
            .collect();
        let mut next_u = self.effective.solve(&rhs)?;
        check_finite(&next_u, "displacement")?;
        let mut next_a: Vec<f64> = next_u
            .iter()
            .zip(state.displacement.iter())
            .zip(state.velocity.iter())
            .zip(state.acceleration.iter())
            .map(|(((plus, u), v), acc)| a[0] * (plus - u) - a[2] * v - a[3] * acc)
            .collect();
        let gamma = self.config.gamma;
        let mut next_v: Vec<f64> = state
            .velocity
            .iter()
            .zip(state.acceleration.iter())
            .zip(next_a.iter())
            .map(|((v, acc), plus)| v + self.dt * ((1.0 - gamma) * acc + gamma * plus))
            .collect();
        if let Some(tolerance) = equilibrium_tolerance {
            let previous_elastic = force_matvec(&self.stiffness, &state.displacement, compensated);
            let previous_damping = self.apply_damping_impl(&state.velocity, compensated);
            // Bound checked-step cost. Ordinary steps retain their existing path.
            let max_equilibrium_refinements = if compensated { 8 } else { 3 };
            for refinement in 0..=max_equilibrium_refinements {
                let inertia = force_matvec(&self.mass, &next_a, compensated);
                let elastic = force_matvec(&self.stiffness, &next_u, compensated);
                let damping = self.apply_damping_impl(&next_v, compensated);
                let residual = (0..self.num_dofs)
                    .map(|i| {
                        h * force_next[i]
                            - self.config.alpha * self.prev_force[i]
                            - inertia[i]
                            - h * elastic[i]
                            - h * damping[i]
                            + self.config.alpha * previous_elastic[i]
                            + self.config.alpha * previous_damping[i]
                    })
                    .collect::<Vec<_>>();
                let residual_norm = residual
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if residual_norm.is_finite() && residual_norm <= tolerance {
                    break;
                }
                if !residual_norm.is_finite() || refinement == max_equilibrium_refinements {
                    return Err(ImplicitDynamicsError::LinearSolve(format!(
                        "true equilibrium residual {residual_norm:.6e} exceeds {tolerance:.6e} after {refinement} refinements"
                    )));
                }
                let correction = self.effective.solve(&residual)?;
                for i in 0..self.num_dofs {
                    next_u[i] += correction[i];
                    next_a[i] += a[0] * correction[i];
                    next_v[i] += a[1] * correction[i];
                }
            }
        }
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
        self.apply_damping_impl(values, false)
    }

    fn apply_damping_impl(&self, values: &[f64], compensated: bool) -> Vec<f64> {
        let mut out = vec![0.0; self.num_dofs];
        if let Some(damping) = &self.damping {
            let part = force_matvec(damping, values, compensated);
            for (slot, value) in out.iter_mut().zip(part.iter()) {
                *slot += value;
            }
        }
        if self.config.rayleigh_mass != 0.0 {
            let part = force_matvec(&self.mass, values, compensated);
            for (slot, value) in out.iter_mut().zip(part.iter()) {
                *slot += self.config.rayleigh_mass * value;
            }
        }
        if self.config.stiffness_damping != 0.0 {
            let part = force_matvec(&self.stiffness, values, compensated);
            for (slot, value) in out.iter_mut().zip(part.iter()) {
                *slot += self.config.stiffness_damping * value;
            }
        }
        out
    }
}

fn force_matvec(matrix: &CsrMatrix<f64>, values: &[f64], compensated: bool) -> Vec<f64> {
    if !compensated {
        return matrix.apply(values);
    }
    (0..matrix.num_rows)
        .map(|row| {
            let mut high = 0.0;
            let mut low = 0.0;
            for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                let coefficient = matrix.values[entry];
                let value = values[matrix.col_indices[entry]];
                let product = coefficient * value;
                let product_error = coefficient.mul_add(value, -product);
                let sum = high + product;
                let recovered = sum - high;
                low += (high - (sum - recovered)) + (product - recovered) + product_error;
                high = sum;
            }
            high + low
        })
        .collect()
}

fn factor_effective(
    system: &ImplicitDynamicsSystem,
    config: &ImplicitDynamicsConfig,
    coeffs: &[f64; 6],
    cholesky: bool,
) -> Result<DynamicsFactor, ImplicitDynamicsError> {
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
    factor_matrix(&matrix, cholesky)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_force_products_retain_exact_cancellation_remainders() {
        let mut entries = TripletMatrix::new(1, 3);
        entries.push(0, 0, 1e16);
        entries.push(0, 1, 1.0);
        entries.push(0, 2, -1e16);
        let matrix = entries.to_csr();
        assert_eq!(force_matvec(&matrix, &[1.0, 1.0, 1.0], true), vec![1.0]);
        let epsilon = 2.0_f64.powi(-27);
        let mut entries = TripletMatrix::new(1, 2);
        entries.push(0, 0, 1.0 + epsilon);
        entries.push(0, 1, -1.0);
        assert_eq!(
            force_matvec(&entries.to_csr(), &[1.0 - epsilon, 1.0], true),
            vec![-epsilon * epsilon]
        );
    }

    #[test]
    fn accurate_checked_hht_preserves_balance_kinematics_and_failed_state() {
        let config = ImplicitDynamicsConfig {
            rayleigh_mass: 0.02,
            stiffness_damping: 0.001,
            ..ImplicitDynamicsConfig::hht(-0.1)
        };
        let mut integrator =
            ImplicitDynamicsIntegrator::new_spd(sdof_system(2.0, 0.4, 11.0), config, 0.02).unwrap();
        let damping = 0.4 + 0.02 * 2.0 + 0.001 * 11.0;
        let mut previous_force = 0.7;
        integrator
            .initialize(&[0.25], &[-0.1], &[previous_force])
            .unwrap();
        for step in 1..=12 {
            if step == 6 {
                integrator.set_timestep(0.01).unwrap();
            }
            let dt = integrator.dt();
            let previous = integrator.state().clone();
            let force = 0.7 + 0.1 * step as f64;
            let next = integrator.step_checked_accurate(&[force], 1e-11).unwrap();
            let h = 1.0 + config.alpha;
            let residual = 2.0 * next.acceleration[0]
                + h * damping * next.velocity[0]
                + h * 11.0 * next.displacement[0]
                - config.alpha * damping * previous.velocity[0]
                - config.alpha * 11.0 * previous.displacement[0]
                - h * force
                + config.alpha * previous_force;
            assert!(residual.abs() <= 1e-11);
            let du = next.displacement[0]
                - previous.displacement[0]
                - dt * previous.velocity[0]
                - dt * dt
                    * ((0.5 - config.beta) * previous.acceleration[0]
                        + config.beta * next.acceleration[0]);
            let dv = next.velocity[0]
                - previous.velocity[0]
                - dt * ((1.0 - config.gamma) * previous.acceleration[0]
                    + config.gamma * next.acceleration[0]);
            assert!(du.abs() <= 1e-13 && dv.abs() <= 1e-13);
            previous_force = force;
        }
        let previous = integrator.state().clone();
        let force_history = integrator.prev_force.clone();
        assert!(integrator.step_checked_accurate(&[1e308], 1e-11).is_err());
        assert_eq!(integrator.state().time, previous.time);
        assert_eq!(integrator.state().displacement, previous.displacement);
        assert_eq!(integrator.state().velocity, previous.velocity);
        assert_eq!(integrator.state().acceleration, previous.acceleration);
        assert_eq!(integrator.prev_force, force_history);
        assert!(integrator.step_checked_accurate(&[0.8], f64::NAN).is_err());
        assert!(integrator.step_checked_accurate(&[0.8], 1e-11).is_ok());
    }

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
    fn spd_constructor_matches_lu_for_coupled_hht_and_timestep_change() {
        let matrix = |entries: &[(usize, usize, f64)]| {
            let mut triplets = TripletMatrix::new(2, 2);
            for &(row, column, value) in entries {
                triplets.push(row, column, value);
            }
            triplets.to_csr()
        };
        let system = ImplicitDynamicsSystem {
            mass: matrix(&[(0, 0, 2.0), (0, 1, 0.3), (1, 0, 0.3), (1, 1, 1.0)]),
            stiffness: matrix(&[(0, 0, 40.0), (0, 1, -10.0), (1, 0, -10.0), (1, 1, 20.0)]),
            damping: Some(matrix(&[
                (0, 0, 0.5),
                (0, 1, -0.1),
                (1, 0, -0.1),
                (1, 1, 0.4),
            ])),
        };
        let config = ImplicitDynamicsConfig {
            rayleigh_mass: 0.02,
            stiffness_damping: 0.001,
            ..ImplicitDynamicsConfig::hht(-0.1)
        };
        let mut lu = ImplicitDynamicsIntegrator::new(system.clone(), config, 0.02).unwrap();
        let mut spd = ImplicitDynamicsIntegrator::new_spd(system, config, 0.02).unwrap();
        for integrator in [&mut lu, &mut spd] {
            integrator
                .initialize(&[0.1, -0.2], &[0.3, 0.1], &[0.5, -0.2])
                .unwrap();
        }
        for step in 0..100 {
            if step == 20 {
                lu.set_timestep(0.01).unwrap();
                spd.set_timestep(0.01).unwrap();
            }
            let force = [(step as f64 * 0.03).sin(), (step as f64 * 0.05).cos()];
            lu.step_checked(&force, 1e-10).unwrap();
            spd.step_checked(&force, 1e-10).unwrap();
            for (a, b) in lu
                .state()
                .displacement
                .iter()
                .zip(&spd.state().displacement)
                .chain(lu.state().velocity.iter().zip(&spd.state().velocity))
                .chain(
                    lu.state()
                        .acceleration
                        .iter()
                        .zip(&spd.state().acceleration),
                )
            {
                assert!((a - b).abs() < 1e-10);
            }
            assert!(matches!(spd.effective, DynamicsFactor::Cholesky(_)));
        }
    }

    #[test]
    fn spd_constructor_rejects_asymmetry_and_nonpositive_mass() {
        let mut mass = TripletMatrix::new(2, 2);
        mass.push(0, 0, 1.0);
        mass.push(1, 1, 1.0);
        let mut stiffness = TripletMatrix::new(2, 2);
        stiffness.push(0, 0, 2.0);
        stiffness.push(1, 1, 2.0);
        stiffness.push(0, 1, 1.0);
        let system = ImplicitDynamicsSystem {
            mass: mass.to_csr(),
            stiffness: stiffness.to_csr(),
            damping: None,
        };
        assert!(
            ImplicitDynamicsIntegrator::new(
                system.clone(),
                ImplicitDynamicsConfig::default(),
                0.01
            )
            .is_ok()
        );
        assert!(matches!(
            ImplicitDynamicsIntegrator::new_spd(system, ImplicitDynamicsConfig::default(), 0.01),
            Err(ImplicitDynamicsError::InvalidConfig(_))
        ));
        assert!(
            ImplicitDynamicsIntegrator::new_spd(
                sdof_system(-1.0, 0.0, 0.0),
                ImplicitDynamicsConfig::default(),
                0.01
            )
            .is_err()
        );
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

    #[test]
    fn checked_step_preserves_stiff_static_equilibrium() {
        let mut stiffness = TripletMatrix::new(2, 2);
        for (row, col, value) in [
            (0, 0, 1e12 + 1.0),
            (0, 1, -1e12),
            (1, 0, -1e12),
            (1, 1, 1e12 + 1.0),
        ] {
            stiffness.push(row, col, value);
        }
        let mut mass = TripletMatrix::new(2, 2);
        mass.push(0, 0, 1.0);
        mass.push(1, 1, 1.0);
        let system = ImplicitDynamicsSystem {
            mass: mass.to_csr(),
            stiffness: stiffness.to_csr(),
            damping: None,
        };
        let mut plain = ImplicitDynamicsIntegrator::new(
            system.clone(),
            ImplicitDynamicsConfig::default(),
            0.01,
        )
        .unwrap();
        let mut checked = ImplicitDynamicsIntegrator::new(
            system.clone(),
            ImplicitDynamicsConfig::default(),
            0.01,
        )
        .unwrap();
        for integrator in [&mut plain, &mut checked] {
            integrator
                .initialize(&[1.0, 1.0], &[0.0, 0.0], &[1.0, 1.0])
                .unwrap();
        }
        let residual = |state: &ImplicitDynamicsState| {
            let inertia = system.mass.apply(&state.acceleration);
            let elastic = system.stiffness.apply(&state.displacement);
            inertia
                .iter()
                .zip(elastic)
                .map(|(m, k)| (m + k - 1.0).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let original_residual = residual(plain.step(&[1.0, 1.0]).unwrap());
        assert!(
            original_residual > 1e-8,
            "counterexample must exercise effective-system cancellation: {original_residual}"
        );
        let state = checked.step_checked(&[1.0, 1.0], 1e-8).unwrap();
        assert!(residual(state) <= 1e-8);
        assert!(
            state
                .displacement
                .iter()
                .all(|value| (*value - 1.0).abs() <= 1e-12)
        );
        for i in 0..2 {
            assert!(
                (state.displacement[i] - 1.0 - 0.25 * 0.01_f64.powi(2) * state.acceleration[i])
                    .abs()
                    <= 5e-16
            );
            assert!((state.velocity[i] - 0.5 * 0.01 * state.acceleration[i]).abs() <= 5e-16);
        }
    }

    #[test]
    fn checked_hht_step_preserves_weighted_balance_and_kinematics() {
        let config = ImplicitDynamicsConfig::hht(-0.1);
        let dt = 0.02;
        let mut integrator =
            ImplicitDynamicsIntegrator::new(sdof_system(2.0, 0.4, 11.0), config, dt).unwrap();
        let mut previous_force = 0.7;
        integrator
            .initialize(&[0.25], &[-0.1], &[previous_force])
            .unwrap();
        for step in 1..=12 {
            let previous = integrator.state().clone();
            let force = 0.7 + 0.1 * step as f64;
            let next = integrator.step_checked(&[force], 1e-11).unwrap();
            let h = 1.0 + config.alpha;
            let residual = 2.0 * next.acceleration[0]
                + h * 0.4 * next.velocity[0]
                + h * 11.0 * next.displacement[0]
                - config.alpha * 0.4 * previous.velocity[0]
                - config.alpha * 11.0 * previous.displacement[0]
                - h * force
                + config.alpha * previous_force;
            assert!(residual.abs() <= 1e-11);
            let displacement_defect = next.displacement[0]
                - previous.displacement[0]
                - dt * previous.velocity[0]
                - dt * dt
                    * ((0.5 - config.beta) * previous.acceleration[0]
                        + config.beta * next.acceleration[0]);
            let velocity_defect = next.velocity[0]
                - previous.velocity[0]
                - dt * ((1.0 - config.gamma) * previous.acceleration[0]
                    + config.gamma * next.acceleration[0]);
            assert!(displacement_defect.abs() <= 1e-13);
            assert!(velocity_defect.abs() <= 1e-13);
            previous_force = force;
        }
    }

    #[test]
    fn failed_checked_step_preserves_state_and_force_history() {
        let mut integrator = ImplicitDynamicsIntegrator::new(
            sdof_system(1.3, 0.31, 7.1),
            ImplicitDynamicsConfig::default(),
            0.01,
        )
        .unwrap();
        integrator.initialize(&[0.2], &[0.3], &[0.7]).unwrap();
        let previous = integrator.state().clone();
        let force = integrator.prev_force.clone();
        assert!(integrator.step_checked(&[1e308], 1e-10).is_err());
        assert_eq!(integrator.state.time, previous.time);
        assert_eq!(integrator.state.displacement, previous.displacement);
        assert_eq!(integrator.state.velocity, previous.velocity);
        assert_eq!(integrator.state.acceleration, previous.acceleration);
        assert_eq!(integrator.prev_force, force);
        assert!(integrator.step_checked(&[0.8], 1e-10).is_ok());
    }
}
