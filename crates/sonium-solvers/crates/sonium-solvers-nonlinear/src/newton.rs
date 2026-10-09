//! Newton-Raphson nonlinear solver
//!
//! Solves `F(x) = 0` given a residual function and a linear solve for the Jacobian system.
//! The caller provides:
//! - A residual function `F(x) -> r`
//! - A Jacobian solve `J(x) \\ r -> delta_x` (the caller chooses how to form and solve J)
//!
//! This keeps the Newton iteration loop generic — no dependence on meshes, elements, or
//! assembly. FEM-specific assembly lives in `sonium-fem`.
//!
//! ## Inexact Newton
//!
//! [`newton_raphson_inexact`] extends the basic iteration with:
//! - an Eisenstat–Walker forcing term: the per-iteration linear-solve tolerance
//!   `η_k = min(η_max, ‖F_k‖^q)` is passed to the Jacobian-solve closure so the
//!   inner solver does not over-solve early Newton steps;
//! - a line-search failure policy ([`LineSearchFailure`]): by default a failed
//!   backtracking search returns [`NewtonError::LineSearchFailed`] instead of
//!   silently accepting the smallest trial step;
//! - an optional trust-region globalisation ([`TrustRegionConfig`]) that clips
//!   the Newton step to a 2-norm radius and accepts/rejects via the
//!   actual/predicted reduction ratio of the Gauss–Newton model.
//!
//! ## Pseudo-transient continuation (Ψtc)
//!
//! [`newton_raphson_continuation`] is a matrix-free Newton–Krylov variant that
//! globalises the iteration with pseudo-transient continuation (Kelley &
//! Keyes, "Convergence analysis of pseudo-transient continuation", SIAM
//! J. Numer. Anal. 35(2), 1998): instead of the Newton system `J δ = −F` it
//! solves the time-accurate system `(I/Δt + J) δ = −F`, which is diagonally
//! dominated for small Δt and reduces to a damped steepest-descent step. The
//! pseudo-timestep Δt is evolved by switched evolution relaxation (SER),
//! `Δt_{n+1} = Δt_n · ‖F_n‖ / ‖F_{n+1}‖`, so the iteration approaches pure
//! Newton (Δt → ∞) as the residual contracts, recovering quadratic terminal
//! convergence. The diagonal shift is applied by wrapping the user's Jacobian
//! matvec — user Jacobians are unchanged.
//!
//! With [`InexactNewtonConfig::auto_pseudo_transient`] the solver starts as
//! plain Newton and escalates to Ψtc when the line search fails.
//!
//! ## Scalar type
//!
//! Generic over [`ComplexField`](crate::ComplexField): works with f32, f64,
//! Complex32, and Complex64.

use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use thiserror::Error;

use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, SolverError, SolverFailure,
    SolverResultContract, SolverStatus,
};
use sonium_solvers_iterative::{GmresConfig, gmres};

/// Line search strategy for Newton-Raphson.
#[derive(Debug, Clone, Copy)]
pub enum LineSearch {
    /// No line search — use fixed damping factor.
    None,
    /// Backtracking Armijo line search.
    ///
    /// Finds α ∈ (0,1] such that `‖F(x + α δ)‖ ≤ (1 − c₁ α) ‖F(x)‖`
    /// by halving α until the sufficient decrease condition is met.
    Backtracking {
        /// Sufficient decrease parameter c₁ (typical: 1e-4).
        c1: f64,
        /// Maximum number of backtracking halvings.
        max_halvings: usize,
    },
}

impl LineSearch {
    /// Default backtracking with c₁ = 1e-4, up to 10 halvings.
    pub fn backtracking() -> Self {
        Self::Backtracking {
            c1: 1e-4,
            max_halvings: 10,
        }
    }
}

/// Forcing-term strategy for inexact Newton: the per-iteration relative
/// tolerance `η_k` handed to the Jacobian-solve closure.
#[derive(Debug, Clone, Copy)]
pub enum ForcingTerm {
    /// No forcing term — `η = 0` is passed and the caller picks its own
    /// (fixed) inner tolerance. This is the legacy behaviour.
    None,
    /// Eisenstat–Walker choice 2: `η_k = min(η_max, ‖F_k‖^q)`.
    ///
    /// Early Newton steps are solved loosely; as `‖F_k‖ → 0` the inner
    /// tolerance tightens, preserving the outer convergence order while
    /// reducing the total number of inner iterations.
    ///
    /// The published safeguard (Eisenstat–Walker 1996, §2.2) is applied:
    /// whenever the previous forcing term satisfies `η_{k-1}^q > 0.1`, the
    /// new term is bounded below by `η_{k-1}^{(1+√5)/2}`. Without this, a
    /// single fast residual drop after a loosely-solved step collapses η and
    /// the next inner system is massively over-solved.
    EisenstatWalker {
        /// Cap on the forcing term (typical: 0.5).
        eta_max: f64,
        /// Exponent q ∈ (0, 2] (q = 2 gives quadratic outer convergence
        /// when the inner systems are solved accurately enough).
        q: f64,
    },
}

/// Policy applied when the backtracking line search finds no acceptable step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchFailure {
    /// Report [`NewtonError::LineSearchFailed`] and restore the pre-step
    /// iterate. Default for [`newton_raphson_inexact`].
    Fail,
    /// Legacy behaviour: accept the smallest trial step and keep iterating.
    AcceptSmallestStep,
}

/// Trust-region globalisation for Newton-Raphson.
///
/// The Newton step is clipped to a 2-norm trust radius (`d = τ δ` with
/// `τ = min(1, radius / ‖δ‖)`) and accepted or rejected by the ratio of
/// actual to predicted residual-norm reduction, where the predicted
/// reduction uses the Gauss–Newton model `J d ≈ −τ F`:
/// `pred = (2τ − τ²) ‖F‖²`.
#[derive(Debug, Clone, Copy)]
pub struct TrustRegionConfig {
    /// Initial trust radius.
    pub initial_radius: f64,
    /// Maximum trust radius.
    pub max_radius: f64,
    /// Minimum trust radius — below this the solve fails with
    /// [`NewtonError::TrustRegionCollapsed`].
    pub min_radius: f64,
    /// Acceptance threshold on the reduction ratio ρ (typical: 1e-4).
    pub accept_ratio: f64,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 100.0,
            min_radius: 1e-12,
            accept_ratio: 1e-4,
        }
    }
}

/// Continuation strategy for Newton-Raphson globalisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContinuationStrategy {
    /// No continuation — plain (possibly line-search globalised) Newton.
    None,
    /// Pseudo-transient continuation (Ψtc, Kelley & Keyes 1998).
    ///
    /// Solves `(I/Δt + J) δ = −F` instead of `J δ = −F`. For small Δt the
    /// system is diagonally dominated and the step is a damped
    /// steepest-descent-like move; as Δt grows the step approaches the full
    /// Newton step and quadratic convergence is recovered.
    ///
    /// After each accepted step the pseudo-timestep is updated by switched
    /// evolution relaxation (SER): `Δt ← clamp(Δt · ‖F_old‖/‖F_new‖, dt0,
    /// max_dt)`, with the per-step growth factor capped at `growth`. When a
    /// step is rejected by the line search, Δt is halved (floored at `dt0`)
    /// and the shifted system is re-solved at the unchanged iterate.
    PseudoTransient {
        /// Initial pseudo-timestep Δt₀ (> 0). Small values give steepest-
        /// descent-like initial steps; typical: 1e-3 to 1e-1.
        dt0: f64,
        /// Cap on the per-step SER growth factor (≥ 1). Bounds
        /// `Δt_{n+1} ≤ growth · Δt_n` regardless of the residual ratio.
        growth: f64,
        /// Maximum pseudo-timestep (≥ dt0). Once Δt ≈ max_dt the shifted
        /// system is numerically indistinguishable from Newton's; a large
        /// value (e.g. 1e12) recovers full quadratic terminal convergence.
        max_dt: f64,
    },
}

impl ContinuationStrategy {
    /// Default Ψtc parameters: Δt₀ = 1e-3, growth cap 10, Δt_max = 1e12.
    pub fn pseudo_transient() -> Self {
        Self::PseudoTransient {
            dt0: 1e-3,
            growth: 10.0,
            max_dt: 1e12,
        }
    }

    /// The `(dt0, growth, max_dt)` parameters, or `None` when continuation
    /// is disabled.
    fn params(self) -> Option<(f64, f64, f64)> {
        match self {
            Self::None => None,
            Self::PseudoTransient {
                dt0,
                growth,
                max_dt,
            } => Some((dt0, growth, max_dt)),
        }
    }
}

/// Golden-ratio exponent `(1+√5)/2` used by the Eisenstat–Walker choice-2
/// safeguard: `η_k ≥ η_{k-1}^φ` whenever `η_{k-1}^q > 0.1`.
const EW_SAFEGUARD_PHI: f64 = std::f64::consts::GOLDEN_RATIO;

/// Eisenstat–Walker choice-2 forcing term with the published safeguard
/// (Eisenstat & Walker, "Choosing the forcing terms in an inexact Newton
/// method", SIAM J. Sci. Comput. 17(1), 1996, §2.2).
///
/// `eta_prev` is the forcing term used at the previous Newton step (pass
/// `None` on the first step, where it defaults to `eta_max`).
fn eisenstat_walker_eta(residual_norm: f64, eta_max: f64, q: f64, eta_prev: Option<f64>) -> f64 {
    let raw = eta_max.min(residual_norm.powf(q));
    let prev = eta_prev.unwrap_or(eta_max);
    if prev.powf(q) > 0.1 {
        raw.max(prev.powf(EW_SAFEGUARD_PHI))
    } else {
        raw
    }
}

/// Fallible `Real::from_f64` for hot paths: exotic [`ComplexField`] impls
/// may not represent a constant that is trivially fine for f32/f64. A
/// conversion failure is a typed error, never a panic.
fn real_from_f64<T: ComplexField>(
    value: f64,
    stage: &'static str,
    iteration: usize,
) -> Result<T::Real, NewtonError>
where
    T::Real: FromPrimitive,
{
    T::Real::from_f64(value).ok_or_else(|| NewtonError::NumericalError {
        iteration,
        stage,
        reason: format!(
            "scalar constant {value:.3e} is not representable in the working precision"
        ),
    })
}

/// Fallible `Real::to_f64` for hot paths (same rationale as
/// [`real_from_f64`]).
fn real_to_f64<T: ComplexField>(
    value: T::Real,
    stage: &'static str,
    iteration: usize,
) -> Result<f64, NewtonError>
where
    T::Real: ToPrimitive,
{
    value.to_f64().ok_or_else(|| NewtonError::NumericalError {
        iteration,
        stage,
        reason: "scalar value is not representable as f64".to_string(),
    })
}

/// Newton-Raphson solver configuration.
///
/// Algorithm parameters are stored as f64 and converted to the working
/// precision at solve time.
#[derive(Debug, Clone)]
pub struct NewtonConfig {
    /// Maximum number of Newton iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the residual norm (absolute).
    pub absolute_tolerance: f64,
    /// Convergence tolerance on the residual norm (relative to initial).
    pub relative_tolerance: f64,
    /// Fixed damping factor (1.0 = full Newton step). Ignored if
    /// `line_search` is not `None`.
    pub damping: f64,
    /// Line search strategy. Set to `LineSearch::Backtracking { .. }` for
    /// globalized Newton suitable for hyperelasticity and other strongly
    /// nonlinear problems.
    pub line_search: LineSearch,
    /// Verbosity level (0 = quiet, 1 = per-iteration, 2+ = detailed).
    pub verbosity: usize,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-8,
            damping: 1.0,
            line_search: LineSearch::None,
            verbosity: 0,
        }
    }
}

/// Configuration for [`newton_raphson_inexact`].
///
/// Wraps the legacy [`NewtonConfig`] with the inexact-Newton extensions:
/// forcing term, line-search failure policy, and optional trust-region
/// globalisation.
#[derive(Debug, Clone)]
pub struct InexactNewtonConfig {
    /// Base Newton parameters (iteration limits, tolerances, damping,
    /// line search, verbosity).
    pub newton: NewtonConfig,
    /// Forcing term `η_k` passed to the Jacobian-solve closure.
    pub forcing: ForcingTerm,
    /// What to do when backtracking finds no acceptable step.
    pub line_search_failure: LineSearchFailure,
    /// Trust-region globalisation. When `Some`, it replaces
    /// `newton.line_search` as the step-acceptance mechanism.
    pub trust_region: Option<TrustRegionConfig>,
    /// Continuation strategy. Honoured only by the matrix-free entry point
    /// [`newton_raphson_continuation`], which applies the `I/Δt` diagonal
    /// shift by wrapping the Jacobian matvec; the Jacobian-*solve* entry
    /// points ([`newton_raphson`], [`newton_raphson_inexact`]) ignore it,
    /// since a shift cannot be pushed through a black-box linear solve.
    pub continuation: ContinuationStrategy,
    /// When `true` and `continuation` is [`ContinuationStrategy::None`],
    /// [`newton_raphson_continuation`] starts as plain Newton and
    /// automatically escalates to pseudo-transient continuation (with the
    /// default parameters of [`ContinuationStrategy::pseudo_transient`])
    /// when the backtracking line search fails, or when an accepted step
    /// needed heavy damping (see `ESCALATION_HALVING_THRESHOLD`) — a sign
    /// the local Newton model is poor.
    pub auto_pseudo_transient: bool,
    /// Inner Krylov budget `(max_iterations, restart)` for the GMRES solves
    /// inside [`newton_raphson_continuation`]. `None` keeps the legacy budget
    /// `(20, min(n, 30))`, which suits small-to-medium systems; large stiff
    /// systems can widen the Krylov basis without editing the crate.
    pub continuation_gmres_budget: Option<(usize, usize)>,
}

impl Default for InexactNewtonConfig {
    fn default() -> Self {
        Self {
            newton: NewtonConfig::default(),
            forcing: ForcingTerm::None,
            line_search_failure: LineSearchFailure::Fail,
            trust_region: None,
            continuation: ContinuationStrategy::None,
            auto_pseudo_transient: false,
            continuation_gmres_budget: None,
        }
    }
}

impl InexactNewtonConfig {
    /// View of a plain [`NewtonConfig`]: no forcing term, fail on line-search
    /// failure, no trust region, no continuation.
    fn from_legacy(newton: NewtonConfig) -> Self {
        Self {
            newton,
            ..Default::default()
        }
    }
}

/// Result of a Newton-Raphson solve, generic over the scalar type.
#[derive(Debug, Clone)]
pub struct NewtonSolution<T: ComplexField> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub residual: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Detailed status.
    pub status: SolverStatus,
    /// Whether the nonlinear residual attained the requested tolerance. A
    /// linear backward-error metric is not defined for a general nonlinear
    /// residual/Jacobian pair.
    pub attainable_accuracy: AccuracyStatus,
    /// Total number of residual evaluations: the initial evaluation, one
    /// per Newton iteration, every line-search / trust-region trial, and
    /// the final check. Always `>= 1` on any returned solution.
    pub residual_evaluations: usize,
    /// Total number of Jacobian evaluations: one per Jacobian-solve call
    /// for the [`newton_raphson`]/[`newton_raphson_inexact`] entry points.
    /// Always 0 for [`newton_raphson_continuation`], which is matrix-free
    /// and never assembles a Jacobian (its linear work is counted in
    /// `inner_iterations` instead).
    pub jacobian_evaluations: usize,
    /// Total inner Krylov iterations, accumulated from the inner GMRES
    /// solves of [`newton_raphson_continuation`]. Always 0 for the
    /// black-box Jacobian-solve entry points, whose inner iteration counts
    /// are not visible through the solve closure.
    pub inner_iterations: usize,
}

impl<T: ComplexField> SolverResultContract for NewtonSolution<T> {
    fn solver_status(&self) -> SolverStatus {
        self.status
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.x.iter().any(|value| !value.norm().is_finite())
            || !self.residual.is_finite()
            || self.residual < 0.0
        {
            return Err(SolverError::OperationFailed {
                operation: "Newton result contract",
                reason: "solution or residual is non-finite or negative".into(),
            });
        }
        if self.converged != (self.status == SolverStatus::Converged) {
            return Err(SolverError::OperationFailed {
                operation: "Newton result contract",
                reason: "converged flag and solver status disagree".into(),
            });
        }
        if self.converged && self.attainable_accuracy != AccuracyStatus::Attained {
            return Err(SolverError::OperationFailed {
                operation: "Newton result contract",
                reason: "converged result must report attained accuracy".into(),
            });
        }
        if self.residual_evaluations == 0 {
            return Err(SolverError::OperationFailed {
                operation: "Newton result contract",
                reason: "a returned solution always evaluates the residual at least once".into(),
            });
        }
        Ok(())
    }
}

/// Errors from the Newton-Raphson solver.
#[derive(Debug, Error)]
pub enum NewtonError {
    #[error("Newton-Raphson did not converge in {0} iterations (residual = {1:.2e})")]
    NotConverged(usize, f64),
    #[error("Linear solve failed at iteration {iteration}: {message}")]
    LinearSolveFailed { iteration: usize, message: String },
    #[error("Newton numerical error at iteration {iteration} during {stage}: {reason}")]
    NumericalError {
        iteration: usize,
        stage: &'static str,
        reason: String,
    },
    #[error("Residual diverged at iteration {0} (residual = {1:.2e})")]
    Diverged(usize, f64),
    #[error(
        "Line search failed at iteration {iteration}: no sufficient decrease after {halvings} halvings (residual = {residual:.2e})"
    )]
    LineSearchFailed {
        iteration: usize,
        halvings: usize,
        residual: f64,
    },
    #[error(
        "Trust region collapsed at iteration {iteration}: radius {radius:.2e} below minimum (residual = {residual:.2e})"
    )]
    TrustRegionCollapsed {
        iteration: usize,
        radius: f64,
        residual: f64,
    },
    #[error("Invalid Newton configuration: {0}")]
    InvalidConfig(String),
}

impl NewtonError {
    /// Return the canonical solver status corresponding to this failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NotConverged(..) => SolverStatus::MaxIterationsReached,
            Self::Diverged(..) => SolverStatus::Diverged,
            Self::LineSearchFailed { .. } | Self::TrustRegionCollapsed { .. } => {
                SolverStatus::Stagnated
            }
            Self::LinearSolveFailed { .. }
            | Self::NumericalError { .. }
            | Self::InvalidConfig(_) => SolverStatus::NumericalError,
        }
    }

    /// Return the nonlinear attainable-accuracy state corresponding to this
    /// failure.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        match self {
            Self::NotConverged(..)
            | Self::Diverged(..)
            | Self::LineSearchFailed { .. }
            | Self::TrustRegionCollapsed { .. } => AccuracyStatus::NotAttained,
            Self::LinearSolveFailed { .. }
            | Self::NumericalError { .. }
            | Self::InvalidConfig(_) => AccuracyStatus::NotChecked,
        }
    }
}

impl SolverFailure for NewtonError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotConverged(..)
            | Self::Diverged(..)
            | Self::LineSearchFailed { .. }
            | Self::TrustRegionCollapsed { .. } => SolverErrorKind::Convergence,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        NewtonError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        NewtonError::attainable_accuracy(self)
    }
}

fn validate_newton_vector<T: ComplexField>(
    values: &[T],
    expected_len: usize,
    iteration: usize,
    stage: &'static str,
) -> Result<(), NewtonError> {
    if values.len() != expected_len {
        return Err(NewtonError::NumericalError {
            iteration,
            stage,
            reason: format!(
                "length {} does not match expected length {expected_len}",
                values.len()
            ),
        });
    }
    if values.iter().any(|value| !value.norm().is_finite()) {
        return Err(NewtonError::NumericalError {
            iteration,
            stage,
            reason: "vector contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn validate_newton_config(config: &InexactNewtonConfig) -> Result<(), NewtonError> {
    let base = &config.newton;
    if base.max_iterations == 0 {
        return Err(NewtonError::InvalidConfig(
            "max_iterations must be non-zero".to_string(),
        ));
    }
    if !base.absolute_tolerance.is_finite() || base.absolute_tolerance < 0.0 {
        return Err(NewtonError::InvalidConfig(
            "absolute_tolerance must be finite and non-negative".to_string(),
        ));
    }
    if !base.relative_tolerance.is_finite() || base.relative_tolerance < 0.0 {
        return Err(NewtonError::InvalidConfig(
            "relative_tolerance must be finite and non-negative".to_string(),
        ));
    }
    if !base.damping.is_finite() || base.damping <= 0.0 {
        return Err(NewtonError::InvalidConfig(
            "damping must be finite and positive".to_string(),
        ));
    }
    if let LineSearch::Backtracking { c1, max_halvings } = base.line_search
        && (!c1.is_finite() || !(0.0..1.0).contains(&c1) || max_halvings == 0)
    {
        return Err(NewtonError::InvalidConfig(
            "backtracking requires 0 < c1 < 1 and max_halvings > 0".to_string(),
        ));
    }
    if let ForcingTerm::EisenstatWalker { eta_max, q } = config.forcing
        && (!eta_max.is_finite()
            || !(0.0..=1.0).contains(&eta_max)
            || eta_max <= 0.0
            || !q.is_finite()
            || !(0.0..=2.0).contains(&q)
            || q <= 0.0)
    {
        return Err(NewtonError::InvalidConfig(
            "EisenstatWalker requires 0 < eta_max <= 1 and 0 < q <= 2 (both finite)".to_string(),
        ));
    }
    if let ContinuationStrategy::PseudoTransient {
        dt0,
        growth,
        max_dt,
    } = config.continuation
        && (!dt0.is_finite()
            || !growth.is_finite()
            || !max_dt.is_finite()
            || dt0 <= 0.0
            || growth < 1.0
            || max_dt < dt0)
    {
        return Err(NewtonError::InvalidConfig(format!(
            "pseudo-transient parameters must satisfy dt0 > 0, growth >= 1, \
             max_dt >= dt0 (got dt0 = {dt0}, growth = {growth}, max_dt = {max_dt})"
        )));
    }
    if let Some((max_iter, restart)) = config.continuation_gmres_budget
        && (max_iter == 0 || restart == 0)
    {
        return Err(NewtonError::InvalidConfig(
            "continuation_gmres_budget requires non-zero max_iterations and restart".to_string(),
        ));
    }
    if let Some(trust) = config.trust_region
        && (!trust.initial_radius.is_finite()
            || !trust.max_radius.is_finite()
            || !trust.min_radius.is_finite()
            || !trust.accept_ratio.is_finite()
            || trust.initial_radius <= 0.0
            || trust.min_radius <= 0.0
            || trust.max_radius < trust.initial_radius
            || trust.min_radius > trust.initial_radius
            || !(0.0..=1.0).contains(&trust.accept_ratio))
    {
        return Err(NewtonError::InvalidConfig(
            "trust-region radii and acceptance ratio are inconsistent or non-finite".to_string(),
        ));
    }
    Ok(())
}

/// Solve a nonlinear system `F(x) = 0` using Newton-Raphson iteration.
///
/// Generic over any [`ComplexField`] type: works with f32, f64, Complex32,
/// and Complex64.
///
/// The caller provides two closures:
/// - `residual_fn(x) -> Vec<T>`: evaluates the residual `F(x)`
/// - `jacobian_solve_fn(x, rhs) -> Result<Vec<T>, String>`: solves `J(x) * delta = rhs`
///   for `delta`, where `J` is the Jacobian.
///
/// The iteration is: `x_{n+1} = x_n - damping * J(x_n)^{-1} F(x_n)`
///
/// This entry point uses no forcing term and no trust region, and a failed
/// backtracking line search returns [`NewtonError::LineSearchFailed`]. Use
/// [`newton_raphson_inexact`] for Eisenstat–Walker forcing, a configurable
/// line-search failure policy, and trust-region globalisation.
pub fn newton_raphson<T, R, J>(
    initial_guess: &[T],
    residual_fn: R,
    jacobian_solve_fn: J,
    config: &NewtonConfig,
) -> Result<NewtonSolution<T>, NewtonError>
where
    T: ComplexField,
    T::Real: FromPrimitive,
    R: Fn(&[T]) -> Vec<T>,
    J: Fn(&[T], &[T]) -> Result<Vec<T>, String>,
{
    let ext = InexactNewtonConfig::from_legacy(config.clone());
    catch_unwind(AssertUnwindSafe(|| {
        newton_core(
            initial_guess,
            residual_fn,
            |x, rhs, _eta| jacobian_solve_fn(x, rhs),
            &ext,
        )
    }))
    .map_err(|payload| callback_panic_error("Newton iteration", payload))?
}

/// Solve `F(x) = 0` with inexact Newton iteration.
///
/// Same contract as [`newton_raphson`], but the Jacobian-solve closure
/// receives a third argument `η_k` — the relative tolerance for the inner
/// linear solve at this Newton step (0.0 when [`ForcingTerm::None`]):
///
/// - `jacobian_solve_fn(x, rhs, eta) -> Result<Vec<T>, String>` should solve
///   `J(x) * delta = rhs` to roughly `‖J delta − rhs‖ ≤ eta * ‖rhs‖`.
///
/// Globalisation is controlled by [`InexactNewtonConfig`]: the base line
/// search (with a configurable failure policy) or an optional trust region.
pub fn newton_raphson_inexact<T, R, J>(
    initial_guess: &[T],
    residual_fn: R,
    jacobian_solve_fn: J,
    config: &InexactNewtonConfig,
) -> Result<NewtonSolution<T>, NewtonError>
where
    T: ComplexField,
    T::Real: FromPrimitive,
    R: Fn(&[T]) -> Vec<T>,
    J: Fn(&[T], &[T], f64) -> Result<Vec<T>, String>,
{
    catch_unwind(AssertUnwindSafe(|| {
        newton_core(initial_guess, residual_fn, jacobian_solve_fn, config)
    }))
    .map_err(|payload| callback_panic_error("inexact Newton iteration", payload))?
}

/// Maximum number of consecutive line-search rejections tolerated in the
/// continuation solver before honouring the [`LineSearchFailure`] policy.
/// Each rejection re-solves the shifted system at the unchanged iterate with
/// a halved Δt (or escalates to Ψtc), so the retry budget also bounds the
/// number of Krylov solves per Newton step.
const MAX_STEP_RETRIES: usize = 8;

/// Number of Armijo halvings beyond which an accepted step counts as
/// "heavily damped" — a sign that the local Newton model is poor and
/// automatic escalation to Ψtc should engage (with
/// [`InexactNewtonConfig::auto_pseudo_transient`]).
const ESCALATION_HALVING_THRESHOLD: usize = 4;

/// Initial Δt for automatic escalation, scaled from the failed (or heavily
/// damped) Newton step: where the diagonal shift dominates, the first Ψtc
/// step has length ≈ Δt·‖F‖ = ‖δ‖/8 — a conservative but steady crawl
/// across the region where the plain Newton model was poor. An oversized
/// value is repaired by the Δt-halving retry loop, so erring small is the
/// safe side. Clamped to `[dt0, max_dt]`.
fn escalation_dt(delta_norm: f64, residual_norm: f64, dt0: f64, max_dt: f64) -> f64 {
    let heuristic = if residual_norm > 0.0 {
        delta_norm / (8.0 * residual_norm)
    } else {
        dt0
    };
    heuristic.clamp(dt0, max_dt)
}

/// SER pseudo-timestep update (Kelley & Keyes 1998, eq. (3.4)):
/// `Δt_{n+1} = Δt_n · ‖F_n‖ / ‖F_{n+1}‖`, with the growth factor capped at
/// `growth` and the result clamped to `[dt0, max_dt]`.
fn pseudo_transient_dt_update(
    dt: f64,
    norm_old: f64,
    norm_new: f64,
    dt0: f64,
    growth: f64,
    max_dt: f64,
) -> f64 {
    let ratio = if norm_new > 0.0 {
        norm_old / norm_new
    } else {
        growth
    };
    (dt * ratio.min(growth)).clamp(dt0, max_dt)
}

/// Matrix-free Jacobian wrapper applying the pseudo-transient diagonal shift.
///
/// Applies `v ↦ (shift · I + J(x)) v`, where `J(x)` is evaluated by the
/// user's matvec closure at the fixed point `x`. `shift = 0` is the plain
/// Jacobian (pure Newton); `shift = 1/Δt` is the Ψtc system matrix. The user
/// Jacobian is untouched — the shift lives entirely in this wrapper.
struct ShiftedJacobian<'a, T: ComplexField, A>
where
    A: Fn(&[T], &[T]) -> Vec<T> + Sync,
{
    /// Point at which the Jacobian is frozen.
    point: &'a [T],
    /// Diagonal shift `1/Δt` (0 for plain Newton).
    shift: T,
    /// User Jacobian matvec `(x, v) ↦ J(x) v`.
    jacobian_apply_fn: &'a A,
}

impl<T, A> LinearOperator<T> for ShiftedJacobian<'_, T, A>
where
    T: ComplexField,
    A: Fn(&[T], &[T]) -> Vec<T> + Sync,
{
    fn num_rows(&self) -> usize {
        self.point.len()
    }

    fn num_cols(&self) -> usize {
        self.point.len()
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        let jv = (self.jacobian_apply_fn)(self.point, x);
        // `LinearOperator::apply_into` cannot return a `Result`; propagate a
        // malformed user matvec as a non-finite vector so the Newton core's
        // normal numerical-error sentinel reports a typed failure instead of
        // panicking inside a production solve.
        if jv.len() != y.len() || jv.iter().any(|value| !value.norm().is_finite()) {
            let nan = T::Real::nan();
            y.fill(T::from_re_im(nan, T::Real::zero()));
            return;
        }
        for ((yi, jvi), xi) in y.iter_mut().zip(jv.iter()).zip(x.iter()) {
            *yi = self.shift * *xi + *jvi;
        }
    }

    /// The transpose is never exercised by the Krylov methods used here
    /// (GMRES only applies the forward operator). Both checked and raw
    /// transpose entry points reject it: no adjoint callback was supplied.
    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Unsupported
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Unsupported
    }

    fn apply_transpose_into(&self, _x: &[T], _y: &mut [T]) {
        panic!("shifted Jacobian has no transpose callback");
    }
}

/// Solve `F(x) = 0` with matrix-free Newton–Krylov iteration and optional
/// pseudo-transient continuation (Ψtc, Kelley & Keyes 1998).
///
/// Same residual contract as [`newton_raphson`], but the Jacobian is given
/// as a matvec closure:
///
/// - `jacobian_apply_fn(x, v) -> Vec<T>` computes `J(x) v`.
///
/// The linear systems are solved by the crate's restarted GMRES. When
/// [`InexactNewtonConfig::continuation`] is
/// [`ContinuationStrategy::PseudoTransient`], the solver wraps the matvec in
/// a diagonally-shifted operator and solves `(I/Δt + J) δ = −F` instead of
/// `J δ = −F`; Δt starts at `dt0`, is evolved by SER (`Δt ← Δt ·
/// ‖F_old‖/‖F_new‖`, growth capped, clamped to `[dt0, max_dt]`), and is
/// halved whenever the Armijo line search rejects a step. With a large
/// `max_dt` the iteration approaches pure Newton and recovers quadratic
/// terminal convergence.
///
/// When `continuation` is [`ContinuationStrategy::None`] and
/// [`InexactNewtonConfig::auto_pseudo_transient`] is `true`, the solver
/// starts as plain Newton–Krylov and escalates to Ψtc (default parameters)
/// on the first line-search failure.
///
/// Notes:
/// - The forcing term ([`ForcingTerm`]) is honoured: `η_k` becomes the
///   GMRES relative tolerance.
/// - Trust-region globalisation is not available here (Ψtc replaces it);
///   combine trust regions with [`newton_raphson_inexact`] instead. Passing
///   a [`TrustRegionConfig`] returns [`NewtonError::InvalidConfig`].
pub fn newton_raphson_continuation<T, R, A>(
    initial_guess: &[T],
    residual_fn: R,
    jacobian_apply_fn: A,
    config: &InexactNewtonConfig,
) -> Result<NewtonSolution<T>, NewtonError>
where
    T: ComplexField,
    T::Real: FromPrimitive,
    R: Fn(&[T]) -> Vec<T>,
    A: Fn(&[T], &[T]) -> Vec<T> + Sync,
{
    catch_unwind(AssertUnwindSafe(|| {
        newton_core_continuation(initial_guess, residual_fn, jacobian_apply_fn, config)
    }))
    .map_err(|payload| callback_panic_error("Newton continuation", payload))?
}

fn callback_panic_error(
    stage: &'static str,
    payload: Box<dyn std::any::Any + Send>,
) -> NewtonError {
    let reason = if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "callback panicked with a non-string payload".to_string()
    };
    NewtonError::NumericalError {
        iteration: 0,
        stage,
        reason,
    }
}

/// Iteration core behind [`newton_raphson_continuation`].
fn newton_core_continuation<T, R, A>(
    initial_guess: &[T],
    residual_fn: R,
    jacobian_apply_fn: A,
    config: &InexactNewtonConfig,
) -> Result<NewtonSolution<T>, NewtonError>
where
    T: ComplexField,
    T::Real: FromPrimitive,
    R: Fn(&[T]) -> Vec<T>,
    A: Fn(&[T], &[T]) -> Vec<T> + Sync,
{
    validate_newton_config(config)?;
    if initial_guess.is_empty() {
        return Err(NewtonError::InvalidConfig(
            "initial guess must contain at least one value".to_string(),
        ));
    }
    let base = &config.newton;
    let abs_tol = base.absolute_tolerance;
    let rel_tol = base.relative_tolerance;
    let damping_t = T::from_real(T::Real::from_f64(base.damping).ok_or_else(|| {
        NewtonError::InvalidConfig(
            "damping is not representable in the working precision".to_string(),
        )
    })?);

    if config.trust_region.is_some() {
        return Err(NewtonError::InvalidConfig(
            "trust-region globalisation is not supported by newton_raphson_continuation; \
             use newton_raphson_inexact"
                .to_string(),
        ));
    }

    // Ψtc parameters: explicit continuation, or the defaults kept in reserve
    // for automatic escalation.
    let pt_params = match config.continuation.params() {
        Some(p) => Some(p),
        None if config.auto_pseudo_transient => ContinuationStrategy::pseudo_transient().params(),
        None => None,
    };
    if let Some((dt0, growth, max_dt)) = pt_params
        && !(dt0 > 0.0 && growth >= 1.0 && max_dt >= dt0)
    {
        return Err(NewtonError::InvalidConfig(format!(
            "pseudo-transient parameters must satisfy dt0 > 0, growth >= 1, \
             max_dt >= dt0 (got dt0 = {dt0}, growth = {growth}, max_dt = {max_dt})"
        )));
    }

    // Same per-call callback guards as `newton_core`: a panicking user
    // closure yields an empty vector, which the length checks below report
    // as a typed `NumericalError` with the real iteration. For the matvec
    // this flows through `ShiftedJacobian::apply_into`, which already maps
    // malformed user output to a NaN sentinel for the GMRES layer.
    // Work counters, shared by the converged return sites below. The
    // continuation core is matrix-free, so `jacobian_evaluations` stays 0
    // and the linear work surfaces through `inner_iterations` instead.
    let residual_evaluations = Cell::new(0usize);
    let jacobian_evaluations = Cell::new(0usize);
    let inner_iterations = Cell::new(0usize);
    let residual_fn = |x: &[T]| -> Vec<T> {
        residual_evaluations.set(residual_evaluations.get() + 1);
        catch_unwind(AssertUnwindSafe(|| residual_fn(x))).unwrap_or_default()
    };
    let jacobian_apply_fn = |x: &[T], v: &[T]| -> Vec<T> {
        catch_unwind(AssertUnwindSafe(|| jacobian_apply_fn(x, v))).unwrap_or_default()
    };

    let mut x = initial_guess.to_vec();
    let mut initial_residual_norm = 0.0;
    // Active pseudo-timestep (None = plain Newton, not yet escalated).
    let mut dt: Option<f64> = config.continuation.params().map(|(dt0, _, _)| dt0);
    // Previous Eisenstat–Walker forcing term, for the safeguard bound.
    let mut eta_prev: Option<f64> = None;

    for iter in 0..base.max_iterations {
        validate_newton_vector(&x, initial_guess.len(), iter, "iterate")?;
        let residual = residual_fn(&x);
        validate_newton_vector(&residual, x.len(), iter, "residual")?;
        let residual_norm = real_to_f64::<T>(T::vec_norm(&residual), "residual norm", iter)?;
        if !residual_norm.is_finite() {
            return Err(NewtonError::NumericalError {
                iteration: iter,
                stage: "residual norm",
                reason: "norm overflowed to NaN or infinity".to_string(),
            });
        }

        if iter == 0 {
            initial_residual_norm = residual_norm;
            if base.verbosity >= 1 {
                log::info!("Newton iter 0: |R| = {residual_norm:.6e}");
            }
        } else if base.verbosity >= 1 {
            let relative = if initial_residual_norm > 0.0 {
                residual_norm / initial_residual_norm
            } else {
                0.0
            };
            log::info!("Newton iter {iter}: |R| = {residual_norm:.6e} (rel = {relative:.6e})");
        }

        // Absolute convergence check (`<=` so an exact initial guess
        // terminates with `iterations: 0` even at `abs_tol = 0`).
        if residual_norm <= abs_tol {
            return Ok(NewtonSolution {
                x,
                iterations: iter,
                residual: residual_norm,
                converged: true,
                status: SolverStatus::Converged,
                attainable_accuracy: AccuracyStatus::Attained,
                residual_evaluations: residual_evaluations.get(),
                jacobian_evaluations: jacobian_evaluations.get(),
                inner_iterations: inner_iterations.get(),
            });
        }

        // Relative convergence check
        if iter > 0 && initial_residual_norm > 0.0 {
            let relative = residual_norm / initial_residual_norm;
            if relative < rel_tol {
                return Ok(NewtonSolution {
                    x,
                    iterations: iter,
                    residual: residual_norm,
                    converged: true,
                    status: SolverStatus::Converged,
                    attainable_accuracy: AccuracyStatus::Attained,
                    residual_evaluations: residual_evaluations.get(),
                    jacobian_evaluations: jacobian_evaluations.get(),
                    inner_iterations: inner_iterations.get(),
                });
            }
        }

        // Divergence guard: a 1e6x blowup over the initial residual is
        // unscaled and can trip on legitimate non-monotone (early-Psi-tc)
        // transients, while slow drifts never trip it. Kept deliberately
        // loose so globalised runs are not aborted spuriously; tighten via
        // `max_iterations`/tolerances for strict runs.
        // Check divergence
        if iter > 0 && residual_norm > 1e6 * initial_residual_norm {
            return Err(NewtonError::Diverged(iter, residual_norm));
        }

        // Eisenstat–Walker forcing term (as in `newton_core`): η_k becomes
        // the GMRES relative tolerance for this Newton step.
        let eta = match config.forcing {
            ForcingTerm::None => 0.0,
            ForcingTerm::EisenstatWalker { eta_max, q } => {
                let eta = eisenstat_walker_eta(residual_norm, eta_max, q, eta_prev);
                eta_prev = Some(eta);
                eta
            }
        };
        // Inner tolerance, floored well above the machine epsilon of the
        // working precision so GMRES can actually terminate.
        let eps = real_to_f64::<T>(T::Real::epsilon(), "machine epsilon", iter)?;
        let forcing_tol = if eta > 0.0 { eta } else { 1e-8 };
        let inner_tol = forcing_tol.max(100.0 * eps);

        // Step computation, with re-solves at shrinking Δt when the line
        // search rejects the step.
        let mut step_retries = 0usize;
        let (new_x, new_norm) = loop {
            let inv_dt = dt.map(|d| 1.0 / d).unwrap_or(0.0);
            let op = ShiftedJacobian {
                point: &x,
                shift: T::from_real(real_from_f64::<T>(inv_dt, "continuation shift", iter)?),
                jacobian_apply_fn: &jacobian_apply_fn,
            };
            let (inner_max_iter, inner_restart) = match config.continuation_gmres_budget {
                Some((max_iter, restart)) => (max_iter, restart.min(x.len().max(1))),
                None => (20, x.len().min(30)),
            };
            let gmres_config = GmresConfig {
                max_iterations: inner_max_iter,
                restart: inner_restart,
                tolerance: real_from_f64::<T>(inner_tol, "inner GMRES tolerance", iter)?,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            };

            // Solve (shift·I + J(x)) δ = −F(x).
            let mut neg_residual = residual.clone();
            sonium_solvers_core::vec_ops::vec_mapv_inplace(&mut neg_residual, |v| -v);
            let sol = gmres(&op, &neg_residual, &gmres_config);
            if sol.status == SolverStatus::NumericalError {
                return Err(NewtonError::NumericalError {
                    iteration: iter,
                    stage: "Jacobian solve",
                    reason: "GMRES detected non-finite numerical data".to_string(),
                });
            }
            if !sol.converged {
                return Err(NewtonError::LinearSolveFailed {
                    iteration: iter,
                    message: format!(
                        "GMRES on the shifted Jacobian did not converge \
                         (shift = {inv_dt:.2e}, relative residual = {:.2e})",
                        sol.residual.to_f64().unwrap_or(f64::NAN)
                    ),
                });
            }
            inner_iterations.set(inner_iterations.get() + sol.iterations);
            let delta = sol.x;
            validate_newton_vector(&delta, x.len(), iter, "Jacobian step")?;

            match base.line_search {
                LineSearch::None => {
                    let mut x_new = x.clone();
                    T::vec_axpy(damping_t, &delta, &mut x_new);
                    let trial = residual_fn(&x_new);
                    validate_newton_vector(&trial, x.len(), iter, "trial residual")?;
                    let trial_norm =
                        real_to_f64::<T>(T::vec_norm(&trial), "trial residual norm", iter)?;
                    if !trial_norm.is_finite() {
                        return Err(NewtonError::NumericalError {
                            iteration: iter,
                            stage: "continuation trial norm",
                            reason: "norm overflowed to NaN or infinity".to_string(),
                        });
                    }
                    break (x_new, trial_norm);
                }
                LineSearch::Backtracking { c1, max_halvings } => {
                    let mut alpha = 1.0;
                    let mut x_trial = x.clone();
                    let mut trial_norm = 0.0;
                    let mut accepted = false;
                    let mut halvings_used = 0;

                    for ls_iter in 0..max_halvings {
                        x_trial.copy_from_slice(&x);
                        let alpha_t =
                            T::from_real(real_from_f64::<T>(alpha, "line-search step", iter)?);
                        T::vec_axpy(alpha_t, &delta, &mut x_trial);

                        let trial_residual = residual_fn(&x_trial);
                        trial_norm = real_to_f64::<T>(
                            T::vec_norm(&trial_residual),
                            "trial residual norm",
                            iter,
                        )?;

                        // Armijo sufficient decrease: ‖F(x_new)‖ ≤ (1 - c₁·α) ‖F(x_old)‖
                        if trial_norm <= (1.0 - c1 * alpha) * residual_norm {
                            if base.verbosity >= 2 {
                                log::info!("  Line search: α = {alpha:.4} (iter {ls_iter})");
                            }
                            accepted = true;
                            halvings_used = ls_iter;
                            break;
                        }
                        alpha *= 0.5;
                    }
                    if accepted {
                        // A heavily damped accept signals a poor local Newton
                        // model: engage Ψtc now, before marginal Armijo steps
                        // random-walk the iterate out of the region where the
                        // Jacobian carries useful information.
                        if dt.is_none()
                            && config.auto_pseudo_transient
                            && halvings_used >= ESCALATION_HALVING_THRESHOLD
                        {
                            let (dt0, _, max_dt) =
                                pt_params.expect("escalation requires Ψtc parameters");
                            let delta_norm =
                                real_to_f64::<T>(T::vec_norm(&delta), "Jacobian step norm", iter)?;
                            let dt_esc = escalation_dt(delta_norm, residual_norm, dt0, max_dt);
                            if base.verbosity >= 1 {
                                log::info!(
                                    "  Heavily damped step ({halvings_used} halvings): escalating to pseudo-transient continuation (Δt = {dt_esc:.2e})"
                                );
                            }
                            dt = Some(dt_esc);
                        }
                        break (x_trial, trial_norm);
                    }

                    // Rejected: shrink Δt (or escalate to Ψtc) and re-solve
                    // at the unchanged iterate — the larger shift yields a
                    // safer, more descent-like direction. This replaces the
                    // abort/legacy behaviour while a retry budget remains.
                    let can_retry = step_retries < MAX_STEP_RETRIES
                        && (dt.is_some() || config.auto_pseudo_transient);
                    if can_retry {
                        let (dt0, _, max_dt) = pt_params.expect("retry requires Ψtc parameters");
                        step_retries += 1;
                        dt = Some(match dt {
                            Some(d) => (d * 0.5).max(dt0),
                            None => {
                                let delta_norm = real_to_f64::<T>(
                                    T::vec_norm(&delta),
                                    "Jacobian step norm",
                                    iter,
                                )?;
                                let dt_esc = escalation_dt(delta_norm, residual_norm, dt0, max_dt);
                                if base.verbosity >= 1 {
                                    log::info!(
                                        "  Line search failed: escalating to pseudo-transient continuation (Δt = {dt_esc:.2e})"
                                    );
                                }
                                dt_esc
                            }
                        });
                        if base.verbosity >= 2 {
                            log::info!(
                                "  Ψtc: step rejected, Δt reduced to {:.3e} (retry {step_retries})",
                                dt.unwrap()
                            );
                        }
                        continue;
                    }
                    match config.line_search_failure {
                        LineSearchFailure::Fail => {
                            return Err(NewtonError::LineSearchFailed {
                                iteration: iter,
                                halvings: max_halvings,
                                residual: residual_norm,
                            });
                        }
                        LineSearchFailure::AcceptSmallestStep => {
                            // Legacy: keep the last trial point (smallest step)
                            break (x_trial, trial_norm);
                        }
                    }
                }
            }
        };

        // SER pseudo-timestep update after an accepted step: Δt grows with
        // the residual contraction, approaching pure Newton in the basin.
        if let (Some(d), Some((dt0, growth, max_dt))) = (dt, pt_params) {
            let new_dt =
                pseudo_transient_dt_update(d, residual_norm, new_norm, dt0, growth, max_dt);
            if base.verbosity >= 2 {
                log::info!(
                    "  Ψtc: Δt {d:.3e} -> {new_dt:.3e} (‖F‖ {residual_norm:.3e} -> {new_norm:.3e})"
                );
            }
            dt = Some(new_dt);
        }
        x = new_x;
    }

    // Final residual check
    let residual = residual_fn(&x);
    validate_newton_vector(&residual, x.len(), base.max_iterations, "final residual")?;
    let residual_norm = real_to_f64::<T>(
        T::vec_norm(&residual),
        "final residual norm",
        base.max_iterations,
    )?;

    Err(NewtonError::NotConverged(
        base.max_iterations,
        residual_norm,
    ))
}

/// Shared Newton iteration core behind [`newton_raphson`] and
/// [`newton_raphson_inexact`].
fn newton_core<T, R, J>(
    initial_guess: &[T],
    residual_fn: R,
    jacobian_solve_fn: J,
    config: &InexactNewtonConfig,
) -> Result<NewtonSolution<T>, NewtonError>
where
    T: ComplexField,
    T::Real: FromPrimitive,
    R: Fn(&[T]) -> Vec<T>,
    J: Fn(&[T], &[T], f64) -> Result<Vec<T>, String>,
{
    validate_newton_config(config)?;
    if initial_guess.is_empty() {
        return Err(NewtonError::InvalidConfig(
            "initial guess must contain at least one value".to_string(),
        ));
    }
    let base = &config.newton;
    let abs_tol = base.absolute_tolerance;
    let rel_tol = base.relative_tolerance;
    let damping_t = T::from_real(T::Real::from_f64(base.damping).ok_or_else(|| {
        NewtonError::InvalidConfig(
            "damping is not representable in the working precision".to_string(),
        )
    })?);

    // The Jacobian-solve entry points cannot push a diagonal shift through a
    // black-box linear solve, so `continuation` is ignored here (only
    // `newton_raphson_continuation` honours it). Say so loudly instead of
    // silently globalising differently than the caller asked.
    if config.continuation != ContinuationStrategy::None {
        log::warn!(
            "newton_core: `continuation` is set but ignored by the Jacobian-solve entry points; \
             use `newton_raphson_continuation` for pseudo-transient continuation"
        );
    }

    // Guard user callbacks per call: a panicking callback yields an empty
    // vector, which the length checks below report as a typed
    // `NumericalError` carrying the REAL iteration — instead of unwinding to
    // the entry-point wrapper, which can only report iteration 0. The outer
    // `catch_unwind` at the entry points remains as a backstop for panics
    // outside user callbacks.
    // Work counters, shared by the converged return sites below. The
    // Jacobian-solve closure is a black box, so its inner iteration count
    // is not visible and `inner_iterations` stays 0.
    let residual_evaluations = Cell::new(0usize);
    let jacobian_evaluations = Cell::new(0usize);
    let inner_iterations = Cell::new(0usize);
    let residual_fn = |x: &[T]| -> Vec<T> {
        residual_evaluations.set(residual_evaluations.get() + 1);
        catch_unwind(AssertUnwindSafe(|| residual_fn(x))).unwrap_or_default()
    };
    let jacobian_solve_fn = |x: &[T], rhs: &[T], eta: f64| -> Result<Vec<T>, String> {
        jacobian_evaluations.set(jacobian_evaluations.get() + 1);
        match catch_unwind(AssertUnwindSafe(|| jacobian_solve_fn(x, rhs, eta))) {
            Ok(result) => result,
            Err(_) => Ok(Vec::new()),
        }
    };

    let mut x = initial_guess.to_vec();
    let mut initial_residual_norm = 0.0;
    let mut trust_radius = config
        .trust_region
        .map(|tr| tr.initial_radius)
        .unwrap_or(0.0);
    // Previous Eisenstat–Walker forcing term, for the safeguard bound.
    let mut eta_prev: Option<f64> = None;

    for iter in 0..base.max_iterations {
        validate_newton_vector(&x, initial_guess.len(), iter, "iterate")?;
        let residual = residual_fn(&x);
        validate_newton_vector(&residual, x.len(), iter, "residual")?;
        let residual_norm = real_to_f64::<T>(T::vec_norm(&residual), "residual norm", iter)?;
        if !residual_norm.is_finite() {
            return Err(NewtonError::NumericalError {
                iteration: iter,
                stage: "residual norm",
                reason: "norm overflowed to NaN or infinity".to_string(),
            });
        }

        if iter == 0 {
            initial_residual_norm = residual_norm;
            if base.verbosity >= 1 {
                log::info!("Newton iter 0: |R| = {residual_norm:.6e}");
            }
        } else if base.verbosity >= 1 {
            let relative = if initial_residual_norm > 0.0 {
                residual_norm / initial_residual_norm
            } else {
                0.0
            };
            log::info!("Newton iter {iter}: |R| = {residual_norm:.6e} (rel = {relative:.6e})");
        }

        // Absolute convergence check (`<=` so an exact initial guess
        // terminates with `iterations: 0` even at `abs_tol = 0`).
        if residual_norm <= abs_tol {
            return Ok(NewtonSolution {
                x,
                iterations: iter,
                residual: residual_norm,
                converged: true,
                status: SolverStatus::Converged,
                attainable_accuracy: AccuracyStatus::Attained,
                residual_evaluations: residual_evaluations.get(),
                jacobian_evaluations: jacobian_evaluations.get(),
                inner_iterations: inner_iterations.get(),
            });
        }

        // Relative convergence check
        if iter > 0 && initial_residual_norm > 0.0 {
            let relative = residual_norm / initial_residual_norm;
            if relative < rel_tol {
                return Ok(NewtonSolution {
                    x,
                    iterations: iter,
                    residual: residual_norm,
                    converged: true,
                    status: SolverStatus::Converged,
                    attainable_accuracy: AccuracyStatus::Attained,
                    residual_evaluations: residual_evaluations.get(),
                    jacobian_evaluations: jacobian_evaluations.get(),
                    inner_iterations: inner_iterations.get(),
                });
            }
        }

        // Divergence guard: a 1e6x blowup over the initial residual is
        // unscaled and can trip on legitimate non-monotone (early-Psi-tc)
        // transients, while slow drifts never trip it. Kept deliberately
        // loose so globalised runs are not aborted spuriously; tighten via
        // `max_iterations`/tolerances for strict runs.
        // Check divergence
        if iter > 0 && residual_norm > 1e6 * initial_residual_norm {
            return Err(NewtonError::Diverged(iter, residual_norm));
        }

        // Eisenstat–Walker forcing term: η_k = min(η_max, ‖F_k‖^q) with the
        // published safeguard against η collapse (see `eisenstat_walker_eta`).
        let eta = match config.forcing {
            ForcingTerm::None => 0.0,
            ForcingTerm::EisenstatWalker { eta_max, q } => {
                let eta = eisenstat_walker_eta(residual_norm, eta_max, q, eta_prev);
                eta_prev = Some(eta);
                eta
            }
        };

        // Solve J(x) * delta = -F(x)
        // Negate in-place to avoid allocating a temporary (zero allocation)
        let mut neg_residual = residual;
        sonium_solvers_core::vec_ops::vec_mapv_inplace(&mut neg_residual, |v| -v);
        let delta = jacobian_solve_fn(&x, &neg_residual, eta).map_err(|msg| {
            NewtonError::LinearSolveFailed {
                iteration: iter,
                message: msg,
            }
        })?;
        validate_newton_vector(&delta, x.len(), iter, "Jacobian step")?;

        // Update with trust region, line search, or fixed damping
        if let Some(tr) = &config.trust_region {
            // Clip the Newton step to the trust radius: d = τ δ.
            let delta_norm = real_to_f64::<T>(T::vec_norm(&delta), "Jacobian step norm", iter)?;
            if !delta_norm.is_finite() {
                return Err(NewtonError::NumericalError {
                    iteration: iter,
                    stage: "Jacobian step norm",
                    reason: "norm overflowed to NaN or infinity".to_string(),
                });
            }
            if delta_norm <= 0.0 {
                return Err(NewtonError::LinearSolveFailed {
                    iteration: iter,
                    message: "zero Newton step with nonzero residual".to_string(),
                });
            }
            // Acceptance/rejection retry loop. On rejection only the trust
            // radius shrinks and the SAME direction δ is re-clipped — the
            // residual and the Jacobian solve are not recomputed at the
            // unchanged iterate (standard trust-region retry). This keeps a
            // rejected step at one extra residual evaluation instead of a
            // full (residual + J-solve) cycle.
            // The trial buffer is reused across retries (`clone_from` keeps
            // the allocation); values are identical to a fresh clone per
            // retry.
            let mut x_trial: Vec<T> = Vec::new();
            loop {
                let tau = if delta_norm > trust_radius {
                    trust_radius / delta_norm
                } else {
                    1.0
                };
                // Predicted reduction of ‖F‖² from the Gauss–Newton model
                // J d ≈ −τ F: pred = ‖F‖² − ‖(1−τ) F‖² = (2τ − τ²) ‖F‖².
                let pred = (2.0 * tau - tau * tau) * residual_norm * residual_norm;

                let tau_t = T::from_real(real_from_f64::<T>(tau, "trust-region scaling", iter)?);
                x_trial.clone_from(&x);
                T::vec_axpy(tau_t, &delta, &mut x_trial);
                let trial_residual = residual_fn(&x_trial);
                validate_newton_vector(&trial_residual, x.len(), iter, "trust-region residual")?;
                let trial_norm =
                    real_to_f64::<T>(T::vec_norm(&trial_residual), "trial residual norm", iter)?;
                if !trial_norm.is_finite() {
                    return Err(NewtonError::NumericalError {
                        iteration: iter,
                        stage: "trust-region residual norm",
                        reason: "norm overflowed to NaN or infinity".to_string(),
                    });
                }
                let ared = residual_norm * residual_norm - trial_norm * trial_norm;
                let rho = if pred > 0.0 { ared / pred } else { 0.0 };

                if rho >= tr.accept_ratio {
                    // Accept the step (move the buffer; no copy).
                    x = std::mem::take(&mut x_trial);
                    // Grow the radius only when the step was clipped to the
                    // boundary (`tau < 1`) and the model is accurate: growing
                    // on interior steps lets inexact solves inflate the
                    // radius on a poor model (classical update).
                    if rho > 0.75 && tau < 1.0 {
                        trust_radius = (2.0 * trust_radius).min(tr.max_radius);
                    }
                    if base.verbosity >= 2 {
                        log::info!(
                            "  Trust region: accepted τ = {tau:.4}, ρ = {rho:.4}, radius = {trust_radius:.4e}"
                        );
                    }
                    break;
                }
                // Reject: shrink the radius and retry with the same iterate
                // and the same direction.
                trust_radius *= 0.25;
                if base.verbosity >= 2 {
                    log::info!(
                        "  Trust region: rejected ρ = {rho:.4}, new radius = {trust_radius:.4e}"
                    );
                }
                if trust_radius < tr.min_radius {
                    return Err(NewtonError::TrustRegionCollapsed {
                        iteration: iter,
                        radius: trust_radius,
                        residual: residual_norm,
                    });
                }
            }
        } else {
            match base.line_search {
                LineSearch::None => {
                    T::vec_axpy(damping_t, &delta, &mut x);
                }
                LineSearch::Backtracking { c1, max_halvings } => {
                    let mut alpha = 1.0;
                    let x_old = x.clone();
                    let norm_old = residual_norm;
                    let mut accepted = false;

                    for ls_iter in 0..max_halvings {
                        // Trial: x = x_old + alpha * delta
                        x.copy_from_slice(&x_old);
                        let alpha_t =
                            T::from_real(real_from_f64::<T>(alpha, "line-search step", iter)?);
                        T::vec_axpy(alpha_t, &delta, &mut x);

                        let trial_residual = residual_fn(&x);
                        validate_newton_vector(
                            &trial_residual,
                            x.len(),
                            iter,
                            "line-search residual",
                        )?;
                        let trial_norm = real_to_f64::<T>(
                            T::vec_norm(&trial_residual),
                            "trial residual norm",
                            iter,
                        )?;
                        if !trial_norm.is_finite() {
                            return Err(NewtonError::NumericalError {
                                iteration: iter,
                                stage: "line-search residual norm",
                                reason: "norm overflowed to NaN or infinity".to_string(),
                            });
                        }

                        // Armijo sufficient decrease: ‖F(x_new)‖ ≤ (1 - c₁·α) ‖F(x_old)‖
                        if trial_norm <= (1.0 - c1 * alpha) * norm_old {
                            if base.verbosity >= 2 {
                                log::info!("  Line search: α = {alpha:.4} (iter {ls_iter})");
                            }
                            accepted = true;
                            break;
                        }
                        alpha *= 0.5;
                    }
                    if !accepted {
                        match config.line_search_failure {
                            LineSearchFailure::Fail => {
                                // Restore the pre-step iterate and report.
                                x.copy_from_slice(&x_old);
                                return Err(NewtonError::LineSearchFailed {
                                    iteration: iter,
                                    halvings: max_halvings,
                                    residual: residual_norm,
                                });
                            }
                            LineSearchFailure::AcceptSmallestStep => {
                                // Legacy: keep the last trial point (smallest step)
                            }
                        }
                    }
                }
            }
        }
    }

    // Final residual check
    let residual = residual_fn(&x);
    validate_newton_vector(&residual, x.len(), base.max_iterations, "final residual")?;
    let residual_norm = real_to_f64::<T>(
        T::vec_norm(&residual),
        "final residual norm",
        base.max_iterations,
    )?;

    Err(NewtonError::NotConverged(
        base.max_iterations,
        residual_norm,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_jacobian_rejects_raw_and_checked_adjoint_requests() {
        let nonsymmetric = |_: &[f64], x: &[f64]| vec![x[0] + 2.0 * x[1], x[1]];
        let op = ShiftedJacobian {
            point: &[0.0, 0.0],
            shift: 1.0,
            jacobian_apply_fn: &nonsymmetric,
        };
        assert_eq!(op.apply(&[1.0, 2.0]), vec![6.0, 4.0]);
        assert!(op.try_apply_transpose(&[1.0, 2.0]).is_err());
        assert!(op.try_apply_hermitian(&[1.0, 2.0]).is_err());
        assert!(std::panic::catch_unwind(|| op.apply_transpose(&[1.0, 2.0])).is_err());
        assert!(std::panic::catch_unwind(|| op.apply_hermitian(&[1.0, 2.0])).is_err());
    }
    use std::cell::Cell;

    #[test]
    fn test_newton_scalar_equation() {
        // Solve x^2 - 4 = 0, starting from x = 1
        let x0 = sonium_solvers_core::array![1.0_f64];
        let config = NewtonConfig {
            absolute_tolerance: 1e-12,
            ..Default::default()
        };

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] - 4.0],
            |x, rhs| Ok(sonium_solvers_core::array![rhs[0] / (2.0 * x[0])]),
            &config,
        )
        .unwrap();

        assert!(result.converged);
        assert_eq!(result.status, SolverStatus::Converged);
        assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
        assert!((result.x[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn newton_errors_expose_canonical_status_and_accuracy() {
        let not_converged = NewtonError::NotConverged(4, 1.0);
        assert_eq!(not_converged.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(
            not_converged.attainable_accuracy(),
            AccuracyStatus::NotAttained
        );

        let line_search = NewtonError::LineSearchFailed {
            iteration: 2,
            halvings: 4,
            residual: 1.0,
        };
        assert_eq!(line_search.status(), SolverStatus::Stagnated);
        assert_eq!(
            line_search.attainable_accuracy(),
            AccuracyStatus::NotAttained
        );

        let invalid = NewtonError::InvalidConfig("bad tolerance".into());
        assert_eq!(invalid.status(), SolverStatus::NumericalError);
        assert_eq!(invalid.attainable_accuracy(), AccuracyStatus::NotChecked);
    }

    #[test]
    fn newton_rejects_non_finite_initial_guess_before_callback() {
        let result = newton_raphson(
            &[f64::NAN],
            |_| panic!("residual callback must not run"),
            |_, _| Ok(vec![0.0]),
            &NewtonConfig::default(),
        );

        assert!(matches!(
            result,
            Err(NewtonError::NumericalError {
                stage: "iterate",
                ..
            })
        ));
    }

    #[test]
    fn newton_rejects_empty_initial_guess_and_zero_iteration_budget() {
        let empty = newton_raphson(
            &[] as &[f64],
            |_| panic!("empty systems must be rejected before callbacks"),
            |_, _| Ok(Vec::new()),
            &NewtonConfig::default(),
        );
        assert!(matches!(empty, Err(NewtonError::InvalidConfig(_))));

        let mut config = NewtonConfig::default();
        config.max_iterations = 0;
        let zero_budget = newton_raphson(&[0.0_f64], |_| vec![0.0], |_, _| Ok(vec![0.0]), &config);
        assert!(matches!(zero_budget, Err(NewtonError::InvalidConfig(_))));
    }

    #[test]
    fn inexact_newton_rejects_invalid_forcing_and_inner_budgets_before_callbacks() {
        let bad_forcing = [
            (
                "eta_max = 0",
                ForcingTerm::EisenstatWalker {
                    eta_max: 0.0,
                    q: 2.0,
                },
            ),
            (
                "eta_max > 1",
                ForcingTerm::EisenstatWalker {
                    eta_max: 2.0,
                    q: 2.0,
                },
            ),
            (
                "eta_max NaN",
                ForcingTerm::EisenstatWalker {
                    eta_max: f64::NAN,
                    q: 2.0,
                },
            ),
            (
                "q = 0",
                ForcingTerm::EisenstatWalker {
                    eta_max: 0.5,
                    q: 0.0,
                },
            ),
            (
                "q > 2",
                ForcingTerm::EisenstatWalker {
                    eta_max: 0.5,
                    q: 3.0,
                },
            ),
            (
                "q NaN",
                ForcingTerm::EisenstatWalker {
                    eta_max: 0.5,
                    q: f64::NAN,
                },
            ),
        ];
        for (label, forcing) in bad_forcing {
            let config = InexactNewtonConfig {
                forcing,
                ..Default::default()
            };
            let result = newton_raphson_inexact(
                &[0.0_f64],
                |_| panic!("residual callback must not run"),
                |_, _, _| panic!("inner solve callback must not run"),
                &config,
            );
            assert!(
                matches!(result, Err(NewtonError::InvalidConfig(_))),
                "{label} must be rejected before callbacks run",
            );
        }

        for budget in [Some((0, 10)), Some((10, 0))] {
            let config = InexactNewtonConfig {
                continuation_gmres_budget: budget,
                ..Default::default()
            };
            let result = newton_raphson_inexact(
                &[0.0_f64],
                |_| panic!("residual callback must not run"),
                |_, _, _| panic!("inner solve callback must not run"),
                &config,
            );
            assert!(
                matches!(result, Err(NewtonError::InvalidConfig(_))),
                "zero continuation GMRES budget {budget:?} must be rejected before callbacks run",
            );
        }
    }

    #[test]
    fn newton_rejects_non_finite_or_wrong_length_residuals() {
        let non_finite = newton_raphson(
            &[0.0_f64],
            |_| vec![f64::INFINITY],
            |_, _| Ok(vec![0.0]),
            &NewtonConfig::default(),
        );
        assert!(matches!(
            non_finite,
            Err(NewtonError::NumericalError {
                stage: "residual",
                ..
            })
        ));

        let wrong_length = newton_raphson(
            &[0.0_f64],
            |_| Vec::new(),
            |_, _| Ok(vec![0.0]),
            &NewtonConfig::default(),
        );
        assert!(matches!(
            wrong_length,
            Err(NewtonError::NumericalError {
                stage: "residual",
                ..
            })
        ));
    }

    #[test]
    fn newton_rejects_malformed_jacobian_steps() {
        let result = newton_raphson(
            &[0.0_f64],
            |_| vec![1.0],
            |_, _| Ok(Vec::new()),
            &NewtonConfig::default(),
        );

        assert!(matches!(
            result,
            Err(NewtonError::NumericalError {
                stage: "Jacobian step",
                ..
            })
        ));
    }

    #[test]
    fn continuation_rejects_non_finite_residual_before_gmres() {
        let result = newton_raphson_continuation(
            &[0.0_f64],
            |_| vec![f64::NAN],
            |_, v| v.to_vec(),
            &InexactNewtonConfig::default(),
        );

        assert!(matches!(
            result,
            Err(NewtonError::NumericalError {
                stage: "residual",
                ..
            })
        ));
    }

    #[test]
    fn test_newton_f32() {
        // Same problem in f32
        let x0 = sonium_solvers_core::array![1.0_f32];
        let config = NewtonConfig {
            absolute_tolerance: 1e-6,
            ..Default::default()
        };

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] - 4.0_f32],
            |x, rhs| Ok(sonium_solvers_core::array![rhs[0] / (2.0_f32 * x[0])]),
            &config,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0_f32).abs() < 1e-5);
    }

    #[test]
    fn test_newton_2d_system() {
        let x0 = sonium_solvers_core::array![0.5, 0.5];
        let config = NewtonConfig {
            absolute_tolerance: 1e-12,
            ..Default::default()
        };

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]],
            |x, rhs| {
                let det = -2.0 * x[0] - 2.0 * x[1];
                Ok(sonium_solvers_core::array![
                    (-rhs[0] - 2.0 * x[1] * rhs[1]) / det,
                    (-rhs[0] + 2.0 * x[0] * rhs[1]) / det,
                ])
            },
            &config,
        )
        .unwrap();

        assert!(result.converged);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((result.x[0] - expected).abs() < 1e-10);
        assert!((result.x[1] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_newton_with_damping() {
        let x0 = sonium_solvers_core::array![0.1_f64];
        let config = NewtonConfig {
            absolute_tolerance: 1e-10,
            damping: 0.5,
            ..Default::default()
        };

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] - 4.0],
            |x, rhs| Ok(sonium_solvers_core::array![rhs[0] / (2.0 * x[0])]),
            &config,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_newton_already_converged() {
        let x0 = sonium_solvers_core::array![2.0_f64];
        let config = NewtonConfig {
            absolute_tolerance: 1e-10,
            ..Default::default()
        };

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] - 4.0],
            |x, rhs| Ok(sonium_solvers_core::array![rhs[0] / (2.0 * x[0])]),
            &config,
        )
        .unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 0, "No Newton steps should be needed");
        assert!((result.x[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_newton_zero_initial_residual() {
        let x0 = sonium_solvers_core::array![0.0_f64, 0.0];
        let config = NewtonConfig::default();

        let result = newton_raphson(
            &x0,
            |_x| sonium_solvers_core::array![0.0, 0.0],
            |_x, _rhs| Ok(sonium_solvers_core::array![0.0, 0.0]),
            &config,
        )
        .unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_newton_linear_solve_failure() {
        let x0 = sonium_solvers_core::array![1.0_f64];
        let config = NewtonConfig::default();

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] - 5.0],
            |_x, _rhs| Err("Singular matrix".to_string()),
            &config,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, NewtonError::LinearSolveFailed { .. }));
    }

    #[test]
    fn test_newton_backtracking_line_search() {
        // Solve x^3 - 1 = 0 starting far away — pure Newton might overshoot
        let x0 = sonium_solvers_core::array![10.0_f64];
        let config = NewtonConfig {
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-12,
            line_search: LineSearch::backtracking(),
            ..Default::default()
        };

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] * x[0] - 1.0], // F(x) = x³ - 1
            |x, rhs| Ok(sonium_solvers_core::array![rhs[0] / (3.0 * x[0] * x[0])]), // J⁻¹ = 1/(3x²)
            &config,
        )
        .unwrap();

        assert!(result.converged, "Should converge with line search");
        assert!((result.x[0] - 1.0).abs() < 1e-8, "x = {}", result.x[0]);
    }

    #[test]
    fn test_newton_line_search_from_bad_initial() {
        // Solve x^2 - 4 = 0 from x = 0.01 — J(0.01) = 0.02, Newton step = -198
        // Without line search, x jumps to -198 (wrong root). With LS, stays positive.
        let x0 = sonium_solvers_core::array![0.01_f64];
        let config = NewtonConfig {
            absolute_tolerance: 1e-10,
            line_search: LineSearch::backtracking(),
            ..Default::default()
        };

        let result = newton_raphson(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] - 4.0],
            |x, rhs| {
                let j = 2.0 * x[0];
                if j.abs() < 1e-15 {
                    return Err("Singular".to_string());
                }
                Ok(sonium_solvers_core::array![rhs[0] / j])
            },
            &config,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.x[0].abs() - 2.0).abs() < 1e-8);
    }

    // ------------------------------------------------------------------
    // Inexact Newton: forcing term, line-search failure, trust region
    // ------------------------------------------------------------------

    /// 1D nonlinear Poisson residual: −u″ + u³ = 1 on (0,1), u(0) = u(1) = 0,
    /// discretised with centered differences. Returns the residual and the
    /// tridiagonal Jacobian diagonal/off-diagonal coefficients.
    fn poisson_residual(u: &[f64]) -> Vec<f64> {
        let n = u.len();
        let h = 1.0 / (n as f64 + 1.0);
        let inv_h2 = 1.0 / (h * h);
        let mut f = vec![0.0; n];
        for i in 0..n {
            let u_im = if i == 0 { 0.0 } else { u[i - 1] };
            let u_ip = if i + 1 == n { 0.0 } else { u[i + 1] };
            f[i] = (2.0 * u[i] - u_im - u_ip) * inv_h2 + u[i].powi(3) - 1.0;
        }
        f
    }

    /// Solve the tridiagonal Jacobian system of the 1D nonlinear Poisson
    /// problem with Jacobi iteration until `‖J δ − rhs‖ ≤ tol_rel * ‖rhs‖`,
    /// counting sweeps in `inner_counter`.
    fn poisson_jacobi_solve(
        u: &[f64],
        rhs: &[f64],
        tol_rel: f64,
        inner_counter: &Cell<usize>,
    ) -> Result<Vec<f64>, String> {
        let n = u.len();
        let h = 1.0 / (n as f64 + 1.0);
        let inv_h2 = 1.0 / (h * h);
        let diag: Vec<f64> = u.iter().map(|&ui| 2.0 * inv_h2 + 3.0 * ui * ui).collect();
        let off = -inv_h2;

        let rhs_norm = rhs.iter().map(|r| r * r).sum::<f64>().sqrt();
        if rhs_norm == 0.0 {
            return Ok(vec![0.0; n]);
        }

        let mut delta = vec![0.0; n];
        for _ in 0..200_000 {
            // One Jacobi sweep; compute residual on the fly.
            let mut res_norm_sqr = 0.0;
            let mut new_delta = vec![0.0; n];
            for i in 0..n {
                let d_im = if i == 0 { 0.0 } else { delta[i - 1] };
                let d_ip = if i + 1 == n { 0.0 } else { delta[i + 1] };
                let jd = diag[i] * delta[i] + off * d_im + off * d_ip;
                let r = rhs[i] - jd;
                res_norm_sqr += r * r;
                new_delta[i] = delta[i] + r / diag[i];
            }
            delta = new_delta;
            inner_counter.set(inner_counter.get() + 1);
            if res_norm_sqr.sqrt() <= tol_rel * rhs_norm {
                return Ok(delta);
            }
        }
        Err("inner Jacobi solve did not converge".to_string())
    }

    #[test]
    fn test_inexact_newton_ew_reduces_inner_iterations() {
        // Nonlinear Poisson: with a fixed tight inner tolerance the early
        // Newton steps are over-solved. The Eisenstat–Walker forcing term
        // should cut the total inner (Jacobi) iterations substantially.
        let n = 32;
        let x0 = vec![0.0_f64; n];

        // Fixed tight inner tolerance (legacy: eta = 0 → closure uses 1e-12).
        let fixed_counter = Cell::new(0usize);
        let fixed_cfg = InexactNewtonConfig {
            newton: NewtonConfig {
                absolute_tolerance: 1e-10,
                ..Default::default()
            },
            ..Default::default()
        };
        let fixed = newton_raphson_inexact(
            &x0,
            poisson_residual,
            |u, rhs, eta| {
                let tol = if eta > 0.0 { eta } else { 1e-12 };
                poisson_jacobi_solve(u, rhs, tol, &fixed_counter)
            },
            &fixed_cfg,
        )
        .unwrap();
        assert!(fixed.converged, "fixed-tolerance Newton should converge");

        // Eisenstat–Walker choice 2: eta_k = min(0.5, ‖F_k‖²).
        let ew_counter = Cell::new(0usize);
        let ew_cfg = InexactNewtonConfig {
            newton: NewtonConfig {
                absolute_tolerance: 1e-10,
                ..Default::default()
            },
            forcing: ForcingTerm::EisenstatWalker {
                eta_max: 0.5,
                q: 2.0,
            },
            ..Default::default()
        };
        let ew = newton_raphson_inexact(
            &x0,
            poisson_residual,
            |u, rhs, eta| {
                let tol = if eta > 0.0 { eta } else { 1e-12 };
                poisson_jacobi_solve(u, rhs, tol, &ew_counter)
            },
            &ew_cfg,
        )
        .unwrap();
        assert!(ew.converged, "EW Newton should converge");

        // Same solution.
        for (a, b) in fixed.x.iter().zip(ew.x.iter()) {
            assert!((a - b).abs() < 1e-6, "solutions differ: {a} vs {b}");
        }

        let fixed_inner = fixed_counter.get();
        let ew_inner = ew_counter.get();
        assert!(
            ew_inner < fixed_inner,
            "EW should reduce inner iterations: ew={ew_inner} fixed={fixed_inner}"
        );
        assert!(
            ew_inner * 3 <= fixed_inner * 2,
            "EW should cut inner iterations by at least a third: ew={ew_inner} fixed={fixed_inner}"
        );
    }

    #[test]
    fn test_inexact_newton_line_search_failure_reported() {
        // F(x) ≡ 1 regardless of x: the Armijo condition can never be met,
        // so every halving fails. The legacy path silently accepted the
        // smallest step; the inexact path must report LineSearchFailed and
        // leave the iterate untouched.
        let x0 = sonium_solvers_core::array![3.0_f64];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 10,
                line_search: LineSearch::backtracking(),
                ..Default::default()
            },
            ..Default::default()
        };

        let result = newton_raphson_inexact(
            &x0,
            |_x| sonium_solvers_core::array![1.0_f64],
            |_x, _rhs, _eta| Ok(sonium_solvers_core::array![1.0_f64]),
            &config,
        );

        match result {
            Err(NewtonError::LineSearchFailed {
                iteration,
                halvings,
                ..
            }) => {
                assert_eq!(iteration, 0);
                assert_eq!(halvings, 10);
            }
            other => panic!("expected LineSearchFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_inexact_newton_line_search_failure_legacy_flag() {
        // Same un-satisfiable line search, but the legacy policy restores the
        // old behaviour: accept the smallest step, keep iterating, and
        // finally report NotConverged (not LineSearchFailed).
        let x0 = sonium_solvers_core::array![3.0_f64];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 5,
                line_search: LineSearch::backtracking(),
                ..Default::default()
            },
            line_search_failure: LineSearchFailure::AcceptSmallestStep,
            ..Default::default()
        };

        let result = newton_raphson_inexact(
            &x0,
            |_x| sonium_solvers_core::array![1.0_f64],
            |_x, _rhs, _eta| Ok(sonium_solvers_core::array![1.0_f64]),
            &config,
        );

        assert!(
            matches!(result, Err(NewtonError::NotConverged(5, _))),
            "expected NotConverged with legacy policy, got {result:?}"
        );
    }

    #[test]
    fn test_inexact_newton_trust_region_where_plain_newton_diverges() {
        // F(x) = atan(x). The Newton step from |x| large is
        // δ = −atan(x)(1 + x²), which overshoots wildly: plain Newton
        // oscillates between huge ±x and never converges.
        let residual = |x: &[f64]| -> Vec<f64> { x.iter().map(|&xi| xi.atan()).collect() };
        let jac_solve = |x: &[f64], rhs: &[f64], _eta: f64| -> Result<Vec<f64>, String> {
            Ok(x.iter()
                .zip(rhs.iter())
                .map(|(&xi, &ri)| ri * (1.0 + xi * xi))
                .collect())
        };

        // Plain Newton (no globalisation) fails from x0 = [10, −20].
        let x0 = vec![10.0_f64, -20.0];
        let plain_cfg = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 50,
                ..Default::default()
            },
            ..Default::default()
        };
        let plain = newton_raphson_inexact(&x0, residual, jac_solve, &plain_cfg);
        assert!(
            plain.is_err(),
            "plain Newton should not converge from x0 = [10, -20]"
        );

        // Trust-region Newton converges.
        let tr_cfg = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 200,
                absolute_tolerance: 1e-12,
                relative_tolerance: 1e-15,
                ..Default::default()
            },
            trust_region: Some(TrustRegionConfig::default()),
            ..Default::default()
        };
        let tr = newton_raphson_inexact(&x0, residual, jac_solve, &tr_cfg).unwrap();
        assert!(tr.converged, "trust-region Newton should converge");
        assert!(tr.x[0].abs() < 1e-10, "x[0] = {}", tr.x[0]);
        assert!(tr.x[1].abs() < 1e-10, "x[1] = {}", tr.x[1]);
    }

    #[test]
    fn test_inexact_newton_trust_region_scalar() {
        // Sanity: trust region also solves the easy scalar problem.
        let x0 = sonium_solvers_core::array![1.0_f64];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                absolute_tolerance: 1e-12,
                ..Default::default()
            },
            trust_region: Some(TrustRegionConfig::default()),
            ..Default::default()
        };

        let result = newton_raphson_inexact(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] - 4.0],
            |x, rhs, _eta| Ok(sonium_solvers_core::array![rhs[0] / (2.0 * x[0])]),
            &config,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ew_safeguard_formula() {
        // Safeguard engaged: previous η was large (η_prev^q = 0.81 > 0.1) and
        // the residual collapsed (‖F‖^q = 1e-8). Without the safeguard η
        // would drop to 1e-8; the paper's bound keeps it at η_prev^φ.
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let eta = eisenstat_walker_eta(1e-4, 0.9, 2.0, Some(0.9));
        let expected = 0.9_f64.powf(phi);
        assert!(
            (eta - expected).abs() < 1e-15,
            "safeguard bound: got {eta}, expected {expected}"
        );

        // Safeguard not engaged: η_prev^q ≤ 0.1, η follows the raw choice 2.
        let eta = eisenstat_walker_eta(1e-4, 0.9, 2.0, Some(0.3));
        assert!(
            (eta - 1e-8).abs() < 1e-20,
            "no safeguard below threshold: got {eta}"
        );

        // Raw value above the bound: safeguard does not loosen η.
        let eta = eisenstat_walker_eta(0.9, 0.9, 2.0, Some(0.5));
        assert!(
            (eta - 0.81).abs() < 1e-15,
            "raw value kept when above bound: got {eta}"
        );

        // First step defaults η_prev to η_max: η = min(η_max, ‖F‖^q).
        let eta = eisenstat_walker_eta(10.0, 0.5, 2.0, None);
        assert!((eta - 0.5).abs() < 1e-15, "first step caps at η_max: {eta}");
    }

    #[test]
    fn test_ew_safeguard_prevents_eta_yo_yo() {
        // Drive the solver through a state where the residual collapses
        // after a loosely-solved step and check, via the η values handed to
        // the Jacobian closure, that the safeguard keeps η from yo-yoing.
        // F(x) = x³ − 1 from x0 slightly off the root: the first Newton step
        // lands near the root (residual collapses), and without the
        // safeguard the next η would drop by orders of magnitude.
        let etas = std::cell::RefCell::new(Vec::new());
        let x0 = sonium_solvers_core::array![1.05_f64];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                absolute_tolerance: 1e-14,
                ..Default::default()
            },
            forcing: ForcingTerm::EisenstatWalker {
                eta_max: 0.5,
                q: 2.0,
            },
            ..Default::default()
        };
        let result = newton_raphson_inexact(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] * x[0] - 1.0],
            |x, rhs, eta| {
                etas.borrow_mut().push(eta);
                Ok(sonium_solvers_core::array![rhs[0] / (3.0 * x[0] * x[0])])
            },
            &config,
        )
        .unwrap();
        assert!(result.converged);

        let etas = etas.borrow();
        assert!(etas.len() >= 2, "need at least two steps: {etas:?}");
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // The residual at x0 is already small (‖F₀‖² ≈ 0.025), so the
        // safeguard binds from the very first step: η₀ = η_max^φ instead of
        // the raw ‖F₀‖², and the cascade continues while η_{k-1}² > 0.1.
        let bound0 = 0.5_f64.powf(phi);
        assert!(
            (etas[0] - bound0).abs() < 1e-12,
            "first η should be the safeguard bound {bound0}, got {} (all: {etas:?})",
            etas[0]
        );
        let bound1 = bound0.powf(phi);
        assert!(
            (etas[1] - bound1).abs() < 1e-12,
            "second η should cascade to {bound1}, got {} (all: {etas:?})",
            etas[1]
        );
        // Invariant form: no step collapses below η_{k-1}^φ while the
        // safeguard threshold η_{k-1}^q > 0.1 holds.
        for w in etas.windows(2) {
            if w[0].powi(2) > 0.1 {
                assert!(
                    w[1] >= w[0].powf(phi) - 1e-15,
                    "η yo-yoed: {} -> {} (bound {})",
                    w[0],
                    w[1],
                    w[0].powf(phi)
                );
            }
        }
    }

    #[test]
    fn test_trust_region_rejection_does_not_resolve_jacobian() {
        // F(x) = atan(x) from a distant start: the Gauss–Newton model is poor
        // there, so the trust region rejects steps. A rejected step must only
        // shrink the radius and re-clip the SAME direction — the Jacobian
        // solve must never run twice at the same iterate.
        let solve_points = std::cell::RefCell::new(Vec::new());
        let residual = |x: &[f64]| -> Vec<f64> { x.iter().map(|&xi| xi.atan()).collect() };
        let jac_solve = |x: &[f64], rhs: &[f64], _eta: f64| -> Result<Vec<f64>, String> {
            solve_points.borrow_mut().push(x.to_vec());
            Ok(x.iter()
                .zip(rhs.iter())
                .map(|(&xi, &ri)| ri * (1.0 + xi * xi))
                .collect())
        };

        let x0 = vec![10.0_f64, -20.0];
        let tr_cfg = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 200,
                absolute_tolerance: 1e-12,
                relative_tolerance: 1e-15,
                ..Default::default()
            },
            trust_region: Some(TrustRegionConfig::default()),
            ..Default::default()
        };
        let tr = newton_raphson_inexact(&x0, residual, jac_solve, &tr_cfg).unwrap();
        assert!(tr.converged, "trust-region Newton should converge");

        // One Jacobian solve per Newton iteration, each at a fresh iterate:
        // no re-solve at an unchanged point after a rejection.
        let points = solve_points.borrow();
        assert_eq!(
            points.len(),
            tr.iterations,
            "exactly one J-solve per iteration (solves: {}, iters: {})",
            points.len(),
            tr.iterations
        );
        for w in points.windows(2) {
            assert_ne!(
                w[0], w[1],
                "Jacobian re-solved at an unchanged iterate after a rejection"
            );
        }
    }

    // ------------------------------------------------------------------
    // Pseudo-transient continuation (Ψtc)
    // ------------------------------------------------------------------

    /// Multi-scale stiff algebraic surrogate (driven-cavity-style): three
    /// decoupled reaction equations `F_i(x) = atan(σ_i (x_i − t_i))` with
    /// scales σ = [10, 40, 160]. From a far initial guess the Jacobian
    /// `J_ii = σ_i / (1 + (σ_i (x_i − t_i))²)` is numerically near zero, so
    /// the Newton step overshoots by orders of magnitude and the Armijo
    /// search cannot repair the direction — plain Newton with line search
    /// stalls. Ψtc follows the pseudo-time flow until the residuals enter
    /// the steep region, where SER grows Δt and Newton takes over.
    const ATAN_SCALES: [f64; 3] = [10.0, 40.0, 160.0];
    const ATAN_TARGETS: [f64; 3] = [1.0, -0.5, 0.2];

    fn atan_residual(x: &[f64]) -> Vec<f64> {
        (0..3)
            .map(|i| (ATAN_SCALES[i] * (x[i] - ATAN_TARGETS[i])).atan())
            .collect()
    }

    fn atan_matvec(x: &[f64], v: &[f64]) -> Vec<f64> {
        (0..3)
            .map(|i| {
                let s = ATAN_SCALES[i] * (x[i] - ATAN_TARGETS[i]);
                ATAN_SCALES[i] / (1.0 + s * s) * v[i]
            })
            .collect()
    }

    fn atan_check_solution(x: &[f64]) {
        for (xi, ti) in x.iter().zip(ATAN_TARGETS.iter()) {
            assert!((xi - ti).abs() < 1e-8, "x = {xi}, target = {ti}");
        }
    }

    #[test]
    fn test_continuation_converges_where_line_search_newton_fails() {
        let x0 = vec![5.0_f64, 3.0, -4.0];
        let base_newton = NewtonConfig {
            max_iterations: 200,
            absolute_tolerance: 1e-12,
            relative_tolerance: 1e-14,
            line_search: LineSearch::backtracking(),
            ..Default::default()
        };

        // Plain matrix-free Newton + Armijo cannot make progress from this far
        // guess: depending on the inner GMRES stopping path it either exhausts
        // the line search or rejects an unconverged Jacobian solve.
        let plain_cfg = InexactNewtonConfig {
            newton: base_newton.clone(),
            ..Default::default()
        };
        let plain = newton_raphson_continuation(&x0, atan_residual, atan_matvec, &plain_cfg);
        assert!(
            matches!(
                plain,
                Err(NewtonError::LineSearchFailed { .. })
                    | Err(NewtonError::LinearSolveFailed { .. })
            ),
            "plain Newton+Armijo should fail from the far guess, got {plain:?}"
        );

        // Ψtc follows the pseudo-time ODE through the flat region and
        // converges.
        let ptc_cfg = InexactNewtonConfig {
            newton: base_newton,
            continuation: ContinuationStrategy::PseudoTransient {
                dt0: 1.0,
                growth: 10.0,
                max_dt: 1e12,
            },
            ..Default::default()
        };
        let ptc = newton_raphson_continuation(&x0, atan_residual, atan_matvec, &ptc_cfg).unwrap();
        assert!(
            ptc.converged,
            "Ψtc should converge, residual = {:.2e}",
            ptc.residual
        );
        atan_check_solution(&ptc.x);
    }

    #[test]
    fn test_continuation_auto_escalation_on_line_search_failure() {
        // Same far guess, but with continuation disabled and the escalation
        // flag set: the solver starts as plain Newton, the line search fails
        // at iteration 0, and it escalates to Ψtc automatically.
        let x0 = vec![5.0_f64, 3.0, -4.0];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 200,
                absolute_tolerance: 1e-12,
                relative_tolerance: 1e-14,
                line_search: LineSearch::backtracking(),
                ..Default::default()
            },
            auto_pseudo_transient: true,
            ..Default::default()
        };

        let result = newton_raphson_continuation(&x0, atan_residual, atan_matvec, &config).unwrap();
        assert!(
            result.converged,
            "escalated Ψtc should converge, residual = {:.2e}",
            result.residual
        );
        atan_check_solution(&result.x);
    }

    #[test]
    fn test_ser_dt_update() {
        // Residual halving doubles Δt (SER: Δt·‖F_old‖/‖F_new‖).
        let dt = pseudo_transient_dt_update(1.0, 2.0, 1.0, 1e-3, 10.0, 1e6);
        assert!((dt - 2.0).abs() < 1e-15, "SER doubling: got {dt}");

        // Growth cap: ratio 100 with growth 10 grows Δt by only ×10.
        let dt = pseudo_transient_dt_update(1.0, 100.0, 1.0, 1e-3, 10.0, 1e6);
        assert!((dt - 10.0).abs() < 1e-12, "growth cap: got {dt}");

        // Clamp to max_dt — Δt → ∞ behaviour: in the basin the shift 1/Δt
        // vanishes and the step is pure Newton.
        let dt = pseudo_transient_dt_update(1e6, 10.0, 1.0, 1e-3, 10.0, 1e6);
        assert!((dt - 1e6).abs() < 1e-3, "max_dt clamp: got {dt}");

        // Residual growth shrinks Δt but never below dt0.
        let dt = pseudo_transient_dt_update(1e-2, 1.0, 100.0, 1e-3, 10.0, 1e6);
        assert!((dt - 1e-3).abs() < 1e-18, "dt0 floor: got {dt}");

        // Zero new residual (exact solve) grows Δt by the growth factor.
        let dt = pseudo_transient_dt_update(1.0, 1.0, 0.0, 1e-3, 10.0, 1e6);
        assert!((dt - 10.0).abs() < 1e-12, "zero residual: got {dt}");
    }

    #[test]
    fn test_continuation_terminal_quadratic_convergence() {
        // Easy problem with a small dt0: Ψtc must still finish with
        // Newton-like steps once SER has grown Δt toward max_dt (shift
        // 1/Δt ≈ 0 → pure Newton). The residual closure is evaluated both
        // after a step and at the next iteration start, so consecutive
        // duplicates are removed before checking the contraction history.
        let history = std::cell::RefCell::new(Vec::new());
        let x0 = sonium_solvers_core::array![1.0_f64];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 100,
                absolute_tolerance: 1e-13,
                relative_tolerance: 1e-15,
                ..Default::default()
            },
            continuation: ContinuationStrategy::PseudoTransient {
                dt0: 0.1,
                growth: 10.0,
                max_dt: 1e12,
            },
            ..Default::default()
        };

        let result = newton_raphson_continuation(
            &x0,
            |x| {
                let r = sonium_solvers_core::array![x[0] * x[0] - 4.0];
                history.borrow_mut().push(r[0].abs());
                r
            },
            |x, v| sonium_solvers_core::array![2.0 * x[0] * v[0]],
            &config,
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-10);

        let mut h = history.borrow().clone();
        h.dedup();
        let n = h.len();
        assert!(n >= 5, "history too short: {h:?}");
        // Accelerating tail: the contraction ratio r_{k+1}/r_k collapses as
        // Δt grows — superlinear approach to the root (not the linear crawl
        // of the small-Δt phase).
        let (r2, r1, r0) = (h[n - 3], h[n - 2], h[n - 1]);
        assert!(
            r1 <= 1e-2 * r2,
            "penultimate step not fast: {r2:.3e} -> {r1:.3e} (history: {h:?})"
        );
        assert!(
            r0 <= 1e-4 * r1,
            "terminal step not superlinear: {r1:.3e} -> {r0:.3e} (history: {h:?})"
        );
    }

    #[test]
    fn test_continuation_large_dt_recovers_newton_quadratic_rate() {
        // With dt0 = 1e10 the shift 1/Δt is negligible from the first step,
        // so Ψtc reduces to pure Newton and must show the textbook quadratic
        // rate of F = x² − 4 (r' ≈ r²/16).
        let history = std::cell::RefCell::new(Vec::new());
        let x0 = sonium_solvers_core::array![3.0_f64];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 20,
                absolute_tolerance: 1e-13,
                relative_tolerance: 1e-15,
                ..Default::default()
            },
            continuation: ContinuationStrategy::PseudoTransient {
                dt0: 1e10,
                growth: 10.0,
                max_dt: 1e12,
            },
            ..Default::default()
        };

        let result = newton_raphson_continuation(
            &x0,
            |x| {
                let r = sonium_solvers_core::array![x[0] * x[0] - 4.0];
                history.borrow_mut().push(r[0].abs());
                r
            },
            |x, v| sonium_solvers_core::array![2.0 * x[0] * v[0]],
            &config,
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-10);

        let mut h = history.borrow().clone();
        h.dedup();
        // Last step from a residual still above the inner-solve inexactness
        // floor (η = 1e-8 relative): must satisfy the quadratic bound with
        // 2x margin on the constant.
        let i = h
            .iter()
            .rposition(|&r| r > 1e-6)
            .expect("need a residual above 1e-6 in {h:?}");
        assert!(i + 1 < h.len(), "no successor residual: {h:?}");
        assert!(
            h[i + 1] <= 0.5 * h[i] * h[i],
            "not quadratic: {:.3e} -> {:.3e} (history: {h:?})",
            h[i],
            h[i + 1]
        );
    }

    #[test]
    fn test_continuation_invalid_config() {
        let x0 = sonium_solvers_core::array![1.0_f64];
        let residual = |x: &[f64]| sonium_solvers_core::array![x[0] * x[0] - 4.0];
        let matvec = |x: &[f64], v: &[f64]| sonium_solvers_core::array![2.0 * x[0] * v[0]];

        // Trust region is not combinable with the continuation entry point.
        let tr_cfg = InexactNewtonConfig {
            trust_region: Some(TrustRegionConfig::default()),
            ..Default::default()
        };
        let result = newton_raphson_continuation(&x0, residual, matvec, &tr_cfg);
        assert!(
            matches!(result, Err(NewtonError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );

        // Degenerate Ψtc parameters are rejected.
        let bad_ptc = InexactNewtonConfig {
            continuation: ContinuationStrategy::PseudoTransient {
                dt0: -1.0,
                growth: 10.0,
                max_dt: 1e12,
            },
            ..Default::default()
        };
        let result = newton_raphson_continuation(&x0, residual, matvec, &bad_ptc);
        assert!(
            matches!(result, Err(NewtonError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    #[test]
    fn test_continuation_f32() {
        // Same easy problem in f32, exercising the generic scalar path.
        let x0 = sonium_solvers_core::array![1.0_f32];
        let config = InexactNewtonConfig {
            newton: NewtonConfig {
                max_iterations: 100,
                absolute_tolerance: 1e-4,
                ..Default::default()
            },
            continuation: ContinuationStrategy::PseudoTransient {
                dt0: 0.1,
                growth: 10.0,
                max_dt: 1e8,
            },
            ..Default::default()
        };

        let result = newton_raphson_continuation(
            &x0,
            |x| sonium_solvers_core::array![x[0] * x[0] - 4.0_f32],
            |x, v| sonium_solvers_core::array![2.0_f32 * x[0] * v[0]],
            &config,
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0_f32).abs() < 1e-3);
    }
}
