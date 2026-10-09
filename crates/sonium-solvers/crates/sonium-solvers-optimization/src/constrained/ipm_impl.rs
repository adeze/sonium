//! Primal-dual interior-point method (IPM) for smooth nonlinear programming.
//!
//! IPOPT-class method scoped to the crate's dense small-to-moderate-`n`
//! regime. Inequalities `c(x) >= 0` are slacked (`c(x) - s = 0`, `s > 0`)
//! and equalities are kept as-is. Box bounds from the config are converted
//! to additional (linear) inequality rows — a deliberate simplification
//! over IPOPT, which treats bounds with dedicated slack pairs; here they
//! simply widen the inequality block, which is exact and keeps one code
//! path at the cost of up to `2n` extra slacks.
//!
//! The perturbed primal-dual KKT system
//!
//! ```text
//! [ H    -Ae^T  -Ai^T   0 ] [dx]     [r_d ]
//! [ Ae     0      0     0 ] [dy]     [r_e ]
//! [ Ai     0      0    -I ] [ds] = - [r_pi]
//! [ 0      0      Z     S ] [dz]     [r_c ]
//! ```
//!
//! is reduced by the standard slack elimination (`ds = Ai dx + r_pi`,
//! `dz = -S^-1 (r_c + Z ds)`) to the condensed symmetric system
//!
//! ```text
//! [ H + Ai^T S^-1 Z Ai   -Ae^T ] [dx]   [ -r_d - Ai^T S^-1 (r_c + Z r_pi) ]
//! [     -Ae                 0  ] [dy] = [             r_e                 ].
//! ```
//!
//! The `(1,1)` block is factored with the existing dense GMW modified
//! Cholesky ([`crate::util::dense_linalg`]), which doubles as
//! the inertia correction: whenever the block is indefinite the GMW
//! diagonal regularisation makes the factorisation succeed. The remaining
//! equality Schur complement (`Ae K^-1 Ae^T`, small and SPD for full-rank
//! `Ae`) is solved by Cholesky with escalating diagonal regularisation.
//!
//! Globalisation is a Fletcher-Leyffer filter on the pair
//! `(theta, phi_mu)` with `theta` the L1 constraint violation and
//! `phi_mu = f - mu * sum(log s)` the barrier objective, combined with an
//! Armijo sufficient-decrease test on `phi_mu` under a switching condition
//! (simplified single-exponent form of IPOPT's two-exponent condition) and
//! the fraction-to-the-boundary rule on both primal slacks and dual
//! variables. The barrier parameter follows the monotone Fiacco-McCormick
//! update: `mu` is only decreased once the barrier KKT error drops below
//! `kappa_eps * mu`, then `mu <- max(mu_min, min(kappa_mu * mu, mu^1.5))`.
//!
//! Documented simplifications vs IPOPT:
//! - no Mehrotra predictor-corrector and no adaptive (LOQO-quality-function)
//!   barrier update — monotone Fiacco-McCormick only;
//! - no warm-starting (multipliers always start from `y = 0`, `z = mu/s`);
//! - the feasibility-restoration phase is a plain Gauss-Newton minimisation
//!   of the squared constraint violation with Levenberg damping and an
//!   Armijo backtrack on the L1 violation `theta` (IPOPT re-runs the full
//!   barrier problem on the violation instead). When the filter line
//!   search fails, up to `max_restoration_iterations` restoration steps
//!   are taken; if `theta` is at least halved (or below
//!   `primal_tolerance`) the barrier loop is re-entered with rebuilt
//!   slacks, `z = mu/s`, `y = 0`, and a cleared filter. Otherwise the run
//!   terminates with [`OptimStatus::RestorationFailed`];
//! - dense algebra throughout: intended for `n` up to a few hundred, not
//!   the sparse large-scale regime IPOPT targets.
//!
//! When no exact Lagrangian Hessian is supplied, a damped BFGS
//! approximation built from Lagrangian-gradient differences (Powell
//! damping, same convention as [`super::sqp_impl`]) is used instead.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{
    Bound, Constraint, ConstraintKind, IpmDiagnostics, KktMultipliers, OptimError, OptimResult,
    OptimStatus, RealScalar, validate_bounds,
};
use crate::util::dense_linalg::{
    cholesky, modified_cholesky, solve_lower_triangular, solve_lower_triangular_transpose,
};

#[inline]
fn cst<T: RealScalar>(v: f64) -> T {
    T::from_f64(v).unwrap()
}

/// Convert the primal-dual IPM multipliers into the public KKT ordering.
///
/// Internally, IPM keeps equality rows first in `y`, user inequality rows
/// first in `z`, and appends bound rows.  The public report preserves the
/// caller's mixed constraint order and separates lower/upper bound entries.
fn native_kkt_multipliers<T: RealScalar>(
    constraints: &[Constraint<T>],
    bounds: &[Bound<T>],
    equality_multipliers: &[T],
    inequality_multipliers: &[T],
    bound_rows: &[(usize, T, bool)],
) -> Result<KktMultipliers<T>, OptimError> {
    let expected_equalities = constraints
        .iter()
        .filter(|constraint| constraint.kind == ConstraintKind::Equality)
        .map(|constraint| constraint.dim)
        .sum::<usize>();
    let expected_user_inequalities = constraints
        .iter()
        .filter(|constraint| constraint.kind == ConstraintKind::Inequality)
        .map(|constraint| constraint.dim)
        .sum::<usize>();
    let expected_inequalities = expected_user_inequalities
        .checked_add(bound_rows.len())
        .ok_or(OptimError::DimensionMismatch {
            expected: usize::MAX,
            got: inequality_multipliers.len(),
        })?;
    let expected_constraints = expected_equalities
        .checked_add(expected_user_inequalities)
        .ok_or(OptimError::DimensionMismatch {
            expected: usize::MAX,
            got: constraints.len(),
        })?;
    if equality_multipliers.len() != expected_equalities {
        return Err(OptimError::DimensionMismatch {
            expected: expected_equalities,
            got: equality_multipliers.len(),
        });
    }
    if inequality_multipliers.len() != expected_inequalities {
        return Err(OptimError::DimensionMismatch {
            expected: expected_inequalities,
            got: inequality_multipliers.len(),
        });
    }
    if let Some(index) = bound_rows
        .iter()
        .map(|(index, _, _)| *index)
        .find(|index| *index >= bounds.len())
    {
        return Err(OptimError::DimensionMismatch {
            expected: bounds.len(),
            got: index.saturating_add(1),
        });
    }

    let mut multipliers = KktMultipliers {
        constraints: Vec::with_capacity(expected_constraints),
        lower_bounds: vec![T::zero(); bounds.len()],
        upper_bounds: vec![T::zero(); bounds.len()],
    };
    let mut equality_row = 0;
    let mut inequality_row = 0;
    for constraint in constraints {
        for _ in 0..constraint.dim {
            let multiplier = match constraint.kind {
                ConstraintKind::Equality => {
                    let value = equality_multipliers[equality_row];
                    equality_row += 1;
                    value
                }
                ConstraintKind::Inequality => {
                    let value = inequality_multipliers[inequality_row];
                    inequality_row += 1;
                    value
                }
            };
            multipliers.constraints.push(multiplier);
        }
    }
    for (row, (index, _, is_lower)) in bound_rows.iter().enumerate() {
        let multiplier = inequality_multipliers[expected_user_inequalities + row];
        if *is_lower {
            multipliers.lower_bounds[*index] = multiplier;
        } else {
            multipliers.upper_bounds[*index] = multiplier;
        }
    }
    Ok(multipliers)
}

fn eval_objective<T, F>(f: &F, x: &[T], iteration: usize) -> Result<T, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let value =
        catch_unwind(AssertUnwindSafe(|| f(x))).map_err(|_| OptimError::CallbackPanicked {
            iteration,
            stage: "objective".to_string(),
        })?;
    if !value.is_finite() {
        return Err(OptimError::NumericalError {
            iteration,
            message: "objective callback returned a non-finite value".to_string(),
        });
    }
    Ok(value)
}

fn eval_gradient<T, Fg>(
    grad_fn: &Fg,
    x: &[T],
    n: usize,
    iteration: usize,
) -> Result<Vec<T>, OptimError>
where
    T: RealScalar,
    Fg: Fn(&[T]) -> Vec<T>,
{
    let gradient = catch_unwind(AssertUnwindSafe(|| grad_fn(x))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: "gradient".to_string(),
        }
    })?;
    if gradient.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: gradient.len(),
        });
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: "gradient callback returned a non-finite value".to_string(),
        });
    }
    Ok(gradient)
}

fn eval_lagrangian_hessian<T, H>(
    hessian_fn: &H,
    x: &[T],
    multipliers: &[T],
    n: usize,
    iteration: usize,
) -> Result<sonium_solvers_core::dense::DenseMatrix<T>, OptimError>
where
    T: RealScalar,
    H: Fn(&[T], &[T]) -> sonium_solvers_core::dense::DenseMatrix<T>,
{
    let hessian = catch_unwind(AssertUnwindSafe(|| hessian_fn(x, multipliers))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: "Lagrangian Hessian".to_string(),
        }
    })?;
    if hessian.nrows() != n || hessian.ncols() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n * n,
            got: hessian.nrows() * hessian.ncols(),
        });
    }
    if (0..n).any(|row| (0..n).any(|column| !hessian[[row, column]].is_finite())) {
        return Err(OptimError::NumericalError {
            iteration,
            message: "Lagrangian Hessian callback returned a non-finite value".to_string(),
        });
    }
    Ok(hessian)
}

/// Configuration for the primal-dual interior-point method.
#[derive(Debug, Clone)]
pub struct IpmConfig<T: RealScalar> {
    /// Maximum primal-dual iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the (scaled) dual infeasibility
    /// `||grad f - Ae^T y - Ai^T z||_inf`.
    pub dual_tolerance: T,
    /// Convergence tolerance on the primal infeasibility
    /// `max(||c_eq||_inf, ||c_ineq - s||_inf)`.
    pub primal_tolerance: T,
    /// Convergence tolerance on the (scaled) complementarity `||S z||_inf`.
    pub complementarity_tolerance: T,
    /// Initial barrier parameter.
    pub mu_init: T,
    /// Lower bound for the barrier parameter (Fiacco-McCormick floor).
    pub mu_min: T,
    /// Linear decrease factor `kappa_mu` of the barrier update.
    pub mu_decrease: T,
    /// Superlinear exponent `theta_mu` of the barrier update.
    pub mu_superlinear_exp: T,
    /// The barrier parameter is only decreased once the barrier KKT error
    /// is below `barrier_tolerance_factor * mu` (IPOPT's `kappa_eps`).
    pub barrier_tolerance_factor: T,
    /// Fraction-to-the-boundary parameter (0.995 is the IPOPT default).
    pub tau: T,
    /// Filter margin on the constraint violation (gamma_theta).
    pub filter_gamma_theta: T,
    /// Filter margin on the barrier objective (gamma_phi).
    pub filter_gamma_phi: T,
    /// Switching-condition constant delta.
    pub filter_delta: T,
    /// Switching-condition exponent s_theta (> 1).
    pub filter_s_theta: T,
    /// Armijo sufficient-decrease constant on the barrier objective.
    pub armijo_eta: T,
    /// Maximum backtracking steps per line search.
    pub max_line_search: usize,
    /// Maximum Gauss-Newton iterations of the feasibility-restoration
    /// phase entered when the filter line search fails.
    pub max_restoration_iterations: usize,
    /// Optional box bounds; converted to additional inequality rows.
    pub bounds: Vec<Bound<T>>,
}

impl<T: RealScalar> Default for IpmConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            dual_tolerance: cst(1e-9),
            primal_tolerance: cst(1e-9),
            complementarity_tolerance: cst(1e-9),
            mu_init: cst(0.1),
            mu_min: cst(1e-12),
            mu_decrease: cst(0.2),
            mu_superlinear_exp: cst(1.5),
            barrier_tolerance_factor: cst(10.0),
            tau: cst(0.995),
            filter_gamma_theta: cst(1e-5),
            filter_gamma_phi: cst(1e-5),
            filter_delta: cst(1.0),
            filter_s_theta: cst(1.5),
            armijo_eta: cst(1e-4),
            max_line_search: 25,
            max_restoration_iterations: 20,
            bounds: Vec::new(),
        }
    }
}

impl<T: RealScalar> IpmConfig<T> {
    /// Validate configuration values before invoking objective or constraint
    /// callbacks. A non-finite or non-positive barrier/KKT parameter can make
    /// the primal-dual arithmetic silently leave its feasible domain.
    pub fn validate(&self) -> Result<(), OptimError> {
        if self.max_iterations == 0 {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "max_iterations must be positive".to_string(),
            });
        }

        for (name, value) in [
            ("dual_tolerance", self.dual_tolerance),
            ("primal_tolerance", self.primal_tolerance),
            ("complementarity_tolerance", self.complementarity_tolerance),
            ("mu_init", self.mu_init),
            ("mu_min", self.mu_min),
            ("barrier_tolerance_factor", self.barrier_tolerance_factor),
            ("filter_delta", self.filter_delta),
        ] {
            if !value.is_finite() || value <= T::zero() {
                return Err(OptimError::NumericalError {
                    iteration: 0,
                    message: format!("{name} must be finite and positive"),
                });
            }
        }
        if self.mu_min > self.mu_init {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "mu_min must not exceed mu_init".to_string(),
            });
        }
        if !self.mu_decrease.is_finite()
            || self.mu_decrease <= T::zero()
            || self.mu_decrease >= T::one()
        {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "mu_decrease must be finite and in (0, 1)".to_string(),
            });
        }
        if !self.mu_superlinear_exp.is_finite() || self.mu_superlinear_exp <= T::one() {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "mu_superlinear_exp must be finite and greater than 1".to_string(),
            });
        }
        if !self.tau.is_finite() || self.tau <= T::zero() || self.tau >= T::one() {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "tau must be finite and in (0, 1)".to_string(),
            });
        }
        for (name, value) in [
            ("filter_gamma_theta", self.filter_gamma_theta),
            ("filter_gamma_phi", self.filter_gamma_phi),
        ] {
            if !value.is_finite() || value <= T::zero() || value >= T::one() {
                return Err(OptimError::NumericalError {
                    iteration: 0,
                    message: format!("{name} must be finite and in (0, 1)"),
                });
            }
        }
        if !self.filter_s_theta.is_finite() || self.filter_s_theta <= T::one() {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "filter_s_theta must be finite and greater than 1".to_string(),
            });
        }
        if !self.armijo_eta.is_finite()
            || self.armijo_eta <= T::zero()
            || self.armijo_eta >= T::one()
        {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "armijo_eta must be finite and in (0, 1)".to_string(),
            });
        }
        if self.max_line_search == 0 {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: "max_line_search must be positive".to_string(),
            });
        }
        Ok(())
    }
}

/// Minimise `f(x)` subject to constraints with a primal-dual
/// interior-point method.
///
/// `lagrangian_hessian_fn`, when `Some`, must return the dense Hessian
/// `∇²_xx L(x, lambda)` with `L = f - sum_i lambda_i c_i` and `lambda`
/// ordered exactly like the rows of `constraints` (equality multipliers
/// free, inequality multipliers non-negative). When `None`, a damped-BFGS
/// approximation is maintained instead.
pub fn ipm<T, F, Fg, H>(
    f: &F,
    grad_fn: &Fg,
    lagrangian_hessian_fn: Option<&H>,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &IpmConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
    H: Fn(&[T], &[T]) -> sonium_solvers_core::dense::DenseMatrix<T>,
{
    config.validate()?;
    if let Some(index) = x0.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("initial point entry {index} is non-finite"),
        });
    }
    validate_bounds(&config.bounds, x0.len())?;
    for (index, bound) in config.bounds.iter().enumerate() {
        if bound.lower.is_some_and(|lower| x0[index] < lower)
            || bound.upper.is_some_and(|upper| x0[index] > upper)
        {
            return Err(OptimError::IncompatibleStartingPoint {
                index,
                value: x0[index].to_f64().unwrap_or(f64::NAN),
                lower: bound.lower.map(|value| value.to_f64().unwrap_or(f64::NAN)),
                upper: bound.upper.map(|value| value.to_f64().unwrap_or(f64::NAN)),
            });
        }
    }
    // Configuration and feasibility checks intentionally precede callbacks:
    // an invalid starting point must not execute user code.
    super::validate_optimizer_start(f, grad_fn, x0, constraints)?;
    if let Some(hessian_fn) = lagrangian_hessian_fn {
        super::validate_hessian_start(
            hessian_fn,
            x0,
            constraints.iter().map(|constraint| constraint.dim).sum(),
            "IPM Hessian preflight",
        )?;
    }
    let (mut result, ipm_diagnostics, native_multipliers) =
        ipm_impl(f, grad_fn, lagrangian_hessian_fn, x0, constraints, config)?;
    if !config.bounds.is_empty() && config.bounds.len() != result.x.len() {
        return Err(OptimError::DimensionMismatch {
            expected: result.x.len(),
            got: config.bounds.len(),
        });
    }
    // IPM already owns the final primal-dual multipliers. Publish those
    // values in the original-problem KKT report instead of replacing them
    // with an identity-Hessian active-set estimate.
    result.attach_kkt(constraints, &config.bounds, native_multipliers)?;
    if let Some(report) = result.kkt.as_mut() {
        report.ipm = Some(ipm_diagnostics);
    }
    let tolerance = config
        .primal_tolerance
        .max(config.dual_tolerance)
        .max(config.complementarity_tolerance);
    result.gate_kkt(tolerance);
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

/// Outcome of [`ipm_impl`]: optimization result, solver diagnostics, and KKT multipliers.
type IpmSolveOutcome<T> = (OptimResult<T>, IpmDiagnostics<T>, KktMultipliers<T>);

fn ipm_impl<T, F, Fg, H>(
    f: &F,
    grad_fn: &Fg,
    lagrangian_hessian_fn: Option<&H>,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &IpmConfig<T>,
) -> Result<IpmSolveOutcome<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
    H: Fn(&[T], &[T]) -> sonium_solvers_core::dense::DenseMatrix<T>,
{
    let n = x0.len();
    let eq_cons: Vec<&Constraint<T>> = constraints
        .iter()
        .filter(|c| c.kind == ConstraintKind::Equality)
        .collect();
    let iq_cons: Vec<&Constraint<T>> = constraints
        .iter()
        .filter(|c| c.kind == ConstraintKind::Inequality)
        .collect();
    let me: usize = eq_cons.iter().map(|c| c.dim).sum();
    let mi_user: usize = iq_cons.iter().map(|c| c.dim).sum();

    // Bounds become extra linear inequality rows: (var, bound_value, is_lower).
    let mut bound_rows: Vec<(usize, T, bool)> = Vec::new();
    if config.bounds.len() == n {
        for (j, b) in config.bounds.iter().enumerate() {
            if let Some(lb) = b.lower {
                bound_rows.push((j, lb, true));
            }
            if let Some(ub) = b.upper {
                bound_rows.push((j, ub, false));
            }
        }
    }
    let mi = mi_user + bound_rows.len();

    let eval_eq = |x: &[T], iteration: usize| eval_eq_rows(&eq_cons, x, me, n, iteration);

    let eval_ineq =
        |x: &[T], iteration: usize| eval_ineq_rows(&iq_cons, &bound_rows, x, mi, n, iteration);

    // Constraint values only (for the filter's violation measure).
    let eval_violation = |x: &[T], iteration: usize| l1_violation(constraints, x, iteration);

    // Starting point: strictly interior w.r.t. bounds, positive slacks.
    let mut x = x0.to_vec();
    if config.bounds.len() == n {
        for (j, b) in config.bounds.iter().enumerate() {
            let margin = |v: T| cst::<T>(1e-4) * T::max(T::one(), v.abs());
            match (b.lower, b.upper) {
                (Some(lb), Some(ub)) => {
                    if lb >= ub {
                        return Err(OptimError::InvalidBounds {
                            index: j,
                            lower: lb.to_f64().unwrap_or(0.0),
                            upper: ub.to_f64().unwrap_or(0.0),
                        });
                    }
                    let m = cst::<T>(0.01) * (ub - lb);
                    x[j] = T::min(T::max(x[j], lb + m), ub - m);
                }
                (Some(lb), None) => {
                    if x[j] <= lb {
                        x[j] = lb + margin(lb);
                    }
                }
                (None, Some(ub)) => {
                    if x[j] >= ub {
                        x[j] = ub - margin(ub);
                    }
                }
                (None, None) => {}
            }
        }
    }

    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;
    let mut hess_evals = 0_usize;
    let mut fx = eval_objective(f, &x, 0)?;
    let mut gx = eval_gradient(grad_fn, &x, n, 0)?;

    let (ci0, _) = eval_ineq(&x, 0)?;

    let mut mu = config.mu_init;
    let mut s: Vec<T> = ci0.iter().map(|&v| T::max(v, cst(1e-4))).collect();
    let mut z: Vec<T> = s.iter().map(|&si| mu / si).collect();
    let mut y = vec![T::zero(); me];

    let mut bfgs = sonium_solvers_core::dense::DenseMatrix::eye(n);
    let mut filter: Vec<(T, T)> = Vec::new();

    let mut status = OptimStatus::MaxIterationsReached;
    let mut converged = false;
    let mut iterations = 0_usize;
    let mut dual_infeas = T::infinity();
    let mut primal_infeas = T::infinity();
    let mut complementarity = T::infinity();
    let mut restoration_attempts = 0_usize;
    let mut restoration_successes = 0_usize;
    let mut restoration_failures = 0_usize;
    let mut line_search_backtracks = 0_usize;

    'outer: for iter in 0..config.max_iterations {
        let (ce, je) = eval_eq(&x, iter)?;
        let (ci, ji) = eval_ineq(&x, iter)?;

        // Primal-dual KKT residuals.
        let mut r_d = gx.clone();
        for r in 0..me {
            for j in 0..n {
                r_d[j] -= je[[r, j]] * y[r];
            }
        }
        for r in 0..mi {
            for j in 0..n {
                r_d[j] -= ji[[r, j]] * z[r];
            }
        }
        let r_pi: Vec<T> = (0..mi).map(|i| ci[i] - s[i]).collect();
        let r_c: Vec<T> = (0..mi).map(|i| s[i] * z[i] - mu).collect();

        let inf = |v: &[T]| v.iter().map(|&a| a.abs()).fold(T::zero(), T::max);

        // IPOPT-style scaling of the optimality error.
        let mult_sum = y
            .iter()
            .chain(z.iter())
            .fold(T::zero(), |a, &v| a + v.abs());
        let s_d = T::max(cst(100.0), mult_sum / cst((me + mi).max(1) as f64)) / cst(100.0);
        let s_c = if mi > 0 {
            let z_avg = z.iter().fold(T::zero(), |a, &v| a + v.abs()) / cst(mi as f64);
            T::max(cst(100.0), z_avg) / cst(100.0)
        } else {
            T::one()
        };

        dual_infeas = inf(&r_d) / s_d;
        primal_infeas = T::max(inf(&ce), inf(&r_pi));
        complementarity = if mi > 0 {
            inf(&(0..mi).map(|i| s[i] * z[i]).collect::<Vec<_>>()) / s_c
        } else {
            T::zero()
        };
        let barrier_err = T::max(
            T::max(dual_infeas, primal_infeas),
            if mi > 0 { inf(&r_c) / s_c } else { T::zero() },
        );

        if dual_infeas < config.dual_tolerance
            && primal_infeas < config.primal_tolerance
            && complementarity < config.complementarity_tolerance
        {
            converged = true;
            status = OptimStatus::Converged;
            iterations = iter;
            break;
        }

        // (1,1) block K = H + Ai^T S^-1 Z Ai.
        let mut kmat = match lagrangian_hessian_fn {
            Some(h) => {
                hess_evals += 1;
                let lambda = user_multipliers(&y, &z, constraints);
                eval_lagrangian_hessian(h, &x, &lambda, n, iter)?
            }
            None => bfgs.clone(),
        };
        for i in 0..mi {
            let sigma = z[i] / s[i];
            for a in 0..n {
                let jia = ji[[i, a]];
                if jia == T::zero() {
                    continue;
                }
                for b in 0..=a {
                    let upd = sigma * jia * ji[[i, b]];
                    kmat[[a, b]] += upd;
                    if b != a {
                        kmat[[b, a]] += upd;
                    }
                }
            }
        }

        // Condensed right-hand side.
        let mut rhs1 = r_d.iter().map(|&v| -v).collect::<Vec<_>>();
        for i in 0..mi {
            let w = (r_c[i] + z[i] * r_pi[i]) / s[i];
            for j in 0..n {
                rhs1[j] -= ji[[i, j]] * w;
            }
        }

        let (dx, dy) = match solve_condensed(&kmat, &je, &rhs1, &ce, n, me) {
            Some(sol) => sol,
            None => {
                status = OptimStatus::NumericalError;
                iterations = iter;
                break;
            }
        };

        // Slack and dual steps from the elimination.
        let ds: Vec<T> = (0..mi)
            .map(|i| {
                let mut v = r_pi[i];
                for j in 0..n {
                    v += ji[[i, j]] * dx[j];
                }
                v
            })
            .collect();
        let dz: Vec<T> = (0..mi).map(|i| -(r_c[i] + z[i] * ds[i]) / s[i]).collect();

        let alpha_p = frac_to_boundary(&s, &ds, config.tau);
        let alpha_d = frac_to_boundary(&z, &dz, config.tau);

        // Filter line search on (theta, phi_mu).
        let theta_k = eval_violation(&x, iter)?;
        let phi_k = barrier_objective(fx, &s, mu);
        let mut dphi = T::zero();
        for j in 0..n {
            dphi += gx[j] * dx[j];
        }
        for i in 0..mi {
            dphi -= mu * ds[i] / s[i];
        }

        let mut alpha = alpha_p;
        let mut accepted = false;
        let mut used_armijo = false;
        let mut x_t = x.clone();
        let mut s_t = s.clone();
        let mut fx_t = fx;

        for _ in 0..config.max_line_search {
            if alpha <= cst(1e-14) {
                break;
            }
            for j in 0..n {
                x_t[j] = x[j] + alpha * dx[j];
            }
            for i in 0..mi {
                s_t[i] = s[i] + alpha * ds[i];
            }
            let fx_trial = match eval_objective(f, &x_t, iter) {
                Ok(value) => value,
                Err(OptimError::NumericalError { .. }) => {
                    alpha *= cst(0.5);
                    line_search_backtracks += 1;
                    continue;
                }
                Err(error) => return Err(error),
            };
            fun_evals += 1;
            let theta_t = eval_violation(&x_t, iter)?;
            let phi_t = barrier_objective(fx_trial, &s_t, mu);

            let m = alpha * dphi;
            let switching =
                m < T::zero() && -m > config.filter_delta * theta_k.powf(config.filter_s_theta);
            let ok = if switching {
                // Armijo sufficient decrease on the barrier objective.
                phi_t <= phi_k + config.armijo_eta * m
            } else {
                // Filter acceptance w.r.t. the current iterate and all entries.
                let current_ok = theta_t <= (T::one() - config.filter_gamma_theta) * theta_k
                    || phi_t <= phi_k - config.filter_gamma_phi * theta_k;
                current_ok
                    && filter_acceptable(
                        &filter,
                        theta_t,
                        phi_t,
                        config.filter_gamma_theta,
                        config.filter_gamma_phi,
                    )
            };
            if ok {
                accepted = true;
                used_armijo = switching;
                fx_t = fx_trial;
                break;
            }
            alpha *= cst(0.5);
            line_search_backtracks += 1;
        }

        if !accepted {
            restoration_attempts += 1;
            // Restoration phase: minimise the constraint violation with
            // Gauss-Newton steps, then re-enter the barrier loop at the
            // restored point (rebuilt slacks, z = mu/s, y = 0, cleared
            // filter). If theta cannot be meaningfully reduced the
            // problem is likely infeasible: fail cleanly.
            match feasibility_restoration(
                f,
                &x,
                mu,
                &eq_cons,
                &iq_cons,
                &bound_rows,
                constraints,
                config,
                &mut fun_evals,
            )? {
                Some((x_r, s_r, z_r, fx_r)) => {
                    restoration_successes += 1;
                    x = x_r;
                    s = s_r;
                    z = z_r;
                    for yi in y.iter_mut() {
                        *yi = T::zero();
                    }
                    fx = fx_r;
                    gx = eval_gradient(grad_fn, &x, n, iter + 1)?;
                    grad_evals += 1;
                    filter.clear();
                    iterations = iter + 1;
                    continue 'outer;
                }
                None => {
                    restoration_failures += 1;
                    status = OptimStatus::RestorationFailed;
                    iterations = iter;
                    break;
                }
            }
        }

        // Dual updates (independent fraction-to-the-boundary step).
        for i in 0..mi {
            z[i] += alpha_d * dz[i];
        }
        for r in 0..me {
            y[r] += alpha * dy[r];
        }

        // A step accepted purely through the filter (not Armijo) makes the
        // current iterate a filter entry.
        if accepted && !used_armijo && theta_k > T::zero() {
            filter.push((theta_k, phi_k));
        }

        let gx_new = eval_gradient(grad_fn, &x_t, n, iter)?;
        grad_evals += 1;

        // Damped BFGS on the Lagrangian Hessian when no exact Hessian is
        // available (same Powell damping as the SQP solver).
        if lagrangian_hessian_fn.is_none() {
            let lambda = user_multipliers(&y, &z, constraints);
            let yv = super::qp_active_set::lagrangian_gradient_difference_checked(
                &gx_new,
                &gx,
                &lambda,
                &x_t,
                &x,
                constraints,
                iter,
            )?;
            let sv: Vec<T> = (0..n).map(|j| x_t[j] - x[j]).collect();
            let bs = bfgs.matvec(&sv);
            let sbs = T::vec_dot(&sv, &bs);
            let sy = T::vec_dot(&sv, &yv);
            let theta_damp = if sy >= cst::<T>(0.2) * sbs {
                T::one()
            } else if sbs > T::machine_epsilon() {
                cst::<T>(0.8) * sbs / (sbs - sy)
            } else {
                T::one()
            };
            let r: Vec<T> = (0..n)
                .map(|j| theta_damp * yv[j] + (T::one() - theta_damp) * bs[j])
                .collect();
            let sr = T::vec_dot(&sv, &r);
            if sr.abs() > T::machine_epsilon() && sbs.abs() > T::machine_epsilon() {
                let rho_r = T::one() / sr;
                let rho_s = T::one() / sbs;
                for a in 0..n {
                    for b in 0..=a {
                        let upd = rho_r * r[a] * r[b] - rho_s * bs[a] * bs[b];
                        bfgs[[a, b]] += upd;
                        if b != a {
                            bfgs[[b, a]] += upd;
                        }
                    }
                }
            }
        }

        // Monotone Fiacco-McCormick barrier update, gated on the barrier
        // KKT error (IPOPT's kappa_eps rule).
        if barrier_err <= config.barrier_tolerance_factor * mu {
            let cand = T::min(config.mu_decrease * mu, mu.powf(config.mu_superlinear_exp));
            mu = T::max(config.mu_min, cand);
        }

        x = x_t;
        s = s_t;
        fx = fx_t;
        gx = gx_new;
        iterations = iter + 1;

        if iter == config.max_iterations - 1 {
            break 'outer;
        }
    }

    let active_inequality_rows: Vec<usize> = s
        .iter()
        .enumerate()
        .filter(|(_, value)| **value <= T::sqrt_epsilon() * T::one().max(value.abs()))
        .map(|(index, _)| index)
        .collect();
    let active_bound_rows: Vec<usize> = s
        .iter()
        .skip(mi_user)
        .enumerate()
        .filter(|(_, value)| **value <= T::sqrt_epsilon() * T::one().max(value.abs()))
        .map(|(index, _)| index)
        .collect();
    let diagnostics = IpmDiagnostics {
        iterations,
        barrier_parameter: mu,
        dual_residual: dual_infeas,
        primal_residual: primal_infeas,
        complementarity,
        restoration_attempts,
        restoration_successes,
        active_inequalities: active_inequality_rows.len(),
        active_bounds: active_bound_rows.len(),
        active_inequality_rows,
        active_bound_rows,
        line_search_backtracks,
        restoration_failures,
    };
    let native_multipliers =
        native_kkt_multipliers(constraints, &config.bounds, &y, &z, &bound_rows)?;
    Ok((
        OptimResult {
            x,
            fun: fx,
            grad: Some(gx),
            iterations,
            fun_evals,
            grad_evals,
            hess_evals,
            grad_norm: dual_infeas,
            converged,
            status,
            kkt: None,
        },
        diagnostics,
        native_multipliers,
    ))
}

/// Equality-constraint values and Jacobian (rows ordered like `eq_cons`).
fn eval_eq_rows<T: RealScalar>(
    eq_cons: &[&Constraint<T>],
    x: &[T],
    me: usize,
    n: usize,
    iteration: usize,
) -> Result<(Vec<T>, sonium_solvers_core::dense::DenseMatrix<T>), OptimError> {
    let mut cv = Vec::with_capacity(me);
    let mut jac = sonium_solvers_core::dense::DenseMatrix::zeros(me, n);
    let mut row = 0;
    for (index, c) in eq_cons.iter().enumerate() {
        let v = super::qp_active_set::checked_constraint_values(c, x, index, iteration)?;
        let j = super::qp_active_set::checked_constraint_jacobian(c, x, n, index, iteration)?;
        for k in 0..c.dim {
            cv.push(v[k]);
            for jj in 0..n {
                jac[[row, jj]] = j[[k, jj]];
            }
            row += 1;
        }
    }
    Ok((cv, jac))
}

/// Inequality-constraint values and Jacobian: user rows (c(x) >= 0 form)
/// followed by the bound rows `(var, bound_value, is_lower)`.
fn eval_ineq_rows<T: RealScalar>(
    iq_cons: &[&Constraint<T>],
    bound_rows: &[(usize, T, bool)],
    x: &[T],
    mi: usize,
    n: usize,
    iteration: usize,
) -> Result<(Vec<T>, sonium_solvers_core::dense::DenseMatrix<T>), OptimError> {
    let mut cv = Vec::with_capacity(mi);
    let mut jac = sonium_solvers_core::dense::DenseMatrix::zeros(mi, n);
    let mut row = 0;
    for (index, c) in iq_cons.iter().enumerate() {
        let v = super::qp_active_set::checked_constraint_values(c, x, index, iteration)?;
        let j = super::qp_active_set::checked_constraint_jacobian(c, x, n, index, iteration)?;
        for k in 0..c.dim {
            cv.push(v[k]);
            for jj in 0..n {
                jac[[row, jj]] = j[[k, jj]];
            }
            row += 1;
        }
    }
    for &(j, val, is_lower) in bound_rows {
        cv.push(if is_lower { x[j] - val } else { val - x[j] });
        jac[[row, j]] = if is_lower { T::one() } else { -T::one() };
        row += 1;
    }
    Ok((cv, jac))
}

/// L1 constraint violation: |c| for equalities, max(-c, 0) for
/// inequalities.
fn l1_violation<T: RealScalar>(
    constraints: &[Constraint<T>],
    x: &[T],
    iteration: usize,
) -> Result<T, OptimError> {
    let mut theta = T::zero();
    for (index, c) in constraints.iter().enumerate() {
        let v = super::qp_active_set::checked_constraint_values(c, x, index, iteration)?;
        for &ci in v.iter() {
            theta += match c.kind {
                ConstraintKind::Equality => ci.abs(),
                ConstraintKind::Inequality => T::max(-ci, T::zero()),
            };
        }
    }
    Ok(theta)
}

/// Feasibility restoration: minimise the constraint violation with
/// damped Gauss-Newton steps on the active residual `r(x) = [c_eq(x);
/// min(c_ineq(x), 0)]` (Levenberg-regularised normal equations,
/// `(JᵀJ + λI) d = -Jᵀr`, and an Armijo backtrack on the L1 violation
/// `theta`). Runs at most `config.max_restoration_iterations` steps.
///
/// Returns `Some((x, s, z, f(x)))` — the restored primal point with
/// rebuilt slacks and dual variables `z = mu/s` — when `theta` was
/// reduced at least by half (or below `primal_tolerance`); `None` when
/// the violation cannot be meaningfully reduced (likely infeasible
/// problem).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn feasibility_restoration<T, F>(
    f: &F,
    x_start: &[T],
    mu: T,
    eq_cons: &[&Constraint<T>],
    iq_cons: &[&Constraint<T>],
    bound_rows: &[(usize, T, bool)],
    constraints: &[Constraint<T>],
    config: &IpmConfig<T>,
    fun_evals: &mut usize,
) -> Result<Option<(Vec<T>, Vec<T>, Vec<T>, T)>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let n = x_start.len();
    let me: usize = eq_cons.iter().map(|c| c.dim).sum();
    let mi_user: usize = iq_cons.iter().map(|c| c.dim).sum();
    let mi = mi_user + bound_rows.len();

    let mut xr = x_start.to_vec();
    let mut th = l1_violation(constraints, &xr, 0)?;
    let th_start = th;

    if th > T::zero() {
        for _ in 0..config.max_restoration_iterations {
            let (ce, je) = eval_eq_rows(eq_cons, &xr, me, n, 0)?;
            let (ci, ji) = eval_ineq_rows(iq_cons, bound_rows, &xr, mi, n, 0)?;

            // Normal equations of the active residual.
            let mut jtj: sonium_solvers_core::dense::DenseMatrix<T> =
                sonium_solvers_core::dense::DenseMatrix::zeros(n, n);
            let mut jtr = vec![T::zero(); n];
            let mut accumulate =
                |jac: &sonium_solvers_core::dense::DenseMatrix<T>, r: usize, resid: T| {
                    for a in 0..n {
                        let ja = jac[[r, a]];
                        if ja == T::zero() {
                            continue;
                        }
                        jtr[a] += ja * resid;
                        for b in 0..=a {
                            let upd = ja * jac[[r, b]];
                            jtj[[a, b]] += upd;
                            if b != a {
                                jtj[[b, a]] += upd;
                            }
                        }
                    }
                };
            for (r, &cer) in ce[..me].iter().enumerate() {
                accumulate(&je, r, cer);
            }
            for (r, &cir) in ci[..mi].iter().enumerate() {
                if cir < T::zero() {
                    accumulate(&ji, r, cir);
                }
            }

            // Levenberg damping, then solve (JᵀJ + λI) d = -Jᵀr.
            let scale = (0..n).map(|i| jtj[[i, i]].abs()).fold(T::one(), T::max);
            for i in 0..n {
                jtj[[i, i]] += cst::<T>(1e-8) * scale;
            }
            let neg_jtr: Vec<T> = jtr.iter().map(|&v| -v).collect();
            let d = match solve_spd_regularized(&jtj, &neg_jtr) {
                Some(direction) => direction,
                None => return Ok(None),
            };

            // Armijo backtrack on theta.
            let mut alpha = T::one();
            let mut accepted = false;
            let mut th_t = th;
            for _ in 0..25 {
                if alpha <= cst(1e-14) {
                    break;
                }
                let x_t: Vec<T> = (0..n).map(|j| xr[j] + alpha * d[j]).collect();
                th_t = l1_violation(constraints, &x_t, 0)?;
                if th_t <= (T::one() - cst::<T>(1e-4) * alpha) * th {
                    xr = x_t;
                    accepted = true;
                    break;
                }
                alpha *= cst(0.5);
            }
            if !accepted {
                break;
            }
            th = th_t;
            if th <= config.primal_tolerance {
                break;
            }
        }
    }

    // Success: the violation was at least halved (or is tiny). A plateau
    // near th_start means the constraints are (numerically) inconsistent.
    if th > T::max(config.primal_tolerance, cst::<T>(0.5) * th_start) {
        return Ok(None);
    }

    let (ci, _) = eval_ineq_rows(iq_cons, bound_rows, &xr, mi, n, 0)?;
    let s: Vec<T> = ci.iter().map(|&v| T::max(v, cst(1e-4))).collect();
    let z: Vec<T> = s.iter().map(|&si| mu / si).collect();
    let fx = eval_objective(f, &xr, 0)?;
    *fun_evals += 1;
    Ok(Some((xr, s, z, fx)))
}

/// Multipliers in the order of the user-supplied `constraints` rows
/// (equalities from `y`, inequalities from `z`; bound rows excluded).
fn user_multipliers<T: RealScalar>(y: &[T], z: &[T], constraints: &[Constraint<T>]) -> Vec<T> {
    let mut lambda = Vec::new();
    let (mut iy, mut iz) = (0_usize, 0_usize);
    for c in constraints {
        for _ in 0..c.dim {
            let v = match c.kind {
                ConstraintKind::Equality => {
                    let v = y[iy];
                    iy += 1;
                    v
                }
                ConstraintKind::Inequality => {
                    let v = z[iz];
                    iz += 1;
                    v
                }
            };
            lambda.push(v);
        }
    }
    lambda
}

/// Fraction-to-the-boundary step length: the largest `alpha <= 1` keeping
/// `v + alpha * dv >= (1 - tau) * v` componentwise.
fn frac_to_boundary<T: RealScalar>(v: &[T], dv: &[T], tau: T) -> T {
    let mut alpha = T::one();
    for i in 0..v.len() {
        if dv[i] < T::zero() {
            let cand = -tau * v[i] / dv[i];
            if cand < alpha {
                alpha = cand;
            }
        }
    }
    alpha
}

/// Barrier objective `f(x) - mu * sum(log s)`.
fn barrier_objective<T: RealScalar>(fx: T, s: &[T], mu: T) -> T {
    let mut phi = fx;
    for &si in s {
        phi -= mu * si.ln();
    }
    phi
}

/// Fletcher-Leyffer filter acceptance of `(theta, phi)` against all entries.
fn filter_acceptable<T: RealScalar>(
    filter: &[(T, T)],
    theta: T,
    phi: T,
    gamma_theta: T,
    gamma_phi: T,
) -> bool {
    filter
        .iter()
        .all(|&(th, ph)| theta <= (T::one() - gamma_theta) * th || phi <= ph - gamma_phi * th)
}

/// Solve the condensed KKT system
///
/// ```text
/// [ K   -Ae^T ] [dx]   [rhs1]
/// [ -Ae   0   ] [dy] = [rhs2]
/// ```
///
/// via GMW modified Cholesky on `K` (the inertia correction) plus the
/// dense Schur complement `Ae K^-1 Ae^T` for the equality block. Returns
/// `None` only on hard numerical breakdown (non-finite factorisation).
fn solve_condensed<T: RealScalar>(
    kmat: &sonium_solvers_core::dense::DenseMatrix<T>,
    je: &sonium_solvers_core::dense::DenseMatrix<T>,
    rhs1: &[T],
    rhs2: &[T],
    n: usize,
    me: usize,
) -> Option<(Vec<T>, Vec<T>)> {
    // `kmat` is the (n+me) x (n+me) condensed KKT matrix by construction.
    let (lk, _) = modified_cholesky(kmat)?;
    let solve_k = |b: &[T]| -> Vec<T> {
        let yv = solve_lower_triangular(&lk, b);
        solve_lower_triangular_transpose(&lk, &yv)
    };

    if me == 0 {
        let dx = solve_k(rhs1);
        return if dx.iter().all(|&v| v.is_finite()) {
            Some((dx, Vec::new()))
        } else {
            None
        };
    }

    let u = solve_k(rhs1);

    // V = K^-1 Ae^T column by column; S = Ae V.
    let mut s_mat = sonium_solvers_core::dense::DenseMatrix::zeros(me, me);
    let mut v_cols: Vec<Vec<T>> = Vec::with_capacity(me);
    for c in 0..me {
        let col: Vec<T> = (0..n).map(|j| je[[c, j]]).collect();
        let v = solve_k(&col);
        for r in 0..=c {
            let mut val = T::zero();
            for j in 0..n {
                val += je[[r, j]] * v[j];
            }
            s_mat[[r, c]] = val;
            s_mat[[c, r]] = val;
        }
        v_cols.push(v);
    }

    // S dy = -rhs2 - Ae u, with escalating regularisation if Ae is
    // (numerically) rank-deficient.
    let rhs_s: Vec<T> = (0..me)
        .map(|r| {
            let mut v = -rhs2[r];
            for j in 0..n {
                v -= je[[r, j]] * u[j];
            }
            v
        })
        .collect();
    let dy = solve_spd_regularized(&s_mat, &rhs_s)?;

    let mut dx = u;
    for c in 0..me {
        for j in 0..n {
            dx[j] += v_cols[c][j] * dy[c];
        }
    }
    if dx.iter().all(|&v| v.is_finite()) && dy.iter().all(|&v| v.is_finite()) {
        Some((dx, dy))
    } else {
        None
    }
}

/// Cholesky solve with escalating diagonal regularisation.
fn solve_spd_regularized<T: RealScalar>(
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
    b: &[T],
) -> Option<Vec<T>> {
    let n = a.nrows();
    let scale = (0..n).map(|i| a[[i, i]].abs()).fold(T::one(), T::max);
    let mut reg = T::zero();
    for _ in 0..8 {
        let mut m = a.clone();
        if reg > T::zero() {
            for i in 0..n {
                m[[i, i]] += reg;
            }
        }
        if let Some(l) = cholesky(&m) {
            let yv = solve_lower_triangular(&l, b);
            let sol = solve_lower_triangular_transpose(&l, &yv);
            if sol.iter().all(|&v| v.is_finite()) {
                return Some(sol);
            }
        }
        reg = if reg == T::zero() {
            cst::<T>(1e-12) * scale
        } else {
            reg * cst(10.0)
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn native_kkt_multipliers_preserve_mixed_constraint_and_bound_order() {
        let constraints: Vec<Constraint<f64>> = vec![
            Constraint {
                kind: ConstraintKind::Inequality,
                dim: 2,
                fun: Box::new(|_| vec![0.0, 0.0]),
                jac: None,
            },
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|_| vec![0.0]),
                jac: None,
            },
            Constraint {
                kind: ConstraintKind::Inequality,
                dim: 1,
                fun: Box::new(|_| vec![0.0]),
                jac: None,
            },
        ];
        let bounds = vec![Bound::both(-1.0, 2.0), Bound::lower(0.0)];
        let bound_rows = vec![(1, 0.0, true), (0, 2.0, false)];

        let multipliers = native_kkt_multipliers(
            &constraints,
            &bounds,
            &[10.0],
            &[20.0, 21.0, 30.0, 40.0, 50.0],
            &bound_rows,
        )
        .expect("valid native multiplier layout");

        assert_eq!(multipliers.constraints, vec![20.0, 21.0, 10.0, 30.0]);
        assert_eq!(multipliers.lower_bounds, vec![0.0, 40.0]);
        assert_eq!(multipliers.upper_bounds, vec![50.0, 0.0]);
    }

    /// Circle problem: min x + y s.t. 1 - x^2 - y^2 >= 0.
    /// KKT point x = y = -1/sqrt(2), multiplier z = 1/sqrt(2).
    #[allow(clippy::type_complexity)]
    fn circle_problem() -> (
        impl Fn(&[f64]) -> f64,
        impl Fn(&[f64]) -> Vec<f64>,
        impl Fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>,
        Vec<Constraint<f64>>,
    ) {
        let f = |x: &[f64]| x[0] + x[1];
        let g = |_: &[f64]| vec![1.0, 1.0];
        // ∇²L = ∇²f - z ∇²c = 0 - z*(-2 I) = 2z I
        let h = |_: &[f64], lambda: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0 * lambda[0], 2.0 * lambda[0]])
        };
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![1.0 - x[0] * x[0] - x[1] * x[1]]),
            jac: Some(Box::new(|x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![-2.0 * x[0], -2.0 * x[1]],
                )
            })),
        }];
        (f, g, h, constraints)
    }

    #[test]
    fn test_ipm_circle_kkt() {
        let (f, g, h, constraints) = circle_problem();
        let config = IpmConfig::default();
        let result = ipm(&f, &g, Some(&h), &[1.0, 0.0], &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let ipm_state = result
            .kkt
            .as_ref()
            .and_then(|report| report.ipm.as_ref())
            .expect("IPM result must expose final barrier diagnostics");
        assert!(ipm_state.barrier_parameter.is_finite());
        assert!(ipm_state.dual_residual.is_finite());
        assert!(ipm_state.primal_residual.is_finite());
        assert!(ipm_state.complementarity.is_finite());
        assert_eq!(ipm_state.iterations, result.iterations);
        assert_eq!(
            ipm_state.active_inequalities,
            ipm_state.active_inequality_rows.len()
        );
        assert_eq!(ipm_state.active_bounds, ipm_state.active_bound_rows.len());
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(result.x[0], -s, epsilon = 1e-7);
        assert_relative_eq!(result.x[1], -s, epsilon = 1e-7);
        assert_relative_eq!(result.fun, -std::f64::consts::SQRT_2, epsilon = 1e-7);
        let report = result.kkt.as_ref().expect("IPM KKT report");
        assert_eq!(report.multipliers.constraints.len(), 1);
        assert_relative_eq!(
            report.multipliers.constraints[0],
            std::f64::consts::FRAC_1_SQRT_2,
            epsilon = 1e-7
        );

        // Analytical KKT residuals at the reported point (< 1e-8).
        let c = 1.0 - result.x[0].powi(2) - result.x[1].powi(2);
        let lambda = s; // z* = 1/sqrt(2)
        let stat0 = 1.0 - lambda * (-2.0 * result.x[0]);
        let stat1 = 1.0 - lambda * (-2.0 * result.x[1]);
        assert!(c.abs() < 1e-8, "primal residual c = {c:e}");
        assert!(stat0.abs() < 1e-8, "stationarity[0] = {stat0:e}");
        assert!(stat1.abs() < 1e-8, "stationarity[1] = {stat1:e}");
        assert!(
            (c.abs() * lambda).abs() < 1e-8,
            "complementarity = {e:e}",
            e = c * lambda
        );
    }

    #[test]
    fn test_ipm_circle_bfgs_hessian() {
        // Same problem without an exact Hessian: damped BFGS path.
        let (f, g, _h, constraints) = circle_problem();
        let config = IpmConfig::default();
        let result = ipm(
            &f,
            &g,
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[1.0, 0.0],
            &constraints,
            &config,
        )
        .unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(result.x[0], -s, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], -s, epsilon = 1e-5);
    }

    #[test]
    fn test_ipm_equality_inequality_mix() {
        // min x^2 + y^2 + z^2  s.t.  x + y + z = 3,  x >= 1.5.
        // Optimum (1.5, 0.75, 0.75), f = 3.375.
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2) + x[2].powi(2);
        let g = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1], 2.0 * x[2]];
        let h = |_: &[f64], _: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0, 2.0, 2.0])
        };
        let constraints = vec![
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|x: &[f64]| vec![x[0] + x[1] + x[2] - 3.0]),
                jac: Some(Box::new(|_: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        3,
                        vec![1.0, 1.0, 1.0],
                    )
                })),
            },
            Constraint {
                kind: ConstraintKind::Inequality,
                dim: 1,
                fun: Box::new(|x: &[f64]| vec![x[0] - 1.5]),
                jac: Some(Box::new(|_: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        3,
                        vec![1.0, 0.0, 0.0],
                    )
                })),
            },
        ];
        let config = IpmConfig::default();
        let result = ipm(&f, &g, Some(&h), &[1.0, 1.0, 1.0], &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.5, epsilon = 1e-7);
        assert_relative_eq!(result.x[1], 0.75, epsilon = 1e-7);
        assert_relative_eq!(result.x[2], 0.75, epsilon = 1e-7);
        assert_relative_eq!(result.fun, 3.375, epsilon = 1e-7);
    }

    #[test]
    fn test_ipm_bounds_via_slacks() {
        // min (x - 3)^2 with bound x in [0, 1] (bounds become inequality
        // rows). Optimum x = 1.
        let f = |x: &[f64]| (x[0] - 3.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 3.0)];
        let h = |_: &[f64], _: &[f64]| sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0]);
        let config: IpmConfig<f64> = IpmConfig {
            bounds: vec![Bound::both(0.0, 1.0)],
            ..IpmConfig::default()
        };
        let result = ipm(&f, &g, Some(&h), &[0.5], &[], &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-6);
        let report = result.kkt.as_ref().expect("IPM KKT report");
        assert_eq!(report.multipliers.upper_bounds.len(), 1);
        assert_relative_eq!(report.multipliers.upper_bounds[0], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn ipm_rejects_invalid_configuration_before_callbacks() {
        let mut config = IpmConfig::<f64>::default();
        config.tau = f64::NAN;
        let error = ipm(
            &|_| panic!("objective must not run"),
            &|_| panic!("gradient must not run"),
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[0.0],
            &[],
            &config,
        )
        .expect_err("invalid IPM configuration must fail before callbacks");
        assert!(matches!(
            error,
            OptimError::NumericalError { iteration: 0, .. }
        ));
    }

    #[test]
    fn ipm_rejects_starting_point_outside_bounds_before_callbacks() {
        let config = IpmConfig {
            bounds: vec![Bound::both(0.0, 1.0)],
            ..IpmConfig::default()
        };
        let error = ipm(
            &|_| panic!("objective must not run"),
            &|_| panic!("gradient must not run"),
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[2.0],
            &[],
            &config,
        )
        .expect_err("incompatible starting point must be rejected");
        assert!(matches!(
            error,
            OptimError::IncompatibleStartingPoint { index: 0, .. }
        ));
    }

    #[test]
    fn ipm_converts_callback_panics_to_typed_errors() {
        let config = IpmConfig::default();
        let error = ipm(
            &|_| panic!("objective callback failed"),
            &|_| vec![0.0],
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[0.0],
            &[],
            &config,
        )
        .expect_err("objective callback panic must not escape IPM");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));

        let error = ipm(
            &|_| 0.0,
            &|_| panic!("gradient callback failed"),
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[0.0],
            &[],
            &config,
        )
        .expect_err("gradient callback panic must not escape IPM");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));

        let constraint = Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|_| panic!("constraint callback failed")),
            jac: Some(Box::new(|_| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
            })),
        };
        let error = ipm(
            &|_| 0.0,
            &|_| vec![0.0],
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[0.0],
            &[constraint],
            &config,
        )
        .expect_err("constraint callback panic must not escape IPM");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));

        let hessian = |_: &[f64], _: &[f64]| {
            panic!("Hessian callback failed");
        };
        let error = ipm(
            &|_| 0.0,
            &|_| vec![1.0],
            Some(&hessian),
            &[0.0],
            &[],
            &IpmConfig {
                max_iterations: 1,
                ..config
            },
        )
        .expect_err("Hessian callback panic must not escape IPM");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));
    }

    #[test]
    fn ipm_revalidates_callback_outputs_after_preflight() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let config = IpmConfig::<f64> {
            max_iterations: 1,
            ..IpmConfig::default()
        };

        let objective_calls = AtomicUsize::new(0);
        let objective = |x: &[f64]| {
            if objective_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                x[0] * x[0]
            } else {
                f64::NAN
            }
        };
        let error = ipm(
            &objective,
            &|_| vec![1.0],
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[1.0],
            &[],
            &config,
        )
        .expect_err("post-preflight objective values must remain checked");
        assert!(matches!(error, OptimError::NumericalError { .. }));

        let gradient_calls = AtomicUsize::new(0);
        let gradient = |_: &[f64]| {
            if gradient_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                vec![1.0]
            } else {
                Vec::new()
            }
        };
        let error = ipm(
            &|x: &[f64]| x[0] * x[0],
            &gradient,
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[1.0],
            &[],
            &config,
        )
        .expect_err("post-preflight gradient dimensions must remain checked");
        assert!(matches!(
            error,
            OptimError::DimensionMismatch {
                expected: 1,
                got: 0
            }
        ));

        let value_calls = Arc::new(AtomicUsize::new(0));
        let value_calls_for_constraint = Arc::clone(&value_calls);
        let constraint = Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(move |x: &[f64]| {
                if value_calls_for_constraint.fetch_add(1, Ordering::SeqCst) == 0 {
                    vec![x[0]]
                } else {
                    Vec::new()
                }
            }),
            jac: Some(Box::new(|_| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
            })),
        };
        let error = ipm(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[1.0],
            &[constraint],
            &config,
        )
        .expect_err("post-preflight constraint dimensions must remain checked");
        assert!(matches!(
            error,
            OptimError::InvalidConstraintDimension {
                index: 0,
                expected: 1,
                got: 0
            }
        ));

        let jacobian_calls = Arc::new(AtomicUsize::new(0));
        let jacobian_calls_for_constraint = Arc::clone(&jacobian_calls);
        let constraint = Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0]]),
            jac: Some(Box::new(move |_| {
                if jacobian_calls_for_constraint.fetch_add(1, Ordering::SeqCst) == 0 {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
                } else {
                    sonium_solvers_core::dense::DenseMatrix::zeros(0, 0)
                }
            })),
        };
        let error = ipm(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            None::<&fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>>,
            &[1.0],
            &[constraint],
            &config,
        )
        .expect_err("post-preflight constraint Jacobian dimensions must remain checked");
        assert!(matches!(
            error,
            OptimError::InvalidConstraintJacobian {
                index: 0,
                expected_rows: 1,
                expected_cols: 1,
                got_rows: 0,
                got_cols: 0
            }
        ));

        let hessian_calls = AtomicUsize::new(0);
        let hessian = |_: &[f64], _: &[f64]| {
            if hessian_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0])
            } else {
                sonium_solvers_core::dense::DenseMatrix::zeros(0, 0)
            }
        };
        let error = ipm(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            Some(&hessian),
            &[1.0],
            &[],
            &config,
        )
        .expect_err("post-preflight Hessian dimensions must remain checked");
        assert!(matches!(
            error,
            OptimError::DimensionMismatch {
                expected: 1,
                got: 0
            }
        ));
    }

    /// Box projection with many inequalities: min ||x - c||^2 s.t. x >= a.
    /// Solution x* = max(c, a); half the constraints are active. The bound
    /// a = 1.05 deliberately avoids any c_i == a_i: at such a degenerate
    /// component strict complementarity fails (s* = z* = 0) and any IPM
    /// only reaches O(sqrt(mu)) accuracy there.
    #[allow(clippy::type_complexity)]
    fn box_projection(
        n: usize,
    ) -> (
        impl Fn(&[f64]) -> f64,
        impl Fn(&[f64]) -> Vec<f64>,
        impl Fn(&[f64], &[f64]) -> sonium_solvers_core::dense::DenseMatrix<f64>,
        Vec<Constraint<f64>>,
        Vec<f64>,
    ) {
        let c: Vec<f64> = (0..n).map(|i| 0.1 * (i + 1) as f64).collect();
        let a = vec![1.05_f64; n];
        let x_star: Vec<f64> = c.iter().zip(&a).map(|(&ci, &ai)| ci.max(ai)).collect();
        let cf = c.clone();
        let f = move |x: &[f64]| x.iter().zip(&cf).map(|(&xi, &ci)| (xi - ci).powi(2)).sum();
        let cg = c.clone();
        let g = move |x: &[f64]| {
            x.iter()
                .zip(&cg)
                .map(|(&xi, &ci)| 2.0 * (xi - ci))
                .collect()
        };
        let h = move |_: &[f64], _: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&vec![2.0; n])
        };
        let ac = a.clone();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: n,
            fun: Box::new(move |x: &[f64]| x.iter().zip(&ac).map(|(&xi, &ai)| xi - ai).collect()),
            jac: Some(Box::new(move |_: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::eye(n)
            })),
        }];
        (f, g, h, constraints, x_star)
    }

    #[test]
    fn test_ipm_many_inequalities_box_projection() {
        let n = 20;
        let (f, g, h, constraints, x_star) = box_projection(n);
        let x0: Vec<f64> = x_star.iter().map(|&v| v + 0.3).collect();
        let config = IpmConfig::default();
        let result = ipm(&f, &g, Some(&h), &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        for j in 0..n {
            assert_relative_eq!(result.x[j], x_star[j], epsilon = 1e-6);
        }
        // Large active set: 10 of 20 constraints active at the solution.
        let active = result
            .x
            .iter()
            .filter(|&&xj| (xj - 1.05).abs() < 1e-5)
            .count();
        assert_eq!(active, 10, "expected 10 active constraints");
    }

    #[test]
    fn test_ipm_vs_sqp_many_inequalities() {
        // Comparison on the box-projection instance: IPM should reach the
        // solution in few primal-dual steps regardless of the active-set
        // size, while the active-set SQP pays per QP iteration.
        let n = 20;
        let (f, g, h, constraints, x_star) = box_projection(n);
        let x0: Vec<f64> = x_star.iter().map(|&v| v + 0.3).collect();

        let config = IpmConfig::default();
        let ipm_result = ipm(&f, &g, Some(&h), &x0, &constraints, &config).unwrap();
        assert!(ipm_result.converged);

        let sqp_config = super::super::sqp_impl::SqpConfig::default();
        let sqp_result =
            super::super::sqp_impl::sqp(&f, &g, &x0, &constraints, &sqp_config).unwrap();

        println!(
            "box projection n={n}: IPM iters={} fun_evals={} grad_evals={} | \
             SQP iters={} fun_evals={} grad_evals={} converged={}",
            ipm_result.iterations,
            ipm_result.fun_evals,
            ipm_result.grad_evals,
            sqp_result.iterations,
            sqp_result.fun_evals,
            sqp_result.grad_evals,
            sqp_result.converged,
        );
        assert!(
            ipm_result.iterations <= 40,
            "IPM should converge in a few dozen iterations: {}",
            ipm_result.iterations
        );
        for j in 0..n {
            assert_relative_eq!(ipm_result.x[j], x_star[j], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ipm_deterministic() {
        let (f, g, h, constraints) = circle_problem();
        let config = IpmConfig::default();
        let r1 = ipm(&f, &g, Some(&h), &[1.0, 0.0], &constraints, &config).unwrap();
        let r2 = ipm(&f, &g, Some(&h), &[1.0, 0.0], &constraints, &config).unwrap();
        assert_eq!(r1.x, r2.x, "IPM must be deterministic");
        assert_eq!(r1.fun, r2.fun);
        assert_eq!(r1.iterations, r2.iterations);
    }

    #[test]
    fn test_ipm_restoration_infeasible_start() {
        // Far infeasible start on two intersecting curved equalities:
        //   min (x - 1)^2 + (y - 1)^2
        //   s.t.  x^2 + y^2 = 2,  (x - 2)^2 + y^2 = 2   →  x = 1, y = ±1
        // from x0 = (8, -6) (theta(x0) = 100).
        //
        // With the restoration phase disabled (max_restoration_iterations
        // = 0) the filter line search stalls on the first iteration and
        // the run fails; with the default budget the Gauss-Newton
        // restoration drives theta down, the barrier loop re-enters, and
        // the run converges to the near intersection point (1, -1).
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 1.0)];
        // ∇²L = ∇²f − y1 ∇²c1 − y2 ∇²c2 = (2 − 2(y1 + y2)) I.
        let h = |_: &[f64], l: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[
                2.0 - 2.0 * (l[0] + l[1]),
                2.0 - 2.0 * (l[0] + l[1]),
            ])
        };
        let mk_constraints = || {
            vec![
                Constraint {
                    kind: ConstraintKind::Equality,
                    dim: 1,
                    fun: Box::new(|x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 2.0]),
                    jac: Some(Box::new(|x: &[f64]| {
                        sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                            1,
                            2,
                            vec![2.0 * x[0], 2.0 * x[1]],
                        )
                    })),
                },
                Constraint {
                    kind: ConstraintKind::Equality,
                    dim: 1,
                    fun: Box::new(|x: &[f64]| vec![(x[0] - 2.0).powi(2) + x[1] * x[1] - 2.0]),
                    jac: Some(Box::new(|x: &[f64]| {
                        sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                            1,
                            2,
                            vec![2.0 * (x[0] - 2.0), 2.0 * x[1]],
                        )
                    })),
                },
            ]
        };

        let no_restoration = IpmConfig {
            max_restoration_iterations: 0,
            ..IpmConfig::default()
        };
        let failed = ipm(
            &f,
            &g,
            Some(&h),
            &[8.0, -6.0],
            &mk_constraints(),
            &no_restoration,
        )
        .unwrap();
        println!(
            "infeasible start without restoration: status={:?} iters={}",
            failed.status, failed.iterations
        );
        assert!(
            !failed.converged,
            "without restoration this start should fail (got {:?})",
            failed.status
        );

        let config = IpmConfig::default();
        let result = ipm(&f, &g, Some(&h), &[8.0, -6.0], &mk_constraints(), &config).unwrap();
        println!(
            "infeasible start with restoration: status={:?} iters={} fun_evals={}",
            result.status, result.iterations, result.fun_evals
        );
        assert!(result.converged, "status: {:?}", result.status);
        let ipm_state = result
            .kkt
            .as_ref()
            .and_then(|report| report.ipm.as_ref())
            .expect("restored IPM result must expose restoration diagnostics");
        assert!(ipm_state.restoration_attempts > 0);
        assert!(ipm_state.restoration_successes > 0);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-6);
        assert_relative_eq!(result.fun, 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_ipm_restoration_genuinely_infeasible() {
        // Inconsistent equalities: x + y = 1 and x + y = 2. The
        // restoration phase cannot reduce theta below 1, so the run must
        // fail cleanly with RestorationFailed at a finite point (not
        // garbage, not a hang).
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let g = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let h =
            |_: &[f64], _: &[f64]| sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0, 2.0]);
        let constraints = vec![
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|x: &[f64]| vec![x[0] + x[1] - 1.0]),
                jac: Some(Box::new(|_: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        2,
                        vec![1.0, 1.0],
                    )
                })),
            },
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|x: &[f64]| vec![x[0] + x[1] - 2.0]),
                jac: Some(Box::new(|_: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        2,
                        vec![1.0, 1.0],
                    )
                })),
            },
        ];
        let config = IpmConfig::default();
        let result = ipm(&f, &g, Some(&h), &[0.5, 0.5], &constraints, &config).unwrap();
        println!(
            "infeasible problem: status={:?} iters={} x={:?}",
            result.status, result.iterations, result.x
        );
        assert_eq!(result.status, OptimStatus::RestorationFailed);
        assert!(!result.converged);
        assert!(result.x.iter().all(|&v| v.is_finite()));
        assert!(result.fun.is_finite());
    }
}
