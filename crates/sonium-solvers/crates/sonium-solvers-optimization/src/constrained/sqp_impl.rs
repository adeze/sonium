//! SQP: Sequential Quadratic Programming with second-order corrections.
//!
//! At each iteration a QP subproblem (equalities and inequalities) is solved
//! for the search direction with the Goldfarb–Idnani dual active-set method
//! (`super::qp_active_set`). Globalisation is either an L1 exact merit
//! function (default, [`SqpAcceptance::Merit`]) or a Fletcher–Leyffer filter
//! ([`SqpAcceptance::Filter`]); the merit penalty follows the
//! multiplier-based rule of Nocedal & Wright (§18.5): the L1 merit is exact
//! once `mu >= ||lambda||_inf`, so `mu` tracks the QP multipliers with a
//! small margin. Second-order corrections mitigate the Maratos effect under
//! both acceptance rules. Constraints without an analytic Jacobian fall back
//! to central finite differences.
//!
//! ## Filter acceptance
//!
//! With [`SqpAcceptance::Filter`] the merit function and its penalty
//! parameter are dropped entirely. The filter collects `(theta, f)` pairs
//! (L1 constraint violation, objective); a trial point is accepted when it
//! is not dominated by any filter entry, with the standard margin
//! `theta <= (1 - gamma_theta) * theta_entry` or
//! `f <= f_entry - gamma_f * theta_entry`. The f-type / theta-type step
//! distinction follows Wächter–Biegler (2006): when the switching condition
//! holds (`alpha * gᵀd` sufficiently negative relative to `theta`), the
//! trial must satisfy an Armijo decrease on `f` and the filter is *not*
//! augmented (f-type step); otherwise acceptance is purely by the filter
//! and the current iterate becomes a filter entry (theta-type step). When
//! the backtracking loop fails outright, a simple feasibility restoration
//! (projected-gradient steps on the violation, [`feasibility_restoration`])
//! is tried before giving up.

use crate::types::{
    Bound, Constraint, ConstraintKind, OptimError, OptimResult, OptimStatus, RealScalar,
    validate_bounds,
};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

/// Globalisation strategy for [`sqp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqpAcceptance {
    /// L1 exact merit function with multiplier-based penalty update
    /// (Nocedal & Wright §18.5).
    #[default]
    Merit,
    /// Fletcher–Leyffer filter on `(constraint violation, objective)` —
    /// no penalty parameter to tune, at the cost of needing a feasibility
    /// restoration fallback when the line search fails.
    Filter,
}

/// Configuration for SQP.
#[derive(Debug, Clone)]
pub struct SqpConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub constraint_tolerance: T,
    /// Use second-order correction steps (Maratos effect mitigation).
    pub second_order_correction: bool,
    /// Acceptance rule: L1 merit (default) or Fletcher–Leyffer filter.
    pub acceptance: SqpAcceptance,
    /// Use the multiplier-based L1 merit penalty update (Nocedal & Wright,
    /// §18.5): after each QP solve, `mu = max(mu, ||lambda||_inf * (1 +
    /// penalty_margin))` over the constraint multipliers. This is the
    /// theoretically sound exactness condition and keeps the penalty
    /// bounded; when `false` (or when the QP returns no multipliers, e.g.
    /// an infeasible linearisation) the legacy predicted-reduction
    /// heuristic is used as a fallback. Only used with
    /// [`SqpAcceptance::Merit`].
    pub use_multiplier_penalty: bool,
    /// Relative margin for the multiplier-based penalty update.
    pub penalty_margin: T,
    /// Filter margin on the constraint violation (`gamma_theta`, 1e-5).
    /// Only used with [`SqpAcceptance::Filter`].
    pub filter_gamma_theta: T,
    /// Filter margin on the objective (`gamma_f`, 1e-5). Only used with
    /// [`SqpAcceptance::Filter`].
    pub filter_gamma_f: T,
    /// Switching-condition constant (`delta`, 1.0): the step is treated as
    /// f-type when `-alpha * gᵀd > delta * theta^s_theta`. Only used with
    /// [`SqpAcceptance::Filter`].
    pub filter_delta: T,
    /// Switching-condition exponent (`s_theta`, 1.5). Only used with
    /// [`SqpAcceptance::Filter`].
    pub filter_s_theta: T,
    /// Armijo sufficient-decrease constant for f-type filter steps (1e-4).
    /// Only used with [`SqpAcceptance::Filter`].
    pub filter_armijo_eta: T,
    pub bounds: Vec<Bound<T>>,
}

impl Default for SqpConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-6,
            constraint_tolerance: 1e-6,
            second_order_correction: true,
            acceptance: SqpAcceptance::Merit,
            use_multiplier_penalty: true,
            penalty_margin: 0.1,
            filter_gamma_theta: 1e-5,
            filter_gamma_f: 1e-5,
            filter_delta: 1.0,
            filter_s_theta: 1.5,
            filter_armijo_eta: 1e-4,
            bounds: Vec::new(),
        }
    }
}

impl Default for SqpConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-4,
            constraint_tolerance: 1e-4,
            second_order_correction: true,
            acceptance: SqpAcceptance::Merit,
            use_multiplier_penalty: true,
            penalty_margin: 0.1,
            filter_gamma_theta: 1e-3,
            filter_gamma_f: 1e-3,
            filter_delta: 1.0,
            filter_s_theta: 1.5,
            filter_armijo_eta: 1e-4,
            bounds: Vec::new(),
        }
    }
}

/// Minimise `f(x)` subject to constraints using SQP.
pub fn sqp<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &SqpConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    validate_bounds(&config.bounds, x0.len())?;
    if let Some(index) = x0.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("initial point contains a non-finite value at index {index}"),
        });
    }

    // SQP has several legacy helper paths that invoke the objective and
    // gradient directly (line search, restoration, and BFGS updates). Guard
    // the callbacks once at this public boundary so a panic or invalid value
    // cannot escape as an optimizer panic or be turned into a successful
    // result after a later backtracking step happens to recover.
    let callback_failure = Arc::new(Mutex::new(None));
    let safe_f = {
        let callback_failure = Arc::clone(&callback_failure);
        move |x: &[T]| match catch_unwind(AssertUnwindSafe(|| f(x))) {
            Ok(value) if value.is_finite() => value,
            Ok(_) => {
                record_callback_failure(
                    &callback_failure,
                    OptimError::NumericalError {
                        iteration: 0,
                        message: "objective callback returned a non-finite value".to_string(),
                    },
                );
                T::nan()
            }
            Err(_) => {
                record_callback_failure(
                    &callback_failure,
                    OptimError::CallbackPanicked {
                        iteration: 0,
                        stage: "objective".to_string(),
                    },
                );
                T::nan()
            }
        }
    };
    let safe_grad = {
        let callback_failure = Arc::clone(&callback_failure);
        move |x: &[T]| match catch_unwind(AssertUnwindSafe(|| grad_fn(x))) {
            Ok(gradient) if gradient.len() != x.len() => {
                record_callback_failure(
                    &callback_failure,
                    OptimError::DimensionMismatch {
                        expected: x.len(),
                        got: gradient.len(),
                    },
                );
                vec![T::nan(); x.len()]
            }
            Ok(gradient) if gradient.iter().any(|value| !value.is_finite()) => {
                record_callback_failure(
                    &callback_failure,
                    OptimError::NumericalError {
                        iteration: 0,
                        message: "gradient callback returned a non-finite value".to_string(),
                    },
                );
                vec![T::nan(); x.len()]
            }
            Ok(gradient) => gradient,
            Err(_) => {
                record_callback_failure(
                    &callback_failure,
                    OptimError::CallbackPanicked {
                        iteration: 0,
                        stage: "gradient".to_string(),
                    },
                );
                vec![T::nan(); x.len()]
            }
        }
    };

    let mut result = match catch_unwind(AssertUnwindSafe(|| {
        sqp_impl(&safe_f, &safe_grad, x0, constraints, config)
    })) {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => return Err(take_callback_failure(&callback_failure).unwrap_or(error)),
        Err(_) => {
            return Err(take_callback_failure(&callback_failure).unwrap_or(
                OptimError::CallbackPanicked {
                    iteration: 0,
                    stage: "constraint or SQP callback".to_string(),
                },
            ));
        }
    };
    if let Some(error) = take_callback_failure(&callback_failure) {
        return Err(error);
    }
    validate_sqp_result(&result, x0.len())?;
    attach_final_kkt(&mut result, constraints, &config.bounds)?;
    result.gate_kkt(
        config.grad_tolerance.max(config.constraint_tolerance) * T::from_f64(100.0).unwrap(),
    );
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn record_callback_failure(failure: &Arc<Mutex<Option<OptimError>>>, error: OptimError) {
    if let Ok(mut slot) = failure.lock()
        && slot.is_none()
    {
        *slot = Some(error);
    }
}

fn take_callback_failure(failure: &Arc<Mutex<Option<OptimError>>>) -> Option<OptimError> {
    failure.lock().ok().and_then(|mut slot| slot.take())
}

fn validate_sqp_result<T: RealScalar>(
    result: &OptimResult<T>,
    expected_dimension: usize,
) -> Result<(), OptimError> {
    if result.x.len() != expected_dimension {
        return Err(OptimError::DimensionMismatch {
            expected: expected_dimension,
            got: result.x.len(),
        });
    }
    if result.x.iter().any(|value| !value.is_finite())
        || !result.fun.is_finite()
        || !result.grad_norm.is_finite()
    {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: "SQP produced a non-finite iterate or objective state".to_string(),
        });
    }
    if let Some(gradient) = result.grad.as_deref() {
        if gradient.len() != expected_dimension {
            return Err(OptimError::DimensionMismatch {
                expected: expected_dimension,
                got: gradient.len(),
            });
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(OptimError::NumericalError {
                iteration: result.iterations,
                message: "SQP produced a non-finite gradient state".to_string(),
            });
        }
    }
    Ok(())
}

fn sqp_impl<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &SqpConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut gx = grad_fn(&x);
    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;
    let half = T::from_f64(0.5).unwrap();

    // BFGS Hessian approximation
    let mut b = sonium_solvers_core::dense::DenseMatrix::eye(n);
    let mut penalty = T::from_f64(1.0).unwrap();
    // Fletcher–Leyffer filter of (theta, f) pairs (Filter acceptance only).
    let mut filter: Vec<(T, T)> = Vec::new();
    // Number of QP rows that correspond to user constraints (bound rows
    // follow them and are not penalised in the merit function).
    let n_constraint_rows: usize = constraints.iter().map(|c| c.dim).sum();

    for iter in 0..config.max_iterations {
        let max_violation = compute_max_violation(&x, constraints);

        // Compute projected gradient norm: project objective gradient onto null space
        // of equality constraint Jacobians.
        let grad_norm = projected_kkt_norm(&gx, &x, constraints, n);

        if grad_norm < config.grad_tolerance && max_violation < config.constraint_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(gx),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Solve QP subproblem for direction d: both the linearised
        // equalities and inequalities (and bounds) enter the direction via
        // the active-set QP solver.
        let qp = solve_sqp_direction(&gx, &b, &x, constraints, &config.bounds, n)?;
        let d = qp.d;

        // Direction-based KKT check: at a constrained optimum with active
        // inequalities the equality-projected gradient does not vanish, but
        // the QP direction does.
        let d_norm = d.iter().map(|&di| di.abs()).fold(T::zero(), T::max);
        if d_norm < config.grad_tolerance && max_violation < config.constraint_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(gx),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm: d_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Directional derivative of the objective along d, used by the
        // merit predicted reduction and by the filter switching condition.
        let dgd = T::vec_dot(&gx, &d);

        // Globalisation: L1 merit line search or Fletcher–Leyffer filter.
        // `restored` marks iterates produced by feasibility restoration
        // (they replace the QP step; SOC is skipped for them).
        let (alpha, mut x_trial, mut fx_trial, f_type_step, restored) = match config.acceptance {
            SqpAcceptance::Merit => {
                // Update merit penalty. Multiplier-based rule (Nocedal &
                // Wright, §18.5): the L1 merit is exact once
                // mu >= ||lambda||_inf, so track the QP constraint
                // multipliers with a small relative margin. This keeps the
                // penalty bounded (no geometric blow-up as the violation
                // shrinks) and avoids under-penalisation stalls. When the
                // QP did not terminate at an optimum (e.g. infeasible
                // linearisation) the multipliers are not meaningful: fall
                // back to the legacy predicted-reduction heuristic.
                let bd = b.dot(&d);
                let pred_reduction = -dgd - half * T::vec_dot(&d, &bd);
                let lambda_inf = if config.use_multiplier_penalty
                    && qp.status == super::qp_active_set::QpStatus::Optimal
                    && n_constraint_rows > 0
                {
                    Some(
                        qp.multipliers[..n_constraint_rows]
                            .iter()
                            .map(|&l| l.abs())
                            .fold(T::zero(), T::max),
                    )
                } else {
                    None
                };
                penalty = merit_penalty_update(
                    penalty,
                    lambda_inf,
                    pred_reduction,
                    max_violation,
                    config.penalty_margin,
                );

                // L1 merit line search
                let (alpha, x_trial, fx_trial) =
                    sqp_line_search(f, &x, &d, fx, &gx, constraints, penalty, &mut fun_evals);
                (alpha, x_trial, fx_trial, false, false)
            }
            SqpAcceptance::Filter => {
                match filter_line_search(
                    f,
                    &x,
                    &d,
                    fx,
                    dgd,
                    constraints,
                    &mut filter,
                    config,
                    &mut fun_evals,
                ) {
                    Some((alpha, x_trial, fx_trial, f_type)) => {
                        (alpha, x_trial, fx_trial, f_type, false)
                    }
                    None => {
                        // Backtracking failed outright: restore feasibility
                        // before giving up.
                        match feasibility_restoration(f, &x, constraints, n, &mut fun_evals) {
                            Some((x_r, f_r)) => (T::zero(), x_r, f_r, false, true),
                            None => {
                                let grad_norm =
                                    gx.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
                                return Ok(OptimResult {
                                    x,
                                    fun: fx,
                                    grad: Some(gx),
                                    iterations: iter,
                                    fun_evals,
                                    grad_evals,
                                    hess_evals: 0,
                                    grad_norm,
                                    converged: false,
                                    status: OptimStatus::LineSearchFailed,
                                    kkt: None,
                                });
                            }
                        }
                    }
                }
            }
        };

        // Second-order correction (Maratos effect). Under filter
        // acceptance the corrected point is kept when it reduces the
        // violation and stays filter-acceptable (f-type steps additionally
        // keep their Armijo guarantee); under merit acceptance the
        // corrected point is kept when it lowers the L1 merit.
        if config.second_order_correction && !restored && alpha < T::from_f64(0.9).unwrap() {
            let violation_trial = compute_max_violation(&x_trial, constraints);
            if violation_trial > T::from_f64(0.9).unwrap() * max_violation {
                // Compute correction step from linearised constraints at x_trial
                let correction = compute_soc(&x_trial, constraints, n);
                let x_corrected = sonium_solvers_core::vec_ops::vec_add(&x_trial, &correction);
                let fx_corrected = f(&x_corrected);
                fun_evals += 1;

                let accept_corrected = match config.acceptance {
                    SqpAcceptance::Merit => {
                        let m_trial = l1_merit(fx_trial, &x_trial, constraints, penalty);
                        let m_corrected =
                            l1_merit(fx_corrected, &x_corrected, constraints, penalty);
                        m_corrected < m_trial
                    }
                    SqpAcceptance::Filter => {
                        let theta_trial = l1_violation_norm(&x_trial, constraints);
                        let theta_corrected = l1_violation_norm(&x_corrected, constraints);
                        let filter_ok = filter_pair_acceptable(
                            &filter,
                            theta_corrected,
                            fx_corrected,
                            fx,
                            l1_violation_norm(&x, constraints),
                            config,
                        );
                        let armijo_ok = !f_type_step
                            || fx_corrected <= fx + config.filter_armijo_eta * alpha * dgd;
                        theta_corrected < theta_trial && filter_ok && armijo_ok
                    }
                };

                if accept_corrected {
                    x_trial = x_corrected;
                    fx_trial = fx_corrected;
                }
            }
        }

        let fx_new = fx_trial;
        let gx_new = grad_fn(&x_trial);
        grad_evals += 1;

        // Damped BFGS update (Powell). The damping theory is stated for
        // the Lagrangian Hessian, so y must be the Lagrangian gradient
        // difference at the QP multiplier estimates,
        //   y = ∇L(x_trial, λ_QP) − ∇L(x, λ_QP),
        // not the objective gradient difference. The two coincide only
        // when every constraint is affine (∇²L = ∇²f).
        let s = sonium_solvers_core::vec_ops::vec_sub(&x_trial, &x);
        let y = super::qp_active_set::lagrangian_gradient_difference_checked(
            &gx_new,
            &gx,
            &qp.multipliers,
            &x_trial,
            &x,
            constraints,
            iter,
        )?;
        let bs = b.dot(&s);
        let sbs = T::vec_dot(&s, &bs);
        let sy = T::vec_dot(&s, &y);

        let theta = if sy >= T::from_f64(0.2).unwrap() * sbs {
            T::one()
        } else if sbs > T::machine_epsilon() {
            T::from_f64(0.8).unwrap() * sbs / (sbs - sy)
        } else {
            T::one()
        };

        let r = sonium_solvers_core::vec_ops::vec_add(
            &sonium_solvers_core::vec_ops::vec_scale(&y, theta),
            &sonium_solvers_core::vec_ops::vec_scale(&bs, T::one() - theta),
        );
        let sr = T::vec_dot(&s, &r);

        if sr.abs() > T::machine_epsilon() && sbs.abs() > T::machine_epsilon() {
            let rho_r = T::one() / sr;
            let rho_s = T::one() / sbs;
            for i in 0..n {
                for j in 0..=i {
                    let update = rho_r * r[i] * r[j] - rho_s * bs[i] * bs[j];
                    b[[i, j]] += update;
                    if i != j {
                        b[[j, i]] += update;
                    }
                }
            }
        }

        x = x_trial;
        fx = fx_new;
        gx = gx_new;
    }

    let grad_norm = gx.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
    Ok(OptimResult {
        x,
        fun: fx,
        grad: Some(gx),
        iterations: config.max_iterations,
        fun_evals,
        grad_evals,
        hess_evals: 0,
        grad_norm,
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
}

fn attach_final_kkt<T: RealScalar>(
    result: &mut OptimResult<T>,
    constraints: &[Constraint<T>],
    bounds: &[Bound<T>],
) -> Result<(), OptimError> {
    if !bounds.is_empty() && bounds.len() != result.x.len() {
        return Err(OptimError::DimensionMismatch {
            expected: result.x.len(),
            got: bounds.len(),
        });
    }
    let Some(gradient) = result.grad.as_deref() else {
        return Ok(());
    };
    let hessian = sonium_solvers_core::dense::DenseMatrix::eye(result.x.len());
    let qp = solve_sqp_direction(
        gradient,
        &hessian,
        &result.x,
        constraints,
        bounds,
        result.x.len(),
    )?;
    let multipliers =
        super::qp_active_set::kkt_multipliers_from_qp(&qp.multipliers, constraints, bounds)?;
    result.attach_kkt(constraints, bounds, multipliers)
}

/// Solve the SQP direction subproblem with the dense active-set QP solver
/// (equalities, inequalities, and bounds all enter the direction). The full
/// [`super::qp_active_set::QpSolution`] is returned so the caller can use
/// the KKT multipliers for the merit penalty update.
fn solve_sqp_direction<T: RealScalar>(
    g: &[T],
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    x: &[T],
    constraints: &[Constraint<T>],
    bounds: &[Bound<T>],
    n: usize,
) -> Result<super::qp_active_set::QpSolution<T>, OptimError> {
    super::qp_active_set::try_solve_linearized_qp(g, b, x, constraints, bounds, n)
}

/// L1 merit penalty update.
///
/// With `lambda_inf = Some(||lambda||_inf)` (QP multipliers available),
/// apply the multiplier-based rule `mu = max(mu, ||lambda||_inf * (1 +
/// margin))` — the exactness condition of Nocedal & Wright (§18.5). With
/// `None`, fall back to the legacy predicted-reduction heuristic, which
/// is only applied while the constraint violation is above machine
/// precision.
fn merit_penalty_update<T: RealScalar>(
    penalty: T,
    lambda_inf: Option<T>,
    pred_reduction: T,
    max_violation: T,
    margin: T,
) -> T {
    match lambda_inf {
        Some(l) => T::max(penalty, l * (T::one() + margin)),
        None => {
            if max_violation > T::machine_epsilon() {
                T::max(penalty, (pred_reduction.abs() + T::one()) / max_violation)
            } else {
                penalty
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sqp_line_search<T, F>(
    f: &F,
    x: &[T],
    d: &[T],
    fx: T,
    gx: &[T],
    constraints: &[Constraint<T>],
    penalty: T,
    fun_evals: &mut usize,
) -> (T, Vec<T>, T)
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let m0 = l1_merit(fx, x, constraints, penalty);
    let c1 = T::from_f64(1e-4).unwrap();
    // Exact directional derivative of the L1 merit φ(x; μ) = f(x) +
    // μ‖c(x)‖₁ along d: Dφ = ∇fᵀd − μ‖c(x)‖₁ (the violated-constraint
    // norm at the current point). Using only ∇fᵀd weakens the Armijo
    // sufficient-decrease test.
    let dm = T::vec_dot(gx, d) - penalty * l1_violation_norm(x, constraints);
    let rho = T::from_f64(0.5).unwrap();
    let mut alpha = T::one();

    for _ in 0..30 {
        let x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, alpha);
        let fx_new = f(&x_new);
        *fun_evals += 1;
        let m_new = l1_merit(fx_new, &x_new, constraints, penalty);
        if m_new <= m0 + c1 * alpha * dm {
            return (alpha, x_new, fx_new);
        }
        alpha *= rho;
    }

    let x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, alpha);
    let fx_new = f(&x_new);
    *fun_evals += 1;
    (alpha, x_new, fx_new)
}

fn l1_merit<T: RealScalar>(fx: T, x: &[T], constraints: &[Constraint<T>], penalty: T) -> T {
    fx + penalty * l1_violation_norm(x, constraints)
}

/// Fletcher–Leyffer filter acceptance of `(theta, f)` against all filter
/// entries, with the standard margin: acceptable iff for every entry
/// `(theta_j, f_j)`, `theta <= (1 - gamma_theta) * theta_j` or
/// `f <= f_j - gamma_f * theta_j`.
fn filter_acceptable<T: RealScalar>(
    filter: &[(T, T)],
    theta: T,
    f_val: T,
    gamma_theta: T,
    gamma_f: T,
) -> bool {
    filter
        .iter()
        .all(|&(th, fj)| theta <= (T::one() - gamma_theta) * th || f_val <= fj - gamma_f * th)
}

/// Filter acceptance of a trial pair against the filter *and* the current
/// iterate `(theta_k, f_k)` (which is only a filter entry once a theta-type
/// step moves away from it).
fn filter_pair_acceptable<T: RealScalar>(
    filter: &[(T, T)],
    theta: T,
    f_val: T,
    f_k: T,
    theta_k: T,
    config: &SqpConfig<T>,
) -> bool {
    let current_ok = theta <= (T::one() - config.filter_gamma_theta) * theta_k
        || f_val <= f_k - config.filter_gamma_f * theta_k;
    current_ok
        && filter_acceptable(
            filter,
            theta,
            f_val,
            config.filter_gamma_theta,
            config.filter_gamma_f,
        )
}

/// Fletcher–Leyffer filter backtracking line search (Filter acceptance).
///
/// Returns `(alpha, x_trial, f_trial, f_type)` on acceptance, where
/// `f_type` marks steps accepted through the switching condition plus an
/// Armijo decrease on `f` (the filter is not augmented for those); for
/// theta-type steps the current iterate is pushed onto the filter. Returns
/// `None` when the backtracking loop fails — the caller then runs
/// feasibility restoration.
#[allow(clippy::too_many_arguments)]
fn filter_line_search<T, F>(
    f: &F,
    x: &[T],
    d: &[T],
    fx: T,
    dgd: T,
    constraints: &[Constraint<T>],
    filter: &mut Vec<(T, T)>,
    config: &SqpConfig<T>,
    fun_evals: &mut usize,
) -> Option<(T, Vec<T>, T, bool)>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let theta_k = l1_violation_norm(x, constraints);
    let rho = T::from_f64(0.5).unwrap();
    let mut alpha = T::one();

    for _ in 0..30 {
        let x_trial = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, alpha);
        let f_trial = f(&x_trial);
        *fun_evals += 1;
        if !f_trial.is_finite() {
            alpha *= rho;
            continue;
        }
        let theta_trial = l1_violation_norm(&x_trial, constraints);

        // Switching condition (Wächter–Biegler 2006, Case I): the step
        // promises enough objective decrease relative to the violation
        // that it must deliver an Armijo decrease on f.
        let m = alpha * dgd;
        let switching =
            m < T::zero() && -m > config.filter_delta * theta_k.powf(config.filter_s_theta);

        let ok = if switching {
            f_trial <= fx + config.filter_armijo_eta * m
        } else {
            filter_pair_acceptable(filter, theta_trial, f_trial, fx, theta_k, config)
        };

        if ok {
            // A step accepted purely through the filter (not Armijo) makes
            // the current iterate a filter entry.
            if !switching && theta_k > T::zero() {
                filter.push((theta_k, fx));
            }
            return Some((alpha, x_trial, f_trial, switching));
        }
        alpha *= rho;
    }
    None
}

/// Feasibility restoration: minimise the L1 constraint violation theta
/// with projected-gradient steps (the `compute_soc` direction, i.e.
/// `-Jᵀc / ||J||²` per violated row) and backtracking on theta. This is a
/// deliberately simple restoration phase: it returns the first point with
/// a meaningfully smaller violation, or `None` when no progress is made
/// (the caller then reports `LineSearchFailed`). Restoration iterates are
/// not added to the filter.
fn feasibility_restoration<T, F>(
    f: &F,
    x: &[T],
    constraints: &[Constraint<T>],
    n: usize,
    fun_evals: &mut usize,
) -> Option<(Vec<T>, T)>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let theta0 = l1_violation_norm(x, constraints);
    if theta0 <= T::machine_epsilon() {
        return None; // nothing to restore
    }
    let nine_tenths = T::from_f64(0.9).unwrap();
    let half = T::from_f64(0.5).unwrap();

    let mut x_r = x.to_vec();
    let mut theta_r = theta0;
    for _ in 0..25 {
        let step = compute_soc(&x_r, constraints, n);
        if step.iter().all(|&si| si.abs() <= T::machine_epsilon()) {
            break;
        }
        // Backtrack until the violation decreases.
        let mut a = T::one();
        let mut improved = false;
        for _ in 0..20 {
            let cand = sonium_solvers_core::vec_ops::vec_add_scaled(&x_r, &step, a);
            let theta_c = l1_violation_norm(&cand, constraints);
            if theta_c < nine_tenths * theta_r {
                x_r = cand;
                theta_r = theta_c;
                improved = true;
                break;
            }
            a *= half;
        }
        if !improved {
            break;
        }
    }

    if theta_r < theta0 {
        let f_r = f(&x_r);
        *fun_evals += 1;
        if f_r.is_finite() {
            return Some((x_r, f_r));
        }
    }
    None
}

/// L1 constraint-violation norm ‖c(x)‖₁: |c| for equalities, max(−c, 0)
/// for inequalities. This is the constraint part of the L1 merit function.
fn l1_violation_norm<T: RealScalar>(x: &[T], constraints: &[Constraint<T>]) -> T {
    let mut norm = T::zero();
    for c in constraints {
        let cv = (c.fun)(x);
        for &ci in cv.iter() {
            norm += match c.kind {
                ConstraintKind::Equality => ci.abs(),
                ConstraintKind::Inequality => (-ci).max(T::zero()),
            };
        }
    }
    norm
}

fn projected_kkt_norm<T: RealScalar>(
    g: &[T],
    x: &[T],
    constraints: &[Constraint<T>],
    n: usize,
) -> T {
    let mut proj_g = g.to_vec();
    for _ in 0..3 {
        for c in constraints {
            if !matches!(c.kind, ConstraintKind::Equality) {
                continue;
            }
            let jac = super::qp_active_set::constraint_jacobian(c, x);
            for k in 0..c.dim {
                let mut jg = T::zero();
                for j in 0..n {
                    jg += jac[[k, j]] * proj_g[j];
                }
                let mut jnorm_sq = T::zero();
                for j in 0..n {
                    jnorm_sq += jac[[k, j]] * jac[[k, j]];
                }
                if jnorm_sq > T::machine_epsilon() {
                    let scale = jg / jnorm_sq;
                    for j in 0..n {
                        proj_g[j] -= scale * jac[[k, j]];
                    }
                }
            }
        }
    }
    proj_g.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max)
}

fn compute_max_violation<T: RealScalar>(x: &[T], constraints: &[Constraint<T>]) -> T {
    let mut max_viol = T::zero();
    for c in constraints {
        let cv = (c.fun)(x);
        for &ci in cv.iter() {
            let v = match c.kind {
                ConstraintKind::Equality => ci.abs(),
                ConstraintKind::Inequality => (-ci).max(T::zero()),
            };
            if v > max_viol {
                max_viol = v;
            }
        }
    }
    max_viol
}

fn compute_soc<T: RealScalar>(x: &[T], constraints: &[Constraint<T>], n: usize) -> Vec<T> {
    // Simple correction: -J^T * c / (J * J^T + eps*I) — projected gradient
    let mut correction = vec![T::zero(); n];
    for c in constraints {
        let cv = (c.fun)(x);
        let jac = super::qp_active_set::constraint_jacobian(c, x);
        for k in 0..c.dim {
            let viol = match c.kind {
                ConstraintKind::Equality => cv[k],
                ConstraintKind::Inequality => (-cv[k]).max(T::zero()).copysign(cv[k]),
            };
            if viol.abs() > T::machine_epsilon() {
                // Compute J_k^T * cv_k / ||J_k||^2
                let mut jnorm_sq = T::zero();
                for j in 0..n {
                    jnorm_sq += jac[[k, j]] * jac[[k, j]];
                }
                if jnorm_sq > T::machine_epsilon() {
                    let scale = -cv[k] / jnorm_sq;
                    for j in 0..n {
                        correction[j] += scale * jac[[k, j]];
                    }
                }
            }
        }
    }
    correction
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sqp_equality() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = SqpConfig::default();
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 5e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 5e-2);
    }

    #[test]
    fn test_sqp_inequality_constrained_optimum() {
        // min x^2 + y^2  s.t.  x + y >= 1.
        // Unconstrained minimiser (0,0) is infeasible; the constrained
        // optimum is (0.5, 0.5) with KKT multiplier lambda = 1.
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = SqpConfig::default();
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        // With a real QP subproblem this converges in a handful of
        // iterations; the projected-gradient substitute ran to the cap.
        assert!(result.iterations < 20, "iterations: {}", result.iterations);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-4);

        // KKT residuals at the reported point.
        let c = result.x[0] + result.x[1] - 1.0;
        assert!(c >= -1e-8, "primal infeasible: c = {c}");
        // Stationarity: grad f - lambda * grad c = 0 with lambda >= 0.
        let gx = g(&result.x);
        let lambda = 0.5 * (gx[0] + gx[1]);
        assert!(lambda >= -1e-8, "dual infeasible: lambda = {lambda}");
        assert!((gx[0] - lambda).abs() < 1e-6, "stationarity residual");
        assert!((gx[1] - lambda).abs() < 1e-6, "stationarity residual");
        // Complementary slackness.
        assert!((c * lambda).abs() < 1e-6, "complementarity residual");
    }

    #[test]
    fn test_sqp_inequality_fd_jacobian_fallback() {
        // Same problem, but the constraint supplies no analytic Jacobian:
        // the solver must fall back to finite differences instead of
        // silently ignoring the constraint.
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: None,
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = SqpConfig::default();
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_sqp_curved_constraint_lagrangian_bfgs() {
        // min  x + y   s.t.  x^2 + y^2 = 1  →  x = y = -1/√2, f = -√2.
        //
        // The constraint is nonlinear (∇²L ≠ ∇²f), so the damped-BFGS
        // update only builds the right curvature when y is the Lagrangian
        // gradient difference ∇L(x⁺, λ⁺) − ∇L(x, λ⁺). With the objective
        // gradient difference instead, y = 0 here (constant objective
        // gradient) and the update degenerates.
        let f = |x: &[f64]| x[0] + x[1];
        let g = |_: &[f64]| vec![1.0, 1.0];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] * x[0] + x[1] * x[1] - 1.0]),
            jac: Some(Box::new(|x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![2.0 * x[0], 2.0 * x[1]],
                )
            })),
        }];
        let x0 = sonium_solvers_core::array![1.0, 0.0];
        let config = SqpConfig::default();
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(result.x[0], -s, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], -s, epsilon = 1e-4);
        assert_relative_eq!(result.fun, -std::f64::consts::SQRT_2, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_penalty_update_multiplier_rule_bounded() {
        // Multiplier rule: repeated updates with a fixed multiplier norm
        // reach a fixed point — the penalty provably stays bounded instead
        // of growing geometrically as the violation shrinks.
        let margin = 0.1;
        let lambda_inf = 7.07;
        let cap = lambda_inf * (1.0 + margin);
        let mut mu = 1.0_f64;
        for _ in 0..50 {
            mu = merit_penalty_update(mu, Some(lambda_inf), 0.0, 1e-14, margin);
        }
        assert!((mu - cap).abs() < 1e-12, "mu = {mu}, cap = {cap}");

        // Legacy heuristic on the same shrinking-violation sequence blows
        // up geometrically (the failure mode the multiplier rule avoids).
        let mut mu_h = 1.0_f64;
        let mut viol = 1.0_f64;
        for _ in 0..15 {
            mu_h = merit_penalty_update(mu_h, None, 1.0, viol, margin);
            viol *= 0.5;
        }
        assert!(mu_h > 1e4, "heuristic should blow up: mu_h = {mu_h}");
    }

    #[test]
    fn test_sqp_penalty_update_fallback_without_multipliers() {
        // No multipliers (e.g. infeasible QP): the predicted-reduction
        // heuristic is used while the violation is significant.
        let mu: f64 = merit_penalty_update(1.0, None, 4.0, 2.0, 0.1);
        assert!((mu - 2.5).abs() < 1e-12, "mu = {mu}"); // (|4| + 1) / 2
        // ... and the penalty never decreases.
        let mu: f64 = merit_penalty_update(10.0, None, 0.0, 2.0, 0.1);
        assert!((mu - 10.0).abs() < 1e-12, "mu = {mu}");
        // Near-feasible iterates leave the penalty untouched.
        let mu: f64 = merit_penalty_update(3.0, None, 100.0, 0.0, 0.1);
        assert!((mu - 3.0).abs() < 1e-12, "mu = {mu}");
    }

    #[test]
    fn test_sqp_multiplier_rule_fixes_underpenalized_stall() {
        // min  -10 (x + y)   s.t.  x^2 + y^2 = 1,  start (1, 0).
        //
        // The KKT multiplier at the solution (1, 1) / sqrt(2) is
        // |lambda*| = 10 / sqrt(2) ~= 7.07, so the L1 merit is exact only
        // for mu >= 7.07. Before the damped-BFGS update was fixed to use
        // the Lagrangian gradient difference, the legacy
        // predicted-reduction heuristic (mu stuck at its initial 1.0)
        // stalled on this instance: the wrong curvature produced QP steps
        // the under-penalised merit kept rejecting. With the corrected
        // ∇L update the legacy heuristic converges too (good steps +
        // second-order correction beat the Maratos effect even at
        // mu < |lambda*|), so the discriminating check is now only that
        // the multiplier-based update — the documented N&W §18.5 default —
        // reaches the KKT point.
        let f = |x: &[f64]| -10.0 * (x[0] + x[1]);
        let g = |_: &[f64]| vec![-10.0, -10.0];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] * x[0] + x[1] * x[1] - 1.0]),
            jac: Some(Box::new(|x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![2.0 * x[0], 2.0 * x[1]],
                )
            })),
        }];
        let x0 = sonium_solvers_core::array![1.0, 0.0];
        let s = std::f64::consts::FRAC_1_SQRT_2;

        // Post-∇L-fix the legacy heuristic converges on this instance as
        // well; keep it under test so a regression of the BFGS update is
        // caught on both penalty paths.
        let legacy = SqpConfig {
            use_multiplier_penalty: false,
            ..SqpConfig::default()
        };
        let res_legacy = sqp(&f, &g, &x0, &constraints, &legacy).unwrap();
        assert!(res_legacy.converged, "status: {:?}", res_legacy.status);
        assert_relative_eq!(res_legacy.x[0], s, epsilon = 1e-4);
        assert_relative_eq!(res_legacy.x[1], s, epsilon = 1e-4);

        let config = SqpConfig::default();
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], s, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], s, epsilon = 1e-4);
    }

    // ------------------------------------------------------------------
    // Filter acceptance (SqpAcceptance::Filter)
    // ------------------------------------------------------------------

    /// The three standard SQP test problems, as (f, g, constraints, x0,
    /// expected solution) closures shared by the filter tests.
    fn circle_equality_constraint() -> Constraint<f64> {
        Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] * x[0] + x[1] * x[1] - 1.0]),
            jac: Some(Box::new(|x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![2.0 * x[0], 2.0 * x[1]],
                )
            })),
        }
    }

    #[test]
    fn test_sqp_filter_equality() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 5e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 5e-2);
    }

    #[test]
    fn test_sqp_filter_inequality_constrained_optimum() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_filter_curved_constraint() {
        // min x + y s.t. x^2 + y^2 = 1 → x = y = -1/sqrt(2).
        let f = |x: &[f64]| x[0] + x[1];
        let g = |_: &[f64]| vec![1.0, 1.0];
        let constraints = vec![circle_equality_constraint()];
        let x0 = sonium_solvers_core::array![1.0, 0.0];
        let config = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(result.x[0], -s, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], -s, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_filter_multiplier_rule_problem() {
        // The under-penalisation stall instance of
        // `test_sqp_multiplier_rule_fixes_underpenalized_stall`: the filter
        // has no penalty parameter at all, so it must converge without any
        // tuning.
        let f = |x: &[f64]| -10.0 * (x[0] + x[1]);
        let g = |_: &[f64]| vec![-10.0, -10.0];
        let constraints = vec![circle_equality_constraint()];
        let x0 = sonium_solvers_core::array![1.0, 0.0];
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let config = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], s, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], s, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_filter_maratos_problem() {
        // The classic Maratos counterexample (Powell; Nocedal & Wright
        // Example 15.4): min 2(x^2 + y^2 - 1) - x  s.t.  x^2 + y^2 = 1,
        // solution (1, 0) with multiplier 3/2. Full SQP steps increase
        // both f and the violation by O(theta^2), so an L1 merit rejects
        // them (Maratos effect); the filter accepts them because the new
        // pair is not dominated.
        let f = |x: &[f64]| 2.0 * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0];
        let g = |x: &[f64]| vec![4.0 * x[0] - 1.0, 4.0 * x[1]];
        let constraints = vec![circle_equality_constraint()];
        // Start on the circle, away from the solution.
        let x0 = sonium_solvers_core::array![0.8, 0.6];

        let filter_cfg = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let res = sqp(&f, &g, &x0, &constraints, &filter_cfg).unwrap();
        assert!(res.converged, "filter status: {:?}", res.status);
        assert_relative_eq!(res.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(res.x[1], 0.0, epsilon = 1e-4);

        // Merit comparison (report only — with the second-order correction
        // the merit path also handles this instance, at extra cost).
        let merit_cfg = SqpConfig::default();
        let merit = sqp(&f, &g, &x0, &constraints, &merit_cfg).unwrap();
        println!(
            "Maratos: filter iters={} f_evals={} | merit iters={} f_evals={}",
            res.iterations, res.fun_evals, merit.iterations, merit.fun_evals
        );
    }

    #[test]
    fn test_sqp_filter_line_search_rejects_dominated_trials() {
        // Unit-level: a filter entry that dominates every point along the
        // search direction forces `filter_line_search` to return None —
        // the condition under which the main loop enters restoration.
        let f = |x: &[f64]| x.dot(x);
        let constraints = vec![circle_equality_constraint()];
        let x = sonium_solvers_core::array![2.0, 0.0];
        let fx = f(&x);
        // Tangential direction: every trial has theta = 3 + alpha^2 and
        // f = 4 + alpha^2, so the entry below dominates them all.
        let d = vec![0.0, -1.0];
        let dgd = 0.0; // g(x)·d with g = 2x
        // Entry with near-zero violation and hugely negative objective:
        // no trial on the ray can beat either margin.
        let mut filter = vec![(1e-12, -1e12)];
        let config = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let mut fun_evals = 0;
        let out = filter_line_search(
            &f,
            &x,
            &d,
            fx,
            dgd,
            &constraints,
            &mut filter,
            &config,
            &mut fun_evals,
        );
        assert!(out.is_none(), "dominated trials must be rejected");
    }

    #[test]
    fn test_sqp_feasibility_restoration_reduces_violation() {
        // Restoration from a deliberately hard start: far from the circle,
        // the projected-gradient restoration must shrink the violation.
        let f = |x: &[f64]| x.dot(x);
        let constraints = vec![circle_equality_constraint()];
        let x = sonium_solvers_core::array![7.0, -3.0];
        let theta0 = l1_violation_norm(&x, &constraints);
        let mut fun_evals = 0;
        let (x_r, _f_r) = feasibility_restoration(&f, &x, &constraints, 2, &mut fun_evals).unwrap();
        let theta_r = l1_violation_norm(&x_r, &constraints);
        assert!(
            theta_r < 0.5 * theta0,
            "restoration made no progress: theta {theta0} -> {theta_r}"
        );
    }

    #[test]
    fn test_sqp_filter_restoration_on_infeasible_problem() {
        // End-to-end restoration exercise: contradictory linear equality
        // constraints (x = 0 and x = 1) make every QP step unacceptable to
        // the filter (violation theta = |x| + |x - 1| >= 1 cannot
        // decrease, and the objective is flat in x), so the backtracking
        // loop fails and the restoration fallback runs. Restoration also
        // cannot reduce theta below its minimum of 1, so the solver must
        // give up cleanly with LineSearchFailed rather than loop or panic.
        let f = |x: &[f64]| x[1] * x[1];
        let g = |x: &[f64]| vec![0.0, 2.0 * x[1]];
        let constraints = vec![
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0]]),
                jac: Some(Box::new(|_x: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        2,
                        vec![1.0, 0.0],
                    )
                })),
            },
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] - 1.0]),
                jac: Some(Box::new(|_x: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        2,
                        vec![1.0, 0.0],
                    )
                })),
            },
        ];
        let x0 = sonium_solvers_core::array![0.5, 5.0];
        let config = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let result = sqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(!result.converged);
        assert_eq!(result.status, OptimStatus::LineSearchFailed);
    }

    #[test]
    fn test_sqp_filter_determinism() {
        let f = |x: &[f64]| 2.0 * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0];
        let g = |x: &[f64]| vec![4.0 * x[0] - 1.0, 4.0 * x[1]];
        let x0 = sonium_solvers_core::array![0.8, 0.6];
        let config = SqpConfig {
            acceptance: SqpAcceptance::Filter,
            ..SqpConfig::default()
        };
        let r1 = sqp(&f, &g, &x0, &[circle_equality_constraint()], &config).unwrap();
        let r2 = sqp(&f, &g, &x0, &[circle_equality_constraint()], &config).unwrap();
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.fun_evals, r2.fun_evals);
    }
}
