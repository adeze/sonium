//! Exact-Hessian SQP (sequential quadratic programming).
//!
//! Each iteration solves the QP subproblem
//!
//! ```text
//! min  ½ dᵀ H d + ∇f(x)ᵀ d
//! s.t. J_eq(x) d = -c_eq(x),  J_ineq(x) d ≥ -c_ineq(x),  lb ≤ x + d ≤ ub
//! ```
//!
//! with the dense Goldfarb–Idnani active-set solver
//! ([`super::qp_active_set::solve_linearized_qp`]): both the linearised
//! equalities and the inequalities (and bounds) enter the direction step,
//! and the QP returns the KKT multipliers used for the next Lagrangian
//! Hessian evaluation. `H` is the user-supplied Hessian of the Lagrangian
//! `∇²_xx L(x, λ)` with `L = f − Σᵢ λᵢ cᵢ`, made positive definite via
//! Levenberg–Marquardt or modified-Cholesky regularisation before the QP
//! solve (the QP solver additionally applies its own GMW correction, so an
//! indefinite `H` can never produce a meaningless direction).
//!
//! Globalisation is an L1 exact merit line search; the merit penalty
//! follows the multiplier-based rule of Nocedal & Wright (§18.5), `μ =
//! max(μ, ‖λ_QP‖∞ · (1 + penalty_margin))`, with a predicted-reduction
//! fallback when the QP did not converge (infeasible linearisation). With
//! the exact Lagrangian Hessian the method is quadratically convergent
//! near a regular KKT point.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{
    Bound, Constraint, ConstraintKind, OptimError, OptimResult, OptimStatus, RealScalar,
    validate_bounds,
};
use crate::util::dense_linalg::modified_cholesky;

/// Hessian regularisation strategy.
#[derive(Debug, Clone, Default)]
pub enum HessianRegularisation<T: RealScalar> {
    LevenbergMarquardt {
        initial_lambda: T,
    },
    #[default]
    ModifiedCholesky,
}

/// Configuration for the exact-Hessian SQP solver.
#[derive(Debug, Clone)]
pub struct AdvancedSqpConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub constraint_tolerance: T,
    pub hessian_regularisation: HessianRegularisation<T>,
    /// Relative margin of the multiplier-based L1 merit penalty update
    /// (Nocedal & Wright §18.5).
    pub penalty_margin: T,
    pub bounds: Vec<Bound<T>>,
}

impl Default for AdvancedSqpConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-6,
            constraint_tolerance: 1e-6,
            hessian_regularisation: HessianRegularisation::ModifiedCholesky,
            penalty_margin: 0.1,
            bounds: Vec::new(),
        }
    }
}

impl Default for AdvancedSqpConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-4,
            constraint_tolerance: 1e-4,
            hessian_regularisation: HessianRegularisation::ModifiedCholesky,
            penalty_margin: 0.1,
            bounds: Vec::new(),
        }
    }
}

/// Minimise `f(x)` subject to constraints with an exact-Hessian SQP
/// iteration (see the module docs): the search direction is the solution
/// of the linearised QP subproblem, and the Lagrange multipliers are the
/// QP multipliers.
///
/// `lagrangian_hessian_fn(x, lambda)` returns the Hessian of the Lagrangian
/// `∇²_xx L(x, λ)` with `L = f − Σᵢ λᵢ cᵢ` (inequality multipliers
/// non-negative) as a dense matrix; `lambda` is ordered like the rows of
/// `constraints`.
pub fn sqp_advanced<T, F, Fg, H>(
    f: &F,
    grad_fn: &Fg,
    lagrangian_hessian_fn: &H,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &AdvancedSqpConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
    H: Fn(&[T], &[T]) -> sonium_solvers_core::dense::DenseMatrix<T>,
{
    super::validate_hessian_start(
        lagrangian_hessian_fn,
        x0,
        constraints.iter().map(|constraint| constraint.dim).sum(),
        "advanced SQP Hessian preflight",
    )?;
    validate_bounds(&config.bounds, x0.len())?;
    super::validate_optimizer_start(f, grad_fn, x0, constraints)?;
    let callback_non_finite = Cell::new(false);
    let checked_f = |point: &[T]| {
        let value = f(point);
        if !value.is_finite() {
            callback_non_finite.set(true);
        }
        value
    };
    let checked_grad = |point: &[T]| {
        let gradient = grad_fn(point);
        if gradient.len() != point.len() || gradient.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
            return vec![T::nan(); point.len()];
        }
        gradient
    };
    let checked_hessian = |point: &[T], multipliers: &[T]| {
        let hessian = lagrangian_hessian_fn(point, multipliers);
        if hessian.nrows() != point.len()
            || hessian.ncols() != point.len()
            || (0..hessian.nrows())
                .any(|row| (0..hessian.ncols()).any(|column| !hessian[[row, column]].is_finite()))
        {
            callback_non_finite.set(true);
            return sonium_solvers_core::dense::DenseMatrix::zeros(point.len(), point.len());
        }
        hessian
    };
    let inner_result = catch_unwind(AssertUnwindSafe(|| {
        sqp_advanced_impl(
            &checked_f,
            &checked_grad,
            &checked_hessian,
            x0,
            constraints,
            config,
            &callback_non_finite,
        )
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "advanced SQP callback".into(),
    })?;
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "advanced SQP callback returned a non-finite value".into(),
        });
    }
    let mut result = inner_result?;
    if !config.bounds.is_empty() && config.bounds.len() != result.x.len() {
        return Err(OptimError::DimensionMismatch {
            expected: result.x.len(),
            got: config.bounds.len(),
        });
    }
    if let Some(gradient) = result.grad.as_deref() {
        let hessian = sonium_solvers_core::dense::DenseMatrix::eye(result.x.len());
        let qp = super::qp_active_set::try_solve_linearized_qp(
            gradient,
            &hessian,
            &result.x,
            constraints,
            &config.bounds,
            result.x.len(),
        )?;
        let multipliers = super::qp_active_set::kkt_multipliers_from_qp(
            &qp.multipliers,
            constraints,
            &config.bounds,
        )?;
        result.attach_kkt(constraints, &config.bounds, multipliers)?;
    }
    result.gate_kkt(
        config.grad_tolerance.max(config.constraint_tolerance) * T::from_f64(100.0).unwrap(),
    );
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn sqp_advanced_impl<T, F, Fg, H>(
    f: &F,
    grad_fn: &Fg,
    lagrangian_hessian_fn: &H,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &AdvancedSqpConfig<T>,
    callback_non_finite: &Cell<bool>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
    H: Fn(&[T], &[T]) -> sonium_solvers_core::dense::DenseMatrix<T>,
{
    let n = x0.len();
    let n_constraint_rows: usize = constraints.iter().map(|c| c.dim).sum();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    if !fx.is_finite() {
        callback_non_finite.set(true);
    }
    let mut gx = grad_fn(&x);
    if gx.len() != n || gx.iter().any(|value| !value.is_finite()) {
        callback_non_finite.set(true);
        gx = vec![T::nan(); n];
    }
    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;
    let mut hess_evals = 0_usize;

    let mut lambda = vec![T::zero(); n_constraint_rows];
    let mut penalty = T::from_f64(1.0).unwrap();
    let half = T::from_f64(0.5).unwrap();

    for iter in 0..config.max_iterations {
        let max_violation = compute_max_violation(&x, constraints, callback_non_finite);

        // Projected gradient norm onto null space of equality constraint Jacobians
        let grad_norm = projected_kkt_norm(&gx, &x, constraints, n, iter)?;

        if grad_norm < config.grad_tolerance && max_violation < config.constraint_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(gx),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals,
                grad_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Lagrangian Hessian at the current multiplier estimates.
        let hess = lagrangian_hessian_fn(&x, &lambda);
        if hess.nrows() != n
            || hess.ncols() != n
            || (0..hess.nrows())
                .any(|row| (0..hess.ncols()).any(|column| !hess[[row, column]].is_finite()))
        {
            callback_non_finite.set(true);
        }
        hess_evals += 1;
        let reg_hess = regularise_hessian(&hess, &config.hessian_regularisation, n);

        // Direction step: solve the linearised QP. Both equalities and
        // inequalities (and bounds) enter the direction; the QP returns
        // the multipliers for the next Hessian evaluation and the merit
        // penalty update.
        let qp = super::qp_active_set::try_solve_linearized_qp(
            &gx,
            &reg_hess,
            &x,
            constraints,
            &config.bounds,
            n,
        )?;
        let d = qp.d;

        // Direction-based KKT check: at a constrained optimum with active
        // inequalities the equality-projected gradient does not vanish,
        // but the QP direction does.
        let d_norm = d.iter().map(|&di| di.abs()).fold(T::zero(), T::max);
        if d_norm < config.grad_tolerance && max_violation < config.constraint_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(gx),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals,
                grad_norm: d_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Multiplier update: the QP multipliers replace the previous
        // estimates (constraint rows only; bound rows are not part of the
        // user Lagrangian). When the QP failed to converge the estimates
        // are not meaningful and are left untouched.
        if qp.status == super::qp_active_set::QpStatus::Optimal {
            lambda.copy_from_slice(&qp.multipliers[..n_constraint_rows]);
        }

        // Merit penalty update. Multiplier-based rule (Nocedal & Wright,
        // §18.5): the L1 merit is exact once mu >= ||lambda||_inf, so
        // track the QP multipliers with a small relative margin. When the
        // QP did not terminate at an optimum, fall back to the
        // predicted-reduction heuristic.
        let dgd = T::vec_dot(&gx, &d);
        let bd = reg_hess.dot(&d);
        let pred_reduction = -dgd - half * T::vec_dot(&d, &bd);
        let lambda_inf =
            if qp.status == super::qp_active_set::QpStatus::Optimal && n_constraint_rows > 0 {
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

        // L1 merit line search.
        let m0 = l1_merit(fx, &x, constraints, penalty, callback_non_finite);
        // Exact directional derivative of the L1 merit φ(x; μ) = f(x) +
        // μ‖c(x)‖₁ along d: Dφ = ∇fᵀd − μ‖c(x)‖₁ (the violated-constraint
        // norm at the current point). Using only ∇fᵀd weakens the Armijo
        // sufficient-decrease test.
        let dm = dgd - penalty * l1_violation_norm(&x, constraints, callback_non_finite);
        let c1 = T::from_f64(1e-4).unwrap();
        let rho = T::from_f64(0.5).unwrap();
        let mut alpha = T::one();
        let mut x_new = x.clone();
        let mut fx_new = fx;

        for _ in 0..30 {
            x_new = x
                .iter()
                .zip(&d)
                .map(|(&xi, &di)| xi + alpha * di)
                .collect::<Vec<_>>();
            fx_new = f(&x_new);
            fun_evals += 1;
            let m_new = l1_merit(fx_new, &x_new, constraints, penalty, callback_non_finite);
            if m_new <= m0 + c1 * alpha * dm {
                break;
            }
            alpha *= rho;
        }

        let gx_new = grad_fn(&x_new);
        if gx_new.len() != n || gx_new.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
        }
        grad_evals += 1;

        x = x_new;
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
        hess_evals,
        grad_norm,
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
}

/// Make the Lagrangian Hessian positive definite for the QP solve.
fn regularise_hessian<T: RealScalar>(
    hess: &sonium_solvers_core::dense::DenseMatrix<T>,
    regularisation: &HessianRegularisation<T>,
    n: usize,
) -> sonium_solvers_core::dense::DenseMatrix<T> {
    match regularisation {
        HessianRegularisation::ModifiedCholesky => {
            // `hess` is the n x n Lagrangian Hessian by construction.
            let (l, _e) = modified_cholesky(hess).expect("SQP Hessian is square by construction");
            // Reconstruct: H_reg = L * L^T
            l.dot(&l.t())
        }
        HessianRegularisation::LevenbergMarquardt { initial_lambda: lm } => {
            let mut h = hess.clone();
            let mut lam = *lm;
            loop {
                let mut h_reg = h.clone();
                for i in 0..n {
                    h_reg[[i, i]] += lam;
                }
                if crate::util::dense_linalg::cholesky(&h_reg).is_some() {
                    break h_reg;
                }
                lam *= T::from_f64(10.0).unwrap();
                if lam > T::from_f64(1e20).unwrap() {
                    h = sonium_solvers_core::dense::DenseMatrix::eye(n);
                    break h;
                }
            }
        }
    }
}

/// L1 merit penalty update (mirrors [`super::sqp_impl`]): with
/// `lambda_inf = Some(||lambda||_inf)` apply the multiplier-based rule
/// `mu = max(mu, ||lambda||_inf * (1 + margin))`; otherwise fall back to
/// the predicted-reduction heuristic, only applied while the constraint
/// violation is above machine precision. The penalty never decreases.
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

fn l1_merit<T: RealScalar>(
    fx: T,
    x: &[T],
    constraints: &[Constraint<T>],
    penalty: T,
    callback_non_finite: &Cell<bool>,
) -> T {
    fx + penalty * l1_violation_norm(x, constraints, callback_non_finite)
}

/// L1 constraint-violation norm ‖c(x)‖₁: |c| for equalities, max(−c, 0)
/// for inequalities. This is the constraint part of the L1 merit function.
fn l1_violation_norm<T: RealScalar>(
    x: &[T],
    constraints: &[Constraint<T>],
    callback_non_finite: &Cell<bool>,
) -> T {
    let mut norm = T::zero();
    for c in constraints {
        let cv = (c.fun)(x);
        if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
            return T::nan();
        }
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
    iteration: usize,
) -> Result<T, OptimError> {
    let mut proj_g = g.to_vec();
    for _ in 0..3 {
        for (index, c) in constraints.iter().enumerate() {
            if !matches!(c.kind, ConstraintKind::Equality) {
                continue;
            }
            let jac = super::qp_active_set::checked_constraint_jacobian(c, x, n, index, iteration)?;
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
    Ok(proj_g.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max))
}

fn compute_max_violation<T: RealScalar>(
    x: &[T],
    constraints: &[Constraint<T>],
    callback_non_finite: &Cell<bool>,
) -> T {
    let mut max_viol = T::zero();
    for c in constraints {
        let cv = (c.fun)(x);
        if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
            return T::nan();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sonium_solvers_core::vec_ops::SliceDot;

    #[test]
    fn test_sqp_advanced_equality() {
        // min x^2 + y^2 s.t. x + y = 1
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        // ∇²L = ∇²f − λ∇²c = 2I (c is affine).
        let hess = |_x: &[f64], _lam: &[f64]| -> sonium_solvers_core::dense::DenseMatrix<f64> {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0, 2.0])
        };
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = AdvancedSqpConfig::default();
        let result = sqp_advanced(&f, &g, &hess, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 5e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 5e-2);
    }

    #[test]
    fn test_sqp_advanced_curved_constraint_kkt() {
        // min  x + y   s.t.  x^2 + y^2 = 1  →  x = y = -1/√2, f = -√2,
        // KKT multiplier λ = -1/√2 (stationarity (1,1) = λ (2x, 2y)).
        //
        // With the exact Lagrangian Hessian a true SQP converges
        // quadratically to the analytical KKT point with the default
        // tolerances — no penalty-throttled asymptotic tail. LM
        // regularisation floors the (initially singular) Hessian.
        let f = |x: &[f64]| x[0] + x[1];
        let g = |_: &[f64]| vec![1.0, 1.0];
        // ∇²L = ∇²f − λ∇²c = −2λ I  (equality c = x² + y² − 1).
        let hess = |_x: &[f64], lam: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[-2.0 * lam[0], -2.0 * lam[0]])
        };
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
        let x0 = sonium_solvers_core::array![-1.0, 0.0];
        let config = AdvancedSqpConfig {
            hessian_regularisation: HessianRegularisation::LevenbergMarquardt {
                initial_lambda: 1.0,
            },
            ..AdvancedSqpConfig::default()
        };
        let result = sqp_advanced(&f, &g, &hess, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        println!(
            "curved constraint: iters={} fun_evals={} hess_evals={}",
            result.iterations, result.fun_evals, result.hess_evals
        );
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(result.x[0], -s, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], -s, epsilon = 1e-4);
        assert_relative_eq!(result.fun, -std::f64::consts::SQRT_2, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_advanced_inequality_direction_step() {
        // min (x - 3)^2  s.t.  1 - x >= 0  (x <= 1): optimum x = 1.
        //
        // From x0 = 0 the unconstrained Newton step d = 3 violates the
        // linearised inequality; the QP direction must respect it and land
        // exactly on the active bound in a single step (the direction-norm
        // KKT check then terminates the run: at the active inequality the
        // equality-projected gradient does not vanish, but the QP
        // direction does).
        let f = |x: &[f64]| (x[0] - 3.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 3.0)];
        // ∇²L = ∇²f − λ∇²c = 2 (c is affine).
        let hess =
            |_x: &[f64], _lam: &[f64]| sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0]);
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![1.0 - x[0]]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![-1.0])
            })),
        }];
        let config = AdvancedSqpConfig::default();
        let result = sqp_advanced(&f, &g, &hess, &[0.0], &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(
            result.iterations <= 2,
            "QP should land on the bound immediately: {}",
            result.iterations
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_sqp_advanced_quadratic_convergence() {
        // Quadratic convergence with the exact Hessian: a smooth, mildly
        // nonlinear problem must take only a handful of iterations with
        // the default (tight) tolerances.
        //   min  e^x + y^2  s.t.  x + y = 1
        // KKT: e^x = 2y = 2(1 - x) → x* ≈ 0.3149230…, y* = 1 − x*.
        let f = |x: &[f64]| x[0].exp() + x[1] * x[1];
        let g = |x: &[f64]| vec![x[0].exp(), 2.0 * x[1]];
        // ∇²L = diag(e^x, 2) − λ·0.
        let hess = |x: &[f64], _lam: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[x[0].exp(), 2.0])
        };
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let config = AdvancedSqpConfig::default();
        let result = sqp_advanced(&f, &g, &hess, &[0.0, 1.0], &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        println!(
            "quadratic convergence: iters={} fun_evals={}",
            result.iterations, result.fun_evals
        );
        assert!(
            result.iterations <= 10,
            "exact Hessian should converge quadratically: {} iterations",
            result.iterations
        );
        // Reference root of e^x + 2x - 2 = 0.
        let x_star = 0.314_923_057_847_803;
        assert_relative_eq!(result.x[0], x_star, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 1.0 - x_star, epsilon = 1e-6);
    }

    #[test]
    fn advanced_sqp_rejects_non_finite_generated_objective() {
        let objective_calls = Cell::new(0_usize);
        let f = |_: &[f64]| {
            let call = objective_calls.get();
            objective_calls.set(call + 1);
            if call == 0 { 0.0 } else { f64::NAN }
        };
        let g = |_x: &[f64]| vec![0.0];
        let hess = |_x: &[f64], _lambda: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[1.0])
        };
        let config = AdvancedSqpConfig {
            max_iterations: 1,
            ..AdvancedSqpConfig::default()
        };

        assert!(matches!(
            sqp_advanced(&f, &g, &hess, &[0.0], &[], &config),
            Err(OptimError::NumericalError { .. })
        ));
    }
}
