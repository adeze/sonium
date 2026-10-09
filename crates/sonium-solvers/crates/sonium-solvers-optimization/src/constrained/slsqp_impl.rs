//! SLSQP: Sequential Least Squares Programming with active-set QP solver.
//!
//! Solves a QP subproblem at each iteration (equalities and inequalities)
//! with the Goldfarb–Idnani dual active-set method
//! (`super::qp_active_set`) to determine the search direction, then performs
//! a line search using an L1 merit function. Constraints without an analytic
//! Jacobian fall back to central finite differences.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{
    Bound, Constraint, ConstraintKind, OptimError, OptimResult, OptimStatus, RealScalar,
    validate_bounds,
};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for SLSQP.
#[derive(Debug, Clone)]
pub struct SlsqpConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub constraint_tolerance: T,
    pub bounds: Vec<Bound<T>>,
}

impl Default for SlsqpConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-6,
            constraint_tolerance: 1e-6,
            bounds: Vec::new(),
        }
    }
}

impl Default for SlsqpConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-4,
            constraint_tolerance: 1e-4,
            bounds: Vec::new(),
        }
    }
}

/// Minimise `f(x)` subject to constraints using SLSQP.
pub fn slsqp<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &SlsqpConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    super::validate_optimizer_start(f, grad_fn, x0, constraints)?;
    validate_bounds(&config.bounds, x0.len())?;
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
        }
        gradient
    };
    let inner_result = catch_unwind(AssertUnwindSafe(|| {
        slsqp_impl(
            &checked_f,
            &checked_grad,
            x0,
            constraints,
            config,
            &callback_non_finite,
        )
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "SLSQP callback".into(),
    })?;
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "SLSQP callback returned a non-finite value".into(),
        });
    }
    let mut result = inner_result?;
    attach_final_kkt(&mut result, constraints, &config.bounds)?;
    result.gate_kkt(
        config.grad_tolerance.max(config.constraint_tolerance) * T::from_f64(100.0).unwrap(),
    );
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn slsqp_impl<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &SlsqpConfig<T>,
    callback_non_finite: &Cell<bool>,
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

    // BFGS Hessian approximation for the Lagrangian
    let mut b = sonium_solvers_core::dense::DenseMatrix::eye(n);

    for iter in 0..config.max_iterations {
        let max_violation = compute_max_violation(&x, constraints, callback_non_finite);

        // Compute projected gradient norm: project objective gradient onto null space
        // of equality constraint Jacobians.
        let grad_norm = projected_kkt_norm(&gx, &x, constraints, n, iter)?;

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

        // Solve QP subproblem: min g^T d + 0.5 d^T B d s.t. linearised
        // constraints (equalities, inequalities and bounds all enter the
        // direction via the active-set QP solver).
        let qp = solve_qp_subproblem(&gx, &b, &x, constraints, &config.bounds, n)?;
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

        // L1 merit function line search
        let merit_penalty = T::max(
            T::from_f64(1.0).unwrap(),
            compute_merit_penalty(&x, constraints, callback_non_finite),
        );
        let (alpha, x_new, fx_new) = l1_merit_line_search(
            f,
            &x,
            &d,
            fx,
            &gx,
            constraints,
            merit_penalty,
            &mut fun_evals,
            callback_non_finite,
        );

        let gx_new = grad_fn(&x_new);
        grad_evals += 1;

        // BFGS update of B (approximate Lagrangian Hessian). Kraft's
        // SLSQP (and Powell's damping theory) forms y from the Lagrangian
        // gradient at the QP multiplier estimates,
        //   y = ∇L(x_new, λ_QP) − ∇L(x, λ_QP),
        // not from the objective gradient; the two coincide only for
        // affine constraints.
        let s = d.iter().map(|&di| di * alpha).collect::<Vec<_>>();
        let y = super::qp_active_set::lagrangian_gradient_difference_checked(
            &gx_new,
            &gx,
            &qp.multipliers,
            &x_new,
            &x,
            constraints,
            iter,
        )?;

        // Damped BFGS update (Powell's modification)
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
    let qp = solve_qp_subproblem(
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

/// Solve the QP subproblem with the dense active-set QP solver
/// (Goldfarb–Idnani dual method; equalities, inequalities, and bounds all
/// enter the direction). The full [`super::qp_active_set::QpSolution`] is
/// returned so the caller can use the KKT multipliers for the damped-BFGS
/// Lagrangian update.
fn solve_qp_subproblem<T: RealScalar>(
    g: &[T],
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    x: &[T],
    constraints: &[Constraint<T>],
    bounds: &[Bound<T>],
    n: usize,
) -> Result<super::qp_active_set::QpSolution<T>, OptimError> {
    super::qp_active_set::try_solve_linearized_qp(g, b, x, constraints, bounds, n)
}

#[allow(clippy::too_many_arguments)]
fn l1_merit_line_search<T, F>(
    f: &F,
    x: &[T],
    d: &[T],
    fx: T,
    gx: &[T],
    constraints: &[Constraint<T>],
    penalty: T,
    fun_evals: &mut usize,
    callback_non_finite: &Cell<bool>,
) -> (T, Vec<T>, T)
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let merit = |xp: &[T], fxp: T| -> T {
        let mut m = fxp;
        if !fxp.is_finite() {
            callback_non_finite.set(true);
        }
        for c in constraints {
            let cv = (c.fun)(xp);
            if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                continue;
            }
            for &ci in cv.iter() {
                match c.kind {
                    ConstraintKind::Equality => m += penalty * ci.abs(),
                    ConstraintKind::Inequality => m += penalty * (-ci).max(T::zero()),
                }
            }
        }
        m
    };

    let m0 = merit(x, fx);
    let c1 = T::from_f64(1e-4).unwrap();
    // Exact directional derivative of the L1 merit φ(x; μ) = f(x) +
    // μ‖c(x)‖₁ along d: Dφ = ∇fᵀd − μ‖c(x)‖₁ (the violated-constraint
    // norm at the current point). Using only ∇fᵀd weakens the Armijo
    // sufficient-decrease test.
    let violation_norm: T = {
        let mut norm = T::zero();
        for c in constraints {
            let cv = (c.fun)(x);
            if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                continue;
            }
            for &ci in cv.iter() {
                norm += match c.kind {
                    ConstraintKind::Equality => ci.abs(),
                    ConstraintKind::Inequality => (-ci).max(T::zero()),
                };
            }
        }
        norm
    };
    let dm = T::vec_dot(gx, d) - penalty * violation_norm;
    let rho = T::from_f64(0.5).unwrap();
    let mut alpha = T::one();

    for _ in 0..30 {
        let x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, alpha);
        let fx_new = f(&x_new);
        if !fx_new.is_finite() {
            callback_non_finite.set(true);
        }
        *fun_evals += 1;
        let m_new = merit(&x_new, fx_new);

        if m_new <= m0 + c1 * alpha * dm {
            return (alpha, x_new, fx_new);
        }
        alpha *= rho;
    }

    let x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, alpha);
    let fx_new = f(&x_new);
    if !fx_new.is_finite() {
        callback_non_finite.set(true);
    }
    *fun_evals += 1;
    (alpha, x_new, fx_new)
}

/// Projected KKT gradient norm: project the objective gradient onto the null space
/// of equality constraint Jacobians. For unconstrained or inequality-only problems,
/// this is just the infinity norm of the gradient.
fn projected_kkt_norm<T: RealScalar>(
    g: &[T],
    x: &[T],
    constraints: &[Constraint<T>],
    n: usize,
    iteration: usize,
) -> Result<T, OptimError> {
    let mut proj_g = g.to_vec();
    // Remove the component of g in the row space of equality constraint Jacobians:
    // proj_g = g - J^T * (J*J^T)^{-1} * J * g
    // For simplicity, iterate: proj_g -= J_k^T * (J_k * proj_g) / ||J_k||^2
    for _ in 0..3 {
        for c in constraints {
            if !matches!(c.kind, ConstraintKind::Equality) {
                continue;
            }
            let jac = super::qp_active_set::checked_constraint_jacobian(c, x, n, 0, iteration)?;
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
            let viol = match c.kind {
                ConstraintKind::Equality => ci.abs(),
                ConstraintKind::Inequality => (-ci).max(T::zero()),
            };
            if viol > max_viol {
                max_viol = viol;
            }
        }
    }
    max_viol
}

fn compute_merit_penalty<T: RealScalar>(
    x: &[T],
    constraints: &[Constraint<T>],
    callback_non_finite: &Cell<bool>,
) -> T {
    let mut penalty = T::one();
    for c in constraints {
        let cv = (c.fun)(x);
        if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
            return T::nan();
        }
        for &ci in cv.iter() {
            let viol = match c.kind {
                ConstraintKind::Equality => ci.abs(),
                ConstraintKind::Inequality => (-ci).max(T::zero()),
            };
            if viol > T::machine_epsilon() {
                penalty = T::max(penalty, viol * T::from_f64(10.0).unwrap());
            }
        }
    }
    penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_slsqp_equality() {
        // min x^2 + y^2 s.t. x + y = 1
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
        let config = SlsqpConfig::default();
        let result = slsqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 5e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 5e-2);
    }

    #[test]
    fn test_slsqp_inequality_constrained_optimum() {
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
        let config = SlsqpConfig::default();
        let result = slsqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        // With a real active-set QP this converges in a handful of
        // iterations; the projected-gradient substitute ran to the cap.
        assert!(result.iterations < 20, "iterations: {}", result.iterations);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-4);

        // KKT residuals at the reported point.
        let c = result.x[0] + result.x[1] - 1.0;
        assert!(c >= -1e-8, "primal infeasible: c = {c}");
        let gx = g(&result.x);
        let lambda = 0.5 * (gx[0] + gx[1]);
        assert!(lambda >= -1e-8, "dual infeasible: lambda = {lambda}");
        assert!((gx[0] - lambda).abs() < 1e-6, "stationarity residual");
        assert!((gx[1] - lambda).abs() < 1e-6, "stationarity residual");
        assert!((c * lambda).abs() < 1e-6, "complementarity residual");
    }

    #[test]
    fn test_slsqp_inequality_fd_jacobian_fallback() {
        // Same problem with no analytic Jacobian: the finite-difference
        // fallback must be used instead of silently dropping the constraint.
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: None,
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = SlsqpConfig::default();
        let result = slsqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_slsqp_curved_constraint_lagrangian_bfgs() {
        // min  x + y   s.t.  x^2 + y^2 <= 1  →  x = y = -1/√2, f = -√2
        // (constraint active at the optimum).
        //
        // The constraint is nonlinear (∇²L ≠ ∇²f), so the damped-BFGS
        // update only builds the right curvature when y is the Lagrangian
        // gradient difference ∇L(x⁺, λ⁺) − ∇L(x, λ⁺) at the QP
        // multipliers (Kraft's SLSQP), not the objective gradient
        // difference.
        let f = |x: &[f64]| x[0] + x[1];
        let g = |_: &[f64]| vec![1.0, 1.0];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![1.0 - x[0] * x[0] - x[1] * x[1]]),
            jac: Some(Box::new(|x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![-2.0 * x[0], -2.0 * x[1]],
                )
            })),
        }];
        let x0 = sonium_solvers_core::array![0.5, 0.5]; // strictly feasible start
        let config = SlsqpConfig::default();
        let result = slsqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(result.x[0], -s, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], -s, epsilon = 1e-4);
        assert_relative_eq!(result.fun, -std::f64::consts::SQRT_2, epsilon = 1e-4);
    }

    #[test]
    fn test_slsqp_bounds() {
        // min (x - 3)^2  s.t.  0 <= x <= 1  →  x = 1.
        let f = |x: &[f64]| (x[0] - 3.0) * (x[0] - 3.0);
        let g = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 3.0)];
        let constraints: Vec<Constraint<f64>> = vec![];
        let x0 = sonium_solvers_core::array![0.5];
        let config = SlsqpConfig {
            bounds: vec![Bound::both(0.0, 1.0)],
            ..SlsqpConfig::default()
        };
        let result = slsqp(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
    }
}
