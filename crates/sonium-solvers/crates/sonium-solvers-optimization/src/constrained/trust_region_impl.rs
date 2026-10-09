//! Trust Region Constrained (TRCON) algorithm.
//!
//! Solves constrained optimisation using a trust-region framework with
//! penalty-based constraint handling.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{Constraint, ConstraintKind, OptimError, OptimResult, OptimStatus, RealScalar};
use crate::util::trust_region_subproblem::{
    TrustRegionSubproblemSolver, more_sorensen, steihaug_cg,
};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for trust-region constrained.
#[derive(Debug, Clone)]
pub struct TrustRegionConstrainedConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub constraint_tolerance: T,
    pub initial_trust_radius: T,
    pub max_trust_radius: T,
    /// Initial quadratic penalty parameter mu.
    pub initial_penalty: T,
    /// Cap for the adaptive penalty parameter.
    pub max_penalty: T,
    /// Geometric multiplier applied to mu when constraint violation stagnates.
    pub penalty_increase: T,
    /// Mu is increased when the max constraint violation fails to decrease by
    /// at least this factor between iterations (Nocedal-Wright style penalty
    /// steering, Framework 17.2).
    pub violation_decrease_factor: T,
    /// How the trust-region subproblem is solved.
    pub subproblem_solver: TrustRegionSubproblemSolver,
}

impl Default for TrustRegionConstrainedConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            grad_tolerance: 1e-6,
            constraint_tolerance: 1e-6,
            initial_trust_radius: 1.0,
            max_trust_radius: 100.0,
            initial_penalty: 10.0,
            max_penalty: 1e8,
            penalty_increase: 10.0,
            violation_decrease_factor: 0.9,
            subproblem_solver: TrustRegionSubproblemSolver::default(),
        }
    }
}

impl Default for TrustRegionConstrainedConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            grad_tolerance: 1e-4,
            constraint_tolerance: 1e-4,
            initial_trust_radius: 1.0,
            max_trust_radius: 100.0,
            initial_penalty: 10.0,
            max_penalty: 1e6,
            penalty_increase: 10.0,
            violation_decrease_factor: 0.9,
            subproblem_solver: TrustRegionSubproblemSolver::default(),
        }
    }
}

/// Minimise `f(x)` subject to constraints using trust-region method.
///
/// `hess_vec_fn(x, v)` computes the Hessian-vector product `H(x) * v`.
pub fn trust_region_constrained<T, F, Fg, Hv>(
    f: &F,
    grad_fn: &Fg,
    hess_vec_fn: &Hv,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &TrustRegionConstrainedConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    super::validate_optimizer_start(f, grad_fn, x0, constraints)?;
    let zero = vec![T::zero(); x0.len()];
    let hessian = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hess_vec_fn(x0, &zero)))
        .map_err(|_| OptimError::CallbackPanicked {
            iteration: 0,
            stage: "trust-region Hessian-vector preflight".into(),
        })?;
    if hessian.len() != x0.len() {
        return Err(OptimError::DimensionMismatch {
            expected: x0.len(),
            got: hessian.len(),
        });
    }
    if hessian.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "trust-region Hessian-vector callback returned a non-finite value".into(),
        });
    }
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
    let checked_hess_vec = |point: &[T], vector: &[T]| {
        let output = hess_vec_fn(point, vector);
        if output.len() != vector.len() || output.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
        }
        output
    };
    let inner_result = catch_unwind(AssertUnwindSafe(|| {
        trust_region_constrained_impl(
            &checked_f,
            &checked_grad,
            &checked_hess_vec,
            x0,
            constraints,
            config,
            &callback_non_finite,
        )
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "trust-region callback".into(),
    })?;
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "trust-region callback returned a non-finite value".into(),
        });
    }
    let mut result = inner_result?;
    super::finalize_identity_kkt(
        &mut result,
        constraints,
        &[],
        config.grad_tolerance.max(config.constraint_tolerance) * T::from_f64(100.0).unwrap(),
    )?;
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn trust_region_constrained_impl<T, F, Fg, Hv>(
    f: &F,
    grad_fn: &Fg,
    hess_vec_fn: &Hv,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &TrustRegionConstrainedConfig<T>,
    callback_non_finite: &Cell<bool>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    let mut x = x0.to_vec();
    let mut delta = config.initial_trust_radius;
    let mut fun_evals = 0_usize;
    let mut grad_evals = 0_usize;
    let mut hess_evals = 0_usize;
    let mut mu = config.initial_penalty;
    let half = T::from_f64(0.5).unwrap();

    // Penalised objective (takes mu as parameter to avoid borrow issues)
    let merit = |xp: &[T], mu_val: T| -> T {
        let mut val = f(xp);
        if !val.is_finite() {
            callback_non_finite.set(true);
        }
        for c in constraints {
            let cv = (c.fun)(xp);
            if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                return T::nan();
            }
            for &ci in cv.iter() {
                match c.kind {
                    ConstraintKind::Equality => val += half * mu_val * ci * ci,
                    ConstraintKind::Inequality => {
                        if ci < T::zero() {
                            val += half * mu_val * ci * ci;
                        }
                    }
                }
            }
        }
        if !val.is_finite() {
            callback_non_finite.set(true);
        }
        val
    };

    let merit_grad = |xp: &[T], mu_val: T| -> Vec<T> {
        let mut grad = grad_fn(xp);
        if grad.len() != xp.len() || grad.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
            return vec![T::nan(); xp.len()];
        }
        for c in constraints {
            let cv = (c.fun)(xp);
            if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                return vec![T::nan(); xp.len()];
            }
            let jac = match super::qp_active_set::checked_constraint_jacobian(c, xp, xp.len(), 0, 0)
            {
                Ok(jac) => jac,
                Err(_) => {
                    callback_non_finite.set(true);
                    return vec![T::nan(); xp.len()];
                }
            };
            for k in 0..c.dim {
                let factor = match c.kind {
                    ConstraintKind::Equality => mu_val * cv[k],
                    ConstraintKind::Inequality => {
                        if cv[k] < T::zero() {
                            mu_val * cv[k]
                        } else {
                            T::zero()
                        }
                    }
                };
                if factor.abs() > T::machine_epsilon() {
                    for j in 0..grad.len() {
                        grad[j] += factor * jac[[k, j]];
                    }
                }
            }
        }
        if grad.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
        }
        grad
    };

    let mut fx = merit(&x, mu);
    fun_evals += 1;
    let mut gx = merit_grad(&x, mu);
    grad_evals += 1;

    let eta_good = T::from_f64(0.75).unwrap();
    let eta_bad = T::from_f64(0.25).unwrap();
    let mut prev_violation = compute_max_violation(&x, constraints, callback_non_finite);

    for iter in 0..config.max_iterations {
        let grad_norm = gx.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
        let max_violation = compute_max_violation(&x, constraints, callback_non_finite);

        if grad_norm < config.grad_tolerance && max_violation < config.constraint_tolerance {
            let actual_fx = f(&x);
            let final_grad = grad_fn(&x);
            return Ok(OptimResult {
                x,
                fun: actual_fx,
                grad: Some(final_grad),
                iterations: iter,
                fun_evals,
                grad_evals: grad_evals + 1,
                hess_evals,
                grad_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Penalised Hessian-vector product: H_f * v + mu * sum_k (J_k^T (J_k * v))
        let mu_val = mu;
        let penalised_hv = |xp: &[T], v: &[T]| -> Vec<T> {
            let mut hv = hess_vec_fn(xp, v);
            if hv.len() != v.len() || hv.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                return vec![T::nan(); v.len()];
            }
            for c in constraints {
                let jac = match super::qp_active_set::checked_constraint_jacobian(
                    c,
                    xp,
                    xp.len(),
                    0,
                    0,
                ) {
                    Ok(jac) => jac,
                    Err(_) => {
                        callback_non_finite.set(true);
                        return vec![T::nan(); v.len()];
                    }
                };
                let cv = (c.fun)(xp);
                if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
                    callback_non_finite.set(true);
                    return vec![T::nan(); v.len()];
                }
                for k in 0..c.dim {
                    // J_k * v
                    let mut jv = T::zero();
                    for j in 0..v.len() {
                        jv += jac[[k, j]] * v[j];
                    }
                    // hv += mu * (J_k^T * (J_k * v))
                    for j in 0..v.len() {
                        hv[j] += mu_val * jac[[k, j]] * jv;
                    }
                }
            }
            if hv.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
            }
            hv
        };

        // Solve trust region subproblem
        let d = match config.subproblem_solver {
            TrustRegionSubproblemSolver::SteihaugCg => {
                let (d, inner_hess) = steihaug_cg(
                    &gx,
                    &penalised_hv,
                    &x,
                    delta,
                    100,
                    grad_norm * T::from_f64(0.1).unwrap(),
                );
                hess_evals += inner_hess;
                d
            }
            TrustRegionSubproblemSolver::MoreSorensen => {
                // Assemble the dense penalised Hessian column-by-column from
                // Hessian-vector products, then solve the subproblem exactly.
                let n = x.len();
                let mut h_dense = sonium_solvers_core::dense::DenseMatrix::zeros(n, n);
                for j in 0..n {
                    let mut e_j = vec![T::zero(); n];
                    e_j[j] = T::one();
                    let col = penalised_hv(&x, &e_j);
                    hess_evals += 1;
                    for i in 0..n {
                        h_dense[[i, j]] = col[i];
                    }
                }
                more_sorensen(&gx, &h_dense, delta)
            }
        };

        // Predicted reduction
        let hd = penalised_hv(&x, &d);
        let pred = -(T::vec_dot(&gx, &d) + half * T::vec_dot(&d, &hd));

        // Actual reduction
        let x_new = sonium_solvers_core::vec_ops::vec_add(&x, &d);
        let fx_new = merit(&x_new, mu);
        fun_evals += 1;
        let actual = fx - fx_new;

        // Ratio
        let rho = if pred.abs() > T::machine_epsilon() {
            actual / pred
        } else {
            T::zero()
        };

        // Update trust region
        if rho < eta_bad {
            delta *= T::from_f64(0.25).unwrap();
        } else if rho > eta_good && (T::vec_norm(&d) - delta).abs() < T::machine_epsilon() * delta {
            delta = T::min(delta * T::from_f64(2.0).unwrap(), config.max_trust_radius);
        }

        if delta < T::machine_epsilon() {
            let actual_fx = f(&x);
            let final_grad = grad_fn(&x);
            return Ok(OptimResult {
                x,
                fun: actual_fx,
                grad: Some(final_grad),
                iterations: iter,
                fun_evals,
                grad_evals: grad_evals + 1,
                hess_evals,
                grad_norm,
                converged: false,
                status: OptimStatus::TrustRegionCollapsed,
                kkt: None,
            });
        }

        // Accept step
        if rho > eta_bad {
            x = x_new;
            fx = fx_new;
            gx = merit_grad(&x, mu);
            grad_evals += 1;
        }

        // Adaptive penalty steering (Nocedal-Wright Framework 17.2): only
        // increase mu when the constraint violation fails to decrease by at
        // least the configured factor, and never past the cap. This keeps the
        // subproblems well-conditioned while driving the limit point to
        // feasibility instead of to the fixed-penalty minimiser.
        let new_violation = compute_max_violation(&x, constraints, callback_non_finite);
        if new_violation > config.constraint_tolerance
            && new_violation > config.violation_decrease_factor * prev_violation
            && mu < config.max_penalty
        {
            mu = T::min(mu * config.penalty_increase, config.max_penalty);
            fx = merit(&x, mu);
            gx = merit_grad(&x, mu);
            fun_evals += 1;
            grad_evals += 1;
        }
        prev_violation = new_violation;
    }

    let actual_fx = f(&x);
    let gx_final = grad_fn(&x);
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: config.max_iterations,
            message: "trust-region callback returned a non-finite value".into(),
        });
    }
    let grad_norm = gx_final.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
    Ok(OptimResult {
        x,
        fun: actual_fx,
        grad: Some(gx_final),
        iterations: config.max_iterations,
        fun_evals,
        grad_evals: grad_evals + 1,
        hess_evals,
        grad_norm,
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
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

    #[test]
    fn rejects_nonfinite_final_objective_after_finite_preflight() {
        let calls = Cell::new(0_usize);
        let objective = |_: &[f64]| {
            let call = calls.get();
            calls.set(call + 1);
            if call == 0 { 1.0 } else { f64::NAN }
        };
        let gradient = |_: &[f64]| vec![0.0];
        let hessian_vector = |_: &[f64], vector: &[f64]| vec![0.0; vector.len()];
        let config = TrustRegionConstrainedConfig {
            max_iterations: 0,
            ..TrustRegionConstrainedConfig::default()
        };

        let error =
            trust_region_constrained(&objective, &gradient, &hessian_vector, &[0.0], &[], &config)
                .expect_err("non-finite final objective must not be returned");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn rejects_nonfinite_final_gradient_after_finite_preflight() {
        let calls = Cell::new(0_usize);
        let objective = |_: &[f64]| 1.0;
        let gradient = |_: &[f64]| {
            let call = calls.get();
            calls.set(call + 1);
            if call == 0 { vec![0.0] } else { vec![f64::NAN] }
        };
        let hessian_vector = |_: &[f64], vector: &[f64]| vec![0.0; vector.len()];
        let config = TrustRegionConstrainedConfig {
            max_iterations: 0,
            ..TrustRegionConstrainedConfig::default()
        };

        let error =
            trust_region_constrained(&objective, &gradient, &hessian_vector, &[0.0], &[], &config)
                .expect_err("non-finite final gradient must not be returned");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }
    use approx::assert_relative_eq;

    #[test]
    fn test_trust_region_equality() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let hv = |_x: &[f64], v: &[f64]| v.iter().map(|&vi| 2.0 * vi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = TrustRegionConstrainedConfig::default();
        let result = trust_region_constrained(&f, &g, &hv, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 5e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 5e-2);
    }

    /// min (x0-1)^2 + x1^2  s.t.  x0 + x1 = 2; optimum x* = (1.5, 0.5).
    /// With a static penalty mu = 10 the penalty minimiser sits at
    /// x0+x1 = 21/11 ~ 1.909 (violation ~0.09), so the static-mu algorithm
    /// can never satisfy the constraint tolerance.
    #[allow(clippy::type_complexity)]
    fn shifted_quadratic_problem() -> (
        impl Fn(&[f64]) -> f64,
        impl Fn(&[f64]) -> Vec<f64>,
        impl Fn(&[f64], &[f64]) -> Vec<f64>,
    ) {
        let f = |x: &[f64]| (x[0] - 1.0) * (x[0] - 1.0) + x[1] * x[1];
        let g = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 1.0), 2.0 * x[1]];
        let hv = |_x: &[f64], v: &[f64]| sonium_solvers_core::array![2.0 * v[0], 2.0 * v[1]];
        (f, g, hv)
    }

    #[test]
    fn test_static_penalty_converges_infeasible() {
        let (f, g, hv) = shifted_quadratic_problem();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 2.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        // Cap mu at its initial value: this emulates the old static-mu code.
        let config = TrustRegionConstrainedConfig {
            max_penalty: 10.0,
            ..TrustRegionConstrainedConfig::default()
        };
        let result = trust_region_constrained(&f, &g, &hv, &x0, &constraints, &config).unwrap();
        let violation = (result.x[0] + result.x[1] - 2.0).abs();
        assert!(
            !result.converged || violation > 1e-3,
            "static mu should stall at the infeasible penalty minimiser: {:?}, violation {violation}",
            result.status
        );
        assert!(
            violation > 1e-3,
            "static-mu limit point should be infeasible, got violation {violation}"
        );
    }

    #[test]
    fn test_adaptive_penalty_reaches_feasible_optimum() {
        let (f, g, hv) = shifted_quadratic_problem();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 2.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = TrustRegionConstrainedConfig::default();
        let result = trust_region_constrained(&f, &g, &hv, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let violation = (result.x[0] + result.x[1] - 2.0).abs();
        assert!(violation < 1e-6, "violation {violation}");
        assert_relative_eq!(result.x[0], 1.5, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_constraint_jacobian_fd_fallback() {
        // Same problem but without a user Jacobian: the finite-difference
        // fallback must supply it instead of silently dropping the constraint.
        let (f, g, hv) = shifted_quadratic_problem();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 2.0]),
            jac: None,
        }];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = TrustRegionConstrainedConfig::default();
        let result = trust_region_constrained(&f, &g, &hv, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let violation = (result.x[0] + result.x[1] - 2.0).abs();
        assert!(violation < 1e-5, "violation {violation}");
        assert_relative_eq!(result.x[0], 1.5, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_trust_region_more_sorensen_subproblem() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let hv = |_x: &[f64], v: &[f64]| v.iter().map(|&vi| 2.0 * vi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![2.0, 2.0];
        let config = TrustRegionConstrainedConfig {
            subproblem_solver:
                crate::util::trust_region_subproblem::TrustRegionSubproblemSolver::MoreSorensen,
            ..TrustRegionConstrainedConfig::default()
        };
        let result = trust_region_constrained(&f, &g, &hv, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 5e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 5e-2);
    }
}
