//! Augmented Lagrangian method with adaptive penalty.
//!
//! Outer loop updates Lagrange multipliers and penalty parameter.
//! Inner loop solves an unconstrained (or bound-constrained) subproblem.

use std::cell::Cell;

use crate::constrained::lbfgsb_impl::{LbfgsbConfig, lbfgsb};
use crate::types::{
    Bound, Constraint, ConstraintKind, OptimError, OptimResult, OptimStatus, RealScalar,
    validate_bounds,
};
use crate::unconstrained::lbfgs_impl::{LbfgsConfig, lbfgs};
use crate::util::finite_diff::jacobian_central_auto;
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Inner solver for the augmented Lagrangian subproblem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnerSolver {
    Lbfgs,
    Lbfgsb,
}

/// Configuration for the augmented Lagrangian method.
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianConfig<T: RealScalar> {
    pub max_outer_iterations: usize,
    pub max_inner_iterations: usize,
    pub grad_tolerance: T,
    pub constraint_tolerance: T,
    pub initial_penalty: T,
    pub penalty_factor: T,
    pub inner_solver: InnerSolver,
    pub bounds: Vec<Bound<T>>,
}

impl<T: RealScalar> Default for AugmentedLagrangianConfig<T> {
    fn default() -> Self {
        Self {
            max_outer_iterations: 100,
            max_inner_iterations: 500,
            grad_tolerance: T::from_f64(1e-6).unwrap(),
            constraint_tolerance: T::from_f64(1e-6).unwrap(),
            initial_penalty: T::from_f64(10.0).unwrap(),
            penalty_factor: T::from_f64(10.0).unwrap(),
            inner_solver: InnerSolver::Lbfgs,
            bounds: Vec::new(),
        }
    }
}

/// Minimise `f(x)` subject to constraints using the augmented Lagrangian method.
pub fn augmented_lagrangian<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &AugmentedLagrangianConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    super::validate_optimizer_start(f, grad_fn, x0, constraints)?;
    validate_bounds(&config.bounds, x0.len())?;
    let mut result = augmented_lagrangian_impl(f, grad_fn, x0, constraints, config)?;
    super::finalize_identity_kkt(
        &mut result,
        constraints,
        &config.bounds,
        config.grad_tolerance.max(config.constraint_tolerance) * T::from_f64(100.0).unwrap(),
    )?;
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn augmented_lagrangian_impl<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &AugmentedLagrangianConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    let mut x = x0.to_vec();
    let mut mu = config.initial_penalty;
    let mut total_fun_evals = 0_usize;
    let mut total_grad_evals = 0_usize;
    let half = T::from_f64(0.5).unwrap();

    // Initialise Lagrange multipliers
    let total_constraint_dim: usize = constraints.iter().map(|c| c.dim).sum();
    let mut lambda = vec![T::zero(); total_constraint_dim];

    let mut prev_violation = T::infinity();

    for outer in 0..config.max_outer_iterations {
        let callback_non_finite = Cell::new(false);
        // Build augmented Lagrangian
        let aug_f = |xp: &[T]| -> T {
            let mut val = f(xp);
            if !val.is_finite() {
                callback_non_finite.set(true);
            }
            let mut idx = 0;
            for c in constraints {
                let cv = (c.fun)(xp);
                if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
                    callback_non_finite.set(true);
                    return T::nan();
                }
                for &cvk in &cv {
                    match c.kind {
                        ConstraintKind::Equality => {
                            val = val + lambda[idx] * cvk + half * mu * cvk * cvk;
                        }
                        ConstraintKind::Inequality => {
                            // AL for inequality: use shifted constraint
                            let shifted = cvk - lambda[idx] / mu;
                            if shifted < T::zero() {
                                val = val + half * mu * shifted * shifted
                                    - half * lambda[idx] * lambda[idx] / mu;
                            }
                        }
                    }
                    idx += 1;
                }
            }
            if !val.is_finite() {
                callback_non_finite.set(true);
            }
            val
        };

        let aug_grad = |xp: &[T]| -> Vec<T> {
            let mut grad = grad_fn(xp);
            if grad.len() != xp.len() || grad.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                return vec![T::nan(); xp.len()];
            }
            let mut idx = 0;
            for c in constraints {
                let cv = (c.fun)(xp);
                if cv.len() != c.dim || cv.iter().any(|value| !value.is_finite()) {
                    callback_non_finite.set(true);
                    return vec![T::nan(); xp.len()];
                }
                // Use the analytic Jacobian when provided; otherwise fall back
                // to central finite differences so the constraint still binds.
                let jac = match c.jac {
                    Some(ref jac_fn) => jac_fn(xp),
                    None => jacobian_central_auto(&c.fun, xp, c.dim),
                };
                if jac.nrows() != c.dim
                    || jac.ncols() != xp.len()
                    || (0..jac.nrows())
                        .any(|row| (0..jac.ncols()).any(|column| !jac[[row, column]].is_finite()))
                {
                    callback_non_finite.set(true);
                    return vec![T::nan(); xp.len()];
                }
                for k in 0..c.dim {
                    match c.kind {
                        ConstraintKind::Equality => {
                            let factor = lambda[idx] + mu * cv[k];
                            for j in 0..grad.len() {
                                grad[j] += factor * jac[[k, j]];
                            }
                        }
                        ConstraintKind::Inequality => {
                            let shifted = cv[k] - lambda[idx] / mu;
                            if shifted < T::zero() {
                                let factor = mu * shifted;
                                for j in 0..grad.len() {
                                    grad[j] += factor * jac[[k, j]];
                                }
                            }
                        }
                    }
                    idx += 1;
                }
            }
            if grad.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
            }
            grad
        };

        // Solve inner subproblem
        let inner_result = match config.inner_solver {
            InnerSolver::Lbfgs => {
                let inner_config = LbfgsConfig {
                    max_iterations: config.max_inner_iterations,
                    grad_tolerance: config.grad_tolerance / mu.max(T::one()),
                    ..Default::default()
                };
                lbfgs(&aug_f, &aug_grad, &x, &inner_config)
            }
            InnerSolver::Lbfgsb => {
                let inner_config = LbfgsbConfig {
                    max_iterations: config.max_inner_iterations,
                    grad_tolerance: config.grad_tolerance / mu.max(T::one()),
                    bounds: config.bounds.clone(),
                    ..Default::default()
                };
                lbfgsb(&aug_f, &aug_grad, &x, &inner_config)
            }
        };
        if callback_non_finite.get() {
            return Err(OptimError::NumericalError {
                iteration: outer,
                message: "augmented-Lagrangian callback generated a non-finite value".into(),
            });
        }
        let inner_result = inner_result?;

        total_fun_evals += inner_result.fun_evals;
        total_grad_evals += inner_result.grad_evals;
        x = inner_result.x;

        // Update Lagrange multipliers
        let mut max_violation = T::zero();
        let mut idx = 0;
        for (constraint_index, c) in constraints.iter().enumerate() {
            let cv =
                super::qp_active_set::checked_constraint_values(c, &x, constraint_index, outer)?;
            for &cvk in &cv {
                match c.kind {
                    ConstraintKind::Equality => {
                        lambda[idx] += mu * cvk;
                        let viol = cvk.abs();
                        if viol > max_violation {
                            max_violation = viol;
                        }
                    }
                    ConstraintKind::Inequality => {
                        lambda[idx] = T::max(T::zero(), lambda[idx] - mu * cvk);
                        let viol = (-cvk).max(T::zero());
                        if viol > max_violation {
                            max_violation = viol;
                        }
                    }
                }
                idx += 1;
            }
        }

        // Check convergence
        if max_violation < config.constraint_tolerance {
            let fx = f(&x);
            let gx = grad_fn(&x);
            let grad_norm = gx.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
            let mut result = OptimResult {
                x: x.clone(),
                fun: fx,
                grad: Some(gx),
                iterations: outer + 1,
                fun_evals: total_fun_evals,
                grad_evals: total_grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            };
            super::attach_identity_kkt(&mut result, constraints, &config.bounds)?;
            let kkt_tolerance = config.grad_tolerance.max(config.constraint_tolerance)
                * T::from_f64(100.0).unwrap();
            if result
                .kkt
                .as_ref()
                .is_some_and(|report| report.norm <= kkt_tolerance)
            {
                return Ok(result);
            }
        }

        // Adaptive penalty: increase mu if constraint violation didn't decrease enough
        if max_violation > T::from_f64(0.25).unwrap() * prev_violation {
            mu *= config.penalty_factor;
        }
        prev_violation = max_violation;
    }

    let fx = f(&x);
    let gx = grad_fn(&x);
    let grad_norm = gx.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
    Ok(OptimResult {
        x,
        fun: fx,
        grad: Some(gx),
        iterations: config.max_outer_iterations,
        fun_evals: total_fun_evals,
        grad_evals: total_grad_evals,
        hess_evals: 0,
        grad_norm,
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_augmented_lagrangian_equality() {
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
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = AugmentedLagrangianConfig::default();
        let result = augmented_lagrangian(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_augmented_lagrangian_inequality() {
        // min (x-2)^2 + (y-2)^2 s.t. x + y <= 2 (i.e. 2 - x - y >= 0)
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
        let g = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![2.0 - x[0] - x[1]]),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![-1.0, -1.0])
            })),
        }];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = AugmentedLagrangianConfig::default();
        let result = augmented_lagrangian(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_augmented_lagrangian_no_jac_idx_advance() {
        // Regression: constraints without Jacobian must still advance the
        // multiplier index so subsequent constraints use the correct lambda.
        // With the finite-difference fallback, a Jacobian-free constraint now
        // binds instead of being silently dropped.
        // Two equality constraints: first has no Jacobian, second has one.
        // min x^2 + y^2 s.t. x - 0.3 = 0 AND y - 0.7 = 0
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let constraints = vec![
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] - 0.3]),
                jac: None, // no Jacobian — FD fallback kicks in
            },
            Constraint {
                kind: ConstraintKind::Equality,
                dim: 1,
                fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[1] - 0.7]),
                jac: Some(Box::new(|_x: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        2,
                        vec![0.0, 1.0],
                    )
                })),
            },
        ];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = AugmentedLagrangianConfig::default();
        let result = augmented_lagrangian(&f, &g, &x0, &constraints, &config).unwrap();
        // Both constraints must bind: x = 0.3 via FD, y = 0.7 via analytic jac.
        assert_relative_eq!(result.x[0], 0.3, epsilon = 5e-2);
        assert_relative_eq!(result.x[1], 0.7, epsilon = 5e-2);
    }

    #[test]
    fn test_augmented_lagrangian_no_jac_matches_analytic() {
        // Same KKT point with and without analytic Jacobians.
        // min x^2 + y^2 s.t. x + y = 1 -> (0.5, 0.5)
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let cfun = |x: &[f64]| sonium_solvers_core::array![x[0] + x[1] - 1.0];
        let with_jac = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(cfun),
            jac: Some(Box::new(|_x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }];
        let without_jac = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(cfun),
            jac: None,
        }];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = AugmentedLagrangianConfig::default();
        let r_analytic = augmented_lagrangian(&f, &g, &x0, &with_jac, &config).unwrap();
        let r_fd = augmented_lagrangian(&f, &g, &x0, &without_jac, &config).unwrap();
        assert!(r_fd.converged, "status: {:?}", r_fd.status);
        assert_relative_eq!(r_fd.x[0], r_analytic.x[0], epsilon = 1e-3);
        assert_relative_eq!(r_fd.x[1], r_analytic.x[1], epsilon = 1e-3);
        assert_relative_eq!(r_fd.x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(r_fd.x[1], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_augmented_lagrangian_inequality_no_jac_binds() {
        // A Jacobian-free inequality constraint must actually bind.
        // min (x-2)^2 + (y-2)^2 s.t. x + y <= 2 (i.e. 2 - x - y >= 0)
        // Unconstrained minimum (2,2) is infeasible; constrained optimum (1,1).
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
        let g = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![2.0 - x[0] - x[1]]),
            jac: None,
        }];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = AugmentedLagrangianConfig::default();
        let result = augmented_lagrangian(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
        // The constraint is active at the solution (binds), not ignored.
        assert!(result.x[0] + result.x[1] <= 2.0 + 1e-4);
    }

    #[test]
    fn test_augmented_lagrangian_equality_f32() {
        // min x^2 + y^2 s.t. x + y = 1
        let f = |x: &[f32]| x.dot(x);
        let g = |x: &[f32]| x.iter().map(|&xi| 2.0_f32 * xi).collect::<Vec<_>>();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f32]| sonium_solvers_core::array![x[0] + x[1] - 1.0_f32]),
            jac: Some(Box::new(|_x: &[f32]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![1.0_f32, 1.0_f32],
                )
            })),
        }];
        let x0 = sonium_solvers_core::array![0.0_f32, 0.0_f32];
        let config: AugmentedLagrangianConfig<f32> = AugmentedLagrangianConfig::default();
        let result = augmented_lagrangian(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!((result.x[0] - 0.5_f32).abs() < 1e-1, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 0.5_f32).abs() < 1e-1, "x[1]={}", result.x[1]);
    }
}
