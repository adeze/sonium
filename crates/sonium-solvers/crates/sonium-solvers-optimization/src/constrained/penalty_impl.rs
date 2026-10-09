//! Penalty methods: quadratic penalty, barrier, and log-barrier.
//!
//! Converts a constrained problem into a sequence of unconstrained subproblems
//! by adding a penalty term for constraint violations.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::constrained::lbfgsb_impl::{LbfgsbConfig, lbfgsb};
use crate::types::{
    Bound, Constraint, ConstraintKind, OptimError, OptimResult, OptimStatus, RealScalar,
    validate_bounds,
};
use crate::unconstrained::lbfgs_impl::{LbfgsConfig, lbfgs};
use crate::util::finite_diff::jacobian_central_auto;
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Penalty type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyType {
    /// Quadratic penalty: `mu/2 * sum(max(0, -c_i(x))^2 + h_i(x)^2)`
    Quadratic,
    /// Barrier: `-sum(1/c_i(x))` (interior point, requires feasible start)
    Barrier,
    /// Log-barrier: `-sum(log(c_i(x)))` (interior point, requires feasible start)
    LogBarrier,
}

/// Configuration for penalty methods.
#[derive(Debug, Clone)]
pub struct PenaltyConfig<T: RealScalar> {
    pub penalty_type: PenaltyType,
    pub max_outer_iterations: usize,
    pub max_inner_iterations: usize,
    /// Initial penalty parameter.
    pub initial_penalty: T,
    /// Penalty increase factor per outer iteration.
    pub penalty_factor: T,
    pub grad_tolerance: T,
    /// Constraint violation tolerance.
    pub constraint_tolerance: T,
    /// Optional bounds.
    pub bounds: Vec<Bound<T>>,
}

impl Default for PenaltyConfig<f64> {
    fn default() -> Self {
        Self {
            penalty_type: PenaltyType::Quadratic,
            max_outer_iterations: 50,
            max_inner_iterations: 500,
            initial_penalty: 1.0,
            penalty_factor: 10.0,
            grad_tolerance: 1e-6,
            constraint_tolerance: 1e-6,
            bounds: Vec::new(),
        }
    }
}

impl Default for PenaltyConfig<f32> {
    fn default() -> Self {
        Self {
            penalty_type: PenaltyType::Quadratic,
            max_outer_iterations: 50,
            max_inner_iterations: 500,
            initial_penalty: 1.0,
            penalty_factor: 10.0,
            grad_tolerance: 1e-4,
            constraint_tolerance: 1e-4,
            bounds: Vec::new(),
        }
    }
}

/// Minimise `f(x)` subject to constraints using a penalty method.
pub fn penalty_method<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &PenaltyConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    super::validate_optimizer_start(f, grad_fn, x0, constraints)?;
    validate_bounds(&config.bounds, x0.len())?;
    let mut result = penalty_method_impl(f, grad_fn, x0, constraints, config)?;
    super::finalize_identity_kkt(
        &mut result,
        constraints,
        &config.bounds,
        config.grad_tolerance.max(config.constraint_tolerance) * T::from_f64(100.0).unwrap(),
    )?;
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn penalty_method_impl<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &PenaltyConfig<T>,
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

    for outer in 0..config.max_outer_iterations {
        let callback_non_finite = Cell::new(false);
        // Build penalised objective
        let penalised_f = |xp: &[T]| -> T {
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
                match (c.kind, config.penalty_type) {
                    (ConstraintKind::Equality, _) => {
                        val += half
                            * mu
                            * sonium_solvers_core::traits::ComplexField::vec_dot(&cv, &cv);
                    }
                    (ConstraintKind::Inequality, PenaltyType::Quadratic) => {
                        for &ci in cv.iter() {
                            if ci < T::zero() {
                                val += half * mu * ci * ci;
                            }
                        }
                    }
                    (ConstraintKind::Inequality, PenaltyType::LogBarrier) => {
                        for &ci in cv.iter() {
                            if ci > T::machine_epsilon() {
                                val -= ci.ln() / mu;
                            } else {
                                val += T::from_f64(1e20).unwrap();
                            }
                        }
                    }
                    (ConstraintKind::Inequality, PenaltyType::Barrier) => {
                        for &ci in cv.iter() {
                            if ci > T::machine_epsilon() {
                                val += T::one() / (mu * ci);
                            } else {
                                val += T::from_f64(1e20).unwrap();
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

        let penalised_grad = |xp: &[T]| -> Vec<T> {
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
                match (c.kind, config.penalty_type) {
                    (ConstraintKind::Equality, _) => {
                        // grad += mu * J^T * c
                        for k in 0..c.dim {
                            for j in 0..grad.len() {
                                grad[j] += mu * jac[[k, j]] * cv[k];
                            }
                        }
                    }
                    (ConstraintKind::Inequality, PenaltyType::Quadratic) => {
                        for k in 0..c.dim {
                            if cv[k] < T::zero() {
                                for j in 0..grad.len() {
                                    grad[j] += mu * jac[[k, j]] * cv[k];
                                }
                            }
                        }
                    }
                    (ConstraintKind::Inequality, PenaltyType::LogBarrier) => {
                        for k in 0..c.dim {
                            if cv[k] > T::machine_epsilon() {
                                for j in 0..grad.len() {
                                    grad[j] -= jac[[k, j]] / (mu * cv[k]);
                                }
                            }
                        }
                    }
                    (ConstraintKind::Inequality, PenaltyType::Barrier) => {
                        for k in 0..c.dim {
                            if cv[k] > T::machine_epsilon() {
                                for j in 0..grad.len() {
                                    grad[j] -= jac[[k, j]] / (mu * cv[k] * cv[k]);
                                }
                            }
                        }
                    }
                }
            }
            if grad.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
            }
            grad
        };

        // Solve inner subproblem
        let inner_result = if config.bounds.is_empty() {
            let inner_config = LbfgsConfig {
                max_iterations: config.max_inner_iterations,
                grad_tolerance: config.grad_tolerance
                    * T::from_f64(0.1).unwrap().max(T::one() / mu),
                ..Default::default()
            };
            lbfgs(&penalised_f, &penalised_grad, &x, &inner_config)
        } else {
            let inner_config = LbfgsbConfig {
                max_iterations: config.max_inner_iterations,
                grad_tolerance: config.grad_tolerance
                    * T::from_f64(0.1).unwrap().max(T::one() / mu),
                bounds: config.bounds.clone(),
                ..Default::default()
            };
            lbfgsb(&penalised_f, &penalised_grad, &x, &inner_config)
        };
        if callback_non_finite.get() {
            return Err(OptimError::NumericalError {
                iteration: outer,
                message: "penalty callback generated a non-finite value".into(),
            });
        }
        let inner_result = inner_result?;

        total_fun_evals += inner_result.fun_evals;
        total_grad_evals += inner_result.grad_evals;
        x = inner_result.x;

        // Check constraint satisfaction
        let max_violation = compute_max_violation(&x, constraints, outer)?;
        if max_violation < config.constraint_tolerance {
            let fx = checked_objective(f, &x, outer)?;
            let gx = checked_gradient(grad_fn, &x, outer)?;
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

        mu *= config.penalty_factor;
    }

    let fx = checked_objective(f, &x, config.max_outer_iterations)?;
    let gx = checked_gradient(grad_fn, &x, config.max_outer_iterations)?;
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

fn checked_objective<T, F>(f: &F, x: &[T], iteration: usize) -> Result<T, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let value =
        catch_unwind(AssertUnwindSafe(|| f(x))).map_err(|_| OptimError::CallbackPanicked {
            iteration,
            stage: "penalty final objective".into(),
        })?;
    if !value.is_finite() {
        return Err(OptimError::NumericalError {
            iteration,
            message: "penalty objective callback generated a non-finite value".into(),
        });
    }
    Ok(value)
}

fn checked_gradient<T, F>(grad_fn: &F, x: &[T], iteration: usize) -> Result<Vec<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> Vec<T>,
{
    let gradient = catch_unwind(AssertUnwindSafe(|| grad_fn(x))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: "penalty final gradient".into(),
        }
    })?;
    if gradient.len() != x.len() || gradient.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: "penalty gradient callback generated invalid data".into(),
        });
    }
    Ok(gradient)
}

fn compute_max_violation<T: RealScalar>(
    x: &[T],
    constraints: &[Constraint<T>],
    iteration: usize,
) -> Result<T, OptimError> {
    let mut max_viol = T::zero();
    for (index, c) in constraints.iter().enumerate() {
        let cv = super::qp_active_set::checked_constraint_values(c, x, index, iteration)?;
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
    Ok(max_viol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn penalty_rejects_nonfinite_final_objective_without_success() {
        let objective_calls = Cell::new(0);
        let objective = |_: &[f64]| {
            let call = objective_calls.get();
            objective_calls.set(call + 1);
            if call == 0 { 0.0 } else { f64::NAN }
        };
        let gradient = |_: &[f64]| vec![0.0];
        let config = PenaltyConfig {
            max_outer_iterations: 0,
            max_inner_iterations: 0,
            ..PenaltyConfig::default()
        };

        let error = penalty_method(&objective, &gradient, &[0.0], &[], &config)
            .expect_err("non-finite final objective must not produce success");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn test_penalty_equality_constraint() {
        // min x^2 + y^2 s.t. x + y = 1
        // Solution: (0.5, 0.5), f = 0.5
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
        let config = PenaltyConfig::default();
        let result = penalty_method(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_penalty_inequality_constraint() {
        // min (x-2)^2 + (y-2)^2 s.t. x + y <= 2  (i.e. 2 - x - y >= 0)
        // Solution: (1, 1), f = 2
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
        let config = PenaltyConfig::default();
        let result = penalty_method(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_penalty_equality_f32() {
        // min x^2 + y^2 s.t. x + y = 1
        // Solution: (0.5, 0.5), f = 0.5
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
        let config: PenaltyConfig<f32> = PenaltyConfig::default();
        let result = penalty_method(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!((result.x[0] - 0.5_f32).abs() < 1e-1, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 0.5_f32).abs() < 1e-1, "x[1]={}", result.x[1]);
    }

    #[test]
    fn test_penalty_no_jac_matches_analytic() {
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
        let config = PenaltyConfig::default();
        let r_analytic = penalty_method(&f, &g, &x0, &with_jac, &config).unwrap();
        let r_fd = penalty_method(&f, &g, &x0, &without_jac, &config).unwrap();
        assert!(r_fd.converged, "status: {:?}", r_fd.status);
        assert_relative_eq!(r_fd.x[0], r_analytic.x[0], epsilon = 1e-3);
        assert_relative_eq!(r_fd.x[1], r_analytic.x[1], epsilon = 1e-3);
        assert_relative_eq!(r_fd.x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(r_fd.x[1], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_penalty_inequality_no_jac_binds() {
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
        let config = PenaltyConfig::default();
        let result = penalty_method(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
        // The constraint is active at the solution (binds), not ignored.
        assert!(result.x[0] + result.x[1] <= 2.0 + 1e-3);
    }

    #[test]
    fn test_penalty_log_barrier() {
        // min x^2 + y^2 s.t. x + y >= 1 (i.e. x + y - 1 >= 0)
        // Solution: (0.5, 0.5), f = 0.5.
        // Start from a feasible point (1, 1).
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
        let x0 = sonium_solvers_core::array![1.0, 1.0]; // feasible starting point
        let config = PenaltyConfig {
            penalty_type: PenaltyType::LogBarrier,
            initial_penalty: 10.0,
            penalty_factor: 10.0,
            max_outer_iterations: 100,
            max_inner_iterations: 1000,
            constraint_tolerance: 1e-6,
            ..Default::default()
        };
        let result = penalty_method(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(
            result.converged,
            "LogBarrier did not converge: status={:?}, x={:?}, fun={}",
            result.status, result.x, result.fun
        );
        // Log-barrier converges to the constrained optimum as mu -> inf.
        // The constraint x + y >= 1 should be approximately active at (0.5, 0.5).
        assert_relative_eq!(result.x[0], 0.5, epsilon = 0.1);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 0.1);
    }
}
