//! Powell's conjugate direction method with Brent line search.
//!
//! Derivative-free. Iteratively minimises along a set of conjugate directions,
//! updating them to build conjugacy without computing gradients.

use std::cell::Cell;

use crate::line_search::{BrentConfig, BrentStatus, brent_minimize};
use crate::types::{OptimError, OptimResult, OptimStatus, RealScalar};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for Powell's method.
#[deprecated(
    note = "superseded by bobyqa (model-based DFO) and cma_es (multimodal); kept for API compatibility"
)]
#[derive(Debug, Clone)]
pub struct PowellConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub max_fun_evals: usize,
    pub fun_tolerance: T,
    pub x_tolerance: T,
}

#[allow(deprecated)]
impl Default for PowellConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            max_fun_evals: 100_000,
            fun_tolerance: 1e-8,
            x_tolerance: 1e-8,
        }
    }
}

#[allow(deprecated)]
impl Default for PowellConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            max_fun_evals: 100_000,
            fun_tolerance: 1e-4,
            x_tolerance: 1e-4,
        }
    }
}

/// Minimise `f(x)` using Powell's direction-set method.
#[deprecated(
    note = "superseded by bobyqa (model-based DFO) and cma_es (multimodal); kept for API compatibility"
)]
#[allow(deprecated)]
pub fn powell<T, F>(f: &F, x0: &[T], config: &PowellConfig<T>) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    super::validate_objective_start(f, x0)?;
    let callback_non_finite = Cell::new(false);
    let checked_f = |point: &[T]| {
        let value = f(point);
        if !value.is_finite() {
            callback_non_finite.set(true);
        }
        value
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        powell_impl(&checked_f, x0, config)
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "Powell callback".into(),
    })??;
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: "Powell callback returned a non-finite objective".into(),
        });
    }
    super::validate_result(&result)?;
    Ok(result)
}

#[allow(deprecated)]
fn powell_impl<T, F>(
    f: &F,
    x0: &[T],
    config: &PowellConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let n = x0.len();
    let two = T::from_f64(2.0).unwrap();

    // Initial direction set: coordinate axes
    let mut directions: Vec<Vec<T>> = (0..n)
        .map(|i| {
            let mut d = vec![T::zero(); n];
            d[i] = T::one();
            d
        })
        .collect();

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut fun_evals = 1;

    let brent_config = BrentConfig {
        tol: config.x_tolerance,
        max_iterations: 200,
    };

    for iter in 0..config.max_iterations {
        let fx_start = fx;
        let x_start = x.clone();
        let mut biggest_decrease = T::zero();
        let mut biggest_decrease_idx = 0;

        // Minimise along each direction
        for (i, dir) in directions.iter().enumerate() {
            if fun_evals >= config.max_fun_evals {
                return Ok(OptimResult {
                    x,
                    fun: fx,
                    grad: None,
                    iterations: iter,
                    fun_evals,
                    grad_evals: 0,
                    hess_evals: 0,
                    grad_norm: T::infinity(),
                    converged: false,
                    status: OptimStatus::MaxFunctionEvals,
                    kkt: None,
                });
            }

            let f_before = fx;
            let (x_new, fx_new, evals, line_status) = line_min(f, &x, dir, fx, &brent_config);
            fun_evals += evals;
            x = x_new;
            fx = fx_new;
            if line_status != BrentStatus::Converged {
                let status = match line_status {
                    BrentStatus::MaxIterations => OptimStatus::MaxIterationsReached,
                    BrentStatus::NumericalError => OptimStatus::NumericalError,
                    BrentStatus::Converged => unreachable!(),
                };
                return Ok(OptimResult {
                    x,
                    fun: fx,
                    grad: None,
                    iterations: iter + 1,
                    fun_evals,
                    grad_evals: 0,
                    hess_evals: 0,
                    grad_norm: T::infinity(),
                    converged: false,
                    status,
                    kkt: None,
                });
            }

            let decrease = f_before - fx;
            if decrease > biggest_decrease {
                biggest_decrease = decrease;
                biggest_decrease_idx = i;
            }
        }

        // Convergence checks
        if two * (fx_start - fx).abs()
            <= config.fun_tolerance * (fx_start.abs() + fx.abs() + T::machine_epsilon())
        {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: None,
                iterations: iter + 1,
                fun_evals,
                grad_evals: 0,
                hess_evals: 0,
                grad_norm: T::infinity(),
                converged: true,
                status: OptimStatus::FunctionTolerance,
                kkt: None,
            });
        }

        let step_norm = T::vec_norm(&(sonium_solvers_core::vec_ops::vec_sub(&x, &x_start)));
        if step_norm < config.x_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: None,
                iterations: iter + 1,
                fun_evals,
                grad_evals: 0,
                hess_evals: 0,
                grad_norm: T::infinity(),
                converged: true,
                status: OptimStatus::StepTolerance,
                kkt: None,
            });
        }

        // Extrapolated point
        let x_e = sonium_solvers_core::vec_ops::vec_add(
            &x,
            &sonium_solvers_core::vec_ops::vec_sub(&x, &x_start),
        );
        let f_e = f(&x_e);
        fun_evals += 1;

        // Update direction set (Powell's heuristic)
        if f_e < fx_start {
            let t = two * (fx_start - two * fx + f_e) * (fx_start - fx - biggest_decrease).powi(2)
                - biggest_decrease * (fx_start - f_e).powi(2);
            if t < T::zero() {
                // Replace direction with largest decrease
                let new_dir = sonium_solvers_core::vec_ops::vec_sub(&x, &x_start);
                let norm = T::vec_norm(&new_dir);
                if norm > T::machine_epsilon() {
                    directions[biggest_decrease_idx] =
                        new_dir.iter().map(|&d| d / norm).collect::<Vec<_>>();
                    // Minimise along the new direction
                    let (x_new, fx_new, evals, line_status) =
                        line_min(f, &x, &directions[biggest_decrease_idx], fx, &brent_config);
                    fun_evals += evals;
                    x = x_new;
                    fx = fx_new;
                    if line_status != BrentStatus::Converged {
                        let status = match line_status {
                            BrentStatus::MaxIterations => OptimStatus::MaxIterationsReached,
                            BrentStatus::NumericalError => OptimStatus::NumericalError,
                            BrentStatus::Converged => unreachable!(),
                        };
                        return Ok(OptimResult {
                            x,
                            fun: fx,
                            grad: None,
                            iterations: iter + 1,
                            fun_evals,
                            grad_evals: 0,
                            hess_evals: 0,
                            grad_norm: T::infinity(),
                            converged: false,
                            status,
                            kkt: None,
                        });
                    }
                }
            }
        }
    }

    Ok(OptimResult {
        x,
        fun: fx,
        grad: None,
        iterations: config.max_iterations,
        fun_evals,
        grad_evals: 0,
        hess_evals: 0,
        grad_norm: T::infinity(),
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
}

/// 1D line minimisation: minimise `f(x + t * d)` over `t` using Brent.
fn line_min<T, F>(
    f: &F,
    x: &[T],
    d: &[T],
    fx: T,
    brent_config: &BrentConfig<T>,
) -> (Vec<T>, T, usize, BrentStatus)
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    // Bracket the minimum along the direction
    let (a, b) = bracket_minimum(f, x, d, fx);

    let phi = |t: T| {
        let x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, t);
        f(&x_new)
    };

    let result = brent_minimize(&phi, a, b, brent_config);
    let x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, result.x_min);

    (x_new, result.f_min, result.fun_evals, result.status)
}

/// Bracket search: find a, b such that the minimum lies in [a, b].
///
/// Uses a three-point bracket strategy: we look for three points a < b < c
/// such that f(b) < f(a) and f(b) < f(c), then return (a, c).
/// If the initial step is too large, we try progressively smaller steps.
fn bracket_minimum<T, F>(f: &F, x: &[T], d: &[T], fx: T) -> (T, T)
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let golden = T::from_f64(std::f64::consts::GOLDEN_RATIO).unwrap();

    // Try different initial step sizes until we find a descent direction
    let initial_steps = [1.0, 0.1, 0.01, 0.001];
    let mut last_step = T::from_f64(0.001).unwrap();
    for &step_f64 in &initial_steps {
        let step = T::from_f64(step_f64).unwrap();
        last_step = step;

        // Try positive direction
        let b_pos = step;
        let x_pos = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, b_pos);
        let f_pos = f(&x_pos);

        // Try negative direction
        let b_neg = -step;
        let x_neg = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, b_neg);
        let f_neg = f(&x_neg);

        // Pick the downhill direction (if any)
        let (mut a, mut fa, mut b, mut fb) = if f_pos < fx && f_pos <= f_neg {
            (T::zero(), fx, b_pos, f_pos)
        } else if f_neg < fx {
            (T::zero(), fx, b_neg, f_neg)
        } else {
            // Both directions uphill at this step size; try a smaller step
            continue;
        };

        // Expand until we pass the minimum
        let max_expand = 50;
        for _ in 0..max_expand {
            if fb >= fa {
                return if a < b { (a, b) } else { (b, a) };
            }
            let c = b + golden * (b - a);
            a = b;
            fa = fb;
            b = c;
            let x_b = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, b);
            fb = f(&x_b);
        }

        return if a < b { (a, b) } else { (b, a) };
    }

    // All step sizes uphill: return a small bracket around zero
    (-last_step, last_step)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_powell_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = PowellConfig::default();
        let result = powell(&f, &x0, &config).unwrap();
        assert!(result.converged);
        assert!(result.fun < 1e-12);
    }

    #[test]
    fn powell_rejects_non_finite_generated_objective() {
        let objective_calls = std::cell::Cell::new(0);
        let f = |_x: &[f64]| {
            let calls = objective_calls.get() + 1;
            objective_calls.set(calls);
            if calls <= 1 { 0.0 } else { f64::NAN }
        };
        let config = PowellConfig {
            max_iterations: 1,
            ..PowellConfig::default()
        };

        let error =
            powell(&f, &[0.0], &config).expect_err("generated non-finite objective must fail");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn test_powell_rosenbrock() {
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let x0 = sonium_solvers_core::array![-1.2, 1.0];
        let config = PowellConfig {
            max_iterations: 5000,
            ..Default::default()
        };
        let result = powell(&f, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_powell_booth() {
        let f = |x: &[f64]| {
            let a = x[0] + 2.0 * x[1] - 7.0;
            let b = 2.0 * x[0] + x[1] - 5.0;
            a * a + b * b
        };
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = PowellConfig::default();
        let result = powell(&f, &x0, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_powell_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: PowellConfig<f32> = PowellConfig::default();
        let result = powell(&f, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-3, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1]={}", result.x[1]);
    }
}
