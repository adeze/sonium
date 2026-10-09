//! Nelder-Mead simplex method with adaptive parameter selection.
//!
//! Derivative-free minimisation. Suitable for non-smooth or noisy objectives.
//! Adaptive parameters follow Gao & Han (2012).

use crate::types::{OptimError, OptimResult, OptimStatus, RealScalar};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for Nelder-Mead.
#[derive(Debug, Clone)]
pub struct NelderMeadConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub max_fun_evals: usize,
    /// Convergence tolerance on the standard deviation of simplex values.
    pub fun_tolerance: T,
    /// Convergence tolerance on the simplex diameter.
    pub x_tolerance: T,
    /// Use adaptive parameters (Gao & Han 2012) scaled by dimension.
    pub adaptive: bool,
    /// Initial simplex edge length (if `None`, auto-scaled from x0).
    pub initial_simplex_size: Option<T>,
}

impl Default for NelderMeadConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            max_fun_evals: 100_000,
            fun_tolerance: 1e-8,
            x_tolerance: 1e-8,
            adaptive: true,
            initial_simplex_size: None,
        }
    }
}

impl Default for NelderMeadConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            max_fun_evals: 100_000,
            fun_tolerance: 1e-4,
            x_tolerance: 1e-4,
            adaptive: true,
            initial_simplex_size: None,
        }
    }
}

/// Minimise `f(x)` using the Nelder-Mead simplex method.
pub fn nelder_mead<T, F>(
    f: &F,
    x0: &[T],
    config: &NelderMeadConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    super::validate_objective_start(f, x0)?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        nelder_mead_impl(f, x0, config)
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "Nelder-Mead callback".into(),
    })??;
    super::validate_result(&result)?;
    Ok(result)
}

fn nelder_mead_impl<T, F>(
    f: &F,
    x0: &[T],
    config: &NelderMeadConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    if x0.is_empty() {
        return Err(OptimError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if let Some(index) = x0.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("initial point contains a non-finite value at index {index}"),
        });
    }
    if config.max_iterations == 0 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "max_iterations must be greater than zero".to_string(),
        });
    }
    if config.max_fun_evals == 0 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "max_fun_evals must be greater than zero".to_string(),
        });
    }
    if !config.fun_tolerance.is_finite() || config.fun_tolerance <= T::zero() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "fun_tolerance must be finite and positive".to_string(),
        });
    }
    if !config.x_tolerance.is_finite() || config.x_tolerance <= T::zero() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "x_tolerance must be finite and positive".to_string(),
        });
    }
    if let Some(edge) = config.initial_simplex_size
        && (!edge.is_finite() || edge <= T::zero())
    {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "initial_simplex_size must be finite and positive".to_string(),
        });
    }

    let evaluate = |point: &[T], iteration: usize| -> Result<T, OptimError> {
        if let Some(index) = point.iter().position(|value| !value.is_finite()) {
            return Err(OptimError::NumericalError {
                iteration,
                message: format!("simplex point contains a non-finite value at index {index}"),
            });
        }
        let value = f(point);
        if !value.is_finite() {
            return Err(OptimError::NumericalError {
                iteration,
                message: "objective returned a non-finite value".to_string(),
            });
        }
        Ok(value)
    };

    let n = x0.len();
    let nf = T::from_usize(n).unwrap();

    // Adaptive parameters (Gao & Han 2012)
    // For small n, clamp rho >= 0.5 and sigma >= 0.5 to avoid degenerate behaviour.
    let (alpha, gamma, rho, sigma) = if config.adaptive {
        (
            T::one(),
            T::one() + T::from_f64(2.0).unwrap() / nf,
            T::max(
                T::from_f64(0.5).unwrap(),
                T::from_f64(0.75).unwrap() - T::one() / (T::from_f64(2.0).unwrap() * nf),
            ),
            T::max(T::from_f64(0.5).unwrap(), T::one() - T::one() / nf),
        )
    } else {
        (
            T::one(),
            T::from_f64(2.0).unwrap(),
            T::from_f64(0.5).unwrap(),
            T::from_f64(0.5).unwrap(),
        )
    };

    // Build initial simplex: x0 plus n axis-aligned perturbations.
    // Use a per-component edge length: if x0[i] is nonzero, use 5% of |x0[i]|,
    // otherwise use an absolute step of 0.00025 (like SciPy).
    let global_edge = config.initial_simplex_size;

    let mut simplex: Vec<Vec<T>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut xi = x0.to_vec();
        let edge_i = match global_edge {
            Some(e) => e,
            None => {
                // Nonzero components: use a fraction of the component magnitude.
                // Zero components: use a small absolute step.
                if x0[i].abs() > T::machine_epsilon() {
                    T::from_f64(0.05).unwrap() * x0[i].abs()
                } else {
                    T::from_f64(0.00025).unwrap()
                }
            }
        };
        // Ensure minimum edge length to avoid degenerate simplices
        let edge_i = T::max(edge_i, T::from_f64(1e-4).unwrap());
        xi[i] += edge_i;
        simplex.push(xi);
    }

    let mut f_values: Vec<T> = simplex
        .iter()
        .map(|x| evaluate(x, 0))
        .collect::<Result<_, _>>()?;
    let mut fun_evals = n + 1;

    // Sort indices by function value
    let mut order: Vec<usize> = (0..=n).collect();

    for iter in 0..config.max_iterations {
        if fun_evals >= config.max_fun_evals {
            let best = order[0];
            return Ok(OptimResult {
                x: simplex[best].clone(),
                fun: f_values[best],
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

        // Sort
        order.sort_by(|&a, &b| f_values[a].partial_cmp(&f_values[b]).unwrap());

        let best = order[0];
        let worst = order[n];
        // For n=1 the simplex has only 2 vertices, so second_worst == best.
        // Use worst as the fallback to keep the reflection acceptance meaningful.
        let second_worst = if n >= 2 { order[n - 1] } else { order[n] };

        // Convergence check: standard deviation of simplex values
        let f_mean = order
            .iter()
            .map(|&i| f_values[i])
            .fold(T::zero(), |a, b| a + b)
            / T::from_usize(n + 1).unwrap();
        let f_std = (order
            .iter()
            .map(|&i| {
                let d = f_values[i] - f_mean;
                d * d
            })
            .fold(T::zero(), |a, b| a + b)
            / T::from_usize(n + 1).unwrap())
        .sqrt();

        if f_std < config.fun_tolerance {
            return Ok(OptimResult {
                x: simplex[best].clone(),
                fun: f_values[best],
                grad: None,
                iterations: iter,
                fun_evals,
                grad_evals: 0,
                hess_evals: 0,
                grad_norm: T::infinity(),
                converged: true,
                status: OptimStatus::FunctionTolerance,
                kkt: None,
            });
        }

        // Check simplex diameter
        let mut max_dist = T::zero();
        for i in 1..=n {
            let dist = T::vec_norm(&sonium_solvers_core::vec_ops::vec_sub(
                &simplex[order[i]],
                &simplex[best],
            ));
            if dist > max_dist {
                max_dist = dist;
            }
        }
        if max_dist < config.x_tolerance {
            return Ok(OptimResult {
                x: simplex[best].clone(),
                fun: f_values[best],
                grad: None,
                iterations: iter,
                fun_evals,
                grad_evals: 0,
                hess_evals: 0,
                grad_norm: T::infinity(),
                converged: true,
                status: OptimStatus::StepTolerance,
                kkt: None,
            });
        }

        // Centroid of all points except worst
        let mut centroid = vec![T::zero(); n];
        for &i in &order[..n] {
            sonium_solvers_core::vec_ops::vec_add_assign(&mut centroid, &simplex[i]);
        }
        sonium_solvers_core::vec_ops::vec_mapv_inplace(&mut centroid, |c| {
            c / T::from_usize(n).unwrap()
        });

        // Reflection: x_r = centroid + alpha * (centroid - x_worst)
        let x_r = sonium_solvers_core::vec_ops::vec_add_scaled(
            &centroid,
            &sonium_solvers_core::vec_ops::vec_sub(&centroid, &simplex[worst]),
            alpha,
        );
        let f_r = evaluate(&x_r, iter)?;
        fun_evals += 1;

        if f_r < f_values[second_worst] && f_r >= f_values[best] {
            // Accept reflection
            simplex[worst] = x_r;
            f_values[worst] = f_r;
            continue;
        }

        if f_r < f_values[best] {
            // Expansion: x_e = centroid + gamma * (x_r - centroid)
            let x_e = sonium_solvers_core::vec_ops::vec_add_scaled(
                &centroid,
                &sonium_solvers_core::vec_ops::vec_sub(&x_r, &centroid),
                gamma,
            );
            let f_e = evaluate(&x_e, iter)?;
            fun_evals += 1;

            if f_e < f_r {
                simplex[worst] = x_e;
                f_values[worst] = f_e;
            } else {
                simplex[worst] = x_r;
                f_values[worst] = f_r;
            }
            continue;
        }

        // Contraction
        if f_r < f_values[worst] {
            // Outside contraction
            let x_c = sonium_solvers_core::vec_ops::vec_add_scaled(
                &centroid,
                &sonium_solvers_core::vec_ops::vec_sub(&x_r, &centroid),
                rho,
            );
            let f_c = evaluate(&x_c, iter)?;
            fun_evals += 1;
            if f_c <= f_r {
                simplex[worst] = x_c;
                f_values[worst] = f_c;
                continue;
            }
        } else {
            // Inside contraction
            let x_c = sonium_solvers_core::vec_ops::vec_add_scaled(
                &centroid,
                &sonium_solvers_core::vec_ops::vec_sub(&simplex[worst], &centroid),
                rho,
            );
            let f_c = evaluate(&x_c, iter)?;
            fun_evals += 1;
            if f_c < f_values[worst] {
                simplex[worst] = x_c;
                f_values[worst] = f_c;
                continue;
            }
        }

        // Shrink: move all points toward the best
        let x_best = simplex[best].clone();
        for &i in &order[1..] {
            simplex[i] = sonium_solvers_core::vec_ops::vec_add(
                &x_best,
                &sonium_solvers_core::vec_ops::vec_sub(&simplex[i], &x_best)
                    .iter()
                    .map(|&d| d * sigma)
                    .collect::<Vec<_>>(),
            );
            f_values[i] = evaluate(&simplex[i], iter)?;
            fun_evals += 1;
        }
    }

    order.sort_by(|&a, &b| f_values[a].partial_cmp(&f_values[b]).unwrap());
    let best = order[0];
    Ok(OptimResult {
        x: simplex[best].clone(),
        fun: f_values[best],
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nelder_mead_rejects_invalid_input_and_objective() {
        let config = NelderMeadConfig::<f64>::default();
        let objective = |x: &[f64]| x.iter().map(|value| value * value).sum();

        assert!(matches!(
            nelder_mead(&objective, &[], &config),
            Err(OptimError::DimensionMismatch {
                expected: 1,
                got: 0
            })
        ));
        assert!(matches!(
            nelder_mead(&objective, &[f64::NAN], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        let non_finite_objective = |_x: &[f64]| f64::INFINITY;
        assert!(matches!(
            nelder_mead(&non_finite_objective, &[0.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        let invalid_config = NelderMeadConfig {
            fun_tolerance: 0.0,
            ..config
        };
        assert!(matches!(
            nelder_mead(&objective, &[0.0], &invalid_config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));
    }

    #[test]
    fn test_nelder_mead_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = NelderMeadConfig::default();
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}, x: {:?}, iters: {}",
            result.status, result.fun, result.x, result.iterations
        );
        assert!(result.fun < 1e-6, "fun={}, x={:?}", result.fun, result.x);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let x0 = sonium_solvers_core::array![-1.2, 1.0];
        let config = NelderMeadConfig {
            max_iterations: 50_000,
            max_fun_evals: 200_000,
            ..Default::default()
        };
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert!(result.fun < 1e-8);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_nelder_mead_booth() {
        let f = |x: &[f64]| {
            let a = x[0] + 2.0 * x[1] - 7.0;
            let b = 2.0 * x[0] + x[1] - 5.0;
            a * a + b * b
        };
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = NelderMeadConfig::default();
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nelder_mead_already_at_min() {
        let f = |x: &[f64]| x.dot(x);
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = NelderMeadConfig::default();
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(result.converged);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_nelder_mead_1d() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2);
        let x0 = sonium_solvers_core::array![0.0];
        let config = NelderMeadConfig::default();
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nelder_mead_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: NelderMeadConfig<f32> = NelderMeadConfig::default();
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-2, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-2, "x[1]={}", result.x[1]);
    }

    #[test]
    fn test_nelder_mead_max_fun_evals() {
        // 10D sphere with very limited function evaluations -- should stop early.
        let f = |x: &[f64]| x.dot(x);
        let x0 = vec![1.0; 10];
        let config = NelderMeadConfig {
            max_fun_evals: 50,
            max_iterations: 100_000,
            ..Default::default()
        };
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert_eq!(
            result.status,
            OptimStatus::MaxFunctionEvals,
            "Expected MaxFunctionEvals, got {:?}",
            result.status
        );
        assert!(!result.converged);
    }

    #[test]
    fn test_nelder_mead_custom_simplex_size() {
        let f = |x: &[f64]| x.dot(x);
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = NelderMeadConfig {
            initial_simplex_size: Some(0.5),
            ..Default::default()
        };
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(
            result.converged,
            "Expected convergence with custom simplex size, status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_nelder_mead_non_adaptive() {
        // With adaptive=false, use standard (non-adaptive) parameters and verify
        // convergence on the Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
        // Minimum at (1, 3) with f = 0
        let f = |x: &[f64]| {
            let a = x[0] + 2.0 * x[1] - 7.0;
            let b = 2.0 * x[0] + x[1] - 5.0;
            a * a + b * b
        };
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = NelderMeadConfig {
            adaptive: false,
            ..Default::default()
        };
        let result = nelder_mead(&f, &x0, &config).unwrap();
        assert!(
            result.converged,
            "Non-adaptive NM failed: {:?}",
            result.status
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-4);
    }
}
