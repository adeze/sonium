//! Brent's method for 1D minimisation on a bracket.
//!
//! Combines golden section search with parabolic interpolation.
//! Used by Powell's direction-set method.

use crate::types::RealScalar;
use sonium_solvers_core::traits::{
    AccuracyStatus, SolverError, SolverResultContract, SolverStatus,
};

/// Configuration for Brent's 1D minimisation.
#[derive(Debug, Clone)]
pub struct BrentConfig<T: RealScalar> {
    pub tol: T,
    pub max_iterations: usize,
}

impl Default for BrentConfig<f64> {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iterations: 500,
        }
    }
}

impl Default for BrentConfig<f32> {
    fn default() -> Self {
        Self {
            tol: 1e-4,
            max_iterations: 500,
        }
    }
}

/// Result of Brent 1D minimisation.
#[derive(Debug, Clone)]
pub struct BrentResult<T: RealScalar> {
    pub x_min: T,
    pub f_min: T,
    pub iterations: usize,
    pub fun_evals: usize,
    pub status: BrentStatus,
}

/// Termination status for Brent's one-dimensional minimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrentStatus {
    /// The bracket met the configured stopping criterion.
    Converged,
    /// The iteration budget ended before the bracket met the criterion.
    MaxIterations,
    /// The objective or an intermediate result was non-finite.
    NumericalError,
}

impl<T: RealScalar> SolverResultContract for BrentResult<T> {
    fn solver_status(&self) -> SolverStatus {
        match self.status {
            BrentStatus::Converged => SolverStatus::Converged,
            BrentStatus::MaxIterations => SolverStatus::MaxIterationsReached,
            BrentStatus::NumericalError => SolverStatus::NumericalError,
        }
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        match self.status {
            BrentStatus::Converged => AccuracyStatus::Attained,
            BrentStatus::MaxIterations => AccuracyStatus::NotAttained,
            BrentStatus::NumericalError => AccuracyStatus::NotChecked,
        }
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if !self.x_min.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "brent result contract",
                reason: "minimizer is non-finite".into(),
            });
        }
        if !self.f_min.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "brent result contract",
                reason: "objective value is non-finite".into(),
            });
        }
        if self.fun_evals == 0 {
            return Err(SolverError::OperationFailed {
                operation: "brent result contract",
                reason: "function evaluation count is zero".into(),
            });
        }
        if self.status == BrentStatus::Converged && self.iterations == 0 {
            return Err(SolverError::OperationFailed {
                operation: "brent result contract",
                reason: "converged result has zero iterations".into(),
            });
        }
        Ok(())
    }
}

/// Minimise a univariate function on the bracket `[a, b]` using Brent's method.
///
/// Returns the minimiser `x_min`, its function value, and iteration count.
pub fn brent_minimize<T, F>(f: &F, mut a: T, mut b: T, config: &BrentConfig<T>) -> BrentResult<T>
where
    T: RealScalar,
    F: Fn(T) -> T,
{
    let golden = T::from_f64(0.3819660112501051).unwrap(); // (3 - sqrt(5)) / 2
    let two = T::from_f64(2.0).unwrap();
    let half = T::from_f64(0.5).unwrap();

    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut fun_evals = 1_usize;

    let mut d = T::zero();
    let mut e = T::zero(); // distance moved on the step before last
    let mut iterations = 0;

    if !fx.is_finite() || !config.tol.is_finite() || config.tol <= T::zero() {
        return BrentResult {
            x_min: x,
            f_min: fx,
            iterations,
            fun_evals,
            status: BrentStatus::NumericalError,
        };
    }

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let midpoint = half * (a + b);
        let tol1 = config.tol * x.abs() + T::machine_epsilon();
        let tol2 = two * tol1;

        // Convergence test
        if (x - midpoint).abs() <= tol2 - half * (b - a) {
            return BrentResult {
                x_min: x,
                f_min: fx,
                iterations,
                fun_evals,
                status: BrentStatus::Converged,
            };
        }

        // Try parabolic interpolation
        let mut use_golden = true;
        if e.abs() > tol1 {
            // Fit parabola through x, w, v
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q = two * (q - r);
            let (p, q) = if q > T::zero() { (-p, q) } else { (p, -q) };

            // Accept parabolic step only if it falls within the bracket
            // and is a sufficient improvement
            if p.abs() < (half * q * e).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                // Don't evaluate too close to endpoints
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if x < midpoint { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            e = if x < midpoint { b - x } else { a - x };
            d = golden * e;
        } else {
            e = d;
        }

        // Evaluate at the new point
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > T::zero() {
            x + tol1
        } else {
            x - tol1
        };

        let fu = f(u);
        fun_evals += 1;

        if !fu.is_finite() {
            return BrentResult {
                x_min: x,
                f_min: fx,
                iterations,
                fun_evals,
                status: BrentStatus::NumericalError,
            };
        }

        // Update bracket and best points
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    BrentResult {
        x_min: x,
        f_min: fx,
        iterations,
        fun_evals,
        status: BrentStatus::MaxIterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_brent_quadratic() {
        // f(x) = (x - 3)^2, min at x = 3
        let f = |x: f64| (x - 3.0) * (x - 3.0);
        let config = BrentConfig::default();
        let result = brent_minimize(&f, 0.0, 10.0, &config);
        assert_relative_eq!(result.x_min, 3.0, epsilon = 1e-7);
        assert_relative_eq!(result.f_min, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_brent_cubic() {
        // f(x) = x^4 - 2x^2, min at x = 1 or x = -1
        let f = |x: f64| x.powi(4) - 2.0 * x * x;
        let config = BrentConfig::default();
        let result = brent_minimize(&f, 0.0, 3.0, &config);
        assert_relative_eq!(result.x_min, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.f_min, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_brent_sin() {
        // f(x) = sin(x) on [2, 5], min near x = 3*pi/2 ≈ 4.712
        let f = |x: f64| x.sin();
        let config = BrentConfig::default();
        let result = brent_minimize(&f, 2.0, 5.0, &config);
        assert_relative_eq!(
            result.x_min,
            3.0 * std::f64::consts::FRAC_PI_2,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_brent_f32() {
        let f = |x: f32| (x - 2.0) * (x - 2.0);
        let config = BrentConfig::<f32>::default();
        let result = brent_minimize(&f, -5.0_f32, 5.0_f32, &config);
        assert!((result.x_min - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_brent_flat_function() {
        // f(x) = constant. The convergence test (bracket width) should trigger
        // quickly since the function provides no directional preference.
        let f = |_x: f64| 42.0;
        let config = BrentConfig::default();
        let result = brent_minimize(&f, -10.0, 10.0, &config);
        // Should converge quickly; the exact x_min is unimportant for a constant.
        assert_relative_eq!(result.f_min, 42.0, epsilon = 1e-12);
        // The algorithm should not need many iterations on a flat function.
        assert!(
            result.iterations <= 100,
            "Expected quick convergence on flat function, got {} iterations",
            result.iterations
        );
    }

    #[test]
    fn test_brent_max_iterations() {
        // Set a very low max_iterations and verify the algorithm respects it.
        let f = |x: f64| (x - 3.0).powi(2);
        let config = BrentConfig {
            max_iterations: 3,
            ..Default::default()
        };
        let result = brent_minimize(&f, 0.0, 10.0, &config);
        // iterations should be exactly max_iterations (did not converge early
        // with only 3 iterations on a wide bracket).
        assert!(
            result.iterations == 3,
            "Expected exactly 3 iterations, got {}",
            result.iterations
        );
    }
}
