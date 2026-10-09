//! Unconstrained optimisation algorithms.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::line_search::LineSearchResult;
use crate::types::{OptimError, OptimResult, RealScalar};

pub(crate) fn validate_objective_start<T, F>(f: &F, x0: &[T]) -> Result<(), OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T + ?Sized,
{
    if let Some(index) = x0.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("initial point contains a non-finite value at index {index}"),
        });
    }
    let value =
        catch_unwind(AssertUnwindSafe(|| f(x0))).map_err(|_| OptimError::CallbackPanicked {
            iteration: 0,
            stage: "objective preflight".into(),
        })?;
    if !value.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "objective callback returned a non-finite value".into(),
        });
    }
    Ok(())
}

pub(crate) fn validate_gradient_start<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
) -> Result<(), OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T + ?Sized,
    G: Fn(&[T]) -> Vec<T> + ?Sized,
{
    validate_objective_start(f, x0)?;
    let gradient = catch_unwind(AssertUnwindSafe(|| grad_fn(x0))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration: 0,
            stage: "gradient preflight".into(),
        }
    })?;
    if gradient.len() != x0.len() {
        return Err(OptimError::DimensionMismatch {
            expected: x0.len(),
            got: gradient.len(),
        });
    }
    if let Some(index) = gradient.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("gradient callback returned a non-finite value at index {index}"),
        });
    }
    Ok(())
}

pub(crate) fn validate_gradient_output<T: RealScalar>(
    gradient: &[T],
    expected_len: usize,
    iteration: usize,
    stage: &'static str,
) -> Result<(), OptimError> {
    if gradient.len() != expected_len {
        return Err(OptimError::DimensionMismatch {
            expected: expected_len,
            got: gradient.len(),
        });
    }
    if let Some(index) = gradient.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: format!("{stage} returned a non-finite gradient at index {index}"),
        });
    }
    Ok(())
}

pub(crate) fn validate_finite_vector<T: RealScalar>(
    values: &[T],
    iteration: usize,
    stage: &'static str,
) -> Result<(), OptimError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: format!("{stage} contains a non-finite value at index {index}"),
        });
    }
    Ok(())
}

pub(crate) fn validate_line_search_result<T: RealScalar>(
    result: &LineSearchResult<T>,
    expected_gradient_len: usize,
    iteration: usize,
    stage: &'static str,
) -> Result<(), OptimError> {
    if !result.alpha.is_finite() || !result.fun.is_finite() {
        return Err(OptimError::NumericalError {
            iteration,
            message: format!("{stage} returned a non-finite step or objective"),
        });
    }
    if let Some(gradient) = result.grad.as_deref() {
        validate_gradient_output(gradient, expected_gradient_len, iteration, stage)?;
    }
    Ok(())
}

pub(crate) fn validate_hessian_vector_start<T, Hv>(
    hess_vec_fn: &Hv,
    x0: &[T],
    stage: &'static str,
) -> Result<(), OptimError>
where
    T: RealScalar,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    let zero = vec![T::zero(); x0.len()];
    let value = catch_unwind(AssertUnwindSafe(|| hess_vec_fn(x0, &zero))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration: 0,
            stage: stage.into(),
        }
    })?;
    if value.len() != x0.len() {
        return Err(OptimError::DimensionMismatch {
            expected: x0.len(),
            got: value.len(),
        });
    }
    if value.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("{stage} returned a non-finite value"),
        });
    }
    Ok(())
}

pub(crate) fn validate_result<T: RealScalar>(result: &OptimResult<T>) -> Result<(), OptimError> {
    crate::types::validate_result_finite(result)
}

pub(crate) mod bfgs_impl;
pub(crate) mod bobyqa_impl;
pub(crate) mod cmaes_impl;
pub(crate) mod conjugate_gradient_impl;
/// Strategy-rich Differential Evolution (mutation strategies, L-SHADE,
/// archive, mixed-integer and penalty-constraint support).
pub mod de;
pub(crate) mod global_impl;
pub(crate) mod lbfgs_impl;
/// Bounded Levenberg-Marquardt nonlinear least-squares solver.
pub mod levenberg_marquardt;
pub(crate) mod nelder_mead_impl;
pub(crate) mod newton_cg_impl;
pub(crate) mod powell_impl;
pub(crate) mod quasi_newton_impl;

pub use bfgs_impl::{BfgsConfig, bfgs};
pub use bobyqa_impl::{BobyqaConfig, bobyqa};
pub use cmaes_impl::{CmaEsConfig, cma_es};
pub use conjugate_gradient_impl::{CgVariant, NonlinearCgConfig, nonlinear_cg};
pub use global_impl::{
    DifferentialEvolutionConfig, RandomSearchConfig, SimulatedAnnealingConfig,
    differential_evolution, random_search, simulated_annealing,
};
pub use lbfgs_impl::{LbfgsConfig, lbfgs};
pub use nelder_mead_impl::{NelderMeadConfig, nelder_mead};
pub use newton_cg_impl::{NewtonCgConfig, newton_cg};
#[allow(deprecated)]
pub use powell_impl::{PowellConfig, powell};
pub use quasi_newton_impl::{QuasiNewtonConfig, QuasiNewtonUpdate, quasi_newton};

#[cfg(test)]
mod validation_tests {
    use super::{LbfgsConfig, NelderMeadConfig, NewtonCgConfig, lbfgs, nelder_mead, newton_cg};
    use crate::types::OptimError;

    #[test]
    fn gradient_optimizer_preflight_rejects_non_finite_callback_data() {
        let error = lbfgs(
            &|_| 0.0,
            &|_| vec![f64::NAN],
            &[0.0],
            &LbfgsConfig::default(),
        )
        .expect_err("non-finite gradient must fail before iteration");
        assert!(matches!(
            error,
            OptimError::NumericalError { iteration: 0, .. }
        ));
    }

    #[test]
    fn derivative_free_optimizer_contains_callback_panics() {
        let error = nelder_mead(
            &|_| panic!("objective callback failed"),
            &[0.0],
            &NelderMeadConfig::default(),
        )
        .expect_err("objective panic must become a typed error");
        assert!(matches!(
            error,
            OptimError::CallbackPanicked { iteration: 0, .. }
        ));
    }

    #[test]
    fn hessian_vector_preflight_rejects_wrong_shape() {
        let error = newton_cg(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            &|_: &[f64], _: &[f64]| vec![],
            &[1.0],
            &NewtonCgConfig::default(),
        )
        .expect_err("malformed Hessian-vector output must fail before iteration");
        assert!(matches!(error, OptimError::DimensionMismatch { .. }));
    }
}
