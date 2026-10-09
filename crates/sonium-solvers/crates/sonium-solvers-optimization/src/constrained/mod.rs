//! Constrained optimisation algorithms.

use std::panic::{AssertUnwindSafe, catch_unwind};

pub(crate) mod augmented_lagrangian_impl;
/// Constrained Optimization BY Response surfaces (surrogate-based).
pub mod cobra;
/// Derivative-free COBYLA local optimizer with inequality constraints.
pub mod cobyla;
/// Hand-translated Powell COBYLA core (pure Rust, no external dependency).
pub mod cobyla_native;
pub(crate) mod epsilon_constraint_impl;
pub(crate) mod ipm_impl;
/// Improved Stochastic Ranking Evolution Strategy (constrained global).
pub mod isres;
pub(crate) mod lbfgsb_impl;
pub(crate) mod mma_impl;
pub(crate) mod penalty_impl;
pub mod qp_active_set;
pub(crate) mod slsqp_impl;
pub(crate) mod sqp_advanced_impl;
pub(crate) mod sqp_impl;
pub(crate) mod trust_region_impl;

/// Validate the first callback evaluation at an optimizer boundary. Several
/// constrained methods delegate to infallible inner algorithms, so this
/// shared preflight prevents NaN/Inf or malformed callback data from entering
/// those loops as an apparently valid initial state.
pub(crate) fn validate_optimizer_start<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[crate::types::Constraint<T>],
) -> Result<(), crate::types::OptimError>
where
    T: crate::types::RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    if let Some(index) = x0.iter().position(|value| !value.is_finite()) {
        return Err(crate::types::OptimError::NumericalError {
            iteration: 0,
            message: format!("initial point contains a non-finite value at index {index}"),
        });
    }

    let objective = catch_unwind(AssertUnwindSafe(|| f(x0))).map_err(|_| {
        crate::types::OptimError::CallbackPanicked {
            iteration: 0,
            stage: "objective preflight".into(),
        }
    })?;
    if !objective.is_finite() {
        return Err(crate::types::OptimError::NumericalError {
            iteration: 0,
            message: "objective callback returned a non-finite value".into(),
        });
    }

    let gradient = catch_unwind(AssertUnwindSafe(|| grad_fn(x0))).map_err(|_| {
        crate::types::OptimError::CallbackPanicked {
            iteration: 0,
            stage: "gradient preflight".into(),
        }
    })?;
    if gradient.len() != x0.len() {
        return Err(crate::types::OptimError::DimensionMismatch {
            expected: x0.len(),
            got: gradient.len(),
        });
    }
    if let Some(index) = gradient.iter().position(|value| !value.is_finite()) {
        return Err(crate::types::OptimError::NumericalError {
            iteration: 0,
            message: format!("gradient callback returned a non-finite value at index {index}"),
        });
    }

    for (index, constraint) in constraints.iter().enumerate() {
        let values = catch_unwind(AssertUnwindSafe(|| (constraint.fun)(x0))).map_err(|_| {
            crate::types::OptimError::CallbackPanicked {
                iteration: 0,
                stage: format!("constraint {index} preflight"),
            }
        })?;
        if values.len() != constraint.dim {
            return Err(crate::types::OptimError::InvalidConstraintDimension {
                index,
                expected: constraint.dim,
                got: values.len(),
            });
        }
        if let Some(value_index) = values.iter().position(|value| !value.is_finite()) {
            return Err(crate::types::OptimError::NumericalError {
                iteration: 0,
                message: format!(
                    "constraint {index} returned a non-finite value at index {value_index}"
                ),
            });
        }
        if let Some(jacobian) = &constraint.jac {
            let jacobian = catch_unwind(AssertUnwindSafe(|| jacobian(x0))).map_err(|_| {
                crate::types::OptimError::CallbackPanicked {
                    iteration: 0,
                    stage: format!("constraint {index} Jacobian preflight"),
                }
            })?;
            if jacobian.nrows() != constraint.dim || jacobian.ncols() != x0.len() {
                return Err(crate::types::OptimError::InvalidConstraintJacobian {
                    index,
                    expected_rows: constraint.dim,
                    expected_cols: x0.len(),
                    got_rows: jacobian.nrows(),
                    got_cols: jacobian.ncols(),
                });
            }
            if (0..jacobian.nrows())
                .any(|row| (0..jacobian.ncols()).any(|column| !jacobian[[row, column]].is_finite()))
            {
                return Err(crate::types::OptimError::NumericalError {
                    iteration: 0,
                    message: format!("constraint {index} Jacobian returned a non-finite value"),
                });
            }
        }
    }
    Ok(())
}

/// Validate a user-supplied Hessian callback at the initial point.  Calling
/// it once here catches panics, malformed shapes, and non-finite generated
/// values before an inner optimiser can turn them into an opaque failure.
pub(crate) fn validate_hessian_start<T, H>(
    hessian_fn: &H,
    x0: &[T],
    lambda_len: usize,
    stage: &'static str,
) -> Result<(), crate::types::OptimError>
where
    T: crate::types::RealScalar,
    H: Fn(&[T], &[T]) -> sonium_solvers_core::dense::DenseMatrix<T>,
{
    let lambda = vec![T::zero(); lambda_len];
    let hessian = catch_unwind(AssertUnwindSafe(|| hessian_fn(x0, &lambda))).map_err(|_| {
        crate::types::OptimError::CallbackPanicked {
            iteration: 0,
            stage: stage.into(),
        }
    })?;
    if hessian.nrows() != x0.len() || hessian.ncols() != x0.len() {
        return Err(crate::types::OptimError::DimensionMismatch {
            expected: x0.len() * x0.len(),
            got: hessian.nrows() * hessian.ncols(),
        });
    }
    if (0..hessian.nrows())
        .any(|row| (0..hessian.ncols()).any(|column| !hessian[[row, column]].is_finite()))
    {
        return Err(crate::types::OptimError::NumericalError {
            iteration: 0,
            message: format!("{stage} returned a non-finite Hessian"),
        });
    }
    Ok(())
}

/// Attach a final KKT report using an identity-Hessian linearised model. This
/// is the common fallback for constrained methods that do not expose their
/// native multiplier state; SQP/IPM wrappers use the same contract directly
/// while they are migrated to native multipliers.
pub(crate) fn attach_identity_kkt<T: crate::types::RealScalar>(
    result: &mut crate::types::OptimResult<T>,
    constraints: &[crate::types::Constraint<T>],
    bounds: &[crate::types::Bound<T>],
) -> Result<(), crate::types::OptimError> {
    crate::types::validate_bounds(bounds, result.x.len())?;
    let Some(gradient) = result.grad.as_deref() else {
        return Ok(());
    };
    let hessian = sonium_solvers_core::dense::DenseMatrix::eye(result.x.len());
    let qp = qp_active_set::try_solve_linearized_qp(
        gradient,
        &hessian,
        &result.x,
        constraints,
        bounds,
        result.x.len(),
    )?;
    let multipliers = qp_active_set::kkt_multipliers_from_qp(&qp.multipliers, constraints, bounds)?;
    result.attach_kkt(constraints, bounds, multipliers)
}

/// Attach the common identity-Hessian KKT estimate and apply the constrained
/// success gate.  A result that did not expose a gradient can remain a
/// non-converged partial result, but it cannot remain successful because
/// [`crate::types::OptimResult::gate_kkt`] rejects the missing report.
pub(crate) fn finalize_identity_kkt<T: crate::types::RealScalar>(
    result: &mut crate::types::OptimResult<T>,
    constraints: &[crate::types::Constraint<T>],
    bounds: &[crate::types::Bound<T>],
    tolerance: T,
) -> Result<(), crate::types::OptimError> {
    attach_identity_kkt(result, constraints, bounds)?;
    result.gate_kkt(tolerance);
    Ok(())
}

pub use augmented_lagrangian_impl::{AugmentedLagrangianConfig, InnerSolver, augmented_lagrangian};
pub use epsilon_constraint_impl::{EpsilonConstraintConfig, ParetoResult, epsilon_constraint};
pub use ipm_impl::{IpmConfig, ipm};
pub use lbfgsb_impl::{LbfgsbConfig, lbfgsb};
pub use mma_impl::{MmaConfig, MmaVariant, mma};
pub use penalty_impl::{PenaltyConfig, PenaltyType, penalty_method};
pub use qp_active_set::{QpSense, QpSolution, QpStatus, solve_qp};
pub use slsqp_impl::{SlsqpConfig, slsqp};
pub use sqp_advanced_impl::{AdvancedSqpConfig, HessianRegularisation, sqp_advanced};
pub use sqp_impl::{SqpAcceptance, SqpConfig, sqp};
pub use trust_region_impl::{TrustRegionConstrainedConfig, trust_region_constrained};

#[cfg(test)]
mod tests {
    use super::{PenaltyConfig, PenaltyType, penalty_method};
    use crate::types::{Constraint, ConstraintKind, OptimError};

    #[test]
    fn constrained_optimizer_preflight_rejects_non_finite_objective() {
        let config = PenaltyConfig::<f64> {
            penalty_type: PenaltyType::Quadratic,
            ..PenaltyConfig::default()
        };
        let error = penalty_method(&|_| f64::NAN, &|_| vec![0.0], &[0.0], &[], &config)
            .expect_err("non-finite objective must fail before iteration");
        assert!(matches!(
            error,
            OptimError::NumericalError { iteration: 0, .. }
        ));
    }

    #[test]
    fn constrained_optimizer_preflight_rejects_non_finite_gradient() {
        let config = PenaltyConfig::<f64>::default();
        let error = penalty_method(&|_| 0.0, &|_| vec![f64::INFINITY], &[0.0], &[], &config)
            .expect_err("non-finite gradient must fail before iteration");
        assert!(matches!(
            error,
            OptimError::NumericalError { iteration: 0, .. }
        ));
    }

    #[test]
    fn constrained_optimizer_preflight_contains_constraint_panics() {
        let config = PenaltyConfig::<f64>::default();
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|_| panic!("constraint callback failed")),
            jac: None,
        }];
        let error = penalty_method(&|_| 0.0, &|_| vec![0.0], &[0.0], &constraints, &config)
            .expect_err("constraint panic must fail before iteration");
        assert!(matches!(
            error,
            OptimError::CallbackPanicked { iteration: 0, .. }
        ));
    }

    #[test]
    fn ipm_hessian_preflight_rejects_wrong_shape() {
        let config = super::IpmConfig::<f64>::default();
        let hessian = |_: &[f64], _: &[f64]| sonium_solvers_core::dense::DenseMatrix::zeros(1, 2);
        let error = super::ipm(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            Some(&hessian),
            &[1.0],
            &[],
            &config,
        )
        .expect_err("malformed Hessian must fail before iteration");
        assert!(matches!(error, OptimError::DimensionMismatch { .. }));
    }
}
