//! Shared checked-result contract for the iterative solver family.

use num_traits::Float;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverError, SolverResultContract, SolverStatus,
};

use super::{
    BiCgstabSolution, BlockFgmresSolution, BlockGmresSolution, CaCgSolution, CgSolution,
    CgsSolution, FgmresSolution, GmresSolution, IdrsSolution, MinresSolution, TfqmrSolution,
};

fn validate_scalar_result<T: ComplexField>(
    solution: &[T],
    status: SolverStatus,
    converged: bool,
    accuracy: AccuracyStatus,
    residual: T::Real,
    estimated_residual: T::Real,
    backward_error: T::Real,
) -> Result<(), SolverError> {
    // Numerical-error results intentionally use infinity as a sentinel for
    // unavailable diagnostics. They remain valid, typed failure records, but
    // must not also claim convergence or attained accuracy.
    if status == SolverStatus::NumericalError {
        if converged || accuracy == AccuracyStatus::Attained {
            return Err(SolverError::OperationFailed {
                operation: "iterative solver result contract",
                reason: "numerical-error result cannot report convergence or attained accuracy"
                    .into(),
            });
        }
        return Ok(());
    }

    if solution.iter().any(|value| !value.norm().is_finite())
        || !residual.is_finite()
        || !estimated_residual.is_finite()
        || !backward_error.is_finite()
    {
        return Err(SolverError::OperationFailed {
            operation: "iterative solver result contract",
            reason: "non-finite solution or diagnostic in non-numerical-error result".into(),
        });
    }
    if residual < residual.abs()
        || estimated_residual < estimated_residual.abs()
        || backward_error < backward_error.abs()
    {
        return Err(SolverError::OperationFailed {
            operation: "iterative solver result contract",
            reason: "residual and backward-error diagnostics must be non-negative".into(),
        });
    }

    if status == SolverStatus::Converged && (!converged || accuracy != AccuracyStatus::Attained) {
        return Err(SolverError::OperationFailed {
            operation: "iterative solver result contract",
            reason: "converged result must report attained accuracy".into(),
        });
    }

    if status != SolverStatus::Converged && converged {
        return Err(SolverError::OperationFailed {
            operation: "iterative solver result contract",
            reason: "non-converged status cannot set converged=true".into(),
        });
    }

    if status != SolverStatus::Converged && accuracy == AccuracyStatus::Attained {
        return Err(SolverError::OperationFailed {
            operation: "iterative solver result contract",
            reason: "non-converged result cannot report attained accuracy".into(),
        });
    }
    Ok(())
}

macro_rules! impl_scalar_contract {
    ($solution:ty) => {
        impl<T: ComplexField> SolverResultContract for $solution {
            fn solver_status(&self) -> SolverStatus {
                self.status
            }

            fn accuracy_status(&self) -> AccuracyStatus {
                self.attainable_accuracy
            }

            fn validate_contract(&self) -> Result<(), SolverError> {
                validate_scalar_result(
                    &self.x,
                    self.status,
                    self.converged,
                    self.attainable_accuracy,
                    self.residual,
                    self.estimated_residual,
                    self.backward_error,
                )
            }
        }
    };
}

impl_scalar_contract!(BiCgstabSolution<T>);
impl_scalar_contract!(CaCgSolution<T>);
impl_scalar_contract!(CgSolution<T>);
impl_scalar_contract!(CgsSolution<T>);
impl_scalar_contract!(FgmresSolution<T>);
impl_scalar_contract!(GmresSolution<T>);
impl_scalar_contract!(IdrsSolution<T>);
impl_scalar_contract!(MinresSolution<T>);
impl_scalar_contract!(TfqmrSolution<T>);

fn block_status<T: ComplexField>(
    x: &sonium_solvers_core::dense::DenseMatrix<T>,
    true_residuals: &[T::Real],
    backward_errors: &[T::Real],
    converged: bool,
) -> SolverStatus {
    if x.as_slice().iter().any(|value| !value.norm().is_finite())
        || true_residuals.iter().any(|value| !value.is_finite())
        || backward_errors.iter().any(|value| !value.is_finite())
    {
        SolverStatus::NumericalError
    } else if converged {
        SolverStatus::Converged
    } else {
        SolverStatus::MaxIterationsReached
    }
}

fn block_accuracy(statuses: &[AccuracyStatus]) -> AccuracyStatus {
    if statuses.is_empty() {
        AccuracyStatus::NotChecked
    } else if statuses
        .iter()
        .all(|status| *status == AccuracyStatus::Attained)
    {
        AccuracyStatus::Attained
    } else if statuses.contains(&AccuracyStatus::NotAttained) {
        AccuracyStatus::NotAttained
    } else {
        AccuracyStatus::NotChecked
    }
}

fn validate_block_result<T: ComplexField>(
    x: &sonium_solvers_core::dense::DenseMatrix<T>,
    residuals: &[T::Real],
    true_residuals: &[T::Real],
    backward_errors: &[T::Real],
    accuracies: &[AccuracyStatus],
    converged: bool,
) -> Result<(), SolverError> {
    let rhs_count = x.ncols();
    if residuals.len() != rhs_count
        || true_residuals.len() != rhs_count
        || backward_errors.len() != rhs_count
        || accuracies.len() != rhs_count
    {
        return Err(SolverError::DimensionMismatch {
            expected: rhs_count,
            got: residuals
                .len()
                .max(true_residuals.len())
                .max(backward_errors.len())
                .max(accuracies.len()),
        });
    }

    if block_status(x, true_residuals, backward_errors, converged) == SolverStatus::NumericalError {
        // Match scalar iterative results: non-finite diagnostics are a valid
        // typed NumericalError sentinel when the result clearly does not
        // claim convergence. Preserve the fail-closed metadata contract.
        if converged || accuracies.contains(&AccuracyStatus::Attained) {
            return Err(SolverError::OperationFailed {
                operation: "block iterative solver result contract",
                reason: "numerical-error block cannot report convergence or attained accuracy"
                    .into(),
            });
        }
        return Ok(());
    }

    if residuals.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "block iterative solver result contract",
            reason: "non-finite projected residual".into(),
        });
    }
    if residuals
        .iter()
        .chain(true_residuals)
        .chain(backward_errors)
        .any(|value| *value < value.abs())
    {
        return Err(SolverError::OperationFailed {
            operation: "block iterative solver result contract",
            reason: "residual and backward-error diagnostics must be non-negative".into(),
        });
    }

    if converged && block_accuracy(accuracies) != AccuracyStatus::Attained {
        return Err(SolverError::OperationFailed {
            operation: "block iterative solver result contract",
            reason: "converged block must report attained accuracy for every RHS".into(),
        });
    }

    if !converged && block_accuracy(accuracies) == AccuracyStatus::Attained {
        return Err(SolverError::OperationFailed {
            operation: "block iterative solver result contract",
            reason: "non-converged block cannot report attained accuracy for every RHS".into(),
        });
    }

    Ok(())
}

macro_rules! impl_block_contract {
    ($solution:ty) => {
        impl<T: ComplexField> SolverResultContract for $solution {
            fn solver_status(&self) -> SolverStatus {
                block_status(
                    &self.x,
                    &self.true_residuals,
                    &self.backward_errors,
                    self.converged,
                )
            }

            fn accuracy_status(&self) -> AccuracyStatus {
                block_accuracy(&self.attainable_accuracy)
            }

            fn validate_contract(&self) -> Result<(), SolverError> {
                validate_block_result(
                    &self.x,
                    &self.residuals,
                    &self.true_residuals,
                    &self.backward_errors,
                    &self.attainable_accuracy,
                    self.converged,
                )
            }
        }
    };
}

impl_block_contract!(BlockGmresSolution<T>);
impl_block_contract!(BlockFgmresSolution<T>);

/// A common bound for all public iterative result types.
pub trait IterativeResultContract: SolverResultContract {}

impl<T> IterativeResultContract for T where T: SolverResultContract {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GmresConfig;
    use sonium_solvers_core::dense::DenseMatrix;
    use sonium_solvers_core::traits::LinearOperator;

    struct Identity;

    impl LinearOperator<f64> for Identity {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            y.copy_from_slice(x);
        }

        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            y.copy_from_slice(x);
        }
    }

    #[test]
    fn checked_gmres_result_satisfies_the_public_contract() {
        let result = crate::try_gmres(
            &Identity,
            &[1.0, -2.0],
            &GmresConfig {
                max_iterations: 4,
                restart: 2,
                tolerance: 1.0e-12,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 1,
                ..Default::default()
            },
        )
        .expect("identity GMRES should converge");
        assert_eq!(result.solver_status(), SolverStatus::Converged);
        assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
        result.validate_contract().unwrap();
    }

    #[test]
    fn scalar_contract_rejects_converged_without_attained_accuracy() {
        let result = GmresSolution::<f64> {
            x: vec![1.0],
            iterations: 1,
            restarts: 0,
            residual: 0.0,
            estimated_residual: 0.0,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts: Default::default(),
        };
        assert!(result.validate_contract().is_err());
    }

    #[test]
    fn scalar_contract_rejects_nonconverged_with_attained_accuracy() {
        let result = GmresSolution::<f64> {
            x: vec![1.0],
            iterations: 1,
            restarts: 0,
            residual: 1.0,
            estimated_residual: 1.0,
            backward_error: 1.0,
            attainable_accuracy: AccuracyStatus::Attained,
            converged: false,
            status: SolverStatus::MaxIterationsReached,
            op_counts: Default::default(),
        };
        assert!(result.validate_contract().is_err());
    }

    #[test]
    fn block_contract_requires_one_metadata_entry_per_rhs() {
        let result = BlockGmresSolution::<f64> {
            x: DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 2.0]),
            iterations: 1,
            residuals: vec![0.0],
            true_residuals: vec![0.0],
            backward_errors: vec![0.0],
            attainable_accuracy: vec![AccuracyStatus::Attained],
            converged: true,
            op_counts: Default::default(),
        };
        assert!(matches!(
            result.validate_contract(),
            Err(SolverError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn block_contract_rejects_nonconverged_with_attained_accuracy() {
        let result = BlockGmresSolution::<f64> {
            x: DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 2.0]),
            iterations: 1,
            residuals: vec![1.0, 1.0],
            true_residuals: vec![1.0, 1.0],
            backward_errors: vec![1.0, 1.0],
            attainable_accuracy: vec![AccuracyStatus::Attained, AccuracyStatus::Attained],
            converged: false,
            op_counts: Default::default(),
        };
        assert!(result.validate_contract().is_err());
    }
}
