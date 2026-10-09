//! Reusable forward and adjoint linear solves.
//!
//! A gradient or sensitivity calculation usually solves the same linear
//! system twice: once with the physical right-hand side and once with an
//! adjoint right-hand side.  [`DifferentiableLinearSolve`] keeps the
//! operator, preconditioner, and Krylov configuration together so both
//! solves use the same validated state.  The adjoint solve is explicit about
//! whether it uses `Aᵀ` or `Aᴴ`; unsupported operations are rejected before a
//! Krylov iteration can start.

use crate::differentiable_contract::CancellationToken;
use crate::iterative::{
    BlockFgmresConfig, BlockFgmresSolution, GmresConfig, GmresSolution,
    gmres_preconditioned_with_guess, try_block_fgmres_preconditioned,
};
use crate::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use num_traits::{Float, ToPrimitive, Zero};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Forward and optional transpose/Hermitian block solve results.
#[derive(Debug)]
pub struct DifferentiableBlockSolveResult<T: ComplexField> {
    /// Forward block solution, one column per right-hand side.
    pub forward: BlockFgmresSolution<T>,
    /// Optional adjoint block solution.
    pub adjoint: Option<BlockFgmresSolution<T>>,
}

impl<T: ComplexField> SolverResultContract for DifferentiableBlockSolveResult<T> {
    fn solver_status(&self) -> SolverStatus {
        let forward_status = self.forward.solver_status();
        if forward_status != SolverStatus::Converged {
            return forward_status;
        }
        self.adjoint
            .as_ref()
            .map_or(forward_status, SolverResultContract::solver_status)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        let forward_accuracy = self.forward.accuracy_status();
        if forward_accuracy != AccuracyStatus::Attained {
            return forward_accuracy;
        }
        self.adjoint
            .as_ref()
            .map_or(forward_accuracy, SolverResultContract::accuracy_status)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        self.forward.validate_contract()?;
        if let Some(adjoint) = &self.adjoint {
            if adjoint.x.nrows() != self.forward.x.nrows() {
                return Err(SolverError::DimensionMismatch {
                    expected: self.forward.x.nrows(),
                    got: adjoint.x.nrows(),
                });
            }
            adjoint.validate_contract()?;
        }
        Ok(())
    }
}

/// Which adjoint linear system should be solved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjointSolve {
    /// Solve `Aᵀ λ = g`.
    Transpose,
    /// Solve `Aᴴ λ = g`.
    Hermitian,
}

/// Results from a forward solve and its optional adjoint solve.
#[derive(Debug)]
pub struct DifferentiableSolveResult<T: ComplexField> {
    /// Solution of `A x = b`.
    pub forward: GmresSolution<T>,
    /// Solution of the requested adjoint system, if one was requested.
    pub adjoint: Option<GmresSolution<T>>,
}

impl<T: ComplexField> SolverResultContract for DifferentiableSolveResult<T> {
    fn solver_status(&self) -> SolverStatus {
        let forward_status = self.forward.solver_status();
        if forward_status != SolverStatus::Converged {
            return forward_status;
        }
        self.adjoint
            .as_ref()
            .map_or(forward_status, SolverResultContract::solver_status)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        let forward_accuracy = self.forward.accuracy_status();
        if forward_accuracy != AccuracyStatus::Attained {
            return forward_accuracy;
        }
        self.adjoint
            .as_ref()
            .map_or(forward_accuracy, SolverResultContract::accuracy_status)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        self.forward.validate_contract()?;
        if let Some(adjoint) = &self.adjoint {
            if adjoint.x.len() != self.forward.x.len() {
                return Err(SolverError::DimensionMismatch {
                    expected: self.forward.x.len(),
                    got: adjoint.x.len(),
                });
            }
            adjoint.validate_contract()?;
        }
        Ok(())
    }
}

/// A reusable, checked forward/adjoint GMRES solve handle.
///
/// The references deliberately borrow the operator and preconditioner.  This
/// lets callers retain factorisations or device-resident preconditioners and
/// reuse them across a forward solve and any number of sensitivity solves.
pub struct DifferentiableLinearSolve<'a, T, A, P>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    operator: &'a A,
    preconditioner: &'a P,
    config: GmresConfig<T::Real>,
}

/// Operator adapter that polls cooperative cancellation at every Krylov
/// action. The native iterative core is infallible for performance, so a
/// cancellation request is raised as a panic here and converted to a typed
/// `SolverError` by the checked solve boundary.
struct CancellableLinearOperator<'a, T, A> {
    inner: &'a A,
    cancellation: &'a CancellationToken,
    marker: std::marker::PhantomData<fn() -> T>,
}

impl<'a, T, A> CancellableLinearOperator<'a, T, A> {
    fn new(inner: &'a A, cancellation: &'a CancellationToken) -> Self {
        Self {
            inner,
            cancellation,
            marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn poll(&self) {
        if self.cancellation.is_cancelled() {
            panic!("differentiable operation cancelled");
        }
    }
}

impl<T, A> LinearOperator<T> for CancellableLinearOperator<'_, T, A>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_cols()
    }

    fn apply(&self, x: &[T]) -> Vec<T> {
        self.poll();
        let value = self.inner.apply(x);
        self.poll();
        value
    }

    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.cancellation.check("differentiable Krylov iteration")?;
        let value = self.inner.try_apply(x)?;
        self.cancellation.check("differentiable Krylov iteration")?;
        Ok(value)
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        self.poll();
        self.inner.apply_into(x, y);
        self.poll();
    }

    fn apply_block(&self, x: &crate::dense::DenseMatrix<T>, y: &mut crate::dense::DenseMatrix<T>) {
        self.poll();
        self.inner.apply_block(x, y);
        self.poll();
    }

    fn apply_transpose(&self, x: &[T]) -> Vec<T> {
        self.poll();
        let value = self.inner.apply_transpose(x);
        self.poll();
        value
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.poll();
        self.inner.apply_transpose_into(x, y);
        self.poll();
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        self.poll();
        let value = self.inner.apply_hermitian(x);
        self.poll();
        value
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.inner.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.inner.hermitian_capability()
    }
}

/// Preconditioner counterpart to [`CancellableLinearOperator`].
struct CancellablePreconditioner<'a, T, P> {
    inner: &'a P,
    cancellation: &'a CancellationToken,
    marker: std::marker::PhantomData<fn() -> T>,
}

impl<'a, T, P> CancellablePreconditioner<'a, T, P> {
    fn new(inner: &'a P, cancellation: &'a CancellationToken) -> Self {
        Self {
            inner,
            cancellation,
            marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn poll(&self) {
        if self.cancellation.is_cancelled() {
            panic!("differentiable operation cancelled");
        }
    }
}

impl<T, P> Preconditioner<T> for CancellablePreconditioner<'_, T, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply(&self, r: &[T]) -> Vec<T> {
        self.poll();
        let value = self.inner.apply(r);
        self.poll();
        value
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        self.poll();
        self.inner.apply_into(r, y);
        self.poll();
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        self.poll();
        let value = self.inner.apply_transpose(r);
        self.poll();
        value
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.poll();
        self.inner.apply_transpose_into(r, y);
        self.poll();
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        self.poll();
        let value = self.inner.apply_hermitian(r);
        self.poll();
        value
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.inner.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.inner.hermitian_capability()
    }
}

impl<'a, T, A, P> DifferentiableLinearSolve<'a, T, A, P>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    /// Create a reusable solve handle after validating the Krylov settings
    /// and the operator shape.
    pub fn new(
        operator: &'a A,
        preconditioner: &'a P,
        config: GmresConfig<T::Real>,
    ) -> Result<Self, SolverError> {
        if operator.num_rows() != operator.num_cols() {
            return Err(SolverError::OperationFailed {
                operation: "differentiable solve",
                reason: format!(
                    "operator must be square ({} rows, {} columns)",
                    operator.num_rows(),
                    operator.num_cols()
                ),
            });
        }
        if config.max_iterations == 0 {
            return Err(SolverError::OperationFailed {
                operation: "differentiable solve",
                reason: "GMRES max_iterations must be greater than zero".to_string(),
            });
        }
        if config.restart == 0 {
            return Err(SolverError::OperationFailed {
                operation: "differentiable solve",
                reason: "GMRES restart must be greater than zero".to_string(),
            });
        }
        let tolerance = config
            .tolerance
            .to_f64()
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "differentiable solve",
                reason: "GMRES tolerance cannot be represented as f64".to_string(),
            })?;
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "differentiable solve",
                reason: "GMRES tolerance must be finite and positive".to_string(),
            });
        }
        Ok(Self {
            operator,
            preconditioner,
            config,
        })
    }

    /// Solve the forward system `A x = b`.
    pub fn solve_forward(&self, rhs: &[T]) -> Result<GmresSolution<T>, SolverError> {
        self.solve_forward_with_guess(rhs, None)
    }

    /// Solve the forward system while polling a cooperative cancellation
    /// token inside every operator and preconditioner action.
    pub fn solve_forward_cancellable(
        &self,
        rhs: &[T],
        cancellation: &CancellationToken,
    ) -> Result<GmresSolution<T>, SolverError> {
        cancellation.check("differentiable forward")?;
        if rhs.len() != self.operator.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.operator.num_cols(),
                got: rhs.len(),
            });
        }
        validate_finite_rhs(rhs, "differentiable forward solve")?;
        let operator = CancellableLinearOperator::new(self.operator, cancellation);
        let preconditioner = CancellablePreconditioner::new(self.preconditioner, cancellation);
        let checked_config = promoted_gmres_config(&self.config);
        let result = checked_gmres(&operator, rhs, self.config.tolerance, || {
            gmres_preconditioned_with_guess(&operator, &preconditioner, rhs, None, &checked_config)
        })?;
        cancellation.check("differentiable forward")?;
        Ok(result)
    }

    /// Solve the forward system `A x = b`, optionally continuing from `x₀`.
    ///
    /// The initial iterate is validated before any operator or preconditioner
    /// callback runs. The checked GMRES boundary still recomputes the true
    /// residual through the original operator.
    pub fn solve_forward_with_guess(
        &self,
        rhs: &[T],
        initial_guess: Option<&[T]>,
    ) -> Result<GmresSolution<T>, SolverError> {
        if rhs.len() != self.operator.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: self.operator.num_cols(),
                got: rhs.len(),
            });
        }
        validate_finite_rhs(rhs, "differentiable forward solve")?;
        validate_initial_guess(
            initial_guess,
            self.operator.num_cols(),
            "differentiable forward initial guess",
        )?;
        let checked_config = promoted_gmres_config(&self.config);
        checked_gmres(self.operator, rhs, self.config.tolerance, || {
            gmres_preconditioned_with_guess(
                self.operator,
                self.preconditioner,
                rhs,
                initial_guess,
                &checked_config,
            )
        })
    }

    /// Solve `Aᵀ λ = rhs` or `Aᴴ λ = rhs` using the corresponding adjoint
    /// action of the same operator and preconditioner.
    pub fn solve_adjoint(
        &self,
        rhs: &[T],
        kind: AdjointSolve,
    ) -> Result<GmresSolution<T>, SolverError> {
        self.solve_adjoint_with_guess(rhs, kind, None)
    }

    /// Solve the transpose/Hermitian adjoint while polling cooperative
    /// cancellation inside every adjoint action.
    pub fn solve_adjoint_cancellable(
        &self,
        rhs: &[T],
        kind: AdjointSolve,
        cancellation: &CancellationToken,
    ) -> Result<GmresSolution<T>, SolverError> {
        cancellation.check("differentiable adjoint")?;
        if rhs.len() != self.operator.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.operator.num_rows(),
                got: rhs.len(),
            });
        }
        validate_finite_rhs(rhs, "differentiable adjoint solve")?;
        self.validate_adjoint(kind)?;
        let operator = CancellableLinearOperator::new(self.operator, cancellation);
        let preconditioner = CancellablePreconditioner::new(self.preconditioner, cancellation);
        let adjoint_operator = AdjointOperator {
            inner: &operator,
            kind,
        };
        let adjoint_preconditioner = AdjointPreconditioner {
            inner: &preconditioner,
            kind,
        };
        let checked_config = promoted_gmres_config(&self.config);
        let result = checked_gmres(&adjoint_operator, rhs, self.config.tolerance, || {
            gmres_preconditioned_with_guess(
                &adjoint_operator,
                &adjoint_preconditioner,
                rhs,
                None,
                &checked_config,
            )
        })?;
        cancellation.check("differentiable adjoint")?;
        Ok(result)
    }

    /// Solve `Aᵀ λ = rhs` or `Aᴴ λ = rhs`, optionally continuing from `λ₀`.
    pub fn solve_adjoint_with_guess(
        &self,
        rhs: &[T],
        kind: AdjointSolve,
        initial_guess: Option<&[T]>,
    ) -> Result<GmresSolution<T>, SolverError> {
        if rhs.len() != self.operator.num_rows() {
            return Err(SolverError::DimensionMismatch {
                expected: self.operator.num_rows(),
                got: rhs.len(),
            });
        }
        validate_finite_rhs(rhs, "differentiable adjoint solve")?;
        validate_initial_guess(
            initial_guess,
            self.operator.num_rows(),
            "differentiable adjoint initial guess",
        )?;
        let (operator_capability, preconditioner_capability) = match kind {
            AdjointSolve::Transpose => (
                self.operator.transpose_capability(),
                self.preconditioner.transpose_capability(),
            ),
            AdjointSolve::Hermitian => (
                self.operator.hermitian_capability(),
                self.preconditioner.hermitian_capability(),
            ),
        };
        if operator_capability != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperatorOperation {
                operation: kind.operation(),
                operator: "linear operator",
            });
        }
        if preconditioner_capability != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: kind.preconditioner_operation(),
                preconditioner: "linear preconditioner",
            });
        }

        let operator = AdjointOperator {
            inner: self.operator,
            kind,
        };
        let preconditioner = AdjointPreconditioner {
            inner: self.preconditioner,
            kind,
        };
        let checked_config = promoted_gmres_config(&self.config);
        checked_gmres(&operator, rhs, self.config.tolerance, || {
            gmres_preconditioned_with_guess(
                &operator,
                &preconditioner,
                rhs,
                initial_guess,
                &checked_config,
            )
        })
    }

    /// Run a forward solve and optionally its adjoint solve while reusing the
    /// borrowed operator/preconditioner state.
    pub fn solve(
        &self,
        rhs: &[T],
        adjoint_rhs: Option<(&[T], AdjointSolve)>,
    ) -> Result<DifferentiableSolveResult<T>, SolverError> {
        let forward = self.solve_forward(rhs)?;
        let adjoint = adjoint_rhs
            .map(|(rhs, kind)| self.solve_adjoint(rhs, kind))
            .transpose()?;
        Ok(DifferentiableSolveResult { forward, adjoint })
    }

    /// Solve a block of forward right-hand sides with checked FGMRES.
    pub fn solve_forward_block(
        &self,
        rhs: &sonium_solvers_core::dense::DenseMatrix<T>,
        config: &BlockFgmresConfig<T::Real>,
    ) -> Result<BlockFgmresSolution<T>, SolverError> {
        self.solve_forward_block_with_guess(rhs, None, config)
    }

    /// Solve block forward right-hand sides while polling cooperative
    /// cancellation inside every operator and preconditioner action.
    pub fn solve_forward_block_cancellable(
        &self,
        rhs: &sonium_solvers_core::dense::DenseMatrix<T>,
        config: &BlockFgmresConfig<T::Real>,
        cancellation: &CancellationToken,
    ) -> Result<BlockFgmresSolution<T>, SolverError> {
        cancellation.check("differentiable block forward")?;
        let operator = CancellableLinearOperator::new(self.operator, cancellation);
        let preconditioner = CancellablePreconditioner::new(self.preconditioner, cancellation);
        let checked_config = promoted_block_gmres_config(config);
        let result = try_block_fgmres_preconditioned(
            &operator,
            rhs,
            &preconditioner,
            None,
            &checked_config,
        )?;
        cancellation.check("differentiable block forward")?;
        Ok(result)
    }

    /// Solve block forward right-hand sides with a checked FGMRES
    /// continuation iterate. The optional initial block has one column per
    /// right-hand side and is validated by the checked block boundary before
    /// any operator or preconditioner callback runs.
    pub fn solve_forward_block_with_guess(
        &self,
        rhs: &sonium_solvers_core::dense::DenseMatrix<T>,
        initial_guess: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
        config: &BlockFgmresConfig<T::Real>,
    ) -> Result<BlockFgmresSolution<T>, SolverError> {
        let checked_config = promoted_block_gmres_config(config);
        try_block_fgmres_preconditioned(
            self.operator,
            rhs,
            self.preconditioner,
            initial_guess,
            &checked_config,
        )
    }

    /// Solve a block transpose or Hermitian-adjoint system with checked FGMRES.
    pub fn solve_adjoint_block(
        &self,
        rhs: &sonium_solvers_core::dense::DenseMatrix<T>,
        kind: AdjointSolve,
        config: &BlockFgmresConfig<T::Real>,
    ) -> Result<BlockFgmresSolution<T>, SolverError> {
        self.solve_adjoint_block_with_guess(rhs, kind, None, config)
    }

    /// Solve block transpose/Hermitian adjoint right-hand sides while
    /// polling cooperative cancellation inside every adjoint action.
    pub fn solve_adjoint_block_cancellable(
        &self,
        rhs: &sonium_solvers_core::dense::DenseMatrix<T>,
        kind: AdjointSolve,
        config: &BlockFgmresConfig<T::Real>,
        cancellation: &CancellationToken,
    ) -> Result<BlockFgmresSolution<T>, SolverError> {
        cancellation.check("differentiable block adjoint")?;
        self.validate_adjoint(kind)?;
        let operator = CancellableLinearOperator::new(self.operator, cancellation);
        let preconditioner = CancellablePreconditioner::new(self.preconditioner, cancellation);
        let adjoint_operator = AdjointOperator {
            inner: &operator,
            kind,
        };
        let adjoint_preconditioner = AdjointPreconditioner {
            inner: &preconditioner,
            kind,
        };
        let checked_config = promoted_block_gmres_config(config);
        let result = try_block_fgmres_preconditioned(
            &adjoint_operator,
            rhs,
            &adjoint_preconditioner,
            None,
            &checked_config,
        )?;
        cancellation.check("differentiable block adjoint")?;
        Ok(result)
    }

    /// Solve a block transpose or Hermitian-adjoint system with a checked
    /// FGMRES continuation iterate.
    pub fn solve_adjoint_block_with_guess(
        &self,
        rhs: &sonium_solvers_core::dense::DenseMatrix<T>,
        kind: AdjointSolve,
        initial_guess: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
        config: &BlockFgmresConfig<T::Real>,
    ) -> Result<BlockFgmresSolution<T>, SolverError> {
        self.validate_adjoint(kind)?;
        let operator = AdjointOperator {
            inner: self.operator,
            kind,
        };
        let preconditioner = AdjointPreconditioner {
            inner: self.preconditioner,
            kind,
        };
        let checked_config = promoted_block_gmres_config(config);
        try_block_fgmres_preconditioned(
            &operator,
            rhs,
            &preconditioner,
            initial_guess,
            &checked_config,
        )
    }

    /// Run forward and optional transpose/Hermitian block solves together.
    pub fn solve_block(
        &self,
        rhs: &sonium_solvers_core::dense::DenseMatrix<T>,
        adjoint_rhs: Option<(&sonium_solvers_core::dense::DenseMatrix<T>, AdjointSolve)>,
        config: &BlockFgmresConfig<T::Real>,
    ) -> Result<DifferentiableBlockSolveResult<T>, SolverError> {
        let forward = self.solve_forward_block(rhs, config)?;
        let adjoint = adjoint_rhs
            .map(|(rhs, kind)| self.solve_adjoint_block(rhs, kind, config))
            .transpose()?;
        Ok(DifferentiableBlockSolveResult { forward, adjoint })
    }

    fn validate_adjoint(&self, kind: AdjointSolve) -> Result<(), SolverError> {
        match kind {
            AdjointSolve::Transpose => {
                if self.operator.transpose_capability() != AdjointCapability::Exact {
                    return Err(SolverError::UnsupportedOperatorOperation {
                        operation: kind.operation(),
                        operator: "linear operator",
                    });
                }
                if self.preconditioner.transpose_capability() != AdjointCapability::Exact {
                    return Err(SolverError::UnsupportedOperation {
                        operation: kind.preconditioner_operation(),
                        preconditioner: "linear preconditioner",
                    });
                }
            }
            AdjointSolve::Hermitian => {
                if self.operator.hermitian_capability() != AdjointCapability::Exact {
                    return Err(SolverError::UnsupportedOperatorOperation {
                        operation: kind.operation(),
                        operator: "linear operator",
                    });
                }
                if self.preconditioner.hermitian_capability() != AdjointCapability::Exact {
                    return Err(SolverError::UnsupportedOperation {
                        operation: kind.preconditioner_operation(),
                        preconditioner: "linear preconditioner",
                    });
                }
            }
        }
        Ok(())
    }
}

fn validate_finite_rhs<T: ComplexField>(
    rhs: &[T],
    operation: &'static str,
) -> Result<(), SolverError> {
    if rhs.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn validate_initial_guess<T: ComplexField>(
    initial_guess: Option<&[T]>,
    expected: usize,
    operation: &'static str,
) -> Result<(), SolverError> {
    let Some(initial_guess) = initial_guess else {
        return Ok(());
    };
    if initial_guess.len() != expected {
        return Err(SolverError::DimensionMismatch {
            expected,
            got: initial_guess.len(),
        });
    }
    if initial_guess.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

impl AdjointSolve {
    fn operation(self) -> &'static str {
        match self {
            Self::Transpose => "operator transpose",
            Self::Hermitian => "operator Hermitian transpose",
        }
    }

    fn preconditioner_operation(self) -> &'static str {
        match self {
            Self::Transpose => "preconditioner transpose",
            Self::Hermitian => "preconditioner Hermitian transpose",
        }
    }
}

/// Differentiable solves must not accept a projected Krylov convergence
/// estimate without checking the original operator.  A caller may disable
/// periodic exact checks on the general GMRES configuration, but doing so here
/// would turn a promoted residual rejection into an immediate error instead
/// of allowing GMRES to restart from the rejected candidate.
fn promoted_gmres_config<R: Clone>(config: &GmresConfig<R>) -> GmresConfig<R> {
    let mut checked = config.clone();
    checked.exact_residual_check_interval = checked.exact_residual_check_interval.max(1);
    checked
}

/// Apply the same original-operator gate to differentiable block solves,
/// regardless of the caller's cheaper projected-residual preference.
fn promoted_block_gmres_config<R: Clone>(config: &BlockFgmresConfig<R>) -> BlockFgmresConfig<R> {
    let mut checked = config.clone();
    checked.true_residual_check = true;
    checked.exact_residual_check_interval = checked.exact_residual_check_interval.max(1);
    checked
}

fn checked_gmres<T, A, F>(
    operator: &A,
    rhs: &[T],
    tolerance: T::Real,
    solve: F,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    F: FnOnce() -> GmresSolution<T>,
{
    let result = catch_unwind(AssertUnwindSafe(solve)).map_err(|payload| {
        let reason = if let Some(message) = payload.downcast_ref::<&'static str>() {
            (*message).to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "GMRES operation panicked".to_string()
        };
        SolverError::OperationFailed {
            operation: "GMRES solve",
            reason,
        }
    })?;
    if !result.residual.is_finite() || result.x.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "GMRES solve",
            reason: "generated solution or residual contains NaN or infinity".to_string(),
        });
    }
    let ax = operator.try_apply(&result.x)?;
    let residual = rhs
        .iter()
        .zip(ax.iter())
        .map(|(&b, &ax_i)| b - ax_i)
        .collect::<Vec<_>>();
    let residual_norm = T::vec_norm(&residual);
    let rhs_norm = T::vec_norm(rhs);
    let ax_norm = T::vec_norm(&ax);
    if !residual_norm.is_finite() || !rhs_norm.is_finite() || !ax_norm.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "GMRES residual verification",
            reason: "original-operator residual contains NaN or infinity".to_string(),
        });
    }
    let true_residual = if rhs_norm > T::Real::zero() {
        residual_norm / rhs_norm
    } else {
        residual_norm
    };
    if !true_residual.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "GMRES residual verification",
            reason: "original-operator relative residual is non-finite".to_string(),
        });
    }

    let true_converged = true_residual <= tolerance;
    let backward_error = T::normwise_backward_error(residual_norm, ax_norm, rhs_norm);
    if !backward_error.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "GMRES residual verification",
            reason: "original-operator backward error is non-finite".to_string(),
        });
    }
    let mut result = result;
    result.residual = true_residual;
    result.backward_error = backward_error;
    result.attainable_accuracy = if true_converged {
        AccuracyStatus::Attained
    } else {
        AccuracyStatus::NotAttained
    };
    result.converged = true_converged;
    if true_converged {
        result.status = SolverStatus::Converged;
        return Ok(result);
    }
    if result.status == SolverStatus::Converged {
        result.status = SolverStatus::MaxIterationsReached;
    }
    Err(SolverError::ConvergenceError {
        status: result.status,
        iterations: result.iterations,
        residual: true_residual.to_f64().unwrap_or(f64::NAN),
    })
}

struct AdjointOperator<'a, A> {
    inner: &'a A,
    kind: AdjointSolve,
}

impl<T, A> LinearOperator<T> for AdjointOperator<'_, A>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    fn num_rows(&self) -> usize {
        self.inner.num_cols()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_rows()
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        let values = match self.kind {
            AdjointSolve::Transpose => self.inner.apply_transpose(x),
            AdjointSolve::Hermitian => self.inner.apply_hermitian(x),
        };
        y.copy_from_slice(&values);
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        match self.kind {
            AdjointSolve::Transpose => self.inner.apply_into(x, y),
            AdjointSolve::Hermitian => {
                let conjugated_x: Vec<_> = x.iter().map(|value| value.conj()).collect();
                let values = self.inner.apply(&conjugated_x);
                for (output, value) in y.iter_mut().zip(values) {
                    *output = value.conj();
                }
            }
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

struct AdjointPreconditioner<'a, P> {
    inner: &'a P,
    kind: AdjointSolve,
}

impl<T, P> Preconditioner<T> for AdjointPreconditioner<'_, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply(&self, rhs: &[T]) -> Vec<T> {
        match self.kind {
            AdjointSolve::Transpose => self.inner.apply_transpose(rhs),
            AdjointSolve::Hermitian => self.inner.apply_hermitian(rhs),
        }
    }

    fn apply_into(&self, rhs: &[T], output: &mut [T]) {
        let values = self.apply(rhs);
        output.copy_from_slice(&values);
    }

    fn apply_transpose(&self, rhs: &[T]) -> Vec<T> {
        match self.kind {
            AdjointSolve::Transpose => self.inner.apply(rhs),
            AdjointSolve::Hermitian => {
                let conjugated_rhs: Vec<_> = rhs.iter().map(|value| value.conj()).collect();
                self.inner
                    .apply(&conjugated_rhs)
                    .into_iter()
                    .map(|value| value.conj())
                    .collect()
            }
        }
    }

    fn apply_hermitian(&self, rhs: &[T]) -> Vec<T> {
        match self.kind {
            AdjointSolve::Transpose => {
                let conjugated_rhs: Vec<_> = rhs.iter().map(|value| value.conj()).collect();
                self.inner
                    .apply(&conjugated_rhs)
                    .into_iter()
                    .map(|value| value.conj())
                    .collect()
            }
            AdjointSolve::Hermitian => self.inner.apply(rhs),
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CompositeOperator, CsrMatrix, ScaledOperator, SumOperator};
    use crate::traits::IdentityPreconditioner;
    use num_complex::Complex64;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn config() -> GmresConfig<f64> {
        GmresConfig {
            max_iterations: 32,
            restart: 8,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        }
    }

    #[test]
    fn differentiable_forward_rejects_non_finite_generated_state() {
        struct NonFiniteOperator;
        impl LinearOperator<f64> for NonFiniteOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
                y[0] = f64::NAN;
            }

            fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
                y[0] = f64::NAN;
            }
        }

        let handle =
            DifferentiableLinearSolve::new(&NonFiniteOperator, &IdentityPreconditioner, config())
                .expect("square matrix-free operator should be accepted");
        let error = handle
            .solve_forward(&[1.0])
            .expect_err("generated NaN must be rejected");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn differentiable_krylov_cancellation_interrupts_operator_and_adjoint_actions() {
        struct CancellingOperator<'a> {
            token: &'a CancellationToken,
        }

        impl LinearOperator<f64> for CancellingOperator<'_> {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply_into(&self, x: &[f64], y: &mut [f64]) {
                y.copy_from_slice(x);
                self.token.cancel();
            }

            fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
                y.copy_from_slice(x);
                self.token.cancel();
            }

            fn transpose_capability(&self) -> AdjointCapability {
                AdjointCapability::Exact
            }

            fn hermitian_capability(&self) -> AdjointCapability {
                AdjointCapability::Exact
            }
        }

        let forward_token = CancellationToken::new();
        let forward_operator = CancellingOperator {
            token: &forward_token,
        };
        let forward_handle =
            DifferentiableLinearSolve::new(&forward_operator, &IdentityPreconditioner, config())
                .expect("square operator should be accepted");
        let forward_error = forward_handle
            .solve_forward_cancellable(&[1.0, 0.0], &forward_token)
            .expect_err("cancellation during operator action must stop forward solve");
        assert!(matches!(
            forward_error,
            SolverError::OperationFailed { ref reason, .. } if reason.contains("cancelled")
        ));

        let adjoint_token = CancellationToken::new();
        let adjoint_operator = CancellingOperator {
            token: &adjoint_token,
        };
        let adjoint_handle =
            DifferentiableLinearSolve::new(&adjoint_operator, &IdentityPreconditioner, config())
                .expect("square operator should be accepted");
        let adjoint_error = adjoint_handle
            .solve_adjoint_cancellable(&[1.0, 0.0], AdjointSolve::Transpose, &adjoint_token)
            .expect_err("cancellation during adjoint action must stop solve");
        assert!(matches!(
            adjoint_error,
            SolverError::OperationFailed { ref reason, .. } if reason.contains("cancelled")
        ));

        let block_token = CancellationToken::new();
        let block_operator = CancellingOperator {
            token: &block_token,
        };
        let block_handle =
            DifferentiableLinearSolve::new(&block_operator, &IdentityPreconditioner, config())
                .expect("square operator should be accepted");
        let block_rhs = crate::dense::DenseMatrix::from_row_major_vec(2, 1, vec![1.0, 0.0]);
        let block_config = BlockFgmresConfig {
            max_iterations: 4,
            restart: 1,
            tolerance: 1.0e-12,
            ..BlockFgmresConfig::default()
        };
        let block_error = block_handle
            .solve_forward_block_cancellable(&block_rhs, &block_config, &block_token)
            .expect_err("cancellation during block action must stop solve");
        assert!(matches!(
            block_error,
            SolverError::OperationFailed { ref reason, .. } if reason.contains("cancelled")
        ));
    }

    #[test]
    fn differentiable_rejects_non_finite_rhs_before_krylov() {
        struct CountingOperator<'a> {
            applications: &'a AtomicUsize,
        }

        impl LinearOperator<f64> for CountingOperator<'_> {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, x: &[f64], y: &mut [f64]) {
                self.applications.fetch_add(1, Ordering::Relaxed);
                y.copy_from_slice(x);
            }

            fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
                self.applications.fetch_add(1, Ordering::Relaxed);
                y.copy_from_slice(x);
            }
        }

        let applications = AtomicUsize::new(0);
        let operator = CountingOperator {
            applications: &applications,
        };
        let handle = DifferentiableLinearSolve::new(&operator, &IdentityPreconditioner, config())
            .expect("square operator should be accepted");

        let forward_error = handle
            .solve_forward(&[f64::NAN])
            .expect_err("non-finite forward RHS must be rejected");
        assert!(matches!(
            forward_error,
            SolverError::OperationFailed {
                operation: "differentiable forward solve",
                ..
            }
        ));
        assert_eq!(applications.load(Ordering::Relaxed), 0);

        let adjoint_error = handle
            .solve_adjoint(&[f64::INFINITY], AdjointSolve::Hermitian)
            .expect_err("non-finite adjoint RHS must be rejected");
        assert!(matches!(
            adjoint_error,
            SolverError::OperationFailed {
                operation: "differentiable adjoint solve",
                ..
            }
        ));
        assert_eq!(applications.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn differentiable_checked_result_gates_on_original_operator_residual() {
        struct IdentityOperator;
        impl LinearOperator<f64> for IdentityOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, x: &[f64], y: &mut [f64]) {
                y.copy_from_slice(x);
            }

            fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
                y.copy_from_slice(x);
            }
        }

        let reported = GmresSolution {
            x: vec![0.0],
            iterations: 1,
            restarts: 0,
            residual: 0.0,
            estimated_residual: 0.0,
            backward_error: 0.0,
            attainable_accuracy: sonium_solvers_core::traits::AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts: Default::default(),
        };
        let error = checked_gmres(&IdentityOperator, &[1.0], 1.0e-8, || reported)
            .expect_err("a false inner estimate must not report success");
        assert!(matches!(
            error,
            SolverError::ConvergenceError { residual, .. } if (residual - 1.0).abs() < 1.0e-12
        ));
    }

    #[test]
    fn real_forward_and_transpose_share_solver_state() {
        let matrix = CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 3.0)],
        );
        let handle = DifferentiableLinearSolve::new(&matrix, &IdentityPreconditioner, config())
            .expect("square system should be accepted");
        let result = handle
            .solve(&[6.0, 8.0], Some((&[8.0, 7.0], AdjointSolve::Transpose)))
            .expect("forward and transpose solves should converge");
        assert_eq!(result.forward.x.len(), 2);
        assert_eq!(result.adjoint.as_ref().expect("adjoint result").x.len(), 2);
        assert_eq!(result.forward.attainable_accuracy, AccuracyStatus::Attained);
        let forward_backward_error: f64 = result.forward.backward_error;
        assert!(forward_backward_error.is_finite());
        assert_eq!(
            result
                .adjoint
                .as_ref()
                .expect("adjoint result")
                .attainable_accuracy,
            AccuracyStatus::Attained
        );
        let adjoint_backward_error: f64 = result
            .adjoint
            .as_ref()
            .expect("adjoint result")
            .backward_error;
        assert!(adjoint_backward_error.is_finite());
        for (actual, expected) in result.forward.x.iter().zip([1.0_f64, 2.0_f64]) {
            assert!((*actual - expected).abs() < 1.0e-10);
        }
        for (actual, expected) in result
            .adjoint
            .expect("adjoint result")
            .x
            .iter()
            .zip([1.0_f64, 2.0_f64])
        {
            assert!((*actual - expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn complex_hermitian_solve_matches_known_solution() {
        let i = Complex64::new(0.0, 1.0);
        let matrix = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(3.0, 1.0)),
                (0, 1, Complex64::new(1.0, -2.0)),
                (1, 0, 2.0 * i),
                (1, 1, Complex64::new(4.0, -1.0)),
            ],
        );
        let x = [Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.5)];
        let rhs = matrix.apply(&x);
        let adjoint_x = [Complex64::new(-0.5, 1.0), Complex64::new(1.5, -2.0)];
        let adjoint_rhs = matrix.apply_hermitian(&adjoint_x);
        let handle = DifferentiableLinearSolve::new(&matrix, &IdentityPreconditioner, config())
            .expect("square system should be accepted");
        let result = handle
            .solve(&rhs, Some((&adjoint_rhs, AdjointSolve::Hermitian)))
            .expect("complex forward and Hermitian solves should converge");
        for (actual, expected) in result.forward.x.iter().zip(x) {
            assert!((*actual - expected).norm() < 1.0e-10);
        }
        for (actual, expected) in result
            .adjoint
            .expect("adjoint result")
            .x
            .iter()
            .zip(adjoint_x)
        {
            assert!((*actual - expected).norm() < 1.0e-10);
        }
    }

    #[test]
    fn composed_complex_adjoint_solves_preserve_transpose_and_hermitian_semantics() {
        let a = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(2.0, 1.0)),
                (0, 1, Complex64::new(-1.0, 0.5)),
                (1, 0, Complex64::new(0.25, -2.0)),
                (1, 1, Complex64::new(3.0, -0.75)),
            ],
        );
        let b = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(1.0, -0.5)),
                (0, 1, Complex64::new(0.75, 1.25)),
                (1, 0, Complex64::new(-1.5, 0.25)),
                (1, 1, Complex64::new(0.5, 2.0)),
            ],
        );
        let sum = SumOperator::new(&a, &b);
        let scaled = ScaledOperator::new(&sum, Complex64::new(0.5, -1.0));
        let composite = CompositeOperator::new(&scaled, &b);
        assert_eq!(composite.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(composite.hermitian_capability(), AdjointCapability::Exact);

        let x = [Complex64::new(1.0, -0.5), Complex64::new(-0.75, 1.25)];
        let lambda = [Complex64::new(-0.25, 0.75), Complex64::new(1.5, -1.0)];
        let rhs = composite.apply(&x);
        let transpose_rhs = composite.apply_transpose(&lambda);
        let hermitian_rhs = composite.apply_hermitian(&lambda);
        assert!(
            transpose_rhs
                .iter()
                .zip(hermitian_rhs.iter())
                .any(|(transpose, hermitian)| (*transpose - *hermitian).norm() > 1.0e-8),
            "complex transpose and Hermitian actions must remain distinct"
        );

        let handle = DifferentiableLinearSolve::new(&composite, &IdentityPreconditioner, config())
            .expect("composed square operator should be accepted");
        let forward = handle
            .solve_forward(&rhs)
            .expect("composed forward solve should converge");
        let transpose = handle
            .solve_adjoint(&transpose_rhs, AdjointSolve::Transpose)
            .expect("composed transpose solve should converge");
        let hermitian = handle
            .solve_adjoint(&hermitian_rhs, AdjointSolve::Hermitian)
            .expect("composed Hermitian solve should converge");

        for (actual, expected) in forward.x.iter().zip(x) {
            assert!((*actual - expected).norm() < 1.0e-9);
        }
        for (actual, expected) in transpose.x.iter().zip(lambda) {
            assert!((*actual - expected).norm() < 1.0e-9);
        }
        for (actual, expected) in hermitian.x.iter().zip(lambda) {
            assert!((*actual - expected).norm() < 1.0e-9);
        }
    }

    struct ForwardOnly;

    impl LinearOperator<f64> for ForwardOnly {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            y.copy_from_slice(x);
        }

        fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("transpose intentionally unavailable");
        }

        fn transpose_capability(&self) -> AdjointCapability {
            AdjointCapability::Unsupported
        }

        fn hermitian_capability(&self) -> AdjointCapability {
            AdjointCapability::Unsupported
        }
    }

    #[test]
    fn unsupported_adjoint_is_rejected_before_krylov_iteration() {
        let handle =
            DifferentiableLinearSolve::new(&ForwardOnly, &IdentityPreconditioner, config())
                .expect("forward-only square system should be accepted");
        let error = handle
            .solve_adjoint(&[1.0], AdjointSolve::Hermitian)
            .expect_err("unsupported Hermitian action must fail");
        assert!(matches!(
            error,
            SolverError::UnsupportedOperatorOperation { .. }
        ));
    }
}
