//! MINRES (Minimal Residual) solver for Hermitian indefinite systems
//!
//! Paige–Saunders MINRES in the short-recurrence form of Choi (2006),
//! Algorithm 4. MINRES is the method of choice for symmetric / Hermitian
//! **possibly indefinite** systems: it minimises the residual over the Krylov
//! space like GMRES but needs only a three-term Lanczos recurrence, so storage
//! is O(n) with no Krylov basis and each iteration costs one operator
//! application, one preconditioner application, and two inner products.
//!
//! **Applicability.** A must be Hermitian (real symmetric or complex
//! Hermitian); it may be indefinite or even singular (consistent systems
//! converge). The preconditioner M must be Hermitian positive definite: the
//! algorithm runs Lanczos on `M⁻¹A` in the M inner product, which is exactly
//! the symmetric-preconditioned system `M^{-1/2} A M^{-1/2}` without ever
//! forming the square root.
//!
//! **Complex Hermitian case.** With conjugated inner products the Lanczos
//! tridiagonal stays real symmetric: `α_k = (z_k, A z_k)` is real by
//! Hermiticity of A and the `β_k` are norms, so the Givens rotations are real
//! and the same code path serves real and complex systems. (This is *not* the
//! complex-symmetric MINRES variant, which needs a different recurrence.)
//!
//! **Guards.** Following the reference implementations (Paige–Saunders /
//! Choi / MATLAB `minres`):
//! - `‖A‖`-scaled breakdown: the Lanczos `β_{k+1}` is compared against a
//!   threshold relative to the running Frobenius estimate `‖T_k‖_F` of the
//!   preconditioned operator norm, not an absolute floor — exact Krylov
//!   exhaustion ("lucky breakdown") then terminates with the exact minimiser.
//! - Ill-conditioning: a running condition estimate `‖T_k‖_F / min_k |R_kk|`
//!   of the projected tridiagonal is compared against
//!   [`MinresConfig::condition_limit`]; beyond it the best iterate is
//!   returned with [`SolverStatus::Breakdown`].
//! - Singular projected system (`γ_k ≈ 0` with `β_{k+1} ≈ 0`): the step is
//!   skipped and the solve exits with [`SolverStatus::Stagnated`] unless the
//!   true residual already meets the tolerance.
//!
//! The stopping test is the standard relative preconditioned residual
//! estimate `‖M⁻¹r_k‖_M / ‖b‖_{M⁻¹}` (which reduces to `‖r_k‖₂/‖b‖₂`
//! unpreconditioned), tracked through the Givens rotations at O(1) cost.
//! Every exit path recomputes the exact unpreconditioned relative residual
//! `‖b − Ax‖₂/‖b‖₂` (one extra SpMV per solve) so the reported `residual`
//! and the `converged` flag are always honest, mirroring the
//! pathological-preconditioner guard in `gmres.rs`.
//!
//! References:
//! - Paige, C.C. & Saunders, M.A. (1975). "Solution of sparse indefinite
//!   systems of linear equations." SIAM J. Numer. Anal. 12(4).
//! - Choi, S.-C.T. (2006). "Iterative Methods for Singular Linear Equations
//!   and Least-Squares Problems." PhD Thesis, Stanford. Algorithm 4.
//! - Fong, D.C.-L. & Saunders, M.A. (2012). "CG versus MINRES: An empirical
//!   comparison." SQU Journal for Science 17(1).

use super::cgs_orth::KrylovOpCounts;
use super::gmres_common::exact_relative_residual;
use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::blas_helpers::{inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

/// MINRES solver configuration.
#[derive(Debug, Clone)]
pub struct MinresConfig<R> {
    /// Maximum number of Lanczos iterations (one SpMV + one preconditioner
    /// application each).
    pub max_iterations: usize,
    /// Relative tolerance on the preconditioned residual norm
    /// `‖M⁻¹r‖_M / ‖b‖_{M⁻¹}` (`‖r‖₂/‖b‖₂` when unpreconditioned).
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output).
    pub print_interval: usize,
    /// Ill-conditioning guard: stop with [`SolverStatus::Breakdown`] when the
    /// running condition estimate of the projected tridiagonal exceeds this
    /// limit (default 1e14, following Paige–Saunders).
    pub condition_limit: R,
}

impl Default for MinresConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
            condition_limit: 1e14,
        }
    }
}

/// MINRES solver result.
#[derive(Debug)]
pub struct MinresSolution<T: ComplexField> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of Lanczos iterations (= number of operator applications).
    pub iterations: usize,
    /// Exact relative residual `‖b − Ax‖₂/‖b‖₂` at exit.
    pub residual: T::Real,
    /// Residual estimate captured before the independent final check.
    pub estimated_residual: T::Real,
    /// Normwise backward error against the original operator.
    pub backward_error: T::Real,
    /// Whether the returned iterate satisfies the requested true-residual tolerance.
    pub attainable_accuracy: AccuracyStatus,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Solver status.
    pub status: SolverStatus,
    /// Final running condition estimate of the Lanczos tridiagonal (the
    /// `‖A‖`-scaled ill-conditioning guard).
    pub condition_estimate: T::Real,
    /// Memory-traffic oriented operation counts for the Krylov solve.
    pub op_counts: KrylovOpCounts,
}

fn numerical_error_solution<T: ComplexField>(x: Vec<T>) -> MinresSolution<T> {
    MinresSolution {
        x,
        iterations: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        condition_estimate: T::Real::infinity(),
        op_counts: KrylovOpCounts::default(),
    }
}

fn compatibility_error_solution<T: ComplexField>(b: &[T], x0: Option<&[T]>) -> MinresSolution<T> {
    numerical_error_solution(match x0 {
        Some(guess) if guess.len() == b.len() => guess.to_vec(),
        _ => vec![T::zero(); b.len()],
    })
}

fn invalid_minres_input<T: ComplexField, A: LinearOperator<T>>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> Option<MinresSolution<T>> {
    let n = b.len();
    let invalid = config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
        || !config.condition_limit.is_finite()
        || config.condition_limit <= T::Real::zero()
        || operator.num_rows() != n
        || operator.num_cols() != n
        || b.iter().any(|value| !value.norm().is_finite())
        || x0.is_some_and(|guess| {
            guess.len() != n || guess.iter().any(|value| !value.norm().is_finite())
        });

    invalid.then(|| {
        numerical_error_solution(match x0 {
            Some(guess) if guess.len() == n => guess.to_vec(),
            _ => vec![T::zero(); n],
        })
    })
}

/// Checked unpreconditioned MINRES entry point.
pub fn try_minres<T, A>(
    operator: &A,
    b: &[T],
    config: &MinresConfig<T::Real>,
) -> Result<MinresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    try_minres_unpreconditioned(operator, b, None, config)
}

/// Checked unpreconditioned MINRES entry point with an optional initial
/// guess.
pub fn try_minres_with_guess<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> Result<MinresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    try_minres_unpreconditioned(operator, b, x0, config)
}

/// Checked preconditioned MINRES entry point.
pub fn try_minres_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &MinresConfig<T::Real>,
) -> Result<MinresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_minres_preconditioned_with_guess(operator, precond, b, None, config)
}

/// Checked preconditioned MINRES entry point with an optional initial guess.
pub fn try_minres_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> Result<MinresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let invalid_preconditioner_output = AtomicBool::new(false);
    let checked_operator = CheckedMinresOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let checked_precond = CheckedMinresPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner_output,
    };
    try_minres_checked(
        &checked_operator,
        Some(&checked_precond),
        operator,
        b,
        x0,
        config,
        &invalid_operator_output,
        &invalid_preconditioner_output,
    )
}

fn try_minres_unpreconditioned<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> Result<MinresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let checked_operator = CheckedMinresOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    try_minres_checked(
        &checked_operator,
        None::<&CheckedMinresPreconditioner<'_, sonium_solvers_core::traits::IdentityPreconditioner>>,
        operator,
        b,
        x0,
        config,
        &invalid_operator_output,
        &AtomicBool::new(false),
    )
}

#[allow(clippy::too_many_arguments)]
fn try_minres_checked<T, A, OrigA, OrigP>(
    checked_operator: &CheckedMinresOperator<'_, OrigA>,
    precond: Option<&CheckedMinresPreconditioner<'_, OrigP>>,
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
    invalid_operator_output: &AtomicBool,
    invalid_preconditioner_output: &AtomicBool,
) -> Result<MinresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    OrigA: LinearOperator<T>,
    OrigP: Preconditioner<T>,
{
    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_minres_inputs(operator, b, x0, config)?;
        let solution = match precond {
            Some(precond) => minres_core(checked_operator, Some(precond), b, x0, config),
            None => minres_core(
                checked_operator,
                None::<&CheckedMinresPreconditioner<'_, OrigP>>,
                b,
                x0,
                config,
            ),
        };
        let solution = finalize_minres_solution(checked_operator, b, config.tolerance, solution);
        finish_checked_minres_solution(
            solution,
            b.len(),
            invalid_operator_output.load(Ordering::Relaxed),
            invalid_preconditioner_output.load(Ordering::Relaxed),
        )
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "MINRES",
        reason: panic_reason(payload),
    })??;
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn validate_minres_inputs<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
        || !config.condition_limit.is_finite()
        || config.condition_limit <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "MINRES",
            reason: "invalid MINRES configuration".to_string(),
        });
    }
    let rows = operator.num_rows();
    let cols = operator.num_cols();
    if rows != cols {
        return Err(SolverError::DimensionMismatch {
            expected: cols,
            got: rows,
        });
    }
    if b.len() != cols {
        return Err(SolverError::DimensionMismatch {
            expected: cols,
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "MINRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(guess) = x0
        && guess.len() != cols
    {
        return Err(SolverError::DimensionMismatch {
            expected: cols,
            got: guess.len(),
        });
    }
    if let Some(guess) = x0
        && guess.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "MINRES",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn finish_checked_minres_solution<T: ComplexField>(
    solution: MinresSolution<T>,
    expected_dimension: usize,
    invalid_operator_output: bool,
    invalid_preconditioner_output: bool,
) -> Result<MinresSolution<T>, SolverError> {
    if invalid_operator_output {
        return Err(SolverError::OperationFailed {
            operation: "MINRES operator",
            reason: "operator generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if invalid_preconditioner_output {
        return Err(SolverError::OperationFailed {
            operation: "MINRES preconditioner",
            reason: "preconditioner generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if solution.status == SolverStatus::NumericalError
        || solution.x.len() != expected_dimension
        || !solution.residual.is_finite()
        || !solution.estimated_residual.is_finite()
        || !solution.backward_error.is_finite()
        || !solution.condition_estimate.is_finite()
        || solution.x.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "MINRES",
            reason: "generated NaN, infinity, or an invalid solution".to_string(),
        });
    }
    Ok(solution)
}

fn finalize_minres_solution<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut result: MinresSolution<T>,
) -> MinresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if result.status == SolverStatus::NumericalError
        || result.x.len() != operator.num_cols()
        || b.len() != operator.num_rows()
        || result.x.iter().any(|value| !value.norm().is_finite())
        || b.iter().any(|value| !value.norm().is_finite())
    {
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        result.converged = false;
        result.status = SolverStatus::NumericalError;
        return result;
    }

    let mut ax = vec![T::zero(); b.len()];
    operator.apply_into(&result.x, &mut ax);
    result.op_counts.spmv_count += 1;
    let mut residual_norm_sq = T::Real::zero();
    let mut ax_norm_sq = T::Real::zero();
    let mut rhs_norm_sq = T::Real::zero();
    for (rhs, applied) in b.iter().zip(&mut ax) {
        let ax_value = *applied;
        *applied = *rhs - ax_value;
        if !applied.norm().is_finite() || !ax_value.norm().is_finite() {
            result.residual = T::Real::infinity();
            result.backward_error = T::Real::infinity();
            result.attainable_accuracy = AccuracyStatus::NotChecked;
            result.converged = false;
            result.status = SolverStatus::NumericalError;
            return result;
        }
        residual_norm_sq += applied.norm_sqr();
        ax_norm_sq += ax_value.norm_sqr();
        rhs_norm_sq += rhs.norm_sqr();
    }
    if !residual_norm_sq.is_finite() || !ax_norm_sq.is_finite() || !rhs_norm_sq.is_finite() {
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        result.converged = false;
        result.status = SolverStatus::NumericalError;
        return result;
    }
    let residual_norm = residual_norm_sq.sqrt();
    let rhs_norm = rhs_norm_sq.sqrt();
    let ax_norm = ax_norm_sq.sqrt();
    result.op_counts.norm_count += 3;
    result.residual = residual_norm / rhs_norm.max(T::Real::min_positive_value());
    result.backward_error = T::normwise_backward_error(residual_norm, ax_norm, rhs_norm);
    if !result.residual.is_finite() || !result.backward_error.is_finite() {
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        result.converged = false;
        result.status = SolverStatus::NumericalError;
    } else if result.residual <= tolerance {
        result.converged = true;
        result.status = SolverStatus::Converged;
        result.attainable_accuracy = AccuracyStatus::Attained;
    } else {
        result.converged = false;
        if result.status == SolverStatus::Converged {
            result.status = SolverStatus::Stagnated;
        }
        result.attainable_accuracy = AccuracyStatus::NotAttained;
    }
    result
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "MINRES callback panicked".to_string()
    }
}

struct CheckedMinresOperator<'a, A> {
    inner: &'a A,
    invalid_output: &'a AtomicBool,
}

impl<T, A> LinearOperator<T> for CheckedMinresOperator<'_, A>
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

    fn apply(&self, input: &[T]) -> Vec<T> {
        let mut output = self.inner.apply(input);
        self.validate_forward(&mut output);
        output
    }

    fn apply_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_into(input, output);
        self.validate_forward(output);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.inner.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.inner.hermitian_capability()
    }

    fn apply_transpose_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_transpose_into(input, output);
        if output.len() != self.num_cols() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }

    fn apply_hermitian(&self, input: &[T]) -> Vec<T> {
        let mut output = self.inner.apply_hermitian(input);
        if output.len() != self.num_cols() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
        output
    }
}

impl<A> CheckedMinresOperator<'_, A> {
    fn validate_forward<T>(&self, output: &mut [T])
    where
        T: ComplexField,
        A: LinearOperator<T>,
    {
        if output.len() != self.inner.num_rows()
            || output.iter().any(|value| !value.norm().is_finite())
        {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

struct CheckedMinresPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for CheckedMinresPreconditioner<'_, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply(&self, input: &[T]) -> Vec<T> {
        let mut output = self.inner.apply(input);
        self.validate(input, &mut output);
        output
    }

    fn apply_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_into(input, output);
        self.validate(input, output);
    }
}

impl<P> CheckedMinresPreconditioner<'_, P> {
    fn validate<T: ComplexField>(&self, input: &[T], output: &mut [T]) {
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

/// Solve Ax = b using MINRES (unpreconditioned).
///
/// A must be Hermitian (possibly indefinite). For SPD systems [`crate::cg`]
/// is slightly cheaper per iteration; for non-symmetric systems use GMRES.
pub fn minres<T, A>(operator: &A, b: &[T], config: &MinresConfig<T::Real>) -> MinresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_minres(operator, b, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b, None),
    }
}

/// Solve Ax = b using MINRES with an initial guess.
pub fn minres_with_guess<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> MinresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_minres_with_guess(operator, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b, x0),
    }
}

/// Solve Ax = b using preconditioned MINRES.
///
/// Both A and the preconditioner M must be Hermitian, and M must be positive
/// definite. The method runs Lanczos on `M⁻¹A` in the M inner product, which
/// is equivalent to the symmetric-preconditioned system
/// `M^{-1/2} A M^{-1/2} x̂ = M^{-1/2} b` without forming `M^{1/2}`.
pub fn minres_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &MinresConfig<T::Real>,
) -> MinresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_minres_preconditioned(operator, precond, b, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b, None),
    }
}

/// Solve Ax = b using preconditioned MINRES with an initial guess.
pub fn minres_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> MinresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_minres_preconditioned_with_guess(operator, precond, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b, x0),
    }
}

fn apply_precond_into<T: ComplexField, P: Preconditioner<T>>(
    precond: Option<&P>,
    r: &[T],
    z: &mut [T],
) {
    match precond {
        Some(pc) => pc.apply_into(r, z),
        None => z.copy_from_slice(r),
    }
}

fn minres_core<T, A, P>(
    operator: &A,
    precond: Option<&P>,
    b: &[T],
    x0: Option<&[T]>,
    config: &MinresConfig<T::Real>,
) -> MinresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if let Some(invalid) = invalid_minres_input(operator, b, x0, config) {
        return invalid;
    }

    let n = b.len();
    let has_preconditioner = precond.is_some();
    let mut op_counts = KrylovOpCounts::default();

    let zero = T::Real::zero();
    let one = T::Real::one();

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };

    // True RHS norm (unpreconditioned) for honest exit reporting, mirroring
    // gmres.rs: a near-zero RHS switches the reference norm to 1.
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    let raw_true_b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let rhs_is_near_zero = raw_true_b_norm < tol_threshold;
    let true_b_norm = if rhs_is_near_zero {
        one
    } else {
        raw_true_b_norm
    };

    // Preconditioned RHS scale ‖b‖_{M⁻¹} = sqrt(<M⁻¹ b, b>) for the relative
    // stopping test (equals ‖b‖₂ when unpreconditioned). Positivity of the
    // inner product is the cheap HPD check on the preconditioner.
    let mut work = vec![T::zero(); n];
    apply_precond_into(precond, b, &mut work);
    if has_preconditioner {
        op_counts.preconditioner_count += 1;
    }
    let pb_sq = inner_product(&work, b).re();
    op_counts.dot_reductions += 1;
    if !rhs_is_near_zero && pb_sq <= zero {
        // <M⁻¹b, b> must be positive — M is not Hermitian positive definite.
        return MinresSolution {
            x,
            iterations: 0,
            residual: one,
            estimated_residual: one,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Breakdown,
            condition_estimate: zero,
            op_counts,
        };
    }
    let preconditioned_b_norm = if rhs_is_near_zero { one } else { pb_sq.sqrt() };

    // r = b − A x (true residual), then z = M⁻¹ r.
    let mut r = b.to_vec();
    if x0.is_some() {
        operator.apply_into(&x, &mut work);
        op_counts.spmv_count += 1;
        for i in 0..n {
            r[i] = b[i] - work[i];
        }
        op_counts.axpy_passes += 1;
    }
    let mut z_cur = vec![T::zero(); n];
    apply_precond_into(precond, &r, &mut z_cur);
    if has_preconditioner {
        op_counts.preconditioner_count += 1;
    }
    let beta1_sq = inner_product(&z_cur, &r).re();
    op_counts.dot_reductions += 1;
    if beta1_sq <= zero {
        // Either r = 0 (the initial guess is the exact solution) or M⁻¹ is
        // not HPD along r. Decide on the true residual.
        let actual =
            exact_relative_residual(operator, &x, b, &mut work, true_b_norm, &mut op_counts);
        let converged = actual <= config.tolerance;
        return MinresSolution {
            x,
            iterations: 0,
            residual: actual,
            estimated_residual: actual,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged,
            status: if converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Breakdown
            },
            condition_estimate: zero,
            op_counts,
        };
    }
    let beta1 = beta1_sq.sqrt();

    // Lanczos state: z_k with z_k = M⁻¹ v_k, ‖z_k‖_M = 1 (i.e. <z_k, v_k> = 1).
    let inv_beta1 = T::from_real(one / beta1);
    let mut v_cur = vec![T::zero(); n];
    for i in 0..n {
        v_cur[i] = r[i] * inv_beta1;
        z_cur[i] *= inv_beta1;
    }
    drop(r);
    op_counts.axpy_passes += 1;

    let mut v_prev = vec![T::zero(); n];
    let mut z_prev = vec![T::zero(); n];
    let mut v_new = vec![T::zero(); n];
    let mut z_new = vec![T::zero(); n];
    let mut w_av = vec![T::zero(); n]; // A z_k
    let mut u = vec![T::zero(); n]; // M⁻¹ A z_k
    let mut w_dir = vec![T::zero(); n]; // w_k update direction
    let mut wl = vec![T::zero(); n]; // w_{k−1}
    let mut wl2 = vec![T::zero(); n]; // w_{k−2}

    // QR-of-tridiagonal scalar state. Everything here is real: A Hermitian
    // and M HPD make the Lanczos tridiagonal real symmetric (α_k = (z_k, A z_k)
    // is real, the β_k are norms), so the Givens rotations are real too.
    let mut beta = zero; // β_k entering the recurrence (β_1 := 0)
    let mut cs1 = one; // c_{k−1} (c_0 = 1)
    let mut sn1 = zero; // s_{k−1} (s_0 = 0)
    let mut cs2 = one; // c_{k−2}
    let mut sn2 = zero; // s_{k−2}
    let mut phi_tilde = beta1; // φ̃_1; the residual estimate is |φ̃_{k+1}|
    let mut anorm_sq = zero; // running ‖T_k‖_F² (≈ ‖M^{-1/2} A M^{-1/2}‖²)
    let mut gmin = T::Real::infinity(); // smallest |R[k,k]| so far
    let mut acond = zero;

    let gama_floor = T::Real::from_f64(1e-30).unwrap();
    // Krylov-exhaustion threshold relative to the running operator-norm
    // estimate (the ‖A‖-scaled breakdown guard).
    let breakdown_rel = T::Real::from_f64(1e-13).unwrap();

    for iter in 1..=config.max_iterations {
        // ============================================================
        // Lanczos three-term recurrence in the M inner product:
        //   β_{k+1} z_{k+1} = M⁻¹A z_k − α_k z_k − β_k z_{k−1}
        // with α_k = (z_k, A z_k) (real) and β_{k+1}² = (ẑ, v̂) where
        // v̂ = M ẑ is tracked through the v vectors.
        // ============================================================
        operator.apply_into(&z_cur, &mut w_av);
        op_counts.spmv_count += 1;
        let alpha = inner_product(&z_cur, &w_av).re();
        op_counts.dot_reductions += 1;
        apply_precond_into(precond, &w_av, &mut u);
        if has_preconditioner {
            op_counts.preconditioner_count += 1;
        }

        let alpha_t = T::from_real(alpha);
        let beta_t = T::from_real(beta);
        for i in 0..n {
            v_new[i] = w_av[i] - alpha_t * v_cur[i] - beta_t * v_prev[i];
            z_new[i] = u[i] - alpha_t * z_cur[i] - beta_t * z_prev[i];
        }
        op_counts.axpy_passes += 2;
        let betan_sq = inner_product(&z_new, &v_new).re().max(zero);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 2;
        let betan = betan_sq.sqrt();

        // Normalise the new Lanczos pair: z_{k+1} = ẑ/β_{k+1}, v_{k+1} = v̂/β_{k+1}.
        // (Skipped on Krylov exhaustion — the pair is unused after the exit.)
        if betan > gama_floor {
            let inv_betan = T::from_real(one / betan);
            for i in 0..n {
                v_new[i] *= inv_betan;
                z_new[i] *= inv_betan;
            }
            op_counts.axpy_passes += 1;
        }

        // ============================================================
        // Givens rotation update of the QR factorisation of T̲_k.
        // Column k of R: (ε_k, δ̄_k, γ_k) at rows (k−2, k−1, k).
        // ============================================================
        let eps = sn2 * beta; // ε_k = s_{k−2} β_k
        let dhat = cs2 * beta; // δ̂_k = c_{k−2} β_k
        let dbar = cs1 * dhat + sn1 * alpha; // δ̄_k
        let gbar = cs1 * alpha - sn1 * dhat; // γ̄_k
        let gama = (gbar * gbar + betan * betan).sqrt();

        if gama <= gama_floor {
            // Singular projected system with an exhausted Krylov space: no
            // representable step remains. Keep the best iterate.
            let actual =
                exact_relative_residual(operator, &x, b, &mut work, true_b_norm, &mut op_counts);
            let converged = actual <= config.tolerance;
            return MinresSolution {
                x,
                iterations: iter - 1,
                residual: actual,
                estimated_residual: actual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged,
                status: if converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Stagnated
                },
                condition_estimate: acond,
                op_counts,
            };
        }

        let cs_k = gbar / gama;
        let sn_k = betan / gama;
        let phi = cs_k * phi_tilde;
        phi_tilde = -sn_k * phi_tilde;

        // ============================================================
        // Short-recurrence solution update:
        //   w_k = (z_k − ε_k w_{k−2} − δ̄_k w_{k−1}) / γ_k
        //   x  += φ_k w_k
        // ============================================================
        let inv_gama = T::from_real(one / gama);
        let eps_t = T::from_real(eps);
        let dbar_t = T::from_real(dbar);
        let phi_t = T::from_real(phi);
        for i in 0..n {
            w_dir[i] = (z_cur[i] - eps_t * wl2[i] - dbar_t * wl[i]) * inv_gama;
            x[i] += phi_t * w_dir[i];
        }
        op_counts.axpy_passes += 2;

        // Running ‖T_k‖_F and condition estimate ‖T_k‖_F / min|R_jj|.
        anorm_sq += alpha * alpha + betan * betan + if iter > 1 { beta * beta } else { zero };
        let anorm = anorm_sq.sqrt();
        if gama < gmin {
            gmin = gama;
        }
        acond = anorm / gmin;

        // Estimated relative preconditioned residual ‖M⁻¹r_k‖_M / ‖b‖_{M⁻¹}.
        let rel_res = phi_tilde.abs() / preconditioned_b_norm;

        if config.print_interval > 0 && iter % config.print_interval == 0 {
            log::info!(
                "MINRES iteration {}: relative residual = {:.6e}",
                iter,
                num_traits::ToPrimitive::to_f64(&rel_res).unwrap_or(0.0)
            );
        }

        if rel_res <= config.tolerance {
            // Verify with the exact unpreconditioned residual to guard
            // against a pathological preconditioner (mirrors gmres.rs).
            let actual =
                exact_relative_residual(operator, &x, b, &mut work, true_b_norm, &mut op_counts);
            if actual <= config.tolerance {
                return MinresSolution {
                    x,
                    iterations: iter,
                    residual: actual,
                    estimated_residual: actual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: true,
                    status: SolverStatus::Converged,
                    condition_estimate: acond,
                    op_counts,
                };
            }
        }

        if betan <= breakdown_rel * anorm.max(one) {
            // Krylov space exhausted ("lucky breakdown"): x is the residual
            // minimiser over the full Krylov space.
            let actual =
                exact_relative_residual(operator, &x, b, &mut work, true_b_norm, &mut op_counts);
            let converged = actual <= config.tolerance;
            return MinresSolution {
                x,
                iterations: iter,
                residual: actual,
                estimated_residual: actual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged,
                status: if converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Breakdown
                },
                condition_estimate: acond,
                op_counts,
            };
        }

        if acond > config.condition_limit {
            // Projected operator too ill-conditioned to trust further steps.
            let actual =
                exact_relative_residual(operator, &x, b, &mut work, true_b_norm, &mut op_counts);
            let converged = actual <= config.tolerance;
            return MinresSolution {
                x,
                iterations: iter,
                residual: actual,
                estimated_residual: actual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged,
                status: if converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Breakdown
                },
                condition_estimate: acond,
                op_counts,
            };
        }

        // Rotate Lanczos and update-direction triples; shift rotations.
        std::mem::swap(&mut v_prev, &mut v_cur);
        std::mem::swap(&mut v_cur, &mut v_new);
        std::mem::swap(&mut z_prev, &mut z_cur);
        std::mem::swap(&mut z_cur, &mut z_new);
        std::mem::swap(&mut wl2, &mut wl);
        std::mem::swap(&mut wl, &mut w_dir);
        cs2 = cs1;
        cs1 = cs_k;
        sn2 = sn1;
        sn1 = sn_k;
        beta = betan;
    }

    let actual = exact_relative_residual(operator, &x, b, &mut work, true_b_norm, &mut op_counts);
    MinresSolution {
        x,
        iterations: config.max_iterations,
        residual: actual,
        estimated_residual: actual,
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        condition_estimate: acond,
        op_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner};
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn minres_max_iteration_reports_true_residual_and_backward_error() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, -2.0),
                (1, 2, 2.0),
                (2, 1, 2.0),
                (2, 2, 3.0),
            ],
        );
        let rhs = [1.0, -2.0, 3.0];
        let config = MinresConfig {
            max_iterations: 1,
            tolerance: 1.0e-14,
            print_interval: 0,
            condition_limit: 1.0e14,
        };
        let solution = minres(&matrix, &rhs, &config);
        let ax = matrix.matvec(&solution.x);
        let residual_norm = ax
            .iter()
            .zip(rhs)
            .map(|(applied, rhs)| (rhs - applied).powi(2))
            .sum::<f64>()
            .sqrt();
        let rhs_norm = rhs.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
        let ax_norm = ax.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();

        assert!(!solution.converged);
        assert_eq!(solution.status, SolverStatus::MaxIterationsReached);
        assert!((solution.residual - residual_norm / rhs_norm).abs() < 1.0e-14);
        assert!(solution.estimated_residual.is_finite());
        assert!(
            (solution.backward_error - residual_norm / (ax_norm + rhs_norm).max(f64::MIN_POSITIVE))
                .abs()
                < 1.0e-14
        );
        assert_eq!(solution.attainable_accuracy, AccuracyStatus::NotAttained);
    }

    /// 2D 5-point Laplacian (unit grid spacing) shifted by −σI.
    /// Eigenvalues of the unshifted operator lie in (0, 8), so 0 < σ < 8
    /// makes the system symmetric indefinite.
    fn shifted_laplacian_2d(nx: usize, sigma: f64) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..n {
            let i = row % nx;
            let j = row / nx;
            if i > 0 {
                col_indices.push(row - 1);
                values.push(-1.0);
            }
            if j > 0 {
                col_indices.push(row - nx);
                values.push(-1.0);
            }
            col_indices.push(row);
            values.push(4.0 - sigma);
            if i < nx - 1 {
                col_indices.push(row + 1);
                values.push(-1.0);
            }
            if j < nx - 1 {
                col_indices.push(row + nx);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }

        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    /// Complex Hermitian tridiagonal with diagonal `diag` and super-diagonal
    /// `off` (sub-diagonal `conj(off)`). Indefinite when `diag` lies inside
    /// the spectral range of the unshifted matrix.
    fn hermitian_tridiag_shifted(n: usize, diag: f64) -> CsrMatrix<Complex64> {
        let off = Complex64::new(0.5, 0.25);
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(off.conj());
            }
            col_indices.push(i);
            values.push(Complex64::new(diag, 0.0));
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(off);
            }
            row_ptrs.push(col_indices.len());
        }

        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    /// Diagonal symmetric indefinite matrix: entries on both sides of zero,
    /// spread so the unpreconditioned solve needs several iterations.
    fn diagonal_indefinite(n: usize) -> CsrMatrix<f64> {
        let row_ptrs: Vec<usize> = (0..=n).collect();
        let col_indices: Vec<usize> = (0..n).collect();
        let values: Vec<f64> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    -(0.5 + i as f64 / n as f64)
                } else {
                    0.5 + 2.0 * i as f64 / n as f64
                }
            })
            .collect();
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    fn true_rel_residual<T: ComplexField>(a: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real {
        let ax = a.apply(x);
        let mut res = T::Real::zero();
        let mut bn = T::Real::zero();
        for i in 0..b.len() {
            res += (b[i] - ax[i]).norm_sqr();
            bn += b[i].norm_sqr();
        }
        res.sqrt() / bn.sqrt()
    }

    #[test]
    fn minres_rejects_invalid_input_before_operator_use() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator must not be called");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator must not be called");
            }
        }

        let config = MinresConfig::default();
        let non_finite = minres(&PanicOperator, &[f64::NAN], &config);
        assert_eq!(non_finite.status, SolverStatus::NumericalError);
        assert!(!non_finite.converged);
        assert_eq!(non_finite.iterations, 0);

        let wrong_dimension = minres(&PanicOperator, &[1.0, 2.0], &config);
        assert_eq!(wrong_dimension.status, SolverStatus::NumericalError);
        assert!(!wrong_dimension.converged);
    }

    #[test]
    fn checked_minres_contains_operator_panics() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("MINRES operator callback failed");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("MINRES operator callback failed");
            }
        }

        let error = try_minres(&PanicOperator, &[1.0], &MinresConfig::default())
            .expect_err("checked MINRES must type callback panics");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "MINRES",
                ..
            }
        ));

        let legacy = minres(&PanicOperator, &[1.0], &MinresConfig::default());
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    #[test]
    fn checked_minres_rejects_non_finite_generated_values() {
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

        let error = try_minres(&NonFiniteOperator, &[1.0], &MinresConfig::default())
            .expect_err("checked MINRES must reject non-finite operator output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "MINRES operator",
                ..
            }
        ));

        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
                output[0] = f64::INFINITY;
            }
        }

        let operator = CsrMatrix::identity(1);
        let precond = NonFinitePreconditioner;
        let error =
            try_minres_preconditioned(&operator, &precond, &[1.0], &MinresConfig::default())
                .expect_err("checked MINRES must reject non-finite preconditioner output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "MINRES preconditioner",
                ..
            }
        ));

        let legacy = minres_preconditioned(&operator, &precond, &[1.0], &MinresConfig::default());
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    #[test]
    fn test_minres_symmetric_indefinite_2d() {
        // Shifted 2D Laplacian: symmetric indefinite (eigenvalues on both
        // sides of zero). CG has no convergence guarantee here; MINRES is
        // the right Krylov method and must converge.
        let nx = 16;
        let n = nx * nx;
        let a = shifted_laplacian_2d(nx, 3.0);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() + 0.5).collect();

        let config = MinresConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);

        assert!(
            result.converged,
            "MINRES should converge on symmetric-indefinite system: {} iters, res {:.2e}",
            result.iterations, result.residual
        );
        assert!(result.residual < 1e-10);
        assert!(true_rel_residual(&a, &result.x, &b) < 1e-10);
    }

    #[test]
    fn test_minres_symmetric_indefinite_3d() {
        // Shifted 3D 7-point Laplacian: eigenvalues of the unshifted
        // operator lie in (0, 12), so σ = 5.3 makes it symmetric indefinite.
        // (σ must avoid the exact eigenvalues 6 − 2Σcos(kπ/9) — a singular
        // inconsistent system makes MINRES diverge by design.)
        let nx = 8;
        let n = nx * nx * nx;
        let sigma = 5.3;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            let i = row % nx;
            let j = (row / nx) % nx;
            let k = row / (nx * nx);
            if i > 0 {
                col_indices.push(row - 1);
                values.push(-1.0);
            }
            if j > 0 {
                col_indices.push(row - nx);
                values.push(-1.0);
            }
            if k > 0 {
                col_indices.push(row - nx * nx);
                values.push(-1.0);
            }
            col_indices.push(row);
            values.push(6.0 - sigma);
            if i < nx - 1 {
                col_indices.push(row + 1);
                values.push(-1.0);
            }
            if j < nx - 1 {
                col_indices.push(row + nx);
                values.push(-1.0);
            }
            if k < nx - 1 {
                col_indices.push(row + nx * nx);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.19).cos() + 0.4).collect();

        let config = MinresConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);

        assert!(
            result.converged,
            "MINRES should converge on the 3D symmetric-indefinite system: {} iters, res {:.2e}",
            result.iterations, result.residual
        );
        assert!(true_rel_residual(&a, &result.x, &b) < 1e-10);
    }

    #[test]
    fn test_minres_converges_where_cg_breaks_down() {
        // Diagonal indefinite matrix with exactly ±-paired eigenvalues and
        // b = ones: (p, Ap) = Σ a_i = 0 exactly at the first CG iteration,
        // so CG breaks down immediately while MINRES solves the system.
        let pairs: Vec<f64> = vec![2.0, 1.5, 1.1, 0.7, 0.4, 2.6, 1.9, 0.9, 0.55, 0.31];
        let n = 2 * pairs.len();
        let mut vals = Vec::with_capacity(n);
        for &p in &pairs {
            vals.push(-p);
            vals.push(p);
        }
        let a = CsrMatrix::from_diagonal(&vals);
        let b = vec![1.0_f64; n];

        let cg_config = crate::CgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let cg_result = crate::cg(&a, &b, &cg_config);
        assert!(
            !cg_result.converged,
            "CG should break down on the sign-symmetric indefinite system"
        );

        let config = MinresConfig {
            max_iterations: 500,
            tolerance: 1e-12,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);
        assert!(
            result.converged,
            "MINRES should converge where CG breaks down: {} iters, res {:.2e}",
            result.iterations, result.residual
        );
        for (i, &xi) in result.x.iter().enumerate() {
            assert!(
                (xi - 1.0 / vals[i]).abs() < 1e-10,
                "x[{i}] = {xi} differs from exact {}",
                1.0 / vals[i]
            );
        }
    }

    #[test]
    fn test_minres_matches_gmres_fewer_mvs() {
        // Same symmetric-indefinite system solved by MINRES and GMRES.
        //
        // - Unrestarted GMRES is the same Krylov minimiser as MINRES in
        //   exact arithmetic: same solution, comparable MV count (MINRES
        //   lags slightly in finite precision — Lanczos loses orthogonality
        //   where Arnoldi reorthogonalises — but needs O(n) storage instead
        //   of the full basis).
        // - Restarted GMRES(50), the practical default, pays for its basis
        //   truncation: it cannot converge here within 1000 MVs, while
        //   MINRES converges in ~200.
        let nx = 16;
        let n = nx * nx;
        let a = shifted_laplacian_2d(nx, 3.0);
        let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 13) as f64 - 6.0).collect();

        let minres_config = MinresConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let minres_result = minres(&a, &b, &minres_config);

        let gmres_full_config = crate::GmresConfig {
            max_iterations: 5,
            restart: 300,
            tolerance: 1e-10,
            ..crate::GmresConfig::default()
        };
        let gmres_full = crate::gmres(&a, &b, &gmres_full_config);

        let gmres50_config = crate::GmresConfig {
            max_iterations: 20,
            restart: 50,
            tolerance: 1e-10,
            ..crate::GmresConfig::default()
        };
        let gmres50 = crate::gmres(&a, &b, &gmres50_config);

        assert!(minres_result.converged);
        assert!(
            gmres_full.converged,
            "unrestarted GMRES should converge: {} MVs, res {:.2e}",
            gmres_full.iterations, gmres_full.residual
        );

        // Solutions agree.
        let diff: f64 = minres_result
            .x
            .iter()
            .zip(gmres_full.x.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>();
        assert!(
            diff < 1e-6,
            "MINRES and GMRES solutions differ by {diff:.2e}"
        );

        // MINRES (one MV per iteration) vs unrestarted GMRES: comparable MVs.
        assert!(
            minres_result.iterations <= gmres_full.iterations * 3 / 2 + 10,
            "MINRES ({} MVs) should be within ~1.5x of unrestarted GMRES ({} MVs)",
            minres_result.iterations,
            gmres_full.iterations
        );
        // ...and strictly fewer MVs than the restarted default.
        assert!(
            minres_result.iterations <= gmres50.iterations,
            "MINRES ({} MVs) should beat restarted GMRES(50) ({} MVs, converged={})",
            minres_result.iterations,
            gmres50.iterations,
            gmres50.converged
        );
    }

    #[test]
    fn test_minres_preconditioned_diagonal() {
        // Diagonal indefinite system. Note plain Jacobi is *indefinite* here
        // and is correctly rejected by the HPD guard; the valid spectral
        // preconditioner M = diag(|a_i|) makes M⁻¹A = diag(±1), so
        // preconditioned MINRES exhausts the Krylov space in at most two
        // steps while the unpreconditioned solve needs several iterations on
        // the spread spectrum.
        let n = 100;
        let a = diagonal_indefinite(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.61).cos() + 0.25).collect();

        let config = MinresConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let plain = minres(&a, &b, &config);

        let abs_diag: Vec<f64> = a.diagonal().iter().map(|&d| d.abs()).collect();
        let precond =
            sonium_solvers_preconditioners::DiagonalPreconditioner::from_diagonal(&abs_diag);
        let preconditioned = minres_preconditioned(&a, &precond, &b, &config);

        assert!(plain.converged);
        assert!(
            preconditioned.converged,
            "preconditioned MINRES should converge: {} iters, res {:.2e}",
            preconditioned.iterations, preconditioned.residual
        );
        assert!(
            preconditioned.iterations <= 2,
            "two-eigenvalue preconditioned spectrum should finish in <= 2 iterations, got {}",
            preconditioned.iterations
        );
        assert!(preconditioned.iterations < plain.iterations);
        assert!(preconditioned.op_counts.preconditioner_count > 0);
        assert!(true_rel_residual(&a, &preconditioned.x, &b) < 1e-10);
    }

    #[test]
    fn test_minres_preconditioned_shifted_laplacian() {
        // Non-trivial SPD preconditioner on an indefinite 2D system: the
        // varying diagonal 4 − σ + d_i ∈ [1, 2] keeps Jacobi HPD while the
        // operator itself stays indefinite (smallest eigenvalue ≈ −2.93).
        let nx = 16;
        let n = nx * nx;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            let i = row % nx;
            let j = row / nx;
            if i > 0 {
                col_indices.push(row - 1);
                values.push(-1.0);
            }
            if j > 0 {
                col_indices.push(row - nx);
                values.push(-1.0);
            }
            col_indices.push(row);
            values.push(1.0 + (row % 3) as f64 * 0.5);
            if i < nx - 1 {
                col_indices.push(row + 1);
                values.push(-1.0);
            }
            if j < nx - 1 {
                col_indices.push(row + nx);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);

        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.23).sin() + 0.1).collect();
        let config = MinresConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };

        let plain = minres(&a, &b, &config);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);
        let preconditioned = minres_preconditioned(&a, &precond, &b, &config);

        assert!(plain.converged);
        assert!(
            preconditioned.converged,
            "preconditioned MINRES should converge: {} iters, res {:.2e}",
            preconditioned.iterations, preconditioned.residual
        );
        assert!(true_rel_residual(&a, &preconditioned.x, &b) < 1e-10);
        assert!(preconditioned.op_counts.preconditioner_count > 0);
    }

    #[test]
    fn test_minres_complex_hermitian_indefinite() {
        // Complex Hermitian indefinite: diag 0.5 with |off| ≈ 0.559 puts the
        // spectrum of the tridiagonal inside [0.5 − 1.12, 0.5 + 1.12].
        let n = 60;
        let a = hermitian_tridiag_shifted(n, 0.5);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.41).sin() + 0.3, (i as f64 * 0.17).cos()))
            .collect();

        let config = MinresConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);

        assert!(
            result.converged,
            "MINRES should converge on complex Hermitian indefinite: {} iters, res {:.2e}",
            result.iterations, result.residual
        );
        assert!(true_rel_residual(&a, &result.x, &b) < 1e-10);

        // Agreement with GMRES on the same system.
        let gmres_config = crate::GmresConfig {
            max_iterations: 10,
            restart: 60,
            tolerance: 1e-10,
            ..crate::GmresConfig::default()
        };
        let gmres_result = crate::gmres(&a, &b, &gmres_config);
        assert!(gmres_result.converged);
        let diff: f64 = result
            .x
            .iter()
            .zip(gmres_result.x.iter())
            .map(|(&a, &b)| (a - b).norm())
            .sum::<f64>();
        assert!(
            diff < 1e-6,
            "complex MINRES and GMRES solutions differ by {diff:.2e}"
        );
    }

    #[test]
    fn test_minres_breakdown_eigenvector_rhs() {
        // b is a single eigenvector: the Krylov space is exhausted after one
        // iteration (β_2 = 0 exactly) — the lucky-breakdown path must return
        // the exact solution as converged.
        let n = 20;
        let a = diagonal_indefinite(n);
        let mut b = vec![0.0_f64; n];
        b[7] = 2.5;

        let config = MinresConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);

        assert!(
            result.converged,
            "lucky breakdown should report converged: status {:?}",
            result.status
        );
        assert!(result.iterations <= 1, "should finish in one iteration");
        // Exact solution: x_7 = b_7 / a_77, all other components zero.
        let diag_val = 0.5 + 2.0 * 7.0 / n as f64; // i = 7 is odd
        let expected = 2.5 / diag_val;
        assert!((result.x[7] - expected).abs() < 1e-10);
        for (i, &xi) in result.x.iter().enumerate() {
            if i != 7 {
                assert!(xi.abs() < 1e-12, "x[{i}] should be zero, got {xi}");
            }
        }
    }

    #[test]
    fn test_minres_zero_rhs() {
        let n = 10;
        let a = diagonal_indefinite(n);
        let b = vec![0.0_f64; n];

        let config = MinresConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert!(result.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_minres_singular_consistent() {
        // 1D pure-Neumann Laplacian: symmetric positive semidefinite with a
        // one-dimensional null space (the constants). b = A x* for a ramp x*
        // is consistent (b ⊥ null(A)); MINRES must find *a* solution.
        let n = 50;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(if i == 0 || i == n - 1 { 1.0 } else { 2.0 });
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);

        let x_star: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b = a.apply(&x_star); // consistent by construction

        let config = MinresConfig {
            max_iterations: 500,
            tolerance: 1e-8,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);

        assert!(
            result.converged,
            "MINRES should solve the consistent singular system: {} iters, status {:?}, res {:.2e}",
            result.iterations, result.status, result.residual
        );
        assert!(true_rel_residual(&a, &result.x, &b) < 1e-8);
    }

    #[test]
    fn test_minres_with_guess() {
        let nx = 12;
        let n = nx * nx;
        let a = shifted_laplacian_2d(nx, 3.0);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.53).sin()).collect();

        let config = MinresConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let plain = minres(&a, &b, &config);
        assert!(plain.converged);

        // Starting from the converged solution should terminate immediately.
        let seeded = minres_with_guess(&a, &b, Some(&plain.x), &config);
        assert!(seeded.converged);
        assert!(
            seeded.iterations <= 1,
            "seeded with the solution should finish at once, got {} iterations",
            seeded.iterations
        );
    }

    #[test]
    fn test_minres_reports_krylov_op_counts() {
        let n = 50;
        let a = diagonal_indefinite(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 0.5).collect();

        let config = MinresConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
            condition_limit: 1e14,
        };
        let result = minres(&a, &b, &config);

        assert!(result.converged);
        assert!(result.op_counts.spmv_count >= result.iterations);
        assert!(result.op_counts.dot_reductions >= 2 * result.iterations);
        assert!(result.op_counts.norm_count > 0);
        assert_eq!(result.op_counts.preconditioner_count, 0);
        // Every exit path recomputes the exact residual once.
        assert!(result.op_counts.exact_residual_checks >= 1);
        assert!(result.condition_estimate > 0.0);
    }
}
