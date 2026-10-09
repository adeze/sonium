//! FGMRES (Flexible Generalized Minimal Residual) solver
//!
//! Implementation of the Flexible GMRES algorithm based on Saad (1993).
//!
//! FGMRES extends standard GMRES to support **variable preconditioning**,
//! where the preconditioner can change at each iteration. This is essential
//! for inner-outer iteration schemes and adaptive preconditioners like AMG
//! with parallel Jacobi smoothing.
//!
//! Key differences from GMRES:
//! - Stores an additional basis Z = [z_1, ..., z_m] where z_j = M_j^{-1} v_j
//! - The Arnoldi process computes w = A * z_j (right preconditioning)
//! - The solution update uses Z: x = x_0 + Z_m * y_m

use super::cgs_orth::{
    FlatBasis, KrylovOpCounts, batched_dots_flat_into, batched_project_flat,
    fused_basis_update_flat,
};
use super::gmres_common::{
    exact_relative_residual, exact_residual_check_due, givens_rotation, solve_upper_triangular,
};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::vector_norm;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

/// FGMRES solver configuration
#[derive(Debug, Clone)]
pub struct FgmresConfig<R> {
    /// Maximum number of outer iterations (restarts)
    pub max_iterations: usize,
    /// Restart parameter (number of inner iterations before restart)
    pub restart: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
    /// Check estimated convergence every N inner iterations.
    pub convergence_check_interval: usize,
    /// Check the candidate iterate against the original operator every N Krylov
    /// iterations, including before the projected estimate claims convergence.
    ///
    /// The exact residual check costs an additional operator application. A
    /// value of 0 preserves the cheap estimated-residual path except for
    /// mandatory breakdown and final checks.
    pub exact_residual_check_interval: usize,
}

impl<R: Float> FgmresConfig<R> {
    /// Validate configuration before entering the Krylov loop.
    pub fn validate(&self) -> Result<(), SolverError> {
        if self.max_iterations == 0 {
            return Err(SolverError::OperationFailed {
                operation: "FGMRES configuration",
                reason: "max_iterations must be greater than zero".to_string(),
            });
        }
        if self.restart == 0 {
            return Err(SolverError::OperationFailed {
                operation: "FGMRES configuration",
                reason: "restart must be greater than zero".to_string(),
            });
        }
        if !self.tolerance.is_finite() || self.tolerance <= R::zero() {
            return Err(SolverError::OperationFailed {
                operation: "FGMRES configuration",
                reason: "tolerance must be finite and positive".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for FgmresConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-6,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        }
    }
}

impl Default for FgmresConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-5,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        }
    }
}

/// Adapter used only by the checked FGMRES entry point.
///
/// The Krylov kernel intentionally uses the infallible `apply_into` trait
/// method for the hot path. This adapter keeps that kernel unchanged while
/// turning an invalid preconditioner vector into a typed failure in checked
/// calls before it can enter the Krylov basis.
struct FiniteCheckedPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: AtomicBool,
}

impl<T, P> Preconditioner<T> for FiniteCheckedPreconditioner<'_, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply_into(&self, r: &[T], y: &mut [T]) {
        self.inner.apply_into(r, y);
        if y.len() != r.len() || y.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            y.fill(T::zero());
        }
    }
}

/// Adapter used only by the checked FGMRES entry point.
///
/// Operator callbacks are infallible for compatibility, but a generated
/// non-finite vector must not be fed into Arnoldi.  Replace it with a finite
/// zero vector after retaining a sticky diagnostic so the checked boundary
/// can stop before another Krylov operator application.
struct FiniteCheckedOperator<'a, A> {
    inner: &'a A,
    invalid_output: AtomicBool,
}

impl<T, A> LinearOperator<T> for FiniteCheckedOperator<'_, A>
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

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        if x.len() != self.num_cols()
            || y.len() != self.num_rows()
            || x.iter().any(|value| !value.norm().is_finite())
        {
            self.invalid_output.store(true, Ordering::Relaxed);
            y.fill(T::zero());
            return;
        }
        self.inner.apply_into(x, y);
        if y.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            y.fill(T::zero());
        }
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        if x.len() != self.num_rows()
            || y.len() != self.num_cols()
            || x.iter().any(|value| !value.norm().is_finite())
        {
            self.invalid_output.store(true, Ordering::Relaxed);
            y.fill(T::zero());
            return;
        }
        self.inner.apply_transpose_into(x, y);
        if y.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            y.fill(T::zero());
        }
    }
}

/// FGMRES solver result
#[derive(Debug)]
pub struct FgmresSolution<T: ComplexField> {
    /// Solution vector
    pub x: Vec<T>,
    /// Total number of matrix-vector products
    pub iterations: usize,
    /// Number of restarts performed
    pub restarts: usize,
    /// Final relative residual
    pub residual: T::Real,
    /// Cheap projected/Hessenberg residual captured before the final true
    /// residual recomputation. This remains available for performance
    /// diagnostics even when the public residual is promoted to high precision.
    pub estimated_residual: T::Real,
    /// Normwise backward-error indicator computed from the returned operator
    /// action: `||b - A*x|| / max(||A*x|| + ||b||, min-positive)`.
    pub backward_error: T::Real,
    /// Whether the requested tolerance was attained by the returned iterate.
    pub attainable_accuracy: AccuracyStatus,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Solver status
    pub status: SolverStatus,
    /// Memory-traffic oriented operation counts for the Krylov solve.
    pub op_counts: KrylovOpCounts,
}

/// Solve Ax = b using FGMRES (no preconditioner — equivalent to standard GMRES)
pub fn fgmres<T, A>(operator: &A, b: &[T], config: &FgmresConfig<T::Real>) -> FgmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    fgmres_preconditioned(operator, &identity, b, None, config)
}

/// Checked FGMRES entry point using the identity preconditioner.
pub fn try_fgmres<T, A>(
    operator: &A,
    b: &[T],
    config: &FgmresConfig<T::Real>,
) -> Result<FgmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    try_fgmres_preconditioned(operator, &identity, b, None, config)
}

/// Checked FGMRES entry point for callers that accept user-controlled solver
/// configuration or matrix dimensions.
pub fn try_fgmres_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &FgmresConfig<T::Real>,
) -> Result<FgmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_fgmres_preconditioned_with_abort(operator, precond, b, x0, config, || false)
}

/// Checked FGMRES with a caller-owned preconditioner failure signal.
///
/// The `should_abort` callback is sampled immediately after each
/// preconditioner application.  This is needed for adapters whose trait
/// surface is intentionally infallible but which retain a sticky failure
/// (for example a mixed-precision cast or factor application).  The checked
/// solve stops before applying the operator to a failed correction; callers
/// still receive a typed error and can preserve a more specific diagnostic
/// from their adapter.
pub fn try_fgmres_preconditioned_with_abort<T, A, P, F>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &FgmresConfig<T::Real>,
    should_abort: F,
) -> Result<FgmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
    F: Fn() -> bool,
{
    config.validate()?;
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if b.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "FGMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(x0) = x0
        && x0.len() != operator.num_cols()
    {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: x0.len(),
        });
    }
    if let Some(x0) = x0
        && x0.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "FGMRES",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }
    let checked_operator = FiniteCheckedOperator {
        inner: operator,
        invalid_output: AtomicBool::new(false),
    };
    let checked_precond = FiniteCheckedPreconditioner {
        inner: precond,
        invalid_output: AtomicBool::new(false),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let abort = || {
            checked_operator.invalid_output.load(Ordering::Relaxed)
                || checked_precond.invalid_output.load(Ordering::Relaxed)
                || should_abort()
        };
        let mut result = fgmres_preconditioned_unchecked(
            &checked_operator,
            &checked_precond,
            b,
            x0,
            config,
            Some(&abort),
        );
        if checked_operator.invalid_output.load(Ordering::Relaxed) {
            return Err(SolverError::OperationFailed {
                operation: "FGMRES operator",
                reason: "operator generated NaN, infinity, or an invalid output".to_string(),
            });
        }
        if checked_precond.invalid_output.load(Ordering::Relaxed) {
            return Err(SolverError::OperationFailed {
                operation: "FGMRES preconditioner",
                reason: "preconditioner generated NaN, infinity, or an invalid output".to_string(),
            });
        }
        if should_abort() {
            return Err(SolverError::OperationFailed {
                operation: "FGMRES preconditioner",
                reason: "preconditioner requested early termination".to_string(),
            });
        }
        finalize_checked_residual(operator, b, config, &mut result)?;
        SolverResultContract::validate_contract(&result)?;
        Ok::<_, SolverError>(result)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "FGMRES",
        reason: panic_reason(payload),
    })??;
    Ok(result)
}

/// Recompute the residual for the checked public API.
///
/// The inner FGMRES loop may stop on a Hessenberg estimate for performance.
/// Checked callers must receive a metric from the original operator and must
/// not observe a successful result when the returned iterate is non-finite or
/// misses the requested tolerance.
fn finalize_checked_residual<T, A>(
    operator: &A,
    b: &[T],
    config: &FgmresConfig<T::Real>,
    result: &mut FgmresSolution<T>,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if result.x.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: result.x.len(),
        });
    }
    if result.x.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "FGMRES",
            reason: "generated solution contains NaN or infinity".to_string(),
        });
    }

    let mut ax = vec![T::zero(); operator.num_rows()];
    operator.apply_into(&result.x, &mut ax);
    result.op_counts.spmv_count += 1;
    if ax.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "FGMRES",
            reason: "operator generated NaN or infinity".to_string(),
        });
    }

    let b_norm = vector_norm(b);
    let mut residual = vec![T::zero(); b.len()];
    for ((out, &rhs), &applied) in residual.iter_mut().zip(b).zip(&ax) {
        *out = rhs - applied;
    }
    let residual_norm = vector_norm(&residual);
    result.op_counts.norm_count += 1;
    // Relative residuals are undefined for an exactly zero RHS; use the
    // absolute residual in that one case.  Non-zero tiny RHS values retain
    // their physical scale instead of being promoted to a unit denominator.
    let true_residual = if b_norm == T::Real::zero() {
        residual_norm
    } else {
        residual_norm / b_norm
    };
    if !true_residual.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "FGMRES",
            reason: "true residual is NaN or infinity".to_string(),
        });
    }

    let ax_norm = vector_norm(&ax);
    let rhs_norm = vector_norm(b);
    result.op_counts.norm_count += 2;
    result.residual = true_residual;
    result.backward_error = T::normwise_backward_error(residual_norm, ax_norm, rhs_norm);
    validate_checked_metrics(result)?;
    result.attainable_accuracy = if true_residual <= config.tolerance {
        AccuracyStatus::Attained
    } else {
        AccuracyStatus::NotAttained
    };
    if true_residual <= config.tolerance {
        result.converged = true;
        result.status = SolverStatus::Converged;
    } else if result.status == SolverStatus::Converged {
        result.converged = false;
        result.status = SolverStatus::MaxIterationsReached;
    }
    Ok(())
}

fn validate_checked_metrics<T: ComplexField>(
    result: &FgmresSolution<T>,
) -> Result<(), SolverError> {
    if !result.estimated_residual.is_finite() || !result.backward_error.is_finite() {
        return Err(SolverError::OperationFailed {
            operation: "FGMRES",
            reason: "generated residual metrics contain NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "FGMRES callback panicked".to_string()
    }
}

/// Solve Ax = b using FGMRES with right preconditioning
///
/// Unlike standard left-preconditioned GMRES, FGMRES applies the
/// preconditioner on the right: it solves A M^{-1} (M x) = b in the
/// Krylov subspace of A, storing z_j = M^{-1} v_j separately.
///
/// This allows the preconditioner to be **variable** (different at each
/// iteration), which is critical for AMG, inner-outer schemes, etc.
pub fn fgmres_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &FgmresConfig<T::Real>,
) -> FgmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let mut result = fgmres_preconditioned_unchecked(operator, precond, b, x0, config, None);
    let finalized = catch_unwind(AssertUnwindSafe(|| {
        finalize_checked_residual(operator, b, config, &mut result)
    }));
    if !matches!(finalized, Ok(Ok(()))) {
        result.residual = T::Real::infinity();
        result.estimated_residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotAttained;
        result.converged = false;
        result.status = SolverStatus::NumericalError;
    }
    result
}

fn fgmres_preconditioned_unchecked<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &FgmresConfig<T::Real>,
    abort_if: Option<&dyn Fn() -> bool>,
) -> FgmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.len();
    let invalid_input = config.max_iterations == 0
        || config.restart == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
        || b.iter().any(|value| !value.norm().is_finite())
        || x0.is_some_and(|guess| {
            guess.len() != n || guess.iter().any(|value| !value.norm().is_finite())
        });
    if invalid_input {
        let x = match x0 {
            Some(guess) if guess.len() == n => guess.to_vec(),
            _ => vec![T::zero(); n],
        };
        return FgmresSolution {
            x,
            iterations: 0,
            restarts: 0,
            residual: T::Real::infinity(),
            estimated_residual: T::Real::infinity(),
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::NumericalError,
            op_counts: KrylovOpCounts::default(),
        };
    }
    let m = config.restart;

    // Initialize solution
    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut op_counts = KrylovOpCounts::default();

    // Compute RHS norm for relative tolerance
    let mut b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        // Guard against a near-zero RHS with a non-zero initial guess:
        // only claim immediate convergence if the initial residual is also tiny.
        let mut w = vec![T::zero(); n];
        operator.apply_into(&x, &mut w);
        op_counts.spmv_count += 1;
        for k in 0..n {
            w[k] = b[k] - w[k];
        }
        let beta = vector_norm(&w);
        op_counts.norm_count += 1;
        if beta < tol_threshold {
            return FgmresSolution {
                x,
                iterations: 0,
                restarts: 0,
                residual: T::Real::zero(),
                estimated_residual: T::Real::zero(),
                backward_error: T::Real::zero(),
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }
        // Relative residual is undefined for a zero RHS. Normalizing by one
        // turns the remaining convergence checks into absolute-residual checks.
        b_norm = T::Real::one();
    }

    let mut total_iterations = 0;
    let mut restarts = 0;
    // Keep the last Hessenberg/Givens estimate separate from the true
    // residual computed when a cycle or the whole solve ends.  Mixed-
    // precision callers rely on this distinction to diagnose how closely
    // the low-precision preconditioner tracked the original operator.
    let mut last_estimated_residual = T::Real::infinity();

    // Pre-allocate Krylov bases and work buffer (reused across restarts)
    let mut v = FlatBasis::new(m + 1, n);
    let mut z = FlatBasis::new(m, n);
    let mut w = vec![T::zero(); n];
    let mut coeffs = Vec::with_capacity(m);
    let mut reorth_coeffs = Vec::with_capacity(m);

    // Hessenberg matrix, Givens coefficients and least-squares RHS, reused
    // across restart cycles (cleared per cycle below) instead of reallocated.
    let mut h: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, T::zero());
    let mut cs: Vec<T> = Vec::with_capacity(m);
    let mut sn: Vec<T> = Vec::with_capacity(m);
    let mut g: Vec<T> = vec![T::zero(); m + 1];

    let dgks_threshold = T::Real::from_f64(0.707).unwrap();
    let breakdown_tol = T::Real::from_f64(1e-20).unwrap();
    let abs_tol = T::Real::from_f64(1e-20).unwrap();

    // Outer iteration (restarts)
    'outer: for _outer in 0..config.max_iterations {
        // Compute residual r = b - Ax (zero allocation)
        operator.apply_into(&x, &mut w);
        op_counts.spmv_count += 1;
        if abort_if.is_some_and(|check| check()) {
            return numerical_error_solution(
                x,
                total_iterations,
                restarts,
                last_estimated_residual,
                op_counts,
            );
        }
        for k in 0..n {
            w[k] = b[k] - w[k];
        }
        let beta = vector_norm(&w);
        op_counts.norm_count += 1;

        // Check convergence
        let rel_residual = beta / b_norm;
        if rel_residual < config.tolerance {
            return FgmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: rel_residual,
                estimated_residual: rel_residual,
                backward_error: T::Real::zero(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // Initialize v[0] = r / beta
        let inv_beta = T::from_real(T::Real::one() / beta);
        {
            let v0 = v.vector_mut(0);
            for k in 0..n {
                v0[k] = w[k] * inv_beta;
            }
        }

        // Reset the reused Hessenberg state for this restart cycle.
        h.fill(T::zero());
        cs.clear();
        sn.clear();
        g.fill(T::zero());
        g[0] = T::from_real(beta);

        let mut inner_converged = false;

        // Inner iteration (Arnoldi with right preconditioning)
        for j in 0..m {
            total_iterations += 1;

            // z_j = M^{-1} v_j  (right preconditioning — the key FGMRES step)
            let zj = z.vector_mut(j);
            precond.apply_into(v.vector(j), zj);
            op_counts.preconditioner_count += 1;

            if abort_if.is_some_and(|check| check()) {
                return numerical_error_solution(
                    x,
                    total_iterations,
                    restarts,
                    last_estimated_residual,
                    op_counts,
                );
            }

            // w = A * z_j (zero allocation via apply_into)
            operator.apply_into(z.vector(j), &mut w);
            op_counts.spmv_count += 1;
            if abort_if.is_some_and(|check| check()) {
                return numerical_error_solution(
                    x,
                    total_iterations,
                    restarts,
                    last_estimated_residual,
                    op_counts,
                );
            }

            // Classical Gram-Schmidt with DGKS reorthogonalization, batched.
            //
            // This mirrors the GMRES Arnoldi path: compute all projection
            // coefficients together, then fuse the projection into one pass
            // over rows of `w` instead of issuing `j + 1` serial axpys.
            let w_norm_before = vector_norm(&w);
            op_counts.norm_count += 1;
            batched_dots_flat_into(&v, j + 1, &w, &mut coeffs);
            for (i, &c) in coeffs.iter().enumerate() {
                h[[i, j]] = c;
            }
            batched_project_flat(&v, &coeffs, &mut w);
            op_counts.add_basis_projection::<T>(j + 1, n);

            let mut w_norm = vector_norm(&w);
            op_counts.norm_count += 1;
            if w_norm < dgks_threshold * w_norm_before {
                batched_dots_flat_into(&v, j + 1, &w, &mut reorth_coeffs);
                for (i, &si) in reorth_coeffs.iter().enumerate() {
                    h[[i, j]] += si;
                }
                batched_project_flat(&v, &reorth_coeffs, &mut w);
                op_counts.add_basis_projection::<T>(j + 1, n);
                w_norm = vector_norm(&w);
                op_counts.norm_count += 1;
            }
            h[[j + 1, j]] = T::from_real(w_norm);

            // Check for breakdown
            if w_norm < breakdown_tol {
                inner_converged = true;
            } else {
                let inv_norm = T::from_real(T::Real::one() / w_norm);
                let next_v = v.vector_mut(j + 1);
                for k in 0..n {
                    next_v[k] = w[k] * inv_norm;
                }
            }

            // Apply previous Givens rotations to new column of H
            for i in 0..j {
                let temp = cs[i].conj() * h[[i, j]] + sn[i].conj() * h[[i + 1, j]];
                h[[i + 1, j]] = T::zero() - sn[i] * h[[i, j]] + cs[i] * h[[i + 1, j]];
                h[[i, j]] = temp;
            }

            // Compute new Givens rotation
            let (c, s) = givens_rotation(h[[j, j]], h[[j + 1, j]]);
            cs.push(c);
            sn.push(s);

            // Apply Givens rotation to H and g
            h[[j, j]] = c.conj() * h[[j, j]] + s.conj() * h[[j + 1, j]];
            h[[j + 1, j]] = T::zero();

            let temp = c.conj() * g[j] + s.conj() * g[j + 1];
            g[j + 1] = T::zero() - s * g[j] + c * g[j + 1];
            g[j] = temp;

            // Check convergence
            let abs_residual = g[j + 1].norm();
            let rel_residual = abs_residual / b_norm;
            last_estimated_residual = rel_residual;

            if config.print_interval > 0 && total_iterations % config.print_interval == 0 {
                log::info!(
                    "FGMRES iteration {} (restart {}): relative residual = {:.6e}",
                    total_iterations,
                    restarts,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }

            let should_check =
                j % config.convergence_check_interval.max(1) == 0 || j == m - 1 || inner_converged;
            // On breakdown we must verify the actual residual; the
            // preconditioned estimate can be fooled by a pathological
            // preconditioner.
            let estimated_converged =
                !inner_converged && (rel_residual < config.tolerance || abs_residual < abs_tol);

            // An optional exact check is useful for mixed-precision and
            // variable-preconditioner solves where the projected residual can
            // lose contact with the original operator. Form the candidate
            // without changing `x`, so a rejected check can continue the
            // current Krylov cycle. If the estimate also claims convergence,
            // this same candidate is the mandatory exact convergence gate.
            let exact_check_due = !inner_converged
                && exact_residual_check_due(config.exact_residual_check_interval, total_iterations);
            if exact_check_due || (should_check && estimated_converged) {
                let y = solve_upper_triangular(&h, &g, j + 1);

                let mut candidate = x.clone();
                // Candidate update: x + Z * y (uses Z, NOT V — the FGMRES
                // difference). Keep the current restart base untouched until
                // the exact residual accepts this candidate.
                fused_basis_update_flat(&z, &y, &mut candidate);
                op_counts.add_basis_update::<T>(y.len(), n);

                let actual_residual = exact_relative_residual(
                    operator,
                    &candidate,
                    b,
                    &mut w,
                    b_norm,
                    &mut op_counts,
                );

                if actual_residual <= config.tolerance || actual_residual <= abs_tol {
                    return FgmresSolution {
                        x: candidate,
                        iterations: total_iterations,
                        restarts,
                        residual: actual_residual,
                        estimated_residual: rel_residual,
                        backward_error: T::Real::zero(),
                        attainable_accuracy: AccuracyStatus::NotChecked,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    };
                }

                if should_check && estimated_converged {
                    // The projected estimate was optimistic. Preserve the
                    // candidate as the new restart base and continue from it;
                    // this is the same mandatory exact-gate behavior as the
                    // previous FGMRES path, but without recomputing the
                    // candidate or its residual.
                    x = candidate;
                    restarts += 1;
                    continue 'outer;
                }
            }

            // Breakdown with unconverged residual — terminate early.
            if inner_converged {
                let y = solve_upper_triangular(&h, &g, j + 1);
                fused_basis_update_flat(&z, &y, &mut x);
                op_counts.add_basis_update::<T>(y.len(), n);

                // Compute the *actual* residual to guard against
                // pathological preconditioners.
                operator.apply_into(&x, &mut w);
                op_counts.spmv_count += 1;
                let mut actual = T::Real::zero();
                for k in 0..n {
                    let rk = b[k] - w[k];
                    actual += rk.norm_sqr();
                }
                actual = actual.sqrt() / b_norm;

                if actual <= config.tolerance {
                    return FgmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: actual,
                        estimated_residual: rel_residual,
                        backward_error: T::Real::zero(),
                        attainable_accuracy: AccuracyStatus::NotChecked,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    };
                }

                return FgmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: actual,
                    estimated_residual: rel_residual,
                    backward_error: T::Real::zero(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: false,
                    status: SolverStatus::Breakdown,
                    op_counts,
                };
            }
        }

        // Maximum inner iterations reached, compute solution and restart
        let y = solve_upper_triangular(&h, &g, m);

        // x = x + Z * y  (uses Z basis)
        fused_basis_update_flat(&z, &y, &mut x);
        op_counts.add_basis_update::<T>(y.len(), n);

        restarts += 1;
    }

    // Compute final residual (zero allocation)
    operator.apply_into(&x, &mut w);
    op_counts.spmv_count += 1;
    for k in 0..n {
        w[k] = b[k] - w[k];
    }
    let rel_residual = vector_norm(&w) / b_norm;
    op_counts.norm_count += 1;

    FgmresSolution {
        x,
        iterations: total_iterations,
        restarts,
        residual: rel_residual,
        estimated_residual: last_estimated_residual,
        backward_error: T::Real::zero(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

fn numerical_error_solution<T: ComplexField>(
    x: Vec<T>,
    iterations: usize,
    restarts: usize,
    estimated_residual: T::Real,
    op_counts: KrylovOpCounts,
) -> FgmresSolution<T> {
    FgmresSolution {
        x,
        iterations,
        restarts,
        residual: T::Real::infinity(),
        estimated_residual,
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use sonium_solvers_preconditioners::DiagonalPreconditioner;
    use sonium_solvers_sparse::CsrMatrix;

    struct NonFinitePreconditioner;

    impl Preconditioner<f64> for NonFinitePreconditioner {
        fn apply(&self, r: &[f64]) -> Vec<f64> {
            vec![f64::NAN; r.len()]
        }

        fn apply_into(&self, r: &[f64], y: &mut [f64]) {
            assert_eq!(r.len(), y.len());
            y.fill(f64::NAN);
        }
    }

    #[test]
    fn checked_fgmres_rejects_invalid_configuration_before_iteration() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let preconditioner = sonium_solvers_core::traits::IdentityPreconditioner;
        let config = FgmresConfig {
            max_iterations: 8,
            restart: 0,
            tolerance: 1.0e-8,
            ..FgmresConfig::default()
        };
        let error = try_fgmres_preconditioned(&matrix, &preconditioner, &[1.0, 2.0], None, &config)
            .expect_err("zero restart must fail before iteration");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let config = FgmresConfig {
            max_iterations: 8,
            restart: 2,
            tolerance: 1.0e-12,
            ..FgmresConfig::default()
        };
        let solution =
            try_fgmres_preconditioned(&matrix, &preconditioner, &[1.0, 2.0], None, &config)
                .expect("valid checked FGMRES solve");
        assert!(solution.converged);
    }

    #[test]
    fn checked_fgmres_rejects_non_finite_published_metrics() {
        let mut result = FgmresSolution {
            x: vec![0.0_f64],
            iterations: 0,
            restarts: 0,
            residual: 0.0,
            estimated_residual: f64::NAN,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::MaxIterationsReached,
            op_counts: KrylovOpCounts::default(),
        };

        let error = validate_checked_metrics(&result)
            .expect_err("non-finite projected residual must not be published");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(
            error
                .to_string()
                .contains("residual metrics contain NaN or infinity")
        );

        result.estimated_residual = 0.0;
        result.backward_error = f64::INFINITY;
        let error = validate_checked_metrics(&result)
            .expect_err("non-finite backward error must not be published");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_fgmres_rejects_non_finite_data_before_iteration() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let config = FgmresConfig {
            max_iterations: 8,
            restart: 2,
            tolerance: 1.0e-8,
            ..FgmresConfig::default()
        };
        let error = try_fgmres(&matrix, &[f64::NAN, 1.0], &config)
            .expect_err("NaN RHS must be rejected by checked FGMRES");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        let error = try_fgmres_preconditioned(
            &matrix,
            &sonium_solvers_core::traits::IdentityPreconditioner,
            &[1.0, 2.0],
            Some(&[f64::INFINITY, 0.0]),
            &config,
        )
        .expect_err("non-finite initial guess must be rejected");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_fgmres_rejects_non_finite_generated_solution() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let config = FgmresConfig {
            max_iterations: 8,
            restart: 2,
            tolerance: 1.0e-8,
            ..FgmresConfig::default()
        };
        let error = try_fgmres_preconditioned(
            &matrix,
            &NonFinitePreconditioner,
            &[1.0, 2.0],
            None,
            &config,
        )
        .expect_err("generated NaN solution must be rejected");
        match error {
            SolverError::OperationFailed { reason, .. } => {
                assert!(reason.contains("preconditioner"));
            }
            other => panic!("expected a checked preconditioner failure, got {other:?}"),
        }
    }

    #[test]
    fn checked_fgmres_rejects_non_finite_operator_output_during_iteration() {
        struct NonFiniteOperator;

        impl LinearOperator<f64> for NonFiniteOperator {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
                y.fill(f64::NAN);
            }

            fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
                y.fill(f64::NAN);
            }
        }

        let error = try_fgmres(
            &NonFiniteOperator,
            &[1.0, 2.0],
            &FgmresConfig {
                max_iterations: 8,
                restart: 2,
                tolerance: 1.0e-8,
                ..FgmresConfig::default()
            },
        )
        .expect_err("non-finite operator output must fail during iteration");
        match error {
            SolverError::OperationFailed { operation, reason } => {
                assert_eq!(operation, "FGMRES operator");
                assert!(reason.contains("operator generated"));
            }
            other => panic!("expected a checked operator failure, got {other:?}"),
        }
    }

    #[test]
    fn test_fgmres_simple_spd() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres(&a, &b, &config);

        assert!(solution.converged, "FGMRES should converge");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution should satisfy Ax = b");
    }

    #[test]
    fn test_fgmres_nonsymmetric() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres(&a, &b, &config);
        assert!(
            solution.converged,
            "FGMRES should converge for non-symmetric"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8);
    }

    #[test]
    fn test_fgmres_complex() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 1.0), Complex64::new(1.0, 0.5)],
            [Complex64::new(1.0, -0.5), Complex64::new(3.0, -1.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres(&a, &b, &config);
        assert!(
            solution.converged,
            "FGMRES should converge for complex system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8);
    }

    #[test]
    fn test_fgmres_with_preconditioner() {
        let n = 10;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = DiagonalPreconditioner::from_csr(&matrix);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = FgmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let sol = fgmres_preconditioned(&matrix, &precond, &b, None, &config);
        assert!(
            sol.converged,
            "FGMRES with Jacobi preconditioner should converge"
        );

        let ax = matrix.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8);
    }

    #[test]
    fn test_fgmres_identity() {
        let n = 5;
        let id: CsrMatrix<Complex64> = CsrMatrix::identity(n);
        let b = (1..=n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();

        let config = FgmresConfig {
            max_iterations: 10,
            restart: 10,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres(&id, &b, &config);

        assert!(solution.converged);
        assert!(solution.iterations <= 2);

        let error: f64 = sonium_solvers_core::vec_ops::vec_sub(&solution.x, &b)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-10);
    }

    #[test]
    fn test_fgmres_accepts_boundary_residual_at_tolerance() {
        let zero: CsrMatrix<f64> = CsrMatrix::from_triplets(1, 1, vec![]);
        let config = FgmresConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        };

        let solution = fgmres_preconditioned_unchecked(
            &zero,
            &sonium_solvers_core::traits::IdentityPreconditioner,
            &[1.0],
            None,
            &config,
            None,
        );

        assert!(solution.converged);
        assert_eq!(solution.status, SolverStatus::Converged);
        assert_eq!(solution.residual, 1.0);
    }

    #[test]
    fn test_fgmres_exact_residual_frequency_preserves_mandatory_gate() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let base_config = FgmresConfig {
            max_iterations: 10,
            restart: 10,
            tolerance: 0.9,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };
        let checked_config = FgmresConfig {
            exact_residual_check_interval: 1,
            ..base_config.clone()
        };

        let base = fgmres(&a, &b, &base_config);
        let checked = fgmres(&a, &b, &checked_config);

        assert!(base.converged);
        assert!(checked.converged);
        assert_eq!(base.iterations, checked.iterations);
        let base_true_residual = {
            let ax = a.matvec(&base.x);
            sonium_solvers_core::vec_ops::vec_sub(&ax, &b)
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt()
                / b.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt()
        };
        let checked_true_residual = {
            let ax = a.matvec(&checked.x);
            sonium_solvers_core::vec_ops::vec_sub(&ax, &b)
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt()
                / b.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt()
        };
        assert!((base.residual - base_true_residual).abs() < 1.0e-12);
        assert!((checked.residual - checked_true_residual).abs() < 1.0e-12);
        assert_eq!(base.op_counts.exact_residual_checks, 1);
        assert_eq!(checked.op_counts.exact_residual_checks, 1);
        assert_eq!(checked.op_counts.spmv_count, base.op_counts.spmv_count);
    }

    #[test]
    fn test_fgmres_honors_exact_residual_check_interval_before_estimated_convergence() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = [1.0_f64, 2.0];
        let base_config = FgmresConfig {
            max_iterations: 4,
            restart: 4,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };
        let checked_config = FgmresConfig {
            exact_residual_check_interval: 1,
            ..base_config.clone()
        };

        let base = fgmres(&a, &b, &base_config);
        let checked = fgmres(&a, &b, &checked_config);

        assert!(base.converged);
        assert!(checked.converged);
        assert!(
            checked.op_counts.exact_residual_checks > base.op_counts.exact_residual_checks,
            "the configured interval must check a candidate before projected convergence: base={}, checked={}",
            base.op_counts.exact_residual_checks,
            checked.op_counts.exact_residual_checks,
        );
        assert!(checked.op_counts.spmv_count > base.op_counts.spmv_count);
    }

    #[test]
    fn test_fgmres_f64() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres(&a, &b, &config);
        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions > 0);
        assert!(solution.op_counts.basis_bytes_touched > 0);

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert_relative_eq!(error, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_fgmres_large_diagonal_uses_batched_arnoldi_path() {
        let n = 20_000;
        let triplets = (0..n)
            .map(|i| {
                let diag = 2.0 + (i % 5) as f64 * 0.25;
                (i, i, Complex64::new(diag, 0.0))
            })
            .collect();
        let a = CsrMatrix::from_triplets(n, n, triplets);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.013).sin(), (i as f64 * 0.017).cos()))
            .collect::<Vec<_>>();

        let config = FgmresConfig {
            max_iterations: 3,
            restart: 8,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres(&a, &b, &config);
        assert!(
            solution.converged,
            "large diagonal FGMRES should converge, residual: {}",
            solution.residual
        );
        assert!(solution.iterations > 1, "test should exercise k > 1");

        let ax = a.matvec(&solution.x);
        let rel_error = sonium_solvers_core::vec_ops::vec_sub(&ax, &b)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / vector_norm(&b);
        assert!(rel_error < 1e-8, "relative error: {rel_error:.3e}");
    }

    #[test]
    fn test_fgmres_zero_rhs() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

        let config = FgmresConfig::default();
        let solution = fgmres(&a, &b, &config);

        assert!(solution.converged);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn fgmres_reports_promoted_metrics_and_preserves_estimate() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![1.0_f64, 2.0];
        let config = FgmresConfig {
            max_iterations: 8,
            restart: 4,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = try_fgmres(&a, &b, &config).expect("checked FGMRES");
        let ax = a.matvec(&solution.x);
        let residual_norm = ax
            .iter()
            .zip(&b)
            .map(|(applied, rhs)| (rhs - applied).powi(2))
            .sum::<f64>()
            .sqrt();
        let expected_backward = residual_norm
            / (ax.iter().map(|value| value * value).sum::<f64>().sqrt()
                + b.iter().map(|value| value * value).sum::<f64>().sqrt())
            .max(1.0);

        assert_eq!(solution.attainable_accuracy, AccuracyStatus::Attained);
        assert!(solution.estimated_residual.is_finite());
        assert!((solution.residual - residual_norm / vector_norm(&b)).abs() < 1e-14);
        assert!((solution.backward_error - expected_backward).abs() < 1e-14);
    }

    #[test]
    fn test_fgmres_zero_rhs_corrects_non_solution_initial_guess() {
        use sonium_solvers_core::traits::IdentityPreconditioner;

        let dense = sonium_solvers_core::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
        let x0 = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)];

        let solution = fgmres_preconditioned(
            &a,
            &IdentityPreconditioner,
            &b,
            Some(&x0),
            &FgmresConfig::default(),
        );
        assert!(solution.converged, "status={:?}", solution.status);
        assert_eq!(solution.status, SolverStatus::Converged);
        assert!(solution.residual < 1e-12);
        assert!(solution.x.iter().all(|value| value.norm() < 1e-12));
    }

    #[test]
    fn test_fgmres_breakdown_not_converged() {
        // Regression test: when FGMRES hits a breakdown (Krylov subspace
        // exhausted or linearly dependent), it should NOT report converged=true
        // unless the residual tolerance was actually met.
        //
        // Strategy: use a zero preconditioner so z_j = 0, w = A*0 = 0,
        // w_norm = 0 → breakdown on first inner iteration. With a large
        // b_norm, the relative residual remains large even though abs_residual
        // is tiny, so status=Breakdown is set.
        use sonium_solvers_core::traits::Preconditioner;

        struct ZeroPreconditioner;
        impl Preconditioner<Complex64> for ZeroPreconditioner {
            fn apply(&self, r: &[Complex64]) -> Vec<Complex64> {
                vec![Complex64::new(0.0, 0.0); r.len()]
            }
        }

        let n = 5;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(1.0, 0.0);
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);

        // Large b_norm so that even though abs_residual ≈ 0 after breakdown,
        // the relative residual abs_residual / b_norm is still > tolerance.
        // abs_residual ≈ 0, b_norm ≈ 1e10, rel_residual ≈ 0 < any tolerance.
        // But the status check is: inner_converged && rel >= tol && abs >= abs_tol
        // abs_tol = 1e-20, abs_residual = 0 → abs_residual < abs_tol → status = Converged
        // So we need a case where abs_residual is NOT tiny...
        //
        // Actually with zero precond, abs_residual IS 0, so status = Converged.
        // The bug is still real: converged=true is set unconditionally.
        // For the invariant test, we verify that converged always matches status.
        let b = sonium_solvers_core::array![
            Complex64::new(1e10, 0.0),
            Complex64::new(2e10, 0.0),
            Complex64::new(3e10, 0.0),
            Complex64::new(4e10, 0.0),
            Complex64::new(5e10, 0.0),
        ];

        let config = FgmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1e-15,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres_preconditioned(&a, &ZeroPreconditioner, &b, None, &config);

        // Invariant: converged must always match status == Converged
        assert_eq!(
            solution.converged,
            solution.status == SolverStatus::Converged,
            "converged={} does not match status={:?}. Breakdown must not report converged=true",
            solution.converged,
            solution.status
        );
    }

    #[test]
    fn test_fgmres_zero_preconditioner_no_false_convergence() {
        // Regression: a zero preconditioner must not produce false
        // convergence in FGMRES.  FGMRES uses right preconditioning
        // (AM^{-1} y = b), so the actual residual is minimised; a zero
        // preconditioner makes AM^{-1} = 0, and the solver should
        // detect that the residual does not improve.
        use sonium_solvers_core::traits::Preconditioner;

        struct ZeroPreconditioner;

        impl<T: ComplexField> Preconditioner<T> for ZeroPreconditioner {
            fn apply(&self, r: &[T]) -> Vec<T> {
                vec![T::zero(); r.len()]
            }
            fn apply_into(&self, _r: &[T], y: &mut [T]) {
                y.fill(T::zero());
            }
        }

        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = FgmresConfig {
            max_iterations: 10,
            restart: 5,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres_preconditioned(&a, &ZeroPreconditioner, &b, None, &config);

        // With a zero preconditioner the true residual never improves,
        // so the solver should NOT claim convergence.
        assert!(
            !solution.converged,
            "Zero preconditioner should not produce false convergence in FGMRES"
        );
    }

    #[test]
    fn test_fgmres_small_restart_reuses_hessenberg_across_cycles() {
        // Guards the hoisted Hessenberg/Givens/least-squares state: a
        // diagonal system with 5 distinct eigenvalues needs more than one
        // restart-2 cycle, so every cycle after the first reuses the cleared
        // buffers. A stale-clear bug would corrupt the projected problem and
        // either stall convergence or fail the true-residual gate below.
        let n = 25;
        let triplets: Vec<(usize, usize, f64)> =
            (0..n).map(|i| (i, i, 1.0 + (i % 5) as f64 * 0.5)).collect();
        let a = CsrMatrix::from_triplets(n, n, triplets);
        let b = vec![1.0_f64; n];

        let config = FgmresConfig {
            max_iterations: 100,
            restart: 2,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };

        let solution = fgmres(&a, &b, &config);
        assert!(
            solution.restarts >= 1,
            "test must span multiple restart cycles, restarts={}",
            solution.restarts
        );
        assert!(
            solution.converged,
            "FGMRES(2) should converge on a diagonal system, residual: {}",
            solution.residual
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt()
            / vector_norm(&b);
        assert!(error < 1e-8, "true relative residual: {error:.3e}");
    }
}
