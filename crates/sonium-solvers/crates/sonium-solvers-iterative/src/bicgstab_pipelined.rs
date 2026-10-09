//! Pipelined BiCGSTAB solver (Cools & Vanroose, 2017)
//!
//! Implements the communication-hiding pipelined BiCGStab method of
//! S. Cools and W. Vanroose, "The communication-hiding pipelined BiCGStab
//! method for the parallel solution of large unsymmetric linear systems",
//! Parallel Computing 65 (2017), Algorithm 9, extended with right
//! preconditioning following Section 3.6 of the same paper.
//!
//! Textbook BiCGSTAB has three global synchronization points per iteration
//! (the dot products for `rho`, `alpha` and `omega`), and each one blocks
//! the matvec that follows it. The pipelined variant recouples the vector
//! recurrences through auxiliary vectors so that
//!
//!   1. the dot products for `omega` fuse into one reduction phase that is
//!      independent of the SpMV `v = A * M^-1 * z`, and
//!   2. the four shadow-residual dot products feeding `beta` and the next
//!      `alpha` fuse into a second reduction phase that is independent of
//!      the SpMV `t = A * M^-1 * w`.
//!
//! Each reduction phase is overlapped with its independent preconditioner +
//! SpMV pair (via `rayon::join` on shared memory; the non-blocking
//! allreduce / SpMV overlap of the original distributed-memory algorithm).
//! The price is a chain of auxiliary vectors (`u, w, m_w, t, z, m_z, v`)
//! and one-time setup cost of two extra operator applications.
//!
//! Notation follows the paper, which differs from textbook BiCGSTAB:
//! `s = A M^-1 p` (textbook `v`), `q = r - alpha s` (textbook `s`),
//! `y = w - alpha z` (textbook `t = A M^-1 q`). Dots against the fixed
//! shadow residual use the unconjugated Lanczos bilinear form; the `omega`
//! dots use the conjugated inner product (least-squares minimizer), exactly
//! as in `bicgstab.rs`.

use super::gmres_common::exact_relative_residual;
use crate::bicgstab::{BiCgstabConfig, BiCgstabSolution};
use crate::bicgstab::{has_invalid_input, invalid_input_solution};
use crate::cgs_orth::KrylovOpCounts;
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{
    axpy, copy_axpy_norm, dot_unconjugated, inner_product, scale_inplace, vector_norm,
};
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, IdentityPreconditioner, LinearOperator,
    Preconditioner, SolverError, SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

/// Default interval (in iterations) at which the recursive residual is
/// replaced by the true residual `r = b - A*x` and the pipelined recurrences
/// are restarted from it (residual replacement, cf. Sleijpen & van der Vorst,
/// "Reliable updated residuals in hybrid Bi-CG methods"). The pipelined
/// recurrences accumulate more rounding error than textbook BiCGSTAB because
/// the auxiliary chains (`u, w, m_w, t`) are updated recursively as well, so
/// without replacement the attainable accuracy saturates well above machine
/// precision. Mirrors [`crate::bicgstab`].
pub const RESIDUAL_REPLACEMENT_INTERVAL: usize = 50;

/// Checked pipelined BiCGSTAB entry point using the identity preconditioner.
pub fn try_bicgstab_pipelined<T, A>(
    operator: &A,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
) -> Result<BiCgstabSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    try_bicgstab_pipelined_checked(
        operator,
        &IdentityPreconditioner,
        b,
        None,
        config,
        RESIDUAL_REPLACEMENT_INTERVAL,
    )
}

/// Checked preconditioned pipelined BiCGSTAB entry point.
pub fn try_bicgstab_pipelined_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
) -> Result<BiCgstabSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_bicgstab_pipelined_checked(
        operator,
        precond,
        b,
        None,
        config,
        RESIDUAL_REPLACEMENT_INTERVAL,
    )
}

/// Checked preconditioned pipelined BiCGSTAB entry point with an optional
/// initial guess.
pub fn try_bicgstab_pipelined_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &BiCgstabConfig<T::Real>,
) -> Result<BiCgstabSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_bicgstab_pipelined_checked(
        operator,
        precond,
        b,
        x0,
        config,
        RESIDUAL_REPLACEMENT_INTERVAL,
    )
}

/// Checked preconditioned pipelined BiCGSTAB entry point with a configurable
/// residual-replacement interval.
pub fn try_bicgstab_pipelined_preconditioned_with_residual_replacement<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
    residual_replacement_interval: usize,
) -> Result<BiCgstabSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_bicgstab_pipelined_checked(
        operator,
        precond,
        b,
        None,
        config,
        residual_replacement_interval,
    )
}

/// Checked unpreconditioned pipelined BiCGSTAB entry point with a
/// configurable residual-replacement interval.
pub fn try_bicgstab_pipelined_with_residual_replacement<T, A>(
    operator: &A,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
    residual_replacement_interval: usize,
) -> Result<BiCgstabSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    try_bicgstab_pipelined_checked(
        operator,
        &IdentityPreconditioner,
        b,
        None,
        config,
        residual_replacement_interval,
    )
}

fn try_bicgstab_pipelined_checked<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &BiCgstabConfig<T::Real>,
    residual_replacement_interval: usize,
) -> Result<BiCgstabSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let invalid_preconditioner_output = AtomicBool::new(false);
    let checked_operator = CheckedPipelinedOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let checked_precond = CheckedPipelinedPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner_output,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_pipelined_inputs(operator, b, x0, config)?;
        let solution = bicgstab_pipelined_core(
            &checked_operator,
            &checked_precond,
            b,
            x0,
            config,
            residual_replacement_interval,
        );
        let solution =
            finalize_pipelined_true_residual(&checked_operator, b, config.tolerance, solution);
        finish_checked_pipelined_solution(
            solution,
            b.len(),
            invalid_operator_output.load(Ordering::Relaxed),
            invalid_preconditioner_output.load(Ordering::Relaxed),
        )
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "pipelined BiCGSTAB",
        reason: panic_reason(payload),
    })??;
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn validate_pipelined_inputs<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &BiCgstabConfig<T::Real>,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "pipelined BiCGSTAB",
            reason: "invalid pipelined BiCGSTAB configuration".to_string(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "pipelined BiCGSTAB",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(guess) = x0 {
        if guess.len() != b.len() {
            return Err(SolverError::DimensionMismatch {
                expected: b.len(),
                got: guess.len(),
            });
        }
        if guess.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "pipelined BiCGSTAB",
                reason: "initial guess contains NaN or infinity".to_string(),
            });
        }
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
    Ok(())
}

fn finish_checked_pipelined_solution<T: ComplexField>(
    solution: BiCgstabSolution<T>,
    expected_dimension: usize,
    invalid_operator_output: bool,
    invalid_preconditioner_output: bool,
) -> Result<BiCgstabSolution<T>, SolverError> {
    if invalid_operator_output {
        return Err(SolverError::OperationFailed {
            operation: "pipelined BiCGSTAB operator",
            reason: "operator generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if invalid_preconditioner_output {
        return Err(SolverError::OperationFailed {
            operation: "pipelined BiCGSTAB preconditioner",
            reason: "preconditioner generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if solution.status == SolverStatus::NumericalError
        || solution.x.len() != expected_dimension
        || !solution.residual.is_finite()
        || !solution.estimated_residual.is_finite()
        || !solution.backward_error.is_finite()
        || solution.x.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "pipelined BiCGSTAB",
            reason: "generated NaN, infinity, or an invalid solution".to_string(),
        });
    }
    Ok(solution)
}

fn finalize_pipelined_true_residual<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut solution: BiCgstabSolution<T>,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let estimated_residual = solution.residual;
    let ax = operator.apply(&solution.x);
    let residual: Vec<T> = b
        .iter()
        .zip(&ax)
        .map(|(&rhs, &applied)| rhs - applied)
        .collect();
    let residual_norm = vector_norm(&residual);
    let rhs_norm = vector_norm(b);
    let ax_norm = vector_norm(&ax);
    let residual_denominator = if rhs_norm > T::Real::zero() {
        rhs_norm
    } else {
        T::Real::one()
    };
    solution.estimated_residual = estimated_residual;
    solution.residual = residual_norm / residual_denominator;
    solution.backward_error = T::normwise_backward_error(residual_norm, ax_norm, rhs_norm);
    solution.attainable_accuracy =
        if solution.residual.is_finite() && solution.residual <= tolerance {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        };
    solution.op_counts.spmv_count += 1;
    solution.op_counts.norm_count += 3;
    if !solution.residual.is_finite() || !solution.backward_error.is_finite() {
        solution.converged = false;
        solution.status = SolverStatus::NumericalError;
        solution.attainable_accuracy = AccuracyStatus::NotChecked;
    } else if solution.converged && solution.residual > tolerance {
        solution.converged = false;
        solution.status = SolverStatus::Stagnated;
    }
    solution
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "pipelined BiCGSTAB callback panicked".to_string()
    }
}

struct CheckedPipelinedOperator<'a, A> {
    inner: &'a A,
    invalid_output: &'a AtomicBool,
}

impl<T, A> LinearOperator<T> for CheckedPipelinedOperator<'_, A>
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

impl<A> CheckedPipelinedOperator<'_, A> {
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

struct CheckedPipelinedPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for CheckedPipelinedPreconditioner<'_, P>
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

impl<P> CheckedPipelinedPreconditioner<'_, P> {
    fn validate<T: ComplexField>(&self, input: &[T], output: &mut [T]) {
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

/// Solve Ax = b using the Pipelined BiCGSTAB method (unpreconditioned).
pub fn bicgstab_pipelined<T, A>(
    operator: &A,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_bicgstab_pipelined(operator, b, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

/// Solve Ax = b using the Pipelined BiCGSTAB method with a preconditioner.
pub fn bicgstab_pipelined_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_bicgstab_pipelined_preconditioned(operator, precond, b, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

/// Solve `Ax = b` using pipelined BiCGSTAB, a right preconditioner, and an
/// optional initial guess.
pub fn bicgstab_pipelined_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &BiCgstabConfig<T::Real>,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_bicgstab_pipelined_preconditioned_with_guess(operator, precond, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, x0),
    }
}

/// Solve `Ax = b` using pipelined BiCGSTAB with a right preconditioner and a
/// configurable residual-replacement interval.
///
/// Every `residual_replacement_interval` iterations the true residual
/// `r = b - A*x` is recomputed with one SpMV and the pipelined recurrences
/// (including the auxiliary chain `u, w, m_w, t`) are regenerated from it.
/// Pass 0 to disable residual replacement.
pub fn bicgstab_pipelined_preconditioned_with_residual_replacement<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
    residual_replacement_interval: usize,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_bicgstab_pipelined_preconditioned_with_residual_replacement(
        operator,
        precond,
        b,
        config,
        residual_replacement_interval,
    ) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

/// Solve `Ax = b` using unpreconditioned pipelined BiCGSTAB with a
/// configurable residual-replacement interval (0 disables).
pub fn bicgstab_pipelined_with_residual_replacement<T, A>(
    operator: &A,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
    residual_replacement_interval: usize,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_bicgstab_pipelined_with_residual_replacement(
        operator,
        b,
        config,
        residual_replacement_interval,
    ) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

/// Core pipelined BiCGSTAB with right preconditioning.
///
/// Auxiliary chain (right-preconditioned, paper Sec. 3.6):
///   u = M^-1 r,  w = A u,  m_w = M^-1 w,  t = A m_w,
/// and the recoupled per-iteration vectors
///   p,   p_h = M^-1 p,  s = A p_h,  s_h = M^-1 s,  z = A s_h,  m_z = M^-1 z,  v = A m_z.
///
/// Per iteration (paper Algorithm 9, preconditioned):
///   alpha from the previous fused reduction (paper eq. (3)), then
///     p   = r + beta * (p - omega * s)        s   = w + beta * (s - omega * z)
///     p_h = u + beta * (p_h - omega * s_h)    s_h = m_w + beta * (s_h - omega * m_z)
///     z   = t + beta * (z - omega * v)
///     q   = r - alpha * s                     y   = w - alpha * z
///   [REDUCTION A: (y,q), (y,y)] || [m_z = M^-1 z, v = A m_z]
///     omega = (y,q) / (y,y)
///     q_h = u - alpha * s_h
///     x  += alpha * p_h + omega * q_h
///     r   = q - omega * y
///     u   = q_h - omega * (m_w - alpha * m_z)
///     w   = y - omega * (t - alpha * v)
///   [REDUCTION B: (r0,r), (r0,w), (r0,s), (r0,z)] || [m_w = M^-1 w, t = A m_w]
///     beta = (alpha / omega) * (rho_new / rho)
///
/// Both reduction phases overlap with the independent preconditioner + SpMV
/// pair via `rayon::join`, so no dot product ever blocks a matvec.
///
/// Residual replacement: every `residual_replacement_interval` iterations
/// (0 disables) the true residual `r = b - A*x` is recomputed with one SpMV,
/// the shadow residual is reset to it, the auxiliary chain
/// `u = M^-1 r, w = A u, m_w = M^-1 w, t = A m_w` is regenerated (the same
/// 2 preconditioner + 2 SpMV setup cost as the initial iteration), and the
/// recoupled chains are restarted by zeroing `beta_prev`: with
/// `beta_prev = 0` the next iteration's `scale_inplace(beta)` annihilates the
/// stale `p, p_h, s, s_h, z` and rebuilds them purely from the refreshed
/// `(r, u, w, m_w, t)`, exactly as in the first iteration. This mirrors the
/// bi-Lanczos restart of `bicgstab.rs` for the pipelined auxiliary chains.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn bicgstab_pipelined_core<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &BiCgstabConfig<T::Real>,
    residual_replacement_interval: usize,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.len();
    if has_invalid_input(b, x0, config)
        || operator.num_rows() != operator.num_cols()
        || n != operator.num_cols()
    {
        return invalid_input_solution(b, x0);
    }
    let zero_tol = T::Real::from_f64(1e-15).unwrap();
    // omega is a ratio (scale-invariant), so an absolute threshold applies
    let omega_tol = T::Real::from_f64(1e-20).unwrap();

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut op_counts = KrylovOpCounts::default();

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;

    // Breakdown threshold scaled with ||b||^2 (same construction as
    // bicgstab.rs): dots against the shadow residual scale like ||b||^2.
    let eps = T::Real::epsilon();
    let scaled = b_norm * b_norm * eps * eps;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_tol = if scaled > floor { scaled } else { floor };

    // Initial residual r = b - Ax0 (unpreconditioned)
    let mut r = if x0.is_some() {
        let mut ax = vec![T::zero(); n];
        operator.apply_into(&x, &mut ax);
        op_counts.spmv_count += 1;
        let mut res = b.to_vec();
        axpy(-T::one(), &ax, &mut res);
        op_counts.axpy_passes += 1;
        res
    } else {
        b.to_vec()
    };
    let initial_residual = vector_norm(&r);
    op_counts.norm_count += 1;
    if b_norm < zero_tol {
        return BiCgstabSolution {
            x,
            iterations: 0,
            residual: initial_residual,
            estimated_residual: initial_residual,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: initial_residual < zero_tol,
            status: if initial_residual < zero_tol {
                SolverStatus::Converged
            } else {
                SolverStatus::Breakdown
            },
            op_counts,
        };
    }
    let initial_relative_residual = initial_residual / b_norm;
    if initial_relative_residual < config.tolerance {
        return BiCgstabSolution {
            x,
            iterations: 0,
            residual: initial_relative_residual,
            estimated_residual: initial_relative_residual,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }

    let mut r0 = r.clone(); // Shadow residual (refreshed on residual replacement)

    // Setup of the auxiliary chain: u = M^-1 r, w = A u, m_w = M^-1 w, t = A m_w.
    // One-time cost of 2 preconditioner applications + 2 SpMVs (paper line 2).
    let mut u = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];
    let mut m_w = vec![T::zero(); n];
    let mut t = vec![T::zero(); n];
    precond.apply_into(&r, &mut u);
    operator.apply_into(&u, &mut w);
    precond.apply_into(&w, &mut m_w);
    operator.apply_into(&m_w, &mut t);
    op_counts.preconditioner_count += 2;
    op_counts.spmv_count += 2;

    // First fused reduction: rho = <r0, r> and r0w = <r0, w>
    // (unconjugated Lanczos dots, as in bicgstab.rs).
    let mut rho = dot_unconjugated(&r0, &r);
    let mut r0w = dot_unconjugated(&r0, &w);
    op_counts.dot_reductions += 2;
    op_counts.synchronization_points += 1;
    if rho.norm() < breakdown_tol {
        return BiCgstabSolution {
            x,
            iterations: 0,
            residual: initial_relative_residual,
            estimated_residual: initial_relative_residual,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts,
        };
    }

    // Scalars from the "previous" fused reduction (beta_-1 = 0 annihilates
    // the correction terms in the first iteration).
    let mut beta_prev = T::zero();
    let mut omega_prev = T::one();
    let mut r0s_prev = T::zero();
    let mut r0z_prev = T::zero();

    // Remaining work buffers (zero allocation in the inner loop).
    let mut p = vec![T::zero(); n];
    let mut p_h = vec![T::zero(); n];
    let mut s = vec![T::zero(); n];
    let mut s_h = vec![T::zero(); n];
    let mut z = vec![T::zero(); n];
    let mut m_z = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut q = vec![T::zero(); n];
    let mut q_h = vec![T::zero(); n];
    let mut y = vec![T::zero(); n];

    for iter in 0..config.max_iterations {
        // Residual replacement: recompute the true residual r = b - A*x with
        // one SpMV and restart the pipelined recurrences from it. The chains
        // p, p_h, s, s_h, z need no explicit clearing: beta_prev = 0 makes the
        // scale_inplace(beta) below annihilate them, so they are rebuilt from
        // the refreshed (r, u, w, m_w, t) exactly as in the first iteration.
        if residual_replacement_interval > 0
            && iter > 0
            && iter % residual_replacement_interval == 0
        {
            operator.apply_into(&x, &mut y);
            op_counts.spmv_count += 1;
            op_counts.exact_residual_checks += 1;
            r.copy_from_slice(b);
            axpy(-T::one(), &y, &mut r);
            op_counts.axpy_passes += 1;

            let true_rel_residual = vector_norm(&r) / b_norm;
            op_counts.norm_count += 1;
            if true_rel_residual < config.tolerance {
                return BiCgstabSolution {
                    x,
                    iterations: iter,
                    residual: true_rel_residual,
                    estimated_residual: true_rel_residual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }

            // New shadow residual and regenerated auxiliary chain
            // u = M^-1 r, w = A u, m_w = M^-1 w, t = A m_w.
            r0.copy_from_slice(&r);
            precond.apply_into(&r, &mut u);
            operator.apply_into(&u, &mut w);
            precond.apply_into(&w, &mut m_w);
            operator.apply_into(&m_w, &mut t);
            op_counts.preconditioner_count += 2;
            op_counts.spmv_count += 2;

            rho = dot_unconjugated(&r0, &r);
            r0w = dot_unconjugated(&r0, &w);
            op_counts.dot_reductions += 2;
            op_counts.synchronization_points += 1;
            if rho.norm() < breakdown_tol {
                return BiCgstabSolution {
                    x,
                    iterations: iter,
                    residual: true_rel_residual,
                    estimated_residual: true_rel_residual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: false,
                    status: SolverStatus::Breakdown,
                    op_counts,
                };
            }

            beta_prev = T::zero();
            omega_prev = T::one();
            r0s_prev = T::zero();
            r0z_prev = T::zero();
        }

        // alpha from the dots of the previous fused reduction (paper eq. (3)):
        //   alpha = rho / (<r0,w> + beta * <r0,s_prev> - beta * omega * <r0,z_prev>)
        let alpha_denom = r0w + beta_prev * r0s_prev - beta_prev * omega_prev * r0z_prev;
        if alpha_denom.norm() < breakdown_tol {
            let residual = vector_norm(&r) / b_norm;
            op_counts.norm_count += 1;
            return BiCgstabSolution {
                x,
                iterations: iter,
                residual,
                estimated_residual: residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }
        let alpha = rho / alpha_denom;

        // Recoupled search-direction updates (paper lines 4-6).
        let neg_beta_omega = -(beta_prev * omega_prev);
        scale_inplace(&mut p, beta_prev);
        axpy(neg_beta_omega, &s, &mut p);
        axpy(T::one(), &r, &mut p);
        scale_inplace(&mut p_h, beta_prev);
        axpy(neg_beta_omega, &s_h, &mut p_h);
        axpy(T::one(), &u, &mut p_h);
        scale_inplace(&mut s, beta_prev);
        axpy(neg_beta_omega, &z, &mut s);
        axpy(T::one(), &w, &mut s);
        scale_inplace(&mut s_h, beta_prev);
        axpy(neg_beta_omega, &m_z, &mut s_h);
        axpy(T::one(), &m_w, &mut s_h);
        scale_inplace(&mut z, beta_prev);
        axpy(neg_beta_omega, &v, &mut z);
        axpy(T::one(), &t, &mut z);
        op_counts.axpy_passes += 15;

        // q = r - alpha * s (paper line 7; textbook "s")
        let q_norm = copy_axpy_norm(&r, -alpha, &s, &mut q);
        op_counts.axpy_passes += 1;
        op_counts.norm_count += 1;
        if q_norm / b_norm < config.tolerance {
            axpy(alpha, &p_h, &mut x);
            op_counts.axpy_passes += 1;
            let true_rel_residual =
                exact_relative_residual(operator, &x, b, &mut r, b_norm, &mut op_counts);
            if true_rel_residual < config.tolerance {
                return BiCgstabSolution {
                    x,
                    iterations: iter + 1,
                    residual: true_rel_residual,
                    estimated_residual: true_rel_residual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
        }

        // y = w - alpha * z (paper line 8; textbook "t" = A M^-1 q)
        y.copy_from_slice(&w);
        axpy(-alpha, &z, &mut y);
        op_counts.axpy_passes += 2;

        // REDUCTION A (paper lines 9-11): the conjugated dots for omega are
        // independent of m_z = M^-1 z and v = A m_z, so the reduction and the
        // operator work proceed concurrently.
        let ((yq, yy), ()) = rayon::join(
            || (inner_product(&y, &q), inner_product(&y, &y)),
            || {
                precond.apply_into(&z, &mut m_z);
                operator.apply_into(&m_z, &mut v);
            },
        );
        op_counts.dot_reductions += 2;
        op_counts.synchronization_points += 1;
        op_counts.preconditioner_count += 1;
        op_counts.spmv_count += 1;

        if yy.norm() < breakdown_tol {
            let residual = vector_norm(&r) / b_norm;
            op_counts.norm_count += 1;
            return BiCgstabSolution {
                x,
                iterations: iter,
                residual,
                estimated_residual: residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }
        let omega = yq / yy;

        // q_h = u - alpha * s_h (= M^-1 q); x += alpha * p_h + omega * q_h
        q_h.copy_from_slice(&u);
        axpy(-alpha, &s_h, &mut q_h);
        axpy(alpha, &p_h, &mut x);
        axpy(omega, &q_h, &mut x);
        op_counts.axpy_passes += 4;

        // r = q - omega * y (paper line 14)
        let rel_residual = copy_axpy_norm(&q, -omega, &y, &mut r) / b_norm;
        op_counts.axpy_passes += 1;
        op_counts.norm_count += 1;

        if config.print_interval > 0 && (iter + 1) % config.print_interval == 0 {
            log::info!(
                "p-BiCGSTAB iteration {}: relative residual = {:.6e}",
                iter + 1,
                rel_residual.to_f64().unwrap_or(0.0)
            );
        }

        if rel_residual < config.tolerance {
            let true_rel_residual =
                exact_relative_residual(operator, &x, b, &mut r, b_norm, &mut op_counts);
            if true_rel_residual < config.tolerance {
                return BiCgstabSolution {
                    x,
                    iterations: iter + 1,
                    residual: true_rel_residual,
                    estimated_residual: true_rel_residual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
        }

        if omega.norm() < omega_tol {
            return BiCgstabSolution {
                x,
                iterations: iter + 1,
                residual: rel_residual,
                estimated_residual: rel_residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Stagnated,
                op_counts,
            };
        }

        // u = q_h - omega * (m_w - alpha * m_z)  (= M^-1 r_new)
        u.copy_from_slice(&q_h);
        axpy(-omega, &m_w, &mut u);
        axpy(omega * alpha, &m_z, &mut u);
        // w = y - omega * (t - alpha * v)  (= A u_new, paper line 15)
        w.copy_from_slice(&y);
        axpy(-omega, &t, &mut w);
        axpy(omega * alpha, &v, &mut w);
        op_counts.axpy_passes += 6;

        // REDUCTION B (paper lines 16-18): the four shadow-residual dots
        // feeding beta and the next alpha are independent of m_w = M^-1 w and
        // t = A m_w, so the reduction and the operator work proceed
        // concurrently.
        let ((rho_new, r0w_new, r0s, r0z), ()) = rayon::join(
            || {
                (
                    dot_unconjugated(&r0, &r),
                    dot_unconjugated(&r0, &w),
                    dot_unconjugated(&r0, &s),
                    dot_unconjugated(&r0, &z),
                )
            },
            || {
                precond.apply_into(&w, &mut m_w);
                operator.apply_into(&m_w, &mut t);
            },
        );
        op_counts.dot_reductions += 4;
        op_counts.synchronization_points += 1;
        op_counts.preconditioner_count += 1;
        op_counts.spmv_count += 1;

        if rho_new.norm() < breakdown_tol {
            return BiCgstabSolution {
                x,
                iterations: iter + 1,
                residual: rel_residual,
                estimated_residual: rel_residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }

        // beta = (alpha / omega) * (rho_new / rho) (paper line 19)
        beta_prev = (alpha / omega) * (rho_new / rho);
        rho = rho_new;
        r0w = r0w_new;
        r0s_prev = r0s;
        r0z_prev = r0z;
        omega_prev = omega;
    }

    let rel_residual = vector_norm(&r) / b_norm;
    op_counts.norm_count += 1;
    BiCgstabSolution {
        x,
        iterations: config.max_iterations,
        residual: rel_residual,
        estimated_residual: rel_residual,
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bicgstab::bicgstab;
    use num_complex::Complex64;
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner};
    use sonium_solvers_preconditioners::DiagonalPreconditioner;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn checked_pipelined_bicgstab_contains_callback_failures() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("pipelined BiCGSTAB operator callback failed");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("pipelined BiCGSTAB operator callback failed");
            }
        }

        let config = BiCgstabConfig {
            max_iterations: 10,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let error = try_bicgstab_pipelined(&PanicOperator, &[1.0], &config)
            .expect_err("checked pipelined BiCGSTAB must type callback panics");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "pipelined BiCGSTAB",
                ..
            }
        ));

        let legacy = bicgstab_pipelined(&PanicOperator, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    #[test]
    fn checked_pipelined_bicgstab_rejects_non_finite_generated_values() {
        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
                output[0] = f64::NAN;
            }
        }

        let operator = CsrMatrix::identity(1);
        let precond = NonFinitePreconditioner;
        let config = BiCgstabConfig::default();
        let error = try_bicgstab_pipelined_preconditioned(&operator, &precond, &[1.0], &config)
            .expect_err("checked pipelined BiCGSTAB must reject non-finite preconditioner output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "pipelined BiCGSTAB preconditioner",
                ..
            }
        ));

        let legacy = bicgstab_pipelined_preconditioned(&operator, &precond, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    #[test]
    fn test_pipelined_bicgstab_accepts_exact_initial_guess() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, -2.0];
        let config = BiCgstabConfig {
            max_iterations: 10,
            tolerance: 1.0e-12,
            print_interval: 0,
        };

        let solution = bicgstab_pipelined_preconditioned_with_guess(
            &a,
            &IdentityPreconditioner,
            &b,
            Some(&b),
            &config,
        );

        assert!(solution.converged);
        assert_eq!(solution.iterations, 0);
        assert!(solution.residual <= config.tolerance);
    }

    #[test]
    fn test_pipelined_bicgstab_zero_rhs_rejects_non_solution_guess() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [0.0, 0.0];
        let x0 = [1.0, -2.0];

        let solution = bicgstab_pipelined_preconditioned_with_guess(
            &a,
            &IdentityPreconditioner,
            &b,
            Some(&x0),
            &BiCgstabConfig::default(),
        );

        assert!(!solution.converged);
        assert_eq!(solution.status, SolverStatus::Breakdown);
        assert!(solution.residual > 0.0);
    }

    #[test]
    fn test_pipelined_bicgstab_simple() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = BiCgstabConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab_pipelined(&a, &b, &config);
        assert!(solution.converged, "Pipelined BiCGSTAB should converge");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "Solution should satisfy Ax = b, error = {error}"
        );
    }

    #[test]
    fn pipelined_bicgstab_reports_independent_accuracy_on_max_iterations() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let config = BiCgstabConfig {
            max_iterations: 1,
            tolerance: 1e-15,
            print_interval: 0,
        };

        let solution =
            try_bicgstab_pipelined(&a, &b, &config).expect("finite pipelined max-iteration result");
        let ax = a.matvec(&solution.x);
        let residual = sonium_solvers_core::vec_ops::vec_sub(&b, &ax);
        let residual_norm = residual
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let rhs_norm = b.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
        let ax_norm = ax.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();

        assert_eq!(solution.status, SolverStatus::MaxIterationsReached);
        assert!(!solution.converged);
        assert!(solution.estimated_residual.is_finite());
        assert!((solution.residual - residual_norm / rhs_norm).abs() < 1e-14);
        assert!(
            (solution.backward_error - residual_norm / (ax_norm + rhs_norm).max(f64::MIN_POSITIVE))
                .abs()
                < 1e-14
        );
        assert_eq!(solution.attainable_accuracy, AccuracyStatus::NotAttained);
    }

    #[test]
    fn test_pipelined_matches_standard() {
        // Verify pipelined BiCGSTAB produces the same solution as standard BiCGSTAB
        let n = 20;
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

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = BiCgstabConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let standard = bicgstab(&a, &b, &config);
        let pipelined = bicgstab_pipelined(&a, &b, &config);

        assert!(standard.converged, "Standard BiCGSTAB should converge");
        assert!(pipelined.converged, "Pipelined BiCGSTAB should converge");

        // Both should produce the same solution within tolerance
        let diff: f64 = sonium_solvers_core::vec_ops::vec_sub(&standard.x, &pipelined.x)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 1e-6,
            "Pipelined and standard should match, diff = {diff}"
        );
    }

    #[test]
    fn test_pipelined_bicgstab_complex() {
        // Complex system with non-zero imaginary parts
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 1.0), Complex64::new(1.0, -0.5)],
            [Complex64::new(1.0, 0.5), Complex64::new(3.0, -1.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(2.0, 1.0), Complex64::new(1.0, -1.0)];

        let config = BiCgstabConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab_pipelined(&a, &b, &config);
        assert!(
            solution.converged,
            "Pipelined BiCGSTAB should converge for complex system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Complex solution error too large: {error}");
    }

    #[test]
    fn test_pipelined_bicgstab_preconditioned() {
        // Preconditioned pipelined BiCGSTAB with diagonal (Jacobi) preconditioner
        let n = 30;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(10.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-2.0, 0.5);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-2.0, -0.5);
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = DiagonalPreconditioner::from_csr(&a);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).cos(), (i as f64).sin()))
            .collect::<Vec<_>>();

        let config = BiCgstabConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab_pipelined_preconditioned(&a, &precond, &b, &config);
        assert!(
            solution.converged,
            "Preconditioned pipelined BiCGSTAB should converge"
        );

        // Verify Ax = b (not M^{-1}Ax = M^{-1}b)
        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-6,
            "Preconditioned solution error too large: {error}"
        );
    }

    #[test]
    fn test_pipelined_bicgstab_nonsymmetric_larger() {
        // Strongly nonsymmetric complex tridiagonal system; the pipelined
        // recurrences must track standard BiCGSTAB.
        let n = 40;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(5.0, 0.25);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-3.0, 0.1);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, -0.2);
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).cos(), 0.3 * (i as f64).sin()))
            .collect::<Vec<_>>();

        let config = BiCgstabConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let standard = bicgstab(&a, &b, &config);
        let pipelined = bicgstab_pipelined(&a, &b, &config);

        assert!(standard.converged, "Standard BiCGSTAB should converge");
        assert!(
            pipelined.converged,
            "Pipelined BiCGSTAB should converge on nonsymmetric system"
        );

        let ax = a.matvec(&pipelined.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-7, "Residual too large: {error}");

        let diff: f64 = sonium_solvers_core::vec_ops::vec_sub(&standard.x, &pipelined.x)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 1e-5,
            "Pipelined and standard should match, diff = {diff}"
        );
    }

    #[test]
    fn test_pipelined_bicgstab_real_f64_nonsymmetric() {
        // Real nonsymmetric 3x3 system (off-diagonal entries differ)
        let dense =
            sonium_solvers_core::array![[4.0f64, 2.0, 0.0], [1.0, 5.0, 1.0], [0.0, 3.0, 6.0],];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0f64, 2.0, 3.0];

        let config = BiCgstabConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab_pipelined(&a, &b, &config);
        assert!(
            solution.converged,
            "Pipelined BiCGSTAB should converge for real nonsymmetric system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution error too large: {error}");
    }

    #[test]
    fn test_pipelined_bicgstab_residual_replacement_accuracy() {
        // Attainable-accuracy regression: an ill-conditioned complex
        // non-Hermitian system solved far below 1e-10 relative. Without
        // residual replacement the recursively updated residual (and the
        // recursively updated auxiliary chains) drift from the true residual
        // and the solver stagnates orders of magnitude above the requested
        // tolerance; with replacement every 50 iterations the true residual
        // must actually reach it.
        let n = 100;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(2.0, 0.3);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.05);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, -0.05);
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();

        let config = BiCgstabConfig {
            max_iterations: 2000,
            tolerance: 1e-12,
            print_interval: 0,
        };

        let solution = bicgstab_pipelined(&a, &b, &config);
        assert!(
            solution.converged,
            "p-BiCGSTAB with residual replacement should converge, residual = {}",
            solution.residual
        );
        assert!(
            solution.iterations > 50,
            "test must exercise the residual-replacement path (iterations = {})",
            solution.iterations
        );

        // The reported residual is recursive; verify against the true residual.
        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            error / b_norm < 1e-10,
            "true relative residual too large after residual replacement: {}",
            error / b_norm
        );
    }

    #[test]
    fn pipelined_bicgstab_rejects_non_finite_rhs_before_operator_use() {
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

        let solution = bicgstab_pipelined(&PanicOperator, &[f64::NAN], &BiCgstabConfig::default());

        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
    }

    #[test]
    fn pipelined_bicgstab_does_not_report_preconditioned_false_convergence() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 100.0 * r[1];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0];
        let config = BiCgstabConfig {
            max_iterations: 1,
            tolerance: 1.0e-1,
            print_interval: 0,
        };
        let solution =
            bicgstab_pipelined_preconditioned(&a, &AnisotropicPreconditioner, &b, &config);
        let ax = a.matvec(&solution.x);
        let true_residual = sonium_solvers_core::vec_ops::vec_sub(&ax, &b)
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / b.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();

        assert!(!solution.converged);
        assert!(true_residual >= config.tolerance);
    }
}
