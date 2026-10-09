//! IDR(s) (Induced Dimension Reduction) solver
//!
//! Implementation of the IDR(s) algorithm by Sonneveld & van Gijzen (2008).
//!
//! IDR(s) is a modern Krylov method for non-symmetric systems that generalizes
//! BiCGSTAB (which corresponds to s=1). With s=4 or s=8, IDR(s) typically
//! converges in significantly fewer iterations than BiCGSTAB for complex
//! non-symmetric systems like those arising from Helmholtz/BEM.
//!
//! The residual is updated recursively (`r -= alpha * g`), so rounding error
//! makes it drift from the true residual `b - A*x` over a long solve, capping
//! the attainable accuracy. Every [`IDR_RESIDUAL_REPLACEMENT_INTERVAL`]
//! matrix-vector products the true residual is recomputed with one SpMV
//! (residual replacement) and the recurrence continues from it. The G/U
//! Krylov columns are kept: they are A-images of previously built search
//! directions and do not depend on the residual, so only `f = P^T r` and the
//! shadow matrix `M = P^T G` must be refreshed, which happens at the top of
//! every outer cycle anyway (cf. Aihara et al., "a variant of IDR(s) with
//! reliable update strategies").

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use super::cgs_orth::KrylovOpCounts;

/// Default interval (in matrix-vector products) at which the recursively
/// updated residual is replaced by the true residual `r = b - A*x`
/// (residual replacement). This curbs the rounding-error build-up that
/// otherwise limits the attainable accuracy of IDR(s), cf. the analogous
/// mechanism in [`super::bicgstab`].
pub const IDR_RESIDUAL_REPLACEMENT_INTERVAL: usize = 50;

/// IDR(s) solver configuration
#[derive(Debug, Clone)]
pub struct IdrsConfig<R> {
    /// Shadow space dimension (s=1 gives BiCGSTAB-like behavior, s=4 is typical)
    pub s: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
}

impl Default for IdrsConfig<f64> {
    fn default() -> Self {
        Self {
            s: 4,
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        }
    }
}

impl Default for IdrsConfig<f32> {
    fn default() -> Self {
        Self {
            s: 4,
            max_iterations: 1000,
            tolerance: 1e-5,
            print_interval: 0,
        }
    }
}

/// IDR(s) solver result
#[derive(Debug)]
pub struct IdrsSolution<T: ComplexField> {
    /// Solution vector
    pub x: Vec<T>,
    /// Number of iterations (matrix-vector products)
    pub iterations: usize,
    /// Final relative residual
    pub residual: T::Real,
    /// Residual estimate captured before the independent final check.
    pub estimated_residual: T::Real,
    /// Normwise backward error against the original operator.
    pub backward_error: T::Real,
    /// Whether the returned iterate satisfies the requested true-residual tolerance.
    pub attainable_accuracy: AccuracyStatus,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Solver status
    pub status: SolverStatus,
    /// Memory-traffic oriented operation counts for the Krylov solve.
    pub op_counts: KrylovOpCounts,
}

fn numerical_error_solution<T: ComplexField>(n: usize) -> IdrsSolution<T> {
    IdrsSolution {
        x: vec![T::zero(); n],
        iterations: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
    }
}

fn compatibility_error_solution<T: ComplexField>(n: usize) -> IdrsSolution<T> {
    numerical_error_solution(n)
}

/// Checked IDR(s) entry point using the identity preconditioner.
pub fn try_idrs<T, A>(
    operator: &A,
    b: &[T],
    config: &IdrsConfig<T::Real>,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    try_idrs_checked(
        operator,
        &identity,
        b,
        config,
        IDR_RESIDUAL_REPLACEMENT_INTERVAL,
    )
}

/// Checked IDR(s) entry point using the identity preconditioner and a
/// configurable residual-replacement interval.
pub fn try_idrs_with_residual_replacement<T, A>(
    operator: &A,
    b: &[T],
    config: &IdrsConfig<T::Real>,
    residual_replacement_interval: usize,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    try_idrs_checked(
        operator,
        &identity,
        b,
        config,
        residual_replacement_interval,
    )
}

/// Checked preconditioned IDR(s) entry point.
pub fn try_idrs_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsConfig<T::Real>,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_idrs_preconditioned_with_residual_replacement(
        operator,
        precond,
        b,
        config,
        IDR_RESIDUAL_REPLACEMENT_INTERVAL,
    )
}

/// Checked preconditioned IDR(s) entry point with a configurable
/// residual-replacement interval.
pub fn try_idrs_preconditioned_with_residual_replacement<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsConfig<T::Real>,
    residual_replacement_interval: usize,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_idrs_checked(operator, precond, b, config, residual_replacement_interval)
}

fn try_idrs_checked<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsConfig<T::Real>,
    residual_replacement_interval: usize,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let invalid_preconditioner_output = AtomicBool::new(false);
    let checked_operator = CheckedIdrsOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let checked_precond = CheckedIdrsPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner_output,
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_idrs_inputs(operator, b, config)?;
        let solution = idrs_core(
            &checked_operator,
            &checked_precond,
            b,
            config,
            residual_replacement_interval,
        );
        let solution = finalize_idrs_solution(&checked_operator, b, config.tolerance, solution);
        finish_checked_idrs_solution(
            solution,
            b.len(),
            invalid_operator_output.load(Ordering::Relaxed),
            invalid_preconditioner_output.load(Ordering::Relaxed),
        )
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "IDR(s)",
        reason: panic_reason(payload),
    })??;

    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn validate_idrs_inputs<T, A>(
    operator: &A,
    b: &[T],
    config: &IdrsConfig<T::Real>,
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
            operation: "IDR(s)",
            reason: "invalid IDR(s) configuration".to_string(),
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
            operation: "IDR(s)",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn finish_checked_idrs_solution<T: ComplexField>(
    solution: IdrsSolution<T>,
    expected_dimension: usize,
    invalid_operator_output: bool,
    invalid_preconditioner_output: bool,
) -> Result<IdrsSolution<T>, SolverError> {
    if invalid_operator_output {
        return Err(SolverError::OperationFailed {
            operation: "IDR(s) operator",
            reason: "operator generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if invalid_preconditioner_output {
        return Err(SolverError::OperationFailed {
            operation: "IDR(s) preconditioner",
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
            operation: "IDR(s)",
            reason: "generated NaN, infinity, or an invalid solution".to_string(),
        });
    }
    Ok(solution)
}

fn finalize_idrs_solution<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut result: IdrsSolution<T>,
) -> IdrsSolution<T>
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

    let mut residual = vec![T::zero(); b.len()];
    operator.apply_into(&result.x, &mut residual);
    result.op_counts.spmv_count += 1;
    let mut residual_norm_sq = T::Real::zero();
    let mut ax_norm_sq = T::Real::zero();
    let mut rhs_norm_sq = T::Real::zero();
    for (rhs, applied) in b.iter().zip(&mut residual) {
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
        "IDR(s) callback panicked".to_string()
    }
}

struct CheckedIdrsOperator<'a, A> {
    inner: &'a A,
    invalid_output: &'a AtomicBool,
}

impl<T, A> LinearOperator<T> for CheckedIdrsOperator<'_, A>
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

impl<A> CheckedIdrsOperator<'_, A> {
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

struct CheckedIdrsPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for CheckedIdrsPreconditioner<'_, P>
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

impl<P> CheckedIdrsPreconditioner<'_, P> {
    fn validate<T: ComplexField>(&self, input: &[T], output: &mut [T]) {
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

/// Solve Ax = b using IDR(s) (no preconditioner)
pub fn idrs<T, A>(operator: &A, b: &[T], config: &IdrsConfig<T::Real>) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_idrs(operator, b, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b.len()),
    }
}

/// Solve Ax = b using IDR(s) with a configurable residual-replacement
/// interval in matrix-vector products (0 disables).
pub fn idrs_with_residual_replacement<T, A>(
    operator: &A,
    b: &[T],
    config: &IdrsConfig<T::Real>,
    residual_replacement_interval: usize,
) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_idrs_with_residual_replacement(operator, b, config, residual_replacement_interval) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b.len()),
    }
}

/// Solve Ax = b using preconditioned IDR(s)
///
/// Uses left preconditioning: M^{-1} A x = M^{-1} b
pub fn idrs_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsConfig<T::Real>,
) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_idrs_preconditioned(operator, precond, b, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b.len()),
    }
}

/// Solve Ax = b using preconditioned IDR(s) with a configurable
/// residual-replacement interval in matrix-vector products (0 disables).
pub fn idrs_preconditioned_with_residual_replacement<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsConfig<T::Real>,
    residual_replacement_interval: usize,
) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_idrs_preconditioned_with_residual_replacement(
        operator,
        precond,
        b,
        config,
        residual_replacement_interval,
    ) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b.len()),
    }
}

fn idrs_core<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsConfig<T::Real>,
    residual_replacement_interval: usize,
) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.len();
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
        || operator.num_rows() != n
        || operator.num_cols() != n
        || b.iter().any(|value| !value.norm().is_finite())
    {
        return numerical_error_solution(n);
    }
    let s = config.s;
    let mut x = vec![T::zero(); n];
    let mut op_counts = KrylovOpCounts::default();
    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    if s == 0 || s > n {
        return IdrsSolution {
            x,
            iterations: 0,
            residual: b_norm,
            estimated_residual: b_norm,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts,
        };
    }

    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        return IdrsSolution {
            x,
            iterations: 0,
            residual: T::Real::zero(),
            estimated_residual: T::Real::zero(),
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }

    // Initial residual
    let mut r = b.to_vec();
    let mut iterations = 0;

    // Breakdown threshold scaled with ||b||^2 (same eps²-based construction
    // as bicgstab.rs): dots against the shadow space scale like ||b||^2, so a
    // fixed absolute threshold either false-triggers on large right-hand
    // sides or never triggers on small ones.
    let eps = T::Real::epsilon();
    let scaled = b_norm * b_norm * eps * eps;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_tol = if scaled > floor { scaled } else { floor };
    // omega is a ratio (scale-invariant), so an absolute threshold applies
    // (mirrors the omega stagnation check in bicgstab.rs).
    let omega_tol = T::Real::from_f64(1e-20).unwrap();

    // Generate deterministic pseudo-random P (n x s)
    let p_columns = generate_shadow_space::<T>(n, s);

    // G and U arrays (n x s), stored as vectors of columns
    let mut g_cols: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();
    let mut u_cols: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();

    // M = P^T G (s x s matrix), tracks current state
    let mut m_matrix = sonium_solvers_core::dense::DenseMatrix::from_elem(s, s, T::zero());

    // omega scalar for residual minimization
    let mut omega = T::one();

    // Track whether G has been initialized
    let mut g_initialized = false;

    let max_outer = config.max_iterations / (s + 1) + 1;
    let mut next_replacement = residual_replacement_interval;

    // Pre-allocate work buffers (reused across all outer/inner iterations)
    let mut f: Vec<T> = vec![T::zero(); s];
    let mut v_buf = vec![T::zero(); n];
    let mut pv_buf = vec![T::zero(); n];
    let mut t_buf = vec![T::zero(); n];
    let mut small_aug = sonium_solvers_core::dense::DenseMatrix::from_elem(s, s + 1, T::zero());
    let mut c_coeffs = vec![T::zero(); s];

    for _outer in 0..max_outer {
        // Residual replacement: recompute the true residual r = b - A*x with
        // one SpMV. The G/U Krylov columns are kept (they do not depend on
        // r); f = P^T r and M = P^T G are refreshed below as usual, so the
        // recurrence continues consistently from the true residual.
        if residual_replacement_interval > 0 && iterations >= next_replacement {
            operator.apply_into(&x, &mut t_buf);
            op_counts.spmv_count += 1;
            op_counts.exact_residual_checks += 1;
            r.copy_from_slice(b);
            axpy(-T::one(), &t_buf, &mut r);
            op_counts.axpy_passes += 1;
            next_replacement += residual_replacement_interval;

            let rel_residual = vector_norm(&r) / b_norm;
            op_counts.norm_count += 1;
            if rel_residual < config.tolerance {
                return IdrsSolution {
                    x,
                    iterations,
                    residual: rel_residual,
                    estimated_residual: rel_residual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
        }

        // f = P^T * r (zero allocation via indexed writes)
        for (fi, p) in f.iter_mut().zip(p_columns.iter()) {
            *fi = inner_product(p, &r);
        }
        op_counts.dot_reductions += s;
        op_counts.synchronization_points += s;
        if g_initialized {
            refresh_shadow_matrix(&p_columns, &g_cols, s, &mut m_matrix);
            op_counts.dot_reductions += s * s;
            op_counts.synchronization_points += s * s;
        }

        for k in 0..s {
            // Compute v = r - G*c where c solves (P^T G) c = f
            if g_initialized {
                if !solve_small_system_into(
                    &m_matrix,
                    &f,
                    s,
                    breakdown_tol,
                    &mut small_aug,
                    &mut c_coeffs,
                ) {
                    let residual = relative_residual(&r, b_norm, &mut op_counts);
                    return IdrsSolution {
                        x,
                        iterations,
                        residual,
                        estimated_residual: residual,
                        backward_error: T::Real::infinity(),
                        attainable_accuracy: AccuracyStatus::NotChecked,
                        converged: false,
                        status: SolverStatus::Breakdown,
                        op_counts,
                    };
                }

                // v_buf = r - G * c
                v_buf.copy_from_slice(&r);
                op_counts.axpy_passes += 1;
                for j in 0..s {
                    let cj = c_coeffs[j];
                    for idx in 0..n {
                        v_buf[idx] -= cj * g_cols[j][idx];
                    }
                }
                op_counts.axpy_passes += s;

                // u_cols[k] = U * c + omega * M^{-1} v
                precond.apply_into(&v_buf, &mut pv_buf);
                op_counts.preconditioner_count += 1;
                // Accumulate U*c into pv_buf (which already holds omega*M^{-1}v after scaling)
                // First compute u_new = sum(c[j]*u_cols[j]) + omega*pv into t_buf, then swap into u_cols[k]
                for idx in 0..n {
                    t_buf[idx] = omega * pv_buf[idx];
                }
                op_counts.axpy_passes += 1;
                for j in 0..s {
                    if j == k {
                        continue; // skip self (u_cols[k] is being overwritten)
                    }
                    let cj = c_coeffs[j];
                    for idx in 0..n {
                        t_buf[idx] += cj * u_cols[j][idx];
                    }
                }
                op_counts.axpy_passes += s.saturating_sub(1);
                // Also add c[k]*u_cols[k] (the old value before overwrite)
                if k < s {
                    let ck = c_coeffs[k];
                    for idx in 0..n {
                        t_buf[idx] += ck * u_cols[k][idx];
                    }
                    op_counts.axpy_passes += 1;
                }
                u_cols[k].copy_from_slice(&t_buf);
                op_counts.axpy_passes += 1;
            } else {
                // G is zero — just use v = r, u_new = omega * M^{-1} r
                precond.apply_into(&r, &mut pv_buf);
                op_counts.preconditioner_count += 1;
                for idx in 0..n {
                    u_cols[k][idx] = omega * pv_buf[idx];
                }
                op_counts.axpy_passes += 1;
            };

            // g_cols[k] = A * u_k (zero allocation via apply_into)
            operator.apply_into(&u_cols[k], &mut g_cols[k]);
            op_counts.spmv_count += 1;
            iterations += 1;

            // Bi-orthogonalize g_cols[k] against previous columns of G
            for i in 0..k {
                let pg = inner_product(&p_columns[i], &g_cols[k]);
                let pp = inner_product(&p_columns[i], &g_cols[i]);
                op_counts.dot_reductions += 2;
                op_counts.synchronization_points += 2;
                if pp.norm() < breakdown_tol {
                    continue;
                }
                let alpha = pg * pp.inv();
                for idx in 0..n {
                    let gi = g_cols[i][idx];
                    let ui = u_cols[i][idx];
                    g_cols[k][idx] -= alpha * gi;
                    u_cols[k][idx] -= alpha * ui;
                }
                op_counts.axpy_passes += 2;
            }

            // Refresh the changed shadow-matrix column once; subsequent solves and
            // f-updates reuse it instead of recomputing all P^T G entries.
            for i in 0..s {
                m_matrix[[i, k]] = inner_product(&p_columns[i], &g_cols[k]);
            }
            op_counts.dot_reductions += s;
            op_counts.synchronization_points += s;

            // beta = p_k^T g_k
            let beta = m_matrix[[k, k]];
            if beta.norm() < breakdown_tol {
                let residual = relative_residual(&r, b_norm, &mut op_counts);
                return IdrsSolution {
                    x,
                    iterations,
                    residual,
                    estimated_residual: residual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: false,
                    status: SolverStatus::Breakdown,
                    op_counts,
                };
            }

            // alpha = f[k] / beta
            let alpha = f[k] * beta.inv();

            // Update x and r
            for idx in 0..n {
                x[idx] += alpha * u_cols[k][idx];
                r[idx] -= alpha * g_cols[k][idx];
            }
            op_counts.axpy_passes += 2;

            // Update f for remaining entries
            for i in (k + 1)..s {
                let pig = m_matrix[[i, k]];
                f[i] -= alpha * pig;
            }

            let rel_residual = vector_norm(&r) / b_norm;
            op_counts.norm_count += 1;

            if config.print_interval > 0 && iterations % config.print_interval == 0 {
                log::info!(
                    "IDR({}) iteration {}: relative residual = {:.6e}",
                    s,
                    iterations,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }

            if rel_residual < config.tolerance {
                return IdrsSolution {
                    x,
                    iterations,
                    residual: rel_residual,
                    estimated_residual: rel_residual,
                    backward_error: T::Real::infinity(),
                    attainable_accuracy: AccuracyStatus::NotChecked,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
        }

        g_initialized = true;

        // Residual minimization step (zero allocation via apply_into)
        precond.apply_into(&r, &mut v_buf);
        op_counts.preconditioner_count += 1;
        operator.apply_into(&v_buf, &mut t_buf);
        op_counts.spmv_count += 1;
        iterations += 1;

        let tt = inner_product(&t_buf, &t_buf);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if tt.norm() < breakdown_tol {
            let residual = relative_residual(&r, b_norm, &mut op_counts);
            return IdrsSolution {
                x,
                iterations,
                residual,
                estimated_residual: residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }
        omega = inner_product(&t_buf, &r) * tt.inv();
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;

        if omega.norm() < omega_tol {
            let residual = relative_residual(&r, b_norm, &mut op_counts);
            return IdrsSolution {
                x,
                iterations,
                residual,
                estimated_residual: residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Stagnated,
                op_counts,
            };
        }

        for idx in 0..n {
            x[idx] += omega * v_buf[idx];
            r[idx] -= omega * t_buf[idx];
        }
        op_counts.axpy_passes += 2;

        let rel_residual = vector_norm(&r) / b_norm;
        op_counts.norm_count += 1;

        if config.print_interval > 0 && iterations % config.print_interval == 0 {
            log::info!(
                "IDR({}) iteration {}: relative residual = {:.6e}",
                s,
                iterations,
                rel_residual.to_f64().unwrap_or(0.0)
            );
        }

        if rel_residual < config.tolerance {
            return IdrsSolution {
                x,
                iterations,
                residual: rel_residual,
                estimated_residual: rel_residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        if iterations >= config.max_iterations {
            break;
        }
    }

    let rel_residual = vector_norm(&r) / b_norm;
    op_counts.norm_count += 1;
    IdrsSolution {
        x,
        iterations,
        residual: rel_residual,
        estimated_residual: rel_residual,
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

fn relative_residual<T: ComplexField>(
    r: &[T],
    b_norm: T::Real,
    op_counts: &mut KrylovOpCounts,
) -> T::Real {
    op_counts.norm_count += 1;
    vector_norm(r) / b_norm
}

/// Generate deterministic pseudo-random shadow space P (n x s)
///
/// Uses a PCG-style LCG to generate values in (-1, 1), avoiding zero.
/// Each column uses a different seed for independence.
///
/// For complex scalar types the shadow vectors carry nonzero imaginary
/// parts: a purely real shadow space can be (nearly) orthogonal to the
/// residual subspaces of a genuinely complex non-symmetric operator
/// (e.g. Helmholtz/BEM), which stalls or breaks down the bi-orthogonal
/// projections. Real scalar types keep the original real-valued stream
/// bit-for-bit.
pub(crate) fn generate_shadow_space<T: ComplexField>(n: usize, s: usize) -> Vec<Vec<T>> {
    // Probe whether T can represent a nonzero imaginary part; for real
    // fields `from_re_im` discards the imaginary component.
    let one = T::Real::one();
    let is_complex = T::from_re_im(one, one).im() != T::Real::zero();

    let mut columns = Vec::with_capacity(s);

    for col in 0..s {
        let mut state: u64 = 6364136223846793005u64
            .wrapping_mul(col as u64 + 1)
            .wrapping_add(1442695040888963407);

        let values: Vec<T> = (0..n)
            .map(|_| {
                let re = next_lcg_value(&mut state);
                if is_complex {
                    let im = next_lcg_value(&mut state);
                    T::from_re_im(
                        T::Real::from_f64(re).unwrap(),
                        T::Real::from_f64(im).unwrap(),
                    )
                } else {
                    T::from_real(T::Real::from_f64(re).unwrap())
                }
            })
            .collect();

        columns.push(values);
    }

    columns
}

/// Advance the LCG one step and map the state to a value in (-1, 1),
/// nudged away from exact zero.
fn next_lcg_value(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Map to (-1, 1) using upper bits, avoiding exact zero
    let val = ((*state >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0;
    if val.abs() < 1e-10 { 0.5 } else { val }
}

fn refresh_shadow_matrix<T: ComplexField>(
    p_columns: &[Vec<T>],
    g_cols: &[Vec<T>],
    s: usize,
    m: &mut sonium_solvers_core::dense::DenseMatrix<T>,
) {
    for i in 0..s {
        for j in 0..s {
            m[[i, j]] = inner_product(&p_columns[i], &g_cols[j]);
        }
    }
}

/// Solve small s x s dense system M * c = f via Gaussian elimination with pivoting.
/// Returns `false` if the matrix is numerically singular (pivot below `tol`).
pub(crate) fn solve_small_system_into<T: ComplexField>(
    m: &sonium_solvers_core::dense::DenseMatrix<T>,
    f: &[T],
    s: usize,
    tol: T::Real,
    aug: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    c: &mut [T],
) -> bool {
    // Build augmented matrix [M | f]
    for i in 0..s {
        for j in 0..s {
            aug[[i, j]] = m[[i, j]];
        }
        aug[[i, s]] = f[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..s {
        // Find pivot
        let mut max_val = aug[[k, k]].norm();
        let mut max_row = k;
        for i in (k + 1)..s {
            let val = aug[[i, k]].norm();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..=s {
                let tmp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[k, k]];
        if pivot.norm() < tol {
            return false;
        }

        for i in (k + 1)..s {
            let mult = aug[[i, k]] * pivot.inv();
            aug[[i, k]] = T::zero();
            for j in (k + 1)..=s {
                let akj = aug[[k, j]];
                aug[[i, j]] -= mult * akj;
            }
        }
    }

    // Back substitution
    for i in (0..s).rev() {
        let mut sum = aug[[i, s]];
        for j in (i + 1)..s {
            sum -= aug[[i, j]] * c[j];
        }
        let diag = aug[[i, i]];
        if diag.norm() < tol {
            return false;
        }
        c[i] = sum * diag.inv();
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner};
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn idrs_max_iteration_reports_true_residual_and_backward_error() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, 2.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 1, -2.0),
                (2, 2, 5.0),
            ],
        );
        let rhs = [1.0, -2.0, 3.0];
        let config = IdrsConfig {
            s: 1,
            max_iterations: 1,
            tolerance: 1.0e-14,
            print_interval: 0,
        };
        let solution = idrs(&matrix, &rhs, &config);
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

    #[test]
    fn checked_idrs_contains_operator_panics() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("IDR(s) operator callback failed");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("IDR(s) operator callback failed");
            }
        }

        let config = IdrsConfig {
            s: 1,
            max_iterations: 10,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let error = try_idrs(&PanicOperator, &[1.0], &config)
            .expect_err("checked IDR(s) must type callback panics");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "IDR(s)",
                ..
            }
        ));

        let legacy = idrs(&PanicOperator, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    #[test]
    fn checked_idrs_rejects_non_finite_generated_values() {
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

        let config = IdrsConfig {
            s: 1,
            max_iterations: 10,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let error = try_idrs(&NonFiniteOperator, &[1.0], &config)
            .expect_err("checked IDR(s) must reject non-finite operator output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "IDR(s) operator",
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
        let error = try_idrs_preconditioned(&operator, &precond, &[1.0], &config)
            .expect_err("checked IDR(s) must reject non-finite preconditioner output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "IDR(s) preconditioner",
                ..
            }
        ));

        let legacy = idrs_preconditioned(&operator, &precond, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    #[test]
    fn idrs_rejects_non_finite_input_before_operator_use() {
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

        let config = IdrsConfig::default();
        let solution = idrs(&PanicOperator, &[f64::NAN], &config);
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn test_idrs_simple_spd() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = IdrsConfig {
            s: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(solution.converged, "IDR(s) should converge for SPD system");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "Solution should satisfy Ax = b, got error {}",
            error
        );
    }

    #[test]
    fn test_idrs_nonsymmetric() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = IdrsConfig {
            s: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(
            solution.converged,
            "IDR(s) should converge for non-symmetric"
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
    fn test_idrs_complex() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 1.0), Complex64::new(1.0, 0.5)],
            [Complex64::new(1.0, -0.5), Complex64::new(3.0, -1.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        let config = IdrsConfig {
            s: 1,
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(
            solution.converged,
            "IDR(s) should converge for complex system"
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
    fn test_idrs_identity() {
        let n = 5;
        let id: CsrMatrix<Complex64> = CsrMatrix::identity(n);
        let b = (1..=n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();

        let config = IdrsConfig {
            s: 2,
            max_iterations: 50,
            tolerance: 1e-12,
            print_interval: 0,
        };

        let solution = idrs(&id, &b, &config);
        assert!(solution.converged);

        let error: f64 = sonium_solvers_core::vec_ops::vec_sub(&solution.x, &b)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-10);
    }

    #[test]
    fn test_idrs_f64() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        let config = IdrsConfig {
            s: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(solution.converged);

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8);
    }

    #[test]
    fn test_idrs_larger_tridiagonal() {
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

        let config = IdrsConfig {
            s: 4,
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(
            solution.converged,
            "IDR(s) should converge on tridiagonal system"
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
    fn test_idrs_s1_bicgstab_like() {
        // s=1 should behave like BiCGSTAB
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = IdrsConfig {
            s: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(solution.converged, "IDR(1) should converge");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8);
    }

    #[test]
    fn test_idrs_reports_krylov_op_counts() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let config = IdrsConfig {
            s: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);

        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.preconditioner_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions >= solution.iterations);
        assert!(solution.op_counts.norm_count > 0);
        assert!(solution.op_counts.axpy_passes >= solution.iterations);
    }

    #[test]
    fn test_idrs_zero_rhs() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

        let config = IdrsConfig {
            s: 1,
            ..Default::default()
        };
        let solution = idrs(&a, &b, &config);

        assert!(solution.converged);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn test_shadow_space_complex_for_complex_type() {
        // Complex scalar types must get a shadow space with nonzero
        // imaginary parts; a purely real P can be nearly orthogonal to the
        // residual subspaces of a genuinely complex operator and stall.
        let p = generate_shadow_space::<Complex64>(16, 4);
        assert_eq!(p.len(), 4);
        for col in &p {
            assert_eq!(col.len(), 16);
            let max_im = col.iter().map(|v| v.im.abs()).fold(0.0_f64, f64::max);
            assert!(
                max_im > 1e-6,
                "complex shadow space must have nonzero imaginary parts"
            );
            let max_re = col.iter().map(|v| v.re.abs()).fold(0.0_f64, f64::max);
            assert!(max_re > 1e-6, "shadow space must be non-degenerate");
        }

        // Real scalar types keep the real-valued shadow space.
        let p_real = generate_shadow_space::<f64>(16, 4);
        for col in &p_real {
            assert!(col.iter().all(|v| v.abs() <= 1.0 && *v != 0.0));
        }
    }

    #[test]
    fn test_idrs_complex_helmholtz_like() {
        // 1D Helmholtz-like operator: shifted Laplacian with a complex
        // wavenumber (damping) plus a first-derivative (advection) term to
        // make it genuinely non-symmetric — the class of system where a
        // real shadow space can stall.
        let n = 40;
        let k2 = Complex64::new(8.0, -2.0); // complex wavenumber² (lossy)
        let adv = Complex64::new(0.5, 0.3);
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(2.0, 0.0) + k2;
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0) - adv;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0) + adv;
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();

        let config = IdrsConfig {
            s: 4,
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(
            solution.converged,
            "IDR(4) should converge on complex Helmholtz-like system, residual = {}",
            solution.residual
        );
        // Well below the iteration budget: guards against stalling.
        assert!(
            solution.iterations <= 250,
            "IDR(4) took {} iterations (possible stall)",
            solution.iterations
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        assert!(error / b_norm < 1e-8);
    }

    #[test]
    fn test_idrs_invalid_s_returns_breakdown() {
        // Regression test: s > n produces a rank-deficient shadow system.
        // With s=4 on a 2x2 system, P is 2×4 (rank ≤ 2), so the s×s system
        // M = P^T * G is rank-deficient and solve_small_system returns garbage.
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = IdrsConfig {
            s: 4,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(!solution.converged);
        assert_eq!(solution.status, SolverStatus::Breakdown);
        assert_eq!(solution.iterations, 0);

        let zero_s = IdrsConfig { s: 0, ..config };
        let solution = idrs(&a, &b, &zero_s);
        assert!(!solution.converged);
        assert_eq!(solution.status, SolverStatus::Breakdown);
    }

    #[test]
    fn test_idrs_residual_replacement_accuracy() {
        // Attainable-accuracy regression: an ill-conditioned complex
        // non-Hermitian system solved far below 1e-10 relative. The purely
        // recursive residual update drifts from the true residual over a long
        // solve and caps the attainable accuracy; with periodic true-residual
        // replacement the true residual must actually reach the tolerance.
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

        let config = IdrsConfig {
            s: 4,
            max_iterations: 2000,
            tolerance: 1e-12,
            print_interval: 0,
        };

        let solution = idrs(&a, &b, &config);
        assert!(
            solution.converged,
            "IDR(4) with residual replacement should converge, residual = {}",
            solution.residual
        );
        assert!(
            solution.iterations > 50,
            "test must exercise the residual-replacement path (iterations = {})",
            solution.iterations
        );
        assert!(
            solution.op_counts.exact_residual_checks > 0,
            "residual replacement must have recomputed the true residual"
        );

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
}
#[cfg(test)]
mod checked_result_tests {
    use super::*;

    #[test]
    fn checked_result_rejects_nonfinite_estimates_and_backward_error() {
        let solution = IdrsSolution {
            x: vec![0.0_f64],
            iterations: 1,
            residual: 0.0,
            estimated_residual: f64::INFINITY,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Stagnated,
            op_counts: KrylovOpCounts::default(),
        };
        let error = finish_checked_idrs_solution(solution, 1, false, false)
            .expect_err("non-finite IDR(s) result metrics must be rejected");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "IDR(s)",
                ..
            }
        ));
    }
}
