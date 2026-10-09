//! TFQMR (Transpose-Free Quasi-Minimal Residual) solver
//!
//! Freund's TFQMR (SIAM J. Sci. Comput., 1993) is the quasi-minimal-residual
//! smoothing of CGS: it runs the same two-term CGS/bi-Lanczos recurrence
//! (shadow residual `r0`, unconjugated dots, no `A^T`) but replaces the CGS
//! iterate with a quasi-minimizing update built from the search direction
//! `d`, the scalars `theta` (ratio of successive quasi-residuals) and `tau`
//! (the quasi-residual norm). The result converges about as fast as CGS per
//! matrix-vector product but with a smooth, monotone-looking residual curve
//! instead of CGS/BiCGSTAB's erratic one.
//!
//! Algorithm structure follows the standard reference formulation (PETSc /
//! SLATEC): vectors `u`, `v`, `w`, `y`, `d`, scalars `rho`, `sigma`, `alpha`,
//! `beta`, `eta`, `theta`, `tau`. Each outer ("double") step costs two
//! operator applications and — in the preconditioned variant — two
//! preconditioner applications, the same as BiCGSTAB.
//!
//! Convergence is decided on Freund's quasi-residual estimate
//! `||r_m|| <= tau * sqrt(m + 1)` (relative to `||b||`), and every exit path
//! recomputes the *true* residual `||b - A x|| / ||b||` so the reported
//! residual is honest.
//!
//! Complex-safety conventions mirror [`super::bicgstab`]: bi-Lanczos inner
//! products use the *unconjugated* dot (correct for complex-symmetric
//! Helmholtz-type systems), breakdown thresholds scale with `||b||^2 * eps^2`,
//! and every [`RESIDUAL_REPLACEMENT_INTERVAL`] outer steps the true residual
//! is recomputed and the recurrence restarted from it to curb rounding-error
//! build-up (Sleijpen & van der Vorst).

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, dot_unconjugated, scale_inplace, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use super::cgs_orth::KrylovOpCounts;

/// Interval (in outer steps) at which the true residual `r = b - A*x` is
/// recomputed and the bi-Lanczos recurrence restarted from it (residual
/// replacement), matching the convention in [`super::bicgstab`].
const RESIDUAL_REPLACEMENT_INTERVAL: usize = 50;

/// TFQMR solver configuration
#[derive(Debug, Clone)]
pub struct TfqmrConfig<R> {
    /// Maximum number of outer iterations (each costs 2 operator applications)
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
}

impl Default for TfqmrConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        }
    }
}

/// TFQMR solver result
#[derive(Debug)]
pub struct TfqmrSolution<T: ComplexField> {
    /// Solution vector
    pub x: Vec<T>,
    /// Number of outer iterations (each performs 2 operator applications)
    pub iterations: usize,
    /// Final *true* relative residual `||b - A x|| / ||b||`
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

fn numerical_error_solution<T: ComplexField>(x: Vec<T>) -> TfqmrSolution<T> {
    TfqmrSolution {
        x,
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

fn compatibility_error_solution<T: ComplexField>(b: &[T], x0: Option<&[T]>) -> TfqmrSolution<T> {
    numerical_error_solution(match x0 {
        Some(guess) if guess.len() == b.len() => guess.to_vec(),
        _ => vec![T::zero(); b.len()],
    })
}

fn invalid_tfqmr_input<T: ComplexField, A: LinearOperator<T>>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &TfqmrConfig<T::Real>,
) -> Option<TfqmrSolution<T>> {
    let n = b.len();
    let invalid = config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
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

/// Checked TFQMR entry point using the identity preconditioner.
pub fn try_tfqmr<T, A>(
    operator: &A,
    b: &[T],
    config: &TfqmrConfig<T::Real>,
) -> Result<TfqmrSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    try_tfqmr_checked(operator, &identity, false, b, None, config)
}

/// Checked preconditioned TFQMR entry point.
pub fn try_tfqmr_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &TfqmrConfig<T::Real>,
) -> Result<TfqmrSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_tfqmr_checked(operator, precond, true, b, None, config)
}

/// Checked preconditioned TFQMR entry point with an optional initial guess.
pub fn try_tfqmr_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &TfqmrConfig<T::Real>,
) -> Result<TfqmrSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_tfqmr_checked(operator, precond, true, b, x0, config)
}

fn try_tfqmr_checked<T, A, P>(
    operator: &A,
    precond: &P,
    with_precond: bool,
    b: &[T],
    x0: Option<&[T]>,
    config: &TfqmrConfig<T::Real>,
) -> Result<TfqmrSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let invalid_preconditioner_output = AtomicBool::new(false);
    let checked_operator = CheckedTfqmrOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let checked_precond = CheckedTfqmrPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner_output,
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_tfqmr_inputs(operator, b, x0, config)?;
        let solution = tfqmr_core(
            &checked_operator,
            &checked_precond,
            with_precond,
            b,
            x0,
            config,
        );
        let solution = finalize_tfqmr_solution(&checked_operator, b, config.tolerance, solution);
        finish_checked_tfqmr_solution(
            solution,
            b.len(),
            invalid_operator_output.load(Ordering::Relaxed),
            invalid_preconditioner_output.load(Ordering::Relaxed),
        )
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "TFQMR",
        reason: panic_reason(payload),
    })??;

    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn validate_tfqmr_inputs<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &TfqmrConfig<T::Real>,
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
            operation: "TFQMR",
            reason: "invalid TFQMR configuration".to_string(),
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
            operation: "TFQMR",
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
            operation: "TFQMR",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn finish_checked_tfqmr_solution<T: ComplexField>(
    solution: TfqmrSolution<T>,
    expected_dimension: usize,
    invalid_operator_output: bool,
    invalid_preconditioner_output: bool,
) -> Result<TfqmrSolution<T>, SolverError> {
    if invalid_operator_output {
        return Err(SolverError::OperationFailed {
            operation: "TFQMR operator",
            reason: "operator generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if invalid_preconditioner_output {
        return Err(SolverError::OperationFailed {
            operation: "TFQMR preconditioner",
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
            operation: "TFQMR",
            reason: "generated NaN, infinity, or an invalid solution".to_string(),
        });
    }
    Ok(solution)
}

fn finalize_tfqmr_solution<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut result: TfqmrSolution<T>,
) -> TfqmrSolution<T>
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
    // Accumulate with hypot rather than sum(norm_sqr). The latter turns
    // perfectly finite vectors near the scalar limit into an infinite
    // metric before the checked result boundary can classify them.
    let mut residual_norm = T::Real::zero();
    let mut ax_norm = T::Real::zero();
    let mut rhs_norm = T::Real::zero();
    for (rhs, applied) in b.iter().zip(&mut ax) {
        let action = *applied;
        let action_entry_norm = action.norm();
        let rhs_entry_norm = rhs.norm();
        if !action_entry_norm.is_finite() || !rhs_entry_norm.is_finite() {
            result.residual = T::Real::infinity();
            result.backward_error = T::Real::infinity();
            result.attainable_accuracy = AccuracyStatus::NotChecked;
            result.converged = false;
            result.status = SolverStatus::NumericalError;
            return result;
        }
        *applied = *rhs - *applied;
        let residual_entry_norm = applied.norm();
        if !residual_entry_norm.is_finite() {
            result.residual = T::Real::infinity();
            result.backward_error = T::Real::infinity();
            result.attainable_accuracy = AccuracyStatus::NotChecked;
            result.converged = false;
            result.status = SolverStatus::NumericalError;
            return result;
        }
        residual_norm = residual_norm.hypot(residual_entry_norm);
        ax_norm = ax_norm.hypot(action_entry_norm);
        rhs_norm = rhs_norm.hypot(rhs_entry_norm);
    }
    if !residual_norm.is_finite() || !ax_norm.is_finite() || !rhs_norm.is_finite() {
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        result.converged = false;
        result.status = SolverStatus::NumericalError;
        return result;
    }
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
        "TFQMR callback panicked".to_string()
    }
}

struct CheckedTfqmrOperator<'a, A> {
    inner: &'a A,
    invalid_output: &'a AtomicBool,
}

impl<T, A> LinearOperator<T> for CheckedTfqmrOperator<'_, A>
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
        self.validate_adjoint(output);
    }

    fn apply_hermitian(&self, input: &[T]) -> Vec<T> {
        let mut output = self.inner.apply_hermitian(input);
        self.validate_adjoint(&mut output);
        output
    }
}

impl<A> CheckedTfqmrOperator<'_, A> {
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

    fn validate_adjoint<T>(&self, output: &mut [T])
    where
        T: ComplexField,
        A: LinearOperator<T>,
    {
        if output.len() != self.inner.num_cols()
            || output.iter().any(|value| !value.norm().is_finite())
        {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

struct CheckedTfqmrPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for CheckedTfqmrPreconditioner<'_, P>
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

impl<P> CheckedTfqmrPreconditioner<'_, P> {
    fn validate<T: ComplexField>(&self, input: &[T], output: &mut [T]) {
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

/// Solve `Ax = b` using TFQMR (no preconditioning).
pub fn tfqmr<T, A>(operator: &A, b: &[T], config: &TfqmrConfig<T::Real>) -> TfqmrSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    match try_tfqmr_checked(operator, &identity, false, b, None, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b, None),
    }
}

/// Solve `Ax = b` using TFQMR with preconditioning.
///
/// Follows the same convention as [`super::bicgstab::bicgstab_preconditioned`]:
/// every direction fed to the operator is passed through the preconditioner
/// first, and the solution is updated with the preconditioned directions, so
/// all residual vectors stay in the original (unpreconditioned) space.
pub fn tfqmr_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &TfqmrConfig<T::Real>,
) -> TfqmrSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_tfqmr_checked(operator, precond, true, b, None, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b, None),
    }
}

/// Solve `Ax = b` using TFQMR with preconditioning and an initial guess.
pub fn tfqmr_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &TfqmrConfig<T::Real>,
) -> TfqmrSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_tfqmr_checked(operator, precond, true, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => compatibility_error_solution(b, x0),
    }
}

/// Core TFQMR loop (see module docs for the algorithm).
///
/// The recurrence follows the PETSc/SLATEC formulation in the preconditioned
/// ("y") space: the method runs on `Ã = A·M⁻¹` with the accumulated solution
/// `y` unwound as `x = x_base + M⁻¹·y` whenever the true residual is needed.
/// This matches [`super::bicgstab`]'s convention that every direction fed to
/// the operator passes through the preconditioner first, and keeps the
/// residual `r` in the original space at all times.
///
/// `with_precond` distinguishes the genuinely preconditioned path from the
/// identity path so operation counts stay honest for the unpreconditioned
/// entry point.
#[allow(clippy::too_many_lines)]
fn tfqmr_core<T, A, P>(
    operator: &A,
    precond: &P,
    with_precond: bool,
    b: &[T],
    x0: Option<&[T]>,
    config: &TfqmrConfig<T::Real>,
) -> TfqmrSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if let Some(invalid) = invalid_tfqmr_input(operator, b, x0, config) {
        return invalid;
    }

    let n = b.len();
    let mut op_counts = KrylovOpCounts::default();

    // Solution base: x = x_base + M^{-1} y. x_base starts at the initial
    // guess and is advanced to the current iterate at every residual
    // replacement (which restarts the recurrence with y = 0).
    let mut x_base = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut y = vec![T::zero(); n]; // accumulated solution in Ã-space
    let mut x = vec![T::zero(); n]; // assembled x_base + M^{-1} y

    let mut b_norm = vector_norm(b);
    op_counts.norm_count += 1;

    // True initial residual r = b - A*x0.
    let mut r = vec![T::zero(); n];
    let mut hat = vec![T::zero(); n]; // scratch for M^{-1}(.) outputs
    if x0.is_some() {
        operator.apply_into(&x_base, &mut r);
        op_counts.spmv_count += 1;
        for k in 0..n {
            r[k] = b[k] - r[k];
        }
    } else {
        r.copy_from_slice(b);
    }
    let initial_residual = vector_norm(&r);
    op_counts.norm_count += 1;

    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        if initial_residual < tol_threshold {
            return TfqmrSolution {
                x: x_base,
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
        // Relative residual is undefined for a zero RHS; normalize by one so
        // the remaining checks become absolute-residual checks.
        b_norm = T::Real::one();
    }
    if initial_residual / b_norm < config.tolerance {
        return TfqmrSolution {
            x: x_base,
            iterations: 0,
            residual: initial_residual / b_norm,
            estimated_residual: initial_residual / b_norm,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }

    // eps²-scaled breakdown threshold, same convention as bicgstab.
    let eps = T::Real::epsilon();
    let scaled = b_norm * b_norm * eps * eps;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_threshold = if scaled > floor { scaled } else { floor };

    // Shadow residual (bi-Lanczos left vector), fixed between restarts.
    let mut r0 = r.clone();

    // CGS directions in Ã-space: u, q, p. v = Ã p lives in residual space.
    let mut u = r.clone();
    let mut q = vec![T::zero(); n];
    let mut p = r.clone();
    let mut v = vec![T::zero(); n];
    precond.apply_into(&p, &mut hat);
    if with_precond {
        op_counts.preconditioner_count += 1;
    }
    operator.apply_into(&hat, &mut v);
    op_counts.spmv_count += 1;

    let mut d = vec![T::zero(); n]; // quasi-minimizing direction (Ã-space)
    let mut t = vec![T::zero(); n]; // scratch: u + q
    let mut auq = vec![T::zero(); n]; // Ã (u + q), residual space
    let mut scratch = vec![T::zero(); n]; // operator-output scratch

    let mut tau = initial_residual; // quasi-residual norm
    let mut dpold = initial_residual; // ||r|| of the previous outer step
    let mut theta_old = T::Real::zero(); // theta from the previous half-step
    let mut eta_old = T::zero(); // eta from the previous half-step
    let mut rho_old = dot_unconjugated(&r0, &r);
    op_counts.dot_reductions += 1;
    op_counts.synchronization_points += 1;
    if rho_old.norm() < breakdown_threshold {
        x_base.clone_into(&mut x);
        let residual =
            true_relative_residual(operator, &x, b, b_norm, &mut scratch, &mut op_counts);
        return TfqmrSolution {
            x,
            iterations: 0,
            residual,
            estimated_residual: residual,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts,
        };
    }

    for iter in 0..config.max_iterations {
        // Residual replacement: fold y into the base, recompute the true
        // residual r = b - A*x, and restart the recurrence from it.
        if iter > 0 && iter % RESIDUAL_REPLACEMENT_INTERVAL == 0 {
            assemble_x(
                precond,
                with_precond,
                &x_base,
                &y,
                &mut x,
                &mut hat,
                &mut op_counts,
            );
            x_base.clone_from(&x);
            y.fill(T::zero());
            operator.apply_into(&x_base, &mut r);
            op_counts.spmv_count += 1;
            for k in 0..n {
                r[k] = b[k] - r[k];
            }
            r0.copy_from_slice(&r);
            u.copy_from_slice(&r);
            p.copy_from_slice(&r);
            precond.apply_into(&p, &mut hat);
            if with_precond {
                op_counts.preconditioner_count += 1;
            }
            operator.apply_into(&hat, &mut v);
            op_counts.spmv_count += 1;
            d.fill(T::zero());
            tau = vector_norm(&r);
            op_counts.norm_count += 1;
            dpold = tau;
            theta_old = T::Real::zero();
            eta_old = T::zero();
            rho_old = dot_unconjugated(&r0, &r);
            op_counts.dot_reductions += 1;
            op_counts.synchronization_points += 1;
        }

        // sigma = (r0, v), alpha = rho / sigma
        let sigma = dot_unconjugated(&r0, &v);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if sigma.norm() < breakdown_threshold {
            assemble_x(
                precond,
                with_precond,
                &x_base,
                &y,
                &mut x,
                &mut hat,
                &mut op_counts,
            );
            let residual =
                true_relative_residual(operator, &x, b, b_norm, &mut scratch, &mut op_counts);
            return TfqmrSolution {
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
        let alpha = rho_old / sigma;

        // q = u - alpha * v  (Ã-space direction for the even half-step)
        q.copy_from_slice(&u);
        axpy(-alpha, &v, &mut q);
        op_counts.axpy_passes += 1;

        // Single combined residual update for both half-steps:
        //   r = r - alpha * Ã (u + q)
        t.copy_from_slice(&u);
        axpy(T::one(), &q, &mut t);
        op_counts.axpy_passes += 1;
        precond.apply_into(&t, &mut hat);
        if with_precond {
            op_counts.preconditioner_count += 1;
        }
        operator.apply_into(&hat, &mut auq);
        op_counts.spmv_count += 1;
        axpy(-alpha, &auq, &mut r);
        op_counts.axpy_passes += 1;

        let dp = vector_norm(&r);
        op_counts.norm_count += 1;

        // Two quasi-minimization half-steps per outer iteration. The odd
        // half-step's residual norm is estimated by the geometric mean
        // sqrt(dp * dpold) (the intermediate CGS residual is unavailable
        // without a second operator application).
        for half in 0..2 {
            let w = if half == 0 { (dp * dpold).sqrt() } else { dp };
            let theta = w / tau;
            let c = T::Real::one() / (T::Real::one() + theta * theta).sqrt();
            tau = tau * theta * c;
            let eta = T::from_real(c * c) * alpha;

            // d = y_k + (theta_old^2 * eta_old / alpha) * d
            let coef = eta_old * T::from_real(theta_old * theta_old) / alpha;
            scale_inplace(&mut d, coef);
            axpy(T::one(), if half == 0 { &u } else { &q }, &mut d);
            op_counts.axpy_passes += 2;

            // y += eta * d
            axpy(eta, &d, &mut y);
            op_counts.axpy_passes += 1;

            // Freund's quasi-residual bound ||r_m|| <= tau * sqrt(m + 1).
            let half_count = 2 * iter + half + 1;
            let quasi_estimate = tau * T::Real::from_usize(half_count + 1).unwrap().sqrt() / b_norm;

            if config.print_interval > 0 && half_count % config.print_interval == 0 {
                log::info!(
                    "TFQMR half-iteration {} (outer {}): quasi-residual estimate = {:.6e}",
                    half_count,
                    iter + 1,
                    quasi_estimate.to_f64().unwrap_or(0.0)
                );
            }

            if quasi_estimate < config.tolerance {
                // Honest exit: verify with the true residual. If the estimate
                // was fooled, keep iterating — the recurrence state is intact.
                assemble_x(
                    precond,
                    with_precond,
                    &x_base,
                    &y,
                    &mut x,
                    &mut hat,
                    &mut op_counts,
                );
                let actual =
                    true_relative_residual(operator, &x, b, b_norm, &mut scratch, &mut op_counts);
                if actual < config.tolerance {
                    return TfqmrSolution {
                        x,
                        iterations: iter + 1,
                        residual: actual,
                        estimated_residual: actual,
                        backward_error: T::Real::infinity(),
                        attainable_accuracy: AccuracyStatus::NotChecked,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    };
                }
            }

            theta_old = theta;
            eta_old = eta;
        }

        // Even-step bi-Lanczos update:
        //   rho_new = (r0, r), beta = rho_new / rho_old
        //   u = r + beta * q
        //   q = q + beta * p
        //   p = u + beta * q   (= u + beta * (q_old + beta * p_old))
        //   v = Ã p
        let rho_new = dot_unconjugated(&r0, &r);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if rho_new.norm() < breakdown_threshold {
            assemble_x(
                precond,
                with_precond,
                &x_base,
                &y,
                &mut x,
                &mut hat,
                &mut op_counts,
            );
            let residual =
                true_relative_residual(operator, &x, b, b_norm, &mut scratch, &mut op_counts);
            return TfqmrSolution {
                x,
                iterations: iter + 1,
                residual,
                estimated_residual: residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }
        let beta = rho_new / rho_old;
        rho_old = rho_new;

        u.copy_from_slice(&r);
        axpy(beta, &q, &mut u);
        op_counts.axpy_passes += 1;
        axpy(beta, &p, &mut q);
        op_counts.axpy_passes += 1;
        p.copy_from_slice(&u);
        axpy(beta, &q, &mut p);
        op_counts.axpy_passes += 1;

        precond.apply_into(&p, &mut hat);
        if with_precond {
            op_counts.preconditioner_count += 1;
        }
        operator.apply_into(&hat, &mut v);
        op_counts.spmv_count += 1;

        dpold = dp;
    }

    assemble_x(
        precond,
        with_precond,
        &x_base,
        &y,
        &mut x,
        &mut hat,
        &mut op_counts,
    );
    let residual = true_relative_residual(operator, &x, b, b_norm, &mut scratch, &mut op_counts);
    TfqmrSolution {
        x,
        iterations: config.max_iterations,
        residual,
        estimated_residual: residual,
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

/// Assemble `x_out = x_base + M^{-1} y` (unwind the Ã-space solution).
fn assemble_x<T, P>(
    precond: &P,
    with_precond: bool,
    x_base: &[T],
    y: &[T],
    x_out: &mut [T],
    hat: &mut [T],
    op_counts: &mut KrylovOpCounts,
) where
    T: ComplexField,
    P: Preconditioner<T>,
{
    precond.apply_into(y, hat);
    if with_precond {
        op_counts.preconditioner_count += 1;
    }
    x_out.copy_from_slice(x_base);
    axpy(T::one(), hat, x_out);
    op_counts.axpy_passes += 1;
}

/// Compute `||b - A x|| / b_norm` into `scratch`, tracking operation counts.
fn true_relative_residual<T, A>(
    operator: &A,
    x: &[T],
    b: &[T],
    b_norm: T::Real,
    scratch: &mut [T],
    op_counts: &mut KrylovOpCounts,
) -> T::Real
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    operator.apply_into(x, scratch);
    op_counts.spmv_count += 1;
    let mut norm_sqr = T::Real::zero();
    for (si, bi) in scratch.iter().zip(b.iter()) {
        norm_sqr += (*bi - *si).norm_sqr();
    }
    op_counts.norm_count += 1;
    op_counts.exact_residual_checks += 1;
    norm_sqr.sqrt() / b_norm
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BiCgstabConfig, bicgstab};
    use num_complex::Complex64;
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner};
    use sonium_solvers_preconditioners::{DiagonalPreconditioner, IluPreconditioner};
    use sonium_solvers_sparse::CsrMatrix;

    /// True relative residual `||b - A x|| / ||b||` for a complex system.
    fn true_rel_residual_complex(
        a: &CsrMatrix<Complex64>,
        x: &[Complex64],
        b: &[Complex64],
    ) -> f64 {
        let ax = a.matvec(x);
        let err: f64 = sonium_solvers_core::vec_ops::vec_sub(&ax, b)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        err / b_norm
    }

    /// Non-symmetric 3x3 system shared with the bicgstab tests.
    fn nonsymmetric_3x3() -> (CsrMatrix<Complex64>, Vec<Complex64>) {
        let dense = sonium_solvers_core::array![
            [
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(1.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(6.0, 0.0)
            ],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        (a, b)
    }

    #[test]
    fn checked_tfqmr_operator_preserves_exact_adjoint_contract() {
        let (operator, _) = nonsymmetric_3x3();
        let invalid_output = AtomicBool::new(false);
        let checked = CheckedTfqmrOperator {
            inner: &operator,
            invalid_output: &invalid_output,
        };
        assert_eq!(checked.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(checked.hermitian_capability(), AdjointCapability::Exact);

        let input = vec![
            Complex64::new(0.5, -0.25),
            Complex64::new(-1.0, 0.75),
            Complex64::new(0.25, 0.5),
        ];
        assert_eq!(
            checked.try_apply_transpose(&input).unwrap(),
            operator.apply_transpose(&input)
        );
        assert_eq!(
            checked.try_apply_hermitian(&input).unwrap(),
            operator.apply_hermitian(&input)
        );
        assert!(!invalid_output.load(Ordering::Relaxed));
    }

    #[test]
    fn tfqmr_rejects_invalid_input_before_operator_use() {
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

        let config = TfqmrConfig::default();
        let non_finite = tfqmr(&PanicOperator, &[f64::NAN], &config);
        assert_eq!(non_finite.status, SolverStatus::NumericalError);
        assert!(!non_finite.converged);
        assert_eq!(non_finite.iterations, 0);

        let wrong_dimension = tfqmr(&PanicOperator, &[1.0, 2.0], &config);
        assert_eq!(wrong_dimension.status, SolverStatus::NumericalError);
        assert!(!wrong_dimension.converged);
    }

    #[test]
    fn checked_tfqmr_contains_operator_panics() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("TFQMR operator callback failed");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("TFQMR operator callback failed");
            }
        }

        let error = try_tfqmr(&PanicOperator, &[1.0], &TfqmrConfig::default())
            .expect_err("checked TFQMR must type callback panics");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "TFQMR",
                ..
            }
        ));

        let legacy = tfqmr(&PanicOperator, &[1.0], &TfqmrConfig::default());
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn checked_tfqmr_rejects_non_finite_operator_output() {
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

        let error = try_tfqmr(&NonFiniteOperator, &[1.0], &TfqmrConfig::default())
            .expect_err("checked TFQMR must reject non-finite operator output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "TFQMR operator",
                ..
            }
        ));

        let legacy = tfqmr(&NonFiniteOperator, &[1.0], &TfqmrConfig::default());
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn checked_tfqmr_rejects_non_finite_preconditioner_output() {
        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
                output[0] = f64::INFINITY;
            }
        }

        let dense = sonium_solvers_core::array![[1.0_f64]];
        let operator = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = NonFinitePreconditioner;
        let config = TfqmrConfig::default();

        let error = try_tfqmr_preconditioned(&operator, &precond, &[1.0], &config)
            .expect_err("checked TFQMR must reject non-finite preconditioner output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "TFQMR preconditioner",
                ..
            }
        ));

        let legacy = tfqmr_preconditioned(&operator, &precond, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn test_tfqmr_simple() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = TfqmrConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = tfqmr(&a, &b, &config);
        assert!(solution.converged, "TFQMR should converge");
        assert!(
            true_rel_residual_complex(&a, &solution.x, &b) < 1e-8,
            "Solution should satisfy Ax = b"
        );
    }

    #[test]
    fn test_tfqmr_nonsymmetric() {
        let (a, b) = nonsymmetric_3x3();
        let config = TfqmrConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = tfqmr(&a, &b, &config);
        assert!(
            solution.converged,
            "TFQMR should converge for non-symmetric system"
        );
        let rel = true_rel_residual_complex(&a, &solution.x, &b);
        assert!(rel < 1e-8, "Solution error too large: {}", rel);
    }

    #[test]
    fn test_tfqmr_real_nonsymmetric_preconditioned_ilu() {
        // Non-symmetric real 40x40 system with an ILU(0) preconditioner,
        // shared with bicgstab's preconditioned test.
        let n = 40;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = 1.5;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = 0.5;
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&a);
        let b = (0..n).map(|i| (i as f64).sin() + 1.0).collect::<Vec<_>>();

        let config = TfqmrConfig {
            max_iterations: 500,
            tolerance: 1e-12,
            print_interval: 0,
        };
        let solution = tfqmr_preconditioned(&a, &precond, &b, &config);
        assert!(
            solution.converged,
            "Preconditioned TFQMR (ILU) should converge, residual = {}",
            solution.residual
        );
        assert!(solution.op_counts.preconditioner_count > 0);

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution error too large: {}", error);
    }

    #[test]
    fn test_tfqmr_complex_nonhermitian_preconditioned() {
        // Non-symmetric complex tridiagonal with a Jacobi preconditioner,
        // shared with bicgstab's complex preconditioned test.
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

        let config = TfqmrConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = tfqmr_preconditioned(&a, &precond, &b, &config);
        assert!(
            solution.converged,
            "Preconditioned TFQMR (Jacobi) should converge, residual = {}",
            solution.residual
        );
        assert!(solution.op_counts.preconditioner_count > 0);
        let rel = true_rel_residual_complex(&a, &solution.x, &b);
        assert!(rel < 1e-6, "Solution error too large: {}", rel);
    }

    #[test]
    fn test_tfqmr_vs_bicgstab_comparison() {
        // Complex shifted-Laplacian-type system: compare TFQMR and BiCGSTAB
        // convergence (printed, not asserted — TFQMR's curve is typically
        // smoother). Both must converge.
        let n = 100;
        let k2 = 0.7; // Helmholtz-like shift, keeps the matrix indefinite
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(2.0 - k2, 0.05);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.1).cos()))
            .collect::<Vec<_>>();

        let tfqmr_config = TfqmrConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        };
        let bicgstab_config = BiCgstabConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        };

        let tfqmr_sol = tfqmr(&a, &b, &tfqmr_config);
        let bicgstab_sol = bicgstab(&a, &b, &bicgstab_config);

        println!(
            "shifted-Laplacian n={n}: TFQMR {} iterations (residual {:.3e}), \
             BiCGSTAB {} iterations (residual {:.3e})",
            tfqmr_sol.iterations,
            tfqmr_sol.residual,
            bicgstab_sol.iterations,
            bicgstab_sol.residual,
        );

        assert!(tfqmr_sol.converged, "TFQMR should converge");
        assert!(bicgstab_sol.converged, "BiCGSTAB should converge");
        assert!(true_rel_residual_complex(&a, &tfqmr_sol.x, &b) < 1e-6);
    }

    #[test]
    fn test_tfqmr_zero_rhs() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

        let config = TfqmrConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = tfqmr(&a, &b, &config);
        assert!(solution.converged, "Zero RHS should converge");
        assert_eq!(solution.iterations, 0, "Zero RHS should need 0 iterations");
    }

    #[test]
    fn test_tfqmr_breakdown_shadow_residual() {
        // b = [1, i] with A = I: the unconjugated bi-Lanczos dot
        // (r0, r0) = 1 + i^2 = 0, so TFQMR must report a breakdown instead of
        // dividing by zero — even though the system is trivially solvable.
        let dense = sonium_solvers_core::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];

        let config = TfqmrConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            print_interval: 0,
        };
        let solution = tfqmr(&a, &b, &config);
        assert!(
            !solution.converged,
            "zero bi-Lanczos dot must not converge silently"
        );
        assert_eq!(solution.status, SolverStatus::Breakdown);
        // The reported residual is the true one: x never moved, so it is ~1.
        assert!(
            (solution.residual - 1.0).abs() < 1e-9,
            "honest residual expected, got {}",
            solution.residual
        );
    }

    #[test]
    fn test_tfqmr_exact_guess_converges_immediately() {
        let (a, b) = nonsymmetric_3x3();
        let precond = DiagonalPreconditioner::from_csr(&a);
        let config = TfqmrConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let reference = tfqmr_preconditioned(&a, &precond, &b, &config);
        assert!(reference.converged);

        let warm = tfqmr_preconditioned_with_guess(&a, &precond, &b, Some(&reference.x), &config);
        assert!(warm.converged, "Exact initial guess should converge");
        assert_eq!(warm.iterations, 0, "Exact initial guess needs 0 iterations");

        // A perturbed guess must still converge to the same solution.
        let x0: Vec<Complex64> = reference
            .x
            .iter()
            .map(|xi| xi + Complex64::new(0.01, -0.01))
            .collect();
        let warm = tfqmr_preconditioned_with_guess(&a, &precond, &b, Some(&x0), &config);
        assert!(warm.converged, "Perturbed guess should still converge");
        let diff: f64 = sonium_solvers_core::vec_ops::vec_sub(&warm.x, &reference.x)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-6, "Warm-start solution mismatch: {}", diff);
    }

    #[test]
    fn test_tfqmr_residual_replacement_and_honesty() {
        // Ill-conditioned system that should need more than
        // RESIDUAL_REPLACEMENT_INTERVAL outer iterations, so the residual
        // replacement path actually runs; the reported residual must be the
        // true residual, not the recursive/quasi estimate.
        let n = 60;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 2.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n).map(|i| (i as f64).cos()).collect::<Vec<_>>();

        let config = TfqmrConfig {
            max_iterations: 2000,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = tfqmr(&a, &b, &config);
        assert!(
            solution.converged,
            "TFQMR with residual replacement should converge, residual = {}",
            solution.residual
        );
        assert!(
            solution.iterations > 50,
            "Test must exercise the residual-replacement path (iterations = {})",
            solution.iterations
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e * e).sum::<f64>().sqrt();
        assert!(
            error / b_norm < 1e-8,
            "True residual too large after residual replacement: {}",
            error / b_norm
        );
        // Honesty: reported residual must match the true residual.
        assert!(
            (solution.residual - error / b_norm).abs() < 1e-12,
            "reported residual {} must equal true residual {}",
            solution.residual,
            error / b_norm
        );
    }

    #[test]
    fn tfqmr_max_iteration_reports_true_residual_and_backward_error() {
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
        let config = TfqmrConfig {
            max_iterations: 1,
            tolerance: 1.0e-14,
            print_interval: 0,
        };
        let solution = tfqmr(&matrix, &rhs, &config);
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
    fn tfqmr_final_metrics_match_independent_scaled_norm_reference() {
        struct IdentityOperator;

        impl LinearOperator<f64> for IdentityOperator {
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

        let scale = f64::MAX / 4.0;
        let rhs = [scale, scale * 0.5];
        let result = TfqmrSolution {
            x: vec![scale * 0.5, scale * 0.25],
            iterations: 1,
            residual: 1.0,
            estimated_residual: 1.0,
            backward_error: 1.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::MaxIterationsReached,
            op_counts: KrylovOpCounts::default(),
        };

        let finalized = finalize_tfqmr_solution(&IdentityOperator, &rhs, 1.0e-12, result);
        let rhs_norm = scale.hypot(scale * 0.5);
        let residual_norm = (scale * 0.5).hypot(scale * 0.25);
        let expected_residual = residual_norm / rhs_norm;
        let expected_backward_error = residual_norm / (residual_norm + rhs_norm);

        assert!(finalized.residual.is_finite());
        assert!((finalized.residual - expected_residual).abs() < 1.0e-15);
        assert!(
            (finalized.backward_error - expected_backward_error).abs() < 1.0e-15,
            "backward error {} != independent reference {}",
            finalized.backward_error,
            expected_backward_error
        );
        assert_eq!(finalized.status, SolverStatus::MaxIterationsReached);
        assert_eq!(finalized.attainable_accuracy, AccuracyStatus::NotAttained);
    }

    #[test]
    fn tfqmr_final_metrics_classify_nonfinite_operator_action_as_failure() {
        struct NonFiniteOperator;

        impl LinearOperator<f64> for NonFiniteOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
                y[0] = f64::INFINITY;
            }

            fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
                y[0] = f64::INFINITY;
            }
        }

        let result = TfqmrSolution {
            x: vec![0.0],
            iterations: 1,
            residual: 0.0,
            estimated_residual: 0.0,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::MaxIterationsReached,
            op_counts: KrylovOpCounts::default(),
        };
        let finalized = finalize_tfqmr_solution(&NonFiniteOperator, &[1.0], 1.0e-12, result);

        assert_eq!(finalized.status, SolverStatus::NumericalError);
        assert!(!finalized.converged);
        assert!(finalized.residual.is_infinite());
        assert_eq!(finalized.attainable_accuracy, AccuracyStatus::NotChecked);
    }

    #[test]
    fn test_tfqmr_reports_krylov_op_counts() {
        let (a, b) = nonsymmetric_3x3();
        let config = TfqmrConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = tfqmr(&a, &b, &config);
        assert!(solution.converged);
        // Two operator applications per outer iteration, plus setup/exit checks.
        assert!(solution.op_counts.spmv_count >= 2 * solution.iterations);
        assert!(solution.op_counts.dot_reductions >= 2 * solution.iterations);
        assert!(solution.op_counts.norm_count > 0);
        assert!(solution.op_counts.axpy_passes >= solution.iterations);
        assert_eq!(solution.op_counts.preconditioner_count, 0);
        // True-residual exit check ran.
        assert!(solution.op_counts.exact_residual_checks >= 1);
    }
}
