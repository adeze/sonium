//! BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver
//!
//! BiCGSTAB is a Krylov subspace method for non-symmetric systems.
//! It often converges faster than GMRES for certain problem types.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{
    axpy, copy_axpy_norm, dot_unconjugated, inner_product, scale_inplace, vector_norm,
};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};

use super::cgs_orth::KrylovOpCounts;
use super::checked_callbacks::{
    CheckedLinearOperator, CheckedPreconditioner, catch_solver_callback,
};

/// Interval (in iterations) at which the recursive residual is replaced by the
/// true residual `r = b - A*x` and the bi-Lanczos recurrence is restarted from
/// it (residual replacement). This curbs the rounding-error build-up in the
/// recursively updated residual that otherwise limits the attainable accuracy
/// of BiCGSTAB (cf. Sleijpen & van der Vorst, "Reliable updated residuals in
/// hybrid Bi-CG methods").
const RESIDUAL_REPLACEMENT_INTERVAL: usize = 50;

/// BiCGSTAB solver configuration
#[derive(Debug, Clone)]
pub struct BiCgstabConfig<R> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
}

impl Default for BiCgstabConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        }
    }
}

/// BiCGSTAB solver result
#[derive(Debug)]
pub struct BiCgstabSolution<T: ComplexField> {
    /// Solution vector
    pub x: Vec<T>,
    /// Number of iterations
    pub iterations: usize,
    /// Final relative residual
    pub residual: T::Real,
    /// Cheap recurrence residual captured before the independent final check.
    pub estimated_residual: T::Real,
    /// Normwise backward error against the original operator.
    pub backward_error: T::Real,
    /// Whether the independently recomputed residual meets the requested tolerance.
    pub attainable_accuracy: AccuracyStatus,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Solver status
    pub status: SolverStatus,
    /// Memory-traffic oriented operation counts for the Krylov solve.
    pub op_counts: KrylovOpCounts,
}

pub(crate) fn invalid_input_solution<T: ComplexField>(
    b: &[T],
    x0: Option<&[T]>,
) -> BiCgstabSolution<T> {
    let x = match x0 {
        Some(x0) if x0.len() == b.len() => x0.to_vec(),
        _ => vec![T::zero(); b.len()],
    };
    BiCgstabSolution {
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

pub(crate) fn has_invalid_input<T: ComplexField>(
    b: &[T],
    x0: Option<&[T]>,
    config: &BiCgstabConfig<T::Real>,
) -> bool {
    config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
        || b.iter().any(|value| !value.norm().is_finite())
        || x0.is_some_and(|guess| {
            guess.len() != b.len() || guess.iter().any(|value| !value.norm().is_finite())
        })
}

fn finalize_true_residual<T, A>(
    operator: &A,
    b: &[T],
    solution: &mut BiCgstabSolution<T>,
    tolerance: T::Real,
) where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let estimated_residual = solution.residual;
    let ax = operator.apply(&solution.x);
    let residual: Vec<T> = b
        .iter()
        .zip(ax.iter())
        .map(|(&b_i, &ax_i)| b_i - ax_i)
        .collect();
    let residual_norm = vector_norm(&residual);
    let rhs_norm = vector_norm(b);
    let ax_norm = vector_norm(&ax);
    let denominator = if rhs_norm > T::Real::zero() {
        rhs_norm
    } else {
        T::Real::one()
    };
    solution.estimated_residual = estimated_residual;
    solution.residual = residual_norm / denominator;
    solution.backward_error = T::normwise_backward_error(residual_norm, ax_norm, rhs_norm);
    solution.op_counts.spmv_count += 1;
    solution.op_counts.norm_count += 3;
    solution.attainable_accuracy =
        if solution.residual.is_finite() && solution.residual <= tolerance {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        };
    if !solution.residual.is_finite() || !solution.backward_error.is_finite() {
        solution.converged = false;
        solution.status = SolverStatus::NumericalError;
        solution.attainable_accuracy = AccuracyStatus::NotChecked;
    } else if solution.converged && solution.residual >= tolerance {
        solution.converged = false;
        solution.status = SolverStatus::Stagnated;
    }
}

fn validate_bicgstab_inputs<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &BiCgstabConfig<T::Real>,
) -> Result<usize, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
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
    if let Some(guess) = x0 {
        if guess.len() != cols {
            return Err(SolverError::DimensionMismatch {
                expected: cols,
                got: guess.len(),
            });
        }
        if guess.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "BiCGSTAB",
                reason: "initial guess contains NaN or infinity".to_string(),
            });
        }
    }
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "BiCGSTAB",
            reason: "invalid BiCGSTAB configuration".to_string(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "BiCGSTAB",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    Ok(cols)
}

fn validate_bicgstab_solution<T: ComplexField>(
    solution: &BiCgstabSolution<T>,
    dimension: usize,
) -> Result<(), SolverError> {
    if solution.status == SolverStatus::NumericalError
        || solution.x.len() != dimension
        || solution.x.iter().any(|value| !value.norm().is_finite())
        || !solution.residual.is_finite()
        || !solution.estimated_residual.is_finite()
        || !solution.backward_error.is_finite()
    {
        return Err(SolverError::OperationFailed {
            operation: "BiCGSTAB",
            reason: "solver generated NaN, infinity, or an invalid result".to_string(),
        });
    }
    Ok(())
}

/// Checked BiCGSTAB entry point with typed callback and generated-value
/// failures.
pub fn try_bicgstab<T, A>(
    operator: &A,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
) -> Result<BiCgstabSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    catch_solver_callback("BiCGSTAB", || {
        let dimension = validate_bicgstab_inputs(operator, b, None, config)?;
        let checked_operator = CheckedLinearOperator::new(operator);
        let mut solution = bicgstab_unchecked(&checked_operator, b, config);
        finalize_true_residual(&checked_operator, b, &mut solution, config.tolerance);
        validate_bicgstab_solution(&solution, dimension)?;
        SolverResultContract::validate_contract(&solution)?;
        Ok(solution)
    })
}

/// Checked preconditioned BiCGSTAB entry point.
pub fn try_bicgstab_preconditioned<T, A, P>(
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
    try_bicgstab_preconditioned_with_guess(operator, precond, b, None, config)
}

/// Checked preconditioned BiCGSTAB entry point with an optional initial guess.
pub fn try_bicgstab_preconditioned_with_guess<T, A, P>(
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
    catch_solver_callback("BiCGSTAB", || {
        let dimension = validate_bicgstab_inputs(operator, b, x0, config)?;
        let checked_operator = CheckedLinearOperator::new(operator);
        let checked_precond = CheckedPreconditioner::new(precond);
        let mut solution =
            bicgstab_preconditioned_core(&checked_operator, &checked_precond, b, x0, config);
        finalize_true_residual(&checked_operator, b, &mut solution, config.tolerance);
        validate_bicgstab_solution(&solution, dimension)?;
        SolverResultContract::validate_contract(&solution)?;
        Ok(solution)
    })
}

/// Solve Ax = b using the BiCGSTAB method
pub fn bicgstab<T, A>(
    operator: &A,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_bicgstab(operator, b, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

fn bicgstab_unchecked<T, A>(
    operator: &A,
    b: &[T],
    config: &BiCgstabConfig<T::Real>,
) -> BiCgstabSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let n = b.len();
    let mut x = vec![T::zero(); n];
    let mut op_counts = KrylovOpCounts::default();

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        return BiCgstabSolution {
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
    let mut r0 = r.clone(); // Shadow residual

    let b_norm_sqr = b_norm * b_norm;
    // Use eps² as the multiplier so the threshold only scales with ||b||²
    // for extremely large problems.  For f64, eps² ≈ 5e-32, which is far
    // smaller than the old 1e-20 and prevents false breakdown on typical
    // large-RHS problems.
    let eps = T::Real::epsilon();
    let scaled = b_norm_sqr * eps * eps;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_threshold = if scaled > floor { scaled } else { floor };

    let mut rho = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();

    let mut p = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut s = vec![T::zero(); n];
    let mut t = vec![T::zero(); n];

    for iter in 0..config.max_iterations {
        // Residual replacement: recompute the true residual r = b - A*x and
        // restart the recurrence from it (new shadow residual, reset scalars).
        if iter > 0 && iter % RESIDUAL_REPLACEMENT_INTERVAL == 0 {
            operator.apply_into(&x, &mut t);
            op_counts.spmv_count += 1;
            r.copy_from_slice(b);
            axpy(-T::one(), &t, &mut r);
            op_counts.axpy_passes += 1;
            r0.copy_from_slice(&r);
            rho = T::one();
            alpha = T::one();
            omega = T::one();
            p.fill(T::zero());
            v.fill(T::zero());
        }

        let rho_new = dot_unconjugated(&r0, &r);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;

        // Check for breakdown
        if rho_new.norm() < breakdown_threshold {
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

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        //   = r + beta*p - beta*omega*v
        // In-place: p *= beta, then p -= beta*omega*v, then p += r
        scale_inplace(&mut p, beta);
        axpy(-(beta * omega), &v, &mut p);
        axpy(T::one(), &r, &mut p);
        op_counts.axpy_passes += 3;

        // v = A * p (zero allocation via apply_into)
        operator.apply_into(&p, &mut v);
        op_counts.spmv_count += 1;

        let r0v = dot_unconjugated(&r0, &v);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if r0v.norm() < breakdown_threshold {
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

        alpha = rho / r0v;

        // s = r - alpha * v  (reuse s buffer)
        let s_norm = copy_axpy_norm(&r, -alpha, &v, &mut s);
        op_counts.axpy_passes += 1;
        op_counts.norm_count += 1;
        if s_norm / b_norm < config.tolerance {
            axpy(alpha, &p, &mut x);
            op_counts.axpy_passes += 1;
            return BiCgstabSolution {
                x,
                iterations: iter + 1,
                residual: s_norm / b_norm,
                estimated_residual: s_norm / b_norm,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // t = A * s (zero allocation via apply_into)
        operator.apply_into(&s, &mut t);
        op_counts.spmv_count += 1;

        // omega = (t, s) / (t, t)
        let tt = inner_product(&t, &t);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if tt.norm() < breakdown_threshold {
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
        omega = inner_product(&t, &s) / tt;
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;

        // x += alpha * p + omega * s
        axpy(alpha, &p, &mut x);
        axpy(omega, &s, &mut x);
        op_counts.axpy_passes += 2;

        // r = s - omega * t  (reuse r buffer)
        let rel_residual = copy_axpy_norm(&s, -omega, &t, &mut r) / b_norm;
        op_counts.axpy_passes += 1;
        op_counts.norm_count += 1;

        if config.print_interval > 0 && (iter + 1) % config.print_interval == 0 {
            log::info!(
                "BiCGSTAB iteration {}: relative residual = {:.6e}",
                iter + 1,
                rel_residual.to_f64().unwrap_or(0.0)
            );
        }

        if rel_residual < config.tolerance {
            return BiCgstabSolution {
                x,
                iterations: iter + 1,
                residual: rel_residual,
                estimated_residual: rel_residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // Check for stagnation
        // omega is a ratio (scale-invariant), so absolute threshold applies
        let omega_threshold = T::Real::from_f64(1e-20).unwrap();
        if omega.norm() < omega_threshold {
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

/// Solve Ax = b using the BiCGSTAB method with a right preconditioner.
pub fn bicgstab_preconditioned<T, A, P>(
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
    match try_bicgstab_preconditioned(operator, precond, b, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

/// Solve `Ax = b` using BiCGSTAB with a right preconditioner and an optional
/// initial guess.
pub fn bicgstab_preconditioned_with_guess<T, A, P>(
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
    match try_bicgstab_preconditioned_with_guess(operator, precond, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, x0),
    }
}

/// Core BiCGSTAB with right preconditioning, an optional initial guess, and
/// residual replacement.
///
/// Right preconditioning keeps all residual vectors in the original space:
///   p_hat = M^{-1} p,  v = A * p_hat
///   s_hat = M^{-1} s,  t = A * s_hat
///   x += alpha * p_hat + omega * s_hat
///
/// Every [`RESIDUAL_REPLACEMENT_INTERVAL`] iterations the true residual
/// `r = b - A*x` is recomputed and the recurrence is restarted from it, which
/// curbs rounding-error build-up in the recursively updated residual.
#[allow(clippy::too_many_lines)]
fn bicgstab_preconditioned_core<T, A, P>(
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
    let n = b.len();
    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut op_counts = KrylovOpCounts::default();

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        return BiCgstabSolution {
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

    // Initial residual r = b - A*x0 (true residual when a guess is given)
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
    let initial_rel_residual = vector_norm(&r) / b_norm;
    op_counts.norm_count += 1;
    if initial_rel_residual < config.tolerance {
        return BiCgstabSolution {
            x,
            iterations: 0,
            residual: initial_rel_residual,
            estimated_residual: initial_rel_residual,
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }
    let mut r0 = r.clone(); // Shadow residual

    let b_norm_sqr = b_norm * b_norm;
    // Same eps²-scaled breakdown threshold as the unpreconditioned solver.
    let eps = T::Real::epsilon();
    let scaled = b_norm_sqr * eps * eps;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_threshold = if scaled > floor { scaled } else { floor };

    let mut rho = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();

    let mut p = vec![T::zero(); n];
    let mut p_hat = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut s = vec![T::zero(); n];
    let mut s_hat = vec![T::zero(); n];
    let mut t = vec![T::zero(); n];

    for iter in 0..config.max_iterations {
        // Residual replacement: recompute the true residual r = b - A*x and
        // restart the recurrence from it (new shadow residual, reset scalars).
        if iter > 0 && iter % RESIDUAL_REPLACEMENT_INTERVAL == 0 {
            operator.apply_into(&x, &mut t);
            op_counts.spmv_count += 1;
            r.copy_from_slice(b);
            axpy(-T::one(), &t, &mut r);
            op_counts.axpy_passes += 1;
            r0.copy_from_slice(&r);
            rho = T::one();
            alpha = T::one();
            omega = T::one();
            p.fill(T::zero());
            v.fill(T::zero());
        }

        let rho_new = dot_unconjugated(&r0, &r);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;

        // Check for breakdown
        if rho_new.norm() < breakdown_threshold {
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

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        scale_inplace(&mut p, beta);
        axpy(-(beta * omega), &v, &mut p);
        axpy(T::one(), &r, &mut p);
        op_counts.axpy_passes += 3;

        // p_hat = M^{-1} p, then v = A * p_hat
        precond.apply_into(&p, &mut p_hat);
        op_counts.preconditioner_count += 1;
        operator.apply_into(&p_hat, &mut v);
        op_counts.spmv_count += 1;

        let r0v = dot_unconjugated(&r0, &v);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if r0v.norm() < breakdown_threshold {
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

        alpha = rho / r0v;

        // s = r - alpha * v  (reuse s buffer)
        let s_norm = copy_axpy_norm(&r, -alpha, &v, &mut s);
        op_counts.axpy_passes += 1;
        op_counts.norm_count += 1;
        if s_norm / b_norm < config.tolerance {
            axpy(alpha, &p_hat, &mut x);
            op_counts.axpy_passes += 1;
            return BiCgstabSolution {
                x,
                iterations: iter + 1,
                residual: s_norm / b_norm,
                estimated_residual: s_norm / b_norm,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // s_hat = M^{-1} s, then t = A * s_hat
        precond.apply_into(&s, &mut s_hat);
        op_counts.preconditioner_count += 1;
        operator.apply_into(&s_hat, &mut t);
        op_counts.spmv_count += 1;

        // omega = (t, s) / (t, t)
        let tt = inner_product(&t, &t);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if tt.norm() < breakdown_threshold {
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
        omega = inner_product(&t, &s) / tt;
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;

        // x += alpha * p_hat + omega * s_hat (preconditioned directions)
        axpy(alpha, &p_hat, &mut x);
        axpy(omega, &s_hat, &mut x);
        op_counts.axpy_passes += 2;

        // r = s - omega * t  (stays in unpreconditioned space)
        let rel_residual = copy_axpy_norm(&s, -omega, &t, &mut r) / b_norm;
        op_counts.axpy_passes += 1;
        op_counts.norm_count += 1;

        if config.print_interval > 0 && (iter + 1) % config.print_interval == 0 {
            log::info!(
                "BiCGSTAB iteration {}: relative residual = {:.6e}",
                iter + 1,
                rel_residual.to_f64().unwrap_or(0.0)
            );
        }

        if rel_residual < config.tolerance {
            return BiCgstabSolution {
                x,
                iterations: iter + 1,
                residual: rel_residual,
                estimated_residual: rel_residual,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // Check for stagnation
        // omega is a ratio (scale-invariant), so absolute threshold applies
        let omega_threshold = T::Real::from_f64(1e-20).unwrap();
        if omega.norm() < omega_threshold {
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
    use num_complex::Complex64;
    use sonium_solvers_preconditioners::{DiagonalPreconditioner, IluPreconditioner};
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn bicgstab_reports_independent_residual_and_backward_error_on_max_iterations() {
        let a = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]],
            1e-15,
        );
        let b = sonium_solvers_core::array![1.0_f64, 2.0];
        let config = BiCgstabConfig {
            max_iterations: 1,
            tolerance: 1e-15,
            print_interval: 0,
        };

        let solution = try_bicgstab(&a, &b, &config).expect("finite max-iteration result");
        let ax = a.matvec(&solution.x);
        let residual = sonium_solvers_core::vec_ops::vec_sub(&b, &ax);
        let residual_norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let rhs_norm = b.iter().map(|value| value * value).sum::<f64>().sqrt();
        let ax_norm = ax.iter().map(|value| value * value).sum::<f64>().sqrt();

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
    fn direct_bicgstab_rejects_non_finite_input_before_operator_use() {
        struct PanicOperator;
        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator must not be called")
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator must not be called")
            }
        }

        let config = BiCgstabConfig {
            max_iterations: 4,
            tolerance: 1.0e-8,
            print_interval: 0,
        };
        let solution = bicgstab(&PanicOperator, &[f64::NAN], &config);
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
    }

    #[test]
    fn checked_bicgstab_contains_callback_panics_and_non_finite_outputs() {
        struct PanicOperator;
        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator callback failed")
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator transpose callback failed")
            }
        }

        let config = BiCgstabConfig::default();
        let error = try_bicgstab(&PanicOperator, &[1.0], &config)
            .expect_err("checked BiCGSTAB must contain operator panics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        #[allow(deprecated)]
        let legacy = bicgstab(&PanicOperator, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);

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

        let error = try_bicgstab(&NonFiniteOperator, &[1.0], &config)
            .expect_err("checked BiCGSTAB must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_preconditioned_bicgstab_contains_preconditioner_failures() {
        struct IdentityOperator;
        impl LinearOperator<f64> for IdentityOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, x: &[f64], y: &mut [f64]) {
                y[0] = x[0];
            }

            fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
                y[0] = x[0];
            }
        }

        struct PanicPreconditioner;
        impl Preconditioner<f64> for PanicPreconditioner {
            fn apply(&self, _r: &[f64]) -> Vec<f64> {
                panic!("preconditioner callback failed")
            }

            fn apply_transpose(&self, _r: &[f64]) -> Vec<f64> {
                panic!("preconditioner transpose callback failed")
            }
        }

        let config = BiCgstabConfig::default();
        let error =
            try_bicgstab_preconditioned(&IdentityOperator, &PanicPreconditioner, &[1.0], &config)
                .expect_err("checked BiCGSTAB must contain preconditioner panics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        struct NonFinitePreconditioner;
        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply(&self, _r: &[f64]) -> Vec<f64> {
                vec![f64::NAN]
            }

            fn apply_transpose(&self, _r: &[f64]) -> Vec<f64> {
                vec![f64::NAN]
            }
        }

        let error = try_bicgstab_preconditioned(
            &IdentityOperator,
            &NonFinitePreconditioner,
            &[1.0],
            &config,
        )
        .expect_err("checked BiCGSTAB must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn test_bicgstab_simple() {
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

        let solution = bicgstab(&a, &b, &config);

        assert!(solution.converged, "BiCGSTAB should converge");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution should satisfy Ax = b");
    }

    #[test]
    fn test_bicgstab_nonsymmetric() {
        // Non-symmetric 3x3 system: off-diagonal entries are not equal
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
        let b = sonium_solvers_core::array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0)
        ];

        let config = BiCgstabConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab(&a, &b, &config);
        assert!(
            solution.converged,
            "BiCGSTAB should converge for non-symmetric system"
        );

        // Verify Ax = b
        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution error too large: {}", error);
    }

    #[test]
    fn test_bicgstab_complex() {
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

        let solution = bicgstab(&a, &b, &config);
        assert!(
            solution.converged,
            "BiCGSTAB should converge for complex system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Complex solution error too large: {}", error);
    }

    #[test]
    fn test_bicgstab_larger_tridiagonal() {
        // 20-node SPD tridiagonal system, verify residual < 1e-8
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

        let solution = bicgstab(&a, &b, &config);
        assert!(
            solution.converged,
            "BiCGSTAB should converge on 20-node tridiagonal"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Residual too large on tridiagonal: {}", error);
    }

    #[test]
    fn test_bicgstab_reports_krylov_op_counts() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let config = BiCgstabConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab(&a, &b, &config);

        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions >= 2 * solution.iterations);
        assert!(solution.op_counts.norm_count > 0);
        assert!(solution.op_counts.axpy_passes >= solution.iterations);
        assert_eq!(solution.op_counts.preconditioner_count, 0);
    }

    #[test]
    fn test_bicgstab_preconditioned_jacobi_nonsymmetric() {
        // Non-symmetric complex tridiagonal with a Jacobi (diagonal) preconditioner
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

        let solution = bicgstab_preconditioned(&a, &precond, &b, &config);
        assert!(
            solution.converged,
            "Preconditioned BiCGSTAB (Jacobi) should converge, residual = {}",
            solution.residual
        );
        assert!(
            solution.op_counts.preconditioner_count > 0,
            "Preconditioner must have been applied"
        );

        // Right preconditioning: the returned x must satisfy the original Ax = b
        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-6,
            "Preconditioned (Jacobi) solution error too large: {}",
            error
        );
    }

    #[test]
    fn test_bicgstab_preconditioned_ilu_nonsymmetric() {
        // Non-symmetric real system with an ILU(0) preconditioner
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

        let config = BiCgstabConfig {
            max_iterations: 500,
            tolerance: 1e-12,
            print_interval: 0,
        };

        let solution = bicgstab_preconditioned(&a, &precond, &b, &config);
        assert!(
            solution.converged,
            "Preconditioned BiCGSTAB (ILU) should converge, residual = {}",
            solution.residual
        );
        assert!(solution.op_counts.preconditioner_count > 0);

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "Preconditioned (ILU) solution error too large: {}",
            error
        );
    }

    #[test]
    fn test_bicgstab_preconditioned_with_guess() {
        // An exact initial guess must converge immediately (0 iterations)
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
        let precond = DiagonalPreconditioner::from_csr(&a);
        let b = sonium_solvers_core::array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0)
        ];

        let config = BiCgstabConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        // Reference solution from a zero guess
        let reference = bicgstab_preconditioned(&a, &precond, &b, &config);
        assert!(reference.converged);

        let warm =
            bicgstab_preconditioned_with_guess(&a, &precond, &b, Some(&reference.x), &config);
        assert!(
            warm.converged,
            "Exact initial guess should converge immediately"
        );
        assert_eq!(
            warm.iterations, 0,
            "Exact initial guess should need 0 iterations"
        );

        // A perturbed guess must still converge to the same solution
        let x0: Vec<Complex64> = reference
            .x
            .iter()
            .map(|xi| xi + Complex64::new(0.01, -0.01))
            .collect();
        let warm = bicgstab_preconditioned_with_guess(&a, &precond, &b, Some(&x0), &config);
        assert!(warm.converged, "Perturbed guess should still converge");
        let diff: f64 = sonium_solvers_core::vec_ops::vec_sub(&warm.x, &reference.x)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-6, "Warm-start solution mismatch: {}", diff);
    }

    #[test]
    fn test_bicgstab_residual_replacement() {
        // Ill-conditioned system that needs more than RESIDUAL_REPLACEMENT_INTERVAL
        // iterations, so the residual-replacement/restart path actually runs; the
        // restarted recurrence must still converge to the true solution.
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

        let config = BiCgstabConfig {
            max_iterations: 2000,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab(&a, &b, &config);
        assert!(
            solution.converged,
            "BiCGSTAB with residual replacement should converge, residual = {}",
            solution.residual
        );
        assert!(
            solution.iterations > 50,
            "Test must exercise the residual-replacement path (iterations = {})",
            solution.iterations
        );

        // The reported residual is recursive; verify against the true residual
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
    }

    #[test]
    fn test_bicgstab_zero_rhs() {
        // Zero RHS should converge immediately with 0 iterations
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

        let config = BiCgstabConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab(&a, &b, &config);
        assert!(solution.converged, "Zero RHS should converge");
        assert_eq!(solution.iterations, 0, "Zero RHS should need 0 iterations");
    }

    #[test]
    fn test_bicgstab_scaled_threshold_not_breakdown() {
        // Regression: breakdown thresholds must scale with ||b||^2.
        // Previously 1e-20 was unconditionally small (never triggered false breakdown)
        // but inconsistent. Now threshold = max(b_norm^2 * 1e-20, 1e-30).
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        // Large ||b|| → intermediate dot products scale with ||b||^2
        let scale = 1e8;
        let b = sonium_solvers_core::array![
            Complex64::new(scale, 0.0),
            Complex64::new(2.0 * scale, 0.0),
        ];

        let config = BiCgstabConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = bicgstab(&a, &b, &config);
        assert!(
            solution.converged,
            "BiCGSTAB should converge for scaled system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        // Relative error should be small
        let b_norm = (scale * scale + 4.0 * scale * scale).sqrt();
        assert!(
            error / b_norm < 1e-8,
            "Scaled system solution too inaccurate: relative error {}",
            error / b_norm
        );
    }
}
