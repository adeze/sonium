//! CG (Conjugate Gradient) solver
//!
//! The Conjugate Gradient method for symmetric positive definite systems.
//! This is the method of choice for SPD matrices as it has optimal convergence.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, scale_inplace, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};

use super::cgs_orth::KrylovOpCounts;
use super::checked_callbacks::{
    CheckedLinearOperator, CheckedPreconditioner, catch_solver_callback,
};

/// Compute `||x||` and `(x, y)` with no threading.
///
/// Runs the two SIMD kernels back to back on the calling thread: the
/// previous `rayon::join` of two bandwidth-bound sweeps saved no time even
/// uncontended and collapsed under oversubscription (per-iteration
/// rendezvous of a many-thread pool). Bit-identical to
/// `(vector_norm(x), inner_product(x, y))`.
#[inline]
fn fused_norm_dot<T: ComplexField>(x: &[T], y: &[T]) -> (T::Real, T)
where
    T::Real: Float,
{
    sonium_solvers_core::blas_helpers::norm_dot(x, y)
}

/// CG solver configuration
#[derive(Debug, Clone)]
pub struct CgConfig<R> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
}

impl Default for CgConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        }
    }
}

/// CG solver result
#[derive(Debug)]
pub struct CgSolution<T: ComplexField> {
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

fn invalid_input_solution<T: ComplexField>(b: &[T], x0: Option<&[T]>) -> CgSolution<T> {
    let x = match x0 {
        Some(x0) if x0.len() == b.len() => x0.to_vec(),
        _ => vec![T::zero(); b.len()],
    };
    CgSolution {
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

fn validate_cg_inputs<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &CgConfig<T::Real>,
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
                operation: "CG",
                reason: "initial guess contains NaN or infinity".to_string(),
            });
        }
    }
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "CG",
            reason: "invalid CG configuration".to_string(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "CG",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    Ok(cols)
}

fn validate_cg_solution<T: ComplexField>(
    solution: &CgSolution<T>,
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
            operation: "CG",
            reason: "solver generated NaN, infinity, or an invalid result".to_string(),
        });
    }
    Ok(())
}

/// Checked CG entry point with typed input, callback, and generated-value
/// failures.
pub fn try_cg<T, A>(
    operator: &A,
    b: &[T],
    config: &CgConfig<T::Real>,
) -> Result<CgSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    try_cg_with_guess(operator, b, None, config)
}

/// Checked CG entry point with an optional initial guess.
pub fn try_cg_with_guess<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &CgConfig<T::Real>,
) -> Result<CgSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    catch_solver_callback("CG", || {
        let dimension = validate_cg_inputs(operator, b, x0, config)?;
        let checked_operator = CheckedLinearOperator::new(operator);
        let mut solution = cg_with_guess_unchecked(&checked_operator, b, x0, config);
        finalize_cg_solution(&checked_operator, b, &mut solution, config.tolerance);
        validate_cg_solution(&solution, dimension)?;
        SolverResultContract::validate_contract(&solution)?;
        Ok(solution)
    })
}

/// Checked preconditioned CG entry point.
pub fn try_pcg<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &CgConfig<T::Real>,
) -> Result<CgSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_pcg_with_guess(operator, precond, b, None, config)
}

/// Checked preconditioned CG entry point with an optional initial guess.
pub fn try_pcg_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &CgConfig<T::Real>,
) -> Result<CgSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    catch_solver_callback("PCG", || {
        let dimension = validate_cg_inputs(operator, b, x0, config)?;
        let checked_operator = CheckedLinearOperator::new(operator);
        let checked_precond = CheckedPreconditioner::new(precond);
        let mut solution =
            pcg_with_guess_unchecked(&checked_operator, &checked_precond, b, x0, config);
        finalize_cg_solution(&checked_operator, b, &mut solution, config.tolerance);
        validate_cg_solution(&solution, dimension)?;
        SolverResultContract::validate_contract(&solution)?;
        Ok(solution)
    })
}

/// Solve Ax = b using the Conjugate Gradient method
///
/// Note: This method is only correct for symmetric positive definite matrices.
/// For non-symmetric systems, use GMRES or BiCGSTAB instead.
pub fn cg<T, A>(operator: &A, b: &[T], config: &CgConfig<T::Real>) -> CgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_cg(operator, b, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

/// Solve Ax = b using CG with an initial guess
pub fn cg_with_guess<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &CgConfig<T::Real>,
) -> CgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_cg_with_guess(operator, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, x0),
    }
}

fn cg_with_guess_unchecked<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &CgConfig<T::Real>,
) -> CgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let n = b.len();
    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut op_counts = KrylovOpCounts::default();

    let ax = operator.apply(&x);
    op_counts.spmv_count += 1;
    let r: Vec<T> = sonium_solvers_core::vec_ops::vec_sub(b, &ax);
    let initial_residual_norm = vector_norm(&r);
    op_counts.norm_count += 1;

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let scaled_thresh = b_norm * T::Real::from_f64(1e-15).unwrap();
    let tol_threshold = if scaled_thresh > floor {
        scaled_thresh
    } else {
        floor
    };
    if initial_residual_norm < tol_threshold {
        return CgSolution {
            x,
            iterations: 0,
            residual: if b_norm > floor {
                initial_residual_norm / b_norm
            } else {
                T::Real::from_f64(0.0).unwrap()
            },
            estimated_residual: if b_norm > floor {
                initial_residual_norm / b_norm
            } else {
                T::Real::from_f64(0.0).unwrap()
            },
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }

    let mut r = r;
    let mut p = r.clone();
    let mut q = vec![T::zero(); n];
    let mut rho = inner_product(&r, &r);
    op_counts.dot_reductions += 1;
    op_counts.synchronization_points += 1;

    // Guard: if b_norm is zero, use the initial residual norm as the
    // reference to avoid division-by-zero in the relative residual.
    let ref_norm = if b_norm > floor {
        b_norm
    } else {
        initial_residual_norm
    };

    let scaled = b_norm * T::Real::from_f64(1e-20).unwrap();
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_threshold = if scaled > floor { scaled } else { floor };

    for iter in 0..config.max_iterations {
        // q = A * p (zero allocation via apply_into)
        operator.apply_into(&p, &mut q);
        op_counts.spmv_count += 1;

        // alpha = rho / (p, q)
        let pq = inner_product(&p, &q);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if pq.norm() < breakdown_threshold {
            op_counts.norm_count += 1;
            return CgSolution {
                x,
                iterations: iter,
                residual: vector_norm(&r) / ref_norm,
                estimated_residual: vector_norm(&r) / ref_norm,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }

        let alpha = rho / pq;

        // x += alpha * p
        axpy(alpha, &p, &mut x);
        op_counts.axpy_passes += 1;

        // r -= alpha * q
        axpy(-alpha, &q, &mut r);
        op_counts.axpy_passes += 1;

        // Fused reduction: ||r|| (convergence test) and (r, r) (beta
        // numerator) in one parallel section — one barrier instead of two.
        let (r_norm, rho_new) = fused_norm_dot(&r, &r);
        op_counts.norm_count += 1;
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        let rel_residual = r_norm / ref_norm;

        if config.print_interval > 0 && (iter + 1) % config.print_interval == 0 {
            log::info!(
                "CG iteration {}: relative residual = {:.6e}",
                iter + 1,
                rel_residual.to_f64().unwrap_or(0.0)
            );
        }

        if rel_residual < config.tolerance {
            return CgSolution {
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

        if rho_new.norm() < breakdown_threshold {
            return CgSolution {
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

        let beta = rho_new / rho;
        rho = rho_new;

        // p = r + beta * p  (in-place: scale p by beta, then add r)
        scale_inplace(&mut p, beta);
        op_counts.axpy_passes += 1;
        axpy(T::one(), &r, &mut p);
        op_counts.axpy_passes += 1;
    }

    let rel_residual = vector_norm(&r) / ref_norm;
    op_counts.norm_count += 1;
    CgSolution {
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

fn true_residual_metrics<T, A>(operator: &A, b: &[T], x: &[T]) -> (T::Real, T::Real)
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let ax = operator.apply(x);
    let residual: Vec<T> = b
        .iter()
        .zip(ax.iter())
        .map(|(&b_i, &ax_i)| b_i - ax_i)
        .collect();
    let residual_norm = vector_norm(&residual);
    let rhs_norm = vector_norm(b);
    let ax_norm = vector_norm(&ax);
    let relative = residual_norm
        / if rhs_norm > T::Real::zero() {
            rhs_norm
        } else {
            T::Real::one()
        };
    let backward = T::normwise_backward_error(residual_norm, ax_norm, rhs_norm);
    (relative, backward)
}

fn finalize_cg_solution<T, A>(
    operator: &A,
    b: &[T],
    solution: &mut CgSolution<T>,
    tolerance: T::Real,
) where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let estimated_residual = solution.residual;
    let (true_residual, backward_error) = true_residual_metrics(operator, b, &solution.x);
    solution.op_counts.spmv_count += 1;
    solution.op_counts.norm_count += 3;
    solution.estimated_residual = estimated_residual;
    solution.residual = true_residual;
    solution.backward_error = backward_error;
    solution.attainable_accuracy = if true_residual.is_finite() && true_residual <= tolerance {
        AccuracyStatus::Attained
    } else {
        AccuracyStatus::NotAttained
    };
    if !true_residual.is_finite() || !backward_error.is_finite() {
        solution.converged = false;
        solution.status = SolverStatus::NumericalError;
        solution.attainable_accuracy = AccuracyStatus::NotChecked;
    } else if solution.converged && true_residual >= tolerance {
        solution.converged = false;
        solution.status = SolverStatus::Stagnated;
    }
}

/// Solve Ax = b using the Preconditioned Conjugate Gradient method
///
/// Uses a preconditioner M to accelerate convergence. At each iteration,
/// the preconditioner is applied as z = M^{-1} r, and the search direction
/// is computed using the preconditioned residual.
///
/// Note: Both A and M^{-1} must be symmetric positive definite.
pub fn pcg<T, A, P>(operator: &A, precond: &P, b: &[T], config: &CgConfig<T::Real>) -> CgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: sonium_solvers_core::traits::Preconditioner<T>,
{
    match try_pcg(operator, precond, b, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, None),
    }
}

/// Solve Ax = b using PCG with an initial guess
pub fn pcg_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &CgConfig<T::Real>,
) -> CgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: sonium_solvers_core::traits::Preconditioner<T>,
{
    match try_pcg_with_guess(operator, precond, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => invalid_input_solution(b, x0),
    }
}

fn pcg_with_guess_unchecked<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &CgConfig<T::Real>,
) -> CgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: sonium_solvers_core::traits::Preconditioner<T>,
{
    let n = b.len();
    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut op_counts = KrylovOpCounts::default();

    let ax = operator.apply(&x);
    op_counts.spmv_count += 1;
    let r: Vec<T> = sonium_solvers_core::vec_ops::vec_sub(b, &ax);
    let initial_residual_norm = vector_norm(&r);
    op_counts.norm_count += 1;

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let scaled_thresh = b_norm * T::Real::from_f64(1e-15).unwrap();
    let tol_threshold = if scaled_thresh > floor {
        scaled_thresh
    } else {
        floor
    };
    if initial_residual_norm < tol_threshold {
        return CgSolution {
            x,
            iterations: 0,
            residual: if b_norm > floor {
                initial_residual_norm / b_norm
            } else {
                T::Real::from_f64(0.0).unwrap()
            },
            estimated_residual: if b_norm > floor {
                initial_residual_norm / b_norm
            } else {
                T::Real::from_f64(0.0).unwrap()
            },
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }

    let mut r = r;
    let mut z = precond.apply(&r);
    op_counts.preconditioner_count += 1;
    let mut p = z.clone();
    let mut q = vec![T::zero(); n];
    let mut rho = inner_product(&r, &z);
    op_counts.dot_reductions += 1;
    op_counts.synchronization_points += 1;

    // Guard: if b_norm is zero, use the initial residual norm as the
    // reference to avoid division-by-zero in the relative residual.
    let ref_norm = if b_norm > floor {
        b_norm
    } else {
        initial_residual_norm
    };

    let scaled = b_norm * T::Real::from_f64(1e-20).unwrap();
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_threshold = if scaled > floor { scaled } else { floor };

    for iter in 0..config.max_iterations {
        // q = A * p (zero allocation via apply_into)
        operator.apply_into(&p, &mut q);
        op_counts.spmv_count += 1;

        let pq = inner_product(&p, &q);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if pq.norm() < breakdown_threshold {
            op_counts.norm_count += 1;
            return CgSolution {
                x,
                iterations: iter,
                residual: vector_norm(&r) / ref_norm,
                estimated_residual: vector_norm(&r) / ref_norm,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }

        let alpha = rho / pq;

        // x += alpha * p
        axpy(alpha, &p, &mut x);
        op_counts.axpy_passes += 1;

        // r -= alpha * q
        axpy(-alpha, &q, &mut r);
        op_counts.axpy_passes += 1;

        // z = M^{-1} r is needed for the fused reduction, so the
        // preconditioner now runs before the convergence test (one extra
        // application on the final iteration — the price of fusing).
        precond.apply_into(&r, &mut z);
        op_counts.preconditioner_count += 1;

        // Fused reduction: ||r|| (convergence test) and (r, z) (beta
        // numerator) in one parallel section — one barrier instead of two.
        let (r_norm, rho_new) = fused_norm_dot(&r, &z);
        op_counts.norm_count += 1;
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        let rel_residual = r_norm / ref_norm;

        if config.print_interval > 0 && (iter + 1) % config.print_interval == 0 {
            log::info!(
                "PCG iteration {}: relative residual = {:.6e}",
                iter + 1,
                rel_residual.to_f64().unwrap_or(0.0)
            );
        }

        if rel_residual < config.tolerance {
            return CgSolution {
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

        if rho_new.norm() < breakdown_threshold {
            return CgSolution {
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

        let beta = rho_new / rho;
        rho = rho_new;

        // p = z + beta * p  (in-place: scale p by beta, then add z)
        scale_inplace(&mut p, beta);
        op_counts.axpy_passes += 1;
        axpy(T::one(), &z, &mut p);
        op_counts.axpy_passes += 1;
    }

    let rel_residual = vector_norm(&r) / ref_norm;
    op_counts.norm_count += 1;
    CgSolution {
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
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn test_cg_spd() {
        // Symmetric positive definite matrix
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0],];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        let config = CgConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cg(&a, &b, &config);

        assert!(solution.converged, "CG should converge for SPD matrix");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution should satisfy Ax = b");
        assert!(solution.estimated_residual.is_finite());
        assert!(solution.residual.is_finite());
        assert!(solution.backward_error.is_finite());
        assert_eq!(solution.attainable_accuracy, AccuracyStatus::Attained);
    }

    #[test]
    fn cg_reports_independent_residual_and_backward_error_on_max_iterations() {
        let a = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]],
            1e-15,
        );
        let b = sonium_solvers_core::array![1.0_f64, 2.0];
        let config = CgConfig {
            max_iterations: 1,
            tolerance: 1e-15,
            print_interval: 0,
        };

        let solution = try_cg(&a, &b, &config).expect("finite max-iteration result");
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
    fn test_cg_identity() {
        let n = 5;
        let id: CsrMatrix<f64> = CsrMatrix::identity(n);
        let b = (1..=n).map(|i| i as f64).collect::<Vec<_>>();

        let config = CgConfig {
            max_iterations: 10,
            tolerance: 1e-12,
            print_interval: 0,
        };

        let solution = cg(&id, &b, &config);

        assert!(solution.converged);
        assert!(solution.iterations <= 2);

        let error: f64 = sonium_solvers_core::vec_ops::vec_sub(&solution.x, &b)
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-10);
    }

    #[test]
    fn test_cg_larger_tridiagonal() {
        // 20-node SPD tridiagonal system
        let n = 20;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n).map(|i| (i as f64).sin()).collect::<Vec<_>>();

        let config = CgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cg(&a, &b, &config);
        assert!(
            solution.converged,
            "CG should converge on 20-node SPD tridiagonal"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Residual too large: {}", error);
    }

    #[test]
    fn test_pcg_with_jacobi() {
        // PCG with DiagonalPreconditioner on a 10-node SPD system
        use sonium_solvers_preconditioners::DiagonalPreconditioner;

        let n = 10;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -1.0;
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = DiagonalPreconditioner::from_csr(&a);
        let b = (0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>();

        let config = CgConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = pcg(&a, &precond, &b, &config);
        assert!(solution.converged, "PCG with Jacobi should converge");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "PCG solution error too large: {}", error);
    }

    #[test]
    fn test_cg_zero_rhs() {
        // Zero RHS should converge immediately with 0 iterations
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![0.0_f64, 0.0];

        let config = CgConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cg(&a, &b, &config);
        assert!(solution.converged, "Zero RHS should converge");
        assert_eq!(solution.iterations, 0, "Zero RHS should need 0 iterations");
    }

    #[test]
    fn test_cg_initial_residual_threshold_scales_with_b_norm() {
        // Regression: the early convergence threshold must scale with ||b||.
        // Previously a fixed 1e-15 threshold meant that for ||b|| = 1e-4,
        // an initial residual of 1e-10 (relative 1e-6) would trigger "converged".
        // Now threshold = b_norm * 1e-15, so it's relative.
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);

        // b_norm = sqrt((1e-4)^2 + (2e-4)^2) ≈ 2.24e-4
        let b = sonium_solvers_core::array![1e-4_f64, 2e-4];
        let config = CgConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cg(&a, &b, &config);
        // Should converge normally (not via early convergence shortcut)
        // because the initial residual norm ≈ 1e-4 > b_norm * 1e-15
        assert!(
            solution.converged,
            "CG should converge on small ||b|| system"
        );
        assert!(
            solution.iterations > 0,
            "Should not trigger early convergence for ||b|| = 1e-4"
        );

        // The residual should be accurate
        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "Solution should satisfy Ax = b, got error {}",
            error
        );
    }

    #[test]
    fn test_cg_zero_rhs_with_guess_max_iters_residual_is_finite() {
        // Regression: the max-iterations exit divided by b_norm, which is 0
        // when b = 0 with a nonzero initial guess — yielding inf/NaN.
        // ref_norm falls back to the initial residual norm in that case.
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![0.0_f64, 0.0];
        let x0 = sonium_solvers_core::array![1.0_f64, 2.0];
        let config = CgConfig {
            max_iterations: 1,
            tolerance: 1e-15,
            print_interval: 0,
        };

        let solution = cg_with_guess(&a, &b, Some(&x0), &config);
        assert!(
            solution.residual.is_finite(),
            "residual must be finite with b = 0, got {}",
            solution.residual
        );

        use sonium_solvers_preconditioners::DiagonalPreconditioner;
        let precond = DiagonalPreconditioner::from_csr(&a);
        let solution = pcg_with_guess(&a, &precond, &b, Some(&x0), &config);
        assert!(
            solution.residual.is_finite(),
            "pcg residual must be finite with b = 0, got {}",
            solution.residual
        );
    }

    #[test]
    fn direct_cg_rejects_non_finite_input_before_operator_use() {
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

        let config = CgConfig {
            max_iterations: 4,
            tolerance: 1.0e-8,
            print_interval: 0,
        };
        let solution = cg(&PanicOperator, &[f64::NAN], &config);
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
    }

    #[test]
    fn checked_cg_contains_callback_panics_and_non_finite_outputs() {
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

        let config = CgConfig::default();
        let error = try_cg(&PanicOperator, &[1.0], &config)
            .expect_err("checked CG must contain operator panics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        #[allow(deprecated)]
        let legacy = cg(&PanicOperator, &[1.0], &config);
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

        let error = try_cg(&NonFiniteOperator, &[1.0], &config)
            .expect_err("checked CG must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_pcg_contains_preconditioner_panics_and_non_finite_outputs() {
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

        let config = CgConfig::default();
        let error = try_pcg(&IdentityOperator, &PanicPreconditioner, &[1.0], &config)
            .expect_err("checked PCG must contain preconditioner panics");
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

        let error = try_pcg(&IdentityOperator, &NonFinitePreconditioner, &[1.0], &config)
            .expect_err("checked PCG must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn test_cg_reports_krylov_op_counts() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];
        let config = CgConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cg(&a, &b, &config);

        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions >= solution.iterations);
        assert!(solution.op_counts.norm_count > 0);
        assert!(solution.op_counts.axpy_passes >= 2 * solution.iterations);
        assert_eq!(solution.op_counts.preconditioner_count, 0);
    }

    #[test]
    fn test_pcg_reports_preconditioner_counts() {
        use sonium_solvers_preconditioners::DiagonalPreconditioner;

        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = DiagonalPreconditioner::from_csr(&a);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];
        let config = CgConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = pcg(&a, &precond, &b, &config);

        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.preconditioner_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions >= solution.iterations);
        assert!(solution.op_counts.norm_count > 0);
        assert!(solution.op_counts.axpy_passes >= 2 * solution.iterations);
    }
}
