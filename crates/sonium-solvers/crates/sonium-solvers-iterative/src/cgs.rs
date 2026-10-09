//! CGS (Conjugate Gradient Squared) solver
//!
//! CGS is a Krylov subspace method for non-symmetric systems.
//! It can converge faster than BiCG but may be less stable.

use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, scale_inplace, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, SolverError, SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::cgs_orth::KrylovOpCounts;

/// CGS solver configuration
#[derive(Debug, Clone)]
pub struct CgsConfig<R> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
}

impl Default for CgsConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        }
    }
}

/// CGS solver result
#[derive(Debug)]
pub struct CgsSolution<T: ComplexField> {
    /// Solution vector
    pub x: Vec<T>,
    /// Number of iterations
    pub iterations: usize,
    /// Final relative residual
    pub residual: T::Real,
    /// Cheap recurrence residual estimate captured by the CGS iteration.
    pub estimated_residual: T::Real,
    /// Normwise backward error against the original operator.
    pub backward_error: T::Real,
    /// Whether the returned iterate satisfies the configured true-residual tolerance.
    pub attainable_accuracy: AccuracyStatus,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Solver status
    pub status: SolverStatus,
    /// Memory-traffic oriented operation counts for the Krylov solve.
    pub op_counts: KrylovOpCounts,
}

fn numerical_error_solution<T: ComplexField>(n: usize) -> CgsSolution<T> {
    CgsSolution {
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

fn finite_slice<T: ComplexField>(values: &[T]) -> bool {
    values.iter().all(|value| value.norm().is_finite())
}

fn apply_into_checked<T, A>(operator: &A, x: &[T], y: &mut [T]) -> bool
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if catch_unwind(AssertUnwindSafe(|| operator.apply_into(x, y))).is_err() {
        return false;
    }
    finite_slice(y)
}

/// Checked CGS entry point.
///
/// The historical CGS API is retained for compatibility, but checked callers
/// receive a typed failure when validation fails or the infallible iteration
/// path detects a generated non-finite value or callback failure.
pub fn try_cgs<T, A>(
    operator: &A,
    b: &[T],
    config: &CgsConfig<T::Real>,
) -> Result<CgsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let result = catch_unwind(AssertUnwindSafe(|| {
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
        if config.max_iterations == 0
            || !config.tolerance.is_finite()
            || config.tolerance <= T::Real::zero()
        {
            return Err(SolverError::OperationFailed {
                operation: "CGS",
                reason: "invalid CGS configuration".to_string(),
            });
        }
        if b.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "CGS",
                reason: "right-hand side contains NaN or infinity".to_string(),
            });
        }

        #[allow(deprecated)]
        let solution = cgs(operator, b, config);
        validate_cgs_solution(&solution, cols)?;
        SolverResultContract::validate_contract(&solution)?;
        Ok(solution)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "CGS",
        reason: panic_reason(payload),
    })??;

    Ok(result)
}

fn validate_cgs_solution<T: ComplexField>(
    solution: &CgsSolution<T>,
    expected_dimension: usize,
) -> Result<(), SolverError> {
    if solution.status == SolverStatus::NumericalError
        || solution.x.len() != expected_dimension
        || solution.x.iter().any(|value| !value.norm().is_finite())
        || !solution.residual.is_finite()
        || !solution.estimated_residual.is_finite()
        || !solution.backward_error.is_finite()
    {
        return Err(SolverError::OperationFailed {
            operation: "CGS",
            reason: "solver generated NaN, infinity, or an invalid result".to_string(),
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
        "CGS callback panicked".to_string()
    }
}

fn cgs_unfinalized<T, A>(operator: &A, b: &[T], config: &CgsConfig<T::Real>) -> CgsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
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
    let mut x = vec![T::zero(); n];
    let mut op_counts = KrylovOpCounts::default();

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    if !b_norm.is_finite() {
        return numerical_error_solution(n);
    }
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        return CgsSolution {
            x,
            iterations: 0,
            residual: T::Real::zero(),
            estimated_residual: T::Real::zero(),
            backward_error: T::Real::zero(),
            attainable_accuracy: AccuracyStatus::Attained,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }

    // Initial residual
    let mut r = b.to_vec();
    let r0 = r.clone(); // Shadow residual

    let mut rho = inner_product(&r0, &r);
    op_counts.dot_reductions += 1;
    op_counts.synchronization_points += 1;
    if !rho.norm().is_finite() {
        return numerical_error_solution(n);
    }
    let mut p = r.clone();
    let mut u = r.clone();
    let mut q = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];
    let mut tmp = vec![T::zero(); n];

    for iter in 0..config.max_iterations {
        // v = A * p (zero allocation via apply_into)
        if !apply_into_checked(operator, &p, &mut v) {
            return numerical_error_solution(n);
        }
        op_counts.spmv_count += 1;

        let sigma = inner_product(&r0, &v);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if !sigma.norm().is_finite() {
            return numerical_error_solution(n);
        }
        if sigma.norm() < T::Real::from_f64(1e-20).unwrap() {
            let residual = vector_norm(&r) / b_norm;
            op_counts.norm_count += 1;
            return CgsSolution {
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

        let alpha = rho / sigma;
        if !alpha.norm().is_finite() {
            return numerical_error_solution(n);
        }

        // q = u - alpha * v
        q.copy_from_slice(&u);
        axpy(-alpha, &v, &mut q);
        op_counts.axpy_passes += 2;
        if !finite_slice(&q) {
            return numerical_error_solution(n);
        }

        // tmp = u + q  (reuse tmp buffer for u+q)
        tmp.copy_from_slice(&u);
        axpy(T::one(), &q, &mut tmp);
        op_counts.axpy_passes += 2;
        if !finite_slice(&tmp) {
            return numerical_error_solution(n);
        }

        // w = A * tmp  (where tmp = u + q, zero allocation via apply_into)
        if !apply_into_checked(operator, &tmp, &mut w) {
            return numerical_error_solution(n);
        }
        op_counts.spmv_count += 1;

        // x += alpha * tmp  (where tmp = u + q)
        axpy(alpha, &tmp, &mut x);
        op_counts.axpy_passes += 1;

        // r -= alpha * w
        axpy(-alpha, &w, &mut r);
        op_counts.axpy_passes += 1;
        if !finite_slice(&x) || !finite_slice(&r) {
            return numerical_error_solution(n);
        }

        let rel_residual = vector_norm(&r) / b_norm;
        op_counts.norm_count += 1;
        if !rel_residual.is_finite() {
            return numerical_error_solution(n);
        }

        if config.print_interval > 0 && (iter + 1) % config.print_interval == 0 {
            log::info!(
                "CGS iteration {}: relative residual = {:.6e}",
                iter + 1,
                rel_residual.to_f64().unwrap_or(0.0)
            );
        }

        if rel_residual < config.tolerance {
            return CgsSolution {
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

        let rho_new = inner_product(&r0, &r);
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        if !rho_new.norm().is_finite() {
            return numerical_error_solution(n);
        }
        if rho_new.norm() < T::Real::from_f64(1e-20).unwrap() {
            return CgsSolution {
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
        if !beta.norm().is_finite() {
            return numerical_error_solution(n);
        }
        rho = rho_new;

        // u = r + beta * q
        u.copy_from_slice(&r);
        axpy(beta, &q, &mut u);
        op_counts.axpy_passes += 2;

        // p = u + beta * (q + beta * p)
        //   = u + beta*q + beta^2*p
        // In-place: p *= beta^2, then p += beta*q, then p += u
        scale_inplace(&mut p, beta * beta);
        axpy(beta, &q, &mut p);
        axpy(T::one(), &u, &mut p);
        op_counts.axpy_passes += 3;
        if !finite_slice(&u) || !finite_slice(&p) {
            return numerical_error_solution(n);
        }
    }

    let rel_residual = vector_norm(&r) / b_norm;
    op_counts.norm_count += 1;
    if !rel_residual.is_finite() || !finite_slice(&x) {
        return numerical_error_solution(n);
    }
    CgsSolution {
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

fn finalize_cgs_solution<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut result: CgsSolution<T>,
) -> CgsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if result.status == SolverStatus::NumericalError {
        return result;
    }
    if result.x.len() != operator.num_cols()
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
    if !apply_into_checked(operator, &result.x, &mut ax) {
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        result.converged = false;
        result.status = SolverStatus::NumericalError;
        return result;
    }
    result.op_counts.spmv_count += 1;

    let mut residual_norm_sq = T::Real::zero();
    let mut ax_norm_sq = T::Real::zero();
    let mut rhs_norm_sq = T::Real::zero();
    for (&rhs, &applied) in b.iter().zip(&ax) {
        let residual = rhs - applied;
        if !residual.norm().is_finite() || !applied.norm().is_finite() {
            result.residual = T::Real::infinity();
            result.backward_error = T::Real::infinity();
            result.attainable_accuracy = AccuracyStatus::NotChecked;
            result.converged = false;
            result.status = SolverStatus::NumericalError;
            return result;
        }
        residual_norm_sq += residual.norm_sqr();
        ax_norm_sq += applied.norm_sqr();
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
        return result;
    }

    if result.residual <= tolerance {
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

/// Solve `A x = b` using CGS with independently recomputed final accuracy metrics.
#[deprecated(note = "legacy: superseded by BiCGSTAB/IDR(s); kept for API compatibility")]
pub fn cgs<T, A>(operator: &A, b: &[T], config: &CgsConfig<T::Real>) -> CgsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    finalize_cgs_solution(
        operator,
        b,
        config.tolerance,
        cgs_unfinalized(operator, b, config),
    )
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn cgs_max_iteration_reports_true_residual_and_backward_error() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0_f64),
                (0, 1, -1.0),
                (1, 0, 2.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 1, -2.0),
                (2, 2, 5.0),
            ],
        );
        let rhs = [1.0, -2.0, 3.0];
        let config = CgsConfig {
            max_iterations: 1,
            tolerance: 1.0e-14,
            print_interval: 0,
        };

        let solution = cgs(&matrix, &rhs, &config);
        let ax = matrix.matvec(&solution.x);
        let residual_norm = ax
            .iter()
            .zip(rhs)
            .map(|(applied, expected)| (expected - applied).powi(2))
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
    fn cgs_rejects_non_finite_input_before_operator_use() {
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

        let config = CgsConfig::default();
        let solution = cgs(&PanicOperator, &[f64::NAN], &config);
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn cgs_rejects_non_finite_operator_output() {
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

        let solution = cgs(&NonFiniteOperator, &[1.0], &CgsConfig::default());
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
    }

    #[test]
    fn try_cgs_rejects_non_finite_operator_output() {
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

        let error = try_cgs(&NonFiniteOperator, &[1.0], &CgsConfig::default())
            .expect_err("checked CGS must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn try_cgs_rejects_non_finite_result_metrics() {
        let solution = CgsSolution {
            x: vec![1.0_f64],
            iterations: 1,
            residual: 0.0,
            estimated_residual: 0.0,
            backward_error: f64::INFINITY,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Converged,
            op_counts: KrylovOpCounts::default(),
        };

        let error = validate_cgs_solution(&solution, 1)
            .expect_err("checked CGS must reject non-finite result metrics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn try_cgs_contains_operator_panics() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator callback failed");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator transpose callback failed");
            }
        }

        let error = try_cgs(&PanicOperator, &[1.0], &CgsConfig::default())
            .expect_err("checked CGS must contain operator panics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        #[allow(deprecated)]
        let legacy = cgs(&PanicOperator, &[1.0], &CgsConfig::default());
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn test_cgs_simple() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = CgsConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cgs(&a, &b, &config);

        assert!(solution.converged, "CGS should converge");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution should satisfy Ax = b");
    }

    #[test]
    fn test_cgs_nonsymmetric() {
        // Non-symmetric 3x3 system
        let dense = sonium_solvers_core::array![
            [
                Complex64::new(5.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 0.0),
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

        let config = CgsConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cgs(&a, &b, &config);
        assert!(
            solution.converged,
            "CGS should converge for non-symmetric system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution error too large: {}", error);
    }

    #[test]
    fn test_cgs_f64() {
        // Real (f64) system instead of Complex64
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        let config = CgsConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cgs(&a, &b, &config);
        assert!(solution.converged, "CGS should converge for f64 system");

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "f64 solution error too large: {}", error);
    }

    #[test]
    fn test_cgs_reports_krylov_op_counts() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let config = CgsConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cgs(&a, &b, &config);

        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions >= solution.iterations);
        assert!(solution.op_counts.norm_count > 0);
        assert!(solution.op_counts.axpy_passes >= solution.iterations);
        assert_eq!(solution.op_counts.preconditioner_count, 0);
    }

    #[test]
    fn test_cgs_complex_hermitian_inner_product() {
        // Complex non-Hermitian system. BiCG/CGS uses the scalar inner product
        // r_tilde^H r; omitting the conjugation destroys the complex
        // bi-orthogonality recurrence and stalls on this system.
        let n = 100;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(3.0, 0.1);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(1.0, 0.3);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-0.1, 0.7);
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).cos(), (i as f64).sin()))
            .collect::<Vec<_>>();

        let config = CgsConfig {
            max_iterations: 2000,
            tolerance: 1e-8,
            print_interval: 0,
        };

        let solution = cgs(&a, &b, &config);
        assert!(
            solution.converged,
            "CGS with the Hermitian bi-Lanczos inner product should converge on complex systems, \
             status = {:?}, residual = {}",
            solution.status, solution.residual
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            error / b_norm < 1e-6,
            "Complex CGS true residual too large: {}",
            error / b_norm
        );
    }

    #[test]
    fn test_cgs_zero_rhs() {
        // Zero RHS should converge immediately with 0 iterations
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

        let config = CgsConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };

        let solution = cgs(&a, &b, &config);
        assert!(solution.converged, "Zero RHS should converge");
        assert_eq!(solution.iterations, 0, "Zero RHS should need 0 iterations");
    }
}
