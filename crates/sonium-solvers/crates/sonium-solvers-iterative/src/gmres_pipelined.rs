//! Pipelined GMRES solver (p-GMRES)
//!
//! Implementation of the Pipelined GMRES algorithm based on Ghysels et al. (2013).
//! This solver overlaps the matrix-vector product with global reductions to hide communication latency
//! and improve scalability on parallel systems.

use crate::gmres::{GmresConfig, GmresSolution};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::vector_norm;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};

use super::cgs_orth::{
    FlatBasis, KrylovOpCounts, batched_dots_flat_into, batched_project_flat,
    fused_basis_update_flat,
};
use super::checked_callbacks::{
    CheckedLinearOperator, CheckedPreconditioner, catch_solver_callback,
};
use super::gmres_common::{exact_relative_residual, givens_rotation, solve_upper_triangular};

fn invalid_pipelined_solution<T: ComplexField>(b: &[T], x0: Option<&[T]>) -> GmresSolution<T> {
    let x = match x0 {
        Some(guess) if guess.len() == b.len() => guess.to_vec(),
        _ => vec![T::zero(); b.len()],
    };
    GmresSolution {
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
    }
}

fn validate_pipelined_inputs<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
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
                operation: "pipelined GMRES",
                reason: "initial guess contains NaN or infinity".to_string(),
            });
        }
    }
    if config.max_iterations == 0
        || config.restart == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "pipelined GMRES",
            reason: "invalid pipelined GMRES configuration".to_string(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "pipelined GMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    Ok(cols)
}

fn validate_pipelined_solution<T: ComplexField>(
    solution: &GmresSolution<T>,
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
            operation: "pipelined GMRES",
            reason: "solver generated NaN, infinity, or an invalid result".to_string(),
        });
    }
    Ok(())
}

/// Checked pipelined GMRES entry point with typed callback and generated-value
/// failures.
pub fn try_gmres_pipelined<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    catch_solver_callback("pipelined GMRES", || {
        let dimension = validate_pipelined_inputs(operator, b, x0, config)?;
        let checked_operator = CheckedLinearOperator::new(operator);
        let checked_precond = CheckedPreconditioner::new(precond);
        let solution = gmres_pipelined_core(&checked_operator, &checked_precond, b, x0, config);
        let solution =
            super::gmres::finalize_gmres_solution(&checked_operator, b, config.tolerance, solution);
        validate_pipelined_solution(&solution, dimension)?;
        SolverResultContract::validate_contract(&solution)?;
        Ok(solution)
    })
}

/// Solve Ax = b using the Pipelined GMRES method
///
/// This moves the SpMV of the *next* iteration to overlap with the dot products of the *current* iteration.
/// It requires maintaining an auxiliary basis Z = AV.
pub fn gmres_pipelined<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_gmres_pipelined(operator, precond, b, x0, config) {
        Ok(solution) => solution,
        Err(_) => invalid_pipelined_solution(b, x0),
    }
}

fn gmres_pipelined_core<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
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
        || operator.num_rows() != operator.num_cols()
        || n != operator.num_cols()
        || b.iter().any(|value| !value.norm().is_finite())
        || x0.is_some_and(|guess| {
            guess.len() != n || guess.iter().any(|value| !value.norm().is_finite())
        });
    if invalid_input {
        let x = match x0 {
            Some(guess) if guess.len() == n => guess.to_vec(),
            _ => vec![T::zero(); n],
        };
        return GmresSolution {
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

    // Initialize solution vector
    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut op_counts = KrylovOpCounts::default();

    // Compute initial residual
    // r0 = b - Ax0
    let ax = operator.apply(&x);
    op_counts.spmv_count += 1;
    let r0: Vec<T> = sonium_solvers_core::vec_ops::vec_sub(b, &ax);
    op_counts.axpy_passes += 1;

    // Apply preconditioner (Left preconditioning: M^-1 A x = M^-1 b)
    let r0 = precond.apply(&r0);
    op_counts.preconditioner_count += 1;

    // Use proper tolerance check from config or default
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    let initial_b_norm = vector_norm(&r0);
    op_counts.norm_count += 1;

    if initial_b_norm < tol_threshold {
        return GmresSolution {
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

    let mut total_iterations = 0;
    let mut restarts = 0;

    // Fixed reference norm: use the preconditioned RHS norm (matches gmres.rs).
    let b_norm = initial_b_norm;

    // Unpreconditioned RHS norm for the true-residual check on the breakdown
    // path. A near-zero RHS needs an absolute check because a relative
    // residual is undefined (matches gmres.rs).
    let raw_true_b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let true_b_norm = if raw_true_b_norm < tol_threshold {
        T::Real::one()
    } else {
        raw_true_b_norm
    };

    // Pre-allocate Krylov bases (reused across restarts)
    let mut v = FlatBasis::new(m + 1, n);
    let mut z = FlatBasis::new(m + 1, n);
    let mut op_buf = vec![T::zero(); n];
    let mut v_new_buf = vec![T::zero(); n];
    let mut coeffs: Vec<T> = Vec::with_capacity(m + 1);
    let mut reorth_coeffs: Vec<T> = Vec::with_capacity(m + 1);

    let mut h: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, T::zero());
    let mut g: Vec<T> = vec![T::zero(); m + 1];
    let mut cs: Vec<T> = Vec::with_capacity(m);
    let mut sn: Vec<T> = Vec::with_capacity(m);

    let mut p_az = vec![T::zero(); n];
    let mut p_q = vec![T::zero(); n];

    for _outer in 0..config.max_iterations {
        // Compute current residual norm and compare to reference b_norm
        operator.apply_into(&x, &mut op_buf);
        op_counts.spmv_count += 1;
        for k in 0..n {
            op_buf[k] = b[k] - op_buf[k];
        }
        op_counts.axpy_passes += 1;
        precond.apply_into(&op_buf, &mut v_new_buf);
        op_counts.preconditioner_count += 1;
        // The preconditioned norm drives the recurrence, but public stopping
        // and reporting use the original residual space.
        let true_rel_residual = vector_norm(&op_buf) / true_b_norm;
        let beta = vector_norm(&v_new_buf);
        op_counts.norm_count += 1;

        if true_rel_residual < config.tolerance {
            return GmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: true_rel_residual,
                estimated_residual: true_rel_residual,
                backward_error: true_rel_residual,
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // v[0] = r / beta
        let inv_beta = T::from_real(T::Real::one() / beta);
        {
            let v0 = v.vector_mut(0);
            for k in 0..n {
                v0[k] = v_new_buf[k] * inv_beta;
            }
        }
        op_counts.axpy_passes += 1;

        // Pre-calculate z[0] = M^{-1} A v[0]
        operator.apply_into(v.vector(0), &mut op_buf);
        op_counts.spmv_count += 1;
        precond.apply_into(&op_buf, z.vector_mut(0));
        op_counts.preconditioner_count += 1;

        h.fill(T::zero());
        g.fill(T::zero());
        cs.clear();
        sn.clear();
        g[0] = T::from_real(beta);

        let mut inner_converged = false;
        let dgks_threshold = T::Real::from_f64(0.707).unwrap();

        for j in 0..m {
            total_iterations += 1;

            let p_az_ref = &mut p_az;
            let p_q_ref = &mut p_q;
            let coeffs_ref = &mut coeffs;

            // Pipelined step: overlap the SpMV/preconditioner for the *next*
            // iteration with the batched dot-product pass of the *current* one.
            rayon::join(
                || {
                    operator.apply_into(z.vector(j), p_az_ref);
                    precond.apply_into(p_az_ref, p_q_ref);
                },
                || {
                    // Column of H: <v[i], z[j]> for i = 0..=j, batched.
                    batched_dots_flat_into(&v, j + 1, z.vector(j), coeffs_ref);
                },
            );
            op_counts.spmv_count += 1;
            op_counts.preconditioner_count += 1;
            op_counts.synchronization_points += 1;

            // Place CGS coefficients into H
            for (i, &c) in coeffs.iter().enumerate() {
                h[[i, j]] = c;
            }

            // Update v[j+1] and z[j+1] with fused projection passes:
            // v_new = z[j] - sum(h_ij * v_i)
            // z_new = q    - sum(h_ij * z_i)
            // Applying the same coefficients to both bases preserves Z = M⁻¹AV.
            v_new_buf.copy_from_slice(z.vector(j));
            op_counts.axpy_passes += 1;
            let w_norm_before = vector_norm(&v_new_buf);
            op_counts.norm_count += 1;

            batched_project_flat(&v, &coeffs, &mut v_new_buf);
            batched_project_flat(&z, &coeffs, &mut p_q);
            op_counts.add_basis_projection::<T>(j + 1, n);
            op_counts.axpy_passes += 1;
            op_counts.basis_bytes_touched += ((j + 1) * n * std::mem::size_of::<T>()) as u64;

            // DGKS reorthogonalisation: redo the projection when the
            // orthogonal complement lost too much norm in a single CGS pass
            // (classic p-GMRES instability fix, mirrors gmres.rs).
            let mut norm_v = vector_norm(&v_new_buf);
            op_counts.norm_count += 1;
            if norm_v < dgks_threshold * w_norm_before {
                batched_dots_flat_into(&v, j + 1, &v_new_buf, &mut reorth_coeffs);
                for (i, &si) in reorth_coeffs.iter().enumerate() {
                    h[[i, j]] += si;
                }
                batched_project_flat(&v, &reorth_coeffs, &mut v_new_buf);
                batched_project_flat(&z, &reorth_coeffs, &mut p_q);
                op_counts.add_basis_projection::<T>(j + 1, n);
                op_counts.axpy_passes += 1;
                op_counts.basis_bytes_touched += ((j + 1) * n * std::mem::size_of::<T>()) as u64;
                norm_v = vector_norm(&v_new_buf);
                op_counts.norm_count += 1;
            }
            h[[j + 1, j]] = T::from_real(norm_v);

            // Breakdown check
            let breakdown_tol = T::Real::from_f64(1e-20).unwrap();

            if norm_v < breakdown_tol {
                inner_converged = true;
            } else {
                // Normalize and store in pre-allocated basis
                let inv_norm = T::from_real(T::Real::one() / norm_v);
                let next_v = v.vector_mut(j + 1);
                let next_z = z.vector_mut(j + 1);
                for k in 0..n {
                    next_v[k] = v_new_buf[k] * inv_norm;
                    next_z[k] = p_q[k] * inv_norm;
                }
                op_counts.axpy_passes += 2;
            }

            // Givens rotations (same as standard GMRES)
            for i in 0..j {
                let temp = cs[i].conj() * h[[i, j]] + sn[i].conj() * h[[i + 1, j]];
                h[[i + 1, j]] = T::zero() - sn[i] * h[[i, j]] + cs[i] * h[[i + 1, j]];
                h[[i, j]] = temp;
            }

            let (c, s) = givens_rotation(h[[j, j]], h[[j + 1, j]]);
            cs.push(c);
            sn.push(s);

            h[[j, j]] = c.conj() * h[[j, j]] + s.conj() * h[[j + 1, j]];
            h[[j + 1, j]] = T::zero();

            let temp = c.conj() * g[j] + s.conj() * g[j + 1];
            g[j + 1] = T::zero() - s * g[j] + c * g[j + 1];
            g[j] = temp;

            let abs_residual = g[j + 1].norm();
            let rel_residual = abs_residual / b_norm;
            let abs_tol = T::Real::from_f64(1e-20).unwrap();

            if config.print_interval > 0 && total_iterations % config.print_interval == 0 {
                log::info!(
                    "p-GMRES iteration {} (restart {}): relative residual = {:.6e}",
                    total_iterations,
                    restarts,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }

            // Arnoldi breakdown before the Krylov estimate reached tolerance:
            // update x and measure the *true* (unpreconditioned) residual.
            // Only a happy breakdown whose true residual is already below tol
            // may report convergence (mirrors gmres.rs).
            if inner_converged {
                let y = solve_upper_triangular(&h, &g, j + 1);
                fused_basis_update_flat(&v, &y, &mut x);
                op_counts.add_basis_update::<T>(y.len(), n);

                operator.apply_into(&x, &mut op_buf);
                op_counts.spmv_count += 1;
                let mut actual_residual = T::Real::zero();
                for k in 0..n {
                    let rk = b[k] - op_buf[k];
                    actual_residual += rk.norm_sqr();
                }
                let actual_residual = actual_residual.sqrt() / true_b_norm;
                op_counts.norm_count += 1;

                if actual_residual < config.tolerance {
                    return GmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: actual_residual,
                        estimated_residual: rel_residual,
                        backward_error: actual_residual,
                        attainable_accuracy: AccuracyStatus::Attained,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    };
                }

                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: actual_residual,
                    estimated_residual: rel_residual,
                    backward_error: actual_residual,
                    attainable_accuracy: AccuracyStatus::NotAttained,
                    converged: false,
                    status: SolverStatus::Breakdown,
                    op_counts,
                };
            }

            if rel_residual < config.tolerance || abs_residual < abs_tol {
                let y = solve_upper_triangular(&h, &g, j + 1);
                fused_basis_update_flat(&v, &y, &mut x);
                op_counts.add_basis_update::<T>(y.len(), n);
                let true_rel_residual = exact_relative_residual(
                    operator,
                    &x,
                    b,
                    &mut op_buf,
                    true_b_norm,
                    &mut op_counts,
                );
                if true_rel_residual >= config.tolerance {
                    continue;
                }

                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: true_rel_residual,
                    estimated_residual: rel_residual,
                    backward_error: true_rel_residual,
                    attainable_accuracy: AccuracyStatus::Attained,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
        }

        // Restart
        let y = solve_upper_triangular(&h, &g, m);
        fused_basis_update_flat(&v, &y, &mut x);
        op_counts.add_basis_update::<T>(y.len(), n);

        // The Krylov estimate is a left-preconditioned norm and can be much
        // more pessimistic than the requested ||b-Ax||/||b|| tolerance. Check
        // the true (unpreconditioned) residual at every restart boundary
        // before spending another whole cycle (mirrors gmres.rs).
        let true_rel_residual =
            exact_relative_residual(operator, &x, b, &mut op_buf, true_b_norm, &mut op_counts);
        if true_rel_residual < config.tolerance {
            return GmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: true_rel_residual,
                estimated_residual: true_rel_residual,
                backward_error: true_rel_residual,
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        restarts += 1;
    }

    // Final check
    let ax = operator.apply(&x);
    op_counts.spmv_count += 1;
    let r = sonium_solvers_core::vec_ops::vec_sub(b, &ax);
    op_counts.axpy_passes += 1;
    let rel_residual = vector_norm(&r) / true_b_norm;
    op_counts.norm_count += 1;

    GmresSolution {
        x,
        iterations: total_iterations,
        restarts,
        residual: rel_residual,
        estimated_residual: rel_residual,
        backward_error: rel_residual,
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_preconditioners::IdentityPreconditioner;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn test_pgmres_rejects_invalid_input_before_iteration() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let precond = IdentityPreconditioner;
        let config = GmresConfig {
            max_iterations: 4,
            restart: 0,
            tolerance: 1.0e-8,
            ..GmresConfig::default()
        };
        let solution = gmres_pipelined(&matrix, &precond, &[f64::NAN, 1.0], None, &config);
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.iterations, 0);
        assert_eq!(solution.op_counts.spmv_count, 0);
    }

    #[test]
    fn checked_pgmres_contains_callback_panics_and_non_finite_outputs() {
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

        let config = GmresConfig::default();
        let error = try_gmres_pipelined(
            &PanicOperator,
            &IdentityPreconditioner,
            &[1.0],
            None,
            &config,
        )
        .expect_err("checked p-GMRES must contain operator panics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let legacy = gmres_pipelined(
            &PanicOperator,
            &IdentityPreconditioner,
            &[1.0],
            None,
            &config,
        );
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

        let error = try_gmres_pipelined(
            &NonFiniteOperator,
            &IdentityPreconditioner,
            &[1.0],
            None,
            &config,
        )
        .expect_err("checked p-GMRES must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_pgmres_contains_preconditioner_failures() {
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

        let config = GmresConfig::default();
        let error = try_gmres_pipelined(
            &IdentityOperator,
            &PanicPreconditioner,
            &[1.0],
            None,
            &config,
        )
        .expect_err("checked p-GMRES must contain preconditioner panics");
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

        let error = try_gmres_pipelined(
            &IdentityOperator,
            &NonFinitePreconditioner,
            &[1.0],
            None,
            &config,
        )
        .expect_err("checked p-GMRES must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn test_pgmres_simple_solver() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let precond = IdentityPreconditioner;

        let config = GmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);

        assert!(solution.converged, "GMRES should converge");

        // Verify solution: Ax ≈ b
        let ax = a.apply(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution should satisfy Ax = b");
    }

    #[test]
    fn test_pgmres_with_jacobi() {
        // Pipelined GMRES with Jacobi (diagonal) preconditioner
        use sonium_solvers_preconditioners::DiagonalPreconditioner;

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

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = DiagonalPreconditioner::from_csr(&a);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);
        assert!(
            solution.converged,
            "Pipelined GMRES with Jacobi should converge"
        );

        let ax = a.apply(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "p-GMRES Jacobi solution error: {}", error);
    }

    #[test]
    fn test_pgmres_f64() {
        // Real f64 system through pipelined GMRES
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IdentityPreconditioner;
        let b = vec![1.0_f64, 2.0];

        let config = GmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);
        assert!(solution.converged, "p-GMRES should converge for f64 system");

        let ax = a.apply(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "p-GMRES f64 solution error: {}", error);
    }

    #[test]
    fn test_pgmres_reports_krylov_op_counts() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IdentityPreconditioner;
        let b = vec![1.0_f64, 2.0];
        let config = GmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);

        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.preconditioner_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions > 0);
        assert!(solution.op_counts.norm_count > 0);
        assert!(solution.op_counts.axpy_passes >= solution.iterations);
        assert!(solution.op_counts.basis_bytes_touched > 0);
    }

    #[test]
    fn test_pgmres_restart_cycle() {
        // Regression: b_norm was zero on the first cycle, causing
        // division-by-zero (rel_residual = inf) and preventing early
        // convergence within the first cycle.  Force multiple cycles
        // with a tight restart on a non-trivial system.
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
        let precond = IdentityPreconditioner;
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 5,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);
        assert!(solution.converged, "p-GMRES should converge with restarts");
        assert!(
            solution.restarts > 0,
            "Should need at least one restart with restart=5 on 20-node system"
        );

        let ax = a.apply(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution error too large: {}", error);
    }

    #[test]
    fn test_pgmres_breakdown_does_not_report_converged() {
        // Regression: a pre-convergence Arnoldi breakdown used to return
        // converged: true together with status: Breakdown.
        //
        // A = diag(1, 0) is singular and b = [1, 1] is not in range(A):
        // the Krylov subspace exhausts at j = 1 (happy breakdown gives the
        // least-squares solution) but the true residual is 1/sqrt(2), far
        // above the tolerance. The solver must report Breakdown and
        // converged == false.
        let dense = sonium_solvers_core::array![[1.0_f64, 0.0], [0.0, 0.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IdentityPreconditioner;
        let b = vec![1.0_f64, 1.0];

        let config = GmresConfig {
            max_iterations: 10,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);
        assert_eq!(
            solution.status,
            SolverStatus::Breakdown,
            "pre-convergence breakdown must report Breakdown"
        );
        assert!(
            !solution.converged,
            "pre-convergence breakdown must NOT report converged"
        );
        assert!(
            solution.residual > config.tolerance,
            "reported residual {} should be above tol",
            solution.residual
        );
    }

    #[test]
    fn test_pgmres_happy_breakdown_reports_converged() {
        // Companion to the regression above: an Arnoldi breakdown where the
        // *true* residual is already below tol is a happy breakdown and may
        // report convergence. A = [[1, 1], [0, 0]], b = [1, 0]: r0 = b spans
        // the solution x = [1, 0] exactly, so the Arnoldi recurrence breaks
        // down at j = 0 with a zero true residual.
        let dense = sonium_solvers_core::array![[1.0_f64, 1.0], [0.0, 0.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IdentityPreconditioner;
        let b = vec![1.0_f64, 0.0];

        let config = GmresConfig {
            max_iterations: 10,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);
        assert!(
            solution.converged,
            "happy breakdown with zero true residual should converge"
        );
        assert_eq!(solution.status, SolverStatus::Converged);
    }

    #[test]
    fn test_pgmres_dgks_reorthogonalization_converges() {
        // Single-pass CGS loses orthogonality when the new Arnoldi vector has
        // only a tiny component orthogonal to the existing basis (the classic
        // p-GMRES instability). A diagonal matrix with one very small
        // eigenvalue triggers exactly that: after the first steps the new
        // direction is dominated by already-spanned components, so the DGKS
        // criterion (||w|| < 0.707 ||w_before||) fires and reorthogonalizes.
        let n = 20;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        dense[[0, 0]] = 2e-6;
        for i in 1..n {
            dense[[i, i]] = i as f64;
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IdentityPreconditioner;
        let b = (0..n)
            .map(|i| ((i + 1) as f64).sin() + 0.5 * ((i + 1) as f64).cos())
            .collect::<Vec<_>>();

        // cond(A) ~ 1e7, so the attainable relative residual floor is
        // ~cond * eps ~ 2e-9; 1e-8 is reachable with DGKS but not with
        // single-pass CGS (which stagnates orders of magnitude earlier).
        let config = GmresConfig {
            // Permit a full Arnoldi basis. Ten steps happened to cross the
            // tolerance on some targets, but cannot robustly resolve all 20
            // distinct diagonal modes of this fixture.
            max_iterations: n,
            restart: n,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);
        assert!(
            solution.converged,
            "p-GMRES with DGKS reorthogonalization should converge (residual {})",
            solution.residual
        );

        let ax = a.apply(&solution.x);
        let b_norm: f64 = b.iter().map(|e| e * e).sum::<f64>().sqrt();
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(
            error / b_norm < 1e-7,
            "relative solution error too large: {}",
            error / b_norm
        );

        // Prove the DGKS path actually fired: without reorthogonalization a
        // full first cycle of `iters` iterations performs exactly
        // iters*(iters+1)/2 dot reductions; each reorthogonalization pass adds
        // another j+1.
        if solution.restarts == 0 {
            let iters = solution.iterations;
            let single_pass_dots = iters * (iters + 1) / 2;
            assert!(
                solution.op_counts.dot_reductions > single_pass_dots,
                "expected DGKS reorthogonalization to add dot reductions ({} <= {})",
                solution.op_counts.dot_reductions,
                single_pass_dots
            );
        }
    }

    #[test]
    fn test_pgmres_restart_accepts_requested_true_residual() {
        // Regression: the restart boundary must check the true
        // (unpreconditioned) residual, not only the left-preconditioned
        // Krylov estimate (mirrors gmres.rs). After one GMRES(1) cycle the
        // true relative residual is about 1.4e-4 while the preconditioned
        // estimate remains about 1e-2; before the fix p-GMRES exhausted the
        // restart budget without ever accepting the solution.
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 100.0 * r[1];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0e-6];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1.0e-3,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
            print_interval: 0,
        };

        let solution = gmres_pipelined(&a, &AnisotropicPreconditioner, &b, None, &config);

        assert!(
            solution.converged,
            "true residual met the requested tolerance at restart: residual={:.3e}",
            solution.residual
        );
        assert!(solution.residual < config.tolerance);
        assert_eq!(solution.iterations, 1);
        assert!(
            solution.op_counts.exact_residual_checks > 0,
            "restart boundary must perform an exact residual check"
        );
    }

    #[test]
    fn test_pgmres_restart_true_residual_tight_tolerance() {
        // Attainable-accuracy regression: ill-conditioned complex
        // non-Hermitian system with small restart, solved to <1e-10 relative.
        // The reported residual must be backed by the true residual
        // ||b - A*x||/||b||.
        let n = 60;
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
        let precond = IdentityPreconditioner;
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 20,
            tolerance: 1e-11,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
            print_interval: 0,
        };

        let solution = gmres_pipelined(&a, &precond, &b, None, &config);
        assert!(
            solution.converged,
            "p-GMRES should converge to tight tolerance, residual = {}",
            solution.residual
        );
        assert!(
            solution.restarts > 0,
            "test must exercise restart boundaries (restarts = {})",
            solution.restarts
        );

        let ax = a.apply(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            error / b_norm < 1e-10,
            "true relative residual too large: {}",
            error / b_norm
        );
    }
}
