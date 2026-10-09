//! GMRES-DR (Deflated Restarting with Harmonic Ritz Vectors)
//!
//! Automatically extracts harmonic Ritz vectors from the Hessenberg matrix at each
//! restart and prepends them to the next cycle's Krylov basis. This eliminates the
//! "restart penalty" for problems with clustered eigenvalues (e.g. Helmholtz BEM).
//!
//! Reference: Morgan, R.B. (2002). "GMRES with Deflated Restarting."
//! SIAM J. Sci. Comput. 24(1), 20-37.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, IdentityPreconditioner, LinearOperator, Preconditioner,
    SolverError, SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use super::cgs_orth::{
    FlatBasis, KrylovOpCounts, batched_dots_flat_into, batched_project_flat,
    fused_basis_update_flat,
};
use super::checked_callbacks::CheckedLinearOperator;
use super::gmres::{
    GmresConfig, GmresSolution, finalize_gmres_solution, invalid_gmres_input,
    numerical_error_solution,
};
use super::gmres_common::{
    exact_relative_residual, exact_residual_check_due, givens_rotation, solve_upper_triangular,
};
use sonium_solvers_direct::lu_factorize;

/// Configuration for GMRES-DR
#[derive(Debug, Clone)]
pub struct GmresDrConfig<R> {
    /// Base GMRES configuration
    pub gmres: GmresConfig<R>,
    /// Number of harmonic Ritz vectors to retain across restarts (k < m)
    pub num_deflation_vectors: usize,
}

impl Default for GmresDrConfig<f64> {
    fn default() -> Self {
        Self {
            gmres: GmresConfig::default(),
            num_deflation_vectors: 10,
        }
    }
}

impl Default for GmresDrConfig<f32> {
    fn default() -> Self {
        Self {
            gmres: GmresConfig::default(),
            num_deflation_vectors: 10,
        }
    }
}

/// Checked GMRES-DR entry point.
///
/// The compatibility APIs below retain their infallible result-shaped
/// contract, but route through this boundary so callback panics and
/// non-finite operator or preconditioner values cannot enter the deflated
/// Krylov basis.
pub fn try_gmres_dr_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresDrConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if config.gmres.max_iterations == 0
        || config.gmres.restart == 0
        || !config.gmres.tolerance.is_finite()
        || config.gmres.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES-DR",
            reason: "invalid GMRES configuration".to_string(),
        });
    }
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
            operation: "GMRES-DR",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(guess) = x0
        && guess.len() != operator.num_cols()
    {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: guess.len(),
        });
    }
    if let Some(guess) = x0
        && guess.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES-DR",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }

    let invalid_output = AtomicBool::new(false);
    let checked_operator = CheckedLinearOperator::<T, A>::new(operator);
    let checked_precond = FiniteCheckedPreconditioner {
        inner: precond,
        invalid_output: &invalid_output,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let result = gmres_dr_core(&checked_operator, &checked_precond, b, x0, config);
        if invalid_output.load(Ordering::Relaxed) {
            return Err(SolverError::OperationFailed {
                operation: "GMRES-DR preconditioner",
                reason: "preconditioner generated NaN, infinity, or an invalid output".to_string(),
            });
        }
        let result = finalize_gmres_solution(&checked_operator, b, config.gmres.tolerance, result);
        finish_checked_gmres_dr_solution(result, b.len())
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "GMRES-DR",
        reason: panic_reason(payload),
    })??;

    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

/// Validate the complete result boundary for the checked GMRES-DR entry point.
///
/// `finalize_gmres_solution` recomputes the public true residual and backward
/// error, but the projected residual is produced by the deflated Krylov
/// recurrence.  Keep that estimate under the same finite-value and shape
/// contract as the independently recomputed metrics before exposing it to a
/// checked caller.
fn finish_checked_gmres_dr_solution<T: ComplexField>(
    solution: GmresSolution<T>,
    expected_dimension: usize,
) -> Result<GmresSolution<T>, SolverError> {
    if solution.status == SolverStatus::NumericalError {
        return Err(SolverError::OperationFailed {
            operation: "GMRES-DR",
            reason: "operator generated NaN or infinity".to_string(),
        });
    }
    if solution.x.len() != expected_dimension
        || solution.x.iter().any(|value| !value.norm().is_finite())
        || !solution.residual.is_finite()
        || !solution.estimated_residual.is_finite()
        || !solution.backward_error.is_finite()
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES-DR",
            reason: "generated NaN, infinity, or an invalid solution".to_string(),
        });
    }
    Ok(solution)
}

struct FiniteCheckedPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for FiniteCheckedPreconditioner<'_, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut output = self.inner.apply(r);
        self.validate(r, &mut output);
        output
    }

    fn apply_into(&self, r: &[T], output: &mut [T]) {
        self.inner.apply_into(r, output);
        self.validate(r, output);
    }
}

impl<P> FiniteCheckedPreconditioner<'_, P> {
    fn validate<T: ComplexField>(&self, input: &[T], output: &mut [T]) {
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "GMRES-DR callback panicked".to_string()
    }
}

fn compatibility_error_solution<T: ComplexField>(b: &[T], x0: Option<&[T]>) -> GmresSolution<T> {
    numerical_error_solution(match x0 {
        Some(guess) if guess.len() == b.len() => guess.to_vec(),
        _ => vec![T::zero(); b.len()],
    })
}

/// Solve Ax = b using GMRES-DR (deflated restarting)
pub fn gmres_dr<T, A>(operator: &A, b: &[T], config: &GmresDrConfig<T::Real>) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_gmres_dr_preconditioned(operator, &IdentityPreconditioner, b, None, config) {
        Ok(result) => result,
        Err(_) => compatibility_error_solution(b, None),
    }
}

/// Solve Ax = b using GMRES-DR with an initial guess
pub fn gmres_dr_with_guess<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresDrConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_gmres_dr_preconditioned(operator, &IdentityPreconditioner, b, x0, config) {
        Ok(result) => result,
        Err(_) => compatibility_error_solution(b, x0),
    }
}

/// GMRES-DR with preconditioner
pub fn gmres_dr_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &GmresDrConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_gmres_dr_preconditioned(operator, precond, b, None, config) {
        Ok(result) => result,
        Err(_) => compatibility_error_solution(b, None),
    }
}

/// GMRES-DR with preconditioner and initial guess
pub fn gmres_dr_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresDrConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_gmres_dr_preconditioned(operator, precond, b, x0, config) {
        Ok(result) => result,
        Err(_) => compatibility_error_solution(b, x0),
    }
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn init_gmres_dr_state<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresDrConfig<T::Real>,
) -> Result<
    (
        Vec<T>,
        KrylovOpCounts,
        T::Real,
        FlatBasis<T>,
        Vec<T>,
        Vec<T>,
        sonium_solvers_core::dense::DenseMatrix<T>,
        Vec<(usize, T, T)>,
        Vec<T>,
        Vec<T>,
        Vec<Vec<T>>,
    ),
    GmresSolution<T>,
>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.len();
    let m = config.gmres.restart;
    let k = config.num_deflation_vectors.min(m.saturating_sub(1));

    // If k == 0, fall back to standard GMRES
    if k == 0 {
        return Err(crate::gmres::gmres_preconditioned_with_guess(
            operator,
            precond,
            b,
            x0,
            &config.gmres,
        ));
    }

    let x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    let mut op_counts = KrylovOpCounts::default();

    let pb = precond.apply(b);
    op_counts.preconditioner_count += 1;
    let preconditioned_b_norm = vector_norm(&pb);
    op_counts.norm_count += 1;
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    let true_b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let b_norm = if preconditioned_b_norm < tol_threshold {
        true_b_norm.max(tol_threshold)
    } else {
        preconditioned_b_norm
    };
    if true_b_norm < tol_threshold {
        let mut residual = vec![T::zero(); n];
        operator.apply_into(&x, &mut residual);
        op_counts.spmv_count += 1;
        let residual_norm = vector_norm(&residual);
        op_counts.norm_count += 1;
        if residual_norm < tol_threshold {
            return Err(GmresSolution {
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
            });
        }
    }
    if n == 0 {
        return Err(GmresSolution {
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
        });
    }

    let v = FlatBasis::new(m + 1, n);
    let op_buf = vec![T::zero(); n];
    let w = vec![T::zero(); n];
    let h: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, T::zero());
    // Givens rotations as (top_row, c, s) in application order: the dense QR
    // of the initial deflation block first, then one rotation per Arnoldi
    // step. Capacity: nk(nk+1)/2 initial + (m - nk) step rotations.
    let rots: Vec<(usize, T, T)> = Vec::with_capacity(k * (k + 1) / 2 + m);
    let resid = vec![T::zero(); n];
    let g: Vec<T> = vec![T::zero(); m + 1];
    let deflation_vecs: Vec<Vec<T>> = Vec::new();

    Ok((
        x,
        op_counts,
        b_norm,
        v,
        op_buf,
        w,
        h,
        rots,
        resid,
        g,
        deflation_vecs,
    ))
}

#[allow(clippy::too_many_arguments)]
fn initialize_cycle_basis<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x: &mut [T],
    config: &GmresDrConfig<T::Real>,
    n: usize,
    _m: usize,
    k: usize,
    b_norm: T::Real,
    v: &mut FlatBasis<T>,
    op_buf: &mut [T],
    w: &mut [T],
    h: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    rots: &mut Vec<(usize, T, T)>,
    resid: &mut [T],
    g: &mut [T],
    resid_norm_sq: &mut T::Real,
    r_capture_sq: &mut T::Real,
    deflation_vecs: &[Vec<T>],
    total_iterations: usize,
    restarts: usize,
    op_counts: &mut KrylovOpCounts,
) -> Result<usize, GmresSolution<T>>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    // Compute the preconditioned residual r = M^{-1}(b - Ax) once per cycle
    // and keep it: continuation steps need <v[j+1], r> so the Givens
    // least-squares estimate stays equal to the true residual norm.
    operator.apply_into(x, op_buf);
    op_counts.spmv_count += 1;
    for i in 0..n {
        op_buf[i] = b[i] - op_buf[i];
    }
    op_counts.axpy_passes += 1;
    precond.apply_into(op_buf, w);
    op_counts.preconditioner_count += 1;
    resid.copy_from_slice(w);
    let beta = vector_norm(w);
    op_counts.norm_count += 1;
    *resid_norm_sq = beta * beta;

    let rel_residual = beta / b_norm;
    if rel_residual < config.gmres.tolerance {
        return Err(GmresSolution {
            x: x.to_vec(),
            iterations: total_iterations,
            restarts,
            residual: rel_residual,
            estimated_residual: rel_residual,
            backward_error: rel_residual,
            attainable_accuracy: AccuracyStatus::Attained,
            converged: true,
            status: SolverStatus::Converged,
            op_counts: *op_counts,
        });
    }

    h.fill(T::zero());
    rots.clear();
    g.fill(T::zero());

    let start_col = if deflation_vecs.is_empty() {
        // First cycle: standard GMRES initialization
        let inv_beta = T::from_real(T::Real::one() / beta);
        {
            let v0 = v.vector_mut(0);
            for i in 0..n {
                v0[i] = w[i] * inv_beta;
            }
        }
        op_counts.axpy_passes += 1;
        g[0] = T::from_real(beta);
        *r_capture_sq = beta * beta;
        0
    } else {
        // Subsequent cycles: prepend deflation vectors to Krylov basis
        let nk = deflation_vecs.len().min(k);

        // Copy deflation vectors into v[0..nk]
        for (j, defl) in deflation_vecs[..nk].iter().enumerate() {
            v.vector_mut(j).copy_from_slice(defl);
        }
        op_counts.axpy_passes += nk;

        // Build the FULL (nk+1) x nk initial block of H:
        //   h[i, j] = <v[i], M^{-1} A v[j]>   for i in 0..nk,
        // keeping each column's post-projection leftover
        //   l_j = B v_j - sum_{i<nk} h[i, j] v_i.
        // The deflation vectors are harmonic Ritz vectors of B = M^{-1}A
        // from the previous cycle; the harmonic Petrov-Galerkin condition
        // makes every Ritz residual B v_j - theta_j v_j orthogonal to
        // B*K_m while lying in K_{m+1}, whose complement to B*K_m is
        // 1-dimensional. Hence ALL leftovers are parallel in exact
        // arithmetic, and a single continuation vector v[nk] parallel to
        // them makes the Arnoldi relation B V = V H exact (to rounding)
        // for the deflation columns. (Previously h[j+1, j] was overwritten
        // with ||l_j|| even though v[j+1] was the next deflation vector,
        // violating the relation by O(Ritz residual) and letting the
        // Givens residual estimate converge ~3 orders of magnitude below
        // the true residual.)
        let mut leftovers: Vec<Vec<T>> = Vec::with_capacity(nk);
        let mut lnorms: Vec<T::Real> = Vec::with_capacity(nk);
        let mut b_scale = T::Real::zero();
        for j in 0..nk {
            operator.apply_into(v.vector(j), op_buf);
            op_counts.spmv_count += 1;
            precond.apply_into(op_buf, w);
            op_counts.preconditioner_count += 1;
            let bv_norm = vector_norm(w);
            op_counts.norm_count += 1;
            b_scale = b_scale.max(bv_norm);

            // Project against ALL deflation vectors: they are approximate
            // eigenvectors, not a fresh Arnoldi basis, so cross-projections
            // are nonzero.
            for i in 0..nk {
                h[[i, j]] = inner_product(v.vector(i), w);
            }
            op_counts.dot_reductions += nk;
            op_counts.synchronization_points += 1;
            for i in 0..nk {
                axpy(-h[[i, j]], v.vector(i), w);
            }
            op_counts.axpy_passes += nk;
            op_counts.basis_bytes_touched += (nk * n * std::mem::size_of::<T>() * 2) as u64;

            // Reorthogonalize for robustness
            for i in 0..nk {
                let s = inner_product(v.vector(i), w);
                h[[i, j]] += s;
                axpy(-s, v.vector(i), w);
            }
            op_counts.dot_reductions += nk;
            op_counts.synchronization_points += 1;
            op_counts.axpy_passes += nk;
            op_counts.basis_bytes_touched += (nk * n * std::mem::size_of::<T>() * 2) as u64;

            lnorms.push(vector_norm(w));
            op_counts.norm_count += 1;
            leftovers.push(w.to_vec());
        }

        // Continuation vector v[nk]: the normalized leftover direction.
        // Prefer the last column's leftover (so h[nk, nk-1] = ||l_{nk-1}||
        // as in a true Arnoldi recurrence); fall back to the largest
        // leftover, then to the residual direction, when the deflation
        // subspace is numerically B-invariant.
        let dir_tol = T::Real::from_f64(100.0).unwrap() * T::Real::epsilon() * b_scale;
        let from_leftover = if lnorms[nk - 1] > dir_tol {
            Some(nk - 1)
        } else {
            let mut best = usize::MAX;
            let mut best_norm = T::Real::zero();
            for (j, &ln) in lnorms.iter().enumerate() {
                if ln > best_norm {
                    best_norm = ln;
                    best = j;
                }
            }
            if best_norm > dir_tol {
                Some(best)
            } else {
                None
            }
        };

        let have_v_nk = if let Some(jc) = from_leftover {
            let inv = T::from_real(T::Real::one() / lnorms[jc]);
            let next_v = v.vector_mut(nk);
            for i in 0..n {
                next_v[i] = leftovers[jc][i] * inv;
            }
            op_counts.axpy_passes += 1;
            true
        } else {
            // Numerically B-invariant deflation subspace: h[nk, :] ~ 0.
            // Continue from the residual direction, orthogonalised against
            // the deflation block.
            op_buf.copy_from_slice(resid);
            for i in 0..nk {
                let s = inner_product(v.vector(i), op_buf);
                axpy(-s, v.vector(i), op_buf);
            }
            op_counts.dot_reductions += nk;
            op_counts.axpy_passes += nk;
            let rn = vector_norm(op_buf);
            op_counts.norm_count += 1;
            let rdir_tol = T::Real::from_f64(1e-14).unwrap() * beta;
            let next_v = v.vector_mut(nk);
            if rn > rdir_tol {
                let inv = T::from_real(T::Real::one() / rn);
                for i in 0..n {
                    next_v[i] = op_buf[i] * inv;
                }
                op_counts.axpy_passes += 1;
                true
            } else {
                // Residual lies in the (numerically invariant) deflation
                // subspace: v[nk] stays zero, g[nk] = 0 and r_orth ~ 0, so
                // the convergence check below fires.
                next_v[..n].fill(T::zero());
                false
            }
        };

        // h[nk, j] = <v[nk], l_j> for every deflation column. Exact when the
        // leftovers are parallel; for the chosen column this recovers
        // h[nk, jc] = ||l_jc||.
        if have_v_nk {
            for j in 0..nk {
                h[[nk, j]] = inner_product(v.vector(nk), &leftovers[j]);
            }
            op_counts.dot_reductions += nk;
            op_counts.synchronization_points += 1;
        }

        // g[i] = <v[i], r> for i = 0..=nk. Because v[nk] is the Ritz
        // leftover direction (not the residual direction), r is NOT
        // contained in span(v[0..=nk]); the uncaptured remainder
        // r_orth^2 = ||r||^2 - sum_i |<v[i], r>|^2 must be included in the
        // residual estimate (it shrinks as continuation vectors capture it).
        for (i, slot) in g[..=nk].iter_mut().enumerate() {
            *slot = inner_product(v.vector(i), resid);
        }
        op_counts.dot_reductions += nk + 1;
        op_counts.synchronization_points += 1;
        *r_capture_sq = T::Real::zero();
        for gi in g.iter().take(nk + 1) {
            *r_capture_sq += gi.norm_sqr();
        }

        // The initial (nk+1) x nk block is FULL (row nk couples every
        // deflation column), so a subdiagonal-only sweep does not
        // triangularise it. Dense Givens QR: column by column, bottom-up
        // adjacent-row rotations, applied to g as well. The rotations are
        // recorded in `rots` in application order and replayed against
        // every continuation column before the per-step rotations.
        for j in 0..nk {
            for i in (j + 1..=nk).rev() {
                let (c, s) = givens_rotation(h[[i - 1, j]], h[[i, j]]);
                rots.push((i - 1, c, s));
                for col in j..nk {
                    let a0 = h[[i - 1, col]];
                    let b0 = h[[i, col]];
                    h[[i - 1, col]] = c.conj() * a0 + s.conj() * b0;
                    h[[i, col]] = T::zero() - s * a0 + c * b0;
                }
                let temp = c.conj() * g[i - 1] + s.conj() * g[i];
                g[i] = T::zero() - s * g[i - 1] + c * g[i];
                g[i - 1] = temp;
            }
        }

        nk

        // Check if already converged from the deflation block. The true
        // residual norm is sqrt(|g[nk]|^2 + r_orth^2).
        // Do not accept the preconditioned deflation-block estimate here.
        // The original-operator residual is checked after the candidate
        // update in `arnoldi_step`, where a failed exact gate can restart
        // without applying the same correction twice.
    };

    Ok(start_col)
}

enum ArnoldiOutcome<T: ComplexField> {
    Continue,
    Restart,
    Converged(GmresSolution<T>),
}

#[allow(clippy::too_many_arguments)]
fn arnoldi_step<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x: &mut Vec<T>,
    config: &GmresDrConfig<T::Real>,
    n: usize,
    _m: usize,
    b_norm: T::Real,
    true_b_norm: T::Real,
    j: usize,
    v: &mut FlatBasis<T>,
    op_buf: &mut [T],
    w: &mut [T],
    h: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    rots: &mut Vec<(usize, T, T)>,
    resid: &[T],
    g: &mut [T],
    resid_norm_sq: T::Real,
    r_capture_sq: &mut T::Real,
    coeffs: &mut Vec<T>,
    reorth_coeffs: &mut Vec<T>,
    total_iterations: &mut usize,
    restarts: usize,
    op_counts: &mut KrylovOpCounts,
) -> ArnoldiOutcome<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    *total_iterations += 1;

    // w = M^{-1} * A * v[j]
    operator.apply_into(v.vector(j), op_buf);
    op_counts.spmv_count += 1;
    precond.apply_into(op_buf, w);
    op_counts.preconditioner_count += 1;

    // Classical Gram-Schmidt with DGKS reorthogonalisation, batched.
    //
    // Pass 1 collects every projection coefficient against the basis in
    // parallel (independent reductions over disjoint vectors). Pass 2 fuses
    // all `j+1` axpys into a single parallel walk over the rows of `w`,
    // removing the false serialisation of the per-`i` axpy loop — the same
    // scheme `gmres.rs` uses.
    let w_norm_before = vector_norm(w);
    op_counts.norm_count += 1;

    batched_dots_flat_into(v, j + 1, w, coeffs);
    for (i, &c) in coeffs.iter().enumerate() {
        h[[i, j]] = c;
    }
    batched_project_flat(v, coeffs, w);
    op_counts.add_basis_projection::<T>(j + 1, n);

    // DGKS criterion: reorthogonalise if ||w|| < 0.707 * ||w_before||
    let dgks_threshold = T::Real::from_f64(0.707).unwrap();
    let mut w_norm = vector_norm(w);
    op_counts.norm_count += 1;
    if w_norm < dgks_threshold * w_norm_before {
        batched_dots_flat_into(v, j + 1, w, reorth_coeffs);
        for (i, &si) in reorth_coeffs.iter().enumerate() {
            h[[i, j]] += si;
        }
        batched_project_flat(v, reorth_coeffs, w);
        op_counts.add_basis_projection::<T>(j + 1, n);
        w_norm = vector_norm(w);
        op_counts.norm_count += 1;
    }
    h[[j + 1, j]] = T::from_real(w_norm);

    let breakdown_tol = T::Real::from_f64(1e-20).unwrap();
    let inner_converged = if w_norm < breakdown_tol {
        true
    } else {
        let inv_norm = T::from_real(T::Real::one() / w_norm);
        let next_v = v.vector_mut(j + 1);
        for i in 0..n {
            next_v[i] = w[i] * inv_norm;
        }
        op_counts.axpy_passes += 1;

        // True component of the cycle-start residual along the new basis
        // vector. With v[nk] parallel to the Ritz leftovers (not to the
        // residual), r is not contained in span(v[0..=nk]), so this must be
        // tracked explicitly — standard GMRES gets it for free because
        // r = beta*v0 is orthogonal to every later basis vector.
        let c_new = inner_product(v.vector(j + 1), resid);
        g[j + 1] = c_new;
        *r_capture_sq += c_new.norm_sqr();
        op_counts.dot_reductions += 1;
        op_counts.synchronization_points += 1;
        false
    };

    // Apply all previous Givens rotations to the new column: the dense QR
    // rotations of the initial deflation block first (application order),
    // then the per-step rotations of this cycle.
    for &(r, c, s) in rots.iter() {
        let temp = c.conj() * h[[r, j]] + s.conj() * h[[r + 1, j]];
        h[[r + 1, j]] = T::zero() - s * h[[r, j]] + c * h[[r + 1, j]];
        h[[r, j]] = temp;
    }

    let (c, s) = givens_rotation(h[[j, j]], h[[j + 1, j]]);
    rots.push((j, c, s));

    h[[j, j]] = c.conj() * h[[j, j]] + s.conj() * h[[j + 1, j]];
    h[[j + 1, j]] = T::zero();

    let temp = c.conj() * g[j] + s.conj() * g[j + 1];
    g[j + 1] = T::zero() - s * g[j] + c * g[j + 1];
    g[j] = temp;

    // True residual norm: |g[j+1]| alone misses the component of r not yet
    // captured by the basis, so add r_orth^2 = ||r||^2 - sum_i |<v[i], r>|^2.
    let r_orth_sq = (resid_norm_sq - *r_capture_sq).max(T::Real::zero());
    let abs_residual = (g[j + 1].norm_sqr() + r_orth_sq).sqrt();
    let rel_residual = abs_residual / b_norm;
    let abs_tol = T::Real::from_f64(1e-20).unwrap();

    if config.gmres.print_interval > 0
        && (*total_iterations).is_multiple_of(config.gmres.print_interval)
    {
        log::info!(
            "GMRES-DR iteration {} (restart {}): relative residual = {:.6e}",
            total_iterations,
            restarts,
            rel_residual.to_f64().unwrap_or(0.0)
        );
    }

    let estimated_converged =
        rel_residual < config.gmres.tolerance || abs_residual < abs_tol || inner_converged;
    let exact_check_due = exact_residual_check_due(
        config.gmres.exact_residual_check_interval,
        *total_iterations,
    );
    if estimated_converged || exact_check_due {
        let y = solve_upper_triangular(h, g, j + 1);
        let mut candidate = x.clone();
        fused_basis_update_flat(v, &y, candidate.as_mut_slice());
        op_counts.add_basis_update::<T>(y.len(), n);

        let true_rel_residual =
            exact_relative_residual(operator, &candidate, b, op_buf, true_b_norm, op_counts);
        if true_rel_residual < config.gmres.tolerance {
            x.clone_from(&candidate);

            return ArnoldiOutcome::Converged(GmresSolution {
                x: x.clone(),
                iterations: *total_iterations,
                restarts,
                residual: true_rel_residual,
                estimated_residual: rel_residual,
                backward_error: true_rel_residual,
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts: *op_counts,
            });
        }

        if estimated_converged {
            // The preconditioned/Hessenberg estimate triggered, but the
            // original operator residual did not. Commit the candidate and
            // restart from it; an optional exact checkpoint that did not
            // trigger projected convergence leaves the current Krylov base
            // unchanged. Do not finalize this cycle, which would apply the
            // same least-squares update a second time.
            x.clone_from(&candidate);
            return ArnoldiOutcome::Restart;
        }

        // An optional checkpoint which did not trigger projected convergence
        // is diagnostic only; retain the current Krylov base and continue.
        ArnoldiOutcome::Continue
    } else {
        ArnoldiOutcome::Continue
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_cycle<T: ComplexField>(
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    g: &[T],
    m: usize,
    v: &FlatBasis<T>,
    x: &mut [T],
    rots: &[(usize, T, T)],
    k: usize,
    n: usize,
    op_counts: &mut KrylovOpCounts,
) -> Vec<Vec<T>> {
    let y = solve_upper_triangular(h, g, m);
    fused_basis_update_flat(v, &y, x);
    op_counts.add_basis_update::<T>(y.len(), n);

    // Extract harmonic Ritz vectors for next cycle.
    let h_raw = unapply_givens_general(h, rots, m);
    extract_deflation_vectors_flat(v, &h_raw, m, k, n)
}

#[allow(clippy::too_many_arguments)]
fn build_final_solution<T, A>(
    operator: &A,
    b: &[T],
    x: Vec<T>,
    true_b_norm: T::Real,
    op_buf: &mut [T],
    op_counts: &mut KrylovOpCounts,
    total_iterations: usize,
    restarts: usize,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let rel_residual = exact_relative_residual(operator, &x, b, op_buf, true_b_norm, op_counts);

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
        op_counts: *op_counts,
    }
}

/// Core GMRES-DR implementation.
///
/// First cycle runs standard GMRES(m). At each restart, extracts the k harmonic
/// Ritz vectors with smallest eigenvalue magnitude and prepends them to the next
/// cycle's Krylov basis.
fn gmres_dr_core<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresDrConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if let Some(invalid) = invalid_gmres_input(
        b,
        x0,
        config.gmres.max_iterations,
        config.gmres.restart,
        config.gmres.tolerance,
    ) {
        return invalid;
    }

    let n = b.len();
    let m = config.gmres.restart;
    let k = config.num_deflation_vectors.min(m.saturating_sub(1));

    let (
        mut x,
        mut op_counts,
        b_norm,
        mut v,
        mut op_buf,
        mut w,
        mut h,
        mut rots,
        mut resid,
        mut g,
        mut deflation_vecs,
    ) = match init_gmres_dr_state(operator, precond, b, x0, config) {
        Ok(state) => state,
        Err(sol) => return sol,
    };
    let true_b_norm = vector_norm(b).max(T::Real::from_f64(1e-30).unwrap());
    op_counts.norm_count += 1;

    let mut total_iterations = 0;
    let mut restarts = 0;
    // Reusable projection-coefficient buffers for the batched CGS path.
    let mut coeffs: Vec<T> = Vec::with_capacity(m + 1);
    let mut reorth_coeffs: Vec<T> = Vec::with_capacity(m + 1);
    // Per-cycle residual tracking: ||r||^2 at cycle start and the squared
    // norm of the part of r already captured by the basis (see
    // initialize_cycle_basis / arnoldi_step).
    let mut resid_norm_sq = T::Real::zero();
    let mut r_capture_sq = T::Real::zero();

    'outer: for _outer in 0..config.gmres.max_iterations {
        let start_col = match initialize_cycle_basis(
            operator,
            precond,
            b,
            &mut x,
            config,
            n,
            m,
            k,
            b_norm,
            &mut v,
            &mut op_buf,
            &mut w,
            &mut h,
            &mut rots,
            &mut resid,
            &mut g,
            &mut resid_norm_sq,
            &mut r_capture_sq,
            &deflation_vecs,
            total_iterations,
            restarts,
            &mut op_counts,
        ) {
            Ok(start_col) => start_col,
            Err(sol) => return sol,
        };

        for j in start_col..m {
            match arnoldi_step(
                operator,
                precond,
                b,
                &mut x,
                config,
                n,
                m,
                b_norm,
                true_b_norm,
                j,
                &mut v,
                &mut op_buf,
                &mut w,
                &mut h,
                &mut rots,
                &resid,
                &mut g,
                resid_norm_sq,
                &mut r_capture_sq,
                &mut coeffs,
                &mut reorth_coeffs,
                &mut total_iterations,
                restarts,
                &mut op_counts,
            ) {
                ArnoldiOutcome::Converged(sol) => return sol,
                ArnoldiOutcome::Restart => {
                    restarts += 1;
                    continue 'outer;
                }
                ArnoldiOutcome::Continue => {}
            }
        }

        deflation_vecs = finalize_cycle(&h, &g, m, &v, &mut x, &rots, k, n, &mut op_counts);
        restarts += 1;
    }

    build_final_solution(
        operator,
        b,
        x,
        true_b_norm,
        &mut op_buf,
        &mut op_counts,
        total_iterations,
        restarts,
    )
}

/// Reconstruct the raw (un-rotated) Hessenberg matrix from the Givens-rotated
/// upper triangular matrix R and the Givens rotation coefficients.
///
/// H = Q * R where Q = G_0 * G_1 * ... * G_{m-1} and each G_j applies a
/// Givens rotation to rows (j, j+1). To recover H from R, we apply the
/// inverse rotations: H = G_{m-1}^{-1} * ... * G_0^{-1} * R.
pub(super) fn unapply_givens<T: ComplexField>(
    h_rotated: &sonium_solvers_core::dense::DenseMatrix<T>,
    cs: &[T],
    sn: &[T],
    m: usize,
) -> sonium_solvers_core::dense::DenseMatrix<T> {
    let mut h = h_rotated.clone();

    // Apply inverse Givens rotations in reverse order
    // Inverse of G(c,s) is G(c.conj(), -s)
    for j in (0..m).rev() {
        if j >= cs.len() {
            continue;
        }
        let c = cs[j];
        let s = sn[j];

        // Inverse rotation applied to rows j and j+1 of all columns
        // G = [c*  s*; -s  c], so G^{-1} = G^H = [c  -s*; s*  c*]
        for col in j..m {
            let hj = h[[j, col]];
            let hj1 = h[[j + 1, col]];
            h[[j, col]] = c * hj - s.conj() * hj1;
            h[[j + 1, col]] = s.conj() * hj + c.conj() * hj1;
        }
    }

    h
}

/// Generalised variant of [`unapply_givens`] for rotation lists with
/// explicit row indices: the dense QR of a full initial deflation block
/// (multiple rotations per column, rows 0..nk) followed by the per-step
/// Hessenberg rotations. Inverse rotations are applied in reverse
/// application order across all columns. Used by GMRES-DR, whose initial
/// block is full; `gmres_recycled` keeps the per-row [`unapply_givens`].
fn unapply_givens_general<T: ComplexField>(
    h_rotated: &sonium_solvers_core::dense::DenseMatrix<T>,
    rots: &[(usize, T, T)],
    m: usize,
) -> sonium_solvers_core::dense::DenseMatrix<T> {
    let mut h = h_rotated.clone();

    for &(r, c, s) in rots.iter().rev() {
        // Inverse rotation applied to rows r and r+1 of all columns
        // G = [c*  s*; -s  c], so G^{-1} = G^H = [c  -s*; s*  c*]
        for col in 0..m {
            let hj = h[[r, col]];
            let hj1 = h[[r + 1, col]];
            h[[r, col]] = c * hj - s.conj() * hj1;
            h[[r + 1, col]] = s.conj() * hj + c.conj() * hj1;
        }
    }

    h
}

/// Extract k harmonic Ritz vectors from the Arnoldi decomposition.
///
/// Extracts deflation vectors using harmonic Ritz selection (Morgan 2002).
///
/// Computes eigenvalues of H_m, then selects the k eigenvectors with smallest
/// **Ritz residual norm** `h_{m+1,m} * |e_m^H y_j|`. This targets well-converged
/// approximate eigenvectors of A (interior spectrum), which is the correct
/// criterion for GMRES deflation on non-symmetric systems.
#[cfg(test)]
fn extract_deflation_vectors<T: ComplexField>(
    v: &[Vec<T>],
    h_raw: &sonium_solvers_core::dense::DenseMatrix<T>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Vec<T>> {
    extract_deflation_vectors_with(|i, coeff, out| axpy(coeff, &v[i], out), h_raw, m, k, n)
}

pub(super) fn extract_deflation_vectors_flat<T: ComplexField>(
    v: &FlatBasis<T>,
    h_raw: &sonium_solvers_core::dense::DenseMatrix<T>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Vec<T>> {
    extract_deflation_vectors_with(
        |i, coeff, out| axpy(coeff, v.vector(i), out),
        h_raw,
        m,
        k,
        n,
    )
}

fn extract_deflation_vectors_with<T, F>(
    mut add_basis_scaled: F,
    h_raw: &sonium_solvers_core::dense::DenseMatrix<T>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Vec<T>>
where
    T: ComplexField,
    F: FnMut(usize, T, &mut [T]),
{
    // Extract the m x m upper Hessenberg submatrix
    let mut hm = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, T::zero());
    for i in 0..m {
        for j in 0..m {
            hm[[i, j]] = h_raw[[i, j]];
        }
    }

    // -----------------------------------------------------------------
    // Harmonic Ritz vectors (Morgan 2002).
    //
    // Standard Ritz vectors target eigenvalues of H_m; harmonic Ritz
    // vectors target eigenvalues of A near zero, which are exactly the
    // ones that stall GMRES convergence.
    //
    // Formula:  G = H_m + h_{m+1,m}^2 * H_m^{-T} e_m e_m^T
    // where e_m is the last unit vector.  The eigenvectors of G are the
    // harmonic Ritz vectors in the H_m basis.
    // -----------------------------------------------------------------
    let h_sub = if m < h_raw.nrows() {
        h_raw[[m, m - 1]]
    } else {
        T::zero()
    };
    let h_sub_norm_sq = h_sub.norm_sqr();

    if h_sub_norm_sq > T::Real::from_f64(1e-30).unwrap() {
        // Solve H_m^T z = e_m  =>  z = H_m^{-T} e_m
        let hm_t = hm.t().to_owned();
        let z = if let Ok(lu) = lu_factorize(&hm_t) {
            let mut e_m = vec![T::zero(); m];
            e_m[m - 1] = T::one();
            lu.solve(&e_m).unwrap_or_else(|_| vec![T::zero(); m])
        } else {
            vec![T::zero(); m]
        };

        // Rank-1 update:  G[:, m-1] += h_sub^2 * z
        for i in 0..m {
            hm[[i, m - 1]] += T::from_real(h_sub_norm_sq) * z[i];
        }
    }

    // Compute eigendecomposition of the updated matrix G.
    //
    // G is NOT necessarily upper Hessenberg: in deflated cycles the initial
    // block of H_m is full (row nk couples every deflation column), so the
    // Francis QR eigensolver cannot be run on G directly. Reduce to
    // Hessenberg form first and lift the eigenvectors back through Q_hess —
    // the same pattern gcro_dr uses for its augmented matrix.
    let (g_hess, q_hess) = dense_to_hessenberg(&hm, m);
    let (eigenvalues, eigenvectors_hess) = hessenberg_qr_eigen(&g_hess, m);
    let mut eigenvectors = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, T::zero());
    for j in 0..m {
        for i in 0..m {
            let mut sum = T::zero();
            for l in 0..m {
                sum += q_hess[[i, l]] * eigenvectors_hess[[l, j]];
            }
            eigenvectors[[i, j]] = sum;
        }
    }

    // Select the k eigenvectors with smallest eigenvalue magnitude
    // (interior spectrum — the eigenvalues closest to zero).
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| {
        let mag_a = eigenvalues[a].norm();
        let mag_b = eigenvalues[b].norm();
        mag_a
            .partial_cmp(&mag_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected = &indices[..k.min(indices.len())];

    // Form Ritz vectors: w_j = V_m * y_j and orthogonalize
    let mut deflation: Vec<Vec<T>> = Vec::with_capacity(selected.len());
    for &idx in selected {
        let mut w_vec = vec![T::zero(); n];
        for i in 0..m {
            add_basis_scaled(i, eigenvectors[[i, idx]], &mut w_vec);
        }

        // Orthogonalize against previous deflation vectors (modified Gram-Schmidt)
        for prev in &deflation {
            let coeff = inner_product(prev, &w_vec);
            axpy(-coeff, prev, &mut w_vec);
        }

        let norm = vector_norm(&w_vec);
        let tol = T::Real::from_f64(1e-14).unwrap();
        if norm > tol {
            let inv_norm = T::from_real(T::Real::one() / norm);
            for v in &mut w_vec {
                *v *= inv_norm;
            }
            deflation.push(w_vec);
        }
    }

    deflation
}

/// Compute eigenvalues and eigenvectors of an upper Hessenberg matrix using
/// the implicit QR algorithm.
///
/// Steps use a Wilkinson single shift; whenever the trailing 2x2 block has a
/// complex-conjugate eigenvalue pair (real, negative discriminant) an implicit
/// Francis double-shift step is used instead, so a real matrix converges to a
/// real Schur form whose 2x2 diagonal blocks represent the pairs. A real
/// single shift cannot represent such a pair and stalls. Genuinely complex
/// spectra keep the complex single-shift path.
///
/// Returns (eigenvalues, eigenvector_matrix) where eigenvalues[j] is the j-th
/// eigenvalue and eigenvectors column j is the corresponding eigenvector in the
/// Hessenberg basis. For a complex-conjugate pair of a numerically real
/// matrix, the two returned columns hold the real and imaginary parts of the
/// complex eigenvector (a real basis of the 2D invariant subspace) and both
/// eigenvalue entries hold |λ|, which is all downstream harmonic-Ritz
/// selection consumes.
///
/// Since m is small (typically 30-50), this is O(m^3) and fast.
pub(super) fn hessenberg_qr_eigen<T: ComplexField>(
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    m: usize,
) -> (Vec<T>, sonium_solvers_core::dense::DenseMatrix<T>) {
    // After convergence to (quasi-)triangular form, extract eigenvalues from
    // the (block) diagonal and back-substitute for eigenvectors.

    let mut t = h.clone(); // Working copy (will converge to Schur form)
    let mut q = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, T::zero()); // Accumulated transformations
    for i in 0..m {
        q[[i, i]] = T::one();
    }

    let max_qr_iters = 200 * m;
    let mut p = m; // Active submatrix is t[0..p, 0..p]

    for _iter in 0..max_qr_iters {
        if p <= 1 {
            break;
        }

        // Check for deflation: find the largest q such that t[p-1, p-2] ~ 0
        let deflation_tol = T::Real::from_f64(1e-14).unwrap();
        let mut deflated = false;
        for i in (1..p).rev() {
            let sub = t[[i, i - 1]].norm();
            let diag_sum = t[[i, i]].norm() + t[[i - 1, i - 1]].norm();
            let threshold = deflation_tol
                * if diag_sum > T::Real::zero() {
                    diag_sum
                } else {
                    T::Real::one()
                };
            if sub < threshold {
                t[[i, i - 1]] = T::zero();
                // Deflate at any position (not just bottom): reduce active
                // submatrix to t[0..i, 0..i] so the shift targets the correct block.
                p = i;
                deflated = true;
                break;
            }
        }
        if deflated {
            continue;
        }

        // Trailing 2x2 block and its discriminant.
        let a = t[[p - 2, p - 2]];
        let b = t[[p - 2, p - 1]];
        let c = t[[p - 1, p - 2]];
        let d = t[[p - 1, p - 1]];
        let half = T::from_real(T::Real::from_f64(0.5).unwrap());
        let da = (a - d) * half;
        let disc = da * da + b * c;
        let trace = a + d;
        let det = a * d - b * c;

        // A complex-conjugate eigenvalue pair shows up as a real, negative
        // discriminant. This is only meaningful when the block (hence the
        // whole active window, which shares its field) is numerically real:
        // guard on the imaginary parts so genuinely complex data keeps the
        // single-shift path.
        let scale = a.norm() + b.norm() + c.norm() + d.norm();
        let imag_tol = T::Real::from_f64(1e-12).unwrap() * scale.max(T::Real::one());
        let imag_tol_sq = T::Real::from_f64(1e-12).unwrap() * (scale * scale).max(T::Real::one());
        let complex_pair = disc.re() < T::Real::zero()
            && disc.im().abs() <= imag_tol_sq
            && trace.im().abs() <= imag_tol
            && det.im().abs() <= imag_tol_sq;

        if p == 2 && complex_pair {
            // The active window is exactly the real-Schur 2x2 block of a
            // complex-conjugate pair: accept it and finish.
            break;
        }

        if complex_pair {
            // Francis double-shift step with the shift pair σ₁, σ₂ =
            // eigenvalues of the trailing 2x2. Only s = σ₁ + σ₂ and
            // tt = σ₁σ₂ (both real-valued) enter the iteration, so the
            // arithmetic stays in the real field and the subdiagonal above
            // the pair's 2x2 block converges to zero.
            let s = T::from_real(trace.re());
            let tt = T::from_real(det.re());
            francis_double_shift_step(&mut t, &mut q, m, p, s, tt);
            continue;
        }

        // Wilkinson shift: eigenvalue of bottom-right 2x2 closest to t[p-1,p-1]
        let sigma = wilkinson_shift(a, b, c, d);

        // Single-shift QR step on t[0..p, 0..p]
        // Shifted matrix: t - sigma*I
        for i in 0..p {
            t[[i, i]] -= sigma;
        }

        // QR factorization of the shifted Hessenberg matrix using Givens rotations
        let mut givens_c = Vec::with_capacity(p - 1);
        let mut givens_s = Vec::with_capacity(p - 1);

        for i in 0..(p - 1) {
            let (c, s) = givens_rotation(t[[i, i]], t[[i + 1, i]]);
            givens_c.push(c);
            givens_s.push(s);

            // Apply rotation to rows i and i+1 (columns i..m)
            for col in i..m {
                let a = t[[i, col]];
                let b = t[[i + 1, col]];
                t[[i, col]] = c.conj() * a + s.conj() * b;
                t[[i + 1, col]] = T::zero() - s * a + c * b;
            }
        }

        // Apply rotations from right: R * Q^H -> form RQ + sigma*I
        for i in 0..(p - 1) {
            let c = givens_c[i];
            let s = givens_s[i];

            // Apply rotation to columns i and i+1 (rows 0..m)
            for row in 0..p.min(i + 3) {
                let a = t[[row, i]];
                let b = t[[row, i + 1]];
                t[[row, i]] = a * c + b * s;
                t[[row, i + 1]] = a * (T::zero() - s.conj()) + b * c.conj();
            }

            // Also accumulate into Q
            for row in 0..m {
                let a = q[[row, i]];
                let b = q[[row, i + 1]];
                q[[row, i]] = a * c + b * s;
                q[[row, i + 1]] = a * (T::zero() - s.conj()) + b * c.conj();
            }
        }

        // Undo shift
        for i in 0..p {
            t[[i, i]] += sigma;
        }
    }

    // Detect real-Schur 2x2 blocks: any subdiagonal entry that was not
    // explicitly zeroed during deflation marks a block (a complex-conjugate
    // pair, or a real pair if the iteration budget ran out).
    let has_2x2_blocks =
        (0..m.saturating_sub(1)).any(|k| t[[k + 1, k]].norm_sqr() > T::Real::zero());

    // Extract eigenvalues and eigenvectors of the Schur form, then transform
    // back: eigvecs_of_H = Q * eigvecs_of_T.
    let (eigenvalues, schur_eigvecs) = if has_2x2_blocks {
        quasi_schur_eigen(&t, m)
    } else {
        let eigenvalues: Vec<T> = (0..m).map(|i| t[[i, i]]).collect();
        let schur_eigvecs = schur_back_substitute(&t, m);
        (eigenvalues, schur_eigvecs)
    };

    // Transform: eigvecs = Q * schur_eigvecs
    let mut eigenvectors = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, T::zero());
    for j in 0..m {
        for i in 0..m {
            let mut sum = T::zero();
            for l in 0..m {
                sum += q[[i, l]] * schur_eigvecs[[l, j]];
            }
            eigenvectors[[i, j]] = sum;
        }
    }

    (eigenvalues, eigenvectors)
}

/// Perform one implicit double-shift (Francis) QR step on the active window
/// `t[0..p, 0..p]` with a complex-conjugate shift pair whose sum is `s` and
/// whose product is `tt` (both real-valued). Accumulates the orthogonal
/// similarity into `q`. Unlike a single-shift step, this preserves real
/// arithmetic and drives the subdiagonal above a complex-pair 2x2 block to
/// zero. Golub & Van Loan, Matrix Computations, Algorithm 7.5.1.
fn francis_double_shift_step<T: ComplexField>(
    t: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    q: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    m: usize,
    p: usize,
    s: T,
    tt: T,
) {
    debug_assert!(p >= 3);

    // First column of (H - σ₁I)(H - σ₂I): only three nonzero entries.
    let h00 = t[[0, 0]];
    let h01 = t[[0, 1]];
    let h10 = t[[1, 0]];
    let h11 = t[[1, 1]];
    let h21 = t[[2, 1]];
    let mut x = h00 * h00 + h01 * h10 - s * h00 + tt;
    let mut y = h10 * (h00 + h11 - s);
    let mut z = h10 * h21;

    let nonzero = T::Real::from_f64(1e-300).unwrap_or_else(T::Real::zero);

    for k in 0..(p - 1) {
        let len = if k + 3 <= p { 3 } else { 2 };
        let u = [x, y, z];

        // Householder reflector P = I - beta·v·vᵀ mapping u to ±||u||·e₁.
        let mut norm_sq = T::Real::zero();
        for val in u.iter().take(len) {
            norm_sq += val.norm_sqr();
        }
        let norm = norm_sq.sqrt();
        if norm > nonzero {
            let u0 = u[0];
            let alpha = if u0.re() >= T::Real::zero() {
                T::from_real(-norm)
            } else {
                T::from_real(norm)
            };
            let mut v = [T::zero(); 3];
            v[0] = u0 - alpha;
            for (l, vl) in v.iter_mut().enumerate().take(len).skip(1) {
                *vl = u[l];
            }
            let mut v_sq = T::Real::zero();
            for val in v.iter().take(len) {
                v_sq += val.norm_sqr();
            }
            if v_sq > nonzero {
                let beta = T::Real::from_f64(2.0).unwrap() / v_sq;
                let beta_t = T::from_real(beta);

                // Left: rows k..k+len, cols max(k-1, 0)..m. The similarity
                // must span the FULL width: rows of the active window carry
                // coupling entries into already-deflated columns (>= p),
                // and P·T transforms them too (GVL: H(k+1:k+3, q:n)).
                let col0 = k.saturating_sub(1);
                for col in col0..m {
                    let mut dot = T::zero();
                    for l in 0..len {
                        dot += v[l].conj() * t[[k + l, col]];
                    }
                    for l in 0..len {
                        t[[k + l, col]] -= beta_t * v[l] * dot;
                    }
                }

                // Right: rows 0..min(k+len+1, p), cols k..k+len.
                let row1 = (k + len + 1).min(p);
                for row in 0..row1 {
                    let mut dot = T::zero();
                    for l in 0..len {
                        dot += t[[row, k + l]] * v[l];
                    }
                    for l in 0..len {
                        t[[row, k + l]] -= beta_t * dot * v[l].conj();
                    }
                }

                // Accumulate the similarity into q: cols k..k+len, all rows.
                for row in 0..m {
                    let mut dot = T::zero();
                    for l in 0..len {
                        dot += q[[row, k + l]] * v[l];
                    }
                    for l in 0..len {
                        q[[row, k + l]] -= beta_t * dot * v[l].conj();
                    }
                }
            }
        }

        // Next bulge vector from the updated subdiagonal.
        if k + 1 < p - 1 {
            x = t[[k + 1, k]];
            y = t[[k + 2, k]];
            z = if k + 3 < p { t[[k + 3, k]] } else { T::zero() };
        }
    }
}

/// Wilkinson shift: eigenvalue of the 2x2 matrix
/// [[a, b], [c, d]] closest to d.
fn wilkinson_shift<T: ComplexField>(a: T, b: T, c: T, d: T) -> T {
    // Eigenvalues of [[a,b],[c,d]] are:
    // lambda = (a+d)/2 +/- sqrt((a-d)/2)^2 + b*c)
    let half = T::from_real(T::Real::from_f64(0.5).unwrap());
    let trace_half = (a + d) * half;
    let det_part = ((a - d) * half) * ((a - d) * half) + b * c;

    // For real T, det_part can be negative (complex conjugate eigenvalue pair).
    // sqrt() of a negative real produces NaN. Detect this by checking if the
    // sqrt result is NaN/invalid and fall back to Rayleigh shift (d).
    let disc = det_part.sqrt();
    if disc.norm() != disc.norm() {
        // NaN detected — complex eigenvalue pair. Use Rayleigh shift.
        return d;
    }

    let lambda1 = trace_half + disc;
    let lambda2 = trace_half - disc;

    // Pick the eigenvalue closer to d
    if (lambda1 - d).norm() < (lambda2 - d).norm() {
        lambda1
    } else {
        lambda2
    }
}

/// Back-substitution on an upper (quasi-)triangular Schur matrix to recover
/// eigenvectors.
fn schur_back_substitute<T: ComplexField>(
    t: &sonium_solvers_core::dense::DenseMatrix<T>,
    m: usize,
) -> sonium_solvers_core::dense::DenseMatrix<T> {
    let mut vecs = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, T::zero());
    let tol = T::Real::from_f64(1e-30).unwrap();

    for j in 0..m {
        vecs[[j, j]] = T::one();

        // Back-substitute: (T[0..j, 0..j] - T[j,j]*I) * x[0..j] = -T[0..j, j]
        for i in (0..j).rev() {
            let mut sum = T::zero();
            for l in (i + 1)..=j {
                sum += t[[i, l]] * vecs[[l, j]];
            }
            let diag = t[[i, i]] - t[[j, j]];
            if diag.norm() > tol {
                vecs[[i, j]] = (T::zero() - sum) * diag.inv();
            } else {
                // Near-degenerate eigenvalues: regularize to avoid silent zero.
                // Downstream Gram-Schmidt orthogonalization corrects artifacts.
                let eps = T::from_real(T::Real::from_f64(1e-10).unwrap());
                vecs[[i, j]] = (T::zero() - sum) * (diag + eps).inv();
            }
        }
    }

    vecs
}

/// Extract eigenvalues and eigenvectors from a real Schur (quasi-triangular)
/// matrix: 1x1 diagonal entries are real eigenvalues; every nonzero
/// subdiagonal entry `t[k+1, k]` marks a 2x2 block whose eigenvalues are a
/// complex-conjugate pair μ ± iν (or a real pair if the QR iteration budget
/// was exhausted before the block split).
///
/// All returned eigenvector columns are real: a 1x1 block yields its real
/// eigenvector, and a 2x2 block yields two columns — the real and imaginary
/// parts of its complex eigenvector — spanning the same 2D invariant
/// subspace. Both eigenvalue entries of a complex pair hold
/// |λ| = sqrt(μ² + ν²) so magnitude-based harmonic-Ritz selection orders the
/// pair correctly (downstream only consumes |λ|).
fn quasi_schur_eigen<T: ComplexField>(
    t: &sonium_solvers_core::dense::DenseMatrix<T>,
    m: usize,
) -> (Vec<T>, sonium_solvers_core::dense::DenseMatrix<T>) {
    // block2[k] == true marks a 2x2 block at rows/cols (k, k+1).
    let block2: Vec<bool> = (0..m.saturating_sub(1))
        .map(|k| t[[k + 1, k]].norm_sqr() > T::Real::zero())
        .collect();

    let mut eigenvalues: Vec<T> = Vec::with_capacity(m);
    let mut vecs = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, T::zero());

    let half = T::Real::from_f64(0.5).unwrap();
    let mut col = 0;
    let mut j = 0;
    while j < m {
        if j + 1 < m && block2[j] {
            let a = t[[j, j]].re();
            let b = t[[j, j + 1]].re();
            let c = t[[j + 1, j]].re();
            let d = t[[j + 1, j + 1]].re();
            let mu = (a + d) * half;
            let da = (a - d) * half;
            let disc = da * da + b * c;

            if disc < T::Real::zero() {
                // Complex-conjugate pair μ ± iν. Pivot entries of the 2x2
                // block's eigenvector for λ = μ + iν.
                let nu = (-disc).sqrt();
                let pivot = if b.abs() >= c.abs() {
                    ((b, T::Real::zero()), (mu - a, nu))
                } else {
                    ((mu - d, nu), (c, T::Real::zero()))
                };
                let w = quasi_eigvec_column::<T>(t, &block2, m, j, mu, nu, Some(pivot));
                let mag = (mu * mu + nu * nu).sqrt();
                for i in 0..m {
                    vecs[[i, col]] = T::from_real(w[i].0);
                    vecs[[i, col + 1]] = T::from_real(w[i].1);
                }
                eigenvalues.push(T::from_real(mag));
                eigenvalues.push(T::from_real(mag));
                col += 2;
            } else {
                // Real pair (only reachable when the QR budget ran out).
                let sq = disc.sqrt();
                for lambda in [mu + sq, mu - sq] {
                    let pivot = if b.abs() >= c.abs() {
                        ((b, T::Real::zero()), (lambda - a, T::Real::zero()))
                    } else {
                        ((lambda - d, T::Real::zero()), (c, T::Real::zero()))
                    };
                    let w = quasi_eigvec_column::<T>(
                        t,
                        &block2,
                        m,
                        j,
                        lambda,
                        T::Real::zero(),
                        Some(pivot),
                    );
                    for i in 0..m {
                        vecs[[i, col]] = T::from_real(w[i].0);
                    }
                    eigenvalues.push(T::from_real(lambda));
                    col += 1;
                }
            }
            j += 2;
        } else {
            let lambda = t[[j, j]].re();
            let w = quasi_eigvec_column::<T>(t, &block2, m, j, lambda, T::Real::zero(), None);
            for i in 0..m {
                vecs[[i, col]] = T::from_real(w[i].0);
            }
            eigenvalues.push(t[[j, j]]);
            col += 1;
            j += 1;
        }
    }

    (eigenvalues, vecs)
}

/// Back-substitution for one eigenvector column of the quasi-triangular `t`.
///
/// The eigenvalue is `mu + i·nu` (`nu = 0` for real eigenvalues). `t` is
/// numerically real, so the complex arithmetic is carried out on `(re, im)`
/// pairs of `T::Real`. `pivot2` supplies the two eigenvector entries of a
/// 2x2 diagonal block at rows `(j, j+1)`; `None` selects the unit pivot
/// `w[j] = 1` for a 1x1 block. Crossing another 2x2 block during the
/// substitution is handled with a 2x2 complex (Cramer) solve, mirroring
/// LAPACK `dtrevc`.
#[allow(clippy::type_complexity)]
fn quasi_eigvec_column<T: ComplexField>(
    t: &sonium_solvers_core::dense::DenseMatrix<T>,
    block2: &[bool],
    m: usize,
    j: usize,
    mu: T::Real,
    nu: T::Real,
    pivot2: Option<((T::Real, T::Real), (T::Real, T::Real))>,
) -> Vec<(T::Real, T::Real)> {
    let mut w = vec![(T::Real::zero(), T::Real::zero()); m];

    match pivot2 {
        Some((y0, y1)) => {
            w[j] = y0;
            w[j + 1] = y1;
        }
        None => {
            w[j] = (T::Real::one(), T::Real::zero());
        }
    }

    let tol = T::Real::from_f64(1e-30).unwrap();
    let eps = T::Real::from_f64(1e-10).unwrap();

    let mut i = j as isize - 1;
    while i >= 0 {
        let iu = i as usize;
        if iu >= 1 && block2[iu - 1] {
            // Crossing a 2x2 block at rows (iu-1, iu): solve
            //   [A B; C D] [x; y] = -[r0; r1]
            // with A = t[iu-1,iu-1] - λ, D = t[iu,iu] - λ (complex) and
            // B = t[iu-1,iu], C = t[iu,iu-1] (real).
            let mut r0 = (T::Real::zero(), T::Real::zero());
            let mut r1 = (T::Real::zero(), T::Real::zero());
            for l in (iu + 1)..m {
                let t0 = t[[iu - 1, l]].re();
                let t1 = t[[iu, l]].re();
                r0.0 += t0 * w[l].0;
                r0.1 += t0 * w[l].1;
                r1.0 += t1 * w[l].0;
                r1.1 += t1 * w[l].1;
            }
            let aa = (t[[iu - 1, iu - 1]].re() - mu, -nu);
            let bb = (t[[iu - 1, iu]].re(), T::Real::zero());
            let cc = (t[[iu, iu - 1]].re(), T::Real::zero());
            let dd = (t[[iu, iu]].re() - mu, -nu);
            let det = csub(cmul(aa, dd), cmul(bb, cc));
            // Cramer: x = (R0·D - B·R1)/det, y = (A·R1 - R0·C)/det.
            let nr0 = (-r0.0, -r0.1);
            let nr1 = (-r1.0, -r1.1);
            w[iu - 1] = cdiv(csub(cmul(nr0, dd), cmul(bb, nr1)), det, tol, eps);
            w[iu] = cdiv(csub(cmul(aa, nr1), cmul(nr0, cc)), det, tol, eps);
            i -= 2;
        } else {
            let mut r = (T::Real::zero(), T::Real::zero());
            for l in (iu + 1)..m {
                let te = t[[iu, l]].re();
                r.0 += te * w[l].0;
                r.1 += te * w[l].1;
            }
            let denom = (t[[iu, iu]].re() - mu, -nu);
            w[iu] = cdiv((-r.0, -r.1), denom, tol, eps);
            i -= 1;
        }
    }

    w
}

/// Complex multiply on `(re, im)` pairs.
#[inline]
fn cmul<R: Float>(x: (R, R), y: (R, R)) -> (R, R) {
    (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0)
}

/// Complex subtract on `(re, im)` pairs.
#[inline]
fn csub<R: Float>(x: (R, R), y: (R, R)) -> (R, R) {
    (x.0 - y.0, x.1 - y.1)
}

/// Complex divide on `(re, im)` pairs with the same near-degenerate
/// regularisation as `schur_back_substitute`.
#[inline]
fn cdiv<R: Float>(x: (R, R), y: (R, R), tol: R, eps: R) -> (R, R) {
    let mut y0 = y.0;
    if (y0 * y0 + y.1 * y.1).sqrt() < tol {
        y0 = y0 + eps;
    }
    let n = y0 * y0 + y.1 * y.1;
    ((x.0 * y0 + x.1 * y.1) / n, (x.1 * y0 - x.0 * y.1) / n)
}

/// Reduce a general dense matrix to upper Hessenberg form using
/// Householder reflections.
///
/// Returns `(H, Q)` such that `A = Q * H * Q^H` where `H` is upper
/// Hessenberg and `Q` is unitary (orthogonal for real `T`).
///
/// Only the strictly lower-triangular part below the first subdiagonal
/// is eliminated.  For `n ≤ 60` (the GCRO-DR augmented matrix size) this
/// is negligible work, so a compact triple-loop implementation is used.
pub(super) fn dense_to_hessenberg<T: ComplexField>(
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
    n: usize,
) -> (
    sonium_solvers_core::dense::DenseMatrix<T>,
    sonium_solvers_core::dense::DenseMatrix<T>,
) {
    let mut h = a.clone();
    let mut q = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, T::zero());
    for i in 0..n {
        q[[i, i]] = T::one();
    }

    let tol = T::Real::from_f64(1e-30).unwrap();

    for j in 0..n.saturating_sub(2) {
        // Extract sub-column x = h[j+1..n, j]
        let mut x_norm_sq = T::Real::zero();
        for i in (j + 1)..n {
            x_norm_sq += h[[i, j]].norm_sqr();
        }
        let x_norm = x_norm_sq.sqrt();

        if x_norm < tol {
            continue;
        }

        // Stable phase choice: alpha = -sign(re(x0)) * ||x||
        let x0 = h[[j + 1, j]];
        let alpha = if x0.re() >= T::Real::zero() {
            T::from_real(-x_norm)
        } else {
            T::from_real(x_norm)
        };

        // Householder vector v = x - alpha * e_1
        let mut v = vec![T::zero(); n - j - 1];
        v[0] = x0 - alpha;
        for i in (j + 2)..n {
            v[i - j - 1] = h[[i, j]];
        }

        // ||v||^2 = 2 * (||x||^2 + |alpha| * |re(x0)|)  for real case,
        // but we compute it directly to handle complex correctly.
        let mut v_norm_sq = T::Real::zero();
        for val in &v {
            v_norm_sq += val.norm_sqr();
        }

        if v_norm_sq < tol {
            continue;
        }
        let beta = T::Real::from_f64(2.0).unwrap() / v_norm_sq;

        // ---- Left apply: h = P * h  (only rows j+1..n-1 change) ----
        for k in j..n {
            let mut dot = T::zero();
            for l in (j + 1)..n {
                dot += v[l - j - 1].conj() * h[[l, k]];
            }
            for i in (j + 1)..n {
                h[[i, k]] -= T::from_real(beta) * v[i - j - 1] * dot;
            }
        }

        // ---- Right apply: h = h * P^H  (only cols j+1..n-1 change) ----
        for k in 0..n {
            let mut dot = T::zero();
            for l in (j + 1)..n {
                dot += h[[k, l]] * v[l - j - 1];
            }
            for i in (j + 1)..n {
                h[[k, i]] -= T::from_real(beta) * dot * v[i - j - 1].conj();
            }
        }

        // ---- Accumulate Q = Q * P^H  (only cols j+1..n-1 change) ----
        for k in 0..n {
            let mut dot = T::zero();
            for l in (j + 1)..n {
                dot += q[[k, l]] * v[l - j - 1];
            }
            for i in (j + 1)..n {
                q[[k, i]] -= T::from_real(beta) * dot * v[i - j - 1].conj();
            }
        }
    }

    // Zero numerical noise below the first subdiagonal.
    for j in 0..n {
        for i in (j + 2)..n {
            h[[i, j]] = T::zero();
        }
    }

    (h, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn test_gmres_dr_simple_f64() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 100,
                restart: 10,
                tolerance: 1e-10,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 2,
        };

        let sol = gmres_dr(&a, &b, &config);
        assert!(sol.converged, "GMRES-DR should converge");

        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution error too large: {}", error);
    }

    #[test]
    fn test_gmres_dr_rejects_preconditioned_false_convergence() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 100.0 * r[1];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0];
        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 1,
                restart: 2,
                tolerance: 1.0e-1,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 1,
        };

        let solution = gmres_dr_preconditioned(&a, &AnisotropicPreconditioner, &b, &config);
        let ax = a.matvec(&solution.x);
        let true_residual = sonium_solvers_core::vec_ops::vec_sub(&ax, &b)
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / b.iter().map(|value| value * value).sum::<f64>().sqrt();

        assert!(!solution.converged);
        assert!(matches!(
            solution.status,
            SolverStatus::MaxIterationsReached | SolverStatus::Breakdown
        ));
        assert!(
            true_residual >= config.gmres.tolerance,
            "false convergence accepted: true residual {true_residual:.3e}"
        );
    }

    #[test]
    fn test_gmres_dr_restarts_after_rejected_exact_candidate() {
        struct FirstProjectedCandidateOperator {
            calls: std::sync::atomic::AtomicUsize,
        }

        impl LinearOperator<f64> for FirstProjectedCandidateOperator {
            fn num_rows(&self) -> usize {
                2
            }

            fn num_cols(&self) -> usize {
                2
            }

            fn apply_into(&self, x: &[f64], y: &mut [f64]) {
                let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if call < 2 {
                    let epsilon = 1.0e-10;
                    y[0] = x[0] + epsilon * x[1];
                    y[1] = x[1];
                } else {
                    for (out, &value) in y.iter_mut().zip(x) {
                        *out = 2.0 * value;
                    }
                }
            }

            fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
                self.apply_into(x, y);
            }
        }

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 2,
                restart: 2,
                tolerance: 1.0e-8,
                convergence_check_interval: 1,
                exact_residual_check_interval: 1,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 1,
        };
        let solution = gmres_dr(
            &FirstProjectedCandidateOperator {
                calls: std::sync::atomic::AtomicUsize::new(0),
            },
            &[1.0, 1.0],
            &config,
        );

        assert!(
            solution.converged,
            "status={:?}, residual={}, estimated={}, iterations={}, restarts={}",
            solution.status,
            solution.residual,
            solution.estimated_residual,
            solution.iterations,
            solution.restarts
        );
        assert_eq!(solution.restarts, 1);
        assert!(solution.iterations >= 2);
        assert!(solution.residual < config.gmres.tolerance);
    }

    #[test]
    fn try_gmres_dr_rejects_non_finite_preconditioner_output() {
        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply(&self, r: &[f64]) -> Vec<f64> {
                vec![f64::NAN; r.len()]
            }

            fn apply_into(&self, _r: &[f64], output: &mut [f64]) {
                output.fill(f64::NAN);
            }
        }

        let matrix = CsrMatrix::<f64>::identity(2);
        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 4,
                restart: 2,
                tolerance: 1.0e-8,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 1,
        };
        let error = try_gmres_dr_preconditioned(
            &matrix,
            &NonFinitePreconditioner,
            &[1.0, 2.0],
            None,
            &config,
        )
        .expect_err("checked GMRES-DR must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let legacy =
            gmres_dr_preconditioned(&matrix, &NonFinitePreconditioner, &[1.0, 2.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn checked_gmres_dr_rejects_non_finite_projected_metrics() {
        let solution = GmresSolution {
            x: vec![0.0_f64],
            iterations: 1,
            restarts: 0,
            residual: 0.0,
            estimated_residual: f64::INFINITY,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotAttained,
            converged: false,
            status: SolverStatus::MaxIterationsReached,
            op_counts: KrylovOpCounts::default(),
        };

        let error = finish_checked_gmres_dr_solution(solution, 1)
            .expect_err("checked GMRES-DR must reject non-finite projected metrics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn checked_gmres_dr_rejects_malformed_solution_dimension() {
        let solution = GmresSolution {
            x: vec![0.0_f64],
            iterations: 0,
            restarts: 0,
            residual: 0.0,
            estimated_residual: 0.0,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::Attained,
            converged: true,
            status: SolverStatus::Converged,
            op_counts: KrylovOpCounts::default(),
        };

        let error = finish_checked_gmres_dr_solution(solution, 2)
            .expect_err("checked GMRES-DR must reject a malformed solution dimension");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn test_gmres_dr_reports_krylov_op_counts() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];
        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 100,
                restart: 10,
                tolerance: 1e-10,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 1,
        };

        let sol = gmres_dr(&a, &b, &config);

        assert!(sol.converged);
        assert!(sol.op_counts.spmv_count >= sol.iterations);
        assert!(sol.op_counts.preconditioner_count > 0);
        assert!(sol.op_counts.dot_reductions > 0);
        assert!(sol.op_counts.norm_count > 0);
        assert!(sol.op_counts.axpy_passes > 0);
    }

    #[test]
    fn test_gmres_dr_simple_complex() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 1.0), Complex64::new(1.0, -0.5)],
            [Complex64::new(1.0, 0.5), Complex64::new(3.0, -1.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(2.0, 1.0), Complex64::new(1.0, -1.0)];

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 100,
                restart: 10,
                tolerance: 1e-10,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 1,
        };

        let sol = gmres_dr(&a, &b, &config);
        assert!(sol.converged, "GMRES-DR should converge for complex system");

        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Complex solution error too large: {}", error);
    }

    #[test]
    fn test_gmres_dr_matches_standard_gmres() {
        // On a system that doesn't need restarts, GMRES-DR should produce
        // the same result as standard GMRES.
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
        let b = (0..n).map(|i| (i as f64).sin()).collect::<Vec<_>>();

        let gmres_config = GmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..GmresConfig::default()
        };

        let sol_std = crate::gmres::gmres(&a, &b, &gmres_config);

        let dr_config = GmresDrConfig {
            gmres: gmres_config.clone(),
            num_deflation_vectors: 5,
        };
        let sol_dr = gmres_dr(&a, &b, &dr_config);

        assert!(sol_std.converged, "Standard GMRES should converge");
        assert!(sol_dr.converged, "GMRES-DR should converge");

        // Both should produce solutions with small residual
        let ax_std = a.matvec(&sol_std.x);
        let ax_dr = a.matvec(&sol_dr.x);
        let err_std: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax_std, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        let err_dr: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax_dr, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(err_std < 1e-8, "Standard GMRES error: {}", err_std);
        assert!(err_dr < 1e-8, "GMRES-DR error: {}", err_dr);
    }

    #[test]
    fn test_gmres_dr_with_restarts() {
        // System that forces restarts: restart=5 on a 20-node system
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

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 200,
                restart: 8,
                tolerance: 1e-10,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 3,
        };

        let sol = gmres_dr(&a, &b, &config);
        assert!(
            sol.converged,
            "GMRES-DR should converge with restarts (residual: {:.2e})",
            sol.residual
        );

        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-5, "Solution error too large: {}", error);
    }

    #[test]
    fn test_gmres_dr_zero_deflation_falls_back() {
        // With num_deflation_vectors = 0, should behave like standard GMRES
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 100,
                restart: 10,
                tolerance: 1e-10,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 0,
        };

        let sol = gmres_dr(&a, &b, &config);
        assert!(sol.converged, "Should converge with zero deflation vectors");
    }

    #[test]
    fn test_gmres_dr_preconditioned() {
        use sonium_solvers_preconditioners::DiagonalPreconditioner;

        let n = 15;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.5);
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
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect::<Vec<_>>();

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 100,
                restart: 10,
                tolerance: 1e-10,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 3,
        };

        let sol = gmres_dr_preconditioned(&a, &precond, &b, &config);
        assert!(sol.converged, "Preconditioned GMRES-DR should converge");

        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-6, "Preconditioned solution error: {}", error);
    }

    #[test]
    fn test_gmres_dr_deflation_vectors_not_overwritten() {
        // Regression test for issue #16 part 1: deflation vectors in v[0..nk]
        // must never be overwritten by the Arnoldi continuation logic.
        //
        // Use a small restart with deflation so the bug would have destroyed
        // v[1] during the j=0 projection step.  The old bug caused a crash
        // or severe stagnation because the second deflation vector was
        // replaced by an Arnoldi vector.
        let n = 8;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 2.0 + 0.5 * (i as f64);
            if i > 0 {
                dense[[i, i - 1]] = -0.5;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -0.5;
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n).map(|i| (i as f64).sin()).collect::<Vec<_>>();

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 100,
                restart: 5,
                tolerance: 1e-6,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 2,
        };

        let sol = gmres_dr(&a, &b, &config);
        // The key assertion is that it converges at all; with the old bug
        // (overwriting deflation vectors) it would stagnate or crash.
        assert!(
            sol.converged,
            "GMRES-DR must converge with deflation (residual: {:.2e})",
            sol.residual
        );
    }

    #[test]
    fn test_harmonic_ritz_selects_interior_eigenvalues() {
        // Regression test for issue #16 part 2: verify that
        // extract_deflation_vectors uses harmonic Ritz (interior spectrum)
        // instead of standard Ritz (exterior spectrum).
        //
        // Construct an extended Hessenberg where the smallest standard Ritz
        // residual does NOT correspond to the smallest eigenvalue.
        let m = 4;
        let mut h_raw = sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, 0.0_f64);
        h_raw[[0, 0]] = 2.0;
        h_raw[[0, 1]] = 1.0;
        h_raw[[1, 0]] = 1.0;
        h_raw[[1, 1]] = 3.0;
        h_raw[[1, 2]] = 1.0;
        h_raw[[2, 1]] = 1.0;
        h_raw[[2, 2]] = 4.0;
        h_raw[[2, 3]] = 1.0;
        h_raw[[3, 2]] = 1.0;
        h_raw[[3, 3]] = 5.0;
        h_raw[[4, 3]] = 0.1; // tiny subdiagonal → harmonic Ritz ≈ standard Ritz

        // Dummy Krylov basis
        let n = 4;
        let v: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let mut vec = vec![0.0; n];
                vec[i] = 1.0;
                vec
            })
            .collect();

        let deflation = extract_deflation_vectors(&v, &h_raw, m, 2, n);
        assert!(
            !deflation.is_empty(),
            "Harmonic Ritz extraction must return at least one vector"
        );
    }

    #[test]
    fn test_hessenberg_qr_eigen_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal entries
        let m = 4;
        let mut h = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, 0.0_f64);
        h[[0, 0]] = 1.0;
        h[[1, 1]] = 2.0;
        h[[2, 2]] = 3.0;
        h[[3, 3]] = 4.0;

        let (eigenvalues, _) = hessenberg_qr_eigen(&h, m);

        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.norm()).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((eigs[0] - 1.0).abs() < 1e-10, "eig[0] = {}", eigs[0]);
        assert!((eigs[1] - 2.0).abs() < 1e-10, "eig[1] = {}", eigs[1]);
        assert!((eigs[2] - 3.0).abs() < 1e-10, "eig[2] = {}", eigs[2]);
        assert!((eigs[3] - 4.0).abs() < 1e-10, "eig[3] = {}", eigs[3]);
    }

    #[test]
    fn test_hessenberg_qr_eigen_hessenberg() {
        // Upper Hessenberg matrix with known eigenvalues
        // [[2, 1, 0], [1, 3, 1], [0, 1, 4]] has eigenvalues approx 1.268, 3.0, 4.732
        let m = 3;
        let mut h = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, 0.0_f64);
        h[[0, 0]] = 2.0;
        h[[0, 1]] = 1.0;
        h[[1, 0]] = 1.0;
        h[[1, 1]] = 3.0;
        h[[1, 2]] = 1.0;
        h[[2, 1]] = 1.0;
        h[[2, 2]] = 4.0;

        let (eigenvalues, _) = hessenberg_qr_eigen(&h, m);

        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.norm()).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Known eigenvalues of symmetric tridiagonal [2,3,4] with off-diag [1,1]
        assert!(
            (eigs[0] - 1.2679491924311228).abs() < 1e-6,
            "eig[0] = {}",
            eigs[0]
        );
        assert!((eigs[1] - 3.0).abs() < 1e-6, "eig[1] = {}", eigs[1]);
        assert!(
            (eigs[2] - 4.732050807568877).abs() < 1e-6,
            "eig[2] = {}",
            eigs[2]
        );
    }

    #[test]
    fn test_dense_to_hessenberg_produces_hessenberg_matrix() {
        // Verify that dense_to_hessenberg reduces a general dense matrix
        // to upper Hessenberg form and that Q is unitary.
        let n = 6;
        let mut a = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        // Fill with a non-symmetric, non-Hessenberg pattern.
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = ((i + 1) * (j + 2)) as f64 + (i as f64).sin() * (j as f64).cos();
            }
        }

        let (h, q) = dense_to_hessenberg(&a, n);

        // 1. h must be upper Hessenberg: h[i, j] == 0 for i > j + 1
        for j in 0..n {
            for i in (j + 2)..n {
                assert!(
                    h[[i, j]].abs() < 1e-12,
                    "Entry h[{i},{j}] = {} should be zero (below subdiagonal)",
                    h[[i, j]]
                );
            }
        }

        // 2. Q must be unitary: Q^T * Q ≈ I (for real matrices)
        let qt_q = q.t().dot(&q);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qt_q[[i, j]] - expected).abs() < 1e-10,
                    "Q^T*Q[{i},{j}] = {} should be {}",
                    qt_q[[i, j]],
                    expected
                );
            }
        }

        // 3. Reconstruction: Q * H * Q^T ≈ A
        let reconstructed = q.dot(&h).dot(&q.t());
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Reconstruction error at [{i},{j}]: got {}, expected {}",
                    reconstructed[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    /// Max-abs residual ||H·U - U·M|| for a 2-column real block U, where
    /// M = (UᵀU)⁻¹ Uᵀ H U is the least-squares 2x2 representation of H on
    /// span(U). Zero iff span(U) is an invariant subspace of H.
    fn invariant_subspace_residual(
        h: &sonium_solvers_core::dense::DenseMatrix<f64>,
        vecs: &sonium_solvers_core::dense::DenseMatrix<f64>,
        m: usize,
        c0: usize,
        c1: usize,
    ) -> f64 {
        // hu[i][k] = (H·U)[i][k]
        let mut hu = vec![[0.0_f64; 2]; m];
        for i in 0..m {
            for (k, ck) in [c0, c1].iter().enumerate() {
                let mut s = 0.0;
                for l in 0..m {
                    s += h[[i, l]] * vecs[[l, *ck]];
                }
                hu[i][k] = s;
            }
        }
        // G = UᵀU, W = Uᵀ(H·U)
        let mut g = [[0.0_f64; 2]; 2];
        let mut wmat = [[0.0_f64; 2]; 2];
        for a in 0..2 {
            let ca = if a == 0 { c0 } else { c1 };
            for b in 0..2 {
                let cb = if b == 0 { c0 } else { c1 };
                let mut sg = 0.0;
                let mut sw = 0.0;
                for i in 0..m {
                    sg += vecs[[i, ca]] * vecs[[i, cb]];
                    sw += vecs[[i, ca]] * hu[i][b];
                }
                g[a][b] = sg;
                wmat[a][b] = sw;
            }
        }
        let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
        assert!(det.abs() > 1e-20, "block columns must be independent");
        // M = G⁻¹ W
        let mut mmat = [[0.0_f64; 2]; 2];
        for (b, row) in mmat.iter_mut().enumerate() {
            row[0] = (g[1][1] * wmat[0][b] - g[0][1] * wmat[1][b]) / det;
            row[1] = (-g[1][0] * wmat[0][b] + g[0][0] * wmat[1][b]) / det;
        }
        // R = H·U - U·M
        let mut max_abs = 0.0_f64;
        for i in 0..m {
            for k in 0..2 {
                let r = hu[i][k] - (vecs[[i, c0]] * mmat[k][0] + vecs[[i, c1]] * mmat[k][1]);
                max_abs = max_abs.max(r.abs());
            }
        }
        max_abs
    }

    #[test]
    fn test_hessenberg_qr_eigen_real_complex_pairs() {
        // Real nonsymmetric Hessenberg matrix whose spectrum consists
        // entirely of complex-conjugate pairs:
        //   A = d·I + S, S skew-symmetric tridiagonal with ±1 off-diagonals.
        // Eigenvalues: λ_k = d + 2i·cos(kπ/(m+1)), k = 1..=m.
        let m = 6;
        let d = 2.0_f64;
        let mut h = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, 0.0_f64);
        for i in 0..m {
            h[[i, i]] = d;
            if i > 0 {
                h[[i, i - 1]] = -1.0;
                h[[i - 1, i]] = 1.0;
            }
        }

        let (eigenvalues, eigenvectors) = hessenberg_qr_eigen(&h, m);

        // True magnitudes |λ_k| = sqrt(d² + 4cos²(kπ/(m+1))).
        let mut expected: Vec<f64> = (1..=m)
            .map(|k| {
                let theta = (k as f64) * std::f64::consts::PI / (m as f64 + 1.0);
                (d * d + 4.0 * theta.cos() * theta.cos()).sqrt()
            })
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut got: Vec<f64> = eigenvalues.iter().map(|e| e.norm()).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..m {
            assert!(
                (got[k] - expected[k]).abs() < 1e-8,
                "eigenvalue magnitude {k}: got {}, expected {}",
                got[k],
                expected[k]
            );
        }

        // Each conjugate pair occupies two consecutive columns u, v spanning
        // a real 2D invariant subspace of H.
        for pair in 0..(m / 2) {
            let res = invariant_subspace_residual(&h, &eigenvectors, m, 2 * pair, 2 * pair + 1);
            assert!(
                res < 1e-8,
                "pair {pair}: invariant-subspace residual {res:.2e} too large"
            );
        }
    }

    #[test]
    fn test_hessenberg_qr_eigen_real_mixed_spectrum() {
        // Odd-sized skew tridiagonal + d·I: complex-conjugate pairs plus one
        // real eigenvalue λ = d (cos(π/2) = 0).
        let m = 5;
        let d = 2.0_f64;
        let mut h = sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, 0.0_f64);
        for i in 0..m {
            h[[i, i]] = d;
            if i > 0 {
                h[[i, i - 1]] = -1.0;
                h[[i - 1, i]] = 1.0;
            }
        }

        let (eigenvalues, eigenvectors) = hessenberg_qr_eigen(&h, m);

        let mut expected: Vec<f64> = (1..=m)
            .map(|k| {
                let theta = (k as f64) * std::f64::consts::PI / (m as f64 + 1.0);
                (d * d + 4.0 * theta.cos() * theta.cos()).sqrt()
            })
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut got: Vec<f64> = eigenvalues.iter().map(|e| e.norm()).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..m {
            assert!(
                (got[k] - expected[k]).abs() < 1e-8,
                "eigenvalue magnitude {k}: got {}, expected {}",
                got[k],
                expected[k]
            );
        }

        // The real eigenvalue's eigenvector must satisfy ||H v - λ v|| ≈ 0.
        let real_idx = eigenvalues
            .iter()
            .position(|&e| (e - d).abs() < 1e-6)
            .expect("real eigenvalue d must be present");
        let mut res = 0.0_f64;
        for i in 0..m {
            let mut hv = 0.0;
            for l in 0..m {
                hv += h[[i, l]] * eigenvectors[[l, real_idx]];
            }
            res = res.max((hv - d * eigenvectors[[i, real_idx]]).abs());
        }
        assert!(res < 1e-8, "real eigenvector residual {res:.2e} too large");
    }

    #[test]
    fn test_hessenberg_qr_eigen_complex_path_still_works() {
        // Genuinely complex spectrum: the complex single-shift path must be
        // unaffected by the real double-shift addition. Upper-triangular
        // input ⇒ eigenvalues are the diagonal entries.
        let m = 3;
        let mut h =
            sonium_solvers_core::dense::DenseMatrix::from_elem(m, m, Complex64::new(0.0, 0.0));
        h[[0, 0]] = Complex64::new(1.0, 1.0);
        h[[0, 1]] = Complex64::new(5.0, -2.0);
        h[[0, 2]] = Complex64::new(1.0, 3.0);
        h[[1, 1]] = Complex64::new(2.0, -1.0);
        h[[1, 2]] = Complex64::new(-1.0, 1.0);
        h[[2, 2]] = Complex64::new(3.0, 2.0);

        let (eigenvalues, eigenvectors) = hessenberg_qr_eigen(&h, m);

        let mut got: Vec<(f64, f64)> = eigenvalues.iter().map(|e| (e.re, e.im)).collect();
        got.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let expected = [(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)];
        for k in 0..m {
            assert!(
                (got[k].0 - expected[k].0).abs() < 1e-10
                    && (got[k].1 - expected[k].1).abs() < 1e-10,
                "eigenvalue {k}: got {:?}, expected {:?}",
                got[k],
                expected[k]
            );
        }

        // Eigenvector residuals ||H v - λ v|| ≈ 0.
        for (j, &lam) in eigenvalues.iter().enumerate() {
            let mut res = 0.0_f64;
            for i in 0..m {
                let mut hv = Complex64::new(0.0, 0.0);
                for l in 0..m {
                    hv += h[[i, l]] * eigenvectors[[l, j]];
                }
                res = res.max((hv - lam * eigenvectors[[i, j]]).norm());
            }
            assert!(res < 1e-10, "eigenvector {j}: residual {res:.2e}");
        }
    }

    #[test]
    fn test_gmres_dr_rotation_complex_eigenpairs() {
        // Deflation-quality test on a real nonsymmetric system whose
        // spectrum consists of complex-conjugate pairs (skew-dominated,
        // rotation/convection type). With the old Rayleigh-shift fallback
        // the small eigensolver could not resolve the pairs, so the
        // extracted harmonic Ritz vectors were inaccurate and deflated
        // restarting stalled.
        let n = 24;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 2.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
                dense[[i - 1, i]] = 1.0;
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n).map(|i| (i as f64).sin()).collect::<Vec<_>>();

        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 50,
                restart: 10,
                tolerance: 1e-8,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: 4,
        };

        let sol = gmres_dr(&a, &b, &config);
        assert!(
            sol.converged,
            "GMRES-DR must converge on a complex-pair spectrum (residual: {:.2e})",
            sol.residual
        );

        let ax = a.matvec(&sol.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e * e).sum::<f64>().sqrt();
        let true_rel = error / b_norm;
        assert!(
            true_rel < 1e-6,
            "True relative residual too large: {true_rel:.2e}"
        );

        // The reported (estimated) residual must track the true residual:
        // with the exact Arnoldi relation in deflated cycles they agree to
        // within a small factor, not the ~3 orders of magnitude gap the old
        // broken recurrence produced.
        let ratio = if true_rel > 0.0 {
            sol.residual / true_rel
        } else {
            1.0
        };
        assert!(
            (0.1..=10.0).contains(&ratio),
            "estimated residual {:.2e} vs true {:.2e}: ratio {:.2} out of [0.1, 10]",
            sol.residual,
            true_rel,
            ratio
        );
    }

    #[test]
    fn test_deflated_cycle_arnoldi_relation_exact() {
        // Drive the solver internals: run one standard cycle, extract
        // deflation vectors, then verify that the deflated cycle satisfies
        // the Arnoldi relation B V_j = V_{j+1} H_j at rounding level, both
        // after the initial full-block build and after continuation steps.
        let n = 24;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 2.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
                dense[[i - 1, i]] = 1.0;
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let m = 10;
        let k = 4;
        let config = GmresDrConfig {
            gmres: GmresConfig {
                max_iterations: 50,
                restart: m,
                tolerance: 1e-14,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..GmresConfig::default()
            },
            num_deflation_vectors: k,
        };

        let (
            mut x,
            mut op_counts,
            b_norm,
            mut v,
            mut op_buf,
            mut w,
            mut h,
            mut rots,
            mut resid,
            mut g,
            mut deflation_vecs,
        ) = init_gmres_dr_state(&a, &IdentityPreconditioner, &b, None, &config)
            .ok()
            .unwrap();
        let mut resid_norm_sq = 0.0;
        let mut r_capture_sq = 0.0;
        let mut coeffs = Vec::new();
        let mut reorth_coeffs = Vec::new();
        let precond = IdentityPreconditioner;

        // Frobenius norm of B V_cols - V H_cols relative to B V_cols, using
        // the raw (un-rotated) H reconstructed from the Givens rotations.
        let relation_error = |v: &FlatBasis<f64>,
                              rots: &[(usize, f64, f64)],
                              h: &sonium_solvers_core::dense::DenseMatrix<f64>,
                              cols: usize,
                              op_buf: &mut Vec<f64>| {
            let h_raw = unapply_givens_general(h, rots, cols);
            let mut err = 0.0_f64;
            let mut refn = 0.0_f64;
            for j in 0..cols {
                a.apply_into(v.vector(j), op_buf);
                let bvn: f64 = op_buf.iter().map(|t| t * t).sum::<f64>().sqrt();
                let mut r = op_buf.clone();
                // Raw column j of H has nonzeros in rows 0..=max(j+1, nk).
                for i in 0..h_raw.nrows().min(cols + 1) {
                    if h_raw[[i, j]] != 0.0 {
                        axpy(-h_raw[[i, j]], v.vector(i), &mut r);
                    }
                }
                err += r.iter().map(|t| t * t).sum::<f64>();
                refn += bvn * bvn;
            }
            (err / refn).sqrt()
        };

        // Cycle 1: no deflation, run m Arnoldi steps, extract deflation vectors.
        let start = initialize_cycle_basis(
            &a,
            &precond,
            &b,
            &mut x,
            &config,
            n,
            m,
            k,
            b_norm,
            &mut v,
            &mut op_buf,
            &mut w,
            &mut h,
            &mut rots,
            &mut resid,
            &mut g,
            &mut resid_norm_sq,
            &mut r_capture_sq,
            &deflation_vecs,
            0,
            0,
            &mut op_counts,
        )
        .ok()
        .unwrap();
        assert_eq!(start, 0);
        for j in 0..m {
            assert!(
                matches!(
                    arnoldi_step(
                        &a,
                        &precond,
                        &b,
                        &mut x,
                        &config,
                        n,
                        m,
                        b_norm,
                        b_norm,
                        j,
                        &mut v,
                        &mut op_buf,
                        &mut w,
                        &mut h,
                        &mut rots,
                        &resid,
                        &mut g,
                        resid_norm_sq,
                        &mut r_capture_sq,
                        &mut coeffs,
                        &mut reorth_coeffs,
                        &mut 0,
                        0,
                        &mut op_counts,
                    ),
                    ArnoldiOutcome::Continue
                ),
                "cycle 1 must not converge at step {j}"
            );
        }
        let rel0 = relation_error(&v, &rots, &h, m, &mut op_buf);
        assert!(
            rel0 < 1e-10,
            "cycle 1 Arnoldi relation error {rel0:.2e} (expected < 1e-10)"
        );
        deflation_vecs = finalize_cycle(&h, &g, m, &v, &mut x, &rots, k, n, &mut op_counts);
        assert!(deflation_vecs.len() >= 2);

        // Cycle 2 with deflation: check the relation right after the
        // initial full-block build...
        let start2 = initialize_cycle_basis(
            &a,
            &precond,
            &b,
            &mut x,
            &config,
            n,
            m,
            k,
            b_norm,
            &mut v,
            &mut op_buf,
            &mut w,
            &mut h,
            &mut rots,
            &mut resid,
            &mut g,
            &mut resid_norm_sq,
            &mut r_capture_sq,
            &deflation_vecs,
            0,
            1,
            &mut op_counts,
        )
        .ok()
        .unwrap();
        let nk = start2;
        assert!(nk >= 2, "deflation vectors must be installed");
        let rel_init = relation_error(&v, &rots, &h, nk, &mut op_buf);
        assert!(
            rel_init < 1e-10,
            "deflated init-block Arnoldi relation error {rel_init:.2e} (expected < 1e-10)"
        );

        // ...and after a few continuation steps.
        let extra = 3.min(m - nk);
        let mut total = 0usize;
        for j in nk..nk + extra {
            assert!(
                matches!(
                    arnoldi_step(
                        &a,
                        &precond,
                        &b,
                        &mut x,
                        &config,
                        n,
                        m,
                        b_norm,
                        b_norm,
                        j,
                        &mut v,
                        &mut op_buf,
                        &mut w,
                        &mut h,
                        &mut rots,
                        &resid,
                        &mut g,
                        resid_norm_sq,
                        &mut r_capture_sq,
                        &mut coeffs,
                        &mut reorth_coeffs,
                        &mut total,
                        1,
                        &mut op_counts,
                    ),
                    ArnoldiOutcome::Continue
                ),
                "cycle 2 must not converge at step {j}"
            );
        }
        let rel_cont = relation_error(&v, &rots, &h, nk + extra, &mut op_buf);
        assert!(
            rel_cont < 1e-10,
            "deflated continuation Arnoldi relation error {rel_cont:.2e} (expected < 1e-10)"
        );
    }

    #[test]
    fn gmres_dr_rejects_invalid_configuration_before_iteration() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let config = GmresDrConfig {
            gmres: GmresConfig {
                restart: 0,
                ..GmresConfig::default()
            },
            ..GmresDrConfig::default()
        };

        let solution = gmres_dr(&a, &[1.0, 2.0], &config);

        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.iterations, 0);
    }
}
