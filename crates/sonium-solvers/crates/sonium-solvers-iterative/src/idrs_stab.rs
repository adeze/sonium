//! IDR(s)stab(ℓ) solver (Sleijpen & van Gijzen, 2010)
//!
//! Implementation of IDR(s)stab(ℓ) from G. L. G. Sleijpen and M. B. van
//! Gijzen, "Exploiting BiCGstab(ℓ) strategies to induce dimension reduction",
//! SIAM J. Sci. Comput. 32(5):2687-2709, 2010, Algorithm 5.3. IDRstab unifies
//! IDR(s) and BiCGstab(ℓ): with ℓ = 1 it is mathematically equivalent to
//! IDR(s), with s = 1 to BiCGstab(ℓ).
//!
//! Where plain IDR(s) applies a single degree-1 stabilization factor
//! `(I - ω A)` per IDR cycle (one ω chosen by a 1-D residual minimization,
//! exactly like BiCGSTAB), IDRstab performs ℓ IDR steps per cycle and then
//! picks a degree-ℓ polynomial `1 - γ1 λ - ... - γℓ λ^ℓ` that minimizes the
//! residual norm over the whole cycle. For strongly non-symmetric or
//! (nearly) skew-symmetric systems — where the single ω is (nearly) zero and
//! BiCGSTAB/IDR(s) stagnate — the higher-degree stabilization keeps the
//! method converging. This makes IDR(s)stab(2) the recommended IDR variant
//! for Helmholtz-type indefinite systems.
//!
//! The algorithm maintains "tall" quantities: the residual chain
//! `r = [r0; ...; rj]` with `r_{i+1} = B r_i` and block matrices
//! `U = [U0; ...; Uj]`, `V = [V0; ...; V_{j+1}]` with `U_{i+1} = B U_i`,
//! where `B = A M^{-1}` is the right-preconditioned operator (so the tracked
//! residual is always the true unpreconditioned residual `b - A x`). The
//! iterate `y` lives in the preconditioned space; the final solution is
//! `x = M^{-1} y` (one extra preconditioner application at exit).
//!
//! Per cycle of ℓ IDR steps the algorithm uses `(s + 1) * ℓ` matrix-vector
//! products; `iterations` in the result counts MVs, as in [`super::idrs`].

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, scale_inplace, vector_norm};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{
    AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use super::cgs_orth::KrylovOpCounts;
use super::idrs::{IdrsSolution, generate_shadow_space, solve_small_system_into};

/// IDR(s)stab(ℓ) solver configuration
#[derive(Debug, Clone)]
pub struct IdrsStabConfig<R> {
    /// Shadow space dimension (s=1 gives BiCGstab(ℓ), s=4 is typical)
    pub s: usize,
    /// Stabilization polynomial degree (ell=1 is equivalent to IDR(s),
    /// ell=2 is the recommended choice for Helmholtz-type systems)
    pub ell: usize,
    /// Maximum number of iterations (matrix-vector products)
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
}

impl Default for IdrsStabConfig<f64> {
    fn default() -> Self {
        Self {
            s: 4,
            ell: 2,
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        }
    }
}

impl Default for IdrsStabConfig<f32> {
    fn default() -> Self {
        Self {
            s: 4,
            ell: 2,
            max_iterations: 1000,
            tolerance: 1e-5,
            print_interval: 0,
        }
    }
}

/// Column-block matrix: `s` columns of length `n` (one block-row of the tall
/// U/V matrices of the paper).
type Block<T> = Vec<Vec<T>>;

fn numerical_error_solution<T: ComplexField>(n: usize) -> IdrsSolution<T> {
    IdrsSolution {
        x: vec![T::zero(); n],
        iterations: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: sonium_solvers_core::traits::AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
    }
}

/// Checked IDR(s)stab(ell) entry point using the identity preconditioner.
pub fn try_idrs_stab<T, A>(
    operator: &A,
    b: &[T],
    config: &IdrsStabConfig<T::Real>,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = sonium_solvers_core::traits::IdentityPreconditioner;
    try_idrs_stab_checked(operator, &identity, b, config)
}

/// Checked preconditioned IDR(s)stab(ell) entry point.
pub fn try_idrs_stab_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsStabConfig<T::Real>,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    try_idrs_stab_checked(operator, precond, b, config)
}

fn try_idrs_stab_checked<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsStabConfig<T::Real>,
) -> Result<IdrsSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let invalid_preconditioner_output = AtomicBool::new(false);
    let checked_operator = CheckedIdrsStabOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let checked_precond = CheckedIdrsStabPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner_output,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_idrs_stab_inputs(operator, b, config)?;
        let solution =
            idrs_stab_preconditioned_unchecked(&checked_operator, &checked_precond, b, config);
        let solution =
            finalize_idrs_stab_solution(&checked_operator, b, config.tolerance, solution);
        finish_checked_idrs_stab_solution(
            solution,
            b.len(),
            invalid_operator_output.load(Ordering::Relaxed),
            invalid_preconditioner_output.load(Ordering::Relaxed),
        )
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "IDR(s)stab",
        reason: panic_reason(payload),
    })??;
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn validate_idrs_stab_inputs<T, A>(
    operator: &A,
    b: &[T],
    config: &IdrsStabConfig<T::Real>,
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
            operation: "IDR(s)stab",
            reason: "invalid IDR(s)stab configuration".to_string(),
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
            operation: "IDR(s)stab",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn finish_checked_idrs_stab_solution<T: ComplexField>(
    solution: IdrsSolution<T>,
    expected_dimension: usize,
    invalid_operator_output: bool,
    invalid_preconditioner_output: bool,
) -> Result<IdrsSolution<T>, SolverError> {
    if invalid_operator_output {
        return Err(SolverError::OperationFailed {
            operation: "IDR(s)stab operator",
            reason: "operator generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if invalid_preconditioner_output {
        return Err(SolverError::OperationFailed {
            operation: "IDR(s)stab preconditioner",
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
            operation: "IDR(s)stab",
            reason: "generated NaN, infinity, or an invalid solution".to_string(),
        });
    }
    Ok(solution)
}

fn finalize_idrs_stab_solution<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut result: IdrsSolution<T>,
) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    use sonium_solvers_core::traits::AccuracyStatus;

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
        "IDR(s)stab callback panicked".to_string()
    }
}

struct CheckedIdrsStabOperator<'a, A> {
    inner: &'a A,
    invalid_output: &'a AtomicBool,
}

impl<T, A> LinearOperator<T> for CheckedIdrsStabOperator<'_, A>
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

impl<A> CheckedIdrsStabOperator<'_, A> {
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

struct CheckedIdrsStabPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for CheckedIdrsStabPreconditioner<'_, P>
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

impl<P> CheckedIdrsStabPreconditioner<'_, P> {
    fn validate<T: ComplexField>(&self, input: &[T], output: &mut [T]) {
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

fn zero_block<T: ComplexField>(n: usize, s: usize) -> Block<T> {
    (0..s).map(|_| vec![T::zero(); n]).collect()
}

/// v -= M_block * c for an n-vector v and a block with s columns.
fn block_vec_sub<T: ComplexField>(c: &[T], m: &Block<T>, v: &mut [T]) {
    for (k, &ck) in c.iter().enumerate() {
        if ck != T::zero() {
            axpy(-ck, &m[k], v);
        }
    }
}

/// v += M_block * c for an n-vector v and a block with s columns.
fn block_vec_add<T: ComplexField>(c: &[T], m: &Block<T>, v: &mut [T]) {
    for (k, &ck) in c.iter().enumerate() {
        if ck != T::zero() {
            axpy(ck, &m[k], v);
        }
    }
}

/// w -= gamma * m, column-wise (both blocks have s columns).
fn block_block_sub<T: ComplexField>(gamma: T, m: &Block<T>, w: &mut Block<T>) {
    for (wq, mq) in w.iter_mut().zip(m.iter()) {
        axpy(-gamma, mq, wq);
    }
}

/// Solve Ax = b using IDR(s)stab(ℓ) (no preconditioner).
pub fn idrs_stab<T, A>(operator: &A, b: &[T], config: &IdrsStabConfig<T::Real>) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_idrs_stab(operator, b, config) {
        Ok(solution) => solution,
        Err(_) => numerical_error_solution(b.len()),
    }
}

/// Solve Ax = b using IDR(s)stab(ℓ) with a right preconditioner.
///
/// The iteration runs on `B = A M^{-1}` (right preconditioning), so the
/// residual tracked — and used for the convergence test — is always the true
/// unpreconditioned residual `b - A x`.
#[allow(clippy::too_many_lines)]
pub fn idrs_stab_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsStabConfig<T::Real>,
) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_idrs_stab_preconditioned(operator, precond, b, config) {
        Ok(solution) => solution,
        Err(_) => numerical_error_solution(b.len()),
    }
}

#[allow(clippy::too_many_lines)]
fn idrs_stab_preconditioned_unchecked<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &IdrsStabConfig<T::Real>,
) -> IdrsSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.len();
    let s = config.s;
    let ell = config.ell;
    let mut op_counts = KrylovOpCounts::default();

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;

    if s == 0 || s > n || ell == 0 {
        return IdrsSolution {
            x: vec![T::zero(); n],
            iterations: 0,
            residual: b_norm,
            estimated_residual: b_norm,
            backward_error: T::Real::infinity(),
            attainable_accuracy: sonium_solvers_core::traits::AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts,
        };
    }

    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        return IdrsSolution {
            x: vec![T::zero(); n],
            iterations: 0,
            residual: T::Real::zero(),
            estimated_residual: T::Real::zero(),
            backward_error: T::Real::infinity(),
            attainable_accuracy: sonium_solvers_core::traits::AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
        };
    }

    // eps²-scaled breakdown threshold (same construction as bicgstab.rs /
    // idrs.rs): dots against the shadow space scale like ||b||^2.
    let eps = T::Real::epsilon();
    let scaled = b_norm * b_norm * eps * eps;
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_tol = if scaled > floor { scaled } else { floor };

    let mut iterations = 0; // matrix-vector products

    // Preconditioned MV: B v = A (M^{-1} v). `pv` is the shared scratch.
    let mut pv = vec![T::zero(); n];
    macro_rules! apply_b {
        ($input:expr, $out:expr) => {{
            precond.apply_into($input, &mut pv);
            operator.apply_into(&pv, $out);
            op_counts.preconditioner_count += 1;
            op_counts.spmv_count += 1;
            iterations += 1;
        }};
    }

    // y: iterate in the preconditioned space (x = M^{-1} y at exit).
    let mut y = vec![T::zero(); n];

    // Residual chain r = [r0]; r0 = b - B y0 = b.
    let mut r: Vec<Vec<T>> = vec![b.to_vec()];

    // Deterministic complex-aware shadow space (same generator as idrs.rs).
    let p_cols = generate_shadow_space::<T>(n, s);

    // ---- Initial U = [U0; U1]: Arnoldi basis of K_s(B, r0), orthonormal in
    // the U0 block (Algorithm 5.3, initialization).
    let mut u: Vec<Block<T>> = vec![zero_block(n, s), zero_block(n, s)];
    for q in 0..s {
        let mut w0 = if q == 0 {
            r[0].clone()
        } else {
            u[1][q - 1].clone()
        };
        let mut w1 = vec![T::zero(); n];
        apply_b!(&w0, &mut w1);
        for (u0k, u1k) in u[0][..q].iter().zip(&u[1][..q]) {
            let mu = inner_product(u0k, &w0);
            op_counts.dot_reductions += 1;
            axpy(-mu, u0k, &mut w0);
            axpy(-mu, u1k, &mut w1);
            op_counts.axpy_passes += 2;
        }
        let nrm = vector_norm(&w0);
        op_counts.norm_count += 1;
        if nrm < breakdown_tol {
            let rel = vector_norm(&r[0]) / b_norm;
            return finish(
                precond,
                &y,
                iterations,
                rel,
                false,
                SolverStatus::Breakdown,
                config.tolerance,
                op_counts,
            );
        }
        let inv = T::from_real(T::Real::one() / nrm);
        scale_inplace(&mut w0, inv);
        scale_inplace(&mut w1, inv);
        u[0][q] = w0;
        u[1][q] = w1;
    }

    // Small dense work arrays.
    let mut sigma = DenseMatrix::from_elem(s, s, T::zero());
    let mut aug = DenseMatrix::from_elem(s, s + 1, T::zero());
    let mut coeffs = vec![T::zero(); s];
    let mut f = vec![T::zero(); s];

    let mv_per_cycle = (s + 1) * ell;
    let max_cycles = config.max_iterations / mv_per_cycle + 1;

    'outer: for _cycle in 0..max_cycles {
        for j in 1..=ell {
            // ---- One IDR step (paper section 5.1) ----
            // sigma = P^H U_j, f = P^H r_{j-1}, alpha = sigma^{-1} f.
            for i in 0..s {
                for k in 0..s {
                    sigma[[i, k]] = inner_product(&p_cols[i], &u[j][k]);
                }
                f[i] = inner_product(&p_cols[i], &r[j - 1]);
            }
            op_counts.dot_reductions += s * s + s;
            op_counts.synchronization_points += 1;
            if !solve_small_system_into(&sigma, &f, s, breakdown_tol, &mut aug, &mut coeffs) {
                let rel = vector_norm(&r[0]) / b_norm;
                return finish(
                    precond,
                    &y,
                    iterations,
                    rel,
                    false,
                    SolverStatus::Breakdown,
                    config.tolerance,
                    op_counts,
                );
            }
            // y += U0 alpha ; r_i -= U_{i+1} alpha (i = 0..j-1)
            block_vec_add(&coeffs, &u[0], &mut y);
            for (i, r_i) in r.iter_mut().enumerate().take(j) {
                block_vec_sub(&coeffs, &u[i + 1], r_i);
            }
            op_counts.axpy_passes += j + 1;
            // Append r_j = B r_{j-1}.
            let r_prev = r[j - 1].clone();
            let mut rj = vec![T::zero(); n];
            apply_b!(&r_prev, &mut rj);
            r.push(rj);

            // ---- Build V column by column (paper inner "for q" loop) ----
            let mut v: Vec<Block<T>> = (0..j + 2).map(|_| zero_block(n, s)).collect();
            let mut w: Vec<Vec<T>> = Vec::new(); // previous column's tall vector
            for q in 0..s {
                // q == 0: wq = r (blocks 0..=j); else shift: wq = w[1..].
                let mut wq: Vec<Vec<T>> = if q == 0 { r.clone() } else { w[1..].to_vec() };
                // beta = sigma^{-1} (P^H w_j); wq -= U beta (blocks 0..=j)
                for (i, f_i) in f.iter_mut().enumerate() {
                    *f_i = inner_product(&p_cols[i], &wq[j]);
                }
                op_counts.dot_reductions += s;
                op_counts.synchronization_points += 1;
                if !solve_small_system_into(&sigma, &f, s, breakdown_tol, &mut aug, &mut coeffs) {
                    let rel = vector_norm(&r[0]) / b_norm;
                    return finish(
                        precond,
                        &y,
                        iterations,
                        rel,
                        false,
                        SolverStatus::Breakdown,
                        config.tolerance,
                        op_counts,
                    );
                }
                for (i, w_i) in wq.iter_mut().enumerate() {
                    block_vec_sub(&coeffs, &u[i], w_i);
                }
                op_counts.axpy_passes += j + 1;
                // Append w_{j+1} = B w_j.
                let w_prev = wq[j].clone();
                let mut wj1 = vec![T::zero(); n];
                apply_b!(&w_prev, &mut wj1);
                wq.push(wj1);

                // Classical Gram-Schmidt against V columns 0..q on block j,
                // then normalize by ||w_j|| (Arnoldi stabilization, paper
                // section 5.4).
                let wj_norm_before = vector_norm(&wq[j]);
                op_counts.norm_count += 1;
                #[expect(
                    clippy::needless_range_loop,
                    reason = "k addresses one column across the independently stored v[i] blocks while wq updates in place (order-critical MGS); no single-vec iterator preserves both"
                )]
                for k in 0..q {
                    let mu = inner_product(&v[j][k], &wq[j]);
                    op_counts.dot_reductions += 1;
                    for (i, w_i) in wq.iter_mut().enumerate() {
                        axpy(-mu, &v[i][k], w_i);
                    }
                    op_counts.axpy_passes += j + 2;
                }
                let wj_norm = vector_norm(&wq[j]);
                op_counts.norm_count += 1;
                if wj_norm < eps * wj_norm_before || wj_norm < breakdown_tol {
                    // Krylov space exhausted (lucky breakdown) or numerically
                    // degenerate; stop here.
                    let rel = vector_norm(&r[0]) / b_norm;
                    return finish(
                        precond,
                        &y,
                        iterations,
                        rel,
                        false,
                        SolverStatus::Breakdown,
                        config.tolerance,
                        op_counts,
                    );
                }
                let inv = T::from_real(T::Real::one() / wj_norm);
                for w_i in wq.iter_mut() {
                    scale_inplace(w_i, inv);
                }
                for (i, w_i) in wq.iter().enumerate() {
                    v[i][q] = w_i.clone();
                }
                w = wq;
            }
            if j < ell {
                u = v;
                continue;
            }

            // ---- Polynomial step (paper section 5.2), j == ell ----
            // gamma = argmin ||r0 - [r1..r_ell] gamma|| via the normal
            // equations (as in BiCGstab(ell)).
            let mut gram = DenseMatrix::from_elem(ell, ell, T::zero());
            let mut rhs = vec![T::zero(); ell];
            for i in 0..ell {
                for k in 0..ell {
                    gram[[i, k]] = inner_product(&r[i + 1], &r[k + 1]);
                }
                rhs[i] = inner_product(&r[i + 1], &r[0]);
            }
            op_counts.dot_reductions += ell * ell + ell;
            op_counts.synchronization_points += 1;
            let mut gamma_aug = DenseMatrix::from_elem(ell, ell + 1, T::zero());
            let mut gamma = vec![T::zero(); ell];
            if !solve_small_system_into(&gram, &rhs, ell, breakdown_tol, &mut gamma_aug, &mut gamma)
            {
                // Fall back to the degree-1 (GMRES(1)/IDR(s)) factor.
                let r1r1 = gram[[0, 0]];
                if r1r1.norm() < breakdown_tol {
                    let rel = vector_norm(&r[0]) / b_norm;
                    return finish(
                        precond,
                        &y,
                        iterations,
                        rel,
                        false,
                        SolverStatus::Stagnated,
                        config.tolerance,
                        op_counts,
                    );
                }
                gamma[0] = rhs[0] * r1r1.inv();
            }

            // y += sum_{i=0}^{ell-1} gamma_{i+1} r_i
            for (i, &gi) in gamma.iter().enumerate() {
                axpy(gi, &r[i], &mut y);
            }
            // r0 = r0 - sum_{i=1}^{ell} gamma_i r_i
            let mut r0_new = r[0].clone();
            for (i, &gi) in gamma.iter().enumerate() {
                axpy(-gi, &r[i + 1], &mut r0_new);
            }
            r = vec![r0_new];
            // U0 = V0 - sum gamma_j V_j ; U1 = V1 - sum gamma_j V_{j+1}
            let mut u0 = v[0].clone();
            let mut u1 = v[1].clone();
            for (jj, &gj) in gamma.iter().enumerate() {
                block_block_sub(gj, &v[jj + 1], &mut u0);
                block_block_sub(gj, &v[jj + 2], &mut u1);
            }
            u = vec![u0, u1];
            op_counts.axpy_passes += 2 * ell + 2;

            let rel_residual = vector_norm(&r[0]) / b_norm;
            op_counts.norm_count += 1;
            if config.print_interval > 0 && iterations % config.print_interval == 0 {
                log::info!(
                    "IDR({})stab({}) iteration {}: relative residual = {:.6e}",
                    s,
                    ell,
                    iterations,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }
            if rel_residual < config.tolerance {
                return finish(
                    precond,
                    &y,
                    iterations,
                    rel_residual,
                    true,
                    SolverStatus::Converged,
                    config.tolerance,
                    op_counts,
                );
            }
            if iterations >= config.max_iterations {
                break 'outer;
            }
        }
    }

    let rel_residual = vector_norm(&r[0]) / b_norm;
    finish(
        precond,
        &y,
        iterations,
        rel_residual,
        false,
        SolverStatus::MaxIterationsReached,
        config.tolerance,
        op_counts,
    )
}

/// Map the y-space iterate to x = M^{-1} y and package the solver statistics.
#[allow(clippy::too_many_arguments)]
fn finish<T, P>(
    precond: &P,
    y: &[T],
    iterations: usize,
    residual: T::Real,
    converged: bool,
    status: SolverStatus,
    tolerance: T::Real,
    mut op_counts: KrylovOpCounts,
) -> IdrsSolution<T>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    let mut x = vec![T::zero(); y.len()];
    precond.apply_into(y, &mut x);
    op_counts.preconditioner_count += 1;
    // A breakdown or stagnation exit can still sit on a converged iterate
    // (e.g. the Krylov space is exhausted exactly when the residual reaches
    // the tolerance); report convergence honestly in that case.
    let (converged, status) = if !converged && residual < tolerance {
        (true, SolverStatus::Converged)
    } else {
        (converged, status)
    };
    IdrsSolution {
        x,
        iterations,
        residual,
        estimated_residual: residual,
        backward_error: T::Real::infinity(),
        attainable_accuracy: sonium_solvers_core::traits::AccuracyStatus::NotChecked,
        converged,
        status,
        op_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idrs::{IdrsConfig, idrs};
    use num_complex::Complex64;
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner};
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn idrs_stab_max_iteration_reports_true_residual_and_backward_error() {
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
        let config = IdrsStabConfig {
            s: 1,
            ell: 1,
            max_iterations: 1,
            tolerance: 1.0e-14,
            print_interval: 0,
        };
        let solution = idrs_stab(&matrix, &rhs, &config);
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
        assert_eq!(
            solution.attainable_accuracy,
            sonium_solvers_core::traits::AccuracyStatus::NotAttained
        );
    }

    #[test]
    fn checked_idrs_stab_contains_operator_panics() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("IDR(s)stab operator callback failed");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("IDR(s)stab operator callback failed");
            }
        }

        let config = IdrsStabConfig {
            s: 1,
            ell: 1,
            max_iterations: 10,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let error = try_idrs_stab(&PanicOperator, &[1.0], &config)
            .expect_err("checked IDR(s)stab must type callback panics");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "IDR(s)stab",
                ..
            }
        ));

        let legacy = idrs_stab(&PanicOperator, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    #[test]
    fn checked_idrs_stab_rejects_non_finite_preconditioner_output() {
        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
                output[0] = f64::NAN;
            }
        }

        let operator = CsrMatrix::identity(1);
        let precond = NonFinitePreconditioner;
        let config = IdrsStabConfig {
            s: 1,
            ell: 1,
            max_iterations: 10,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let error = try_idrs_stab_preconditioned(&operator, &precond, &[1.0], &config)
            .expect_err("checked IDR(s)stab must reject non-finite preconditioner output");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "IDR(s)stab preconditioner",
                ..
            }
        ));

        let legacy = idrs_stab_preconditioned(&operator, &precond, &[1.0], &config);
        assert_eq!(legacy.status, SolverStatus::NumericalError);
    }

    fn helmholtz_like(n: usize) -> CsrMatrix<Complex64> {
        // 1D Helmholtz-like operator: shifted Laplacian with a complex
        // wavenumber (damping) plus a first-derivative (advection) term —
        // the same matrix idrs.rs tests against.
        let k2 = Complex64::new(8.0, -2.0);
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
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn true_rel_residual<T: ComplexField>(a: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real {
        let ax = a.matvec(x);
        let mut err = T::Real::zero();
        let mut bnorm = T::Real::zero();
        for (ai, bi) in ax.iter().zip(b.iter()) {
            err += (*bi - *ai).norm_sqr();
            bnorm += bi.norm_sqr();
        }
        err.sqrt() / bnorm.sqrt()
    }

    #[test]
    fn test_idrs_stab_helmholtz_like_complex() {
        // IDR(4)stab(2) on the same Helmholtz-like complex non-symmetric
        // system idrs.rs uses: must converge with an accurate true residual.
        // Iteration counts must match plain IDR(4) within one stabilization
        // cycle (IDRstab spends (s+1)*ell MVs per cycle against IDR(s)'s
        // s+1); on harder members of the same family it wins outright, see
        // the next test.
        let n = 40;
        let a = helmholtz_like(n);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();

        let stab_config = IdrsStabConfig {
            s: 4,
            ell: 2,
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let stab = idrs_stab(&a, &b, &stab_config);
        assert!(
            stab.converged,
            "IDR(4)stab(2) should converge on Helmholtz-like system, residual = {}",
            stab.residual
        );
        let rel = true_rel_residual(&a, &stab.x, &b);
        assert!(rel < 1e-8, "true relative residual too large: {rel}");

        let idrs_config = IdrsConfig {
            s: 4,
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let plain = idrs(&a, &b, &idrs_config);
        assert!(plain.converged, "reference IDR(4) should converge too");
        assert!(
            stab.iterations <= plain.iterations + (4 + 1) * 2,
            "IDR(4)stab(2) ({} MVs) should match IDR(4) ({} MVs) within one cycle",
            stab.iterations,
            plain.iterations
        );
    }

    #[test]
    fn test_idrs_stab_harder_helmholtz_beats_idrs() {
        // Harder member of the same Helmholtz-like family (weaker damping,
        // larger n): the degree-1 omega recursion of plain IDR(s) becomes
        // less effective and IDR(4)stab(2) converges in strictly fewer
        // matrix-vector products.
        let n = 200;
        let k2 = Complex64::new(4.0, -1.0); // complex wavenumber² (weakly lossy)
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

        let stab = idrs_stab(
            &a,
            &b,
            &IdrsStabConfig {
                s: 4,
                ell: 2,
                max_iterations: 500,
                tolerance: 1e-10,
                print_interval: 0,
            },
        );
        assert!(
            stab.converged,
            "IDR(4)stab(2) should converge, residual = {}",
            stab.residual
        );
        let rel = true_rel_residual(&a, &stab.x, &b);
        assert!(rel < 1e-8, "true relative residual too large: {rel}");

        let plain = idrs(
            &a,
            &b,
            &IdrsConfig {
                s: 4,
                max_iterations: 500,
                tolerance: 1e-10,
                print_interval: 0,
            },
        );
        assert!(plain.converged, "reference IDR(4) should converge too");
        assert!(
            stab.iterations <= plain.iterations,
            "IDR(4)stab(2) ({} MVs) should match or beat IDR(4) ({} MVs)",
            stab.iterations,
            plain.iterations
        );
    }

    #[test]
    fn test_idrs_stab_skew_like_beats_idrs() {
        // Nearly skew-symmetric (convection-dominated) real system: the
        // degree-1 omega of IDR(s) is nearly zero and plain IDR(s) stagnates
        // or crawls; the degree-2 stabilization of IDR(s)stab(2) keeps
        // converging (this is the motivating case of the IDRstab paper).
        let n = 40;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 1.0;
            if i > 0 {
                dense[[i, i - 1]] = -4.0; // strong upwind convection
            }
            if i < n - 1 {
                dense[[i, i + 1]] = 4.0;
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin() + 0.5).collect();

        let stab_config = IdrsStabConfig {
            s: 4,
            ell: 2,
            max_iterations: 2000,
            tolerance: 1e-8,
            print_interval: 0,
        };
        let stab = idrs_stab(&a, &b, &stab_config);
        assert!(
            stab.converged,
            "IDR(4)stab(2) should converge on skew-like system, residual = {}",
            stab.residual
        );
        assert!(true_rel_residual(&a, &stab.x, &b) < 1e-6);

        let idrs_config = IdrsConfig {
            s: 4,
            max_iterations: 2000,
            tolerance: 1e-8,
            print_interval: 0,
        };
        let plain = idrs(&a, &b, &idrs_config);
        assert!(
            !plain.converged || stab.iterations <= plain.iterations,
            "IDR(4)stab(2) ({} MVs) should match or beat IDR(4) ({} MVs, converged = {})",
            stab.iterations,
            plain.iterations,
            plain.converged
        );
    }

    #[test]
    fn test_idrs_stab_simple_and_identity() {
        // Small complex SPD sanity check, s = 2.
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = IdrsStabConfig {
            s: 2,
            ell: 2,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = idrs_stab(&a, &b, &config);
        assert!(solution.converged, "IDR(2)stab(2) should converge");
        assert!(true_rel_residual(&a, &solution.x, &b) < 1e-8);

        // Identity matrix: converges almost immediately.
        let id: CsrMatrix<Complex64> = CsrMatrix::identity(5);
        let b: Vec<Complex64> = (1..=5).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let solution = idrs_stab(&id, &b, &config);
        assert!(solution.converged);
        assert!(true_rel_residual(&id, &solution.x, &b) < 1e-10);
    }

    #[test]
    fn test_idrs_stab_ell1_matches_idrs_behaviour() {
        // ell = 1 is mathematically equivalent to IDR(s): it must converge
        // on the standard tridiagonal system.
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
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect();

        let config = IdrsStabConfig {
            s: 4,
            ell: 1,
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = idrs_stab(&a, &b, &config);
        assert!(
            solution.converged,
            "IDR(4)stab(1) ~ IDR(4) should converge, residual = {}",
            solution.residual
        );
        assert!(true_rel_residual(&a, &solution.x, &b) < 1e-8);
    }

    #[test]
    fn idrs_stab_rejects_non_finite_input_before_operator_use() {
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

        let config = IdrsStabConfig::default();
        let solution = idrs_stab(&PanicOperator, &[f64::NAN], &config);
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn test_idrs_stab_invalid_params_and_zero_rhs() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        // s > n is rank-deficient.
        let bad = IdrsStabConfig {
            s: 4,
            ell: 2,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = idrs_stab(&a, &b, &bad);
        assert!(!solution.converged);
        assert_eq!(solution.status, SolverStatus::Breakdown);

        // ell = 0 is invalid.
        let bad = IdrsStabConfig { ell: 0, ..bad };
        let solution = idrs_stab(&a, &b, &bad);
        assert!(!solution.converged);
        assert_eq!(solution.status, SolverStatus::Breakdown);

        // Zero RHS converges immediately.
        let zero_b =
            sonium_solvers_core::array![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
        let ok = IdrsStabConfig {
            s: 1,
            ell: 2,
            max_iterations: 100,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let solution = idrs_stab(&a, &zero_b, &ok);
        assert!(solution.converged);
        assert_eq!(solution.iterations, 0);
    }
}
