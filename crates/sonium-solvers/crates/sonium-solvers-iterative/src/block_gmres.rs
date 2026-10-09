//! Block GMRES for multiple right-hand sides
//!
//! Solves `A * X = B` where `B` is `n × s` (s right-hand sides simultaneously)
//! with a *true* block method: a single block Arnoldi recurrence builds one
//! shared block Krylov basis
//!
//! ```text
//! K_m(A, R₀) = span{R₀, A·R₀, …, A^{m-1}·R₀}
//! ```
//!
//! and every RHS is solved by a least-squares projection onto that shared
//! basis at each block step.
//!
//! **Algorithm (block GMRES(m) with deflation):**
//! - Initial block QR of the (preconditioned) residual block `R₀ = M⁻¹(B − A·X₀)`
//!   with rank-drop deflation (CGS2 per column; a direction whose norm after
//!   orthogonalisation falls below `sqrt(eps)` of its original norm is dropped).
//! - Block Arnoldi: `W = M⁻¹·A·V_j`, then iterated block classical Gram-Schmidt
//!   against the whole basis with a DGKS reorthogonalisation trigger. The block
//!   Gram matrices `Vᴴ·W` and the projection `W −= V·(Vᴴ·W)` are evaluated as
//!   BLAS-3 products through `T::mat_mul_nn` (SME2-accelerated where available).
//! - Each new block is QR-factorised column-by-column (CGS2) with the same
//!   rank-drop deflation, so numerically rank-deficient blocks (e.g. two
//!   identical RHS) shrink the block instead of breaking down.
//! - The small block-Hessenberg least-squares problem is solved after every
//!   block step by a dense Givens QR of the `(p + s') × p` Hessenberg matrix,
//!   applied to all `s` right-hand sides at once; the per-RHS residual norms
//!   are the tails of the transformed right-hand side.
//! - Block products `A·X` (initial/restart residual, final true-residual
//!   recomputation) go through [`LinearOperator::apply_block`], so dense
//!   operators pay one GEMM instead of `s` matvecs. An optional block
//!   initial guess `X₀` is supported (`R₀ = B − A·X₀`).
//! - At exit the per-RHS *true* residuals `‖b_j − A·x_j‖` are recomputed
//!   with a single block product into `true_residuals`; `converged` gates on
//!   them unless `true_residual_check` is disabled.
//!
//! **Advantages over solving s systems independently:**
//! - Single preconditioner setup (AMG, ILU, etc.)
//! - One shared Krylov basis: each RHS also searches directions generated from
//!   the *other* right-hand sides, which often speeds up convergence.
//! - Block inner products via BLAS-3 reduce synchronisation overhead and
//!   improve cache utilisation.
//!
//! **Use case:** Frequency sweeps where the matrix changes but the preconditioner
//! is built once, or multiple load cases in structural analysis.

use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, Preconditioner, SolverError, SolverResultContract,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::cgs_orth::KrylovOpCounts;
use super::checked_callbacks::{CheckedLinearOperator, CheckedPreconditioner};
use super::gmres_common::{exact_residual_check_due, givens_rotation, solve_upper_triangular};

/// Configuration for block GMRES.
#[derive(Debug, Clone)]
pub struct BlockGmresConfig<R> {
    /// Maximum outer iterations (restarts).
    pub max_iterations: usize,
    /// Restart parameter: number of block Arnoldi steps per restart cycle.
    /// The shared basis grows by up to `n_rhs` vectors per block step, so
    /// each RHS sees the same Krylov depth as scalar GMRES(restart).
    pub restart: usize,
    /// Relative tolerance for convergence (per RHS).
    pub tolerance: R,
    /// Gate `converged` on the recomputed per-RHS *true* residuals
    /// (`‖b_j − A·x_j‖`) instead of the projected Arnoldi estimates.
    ///
    /// The true residuals are always recomputed at exit (one
    /// [`LinearOperator::apply_block`]) and reported in
    /// [`BlockGmresSolution::true_residuals`]; this flag only selects which
    /// residual vector `converged` reflects. Default true.
    pub true_residual_check: bool,
    /// Recompute true residuals every N block Arnoldi steps while iterating.
    /// A value of zero disables optional checkpoints; projected convergence
    /// is still gated by `true_residual_check`.
    pub exact_residual_check_interval: usize,
}

impl Default for BlockGmresConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-6,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        }
    }
}

impl<R: Float> BlockGmresConfig<R> {
    /// Validate configuration before the block core allocates Krylov state.
    pub fn validate(&self) -> Result<(), SolverError> {
        if self.max_iterations == 0 {
            return Err(SolverError::OperationFailed {
                operation: "block GMRES configuration",
                reason: "max_iterations must be greater than zero".into(),
            });
        }
        if self.restart == 0 {
            return Err(SolverError::OperationFailed {
                operation: "block GMRES configuration",
                reason: "restart must be greater than zero".into(),
            });
        }
        if !self.tolerance.is_finite() || self.tolerance <= R::zero() {
            return Err(SolverError::OperationFailed {
                operation: "block GMRES configuration",
                reason: "tolerance must be finite and positive".into(),
            });
        }
        Ok(())
    }
}

/// Result of block GMRES.
#[derive(Debug)]
pub struct BlockGmresSolution<T: ComplexField> {
    /// Solution matrix: n × k (one column per RHS).
    pub x: sonium_solvers_core::dense::DenseMatrix<T>,
    /// Total block Arnoldi steps performed.
    pub iterations: usize,
    /// Projected relative residual estimate per RHS at termination: the
    /// tails of the Givens-transformed block-Hessenberg right-hand side.
    /// Left-preconditioned runs normalize by `‖M⁻¹·b_j‖`, matching the
    /// preconditioned Krylov space that drives the convergence test.
    pub residuals: Vec<T::Real>,
    /// Recomputed true relative residual per RHS, `‖b_j − A·x_j‖ / ‖b_j‖`
    /// (unpreconditioned, the convention external residual gates use).
    /// Always computed at exit with a single [`LinearOperator::apply_block`].
    pub true_residuals: Vec<T::Real>,
    /// Normwise backward error per RHS:
    /// `‖b_j − A·x_j‖ / max(‖A·x_j‖ + ‖b_j‖, min-positive)`.
    pub backward_errors: Vec<T::Real>,
    /// Whether the requested tolerance was attained for each RHS.
    pub attainable_accuracy: Vec<AccuracyStatus>,
    /// Whether all RHS converged (on `true_residuals` when
    /// [`BlockGmresConfig::true_residual_check`] is set, else on the
    /// projected `residuals`).
    pub converged: bool,
    /// Memory-traffic oriented operation counts for the block Krylov solve.
    pub op_counts: KrylovOpCounts,
}

/// Solve A*X = B for k RHS simultaneously using block GMRES.
pub fn block_gmres<T, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    config: &BlockGmresConfig<T::Real>,
) -> BlockGmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    block_gmres_with_guess(operator, b, None, config)
}

/// Solve A*X = B with preconditioner using block GMRES.
pub fn block_gmres_preconditioned<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    precond: &P,
    config: &BlockGmresConfig<T::Real>,
) -> BlockGmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    block_gmres_preconditioned_with_guess(operator, b, precond, None, config)
}

/// Solve A*X = B from an optional block initial guess `x0` (n × k, one
/// column per RHS; `None` starts from zero). The initial residual block
/// `R₀ = B − A·X₀` is formed with a single [`LinearOperator::apply_block`].
pub fn block_gmres_with_guess<T, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    config: &BlockGmresConfig<T::Real>,
) -> BlockGmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    block_gmres_inner(
        operator,
        b,
        x0,
        None::<&sonium_solvers_core::traits::IdentityPreconditioner>,
        config,
        None,
    )
    .expect("unabortable block GMRES cannot fail")
}

/// Preconditioned variant of [`block_gmres_with_guess`].
pub fn block_gmres_preconditioned_with_guess<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    precond: &P,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    config: &BlockGmresConfig<T::Real>,
) -> BlockGmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    block_gmres_inner(operator, b, x0, Some(precond), config, None)
        .expect("unabortable block GMRES cannot fail")
}

/// Checked block GMRES boundary for an unpreconditioned solve.
pub fn try_block_gmres<T, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    config: &BlockGmresConfig<T::Real>,
) -> Result<BlockGmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    try_block_gmres_preconditioned(
        operator,
        b,
        &sonium_solvers_core::traits::IdentityPreconditioner,
        None,
        config,
    )
}

/// Checked block GMRES boundary with an optional block initial guess.
///
/// This validates public dimensions and numerical inputs before entering the
/// infallible compatibility kernel, and converts callback panics or generated
/// non-finite values into typed solver failures.
pub fn try_block_gmres_preconditioned<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    precond: &P,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    config: &BlockGmresConfig<T::Real>,
) -> Result<BlockGmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    config.validate()?;

    let rows = operator.num_rows();
    let cols = operator.num_cols();
    if rows != cols {
        return Err(SolverError::DimensionMismatch {
            expected: cols,
            got: rows,
        });
    }
    if b.nrows() != cols {
        return Err(SolverError::DimensionMismatch {
            expected: cols,
            got: b.nrows(),
        });
    }
    if b.ncols() == 0 {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "at least one right-hand side is required".to_string(),
        });
    }
    if b.as_slice().iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(x0) = x0
        && (x0.nrows() != cols || x0.ncols() != b.ncols())
    {
        return Err(SolverError::DimensionMismatch {
            expected: cols * b.ncols(),
            got: x0.nrows() * x0.ncols(),
        });
    }
    if let Some(x0) = x0
        && x0.as_slice().iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }

    let checked_operator = CheckedLinearOperator::<T, A>::new(operator);
    let checked_precond = CheckedPreconditioner::<T, P>::new(precond);
    catch_unwind(AssertUnwindSafe(|| {
        let result = block_gmres_inner(
            &checked_operator,
            b,
            x0,
            Some(&checked_precond),
            config,
            None,
        )?;
        validate_checked_block_gmres_result(&result, cols, b.ncols())?;
        SolverResultContract::validate_contract(&result)?;
        Ok(result)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "block GMRES",
        reason: panic_reason(payload),
    })?
}

/// Checked block GMRES with a cooperative abort callback.
///
/// The callback is sampled before each restart cycle and between block
/// operator/preconditioner applications. It is intended for an atomic,
/// caller-owned cancellation flag and never crosses into a language-runtime
/// callback from a solver worker. An abort is returned as a typed solver
/// error, so callers cannot accidentally publish a partial result as success.
pub fn try_block_gmres_preconditioned_with_abort<T, A, P, F>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    precond: &P,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    config: &BlockGmresConfig<T::Real>,
    should_abort: F,
) -> Result<BlockGmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
    F: Fn() -> bool,
{
    config.validate()?;

    let rows = operator.num_rows();
    let cols = operator.num_cols();
    if rows != cols {
        return Err(SolverError::DimensionMismatch {
            expected: cols,
            got: rows,
        });
    }
    if b.nrows() != cols {
        return Err(SolverError::DimensionMismatch {
            expected: cols,
            got: b.nrows(),
        });
    }
    if b.ncols() == 0 {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "at least one right-hand side is required".to_string(),
        });
    }
    if should_abort() {
        return Err(block_gmres_aborted());
    }
    if b.as_slice().iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(x0) = x0
        && (x0.nrows() != cols || x0.ncols() != b.ncols())
    {
        return Err(SolverError::DimensionMismatch {
            expected: cols * b.ncols(),
            got: x0.nrows() * x0.ncols(),
        });
    }
    if let Some(x0) = x0
        && x0.as_slice().iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }

    let checked_operator = CheckedLinearOperator::<T, A>::new(operator);
    let checked_precond = CheckedPreconditioner::<T, P>::new(precond);
    catch_unwind(AssertUnwindSafe(|| {
        let result = block_gmres_inner(
            &checked_operator,
            b,
            x0,
            Some(&checked_precond),
            config,
            Some(&should_abort),
        )?;
        validate_checked_block_gmres_result(&result, cols, b.ncols())?;
        SolverResultContract::validate_contract(&result)?;
        Ok(result)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "block GMRES",
        reason: panic_reason(payload),
    })?
}

fn block_gmres_aborted() -> SolverError {
    SolverError::OperationFailed {
        operation: "block GMRES",
        reason: "aborted by caller".to_string(),
    }
}

fn validate_checked_block_gmres_result<T: ComplexField>(
    result: &BlockGmresSolution<T>,
    rows: usize,
    rhs_count: usize,
) -> Result<(), SolverError> {
    if result.x.nrows() != rows || result.x.ncols() != rhs_count {
        return Err(SolverError::DimensionMismatch {
            expected: rows * rhs_count,
            got: result.x.nrows() * result.x.ncols(),
        });
    }
    if result
        .x
        .as_slice()
        .iter()
        .any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "solver generated NaN or infinity in the solution".to_string(),
        });
    }
    if result.residuals.len() != rhs_count
        || result.true_residuals.len() != rhs_count
        || result.backward_errors.len() != rhs_count
        || result.attainable_accuracy.len() != rhs_count
    {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "solver generated invalid result-vector lengths".to_string(),
        });
    }
    if result
        .residuals
        .iter()
        .chain(result.true_residuals.iter())
        .chain(result.backward_errors.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "block GMRES",
            reason: "solver generated NaN or infinity in residual metrics".to_string(),
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
        "block GMRES callback panicked".to_string()
    }
}

/// `out[i, j] = <v_i, w_j> = Σ_r conj(V[r, i]) · W[r, j]` (conjugated block Gram).
///
/// `v` is a column-major block of `p` basis vectors (`p·n`, vector `i` at
/// `v[i·n..]`), `w` a column-major block of `sj` vectors (`sj·n`), and `out`
/// is `p × sj` row-major.
///
/// Computed through the BLAS-3 path as `conj(V_flat · conj(W)ᵀ)` so that the
/// `n`-long reductions run inside `T::mat_mul_nn`.
pub(crate) fn block_gram_into<T: ComplexField>(
    v: &[T],
    p: usize,
    w: &[T],
    sj: usize,
    n: usize,
    wct: &mut [T],
    out: &mut [T],
) {
    debug_assert!(v.len() >= p * n);
    debug_assert!(w.len() >= sj * n);
    debug_assert!(wct.len() >= n * sj);
    debug_assert!(out.len() >= p * sj);
    // wct[r, j] = conj(W[r, j]) as an n × sj row-major matrix.
    for j in 0..sj {
        for r in 0..n {
            wct[r * sj + j] = w[j * n + r].conj();
        }
    }
    T::mat_mul_nn(p, sj, n, v, wct, out);
    for val in out[..p * sj].iter_mut() {
        *val = val.conj();
    }
}

/// `W −= V · Gram` where `Gram` is `p × sj` row-major and the blocks use the
/// same column-major layout as [`block_gram_into`].
///
/// Evaluated as `(Wᵀ) −= Gramᵀ · V_flat` through `T::mat_mul_nn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_project_into<T: ComplexField>(
    v: &[T],
    p: usize,
    gram: &[T],
    sj: usize,
    n: usize,
    gram_t: &mut [T],
    proj: &mut [T],
    w: &mut [T],
) {
    debug_assert!(gram_t.len() >= sj * p);
    debug_assert!(proj.len() >= sj * n);
    for i in 0..p {
        for j in 0..sj {
            gram_t[j * p + i] = gram[i * sj + j];
        }
    }
    T::mat_mul_nn(sj, n, p, gram_t, v, proj);
    for k in 0..sj * n {
        w[k] -= proj[k];
    }
}

/// Frobenius norm of a column-major block of `count` vectors of length `n`.
pub(crate) fn block_frob_norm<T: ComplexField>(w: &[T]) -> T::Real {
    let mut acc = T::Real::zero();
    for val in w {
        acc += val.norm_sqr();
    }
    acc.sqrt()
}

/// `X += V · Y` with `V` the column-major basis block (`p` vectors of length
/// `n` in `vstore`) and `Y` the `p × s` row-major minimiser, evaluated as
/// `(V·Y)ᵀ = Yᵀ·V_flat` through `T::mat_mul_nn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_block_update<T: ComplexField>(
    x: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    vstore: &[T],
    y: &[T],
    p: usize,
    s: usize,
    n: usize,
    yt: &mut [T],
    proj: &mut [T],
) {
    for i in 0..p {
        for j in 0..s {
            yt[j * p + i] = y[i * s + j];
        }
    }
    T::mat_mul_nn(s, n, p, yt, vstore, proj);
    for j in 0..s {
        for i in 0..n {
            x[[i, j]] += proj[j * n + i];
        }
    }
}

fn block_true_metrics<T, A>(
    operator: &A,
    x: &sonium_solvers_core::dense::DenseMatrix<T>,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    b_norms: &[T::Real],
    ax_block: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    op_counts: &mut KrylovOpCounts,
) -> (Vec<T::Real>, Vec<T::Real>)
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let n = b.nrows();
    let s = b.ncols();
    let tiny = T::Real::from_f64(1e-30).unwrap();
    operator.apply_block(x, ax_block);
    op_counts.spmv_count += s;
    op_counts.exact_residual_checks += 1;

    let mut true_residuals = Vec::with_capacity(s);
    let mut backward_errors = Vec::with_capacity(s);
    for j in 0..s {
        let mut residual_sq = T::Real::zero();
        let mut ax_sq = T::Real::zero();
        let mut rhs_sq = T::Real::zero();
        for i in 0..n {
            let applied = ax_block[[i, j]];
            let rhs = b[[i, j]];
            residual_sq += (rhs - applied).norm_sqr();
            ax_sq += applied.norm_sqr();
            rhs_sq += rhs.norm_sqr();
        }
        op_counts.axpy_passes += 1;
        op_counts.norm_count += 1;
        let abs_residual = residual_sq.sqrt();
        true_residuals.push(if b_norms[j] <= tiny {
            abs_residual
        } else {
            abs_residual / b_norms[j]
        });
        backward_errors.push(T::normwise_backward_error(
            abs_residual,
            ax_sq.sqrt(),
            rhs_sq.sqrt(),
        ));
    }
    (true_residuals, backward_errors)
}

/// Solve the small block-Hessenberg least-squares problem shared by all RHS:
///
/// ```text
/// min_{y_j} || g_j − H̄·y_j ||,   j = 0..s
/// ```
///
/// where `H̄` is the `rows × cols` leading block of `h` (`rows ≥ cols`) and
/// `g` has the initial block-QR coefficients `c0` (`s1 × s` row-major) in its
/// first `s1` rows and zeros below.
///
/// A dense Givens QR of `H̄` is computed once and applied to all `s`
/// right-hand sides simultaneously. Returns `(y, tail_norms)` with `y` the
/// `cols × s` row-major minimiser and `tail_norms[j]` the resulting residual
/// norm of RHS `j` (the norm of rows `cols..rows` of the transformed `g_j`,
/// exact because the Arnoldi basis is orthonormal).
pub(crate) fn solve_block_ls<T: ComplexField>(
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    rows: usize,
    cols: usize,
    c0: &[T],
    s1: usize,
    s: usize,
) -> (Vec<T>, Vec<T::Real>) {
    let tiny = T::Real::from_f64(1e-30).unwrap();

    let mut hw = sonium_solvers_core::dense::DenseMatrix::from_elem(rows, cols, T::zero());
    for i in 0..rows {
        for j in 0..cols {
            hw[[i, j]] = h[[i, j]];
        }
    }
    let mut gw = sonium_solvers_core::dense::DenseMatrix::from_elem(rows, s, T::zero());
    for i in 0..s1.min(rows) {
        for j in 0..s {
            gw[[i, j]] = c0[i * s + j];
        }
    }

    // Givens QR of H̄, applying each rotation to the s right-hand sides too.
    for col in 0..cols {
        for row in (col + 1)..rows {
            if hw[[row, col]].norm() < tiny {
                continue;
            }
            let (cr, sr) = givens_rotation(hw[[col, col]], hw[[row, col]]);
            for j in col..cols {
                let a = hw[[col, j]];
                let bb = hw[[row, j]];
                hw[[col, j]] = cr.conj() * a + sr.conj() * bb;
                hw[[row, j]] = T::zero() - sr * a + cr * bb;
            }
            for j in 0..s {
                let a = gw[[col, j]];
                let bb = gw[[row, j]];
                gw[[col, j]] = cr.conj() * a + sr.conj() * bb;
                gw[[row, j]] = T::zero() - sr * a + cr * bb;
            }
        }
    }

    // Back-substitution per RHS on the triangular factor.
    let mut y = vec![T::zero(); cols * s];
    let mut gj = vec![T::zero(); cols];
    for j in 0..s {
        for i in 0..cols {
            gj[i] = gw[[i, j]];
        }
        let yj = solve_upper_triangular(&hw, &gj, cols);
        for i in 0..cols {
            y[i * s + j] = yj[i];
        }
    }

    let mut tail = vec![T::Real::zero(); s];
    for i in cols..rows {
        for j in 0..s {
            tail[j] += gw[[i, j]].norm_sqr();
        }
    }
    for t in tail.iter_mut() {
        *t = t.sqrt();
    }

    (y, tail)
}

fn block_gmres_inner<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    precond: Option<&P>,
    config: &BlockGmresConfig<T::Real>,
    should_abort: Option<&dyn Fn() -> bool>,
) -> Result<BlockGmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    abort_if_requested(should_abort)?;
    let n = b.nrows();
    let s = b.ncols();
    let mut op_counts = KrylovOpCounts::default();
    let tiny = T::Real::from_f64(1e-30).unwrap();
    // Rank-drop tolerance for deflation, relative to the direction's
    // pre-orthogonalisation norm.
    let defl_tol = T::Real::epsilon().sqrt();
    let dgks_threshold = T::Real::from_f64(0.707).unwrap();

    // Compute b norms for relative residual, one multi-vector preconditioner
    // application for the whole block.
    // When left-preconditioned, use ||M^{-1} b|| for consistency with
    // the preconditioned Krylov space.
    let b_norms: Vec<T::Real> = match precond {
        Some(pc) => {
            let mut b_eff = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero());
            pc.apply_block(b, &mut b_eff);
            op_counts.preconditioner_count += s;
            (0..s)
                .map(|j| {
                    op_counts.norm_count += 1;
                    let mut acc = T::Real::zero();
                    for i in 0..n {
                        acc += b_eff[[i, j]].norm_sqr();
                    }
                    acc.sqrt()
                })
                .collect()
        }
        None => (0..s)
            .map(|j| {
                op_counts.norm_count += 1;
                let bj = b.column(j).to_owned();
                vector_norm(&bj)
            })
            .collect(),
    };

    abort_if_requested(should_abort)?;

    let mut x = match x0 {
        Some(x0) => {
            assert_eq!(x0.nrows(), n, "block initial guess row mismatch");
            assert_eq!(x0.ncols(), s, "block initial guess column mismatch");
            x0.clone()
        }
        None => sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero()),
    };
    // Unpreconditioned ‖b_j‖ for the true-residual recomputation at exit:
    // `true_residuals` reports ‖b_j − A·x_j‖ / ‖b_j‖, the same convention
    // callers use for post-solve residual gates, independent of the
    // preconditioned norm that drives the Krylov convergence test.
    let b_true_norms: Vec<T::Real> = (0..s)
        .map(|j| {
            let mut acc = T::Real::zero();
            for i in 0..n {
                acc += b[[i, j]].norm_sqr();
            }
            acc.sqrt()
        })
        .collect();
    let mut total_iters = 0usize;
    // Projected per-RHS relative residuals from the most recent least-squares
    // solve (set from the initial residual block at the top of each cycle;
    // infinity until then so a zero-cycle run cannot report convergence).
    let mut projected = vec![T::Real::infinity(); s];

    // Block GMRES(m): m = restart block Arnoldi steps per cycle. The initial
    // residual block contributes up to s vectors, then each Arnoldi step adds
    // up to s more. Keep room for the initial block plus all requested steps;
    // otherwise restart = 1 would fill the budget before the first step.
    let max_steps = config.restart.max(1);
    let max_basis = (max_steps + 1) * s;

    // Shared block Krylov basis, column-major: vector `i` at `vstore[i·n..]`.
    // Capacity: max_basis accepted vectors + one in-flight block.
    let mut vstore = vec![T::zero(); (max_basis + s) * n];

    let mut r_flat = vec![T::zero(); s * n]; // residual block, column-major
    let mut w_flat = vec![T::zero(); s * n]; // new Arnoldi block, column-major
    let mut axj_buf = vec![T::zero(); n];
    let mut w_buf = vec![T::zero(); n];
    let mut dots = Vec::new();
    let mut gram = vec![T::zero(); max_basis * s];
    let mut gram_t = vec![T::zero(); s * max_basis];
    let mut proj = vec![T::zero(); s * n];
    let mut wct = vec![T::zero(); n * s];
    let mut c0 = vec![T::zero(); max_basis * s];
    let mut yt = vec![T::zero(); s * max_basis];
    // Row-major n × s scratch for multi-vector operator/preconditioner
    // products (the Krylov basis itself stays column-major).
    let mut ax_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero());
    let mut r_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero());
    let mut rp_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero());

    // Block Hessenberg matrix and per-RHS residual norms, reused across
    // restart cycles instead of reallocated. `h` is zeroed per cycle (the
    // Arnoldi updates below accumulate with `+=`); `col_norms` is fully
    // overwritten at the top of each cycle before any read.
    let mut h =
        sonium_solvers_core::dense::DenseMatrix::from_elem(max_basis + s, max_basis, T::zero());
    let mut col_norms = vec![T::Real::zero(); s];

    // Outer restart loop
    'cycle: for _cycle in 0..config.max_iterations {
        abort_if_requested(should_abort)?;
        // ---- Preconditioned residual block R₀ = M⁻¹(B − A·X), column-major ----
        // A·X is formed with ONE multi-vector product (a GEMM for dense
        // operators), then scattered into the column-major Krylov layout.
        operator.apply_block(&x, &mut ax_block);
        op_counts.spmv_count += s;
        for k in 0..n * s {
            r_block.as_mut_slice()[k] = b.as_slice()[k] - ax_block.as_slice()[k];
        }
        op_counts.axpy_passes += 1;
        let rp: &sonium_solvers_core::dense::DenseMatrix<T> = if let Some(pc) = precond {
            pc.apply_block(&r_block, &mut rp_block);
            op_counts.preconditioner_count += s;
            &rp_block
        } else {
            &r_block
        };
        let mut all_converged = true;
        for j in 0..s {
            let mut acc = T::Real::zero();
            for i in 0..n {
                let v = rp[[i, j]];
                acc += v.norm_sqr();
                r_flat[j * n + i] = v;
            }
            let rn = acc.sqrt();
            op_counts.norm_count += 1;
            col_norms[j] = rn;
            projected[j] = if b_norms[j] >= tiny {
                rn / b_norms[j]
            } else {
                T::Real::zero()
            };
            if b_norms[j] >= tiny && projected[j] > config.tolerance {
                all_converged = false;
            }
        }
        if all_converged {
            break;
        }

        // ---- Initial block QR of R₀ with rank-drop deflation (CGS2) ----
        let mut p = 0usize; // accepted basis vectors
        for j in 0..s {
            w_buf.copy_from_slice(&r_flat[j * n..(j + 1) * n]);
            for _pass in 0..2 {
                if p == 0 {
                    break;
                }
                for a in 0..p {
                    let qa = &vstore[a * n..(a + 1) * n];
                    dots.push(inner_product(qa, &w_buf));
                }
                for (a, d) in dots.drain(..).enumerate() {
                    let qa = &vstore[a * n..(a + 1) * n];
                    axpy(T::zero() - d, qa, &mut w_buf);
                }
                op_counts.add_basis_projection::<T>(p, n);
            }
            let wn = vector_norm(&w_buf);
            op_counts.norm_count += 1;
            if wn > defl_tol * col_norms[j].max(tiny) {
                let inv = T::from_real(T::Real::one() / wn);
                let slot = &mut vstore[p * n..(p + 1) * n];
                for i in 0..n {
                    slot[i] = w_buf[i] * inv;
                }
                p += 1;
            }
        }
        if p == 0 {
            // Numerically zero residual block; nothing to project onto.
            break;
        }
        let s1 = p;

        // C₀ = V₁ᴴ·R₀ (s1 × s row-major) via BLAS-3. Columns of R₀ that were
        // deflated away are represented by their projection onto the kept
        // basis; the dropped component is below the deflation tolerance.
        block_gram_into(&vstore, p, &r_flat, s, n, &mut wct, &mut c0);
        op_counts.add_basis_projection::<T>(p * s, n);

        // Block Hessenberg matrix; rows grow past `p` by one block at most.
        // Zero the reused buffer: entries accumulate with `+=` below.
        h.fill(T::zero());

        let mut converged_cycle = false;
        let mut sj = s1; // size of the latest basis block

        // ---- Block Arnoldi loop ----
        for _step in 0..max_steps {
            abort_if_requested(should_abort)?;
            if p >= max_basis {
                break;
            }
            let p_start = p - sj; // start of the latest block

            // W = M⁻¹·A·V_j, column by column. The basis is column-major,
            // so routing this through `apply_block` would add an O(n·sj)
            // gather/scatter transpose pair per step on top of the matvecs,
            // and for matrix-free operators the default `apply_block`
            // degenerates to this same per-column loop anyway. The block
            // Gram/projection updates below already run as BLAS-3, so the
            // per-column matvec is kept here deliberately.
            for c in 0..sj {
                abort_if_requested(should_abort)?;
                let v_col = &vstore[(p_start + c) * n..(p_start + c + 1) * n];
                operator.apply_into(v_col, &mut axj_buf);
                op_counts.spmv_count += 1;
                if let Some(pc) = precond {
                    pc.apply_into(&axj_buf, &mut w_buf);
                    op_counts.preconditioner_count += 1;
                } else {
                    w_buf.copy_from_slice(&axj_buf);
                }
                abort_if_requested(should_abort)?;
                w_flat[c * n..(c + 1) * n].copy_from_slice(&w_buf);
            }

            // Iterated block CGS against the whole basis, with a DGKS
            // reorthogonalisation trigger on the block Frobenius norm.
            let frob_before = block_frob_norm::<T>(&w_flat[..sj * n]);
            op_counts.norm_count += 1;
            for pass in 0..2 {
                block_gram_into(&vstore, p, &w_flat, sj, n, &mut wct, &mut gram);
                for i in 0..p {
                    for c in 0..sj {
                        h[[i, p_start + c]] += gram[i * sj + c];
                    }
                }
                block_project_into(
                    &vstore,
                    p,
                    &gram,
                    sj,
                    n,
                    &mut gram_t,
                    &mut proj,
                    &mut w_flat,
                );
                op_counts.add_basis_projection::<T>(p * sj, n);
                if pass == 0 {
                    let frob_after = block_frob_norm::<T>(&w_flat[..sj * n]);
                    op_counts.norm_count += 1;
                    if frob_after >= dgks_threshold * frob_before {
                        break;
                    }
                }
            }

            // QR of the new block with rank-drop deflation (CGS2 per column
            // against the already-accepted directions of this block). New
            // directions are only accepted while the basis budget lasts.
            let budget = max_basis - p;
            let mut s_next = 0usize;
            for c in 0..sj {
                w_buf.copy_from_slice(&w_flat[c * n..(c + 1) * n]);
                let orig = vector_norm(&w_buf);
                op_counts.norm_count += 1;
                for _pass in 0..2 {
                    for a in 0..s_next {
                        let qa = &vstore[(p + a) * n..(p + a + 1) * n];
                        dots.push(inner_product(qa, &w_buf));
                    }
                    for (a, d) in dots.drain(..).enumerate() {
                        let qa = &vstore[(p + a) * n..(p + a + 1) * n];
                        h[[p + a, p_start + c]] += d;
                        axpy(T::zero() - d, qa, &mut w_buf);
                    }
                }
                if s_next > 0 {
                    op_counts.add_basis_projection::<T>(s_next, n);
                }
                let wn = vector_norm(&w_buf);
                op_counts.norm_count += 1;
                if wn > defl_tol * orig.max(tiny) && s_next < budget {
                    h[[p + s_next, p_start + c]] = T::from_real(wn);
                    let inv = T::from_real(T::Real::one() / wn);
                    let slot = &mut vstore[(p + s_next) * n..(p + s_next + 1) * n];
                    for i in 0..n {
                        slot[i] = w_buf[i] * inv;
                    }
                    s_next += 1;
                }
            }

            // Small dense least-squares solve for all RHS at once.
            total_iters += 1;
            let rows = p + s_next;
            let (y, tail) = solve_block_ls(&h, rows, p, &c0, s1, s);
            for j in 0..s {
                projected[j] = if b_norms[j] >= tiny {
                    tail[j] / b_norms[j]
                } else {
                    T::Real::zero()
                };
            }
            let cycle_done = (0..s).all(|j| projected[j] <= config.tolerance);

            // Use the exact candidate both for optional interval checks and
            // for the mandatory gate when the projected estimate claims that
            // every RHS converged. A rejected candidate does not replace the
            // current Krylov base unless that projected gate was optimistic;
            // in that case it becomes the next restart base.
            let exact_check_due = s_next > 0
                && (exact_residual_check_due(config.exact_residual_check_interval, total_iters)
                    || (cycle_done && config.true_residual_check));
            if exact_check_due {
                abort_if_requested(should_abort)?;
                let mut candidate = x.clone();
                apply_block_update(&mut candidate, &vstore, &y, p, s, n, &mut yt, &mut proj);
                op_counts.add_basis_update::<T>(p * s, n);
                let (candidate_true_residuals, _) = block_true_metrics(
                    operator,
                    &candidate,
                    b,
                    &b_true_norms,
                    &mut ax_block,
                    &mut op_counts,
                );
                let exact_converged = candidate_true_residuals
                    .iter()
                    .all(|&residual| residual <= config.tolerance);
                if exact_converged {
                    x = candidate;
                    converged_cycle = true;
                    break;
                }
                if cycle_done && config.true_residual_check {
                    x = candidate;
                    continue 'cycle;
                }
            }

            if cycle_done || s_next == 0 {
                // Update X += V·Y on the current basis and stop the cycle:
                // either converged, or the Krylov space is exhausted (an
                // invariant subspace was found; the LS residual is then
                // minimal for this basis).
                apply_block_update(&mut x, &vstore, &y, p, s, n, &mut yt, &mut proj);
                op_counts.add_basis_update::<T>(p * s, n);
                converged_cycle = cycle_done;
                break;
            }

            p += s_next;
            sj = s_next;
        }

        if !converged_cycle && p > s1 {
            // Block-step budget exhausted: final least-squares solve over the
            // full basis (the Hessenberg is now square), then restart.
            let (y, tail) = solve_block_ls(&h, p, p, &c0, s1, s);
            for j in 0..s {
                projected[j] = if b_norms[j] >= tiny {
                    tail[j] / b_norms[j]
                } else {
                    T::Real::zero()
                };
            }
            apply_block_update(&mut x, &vstore, &y, p, s, n, &mut yt, &mut proj);
            op_counts.add_basis_update::<T>(p * s, n);
        }

        if converged_cycle {
            break;
        }
    }

    // Recompute the per-RHS true residuals ‖b_j − A·x_j‖ (unpreconditioned,
    // relative to ‖b_j‖) with a single multi-vector product.
    abort_if_requested(should_abort)?;
    let (true_residuals, backward_errors) = block_true_metrics(
        operator,
        &x,
        b,
        &b_true_norms,
        &mut ax_block,
        &mut op_counts,
    );
    let attainable_accuracy = true_residuals
        .iter()
        .map(|&residual| {
            if !residual.is_finite() {
                AccuracyStatus::NotChecked
            } else if residual <= config.tolerance {
                AccuracyStatus::Attained
            } else {
                AccuracyStatus::NotAttained
            }
        })
        .collect();

    let converged = if config.true_residual_check {
        true_residuals.iter().all(|&r| r <= config.tolerance)
    } else {
        projected.iter().all(|&r| r <= config.tolerance)
    };

    Ok(BlockGmresSolution {
        x,
        iterations: total_iters,
        residuals: projected,
        true_residuals,
        backward_errors,
        attainable_accuracy,
        converged,
        op_counts,
    })
}

fn abort_if_requested(should_abort: Option<&dyn Fn() -> bool>) -> Result<(), SolverError> {
    if should_abort.is_some_and(|callback| callback()) {
        Err(block_gmres_aborted())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_sparse::CsrMatrix;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(2.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    fn laplacian_1d_complex(n: usize) -> CsrMatrix<Complex64> {
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            col_indices.push(i);
            values.push(Complex64::new(2.0, 0.5)); // complex shift: non-Hermitian
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    fn check_residual<T: ComplexField, A: sonium_solvers_core::traits::LinearOperator<T>>(
        a: &A,
        x: &sonium_solvers_core::dense::DenseMatrix<T>,
        b: &sonium_solvers_core::dense::DenseMatrix<T>,
        j: usize,
    ) -> T::Real {
        let xj = x.column(j).to_owned();
        let bj = b.column(j).to_owned();
        let axj = a.apply(&xj);
        let mut res_sqr = T::Real::zero();
        let mut b_sqr = T::Real::zero();
        for i in 0..bj.len() {
            res_sqr += (axj[i] - bj[i]).norm_sqr();
            b_sqr += bj[i].norm_sqr();
        }
        res_sqr.sqrt() / b_sqr.sqrt()
    }

    #[test]
    fn test_try_block_gmres_rejects_non_finite_operator_output() {
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

        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(2, 1, |_, _| 1.0);
        let error = try_block_gmres(&NonFiniteOperator, &b, &BlockGmresConfig::default())
            .expect_err("non-finite block operator output must be a typed failure");

        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("block GMRES"));
    }

    #[test]
    fn test_try_block_gmres_abort_is_typed_and_sampled_between_steps() {
        let n = 80;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| (i + j + 1) as f64);
        let preconditioner = sonium_solvers_core::traits::IdentityPreconditioner;
        let calls = AtomicUsize::new(0);
        let config = BlockGmresConfig {
            max_iterations: 20,
            restart: 4,
            tolerance: 1e-14,
            true_residual_check: true,
            exact_residual_check_interval: 1,
        };

        let error = try_block_gmres_preconditioned_with_abort(
            &a,
            &b,
            &preconditioner,
            None,
            &config,
            || calls.fetch_add(1, Ordering::Relaxed) >= 3,
        )
        .expect_err("the cooperative abort must stop the block solve");

        assert!(error.to_string().contains("aborted by caller"));
        assert!(calls.load(Ordering::Relaxed) >= 3);
    }

    #[test]
    fn test_block_gmres_single_rhs() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 1, |i, _| i as f64 + 1.0);

        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 30,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(
            result.converged,
            "Single RHS should converge: res={:.2e}",
            result.residuals[0]
        );
    }

    #[test]
    fn test_block_gmres_restart_one_performs_an_arnoldi_step() {
        let a = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);
        let b =
            sonium_solvers_core::dense::DenseMatrix::from_fn(
                2,
                1,
                |i, _| {
                    if i == 0 { 2.0 } else { -3.0 }
                },
            );
        let config = BlockGmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1e-12,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);

        assert!(
            result.converged,
            "restart=1 must still make one Arnoldi step"
        );
        assert_eq!(result.iterations, 1);
        assert!(result.true_residuals[0] < 1e-12);
        assert!((result.x[[0, 0]] - 2.0).abs() < 1e-12);
        assert!((result.x[[1, 0]] + 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_block_gmres_reports_krylov_op_counts() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            if j == 0 { 1.0 } else { (i as f64).sin() }
        });

        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 30,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);

        assert!(result.converged);
        assert!(result.op_counts.spmv_count >= result.iterations);
        assert!(result.op_counts.dot_reductions > 0);
        assert!(result.op_counts.norm_count > 0);
        assert!(result.op_counts.axpy_passes > 0);
        assert!(result.op_counts.basis_bytes_touched > 0);
    }

    #[test]
    fn test_block_gmres_multiple_rhs() {
        let n = 50;
        let a = laplacian_1d(n);

        // 3 different RHS
        let mut b = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 3, 0.0_f64);
        for i in 0..n {
            b[[i, 0]] = 1.0;
            b[[i, 1]] = (i as f64).sin();
            b[[i, 2]] = if i < n / 2 { 1.0 } else { -1.0 };
        }

        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 40,
            tolerance: 1e-8,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(
            result.converged,
            "All RHS should converge: {:?}",
            result.residuals
        );
        assert_eq!(result.x.ncols(), 3);

        // Verify each solution
        for j in 0..3 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-7, "RHS {j}: rel residual = {rel:.2e}");
        }
    }

    #[test]
    fn test_block_gmres_preconditioned() {
        let n = 100;
        let a = laplacian_1d(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let mut b = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 2, 0.0_f64);
        for i in 0..n {
            b[[i, 0]] = 1.0;
            b[[i, 1]] = (i as f64 * 0.1).cos();
        }

        let config = BlockGmresConfig {
            max_iterations: 200,
            restart: 50,
            tolerance: 1e-8,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres_preconditioned(&a, &b, &precond, &config);
        // Verify solutions are reasonable (preconditioned residual may differ from true)
        for j in 0..2 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-4, "Preconditioned RHS {j}: rel res = {rel:.2e}");
        }
    }

    #[test]
    fn test_block_gmres_matches_individual() {
        let n = 30;
        let a = laplacian_1d(n);
        let b1 = vec![1.0_f64; n];
        let b2 = vec![2.0_f64; n];

        // Solve individually
        let cfg = crate::GmresConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let r1 = crate::gmres(&a, &b1, &cfg);
        let r2 = crate::gmres(&a, &b2, &cfg);

        // Solve as block
        let mut b_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 2, 0.0_f64);
        for i in 0..n {
            b_block[[i, 0]] = 1.0;
            b_block[[i, 1]] = 2.0;
        }
        let block_cfg = BlockGmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };
        let rb = block_gmres(&a, &b_block, &block_cfg);

        assert!(rb.converged);

        // Solutions should match
        for i in 0..n {
            assert!(
                (rb.x[[i, 0]] - r1.x[i]).abs() < 1e-6,
                "RHS 0 mismatch at {i}"
            );
            assert!(
                (rb.x[[i, 1]] - r2.x[i]).abs() < 1e-6,
                "RHS 1 mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_block_gmres_matches_scalar_accuracy() {
        // Distinct, non-trivial RHS: the shared block basis must reach the
        // same per-RHS accuracy as scalar GMRES.
        let n = 50;
        let a = laplacian_1d(n);

        let rhs: Vec<Vec<f64>> = vec![
            (0..n).map(|i| (i as f64 * 0.3).sin() + 1.0).collect(),
            (0..n).map(|i| (i as f64 * 0.7).cos()).collect(),
            (0..n)
                .map(|i| if i % 3 == 0 { 1.0 } else { -0.5 })
                .collect(),
        ];

        let cfg = crate::GmresConfig {
            max_iterations: 100,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let scalar_x: Vec<Vec<f64>> = rhs.iter().map(|b| crate::gmres(&a, b, &cfg).x).collect();

        let s = rhs.len();
        let mut b_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, 0.0_f64);
        for (j, bj) in rhs.iter().enumerate() {
            for i in 0..n {
                b_block[[i, j]] = bj[i];
            }
        }
        let block_cfg = BlockGmresConfig {
            max_iterations: 100,
            restart: 40,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };
        let rb = block_gmres(&a, &b_block, &block_cfg);

        assert!(rb.converged, "block residuals: {:?}", rb.residuals);
        for j in 0..s {
            let rel = check_residual(&a, &rb.x, &b_block, j);
            assert!(rel < 1e-9, "RHS {j}: block rel residual = {rel:.2e}");
            for i in 0..n {
                assert!(
                    (rb.x[[i, j]] - scalar_x[j][i]).abs() < 1e-5,
                    "RHS {j} mismatch at {i}: block={:.6e} scalar={:.6e}",
                    rb.x[[i, j]],
                    scalar_x[j][i]
                );
            }
        }
    }

    #[test]
    fn test_block_gmres_rank_deficient_block() {
        // Two identical RHS plus one independent: the block is numerically
        // rank-deficient and must be deflated, not break down.
        let n = 40;
        let a = laplacian_1d(n);

        let mut b = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 3, 0.0_f64);
        for i in 0..n {
            b[[i, 0]] = (i as f64 + 1.0).sqrt();
            b[[i, 1]] = b[[i, 0]]; // duplicate direction
            b[[i, 2]] = (i as f64 * 0.2).cos();
        }

        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 30,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(
            result.converged,
            "Rank-deficient block must still converge: {:?}",
            result.residuals
        );

        // Identical RHS → identical solutions.
        for i in 0..n {
            assert!(
                (result.x[[i, 0]] - result.x[[i, 1]]).abs() < 1e-8,
                "duplicated RHS solutions differ at {i}"
            );
        }
        for j in 0..3 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-9, "RHS {j}: rel residual = {rel:.2e}");
        }
    }

    #[test]
    fn test_block_gmres_zero_rhs_column() {
        let n = 32;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            if j == 0 {
                0.0
            } else {
                (i as f64 * 0.31).sin() + 0.25
            }
        });
        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 24,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(
            result.converged,
            "zero-RHS block must still converge: {:?}",
            result.residuals
        );
        for i in 0..n {
            assert_eq!(
                result.x[[i, 0]],
                0.0,
                "zero RHS produced a nonzero solution at {i}"
            );
        }
        assert!(
            result.true_residuals[0] < 1e-12,
            "zero RHS absolute residual = {:.2e}",
            result.true_residuals[0]
        );
        let rel = check_residual(&a, &result.x, &b, 1);
        assert!(rel < 1e-9, "nonzero RHS rel residual = {rel:.2e}");
    }

    #[test]
    fn test_block_gmres_scaled_dependent_rhs_columns() {
        let n = 36;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 3, |i, j| {
            let base = (i as f64 * 0.23).cos() + 0.4;
            match j {
                0 => base,
                1 => 2.0 * base,
                _ => (i as f64 * 0.47).sin() - 0.1,
            }
        });
        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 24,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(
            result.converged,
            "scaled-dependent block must still converge: {:?}",
            result.residuals
        );
        for i in 0..n {
            assert!(
                (result.x[[i, 1]] - 2.0 * result.x[[i, 0]]).abs() < 1e-8,
                "scaled dependent solutions differ at {i}"
            );
        }
        for j in 0..3 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-9, "RHS {j}: rel residual = {rel:.2e}");
        }
    }

    #[test]
    fn test_block_gmres_complex64() {
        let n = 40;
        let a = laplacian_1d_complex(n);

        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            if j == 0 {
                Complex64::new((i as f64).sin(), 1.0)
            } else {
                Complex64::new(1.0, (i as f64 * 0.1).cos())
            }
        });

        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 40,
            tolerance: 1e-8,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(
            result.converged,
            "Complex block GMRES should converge: {:?}",
            result.residuals
        );
        for j in 0..2 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-7, "Complex RHS {j}: rel residual = {rel:.2e}");
        }
    }

    /// Dense operator wrapper whose `apply_block` is a single GEMM, mirroring
    /// the production wrappers in sonium-bem / sonium-sim-speaker.
    struct DenseOp<'a>(&'a sonium_solvers_core::dense::DenseMatrix<f64>);

    impl LinearOperator<f64> for DenseOp<'_> {
        fn num_rows(&self) -> usize {
            self.0.nrows()
        }
        fn num_cols(&self) -> usize {
            self.0.ncols()
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            self.0.matvec_into(x, y);
        }
        fn apply_block(
            &self,
            x: &sonium_solvers_core::dense::DenseMatrix<f64>,
            y: &mut sonium_solvers_core::dense::DenseMatrix<f64>,
        ) {
            self.0.matmul_into(x, y);
        }
        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            self.0.matvec_transpose_into(x, y);
        }
    }

    #[test]
    fn test_apply_block_matches_per_column_matvec() {
        // Deterministic pseudo-random dense operator.
        let n = 24;
        let k = 3;
        let prng = |s: usize| {
            let h = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                >> 11;
            (h % 2000) as f64 / 1000.0 - 1.0
        };
        let a = sonium_solvers_core::dense::DenseMatrix::from_fn(n, n, |i, j| prng(i * n + j));
        let op = DenseOp(&a);
        let x =
            sonium_solvers_core::dense::DenseMatrix::from_fn(n, k, |i, j| prng(n * n + j * n + i));

        let mut y_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, k, 0.0);
        op.apply_block(&x, &mut y_block);

        for j in 0..k {
            let xj = x.column_to_vec(j);
            let mut yj = vec![0.0; n];
            op.apply_into(&xj, &mut yj);
            let err = (0..n)
                .map(|i| (y_block[[i, j]] - yj[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale = yj.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
            assert!(
                err <= 1e-12 * scale,
                "GEMM apply_block vs matvec column {j}: err={err:.2e}"
            );
        }
    }

    #[test]
    fn test_default_apply_block_matches_per_column_matvec() {
        // CsrMatrix does not override `apply_block`: the trait default must
        // reproduce per-column `apply_into` exactly.
        let n = 30;
        let a = laplacian_1d(n);
        let x = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 3, |i, j| {
            ((i * 7 + j * 13) % 11) as f64
        });

        let mut y_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 3, 0.0);
        a.apply_block(&x, &mut y_block);

        for j in 0..3 {
            let xj = x.column_to_vec(j);
            let mut yj = vec![0.0; n];
            a.apply_into(&xj, &mut yj);
            for i in 0..n {
                assert_eq!(y_block[[i, j]], yj[i], "default apply_block at ({i}, {j})");
            }
        }
    }

    #[test]
    fn test_block_gmres_with_guess_reduces_iterations() {
        // Shifted Laplacian: well-conditioned enough that GMRES reduces the
        // residual geometrically, so a warm start measurably saves steps.
        let n = 50;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(6.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            if j == 0 { 1.0 } else { (i as f64).sin() }
        });
        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 40,
            tolerance: 1e-8,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let cold = block_gmres(&a, &b, &config);
        assert!(cold.converged);
        assert!(cold.iterations > 0);

        // Warm start from the exact solution: the initial residual check
        // accepts the guess before a single Arnoldi step runs.
        let exact = block_gmres_with_guess(&a, &b, Some(&cold.x), &config);
        assert!(exact.converged);
        assert_eq!(exact.iterations, 0, "exact guess needs no Arnoldi steps");
        assert!(exact.true_residuals.iter().all(|&r| r <= config.tolerance));

        // Perturbed guess (~1e-3 relative residual): the warm run skips the
        // first three decades the cold run has to earn from scratch.
        let perturbed =
            sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| cold.x[[i, j]] * 1.001);
        let warm = block_gmres_with_guess(&a, &b, Some(&perturbed), &config);
        assert!(warm.converged, "warm residuals: {:?}", warm.true_residuals);
        assert!(
            warm.iterations < cold.iterations,
            "warm start ({}) should beat cold start ({})",
            warm.iterations,
            cold.iterations
        );
        for j in 0..2 {
            let rel = check_residual(&a, &warm.x, &b, j);
            assert!(rel < 1e-7, "warm RHS {j}: rel residual = {rel:.2e}");
        }
    }

    #[test]
    fn test_block_gmres_true_residuals_match_direct_computation() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => (i as f64).sin(),
            _ => {
                if i < n / 2 {
                    1.0
                } else {
                    -1.0
                }
            }
        });
        let config = BlockGmresConfig {
            max_iterations: 50,
            restart: 40,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(result.converged);
        // Unpreconditioned run: b_norms are ‖b_j‖, so `true_residuals` must
        // equal the directly computed ‖b_j − A·x_j‖ / ‖b_j‖.
        for j in 0..3 {
            let direct = check_residual(&a, &result.x, &b, j);
            assert!(
                (result.true_residuals[j] - direct).abs() < 1e-14,
                "RHS {j}: reported true residual {:.3e} vs direct {direct:.3e}",
                result.true_residuals[j]
            );
            // On a converged Arnoldi projection the true residual tracks the
            // projected estimate to within rounding (the projected value can
            // be exactly zero when an invariant subspace is found).
            assert!(
                result.true_residuals[j] <= result.residuals[j] * 10.0 + 1e-12,
                "RHS {j}: true {:.3e} vs projected {:.3e}",
                result.true_residuals[j],
                result.residuals[j]
            );
            assert!(result.backward_errors[j].is_finite());
            assert_eq!(result.attainable_accuracy[j], AccuracyStatus::Attained);
        }
    }

    #[test]
    fn test_block_gmres_converged_gates_on_configured_residual() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            if j == 0 { 1.0 } else { (i as f64 * 0.2).cos() }
        });
        let tolerance = 1e-10;
        for true_residual_check in [true, false] {
            let config = BlockGmresConfig {
                max_iterations: 50,
                restart: 40,
                tolerance,
                true_residual_check,
                exact_residual_check_interval: 0,
            };
            let result = block_gmres(&a, &b, &config);
            let gated = if true_residual_check {
                &result.true_residuals
            } else {
                &result.residuals
            };
            assert_eq!(
                result.converged,
                gated.iter().all(|&r| r <= tolerance),
                "converged must reflect the configured residual vector \
                 (true_residual_check={true_residual_check})"
            );
            assert!(result.converged);
        }

        // A run that never enters a cycle (max_iterations = 0) must not
        // report convergence under either gate.
        let config = BlockGmresConfig {
            max_iterations: 0,
            restart: 40,
            tolerance,
            true_residual_check: false,
            exact_residual_check_interval: 0,
        };
        let result = block_gmres(&a, &b, &config);
        assert!(!result.converged, "zero-cycle run must not converge");
        assert!(
            result.true_residuals.iter().all(|&r| r > tolerance),
            "zero-cycle true residuals are the initial residuals"
        );
    }

    #[test]
    fn test_block_gmres_honors_exact_residual_interval_before_projected_convergence() {
        let n = 8;
        let a = laplacian_1d(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            (i as f64 + 1.0) * (j as f64 + 1.5)
        });
        let base_config = BlockGmresConfig {
            max_iterations: 20,
            restart: 20,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };
        let checked_config = BlockGmresConfig {
            exact_residual_check_interval: 1,
            ..base_config.clone()
        };

        let base = block_gmres_preconditioned(&a, &b, &precond, &base_config);
        let checked = block_gmres_preconditioned(&a, &b, &precond, &checked_config);

        assert!(base.converged, "base residuals: {:?}", base.true_residuals);
        assert!(
            checked.converged,
            "checked residuals: {:?}",
            checked.true_residuals
        );
        assert!(
            checked.op_counts.exact_residual_checks > base.op_counts.exact_residual_checks,
            "configured interval must check before projected convergence: base={}, checked={}",
            base.op_counts.exact_residual_checks,
            checked.op_counts.exact_residual_checks,
        );
    }

    #[test]
    fn test_block_gmres_restart_one_reuses_hessenberg_across_cycles() {
        // Guards the hoisted Hessenberg/column-norm state: with restart 1
        // every block Arnoldi step ends its cycle, so a converging solve
        // reuses the cleared buffers across many cycles. Block solutions
        // carry no restart counter, so iterations > 1 proves multi-cycle
        // execution (one step per cycle at restart 1).
        // Diagonal system: restarted block GMRES converges here (unlike on
        // the Laplacian, where restart-1 stalls), while still spanning many
        // single-step cycles.
        let n = 25;
        let triplets: Vec<(usize, usize, f64)> =
            (0..n).map(|i| (i, i, 1.0 + (i % 5) as f64 * 0.5)).collect();
        let a = CsrMatrix::from_triplets(n, n, triplets);
        let mut b = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 2, 0.0_f64);
        for i in 0..n {
            b[[i, 0]] = 1.0;
            b[[i, 1]] = 2.0;
        }
        let config = BlockGmresConfig {
            max_iterations: 100,
            restart: 1,
            tolerance: 1e-8,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_gmres(&a, &b, &config);
        assert!(
            result.iterations > 1,
            "test must span multiple restart cycles, iterations={}",
            result.iterations
        );
        assert!(
            result.converged,
            "true residuals: {:?}",
            result.true_residuals
        );
        for j in 0..2 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-6, "RHS {j}: true rel res = {rel:.2e}");
        }
    }
}
