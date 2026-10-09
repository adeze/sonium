//! Block FGMRES for multiple right-hand sides with flexible right
//! preconditioning
//!
//! Solves `A * X = B` where `B` is `n × s` (s right-hand sides simultaneously)
//! with a true block method: a single block Arnoldi recurrence builds one
//! shared block Krylov basis, and every RHS is solved by a least-squares
//! projection onto that shared basis at each block step.
//!
//! This is the block analogue of [`fgmres`](super::fgmres): like scalar
//! FGMRES it applies the preconditioner on the **right** and stores the
//! preconditioned basis blocks
//!
//! ```text
//! Z_j = M_j⁻¹ · V_j,   W = A · Z_j,
//! ```
//!
//! so the preconditioner may **vary from step to step** (inner-outer schemes,
//! AMG with variable smoothing, iteration-dependent scaling) — the solution
//! update `X += Z·Y` uses exactly the `Z` directions that generated the
//! space. A block GMRES recurrence (which applies a fixed `M⁻¹` to the
//! residual and to every Arnoldi vector) has no such guarantee when `M`
//! changes between steps.
//!
//! Because the preconditioning is on the right, the Arnoldi minimisation
//! targets the **true** residual `B − A·X`, so the projected residual
//! estimates are normalized by the unpreconditioned `‖b_j‖` (unlike the
//! left-preconditioned [`block_gmres`](super::block_gmres), which normalizes
//! by `‖M⁻¹·b_j‖`).
//!
//! **Structure** (mirrors [`block_gmres`](super::block_gmres)):
//! - Initial block QR of the residual block `R₀ = B − A·X₀` with rank-drop
//!   deflation (CGS2 per column).
//! - Block Arnoldi with iterated block classical Gram-Schmidt and a DGKS
//!   reorthogonalisation trigger; block Gram matrices and projections run as
//!   BLAS-3 products through `T::mat_mul_nn`.
//! - The small block-Hessenberg least-squares problem is solved after every
//!   block step by a dense Givens QR applied to all `s` right-hand sides at
//!   once; the per-RHS residual norms are the tails of the transformed
//!   right-hand side.
//! - Block products `A·X` (initial/restart residual, final true-residual
//!   recomputation) go through [`LinearOperator::apply_block`]. An optional
//!   block initial guess `X₀` is supported (warm start: `R₀ = B − A·X₀`).
//! - At exit the per-RHS *true* residuals `‖b_j − A·x_j‖` are recomputed
//!   with a single block product into `true_residuals`; `converged` gates on
//!   them unless `true_residual_check` is disabled.
//!
//! **Inner-loop layout note:** the Krylov basis is column-major, so the
//! per-step `Z = M⁻¹·V_j` and `W = A·Z_j` products are applied per column
//! (equivalent to what the default per-column `apply_block` would do after
//! an `O(n·s)` transpose round-trip). The block Gram/projection updates —
//! the dominant `O(p·s·n)` work — already run as BLAS-3.
//!
//! **Subspace recycling:** there is no block GCRO-DR here. Warm-start `X₀`
//! (previous solution block) is the supported reuse mechanism; true
//! recycled-subspace block GCRO-DR is future work.

use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, LinearOperator, Preconditioner, SolverError, SolverResultContract,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::block_gmres::{
    apply_block_update, block_frob_norm, block_gram_into, block_project_into, solve_block_ls,
};
use super::cgs_orth::KrylovOpCounts;
use super::checked_callbacks::CheckedLinearOperator;
use super::gmres_common::exact_residual_check_due;
use std::sync::atomic::{AtomicBool, Ordering};

/// Configuration for block FGMRES.
#[derive(Debug, Clone)]
pub struct BlockFgmresConfig<R> {
    /// Maximum outer iterations (restarts).
    pub max_iterations: usize,
    /// Restart parameter: number of block Arnoldi steps per restart cycle.
    /// The shared basis grows by up to `n_rhs` vectors per block step, so
    /// each RHS sees the same Krylov depth as scalar FGMRES(restart).
    pub restart: usize,
    /// Relative tolerance for convergence (per RHS), on the true residual
    /// `‖b_j − A·x_j‖ / ‖b_j‖` (right preconditioning minimises the true
    /// residual directly).
    pub tolerance: R,
    /// Gate `converged` on the recomputed per-RHS *true* residuals
    /// (`‖b_j − A·x_j‖`) instead of the projected Arnoldi estimates.
    ///
    /// The true residuals are always recomputed at exit (one
    /// [`LinearOperator::apply_block`]) and reported in
    /// [`BlockFgmresSolution::true_residuals`]; this flag only selects which
    /// residual vector `converged` reflects. Default true.
    pub true_residual_check: bool,
    /// Check candidate blocks against the original operator every N block
    /// Arnoldi steps. Zero disables optional interval checks; a projected
    /// convergence gate still performs an exact check when
    /// `true_residual_check` is enabled.
    pub exact_residual_check_interval: usize,
}

impl<R: Float> BlockFgmresConfig<R> {
    /// Validate configuration before allocating a block Krylov basis.
    pub fn validate(&self) -> Result<(), SolverError> {
        if self.max_iterations == 0 {
            return Err(SolverError::OperationFailed {
                operation: "block FGMRES configuration",
                reason: "max_iterations must be greater than zero".to_string(),
            });
        }
        if self.restart == 0 {
            return Err(SolverError::OperationFailed {
                operation: "block FGMRES configuration",
                reason: "restart must be greater than zero".to_string(),
            });
        }
        if !self.tolerance.is_finite() || self.tolerance <= R::zero() {
            return Err(SolverError::OperationFailed {
                operation: "block FGMRES configuration",
                reason: "tolerance must be finite and positive".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for BlockFgmresConfig<f64> {
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

/// Result of block FGMRES.
#[derive(Debug)]
pub struct BlockFgmresSolution<T: ComplexField> {
    /// Solution matrix: n × k (one column per RHS).
    pub x: sonium_solvers_core::dense::DenseMatrix<T>,
    /// Total block Arnoldi steps performed.
    pub iterations: usize,
    /// Projected relative residual estimate per RHS at termination: the
    /// tails of the Givens-transformed block-Hessenberg right-hand side,
    /// normalized by the unpreconditioned `‖b_j‖` (right preconditioning
    /// minimises the true residual in the Arnoldi space of `A·M⁻¹`, so these
    /// estimates track the true residuals up to basis orthonormality).
    pub residuals: Vec<T::Real>,
    /// Recomputed true relative residual per RHS, `‖b_j − A·x_j‖ / ‖b_j‖`.
    /// Always computed at exit with a single [`LinearOperator::apply_block`].
    pub true_residuals: Vec<T::Real>,
    /// Normwise backward error per RHS:
    /// `‖b_j − A·x_j‖ / max(‖A·x_j‖ + ‖b_j‖, min-positive)`.
    pub backward_errors: Vec<T::Real>,
    /// Whether the requested tolerance was attained for each RHS.
    pub attainable_accuracy: Vec<AccuracyStatus>,
    /// Whether all RHS converged (on `true_residuals` when
    /// [`BlockFgmresConfig::true_residual_check`] is set, else on the
    /// projected `residuals`).
    pub converged: bool,
    /// Memory-traffic oriented operation counts for the block Krylov solve.
    pub op_counts: KrylovOpCounts,
}

/// Solve A*X = B for k RHS simultaneously using block FGMRES with a (possibly
/// variable) right preconditioner.
pub fn block_fgmres_preconditioned<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    precond: &P,
    config: &BlockFgmresConfig<T::Real>,
) -> BlockFgmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    block_fgmres_preconditioned_with_guess(operator, b, precond, None, config)
}

/// Checked block FGMRES entry point for user-controlled shapes and
/// configuration. The unchecked historical entry point remains available for
/// callers that have already validated their solver inputs.
pub fn try_block_fgmres_preconditioned<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    precond: &P,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    config: &BlockFgmresConfig<T::Real>,
) -> Result<BlockFgmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    config.validate()?;
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if b.nrows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: b.nrows(),
        });
    }
    if b.as_slice().iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "block FGMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if b.ncols() == 0 {
        return Err(SolverError::OperationFailed {
            operation: "block FGMRES",
            reason: "at least one right-hand side is required".to_string(),
        });
    }
    if let Some(x0) = x0
        && (x0.nrows() != operator.num_cols() || x0.ncols() != b.ncols())
    {
        return Err(SolverError::DimensionMismatch {
            expected: b.nrows() * b.ncols(),
            got: x0.nrows() * x0.ncols(),
        });
    }
    if let Some(x0) = x0
        && x0.as_slice().iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "block FGMRES",
            reason: "initial guess contains NaN or infinity".to_string(),
        });
    }

    let invalid_preconditioner = AtomicBool::new(false);
    let checked_precond = FiniteCheckedPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner,
    };
    let checked_operator = CheckedLinearOperator::<T, A>::new(operator);
    catch_unwind(AssertUnwindSafe(|| {
        let mut result = block_fgmres_preconditioned_with_guess(
            &checked_operator,
            b,
            &checked_precond,
            x0,
            config,
        );
        if invalid_preconditioner.load(Ordering::Relaxed) {
            return Err(SolverError::OperationFailed {
                operation: "block FGMRES preconditioner",
                reason: "preconditioner generated NaN or infinity".to_string(),
            });
        }
        finalize_checked_block_result(&checked_operator, b, config, &mut result)?;
        SolverResultContract::validate_contract(&result)?;
        Ok(result)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "block FGMRES",
        reason: panic_reason(payload),
    })?
}

/// Checked adapter for the infallible preconditioner callback used by the
/// block kernel. Invalid values are replaced before they can contaminate
/// later arithmetic, while the atomic flag preserves the typed failure at the
/// public boundary.
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
        let mut y = vec![T::zero(); r.len()];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        self.inner.apply_into(r, y);
        if y.len() != r.len() || y.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            y.fill(T::zero());
        }
    }
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "block FGMRES callback panicked".to_string()
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

/// Independently finalize a checked block-FGMRES result against the original
/// operator. The infallible kernel owns its fast projected estimates; this
/// boundary owns the public true-residual, backward-error, and accuracy
/// contract.
fn finalize_checked_block_result<T, A>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    config: &BlockFgmresConfig<T::Real>,
    result: &mut BlockFgmresSolution<T>,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let n = operator.num_cols();
    let s = b.ncols();
    if result.x.nrows() != n || result.x.ncols() != s {
        return Err(SolverError::DimensionMismatch {
            expected: n * s,
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
            operation: "block FGMRES",
            reason: "solver generated NaN or infinity in the solution".to_string(),
        });
    }
    if result.residuals.len() != s {
        return Err(SolverError::OperationFailed {
            operation: "block FGMRES",
            reason: "solver generated an invalid projected-residual vector length".to_string(),
        });
    }
    if result.residuals.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "block FGMRES",
            reason: "solver generated a non-finite projected residual".to_string(),
        });
    }

    let b_norms: Vec<T::Real> = (0..s)
        .map(|j| vector_norm(&b.column(j).to_owned()))
        .collect();
    let mut ax = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero());
    let mut counts = KrylovOpCounts::default();
    let (true_residuals, backward_errors) =
        block_true_metrics(operator, &result.x, b, &b_norms, &mut ax, &mut counts);
    if true_residuals
        .iter()
        .chain(backward_errors.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "block FGMRES",
            reason: "original-operator residual metrics are NaN or infinity".to_string(),
        });
    }

    result.true_residuals = true_residuals;
    result.backward_errors = backward_errors;
    result.attainable_accuracy = result
        .true_residuals
        .iter()
        .map(|&residual| {
            if residual <= config.tolerance {
                AccuracyStatus::Attained
            } else {
                AccuracyStatus::NotAttained
            }
        })
        .collect();
    if config.true_residual_check {
        result.converged = result
            .true_residuals
            .iter()
            .all(|&residual| residual <= config.tolerance);
    }
    Ok(())
}

/// Preconditioned block FGMRES from an optional block initial guess `x0`
/// (n × k, one column per RHS; `None` starts from zero). The initial
/// residual block `R₀ = B − A·X₀` is formed with a single
/// [`LinearOperator::apply_block`].
pub fn block_fgmres_preconditioned_with_guess<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    precond: &P,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    config: &BlockFgmresConfig<T::Real>,
) -> BlockFgmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    block_fgmres_inner(operator, b, x0, precond, config)
}

fn block_fgmres_inner<T, A, P>(
    operator: &A,
    b: &sonium_solvers_core::dense::DenseMatrix<T>,
    x0: Option<&sonium_solvers_core::dense::DenseMatrix<T>>,
    precond: &P,
    config: &BlockFgmresConfig<T::Real>,
) -> BlockFgmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.nrows();
    let s = b.ncols();
    let mut op_counts = KrylovOpCounts::default();
    let tiny = T::Real::from_f64(1e-30).unwrap();
    // Rank-drop tolerance for deflation, relative to the direction's
    // pre-orthogonalisation norm.
    let defl_tol = T::Real::epsilon().sqrt();
    let dgks_threshold = T::Real::from_f64(0.707).unwrap();

    // Right preconditioning minimises the true residual, so both the
    // projected estimates and the exit recomputation normalize by the
    // unpreconditioned ‖b_j‖.
    let b_norms: Vec<T::Real> = (0..s)
        .map(|j| {
            op_counts.norm_count += 1;
            let bj = b.column(j).to_owned();
            vector_norm(&bj)
        })
        .collect();

    let mut x = match x0 {
        Some(x0) => {
            assert_eq!(x0.nrows(), n, "block initial guess row mismatch");
            assert_eq!(x0.ncols(), s, "block initial guess column mismatch");
            x0.clone()
        }
        None => sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero()),
    };
    let mut total_iters = 0usize;
    // Projected per-RHS relative residuals from the most recent least-squares
    // solve (set from the initial residual block at the top of each cycle;
    // infinity until then so a zero-cycle run cannot report convergence).
    let mut projected = vec![T::Real::infinity(); s];

    // Block FGMRES(m): m = restart block Arnoldi steps per cycle. The shared
    // basis grows by up to s vectors per step, so its dimension is at most
    // restart·s (matching the per-RHS Krylov depth of scalar FGMRES(restart)).
    let max_steps = config.restart.max(1);
    let max_basis = max_steps * s;

    // Shared block Krylov basis V and preconditioned basis Z, column-major:
    // vector `i` at `store[i·n..]`. Z is stored per accepted direction, which
    // is what makes a step-dependent (flexible) preconditioner correct.
    // Capacity: max_basis accepted vectors + one in-flight block.
    let mut vstore = vec![T::zero(); (max_basis + s) * n];
    let mut zstore = vec![T::zero(); (max_basis + s) * n];

    let mut r_flat = vec![T::zero(); s * n]; // residual block, column-major
    let mut w_flat = vec![T::zero(); s * n]; // new Arnoldi block, column-major
    let mut axj_buf = vec![T::zero(); n];
    let mut w_buf = vec![T::zero(); n];
    let mut z_buf = vec![T::zero(); n];
    let mut dots = Vec::new();
    let mut gram = vec![T::zero(); max_basis * s];
    let mut gram_t = vec![T::zero(); s * max_basis];
    let mut proj = vec![T::zero(); s * n];
    let mut wct = vec![T::zero(); n * s];
    let mut c0 = vec![T::zero(); max_basis * s];
    let mut yt = vec![T::zero(); s * max_basis];
    // Row-major n × s scratch for multi-vector operator products (the Krylov
    // basis itself stays column-major).
    let mut ax_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero());
    let mut r_block = sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, T::zero());

    // Block Hessenberg matrix and per-RHS residual norms, reused across
    // restart cycles instead of reallocated. `h` is zeroed per cycle (the
    // Arnoldi updates below accumulate with `+=`); `col_norms` is fully
    // overwritten at the top of each cycle before any read.
    let mut h =
        sonium_solvers_core::dense::DenseMatrix::from_elem(max_basis + s, max_basis, T::zero());
    let mut col_norms = vec![T::Real::zero(); s];

    // Outer restart loop
    'cycle: for _cycle in 0..config.max_iterations {
        // ---- Residual block R₀ = B − A·X, column-major ----
        // A·X is formed with ONE multi-vector product, then scattered into
        // the column-major Krylov layout. No preconditioner is applied to
        // the residual: FGMRES is right-preconditioned.
        operator.apply_block(&x, &mut ax_block);
        op_counts.spmv_count += s;
        for k in 0..n * s {
            r_block.as_mut_slice()[k] = b.as_slice()[k] - ax_block.as_slice()[k];
        }
        op_counts.axpy_passes += 1;
        let mut all_converged = true;
        for j in 0..s {
            let mut acc = T::Real::zero();
            for i in 0..n {
                let v = r_block[[i, j]];
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

        // ---- Block Arnoldi loop (right-preconditioned) ----
        for _step in 0..max_steps {
            if p >= max_basis {
                break;
            }
            let p_start = p - sj; // start of the latest block

            // Z_j = M⁻¹·V_j, W = A·Z_j, column by column. The basis is
            // column-major, so routing through `apply_block` would add an
            // O(n·sj) gather/scatter transpose pair per step on top of the
            // applies, and the default `apply_block` degenerates to this
            // same per-column loop anyway. The block Gram/projection updates
            // below already run as BLAS-3, so the per-column apply is kept
            // here deliberately (same trade-off as block GMRES).
            for c in 0..sj {
                let v_col = &vstore[(p_start + c) * n..(p_start + c + 1) * n];
                precond.apply_into(v_col, &mut z_buf);
                op_counts.preconditioner_count += 1;
                // Store Z per direction: the update X += Z·Y must use the
                // exact directions that generated the space, which is what
                // makes a step-dependent (flexible) preconditioner correct.
                zstore[(p_start + c) * n..(p_start + c + 1) * n].copy_from_slice(&z_buf);
                operator.apply_into(&z_buf, &mut axj_buf);
                op_counts.spmv_count += 1;
                w_flat[c * n..(c + 1) * n].copy_from_slice(&axj_buf);
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
                let mut candidate = x.clone();
                apply_block_update(&mut candidate, &zstore, &y, p, s, n, &mut yt, &mut proj);
                op_counts.add_basis_update::<T>(p * s, n);
                let (candidate_true_residuals, _) = block_true_metrics(
                    operator,
                    &candidate,
                    b,
                    &b_norms,
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
                // Update X += Z·Y on the current *preconditioned* basis (the
                // FGMRES difference from block GMRES) and stop the cycle:
                // either converged, or the Krylov space is exhausted (an
                // invariant subspace was found; the LS residual is then
                // minimal for this basis).
                apply_block_update(&mut x, &zstore, &y, p, s, n, &mut yt, &mut proj);
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
            apply_block_update(&mut x, &zstore, &y, p, s, n, &mut yt, &mut proj);
            op_counts.add_basis_update::<T>(p * s, n);
        }

        if converged_cycle {
            break;
        }
    }

    // Recompute the per-RHS true residuals ‖b_j − A·x_j‖ (relative to ‖b_j‖)
    // with a single multi-vector product.
    let (true_residuals, backward_errors) =
        block_true_metrics(operator, &x, b, &b_norms, &mut ax_block, &mut op_counts);
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

    BlockFgmresSolution {
        x,
        iterations: total_iters,
        residuals: projected,
        true_residuals,
        backward_errors,
        attainable_accuracy,
        converged,
        op_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_sparse::CsrMatrix;

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

    fn config(tol: f64) -> BlockFgmresConfig<f64> {
        BlockFgmresConfig {
            max_iterations: 100,
            restart: 40,
            tolerance: tol,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        }
    }

    #[test]
    fn test_block_fgmres_multiple_rhs() {
        let n = 50;
        let a = laplacian_1d(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let mut b = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 3, 0.0_f64);
        for i in 0..n {
            b[[i, 0]] = 1.0;
            b[[i, 1]] = (i as f64).sin();
            b[[i, 2]] = if i < n / 2 { 1.0 } else { -1.0 };
        }

        let result = block_fgmres_preconditioned(&a, &b, &precond, &config(1e-8));
        assert!(
            result.converged,
            "All RHS should converge: proj={:?} true={:?}",
            result.residuals, result.true_residuals
        );
        assert_eq!(result.x.ncols(), 3);
        for j in 0..3 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-7, "RHS {j}: rel residual = {rel:.2e}");
            assert!(result.backward_errors[j].is_finite());
            assert_eq!(result.attainable_accuracy[j], AccuracyStatus::Attained);
        }
        assert!(result.op_counts.spmv_count >= result.iterations);
        assert!(result.op_counts.preconditioner_count >= result.iterations);
    }

    #[test]
    fn test_block_fgmres_matches_single_rhs_fgmres() {
        // Parity with scalar FGMRES per column: same tolerance, residuals
        // comparable and solutions close.
        let n = 40;
        let a = laplacian_1d_complex(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let rhs: Vec<Vec<Complex64>> = vec![
            (0..n)
                .map(|i| Complex64::new((i as f64 * 0.3).sin() + 1.0, 0.1))
                .collect(),
            (0..n)
                .map(|i| Complex64::new((i as f64 * 0.7).cos(), -0.2))
                .collect(),
        ];

        let fcfg = crate::FgmresConfig {
            max_iterations: 100,
            restart: 40,
            tolerance: 1e-9,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
        };
        let scalar: Vec<_> = rhs
            .iter()
            .map(|b| crate::fgmres_preconditioned(&a, &precond, b, None, &fcfg))
            .collect();

        let s = rhs.len();
        let mut b_block =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, s, Complex64::new(0.0, 0.0));
        for (j, bj) in rhs.iter().enumerate() {
            for i in 0..n {
                b_block[[i, j]] = bj[i];
            }
        }
        let rb = block_fgmres_preconditioned(&a, &b_block, &precond, &config(1e-9));

        assert!(rb.converged, "block residuals: {:?}", rb.true_residuals);
        for j in 0..s {
            assert!(
                scalar[j].converged,
                "scalar FGMRES should converge for RHS {j}"
            );
            let rel = check_residual(&a, &rb.x, &b_block, j);
            assert!(
                rel < 10.0 * scalar[j].residual.max(1e-12) + 1e-9,
                "RHS {j}: block rel residual {rel:.2e} vs scalar {:.2e}",
                scalar[j].residual
            );
            for i in 0..n {
                let diff = (rb.x[[i, j]] - scalar[j].x[i]).norm();
                assert!(diff < 1e-5, "RHS {j} solution mismatch at {i}: {diff:.2e}");
            }
        }
    }

    #[test]
    fn test_block_fgmres_deflates_rank_deficient_rhs() {
        // Two identical RHS columns: the block must deflate the duplicate
        // instead of breaking down.
        let n = 30;
        let a = laplacian_1d(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, _| {
            (i as f64 * 0.4).sin() + 0.5
        });
        let result = block_fgmres_preconditioned(&a, &b, &precond, &config(1e-9));
        assert!(
            result.converged,
            "rank-deficient block should still converge: {:?}",
            result.true_residuals
        );
        for j in 0..2 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-8, "RHS {j}: rel residual = {rel:.2e}");
        }
    }

    #[test]
    fn test_block_fgmres_with_guess_reduces_iterations() {
        let n = 60;
        let a = laplacian_1d(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            ((i * 31 + j * 17) % 23) as f64 - 11.0
        });
        let cfg = config(1e-8);

        let cold = block_fgmres_preconditioned(&a, &b, &precond, &cfg);
        assert!(cold.converged);

        // Warm start from the converged solution very slightly perturbed:
        // the initial residual then sits right at a moderate tolerance, so
        // the warm run exits after zero or few block steps while the cold
        // run climbs from O(1).
        let mut x0 = cold.x.clone();
        for v in x0.as_mut_slice() {
            *v += 1e-3;
        }
        let warm_cfg = config(1e-4);
        let warm = block_fgmres_preconditioned_with_guess(&a, &b, &precond, Some(&x0), &warm_cfg);
        assert!(
            warm.converged,
            "warm start residuals: {:?}",
            warm.true_residuals
        );
        assert!(
            warm.iterations < cold.iterations,
            "warm start should cut iterations: warm={} cold={}",
            warm.iterations,
            cold.iterations
        );
    }

    /// Preconditioner that changes at every application: Jacobi scaling
    /// times an iteration-dependent diagonal. This is exactly the case
    /// FGMRES (and block FGMRES) exists for — the preconditioned directions
    /// Z_j differ per step, so a fixed-operator Krylov recurrence is invalid.
    struct NonStationaryJacobi {
        diag_inv: Vec<f64>,
        apply_count: std::sync::atomic::AtomicUsize,
    }

    impl Preconditioner<f64> for NonStationaryJacobi {
        fn apply(&self, r: &[f64]) -> Vec<f64> {
            let mut y = vec![0.0; r.len()];
            self.apply_into(r, &mut y);
            y
        }

        fn apply_into(&self, r: &[f64], y: &mut [f64]) {
            let step = self
                .apply_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Vary the scaling per apply: diag factor oscillates in [0.5, 1.5].
            let phase = 1.0 + 0.5 * (step as f64 * 0.7).sin();
            for (i, (yi, ri)) in y.iter_mut().zip(r.iter()).enumerate() {
                let per_entry = 1.0 + 0.25 * ((i + step) % 5) as f64 * 0.1;
                *yi = phase * per_entry * self.diag_inv[i] * ri;
            }
        }
    }

    #[test]
    fn test_block_fgmres_nonstationary_preconditioner() {
        let n = 50;
        let a = laplacian_1d(n);
        let diag_inv: Vec<f64> = (0..n).map(|i| 1.0 / a.get(i, i)).collect();
        let precond = NonStationaryJacobi {
            diag_inv,
            apply_count: std::sync::atomic::AtomicUsize::new(0),
        };

        let mut b = sonium_solvers_core::dense::DenseMatrix::from_elem(n, 2, 0.0_f64);
        for i in 0..n {
            b[[i, 0]] = 1.0;
            b[[i, 1]] = (i as f64 * 0.2).cos();
        }

        let cfg = BlockFgmresConfig {
            max_iterations: 200,
            restart: 30,
            tolerance: 1e-7,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };
        let result = block_fgmres_preconditioned(&a, &b, &precond, &cfg);
        assert!(
            result.converged,
            "non-stationary preconditioner should still converge: {:?}",
            result.true_residuals
        );
        for j in 0..2 {
            let rel = check_residual(&a, &result.x, &b, j);
            assert!(rel < 1e-6, "RHS {j}: rel residual = {rel:.2e}");
        }
    }

    #[test]
    fn test_block_fgmres_true_residual_gate() {
        // A zero preconditioner produces Z = 0, W = 0: the whole block
        // deflates immediately. The projected residual stays large and the
        // true-residual gate must report non-convergence (no false positive).
        struct ZeroPreconditioner;
        impl Preconditioner<f64> for ZeroPreconditioner {
            fn apply(&self, r: &[f64]) -> Vec<f64> {
                vec![0.0; r.len()]
            }
            fn apply_into(&self, _r: &[f64], y: &mut [f64]) {
                y.fill(0.0);
            }
        }

        let n = 20;
        let a = laplacian_1d(n);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            i as f64 + j as f64 + 1.0
        });

        let result = block_fgmres_preconditioned(&a, &b, &ZeroPreconditioner, &config(1e-8));
        assert!(
            !result.converged,
            "zero preconditioner must not report convergence"
        );
        assert!(
            result.true_residuals.iter().all(|&r| r > 1e-3),
            "true residuals should expose the failure: {:?}",
            result.true_residuals
        );
    }

    #[test]
    fn test_block_fgmres_honors_exact_residual_interval_before_projected_convergence() {
        let n = 8;
        let a = laplacian_1d(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            (i as f64 + 1.0) * (j as f64 + 1.5)
        });
        let base_config = BlockFgmresConfig {
            max_iterations: 20,
            restart: 20,
            tolerance: 1e-10,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };
        let checked_config = BlockFgmresConfig {
            exact_residual_check_interval: 1,
            ..base_config.clone()
        };

        let base = block_fgmres_preconditioned(&a, &b, &precond, &base_config);
        let checked = block_fgmres_preconditioned(&a, &b, &precond, &checked_config);

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
    fn test_try_block_fgmres_validates_and_solves() {
        let n = 4;
        let a = CsrMatrix::from_raw_parts(
            n,
            n,
            vec![0, 1, 2, 3, 4],
            vec![0, 1, 2, 3],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |i, j| {
            (i as f64 + 1.0) * (j as f64 + 2.0)
        });

        let invalid = BlockFgmresConfig {
            max_iterations: 1,
            restart: 0,
            tolerance: 1e-8,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };
        assert!(try_block_fgmres_preconditioned(&a, &b, &precond, None, &invalid).is_err());

        let result = try_block_fgmres_preconditioned(&a, &b, &precond, None, &config(1e-10))
            .expect("identity block system should solve");
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        for j in 0..b.ncols() {
            for i in 0..b.nrows() {
                assert!((result.x[[i, j]] - b[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_try_block_fgmres_rejects_non_finite_preconditioner_output() {
        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply(&self, r: &[f64]) -> Vec<f64> {
                vec![f64::NAN; r.len()]
            }

            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                assert_eq!(r.len(), y.len());
                y.fill(f64::NAN);
            }
        }

        let matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(2, 2, |i, j| (i + j + 1) as f64);
        let config = config(1e-10);
        let error =
            try_block_fgmres_preconditioned(&matrix, &b, &NonFinitePreconditioner, None, &config)
                .expect_err("non-finite block preconditioner output must be typed failure");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("preconditioner"));
    }

    #[test]
    fn test_try_block_fgmres_rejects_non_finite_operator_output() {
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
        let error = try_block_fgmres_preconditioned(
            &NonFiniteOperator,
            &b,
            &sonium_solvers_core::traits::IdentityPreconditioner,
            None,
            &config(1e-10),
        )
        .expect_err("non-finite block operator output must be typed failure");

        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("operator"));
    }

    #[test]
    fn test_block_fgmres_restart_one_reuses_hessenberg_across_cycles() {
        // Guards the hoisted Hessenberg/column-norm state: with restart 1
        // every block Arnoldi step ends its cycle, so a converging solve
        // reuses the cleared buffers across many cycles. Block solutions
        // carry no restart counter, so iterations > 1 proves multi-cycle
        // execution (one step per cycle at restart 1).
        // Diagonal system: restarted block FGMRES converges here (unlike on
        // the Laplacian, where restart-1 stalls), while still spanning many
        // single-step cycles.
        let n = 25;
        let triplets: Vec<(usize, usize, f64)> =
            (0..n).map(|i| (i, i, 1.0 + (i % 5) as f64 * 0.5)).collect();
        let a = CsrMatrix::from_triplets(n, n, triplets);
        let b = sonium_solvers_core::dense::DenseMatrix::from_fn(n, 2, |_, j| (j + 1) as f64);
        // Identity preconditioner: a diagonal preconditioner would solve this
        // system in one step, leaving nothing multi-cycle to guard.
        let precond = sonium_solvers_core::traits::IdentityPreconditioner;
        let config = BlockFgmresConfig {
            max_iterations: 100,
            restart: 1,
            tolerance: 1e-8,
            true_residual_check: true,
            exact_residual_check_interval: 0,
        };

        let result = block_fgmres_preconditioned(&a, &b, &precond, &config);
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
