//! LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
//!
//! Computes the `k` smallest eigenvalues and eigenvectors of a large sparse
//! symmetric positive-definite matrix `A`:
//!
//! ```text
//! A x_i = lambda_i x_i,    i = 1, ..., k
//! ```
//!
//! The generalized entry point solves
//!
//! ```text
//! A x_i = lambda_i B x_i
//! ```
//!
//! for symmetric `A` and symmetric positive-definite `B`.
//!
//! Only needs `A.apply(x)` (SpMV) and optionally `M^{-1}.apply(r)` (preconditioner).
//!
//! Reference: Knyazev, A.V. (2001). "Toward the Optimal Preconditioned Eigensolver:
//! Locally Optimal Block Preconditioned Conjugate Gradient Method." SIAM J. Sci. Comput.
//!
//! # Algorithm
//!
//! Each iteration:
//! 1. Compute residuals `R = A*X - B*X*diag(eigenvalues)` for the active block
//! 2. Soft-lock converged Ritz vectors (verified with a fresh operator apply)
//! 3. Apply preconditioner: `W = M^{-1} R`
//! 4. Orthonormalise `W` against `[locked, X, P]` (block CGS + MGS with
//!    reorthogonalisation; block Gram products go through BLAS-3 `matmul`)
//! 5. Solve the Rayleigh-Ritz problem on `span{X, W, P}` (dense symmetric
//!    eigenproblem via Householder tridiagonalisation + implicit-shift QL)
//! 6. Update `X` (eigenvectors) and `P` (search directions) with one gemm each;
//!    `A*X` / `B*X` are carried through the recurrence as `(A*S)*Z`
//!
//! # Implementation notes
//!
//! - **Soft locking**: once a Ritz pair's residual drops below `tolerance`
//!   (confirmed with a fresh SpMV so roundoff in the carried products cannot
//!   cause false convergence), the vector is frozen out of the active block.
//!   No SpMV, preconditioner, or orthogonalisation work is spent on it again;
//!   new directions are orthogonalised against the locked set, which keeps the
//!   whole computed basis B-orthonormal.
//! - **Block oversampling**: with `block_padding > 0` the block holds
//!   `k + padding` columns while only the `k` smallest pairs are sought and
//!   returned. The extra directions enlarge the trial subspace, which
//!   accelerates convergence on clustered spectra (the block convergence rate
//!   is otherwise set by the worst cluster member). Iteration stops once the
//!   `k` smallest pairs are locked; the unconverged padding columns are
//!   dropped from the output.
//! - **Orthogonalisation**: two block-CGS passes against the orthonormal
//!   basis (one block Gram + one block update, both gemms) followed by
//!   CholQR2 inside the block — QR via the Cholesky factor of the Gram
//!   matrix, applied twice (SLEPc-style one-reorth refinement). When the
//!   Gram is numerically rank-deficient the Cholesky breaks down and the
//!   code falls back to MGS with one reorthogonalisation pass, dropping
//!   dependent columns so the projected Gram matrix stays SPD.
//! - **Projected eigensolver**: Householder tridiagonalisation followed by
//!   the implicit-shift symmetric QL algorithm with Wilkinson shift (the
//!   LAPACK `sytrd`/`steqr` pipeline) — no per-rotation matrix clones.

use sonium_solvers_core::traits::{
    AccuracyStatus, LinearOperator, Preconditioner, SolverError, SolverResultContract, SolverStatus,
};

use crate::result_contract::real_result_shape_valid;

/// Configuration for the LOBPCG solver.
#[derive(Debug, Clone)]
pub struct LobpcgConfig {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on relative residual norm.
    pub tolerance: f64,
    /// Block oversampling: the block holds `num_eigenvalues + block_padding`
    /// columns, but only the `num_eigenvalues` smallest pairs are converged
    /// and returned. The padding directions enlarge the trial subspace and
    /// accelerate clustered spectra. Default 0 (block size == number of
    /// requested eigenvalues).
    pub block_padding: usize,
}

/// Marker operator used to infer the unused generalized-operator type for
/// the standard (non-generalized) entry points.
struct UnusedGeneralizedOperator;

impl LinearOperator<f64> for UnusedGeneralizedOperator {
    fn num_rows(&self) -> usize {
        0
    }

    fn num_cols(&self) -> usize {
        0
    }

    fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
        unreachable!("the unused generalized operator is never applied")
    }

    fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
        unreachable!("the unused generalized operator is never applied")
    }
}

impl Default for LobpcgConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            max_iterations: 200,
            tolerance: 1e-8,
            block_padding: 0,
        }
    }
}

/// Result of the LOBPCG eigenvalue solver.
#[derive(Debug, Clone)]
pub struct LobpcgResult {
    /// Computed eigenvalues (smallest first), length = k.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns of an n×k matrix.
    pub eigenvectors: sonium_solvers_core::dense::DenseMatrix<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Residual norms for each eigenpair at convergence.
    pub residual_norms: Vec<f64>,
    /// Whether all eigenpairs converged.
    pub converged: bool,
}

impl LobpcgResult {
    /// Return the canonical status for the returned Ritz pairs.
    pub fn solver_status(&self) -> SolverStatus {
        let shape_valid = real_result_shape_valid(
            &self.eigenvalues,
            &self.eigenvectors,
            &self.residual_norms,
            false,
        );
        if !shape_valid {
            SolverStatus::NumericalError
        } else if self.converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        }
    }

    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached => AccuracyStatus::NotAttained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverResultContract for LobpcgResult {
    fn solver_status(&self) -> SolverStatus {
        LobpcgResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        LobpcgResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        let status = LobpcgResult::solver_status(self);
        let accuracy = LobpcgResult::accuracy_status(self);
        if status == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "LOBPCG result contract",
                reason: "result shape or diagnostic is non-finite".into(),
            });
        }
        if (status == SolverStatus::Converged) != (accuracy == AccuracyStatus::Attained) {
            return Err(SolverError::OperationFailed {
                operation: "LOBPCG result contract",
                reason: "status and accuracy disagree".into(),
            });
        }
        Ok(())
    }
}

/// Compute the `k` smallest eigenvalues of a real SPD operator without preconditioning.
///
/// Returns `Err(SolverError::ConvergenceError)` if the dense projected
/// eigensolver (implicit-shift QL) fails to converge within its sweep limit.
pub fn lobpcg<A: LinearOperator<f64>>(
    operator: &A,
    config: &LobpcgConfig,
) -> Result<LobpcgResult, SolverError> {
    lobpcg_core(
        operator,
        None::<&UnusedGeneralizedOperator>,
        None::<&sonium_solvers_core::traits::IdentityPreconditioner>,
        config,
        true,
    )
}

/// Compute the `k` smallest eigenvalues of a real SPD operator with preconditioning.
///
/// Returns `Err(SolverError::ConvergenceError)` if the dense projected
/// eigensolver (implicit-shift QL) fails to converge within its sweep limit.
pub fn lobpcg_preconditioned<A, P>(
    operator: &A,
    precond: &P,
    config: &LobpcgConfig,
) -> Result<LobpcgResult, SolverError>
where
    A: LinearOperator<f64>,
    P: Preconditioner<f64>,
{
    lobpcg_core(
        operator,
        None::<&UnusedGeneralizedOperator>,
        Some(precond),
        config,
        true,
    )
}

/// Compute the `k` smallest generalized eigenvalues of `A x = lambda B x`.
///
/// This is the Maxwell/FEM cutoff-frequency variant: `A` is usually a
/// stiffness-like operator and `B` a mass-like SPD operator. The returned
/// eigenvectors are approximately `B`-orthonormal.
///
/// Returns `Err(SolverError::ConvergenceError)` if the dense projected
/// eigensolver (implicit-shift QL) fails to converge within its sweep limit.
pub fn lobpcg_generalized<A, B>(
    a: &A,
    b: &B,
    config: &LobpcgConfig,
) -> Result<LobpcgResult, SolverError>
where
    A: LinearOperator<f64>,
    B: LinearOperator<f64>,
{
    lobpcg_core(
        a,
        Some(b),
        None::<&sonium_solvers_core::traits::IdentityPreconditioner>,
        config,
        true,
    )
}

/// Compute generalized eigenvalues of `A x = lambda B x` with preconditioning.
///
/// The preconditioner is applied to the generalized residual
/// `r = A x - lambda B x`.
///
/// Returns `Err(SolverError::ConvergenceError)` if the dense projected
/// eigensolver (implicit-shift QL) fails to converge within its sweep limit.
pub fn lobpcg_generalized_preconditioned<A, B, P>(
    a: &A,
    b: &B,
    precond: &P,
    config: &LobpcgConfig,
) -> Result<LobpcgResult, SolverError>
where
    A: LinearOperator<f64>,
    B: LinearOperator<f64>,
    P: Preconditioner<f64>,
{
    lobpcg_core(a, Some(b), Some(precond), config, true)
}

/// Deterministic initial block for LOBPCG.
///
/// Every call with the same `(n, k_block)` returns the same block, so tests
/// and downstream QA binaries that depend on reproducible bases are stable.
/// The block must have full numerical rank: `orthonormalize_block_b` drops
/// dependent columns, so a rank-deficient initial block silently shrinks the
/// trial subspace and caps the number of eigenpairs LOBPCG can return.
///
/// The block is filled from a seeded xorshift64* PRNG (fixed seed, uniform
/// on [-0.5, 0.5)); i.i.d. dense random blocks have full numerical rank with
/// probability one and are well-conditioned after one orthonormalisation.
///
/// This replaces the earlier hash generator
/// `((i·2654435761) ^ (j·1013904223)) & 0xFFFF`, whose entries had the form
/// `g(t_i ⊕ c_j)` with `t_i = i·a mod 2¹⁶` and `c_j = j·b mod 2¹⁶`. Such an
/// XOR-Cayley matrix has rank equal to the number of nonzero Walsh–Fourier
/// coefficients of `g`; since `g(x) = x/65536 − 0.5` expands into the 16
/// single-bit parity characters plus the constant character, the rank was
/// capped at 17 regardless of block size, and LOBPCG could return no more
/// than ~17–20 eigenpairs.
fn deterministic_initial_block(
    n: usize,
    k_block: usize,
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    let mut x = sonium_solvers_core::dense::DenseMatrix::zeros(n, k_block);
    let mut rng = XorShift64Star::new(0x853c_49e6_748f_ea9b);
    for j in 0..k_block {
        for i in 0..n {
            x[[i, j]] = rng.next_f64() - 0.5;
        }
    }
    x
}

/// Deterministic xorshift64* generator, mirroring the one in
/// `optimization::unconstrained::cmaes_impl` (kept private so the solver
/// needs no `rand` dependency; a fixed seed gives bit-reproducible initial
/// blocks across runs and platforms).
struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    fn new(seed: u64) -> Self {
        // Offset the seed so a zero seed does not hit the all-zero fixed
        // point of the xorshift recurrence.
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.state = s;
        s.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform on [0, 1) with 53 bits of precision.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Shared LOBPCG core for the standard (`b == None`, identity inner product)
/// and generalized (`b == Some(B)`) pencils.
///
/// `soft_locking` enables deflation of converged Ritz vectors out of the
/// active block. It is always on through the public API; the flag exists so
/// tests can measure the work it saves.
fn lobpcg_validation_error(operation: &'static str, error: SolverError) -> SolverError {
    SolverError::from_failure(operation, &error)
}

fn ensure_finite_dense(
    operation: &'static str,
    values: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> Result<(), SolverError> {
    if values.as_slice().iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SolverError::OperationFailed {
            operation,
            reason: "operator generated NaN or infinity".into(),
        })
    }
}

fn ensure_finite_values(operation: &'static str, values: &[f64]) -> Result<(), SolverError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SolverError::OperationFailed {
            operation,
            reason: "calculation generated NaN or infinity".into(),
        })
    }
}

fn ensure_non_nan_values(operation: &'static str, values: &[f64]) -> Result<(), SolverError> {
    if values.iter().all(|value| !value.is_nan()) {
        Ok(())
    } else {
        Err(SolverError::OperationFailed {
            operation,
            reason: "calculation generated NaN".into(),
        })
    }
}

fn lobpcg_core<A, B, P>(
    a: &A,
    b: Option<&B>,
    precond: Option<&P>,
    config: &LobpcgConfig,
    soft_locking: bool,
) -> Result<LobpcgResult, SolverError>
where
    A: LinearOperator<f64>,
    B: LinearOperator<f64>,
    P: Preconditioner<f64>,
{
    let n = a.num_rows();
    if n == 0 {
        return Err(SolverError::OperationFailed {
            operation: "LOBPCG configuration",
            reason: "operator must have at least one row".into(),
        });
    }
    if a.num_cols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: a.num_cols(),
        });
    }
    if config.num_eigenvalues == 0 || config.num_eigenvalues > n {
        return Err(SolverError::OperationFailed {
            operation: "LOBPCG configuration",
            reason: format!(
                "num_eigenvalues must be in 1..={n}, got {}",
                config.num_eigenvalues
            ),
        });
    }
    if config.max_iterations == 0 {
        return Err(SolverError::OperationFailed {
            operation: "LOBPCG configuration",
            reason: "max_iterations must be greater than zero".into(),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "LOBPCG configuration",
            reason: "tolerance must be finite and positive".into(),
        });
    }
    if let Some(bb) = b
        && (bb.num_rows() != n || bb.num_cols() != n)
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: bb.num_rows(),
        });
    }

    let zero_probe = vec![0.0; n];
    let a_zero = a
        .try_apply(&zero_probe)
        .map_err(|error| lobpcg_validation_error("operator", error))?;
    ensure_finite_values("operator validation", &a_zero)?;
    if let Some(bb) = b {
        let b_zero = bb
            .try_apply(&zero_probe)
            .map_err(|error| lobpcg_validation_error("mass operator", error))?;
        ensure_finite_values("mass operator validation", &b_zero)?;
    }
    if let Some(pc) = precond {
        let precond_zero = pc
            .try_apply(&zero_probe)
            .map_err(|error| lobpcg_validation_error("preconditioner", error))?;
        ensure_finite_values("preconditioner validation", &precond_zero)?;
    }
    // k pairs are sought; the block carries k + padding columns (padding = 0
    // keeps the classic block size). Padding directions are iterated along
    // but never returned.
    let k = config.num_eigenvalues.min(n);
    let k_block = k
        .checked_add(config.block_padding)
        .ok_or_else(|| SolverError::OperationFailed {
            operation: "LOBPCG configuration",
            reason: "num_eigenvalues plus block_padding overflows usize".into(),
        })?
        .min(n);
    let generalized = b.is_some();

    // Initialise X with deterministic pseudo-random vectors.
    let x = deterministic_initial_block(n, k_block);
    let bx_init =
        try_apply_b(b, &x).map_err(|error| lobpcg_validation_error("mass operator", error))?;
    let (x_init, bx_init) = orthonormalize_block_b(&x, &bx_init, None);
    if x_init.ncols() != k_block || bx_init.ncols() != k_block {
        return Err(SolverError::Breakdown {
            operation: "LOBPCG initial block",
            reason: format!(
                "orthonormalization retained {} of {k_block} required columns",
                x_init.ncols()
            ),
        });
    }
    ensure_finite_dense("LOBPCG initial block", &x_init)?;
    ensure_finite_dense("LOBPCG initial mass block", &bx_init)?;
    let mut x = x_init;
    let mut bx = bx_init;
    let mut ax =
        try_apply_columns(a, &x).map_err(|error| lobpcg_validation_error("operator", error))?;
    let mut eigenvalues = generalized_rayleigh_quotients(&x, &ax, &bx);
    ensure_finite_values("LOBPCG initial Rayleigh quotient", &eigenvalues)?;

    // Search directions P (empty on the first iteration).
    let mut p = sonium_solvers_core::dense::DenseMatrix::zeros(n, 0);
    let mut residual_norms = vec![f64::INFINITY; k_block];

    // Deflated (locked) eigenpairs: B-orthonormal, frozen out of the active
    // block, and carried along only for orthogonalisation and final output.
    let mut locked_vals: Vec<f64> = Vec::new();
    let mut locked_res: Vec<f64> = Vec::new();
    let mut locked_q = sonium_solvers_core::dense::DenseMatrix::zeros(n, 0);
    let mut locked_bq = sonium_solvers_core::dense::DenseMatrix::zeros(n, 0);

    let mut converged = false;
    let mut iteration = 0;

    for iter in 0..config.max_iterations {
        iteration = iter + 1;

        // Residuals R = A*X - B*X*Lambda for the active block.
        let mut r = ax.clone();
        for j in 0..x.ncols() {
            for i in 0..n {
                r[[i, j]] -= eigenvalues[j] * bx[[i, j]];
            }
        }
        ensure_finite_dense("LOBPCG residual", &r)?;
        for j in 0..x.ncols() {
            residual_norms[j] =
                column_norm(&r, j) / residual_scale(&ax, &bx, j, eigenvalues[j], generalized);
        }
        ensure_finite_values("LOBPCG residual norm", &residual_norms[..x.ncols()])?;

        if soft_locking {
            // Soft locking: verify convergence with a fresh operator apply
            // (the carried A*x / B*x products accumulate roundoff over
            // iterations, so the cheap residual estimate alone could lock a
            // vector that is not actually converged), then freeze the vector.
            let mut newly_locked: Vec<Vec<f64>> = Vec::new();
            let mut newly_locked_b: Vec<Vec<f64>> = Vec::new();
            let mut keep: Vec<usize> = Vec::with_capacity(x.ncols());
            for j in 0..x.ncols() {
                if residual_norms[j] > config.tolerance {
                    keep.push(j);
                    continue;
                }
                let xj = x.column_to_vec(j);
                let axj = a
                    .try_apply(&xj)
                    .map_err(|error| lobpcg_validation_error("operator", error))?;
                let bxj = match b {
                    Some(bb) => bb
                        .try_apply(&xj)
                        .map_err(|error| lobpcg_validation_error("mass operator", error))?,
                    None => xj.clone(),
                };
                let mut rnorm_sq = 0.0;
                let mut ax_sq = 0.0;
                let mut bx_sq = 0.0;
                for i in 0..n {
                    let ri = axj[i] - eigenvalues[j] * bxj[i];
                    rnorm_sq += ri * ri;
                    ax_sq += axj[i] * axj[i];
                    bx_sq += bxj[i] * bxj[i];
                }
                let scale = if generalized {
                    (ax_sq.sqrt() + eigenvalues[j].abs() * bx_sq.sqrt()).max(1e-30)
                } else {
                    eigenvalues[j].abs().max(1e-30)
                };
                let res_true = rnorm_sq.sqrt() / scale;
                ensure_finite_values("LOBPCG verified residual", &[res_true])?;
                if res_true <= config.tolerance {
                    locked_vals.push(eigenvalues[j]);
                    locked_res.push(res_true);
                    newly_locked.push(xj);
                    newly_locked_b.push(bxj);
                } else {
                    // Drift in the carried products made the vector look
                    // converged; refresh them and keep iterating.
                    ax.set_column(j, &axj);
                    bx.set_column(j, &bxj);
                    residual_norms[j] = res_true;
                    keep.push(j);
                }
            }
            if !newly_locked.is_empty() {
                let new_q = columns_to_matrix(n, newly_locked);
                let new_bq = columns_to_matrix(n, newly_locked_b);
                locked_q = hstack(&[&locked_q, &new_q]);
                locked_bq = hstack(&[&locked_bq, &new_bq]);
            }
            if keep.len() < x.ncols() {
                x = take_columns(&x, &keep);
                ax = take_columns(&ax, &keep);
                bx = take_columns(&bx, &keep);
                p = take_columns(&p, &keep);
                r = take_columns(&r, &keep);
                eigenvalues = keep.iter().map(|&j| eigenvalues[j]).collect();
                residual_norms = keep.iter().map(|&j| residual_norms[j]).collect();
            }
            // Convergence: every requested pair is locked. With oversampling
            // (k_block > k) the padding columns need not converge — stop once
            // the k smallest locked values sit at or below the smallest
            // active Ritz value, so the k returned pairs are all verified
            // converged (locked values are confirmed with a fresh SpMV).
            if locked_vals.len() >= k {
                let active_floor = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
                let mut locked_sorted = locked_vals.clone();
                locked_sorted.sort_by(f64::total_cmp);
                if x.ncols() == 0 || locked_sorted[k - 1] <= active_floor {
                    converged = true;
                    break;
                }
            }
            if x.ncols() == 0 {
                converged = true;
                break;
            }
        } else if residual_norms
            .iter()
            .take(k)
            .all(|&res| res <= config.tolerance)
        {
            // Test-only path without soft locking: the active block stays
            // sorted smallest-first, so the first k columns are the sought
            // pairs (padding columns are allowed to lag behind).
            converged = true;
            break;
        }

        // W = M^{-1} R on the active block only.
        let w0 = match precond {
            Some(pc) => try_apply_precond_columns(pc, &r)
                .map_err(|error| lobpcg_validation_error("preconditioner", error))?,
            None => r,
        };
        let bw0 =
            try_apply_b(b, &w0).map_err(|error| lobpcg_validation_error("mass operator", error))?;

        // Re-orthonormalise the search directions P against X (P comes out of
        // the Rayleigh-Ritz reconstruction non-orthonormal).
        let mut bp = sonium_solvers_core::dense::DenseMatrix::zeros(n, 0);
        if p.ncols() > 0 {
            let bp_raw = try_apply_b(b, &p)
                .map_err(|error| lobpcg_validation_error("mass operator", error))?;
            let (pn, bpn) = orthonormalize_block_b(&p, &bp_raw, Some((&x, &bx)));
            ensure_finite_dense("LOBPCG search directions", &pn)?;
            ensure_finite_dense("LOBPCG mass search directions", &bpn)?;
            p = pn;
            bp = bpn;
        }

        // Orthogonalise W against [locked, X, P] and orthonormalise it.
        let q = hstack(&[&locked_q, &x, &p]);
        let bq = hstack(&[&locked_bq, &bx, &bp]);
        let (w, bw) = orthonormalize_block_b(&w0, &bw0, Some((&q, &bq)));
        ensure_finite_dense("LOBPCG preconditioned directions", &w)?;
        ensure_finite_dense("LOBPCG mass preconditioned directions", &bw)?;
        if w.ncols() == 0 {
            // The preconditioned residual lies entirely in the current
            // subspace: the Ritz pairs cannot improve any further.
            return Err(SolverError::Breakdown {
                operation: "LOBPCG iteration",
                reason: "preconditioned residual produced no independent search direction".into(),
            });
        }

        // Operator products for the new directions only; A*X and B*X are
        // carried through the recurrence below.
        let aw =
            try_apply_columns(a, &w).map_err(|error| lobpcg_validation_error("operator", error))?;
        let ap =
            try_apply_columns(a, &p).map_err(|error| lobpcg_validation_error("operator", error))?;

        // Rayleigh-Ritz on S = [X, W, P] (or [X, W] on the first sweep).
        let s = hstack(&[&x, &w, &p]);
        let s_a = hstack(&[&ax, &aw, &ap]);
        let s_b = hstack(&[&bx, &bw, &bp]);
        let m = s.ncols();
        let gram_a = gram_symmetric_atb(&s, &s_a);
        let gram_b = gram_symmetric_atb(&s, &s_b);
        ensure_finite_dense("LOBPCG projected operator", &gram_a)?;
        ensure_finite_dense("LOBPCG projected mass operator", &gram_b)?;
        let (evals, evecs) = generalized_sym_eigen(&gram_a, &gram_b, m)?;
        // Rank-deficient projected directions are represented internally by
        // +infinity so they sort last. NaN is never a valid sentinel.
        ensure_non_nan_values("LOBPCG projected eigenvalues", &evals)?;
        ensure_finite_dense("LOBPCG projected eigenvectors", &evecs)?;

        let mut idx: Vec<usize> = (0..m).collect();
        idx.sort_by(|&lhs, &rhs| evals[lhs].total_cmp(&evals[rhs]));

        // Ritz coefficients for the ka smallest pairs, as one block of
        // columns so the reconstruction below is a single gemm per matrix.
        let ka = x.ncols();
        let mut z = sonium_solvers_core::dense::DenseMatrix::zeros(m, ka);
        for (jj, &col) in idx.iter().take(ka).enumerate() {
            eigenvalues[jj] = evals[col];
            for i in 0..m {
                z[[i, jj]] = evecs[[i, col]];
            }
        }
        ensure_finite_values("LOBPCG selected Ritz values", &eigenvalues)?;
        ensure_finite_dense("LOBPCG Ritz coefficients", &z)?;
        // Search directions keep only the non-X part of each Ritz vector.
        // Using X_new - X_old here would discard the W/P coefficients
        // selected by Rayleigh-Ritz and degenerate to a poorly conditioned
        // steepest-descent recurrence on hard pencils.
        let mut z_dir = z.clone();
        for jj in 0..ka {
            for i in 0..ka {
                z_dir[[i, jj]] = 0.0;
            }
        }

        x = s.matmul(&z);
        ax = s_a.matmul(&z);
        bx = s_b.matmul(&z);
        p = s.matmul(&z_dir);
        ensure_finite_dense("LOBPCG Ritz vectors", &x)?;
        ensure_finite_dense("LOBPCG Ritz operator products", &ax)?;
        ensure_finite_dense("LOBPCG Ritz mass products", &bx)?;
        ensure_finite_dense("LOBPCG updated search directions", &p)?;
    }

    // Merge locked and still-active pairs and return the k smallest
    // (oversampled padding columns are dropped here).
    let mut all_vals = locked_vals;
    all_vals.extend_from_slice(&eigenvalues);
    let mut all_res = locked_res;
    all_res.extend_from_slice(&residual_norms);
    let mut all_vecs: Vec<Vec<f64>> = (0..locked_q.ncols())
        .map(|j| locked_q.column_to_vec(j))
        .collect();
    all_vecs.extend((0..x.ncols()).map(|j| x.column_to_vec(j)));

    let mut order: Vec<usize> = (0..all_vals.len()).collect();
    ensure_non_nan_values("LOBPCG final eigenvalues", &all_vals)?;
    order.sort_by(|&lhs, &rhs| all_vals[lhs].total_cmp(&all_vals[rhs]));
    order.truncate(k);

    if order.len() != k {
        return Err(SolverError::Breakdown {
            operation: "LOBPCG finalization",
            reason: format!("computed {} of {k} requested eigenpairs", order.len()),
        });
    }

    let selected_eigenvalues: Vec<f64> = order.iter().map(|&i| all_vals[i]).collect();
    let selected_residual_norms: Vec<f64> = order.iter().map(|&i| all_res[i]).collect();
    let selected_eigenvectors =
        columns_to_matrix(n, order.iter().map(|&i| all_vecs[i].clone()).collect());
    ensure_finite_values("LOBPCG final eigenvalues", &selected_eigenvalues)?;
    ensure_finite_values("LOBPCG final residual norms", &selected_residual_norms)?;
    ensure_finite_dense("LOBPCG final eigenvectors", &selected_eigenvectors)?;

    Ok(LobpcgResult {
        eigenvalues: selected_eigenvalues,
        eigenvectors: selected_eigenvectors,
        iterations: iteration,
        residual_norms: selected_residual_norms,
        converged,
    })
}

// =====================================================================
// Helper functions
// =====================================================================

/// Apply operator to a block of columns.
///
/// Routes through the [`LinearOperator::try_apply_block`] multi-vector
/// contract: operators with a native block kernel (sparse SpMM, dense GEMM,
/// backend buffers) cover all `k` right-hand sides in one matrix walk
/// instead of `k` gather/scatter + matvec rounds. Operators without an
/// override fall back to the trait's per-column default.
fn try_apply_columns<A: LinearOperator<f64>>(
    op: &A,
    x: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> Result<sonium_solvers_core::dense::DenseMatrix<f64>, SolverError> {
    let mut result = sonium_solvers_core::dense::DenseMatrix::zeros(op.num_rows(), x.ncols());
    op.try_apply_block(x, &mut result)?;
    Ok(result)
}

/// Apply the mass operator `B`, or the identity for standard problems.
fn try_apply_b<B: LinearOperator<f64>>(
    b: Option<&B>,
    x: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> Result<sonium_solvers_core::dense::DenseMatrix<f64>, SolverError> {
    match b {
        Some(b) => try_apply_columns(b, x),
        None => Ok(x.clone()),
    }
}

/// Apply preconditioner to a block of columns.
///
/// Routes through the [`Preconditioner::try_apply_block`] multi-vector
/// contract, mirroring [`try_apply_columns`]: one checked block call instead
/// of `k` per-column applies, so native block kernels apply when present.
fn try_apply_precond_columns<P: Preconditioner<f64>>(
    pc: &P,
    r: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> Result<sonium_solvers_core::dense::DenseMatrix<f64>, SolverError> {
    let (n, k) = (r.nrows(), r.ncols());
    let mut result = sonium_solvers_core::dense::DenseMatrix::zeros(n, k);
    pc.try_apply_block(r, &mut result)?;
    Ok(result)
}

/// Orthogonalise a block `v` against the B-orthonormal basis `q` and
/// B-orthonormalise it internally.
///
/// `bv` carries `B·v` (for the standard problem it is simply a copy of `v`,
/// i.e. the Euclidean inner product); `bq` likewise carries `B·q`. Both are
/// updated with the exact same operations as `v`, so the invariant `bv = B·v`
/// is preserved without extra operator applies.
///
/// Procedure:
/// 1. Two block-CGS passes against `q`: `H = Q^T (B V)`, `V -= Q H`. The Gram
///    block and the update are single gemms.
/// 2. CholQR2 inside the block: QR via the Cholesky factor of the Gram
///    matrix `G = Vᵀ (B V) = Rᵀ R`, `V ← V R⁻¹`, applied twice (the
///    SLEPc-style one-reorth refinement). Both Gram products and both
///    triangular updates are BLAS-3 gemms, and the refinement pass brings
///    B-orthonormality down to machine level for well-conditioned Grams.
/// 3. Fallback: when the Gram is numerically rank-deficient the Cholesky
///    pivot breaks down; the code then uses MGS with one reorthogonalisation
///    pass, dropping columns whose B-norm collapses (numerically dependent
///    on `q` or on earlier kept columns) so the projected Gram matrix stays
///    SPD.
///
/// Returns the kept `(v, bv)` pair.
fn orthonormalize_block_b(
    v: &sonium_solvers_core::dense::DenseMatrix<f64>,
    bv: &sonium_solvers_core::dense::DenseMatrix<f64>,
    q: Option<(
        &sonium_solvers_core::dense::DenseMatrix<f64>,
        &sonium_solvers_core::dense::DenseMatrix<f64>,
    )>,
) -> (
    sonium_solvers_core::dense::DenseMatrix<f64>,
    sonium_solvers_core::dense::DenseMatrix<f64>,
) {
    let n = v.nrows();
    let kc = v.ncols();
    if kc == 0 {
        return (v.clone(), bv.clone());
    }

    let mut v = v.clone();
    let mut bv = bv.clone();

    if let Some((q, bq)) = q.filter(|(q, _)| q.ncols() > 0) {
        for _ in 0..2 {
            let h = gram_atb(q, &bv); // kq × kc Gram block
            sub_assign(&mut v, &q.matmul(&h));
            sub_assign(&mut bv, &bq.matmul(&h));
        }
    }

    // CholQR2 fast path (well-conditioned Gram).
    if let Some((vq, bvq)) = cholqr2_b(&v, &bv) {
        return (vq, bvq);
    }

    // MGS fallback with column dropping for rank-deficient blocks.
    let cols: Vec<Vec<f64>> = (0..kc).map(|j| v.column_to_vec(j)).collect();
    let bcols: Vec<Vec<f64>> = (0..kc).map(|j| bv.column_to_vec(j)).collect();
    let mut kept_v: Vec<Vec<f64>> = Vec::with_capacity(kc);
    let mut kept_bv: Vec<Vec<f64>> = Vec::with_capacity(kc);
    for j in 0..kc {
        let mut cj = cols[j].clone();
        let mut bcj = bcols[j].clone();
        let norm0 = bdot(&cj, &bcj).sqrt();
        for _ in 0..2 {
            for (ci, bci) in kept_v.iter().zip(kept_bv.iter()) {
                // Projection coefficient: signed inner product (a negative
                // value is meaningful here, so no roundoff clamp).
                let h = bdot_signed(ci, &bcj);
                for row in 0..n {
                    cj[row] -= h * ci[row];
                    bcj[row] -= h * bci[row];
                }
            }
        }
        let norm = bdot(&cj, &bcj).sqrt();
        if norm > 1e-12 * norm0.max(1e-300) {
            for row in 0..n {
                cj[row] /= norm;
                bcj[row] /= norm;
            }
            kept_v.push(cj);
            kept_bv.push(bcj);
        }
    }

    (columns_to_matrix(n, kept_v), columns_to_matrix(n, kept_bv))
}

/// CholQR2: B-orthonormalise `v` via the Cholesky factor of the Gram matrix,
/// applied twice (one refinement pass, SLEPc-style).
///
/// Each pass forms `G = Vᵀ (B V)`, factors `G = Rᵀ R`, and applies
/// `V ← V R⁻¹` to both `v` and its carried image `bv = B·v`, preserving the
/// `bv = B·v` invariant. Returns `None` when a Cholesky pivot breaks down
/// (Gram numerically rank-deficient or too ill-conditioned for CholQR to be
/// reliable); the caller then falls back to MGS with column dropping.
fn cholqr2_b(
    v: &sonium_solvers_core::dense::DenseMatrix<f64>,
    bv: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> Option<(
    sonium_solvers_core::dense::DenseMatrix<f64>,
    sonium_solvers_core::dense::DenseMatrix<f64>,
)> {
    let mut v = v.clone();
    let mut bv = bv.clone();
    for _ in 0..2 {
        let gram = gram_symmetric_atb(&v, &bv);
        let r = cholesky_upper(&gram)?;
        let r_inv = invert_upper_triangular(&r);
        v = v.matmul(&r_inv);
        bv = bv.matmul(&r_inv);
    }
    Some((v, bv))
}

/// Cholesky factorisation of a small SPD matrix into the upper factor `R`
/// with `G = Rᵀ R`. Returns `None` on pivot breakdown (non-SPD or
/// numerically rank-deficient input).
fn cholesky_upper(
    g: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> Option<sonium_solvers_core::dense::DenseMatrix<f64>> {
    let k = g.nrows();
    let mut r = sonium_solvers_core::dense::DenseMatrix::zeros(k, k);
    for j in 0..k {
        for i in 0..j {
            let mut s = g[[i, j]];
            for l in 0..i {
                s -= r[[l, i]] * r[[l, j]];
            }
            r[[i, j]] = s / r[[i, i]];
        }
        let mut d = g[[j, j]];
        for l in 0..j {
            d -= r[[l, j]] * r[[l, j]];
        }
        // Pivot breakdown: the remaining column direction is (nearly)
        // dependent on the previous ones. A tiny positive pivot would make
        // R⁻¹ blow up and destroy orthonormality, so bail out to the MGS
        // fallback instead.
        if !d.is_finite() || d <= 1e-14 * g[[j, j]].abs().max(1e-300) {
            return None;
        }
        r[[j, j]] = d.sqrt();
    }
    Some(r)
}

/// Invert a small upper-triangular matrix by back substitution. The
/// diagonal must be nonzero (guaranteed by [`cholesky_upper`]).
fn invert_upper_triangular(
    r: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    let k = r.nrows();
    let mut inv = sonium_solvers_core::dense::DenseMatrix::zeros(k, k);
    for col in 0..k {
        for i in (0..=col).rev() {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for l in (i + 1)..=col {
                s -= r[[i, l]] * inv[[l, col]];
            }
            inv[[i, col]] = s / r[[i, i]];
        }
    }
    inv
}

/// B-inner product `<x, B y>` given `x` and the carried image `by = B y`,
/// clamped against small negative roundoff. Use for (squared) B-norms only;
/// projection coefficients need [`bdot_signed`].
fn bdot(x: &[f64], by: &[f64]) -> f64 {
    bdot_signed(x, by).max(0.0)
}

/// Signed B-inner product `<x, B y>` given `x` and `by = B y`. Negative
/// values are legitimate for cross terms, so no clamping is applied.
fn bdot_signed(x: &[f64], by: &[f64]) -> f64 {
    // AVX2+FMA dot on x86_64 via the core BLAS helper (scalar fallback
    // elsewhere). Same value up to summation order.
    sonium_solvers_core::blas_helpers::dot_f64(x, by)
}

/// `a -= b` elementwise (same shape).
fn sub_assign(
    a: &mut sonium_solvers_core::dense::DenseMatrix<f64>,
    b: &sonium_solvers_core::dense::DenseMatrix<f64>,
) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice().iter()) {
        *x -= *y;
    }
}

/// Build an n×k matrix from owned columns (empty slice → n×0 matrix).
fn columns_to_matrix(
    n: usize,
    cols: Vec<Vec<f64>>,
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    let mut m = sonium_solvers_core::dense::DenseMatrix::zeros(n, cols.len());
    for (j, c) in cols.iter().enumerate() {
        m.set_column(j, c);
    }
    m
}

/// Keep only the given columns of a matrix.
fn take_columns(
    m: &sonium_solvers_core::dense::DenseMatrix<f64>,
    keep: &[usize],
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    columns_to_matrix(
        m.nrows(),
        keep.iter().map(|&j| m.column_to_vec(j)).collect(),
    )
}

/// Horizontally concatenate matrices with the same row count.
///
/// Copies row segments with unit-stride `memcpy`s instead of strided
/// column round-trips (`column_to_vec` + `set_column` touch each element
/// twice with an `n`-element stride, which stalls on cache lines for tall
/// blocks). Bitwise-identical result.
fn hstack(
    mats: &[&sonium_solvers_core::dense::DenseMatrix<f64>],
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    let n = mats.first().map_or(0, |m| m.nrows());
    let total: usize = mats.iter().map(|m| m.ncols()).sum();
    let mut out = sonium_solvers_core::dense::DenseMatrix::zeros(n, total);
    if n == 0 || total == 0 {
        return out;
    }
    let dst = out.as_mut_slice();
    for i in 0..n {
        let mut o = i * total;
        for m in mats {
            let cols = m.ncols();
            dst[o..o + cols].copy_from_slice(m.row(i));
            o += cols;
        }
    }
    out
}

/// `AᵀB` Gram product through the tuned transposed GEMM
/// (`DenseMatrix::matmul_transa`), with no `n × m` transpose allocation —
/// unlike the `a.t().dot(&b)` pattern it replaces. Same result up to
/// floating-point summation order.
fn gram_atb(
    a: &sonium_solvers_core::dense::DenseMatrix<f64>,
    b: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    a.matmul_transa(b)
}

/// Symmetric Gram product: [`gram_atb`] followed by `(G + Gᵀ)/2`
/// averaging, done in place.
fn gram_symmetric_atb(
    a: &sonium_solvers_core::dense::DenseMatrix<f64>,
    b: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    let mut g = gram_atb(a, b);
    let n = g.nrows();
    debug_assert_eq!(n, g.ncols());
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (g[[i, j]] + g[[j, i]]);
            g[[i, j]] = v;
            g[[j, i]] = v;
        }
    }
    g
}

/// Symmetrize a small square matrix: `G ← (G + Gᵀ)/2`.
#[allow(dead_code)]
fn symmetrize(
    g: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> sonium_solvers_core::dense::DenseMatrix<f64> {
    let n = g.nrows();
    let mut out = g.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (g[[i, j]] + g[[j, i]]);
            out[[i, j]] = v;
            out[[j, i]] = v;
        }
    }
    out
}

/// Compute generalized Rayleigh quotients: lambda_j = x_j^T A x_j / x_j^T B x_j.
fn generalized_rayleigh_quotients(
    x: &sonium_solvers_core::dense::DenseMatrix<f64>,
    ax: &sonium_solvers_core::dense::DenseMatrix<f64>,
    bx: &sonium_solvers_core::dense::DenseMatrix<f64>,
) -> Vec<f64> {
    let k = x.ncols();
    let mut eigenvalues = vec![0.0; k];
    for (j, eigenvalue) in eigenvalues.iter_mut().enumerate() {
        let xax = x.column(j).dot(&ax.column(j));
        let xbx = x.column(j).dot(&bx.column(j));
        *eigenvalue = if xbx.abs() > 1e-30 { xax / xbx } else { 0.0 };
    }
    eigenvalues
}

/// Relative residual scaling for column `j`.
fn residual_scale(
    ax: &sonium_solvers_core::dense::DenseMatrix<f64>,
    bx: &sonium_solvers_core::dense::DenseMatrix<f64>,
    j: usize,
    eigenvalue: f64,
    generalized: bool,
) -> f64 {
    if generalized {
        (column_norm(ax, j) + eigenvalue.abs() * column_norm(bx, j)).max(1e-30)
    } else {
        // For a standard eigenproblem, ||A*x|| equals |lambda| only for an
        // exact Ritz pair.  Using |lambda| alone makes the relative residual
        // scale collapse near a zero eigenvalue while the iterate still has a
        // non-zero operator image, overstating the residual by many orders of
        // magnitude and potentially preventing convergence.
        column_norm(ax, j).max(eigenvalue.abs()).max(1e-30)
    }
}

/// Euclidean norm of column j.
fn column_norm(m: &sonium_solvers_core::dense::DenseMatrix<f64>, j: usize) -> f64 {
    m.column(j).dot(&m.column(j)).sqrt()
}

/// Solve the symmetric generalised eigenproblem A z = lambda B z.
///
/// Uses eigendecomposition of B to whiten A, then solves the resulting standard
/// eigenproblem. This handles cases where B deviates from the identity due to
/// floating-point loss of orthogonality in the block orthogonalisation.
///
/// Returns (eigenvalues, eigenvectors-as-columns), or
/// `Err(SolverError::ConvergenceError)` if the underlying QL iteration
/// exhausts its sweep limit.
fn generalized_sym_eigen(
    a: &sonium_solvers_core::dense::DenseMatrix<f64>,
    b: &sonium_solvers_core::dense::DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, sonium_solvers_core::dense::DenseMatrix<f64>), SolverError> {
    // B = V D V^T  (V orthonormal, D diagonal with non-negative entries)
    let (b_evals, b_evecs) = symmetric_eigen_ql(b, n)?;
    ensure_finite_values("LOBPCG projected mass eigenvalues", &b_evals)?;
    if b_evals.iter().any(|&value| value <= 1.0e-14) {
        return Err(SolverError::Breakdown {
            operation: "LOBPCG projected mass eigensolve",
            reason: "projected mass matrix is not positive definite".into(),
        });
    }

    // Whitened problem: Ã = D^{-1/2} V^T A V D^{-1/2},  solve Ã y = lambda y
    let av = a.matmul(&b_evecs);
    let vtav = gram_atb(&b_evecs, &av);
    let mut a_white = sonium_solvers_core::dense::DenseMatrix::zeros(n, n);
    for i in 0..n {
        let si = b_evals[i].max(0.0).sqrt();
        if si < 1e-14 {
            continue;
        }
        for j in 0..n {
            let sj = b_evals[j].max(0.0).sqrt();
            if sj > 1e-14 {
                a_white[[i, j]] = vtav[[i, j]] / (si * sj);
            }
        }
    }

    let (mut evals, y_vecs) = symmetric_eigen_ql(&a_white, n)?;

    // Transform back to original space: z = V D^{-1/2} y
    let mut evecs = sonium_solvers_core::dense::DenseMatrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let si = b_evals[i].max(0.0).sqrt();
            if si > 1e-14 {
                let scale = y_vecs[[i, j]] / si;
                for l in 0..n {
                    evecs[[l, j]] += b_evecs[[l, i]] * scale;
                }
            }
        }
    }

    // Rank-deficient projected bases produce null-space Ritz vectors. They are
    // not valid generalized eigenvectors, so keep them out of the "smallest"
    // selection without discarding legitimate non-null zero modes.
    let b_evecs_prod = b.dot(&evecs);
    for j in 0..n {
        let b_norm: f64 = (0..n)
            .map(|row| evecs[[row, j]] * b_evecs_prod[[row, j]])
            .sum();
        if b_norm.abs() < 1e-12 {
            evals[j] = f64::INFINITY;
        }
    }

    Ok((evals, evecs))
}

/// Symmetric eigendecomposition of a dense matrix.
///
/// Small problems go through Householder tridiagonalisation followed by the
/// implicit-shift symmetric QL algorithm with Wilkinson shift — the standard
/// LAPACK-style pipeline (`sytrd` + `steqr`). At or above
/// [`crate::divide_conquer::DC_THRESHOLD`] the divide-and-conquer route
/// (Cuppen, LAPACK `dstedc`) takes over: it wins by a growing margin there
/// while the scalar QL path stays faster below. Returns (eigenvalues,
/// eigenvectors-as-columns), unordered, or `Err(SolverError::ConvergenceError)`
/// if the underlying iteration exhausts its limit.
pub fn symmetric_eigen(
    a: &sonium_solvers_core::dense::DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, sonium_solvers_core::dense::DenseMatrix<f64>), SolverError> {
    if a.nrows() != n || a.ncols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: a.nrows().max(a.ncols()),
        });
    }
    ensure_finite_dense("symmetric eigensolver input", a)?;
    if n >= crate::divide_conquer::DC_THRESHOLD {
        let (evals, evecs) = crate::divide_conquer::symmetric_eigen_dc(a, n)?;
        ensure_finite_values("symmetric eigensolver eigenvalues", &evals)?;
        ensure_finite_dense("symmetric eigensolver eigenvectors", &evecs)?;
        return Ok((evals, evecs));
    }
    symmetric_eigen_ql(a, n)
}

/// Symmetric eigendecomposition via the Householder + implicit-shift QL
/// pipeline only, regardless of size.
///
/// Internal solvers (LOBPCG/FEAST Ritz and projected solves) use this
/// directly: their convergence is tuned against the QL path's exact
/// behaviour (value order, sub-ulp ties), and re-routing per-iteration inner
/// solves through divide-and-conquer would perturb it.
pub(crate) fn symmetric_eigen_ql(
    a: &sonium_solvers_core::dense::DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, sonium_solvers_core::dense::DenseMatrix<f64>), SolverError> {
    if a.nrows() != n || a.ncols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: a.nrows().max(a.ncols()),
        });
    }
    ensure_finite_dense("symmetric eigensolver input", a)?;
    let mut z = a.as_slice().to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    householder_tridiag(&mut z, n, &mut d, &mut e);
    implicit_shift_ql(&mut d, &mut e, n, &mut z, 100 * n + 100)?;
    ensure_finite_values("symmetric eigensolver eigenvalues", &d)?;
    ensure_finite_values("symmetric eigensolver eigenvectors", &z)?;
    Ok((
        d,
        sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(n, n, z),
    ))
}

/// Householder tridiagonalisation of a row-major symmetric matrix.
///
/// On entry `a` holds the symmetric matrix (only the lower triangle is
/// referenced); on exit `a` holds the accumulated orthogonal transformation
/// Q, `d` the diagonal and `e` the subdiagonal of the tridiagonal matrix
/// (with `e[0] = 0`). Shared with the divide-and-conquer path
/// ([`crate::divide_conquer`]).
pub(crate) fn householder_tridiag(a: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let mut scale = 0.0;
            for k in 0..=l {
                scale += a[i * n + k].abs();
            }
            if scale == 0.0 {
                e[i] = a[i * n + l];
            } else {
                for k in 0..=l {
                    a[i * n + k] /= scale;
                    h += a[i * n + k] * a[i * n + k];
                }
                let f = a[i * n + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[i * n + l] = f - g;
                let mut fsum = 0.0;
                for j in 0..=l {
                    a[j * n + i] = a[i * n + j] / h;
                    let mut gsum = 0.0;
                    for k in 0..=j {
                        gsum += a[j * n + k] * a[i * n + k];
                    }
                    for k in (j + 1)..=l {
                        gsum += a[k * n + j] * a[i * n + k];
                    }
                    e[j] = gsum / h;
                    fsum += e[j] * a[i * n + j];
                }
                let hh = fsum / (h + h);
                for j in 0..=l {
                    let f2 = a[i * n + j];
                    let g2 = e[j] - hh * f2;
                    e[j] = g2;
                    for k in 0..=j {
                        a[j * n + k] -= f2 * e[k] + g2 * a[i * n + k];
                    }
                }
            }
        } else {
            e[i] = a[i * n + l];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;

    // Accumulate Q in place.
    for i in 0..n {
        if i > 0 && d[i] != 0.0 {
            for j in 0..i {
                let mut g = 0.0;
                for k in 0..i {
                    g += a[i * n + k] * a[k * n + j];
                }
                for k in 0..i {
                    a[k * n + j] -= g * a[k * n + i];
                }
            }
        }
        d[i] = a[i * n + i];
        a[i * n + i] = 1.0;
        for j in 0..i {
            a[i * n + j] = 0.0;
            a[j * n + i] = 0.0;
        }
    }
}

/// Implicit-shift symmetric QL iteration with Wilkinson shift.
///
/// Reduces the tridiagonal matrix (`d`, `e`) to diagonal form, accumulating
/// the rotations into the row-major orthogonal matrix `z`. Eigenvalues are
/// returned in `d` (unordered); column `i` of `z` is the matching
/// eigenvector.
///
/// Panel width for [`householder_tridiag_blocked`]: balances GEMM
/// efficiency (tall-skinny kernels) against level-2 panel work O(n²·nb).
/// 64 was retried after the column update went AVX2 and still loses at
/// n ≤ 300 (wider mirrors + fatter Q-accumulation panels outweigh the
/// level-3 shift), so 32 stands.
pub(crate) const HH_PANEL: usize = 32;

/// Blocked Householder tridiagonalisation (LAPACK `sytrd` structure):
/// panel factorisation by the `dlatrd` recurrence (explicit per-panel
/// reflectors `V`, compact-WY `T`, auxiliary `W`), trailing `syr2k`
/// updates (`A −= V·Wᵀ + W·Vᵀ`) through the tuned GEMM, and blocked Q
/// accumulation. Same output convention as [`householder_tridiag`]:
/// `d` the diagonal, `e` the subdiagonal with `e[0] = 0`, `a` holding Q
/// on exit. (The reduced `T` itself legitimately differs from the
/// descending NR reduction in [`householder_tridiag`]; both satisfy
/// `Q·T·Qᵀ = A` with identical eigenvalues.)
pub(crate) fn householder_tridiag_blocked(a: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
    use sonium_solvers_core::dense::DenseMatrix;
    if n == 0 {
        return;
    }
    if n == 1 {
        d[0] = a[0];
        return;
    }
    e[0] = 0.0;
    // Panels cover columns 0..n-1 (the last column needs no reflector).
    // Store each panel's (V, T): V is n×kb with reflector j supported on
    // rows ≥ k+j+1 and an explicit 1.0 at row k+j+1... (leading one at the
    // first subdiagonal position); T is kb×kb upper triangular.
    struct Panel {
        k: usize,
        kb: usize,
        v: DenseMatrix<f64>,
        t: DenseMatrix<f64>,
    }
    // Profile-gated phase split for the vendor-parity work (same
    // SONIUM_EIGEN_PROFILE flag as the D&C dispatcher): panel level-2 work
    // vs trailing syr2k GEMMs vs blocked Q accumulation.
    let prof = std::env::var_os("SONIUM_EIGEN_PROFILE").is_some();
    let mut ms_panel = 0.0;
    let mut ms_trail = 0.0;
    let mut ms_q = 0.0;
    let mut panels: Vec<Panel> = Vec::new();
    // Trailing-update scratch: U output (m×m, m < n) and the Wt-transpose
    // pack (kb×m), reused across panels (each panel overwrites fully).
    let mut uscratch = vec![0.0f64; n * n];
    let mut wscratch = vec![0.0f64; HH_PANEL * n];
    // Panel V/W column-major mirrors, reused across panels: every panel
    // snapshots its kb columns for rows ≥ its start, and every mirror
    // read runs at rows ≥ the current column (colupd from c, the rest
    // from c+1) — i.e. strictly inside the freshly snapshotted range —
    // so stale rows from earlier panels are never read and no
    // per-panel clear is needed. Same values read: bitwise-identical.
    let mut vcm = vec![0.0f64; HH_PANEL * n];
    let mut wcm = vec![0.0f64; HH_PANEL * n];
    let mut k = 0;
    while k < n - 1 {
        let kb = HH_PANEL.min(n - 1 - k);
        let t_ph = std::time::Instant::now();
        let (v, t, wmat) = hh_panel(a, n, d, e, k, kb, &mut vcm[..kb * n], &mut wcm[..kb * n]);
        ms_panel += t_ph.elapsed().as_secs_f64() * 1000.0;
        // Trailing syr2k update of rows/cols ≥ k+kb (nothing to do for the
        // final panel).
        if k + kb < n {
            let t_ph = std::time::Instant::now();
            hh_trailing_update(a, n, k + kb, &v, &wmat, kb, &mut uscratch, &mut wscratch);
            ms_trail += t_ph.elapsed().as_secs_f64() * 1000.0;
        }
        panels.push(Panel { k, kb, v, t });
        k += kb;
    }
    d[n - 1] = a[(n - 1) * n + (n - 1)];
    // Blocked Q accumulation, LAPACK-dorgtr order: Q = H_0·...·H_{P-1}
    // via last-panel-first LEFT updates
    // Q[r0..,:] −= V2·T·V2ᵀ·Q[r0..,:] (~3x fewer flops than the forward
    // full-width form: 4(n−k)²kb per panel instead of 4n²kb). Rows above
    // r0 are exactly identity-transformed (V is explicitly zero there),
    // so skipping them is bitwise-neutral; trailing-row groupings drift
    // by ulps, covered by the reconstruction/agreement tests.
    // (A forward trailing-only form was tried: same time — tall-skinny
    // GEMM efficiency, not flops, binds there. Wider panels net wash.)
    let mut q = DenseMatrix::eye(n);
    let t_ph = std::time::Instant::now();
    // A1/A2 panel outputs reuse two scratch buffers (every panel
    // overwrites them fully: beta=0 GEMMs): 2 allocs total instead of 2
    // per panel. V2 and the Q trailing rows are sliced in place — both
    // are contiguous row ranges — so the per-panel packs are gone too.
    // All three updates go through the ComplexField kernels directly:
    // - A1 = V2ᵀ·Qr: the serial transpose kernel. Below the dense
    //   wrapper's parallel gate this is the same call it would make
    //   (bitwise identical); the few biggest panels give up the
    //   wrapper's M-split (~+60µs each, hot) in exchange for no packs
    //   (~-200µs each) — a net win in-situ.
    // - A2 = T·A1: inline triangular loop ([`triu_mul_dense`]) —
    //   the GEMM dispatch (env reads + packing) dominates the kernel
    //   at this ≤32-wide shape; the loop is ulp-level (pinned by
    //   `qaccum_triu_loop_matches_gemm`).
    // - Q −= V2·A2: fused subtract (alpha=-1, beta=1) straight into Q's
    //   trailing rows — no A3 output alloc, no separate subtract pass;
    //   bitwise identical to GEMM-then-subtract (see
    //   qaccum_fused_subtract_matches_unfused).
    use sonium_solvers_core::traits::ComplexField;
    let mut a1buf = vec![0.0f64; HH_PANEL * n];
    let mut a2buf = vec![0.0f64; HH_PANEL * n];
    for p in panels.iter().rev() {
        let r0 = p.k + 1;
        let nr = n - r0;
        let kb = p.kb;
        let v2s = &p.v.as_slice()[r0 * kb..(r0 + nr) * kb];
        let qrs = &q.as_slice()[r0 * n..(r0 + nr) * n];
        let a1 = &mut a1buf[..kb * n];
        f64::mat_mul_tn(kb, n, nr, v2s, qrs, a1);
        let a2 = &mut a2buf[..kb * n];
        triu_mul_dense(p.t.as_slice(), a1, a2, kb, n);
        // Q[r0.., :] −= V2·A2 in the GEMM epilogue (C is the nr×n row
        // range at ldc=n inside Q).
        let tgt = &mut q.as_mut_slice()[r0 * n..(r0 + nr) * n];
        f64::mat_mul_nn_sub(nr, n, kb, v2s, a2, tgt, n);
    }
    ms_q += t_ph.elapsed().as_secs_f64() * 1000.0;
    if prof {
        eprintln!(
            "[sonium-eigen-tridiag] n={n} panel={ms_panel:.3}ms trailing={ms_trail:.3}ms qaccum={ms_q:.3}ms"
        );
    }
    a.copy_from_slice(q.as_slice());
}

/// Upper-triangular times dense (`A2 = T·A1`) for the blocked Q
/// accumulation: `t` is kb×kb upper triangular (row-major), `a1`/`a2`
/// are kb×n (row-major).
///
/// An inline AXPY loop (t-outer, i-inner over the nonzero triangle)
/// over contiguous n-length rows — L2-resident, auto-vectorized —
/// replacing a full GEMM dispatch whose env reads and packing
/// dominate the kernel at this ≤32-wide shape (~10 dispatches saved
/// at n = 300). Summation order differs from the GEMM by ulps
/// (pinned by `qaccum_triu_loop_matches_gemm`); the Q it feeds is
/// governed by the reconstruction/agreement tests.
fn triu_mul_dense(t: &[f64], a1: &[f64], a2: &mut [f64], kb: usize, n: usize) {
    debug_assert_eq!(t.len(), kb * kb);
    debug_assert_eq!(a1.len(), kb * n);
    debug_assert_eq!(a2.len(), kb * n);
    a2.fill(0.0);
    for tt in 0..kb {
        let a1row = &a1[tt * n..(tt + 1) * n];
        for i in 0..=tt {
            let c = t[i * kb + tt];
            let row = &mut a2[i * n..(i + 1) * n];
            for j in 0..n {
                row[j] += c * a1row[j];
            }
        }
    }
}

/// Factorise one panel (global columns `k..k+kb`) of the trailing matrix in
/// `a` (full symmetric storage): the LAPACK `dlatrd` recurrence (lower
/// case), adapted to full storage by mirroring every column update into
/// the upper triangle. Returns explicit `(V, T, W)`: `V` is n×kb with
/// column j holding reflector j (leading 1.0 at row `k+j+1`, support
/// below), `T` kb×kb upper triangular with the compact-WY coefficients
/// (for blocked Q accumulation), and `W` n×kb the dlatrd auxiliary matrix
/// driving the trailing `syr2k` update `A ← A − V·Wᵀ − W·Vᵀ`.
/// Finalises `d[k..k+kb]` and `e[k+1..k+kb+1]`; on exit the panel block of
/// `a` holds the finished tridiagonal and every row `≥ k` of later columns
/// carries the panel's contribution (only the trailing `≥ k+kb` block is
/// deferred to [`hh_trailing_update`]).
fn hh_panel(
    a: &mut [f64],
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    k: usize,
    kb: usize,
    vcm: &mut [f64],
    wcm: &mut [f64],
) -> (
    sonium_solvers_core::dense::DenseMatrix<f64>,
    sonium_solvers_core::dense::DenseMatrix<f64>,
    sonium_solvers_core::dense::DenseMatrix<f64>,
) {
    use sonium_solvers_core::dense::DenseMatrix;
    debug_assert_eq!(vcm.len(), kb * n);
    debug_assert_eq!(wcm.len(), kb * n);
    let mut v = DenseMatrix::zeros(n, kb);
    let mut t = DenseMatrix::zeros(kb, kb);
    let mut wmat = DenseMatrix::zeros(n, kb);
    // Column-major mirrors of V and W (column j at `j*n`), filled as each
    // column generates: single-column reads below then run over contiguous
    // memory (row-major layout would stride by kb and stall the
    // vectorizer/ILP). Borrowed from the caller's per-solve scratch
    // (see the reuse-safety note there); this panel snapshots every
    // column it reads before reading it.
    // Profile-gated section split (same SONIUM_EIGEN_PROFILE flag):
    // column update vs reflector gen vs DSYMV vs corrections vs dlarft.
    let pprof = std::env::var_os("SONIUM_EIGEN_PROFILE").is_some();
    let mut s_colupd = 0.0;
    let mut s_refgen = 0.0;
    let mut s_dsymv = 0.0;
    let mut s_corr = 0.0;
    let mut s_larft = 0.0;
    // Per-column small-vector scratch (wrow/wrow2/correction dots, larft
    // weights): hoisted to panel scope — each column overwrites [..j]
    // fully, so one kb-length buffer each replaces ~3·kb allocs per
    // panel with identical values.
    let mut wrow_buf = vec![0.0f64; kb];
    let mut wrow2_buf = vec![0.0f64; kb];
    let mut larft_buf = vec![0.0f64; kb];
    // Hoisted AVX2 dispatch: the x86 kernels probe CPU features on
    // every call (~150k dot/mirror calls at n = 300); probing once per
    // panel and dispatching through these closures recovers that. When
    // AVX2+FMA is present every call takes the vector path regardless
    // of length (the x86 length gates are throughput heuristics, not
    // correctness: both paths are exact for all inputs, and the rare
    // short vector call costs single-digit ns extra); without it every
    // call stays scalar. Short inputs differ from the gated dispatch
    // by reassociation ulps — the panel's accepted convention.
    #[cfg(target_arch = "x86_64")]
    let avx2 = crate::x86::has_avx2_fma();
    let dot = |x: &[f64], y: &[f64]| -> f64 {
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            // SAFETY: AVX2+FMA presence was probed once above (CPU
            // features are process-invariant); both slices span the
            // same index range at every call site below (equal
            // lengths, unaligned loads throughout the kernel).
            return unsafe { crate::x86::dot_avx2(x, y) };
        }
        // No-AVX2 / non-x86 fallback delegates to the gated kernel
        // (also keeps it live: this panel is its only caller).
        crate::x86::dot(x, y)
    };
    let mirror4 = |mirror: &[f64], jj: usize, r0: usize, u: &[f64], ustride: usize| -> [f64; 4] {
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            // SAFETY: AVX2+FMA probed above; the bounds are exactly
            // the delegated kernel's contract (jj ≤ kb live columns
            // of stride n with rows r0..r0+4 resident; u holds jj
            // elements at stride ustride) — the same arguments the
            // delegated call below receives.
            return unsafe { crate::x86::mirror_gemv4_avx2(mirror, n, jj, r0, u, ustride) };
        }
        crate::x86::mirror_gemv4(mirror, n, jj, r0, u, ustride)
    };
    for (j, c) in (k..k + kb).enumerate() {
        let mut t_s = std::time::Instant::now();
        // dlatrd column update: A[c..n, c] −= V[c..n, ..j]·W[c, ..j]ᵀ
        //                                  −= W[c..n, ..j]·V[c, ..j]ᵀ,
        // with V/W rows read over full support (panel rows included).
        // Hoisted row-major slices (strides n / kb): plain indexing
        // auto-vectorizes, opaque `Index` calls do not.
        {
            // 4-row mirror GEMVs over V/W (see `x86::mirror_gemv4`): the two
            // reductions share one pass shape; the row remainder stays on
            // the original scalar loop. `j < kb` always, so the strided
            // column-`c` reads stay in bounds (see the kernel contract).
            let mut r = c;
            while r + 4 <= n {
                let s1 = mirror4(vcm, j, r, &wcm[c..], n);
                let s2 = mirror4(wcm, j, r, &vcm[c..], n);
                a[r * n + c] -= s1[0] + s2[0];
                a[(r + 1) * n + c] -= s1[1] + s2[1];
                a[(r + 2) * n + c] -= s1[2] + s2[2];
                a[(r + 3) * n + c] -= s1[3] + s2[3];
                r += 4;
            }
            while r < n {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                for t_ in 0..j {
                    s1 += vcm[t_ * n + r] * wcm[t_ * n + c];
                    s2 += wcm[t_ * n + r] * vcm[t_ * n + c];
                }
                a[r * n + c] -= s1 + s2;
                r += 1;
            }
        }
        // Mirror into the upper triangle (full symmetric storage: the
        // trailing GEMM and the symmetric matvec below read full rows).
        for r in c + 1..n {
            a[c * n + r] = a[r * n + c];
        }
        d[c] = a[c * n + c];
        s_colupd += t_s.elapsed().as_secs_f64() * 1000.0;
        t_s = std::time::Instant::now();
        // LAPACK dlarfg on x = column segment rows c+1..n-1 (nonempty: c ≤ n-2).
        debug_assert!(n - c > 1);
        let alpha = a[(c + 1) * n + c];
        let mut xnorm_sq = 0.0;
        for r in c + 2..n {
            let x = a[r * n + c];
            xnorm_sq += x * x;
        }
        let xnorm = xnorm_sq.sqrt();
        let (tau, beta);
        if xnorm == 0.0 {
            tau = 0.0;
            beta = alpha;
        } else {
            let norm = alpha.hypot(xnorm);
            beta = -norm.copysign(alpha);
            tau = (beta - alpha) / beta;
            let denom = alpha - beta;
            let vsm = v.as_mut_slice();
            vsm[(c + 1) * kb + j] = 1.0;
            // Scale fused with the eliminated-segment zeroing (keeps
            // storage tridiagonal; the subdiagonal slot takes beta
            // below): one pass over rows c+2..n, not two. Same values.
            for r in c + 2..n {
                vsm[r * kb + j] = a[r * n + c] / denom;
                a[r * n + c] = 0.0;
                a[c * n + r] = 0.0;
            }
        }
        if xnorm == 0.0 {
            // Identity reflector: still record the (unused) unit vector so
            // downstream shapes stay uniform.
            v.as_mut_slice()[(c + 1) * kb + j] = 1.0;
        }
        // Snapshot column j into the column-major mirror (rows below c+1
        // are zero in both layouts: support starts at row c+1).
        {
            let vs = v.as_slice();
            let col = &mut vcm[j * n..(j + 1) * n];
            for r in c + 1..n {
                col[r] = vs[r * kb + j];
            }
        }
        e[c + 1] = beta;
        a[(c + 1) * n + c] = beta;
        a[c * n + (c + 1)] = beta;
        s_refgen += t_s.elapsed().as_secs_f64() * 1000.0;
        t_s = std::time::Instant::now();
        // dlatrd W column: symmetric matvec over rows/cols ≥ c+1, the two
        // correction GEMVs, τ scaling, and the α rank-1 fix. Skipped for an
        // identity reflector (dlatrd's DSCAL/DAXPY then force W[.., j] = 0).
        if tau != 0.0 {
            // Contiguous column-j (and column-t_) streams via the mirror.
            let vcol = &vcm[j * n..(j + 1) * n];
            // NOTE: threading this row loop was tried (rayon collect +
            // strided scatter): it loses. The working set fits private
            // cache and the dots are streaming, so splitting rows across
            // threads only shatters L2 locality for no compute gain —
            // unlike the division-heavy secular roots, which do parallelize.
            {
                // NOTE: a true symmetric (triangle-only) dual-update DSYMV
                // was A/B-tested here (within-block pairs + fused
                // below-block rank-1/transpose pass, exact pair coverage
                // verified by review): it halves the A traffic but measured
                // within noise of these full-square dots (24.9 vs 24.1 ms
                // medians at n=300 under shared-machine load) — the working
                // set is L3-resident, so the traffic saving does not pay,
                // and the scalar fused pass cannot beat AVX2 streaming rate.
                // Kept the simpler full-square per-row dots, fused 4 rows
                // at a time (fewer dispatches, shared L1 vector).
                let wms = wmat.as_mut_slice();
                // Fused 4-row dots (one kernel call per 4 rows instead of
                // one `dot` per row: fewer dispatches, shared L1 vector,
                // next-group prefetch). Reassociation ulps vs the old
                // per-row order are panel-accepted; tail stays scalar.
                // NOTE (triangle attempt, 2026-09-28): a triangle-only
                // composition (row part + scalar diag + `mirror_gemv4`
                // column part) REGRESSED +0.75ms at n=300 (9.14 -> 9.89):
                // the unprefetched strided column stream is latency-bound
                // (removed; OpenBLAS `dsymv` proves 1.4x is achievable with
                // a prefetch-carrying kernel — future retry needs one).
                let k0 = c + 1;
                let vseg = &vcol[k0..n];
                let mut r4 = k0;
                while r4 + 4 <= n {
                    crate::x86::rowdots4(a, n, r4, k0, n - k0, vseg, wms, r4 * kb + j, kb);
                    r4 += 4;
                }
                for r in r4..n {
                    wms[r * kb + j] = dot(&a[r * n + k0..r * n + n], vseg);
                }
            }
            s_dsymv += t_s.elapsed().as_secs_f64() * 1000.0;
            t_s = std::time::Instant::now();
            // Fused 4-row dots over the mirror rows (one kernel call per
            // 4 rows; short-j tails stay scalar). Same sums up to the
            // accepted reassociation ulps.
            let vseg = &vcol[c + 1..n];
            let wrow = &mut wrow_buf[..j];
            let mut t4 = 0;
            while t4 + 4 <= j {
                crate::x86::rowdots4(wcm, n, t4, c + 1, n - c - 1, vseg, &mut *wrow, t4, 1);
                t4 += 4;
            }
            for t_ in t4..j {
                wrow[t_] = dot(&wcm[t_ * n + c + 1..t_ * n + n], vseg);
            }
            let wrow2 = &mut wrow2_buf[..j];
            t4 = 0;
            while t4 + 4 <= j {
                crate::x86::rowdots4(vcm, n, t4, c + 1, n - c - 1, vseg, &mut *wrow2, t4, 1);
                t4 += 4;
            }
            for t_ in t4..j {
                wrow2[t_] = dot(&vcm[t_ * n + c + 1..(t_ + 1) * n], vseg);
            }
            // Fused correction: both GEMV subtractions (V·wrow and W·wrow2)
            // share one 4-row mirror pass, with τ scaling and the α dot
            // folded into the same pass. W reads through its column-major
            // mirror — same values the retired row-major loop read from
            // `wmat` (columns t_ < j are snapshotted for every row read
            // here). This retires the scalar (n−c)×j W-correction GEMV
            // (~0.2 ms at n = 300) and two extra passes over column j.
            // Values differ from the retired two-pass form by
            // mirror-kernel reassociation ulps — the accepted convention
            // for every AVX2 consumer in this panel (backward stability
            // unchanged; the reconstruction/agreement tests and the
            // orthogonality gate govern).
            // NOTE: dot accumulates here over the just-computed column —
            // the mirror snapshot below happens only after the α fix.
            let mut dotalpha = 0.0;
            {
                let wms = wmat.as_mut_slice();
                let mut r = c + 1;
                while r + 4 <= n {
                    let s1 = mirror4(vcm, j, r, wrow, 1);
                    let s2 = mirror4(wcm, j, r, wrow2, 1);
                    // Ascending within the block: same dot order as the
                    // scalar loop it replaces.
                    let e0 = &mut wms[r * kb + j];
                    *e0 = (*e0 - s1[0] - s2[0]) * tau;
                    dotalpha += *e0 * vcol[r];
                    let e1 = &mut wms[(r + 1) * kb + j];
                    *e1 = (*e1 - s1[1] - s2[1]) * tau;
                    dotalpha += *e1 * vcol[r + 1];
                    let e2 = &mut wms[(r + 2) * kb + j];
                    *e2 = (*e2 - s1[2] - s2[2]) * tau;
                    dotalpha += *e2 * vcol[r + 2];
                    let e3 = &mut wms[(r + 3) * kb + j];
                    *e3 = (*e3 - s1[3] - s2[3]) * tau;
                    dotalpha += *e3 * vcol[r + 3];
                    r += 4;
                }
                while r < n {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    for t_ in 0..j {
                        s1 += vcm[t_ * n + r] * wrow[t_];
                        s2 += wcm[t_ * n + r] * wrow2[t_];
                    }
                    let e = &mut wms[r * kb + j];
                    *e = (*e - s1 - s2) * tau;
                    dotalpha += *e * vcol[r];
                    r += 1;
                }
            }
            let alpha = -0.5 * tau * dotalpha;
            // Unrolled DAXPY over the strided W column, fused with the W
            // mirror snapshot: the strided read feeds both the update and
            // the mirror write (one pass, not two). Same rows, same values.
            {
                let wms = wmat.as_mut_slice();
                let col = &mut wcm[j * n..(j + 1) * n];
                let mut r = c + 1;
                while r + 4 <= n {
                    wms[r * kb + j] += alpha * vcol[r];
                    col[r] = wms[r * kb + j];
                    wms[(r + 1) * kb + j] += alpha * vcol[r + 1];
                    col[r + 1] = wms[(r + 1) * kb + j];
                    wms[(r + 2) * kb + j] += alpha * vcol[r + 2];
                    col[r + 2] = wms[(r + 2) * kb + j];
                    wms[(r + 3) * kb + j] += alpha * vcol[r + 3];
                    col[r + 3] = wms[(r + 3) * kb + j];
                    r += 4;
                }
                while r < n {
                    wms[r * kb + j] += alpha * vcol[r];
                    col[r] = wms[r * kb + j];
                    r += 1;
                }
            }
        } else {
            // Identity reflector: W[.., j] stays zero, so the mirror
            // column must read zero too. With per-panel mirrors this
            // held by fresh zero-init; with reused mirrors the rows
            // future columns read must be cleared explicitly, else
            // stale values from earlier panels leak into later
            // columns' updates. Same zeros either way.
            wcm[j * n + c + 1..(j + 1) * n].fill(0.0);
        }
        s_corr += t_s.elapsed().as_secs_f64() * 1000.0;
        t_s = std::time::Instant::now();
        // dlarft (forward): T[j,j] = τ; T[i,j] = −τ·T[0..j,0..j]·w.
        t.as_mut_slice()[j * kb + j] = tau;
        if tau != 0.0 {
            let vcol = &vcm[j * n..(j + 1) * n];
            let w = &mut larft_buf[..j];
            // Dots run over rows c+1..n only: the current reflector
            // vanishes above row c+1 (leading 1.0 there), so the dropped
            // leading terms are exact ±0.0 — ulp-level regrouping of the
            // same sum, ~half the larft flops. Fused 4-row form (tails
            // scalar), same convention as the correction dots above.
            let vseg = &vcol[c + 1..n];
            let mut i4 = 0;
            while i4 + 4 <= j {
                crate::x86::rowdots4(vcm, n, i4, c + 1, n - c - 1, vseg, &mut *w, i4, 1);
                i4 += 4;
            }
            for i in i4..j {
                w[i] = dot(&vcm[i * n + c + 1..(i + 1) * n], vseg);
            }
            let tsm = t.as_mut_slice();
            for i in 0..j {
                let mut s = 0.0;
                for tt in i..j {
                    s += tsm[i * kb + tt] * w[tt];
                }
                tsm[i * kb + j] = -tau * s;
            }
        }
        s_larft += t_s.elapsed().as_secs_f64() * 1000.0;
    }
    if pprof {
        eprintln!(
            "[sonium-eigen-panel] k={k} kb={kb} colupd={s_colupd:.3}ms refgen={s_refgen:.3}ms dsymv={s_dsymv:.3}ms corr={s_corr:.3}ms larft={s_larft:.3}ms"
        );
    }
    (v, t, wmat)
}

/// Trailing `syr2k` update `A ← A − V·Wᵀ − W·Vᵀ` on rows/cols `≥ start`
/// (LAPACK `dsytrd`'s `DSYR2K` call): applies the whole panel block
/// reflector to the trailing submatrix through a single GEMM
/// (`U = V·Wᵀ`, then `A −= U + Uᵀ`). The panel block itself and the
/// panel-rows × trailing-cols rectangle are already final — the panel
/// factorisation ([`hh_panel`]) applied them column by column — so only
/// the trailing `≥ start` block is touched here. `v`/`w` are the panel
/// n×kb reflector/auxiliary matrices, `kb` the panel width.
fn hh_trailing_update(
    a: &mut [f64],
    n: usize,
    start: usize,
    v: &sonium_solvers_core::dense::DenseMatrix<f64>,
    w: &sonium_solvers_core::dense::DenseMatrix<f64>,
    kb: usize,
    uscratch: &mut [f64],
    wscratch: &mut [f64],
) {
    use sonium_solvers_core::traits::ComplexField;
    let m = n - start;
    // Vt rows are a contiguous row range — slice, no pack. Wt must feed
    // a trans-B GEMM the dense wrapper does not expose, so pack its
    // transpose with flat offset writes (the from_fn pair + transposed()
    // did three bounds-checked passes here).
    let vt = &v.as_slice()[start * kb..(start + m) * kb];
    let ws = w.as_slice();
    // 8×8-blocked transpose pack: the naive i-outer form strides the
    // output by m (one cache line per element); blocking keeps both
    // sides L1-resident. Same values, bitwise-identical GEMM input.
    let wtt = &mut wscratch[..kb * m];
    const TB: usize = 8;
    let mut i0 = 0;
    while i0 < m {
        let ie = (i0 + TB).min(m);
        let mut j0 = 0;
        while j0 < kb {
            let je = (j0 + TB).min(kb);
            for i in i0..ie {
                let row = (start + i) * kb;
                for j in j0..je {
                    wtt[j * m + i] = ws[row + j];
                }
            }
            j0 = je;
        }
        i0 = ie;
    }
    // U = Vt·Wtᵀ (m×m) through the serial kernel into reused scratch;
    // A −= U + Uᵀ keeps exact symmetry. Only the largest panel clears
    // the dense wrapper's parallel gate; the direct call matches its
    // serial path element-for-element (bitwise identical) and wins
    // in-situ there too by skipping the M-split's per-panel B re-packs.
    let u = &mut uscratch[..m * m];
    f64::mat_mul_nn(m, m, kb, vt, wtt, u);
    // Symmetric scatter fused with the upper-triangle mirror: each (r,c)
    // writes both triangles while the U pair is hot (the separate mirror
    // pass re-streamed the whole trailing block for a copy). Same values
    // the old scatter-then-mirror wrote, bitwise.
    for c in 0..m {
        for r in c..m {
            let v = u[r * m + c] + u[c * m + r];
            a[(r + start) * n + (c + start)] -= v;
            if r != c {
                a[(c + start) * n + (r + start)] -= v;
            }
        }
    }
}

/// Implicit-shift symmetric QL iteration with Wilkinson shift.
///
/// Reduces the tridiagonal matrix (`d`, `e`) to diagonal form, accumulating
/// the rotations into the row-major orthogonal matrix `z`. Eigenvalues are
/// returned in `d` (unordered); column `i` of `z` is the matching
/// eigenvector.
///
/// `max_sweeps` caps the number of QL sweeps per eigenvalue; the production
/// path uses `100 * n + 100`. If the cap is exceeded the routine returns
/// `Err(SolverError::ConvergenceError)` with the largest remaining
/// subdiagonal magnitude as the residual, instead of silently returning a
/// partially reduced matrix. The limit is effectively never reached for the
/// small well-conditioned projected problems LOBPCG produces; it exists to
/// avoid an infinite loop on pathological input. Shared with the
/// divide-and-conquer path ([`crate::divide_conquer`]), which runs it
/// directly on small tridiagonal leaves.
pub(crate) fn implicit_shift_ql(
    d: &mut [f64],
    e: &mut [f64],
    n: usize,
    z: &mut [f64],
    max_sweeps: usize,
) -> Result<(), SolverError> {
    if n == 0 {
        return Ok(());
    }
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    // Transposed eigenvector accumulator: QL rotations hit COLUMNS of Z
    // (stride-n pairs), so accumulating Zᵀ instead turns every rotation's
    // Z-update into two contiguous row AXPYs that auto-vectorize. Same
    // per-element operations in the same order → bitwise-identical;
    // transposed back into `z` on every exit (including the error path,
    // which preserves the partial-accumulation postcondition).
    let mut zt = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            zt[j * n + i] = z[i * n + j];
        }
    }
    let transpose_back = |zt: &[f64], z: &mut [f64]| {
        for i in 0..n {
            for j in 0..n {
                z[i * n + j] = zt[j * n + i];
            }
        }
    };

    for l in 0..n {
        let mut sweeps = 0;
        loop {
            // Look for a single small subdiagonal element to split the matrix.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            sweeps += 1;
            if sweeps > max_sweeps {
                // Report the largest unconverged coupling as the residual so
                // the caller can see how far from convergence the sweep died.
                let residual = e[l..n - 1].iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                transpose_back(&zt, z);
                return Err(SolverError::ConvergenceError {
                    status: SolverStatus::MaxIterationsReached,
                    iterations: sweeps,
                    residual,
                });
            }

            // Implicit QL step with Wilkinson shift.
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut restarted = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r2 = f.hypot(g);
                e[i + 1] = r2;
                if r2 == 0.0 {
                    // Recover from underflow: restart the sweep.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    restarted = true;
                    break;
                }
                s = f / r2;
                c = g / r2;
                g = d[i + 1] - p;
                let rr = (d[i] - g) * s + 2.0 * c * b;
                p = s * rr;
                d[i + 1] = g + p;
                g = c * rr - b;
                // Row pair of ZT == column pair of Z: contiguous AXPYs.
                let (lo, hi) = zt.split_at_mut((i + 1) * n);
                let (ra, rb) = (&mut lo[i * n..(i + 1) * n], &mut hi[..n]);
                for r in 0..n {
                    let a = ra[r];
                    let b = rb[r];
                    ra[r] = c * a - s * b;
                    rb[r] = s * a + c * b;
                }
            }
            if restarted {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    transpose_back(&zt, z);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn diagonal(values_in: &[f64]) -> CsrMatrix<f64> {
        let n = values_in.len();
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        row_ptrs.push(0);

        for (i, value) in values_in.iter().enumerate() {
            col_indices.push(i);
            values.push(*value);
            row_ptrs.push(values.len());
        }

        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    /// LinearOperator wrapper that counts matrix-vector products, used to
    /// verify that soft locking excludes converged vectors from SpMV work.
    struct CountingOperator {
        inner: CsrMatrix<f64>,
        applies: AtomicUsize,
    }

    impl CountingOperator {
        fn new(inner: CsrMatrix<f64>) -> Self {
            Self {
                inner,
                applies: AtomicUsize::new(0),
            }
        }

        fn apply_count(&self) -> usize {
            self.applies.load(Ordering::Relaxed)
        }
    }

    impl LinearOperator<f64> for CountingOperator {
        fn num_rows(&self) -> usize {
            self.inner.num_rows()
        }

        fn num_cols(&self) -> usize {
            self.inner.num_cols()
        }

        fn apply(&self, x: &[f64]) -> Vec<f64> {
            self.applies.fetch_add(1, Ordering::Relaxed);
            self.inner.apply(x)
        }

        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            self.applies.fetch_add(1, Ordering::Relaxed);
            self.inner.apply_into(x, y);
        }

        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            self.inner.apply_transpose_into(x, y);
        }
    }

    #[test]
    fn qaccum_fused_subtract_matches_unfused() {
        // The blocked Q accumulation fuses V2·A2 into Q via mat_mul_nn_sub
        // (alpha=-1, beta=1); it must equal GEMM-then-subtract bitwise at
        // the accumulation shapes (first/last panel geometries at n=300).
        use sonium_solvers_core::dense::DenseMatrix;
        use sonium_solvers_core::traits::ComplexField;
        fn gen_vec(seed: &mut u64, len: usize) -> Vec<f64> {
            (0..len)
                .map(|_| {
                    *seed = seed
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(0xBF58476D1CE4E5B9);
                    ((*seed >> 11) as f64) / (u64::MAX as f64) * 2.0 - 1.0
                })
                .collect()
        }
        let mut seed = 0x51ab_c07d_u64;
        for &(nr, kb, n) in &[(299usize, 32usize, 300usize), (11, 11, 300), (75, 32, 150)] {
            let v2 = gen_vec(&mut seed, nr * kb);
            let a2 = gen_vec(&mut seed, kb * n);
            let q0 = gen_vec(&mut seed, nr * n);
            // Unfused reference: owned GEMM then subtract.
            let vm = DenseMatrix::from_row_major_vec(nr, kb, v2.clone());
            let am = DenseMatrix::from_row_major_vec(kb, n, a2.clone());
            let prod = vm.matmul(&am);
            let mut want = q0.clone();
            for (w, &p) in want.iter_mut().zip(prod.as_slice().iter()) {
                *w -= p;
            }
            // Fused: straight into the target (strided form, ldc=n, as the
            // panel loop passes Q's trailing rows).
            let mut got = q0;
            f64::mat_mul_nn_sub(nr, n, kb, &v2, &a2, &mut got, n);
            assert_eq!(got.len(), want.len());
            for (t, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "nr={nr} kb={kb} n={n} elem {t}: {g} vs {w}"
                );
            }
        }
    }

    #[test]
    fn qaccum_triu_loop_matches_gemm() {
        // The blocked Q accumulation forms A2 = T·A1 (T upper
        // triangular) with an inline loop instead of a GEMM dispatch;
        // it must equal the kernel up to summation-order ulps at
        // accumulation shapes (full + final-panel geometries). A wrong
        // triangle traversal would show up at O(1), far above the bar.
        use super::triu_mul_dense;
        use sonium_solvers_core::traits::ComplexField;
        let mut seed = 0x7ac0_5517_u64;
        let mut genv = |len: usize| -> Vec<f64> {
            (0..len)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(0xBF58476D1CE4E5B9);
                    ((seed >> 11) as f64) / (u64::MAX as f64) * 2.0 - 1.0
                })
                .collect()
        };
        for &(kb, n) in &[(32usize, 300usize), (11, 300), (32, 150), (7, 200)] {
            let mut t = genv(kb * kb);
            for i in 0..kb {
                for tt in 0..i {
                    t[i * kb + tt] = 0.0;
                }
            }
            let a1 = genv(kb * n);
            let mut want = vec![0.0; kb * n];
            f64::mat_mul_nn(kb, n, kb, &t, &a1, &mut want);
            let mut got = vec![0.0; kb * n];
            triu_mul_dense(&t, &a1, &mut got, kb, n);
            let scale = want
                .iter()
                .map(|v| v.abs())
                .fold(0.0f64, f64::max)
                .max(1e-300);
            let mut worst = 0.0f64;
            for (g, w) in got.iter().zip(want.iter()) {
                worst = worst.max((g - w).abs());
            }
            eprintln!("DIAG triu-loop kb={kb} n={n} worst={worst:.3e} scale={scale:.3e}");
            assert!(
                worst <= 1e-12 * scale,
                "kb={kb} n={n}: worst {worst:.3e} scale {scale:.3e}"
            );
        }
    }

    /// Spectrum with well-separated small eigenvalues (which converge quickly
    /// and get locked) followed by a tight cluster (which dominates the
    /// iteration count).
    fn separated_plus_cluster(n: usize) -> Vec<f64> {
        let mut diag = vec![0.05, 0.4, 1.5];
        for i in 3..n {
            diag.push(8.0 + 0.02 * (i - 3) as f64);
        }
        diag
    }

    #[test]
    fn test_lobpcg_smallest_eigenvalue_1d_laplacian() {
        // 1D Laplacian eigenvalues: lambda_k = 2 - 2*cos(k*pi/(n+1))
        let n = 50;
        let a = laplacian_1d(n);

        let config = LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 200,
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = lobpcg(&a, &config).unwrap();

        // Analytical eigenvalues
        let lambda_1 = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        let lambda_2 = 2.0 - 2.0 * (2.0 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
        let lambda_3 = 2.0 - 2.0 * (3.0 * std::f64::consts::PI / (n as f64 + 1.0)).cos();

        assert!(
            result.converged,
            "LOBPCG should converge (iters: {}, res: {:?})",
            result.iterations, result.residual_norms
        );

        let tol = 1e-4;
        assert!(
            (result.eigenvalues[0] - lambda_1).abs() < tol,
            "lambda_1: got {}, expected {}, diff {}",
            result.eigenvalues[0],
            lambda_1,
            (result.eigenvalues[0] - lambda_1).abs()
        );
        assert!(
            (result.eigenvalues[1] - lambda_2).abs() < tol,
            "lambda_2: got {}, expected {}",
            result.eigenvalues[1],
            lambda_2
        );
        assert!(
            (result.eigenvalues[2] - lambda_3).abs() < tol,
            "lambda_3: got {}, expected {}",
            result.eigenvalues[2],
            lambda_3
        );
    }

    #[test]
    fn test_lobpcg_eigenvectors_orthogonal() {
        let n = 30;
        let a = laplacian_1d(n);

        let config = LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 200,
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = lobpcg(&a, &config).unwrap();
        let k = result.eigenvalues.len();

        // Check orthogonality: v_i^T v_j ≈ delta_ij
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = result
                    .eigenvectors
                    .column(i)
                    .dot(&result.eigenvectors.column(j));
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "Eigenvectors not orthonormal: v{i}^T v{j} = {dot:.6} (expected {expected})"
                );
            }
        }
    }

    #[test]
    fn test_lobpcg_with_preconditioner() {
        let n = 50;
        let a = laplacian_1d(n);
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let config = LobpcgConfig {
            num_eigenvalues: 2,
            max_iterations: 200,
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = lobpcg_preconditioned(&a, &precond, &config).unwrap();

        let lambda_1 = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        assert!(
            (result.eigenvalues[0] - lambda_1).abs() < 1e-4,
            "Preconditioned LOBPCG lambda_1: got {}, expected {}",
            result.eigenvalues[0],
            lambda_1
        );
    }

    #[test]
    fn test_lobpcg_generalized_diagonal_mass() {
        let n = 8;
        let mass_diag: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        let expected: Vec<f64> = (0..n).map(|i| 2.0 + 0.75 * i as f64).collect();
        let stiffness_diag: Vec<f64> = mass_diag
            .iter()
            .zip(expected.iter())
            .map(|(b, lambda)| b * lambda)
            .collect();

        let a = diagonal(&stiffness_diag);
        let b = diagonal(&mass_diag);
        let config = LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 100,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = lobpcg_generalized(&a, &b, &config).unwrap();

        assert!(
            result.converged,
            "generalized LOBPCG should converge (iters: {}, res: {:?})",
            result.iterations, result.residual_norms
        );
        for (j, expected_lambda) in expected.iter().take(3).enumerate() {
            assert!(
                (result.eigenvalues[j] - expected_lambda).abs() < 1e-6,
                "lambda_{j}: got {}, expected {}",
                result.eigenvalues[j],
                expected_lambda
            );
        }

        let bx = try_apply_columns(&b, &result.eigenvectors).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let dot = result.eigenvectors.column(i).dot(&bx.column(j));
                let expected_dot = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected_dot).abs() < 1e-5,
                    "eigenvectors not B-orthonormal: v{i}^T B v{j} = {dot:.6}"
                );
            }
        }
    }

    #[test]
    fn test_lobpcg_single_eigenvalue() {
        let n = 20;
        let a = laplacian_1d(n);

        let config = LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 100,
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = lobpcg(&a, &config).unwrap();
        assert_eq!(result.eigenvalues.len(), 1);

        let lambda_1 = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        assert!(
            (result.eigenvalues[0] - lambda_1).abs() < 1e-4,
            "Single eigenvalue: got {}, expected {}",
            result.eigenvalues[0],
            lambda_1
        );
    }

    #[test]
    fn standard_residual_scale_uses_operator_norm_near_zero_eigenvalue() {
        let ax = sonium_solvers_core::dense::DenseMatrix::from_fn(2, 1, |i, _| {
            if i == 0 { 1.0e-6 } else { 0.0 }
        });
        let bx = sonium_solvers_core::dense::DenseMatrix::zeros(2, 1);

        let scale = residual_scale(&ax, &bx, 0, 0.0, false);

        assert!((scale - 1.0e-6).abs() < 1.0e-18);
    }

    #[test]
    fn test_lobpcg_soft_locking_excludes_converged_from_spmv() {
        let n = 60;
        let diag = separated_plus_cluster(n);
        let a = CountingOperator::new(diagonal(&diag));

        let k = 4;
        let config = LobpcgConfig {
            num_eigenvalues: k,
            max_iterations: 500,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = lobpcg(&a, &config).unwrap();
        assert!(
            result.converged,
            "LOBPCG should converge (iters: {}, res: {:?})",
            result.iterations, result.residual_norms
        );
        for (j, expected) in diag.iter().take(k).enumerate() {
            assert!(
                (result.eigenvalues[j] - expected).abs() < 1e-6,
                "lambda_{j}: got {}, expected {}",
                result.eigenvalues[j],
                expected
            );
        }

        // Without locking, every Rayleigh-Ritz sweep applies the operator to
        // the full W and P blocks: k initial applies, k on the first sweep
        // (no P yet), 2k on every later sweep. Soft locking must do strictly
        // fewer even after accounting for one verification apply per lock.
        let total = a.apply_count();
        let sweeps = result.iterations.saturating_sub(1);
        let nolock_applies = if sweeps <= 1 {
            k + k * sweeps
        } else {
            k + k + 2 * k * (sweeps - 1)
        };
        assert!(
            total + k <= nolock_applies,
            "locked vectors should be excluded from SpMV work: {total} applies \
             (no-locking baseline {nolock_applies}, iters {})",
            result.iterations
        );
    }

    #[test]
    fn test_lobpcg_soft_locking_reduces_work_on_clustered_spectrum() {
        let n = 60;
        let diag = separated_plus_cluster(n);
        let k = 4;
        let config = LobpcgConfig {
            num_eigenvalues: k,
            max_iterations: 500,
            tolerance: 1e-8,
            ..Default::default()
        };

        let a_lock = CountingOperator::new(diagonal(&diag));
        let res_lock = lobpcg_core(
            &a_lock,
            None::<&CsrMatrix<f64>>,
            None::<&sonium_solvers_core::traits::IdentityPreconditioner>,
            &config,
            true,
        )
        .unwrap();

        let a_nolock = CountingOperator::new(diagonal(&diag));
        let res_nolock = lobpcg_core(
            &a_nolock,
            None::<&CsrMatrix<f64>>,
            None::<&sonium_solvers_core::traits::IdentityPreconditioner>,
            &config,
            false,
        )
        .unwrap();

        assert!(res_lock.converged && res_nolock.converged);
        for j in 0..k {
            assert!(
                (res_lock.eigenvalues[j] - res_nolock.eigenvalues[j]).abs() < 1e-6,
                "locking changed eigenpair {j}: {} vs {}",
                res_lock.eigenvalues[j],
                res_nolock.eigenvalues[j]
            );
        }

        let locked = a_lock.apply_count();
        let unlocked = a_nolock.apply_count();
        assert!(
            locked < unlocked,
            "locking should reduce SpMV work: {locked} vs {unlocked} applies"
        );
        // On this clustered spectrum the three separated pairs lock long
        // before the clustered one converges, so most sweeps run at a
        // quarter of the block size.
        assert!(
            locked * 2 <= unlocked,
            "locking should at least halve SpMV work on a clustered spectrum: \
             {locked} vs {unlocked} applies (iters {} vs {})",
            res_lock.iterations,
            res_nolock.iterations
        );
    }

    #[test]
    fn test_lobpcg_generalized_b_orthonormality_tight() {
        let n = 40;
        let mass_diag: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        // Well-separated quadratic spectrum for fast convergence.
        let expected: Vec<f64> = (0..n)
            .map(|i| 0.2 * (i + 1) as f64 * (i + 1) as f64)
            .collect();
        let stiffness_diag: Vec<f64> = mass_diag
            .iter()
            .zip(expected.iter())
            .map(|(b, lambda)| b * lambda)
            .collect();

        let a = diagonal(&stiffness_diag);
        let b = diagonal(&mass_diag);
        let k = 4;
        let config = LobpcgConfig {
            num_eigenvalues: k,
            max_iterations: 300,
            tolerance: 1e-9,
            ..Default::default()
        };

        let result = lobpcg_generalized(&a, &b, &config).unwrap();
        assert!(
            result.converged,
            "generalized LOBPCG should converge (iters: {}, res: {:?})",
            result.iterations, result.residual_norms
        );
        for (j, expected_lambda) in expected.iter().take(k).enumerate() {
            assert!(
                (result.eigenvalues[j] - expected_lambda).abs() < 1e-6,
                "lambda_{j}: got {}, expected {}",
                result.eigenvalues[j],
                expected_lambda
            );
        }

        // ||X^T B X - I|| < 1e-10 (elementwise max)
        let bx = try_apply_columns(&b, &result.eigenvectors).unwrap();
        let gram = result.eigenvectors.t().dot(&bx);
        for i in 0..k {
            for j in 0..k {
                let expected_dot = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - expected_dot).abs() < 1e-10,
                    "eigenvectors not B-orthonormal: v{i}^T B v{j} = {:.3e}",
                    gram[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_implicit_shift_ql_reports_nonconvergence() {
        // Non-trivial tridiagonal problem: with a zero sweep budget the QL
        // iteration cannot converge, and it must report that loudly instead
        // of returning a partially reduced matrix.
        let n = 8;
        let mut d: Vec<f64> = (0..n).map(|i| 2.0 + 0.5 * i as f64).collect();
        let mut e: Vec<f64> = (0..n).map(|i| 1.0 - 0.1 * i as f64).collect();
        let mut z = vec![0.0; n * n];
        for i in 0..n {
            z[i * n + i] = 1.0;
        }

        let result = implicit_shift_ql(&mut d, &mut e, n, &mut z, 0);
        match result {
            Err(SolverError::ConvergenceError {
                status,
                iterations,
                residual,
            }) => {
                assert_eq!(status, SolverStatus::MaxIterationsReached);
                assert_eq!(iterations, 1);
                assert!(
                    residual > 0.0,
                    "residual should carry the unconverged coupling"
                );
            }
            other => panic!("expected ConvergenceError, got {other:?}"),
        }
    }

    #[test]
    fn test_implicit_shift_ql_converges_with_default_budget() {
        // The production sweep budget `100 * n + 100` must still converge on
        // the same problem and return the correct eigenvalues (unordered).
        let n = 8;
        let mut d: Vec<f64> = (0..n).map(|i| 2.0 + 0.5 * i as f64).collect();
        let mut e: Vec<f64> = (0..n).map(|i| 1.0 - 0.1 * i as f64).collect();
        let mut z = vec![0.0; n * n];
        for i in 0..n {
            z[i * n + i] = 1.0;
        }

        implicit_shift_ql(&mut d, &mut e, n, &mut z, 100 * n + 100).unwrap();

        // Residual check: T z_j ≈ d_j z_j for every column of z, where T is
        // the original tridiagonal matrix.
        let t_diag: Vec<f64> = (0..n).map(|i| 2.0 + 0.5 * i as f64).collect();
        let t_off: Vec<f64> = (0..n - 1).map(|i| 1.0 - 0.1 * (i + 1) as f64).collect();
        for j in 0..n {
            let mut residual = 0.0;
            for row in 0..n {
                let mut tz = t_diag[row] * z[row * n + j];
                if row > 0 {
                    tz += t_off[row - 1] * z[(row - 1) * n + j];
                }
                if row + 1 < n {
                    tz += t_off[row] * z[(row + 1) * n + j];
                }
                residual += (tz - d[j] * z[row * n + j]).powi(2);
            }
            assert!(
                residual.sqrt() < 1e-10,
                "eigenpair {j} residual too large: {:.3e}",
                residual.sqrt()
            );
        }
    }

    #[test]
    fn test_lobpcg_block_padding_accelerates_clustered_spectrum() {
        // Tight cluster at the bottom of the spectrum: with block size == k
        // the convergence rate of the second eigenvalue is governed by
        // λ₂/λ₃ ≈ 0.999 (worst cluster member); padding the block by 2
        // lifts the relevant ratio to λ₂/λ₅ ≈ 0.125.
        let n = 40;
        let mut diag = vec![1.0, 1.001, 1.002];
        for i in 3..n {
            diag.push(8.0 + 0.02 * (i - 3) as f64);
        }

        let k = 2;
        let base = LobpcgConfig {
            num_eigenvalues: k,
            max_iterations: 2000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let a_plain = diagonal(&diag);
        let plain = lobpcg(&a_plain, &base).unwrap();
        assert!(
            plain.converged,
            "unpadded LOBPCG should converge (iters: {})",
            plain.iterations
        );

        let padded_cfg = LobpcgConfig {
            block_padding: 2,
            ..base.clone()
        };
        let a_padded = diagonal(&diag);
        let padded = lobpcg(&a_padded, &padded_cfg).unwrap();
        assert!(
            padded.converged,
            "padded LOBPCG should converge (iters: {}, res: {:?})",
            padded.iterations, padded.residual_norms
        );

        // Only the k smallest pairs are returned, and they are accurate.
        assert_eq!(padded.eigenvalues.len(), k);
        assert_eq!(padded.eigenvectors.ncols(), k);
        for (j, expected) in diag.iter().take(k).enumerate() {
            assert!(
                (padded.eigenvalues[j] - expected).abs() < 1e-6,
                "lambda_{j}: got {}, expected {}",
                padded.eigenvalues[j],
                expected
            );
        }

        assert!(
            padded.iterations < plain.iterations,
            "padding should cut iterations on a clustered spectrum: padded={} plain={}",
            padded.iterations,
            plain.iterations
        );
    }

    #[test]
    fn test_lobpcg_padding_returns_k_smallest_of_larger_block() {
        // Padding columns that converge slowly must not leak into the
        // output: the result holds exactly the k smallest eigenpairs.
        let n = 30;
        let a = laplacian_1d(n);
        let k = 3;
        let config = LobpcgConfig {
            num_eigenvalues: k,
            max_iterations: 200,
            tolerance: 1e-6,
            block_padding: 2,
        };

        let result = lobpcg(&a, &config).unwrap();
        assert!(result.converged, "iters: {}", result.iterations);
        assert_eq!(result.eigenvalues.len(), k);
        assert_eq!(result.residual_norms.len(), k);
        assert_eq!(result.eigenvectors.ncols(), k);

        for (j, ev) in result.eigenvalues.iter().enumerate() {
            let expected =
                2.0 - 2.0 * (((j + 1) as f64) * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            assert!(
                (ev - expected).abs() < 1e-4,
                "lambda_{j}: got {ev}, expected {expected}"
            );
        }
    }

    /// Deterministic pseudo-random n×k block (same generator as the solver
    /// initialisation).
    fn pseudo_random_block(n: usize, k: usize) -> sonium_solvers_core::dense::DenseMatrix<f64> {
        deterministic_initial_block(n, k)
    }

    /// Numerical rank of a dense block via MGS: a column counts towards the
    /// rank when its norm after two reorthogonalisation passes stays above
    /// `rtol` times its original norm.
    fn mgs_numerical_rank(v: &sonium_solvers_core::dense::DenseMatrix<f64>, rtol: f64) -> usize {
        let n = v.nrows();
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut rank = 0;
        for j in 0..v.ncols() {
            let mut cj = v.column_to_vec(j);
            let norm0 = cj.iter().map(|x| x * x).sum::<f64>().sqrt();
            for _ in 0..2 {
                for q in &basis {
                    let h: f64 = q.iter().zip(cj.iter()).map(|(a, b)| a * b).sum();
                    for row in 0..n {
                        cj[row] -= h * q[row];
                    }
                }
            }
            let norm = cj.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > rtol * norm0 {
                for row in 0..n {
                    cj[row] /= norm;
                }
                basis.push(cj);
                rank += 1;
            }
        }
        rank
    }

    #[test]
    fn test_initial_block_has_full_numerical_rank() {
        // Regression test: the original hash-based generator
        //   ((i·2654435761) ^ (j·1013904223)) & 0xFFFF
        // produces a block of numerical rank ≈ 17 no matter how large the
        // requested block is, so LOBPCG could never return more than ~17–20
        // eigenpairs. The initial block must have full numerical rank for
        // every requested block size.
        let n = 256;
        for &k_block in &[10usize, 17, 25, 40, 72] {
            let x = deterministic_initial_block(n, k_block);
            let rank = mgs_numerical_rank(&x, 1e-10);
            assert_eq!(
                rank, k_block,
                "initial block rank {rank} < block size {k_block}: \
                 low-entropy initialisation caps the number of eigenpairs"
            );
        }
    }

    #[test]
    fn test_lobpcg_more_than_20_eigenpairs_1d_laplacian() {
        // Regression test for the low-entropy initial block: with the old
        // hash generator the initial block had numerical rank ≈ 17, so
        // requesting 30 eigenpairs returned only ~17–20 with
        // `converged = true`. The full-rank generator must deliver all 30.
        //
        // 1D Laplacian eigenvalues: lambda_k = 2 - 2*cos(k*pi/(n+1)).
        let n = 200;
        let k = 30;
        let a = laplacian_1d(n);

        let config = LobpcgConfig {
            num_eigenvalues: k,
            max_iterations: 500,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = lobpcg(&a, &config).unwrap();

        assert!(
            result.converged,
            "LOBPCG should converge all {k} pairs (iters: {}, worst residual: {:?})",
            result.iterations,
            result
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max)
        );
        assert_eq!(result.eigenvalues.len(), k);

        for (idx, &lam) in result.eigenvalues.iter().enumerate() {
            let analytic =
                2.0 - 2.0 * (((idx + 1) as f64) * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            assert!(
                (lam - analytic).abs() < 1e-5,
                "lambda_{}: got {lam}, expected {analytic}, diff {:.3e}",
                idx + 1,
                (lam - analytic).abs()
            );
        }
    }

    #[test]
    fn test_cholqr2_b_orthonormalizes_well_conditioned_block() {
        // CholQR2 must bring a skewed block to machine-level B-orthonormality
        // with a non-trivial (diagonal) mass operator.
        let n = 64;
        let k = 6;
        let b = diagonal(&((0..n).map(|i| 0.5 + 0.1 * i as f64).collect::<Vec<_>>()));
        let v = pseudo_random_block(n, k);
        let bv = try_apply_columns(&b, &v).unwrap();

        let (q, bq) =
            cholqr2_b(&v, &bv).expect("CholQR2 should succeed on a well-conditioned Gram");

        // The carried image is preserved: bq == B q.
        let bq_fresh = try_apply_columns(&b, &q).unwrap();
        for j in 0..k {
            for i in 0..n {
                assert!(
                    (bq[[i, j]] - bq_fresh[[i, j]]).abs() < 1e-12,
                    "carried B·v drifted at ({i}, {j})"
                );
            }
        }

        // ‖Qᵀ B Q − I‖∞ < 1e-12.
        let gram = q.t().dot(&bq);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - expected).abs() < 1e-12,
                    "not B-orthonormal: q{i}ᵀ B q{j} = {:.3e}",
                    gram[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_orthonormalize_block_b_falls_back_on_rank_deficiency() {
        // A duplicated column makes the Gram singular: the Cholesky breaks
        // down and the MGS fallback must drop the dependent column while
        // keeping the result B-orthonormal.
        let n = 32;
        let b = diagonal(&((0..n).map(|i| 1.0 + 0.05 * i as f64).collect::<Vec<_>>()));
        let mut v = pseudo_random_block(n, 3);
        // Column 3 = column 0 (exactly dependent).
        let c0 = v.column_to_vec(0);
        let mut cols: Vec<Vec<f64>> = (0..3).map(|j| v.column_to_vec(j)).collect();
        cols.push(c0);
        v = columns_to_matrix(n, cols);
        let bv = try_apply_columns(&b, &v).unwrap();

        assert!(
            cholqr2_b(&v, &bv).is_none(),
            "CholQR2 must report breakdown on a rank-deficient Gram"
        );

        let (q, bq) = orthonormalize_block_b(&v, &bv, None);
        assert_eq!(
            q.ncols(),
            3,
            "dependent column should be dropped by the fallback"
        );
        let gram = q.t().dot(&bq);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - expected).abs() < 1e-10,
                    "fallback result not B-orthonormal: q{i}ᵀ B q{j} = {:.3e}",
                    gram[[i, j]]
                );
            }
        }
    }
}
