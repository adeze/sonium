//! Communication-Avoiding Conjugate Gradient (CA-CG)
//!
//! Genuine s-step formulation. Each outer ("block") iteration performs `s`
//! CG steps while requiring only two global reductions:
//!
//! 1. **Matrix-powers kernel.** Two Krylov chains of length `s` are built in
//!    one pass with no global reductions: one from the search direction `p`
//!    and one from the preconditioned residual `z = M^{-1} r`:
//!    `V = [p, B p, ..., B^s p | z, B z, ..., B^s z]` with `B = M^{-1} A`
//!    (`B = A` when unpreconditioned). The `A`-images `U` of all but the two
//!    chain-final columns are kept as well — they come out of the chain
//!    build at no extra SpMV cost. Two chains are required so that the block
//!    recurrence can represent `z_{k+1} = M^{-1} r_{k+1}` and the updated
//!    search direction without restarting; restarting `p = z` every block
//!    would destroy CG's cross-block conjugacy and its convergence rate.
//! 2. **Single batched reduction.** Every inner product needed by the next
//!    `s` CG steps — the `(2s+2) x 2s` Gram block `V^H U`, the basis column
//!    norms (Gram diagonal), and `V^H r` — is computed in one parallel
//!    reduction pass.
//! 3. **s local CG steps.** The CG coefficients for `s` steps are computed
//!    entirely in `(2s+2)`-dimensional coordinate space: small dense
//!    algebra, no length-`n` traffic, no communication. The basis columns
//!    are normalised in coordinates via the Gram diagonal to keep the small
//!    algebra well conditioned.
//! 4. **Residual replacement.** `x` and `p` are reconstructed with fused
//!    basis-combination passes and the true residual `r = b - A x` is
//!    recomputed (one extra SpMV per block), bounding the drift of the
//!    monomial-basis recurrence.
//!
//! Synchronisation drops from 2 global reductions per CG iteration (PCG) to
//! 2 per `s` iterations. The price is ~2 SpMVs and ~2 preconditioner
//! applications per step (the two chains) plus O(n·s) basis memory — the
//! classic communication-avoiding trade of extra local work for fewer
//! global barriers.
//!
//! **Stability.** Two bases are available for the Krylov chains
//! ([`CaCgBasis`]):
//! - **Monomial (power) basis** (default). With coordinate normalisation and
//!   residual replacement this is adequate for the recommended range
//!   `s <= 4..5`, especially with preconditioning; larger `s` progressively
//!   degrades convergence (Hoemmen §7.2).
//! - **Chebyshev basis** on a user-supplied spectral interval
//!   `[λmin, λmax]` of the preconditioned operator `B = M⁻¹A`. The chains are
//!   built with the scaled three-term Chebyshev recurrence instead of raw
//!   powers, keeping the basis conditioned like `O(κ^{s/2})`-free growth —
//!   `s = 8..16` stays usable on ill-conditioned SPD systems where the
//!   monomial basis at `s = 8` is numerically rank-deficient. The two-chain
//!   structure, residual replacement, and chain-deficiency truncation are
//!   unchanged; only the chain recurrence and the (basis-dependent) mapping
//!   from `U` columns to their preconditioned images differ.
//!
//! Reference: Hoemmen, M. (2010). "Communication-Avoiding Krylov Subspace
//! Methods." PhD Thesis, UC Berkeley. Chronopoulos, A.T. & Gear, C.W.
//! (1989). "s-step iterative methods for symmetric linear systems."

use num_traits::{Float, FromPrimitive, One, Zero};
use rayon::prelude::*;
use sonium_solvers_core::blas_helpers::{inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use super::cgs_orth::{FlatBasis, KrylovOpCounts, fused_basis_update_flat};

/// Above this vector length the batched Gram pass parallelises over pairs
/// (shared [`super::cgs_orth::PAR_REDUCTION_THRESHOLD`]).
use super::cgs_orth::par_reduction_threshold;

/// Krylov basis used by the s-step matrix-powers kernel.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CaCgBasis<R> {
    /// Monomial (power) basis: `V[k] = B^k v0` with `B = M⁻¹A`. Stable for
    /// `s <= 4..5`; the default for backward compatibility.
    #[default]
    Monomial,
    /// Scaled-and-shifted Chebyshev basis on `[\lambda_min, \lambda_max]`,
    /// an interval containing the spectrum of the preconditioned operator
    /// `B = M⁻¹A` (`A` when unpreconditioned):
    /// `p_{k+1}(λ) = 2(λ − c)/d · p_k(λ) − p_{k−1}(λ)` with
    /// `c = (λmax + λmin)/2`, `d = (λmax − λmin)/2`.
    ///
    /// The interval does not need to be tight — a padded enclosing interval
    /// works — but `0 < λmin < λmax` is required (SPD spectrum); invalid
    /// intervals fall back to the monomial basis with a warning.
    Chebyshev {
        /// Lower bound on the smallest eigenvalue of `B = M⁻¹A`.
        lambda_min: R,
        /// Upper bound on the largest eigenvalue of `B = M⁻¹A`.
        lambda_max: R,
    },
}

/// Configuration for CA-CG solver.
#[derive(Debug, Clone)]
pub struct CaCgConfig<R> {
    /// Maximum total CG iterations (across all outer steps).
    pub max_iterations: usize,
    /// Relative residual tolerance for convergence.
    pub tolerance: R,
    /// Number of CG steps per outer iteration (s-step parameter).
    /// Typical values: 2-5 with the monomial basis, up to ~16 with the
    /// Chebyshev basis. Larger s = fewer syncs but less stable with the
    /// monomial basis.
    pub step_size: usize,
    /// Krylov basis for the matrix-powers kernel (default monomial).
    pub basis: CaCgBasis<R>,
    /// Print progress every N iterations (0 = no output).
    pub print_interval: usize,
}

impl Default for CaCgConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        }
    }
}

/// CA-CG solver result.
#[derive(Debug)]
pub struct CaCgSolution<T: ComplexField> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Total number of CG iterations performed.
    pub iterations: usize,
    /// Final relative residual norm.
    pub residual: T::Real,
    /// Cheap recurrence residual captured before the independent final check.
    pub estimated_residual: T::Real,
    /// Normwise backward error against the original operator.
    pub backward_error: T::Real,
    /// Whether the independently recomputed residual meets the requested tolerance.
    pub attainable_accuracy: AccuracyStatus,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Solver status.
    pub status: SolverStatus,
    /// Memory-traffic oriented operation counts for the Krylov solve.
    pub op_counts: KrylovOpCounts,
    /// Whether an invalid Chebyshev interval fell back to the monomial basis.
    /// When true, the solve ran a different algorithm than requested; check
    /// this alongside `converged` when Chebyshev stability matters (s=8..16).
    pub chebyshev_fell_back_to_monomial: bool,
}

fn numerical_error_solution<T: ComplexField>(n: usize) -> CaCgSolution<T> {
    CaCgSolution {
        x: vec![T::zero(); n],
        iterations: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
        chebyshev_fell_back_to_monomial: false,
    }
}

/// Solve Ax = b using s-step Communication-Avoiding CG.
///
/// For symmetric/Hermitian positive definite A only.
pub fn ca_cg<T, A>(operator: &A, b: &[T], config: &CaCgConfig<T::Real>) -> CaCgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_ca_cg(operator, b, config) {
        Ok(solution) => solution,
        Err(_) => numerical_error_solution(b.len()),
    }
}

/// Checked CA-CG entry point.
pub fn try_ca_cg<T, A>(
    operator: &A,
    b: &[T],
    config: &CaCgConfig<T::Real>,
) -> Result<CaCgSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let checked_operator = CheckedCaCgOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_ca_cg_inputs(operator, b, config, "CA-CG")?;
        let solution = ca_cg_inner(
            &checked_operator,
            b,
            None::<&sonium_solvers_core::traits::IdentityPreconditioner>,
            config,
        );
        let solution = finalize_ca_cg_solution(&checked_operator, b, config.tolerance, solution);
        finish_checked_ca_cg_solution(
            solution,
            b.len(),
            invalid_operator_output.load(Ordering::Relaxed),
            false,
            "CA-CG",
        )
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "CA-CG",
        reason: panic_reason(payload),
    })??;
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

/// Solve Ax = b using preconditioned s-step CA-CG.
///
/// Both A and M^{-1} must be symmetric/Hermitian positive definite.
pub fn ca_pcg<T, A, P>(
    operator: &A,
    b: &[T],
    precond: &P,
    config: &CaCgConfig<T::Real>,
) -> CaCgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_ca_pcg(operator, b, precond, config) {
        Ok(solution) => solution,
        Err(_) => numerical_error_solution(b.len()),
    }
}

/// Checked preconditioned CA-CG entry point.
pub fn try_ca_pcg<T, A, P>(
    operator: &A,
    b: &[T],
    precond: &P,
    config: &CaCgConfig<T::Real>,
) -> Result<CaCgSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let invalid_operator_output = AtomicBool::new(false);
    let invalid_preconditioner_output = AtomicBool::new(false);
    let checked_operator = CheckedCaCgOperator {
        inner: operator,
        invalid_output: &invalid_operator_output,
    };
    let checked_precond = CheckedCaCgPreconditioner {
        inner: precond,
        invalid_output: &invalid_preconditioner_output,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_ca_cg_inputs(operator, b, config, "CA-PCG")?;
        let solution = ca_cg_inner(&checked_operator, b, Some(&checked_precond), config);
        let solution = finalize_ca_cg_solution(&checked_operator, b, config.tolerance, solution);
        finish_checked_ca_cg_solution(
            solution,
            b.len(),
            invalid_operator_output.load(Ordering::Relaxed),
            invalid_preconditioner_output.load(Ordering::Relaxed),
            "CA-PCG",
        )
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "CA-PCG",
        reason: panic_reason(payload),
    })??;
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn validate_ca_cg_inputs<T, A>(
    operator: &A,
    b: &[T],
    config: &CaCgConfig<T::Real>,
    operation: &'static str,
) -> Result<(), SolverError>
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
    if config.max_iterations == 0
        || config.step_size == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!("invalid {operation} configuration"),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    Ok(())
}

fn finish_checked_ca_cg_solution<T: ComplexField>(
    solution: CaCgSolution<T>,
    expected_dimension: usize,
    invalid_operator_output: bool,
    invalid_preconditioner_output: bool,
    operation: &'static str,
) -> Result<CaCgSolution<T>, SolverError> {
    if invalid_operator_output {
        return Err(SolverError::OperationFailed {
            operation: match operation {
                "CA-PCG" => "CA-PCG operator",
                _ => "CA-CG operator",
            },
            reason: "operator generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if invalid_preconditioner_output {
        return Err(SolverError::OperationFailed {
            operation: match operation {
                "CA-PCG" => "CA-PCG preconditioner",
                _ => "CA-CG preconditioner",
            },
            reason: "preconditioner generated NaN, infinity, or an invalid output".to_string(),
        });
    }
    if solution.status == SolverStatus::NumericalError
        || solution.x.len() != expected_dimension
        || !solution.estimated_residual.is_finite()
        || !solution.backward_error.is_finite()
        || !solution_is_finite(&solution)
    {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "generated NaN, infinity, or an invalid solution".to_string(),
        });
    }
    Ok(solution)
}

fn finalize_ca_cg_solution<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut solution: CaCgSolution<T>,
) -> CaCgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let estimated_residual = solution.residual;
    let ax = operator.apply(&solution.x);
    let residual: Vec<T> = b
        .iter()
        .zip(&ax)
        .map(|(&rhs, &applied)| rhs - applied)
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
    solution.attainable_accuracy =
        if solution.residual.is_finite() && solution.residual <= tolerance {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        };
    solution.op_counts.spmv_count += 1;
    solution.op_counts.norm_count += 3;
    if !solution.residual.is_finite() || !solution.backward_error.is_finite() {
        solution.converged = false;
        solution.status = SolverStatus::NumericalError;
        solution.attainable_accuracy = AccuracyStatus::NotChecked;
    } else if solution.converged && solution.residual > tolerance {
        solution.converged = false;
        solution.status = SolverStatus::Stagnated;
    }
    solution
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "CA-CG callback panicked".to_string()
    }
}

struct CheckedCaCgOperator<'a, A> {
    inner: &'a A,
    invalid_output: &'a AtomicBool,
}

impl<T, A> LinearOperator<T> for CheckedCaCgOperator<'_, A>
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
        if output.len() != self.num_rows() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output = vec![T::zero(); self.num_rows()];
        }
        output
    }

    fn apply_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_into(input, output);
        if output.len() != self.num_rows() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
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

struct CheckedCaCgPreconditioner<'a, P> {
    inner: &'a P,
    invalid_output: &'a AtomicBool,
}

impl<T, P> Preconditioner<T> for CheckedCaCgPreconditioner<'_, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply(&self, input: &[T]) -> Vec<T> {
        let output = self.inner.apply(input);
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            vec![T::zero(); input.len()]
        } else {
            output
        }
    }

    fn apply_into(&self, input: &[T], output: &mut [T]) {
        self.inner.apply_into(input, output);
        if output.len() != input.len() || output.iter().any(|value| !value.norm().is_finite()) {
            self.invalid_output.store(true, Ordering::Relaxed);
            output.fill(T::zero());
        }
    }
}

fn solution_is_finite<T: ComplexField>(solution: &CaCgSolution<T>) -> bool {
    solution.residual.is_finite() && solution.x.iter().all(|value| value.norm().is_finite())
}

/// `(r, z)` in coordinate space: `r = r_0 - Û ŵ` and `z = V̂ ẑ`, using the
/// normalised Gram data `cv = V̂^H r_0` and `g_vu = V̂^H Û`.
///
/// `(r, z) = Σ_i conj(cv[i])·ẑ[i] − Σ_j conj(ŵ[j]) · Σ_i conj(g_vu[i][j])·ẑ[i]`
fn small_rz<T: ComplexField>(
    cv: &[T],
    g_vu: &[T],
    z_hat: &[T],
    w_hat: &[T],
    mv: usize,
    mu: usize,
) -> T {
    let mut rz = T::zero();
    for i in 0..mv {
        rz += cv[i].conj() * z_hat[i];
    }
    for j in 0..mu {
        let mut t = T::zero();
        for i in 0..mv {
            t += g_vu[i * mu + j].conj() * z_hat[i];
        }
        rz -= w_hat[j].conj() * t;
    }
    rz
}

fn ca_cg_inner<T, A, P>(
    operator: &A,
    b: &[T],
    precond: Option<&P>,
    config: &CaCgConfig<T::Real>,
) -> CaCgSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.len();
    if config.max_iterations == 0
        || config.step_size == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
        || operator.num_rows() != n
        || operator.num_cols() != n
        || b.iter().any(|value| !value.norm().is_finite())
    {
        return numerical_error_solution(n);
    }
    let s = config.step_size.max(1);
    let mut x = vec![T::zero(); n];
    let mut op_counts = KrylovOpCounts::default();
    let has_preconditioner = precond.is_some();
    let two = T::Real::from_f64(2.0).unwrap();

    // Chebyshev interval → (center, half-width) of the mapped recurrence.
    // The interval refers to the spectrum of the preconditioned operator
    // B = M⁻¹A. Invalid intervals fall back to the monomial basis and set
    // `chebyshev_fell_back_to_monomial` on the solution so the caller learns
    // the basis changed (a `log::warn` alone is not machine-readable).
    let mut chebyshev_fell_back_to_monomial = false;
    let cheb = match config.basis {
        CaCgBasis::Monomial => None,
        CaCgBasis::Chebyshev {
            lambda_min,
            lambda_max,
        } => {
            if lambda_min > T::Real::zero() && lambda_max > lambda_min {
                Some((
                    (lambda_max + lambda_min) / two,
                    (lambda_max - lambda_min) / two,
                ))
            } else {
                log::warn!("CA-CG: invalid Chebyshev interval; falling back to monomial basis");
                chebyshev_fell_back_to_monomial = true;
                None
            }
        }
    };

    let b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tol_threshold {
        return CaCgSolution {
            x,
            iterations: 0,
            residual: T::Real::from_f64(0.0).unwrap(),
            estimated_residual: T::Real::from_f64(0.0).unwrap(),
            backward_error: T::Real::infinity(),
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: true,
            status: SolverStatus::Converged,
            op_counts,
            chebyshev_fell_back_to_monomial,
        };
    }

    // Breakdown floor, mirroring cg.rs.
    let scaled = b_norm * T::Real::from_f64(1e-20).unwrap();
    let floor = T::Real::from_f64(1e-30).unwrap();
    let breakdown_threshold = if scaled > floor { scaled } else { floor };

    // Chain-deficiency threshold: a basis vector whose norm drops below
    // ~eps of its chain seed norm is treated as Krylov exhaustion.
    let def_thresh = T::Real::from_f64(100.0).unwrap() * T::Real::epsilon();

    // Block state: r = b - A x (true residual), z = M^{-1} r, p = direction.
    let mut r = b.to_vec();
    let mut z = apply_precond(precond, &r);
    if has_preconditioner {
        op_counts.preconditioner_count += 1;
    }
    let mut p = z.clone();
    let mut ax = vec![T::zero(); n];

    // Bases sized for a full block; short final blocks use prefixes.
    let mut basis_v = FlatBasis::new(2 * s + 2, n);
    let mut basis_u = FlatBasis::new(2 * s, n);
    let mut scratch = vec![T::zero(); n];
    let mut scratch2 = vec![T::zero(); n];

    let mut total_iters = 0usize;

    // Outer loop: each block performs up to s CG steps.
    loop {
        if total_iters >= config.max_iterations {
            break;
        }
        let m = s.min(config.max_iterations - total_iters);
        let mv = 2 * m + 2; // V columns: two chains of length m+1
        let mu = 2 * m; // U columns: A-images of all but the chain ends

        // ============================================================
        // Phase 1: matrix-powers kernel — two Krylov chains, no reductions.
        //   p-chain: V[k]     = p_k(B) p,  U[k]   = A V[k]
        //   z-chain: V[m+1+k] = p_k(B) z,  U[m+k] = A V[m+1+k]
        // with p_k the monomial λ^k or the scaled Chebyshev polynomial,
        // depending on the configured basis.
        // ============================================================
        basis_v.vector_mut(0).copy_from_slice(&p);
        basis_v.vector_mut(m + 1).copy_from_slice(&z);
        for k in 0..m {
            chain_column(
                operator,
                precond,
                &mut basis_v,
                &mut basis_u,
                k,
                k,
                k + 1,
                k,
                cheb,
                &mut scratch,
                &mut scratch2,
                has_preconditioner,
                &mut op_counts,
            );
            chain_column(
                operator,
                precond,
                &mut basis_v,
                &mut basis_u,
                m + k,
                m + 1 + k,
                m + 2 + k,
                k,
                cheb,
                &mut scratch,
                &mut scratch2,
                has_preconditioner,
                &mut op_counts,
            );
        }

        // ============================================================
        // Phase 2: one batched reduction for the whole block.
        //   dots = [diag(V^H V) | diag(U^H U) | V^H U | V^H r]
        // ============================================================
        let r_idx = mv + mu;
        let mut cols: Vec<&[T]> = Vec::with_capacity(r_idx + 1);
        for i in 0..mv {
            cols.push(basis_v.vector(i));
        }
        for j in 0..mu {
            cols.push(basis_u.vector(j));
        }
        cols.push(&r);

        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(mv + mu + mv * mu + mv);
        for i in 0..mv {
            pairs.push((i, i));
        }
        for j in 0..mu {
            pairs.push((mv + j, mv + j));
        }
        for i in 0..mv {
            for j in 0..mu {
                pairs.push((i, mv + j));
            }
        }
        for i in 0..mv {
            pairs.push((i, r_idx));
        }

        let ndots = pairs.len();
        let dots: Vec<T> = if n >= par_reduction_threshold() && ndots > 1 {
            pairs
                .par_iter()
                .map(|&(a, b)| inner_product(cols[a], cols[b]))
                .collect()
        } else {
            pairs
                .iter()
                .map(|&(a, b)| inner_product(cols[a], cols[b]))
                .collect()
        };
        op_counts.dot_reductions += ndots;
        op_counts.synchronization_points += 1;

        let g_base = mv + mu;
        let c_base = g_base + mv * mu;

        // Column norms from the Gram diagonal.
        let mut d_v = vec![T::Real::zero(); mv];
        let mut d_u = vec![T::Real::zero(); mu];
        for i in 0..mv {
            d_v[i] = dots[i].norm().sqrt();
        }
        for j in 0..mu {
            d_u[j] = dots[mv + j].norm().sqrt();
        }

        // p ≈ 0 or z ≈ 0: nothing to build on. r is the true residual here
        // (residual replacement), so decide converged-vs-breakdown exactly.
        if d_v[0] < breakdown_threshold || d_v[m + 1] < breakdown_threshold {
            let r_norm = vector_norm(&r);
            op_counts.norm_count += 1;
            let rel_res = r_norm / b_norm;
            let converged = rel_res <= config.tolerance;
            return CaCgSolution {
                x,
                iterations: total_iters,
                residual: rel_res,
                estimated_residual: rel_res,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged,
                status: if converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Breakdown
                },
                op_counts,
                chebyshev_fell_back_to_monomial,
            };
        }

        // Truncate the block if a chain exhausts its Krylov space.
        let p_def = (1..=m)
            .find(|&k| d_v[k] < def_thresh * d_v[0])
            .unwrap_or(m + 1);
        let z_def = (1..=m)
            .find(|&k| d_v[m + 1 + k] < def_thresh * d_v[m + 1])
            .unwrap_or(m + 1);
        let m_eff = m.min(p_def.saturating_sub(1)).min(z_def.saturating_sub(1));
        if m_eff == 0 {
            let r_norm = vector_norm(&r);
            op_counts.norm_count += 1;
            let rel_res = r_norm / b_norm;
            let converged = rel_res <= config.tolerance;
            return CaCgSolution {
                x,
                iterations: total_iters,
                residual: rel_res,
                estimated_residual: rel_res,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged,
                status: if converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Breakdown
                },
                op_counts,
                chebyshev_fell_back_to_monomial,
            };
        }

        // Normalised Gram data for the coordinate recurrences:
        //   g_vu[i][j] = (v̂_i, û_j),  cv[i] = (v̂_i, r_0)
        let mut g_vu = vec![T::zero(); mv * mu];
        for i in 0..mv {
            for j in 0..mu {
                let denom = d_v[i] * d_u[j];
                g_vu[i * mu + j] = if denom > T::Real::zero() {
                    dots[g_base + i * mu + j] / T::from_real(denom)
                } else {
                    T::zero()
                };
            }
        }
        let mut cv = vec![T::zero(); mv];
        for i in 0..mv {
            cv[i] = if d_v[i] > T::Real::zero() {
                dots[c_base + i] / T::from_real(d_v[i])
            } else {
                T::zero()
            };
        }

        // ============================================================
        // Phase 3: m_eff CG steps in coordinate space (no length-n ops).
        //
        // Tracked coordinates (normalised basis V̂ = V D_v^{-1}, Û = U D_u^{-1}):
        //   x update: x̂      with x += V̂ x̂
        //   residual: ŵ      with r = r_0 − Û ŵ
        //   precond:  ẑ      with z = V̂ ẑ   (z = z_0 − M^{-1} Û ŵ, with
        //             M^{-1} û_j expanded in V̂ via nxt_map — see below)
        //   direction: p̂     with p = V̂ p̂
        //
        // nxt_map[j] lists the (v-index, real coefficient) terms of M⁻¹ û_j
        // in the normalised basis:
        //   monomial:  M⁻¹ u_j = V[nxt]                        (1 term)
        //   Chebyshev: B v_k = (d/2) v_{k+1} + c v_k + (d/2) v_{k−1}
        //              (B v_0 = d v_1 + c v_0 at a chain start) (2-3 terms)
        // since U[j] = A V[src] and M⁻¹ A = B.
        // ============================================================
        let mut nxt_map = vec![[(0usize, T::Real::zero()); 3]; mu];
        let mut nxt_len = vec![0usize; mu];
        for j in 0..mu {
            let src = if j < m { j } else { j + 1 };
            if d_u[j] <= T::Real::zero() {
                continue;
            }
            let inv_du = T::Real::one() / d_u[j];
            match cheb {
                None => {
                    let nxt = if j < m { j + 1 } else { j + 2 };
                    nxt_map[j][0] = (nxt, d_v[nxt] * inv_du);
                    nxt_len[j] = 1;
                }
                Some((center, half_width)) => {
                    if d_v[src] <= T::Real::zero() {
                        continue;
                    }
                    let local = if j < m { j } else { j - m };
                    let up = if local == 0 {
                        half_width
                    } else {
                        half_width / two
                    };
                    let mut t = 0;
                    nxt_map[j][t] = (src + 1, up * d_v[src + 1] * inv_du);
                    t += 1;
                    nxt_map[j][t] = (src, center * d_v[src] * inv_du);
                    t += 1;
                    if local > 0 {
                        nxt_map[j][t] = (src - 1, (half_width / two) * d_v[src - 1] * inv_du);
                        t += 1;
                    }
                    nxt_len[j] = t;
                }
            }
        }
        let mut p_hat = vec![T::zero(); mv];
        let mut z_hat = vec![T::zero(); mv];
        let mut x_hat = vec![T::zero(); mv];
        let mut w_hat = vec![T::zero(); mu];
        let mut c = vec![T::zero(); mu];
        p_hat[0] = T::from_real(d_v[0]);
        z_hat[m + 1] = T::from_real(d_v[m + 1]);

        let mut rz = small_rz(&cv, &g_vu, &z_hat, &w_hat, mv, mu);
        let mut steps_done = 0usize;
        let mut broke_down = false;

        for _k in 0..m_eff {
            // A p in Û coordinates: A v̂_i = (d_u[img] / d_v[i]) û_{img}.
            // src[j] maps U column j back to its V source column.
            for j in 0..mu {
                let src = if j < m { j } else { j + 1 };
                c[j] = if d_v[src] > T::Real::zero() {
                    p_hat[src] * T::from_real(d_u[j] / d_v[src])
                } else {
                    T::zero()
                };
            }

            // (p, A p) = p̂^H · G_vu · c
            let mut pap = T::zero();
            for i in 0..mv {
                let mut t = T::zero();
                for j in 0..mu {
                    t += g_vu[i * mu + j] * c[j];
                }
                pap += p_hat[i].conj() * t;
            }
            if pap.norm() < breakdown_threshold {
                broke_down = true;
                break;
            }

            let alpha = rz / pap;
            for i in 0..mv {
                x_hat[i] += alpha * p_hat[i];
            }
            for j in 0..mu {
                w_hat[j] += alpha * c[j];
            }

            // z = M^{-1} r in V̂ coordinates: z = z_0 − M^{-1} Û ŵ.
            // nxt_map[j] gives the V̂ expansion of M^{-1} u_j (1 term for the
            // monomial basis, up to 3 for the Chebyshev three-term recurrence).
            z_hat.fill(T::zero());
            z_hat[m + 1] = T::from_real(d_v[m + 1]);
            for j in 0..mu {
                for &(idx, coef) in &nxt_map[j][..nxt_len[j]] {
                    z_hat[idx] -= w_hat[j] * T::from_real(coef);
                }
            }

            let rz_new = small_rz(&cv, &g_vu, &z_hat, &w_hat, mv, mu);
            steps_done += 1;
            if rz_new.norm() < breakdown_threshold {
                // r ≈ 0 — verify with the true residual after reconstruction.
                broke_down = true;
                break;
            }
            let beta = rz_new / rz;
            rz = rz_new;
            for i in 0..mv {
                p_hat[i] = z_hat[i] + beta * p_hat[i];
            }
        }

        // ============================================================
        // Phase 4: fused reconstruction + residual replacement.
        // ============================================================
        let mut coeffs = vec![T::zero(); mv];
        for i in 0..mv {
            coeffs[i] = if d_v[i] > T::Real::zero() {
                x_hat[i] / T::from_real(d_v[i])
            } else {
                T::zero()
            };
        }
        fused_basis_update_flat(&basis_v, &coeffs, &mut x);
        op_counts.axpy_passes += 1;

        for i in 0..mv {
            coeffs[i] = if d_v[i] > T::Real::zero() {
                p_hat[i] / T::from_real(d_v[i])
            } else {
                T::zero()
            };
        }
        p.fill(T::zero());
        fused_basis_update_flat(&basis_v, &coeffs, &mut p);
        op_counts.axpy_passes += 1;

        total_iters += steps_done;

        // True residual r = b - A x: one extra SpMV per block.
        operator.apply_into(&x, &mut ax);
        op_counts.spmv_count += 1;
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
        op_counts.axpy_passes += 1;
        op_counts.exact_residual_checks += 1;
        let r_norm = vector_norm(&r);
        op_counts.norm_count += 1;
        op_counts.synchronization_points += 1;
        let rel_res = r_norm / b_norm;

        if config.print_interval > 0 {
            let rel_f64 = num_traits::ToPrimitive::to_f64(&rel_res).unwrap_or(0.0);
            log::info!("CA-CG iter {}: rel_res = {:.4e}", total_iters, rel_f64);
        }

        if rel_res <= config.tolerance {
            return CaCgSolution {
                x,
                iterations: total_iters,
                residual: rel_res,
                estimated_residual: rel_res,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
                chebyshev_fell_back_to_monomial,
            };
        }
        if broke_down || steps_done == 0 {
            return CaCgSolution {
                x,
                iterations: total_iters,
                residual: rel_res,
                estimated_residual: rel_res,
                backward_error: T::Real::infinity(),
                attainable_accuracy: AccuracyStatus::NotChecked,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
                chebyshev_fell_back_to_monomial,
            };
        }

        // z = M^{-1} r for the next block.
        apply_precond_into(precond, &r, &mut z);
        if has_preconditioner {
            op_counts.preconditioner_count += 1;
        }
    }

    // r already holds the true residual from the last block.
    let r_norm = vector_norm(&r);
    op_counts.norm_count += 1;
    CaCgSolution {
        x,
        iterations: total_iters,
        residual: r_norm / b_norm,
        estimated_residual: r_norm / b_norm,
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
        chebyshev_fell_back_to_monomial,
    }
}

/// Advance one Krylov chain by one column, recording the A-image in `U`.
///
/// Monomial:   `V[v_dst] = B V[v_src]`
/// Chebyshev:  `V[v_dst] = a_k B V[v_src] + b_k V[v_src] + c_k V[v_src−1]`
/// with `(a_0, b_0, c_0) = (1/d, −c/d, 0)` at a chain start (`local_k = 0`)
/// and `(a_k, b_k, c_k) = (2/d, −2c/d, −1)` afterwards, where `(c, d)` are
/// the interval center and half-width. In both cases `U[u_dst] = A V[v_src]`
/// is kept as a by-product of the chain build (no extra SpMV).
#[allow(clippy::too_many_arguments)]
fn chain_column<T, A, P>(
    operator: &A,
    precond: Option<&P>,
    basis_v: &mut FlatBasis<T>,
    basis_u: &mut FlatBasis<T>,
    u_dst: usize,
    v_src: usize,
    v_dst: usize,
    local_k: usize,
    cheb: Option<(T::Real, T::Real)>,
    scratch: &mut [T],
    scratch2: &mut [T],
    has_preconditioner: bool,
    op_counts: &mut KrylovOpCounts,
) where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    operator.apply_into(basis_v.vector(v_src), scratch);
    op_counts.spmv_count += 1;
    basis_u.vector_mut(u_dst).copy_from_slice(scratch);
    apply_precond_into(precond, scratch, scratch2);
    if has_preconditioner {
        op_counts.preconditioner_count += 1;
    }

    match cheb {
        None => basis_v.vector_mut(v_dst).copy_from_slice(scratch2),
        Some((center, half_width)) => {
            let (a, b_coef, c_coef) = if local_k == 0 {
                (
                    T::Real::one() / half_width,
                    -center / half_width,
                    T::Real::zero(),
                )
            } else {
                let two = T::Real::from_f64(2.0).unwrap();
                (
                    two / half_width,
                    -two * center / half_width,
                    -T::Real::one(),
                )
            };
            let a_t = T::from_real(a);
            let b_t = T::from_real(b_coef);
            let c_t = T::from_real(c_coef);
            let n = scratch.len();
            if local_k == 0 {
                let v_k = basis_v.vector(v_src);
                for i in 0..n {
                    scratch[i] = a_t * scratch2[i] + b_t * v_k[i];
                }
            } else {
                let v_k = basis_v.vector(v_src);
                let v_km1 = basis_v.vector(v_src - 1);
                for i in 0..n {
                    scratch[i] = a_t * scratch2[i] + b_t * v_k[i] + c_t * v_km1[i];
                }
            }
            basis_v.vector_mut(v_dst).copy_from_slice(scratch);
        }
    }
}

fn apply_precond<T: ComplexField, P: Preconditioner<T>>(precond: Option<&P>, r: &[T]) -> Vec<T> {
    match precond {
        Some(pc) => pc.apply(r),
        None => r.to_vec(),
    }
}

fn apply_precond_into<T: ComplexField, P: Preconditioner<T>>(
    precond: Option<&P>,
    r: &[T],
    z: &mut [T],
) {
    match precond {
        Some(pc) => pc.apply_into(r, z),
        None => z.copy_from_slice(r),
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

    /// Complex Hermitian positive definite tridiagonal:
    /// diagonal 4, super-diagonal `off`, sub-diagonal `conj(off)`.
    /// Strictly diagonally dominant (4 > 2·|off|), hence HPD.
    fn hermitian_tridiag(n: usize) -> CsrMatrix<Complex64> {
        let off = Complex64::new(0.5, 0.25);
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(off.conj());
            }
            col_indices.push(i);
            values.push(Complex64::new(4.0, 0.0));
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(off);
            }
            row_ptrs.push(col_indices.len());
        }

        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    #[test]
    fn ca_cg_rejects_non_finite_input_before_operator_use() {
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

        let solution = ca_cg(&PanicOperator, &[f64::NAN], &CaCgConfig::default());
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn ca_cg_rejects_non_finite_operator_output() {
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

        let solution = ca_cg(&NonFiniteOperator, &[1.0], &CaCgConfig::default());
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert!(solution.x.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn try_ca_cg_rejects_non_finite_operator_output() {
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

        let error = try_ca_cg(&NonFiniteOperator, &[1.0], &CaCgConfig::default())
            .expect_err("checked CA-CG must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn try_ca_cg_contains_operator_panics() {
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

        let error = try_ca_cg(&PanicOperator, &[1.0], &CaCgConfig::default())
            .expect_err("checked CA-CG must contain operator panics");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let legacy = ca_cg(&PanicOperator, &[1.0], &CaCgConfig::default());
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn try_ca_pcg_rejects_non_finite_preconditioner_output() {
        struct NonFinitePreconditioner;

        impl Preconditioner<f64> for NonFinitePreconditioner {
            fn apply(&self, input: &[f64]) -> Vec<f64> {
                vec![f64::NAN; input.len()]
            }

            fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
                output.fill(f64::NAN);
            }
        }

        let operator = CsrMatrix::<f64>::identity(2);
        let error = try_ca_pcg(
            &operator,
            &[1.0, 2.0],
            &NonFinitePreconditioner,
            &CaCgConfig {
                max_iterations: 4,
                step_size: 2,
                ..CaCgConfig::default()
            },
        )
        .expect_err("checked CA-PCG must reject generated non-finite values");
        assert!(matches!(error, SolverError::OperationFailed { .. }));

        let legacy = ca_pcg(
            &operator,
            &[1.0, 2.0],
            &NonFinitePreconditioner,
            &CaCgConfig {
                max_iterations: 4,
                step_size: 2,
                ..CaCgConfig::default()
            },
        );
        assert_eq!(legacy.status, SolverStatus::NumericalError);
        assert!(!legacy.converged);
    }

    #[test]
    fn test_ca_cg_converges() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let config = CaCgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };

        let result = ca_cg(&a, &b, &config);
        assert!(
            result.converged,
            "CA-CG should converge: {} iters, res {:.2e}",
            result.iterations, result.residual
        );
        assert!(result.residual < 1e-10);
        assert!(result.estimated_residual.is_finite());
        assert!(result.backward_error.is_finite());
        assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
    }

    #[test]
    fn test_ca_cg_matches_standard_cg() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let cg_config = crate::CgConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let cg_result = crate::cg(&a, &b, &cg_config);

        let ca_config = CaCgConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };
        let ca_result = ca_cg(&a, &b, &ca_config);

        assert!(cg_result.converged);
        assert!(ca_result.converged);

        // In exact arithmetic CA-CG and CG produce the same iterates; with
        // the monomial basis we tolerate a small iteration drift.
        let iter_diff = cg_result.iterations.abs_diff(ca_result.iterations);
        assert!(
            iter_diff <= 4,
            "CA-CG ({}) and CG ({}) iteration counts diverged",
            ca_result.iterations,
            cg_result.iterations
        );

        // Solutions should be close
        let diff: f64 = cg_result
            .x
            .iter()
            .zip(ca_result.x.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>();
        assert!(diff < 1e-6, "CA-CG and CG solutions differ by {diff:.2e}");
    }

    #[test]
    fn test_ca_cg_different_step_sizes() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        // Monomial basis: stable for s <= 4..5 (documented limitation).
        for s in [1, 2, 4] {
            let config = CaCgConfig {
                max_iterations: 500,
                tolerance: 1e-10,
                step_size: s,
                basis: CaCgBasis::Monomial,
                print_interval: 0,
            };
            let result = ca_cg(&a, &b, &config);
            assert!(
                result.converged,
                "CA-CG with s={s} should converge: {} iters, res {:.2e}",
                result.iterations, result.residual
            );
        }
    }

    #[test]
    fn test_ca_cg_reports_krylov_op_counts() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let config = CaCgConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };

        let result = ca_cg(&a, &b, &config);

        assert!(result.converged);
        assert!(result.op_counts.spmv_count >= result.iterations);
        assert!(result.op_counts.dot_reductions >= result.iterations);
        assert!(result.op_counts.norm_count > 0);
        // Reconstruction is batched: ~3 fused passes per block of s steps.
        assert!(result.op_counts.axpy_passes >= result.iterations / 4);
        assert_eq!(result.op_counts.preconditioner_count, 0);
        // Residual replacement runs once per block.
        assert!(result.op_counts.exact_residual_checks > 0);
    }

    #[test]
    fn test_ca_cg_reduces_synchronisation_points() {
        // The whole point of CA-CG: ~s× fewer global reductions than CG.
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let cg_config = crate::CgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let cg_result = crate::cg(&a, &b, &cg_config);

        let ca_config = CaCgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };
        let ca_result = ca_cg(&a, &b, &ca_config);

        assert!(cg_result.converged);
        assert!(ca_result.converged);

        // CG uses >= 2 syncs per iteration; CA-CG uses 2 per block of 4.
        let cg_syncs = cg_result.op_counts.synchronization_points;
        let ca_syncs = ca_result.op_counts.synchronization_points;
        assert!(
            ca_syncs * 2 <= cg_syncs + 4,
            "CA-CG syncs ({ca_syncs}) should be at least 2x fewer than CG ({cg_syncs})"
        );
    }

    #[test]
    fn test_ca_pcg_with_preconditioner() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);
        let config = CaCgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };

        let result_no_pc = ca_cg(&a, &b, &config);
        let result_pc = ca_pcg(&a, &b, &precond, &config);

        assert!(result_no_pc.converged);
        assert!(result_pc.converged);
        // Preconditioned should converge in fewer or equal iterations
        assert!(
            result_pc.iterations <= result_no_pc.iterations,
            "Preconditioned ({}) should be no slower than unpreconditioned ({})",
            result_pc.iterations,
            result_no_pc.iterations
        );
    }

    #[test]
    fn test_ca_pcg_matches_pcg() {
        // CA-PCG must converge to the same answer as reference PCG.
        let n = 100;
        let a = laplacian_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 0.5).collect();

        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let cg_config = crate::CgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let pcg_result = crate::pcg(&a, &precond, &b, &cg_config);

        let ca_config = CaCgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };
        let ca_result = ca_pcg(&a, &b, &precond, &ca_config);

        assert!(pcg_result.converged);
        assert!(ca_result.converged);

        let iter_diff = pcg_result.iterations.abs_diff(ca_result.iterations);
        assert!(
            iter_diff <= 4,
            "CA-PCG ({}) and PCG ({}) iteration counts diverged",
            ca_result.iterations,
            pcg_result.iterations
        );

        let diff: f64 = pcg_result
            .x
            .iter()
            .zip(ca_result.x.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>();
        assert!(diff < 1e-6, "CA-PCG and PCG solutions differ by {diff:.2e}");
    }

    #[test]
    fn test_ca_pcg_reports_preconditioner_counts() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);
        let config = CaCgConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };

        let result = ca_pcg(&a, &b, &precond, &config);

        assert!(result.converged);
        assert!(result.op_counts.spmv_count >= result.iterations);
        assert!(result.op_counts.preconditioner_count >= result.iterations);
        assert!(result.op_counts.dot_reductions >= result.iterations);
        assert!(result.op_counts.norm_count > 0);
    }

    #[test]
    fn test_ca_cg_complex_hermitian() {
        // Complex Hermitian positive definite system: CA-CG must match CG.
        let n = 60;
        let a = hermitian_tridiag(n);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin() + 1.0, 0.3 * (i as f64).cos()))
            .collect();

        let cg_config = crate::CgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let cg_result = crate::cg(&a, &b, &cg_config);

        let ca_config = CaCgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };
        let ca_result = ca_cg(&a, &b, &ca_config);

        assert!(cg_result.converged, "CG should converge on complex HPD");
        assert!(
            ca_result.converged,
            "CA-CG should converge on complex HPD: {} iters, res {:.2e}",
            ca_result.iterations, ca_result.residual
        );

        let diff: f64 = cg_result
            .x
            .iter()
            .zip(ca_result.x.iter())
            .map(|(&a, &b)| (a - b).norm())
            .sum::<f64>();
        assert!(
            diff < 1e-6,
            "CA-CG and CG complex solutions differ by {diff:.2e}"
        );
    }

    #[test]
    fn test_ca_pcg_complex_hermitian_preconditioned() {
        // Complex HPD system with Jacobi preconditioning: CA-PCG vs PCG.
        let n = 60;
        let a = hermitian_tridiag(n);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).cos() + 0.5, (i as f64).sin()))
            .collect();
        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);

        let cg_config = crate::CgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let pcg_result = crate::pcg(&a, &precond, &b, &cg_config);

        let ca_config = CaCgConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            step_size: 4,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        };
        let ca_result = ca_pcg(&a, &b, &precond, &ca_config);

        assert!(pcg_result.converged);
        assert!(
            ca_result.converged,
            "CA-PCG should converge on complex HPD: {} iters, res {:.2e}",
            ca_result.iterations, ca_result.residual
        );

        let diff: f64 = pcg_result
            .x
            .iter()
            .zip(ca_result.x.iter())
            .map(|(&a, &b)| (a - b).norm())
            .sum::<f64>();
        assert!(
            diff < 1e-6,
            "CA-PCG and PCG complex solutions differ by {diff:.2e}"
        );
    }

    /// Exact spectral interval of the 1D Laplacian of size n, padded by the
    /// given relative margins.
    fn laplacian_1d_spectrum(n: usize, lo_pad: f64, hi_pad: f64) -> (f64, f64) {
        let lam_min = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        let lam_max = 2.0 - 2.0 * (n as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
        (lam_min * (1.0 - lo_pad), lam_max * (1.0 + hi_pad))
    }

    fn true_rel_residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let ax = a.apply(x);
        let mut res = 0.0;
        let mut bn = 0.0;
        for i in 0..b.len() {
            res += (b[i] - ax[i]).powi(2);
            bn += b[i].powi(2);
        }
        res.sqrt() / bn.sqrt()
    }

    #[test]
    fn test_ca_cg_chebyshev_s8_beats_monomial_s8() {
        // Moderately ill-conditioned SPD system (κ ≈ 2.6e4). At s = 8 the
        // monomial Krylov basis has a Gram condition number ~κ^s, far beyond
        // f64 resolution, so it stagnates; the Chebyshev basis on a padded
        // spectral interval stays well conditioned and converges.
        let n = 256;
        let a = laplacian_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() + 0.25).collect();
        let (lam_min, lam_max) = laplacian_1d_spectrum(n, 0.1, 0.05);

        let mk_config = |basis| CaCgConfig {
            max_iterations: 2000,
            tolerance: 1e-8,
            step_size: 8,
            basis,
            print_interval: 0,
        };

        let mono = ca_cg(&a, &b, &mk_config(CaCgBasis::Monomial));
        let cheb = ca_cg(
            &a,
            &b,
            &mk_config(CaCgBasis::Chebyshev {
                lambda_min: lam_min,
                lambda_max: lam_max,
            }),
        );

        let mono_rel = true_rel_residual(&a, &mono.x, &b);
        let cheb_rel = true_rel_residual(&a, &cheb.x, &b);

        assert!(
            cheb.converged,
            "Chebyshev CA-CG at s=8 should converge: {} iters, true rel res {:.2e}",
            cheb.iterations, cheb_rel
        );
        assert!(cheb_rel < 1e-8);
        assert!(
            !mono.converged || mono_rel > 1e2 * cheb_rel,
            "monomial s=8 should be far less accurate than Chebyshev s=8: \
             mono rel res {mono_rel:.2e} (converged={}), cheb rel res {cheb_rel:.2e}",
            mono.converged
        );
    }

    #[test]
    fn test_ca_cg_chebyshev_matches_cg_at_s8() {
        // Chebyshev CA-CG at s = 8 must produce CG's answer.
        let n = 256;
        let a = laplacian_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() + 0.25).collect();
        let (lam_min, lam_max) = laplacian_1d_spectrum(n, 0.1, 0.05);

        let cg_config = crate::CgConfig {
            max_iterations: 2000,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let cg_result = crate::cg(&a, &b, &cg_config);

        let cheb = ca_cg(
            &a,
            &b,
            &CaCgConfig {
                max_iterations: 2000,
                tolerance: 1e-10,
                step_size: 8,
                basis: CaCgBasis::Chebyshev {
                    lambda_min: lam_min,
                    lambda_max: lam_max,
                },
                print_interval: 0,
            },
        );

        assert!(cg_result.converged);
        assert!(
            cheb.converged,
            "Chebyshev CA-CG at s=8 should converge: {} iters, res {:.2e}",
            cheb.iterations, cheb.residual
        );

        let diff: f64 = cg_result
            .x
            .iter()
            .zip(cheb.x.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>();
        assert!(
            diff < 1e-5,
            "Chebyshev CA-CG and CG solutions differ by {diff:.2e}"
        );
    }

    #[test]
    fn test_ca_pcg_chebyshev_preconditioned() {
        // Chebyshev basis on the *preconditioned* spectrum. For the 1D
        // Laplacian with Jacobi, B = A / 2, so the interval is the
        // unpreconditioned one scaled by 1/2.
        let n = 256;
        let a = laplacian_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).cos() + 0.5).collect();
        let (lam_min, lam_max) = laplacian_1d_spectrum(n, 0.1, 0.05);

        let precond = sonium_solvers_preconditioners::DiagonalPreconditioner::from_csr(&a);
        let result = ca_pcg(
            &a,
            &b,
            &precond,
            &CaCgConfig {
                max_iterations: 2000,
                tolerance: 1e-8,
                step_size: 8,
                basis: CaCgBasis::Chebyshev {
                    lambda_min: lam_min / 2.0,
                    lambda_max: lam_max / 2.0,
                },
                print_interval: 0,
            },
        );

        assert!(
            result.converged,
            "preconditioned Chebyshev CA-CG at s=8 should converge: {} iters, res {:.2e}",
            result.iterations, result.residual
        );
        assert!(true_rel_residual(&a, &result.x, &b) < 1e-8);
        assert!(result.op_counts.preconditioner_count >= result.iterations);
    }

    #[test]
    fn test_ca_cg_chebyshev_invalid_interval_falls_back() {
        // An invalid interval (λmin ≤ 0) falls back to the monomial basis:
        // at s = 4 the behaviour should match the plain monomial config.
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        let fallback = ca_cg(
            &a,
            &b,
            &CaCgConfig {
                max_iterations: 500,
                tolerance: 1e-10,
                step_size: 4,
                basis: CaCgBasis::Chebyshev {
                    lambda_min: -1.0,
                    lambda_max: 4.0,
                },
                print_interval: 0,
            },
        );
        let mono = ca_cg(
            &a,
            &b,
            &CaCgConfig {
                max_iterations: 500,
                tolerance: 1e-10,
                step_size: 4,
                basis: CaCgBasis::Monomial,
                print_interval: 0,
            },
        );

        assert!(fallback.converged);
        assert_eq!(fallback.iterations, mono.iterations);
        assert!(
            fallback.chebyshev_fell_back_to_monomial,
            "invalid interval must be reported on the solution"
        );
        assert!(!mono.chebyshev_fell_back_to_monomial);
    }
}
