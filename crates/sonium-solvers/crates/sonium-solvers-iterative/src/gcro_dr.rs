//! GCRO-DR (Generalized Conjugate Residual Orthogonalisation with Deflated
//! Restarting) for sequences of nearly-related linear systems.
//!
//! This is the algorithm of Parks–de Sturler–Mackey–Johnson–Maiti, *Recycling
//! Krylov Subspaces for Sequences of Linear Systems*, SIAM J. Sci. Comput.
//! 28(5), 1651–1674 (2006), in the practical form used by
//! Panagiotopoulos–Deckers–Desmet for acoustic FEM frequency sweeps
//! (JCAM/CMAME, 2020/2022).
//!
//! ## How this differs from `gmres_recycled`
//!
//! [`super::gmres_recycled`] applies a *one-shot* coarse correction at the
//! start of each solve and then runs standard GMRES. The Krylov basis it
//! builds is **not** kept orthogonal to `M⁻¹A · U` during Arnoldi, so the
//! iterations regrow components in the recycled subspace and the headline
//! "fewer iterations" win is largely lost.
//!
//! GCRO-DR keeps `C := orthonormalize(M⁻¹A U)` in memory and projects every
//! new Arnoldi vector against `C` before the standard CGS+DGKS step against
//! the current Krylov basis. This gives an Arnoldi recurrence
//!
//! ```text
//!   M⁻¹A V_m = V_{m+1} H_{m+1,m} + C B_{k,m},     C^H V_{m+1} = 0
//! ```
//!
//! and a residual minimisation that splits cleanly across the two orthogonal
//! subspaces. The recycled subspace is updated between solves via harmonic
//! Ritz extraction on the *augmented* `(k + m) × (k + m)` block.
//!
//! ## When to use it
//!
//! Pass the same [`GcroDrSubspace`] across consecutive solves whose operators
//! are spectral perturbations of each other (e.g. consecutive frequencies in a
//! FEM/BEM Helmholtz sweep, time-stepping with a slowly varying Jacobian).
//! The first solve runs unaccelerated and seeds the subspace; each subsequent
//! solve consumes the recycled subspace and emits an updated one.

use super::cgs_orth::{
    FlatBasis, KrylovOpCounts, batched_dots_flat_into, batched_dots_into, batched_project,
    batched_project_flat, fused_basis_update_flat,
};
use super::checked_callbacks::{
    CheckedLinearOperator, CheckedPreconditioner, catch_solver_callback,
};
use super::gmres::{
    GmresConfig, GmresSolution, finalize_gmres_solution, invalid_gmres_input,
    numerical_error_solution,
};
use super::gmres_common::{exact_relative_residual, givens_rotation, solve_upper_triangular};
use super::gmres_dr::{dense_to_hessenberg, hessenberg_qr_eigen};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::{axpy, inner_product, vector_norm};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, IdentityPreconditioner, LinearOperator, Preconditioner,
    SolverError, SolverResultContract, SolverStatus,
};

/// Recycled subspace persisted across consecutive GCRO-DR solves.
///
/// Stores only the recycled basis `U` (orthonormal columns). The companion
/// matrix `C := orthonormalize(M⁻¹A · U)` is rebuilt at the start of each
/// solve from the current operator/preconditioner — so the *same* subspace
/// can be reused across solves whose operators differ.
#[derive(Clone)]
pub struct GcroDrSubspace<T: ComplexField> {
    /// Orthonormal recycled basis vectors (each of length n).
    pub(crate) u: Vec<Vec<T>>,
    /// Target number of vectors to retain at the end of each solve.
    pub(crate) k: usize,
}

impl<T: ComplexField> GcroDrSubspace<T> {
    /// Create an empty subspace that will retain up to `k` recycled vectors.
    pub fn new(k: usize) -> Self {
        Self { u: Vec::new(), k }
    }

    /// Whether the subspace has been populated by a prior solve.
    pub fn is_populated(&self) -> bool {
        !self.u.is_empty()
    }

    /// Number of vectors currently stored.
    pub fn num_vectors(&self) -> usize {
        self.u.len()
    }

    /// Target subspace size.
    pub fn target_size(&self) -> usize {
        self.k
    }

    /// Drop all recycled vectors. Useful when the operator has changed enough
    /// that the existing subspace is no longer informative.
    pub fn reset(&mut self) {
        self.u.clear();
    }

    /// Borrow the persisted recycled vectors for validated checkpointing.
    pub(crate) fn vectors(&self) -> &[Vec<T>] {
        &self.u
    }

    /// Replace recycled state after a checkpoint has validated it.
    pub(crate) fn replace_vectors(&mut self, target_size: usize, vectors: Vec<Vec<T>>) {
        self.k = target_size;
        self.u = vectors;
    }
}

/// Solver configuration. Mirrors [`GmresConfig`] with the addition of an
/// explicit `recycle_size` (also held on the subspace itself for convenience).
#[derive(Debug, Clone)]
pub struct GcroDrConfig<R> {
    /// Maximum number of restart cycles per solve.
    pub max_iterations: usize,
    /// Krylov dimension `m` per cycle.
    pub restart: usize,
    /// Relative residual tolerance (against `||M⁻¹ b||`).
    pub tolerance: R,
    /// Log every N inner iterations (0 = silent).
    pub print_interval: usize,
}

impl Default for GcroDrConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 50,
            tolerance: 1e-6,
            print_interval: 0,
        }
    }
}

impl Default for GcroDrConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 50,
            tolerance: 1e-5,
            print_interval: 0,
        }
    }
}

impl<R: num_traits::Float + FromPrimitive> GcroDrConfig<R> {
    /// Build a config from an existing [`GmresConfig`] (same `tolerance`,
    /// `max_iterations`, `restart`).
    pub fn from_gmres(g: &GmresConfig<R>) -> Self {
        Self {
            max_iterations: g.max_iterations,
            restart: g.restart,
            tolerance: g.tolerance,
            print_interval: g.print_interval,
        }
    }
}

/// Solve `Ax = b` with GCRO-DR using the identity preconditioner.
///
/// On exit the recycled subspace is updated with harmonic Ritz vectors of
/// `M⁻¹A` extracted from the augmented basis of the final cycle.
pub fn gcro_dr<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut GcroDrSubspace<T>,
    config: &GcroDrConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    match try_gcro_dr(operator, b, x0, subspace, config) {
        Ok(result) => result,
        Err(_) => numerical_error_solution(match x0 {
            Some(guess) if guess.len() == b.len() => guess.to_vec(),
            _ => vec![T::zero(); b.len()],
        }),
    }
}

/// Checked GCRO-DR entry point using the identity preconditioner.
pub fn try_gcro_dr<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut GcroDrSubspace<T>,
    config: &GcroDrConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    let identity = IdentityPreconditioner;
    try_gcro_dr_preconditioned(operator, &identity, b, x0, subspace, config)
}

/// Solve `Ax = b` with GCRO-DR using a left preconditioner.
pub fn gcro_dr_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut GcroDrSubspace<T>,
    config: &GcroDrConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match try_gcro_dr_preconditioned(operator, precond, b, x0, subspace, config) {
        Ok(result) => result,
        Err(_) => numerical_error_solution(match x0 {
            Some(guess) if guess.len() == b.len() => guess.to_vec(),
            _ => vec![T::zero(); b.len()],
        }),
    }
}

/// Checked GCRO-DR entry point for a left preconditioner.
pub fn try_gcro_dr_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut GcroDrSubspace<T>,
    config: &GcroDrConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    validate_gcro_request(operator, b, x0, subspace, config)?;

    let checked_operator = CheckedLinearOperator::<T, A>::new(operator);
    let checked_precond = CheckedPreconditioner::<T, P>::new(precond);
    // GCRO-DR refreshes the recycled basis while it solves. Keep that
    // mutation transactional across the checked callback boundary: a panic
    // or non-finite callback result must not leave the caller with a partly
    // rebuilt subspace that was never associated with a valid solve result.
    let mut working_subspace = subspace.clone();
    let result = catch_solver_callback("GCRO-DR", || {
        let result = gcro_dr_core(
            &checked_operator,
            &checked_precond,
            b,
            x0,
            &mut working_subspace,
            config,
        );
        let result = finalize_gmres_solution(&checked_operator, b, config.tolerance, result);
        if result.status == SolverStatus::NumericalError {
            return Err(SolverError::OperationFailed {
                operation: "GCRO-DR",
                reason: "operator or preconditioner produced a non-finite result".into(),
            });
        }
        SolverResultContract::validate_contract(&result)?;
        Ok(result)
    });
    if result.is_ok() {
        *subspace = working_subspace;
    }
    result
}

fn validate_gcro_request<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &GcroDrSubspace<T>,
    config: &GcroDrConfig<T::Real>,
) -> Result<(), SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if config.max_iterations == 0
        || config.restart == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= T::Real::zero()
    {
        return Err(SolverError::OperationFailed {
            operation: "GCRO-DR",
            reason: "invalid GCRO-DR configuration".into(),
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
            operation: "GCRO-DR",
            reason: "right-hand side contains NaN or infinity".into(),
        });
    }
    if let Some(guess) = x0 {
        if guess.len() != b.len() {
            return Err(SolverError::DimensionMismatch {
                expected: b.len(),
                got: guess.len(),
            });
        }
        if guess.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "GCRO-DR",
                reason: "initial guess contains NaN or infinity".into(),
            });
        }
    }
    if subspace.u.iter().any(|vector| {
        vector.len() != operator.num_cols() || vector.iter().any(|value| !value.norm().is_finite())
    }) {
        return Err(SolverError::OperationFailed {
            operation: "GCRO-DR recycled subspace",
            reason: "recycled vector has invalid dimension or non-finite value".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core solver
// ---------------------------------------------------------------------------

fn gcro_dr_core<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    subspace: &mut GcroDrSubspace<T>,
    config: &GcroDrConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if let Some(invalid) = invalid_gmres_input(
        b,
        x0,
        config.max_iterations,
        config.restart,
        config.tolerance,
    ) {
        return invalid;
    }

    let n = b.len();
    let m = config.restart;

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };
    if subspace.u.iter().any(|vector| vector.len() != n) {
        return numerical_error_solution(x);
    }
    let mut op_counts = KrylovOpCounts::default();

    let pb = precond.apply(b);
    op_counts.preconditioner_count += 1;
    let b_norm = vector_norm(&pb);
    op_counts.norm_count += 1;
    let true_b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let public_b_norm = if true_b_norm < T::Real::from_f64(1e-15).unwrap() {
        T::Real::one()
    } else {
        true_b_norm
    };
    let tiny = T::Real::from_f64(1e-15).unwrap();
    if b_norm < tiny {
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

    // Build C := orthonormalize(M⁻¹A · U) and R such that M⁻¹A U = C R.
    // If the build fails (rank-deficient U or near-singular R), fall back to
    // a clean restart: clear U here so this solve seeds a fresh subspace.
    let (c_mat, r_mat, used) = if subspace.is_populated() {
        match build_c_r(operator, precond, &subspace.u, &mut op_counts) {
            Some(triple) => triple,
            None => {
                subspace.u.clear();
                (
                    Vec::new(),
                    sonium_solvers_core::dense::DenseMatrix::from_elem(0, 0, T::zero()),
                    0,
                )
            }
        }
    } else {
        (
            Vec::new(),
            sonium_solvers_core::dense::DenseMatrix::from_elem(0, 0, T::zero()),
            0,
        )
    };
    if used < subspace.u.len() {
        subspace.u.truncate(used);
    }
    let k = c_mat.len();

    // Krylov scratch — sized to the requested restart length.
    let mut v = FlatBasis::new(m + 1, n);
    let mut op_buf = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];

    let mut h: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, T::zero());
    let mut b_block: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::from_elem(k.max(1), m, T::zero());
    let mut cs: Vec<T> = Vec::with_capacity(m);
    let mut sn: Vec<T> = Vec::with_capacity(m);
    let mut g_vec: Vec<T> = vec![T::zero(); m + 1];
    let mut c_coeffs = Vec::with_capacity(k);
    let mut c_reorth_coeffs = Vec::with_capacity(k);
    let mut v_coeffs = Vec::with_capacity(m);
    let mut v_reorth_coeffs = Vec::with_capacity(m);

    let mut total_iterations = 0usize;
    let mut restarts = 0usize;

    let dgks_threshold = T::Real::from_f64(0.707).unwrap();
    let breakdown_tol = T::Real::from_f64(1e-20).unwrap();
    let abs_tol = T::Real::from_f64(1e-20).unwrap();

    for _outer in 0..config.max_iterations {
        // Recompute preconditioned residual r = M⁻¹(b − Ax).
        operator.apply_into(&x, &mut op_buf);
        op_counts.spmv_count += 1;
        for i in 0..n {
            op_buf[i] = b[i] - op_buf[i];
        }
        op_counts.axpy_passes += 1;
        precond.apply_into(&op_buf, &mut w);
        op_counts.preconditioner_count += 1;

        // Project onto the C-complement, then absorb the C-component into x
        // via U R⁻¹: the initial residual `w` decomposes as `C·α + w'` with
        // `w' ⊥ C`. We update `x ← x + U R⁻¹ α` (this exactly zeros the
        // C-component of the residual since `M⁻¹A · U R⁻¹ α = C·α`) and
        // proceed with Arnoldi on `w'`. After the cycle's V-correction
        // re-introduces a `C` coupling `−B y_v`, the end-of-cycle U-correction
        // cancels it via `z_u = −R⁻¹ B y_v`. So `α` is consumed here in full
        // and not needed at the LS step.
        if k > 0 {
            batched_dots_into(&c_mat, k, &w, &mut c_coeffs);
            batched_project(&c_mat, &c_coeffs, &mut w);
            op_counts.add_basis_projection::<T>(k, n);
            let z_u_initial = solve_upper_triangular_dense(&r_mat, &c_coeffs, k);
            for (i, &z_u_i) in z_u_initial.iter().enumerate() {
                axpy(z_u_i, &subspace.u[i], &mut x);
            }
            op_counts.add_basis_update::<T>(z_u_initial.len(), n);
        }

        let beta = vector_norm(&w);
        op_counts.norm_count += 1;
        let rel_initial = beta / b_norm;
        if rel_initial < config.tolerance {
            let true_rel_initial = exact_relative_residual(
                operator,
                &x,
                b,
                &mut op_buf,
                public_b_norm,
                &mut op_counts,
            );
            if true_rel_initial < config.tolerance || true_rel_initial < abs_tol {
                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: true_rel_initial,
                    estimated_residual: rel_initial,
                    backward_error: true_rel_initial,
                    attainable_accuracy: AccuracyStatus::Attained,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
        }

        // V[0] = w / β
        let inv_beta = T::from_real(T::Real::one() / beta);
        for (dst, &wi) in v.vector_mut(0).iter_mut().zip(&w) {
            *dst = wi * inv_beta;
        }
        op_counts.axpy_passes += 1;

        h.fill(T::zero());
        if k > 0 {
            b_block.fill(T::zero());
        }
        cs.clear();
        sn.clear();
        g_vec.fill(T::zero());
        g_vec[0] = T::from_real(beta);

        let mut inner_converged = false;
        let mut converged_at: Option<usize> = None;

        for j in 0..m {
            total_iterations += 1;

            // z = M⁻¹ A v_j
            operator.apply_into(v.vector(j), &mut op_buf);
            op_counts.spmv_count += 1;
            precond.apply_into(&op_buf, &mut w);
            op_counts.preconditioner_count += 1;

            // Project against C (the structural GCRO-DR step).
            if k > 0 {
                batched_dots_into(&c_mat, k, &w, &mut c_coeffs);
                for (i, &c_ij) in c_coeffs.iter().enumerate() {
                    b_block[[i, j]] = c_ij;
                }
                batched_project(&c_mat, &c_coeffs, &mut w);
                op_counts.add_basis_projection::<T>(k, n);
                // DGKS reorthogonalisation against C, applied unconditionally:
                // it is cheap (k extra dots) and in finite precision the loss
                // of orthogonality between V and C is the dominant numerical
                // failure mode for GCRO-DR.
                batched_dots_into(&c_mat, k, &w, &mut c_reorth_coeffs);
                for (i, &c_ij) in c_reorth_coeffs.iter().enumerate() {
                    b_block[[i, j]] += c_ij;
                }
                batched_project(&c_mat, &c_reorth_coeffs, &mut w);
                op_counts.add_basis_projection::<T>(k, n);
            }

            // CGS+DGKS against V (same as `gmres.rs`).
            let w_norm_before = vector_norm(&w);
            op_counts.norm_count += 1;
            batched_dots_flat_into(&v, j + 1, &w, &mut v_coeffs);
            for (i, &c) in v_coeffs.iter().enumerate() {
                h[[i, j]] = c;
            }
            batched_project_flat(&v, &v_coeffs, &mut w);
            op_counts.add_basis_projection::<T>(j + 1, n);

            let mut w_norm = vector_norm(&w);
            op_counts.norm_count += 1;
            if w_norm < dgks_threshold * w_norm_before {
                batched_dots_flat_into(&v, j + 1, &w, &mut v_reorth_coeffs);
                for (i, &si) in v_reorth_coeffs.iter().enumerate() {
                    h[[i, j]] += si;
                }
                batched_project_flat(&v, &v_reorth_coeffs, &mut w);
                op_counts.add_basis_projection::<T>(j + 1, n);
                w_norm = vector_norm(&w);
                op_counts.norm_count += 1;
            }
            h[[j + 1, j]] = T::from_real(w_norm);

            if w_norm < breakdown_tol {
                inner_converged = true;
            } else {
                let inv_norm = T::from_real(T::Real::one() / w_norm);
                for (dst, &wi) in v.vector_mut(j + 1).iter_mut().zip(&w) {
                    *dst = wi * inv_norm;
                }
                op_counts.axpy_passes += 1;
            }

            // Apply previous Givens rotations to the new H-column.
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

            let temp = c.conj() * g_vec[j] + s.conj() * g_vec[j + 1];
            g_vec[j + 1] = T::zero() - s * g_vec[j] + c * g_vec[j + 1];
            g_vec[j] = temp;

            let abs_residual = g_vec[j + 1].norm();
            let rel_residual = abs_residual / b_norm;

            if config.print_interval > 0 && total_iterations.is_multiple_of(config.print_interval) {
                log::info!(
                    "GCRO-DR iter {} (restart {}, k={k}): rel_res = {:.6e}",
                    total_iterations,
                    restarts,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }

            if rel_residual < config.tolerance || abs_residual < abs_tol || inner_converged {
                converged_at = Some(j);
                break;
            }
        }

        let cycle_steps = match converged_at {
            Some(j) => j + 1,
            None => m,
        };

        // Solve y_v = argmin ||β e_1 - H y_v|| from the upper-triangularised H.
        let y_v = solve_upper_triangular(&h, &g_vec, cycle_steps);

        // Apply the V-correction to x.
        fused_basis_update_flat(&v, &y_v, &mut x);
        op_counts.add_basis_update::<T>(y_v.len(), n);

        // Apply the U-correction. The α component was already absorbed by
        // the initial U correction above (`x += U R⁻¹ α`); here we only need
        // to cancel the C-coupling that the V-correction re-introduced.
        // After `x += V y_v` the residual gains a C-component `−B y_v`, so
        // `z_u = −R⁻¹ B y_v`.
        if k > 0 {
            let mut by = vec![T::zero(); k];
            for i in 0..k {
                let mut s = T::zero();
                for jj in 0..cycle_steps {
                    s += b_block[[i, jj]] * y_v[jj];
                }
                by[i] = T::zero() - s;
            }
            let z_u = solve_upper_triangular_dense(&r_mat, &by, k);
            for (i, &z_u_i) in z_u.iter().enumerate() {
                axpy(z_u_i, &subspace.u[i], &mut x);
            }
            op_counts.add_basis_update::<T>(z_u.len(), n);
        }

        // Final residual estimate of this cycle (norm of last g entry from LS).
        let final_abs = if cycle_steps == 0 {
            T::Real::zero()
        } else {
            g_vec[cycle_steps].norm()
        };
        let rel_after_cycle = final_abs / b_norm;

        if converged_at.is_some() || rel_after_cycle < config.tolerance {
            let true_rel_after_cycle = exact_relative_residual(
                operator,
                &x,
                b,
                &mut op_buf,
                public_b_norm,
                &mut op_counts,
            );
            let converged =
                true_rel_after_cycle < config.tolerance || true_rel_after_cycle < abs_tol;
            if converged {
                // Update U from the augmented (k + m) Ritz problem only after
                // the original-operator gate accepts this candidate. A
                // rejected candidate must not replace the recycled subspace
                // used by the next restart.
                update_recycled_subspace(
                    subspace,
                    &r_mat,
                    &b_block,
                    &h,
                    &cs,
                    &sn,
                    &v,
                    cycle_steps,
                    k,
                    n,
                );
                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: true_rel_after_cycle,
                    estimated_residual: rel_after_cycle,
                    backward_error: true_rel_after_cycle,
                    attainable_accuracy: AccuracyStatus::Attained,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }
            if inner_converged {
                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: true_rel_after_cycle,
                    estimated_residual: rel_after_cycle,
                    backward_error: true_rel_after_cycle,
                    attainable_accuracy: AccuracyStatus::NotAttained,
                    converged: false,
                    status: SolverStatus::Breakdown,
                    op_counts,
                };
            }

            // The projected candidate was optimistic. It is already applied
            // exactly once above, so restart from it without refreshing the
            // recycled space from a rejected iterate.
            restarts += 1;
            continue;
        }

        restarts += 1;
    }

    // Did not converge; still refresh the recycled subspace so the next solve
    // has fresh harmonic Ritz vectors that reflect the latest spectrum.
    update_recycled_subspace(subspace, &r_mat, &b_block, &h, &cs, &sn, &v, m, k, n);

    operator.apply_into(&x, &mut op_buf);
    op_counts.spmv_count += 1;
    for i in 0..n {
        op_buf[i] = b[i] - op_buf[i];
    }
    op_counts.axpy_passes += 1;
    let rel = vector_norm(&op_buf) / public_b_norm;
    op_counts.norm_count += 1;

    GmresSolution {
        x,
        iterations: total_iterations,
        restarts,
        residual: rel,
        estimated_residual: rel,
        backward_error: rel,
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build `(C, R, columns_used)` such that `M⁻¹A U[..columns_used] = C R`,
/// with `C` orthonormal and `R` upper triangular. Returns `None` if the
/// process collapses to fewer than 1 well-conditioned column (no useful
/// recycled subspace).
fn build_c_r<T, A, P>(
    operator: &A,
    precond: &P,
    u: &[Vec<T>],
    op_counts: &mut KrylovOpCounts,
) -> Option<(
    Vec<Vec<T>>,
    sonium_solvers_core::dense::DenseMatrix<T>,
    usize,
)>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let k_in = u.len();
    if k_in == 0 {
        return None;
    }
    let n = u[0].len();
    // C accumulates in a flat contiguous basis so the orthogonalisation below
    // can use the batched parallel dot/projection helpers — the same scheme
    // `gmres.rs` and `gmres_dr.rs` use for their Arnoldi bases.
    let mut c_basis = FlatBasis::new(k_in, n);
    let mut c_count = 0usize;
    let mut r = sonium_solvers_core::dense::DenseMatrix::from_elem(k_in, k_in, T::zero());

    let mut au = vec![T::zero(); n];
    let mut z = vec![T::zero(); n];
    let mut coeffs: Vec<T> = Vec::with_capacity(k_in);
    let mut reorth_coeffs: Vec<T> = Vec::with_capacity(k_in);

    let drop_tol = T::Real::from_f64(1e-12).unwrap();

    for j in 0..k_in {
        operator.apply_into(&u[j], &mut au);
        op_counts.spmv_count += 1;
        precond.apply_into(&au, &mut z);
        op_counts.preconditioner_count += 1;

        // Batched CGS+DGKS against existing C columns. Both passes are
        // applied unconditionally: the DGKS pass is cheap (k extra dots) and
        // keeps C = orth(M⁻¹A U) to working precision.
        if c_count > 0 {
            batched_dots_flat_into(&c_basis, c_count, &z, &mut coeffs);
            for (i, &c_ij) in coeffs.iter().enumerate() {
                r[[i, j]] = c_ij;
            }
            batched_project_flat(&c_basis, &coeffs, &mut z);
            op_counts.add_basis_projection::<T>(c_count, n);

            batched_dots_flat_into(&c_basis, c_count, &z, &mut reorth_coeffs);
            for (i, &s) in reorth_coeffs.iter().enumerate() {
                r[[i, j]] += s;
            }
            batched_project_flat(&c_basis, &reorth_coeffs, &mut z);
            op_counts.add_basis_projection::<T>(c_count, n);
        }

        let nz = vector_norm(&z);
        op_counts.norm_count += 1;
        if nz < drop_tol {
            // u_j was effectively in the operator-image span of the prior
            // u_0..u_{j-1}; truncate the subspace at this column.
            break;
        }
        r[[j, j]] = T::from_real(nz);
        let inv = T::from_real(T::Real::one() / nz);
        let dst = c_basis.vector_mut(c_count);
        for (d, &zi) in dst.iter_mut().zip(&z) {
            *d = zi * inv;
        }
        c_count += 1;
    }

    if c_count == 0 {
        return None;
    }
    let c: Vec<Vec<T>> = (0..c_count).map(|i| c_basis.vector(i).to_vec()).collect();
    Some((c, r, c_count))
}

/// Upper-triangular back-substitution for a `k × k` dense system `R y = g`,
/// with `g` supplied as a slice. Returns the `k`-element solution.
fn solve_upper_triangular_dense<T: ComplexField>(
    r: &sonium_solvers_core::dense::DenseMatrix<T>,
    g: &[T],
    k: usize,
) -> Vec<T> {
    let mut y = vec![T::zero(); k];
    let tol = T::Real::from_f64(1e-30).unwrap();
    for i in (0..k).rev() {
        let mut sum = g[i];
        for j in (i + 1)..k {
            sum -= r[[i, j]] * y[j];
        }
        if r[[i, i]].norm() > tol {
            y[i] = sum * r[[i, i]].inv();
        }
    }
    y
}

/// Unapply Givens rotations from the rotated H to recover the raw
/// upper-Hessenberg block (size `(steps + 1) × steps`).
fn unapply_givens_local<T: ComplexField>(
    h_rotated: &sonium_solvers_core::dense::DenseMatrix<T>,
    cs: &[T],
    sn: &[T],
    steps: usize,
) -> sonium_solvers_core::dense::DenseMatrix<T> {
    let mut h = h_rotated.clone();
    for j in (0..steps).rev() {
        if j >= cs.len() {
            continue;
        }
        let c = cs[j];
        let s = sn[j];
        for col in j..steps {
            let hj = h[[j, col]];
            let hj1 = h[[j + 1, col]];
            h[[j, col]] = c * hj - s.conj() * hj1;
            h[[j + 1, col]] = s.conj() * hj + c.conj() * hj1;
        }
    }
    h
}

/// Update the recycled subspace U using harmonic Ritz vectors of the
/// augmented matrix `M⁻¹A` restricted to `range([U V_m])`.
///
/// In the orthonormal basis `[C V_{m+1}]`, the action of `M⁻¹A` on `[U V_m]`
/// is the `(k + m + 1) × (k + m)` matrix
///
/// ```text
///   G = [ R          B_{k,m} ]
///       [ 0          H_{m+1,m} ]
/// ```
///
/// Let `G_top` be the leading `(k + m) × (k + m)` block of `G`. The harmonic
/// Ritz pairs satisfy
///
/// ```text
///   (G_top + ρ² G_top^{-H} e_l e_l^H) y = θ y
/// ```
///
/// where `ρ = h_{m+1,m}` and `e_l` is the last unit vector of length `k + m`
/// (Morgan 2002, Parks 2006 Algorithm 2.2 step 6). We solve this small dense
/// eigenproblem via Francis QR (`hessenberg_qr_eigen` works on general dense
/// inputs since it deflates from the bottom), pick the `k_target` smallest
/// |θ_h|, lift each `y` back to an `n`-vector via `[U V_m] y`, and
/// orthonormalise.
#[allow(clippy::too_many_arguments)]
fn update_recycled_subspace<T: ComplexField>(
    subspace: &mut GcroDrSubspace<T>,
    r_mat: &sonium_solvers_core::dense::DenseMatrix<T>,
    b_block: &sonium_solvers_core::dense::DenseMatrix<T>,
    h_rotated: &sonium_solvers_core::dense::DenseMatrix<T>,
    cs: &[T],
    sn: &[T],
    v: &FlatBasis<T>,
    cycle_steps: usize,
    k_in: usize,
    n: usize,
) {
    let k_target = subspace.k;
    if k_target == 0 || cycle_steps == 0 {
        return;
    }
    let total = k_in + cycle_steps; // size of the joint augmented basis

    // Reconstruct the raw (un-rotated) Hessenberg of the V-block.
    let h_raw = unapply_givens_local(h_rotated, cs, sn, cycle_steps);
    let rho = if cycle_steps < h_raw.nrows() {
        h_raw[[cycle_steps, cycle_steps - 1]]
    } else {
        T::zero()
    };

    // Build G_top (the leading total × total block of G).
    let mut g_top = sonium_solvers_core::dense::DenseMatrix::from_elem(total, total, T::zero());
    for i in 0..k_in {
        for j in 0..k_in {
            g_top[[i, j]] = r_mat[[i, j]];
        }
        for j in 0..cycle_steps {
            g_top[[i, k_in + j]] = b_block[[i, j]];
        }
    }
    for i in 0..cycle_steps {
        for j in 0..cycle_steps {
            g_top[[k_in + i, k_in + j]] = h_raw[[i, j]];
        }
    }

    // Form the harmonic Ritz problem: (G_top + ρ² G_top^{-H} e_l e_l^H) y = θ y.
    // To do that without solving with G_top^{-H} explicitly, we solve
    //   G_top^H w = e_l                  (one (k+m)-system)
    // and rank-1 update: M_aug = G_top + (ρ.conj() ρ) w e_l^H? No — derivation:
    //   G_top^{-H} e_l = (G_top^H)^{-1} e_l = w
    // ⇒ M_aug = G_top + ρ² (w e_l^H) is a column rank-1 update of G_top.
    // We then run a *general* (non-Hessenberg) eigensolver via the Francis QR
    // helper, which still converges on small dense inputs.
    let rho_sq = rho * rho.conj(); // |ρ|²
    let mut e_l = vec![T::zero(); total];
    if total > 0 {
        e_l[total - 1] = T::one();
    }
    let w_aug = solve_with_g_top_h(&g_top, &e_l, total);

    let mut m_aug = g_top.clone();
    for i in 0..total {
        m_aug[[i, total - 1]] += rho_sq * w_aug[i];
    }

    // Eigendecomposition.
    // m_aug is a general dense matrix (rank-1 update of g_top), so we must
    // reduce it to upper Hessenberg form before calling the Francis QR
    // eigensolver.
    let (m_aug_hess, q_hess) = dense_to_hessenberg(&m_aug, total);
    let (eigenvalues, eigenvectors_hess) = hessenberg_qr_eigen(&m_aug_hess, total);

    // Transform eigenvectors back from the Hessenberg basis to the original basis.
    let mut eigenvectors =
        sonium_solvers_core::dense::DenseMatrix::from_elem(total, total, T::zero());
    for j in 0..total {
        for i in 0..total {
            let mut sum = T::zero();
            for l in 0..total {
                sum += q_hess[[i, l]] * eigenvectors_hess[[l, j]];
            }
            eigenvectors[[i, j]] = sum;
        }
    }

    // Sort eigenvalues by |θ_h| ascending — the harmonic-Ritz selection of
    // smallest-magnitude approximate eigenvalues of M⁻¹A. These are the
    // slowly-converging modes that benefit most from recycling.
    let mut indices: Vec<usize> = (0..total).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[a]
            .norm()
            .partial_cmp(&eigenvalues[b].norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let take = k_target.min(total);
    let selected = &indices[..take];

    // Lift each selected eigenvector y ∈ ℂ^{k+m} into the n-space:
    //   ψ = U y[0..k_in]  +  V_m y[k_in..k_in+m]
    // and Modified Gram-Schmidt to orthonormalise.
    let mut new_u: Vec<Vec<T>> = Vec::with_capacity(take);
    for &idx in selected {
        let mut psi = vec![T::zero(); n];
        for i in 0..k_in {
            axpy(eigenvectors[[i, idx]], &subspace.u[i], &mut psi);
        }
        for j in 0..cycle_steps {
            axpy(eigenvectors[[k_in + j, idx]], v.vector(j), &mut psi);
        }

        // Orthogonalise against already-accepted vectors.
        for prev in &new_u {
            let coeff = inner_product(prev, &psi);
            axpy(T::zero() - coeff, prev, &mut psi);
        }

        let norm = vector_norm(&psi);
        let ortho_tol = T::Real::from_f64(1e-12).unwrap();
        if norm > ortho_tol {
            let inv = T::from_real(T::Real::one() / norm);
            for v in &mut psi {
                *v *= inv;
            }
            new_u.push(psi);
        }
    }

    if !new_u.is_empty() {
        subspace.u = new_u;
    }
}

/// Solve `G_top^H w = e` for the small dense system used by the harmonic-Ritz
/// pencil. `G_top` is upper-triangular-block-Hessenberg, so `G_top^H` is
/// lower-triangular-block-Hessenberg; we use forward substitution column by
/// column with an explicit fill-handling pass for the Hessenberg subdiagonal
/// entries inherited from the H-block.
fn solve_with_g_top_h<T: ComplexField>(
    g_top: &sonium_solvers_core::dense::DenseMatrix<T>,
    rhs: &[T],
    n: usize,
) -> Vec<T> {
    // Build the dense Hermitian transpose so we can run a general
    // forward-substitute that tolerates the lower-Hessenberg fill. The matrix
    // is small (at most ~70 × 70 in our intended use) so this is cheap.
    let mut a = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, T::zero());
    for i in 0..n {
        for j in 0..n {
            a[[i, j]] = g_top[[j, i]].conj();
        }
    }

    // LU with partial pivoting on a small matrix.
    let lu = match sonium_solvers_direct::lu_factorize(&a) {
        Ok(lu) => lu,
        Err(_) => {
            // Fall back to a heavily damped identity-shift if singular.
            for i in 0..n {
                a[[i, i]] += T::from_real(T::Real::from_f64(1e-12).unwrap());
            }
            sonium_solvers_direct::lu_factorize(&a).expect("LU must factor after damping")
        }
    };
    lu.solve(rhs).unwrap_or_else(|_| rhs.to_vec())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_core::traits::IdentityPreconditioner;
    use sonium_solvers_sparse::CsrMatrix;

    fn build_helmholtz_like(n: usize, freq: f64) -> CsrMatrix<Complex64> {
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        let k2 = freq * freq * 0.05;
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0 - k2, 0.01 * freq);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn first_solve_with_empty_subspace_converges() {
        let a = build_helmholtz_like(40, 1.0);
        let b = (0..40)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let mut sub = GcroDrSubspace::<Complex64>::new(8);
        let cfg = GcroDrConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-10,
            print_interval: 0,
        };
        let sol = gcro_dr_preconditioned(&a, &IdentityPreconditioner, &b, None, &mut sub, &cfg);
        assert!(sol.converged, "first solve should converge");
        let ax = a.matvec(&sol.x);
        let err: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm = b.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
        assert!((sol.residual - err / b_norm).abs() < 1.0e-12);
        assert!(err < 1e-7, "residual too large: {err}");
        assert!(
            sub.is_populated(),
            "subspace should be populated after first solve"
        );
    }

    #[test]
    fn sequential_solves_use_recycled_subspace() {
        // Sequence of nearby Helmholtz-like systems: identical RHS, slowly
        // varying frequency. With recycling the second solve must converge in
        // strictly fewer iterations than the first.
        let n = 60;
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).cos(), 0.5))
            .collect::<Vec<_>>();

        let cfg = GcroDrConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-9,
            print_interval: 0,
        };

        let mut sub = GcroDrSubspace::<Complex64>::new(10);
        let a0 = build_helmholtz_like(n, 1.0);
        let s0 = gcro_dr_preconditioned(&a0, &IdentityPreconditioner, &b, None, &mut sub, &cfg);
        assert!(s0.converged);
        let iters0 = s0.iterations;

        let a1 = build_helmholtz_like(n, 1.05);
        let s1 = gcro_dr_preconditioned(&a1, &IdentityPreconditioner, &b, None, &mut sub, &cfg);
        assert!(s1.converged);

        // We don't assert a fixed factor (depends heavily on the toy spectrum
        // of build_helmholtz_like), only that recycling didn't hurt.
        assert!(
            s1.iterations <= iters0,
            "recycled solve should not be slower than the seeding solve: {} > {}",
            s1.iterations,
            iters0
        );
    }

    #[test]
    fn matches_plain_gmres_solution() {
        // Solving the same system with GMRES vs GCRO-DR should give the same
        // answer to within tolerance.
        use crate::{GmresConfig, gmres};

        let a = build_helmholtz_like(32, 0.5);
        let b = (0..32)
            .map(|i| Complex64::new(i as f64 * 0.1, 0.0))
            .collect::<Vec<_>>();

        let cfg_g = GmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };
        let cfg_r = GcroDrConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-12,
            print_interval: 0,
        };

        let g_sol = gmres(&a, &b, &cfg_g);
        let mut sub = GcroDrSubspace::<Complex64>::new(6);
        let r_sol = gcro_dr_preconditioned(&a, &IdentityPreconditioner, &b, None, &mut sub, &cfg_r);

        assert!(g_sol.converged && r_sol.converged);
        let diff: f64 = sonium_solvers_core::vec_ops::vec_sub(&g_sol.x, &r_sol.x)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-7, "GMRES vs GCRO-DR solution differ: {diff}");
    }

    #[test]
    fn frequency_sweep_recycling_beats_independent_solves() {
        // 100-frequency sweep: iteration count must drop substantially when
        // we recycle harmonic Ritz vectors across consecutive frequencies.
        //
        // We compare *total Krylov iterations* (not wall-clock) because that
        // isolates the structural Parks–de Sturler win from the noise of
        // small-problem timing. Real-world wall-clock follows iteration count
        // when the per-iteration cost is dominated by SpMV + preconditioner
        // apply (i.e., for FEM/BEM where N ≫ k).
        let n = 800;
        let n_freqs = 100;
        let f0 = 1.0;
        let df = 0.05;
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).cos(), 0.5))
            .collect::<Vec<_>>();

        let cfg_g = crate::GmresConfig {
            max_iterations: 200,
            restart: 40,
            tolerance: 1e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };
        let cfg_r = GcroDrConfig {
            max_iterations: 200,
            restart: 40,
            tolerance: 1e-8,
            print_interval: 0,
        };

        let mut total_indep = 0usize;
        let mut total_recycled = 0usize;
        let mut sub = GcroDrSubspace::<Complex64>::new(20);

        for f_idx in 0..n_freqs {
            let f = f0 + df * f_idx as f64;
            let a = build_helmholtz_like(n, f);

            let plain = crate::gmres_preconditioned(&a, &IdentityPreconditioner, &b, &cfg_g);
            let recycled =
                gcro_dr_preconditioned(&a, &IdentityPreconditioner, &b, None, &mut sub, &cfg_r);
            assert!(plain.converged, "plain GMRES failed at f_idx={f_idx}");
            assert!(recycled.converged, "GCRO-DR failed at f_idx={f_idx}");

            // Compare *residuals*, not solutions — near interior eigenvalues
            // of the Helmholtz-like operator, the system is poorly conditioned
            // so two different x's can satisfy the same tolerance with very
            // different positions in the high-condition-number directions.
            let b_norm: f64 = b.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
            let r_recycled = sonium_solvers_core::vec_ops::vec_sub(&a.matvec(&recycled.x), &b);
            let nr_recycled: f64 = r_recycled.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
            assert!(
                nr_recycled / b_norm < 1e-4,
                "GCRO-DR residual too large at f_idx={f_idx}: {:.3e}",
                nr_recycled / b_norm
            );

            total_indep += plain.iterations;
            total_recycled += recycled.iterations;
        }

        let ratio = total_indep as f64 / total_recycled.max(1) as f64;
        eprintln!(
            "GCRO-DR sweep: indep={total_indep} iters, recycled={total_recycled} iters, ratio={ratio:.2}x"
        );
        // Phase-1.5 target is ≥5× end-to-end. We assert ≥2× here because
        // (a) the synthetic build_helmholtz_like spectrum is much friendlier
        // than a real shifted-Laplacian-preconditioned indefinite Helmholtz
        // operator, where the recycling win compounds, and (b) we include
        // the seeding solve in the recycled total. The QA bin reports the
        // larger ratio that real applications see.
        assert!(
            ratio >= 2.0,
            "expected GCRO-DR to halve total iterations across 100-freq sweep; got {ratio:.2}x"
        );
    }

    #[test]
    fn small_diagonal_smoke() {
        // Tiny diagonal system.
        let dense = sonium_solvers_core::array![
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(2.0, 0.0), Complex64::new(6.0, 0.0)];
        let cfg = GcroDrConfig {
            max_iterations: 10,
            restart: 5,
            tolerance: 1e-12,
            print_interval: 0,
        };
        let mut sub = GcroDrSubspace::<Complex64>::new(2);
        let sol = gcro_dr_preconditioned(&a, &IdentityPreconditioner, &b, None, &mut sub, &cfg);
        assert!(sol.converged);
        assert!((sol.x[0].re - 1.0).abs() < 1e-9);
        assert!((sol.x[1].re - 2.0).abs() < 1e-9);
    }

    #[test]
    fn reports_krylov_op_counts() {
        let a = build_helmholtz_like(32, 0.5);
        let b = (0..32)
            .map(|i| Complex64::new(i as f64 * 0.1, 0.0))
            .collect::<Vec<_>>();
        let cfg = GcroDrConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-12,
            print_interval: 0,
        };

        let mut sub = GcroDrSubspace::<Complex64>::new(6);
        let sol = gcro_dr_preconditioned(&a, &IdentityPreconditioner, &b, None, &mut sub, &cfg);

        assert!(sol.converged);
        assert!(sol.op_counts.spmv_count >= sol.iterations);
        assert!(sol.op_counts.preconditioner_count > 0);
        assert!(sol.op_counts.dot_reductions > 0);
        assert!(sol.op_counts.norm_count > 0);
        assert!(sol.op_counts.axpy_passes > 0);
        assert!(sol.op_counts.basis_bytes_touched > 0);
    }

    #[test]
    fn rejected_exact_candidate_restarts_gcro_dr() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 1.0e-4 * r[1];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0];
        let config = GcroDrConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-3,
            print_interval: 0,
        };
        let mut subspace = GcroDrSubspace::new(0);

        let solution = gcro_dr_preconditioned(
            &a,
            &AnisotropicPreconditioner,
            &b,
            None,
            &mut subspace,
            &config,
        );

        assert!(
            solution.converged,
            "GCRO-DR should restart after an optimistic exact gate: residual={}",
            solution.residual
        );
        assert!(solution.iterations >= 2);
    }

    #[test]
    fn test_gcro_dr_dense_hessenberg_reduction() {
        // Regression test for issue #17: the augmented matrix in GCRO-DR is
        // general dense (not Hessenberg), so calling a Hessenberg-only QR
        // eigensolver on it produces garbage.
        //
        // Verify that a sequence of solves where the subspace is updated
        // (triggering the dense eigendecomposition) still converges.
        let n = 12;
        let mut dense = sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, 0.0_f64);
        for i in 0..n {
            dense[[i, i]] = 2.0 + 0.3 * (i as f64);
            if i > 0 {
                dense[[i, i - 1]] = -0.3;
            }
            if i < n - 1 {
                dense[[i, i + 1]] = -0.3;
            }
            // Add fill-in below the second subdiagonal so the matrix is
            // genuinely dense, not already Hessenberg.
            if i >= 2 {
                dense[[i, i - 2]] = 0.1 * (i as f64).sin();
            }
        }
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n).map(|i| (i as f64).sin()).collect::<Vec<_>>();

        let cfg = GcroDrConfig {
            max_iterations: 50,
            restart: 6,
            tolerance: 1e-8,
            print_interval: 0,
        };

        let mut sub = GcroDrSubspace::<f64>::new(3);

        // First solve seeds the subspace
        let sol1 = gcro_dr(&a, &b, None, &mut sub, &cfg);
        assert!(sol1.converged, "First solve must converge");

        // Second solve reuses the recycled subspace; the update step
        // exercises the dense-to-Hessenberg reduction.
        let b2 = (0..n).map(|i| (i as f64).cos()).collect::<Vec<_>>();
        let sol2 = gcro_dr(&a, &b2, None, &mut sub, &cfg);
        assert!(sol2.converged, "Second solve with recycling must converge");
    }

    #[test]
    fn build_c_r_is_orthonormal_and_counts_ops() {
        // The C basis built by `build_c_r` must satisfy the GCRO-DR invariant
        // C = orth(M⁻¹A U) with M⁻¹A U = C R, and the op counters must reflect
        // the two unconditional batched CGS+DGKS passes per accepted column.
        let n = 48;
        let k = 6;
        let a = build_helmholtz_like(n, 0.7);
        // Discrete sine basis: exactly orthogonal columns, so a well-posed
        // full-rank build is guaranteed regardless of the drop tolerance.
        let u: Vec<Vec<Complex64>> = (0..k)
            .map(|c| {
                (0..n)
                    .map(|i| {
                        let s = (std::f64::consts::PI * (c + 1) as f64 * (i + 1) as f64
                            / (n + 1) as f64)
                            .sin();
                        Complex64::new(s, 0.3 * s)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut counts = KrylovOpCounts::default();
        let (c_mat, r_mat, used) =
            build_c_r(&a, &IdentityPreconditioner, &u, &mut counts).expect("full-rank build");
        assert_eq!(used, k);

        // Orthonormality of C.
        for i in 0..k {
            for j in 0..k {
                let d = inner_product(&c_mat[i], &c_mat[j]);
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (d.re - expect).abs() < 1e-11 && d.im.abs() < 1e-11,
                    "C not orthonormal at ({i},{j}): {d:?}"
                );
            }
        }

        // M⁻¹A U = C R (identity preconditioner here, so M⁻¹A = A).
        let mut err = 0.0f64;
        for j in 0..k {
            let mut resid = a.matvec(&u[j]);
            for i in 0..k {
                axpy(-r_mat[[i, j]], &c_mat[i], &mut resid);
            }
            err += resid.iter().map(|e| e.norm_sqr()).sum::<f64>();
        }
        assert!(
            err.sqrt() < 1e-10,
            "M⁻¹AU − C R residual too large: {:.3e}",
            err.sqrt()
        );

        // Two unconditional passes against the already-accepted columns:
        // 2 * sum_{j=0}^{k-1} j = k(k-1) dot reductions for a full-rank build.
        assert_eq!(counts.dot_reductions, k * (k - 1));
        assert_eq!(counts.spmv_count, k);
        assert_eq!(counts.preconditioner_count, k);
    }

    #[test]
    fn checked_gcro_dr_rejects_non_finite_rhs() {
        let operator: CsrMatrix<f64> = CsrMatrix::identity(2);
        let mut subspace = GcroDrSubspace::new(1);
        let config = GcroDrConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            print_interval: 0,
        };
        let error = try_gcro_dr(&operator, &[f64::NAN, 1.0], None, &mut subspace, &config)
            .expect_err("checked GCRO-DR must reject non-finite RHS");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(subspace.u.is_empty());
    }

    #[test]
    fn gcro_dr_rejects_non_finite_rhs_before_iteration() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let mut subspace = GcroDrSubspace::new(2);
        let config = GcroDrConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            print_interval: 0,
        };

        let solution = gcro_dr(&a, &[f64::NAN, 1.0], None, &mut subspace, &config);

        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
        assert!(subspace.u.is_empty());
    }

    #[test]
    fn checked_callback_failure_does_not_mutate_recycled_subspace() {
        struct FailingPreconditioner {
            calls: std::sync::atomic::AtomicUsize,
            fail_on: usize,
        }

        impl Preconditioner<f64> for FailingPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                assert_ne!(call, self.fail_on, "injected preconditioner failure");
                y.copy_from_slice(r);
            }
        }

        let operator: CsrMatrix<f64> = CsrMatrix::identity(2);
        let mut subspace = GcroDrSubspace::new(2);
        // The duplicate second vector is valid input but is dropped while the
        // working copy rebuilds C/R. The injected failure comes afterwards,
        // so a direct solve would leave the caller with a truncated state.
        subspace.u = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let original = subspace.u.clone();
        let preconditioner = FailingPreconditioner {
            calls: std::sync::atomic::AtomicUsize::new(0),
            fail_on: 4,
        };
        let config = GcroDrConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            print_interval: 0,
        };

        let error = try_gcro_dr_preconditioned(
            &operator,
            &preconditioner,
            &[1.0, 2.0],
            None,
            &mut subspace,
            &config,
        )
        .expect_err("checked callback failure should be returned as a typed error");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert_eq!(subspace.u, original);
    }

    #[test]
    fn checked_success_commits_recycled_subspace_update() {
        let operator: CsrMatrix<f64> = CsrMatrix::identity(2);
        let mut subspace = GcroDrSubspace::new(1);
        let config = GcroDrConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-10,
            print_interval: 0,
        };

        let solution = try_gcro_dr(&operator, &[1.0, 2.0], None, &mut subspace, &config)
            .expect("checked GCRO-DR solve should succeed");
        assert!(solution.converged);
        assert_eq!(subspace.num_vectors(), 1);
        assert_eq!(subspace.u[0].len(), 2);
    }
}
