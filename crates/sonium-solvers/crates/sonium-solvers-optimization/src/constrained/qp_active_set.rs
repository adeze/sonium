//! Dense active-set QP solver (Goldfarb–Idnani dual method).
//!
//! Solves the convex quadratic program
//!
//! ```text
//! min  ½ dᵀ H d + gᵀ d
//! s.t. a_iᵀ d = b_i   (equalities)
//!      a_iᵀ d ≥ b_i   (inequalities)
//! ```
//!
//! The dual method starts from the unconstrained minimiser `d = -H⁻¹ g`
//! (optimal for the empty working set) and enforces the most violated
//! constraint at a time, moving along directions that maintain stationarity
//! with non-negative multipliers for the active inequalities. An active
//! inequality is dropped from the working set when its multiplier reaches
//! zero before the newly enforced constraint becomes feasible.
//!
//! All linear algebra goes through the Gill–Murray–Wright modified Cholesky
//! factorisation in [`crate::util::dense_linalg`], so an
//! indefinite Hessian is regularised (the correction is zero when `H` is
//! already positive definite) and every solve is positive-definite.
//!
//! [`solve_linearized_qp`] assembles the SQP subproblem
//!
//! ```text
//! min  ½ dᵀ B d + gᵀ d
//! s.t. J_eq d = -c_eq(x),  J_ineq d ≥ -c_ineq(x),  bounds on x + d
//! ```
//!
//! from a [`Constraint`] list, falling back to central finite differences
//! ([`crate::util::finite_diff::jacobian_central_auto`]) for
//! constraints without a user-supplied Jacobian.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{
    Bound, Constraint, ConstraintKind, KktMultipliers, OptimError, RealScalar, validate_bounds,
};
use crate::util::dense_linalg;
use crate::util::finite_diff::jacobian_central_auto;
use sonium_solvers_core::traits::{
    AccuracyStatus, SolverError, SolverResultContract, SolverStatus,
};

/// Sense of a linear QP constraint row `aᵀ d (sense) b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpSense {
    /// `aᵀ d = b`
    Equality,
    /// `aᵀ d ≥ b`
    Inequality,
}

/// Exit status of [`solve_qp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// KKT point of the QP found.
    Optimal,
    /// Iteration cap reached (degenerate-cycling guard); the returned
    /// direction still satisfies the current working set.
    MaxIterations,
    /// A violated constraint is linearly dependent on the working set and
    /// cannot be satisfied: the linearised constraints are inconsistent.
    Infeasible,
}

/// Solution of a dense QP.
#[derive(Debug, Clone)]
pub struct QpSolution<T: RealScalar> {
    /// Primal solution `d`.
    pub d: Vec<T>,
    /// Lagrange multipliers per constraint row, with the sign convention
    /// `H d + g = Σ λ_i a_i` (zero for inactive inequalities, `λ ≥ 0` for
    /// active inequalities, free sign for equalities).
    pub multipliers: Vec<T>,
    /// Indices of the constraints in the final working set.
    #[allow(dead_code)] // exercised by unit tests; part of the API surface
    pub active_set: Vec<usize>,
    /// Exit status.
    pub status: QpStatus,
}

impl<T: RealScalar> SolverResultContract for QpSolution<T> {
    fn solver_status(&self) -> SolverStatus {
        match self.status {
            QpStatus::Optimal => SolverStatus::Converged,
            QpStatus::MaxIterations => SolverStatus::MaxIterationsReached,
            // The shared vocabulary has no separate infeasibility variant.
            // Match OptimStatus::RestorationFailed, which uses Stagnated for
            // a finite optimization result that could not establish feasibility.
            QpStatus::Infeasible => SolverStatus::Stagnated,
        }
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        match self.status {
            QpStatus::Optimal => AccuracyStatus::Attained,
            QpStatus::MaxIterations | QpStatus::Infeasible => AccuracyStatus::NotAttained,
        }
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        let contract_error = |reason: String| SolverError::OperationFailed {
            operation: "qp_result_contract",
            reason,
        };

        if let Some(index) = self.d.iter().position(|value| !value.is_finite()) {
            return Err(contract_error(format!(
                "primal solution contains non-finite value at index {index}"
            )));
        }
        if let Some(index) = self.multipliers.iter().position(|value| !value.is_finite()) {
            return Err(contract_error(format!(
                "multiplier contains non-finite value at index {index}"
            )));
        }

        let mut seen = vec![false; self.multipliers.len()];
        for &index in &self.active_set {
            let Some(was_seen) = seen.get_mut(index) else {
                return Err(contract_error(format!(
                    "active-set index {index} exceeds multiplier vector length {}",
                    self.multipliers.len()
                )));
            };
            if *was_seen {
                return Err(contract_error(format!(
                    "active-set index {index} appears more than once"
                )));
            }
            *was_seen = true;
        }

        Ok(())
    }
}

type CheckedQpAssembly<T> = (
    sonium_solvers_core::dense::DenseMatrix<T>,
    Vec<T>,
    Vec<QpSense>,
);

/// Solve a dense QP with the Goldfarb–Idnani dual active-set method.
///
/// * `h` — symmetric Hessian, `n × n` (indefinite input is regularised via
///   the GMW modified Cholesky).
/// * `g` — linear term, length `n`.
/// * `a` — constraint matrix, `m × n`, one row per constraint.
/// * `b` — right-hand side, length `m`.
/// * `senses` — sense of each constraint row, length `m`.
pub fn solve_qp<T: RealScalar>(
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    g: &[T],
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
    b: &[T],
    senses: &[QpSense],
) -> QpSolution<T> {
    let n = g.len();
    let m = senses.len();
    debug_assert_eq!(a.nrows(), m);
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.len(), m);

    // Factor H. The GMW modified Cholesky only adds a minimal diagonal
    // correction, so for an indefinite H the factorised matrix can still be
    // near-singular and solves against it can be meaningless (huge,
    // cancellation-ridden). If the correction is non-trivial, escalate an
    // explicit diagonal shift until H + shift·I is numerically positive
    // definite, which keeps every subsequent solve well conditioned (this
    // is the QP analogue of Nocedal & Wright's "Cholesky with added
    // multiple of the identity"). For a positive-definite H — e.g. a
    // damped-BFGS Hessian — the correction is ~eps and the loop exits
    // after the first factorisation.
    let a_norm = (0..n)
        .flat_map(|i| (0..n).map(move |j| h[[i, j]].abs()))
        .fold(T::zero(), T::max);
    let scale = T::max(a_norm, T::one());
    let e_tol = T::sqrt_epsilon() * scale;
    let ten = T::from_f64(10.0).unwrap();
    let mut shift = T::zero();
    debug_assert_eq!(h.nrows(), n);
    debug_assert_eq!(h.ncols(), n);
    // `h` / `hs` are n x n by the debug-checked caller contract above.
    let mut hl = dense_linalg::modified_cholesky(h)
        .expect("QP Hessian is square by contract")
        .0;
    for attempt in 0..=12 {
        let (l, e) = if shift == T::zero() {
            dense_linalg::modified_cholesky(h).expect("QP Hessian is square by contract")
        } else {
            let mut hs = h.clone();
            for i in 0..n {
                hs[[i, i]] += shift;
            }
            dense_linalg::modified_cholesky(&hs).expect("shifted Hessian stays square")
        };
        let e_max = e.iter().map(|&ei| ei.abs()).fold(T::zero(), T::max);
        hl = l;
        if e_max <= e_tol || attempt == 12 {
            break;
        }
        shift = if shift == T::zero() {
            e_tol
        } else {
            shift * ten
        };
    }
    let solve_h = |rhs: &[T]| -> Vec<T> {
        let y = dense_linalg::solve_lower_triangular(&hl, rhs);
        dense_linalg::solve_lower_triangular_transpose(&hl, &y)
    };

    // Unconstrained minimiser: H d = -g.
    let g_neg: Vec<T> = g.iter().map(|&gi| -gi).collect();
    let mut x = solve_h(&g_neg);

    let mut active: Vec<usize> = Vec::new();
    let mut rows: Vec<Vec<T>> = Vec::new(); // oriented normals, sign * a_row
    let mut signs: Vec<T> = Vec::new();
    let mut mult: Vec<T> = Vec::new();

    let tol = T::sqrt_epsilon();
    // The dual method adds one constraint per outer iteration and drops at
    // most one per inner iteration; the caps guard against degenerate cycling.
    let max_outer = 20 * (m + 1) + 10;
    let max_inner = 20 * (m + 1) + 10;

    let mut status = QpStatus::MaxIterations;
    'outer: for _ in 0..max_outer {
        // Most-violated constraint not in the working set.
        let mut p_best: Option<usize> = None;
        let mut viol_best = T::zero();
        for i in 0..m {
            if active.contains(&i) {
                continue;
            }
            let row_i: Vec<T> = (0..n).map(|j| a[[i, j]]).collect();
            let ax = T::vec_dot(&row_i, &x);
            let viol = match senses[i] {
                QpSense::Equality => (b[i] - ax).abs(),
                QpSense::Inequality => T::max(T::zero(), b[i] - ax),
            };
            let tol_i = tol * (T::one() + b[i].abs());
            if viol > tol_i && (p_best.is_none() || viol > viol_best) {
                p_best = Some(i);
                viol_best = viol;
            }
        }
        let Some(p) = p_best else {
            status = QpStatus::Optimal;
            break 'outer;
        };

        // Orient constraint p so that b̂ - âᵀx > 0 (equalities may need a
        // sign flip; inequalities always keep s = +1 when violated).
        let row_p_raw: Vec<T> = (0..n).map(|j| a[[p, j]]).collect();
        let s = if b[p] - T::vec_dot(&row_p_raw, &x) >= T::zero() {
            T::one()
        } else {
            -T::one()
        };
        let row_p: Vec<T> = row_p_raw.iter().map(|&aij| s * aij).collect();
        let b_p = s * b[p];

        // Inner loop: enforce p while keeping the working set optimal.
        //
        // With W the working set, the step z and multiplier rates (w, v)
        // satisfy the homogeneous system
        //   H z - A_Wᵀ w - v â = 0,   A_W z = 0,   âᵀ z = 1,
        // solved exactly via the (SPD) range-space Schur complement with
        //   M = A_W H⁻¹ A_Wᵀ,  q = A_W H⁻¹ â,  s0 = âᵀ H⁻¹ â:
        //   ŵ = M⁻¹ q,  v = 1 / (s0 - qᵀ ŵ),  z = v (H⁻¹ â - H⁻¹ A_Wᵀ ŵ).
        // mult_p accumulates the multiplier of p across partial steps.
        let mut mult_p = T::zero();
        for _ in 0..max_inner {
            let z_p = solve_h(&row_p);
            let zs: Vec<Vec<T>> = rows.iter().map(|r| solve_h(r)).collect();
            let k = rows.len();

            let s0 = T::vec_dot(&row_p, &z_p);
            let mut w_hat: Vec<T> = Vec::new();
            let mut denom = s0;
            if k > 0 {
                let mut m_mat = sonium_solvers_core::dense::DenseMatrix::zeros(k, k);
                let mut qv = vec![T::zero(); k];
                for (ii, ri) in rows.iter().enumerate() {
                    for (jj, zj) in zs.iter().enumerate() {
                        m_mat[[ii, jj]] = T::vec_dot(ri, zj);
                    }
                    qv[ii] = T::vec_dot(ri, &z_p);
                }
                w_hat = match dense_linalg::solve_spd(&m_mat, &qv) {
                    Some(w) => w,
                    // Rank-deficient working set: GMW-regularised solve.
                    // `m_mat` is the k x k working-set Gram matrix built
                    // above; `qv` has length k.
                    None => {
                        dense_linalg::solve_modified_cholesky(&m_mat, &qv)
                            .expect("working-set Gram matrix is square by construction")
                            .0
                    }
                };
                denom = s0 - T::vec_dot(&qv, &w_hat);
            }
            let denom_min =
                T::machine_epsilon() * T::from_f64(100.0).unwrap() * T::max(s0, T::one());
            if denom <= denom_min {
                // denom = ||â - A_Wᵀ ŵ||²_{H⁻¹} ≈ 0: â lies in the span of
                // the working-set normals, so no direction keeps the whole
                // working set active while moving towards p. Exchange
                // instead: the multipliers along u_j(t) = u_j - t·ŵ_j,
                // λ_p(t) = t stay feasible until the first active
                // inequality multiplier hits zero; drop that constraint
                // (the primal point does not move) and retry.
                let mut t_star: Option<T> = None;
                let mut drop_idx = 0_usize;
                for ii in 0..k {
                    if senses[active[ii]] != QpSense::Inequality || w_hat[ii] <= T::zero() {
                        continue;
                    }
                    let t = mult[ii] / w_hat[ii];
                    if t_star.is_none() || t < t_star.unwrap() {
                        t_star = Some(t);
                        drop_idx = ii;
                    }
                }
                let Some(t) = t_star else {
                    // Only equalities (or non-positive rates) span â while
                    // p is violated: inconsistent constraints.
                    status = QpStatus::Infeasible;
                    break 'outer;
                };
                for ii in 0..k {
                    mult[ii] -= t * w_hat[ii];
                }
                mult_p += t;
                active.remove(drop_idx);
                rows.remove(drop_idx);
                signs.remove(drop_idx);
                mult.remove(drop_idx);
                continue;
            }
            let v = T::one() / denom;

            // z = v (z_p - Σ ŵ_j Z_j); multiplier rates r_j = -v ŵ_j.
            let mut z = z_p;
            for (ii, zj) in zs.iter().enumerate() {
                for l in 0..n {
                    z[l] -= w_hat[ii] * zj[l];
                }
            }
            for zi in z.iter_mut() {
                *zi *= v;
            }

            // Full step length: makes constraint p exactly satisfied.
            let t_full = b_p - T::vec_dot(&row_p, &x);

            // Blocking step: first active-inequality multiplier hitting zero.
            let mut t_block: Option<T> = None;
            let mut block_idx = 0_usize;
            for ii in 0..k {
                if senses[active[ii]] != QpSense::Inequality {
                    continue;
                }
                let rate = -v * w_hat[ii];
                if rate < T::zero() {
                    let t = -mult[ii] / rate;
                    if t_block.is_none() || t < t_block.unwrap() {
                        t_block = Some(t);
                        block_idx = ii;
                    }
                }
            }

            let full = t_block.is_none() || t_full <= t_block.unwrap();
            let t = if full { t_full } else { t_block.unwrap() };
            for l in 0..n {
                x[l] += t * z[l];
            }
            for ii in 0..k {
                mult[ii] += t * (-v * w_hat[ii]);
            }
            mult_p += t * v;

            if full {
                active.push(p);
                rows.push(row_p.clone());
                signs.push(s);
                mult.push(mult_p);
                break; // constraint p enforced; pick the next violated one
            }
            // Drop the blocking inequality and re-solve with the same p.
            active.remove(block_idx);
            rows.remove(block_idx);
            signs.remove(block_idx);
            mult.remove(block_idx);
        }
    }

    let mut multipliers = vec![T::zero(); m];
    for (ii, &ai) in active.iter().enumerate() {
        multipliers[ai] = mult[ii] * signs[ii];
    }
    QpSolution {
        d: x,
        multipliers,
        active_set: active,
        status,
    }
}

/// Jacobian of a constraint at `x`: analytic if supplied, otherwise a
/// central finite-difference approximation via [`jacobian_central_auto`].
/// Constraints are never silently ignored.
pub(crate) fn constraint_jacobian<T: RealScalar>(
    c: &Constraint<T>,
    x: &[T],
) -> sonium_solvers_core::dense::DenseMatrix<T> {
    match c.jac {
        Some(ref jac_fn) => jac_fn(x),
        None => jacobian_central_auto(&c.fun, x, c.dim),
    }
}

/// Checked Lagrangian-gradient difference for optimizer paths
/// that must not re-enter infallible constraint/Jacobian callbacks.
pub(crate) fn lagrangian_gradient_difference_checked<T: RealScalar>(
    g_new: &[T],
    g: &[T],
    lambda: &[T],
    x_new: &[T],
    x: &[T],
    constraints: &[Constraint<T>],
    iteration: usize,
) -> Result<Vec<T>, OptimError> {
    let n = g.len();
    if g_new.len() != n || x.len() != n || x_new.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: g_new.len().max(x.len()).max(x_new.len()),
        });
    }
    let expected_lambda: usize = constraints.iter().map(|constraint| constraint.dim).sum();
    if lambda.len() < expected_lambda {
        return Err(OptimError::DimensionMismatch {
            expected: expected_lambda,
            got: lambda.len(),
        });
    }
    let mut y: Vec<T> = (0..n).map(|j| g_new[j] - g[j]).collect();
    let mut row = 0_usize;
    for (index, constraint) in constraints.iter().enumerate() {
        let jac_new = checked_constraint_jacobian(constraint, x_new, n, index, iteration)?;
        let jac = checked_constraint_jacobian(constraint, x, n, index, iteration)?;
        for k in 0..constraint.dim {
            let lam = lambda[row];
            if lam != T::zero() {
                for j in 0..n {
                    y[j] -= lam * (jac_new[[k, j]] - jac[[k, j]]);
                }
            }
            row += 1;
        }
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: "checked Lagrangian gradient difference is non-finite".to_string(),
        });
    }
    Ok(y)
}

/// Checked entry point for the linearised QP assembly.  In particular, never
/// truncates a caller-provided bound vector: a non-empty vector must describe
/// every variable before any constraint Jacobian or QP iteration is run.
pub(crate) fn try_solve_linearized_qp<T: RealScalar>(
    g: &[T],
    b_hess: &sonium_solvers_core::dense::DenseMatrix<T>,
    x: &[T],
    constraints: &[Constraint<T>],
    bounds: &[Bound<T>],
    n: usize,
) -> Result<QpSolution<T>, OptimError> {
    if g.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: g.len(),
        });
    }
    if x.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: x.len(),
        });
    }
    if b_hess.nrows() != n || b_hess.ncols() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n * n,
            got: b_hess.nrows() * b_hess.ncols(),
        });
    }
    validate_bounds(bounds, n)?;
    let (a, rhs, senses) = assemble_linearized_qp_checked(g, b_hess, x, constraints, bounds, n)?;
    Ok(solve_qp(b_hess, g, &a, &rhs, &senses))
}

/// Assemble the checked active-set QP without re-entering the legacy
/// infallible callback path. This is deliberately separate from
/// `solve_linearized_qp`: all production constrained optimizers use this
/// function through `try_solve_linearized_qp`, while the old helper remains a
/// compatibility path for callers that already proved its invariants.
fn assemble_linearized_qp_checked<T: RealScalar>(
    g: &[T],
    b_hess: &sonium_solvers_core::dense::DenseMatrix<T>,
    x: &[T],
    constraints: &[Constraint<T>],
    bounds: &[Bound<T>],
    n: usize,
) -> Result<CheckedQpAssembly<T>, OptimError> {
    if g.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: g.len(),
        });
    }
    if x.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: x.len(),
        });
    }
    if b_hess.nrows() != n || b_hess.ncols() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n * n,
            got: b_hess.nrows() * b_hess.ncols(),
        });
    }
    if let Some(index) = g.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("QP gradient entry {index} is non-finite"),
        });
    }
    if let Some(index) = x.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("QP point entry {index} is non-finite"),
        });
    }
    for row in 0..n {
        for column in 0..n {
            if !b_hess[[row, column]].is_finite() {
                return Err(OptimError::NumericalError {
                    iteration: 0,
                    message: format!("QP Hessian entry ({row}, {column}) is non-finite"),
                });
            }
        }
    }

    let row_count = constraints
        .iter()
        .map(|constraint| constraint.dim)
        .sum::<usize>()
        + bounds
            .iter()
            .map(|bound| usize::from(bound.lower.is_some()) + usize::from(bound.upper.is_some()))
            .sum::<usize>();
    let mut matrix = sonium_solvers_core::dense::DenseMatrix::zeros(row_count, n);
    let mut rhs = vec![T::zero(); row_count];
    let mut senses = Vec::with_capacity(row_count);
    let mut row = 0_usize;

    for (index, constraint) in constraints.iter().enumerate() {
        let values = checked_constraint_values(constraint, x, index, 0)?;
        let jacobian = checked_constraint_jacobian(constraint, x, n, index, 0)?;
        for k in 0..constraint.dim {
            for column in 0..n {
                matrix[[row, column]] = jacobian[[k, column]];
            }
            rhs[row] = -values[k];
            senses.push(match constraint.kind {
                ConstraintKind::Equality => QpSense::Equality,
                ConstraintKind::Inequality => QpSense::Inequality,
            });
            row += 1;
        }
    }
    for (index, bound) in bounds.iter().enumerate() {
        if let Some(lower) = bound.lower {
            matrix[[row, index]] = T::one();
            rhs[row] = lower - x[index];
            senses.push(QpSense::Inequality);
            row += 1;
        }
        if let Some(upper) = bound.upper {
            matrix[[row, index]] = -T::one();
            rhs[row] = x[index] - upper;
            senses.push(QpSense::Inequality);
            row += 1;
        }
    }
    Ok((matrix, rhs, senses))
}

/// Evaluate a constraint value callback without allowing a user panic to
/// escape the checked optimizer boundary.
pub(crate) fn checked_constraint_values<T: RealScalar>(
    constraint: &Constraint<T>,
    x: &[T],
    index: usize,
    iteration: usize,
) -> Result<Vec<T>, OptimError> {
    let values = catch_unwind(AssertUnwindSafe(|| (constraint.fun)(x))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: format!("constraint {index} values"),
        }
    })?;
    if values.len() != constraint.dim {
        return Err(OptimError::InvalidConstraintDimension {
            index,
            expected: constraint.dim,
            got: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration,
            message: format!("constraint {index} returned non-finite value"),
        });
    }
    Ok(values)
}

/// Checked Jacobian evaluation for the active-set boundary. The existing
/// infallible helper is retained for legacy internal callers, but this path
/// must also protect finite-difference callbacks whose output shape changes
/// at a perturbed point.
pub(crate) fn checked_constraint_jacobian<T: RealScalar>(
    constraint: &Constraint<T>,
    x: &[T],
    n: usize,
    index: usize,
    iteration: usize,
) -> Result<sonium_solvers_core::dense::DenseMatrix<T>, OptimError> {
    if let Some(jacobian_fn) = constraint.jac.as_ref() {
        let jacobian = catch_unwind(AssertUnwindSafe(|| jacobian_fn(x))).map_err(|_| {
            OptimError::CallbackPanicked {
                iteration,
                stage: format!("constraint {index} Jacobian"),
            }
        })?;
        if jacobian.nrows() != constraint.dim || jacobian.ncols() != n {
            return Err(OptimError::InvalidConstraintJacobian {
                index,
                expected_rows: constraint.dim,
                expected_cols: n,
                got_rows: jacobian.nrows(),
                got_cols: jacobian.ncols(),
            });
        }
        if (0..jacobian.nrows())
            .any(|row| (0..jacobian.ncols()).any(|column| !jacobian[[row, column]].is_finite()))
        {
            return Err(OptimError::NumericalError {
                iteration,
                message: format!("constraint {index} returned non-finite Jacobian"),
            });
        }
        return Ok(jacobian);
    }

    let mut jacobian = sonium_solvers_core::dense::DenseMatrix::zeros(constraint.dim, n);
    let step_scale = T::from_f64(f64::EPSILON.powf(1.0 / 3.0)).unwrap();
    let two = T::from_f64(2.0).unwrap();
    for column in 0..n {
        let step = step_scale * T::max(T::one(), x[column].abs());
        let mut plus = x.to_vec();
        let mut minus = x.to_vec();
        plus[column] += step;
        minus[column] -= step;
        let plus_values = checked_constraint_values(constraint, &plus, index, iteration)?;
        let minus_values = checked_constraint_values(constraint, &minus, index, iteration)?;
        for row in 0..constraint.dim {
            let value = (plus_values[row] - minus_values[row]) / (two * step);
            if !value.is_finite() {
                return Err(OptimError::NumericalError {
                    iteration,
                    message: format!("constraint {index} returned non-finite Jacobian"),
                });
            }
            jacobian[[row, column]] = value;
        }
    }
    Ok(jacobian)
}

/// Convert the row ordering used by the linearised QP into the public KKT
/// multiplier layout.  QP rows are user constraints first, then each
/// variable's lower and upper bound row in that order.
pub(crate) fn kkt_multipliers_from_qp<T: RealScalar>(
    qp_multipliers: &[T],
    constraints: &[Constraint<T>],
    bounds: &[Bound<T>],
) -> Result<KktMultipliers<T>, OptimError> {
    let constraint_count: usize = constraints.iter().map(|constraint| constraint.dim).sum();
    let bound_row_count: usize = bounds
        .iter()
        .map(|bound| usize::from(bound.lower.is_some()) + usize::from(bound.upper.is_some()))
        .sum();
    let expected = constraint_count + bound_row_count;
    if qp_multipliers.len() != expected {
        return Err(OptimError::DimensionMismatch {
            expected,
            got: qp_multipliers.len(),
        });
    }
    let mut result = KktMultipliers {
        constraints: qp_multipliers[..constraint_count].to_vec(),
        lower_bounds: vec![T::zero(); bounds.len()],
        upper_bounds: vec![T::zero(); bounds.len()],
    };
    let mut row = constraint_count;
    for (index, bound) in bounds.iter().enumerate() {
        if bound.lower.is_some() {
            result.lower_bounds[index] = qp_multipliers[row];
            row += 1;
        }
        if bound.upper.is_some() {
            result.upper_bounds[index] = qp_multipliers[row];
            row += 1;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn dense(
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    ) -> sonium_solvers_core::dense::DenseMatrix<f64> {
        sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(rows, cols, data)
    }

    #[test]
    fn test_qp_unconstrained_empty_active_set() {
        // min ½ dᵀ d + [1, 2]ᵀ d → d = (-1, -2); no constraints.
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![1.0, 2.0];
        let a = dense(0, 2, vec![]);
        let sol = solve_qp(&h, &g, &a, &[], &[]);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(sol.active_set.is_empty());
        assert_relative_eq!(sol.d[0], -1.0, epsilon = 1e-12);
        assert_relative_eq!(sol.d[1], -2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_qp_equality_only() {
        // min ½ (x² + y²)  s.t.  x + y = 1  →  (0.5, 0.5), λ = 0.5.
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![0.0, 0.0];
        let a = dense(1, 2, vec![1.0, 1.0]);
        let b = sonium_solvers_core::array![1.0];
        let sol = solve_qp(&h, &g, &a, &b, &[QpSense::Equality]);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.d[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(sol.d[1], 0.5, epsilon = 1e-12);
        // Stationarity: H d + g = λ a → λ = 0.5.
        assert_relative_eq!(sol.multipliers[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_qp_inequality_active() {
        // min ½ ||d||² - [2,2]ᵀ d (unconstrained min (2,2))
        // s.t. -x - y ≥ -1   ⇔   x + y ≤ 1  →  (0.5, 0.5), λ = 1.5.
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![-2.0, -2.0];
        let a = dense(1, 2, vec![-1.0, -1.0]);
        let b = sonium_solvers_core::array![-1.0];
        let sol = solve_qp(&h, &g, &a, &b, &[QpSense::Inequality]);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_eq!(sol.active_set, vec![0]);
        assert_relative_eq!(sol.d[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(sol.d[1], 0.5, epsilon = 1e-12);
        // Stationarity: d + g = λ a → (-1.5, -1.5) = λ (-1, -1) → λ = 1.5 ≥ 0.
        assert_relative_eq!(sol.multipliers[0], 1.5, epsilon = 1e-12);
    }

    #[test]
    fn test_qp_inequality_inactive() {
        // Same objective, constraint x + y ≤ 10: inactive at the optimum.
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![-2.0, -2.0];
        let a = dense(1, 2, vec![-1.0, -1.0]);
        let b = sonium_solvers_core::array![-10.0];
        let sol = solve_qp(&h, &g, &a, &b, &[QpSense::Inequality]);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(sol.active_set.is_empty());
        assert_relative_eq!(sol.d[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(sol.d[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(sol.multipliers[0], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_qp_degenerate_duplicate_constraints() {
        // Duplicate equality rows (rank-deficient if both were activated):
        // the second is never violated once the first is enforced. The
        // inequality x + y ≤ 1 is redundant with the equality.
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![0.0, 0.0];
        let a = dense(3, 2, vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0]);
        let b = sonium_solvers_core::array![1.0, 1.0, -1.0];
        let senses = [QpSense::Equality, QpSense::Equality, QpSense::Inequality];
        let sol = solve_qp(&h, &g, &a, &b, &senses);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_eq!(sol.active_set.len(), 1);
        assert_relative_eq!(sol.d[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(sol.d[1], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_qp_degenerate_span_drop() {
        // min ½((x-1)² + (y-1)²) s.t. x + y ≤ 0.5, x ≤ 0.2, y ≤ 0.2.
        // Optimum (0.2, 0.2): only x ≤ 0.2 and y ≤ 0.2 active, λ = 0.8 each.
        // Enforcing y ≤ 0.2 requires dropping x + y ≤ 0.5 from the working
        // set (its normal spans R² together with x ≤ 0.2, the degenerate
        // zero-direction case of the dual method).
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![-1.0, -1.0];
        let a = dense(3, 2, vec![-1.0, -1.0, -1.0, 0.0, 0.0, -1.0]);
        let b = sonium_solvers_core::array![-0.5, -0.2, -0.2];
        let senses = [
            QpSense::Inequality,
            QpSense::Inequality,
            QpSense::Inequality,
        ];
        let sol = solve_qp(&h, &g, &a, &b, &senses);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.d[0], 0.2, epsilon = 1e-10);
        assert_relative_eq!(sol.d[1], 0.2, epsilon = 1e-10);
        assert_eq!(sol.active_set, vec![1, 2]);
        assert_relative_eq!(sol.multipliers[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(sol.multipliers[1], 0.8, epsilon = 1e-10);
        assert_relative_eq!(sol.multipliers[2], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_qp_infeasible_detected() {
        // x + y = 1 and x + y = 2: inconsistent.
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![0.0, 0.0];
        let a = dense(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let b = sonium_solvers_core::array![1.0, 2.0];
        let sol = solve_qp(&h, &g, &a, &b, &[QpSense::Equality, QpSense::Equality]);
        assert_eq!(sol.status, QpStatus::Infeasible);
    }

    #[test]
    fn test_qp_mixed_equality_inequality() {
        // min ½ ((x-2)² + (y-1)²)  s.t.  x + y = 2,  x ≥ 0.25.
        // Along x + y = 2 the objective is minimised at (1.5, 0.5);
        // x ≥ 0.25 is slack there.
        let h = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![-2.0, -1.0];
        let a = dense(2, 2, vec![1.0, 1.0, 1.0, 0.0]);
        let b = sonium_solvers_core::array![2.0, 0.25];
        let sol = solve_qp(&h, &g, &a, &b, &[QpSense::Equality, QpSense::Inequality]);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.d[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(sol.d[1], 0.5, epsilon = 1e-10);
        assert_eq!(sol.active_set, vec![0]);
    }

    #[test]
    fn test_qp_indefinite_hessian_regularised() {
        // Indefinite H (eigenvalues -1, 3): GMW regularisation must still
        // produce a finite step that satisfies the equality exactly.
        let h = dense(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        let g = sonium_solvers_core::array![1.0, 0.0];
        let a = dense(1, 2, vec![1.0, 1.0]);
        let b = sonium_solvers_core::array![1.0];
        let sol = solve_qp(&h, &g, &a, &b, &[QpSense::Equality]);
        assert!(sol.d.iter().all(|&di| di.is_finite()));
        assert_relative_eq!(sol.d[0] + sol.d[1], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_qp_f32() {
        let h = sonium_solvers_core::dense::DenseMatrix::<f32>::eye(2);
        let g = sonium_solvers_core::array![0.0_f32, 0.0_f32];
        let a =
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0_f32, 1.0]);
        let b = sonium_solvers_core::array![1.0_f32];
        let sol = solve_qp(&h, &g, &a, &b, &[QpSense::Equality]);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.d[0] - 0.5).abs() < 1e-5, "d = {:?}", sol.d);
        assert!((sol.d[1] - 0.5).abs() < 1e-5, "d = {:?}", sol.d);
    }

    #[test]
    fn test_linearized_qp_bounds() {
        // min ½ d² - 5 d  (unconstrained d = 5) with x = 0, x + d ≤ 1.
        let b_hess = dense(1, 1, vec![1.0]);
        let g = sonium_solvers_core::array![-5.0];
        let x = sonium_solvers_core::array![0.0];
        let bounds = [Bound::<f64>::upper(1.0)];
        let sol = try_solve_linearized_qp(&g, &b_hess, &x, &[], &bounds, 1).unwrap();
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.d[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linearized_qp_fd_jacobian_matches_analytic() {
        // Equality-constrained SQP step: analytic vs finite-difference
        // Jacobian must give (nearly) the same direction.
        let b_hess = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let g = sonium_solvers_core::array![0.0, 0.0];
        let x = sonium_solvers_core::array![2.0, 2.0];
        let mk = |with_jac: bool| Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| sonium_solvers_core::array![x[0] * x[0] + x[1] - 3.0]),
            jac: if with_jac {
                Some(Box::new(|x: &[f64]| {
                    sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                        1,
                        2,
                        vec![2.0 * x[0], 1.0],
                    )
                }))
            } else {
                None
            },
        };
        let d_exact = try_solve_linearized_qp(&g, &b_hess, &x, &[mk(true)], &[], 2)
            .unwrap()
            .d;
        let d_fd = try_solve_linearized_qp(&g, &b_hess, &x, &[mk(false)], &[], 2)
            .unwrap()
            .d;
        assert_relative_eq!(d_fd[0], d_exact[0], epsilon = 1e-5);
        assert_relative_eq!(d_fd[1], d_exact[1], epsilon = 1e-5);
    }

    #[test]
    fn checked_linearized_qp_rejects_truncated_bounds() {
        let hessian = dense(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let error = try_solve_linearized_qp(
            &sonium_solvers_core::array![0.0, 0.0],
            &hessian,
            &sonium_solvers_core::array![0.0, 0.0],
            &[],
            &[Bound::lower(0.0)],
            2,
        )
        .expect_err("a partial bound vector must not be truncated");
        assert!(matches!(
            error,
            OptimError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn checked_linearized_qp_rejects_nonfinite_numeric_inputs() {
        let hessian = dense(1, 1, vec![1.0]);
        let error = try_solve_linearized_qp(&[f64::NAN], &hessian, &[0.0], &[], &[], 1)
            .expect_err("non-finite QP gradient must fail before assembly");
        assert!(matches!(error, OptimError::NumericalError { .. }));

        let error = try_solve_linearized_qp(
            &[0.0],
            &dense(1, 1, vec![f64::INFINITY]),
            &[0.0],
            &[],
            &[],
            1,
        )
        .expect_err("non-finite QP Hessian must fail before factorization");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn checked_linearized_qp_rejects_malformed_constraint_callbacks() {
        let hessian = dense(1, 1, vec![1.0]);
        let wrong_values = Constraint {
            kind: ConstraintKind::Equality,
            dim: 2,
            fun: Box::new(|_| vec![0.0]),
            jac: None,
        };
        let error = try_solve_linearized_qp(&[0.0], &hessian, &[0.0], &[wrong_values], &[], 1)
            .expect_err("wrong constraint output shape must fail before QP assembly");
        assert!(matches!(
            error,
            OptimError::InvalidConstraintDimension {
                expected: 2,
                got: 1,
                ..
            }
        ));

        let wrong_jacobian = Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|_| vec![0.0]),
            jac: Some(Box::new(|_| dense(1, 2, vec![1.0, 0.0]))),
        };
        let error = try_solve_linearized_qp(&[0.0], &hessian, &[0.0], &[wrong_jacobian], &[], 1)
            .expect_err("wrong Jacobian shape must fail before QP assembly");
        assert!(matches!(
            error,
            OptimError::InvalidConstraintJacobian {
                expected_rows: 1,
                expected_cols: 1,
                got_rows: 1,
                got_cols: 2,
                ..
            }
        ));

        let nonfinite = Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|_| vec![f64::NAN]),
            jac: Some(Box::new(|_| dense(1, 1, vec![1.0]))),
        };
        let error = try_solve_linearized_qp(&[0.0], &hessian, &[0.0], &[nonfinite], &[], 1)
            .expect_err("non-finite constraint output must fail before QP assembly");
        assert!(matches!(error, OptimError::NumericalError { .. }));

        let panicking_values = Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|_| panic!("constraint callback failed")),
            jac: Some(Box::new(|_| dense(1, 1, vec![1.0]))),
        };
        let error = try_solve_linearized_qp(&[0.0], &hessian, &[0.0], &[panicking_values], &[], 1)
            .expect_err("constraint callback panic must become a typed error");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));

        let panicking_jacobian = Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|_| vec![0.0]),
            jac: Some(Box::new(|_| panic!("Jacobian callback failed"))),
        };
        let error =
            try_solve_linearized_qp(&[0.0], &hessian, &[0.0], &[panicking_jacobian], &[], 1)
                .expect_err("Jacobian callback panic must become a typed error");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));
    }
}
