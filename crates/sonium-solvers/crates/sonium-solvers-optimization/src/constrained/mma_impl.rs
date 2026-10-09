//! MMA: Method of Moving Asymptotes (Svanberg 1987), with the optional
//! globally-convergent GCMMA variant (Svanberg 2002).
//!
//! The de-facto standard for bound-constrained topology optimisation with
//! many design variables and few constraints (NLopt ships the same
//! algorithm). Each outer iteration builds a convex separable approximation
//! of the problem at `x^k`,
//!
//! ```text
//! min  sum_j [ p0_j / (U_j - x_j) + q0_j / (x_j - L_j) ]
//! s.t. sum_j [ p_ij / (U_j - x_j) + q_ij / (x_j - L_j) ] <= b_i,  i = 1..m
//!      alpha_j <= x_j <= beta_j
//! ```
//!
//! with moving asymptotes `L_j < x_j < U_j` updated by Svanberg's rule
//! (shrink by `asym_decr` on oscillation of successive iterates, expand by
//! `asym_incr` on monotone movement) and subproblem bounds `alpha`, `beta`
//! combining the box bounds, an `albefa` fraction of the asymptote
//! distance, and the move limit. The subproblem is solved exactly through
//! its dual: for `lambda >= 0` the primal minimiser is separable in closed
//! form, and the (small, `m`-dimensional) dual is maximised by Newton
//! steps on the explicit dual Hessian with an Armijo ascent line search.
//!
//! With [`MmaVariant::Mma`] (default) this is plain MMA: one subproblem
//! solve per outer iteration, which is what NLopt and most
//! topology-optimisation codes use in practice.
//!
//! With [`MmaVariant::Gcmma`] each outer iteration runs an inner
//! conservative-approximation loop (Svanberg 2002, "A class of globally
//! convergent optimization methods based on conservative convex separable
//! approximations"): the `raa`-regularisation of the coefficients is
//! replaced by per-function parameters `rho_i` (strictly positive, so the
//! approximations are strictly convex and first-order consistent). After
//! each subproblem solve the true functions are evaluated at the
//! subproblem optimum `x̂`; whenever an approximation is not conservative
//! (`f̃_i(x̂) < f_i(x̂)`), its `rho_i` is increased by Svanberg's rule
//! `rho_i <- min(1.1 (rho_i + delta_i), 10 rho_i)` with
//! `delta_i = (f_i(x̂) - f̃_i(x̂)) / d(x̂)`,
//! `d(x̂) = sum_j (U_j-L_j)(x̂_j-x_j)^2 / ((U_j-x̂_j)(x̂_j-L_j)(xmax_j-xmin_j))`,
//! and the subproblem is re-solved (no new gradient evaluations needed).
//! Between outer iterations `rho_i` decays slowly (`rho_i / 2`) towards
//! the gradient-scaled floor `0.1/n * sum_j |df_i/dx_j| (xmax_j-xmin_j)`,
//! so the method reverts to plain MMA speed near the solution. Accepted
//! iterates satisfy `f_0(x^{k+1}) <= f_0(x^k)` at feasible points (the
//! subproblem optimum beats `x^k` under the conservative approximation),
//! which is the global-convergence mechanism.
//!
//! Complexity per outer iteration is `O(m * n)` plus the `O(m^2 n)` dual
//! Hessian — no dense `n x n` Hessian is ever formed (that would defeat
//! the purpose for large `n`).
//!
//! Limitations: only inequality constraints are supported (the crate's
//! [`ConstraintKind::Inequality`], `c(x) >= 0`, is negated internally to
//! MMA's `f_i(x) <= 0` form); equality constraints return an error.
//! Variables need box bounds; missing entries fall back to a wide default
//! box around `x0`.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{
    Bound, Constraint, ConstraintKind, OptimError, OptimResult, OptimStatus, RealScalar,
    validate_bounds,
};
use crate::util::dense_linalg::{
    cholesky, modified_cholesky, solve_lower_triangular, solve_lower_triangular_transpose,
};

#[inline]
fn cst<T: RealScalar>(v: f64) -> T {
    T::from_f64(v).unwrap()
}

/// Algorithm variant: plain MMA (default) or the globally-convergent
/// GCMMA (Svanberg 2002).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MmaVariant {
    /// Plain MMA (Svanberg 1987): one subproblem solve per outer
    /// iteration. Fast in practice, no convergence guarantee.
    #[default]
    Mma,
    /// GCMMA (Svanberg 2002): inner conservative-approximation loop with
    /// `rho` updates; provably converges to a KKT point from arbitrary
    /// starts at the cost of extra function evaluations (gradients are
    /// still evaluated only once per outer iteration).
    Gcmma,
}

/// Configuration for MMA (defaults are Svanberg's standard values).
#[derive(Debug, Clone)]
pub struct MmaConfig<T: RealScalar> {
    /// Maximum outer (approximation) iterations.
    pub max_iterations: usize,
    /// Maximum Newton iterations per dual subproblem solve.
    pub max_inner_iterations: usize,
    /// Convergence tolerance on the KKT residual of the original problem
    /// (max of projected stationarity and constraint violation).
    pub kkt_tolerance: T,
    /// Tolerance on the dual subproblem's projected gradient.
    pub inner_tolerance: T,
    /// Move limit as a fraction of the box width (Svanberg: 0.5).
    pub move_limit: T,
    /// Initial asymptote distance as a fraction of the box width (0.5).
    pub asym_init: T,
    /// Asymptote expansion factor on monotone movement (1.2).
    pub asym_incr: T,
    /// Asymptote shrink factor on oscillation (0.7).
    pub asym_decr: T,
    /// Subproblem bound margin as a fraction of the asymptote distance (0.1).
    pub albefa: T,
    /// Regularisation of the objective approximation coefficients (1e-5).
    pub raa0: T,
    /// Regularisation of the constraint approximation coefficients (1e-5).
    pub raa: T,
    /// Algorithm variant: plain MMA (default) or GCMMA.
    pub variant: MmaVariant,
    /// Maximum conservative-approximation (inner) iterations per outer
    /// iteration under [`MmaVariant::Gcmma`]. If the budget is exhausted
    /// the last subproblem optimum is accepted anyway.
    pub max_conservative_iterations: usize,
    /// Box bounds. Missing entries default to a wide box around `x0`.
    pub bounds: Vec<Bound<T>>,
}

impl<T: RealScalar> Default for MmaConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            max_inner_iterations: 50,
            kkt_tolerance: cst(1e-6),
            inner_tolerance: cst(1e-9),
            move_limit: cst(0.5),
            asym_init: cst(0.5),
            asym_incr: cst(1.2),
            asym_decr: cst(0.7),
            albefa: cst(0.1),
            raa0: cst(1e-5),
            raa: cst(1e-5),
            variant: MmaVariant::Mma,
            max_conservative_iterations: 15,
            bounds: Vec::new(),
        }
    }
}

/// Minimise `f(x)` subject to inequality constraints and bounds using MMA.
///
/// Constraints must all be [`ConstraintKind::Inequality`] (`c(x) >= 0`);
/// they are converted to MMA's `f_i(x) = -c_i(x) <= 0` form. Passing an
/// equality constraint is an error (MMA has no mechanism for equalities).
pub fn mma<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &MmaConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    if config.bounds.len() > x0.len() {
        return Err(OptimError::DimensionMismatch {
            expected: x0.len(),
            got: config.bounds.len(),
        });
    }
    super::validate_optimizer_start(f, grad_fn, x0, constraints)?;
    let full_bounds = (0..x0.len())
        .map(|index| {
            config.bounds.get(index).copied().unwrap_or_else(|| {
                let wide = T::from_f64(100.0).unwrap() * T::one().max(x0[index].abs());
                Bound::both(x0[index] - wide, x0[index] + wide)
            })
        })
        .collect::<Vec<_>>();
    validate_bounds(&full_bounds, x0.len())?;
    let callback_non_finite = Cell::new(false);
    let checked_f = |point: &[T]| {
        let value = f(point);
        if !value.is_finite() {
            callback_non_finite.set(true);
        }
        value
    };
    let checked_grad = |point: &[T]| {
        let gradient = grad_fn(point);
        if gradient.len() != point.len() || gradient.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
            return vec![T::nan(); point.len()];
        }
        gradient
    };
    let inner_result = catch_unwind(AssertUnwindSafe(|| {
        mma_impl(
            &checked_f,
            &checked_grad,
            x0,
            constraints,
            config,
            &callback_non_finite,
        )
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "MMA callback".into(),
    })?;
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "MMA callback returned a non-finite value".into(),
        });
    }
    let mut result = inner_result?;
    super::finalize_identity_kkt(&mut result, constraints, &full_bounds, config.kkt_tolerance)?;
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn mma_impl<T, F, Fg>(
    f: &F,
    grad_fn: &Fg,
    x0: &[T],
    constraints: &[Constraint<T>],
    config: &MmaConfig<T>,
    callback_non_finite: &Cell<bool>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    Fg: Fn(&[T]) -> Vec<T>,
{
    if constraints
        .iter()
        .any(|c| c.kind == ConstraintKind::Equality)
    {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "MMA supports inequality constraints only".to_string(),
        });
    }

    let n = x0.len();
    let m: usize = constraints.iter().map(|c| c.dim).sum();

    // Box bounds; missing entries get a wide default box around x0.
    let mut xmin = vec![T::zero(); n];
    let mut xmax = vec![T::zero(); n];
    for j in 0..n {
        let b = config.bounds.get(j).copied().unwrap_or(Bound::unbounded());
        let wide = cst::<T>(100.0) * T::max(T::one(), x0[j].abs());
        xmin[j] = b.lower.unwrap_or(x0[j] - wide);
        xmax[j] = b.upper.unwrap_or(x0[j] + wide);
        if xmin[j] >= xmax[j] {
            return Err(OptimError::InvalidBounds {
                index: j,
                lower: xmin[j].to_f64().unwrap_or(0.0),
                upper: xmax[j].to_f64().unwrap_or(0.0),
            });
        }
    }

    // Constraint values and gradients in MMA's f_i(x) <= 0 form.
    let eval_constraints = |x: &[T]| -> (Vec<T>, Vec<Vec<T>>) {
        let mut fval = Vec::with_capacity(m);
        let mut grad: Vec<Vec<T>> = vec![vec![T::zero(); n]; m];
        let mut row = 0;
        for c in constraints {
            let v = (c.fun)(x);
            if v.len() != c.dim || v.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                return (vec![T::nan(); m], vec![vec![T::nan(); n]; m]);
            }
            let j = match super::qp_active_set::checked_constraint_jacobian(c, x, n, 0, 0) {
                Ok(jacobian) => jacobian,
                Err(_) => {
                    callback_non_finite.set(true);
                    return (vec![T::nan(); m], vec![vec![T::nan(); n]; m]);
                }
            };
            for k in 0..c.dim {
                fval.push(-v[k]);
                for jj in 0..n {
                    grad[row][jj] = -j[[k, jj]];
                }
                row += 1;
            }
        }
        (fval, grad)
    };

    // Constraint values only (f_i(x) <= 0 form) — the GCMMA inner loop
    // does not need gradients.
    let eval_constraint_values = |x: &[T]| -> Vec<T> {
        let mut fval = Vec::with_capacity(m);
        for c in constraints {
            let v = (c.fun)(x);
            if v.len() != c.dim || v.iter().any(|value| !value.is_finite()) {
                callback_non_finite.set(true);
                return vec![T::nan(); m];
            }
            for &vi in v.iter() {
                fval.push(-vi);
            }
        }
        fval
    };

    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;

    let mut x: Vec<T> = (0..n)
        .map(|j| T::min(T::max(x0[j], xmin[j]), xmax[j]))
        .collect();
    let mut fx = f(&x);
    if !fx.is_finite() {
        callback_non_finite.set(true);
    }
    let mut gx = grad_fn(&x);
    if gx.len() != n || gx.iter().any(|value| !value.is_finite()) {
        callback_non_finite.set(true);
        gx = vec![T::nan(); n];
    }
    let (mut fval, mut gcon) = eval_constraints(&x);

    let mut xold1 = x.clone();
    let mut xold2 = x.clone();
    let mut low: Vec<T> = (0..n)
        .map(|j| x[j] - config.asym_init * (xmax[j] - xmin[j]))
        .collect();
    let mut upp: Vec<T> = (0..n)
        .map(|j| x[j] + config.asym_init * (xmax[j] - xmin[j]))
        .collect();
    let mut lambda = vec![T::zero(); m];

    let mut status = OptimStatus::MaxIterationsReached;
    let mut converged = false;
    let mut iterations = 0_usize;
    let mut kkt_stationarity = T::infinity();

    let gcmma = matches!(config.variant, MmaVariant::Gcmma);
    // GCMMA conservativeness parameters: rho[0] for the objective,
    // rho[1..] per constraint (Svanberg 2002). Zero until initialised.
    let mut rho = vec![T::zero(); m + 1];
    let inner_budget = if gcmma {
        config.max_conservative_iterations
    } else {
        0
    };

    for iter in 0..config.max_iterations {
        // Asymptote adaptation (skip on the first two iterates).
        if iter >= 2 {
            update_asymptotes(
                &x,
                &xold1,
                &xold2,
                &mut low,
                &mut upp,
                &xmin,
                &xmax,
                config.asym_decr,
                config.asym_incr,
            );
        }

        // Subproblem bounds.
        let (alpha, beta) = subproblem_bounds(&x, &low, &upp, &xmin, &xmax, config);

        // Coefficient regularisation: `raa` for plain MMA; for GCMMA the
        // conservativeness parameters rho, reset at the start of each
        // outer iteration to Svanberg's gradient-scaled floor
        // `0.1/n * sum_j |df_i/dx_j| (xmax_j - xmin_j)` (below 1e-6
        // clamped to 1e-6) — the carried-over value decays by 1/2 per
        // outer iteration, so an elevated rho fades back towards the
        // floor once the approximation is conservative again.
        let mut reg = vec![T::zero(); m + 1];
        if gcmma {
            let n_t = cst::<T>(n as f64);
            let floor = |grad: &[T]| -> T {
                let mut s = T::zero();
                for j in 0..n {
                    s += grad[j].abs() * (xmax[j] - xmin[j]);
                }
                T::max(cst::<T>(1e-6), cst::<T>(0.1) * s / n_t)
            };
            rho[0] = T::max(rho[0] / cst::<T>(2.0), floor(&gx));
            for (i, gi) in gcon.iter().enumerate() {
                rho[i + 1] = T::max(rho[i + 1] / cst::<T>(2.0), floor(gi));
            }
            reg.clone_from_slice(&rho);
        } else {
            reg[0] = config.raa0;
            for r in reg.iter_mut().skip(1) {
                *r = config.raa;
            }
        }

        // Inner conservative-approximation loop: plain MMA solves the
        // subproblem exactly once; GCMMA re-solves with increased rho
        // until every approximation is conservative at the subproblem
        // optimum. Gradients are NOT re-evaluated inside the loop.
        let mut x_new = x.clone();
        let mut lambda_new = lambda.clone();
        let mut fx_new = fx;
        // Assigned on every inner-loop pass before any read (the loop
        // always runs at least once).
        #[allow(unused_assignments)]
        let mut fval_new: Vec<T> = Vec::new();
        for _inner in 0..=inner_budget {
            // Approximation coefficients.
            let (p0, q0, p, q, bvec) =
                approximation_coefficients(&gx, &gcon, &fval, &x, &low, &upp, &xmin, &xmax, &reg);

            // Solve the subproblem via its dual.
            let (x_hat, lambda_hat) = solve_dual(
                &p0,
                &q0,
                &p,
                &q,
                &bvec,
                &low,
                &upp,
                &alpha,
                &beta,
                &lambda_new,
                config.max_inner_iterations,
                config.inner_tolerance,
            );

            let fx_hat = f(&x_hat);
            fun_evals += 1;
            let fv_hat = eval_constraint_values(&x_hat);

            x_new = x_hat;
            lambda_new = lambda_hat;
            fx_new = fx_hat;
            fval_new = fv_hat;

            if !gcmma {
                break;
            }

            // Conservativeness check: f̃_i(x̂) >= f_i(x̂) for all i
            // (objective and constraints). The approximation value uses
            // the incremental form f̃_i(x̂) = f_i(x^k) + sum_j [p_ij
            // (1/(U-x̂) - 1/(U-x^k)) + q_ij (1/(x̂-L) - 1/(x^k-L))].
            let d_x = conservativeness_d(&x, &x_new, &low, &upp, &xmin, &xmax);
            let mut all_conservative = true;
            let f0_app = approximation_value(&p0, &q0, fx, &x, &x_new, &low, &upp);
            let slack0 = cst::<T>(1e-12) * T::max(T::one(), fx_new.abs());
            if f0_app < fx_new - slack0 {
                all_conservative = false;
                increase_rho(&mut rho[0], fx_new - f0_app, d_x);
            }
            for (i, fv) in fval_new.iter().enumerate() {
                let fi_app = approximation_value(&p[i], &q[i], fval[i], &x, &x_new, &low, &upp);
                let slack = cst::<T>(1e-12) * T::max(T::one(), fv.abs());
                if fi_app < *fv - slack {
                    all_conservative = false;
                    increase_rho(&mut rho[i + 1], *fv - fi_app, d_x);
                }
            }
            if all_conservative {
                break;
            }
            reg.clone_from_slice(&rho);
        }

        xold2 = xold1;
        xold1 = x;
        x = x_new;
        lambda = lambda_new;

        fx = fx_new;
        gx = grad_fn(&x);
        grad_evals += 1;
        let (fv, gc) = eval_constraints(&x);
        fval = fv;
        gcon = gc;

        // KKT residual of the ORIGINAL problem at the new iterate.
        let (stat, cons_viol) = kkt_residual(&gx, &gcon, &fval, &lambda, &x, &xmin, &xmax);
        kkt_stationarity = stat;
        iterations = iter + 1;
        if T::max(stat, cons_viol) < config.kkt_tolerance {
            converged = true;
            status = OptimStatus::Converged;
            break;
        }
    }

    Ok(OptimResult {
        x,
        fun: fx,
        grad: Some(gx),
        iterations,
        fun_evals,
        grad_evals,
        hess_evals: 0,
        grad_norm: kkt_stationarity,
        converged,
        status,
        kkt: None,
    })
}

/// Svanberg's asymptote update: shrink by `asym_decr` when successive
/// iterates oscillate (sign change of the step), expand by `asym_incr`
/// otherwise, then clip so `L_j < x_j < U_j` with a margin and the
/// asymptotes stay within a generous multiple of the box.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_asymptotes<T: RealScalar>(
    x: &[T],
    xold1: &[T],
    xold2: &[T],
    low: &mut [T],
    upp: &mut [T],
    xmin: &[T],
    xmax: &[T],
    asym_decr: T,
    asym_incr: T,
) {
    for j in 0..x.len() {
        let range = xmax[j] - xmin[j];
        let oscillating = (x[j] - xold1[j]) * (xold1[j] - xold2[j]) < T::zero();
        let (mut l, mut u) = if oscillating {
            (
                x[j] - asym_decr * (xold1[j] - low[j]),
                x[j] + asym_decr * (upp[j] - xold1[j]),
            )
        } else {
            (
                x[j] - asym_incr * (xold1[j] - low[j]),
                x[j] + asym_incr * (upp[j] - xold1[j]),
            )
        };
        l = T::min(l, x[j] - cst::<T>(0.01) * range);
        l = T::max(l, xmin[j] - cst::<T>(10.0) * range);
        u = T::max(u, x[j] + cst::<T>(0.01) * range);
        u = T::min(u, xmax[j] + cst::<T>(10.0) * range);
        low[j] = l;
        upp[j] = u;
    }
}

/// MMA approximation coefficients `p0, q0` (objective) and `p, q`
/// (per constraint) plus the constraint right-hand sides `b`.
///
/// `p*_j = (U_j - x_j)^2 * (df^+ + 1e-3 |df| + reg / (xmax_j - xmin_j))`,
/// `q*_j` analogously with `df^-` and `(x_j - L_j)^2`; both are strictly
/// positive so the subproblem is strictly convex and its dual well-posed.
/// `reg[0]` is the objective regularisation (`raa0` for MMA, `rho_0` for
/// GCMMA), `reg[1..]` the per-constraint values (`raa` / `rho_i`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn approximation_coefficients<T: RealScalar>(
    gx: &[T],
    gcon: &[Vec<T>],
    fval: &[T],
    x: &[T],
    low: &[T],
    upp: &[T],
    xmin: &[T],
    xmax: &[T],
    reg: &[T],
) -> (Vec<T>, Vec<T>, Vec<Vec<T>>, Vec<Vec<T>>, Vec<T>) {
    let n = x.len();
    let m = fval.len();
    let coeff = |df: T, ux: T, xl: T, range: T, raa: T| -> (T, T) {
        let reg = raa / range;
        (
            ux * ux * (T::max(df, T::zero()) + cst::<T>(1e-3) * df.abs() + reg),
            xl * xl * (T::max(-df, T::zero()) + cst::<T>(1e-3) * df.abs() + reg),
        )
    };

    let mut p0 = vec![T::zero(); n];
    let mut q0 = vec![T::zero(); n];
    for j in 0..n {
        let (pj, qj) = coeff(
            gx[j],
            upp[j] - x[j],
            x[j] - low[j],
            xmax[j] - xmin[j],
            reg[0],
        );
        p0[j] = pj;
        q0[j] = qj;
    }

    let mut p = vec![vec![T::zero(); n]; m];
    let mut q = vec![vec![T::zero(); n]; m];
    let mut bvec = vec![T::zero(); m];
    for i in 0..m {
        let mut bi = -fval[i];
        for j in 0..n {
            let ux = upp[j] - x[j];
            let xl = x[j] - low[j];
            let (pj, qj) = coeff(gcon[i][j], ux, xl, xmax[j] - xmin[j], reg[i + 1]);
            p[i][j] = pj;
            q[i][j] = qj;
            bi += pj / ux + qj / xl;
        }
        bvec[i] = bi;
    }
    (p0, q0, p, q, bvec)
}

/// Approximating-function value at `x_hat` in incremental form:
/// `f̃(x̂) = f(x^k) + sum_j [p_j (1/(U_j-x̂_j) - 1/(U_j-x^k_j)) + q_j
/// (1/(x̂_j-L_j) - 1/(x^k_j-L_j))]`. Identical to `sum_j p/(U-x̂) +
/// q/(x̂-L) + r` but avoids materialising `r`.
fn approximation_value<T: RealScalar>(
    p: &[T],
    q: &[T],
    f_at_xk: T,
    x: &[T],
    x_hat: &[T],
    low: &[T],
    upp: &[T],
) -> T {
    let mut v = f_at_xk;
    for j in 0..x.len() {
        let ux = upp[j] - x[j];
        let xl = x[j] - low[j];
        v += p[j] * (T::one() / (upp[j] - x_hat[j]) - T::one() / ux)
            + q[j] * (T::one() / (x_hat[j] - low[j]) - T::one() / xl);
    }
    v
}

/// Svanberg's `d(x)` (GCMMA): the rho-dependent part of every
/// approximating function, `f̃_i(x) = h_i(x) + rho_i d(x)` with `h_i`
/// independent of `rho_i`:
/// `d(x) = sum_j (U_j-L_j)(x_j-x^k_j)^2 / ((U_j-x_j)(x_j-L_j)(xmax_j-xmin_j))`.
fn conservativeness_d<T: RealScalar>(
    x: &[T],
    x_hat: &[T],
    low: &[T],
    upp: &[T],
    xmin: &[T],
    xmax: &[T],
) -> T {
    let mut d = T::zero();
    for j in 0..x.len() {
        let dx = x_hat[j] - x[j];
        d += (upp[j] - low[j]) * dx * dx
            / ((upp[j] - x_hat[j]) * (x_hat[j] - low[j]) * (xmax[j] - xmin[j]));
    }
    d
}

/// Svanberg's rho increase (GCMMA) for a non-conservative approximation:
/// `delta = deficit / d(x̂)` makes `rho + delta` exactly conservative at
/// `x̂`; the update dampens it to `rho <- min(1.1 (rho + delta), 10 rho)`.
/// When `d(x̂)` vanishes (subproblem optimum ≈ x^k) the geometric factor
/// dominates.
fn increase_rho<T: RealScalar>(rho: &mut T, deficit: T, d_x: T) {
    let delta = deficit / T::max(d_x, T::machine_epsilon());
    *rho = T::min(cst::<T>(1.1) * (*rho + delta), cst::<T>(10.0) * *rho);
}

/// Subproblem bounds: box bounds intersected with the asymptote margin
/// (`albefa`) and the move limit.
fn subproblem_bounds<T: RealScalar>(
    x: &[T],
    low: &[T],
    upp: &[T],
    xmin: &[T],
    xmax: &[T],
    config: &MmaConfig<T>,
) -> (Vec<T>, Vec<T>) {
    let n = x.len();
    let mut alpha = vec![T::zero(); n];
    let mut beta = vec![T::zero(); n];
    for j in 0..n {
        let range = xmax[j] - xmin[j];
        alpha[j] = T::max(
            T::max(xmin[j], low[j] + config.albefa * (x[j] - low[j])),
            x[j] - config.move_limit * range,
        );
        beta[j] = T::min(
            T::min(xmax[j], upp[j] - config.albefa * (upp[j] - x[j])),
            x[j] + config.move_limit * range,
        );
    }
    (alpha, beta)
}

/// Primal minimiser of the separable subproblem for a fixed dual point.
///
/// Returns `(x, P, Q)` with `P_j = p0_j + lambda^T p_:j`,
/// `Q_j = q0_j + lambda^T q_:j`; the unconstrained per-variable minimiser
/// of `P/(U - x) + Q/(x - L)` is `(L sqrt(P) + U sqrt(Q)) / (sqrt(P) +
/// sqrt(Q))`, clipped to `[alpha_j, beta_j]`.
#[allow(clippy::too_many_arguments)]
fn subproblem_primal<T: RealScalar>(
    lambda: &[T],
    p0: &[T],
    q0: &[T],
    p: &[Vec<T>],
    q: &[Vec<T>],
    low: &[T],
    upp: &[T],
    alpha: &[T],
    beta: &[T],
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let n = p0.len();
    let m = lambda.len();
    let mut x = vec![T::zero(); n];
    let mut pp = vec![T::zero(); n];
    let mut qq = vec![T::zero(); n];
    let tiny = cst::<T>(1e-14);
    for j in 0..n {
        let mut pj = p0[j];
        let mut qj = q0[j];
        for i in 0..m {
            pj += lambda[i] * p[i][j];
            qj += lambda[i] * q[i][j];
        }
        pp[j] = pj;
        qq[j] = qj;
        let sp = T::max(pj, tiny).sqrt();
        let sq = T::max(qj, tiny).sqrt();
        let xj = (low[j] * sp + upp[j] * sq) / (sp + sq);
        x[j] = T::min(T::max(xj, alpha[j]), beta[j]);
    }
    (x, pp, qq)
}

/// Dual objective `W(lambda) = sum_j [P/(U-x) + Q/(x-L)] - lambda^T b`
/// (constant terms dropped) and its gradient
/// `g_i = sum_j [p_ij/(U-x) + q_ij/(x-L)] - b_i`.
#[allow(clippy::too_many_arguments)]
fn dual_value_grad<T: RealScalar>(
    lambda: &[T],
    p: &[Vec<T>],
    q: &[Vec<T>],
    bvec: &[T],
    low: &[T],
    upp: &[T],
    x: &[T],
    pp: &[T],
    qq: &[T],
) -> (T, Vec<T>) {
    let n = x.len();
    let m = lambda.len();
    let mut w = T::zero();
    for j in 0..n {
        w += pp[j] / (upp[j] - x[j]) + qq[j] / (x[j] - low[j]);
    }
    let mut g = vec![T::zero(); m];
    for i in 0..m {
        let mut gi = -bvec[i];
        for j in 0..n {
            gi += p[i][j] / (upp[j] - x[j]) + q[i][j] / (x[j] - low[j]);
        }
        g[i] = gi;
        w -= lambda[i] * bvec[i];
    }
    (w, g)
}

/// Solve the convex separable subproblem through its dual with Newton
/// steps on the explicit dual Hessian and an Armijo ascent line search.
/// Returns the subproblem-optimal `(x, lambda)`.
#[allow(clippy::too_many_arguments)]
fn solve_dual<T: RealScalar>(
    p0: &[T],
    q0: &[T],
    p: &[Vec<T>],
    q: &[Vec<T>],
    bvec: &[T],
    low: &[T],
    upp: &[T],
    alpha: &[T],
    beta: &[T],
    lambda0: &[T],
    max_iter: usize,
    tol: T,
) -> (Vec<T>, Vec<T>) {
    let n = p0.len();
    let m = bvec.len();
    let (x0, _, _) = subproblem_primal(lambda0, p0, q0, p, q, low, upp, alpha, beta);
    if m == 0 {
        return (x0, Vec::new());
    }

    let mut lambda = lambda0.to_vec();
    let tiny = cst::<T>(1e-13);

    for _ in 0..max_iter {
        let (x, pp, qq) = subproblem_primal(&lambda, p0, q0, p, q, low, upp, alpha, beta);
        let (w, g) = dual_value_grad(&lambda, p, q, bvec, low, upp, &x, &pp, &qq);

        // Projected gradient of the dual over lambda >= 0.
        let mut res = T::zero();
        for i in 0..m {
            let ri = if lambda[i] > T::zero() {
                g[i].abs()
            } else {
                T::max(g[i], T::zero())
            };
            if ri > res {
                res = ri;
            }
        }
        if res < tol {
            break;
        }

        // Dual Hessian (negative semidefinite):
        // H_il = -sum_j free [ e_ij * e_lj / D_j ] with
        // e_ij = p_ij/u^2 - q_ij/l^2, u = U-x, l = x-L,
        // D_j = 2P/u^3 + 2Q/l^3; clipped variables (x at alpha/beta)
        // contribute nothing.
        let mut neg_h = sonium_solvers_core::dense::DenseMatrix::zeros(m, m);
        // Precompute per-variable factors.
        let mut dvec = vec![T::zero(); n];
        let mut free = vec![false; n];
        for j in 0..n {
            let u = upp[j] - x[j];
            let l = x[j] - low[j];
            let clipped = (x[j] - alpha[j]).abs() <= tiny * T::max(T::one(), alpha[j].abs())
                || (beta[j] - x[j]).abs() <= tiny * T::max(T::one(), beta[j].abs());
            free[j] = !clipped;
            dvec[j] = cst::<T>(2.0) * pp[j] / (u * u * u) + cst::<T>(2.0) * qq[j] / (l * l * l);
        }
        for i in 0..m {
            for l2 in 0..=i {
                let mut val = T::zero();
                for j in 0..n {
                    if !free[j] || dvec[j] <= T::zero() {
                        continue;
                    }
                    let u = upp[j] - x[j];
                    let l = x[j] - low[j];
                    let ei = p[i][j] / (u * u) - q[i][j] / (l * l);
                    let el = p[l2][j] / (u * u) - q[l2][j] / (l * l);
                    val += ei * el / dvec[j];
                }
                neg_h[[i, l2]] = val;
                neg_h[[l2, i]] = val;
            }
        }

        // Newton direction: (-H) delta = g (ascent direction since H is
        // negative semidefinite). Fall back to modified Cholesky. The
        // regularisation must scale with the problem: when every variable
        // sits at a subproblem bound the true dual Hessian is ~0 (W is
        // piecewise linear there), and an unscaled epsilon would make the
        // Newton step astronomically large so no Armijo backtrack could
        // ever recover the kink where variables unclip.
        let h_scale = (0..m).map(|i| neg_h[[i, i]].abs()).fold(T::one(), T::max);
        for i in 0..m {
            neg_h[[i, i]] += cst::<T>(1e-9) * h_scale;
        }
        let mut delta = match cholesky(&neg_h) {
            Some(lf) => {
                let yv = solve_lower_triangular(&lf, &g);
                solve_lower_triangular_transpose(&lf, &yv)
            }
            None => {
                // `neg_h` is the m x m dual Hessian assembled above.
                let (lf, _) =
                    modified_cholesky(&neg_h).expect("MMA dual Hessian is square by construction");
                let yv = solve_lower_triangular(&lf, &g);
                solve_lower_triangular_transpose(&lf, &yv)
            }
        };
        if !delta.iter().all(|&v| v.is_finite()) {
            break;
        }

        // Trust-region-style cap on the dual step: keeps the line search
        // in the basin where the concave quadratic model is meaningful.
        let delta_inf = delta.iter().map(|&v| v.abs()).fold(T::zero(), T::max);
        let lambda_inf = lambda.iter().map(|&v| v.abs()).fold(T::zero(), T::max);
        let step_cap = T::max(T::one(), cst::<T>(10.0) * lambda_inf);
        if delta_inf > step_cap {
            let scale = step_cap / delta_inf;
            for d in delta.iter_mut() {
                *d *= scale;
            }
        }

        // Armijo ascent line search with projection onto lambda >= 0.
        let slope: T = (0..m)
            .map(|i| g[i] * delta[i])
            .fold(T::zero(), |a, v| a + v);
        let mut t = T::one();
        let mut accepted = false;
        for _ in 0..50 {
            let lambda_t: Vec<T> = (0..m)
                .map(|i| T::max(lambda[i] + t * delta[i], T::zero()))
                .collect();
            let (x_t, pp_t, qq_t) =
                subproblem_primal(&lambda_t, p0, q0, p, q, low, upp, alpha, beta);
            let (w_t, _) = dual_value_grad(&lambda_t, p, q, bvec, low, upp, &x_t, &pp_t, &qq_t);
            if w_t >= w + cst::<T>(1e-4) * t * slope {
                lambda = lambda_t;
                accepted = true;
                break;
            }
            t *= cst(0.5);
            if t < cst(1e-12) {
                break;
            }
        }
        if !accepted {
            break;
        }
    }

    let (x_fin, _, _) = subproblem_primal(&lambda, p0, q0, p, q, low, upp, alpha, beta);
    (x_fin, lambda)
}

/// KKT residual of the original problem: projected stationarity
/// `max_j |df0_j + sum_i lambda_i dfi_j|` (zeroing components where the
/// box bound allows the residual) and the max constraint violation.
fn kkt_residual<T: RealScalar>(
    gx: &[T],
    gcon: &[Vec<T>],
    fval: &[T],
    lambda: &[T],
    x: &[T],
    xmin: &[T],
    xmax: &[T],
) -> (T, T) {
    let n = x.len();
    let m = fval.len();
    let mut stat = T::zero();
    for j in 0..n {
        let mut r = gx[j];
        for i in 0..m {
            r += lambda[i] * gcon[i][j];
        }
        let bound_tol = cst::<T>(1e-10) * T::max(T::one(), xmax[j] - xmin[j]);
        let viol = if x[j] <= xmin[j] + bound_tol {
            // Lower bound active: KKT requires r >= 0.
            T::max(-r, T::zero())
        } else if x[j] >= xmax[j] - bound_tol {
            // Upper bound active: KKT requires r <= 0.
            T::max(r, T::zero())
        } else {
            r.abs()
        };
        if viol > stat {
            stat = viol;
        }
    }
    let cons_viol = fval
        .iter()
        .map(|&v| T::max(v, T::zero()))
        .fold(T::zero(), T::max);
    (stat, cons_viol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// min sum_i x_i  s.t.  n - sum_i 1/x_i >= 0,  x in [0.2, 5]^n.
    /// KKT point: x_i = 1 for all i (multiplier lambda = 1).
    #[allow(clippy::type_complexity)]
    fn reciprocal_sum_problem(
        n: usize,
    ) -> (
        impl Fn(&[f64]) -> f64,
        impl Fn(&[f64]) -> Vec<f64>,
        Vec<Constraint<f64>>,
        MmaConfig<f64>,
    ) {
        let f = move |x: &[f64]| x.iter().sum();
        let g = |x: &[f64]| vec![1.0; x.len()];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(move |x: &[f64]| {
                vec![n as f64 - x.iter().map(|&xi| 1.0 / xi).sum::<f64>()]
            }),
            jac: Some(Box::new(|x: &[f64]| {
                let n = x.len();
                let row: Vec<f64> = x.iter().map(|&xi| 1.0 / (xi * xi)).collect();
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, n, row)
            })),
        }];
        let config = MmaConfig {
            bounds: (0..n).map(|_| Bound::both(0.2, 5.0)).collect(),
            ..MmaConfig::default()
        };
        (f, g, constraints, config)
    }

    #[test]
    fn test_mma_reciprocal_sum() {
        let n = 5;
        let (f, g, constraints, config) = reciprocal_sum_problem(n);
        let x0 = vec![2.0; n];
        let result = mma(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        for j in 0..n {
            assert_relative_eq!(result.x[j], 1.0, epsilon = 1e-4);
        }
        assert_relative_eq!(result.fun, n as f64, epsilon = 1e-3);
    }

    #[test]
    fn test_mma_reciprocal_sum_large_n() {
        // Topology-optimisation-flavoured scaling: many variables, one
        // constraint. Iteration count should stay modest in n.
        let n = 50;
        let (f, g, constraints, config) = reciprocal_sum_problem(n);
        let x0 = vec![2.0; n];
        let result = mma(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(
            result.iterations < 60,
            "iterations should not blow up with n: {}",
            result.iterations
        );
        for j in 0..n {
            assert_relative_eq!(result.x[j], 1.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_mma_compliance_like_2d() {
        // Compliance-flavoured quadratic with a reciprocal constraint:
        //   min 0.5 (x1^2 + x2^2)  s.t.  1/x1 + 2/x2 <= 3, x in [0.1, 10].
        // KKT: lambda = x1^3, x2 = 2^{1/3} x1,
        //      x1* = (1 + 2^{2/3}) / 3, x2* = 2^{1/3} x1*.
        let f = |x: &[f64]| 0.5 * (x[0].powi(2) + x[1].powi(2));
        let g = |x: &[f64]| vec![x[0], x[1]];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![3.0 - 1.0 / x[0] - 2.0 / x[1]]),
            jac: Some(Box::new(|x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![1.0 / (x[0] * x[0]), 2.0 / (x[1] * x[1])],
                )
            })),
        }];
        let config = MmaConfig {
            bounds: vec![Bound::both(0.1, 10.0), Bound::both(0.1, 10.0)],
            ..MmaConfig::default()
        };
        let result = mma(&f, &g, &[2.0, 2.0], &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let x1s = (1.0 + 2.0_f64.powf(2.0 / 3.0)) / 3.0;
        let x2s = 2.0_f64.powf(1.0 / 3.0) * x1s;
        assert_relative_eq!(result.x[0], x1s, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], x2s, epsilon = 1e-4);
        // Constraint active at the optimum.
        let c = 3.0 - 1.0 / result.x[0] - 2.0 / result.x[1];
        assert!(c.abs() < 1e-4, "constraint should be active: c = {c:e}");
    }

    #[test]
    fn test_mma_bounds_respected() {
        // min (x + 2)^2 with box [-1, 1]: optimum at the lower bound.
        let f = |x: &[f64]| (x[0] + 2.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] + 2.0)];
        let config = MmaConfig {
            bounds: vec![Bound::both(-1.0, 1.0)],
            ..MmaConfig::default()
        };
        let result = mma(&f, &g, &[0.7], &[], &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], -1.0, epsilon = 1e-6);
        assert!(result.x[0] >= -1.0 && result.x[0] <= 1.0);
    }

    #[test]
    fn test_mma_asymptote_oscillation_rule() {
        // Oscillating history (x: 1.0 -> 0.9 -> 1.0) must shrink the
        // asymptote distance; monotone history must expand it.
        let xmin = vec![-10.0];
        let xmax = vec![10.0];

        let mut low_os = vec![0.5];
        let mut upp_os = vec![1.5];
        update_asymptotes(
            &[1.0],
            &[0.9],
            &[1.0],
            &mut low_os,
            &mut upp_os,
            &xmin,
            &xmax,
            0.7,
            1.2,
        );
        // l = 1 - 0.7 * (0.9 - 0.5) = 0.72 (shrunk distance 0.28 < 0.4)
        assert_relative_eq!(low_os[0], 0.72, epsilon = 1e-12);
        assert_relative_eq!(upp_os[0], 1.42, epsilon = 1e-12);

        let mut low_mo = vec![0.5];
        let mut upp_mo = vec![1.5];
        update_asymptotes(
            &[1.0],
            &[0.9],
            &[0.8],
            &mut low_mo,
            &mut upp_mo,
            &xmin,
            &xmax,
            0.7,
            1.2,
        );
        // l = 1 - 1.2 * (0.9 - 0.5) = 0.52 (expanded distance 0.48 > 0.4)
        assert_relative_eq!(low_mo[0], 0.52, epsilon = 1e-12);
        assert_relative_eq!(upp_mo[0], 1.72, epsilon = 1e-12);
    }

    #[test]
    fn test_mma_oscillating_problem_converges() {
        // Steep quartic pulls the iterate far each step while the box and
        // move limit push back — a setting where successive iterates
        // oscillate and the asymptote shrink rule must engage. The solve
        // must still reach the optimum x = 0.5.
        let f = |x: &[f64]| (x[0] - 0.5).powi(4) + (x[1] - 0.5).powi(4);
        let g = |x: &[f64]| vec![4.0 * (x[0] - 0.5).powi(3), 4.0 * (x[1] - 0.5).powi(3)];
        let config = MmaConfig {
            bounds: vec![Bound::both(0.0, 1.0), Bound::both(0.0, 1.0)],
            ..MmaConfig::default()
        };
        let result = mma(&f, &g, &[0.05, 0.95], &[], &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_mma_deterministic() {
        let n = 5;
        let (f, g, constraints, config) = reciprocal_sum_problem(n);
        let x0 = vec![2.0; n];
        let r1 = mma(&f, &g, &x0, &constraints, &config).unwrap();
        let r2 = mma(&f, &g, &x0, &constraints, &config).unwrap();
        assert_eq!(r1.x, r2.x, "MMA must be deterministic");
        assert_eq!(r1.fun, r2.fun);
        assert_eq!(r1.iterations, r2.iterations);
    }

    #[test]
    fn test_mma_rejects_equalities() {
        let f = |x: &[f64]| x[0] * x[0];
        let g = |x: &[f64]| vec![2.0 * x[0]];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0] - 1.0]),
            jac: None,
        }];
        let config = MmaConfig::default();
        let err = mma(&f, &g, &[2.0], &constraints, &config).unwrap_err();
        assert!(matches!(err, OptimError::NumericalError { .. }));
    }

    #[test]
    fn test_gcmma_reciprocal_sum() {
        // GCMMA must converge on the standard MMA test-suite problems.
        let n = 5;
        let (f, g, constraints, config) = reciprocal_sum_problem(n);
        let config = MmaConfig {
            variant: MmaVariant::Gcmma,
            ..config
        };
        let x0 = vec![2.0; n];
        let result = mma(&f, &g, &x0, &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        println!(
            "gcmma reciprocal_sum n=5: iters={} fun_evals={}",
            result.iterations, result.fun_evals
        );
        for j in 0..n {
            assert_relative_eq!(result.x[j], 1.0, epsilon = 1e-3);
        }
        assert_relative_eq!(result.fun, n as f64, epsilon = 1e-3);
    }

    #[test]
    fn test_gcmma_compliance_like_2d() {
        // Same compliance-flavoured problem as test_mma_compliance_like_2d.
        let f = |x: &[f64]| 0.5 * (x[0].powi(2) + x[1].powi(2));
        let g = |x: &[f64]| vec![x[0], x[1]];
        let constraints = vec![Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![3.0 - 1.0 / x[0] - 2.0 / x[1]]),
            jac: Some(Box::new(|x: &[f64]| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![1.0 / (x[0] * x[0]), 2.0 / (x[1] * x[1])],
                )
            })),
        }];
        let config = MmaConfig {
            variant: MmaVariant::Gcmma,
            bounds: vec![Bound::both(0.1, 10.0), Bound::both(0.1, 10.0)],
            ..MmaConfig::default()
        };
        let result = mma(&f, &g, &[2.0, 2.0], &constraints, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        let x1s = (1.0 + 2.0_f64.powf(2.0 / 3.0)) / 3.0;
        let x2s = 2.0_f64.powf(1.0 / 3.0) * x1s;
        assert_relative_eq!(result.x[0], x1s, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], x2s, epsilon = 1e-3);
    }

    #[test]
    fn test_gcmma_oscillating_problem_converges() {
        // The steep-quartic oscillation problem under GCMMA.
        let f = |x: &[f64]| (x[0] - 0.5).powi(4) + (x[1] - 0.5).powi(4);
        let g = |x: &[f64]| vec![4.0 * (x[0] - 0.5).powi(3), 4.0 * (x[1] - 0.5).powi(3)];
        let config = MmaConfig {
            variant: MmaVariant::Gcmma,
            bounds: vec![Bound::both(0.0, 1.0), Bound::both(0.0, 1.0)],
            ..MmaConfig::default()
        };
        let result = mma(&f, &g, &[0.05, 0.95], &[], &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        // The quartic gradient 4(x-0.5)^3 is flat near the optimum, so
        // the 1e-6 KKT tolerance corresponds to ~6e-3 in x.
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_gcmma_nonconvex_monotone_vs_mma() {
        // Strongly nonconvex objective: f(x) = x^4 - 4 x^2 + x (asymmetric
        // double well, global minimum near x ≈ -1.52, local minimum near
        // x ≈ 1.35). The MMA convex separable approximation of a concave
        // region underestimates the true function, so plain MMA takes large
        // non-conservative steps and oscillates between the wells; GCMMA
        // detects the non-conservativeness at each candidate and re-solves
        // with increased rho, which keeps it in the starting well and
        // converges immediately.
        //
        // Measurement (also printed below): on this instance plain MMA
        // hits the 200-iteration cap WITHOUT converging (the iterates
        // bounce between wells, f ending mid-oscillation), while GCMMA
        // converges in 7 outer iterations / 16 function evaluations.
        let f = |x: &[f64]| x[0].powi(4) - 4.0 * x[0].powi(2) + x[0];
        let g = |x: &[f64]| vec![4.0 * x[0].powi(3) - 8.0 * x[0] + 1.0];
        let mk_config = |variant: MmaVariant| MmaConfig {
            variant,
            bounds: vec![Bound::both(-3.0, 3.0)],
            ..MmaConfig::default()
        };
        let x0 = [0.9_f64];

        let mma_res = mma(&f, &g, &x0, &[], &mk_config(MmaVariant::Mma)).unwrap();
        let gcmma_res = mma(&f, &g, &x0, &[], &mk_config(MmaVariant::Gcmma)).unwrap();
        println!(
            "nonconvex double well: MMA iters={} evals={} f={:.6} converged={} | \
             GCMMA iters={} evals={} f={:.6} converged={}",
            mma_res.iterations,
            mma_res.fun_evals,
            mma_res.fun,
            mma_res.converged,
            gcmma_res.iterations,
            gcmma_res.fun_evals,
            gcmma_res.fun,
            gcmma_res.converged,
        );
        assert!(gcmma_res.converged, "status: {:?}", gcmma_res.status);
        assert!(gcmma_res.grad_norm < 1e-6);
        // GCMMA stays in the starting well (conservative steps cannot jump
        // the barrier between the wells): local minimum near x ≈ 1.35.
        assert_relative_eq!(gcmma_res.x[0], 1.347, epsilon = 1e-2);
        // Plain MMA oscillates between the wells and does not converge.
        assert!(
            !mma_res.converged,
            "documented divergence case: MMA unexpectedly converged (iters={}, f={})",
            mma_res.iterations, mma_res.fun
        );
        assert!(
            gcmma_res.iterations <= 20,
            "GCMMA should converge quickly: {} iterations",
            gcmma_res.iterations
        );
    }

    #[test]
    fn test_gcmma_deterministic() {
        let n = 5;
        let (f, g, constraints, config) = reciprocal_sum_problem(n);
        let config = MmaConfig {
            variant: MmaVariant::Gcmma,
            ..config
        };
        let x0 = vec![2.0; n];
        let r1 = mma(&f, &g, &x0, &constraints, &config).unwrap();
        let r2 = mma(&f, &g, &x0, &constraints, &config).unwrap();
        assert_eq!(r1.x, r2.x, "GCMMA must be deterministic");
        assert_eq!(r1.iterations, r2.iterations);
    }
}
