//! L-BFGS-B: limited-memory BFGS with bound constraints.
//!
//! Implements the Byrd-Lu-Nocedal-Zhu (1995) algorithm (see also Nocedal &
//! Wright, *Numerical Optimization*, 2nd ed., §16.7). Each iteration:
//!
//! 1. Computes the *generalised Cauchy point* by following the piecewise
//!    linear projected-gradient path `P(x - t g)` and minimising the
//!    limited-memory quadratic model along it, using the compact `(W, M)`
//!    representation of the L-BFGS matrix.
//! 2. Fixes the variables active at the Cauchy point and computes a subspace
//!    step on the remaining free variables with the two-loop L-BFGS recursion
//!    restricted to the free block (including the exact coupling term
//!    `B (x^c - x)` between the fixed and free variables).
//! 3. Obtains the new iterate by a projected backtracking Armijo line search
//!    from the Cauchy point along the subspace direction, falling back to the
//!    Cauchy point itself (or to a search along the Cauchy path) when the
//!    subspace step is not a descent direction or the line search fails.
//!
//! Convergence is declared when the infinity norm of the projected gradient
//! falls below `grad_tolerance`.

use std::cell::Cell;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::types::{
    Bound, Bounds, OptimError, OptimResult, OptimStatus, RealScalar, project_bounds,
    validate_bounds,
};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for L-BFGS-B.
#[derive(Debug, Clone)]
pub struct LbfgsbConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub fun_tolerance: T,
    /// Number of correction pairs (m).
    pub memory: usize,
    /// Variable bounds.
    pub bounds: Bounds<T>,
}

impl<T: RealScalar> Default for LbfgsbConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: T::from_f64(1e-6).unwrap(),
            fun_tolerance: T::from_f64(1e-12).unwrap(),
            memory: 10,
            bounds: Vec::new(),
        }
    }
}

/// Minimise `f(x)` subject to bound constraints using L-BFGS-B.
pub fn lbfgsb<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &LbfgsbConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    super::validate_optimizer_start(f, grad_fn, x0, &[])?;
    validate_bounds(&config.bounds, x0.len())?;
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
        if gradient.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
        }
        gradient
    };
    let mut result = catch_unwind(AssertUnwindSafe(|| {
        lbfgsb_impl(&checked_f, &checked_grad, x0, config)
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "L-BFGS-B callback".into(),
    })??;
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: "L-BFGS-B callback returned a non-finite value".into(),
        });
    }
    let bounds = if config.bounds.is_empty() {
        vec![Bound::unbounded(); x0.len()]
    } else {
        config.bounds.clone()
    };
    super::finalize_identity_kkt(
        &mut result,
        &[],
        &bounds,
        config.grad_tolerance * T::from_f64(100.0).unwrap(),
    )?;
    crate::types::validate_result_finite(&result)?;
    Ok(result)
}

fn lbfgsb_impl<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &LbfgsbConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let n = x0.len();
    let bounds = if config.bounds.is_empty() {
        vec![Bound::unbounded(); n]
    } else {
        if config.bounds.len() != n {
            return Err(OptimError::DimensionMismatch {
                expected: n,
                got: config.bounds.len(),
            });
        }
        config.bounds.clone()
    };

    // Project initial guess
    let mut x = x0.to_vec();
    project_bounds(&mut x, &bounds);

    let mut fx = f(&x);
    let mut g = grad_fn(&x);
    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;

    let mut s_history: VecDeque<Vec<T>> = VecDeque::with_capacity(config.memory);
    let mut y_history: VecDeque<Vec<T>> = VecDeque::with_capacity(config.memory);
    let mut rho_history: VecDeque<T> = VecDeque::with_capacity(config.memory);

    for iter in 0..config.max_iterations {
        // Projected gradient norm (infinity norm)
        let grad_norm = projected_gradient_norm(&x, &g, &bounds);

        if grad_norm < config.grad_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(g),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Scaling of the initial matrix B0 = theta*I (inverse of the two-loop
        // gamma); theta = y^T y / s^T y from the most recent correction pair.
        let theta = match (s_history.back(), y_history.back()) {
            (Some(s), Some(y)) => {
                let sy = T::vec_dot(s, y);
                let yy = T::vec_dot(y, y);
                if sy > T::zero() && yy > T::zero() {
                    yy / sy
                } else {
                    T::one()
                }
            }
            _ => T::one(),
        };

        // Compact (W, M) representation of the L-BFGS matrix; `None` when no
        // pairs are stored or the small factorisation is numerically singular
        // (in which case the model reduces to B = theta*I).
        let compact = CompactForm::new(&s_history, &y_history, theta);

        // 1. Generalised Cauchy point along the piecewise-linear path.
        let (x_c, active) = generalized_cauchy_point(&x, &g, &bounds, compact.as_ref(), theta);

        // Function and gradient at the Cauchy point (needed both for the
        // subspace line search and for the fallback decision).
        let f_c = f(&x_c);
        let g_c = grad_fn(&x_c);
        fun_evals += 1;
        grad_evals += 1;

        // 2. Subspace minimisation over the free variables.
        let free: Vec<usize> = (0..n).filter(|&i| !active[i]).collect();
        let mut accepted: Option<(Vec<T>, T, Vec<T>)> = None;
        if !free.is_empty() {
            let dc = sonium_solvers_core::vec_ops::vec_sub(&x_c, &x);
            let mut dbar = vec![T::zero(); n];
            subspace_direction(
                &g,
                &dc,
                &s_history,
                &y_history,
                &rho_history,
                &free,
                compact.as_ref(),
                theta,
                &mut dbar,
            );
            // 3. Projected backtracking line search from the Cauchy point.
            let gc_d = T::vec_dot(&g_c, &dbar);
            if gc_d < T::zero() {
                let (x_t, f_t, g_t, ls_f, ls_g, ok) =
                    projected_line_search(f, grad_fn, &x_c, &dbar, f_c, &g_c, &bounds);
                fun_evals += ls_f;
                grad_evals += ls_g;
                if ok {
                    accepted = Some((x_t, f_t, g_t));
                }
            }
        }

        // Fallbacks: accept the Cauchy point when it decreases f, otherwise
        // search along the Cauchy path from x (a descent path by construction).
        let (x_new, fx_new, g_new) = match accepted {
            Some(step) => step,
            None => {
                if f_c < fx {
                    (x_c, f_c, g_c)
                } else {
                    let dc = sonium_solvers_core::vec_ops::vec_sub(&x_c, &x);
                    let (x_t, f_t, g_t, ls_f, ls_g, ok) =
                        projected_line_search(f, grad_fn, &x, &dc, fx, &g, &bounds);
                    fun_evals += ls_f;
                    grad_evals += ls_g;
                    if ok {
                        (x_t, f_t, g_t)
                    } else {
                        return Ok(OptimResult {
                            x,
                            fun: fx,
                            grad: Some(g),
                            iterations: iter,
                            fun_evals,
                            grad_evals,
                            hess_evals: 0,
                            grad_norm,
                            converged: false,
                            status: OptimStatus::LineSearchFailed,
                            kkt: None,
                        });
                    }
                }
            }
        };

        // Function tolerance — only check if we actually made a step
        let s = sonium_solvers_core::vec_ops::vec_sub(&x_new, &x);
        let step_norm = T::vec_norm(&s);
        if step_norm > T::machine_epsilon()
            && (fx - fx_new).abs() < config.fun_tolerance * (T::one() + fx.abs())
        {
            return Ok(OptimResult {
                x: x_new,
                fun: fx_new,
                grad: Some(g_new),
                iterations: iter + 1,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: true,
                status: OptimStatus::FunctionTolerance,
                kkt: None,
            });
        }

        // Store correction pair
        let y = sonium_solvers_core::vec_ops::vec_sub(&g_new, &g);
        let sy = T::vec_dot(&s, &y);

        if config.memory > 0 && sy > T::machine_epsilon() * T::vec_norm(&s) * T::vec_norm(&y) {
            if s_history.len() >= config.memory {
                s_history.pop_front();
                y_history.pop_front();
                rho_history.pop_front();
            }
            rho_history.push_back(T::one() / sy);
            s_history.push_back(s);
            y_history.push_back(y);
        }

        x = x_new;
        fx = fx_new;
        g = g_new;
    }

    let grad_norm = projected_gradient_norm(&x, &g, &bounds);
    Ok(OptimResult {
        x,
        fun: fx,
        grad: Some(g),
        iterations: config.max_iterations,
        fun_evals,
        grad_evals,
        hess_evals: 0,
        grad_norm,
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
}

/// Compute the projected gradient infinity norm.
fn projected_gradient_norm<T: RealScalar>(x: &[T], g: &[T], bounds: &[Bound<T>]) -> T {
    let mut norm = T::zero();
    for (i, (&xi, &gi)) in x.iter().zip(g.iter()).enumerate() {
        let pg = if let Some(lb) = bounds[i].lower {
            if xi <= lb && gi > T::zero() {
                T::zero()
            } else {
                gi
            }
        } else {
            gi
        };
        let pg = if let Some(ub) = bounds[i].upper {
            if xi >= ub && pg < T::zero() {
                T::zero()
            } else {
                pg
            }
        } else {
            pg
        };
        if pg.abs() > norm {
            norm = pg.abs();
        }
    }
    norm
}

/// Compact representation of the L-BFGS Hessian approximation
/// `B = theta*I - W M W^T` with `W = [Y, theta*S]` (n x 2m) and
/// `M = A^{-1}`, `A = [[-D, L^T], [L, theta*S^T S]]` where `S^T Y = L + D + R`
/// (Byrd-Nocedal-Schnabel; Nocedal & Wright, eq. (16.74)).
///
/// Only the `2m x 2m` LU factorisation of `A` is stored; rows of `W` are
/// gathered on demand from the pair history.
struct CompactForm<'a, T: RealScalar> {
    s_hist: &'a VecDeque<Vec<T>>,
    y_hist: &'a VecDeque<Vec<T>>,
    theta: T,
    m: usize,
    /// LU factors of `A` (row-major, unit lower triangular implicit).
    lu: Vec<T>,
    /// Row permutation from partial pivoting: row `i` of `P A` is row
    /// `perm[i]` of `A`.
    perm: Vec<usize>,
}

impl<'a, T: RealScalar> CompactForm<'a, T> {
    /// Build the compact form; returns `None` when no pairs are stored or `A`
    /// is numerically singular.
    fn new(s_hist: &'a VecDeque<Vec<T>>, y_hist: &'a VecDeque<Vec<T>>, theta: T) -> Option<Self> {
        let m = s_hist.len();
        if m == 0 {
            return None;
        }
        let k = 2 * m;
        let mut a = vec![T::zero(); k * k];
        for i in 0..m {
            // Strictly lower part L of S^T Y (and its transpose).
            for (j, yj) in y_hist.iter().enumerate().take(i) {
                let lij = T::vec_dot(&s_hist[i], yj);
                a[(m + i) * k + j] = lij;
                a[j * k + (m + i)] = lij;
            }
            // Diagonal D of S^T Y.
            a[i * k + i] = -T::vec_dot(&s_hist[i], &y_hist[i]);
            // theta * S^T S block.
            for (j, sj) in s_hist.iter().enumerate() {
                a[(m + i) * k + (m + j)] = theta * T::vec_dot(&s_hist[i], sj);
            }
        }
        let mut perm = vec![0_usize; k];
        if !lu_factor(&mut a, k, &mut perm) {
            return None;
        }
        Some(Self {
            s_hist,
            y_hist,
            theta,
            m,
            lu: a,
            perm,
        })
    }

    /// Row `i` of `W = [Y, theta*S]` into `out` (length 2m).
    fn w_row(&self, i: usize, out: &mut [T]) {
        for j in 0..self.m {
            out[j] = self.y_hist[j][i];
            out[self.m + j] = self.theta * self.s_hist[j][i];
        }
    }

    /// `W^T v` (length 2m).
    fn wt_vec(&self, v: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); 2 * self.m];
        for j in 0..self.m {
            out[j] = T::vec_dot(&self.y_hist[j], v);
            out[self.m + j] = self.theta * T::vec_dot(&self.s_hist[j], v);
        }
        out
    }

    /// `M v = A^{-1} v` via the LU factorisation.
    fn apply_m(&self, v: &[T]) -> Vec<T> {
        lu_solve(&self.lu, 2 * self.m, &self.perm, v)
    }

    /// `B v = theta*v - W (M (W^T v))`.
    #[cfg(test)]
    fn apply_b(&self, v: &[T]) -> Vec<T> {
        let z = self.apply_m(&self.wt_vec(v));
        let mut out = vec![T::zero(); v.len()];
        let mut w_row = vec![T::zero(); 2 * self.m];
        for (i, oi) in out.iter_mut().enumerate() {
            self.w_row(i, &mut w_row);
            *oi = self.theta * v[i] - T::vec_dot(&w_row, &z);
        }
        out
    }
}

/// LU-factorise the `k x k` row-major matrix `a` with partial pivoting.
/// Returns `false` when a pivot is numerically zero.
fn lu_factor<T: RealScalar>(a: &mut [T], k: usize, perm: &mut [usize]) -> bool {
    let mut amax = T::zero();
    for ai in a.iter() {
        if ai.abs() > amax {
            amax = ai.abs();
        }
    }
    let tol = amax * T::from_f64(1e-12).unwrap();
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    for col in 0..k {
        let mut piv = col;
        let mut pmax = a[col * k + col].abs();
        for (r, row) in a.chunks_exact(k).enumerate().skip(col + 1) {
            let v = row[col].abs();
            if v > pmax {
                pmax = v;
                piv = r;
            }
        }
        if pmax <= tol {
            return false;
        }
        if piv != col {
            for j in 0..k {
                a.swap(col * k + j, piv * k + j);
            }
            perm.swap(col, piv);
        }
        let inv = T::one() / a[col * k + col];
        for r in col + 1..k {
            let f = a[r * k + col] * inv;
            a[r * k + col] = f;
            for j in col + 1..k {
                let pivot_row = a[col * k + j];
                a[r * k + j] -= f * pivot_row;
            }
        }
    }
    true
}

/// Solve `A x = b` given the LU factors and permutation from [`lu_factor`].
fn lu_solve<T: RealScalar>(lu: &[T], k: usize, perm: &[usize], b: &[T]) -> Vec<T> {
    // Apply the row permutation: P b.
    let mut x: Vec<T> = perm.iter().map(|&p| b[p]).collect();
    // Forward substitution (unit lower triangular).
    for i in 0..k {
        let mut acc = x[i];
        for j in 0..i {
            acc -= lu[i * k + j] * x[j];
        }
        x[i] = acc;
    }
    // Back substitution (upper triangular).
    for i in (0..k).rev() {
        let mut acc = x[i];
        for j in i + 1..k {
            acc -= lu[i * k + j] * x[j];
        }
        x[i] = acc / lu[i * k + i];
    }
    x
}

/// Generalised Cauchy point: first local minimiser of the quadratic model
/// `m(x + d) = g^T d + 1/2 d^T B d` along the piecewise-linear path
/// `x(t) = P(x - t g)`, together with the active set at the Cauchy point.
///
/// Follows the path segment by segment. On the current segment the moving
/// coordinates `M` (breakpoints not yet crossed) satisfy
///   F'  = -q + theta * t * q + c^T v,
///   F'' = theta * q - v^T M v,
/// with `q = sum_M g_i^2`, `v = sum_M g_i w_i` (`w_i` = row `i` of `W`) and
/// `c = M W^T d`; all three quantities are updated in O(m^2) when a
/// breakpoint is crossed.
fn generalized_cauchy_point<T: RealScalar>(
    x: &[T],
    g: &[T],
    bounds: &[Bound<T>],
    compact: Option<&CompactForm<T>>,
    theta: T,
) -> (Vec<T>, Vec<bool>) {
    let n = x.len();
    let k = compact.map_or(0, |c| 2 * c.m);
    let inf = T::infinity();

    // Breakpoints t_i at which x_i(t) hits a bound. t_i = 0 when x_i already
    // sits at the bound its negative gradient points into; t_i = inf when the
    // coordinate never hits a bound (also when g_i = 0).
    let mut t = vec![inf; n];
    for i in 0..n {
        if g[i] > T::zero() {
            if let Some(lb) = bounds[i].lower {
                t[i] = (x[i] - lb) / g[i];
            }
        } else if g[i] < T::zero()
            && let Some(ub) = bounds[i].upper
        {
            t[i] = (x[i] - ub) / g[i];
        }
        if t[i] < T::zero() {
            t[i] = T::zero();
        }
    }

    // Finite, strictly positive breakpoints in increasing order.
    let mut order: Vec<usize> = (0..n).filter(|&i| t[i] > T::zero() && t[i] < inf).collect();
    order.sort_by(|&a, &b| t[a].partial_cmp(&t[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Moving-set aggregates q and v.
    let mut q = T::zero();
    let mut v = vec![T::zero(); k];
    let mut w_row = vec![T::zero(); k];
    for i in 0..n {
        if t[i] > T::zero() {
            q += g[i] * g[i];
            if let Some(cf) = compact {
                cf.w_row(i, &mut w_row);
                for (vj, &wj) in v.iter_mut().zip(w_row.iter()) {
                    *vj += g[i] * wj;
                }
            }
        }
    }

    let mut c = vec![T::zero(); k];
    let mut active = vec![false; n];
    for (i, &ti) in t.iter().enumerate() {
        if ti == T::zero() {
            active[i] = true;
        }
    }

    let mut t_star = T::zero();
    let mut t_prev = T::zero();
    for &b in &order {
        // Model derivatives at t_prev.
        let (fp, fpp) = if let Some(cf) = compact {
            let z = cf.apply_m(&v);
            let cv = T::vec_dot(&c, &v);
            let vz = T::vec_dot(&v, &z);
            (-q + theta * t_prev * q + cv, theta * q - vz)
        } else {
            (-q + theta * t_prev * q, theta * q)
        };
        if fp >= T::zero() {
            // Model is already increasing: minimiser at t_prev.
            break;
        }
        let dt = t[b] - t_prev;
        if fpp > T::zero() {
            let dt_star = -fp / fpp;
            if dt_star < dt {
                // Unconstrained minimiser inside this segment.
                t_star = t_prev + dt_star;
                break;
            }
        }
        // Cross breakpoint t[b]: coordinate b becomes fixed.
        if let Some(cf) = compact {
            let z = cf.apply_m(&v);
            for (cj, &zj) in c.iter_mut().zip(z.iter()) {
                *cj -= dt * zj;
            }
            cf.w_row(b, &mut w_row);
            for (vj, &wj) in v.iter_mut().zip(w_row.iter()) {
                *vj -= g[b] * wj;
            }
        }
        q -= g[b] * g[b];
        active[b] = true;
        t_prev = t[b];
        t_star = t_prev;
    }

    let mut x_c: Vec<T> = x
        .iter()
        .zip(g.iter())
        .map(|(&xi, &gi)| xi - t_star * gi)
        .collect();
    project_bounds(&mut x_c, bounds);
    (x_c, active)
}

/// Subspace step on the free variables: approximately solve
/// `Z^T B Z d_F = -Z^T (g + B (x^c - x))` with the two-loop L-BFGS recursion
/// restricted to the free block. The coupling term `B (x^c - x)` is evaluated
/// exactly with the compact form when available. The result (negated) is
/// scattered into `dbar`, which stays zero on the active set.
#[allow(clippy::too_many_arguments)]
fn subspace_direction<T: RealScalar>(
    g: &[T],
    dc: &[T],
    s_hist: &VecDeque<Vec<T>>,
    y_hist: &VecDeque<Vec<T>>,
    rho_hist: &VecDeque<T>,
    free: &[usize],
    compact: Option<&CompactForm<T>>,
    theta: T,
    dbar: &mut [T],
) {
    let nf = free.len();
    // Coupling c2 = M W^T (x^c - x); (B dc)_i = theta*dc_i - w_i^T c2.
    let c2 = compact.map(|cf| cf.apply_m(&cf.wt_vec(dc)));
    let mut w_row = vec![T::zero(); compact.map_or(0, |c| 2 * c.m)];

    let mut q = vec![T::zero(); nf];
    for (j, &i) in free.iter().enumerate() {
        let mut bdc = theta * dc[i];
        if let (Some(cf), Some(c2v)) = (compact, c2.as_ref()) {
            cf.w_row(i, &mut w_row);
            bdc -= T::vec_dot(&w_row, c2v);
        }
        q[j] = g[i] + bdc;
    }

    // Two-loop recursion with the pairs restricted to the free block.
    let m = s_hist.len();
    let mut alphas = vec![T::zero(); m];
    for i in (0..m).rev() {
        let mut dotp = T::zero();
        for (j, &fi) in free.iter().enumerate() {
            dotp += s_hist[i][fi] * q[j];
        }
        alphas[i] = rho_hist[i] * dotp;
        for (j, &fi) in free.iter().enumerate() {
            q[j] -= alphas[i] * y_hist[i][fi];
        }
    }
    if m > 0 {
        let sy = T::vec_dot(&s_hist[m - 1], &y_hist[m - 1]);
        let yy = T::vec_dot(&y_hist[m - 1], &y_hist[m - 1]);
        if yy > T::zero() {
            let gamma = sy / yy;
            for qi in q.iter_mut() {
                *qi *= gamma;
            }
        }
    }
    for i in 0..m {
        let mut dotp = T::zero();
        for (j, &fi) in free.iter().enumerate() {
            dotp += y_hist[i][fi] * q[j];
        }
        let beta = rho_hist[i] * dotp;
        for (j, &fi) in free.iter().enumerate() {
            q[j] += (alphas[i] - beta) * s_hist[i][fi];
        }
    }
    for (j, &fi) in free.iter().enumerate() {
        dbar[fi] = -q[j];
    }
}

/// Projected backtracking Armijo line search along `x(alpha) = P(x + alpha*d)`.
/// Returns the accepted point plus evaluation counts and a success flag; on
/// failure returns the base point unchanged.
fn projected_line_search<T, F, G>(
    f: &F,
    grad_fn: &G,
    x: &[T],
    d: &[T],
    fx: T,
    g: &[T],
    bounds: &[Bound<T>],
) -> (Vec<T>, T, Vec<T>, usize, usize, bool)
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let c1 = T::from_f64(1e-4).unwrap();
    let rho = T::from_f64(0.5).unwrap();
    let dg = T::vec_dot(g, d);
    if dg >= T::zero() {
        return (x.to_vec(), fx, g.to_vec(), 0, 0, false);
    }
    let mut fun_evals = 0;

    // Compute maximum feasible step
    let alpha_max = max_feasible_step(x, d, bounds);
    let mut alpha = T::min(T::one(), alpha_max);

    for _ in 0..40 {
        let mut x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, alpha);
        project_bounds(&mut x_new, bounds);
        let fx_new = f(&x_new);
        fun_evals += 1;

        if fx_new <= fx + c1 * alpha * dg {
            let g_new = grad_fn(&x_new);
            return (x_new, fx_new, g_new, fun_evals, 1, true);
        }

        alpha *= rho;
    }

    (x.to_vec(), fx, g.to_vec(), fun_evals, 0, false)
}

/// Compute the maximum step size that keeps all variables within bounds.
fn max_feasible_step<T: RealScalar>(x: &[T], d: &[T], bounds: &[Bound<T>]) -> T {
    let mut alpha_max = T::from_f64(1e20).unwrap();
    for (i, (&xi, &di)) in x.iter().zip(d.iter()).enumerate() {
        if di > T::machine_epsilon() {
            if let Some(ub) = bounds[i].upper {
                let a = (ub - xi) / di;
                if a < alpha_max {
                    alpha_max = a;
                }
            }
        } else if di < -T::machine_epsilon()
            && let Some(lb) = bounds[i].lower
        {
            let a = (lb - xi) / di;
            if a < alpha_max {
                alpha_max = a;
            }
        }
    }
    alpha_max
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sonium_solvers_core::traits::ComplexField;

    /// Full two-loop recursion on the pair history: returns `-H g` where `H`
    /// is the inverse of the compact-form L-BFGS matrix. Used to cross-check
    /// the compact representation.
    fn two_loop_direction<T: RealScalar>(
        g: &[T],
        s_hist: &VecDeque<Vec<T>>,
        y_hist: &VecDeque<Vec<T>>,
        rho_hist: &VecDeque<T>,
    ) -> Vec<T> {
        let m = s_hist.len();
        let mut q = g.to_vec();
        let mut alphas = vec![T::zero(); m];

        for i in (0..m).rev() {
            alphas[i] = rho_hist[i] * T::vec_dot(&s_hist[i], &q);
            T::vec_axpy(-alphas[i], &y_hist[i], &mut q);
        }

        if m > 0 {
            let sy = T::vec_dot(&s_hist[m - 1], &y_hist[m - 1]);
            let yy = T::vec_dot(&y_hist[m - 1], &y_hist[m - 1]);
            if yy > T::zero() {
                let gamma = sy / yy;
                sonium_solvers_core::vec_ops::vec_mapv_inplace(&mut q, |qi| qi * gamma);
            }
        }

        for i in 0..m {
            let beta = rho_hist[i] * T::vec_dot(&y_hist[i], &q);
            T::vec_axpy(alphas[i] - beta, &s_hist[i], &mut q);
        }

        q.iter().map(|&qi| -qi).collect::<Vec<_>>()
    }

    #[test]
    fn test_compact_form_matches_two_loop() {
        // The compact form B = theta*I - W M W^T and the two-loop recursion
        // must satisfy B (H v) = v for arbitrary v.
        let s_vecs = [
            vec![1.0_f64, 0.5, -0.3, 0.2],
            vec![0.4_f64, -0.8, 0.6, 0.1],
            vec![-0.2_f64, 0.3, 0.9, -0.5],
        ];
        let y_vecs = [
            vec![2.0_f64, 0.7, -0.4, 0.3],
            vec![0.6_f64, -1.5, 1.1, 0.2],
            vec![-0.3_f64, 0.5, 1.7, -0.9],
        ];
        let mut s_hist: VecDeque<Vec<f64>> = VecDeque::new();
        let mut y_hist: VecDeque<Vec<f64>> = VecDeque::new();
        let mut rho_hist: VecDeque<f64> = VecDeque::new();
        for (s, y) in s_vecs.iter().zip(y_vecs.iter()) {
            let sy = f64::vec_dot(s, y);
            assert!(sy > 0.0, "test pairs must satisfy curvature");
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(s.clone());
            y_hist.push_back(y.clone());
        }
        let sy = f64::vec_dot(s_hist.back().unwrap(), y_hist.back().unwrap());
        let yy = f64::vec_dot(y_hist.back().unwrap(), y_hist.back().unwrap());
        let theta = yy / sy;
        let compact = CompactForm::new(&s_hist, &y_hist, theta).expect("nonsingular");

        for v in [vec![1.0_f64, -2.0, 0.5, 3.0], vec![0.1_f64, 0.2, -0.7, 1.3]] {
            // H v from the two-loop recursion (direction for g = -v).
            let hv = two_loop_direction(
                &v.iter().map(|&vi| -vi).collect::<Vec<_>>(),
                &s_hist,
                &y_hist,
                &rho_hist,
            );
            let bhv = compact.apply_b(&hv);
            for (a, b) in bhv.iter().zip(v.iter()) {
                assert_relative_eq!(a, b, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_lbfgsb_unconstrained_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = LbfgsbConfig::default();
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_lbfgsb_bounded_rosenbrock() {
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let g = |x: &[f64]| {
            let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dy = 200.0 * (x[1] - x[0] * x[0]);
            sonium_solvers_core::array![dx, dy]
        };
        let x0 = sonium_solvers_core::array![0.5, 0.5];
        let config = LbfgsbConfig {
            bounds: vec![Bound::both(0.0, 2.0), Bound::both(0.0, 2.0)],
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_lbfgsb_active_bound() {
        // min x^2 + y^2 subject to x >= 1, y >= 2
        // min at (1, 2), f = 5
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![3.0, 5.0];
        let config = LbfgsbConfig {
            bounds: vec![Bound::lower(1.0), Bound::lower(2.0)],
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lbfgsb_bounds_length_mismatch_returns_error() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![1.0, 2.0, 3.0]; // dim = 3
        let config = LbfgsbConfig {
            bounds: vec![Bound::unbounded(); 2], // dim = 2 -- mismatch
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_lbfgsb_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let g = |x: &[f32]| x.iter().map(|&xi| 2.0_f32 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: LbfgsbConfig<f32> = LbfgsbConfig::default();
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-3, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1]={}", result.x[1]);
    }

    #[test]
    fn test_lbfgsb_all_bounds_active() {
        // min x^2 + y^2 s.t. x >= 3, y >= 4
        // Unconstrained min at (0,0), but bounds push solution to (3, 4) with f = 25.
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 6.0];
        let config = LbfgsbConfig {
            bounds: vec![Bound::lower(3.0), Bound::lower(4.0)],
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 4.0, epsilon = 1e-4);
        assert_relative_eq!(result.fun, 25.0, epsilon = 1e-3);
    }

    #[test]
    fn test_lbfgsb_upper_bounds() {
        // min (x-5)^2 + (y-5)^2 s.t. x <= 2, y <= 3
        // Unconstrained min at (5,5), but upper bounds push solution to (2, 3).
        let f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
        let g = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 5.0), 2.0 * (x[1] - 5.0)];
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = LbfgsbConfig {
            bounds: vec![Bound::upper(2.0), Bound::upper(3.0)],
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lbfgsb_quadratic_optimum_at_bound() {
        // min (x+2)^2 + 3(y-1)^2 s.t. x >= 0: constrained optimum (0, 1), f = 4.
        // The Cauchy point identifies the active set immediately and the
        // subspace step solves the remaining 1-D quadratic exactly.
        let f = |x: &[f64]| (x[0] + 2.0).powi(2) + 3.0 * (x[1] - 1.0).powi(2);
        let g = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] + 2.0), 6.0 * (x[1] - 1.0)];
        let x0 = sonium_solvers_core::array![3.0, 5.0];
        let config = LbfgsbConfig {
            bounds: vec![Bound::lower(0.0), Bound::unbounded()],
            grad_tolerance: 1e-10,
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(
            result.iterations <= 4,
            "bound-active quadratic should take few iterations, got {}",
            result.iterations
        );
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.fun, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lbfgsb_quadratic_interior_optimum() {
        // min (x-1)^2 + 4(y+2)^2 with wide bounds: interior optimum (1, -2).
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2);
        let g = |x: &[f64]| sonium_solvers_core::array![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)];
        let x0 = sonium_solvers_core::array![5.0, 5.0];
        let config = LbfgsbConfig {
            bounds: vec![Bound::both(-10.0, 10.0), Bound::both(-10.0, 10.0)],
            grad_tolerance: 1e-10,
            fun_tolerance: 0.0, // converge on the projected gradient, not fun change
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(
            result.iterations <= 12,
            "interior quadratic should take few iterations, got {}",
            result.iterations
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], -2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_lbfgsb_rosenbrock_optimum_at_bound() {
        // Rosenbrock with x <= 0.5: unconstrained minimiser (1, 1) is cut off;
        // the constrained optimum is (0.5, 0.25) with f = 0.25 and the upper
        // bound on x active.
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let g = |x: &[f64]| {
            let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dy = 200.0 * (x[1] - x[0] * x[0]);
            sonium_solvers_core::array![dx, dy]
        };
        let x0 = sonium_solvers_core::array![-1.0, 1.0];
        let config = LbfgsbConfig {
            bounds: vec![Bound::both(-2.0, 0.5), Bound::both(-2.0, 2.0)],
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}, x: {:?}",
            result.status, result.fun, result.x
        );
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.25, epsilon = 1e-3);
        assert_relative_eq!(result.fun, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_lbfgsb_bound_active_iteration_count() {
        // Bound-active benchmark: half of the 50 variables want to sit outside
        // the box. The generalised Cauchy point identifies the whole active
        // set in one step; the old projected two-loop approach needed
        // O(100) iterations here.
        let n = 50;
        let c: Vec<f64> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    -3.0 - i as f64 * 0.1
                } else {
                    0.4
                }
            })
            .collect();
        let f = |x: &[f64]| {
            x.iter()
                .zip(c.iter())
                .map(|(&xi, &ci)| (xi - ci).powi(2))
                .sum()
        };
        let g = |x: &[f64]| {
            x.iter()
                .zip(c.iter())
                .map(|(&xi, &ci)| 2.0 * (xi - ci))
                .collect::<Vec<_>>()
        };
        let x0 = vec![0.0; n];
        let config = LbfgsbConfig {
            bounds: vec![Bound::both(-1.0, 1.0); n],
            grad_tolerance: 1e-8,
            ..Default::default()
        };
        let result = lbfgsb(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(
            result.iterations <= 15,
            "expected fast convergence on bound-active problem, got {} iterations",
            result.iterations
        );
        for (i, &xi) in result.x.iter().enumerate() {
            let expected = if i % 2 == 0 { -1.0 } else { 0.4 };
            assert_relative_eq!(xi, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn lbfgsb_maps_callback_panics_to_typed_error() {
        let objective = |_x: &[f64]| -> f64 { panic!("objective failure") };
        let gradient = |_x: &[f64]| vec![0.0];
        let error = lbfgsb(&objective, &gradient, &[0.0], &LbfgsbConfig::default())
            .expect_err("callback panic must not escape the public API");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));
    }
}
