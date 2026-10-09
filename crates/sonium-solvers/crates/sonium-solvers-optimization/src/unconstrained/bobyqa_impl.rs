//! BOBYQA: model-based trust-region derivative-free optimisation with bound
//! constraints, after Powell (2009), *"The BOBYQA algorithm for bound
//! constrained optimization without derivatives"* (see also Zhang's PRIMA
//! reformulation, 2014/2023).
//!
//! Unlike the direction-set [`powell`](super::powell) method, BOBYQA builds an
//! underdetermined quadratic interpolation model of the objective on `m`
//! interpolation points (default `m = 2n + 1`) and minimises that model inside
//! a trust region, so it needs far fewer objective evaluations on expensive
//! black-box objectives (e.g. BEM/FEM solves) than any line-search-based
//! derivative-free method.
//!
//! # Algorithm outline
//!
//! 1. **Interpolation model.** `q(x) = c + gᵀ(x - x_base) + ½ (x - x_base)ᵀ H
//!    (x - x_base)` interpolates `f` on `m ≤ (n+1)(n+2)/2` points. Because `m`
//!    is smaller than the number of free parameters, the Hessian is chosen by
//!    the *minimum-Frobenius-norm update*: when a point is replaced, `H` moves
//!    as little as possible subject to the interpolation conditions. This is
//!    solved exactly via the `(m + n + 1) × (m + n + 1)` symmetric KKT system
//!    `[[Q, A], [Aᵀ, 0]]` with `Q_kj = ½ (s_k · s_j)²`, `A = [1, S]`, where
//!    `s_k = (y_k - x_base)/δ` are the interpolation points scaled by the
//!    trust-region radius (scaling keeps the system conditioned independently
//!    of `δ`).
//! 2. **Trust-region subproblem.** The model is minimised over
//!    `{x_base + d : ‖d‖ ≤ Δ, xl ≤ x_base + d ≤ xu}` by a projected truncated
//!    conjugate-gradient iteration with active-set freezing at the bounds
//!    (the shared `util::trust_region_subproblem` solvers do not handle box
//!    constraints, hence a local solver).
//! 3. **Acceptance.** Actual/predicted reduction ratio with standard grow /
//!    shrink rules for `Δ`. The new point always enters the interpolation set:
//!    the evicted point maximises `|ℓ_j(x_new)| · max(1, (dist_j/Δ)³)` over
//!    `j ≠ k_opt`, where `ℓ_j` are the Lagrange functions of the interpolation
//!    system (Powell's weighted criterion).
//! 4. **Geometry steps.** On unsuccessful iterations (or model steps shorter
//!    than `ρ/2`), if the farthest interpolation point is more than `2Δ` from
//!    `x_base` the model is deemed not adequately poised and that point is
//!    replaced by an approximate maximiser of `|ℓ_t|` inside the trust region.
//! 5. **Termination.** `ρ ≤ rhoend`, `max_fun_evals`, or a model-gradient
//!    surrogate (`‖projected gradient of q at x_base‖ < gradient_tolerance`
//!    while the geometry is adequately poised and `Δ ≤ 4ρ`).
//!
//! # Simplifications vs Powell's Fortran BOBYQA
//!
//! - The KKT interpolation system is re-factorised (dense LU with partial
//!   pivoting) on every update instead of rank-1 updating the factors of the
//!   interpolation matrix. This is `O((m+n)³)` per iteration — irrelevant next
//!   to an expensive objective evaluation.
//! - The bound-constrained subproblem uses projected CG + active-set freezing
//!   instead of Powell's exact `trsbox` active-set iteration.
//! - Geometry steps maximise `|ℓ_t|` over a small candidate set (the `2n`
//!   coordinate points `x_base ± δ e_i` plus the normalised gradient of `ℓ_t`
//!   at `x_base`) instead of Powell's two-stage Cauchy iteration.
//! - No alternative/backup model steps and no explicit KKT residual checks.
//! - `ρ` is reduced by the fixed factor `1/10`; `x_base` is always the best
//!   known point (Powell keeps `x_base = x0` for long stretches).
//! - Variables with `lower == upper` are rejected rather than eliminated;
//!   fix them outside the optimisation.

use crate::types::{
    Bound, Bounds, OptimError, OptimResult, OptimStatus, RealScalar, project_bounds,
};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for [`bobyqa`].
#[derive(Debug, Clone)]
pub struct BobyqaConfig<T: RealScalar> {
    /// Variable bounds. Empty means unconstrained. `lower` must be strictly
    /// less than `upper` when both are given.
    pub bounds: Bounds<T>,
    /// Number of interpolation points, `n+2 ≤ npt ≤ (n+1)(n+2)/2`.
    /// `None` uses Powell's default `2n + 1`.
    pub npt: Option<usize>,
    /// Initial trust-region radius. `None` uses `0.2 × min(bound range)` if
    /// any variable is bounded, else `1.0`. Clamped to half the smallest
    /// bounded range.
    pub rhobeg: Option<T>,
    /// Final trust-region radius; the run ends (converged) once `ρ ≤ rhoend`.
    pub rhoend: T,
    /// Maximum number of objective evaluations.
    pub max_fun_evals: usize,
    /// Maximum number of trust-region iterations.
    pub max_iterations: usize,
    /// Termination tolerance on the infinity norm of the model gradient at
    /// the best point (projected onto the feasible box). Only active when the
    /// interpolation geometry is adequately poised.
    pub gradient_tolerance: T,
}

impl Default for BobyqaConfig<f64> {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            npt: None,
            rhobeg: None,
            rhoend: 1e-6,
            max_fun_evals: 10_000,
            max_iterations: 10_000,
            gradient_tolerance: 1e-8,
        }
    }
}

impl Default for BobyqaConfig<f32> {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            npt: None,
            rhobeg: None,
            rhoend: 1e-3,
            max_fun_evals: 10_000,
            max_iterations: 10_000,
            gradient_tolerance: 1e-4,
        }
    }
}

/// Minimise `f(x)` subject to bound constraints using the BOBYQA algorithm.
///
/// Derivative-free; only objective values are used. See the module
/// documentation for the algorithm outline and the simplifications relative
/// to Powell's reference implementation.
pub fn bobyqa<T, F>(f: &F, x0: &[T], config: &BobyqaConfig<T>) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    super::validate_objective_start(f, x0)?;
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bobyqa_impl(f, x0, config)))
            .map_err(|_| OptimError::CallbackPanicked {
                iteration: 0,
                stage: "BOBYQA callback".into(),
            })??;
    super::validate_result(&result)?;
    Ok(result)
}

fn bobyqa_impl<T, F>(
    f: &F,
    x0: &[T],
    config: &BobyqaConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "bobyqa requires at least one variable".to_string(),
        });
    }
    if let Some(index) = x0.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("initial point is non-finite at index {index}"),
        });
    }
    if config.max_fun_evals == 0 || config.max_iterations == 0 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "max_fun_evals and max_iterations must be greater than zero".to_string(),
        });
    }
    if !config.rhoend.is_finite()
        || config.rhoend <= T::zero()
        || !config.gradient_tolerance.is_finite()
        || config.gradient_tolerance <= T::zero()
    {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "rhoend and gradient_tolerance must be finite and positive".to_string(),
        });
    }

    // ---- Bounds -----------------------------------------------------------
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
    let mut xl = vec![T::neg_infinity(); n];
    let mut xu = vec![T::infinity(); n];
    for (i, b) in bounds.iter().enumerate() {
        if let Some(lb) = b.lower {
            if !lb.is_finite() {
                return Err(OptimError::NumericalError {
                    iteration: 0,
                    message: format!("lower bound at index {i} is non-finite"),
                });
            }
            xl[i] = lb;
        }
        if let Some(ub) = b.upper {
            if !ub.is_finite() {
                return Err(OptimError::NumericalError {
                    iteration: 0,
                    message: format!("upper bound at index {i} is non-finite"),
                });
            }
            xu[i] = ub;
        }
        if xl[i] >= xu[i] {
            return Err(OptimError::InvalidBounds {
                index: i,
                lower: xl[i].to_f64().unwrap_or(f64::NEG_INFINITY),
                upper: xu[i].to_f64().unwrap_or(f64::INFINITY),
            });
        }
    }

    // ---- Configuration ----------------------------------------------------
    let npt_default = n
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| OptimError::NumericalError {
            iteration: 0,
            message: "dimension overflows the default interpolation-point count".to_string(),
        })?;
    let npt = config.npt.unwrap_or(npt_default);
    let npt_max = n
        .checked_add(1)
        .and_then(|value| value.checked_mul(n.checked_add(2)?))
        .map(|value| value / 2)
        .ok_or_else(|| OptimError::NumericalError {
            iteration: 0,
            message: "dimension overflows the interpolation-point limit".to_string(),
        })?;
    let npt_min = n.checked_add(2).ok_or_else(|| OptimError::NumericalError {
        iteration: 0,
        message: "dimension overflows the minimum interpolation-point count".to_string(),
    })?;
    if npt < npt_min || npt > npt_max {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("npt must be in [{npt_min}, {npt_max}], got {npt}"),
        });
    }
    if config.rhoend <= T::zero() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "rhoend must be positive".to_string(),
        });
    }
    let mut rhobeg = match config.rhobeg {
        Some(r) => r,
        None => {
            let mut r = T::one();
            for i in 0..n {
                if xl[i].is_finite() && xu[i].is_finite() {
                    r = T::min(r, T::from_f64(0.2).unwrap() * (xu[i] - xl[i]));
                }
            }
            r
        }
    };
    // Clamp so each coordinate's initial stencil fits inside its range.
    for i in 0..n {
        if xl[i].is_finite() && xu[i].is_finite() {
            rhobeg = T::min(rhobeg, T::from_f64(0.5).unwrap() * (xu[i] - xl[i]));
        }
    }
    if !rhobeg.is_finite() || rhobeg <= T::zero() || rhobeg <= config.rhoend {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!(
                "rhobeg ({:?}) must exceed rhoend ({:?})",
                rhobeg, config.rhoend
            ),
        });
    }

    let mut x0 = x0.to_vec();
    project_bounds(&mut x0, &bounds);

    // ---- Initial interpolation set ----------------------------------------
    let mut pts: Vec<Vec<T>> = vec![x0.clone()];
    // Per-coordinate stencil pair, shifted inward when x0 sits at a bound.
    let mut pair_pts = Vec::with_capacity(n);
    for i in 0..n {
        let mut a = x0[i] - rhobeg;
        let mut b = x0[i] + rhobeg;
        if a < xl[i] {
            a = xl[i];
            b = T::min(xl[i] + T::from_f64(2.0).unwrap() * rhobeg, xu[i]);
        }
        if b > xu[i] {
            b = xu[i];
            a = T::max(xu[i] - T::from_f64(2.0).unwrap() * rhobeg, xl[i]);
        }
        pair_pts.push((a, b));
    }
    let dedup_tol = T::from_f64(1e-6).unwrap() * rhobeg;
    let push_distinct = |pts: &mut Vec<Vec<T>>, cand: Vec<T>| {
        let distinct = pts
            .iter()
            .all(|p| T::vec_norm(&sonium_solvers_core::vec_ops::vec_sub(p, &cand)) > dedup_tol);
        if distinct {
            pts.push(cand);
        }
    };
    // Powell's layout: x0, then the "+rho" side for each coordinate, then the
    // "-rho" side, so npt = n + 2 still gets a full one-sided stencil.
    for i in 0..n {
        if pts.len() >= npt {
            break;
        }
        let mut p = x0.clone();
        p[i] = pair_pts[i].1;
        push_distinct(&mut pts, p);
    }
    for i in 0..n {
        if pts.len() >= npt {
            break;
        }
        let mut p = x0.clone();
        p[i] = pair_pts[i].0;
        push_distinct(&mut pts, p);
    }
    // Fill up to npt with diagonal points (only reached for npt > 2n + 1 or
    // after bound-induced duplicates).
    'fill: for i in 0..n {
        for j in (i + 1)..n {
            for &(si, sj) in &[(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                if pts.len() >= npt {
                    break 'fill;
                }
                let mut p = x0.clone();
                p[i] += T::from_f64(si).unwrap() * rhobeg;
                p[j] += T::from_f64(sj).unwrap() * rhobeg;
                project_bounds(&mut p, &bounds);
                push_distinct(&mut pts, p);
            }
        }
    }
    let m = pts.len();

    let mut fval: Vec<T> = Vec::with_capacity(m);
    for (k, p) in pts.iter().enumerate() {
        let v = f(p);
        if !v.is_finite() {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: if k == 0 {
                    "objective is not finite at x0".to_string()
                } else {
                    format!("objective is not finite at interpolation point {k}")
                },
            });
        }
        fval.push(v);
    }
    let mut fun_evals = m;

    let kopt = fval
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut st = State {
        n,
        pts,
        fval,
        kopt,
        model: QuadraticModel::zeros(x0),
        xl,
        xu,
        delta: rhobeg,
        rho: rhobeg,
    };
    // The model is expanded around the best known point.
    let base = st.pts[st.kopt].clone();
    st.model.shift_base(&base);
    if !update_model(&mut st) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "initial interpolation system is singular".to_string(),
        });
    }

    let rhoend = config.rhoend;
    let mut iterations = 0usize;
    let status;
    let mut converged = false;

    // ---- Main loop ----------------------------------------------------------
    loop {
        if st.rho <= rhoend {
            status = OptimStatus::StepTolerance;
            converged = true;
            break;
        }
        if fun_evals >= config.max_fun_evals {
            status = OptimStatus::MaxFunctionEvals;
            break;
        }
        if iterations >= config.max_iterations {
            status = OptimStatus::MaxIterationsReached;
            break;
        }

        let maxdist = max_distance(&st);

        // Model-gradient surrogate termination (poised geometry only).
        if st.delta <= T::from_f64(4.0).unwrap() * st.rho
            && maxdist <= T::from_f64(2.0).unwrap() * st.delta
        {
            let pg = projected_gradient_norm(&st.model.g, &st.model.xbase, &st.xl, &st.xu);
            if pg < config.gradient_tolerance {
                status = OptimStatus::Converged;
                converged = true;
                break;
            }
        }

        iterations += 1;

        // Trust-region subproblem.
        let dl: Vec<T> = (0..n).map(|i| st.xl[i] - st.model.xbase[i]).collect();
        let du: Vec<T> = (0..n).map(|i| st.xu[i] - st.model.xbase[i]).collect();
        let d = trsbox(&st.model.g, &st.model.h, &dl, &du, st.delta);
        let dnorm = T::vec_norm(&d);

        if dnorm < T::from_f64(0.5).unwrap() * st.rho {
            // Step too short to be worth an evaluation: fix geometry, shrink
            // delta, or reduce rho.
            if maxdist > T::from_f64(2.0).unwrap() * st.delta
                && geometry_step(f, &mut st, &mut fun_evals, config.max_fun_evals)?
            {
                continue;
            }
            if st.delta > st.rho {
                st.delta = T::max(T::from_f64(0.5).unwrap() * st.delta, st.rho);
                continue;
            }
            if !reduce_rho(&mut st, rhoend) {
                status = OptimStatus::StepTolerance;
                converged = true;
                break;
            }
            continue;
        }

        // Trust-region step: evaluate the objective.
        let xnew = sonium_solvers_core::vec_ops::vec_add(&st.model.xbase, &d);
        let fnew_raw = f(&xnew);
        fun_evals += 1;
        if !fnew_raw.is_finite() {
            return Err(OptimError::NumericalError {
                iteration: iterations,
                message: "objective callback returned non-finite value".to_string(),
            });
        }
        let pred = st.model.c - st.model.value(&xnew);
        let fopt = st.fval[st.kopt];
        let ratio = if fnew_raw.is_finite() && pred > T::zero() {
            (fopt - fnew_raw) / pred
        } else {
            T::neg_infinity()
        };

        // Always add the new information to the interpolation set.
        if fnew_raw.is_finite() {
            let old_pt;
            let old_f;
            {
                let t = eviction_candidate(&st, &xnew);
                old_pt = st.pts[t].clone();
                old_f = st.fval[t];
                st.pts[t] = xnew.clone();
                st.fval[t] = fnew_raw;
                if update_model(&mut st) {
                    if fnew_raw < fopt {
                        st.kopt = t;
                        let base = st.pts[t].clone();
                        st.model.shift_base(&base);
                    }
                } else {
                    // New point made the KKT system singular: revert.
                    st.pts[t] = old_pt;
                    st.fval[t] = old_f;
                }
            }
        }

        // Radius update.
        let half = T::from_f64(0.5).unwrap();
        let two = T::from_f64(2.0).unwrap();
        if ratio >= T::from_f64(0.7).unwrap() {
            st.delta = T::max(st.delta, two * dnorm);
        } else if ratio >= T::from_f64(0.1).unwrap() {
            st.delta = T::max(T::max(half * st.delta, dnorm), st.rho);
        } else {
            st.delta = T::max(half * st.delta, st.rho);
        }

        if ratio <= T::zero() {
            let maxdist = max_distance(&st);
            if maxdist > two * st.delta
                && geometry_step(f, &mut st, &mut fun_evals, config.max_fun_evals)?
            {
                continue;
            }
            if st.delta <= st.rho && !reduce_rho(&mut st, rhoend) {
                status = OptimStatus::StepTolerance;
                converged = true;
                break;
            }
        }
    }

    let x = st.pts[st.kopt].clone();
    let fun = st.fval[st.kopt];
    let grad_norm = projected_gradient_norm(&st.model.g, &st.model.xbase, &st.xl, &st.xu);
    Ok(OptimResult {
        x,
        fun,
        grad: Some(st.model.g.clone()),
        iterations,
        fun_evals,
        grad_evals: 0,
        hess_evals: 0,
        grad_norm,
        converged,
        status,
        kkt: None,
    })
}

// ---------------------------------------------------------------------------
// Quadratic interpolation model
// ---------------------------------------------------------------------------

/// Quadratic model `q(x) = c + gᵀ(x - xbase) + ½ (x - xbase)ᵀ H (x - xbase)`.
struct QuadraticModel<T: RealScalar> {
    xbase: Vec<T>,
    c: T,
    g: Vec<T>,
    h: DenseMatrix<T>,
}

impl<T: RealScalar> QuadraticModel<T> {
    fn zeros(xbase: Vec<T>) -> Self {
        let n = xbase.len();
        Self {
            xbase,
            c: T::zero(),
            g: vec![T::zero(); n],
            h: DenseMatrix::zeros(n, n),
        }
    }

    fn value(&self, x: &[T]) -> T {
        let d = sonium_solvers_core::vec_ops::vec_sub(x, &self.xbase);
        let hd = self.h.matvec(&d);
        self.c + self.g.dot(&d) + T::from_f64(0.5).unwrap() * d.dot(&hd)
    }

    /// Exact shift of the expansion point (valid for any quadratic).
    fn shift_base(&mut self, new_base: &[T]) {
        let delta = sonium_solvers_core::vec_ops::vec_sub(new_base, &self.xbase);
        let hd = self.h.matvec(&delta);
        self.c += self.g.dot(&delta) + T::from_f64(0.5).unwrap() * delta.dot(&hd);
        for (i, slot) in self.g.iter_mut().enumerate() {
            *slot += hd[i];
        }
        self.xbase = new_base.to_vec();
    }
}

struct State<T: RealScalar> {
    n: usize,
    /// Interpolation points (absolute coordinates), length m.
    pts: Vec<Vec<T>>,
    fval: Vec<T>,
    kopt: usize,
    model: QuadraticModel<T>,
    xl: Vec<T>,
    xu: Vec<T>,
    delta: T,
    rho: T,
}

fn max_distance<T: RealScalar>(st: &State<T>) -> T {
    let mut dmax = T::zero();
    for p in &st.pts {
        let d = T::vec_norm(&sonium_solvers_core::vec_ops::vec_sub(p, &st.model.xbase));
        if d > dmax {
            dmax = d;
        }
    }
    dmax
}

/// Reduce `rho` by a factor of 10 (clamped to `rhoend`); returns false when
/// `rho` is already at or below `rhoend` (i.e. the run should terminate).
fn reduce_rho<T: RealScalar>(st: &mut State<T>, rhoend: T) -> bool {
    if st.rho <= rhoend {
        return false;
    }
    st.rho = T::max(T::from_f64(0.1).unwrap() * st.rho, rhoend);
    st.delta = T::max(st.delta, st.rho);
    true
}

// ---------------------------------------------------------------------------
// Dense LU with partial pivoting (for the symmetric-indefinite KKT system)
// ---------------------------------------------------------------------------

struct Lu<T: RealScalar> {
    lu: Vec<T>, // row-major combined L (unit diagonal) and U
    n: usize,
    perm: Vec<usize>,
}

fn lu_factor<T: RealScalar>(a: &[T], n: usize) -> Option<Lu<T>> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut piv = k;
        let mut best = lu[k * n + k].abs();
        for i in (k + 1)..n {
            let v = lu[i * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best <= T::machine_epsilon() * T::machine_epsilon() {
            return None;
        }
        if piv != k {
            for j in 0..n {
                lu.swap(k * n + j, piv * n + j);
            }
            perm.swap(k, piv);
        }
        let d = lu[k * n + k];
        for i in (k + 1)..n {
            lu[i * n + k] /= d;
            let lik = lu[i * n + k];
            for j in (k + 1)..n {
                lu[i * n + j] = lu[i * n + j] - lik * lu[k * n + j];
            }
        }
    }
    Some(Lu { lu, n, perm })
}

fn lu_solve<T: RealScalar>(lu: &Lu<T>, b: &[T]) -> Vec<T> {
    let n = lu.n;
    let mut x: Vec<T> = (0..n).map(|i| b[lu.perm[i]]).collect();
    // Forward substitution (unit lower).
    for i in 0..n {
        let mut s = x[i];
        for (j, &xj) in x[..i].iter().enumerate() {
            s -= lu.lu[i * n + j] * xj;
        }
        x[i] = s;
    }
    // Back substitution.
    for i in (0..n).rev() {
        let mut s = x[i];
        for (j, &xj) in x.iter().enumerate().skip(i + 1) {
            s -= lu.lu[i * n + j] * xj;
        }
        x[i] = s / lu.lu[i * n + i];
    }
    x
}

// ---------------------------------------------------------------------------
// Interpolation KKT system
// ---------------------------------------------------------------------------

/// The factorised interpolation system for the current point set.
///
/// Solves `[[Q, A], [Aᵀ, 0]] z = rhs` with `Q_kj = ½ (s_k·s_j)²`,
/// `A = [1, S]`, `s_k = (y_k - xbase)/scale`. Nonsingular whenever the point
/// set contains `n + 1` affinely independent points (Powell 2004, Lemma).
struct KktSystem<T: RealScalar> {
    lu: Lu<T>,
    m: usize,
    n: usize,
    /// Scaled points `s_k`, length m of length-n vectors.
    s: Vec<Vec<T>>,
}

impl<T: RealScalar> KktSystem<T> {
    fn new(pts: &[Vec<T>], xbase: &[T], scale: T) -> Option<Self> {
        let m = pts.len();
        let n = xbase.len();
        let p = m + 1 + n;
        let s: Vec<Vec<T>> = pts
            .iter()
            .map(|pt| {
                (0..n)
                    .map(|i| (pt[i] - xbase[i]) / scale)
                    .collect::<Vec<T>>()
            })
            .collect();
        let half = T::from_f64(0.5).unwrap();
        let mut a = vec![T::zero(); p * p];
        for k in 0..m {
            for j in 0..m {
                let d = s[k].dot(&s[j]);
                a[k * p + j] = half * d * d;
            }
            a[k * p + m] = T::one();
            for i in 0..n {
                a[k * p + m + 1 + i] = s[k][i];
            }
        }
        for k in 0..m {
            a[m * p + k] = T::one();
        }
        for i in 0..n {
            for k in 0..m {
                a[(m + 1 + i) * p + k] = s[k][i];
            }
        }
        lu_factor(&a, p).map(|lu| Self { lu, m, n, s })
    }

    /// Lagrange function values `ℓ_j(x)` for all j: the first m entries of
    /// `KKT⁻¹ w(x)` where `w(x) = [½(s·s_k)², 1, s]` (the KKT matrix is
    /// symmetric, so `ℓ(x) = KKT⁻¹ w(x)` restricted to the point block).
    fn lagrange_values(&self, x: &[T], xbase: &[T], scale: T) -> Vec<T> {
        let p = self.m + 1 + self.n;
        let s: Vec<T> = (0..self.n).map(|i| (x[i] - xbase[i]) / scale).collect();
        let half = T::from_f64(0.5).unwrap();
        let mut w = vec![T::zero(); p];
        for (k, slot) in w[..self.m].iter_mut().enumerate() {
            let d = s.dot(&self.s[k]);
            *slot = half * d * d;
        }
        w[self.m] = T::one();
        for i in 0..self.n {
            w[self.m + 1 + i] = s[i];
        }
        let z = lu_solve(&self.lu, &w);
        z[..self.m].to_vec()
    }

    /// Gradient of the Lagrange function `ℓ_t` at `xbase` (in scaled
    /// coordinates): the gradient block of `KKT⁻¹ e_t`.
    fn lagrange_gradient(&self, t: usize) -> Vec<T> {
        let p = self.m + 1 + self.n;
        let mut e = vec![T::zero(); p];
        e[t] = T::one();
        let z = lu_solve(&self.lu, &e);
        z[self.m + 1..].to_vec()
    }
}

/// Minimum-Frobenius-norm update of the model so that it interpolates `fval`
/// on `pts`. Solves the KKT correction system with residuals
/// `r_k = fval_k - q(y_k)` and applies `H += Σ μ_k s_k s_kᵀ / scale²`,
/// `g += ĝ/scale`, `c += δc`. Returns false if the system is singular.
fn update_model<T: RealScalar>(st: &mut State<T>) -> bool {
    let m = st.pts.len();
    let n = st.n;
    let scale = st.delta;
    let Some(kkt) = KktSystem::new(&st.pts, &st.model.xbase, scale) else {
        return false;
    };
    let p = m + 1 + n;
    let mut rhs = vec![T::zero(); p];
    for (k, slot) in rhs[..m].iter_mut().enumerate() {
        *slot = st.fval[k] - st.model.value(&st.pts[k]);
    }
    let z = lu_solve(&kkt.lu, &rhs);
    st.model.c += z[m];
    for i in 0..n {
        st.model.g[i] += z[m + 1 + i] / scale;
    }
    let scale_sq = scale * scale;
    for (k, &mu) in z[..m].iter().enumerate() {
        if mu == T::zero() {
            continue;
        }
        for i in 0..n {
            for j in 0..=i {
                let upd = mu * kkt.s[k][i] * kkt.s[k][j] / scale_sq;
                st.model.h[[i, j]] += upd;
                if i != j {
                    st.model.h[[j, i]] += upd;
                }
            }
        }
    }
    true
}

/// Choose the interpolation point to evict when `xnew` enters the set:
/// `argmax_{j ≠ kopt} |ℓ_j(xnew)| · max(1, (dist_j/Δ)³)`.
fn eviction_candidate<T: RealScalar>(st: &State<T>, xnew: &[T]) -> usize {
    let m = st.pts.len();
    let mut weights = vec![T::zero(); m];
    if let Some(kkt) = KktSystem::new(&st.pts, &st.model.xbase, st.delta) {
        let lv = kkt.lagrange_values(xnew, &st.model.xbase, st.delta);
        for j in 0..m {
            weights[j] = lv[j].abs();
        }
    } else {
        // Singular system (should not happen): fall back to distance only.
        weights.fill(T::one());
    }
    let mut best = usize::MAX;
    let mut best_w = T::neg_infinity();
    for (j, &wgt) in weights.iter().enumerate() {
        if j == st.kopt {
            continue;
        }
        let dist = T::vec_norm(&sonium_solvers_core::vec_ops::vec_sub(
            &st.pts[j],
            &st.model.xbase,
        ));
        let r = dist / st.delta;
        let mut w = wgt;
        if r > T::one() {
            w *= r * r * r;
        }
        if best == usize::MAX || w > best_w {
            best = j;
            best_w = w;
        }
    }
    // m >= n + 2 >= 3 guarantees a j != kopt exists.
    debug_assert!(best != usize::MAX);
    best
}

// ---------------------------------------------------------------------------
// Geometry-improving step
// ---------------------------------------------------------------------------

/// Replace the farthest interpolation point by an approximate maximiser of
/// its Lagrange function inside the trust region. Returns false if no valid
/// new point could be placed (callers then fall back to reducing rho).
fn geometry_step<T, F>(
    f: &F,
    st: &mut State<T>,
    fun_evals: &mut usize,
    max_fun_evals: usize,
) -> Result<bool, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    if *fun_evals >= max_fun_evals {
        return Ok(false);
    }
    let n = st.n;

    // Eviction target: farthest point from xbase (never kopt).
    let mut t = usize::MAX;
    let mut dmax = T::zero();
    for (j, p) in st.pts.iter().enumerate() {
        if j == st.kopt {
            continue;
        }
        let d = T::vec_norm(&sonium_solvers_core::vec_ops::vec_sub(p, &st.model.xbase));
        if t == usize::MAX || d > dmax {
            t = j;
            dmax = d;
        }
    }
    if t == usize::MAX {
        return Ok(false);
    }

    let Some(kkt) = KktSystem::new(&st.pts, &st.model.xbase, st.delta) else {
        return Ok(false);
    };

    // Candidates: coordinate points xbase ± delta·e_i and the normalised
    // gradient of ℓ_t at xbase, all projected onto the bounds.
    let mut cands: Vec<Vec<T>> = Vec::with_capacity(2 * n + 1);
    for i in 0..n {
        for &sign in &[1.0, -1.0] {
            let mut c = st.model.xbase.clone();
            c[i] += T::from_f64(sign).unwrap() * st.delta;
            cands.push(c);
        }
    }
    let gt = kkt.lagrange_gradient(t);
    let gnorm = T::vec_norm(&gt);
    if gnorm > T::zero() {
        let mut c = st.model.xbase.clone();
        for i in 0..n {
            c[i] += st.delta * gt[i] / gnorm;
        }
        cands.push(c);
    }

    let min_sep = T::from_f64(1e-2).unwrap() * st.delta;
    let mut best_cand: Option<Vec<T>> = None;
    let mut best_score = T::zero();
    for c in cands.iter_mut() {
        for (i, slot) in c[..n].iter_mut().enumerate() {
            *slot = T::max(*slot, st.xl[i]);
            *slot = T::min(*slot, st.xu[i]);
        }
        let separated = st
            .pts
            .iter()
            .all(|p| T::vec_norm(&sonium_solvers_core::vec_ops::vec_sub(p, c)) >= min_sep);
        if !separated {
            continue;
        }
        let lv = kkt.lagrange_values(c, &st.model.xbase, st.delta);
        let score = lv[t].abs();
        if best_cand.is_none() || score > best_score {
            best_score = score;
            best_cand = Some(c.clone());
        }
    }
    let Some(cand) = best_cand else {
        return Ok(false);
    };

    let fnew = f(&cand);
    *fun_evals += 1;
    if !fnew.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: *fun_evals,
            message: "objective callback returned non-finite value".to_string(),
        });
    }

    let old_pt = st.pts[t].clone();
    let old_f = st.fval[t];
    st.pts[t] = cand;
    st.fval[t] = fnew;
    if !update_model(st) {
        st.pts[t] = old_pt;
        st.fval[t] = old_f;
        return Ok(false);
    }
    if fnew < st.fval[st.kopt] {
        st.kopt = t;
        let base = st.pts[t].clone();
        st.model.shift_base(&base);
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Bound-constrained trust-region subproblem (projected truncated CG)
// ---------------------------------------------------------------------------

/// Approximately minimise `q(d) = gᵀ d + ½ dᵀ H d` subject to
/// `dl ≤ d ≤ du` and `‖d‖ ≤ delta` by projected Steihaug-CG with active-set
/// freezing: when a CG step would cross a variable bound the step is
/// truncated there, that variable is frozen, and CG restarts on the new face.
fn trsbox<T: RealScalar>(g: &[T], h: &DenseMatrix<T>, dl: &[T], du: &[T], delta: T) -> Vec<T> {
    let n = g.len();
    let mut d = vec![T::zero(); n];
    let mut free = vec![true; n];
    // Variables pinned at a bound with an outward-pointing gradient stay put.
    for i in 0..n {
        if dl[i] >= T::zero() && g[i] > T::zero() {
            free[i] = false;
        }
        if du[i] <= T::zero() && g[i] < T::zero() {
            free[i] = false;
        }
    }
    let g_scale = T::max(T::one(), T::vec_norm(g));
    let tol = T::sqrt_epsilon() * g_scale;

    let mut frozen_count = free.iter().filter(|&&f| !f).count();
    // Each restart freezes at least one more variable, so this terminates.
    for _ in 0..(n + 1) {
        // Gradient at current d, restricted to free variables.
        let hd = h.matvec(&d);
        let mut r = sonium_solvers_core::vec_ops::vec_add(g, &hd);
        for i in 0..n {
            if !free[i] {
                r[i] = T::zero();
            }
        }
        let mut rr = r.dot(&r);
        if rr.sqrt() <= tol {
            break;
        }
        let mut p = sonium_solvers_core::vec_ops::vec_neg(&r);
        let mut froze = false;
        for _ in 0..(2 * n + 5) {
            let hp = h.matvec(&p);
            let php = p.dot(&hp);

            // Step lengths to the first bound crossing and to the TR boundary.
            let (tau_b, hit) = first_bound_crossing(&d, &p, &free, dl, du);
            let tau_tr = tr_boundary_step(&d, &p, delta);

            if php <= T::zero() {
                // Negative curvature: go to the nearer constraint.
                let tau = T::min(tau_tr, tau_b);
                T::vec_axpy(tau, &p, &mut d);
                if tau_b < tau_tr {
                    if let Some(i) = hit {
                        free[i] = false;
                        frozen_count += 1;
                        froze = true;
                    }
                    break;
                }
                return d;
            }

            let alpha = rr / php;
            if alpha <= tau_b && alpha <= tau_tr {
                // Full CG step stays feasible.
                T::vec_axpy(alpha, &p, &mut d);
                T::vec_axpy(alpha, &hp, &mut r);
                for i in 0..n {
                    if !free[i] {
                        r[i] = T::zero();
                    }
                }
                let rr_new = r.dot(&r);
                if rr_new.sqrt() <= tol {
                    return d;
                }
                let beta = rr_new / rr;
                for i in 0..n {
                    p[i] = -r[i] + beta * p[i];
                }
                rr = rr_new;
            } else {
                let tau = T::min(tau_b, tau_tr);
                T::vec_axpy(tau, &p, &mut d);
                if tau_b <= tau_tr {
                    if let Some(i) = hit {
                        free[i] = false;
                        frozen_count += 1;
                        froze = true;
                    }
                    break;
                }
                return d;
            }
        }
        if !froze || frozen_count >= n {
            break;
        }
    }
    d
}

/// Smallest `tau ≥ 0` such that `d + tau·p` first touches a variable bound,
/// over free variables only. Returns `(tau, Some(index))`.
fn first_bound_crossing<T: RealScalar>(
    d: &[T],
    p: &[T],
    free: &[bool],
    dl: &[T],
    du: &[T],
) -> (T, Option<usize>) {
    let mut tau = T::infinity();
    let mut hit = None;
    for i in 0..d.len() {
        if !free[i] || p[i] == T::zero() {
            continue;
        }
        let t = if p[i] > T::zero() {
            (du[i] - d[i]) / p[i]
        } else {
            (dl[i] - d[i]) / p[i]
        };
        if t < tau {
            tau = t;
            hit = Some(i);
        }
    }
    (T::max(tau, T::zero()), hit)
}

/// Positive root of `‖d + tau·p‖ = delta` (infinity if p = 0).
fn tr_boundary_step<T: RealScalar>(d: &[T], p: &[T], delta: T) -> T {
    let pp = p.dot(p);
    if pp <= T::machine_epsilon() {
        return T::infinity();
    }
    let dp = d.dot(p);
    let dd = d.dot(d);
    let a = pp;
    let b = T::from_f64(2.0).unwrap() * dp;
    let c = dd - delta * delta;
    let disc = b * b - T::from_f64(4.0).unwrap() * a * c;
    if disc < T::zero() {
        return T::zero();
    }
    T::max(
        (-b + disc.sqrt()) / (T::from_f64(2.0).unwrap() * a),
        T::zero(),
    )
}

/// Infinity norm of the gradient projected onto the feasible box.
fn projected_gradient_norm<T: RealScalar>(g: &[T], x: &[T], xl: &[T], xu: &[T]) -> T {
    let mut norm = T::zero();
    for i in 0..g.len() {
        let at_lower = x[i] <= xl[i] && g[i] > T::zero();
        let at_upper = x[i] >= xu[i] && g[i] < T::zero();
        if !at_lower && !at_upper {
            norm = T::max(norm, g[i].abs());
        }
    }
    norm
}

#[cfg(test)]
#[allow(deprecated)] // head-to-head benchmarks against the deprecated powell
mod tests {
    use super::*;
    use crate::test_functions::{rosenbrock, rosenbrock_nd, rosenbrock_nd_start, rosenbrock_start};
    use crate::unconstrained::{PowellConfig, powell};
    use approx::assert_relative_eq;
    use sonium_solvers_core::traits::ComplexField;
    use std::cell::Cell;

    // ---- Model construction sanity ----------------------------------------

    #[test]
    fn test_model_exact_on_quadratic_full_npt() {
        // With npt = (n+1)(n+2)/2 the interpolation system is fully
        // determined, so the model must reproduce any quadratic exactly.
        let n = 3;
        let c_true = 1.5;
        let g_true = sonium_solvers_core::array![0.5_f64, -1.0, 2.0];
        let h_true =
            sonium_solvers_core::array![[3.0_f64, 0.5, 0.0], [0.5, 2.0, -0.25], [0.0, -0.25, 1.0]];
        let f = |x: &[f64]| {
            let hd = h_true.matvec(x);
            c_true + g_true.dot(x) + 0.5 * x.dot(&hd)
        };
        let x0 = sonium_solvers_core::array![0.3_f64, -0.2, 0.7];
        let npt_full = (n + 1) * (n + 2) / 2;
        let rhobeg = 1.0_f64;
        let mut pts: Vec<Vec<f64>> = vec![x0.to_vec()];
        for i in 0..n {
            let mut p = x0.to_vec();
            p[i] += rhobeg;
            pts.push(p);
        }
        for i in 0..n {
            let mut p = x0.to_vec();
            p[i] -= rhobeg;
            pts.push(p);
        }
        // Full quadratic poise requires one diagonal point per (i, j) pair.
        'fill: for i in 0..n {
            for j in (i + 1)..n {
                if pts.len() >= npt_full {
                    break 'fill;
                }
                let mut p = x0.to_vec();
                p[i] += rhobeg;
                p[j] += rhobeg;
                pts.push(p);
            }
        }
        assert_eq!(pts.len(), npt_full);
        let fval: Vec<f64> = pts.iter().map(|p| f(p)).collect();
        let kopt = fval
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let mut st = State {
            n,
            pts,
            fval,
            kopt,
            model: QuadraticModel::zeros(x0.to_vec()),
            xl: vec![f64::NEG_INFINITY; n],
            xu: vec![f64::INFINITY; n],
            delta: rhobeg,
            rho: rhobeg,
        };
        let base = st.pts[kopt].clone();
        st.model.shift_base(&base);
        assert!(update_model(&mut st));

        // Model must match the true quadratic at arbitrary points.
        for x in [
            &[0.0, 0.0, 0.0][..],
            &[1.0, -1.0, 0.5][..],
            &[2.0, 3.0, -1.0][..],
        ] {
            assert_relative_eq!(st.model.value(x), f(x), epsilon = 1e-8, max_relative = 1e-8);
        }
        // Gradient at xbase should match the true gradient there.
        let xb = st.model.xbase.clone();
        let g_at_base: Vec<f64> = (0..n).map(|i| g_true[i] + h_true.matvec(&xb)[i]).collect();
        for i in 0..n {
            assert_relative_eq!(
                st.model.g[i],
                g_at_base[i],
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }
    }

    // ---- TRSBOX ------------------------------------------------------------

    #[test]
    fn test_trsbox_unconstrained_interior() {
        // q(d) = g·d + ½ dᵀ(2I)d → min at d = -g/2, inside TR and box.
        let g = sonium_solvers_core::array![2.0_f64, -4.0];
        let h = sonium_solvers_core::array![[2.0_f64, 0.0], [0.0, 2.0]];
        let dl = sonium_solvers_core::array![-10.0_f64, -10.0];
        let du = sonium_solvers_core::array![10.0_f64, 10.0];
        let d = trsbox(&g, &h, &dl, &du, 100.0);
        assert_relative_eq!(d[0], -1.0, epsilon = 1e-8);
        assert_relative_eq!(d[1], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_trsbox_respects_trust_region() {
        let g = sonium_solvers_core::array![10.0_f64, 10.0];
        let h = sonium_solvers_core::array![[1.0_f64, 0.0], [0.0, 1.0]];
        let dl = sonium_solvers_core::array![-100.0_f64, -100.0];
        let du = sonium_solvers_core::array![100.0_f64, 100.0];
        let d = trsbox(&g, &h, &dl, &du, 0.5);
        assert!(f64::vec_norm(&d) <= 0.5 + 1e-12);
    }

    #[test]
    fn test_trsbox_respects_bounds() {
        // Unconstrained min at d = [-5, 5]; box clips to dl[0] = -1, du[1] = 2.
        let g = sonium_solvers_core::array![10.0_f64, -10.0];
        let h = sonium_solvers_core::array![[2.0_f64, 0.0], [0.0, 2.0]];
        let dl = sonium_solvers_core::array![-1.0_f64, -10.0];
        let du = sonium_solvers_core::array![10.0_f64, 2.0];
        let d = trsbox(&g, &h, &dl, &du, 100.0);
        assert!(d[0] >= -1.0 - 1e-12 && d[1] <= 2.0 + 1e-12);
        assert_relative_eq!(d[0], -1.0, epsilon = 1e-8);
        assert_relative_eq!(d[1], 2.0, epsilon = 1e-8);
    }

    // ---- Convergence --------------------------------------------------------

    #[test]
    fn test_bobyqa_rejects_non_finite_and_zero_budget_configuration() {
        let objective = |x: &[f64]| x.iter().map(|value| value * value).sum();
        let mut config = BobyqaConfig::default();

        assert!(matches!(
            bobyqa(&objective, &[f64::NAN], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        config.max_fun_evals = 0;
        assert!(matches!(
            bobyqa(&objective, &[1.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        config = BobyqaConfig::default();
        config.rhoend = f64::NAN;
        assert!(matches!(
            bobyqa(&objective, &[1.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        config = BobyqaConfig::default();
        config.bounds = vec![Bound {
            lower: Some(f64::NAN),
            upper: Some(1.0),
        }];
        assert!(matches!(
            bobyqa(&objective, &[0.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));
    }

    #[test]
    fn test_bobyqa_rejects_non_finite_generated_objective() {
        let objective = |x: &[f64]| {
            if x.len() == 1 && x[0] == 1.0 {
                0.0
            } else {
                f64::NAN
            }
        };

        let error = bobyqa(&objective, &[1.0], &BobyqaConfig::default())
            .expect_err("non-finite interpolation objective must fail closed");
        assert!(matches!(
            error,
            OptimError::NumericalError { iteration: 0, .. }
        ));
        assert!(error.to_string().contains("not finite"));
    }

    #[test]
    fn test_bobyqa_rejects_non_finite_trust_region_objective() {
        let calls = Cell::new(0usize);
        let objective = |x: &[f64]| {
            let call = calls.get();
            calls.set(call + 1);
            if call < 4 {
                (x[0] - 0.5).powi(2)
            } else {
                f64::NAN
            }
        };

        let error = bobyqa(&objective, &[0.0], &BobyqaConfig::default())
            .expect_err("non-finite trust-region objective must fail closed");
        assert!(matches!(
            error,
            OptimError::NumericalError { iteration, .. } if iteration > 0
        ));
    }

    #[test]
    fn test_bobyqa_rosenbrock_2d() {
        let x0 = rosenbrock_start();
        let config = BobyqaConfig::default();
        let result = bobyqa(&rosenbrock, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert!(result.fun < 1e-6, "fun = {}", result.fun);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_bobyqa_rosenbrock_5d() {
        let x0 = rosenbrock_nd_start(5);
        let f = |x: &[f64]| rosenbrock_nd(x);
        let config = BobyqaConfig::default();
        let result = bobyqa(&f, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}, evals: {}",
            result.status, result.fun, result.fun_evals
        );
        assert!(result.fun < 1e-6, "fun = {}", result.fun);
    }

    #[test]
    fn test_bobyqa_sphere_custom_npt() {
        let f = |x: &[f64]| x.dot(x);
        let x0 = sonium_solvers_core::array![3.0_f64, -2.0, 1.5];
        let config = BobyqaConfig {
            npt: Some(5), // n + 2
            ..Default::default()
        };
        let result = bobyqa(&f, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-8, "fun = {}", result.fun);
    }

    // ---- Bounds --------------------------------------------------------------

    #[test]
    fn test_bobyqa_optimum_at_bound_1d() {
        // min (x - 3)^2 over [0, 1]: optimum at the upper bound x = 1.
        let f = |x: &[f64]| (x[0] - 3.0) * (x[0] - 3.0);
        let x0 = sonium_solvers_core::array![0.5_f64];
        let config = BobyqaConfig {
            bounds: vec![Bound::both(0.0, 1.0)],
            ..Default::default()
        };
        let result = bobyqa(&f, &x0, &config).unwrap();
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.fun, 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_bobyqa_respects_bounds_2d() {
        // min (x0 - 3)^2 + (x1 + 2)^2 over [-1, 1]^2: optimum at (1, -1).
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let x0 = sonium_solvers_core::array![0.0_f64, 0.0];
        let config = BobyqaConfig {
            bounds: vec![Bound::both(-1.0, 1.0), Bound::both(-1.0, 1.0)],
            ..Default::default()
        };
        let result = bobyqa(&f, &x0, &config).unwrap();
        assert!(result.x[0] <= 1.0 + 1e-12 && result.x[0] >= -1.0 - 1e-12);
        assert!(result.x[1] <= 1.0 + 1e-12 && result.x[1] >= -1.0 - 1e-12);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-5);
        assert_relative_eq!(result.fun, 5.0, epsilon = 1e-6);
    }

    // ---- Noisy objective ------------------------------------------------------

    /// Deterministic "noisy" quadratic: smooth base plus a small oscillatory
    /// perturbation (same value on every call, so results are deterministic).
    fn noisy_quadratic(x: &[f64]) -> f64 {
        let target = [0.5, -1.0, 2.0];
        let mut base = 0.0;
        for i in 0..3 {
            base += (x[i] - target[i]).powi(2);
        }
        let phase = 1.0e5 * (1.1 * x[0] + 1.7 * x[1] + 2.3 * x[2]);
        base + 1e-3 * phase.sin()
    }

    #[test]
    fn test_bobyqa_noisy_quadratic_better_than_powell() {
        let x0 = sonium_solvers_core::array![0.0_f64, 0.0, 0.0];
        let bob = bobyqa(&noisy_quadratic, &x0, &BobyqaConfig::default()).unwrap();
        let pow = powell(
            &noisy_quadratic,
            &x0,
            &PowellConfig {
                max_iterations: 5000,
                ..Default::default()
            },
        )
        .unwrap();
        let target = [0.5, -1.0, 2.0];
        let err = |x: &[f64]| {
            (0..3)
                .map(|i| (x[i] - target[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let bob_err = err(&bob.x);
        let pow_err = err(&pow.x);
        assert!(
            bob_err <= pow_err.max(1e-8),
            "bobyqa error {bob_err:e} should not exceed powell error {pow_err:e}"
        );
        // And bobyqa should be genuinely close to the minimiser (the noise
        // floor is ~sqrt(1e-3) ~ 3e-2).
        assert!(bob_err < 2e-2, "bobyqa error {bob_err:e}");
    }

    // ---- Evaluation-count comparison (headline claim) --------------------------

    #[test]
    fn test_bobyqa_fewer_evals_than_powell_rosenbrock_2d() {
        let count = Cell::new(0usize);
        let f = |x: &[f64]| {
            count.set(count.get() + 1);
            rosenbrock(x)
        };
        let x0 = rosenbrock_start();
        let bob = bobyqa(&f, &x0, &BobyqaConfig::default()).unwrap();
        assert!(bob.fun < 1e-6, "bobyqa fun = {}", bob.fun);
        let bob_evals = count.get();

        count.set(0);
        let pow = powell(
            &f,
            &x0,
            &PowellConfig {
                max_iterations: 5000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(pow.converged, "powell status: {:?}", pow.status);
        let pow_evals = count.get();

        assert!(
            bob_evals < pow_evals,
            "bobyqa used {bob_evals} evals, powell used {pow_evals}"
        );
    }

    #[test]
    fn test_bobyqa_fewer_evals_than_powell_quartic_4d() {
        // f(x) = sum (x_i - c_i)^4, min 0 at c.
        let c = [0.5_f64, -1.0, 2.0, 1.5];
        let count = Cell::new(0usize);
        let f = |x: &[f64]| {
            count.set(count.get() + 1);
            (0..4).map(|i| (x[i] - c[i]).powi(4)).sum()
        };
        let x0 = sonium_solvers_core::array![0.0_f64, 0.0, 0.0, 0.0];
        let bob = bobyqa(&f, &x0, &BobyqaConfig::default()).unwrap();
        assert!(bob.fun < 1e-6, "bobyqa fun = {}", bob.fun);
        let bob_evals = count.get();

        count.set(0);
        let pow = powell(
            &f,
            &x0,
            &PowellConfig {
                max_iterations: 5000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(pow.fun < 1e-6, "powell fun = {}", pow.fun);
        let pow_evals = count.get();

        assert!(
            bob_evals < pow_evals,
            "bobyqa used {bob_evals} evals, powell used {pow_evals}"
        );
    }

    // ---- Determinism ---------------------------------------------------------

    #[test]
    fn test_bobyqa_deterministic() {
        let x0 = rosenbrock_start();
        let config = BobyqaConfig::default();
        let r1 = bobyqa(&rosenbrock, &x0, &config).unwrap();
        let r2 = bobyqa(&rosenbrock, &x0, &config).unwrap();
        assert_eq!(r1.fun_evals, r2.fun_evals);
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.fun.to_bits(), r2.fun.to_bits());
    }

    // ---- f32 ------------------------------------------------------------------

    #[test]
    fn test_bobyqa_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let x0 = sonium_solvers_core::array![3.0_f32, -2.0];
        let config: BobyqaConfig<f32> = BobyqaConfig::default();
        let result = bobyqa(&f, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-3, "fun = {}", result.fun);
    }
}
