//! Anderson acceleration for fixed-point iterations
//!
//! Accelerates the convergence of a fixed-point iteration `x_{k+1} = g(x_k)`
//! without requiring the Jacobian of `g`. Anderson acceleration (Anderson
//! mixing) keeps the last `m + 1` iterates and, at each step, forms the new
//! iterate as an affine combination of the recent `g(x_i)` whose coefficients
//! minimise the combined residual `f(x) = g(x) − x` in the 2-norm.
//!
//! The small least-squares problem is solved in the difference formulation of
//! Walker–Ni (2011): with `Δf_i = f_{i+1} − f_i` and `Δg_i = g_{i+1} − g_i`,
//! solve `min_γ ‖f_k − Σ γ_i Δf_i‖₂` by modified Gram-Schmidt QR of the
//! `n × m` ΔF matrix, then
//!
//! ```text
//! x_{k+1} = g_k − Σ γ_i Δg_i − (1 − β) (f_k − Σ γ_i Δf_i)
//! ```
//!
//! where `β` is the mixing parameter (`β = 1` is undamped Anderson).
//!
//! QR is used instead of the normal equations because the ΔF columns of an
//! Anderson history are routinely near-dependent, and the normal equations
//! square the condition number — Walker–Ni (2011) and Fang–Saad (2009) both
//! recommend QR here. Columns whose norm after orthogonalisation falls below
//! `sqrt(eps) · ‖Δf_i‖` are dropped from the basis (γ = 0), which keeps the
//! effective conditioning bounded and handles exactly rank-deficient ΔF
//! gracefully.
//!
//! ## Scalar type
//!
//! Generic over [`ComplexField`](crate::ComplexField): works with f32, f64,
//! Complex32, and Complex64.

use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::panic::{AssertUnwindSafe, catch_unwind};
use thiserror::Error;

use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverError, SolverFailure, SolverResultContract, SolverStatus,
};

/// Anderson acceleration configuration.
///
/// Algorithm parameters are stored as f64 and converted to the working
/// precision at solve time.
#[derive(Debug, Clone)]
pub struct AndersonConfig {
    /// Anderson depth m: the number of previous iterates (beyond the current
    /// one) used in the affine combination. Typical: 3–10.
    /// `depth = 0` disables the affine combination and reduces to the
    /// damped fixed-point step.
    pub depth: usize,
    /// Mixing parameter β ∈ (0, 1]. β = 1 is undamped Anderson acceleration;
    /// smaller values damp the update (more robust, slower).
    pub mixing: f64,
    /// Maximum number of fixed-point evaluations.
    pub max_iterations: usize,
    /// Convergence tolerance on `‖g(x) − x‖` (absolute).
    pub absolute_tolerance: f64,
    /// Convergence tolerance on `‖g(x) − x‖` (relative to the initial value).
    pub relative_tolerance: f64,
    /// Safeguarding: evaluate the candidate before accepting it; if its
    /// residual does not decrease, fall back to the damped fixed-point step.
    /// Costs at most one extra evaluation of `g` per iteration — and, more
    /// importantly, every rejected candidate discards an Anderson update, so
    /// on smooth problems safeguarding can more than double the step count
    /// (measured 66 vs 27 steps on a ρ = 0.95 contraction at n = 32000).
    /// PETSc SNESANDERSON applies the update unconditionally, so the default
    /// is `false` for vendor parity; opt into `true` for fragile maps where
    /// raw combinations overshoot (e.g. expansive regions far from the
    /// fixed point).
    pub safeguarding: bool,
    /// Verbosity level (0 = quiet, 1 = per-iteration, 2+ = detailed).
    pub verbosity: usize,
}

impl Default for AndersonConfig {
    fn default() -> Self {
        Self {
            depth: 5,
            mixing: 1.0,
            max_iterations: 100,
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-8,
            safeguarding: false,
            verbosity: 0,
        }
    }
}

impl AndersonConfig {
    /// Validate the public configuration before evaluating the user callback.
    pub fn validate(&self) -> Result<(), AndersonError> {
        if self.max_iterations == 0 {
            return Err(AndersonError::InvalidConfig(
                "max_iterations must be non-zero".into(),
            ));
        }
        if !self.mixing.is_finite() || !(0.0..=1.0).contains(&self.mixing) || self.mixing == 0.0 {
            return Err(AndersonError::InvalidConfig(
                "mixing must be finite and in (0, 1]".into(),
            ));
        }
        if !self.absolute_tolerance.is_finite() || self.absolute_tolerance < 0.0 {
            return Err(AndersonError::InvalidConfig(
                "absolute_tolerance must be finite and non-negative".into(),
            ));
        }
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(AndersonError::InvalidConfig(
                "relative_tolerance must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// Result of an Anderson-accelerated solve, generic over the scalar type.
#[derive(Debug, Clone)]
pub struct AndersonSolution<T: ComplexField> {
    /// Solution vector (approximate fixed point).
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final fixed-point residual norm `‖g(x) − x‖`.
    pub residual: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Detailed status.
    pub status: SolverStatus,
    /// Whether the nonlinear fixed-point residual attained the requested
    /// tolerance. A linear backward-error metric is not defined for a
    /// general user-supplied fixed-point map.
    pub attainable_accuracy: AccuracyStatus,
}

impl<T: ComplexField> SolverResultContract for AndersonSolution<T> {
    fn solver_status(&self) -> SolverStatus {
        self.status
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.x.iter().any(|value| !value.norm().is_finite())
            || !self.residual.is_finite()
            || self.residual < 0.0
        {
            return Err(SolverError::OperationFailed {
                operation: "Anderson result contract",
                reason: "solution or residual is non-finite or negative".into(),
            });
        }
        if self.converged != (self.status == SolverStatus::Converged) {
            return Err(SolverError::OperationFailed {
                operation: "Anderson result contract",
                reason: "converged flag and solver status disagree".into(),
            });
        }
        if self.converged && self.attainable_accuracy != AccuracyStatus::Attained {
            return Err(SolverError::OperationFailed {
                operation: "Anderson result contract",
                reason: "converged result must report attained accuracy".into(),
            });
        }
        Ok(())
    }
}

/// Errors from Anderson acceleration.
#[derive(Debug, Error)]
pub enum AndersonError {
    #[error("Anderson acceleration did not converge in {0} iterations (residual = {1:.2e})")]
    NotConverged(usize, f64),
    #[error("Residual diverged at iteration {0} (residual = {1:.2e})")]
    Diverged(usize, f64),
    #[error("Invalid Anderson configuration: {0}")]
    InvalidConfig(String),
    #[error("Anderson callback returned length {got}, expected {expected} at {stage}")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        stage: &'static str,
    },
    #[error("Anderson callback returned a non-finite value at index {index} during {stage}")]
    NonFinite { index: usize, stage: &'static str },
    #[error("Anderson callback panicked during {stage}: {reason}")]
    CallbackPanicked { stage: &'static str, reason: String },
    #[error("Anderson numerical error at iteration {iteration} during {stage}: {reason}")]
    NumericalError {
        iteration: usize,
        stage: &'static str,
        reason: String,
    },
}

impl AndersonError {
    /// Return the canonical solver status corresponding to this failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NotConverged(..) => SolverStatus::MaxIterationsReached,
            Self::Diverged(..) => SolverStatus::Diverged,
            Self::InvalidConfig(_)
            | Self::DimensionMismatch { .. }
            | Self::NonFinite { .. }
            | Self::NumericalError { .. }
            | Self::CallbackPanicked { .. } => SolverStatus::NumericalError,
        }
    }

    /// Return the nonlinear attainable-accuracy state corresponding to this
    /// failure.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        match self {
            Self::NotConverged(..) | Self::Diverged(..) => AccuracyStatus::NotAttained,
            Self::InvalidConfig(_)
            | Self::DimensionMismatch { .. }
            | Self::NonFinite { .. }
            | Self::NumericalError { .. }
            | Self::CallbackPanicked { .. } => AccuracyStatus::NotChecked,
        }
    }
}

impl SolverFailure for AndersonError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotConverged(..) | Self::Diverged(..) => SolverErrorKind::Convergence,
            Self::DimensionMismatch { .. } => SolverErrorKind::DimensionMismatch,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        AndersonError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AndersonError::attainable_accuracy(self)
    }
}

/// Fallible scalar conversions for hot paths: exotic [`ComplexField`]
/// impls may not represent a constant that is trivially fine for f32/f64.
/// A conversion failure is a typed [`AndersonError::NumericalError`],
/// never a panic.
fn anderson_real_to_f64<T: ComplexField>(
    value: T::Real,
    stage: &'static str,
    iteration: usize,
) -> Result<f64, AndersonError>
where
    T::Real: ToPrimitive,
{
    value.to_f64().ok_or_else(|| AndersonError::NumericalError {
        iteration,
        stage,
        reason: "scalar value is not representable as f64".to_string(),
    })
}

/// Solve the fixed-point problem `x = g(x)` with Anderson acceleration.
///
/// Generic over any [`ComplexField`] type: works with f32, f64, Complex32,
/// and Complex64.
///
/// The caller provides a single closure `g(x) -> Vec<T>` evaluating the
/// fixed-point map. No Jacobian is needed. Solving `F(x) = 0` fits this
/// interface via `g(x) = x + ω F(x)` whenever that map is (nearly) a
/// contraction.
pub fn anderson_acceleration<T, G>(
    initial_guess: &[T],
    fixed_point_fn: G,
    config: &AndersonConfig,
) -> Result<AndersonSolution<T>, AndersonError>
where
    T: ComplexField,
    T::Real: FromPrimitive,
    G: Fn(&[T]) -> Vec<T>,
{
    config.validate()?;
    if initial_guess.is_empty() {
        return Err(AndersonError::InvalidConfig(
            "initial_guess must contain at least one value".into(),
        ));
    }
    if let Some(index) = initial_guess
        .iter()
        .position(|value| !value.norm().is_finite())
    {
        return Err(AndersonError::NonFinite {
            index,
            stage: "initial guess",
        });
    }
    let one_minus_beta = T::from_real(T::Real::from_f64(1.0 - config.mixing).ok_or_else(|| {
        AndersonError::InvalidConfig("mixing is not representable in the working precision".into())
    })?);

    let mut x = initial_guess.to_vec();
    let mut g = evaluate_fixed_point(&fixed_point_fn, &x, x.len(), "initial evaluation")?;
    let mut f = sonium_solvers_core::vec_ops::vec_sub(&g, &x);
    let mut f_norm = anderson_real_to_f64::<T>(T::vec_norm(&f), "initial residual norm", 0)?;
    let mut initial_norm = 0.0;

    // History of the last depth+1 residuals f_i and map values g_i.
    let mut f_hist: Vec<Vec<T>> = Vec::with_capacity(config.depth + 1);
    let mut g_hist: Vec<Vec<T>> = Vec::with_capacity(config.depth + 1);

    // Scratch buffers reused across iterations. The math is bit-identical to
    // allocating fresh vectors per step, but n-sized vectors exceed the
    // allocator mmap threshold, so per-step allocation churn shows up as
    // syscalls + page faults in profiles.
    let mut x_new: Vec<T> = Vec::new();
    let mut f_res: Vec<T> = Vec::new();
    let mut df_buf: Vec<Vec<T>> = Vec::with_capacity(config.depth);
    let mut dg_buf: Vec<Vec<T>> = Vec::with_capacity(config.depth);
    let mut qr_scratch = AndersonQrScratch::<T>::default();

    for iter in 0..config.max_iterations {
        if iter == 0 {
            initial_norm = f_norm;
        }
        if config.verbosity >= 1 {
            log::info!("Anderson iter {iter}: |f| = {f_norm:.6e}");
        }

        // Convergence checks (`<=` so an exact initial guess terminates
        // with `iterations: 0` even at `absolute_tolerance = 0`).
        if f_norm <= config.absolute_tolerance {
            return Ok(AndersonSolution {
                x,
                iterations: iter,
                residual: f_norm,
                converged: true,
                status: SolverStatus::Converged,
                attainable_accuracy: AccuracyStatus::Attained,
            });
        }
        if iter > 0 && initial_norm > 0.0 && f_norm / initial_norm < config.relative_tolerance {
            return Ok(AndersonSolution {
                x,
                iterations: iter,
                residual: f_norm,
                converged: true,
                status: SolverStatus::Converged,
                attainable_accuracy: AccuracyStatus::Attained,
            });
        }
        if iter > 0 && f_norm > 1e6 * initial_norm {
            return Err(AndersonError::Diverged(iter, f_norm));
        }

        // Update the history (drop the oldest entry beyond depth+1).
        f_hist.push(f.clone());
        g_hist.push(g.clone());
        if f_hist.len() > config.depth + 1 {
            f_hist.remove(0);
            g_hist.remove(0);
        }
        let len = f_hist.len();
        let mk = config.depth.min(len - 1);

        // Anderson candidate (or damped fixed-point step when no history).
        // `clone_from` reuses the scratch allocation; values equal `g.clone()`.
        x_new.clone_from(&g);
        if mk > 0 {
            let start = len - 1 - mk;
            while df_buf.len() < mk {
                df_buf.push(Vec::new());
                dg_buf.push(Vec::new());
            }
            for (k, j) in (start..len - 1).enumerate() {
                df_buf[k].clone_from(&f_hist[j + 1]);
                sonium_solvers_core::vec_ops::vec_sub_assign(&mut df_buf[k], &f_hist[j]);
                dg_buf[k].clone_from(&g_hist[j + 1]);
                sonium_solvers_core::vec_ops::vec_sub_assign(&mut dg_buf[k], &g_hist[j]);
            }
            let df = &df_buf[..mk];
            let dg = &dg_buf[..mk];

            if let Some(gamma) = anderson_gamma(df, &f, &mut qr_scratch) {
                for (j, gj) in gamma.iter().enumerate() {
                    T::vec_axpy(-*gj, &dg[j], &mut x_new);
                }
                // f_res = f_k − Σ γ_j Δf_j
                f_res.clone_from(&f);
                for (j, gj) in gamma.iter().enumerate() {
                    T::vec_axpy(-*gj, &df[j], &mut f_res);
                }
                T::vec_axpy(-one_minus_beta, &f_res, &mut x_new);
            } else {
                // Degenerate Gram system: plain damped fixed-point step.
                T::vec_axpy(-one_minus_beta, &f, &mut x_new);
            }
        } else {
            T::vec_axpy(-one_minus_beta, &f, &mut x_new);
        }

        if config.safeguarding {
            // Evaluate the candidate; accept only on residual decrease,
            // otherwise fall back to the damped fixed-point step.
            let g_cand =
                evaluate_fixed_point(&fixed_point_fn, &x_new, x.len(), "safeguard candidate")?;
            let f_cand = sonium_solvers_core::vec_ops::vec_sub(&g_cand, &x_new);
            let f_cand_norm =
                anderson_real_to_f64::<T>(T::vec_norm(&f_cand), "candidate residual norm", iter)?;
            if f_cand_norm <= f_norm {
                std::mem::swap(&mut x, &mut x_new);
                g = g_cand;
                f = f_cand;
                f_norm = f_cand_norm;
            } else {
                if config.verbosity >= 2 {
                    log::info!(
                        "  Anderson safeguard: rejected candidate (|f| {f_cand_norm:.6e} > {f_norm:.6e})"
                    );
                }
                let mut x_fallback = g.clone();
                T::vec_axpy(-one_minus_beta, &f, &mut x_fallback);
                x = x_fallback;
                g = evaluate_fixed_point(&fixed_point_fn, &x, x.len(), "safeguard fallback")?;
                f = sonium_solvers_core::vec_ops::vec_sub(&g, &x);
                f_norm = anderson_real_to_f64::<T>(T::vec_norm(&f), "residual norm", iter)?;
            }
        } else {
            std::mem::swap(&mut x, &mut x_new);
            g = evaluate_fixed_point(&fixed_point_fn, &x, x.len(), "iteration")?;
            f = sonium_solvers_core::vec_ops::vec_sub(&g, &x);
            f_norm = anderson_real_to_f64::<T>(T::vec_norm(&f), "residual norm", iter)?;
        }
    }

    Err(AndersonError::NotConverged(config.max_iterations, f_norm))
}

fn evaluate_fixed_point<T, G>(
    fixed_point_fn: &G,
    x: &[T],
    expected: usize,
    stage: &'static str,
) -> Result<Vec<T>, AndersonError>
where
    T: ComplexField,
    G: Fn(&[T]) -> Vec<T>,
{
    let values = catch_unwind(AssertUnwindSafe(|| fixed_point_fn(x))).map_err(|payload| {
        AndersonError::CallbackPanicked {
            stage,
            reason: panic_reason(payload),
        }
    })?;
    if values.len() != expected {
        return Err(AndersonError::DimensionMismatch {
            expected,
            got: values.len(),
            stage,
        });
    }
    if let Some(index) = values.iter().position(|value| !value.norm().is_finite()) {
        return Err(AndersonError::NonFinite { index, stage });
    }
    Ok(values)
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "callback panicked with a non-string payload".to_string()
    }
}

/// Solve the Anderson least-squares problem `min_γ ‖f_k − Σ γ_i Δf_i‖₂` by
/// modified Gram-Schmidt QR of the ΔF matrix, following the Walker–Ni (2011)
/// and Fang–Saad (2009) recommendation. QR avoids squaring the condition
/// number the way the normal equations do, which matters because the ΔF
/// columns of an Anderson history are routinely near-dependent.
///
/// The QR is rebuilt from scratch at every Anderson step: `m ≤ depth ≤ ~10`,
/// so the O(m²·n) MGS cost is negligible next to a fixed-point evaluation,
/// and no update/downdate bookkeeping is needed.
///
/// Column-dependence detection: a column whose norm after orthogonalisation
/// against the previously accepted columns falls below `sqrt(eps) · ‖Δf_j‖`
/// is dropped from the basis (its γ entry is 0). This keeps the effective
/// conditioning bounded by roughly `1/sqrt(eps)` and makes exactly
/// rank-deficient ΔF a non-event.
///
/// Returns `None` when every column is dropped (ΔF carries no usable
/// information) — the caller then falls back to a damped fixed-point step,
/// exactly as it did for a singular Gram system.
fn anderson_gamma<T: ComplexField>(
    df: &[Vec<T>],
    f: &[T],
    scratch: &mut AndersonQrScratch<T>,
) -> Option<Vec<T>> {
    let mk = df.len();
    let tau = T::Real::epsilon().sqrt();

    // q[k]: k-th accepted orthonormal basis vector (pooled buffers).
    // r[k]: column k of R (entries r_0k..r_kk, last entry is the diagonal).
    // kept[k]: original ΔF index of the k-th accepted column.
    // All buffers are reused across Anderson steps; every value written here
    // equals the fresh-allocation version, so the math is bit-identical.
    let AndersonQrScratch {
        q,
        r,
        kept,
        v,
        rk,
        rhs,
        gamma_kept,
    } = scratch;
    let mut nq = 0usize;
    kept.clear();

    for (j, col) in df.iter().enumerate() {
        let col_norm = T::vec_norm(col);
        v.clear();
        v.extend_from_slice(col);
        rk.clear();
        for qk in q.iter().take(nq) {
            let rij = T::vec_dot(qk, v);
            rk.push(rij);
            T::vec_axpy(-rij, qk, v);
        }
        let res_norm = T::vec_norm(v);
        if res_norm <= tau * col_norm {
            // Linearly dependent on the accepted basis (or zero): drop.
            continue;
        }
        T::vec_scale(v, T::from_real(res_norm.recip()));
        rk.push(T::from_real(res_norm));
        if nq == q.len() {
            q.push(Vec::new());
            r.push(Vec::new());
        }
        q[nq].clone_from(v);
        r[nq].clone_from(rk);
        nq += 1;
        kept.push(j);
    }

    if kept.is_empty() {
        return None;
    }

    // γ = R⁻¹ Q^H f by back substitution over the accepted columns.
    let p = kept.len();
    rhs.clear();
    for qk in q.iter().take(p) {
        rhs.push(T::vec_dot(qk, f));
    }
    gamma_kept.clear();
    gamma_kept.resize(p, T::zero());
    for i in (0..p).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..p {
            s -= r[j][i] * gamma_kept[j];
        }
        gamma_kept[i] = s / r[i][i];
    }

    let mut gamma = vec![T::zero(); mk];
    for (k, &j) in kept.iter().enumerate() {
        gamma[j] = gamma_kept[k];
    }
    Some(gamma)
}

/// Reusable n-sized scratch buffers for [`anderson_gamma`].
///
/// Column buffers grow to the problem dimension on first use and are then
/// reused across Anderson steps (lengths reset, allocations kept), avoiding
/// per-step mmap churn for vectors above the allocator threshold.
struct AndersonQrScratch<T> {
    q: Vec<Vec<T>>,
    r: Vec<Vec<T>>,
    kept: Vec<usize>,
    v: Vec<T>,
    rk: Vec<T>,
    rhs: Vec<T>,
    gamma_kept: Vec<T>,
}

impl<T> Default for AndersonQrScratch<T> {
    fn default() -> Self {
        Self {
            q: Vec::new(),
            r: Vec::new(),
            kept: Vec::new(),
            v: Vec::new(),
            rk: Vec::new(),
            rhs: Vec::new(),
            gamma_kept: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_anderson_cos_fixed_point() {
        // g(x) = cos(x): linearly convergent Picard iteration (factor ≈ 0.67
        // near the fixed point 0.739085...). Anderson should need far fewer
        // iterations than plain Picard.
        let x0 = sonium_solvers_core::array![0.5_f64];
        let config = AndersonConfig {
            depth: 5,
            absolute_tolerance: 1e-12,
            relative_tolerance: 1e-15,
            max_iterations: 200,
            ..Default::default()
        };

        let result =
            anderson_acceleration(&x0, |x| sonium_solvers_core::array![x[0].cos()], &config)
                .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 0.7390851332151607).abs() < 1e-10);

        // Plain Picard for comparison.
        let mut xp = 0.5_f64;
        let mut picard_iters = 0;
        for k in 1..1000 {
            let g = xp.cos();
            if (g - xp).abs() < 1e-12 {
                picard_iters = k;
                break;
            }
            xp = g;
        }
        assert!(picard_iters > 0, "Picard should converge");
        assert!(
            result.iterations < picard_iters,
            "Anderson ({} iters) should beat Picard ({} iters)",
            result.iterations,
            picard_iters
        );
    }

    #[test]
    fn test_anderson_linear_contraction_2d() {
        // g(x) = A x + b with ρ(A) < 1: g is a contraction, fixed point
        // solves (I − A) x = b. Anderson on a 2-D linear problem should
        // converge in very few iterations.
        let a = [[0.5_f64, 0.2], [0.0, 0.4]];
        let b = [1.0_f64, 2.0];
        let g = |x: &[f64]| -> Vec<f64> {
            sonium_solvers_core::array![
                a[0][0] * x[0] + a[0][1] * x[1] + b[0],
                a[1][0] * x[0] + a[1][1] * x[1] + b[1],
            ]
        };
        // Exact fixed point: x0 = (b0 + 0.2/0.6 * b1) / 0.5, x1 = b1 / 0.6
        let x1_exact = b[1] / (1.0 - a[1][1]);
        let x0_exact = (b[0] + a[0][1] * x1_exact) / (1.0 - a[0][0]);

        let config = AndersonConfig {
            depth: 3,
            absolute_tolerance: 1e-14,
            ..Default::default()
        };
        let result = anderson_acceleration(&[0.0, 0.0], g, &config).unwrap();
        assert!(result.converged);
        assert!(
            result.iterations <= 8,
            "linear 2-D problem should converge fast, took {}",
            result.iterations
        );
        assert!((result.x[0] - x0_exact).abs() < 1e-10);
        assert!((result.x[1] - x1_exact).abs() < 1e-10);
    }

    #[test]
    fn test_anderson_complex64() {
        // g(z) = 0.5 z + (1 + i): contraction with fixed point z* = 2(1 + i).
        let c = Complex64::new(1.0, 1.0);
        let g = move |x: &[Complex64]| -> Vec<Complex64> { vec![x[0] * 0.5 + c] };

        let config = AndersonConfig {
            depth: 3,
            absolute_tolerance: 1e-12,
            ..Default::default()
        };
        let result = anderson_acceleration(&[Complex64::new(0.0, 0.0)], g, &config).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - Complex64::new(2.0, 2.0)).norm() < 1e-10);
    }

    #[test]
    fn test_anderson_f32() {
        let x0 = sonium_solvers_core::array![0.5_f32];
        let config = AndersonConfig {
            depth: 4,
            absolute_tolerance: 1e-5,
            ..Default::default()
        };
        let result =
            anderson_acceleration(&x0, |x| sonium_solvers_core::array![x[0].cos()], &config)
                .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 0.73908513_f32).abs() < 1e-4);
    }

    #[test]
    fn test_anderson_already_converged() {
        // Initial guess is the fixed point: zero iterations.
        let config = AndersonConfig::default();
        let fp = 0.7390851332151607_f64;
        let result =
            anderson_acceleration(&[fp], |x| sonium_solvers_core::array![x[0].cos()], &config)
                .unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.status, SolverStatus::Converged);
        assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
    }

    #[test]
    fn test_anderson_default_matches_vendor_parity_setup() {
        // Regression guard for the x86 vendor-parity work: the default must
        // stay unsafeguarded depth-5 / mixing-1 (PETSc SNESANDERSON applies
        // the update unconditionally). Safeguarding by default more than
        // doubled the step count on smooth contractions (66 vs 27 steps at
        // n = 32000, ρ = 0.95); opt in explicitly for fragile maps.
        let config = AndersonConfig::default();
        assert_eq!(config.depth, 5);
        assert_eq!(config.mixing, 1.0);
        assert!(!config.safeguarding);
    }

    #[test]
    fn test_anderson_accelerates_stiff_contraction() {
        // G_i(x) = ρ·x_i + (1−ρ)·φ_i(x), ρ = 0.95: plain Picard needs ~360
        // steps for 1e-8 relative. Anderson(5) must do far better; the bound
        // below (measured ~25 steps at n = 500) guards both the least-squares
        // path and the default setup against regressions.
        let n = 500usize;
        let x0 = vec![0.0_f64; n];
        let config = AndersonConfig {
            max_iterations: 500,
            ..Default::default()
        };
        let result = anderson_acceleration(
            &x0,
            |x: &[f64]| {
                let mut out = vec![0.0; n];
                for i in 0..n {
                    let xm = x[(i + n - 1) % n];
                    let xp = x[(i + 1) % n];
                    out[i] =
                        0.95 * x[i] + 0.05 * (((xm + xp) * 0.25 + 0.01 * i as f64).sin() * 0.5);
                }
                out
            },
            &config,
        )
        .unwrap();
        assert!(result.converged);
        assert!(
            result.iterations <= 60,
            "Anderson lost its acceleration: {} iters",
            result.iterations
        );
    }

    #[test]
    fn anderson_errors_expose_canonical_status_and_accuracy() {
        let not_converged = AndersonError::NotConverged(4, 1.0);
        assert_eq!(not_converged.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(
            not_converged.attainable_accuracy(),
            AccuracyStatus::NotAttained
        );

        let diverged = AndersonError::Diverged(2, 2.0);
        assert_eq!(diverged.status(), SolverStatus::Diverged);
    }

    #[test]
    fn test_anderson_safeguarding_fallback() {
        // A mildly expansive map far from the fixed point: g(x) = cos(x) with
        // a large perturbation, so raw Anderson combinations can overshoot.
        // Safeguarding must still drive the residual to zero.
        let config = AndersonConfig {
            depth: 8,
            mixing: 0.7,
            absolute_tolerance: 1e-10,
            // Tightened so the run converges on the absolute tolerance: the
            // default relative tolerance (1e-8) stops at residual ~4e-8,
            // which only bounds the fixed-point error to ~2e-8 — looser than
            // the 1e-8 assertion below.
            relative_tolerance: 1e-14,
            safeguarding: true,
            max_iterations: 300,
            ..Default::default()
        };
        let result = anderson_acceleration(
            &[3.0_f64],
            |x| sonium_solvers_core::array![x[0].cos()],
            &config,
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 0.7390851332151607).abs() < 1e-8);
    }

    #[test]
    fn test_anderson_not_converged_reports_error() {
        // Non-contractive map g(x) = x + 1: residual never shrinks.
        let config = AndersonConfig {
            max_iterations: 7,
            safeguarding: false,
            ..Default::default()
        };
        let result = anderson_acceleration(
            &[0.0_f64],
            |x| sonium_solvers_core::array![x[0] + 1.0],
            &config,
        );
        assert!(matches!(
            result,
            Err(AndersonError::NotConverged(7, _)) | Err(AndersonError::Diverged(_, _))
        ));
    }

    #[test]
    fn test_anderson_gamma_exact_rank_deficiency() {
        // ΔF with an exactly duplicated column (col1 = 2·col0): the MGS-QR
        // basis must drop it (γ = 0) and still return the exact least-squares
        // solution instead of failing on a singular Gram matrix.
        let c0 = vec![1.0_f64, 2.0, -1.0, 0.5];
        let c1 = sonium_solvers_core::vec_ops::vec_scale(&c0, 2.0);
        let c2 = vec![0.0_f64, 1.0, 1.0, -2.0];
        let f = vec![3.0_f64, -1.0, 2.0, 1.0];

        let df = [c0.clone(), c1, c2.clone()];
        let gamma = anderson_gamma(&df, &f, &mut AndersonQrScratch::default())
            .expect("rank-deficient ΔF must not fail");
        assert_eq!(gamma.len(), 3);
        assert_eq!(gamma[1], 0.0, "dependent column must be dropped with γ = 0");

        // Reference: the same problem with only the independent columns.
        let df_indep = [c0, c2];
        let gamma_ref = anderson_gamma(&df_indep, &f, &mut AndersonQrScratch::default()).unwrap();
        let residual = |df: &[Vec<f64>], g: &[f64]| -> f64 {
            let mut r = f.clone();
            for (j, &gj) in g.iter().enumerate() {
                for (ri, dfi) in r.iter_mut().zip(df[j].iter()) {
                    *ri -= gj * dfi;
                }
            }
            r.iter().map(|x| x * x).sum::<f64>().sqrt()
        };
        let res = residual(&df, &gamma);
        let res_indep = residual(&df_indep, &gamma_ref);
        assert!(
            (res - res_indep).abs() < 1e-13,
            "dropped-column solution must match the independent-basis solution: {res} vs {res_indep}"
        );
    }

    #[test]
    fn test_anderson_gamma_zero_history_returns_none() {
        // An all-zero ΔF carries no information: the solver must report None
        // so the caller falls back to a damped fixed-point step.
        let df = [vec![0.0_f64; 3], vec![0.0_f64; 3]];
        let f = vec![1.0_f64, 2.0, 3.0];
        assert!(anderson_gamma(&df, &f, &mut AndersonQrScratch::default()).is_none());
    }

    #[test]
    fn test_anderson_gamma_nearly_dependent_columns_f32() {
        // Two columns dependent at relative ~1e-3 in f32: κ(ΔF) ≈ 2e3, so the
        // normal equations face κ²·eps_f32 ≈ 0.5 and lose most digits, while
        // the MGS-QR faces only κ·eps_f32 ≈ 2e-4. (At f64 both methods get by
        // until the relative dependence reaches ~1e-8, at which point the
        // dependence detector drops the column instead — see
        // test_anderson_gamma_exact_rank_deficiency.)
        //
        // Columns: c1 = e1, c2 = e1 + 1e-3 e2, c3 = e3 with orthonormal
        // e1, e2, e3; f = 2 e1 + 5 e2 + 3 e3. Exact solution:
        // γ2 = 5000, γ1 = 2 − γ2 = −4998, γ3 = 3, residual 0.
        let s = 0.5_f32.sqrt();
        let e1 = vec![s, s, 0.0, 0.0];
        let e2 = vec![s, -s, 0.0, 0.0];
        let e3 = vec![0.0_f32, 0.0, 1.0, 0.0];
        let c1 = e1.clone();
        let c2: Vec<f32> = e1.iter().zip(&e2).map(|(a, b)| a + 1e-3 * b).collect();
        let c3 = e3.clone();
        let f: Vec<f32> = (0..4)
            .map(|i| 2.0 * e1[i] + 5.0 * e2[i] + 3.0 * e3[i])
            .collect();

        let df = [c1, c2, c3];
        let gamma = anderson_gamma(&df, &f, &mut AndersonQrScratch::default())
            .expect("near-dependent ΔF must be solvable");
        // The third column is unaffected by the ill-conditioning.
        assert!(
            (gamma[2] - 3.0).abs() < 1e-3,
            "γ3 = {} should be 3",
            gamma[2]
        );
        // Residual of the combination, evaluated in f64 for an accurate norm.
        let mut r: Vec<f64> = f.iter().map(|x| *x as f64).collect();
        for (j, &gj) in gamma.iter().enumerate() {
            for (ri, &dfi) in r.iter_mut().zip(df[j].iter()) {
                *ri -= gj as f64 * dfi as f64;
            }
        }
        let res = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            res < 1e-2,
            "QR residual {res} should stay small; the normal equations give O(1) here"
        );
    }

    #[test]
    fn test_anderson_rank_deficient_history_end_to_end() {
        // 1-D linear map: every Δf column is an exact scalar multiple of the
        // others, so ΔF is exactly rank 1 once the history holds ≥ 2 entries.
        // Column-dependence detection must shrink the basis gracefully and
        // still converge to the fixed point x* = 2.
        let config = AndersonConfig {
            depth: 5,
            absolute_tolerance: 1e-14,
            ..Default::default()
        };
        let result = anderson_acceleration(
            &[0.0_f64],
            |x| sonium_solvers_core::array![0.3 * x[0] + 1.4],
            &config,
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-12);
    }
}
