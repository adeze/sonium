//! CMA-ES: bounded full-covariance Covariance Matrix Adaptation Evolution
//! Strategy for continuous black-box minimisation, after Hansen & Ostermeier
//! (2001). Internally the search runs in a normalised `[0, 1]^n` box so
//! parameters of different scales share one covariance matrix without manual
//! preconditioning; sampled points are clipped into the box before evaluation.
//!
//! This is a port of the proven implementation in `math-optimisation::cmaes`
//! with the following adaptations to this crate:
//!
//! - The per-generation symmetric eigendecomposition of the covariance matrix
//!   uses the crate's own Householder + implicit-shift QL dense eigensolver
//!   ([`sonium_solvers_eigen`]) instead of nalgebra's `SymmetricEigen`.
//! - Gaussian sampling uses a deterministic xorshift64* generator with
//!   Box-Muller (mirroring `direct::multifrontal_blr::truncation`) instead of
//!   the `rand` crate. With `seed` set, runs are bitwise reproducible.
//! - Offspring are evaluated with `rayon` on the ambient thread pool once the
//!   generation size reaches 4; below that a serial loop avoids thread
//!   overhead. No private thread pool is built.
//! - The result follows the crate's [`OptimResult`] convention. Termination
//!   reasons map onto [`OptimStatus`]: target objective reached or stagnation
//!   -> [`OptimStatus::FunctionTolerance`], step-size collapse ->
//!   [`OptimStatus::StepTolerance`], budget exhausted ->
//!   [`OptimStatus::MaxFunctionEvals`].
//!
//! f64 only: the dense covariance eigendecomposition is f64-only in this
//! crate, and CMA-ES needs the covariance spectrum in full precision — an
//! f32 port would need its own eigensolver and offers no practical win for a
//! black-box optimiser dominated by objective evaluations.

use rayon::prelude::*;

use crate::types::{Bounds, OptimError, OptimResult, OptimStatus};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_eigen::symmetric_eigen;

/// Configuration for [`cma_es`].
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Variable bounds. Every variable must have finite lower and upper
    /// bounds (`Bound::both`): the algorithm normalises the box to
    /// `[0, 1]^n` internally. `lower == upper` is accepted (the variable is
    /// then effectively fixed); `lower > upper` is an error.
    pub bounds: Bounds<f64>,
    /// Optional initial mean in original coordinates. `None` starts at the
    /// centre of the box. Values outside the bounds are clipped.
    pub x0: Option<Vec<f64>>,
    /// Initial global step size in normalised `[0, 1]` coordinates.
    /// The default `0.3` is the standard broad-search setting for bounded
    /// CMA-ES; use smaller values for local refinement.
    pub sigma0: f64,
    /// Offspring population size. `0` uses `4 + floor(3 ln n)`.
    pub lambda: usize,
    /// Number of parents used for recombination. `0` uses `lambda / 2`.
    pub mu: usize,
    /// Maximum number of objective evaluations.
    pub max_fun_evals: usize,
    /// Optional RNG seed for deterministic runs. `None` seeds from the
    /// system clock.
    pub seed: Option<u64>,
    /// Stop after this many generations with best-objective improvement
    /// below [`Self::f_tol`]. `0` disables stagnation detection.
    pub stagnation_window: usize,
    /// Objective-improvement tolerance for stagnation detection.
    pub f_tol: f64,
    /// Stop (converged) once the best objective is at or below this value.
    pub target_f: f64,
    /// Evaluate offspring in parallel with rayon once the per-generation
    /// evaluation count reaches 4. Offspring generation stays serial, so
    /// results are identical with or without parallelism.
    pub parallel: bool,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            x0: None,
            sigma0: 0.3,
            lambda: 0,
            mu: 0,
            max_fun_evals: 10_000,
            seed: None,
            stagnation_window: 80,
            f_tol: 1e-10,
            target_f: f64::NEG_INFINITY,
            parallel: true,
        }
    }
}

/// Minimise `f(x)` subject to box constraints with bounded full-covariance
/// CMA-ES. Derivative-free; only objective values are used.
///
/// The objective receives points in the original coordinate system; bounds
/// are enforced by clipping the normalised samples before evaluation. See the
/// module documentation for the termination-status mapping.
pub fn cma_es<F>(f: &F, config: &CmaEsConfig) -> Result<OptimResult<f64>, OptimError>
where
    F: Fn(&[f64]) -> f64 + Sync + ?Sized,
{
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cma_es_impl(f, config)))
        .map_err(|_| OptimError::CallbackPanicked {
            iteration: 0,
            stage: "CMA-ES callback".into(),
        })??;
    super::validate_result(&result)?;
    Ok(result)
}

fn cma_es_impl<F>(f: &F, config: &CmaEsConfig) -> Result<OptimResult<f64>, OptimError>
where
    F: Fn(&[f64]) -> f64 + Sync + ?Sized,
{
    let n = config.bounds.len();
    if n == 0 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "cma_es requires at least one bounded variable".to_string(),
        });
    }
    let mut lower = vec![0.0; n];
    let mut upper = vec![0.0; n];
    for (i, b) in config.bounds.iter().enumerate() {
        let (Some(lo), Some(hi)) = (b.lower, b.upper) else {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: format!(
                    "cma_es requires finite lower and upper bounds (variable {i} is unbounded)"
                ),
            });
        };
        if lo > hi {
            return Err(OptimError::InvalidBounds {
                index: i,
                lower: lo,
                upper: hi,
            });
        }
        lower[i] = lo;
        upper[i] = hi;
    }
    if let Some(ref x0) = config.x0
        && x0.len() != n
    {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: x0.len(),
        });
    }
    let preflight_x = config.x0.clone().unwrap_or_else(|| {
        lower
            .iter()
            .zip(&upper)
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect()
    });
    super::validate_objective_start(f, &preflight_x)?;

    let lambda = if config.lambda == 0 {
        (4.0 + (3.0 * (n as f64).ln()).floor()).max(4.0) as usize
    } else {
        config.lambda
    };
    if lambda < 2 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("population size (lambda) must be >= 2, got {lambda}"),
        });
    }
    let mu = if config.mu == 0 {
        lambda / 2
    } else {
        config.mu.min(lambda)
    }
    .max(1);

    let weights = recombination_weights(mu);
    let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    let n_f = n as f64;

    let cc = (4.0 + mueff / n_f) / (n_f + 4.0 + 2.0 * mueff / n_f);
    let cs = (mueff + 2.0) / (n_f + mueff + 5.0);
    let c1 = 2.0 / ((n_f + 1.3).powi(2) + mueff);
    let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n_f + 2.0).powi(2) + mueff));
    let damps = 1.0 + 2.0 * ((mueff - 1.0) / (n_f + 1.0)).sqrt().max(1.0) - 2.0 + cs;
    let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

    let mut mean = initial_mean(config, &lower, &upper);
    let mut old_mean = vec![0.0; n];
    let mut sigma = config.sigma0.clamp(1e-12, 2.0);
    // Dense n×n matrices in flat row-major layout.
    let mut covariance = identity(n);
    let mut b = identity(n);
    let mut d = vec![1.0; n];
    let mut invsqrt_c = identity(n);
    let mut pc = vec![0.0; n];
    let mut ps = vec![0.0; n];

    // Reusable per-generation scratch buffers.
    let mut z_buf = vec![0.0; n];
    let mut scaled_z = vec![0.0; n];
    let mut step_buf = vec![0.0; n];
    let mut y_w = vec![0.0; n];
    let mut tmp_n = vec![0.0; n];
    let mut rank_mu = vec![0.0; n * n];

    // Pre-allocated population storage so offspring vectors are reused across
    // generations instead of re-allocated each iteration.
    let mut y_pool: Vec<Vec<f64>> = (0..lambda).map(|_| vec![0.0; n]).collect();
    let mut funs: Vec<f64> = vec![0.0; lambda];
    let mut order: Vec<usize> = (0..lambda).collect();

    let mut rng = XorShift64Star::new(match config.seed {
        Some(s) => s,
        None => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x853c_49e6_748f_ea9b),
    });

    let mut best_x = denormalise(&mean, &lower, &upper);
    let mut best_fun = evaluate_objective(f, &best_x, 0)?;
    let mut nfev = 1usize;
    let mut nit = 0usize;
    let mut last_improvement_fun = best_fun;
    let mut stagnation_counter = 0usize;
    let mut status = OptimStatus::MaxFunctionEvals;
    let mut converged = false;

    while nfev < config.max_fun_evals {
        std::mem::swap(&mut mean, &mut old_mean);
        let eval_budget = (config.max_fun_evals - nfev).min(lambda);

        // Generate offspring in normalised coordinates. Instead of forming
        // the full `B * diag(d)` transform, sample `z ~ N(0, I)` and apply
        // the equivalent cheaper `B * (d .* z)`.
        for y_i in y_pool.iter_mut().take(eval_budget) {
            fill_standard_normal(&mut z_buf, &mut rng);
            for j in 0..n {
                scaled_z[j] = d[j] * z_buf[j];
            }
            gemv(&b, &scaled_z, &mut step_buf);
            for j in 0..n {
                y_i[j] = (old_mean[j] + step_buf[j] * sigma).clamp(0.0, 1.0);
            }
        }

        // Evaluate offspring. The denormalised points are computed first so
        // the parallel path only shares read-only data.
        let xs: Vec<Vec<f64>> = y_pool[..eval_budget]
            .iter()
            .map(|y| denormalise(y, &lower, &upper))
            .collect();
        let evaluated: Result<Vec<_>, _> = if config.parallel && eval_budget >= 4 {
            xs.par_iter()
                .enumerate()
                .map(|(i, x)| evaluate_objective(f, x, nfev + i))
                .collect()
        } else {
            xs.iter()
                .enumerate()
                .map(|(i, x)| evaluate_objective(f, x, nfev + i))
                .collect()
        };
        funs[..eval_budget].copy_from_slice(&evaluated?);
        nfev += eval_budget;

        // Track the best point seen so far.
        for i in 0..eval_budget {
            if funs[i] < best_fun {
                best_fun = funs[i];
                best_x.clone_from(&xs[i]);
            }
        }

        if eval_budget == 0 {
            break;
        }

        // Sort by fitness and select the best `mu_used` parents.
        order[..eval_budget].sort_by(|&a, &b| funs[a].total_cmp(&funs[b]));
        let mu_used = mu.min(eval_budget);

        // Recombine parents into the new mean.
        mean.fill(0.0);
        for i in 0..mu_used {
            let idx = order[i];
            let w = weights[i];
            let y_i = &y_pool[idx];
            for j in 0..n {
                mean[j] += w * y_i[j];
            }
        }
        for v in mean.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }

        // Evolution paths.
        for j in 0..n {
            y_w[j] = (mean[j] - old_mean[j]) / sigma.max(1e-30);
        }
        gemv(&invsqrt_c, &y_w, &mut tmp_n);
        let ps_factor = (cs * (2.0 - cs) * mueff).sqrt();
        for j in 0..n {
            ps[j] = ps[j] * (1.0 - cs) + tmp_n[j] * ps_factor;
        }
        let norm_ps = ps.iter().map(|v| v * v).sum::<f64>().sqrt();

        let hsig_den = (1.0 - (1.0 - cs).powi(2 * (nit as i32 + 1))).sqrt() * chi_n;
        let hsig = if hsig_den > 0.0 {
            norm_ps / hsig_den < 1.4 + 2.0 / (n_f + 1.0)
        } else {
            true
        };

        let pc_factor = (cc * (2.0 - cc) * mueff).sqrt();
        for j in 0..n {
            pc[j] *= 1.0 - cc;
            if hsig {
                pc[j] += y_w[j] * pc_factor;
            }
        }

        // Rank-mu update matrix.
        rank_mu.fill(0.0);
        let inv_sigma = 1.0 / sigma.max(1e-30);
        for i in 0..mu_used {
            let idx = order[i];
            let w = weights[i];
            let y_i = &y_pool[idx];
            for j in 0..n {
                let diff_j = (y_i[j] - old_mean[j]) * inv_sigma;
                for k in 0..n {
                    let diff_k = (y_i[k] - old_mean[k]) * inv_sigma;
                    rank_mu[j * n + k] += w * diff_j * diff_k;
                }
            }
        }

        // Covariance matrix update.
        let hsig_correction = if hsig { 0.0 } else { c1 * cc * (2.0 - cc) };
        let cov_scale = 1.0 - c1 - cmu + hsig_correction;
        for j in 0..n {
            for k in 0..n {
                covariance[j * n + k] = covariance[j * n + k] * cov_scale
                    + c1 * pc[j] * pc[k]
                    + cmu * rank_mu[j * n + k];
            }
        }
        symmetrise_and_regularise(&mut covariance, n);

        // Step-size update.
        sigma *= ((cs / damps) * (norm_ps / chi_n - 1.0)).exp();
        sigma = sigma.clamp(1e-14, 10.0);

        // Eigen-decomposition of the updated covariance matrix (Householder
        // tridiagonalisation + implicit-shift QL). Eigenpairs come back
        // unordered, which is fine: `C = B diag(d^2) B^T` for any ordering.
        let cov_mat = DenseMatrix::from_row_major_vec(n, n, covariance.clone());
        let (eigenvalues, eigenvectors) =
            symmetric_eigen(&cov_mat, n).map_err(|e| OptimError::NumericalError {
                iteration: nit,
                message: format!("covariance eigendecomposition failed: {e}"),
            })?;
        for j in 0..n {
            for l in 0..n {
                b[j * n + l] = eigenvectors[[j, l]];
            }
        }
        for l in 0..n {
            d[l] = eigenvalues[l].max(1e-30).sqrt();
        }

        // Recompute C^{-1/2} = B * diag(1/d) * B^T without materialising the
        // intermediate diagonal matrix.
        for j in 0..n {
            for k in 0..n {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += b[j * n + l] * b[k * n + l] / d[l].max(1e-30);
                }
                invsqrt_c[j * n + k] = sum;
            }
        }

        nit += 1;
        if (last_improvement_fun - best_fun).abs() <= config.f_tol {
            stagnation_counter += 1;
        } else {
            stagnation_counter = 0;
            last_improvement_fun = best_fun;
        }

        if best_fun <= config.target_f {
            status = OptimStatus::FunctionTolerance;
            converged = true;
            break;
        }
        if config.stagnation_window > 0 && stagnation_counter >= config.stagnation_window {
            status = OptimStatus::FunctionTolerance;
            converged = true;
            break;
        }
        if sigma < 1e-12 {
            status = OptimStatus::StepTolerance;
            converged = true;
            break;
        }
    }

    Ok(OptimResult {
        x: best_x,
        fun: best_fun,
        grad: None,
        iterations: nit,
        fun_evals: nfev,
        grad_evals: 0,
        hess_evals: 0,
        grad_norm: f64::INFINITY,
        converged,
        status,
        kkt: None,
    })
}

fn recombination_weights(mu: usize) -> Vec<f64> {
    let mu_f = mu as f64;
    let mut weights: Vec<f64> = (1..=mu)
        .map(|i| (mu_f + 0.5).ln() - (i as f64).ln())
        .collect();
    let sum = weights.iter().sum::<f64>();
    for w in &mut weights {
        *w /= sum;
    }
    weights
}

fn initial_mean(config: &CmaEsConfig, lower: &[f64], upper: &[f64]) -> Vec<f64> {
    let n = lower.len();
    if let Some(ref x0) = config.x0 {
        let mut y = vec![0.0; n];
        for i in 0..n {
            let span = upper[i] - lower[i];
            y[i] = if span > 0.0 {
                ((x0[i].clamp(lower[i], upper[i]) - lower[i]) / span).clamp(0.0, 1.0)
            } else {
                0.5
            };
        }
        y
    } else {
        vec![0.5; n]
    }
}

fn denormalise(y: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    (0..y.len())
        .map(|i| lower[i] + y[i].clamp(0.0, 1.0) * (upper[i] - lower[i]))
        .collect()
}

fn evaluate_objective<F>(f: &F, x: &[f64], iteration: usize) -> Result<f64, OptimError>
where
    F: Fn(&[f64]) -> f64 + ?Sized,
{
    let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(x))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: "CMA-ES objective".into(),
        }
    })?;
    if !value.is_finite() {
        return Err(OptimError::NumericalError {
            iteration,
            message: "objective callback returned a non-finite value".into(),
        });
    }
    Ok(value)
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// In-place dense matrix-vector multiply `y = A * x` (row-major flat `A`).
fn gemv(a: &[f64], x: &[f64], y: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

fn symmetrise_and_regularise(c: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (c[i * n + j] + c[j * n + i]);
            c[i * n + j] = v;
            c[j * n + i] = v;
        }
        c[i * n + i] = c[i * n + i].max(1e-30);
    }
}

/// Deterministic xorshift64* generator, mirroring the one in
/// `direct::multifrontal_blr::truncation`.
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

/// Fill `out` with standard normal samples via Box-Muller.
fn fill_standard_normal(out: &mut [f64], rng: &mut XorShift64Star) {
    let n = out.len();
    let mut i = 0usize;
    while i < n {
        let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
        let u2 = rng.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out[i] = radius * theta.cos();
        if i + 1 < n {
            out[i + 1] = radius * theta.sin();
        }
        i += 2;
    }
}

#[cfg(test)]
#[allow(deprecated)] // head-to-head benchmarks against the deprecated powell
mod tests {
    use super::*;
    use crate::test_functions::{rosenbrock_nd, rosenbrock_nd_start};
    use crate::types::Bound;
    use crate::unconstrained::{BobyqaConfig, PowellConfig, bobyqa, powell};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    // ---- Ported from the source implementation's test suite -------------------

    #[test]
    fn cma_es_converges_on_sphere() {
        let report = cma_es(
            &sphere,
            &CmaEsConfig {
                bounds: vec![Bound::both(-5.0, 5.0); 4],
                max_fun_evals: 5_000,
                seed: Some(42),
                target_f: 1e-10,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");

        assert!(
            report.fun < 1e-6,
            "CMA-ES should converge near origin, got {}",
            report.fun
        );
    }

    #[test]
    fn cma_es_handles_coupled_rotated_quadratic() {
        let rotated = |x: &[f64]| {
            let u = (x[0] + x[1]) / 2.0_f64.sqrt();
            let v = (x[0] - x[1]) / 2.0_f64.sqrt();
            1_000.0 * u * u + v * v
        };
        let report = cma_es(
            &rotated,
            &CmaEsConfig {
                bounds: vec![Bound::both(-3.0, 3.0), Bound::both(-3.0, 3.0)],
                max_fun_evals: 4_000,
                seed: Some(7),
                target_f: 1e-9,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");

        assert!(
            report.fun < 1e-5,
            "CMA-ES should solve rotated ill-conditioned quadratic, got {}",
            report.fun
        );
    }

    #[test]
    fn cma_es_rejects_empty_bounds() {
        let result = cma_es(&sphere, &CmaEsConfig::default());
        assert!(result.is_err(), "Empty bounds should error");
    }

    #[test]
    fn cma_es_rejects_inverted_bounds() {
        let result = cma_es(
            &sphere,
            &CmaEsConfig {
                bounds: vec![Bound::both(1.0, -1.0)],
                ..Default::default()
            },
        );
        assert!(result.is_err(), "Inverted bounds should error");
    }

    #[test]
    fn cma_es_rejects_unbounded_variable() {
        let result = cma_es(
            &sphere,
            &CmaEsConfig {
                bounds: vec![Bound::lower(-1.0)],
                ..Default::default()
            },
        );
        assert!(result.is_err(), "Unbounded variable should error");
    }

    #[test]
    fn cma_es_rejects_mismatched_x0() {
        let result = cma_es(
            &sphere,
            &CmaEsConfig {
                bounds: vec![Bound::both(-5.0, 5.0); 3],
                x0: Some(vec![0.0, 0.0]),
                ..Default::default()
            },
        );
        assert!(result.is_err(), "Mismatched x0 dimension should error");
    }

    #[test]
    fn cma_es_rejects_nonfinite_generated_objective_values() {
        let calls = AtomicUsize::new(0);
        let objective = |_: &[f64]| {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                0.0
            } else {
                f64::NAN
            }
        };
        let error = cma_es(
            &objective,
            &CmaEsConfig {
                bounds: vec![Bound::both(-1.0, 1.0)],
                max_fun_evals: 2,
                seed: Some(1),
                ..Default::default()
            },
        )
        .expect_err("generated non-finite objective must fail the CMA-ES contract");
        assert!(matches!(
            error,
            OptimError::NumericalError { message, .. }
                if message.contains("non-finite")
        ));
    }

    #[test]
    fn cma_es_target_f_reached() {
        let report = cma_es(
            &sphere,
            &CmaEsConfig {
                bounds: vec![Bound::both(-5.0, 5.0); 2],
                max_fun_evals: 10_000,
                seed: Some(42),
                target_f: 1.0,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun <= 1.0,
            "Should stop when target_f reached: f={}",
            report.fun
        );
        assert!(report.converged);
        assert_eq!(report.status, OptimStatus::FunctionTolerance);
    }

    #[test]
    fn cma_es_single_dimension() {
        let one_d = |x: &[f64]| x[0] * x[0];
        let report = cma_es(
            &one_d,
            &CmaEsConfig {
                bounds: vec![Bound::both(-5.0, 5.0)],
                max_fun_evals: 2_000,
                seed: Some(42),
                target_f: 1e-8,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun < 1e-6,
            "1D CMA-ES should converge: f={}",
            report.fun
        );
    }

    #[test]
    fn cma_es_custom_lambda_mu() {
        let report = cma_es(
            &sphere,
            &CmaEsConfig {
                bounds: vec![Bound::both(-5.0, 5.0); 2],
                lambda: 20,
                mu: 8,
                max_fun_evals: 3_000,
                seed: Some(42),
                target_f: 1e-8,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun < 1e-6,
            "Custom lambda/mu should converge: f={}",
            report.fun
        );
    }

    #[test]
    fn cma_es_sigma0_override() {
        let report = cma_es(
            &sphere,
            &CmaEsConfig {
                bounds: vec![Bound::both(-5.0, 5.0); 2],
                sigma0: 0.1,
                max_fun_evals: 3_000,
                seed: Some(42),
                target_f: 1e-8,
                ..Default::default()
            },
        )
        .expect("CMA-ES should run");
        assert!(
            report.fun < 1e-6,
            "Small sigma0 should still converge: f={}",
            report.fun
        );
    }

    // ---- Determinism -----------------------------------------------------------

    #[test]
    fn cma_es_same_seed_is_bitwise_reproducible() {
        let config = CmaEsConfig {
            bounds: vec![Bound::both(-5.0, 5.0); 5],
            x0: Some(rosenbrock_nd_start(5)),
            max_fun_evals: 20_000,
            seed: Some(123),
            ..Default::default()
        };
        let r1 = cma_es(&rosenbrock_nd, &config).unwrap();
        let r2 = cma_es(&rosenbrock_nd, &config).unwrap();
        assert_eq!(r1.fun_evals, r2.fun_evals);
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.fun.to_bits(), r2.fun.to_bits());
    }

    #[test]
    fn cma_es_different_seeds_both_converge() {
        for seed in [1u64, 7, 42, 999] {
            let report = cma_es(
                &sphere,
                &CmaEsConfig {
                    bounds: vec![Bound::both(-5.0, 5.0); 4],
                    max_fun_evals: 5_000,
                    seed: Some(seed),
                    target_f: 1e-8,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(
                report.fun < 1e-6,
                "seed {seed} should converge, got {}",
                report.fun
            );
        }
    }

    // ---- Head-to-head benchmark: CMA-ES vs BOBYQA vs Powell -------------------

    struct BenchEntry {
        solver: &'static str,
        evals: usize,
        evals_to_tol: Option<usize>,
        fun: f64,
        converged: bool,
    }

    fn print_table(problem: &str, tol: f64, rows: &[BenchEntry]) {
        println!("\n=== {problem} (tolerance {tol:e}) ===");
        println!(
            "{:<8} {:>12} {:>14} {:>14} {:>10}",
            "solver", "final f", "evals", "evals-to-tol", "converged"
        );
        for r in rows {
            let ett = r
                .evals_to_tol
                .map(|e| e.to_string())
                .unwrap_or_else(|| "-".to_string());
            println!(
                "{:<8} {:>12.6e} {:>14} {:>14} {:>10}",
                r.solver, r.fun, r.evals, ett, r.converged
            );
        }
    }

    fn run_cma(
        f: &(dyn Fn(&[f64]) -> f64 + Sync),
        x0: &[f64],
        bounds: &[(f64, f64)],
        tol: f64,
        max_evals: usize,
        seed: u64,
    ) -> BenchEntry {
        let config = CmaEsConfig {
            bounds: bounds.iter().map(|&(lo, hi)| Bound::both(lo, hi)).collect(),
            x0: Some(x0.to_vec()),
            max_fun_evals: max_evals,
            seed: Some(seed),
            target_f: tol,
            ..Default::default()
        };
        let r = cma_es(f, &config).unwrap();
        BenchEntry {
            solver: "cma_es",
            evals: r.fun_evals,
            evals_to_tol: (r.fun <= tol).then_some(r.fun_evals),
            fun: r.fun,
            converged: r.converged,
        }
    }

    fn run_bobyqa(
        f: &(dyn Fn(&[f64]) -> f64 + Sync),
        x0: &[f64],
        bounds: &[(f64, f64)],
        tol: f64,
        max_evals: usize,
    ) -> BenchEntry {
        let count = Cell::new(0usize);
        let first_tol = Cell::new(None::<usize>);
        let wrapped = |x: &[f64]| {
            let c = count.get() + 1;
            count.set(c);
            let v = f(x);
            if v <= tol && first_tol.get().is_none() {
                first_tol.set(Some(c));
            }
            v
        };
        let config = BobyqaConfig {
            bounds: bounds.iter().map(|&(lo, hi)| Bound::both(lo, hi)).collect(),
            max_fun_evals: max_evals,
            ..Default::default()
        };
        let r = bobyqa(&wrapped, x0, &config).unwrap();
        BenchEntry {
            solver: "bobyqa",
            evals: r.fun_evals,
            evals_to_tol: first_tol.get(),
            fun: r.fun,
            converged: r.converged,
        }
    }

    fn run_powell(
        f: &(dyn Fn(&[f64]) -> f64 + Sync),
        x0: &[f64],
        tol: f64,
        max_evals: usize,
    ) -> BenchEntry {
        let count = Cell::new(0usize);
        let first_tol = Cell::new(None::<usize>);
        let wrapped = |x: &[f64]| {
            let c = count.get() + 1;
            count.set(c);
            let v = f(x);
            if v <= tol && first_tol.get().is_none() {
                first_tol.set(Some(c));
            }
            v
        };
        let config = PowellConfig {
            max_iterations: 5_000,
            max_fun_evals: max_evals,
            ..Default::default()
        };
        let r = powell(&wrapped, x0, &config).unwrap();
        BenchEntry {
            solver: "powell",
            evals: r.fun_evals,
            evals_to_tol: first_tol.get(),
            fun: r.fun,
            converged: r.converged,
        }
    }

    #[test]
    fn bench_sphere_10d() {
        let tol = 1e-6;
        let x0 = vec![2.0; 10];
        let bounds = vec![(-5.0, 5.0); 10];
        let rows = [
            run_cma(&sphere, &x0, &bounds, tol, 50_000, 42),
            run_bobyqa(&sphere, &x0, &bounds, tol, 100_000),
            run_powell(&sphere, &x0, tol, 100_000),
        ];
        print_table("sphere 10D", tol, &rows);
        assert!(rows[0].fun <= tol, "cma_es fun = {}", rows[0].fun);
    }

    #[test]
    fn bench_rosenbrock_2d() {
        let tol = 1e-6;
        let x0 = rosenbrock_nd_start(2);
        let bounds = vec![(-5.0, 5.0); 2];
        let rows = [
            run_cma(&rosenbrock_nd, &x0, &bounds, tol, 30_000, 42),
            run_bobyqa(&rosenbrock_nd, &x0, &bounds, tol, 100_000),
            run_powell(&rosenbrock_nd, &x0, tol, 100_000),
        ];
        print_table("rosenbrock 2D", tol, &rows);
        assert!(rows[0].fun <= tol, "cma_es fun = {}", rows[0].fun);
    }

    #[test]
    fn bench_rosenbrock_10d() {
        let tol = 1e-4;
        let x0 = rosenbrock_nd_start(10);
        let bounds = vec![(-5.0, 5.0); 10];
        let rows = [
            run_cma(&rosenbrock_nd, &x0, &bounds, tol, 200_000, 42),
            run_bobyqa(&rosenbrock_nd, &x0, &bounds, tol, 100_000),
            run_powell(&rosenbrock_nd, &x0, tol, 100_000),
        ];
        print_table("rosenbrock 10D", tol, &rows);
        assert!(rows[0].fun <= tol, "cma_es fun = {}", rows[0].fun);
    }

    /// Rotated ellipsoid with 1e6:1 conditioning: `f(x) = sum c_i (Qx)_i^2`
    /// with `c_i = 10^(6i/(n-1))` and `Q` a fixed Householder reflector.
    fn rotated_ellipsoid(x: &[f64]) -> f64 {
        let n = x.len();
        let v: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let vv: f64 = v.iter().map(|t| t * t).sum();
        // y = Q x with Q = I - 2 v v^T / (v^T v)
        let vx: f64 = v.iter().zip(x).map(|(a, b)| a * b).sum();
        let mut f = 0.0;
        for i in 0..n {
            let y = x[i] - 2.0 * v[i] * vx / vv;
            let c = 10f64.powf(6.0 * i as f64 / (n as f64 - 1.0));
            f += c * y * y;
        }
        f
    }

    #[test]
    fn bench_rotated_ellipsoid_10d() {
        let tol = 1e-4;
        let x0 = vec![1.0; 10];
        let bounds = vec![(-10.0, 10.0); 10];
        let rows = [
            run_cma(&rotated_ellipsoid, &x0, &bounds, tol, 100_000, 42),
            run_bobyqa(&rotated_ellipsoid, &x0, &bounds, tol, 100_000),
            run_powell(&rotated_ellipsoid, &x0, tol, 100_000),
        ];
        print_table("rotated ellipsoid 10D (cond 1e6)", tol, &rows);
        assert!(rows[0].fun <= tol, "cma_es fun = {}", rows[0].fun);
    }

    /// Ackley: multimodal, global min f = 0 at the origin, nearly flat outer
    /// region full of local minima.
    fn ackley(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|v| v * v).sum();
        let sum_cos: f64 = x
            .iter()
            .map(|v| (2.0 * std::f64::consts::PI * v).cos())
            .sum();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    }

    #[test]
    fn bench_ackley_5d_multimodal() {
        let tol = 1e-2;
        let x0 = vec![5.0; 5];
        let bounds = vec![(-15.0, 15.0); 5];
        let rows = [
            run_cma(&ackley, &x0, &bounds, tol, 100_000, 42),
            run_bobyqa(&ackley, &x0, &bounds, tol, 100_000),
            run_powell(&ackley, &x0, tol, 100_000),
        ];
        print_table("ackley 5D (multimodal)", tol, &rows);
        assert!(rows[0].fun <= tol, "cma_es fun = {}", rows[0].fun);
        // From this start, the local solvers settle into a local minimum
        // while CMA-ES finds the global basin.
        assert!(
            rows[0].fun < rows[1].fun,
            "cma_es ({}) should beat bobyqa ({}) on ackley",
            rows[0].fun,
            rows[1].fun
        );
        assert!(
            rows[0].fun < rows[2].fun,
            "cma_es ({}) should beat powell ({}) on ackley",
            rows[0].fun,
            rows[2].fun
        );
    }

    /// Deterministic mildly noisy quadratic (same value on every call).
    fn noisy_quadratic_5d(x: &[f64]) -> f64 {
        let target = [0.5, -1.0, 2.0, -0.5, 1.5];
        let base: f64 = (0..5).map(|i| (x[i] - target[i]).powi(2)).sum();
        let phase = 1.0e5 * (1.1 * x[0] + 1.7 * x[1] + 2.3 * x[2] + 0.7 * x[3] + 1.9 * x[4]);
        base + 1e-3 * phase.sin()
    }

    #[test]
    fn bench_noisy_quadratic_5d() {
        // Noise floor: the objective jitters by ±1e-3, so "converged" means
        // reaching the minimiser region, not driving f to 1e-6.
        let tol = 2e-3;
        let target = [0.5, -1.0, 2.0, -0.5, 1.5];
        let x0 = vec![0.0; 5];
        let bounds = vec![(-5.0, 5.0); 5];
        let rows = [
            run_cma(&noisy_quadratic_5d, &x0, &bounds, tol, 50_000, 42),
            run_bobyqa(&noisy_quadratic_5d, &x0, &bounds, tol, 100_000),
            run_powell(&noisy_quadratic_5d, &x0, tol, 100_000),
        ];
        print_table("noisy quadratic 5D (amplitude 1e-3)", tol, &rows);
        assert!(rows[0].fun <= tol, "cma_es fun = {}", rows[0].fun);
        // CMA-ES should land genuinely close to the minimiser.
        let cma_x_err = |x: &[f64]| {
            (0..5)
                .map(|i| (x[i] - target[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        // With target_f set, CMA-ES stops on the first sub-tolerance sample,
        // which the ±1e-3 noise can trigger early. A run stopped by
        // stagnation instead polishes the mean into the true minimiser.
        let config = CmaEsConfig {
            bounds: bounds.iter().map(|&(lo, hi)| Bound::both(lo, hi)).collect(),
            x0: Some(x0.clone()),
            max_fun_evals: 50_000,
            seed: Some(42),
            ..Default::default()
        };
        let r = cma_es(&noisy_quadratic_5d, &config).unwrap();
        assert!(
            cma_x_err(&r.x) < 5e-2,
            "cma_es distance to minimiser = {}",
            cma_x_err(&r.x)
        );
    }
}
