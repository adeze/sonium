//! Preconditioned Richardson iteration — the WaveSim algorithm.
//!
//! Per `wavesim_py.iteration.preconditioned_iteration`:
//!
//! ```text
//! tmp = B·x
//! tmp = B·x + c·y
//! tmp = (L+1)⁻¹ · tmp
//! tmp = x − tmp
//! tmp = B·tmp
//! x   = x − α · tmp
//! ```
//!
//! The residual is `‖tmp‖²` after the final medium application, normalized by
//! `‖B(L+1)⁻¹y‖²` (the preconditioned source norm).

use num_complex::Complex64;

use crate::array::Array3;

use super::HelmholtzDomain;

/// Controls for the WaveSim Richardson iteration.
#[derive(Debug, Clone, Copy)]
pub struct IterationOptions {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold on the normalized **squared** residual norm
    /// `‖tmp‖²/init_norm` (not the RMS residual). The default `1e-6` is
    /// therefore equivalent to an RMS tolerance of `1e-3`; set `1e-12` for
    /// an RMS tolerance of `1e-6`.
    pub threshold: f64,
    /// Relaxation factor α (default 0.75 per wavesim_py).
    pub alpha: f64,
}

impl Default for IterationOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100_000,
            threshold: 1.0e-6,
            alpha: 0.75,
        }
    }
}

/// Run the preconditioned Richardson iteration. Returns `(u, iterations, residual)`.
pub fn run_richardson(
    domain: &HelmholtzDomain,
    source: &Array3<Complex64>,
    options: IterationOptions,
) -> (Array3<Complex64>, usize, f64) {
    assert_eq!(source.shape(), domain.padded_shape());
    let mut x = Array3::<Complex64>::zeros(domain.padded_shape());

    // init_norm = ‖c · B(L+1)⁻¹ y‖² (matches wavesim_py.iteration.preconditioner).
    let init_norm = {
        let mut tmp = source.clone();
        domain.propagator_inplace(&mut tmp);
        domain.medium_inplace(&mut tmp);
        for v in tmp.as_mut_slice().iter_mut() {
            *v *= domain.scale;
        }
        norm_squared(&tmp).max(1e-300)
    };

    let mut residual = f64::INFINITY;
    let mut iterations = 0;

    // Reused scratch buffer: the loop below used to allocate a full-grid
    // `domain.medium(&x)` clone per iteration. Copy + in-place apply keeps
    // the exact same operation order with one allocation total.
    let mut tmp = Array3::<Complex64>::zeros(domain.padded_shape());
    for it in 0..options.max_iterations {
        // tmp = B · x
        tmp.as_mut_slice().copy_from_slice(x.as_slice());
        domain.medium_inplace(&mut tmp);
        // tmp = B·x − c·y  (matches wavesim_py.iteration.preconditioned_iteration's `scale(-c, src, offset=tmp)`).
        for (t, s) in tmp.as_mut_slice().iter_mut().zip(source.as_slice().iter()) {
            *t -= domain.scale * *s;
        }
        // tmp = (L+1)⁻¹ tmp
        domain.propagator_inplace(&mut tmp);
        // tmp = x − tmp
        for (t, xi) in tmp.as_mut_slice().iter_mut().zip(x.as_slice().iter()) {
            *t = *xi - *t;
        }
        // tmp = B · tmp
        domain.medium_inplace(&mut tmp);

        residual = norm_squared(&tmp) / init_norm;

        // x ← x − α · tmp
        let alpha = Complex64::new(options.alpha, 0.0);
        for (xi, t) in x.as_mut_slice().iter_mut().zip(tmp.as_slice().iter()) {
            *xi -= alpha * *t;
        }

        iterations = it + 1;
        if residual < options.threshold {
            break;
        }
    }

    (x, iterations, residual)
}

fn norm_squared(a: &Array3<Complex64>) -> f64 {
    let mut s = 0.0;
    for v in a.as_slice().iter() {
        s += v.norm_sqr();
    }
    s
}
