//! Parallel Classical Gram-Schmidt building blocks for GMRES Arnoldi.
//!
//! The standard CGS step at iteration `j` of Arnoldi computes
//!
//! ```text
//!   h[i, j] = <v[i], w>      for i = 0..=j
//!   w      -= sum_i h[i, j] * v[i]
//! ```
//!
//! Both passes are independent across `i` (pass 1) or across rows of `w`
//! (pass 2), so they parallelise cleanly. For small problems we keep the
//! sequential path to avoid rayon dispatch overhead.
//!
//! These helpers replace per-`i` calls to `inner_product` / `axpy` and
//! collapse what was effectively `O(j)` sequential BLAS calls per Arnoldi
//! step into one parallel section over the basis (or one parallel section
//! over rows of `w`).

use rayon::prelude::*;
use sonium_solvers_core::blas_helpers::{axpy, inner_product};
use sonium_solvers_core::traits::ComplexField;

/// Below this vector length the parallel reduction paths (rayon `join` for a
/// fused norm+dot pair, batched Gram pairs) add more dispatch overhead than
/// they save. Single shared value for all Krylov reduction gates in this
/// crate; measured on Apple Silicon (M4). Re-tune per platform with a
/// `dense_gemv`-style sweep before changing the default; dot-vs-Gram
/// crossovers differ, so keep one value until such a sweep justifies
/// splitting it. Env-overridable (`SONIUM_PAR_REDUCTION`).
pub(crate) const PAR_REDUCTION_THRESHOLD: usize = 8_192;

/// Projection pass benefits from parallelism a bit later because each row
/// also walks the basis dimension. Env-overridable
/// (`SONIUM_PAR_PROJECT`).
const PAR_PROJECT_THRESHOLD: usize = 16_384;

pub(crate) fn par_reduction_threshold() -> usize {
    // Resolved once per process: Arnoldi consults this on every iteration
    // (dots + projection) and a `std::env::var` round-trip per call is pure
    // overhead. Same caching contract as the sparse SpMV thresholds — set
    // the variable before process start.
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("SONIUM_PAR_REDUCTION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(PAR_REDUCTION_THRESHOLD)
    })
}

fn par_project_threshold() -> usize {
    // Same once-per-process caching as `par_reduction_threshold` above.
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("SONIUM_PAR_PROJECT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(PAR_PROJECT_THRESHOLD)
    })
}

/// Contiguous Krylov basis storage.
///
/// Basis vector `i` occupies `data[i * len..(i + 1) * len]`. Keeping the
/// restart basis flat gives GMRES/FGMRES a single allocation, predictable
/// streaming access, and a straightforward byte model.
#[derive(Debug, Clone)]
pub(super) struct FlatBasis<T: ComplexField> {
    data: Vec<T>,
    vectors: usize,
    len: usize,
}

impl<T: ComplexField> FlatBasis<T> {
    pub(super) fn new(vectors: usize, len: usize) -> Self {
        Self {
            data: vec![T::zero(); vectors * len],
            vectors,
            len,
        }
    }

    pub(super) fn resize_vectors(&mut self, vectors: usize) {
        self.data.resize(vectors * self.len, T::zero());
        self.vectors = vectors;
    }

    pub(super) fn vector(&self, index: usize) -> &[T] {
        debug_assert!(index < self.vectors);
        let start = index * self.len;
        &self.data[start..start + self.len]
    }

    pub(super) fn vector_mut(&mut self, index: usize) -> &mut [T] {
        debug_assert!(index < self.vectors);
        let start = index * self.len;
        &mut self.data[start..start + self.len]
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn vectors(&self) -> usize {
        self.vectors
    }
}

/// Compute `out[i] = <basis[i], w>` for `i in 0..k`.
///
/// Parallelises over `i` (independent reductions, no false sharing) for
/// vectors above the threshold; the per-`i` reduction itself is the existing
/// SIMD/BLAS-backed `inner_product`.
/// Compute `out[i] = <basis[i], w>` for `i in 0..k`, reusing `out`.
pub(super) fn batched_dots_into<T: ComplexField>(
    basis: &[Vec<T>],
    k: usize,
    w: &[T],
    out: &mut Vec<T>,
) {
    debug_assert!(k <= basis.len());
    out.resize(k, T::zero());
    if k == 0 {
        return;
    }

    let n = w.len();
    if n < par_reduction_threshold() || k == 1 {
        for i in 0..k {
            out[i] = inner_product(&basis[i], w);
        }
        return;
    }

    out.par_iter_mut()
        .enumerate()
        .for_each(|(i, coeff)| *coeff = inner_product(&basis[i], w));
}

/// Compute `w -= sum_{i=0..k} coeffs[i] * basis[i]`.
///
/// For large vectors we parallelise over chunks of `w` and have each task
/// walk the basis dimension; this fuses what was a serial loop of `k` axpys
/// into one parallel scan.
pub(super) fn batched_project<T: ComplexField>(basis: &[Vec<T>], coeffs: &[T], w: &mut [T]) {
    debug_assert_eq!(basis.len().min(coeffs.len()), coeffs.len());

    let k = coeffs.len();
    if k == 0 {
        return;
    }

    let n = w.len();
    if n < par_project_threshold() || k == 1 {
        for i in 0..k {
            axpy(T::zero() - coeffs[i], &basis[i], w);
        }
        return;
    }

    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads).max(4_096);

    w.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(c_idx, w_chunk)| {
            let row0 = c_idx * chunk;
            for (off, w_k) in w_chunk.iter_mut().enumerate() {
                let row = row0 + off;
                let mut sum = T::zero();
                for i in 0..k {
                    sum += coeffs[i] * basis[i][row];
                }
                *w_k -= sum;
            }
        });
}

/// Compute `out[i] = <basis[i], w>` for a contiguous basis.
pub(super) fn batched_dots_flat_into<T: ComplexField>(
    basis: &FlatBasis<T>,
    k: usize,
    w: &[T],
    out: &mut Vec<T>,
) {
    debug_assert!(k <= basis.vectors());
    debug_assert_eq!(w.len(), basis.len());
    out.resize(k, T::zero());
    if k == 0 {
        return;
    }

    let n = w.len();
    if n < par_reduction_threshold() || k == 1 {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = inner_product(basis.vector(i), w);
        }
        return;
    }

    out.par_iter_mut()
        .enumerate()
        .for_each(|(i, coeff)| *coeff = inner_product(basis.vector(i), w));
}

/// Compute `w -= sum_i coeffs[i] * basis[i]` for a contiguous basis.
pub(super) fn batched_project_flat<T: ComplexField>(
    basis: &FlatBasis<T>,
    coeffs: &[T],
    w: &mut [T],
) {
    debug_assert!(coeffs.len() <= basis.vectors());
    debug_assert_eq!(w.len(), basis.len());

    let k = coeffs.len();
    if k == 0 {
        return;
    }

    let n = w.len();
    if n < par_project_threshold() || k == 1 {
        for (i, &coef) in coeffs.iter().enumerate() {
            axpy(T::zero() - coef, basis.vector(i), w);
        }
        return;
    }

    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads).max(4_096);

    w.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(c_idx, w_chunk)| {
            let row0 = c_idx * chunk;
            for (off, w_k) in w_chunk.iter_mut().enumerate() {
                let row = row0 + off;
                let mut sum = T::zero();
                for (i, &coef) in coeffs.iter().enumerate() {
                    sum += coef * basis.vector(i)[row];
                }
                *w_k -= sum;
            }
        });
}

/// Compute `x += basis * coeffs` in one streamed pass.
pub(super) fn fused_basis_update_flat<T: ComplexField>(
    basis: &FlatBasis<T>,
    coeffs: &[T],
    x: &mut [T],
) {
    debug_assert!(coeffs.len() <= basis.vectors());
    debug_assert_eq!(x.len(), basis.len());

    let k = coeffs.len();
    if k == 0 {
        return;
    }

    let n = x.len();
    if n < par_project_threshold() || k == 1 {
        for (i, &coef) in coeffs.iter().enumerate() {
            axpy(coef, basis.vector(i), x);
        }
        return;
    }

    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads).max(4_096);

    x.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(c_idx, x_chunk)| {
            let row0 = c_idx * chunk;
            for (off, x_k) in x_chunk.iter_mut().enumerate() {
                let row = row0 + off;
                let mut sum = T::zero();
                for (i, &coef) in coeffs.iter().enumerate() {
                    sum += coef * basis.vector(i)[row];
                }
                *x_k += sum;
            }
        });
}

/// Krylov operation and byte-budget counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KrylovOpCounts {
    pub spmv_count: usize,
    pub preconditioner_count: usize,
    pub dot_reductions: usize,
    pub norm_count: usize,
    pub axpy_passes: usize,
    pub basis_bytes_touched: u64,
    pub synchronization_points: usize,
    pub exact_residual_checks: usize,
}

impl KrylovOpCounts {
    pub fn add_basis_projection<T: ComplexField>(&mut self, basis_vectors: usize, n: usize) {
        self.dot_reductions += basis_vectors;
        self.axpy_passes += 1;
        self.synchronization_points += 1;
        self.basis_bytes_touched += (basis_vectors * n * std::mem::size_of::<T>() * 2) as u64;
    }

    pub fn add_basis_update<T: ComplexField>(&mut self, basis_vectors: usize, n: usize) {
        self.axpy_passes += 1;
        self.basis_bytes_touched +=
            (basis_vectors * n * std::mem::size_of::<T>() + n * std::mem::size_of::<T>()) as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn make_basis(n: usize, k: usize) -> Vec<Vec<Complex64>> {
        (0..k)
            .map(|i| {
                (0..n)
                    .map(|r| {
                        Complex64::new(
                            ((i + 1) as f64 * (r as f64).sin()).cos(),
                            ((i + 1) as f64 * (r as f64).cos()).sin(),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn batched_dots_matches_serial() {
        let n = 33_000; // above the parallel threshold
        let k = 7;
        let basis = make_basis(n, k);
        let w = (0..n)
            .map(|r| Complex64::new(r as f64, -(r as f64) * 0.5))
            .collect::<Vec<_>>();

        let mut par = Vec::new();
        batched_dots_into(&basis, k, &w, &mut par);
        let serial: Vec<_> = (0..k).map(|i| inner_product(&basis[i], &w)).collect();

        assert_eq!(par.len(), serial.len());
        // The parallel-over-i path runs each `inner_product` on a rayon
        // worker (sequential SIMD kernel) while the serial path calls it
        // from the test thread (parallel chunked-sum kernel). Both paths
        // are correct but accumulate in different orders, so the values
        // can differ by a few ULPs. Compare relatively, not absolutely.
        for (a, b) in par.iter().zip(serial.iter()) {
            let scale = a.norm().max(b.norm()).max(1.0);
            let rel = (a - b).norm() / scale;
            assert!(rel < 1e-12, "rel diff {rel:.3e} exceeds 1e-12");
        }
    }

    #[test]
    fn batched_project_matches_serial() {
        let n = 50_000;
        let k = 5;
        let basis = make_basis(n, k);
        let coeffs: Vec<Complex64> = (0..k)
            .map(|i| Complex64::new((i as f64).cos(), (i as f64).sin()))
            .collect();

        let mut w = (0..n)
            .map(|r| Complex64::new(r as f64, 1.0))
            .collect::<Vec<_>>();
        let mut w_ref = w.clone();

        batched_project(&basis, &coeffs, &mut w);
        for i in 0..k {
            axpy(Complex64::new(0.0, 0.0) - coeffs[i], &basis[i], &mut w_ref);
        }

        for (a, b) in w.iter().zip(w_ref.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-9);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-9);
        }
    }

    #[test]
    fn flat_basis_matches_vec_basis_helpers() {
        let n = 50_000;
        let k = 5;
        let basis_vec = make_basis(n, k);
        let mut basis_flat = FlatBasis::new(k, n);
        for (i, v) in basis_vec.iter().enumerate() {
            basis_flat.vector_mut(i).copy_from_slice(v);
        }
        let coeffs: Vec<Complex64> = (0..k)
            .map(|i| Complex64::new((i as f64).cos(), (i as f64).sin()))
            .collect();
        let mut w_vec = (0..n)
            .map(|r| Complex64::new(r as f64, 1.0))
            .collect::<Vec<_>>();
        let mut w_flat = w_vec.clone();
        let mut dots_vec = Vec::new();
        let mut dots_flat = Vec::new();

        batched_dots_into(&basis_vec, k, &w_vec, &mut dots_vec);
        batched_dots_flat_into(&basis_flat, k, &w_vec, &mut dots_flat);
        batched_project(&basis_vec, &coeffs, &mut w_vec);
        batched_project_flat(&basis_flat, &coeffs, &mut w_flat);

        for (a, b) in dots_vec.iter().zip(dots_flat.iter()) {
            let scale = a.norm().max(b.norm()).max(1.0);
            assert!((a - b).norm() / scale < 1e-12);
        }
        for (a, b) in w_vec.iter().zip(w_flat.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-9);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-9);
        }
    }

    #[test]
    fn small_vectors_use_serial_path() {
        let n = 32; // well below threshold
        let k = 3;
        let basis = make_basis(n, k);
        let w = (0..n)
            .map(|r| Complex64::new(r as f64, 0.0))
            .collect::<Vec<_>>();

        let mut dots = Vec::new();
        batched_dots_into(&basis, k, &w, &mut dots);
        assert_eq!(dots.len(), k);

        let coeffs = vec![Complex64::new(1.0, 0.0); k];
        let mut wm = w.clone();
        batched_project(&basis, &coeffs, &mut wm); // must not panic
    }
}
