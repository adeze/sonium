//! CUDA symmetric eigendecomposition via cuSOLVER (`cusolverDnDsyevd`).
//!
//! The CPU path ([`crate::lobpcg::symmetric_eigen`]) is a from-scratch
//! Householder + divide-and-conquer implementation tuned for x86. On the
//! GPU the state of the art is NVIDIA's own divide-and-conquer `Dsyevd`
//! (the same family every vendor stack — MAGMA included — routes standard
//! dense symmetric problems through), so this entry point is a thin,
//! convention-preserving wrapper over
//! [`sonium_compute::CudaBackend::dense_syevd_f64`]: same input/output
//! contract as [`crate::lobpcg::symmetric_eigen`] (eigenvalues ascending
//! only up to the vendor's ordering — cuSOLVER returns ascending;
//! column `i` of the returned matrix is the matching eigenvector),
//! host-side allocation discipline, and [`SolverError`] mapping. There is
//! no custom GPU secular kernel to maintain: matching cuSOLVER here means
//! calling it.
//!
//! Available with the crate's `cuda` feature. Requires a CUDA-capable
//! device at runtime ([`sonium_compute::CudaBackend::new_default`] errors
//! otherwise).

pub use sonium_compute::CudaBackend;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::SolverError;

/// Dense symmetric eigendecomposition on the GPU.
///
/// Convenience wrapper: builds a [`CudaBackend`] per call (device setup
/// ~0.1 s, amortized away at large `n` but dominant below `n ~ 1000`).
/// For benchmarks and repeated solves, build the backend once and call
/// [`symmetric_eigen_cuda_on`].
///
/// `a` must be square `n × n` and finite (checked here, mirroring the CPU
/// entry). Returns eigenvalues in ascending order with eigenvectors as the
/// columns of the returned row-major matrix — the same contract as
/// [`crate::lobpcg::symmetric_eigen`], verified element-wise by
/// `tests/cuda_syevd.rs`.
pub fn symmetric_eigen_cuda(
    a: &DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    let backend = CudaBackend::new_default().map_err(|e| SolverError::OperationFailed {
        operation: "symmetric CUDA eigensolver setup",
        reason: e.to_string(),
    })?;
    symmetric_eigen_cuda_on(&backend, a, n)
}

/// Dense symmetric eigendecomposition on an existing [`CudaBackend`].
///
/// Same contract as [`symmetric_eigen_cuda`], without per-call device
/// setup: build the backend once with [`CudaBackend::new_default`] and
/// reuse it across solves.
pub fn symmetric_eigen_cuda_on(
    backend: &CudaBackend,
    a: &DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    if a.nrows() != n || a.ncols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: a.nrows().max(a.ncols()),
        });
    }
    if !a.as_slice().iter().all(|v| v.is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "symmetric CUDA eigensolver input",
            reason: "operator generated NaN or infinity".into(),
        });
    }
    // Row-major symmetric input reads as column-major without a transpose.
    let (evals, colmajor_vecs) =
        backend
            .dense_syevd_f64(n, a.as_slice())
            .map_err(|e| SolverError::OperationFailed {
                operation: "cusolverDnDsyevd",
                reason: e.to_string(),
            })?;
    if !evals.iter().all(|v| v.is_finite()) || !colmajor_vecs.iter().all(|v| v.is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "symmetric CUDA eigensolver output",
            reason: "calculation generated NaN or infinity".into(),
        });
    }
    // Column-major eigenvectors to the row-major matrix convention.
    // Blocked (tile) transpose: the naive double loop runs at scalar
    // gather/scatter rate (~50-100 ms at n=4000); 32x32 tiles stream
    // both sides contiguously and auto-vectorize. Pure data movement —
    // bitwise-identical to the naive loop.
    let mut z = vec![0.0f64; n * n];
    const TILE: usize = 32;
    for jj in (0..n).step_by(TILE) {
        let je = (jj + TILE).min(n);
        for ii in (0..n).step_by(TILE) {
            let ie = (ii + TILE).min(n);
            for i in ii..ie {
                for j in jj..je {
                    z[i * n + j] = colmajor_vecs[j * n + i];
                }
            }
        }
    }
    Ok((evals, DenseMatrix::from_row_major_vec(n, n, z)))
}

/// Default crossover size for [`symmetric_eigen_auto_on`].
///
/// Measured on Blackwell RTX PRO 6000 + EPYC 9654: the CPU path
/// (Householder + divide-and-conquer) wins at `n <= 500`, cuSOLVER wins
/// at `n >= 700`, crossover sits at `n ~ 550-600`. The default adds a
/// noise margin above the crossover for the ±15% GPU contention seen on
/// shared boxes. Override per-process with `SONIUM_EIGEN_GPU_MIN_N`
/// (positive integer; invalid values fall back to this default).
pub const AUTO_GPU_MIN_N: usize = 640;

fn resolve_gpu_min_n(var: Option<String>) -> usize {
    var.and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(AUTO_GPU_MIN_N)
}

fn auto_gpu_min_n() -> usize {
    resolve_gpu_min_n(std::env::var("SONIUM_EIGEN_GPU_MIN_N").ok())
}

/// Dense symmetric eigendecomposition with automatic CPU/GPU routing.
///
/// Below the crossover ([`AUTO_GPU_MIN_N`], tunable via
/// `SONIUM_EIGEN_GPU_MIN_N`) this calls
/// [`crate::lobpcg::symmetric_eigen`]; at or above it calls
/// [`symmetric_eigen_cuda_on`] on the provided backend. Same
/// input/output contract on both branches (finite square input,
/// eigenvalues with matching column eigenvectors; ordering follows
/// whichever branch ran — ascending on GPU, unspecified on CPU).
pub fn symmetric_eigen_auto_on(
    backend: &CudaBackend,
    a: &DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    if n >= auto_gpu_min_n() {
        symmetric_eigen_cuda_on(backend, a, n)
    } else {
        crate::lobpcg::symmetric_eigen(a, n)
    }
}

#[cfg(test)]
mod tests {
    use super::AUTO_GPU_MIN_N;
    use super::resolve_gpu_min_n;

    #[test]
    fn gpu_threshold_override_parsing() {
        assert_eq!(resolve_gpu_min_n(None), AUTO_GPU_MIN_N);
        assert_eq!(resolve_gpu_min_n(Some("700".into())), 700);
        assert_eq!(resolve_gpu_min_n(Some("0".into())), AUTO_GPU_MIN_N);
        assert_eq!(resolve_gpu_min_n(Some("abc".into())), AUTO_GPU_MIN_N);
        assert_eq!(resolve_gpu_min_n(Some("".into())), AUTO_GPU_MIN_N);
    }
}
