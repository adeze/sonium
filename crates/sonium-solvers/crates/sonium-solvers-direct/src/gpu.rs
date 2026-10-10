//! CUDA-backed dense LU factorization and solve for `f64`.
//!
//! Solvers-facing wrapper over the `sonium-compute` cuSOLVERDN path
//! (`cusolverDnDgetrf`/`Dgetrs`). Row-major [`DenseMatrix`] input is
//! transposed to column-major once on upload; factors stay resident on the
//! device behind [`GpuDenseLuF64`]. Single solves upload one right-hand side
//! per call ([`lu_solve_gpu`]); [`lu_solve_many_gpu`] batches `nrhs` columns
//! into one call; [`GpuDenseRhsF64`] uploads once and serves repeated solves
//! with no further host-to-device traffic.
//!
//! The Cholesky twins ([`cholesky_factorize_gpu`], [`cholesky_solve_gpu`],
//! [`cholesky_solve_many_gpu`]) wrap `cusolverDnDpotrf`/`Dpotrs` the same
//! way; symmetric input needs no device transpose.

use sonium_compute::{
    Backend, Buffer, CudaBackend, CudaDenseCholF64, CudaDenseLuF64, MemoryPool, OomError, Transpose,
};
use sonium_solvers_core::dense::DenseMatrix;
use thiserror::Error;

/// Failures from the GPU dense-LU path.
#[derive(Debug, Error)]
pub enum GpuLuError {
    /// Host-side shape violation detected before touching the device.
    #[error("GPU LU dimension mismatch: {0}")]
    DimensionMismatch(String),
    /// cuSOLVERDN, transfer, or allocation failure from the compute backend.
    #[error("GPU LU backend error: {0}")]
    Backend(#[from] OomError),
}

/// Device-resident `f64` dense LU factors (cuSOLVERDN `Dgetrf`).
pub struct GpuDenseLuF64 {
    factors: CudaDenseLuF64,
}

impl GpuDenseLuF64 {
    /// Matrix dimension of the factorization.
    pub fn n(&self) -> usize {
        self.factors.n()
    }

    /// Number of `f64` workspace elements retained on the device.
    pub fn workspace_len(&self) -> usize {
        self.factors.workspace_len()
    }
}

/// Factor a square row-major `f64` matrix on `backend`, keeping the LU
/// factors on the device for repeated solves.
///
/// The row-major bytes upload untouched; the transpose to column-major runs
/// on the device (`cublasDgeam`), not in a host-side O(n²) scalar loop.
pub fn lu_factorize_gpu(
    backend: &CudaBackend,
    a: &DenseMatrix<f64>,
) -> Result<GpuDenseLuF64, GpuLuError> {
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(GpuLuError::DimensionMismatch(format!(
            "expected a square matrix, got {rows}x{cols}"
        )));
    }
    let n = rows;
    let row_major = a.as_slice();
    debug_assert_eq!(row_major.len(), n * n);
    let factors = backend.dense_lu_f64_from_row_major(n, row_major)?;
    Ok(GpuDenseLuF64 { factors })
}

/// Solve `A x = b` — or `Aᵀ x = b` when `transposed` — from resident GPU
/// factors, returning the solution on the host.
pub fn lu_solve_gpu(
    pool: &MemoryPool<CudaBackend>,
    lu: &GpuDenseLuF64,
    b: &[f64],
    transposed: bool,
) -> Result<Vec<f64>, GpuLuError> {
    let n = lu.n();
    if b.len() != n {
        return Err(GpuLuError::DimensionMismatch(format!(
            "expected a right-hand side of length {n}, got {}",
            b.len()
        )));
    }
    let backend = pool.backend();
    let mut rhs_dev = pool.try_alloc::<f64>(n)?;
    let mut out_dev = pool.try_alloc::<f64>(n)?;
    backend.copy_h2d(b, rhs_dev.raw_mut());
    let operation = if transposed {
        Transpose::Trans
    } else {
        Transpose::NoTrans
    };
    backend.dense_lu_f64_solve(&lu.factors, rhs_dev.raw(), out_dev.raw_mut(), operation)?;
    let mut x = vec![0.0f64; n];
    backend.copy_d2h(out_dev.raw(), &mut x);
    Ok(x)
}

/// Solve `A X = B` — or `Aᵀ X = B` when `transposed` — for `nrhs`
/// right-hand sides in a single device call from resident GPU factors.
///
/// Both `b` and the returned solution use row-major `n × nrhs` storage
/// (all right-hand sides for a matrix row are contiguous), matching
/// [`crate::LuFactorization::solve_many_into`]. The host-side layout transpose to
/// the column-major order cuSOLVER expects is O(`n * nrhs`); all
/// `O(n²)` work stays on the device, and launch plus transfer overhead is
/// paid once for all columns instead of once per right-hand side.
pub fn lu_solve_many_gpu(
    pool: &MemoryPool<CudaBackend>,
    lu: &GpuDenseLuF64,
    b: &[f64],
    nrhs: usize,
    transposed: bool,
) -> Result<Vec<f64>, GpuLuError> {
    let n = lu.n();
    let expected = n.checked_mul(nrhs).ok_or_else(|| {
        GpuLuError::DimensionMismatch(format!(
            "n={n} x nrhs={nrhs} right-hand-side size overflows usize"
        ))
    })?;
    if nrhs == 0 || b.len() != expected {
        return Err(GpuLuError::DimensionMismatch(format!(
            "expected a row-major {n}x{nrhs} right-hand side ({expected} values), got {}",
            b.len()
        )));
    }
    // Row-major host (i, j) -> column-major device (i, j): one streaming pass
    // each way, dwarfed by the O(n^2 * nrhs) triangular solves.
    let mut col_major = vec![0.0f64; expected];
    for i in 0..n {
        for j in 0..nrhs {
            col_major[i + j * n] = b[i * nrhs + j];
        }
    }
    let backend = pool.backend();
    let mut rhs_dev = pool.try_alloc::<f64>(expected)?;
    let mut out_dev = pool.try_alloc::<f64>(expected)?;
    backend.copy_h2d(&col_major, rhs_dev.raw_mut());
    let operation = if transposed {
        Transpose::Trans
    } else {
        Transpose::NoTrans
    };
    backend.dense_lu_f64_solve_many(
        &lu.factors,
        rhs_dev.raw(),
        out_dev.raw_mut(),
        nrhs,
        operation,
    )?;
    backend.copy_d2h(out_dev.raw(), &mut col_major);
    let mut x = vec![0.0f64; expected];
    for i in 0..n {
        for j in 0..nrhs {
            x[i * nrhs + j] = col_major[i + j * n];
        }
    }
    Ok(x)
}

/// Device-resident `f64` right-hand side(s) for repeated GPU solves.
///
/// Holds `nrhs` columns in column-major `n x nrhs` order on the device,
/// uploaded once by [`lu_rhs_to_gpu`]. Each [`lu_solve_resident_gpu`] call
/// then pays only the device-side triangular solves plus one
/// device-to-host download — no host-to-device transfer per solve.
/// The guard is pool-owned (`Arc` inside), so the handle outlives the
/// borrow of the pool it was allocated from and frees its quota on drop.
pub struct GpuDenseRhsF64 {
    buf: Buffer<f64, CudaBackend>,
    n: usize,
    nrhs: usize,
}

impl GpuDenseRhsF64 {
    /// Matrix dimension of the resident right-hand side.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of resident right-hand-side columns.
    pub fn nrhs(&self) -> usize {
        self.nrhs
    }

    /// Device element count (`n * nrhs`).
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the handle holds no entries.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// Upload row-major `n x nrhs` right-hand sides once, returning a
/// device-resident handle for repeated [`lu_solve_resident_gpu`] calls.
///
/// The row count `n` is derived from `b.len() / nrhs` and checked against
/// the factorization at solve time, so one handle serves any same-size
/// system. The host-side layout transpose is O(`n * nrhs`); it runs once
/// here, never per solve.
pub fn lu_rhs_to_gpu(
    pool: &MemoryPool<CudaBackend>,
    b: &[f64],
    nrhs: usize,
) -> Result<GpuDenseRhsF64, GpuLuError> {
    if nrhs == 0 || b.is_empty() || !b.len().is_multiple_of(nrhs) {
        return Err(GpuLuError::DimensionMismatch(format!(
            "expected a non-empty row-major n x {nrhs} right-hand side, got {} values",
            b.len()
        )));
    }
    let n = b.len() / nrhs;
    let expected = b.len();
    let mut col_major = vec![0.0f64; expected];
    for i in 0..n {
        for j in 0..nrhs {
            col_major[i + j * n] = b[i * nrhs + j];
        }
    }
    let backend = pool.backend();
    let mut buf = pool.try_alloc::<f64>(expected)?;
    backend.copy_h2d(&col_major, buf.raw_mut());
    Ok(GpuDenseRhsF64 { buf, n, nrhs })
}

/// Solve `A X = B` — or `Aᵀ X = B` when `transposed` — from resident GPU
/// factors against a resident right-hand side, returning the row-major
/// `n x nrhs` solution on the host.
///
/// Unlike [`lu_solve_many_gpu`], this performs zero host-to-device traffic:
/// the right-hand side was uploaded once by [`lu_rhs_to_gpu`]. Use it for
/// iterative refinement or repeated solves against a fixed `B`.
pub fn lu_solve_resident_gpu(
    pool: &MemoryPool<CudaBackend>,
    lu: &GpuDenseLuF64,
    rhs: &GpuDenseRhsF64,
    transposed: bool,
) -> Result<Vec<f64>, GpuLuError> {
    let n = lu.n();
    if rhs.n != n {
        return Err(GpuLuError::DimensionMismatch(format!(
            "resident right-hand side has n={}, factorization has n={n}",
            rhs.n
        )));
    }
    let expected = n * rhs.nrhs;
    let backend = pool.backend();
    let mut out_dev = pool.try_alloc::<f64>(expected)?;
    let operation = if transposed {
        Transpose::Trans
    } else {
        Transpose::NoTrans
    };
    backend.dense_lu_f64_solve_many(
        &lu.factors,
        rhs.buf.raw(),
        out_dev.raw_mut(),
        rhs.nrhs,
        operation,
    )?;
    let mut col_major = vec![0.0f64; expected];
    backend.copy_d2h(out_dev.raw(), &mut col_major);
    let mut x = vec![0.0f64; expected];
    for i in 0..n {
        for j in 0..rhs.nrhs {
            x[i * rhs.nrhs + j] = col_major[i + j * n];
        }
    }
    Ok(x)
}

/// Failures from the GPU dense-Cholesky path.
#[derive(Debug, Error)]
pub enum GpuCholError {
    /// Host-side shape violation detected before touching the device.
    #[error("GPU Cholesky dimension mismatch: {0}")]
    DimensionMismatch(String),
    /// cuSOLVERDN, transfer, or allocation failure from the compute backend.
    #[error("GPU Cholesky backend error: {0}")]
    Backend(#[from] OomError),
}

/// Device-resident `f64` dense Cholesky factors (cuSOLVERDN `Dpotrf`,
/// lower triangle).
pub struct GpuDenseCholF64 {
    factors: CudaDenseCholF64,
}

impl GpuDenseCholF64 {
    /// Matrix dimension of the factorization.
    pub fn n(&self) -> usize {
        self.factors.n()
    }

    /// Number of `f64` workspace elements retained on the device.
    pub fn workspace_len(&self) -> usize {
        self.factors.workspace_len()
    }
}

/// Factor a square symmetric positive-definite row-major `f64` matrix on
/// `backend`, keeping the Cholesky factors on the device for repeated
/// solves.
///
/// Symmetry makes the row-major bytes identical to the column-major
/// layout cuSOLVER reads, so the upload needs no transpose (unlike
/// [`lu_factorize_gpu`]).
pub fn cholesky_factorize_gpu(
    backend: &CudaBackend,
    a: &DenseMatrix<f64>,
) -> Result<GpuDenseCholF64, GpuCholError> {
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(GpuCholError::DimensionMismatch(format!(
            "expected a square matrix, got {rows}x{cols}"
        )));
    }
    let n = rows;
    let row_major = a.as_slice();
    debug_assert_eq!(row_major.len(), n * n);
    let factors = backend.dense_cholesky_f64_from_row_major(n, row_major)?;
    Ok(GpuDenseCholF64 { factors })
}

/// Solve `A x = b` from resident GPU Cholesky factors, returning the
/// solution on the host.
pub fn cholesky_solve_gpu(
    pool: &MemoryPool<CudaBackend>,
    chol: &GpuDenseCholF64,
    b: &[f64],
) -> Result<Vec<f64>, GpuCholError> {
    let n = chol.n();
    if b.len() != n {
        return Err(GpuCholError::DimensionMismatch(format!(
            "expected a right-hand side of length {n}, got {}",
            b.len()
        )));
    }
    let backend = pool.backend();
    let mut rhs_dev = pool.try_alloc::<f64>(n)?;
    let mut out_dev = pool.try_alloc::<f64>(n)?;
    backend.copy_h2d(b, rhs_dev.raw_mut());
    backend.dense_cholesky_f64_solve(&chol.factors, rhs_dev.raw(), out_dev.raw_mut())?;
    let mut x = vec![0.0f64; n];
    backend.copy_d2h(out_dev.raw(), &mut x);
    Ok(x)
}

/// Solve `A X = B` for `nrhs` right-hand sides in a single device call
/// from resident GPU Cholesky factors.
///
/// Both `b` and the returned solution use row-major `n × nrhs` storage
/// (all right-hand sides for a matrix row are contiguous), matching
/// [`lu_solve_many_gpu`]. The host-side layout transpose is O(`n * nrhs`);
/// launch plus transfer overhead is paid once for all columns instead of
/// once per right-hand side.
pub fn cholesky_solve_many_gpu(
    pool: &MemoryPool<CudaBackend>,
    chol: &GpuDenseCholF64,
    b: &[f64],
    nrhs: usize,
) -> Result<Vec<f64>, GpuCholError> {
    let n = chol.n();
    let expected = n.checked_mul(nrhs).ok_or_else(|| {
        GpuCholError::DimensionMismatch(format!(
            "n={n} x nrhs={nrhs} right-hand-side size overflows usize"
        ))
    })?;
    if nrhs == 0 || b.len() != expected {
        return Err(GpuCholError::DimensionMismatch(format!(
            "expected a row-major {n}x{nrhs} right-hand side ({expected} values), got {}",
            b.len()
        )));
    }
    // Row-major host (i, j) -> column-major device (i, j): one streaming pass
    // each way, dwarfed by the O(n^2 * nrhs) triangular solves.
    let mut col_major = vec![0.0f64; expected];
    for i in 0..n {
        for j in 0..nrhs {
            col_major[i + j * n] = b[i * nrhs + j];
        }
    }
    let backend = pool.backend();
    let mut rhs_dev = pool.try_alloc::<f64>(expected)?;
    let mut out_dev = pool.try_alloc::<f64>(expected)?;
    backend.copy_h2d(&col_major, rhs_dev.raw_mut());
    backend.dense_cholesky_f64_solve_many(&chol.factors, rhs_dev.raw(), out_dev.raw_mut(), nrhs)?;
    backend.copy_d2h(out_dev.raw(), &mut col_major);
    let mut x = vec![0.0f64; expected];
    for i in 0..n {
        for j in 0..nrhs {
            x[i * nrhs + j] = col_major[i + j * n];
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::{CudaBackend, MemoryPoolConfig};

    fn try_test_pool() -> Option<MemoryPool<CudaBackend>> {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("CUDA unavailable, skipping GPU LU test: {error}");
                return None;
            }
        };
        match MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)) {
            Ok(pool) => Some(pool),
            Err(error) => {
                eprintln!("CUDA pool unavailable, skipping GPU LU test: {error}");
                None
            }
        }
    }

    #[test]
    fn gpu_lu_f64_matches_host_solve() {
        let Some(pool) = try_test_pool() else {
            return;
        };
        // Diagonally dominant 64x64 system with a non-trivial RHS.
        let n = 64;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if i + 1 == j || j + 1 == i {
                -1.0
            } else {
                0.0
            }
        });
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let lu = lu_factorize_gpu(pool.backend(), &a).expect("GPU factorization should succeed");
        assert_eq!(lu.n(), n);
        let x_gpu = lu_solve_gpu(&pool, &lu, &b, false).expect("GPU solve should succeed");
        let x_host = crate::lu_solve(&a, &b).expect("host solve should succeed");

        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let residual = |x: &[f64]| {
            let mut norm_sqr = 0.0;
            for i in 0..n {
                let mut ax = 0.0;
                for j in 0..n {
                    ax += a.as_slice()[i * n + j] * x[j];
                }
                let r = ax - b[i];
                norm_sqr += r * r;
            }
            norm_sqr.sqrt() / b_norm
        };
        assert!(
            residual(&x_gpu) < 1e-12,
            "GPU residual {} exceeds tolerance",
            residual(&x_gpu)
        );
        let diff = x_gpu
            .iter()
            .zip(x_host.iter())
            .map(|(g, h)| (g - h) * (g - h))
            .sum::<f64>()
            .sqrt()
            / x_host.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            diff < 1e-12,
            "GPU/host solution mismatch {diff:e} exceeds tolerance"
        );

        // Transposed system against the host transpose solve.
        let x_gpu_t = lu_solve_gpu(&pool, &lu, &b, true).expect("GPU transpose solve");
        let x_host_t = crate::lu_solve_transpose(&a, &b).expect("host transpose solve");
        let diff_t = x_gpu_t
            .iter()
            .zip(x_host_t.iter())
            .map(|(g, h)| (g - h) * (g - h))
            .sum::<f64>()
            .sqrt()
            / x_host_t.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            diff_t < 1e-12,
            "GPU/host transpose mismatch {diff_t:e} exceeds tolerance"
        );
    }

    #[test]
    fn gpu_chol_f64_matches_host_solve() {
        let Some(pool) = try_test_pool() else {
            return;
        };
        // Strictly diagonally dominant tridiagonal: symmetric positive-definite.
        let n = 64;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if i + 1 == j || j + 1 == i {
                -1.0
            } else {
                0.0
            }
        });
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let chol =
            cholesky_factorize_gpu(pool.backend(), &a).expect("GPU factorization should succeed");
        assert_eq!(chol.n(), n);
        let x_gpu = cholesky_solve_gpu(&pool, &chol, &b).expect("GPU solve should succeed");
        let host = crate::cholesky_factorize(&a).expect("host factorization should succeed");
        let x_host = host.solve(&b).expect("host solve should succeed");

        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut norm_sqr = 0.0;
        for i in 0..n {
            let mut ax = 0.0;
            for j in 0..n {
                ax += a.as_slice()[i * n + j] * x_gpu[j];
            }
            let r = ax - b[i];
            norm_sqr += r * r;
        }
        let residual = norm_sqr.sqrt() / b_norm;
        assert!(
            residual < 1e-12,
            "GPU residual {residual} exceeds tolerance"
        );
        let diff = x_gpu
            .iter()
            .zip(x_host.iter())
            .map(|(g, h)| (g - h) * (g - h))
            .sum::<f64>()
            .sqrt()
            / x_host.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            diff < 1e-12,
            "GPU/host solution mismatch {diff:e} exceeds tolerance"
        );
    }

    #[test]
    fn gpu_chol_f64_multi_rhs_matches_host() {
        let Some(pool) = try_test_pool() else {
            return;
        };
        let n = 64;
        let nrhs = 8;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if i + 1 == j || j + 1 == i {
                -1.0
            } else {
                0.0
            }
        });
        let b: Vec<f64> = (0..n * nrhs)
            .map(|k| ((k * 7919) % 101) as f64 / 17.0)
            .collect();

        let chol =
            cholesky_factorize_gpu(pool.backend(), &a).expect("GPU factorization should succeed");
        let x_gpu = cholesky_solve_many_gpu(&pool, &chol, &b, nrhs).expect("GPU multi-RHS solve");
        let host = crate::cholesky_factorize(&a).expect("host factorization should succeed");
        let mut x_host = vec![0.0; n * nrhs];
        host.solve_many_into(&b, &mut x_host, nrhs)
            .expect("host multi-RHS solve");

        let host_norm = x_host.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff = x_gpu
            .iter()
            .zip(x_host.iter())
            .map(|(g, h)| (g - h) * (g - h))
            .sum::<f64>()
            .sqrt()
            / host_norm;
        assert!(
            diff < 1e-12,
            "GPU/host multi-RHS mismatch {diff:e} exceeds tolerance"
        );

        // Rejections: wrong length and zero nrhs.
        assert!(cholesky_solve_many_gpu(&pool, &chol, &b[1..], nrhs).is_err());
        assert!(cholesky_solve_many_gpu(&pool, &chol, &[], 0).is_err());
    }

    #[test]
    fn gpu_lu_f64_multi_rhs_matches_host() {
        let Some(pool) = try_test_pool() else {
            return;
        };
        let n = 64;
        let nrhs = 8;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if i + 1 == j || j + 1 == i {
                -1.0
            } else {
                0.0
            }
        });
        // Row-major n x nrhs right-hand sides.
        let b: Vec<f64> = (0..n * nrhs)
            .map(|k| ((k * 7919) % 101) as f64 / 17.0)
            .collect();

        let lu = lu_factorize_gpu(pool.backend(), &a).expect("GPU factorization should succeed");
        let x_gpu = lu_solve_many_gpu(&pool, &lu, &b, nrhs, false).expect("GPU multi-RHS solve");
        let host_lu = crate::lu_factorize(&a).expect("host factorization should succeed");
        let x_host = host_lu.solve_many(&b, nrhs).expect("host multi-RHS solve");

        let host_norm = x_host.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff = x_gpu
            .iter()
            .zip(x_host.iter())
            .map(|(g, h)| (g - h) * (g - h))
            .sum::<f64>()
            .sqrt()
            / host_norm;
        assert!(
            diff < 1e-12,
            "GPU/host multi-RHS mismatch {diff:e} exceeds tolerance"
        );

        // Batched call must agree column-by-column with repeated single solves.
        for j in 0..nrhs {
            let col: Vec<f64> = (0..n).map(|i| b[i * nrhs + j]).collect();
            let x_single = lu_solve_gpu(&pool, &lu, &col, false).expect("single RHS solve");
            for i in 0..n {
                let g = x_gpu[i * nrhs + j];
                assert!(
                    (g - x_single[i]).abs() <= 1e-12 * (1.0 + x_single[i].abs()),
                    "batched/single mismatch at ({i},{j}): {g} vs {}",
                    x_single[i]
                );
            }
        }

        // Rejections: wrong length and zero nrhs.
        assert!(lu_solve_many_gpu(&pool, &lu, &b[1..], nrhs, false).is_err());
        assert!(lu_solve_many_gpu(&pool, &lu, &[], 0, false).is_err());
    }

    #[test]
    fn gpu_lu_f64_resident_rhs_matches_batched() {
        let Some(pool) = try_test_pool() else {
            return;
        };
        let n = 64;
        let nrhs = 4;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if i + 1 == j || j + 1 == i {
                -1.0
            } else {
                0.0
            }
        });
        let b: Vec<f64> = (0..n * nrhs)
            .map(|k| ((k * 104729) % 101) as f64 / 13.0)
            .collect();

        let lu = lu_factorize_gpu(pool.backend(), &a).expect("GPU factorization should succeed");
        let rhs = lu_rhs_to_gpu(&pool, &b, nrhs).expect("RHS upload should succeed");
        assert_eq!((rhs.n(), rhs.nrhs(), rhs.len()), (n, nrhs, n * nrhs));
        assert!(!rhs.is_empty());

        // Repeated resident solves agree with the one-shot batched call.
        for _ in 0..3 {
            let x_res = lu_solve_resident_gpu(&pool, &lu, &rhs, false).expect("resident solve");
            let x_many = lu_solve_many_gpu(&pool, &lu, &b, nrhs, false).expect("batched solve");
            let norm = x_many.iter().map(|v| v * v).sum::<f64>().sqrt();
            let diff = x_res
                .iter()
                .zip(x_many.iter())
                .map(|(r, m)| (r - m) * (r - m))
                .sum::<f64>()
                .sqrt()
                / norm;
            assert!(
                diff < 1e-14,
                "resident/batched mismatch {diff:e} exceeds tolerance"
            );
        }

        // Transposed resident solve matches the host transpose multi-RHS solve.
        let x_res_t = lu_solve_resident_gpu(&pool, &lu, &rhs, true).expect("resident transpose");
        let host_lu = crate::lu_factorize(&a).expect("host factorization should succeed");
        let mut x_host_t = vec![0.0; n * nrhs];
        for j in 0..nrhs {
            let col: Vec<f64> = (0..n).map(|i| b[i * nrhs + j]).collect();
            let mut xcol = vec![0.0; n];
            host_lu
                .solve_transpose_into(&col, &mut xcol)
                .expect("host transpose solve");
            for i in 0..n {
                x_host_t[i * nrhs + j] = xcol[i];
            }
        }
        let host_norm = x_host_t.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff_t = x_res_t
            .iter()
            .zip(x_host_t.iter())
            .map(|(g, h)| (g - h) * (g - h))
            .sum::<f64>()
            .sqrt()
            / host_norm;
        assert!(
            diff_t < 1e-12,
            "resident/host transpose mismatch {diff_t:e} exceeds tolerance"
        );

        // Rejections: bad uploads and n mismatch.
        assert!(lu_rhs_to_gpu(&pool, &b, 0).is_err());
        assert!(lu_rhs_to_gpu(&pool, &b[1..], nrhs).is_err());
        let other = DenseMatrix::from_fn(n + 1, n + 1, |i, j| if i == j { 2.0 } else { 0.0 });
        let lu_other = lu_factorize_gpu(pool.backend(), &other).expect("second factorization");
        assert!(lu_solve_resident_gpu(&pool, &lu_other, &rhs, false).is_err());
    }
}
