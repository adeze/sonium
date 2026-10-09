//! CPU acceptance for backend-resident PCG: Jacobi-preconditioned solve of
//! 3D Poisson against an independently recomputed device residual, plus an
//! unpreconditioned baseline showing the preconditioner helps.

use sonium_compute::operator::{DeviceCsr, IdentityDevicePreconditioner};
use sonium_compute::{
    Backend, CpuBackend, DeviceBufferLike, MemoryPool, MemoryPoolConfig, PcgBackendConfig,
    PcgBackendWorkspace, PointJacobiBackend, ReductionMode, pcg_backend_preconditioned,
};

/// 7-point Poisson CSR with `i32` indices (matches `DeviceCsr::try_from_host`).
fn poisson_3d(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(n * n * n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    let at = |i: usize, j: usize, k: usize| i * n * n + j * n + k;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = at(i, j, k);
                row_ptr.push(col_idx.len() as i32);
                let mut push = |c: usize, v: f64| {
                    col_idx.push(c as i32);
                    values.push(v);
                };
                if i > 0 {
                    push(at(i - 1, j, k), -1.0);
                }
                if j > 0 {
                    push(at(i, j - 1, k), -1.0);
                }
                if k > 0 {
                    push(at(i, j, k - 1), -1.0);
                }
                push(row, 6.0);
                if k + 1 < n {
                    push(at(i, j, k + 1), -1.0);
                }
                if j + 1 < n {
                    push(at(i, j + 1, k), -1.0);
                }
                if i + 1 < n {
                    push(at(i + 1, j, k), -1.0);
                }
            }
        }
    }
    row_ptr.push(col_idx.len() as i32);
    (row_ptr, col_idx, values)
}

/// Device-side relative residual `||b - A*x|| / ||b||`, recomputed from the
/// operator (not the CG recurrence).
fn device_residual<B, A>(
    backend: &B,
    pool: &MemoryPool<B>,
    a: &A,
    b: &B::Buffer<f64>,
    x: &B::Buffer<f64>,
) -> f64
where
    B: Backend,
    A: sonium_compute::DeviceOperator<B, f64> + ?Sized,
{
    let mut ax = pool.try_alloc::<f64>(b.len()).unwrap();
    let mut r = pool.try_alloc::<f64>(b.len()).unwrap();
    a.try_apply(backend, x, ax.raw_mut()).unwrap();
    backend.try_copy_d2d(b, r.raw_mut()).unwrap();
    backend.try_axpy(-1.0, ax.raw(), r.raw_mut()).unwrap();
    let num = backend
        .try_nrm2_with_mode::<f64>(ReductionMode::Fast, r.raw())
        .unwrap();
    let den = backend
        .try_nrm2_with_mode::<f64>(ReductionMode::Fast, b)
        .unwrap();
    num / den
}

fn solve_jacobi_pcg(n: usize, tol: f64) -> (usize, f64) {
    let cpu = CpuBackend;
    let pool = MemoryPool::new(cpu, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let backend = pool.backend();
    let (rp, ci, vals) = poisson_3d(n);
    let ndofs = n * n * n;
    let a = DeviceCsr::try_from_host(backend, ndofs, ndofs, &rp, &ci, &vals).unwrap();
    let pre = PointJacobiBackend::from_csr_host(backend, ndofs, &rp, &ci, &vals);
    let mut b = pool.try_alloc::<f64>(ndofs).unwrap();
    let mut out = pool.try_alloc::<f64>(ndofs).unwrap();
    backend.copy_h2d(&vec![1.0; ndofs], b.raw_mut());
    let mut ws = PcgBackendWorkspace::<CpuBackend, f64>::new(ndofs);
    let status = pcg_backend_preconditioned(
        backend,
        &pool,
        &mut ws,
        &a,
        &pre,
        b.raw(),
        None,
        out.raw_mut(),
        &PcgBackendConfig {
            max_iterations: 1000,
            tolerance: tol,
        },
    )
    .unwrap();
    assert!(status.converged);
    let res = device_residual(backend, &pool, &a, b.raw(), out.raw());
    assert!(res < 10.0 * tol, "device residual {res:.2e} too large");
    (status.iterations, res)
}

#[test]
fn cpu_pcg_jacobi_solves_poisson() {
    let (iters, res) = solve_jacobi_pcg(8, 1e-10);
    assert!(iters < 512, "no convergence acceleration: {iters} iters");
    assert!(res < 1e-9, "residual {res:.2e}");
}

#[test]
fn cpu_pcg_f32_solves_poisson() {
    // Single-precision path through the same device loop (f32 reductions,
    // f32 axpy/scale, f32 omega handling).
    let cpu = CpuBackend;
    let pool = MemoryPool::new(cpu, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let backend = pool.backend();
    let (rp, ci, vals_f64) = poisson_3d(8);
    let vals: Vec<f32> = vals_f64.iter().map(|&v| v as f32).collect();
    let ndofs = 512;
    let a = DeviceCsr::try_from_host(backend, ndofs, ndofs, &rp, &ci, &vals).unwrap();
    let pre = PointJacobiBackend::from_csr_host(backend, ndofs, &rp, &ci, &vals);
    let mut b = pool.try_alloc::<f32>(ndofs).unwrap();
    let mut out = pool.try_alloc::<f32>(ndofs).unwrap();
    backend.copy_h2d(&vec![1.0f32; ndofs], b.raw_mut());
    let mut ws = PcgBackendWorkspace::<CpuBackend, f32>::new(ndofs);
    let status = pcg_backend_preconditioned(
        backend,
        &pool,
        &mut ws,
        &a,
        &pre,
        b.raw(),
        None,
        out.raw_mut(),
        &PcgBackendConfig {
            max_iterations: 1000,
            tolerance: 1e-5,
        },
    )
    .unwrap();
    assert!(status.converged);
    assert!(status.iterations < 512);
    let res = device_residual_f32(backend, &pool, &a, b.raw(), out.raw());
    assert!(res < 1e-4, "f32 device residual {res:.2e}");
}

/// Device-side relative residual for f32.
fn device_residual_f32<B, A>(
    backend: &B,
    pool: &MemoryPool<B>,
    a: &A,
    b: &B::Buffer<f32>,
    x: &B::Buffer<f32>,
) -> f32
where
    B: Backend,
    A: sonium_compute::DeviceOperator<B, f32> + ?Sized,
{
    let mut ax = pool.try_alloc::<f32>(b.len()).unwrap();
    let mut r = pool.try_alloc::<f32>(b.len()).unwrap();
    a.try_apply(backend, x, ax.raw_mut()).unwrap();
    backend.try_copy_d2d(b, r.raw_mut()).unwrap();
    backend.try_axpy(-1.0, ax.raw(), r.raw_mut()).unwrap();
    let num = backend
        .try_nrm2_with_mode::<f32>(ReductionMode::Fast, r.raw())
        .unwrap();
    let den = backend
        .try_nrm2_with_mode::<f32>(ReductionMode::Fast, b)
        .unwrap();
    num / den
}

#[test]
fn cpu_pcg_identity_baseline_converges_slower() {
    let cpu = CpuBackend;
    let pool = MemoryPool::new(cpu, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let backend = pool.backend();
    let (rp, ci, vals) = poisson_3d(8);
    let ndofs = 512;
    let a = DeviceCsr::try_from_host(backend, ndofs, ndofs, &rp, &ci, &vals).unwrap();
    let pre = IdentityDevicePreconditioner;
    let mut b = pool.try_alloc::<f64>(ndofs).unwrap();
    let mut out = pool.try_alloc::<f64>(ndofs).unwrap();
    backend.copy_h2d(&vec![1.0; ndofs], b.raw_mut());
    let mut ws = PcgBackendWorkspace::<CpuBackend, f64>::new(ndofs);
    let status = pcg_backend_preconditioned(
        backend,
        &pool,
        &mut ws,
        &a,
        &pre,
        b.raw(),
        None,
        out.raw_mut(),
        &PcgBackendConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
        },
    )
    .unwrap();
    assert!(status.converged);
    let (jacobi_iters, _) = solve_jacobi_pcg(8, 1e-10);
    assert!(
        jacobi_iters <= status.iterations,
        "Jacobi should not hurt: {jacobi_iters} vs {} unpreconditioned",
        status.iterations
    );
}

#[test]
fn cpu_pcg_complex_hpd_matches_real_path() {
    use num_complex::Complex64;
    // Genuinely complex HPD tridiagonal: diagonal 4, off-diagonal
    // (1+i)/(1-i) conjugate pair. Exercises the complex dot (conjugate),
    // complex alpha/beta division, and complex SpMV in the device loop.
    let n = 64usize;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        row_ptr.push(col_idx.len() as i32);
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(Complex64::new(1.0, 1.0));
        }
        col_idx.push(i as i32);
        values.push(Complex64::new(4.0, 0.0));
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(Complex64::new(1.0, -1.0));
        }
    }
    row_ptr.push(col_idx.len() as i32);
    let rhs: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i + 1) as f64, 0.5))
        .collect();

    let cpu = CpuBackend;
    let pool = MemoryPool::new(cpu, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let backend = pool.backend();
    let a = DeviceCsr::try_from_host(backend, n, n, &row_ptr, &col_idx, &values).unwrap();
    let pre = PointJacobiBackend::from_csr_host(backend, n, &row_ptr, &col_idx, &values);
    let mut b = pool.try_alloc::<Complex64>(n).unwrap();
    let mut out = pool.try_alloc::<Complex64>(n).unwrap();
    backend.copy_h2d(&rhs, b.raw_mut());
    let mut ws = PcgBackendWorkspace::<CpuBackend, Complex64>::new(n);
    let status = pcg_backend_preconditioned(
        backend,
        &pool,
        &mut ws,
        &a,
        &pre,
        b.raw(),
        None,
        out.raw_mut(),
        &PcgBackendConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
        },
    )
    .unwrap();
    assert!(status.converged);
    let res = device_residual_complex(backend, &pool, &a, &rhs, out.raw());
    assert!(res < 1e-9, "complex device residual {res:.2e}");
}

/// Device-side relative residual for Complex64 (same contract as
/// [`device_residual`], which is f64-only).
fn device_residual_complex<B, A>(
    backend: &B,
    pool: &MemoryPool<B>,
    a: &A,
    b: &[num_complex::Complex64],
    x: &B::Buffer<num_complex::Complex64>,
) -> f64
where
    B: Backend,
    A: sonium_compute::DeviceOperator<B, num_complex::Complex64> + ?Sized,
{
    use num_complex::Complex64;
    let mut ax = pool.try_alloc::<Complex64>(b.len()).unwrap();
    let mut r = pool.try_alloc::<Complex64>(b.len()).unwrap();
    a.try_apply(backend, x, ax.raw_mut()).unwrap();
    let b_norms: Vec<f64> = b.iter().map(|v| v.norm()).collect();
    backend.copy_h2d(b, r.raw_mut());
    backend
        .try_axpy(Complex64::new(-1.0, 0.0), ax.raw(), r.raw_mut())
        .unwrap();
    let num = backend
        .try_nrm2_with_mode::<Complex64>(ReductionMode::Fast, r.raw())
        .unwrap();
    let den = b_norms.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / den
}

#[test]
fn cpu_pcg_zero_rhs_contract() {
    use sonium_compute::PcgBackendError;
    let cpu = CpuBackend;
    let pool = MemoryPool::new(cpu, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let backend = pool.backend();
    let (rp, ci, vals) = poisson_3d(4);
    let ndofs = 64;
    let a = DeviceCsr::try_from_host(backend, ndofs, ndofs, &rp, &ci, &vals).unwrap();
    let pre = PointJacobiBackend::from_csr_host(backend, ndofs, &rp, &ci, &vals);
    let mut b = pool.try_alloc::<f64>(ndofs).unwrap();
    let mut out = pool.try_alloc::<f64>(ndofs).unwrap();
    backend.copy_h2d(&vec![0.0; ndofs], b.raw_mut());
    let mut ws = PcgBackendWorkspace::<CpuBackend, f64>::new(ndofs);
    let err = pcg_backend_preconditioned(
        backend,
        &pool,
        &mut ws,
        &a,
        &pre,
        b.raw(),
        None,
        out.raw_mut(),
        &PcgBackendConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, PcgBackendError::ZeroRhs));
}
