//! CUDA acceptance for backend-resident PCG: the same Jacobi-preconditioned
//! Poisson solve as `pcg_backend.rs`, run on CUDA, agreeing with the CPU
//! solve and satisfying an independently recomputed device residual.
//!
//! Skip-capable on hosts without an NVIDIA driver; CI jobs with
//! `SONIUM_REQUIRE_CUDA=1` turn that skip into a failure.

#![cfg(feature = "cuda")]

use sonium_compute::operator::DeviceCsr;
use sonium_compute::{
    Backend, CpuBackend, CudaBackend, MemoryPool, MemoryPoolConfig, PcgBackendConfig,
    PcgBackendWorkspace, PointJacobiBackend, ReductionMode, pcg_backend_preconditioned,
};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping PCG acceptance: {error}");
            None
        }
    }
}

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

fn device_residual<B, A>(
    backend: &B,
    pool: &MemoryPool<B>,
    a: &A,
    b: &[f64],
    x: &B::Buffer<f64>,
) -> f64
where
    B: Backend,
    A: sonium_compute::DeviceOperator<B, f64> + ?Sized,
{
    let mut bb = pool.try_alloc::<f64>(b.len()).unwrap();
    let mut ax = pool.try_alloc::<f64>(b.len()).unwrap();
    let mut r = pool.try_alloc::<f64>(b.len()).unwrap();
    backend.copy_h2d(b, bb.raw_mut());
    a.try_apply(backend, x, ax.raw_mut()).unwrap();
    backend.try_copy_d2d(bb.raw(), r.raw_mut()).unwrap();
    backend.try_axpy(-1.0, ax.raw(), r.raw_mut()).unwrap();
    let num = backend
        .try_nrm2_with_mode::<f64>(ReductionMode::Fast, r.raw())
        .unwrap();
    let den = backend
        .try_nrm2_with_mode::<f64>(ReductionMode::Fast, bb.raw())
        .unwrap();
    num / den
}

#[test]
fn cuda_pcg_jacobi_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 8usize;
    let ndofs = n * n * n;
    let (rp, ci, vals) = poisson_3d(n);
    let rhs = vec![1.0; ndofs];

    let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cuda_pool = MemoryPool::new(cuda, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cpu = cpu_pool.backend();
    let cuda = cuda_pool.backend();

    let a_cpu = DeviceCsr::try_from_host(cpu, ndofs, ndofs, &rp, &ci, &vals).unwrap();
    let a_cuda = DeviceCsr::try_from_host(cuda, ndofs, ndofs, &rp, &ci, &vals).unwrap();
    let pre_cpu = PointJacobiBackend::from_csr_host(cpu, ndofs, &rp, &ci, &vals);
    let pre_cuda = PointJacobiBackend::from_csr_host(cuda, ndofs, &rp, &ci, &vals);

    let config = PcgBackendConfig {
        max_iterations: 1000,
        tolerance: 1e-10,
    };
    let mut ws_cpu = PcgBackendWorkspace::<CpuBackend, f64>::new(ndofs);
    let mut ws_cuda = PcgBackendWorkspace::<CudaBackend, f64>::new(ndofs);
    let mut b_cpu = cpu_pool.try_alloc::<f64>(ndofs).unwrap();
    let mut b_cuda = cuda_pool.try_alloc::<f64>(ndofs).unwrap();
    let mut x_cpu = cpu_pool.try_alloc::<f64>(ndofs).unwrap();
    let mut x_cuda = cuda_pool.try_alloc::<f64>(ndofs).unwrap();
    cpu.copy_h2d(&rhs, b_cpu.raw_mut());
    cuda.copy_h2d(&rhs, b_cuda.raw_mut());

    let cpu_status = pcg_backend_preconditioned(
        cpu,
        &cpu_pool,
        &mut ws_cpu,
        &a_cpu,
        &pre_cpu,
        b_cpu.raw(),
        None,
        x_cpu.raw_mut(),
        &config,
    )
    .unwrap();
    let cuda_status = pcg_backend_preconditioned(
        cuda,
        &cuda_pool,
        &mut ws_cuda,
        &a_cuda,
        &pre_cuda,
        b_cuda.raw(),
        None,
        x_cuda.raw_mut(),
        &config,
    )
    .unwrap();
    cuda.try_synchronize().unwrap();
    assert!(cpu_status.converged && cuda_status.converged);

    let cuda_res = device_residual(cuda, &cuda_pool, &a_cuda, &rhs, x_cuda.raw());
    assert!(cuda_res < 1e-9, "CUDA device residual {cuda_res:.2e}");

    let mut x_cpu_host = vec![0.0; ndofs];
    let mut x_cuda_host = vec![0.0; ndofs];
    cpu.try_copy_d2h(x_cpu.raw(), &mut x_cpu_host).unwrap();
    cuda.try_copy_d2h(x_cuda.raw(), &mut x_cuda_host).unwrap();
    let max_diff = x_cpu_host
        .iter()
        .zip(&x_cuda_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let max_ref = x_cpu_host.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(
        max_diff / max_ref < 1e-10,
        "CPU/CUDA solution mismatch {max_diff:.2e}"
    );
}
