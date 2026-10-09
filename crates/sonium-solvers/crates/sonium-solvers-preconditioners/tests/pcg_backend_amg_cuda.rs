//! CUDA bridge acceptance: device-resident PCG preconditioned by an uploaded
//! AMG hierarchy agrees with host PCG on the same hierarchy.
//!
//! The device AMG path substitutes the CUDA-safe Jacobi smoother wherever
//! the host hierarchy requests a Gauss-Seidel variant (recorded in the
//! runtime diagnostics), so iteration counts are not expected to match —
//! only convergence and the final solution are. Skip-capable on hosts
//! without an NVIDIA driver; `SONIUM_REQUIRE_CUDA=1` turns the skip into a
//! failure.

#![cfg(feature = "cuda")]

use sonium_compute::operator::DeviceCsr;
use sonium_compute::{
    Backend, CudaBackend, MemoryPool, MemoryPoolConfig, PcgBackendConfig, PcgBackendWorkspace,
    pcg_backend_preconditioned,
};
use sonium_solvers_iterative::{CgConfig, pcg};
use sonium_solvers_preconditioners::{AmgConfig, AmgPreconditioner, BackendAmgPreconditioner};
use sonium_solvers_sparse::CsrMatrix;

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping PCG+AMG bridge acceptance: {error}");
            None
        }
    }
}

fn poisson_3d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n * n;
    let mut triplets = Vec::with_capacity(7 * nn);
    let at = |i: usize, j: usize, k: usize| i * n * n + j * n + k;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = at(i, j, k);
                triplets.push((row, row, 6.0));
                if i > 0 {
                    triplets.push((row, at(i - 1, j, k), -1.0));
                }
                if j > 0 {
                    triplets.push((row, at(i, j - 1, k), -1.0));
                }
                if k > 0 {
                    triplets.push((row, at(i, j, k - 1), -1.0));
                }
                if k + 1 < n {
                    triplets.push((row, at(i, j, k + 1), -1.0));
                }
                if j + 1 < n {
                    triplets.push((row, at(i, j + 1, k), -1.0));
                }
                if i + 1 < n {
                    triplets.push((row, at(i + 1, j, k), -1.0));
                }
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

#[test]
fn cuda_pcg_amg_matches_host_solution() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 16usize;
    let ndofs = n * n * n;
    let matrix = poisson_3d(n);
    let b = vec![1.0; ndofs];

    // Host reference: PCG + mirror AMG hierarchy.
    let host_amg =
        AmgPreconditioner::from_csr(&matrix, AmgConfig::for_hypre_boomeramg_comparison());
    let host_sol = pcg(
        &matrix,
        &host_amg,
        &b,
        &CgConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            print_interval: 0,
        },
    );
    assert!(host_sol.converged);

    // Device solve: same hierarchy uploaded, PCG fully device-resident.
    let pool = MemoryPool::new(cuda, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let backend = pool.backend();
    let rp: Vec<i32> = matrix.row_ptrs.iter().map(|&v| v as i32).collect();
    let ci: Vec<i32> = matrix.col_indices.iter().map(|&v| v as i32).collect();
    let a = DeviceCsr::try_from_host(backend, ndofs, ndofs, &rp, &ci, &matrix.values).unwrap();
    let dev_amg = BackendAmgPreconditioner::from_host_amg(backend, &host_amg);
    let mut b_dev = pool.try_alloc::<f64>(ndofs).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(ndofs).unwrap();
    backend.copy_h2d(&b, b_dev.raw_mut());
    let mut ws = PcgBackendWorkspace::<CudaBackend, f64>::new(ndofs);
    let status = pcg_backend_preconditioned(
        backend,
        &pool,
        &mut ws,
        &a,
        &dev_amg,
        b_dev.raw(),
        None,
        x_dev.raw_mut(),
        &PcgBackendConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
        },
    )
    .unwrap();
    backend.try_synchronize().unwrap();
    assert!(status.converged, "device PCG+AMG did not converge");

    let mut x_host = vec![0.0; ndofs];
    backend.try_copy_d2h(x_dev.raw(), &mut x_host).unwrap();
    let max_diff = x_host
        .iter()
        .zip(&host_sol.x)
        .map(|(d, h)| (d - h).abs())
        .fold(0.0, f64::max);
    let max_ref = host_sol.x.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(
        max_diff / max_ref < 1e-6,
        "device/host solution mismatch {max_diff:.2e} (device iters {}, host iters {})",
        status.iterations,
        host_sol.iterations
    );
}
