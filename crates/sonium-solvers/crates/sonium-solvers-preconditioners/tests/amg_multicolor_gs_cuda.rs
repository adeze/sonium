//! CUDA acceptance for device multicolor Gauss-Seidel smoothing inside AMG:
//! a hierarchy built with `AmgSmoother::MulticolorGaussSeidel` uploads its
//! colorings, takes the native kernel path (no smoother fallback is
//! recorded), and device PCG over it matches host PCG on the same hierarchy.
//!
//! Skip-capable on hosts without an NVIDIA driver; `SONIUM_REQUIRE_CUDA=1`
//! turns the skip into a failure.

#![cfg(feature = "cuda")]

use sonium_compute::operator::DeviceCsr;
use sonium_compute::{
    Backend, CudaBackend, MemoryPool, MemoryPoolConfig, PcgBackendConfig, PcgBackendWorkspace,
    pcg_backend_preconditioned,
};
use sonium_solvers_iterative::{CgConfig, pcg};
use sonium_solvers_preconditioners::{
    AmgConfig, AmgPreconditioner, AmgSmoother, BackendAmgPreconditioner,
};
use sonium_solvers_sparse::CsrMatrix;

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping multicolor-GS bridge acceptance: {error}");
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
fn cuda_multicolor_gs_needs_no_smoother_fallback() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 16usize;
    let ndofs = n * n * n;
    let matrix = poisson_3d(n);
    let b = vec![1.0; ndofs];

    let mut cfg = AmgConfig::for_hypre_boomeramg_comparison();
    cfg.smoother = AmgSmoother::MulticolorGaussSeidel;
    let host_amg = AmgPreconditioner::from_csr(&matrix, cfg);
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
    assert!(
        status.converged,
        "device PCG+multicolor-AMG did not converge"
    );

    // The multicolor path must not silently substitute Jacobi smoothing.
    let diag = dev_amg.diagnostics();
    assert_eq!(
        diag.runtime.fallback_count, 0,
        "multicolor GS fell back to Jacobi"
    );

    let mut x_host = vec![0.0; ndofs];
    backend.try_copy_d2h(x_dev.raw(), &mut x_host).unwrap();
    let max_diff = x_host
        .iter()
        .zip(&host_sol.x)
        .map(|(d, h)| (d - h).abs())
        .fold(0.0, f64::max);
    let max_ref = host_sol.x.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(
        max_diff / max_ref < 1e-8,
        "device/host solution mismatch {max_diff:.2e} (device iters {}, host iters {})",
        status.iterations,
        host_sol.iterations
    );
}
