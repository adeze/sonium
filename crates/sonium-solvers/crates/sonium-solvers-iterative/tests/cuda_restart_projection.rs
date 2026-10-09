//! CUDA regression: restart cycles whose length does not divide
//! `max_iterations` must not fail transfers.
//!
//! A short final cycle (e.g. restart=30, max_iterations=100 ends with a
//! 10-iteration cycle) previously shrank host staging below the retained
//! device projection buffers, failing exact-length D2H/H2D transfers with
//! `host length 10 != device length 30`. Skip-capable on hosts without an
//! NVIDIA driver; `SONIUM_REQUIRE_CUDA=1` turns the skip into a failure.

#![cfg(feature = "cuda")]

use sonium_compute::{
    Backend, BackendCsr, CudaBackend, IdentityDevicePreconditioner, MemoryPool, MemoryPoolConfig,
};
use sonium_solvers_iterative::{BackendKrylovConfig, BackendKrylovWorkspace, try_gmres_backend};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping restart-size regression: {error}");
            None
        }
    }
}

#[test]
fn cuda_gmres_short_final_restart_cycle_transfers() {
    let Some(backend) = cuda_or_skip() else {
        return;
    };
    // 2D Poisson with a checkerboard right-hand side: unpreconditioned
    // GMRES(30) cannot reach 1e-15 in 100 iterations, so with the
    // unattainable tolerance every cycle runs, including the short final
    // one (30, 30, 30, 10).
    let grid = 32;
    let n = grid * grid;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..grid {
        for j in 0..grid {
            row_ptr.push(col_idx.len() as i32);
            let row = i * grid + j;
            let mut cols = vec![row];
            if i > 0 {
                cols.push((i - 1) * grid + j);
            }
            if i + 1 < grid {
                cols.push((i + 1) * grid + j);
            }
            if j > 0 {
                cols.push(i * grid + j - 1);
            }
            if j + 1 < grid {
                cols.push(i * grid + j + 1);
            }
            cols.sort_unstable();
            for col in cols {
                col_idx.push(col as i32);
                values.push(if col == row { 4.0 } else { -1.0 });
            }
        }
    }
    row_ptr.push(col_idx.len() as i32);

    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).expect("pool");
    let backend = pool.backend();
    let matrix =
        BackendCsr::try_from_host(backend, n, n, &row_ptr, &col_idx, &values).expect("CSR upload");
    let mut rhs = pool.try_alloc::<f64>(n).expect("rhs alloc");
    let mut out = pool.try_alloc::<f64>(n).expect("out alloc");
    let checker: Vec<f64> = (0..n)
        .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    backend.try_copy_h2d(&checker, rhs.raw_mut()).expect("h2d");
    backend
        .try_copy_h2d(&vec![0.0; n], out.raw_mut())
        .expect("h2d zero");

    let mut workspace = BackendKrylovWorkspace::new(n);
    let cfg = BackendKrylovConfig {
        max_iterations: 100,
        restart: 30,
        tolerance: 1e-15,
        require_native: true,
        exact_residual_check_interval: 1,
    };
    // Must not error with an InvalidArgument transfer-length mismatch on
    // the 10-iteration final cycle; convergence is not required. Pinning
    // the full iteration count proves the short cycle actually executed
    // instead of the solve converging early.
    let status = try_gmres_backend(
        backend,
        &pool,
        &mut workspace,
        &matrix,
        &IdentityDevicePreconditioner,
        rhs.raw(),
        None,
        out.raw_mut(),
        &cfg,
        None,
    )
    .expect("short final restart cycle must transfer cleanly");
    assert_eq!(status.backend, "cuda");
    assert_eq!(status.iterations, 100);
}
