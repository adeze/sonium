//! Required-device no-fallback gate: `require_native: true` must run fully
//! native on CUDA (never silently host-staged) and must refuse cells that
//! would fall back, instead of converging through the fallback path.
//!
//! The positive test asserts the solve ran on the device (`status.backend ==
//! "cuda"` — the backend refuses to report host-staged work as a CUDA
//! solve). The negative test is self-adapting: it finds a genuinely
//! fallback cell at runtime and asserts refusal, skipping when every cell is
//! native. Skip-capable on hosts without an NVIDIA driver.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use num_traits::FromPrimitive;
use sonium_compute::{
    Backend, BackendCsr, CudaBackend, DeviceScalar, IdentityDevicePreconditioner, MemoryPool,
    MemoryPoolConfig,
};
use sonium_solvers_iterative::{
    BackendKrylovConfig, BackendKrylovError, BackendKrylovWorkspace, try_gmres_backend,
};

fn cuda_or_skip(label: &str) -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping {label}: {error}");
            None
        }
    }
}

fn poisson_5pt(grid: usize) -> (usize, Vec<i32>, Vec<i32>, Vec<f64>) {
    let n = grid * grid;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..grid {
        for j in 0..grid {
            row_ptr.push(col_idx.len() as i32);
            let row = (i * grid + j) as i32;
            let mut cols = vec![row];
            if i > 0 {
                cols.push(row - grid as i32);
            }
            if i + 1 < grid {
                cols.push(row + grid as i32);
            }
            if j > 0 {
                cols.push(row - 1);
            }
            if j + 1 < grid {
                cols.push(row + 1);
            }
            cols.sort_unstable();
            for col in cols {
                col_idx.push(col);
                values.push(if col == row { 4.0 } else { -1.0 });
            }
        }
    }
    row_ptr.push(col_idx.len() as i32);
    (n, row_ptr, col_idx, values)
}

#[test]
fn cuda_gmres_require_native_runs_without_fallback() {
    let Some(backend) = cuda_or_skip("native-execution gate") else {
        return;
    };
    let (n, row_ptr, col_idx, values) = poisson_5pt(16);
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let backend = pool.backend();
    let matrix = BackendCsr::try_from_host(backend, n, n, &row_ptr, &col_idx, &values).unwrap();
    let mut rhs = pool.try_alloc::<f64>(n).unwrap();
    let mut out = pool.try_alloc::<f64>(n).unwrap();
    let checker: Vec<f64> = (0..n)
        .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    backend.try_copy_h2d(&checker, rhs.raw_mut()).unwrap();
    backend.try_copy_h2d(&vec![0.0; n], out.raw_mut()).unwrap();

    let mut workspace = BackendKrylovWorkspace::new(n);
    let cfg = BackendKrylovConfig {
        max_iterations: 100,
        restart: 30,
        tolerance: 1.0e-8,
        require_native: true,
        exact_residual_check_interval: 0,
    };
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
    .expect("native GMRES must succeed");
    assert!(status.converged, "native GMRES must converge");
    // The backend refuses to report host-staged work as a CUDA solve, so
    // this equality is the no-fallback assertion.
    assert_eq!(status.backend, "cuda");
}

/// True when scalar `T` has a GMRES execution cell that is implemented but
/// would use host fallback (the exact condition `require_native` refuses).
fn gmres_cell_uses_host_fallback<T: DeviceScalar>(backend: &CudaBackend) -> bool {
    let caps = backend.capabilities();
    let states = [
        caps.blas1.state_for::<T>(),
        caps.csr_spmv.state_for::<T>(),
        caps.axpy_many.state_for::<T>(),
        caps.multi_dot.state_for::<T>(),
        caps.reduction.fast.state_for::<T>(),
    ];
    states.iter().all(|state| state.is_available())
        && states.iter().any(|state| state.uses_host_fallback())
}

fn refused_for_scalar<T: DeviceScalar>(backend: CudaBackend, tolerance: T::Real, label: &str) {
    let (n, row_ptr, col_idx, values_f64) = poisson_5pt(8);
    let real =
        |v: f64| T::Real::from_f64(v).expect("small test constants convert to every real type");
    let values: Vec<T> = values_f64.iter().map(|&v| T::from_real(real(v))).collect();
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let backend = pool.backend();
    let matrix = BackendCsr::try_from_host(backend, n, n, &row_ptr, &col_idx, &values).unwrap();
    let mut rhs = pool.try_alloc::<T>(n).unwrap();
    let mut out = pool.try_alloc::<T>(n).unwrap();
    let checker: Vec<T> = (0..n)
        .map(|k| if k % 2 == 0 { T::one() } else { -T::one() })
        .collect();
    backend.try_copy_h2d(&checker, rhs.raw_mut()).unwrap();
    backend
        .try_copy_h2d(&vec![T::zero(); n], out.raw_mut())
        .unwrap();

    let mut workspace = BackendKrylovWorkspace::new(n);
    let cfg = BackendKrylovConfig {
        max_iterations: 100,
        restart: 30,
        tolerance,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let error = try_gmres_backend(
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
    .expect_err(&format!(
        "{label}: fallback cell must be refused under require_native"
    ));
    let BackendKrylovError::Capability { reason, .. } = error else {
        panic!("{label}: expected a Capability refusal, got: {error}");
    };
    assert!(
        reason.contains("host fallback"),
        "{label}: refusal must name the fallback, got: {reason}"
    );
}

#[test]
fn cuda_require_native_refuses_fallback_cell() {
    let Some(backend) = cuda_or_skip("fallback-refusal gate") else {
        return;
    };
    // Probe every scalar family; refuse through the first genuinely fallback
    // GMRES cell. Skips only when the backend is native everywhere.
    if gmres_cell_uses_host_fallback::<Complex64>(&backend) {
        refused_for_scalar::<Complex64>(backend, 1.0e-8, "complex64");
    } else if gmres_cell_uses_host_fallback::<Complex32>(&backend) {
        refused_for_scalar::<Complex32>(backend, 1.0e-5, "complex32");
    } else if gmres_cell_uses_host_fallback::<f32>(&backend) {
        refused_for_scalar::<f32>(backend, 1.0e-5, "f32");
    } else if gmres_cell_uses_host_fallback::<f64>(&backend) {
        refused_for_scalar::<f64>(backend, 1.0e-8, "f64");
    } else {
        eprintln!("backend native for every scalar; fallback refusal vacuously holds");
    }
}

#[test]
fn cuda_host_staged_opt_in_still_solves() {
    // require_native: false is the explicit opt-in for the compatibility
    // path; it must keep working while the default stays strict.
    let Some(backend) = cuda_or_skip("staged opt-in gate") else {
        return;
    };
    let (n, row_ptr, col_idx, values) = poisson_5pt(16);
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let backend = pool.backend();
    let matrix = BackendCsr::try_from_host(backend, n, n, &row_ptr, &col_idx, &values).unwrap();
    let mut rhs = pool.try_alloc::<f64>(n).unwrap();
    let mut out = pool.try_alloc::<f64>(n).unwrap();
    let checker: Vec<f64> = (0..n)
        .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    backend.try_copy_h2d(&checker, rhs.raw_mut()).unwrap();
    backend.try_copy_h2d(&vec![0.0; n], out.raw_mut()).unwrap();

    let mut workspace = BackendKrylovWorkspace::new(n);
    let cfg = BackendKrylovConfig {
        max_iterations: 100,
        restart: 30,
        tolerance: 1.0e-8,
        require_native: false,
        exact_residual_check_interval: 0,
    };
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
    .expect("staged opt-in must solve");
    assert!(status.converged);
}
