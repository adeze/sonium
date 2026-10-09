use sonium_compute::{
    Backend, CpuBackend, DeviceCsr, IdentityDevicePreconditioner, MemoryPool, MemoryPoolConfig,
};
use sonium_solvers::{
    BackendKrylovConfig, BackendKrylovWorkspace, BackendStoppingReason,
    backend_krylov_capability_matrix, try_block_gmres_backend, try_gmres_backend,
};

// Fixture intentionally uses the panicking convenience constructor on a tiny
// fixed matrix; production paths must use the `try_*` variants.
#[allow(deprecated)]
fn operator(backend: &CpuBackend) -> DeviceCsr<CpuBackend, f64> {
    DeviceCsr::from_host(
        backend,
        2,
        2,
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[4.0, 1.0, 1.0, 3.0],
    )
}

#[test]
fn backend_capabilities_and_scalar_solve_are_public() {
    let backend = CpuBackend;
    let cells = backend_krylov_capability_matrix(&backend);
    assert_eq!(cells.len(), 16);
    assert!(cells.iter().all(|cell| cell.native_execution));

    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let mut b = pool.backend().raw_alloc::<f64>(2).unwrap();
    pool.backend().copy_h2d(&[9.0, 8.0], &mut b);
    let mut out = pool.backend().raw_alloc::<f64>(2).unwrap();
    let mut workspace = BackendKrylovWorkspace::new(2);
    let status = try_gmres_backend(
        pool.backend(),
        &pool,
        &mut workspace,
        &operator(pool.backend()),
        &IdentityDevicePreconditioner,
        &b,
        None,
        &mut out,
        &BackendKrylovConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-10,
            require_native: true,
            exact_residual_check_interval: 0,
        },
        None,
    )
    .unwrap();
    assert!(status.converged);
    assert_eq!(status.stopping_reason, BackendStoppingReason::Converged);
    assert!(status.true_residual <= 1.0e-10);
}

#[test]
fn backend_block_solve_uses_reusable_workspace() {
    let backend = CpuBackend;
    let matrix = operator(&backend);
    let mut b = backend.raw_alloc::<f64>(4).unwrap();
    backend.copy_h2d(&[9.0, 8.0, 2.0, 1.0], &mut b);
    let mut out = backend.raw_alloc::<f64>(4).unwrap();
    let mut workspace = BackendKrylovWorkspace::new(2);
    let config = BackendKrylovConfig {
        max_iterations: 4,
        restart: 2,
        tolerance: 1.0e-10,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let first = try_block_gmres_backend(
        &backend,
        &mut workspace,
        &matrix,
        &IdentityDevicePreconditioner,
        &b,
        None,
        &mut out,
        2,
        &config,
        None,
    )
    .unwrap();
    let capacity = first.workspace_device_capacity;
    assert!(first.converged);

    let second = try_block_gmres_backend(
        &backend,
        &mut workspace,
        &matrix,
        &IdentityDevicePreconditioner,
        &b,
        None,
        &mut out,
        2,
        &config,
        None,
    )
    .unwrap();
    assert!(second.converged);
    assert_eq!(second.workspace_device_capacity, capacity);
}
