//! Block-GMRES backend benchmark.
//!
//! Fixture matrices use the checked `try_*` constructors, matching the
//! production contract.
//!
//! The block Krylov implementation keeps the large Arnoldi state on the
//! backend and exposes transfer counters alongside the timing.  The benchmark
//! warms a reusable workspace before measuring, then runs the identical
//! workload on CPU and, when enabled, CUDA and Metal.
//!
//! ```text
//! cargo bench -p sonium-compute --bench block_gmres_accelerator
//! cargo bench -p sonium-compute --bench block_gmres_accelerator --features cuda
//! cargo bench -p sonium-compute --bench block_gmres_accelerator --features metal
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use sonium_compute::{
    Backend, BlockGmresConfig, BlockGmresWorkspace, CpuBackend, DeviceCsr,
    IdentityDevicePreconditioner, gmres_backend_block_preconditioned_with_workspace,
};
use std::hint::black_box;

fn bench_backend<B: Backend>(criterion: &mut Criterion, name: &str, backend: &B) {
    const N: usize = 512;
    const RHS: usize = 8;
    let diagonal = vec![2.0_f32; N];
    let operator = DeviceCsr::try_from_host_diagonal(backend, &diagonal)
        .expect("benchmark diagonal CSR upload");
    let mut rhs_host = vec![0.0_f32; N * RHS];
    for column in 0..RHS {
        for row in 0..N {
            rhs_host[column * N + row] = 1.0 + (row + column) as f32 * 1.0e-3;
        }
    }
    let mut rhs = backend
        .raw_alloc::<f32>(rhs_host.len())
        .expect("benchmark RHS allocation");
    backend.copy_h2d(&rhs_host, &mut rhs);
    let mut out = backend
        .raw_alloc::<f32>(rhs_host.len())
        .expect("benchmark solution allocation");
    let config = BlockGmresConfig {
        max_block_iterations: 4,
        tolerance: 1.0e-5,
    };
    let preconditioner = IdentityDevicePreconditioner;
    let mut workspace = BlockGmresWorkspace::new(N);

    criterion.bench_function(&format!("block_gmres/{name}/n{N}/rhs{RHS}"), |bencher| {
        bencher.iter(|| {
            let before = backend.transfer_stats();
            let status = gmres_backend_block_preconditioned_with_workspace(
                backend,
                &operator,
                &preconditioner,
                &rhs,
                None,
                &mut out,
                RHS,
                &config,
                &mut workspace,
            )
            .expect("diagonal benchmark system should converge");
            let after = backend.transfer_stats();
            black_box((
                status.block_iterations,
                status.final_residual,
                after.d2h_operations.saturating_sub(before.d2h_operations),
                after.d2h_bytes.saturating_sub(before.d2h_bytes),
            ));
        });
    });
}

fn block_gmres_accelerator(c: &mut Criterion) {
    let cpu = CpuBackend::new();
    bench_backend(c, "cpu", &cpu);

    #[cfg(feature = "cuda")]
    if let Ok(cuda) = sonium_compute::CudaBackend::new_default() {
        bench_backend(c, "cuda", &cuda);
    }

    #[cfg(feature = "metal")]
    if let Ok(metal) = sonium_compute::MetalBackend::new() {
        bench_backend(c, "metal", &metal);
    }
}

criterion_group!(benches, block_gmres_accelerator);
criterion_main!(benches);
