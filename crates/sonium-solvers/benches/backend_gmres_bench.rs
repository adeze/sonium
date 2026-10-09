//! Backend-aware GMRES vs legacy CPU GMRES — Phase 4 of the GPU-offload
//! roadmap.
//!
//! Compares per-solve wall-time for the same linear system across:
//!
//! - **CPU + ILU (legacy)** — `gmres_preconditioned_with_guess` over
//!   in-house BLAS-1, with ILU(0) preconditioning. Strong but inherently
//!   serial (triangular solve in apply).
//! - **CpuBackend + Polynomial Jacobi** — `gmres_backend_preconditioned`
//!   driving `PolynomialJacobiBackend` at degrees 1, 2, 3. Same kernel
//!   code path that CUDA/Metal hit; on CPU it falls through to the trivial
//!   `Vec<f64>` BLAS-1 implementation.
//!
//! Under `--features cuda` / `--features metal`, the same GMRES-via-backend
//! arm is exercised on the GPU so the bench tells you the actual offload
//! win. Without those features the bench still runs CpuBackend, which
//! serves as a regression watch on the kernel surface.
//!
//! # What this bench answers
//!
//! - "How much do I pay in iterations to drop ILU for a GPU-friendly
//!   preconditioner?"  → Compare iter counts in the criterion `extra_info`
//!   labels (printed before each group).
//! - "Where does polynomial degree start helping?" → Compare `poly_K=1` vs
//!   `poly_K=2` vs `poly_K=3` time-per-solve.
//! - "Is the CpuBackend dispatcher costing measurable wall-time vs the
//!   inline legacy path?" → Compare `cpu_legacy` to `cpu_backend_poly_K=1`.

#![allow(clippy::needless_range_loop)]
// Bench fixtures intentionally use the panicking convenience constructors on
// fixed matrices; production paths must use the `try_*` variants.
#![allow(deprecated)]

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_compute::gmres::{BackendCsr, GmresBackendConfig, gmres_backend_preconditioned};
use sonium_compute::gmres_ir::{BackendCsrF32, GmresIrConfig, gmres_ir_backend_preconditioned};
use sonium_compute::{
    Backend, ChebyshevBackend, ChebyshevBackendF32, CpuBackend, PolynomialJacobiBackend,
    PolynomialJacobiBackendF32,
};
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres_preconditioned_with_guess};
use std::time::Duration;

/// Convert a `CsrMatrix<f64>` to the i32 host-arrays expected by the
/// backend layer.
fn csr_to_i32_host(csr: &CsrMatrix<f64>) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let row_ptr: Vec<i32> = csr.row_ptrs.iter().map(|&v| v as i32).collect();
    let col_idx: Vec<i32> = csr.col_indices.iter().map(|&v| v as i32).collect();
    (row_ptr, col_idx, csr.values.clone())
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.0123).sin() + 1.0).collect()
}

fn bench_legacy_cpu_ilu(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    csr: &CsrMatrix<f64>,
    b_vec: &[f64],
) {
    let nn = csr.num_rows;
    group.bench_with_input(BenchmarkId::new("cpu_legacy_ilu", nn), &(), |bencher, _| {
        bencher.iter(|| {
            let precond = IluPreconditioner::from_csr(csr);
            let cfg = GmresConfig {
                max_iterations: 200,
                tolerance: 1.0e-8,
                ..Default::default()
            };
            let result = gmres_preconditioned_with_guess(csr, &precond, b_vec, None, &cfg);
            std::hint::black_box(result.x);
        });
    });
}

fn bench_backend_poly<B: Backend>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    backend: &B,
    csr: &CsrMatrix<f64>,
    b_vec: &[f64],
    label: &str,
    degree: usize,
    damping: f64,
) {
    let nn = csr.num_rows;
    let (rp, ci, va) = csr_to_i32_host(csr);
    let device_csr = BackendCsr::from_host(backend, nn, nn, &rp, &ci, &va);
    let mut b_dev = backend.raw_alloc::<f64>(nn).unwrap();
    backend.copy_h2d(b_vec, &mut b_dev);

    group.bench_with_input(
        BenchmarkId::new(format!("{label}_poly_K={degree}"), nn),
        &(),
        |bencher, _| {
            bencher.iter(|| {
                let precond = PolynomialJacobiBackend::from_csr_host(
                    backend,
                    &device_csr,
                    &rp,
                    &ci,
                    &va,
                    degree,
                    damping,
                );
                let mut out = backend.raw_alloc::<f64>(nn).unwrap();
                let _ = gmres_backend_preconditioned(
                    backend,
                    &device_csr,
                    &precond,
                    &b_dev,
                    None,
                    &mut out,
                    &GmresBackendConfig {
                        max_iterations: 200,
                        tolerance: 1.0e-8,
                    },
                )
                .expect("backend GMRES");
                std::hint::black_box(&out);
            });
        },
    );
}

/// Mixed-precision GMRES-IR (FGMRES outer in f64, inner Krylov in f32).
/// Drives the native f32 MSL kernels on Metal — this is where the Apple
/// Silicon win shows up.
fn bench_backend_ir<B: Backend>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    backend: &B,
    csr: &CsrMatrix<f64>,
    b_vec: &[f64],
    label: &str,
    inner_degree: usize,
) {
    let nn = csr.num_rows;
    let (rp, ci, va_f64) = csr_to_i32_host(csr);
    let device_csr_f64 = BackendCsr::from_host(backend, nn, nn, &rp, &ci, &va_f64);
    let device_csr_f32 = BackendCsrF32::from_host_f64_values(backend, nn, nn, &rp, &ci, &va_f64);
    let mut b_dev = backend.raw_alloc::<f64>(nn).unwrap();
    backend.copy_h2d(b_vec, &mut b_dev);
    let va_f32: Vec<f32> = va_f64.iter().map(|&v| v as f32).collect();

    group.bench_with_input(
        BenchmarkId::new(format!("{label}_ir_inner_K={inner_degree}"), nn),
        &(),
        |bencher, _| {
            bencher.iter(|| {
                let pc = PolynomialJacobiBackendF32::from_csr_f32(
                    backend,
                    &device_csr_f32,
                    &rp,
                    &ci,
                    &va_f32,
                    inner_degree,
                    1.0,
                );
                let mut out = backend.raw_alloc::<f64>(nn).unwrap();
                let _ = gmres_ir_backend_preconditioned(
                    backend,
                    &device_csr_f64,
                    &device_csr_f32,
                    &pc,
                    &b_dev,
                    None,
                    &mut out,
                    &GmresIrConfig {
                        max_outer_iterations: 60,
                        outer_tolerance: 1.0e-8,
                        max_inner_iterations: 80,
                        inner_tolerance: 1.0e-3,
                        ..Default::default()
                    },
                )
                .expect("GMRES-IR");
                std::hint::black_box(&out);
            });
        },
    );
}

/// FGMRES-IR with Chebyshev_F32 inner preconditioner — the strongest
/// available f32 inner choice.
fn bench_backend_ir_chebyshev<B: Backend>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    backend: &B,
    csr: &CsrMatrix<f64>,
    b_vec: &[f64],
    label: &str,
    inner_degree: usize,
) {
    let nn = csr.num_rows;
    let (rp, ci, va_f64) = csr_to_i32_host(csr);
    let device_csr_f64 = BackendCsr::from_host(backend, nn, nn, &rp, &ci, &va_f64);
    let device_csr_f32 = BackendCsrF32::from_host_f64_values(backend, nn, nn, &rp, &ci, &va_f64);
    let mut b_dev = backend.raw_alloc::<f64>(nn).unwrap();
    backend.copy_h2d(b_vec, &mut b_dev);
    let va_f32: Vec<f32> = va_f64.iter().map(|&v| v as f32).collect();

    group.bench_with_input(
        BenchmarkId::new(format!("{label}_ir_cheb_K={inner_degree}"), nn),
        &(),
        |bencher, _| {
            bencher.iter(|| {
                let pc = ChebyshevBackendF32::from_csr_f32(
                    backend,
                    &device_csr_f32,
                    &rp,
                    &ci,
                    &va_f32,
                    inner_degree,
                    30.0,
                );
                let mut out = backend.raw_alloc::<f64>(nn).unwrap();
                let _ = gmres_ir_backend_preconditioned(
                    backend,
                    &device_csr_f64,
                    &device_csr_f32,
                    &pc,
                    &b_dev,
                    None,
                    &mut out,
                    &GmresIrConfig {
                        max_outer_iterations: 60,
                        outer_tolerance: 1.0e-8,
                        max_inner_iterations: 80,
                        inner_tolerance: 1.0e-3,
                        ..Default::default()
                    },
                )
                .expect("GMRES-IR");
                std::hint::black_box(&out);
            });
        },
    );
}

fn bench_backend_chebyshev<B: Backend>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    backend: &B,
    csr: &CsrMatrix<f64>,
    b_vec: &[f64],
    label: &str,
    degree: usize,
) {
    let nn = csr.num_rows;
    let (rp, ci, va) = csr_to_i32_host(csr);
    let device_csr = BackendCsr::from_host(backend, nn, nn, &rp, &ci, &va);
    let mut b_dev = backend.raw_alloc::<f64>(nn).unwrap();
    backend.copy_h2d(b_vec, &mut b_dev);

    group.bench_with_input(
        BenchmarkId::new(format!("{label}_cheb_K={degree}"), nn),
        &(),
        |bencher, _| {
            bencher.iter(|| {
                let precond = ChebyshevBackend::from_csr_host(
                    backend,
                    &device_csr,
                    &rp,
                    &ci,
                    &va,
                    degree,
                    30.0,
                );
                let mut out = backend.raw_alloc::<f64>(nn).unwrap();
                let _ = gmres_backend_preconditioned(
                    backend,
                    &device_csr,
                    &precond,
                    &b_dev,
                    None,
                    &mut out,
                    &GmresBackendConfig {
                        max_iterations: 200,
                        tolerance: 1.0e-8,
                    },
                )
                .expect("backend GMRES");
                std::hint::black_box(&out);
            });
        },
    );
}

fn bench_full_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_gmres/poisson_2d");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let cpu_backend = CpuBackend::new();

    // Three problem sizes spanning the small/medium/large bands.
    for &target_dofs in &[1_000_usize, 10_000, 50_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let csr = matrices::poisson_2d(n);
        let nn = csr.num_rows;
        let b_vec = rhs(nn);

        group.throughput(Throughput::Elements(nn as u64));

        bench_legacy_cpu_ilu(&mut group, &csr, &b_vec);
        for &k in &[1_usize, 2, 3] {
            bench_backend_poly(
                &mut group,
                &cpu_backend,
                &csr,
                &b_vec,
                "cpu_backend",
                k,
                1.0,
            );
        }
        for &k in &[3_usize, 5] {
            bench_backend_chebyshev(&mut group, &cpu_backend, &csr, &b_vec, "cpu_backend", k);
        }
        // Mixed-precision GMRES-IR on CpuBackend — useful baseline for
        // comparing against Metal IR (same algorithm, different backend).
        for &k in &[2_usize, 3] {
            bench_backend_ir(&mut group, &cpu_backend, &csr, &b_vec, "cpu_backend", k);
        }
        for &k in &[3_usize, 5] {
            bench_backend_ir_chebyshev(&mut group, &cpu_backend, &csr, &b_vec, "cpu_backend", k);
        }

        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda) = sonium_compute::cuda::CudaBackend::new_default() {
                for &k in &[1_usize, 2, 3] {
                    bench_backend_poly(&mut group, &cuda, &csr, &b_vec, "cuda", k, 1.0);
                }
                for &k in &[3_usize, 5] {
                    bench_backend_chebyshev(&mut group, &cuda, &csr, &b_vec, "cuda", k);
                }
                for &k in &[2_usize, 3] {
                    bench_backend_ir(&mut group, &cuda, &csr, &b_vec, "cuda", k);
                }
            }
        }

        #[cfg(feature = "metal")]
        {
            if let Ok(metal) = sonium_compute::metal::MetalBackend::new() {
                for &k in &[1_usize, 2, 3] {
                    bench_backend_poly(&mut group, &metal, &csr, &b_vec, "metal", k, 1.0);
                }
                for &k in &[3_usize, 5] {
                    bench_backend_chebyshev(&mut group, &metal, &csr, &b_vec, "metal", k);
                }
                // The headline arms — these exercise the native f32 MSL
                // kernels via FGMRES-IR + multi_dot CGS2.
                for &k in &[2_usize, 3] {
                    bench_backend_ir(&mut group, &metal, &csr, &b_vec, "metal", k);
                }
                for &k in &[3_usize, 5] {
                    bench_backend_ir_chebyshev(&mut group, &metal, &csr, &b_vec, "metal", k);
                }
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_full_solve);
criterion_main!(benches);
