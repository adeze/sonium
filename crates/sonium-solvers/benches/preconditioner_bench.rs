//! Preconditioner setup and apply benchmarks
//!
//! Separates preconditioner construction cost (one-time) from
//! per-iteration apply cost, across all preconditioner types and matrix sizes.

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgPreconditioner, AmgSmoother,
    BlockJacobiPreconditioner, DiagonalPreconditioner, HiluPreconditioner, IluPreconditioner,
    IlutConfig, IlutPreconditioner, Preconditioner,
};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Preconditioner setup time
// ---------------------------------------------------------------------------

fn bench_precond_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("precond/setup");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    for &target_dofs in &[1_000, 10_000, 100_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;

        group.throughput(Throughput::Elements(nn as u64));

        group.bench_with_input(BenchmarkId::new("jacobi", nn), &matrix, |b, m| {
            b.iter(|| DiagonalPreconditioner::from_csr(m))
        });

        group.bench_with_input(BenchmarkId::new("ilu0", nn), &matrix, |b, m| {
            b.iter(|| IluPreconditioner::from_csr(m))
        });

        group.bench_with_input(BenchmarkId::new("ilut_default", nn), &matrix, |b, m| {
            b.iter(|| IlutPreconditioner::from_csr(m))
        });

        let ilut_tight = IlutConfig::new(1e-8, 20, 1e-12);
        group.bench_with_input(BenchmarkId::new("ilut_tight", nn), &matrix, |b, m| {
            b.iter(|| IlutPreconditioner::from_csr_with_config(m, &ilut_tight))
        });

        group.bench_with_input(BenchmarkId::new("amg_default", nn), &matrix, |b, m| {
            b.iter(|| AmgPreconditioner::from_csr(m, AmgConfig::default()))
        });

        group.bench_with_input(BenchmarkId::new("amg_aggregation", nn), &matrix, |b, m| {
            b.iter(|| AmgPreconditioner::from_csr(m, AmgConfig::for_aggregation()))
        });

        let amg_cheb = AmgConfig {
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            ..AmgConfig::default()
        };
        group.bench_with_input(BenchmarkId::new("amg_chebyshev", nn), &matrix, |b, m| {
            b.iter(|| AmgPreconditioner::from_csr(m, amg_cheb.clone()))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Preconditioner apply time (per-call cost in the Krylov loop)
// ---------------------------------------------------------------------------

fn bench_precond_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("precond/apply");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    for &target_dofs in &[1_000, 10_000, 100_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // Jacobi
        let jacobi = DiagonalPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("jacobi", nn),
            &(&jacobi, &rhs),
            |b, &(p, r)| b.iter(|| p.apply(r)),
        );

        // ILU(0)
        let ilu = IluPreconditioner::from_csr(&matrix);
        group.bench_with_input(BenchmarkId::new("ilu0", nn), &(&ilu, &rhs), |b, &(p, r)| {
            b.iter(|| p.apply(r))
        });

        // ILUT
        let ilut = IlutPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("ilut", nn),
            &(&ilut, &rhs),
            |b, &(p, r)| b.iter(|| p.apply(r)),
        );

        // AMG
        let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::default());
        group.bench_with_input(BenchmarkId::new("amg", nn), &(&amg, &rhs), |b, &(p, r)| {
            b.iter(|| p.apply(r))
        });

        let amg_aggregation = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
        group.bench_with_input(
            BenchmarkId::new("amg_aggregation", nn),
            &(&amg_aggregation, &rhs),
            |b, &(p, r)| b.iter(|| p.apply(r)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Preconditioner apply_into time (buffer-reuse path used by Krylov solvers)
// ---------------------------------------------------------------------------

fn bench_precond_apply_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("precond/apply_into");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(8));

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs(nn);
        let mut y = vec![0.0; nn];

        group.throughput(Throughput::Elements(nn as u64));

        let ilut = IlutPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("ilut", nn),
            &(&ilut, &rhs),
            |b, &(p, r)| {
                b.iter(|| {
                    p.apply_into(black_box(r), black_box(y.as_mut_slice()));
                    black_box(&y);
                })
            },
        );

        let block_jacobi = BlockJacobiPreconditioner::from_csr(&matrix, 8, IlutConfig::default());
        group.bench_with_input(
            BenchmarkId::new("block_jacobi", nn),
            &(&block_jacobi, &rhs),
            |b, &(p, r)| {
                b.iter(|| {
                    p.apply_into(black_box(r), black_box(y.as_mut_slice()));
                    black_box(&y);
                })
            },
        );

        let schwarz = AdditiveSchwarzPreconditioner::from_csr(&matrix, 8, 1);
        group.bench_with_input(
            BenchmarkId::new("schwarz", nn),
            &(&schwarz, &rhs),
            |b, &(p, r)| {
                b.iter(|| {
                    p.apply_into(black_box(r), black_box(y.as_mut_slice()));
                    black_box(&y);
                })
            },
        );

        let hilu = HiluPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("hilu", nn),
            &(&hilu, &rhs),
            |b, &(p, r)| {
                b.iter(|| {
                    p.apply_into(black_box(r), black_box(y.as_mut_slice()));
                    black_box(&y);
                })
            },
        );

        let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::default());
        group.bench_with_input(BenchmarkId::new("amg", nn), &(&amg, &rhs), |b, &(p, r)| {
            b.iter(|| {
                p.apply_into(black_box(r), black_box(y.as_mut_slice()));
                black_box(&y);
            })
        });

        let amg_aggregation = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
        group.bench_with_input(
            BenchmarkId::new("amg_aggregation", nn),
            &(&amg_aggregation, &rhs),
            |b, &(p, r)| {
                b.iter(|| {
                    p.apply_into(black_box(r), black_box(y.as_mut_slice()));
                    black_box(&y);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = precond_benches;
    config = Criterion::default();
    targets = bench_precond_setup, bench_precond_apply, bench_precond_apply_into
);
criterion_main!(precond_benches);
