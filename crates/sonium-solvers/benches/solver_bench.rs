//! Iterative solver benchmarks
//!
//! Benchmarks all solver × preconditioner combinations across matrix sizes.
//! Each benchmark measures full solve time (not including preconditioner setup).

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::{
    AdditiveSchwarzPreconditioner, AmgCoarsening, AmgConfig, AmgCycle, AmgPreconditioner,
    AmgSmoother, BiCgstabConfig, BlockJacobiPreconditioner, CgConfig, DiagonalPreconditioner,
    FgmresConfig, GmresConfig, HiluConfig, HiluPreconditioner, IdrsConfig, IluPreconditioner,
    IlutConfig, IlutPreconditioner, bicgstab, cg, fgmres_preconditioned, gmres,
    gmres_preconditioned, idrs, idrs_preconditioned, pcg,
};
use std::time::Duration;

/// Standard solver tolerance for benchmarks.
const TOL: f64 = 1e-6;
/// Max iterations — high enough to always converge on these test problems.
const MAX_ITER: usize = 500;

// ---------------------------------------------------------------------------
// GMRES benchmarks (f64, Poisson)
// ---------------------------------------------------------------------------

fn bench_gmres(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/gmres");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = GmresConfig {
        max_iterations: MAX_ITER,
        restart: 30,
        tolerance: TOL,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // No preconditioner — skip above 10k DOFs (too slow)
        if nn <= 10_000 {
            group.bench_with_input(
                BenchmarkId::new("none", nn),
                &(&matrix, &rhs, &config),
                |b, &(m, r, cfg)| b.iter(|| gmres(m, r, cfg)),
            );
        }

        // Jacobi — skip above 10k DOFs (too slow)
        if nn <= 10_000 {
            let jacobi = DiagonalPreconditioner::from_csr(&matrix);
            group.bench_with_input(
                BenchmarkId::new("jacobi", nn),
                &(&matrix, &jacobi, &rhs, &config),
                |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
            );
        }

        // ILU(0)
        let ilu = IluPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("ilu0", nn),
            &(&matrix, &ilu, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );

        // ILUT
        let ilut = IlutPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("ilut", nn),
            &(&matrix, &ilut, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );

        // AMG tuned for Poisson-like elliptic systems.
        let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
        group.bench_with_input(
            BenchmarkId::new("amg", nn),
            &(&matrix, &amg, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// GMRES preconditioner choice benchmarks (f64, Poisson)
// ---------------------------------------------------------------------------

fn bench_gmres_preconditioner_choice(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/gmres_preconditioner_choice");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    const NUM_SUBDOMAINS: usize = 8;
    const OVERLAP: usize = 1;

    let config = GmresConfig {
        max_iterations: MAX_ITER,
        restart: 30,
        tolerance: TOL,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };

    let n = matrices::grid_size_2d(10_000);
    let matrix = matrices::poisson_2d(n);
    let nn = matrix.num_rows;
    let rhs = matrices::ones_rhs(nn);
    let ilut_config = IlutConfig::default();

    group.throughput(Throughput::Elements(nn as u64));

    let ilut = IlutPreconditioner::from_csr_with_config(&matrix, &ilut_config);
    let block_jacobi =
        BlockJacobiPreconditioner::from_csr(&matrix, NUM_SUBDOMAINS, ilut_config.clone());
    let schwarz = AdditiveSchwarzPreconditioner::from_csr(&matrix, NUM_SUBDOMAINS, OVERLAP);
    let hilu = HiluPreconditioner::from_csr(&matrix);
    let hilu_1024 = HiluPreconditioner::from_csr_with_config(
        &matrix,
        &HiluConfig {
            min_block_size: 1024,
            max_levels: None,
        },
    );
    let hilu_2048 = HiluPreconditioner::from_csr_with_config(
        &matrix,
        &HiluConfig {
            min_block_size: 2048,
            max_levels: None,
        },
    );
    let hilu_4096 = HiluPreconditioner::from_csr_with_config(
        &matrix,
        &HiluConfig {
            min_block_size: 4096,
            max_levels: None,
        },
    );
    let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::default());
    let amg_sgs = AmgPreconditioner::from_csr(
        &matrix,
        AmgConfig {
            smoother: AmgSmoother::SymmetricGaussSeidel,
            ..AmgConfig::default()
        },
    );
    let amg_chebyshev = AmgPreconditioner::from_csr(
        &matrix,
        AmgConfig {
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            ..AmgConfig::default()
        },
    );
    let amg_w_cycle = AmgPreconditioner::from_csr(
        &matrix,
        AmgConfig {
            cycle: AmgCycle::WCycle,
            ..AmgConfig::default()
        },
    );
    let amg_sa = AmgPreconditioner::from_csr(
        &matrix,
        AmgConfig {
            coarsening: AmgCoarsening::SmoothedAggregation,
            ..AmgConfig::default()
        },
    );
    let amg_agg = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());

    eprintln!("GMRES preconditioner diagnostics ({nn} DOFs, tol={TOL:e})");
    eprintln!(
        "{:<14} {:>10} {:>8} {:>13} {:>10} {:>12}",
        "preconditioner", "iterations", "restarts", "residual", "converged", "status"
    );
    let report = |name: &str, solution: sonium_solvers::GmresSolution<f64>| {
        eprintln!(
            "{name:<14} {:>10} {:>8} {:>13.3e} {:>10} {:?}",
            solution.iterations,
            solution.restarts,
            solution.residual,
            solution.converged,
            solution.status
        );
    };
    report("ilut", gmres_preconditioned(&matrix, &ilut, &rhs, &config));
    report(
        "block_jacobi",
        gmres_preconditioned(&matrix, &block_jacobi, &rhs, &config),
    );
    report(
        "schwarz",
        gmres_preconditioned(&matrix, &schwarz, &rhs, &config),
    );
    report("hilu", gmres_preconditioned(&matrix, &hilu, &rhs, &config));
    report(
        "hilu_1024",
        gmres_preconditioned(&matrix, &hilu_1024, &rhs, &config),
    );
    report(
        "hilu_2048",
        gmres_preconditioned(&matrix, &hilu_2048, &rhs, &config),
    );
    report(
        "hilu_4096",
        gmres_preconditioned(&matrix, &hilu_4096, &rhs, &config),
    );
    report("amg", gmres_preconditioned(&matrix, &amg, &rhs, &config));
    report(
        "amg_sgs",
        gmres_preconditioned(&matrix, &amg_sgs, &rhs, &config),
    );
    report(
        "amg_cheb",
        gmres_preconditioned(&matrix, &amg_chebyshev, &rhs, &config),
    );
    report(
        "amg_w",
        gmres_preconditioned(&matrix, &amg_w_cycle, &rhs, &config),
    );
    report(
        "amg_sa",
        gmres_preconditioned(&matrix, &amg_sa, &rhs, &config),
    );
    report(
        "amg_agg",
        gmres_preconditioned(&matrix, &amg_agg, &rhs, &config),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/ilut", nn),
        &(&matrix, &ilut, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/block_jacobi", nn),
        &(&matrix, &block_jacobi, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/schwarz", nn),
        &(&matrix, &schwarz, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/hilu", nn),
        &(&matrix, &hilu, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/amg", nn),
        &(&matrix, &amg, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/amg_sgs", nn),
        &(&matrix, &amg_sgs, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/amg_cheb", nn),
        &(&matrix, &amg_chebyshev, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/amg_sa", nn),
        &(&matrix, &amg_sa, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_with_input(
        BenchmarkId::new("solve_only/amg_agg", nn),
        &(&matrix, &amg_agg, &rhs, &config),
        |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
    );

    group.bench_function(BenchmarkId::new("setup_solve/ilut", nn), |b| {
        b.iter(|| {
            let precond = IlutPreconditioner::from_csr_with_config(&matrix, &ilut_config);
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.bench_function(BenchmarkId::new("setup_solve/block_jacobi", nn), |b| {
        b.iter(|| {
            let precond =
                BlockJacobiPreconditioner::from_csr(&matrix, NUM_SUBDOMAINS, ilut_config.clone());
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.bench_function(BenchmarkId::new("setup_solve/schwarz", nn), |b| {
        b.iter(|| {
            let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, NUM_SUBDOMAINS, OVERLAP);
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.bench_function(BenchmarkId::new("setup_solve/hilu", nn), |b| {
        b.iter(|| {
            let precond = HiluPreconditioner::from_csr(&matrix);
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.bench_function(BenchmarkId::new("setup_solve/amg", nn), |b| {
        b.iter(|| {
            let precond = AmgPreconditioner::from_csr(&matrix, AmgConfig::default());
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.bench_function(BenchmarkId::new("setup_solve/amg_sgs", nn), |b| {
        b.iter(|| {
            let precond = AmgPreconditioner::from_csr(
                &matrix,
                AmgConfig {
                    smoother: AmgSmoother::SymmetricGaussSeidel,
                    ..AmgConfig::default()
                },
            );
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.bench_function(BenchmarkId::new("setup_solve/amg_sa", nn), |b| {
        b.iter(|| {
            let precond = AmgPreconditioner::from_csr(
                &matrix,
                AmgConfig {
                    coarsening: AmgCoarsening::SmoothedAggregation,
                    ..AmgConfig::default()
                },
            );
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.bench_function(BenchmarkId::new("setup_solve/amg_agg", nn), |b| {
        b.iter(|| {
            let precond = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
            gmres_preconditioned(&matrix, &precond, &rhs, &config)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// FGMRES benchmarks (right-preconditioned, important for variable preconds)
// ---------------------------------------------------------------------------

fn bench_fgmres(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/fgmres");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = FgmresConfig {
        max_iterations: MAX_ITER,
        restart: 30,
        tolerance: TOL,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // Jacobi
        let jacobi = DiagonalPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("jacobi", nn),
            &(&matrix, &jacobi, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| fgmres_preconditioned(m, p, r, None, cfg)),
        );

        // AMG tuned for Poisson-like elliptic systems.
        let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
        group.bench_with_input(
            BenchmarkId::new("amg", nn),
            &(&matrix, &amg, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| fgmres_preconditioned(m, p, r, None, cfg)),
        );

        // AMG with Chebyshev smoother
        let amg_cheb = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::Chebyshev,
                num_pre_smooth: 2,
                num_post_smooth: 2,
                ..AmgConfig::default()
            },
        );
        group.bench_with_input(
            BenchmarkId::new("amg_chebyshev", nn),
            &(&matrix, &amg_cheb, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| fgmres_preconditioned(m, p, r, None, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// IDR(s) benchmarks
// ---------------------------------------------------------------------------

fn bench_idrs(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/idrs");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // Vary s parameter
        for &s in &[2, 4, 8] {
            let config = IdrsConfig {
                s,
                max_iterations: MAX_ITER,
                tolerance: TOL,
                print_interval: 0,
            };

            group.bench_with_input(
                BenchmarkId::new(format!("s{s}/none"), nn),
                &(&matrix, &rhs, &config),
                |b, &(m, r, cfg)| b.iter(|| idrs(m, r, cfg)),
            );
        }

        // IDR(4) with ILUT preconditioner
        let ilut = IlutPreconditioner::from_csr(&matrix);
        let config_s4 = IdrsConfig {
            s: 4,
            max_iterations: MAX_ITER,
            tolerance: TOL,
            print_interval: 0,
        };

        group.bench_with_input(
            BenchmarkId::new("s4/ilut", nn),
            &(&matrix, &ilut, &rhs, &config_s4),
            |b, &(m, p, r, cfg)| b.iter(|| idrs_preconditioned(m, p, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// BiCGSTAB benchmarks (unpreconditioned only)
// ---------------------------------------------------------------------------

fn bench_bicgstab(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/bicgstab");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = BiCgstabConfig {
        max_iterations: MAX_ITER,
        tolerance: TOL,
        print_interval: 0,
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs(nn);

        group.throughput(Throughput::Elements(nn as u64));

        group.bench_with_input(
            BenchmarkId::new("none", nn),
            &(&matrix, &rhs, &config),
            |b, &(m, r, cfg)| b.iter(|| bicgstab(m, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// CG / PCG benchmarks (SPD only)
// ---------------------------------------------------------------------------

fn bench_cg(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/cg");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = CgConfig {
        max_iterations: MAX_ITER,
        tolerance: TOL,
        print_interval: 0,
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // Unpreconditioned CG
        group.bench_with_input(
            BenchmarkId::new("none", nn),
            &(&matrix, &rhs, &config),
            |b, &(m, r, cfg)| b.iter(|| cg(m, r, cfg)),
        );

        // PCG + Jacobi (SPD preconditioner — safe for PCG)
        let jacobi = DiagonalPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("jacobi", nn),
            &(&matrix, &jacobi, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| pcg(m, p, r, cfg)),
        );

        // PCG + AMG (symmetric GS smoother — safe for PCG)
        let amg_sym = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::SymmetricGaussSeidel,
                ..AmgConfig::default()
            },
        );
        group.bench_with_input(
            BenchmarkId::new("amg_sgs", nn),
            &(&matrix, &amg_sym, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| pcg(m, p, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Helmholtz benchmarks (complex, non-symmetric — the hard case)
// ---------------------------------------------------------------------------

fn bench_helmholtz(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/helmholtz");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Test different wavenumbers at fixed grid size
    let n = matrices::grid_size_2d(5_000); // ~5K DOFs
    let gmres_config = GmresConfig {
        max_iterations: MAX_ITER,
        restart: 50,
        tolerance: TOL,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };
    let idrs_config = IdrsConfig {
        s: 4,
        max_iterations: MAX_ITER,
        tolerance: TOL,
        print_interval: 0,
    };

    for &k in &[1.0, 5.0, 10.0] {
        let matrix = matrices::helmholtz_2d(n, k);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs_c64(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // GMRES (no precond)
        group.bench_with_input(
            BenchmarkId::new(format!("gmres/k{k}"), nn),
            &(&matrix, &rhs, &gmres_config),
            |b, &(m, r, cfg)| b.iter(|| sonium_solvers::gmres(m, r, cfg)),
        );

        // IDR(4) (no precond)
        group.bench_with_input(
            BenchmarkId::new(format!("idrs4/k{k}"), nn),
            &(&matrix, &rhs, &idrs_config),
            |b, &(m, r, cfg)| b.iter(|| sonium_solvers::idrs(m, r, cfg)),
        );

        // GMRES + Jacobi
        let jacobi = sonium_solvers::DiagonalPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new(format!("gmres_jacobi/k{k}"), nn),
            &(&matrix, &jacobi, &rhs, &gmres_config),
            |b, &(m, p, r, cfg)| b.iter(|| sonium_solvers::gmres_preconditioned(m, p, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// f32 GMRES benchmarks (Poisson)
// ---------------------------------------------------------------------------

fn bench_gmres_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/gmres_f32");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = GmresConfig {
        max_iterations: MAX_ITER,
        restart: 30,
        tolerance: 1e-4_f32,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d_f32(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs_f32(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // ILU(0)
        let ilu = IluPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("ilu0", nn),
            &(&matrix, &ilu, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );

        // ILUT
        let ilut = IlutPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("ilut", nn),
            &(&matrix, &ilut, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );

        // AMG tuned for Poisson-like elliptic systems.
        let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
        group.bench_with_input(
            BenchmarkId::new("amg", nn),
            &(&matrix, &amg, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// f32 CG benchmarks (SPD only)
// ---------------------------------------------------------------------------

fn bench_cg_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/cg_f32");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = CgConfig {
        max_iterations: MAX_ITER,
        tolerance: 1e-4_f32,
        print_interval: 0,
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d_f32(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs_f32(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // Unpreconditioned CG
        group.bench_with_input(
            BenchmarkId::new("none", nn),
            &(&matrix, &rhs, &config),
            |b, &(m, r, cfg)| b.iter(|| cg(m, r, cfg)),
        );

        // PCG + Jacobi
        let jacobi = DiagonalPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("jacobi", nn),
            &(&matrix, &jacobi, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| pcg(m, p, r, cfg)),
        );

        // PCG + AMG (symmetric GS smoother)
        let amg_sym = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::SymmetricGaussSeidel,
                ..AmgConfig::default()
            },
        );
        group.bench_with_input(
            BenchmarkId::new("amg_sgs", nn),
            &(&matrix, &amg_sym, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| pcg(m, p, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// f32 BiCGSTAB benchmarks
// ---------------------------------------------------------------------------

fn bench_bicgstab_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/bicgstab_f32");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = BiCgstabConfig {
        max_iterations: MAX_ITER,
        tolerance: 1e-4_f32,
        print_interval: 0,
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d_f32(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs_f32(nn);

        group.throughput(Throughput::Elements(nn as u64));

        group.bench_with_input(
            BenchmarkId::new("none", nn),
            &(&matrix, &rhs, &config),
            |b, &(m, r, cfg)| b.iter(|| bicgstab(m, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// f16 GMRES benchmarks (Poisson)
// ---------------------------------------------------------------------------

fn bench_gmres_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/gmres_f16");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = GmresConfig {
        max_iterations: MAX_ITER,
        restart: 30,
        tolerance: half::f16::from_f32(1e-2),
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        min_restart: None,
        max_restart: None,
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d_f16(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs_f16(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // ILU(0)
        let ilu = IluPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("ilu0", nn),
            &(&matrix, &ilu, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );

        // Jacobi
        let jacobi = DiagonalPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("jacobi", nn),
            &(&matrix, &jacobi, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| gmres_preconditioned(m, p, r, cfg)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// f16 CG benchmarks (SPD only)
// ---------------------------------------------------------------------------

fn bench_cg_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver/cg_f16");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let config = CgConfig {
        max_iterations: MAX_ITER,
        tolerance: half::f16::from_f32(1e-2),
        print_interval: 0,
    };

    for &target_dofs in &[1_000, 10_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d_f16(n);
        let nn = matrix.num_rows;
        let rhs = matrices::ones_rhs_f16(nn);

        group.throughput(Throughput::Elements(nn as u64));

        // Unpreconditioned CG
        group.bench_with_input(
            BenchmarkId::new("none", nn),
            &(&matrix, &rhs, &config),
            |b, &(m, r, cfg)| b.iter(|| cg(m, r, cfg)),
        );

        // PCG + Jacobi
        let jacobi = DiagonalPreconditioner::from_csr(&matrix);
        group.bench_with_input(
            BenchmarkId::new("jacobi", nn),
            &(&matrix, &jacobi, &rhs, &config),
            |b, &(m, p, r, cfg)| b.iter(|| pcg(m, p, r, cfg)),
        );
    }

    group.finish();
}

criterion_group!(
    name = solver_benches;
    config = Criterion::default();
    targets = bench_gmres, bench_gmres_preconditioner_choice, bench_fgmres,
              bench_idrs, bench_bicgstab, bench_cg, bench_helmholtz,
              bench_gmres_f32, bench_cg_f32, bench_bicgstab_f32,
              bench_gmres_f16, bench_cg_f16
);
criterion_main!(solver_benches);
