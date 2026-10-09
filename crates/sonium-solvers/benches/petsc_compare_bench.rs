//! PETSc comparison benchmark
//!
//! Measures GMRES solve time (setup + solve) with ILU and SA-AMG preconditioners
//! across problem sizes from 1K to 1M DOFs.
//!
//! None and Jacobi are included only at n=1024 as a baseline reference.
//!
//! A summary table is printed at the end of the run.
//!
//! Run:
//!   cargo bench -p sonium-solvers --bench petsc_compare_bench

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::{
    AmgConfig, AmgDiagnostics, AmgPreconditioner, CsrMatrix, DiagonalPreconditioner, GmresConfig,
    IluPreconditioner, gmres, gmres_preconditioned,
};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(feature = "cli")]
use sonium_solvers::perf::{self, PerfRun};

const TOL: f64 = 1e-6;
const MAX_ITER: usize = 500;

fn gmres_config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: MAX_ITER,
        restart: 30,
        tolerance: TOL,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    }
}

/// Build matrix + rhs for a given target DOF count.
fn setup(target_dofs: usize) -> (CsrMatrix<f64>, Vec<f64>) {
    let n = matrices::grid_size_2d(target_dofs);
    let matrix = matrices::poisson_2d(n);
    let nn = matrix.num_rows;
    let rhs = matrices::ones_rhs(nn);
    (matrix, rhs)
}

// ---------------------------------------------------------------------------
// Table collection
// ---------------------------------------------------------------------------

struct SolverRow {
    section: &'static str,
    variant: &'static str,
    n: usize,
    ms: f64,
}

static SOLVER_TABLE: Mutex<Vec<SolverRow>> = Mutex::new(Vec::new());

fn record(section: &'static str, variant: &'static str, n: usize, ms: f64) {
    SOLVER_TABLE.lock().unwrap().push(SolverRow {
        section,
        variant,
        n,
        ms,
    });
}

#[cfg(feature = "cli")]
struct SolvePerfMeta<'a> {
    section: &'a str,
    variant: &'a str,
    preconditioner: &'a str,
    n: usize,
    nnz: usize,
    setup_ms: f64,
    solve_ms: f64,
    iterations: usize,
    residual: f64,
    converged: bool,
    krylov_op_counts: sonium_solvers::KrylovOpCounts,
    amg_profile: Option<&'a str>,
    amg_diagnostics: Option<AmgDiagnostics>,
    notes: &'a str,
}

#[cfg(feature = "cli")]
fn record_solve_perf(meta: SolvePerfMeta<'_>) {
    if perf::perf_json_path().is_none() {
        return;
    }

    let total_ms = meta.setup_ms + meta.solve_ms;
    let mut run = PerfRun::new(
        "petsc_compare_bench",
        format!("petsc_compare/{}", meta.section),
        format!("{}/{}", meta.variant, meta.n),
    );
    run.operation = "solve".to_string();
    run.matrix = Some("poisson_2d".to_string());
    run.scalar = "f64".to_string();
    run.nrows = Some(meta.n);
    run.ncols = Some(meta.n);
    run.nnz = Some(meta.nnz);
    run.measured_iterations = 1;
    run.setup_ms = Some(meta.setup_ms);
    run.solve_ms = Some(meta.solve_ms);
    run.total_ms = total_ms;
    run.ms_per_iter = total_ms;
    run.solver_iterations = Some(meta.iterations);
    run.residual = Some(meta.residual);
    run.converged = Some(meta.converged);
    run.record_setup_reuse_economics(1, setup_reuse_scope(meta.preconditioner));
    run = run.with_krylov_counts(meta.krylov_op_counts);
    if let Some(profile) = meta.amg_profile {
        run.amg_profile = Some(profile.to_string());
    }
    if let Some(diag) = meta.amg_diagnostics {
        run.amg_num_levels = Some(diag.num_levels);
        run.amg_grid_complexity = Some(diag.grid_complexity);
        run.amg_operator_complexity = Some(diag.operator_complexity);
        run.amg_max_grid_complexity = diag.max_grid_complexity;
        run.amg_max_operator_complexity = diag.max_operator_complexity;
        run.amg_grid_complexity_within_target = Some(diag.grid_complexity_within_target);
        run.amg_operator_complexity_within_target = Some(diag.operator_complexity_within_target);
        run.amg_apply_count = Some(diag.apply_count);
        run.amg_total_apply_ms = Some(diag.total_apply_ms);
        run.amg_mean_apply_ms = Some(diag.mean_apply_ms);
        run.amg_total_nnz = Some(diag.total_nnz);
        run.amg_finest_nnz = Some(diag.finest_nnz);
        run.amg_coarsest_dofs = Some(diag.coarsest_dofs);
    }
    run.notes = Some(format!(
        "solver=gmres; preconditioner={}; tolerance={TOL:.1e}; max_iterations={MAX_ITER}; {}",
        meta.preconditioner, meta.notes
    ));

    if let Err(err) = perf::append_run_from_env(run) {
        eprintln!("failed to write {} report: {err}", perf::PERF_JSON_ENV);
    }
}

#[cfg(feature = "cli")]
fn setup_reuse_scope(preconditioner: &str) -> &'static str {
    match preconditioner {
        "none" => "no_setup",
        "jacobi" => "cheap_same_matrix_repeated_rhs",
        "ilu" | "amg" => "same_matrix_repeated_rhs",
        _ => "setup_not_reused",
    }
}

/// Single-shot timer (warmup + one timed call). Good for operations > ~1 ms.
fn time_ms<F, T>(f: F) -> f64
where
    F: Fn() -> T,
{
    std::hint::black_box(f()); // warmup
    let t = Instant::now();
    std::hint::black_box(f());
    t.elapsed().as_secs_f64() * 1e3
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{} M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{} K", n / 1_000)
    } else {
        n.to_string()
    }
}

// ---------------------------------------------------------------------------
// Baseline: none + jacobi at 1K only
// ---------------------------------------------------------------------------

fn bench_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("petsc_compare/baseline");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let (matrix, rhs) = setup(1_024);
    let nn = matrix.num_rows;
    let cfg = gmres_config();

    group.throughput(Throughput::Elements(nn as u64));

    #[cfg(feature = "cli")]
    if perf::perf_json_path().is_some() {
        let (solution, solve_ms) = perf::time_once(|| gmres(&matrix, &rhs, &cfg));
        record_solve_perf(SolvePerfMeta {
            section: "baseline",
            variant: "none",
            preconditioner: "none",
            n: nn,
            nnz: matrix.nnz(),
            setup_ms: 0.0,
            solve_ms,
            iterations: solution.iterations,
            residual: solution.residual,
            converged: solution.converged,
            krylov_op_counts: solution.op_counts,
            amg_profile: None,
            amg_diagnostics: None,
            notes: "setup excluded",
        });
    }

    record(
        "baseline",
        "none",
        nn,
        time_ms(|| gmres(&matrix, &rhs, &cfg)),
    );

    group.bench_function(BenchmarkId::new("none", nn), |b| {
        b.iter(|| gmres(&matrix, &rhs, &cfg))
    });

    let jacobi = DiagonalPreconditioner::from_csr(&matrix);
    #[cfg(feature = "cli")]
    if perf::perf_json_path().is_some() {
        let (solution, solve_ms) =
            perf::time_once(|| gmres_preconditioned(&matrix, &jacobi, &rhs, &cfg));
        record_solve_perf(SolvePerfMeta {
            section: "baseline",
            variant: "jacobi",
            preconditioner: "jacobi",
            n: nn,
            nnz: matrix.nnz(),
            setup_ms: 0.0,
            solve_ms,
            iterations: solution.iterations,
            residual: solution.residual,
            converged: solution.converged,
            krylov_op_counts: solution.op_counts,
            amg_profile: None,
            amg_diagnostics: None,
            notes: "setup excluded",
        });
    }

    record(
        "baseline",
        "jacobi",
        nn,
        time_ms(|| gmres_preconditioned(&matrix, &jacobi, &rhs, &cfg)),
    );

    group.bench_function(BenchmarkId::new("jacobi", nn), |b| {
        b.iter(|| gmres_preconditioned(&matrix, &jacobi, &rhs, &cfg))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// ILU(0): 1K, 10K, 100K, 1M — includes setup in the measurement
// ---------------------------------------------------------------------------

fn bench_ilu(c: &mut Criterion) {
    let mut group = c.benchmark_group("petsc_compare/ilu");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(30));

    for &target_dofs in &[1_024, 10_000, 100_000, 1_000_000] {
        let (matrix, rhs) = setup(target_dofs);
        let nn = matrix.num_rows;
        let cfg = gmres_config();

        group.throughput(Throughput::Elements(nn as u64));

        #[cfg(feature = "cli")]
        if perf::perf_json_path().is_some() {
            let (ilu, setup_ms) = perf::time_once(|| IluPreconditioner::from_csr(&matrix));
            let (solution, solve_ms) =
                perf::time_once(|| gmres_preconditioned(&matrix, &ilu, &rhs, &cfg));
            record_solve_perf(SolvePerfMeta {
                section: "ilu",
                variant: "setup+solve",
                preconditioner: "ilu",
                n: nn,
                nnz: matrix.nnz(),
                setup_ms,
                solve_ms,
                iterations: solution.iterations,
                residual: solution.residual,
                converged: solution.converged,
                krylov_op_counts: solution.op_counts,
                amg_profile: None,
                amg_diagnostics: None,
                notes: "setup and solve timed separately",
            });
        }

        // Setup + solve together (what matters for PETSc comparison)
        record(
            "ilu",
            "setup+solve",
            nn,
            time_ms(|| {
                let ilu = IluPreconditioner::from_csr(&matrix);
                gmres_preconditioned(&matrix, &ilu, &rhs, &cfg)
            }),
        );

        group.bench_function(BenchmarkId::new("setup+solve", nn), |b| {
            b.iter(|| {
                let ilu = IluPreconditioner::from_csr(&matrix);
                gmres_preconditioned(&matrix, &ilu, &rhs, &cfg)
            })
        });

        // Solve only (preconditioner reuse scenario)
        let ilu = IluPreconditioner::from_csr(&matrix);
        #[cfg(feature = "cli")]
        if perf::perf_json_path().is_some() {
            let (solution, solve_ms) =
                perf::time_once(|| gmres_preconditioned(&matrix, &ilu, &rhs, &cfg));
            record_solve_perf(SolvePerfMeta {
                section: "ilu",
                variant: "solve_only",
                preconditioner: "ilu",
                n: nn,
                nnz: matrix.nnz(),
                setup_ms: 0.0,
                solve_ms,
                iterations: solution.iterations,
                residual: solution.residual,
                converged: solution.converged,
                krylov_op_counts: solution.op_counts,
                amg_profile: None,
                amg_diagnostics: None,
                notes: "preconditioner setup excluded",
            });
        }

        record(
            "ilu",
            "solve_only",
            nn,
            time_ms(|| gmres_preconditioned(&matrix, &ilu, &rhs, &cfg)),
        );

        group.bench_function(BenchmarkId::new("solve_only", nn), |b| {
            b.iter(|| gmres_preconditioned(&matrix, &ilu, &rhs, &cfg))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// SA-AMG (energy-minimizing): 1K, 10K, 100K, 1M
// ---------------------------------------------------------------------------

fn bench_sa_amg(c: &mut Criterion) {
    let mut group = c.benchmark_group("petsc_compare/sa_amg");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(60));

    for &target_dofs in &[1_024, 10_000, 100_000, 1_000_000] {
        let (matrix, rhs) = setup(target_dofs);
        let nn = matrix.num_rows;
        let cfg = gmres_config();
        let amg_cfg = AmgConfig::for_energy_minimizing();

        group.throughput(Throughput::Elements(nn as u64));

        #[cfg(feature = "cli")]
        if perf::perf_json_path().is_some() {
            let (amg, setup_ms) =
                perf::time_once(|| AmgPreconditioner::from_csr(&matrix, amg_cfg.clone()));
            let (solution, solve_ms) =
                perf::time_once(|| gmres_preconditioned(&matrix, &amg, &rhs, &cfg));
            let amg_diag = amg.diagnostics();
            record_solve_perf(SolvePerfMeta {
                section: "sa_amg",
                variant: "setup+solve",
                preconditioner: "sa_amg",
                n: nn,
                nnz: matrix.nnz(),
                setup_ms,
                solve_ms,
                iterations: solution.iterations,
                residual: solution.residual,
                converged: solution.converged,
                krylov_op_counts: solution.op_counts,
                amg_profile: Some("energy_minimizing_sa"),
                amg_diagnostics: Some(amg_diag),
                notes: "setup and solve timed separately",
            });
        }

        // Setup + solve together
        record(
            "sa_amg",
            "setup+solve",
            nn,
            time_ms(|| {
                let amg = AmgPreconditioner::from_csr(&matrix, amg_cfg.clone());
                gmres_preconditioned(&matrix, &amg, &rhs, &cfg)
            }),
        );

        group.bench_function(BenchmarkId::new("setup+solve", nn), |b| {
            b.iter(|| {
                let amg = AmgPreconditioner::from_csr(&matrix, amg_cfg.clone());
                gmres_preconditioned(&matrix, &amg, &rhs, &cfg)
            })
        });

        // Solve only
        let amg = AmgPreconditioner::from_csr(&matrix, amg_cfg);
        #[cfg(feature = "cli")]
        if perf::perf_json_path().is_some() {
            let (solution, solve_ms) =
                perf::time_once(|| gmres_preconditioned(&matrix, &amg, &rhs, &cfg));
            let amg_diag = amg.diagnostics();
            record_solve_perf(SolvePerfMeta {
                section: "sa_amg",
                variant: "solve_only",
                preconditioner: "sa_amg",
                n: nn,
                nnz: matrix.nnz(),
                setup_ms: 0.0,
                solve_ms,
                iterations: solution.iterations,
                residual: solution.residual,
                converged: solution.converged,
                krylov_op_counts: solution.op_counts,
                amg_profile: Some("energy_minimizing_sa"),
                amg_diagnostics: Some(amg_diag),
                notes: "preconditioner setup excluded",
            });
        }

        record(
            "sa_amg",
            "solve_only",
            nn,
            time_ms(|| gmres_preconditioned(&matrix, &amg, &rhs, &cfg)),
        );

        group.bench_function(BenchmarkId::new("solve_only", nn), |b| {
            b.iter(|| gmres_preconditioned(&matrix, &amg, &rhs, &cfg))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// SOTA-style AMG profiles: hypre BoomerAMG-like and Trilinos MueLu-like
// ---------------------------------------------------------------------------

type AmgProfile = (&'static str, fn() -> AmgConfig);

fn bench_sota_amg_profiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("petsc_compare/sota_amg_profiles");
    group.sample_size(5);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(30));

    let profiles: &[AmgProfile] = &[
        (
            "hypre_boomeramg_like",
            AmgConfig::for_hypre_boomeramg_comparison,
        ),
        ("muelu_sa_like", AmgConfig::for_muelu_sa_comparison),
    ];

    for &target_dofs in &[10_000, 100_000] {
        let (matrix, rhs) = setup(target_dofs);
        let nn = matrix.num_rows;
        let cfg = gmres_config();
        group.throughput(Throughput::Elements(nn as u64));

        for &(profile_name, make_config) in profiles {
            #[cfg(feature = "cli")]
            if perf::perf_json_path().is_some() {
                let (amg, setup_ms) =
                    perf::time_once(|| AmgPreconditioner::from_csr(&matrix, make_config()));
                let (solution, solve_ms) =
                    perf::time_once(|| gmres_preconditioned(&matrix, &amg, &rhs, &cfg));
                let amg_diag = amg.diagnostics();
                record_solve_perf(SolvePerfMeta {
                    section: "sota_amg_profiles",
                    variant: profile_name,
                    preconditioner: "amg",
                    n: nn,
                    nnz: matrix.nnz(),
                    setup_ms,
                    solve_ms,
                    iterations: solution.iterations,
                    residual: solution.residual,
                    converged: solution.converged,
                    krylov_op_counts: solution.op_counts,
                    amg_profile: Some(profile_name),
                    amg_diagnostics: Some(amg_diag),
                    notes: "setup and solve timed separately; profile is solver-side configuration for comparison against external hypre/MueLu runs",
                });
            }

            record(
                "sota_amg_profiles",
                profile_name,
                nn,
                time_ms(|| {
                    let amg = AmgPreconditioner::from_csr(&matrix, make_config());
                    gmres_preconditioned(&matrix, &amg, &rhs, &cfg)
                }),
            );

            group.bench_function(BenchmarkId::new(profile_name, nn), |b| {
                b.iter(|| {
                    let amg = AmgPreconditioner::from_csr(&matrix, make_config());
                    gmres_preconditioned(&matrix, &amg, &rhs, &cfg)
                })
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Summary table
// ---------------------------------------------------------------------------

fn print_solver_table(_c: &mut Criterion) {
    let rows = SOLVER_TABLE.lock().unwrap();

    println!("\n{}", "=".repeat(74));
    println!(" Sonium Solver Summary  (compare against PETSc: see benches/ffi/)");
    println!("{}\n", "=".repeat(74));

    // Sections in display order, with the variants to show as columns
    let sections: &[(&str, &[&str])] = &[
        ("baseline", &["none", "jacobi"]),
        ("ilu", &["setup+solve", "solve_only"]),
        ("sa_amg", &["setup+solve", "solve_only"]),
        (
            "sota_amg_profiles",
            &["hypre_boomeramg_like", "muelu_sa_like"],
        ),
    ];

    for &(section, variants) in sections {
        let section_rows: Vec<&SolverRow> = rows.iter().filter(|r| r.section == section).collect();
        if section_rows.is_empty() {
            continue;
        }

        // Unique DOF sizes in insertion order
        let mut sizes: Vec<usize> = Vec::new();
        for r in &section_rows {
            if !sizes.contains(&r.n) {
                sizes.push(r.n);
            }
        }

        // Header
        let col_w = 16usize;
        let title = section.replace('_', "-").to_uppercase();
        println!("  {}  (GMRES + {})", title, section.replace('_', "-"));
        let header_variants: String = variants
            .iter()
            .map(|v| format!("{:>col_w$}", v))
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {:>8}  {}", "DOFs", header_variants);
        println!("  {}", "-".repeat(8 + 2 + variants.len() * (col_w + 2)));

        for &n in &sizes {
            let cells: String = variants
                .iter()
                .map(|&v| {
                    let ms = section_rows
                        .iter()
                        .find(|r| r.n == n && r.variant == v)
                        .map(|r| r.ms);
                    match ms {
                        Some(t) if t >= 1_000.0 => format!("{:>col_w$.2} s ", t / 1_000.0),
                        Some(t) if t >= 1.0 => format!("{:>col_w$.1} ms", t),
                        Some(t) => format!("{:>col_w$.2} ms", t),
                        None => format!("{:>col_w$}", "—"),
                    }
                })
                .collect::<Vec<_>>()
                .join("  ");
            println!("  {:>8}  {}", fmt_n(n), cells);
        }
        println!();
    }

    println!("{}", "=".repeat(74));
}

criterion_group!(
    name = petsc_compare;
    config = Criterion::default();
    targets = bench_baseline, bench_ilu, bench_sa_amg, bench_sota_amg_profiles, print_solver_table
);
criterion_main!(petsc_compare);
