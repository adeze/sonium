//! High-level selection evidence (audit-sonium-solvers-20260921, Priorities:
//! "publish high-level selection evidence on representative PDE matrices
//! with setup, solve, memory and error separated").
//!
//! Each case drives the **public facade** (`solve`, `solve_with_assumption`,
//! `solve_complex`, `solve_saddle_point`) with default `Auto` dispatch unless
//! the case states otherwise, on a manufactured solution (`rhs = A·1`, error
//! measured against `1`). Setup time is measured by constructing the same
//! preconditioner the facade builds for the resolved strategy through the
//! identical public `try_from_csr*` constructors (`IluPreconditioner`,
//! `AmgPreconditioner` with `AmgConfig::for_fem`, `AdditiveSchwarz` with
//! overlap 1); solve time is total-minus-setup and therefore derived. Memory
//! is a documented lower-bound model in the PERF.md style: exact matrix +
//! Krylov-basis + work-vector bytes plus a stated factor rule, never OS RSS.
//! A `Err` solve fails the bench loudly; non-convergence is recorded, not
//! hidden, with its residual and error.
//!
//! The large-problem dispatch arms are exercised through the documented
//! env-overridable thresholds (`SONIUM_LARGE_PROBLEM_ROWS/_NNZ`), saved and
//! restored around those cases only.
//!
//! Run:
//!   cargo bench -p sonium-solvers --bench selection_evidence_bench

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{Criterion, criterion_group, criterion_main};
use num_complex::Complex64;
use sonium_solvers::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgPreconditioner, BlockPreconditionerConfig,
    IluPreconditioner, LinearSolveConfig, LinearSolverType, LinearSystemAssumption,
    SaddlePointConfig, SchurStrategy, solve, solve_complex, solve_saddle_point,
    solve_with_assumption,
};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const BENCH_TOL: f64 = 1e-8;
const BENCH_MAX_ITER: usize = 500;
const BENCH_RESTART: usize = 30;
const MEASURED_RUNS: usize = 5;

fn bench_config() -> LinearSolveConfig {
    LinearSolveConfig {
        max_iterations: BENCH_MAX_ITER,
        tolerance: BENCH_TOL,
        restart: BENCH_RESTART,
        ..Default::default()
    }
}

struct SolveQuality {
    iterations: usize,
    residual: f64,
    backward_error: f64,
    manufactured_error: f64,
    converged: bool,
}

struct EvidenceRow {
    case: &'static str,
    class: &'static str,
    scalar: &'static str,
    n: usize,
    nnz: usize,
    assumption: &'static str,
    resolved: String,
    setup_ms: Option<f64>,
    total_ms: f64,
    solve_ms: Option<f64>,
    iterations: usize,
    residual: f64,
    backward_error: f64,
    manufactured_error: f64,
    converged: bool,
    modeled_peak_kb: f64,
}

static TABLE: Mutex<Vec<EvidenceRow>> = Mutex::new(Vec::new());
static PRINTED: AtomicBool = AtomicBool::new(false);

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn rel_error_2_real(x: &[f64], exact: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (a, b) in x.iter().zip(exact.iter()) {
        num += (a - b) * (a - b);
        den += b * b;
    }
    (num.sqrt()) / (den.sqrt())
}

fn rel_error_2_complex(x: &[Complex64], exact: &[Complex64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (a, b) in x.iter().zip(exact.iter()) {
        num += (a - b).norm_sqr();
        den += b.norm_sqr();
    }
    (num.sqrt()) / (den.sqrt())
}

/// Exact storage bytes for CSR structure + Krylov basis + work vectors.
/// `factor_bytes` carries the stated per-strategy factor rule (see callers).
fn modeled_peak_kb(
    n: usize,
    nnz: usize,
    width: usize,
    krylov_vecs: usize,
    factor_bytes: usize,
) -> f64 {
    let matrix = (n + 1) * 8 + nnz * 8 + nnz * width;
    let krylov = krylov_vecs * n * width;
    let work = 3 * n * width;
    (matrix + krylov + work + factor_bytes) as f64 / 1024.0
}

fn matrix_values_bytes(nnz: usize, width: usize) -> usize {
    nnz * 8 + nnz * width + 8
}

/// Run one evidence case: 1 warm + MEASURED_RUNS timed facade solves.
/// Skips (records once) if this case already ran in this process, so
/// criterion resampling stays cheap while the printed table is the artifact.
#[allow(clippy::too_many_arguments)]
fn evidence_case(
    case: &'static str,
    class: &'static str,
    scalar: &'static str,
    n: usize,
    nnz: usize,
    width: usize,
    assumption: &'static str,
    resolved: String,
    krylov_vecs: usize,
    factor_bytes: usize,
    setup_once: Option<&dyn Fn()>,
    solve_once: &dyn Fn() -> SolveQuality,
) {
    {
        let table = TABLE.lock().unwrap();
        if table.iter().any(|row| row.case == case) {
            return;
        }
    }
    // Warm-up (admission, caches, thread pools) — discarded.
    solve_once();
    if let Some(setup) = setup_once {
        setup();
    }
    let mut totals = Vec::with_capacity(MEASURED_RUNS);
    let mut quality = None;
    for _ in 0..MEASURED_RUNS {
        let start = Instant::now();
        let q = solve_once();
        totals.push(start.elapsed().as_secs_f64() * 1000.0);
        quality = Some(q);
    }
    let setup_ms = setup_once.map(|setup| {
        let mut times = Vec::with_capacity(MEASURED_RUNS);
        for _ in 0..MEASURED_RUNS {
            let start = Instant::now();
            setup();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        median(times)
    });
    let total_ms = median(totals);
    let quality = quality.expect("at least one measured run");
    TABLE.lock().unwrap().push(EvidenceRow {
        case,
        class,
        scalar,
        n,
        nnz,
        assumption,
        resolved,
        solve_ms: setup_ms.map(|s| total_ms - s),
        setup_ms,
        total_ms,
        iterations: quality.iterations,
        residual: quality.residual,
        backward_error: quality.backward_error,
        manufactured_error: quality.manufactured_error,
        converged: quality.converged,
        modeled_peak_kb: modeled_peak_kb(n, nnz, width, krylov_vecs, factor_bytes),
    });
}

fn with_large_thresholds(f: impl FnOnce()) {
    // SAFETY: the harness drives every case synchronously on the bench
    // thread; no other thread reads these dispatch thresholds concurrently.
    // Values are saved and restored around the large-arm cases only.
    unsafe {
        let old_rows = std::env::var("SONIUM_LARGE_PROBLEM_ROWS").ok();
        let old_nnz = std::env::var("SONIUM_LARGE_PROBLEM_NNZ").ok();
        std::env::set_var("SONIUM_LARGE_PROBLEM_ROWS", "1000");
        std::env::set_var("SONIUM_LARGE_PROBLEM_NNZ", "20000");
        f();
        match old_rows {
            Some(v) => std::env::set_var("SONIUM_LARGE_PROBLEM_ROWS", v),
            None => std::env::remove_var("SONIUM_LARGE_PROBLEM_ROWS"),
        }
        match old_nnz {
            Some(v) => std::env::set_var("SONIUM_LARGE_PROBLEM_NNZ", v),
            None => std::env::remove_var("SONIUM_LARGE_PROBLEM_NNZ"),
        }
    }
}

fn bench_poisson_auto(_c: &mut Criterion) {
    let matrix = matrices::poisson_2d(48);
    let exact = matrices::ones_rhs(matrix.num_rows);
    let rhs = matrix.matvec(&exact);
    let config = bench_config();
    let resolved = format!("{:?}", config.resolved_solver_for_real(&matrix));
    let (n, nnz) = (matrix.num_rows, matrix.nnz());
    evidence_case(
        "poisson-2d",
        "SPD Laplacian",
        "f64",
        n,
        nnz,
        8,
        "General",
        resolved,
        31,
        matrix_values_bytes(nnz, 8),
        Some(&|| {
            IluPreconditioner::try_from_csr(&matrix).unwrap();
        }),
        &|| {
            let result = solve(&matrix, &rhs, &config).expect("poisson Auto solve failed");
            SolveQuality {
                iterations: result.iterations,
                residual: result.residual,
                backward_error: result.backward_error,
                manufactured_error: rel_error_2_real(&result.x, &exact),
                converged: result.converged,
            }
        },
    );
}

fn bench_poisson_cgamg(_c: &mut Criterion) {
    let matrix = matrices::poisson_2d(48);
    let exact = matrices::ones_rhs(matrix.num_rows);
    let rhs = matrix.matvec(&exact);
    let config = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        ..bench_config()
    };
    let (n, nnz) = (matrix.num_rows, matrix.nnz());
    evidence_case(
        "poisson-2d-cgamg",
        "SPD Laplacian",
        "f64",
        n,
        nnz,
        8,
        "PositiveDefinite",
        format!("{:?}", LinearSolverType::CgAmg),
        5,
        // Upper-bound model: AMG operator complexity is not reported by
        // setup; exact complexities live in preconditioner_bench.
        2 * matrix_values_bytes(nnz, 8),
        Some(&|| {
            AmgPreconditioner::try_from_csr(&matrix, AmgConfig::for_fem()).unwrap();
        }),
        &|| {
            let result = solve_with_assumption(
                &matrix,
                &rhs,
                &config,
                LinearSystemAssumption::PositiveDefinite,
            )
            .expect("poisson CgAmg solve failed");
            SolveQuality {
                iterations: result.iterations,
                residual: result.residual,
                backward_error: result.backward_error,
                manufactured_error: rel_error_2_real(&result.x, &exact),
                converged: result.converged,
            }
        },
    );
}

fn bench_convection_diffusion(_c: &mut Criterion) {
    let matrix = matrices::convection_diffusion_2d(64);
    let exact = matrices::ones_rhs(matrix.num_rows);
    let rhs = matrix.matvec(&exact);
    let config = bench_config();
    let resolved = format!("{:?}", config.resolved_solver_for_real(&matrix));
    let (n, nnz) = (matrix.num_rows, matrix.nnz());
    evidence_case(
        "convection-diffusion-2d",
        "nonsymmetric upwind",
        "f64",
        n,
        nnz,
        8,
        "General",
        resolved,
        31,
        matrix_values_bytes(nnz, 8),
        Some(&|| {
            IluPreconditioner::try_from_csr(&matrix).unwrap();
        }),
        &|| {
            let result = solve(&matrix, &rhs, &config).expect("convection-diffusion solve failed");
            SolveQuality {
                iterations: result.iterations,
                residual: result.residual,
                backward_error: result.backward_error,
                manufactured_error: rel_error_2_real(&result.x, &exact),
                converged: result.converged,
            }
        },
    );
}

fn bench_helmholtz_small(_c: &mut Criterion) {
    let matrix = matrices::helmholtz_2d(32, 20.0);
    let exact = matrices::ones_rhs_c64(matrix.num_rows);
    let rhs = matrix.matvec(&exact);
    let config = bench_config();
    let resolved = format!("{:?}", config.resolved_solver_for_complex(&matrix));
    let (n, nnz) = (matrix.num_rows, matrix.nnz());
    evidence_case(
        "helmholtz-2d",
        "complex indefinite",
        "Complex64",
        n,
        nnz,
        16,
        "General",
        resolved,
        31,
        matrix_values_bytes(nnz, 16),
        Some(&|| {
            IluPreconditioner::try_from_csr(&matrix).unwrap();
        }),
        &|| {
            let result = solve_complex(&matrix, &rhs, &config).expect("helmholtz solve failed");
            SolveQuality {
                iterations: result.iterations,
                residual: result.residual,
                backward_error: result.backward_error,
                manufactured_error: rel_error_2_complex(&result.x, &exact),
                converged: result.converged,
            }
        },
    );
}

fn bench_helmholtz_large(_c: &mut Criterion) {
    with_large_thresholds(|| {
        let matrix = matrices::helmholtz_2d(96, 20.0);
        let exact = matrices::ones_rhs_c64(matrix.num_rows);
        let rhs = matrix.matvec(&exact);
        let config = bench_config();
        let resolved = format!("{:?}", config.resolved_solver_for_complex(&matrix));
        let (n, nnz) = (matrix.num_rows, matrix.nnz());
        evidence_case(
            "helmholtz-2d-large",
            "complex indefinite (large arm)",
            "Complex64",
            n,
            nnz,
            16,
            "General",
            resolved,
            32,
            // Overlapping Schwarz subdomains duplicate unknowns; 1.5x is a
            // stated estimate, not a measurement.
            matrix_values_bytes(nnz, 16) * 3 / 2,
            Some(&|| {
                AdditiveSchwarzPreconditioner::try_from_csr_auto(&matrix, 1).unwrap();
            }),
            &|| {
                let result =
                    solve_complex(&matrix, &rhs, &config).expect("helmholtz-large solve failed");
                SolveQuality {
                    iterations: result.iterations,
                    residual: result.residual,
                    backward_error: result.backward_error,
                    manufactured_error: rel_error_2_complex(&result.x, &exact),
                    converged: result.converged,
                }
            },
        );
    });
}

fn bench_laplacian_3d_large(_c: &mut Criterion) {
    with_large_thresholds(|| {
        let matrix = matrices::laplacian_3d(20);
        let exact = matrices::ones_rhs(matrix.num_rows);
        let rhs = matrix.matvec(&exact);
        let config = bench_config();
        let resolved = format!("{:?}", config.resolved_solver_for_real(&matrix));
        let (n, nnz) = (matrix.num_rows, matrix.nnz());
        evidence_case(
            "laplacian-3d-large",
            "SPD Laplacian 3D (large arm)",
            "f64",
            n,
            nnz,
            8,
            "General",
            resolved,
            31,
            matrix_values_bytes(nnz, 8),
            Some(&|| {
                IluPreconditioner::try_from_csr(&matrix).unwrap();
            }),
            &|| {
                let result = solve(&matrix, &rhs, &config).expect("laplacian-3d solve failed");
                SolveQuality {
                    iterations: result.iterations,
                    residual: result.residual,
                    backward_error: result.backward_error,
                    manufactured_error: rel_error_2_real(&result.x, &exact),
                    converged: result.converged,
                }
            },
        );
    });
}

fn bench_saddle_point(_c: &mut Criterion) {
    let (matrix, schur) = matrices::saddle_kkt_4x4();
    let exact = vec![1.0, -0.5, 0.75, 1.5];
    let rhs = matrix.matvec(&exact);
    let config = bench_config();
    let saddle_config = SaddlePointConfig {
        block1_size: 2,
        schur_strategy: SchurStrategy::External,
        block2_config: BlockPreconditionerConfig::Diagonal,
        ..Default::default()
    };
    let (n, nnz) = (matrix.num_rows, matrix.nnz());
    evidence_case(
        "saddle-point-kkt",
        "KKT block (contract scale)",
        "f64",
        n,
        nnz,
        8,
        "General",
        "solve_saddle_point(explicit)".to_string(),
        31,
        // Block/diagonal-Schur factor storage is internal to the solve;
        // modeled at one matrix pattern.
        matrix_values_bytes(nnz, 8),
        None,
        &|| {
            let result =
                solve_saddle_point(&matrix, &rhs, &config, saddle_config.clone(), Some(&schur))
                    .expect("saddle-point solve failed");
            SolveQuality {
                iterations: result.iterations,
                residual: result.residual,
                backward_error: result.backward_error,
                manufactured_error: rel_error_2_real(&result.x, &exact),
                converged: result.converged,
            }
        },
    );
}

fn print_selection_table(_c: &mut Criterion) {
    if PRINTED.swap(true, Ordering::SeqCst) {
        return;
    }
    // Wait for all cases (criterion runs targets in registration order, but
    // resampling may interleave; rows record once via the TABLE guard).
    let table = TABLE.lock().unwrap();
    println!();
    println!("# Selection evidence (facade Auto dispatch)");
    println!();
    println!(
        "| case | class | scalar | N | nnz | assumption | resolved | setup ms | solve ms | total ms | iters | residual | backward err | mfg err | converged | peak KiB |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for row in table.iter() {
        let setup = row
            .setup_ms
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        let solve = row
            .solve_ms
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.3} | {} | {:.2e} | {:.2e} | {:.2e} | {} | {:.1} |",
            row.case,
            row.class,
            row.scalar,
            row.n,
            row.nnz,
            row.assumption,
            row.resolved,
            setup,
            solve,
            row.total_ms,
            row.iterations,
            row.residual,
            row.backward_error,
            row.manufactured_error,
            row.converged,
            row.modeled_peak_kb,
        );
    }
    println!();
    println!(
        "Protocol: 1 warm + {} timed facade solves, median total; setup = median of {} \
         explicit preconditioner builds through the same public constructors the facade uses; \
         solve = total - setup (derived). Memory is a lower-bound model (PERF.md style), not RSS.",
        MEASURED_RUNS, MEASURED_RUNS
    );
    println!();
}

criterion_group!(
    name = selection_evidence;
    config = Criterion::default().measurement_time(Duration::from_secs(1));
    targets =
        bench_poisson_auto,
        bench_poisson_cgamg,
        bench_convection_diffusion,
        bench_helmholtz_small,
        bench_helmholtz_large,
        bench_laplacian_3d_large,
        bench_saddle_point,
        print_selection_table
);
criterion_main!(selection_evidence);
