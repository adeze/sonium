//! sonium-benchmark: exhaustive QA benchmark suite for sonium-solvers.
//!
//! Runs all solver/preconditioner/matrix/scalar combinations, optionally
//! compares with PETSc, validates correctness, and generates a results website.

mod combinations;
mod matrices;
mod matrix_market;
mod petsc;
mod platform;
mod regression;
mod results;
mod runner;
mod site;
mod validation;

use std::path::PathBuf;

use clap::Parser;

use combinations::{all_cases, filter_cases};
use matrix_market::{standard_matrices, standard_rhs};
use platform::detect_platform;
use regression::RegressionGateConfig;
use results::{BenchmarkReport, ValidationSummary};
use validation::validate;

#[derive(Parser)]
#[command(
    name = "sonium-benchmark",
    about = "SOTA QA benchmark suite for sonium-solvers"
)]
struct Cli {
    /// Path to petsc_reference binary for comparison
    #[arg(long)]
    petsc: Option<String>,

    /// Run Matrix Market accuracy benchmark (downloads real matrices, compares vs PETSc)
    #[arg(long)]
    matrix_market: bool,

    /// Use a local matrix file instead of downloading (MTX format, gzipped or plain)
    #[arg(long)]
    local_matrix: Option<PathBuf>,

    /// Matrix Market matrix to benchmark (e.g., "bcsstk15"). If omitted, runs all.
    #[arg(long)]
    mtx_matrix: Option<String>,

    /// Solver to use for Matrix Market benchmarks (default: gmres)
    #[arg(long, default_value = "gmres")]
    mtx_solver: String,

    /// Preconditioner for Matrix Market benchmarks (default: ilu)
    #[arg(long, default_value = "ilu")]
    mtx_precond: String,

    /// Write JSON results to file (default: stdout)
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Platform name override (default: auto-detected)
    #[arg(long)]
    platform_name: Option<String>,

    /// Comma-separated target DOF sizes (default: 1000,10000,100000)
    #[arg(long, default_value = "1000,10000,100000")]
    sizes: String,

    /// Filter cases by pattern: "solver/precond/matrix"
    #[arg(long)]
    filter: Option<String>,

    /// Generate static website from JSON files in a directory
    #[arg(long)]
    generate_site: Option<PathBuf>,

    /// Output directory for generated site
    #[arg(long, default_value = "docs/benchmarks")]
    output_dir: PathBuf,

    /// Print progress to stderr
    #[arg(long, short)]
    verbose: bool,

    /// Only run smallest size for quick validation
    #[arg(long)]
    quick: bool,

    /// Number of RHS solves to run after a single matrix/preconditioner setup
    #[arg(long, default_value_t = 1)]
    rhs_count: usize,

    /// Compare results against a previous benchmark JSON and fail on regressions
    #[arg(long)]
    regression_baseline: Option<PathBuf>,

    /// Maximum allowed total-time slowdown vs baseline, as a fraction
    #[arg(long, default_value_t = 0.10)]
    max_total_slowdown: f64,

    /// Maximum allowed solve-time slowdown vs baseline, as a fraction
    #[arg(long, default_value_t = 0.10)]
    max_solve_slowdown: f64,

    /// Maximum allowed iteration-count increase vs baseline, as a fraction
    #[arg(long, default_value_t = 0.10)]
    max_iteration_regression: f64,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Website generation mode
    if let Some(ref json_dir) = cli.generate_site {
        if cli.verbose {
            eprintln!(
                "Generating site from {:?} -> {:?}",
                json_dir, cli.output_dir
            );
        }
        site::generate_site(json_dir, &cli.output_dir);
        eprintln!("Site generated at {:?}", cli.output_dir);
        return;
    }

    // Matrix Market accuracy benchmark mode
    if cli.matrix_market {
        run_matrix_market_benchmark(&cli);
        return;
    }

    // Benchmark mode
    let platform = detect_platform(cli.platform_name.as_deref());
    if cli.verbose {
        eprintln!(
            "Platform: {} ({} {})",
            platform.name, platform.os, platform.arch
        );
        eprintln!(
            "CPU: {} ({} cores)",
            platform.cpu_model, platform.total_cores
        );
    }

    // Parse sizes
    let sizes: Vec<usize> = if cli.quick {
        vec![1000]
    } else {
        cli.sizes
            .split(',')
            .map(|s| s.trim().parse::<usize>().expect("Invalid size"))
            .collect()
    };

    // Get and filter cases
    let mut cases = all_cases();
    if let Some(ref filter) = cli.filter {
        cases = filter_cases(&cases, filter);
    }

    // Override sizes, but cap GMRES+none/jacobi at 10k DOF (too slow above)
    for case in &mut cases {
        if case.solver == combinations::SolverKind::Gmres
            && matches!(
                case.preconditioner,
                combinations::PrecondKind::None | combinations::PrecondKind::Jacobi
            )
        {
            case.sizes = sizes.iter().copied().filter(|&s| s <= 10_000).collect();
        } else {
            case.sizes = sizes.clone();
        }
    }
    // Drop cases that ended up with no sizes after filtering
    cases.retain(|c| !c.sizes.is_empty());

    if cli.verbose {
        let total_runs: usize = cases.iter().map(|c| c.sizes.len()).sum();
        eprintln!(
            "Running {total_runs} benchmark cases ({} combos × {} sizes)...",
            cases.len(),
            sizes.len()
        );
    }

    // Temp dir for MTX files
    let tmp = std::env::temp_dir().join("sonium_benchmark");
    std::fs::create_dir_all(&tmp).unwrap();

    let mut all_runs = Vec::new();
    let rhs_count = cli.rhs_count.max(1);

    // Print table header
    eprintln!(
        "\n{:<10} {:<10} {:<8} {:<14} {:<8} {:>8} {:>5} {:>10} {:>10} {:>10} {:>8} {:>9} {:>12} {:>5} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:<18} {:>9} {:>9}",
        "engine",
        "solver",
        "precond",
        "matrix",
        "scalar",
        "n",
        "rhs",
        "setup_ms",
        "solve_ms",
        "total_ms",
        "iters",
        "solve/rhs",
        "residual",
        "ok",
        "B/nnz",
        "SpMV GB/s",
        "k-spmv",
        "k-dots",
        "k-axpy",
        "setup%",
        "limit",
        "rhs@4",
        "rhs@16"
    );
    eprintln!("{}", "-".repeat(240));

    for case in &cases {
        for &size in &case.sizes {
            // Run sonium
            if let Some(result) = runner::run_case(case, size, cli.verbose, rhs_count) {
                print_row(&result);
                all_runs.push(result);
            }

            // Run PETSc (f64 only)
            if rhs_count == 1
                && let Some(ref petsc_bin) = cli.petsc
                && let Some(mtx_path) = runner::write_case_matrix(case, size, &tmp)
            {
                if let Some(result) = petsc::run_petsc(
                    petsc_bin,
                    &mtx_path,
                    &case.solver.to_string(),
                    &case.preconditioner.to_string(),
                    1e-6,
                    500,
                    &case.matrix.to_string(),
                    &case.scalar.to_string(),
                ) {
                    print_row(&result);
                    all_runs.push(result);
                }
                let _ = std::fs::remove_file(&mtx_path);
            }
        }
    }

    // Validate
    let validation = if cli.petsc.is_some() {
        validate(&all_runs)
    } else {
        ValidationSummary {
            total_comparisons: 0,
            passed: 0,
            failed: 0,
            warnings: 0,
            failures: Vec::new(),
        }
    };

    if cli.verbose && cli.petsc.is_some() {
        eprintln!(
            "\nValidation: {} passed, {} failed, {} warnings out of {} comparisons",
            validation.passed, validation.failed, validation.warnings, validation.total_comparisons
        );
        for f in &validation.failures {
            eprintln!(
                "  FAIL: {}/{}/{} n={}: {}",
                f.solver, f.preconditioner, f.matrix, f.n, f.reason
            );
        }
    }

    // Build report
    let report = BenchmarkReport {
        schema_version: 1,
        timestamp: chrono::Utc::now().to_rfc3339(),
        platform,
        sonium_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: platform::detect_git_sha(),
        petsc_available: cli.petsc.is_some(),
        runs: all_runs,
        validation,
    };

    let regression = cli.regression_baseline.as_ref().map(|baseline| {
        let config = RegressionGateConfig {
            max_total_slowdown: cli.max_total_slowdown,
            max_solve_slowdown: cli.max_solve_slowdown,
            max_iteration_regression: cli.max_iteration_regression,
        };
        regression::check_report_against_path(&report, baseline, config)
            .unwrap_or_else(|err| panic!("failed to check regression baseline: {err}"))
    });

    if let Some(summary) = &regression {
        eprintln!(
            "\nRegression gates: {} passed, {} failed, {} warnings out of {} comparisons",
            summary.passed, summary.failed, summary.warnings, summary.total_comparisons
        );
        for warning in &summary.warning_messages {
            eprintln!("  WARN: {warning}");
        }
        for failure in &summary.failures {
            eprintln!("  FAIL: {}", failure.reason);
        }
    }

    // Output JSON
    let json = serde_json::to_string_pretty(&report).unwrap();
    if let Some(ref path) = cli.output {
        std::fs::write(path, &json).unwrap();
        eprintln!("\nResults written to {}", path.display());
    } else {
        println!("{json}");
    }

    // Clean up
    let _ = std::fs::remove_dir(&tmp);

    // Exit with error if validation failed
    if report.validation.failed > 0
        || regression
            .as_ref()
            .is_some_and(|summary| summary.failed > 0)
    {
        std::process::exit(1);
    }
}

fn print_row(r: &results::RunResult) {
    eprintln!(
        "{:<10} {:<10} {:<8} {:<14} {:<8} {:>8} {:>5} {:>10.2} {:>10.2} {:>10.2} {:>8} {:>9} {:>12.2e} {:>5} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:<18} {:>9} {:>9}",
        r.engine,
        r.solver,
        r.preconditioner,
        r.matrix,
        r.scalar,
        r.n,
        r.rhs_count,
        r.setup_ms,
        r.solve_ms,
        r.total_ms,
        r.iterations,
        format_optional_f64(r.mean_solve_ms_per_rhs, 2),
        r.residual,
        r.converged,
        format_optional_f64(r.bytes_per_nnz, 1),
        format_optional_f64(r.estimated_spmv_gb_s, 1),
        format_optional_usize(r.krylov_spmv_count),
        format_optional_usize(r.krylov_dot_reductions),
        format_optional_usize(r.krylov_axpy_passes),
        format_optional_percent(r.setup_fraction),
        r.time_limited_by.as_deref().unwrap_or("-"),
        format_optional_f64(r.amortized_total_ms_per_rhs_4, 2),
        format_optional_f64(r.amortized_total_ms_per_rhs_16, 2),
    );
}

fn format_optional_usize(value: Option<usize>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "-".to_string(),
    }
}

fn format_optional_f64(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(v) => format!("{v:.precision$}"),
        None => "-".to_string(),
    }
}

fn format_optional_percent(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{:.0}%", v * 100.0),
        None => "-".to_string(),
    }
}

/// Run the Matrix Market accuracy benchmark: download real matrices,
/// solve with sonium and PETSc, compare solution accuracy.
fn run_matrix_market_benchmark(cli: &Cli) {
    use matrix_market::{MatrixMarketId, fetch_and_parse, parse_matrix_market};
    use sonium_solvers::{
        AmgConfig, AmgPreconditioner, DiagonalPreconditioner, GmresConfig, IluPreconditioner,
        gmres_preconditioned,
    };
    use std::fs::File;
    use std::io::BufReader;

    let platform = detect_platform(cli.platform_name.as_deref());

    // Local matrix mode: load from file and run single benchmark
    if let Some(ref local_path) = cli.local_matrix {
        let file = File::open(local_path).expect("Failed to open local matrix file");
        let reader: Box<dyn std::io::BufRead> = if local_path.extension().is_some_and(|e| e == "gz")
        {
            let decoder = flate2::read::GzDecoder::new(file);
            Box::new(BufReader::new(decoder))
        } else {
            Box::new(BufReader::new(file))
        };
        let (matrix, nnz) = parse_matrix_market(reader).expect("Failed to parse local matrix");
        let n = matrix.num_rows;
        let rhs = standard_rhs(n);

        eprintln!(
            "\nLocal Matrix Benchmark — {}x{}, {} nnz, solver={}, precond={}",
            n, n, nnz, cli.mtx_solver, cli.mtx_precond
        );
        eprintln!(
            "Platform: {} ({} {})",
            platform.name, platform.os, platform.arch
        );
        eprintln!("{}", "-".repeat(80));

        let config = GmresConfig {
            max_iterations: 500,
            restart: 30,
            tolerance: 1e-6,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let (sonium_converged, sonium_residual, sonium_iters) = match cli.mtx_precond.as_str() {
            "none" => {
                let r = sonium_solvers::gmres(&matrix, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            "jacobi" => {
                let precond = DiagonalPreconditioner::from_csr(&matrix);
                let r = gmres_preconditioned(&matrix, &precond, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            "ilu" => {
                let precond = IluPreconditioner::from_csr(&matrix);
                let r = gmres_preconditioned(&matrix, &precond, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            "amg" => {
                let precond = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
                let r = gmres_preconditioned(&matrix, &precond, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            _ => {
                eprintln!("Unknown preconditioner '{}'", cli.mtx_precond);
                return;
            }
        };

        eprintln!(
            "sonium: converged={}, residual={:.2e}, iterations={}",
            sonium_converged, sonium_residual, sonium_iters
        );

        // Run PETSc if available
        if let Some(ref petsc_bin) = cli.petsc {
            let tmp_path = std::env::temp_dir().join("local_matrix.mtx");
            matrices::write_matrix_market_real(&matrix, &tmp_path);
            if let Some(result) = petsc::run_petsc(
                petsc_bin,
                &tmp_path,
                &cli.mtx_solver,
                &cli.mtx_precond,
                1e-6,
                500,
                "local_matrix",
                "f64",
            ) {
                eprintln!(
                    "petsc:  converged={}, residual={:.2e}, iterations={}",
                    result.converged, result.residual, result.iterations
                );
            }
            let _ = std::fs::remove_file(&tmp_path);
        }

        return;
    }

    // Filter matrices if one was specified
    let matrices: Vec<MatrixMarketId> = if let Some(ref name) = cli.mtx_matrix {
        standard_matrices()
            .into_iter()
            .filter(|m| m.name == *name || m.display_name() == *name)
            .collect()
    } else {
        standard_matrices()
    };

    if matrices.is_empty() {
        eprintln!("No matching Matrix Market matrices found.");
        if cli.mtx_matrix.is_some() {
            eprintln!("Available matrices:");
            for m in standard_matrices() {
                eprintln!(
                    "  {} ({}) - {}",
                    m.display_name(),
                    m.expected_size,
                    m.description
                );
            }
        }
        return;
    }

    eprintln!(
        "\nMatrix Market Accuracy Benchmark — {} matrices, solver={}, precond={}",
        matrices.len(),
        cli.mtx_solver,
        cli.mtx_precond
    );
    eprintln!(
        "Platform: {} ({} {})",
        platform.name, platform.os, platform.arch
    );
    eprintln!("{}", "-".repeat(100));

    // Print header for accuracy table
    eprintln!(
        "{:<22} {:>8} {:>10} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "matrix", "n", "nnz", "sonium_res", "petsc_res", "iters_s", "iters_p", "ok"
    );
    eprintln!("{}", "-".repeat(100));

    let mut total_sonum_iter = 0;
    let mut total_petsc_iter = 0;
    let mut compared = 0usize;
    let mut failed = 0usize;

    let config = GmresConfig {
        max_iterations: 500,
        restart: 30,
        tolerance: 1e-6,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };

    for mtx_id in &matrices {
        // Fetch and parse
        let (matrix, nnz) = match fetch_and_parse(mtx_id) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "{:<22} ERROR: failed to fetch/parse: {}",
                    mtx_id.display_name(),
                    e
                );
                continue;
            }
        };

        let n = matrix.num_rows;
        let rhs = standard_rhs(n);

        if cli.verbose {
            eprintln!(
                "\nMatrix {} ({}x{}, {} nnz): {}",
                mtx_id.display_name(),
                n,
                n,
                nnz,
                mtx_id.description
            );
        }

        // Run sonium with the selected preconditioner
        let (sonium_converged, sonium_residual, sonium_iters) = match cli.mtx_precond.as_str() {
            "none" => {
                let r = sonium_solvers::gmres(&matrix, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            "jacobi" => {
                let precond = DiagonalPreconditioner::from_csr(&matrix);
                let r = gmres_preconditioned(&matrix, &precond, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            "ilu" => {
                let precond = IluPreconditioner::from_csr(&matrix);
                let r = gmres_preconditioned(&matrix, &precond, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            "amg" => {
                let precond = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
                let r = gmres_preconditioned(&matrix, &precond, &rhs, &config);
                (r.converged, r.residual, r.iterations)
            }
            _ => {
                eprintln!("Unknown preconditioner '{}', skipping", cli.mtx_precond);
                continue;
            }
        };
        total_sonum_iter += sonium_iters;

        // Run PETSc if available (write matrix to temp file first)
        let (petsc_residual, petsc_iters) = if let Some(ref petsc_bin) = cli.petsc {
            let tmp_path = std::env::temp_dir().join(format!("{}.mtx", mtx_id.name));
            matrices::write_matrix_market_real(&matrix, &tmp_path);
            let result = petsc::run_petsc(
                petsc_bin,
                &tmp_path,
                &cli.mtx_solver,
                &cli.mtx_precond,
                1e-6,
                500,
                mtx_id.name,
                "f64",
            );
            let _ = std::fs::remove_file(&tmp_path);
            if let Some(petsc_result) = result {
                (petsc_result.residual, petsc_result.iterations)
            } else {
                (f64::NAN, 0)
            }
        } else {
            (f64::NAN, 0)
        };

        if petsc_iters > 0 {
            total_petsc_iter += petsc_iters;
            compared += 1;
        }

        // Check accuracy: sonium residual should be similar to PETSc
        let ok = if petsc_iters > 0 {
            sonium_converged
                && sonium_residual / petsc_residual < 100.0
                && petsc_residual / sonium_residual < 100.0
        } else {
            sonium_converged
        };

        if !ok {
            failed += 1;
        }

        eprintln!(
            "{:<22} {:>8} {:>10} {:>12.2e} {:>12.2e} {:>10} {:>10} {:>5}",
            mtx_id.display_name(),
            n,
            nnz,
            sonium_residual,
            if petsc_iters > 0 {
                petsc_residual
            } else {
                f64::NAN
            },
            sonium_iters,
            if petsc_iters > 0 { petsc_iters } else { 0 },
            if ok { "OK" } else { "FAIL" }
        );
    }

    eprintln!("{}", "-".repeat(100));
    eprintln!(
        "Summary: {} matrices, {} compared vs PETSc, {} failed",
        matrices.len(),
        compared,
        failed
    );
    if compared > 0 {
        let avg_ratio = total_sonum_iter as f64 / total_petsc_iter as f64;
        eprintln!("Avg iteration ratio (sonium/PETSc): {:.2}x", avg_ratio);
    }

    // Write JSON output
    let report = BenchmarkReport {
        schema_version: 2, // Bump schema for matrix market results
        timestamp: chrono::Utc::now().to_rfc3339(),
        platform,
        sonium_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: platform::detect_git_sha(),
        petsc_available: cli.petsc.is_some(),
        runs: vec![], // Matrix market results use different format
        validation: ValidationSummary {
            total_comparisons: compared,
            passed: compared - failed,
            failed,
            warnings: 0,
            failures: vec![],
        },
    };

    let json = serde_json::to_string_pretty(&report).unwrap();
    if let Some(ref path) = cli.output {
        std::fs::write(path, &json).unwrap();
        eprintln!("\nResults written to {}", path.display());
    } else {
        println!("{json}");
    }

    if failed > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod main_tests {
    use super::*;

    #[test]
    fn formats_optional_cli_metrics() {
        assert_eq!(format_optional_f64(Some(24.25), 1), "24.2");
        assert_eq!(format_optional_f64(None, 1), "-");
        assert_eq!(format_optional_percent(Some(0.125)), "12%");
        assert_eq!(format_optional_percent(None), "-");
        assert_eq!(format_optional_usize(Some(42)), "42");
        assert_eq!(format_optional_usize(None), "-");
    }
}
