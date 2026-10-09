//! Benchmark runner: executes all solver/preconditioner/matrix combinations.

use std::path::Path;
use std::time::Instant;

use sonium_solvers::perf;
use sonium_solvers::*;

use crate::combinations::*;
use crate::matrices;
use crate::results::RunResult;

/// Run a single benchmark case (one solver/precond/matrix/scalar/size combo).
pub fn run_case(
    case: &BenchmarkCase,
    target_dofs: usize,
    verbose: bool,
    rhs_count: usize,
) -> Option<RunResult> {
    match case.scalar {
        ScalarKind::F64 => run_real(case, target_dofs, verbose, rhs_count),
        ScalarKind::Complex64 => run_complex(case, target_dofs, verbose, rhs_count),
    }
}

fn run_real(
    case: &BenchmarkCase,
    target_dofs: usize,
    verbose: bool,
    rhs_count: usize,
) -> Option<RunResult> {
    let (matrix, actual_n) = match case.matrix {
        MatrixKind::Poisson2D => matrices::poisson_2d(target_dofs),
        MatrixKind::AnisotropicDiffusion2D => matrices::anisotropic_diffusion_2d(target_dofs),
        MatrixKind::SaddlePoint2D => matrices::saddle_point_2d(target_dofs),
        MatrixKind::Laplacian3D => matrices::laplacian_3d(target_dofs),
        MatrixKind::Helmholtz2D => return None, // complex only
    };

    let n = matrix.num_rows;
    let nnz = matrix.nnz();
    let rhs = matrices::ones_rhs_real(n);
    let rhs_count = rhs_count.max(1);

    if verbose {
        eprintln!(
            "  Running {}/{}/{} f64 n={n} rhs={rhs_count} (grid={actual_n})...",
            case.solver, case.preconditioner, case.matrix,
        );
    }

    let config_gmres = GmresConfig {
        max_iterations: 500,
        restart: 30,
        tolerance: 1e-6,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };
    let config_cg = CgConfig {
        max_iterations: 500,
        tolerance: 1e-6,
        print_interval: 0,
    };
    let config_bicgstab = BiCgstabConfig {
        max_iterations: 500,
        tolerance: 1e-6,
        print_interval: 0,
    };
    let config_idrs = IdrsConfig {
        max_iterations: 500,
        tolerance: 1e-6,
        s: 4,
        print_interval: 0,
    };

    // Build preconditioner
    let t_setup = Instant::now();

    let solve_stats = match case.preconditioner {
        PrecondKind::None => {
            let setup_ms = 0.0;
            let stats = repeat_solves(rhs_count, || match case.solver {
                SolverKind::Gmres => {
                    let s = gmres(&matrix, &rhs, &config_gmres);
                    (s.iterations, s.residual, s.converged, Some(s.op_counts))
                }
                SolverKind::BiCgstab => {
                    let s = bicgstab(&matrix, &rhs, &config_bicgstab);
                    (s.iterations, s.residual, s.converged, Some(s.op_counts))
                }
                SolverKind::Cg => {
                    let s = cg(&matrix, &rhs, &config_cg);
                    (s.iterations, s.residual, s.converged, Some(s.op_counts))
                }
                SolverKind::Idrs => {
                    let s = idrs(&matrix, &rhs, &config_idrs);
                    (s.iterations, s.residual, s.converged, Some(s.op_counts))
                }
                SolverKind::Fgmres => {
                    let s = fgmres(
                        &matrix,
                        &rhs,
                        &FgmresConfig {
                            max_iterations: 500,
                            restart: 30,
                            tolerance: 1e-6,
                            print_interval: 0,
                            convergence_check_interval: 1,
                            exact_residual_check_interval: 0,
                        },
                    );
                    (s.iterations, s.residual, s.converged, Some(s.op_counts))
                }
            });
            stats.with_setup_ms(setup_ms)
        }
        PrecondKind::Jacobi => {
            let p = DiagonalPreconditioner::from_csr(&matrix);
            solve_with_precond_real(&matrix, &rhs, &p, case, t_setup, rhs_count)
        }
        PrecondKind::Ilu => {
            let p = IluPreconditioner::from_csr(&matrix);
            solve_with_precond_real(&matrix, &rhs, &p, case, t_setup, rhs_count)
        }
        PrecondKind::Ilut => {
            let p = IlutPreconditioner::from_csr_with_config(&matrix, &IlutConfig::default());
            solve_with_precond_real(&matrix, &rhs, &p, case, t_setup, rhs_count)
        }
        PrecondKind::Amg => {
            let p = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
            solve_with_precond_real(&matrix, &rhs, &p, case, t_setup, rhs_count)
        }
    };

    Some(with_solver_perf_fields(
        RunResult {
            engine: "sonium".into(),
            solver: case.solver.to_string(),
            preconditioner: case.preconditioner.to_string(),
            matrix: case.matrix.to_string(),
            scalar: "f64".into(),
            n,
            nnz,
            rhs_count,
            setup_ms: solve_stats.setup_ms,
            solve_ms: solve_stats.solve_ms,
            total_ms: solve_stats.setup_ms + solve_stats.solve_ms,
            iterations: solve_stats.iterations,
            residual: solve_stats.residual,
            converged: solve_stats.converged,
            bytes_per_spmv: None,
            bytes_per_nnz: None,
            estimated_spmv_gb_s: None,
            setup_fraction: None,
            time_limited_by: None,
            mean_solve_ms_per_rhs: None,
            mean_iterations_per_rhs: None,
            setup_reuse_scope: None,
            setup_reuse_break_even_rhs: None,
            amortized_total_ms_per_rhs_4: None,
            amortized_total_ms_per_rhs_16: None,
            krylov_spmv_count: solve_stats.op_counts.map(|counts| counts.spmv_count),
            krylov_preconditioner_count: solve_stats
                .op_counts
                .map(|counts| counts.preconditioner_count),
            krylov_dot_reductions: solve_stats.op_counts.map(|counts| counts.dot_reductions),
            krylov_norm_count: solve_stats.op_counts.map(|counts| counts.norm_count),
            krylov_axpy_passes: solve_stats.op_counts.map(|counts| counts.axpy_passes),
            krylov_basis_bytes_touched: solve_stats
                .op_counts
                .map(|counts| counts.basis_bytes_touched),
            krylov_synchronization_points: solve_stats
                .op_counts
                .map(|counts| counts.synchronization_points),
            krylov_exact_residual_checks: solve_stats
                .op_counts
                .map(|counts| counts.exact_residual_checks),
        },
        std::mem::size_of::<f64>(),
    ))
}

fn with_solver_perf_fields(mut run: RunResult, scalar_bytes: usize) -> RunResult {
    let model = perf::csr_spmv_byte_model(
        run.n,
        run.n,
        run.nnz,
        scalar_bytes,
        std::mem::size_of::<usize>(),
    );
    run.bytes_per_spmv = Some(model.total_bytes);
    run.bytes_per_nnz = Some(model.bytes_per_nnz);
    run.estimated_spmv_gb_s = if run.solve_ms > 0.0 && run.iterations > 0 {
        Some((model.total_bytes as f64 * run.iterations as f64) / run.solve_ms / 1.0e6)
    } else {
        None
    };
    let rhs_count = run.rhs_count.max(1);
    run.rhs_count = rhs_count;
    run.mean_solve_ms_per_rhs = Some(run.solve_ms / rhs_count as f64);
    run.mean_iterations_per_rhs = Some(run.iterations as f64 / rhs_count as f64);
    run.setup_fraction = if run.total_ms > 0.0 {
        Some(run.setup_ms / run.total_ms)
    } else {
        None
    };
    run.time_limited_by = Some(classify_time_limit(&run));
    run.setup_reuse_scope = Some(classify_setup_reuse_scope(&run));
    run.setup_reuse_break_even_rhs = setup_reuse_break_even_rhs(&run);
    run.amortized_total_ms_per_rhs_4 = Some(amortized_total_ms_per_rhs(&run, 4));
    run.amortized_total_ms_per_rhs_16 = Some(amortized_total_ms_per_rhs(&run, 16));
    run
}

fn classify_time_limit(run: &RunResult) -> String {
    if !run.converged {
        return "convergence".to_string();
    }
    if run.setup_fraction.unwrap_or(0.0) >= 0.5 {
        return "setup".to_string();
    }
    if run.iterations > 0 {
        return "solve_spmv_lower_bound".to_string();
    }
    "solve".to_string()
}

fn classify_setup_reuse_scope(run: &RunResult) -> String {
    if run.setup_ms <= 0.0 {
        return "no_setup".to_string();
    }
    match run.preconditioner.as_str() {
        "ilu" | "ilut" | "amg" => "same_matrix_repeated_rhs".to_string(),
        "jacobi" => "cheap_same_matrix_repeated_rhs".to_string(),
        _ => "setup_not_reused".to_string(),
    }
}

fn setup_reuse_break_even_rhs(run: &RunResult) -> Option<f64> {
    let solve_ms_per_rhs = run.mean_solve_ms_per_rhs.unwrap_or(run.solve_ms);
    if run.setup_ms <= 0.0 || solve_ms_per_rhs <= 0.0 {
        return None;
    }
    Some(run.setup_ms / solve_ms_per_rhs)
}

fn amortized_total_ms_per_rhs(run: &RunResult, rhs_count: usize) -> f64 {
    let solve_ms_per_rhs = run.mean_solve_ms_per_rhs.unwrap_or(run.solve_ms);
    solve_ms_per_rhs + run.setup_ms / rhs_count.max(1) as f64
}

#[derive(Debug, Clone, Copy)]
struct RepeatedSolveStats {
    setup_ms: f64,
    solve_ms: f64,
    iterations: usize,
    residual: f64,
    converged: bool,
    op_counts: Option<KrylovOpCounts>,
}

impl RepeatedSolveStats {
    fn with_setup_ms(mut self, setup_ms: f64) -> Self {
        self.setup_ms = setup_ms;
        self
    }
}

fn repeat_solves<F>(rhs_count: usize, mut solve_once: F) -> RepeatedSolveStats
where
    F: FnMut() -> (usize, f64, bool, Option<KrylovOpCounts>),
{
    let t_solve = Instant::now();
    let mut iterations = 0usize;
    let mut residual = 0.0_f64;
    let mut converged = true;
    let mut op_counts = KrylovOpCounts::default();
    let mut saw_op_counts = false;

    for _ in 0..rhs_count.max(1) {
        let (it, res, ok, counts) = solve_once();
        iterations += it;
        residual = residual.max(res);
        converged &= ok;
        if let Some(counts) = counts {
            saw_op_counts = true;
            add_krylov_counts(&mut op_counts, counts);
        }
    }

    RepeatedSolveStats {
        setup_ms: 0.0,
        solve_ms: t_solve.elapsed().as_secs_f64() * 1000.0,
        iterations,
        residual,
        converged,
        op_counts: saw_op_counts.then_some(op_counts),
    }
}

fn add_krylov_counts(total: &mut KrylovOpCounts, counts: KrylovOpCounts) {
    total.spmv_count += counts.spmv_count;
    total.preconditioner_count += counts.preconditioner_count;
    total.dot_reductions += counts.dot_reductions;
    total.norm_count += counts.norm_count;
    total.axpy_passes += counts.axpy_passes;
    total.basis_bytes_touched += counts.basis_bytes_touched;
    total.synchronization_points += counts.synchronization_points;
    total.exact_residual_checks += counts.exact_residual_checks;
}

fn solve_with_precond_real<P: Preconditioner<f64>>(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    precond: &P,
    case: &BenchmarkCase,
    t_setup: Instant,
    rhs_count: usize,
) -> RepeatedSolveStats {
    let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;

    repeat_solves(rhs_count, || match case.solver {
        SolverKind::Gmres => {
            let config = GmresConfig {
                max_iterations: 500,
                restart: 30,
                tolerance: 1e-6,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                ..Default::default()
            };
            let s = gmres_preconditioned(matrix, precond, rhs, &config);
            (s.iterations, s.residual, s.converged, Some(s.op_counts))
        }
        SolverKind::Fgmres => {
            let config = FgmresConfig {
                max_iterations: 500,
                restart: 30,
                tolerance: 1e-6,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
            };
            let s = fgmres_preconditioned(matrix, precond, rhs, None, &config);
            (s.iterations, s.residual, s.converged, Some(s.op_counts))
        }
        SolverKind::Cg => {
            let config = CgConfig {
                max_iterations: 500,
                tolerance: 1e-6,
                print_interval: 0,
            };
            let s = pcg(matrix, precond, rhs, &config);
            (s.iterations, s.residual, s.converged, Some(s.op_counts))
        }
        SolverKind::Idrs => {
            let config = IdrsConfig {
                max_iterations: 500,
                tolerance: 1e-6,
                s: 4,
                print_interval: 0,
            };
            let s = idrs_preconditioned(matrix, precond, rhs, &config);
            (s.iterations, s.residual, s.converged, Some(s.op_counts))
        }
        SolverKind::BiCgstab => {
            // BiCGSTAB doesn't have a preconditioned variant — run unpreconditioned
            let config = BiCgstabConfig {
                max_iterations: 500,
                tolerance: 1e-6,
                print_interval: 0,
            };
            let s = bicgstab(matrix, rhs, &config);
            (s.iterations, s.residual, s.converged, Some(s.op_counts))
        }
    })
    .with_setup_ms(setup_ms)
}

fn run_complex(
    case: &BenchmarkCase,
    target_dofs: usize,
    verbose: bool,
    rhs_count: usize,
) -> Option<RunResult> {
    let (matrix, actual_n) = match case.matrix {
        MatrixKind::Helmholtz2D => matrices::helmholtz_2d(target_dofs, 5.0),
        MatrixKind::Poisson2D
        | MatrixKind::AnisotropicDiffusion2D
        | MatrixKind::SaddlePoint2D
        | MatrixKind::Laplacian3D => return None,
    };

    let n = matrix.num_rows;
    let nnz = matrix.nnz();
    let rhs = matrices::ones_rhs_complex(n);
    let rhs_count = rhs_count.max(1);

    if verbose {
        eprintln!(
            "  Running {}/{}/{} complex64 n={n} rhs={rhs_count} (grid={actual_n})...",
            case.solver, case.preconditioner, case.matrix,
        );
    }

    let config_gmres = GmresConfig {
        max_iterations: 500,
        restart: 30,
        tolerance: 1e-6,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };
    let config_bicgstab = BiCgstabConfig {
        max_iterations: 500,
        tolerance: 1e-6,
        print_interval: 0,
    };

    // For complex, only unpreconditioned or Jacobi
    let t_setup = Instant::now();

    let solve_stats = match case.preconditioner {
        PrecondKind::None => match case.solver {
            SolverKind::Gmres => repeat_solves(rhs_count, || {
                let s = gmres(&matrix, &rhs, &config_gmres);
                (s.iterations, s.residual, s.converged, Some(s.op_counts))
            }),
            SolverKind::BiCgstab => repeat_solves(rhs_count, || {
                let s = bicgstab(&matrix, &rhs, &config_bicgstab);
                (s.iterations, s.residual, s.converged, Some(s.op_counts))
            }),
            _ => return None,
        },
        PrecondKind::Jacobi => {
            let p = DiagonalPreconditioner::from_csr(&matrix);
            let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;
            match case.solver {
                SolverKind::Gmres => repeat_solves(rhs_count, || {
                    let s = gmres_preconditioned(&matrix, &p, &rhs, &config_gmres);
                    (s.iterations, s.residual, s.converged, Some(s.op_counts))
                })
                .with_setup_ms(setup_ms),
                _ => return None,
            }
        }
        PrecondKind::Ilu => {
            let p = IluPreconditioner::from_csr(&matrix);
            let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;
            repeat_solves(rhs_count, || {
                let s = gmres_preconditioned(&matrix, &p, &rhs, &config_gmres);
                (s.iterations, s.residual, s.converged, Some(s.op_counts))
            })
            .with_setup_ms(setup_ms)
        }
        _ => return None, // AMG/ILUT not supported for complex
    };

    Some(with_solver_perf_fields(
        RunResult {
            engine: "sonium".into(),
            solver: case.solver.to_string(),
            preconditioner: case.preconditioner.to_string(),
            matrix: case.matrix.to_string(),
            scalar: "complex64".into(),
            n,
            nnz,
            rhs_count,
            setup_ms: solve_stats.setup_ms,
            solve_ms: solve_stats.solve_ms,
            total_ms: solve_stats.setup_ms + solve_stats.solve_ms,
            iterations: solve_stats.iterations,
            residual: solve_stats.residual,
            converged: solve_stats.converged,
            bytes_per_spmv: None,
            bytes_per_nnz: None,
            estimated_spmv_gb_s: None,
            setup_fraction: None,
            time_limited_by: None,
            mean_solve_ms_per_rhs: None,
            mean_iterations_per_rhs: None,
            setup_reuse_scope: None,
            setup_reuse_break_even_rhs: None,
            amortized_total_ms_per_rhs_4: None,
            amortized_total_ms_per_rhs_16: None,
            krylov_spmv_count: solve_stats.op_counts.map(|counts| counts.spmv_count),
            krylov_preconditioner_count: solve_stats
                .op_counts
                .map(|counts| counts.preconditioner_count),
            krylov_dot_reductions: solve_stats.op_counts.map(|counts| counts.dot_reductions),
            krylov_norm_count: solve_stats.op_counts.map(|counts| counts.norm_count),
            krylov_axpy_passes: solve_stats.op_counts.map(|counts| counts.axpy_passes),
            krylov_basis_bytes_touched: solve_stats
                .op_counts
                .map(|counts| counts.basis_bytes_touched),
            krylov_synchronization_points: solve_stats
                .op_counts
                .map(|counts| counts.synchronization_points),
            krylov_exact_residual_checks: solve_stats
                .op_counts
                .map(|counts| counts.exact_residual_checks),
        },
        std::mem::size_of::<num_complex::Complex64>(),
    ))
}

/// Write the matrix for this case to a temp file for PETSc (f64 only).
pub fn write_case_matrix(
    case: &BenchmarkCase,
    target_dofs: usize,
    dir: &Path,
) -> Option<std::path::PathBuf> {
    if case.scalar != ScalarKind::F64 {
        return None; // PETSc comparison is f64 only for now
    }

    let (matrix, _) = match case.matrix {
        MatrixKind::Poisson2D => matrices::poisson_2d(target_dofs),
        MatrixKind::AnisotropicDiffusion2D => matrices::anisotropic_diffusion_2d(target_dofs),
        MatrixKind::SaddlePoint2D => matrices::saddle_point_2d(target_dofs),
        MatrixKind::Laplacian3D => matrices::laplacian_3d(target_dofs),
        MatrixKind::Helmholtz2D => return None,
    };

    let path = dir.join(format!(
        "{}_{}_n{}.mtx",
        case.matrix, case.scalar, target_dofs,
    ));
    matrices::write_matrix_market_real(&matrix, &path);
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_run(setup_ms: f64, solve_ms: f64, iterations: usize, converged: bool) -> RunResult {
        RunResult {
            engine: "sonium".into(),
            solver: "gmres".into(),
            preconditioner: "none".into(),
            matrix: "poisson_2d".into(),
            scalar: "f64".into(),
            n: 10,
            nnz: 50,
            rhs_count: 1,
            setup_ms,
            solve_ms,
            total_ms: setup_ms + solve_ms,
            iterations,
            residual: if converged { 1e-8 } else { 1e-2 },
            converged,
            bytes_per_spmv: None,
            bytes_per_nnz: None,
            estimated_spmv_gb_s: None,
            setup_fraction: None,
            time_limited_by: None,
            mean_solve_ms_per_rhs: None,
            mean_iterations_per_rhs: None,
            setup_reuse_scope: None,
            setup_reuse_break_even_rhs: None,
            amortized_total_ms_per_rhs_4: None,
            amortized_total_ms_per_rhs_16: None,
            krylov_spmv_count: None,
            krylov_preconditioner_count: None,
            krylov_dot_reductions: None,
            krylov_norm_count: None,
            krylov_axpy_passes: None,
            krylov_basis_bytes_touched: None,
            krylov_synchronization_points: None,
            krylov_exact_residual_checks: None,
        }
    }

    #[test]
    fn run_result_perf_fields_include_csr_byte_model() {
        let run =
            with_solver_perf_fields(sample_run(2.0, 8.0, 5, true), std::mem::size_of::<f64>());

        assert!(run.bytes_per_spmv.unwrap() > 0);
        assert!(run.bytes_per_nnz.unwrap() > 0.0);
        assert!(run.estimated_spmv_gb_s.unwrap() > 0.0);
        assert_eq!(run.setup_fraction, Some(0.2));
        assert_eq!(
            run.time_limited_by.as_deref(),
            Some("solve_spmv_lower_bound")
        );
        assert_eq!(run.setup_reuse_scope.as_deref(), Some("setup_not_reused"));
        assert_eq!(run.setup_reuse_break_even_rhs, Some(0.25));
        assert_eq!(run.amortized_total_ms_per_rhs_4, Some(8.5));
        assert_eq!(run.amortized_total_ms_per_rhs_16, Some(8.125));
    }

    #[test]
    fn run_result_perf_fields_classify_setup_and_convergence_limits() {
        let setup_limited =
            with_solver_perf_fields(sample_run(10.0, 2.0, 5, true), std::mem::size_of::<f64>());
        let unconverged =
            with_solver_perf_fields(sample_run(1.0, 2.0, 5, false), std::mem::size_of::<f64>());

        assert_eq!(setup_limited.time_limited_by.as_deref(), Some("setup"));
        assert_eq!(unconverged.time_limited_by.as_deref(), Some("convergence"));
    }

    #[test]
    fn run_result_perf_fields_expose_setup_reuse_economics() {
        let mut run = sample_run(12.0, 4.0, 5, true);
        run.preconditioner = "amg".into();
        let run = with_solver_perf_fields(run, std::mem::size_of::<f64>());

        assert_eq!(
            run.setup_reuse_scope.as_deref(),
            Some("same_matrix_repeated_rhs")
        );
        assert_eq!(run.setup_reuse_break_even_rhs, Some(3.0));
        assert_eq!(run.amortized_total_ms_per_rhs_4, Some(7.0));
        assert_eq!(run.amortized_total_ms_per_rhs_16, Some(4.75));
    }

    #[test]
    fn run_result_perf_fields_average_repeated_rhs_solves() {
        let mut run = sample_run(12.0, 20.0, 40, true);
        run.preconditioner = "amg".into();
        run.rhs_count = 4;
        let run = with_solver_perf_fields(run, std::mem::size_of::<f64>());

        assert_eq!(run.mean_solve_ms_per_rhs, Some(5.0));
        assert_eq!(run.mean_iterations_per_rhs, Some(10.0));
        assert_eq!(run.setup_reuse_break_even_rhs, Some(2.4));
        assert_eq!(run.amortized_total_ms_per_rhs_4, Some(8.0));
    }

    #[test]
    fn repeat_solves_aggregates_krylov_counts() {
        let mut calls = 0usize;
        let stats = repeat_solves(2, || {
            calls += 1;
            let counts = KrylovOpCounts {
                spmv_count: calls,
                preconditioner_count: 2,
                dot_reductions: 3,
                norm_count: 4,
                axpy_passes: 5,
                basis_bytes_touched: 6,
                synchronization_points: 7,
                exact_residual_checks: 8,
            };
            (10, 1e-8, true, Some(counts))
        });
        let counts = stats.op_counts.unwrap();

        assert_eq!(stats.iterations, 20);
        assert_eq!(counts.spmv_count, 3);
        assert_eq!(counts.preconditioner_count, 4);
        assert_eq!(counts.dot_reductions, 6);
        assert_eq!(counts.norm_count, 8);
        assert_eq!(counts.axpy_passes, 10);
        assert_eq!(counts.basis_bytes_touched, 12);
        assert_eq!(counts.synchronization_points, 14);
        assert_eq!(counts.exact_residual_checks, 16);
    }
}
