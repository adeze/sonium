//! Integration tests for iterative solvers
//!
//! Tests GMRES, BiCGSTAB, CGS, CG, and IDR(s) against known problems.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::{
    BiCgstabConfig, CgConfig, CgsConfig, CsrBuilder, CsrMatrix, GmresConfig, IdrsConfig,
    IluPreconditioner, SolverStatus, bicgstab, cg, gmres, gmres_preconditioned, idrs,
};
use std::time::Instant;

fn tridiagonal_real(n: usize, diag: f64, off_diag: f64) -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(n, n);
    for i in 0..n {
        let mut entries = vec![(i, diag)];
        if i > 0 {
            entries.push((i - 1, off_diag));
        }
        if i < n - 1 {
            entries.push((i + 1, off_diag));
        }
        builder.add_row_entries(entries.into_iter());
    }
    builder.finish()
}

fn tridiagonal_complex(n: usize, diag: Complex64, off_diag: Complex64) -> CsrMatrix<Complex64> {
    let mut builder = CsrBuilder::new(n, n);
    for i in 0..n {
        let mut entries = vec![(i, diag)];
        if i > 0 {
            entries.push((i - 1, off_diag));
        }
        if i < n - 1 {
            entries.push((i + 1, off_diag));
        }
        builder.add_row_entries(entries.into_iter());
    }
    builder.finish()
}

fn test_gmres_convergence_tridiagonal() {
    let n = 50;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    let config = GmresConfig {
        max_iterations: 200,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &b, &config);

    assert!(
        solution.converged,
        "GMRES should converge for tridiagonal system"
    );
    assert!(solution.residual < 1e-8, "Residual should be small");

    let ax = matrix.matvec(&solution.x);
    let error: f64 = ax.iter().zip(b.iter()).map(|(a, bi)| (a - bi).abs()).sum();
    assert!(error < 1e-8, "Solution error: {}", error);
}

fn test_gmres_with_ilu_preconditioner() {
    let n = 30;
    let matrix = tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0));
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.0))
        .collect();

    let precond = IluPreconditioner::from_csr(&matrix);

    let config = GmresConfig {
        max_iterations: 100,
        restart: 30,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres_preconditioned(&matrix, &precond, &b, &config);

    assert!(solution.converged, "Preconditioned GMRES should converge");
    assert!(
        solution.iterations < 50,
        "ILU should reduce iterations significantly"
    );
}

fn test_bicgstab_convergence() {
    let n = 40;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).cos()).collect();

    let config = BiCgstabConfig {
        max_iterations: 200,
        tolerance: 1e-10,
        print_interval: 0,
    };

    let solution = bicgstab(&matrix, &b, &config);

    assert!(
        solution.converged,
        "BiCGSTAB should converge for this system"
    );
    assert!(solution.residual < 1e-8, "Residual: {}", solution.residual);
}

fn test_cg_convergence_spd() {
    let n = 50;
    let matrix = tridiagonal_real(n, 2.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();

    let config = CgConfig {
        max_iterations: 200,
        tolerance: 1e-10,
        print_interval: 0,
    };

    let solution = cg(&matrix, &b, &config);

    assert!(solution.converged, "CG should converge for SPD tridiagonal");
    assert!(solution.residual < 1e-8, "Residual: {}", solution.residual);
}

/// Retains compatibility coverage for the deprecated public CGS entry point.
#[allow(deprecated)]
fn test_cgs_convergence() {
    let n = 30;
    let matrix = tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0));
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.0))
        .collect();

    let config = CgsConfig {
        max_iterations: 200,
        tolerance: 1e-10,
        print_interval: 0,
    };

    let solution = sonium_solvers::cgs(&matrix, &b, &config);

    assert!(solution.converged, "CGS should converge");
    assert!(solution.residual < 1e-8, "Residual: {}", solution.residual);
}

fn test_idrs_convergence() {
    let n = 40;
    let matrix = tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0));
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).cos(), 0.0))
        .collect();

    let config = IdrsConfig {
        max_iterations: 200,
        tolerance: 1e-10,
        print_interval: 0,
        s: 4,
    };

    let solution = idrs(&matrix, &b, &config);

    assert!(solution.converged, "IDR(s) should converge");
    assert!(solution.residual < 1e-8, "Residual: {}", solution.residual);
}

fn test_gmres_complex() {
    let n = 30;
    let diag = Complex64::new(4.0, 1.0);
    let off_diag = Complex64::new(-1.0, 0.5);
    let matrix = tridiagonal_complex(n, diag, off_diag);
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), (i as f64 + 1.0).cos()))
        .collect();

    let config = GmresConfig {
        max_iterations: 200,
        restart: 30,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &b, &config);

    assert!(
        solution.converged,
        "GMRES should converge for complex system"
    );
    assert!(solution.residual < 1e-8, "Residual: {}", solution.residual);

    let ax = matrix.matvec(&solution.x);
    let error: f64 = ax.iter().zip(b.iter()).map(|(a, bi)| (a - bi).norm()).sum();
    assert!(error < 1e-7, "Solution error: {}", error);
}

fn test_gmres_residual_history() {
    let n = 20;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    let config = GmresConfig {
        max_iterations: 100,
        restart: 5,
        tolerance: 1e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &b, &config);

    assert!(solution.converged, "GMRES should converge");
    assert!(
        solution.restarts > 0,
        "Small restart should force multiple restarts"
    );
}

fn test_status_contracts_across_family() {
    let n = 12;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    let gmres_solution = gmres(
        &matrix,
        &rhs,
        &GmresConfig {
            max_iterations: 100,
            restart: n,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 1,
        },
    );
    assert!(gmres_solution.converged);
    assert!(matches!(gmres_solution.status, SolverStatus::Converged));

    let bicgstab = bicgstab(
        &matrix,
        &rhs,
        &BiCgstabConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            print_interval: 0,
        },
    );
    assert!(bicgstab.converged);
    assert!(matches!(bicgstab.status, SolverStatus::Converged));

    let cg = cg(
        &tridiagonal_real(n, 2.0, -1.0),
        &rhs,
        &CgConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            print_interval: 0,
        },
    );
    assert!(cg.converged);
    assert!(matches!(cg.status, SolverStatus::Converged));

    #[allow(deprecated)]
    let cgs = sonium_solvers::cgs(
        &tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0)),
        &rhs.iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>(),
        &CgsConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            print_interval: 0,
        },
    );
    assert!(cgs.converged);
    assert!(matches!(cgs.status, SolverStatus::Converged));

    let idrs = idrs(
        &tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0)),
        &rhs.iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>(),
        &IdrsConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            print_interval: 0,
            s: 4,
        },
    );
    assert!(idrs.converged);
    assert!(matches!(idrs.status, SolverStatus::Converged));

    let bounded = gmres(
        &matrix,
        &rhs,
        &GmresConfig {
            max_iterations: 1,
            restart: 1,
            tolerance: 1e-30,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 1,
        },
    );
    assert!(!bounded.converged);
    assert!(!matches!(bounded.status, SolverStatus::Converged));
}

fn test_bicgstab_vs_gmres_same_solution() {
    let n = 30;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).cos()).collect();

    let gmres_config = GmresConfig {
        max_iterations: 200,
        restart: 30,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };
    let bicgstab_config = BiCgstabConfig {
        max_iterations: 200,
        tolerance: 1e-10,
        print_interval: 0,
    };

    let gmres_sol = gmres(&matrix, &b, &gmres_config);
    let bicgstab_sol = bicgstab(&matrix, &b, &bicgstab_config);

    assert!(gmres_sol.converged);
    assert!(bicgstab_sol.converged);

    let diff: f64 = gmres_sol
        .x
        .iter()
        .zip(bicgstab_sol.x.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff < 1e-6,
        "GMRES and BiCGSTAB should give similar solutions"
    );
}

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(|| {
        test_gmres_convergence_tridiagonal();
        test_gmres_with_ilu_preconditioner();
        test_bicgstab_convergence();
        test_cg_convergence_spd();
        test_cgs_convergence();
        test_idrs_convergence();
        test_gmres_complex();
        test_gmres_residual_history();
        test_status_contracts_across_family();
        test_bicgstab_vs_gmres_same_solution();
    });
    match result {
        Ok(()) => {
            let matrix = tridiagonal_real(20, 4.0, -1.0);
            let rhs: Vec<f64> = (0..20).map(|i| (i as f64 + 1.0).sin()).collect();
            let solve = gmres(
                &matrix,
                &rhs,
                &GmresConfig {
                    max_iterations: 100,
                    restart: 20,
                    tolerance: 1.0e-12,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    min_restart: None,
                    max_restart: None,
                    exact_residual_check_interval: 1,
                },
            );
            let record = QaResult::passed("solvers.iterative-contracts.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("gmres_true_residual", solve.residual)
                .metric("gmres_iterations", solve.iterations as f64)
                .metric("status_contract_count", 6.0)
                .maximum("gmres_true_residual", 1.0e-8)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(_) => {
            if let Ok(json) = QaResult::failed(
                "solvers.iterative-contracts.v1",
                "one or more iterative solver contract fixtures panicked",
            )
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64)
            .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
