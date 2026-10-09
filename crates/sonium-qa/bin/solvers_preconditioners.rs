//! Integration tests for preconditioners
//!
//! Tests ILU, ILUT, Diagonal, Block Jacobi preconditioners.

use num_complex::Complex64;
use sonium_solvers::{
    BlockJacobiPreconditioner, CsrBuilder, CsrMatrix, DiagonalPreconditioner, GmresConfig,
    IluPreconditioner, IlutConfig, IlutPreconditioner,
};

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

fn test_ilu_preconditioner_reduces_iterations() {
    let n = 50;
    let matrix = tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0));
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.0))
        .collect();

    let precond = IluPreconditioner::from_csr(&matrix);

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

    let solution = sonium_solvers::gmres_preconditioned(&matrix, &precond, &b, &config);

    assert!(solution.converged, "Preconditioned GMRES should converge");
    assert!(
        solution.iterations < 50,
        "ILU should significantly reduce iterations"
    );
}

fn test_ilut_preconditioner() {
    let n = 40;
    let matrix = tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0));
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.0))
        .collect();

    let config = IlutConfig::new(1e-4, 10, 1e-12);
    let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

    let gmres_config = GmresConfig {
        max_iterations: 200,
        restart: 40,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = sonium_solvers::gmres_preconditioned(&matrix, &precond, &b, &gmres_config);

    assert!(
        solution.converged,
        "ILUT preconditioned GMRES should converge"
    );
}

fn test_diagonal_preconditioner_real() {
    let n = 30;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    let precond = DiagonalPreconditioner::from_csr(&matrix);

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

    let solution = sonium_solvers::gmres_preconditioned(&matrix, &precond, &b, &config);

    assert!(
        solution.converged,
        "Diagonal preconditioned GMRES should converge"
    );
}

fn test_diagonal_preconditioner_complex() {
    let n = 30;
    let matrix = tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0));
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), (i as f64 + 1.0).cos()))
        .collect();

    let precond = DiagonalPreconditioner::from_csr(&matrix);

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

    let solution = sonium_solvers::gmres_preconditioned(&matrix, &precond, &b, &config);

    assert!(
        solution.converged,
        "Diagonal preconditioned GMRES should converge for complex"
    );
}

fn test_block_jacobi_preconditioner() {
    let n = 30;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    let ilut_config = IlutConfig::new(1e-4, 10, 1e-12);
    let precond = BlockJacobiPreconditioner::from_csr(&matrix, 3, ilut_config);

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

    let solution = sonium_solvers::gmres_preconditioned(&matrix, &precond, &b, &config);

    assert!(
        solution.converged,
        "Block Jacobi preconditioned GMRES should converge"
    );
}

fn test_ilu_complex_matrix() {
    let n = 30;
    let diag = Complex64::new(4.0, 1.0);
    let off_diag = Complex64::new(-1.0, 0.5);
    let matrix = tridiagonal_complex(n, diag, off_diag);
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), (i as f64 + 1.0).cos()))
        .collect();

    let precond = IluPreconditioner::from_csr(&matrix);

    let config = GmresConfig {
        max_iterations: 200,
        restart: 30,
        tolerance: 1e-9,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = sonium_solvers::gmres_preconditioned(&matrix, &precond, &b, &config);

    assert!(
        solution.converged,
        "ILU should work with complex matrices: iterations={}, residual={:.3e}, status={:?}",
        solution.iterations, solution.residual, solution.status
    );
}

fn test_ilut_vs_ilu_iteration_comparison() {
    let n = 40;
    let matrix = tridiagonal_complex(n, Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0));
    let b: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.0))
        .collect();

    let ilu_precond = IluPreconditioner::from_csr(&matrix);
    let ilut_config = IlutConfig::new(1e-3, 10, 1e-12);
    let ilut_precond = IlutPreconditioner::from_csr_with_config(&matrix, &ilut_config);

    let config = GmresConfig {
        max_iterations: 200,
        restart: 40,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let ilu_sol = sonium_solvers::gmres_preconditioned(&matrix, &ilu_precond, &b, &config);
    let ilut_sol = sonium_solvers::gmres_preconditioned(&matrix, &ilut_precond, &b, &config);

    assert!(ilu_sol.converged, "ILU should converge");
    assert!(ilut_sol.converged, "ILUT should converge");

    println!(
        "ILU iterations: {}, ILUT iterations: {}",
        ilu_sol.iterations, ilut_sol.iterations
    );
}

fn test_preconditioner_identity_equivalence() {
    let n = 20;
    let matrix = tridiagonal_real(n, 4.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    let identity_precond = sonium_solvers::IdentityPreconditioner;

    let config = GmresConfig {
        max_iterations: 200,
        restart: 20,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = sonium_solvers::gmres_preconditioned(&matrix, &identity_precond, &b, &config);

    assert!(
        solution.converged,
        "GMRES with identity preconditioner should work"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_ilu_preconditioner_reduces_iterations();
        test_ilut_preconditioner();
        test_diagonal_preconditioner_real();
        test_diagonal_preconditioner_complex();
        test_block_jacobi_preconditioner();
        test_ilu_complex_matrix();
        test_ilut_vs_ilu_iteration_comparison();
        test_preconditioner_identity_equivalence();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
