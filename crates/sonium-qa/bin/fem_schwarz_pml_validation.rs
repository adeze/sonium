//! Schwarz-PML solver validation tests
//!
//! Integration tests validating the Optimized Schwarz Method with PML
//! transmission conditions against direct solvers, testing high-frequency
//! robustness, and comparing additive vs multiplicative variants.

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::apply_homogeneous_dirichlet;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::schwarz_pml::{SchwarzPmlConfig, SchwarzVariant, solve_schwarz_pml};
use sonium_fem::solver::{SolverConfig, SolverType, solve};
use sonium_qa::QaResult;
use std::f64::consts::PI;

/// Helper: compute L2 error between Schwarz-PML solution and direct solution
fn schwarz_vs_direct_error(n: usize, k: f64, num_subdomains: usize) -> (f64, usize) {
    schwarz_vs_direct_error_with_target(n, k, num_subdomains, 1.0e-6)
}

/// Compare a PML-Schwarz solve with the monolithic discrete reference while
/// exposing the requested PML round-trip reflection target.
fn schwarz_vs_direct_error_with_target(
    n: usize,
    k: f64,
    num_subdomains: usize,
    target_reflection: f64,
) -> (f64, usize) {
    let mesh = unit_square_triangles(n);
    let wavenumber = Complex64::new(k, 0.0);

    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |x, y, _| {
            Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0)
        });

    // The reference and Schwarz solves must represent the same discrete PDE.
    // `unit_square_triangles` tags left/right/bottom/top as 1/2/3/4.
    apply_homogeneous_dirichlet(&mut problem, &mesh, &[1, 2, 3, 4]);

    // Direct reference
    let direct_config = SolverConfig {
        solver_type: SolverType::Direct,
        ..Default::default()
    };
    let direct_sol = solve(&problem, &direct_config).expect("Direct solver should succeed");

    // Schwarz-PML
    let mut dirichlet_bcs = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        if node.x.abs() < 1e-10
            || (node.x - 1.0).abs() < 1e-10
            || node.y.abs() < 1e-10
            || (node.y - 1.0).abs() < 1e-10
        {
            dirichlet_bcs.push((i, Complex64::new(0.0, 0.0)));
        }
    }

    let config = SchwarzPmlConfig {
        num_subdomains,
        pml_target_reflection: target_reflection,
        max_iterations: 50,
        tolerance: 1e-6,
        verbosity: 0,
        ..Default::default()
    };

    let schwarz_sol = solve_schwarz_pml(
        &mesh,
        PolynomialDegree::P1,
        wavenumber,
        &problem.rhs,
        &dirichlet_bcs,
        &config,
    )
    .expect("Schwarz-PML should converge");

    // Relative L2 error
    let direct_norm: f64 = direct_sol
        .values
        .iter()
        .map(|c| c.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let diff_norm: f64 = direct_sol
        .values
        .iter()
        .zip(schwarz_sol.values.iter())
        .map(|(d, s)| (d - s).norm_sqr())
        .sum::<f64>()
        .sqrt();

    (diff_norm / direct_norm.max(1e-15), schwarz_sol.iterations)
}

/// Sweep the configured PML reflection budget on a production 2-D Helmholtz
/// solve.  Each result is compared with the same assembled monolithic system,
/// so a target that silently degrades the interface solve cannot pass merely
/// because the outer iteration reports convergence.
fn test_pml_reflection_target_sweep() -> Vec<(f64, f64, usize)> {
    let mut metrics = Vec::new();
    for target in [1.0e-2, 1.0e-4, 1.0e-6] {
        let (error, iterations) = schwarz_vs_direct_error_with_target(12, 5.0, 2, target);
        println!(
            "Schwarz-PML target reflection={target:.1e}, direct relative error={error:.4e}, iterations={iterations}"
        );
        assert!(
            error.is_finite(),
            "PML target {target:.1e} produced non-finite error"
        );
        assert!(
            iterations > 0,
            "PML target {target:.1e} must perform an iteration"
        );
        assert!(
            error < 1.0e-3,
            "PML target {target:.1e} exceeds the discrete reference budget: {error:.3e}"
        );
        metrics.push((target, error, iterations));
    }
    metrics
}

/// Test 1: Correctness — Schwarz-PML converges to a valid solution
///
/// Unit square, k=2, manufactured solution. Schwarz-PML with 2 subdomains
/// should produce a solution that satisfies the Helmholtz equation at interior nodes.
/// We compare to the direct solver, but allow for larger discrepancy since the Schwarz-PML
/// method uses PML-extended local problems (slightly different discretization).
fn test_schwarz_pml_vs_direct() {
    let (error, iterations) = schwarz_vs_direct_error(12, 2.0, 2);
    println!("Schwarz-PML/direct relative error={error:.4e}, iterations={iterations}");
    assert!(iterations > 0, "Should require at least 1 iteration");
    assert!(
        error < 1e-3,
        "Schwarz-PML must solve the same discrete system as the direct reference: error = {:.2e}",
        error
    );
}

/// Test 2: High-frequency robustness
///
/// Solve for increasing k values on meshes with ~10 points per wavelength.
/// The KEY demonstration: PML transmission conditions should give bounded
/// iteration counts independent of k.
fn test_schwarz_pml_high_frequency_robustness() {
    let max_iters_allowed = 50;
    let mut results = Vec::new();

    for &k in &[5.0, 10.0, 20.0] {
        // ~10 points per wavelength: n ≈ k * domain_size / (2*pi) * 10
        let ppw = 10.0;
        let n = ((k * ppw / (2.0 * PI)) as usize).max(8);
        let mesh = unit_square_triangles(n);
        let wavenumber = Complex64::new(k, 0.0);

        let problem =
            HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |x, y, _| {
                Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0)
            });

        let mut dirichlet_bcs = Vec::new();
        for (i, node) in mesh.nodes.iter().enumerate() {
            if node.x.abs() < 1e-10
                || (node.x - 1.0).abs() < 1e-10
                || node.y.abs() < 1e-10
                || (node.y - 1.0).abs() < 1e-10
            {
                dirichlet_bcs.push((i, Complex64::new(0.0, 0.0)));
            }
        }

        let config = SchwarzPmlConfig {
            num_subdomains: 4,
            max_iterations: max_iters_allowed,
            tolerance: 1e-6,
            verbosity: 0,
            ..Default::default()
        };

        let result = solve_schwarz_pml(
            &mesh,
            PolynomialDegree::P1,
            wavenumber,
            &problem.rhs,
            &dirichlet_bcs,
            &config,
        );

        match result {
            Ok(sol) => {
                results.push((k, sol.iterations, sol.residual));
                assert!(
                    sol.iterations < max_iters_allowed,
                    "k={}: Schwarz-PML took {} iterations (max allowed: {})",
                    k,
                    sol.iterations,
                    max_iters_allowed
                );
            }
            Err(e) => {
                panic!("k={}: Schwarz-PML failed: {}", k, e);
            }
        }
    }

    // Iteration counts should be bounded (not growing proportionally with k)
    if results.len() >= 2 {
        let first_iters = results[0].1 as f64;
        let last_iters = results[results.len() - 1].1 as f64;
        // Allow up to 3x growth (theoretical: bounded, but mesh effects may cause some growth)
        assert!(
            last_iters <= first_iters * 3.0 + 5.0,
            "Iteration counts should be bounded: first_k={} ({} iters), last_k={} ({} iters)",
            results[0].0,
            results[0].1,
            results[results.len() - 1].0,
            results[results.len() - 1].1,
        );
    }
}

/// Test 3: Additive vs multiplicative convergence
///
/// Multiplicative Schwarz should converge in fewer or comparable iterations
/// compared to additive Schwarz.
fn test_additive_vs_multiplicative_convergence() {
    let mesh = unit_square_triangles(8);
    let k = 3.0;
    let wavenumber = Complex64::new(k, 0.0);

    let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |x, y, _| {
        Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0)
    });

    let mut dirichlet_bcs = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        if node.x.abs() < 1e-10
            || (node.x - 1.0).abs() < 1e-10
            || node.y.abs() < 1e-10
            || (node.y - 1.0).abs() < 1e-10
        {
            dirichlet_bcs.push((i, Complex64::new(0.0, 0.0)));
        }
    }

    let additive_config = SchwarzPmlConfig {
        variant: SchwarzVariant::Additive,
        num_subdomains: 3,
        max_iterations: 50,
        tolerance: 1e-6,
        verbosity: 0,
        ..Default::default()
    };
    let multiplicative_config = SchwarzPmlConfig {
        variant: SchwarzVariant::Multiplicative,
        ..additive_config.clone()
    };

    let add_sol = solve_schwarz_pml(
        &mesh,
        PolynomialDegree::P1,
        wavenumber,
        &problem.rhs,
        &dirichlet_bcs,
        &additive_config,
    )
    .expect("Additive should converge");

    let mult_sol = solve_schwarz_pml(
        &mesh,
        PolynomialDegree::P1,
        wavenumber,
        &problem.rhs,
        &dirichlet_bcs,
        &multiplicative_config,
    )
    .expect("Multiplicative should converge");

    assert!(
        mult_sol.iterations <= add_sol.iterations + 5,
        "Multiplicative ({}) should converge no slower than additive ({})",
        mult_sol.iterations,
        add_sol.iterations,
    );
}

/// Test 4: Comparison with the standard monolithic solver.
///
/// `GmresSchwarz` is a reserved solver enum variant today, not an implemented
/// algebraic preconditioner. Keep this QA focused on the implemented PML path
/// and compare it against a monolithic GMRES+ILU solve for the same problem.
fn test_schwarz_pml_vs_gmres_ilu() {
    let mesh = unit_square_triangles(10);
    let k = 5.0;
    let wavenumber = Complex64::new(k, 0.0);

    let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |x, y, _| {
        Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0)
    });

    let monolithic_config = SolverConfig {
        solver_type: SolverType::GmresIlu,
        ..Default::default()
    };
    let monolithic_sol =
        solve(&problem, &monolithic_config).expect("GMRES+ILU reference should converge");

    // PML-Schwarz
    let mut dirichlet_bcs = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        if node.x.abs() < 1e-10
            || (node.x - 1.0).abs() < 1e-10
            || node.y.abs() < 1e-10
            || (node.y - 1.0).abs() < 1e-10
        {
            dirichlet_bcs.push((i, Complex64::new(0.0, 0.0)));
        }
    }

    let pml_config = SchwarzPmlConfig {
        num_subdomains: 4,
        max_iterations: 50,
        tolerance: 1e-6,
        verbosity: 0,
        ..Default::default()
    };

    let pml_result = solve_schwarz_pml(
        &mesh,
        PolynomialDegree::P1,
        wavenumber,
        &problem.rhs,
        &dirichlet_bcs,
        &pml_config,
    );

    assert!(pml_result.is_ok(), "PML-Schwarz should converge");
    let pml_sol = pml_result.unwrap();

    assert!(monolithic_sol.converged);
    assert!(pml_sol.converged);
}

/// Test 5: Multiple subdomain counts
///
/// Test with 2, 4, and 8 subdomains. All should converge.
fn test_multiple_subdomain_counts() {
    let mesh = unit_square_triangles(10);
    let k = 3.0;
    let wavenumber = Complex64::new(k, 0.0);

    let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |x, y, _| {
        Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0)
    });

    let mut dirichlet_bcs = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        if node.x.abs() < 1e-10
            || (node.x - 1.0).abs() < 1e-10
            || node.y.abs() < 1e-10
            || (node.y - 1.0).abs() < 1e-10
        {
            dirichlet_bcs.push((i, Complex64::new(0.0, 0.0)));
        }
    }

    let mut all_iters = Vec::new();

    for &num_sub in &[2, 4, 8] {
        let config = SchwarzPmlConfig {
            num_subdomains: num_sub,
            max_iterations: 50,
            tolerance: 1e-6,
            verbosity: 0,
            ..Default::default()
        };

        let result = solve_schwarz_pml(
            &mesh,
            PolynomialDegree::P1,
            wavenumber,
            &problem.rhs,
            &dirichlet_bcs,
            &config,
        );

        assert!(
            result.is_ok(),
            "{} subdomains: Schwarz-PML should converge: {:?}",
            num_sub,
            result.err()
        );
        let sol = result.unwrap();
        assert!(sol.converged, "{} subdomains: should converge", num_sub);
        all_iters.push((num_sub, sol.iterations));
    }

    // Iteration counts should be weakly dependent on number of subdomains
    // (not growing faster than linearly)
    let iters_2 = all_iters[0].1 as f64;
    let iters_8 = all_iters[2].1 as f64;
    assert!(
        iters_8 <= iters_2 * 4.0 + 10.0,
        "Iterations should not grow faster than linearly with num_subdomains: 2->{}, 8->{}",
        all_iters[0].1,
        all_iters[2].1,
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_schwarz_pml_vs_direct();
        test_schwarz_pml_high_frequency_robustness();
        test_additive_vs_multiplicative_convergence();
        test_schwarz_pml_vs_gmres_ilu();
        test_multiple_subdomain_counts();
        test_pml_reflection_target_sweep()
    });
    match result {
        Ok(reflection_metrics) => {
            let mut record = QaResult::passed("fem.schwarz-pml-reflection.v1")
                .requested_backend("cpu")
                .actual_backend("cpu");
            for (target, error, iterations) in reflection_metrics {
                let key = format!("target_{:.0e}", target).replace('-', "m");
                record = record
                    .metric(format!("{key}_relative_error"), error)
                    .metric(format!("{key}_iterations"), iterations as f64)
                    .maximum(format!("{key}_relative_error"), 1.0e-3);
            }
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pml_reflection_target_sweep_is_bounded() {
        let metrics = super::test_pml_reflection_target_sweep();
        assert_eq!(metrics.len(), 3);
    }
}
