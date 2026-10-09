//! WaveHoltz solver validation tests
//!
//! These integration tests validate the WaveHoltz solver against
//! direct solvers and analytical expectations.

use num_complex::Complex64;
use sonium_fem::assembly::{HelmholtzAssembler, HelmholtzProblem};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::solver::{SolverConfig, SolverType, solve};
use sonium_fem::waveholtz::{WaveHoltzConfig, solve_waveholtz, solve_waveholtz_multi_frequency};
use sonium_qa::relative_error;
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;

/// Helper: compute relative L2 error between WaveHoltz and direct solutions
fn waveholtz_vs_direct_error(n: usize, k: f64, steps_per_period: usize) -> (f64, usize) {
    let mesh = unit_square_triangles(n);
    let omega = k;

    let problem = HelmholtzProblem::assemble(
        &mesh,
        PolynomialDegree::P1,
        Complex64::new(k, 0.0),
        |x, y, _| Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0),
    );

    // Direct reference
    let direct_config = SolverConfig {
        solver_type: SolverType::Direct,
        ..Default::default()
    };
    let direct_sol = solve(&problem, &direct_config).expect("Direct solver should succeed");

    // WaveHoltz
    let assembler = HelmholtzAssembler::new(&mesh, PolynomialDegree::P1);
    let rhs_real: Vec<f64> = problem.rhs.iter().map(|c| c.re).collect();

    let wh_config = WaveHoltzConfig {
        steps_per_period,
        tolerance: 1e-10,
        inner_tolerance: 1e-12,
        dispersion_correction: false,
        ..Default::default()
    };
    let wh_sol = solve_waveholtz(&assembler, &rhs_real, omega, &wh_config)
        .expect("WaveHoltz should succeed");
    assert!(
        wh_sol.converged,
        "WaveHoltz did not converge for n={n}, k={k}"
    );

    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let mut w = VtuWriter::new(&mesh);
    w.add_point_scalar_complex("pressure", wh_sol.values.as_slice());
    let _ = w.write(dir.join(format!("waveholtz_n{n}_k{k}_steps{steps_per_period}.vtu")));

    (
        relative_error(&wh_sol.values, &direct_sol.values),
        wh_sol.iterations,
    )
}

fn test_waveholtz_time_convergence() {
    // Verify O(dt²) convergence: doubling steps should quarter the error
    let n = 8;
    let k = 1.0;

    let (err_20, _) = waveholtz_vs_direct_error(n, k, 20);
    let (err_40, _) = waveholtz_vs_direct_error(n, k, 40);

    let ratio = err_20 / err_40;
    assert!(
        ratio > 2.5,
        "Expected O(dt²) convergence (ratio > 2.5), got {:.2} (errors: {:.2e}, {:.2e})",
        ratio,
        err_20,
        err_40
    );
}

fn test_waveholtz_iteration_stability_across_meshes() {
    // For fixed k, iterations should not grow with mesh refinement
    let k = 1.0;
    let steps = 20;

    let (_, iters_4) = waveholtz_vs_direct_error(4, k, steps);
    let (_, iters_8) = waveholtz_vs_direct_error(8, k, steps);
    let (_, iters_16) = waveholtz_vs_direct_error(16, k, steps);

    // Iterations should stay roughly constant (allow some growth)
    assert!(
        iters_16 <= iters_4 * 3 + 10,
        "Iterations should not grow significantly: n=4: {}, n=8: {}, n=16: {}",
        iters_4,
        iters_8,
        iters_16
    );
}

fn test_waveholtz_accuracy_at_different_wavenumbers() {
    let n = 8;
    let steps = 80;

    for &k in &[0.5, 1.0, 3.0] {
        let (error, iterations) = waveholtz_vs_direct_error(n, k, steps);
        println!(
            "WaveHoltz k={k}: complex direct-reference error={error:.3e}, iterations={iterations}"
        );
        assert!(
            error < 1e-3,
            "WaveHoltz complex error should be below 0.1% for k={}: got {:.2e}",
            k,
            error
        );
    }
}

fn test_waveholtz_via_solver_dispatch() {
    // Verify WaveHoltz works through the main solve() dispatch
    let mesh = unit_square_triangles(4);
    let k = 1.0;

    let problem = HelmholtzProblem::assemble(
        &mesh,
        PolynomialDegree::P1,
        Complex64::new(k, 0.0),
        |_, _, _| Complex64::new(1.0, 0.0),
    );

    let config = SolverConfig {
        solver_type: SolverType::WaveHoltz,
        wavenumber: Some(k),
        waveholtz: Some(WaveHoltzConfig {
            steps_per_period: 30,
            tolerance: 1e-8,
            dispersion_correction: false,
            ..Default::default()
        }),
        ..Default::default()
    };

    let sol = solve(&problem, &config).expect("WaveHoltz via dispatch should succeed");
    assert!(sol.converged);
    assert_eq!(sol.values.len(), mesh.num_nodes());
}

fn test_waveholtz_multi_frequency_independence() {
    // Verify multi-frequency results match single-frequency results
    let mesh = unit_square_triangles(4);
    let assembler = HelmholtzAssembler::new(&mesh, PolynomialDegree::P1);

    let omegas = [1.0, 2.0];
    let mut rhs_vec = Vec::new();

    for &omega in &omegas {
        let problem = HelmholtzProblem::assemble(
            &mesh,
            PolynomialDegree::P1,
            Complex64::new(omega, 0.0),
            |_, _, _| Complex64::new(1.0, 0.0),
        );
        let rhs: Vec<f64> = problem.rhs.iter().map(|c| c.re).collect();
        rhs_vec.push((omega, rhs));
    }

    let wh_config = WaveHoltzConfig {
        steps_per_period: 30,
        tolerance: 1e-8,
        dispersion_correction: false,
        ..Default::default()
    };

    // Solve multi-frequency
    let multi_sols = solve_waveholtz_multi_frequency(&assembler, &rhs_vec, &wh_config)
        .expect("Multi-frequency should succeed");

    // Solve each independently
    for (i, (omega, rhs)) in rhs_vec.iter().enumerate() {
        let single_sol = solve_waveholtz(&assembler, rhs, *omega, &wh_config)
            .expect("Single-frequency should succeed");
        assert!(
            multi_sols[i].converged,
            "multi-frequency WaveHoltz solve {i} did not converge"
        );
        assert!(
            single_sol.converged,
            "single-frequency WaveHoltz solve {i} did not converge"
        );

        // Solutions should match
        let diff: f64 = multi_sols[i]
            .values
            .iter()
            .zip(single_sol.values.iter())
            .map(|(a, b)| (a - b).norm())
            .sum();
        assert!(
            diff < 1e-10,
            "Multi-frequency solution {} should match single: diff = {:.2e}",
            i,
            diff
        );
    }
}

fn test_waveholtz_zero_rhs() {
    // Zero RHS should give zero solution
    let mesh = unit_square_triangles(4);
    let assembler = HelmholtzAssembler::new(&mesh, PolynomialDegree::P1);
    let omega = 1.0;
    let rhs: Vec<f64> = vec![0.0; mesh.num_nodes()];

    let wh_config = WaveHoltzConfig {
        steps_per_period: 10,
        tolerance: 1e-8,
        dispersion_correction: false,
        ..Default::default()
    };

    let sol = solve_waveholtz(&assembler, &rhs, omega, &wh_config)
        .expect("Zero RHS should converge trivially");

    let sol_norm: f64 = sol.values.iter().map(|c| c.norm()).sum();
    assert!(
        sol_norm < 1e-10,
        "Zero RHS should give zero solution: ||u|| = {:.2e}",
        sol_norm
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_waveholtz_time_convergence();
        test_waveholtz_iteration_stability_across_meshes();
        test_waveholtz_accuracy_at_different_wavenumbers();
        test_waveholtz_via_solver_dispatch();
        test_waveholtz_multi_frequency_independence();
        test_waveholtz_zero_rhs();
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
