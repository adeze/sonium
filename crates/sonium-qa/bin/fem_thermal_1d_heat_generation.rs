//! QA: FEM thermal — 1D steady conduction with uniform heat generation.
//!
//! Analytical: T(x) = T_wall + q_dot/(2k) * x * (L - x)

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::analytical::thermal_1d_heat_generation;
use sonium_solvers::{GmresConfig, gmres};

fn test_thermal_1d_heat_generation() {
    let l = 1.0;
    let w = 0.05;
    let k = 50.0;
    let q_dot = 1e5;
    let t_wall = 100.0;

    let nx = 40;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let _n_dofs = space.num_global_dofs(&mesh);

    let conduction = ConductionForm::from_conductivity(k);
    let source = HeatSourceForm {
        source_fn: |_x, _y, _z| q_dot,
    };

    let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
    let rhs = assemble_linear(&source, &space, &mesh, 1);

    let mut system = FemSystem::new(stiffness, rhs);

    // Left boundary (marker 1): x = 0
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| t_wall, None);
    // Right boundary (marker 2): x = L
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| t_wall, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 1000,
            restart: 50,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES should converge for thermal");

    let mut max_err: f64 = 0.0;
    for (i, &t_fem) in result.x.iter().enumerate().take(mesh.num_nodes()) {
        let x = mesh.nodes[i].x;
        let t_exact = thermal_1d_heat_generation(x, l, k, q_dot, t_wall);
        max_err = max_err.max((t_fem - t_exact).abs());
    }

    let t_max_exact = thermal_1d_heat_generation(l / 2.0, l, k, q_dot, t_wall);
    println!(
        "Thermal 1D heat generation: max_err = {:.4e}, T_max_exact = {:.2}",
        max_err, t_max_exact
    );
    assert!(max_err < 1.0e-9, "Max error too large: {:.4e}", max_err);
}

fn test_thermal_1d_heat_generation_convergence() {
    let l = 1.0;
    let w = 0.05;
    let k = 50.0;
    let q_dot = 1e5;
    let t_wall = 100.0;

    let errors: Vec<f64> = [10, 20, 40]
        .iter()
        .map(|&nx| {
            let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, 2);
            let space = H1Space::new(PolynomialDegree::P1);
            let _n_dofs = space.num_global_dofs(&mesh);

            let conduction = ConductionForm::from_conductivity(k);
            let source = HeatSourceForm {
                source_fn: |_x, _y, _z| q_dot,
            };

            let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
            let rhs = assemble_linear(&source, &space, &mesh, 1);

            let mut system = FemSystem::new(stiffness, rhs);
            apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| t_wall, None);
            apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| t_wall, None);

            let csr = system.matrix.to_csr();
            let result = gmres(
                &csr,
                &system.rhs,
                &GmresConfig {
                    max_iterations: 1000,
                    restart: 50,
                    tolerance: 1e-12,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    min_restart: None,
                    max_restart: None,
                    exact_residual_check_interval: 0,
                },
            );
            assert!(result.converged);

            let mut max_err: f64 = 0.0;
            for (i, &t_fem) in result.x.iter().enumerate().take(mesh.num_nodes()) {
                let x = mesh.nodes[i].x;
                let t_exact = thermal_1d_heat_generation(x, l, k, q_dot, t_wall);
                max_err = max_err.max((t_fem - t_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (0.5_f64).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (0.5_f64).log(std::f64::consts::LN_10);

    println!(
        "Convergence heat gen: nx=10 err={:.2e}, nx=20 err={:.2e}, nx=40 err={:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );

    // When discretization error is below solver tolerance, rates are noisy.
    // Accept if all errors are small (< 1e-8) or if rates indicate convergence.
    let all_small = errors.iter().all(|&e| e < 1e-8);
    assert!(
        all_small || (rate_1 > 1.5 && rate_2 > 1.5),
        "Convergence rate too low: {:.2}, {:.2}",
        rate_1,
        rate_2
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_thermal_1d_heat_generation();
        test_thermal_1d_heat_generation_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
