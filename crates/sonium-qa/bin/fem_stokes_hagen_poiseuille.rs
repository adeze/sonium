//! QA: FEM Stokes — Hagen-Poiseuille flow in a circular pipe.
//!
//! Analytical: u(r) = Δp*(R² - r²)/(4*μ*L)
//!
//! This is solved as a scalar Poisson problem on the disk cross-section:
//!   -μ ∇²w = dp/dz   (w = axial velocity, analog to temperature)
//! with w=0 on the pipe wall.

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::circular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm};
use sonium_fem::spaces::H1Space;
use sonium_qa::analytical::stokes_circular_poiseuille;
use sonium_solvers::{GmresConfig, gmres};

fn test_hagen_poiseuille() {
    let r_pipe = 1.0;
    let mu = 1.0;
    let dpdx = -8.0; // pressure gradient driving the flow

    let n_radial = 10;
    let n_angular = 24;
    let mesh = circular_mesh_triangles(0.0, 0.0, r_pipe, n_radial, n_angular);

    let space = H1Space::new(PolynomialDegree::P1);

    // Scalar Poisson analogy: k=μ, source=-dp/dx
    let conduction = ConductionForm::from_conductivity(mu);
    let source = HeatSourceForm {
        source_fn: |_x, _y, _z| -dpdx,
    };

    let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
    let rhs = assemble_linear(&source, &space, &mesh, 1);

    let mut system = FemSystem::new(stiffness, rhs);

    // Apply Dirichlet w=0 on all boundary nodes (marker 1 = outer wall)
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);

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
    assert!(
        result.converged,
        "GMRES should converge for Hagen-Poiseuille"
    );

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let r = (node.x * node.x + node.y * node.y).sqrt();
        let w_fem = result.x[i];
        let w_exact = stokes_circular_poiseuille(r, r_pipe, dpdx, mu);
        max_err = max_err.max((w_fem - w_exact).abs());
    }

    println!("Hagen-Poiseuille: max_err = {:.4e}", max_err);
    assert!(max_err < 0.03, "Max error too large: {:.4e}", max_err);
}

fn test_hagen_poiseuille_convergence() {
    let r_pipe = 1.0;
    let mu = 1.0;
    let dpdx = -8.0;

    let errors: Vec<f64> = [(5, 12), (8, 20), (12, 32)]
        .iter()
        .map(|&(n_radial, n_angular)| {
            let mesh = circular_mesh_triangles(0.0, 0.0, r_pipe, n_radial, n_angular);
            let space = H1Space::new(PolynomialDegree::P1);

            let conduction = ConductionForm::from_conductivity(mu);
            let source = HeatSourceForm {
                source_fn: |_x, _y, _z| -dpdx,
            };

            let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
            let rhs = assemble_linear(&source, &space, &mesh, 1);

            let mut system = FemSystem::new(stiffness, rhs);
            apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);

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
            for (i, node) in mesh.nodes.iter().enumerate() {
                let r = (node.x * node.x + node.y * node.y).sqrt();
                let w_fem = result.x[i];
                let w_exact = stokes_circular_poiseuille(r, r_pipe, dpdx, mu);
                max_err = max_err.max((w_fem - w_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / ((5.0_f64 / 8.0_f64).log(std::f64::consts::LN_10));
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / ((8.0_f64 / 12.0_f64).log(std::f64::consts::LN_10));

    println!(
        "Convergence Hagen-Poiseuille: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );

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
        test_hagen_poiseuille();
        test_hagen_poiseuille_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
