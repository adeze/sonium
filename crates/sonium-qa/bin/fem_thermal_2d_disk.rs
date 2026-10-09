//! QA: FEM thermal — 2D steady conduction on a disk with uniform heat generation.
//!
//! Analytical: T(r) = T_R + q_dot/(4k) * (R² - r²)

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::circular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm};
use sonium_fem::spaces::H1Space;
use sonium_solvers::{GmresConfig, gmres};

fn test_thermal_2d_disk_uniform_generation() {
    let r = 1.0;
    let k = 10.0;
    let q_dot = 1e4;
    let t_boundary = 50.0;

    let n_radial = 40;
    let n_angular = 80;
    let mesh = circular_mesh_triangles(0.0, 0.0, r, n_radial, n_angular);

    let space = H1Space::new(PolynomialDegree::P1);

    let conduction = ConductionForm::from_conductivity(k);
    let source = HeatSourceForm {
        source_fn: |_x, _y, _z| q_dot,
    };

    let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
    let rhs = assemble_linear(&source, &space, &mesh, 1);

    let mut system = FemSystem::new(stiffness, rhs);

    // Outer boundary marker is 1 in circular_mesh_triangles
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| t_boundary, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 50,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES should converge for 2D disk");

    // Analytical: T(r) = T_R + q_dot/(4k) * (R² - r²)
    let mut max_err: f64 = 0.0;
    let mut t_center_fem = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let ri = (node.x * node.x + node.y * node.y).sqrt();
        let t_exact = t_boundary + (q_dot / (4.0 * k)) * (r * r - ri * ri);
        let t_fem = result.x[i];
        let err = (t_fem - t_exact).abs();
        max_err = max_err.max(err);
        if ri < 1e-10 {
            t_center_fem = t_fem;
        }
    }

    let t_center_exact = t_boundary + q_dot * r * r / (4.0 * k);
    println!(
        "Thermal 2D disk: max_err = {:.4e}, T_center_exact = {:.2}, T_center_fem = {:.2}",
        max_err, t_center_exact, t_center_fem
    );
    let temperature_rise = q_dot * r * r / (4.0 * k);
    let normalized_error = max_err / temperature_rise;
    assert!(
        normalized_error < 0.002,
        "Normalized disk temperature error {:.3}% exceeds 0.2%",
        100.0 * normalized_error
    );
}

fn test_thermal_2d_disk_convergence() {
    let r = 1.0;
    let k = 10.0;
    let q_dot = 1e4;
    let t_boundary = 50.0;

    let errors: Vec<f64> = [(10, 20), (20, 40), (40, 80)]
        .iter()
        .map(|&(nr, na)| {
            let mesh = circular_mesh_triangles(0.0, 0.0, r, nr, na);
            let space = H1Space::new(PolynomialDegree::P1);

            let conduction = ConductionForm::from_conductivity(k);
            let source = HeatSourceForm {
                source_fn: |_x, _y, _z| q_dot,
            };

            let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1);
            let rhs = assemble_linear(&source, &space, &mesh, 1);

            let mut system = FemSystem::new(stiffness, rhs);
            apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| t_boundary, None);

            let csr = system.matrix.to_csr();
            let result = gmres(
                &csr,
                &system.rhs,
                &GmresConfig {
                    max_iterations: 2000,
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
                let ri = (node.x * node.x + node.y * node.y).sqrt();
                let t_exact = t_boundary + (q_dot / (4.0 * k)) * (r * r - ri * ri);
                max_err = max_err.max((result.x[i] - t_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (0.5_f64).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (0.5_f64).log(std::f64::consts::LN_10);

    println!(
        "Convergence 2D disk: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );

    assert!(
        rate_1 > 1.5 && rate_2 > 1.5,
        "Convergence rate too low: {:.2}, {:.2}",
        rate_1,
        rate_2
    );
    let temperature_rise = q_dot * r * r / (4.0 * k);
    let normalized_error = errors[2] / temperature_rise;
    assert!(
        normalized_error < 0.002,
        "Finest normalized disk error {:.3}% exceeds 0.2%",
        100.0 * normalized_error
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_thermal_2d_disk_uniform_generation();
        test_thermal_2d_disk_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
