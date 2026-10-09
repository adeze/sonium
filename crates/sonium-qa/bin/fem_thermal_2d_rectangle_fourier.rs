//! QA: FEM thermal — 2D rectangular domain with Fourier-series exact solution.
//!
//! Domain: [0,a]×[0,b]. Three sides at T=0, top side at T=T0.
//! Exact: T(x,y) = (4T0/π) Σ_{n odd} sin(nπx/a) sinh(nπy/a) / (n sinh(nπb/a)).

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::LaplacianForm;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::analytical::thermal_2d_rectangle_fourier;
use sonium_solvers::{GmresConfig, gmres};

fn test_thermal_2d_rectangle_fourier() {
    let a = 2.0;
    let b = 1.0;
    let t0 = 100.0;
    let nx = 40;
    let ny = 20;
    let mesh = rectangular_mesh_triangles(0.0, a, 0.0, b, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let form = LaplacianForm;
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let n_dofs = space.num_global_dofs(&mesh);
    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);

    // Dirichlet T=0 on bottom (tag 3), left (tag 1), right (tag 2)
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 0.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 3, &|_, _, _| 0.0, None);
    // Dirichlet T=T0 on top (tag 4)
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        4,
        &|x, _, _| t0 * (std::f64::consts::PI * x / a).sin(),
        None,
    );

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
        "GMRES should converge for thermal Fourier"
    );

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let x = node.x;
        let y = node.y;
        let t_exact = thermal_2d_rectangle_fourier(x, y, a, b, t0);
        max_err = max_err.max((result.x[i] - t_exact).abs());
    }

    println!("Thermal 2D rectangle Fourier: max_err = {:.4e}", max_err);
    assert!(
        max_err < 0.02,
        "Max temperature error too large: {:.4e}",
        max_err
    );
}

fn test_thermal_2d_rectangle_fourier_convergence() {
    let a = 2.0;
    let b = 1.0;
    let t0 = 100.0;

    let errors: Vec<f64> = [20, 40, 80]
        .iter()
        .map(|&n| {
            let nx = n;
            let ny = n / 2;
            let mesh = rectangular_mesh_triangles(0.0, a, 0.0, b, nx, ny);
            let space = H1Space::new(PolynomialDegree::P1);
            let form = LaplacianForm;
            let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

            let n_dofs = space.num_global_dofs(&mesh);
            let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);
            apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
            apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 0.0, None);
            apply_dirichlet_values(&mut system, &mesh, &space, 3, &|_, _, _| 0.0, None);
            apply_dirichlet_values(
                &mut system,
                &mesh,
                &space,
                4,
                &|x, _, _| t0 * (std::f64::consts::PI * x / a).sin(),
                None,
            );

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
                let t_exact = thermal_2d_rectangle_fourier(node.x, node.y, a, b, t0);
                max_err = max_err.max((result.x[i] - t_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (20.0_f64 / 40.0).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (40.0_f64 / 80.0).log(std::f64::consts::LN_10);

    println!(
        "Thermal 2D rectangle convergence: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );
    assert!(rate_1 > 1.0 && rate_2 > 1.0, "Convergence rate too low");
    assert!(
        errors[2] < 0.005,
        "Finest Fourier temperature error {:.3e} exceeds 0.005 K",
        errors[2]
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_thermal_2d_rectangle_fourier();
        test_thermal_2d_rectangle_fourier_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
