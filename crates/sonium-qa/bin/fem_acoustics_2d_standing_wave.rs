//! QA: FEM acoustics — 2D manufactured Helmholtz solution with source term.
//!
//! Exact: p(x,y) = sin(πx)*sin(πy) on [0,1]×[0,1].
//! Satisfies ∇²p = -2π²p. We solve ∇²p + k²p = f with k=1.0
//! and source f = (k² - 2π²)*sin(πx)*sin(πy).

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::{LaplacianForm, MassForm, SourceForm};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::analytical::acoustics_2d_dirichlet_standing;
use sonium_solvers::{GmresConfig, gmres};

fn test_acoustics_2d_standing_wave() {
    let lx = 1.0;
    let ly = 1.0;
    let nx = 40;
    let ny = 40;
    let mesh = rectangular_mesh_triangles(0.0, lx, 0.0, ly, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let _n_dofs = space.num_global_dofs(&mesh);

    let k = 1.0;
    let k_sq = k * k;
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    let source_coeff = k_sq - two_pi_sq;

    let stiffness = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1);
    let mass = assemble_bilinear(&MassForm, &space, &space, &mesh, 1);

    // Build K - k²M
    let mut combined = stiffness;
    for i in 0..mass.nnz() {
        combined.push(mass.rows[i], mass.cols[i], -k_sq * mass.values[i]);
    }

    // Source term: f = -(k² - 2π²) * sin(πx)*sin(πy) = (2π² - k²) * sin(πx)*sin(πy)
    let source = SourceForm::new(|x, y, _z| {
        -source_coeff * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
    });
    let rhs = assemble_linear(&source, &space, &mesh, 1);

    let mut system = FemSystem::new(combined, rhs);

    // Dirichlet on all four sides matching exact solution
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        1,
        &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
        None,
    );
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        2,
        &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
        None,
    );
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        3,
        &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
        None,
    );
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        4,
        &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
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
        "GMRES should converge for 2D standing wave"
    );

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let p_exact = acoustics_2d_dirichlet_standing(node.x, node.y, lx, ly, 1, 1);
        max_err = max_err.max((result.x[i] - p_exact).abs());
    }

    println!("Acoustics 2D standing wave: max_err = {:.4e}", max_err);
    assert!(
        max_err < 1e-2,
        "Max pressure error too large: {:.4e}",
        max_err
    );
}

fn test_acoustics_2d_standing_wave_convergence() {
    let lx = 1.0;
    let ly = 1.0;
    let k = 1.0;
    let k_sq = k * k;
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    let source_coeff = k_sq - two_pi_sq;

    let errors: Vec<f64> = [20, 40, 80]
        .iter()
        .map(|&n| {
            let mesh = rectangular_mesh_triangles(0.0, lx, 0.0, ly, n, n);
            let space = H1Space::new(PolynomialDegree::P1);
            let _n_dofs = space.num_global_dofs(&mesh);

            let stiffness = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1);
            let mass = assemble_bilinear(&MassForm, &space, &space, &mesh, 1);

            let mut combined = stiffness;
            for i in 0..mass.nnz() {
                combined.push(mass.rows[i], mass.cols[i], -k_sq * mass.values[i]);
            }

            let source = SourceForm::new(|x, y, _z| {
                -source_coeff * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
            });
            let rhs = assemble_linear(&source, &space, &mesh, 1);

            let mut system = FemSystem::new(combined, rhs);
            apply_dirichlet_values(
                &mut system,
                &mesh,
                &space,
                1,
                &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
                None,
            );
            apply_dirichlet_values(
                &mut system,
                &mesh,
                &space,
                2,
                &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
                None,
            );
            apply_dirichlet_values(
                &mut system,
                &mesh,
                &space,
                3,
                &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
                None,
            );
            apply_dirichlet_values(
                &mut system,
                &mesh,
                &space,
                4,
                &|x, y, _| acoustics_2d_dirichlet_standing(x, y, lx, ly, 1, 1),
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
                let p_exact = acoustics_2d_dirichlet_standing(node.x, node.y, lx, ly, 1, 1);
                max_err = max_err.max((result.x[i] - p_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (20.0_f64 / 40.0).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (40.0_f64 / 80.0).log(std::f64::consts::LN_10);

    println!(
        "Acoustics 2D standing wave convergence: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );
    assert!(rate_1 > 1.0 && rate_2 > 1.0, "Convergence rate too low");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_acoustics_2d_standing_wave();
        test_acoustics_2d_standing_wave_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
