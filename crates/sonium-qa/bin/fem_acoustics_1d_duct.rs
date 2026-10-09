//! QA: FEM acoustics — 1D duct low-k limit (Laplace equation).
//!
//! At k→0, Helmholtz reduces to Laplace: ∇²p = 0.
//! With Dirichlet BCs p(0)=0, p(L)=1, the exact solution is p(x)=x/L.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::{LaplacianForm, MassForm};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn test_acoustics_1d_duct_low_k() {
    let l = 1.0;
    let w = 0.05;
    let nx = 20;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    // Assemble K and M separately
    let stiffness =
        sonium_fem::assembly::assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1);
    let mass = sonium_fem::assembly::assemble_bilinear(&MassForm, &space, &space, &mesh, 1);

    // For low k, Helmholtz: A = K - k²M ≈ K
    let k = 1e-6;
    let k_sq = k * k;

    // Build K - k²M manually
    let mut combined = stiffness;
    for i in 0..mass.nnz() {
        combined.push(mass.rows[i], mass.cols[i], -k_sq * mass.values[i]);
    }

    let mut system = FemSystem::new(combined, vec![0.0; n_dofs]);

    // Left boundary: p = 0
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
    // Right boundary: p = 1
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 1.0, None);

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
        "GMRES should converge for low-k acoustics"
    );

    // Analytical: p(x) = x/L
    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let x = node.x;
        let p_exact = x / l;
        let p_fem = result.x[i];
        max_err = max_err.max((p_fem - p_exact).abs());
    }

    println!("Acoustics 1D duct (low-k): max_err = {:.4e}", max_err);
    assert!(max_err < 1e-6, "Max error too large: {:.4e}", max_err);
}

fn test_acoustics_1d_duct_convergence() {
    let l = 1.0;
    let w = 0.05;
    let k = 1e-6;
    let k_sq = k * k;

    let errors: Vec<f64> = [10, 20, 40]
        .iter()
        .map(|&nx| {
            let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, 2);
            let space = H1Space::new(PolynomialDegree::P1);
            let n_dofs = space.num_global_dofs(&mesh);

            let stiffness =
                sonium_fem::assembly::assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1);
            let mass = sonium_fem::assembly::assemble_bilinear(&MassForm, &space, &space, &mesh, 1);

            let mut combined = stiffness;
            for i in 0..mass.nnz() {
                combined.push(mass.rows[i], mass.cols[i], -k_sq * mass.values[i]);
            }

            let mut system = FemSystem::new(combined, vec![0.0; n_dofs]);
            apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
            apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 1.0, None);

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
                let x = node.x;
                let p_exact = x / l;
                max_err = max_err.max((result.x[i] - p_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (0.5_f64).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (0.5_f64).log(std::f64::consts::LN_10);

    println!(
        "Acoustics duct convergence: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );
    let all_small = errors.iter().all(|&e| e < 1e-6);
    assert!(
        all_small || (rate_1 > 1.5 && rate_2 > 1.5),
        "Convergence rate too low"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_acoustics_1d_duct_low_k();
        test_acoustics_1d_duct_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
