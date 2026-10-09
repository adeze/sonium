//! QA: FEM electrostatics — conducting sphere in uniform external field.
//!
//! Domain: spherical shell [a, R] (tetrahedral mesh).
//! PDE:   ∇²φ = 0
//! BC:    φ = 0          on inner sphere  (r = a)
//!        φ = -E0·z·(1-a³/R³) on outer sphere (r = R)
//!
//! Analytical: φ = -E0·z·(1 - a³/r³)   for a ≤ r ≤ R.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::WeightedLaplacianForm;
use sonium_fem::mesh::generators::spherical_shell_mesh_tetrahedra;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn test_electrostatics_sphere_uniform_field() {
    let a = 0.01; // Sphere radius (m)
    let r_outer = 0.05; // Outer domain radius (m)
    let e0 = 1e3; // External field (V/m)

    let mesh = spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, a, r_outer, 2, 4);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = WeightedLaplacianForm { coefficient: 1.0 };
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let n_dofs = space.num_global_dofs(&mesh);
    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);

    // Inner sphere: φ = 0
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);

    // Outer sphere: φ = -E0·z·(1 - a³/R³)
    let scale = 1.0 - a.powi(3) / r_outer.powi(3);
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        2,
        &|_x, _y, z| -e0 * z * scale,
        None,
    );

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 3000,
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
        "GMRES should converge for electrostatics Laplace problem"
    );

    let mut max_err: f64 = 0.0;
    let mut max_exact: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let r = (node.x * node.x + node.y * node.y + node.z * node.z).sqrt();
        let theta = if r > 1e-12 { (node.z / r).acos() } else { 0.0 };
        let exact = if r <= a {
            0.0
        } else {
            -e0 * r * theta.cos() * (1.0 - a.powi(3) / r.powi(3))
        };
        let err = (result.x[i] - exact).abs();
        max_err = max_err.max(err);
        max_exact = max_exact.max(exact.abs());
    }

    let relative_error = max_err / max_exact.max(f64::MIN_POSITIVE);

    println!(
        "Electrostatics sphere uniform field: max error = {:.4e} V ({:.3}%)",
        max_err,
        100.0 * relative_error
    );
    assert!(
        relative_error < 0.02,
        "Relative potential error too large: {:.4e}",
        relative_error
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_electrostatics_sphere_uniform_field();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
