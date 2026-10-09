//! QA: FEM electrostatics — parallel plate capacitor (1D approximation).
//!
//! Analytical: E = V/d, φ(x) = V*x/d

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::WeightedLaplacianForm;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn test_electrostatics_parallel_plate() {
    let d = 1e-3; // plate separation
    let w = 0.05; // width (1D approximation)
    let v_left = 0.0;
    let v_right = 1.0;

    let nx = 40;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, d, 0.0, w, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let form = WeightedLaplacianForm { coefficient: 1.0 };
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let n_dofs = space.num_global_dofs(&mesh);
    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);

    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| v_left, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| v_right, None);

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
    assert!(result.converged, "GMRES should converge for electrostatics");

    // Analytical: φ(x) = x/d
    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let x = node.x;
        let phi_exact = v_left + (v_right - v_left) * x / d;
        max_err = max_err.max((result.x[i] - phi_exact).abs());
    }

    println!("Electrostatics parallel plate: max_err = {:.4e}", max_err);
    assert!(
        max_err < 1e-2,
        "Max potential error too large: {:.4e}",
        max_err
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_electrostatics_parallel_plate();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
