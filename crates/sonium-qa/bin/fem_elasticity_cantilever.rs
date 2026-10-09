//! 2D Linear Elasticity Cantilever Beam Example
//!
//! This example demonstrates solving a structural mechanics problem:
//! A 2D beam is fixed at one end and loaded at the other.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};
use sonium_xem::io::binary_output_dir;

fn test_elasticity_cantilever() {
    println!("--- Sonium FEM: Elasticity Cantilever Example ---");

    // 1. Define geometry and material
    let l = 1.0;
    let h = 0.2;
    let e_mod = 2.1e11; // Steel (Pa)
    let nu = 0.3; // Poisson's ratio
    let p_load = 1e6; // Tip load (N/m)

    println!("Beam: {}m x {}m", l, h);
    println!("Material: E = {:.1e} Pa, nu = {:.2}", e_mod, nu);

    // 2. Create mesh
    let nx = 40;
    let ny = 8;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // 3. Setup function space (Vector H1 for displacement [u, v])
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let n_dofs = space.num_global_dofs(&mesh);
    println!("Degrees of Freedom: {}", n_dofs);

    // 4. Define bilinear form
    let material = IsotropicMaterial::new(e_mod, nu);
    // Euler-Bernoulli theory with unit out-of-plane thickness corresponds to
    // a plane-stress slice.  Plane strain has a different flexural rigidity
    // and must not be compared to the same analytical denominator.
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);

    // 5. Assemble stiffness matrix
    println!("Assembling stiffness matrix...");
    let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);

    // 6. Define RHS (point load at the tip)
    let mut rhs = vec![0.0; n_dofs];
    let tol = 1e-10;
    let tip_nodes: Vec<usize> = (0..mesh.num_nodes())
        .filter(|&i| (mesh.nodes[i].x - l).abs() < tol)
        .collect();

    let load_per_node = -p_load / tip_nodes.len() as f64;
    for &node in &tip_nodes {
        let v_dof = node * 2 + 1; // y-displacement
        rhs[v_dof] = load_per_node;
    }

    // 7. Apply Boundary Conditions
    let mut system = FemSystem::new(stiffness, rhs);
    println!("Applying fixed BC at x=0...");
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    // 8. Solve linear system
    println!("Solving system with GMRES...");
    let csr = system.matrix.to_csr();

    let config = GmresConfig {
        max_iterations: 5000,
        restart: 100,
        tolerance: 1e-10,
        print_interval: 500,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let preconditioner = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(&csr, &preconditioner, &system.rhs, &config);

    assert!(result.converged, "Solver failed to converge!");
    println!("Converged in {} iterations", result.iterations);

    // 9. Post-process results
    let mut max_v = 0.0;
    for &node in &tip_nodes {
        let v = result.x[node * 2 + 1].abs();
        if v > max_v {
            max_v = v;
        }
    }

    // Analytical Euler-Bernoulli solution
    let i_inertia = 1.0 * h.powi(3) / 12.0;
    let theory_delta = p_load * l.powi(3) / (3.0 * e_mod * i_inertia);

    let rel_error = (max_v - theory_delta).abs() / theory_delta;
    println!("Max tip deflection: {:.6e} m", max_v);
    println!("Analytical (1D theory): {:.6e} m", theory_delta);
    println!("Relative error: {:.1}%", rel_error * 100.0);

    let n = mesh.num_nodes();
    let ux: Vec<f64> = (0..n).map(|i| result.x[i * 2]).collect();
    let uy: Vec<f64> = (0..n).map(|i| result.x[i * 2 + 1]).collect();
    let uz: Vec<f64> = vec![0.0; n];
    let mag: Vec<f64> = (0..n)
        .map(|i| (ux[i] * ux[i] + uy[i] * uy[i]).sqrt())
        .collect();
    let mut w = VtuWriter::new(&mesh);
    w.add_point_vector("displacement", &ux, &uy, &uz);
    w.add_point_scalar("displacement_magnitude", &mag);
    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let _ = w.write(dir.join("cantilever.vtu"));

    assert!(
        rel_error < 0.03,
        "Tip deflection relative error {:.1}% exceeds 3% threshold",
        rel_error * 100.0
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elasticity_cantilever();
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
