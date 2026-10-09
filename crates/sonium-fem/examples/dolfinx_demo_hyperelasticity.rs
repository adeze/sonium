//! Hyperelasticity demo - translated from DOLFINx
//!
//! This example demonstrates:
//! - Solving a nonlinear PDE (hyperelasticity)
//! - Neo-Hookean material model (compressible)
//! - Using Newton-Raphson iteration
//! - 3D mesh and vector function spaces
//!
//! Equation: Solve for displacement u in a Neo-Hookean body
//! Domain: Unit cube [0,1]³
//!
//! This matches the DOLFINx demo_hyperelasticity example.
#![allow(deprecated)]

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::box_mesh_tetrahedra;
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_with_bcs};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};

fn main() {
    println!("=== DOLFINx Demo: Hyperelasticity (Neo-Hookean) ===\n");

    // Parameters matching DOLFINx demo
    let nx = 10;
    let ny = 10;
    let nz = 10;

    // Material parameters (matching DOLFINx)
    let young = 10.0;
    let poisson = 0.3;

    // 1. Create 3D mesh - unit cube [0,1]³
    let mesh = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz);
    println!(
        "Mesh: {} nodes, {} tetrahedra",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // 2. Create vector-valued function space (3D displacement)
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let dofs = space.num_global_dofs(&mesh);
    println!("Function space: P1 Vector (3D), {} DOFs", dofs);

    // 3. Create Neo-Hookean material
    let material = NeoHookean::from_young(young, poisson, 3);
    println!("Material: Neo-Hookean, E={}, ν={}", young, poisson);

    // 4. Define hyperelastic form
    let form = HyperelasticForm::new(material);

    // 5. Set up boundary conditions
    // Left face (x=0): fixed (u=0)
    // Right face (x=1): displaced (rotated)
    let mut dirichlet_dofs: Vec<(usize, f64)> = Vec::new();

    let scale = 0.005; // Small deformation as in DOLFINx
    let mut n_fixed = 0usize;
    let mut n_displaced = 0usize;

    for (node_idx, node) in mesh.nodes.iter().enumerate() {
        let tol = 1e-8;

        if node.x < tol {
            // Left face: fixed
            for d in 0..3 {
                dirichlet_dofs.push((node_idx * 3 + d, 0.0));
            }
            n_fixed += 1;
        } else if (1.0 - node.x) < tol {
            // Right face: apply small rotation
            // DOLFINx rotates by 60 degrees around z-axis at center (0.5, 0.5)
            let x1_c = 0.5;
            let x2_c = 0.5;
            let theta = std::f64::consts::PI / 3.0; // 60 degrees

            let dx = node.x - x1_c;
            let dy = node.y - x2_c;

            let new_x = dx * theta.cos() - dy * theta.sin() + x1_c - dx;
            let new_y = dx * theta.sin() + dy * theta.cos() + x2_c - dy;

            dirichlet_dofs.push((node_idx * 3, scale * new_x));
            dirichlet_dofs.push((node_idx * 3 + 1, scale * new_y));
            dirichlet_dofs.push((node_idx * 3 + 2, 0.0));
            n_displaced += 1;
        }
    }

    println!("Dirichlet BC: {} fixed, {} displaced", n_fixed, n_displaced);

    // 6. Initial guess (zero displacement)
    let initial_guess = vec![0.0; dofs];

    // 7. Configure Newton solver
    let config = FemNewtonConfig::default();
    println!("\nSolving with Newton-Raphson...");

    // 8. Solve
    match fem_newton_solve_with_bcs(
        &form,
        &space,
        &mesh,
        &dirichlet_dofs,
        &initial_guess,
        &config,
    ) {
        Ok(result) => {
            println!("Newton converged in {} iterations", result.iterations);

            // Compute max displacement
            let max_disp = result.x.iter().map(|&v| v.abs()).fold(0.0, f64::max);
            println!("Max displacement: {:.6}", max_disp);

            // Compute some statistics
            let mean_disp: f64 = result.x.iter().sum::<f64>() / dofs as f64;
            println!("Mean displacement: {:.6}", mean_disp);

            // VTK output
            let n_nodes = mesh.num_nodes();
            let u = &result.x;
            let ux: Vec<f64> = (0..n_nodes).map(|i| u[i * 3]).collect();
            let uy: Vec<f64> = (0..n_nodes).map(|i| u[i * 3 + 1]).collect();
            let uz: Vec<f64> = (0..n_nodes).map(|i| u[i * 3 + 2]).collect();
            let mag: Vec<f64> = (0..n_nodes)
                .map(|i| (ux[i] * ux[i] + uy[i] * uy[i] + uz[i] * uz[i]).sqrt())
                .collect();

            let mut writer = VtuWriter::new(&mesh);
            writer.add_point_vector("displacement", &ux, &uy, &uz);
            writer.add_point_scalar("displacement_magnitude", &mag);

            match writer.write("hyperelasticity.vtu") {
                Ok(_) => println!("\nWrote hyperelasticity.vtu"),
                Err(e) => println!("\nFailed to write VTK: {e}"),
            }
        }
        Err(e) => {
            println!("Newton failed: {:?}", e);
        }
    }

    println!("\n=== Demo Complete ===");
}
