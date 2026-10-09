//! Basic acoustics example using PrebakedAssembler
//!
//! This example demonstrates how to set up a simple Helmholtz problem
//! with a velocity source and an impedance boundary condition.

use num_complex::Complex64;
use sonium_fem::assembly::assemble_boundary_mass;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::physics::acoustics::pressure::impedance::impedance_to_robin_coefficient;
use sonium_fem::physics::acoustics::pressure::sources::{NeumannFlux, NormalVelocitySource};
use sonium_fem::physics::acoustics::pressure::{
    AcousticsAssemblerExt, build_acoustics_assembler_with_boundaries,
};
use sonium_solvers::{GmresConfig, gmres};
use std::collections::HashMap;

fn main() {
    println!("--- Sonium FEM: Basic Acoustics Example ---");

    // 1. Create mesh (1.0m x 1.0m unit square)
    let mesh = unit_square_triangles(20);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // 2. Build assembler
    // We'll use Tag 2 for an absorbing boundary
    let boundary_tags = vec![2];
    let assembler =
        build_acoustics_assembler_with_boundaries(&mesh, PolynomialDegree::P1, &boundary_tags);

    // 3. Define physics properties
    let freq = 200.0;
    let rho = 1.204;
    let c = 343.0;
    let k_val = 2.0 * std::f64::consts::PI * freq / c;
    let k = Complex64::new(k_val, 0.0);
    println!("Frequency: {} Hz (k = {:.3} rad/m)", freq, k_val);

    // 4. Boundary conditions
    // Tag 2: Absorption (Impedance) - 80% absorption coefficient
    let alpha = impedance_to_robin_coefficient(Complex64::new(rho * c * 2.0, 0.0), k_val, rho, c);
    let mut boundaries = HashMap::new();
    boundaries.insert(2, alpha);

    // 5. Assemble system matrix
    let system = assembler.assemble_helmholtz(k, &boundaries);

    // 6. Source: Normal velocity on Tag 1 (Left wall)
    let source = NormalVelocitySource {
        amplitude: Complex64::new(1.0, 0.0),
        omega: 2.0 * std::f64::consts::PI * freq,
        density: rho,
    };
    let flux = source.neumann_flux();
    let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 1);
    let mut rhs = vec![Complex64::new(0.0, 0.0); mesh.num_nodes()];
    for i in 0..b_mass.nnz() {
        rhs[b_mass.rows[i]] += flux * b_mass.values[i];
    }
    // 7. Solve
    let config = GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 100,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    println!("Solving system...");
    let result = gmres(&system, &rhs, &config);

    if result.converged {
        println!(
            "Converged in {} iterations with residual {:.2e}",
            result.iterations, result.residual
        );

        // Calculate max pressure magnitude
        let max_p = result.x.iter().map(|c| c.norm()).fold(0.0, f64::max);
        println!("Max pressure magnitude: {:.2} Pa", max_p);
    } else {
        println!("Failed to converge!");
    }
}
