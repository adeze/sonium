//! Hypersingular-operator static-limit validation.
//!
//! For the Laplace hypersingular boundary operator, a constant density is in
//! the nullspace: `W₀[1] = 0`. At finite Helmholtz wavenumber `Wₖ[1]` is not
//! zero, so this test uses a very small `ka` and never infers `W` from the
//! combined Burton–Miller matrix.

use num_complex::Complex64;
use sonium_bem::core::assembly::operators::assemble_hypersingular;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
use std::f64::consts::PI;

fn static_constant_error(subdivisions: usize) -> f64 {
    let radius = 0.1;
    let speed_of_sound = 343.0;
    let ka = 1e-4;
    let frequency = ka * speed_of_sound / (2.0 * PI * radius);
    let physics = PhysicsParams::new(frequency, speed_of_sound, 1.21, false);
    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        element.dof_addresses = vec![index];
    }

    let operator = assemble_hypersingular(&elements, &mesh.nodes, &physics);
    let mean_sqr = (0..operator.nrows())
        .map(|row| {
            (0..operator.ncols())
                .map(|column| operator[[row, column]])
                .sum::<Complex64>()
                .norm_sqr()
        })
        .sum::<f64>()
        / operator.nrows() as f64;
    // W has dimensions 1/length. Scale by the sphere radius to obtain a
    // dimensionless static-consistency residual.
    let error = radius * mean_sqr.sqrt();
    println!(
        "subdiv={subdivisions}, elements={}, dimensionless RMS |a W[1]|={error:.3e}",
        elements.len()
    );
    error
}

fn run() {
    println!("=== Hypersingular static-limit validation ===");
    let coarse = static_constant_error(1);
    let fine = static_constant_error(2);

    assert!(
        coarse.is_finite() && fine.is_finite(),
        "non-finite hypersingular static-limit metric"
    );
    assert!(
        coarse < 1e-3 && fine < 1e-3,
        "dimensionless static W[1] residual exceeds 1e-3: coarse={coarse:.3e}, fine={fine:.3e}"
    );
}

fn main() {
    match std::panic::catch_unwind(run) {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
